"""Block 7 (inversion, flags, fill): oracles from SPEC.md section 2.4 and Block 7."""
import dataclasses
import math

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import ndimage

from allen_diameter.config import default_config
from allen_diameter.analysis import fill, invert, table as TB

BASE = default_config()
CFG = dataclasses.replace(BASE, correction=dataclasses.replace(BASE.correction, spline_smoothing=0.0))
C0 = 0.1


def _exact_table(b_fun, n=400, seed=0):
    rng = np.random.default_rng(seed)
    d = np.exp(rng.uniform(math.log(0.2), math.log(4.0), n))
    phi = np.radians(rng.uniform(0, 90, n))
    rows = [dict(d_um=d[i], phi_rad=phi[i], ratio=b_fun(d[i], phi[i]), in_S=True, reject="") for i in range(n)]
    return TB.fit_table(rows, CFG)


T = _exact_table(lambda d, p: 1 + C0 / d)


@pytest.mark.parametrize("d", [0.5, 1.0, 2.0, 3.0])
def test_inversion_recovers_d(d):
    phi = math.radians(30)
    d_hat = d * (1 + C0 / d)
    inv = invert.invert_node(d_hat, phi, T, CFG)
    assert not inv.flags
    assert abs(inv.d_tilde_um - d) <= 1e-3 * d       # spline interpolation error of exact data
    assert float(T.m_hat(inv.d_tilde_um, phi)) == pytest.approx(d_hat, rel=1e-9)
    # shortcut closed form c0^2 / (d (d + 2 c0)) (relative error of d_hat / b(d_hat))
    sc = invert.shortcut(d_hat, phi, T)
    assert abs((sc - d) / d - C0 ** 2 / (d * (d + 2 * C0))) < 1e-3


def test_out_of_domain_and_large_correction():
    phi = 0.3
    lo = float(T.m_hat(0.2, phi))
    inv = invert.invert_node(0.9 * lo, phi, T, CFG)
    assert "out_of_domain" in inv.flags and math.isnan(inv.d_tilde_um)
    hi = float(T.m_hat(4.0, phi))
    assert "out_of_domain" in invert.invert_node(1.1 * hi, phi, T, CFG).flags
    # b = 1 + 0.1/d > 1.2 for d < 0.5: large_correction, d_tilde NaN, root kept
    inv = invert.invert_node(0.3 * (1 + C0 / 0.3), phi, T, CFG)
    assert "large_correction" in inv.flags and math.isnan(inv.d_tilde_um)
    assert inv.d_root_um == pytest.approx(0.3, rel=1e-3)
    assert "no_estimate" in invert.invert_node(float("nan"), phi, T, CFG).flags


def test_non_monotone_flag():
    # m(d) = d b(d) with b = 1 + 0.6 sin(3 ln d) / d ... any m with two crossings of one level
    Tn = _exact_table(lambda d, p: 1.0 + 0.8 * math.exp(-((math.log(d) - math.log(1.0)) / 0.25) ** 2) * 1.0 / d)
    grid = np.exp(np.linspace(math.log(0.2), math.log(4.0), 400))
    m = Tn.m_hat(grid, np.full(grid.shape, 0.2))
    assert np.any(np.diff(m) < 0)                    # the fixture is non-monotone
    dm = np.diff(m)
    i = int(np.flatnonzero((dm[:-1] > 0) & (dm[1:] <= 0))[0]) + 1      # local maximum
    j = i + int(np.argmin(m[i:]))                                        # following local minimum
    level = 0.5 * (m[i] + m[j])                                          # crossed three times
    inv = invert.invert_node(level, 0.2, Tn, CFG)
    assert "non_monotone" in inv.flags


def test_mismatched_estimator_refused():
    other = dataclasses.replace(CFG, measure=dataclasses.replace(CFG.measure, sigma_fit_um=0.125))
    with pytest.raises(ValueError):
        invert.correct_nodes([], T, other)


def _fill_cfg(policy="same_branch_then_allen", w=1):
    return dataclasses.replace(CFG, correction=dataclasses.replace(CFG.correction, fill_policy=policy,
                                                                   median_window_nodes=w))


def test_fill_rules():
    nan = float("nan")
    allen = np.array([9.0, 9.0, 9.0, 9.0, 9.0])
    d, src = fill.fill_stretch([1.0, nan, 3.0, nan, nan], allen, _fill_cfg())
    assert_allclose(d, [1.0, 2.0, 3.0, 3.0, 3.0])            # interior: median(1, 3); end: nearest only
    assert list(src) == ["self", "neighbours", "self", "neighbours", "neighbours"]
    d, src = fill.fill_stretch([nan] * 3, allen[:3], _fill_cfg())
    assert_allclose(d, 9.0) and list(src) == ["allen"] * 3
    d, src = fill.fill_stretch([1.0, nan], allen[:2], _fill_cfg("allen_only"))
    assert_allclose(d, [1.0, 9.0])
    d, src = fill.fill_stretch([1.0, nan], allen[:2], _fill_cfg("none"))
    assert math.isnan(d[1]) and src[1] == "none"
    spike = np.array([1.0, 1.0, 5.0, 1.0, 1.0])
    d, _ = fill.fill_stretch(spike, allen, _fill_cfg(w=3))
    assert_allclose(d, ndimage.median_filter(spike, size=3, mode="nearest"))
    assert_allclose(d, 1.0)
