"""Spec-derived tests of Block 7 (inversion, flags, fill) -- integration
tester, run D1, 2026-11-13. Tables are built by the producer
(`table.fit_table`, Block 14's local linear estimator) from noise-free rows
whose ratio is a known function b(ln d, phi, ln(mu_hat d_hat)) that is linear
in the coordinates, which the local linear fit and its cubic read-out
reproduce exactly; the expected roots come from scipy.optimize.brentq on the
true b, and the fill from scipy.ndimage.median_filter. The code under test
only produces the outputs that are checked.

Run (from the workstream directory):
    python -m pytest tests/spec/test_spec_block7_invert.py -q
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import ndimage, optimize

import spec_helpers_D1 as H

from allen_diameter import config as C
from allen_diameter.analysis import fill as FL
from allen_diameter.analysis import invert as INV
from allen_diameter.analysis import node_pipeline as NP
from allen_diameter.analysis import table as T

NAN = float("nan")


def small_cfg(**corr):
    cfg = C.default_config()
    ph = dataclasses.replace(cfg.phantom, d_range_um=(0.5, 2.0), phi_range_deg=(0.0, 20.0),
                             mu_range_per_um=(0.5, 2.0))
    co = dataclasses.replace(cfg.correction, bandwidth_cv_grid=(1.0,), **corr)
    return dataclasses.replace(cfg, phantom=ph, correction=co)


def make_rows(cfg, bfun, n_draws=1000, n_real=2, seed=3, q_max=None, fail=None):
    """Noise-free rows: draw g's (d, phi, mu) by the design's laws; each
    realization a third coordinate q = ln(mu_hat d_hat) uniform on the design
    band (capped at q_max), ratio y = b(ln d, phi, q), d_hat = d y,
    mu_hat = e^q / d_hat; fail(d, phi) gives the probability of leaving S."""
    p = cfg.phantom
    rng = np.random.default_rng(seed)
    qlo = math.log(p.mu_range_per_um[0] * p.d_range_um[0])
    qhi = math.log(p.mu_range_per_um[1] * p.d_range_um[1])
    rows = []
    for g in range(1, n_draws + 1):
        d = math.exp(rng.uniform(*np.log(p.d_range_um)))
        phi = math.radians(rng.uniform(*p.phi_range_deg))
        mu = math.exp(rng.uniform(*np.log(p.mu_range_per_um)))
        for r in range(1, n_real + 1):
            q = rng.uniform(qlo, qhi if q_max is None else q_max)
            y = bfun(math.log(d), phi, q)
            dh = d * y
            ins = True if fail is None else bool(rng.uniform() >= fail(d, phi))
            rows.append(dict(draw_index=g, realization_index=r, d_um=d, phi_rad=phi, theta_rad=0.0,
                             mu_per_um=mu, d_hat_um=dh, mu_hat_per_um=math.exp(q) / dh, ratio=y,
                             in_S=ins, reject="" if ins else "crossing", sim_hash="sim-x"))
    return rows


def b_lin(ld, phi, q):
    return 1.05 + 0.03 * ld + 0.05 * phi - 0.02 * q


@pytest.fixture(scope="module")
def lin():
    cfg = small_cfg()
    return cfg, T.fit_table(make_rows(cfg, b_lin), cfg)


def ref_root(bfun, d_hat, phi, q, lo, hi):
    return optimize.brentq(lambda d: d * bfun(math.log(d), phi, q) - d_hat, lo, hi, xtol=1e-14, rtol=1e-15)


# ----------------------------------------------------------- O7.1, O7.2 ---

def test_table_reproduces_linear_b(lin):
    cfg, tab = lin
    pts = [(0.8, 0.1, -0.2), (1.0, 0.2, 0.0), (1.5, 0.05, 0.5), (0.7, 0.3, -0.5)]
    for d, phi, q in pts:
        assert abs(tab.b_hat(d, phi, math.exp(q)) - b_lin(math.log(d), phi, q)) < 1e-8


@pytest.mark.parametrize("d,phi,q", [(1.0, 0.2, 0.0), (0.8, 0.1, -0.3), (1.6, 0.3, 0.6), (0.6, 0.05, -0.5)])
def test_root_recovers_d(lin, d, phi, q):
    cfg, tab = lin
    y = b_lin(math.log(d), phi, q)
    d_hat = d * y
    mu_hat = math.exp(q) / d_hat
    inv = INV.invert_node(d_hat, phi, mu_hat, tab, cfg)
    assert inv.flags == () or list(inv.flags) == [], inv.flags
    assert abs(inv.d_tilde_um - d) < 1e-9 * d
    assert abs(inv.d_root_um - inv.d_tilde_um) == 0
    assert abs(inv.b_hat - y) < 1e-8
    assert abs(tab.m_hat(inv.d_tilde_um, phi, mu_hat * d_hat) - d_hat) < 1e-9     # m_hat(d~) = d^
    assert abs(inv.d_tilde_um - ref_root(b_lin, d_hat, phi, q, 0.5, 2.0)) < 1e-9


def test_shortcut_is_d_hat_over_b_hat_at_d_hat(lin):
    cfg, tab = lin
    d_hat, phi, mu_hat = 1.2, 0.15, 0.9
    s = INV.shortcut(d_hat, phi, mu_hat, tab)
    exp = d_hat / b_lin(math.log(d_hat), phi, math.log(mu_hat * d_hat))
    assert abs(float(s) - exp) < 1e-8


@pytest.mark.parametrize("args,flag", [
    ((NAN, 0.1, 1.0), "no_estimate"),
    ((-1.0, 0.1, 1.0), "no_estimate"),
    ((1.0, 0.1, 0.0), "no_estimate"),
    ((1.0, NAN, 1.0), "no_estimate"),
    ((50.0, 0.1, 0.02), "out_of_domain"),
    ((0.05, 0.1, 20.0), "out_of_domain"),
    ((1.0, math.radians(40.0), 1.0), "phi_out_of_domain"),
])
def test_flags_leave_d_tilde_nan(lin, args, flag):
    cfg, tab = lin
    inv = INV.invert_node(*args, tab, cfg)
    assert flag in inv.flags, inv.flags
    assert math.isnan(inv.d_tilde_um)


def test_mu_d_out_of_domain_is_a_label(lin):
    cfg, tab = lin
    rng_ = tab.mu_d_range() if callable(tab.mu_d_range) else tab.mu_d_range
    lo, hi = rng_
    q_out = math.log(hi) + 0.4                      # beyond the third axis: clamped to its end
    d_hat = 1.0
    inv = INV.invert_node(d_hat, 0.17, math.exp(q_out) / d_hat, tab, cfg)
    assert "mu_d_out_of_domain" in inv.flags
    exp = ref_root(b_lin, d_hat, 0.17, math.log(hi), 0.5, 2.0)
    assert abs(inv.d_root_um - exp) < 1e-8              # inverted at the clamped coordinate
    others = set(inv.flags) - set(INV.LABEL_FLAGS)
    # the label alone never sends the node to the fill; M is read at the rows' TRUE mu d
    # (D-062.5, D-063.1), so off_support may also fire at the end of the axis
    assert math.isfinite(inv.d_tilde_um) == (not others), inv.flags


def test_large_correction_threshold(lin):
    cfg, tab = lin
    d, phi, q = 1.0, 0.2, 0.0
    y = b_lin(0.0, phi, q)                             # 1.06: |b - 1| = 0.06
    strict = dataclasses.replace(cfg, correction=dataclasses.replace(cfg.correction, bias_flag_threshold=0.05))
    inv = INV.invert_node(d * y, phi, math.exp(q) / (d * y), tab, strict)
    assert "large_correction" in inv.flags and math.isnan(inv.d_tilde_um)
    assert abs(inv.d_root_um - d) < 1e-9                # the root kept for diagnostics
    inv2 = INV.invert_node(d * y, phi, math.exp(q) / (d * y), tab, cfg)
    assert "large_correction" not in inv2.flags


def test_non_monotone():
    cfg = small_cfg(bias_flag_threshold=10.0)

    def b_nm(ld, phi, q):                               # m(d) = d (1 - ln d): maximum at d = 1
        return 1.0 - ld

    tab = T.fit_table(make_rows(cfg, b_nm, n_draws=800), cfg)
    inv = INV.invert_node(0.9, 0.1, 1.0, tab, cfg)      # two roots on [0.5, 2]
    assert "non_monotone" in inv.flags and math.isnan(inv.d_tilde_um)


def test_high_failure_and_off_support():
    cfg = small_cfg()
    fail = lambda d, phi: 0.6 if phi > math.radians(10.0) else 0.0
    tab = T.fit_table(make_rows(cfg, b_lin, n_draws=1200, fail=fail, q_max=0.3), cfg)
    d, q = 1.0, 0.0
    y = b_lin(0.0, math.radians(15), q)
    inv = INV.invert_node(d * y, math.radians(15), math.exp(q) / (d * y), tab, cfg)
    assert "high_failure" in inv.flags and math.isnan(inv.d_tilde_um)
    y = b_lin(0.0, math.radians(5), q)
    inv = INV.invert_node(d * y, math.radians(5), math.exp(q) / (d * y), tab, cfg)
    assert "high_failure" not in inv.flags and math.isfinite(inv.d_tilde_um)
    # rows only up to q = 0.3; the axis spans the design band up to ln 4 = 1.39: no support at q = 1.2
    q = 1.2
    y = b_lin(0.0, math.radians(5), q)
    inv = INV.invert_node(d * y, math.radians(5), math.exp(q) / (d * y), tab, cfg)
    assert "off_support" in inv.flags and math.isnan(inv.d_tilde_um)


# ------------------------------------------------------------- O7.3 -------

def node(**kw):
    import test_spec_block6_phantoms as B6
    return B6.make_result(**kw)


def test_correct_nodes(lin):
    cfg, tab = lin
    y = b_lin(0.0, 0.2, 0.0)
    ok = dict(d_hat_um=y, mu_hat_per_um=1.0 / y, phi_rad=0.2)
    res = [node(**ok), node(flags=("faint", "dark"), **ok), node(flags=("crossing",), **ok),
           node(fit_status="at_bound", **ok)]
    inv = INV.correct_nodes(res, tab, cfg)
    assert len(inv) == 4
    assert abs(inv[0].d_tilde_um - 1.0) < 1e-9
    assert abs(inv[1].d_tilde_um - 1.0) < 1e-9                 # faint and dark are labels (D-030, D-096)
    assert math.isnan(inv[2].d_tilde_um) and "crossing" in inv[2].flags
    assert math.isnan(inv[3].d_tilde_um) and any("at_bound" in f for f in inv[3].flags)
    other = dataclasses.replace(cfg, measure=dataclasses.replace(cfg.measure, sigma_fit_um=0.12))
    with pytest.raises(ValueError):
        INV.correct_nodes(res, tab, other)


# ------------------------------------------------------------- O7.4 -------

def ref_fill(d_tilde, allen, policy, window):
    d = np.asarray(d_tilde, float)
    n = d.size
    out = d.copy()
    src = np.where(np.isfinite(d), "self", "none").astype(object)
    good = np.flatnonzero(np.isfinite(d))
    for i in np.flatnonzero(~np.isfinite(d)):
        if policy == "none":
            continue
        if policy == "same_branch_then_allen" and good.size:
            left = good[good < i]
            right = good[good > i]
            vals = ([d[left[-1]]] if left.size else []) + ([d[right[0]]] if right.size else [])
            out[i] = float(np.median(vals))
            src[i] = "neighbours"
        else:
            out[i] = allen[i]
            src[i] = "allen"
    return ndimage.median_filter(out, size=window, mode="nearest"), list(src)


@pytest.mark.parametrize("d_tilde", [
    [1.0, NAN, 1.2, NAN, NAN, 0.9],          # interior gaps
    [NAN, 1.0, 1.2, 1.1],                    # end gap
    [NAN, NAN, NAN],                         # the whole stretch flagged
    [1.0, 3.0, 1.1, 1.05, 1.2],              # a spike for the running median
])
@pytest.mark.parametrize("policy", ["same_branch_then_allen", "allen_only"])
def test_fill_policies(d_tilde, policy):
    cfg = C.default_config()
    cfg = dataclasses.replace(cfg, correction=dataclasses.replace(cfg.correction, fill_policy=policy))
    allen = np.full(len(d_tilde), 0.6)
    got, src = FL.fill_stretch(np.array(d_tilde), allen, cfg)
    exp, esrc = ref_fill(d_tilde, allen, policy, cfg.correction.median_window_nodes)
    assert_allclose(got, exp, rtol=0, atol=1e-15)
    assert list(src) == esrc


def test_fill_none_keeps_nan_and_does_not_spread_it():
    """`none`: NaN stays; the running median over the filled values must not
    turn a measured neighbour into NaN or into a value it never had."""
    cfg = C.default_config()
    cfg = dataclasses.replace(cfg, correction=dataclasses.replace(cfg.correction, fill_policy="none"))
    d = np.array([1.0, NAN, 1.2, 1.3, 1.4])
    got, src = FL.fill_stretch(d, np.full(5, 0.6), cfg)
    assert math.isnan(got[1]) and list(src) == ["self", "none", "self", "self", "self"]
    assert np.all(np.isfinite(got[[0, 2, 3, 4]]))
