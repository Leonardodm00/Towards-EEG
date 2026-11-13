"""Spec-derived tests, Block 2: noise_law (SPEC 2.1 (L1)-(L3), the closed
forms (7a), (7b), (9), (10), (8), the window-mean covariance and the
semivariogram). Expected values: brute-force covariance matrices, my own
AR(1) recursion on the same unit draws (scipy.signal.lfilter), Monte Carlo
with fixed seeds, and the topic doc's table 3.5.3 (validation target M8).
"""
import dataclasses
import math
import os
import re

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import noise_law as nl
from noise_law import NoiseLaw
import _fixtures as fx

ROOT = fx.os.path.dirname(fx.os.path.dirname(fx.os.path.dirname(fx.os.path.abspath(fx.__file__))))
DT = 2e-5
LAWS = [NoiseLaw(0.038, 0.175, 0.0508, 0.0366), NoiseLaw(0.1, 0.9, 0.02, 0.005), NoiseLaw(0.05, 0.5, 0.2, 0.3)]


def _gamma_matrix(law, dt, n):
    k = np.arange(n)
    g = fx.gamma((law.sigma_f, law.rho_f, law.sigma_s, law.tau_s), dt, k)
    return g[np.abs(k[:, None] - k[None, :])]


# --- O2.1 / O2.2 / O2.3 the law object ---------------------------------------
@pytest.mark.parametrize("bad", [(-1e-9, 0.5, 0.05, 0.04), (0.04, 0.5, -1e-9, 0.04), (0.04, 0.0, 0.05, 0.04),
                                 (0.04, 1.0, 0.05, 0.04), (0.04, 0.5, 0.05, 0.0), (0.04, 0.5, 0.05, -1.0),
                                 (math.nan, 0.5, 0.05, 0.04), (0.04, math.nan, 0.05, 0.04),
                                 (0.04, 0.5, math.inf, 0.04), (0.04, 0.5, 0.05, math.inf)])
def test_O2_1_invalid_laws_refused(bad):
    with pytest.raises(ValueError):
        NoiseLaw(*bad)


def test_O2_1_zero_sd_accepted_spec_conflict():
    """SPEC 2.1 defines Theta with sigma_f, sigma_s > 0, but Block 4's oracles
    F2/F4/F5 use sigma_s = 0 and sigma_f = sigma_s = 0. The code accepts 0
    (documented here; the conflict is listed under Problems in the spec)."""
    NoiseLaw(0.0, 0.5, 0.05, 0.04)
    NoiseLaw(0.04, 0.5, 0.0, 0.04)


def test_O2_1_frozen_and_round_trip():
    law = LAWS[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        law.sigma_f = 1.0
    a = law.as_array()
    assert a.dtype == np.float64 and a.shape == (4,)
    assert_array_equal(a, [0.038, 0.175, 0.0508, 0.0366])
    assert NoiseLaw.from_array(a) == law
    with pytest.raises(ValueError):
        NoiseLaw.from_array(np.ones(3))


def test_O2_3_rho_s_tau_f():
    law = LAWS[0]
    assert law.rho_s(DT) == pytest.approx(math.exp(-DT / 0.0366), rel=1e-15)
    assert law.tau_f(DT) == pytest.approx(-DT / math.log(0.175), rel=1e-15)


def test_O2_4_autocovariance_closed_form():
    for law in LAWS:
        k = np.array([0, 1, 2, 50, 500, -3])
        ref = law.sigma_f ** 2 * law.rho_f ** np.abs(k) + law.sigma_s ** 2 * np.exp(-np.abs(k) * DT / law.tau_s)
        assert_allclose(nl.autocovariance(law, DT, k), ref, rtol=1e-14, atol=0)


# --- O2.5 / O2.6 / O2.8 / O2.9 the simulator ----------------------------------
def test_O2_5_draw_order():
    u = nl.draw_units(np.random.default_rng(11), 3, 7)
    r = np.random.default_rng(11)
    assert_array_equal(u.fast, r.standard_normal((3, 7)))
    assert_array_equal(u.slow, r.standard_normal((3, 7)))
    assert_array_equal(u.tilt, r.standard_normal(3))
    assert_array_equal(u.offset, r.standard_normal(3))
    with pytest.raises(ValueError):
        nl.draw_units(np.random.default_rng(0), 1, 1)


def _reference_traces(law, dt, u, tilt=None, sigma_b=0.0, sigma_a=0.0):
    rho_s = math.exp(-dt / law.tau_s)
    x = fx.ar1_from_units(u.fast, law.sigma_f, law.rho_f)
    # slow innovations use 1 - rho_s^2 = -expm1(-2 dt / tau_s) (SPEC Block 2, "numpy.expm1 for 1 - rho_s^2")
    xi = np.asarray(u.slow, float)
    eta = law.sigma_s * math.sqrt(-math.expm1(-2.0 * dt / law.tau_s)) * xi
    eta[:, 0] = law.sigma_s * xi[:, 0]
    from scipy.signal import lfilter
    x = x + lfilter([1.0], [1.0, -rho_s], eta, axis=1)
    n = xi.shape[1]
    tc = (np.arange(n) - (n - 1) / 2.0) * dt
    if tilt is not None:
        x = x + np.asarray(tilt)[:, None] * tc[None, :]
    elif sigma_b > 0:
        x = x + (sigma_b * u.tilt)[:, None] * tc[None, :]
    if sigma_a > 0:
        x = x + (sigma_a * u.offset)[:, None]
    return x


def test_O2_6_simulate_traces_equals_independent_recursion():
    u = nl.draw_units(np.random.default_rng(12), 4, 300)
    for law in LAWS:
        got = nl.simulate_traces(law, DT, u.fast, u.slow)
        assert got.dtype == np.float64 and got.shape == (4, 300)
        # 1e-12 mV: roundoff of two lfilter recursions of 300 steps on mV-sized values
        assert_allclose(got, _reference_traces(law, DT, u), rtol=0, atol=1e-12)
        got2 = nl.simulate_traces(law, DT, u.fast, u.slow, xi_tilt=u.tilt, sigma_b=0.3,
                                  xi_offset=u.offset, sigma_a=2.0)
        assert_allclose(got2, _reference_traces(law, DT, u, sigma_b=0.3, sigma_a=2.0), rtol=0, atol=1e-12)
        assert_array_equal(nl.simulate_traces(law, DT, u.fast, u.slow), got)       # bit-identical on the same draws


def test_O2_8_fixed_tilt_exact_centred_and_refusals():
    u = nl.draw_units(np.random.default_rng(13), 3, 400)
    law = LAWS[0]
    b = np.array([0.25, -0.1, 0.0])
    with_t = nl.simulate_traces(law, DT, u.fast, u.slow, tilt_mV_per_s=b)
    without = nl.simulate_traces(law, DT, u.fast, u.slow)
    tc = (np.arange(400) - 199.5) * DT
    assert_allclose(with_t - without, b[:, None] * tc[None, :], rtol=0, atol=1e-12)
    assert np.all(np.abs((with_t - without).sum(axis=1)) < 1e-12)    # the tilt sums to 0 over the trace
    for kw in (dict(tilt_mV_per_s=b, xi_tilt=u.tilt, sigma_b=0.1), dict(tilt_mV_per_s=b[:2]),
               dict(tilt_mV_per_s=np.array([0.1, np.nan, 0.0])), dict(sigma_b=0.1),
               dict(sigma_a=0.1)):
        with pytest.raises(ValueError):
            nl.simulate_traces(law, DT, u.fast, u.slow, **kw)


def test_O2_8_nan_sigma_b_is_not_silently_ignored():
    """Silent-failure hunt: sigma_b is validated by comparison (< 0 refused).
    A NaN sigma_b passes that test and the tilt is then left out without an
    error (NaN > 0 is False). Spec: sigma_b in mV/s (a number); a NaN should
    not be accepted silently."""
    u = nl.draw_units(np.random.default_rng(14), 2, 50)
    with pytest.raises(ValueError):
        nl.simulate_traces(LAWS[0], DT, u.fast, u.slow, xi_tilt=u.tilt, sigma_b=float("nan"))


def test_O2_9_simulate_sweep_is_a_row():
    u = nl.draw_units(np.random.default_rng(15), 3, 120)
    law = LAWS[1]
    full = nl.simulate_traces(law, DT, u.fast, u.slow, tilt_mV_per_s=np.array([0.0, 0.7, 0.0]))
    assert_array_equal(nl.simulate_sweep(law, DT, 120, u, 1, tilt_mV_per_s=0.7), full[1])
    with pytest.raises(ValueError):
        nl.simulate_sweep(law, DT, 119, u, 0)


def test_O2_7_monte_carlo_autocovariance_and_stationarity():
    """2000 traces x 600 samples at a law with both components visible: the
    across-trace covariance at lags 0, 1, 50 (anchored at sample 300) and the
    first/last sample variances equal (L3) within 3.5 SE (fixed seed)."""
    law = NoiseLaw(0.05, 0.6, 0.08, 0.004)
    u = nl.draw_units(np.random.default_rng(16), 2000, 600)
    x = nl.simulate_traces(law, DT, u.fast, u.slow)
    for (i, j) in ((300, 300), (300, 301), (300, 350), (0, 0), (599, 599)):
        prod = x[:, i] * x[:, j]
        se = prod.std(ddof=1) / math.sqrt(prod.size)
        assert abs(prod.mean() - nl.autocovariance(law, DT, [j - i])[0]) < 3.5 * se, (i, j)


# --- O2.10 / O2.11 moments ---------------------------------------------------
@pytest.mark.parametrize("law", LAWS)
def test_O2_10_moments_against_brute_force(law):
    n = 240
    G = _gamma_matrix(law, DT, n)
    Pm = np.eye(n) - np.ones((n, n)) / n
    PGP = Pm @ G @ Pm
    var_mean = G.sum() / n ** 2
    es0 = np.trace(PGP)
    lags = np.array([1, 2, 17, 120, 239])
    esl = np.array([np.trace(PGP, offset=L) for L in lags])
    # 1e-10 relative: brute force sums 240^2 terms
    assert nl.var_window_mean(law, DT, n) == pytest.approx(var_mean, rel=1e-10)
    assert nl.expected_s0(law, DT, n) == pytest.approx(es0, rel=1e-10)
    assert_allclose(nl.expected_sl(law, DT, n, lags), esl, rtol=1e-9, atol=1e-16)
    assert nl.closed_form_sd(law, DT, n) == pytest.approx(math.sqrt(es0 / (n - 1)), rel=1e-10)
    assert_allclose(nl.closed_form_acf(law, DT, n, lags), esl / es0, rtol=1e-9, atol=1e-14)
    assert np.all(np.isnan(nl.expected_sl(law, DT, n, [0, -1, 240, 300])))
    assert np.all(np.isnan(nl.closed_form_acf(law, DT, n, [0, 240])))
    # my own O(n) derivation agrees too (it is the oracle used for long windows below)
    vm, s0, sl = fx.demeaned_moments((law.sigma_f, law.rho_f, law.sigma_s, law.tau_s), DT, n, lags)
    assert vm == pytest.approx(var_mean, rel=1e-10) and s0 == pytest.approx(es0, rel=1e-10)
    assert_allclose(sl, esl, rtol=1e-9, atol=1e-16)


def test_O2_11_visible_share():
    z = np.array([0.25, 1.26, 6.67, 30.0])
    assert_allclose(nl.visible_share(z), 1 - 2 * (z - 1 + np.exp(-z)) / z ** 2, rtol=1e-12)
    for zz in (0.25, 1.26, 6.67):
        n = 4000
        tau = n * DT / zz
        law = NoiseLaw(0.0, 0.5, 0.05, tau)                  # slow-only (zero sigma_f, as M7)
        ratio = nl.expected_s0(law, DT, n) / (n * 0.05 ** 2)
        assert abs(ratio - nl.visible_share(zz)) < 2 * zz / n      # SPEC M7's tolerance


# --- O2.12 / O2.13 window means ----------------------------------------------
def test_O2_12_window_mean_cov_brute_force():
    law = LAWS[1]
    k = np.arange(400)
    for n1, n2, off in ((50, 50, 0), (50, 50, 30), (50, 70, 120), (40, 25, -60), (30, 30, 200)):
        i = np.arange(n1)[:, None]
        j = np.arange(n2)[None, :]
        ref = fx.gamma((law.sigma_f, law.rho_f, law.sigma_s, law.tau_s), DT, off + j - i).mean()
        assert nl.window_mean_cov(law, DT, n1, n2, off) == pytest.approx(ref, rel=1e-10)


def test_O2_13_semivariogram_equal_windows_and_tilt():
    law = LAWS[0]
    n, off = 9500, 4 * 10025
    v = nl.var_window_mean(law, DT, n)
    c = nl.window_mean_cov(law, DT, n, n, off)
    h = off * DT
    assert nl.semivariogram(law, DT, n, n, off, sigma_b=0.1) == pytest.approx(v - c + 0.5 * 0.01 * h * h, rel=1e-12)


def test_O2_13_continuous_form_close_to_discrete():
    """M4: for a slow-only law the exact-discrete V and eq. (25)'s continuous
    form agree within 3 dt/tau_s (relative)."""
    law = NoiseLaw(0.0, 0.5, 0.05, 0.0366)
    n = 9500
    for off in (10025, 4 * 10025):
        Vd = nl.semivariogram(law, DT, n, n, off)
        Vc = nl.semivariogram_continuous(law, DT, n * DT, n * DT, off * DT)
        assert abs(Vd - Vc) / Vc < 3 * DT / law.tau_s


# --- O2.15 validation target: topic doc 3.5.3 (q_cf at theta_cf) -------------
QCF_DOC = {  # topic doc TEEG_Stage7_noise_model_2026-09-28.md, table of 3.5.3, column q_cf(theta_cf)
    ("ss", 500): (0.0408, [0.280, 0.123, 0.070]),
    ("ls", 13250): (0.0584, [0.650, 0.574, 0.553]),
    ("first", 2500): (0.0480, []),
    ("last", 6625): (0.0546, [0.599, 0.512, 0.487, 0.272]),
}
LAGS = {"ss": [1, 5, 50], "ls": [1, 5, 50], "first": [], "last": [1, 5, 50, 500]}


def test_O2_15_topic_doc_q_cf_reproduced():
    """theta_cf is printed to 3 significant digits, hence the spec's
    tolerances 3e-4 mV (SD) and 3e-3 (autocorrelation)."""
    law = NoiseLaw(0.0380, 0.175, 0.0508, 0.0366)
    for (name, n), (sd, rhos) in QCF_DOC.items():
        assert abs(nl.closed_form_sd(law, DT, n) - sd) < 3e-4, name
        if rhos:
            got = nl.closed_form_acf(law, DT, n, LAGS[name])
            assert np.max(np.abs(got - np.array(rhos))) < 3e-3, (name, got)
        # the same from my own derivation (checks the doc's numbers, not the code)
        _, s0, sl = fx.demeaned_moments((0.0380, 0.175, 0.0508, 0.0366), DT, n, LAGS[name] or [1])
        assert abs(math.sqrt(s0 / (n - 1)) - sd) < 3e-4
        if rhos:
            assert np.max(np.abs(sl / s0 - np.array(rhos))) < 3e-3


# --- O2.14 layer rules --------------------------------------------------------
def test_O2_14_no_module_rng_no_io_ascii():
    src = open(os.path.join(ROOT, "noise_law.py")).read()
    code = re.sub(r'""".*?"""', "", src, flags=re.S)
    assert not re.search(r"np\.random\.(seed|normal|rand|randn|standard_normal|default_rng)\(", code)
    assert not re.search(r"\bopen\(|np\.save|np\.load|savefig|plt\.", code)
    b = src.encode()
    assert all(c < 128 for c in b) and b"\r" not in b
