"""Block 2 (noise_law.py) against SPEC 2.6 (a), (b), (h): the three-component law.

Oracles derived from the spec, not from the code:
  * theta = (sigma_f, sigma_m, tau_m, sigma_s, tau_s), this order everywhere;
    NoiseLaw accepts sigma >= 0, tau > 0, finite; it does not enforce the fit bounds.
  * gamma(k) = sigma_f^2 [k=0] + sigma_m^2 exp(-|k| dt/tau_m) + sigma_s^2 exp(-|k| dt/tau_s)   (L3')
  * draws: fast, mid, slow (n_traces, n), tilt, offset (n_traces,), in THIS order (2.6 (b)).
  * e_fast = sigma_f xi_fast; e_mid, e_slow stationary AR(1) started in the stationary law.
  * closed-form moments keep their formula in gamma: checked against the full covariance matrix.
"""
import math

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.linalg import toeplitz

import noise_law as nl

DT = 1.0 / 50000.0
LAW = nl.NoiseLaw(0.035, 0.017, 3e-4, 0.055, 0.040)


def gamma_spec(law, dt, k):
    k = np.abs(np.asarray(k, dtype=float))
    return (law.sigma_f ** 2 * (k == 0) + law.sigma_m ** 2 * np.exp(-k * dt / law.tau_m)
            + law.sigma_s ** 2 * np.exp(-k * dt / law.tau_s))


# ---------------------------------------------------------------- the law object
def test_field_order_and_round_trip():
    th = np.array([0.1, 0.2, 3e-4, 0.4, 0.05])
    law = nl.NoiseLaw.from_array(th)
    assert (law.sigma_f, law.sigma_m, law.tau_m, law.sigma_s, law.tau_s) == tuple(th)
    assert_array_equal(law.as_array(), th)
    assert law.as_array().shape == (5,) and law.as_array().dtype == np.float64
    with pytest.raises(ValueError):
        nl.NoiseLaw.from_array(th[:4])


def test_rho_f_and_tau_f_removed():
    # 2.6 (a): "rho_f no longer exists"; 2.6 (h): tau_f removed
    assert not hasattr(LAW, "rho_f") and not hasattr(LAW, "tau_f")


def test_rho_m_rho_s():
    # exp(-dt / tau), exact up to one rounding (1e-15 relative)
    assert_allclose(LAW.rho_m(DT), math.exp(-DT / 3e-4), rtol=1e-15)
    assert_allclose(LAW.rho_s(DT), math.exp(-DT / 0.040), rtol=1e-15)


@pytest.mark.parametrize("args", [
    (-1e-3, 0.1, 1e-4, 0.1, 0.1), (0.1, -1e-3, 1e-4, 0.1, 0.1), (0.1, 0.1, 1e-4, -1e-3, 0.1),
    (0.1, 0.1, 0.0, 0.1, 0.1), (0.1, 0.1, 1e-4, 0.1, 0.0), (0.1, 0.1, -1e-4, 0.1, 0.1),
    (float("nan"), 0.1, 1e-4, 0.1, 0.1), (0.1, 0.1, float("inf"), 0.1, 0.1), (0.1, 0.1, 1e-4, 0.1, float("nan"))])
def test_invalid_laws_refused(args):
    with pytest.raises(ValueError):
        nl.NoiseLaw(*args)


def test_zero_sd_and_out_of_fit_bounds_accepted():
    # 2.6 (a): any sigma >= 0, any tau > 0; the fit's bounds are not enforced
    nl.NoiseLaw(0.0, 0.0, 1e-4, 0.0, 1.0)
    nl.NoiseLaw(100.0, 50.0, 0.5, 20.0, 100.0)      # tau_m far above 1 ms, tau_s above 10 s
    nl.NoiseLaw(1e-9, 1e-9, 1e-7, 1e-9, 1e-7)


# ---------------------------------------------------------------- (L3')
def test_autocovariance_L3prime_any_sign():
    lags = np.array([-500, -50, -1, 0, 1, 2, 5, 15, 50, 500])
    # same formula evaluated twice: only rounding differs (1e-14 relative)
    assert_allclose(nl.autocovariance(LAW, DT, lags), gamma_spec(LAW, DT, lags), rtol=1e-14)
    g0 = nl.autocovariance(LAW, DT, [0])[0]
    assert_allclose(g0, 0.035 ** 2 + 0.017 ** 2 + 0.055 ** 2, rtol=1e-14)


# ---------------------------------------------------------------- draws and simulator
def test_draw_order():
    rng = np.random.default_rng(5)
    u = nl.draw_units(rng, 3, 40)
    ref = np.random.default_rng(5)
    exp = [ref.standard_normal((3, 40)) for _ in range(3)] + [ref.standard_normal(3) for _ in range(2)]
    for got, e in zip((u.fast, u.mid, u.slow, u.tilt, u.offset), exp):
        assert_array_equal(got, e)


def _loop_ar1(sigma, rho, xi):
    out = np.empty_like(xi)
    out[:, 0] = sigma * xi[:, 0]
    s = sigma * math.sqrt(1.0 - rho * rho)
    for j in range(1, xi.shape[1]):
        out[:, j] = rho * out[:, j - 1] + s * xi[:, j]
    return out


def test_simulator_equals_written_out_law():
    rng = np.random.default_rng(11)
    u = nl.draw_units(rng, 4, 3000)
    copies = [a.copy() for a in (u.fast, u.mid, u.slow)]
    x = nl.simulate_traces(LAW, DT, u.fast, u.mid, u.slow)
    for a, c in zip((u.fast, u.mid, u.slow), copies):
        assert_array_equal(a, c)                                   # inputs not mutated
    exp = (LAW.sigma_f * u.fast + _loop_ar1(LAW.sigma_m, math.exp(-DT / LAW.tau_m), u.mid)
           + _loop_ar1(LAW.sigma_s, math.exp(-DT / LAW.tau_s), u.slow))
    assert x.dtype == np.float64 and x.shape == (4, 3000)
    # recursion of 3000 mV-sized steps in float64: roundoff ~1e-16 per step, 1e-12 mV is ample
    assert_allclose(x, exp, rtol=0, atol=1e-12)


def test_components_switch_off_and_use_their_own_draws():
    rng = np.random.default_rng(12)
    u = nl.draw_units(rng, 2, 500)
    only_mid = nl.NoiseLaw(0.0, 0.02, 2e-4, 0.0, 0.1)
    x = nl.simulate_traces(only_mid, DT, u.fast, u.mid, u.slow)
    assert_allclose(x, _loop_ar1(0.02, math.exp(-DT / 2e-4), u.mid), atol=1e-13)
    only_slow = nl.NoiseLaw(0.0, 0.0, 2e-4, 0.03, 0.05)
    x = nl.simulate_traces(only_slow, DT, u.fast, u.mid, u.slow)
    assert_allclose(x, _loop_ar1(0.03, math.exp(-DT / 0.05), u.slow), atol=1e-13)
    only_fast = nl.NoiseLaw(0.04, 0.0, 2e-4, 0.0, 0.05)
    assert_array_equal(nl.simulate_traces(only_fast, DT, u.fast, u.mid, u.slow), 0.04 * u.fast)


def test_monte_carlo_autocovariance_matches_L3prime():
    # across-trace products at a fixed origin j0 = 0 (stationary start) and j0 = 400
    # tolerance 3.5 SE per lag, 14 checks: family-wise false-alarm ~ 0.7 %; fixed seed
    rng = np.random.default_rng(2026)
    N, n = 6000, 1001
    u = nl.draw_units(rng, N, n)
    x = nl.simulate_traces(LAW, DT, u.fast, u.mid, u.slow)
    lags = [0, 1, 2, 5, 15, 50, 500]
    bad = []
    for j0 in (0, 400):
        for L in lags:
            if j0 + L >= n:
                continue
            prod = x[:, j0] * x[:, j0 + L]
            m, se = prod.mean(), prod.std(ddof=1) / math.sqrt(N)
            g = gamma_spec(LAW, DT, L)
            if abs(m - g) > 3.5 * se:
                bad.append((j0, L, m, g, se))
    assert not bad, bad


def test_fixed_tilt_and_sweep_row():
    rng = np.random.default_rng(3)
    u = nl.draw_units(rng, 3, 200)
    b = np.array([0.1, -0.25, 0.0])
    x0 = nl.simulate_traces(LAW, DT, u.fast, u.mid, u.slow)
    x1 = nl.simulate_traces(LAW, DT, u.fast, u.mid, u.slow, tilt_mV_per_s=b)
    t = np.arange(200) * DT
    # mV-sized additions in float64: 1e-15 mV roundoff, 1e-12 ample
    assert_allclose(x1 - x0, b[:, None] * (t - t.mean())[None, :], atol=1e-12)
    row = nl.simulate_sweep(LAW, DT, 200, u, 1, tilt_mV_per_s=-0.25)
    assert_array_equal(row, x1[1])
    with pytest.raises(ValueError):
        nl.simulate_traces(LAW, DT, u.fast, u.mid, u.slow, tilt_mV_per_s=b, sigma_b=0.1, xi_tilt=u.tilt)


# ---------------------------------------------------------------- closed-form moments vs brute force
@pytest.mark.parametrize("law", [LAW, nl.NoiseLaw(0.05, 0.0, 1e-4, 0.0, 1.0), nl.NoiseLaw(0.0, 0.03, 1e-4, 0.0, 1.0),
                                 nl.NoiseLaw(0.01, 0.02, 4e-5, 0.03, 2e-3)])
def test_closed_form_moments_brute_force(law):
    n = 240
    G = toeplitz(gamma_spec(law, DT, np.arange(n)))
    P = np.eye(n) - np.ones((n, n)) / n
    PGP = P @ G @ P
    lags = np.array([1, 2, 5, 15, 50, 239])
    var_mean = G.sum() / n ** 2
    es0 = np.trace(PGP)
    esl = np.array([np.trace(PGP, offset=L) for L in lags])
    # n x n matrix products in float64: relative roundoff ~ n eps ~ 1e-13; 1e-10 as in M1 of the spec
    assert_allclose(nl.var_window_mean(law, DT, n), var_mean, rtol=1e-10)
    assert_allclose(nl.expected_s0(law, DT, n), es0, rtol=1e-10)
    assert_allclose(nl.expected_sl(law, DT, n, lags), esl, rtol=1e-10, atol=1e-10 * es0)
    assert_allclose(nl.closed_form_sd(law, DT, n), math.sqrt(es0 / (n - 1)), rtol=1e-10)
    assert_allclose(nl.closed_form_acf(law, DT, n, lags), esl / es0, rtol=1e-10, atol=1e-10)
    out = nl.expected_sl(law, DT, n, [0, n])
    assert np.all(np.isnan(out))


def test_white_part_analytic_moments():
    # white noise: E[S_0] = (n-1) s^2, E[S_L] = -(n-L) s^2 / n (stated in Block 2's as-built note)
    law = nl.NoiseLaw(0.05, 0.0, 1e-4, 0.0, 1.0)
    n = 300
    assert_allclose(nl.expected_s0(law, DT, n), (n - 1) * 0.05 ** 2, rtol=1e-12)
    L = np.array([1, 7, 100])
    assert_allclose(nl.expected_sl(law, DT, n, L), -(n - L) * 0.05 ** 2 / n, rtol=1e-10)
