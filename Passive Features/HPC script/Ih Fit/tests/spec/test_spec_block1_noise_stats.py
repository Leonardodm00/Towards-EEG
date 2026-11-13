"""Spec-derived tests, Block 1: noise_stats.sample_acf (SPEC 2.2 (S1)-(S2),
D-028, D-029). Written by the integration tester from SPEC.md; expected
values come from statsmodels, numpy and hand sums, never from the code under
test.
"""
import os
import subprocess

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from statsmodels.tsa.stattools import acf

import noise_stats
from noise_stats import sample_acf

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _hand(x, L):
    """(S2) written out: S_L / S_0 on the demeaned window."""
    u = np.asarray(x, float) - np.mean(x)
    return float(np.dot(u[:-L], u[L:]) / np.dot(u, u))


def test_O1_1_equals_statsmodels_and_hand_sum():
    rng = np.random.default_rng(1)
    for n in (12, 37, 500, 4750):
        x = np.cumsum(rng.standard_normal(n)) * 0.01 + rng.standard_normal(n)
        lags = np.arange(1, n - 9)
        got = sample_acf(x, lags)
        ref = acf(x, adjusted=False, nlags=int(lags.max()), fft=False)[lags]
        # 1e-12: the reference uses the direct (non-FFT) sum; FFT roundoff on sums of <= 4750 terms of O(1)
        assert_allclose(got, ref, rtol=0, atol=1e-12)
        for L in sorted({1, 2, max(1, n // 3), n - 10}):
            if 1 <= L <= n - 10:
                assert abs(got[L - 1] - _hand(x, L)) < 1e-12


def test_O1_2_textbook_equals_scaled_adjusted_form():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(300)
    n = x.size
    u = x - x.mean()
    for L in (1, 7, 150, n - 10):
        rho_adj = (np.dot(u[:-L], u[L:]) / (n - L)) / (np.dot(u, u) / n)
        # D-028.2 identity; 1e-12 for roundoff of two sums of 300 terms
        assert abs(sample_acf(x, L)[0] - (n - L) / n * rho_adj) < 1e-12


def test_O1_3_nan_rules_boundaries():
    rng = np.random.default_rng(3)
    x = rng.standard_normal(50)
    out = sample_acf(x, [0, -1, 1, 40, 41])
    assert np.isnan(out[0]) and np.isnan(out[1])          # L < 1
    assert np.isfinite(out[2]) and np.isfinite(out[3])    # n - L = 10 is allowed
    assert np.isnan(out[4])                               # n - L = 9 is not
    for bad in (np.nan, np.inf, -np.inf):
        y = x.copy()
        y[17] = bad
        assert np.all(np.isnan(sample_acf(y, [1, 2, 3])))


def test_O1_3_flat_window_exactly_representable_mean():
    for c in (0.0, 1.0, -70.0, 2.5):
        assert np.all(np.isnan(sample_acf(np.full(500, c), [1, 2, 5])))


@pytest.mark.parametrize("c,n", [(0.1, 12), (0.1, 13250), (0.3, 500), (-70.123, 13250), (65.4321, 4750)])
def test_O1_3_flat_window_any_constant_is_nan(c, n):
    """D-029 (iii): NaN at every lag when S_0 = 0. A constant window has S_0 = 0
    exactly whatever the constant; the float mean must not turn it into a
    finite autocorrelation. (Finding: fails for constants whose float mean
    carries roundoff -- the result is (n - L)/n, e.g. 0.9999 at n = 13250.)"""
    out = sample_acf(np.full(n, c), [1, 2])
    assert np.all(np.isnan(out)), "flat window of %r (n=%d) gave %r" % (c, n, out)


def test_O1_4_contract_order_scalar_dtype_list_input():
    rng = np.random.default_rng(4)
    x = rng.standard_normal(100)
    lags = [5, 1, 5, 3]
    out = sample_acf(list(x), lags)
    assert out.shape == (4,) and out.dtype == np.float64
    ref = acf(x, adjusted=False, nlags=5, fft=False)
    assert_allclose(out, ref[lags], rtol=0, atol=1e-13)   # roundoff only
    one = sample_acf(x, 3)
    assert one.shape == (1,) and abs(one[0] - ref[3]) < 1e-13


def test_O1_5_bound_and_affine_invariance():
    rng = np.random.default_rng(5)
    worst = 0.0
    for _ in range(200):
        n = int(rng.integers(12, 400))
        r = rng.uniform(-0.99, 0.99)
        x = np.zeros(n)
        e = rng.standard_normal(n)
        for j in range(1, n):
            x[j] = r * x[j - 1] + e[j]
        v = sample_acf(x, np.arange(1, n - 9))
        worst = max(worst, np.nanmax(np.abs(v)))
        y = -3.7 * x + 12.0
        # rho is invariant under x -> a x + c; 1e-11 allows roundoff after scaling by 3.7 and shifting by 12
        assert_allclose(sample_acf(y, [1, 2]), sample_acf(x, [1, 2]), rtol=0, atol=1e-11)
    assert worst <= 1.0


def test_O1_6_no_mutation_constant_and_no_adjusted_opt_in():
    rng = np.random.default_rng(6)
    x = rng.standard_normal(64)
    x0 = x.copy()
    sample_acf(x, [1, 2])
    assert_array_equal(x, x0)
    assert noise_stats.MIN_PRODUCTS == 10
    # D-029 (ii): no adjusted-form opt-in anywhere in Ih Fit/ (git grep over tracked .py files)
    out = subprocess.run(["git", "grep", "-n", "adjusted=True", "--", "*.py"], cwd=ROOT,
                         capture_output=True, text=True)
    assert out.stdout.strip() == "", out.stdout


def test_O1_7_call_sites_give_the_textbook_lag1():
    """noise_calibration.lag_autocorr and the fitter's two estimators go
    through sample_acf: their lag-1 value equals (S2) by hand."""
    import noise_calibration
    rng = np.random.default_rng(7)
    x = rng.standard_normal(1000).cumsum() * 0.001 + rng.standard_normal(1000)
    assert abs(noise_calibration.lag_autocorr(x, 1) - _hand(x, 1)) < 1e-12
    assert abs(noise_calibration.lag_autocorr(x, 37) - _hand(x, 37)) < 1e-12
    assert np.isnan(noise_calibration.lag_autocorr(x[:10], 1))     # n - L = 9


def test_O1_7_fitter_call_sites_source():
    """The two fitter lines named by the spec compute rho through sample_acf
    (static check of the source lines; the monolith imports NEURON)."""
    src = open(os.path.join(ROOT, "passive_fitting_hpc_fixed.py")).read()
    for fn in ("def _estimate_noise(", "def _estimate_residual_noise_at_mle("):
        i = src.index(fn)
        body = src[i:src.index("\ndef ", i + 10)]
        assert "sample_acf(" in body, fn
        assert "np.corrcoef" not in body and "/ (n - 1)" not in body


def test_O1_8_ascii_lf():
    for f in ("noise_stats.py", "smoke_noise_stats.py"):
        b = open(os.path.join(ROOT, f), "rb").read()
        assert all(c < 128 for c in b) and b"\r" not in b, f


def test_note_float_lag_and_2d_input_are_not_refused():
    """Spec: lags are ints in samples, x is 1-D. Documents the behaviour on
    out-of-contract input (NOTE in the report): a float lag is truncated
    toward zero and a 2-D array is flattened, both silently."""
    rng = np.random.default_rng(8)
    x = rng.standard_normal(200)
    assert sample_acf(x, 2.9)[0] == sample_acf(x, 2)[0]
    X = rng.standard_normal((3, 100))
    assert sample_acf(X, 1)[0] == sample_acf(X.ravel(), 1)[0]


def _ls_bundle(v):
    import passive_fitting_hpc_fixed as mono
    n = v.size
    return mono.SweepBundle(polarity="hyp", amplitude_pA=-50.0, t=np.arange(n) / 50000.0, v_mV=v,
                            i_pA=np.zeros(n), stim_onset_s=0.270, stim_duration_s=1.0, n_repeats_averaged=1,
                            sweep_numbers=[60], sampling_rate_Hz=50000.0)


def test_O1_7_fitter_estimate_noise_lag1_is_textbook():
    """passive_fitting_hpc_fixed._estimate_noise (an LS bundle): rho_lag1 of
    the pre-window [0, 265) ms equals (S2) by hand."""
    import passive_fitting_hpc_fixed as mono
    rng = np.random.default_rng(9)
    v = -70.0 + np.cumsum(rng.standard_normal(20000)) * 0.002 + 0.05 * rng.standard_normal(20000)
    sigma, rho, corr = mono._estimate_noise(_ls_bundle(v))
    pre = v[:13250]
    assert sigma == pytest.approx(pre.std(ddof=1), rel=1e-12)
    assert abs(rho - _hand(pre, 1)) < 1e-12 and corr == (rho > 0.5)


def test_O1_7_fitter_flat_prewindow_gives_nan_not_a_correlation_flag():
    """D-029 (iii) carried to the fitter: a flat pre-window has no lag-1
    autocorrelation (NaN) and must not be flagged as correlated. (Finding:
    with a non-representable constant the fitter reports rho_lag1 = 0.9999
    and is_correlated = True.)"""
    import passive_fitting_hpc_fixed as mono
    sigma, rho, corr = mono._estimate_noise(_ls_bundle(np.full(20000, -70.3)))
    assert np.isnan(rho) and corr is False, (sigma, rho, corr)
