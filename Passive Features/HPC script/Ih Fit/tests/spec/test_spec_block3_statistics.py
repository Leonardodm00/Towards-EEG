"""Block 3 (noise_cell_stats.py; noise_windows.py as neighbour) against SPEC 2.6 (c), 2.2 (S1)-(S2).

Oracles derived from the spec:
  * d = 21 statistics in the order SS (sd + 6 lags), LS pre-window (sd + 5 lags),
    LS first 50 ms (sd), LS last 132.5 ms (sd + 6 lags); labels as printed with %g.
  * lags in samples round(lag_ms 1e-3 f_s) = 1, 2, 5, 15, 50, 500 at 50 kHz.
  * sigma_hat = SD ddof 1; rho_hat = statsmodels acf(adjusted=False) (D-028) -- computed here
    with statsmodels directly, not through noise_stats.
  * N_SS = 7, N_LS = 14, from noise_cell_stats.
"""
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from statsmodels.tsa.stattools import acf

import noise_cell_stats as ncs
import noise_windows as nw

FS = 50000.0
SPEC_LABELS = (["ss_sd"] + ["ss_rho_%sms" % s for s in ("0.02", "0.04", "0.1", "0.3", "1", "10")]
               + ["ls_sd"] + ["ls_rho_%sms" % s for s in ("0.02", "0.04", "0.1", "0.3", "1")]
               + ["ls_first_sd"]
               + ["ls_last_sd"] + ["ls_last_rho_%sms" % s for s in ("0.02", "0.04", "0.1", "0.3", "1", "10")])


def test_labels_counts_and_blocks():
    assert list(ncs.STAT_LABELS) == SPEC_LABELS
    assert len(ncs.STAT_LABELS) == 21 and ncs.N_SS_STATS == 7 and ncs.N_LS_STATS == 14
    assert list(ncs.STAT_BLOCK) == ["ss"] * 7 + ["ls"] * 14


def test_lags_in_samples():
    assert_array_equal(ncs.lags_in_samples(ncs.SS_LAGS_MS, FS), [1, 2, 5, 15, 50, 500])
    assert_array_equal(ncs.lags_in_samples(ncs.LS_FULL_LAGS_MS, FS), [1, 2, 5, 15, 50])
    assert_array_equal(ncs.lags_in_samples(ncs.LS_LAST_LAGS_MS, FS), [1, 2, 5, 15, 50, 500])


def _ref_stats(x, lags):
    a = acf(x, adjusted=False, nlags=max(lags), fft=False)
    return [np.std(x, ddof=1)] + [a[L] for L in lags]


def test_ss_window_stats_against_statsmodels():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((3, 6000)).cumsum(axis=1) * 0.01 + rng.standard_normal((3, 6000))
    got = ncs.ss_window_stats(X, FS)
    assert got.shape == (3, 7)
    exp = np.array([_ref_stats(x, [1, 2, 5, 15, 50, 500]) for x in X])
    # fft vs direct sums of 6000 terms: ~1e-15 relative roundoff; 1e-12 ample
    assert_allclose(got, exp, rtol=1e-12, atol=1e-12)


def test_ls_window_stats_layout():
    rng = np.random.default_rng(2)
    pre = [rng.standard_normal(13250) + 0.3 * np.sin(np.arange(13250) / 300.0) for _ in range(2)]
    got = ncs.ls_window_stats(pre, FS, nw.WindowConfig())
    assert got.shape == (2, 14)
    for k, x in enumerate(pre):
        first, last = x[:2500], x[-6625:]                # 50 ms and 132.5 ms at 50 kHz
        exp = (_ref_stats(x, [1, 2, 5, 15, 50]) + [np.std(first, ddof=1)]
               + _ref_stats(last, [1, 2, 5, 15, 50, 500]))
        assert_allclose(got[k], exp, rtol=1e-12, atol=1e-12)


def test_statistic_vector_shapes_and_lengths():
    rng = np.random.default_rng(3)
    ss = rng.standard_normal((9, 7))
    ls = rng.standard_normal((4, 14))
    sv = ncs.statistic_vector(ss, 6000, ls, 13250)
    assert sv.q.shape == (21,) and sv.d == 21
    assert_array_equal(sv.n, [6000] * 7 + [13250] * 6 + [2500] + [6625] * 7)
    assert_array_equal(sv.K, [9] * 7 + [4] * 14)
    assert_allclose(sv.q, np.concatenate([np.median(ss, axis=0), np.median(ls, axis=0)]), rtol=0, atol=0)


def test_pipeline_vector_matches_cell(cell):
    # boundary Block 3 -> Block 4: CellTarget's q_hat equals the medians of the statistics recomputed here
    nwin, ls, tg = cell["nwin"], cell["ls"], cell["target"]
    X = nwin.X_tilde
    ss_exp = np.array([_ref_stats(x, [1, 2, 5, 15, 50, 500]) for x in X])
    assert tg.per_window_ss.shape == (X.shape[0], 7)
    assert_allclose(tg.per_window_ss, ss_exp, rtol=1e-12, atol=1e-12)
    assert tg.per_window_ls.shape == (len(ls.pre), 14)
    assert_allclose(tg.q_hat[:7], np.median(ss_exp, axis=0), rtol=1e-12)
    assert tg.q_hat.shape == (21,) and np.all(np.isfinite(tg.q_hat))
