"""noise_cell_stats -- Block 3 (analysis layer) of specs/SPEC.md: the D-051
statistics per window, per sweep and per cell, the window-level bootstrap SE
and the data semivariogram of the SS window means (D-049, D-051, D-073,
D-076).

In plain words: every noise window (an enlarged SS window after the
mean-window subtraction, or an LS pre-window and its two segments) gives a
few numbers -- its SD and its textbook autocorrelation at a few lags. The
cell's statistic vector q_hat(c) is the median of each number over the
cell's windows; a sweep's vector q_hat(c, s) is the median over that sweep's
windows. The same functions run on simulated windows in Block 4, so data and
simulation go through one estimator (D-033). No I/O, no plotting here.

Units: SDs in mV, autocorrelations dimensionless, the semivariogram in
mV^2; lags are given in ms and converted to samples by round(lag_ms f_s / 1e3).
Pure ASCII, LF.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np

import noise_stats
from noise_windows import SSWindows, WindowConfig, ls_segments

SS_LAGS_MS: Tuple[float, ...] = (0.02, 0.1, 1.0, 10.0)
LS_FULL_LAGS_MS: Tuple[float, ...] = (0.02, 0.1, 1.0)
LS_LAST_LAGS_MS: Tuple[float, ...] = (0.02, 0.1, 1.0, 10.0)


def _lag_label(prefix: str, lag_ms: float) -> str:
    return "%s_rho_%gms" % (prefix, lag_ms)


STAT_LABELS: Tuple[str, ...] = (
    ("ss_sd",) + tuple(_lag_label("ss", l) for l in SS_LAGS_MS)
    + ("ls_sd",) + tuple(_lag_label("ls", l) for l in LS_FULL_LAGS_MS)
    + ("ls_first_sd",)
    + ("ls_last_sd",) + tuple(_lag_label("ls_last", l) for l in LS_LAST_LAGS_MS))
STAT_BLOCK: Tuple[str, ...] = tuple("ss" if lab.startswith("ss") else "ls" for lab in STAT_LABELS)
N_SS_STATS = 1 + len(SS_LAGS_MS)          # 5
N_LS_STATS = len(STAT_LABELS) - N_SS_STATS  # 10


def lags_in_samples(lags_ms: Sequence[float], fs: float) -> np.ndarray:
    return np.array([int(round(l * 1e-3 * fs)) for l in lags_ms], dtype=int)


# ===========================================================================
#  Per-window statistics
# ===========================================================================
def window_stats(X: np.ndarray, fs: float, lags_ms: Sequence[float]) -> np.ndarray:
    """(K, 1 + len(lags)): sigma_hat (ddof 1) then rho_hat at each lag, per
    window, through noise_stats.sample_acf (D-028, D-029)."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must be (K, n)")
    lags = lags_in_samples(lags_ms, fs)
    out = np.empty((X.shape[0], 1 + lags.size))
    for k in range(X.shape[0]):
        out[k, 0] = X[k].std(ddof=1)
        out[k, 1:] = noise_stats.sample_acf(X[k], lags)
    return out


def ss_window_stats(X_tilde: np.ndarray, fs: float) -> np.ndarray:
    """The 5 SS statistics per enlarged window (after the subtraction)."""
    return window_stats(X_tilde, fs, SS_LAGS_MS)


def ls_window_stats(pre_list: Sequence[np.ndarray], fs: float,
                    cfg: WindowConfig = WindowConfig()) -> np.ndarray:
    """The 10 LS statistics per pre-window: full window sigma_hat and
    rho_hat at LS_FULL_LAGS_MS; first-segment sigma_hat; last-segment
    sigma_hat and rho_hat at LS_LAST_LAGS_MS (D-051 (i))."""
    out = np.empty((len(pre_list), N_LS_STATS))
    lags_full = lags_in_samples(LS_FULL_LAGS_MS, fs)
    lags_last = lags_in_samples(LS_LAST_LAGS_MS, fs)
    for k, pre in enumerate(pre_list):
        pre = np.asarray(pre, dtype=np.float64)
        first, last = ls_segments(pre, fs, cfg)
        row = [pre.std(ddof=1)] + list(noise_stats.sample_acf(pre, lags_full))
        row += [first.std(ddof=1)]
        row += [last.std(ddof=1)] + list(noise_stats.sample_acf(last, lags_last))
        out[k] = row
    return out


# ===========================================================================
#  The statistic vector, per cell and per sweep
# ===========================================================================
@dataclass
class StatisticVector:
    q: np.ndarray          # (d,) medians
    labels: Tuple[str, ...]
    block: Tuple[str, ...]
    n: np.ndarray          # (d,) samples of the windows behind each statistic
    K: np.ndarray          # (d,) windows behind each statistic (finite values)

    @property
    def d(self) -> int:
        return int(self.q.size)


def _nanmedian_cols(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    A = np.asarray(A, dtype=np.float64)
    if A.size == 0:
        return np.full(A.shape[1] if A.ndim == 2 else 0, np.nan), np.zeros(A.shape[1] if A.ndim == 2 else 0, dtype=int)
    counts = np.isfinite(A).sum(axis=0)
    med = np.full(A.shape[1], np.nan)
    for j in range(A.shape[1]):
        if counts[j]:
            med[j] = np.nanmedian(A[:, j])
    return med, counts.astype(int)


def statistic_vector(ss_stats: np.ndarray, m: int, ls_stats: np.ndarray, n_ls: int,
                     cfg: WindowConfig = WindowConfig(), fs: float = 50000.0) -> StatisticVector:
    """q_hat(c): NaN-ignoring medians of the per-window statistics, with the
    window lengths n_m and counts K_m per statistic (D-051 (i), 2.2)."""
    ss_stats = np.asarray(ss_stats, dtype=np.float64).reshape(-1, N_SS_STATS)
    ls_stats = np.asarray(ls_stats, dtype=np.float64).reshape(-1, N_LS_STATS)
    q_ss, K_ss = _nanmedian_cols(ss_stats)
    q_ls, K_ls = _nanmedian_cols(ls_stats)
    n1 = int(round(cfg.ls_first_ms * 1e-3 * fs))
    n2 = int(round(cfg.ls_last_ms * 1e-3 * fs))
    n = np.array([m] * N_SS_STATS + [n_ls] * (1 + len(LS_FULL_LAGS_MS)) + [n1]
                 + [n2] * (1 + len(LS_LAST_LAGS_MS)), dtype=int)
    return StatisticVector(q=np.concatenate([q_ss, q_ls]), labels=STAT_LABELS, block=STAT_BLOCK,
                           n=n, K=np.concatenate([K_ss, K_ls]))


def per_sweep_vectors(stats: np.ndarray, sweep_number: np.ndarray) -> Dict[int, np.ndarray]:
    """{sweep: median over that sweep's windows of each statistic} (D-076 (i))."""
    stats = np.asarray(stats, dtype=np.float64)
    sweep_number = np.asarray(sweep_number)
    out: Dict[int, np.ndarray] = {}
    for s in np.unique(sweep_number):
        med, _ = _nanmedian_cols(stats[sweep_number == s])
        out[int(s)] = med
    return out


# ===========================================================================
#  Window-level bootstrap SE of the medians (D-049: W_0 = diag 1/SE^2)
# ===========================================================================
def bootstrap_median_se(per_window: np.ndarray, B: int, rng: np.random.Generator) -> np.ndarray:
    """SE of each column's median under resampling of whole windows with
    replacement (B draws); NaN entries are ignored in the medians. The
    windows are resampled as units so each keeps its own autocorrelation."""
    A = np.asarray(per_window, dtype=np.float64)
    K = A.shape[0]
    if K < 2:
        return np.full(A.shape[1], np.nan)
    idx = rng.integers(0, K, size=(int(B), K))
    meds = np.nanmedian(A[idx], axis=1)           # (B, d)
    return meds.std(axis=0, ddof=1)


# ===========================================================================
#  The data semivariogram of the window means (D-034.1, D-051 (iii))
# ===========================================================================
def window_mean_semivariogram(X_tilde: np.ndarray, ss: SSWindows, lag_periods: int = 4
                              ) -> Tuple[float, int]:
    """V_hat_c(h), h = lag_periods onset periods: half the mean squared
    difference of the window means of two windows of the same sweep and the
    same polarity whose positions differ by lag_periods. (NaN, 0) without a
    pair. The subtraction of (D-051.1), a common vector per polarity, cancels
    in every difference."""
    # Custom: pairs over positions (no library function for the archive's layout).
    means = np.asarray(X_tilde, dtype=np.float64).mean(axis=1)
    key = {(int(s), int(p)): k for k, (s, p) in enumerate(zip(ss.sweep_number, ss.position))}
    d2 = []
    for (s, p), k in key.items():
        j = key.get((s, p + int(lag_periods)))
        if j is not None and ss.polarity_is_dep[j] == ss.polarity_is_dep[k]:
            d2.append((means[j] - means[k]) ** 2)
    if not d2:
        return float("nan"), 0
    return 0.5 * float(np.mean(d2)), len(d2)
