"""noise_cell_stats -- Block 3 (analysis layer) of specs/SPEC.md: the D-051
statistics per window, per sweep and per cell, the window-level bootstrap SE
and the tilt of every sweep (D-049, D-051, D-073, D-076, D-102, D-107).

In plain words: every noise window (an enlarged SS window after the
subtraction of its group's median window, or an LS pre-window and its two
segments) gives a few numbers -- its SD and its textbook autocorrelation at a
few lags. The cell's statistic vector q_hat(c) is the median of each number
over the cell's windows; a sweep's vector q_hat(c, s) is the median over that
sweep's windows. The same functions run on simulated windows in Block 4, so
data and simulation go through one estimator (D-033). Each sweep's tilt --
the slow straight-line drift of its baseline -- is read here too: on an SS
sweep as the least-squares slope of its window means on time (TL1), on an LS
sweep as the line through its pre-window and end-window means (TL2); the
twins and the fit's simulator copy these numbers (D-102, D-107); the SS reading
(TL1) is defined in noise_windows since 2026-10-30, where the window pipeline
reads it at the 80 ms floor before the start (D-115), and is re-exported here.
No I/O, no plotting here.

Units: SDs in mV, autocorrelations dimensionless, tilts in mV/s, the legacy
semivariogram in mV^2; lags are given in ms and converted to samples by
round(lag_ms f_s / 1e3). Pure ASCII, LF.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

import noise_stats
from noise_windows import LSWindows, SSWindows, SweepTilts, WindowConfig, ls_segments, ss_sweep_tilts  # noqa: F401
# SweepTilts and ss_sweep_tilts ((TL1)) live in noise_windows since 2026-10-30 (D-115: the window pipeline reads
# the SS tilt before the start and removes its offsets); they are re-exported here for the callers of Block 3.

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
#  The tilt of each sweep (D-102.2: SS, (TL1); D-107.1: LS, (TL2))
# ===========================================================================
# SweepTilts and ss_sweep_tilts ((TL1), D-102.2 as moved by D-115.2): see noise_windows.


def ls_sweep_tilts(ls: LSWindows) -> SweepTilts:
    """(TL2): per LS sweep, (mean of the end window - mean of the finite
    pre-window samples) / (distance between the two windows' centres), the
    centres the means of their sample times from the sweep's first sample.
    NaN with 'ls_tilt_none:<sweep>:tail_short' when the end window is not
    usable (ls.end_status), ':nonfinite' when a mean is not finite."""
    b, flags = [], []
    for k, s in enumerate(ls.sweep_number):
        if str(ls.end_status[k]) != "ok":
            b.append(float("nan")); flags.append("ls_tilt_none:%d:tail_short" % int(s))
            continue
        pre = np.asarray(ls.pre[k], dtype=np.float64)
        jj = np.flatnonzero(np.isfinite(pre))
        end = np.asarray(ls.end[k], dtype=np.float64)
        if jj.size == 0 or end.size == 0 or not np.all(np.isfinite(end)):
            b.append(float("nan")); flags.append("ls_tilt_none:%d:nonfinite" % int(s))
            continue
        y_pre, t_pre = float(pre[jj].mean()), float(jj.mean()) * ls.dt
        y_end = float(end.mean())
        t_end = float(ls.end_start_s[k]) + 0.5 * (end.size - 1) * ls.dt
        b.append((y_end - y_pre) / (t_end - t_pre))
    K = len(ls.sweep_number)
    return SweepTilts(protocol="ls", sweep_number=np.asarray(ls.sweep_number, dtype=np.int64),
                      b_mV_per_s=np.asarray(b, dtype=np.float64), se_mV_per_s=np.full(K, np.nan),
                      n_windows=np.full(K, 2, dtype=np.int64), flags=tuple(flags))


# ===========================================================================
#  LEGACY (superseded by D-102 / D-107; unused by the pipeline): the data
#  semivariogram of the window means, (T1)'s data side (D-034.1, D-051 (iii))
# ===========================================================================
def window_mean_semivariogram(X_tilde: np.ndarray, ss: SSWindows, lag_periods: int = 4
                              ) -> Tuple[float, int]:
    """LEGACY (kept for the dev checks of the tilt study). V_hat_c(h), h =
    lag_periods onset periods: half the mean squared
    difference of the window means of two windows of the same sweep and the
    same polarity whose positions differ by lag_periods. (NaN, 0) without a
    pair. A subtraction of a common vector per polarity and sweep (D-051.1 or
    D-103.1) cancels in every difference."""
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
