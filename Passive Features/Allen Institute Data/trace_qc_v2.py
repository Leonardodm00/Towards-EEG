"""
trace_qc_v2.py
==============

Per-scale, decimated-baseline AR null for the Square-Subthreshold (SS) trace-QC,
with per-cell self-calibration of the noise null, Bonferroni cross-scale
combination, and a mean-residual misfit diagnostic.

This module is a *layer on top of* ``trace_qc.py``.  It re-uses the pure
primitives there (decimation, lag-1 autocorrelation, Yule-Walker AR fit, AR
simulation, the bounded single-exponential fit, onset/offset detection) and adds
only the new statistics.  The original full-rate AR(15) path in ``trace_qc.py``
is left untouched and is reachable here as ``null_generator="ar_fullrate"`` for
A/B comparison; it is also the reference the smoke test uses to *reproduce the
over-rejection bug*.

WHAT IS NEW vs trace_qc.py  (maps to Part III of trace_qc_description.md)
------------------------------------------------------------------------
III.A  Per-scale decimated-baseline AR null.  For each fixed decimation scale
       s (ms), the pooled signal-free baseline is decimated by q(s)=round(s*f_s),
       a pooled Yule-Walker AR of order  p(s)=round(M_ms / s)  (AIC-selected up
       to that cap) is fit IN THE DECIMATED DOMAIN, and the null distribution of
       the lag-1 autocorrelation rho_1(s) is simulated directly at the decimated
       sample count of one test window.  The memory  M_ms = p(s)*s  is a single,
       rate- and scale-invariant, physically meaningful parameter ("p in ms").
       Cross-scale combination is BONFERRONI on the per-scale p-values:
       reject iff  min_s p_s < alpha / S,  combined p = min(1, S * min_s p_s),
       where S is the number of admissible scales for the cell.

III.B  Baseline source.  The noise model is fit from the long pre-first-pulse
       rest epoch(s) supplied by the caller (``baseline_segments``), pooled across
       sweeps/sessions, NOT from the short per-pulse pre-pulse baselines.  A small
       lead (``selfcal_lead_trim_ms``) is trimmed from each epoch start to drop
       amplifier settling.

III.C  Widened tau band (defaults tau_lo_factor=0.25, tau_hi_factor=4.0).

III.D  Two diagnostics, saved per cell:
       (1) Null self-calibration.  Over R repeated random 50/50 splits of equal-
           length contiguous baseline chunks into a fit set and a held-out test
           set, the per-scale AR is fit on the fit set and the IDENTICAL structure
           test (no exponential fit) is run on each test chunk against that null.
           Under a correct null the per-scale p-values are ~ Uniform(0,1).  The
           memory M_ms is swept over a grid; the LARGEST M_ms whose pooled
           p-value histogram is not rejected as non-uniform (KS) is selected.
           The self-cal-admissible scale bank also becomes the per-pulse scoring
           bank for the cell.
       (2) Mean residual across pulses.  Per-pulse residuals are aligned at the
           fit-window start and averaged (+/- SEM) within (polarity, amplitude
           band) groups.  A deterministic multi-exponential misfit survives
           averaging as a coherent curve; pure noise averages to ~ 0.

III.E  Smoke test (``run_smoke_test_v2`` / ``--smoke-test``) asserting:
       (i)   the per-scale null gives ~ Uniform p-values on pure AR-coloured noise
             (calibrated);
       (ii)  it still rejects an injected deterministic double-exponential misfit
             (retains power), and the mean residual is coherent;
       (iii) the OLD full-rate AR(15) null over-rejects the same coloured noise
             (reproduces the bug) -> establishes the fix as causal.

ARCHITECTURE (separation of concerns preserved)
-----------------------------------------------
re-used primitives : trace_qc.{decimate_rows, lag1_autocorr_rows, fit_ar_pooled,
                               simulate_ar, fit_bounded_single_exp, _exp_decay,
                               _detect_onset_offset}
null layer (new)   : fit_perscale_ar, perscale_null_rho1, build_scoring_nulls
admissibility      : compute_admissible_bank
self-cal layer     : run_self_calibration
test layer (new)   : score_pulse_perscale          (per-pulse, Bonferroni)
diagnostics        : compute_mean_residuals
orchestration      : run_trace_qc_for_cell_v2       (no plotting / no I/O inside scoring)
plotting layer     : plot_* (downstream)
I/O layer          : save_cell_qc_outputs_v2
smoke test         : run_smoke_test_v2

Every scoring function is pure (numpy in -> numbers out); plotting and disk I/O
are strictly downstream.

Dependencies: numpy, scipy, matplotlib, and trace_qc.py on the path.
"""
from __future__ import annotations

import os
import json
import argparse
import warnings
from dataclasses import dataclass, field, asdict
from typing import Sequence

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- re-use the tested primitives from the live module -----------------------
import trace_qc as tq
from trace_qc import (
    decimate_rows,
    lag1_autocorr_rows,
    fit_ar_pooled,
    simulate_ar,
    fit_bounded_single_exp,
    _exp_decay,
    _detect_onset_offset,
    QCConfig as _QCConfigFullrate,   # for the "ar_fullrate" A/B path
)


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class QCConfigV2:
    # --- windowing (all in ms) ---  (kept identical to trace_qc.QCConfig)
    fit_delay_ms: float = 4.0
    fit_window_ms: float = 100.0
    test_window_ms: float | None = None      # None -> = fit_window_ms; also the self-cal chunk length
    baseline_guard_ms: float = 1.0
    min_baseline_ms: float = 3.0

    # --- bounded single-exponential fit ---  (III.C: widened band) ---
    tau_lo_factor: float = 0.25
    tau_hi_factor: float = 4.0

    # --- decimation-timescale CANDIDATE bank (ms); admissibility prunes it ---
    decim_scales_ms: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0, 20.0)

    # --- structure statistic (lag-1 only in v2) ---
    statistic: str = "lag1"

    # --- noise null ---
    #   "ar_perscale" : the new per-scale decimated-baseline AR null (default)
    #   "ar_fullrate" : the OLD full-rate AR(ar_max_order) null (delegated to trace_qc)
    #   "block"       : moving-block bootstrap (delegated to trace_qc)
    null_generator: str = "ar_perscale"
    n_surrogates: int = 500                  # B
    ar_max_memory_ms: float = 5.0            # M_ms starting value / used if self-cal disabled
    m_ms_grid: tuple[float, ...] = (2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0)
    # full-rate / block knobs only used by the delegated A/B paths
    ar_max_order: int = 15
    block_len_ms: float = 4.0

    # --- self-calibration (III.D.1) ---
    selfcal_enable: bool = True
    selfcal_n_repeats: int = 80              # R repeated 50/50 splits
    selfcal_lead_trim_ms: float = 5.0        # drop this much from each baseline epoch start
    selfcal_min_window_samples: int = 5      # min decimated samples per window to estimate rho_1(s)
    selfcal_min_fit_samples: int = 20        # "n_min": min decimated baseline samples to fit AR(s)
    selfcal_min_chunks: int = 4              # min number of test-window-length chunks to attempt self-cal
    # Flatness is judged by the KS DISTANCE D_n = sup_x |F_n(x) - x| (bounded, N-stable),
    # NOT the KS p-value (over-powered at the thousands of pooled p-values self-cal produces).
    # A scale is "flat" iff its per-scale D_n <= selfcal_flat_ks_dist; M_ms is the LARGEST
    # value for which max_s D_n(s) <= this. (KS p-value retained only as a reported diagnostic.)
    selfcal_flat_ks_dist: float = 0.06
    selfcal_flat_ks_alpha: float = 0.05      # reported only (uniformity p-value), not used for selection
    # Only scales with at least this many decimated samples per window contribute to the
    # flatness verdict (max_s D_n). Coarse scales with too few samples have an inherently
    # granular p-value CDF (large D_n that reflects sampling, not miscalibration); they are
    # still SCORED, but do not decide M_ms. Set to 0 to let every admissible scale vote.
    selfcal_flat_min_samples: int = 15
    # Under a fit/test split the held-out null inherits AR-estimation uncertainty, which
    # imposes an irreducible D_n floor (the plug-in surrogate ignores parameter uncertainty;
    # a 50/50 split roughly doubles it relative to the full-baseline production fit). So if
    # the absolute threshold above is unreachable, M_ms is chosen as the LARGEST value whose
    # D_n is within this slack of the minimum achievable D_n ("largest as-flat-as-the-best"),
    # which still caps M_ms so the null cannot quietly absorb slow misfit.
    selfcal_flat_slack: float = 0.03

    # --- decision ---
    alpha: float = 0.05

    # --- auxiliary sharp-transient gate ---
    use_spike_gate: bool = True
    spike_rel_thresh: float = 6.0

    # --- mean-residual diagnostic (III.D.2) ---
    amp_band_round_pA: float = 10.0          # group pulses by round(peak_pA / this) * this

    # --- cell-level rule ---
    min_viable_training_pulses: int = 10

    # --- which polarities to QC, and which is "training" ---
    qc_polarities: tuple[str, ...] = ("dep", "hyp")
    fit_target: str = "hyp"

    # --- reproducibility ---
    seed: int = 0


# =============================================================================
# Result containers
# =============================================================================
@dataclass
class PulseQCResultV2:
    pulse_index: int
    polarity: str
    peak_pA: float
    qced: bool
    passed: bool
    reason: str                              # 'ok'|'pathological'|'spike'|'fit_failed'|'short_window'|'not_qced'|'no_null'
    p_combined: float = np.nan               # Bonferroni-combined p (min(1, S*min_s p_s))
    p_min: float = np.nan                    # min_s p_s (the quantity compared to alpha/S)
    winning_scale_ms: float = np.nan         # argmin_s p_s
    fit_A_mV: float = np.nan
    fit_tau_ms: float = np.nan
    fit_c_mV: float = np.nan
    spike_ratio: float = np.nan
    p_by_scale: dict = field(default_factory=dict)     # {scale_ms: p_s}
    rho1_by_scale: dict = field(default_factory=dict)  # {scale_ms: rho_1(s)}
    amp_band_pA: float = np.nan
    # heavy diagnostics (not serialised to CSV)
    _u_ms: np.ndarray | None = field(default=None, repr=False)
    _d_mV: np.ndarray | None = field(default=None, repr=False)
    _fit_mV: np.ndarray | None = field(default=None, repr=False)
    _resid_mV: np.ndarray | None = field(default=None, repr=False)


@dataclass
class SelfCalResult:
    status: str                              # 'flat'|'flat_relative'|'uncalibrated'|'insufficient_baseline'|'disabled'
    chosen_M_ms: float
    chosen_ks_p: float
    admissible_scales_ms: list
    n_chunks: int
    chunk_len_samples: int
    m_ms_grid: list
    ks_p_pooled: list                        # per M_ms
    reject_rate_at_alpha: list               # per M_ms (per-scale p_s < alpha rate; should ~ alpha)
    n_pvalues: list                          # per M_ms
    per_scale_ks_p_chosen: dict              # {scale_ms: KS p} at chosen M_ms (reported only)
    # KS-DISTANCE flatness (the SELECTION criterion; N-stable). D_n(s) = sup_x |F_n(x) - x|.
    ks_dist_max: list = field(default_factory=list)              # per M_ms: max_s D_n(s) over flat-voting scales
    chosen_ks_dist_max: float = np.nan                           # max_s D_n(s) at chosen M_ms
    per_scale_ks_dist_chosen: dict = field(default_factory=dict) # {scale_ms: D_n(s)} at chosen M_ms (all scales)
    flat_scales_ms: list = field(default_factory=list)           # scales that voted on the flatness verdict
    # heavy: arrays for plotting (not serialised)
    _pooled_pvalues_chosen: np.ndarray | None = field(default=None, repr=False)
    _per_scale_pvalues_chosen: dict | None = field(default=None, repr=False)


@dataclass
class MeanResidualResult:
    groups: dict                             # {(polarity, amp_band_pA): {'u_ms','mean','sem','n'}}
    coherence_index: dict                    # {(polarity, amp_band_pA): max|mean| / median(sem)}


@dataclass
class CellQCResultV2:
    specimen_id: object
    n_pulses: int
    n_qced: int
    n_passed_total: int
    n_viable_training: int
    cell_rejected: bool
    cell_reason: str
    null_generator: str
    selfcal: SelfCalResult | None
    scoring_bank_ms: list
    per_scale_ar: dict                       # {scale_ms: {'order','sigma2'}} for the scoring nulls
    config: dict
    pulse_results: list
    mean_residual: MeanResidualResult | None
    kept_pulses: list
    # heavy: scoring null ensembles for the rho-band plot (not serialised)
    _scoring_nulls: dict | None = field(default=None, repr=False)


# =============================================================================
# Small helpers
# =============================================================================
def _round_half_up(x: float) -> int:
    """Standard rounding (0.5 -> 1), avoiding numpy's banker's rounding."""
    return int(np.floor(float(x) + 0.5))


def _q_of_scale(scale_ms: float, fs_khz: float) -> int:
    return max(1, _round_half_up(scale_ms * fs_khz))


def _decimated_len(n_samples: int, q: int) -> int:
    """Exact length of decimate_rows output for an input of length n_samples."""
    if q <= 1:
        return int(n_samples)
    return int(decimate_rows(np.zeros((1, n_samples)), q).shape[1])


def _order_cap(M_ms: float, scale_ms: float) -> int:
    """p(s) = round(M_ms / s):  AR order cap in the decimated domain ('p in ms')."""
    return max(0, _round_half_up(M_ms / scale_ms))


def _prepare_baseline_segments(segments: Sequence[np.ndarray], fs_khz: float,
                               cfg: QCConfigV2) -> list[np.ndarray]:
    """Lead-trim and mean-subtract each raw pre-first-pulse epoch (III.B)."""
    lead = int(round(cfg.selfcal_lead_trim_ms * fs_khz))
    out = []
    for seg in segments:
        s = np.asarray(seg, float).ravel()
        if s.size > lead + 2:
            s = s[lead:]
        s = s - s.mean()
        if s.size >= 4:
            out.append(s)
    return out


def _chunk_segments(segments: Sequence[np.ndarray], chunk_len: int) -> list[np.ndarray]:
    """Chop each (contiguous) epoch into non-overlapping chunks of chunk_len samples.

    Chunks are the self-cal scoring unit; their length equals the per-pulse test
    window so each scored rho_1(s) has the SAME decimated sample count as the real
    test (finite-sample matching).  Within-chunk contiguity is preserved so each
    chunk is a valid input for decimation + AR fitting.
    """
    chunks = []
    for seg in segments:
        n = seg.size
        k = n // chunk_len
        for j in range(k):
            chunks.append(seg[j * chunk_len:(j + 1) * chunk_len])
    return chunks


# =============================================================================
# Admissibility (which scales survive for this cell)
# =============================================================================
def compute_admissible_bank(candidate_scales_ms: Sequence[float], fs_khz: float,
                            chunk_len: int, n_fit_chunks: int,
                            cfg: QCConfigV2) -> list[float]:
    """Drop coarse scales the baseline cannot support (III.D.1 'lower the maximum
    decimation factor when the window is too short').

    A scale s is admissible iff
      (a) per-window decimated count  n_win(s) = decimated_len(chunk_len, q(s))
          >= selfcal_min_window_samples         (enough to estimate rho_1(s)), AND
      (b) fit-set decimated count     n_fit(s) = n_fit_chunks * n_win(s)
          >= selfcal_min_fit_samples            (the "n_min" floor, enough to fit AR).
    Because n_win(s) is monotone decreasing in s, the failing scales are exactly
    the coarsest tail -> this is "lowering the maximum decimation factor".
    """
    bank = []
    for s in candidate_scales_ms:
        q = _q_of_scale(s, fs_khz)
        n_win = _decimated_len(chunk_len, q)
        n_fit = n_fit_chunks * n_win
        if n_win >= cfg.selfcal_min_window_samples and n_fit >= cfg.selfcal_min_fit_samples:
            bank.append(float(s))
    return bank


# =============================================================================
# Per-scale null layer (pure)
# =============================================================================
def fit_perscale_ar(fit_segments: Sequence[np.ndarray], scale_ms: float, M_ms: float,
                    fs_khz: float):
    """Fit a pooled Yule-Walker AR in the DECIMATED domain of one scale.

    Decimate each (mean-free, contiguous) fit segment by q(s), pool, and fit
    AR(p) with order cap p(s)=round(M_ms/s) (AIC-selected up to that cap; lags
    never cross a decimated-segment boundary).

    Returns (phi:(p,), sigma2:float, order:int).
    """
    q = _q_of_scale(scale_ms, fs_khz)
    dec_segs = []
    n_min_dec = None
    for seg in fit_segments:
        d = decimate_rows(np.asarray(seg, float)[None, :], q)[0]
        if d.size >= 2:
            dec_segs.append(d - d.mean())
            n_min_dec = d.size if n_min_dec is None else min(n_min_dec, d.size)
    if not dec_segs:
        return np.zeros(0), 1.0, 0
    cap = _order_cap(M_ms, scale_ms)
    # cannot fit an order >= available decimated length within a segment
    cap = max(0, min(cap, (n_min_dec - 2) if n_min_dec is not None else 0))
    return fit_ar_pooled(dec_segs, cap)


def perscale_null_rho1(phi: np.ndarray, sigma2: float, n_dec: int, B: int,
                       rng: np.random.Generator) -> np.ndarray:
    """Null distribution of lag-1 autocorrelation rho_1(s) at decimated length n_dec."""
    paths = simulate_ar(phi, sigma2, n_dec, B, rng)        # (B, n_dec) in the decimated domain
    null = lag1_autocorr_rows(paths)
    return null[np.isfinite(null)]


def build_scoring_nulls(baseline_segments: Sequence[np.ndarray], scoring_bank_ms: Sequence[float],
                        M_ms: float, test_len_samples: int, fs_khz: float, B: int,
                        rng: np.random.Generator) -> dict:
    """Per-scale AR fit on the FULL pooled baseline at the chosen M_ms, plus the
    rho_1(s) null ensemble at the decimated length of one TEST window.

    The null is simulated at  n_dec(s) = decimated_len(test_len_samples, q(s))  so
    it matches the decimated length of a real residual exactly (finite-sample
    matching).  Returns {scale_ms: {'phi','sigma2','order','null_rho1','n_dec'}}.
    """
    nulls = {}
    for s in scoring_bank_ms:
        q = _q_of_scale(s, fs_khz)
        phi, sig2, order = fit_perscale_ar(baseline_segments, s, M_ms, fs_khz)
        n_dec = _decimated_len(test_len_samples, q)
        null = perscale_null_rho1(phi, sig2, n_dec, B, rng)
        nulls[float(s)] = dict(phi=phi, sigma2=float(sig2), order=int(order),
                               null_rho1=null, n_dec=int(n_dec))
    return nulls


# =============================================================================
# Self-calibration (III.D.1)
# =============================================================================
def run_self_calibration(baseline_segments: Sequence[np.ndarray], fs_khz: float,
                         cfg: QCConfigV2, rng: np.random.Generator) -> SelfCalResult:
    """Choose M_ms (and the admissible scale bank) by held-out null calibration.

    For each M_ms on the grid, over R repeated random 50/50 splits of equal-length
    contiguous baseline chunks into a fit set and a held-out test set:
        - fit the per-scale AR on the fit set (decimated domain),
        - simulate the rho_1(s) null at the decimated test-window length,
        - score each held-out test chunk -> per-scale p_s,
    pool all p_s, and test uniformity by KS.  Select the LARGEST M_ms whose pooled
    p-value histogram is not rejected as non-uniform (KS p >= selfcal_flat_ks_alpha);
    fall back to the maximally-uniform M_ms otherwise.
    """
    prepared = _prepare_baseline_segments(baseline_segments, fs_khz, cfg)
    test_len = int(round((cfg.test_window_ms or cfg.fit_window_ms) * fs_khz))
    chunks = _chunk_segments(prepared, test_len)
    n_chunks = len(chunks)

    grid = list(cfg.m_ms_grid)

    if n_chunks < cfg.selfcal_min_chunks:
        return SelfCalResult(
            status="insufficient_baseline", chosen_M_ms=float(cfg.ar_max_memory_ms),
            chosen_ks_p=np.nan, admissible_scales_ms=[], n_chunks=n_chunks,
            chunk_len_samples=test_len, m_ms_grid=grid, ks_p_pooled=[], reject_rate_at_alpha=[],
            n_pvalues=[], per_scale_ks_p_chosen={})

    n_fit_chunks = n_chunks // 2
    admissible = compute_admissible_bank(cfg.decim_scales_ms, fs_khz, test_len,
                                         n_fit_chunks, cfg)
    if not admissible:
        return SelfCalResult(
            status="insufficient_baseline", chosen_M_ms=float(cfg.ar_max_memory_ms),
            chosen_ks_p=np.nan, admissible_scales_ms=[], n_chunks=n_chunks,
            chunk_len_samples=test_len, m_ms_grid=grid, ks_p_pooled=[], reject_rate_at_alpha=[],
            n_pvalues=[], per_scale_ks_p_chosen={})

    B = cfg.n_surrogates
    idx_all = np.arange(n_chunks)
    n_dec_by_scale = {s: _decimated_len(test_len, _q_of_scale(s, fs_khz)) for s in admissible}

    # scales that are allowed to decide the flatness verdict (enough samples for a smooth CDF)
    flat_scales = [s for s in admissible
                   if n_dec_by_scale[s] >= cfg.selfcal_flat_min_samples] or list(admissible)

    ks_p_pooled, reject_rate, n_pvalues, ks_dist_max = [], [], [], []
    pooled_by_mms, per_scale_by_mms = {}, {}

    def _ks_dist(arr):
        """KS distance D_n = sup_x |F_n(x) - x| against Uniform(0,1)."""
        return float(stats.kstest(arr, "uniform").statistic) if arr.size >= 5 else np.nan

    for M_ms in grid:
        pooled = []
        per_scale = {s: [] for s in admissible}
        for _ in range(cfg.selfcal_n_repeats):
            rng.shuffle(idx_all)
            fit_idx = idx_all[:n_fit_chunks]
            test_idx = idx_all[n_fit_chunks:]
            fit_chunks = [chunks[i] for i in fit_idx]
            test_chunks = [chunks[i] for i in test_idx]
            for s in admissible:
                q = _q_of_scale(s, fs_khz)
                phi, sig2, _ = fit_perscale_ar(fit_chunks, s, M_ms, fs_khz)
                null = perscale_null_rho1(phi, sig2, n_dec_by_scale[s], B, rng)
                if null.size == 0:
                    continue
                for tc in test_chunks:
                    d = decimate_rows(tc[None, :], q)
                    rho = lag1_autocorr_rows(d)[0]
                    if not np.isfinite(rho):
                        continue
                    p_s = (1 + int(np.sum(null >= rho))) / (1 + null.size)
                    pooled.append(p_s)
                    per_scale[s].append(p_s)
        pooled = np.asarray(pooled, float)
        per_scale = {s: np.asarray(v, float) for s, v in per_scale.items()}
        pooled_by_mms[M_ms] = pooled
        per_scale_by_mms[M_ms] = per_scale
        # SELECTION criterion: worst per-scale KS distance over the FLAT-VOTING scales
        # (each sufficiently-sampled scale must be flat).
        dns = [_ks_dist(per_scale[s]) for s in flat_scales]
        dmax = float(np.nanmax(dns)) if np.any(np.isfinite(dns)) else np.nan
        ks_dist_max.append(dmax)
        # reported-only pooled diagnostics
        ks_p_pooled.append(float(stats.kstest(pooled, "uniform").pvalue) if pooled.size >= 5 else np.nan)
        reject_rate.append(float(np.mean(pooled < cfg.alpha)) if pooled.size >= 5 else np.nan)
        n_pvalues.append(int(pooled.size))

    dmax_arr = np.asarray(ks_dist_max, float)
    finite = np.isfinite(dmax_arr)
    abs_flat = (dmax_arr <= cfg.selfcal_flat_ks_dist) & finite
    if np.any(abs_flat):
        # LARGEST M_ms whose worst-scale KS distance is within the absolute tolerance
        chosen_i = int(max(np.flatnonzero(abs_flat)))
        status = "flat"
    elif np.any(finite):
        # absolute floor unreachable (split-induced fit uncertainty): pick the LARGEST M_ms
        # whose D_n is within slack of the best achievable D_n
        d_min = float(np.nanmin(dmax_arr))
        near = (dmax_arr <= d_min + cfg.selfcal_flat_slack) & finite
        chosen_i = int(max(np.flatnonzero(near)))
        status = "flat_relative"
    else:
        chosen_i = 0
        status = "uncalibrated"

    chosen_M = float(grid[chosen_i])
    chosen_ksp = float(np.asarray(ks_p_pooled, float)[chosen_i])
    chosen_dmax = float(dmax_arr[chosen_i])
    per_scale_ks, per_scale_dn = {}, {}
    for s, v in per_scale_by_mms[chosen_M].items():
        per_scale_ks[float(s)] = float(stats.kstest(v, "uniform").pvalue) if v.size >= 5 else np.nan
        per_scale_dn[float(s)] = _ks_dist(v)

    return SelfCalResult(
        status=status, chosen_M_ms=chosen_M, chosen_ks_p=chosen_ksp,
        admissible_scales_ms=[float(s) for s in admissible], n_chunks=n_chunks,
        chunk_len_samples=test_len, m_ms_grid=grid, ks_p_pooled=ks_p_pooled,
        reject_rate_at_alpha=reject_rate, n_pvalues=n_pvalues,
        per_scale_ks_p_chosen=per_scale_ks,
        ks_dist_max=ks_dist_max, chosen_ks_dist_max=chosen_dmax,
        per_scale_ks_dist_chosen=per_scale_dn, flat_scales_ms=[float(s) for s in flat_scales],
        _pooled_pvalues_chosen=pooled_by_mms[chosen_M],
        _per_scale_pvalues_chosen=per_scale_by_mms[chosen_M])



# =============================================================================
# Per-pulse residual (re-uses the tested fit pipeline) + per-scale scoring
# =============================================================================
def _pulse_residual(pulse: dict, tau_m_ms: float, cfg: QCConfigV2):
    """Compute the post-pulse residual over the test window using the bounded
    single-exponential fit.  Returns (status, info) where status in
    {'ok','fit_failed','short_window'} and info carries the residual + fit + axes.
    """
    t = np.asarray(pulse["t"], float)
    v = np.asarray(pulse["v"], float)
    i = np.asarray(pulse["i"], float)
    fs_khz = float(pulse.get("sampling_rate_Hz", 1.0 / np.median(np.diff(t)))) / 1e3

    onset, offset = _detect_onset_offset(t, i, float(pulse["stim_duration_s"]))
    if onset is None:
        return "fit_failed", None

    guard = int(round(cfg.baseline_guard_ms * fs_khz))
    pre = v[: max(0, onset - guard)]
    if pre.size < int(round(cfg.min_baseline_ms * fs_khz)):
        return "short_window", None
    B_p = float(pre.mean())

    test_len_ms = cfg.test_window_ms if cfg.test_window_ms is not None else cfg.fit_window_ms
    w0 = offset + int(round(cfg.fit_delay_ms * fs_khz))
    w1_fit = w0 + int(round(cfg.fit_window_ms * fs_khz))
    w1_test = w0 + int(round(test_len_ms * fs_khz))
    if w1_test > v.size or (w1_fit - w0) < 8:
        return "short_window", None

    u_fit = (t[w0:w1_fit] - t[w0]) * 1e3
    y_fit = v[w0:w1_fit] - B_p
    fit = fit_bounded_single_exp(u_fit, y_fit, tau_m_ms, cfg.tau_lo_factor, cfg.tau_hi_factor)
    if not fit["ok"]:
        return "fit_failed", None

    u_test = (t[w0:w1_test] - t[w0]) * 1e3
    d_test = v[w0:w1_test] - B_p
    fit_test = _exp_decay(u_test, fit["A"], fit["tau"], fit["c"])
    resid = d_test - fit_test
    return "ok", dict(fit=fit, u=u_test, d=d_test, fit_mV=fit_test, resid=resid, fs_khz=fs_khz)


def score_pulse_perscale(pulse: dict, idx: int, tau_m_ms: float, cfg: QCConfigV2,
                         scoring_nulls: dict, scoring_bank_ms: Sequence[float]) -> PulseQCResultV2:
    """Score one pulse against the per-scale nulls and combine by Bonferroni.

    reject iff  min_s p_s < alpha / S,  combined p = min(1, S * min_s p_s).
    """
    pol = pulse["polarity"]
    peak = float(pulse.get("peak_pA", np.nan))
    amp_band = (np.round(peak / cfg.amp_band_round_pA) * cfg.amp_band_round_pA
                if np.isfinite(peak) else np.nan)
    res = PulseQCResultV2(idx, pol, peak, qced=True, passed=False, reason="ok",
                          amp_band_pA=float(amp_band) if np.isfinite(amp_band) else np.nan)

    status, info = _pulse_residual(pulse, tau_m_ms, cfg)
    if status != "ok":
        res.passed, res.reason = False, status
        return res

    fit, resid, fs_khz = info["fit"], info["resid"], info["fs_khz"]
    res.fit_A_mV, res.fit_tau_ms, res.fit_c_mV = fit["A"], fit["tau"], fit["c"]
    res._u_ms, res._d_mV, res._fit_mV, res._resid_mV = info["u"], info["d"], info["fit_mV"], resid

    # auxiliary sharp-transient gate (decimation would smooth these away)
    robust_amp = max(abs(fit["A"]), 1e-6)
    res.spike_ratio = float(np.max(np.abs(resid)) / robust_amp)

    if not scoring_bank_ms:
        res.reason = "no_null"
        # still apply the spike gate verdict if it fires
        if cfg.use_spike_gate and res.spike_ratio > cfg.spike_rel_thresh:
            res.passed, res.reason = False, "spike"
        else:
            res.passed = True   # cannot test shape; do not reject on a missing null
        return res

    # per-scale p-values
    p_by_scale, rho_by_scale = {}, {}
    for s in scoring_bank_ms:
        q = _q_of_scale(s, fs_khz)
        d = decimate_rows(resid[None, :], q)
        rho = lag1_autocorr_rows(d)[0]
        null = scoring_nulls[float(s)]["null_rho1"]
        if not np.isfinite(rho) or null.size == 0:
            continue
        p_s = (1 + int(np.sum(null >= rho))) / (1 + null.size)
        p_by_scale[float(s)] = p_s
        rho_by_scale[float(s)] = float(rho)

    res.p_by_scale, res.rho1_by_scale = p_by_scale, rho_by_scale
    if not p_by_scale:
        res.reason = "no_null"
        res.passed = not (cfg.use_spike_gate and res.spike_ratio > cfg.spike_rel_thresh)
        if not res.passed:
            res.reason = "spike"
        return res

    S = len(p_by_scale)
    win_scale = min(p_by_scale, key=p_by_scale.get)
    p_min = p_by_scale[win_scale]
    res.p_min = float(p_min)
    res.winning_scale_ms = float(win_scale)
    res.p_combined = float(min(1.0, S * p_min))

    # spike gate first (it is a different failure mode), then the structure test
    if cfg.use_spike_gate and res.spike_ratio > cfg.spike_rel_thresh:
        res.passed, res.reason = False, "spike"
        return res
    if p_min < cfg.alpha / S:
        res.passed, res.reason = False, "pathological"
    else:
        res.passed, res.reason = True, "ok"
    return res


# =============================================================================
# Mean-residual diagnostic (III.D.2)
# =============================================================================
def compute_mean_residuals(pulse_results: Sequence[PulseQCResultV2],
                           cfg: QCConfigV2) -> MeanResidualResult:
    """Average aligned per-pulse residuals within (polarity, amplitude band) groups.

    A deterministic multi-exponential misfit survives averaging as a coherent
    non-zero curve; pure noise averages to ~ 0.  coherence_index = max|mean| /
    median(SEM); >> 1 indicates a coherent (deterministic) component.
    """
    buckets: dict = {}
    for r in pulse_results:
        if not r.qced or r._resid_mV is None or r._u_ms is None:
            continue
        key = (r.polarity, float(r.amp_band_pA) if np.isfinite(r.amp_band_pA) else np.nan)
        buckets.setdefault(key, []).append((r._u_ms, r._resid_mV))

    groups, coherence = {}, {}
    for key, items in buckets.items():
        nmin = min(u.size for u, _ in items)
        if nmin < 4:
            continue
        u0 = items[0][0][:nmin]
        R = np.vstack([res[:nmin] for _, res in items])    # (n_pulses, nmin)
        mean = R.mean(axis=0)
        sem = R.std(axis=0, ddof=1) / np.sqrt(R.shape[0]) if R.shape[0] > 1 else np.full(nmin, np.nan)
        groups[key] = dict(u_ms=u0, mean=mean, sem=sem, n=int(R.shape[0]))
        med_sem = np.nanmedian(sem) if np.isfinite(sem).any() else np.nan
        coherence[key] = float(np.max(np.abs(mean)) / med_sem) if (med_sem and med_sem > 0) else np.nan
    return MeanResidualResult(groups=groups, coherence_index=coherence)


# =============================================================================
# Orchestration (per cell)
# =============================================================================
def run_trace_qc_for_cell_v2(pulses: Sequence[dict], tau_m_ms: float,
                             baseline_segments: Sequence[np.ndarray] | None = None,
                             specimen_id: object = None, cfg: QCConfigV2 | None = None,
                             output_dir: str | None = None, make_plots: bool = True) -> CellQCResultV2:
    """Full per-cell trace QC with the per-scale self-calibrated null.

    Parameters
    ----------
    pulses : list of pulse dicts {t, v, i, polarity, peak_pA, stim_duration_s, sampling_rate_Hz}.
    tau_m_ms : dataset membrane time constant for this cell (ms).
    baseline_segments : list of raw pre-first-pulse rest epochs (1D arrays, full rate,
        same f_s as the pulses), pooled across sweeps/sessions (III.B).  If None,
        falls back to the pooled per-pulse pre-pulse baselines (compatibility only).
    """
    cfg = cfg or QCConfigV2()
    rng = np.random.default_rng(cfg.seed)
    fs_khz = float(pulses[0].get("sampling_rate_Hz", 1.0 / np.median(np.diff(np.asarray(pulses[0]["t"], float))))) / 1e3
    test_len = int(round((cfg.test_window_ms or cfg.fit_window_ms) * fs_khz))

    # ---- A/B paths: delegate the OLD full-rate / block nulls to trace_qc -----
    if cfg.null_generator in ("ar_fullrate", "block"):
        return _run_via_fullrate(pulses, tau_m_ms, cfg, specimen_id, output_dir, make_plots)

    # ---- baseline source (III.B) --------------------------------------------
    if baseline_segments is None:
        baseline_segments = _fallback_perpulse_baselines(pulses, cfg)

    # ---- 1. self-calibration: choose M_ms + admissible bank (III.D.1) -------
    if cfg.selfcal_enable:
        selfcal = run_self_calibration(baseline_segments, fs_khz, cfg, rng)
        scoring_bank = list(selfcal.admissible_scales_ms)
        M_ms = selfcal.chosen_M_ms
    else:
        selfcal = SelfCalResult(status="disabled", chosen_M_ms=float(cfg.ar_max_memory_ms),
                                chosen_ks_p=np.nan, admissible_scales_ms=list(cfg.decim_scales_ms),
                                n_chunks=0, chunk_len_samples=test_len, m_ms_grid=list(cfg.m_ms_grid),
                                ks_p_pooled=[], reject_rate_at_alpha=[], n_pvalues=[],
                                per_scale_ks_p_chosen={})
        scoring_bank = list(cfg.decim_scales_ms)
        M_ms = cfg.ar_max_memory_ms

    # ---- 2. per-scale scoring nulls on the FULL baseline at M_ms ------------
    prepared_full = _prepare_baseline_segments(baseline_segments, fs_khz, cfg)
    scoring_nulls = (build_scoring_nulls(prepared_full, scoring_bank, M_ms, test_len, fs_khz,
                                         cfg.n_surrogates, rng)
                     if scoring_bank and prepared_full else {})
    if not scoring_nulls:
        scoring_bank = []   # nothing to test against
    per_scale_ar = {s: {"order": d["order"], "sigma2": d["sigma2"]}
                    for s, d in scoring_nulls.items()}

    # ---- 3. per-pulse QC ----------------------------------------------------
    results: list[PulseQCResultV2] = []
    for k, p in enumerate(pulses):
        if p["polarity"] in cfg.qc_polarities:
            results.append(score_pulse_perscale(p, k, tau_m_ms, cfg, scoring_nulls, scoring_bank))
        else:
            results.append(PulseQCResultV2(k, p["polarity"], float(p.get("peak_pA", np.nan)),
                                           qced=False, passed=True, reason="not_qced"))

    # ---- 4. cell-level viability rule ---------------------------------------
    n_viable_train = sum(r.passed for r, p in zip(results, pulses)
                         if p["polarity"] == cfg.fit_target)
    cell_rejected = n_viable_train < cfg.min_viable_training_pulses
    cell_reason = (f"only {n_viable_train} viable {cfg.fit_target} pulses "
                   f"(< {cfg.min_viable_training_pulses})") if cell_rejected else "ok"
    kept = [] if cell_rejected else [p for p, r in zip(pulses, results) if r.passed]

    # ---- 5. mean-residual diagnostic (III.D.2) ------------------------------
    mean_resid = compute_mean_residuals(results, cfg)

    cell = CellQCResultV2(
        specimen_id=specimen_id, n_pulses=len(pulses),
        n_qced=sum(r.qced for r in results), n_passed_total=sum(r.passed for r in results),
        n_viable_training=n_viable_train, cell_rejected=cell_rejected, cell_reason=cell_reason,
        null_generator=cfg.null_generator, selfcal=selfcal, scoring_bank_ms=list(scoring_bank),
        per_scale_ar=per_scale_ar, config=asdict(cfg), pulse_results=results,
        mean_residual=mean_resid, kept_pulses=kept, _scoring_nulls=scoring_nulls)

    if output_dir is not None:
        save_cell_qc_outputs_v2(cell, output_dir, make_plots=make_plots, cfg=cfg)
    return cell


def _fallback_perpulse_baselines(pulses, cfg: QCConfigV2) -> list[np.ndarray]:
    """Compatibility fallback when no pre-first-pulse epoch is supplied (NOT the
    III.B path): pool the short per-pulse pre-pulse baselines."""
    segs = []
    for p in pulses:
        if p["polarity"] not in cfg.qc_polarities:
            continue
        t = np.asarray(p["t"], float); v = np.asarray(p["v"], float); i = np.asarray(p["i"], float)
        fs_khz = float(p.get("sampling_rate_Hz", 1.0 / np.median(np.diff(t)))) / 1e3
        onset, _ = _detect_onset_offset(t, i, float(p["stim_duration_s"]))
        if onset is None:
            continue
        guard = int(round(cfg.baseline_guard_ms * fs_khz))
        pre = v[: max(0, onset - guard)]
        if pre.size >= int(round(cfg.min_baseline_ms * fs_khz)):
            segs.append(pre - pre.mean())
    return segs


def _run_via_fullrate(pulses, tau_m_ms, cfg: QCConfigV2, specimen_id, output_dir, make_plots):
    """Delegate to the original trace_qc.run_trace_qc_for_cell for the A/B null
    generators ('ar_fullrate' -> trace_qc 'ar'; 'block' -> 'block')."""
    gen = "ar" if cfg.null_generator == "ar_fullrate" else "block"
    old_scales = tuple(s for s in cfg.decim_scales_ms)   # keep the same bank for a fair A/B
    cfg_old = _QCConfigFullrate(
        fit_delay_ms=cfg.fit_delay_ms, fit_window_ms=cfg.fit_window_ms,
        test_window_ms=cfg.test_window_ms, baseline_guard_ms=cfg.baseline_guard_ms,
        min_baseline_ms=cfg.min_baseline_ms, tau_lo_factor=cfg.tau_lo_factor,
        tau_hi_factor=cfg.tau_hi_factor, decim_scales_ms=old_scales, statistic="lag1",
        null_generator=gen, ar_max_order=cfg.ar_max_order, block_len_ms=cfg.block_len_ms,
        n_surrogates=cfg.n_surrogates, alpha=cfg.alpha, use_spike_gate=cfg.use_spike_gate,
        spike_rel_thresh=cfg.spike_rel_thresh, min_viable_training_pulses=cfg.min_viable_training_pulses,
        qc_polarities=cfg.qc_polarities, fit_target=cfg.fit_target, seed=cfg.seed)
    old = tq.run_trace_qc_for_cell(pulses, tau_m_ms, specimen_id=specimen_id, cfg=cfg_old,
                                   output_dir=output_dir, make_plots=make_plots)
    # wrap into the v2 container shape (per-pulse fields mapped; no self-cal)
    res = []
    for r in old.pulse_results:
        res.append(PulseQCResultV2(
            r.pulse_index, r.polarity, r.peak_pA, r.qced, r.passed, r.reason,
            p_combined=r.p_value, p_min=r.p_value, winning_scale_ms=r.argmax_scale_ms,
            fit_A_mV=r.fit_A_mV, fit_tau_ms=r.fit_tau_ms, fit_c_mV=r.fit_c_mV,
            spike_ratio=r.spike_ratio))
    return CellQCResultV2(
        specimen_id=old.specimen_id, n_pulses=old.n_pulses, n_qced=old.n_qced,
        n_passed_total=old.n_passed_total, n_viable_training=old.n_viable_training,
        cell_rejected=old.cell_rejected, cell_reason=old.cell_reason,
        null_generator=cfg.null_generator, selfcal=None, scoring_bank_ms=list(old_scales),
        per_scale_ar={}, config=asdict(cfg), pulse_results=res, mean_residual=None,
        kept_pulses=old.kept_pulses)


# =============================================================================
# Plotting layer (downstream)
# =============================================================================
def plot_selfcal_histogram(selfcal: SelfCalResult, path: str, alpha: float):
    if selfcal is None or selfcal._pooled_pvalues_chosen is None or selfcal._pooled_pvalues_chosen.size == 0:
        return
    p = selfcal._pooled_pvalues_chosen
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    nb = 20
    ax.hist(p, bins=nb, range=(0, 1), color="0.7", edgecolor="0.4", density=True)
    ax.axhline(1.0, color="tab:green", ls="--", lw=1.4, label="Uniform(0,1)")
    ax.set_xlabel("held-out per-scale p-value")
    ax.set_ylabel("density")
    ax.set_title(f"Null self-calibration — M_ms*={selfcal.chosen_M_ms:g} ms  "
                 f"(max_s D_n={selfcal.chosen_ks_dist_max:.3f}, KS p={selfcal.chosen_ks_p:.3f}, "
                 f"status={selfcal.status})")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_selfcal_mms_sweep(selfcal: SelfCalResult, path: str, dist_thresh: float):
    if selfcal is None or not selfcal.m_ms_grid:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(selfcal.m_ms_grid, selfcal.ks_dist_max, "o-", color="tab:blue",
            label=r"worst-scale KS distance $\max_s D_n(s)$")
    ax.axhline(dist_thresh, color="tab:red", ls=":", lw=1.4,
               label=f"flat threshold (D_n={dist_thresh:g})")
    ax.axvline(selfcal.chosen_M_ms, color="tab:green", ls="--", lw=1.4,
               label=f"chosen M_ms*={selfcal.chosen_M_ms:g} ms")
    ax.set_xlabel("AR memory M_ms (ms)")
    ax.set_ylabel(r"worst-scale KS distance $\max_s D_n(s)$")
    ax.set_title("M_ms sweep: pick the LARGEST still-flat memory (D_n <= threshold)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_rho_bands(cell: CellQCResultV2, path: str):
    """Per-scale null rho_1(s) band with the real pulses' observed rho_1(s)."""
    nulls = cell._scoring_nulls
    if not nulls:
        return
    scales = sorted(nulls.keys())
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    xs = np.arange(len(scales))
    for j, s in enumerate(scales):
        nb = nulls[s]["null_rho1"]
        lo, md, hi = np.percentile(nb, [2.5, 50, 97.5])
        ax.add_patch(plt.Rectangle((j - 0.28, lo), 0.56, hi - lo, color="0.8", zorder=1))
        ax.plot([j - 0.28, j + 0.28], [md, md], color="0.4", lw=1.4, zorder=2)
    # overlay observed rho_1(s), coloured by verdict
    for r in cell.pulse_results:
        if not r.qced or not r.rho1_by_scale:
            continue
        col = "tab:green" if r.passed else "tab:red"
        for s, rho in r.rho1_by_scale.items():
            j = scales.index(float(s))
            ax.plot(j + np.random.uniform(-0.12, 0.12), rho, ".", color=col, ms=3, alpha=0.5, zorder=3)
    ax.set_xticks(xs); ax.set_xticklabels([f"{s:g}" for s in scales])
    ax.set_xlabel("decimation scale s (ms)")
    ax.set_ylabel(r"lag-1 autocorrelation $\rho_1(s)$")
    ax.set_title("Data $\\rho_1(s)$ vs null band (grey = 2.5–97.5% of surrogates; "
                 "green=kept, red=rejected)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_mean_residuals(meanres: MeanResidualResult, path: str):
    if meanres is None or not meanres.groups:
        return
    keys = list(meanres.groups.keys())
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    cmap = plt.get_cmap("tab10")
    for c, key in enumerate(keys):
        g = meanres.groups[key]
        ci = meanres.coherence_index.get(key, np.nan)
        lbl = f"{key[0]}, {key[1]:g} pA (n={g['n']}, coh={ci:.1f})"
        ax.plot(g["u_ms"], g["mean"], color=cmap(c % 10), lw=1.4, label=lbl)
        if np.isfinite(g["sem"]).any():
            ax.fill_between(g["u_ms"], g["mean"] - g["sem"], g["mean"] + g["sem"],
                            color=cmap(c % 10), alpha=0.2)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xlabel("time from fit-window start (ms)")
    ax.set_ylabel("mean residual (mV)")
    ax.set_title("Mean residual per (polarity, amplitude band)  "
                 "[coherent curve => deterministic misfit]")
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_traces_v2(cell: CellQCResultV2, path: str, max_each: int = 24):
    qc = [r for r in cell.pulse_results if r.qced and r._d_mV is not None]
    acc = [r for r in qc if r.passed][:max_each]
    rej = [r for r in qc if not r.passed][:max_each]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, group, title, col in (
        (axes[0], acc, f"Accepted (n={sum(r.passed for r in qc)})", "tab:green"),
        (axes[1], rej, f"Rejected (n={sum(not r.passed for r in qc)})", "tab:red")):
        for r in group:
            ax.plot(r._u_ms, r._d_mV, color=col, alpha=0.35, lw=0.7)
            ax.plot(r._u_ms, r._fit_mV, color="k", alpha=0.5, lw=0.8, ls="--")
        ax.axhline(0, color="0.6", lw=0.6); ax.set_title(title)
        ax.set_xlabel("time from window start (ms)")
    axes[0].set_ylabel("baseline-subtracted deflection (mV)")
    fig.suptitle(f"Trace QC v2 — specimen {cell.specimen_id}  (null='{cell.null_generator}', "
                 f"bank={[f'{s:g}' for s in cell.scoring_bank_ms]} ms)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# =============================================================================
# I/O layer (downstream)
# =============================================================================
def _fmt(x) -> str:
    if isinstance(x, float):
        return "" if not np.isfinite(x) else f"{x:.6g}"
    return str(x)


def save_cell_qc_outputs_v2(cell: CellQCResultV2, output_dir: str, make_plots: bool, cfg: QCConfigV2):
    os.makedirs(output_dir, exist_ok=True)
    scales = list(cell.scoring_bank_ms)

    # --- per-pulse table (CSV) ---
    base_cols = ["pulse_index", "polarity", "peak_pA", "amp_band_pA", "qced", "passed",
                 "reason", "p_combined", "p_min", "winning_scale_ms",
                 "fit_A_mV", "fit_tau_ms", "fit_c_mV", "spike_ratio"]
    pcols = [f"p_s_{s:g}ms" for s in scales]
    rcols = [f"rho1_{s:g}ms" for s in scales]
    header = base_cols + pcols + rcols
    lines = [",".join(header)]
    for r in cell.pulse_results:
        d = asdict(r)
        row = [_fmt(d.get(c)) for c in base_cols]
        row += [_fmt(r.p_by_scale.get(float(s), np.nan)) for s in scales]
        row += [_fmt(r.rho1_by_scale.get(float(s), np.nan)) for s in scales]
        lines.append(",".join(row))
    with open(os.path.join(output_dir, "trace_qc_results.csv"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # --- cell-level summary (JSON) ---
    summary = {
        "specimen_id": str(cell.specimen_id), "n_pulses": cell.n_pulses,
        "n_qced": cell.n_qced, "n_passed_total": cell.n_passed_total,
        "n_viable_training": cell.n_viable_training, "cell_rejected": cell.cell_rejected,
        "cell_reason": cell.cell_reason, "null_generator": cell.null_generator,
        "scoring_bank_ms": scales, "per_scale_ar": cell.per_scale_ar,
        "selfcal": (None if cell.selfcal is None else {
            k: v for k, v in asdict(cell.selfcal).items()
            if not k.startswith("_")}),
        "mean_residual_coherence_index": (
            None if cell.mean_residual is None else
            {f"{k[0]}_{k[1]:g}pA": v for k, v in cell.mean_residual.coherence_index.items()}),
    }
    with open(os.path.join(output_dir, "trace_qc_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # --- self-calibration sweep table (CSV) ---
    if cell.selfcal is not None and cell.selfcal.m_ms_grid:
        sc = cell.selfcal
        sl = ["M_ms,ks_dist_max,ks_p_pooled,reject_rate_at_alpha,n_pvalues"]
        dmaxlist = sc.ks_dist_max if sc.ks_dist_max else [np.nan] * len(sc.m_ms_grid)
        for m, dm, k, rr, n in zip(sc.m_ms_grid, dmaxlist, sc.ks_p_pooled,
                                   sc.reject_rate_at_alpha, sc.n_pvalues):
            sl.append(f"{m:g},{_fmt(float(dm))},{_fmt(float(k))},{_fmt(float(rr))},{n}")
        with open(os.path.join(output_dir, "trace_qc_selfcal.csv"), "w") as f:
            f.write("\n".join(sl) + "\n")

    # --- mean-residual curves (long-format CSV) ---
    if cell.mean_residual is not None and cell.mean_residual.groups:
        ml = ["polarity,amp_band_pA,u_ms,mean_mV,sem_mV,n"]
        for (pol, amp), g in cell.mean_residual.groups.items():
            for u, mn, se in zip(g["u_ms"], g["mean"], g["sem"]):
                ml.append(f"{pol},{amp:g},{u:.6g},{mn:.6g},{_fmt(float(se))},{g['n']}")
        with open(os.path.join(output_dir, "trace_qc_meanresid.csv"), "w") as f:
            f.write("\n".join(ml) + "\n")

    # --- figures ---
    if make_plots:
        plot_traces_v2(cell, os.path.join(output_dir, "trace_qc_traces.png"))
        plot_rho_bands(cell, os.path.join(output_dir, "trace_qc_rho_bands.png"))
        if cell.selfcal is not None:
            plot_selfcal_histogram(cell.selfcal, os.path.join(output_dir, "trace_qc_selfcal_hist.png"), cfg.alpha)
            plot_selfcal_mms_sweep(cell.selfcal, os.path.join(output_dir, "trace_qc_selfcal_mms.png"),
                                   cfg.selfcal_flat_ks_dist)
        if cell.mean_residual is not None:
            plot_mean_residuals(cell.mean_residual, os.path.join(output_dir, "trace_qc_meanresid.png"))


# =============================================================================
# Smoke test (III.E)
# =============================================================================
# Two-component membrane-noise model used by the smoke test: a dominant fast
# white component (averaged DOWN by decimation) plus a small slow AR(1) component
# (CONCENTRATED by decimation).  The slow timescale (~5 ms) is short enough that
# the bounded single-exponential fit cannot absorb it over the test window, yet
# long enough to survive decimation -> the full-rate AR(15) fit is dominated by
# the fast part and its decimated surrogate is ~ white (misses the slow colour ->
# over-rejects), while the per-scale null fit in the decimated domain captures it.
_SMOKE_FAST_SD = 0.06          # mV, dominant fast (white) component
_SMOKE_SLOW_SD = 0.015         # mV, small slow (coloured) component
_SMOKE_TAU_SLOW_MS = 5.0       # ms, slow-component correlation time


def _two_component_noise(rng, n: int, fs_khz: float, fast_sd=_SMOKE_FAST_SD,
                         slow_sd=_SMOKE_SLOW_SD, tau_slow_ms=_SMOKE_TAU_SLOW_MS) -> np.ndarray:
    """Fast white + slow AR(1) membrane-noise surrogate (see note above)."""
    from scipy.signal import lfilter
    fast = rng.normal(0.0, fast_sd, size=n)
    phi = float(np.exp(-(1.0 / fs_khz) / tau_slow_ms))                 # AR(1) pole, full rate
    innov = slow_sd * np.sqrt(max(1e-12, 1.0 - phi ** 2))             # innovation sd for target var
    e = rng.normal(0.0, innov, size=n + 2000)
    slow = lfilter([1.0], [1.0, -phi], e)[2000:]
    return fast + slow


def _synth_baseline_epoch(rng, fs_khz=50.0, dur_ms=900.0):
    """One long signal-free two-component rest epoch (the III.B baseline source)."""
    return _two_component_noise(rng, int(round(dur_ms * fs_khz)), fs_khz)


def _synth_clean_pulse(rng, tau_m_ms=20.0, fs_khz=50.0, pre_ms=10.0, post_ms=200.0,
                       stim_dur_ms=0.5, amp_mV=-1.2):
    """A clean pulse: single-exponential passive transient + two-component noise."""
    dt = 1.0 / fs_khz
    n = int(round((pre_ms + post_ms) / dt))
    t_s = np.arange(n) * dt / 1e3
    onset = int(round(pre_ms / dt)); offset = onset + int(round(stim_dur_ms / dt))
    i_pA = np.zeros(n); i_pA[onset:offset] = -200.0
    v = np.zeros(n)
    u = np.arange(n - offset) * dt
    v[offset:] = amp_mV * np.exp(-u / tau_m_ms)
    v = v + _two_component_noise(rng, n, fs_khz) - 65.0
    return {"t": t_s, "v": v, "i": i_pA, "polarity": "hyp", "peak_pA": -200.0,
            "stim_duration_s": stim_dur_ms / 1e3, "sampling_rate_Hz": fs_khz * 1e3}


def _synth_misfit_pulse(rng, tau_m_ms=20.0, fs_khz=50.0, pre_ms=10.0, post_ms=200.0,
                        stim_dur_ms=0.5, A_fast=-1.4, tau_fast=6.0, A_slow=-0.6, tau_slow=55.0):
    """A pulse whose TRUE relaxation is a SUM of two exponentials but which the QC
    fits with a single bounded exponential -> deterministic misfit in the residual.
    Noise is the same two-component membrane model as the clean/baseline epochs.
    """
    dt = 1.0 / fs_khz
    n = int(round((pre_ms + post_ms) / dt))
    t_s = np.arange(n) * dt / 1e3
    onset = int(round(pre_ms / dt)); offset = onset + int(round(stim_dur_ms / dt))
    i_pA = np.zeros(n); i_pA[onset:offset] = -200.0
    v = np.zeros(n)
    u = np.arange(n - offset) * dt
    v[offset:] = A_fast * np.exp(-u / tau_fast) + A_slow * np.exp(-u / tau_slow)
    v = v + _two_component_noise(rng, n, fs_khz) - 65.0
    return {"t": t_s, "v": v, "i": i_pA, "polarity": "hyp", "peak_pA": -200.0,
            "stim_duration_s": stim_dur_ms / 1e3, "sampling_rate_Hz": fs_khz * 1e3}


def run_smoke_test_v2(outdir: str = "smoke_out_v2", seed: int = 1) -> dict:
    """Self-contained correctness check for the per-scale self-calibrated null.

    (i)   calibration : per-scale null gives ~ Uniform p-values on pure AR noise;
          per-pulse Bonferroni false-positive rate on clean coloured-noise pulses ~ alpha.
    (ii)  power       : it rejects an injected double-exponential misfit, and the mean
          residual is coherent (coherence index >> 1).
    (iii) bug repro   : the OLD full-rate AR(15) null over-rejects the SAME clean pulses.
    """
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(seed)
    tau_m = 20.0
    fs = 50.0
    cfg = QCConfigV2(n_surrogates=400, alpha=0.05, seed=seed, selfcal_n_repeats=40,
                     min_viable_training_pulses=10)

    # shared long baseline epochs (III.B source), pooled "across sweeps"
    base = [_synth_baseline_epoch(rng, fs_khz=fs, dur_ms=900.0) for _ in range(3)]

    # ---- (i) calibration on CLEAN pulses (single-exp + two-component noise) ----
    n_clean = 90
    clean = [_synth_clean_pulse(rng, tau_m_ms=tau_m, fs_khz=fs) for _ in range(n_clean)]
    cell_clean = run_trace_qc_for_cell_v2(clean, tau_m, baseline_segments=base,
                                          specimen_id="SMOKE_clean", cfg=cfg,
                                          output_dir=outdir, make_plots=True)
    fpr = 1.0 - cell_clean.n_passed_total / n_clean
    sc = cell_clean.selfcal
    selfcal_ksp = sc.chosen_ks_p if sc is not None else np.nan
    selfcal_dmax = sc.chosen_ks_dist_max if sc is not None else np.nan

    # ---- (ii) power on MISFIT pulses (double-exp truth, single-exp fit) ----
    n_mis = 50
    mis = [_synth_misfit_pulse(rng, tau_m_ms=tau_m, fs_khz=fs) for _ in range(n_mis)]
    cell_mis = run_trace_qc_for_cell_v2(mis, tau_m, baseline_segments=base,
                                        specimen_id="SMOKE_misfit", cfg=cfg,
                                        output_dir=None, make_plots=False)
    power = 1.0 - cell_mis.n_passed_total / n_mis
    coh = (max((v for v in cell_mis.mean_residual.coherence_index.values()
                if np.isfinite(v)), default=np.nan)
           if cell_mis.mean_residual else np.nan)

    # ---- (iii) bug reproduction: OLD full-rate AR(15) over-rejects clean ----
    cfg_old = QCConfigV2(n_surrogates=400, alpha=0.05, seed=seed, null_generator="ar_fullrate",
                         ar_max_order=15, decim_scales_ms=(1.0, 2.0, 5.0, 10.0),
                         min_viable_training_pulses=10)
    cell_old = run_trace_qc_for_cell_v2(clean, tau_m, baseline_segments=base,
                                        specimen_id="SMOKE_clean_OLD", cfg=cfg_old,
                                        output_dir=None, make_plots=False)
    fpr_old = 1.0 - cell_old.n_passed_total / n_clean

    metrics = dict(
        selfcal_status=(sc.status if sc else "n/a"), chosen_M_ms=(sc.chosen_M_ms if sc else np.nan),
        selfcal_ks_p=selfcal_ksp, selfcal_ks_dist_max=selfcal_dmax,
        flat_scales_ms=(sc.flat_scales_ms if sc else []),
        admissible_scales=(sc.admissible_scales_ms if sc else []),
        fpr_new=fpr, fpr_old_fullrate=fpr_old, power_misfit=power, misfit_coherence=coh)

    print("\n==============  TRACE-QC v2  SMOKE TEST  ==============")
    print(f"  self-cal status / M_ms*        : {metrics['selfcal_status']} / "
          f"{metrics['chosen_M_ms']} ms")
    print(f"  flatness max_s D_n (on scales {metrics['flat_scales_ms']} ms) : "
          f"{selfcal_dmax:.3f}   (target <= {cfg.selfcal_flat_ks_dist})")
    print(f"  admissible scoring bank (ms)   : {metrics['admissible_scales']}")
    print(f"  (i)   calibration  FPR new     : {fpr:6.3f}   (target ~ alpha = {cfg.alpha})")
    print(f"  (iii) bug repro    FPR old AR15: {fpr_old:6.3f}   (target HIGH >> alpha)")
    print(f"  (ii)  power        misfit rej  : {power:6.3f}   (target high)")
    print(f"  (ii)  misfit mean-resid coherence index : {coh:6.2f}   (target >> 1)")
    print(f"  figures        : {outdir}/trace_qc_selfcal_hist.png, _selfcal_mms.png, "
          f"_rho_bands.png, _meanresid.png, _traces.png")

    # ---- assertions (loose, demonstrational) ----
    assert fpr <= 0.15, f"new null FPR {fpr:.3f} too high — per-scale null miscalibrated"
    assert power >= 0.60, f"power {power:.3f} too low — misfit not detected"
    assert np.isfinite(coh) and coh >= 3.0, f"mean-residual coherence {coh} too low — misfit not coherent"
    assert fpr_old >= 0.30, f"old full-rate FPR {fpr_old:.3f} not high — bug not reproduced"
    assert fpr_old >= fpr + 0.15, "old null is not materially worse than new — fix not shown causal"
    assert sc is not None and sc.status in ("flat", "flat_relative"), \
        f"self-calibration failed (status={sc.status if sc else None})"
    assert np.isfinite(selfcal_dmax) and selfcal_dmax <= float(np.nanmin(sc.ks_dist_max)) + cfg.selfcal_flat_slack + 1e-9, \
        "chosen M_ms is not within slack of the best achievable calibration"
    print("  ALL ASSERTIONS PASSED ✔")
    print("======================================================\n")
    return metrics


# =============================================================================
# CLI
# =============================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Trace QC v2 (per-scale self-calibrated null).")
    ap.add_argument("--smoke-test", action="store_true", help="run the self-contained smoke test")
    ap.add_argument("--outdir", default="smoke_out_v2")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    if args.smoke_test:
        run_smoke_test_v2(args.outdir, seed=args.seed)
    else:
        ap.print_help()
