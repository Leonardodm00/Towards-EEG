"""
trace_qc.py
===========

Response-shape quality control for Square-Subthreshold (SS) current-pulse traces
used in passive (C_m, R_m) fitting of human cortical pyramidal neurons.

PURPOSE
-------
The archive builder already screens pulses on the *command current* (amplitude /
duration tolerance).  It does NOT screen the *voltage response*.  This module adds
the missing screen: it flags pulses whose post-pulse decay is not a clean (single-
exponential) passive transient, i.e. pulses contaminated by spontaneous synaptic /
epileptiform humps or sharp transients.

STATISTICAL DESIGN (agreed)
---------------------------
For one pulse p, over a delayed post-pulse window  W = [t_off + delay, t_off + delay + L]:

    1. Baseline-subtract:  d_p(t_n) = v_p(t_n) - B_p,  B_p = mean of v_p over the
       signal-free pre-pulse window W_pre.
    2. Bounded single-exponential fit:  mu_hat(u) = A * exp(-u / tau) + c,
       u = t - (start of W),  with tau CONSTRAINED to [tau_lo, tau_hi] around the
       dataset membrane time constant tau_m (NOT fixed: a bounded tau neither
       manufactures structure from a wrong-tau bias nor lets the fit absorb a slow
       bump).  A and c are free.
    3. Residual:  r_p(t_n) = d_p(t_n) - mu_hat(u_n).
    4. Structure statistic over a bank of decimation timescales S = {s_1,...,s_K}:
       at each s, decimate r_p (anti-aliased) to interval s and compute a per-scale
       structure measure g_s (default: lag-1 autocorrelation of the decimated
       residual).  Combine:  T*(r) = max_{s in S} g_s.
    5. Monte-Carlo p-value against a NULL that bakes in the membrane noise colour:
            p = (1 + #{ T*(eps_b) >= T*(r_p) }) / (1 + B)
       where {eps_b}_{b=1..B} are surrogate residuals.  Two null generators:
         - 'ar'    : AR(p) fit to the cell's POOLED signal-free baselines, B paths
                     simulated, decimated through the IDENTICAL pipeline.  (default)
         - 'block' : moving-block bootstrap of the residual itself (block length must
                     satisfy  noise_corr << L_block << bump_width).
       Because the surrogates pass through the same decimation, the reference
       distribution lives at the same (small) decimated sample count as the test
       statistic -> exact finite-sample size at alpha, no chi^2 asymptotics.
    6. Flag pulse as PATHOLOGICAL if p < alpha (primary test).  An optional cheap
       auxiliary gate flags sharp single-sample transients that decimation would
       smooth over (max|r| / robust_amp > spike_rel_thresh).

CELL-LEVEL RULE
---------------
After per-pulse QC, if the number of surviving training-polarity pulses is below
`min_viable_training_pulses`, the whole cell is rejected.

ARCHITECTURE (separation of concerns)
-------------------------------------
fit layer       : fit_bounded_single_exp
statistic layer : decimate_rows, lag1_autocorr_rows, ljungbox_rows, structure_over_scales
null layer      : fit_ar_pooled, simulate_ar, block_bootstrap
test layer      : monte_carlo_pvalue
orchestration   : qc_single_pulse, run_trace_qc_for_cell      (no plotting / no I/O inside scoring)
plotting layer  : plot_accepted_rejected, plot_null_distribution
I/O layer       : save_cell_qc_outputs
smoke test      : run_smoke_test  (also: `python trace_qc.py --smoke-test`)

Every scoring function is pure (numpy in -> numbers out); plotting and disk I/O are
strictly downstream so thresholds, statistics, or the null generator can be swapped
without touching the rest of the pipeline.

Dependencies: numpy, scipy, matplotlib only (no statsmodels) so it is HPC-portable.
"""
from __future__ import annotations

import os
import json
import argparse
import warnings
from dataclasses import dataclass, field, asdict
from typing import Callable, Literal, Sequence

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import resample_poly, lfilter
from scipy.linalg import solve_toeplitz

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class QCConfig:
    # --- windowing (all in ms) ---
    fit_delay_ms: float = 4.0          # delayed-window start, after pulse OFFSET (lets fast cable modes decay)
    fit_window_ms: float = 100.0       # length of fit window after the delay (matches the loss window)
    test_window_ms: float | None = None  # structure-test window length; None -> = fit_window_ms (set larger to catch late bumps)
    baseline_guard_ms: float = 1.0     # discard this much pre-pulse baseline immediately before onset
    min_baseline_ms: float = 3.0       # minimum usable pre-pulse baseline to trust the noise model

    # --- bounded single-exponential fit ---
    tau_lo_factor: float = 0.5         # tau in [tau_lo_factor, tau_hi_factor] * tau_m
    tau_hi_factor: float = 2.0

    # --- decimation-timescale bank (ms) ---
    decim_scales_ms: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0)

    # --- structure statistic ---
    statistic: Literal["lag1", "ljungbox"] = "lag1"
    ljungbox_lags: int = 5

    # --- noise null ---
    null_generator: Literal["ar", "block"] = "ar"
    ar_max_order: int = 15
    block_len_ms: float = 4.0
    n_surrogates: int = 500

    # --- decision ---
    alpha: float = 0.05

    # --- auxiliary sharp-transient gate ---
    use_spike_gate: bool = True
    spike_rel_thresh: float = 6.0      # reject if max|r| / robust_amp exceeds this

    # --- cell-level rule ---
    min_viable_training_pulses: int = 10

    # --- which polarities to QC, and which is "training" ---
    qc_polarities: tuple[str, ...] = ("hyp",)   # dep pulses may deviate for active-current reasons; QC hyp by default
    fit_target: str = "hyp"

    # --- reproducibility ---
    seed: int = 0


# =============================================================================
# Result containers
# =============================================================================
@dataclass
class PulseQCResult:
    pulse_index: int
    polarity: str
    peak_pA: float
    qced: bool                       # was this pulse subjected to the shape test?
    passed: bool                     # True = kept
    reason: str                      # 'ok' | 'pathological' | 'spike' | 'fit_failed' | 'short_window' | 'not_qced'
    p_value: float = np.nan
    T_obs: float = np.nan
    argmax_scale_ms: float = np.nan
    fit_A_mV: float = np.nan
    fit_tau_ms: float = np.nan
    fit_c_mV: float = np.nan
    spike_ratio: float = np.nan
    # heavy diagnostics for plotting (not serialised to CSV)
    _u_ms: np.ndarray | None = field(default=None, repr=False)
    _d_mV: np.ndarray | None = field(default=None, repr=False)
    _fit_mV: np.ndarray | None = field(default=None, repr=False)
    _resid_mV: np.ndarray | None = field(default=None, repr=False)
    _T_null: np.ndarray | None = field(default=None, repr=False)


@dataclass
class CellQCResult:
    specimen_id: object
    n_pulses: int
    n_qced: int
    n_passed_total: int
    n_viable_training: int
    cell_rejected: bool
    cell_reason: str
    ar_order: int
    ar_sigma2: float
    config: dict
    pulse_results: list[PulseQCResult]
    kept_pulses: list[dict]          # the cleaned pool (empty if the cell is rejected)


# =============================================================================
# Fit layer (pure)
# =============================================================================
def _exp_decay(u: np.ndarray, A: float, tau: float, c: float) -> np.ndarray:
    return A * np.exp(-u / tau) + c


def fit_bounded_single_exp(u_ms: np.ndarray, y_mV: np.ndarray, tau_m_ms: float,
                           tau_lo_factor: float, tau_hi_factor: float) -> dict:
    """Bounded single-exponential fit  y ~ A*exp(-u/tau)+c  with tau constrained.

    Parameters
    ----------
    u_ms : (N,) time within the fit window, in ms, starting at 0.
    y_mV : (N,) baseline-subtracted deflection, in mV.
    tau_m_ms : dataset membrane time constant (window centre for tau), ms.

    Returns
    -------
    dict with keys A, tau, c, ok, resid.
    """
    tau_lo = float(tau_m_ms) * tau_lo_factor
    tau_hi = float(tau_m_ms) * tau_hi_factor
    tail = max(3, y_mV.size // 20)
    c0 = float(np.median(y_mV[-tail:]))
    A0 = float(y_mV[0] - c0)
    tau0 = float(np.clip(tau_m_ms, tau_lo, tau_hi))
    p0 = [A0 if A0 != 0 else -1.0, tau0, c0]
    bounds = ([-np.inf, tau_lo, -np.inf], [np.inf, tau_hi, np.inf])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(_exp_decay, u_ms, y_mV, p0=p0, bounds=bounds, maxfev=20000)
        A, tau, c = (float(popt[0]), float(popt[1]), float(popt[2]))
        resid = y_mV - _exp_decay(u_ms, A, tau, c)
        return {"A": A, "tau": tau, "c": c, "ok": True, "resid": resid}
    except Exception:
        return {"A": np.nan, "tau": np.nan, "c": np.nan, "ok": False, "resid": None}


# =============================================================================
# Statistic layer (pure) — all operate row-wise on (B, n) so the observed trace
# (B=1) and the surrogate ensemble share one code path.
# =============================================================================
def decimate_rows(X: np.ndarray, q: int) -> np.ndarray:
    """Anti-aliased decimation by integer factor q along axis=1. q<=1 -> identity."""
    if q <= 1:
        return X
    return resample_poly(X, up=1, down=q, axis=1)


def lag1_autocorr_rows(X: np.ndarray) -> np.ndarray:
    """Lag-1 autocorrelation of each row (mean-subtracted). Returns (B,)."""
    Xc = X - X.mean(axis=1, keepdims=True)
    num = np.einsum("ij,ij->i", Xc[:, :-1], Xc[:, 1:])
    den = np.einsum("ij,ij->i", Xc, Xc)
    return np.where(den > 0, num / den, np.nan)


def ljungbox_rows(X: np.ndarray, n_lags: int) -> np.ndarray:
    """Ljung-Box Q statistic over lags 1..n_lags for each row. Returns (B,)."""
    Xc = X - X.mean(axis=1, keepdims=True)
    N = Xc.shape[1]
    den = np.einsum("ij,ij->i", Xc, Xc)
    Q = np.zeros(Xc.shape[0])
    n_lags = min(n_lags, N - 2)
    for ell in range(1, n_lags + 1):
        num = np.einsum("ij,ij->i", Xc[:, :-ell], Xc[:, ell:])
        rho = np.where(den > 0, num / den, 0.0)
        Q += rho ** 2 / (N - ell)
    return N * (N + 2) * np.where(den > 0, Q, np.nan)


def structure_over_scales(R: np.ndarray, fs_khz: float, scales_ms: Sequence[float],
                          statistic: str, ljb_lags: int):
    """Per-scale structure measure and its max over the decimation bank.

    R : (B, n_full) full-rate residual rows.
    Returns (T_max:(B,), G:(B,K), argmax_idx:(B,)).
    """
    cols = []
    for s in scales_ms:
        q = max(1, int(round(s * fs_khz)))
        Rd = decimate_rows(R, q)
        if Rd.shape[1] < 4:
            cols.append(np.full(R.shape[0], np.nan))
            continue
        if statistic == "lag1":
            cols.append(lag1_autocorr_rows(Rd))
        elif statistic == "ljungbox":
            cols.append(ljungbox_rows(Rd, ljb_lags))
        else:
            raise ValueError(f"unknown statistic {statistic!r}")
    G = np.column_stack(cols)                              # (B, K)
    Gsafe = np.where(np.isnan(G), -np.inf, G)
    T_max = np.nanmax(G, axis=1)
    argmax_idx = np.argmax(Gsafe, axis=1)
    return T_max, G, argmax_idx


# =============================================================================
# Null layer (pure)
# =============================================================================
def fit_ar_pooled(segments: Sequence[np.ndarray], max_order: int):
    """Yule-Walker AR fit on POOLED baseline segments (lags never cross segment
    boundaries), order chosen by AIC.

    Returns (phi:(p,), sigma2:float, order:int).
    """
    maxlag = max_order
    acc = np.zeros(maxlag + 1)
    ntot = 0
    for seg in segments:
        s = np.asarray(seg, float)
        s = s - s.mean()
        n = s.size
        if n < 2:
            continue
        kmax = min(maxlag, n - 1)
        for k in range(kmax + 1):
            acc[k] += np.dot(s[: n - k], s[k:])
        ntot += n
    if ntot == 0 or acc[0] <= 0:
        # degenerate: fall back to white noise
        return np.zeros(0), float(acc[0] / max(ntot, 1)) or 1.0, 0
    gamma = acc / ntot                                    # biased -> PSD Toeplitz

    best = None
    for p in range(1, max_order + 1):
        try:
            phi = solve_toeplitz(gamma[:p], gamma[1 : p + 1])
        except Exception:
            continue
        sigma2 = float(gamma[0] - phi @ gamma[1 : p + 1])
        if sigma2 <= 0:
            continue
        aic = ntot * np.log(sigma2) + 2 * p
        if best is None or aic < best[0]:
            best = (aic, p, phi, sigma2)
    if best is None:
        return np.zeros(0), float(gamma[0]), 0
    _, order, phi, sigma2 = best
    return phi, sigma2, order


def simulate_ar(phi: np.ndarray, sigma2: float, n: int, B: int,
                rng: np.random.Generator, burnin: int = 200) -> np.ndarray:
    """Simulate B AR paths of length n from coefficients phi and innovation var sigma2."""
    sd = np.sqrt(max(sigma2, 1e-30))
    eta = rng.normal(0.0, sd, size=(B, n + burnin))
    if phi.size == 0:
        x = eta
    else:
        a = np.r_[1.0, -np.asarray(phi, float)]
        x = lfilter([1.0], a, eta, axis=1)
    return x[:, burnin:]


def block_bootstrap(resid: np.ndarray, block_len: int, B: int,
                    rng: np.random.Generator) -> np.ndarray:
    """Circular moving-block bootstrap (with replacement) of a residual series."""
    n = resid.size
    L = max(1, int(block_len))
    n_blocks = int(np.ceil(n / L))
    starts = rng.integers(0, n, size=(B, n_blocks))                     # (B, n_blocks)
    idx = (starts[:, :, None] + np.arange(L)[None, None, :]) % n         # (B, n_blocks, L)
    out = resid[idx].reshape(B, n_blocks * L)[:, :n]
    return out


# =============================================================================
# Test layer (pure)
# =============================================================================
def monte_carlo_pvalue(resid_full: np.ndarray, null_paths: np.ndarray,
                       fs_khz: float, cfg: QCConfig):
    """Add-one Monte-Carlo p-value for 'residual is more structured than the null'.

    Returns (p, T_obs, T_null:(B,), argmax_scale_ms).
    """
    T_obs, _, arg_obs = structure_over_scales(
        resid_full[None, :], fs_khz, cfg.decim_scales_ms, cfg.statistic, cfg.ljungbox_lags)
    T_null, _, _ = structure_over_scales(
        null_paths, fs_khz, cfg.decim_scales_ms, cfg.statistic, cfg.ljungbox_lags)
    t_obs = float(T_obs[0])
    T_null = T_null[np.isfinite(T_null)]
    B = T_null.size
    p = (1 + int(np.sum(T_null >= t_obs))) / (1 + B)
    argmax_scale = float(cfg.decim_scales_ms[int(arg_obs[0])])
    return p, t_obs, T_null, argmax_scale


# =============================================================================
# Orchestration (per pulse) — pure scoring, no plotting / no I/O
# =============================================================================
def _detect_onset_offset(t_s: np.ndarray, i_pA: np.ndarray, stim_duration_s: float):
    """Detect pulse onset / offset sample indices from the command current."""
    base = np.median(i_pA[: max(5, i_pA.size // 50)])
    dev = np.abs(i_pA - base)
    thr = 0.3 * dev.max() if dev.max() > 0 else np.inf
    above = np.flatnonzero(dev > thr)
    if above.size == 0:
        return None, None
    onset = int(above[0])
    fs = 1.0 / np.median(np.diff(t_s))
    offset = onset + int(round(stim_duration_s * fs))
    return onset, offset


def _build_null_paths(resid_full: np.ndarray, ar_model, cfg: QCConfig,
                      fs_khz: float, rng: np.random.Generator) -> np.ndarray:
    """Build the surrogate residual ensemble per the configured null generator."""
    n = resid_full.size
    if cfg.null_generator == "ar":
        phi, sigma2, _ = ar_model
        return simulate_ar(phi, sigma2, n, cfg.n_surrogates, rng)
    elif cfg.null_generator == "block":
        L = max(1, int(round(cfg.block_len_ms * fs_khz)))
        return block_bootstrap(resid_full, L, cfg.n_surrogates, rng)
    raise ValueError(f"unknown null_generator {cfg.null_generator!r}")


def qc_single_pulse(pulse: dict, idx: int, tau_m_ms: float, cfg: QCConfig,
                    ar_model, rng: np.random.Generator) -> PulseQCResult:
    """Score one pulse. `ar_model` = (phi, sigma2, order) from the cell's pooled baselines."""
    t = np.asarray(pulse["t"], float)
    v = np.asarray(pulse["v"], float)
    i = np.asarray(pulse["i"], float)
    pol = pulse["polarity"]
    peak = float(pulse.get("peak_pA", np.nan))
    fs_khz = float(pulse.get("sampling_rate_Hz", 1.0 / np.median(np.diff(t)))) / 1e3

    res = PulseQCResult(idx, pol, peak, qced=True, passed=False, reason="ok")

    onset, offset = _detect_onset_offset(t, i, float(pulse["stim_duration_s"]))
    if onset is None:
        res.passed, res.reason = False, "fit_failed"
        return res

    # --- baseline (pre-pulse) ---
    guard = int(round(cfg.baseline_guard_ms * fs_khz))
    pre = v[: max(0, onset - guard)]
    if pre.size < int(round(cfg.min_baseline_ms * fs_khz)):
        res.passed, res.reason = False, "short_window"
        return res
    B_p = float(pre.mean())

    # --- fit window (delayed) ---
    test_len_ms = cfg.test_window_ms if cfg.test_window_ms is not None else cfg.fit_window_ms
    w0 = offset + int(round(cfg.fit_delay_ms * fs_khz))
    w1_fit = w0 + int(round(cfg.fit_window_ms * fs_khz))
    w1_test = w0 + int(round(test_len_ms * fs_khz))
    if w1_test > v.size or (w1_fit - w0) < 8:
        res.passed, res.reason = False, "short_window"
        return res

    u_fit = (t[w0:w1_fit] - t[w0]) * 1e3
    y_fit = v[w0:w1_fit] - B_p

    fit = fit_bounded_single_exp(u_fit, y_fit, tau_m_ms, cfg.tau_lo_factor, cfg.tau_hi_factor)
    if not fit["ok"]:
        res.passed, res.reason = False, "fit_failed"
        return res
    res.fit_A_mV, res.fit_tau_ms, res.fit_c_mV = fit["A"], fit["tau"], fit["c"]

    # residual over the (possibly longer) test window, using the fitted parameters
    u_test = (t[w0:w1_test] - t[w0]) * 1e3
    d_test = v[w0:w1_test] - B_p
    fit_test = _exp_decay(u_test, fit["A"], fit["tau"], fit["c"])
    resid = d_test - fit_test

    robust_amp = max(abs(fit["A"]), 1e-6)
    res._u_ms, res._d_mV, res._fit_mV, res._resid_mV = u_test, d_test, fit_test, resid

    # --- auxiliary sharp-transient gate (decimation would smooth these away) ---
    spike_ratio = float(np.max(np.abs(resid)) / robust_amp)
    res.spike_ratio = spike_ratio
    if cfg.use_spike_gate and spike_ratio > cfg.spike_rel_thresh:
        res.passed, res.reason = False, "spike"
        # still record the MC p-value below for diagnostics
    # --- primary Monte-Carlo structure test ---
    null_paths = _build_null_paths(resid, ar_model, cfg, fs_khz, rng)
    p, t_obs, T_null, argmax = monte_carlo_pvalue(resid, null_paths, fs_khz, cfg)
    res.p_value, res.T_obs, res.argmax_scale_ms, res._T_null = p, t_obs, argmax, T_null

    if res.reason == "spike":
        return res
    if p < cfg.alpha:
        res.passed, res.reason = False, "pathological"
    else:
        res.passed, res.reason = True, "ok"
    return res


# =============================================================================
# Orchestration (per cell)
# =============================================================================
def _extract_baseline(pulse: dict, cfg: QCConfig) -> np.ndarray | None:
    t = np.asarray(pulse["t"], float)
    v = np.asarray(pulse["v"], float)
    i = np.asarray(pulse["i"], float)
    fs_khz = float(pulse.get("sampling_rate_Hz", 1.0 / np.median(np.diff(t)))) / 1e3
    onset, _ = _detect_onset_offset(t, i, float(pulse["stim_duration_s"]))
    if onset is None:
        return None
    guard = int(round(cfg.baseline_guard_ms * fs_khz))
    pre = v[: max(0, onset - guard)]
    if pre.size < int(round(cfg.min_baseline_ms * fs_khz)):
        return None
    return pre - pre.mean()


def run_trace_qc_for_cell(pulses: Sequence[dict], tau_m_ms: float,
                          specimen_id: object = None, cfg: QCConfig | None = None,
                          output_dir: str | None = None, make_plots: bool = True) -> CellQCResult:
    """Full per-cell trace QC: fit per-cell noise model, score each pulse, apply the
    cell-level viability rule, optionally save tables + figures.

    Parameters
    ----------
    pulses : list of pulse dicts {t, v, i, polarity, peak_pA, stim_duration_s, sampling_rate_Hz}.
    tau_m_ms : dataset membrane time constant for this cell (ms).
    """
    cfg = cfg or QCConfig()
    rng = np.random.default_rng(cfg.seed)

    _validate_block_band(cfg)

    # --- 1. per-cell noise model from pooled baselines of the QC'd polarities ---
    segs = []
    for p in pulses:
        if p["polarity"] in cfg.qc_polarities:
            seg = _extract_baseline(p, cfg)
            if seg is not None:
                segs.append(seg)
    if cfg.null_generator == "ar" and len(segs) == 0:
        ar_model = (np.zeros(0), 1.0, 0)
    elif cfg.null_generator == "ar":
        ar_model = fit_ar_pooled(segs, cfg.ar_max_order)
    else:
        ar_model = (np.zeros(0), 1.0, 0)   # unused for block bootstrap

    # --- 2. per-pulse QC ---
    results: list[PulseQCResult] = []
    for k, p in enumerate(pulses):
        if p["polarity"] in cfg.qc_polarities:
            results.append(qc_single_pulse(p, k, tau_m_ms, cfg, ar_model, rng))
        else:
            results.append(PulseQCResult(k, p["polarity"], float(p.get("peak_pA", np.nan)),
                                         qced=False, passed=True, reason="not_qced"))

    # --- 3. cell-level viability rule (training-polarity survivors) ---
    n_viable_train = sum(r.passed for r, p in zip(results, pulses)
                         if p["polarity"] == cfg.fit_target)
    cell_rejected = n_viable_train < cfg.min_viable_training_pulses
    cell_reason = (f"only {n_viable_train} viable {cfg.fit_target} pulses "
                   f"(< {cfg.min_viable_training_pulses})") if cell_rejected else "ok"

    kept = [] if cell_rejected else [p for p, r in zip(pulses, results) if r.passed]

    cell = CellQCResult(
        specimen_id=specimen_id,
        n_pulses=len(pulses),
        n_qced=sum(r.qced for r in results),
        n_passed_total=sum(r.passed for r in results),
        n_viable_training=n_viable_train,
        cell_rejected=cell_rejected,
        cell_reason=cell_reason,
        ar_order=int(ar_model[2]),
        ar_sigma2=float(ar_model[1]),
        config=asdict(cfg),
        pulse_results=results,
        kept_pulses=kept,
    )

    # --- 4. I/O (downstream of scoring) ---
    if output_dir is not None:
        save_cell_qc_outputs(cell, output_dir, make_plots=make_plots, cfg=cfg)
    return cell


def _validate_block_band(cfg: QCConfig):
    if cfg.null_generator == "block":
        if cfg.block_len_ms >= max(cfg.decim_scales_ms):
            warnings.warn(
                f"block_len_ms={cfg.block_len_ms} >= max decimation scale "
                f"{max(cfg.decim_scales_ms)}: the block null retains structure at every "
                f"tested scale, so the test will have no power. Reduce block_len_ms.")
        if cfg.block_len_ms <= min(cfg.decim_scales_ms):
            warnings.warn(
                f"block_len_ms={cfg.block_len_ms} <= min decimation scale: blocks may be "
                f"too short to carry the membrane colour; the null can be anti-conservative.")


# =============================================================================
# Plotting layer (downstream; consumes results)
# =============================================================================
def plot_accepted_rejected(cell: CellQCResult, path: str, max_each: int = 24):
    """Overlay accepted vs rejected deflections with the fitted exponential."""
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
        ax.axhline(0, color="0.6", lw=0.6)
        ax.set_title(title)
        ax.set_xlabel("time from window start (ms)")
    axes[0].set_ylabel("baseline-subtracted deflection (mV)")
    fig.suptitle(f"Trace QC — specimen {cell.specimen_id}  "
                 f"(dashed = bounded single-exp fit; null='{cell.config['null_generator']}', "
                 f"alpha={cell.config['alpha']})")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_null_distribution(result: PulseQCResult, path: str, alpha: float):
    """Histogram of the surrogate T* with the observed T* and p-value."""
    if result._T_null is None:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.hist(result._T_null, bins=40, color="0.7", edgecolor="0.4",
            label=f"null T*  (B={result._T_null.size})")
    crit = np.quantile(result._T_null, 1 - alpha)
    ax.axvline(crit, color="tab:orange", ls=":", lw=1.6, label=f"{1-alpha:.0%} crit.")
    ax.axvline(result.T_obs, color="tab:red", lw=2.0,
               label=f"observed T*={result.T_obs:.3f}")
    ax.set_xlabel(f"structure statistic T* = max over scales of lag-1 autocorr "
                  f"(argmax @ {result.argmax_scale_ms:g} ms)")
    ax.set_ylabel("surrogate count")
    verdict = "REJECT" if not result.passed else "keep"
    ax.set_title(f"Pulse {result.pulse_index} ({result.polarity}) — "
                 f"p={result.p_value:.3f} → {verdict}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# =============================================================================
# I/O layer (downstream)
# =============================================================================
def save_cell_qc_outputs(cell: CellQCResult, output_dir: str, make_plots: bool, cfg: QCConfig):
    os.makedirs(output_dir, exist_ok=True)

    # per-pulse table (CSV, no heavy arrays)
    cols = ["pulse_index", "polarity", "peak_pA", "qced", "passed", "reason",
            "p_value", "T_obs", "argmax_scale_ms", "fit_A_mV", "fit_tau_ms",
            "fit_c_mV", "spike_ratio"]
    lines = [",".join(cols)]
    for r in cell.pulse_results:
        d = asdict(r)
        lines.append(",".join(_fmt(d[c]) for c in cols))
    with open(os.path.join(output_dir, "trace_qc_results.csv"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # cell-level summary (JSON)
    summary = {k: v for k, v in asdict(cell).items()
               if k not in ("pulse_results", "kept_pulses")}
    with open(os.path.join(output_dir, "trace_qc_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    if make_plots:
        plot_accepted_rejected(cell, os.path.join(output_dir, "trace_qc_traces.png"))
        # one representative null plot for a kept and a rejected pulse
        qc = [r for r in cell.pulse_results if r.qced and r._T_null is not None]
        kept = next((r for r in qc if r.passed), None)
        rej = next((r for r in qc if not r.passed), None)
        if kept is not None:
            plot_null_distribution(kept, os.path.join(output_dir, "trace_qc_null_kept.png"), cfg.alpha)
        if rej is not None:
            plot_null_distribution(rej, os.path.join(output_dir, "trace_qc_null_rejected.png"), cfg.alpha)


def _fmt(x) -> str:
    if isinstance(x, float):
        return "" if not np.isfinite(x) else f"{x:.6g}"
    return str(x)


# =============================================================================
# Smoke test
# =============================================================================
def _make_synthetic_pulse(kind: str, rng: np.random.Generator, *, tau_ms: float = 20.0,
                          fs_khz: float = 50.0, pre_ms: float = 10.0, post_ms: float = 200.0,
                          stim_dur_ms: float = 0.5, amp_mV: float = -1.2,
                          ar_phi=(0.85, -0.10), noise_sd: float = 0.05,
                          bump_h_frac: float = 0.35, bump_t0_ms: float = 70.0,
                          bump_sigma_ms: float = 14.0, spike_frac: float = 8.0) -> dict:
    """Build one synthetic hyperpolarising pulse dict.

    kind in {'clean','bump','spike'}.  The passive transient is a single exp of tau_ms;
    noise is AR(2)-coloured (membrane-like); 'bump' adds a smooth depolarising Gaussian
    in the post-pulse window; 'spike' adds a single sharp sample.
    """
    dt_ms = 1.0 / fs_khz
    n = int(round((pre_ms + post_ms) / dt_ms))
    t_s = np.arange(n) * dt_ms / 1e3
    onset = int(round(pre_ms / dt_ms))
    offset = onset + int(round(stim_dur_ms / dt_ms))

    # command current
    i_pA = np.zeros(n)
    i_pA[onset:offset] = -200.0

    # passive deflection: charge during pulse (negligible width) then exp decay
    v = np.zeros(n)
    u = (np.arange(n - offset)) * dt_ms
    v[offset:] = amp_mV * np.exp(-u / tau_ms)

    # AR(2) coloured noise
    a = np.r_[1.0, -np.asarray(ar_phi, float)]
    eta = rng.normal(0, noise_sd, size=n + 200)
    col = lfilter([1.0], a, eta)[200:]
    v = v + col

    if kind == "bump":
        tb = (np.arange(n) - offset) * dt_ms
        bump = (abs(amp_mV) * bump_h_frac) * np.exp(-(tb - bump_t0_ms) ** 2 / (2 * bump_sigma_ms ** 2))
        v = v + bump                     # depolarising (positive) hump
    elif kind == "spike":
        j = offset + int(round((bump_t0_ms) / dt_ms))
        v[j] += abs(amp_mV) * spike_frac

    v_rest = -65.0
    return {"t": t_s, "v": v + v_rest, "i": i_pA, "polarity": "hyp",
            "peak_pA": -200.0, "stim_duration_s": stim_dur_ms / 1e3,
            "sampling_rate_Hz": fs_khz * 1e3}


def run_smoke_test(outdir: str = "smoke_out", seed: int = 1) -> dict:
    """Self-contained correctness check: calibration (false-positive rate ~ alpha on
    clean coloured-noise traces) and power (rejection rate on bumped traces), plus
    figures. Returns a small dict of metrics.
    """
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(seed)
    tau_ms = 20.0
    cfg = QCConfig(n_surrogates=600, alpha=0.05, seed=seed)

    # ---- (1) calibration: many CLEAN pulses -> FPR should ~ alpha ----
    n_clean = 150
    clean = [_make_synthetic_pulse("clean", rng, tau_ms=tau_ms) for _ in range(n_clean)]
    cell_clean = run_trace_qc_for_cell(clean, tau_ms, specimen_id="SMOKE_clean",
                                       cfg=cfg, output_dir=None, make_plots=False)
    fpr = 1.0 - cell_clean.n_passed_total / n_clean

    # ---- (2) power: many BUMP pulses -> rejection rate should be high ----
    n_bump = 60
    bump = [_make_synthetic_pulse("bump", rng, tau_ms=tau_ms) for _ in range(n_bump)]
    cell_bump = run_trace_qc_for_cell(bump, tau_ms, specimen_id="SMOKE_bump",
                                      cfg=cfg, output_dir=None, make_plots=False)
    power = 1.0 - cell_bump.n_passed_total / n_bump

    # ---- (3) spike gate ----
    spikes = [_make_synthetic_pulse("spike", rng, tau_ms=tau_ms) for _ in range(20)]
    cell_spike = run_trace_qc_for_cell(spikes, tau_ms, specimen_id="SMOKE_spike",
                                       cfg=cfg, output_dir=None, make_plots=False)
    spike_catch = 1.0 - cell_spike.n_passed_total / len(spikes)

    # ---- (4) a mixed cell for the figures + cell-level rule ----
    mixed = ([_make_synthetic_pulse("clean", rng, tau_ms=tau_ms) for _ in range(16)] +
             [_make_synthetic_pulse("bump", rng, tau_ms=tau_ms) for _ in range(10)] +
             [_make_synthetic_pulse("spike", rng, tau_ms=tau_ms) for _ in range(2)])
    rng2 = np.random.default_rng(seed)  # fresh order
    cell_mixed = run_trace_qc_for_cell(mixed, tau_ms, specimen_id="SMOKE_mixed",
                                       cfg=cfg, output_dir=outdir, make_plots=True)

    # ---- (5) cell-level rejection rule: a cell with too few survivors ----
    few = [_make_synthetic_pulse("bump", rng, tau_ms=tau_ms) for _ in range(8)]
    cell_few = run_trace_qc_for_cell(few, tau_ms, specimen_id="SMOKE_few",
                                     cfg=cfg, output_dir=None, make_plots=False)

    metrics = dict(false_positive_rate=fpr, power_bump=power, spike_catch=spike_catch,
                   ar_order=cell_clean.ar_order, mixed_kept=cell_mixed.n_passed_total,
                   mixed_total=cell_mixed.n_pulses,
                   few_cell_rejected=cell_few.cell_rejected)

    print("\n================  TRACE-QC SMOKE TEST  ================")
    print(f"  AR(p) order fitted from pooled baselines : {metrics['ar_order']}")
    print(f"  Calibration  : false-positive rate on {n_clean} clean = {fpr:6.3f}   "
          f"(target ~ alpha = {cfg.alpha})")
    print(f"  Power        : rejection rate on {n_bump} bumped     = {power:6.3f}   (target high)")
    print(f"  Spike gate   : catch rate on {len(spikes)} spikes        = {spike_catch:6.3f}")
    print(f"  Mixed cell   : kept {cell_mixed.n_passed_total}/{cell_mixed.n_pulses}, "
          f"viable hyp = {cell_mixed.n_viable_training}, rejected = {cell_mixed.cell_rejected}")
    print(f"  Cell rule    : 8-survivor cell rejected   = {cell_few.cell_rejected}  "
          f"(expected True: < {cfg.min_viable_training_pulses})")
    print(f"  Figures      : {outdir}/trace_qc_traces.png, trace_qc_null_kept.png, "
          f"trace_qc_null_rejected.png")

    # ---- assertions (loose, demonstrational) ----
    assert fpr <= 0.13, f"false-positive rate {fpr} too high — null miscalibrated"
    assert power >= 0.70, f"power {power} too low — test missing the bumps"
    assert spike_catch >= 0.90, f"spike gate catch {spike_catch} too low"
    assert cell_few.cell_rejected is True, "cell-level <10 rule did not fire"
    print("  ALL ASSERTIONS PASSED ✔")
    print("======================================================\n")
    return metrics


# =============================================================================
# CLI
# =============================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Trace QC for SS passive-fit pulses.")
    ap.add_argument("--smoke-test", action="store_true", help="run the self-contained smoke test")
    ap.add_argument("--outdir", default="smoke_out")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    if args.smoke_test:
        run_smoke_test(args.outdir, seed=args.seed)
    else:
        ap.print_help()
