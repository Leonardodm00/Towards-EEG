# -*- coding: utf-8 -*-
"""
passive_long_step_training.py
=============================

Drop-in patch for `passive_fitting_hpc_fixed.py` that folds a long
subthreshold hyperpolarising step into the TRAINING set, so the fit is
directly constrained by the two quantities a 0.5 ms brief pulse cannot pin
down:

    tau_m = Rm * Cm        (membrane time constant)   -- from the step's
                            charging time course
    R_in  = dV_ss / I      (somatic input resistance)  -- from the step's
                            plateau level

It provides (a) a bundle-type-aware loss and (b) a training-set builder, both
decoupled from NEURON: they consume only the duck-typed `cell` API
(`set_passive`, `set_e_pas`, `simulate`) and `SweepBundle`-like objects (any
object exposing `t, v_mV, amplitude_pA, polarity, stim_onset_s,
stim_duration_s`).

Notation (units carried everywhere; cf. notation directive)
-----------------------------------------------------------
    Cm   [uF/cm^2]   Rm  [Ohm*cm^2]   Ra [Ohm*cm]
    tau_m[ms] = Rm * Cm * 1e-3        (Ohm*uF = 1e-6 s = 1e-3 ms)
    R_in [MOhm] = dV[mV] / I[pA] * 1e3
    The optimiser works in q = log(p); the loss converts p = exp(q) before
    calling NEURON, matching the existing pipeline's convention exactly.

The I_h problem (read before using on real data)
------------------------------------------------
A purely passive model's step response is monotonic to steady state. A real
cell with I_h sags: the deflection peaks (pre-sag) then relaxes to a smaller
steady level. Hence the passive "R_in" is ambiguous for high-I_h cells:

  * `r_in_target="peak"`  matches the model steady state to the experimental
    PRE-SAG PEAK deflection -> recovers the I_h-removed (passive) R_in. Use
    this when I_h will be added back as an explicit channel in the full model.
  * `r_in_target="steady"` matches the experimental SAGGED steady state ->
    recovers the apparent (Allen-style, sag-reduced) R_in. Use this when the
    network model stays purely passive.

`ls_train_window_ms_after_onset` bounds the step window. Keep it long enough
to resolve charging (>= ~5 * tau_m) but short enough to precede heavy sag
(150 ms is a reasonable default for tau_m in 10-40 ms). The window is applied
from step onset; for `r_in_target="peak"` the plateau target is taken as the
max |deflection| within the window rather than its tail mean.
"""
from __future__ import annotations

import itertools
import multiprocessing
import numbers
import pickle
import traceback
import weakref
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Brief vs long discriminator (matches the monolith's _BRIEF_PULSE_MAX_DURATION_S)
_BRIEF_PULSE_MAX_DURATION_S = 0.01

# ---------------------------------------------------------------------------
#  Integration time step
# ---------------------------------------------------------------------------
# These mirror the monolith's DEFAULT_DT_BRIEF_MS / DEFAULT_DT_LONG_MS. They are
# duplicated rather than imported because this module is deliberately decoupled
# from the monolith (it receives `mono` as an argument, never imports it), and
# they are resolved at CALL time so `integrate_long_step(..., dt_long_ms=...)`
# can retune them for a whole run.
#
# The historical `dt_ms = 0.1` on the long-step replay NEVER took effect:
# stdrun's setdt() rewrote it to 0.025 ms because the legacy plotting variable
# steps_per_ms defaults to 40. PassiveCell.simulate now enforces the requested
# dt, so these values are, for the first time, the values actually integrated
# at. Both default to 0.025 ms -- i.e. the behaviour every previous run really
# had -- so adopting this patch alone changes no numbers. Raising
# DEFAULT_DT_LONG_MS to 0.1 is a real 4x saving on the long steps and a real
# first-order accuracy change: validate it before using it.
DEFAULT_DT_BRIEF_MS = 0.025
DEFAULT_DT_LONG_MS = 0.025

# Set by integrate_long_step; the spec the patched loss falls back to
# when fit_one_cell does not pass one (Stage 3/4).
_INTEGRATE_SPEC = None


# ---------------------------------------------------------------------------
#  Bundle helpers (duck-typed; SweepBundle satisfies the attribute contract)
# ---------------------------------------------------------------------------
def _is_brief(bundle) -> bool:
    return float(bundle.stim_duration_s) < _BRIEF_PULSE_MAX_DURATION_S


def _simulate_brief(cell, b, v_rest_mV: float,
                    pre_pad_ms: float = 10.0, post_pad_ms: float = 100.0,
                    dt_ms: Optional[float] = None
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Replay a brief (Square-Subthreshold) pulse; t=0 at pulse onset.

    `dt_ms=None` resolves DEFAULT_DT_BRIEF_MS at call time."""
    dt_ms = float(DEFAULT_DT_BRIEF_MS if dt_ms is None else dt_ms)
    dur_ms = float(b.stim_duration_s) * 1e3
    tstop_ms = pre_pad_ms + dur_ms + post_pad_ms
    t_ms, v = cell.simulate(
        stim_amp_pA=float(b.amplitude_pA), stim_delay_ms=pre_pad_ms,
        stim_dur_ms=dur_ms, tstop_ms=tstop_ms,
        v_init_mV=v_rest_mV, dt_ms=dt_ms)
    return np.asarray(t_ms) * 1e-3 - pre_pad_ms * 1e-3, np.asarray(v)


def _simulate_long(cell, b, v_rest_mV: float,
                   dt_ms: Optional[float] = None
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Replay a long step using the bundle's own onset/duration; t in s on the
    simulation grid (interpolated to the experimental grid by the RMSD step).

    `dt_ms=None` resolves DEFAULT_DT_LONG_MS at call time. The former default
    of 0.1 was silently overridden to 0.025 by stdrun's setdt(); PassiveCell
    .simulate now enforces whatever is requested here."""
    dt_ms = float(DEFAULT_DT_LONG_MS if dt_ms is None else dt_ms)
    delay_ms = float(b.stim_onset_s) * 1e3
    dur_ms = float(b.stim_duration_s) * 1e3
    tstop_ms = float(b.t[-1]) * 1e3
    t_ms, v = cell.simulate(
        stim_amp_pA=float(b.amplitude_pA), stim_delay_ms=delay_ms,
        stim_dur_ms=dur_ms, tstop_ms=tstop_ms,
        v_init_mV=v_rest_mV, dt_ms=dt_ms)
    return np.asarray(t_ms) * 1e-3, np.asarray(v)


# ---------------------------------------------------------------------------
#  Per-sample time-weighting for the brief (SS) pulse RMSD
# ---------------------------------------------------------------------------
#  Statistical reading (cf. notation/likelihood directive): weighting the
#  squared residual at time t_s by w(t_s) is EXACTLY a Gaussian likelihood with
#  time-varying noise sigma_b^2(t_s) = 1 / w_b(t_s). The exponential default
#  below therefore asserts sigma_b^2(t) ~ exp(+(t - t0)/tau_w) -- a standard,
#  defensible heteroscedastic form -- and concentrates leverage on the early,
#  Cm-bearing samples just after the pulse peak. The resulting (Cm,Rm,Ra) is a
#  weighted-LS / quasi-MLE (NOT the homoscedastic MLE); report it as such and
#  propagate uncertainty with the sandwich covariance A^{-1} B A^{-1} or a
#  bootstrap under the TRUE (stationary) noise -- never the naive weighted
#  Fisher inverse.
#
#  Factories take/return SECONDS to match this module's internal t-grid, and
#  are pure + decoupled so the shape can be swapped without touching the RMSD
#  or loss code. They are applied to BRIEF bundles only (see bundle_rmsd).
def exp_time_weight(t0_s: float,
                    tau_w_s: float) -> Callable[[np.ndarray], np.ndarray]:
    """w(t) = exp(-(t - t0)/tau_w) for t >= t0, clamped to 1 for t < t0.
    Implied noise sigma^2(t)  prop_to  exp(+(t - t0)/tau_w): variance e-folds in tau_w,
    std in 2*tau_w. DEFAULT shape (see module note for why over the Gaussian)."""
    if not (tau_w_s > 0):
        raise ValueError(f"tau_w_s must be > 0, got {tau_w_s}")

    def _w(t_s: np.ndarray) -> np.ndarray:
        dt = np.maximum(np.asarray(t_s, dtype=float) - t0_s, 0.0)
        return np.exp(-dt / tau_w_s)
    return _w


def gauss_time_weight(t0_s: float,
                      sigma_w_s: float) -> Callable[[np.ndarray], np.ndarray]:
    """w(t) = exp(-(t - t0)^2 / (2 sigma_w^2)) for t >= t0, clamped to 1 below.
    Implied noise sigma^2(t)  prop_to  exp(+(t - t0)^2 / (2 sigma_w^2)) grows SUPER-
    exponentially -> sharper tail cutoff, flat top near t0. Provided as a
    swap-in for the tau_w/shape sweep; exponential is the default."""
    if not (sigma_w_s > 0):
        raise ValueError(f"sigma_w_s must be > 0, got {sigma_w_s}")

    def _w(t_s: np.ndarray) -> np.ndarray:
        dt = np.maximum(np.asarray(t_s, dtype=float) - t0_s, 0.0)
        return np.exp(-(dt * dt) / (2.0 * sigma_w_s * sigma_w_s))
    return _w


def _baseline_subtracted_rmsd(
        t_exp_s, v_exp, t_sim_s, v_sim, pre_window_s, rmsd_window_s,
        sample_weight_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> float:
    """RMSD in rmsd_window_s after removing each trace's own pre-window mean.

    If `sample_weight_fn` is given, returns the WEIGHTED root-mean-square
        sqrt( sum_s w(t_s) d_s^2 / sum_s w(t_s) ),   w = sample_weight_fn(t_s),
    normalised by sum(w) (NOT N) so the value stays in mV and on the same scale
    as the unweighted long-step RMSDs and the monolith's mV thresholds.
    `sample_weight_fn=None` reproduces the uniform RMSD exactly. Identical in
    spirit to the monolith's _baseline_subtracted_rmsd."""
    def _mean_in(t, v, w):
        m = (t >= w[0]) & (t <= w[1])
        return float(np.mean(v[m])) if m.any() else np.nan
    be = _mean_in(t_exp_s, v_exp, pre_window_s)
    bs = _mean_in(t_sim_s, v_sim, pre_window_s)
    if not (np.isfinite(be) and np.isfinite(bs)):
        return float("inf")
    m = (t_exp_s >= rmsd_window_s[0]) & (t_exp_s <= rmsd_window_s[1])
    if m.sum() < 2:
        return float("inf")
    tg = t_exp_s[m]
    ve = (v_exp - be)[m]
    vs = np.interp(tg, t_sim_s, v_sim - bs)
    d = ve - vs
    if sample_weight_fn is None:
        return float(np.sqrt(np.mean(d * d)))
    w = np.asarray(sample_weight_fn(tg), dtype=float)
    sw = float(np.sum(w))
    if not np.all(np.isfinite(w)) or not (sw > 0):
        return float("inf")
    return float(np.sqrt(np.sum(w * d * d) / sw))


def _peak_deflection(bundle, pre_window_s, win_s) -> float:
    """Max |v - baseline| of the EXPERIMENTAL bundle inside win_s (mV)."""
    t = np.asarray(bundle.t)
    v = np.asarray(bundle.v_mV)
    pm = (t >= pre_window_s[0]) & (t <= pre_window_s[1])
    base = float(np.mean(v[pm])) if pm.any() else 0.0
    wm = (t >= win_s[0]) & (t <= win_s[1])
    return float(np.max(np.abs(v[wm] - base))) if wm.any() else np.nan


# ---------------------------------------------------------------------------
#  Per-bundle RMSD with protocol-appropriate window
# ---------------------------------------------------------------------------
LS_WINDOW_MODES = ("after_onset", "step", "sweep")


def ls_rmsd_window_s(bundle, mode: str, ls_window_ms_after_onset: float
                     ) -> Tuple[float, float]:
    """RMSD window [s] of ONE long-square bundle, by mode.

      "after_onset"  [t_on, t_on + ls_window_ms_after_onset]   (legacy; the
                     truncation that kept I_h OUT of the passive loss)
      "step"         [t_on, t_off]                             (D-006 default
                     for the I_h fit: the whole step, so the sag and its
                     amplitude dependence are IN the loss)
      "sweep"        [t_on, t_end]                             (adds the
                     post-offset rebound; excluded from TRAINING by D-006
                     because the conductances that shape the post-inhibitory
                     rebound are not in this model, and reported instead)
    """
    if mode not in LS_WINDOW_MODES:
        raise ValueError("ls_window mode must be one of %s, got %r"
                         % (LS_WINDOW_MODES, mode))
    onset = float(bundle.stim_onset_s)
    if mode == "after_onset":
        return (onset, onset + float(ls_window_ms_after_onset) * 1e-3)
    if mode == "step":
        return (onset, onset + float(bundle.stim_duration_s))
    return (onset, float(np.asarray(bundle.t)[-1]))


def ls_subwindows_s(bundle, *, charge_ms: float = 150.0
                    ) -> "Dict[str, Tuple[float, float]]":
    """The three diagnostic sub-windows of a long step, reported per bundle
    and never minimised (plan section 7): charging (where tau_m and C_m act),
    sag (where gbar and dv_h act), rebound (after the offset, where kappa_tau
    acts most cleanly because no stimulus is on)."""
    onset = float(bundle.stim_onset_s)
    off = onset + float(bundle.stim_duration_s)
    end = float(np.asarray(bundle.t)[-1])
    charge_end = min(onset + float(charge_ms) * 1e-3, off)
    return {"charge": (onset, charge_end),
            "sag": (charge_end, off),
            "rebound": (off, end)}


# ---------------------------------------------------------------------------
#  The long square's sample weight (SPEC 2.8 (d), D-207 (3); name peak_decay)
# ---------------------------------------------------------------------------
LS_SAMPLE_WEIGHTS = ("none", "peak_decay")
_PEAK_GUARD_S = 1e-3          # the peak is searched from t_on + 1 ms (2.8 (d))


def _check_ls_sample_weight(name: str) -> None:
    if name not in LS_SAMPLE_WEIGHTS:
        raise ValueError("ls_sample_weight must be one of %s, got %r"
                         % (LS_SAMPLE_WEIGHTS, name))


def _is_depolarising_ls(bundle) -> bool:
    """A long square is depolarising when its current is positive."""
    return float(bundle.amplitude_pA) > 0.0


def ls_peak_time_s(bundle, *, smooth_ms: float = 1.0) -> float:
    """t_pk of 2.8 (d), in s on the bundle's own time axis.

    The time of the minimum (the maximum for a depolarising step) over the
    grid samples t in [t_on + 1 ms, t_off], t_off = t_on + stim_duration_s,
    of the recorded voltage smoothed by a centred boxcar of `smooth_ms`. The
    boxcar holds 2 h + 1 samples, h = round(smooth_ms / 2 / dt), and uses the
    samples outside the interval wherever it reaches them; at the ends of the
    trace, and over non-finite samples, it averages the finite samples it
    covers. Ties: the earliest sample. Depends on the recorded trace only.
    """
    # Custom: the peak rule of SPEC 2.8 (d) (interval, guard, tie rule) is
    # not a library function. Checked: scipy.ndimage.uniform_filter1d (pads
    # rather than averaging the samples it covers; no NaN handling) and
    # scipy.signal.find_peaks (local extrema, not the interval's extremum).
    # The boxcar itself is np.convolve.
    if _is_brief(bundle):
        raise ValueError("ls_peak_time_s: a brief (short-square) bundle has no "
                         "long-square peak (stim_duration_s=%g s)"
                         % float(bundle.stim_duration_s))
    if not (float(smooth_ms) > 0):
        raise ValueError("smooth_ms must be > 0, got %r" % (smooth_ms,))
    t = np.asarray(bundle.t, dtype=float)
    v = np.asarray(bundle.v_mV, dtype=float)
    t_on = float(bundle.stim_onset_s)
    t_off = t_on + float(bundle.stim_duration_s)
    idx = np.flatnonzero((t >= t_on + _PEAK_GUARD_S) & (t <= t_off))
    if idx.size < 3:
        raise ValueError("ls_peak_time_s: %d sample(s) in [t_on + 1 ms, t_off] "
                         "= [%.6g, %.6g] s; at least 3 are needed"
                         % (idx.size, t_on + _PEAK_GUARD_S, t_off))
    dt = float(np.median(np.diff(t)))
    half = int(round(0.5 * float(smooth_ms) * 1e-3 / dt))
    i0 = max(int(idx[0]) - half, 0)
    i1 = min(int(idx[-1]) + half + 1, v.size)
    seg = v[i0:i1]
    fin = np.isfinite(seg)
    kern = np.ones(2 * half + 1)
    num = np.convolve(np.where(fin, seg, 0.0), kern, mode="same")
    cnt = np.convolve(fin.astype(float), kern, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        sm = num / cnt
    sm_in = sm[int(idx[0]) - i0:int(idx[-1]) - i0 + 1]   # NaN where cnt == 0
    if not np.any(np.isfinite(sm_in)):
        raise ValueError("ls_peak_time_s: no finite sample in the interval")
    k = (int(np.nanargmax(sm_in)) if _is_depolarising_ls(bundle)
         else int(np.nanargmin(sm_in)))
    return float(t[int(idx[0]) + k])


def peak_decay_weight(t_on_s: float, t_pk_s: float, tau_d_s: float
                      ) -> Callable[[np.ndarray], np.ndarray]:
    """w_b of 2.8 (d) as a function of an array of times (s):
    0 before t_on; (t - t_on) / (t_pk - t_on) on [t_on, t_pk];
    exp(-(t - t_pk) / tau_d) after t_pk."""
    # Custom: the piecewise weight of SPEC 2.8 (d); no library function.
    t_on_s = float(t_on_s)
    t_pk_s = float(t_pk_s)
    tau_d_s = float(tau_d_s)
    if not (np.isfinite(t_on_s) and np.isfinite(t_pk_s) and t_pk_s > t_on_s):
        raise ValueError("peak_decay_weight needs t_pk_s > t_on_s, got "
                         "t_on_s=%r, t_pk_s=%r" % (t_on_s, t_pk_s))
    if not (np.isfinite(tau_d_s) and tau_d_s > 0):
        raise ValueError("tau_d_s must be > 0, got %r" % (tau_d_s,))

    def _w(t_s: np.ndarray) -> np.ndarray:
        t = np.asarray(t_s, dtype=float)
        rise = (t - t_on_s) / (t_pk_s - t_on_s)
        decay = np.exp(-np.maximum(t - t_pk_s, 0.0) / tau_d_s)
        return np.where(t < t_on_s, 0.0, np.where(t <= t_pk_s, rise, decay))
    return _w


def _ls_weight_fn(bundle, ls_sample_weight: str, ls_tau_d_ms: float
                  ) -> Optional[Callable[[np.ndarray], np.ndarray]]:
    """The sample weight of one LONG bundle's RMSD (None = uniform)."""
    _check_ls_sample_weight(ls_sample_weight)
    if ls_sample_weight == "none":
        return None
    return peak_decay_weight(float(bundle.stim_onset_s),
                             ls_peak_time_s(bundle),
                             float(ls_tau_d_ms) * 1e-3)


def bundle_rmsd(cell, bundle, v_rest_mV: float, *,
                ss_window_ms: Tuple[float, float] = (0.5, 100.0),
                ls_window_ms_after_onset: float = 150.0,
                ls_window_mode: str = "after_onset",
                r_in_target: str = "peak",
                ss_sample_weight_fn: Optional[Callable[[np.ndarray], np.ndarray]]
                = None,
                dt_brief_ms: Optional[float] = None,
                dt_long_ms: Optional[float] = None,
                ls_sample_weight: str = "none",
                ls_tau_d_ms: float = 500.0) -> Tuple[float, float]:
    """Return (rmsd_mV, deflection_mV) for one bundle.

    `ls_sample_weight` (SPEC 2.8 (d), D-207 (3)) is the per-sample weight of
    a LONG bundle's RMSD: "none" (default) is the uniform RMSD, bit for bit
    the previous behaviour; "peak_decay" weights it by w_b of
    `peak_decay_weight`, with t_pk from the recorded trace
    (`ls_peak_time_s`) and tau_d = `ls_tau_d_ms`. Brief bundles ignore it.
    The deflection is never weighted.

    `ss_window_ms` defaults to (0.5, 100.0): the start is the 0.5 ms pulse
    OFFSET (not 1.0 ms), so the Cm-bearing early decay just after the peak is
    INSIDE the window. `ss_sample_weight_fn`, if given, applies the per-sample
    time-weight to BRIEF bundles only; long bundles always keep the uniform
    RMSD (the time-weight is SS-only by design).

    `deflection_mV` is the |peak| (r_in_target='peak') or |tail-plateau|
    (r_in_target='steady') experimental deflection in the fit window, used
    only for optional relative weighting; it does not change the RMSD itself.
    """
    _check_ls_sample_weight(ls_sample_weight)
    if _is_brief(bundle):
        t_s, v = _simulate_brief(cell, bundle, v_rest_mV, dt_ms=dt_brief_ms)
        pre_w = (-10e-3, 0.0)
        rmsd_w = (ss_window_ms[0] * 1e-3, ss_window_ms[1] * 1e-3)
        weight_fn = ss_sample_weight_fn
    else:
        t_s, v = _simulate_long(cell, bundle, v_rest_mV, dt_ms=dt_long_ms)
        onset = float(bundle.stim_onset_s)
        pre_w = (0.0, onset)
        rmsd_w = ls_rmsd_window_s(bundle, ls_window_mode, ls_window_ms_after_onset)
        weight_fn = _ls_weight_fn(bundle, ls_sample_weight, ls_tau_d_ms)
    rmsd = _baseline_subtracted_rmsd(
        np.asarray(bundle.t), np.asarray(bundle.v_mV), t_s, v, pre_w, rmsd_w,
        sample_weight_fn=weight_fn)
    if r_in_target == "steady":
        defl = _peak_deflection(bundle, pre_w,
                                (rmsd_w[1] - 20e-3, rmsd_w[1]))
    else:
        defl = _peak_deflection(bundle, pre_w, rmsd_w)
    return rmsd, defl


# ---------------------------------------------------------------------------
#  The sweeps of one loss evaluation in parallel (SPEC 2.9 (c), Block 19)
# ---------------------------------------------------------------------------
#  A forked worker inherits the parent's memory, so the cell, the bundles and
#  the RMSD options reach it through this registry, never through pickling
#  (a NEURON cell cannot be pickled). Each loss with n_workers > 1 registers
#  its context under a token right before its pool forks, and removes it when
#  the pool is closed. Only (token, q, bundle index, dt) cross the pipe.
_POOL_CONTEXTS: Dict[int, tuple] = {}
_POOL_TOKENS = itertools.count(1)
# One live pool per process (SPEC Block 19, amendment 2, D-237): the state of
# every loss whose pool is alive, by id. A pool keeps a manager thread in the
# parent, and a fork made while another thread runs can deadlock the child,
# so a loss closes every other live pool before it forks its own.
_LIVE_POOLS: Dict[int, dict] = {}


class _RemoteDtEnforcementError(RuntimeError):
    """Raised in the parent for a worker's dt-enforcement failure when the
    worker's own exception could not be sent back (not picklable)."""


# The class name is the contract (SPEC Block 19): callers match by name.
_RemoteDtEnforcementError.__name__ = "DtEnforcementError"
_RemoteDtEnforcementError.__qualname__ = "DtEnforcementError"


def _check_n_workers(n_workers) -> int:
    """n_workers must be an integer >= 1 (bool refused); returns it as int."""
    if (isinstance(n_workers, bool)
            or not isinstance(n_workers, numbers.Integral)
            or int(n_workers) < 1):
        raise ValueError("n_workers must be an integer >= 1, got %r"
                         % (n_workers,))
    return int(n_workers)


def _pool_task(token: int, q: tuple, i: int, dt_brief_ms: float,
               dt_long_ms: float) -> tuple:
    """Worker side: apply q to this process's copy of the cell, return the
    pair (RMSD_b, D_b) of bundle i as the serial loop computes it.

    Every exception is caught and sent back tagged, so the parent can apply
    the serial loop's rules in the bundles' order:
        ("ok", rmsd, defl) | ("dt", pickled exception or None, text)
        | ("err", class name, text)
    """
    try:
        cell, spec, bundles, v_rest_mV, kw = _POOL_CONTEXTS[token]
        spec.apply_q(cell, q, v_rest_mV)
        rmsd, defl = bundle_rmsd(cell, bundles[i], v_rest_mV,
                                 dt_brief_ms=dt_brief_ms,
                                 dt_long_ms=dt_long_ms, **kw)
        return ("ok", rmsd, defl)
    except Exception as exc:  # noqa: BLE001 - sent back, judged by the parent
        text = "".join(traceback.format_exception(type(exc), exc,
                                                  exc.__traceback__))
        if type(exc).__name__ == "DtEnforcementError":
            try:
                blob = pickle.dumps(exc)
            except Exception:  # noqa: BLE001 - fall back to the named class
                blob = None
            return ("dt", blob, text)
        return ("err", type(exc).__name__, text)


def _raise_remote_dt(blob, text):
    """Raise a worker's DtEnforcementError in the parent: the original
    exception when it could be sent back, else one of the same class name."""
    exc = None
    if blob is not None:
        try:
            exc = pickle.loads(blob)
        except Exception:  # noqa: BLE001
            exc = None
    if exc is None:
        exc = _RemoteDtEnforcementError(text)
    raise exc


def _close_pool(state: dict) -> None:
    """Shut the pool of one loss down and drop its context. Idempotent; used
    by loss.close(), by the garbage-collection finalizer and at exit. The
    workers are idle between evaluations (the loss waits for every task), so
    shutdown(wait=True) returns once they have exited."""
    pool, token = state.get("pool"), state.get("token")
    state["pool"] = None
    state["token"] = None
    _LIVE_POOLS.pop(id(state), None)
    try:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
    finally:
        if token is not None:
            _POOL_CONTEXTS.pop(token, None)


def _make_pooled_loss(cell, train_bundles, v_rest_mV, spec, n_workers,
                      rmsd_kw, dt_brief_ms, dt_long_ms, combine):
    """The loss of build_multi_protocol_loss with the pairs (RMSD_b, D_b)
    computed by a fork pool (SPEC 2.9 (c)). `combine(outs)` is the serial
    loop's body over the workers' tagged outcomes, in the bundles' order."""
    bundles = tuple(train_bundles)
    n_proc = min(int(n_workers), len(bundles))
    state = {"pool": None, "token": None}

    def _pool():
        if state["pool"] is None:
            # One live pool per process: close every other loss's pool first
            # (as its close() does; it makes a new pool at its next call).
            for other in list(_LIVE_POOLS.values()):
                if other is not state:
                    _close_pool(other)
            token = next(_POOL_TOKENS)
            _POOL_CONTEXTS[token] = (cell, spec, bundles, v_rest_mV, rmsd_kw)
            state["token"] = token
            # The workers fork at the first submit, with the cell as it is
            # now (SPEC Block 19: the pool is created at the first call).
            state["pool"] = ProcessPoolExecutor(
                max_workers=n_proc,
                mp_context=multiprocessing.get_context("fork"))
            _LIVE_POOLS[id(state)] = state
        return state["pool"], state["token"]

    def loss(*q: float) -> float:
        try:
            # The serial code's side effect: the parent's cell is left at q.
            spec.apply_q(cell, q, v_rest_mV)
        except Exception as exc:  # noqa: BLE001 - as the serial loss
            if type(exc).__name__ == "DtEnforcementError":
                raise
            return 1e6
        if not bundles:
            try:
                return combine([])
            except Exception:  # noqa: BLE001 - as the serial loss
                return 1e6
        # dt resolved now, as the serial loop resolves it at each call
        dtb = float(DEFAULT_DT_BRIEF_MS if dt_brief_ms is None else dt_brief_ms)
        dtl = float(DEFAULT_DT_LONG_MS if dt_long_ms is None else dt_long_ms)
        pool, token = _pool()
        n = len(bundles)
        try:
            outs = list(pool.map(_pool_task, [token] * n, [tuple(q)] * n,
                                 range(n), [dtb] * n, [dtl] * n, chunksize=1))
        except BrokenProcessPool:
            # A worker died (not an exception inside it): the pool is
            # unusable and this evaluation has no value. Never a penalty.
            _close_pool(state)
            raise
        try:
            return combine(outs)
        except Exception as exc:  # noqa: BLE001 - as the serial loss
            if type(exc).__name__ == "DtEnforcementError":
                raise
            return 1e6

    loss.n_workers = int(n_workers)
    loss.close = lambda: _close_pool(state)
    # terminated when the loss is garbage-collected, and at interpreter exit
    weakref.finalize(loss, _close_pool, state)
    return loss


# ---------------------------------------------------------------------------
#  Multi-protocol loss (log-space, drop-in for _build_loss_function)
# ---------------------------------------------------------------------------
def build_multi_protocol_loss(
    cell, train_bundles: Sequence, v_rest_mV: float, *,
    ss_window_ms: Tuple[float, float] = (0.5, 100.0),
    ls_window_ms_after_onset: float = 150.0,
    ls_window_mode: str = "after_onset",
    r_in_target: str = "peak",
    weighting: str = "relative",
    weights: Optional[Sequence[float]] = None,
    ss_sample_weight_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    dt_brief_ms: Optional[float] = None,
    dt_long_ms: Optional[float] = None,
    spec: Optional[object] = None,
    ls_sample_weight: str = "none",
    ls_tau_d_ms: float = 500.0,
    n_workers: int = 1,
) -> Callable[..., float]:
    """Return loss(*q) -> scalar, q in the spec's axis order.

    `spec` is a param_spec.ParamSpec. None means the 3-D passive spec, for
    which the signature is loss(cm_log, rm_log, ra_log) -- the legacy one,
    unchanged, so every existing call site keeps working (Stage 3, D-005).

    ss_sample_weight_fn:
      Per-sample time-weight (see exp_time_weight) applied to BRIEF bundles
      only -- the WITHIN-bundle, across-time axis. None -> uniform brief-pulse
      RMSD (previous behaviour). Orthogonal to `weighting` below.

    weighting (the cross-bundle axis):
      'relative' (default) -> mean of (RMSD_bundle / deflection_bundle): a
                    unitless relative RMSD so the small-deflection brief pulse
                    and the larger long step contribute commensurately. The
                    returned scalar is UNITLESS -- the monolith's mV-based
                    thresholds (train_rmsd_fail_mV, valid_rmsd_good_mV) and the
                    valid/train ratio must be re-tuned as fractions.
      'equal'    -> mean of per-bundle mV RMSDs (stays in mV). The long step's
                    larger signal then dominates the mean; the brief pulse keeps
                    its Cm constraint but with less leverage.
      'custom'   -> use `weights` (len == len(train_bundles)).
      'ls_amplitude' -> SPEC 2.8 (c) (D-207 (2)):
                    L = [sum_SS r_b/D_b + n_LS sum_LS omega_b r_b/D_b]
                        / (n_SS + n_LS),
                    omega_b = |I_b| / sum_LS |I_b'|, r_b/D_b the 'relative'
                    term; the long bundles' weights sum to n_LS, so equal
                    amplitudes give the 'relative' loss.

    ls_sample_weight, ls_tau_d_ms:
      the long bundles' per-sample weight of 2.8 (d) ('none' | 'peak_decay'),
      passed to bundle_rmsd. 'none' (default) is the previous behaviour.

    n_workers (SPEC 2.9 (c), Block 19):
      1 (default) -> the serial loop below, unchanged. W > 1 -> the pairs
      (RMSD_b, D_b) are computed by min(W, number of bundles) processes of a
      fork pool created at the FIRST call of the loss (the workers inherit
      the cell as it is then) and reused; each worker applies q to its own
      copy of the cell, the parent applies q to its cell too and combines
      the pairs in the bundles' order by the serial arithmetic, so the value
      equals the serial one exactly. A worker's exception gives 1e6 as the
      serial loop does, except a DtEnforcementError (raised in the parent
      under that class name); a worker that dies raises BrokenProcessPool.
      The loss carries `loss.n_workers` and `loss.close()` (terminates the
      pool; idempotent; a later call creates a new pool; also run when the
      loss is garbage-collected and at exit; a no-op for W = 1). One live
      pool per process: a loss about to create its pool first closes the
      live pool of any other loss (which makes a new one at its next call).
      ValueError unless an integer >= 1.
    """
    n_workers = _check_n_workers(n_workers)
    _check_ls_sample_weight(ls_sample_weight)
    if weighting == "custom":
        if weights is None or len(weights) != len(train_bundles):
            raise ValueError("weighting='custom' needs weights of matching length")
        w = np.asarray(weights, dtype=float)
    if ls_sample_weight != "none":
        # t_pk and w_b depend on the recorded trace only: build them once
        # here so a bundle the rule cannot serve raises now, instead of being
        # folded into the loss's 1e6 penalty at every evaluation.
        for b in train_bundles:
            if not _is_brief(b):
                _ls_weight_fn(b, ls_sample_weight, ls_tau_d_ms)
    if weighting == "ls_amplitude":
        is_ls = np.array([not _is_brief(b) for b in train_bundles], dtype=bool)
        n_ls = int(is_ls.sum())
        n_all = len(train_bundles)
        amps = np.array([abs(float(b.amplitude_pA)) for b in train_bundles],
                        dtype=float)
        tot = float(np.sum(amps[is_ls]))
        if n_ls and not (np.isfinite(tot) and tot > 0):
            raise ValueError("weighting='ls_amplitude' needs a positive total "
                             "|amplitude| over the training long squares, got %r"
                             % tot)
        if n_all == 0:
            raise ValueError("weighting='ls_amplitude' needs training bundles")
        # c_b: 1 for a short square, n_LS * omega_b for a long square
        w = np.where(is_ls, n_ls * amps / (tot if n_ls else 1.0), 1.0)
    if spec is None:
        import param_spec as _ps
        spec = _ps.PASSIVE_3D

    def loss(*q: float) -> float:
        try:
            spec.apply_q(cell, q, v_rest_mV)
            terms = []
            for b in train_bundles:
                rmsd, defl = bundle_rmsd(
                    cell, b, v_rest_mV, ss_window_ms=ss_window_ms,
                    ls_window_ms_after_onset=ls_window_ms_after_onset,
                    ls_window_mode=ls_window_mode,
                    r_in_target=r_in_target,
                    ss_sample_weight_fn=ss_sample_weight_fn,
                    dt_brief_ms=dt_brief_ms, dt_long_ms=dt_long_ms,
                    ls_sample_weight=ls_sample_weight,
                    ls_tau_d_ms=ls_tau_d_ms)
                if not np.isfinite(rmsd):
                    return 1e6
                if weighting in ("relative", "ls_amplitude"):
                    terms.append(rmsd / max(abs(defl), 1e-6))
                else:
                    terms.append(rmsd)
            terms = np.asarray(terms, dtype=float)
            if weighting == "custom":
                return float(np.sum(w * terms) / np.sum(w))
            if weighting == "ls_amplitude":
                return float(np.sum(w * terms) / len(terms))
            return float(np.mean(terms))
        except Exception as exc:  # noqa: BLE001 - mirror monolith: penalise
            # A dt-enforcement failure is a broken integrator configuration,
            # not a bad region of parameter space. Folding it into the 1e6
            # penalty would let the whole fit proceed on wrong simulations, so
            # it is re-raised. Matched by CLASS NAME rather than by import so
            # this module stays decoupled from the monolith.
            if type(exc).__name__ == "DtEnforcementError":
                raise
            return 1e6

    # SPEC Block 23: the residual vector of 2.10 (a) of this loss's protocol
    make_residuals = _residuals_factory(
        cell, train_bundles, v_rest_mV, weighting=weighting, spec=spec,
        dt_brief_ms=dt_brief_ms, dt_long_ms=dt_long_ms, n_workers=n_workers,
        protocol_kw=dict(ss_window_ms=ss_window_ms,
                         ls_window_ms_after_onset=ls_window_ms_after_onset,
                         ls_window_mode=ls_window_mode,
                         r_in_target=r_in_target,
                         ss_sample_weight_fn=ss_sample_weight_fn,
                         ls_sample_weight=ls_sample_weight,
                         ls_tau_d_ms=ls_tau_d_ms))

    if n_workers == 1:
        loss.n_workers = 1
        loss.close = lambda: None
        loss.make_residuals = make_residuals
        return loss

    def combine(outs) -> float:
        # The body of the serial loop above, on the workers' outcomes in the
        # bundles' order: same tests, same order, same arithmetic.
        terms = []
        for out in outs:
            if out[0] == "dt":
                _raise_remote_dt(out[1], out[2])
            if out[0] == "err":
                return 1e6
            rmsd, defl = out[1], out[2]
            if not np.isfinite(rmsd):
                return 1e6
            if weighting in ("relative", "ls_amplitude"):
                terms.append(rmsd / max(abs(defl), 1e-6))
            else:
                terms.append(rmsd)
        terms = np.asarray(terms, dtype=float)
        if weighting == "custom":
            return float(np.sum(w * terms) / np.sum(w))
        if weighting == "ls_amplitude":
            return float(np.sum(w * terms) / len(terms))
        return float(np.mean(terms))

    rmsd_kw = dict(ss_window_ms=ss_window_ms,
                   ls_window_ms_after_onset=ls_window_ms_after_onset,
                   ls_window_mode=ls_window_mode, r_in_target=r_in_target,
                   ss_sample_weight_fn=ss_sample_weight_fn,
                   ls_sample_weight=ls_sample_weight, ls_tau_d_ms=ls_tau_d_ms)
    pooled = _make_pooled_loss(cell, train_bundles, v_rest_mV, spec, n_workers,
                               rmsd_kw, dt_brief_ms, dt_long_ms, combine)
    pooled.make_residuals = make_residuals
    return pooled


# ---------------------------------------------------------------------------
#  Training-set builder: brief hyp pulses + two smallest hyp long steps
# ---------------------------------------------------------------------------
def split_train_validation_with_long_step(
    cell_data, *,
    fit_target: str = "hyp",
    n_long_train: int = 2,
    max_ls_train_deflection_mV: Optional[float] = 12.0,
    max_sag_amplitude_mV: Optional[float] = None,
    max_ls_train_amplitude_pA: Optional[float] = None,
    verbose: bool = True,
) -> Tuple[List, List]:
    """Return (train_bundles, validation_bundles).

    train      = brief SS bundles of the chosen polarity
                 + the `n_long_train` smallest-|amplitude| hyperpolarising
                   long-square bundles.
    validation = remaining brief SS bundles (opposite polarity) + remaining
                 long-square bundles.

    SS-ONLY training: set `n_long_train=0` to train on the brief pulses ALONE
    (no long step is forced in; every long step goes to validation). In that
    mode the brief pulse must carry tau_m / R_in as well as Cm, so prefer a
    GENTLER time-weight (larger tau_w) than in the multi-protocol mode, where
    the long steps already pin tau_m / R_in and an aggressive early up-weight on
    the SS pulse is safe.

    Why the TWO smallest long-square amplitudes (default n_long_train=2), not one
    ---------------------------------------------------------------------------
    I_h is hyperpolarisation-activated: a larger hyperpolarising step drives V
    further below V_rest, recruits more I_h, and sags more. The smallest-|amp|
    steps therefore stay closest to rest in the lowest-I_h regime. Using the two
    smallest (rather than a single one) dampens the residual I_h contamination of
    the recovered passive (Cm, Rm, Ra) in two ways:
      (1) SNR/robustness -- the single smallest step has the worst SNR; averaging
          the constraint over the two smallest de-weights any one trace's noise
          and the chance that the very smallest step is poorly resolved;
      (2) joint linearity -- a sagging cell's APPARENT R_in = dV/I already differs
          between two amplitudes, and one amplitude-independent passive (Cm,Rm,Ra)
          cannot absorb that amplitude dependence into biased Rm. Fitting both at
          once therefore pulls the best passive compromise toward the near-rest,
          low-I_h regime instead of over-fitting one sag-contaminated trace.
    This reduces, but does not eliminate, I_h bias (both traces still sag a
    little); full removal needs an explicit I_h channel or a pre-sag-only window.

    The long-square list is assumed sorted smallest-|amp| first (the loader sorts
    by abs amplitude); we re-sort defensively.

    I_h guard (deflection-based, because I_h is VOLTAGE-gated)
    ---------------------------------------------------------
    "Two smallest amplitudes" is not safe by itself: a cell with sparse sweeps
    may have its second-smallest step at a large amplitude (e.g. a cell whose
    only steps are -10 and -90 pA), which would re-inject the very I_h sag we
    are avoiding. Since I_h activation depends on how far V is driven below rest,
    the cut is on the estimated steady-state deflection
        |dV|[mV] ~= |amplitude_pA| * R_in[MOhm] * 1e-3
    not on the current: a -70 pA step is ~6 mV in a 93 MOhm cell but ~24 mV in a
    339 MOhm cell. A step is admissible for TRAINING iff
        |dV| <= max_ls_train_deflection_mV   (default 12 mV; None disables).
    Rejected steps are HELD OUT (kept in validation), not discarded. If R_in is
    NaN the deflection guard is skipped. An optional hard current cap
    max_ls_train_amplitude_pA is also available. If the guard rejects every
    step, training falls back to the single smallest-|amplitude| step so a
    long-step constraint is always present.

    Sag-ratio-gated guard (optional; `max_sag_amplitude_mV`)
    -------------------------------------------------------
    A fixed mV ceiling is a blunt proxy for "how much I_h will this step
    recruit". If the cell's sag ratio is known (carried on
    `cell_data.sag_ratio`, written by the archive saver from Allen's `sag`
    feature), the guard can instead bound the *estimated I_h sag amplitude*
    the step will inject -- the quantity a purely passive model literally
    cannot fit -- rather than the deflection. With

        s   = cell_data.sag_ratio                       (Allen sag, ~0..0.5)
        dV  = |amplitude_pA| * R_in[MOhm] * 1e-3          (steady deflection, mV)

    the peak (pre-sag) deflection is dV/(1-s) and the sag amplitude is

        sag_amp[mV] = s * dV / (1 - s)        (s clamped to [0, 0.95])

    A step is admissible iff sag_amp <= `max_sag_amplitude_mV`. This adapts the
    effective deflection ceiling to each cell's I_h: a near-passive cell
    (s ~ 0) admits large steps, a high-sag cell is held to the smallest --
    exactly the per-cell behaviour a single fixed mV cap cannot give. It is
    opt-in (None disables) and is applied IN ADDITION to any fixed
    `max_ls_train_deflection_mV`; a step must pass every active criterion. When
    sag-gating is the intended control, set `max_ls_train_deflection_mV=None`
    so the sag bound alone drives admissibility. If `cell_data.sag_ratio` is
    NaN/absent (e.g. an archive built before sag was saved) the sag criterion
    is skipped and behaviour falls back to the fixed-mV guard, so nothing
    breaks on older archives.
    """
    ss = list(cell_data.square_subthreshold)
    if fit_target == "hyp":
        train_ss = [b for b in ss if b.polarity == "hyp"]
        held_ss = [b for b in ss if b.polarity == "dep"]
    elif fit_target == "dep":
        train_ss = [b for b in ss if b.polarity == "dep"]
        held_ss = [b for b in ss if b.polarity == "hyp"]
    elif fit_target == "both":
        train_ss, held_ss = list(ss), []
    else:
        raise ValueError(f"fit_target must be 'hyp'|'dep'|'both', got {fit_target!r}")

    ls_sorted = sorted(cell_data.long_square_subthreshold,
                       key=lambda b: abs(float(b.amplitude_pA)))
    rin = float(getattr(cell_data, "rin_MOhm", float("nan")))
    sag = float(getattr(cell_data, "sag_ratio", float("nan")))

    def _admissible(b) -> Tuple[bool, str]:
        amp = abs(float(b.amplitude_pA))
        dv = amp * rin * 1e-3 if np.isfinite(rin) else float("nan")
        if (max_ls_train_amplitude_pA is not None
                and amp > float(max_ls_train_amplitude_pA)):
            return False, f"|amp|={amp:.0f}pA>{max_ls_train_amplitude_pA:.0f}pA"
        if max_ls_train_deflection_mV is not None and np.isfinite(dv):
            if dv > float(max_ls_train_deflection_mV):
                return False, f"|dV|~{dv:.1f}mV>{max_ls_train_deflection_mV:.0f}mV"
        if (max_sag_amplitude_mV is not None
                and np.isfinite(sag) and np.isfinite(dv)):
            s = min(max(sag, 0.0), 0.95)             # clamp; s>=0.95 -> reject
            sag_amp = s * dv / (1.0 - s)             # estimated I_h sag amplitude
            if sag_amp > float(max_sag_amplitude_mV):
                return False, (f"sag~{sag_amp:.1f}mV(s={sag:.2f},dV={dv:.1f})"
                               f">{max_sag_amplitude_mV:.1f}mV")
        return True, ""

    n = max(0, int(n_long_train))
    ls_train: List = []
    ls_reject: List[Tuple[object, str]] = []
    for b in ls_sorted:
        if len(ls_train) >= n:
            break
        ok, why = _admissible(b)
        if ok:
            ls_train.append(b)
        else:
            ls_reject.append((b, why))

    # Fallback ONLY when the user asked for long steps (n>0) but the I_h guard
    # rejected them all. With n_long_train=0 the caller wants SS-ONLY training
    # (brief pulses alone) -> no fallback, no long step is forced in.
    if n > 0 and not ls_train and ls_sorted:
        ls_train = [ls_sorted[0]]
        if verbose:
            print(f"[split] I_h guard rejected all steps; falling back to "
                  f"smallest |amp|={abs(float(ls_sorted[0].amplitude_pA)):.0f} pA")

    chosen = {id(b) for b in ls_train}
    ls_valid = [b for b in ls_sorted if id(b) not in chosen]

    if verbose:
        if n == 0:
            print(f"[split] SS-ONLY training (n_long_train=0): "
                  f"{len(train_ss)} brief bundle(s), 0 long steps; "
                  f"{len(ls_sorted)} long step(s) -> validation")
        else:
            amps = ", ".join(f"{b.amplitude_pA:+.0f}" for b in ls_train)
            msg = f"[split] LS training steps ({len(ls_train)}): [{amps}] pA"
            if ls_reject:
                msg += "; held out for I_h: " + ", ".join(
                    f"{b.amplitude_pA:+.0f}pA({why})" for b, why in ls_reject)
            print(msg)

    return (train_ss + ls_train), (held_ss + ls_valid)


# ---------------------------------------------------------------------------
#  The I_h campaign's role assignment (Stage 4; D-006 Q6/Q7/Q12, D-007)
# ---------------------------------------------------------------------------
#  Every device the passive loss used to keep I_h OUT of the fit is removed
#  here -- the 12 mV deflection cap, the "two smallest steps" rule, the
#  60/150 ms truncation -- because with I_h in the model each of them deletes
#  the signal that identifies it. What replaces them is an explicit role per
#  sweep, recorded per cell so that a result can be read without guessing
#  which trace did what:
#
#    TRAIN        the SS hyperpolarising pulses (they carry C_m; nothing else
#                 does) + h_2 .. h_{n-1}, each on its whole step window.
#    VALIDATE     the spike-free depolarising long steps (I_h DE-activation,
#                 a regime never trained on) + h_1 (the weakest step) + the
#                 SS depolarising pulses.
#    REPORT ONLY  h_n, the strongest step, withheld because the conductances
#                 that shape the post-inhibitory rebound are not modelled;
#                 and every long step's post-offset window, for the same
#                 reason. Their residuals are WRITTEN, so the exclusion is
#                 checked rather than assumed.
#
#  Ordering convention used throughout: h_1 .. h_n are the hyperpolarising
#  long-square bundles sorted by INCREASING |amplitude|, so h_1 is the
#  weakest step and h_n the strongest.

def assign_ls_roles(cell_data, *, n_drop_weakest: int = 1,
                    n_drop_strongest: int = 1,
                    dep_n_validation: Optional[int] = 3,
                    v_trough_min_mV: Optional[float] = None,
                    train_all_hyp: bool = False,
                    verbose: bool = True,
                    role_rule: str = "d006") -> Dict[str, object]:
    """Assign every long-square bundle of one cell to train / validate /
    report, per D-006 (`role_rule="d006"`, default) or per SPEC 2.8 (b)
    (`role_rule="d207"`, D-207 (1)). Returns a dict with the three bundle
    lists and a per-bundle record for the log.

    role_rule="d207"
        With h_1..h_n by increasing |I| and m = ceil(n / 2): train h_i for
        2 <= i <= n, i != m; validate h_1 and h_m (one bundle when m = 1);
        no report-only bundle except through `v_trough_min_mV`. If that
        leaves no training step and n >= 1 (n = 1), every h_i is trained on.
        `n_drop_weakest` and `n_drop_strongest` are ignored;
        `train_all_hyp=True` raises ValueError. The depolarising steps are
        handled exactly as under D-006.

    n_drop_weakest / n_drop_strongest
        How many of the extreme hyperpolarising amplitudes are withheld from
        training. The D-006 defaults are 1 and 1, i.e. train on h_2..h_{n-1}.
    dep_n_validation
        Keep at most this many of the SMALLEST spike-free depolarising steps
        as validation (None = all of them). The loader has already excluded
        spiking sweeps and applied its own amplitude cap.
    v_trough_min_mV
        Optional floor on V_trough: a training step whose trough is BELOW
        this voltage is demoted to report-only. Off by default; it exists for
        a low-R_in cell whose h_{n-1} is itself very deep.
    train_all_hyp
        The labelled opt-in of D-007: train on every hyperpolarising step,
        h_1 and h_n included. Off by default.
    """
    hyp = sorted(list(getattr(cell_data, "long_square_subthreshold", [])),
                 key=lambda b: abs(float(b.amplitude_pA)))
    dep = sorted(list(getattr(cell_data, "long_square_depolarising", [])),
                 key=lambda b: abs(float(b.amplitude_pA)))
    if role_rule not in LS_ROLE_RULES:
        raise ValueError("role_rule must be one of %s, got %r"
                         % (LS_ROLE_RULES, role_rule))
    if role_rule == "d207" and train_all_hyp:
        raise ValueError("train_all_hyp=True is the D-007 opt-in of the d006 "
                         "rule; it cannot be combined with role_rule='d207'")
    n = len(hyp)
    records: List[Dict[str, object]] = []

    if role_rule == "d207":
        roles_1b = _d207_roles(n)
    else:
        if train_all_hyp:
            lo, hi = 0, n
        else:
            lo = min(int(n_drop_weakest), max(n - 1, 0))
            hi = max(n - int(n_drop_strongest), lo)
        # A cell with too few steps to drop any: train on what there is rather
        # than on nothing. Recorded in the reason string, never silent.
        if hi <= lo and n:
            lo, hi = 0, n

    train_ls: List = []
    valid_ls: List = []
    report_ls: List = []
    for i, b in enumerate(hyp):
        trough = _trough_mV(b)
        if role_rule == "d207":
            role, why = roles_1b[i]
        elif lo <= i < hi:
            role, why = "train", "h_%d of %d" % (i + 1, n)
        elif i < lo:
            role, why = "validate", "weakest step (held out)"
        else:
            role, why = "report", "strongest step (PIR conductances unmodelled)"
        if (role == "train" and v_trough_min_mV is not None
                and np.isfinite(trough) and trough < float(v_trough_min_mV)):
            role, why = "report", ("V_trough %.1f mV below floor %.1f"
                                   % (trough, float(v_trough_min_mV)))
        {"train": train_ls, "validate": valid_ls, "report": report_ls}[role].append(b)
        records.append({"amplitude_pA": float(b.amplitude_pA), "polarity": "hyp",
                        "role": role, "reason": why, "v_trough_mV": trough})

    dep_keep = dep if dep_n_validation is None else dep[:int(dep_n_validation)]
    for b in dep:
        keep = any(b is k for k in dep_keep)
        records.append({"amplitude_pA": float(b.amplitude_pA), "polarity": "dep",
                        "role": "validate" if keep else "unused",
                        "reason": "spike-free depolarising step"
                                  if keep else "beyond dep_n_validation",
                        "v_trough_mV": float("nan")})
    valid_ls.extend(dep_keep)

    if verbose:
        def _fmt(bs):
            return "[" + ", ".join("%+.0f" % float(b.amplitude_pA) for b in bs) + "]"
        print("[roles] LS train %s pA | validate %s pA | report-only %s pA"
              % (_fmt(train_ls), _fmt(valid_ls), _fmt(report_ls)))
        if not dep:
            print("[roles] WARNING: no depolarising validation steps for this "
                  "cell -- " + _dep_absence_reason(cell_data))
    return {"train": train_ls, "validate": valid_ls, "report": report_ls,
            "records": records, "n_hyp": n, "n_dep": len(dep)}


LS_ROLE_RULES = ("d006", "d207")


def _d207_roles(n: int) -> List[Tuple[str, str]]:
    """(role, reason) of h_1..h_n under SPEC 2.8 (b), 0-based list."""
    m = -(-int(n) // 2)                      # ceil(n / 2), 1-based
    out: List[Tuple[str, str]] = []
    for i in range(1, n + 1):
        if i == 1:
            out.append(("validate", "weakest step (held out)"))
        elif i == m:
            out.append(("validate", "middle step h_m (held out, d207)"))
        else:
            out.append(("train", "h_%d of %d (d207)" % (i, n)))
    # T_LS empty with n >= 1 (n = 1): train on every h_i, as D-006 does.
    if n and not any(r == "train" for r, _ in out):
        out = [("train", "h_%d of %d (d207)" % (i, n)) for i in range(1, n + 1)]
    return out


def _dep_absence_reason(cell_data) -> str:
    """Why a cell has no depolarising validation step, read from the loader's
    screen (``CellData.ls_dep_screen``, D-016) rather than guessed. An empty
    screen means the loader was not asked for depolarising sweeps."""
    screen = getattr(cell_data, "ls_dep_screen", None) or {}
    if not screen:
        return ("the loader was not asked for them "
                "(load_depolarising_ls=False)")
    n = int(screen.get("n_depolarising", 0))
    if n == 0:
        return "the archive holds no depolarising Long Square sweep"
    cap = screen.get("cap_pA")
    bits = []
    if screen.get("n_above_cap", 0):
        bits.append("%d above the %s pA cap" % (int(screen["n_above_cap"]),
                                              "%g" % cap if cap is not None else "?"))
    for key, label in (("n_action_potential", "with an action potential"),
                       ("n_peak_above_threshold", "peaking above the catch-all voltage"),
                       ("n_unusable", "unusable")):
        if screen.get(key, 0):
            bits.append("%d %s" % (int(screen[key]), label))
    return ("all %d depolarising sweep(s) were screened out: %s"
            % (n, ", ".join(bits) if bits else "reason not recorded"))


def _trough_mV(bundle, baseline_guard_s: float = 5e-3) -> float:
    """Minimum voltage during the step (duck-typed copy of the monolith's
    bundle_trough_mV, so this module stays importable without it)."""
    if _is_brief(bundle):
        return float("nan")
    t = np.asarray(bundle.t, dtype=float)
    v = np.asarray(bundle.v_mV, dtype=float)
    on = float(bundle.stim_onset_s) + float(baseline_guard_s)
    off = float(bundle.stim_onset_s) + float(bundle.stim_duration_s)
    m = (t >= on) & (t <= off)
    return float(np.min(v[m])) if m.any() else float("nan")


def split_train_validation_ih(cell_data, *, fit_target: str = "hyp",
                              n_drop_weakest: int = 1, n_drop_strongest: int = 1,
                              dep_n_validation: Optional[int] = 3,
                              v_trough_min_mV: Optional[float] = None,
                              train_all_hyp: bool = False,
                              verbose: bool = True,
                              role_rule: str = "d006"):
    """(train, validation, report, roles) for the I_h campaign.

    `role_rule` is passed to assign_ls_roles ("d006" default, or "d207").

    Brief SS pulses follow `fit_target` exactly as before; the long steps
    follow assign_ls_roles. This REPLACES split_train_validation_with_long_step
    for the I_h arms; that function is untouched and still serves the run-B
    reproduction (regression_passive_identity.py).
    """
    ss = list(cell_data.square_subthreshold)
    if fit_target == "hyp":
        train_ss = [b for b in ss if b.polarity == "hyp"]
        held_ss = [b for b in ss if b.polarity == "dep"]
    elif fit_target == "dep":
        train_ss = [b for b in ss if b.polarity == "dep"]
        held_ss = [b for b in ss if b.polarity == "hyp"]
    elif fit_target == "both":
        train_ss, held_ss = list(ss), []
    else:
        raise ValueError("fit_target must be 'hyp'|'dep'|'both', got %r" % fit_target)

    roles = assign_ls_roles(cell_data, n_drop_weakest=n_drop_weakest,
                            n_drop_strongest=n_drop_strongest,
                            dep_n_validation=dep_n_validation,
                            v_trough_min_mV=v_trough_min_mV,
                            train_all_hyp=train_all_hyp, verbose=verbose,
                            role_rule=role_rule)
    return (train_ss + list(roles["train"]),
            held_ss + list(roles["validate"]),
            list(roles["report"]),
            roles)


# ---------------------------------------------------------------------------
#  Decoupled integration into the monolith (no hand-editing of 7,500 lines)
# ---------------------------------------------------------------------------
def integrate_long_step(
    mono, *,
    spec: Optional[object] = None,
    ih_protocol: bool = False,
    ls_window_mode: str = "after_onset",
    n_drop_weakest: int = 1,
    n_drop_strongest: int = 1,
    dep_n_validation: Optional[int] = 3,
    v_trough_min_mV: Optional[float] = None,
    train_all_hyp: bool = False,
    n_long_train: int = 2,
    max_ls_train_deflection_mV: Optional[float] = 12.0,
    max_sag_amplitude_mV: Optional[float] = None,
    max_ls_train_amplitude_pA: Optional[float] = None,
    r_in_target: str = "peak",
    ls_window_ms_after_onset: float = 150.0,
    weighting: str = "relative",
    ss_window_ms: Tuple[float, float] = (0.5, 100.0),
    ss_time_weight: str = "exp",
    ss_tau_w_ms: float = 5.0,
    ss_t0_ms: Optional[float] = None,
    dt_brief_ms: Optional[float] = None,
    dt_long_ms: Optional[float] = None,
    verbose: bool = True,
    role_rule: str = "d006",
    ls_sample_weight: str = "none",
    ls_tau_d_ms: float = 500.0,
    n_workers: int = 1,
):
    """Monkeypatch the imported monolith module `mono` so Phase 2 trains on the
    brief hyperpolarising pulses PLUS the `n_long_train` smallest hyperpolarising
    long-square steps (default 2, to dampen I_h; see
    split_train_validation_with_long_step for the rationale), using the
    multi-protocol loss.

    Patches two monolith globals:
        mono.prepare_optimiser_inputs   (training/validation split)
        mono._build_loss_function       (loss used by fit_one_cell)

    `fit_one_cell` resolves `_build_loss_function` from the monolith's own
    globals, so the loss patch always takes effect. The split patch, however,
    must be in place BEFORE your runner copies names out of the monolith.

    Usage in the Colab/HPC runner (note the ordering):

        spec.loader.exec_module(mono)
        from passive_long_step_training import integrate_long_step
        integrate_long_step(mono, n_long_train=2, r_in_target="peak")
        # THEN copy names, exactly as before:
        globals().update({n: getattr(mono, n) for n in dir(mono)
                          if not n.startswith('_')})

    If you instead launch the monolith's own __main__, call this right after
    `import`-ing it and before `main()`.

    SPEC 2.8 options (Block 14; defaults keep the previous behaviour):
        role_rule         "d006" | "d207", passed to assign_ls_roles, and
                          only when ih_protocol=True.
        weighting, ls_sample_weight, ls_tau_d_ms
                          reach the TRAINING loss (_build_loss_function)
                          only. The monolith's validation RMSDs and every
                          diagnostic are not routed through them and stay
                          unweighted.
        n_workers         SPEC 2.9 (c) (Block 19): passed to the loss the
                          patched _build_loss_function builds (1: serial);
                          ValueError unless an integer >= 1.

    Returns the original `prepare_optimiser_inputs` so you can restore it.
    """
    # Refuse a bad option now, not inside the loss (whose exceptions become
    # the 1e6 penalty).
    if role_rule not in LS_ROLE_RULES:
        raise ValueError("role_rule must be one of %s, got %r"
                         % (LS_ROLE_RULES, role_rule))
    if role_rule == "d207" and train_all_hyp:
        raise ValueError("train_all_hyp=True cannot be combined with "
                         "role_rule='d207'")
    _check_ls_sample_weight(ls_sample_weight)
    if ls_sample_weight != "none" and not (float(ls_tau_d_ms) > 0):
        raise ValueError("ls_tau_d_ms must be > 0, got %r" % (ls_tau_d_ms,))
    n_workers = _check_n_workers(n_workers)
    OptimiserInputs = mono.OptimiserInputs
    PassiveSearchSpace = mono.PassiveSearchSpace
    orig_prepare = mono.prepare_optimiser_inputs

    # --- dt: one choice, applied to EVERY code path ------------------------
    # The training loss is not the only thing that simulates. fit_one_cell's
    # per-bundle VALIDATION replay, Phase 2.5's Ra profile and Phase 3's
    # bootstrap all go through the monolith's own _simulate_* helpers. Those
    # now resolve their dt from the monolith's module globals at call time, so
    # setting them here makes the dt choice global instead of leaving training
    # and validation free to disagree. This must happen BEFORE any simulation.
    global DEFAULT_DT_BRIEF_MS, DEFAULT_DT_LONG_MS
    if dt_brief_ms is not None:
        DEFAULT_DT_BRIEF_MS = float(dt_brief_ms)
    if dt_long_ms is not None:
        DEFAULT_DT_LONG_MS = float(dt_long_ms)
    mono.DEFAULT_DT_BRIEF_MS = float(DEFAULT_DT_BRIEF_MS)
    mono.DEFAULT_DT_LONG_MS = float(DEFAULT_DT_LONG_MS)

    def _prepare(cell_data, fit_target="hyp",
                 train_window_ms=None, n_long_validation=None):
        if train_window_ms is None:           # default to the [0.5,100] window
            train_window_ms = ss_window_ms
        report: List = []
        roles = None
        if ih_protocol:
            train, validation, report, roles = split_train_validation_ih(
                cell_data, fit_target=fit_target,
                n_drop_weakest=n_drop_weakest, n_drop_strongest=n_drop_strongest,
                dep_n_validation=dep_n_validation,
                v_trough_min_mV=v_trough_min_mV,
                train_all_hyp=train_all_hyp, verbose=verbose,
                role_rule=role_rule)
        else:
            train, validation = split_train_validation_with_long_step(
                cell_data, fit_target=fit_target, n_long_train=n_long_train,
                max_ls_train_deflection_mV=max_ls_train_deflection_mV,
                max_sag_amplitude_mV=max_sag_amplitude_mV,
                max_ls_train_amplitude_pA=max_ls_train_amplitude_pA,
                verbose=verbose)
        if n_long_validation is not None and n_long_validation >= 0:
            briefs = [b for b in validation
                      if float(b.stim_duration_s) < _BRIEF_PULSE_MAX_DURATION_S]
            longs = [b for b in validation
                     if float(b.stim_duration_s) >= _BRIEF_PULSE_MAX_DURATION_S]
            validation = briefs + longs[:int(n_long_validation)]
        space = PassiveSearchSpace()
        dims = (spec.as_skopt_dimensions() if spec is not None
                else space.as_skopt_dimensions())
        oi = OptimiserInputs(
            search_space=space,
            skopt_dimensions=dims,
            train_bundles=train,
            train_window_ms=tuple(train_window_ms),
            validation_bundles=validation,
            rin_MOhm=cell_data.rin_MOhm,
            tau_ms=cell_data.tau_ms,
            v_rest_mV=cell_data.v_rest_mV,
        )
        # Attached rather than passed positionally so an OptimiserInputs from
        # an older monolith (no param_spec field) still works.
        try:
            oi.param_spec = spec
        except Exception:                      # pragma: no cover
            pass
        oi.report_bundles = report
        oi.ls_roles = roles
        return oi

    def _make_ss_weight_fn(win_ms):
        """Build the per-sample SS time-weight; t0 defaults to the window start
        (the pulse offset). Returns None when disabled."""
        if ss_time_weight == "none":
            return None
        t0_s = (ss_t0_ms if ss_t0_ms is not None else win_ms[0]) * 1e-3
        scale_s = ss_tau_w_ms * 1e-3
        if ss_time_weight == "exp":
            return exp_time_weight(t0_s, scale_s)
        if ss_time_weight == "gauss":
            return gauss_time_weight(t0_s, scale_s)
        raise ValueError("ss_time_weight must be 'exp'|'gauss'|'none', "
                         f"got {ss_time_weight!r}")

    def _build_loss(cell, train_bundles, v_rest_mV, train_window_ms,
                    pre_window_ms=(-10.0, 0.0), spec=None):
        # `spec` arrives from fit_one_cell (Stage 3). A spec passed there wins
        # over the one integrate_long_step was configured with, so a single
        # patched monolith can fit a 3-D control arm and a 6-D I_h arm in the
        # same process.
        return build_multi_protocol_loss(
            cell, train_bundles, v_rest_mV,
            ss_window_ms=tuple(train_window_ms),
            ls_window_ms_after_onset=ls_window_ms_after_onset,
            ls_window_mode=ls_window_mode,
            r_in_target=r_in_target, weighting=weighting,
            ss_sample_weight_fn=_make_ss_weight_fn(tuple(train_window_ms)),
            dt_brief_ms=DEFAULT_DT_BRIEF_MS, dt_long_ms=DEFAULT_DT_LONG_MS,
            spec=(spec if spec is not None else globals().get("_INTEGRATE_SPEC")),
            ls_sample_weight=ls_sample_weight, ls_tau_d_ms=ls_tau_d_ms,
            n_workers=n_workers)

    global _INTEGRATE_SPEC
    _INTEGRATE_SPEC = spec
    mono.prepare_optimiser_inputs = _prepare
    mono._build_loss_function = _build_loss
    if verbose:
        if ih_protocol:
            # n_long_train does not apply: assign_ls_roles decides per cell.
            ntrain = (f"brief + long steps by role (D-006: drop "
                      f"{n_drop_weakest} weakest / {n_drop_strongest} "
                      f"strongest)" if not train_all_hyp
                      else "brief + EVERY hyperpolarising long step")
            if role_rule == "d207":
                ntrain = "brief + long steps by role (SPEC 2.8 (b), d207)"
        else:
            ntrain = "SS-only" if int(n_long_train) == 0 \
                else f"brief + {n_long_train} long step(s)"
        wtag = (f"{ss_time_weight}(tau_w={ss_tau_w_ms} ms)"
                if ss_time_weight != "none" else "uniform")
        print(f"[integrate_long_step] training = {ntrain}; SS window "
              f"{tuple(ss_window_ms)} ms, SS time-weight = {wtag}; "
              f"r_in_target={r_in_target!r}, ls_window={ls_window_ms_after_onset} "
              f"ms, cross-bundle weighting={weighting!r}")
        if (role_rule != "d006" or ls_sample_weight != "none"
                or weighting == "ls_amplitude"):
            print(f"[integrate_long_step] SPEC 2.8 protocol: role_rule="
                  f"{role_rule!r}"
                  f"{'' if ih_protocol else ' (ignored: ih_protocol=False)'}, "
                  f"ls_sample_weight={ls_sample_weight!r}, ls_tau_d_ms="
                  f"{float(ls_tau_d_ms):g} (training loss only)")
        print(f"[integrate_long_step] dt ENFORCED: brief="
              f"{DEFAULT_DT_BRIEF_MS:g} ms, long={DEFAULT_DT_LONG_MS:g} ms "
              f"(applied to training, validation, Phase 2.5 and Phase 3)")
    return orig_prepare




# ---------------------------------------------------------------------------
#  SPEC 2.10 (a), Block 23: the residual vector and the trace evaluator
# ---------------------------------------------------------------------------
#  r_{b,s}(q) = sqrt(a_{b,s}) d_{b,s}(q) / D_b,
#  a_{b,s} = c_b w_b(t_s) / sum_{s' in W_b} w_b(t_s'),
#  d_{b,s}(q) = Vhat_b(t_s; q) - V_b(t_s), both minus their pre-window means,
#  so that sum_s r_{b,s}^2 = c_b (RMSD_b / D_b)^2 with RMSD_b of bundle_rmsd.
#  The model side (bundle_trace) and the data side (residual_layout) are
#  separate: the layout depends on the recorded traces only and is built
#  once; the traces are what an evaluation costs.
#  Custom: the assembly of SPEC 2.10 (a) is the project's own loss; no library
#  function computes it. Checked: none applies (np.interp does the resampling).

def _bundle_windows_s(bundle, ss_window_ms, ls_window_mode,
                      ls_window_ms_after_onset):
    """(pre-window, RMSD window) in s of one bundle, exactly as bundle_rmsd
    chooses them."""
    if _is_brief(bundle):
        return ((-10e-3, 0.0),
                (ss_window_ms[0] * 1e-3, ss_window_ms[1] * 1e-3))
    return ((0.0, float(bundle.stim_onset_s)),
            ls_rmsd_window_s(bundle, ls_window_mode, ls_window_ms_after_onset))


def bundle_trace(cell, bundle, v_rest_mV: float, *,
                 dt_brief_ms: Optional[float] = None,
                 dt_long_ms: Optional[float] = None) -> np.ndarray:
    """float64 (len(bundle.t),), mV: the model's trace of `bundle` at the
    cell's CURRENT parameters minus its own pre-window mean (taken on the
    simulation grid), interpolated on bundle.t (np.interp: constant outside
    the simulated interval). SPEC Block 23.

    Simulated as bundle_rmsd simulates it (_simulate_brief / _simulate_long,
    the same dt resolution); pre-window (-10 ms, 0) for a brief bundle,
    (0, onset) for a long one, ends included as in _baseline_subtracted_rmsd.
    The subtraction is made on the simulation grid before the interpolation,
    as _baseline_subtracted_rmsd makes it, so the window samples equal the
    ones the RMSD uses. A pre-window with no simulated sample gives an
    all-NaN trace (the RMSD is inf there); the caller judges finiteness.
    """
    if _is_brief(bundle):
        t_s, v = _simulate_brief(cell, bundle, v_rest_mV, dt_ms=dt_brief_ms)
        pre_w = (-10e-3, 0.0)
    else:
        t_s, v = _simulate_long(cell, bundle, v_rest_mV, dt_ms=dt_long_ms)
        pre_w = (0.0, float(bundle.stim_onset_s))
    t_s = np.asarray(t_s, dtype=float)
    v = np.asarray(v, dtype=float)
    pm = (t_s >= pre_w[0]) & (t_s <= pre_w[1])
    bs = float(np.mean(v[pm])) if pm.any() else np.nan
    t_exp = np.asarray(bundle.t, dtype=float)
    if not np.isfinite(bs):
        return np.full(t_exp.shape, np.nan, dtype=np.float64)
    return np.asarray(np.interp(t_exp, t_s, v - bs), dtype=np.float64)


@dataclass(frozen=True)
class ResidualLayout:
    """The data-only part of the residual vector of SPEC 2.10 (a), one entry
    per training bundle in the training order.

    masks      list of bool (n_t_b,): the samples of W_b (the RMSD window)
    pre_masks  list of bool (n_t_b,): the pre-window samples
    sqrt_a     list of float64 (|W_b|,): sqrt(a_{b,s})
    v_exp      list of float64 (|W_b|,): recorded samples of W_b minus the
               recorded pre-window mean, mV
    D          float64 (n_bundles,): D_b, mV
    c          float64 (n_bundles,): c_b, sums to 1
    slices     list of slice: bundle b's entries of the residual vector
    n_residuals  int N = sum_b |W_b|
    """
    masks: list
    pre_masks: list
    sqrt_a: list
    v_exp: list
    D: np.ndarray
    c: np.ndarray
    slices: list
    n_residuals: int


def _bundle_weights(train_bundles, weighting: str) -> np.ndarray:
    """c_b of SPEC 2.10 (a): the loss's cross-bundle weight divided by the
    number of bundles, so that L = sum_b c_b RMSD_b / D_b is the loss of
    build_multi_protocol_loss with the same arithmetic for the weights."""
    n_all = len(train_bundles)
    if weighting == "relative":
        return np.full(n_all, 1.0 / n_all, dtype=np.float64)
    # ls_amplitude: as build_multi_protocol_loss builds w, then w / n
    is_ls = np.array([not _is_brief(b) for b in train_bundles], dtype=bool)
    n_ls = int(is_ls.sum())
    amps = np.array([abs(float(b.amplitude_pA)) for b in train_bundles],
                    dtype=float)
    tot = float(np.sum(amps[is_ls]))
    if n_ls and not (np.isfinite(tot) and tot > 0):
        raise ValueError("weighting='ls_amplitude' needs a positive total "
                         "|amplitude| over the training long squares, got %r"
                         % tot)
    w = np.where(is_ls, n_ls * amps / (tot if n_ls else 1.0), 1.0)
    return np.asarray(w / n_all, dtype=np.float64)


def residual_layout(train_bundles: Sequence, *,
                    ss_window_ms: Tuple[float, float] = (0.5, 100.0),
                    ls_window_ms_after_onset: float = 150.0,
                    ls_window_mode: str = "after_onset",
                    r_in_target: str = "peak",
                    weighting: str = "relative",
                    ss_sample_weight_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                    ls_sample_weight: str = "none",
                    ls_tau_d_ms: float = 500.0) -> ResidualLayout:
    """ResidualLayout of SPEC 2.10 (a): masks, pre_masks, sqrt_a, v_exp, D,
    c, slices, n_residuals -- from the recorded traces only (no cell).
    SPEC Block 23.

    Windows, sample weights and D_b are those of bundle_rmsd for the same
    arguments (D_b = _peak_deflection: r_in_target "steady" takes the last
    20 ms of the window, anything else the whole window, as bundle_rmsd).
    ValueError: weighting not 'relative' or 'ls_amplitude'; no bundle; a
    bundle with no pre-window sample, fewer than 2 window samples, a
    non-finite recorded sample in either, sample weights non-finite or of
    non-positive sum, or D_b not finite and positive; and the refusals of
    build_multi_protocol_loss (ls_sample_weight, a long square peak_decay
    cannot serve, ls_amplitude with no positive long-square amplitude).
    """
    if weighting not in ("relative", "ls_amplitude"):
        raise ValueError("residual_layout: weighting must be 'relative' or "
                         "'ls_amplitude', got %r" % (weighting,))
    _check_ls_sample_weight(ls_sample_weight)
    bundles = list(train_bundles)
    if not bundles:
        raise ValueError("residual_layout: no training bundle")
    c = _bundle_weights(bundles, weighting)
    masks, pre_masks, sqrt_a, v_exp, slices = [], [], [], [], []
    D = np.empty(len(bundles), dtype=np.float64)
    start = 0
    for i, b in enumerate(bundles):
        pre_w, win = _bundle_windows_s(b, ss_window_ms, ls_window_mode,
                                       ls_window_ms_after_onset)
        if _is_brief(b):
            weight_fn = ss_sample_weight_fn
        else:
            weight_fn = _ls_weight_fn(b, ls_sample_weight, ls_tau_d_ms)
        t = np.asarray(b.t, dtype=float)
        v = np.asarray(b.v_mV, dtype=float)
        pm = (t >= pre_w[0]) & (t <= pre_w[1])
        wm = (t >= win[0]) & (t <= win[1])
        tag = "residual_layout: bundle %d (%g pA)" % (i, float(b.amplitude_pA))
        if not pm.any():
            raise ValueError("%s has no pre-window sample in [%g, %g] s"
                             % (tag, pre_w[0], pre_w[1]))
        if int(wm.sum()) < 2:
            raise ValueError("%s has %d sample(s) in its window [%g, %g] s; "
                             "at least 2 are needed"
                             % (tag, int(wm.sum()), win[0], win[1]))
        if not (np.all(np.isfinite(v[pm])) and np.all(np.isfinite(v[wm]))):
            raise ValueError("%s has a non-finite recorded sample in its "
                             "pre-window or window" % tag)
        tg = t[wm]
        if weight_fn is None:
            w = np.ones(tg.shape, dtype=float)
        else:
            w = np.asarray(weight_fn(tg), dtype=float)
        sw = float(np.sum(w))
        if w.shape != tg.shape or not np.all(np.isfinite(w)) or not (sw > 0):
            raise ValueError("%s: sample weights non-finite, of the wrong shape "
                             "or of non-positive sum (%r)" % (tag, sw))
        if r_in_target == "steady":
            d_b = _peak_deflection(b, pre_w, (win[1] - 20e-3, win[1]))
        else:
            d_b = _peak_deflection(b, pre_w, win)
        if not (np.isfinite(d_b) and d_b > 0):
            raise ValueError("%s: deflection D_b = %r mV is not finite and "
                             "positive" % (tag, d_b))
        be = float(np.mean(v[pm]))
        masks.append(wm)
        pre_masks.append(pm)
        sqrt_a.append(np.sqrt(c[i] * w / sw).astype(np.float64))
        v_exp.append(np.asarray((v - be)[wm], dtype=np.float64))
        D[i] = float(d_b)
        n_b = int(wm.sum())
        slices.append(slice(start, start + n_b))
        start += n_b
    return ResidualLayout(masks=masks, pre_masks=pre_masks, sqrt_a=sqrt_a,
                          v_exp=v_exp, D=D, c=c, slices=slices,
                          n_residuals=int(start))


def assemble_residuals(layout, traces: Sequence[np.ndarray]) -> np.ndarray:
    """float64 (N,): sqrt_a[b] * (traces[b][masks[b]] - v_exp[b]) / D[b],
    bundles in order. SPEC Block 23."""
    if len(traces) != len(layout.masks):
        raise ValueError("assemble_residuals: %d traces for %d bundles"
                         % (len(traces), len(layout.masks)))
    r = np.empty(int(layout.n_residuals), dtype=np.float64)
    for b, tr in enumerate(traces):
        tr = np.asarray(tr, dtype=np.float64)
        if tr.shape != layout.masks[b].shape:
            raise ValueError("assemble_residuals: trace %d has shape %s, "
                             "expected %s" % (b, tr.shape, layout.masks[b].shape))
        r[layout.slices[b]] = (layout.sqrt_a[b]
                               * (tr[layout.masks[b]] - layout.v_exp[b])
                               / layout.D[b])
    return r


def l_and_l2(layout, r: np.ndarray) -> Tuple[float, float]:
    """(L, L_2) of SPEC 2.10 (a) from a residual vector (N,):
    L_2 = sum r^2;  L = sum_b c_b [sum_{s in W_b} r_{b,s}^2 / c_b]^(1/2)
    (= sum_b c_b RMSD_b / D_b, the loss of 2.8). A bundle with c_b = 0
    contributes 0 to L (its residuals are 0)."""
    r = np.asarray(r, dtype=np.float64)
    if r.shape != (int(layout.n_residuals),):
        raise ValueError("l_and_l2: r has shape %s, expected (%d,)"
                         % (r.shape, int(layout.n_residuals)))
    l2 = float(np.sum(r * r))
    terms = []
    for b, sl in enumerate(layout.slices):
        cb = float(layout.c[b])
        if cb > 0:
            rb = r[sl]
            terms.append(cb * np.sqrt(float(np.sum(rb * rb)) / cb))
        else:
            terms.append(0.0)
    return float(np.sum(np.asarray(terms, dtype=np.float64))), l2


# ------------------------------------------------------- the evaluator ---
def _spec_dim(spec) -> Optional[int]:
    """p of a ParamSpec (len); None for a duck-typed spec without one."""
    try:
        return int(len(spec))
    except TypeError:
        names = getattr(spec, "names", None)
        return None if names is None else len(tuple(names))


def _trace_task(token: int, q: tuple, i: int, dt_brief_ms: float,
                dt_long_ms: float) -> tuple:
    """Worker side of the evaluator: apply q to this process's copy of the
    cell, return bundle i's trace as the serial loop computes it, tagged:
        ("ok", trace) | ("dt", pickled exception or None, text)
        | ("err", class name, text)
    """
    try:
        cell, spec, bundles, v_rest_mV = _POOL_CONTEXTS[token]
        spec.apply_q(cell, q, v_rest_mV)
        tr = bundle_trace(cell, bundles[i], v_rest_mV,
                          dt_brief_ms=dt_brief_ms, dt_long_ms=dt_long_ms)
        return ("ok", tr)
    except Exception as exc:  # noqa: BLE001 - sent back, judged by the parent
        text = "".join(traceback.format_exception(type(exc), exc,
                                                  exc.__traceback__))
        if type(exc).__name__ == "DtEnforcementError":
            try:
                blob = pickle.dumps(exc)
            except Exception:  # noqa: BLE001 - fall back to the named class
                blob = None
            return ("dt", blob, text)
        return ("err", type(exc).__name__, text)


class _TraceEvaluator:
    """ev(Q) of SPEC Block 23 (build_trace_evaluator)."""

    def __init__(self, cell, bundles, v_rest_mV, spec, dt_brief_ms,
                 dt_long_ms, n_workers):
        self._cell = cell
        self._bundles = tuple(bundles)
        self._v_rest = v_rest_mV
        self._spec = spec
        self._p = _spec_dim(spec)
        self._dtb = dt_brief_ms
        self._dtl = dt_long_ms
        self.n_workers = int(n_workers)
        self.n_points = 0
        self.n_failed = 0
        self._state = {"pool": None, "token": None}
        # terminated when the evaluator is garbage-collected, and at exit
        weakref.finalize(self, _close_pool, self._state)

    def close(self) -> None:
        """Terminate the pool if one was made (idempotent; a later call makes
        a new one). A no-op for n_workers == 1."""
        _close_pool(self._state)

    def _check_Q(self, Q) -> np.ndarray:
        Q = np.asarray(Q, dtype=np.float64)
        if Q.ndim != 2 or (self._p is not None and Q.shape[1] != self._p):
            raise ValueError("ev: Q must be 2-D with %s columns, got shape %s"
                             % (self._p, Q.shape))
        if not np.all(np.isfinite(Q)):
            raise ValueError("ev: Q holds a non-finite value")
        return Q

    def _pool(self):
        state = self._state
        if state["pool"] is None:
            # One live pool per process (Block 19, amendment 2): close every
            # other loss's or evaluator's pool before this one forks.
            for other in list(_LIVE_POOLS.values()):
                if other is not state:
                    _close_pool(other)
            token = next(_POOL_TOKENS)
            _POOL_CONTEXTS[token] = (self._cell, self._spec, self._bundles,
                                     self._v_rest)
            state["token"] = token
            state["pool"] = ProcessPoolExecutor(
                max_workers=self.n_workers,
                mp_context=multiprocessing.get_context("fork"))
            _LIVE_POOLS[id(state)] = state
        return state["pool"], state["token"]

    def _serial_point(self, q: tuple):
        """One point in the process: the list of traces, or None."""
        try:
            self._spec.apply_q(self._cell, q, self._v_rest)
            out = []
            for b in self._bundles:
                tr = bundle_trace(self._cell, b, self._v_rest,
                                  dt_brief_ms=self._dtb, dt_long_ms=self._dtl)
                if not np.all(np.isfinite(tr)):
                    return None
                out.append(tr)
            return out
        except Exception as exc:  # noqa: BLE001 - a failed point, judged here
            if type(exc).__name__ == "DtEnforcementError":
                raise
            return None

    def _pooled(self, qs):
        dtb = float(DEFAULT_DT_BRIEF_MS if self._dtb is None else self._dtb)
        dtl = float(DEFAULT_DT_LONG_MS if self._dtl is None else self._dtl)
        n_b = len(self._bundles)
        pool, token = self._pool()
        n_t = len(qs) * n_b
        q_args = [qs[k // n_b] for k in range(n_t)]
        i_args = [k % n_b for k in range(n_t)]
        try:
            outs = list(pool.map(_trace_task, [token] * n_t, q_args, i_args,
                                 [dtb] * n_t, [dtl] * n_t, chunksize=1))
        except BrokenProcessPool:
            # A worker died: no value for this call, never a failed point.
            _close_pool(self._state)
            raise
        # The serial code's side effect: the parent's cell is left at Q[-1].
        # A DtEnforcementError here is also one the workers met at the same
        # point (same code); it is raised only if the outcomes below do not
        # raise first, as the serial loop would order them.
        parent_dt = None
        try:
            self._spec.apply_q(self._cell, qs[-1], self._v_rest)
        except Exception as exc:  # noqa: BLE001 - judged as the serial loop
            if type(exc).__name__ == "DtEnforcementError":
                parent_dt = exc
        # The serial loop's outcome, point by point, bundle by bundle.
        res = []
        for k in range(len(qs)):
            traces = []
            for out in outs[k * n_b:(k + 1) * n_b]:
                if out[0] == "dt":
                    _raise_remote_dt(out[1], out[2])
                if out[0] == "err" or not np.all(np.isfinite(out[1])):
                    traces = None
                    break
                traces.append(out[1])
            res.append(traces)
        if parent_dt is not None:
            raise parent_dt
        return res

    def __call__(self, Q) -> list:
        Q = self._check_Q(Q)
        qs = [tuple(float(x) for x in row) for row in Q]
        self.n_points += len(qs)
        if self.n_workers == 1 or not self._bundles or not qs:
            res = [self._serial_point(q) for q in qs]
        else:
            res = self._pooled(qs)
        self.n_failed += sum(1 for x in res if x is None)
        return res


def build_trace_evaluator(cell, bundles: Sequence, v_rest_mV: float, *,
                          spec: Optional[object] = None,
                          dt_brief_ms: Optional[float] = None,
                          dt_long_ms: Optional[float] = None,
                          n_workers: int = 1):
    """ev(Q (m, p) in q) -> list of m entries: the list of bundle_trace
    arrays (bundles in order) at Q[k], or None when spec.apply_q or a
    bundle's simulation raised (a DtEnforcementError is raised in the caller
    under that class name) or a trace is non-finite. SPEC Block 23.

    n_workers == 1: a serial loop in the process. n_workers > 1: the
    m * n_bundles (point, bundle) tasks on a fork ProcessPoolExecutor of
    n_workers processes, created at the first call and reused (chunksize=1);
    the values are exactly the serial loop's; a worker that dies raises
    BrokenProcessPool; one live pool per process, shared with the losses of
    build_multi_protocol_loss (Block 19, amendment 2). After a call the
    parent's cell is at Q[-1]. dt None resolves DEFAULT_DT_* at call time.
    Attributes: n_workers, n_points (points asked since built), n_failed
    (points returned None), close(). ValueError: n_workers not an integer
    >= 1; Q not 2-D with p columns or non-finite.
    """
    n_workers = _check_n_workers(n_workers)
    if spec is None:
        import param_spec as _ps
        spec = _ps.PASSIVE_3D
    return _TraceEvaluator(cell, bundles, v_rest_mV, spec, dt_brief_ms,
                           dt_long_ms, n_workers)


class _MultiProtocolResiduals:
    """resid(Q) of SPEC Block 23 (build_multi_protocol_residuals)."""

    def __init__(self, layout, evaluator, fail_l2):
        self.layout = layout
        self.evaluator = evaluator
        self.n_residuals = int(layout.n_residuals)
        self.fail_l2 = float(fail_l2)
        self._fail_value = float(np.sqrt(self.fail_l2 / self.n_residuals))

    @property
    def n_points(self) -> int:
        return self.evaluator.n_points

    @property
    def n_failed(self) -> int:
        return self.evaluator.n_failed

    def l_and_l2(self, row) -> Tuple[float, float]:
        return l_and_l2(self.layout, row)

    def close(self) -> None:
        self.evaluator.close()

    def __call__(self, Q) -> np.ndarray:
        traces = self.evaluator(Q)
        out = np.empty((len(traces), self.n_residuals), dtype=np.float64)
        for k, tr in enumerate(traces):
            if tr is None:
                out[k] = self._fail_value
            else:
                out[k] = assemble_residuals(self.layout, tr)
        return out


def build_multi_protocol_residuals(
        cell, train_bundles: Sequence, v_rest_mV: float, *,
        ss_window_ms: Tuple[float, float] = (0.5, 100.0),
        ls_window_ms_after_onset: float = 150.0,
        ls_window_mode: str = "after_onset",
        r_in_target: str = "peak",
        weighting: str = "relative",
        ss_sample_weight_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        ls_sample_weight: str = "none",
        ls_tau_d_ms: float = 500.0,
        spec: Optional[object] = None,
        dt_brief_ms: Optional[float] = None,
        dt_long_ms: Optional[float] = None,
        n_workers: int = 1,
        fail_l2: float = 100.0):
    """resid(Q (m, p) in q) -> float64 (m, N), row k = r(Q[k]) of SPEC
    2.10 (a); a failed point the constant row sqrt(fail_l2 / N), so its
    L_2 is fail_l2. Attributes layout, evaluator, n_residuals, n_points,
    n_failed, l_and_l2(row), close(). SPEC Block 23.

    ValueError: the refusals of residual_layout and build_trace_evaluator;
    fail_l2 not finite and positive.
    """
    if not (np.isfinite(float(fail_l2)) and float(fail_l2) > 0):
        raise ValueError("fail_l2 must be finite and > 0, got %r" % (fail_l2,))
    n_workers = _check_n_workers(n_workers)
    layout = residual_layout(
        train_bundles, ss_window_ms=ss_window_ms,
        ls_window_ms_after_onset=ls_window_ms_after_onset,
        ls_window_mode=ls_window_mode, r_in_target=r_in_target,
        weighting=weighting, ss_sample_weight_fn=ss_sample_weight_fn,
        ls_sample_weight=ls_sample_weight, ls_tau_d_ms=ls_tau_d_ms)
    ev = build_trace_evaluator(cell, train_bundles, v_rest_mV, spec=spec,
                               dt_brief_ms=dt_brief_ms, dt_long_ms=dt_long_ms,
                               n_workers=n_workers)
    return _MultiProtocolResiduals(layout, ev, fail_l2)


def _residuals_factory(cell, train_bundles, v_rest_mV, *, weighting, spec,
                       dt_brief_ms, dt_long_ms, n_workers, protocol_kw):
    """loss.make_residuals of SPEC Block 23: build_multi_protocol_residuals
    with the loss's own cell, bundles, protocol arguments, spec, dt and
    n_workers. Holds no reference to the loss itself."""
    def make_residuals(fail_l2: float = 100.0):
        if weighting not in ("relative", "ls_amplitude"):
            raise ValueError("make_residuals: the residual vector is defined "
                             "for weighting 'relative' or 'ls_amplitude', "
                             "this loss has %r" % (weighting,))
        return build_multi_protocol_residuals(
            cell, list(train_bundles), v_rest_mV, weighting=weighting,
            spec=spec,
            dt_brief_ms=dt_brief_ms, dt_long_ms=dt_long_ms,
            n_workers=n_workers, fail_l2=fail_l2, **protocol_kw)
    return make_residuals
