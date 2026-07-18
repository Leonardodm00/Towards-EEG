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

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Brief vs long discriminator (matches the monolith's _BRIEF_PULSE_MAX_DURATION_S)
_BRIEF_PULSE_MAX_DURATION_S = 0.01


# ---------------------------------------------------------------------------
#  Bundle helpers (duck-typed; SweepBundle satisfies the attribute contract)
# ---------------------------------------------------------------------------
def _is_brief(bundle) -> bool:
    return float(bundle.stim_duration_s) < _BRIEF_PULSE_MAX_DURATION_S


def _simulate_brief(cell, b, v_rest_mV: float,
                    pre_pad_ms: float = 10.0, post_pad_ms: float = 100.0,
                    dt_ms: float = 0.025) -> Tuple[np.ndarray, np.ndarray]:
    """Replay a brief (Square-Subthreshold) pulse; t=0 at pulse onset."""
    dur_ms = float(b.stim_duration_s) * 1e3
    tstop_ms = pre_pad_ms + dur_ms + post_pad_ms
    t_ms, v = cell.simulate(
        stim_amp_pA=float(b.amplitude_pA), stim_delay_ms=pre_pad_ms,
        stim_dur_ms=dur_ms, tstop_ms=tstop_ms,
        v_init_mV=v_rest_mV, dt_ms=dt_ms)
    return np.asarray(t_ms) * 1e-3 - pre_pad_ms * 1e-3, np.asarray(v)


def _simulate_long(cell, b, v_rest_mV: float,
                   dt_ms: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """Replay a long step using the bundle's own onset/duration; t in s on the
    simulation grid (interpolated to the experimental grid by the RMSD step)."""
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
    Implied noise sigma^2(t) ∝ exp(+(t - t0)/tau_w): variance e-folds in tau_w,
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
    Implied noise sigma^2(t) ∝ exp(+(t - t0)^2 / (2 sigma_w^2)) grows SUPER-
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
def bundle_rmsd(cell, bundle, v_rest_mV: float, *,
                ss_window_ms: Tuple[float, float] = (0.5, 100.0),
                ls_window_ms_after_onset: float = 150.0,
                r_in_target: str = "peak",
                ss_sample_weight_fn: Optional[Callable[[np.ndarray], np.ndarray]]
                = None) -> Tuple[float, float]:
    """Return (rmsd_mV, deflection_mV) for one bundle.

    `ss_window_ms` defaults to (0.5, 100.0): the start is the 0.5 ms pulse
    OFFSET (not 1.0 ms), so the Cm-bearing early decay just after the peak is
    INSIDE the window. `ss_sample_weight_fn`, if given, applies the per-sample
    time-weight to BRIEF bundles only; long bundles always keep the uniform
    RMSD (the time-weight is SS-only by design).

    `deflection_mV` is the |peak| (r_in_target='peak') or |tail-plateau|
    (r_in_target='steady') experimental deflection in the fit window, used
    only for optional relative weighting; it does not change the RMSD itself.
    """
    if _is_brief(bundle):
        t_s, v = _simulate_brief(cell, bundle, v_rest_mV)
        pre_w = (-10e-3, 0.0)
        rmsd_w = (ss_window_ms[0] * 1e-3, ss_window_ms[1] * 1e-3)
        weight_fn = ss_sample_weight_fn
    else:
        t_s, v = _simulate_long(cell, bundle, v_rest_mV)
        onset = float(bundle.stim_onset_s)
        pre_w = (0.0, onset)
        rmsd_w = (onset, onset + ls_window_ms_after_onset * 1e-3)
        weight_fn = None                      # time-weight is SS-only by design
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
#  Multi-protocol loss (log-space, drop-in for _build_loss_function)
# ---------------------------------------------------------------------------
def build_multi_protocol_loss(
    cell, train_bundles: Sequence, v_rest_mV: float, *,
    ss_window_ms: Tuple[float, float] = (0.5, 100.0),
    ls_window_ms_after_onset: float = 150.0,
    r_in_target: str = "peak",
    weighting: str = "relative",
    weights: Optional[Sequence[float]] = None,
    ss_sample_weight_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Callable[[float, float, float], float]:
    """Return loss(cm_log, rm_log, ra_log) -> scalar.

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
    """
    if weighting == "custom":
        if weights is None or len(weights) != len(train_bundles):
            raise ValueError("weighting='custom' needs weights of matching length")
        w = np.asarray(weights, dtype=float)

    def loss(cm_log: float, rm_log: float, ra_log: float) -> float:
        try:
            cell.set_passive(Cm=float(np.exp(cm_log)),
                             Rm=float(np.exp(rm_log)),
                             Ra=float(np.exp(ra_log)))
            cell.set_e_pas(v_rest_mV)
            terms = []
            for b in train_bundles:
                rmsd, defl = bundle_rmsd(
                    cell, b, v_rest_mV, ss_window_ms=ss_window_ms,
                    ls_window_ms_after_onset=ls_window_ms_after_onset,
                    r_in_target=r_in_target,
                    ss_sample_weight_fn=ss_sample_weight_fn)
                if not np.isfinite(rmsd):
                    return 1e6
                if weighting == "relative":
                    terms.append(rmsd / max(abs(defl), 1e-6))
                else:
                    terms.append(rmsd)
            terms = np.asarray(terms, dtype=float)
            if weighting == "custom":
                return float(np.sum(w * terms) / np.sum(w))
            return float(np.mean(terms))
        except Exception:  # noqa: BLE001 - mirror monolith: penalise, never crash
            return 1e6

    return loss


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
#  Decoupled integration into the monolith (no hand-editing of 7,500 lines)
# ---------------------------------------------------------------------------
def integrate_long_step(
    mono, *,
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
    verbose: bool = True,
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

    Returns the original `prepare_optimiser_inputs` so you can restore it.
    """
    OptimiserInputs = mono.OptimiserInputs
    PassiveSearchSpace = mono.PassiveSearchSpace
    orig_prepare = mono.prepare_optimiser_inputs

    def _prepare(cell_data, fit_target="hyp",
                 train_window_ms=None, n_long_validation=None):
        if train_window_ms is None:           # default to the [0.5,100] window
            train_window_ms = ss_window_ms
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
        return OptimiserInputs(
            search_space=space,
            skopt_dimensions=space.as_skopt_dimensions(),
            train_bundles=train,
            train_window_ms=tuple(train_window_ms),
            validation_bundles=validation,
            rin_MOhm=cell_data.rin_MOhm,
            tau_ms=cell_data.tau_ms,
            v_rest_mV=cell_data.v_rest_mV,
        )

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
                    pre_window_ms=(-10.0, 0.0)):
        return build_multi_protocol_loss(
            cell, train_bundles, v_rest_mV,
            ss_window_ms=tuple(train_window_ms),
            ls_window_ms_after_onset=ls_window_ms_after_onset,
            r_in_target=r_in_target, weighting=weighting,
            ss_sample_weight_fn=_make_ss_weight_fn(tuple(train_window_ms)))

    mono.prepare_optimiser_inputs = _prepare
    mono._build_loss_function = _build_loss
    if verbose:
        ntrain = "SS-only" if int(n_long_train) == 0 \
            else f"brief + {n_long_train} long step(s)"
        wtag = (f"{ss_time_weight}(tau_w={ss_tau_w_ms} ms)"
                if ss_time_weight != "none" else "uniform")
        print(f"[integrate_long_step] training = {ntrain}; SS window "
              f"{tuple(ss_window_ms)} ms, SS time-weight = {wtag}; "
              f"r_in_target={r_in_target!r}, ls_window={ls_window_ms_after_onset} "
              f"ms, cross-bundle weighting={weighting!r}")
    return orig_prepare
