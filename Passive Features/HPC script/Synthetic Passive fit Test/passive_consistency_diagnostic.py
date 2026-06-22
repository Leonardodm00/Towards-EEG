# -*- coding: utf-8 -*-
"""
passive_consistency_diagnostic.py
=================================

A *read-only* diagnostic instrument for the passive-fitting pipeline
(`passive_fitting_hpc_fixed.py`).  It does **not** change any fit; it only
*measures* quantities the brief-pulse loss never targets, so you can see why
the fitted product  Rm * Cm  disagrees with the Allen membrane time constant
tau_m, and why the long-square validation diverges.

Notation (carried explicitly throughout; cf. user notation directive)
---------------------------------------------------------------------
    Cm      specific membrane capacitance               [uF / cm^2]
    Rm      specific membrane resistance (= 1 / g_pas)   [Ohm * cm^2]
    Ra      axial resistivity                            [Ohm * cm]
    tau_m   membrane time constant, tau_m = Rm * Cm      [ms]  (see note below)
    tau_0   slowest passive eigen-time-constant of the
            multi-compartment model                      [ms]
    R_in    somatic input resistance (steady state)      [MOhm]
    V_rest  resting membrane potential (= e_pas)         [mV]

Unit note for tau_m
-------------------
With Cm in uF/cm^2 and Rm in Ohm*cm^2,
    Rm * Cm  has units  Ohm * uF  =  Ohm * 1e-6 F  =  1e-6 s  =  1e-3 ms.
Hence, for *every* call,
    tau_m[ms] = Rm[Ohm*cm^2] * Cm[uF/cm^2] * 1e-3.
The factor 1e-3 is written out wherever it is used; it is never absorbed
silently.

Cable-theory fact used here (Rall 1969)
---------------------------------------
For a passive neuron with sealed ends the somatic transient is a sum of
exponentials  V(t) = sum_i A_i exp(-t / tau_i)  with  tau_0 > tau_1 > ...,
and the *slowest* constant equals the membrane time constant for every
geometry:  tau_0 = Rm * Cm.  The faster tau_{i>=1} are dendritic equalising
constants.  Therefore tau_0 extracted from the late tail of a simulated
brief-pulse decay must equal Rm*Cm*1e-3 to within the tail SNR; this identity
is what the NEURON smoke test checks on a single compartment, where
tau_0 = Rm*Cm exactly.

Required `cell` interface (duck-typed; your PassiveCell satisfies it)
--------------------------------------------------------------------
    cell.set_passive(Cm: float, Rm: float, Ra: float) -> None
    cell.set_e_pas(e_pas_mV: float)                    -> None
    cell.simulate(stim_amp_pA, stim_delay_ms, stim_dur_ms,
                  tstop_ms, v_init_mV, dt_ms) -> (t_ms, v_mV)   # np arrays

Dependencies: numpy, scipy (curve_fit).  No NEURON import here -- the cell
object owns NEURON; this module only consumes its public API.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import curve_fit


# ===========================================================================
#  1. Robust single-exponential tail fit
# ===========================================================================
def fit_single_exponential(
    t_ms: np.ndarray,
    v_mV: np.ndarray,
    fit_window_ms: Tuple[float, float],
    *,
    p0_tau_ms: float = 15.0,
) -> Dict[str, float]:
    """Fit  v(t) = A * exp(-(t - t0) / tau) + C  on  t in fit_window_ms.

    t0 is fixed to fit_window_ms[0] so A is the tail amplitude at the window
    start (this de-correlates A and tau and stabilises the fit).  No sign is
    assumed: A is free, so the same routine fits depolarising or
    hyperpolarising decays.

    Returns a dict with keys:
        tau_ms, amp_mV (A), offset_mV (C), r2, n_points, ok, reason
    All physical quantities keep their units; `ok=False` carries `reason`.
    """
    t = np.asarray(t_ms, dtype=float)
    v = np.asarray(v_mV, dtype=float)
    lo, hi = fit_window_ms
    m = (t >= lo) & (t <= hi)
    n = int(m.sum())
    out = dict(tau_ms=np.nan, amp_mV=np.nan, offset_mV=np.nan,
               r2=np.nan, n_points=n, ok=False, reason="")
    if n < 5:
        out["reason"] = f"only {n} points in window {fit_window_ms}"
        return out

    tt = t[m] - lo            # shift so t0 = 0 inside the window
    vv = v[m]
    A0 = float(vv[0] - vv[-1])
    C0 = float(vv[-1])
    if A0 == 0.0:
        A0 = float(np.sign(vv[0] - np.mean(vv)) or 1.0) * (np.std(vv) + 1e-9)

    def _model(x, A, tau, C):
        # guard tau > 0 via bounds below
        return A * np.exp(-x / tau) + C

    try:
        popt, _ = curve_fit(
            _model, tt, vv,
            p0=[A0, p0_tau_ms, C0],
            bounds=([-np.inf, 1e-3, -np.inf], [np.inf, 1e4, np.inf]),
            maxfev=20000,
        )
    except Exception as e:  # noqa: BLE001 - report, do not raise
        out["reason"] = f"curve_fit failed: {type(e).__name__}: {e}"
        return out

    A, tau, C = (float(popt[0]), float(popt[1]), float(popt[2]))
    resid = vv - _model(tt, A, tau, C)
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((vv - vv.mean()) ** 2)) + 1e-30
    out.update(tau_ms=tau, amp_mV=A, offset_mV=C,
               r2=1.0 - ss_res / ss_tot, ok=True)
    return out


# ===========================================================================
#  2. Model-derived quantities the brief-pulse loss does NOT target
# ===========================================================================
def model_tau0_from_brief_pulse(
    cell,
    *,
    amplitude_pA: float = -200.0,
    stim_dur_ms: float = 0.5,
    v_rest_mV: float = -70.0,
    pre_pad_ms: float = 10.0,
    post_pad_ms: float = 200.0,
    dt_ms: float = 0.025,
    tail_window_ms: Tuple[float, float] = (30.0, 120.0),
) -> Dict[str, float]:
    """Simulate ONE brief pulse (mirrors `_simulate_square_subthreshold`) and
    fit a single exponential to the *late* tail to recover the model's slow
    constant tau_0.  For a passive cell tau_0 = Rm*Cm; this is the internal
    cross-check that the fitted Rm*Cm really is the model's tau_m.

    `cell` must already be at the parameter point of interest
    (call set_passive / set_e_pas before this)."""
    tstop_ms = pre_pad_ms + stim_dur_ms + post_pad_ms
    t_ms, v_mV = cell.simulate(
        stim_amp_pA=amplitude_pA, stim_delay_ms=pre_pad_ms,
        stim_dur_ms=stim_dur_ms, tstop_ms=tstop_ms,
        v_init_mV=v_rest_mV, dt_ms=dt_ms,
    )
    t_ms = np.asarray(t_ms) - pre_pad_ms          # t = 0 at pulse onset
    fit = fit_single_exponential(t_ms, np.asarray(v_mV), tail_window_ms)
    fit["peak_deflection_mV"] = float(np.max(np.abs(
        np.asarray(v_mV) - v_rest_mV)))
    return fit


def model_input_resistance(
    cell,
    *,
    amplitude_pA: float = -30.0,
    settle_ms: float = 600.0,
    measure_last_ms: float = 20.0,
    v_rest_mV: float = -70.0,
    dt_ms: float = 0.05,
) -> Dict[str, float]:
    """Steady-state somatic input resistance from a long, subthreshold step.

    R_in[MOhm] = (V_ss - V_rest)[mV] / I[pA] * 1e3.
    (mV/pA = GOhm; * 1e3 -> MOhm.)  Brief 0.5 ms pulses never reach steady
    state, so this is a *free prediction* of the fit -- comparing it to the
    Allen R_in quantifies the steady-state under-constraint."""
    pre_ms = 50.0
    tstop_ms = pre_ms + settle_ms
    t_ms, v_mV = cell.simulate(
        stim_amp_pA=amplitude_pA, stim_delay_ms=pre_ms,
        stim_dur_ms=settle_ms, tstop_ms=tstop_ms,
        v_init_mV=v_rest_mV, dt_ms=dt_ms,
    )
    t_ms = np.asarray(t_ms)
    v_mV = np.asarray(v_mV)
    base_m = (t_ms >= pre_ms - 20.0) & (t_ms < pre_ms)
    ss_m = (t_ms >= (pre_ms + settle_ms - measure_last_ms)) & \
           (t_ms <= (pre_ms + settle_ms))
    v_base = float(v_mV[base_m].mean())
    v_ss = float(v_mV[ss_m].mean())
    dv = v_ss - v_base
    rin_MOhm = dv / float(amplitude_pA) * 1e3
    return dict(rin_MOhm=rin_MOhm, dv_mV=dv, v_base_mV=v_base, v_ss_mV=v_ss)


def tail_snr_of_trace(
    t_ms: np.ndarray,
    v_mV: np.ndarray,
    v_rest_mV: float,
    noise_sigma_mV: float,
    probe_times_ms: Sequence[float] = (20.0, 50.0, 80.0),
    avg_halfwidth_ms: float = 2.0,
) -> Dict[float, float]:
    """|deflection| / noise_sigma at chosen post-onset times.  Shows *why*
    tau_m is loosely constrained: the slow-tail SNR is what an exponential
    fit has to work with.  t=0 is assumed to be the pulse onset."""
    t = np.asarray(t_ms)
    v = np.asarray(v_mV)
    snr: Dict[float, float] = {}
    for tp in probe_times_ms:
        m = (t >= tp - avg_halfwidth_ms) & (t <= tp + avg_halfwidth_ms)
        if not m.any():
            snr[float(tp)] = np.nan
            continue
        defl = abs(float(v[m].mean()) - v_rest_mV)
        snr[float(tp)] = defl / max(noise_sigma_mV, 1e-9)
    return snr


# ===========================================================================
#  3. One-cell consistency report
# ===========================================================================
@dataclass
class ConsistencyReport:
    specimen_id: int
    cm: float
    rm: float
    ra: float
    tau_m_from_product_ms: float          # Rm*Cm*1e-3
    tau0_model_ms: float                  # fit of model brief-pulse tail
    tau0_model_r2: float
    tau_exp_tail_ms: float                # fit of EXPERIMENTAL brief-pulse tail
    tau_exp_tail_r2: float
    tau_m_allen_ms: float                 # reference
    rin_model_MOhm: float                 # steady-state prediction
    rin_allen_MOhm: float                 # reference
    cm_railed: bool
    ra_railed: bool
    tail_snr: Dict[float, float] = field(default_factory=dict)

    def pretty(self) -> str:
        def _ratio(a, b):
            return (a / b) if (np.isfinite(a) and np.isfinite(b) and b != 0)\
                else np.nan
        lines = [
            f"--- consistency report: specimen {self.specimen_id} ---",
            f"  fitted (Cm, Rm, Ra) = ({self.cm:.3f} uF/cm^2, "
            f"{self.rm:.0f} Ohm*cm^2, {self.ra:.0f} Ohm*cm)"
            + ("   [Cm RAILED]" if self.cm_railed else "")
            + ("   [Ra RAILED]" if self.ra_railed else ""),
            "",
            "  tau_m comparison [ms]:",
            f"    Rm*Cm*1e-3 (analytic model tau_m) : "
            f"{self.tau_m_from_product_ms:8.2f}",
            f"    tau_0 from model brief-pulse tail : "
            f"{self.tau0_model_ms:8.2f}   (r2={self.tau0_model_r2:.3f})  "
            f"<- must match line above if extraction is sound",
            f"    tau   from EXPERIMENT pulse tail  : "
            f"{self.tau_exp_tail_ms:8.2f}   (r2={self.tau_exp_tail_r2:.3f})",
            f"    tau_m Allen (reference)           : "
            f"{self.tau_m_allen_ms:8.2f}",
            f"    ratio  model/Allen                : "
            f"{_ratio(self.tau_m_from_product_ms, self.tau_m_allen_ms):8.2f}",
            "",
            "  R_in comparison [MOhm]  (brief pulses never reach steady state):",
            f"    R_in model (steady-state sim)     : {self.rin_model_MOhm:8.1f}",
            f"    R_in Allen (reference)            : {self.rin_allen_MOhm:8.1f}",
            f"    ratio  model/Allen                : "
            f"{_ratio(self.rin_model_MOhm, self.rin_allen_MOhm):8.2f}",
            "",
            "  slow-tail SNR of EXPERIMENT pulse (|deflection| / sigma):",
        ]
        for tp in sorted(self.tail_snr):
            lines.append(f"    t = {tp:5.0f} ms : SNR = {self.tail_snr[tp]:6.1f}")
        return "\n".join(lines)


def consistency_report(
    cell,
    *,
    specimen_id: int,
    cm: float,
    rm: float,
    ra: float,
    v_rest_mV: float,
    tau_m_allen_ms: float,
    rin_allen_MOhm: float,
    exp_pulse_t_ms: Optional[np.ndarray] = None,
    exp_pulse_v_mV: Optional[np.ndarray] = None,
    noise_sigma_mV: float = 0.01,
    cm_bounds: Tuple[float, float] = (0.3, 3.0),
    ra_bounds: Tuple[float, float] = (50.0, 1000.0),
    tail_window_ms: Tuple[float, float] = (30.0, 120.0),
    brief_amplitude_pA: float = -200.0,
    brief_dur_ms: float = 0.5,
    rail_rtol: float = 1e-3,
) -> ConsistencyReport:
    """Build a ConsistencyReport for one fitted cell.

    `exp_pulse_{t,v}` are the averaged *hyperpolarising* brief-pulse bundle
    (t in ms, t=0 at onset; v in mV).  Pass `SweepBundle.t * 1e3` and
    `SweepBundle.v_mV`.  If omitted, the experimental-tail fit and SNR are
    skipped (NaN)."""
    cell.set_passive(Cm=cm, Rm=rm, Ra=ra)
    cell.set_e_pas(v_rest_mV)

    tau_product = rm * cm * 1e-3
    m_tau0 = model_tau0_from_brief_pulse(
        cell, amplitude_pA=brief_amplitude_pA, stim_dur_ms=brief_dur_ms,
        v_rest_mV=v_rest_mV, tail_window_ms=tail_window_ms)
    m_rin = model_input_resistance(cell, v_rest_mV=v_rest_mV)

    tau_exp, tau_exp_r2 = np.nan, np.nan
    tail_snr: Dict[float, float] = {}
    if exp_pulse_t_ms is not None and exp_pulse_v_mV is not None:
        fe = fit_single_exponential(
            np.asarray(exp_pulse_t_ms), np.asarray(exp_pulse_v_mV),
            tail_window_ms)
        tau_exp, tau_exp_r2 = fe["tau_ms"], fe["r2"]
        tail_snr = tail_snr_of_trace(
            np.asarray(exp_pulse_t_ms), np.asarray(exp_pulse_v_mV),
            v_rest_mV, noise_sigma_mV)

    def _railed(x, bounds):
        return bool(np.isclose(x, bounds[0], rtol=rail_rtol) or
                    np.isclose(x, bounds[1], rtol=rail_rtol))

    return ConsistencyReport(
        specimen_id=specimen_id, cm=cm, rm=rm, ra=ra,
        tau_m_from_product_ms=tau_product,
        tau0_model_ms=m_tau0["tau_ms"], tau0_model_r2=m_tau0["r2"],
        tau_exp_tail_ms=tau_exp, tau_exp_tail_r2=tau_exp_r2,
        tau_m_allen_ms=tau_m_allen_ms,
        rin_model_MOhm=m_rin["rin_MOhm"], rin_allen_MOhm=rin_allen_MOhm,
        cm_railed=_railed(cm, cm_bounds), ra_railed=_railed(ra, ra_bounds),
        tail_snr=tail_snr,
    )
