"""ih_mechanism.py -- the I_h layer of the fitting pipeline (Stage 1 of
TEEG_Ih_fit_staged_plan.md; decisions D-005, D-006, D-007).

What this module owns
---------------------
  * IhSpec            the configuration of the h-current: which compiled
                      mechanism, which spatial law, which regions, the
                      configuration shift of the base kinetics.
  * attach_ih         insert the mechanism ONCE into a cell and precompute the
                      per-segment spatial factor (uniform = 1 everywhere).
  * set_ih            the per-loss-evaluation update: density, activation
                      shift, tau scale.
  * balance_e_pas     the analytic rest balance, plan Eq. (11.3) / (D-005.2):
                      per segment, e_pas is set so that at v = V_rest the leak
                      cancels the resting h-current. A uniform v carries no
                      axial current, so v == V_rest is then a stationary state
                      of the whole cable and finitialize(V_rest) starts the
                      cell at rest with m = mInf(V_rest).
  * ih_rest_summary   the derived quantities the campaign reports next to the
                      fitted parameters (plan section 6.5).
  * rest_drift        the check behind smoke S3: simulate with no stimulus
                      and return max |v(t) - V_rest|.

Cell contract (duck-typed; the monolith's PassiveCell satisfies it)
------------------------------------------------------------------
  cell.soma, cell.dend, cell.apic, cell.axon : lists of h.Section
  cell.simulate(stim_amp_pA, stim_delay_ms, stim_dur_ms, tstop_ms,
                v_init_mV, dt_ms)            : (t_ms, v_mV) at soma centre
  seg.g_pas, seg.e_pas                        : the pas mechanism is inserted

Order of operations at every loss evaluation (Stage 3 puts this in
ParamSpec.apply): cell.set_passive(Cm, Rm, Ra) -> set_ih(...) ->
balance_e_pas(...). The balance depends on g_pas (hence on Rm and F) and on
the I_h density and shift, so it is always last.

Sign conventions (fixed in D-005; see mod/Ih.mod)
-------------------------------------------------
    minf(v) = minf_base(v - vshift_base - dv_h)
    mtau(v) = kappa_tau * mtau_base(v - vshift_base)
Positive dv_h / vshift_base = curves moved to MORE depolarised voltages = more
activation at a given v. Kalmbach 2018's "-20 mV shift" of the Kole rates is
vshift_base = +20 in this convention.

Notation (units carried explicitly)
-----------------------------------
    v, V_rest, e_pas, ehcn   [mV]      gbar, g_pas   [S/cm2]
    dis                      [um]      dv_h, vshift  [mV]
    kappa_tau                [1]       mtau          [ms]
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import human_ih_params as _hip

__all__ = [
    "IhSpec", "SPATIAL_LAWS", "REGIONS_DEFAULT",
    "eyal_factor_323", "hay_factor_dmax",
    "attach_ih", "set_ih", "balance_e_pas", "ih_rest_summary", "rest_drift",
    "sag_metrics",
]

# Spatial laws for gbar(s). "uniform" is the deliverable (D-005); the two
# exponential laws are labelled comparisons (plan section 6.2).
SPATIAL_LAWS: Tuple[str, ...] = ("uniform", "eyal_exp_323", "hay_exp_dmax")

# Regions that carry I_h. The Hay axon stub is EXCLUDED by default (D-005 open
# point, assistant's proposal); pass regions=("soma","dend","apic","axon") to
# include it, and the output will say so.
REGIONS_DEFAULT: Tuple[str, ...] = ("soma", "dend", "apic")

# Eyal 2016 Eq. 4 (sign verified from the PDF, plan section 3): the apical
# density is gbar * (-0.8696 + 2.087 * exp(dis / 323 um)); soma and basal carry
# gbar. It is Hay 2011's rule with the length scale frozen at 323 um.
_EYAL_A = -0.8696
_EYAL_B = 2.087
_EYAL_LAMBDA_UM = 323.0
_HAY_K = 3.6161


def eyal_factor_323(dis_um: float) -> float:
    """Apical multiplier of Eyal 2016 Eq. 4 at path distance dis_um [um]."""
    return _EYAL_A + _EYAL_B * math.exp(float(dis_um) / _EYAL_LAMBDA_UM)


def hay_factor_dmax(dis_um: float, dmax_um: float) -> float:
    """Apical multiplier of Hay 2011 normalised by the cell's own apical
    maximum path distance dmax_um [um] (the synthetic generator's
    'hay_exponential'). Clamped at 0 as the generator does."""
    if not (float(dmax_um) > 0.0):
        return 1.0
    return max(_EYAL_A + _EYAL_B * math.exp(_HAY_K * float(dis_um) / float(dmax_um)), 0.0)


@dataclass
class IhSpec:
    """Configuration of the h-current (everything that is NOT fitted)."""
    mechanism: str = "Ih_human"            # D-007 C1: Rich 2021 base by default
    distribution: str = "uniform"          # D-005
    regions: Tuple[str, ...] = REGIONS_DEFAULT
    vshift_base_mV: float = 0.0            # +20 on "Ih" reproduces Kalmbach 2018
    ehcn_mV: Optional[float] = None        # None -> the mechanism's own default
    mtau_min_ms: float = 0.0               # Ih_human only; 0 = published model

    def __post_init__(self) -> None:
        if self.mechanism not in _hip.IH_MECHANISMS:
            raise ValueError("mechanism must be one of %s, got %r"
                             % (_hip.IH_MECHANISMS, self.mechanism))
        if self.distribution not in SPATIAL_LAWS:
            raise ValueError("distribution must be one of %s, got %r"
                             % (SPATIAL_LAWS, self.distribution))
        bad = [r for r in self.regions if r not in ("soma", "dend", "apic", "axon")]
        if bad:
            raise ValueError("unknown regions %s" % bad)
        if self.ehcn_mV is None:
            self.ehcn_mV = _hip.ehcn_default_mV(self.mechanism)

    def label(self) -> str:
        """Short tag for file names / table rows (plan: every output row names
        its mechanism and law)."""
        tag = "%s_%s" % (self.mechanism, self.distribution)
        if abs(float(self.vshift_base_mV)) > 0:
            tag += "_vsb%+g" % float(self.vshift_base_mV)
        if "axon" in self.regions:
            tag += "_axonIh"
        return tag

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


# ---------------------------------------------------------------------------
#  Internal helpers
# ---------------------------------------------------------------------------
def _sections_of(cell, regions: Sequence[str]) -> List:
    out: List = []
    for r in regions:
        out.extend(list(getattr(cell, r, [])))
    return out


def _all_sections(cell) -> List:
    return _sections_of(cell, ("soma", "dend", "apic", "axon"))


def _path_distances(cell, h) -> Dict[Tuple[str, float], float]:
    """Path distance [um] from the soma centre to every segment midpoint, keyed
    by (section name, seg.x). Uses the same NEURON call pattern as the
    monolith's PassiveCell._precompute_F_per_segment."""
    h.distance(0, cell.soma[0](0.5))
    out: Dict[Tuple[str, float], float] = {}
    for sec in _all_sections(cell):
        for seg in sec:
            out[(sec.name(), seg.x)] = float(h.distance(seg.x, sec=sec))
    return out


def _mech_obj(seg, mechanism: str):
    return getattr(seg, mechanism)


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------
def attach_ih(cell, spec: IhSpec) -> Dict[Tuple[str, float], float]:
    """Insert the mechanism into the spec's regions (idempotent) and store the
    per-segment spatial factor on the cell as ``cell._ih_factor``.

    Returns the factor dict {(section name, seg.x): factor}. Segments outside
    the regions are absent from the dict (they carry no I_h). gbar itself is
    NOT set here -- call set_ih() -- so that attach_ih is a pure build step.
    """
    from neuron import h  # local import: NEURON is loaded once per process

    secs = _sections_of(cell, spec.regions)
    if not secs:
        raise RuntimeError("attach_ih: no sections in regions %s" % (spec.regions,))

    dist = _path_distances(cell, h)
    dmax_apic = 0.0
    if spec.distribution == "hay_exp_dmax":
        apic_d = [dist[(sec.name(), seg.x)] for sec in cell.apic for seg in sec]
        dmax_apic = max(apic_d) if apic_d else 0.0

    factor: Dict[Tuple[str, float], float] = {}
    apic_names = {sec.name() for sec in cell.apic}
    for sec in secs:
        if not sec.has_membrane(spec.mechanism):
            sec.insert(spec.mechanism)
        is_apic = sec.name() in apic_names
        for seg in sec:
            key = (sec.name(), seg.x)
            if spec.distribution == "uniform" or not is_apic:
                f = 1.0
            elif spec.distribution == "eyal_exp_323":
                f = eyal_factor_323(dist[key])
            else:
                f = hay_factor_dmax(dist[key], dmax_apic)
            factor[key] = float(f)
            mobj = _mech_obj(seg, spec.mechanism)
            mobj.ehcn = float(spec.ehcn_mV)
            mobj.vshift = float(spec.vshift_base_mV)
            if spec.mechanism == "Ih_human":
                mobj.mTauMin = float(spec.mtau_min_ms)

    cell._ih_spec = spec
    cell._ih_factor = factor
    cell._ih_dmax_apic_um = dmax_apic
    return factor


def set_ih(cell, gbar_S_cm2: float, dv_h_mV: float = 0.0,
           kappa_tau: float = 1.0) -> None:
    """Per-evaluation update of the three fitted I_h quantities.

    gIhbar(s) = gbar * factor(s);  vshift_minf = dv_h;  tau_scale = kappa_tau.
    Requires attach_ih() to have been called on this cell.
    """
    spec: IhSpec = getattr(cell, "_ih_spec", None)
    factor = getattr(cell, "_ih_factor", None)
    if spec is None or factor is None:
        raise RuntimeError("set_ih: call attach_ih(cell, spec) first")
    if not (float(gbar_S_cm2) >= 0.0):
        raise ValueError("gbar_S_cm2 must be >= 0, got %r" % (gbar_S_cm2,))
    if not (float(kappa_tau) > 0.0):
        raise ValueError("kappa_tau must be > 0, got %r" % (kappa_tau,))
    for sec in _sections_of(cell, spec.regions):
        for seg in sec:
            mobj = _mech_obj(seg, spec.mechanism)
            mobj.gIhbar = float(gbar_S_cm2) * factor[(sec.name(), seg.x)]
            mobj.vshift_minf = float(dv_h_mV)
            mobj.tau_scale = float(kappa_tau)
    cell._ih_last = (float(gbar_S_cm2), float(dv_h_mV), float(kappa_tau))


def balance_e_pas(cell, v_rest_mV: float) -> Dict[str, float]:
    """Plan Eq. (11.3) / (D-005.2), per segment s:

        e_pas(s) = V_rest + gIhbar(s) * mInf(V_rest) * (V_rest - ehcn) / g_pas(s)

    with mInf the SHIFTED activation (vshift_base + dv_h), evaluated by the
    pure-Python reference (smoke S1 pins the compiled mechanism to it).
    Segments without I_h get e_pas = V_rest. Returns a small summary
    (soma e_pas, min, max) for logging.
    """
    spec: IhSpec = getattr(cell, "_ih_spec", None)
    factor = getattr(cell, "_ih_factor", None)
    last = getattr(cell, "_ih_last", None)
    if spec is None or factor is None or last is None:
        raise RuntimeError("balance_e_pas: call attach_ih() and set_ih() first")
    gbar, dv_h, _kappa = last
    v_rest = float(v_rest_mV)
    m_rest = _hip.minf(v_rest, spec.mechanism,
                       vshift_mV=spec.vshift_base_mV, vshift_minf_mV=dv_h)
    drive = v_rest - float(spec.ehcn_mV)          # mV; < 0 at rest (V_rest below ehcn)
    vals: List[float] = []
    for sec in _all_sections(cell):
        for seg in sec:
            key = (sec.name(), seg.x)
            g_pas = float(seg.g_pas)
            if key in factor and g_pas > 0.0:
                gh_rest = gbar * factor[key] * m_rest       # S/cm2
                e = v_rest + gh_rest * drive / g_pas
            else:
                e = v_rest
            seg.e_pas = e
            vals.append(e)
    soma_val = float(cell.soma[0](0.5).e_pas)
    return {"e_pas_soma_mV": soma_val,
            "e_pas_min_mV": float(min(vals)), "e_pas_max_mV": float(max(vals)),
            "m_inf_at_rest": float(m_rest)}


def ih_rest_summary(cell, v_rest_mV: float, rm_Ohm_cm2: float) -> Dict[str, float]:
    """Derived quantities reported alongside theta (plan section 6.5)."""
    spec: IhSpec = getattr(cell, "_ih_spec", None)
    last = getattr(cell, "_ih_last", None)
    if spec is None or last is None:
        raise RuntimeError("ih_rest_summary: call attach_ih() and set_ih() first")
    gbar, dv_h, kappa = last
    v_rest = float(v_rest_mV)
    m_rest = _hip.minf(v_rest, spec.mechanism,
                       vshift_mV=spec.vshift_base_mV, vshift_minf_mV=dv_h)
    tau_rest = _hip.mtau_ms(v_rest, spec.mechanism, vshift_mV=spec.vshift_base_mV,
                            tau_scale=kappa, mtau_min_ms=spec.mtau_min_ms)
    v_pk, tau_pk = _hip.peak_mtau_shifted_ms(spec.mechanism, vshift_mV=spec.vshift_base_mV,
                                             tau_scale=kappa)
    vhalf = _hip.v_half_mV(spec.mechanism, vshift_mV=spec.vshift_base_mV,
                           vshift_minf_mV=dv_h)
    soma_e = float(cell.soma[0](0.5).e_pas)
    return {
        "m_inf_at_rest": float(m_rest),
        "tau_h_at_rest_ms": float(tau_rest),
        "tau_h_peak_ms": float(tau_pk),
        "v_at_tau_h_peak_mV": float(v_pk),
        "v_half_mV": float(vhalf),
        "gh_rest_over_gpas": float(gbar * m_rest * float(rm_Ohm_cm2)),  # soma: factor 1
        "e_pas_soma_mV": soma_e,
        "ehcn_mV": float(spec.ehcn_mV),
    }


def rest_drift(cell, v_rest_mV: float, *, t_ms: float = 2000.0,
               dt_ms: float = 0.1) -> float:
    """max_t |v_soma(t) - V_rest| with no stimulus, from finitialize(V_rest).
    The quantity smoke S3 bounds (plan: < 1e-3 mV)."""
    t, v = cell.simulate(stim_amp_pA=0.0, stim_delay_ms=0.0, stim_dur_ms=0.0,
                         tstop_ms=float(t_ms), v_init_mV=float(v_rest_mV),
                         dt_ms=float(dt_ms))
    return float(np.max(np.abs(np.asarray(v) - float(v_rest_mV))))


def sag_metrics(t_ms: np.ndarray, v_mV: np.ndarray, *, v_rest_mV: float,
                onset_ms: float, offset_ms: float,
                ss_window_ms: float = 100.0) -> Dict[str, float]:
    """Allen-style sag fraction and a few time-course numbers on ONE step.

    sag_fraction = (V_trough - V_ss) / (V_trough - V_rest)   (Allen whitepaper
    definition: difference between the minimum and the steady-state value,
    divided by the peak deflection; 0 = no sag).
    t63_recovery_ms = time after the trough at which v has recovered 63 % of
    (V_ss - V_trough): a crude sag time constant, monotone in kappa_tau.
    rebound_mV = max(v - V_rest) after offset (positive = rebound depolarisation).
    """
    t = np.asarray(t_ms, dtype=float); v = np.asarray(v_mV, dtype=float)
    step = (t >= onset_ms) & (t <= offset_ms)
    ss = (t >= offset_ms - ss_window_ms) & (t <= offset_ms)
    post = t > offset_ms
    i_tr = int(np.argmin(np.where(step, v, np.inf)))
    v_tr = float(v[i_tr]); v_ss = float(np.mean(v[ss]))
    denom = v_tr - float(v_rest_mV)
    sag = (v_tr - v_ss) / denom if abs(denom) > 1e-9 else 0.0
    target = v_tr + 0.63 * (v_ss - v_tr)
    after = (t >= t[i_tr]) & step
    idx = np.where(after & (v >= target))[0]
    t63 = float(t[idx[0]] - t[i_tr]) if idx.size else float("nan")
    rebound = float(np.max(v[post] - float(v_rest_mV))) if post.any() else float("nan")
    return {"v_trough_mV": v_tr, "v_ss_mV": v_ss, "sag_fraction": float(max(sag, 0.0)),
            "t63_recovery_ms": t63, "rebound_mV": rebound}
