# -*- coding: utf-8 -*-
"""
synthetic_ground_truth.py
=========================

Forward / ground-truth generator for validating the passive GP fit by
**parameter recovery**: build each reconstructed morphology in NEURON with
*known* (Cm, Rm, Ra) -- optionally plus Mainen Na/Kv active channels -- run the
SAME stimulation protocols the fitter sees (brief Square-Subthreshold pulses +
Long-Square steps), add whole-cell recording noise, and emit the somatic Vm
traces in a form the GP pipeline consumes directly. Feed the result to the GP;
if it cannot recover the injected (Cm, Rm, Ra) from clean passive data, the
failure is in the optimiser/loss/sim, not in the biology.

Design contract (read this)
---------------------------
1. PARITY. The cell here is built the SAME way the fitter builds it
   (Import3d -> Hay axon stub -> insert pas -> path-distance spine factor F).
   For a real assessment pass the monolith's own builder via
   `passive_cell_factory=build_neuron_model` so generator and fitter are
   literally the same code; the bundled `SyntheticPassiveCell` mirrors it for
   standalone use/tests.
2. WHAT THE CHANNELS DO. Na/Kv are depolarisation-activated, so on the
   hyperpolarising subthreshold fitting data they are inert -- "active" and
   "passive-only" data are nearly identical there; the channels only spike on
   depolarising/suprathreshold steps (whole-cell realism). To make synthetic
   data fail the way real cells fail (sag, tau_m mismatch on high-R_in cells)
   you need an I_h mechanism, NOT Na/Kv: see `GroundTruthParams.extra_mechs`,
   which inserts any additional density mechanism (e.g. an `Ih` SUFFIX) the
   same way -- drop the compiled mod in and name it.
3. GROUND TRUTH = the injected (Cm, Rm, Ra). Derived R_in (steady-state sim)
   and tau_m (= Rm*Cm for a passive cell) are stored as secondary references
   the recovered parameters must also reproduce.

Notation (units carried; cf. notation directive)
-------------------------------------------------
    Cm [uF/cm^2]   Rm [Ohm*cm^2]   Ra [Ohm*cm]   e_pas/V [mV]
    gbar_* [pS/um^2]  (Mainen convention; 1 pS/um^2 = 1e-4 S/cm^2)
    tau_m[ms] = Rm * Cm * 1e-3      R_in[MOhm] = dV[mV]/I[pA]*1e3

Dependencies: numpy, neuron. (scipy only if you reuse the consistency
diagnostic for tau extraction; an analytic fallback is used otherwise.)
"""
from __future__ import annotations

import math
import pickle
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ===========================================================================
#  Ground-truth parameter set
# ===========================================================================
@dataclass
class GroundTruthParams:
    """The KNOWN biophysics injected into the model.

    Passive defaults are placeholders -- set them to YOUR intended ground truth
    (e.g. the Eyal 2016 human L2/3 values). The Cm value is itself debated in
    the literature, so it is an explicit input, not a baked-in constant."""
    cm_uF_cm2: float = 1.0
    rm_Ohm_cm2: float = 15000.0
    ra_Ohm_cm: float = 150.0
    e_pas_mV: float = -70.0
    spine_factor_F: float = 1.9
    spine_cutoff_um: float = 60.0

    # --- optional Mainen active channels (depolarisation-activated) ---
    active: bool = False
    gbar_na_pS_um2: float = 1000.0       # ~0.1 S/cm^2 in soma
    gbar_kv_pS_um2: float = 100.0
    ena_mV: float = 60.0
    ek_mV: float = -90.0
    celsius: float = 37.0                # Mainen channels are tuned for 37 C
    active_regions: Tuple[str, ...] = ("soma", "axon")   # where Na/Kv go

    # --- optional I_h (Hay 2011 / Kole 2006 kinetics; the mechanism Eyal used) ---
    ih: Optional["IhConfig"] = None

    # --- optional extra density mechanisms (any other SUFFIX) ---
    # name -> {param_name: value}; inserted uniformly in `extra_regions`.
    extra_mechs: Dict[str, Dict[str, float]] = field(default_factory=dict)
    extra_regions: Tuple[str, ...] = ("soma", "dend", "apic")

    @property
    def tau_m_ms(self) -> float:
        """Passive membrane time constant Rm*Cm. NOTE: with I_h (or any active
        mechanism) present this is NO LONGER the cell's effective time constant
        -- I_h adds a resting conductance, so the measured tau and R_in deviate.
        That deviation is precisely the contamination the I_h mode injects."""
        return self.rm_Ohm_cm2 * self.cm_uF_cm2 * 1e-3


@dataclass
class IhConfig:
    """Hay et al. 2011 / Kole et al. 2006 h-current (the model Eyal 2018 used).

    distribution:
        "uniform"         -- gIhbar_S_cm2 in every `regions` segment (robust on
                             any morphology; matches the Kalmbach human-L3 setup).
        "hay_exponential" -- in apical segments, gIhbar increases with path
                             distance d from the soma exactly as in Hay 2011:
                               gIhbar(d) = gIhbar * (-0.8696 + 2.0870*exp(
                                           3.6161 * d / d_max_apical))
                             soma + basal stay uniform at gIhbar. Faithful to
                             Eyal's distribution but needs apical sections."""
    gIhbar_S_cm2: float = 5e-5
    ehcn_mV: float = -45.0
    distribution: str = "uniform"            # "uniform" | "hay_exponential"
    regions: Tuple[str, ...] = ("soma", "dend", "apic")


@dataclass
class ProtocolConfig:
    """Stimulus protocols mirroring the Allen archive the fitter expects."""
    # Square Subthreshold brief pulses
    ss_amplitude_pA: float = 200.0
    ss_duration_ms: float = 0.5
    ss_n_repeats: int = 30               # per polarity (averaged + stored)
    ss_polarities: Tuple[str, ...] = ("hyp", "dep")
    ss_pre_pad_ms: float = 10.0
    ss_post_pad_ms: float = 200.0
    ss_sampling_rate_Hz: float = 50000.0

    # Long Square steps (hyperpolarising = fit/validation; depolarising = extra)
    ls_hyp_amplitudes_pA: Tuple[float, ...] = (-10.0, -30.0, -50.0, -70.0, -90.0)
    ls_dep_amplitudes_pA: Tuple[float, ...] = ()      # e.g. (50, 100, 200) to show spikes
    ls_onset_ms: float = 100.0
    ls_duration_ms: float = 1000.0
    ls_post_ms: float = 200.0
    ls_n_repeats: int = 1
    ls_sampling_rate_Hz: float = 20000.0


@dataclass
class NoiseConfig:
    """Whole-cell recording noise added to EACH individual sweep/pulse before
    averaging. Two components reproduce the structure seen in real recordings:

    FAST (within-sweep fuzz): `sigma_mV` white RMS, optionally AR(1)-correlated
        via `rho_lag1`. Averages down as ~sigma_mV/sqrt(n_repeats).

    SLOW (sweep-to-sweep variability -- the trace FANNING in the data):
        `baseline_sigma_mV` -- a single random DC offset drawn ONCE per sweep,
            so individual sweeps sit at slightly different levels (the ~0.8 mV
            band width in the example figure). Also averages as
            ~baseline_sigma_mV/sqrt(n_repeats), so the bundle mean stays clean.
        `drift_sigma_mV` -- a random slow linear tilt per sweep (electrode/seal
            drift), so the sweeps are NOT parallel.
    To match the attached figure, the slow terms dominate the visible spread;
    set e.g. baseline_sigma_mV ~ 0.2, drift_sigma_mV ~ 0.1, sigma_mV ~ 0.05."""
    sigma_mV: float = 0.10               # per-sample white RMS (within-sweep fuzz)
    rho_lag1: float = 0.0                # AR(1) coefficient (0 = white)
    baseline_sigma_mV: float = 0.0       # per-sweep DC offset RMS (trace fanning)
    drift_sigma_mV: float = 0.0          # per-sweep slow linear-tilt RMS (mV across sweep)
    seed: int = 0


# ===========================================================================
#  Lightweight bundle (mirrors the monolith SweepBundle attributes)
# ===========================================================================
@dataclass
class Bundle:
    polarity: str
    amplitude_pA: float
    t: np.ndarray                 # seconds, t=0 at onset (SS) or sweep start (LS)
    v_mV: np.ndarray
    i_pA: np.ndarray
    stim_onset_s: float
    stim_duration_s: float
    n_repeats_averaged: int
    sweep_numbers: List[int]
    sampling_rate_Hz: float
    stimulus_name: str = ""


@dataclass
class SyntheticCell:
    specimen_id: int
    swc_path: str
    ground_truth: GroundTruthParams
    square_subthreshold: List[Bundle]
    long_square_subthreshold: List[Bundle]
    ss_individual_pulses: List[Dict[str, Any]]
    rin_MOhm: float                       # measured (steady-state sim)
    tau_ms: float                         # Rm*Cm (passive) / measured (active)
    v_rest_mV: float
    sag_ratio: float                      # 0 for passive; measured if I_h present
    # depolarising LS sweeps (not used by the passive fitter; kept for realism)
    long_square_depolarising: List[Bundle] = field(default_factory=list)


# ===========================================================================
#  Cell builder (mirrors the monolith PassiveCell; adds channels)
# ===========================================================================
class SyntheticPassiveCell:
    """Compartmental model mirroring the fitter's PassiveCell, plus optional
    active/extra mechanisms. Public API matches PassiveCell:
    set_passive / set_e_pas / simulate, and exposes soma/dend/apic/axon lists."""

    def __init__(self, swc_path: Path, gt: GroundTruthParams):
        from neuron import h
        h.load_file("stdrun.hoc")
        h.load_file("import3d.hoc")
        self.h = h
        self.gt = gt
        self.soma: List = []
        self.dend: List = []
        self.apic: List = []
        self.axon: List = []

        self._import_swc(Path(swc_path))
        self._categorise()
        self._replace_axon_hay_stub()
        for sec in h.allsec():
            sec.insert("pas")
        self._precompute_F()
        self._insert_active()
        self._insert_ih()
        self._insert_extra()

        self._ic = h.IClamp(self.soma[0](0.5))
        self._ic.delay = self._ic.dur = self._ic.amp = 0
        self._t = h.Vector().record(h._ref_t)
        self._v = h.Vector().record(self.soma[0](0.5)._ref_v)

    # ---- construction ----
    def _import_swc(self, swc_path: Path) -> None:
        reader = self.h.Import3d_SWC_read()
        reader.input(str(swc_path))
        self.h.Import3d_GUI(reader, 0).instantiate(None)

    def _categorise(self) -> None:
        for sec in self.h.allsec():
            n = sec.name().lower()
            (self.soma if "soma" in n else self.apic if "apic" in n
             else self.dend if "dend" in n else self.axon if "axon" in n
             else self.dend).append(sec)
        if not self.soma:
            raise RuntimeError("no soma after SWC import")

    def _replace_axon_hay_stub(self) -> None:
        for sec in list(self.axon):
            self.h.delete_section(sec=sec)
        self.axon = []
        for i in range(2):
            ax = self.h.Section(name=f"axon[{i}]")
            ax.L, ax.diam, ax.nseg = 30.0, 1.0, 5
            self.axon.append(ax)
        self.axon[0].connect(self.soma[0](1.0), 0)
        self.axon[1].connect(self.axon[0](1.0), 0)

    def _precompute_F(self) -> None:
        h = self.h
        h.distance(0, self.soma[0](0.5))
        self._Fmul: Dict[Tuple[str, float], float] = {}
        for sec in self.dend + self.apic:
            for seg in sec:
                d = h.distance(seg.x, sec=sec)
                self._Fmul[(sec.name(), seg.x)] = (
                    self.gt.spine_factor_F if d > self.gt.spine_cutoff_um else 1.0)
        for sec in self.soma + self.axon:
            for seg in sec:
                self._Fmul[(sec.name(), seg.x)] = 1.0

    def _regions(self, names: Sequence[str]) -> List:
        out = []
        for nm in names:
            out += {"soma": self.soma, "dend": self.dend,
                    "apic": self.apic, "axon": self.axon}.get(nm, [])
        return out

    def _insert_active(self) -> None:
        if not self.gt.active:
            return
        self.h.celsius = float(self.gt.celsius)
        for sec in self._regions(self.gt.active_regions):
            sec.insert("na"); sec.insert("kv")
            for seg in sec:
                seg.na.gbar = float(self.gt.gbar_na_pS_um2)
                seg.kv.gbar = float(self.gt.gbar_kv_pS_um2)
            sec.ena = float(self.gt.ena_mV)
            sec.ek = float(self.gt.ek_mV)

    def _insert_extra(self) -> None:
        for mech, params in self.gt.extra_mechs.items():
            for sec in self._regions(self.gt.extra_regions):
                sec.insert(mech)
                for seg in sec:
                    mobj = getattr(seg, mech)
                    for pname, pval in params.items():
                        setattr(mobj, pname, float(pval))

    def _insert_ih(self) -> None:
        """Insert the Hay/Kole Ih (the mechanism Eyal used), uniform or with the
        Hay apical exponential-with-distance distribution."""
        ih = self.gt.ih
        if ih is None:
            return
        h = self.h
        secs = self._regions(ih.regions)
        if ih.distribution == "hay_exponential":
            # max apical path distance for the normalisation
            h.distance(0, self.soma[0](0.5))
            apic_d = [h.distance(seg.x, sec=sec) for sec in self.apic for seg in sec]
            d_max = max(apic_d) if apic_d else 0.0
        for sec in secs:
            sec.insert("Ih")
            is_apic = sec in self.apic
            for seg in sec:
                if (ih.distribution == "hay_exponential" and is_apic and d_max > 0):
                    d = h.distance(seg.x, sec=sec)
                    factor = -0.8696 + 2.0870 * math.exp(3.6161 * d / d_max)
                    seg.Ih.gIhbar = float(ih.gIhbar_S_cm2) * max(factor, 0.0)
                else:
                    seg.Ih.gIhbar = float(ih.gIhbar_S_cm2)
                seg.Ih.ehcn = float(ih.ehcn_mV)

    # ---- public API (identical signature to PassiveCell) ----
    def set_passive(self, Cm: float, Rm: float, Ra: float) -> None:
        g = 1.0 / float(Rm)
        for sec in self.h.allsec():
            sec.Ra = float(Ra)
            for seg in sec:
                m = self._Fmul[(sec.name(), seg.x)]
                seg.cm = float(Cm) * m
                seg.g_pas = g * m

    def set_e_pas(self, e_pas_mV: float) -> None:
        for sec in self.h.allsec():
            for seg in sec:
                seg.e_pas = float(e_pas_mV)

    def simulate(self, stim_amp_pA, stim_delay_ms, stim_dur_ms, tstop_ms,
                 v_init_mV=-70.0, dt_ms=0.025) -> Tuple[np.ndarray, np.ndarray]:
        h = self.h
        self._ic.delay = float(stim_delay_ms)
        self._ic.dur = float(stim_dur_ms)
        self._ic.amp = float(stim_amp_pA) * 1e-3      # pA -> nA
        h.dt = float(dt_ms)
        h.tstop = float(tstop_ms)
        h.v_init = float(v_init_mV)
        h.finitialize(h.v_init)
        h.run()
        return np.array(self._t), np.array(self._v)


def build_synthetic_cell(swc_path: Path, gt: GroundTruthParams,
                         passive_cell_factory: Optional[Callable] = None):
    """Build the cell and apply (Cm, Rm, Ra, e_pas).

    `passive_cell_factory(swc_path, F)` lets you inject the monolith's
    `build_neuron_model` for exact parity; if None, `SyntheticPassiveCell` is
    used (and active/extra mechs are only available on that path)."""
    if passive_cell_factory is not None:
        cell = passive_cell_factory(swc_path, F=gt.spine_factor_F)
        if gt.active or gt.extra_mechs or gt.ih is not None:
            raise ValueError("active/I_h/extra mechs require the bundled "
                             "SyntheticPassiveCell (pass passive_cell_factory=None)")
    else:
        cell = SyntheticPassiveCell(Path(swc_path), gt)
    cell.set_passive(gt.cm_uF_cm2, gt.rm_Ohm_cm2, gt.ra_Ohm_cm)
    cell.set_e_pas(gt.e_pas_mV)
    return cell


# ===========================================================================
#  Recording noise
# ===========================================================================
def add_recording_noise(v: np.ndarray, sigma_mV: float, rho_lag1: float,
                        rng: np.random.Generator, *,
                        baseline_sigma_mV: float = 0.0,
                        drift_sigma_mV: float = 0.0) -> np.ndarray:
    """Add one sweep's recording noise to a clean trace.

    Fast component: white (rho_lag1=0) or stationary AR(1) Gaussian, RMS
    sigma_mV. Slow components (drawn ONCE per call, i.e. per sweep): a DC offset
    ~N(0, baseline_sigma_mV) and a linear tilt of total span ~N(0, drift_sigma_mV)
    centred on the sweep. The slow terms create the sweep-to-sweep fanning and
    non-parallel drift seen in real whole-cell recordings; they average down as
    ~1/sqrt(n_repeats) so the averaged bundle stays clean."""
    n = len(v)
    out = np.asarray(v, dtype=float).copy()
    if sigma_mV > 0:
        if rho_lag1 == 0.0:
            out = out + rng.normal(0.0, sigma_mV, size=n)
        else:
            rho = float(np.clip(rho_lag1, -0.999, 0.999))
            innov = rng.normal(0.0, sigma_mV * math.sqrt(1 - rho * rho), size=n)
            e = np.empty(n)
            e[0] = rng.normal(0.0, sigma_mV)
            for k in range(1, n):
                e[k] = rho * e[k - 1] + innov[k]
            out = out + e
    if baseline_sigma_mV > 0:                       # per-sweep DC offset (fanning)
        out = out + rng.normal(0.0, baseline_sigma_mV)
    if drift_sigma_mV > 0:                           # per-sweep slow linear tilt
        out = out + np.linspace(-0.5, 0.5, n) * rng.normal(0.0, drift_sigma_mV)
    return out


# ===========================================================================
#  Protocol simulation
# ===========================================================================
def _resample(t_ms: np.ndarray, v: np.ndarray, sr_Hz: float,
              t0_ms: float, t1_ms: float) -> Tuple[np.ndarray, np.ndarray]:
    """Resample a NEURON trace onto a uniform grid in [t0,t1] ms at sr_Hz.
    Returns (t_seconds, v) with t starting at 0."""
    dt = 1.0 / sr_Hz
    tg_ms = np.arange(t0_ms, t1_ms, dt * 1e3)
    vg = np.interp(tg_ms, t_ms, v)
    return (tg_ms - t0_ms) * 1e-3, vg


def simulate_ss_protocol(cell, gt: GroundTruthParams, proto: ProtocolConfig,
                         noise: NoiseConfig, rng: np.random.Generator,
                         v_init_mV: Optional[float] = None
                         ) -> Tuple[List[Bundle], List[Dict[str, Any]]]:
    """Brief Square-Subthreshold pulses: one clean response per polarity, then
    `ss_n_repeats` independent noisy realisations averaged into a bundle (and
    every individual pulse stored, as in the Allen archive).

    `v_init_mV` (default e_pas) should be the cell's measured resting potential
    when active/I_h mechanisms are present, so the pre-pulse baseline is settled
    rather than drifting toward rest from e_pas."""
    v0 = gt.e_pas_mV if v_init_mV is None else float(v_init_mV)
    bundles: List[Bundle] = []
    individuals: List[Dict[str, Any]] = []
    tstop = proto.ss_pre_pad_ms + proto.ss_duration_ms + proto.ss_post_pad_ms
    sweep_id = 0
    for pol in proto.ss_polarities:
        amp = proto.ss_amplitude_pA * (1 if pol == "dep" else -1)
        t_ms, v = cell.simulate(
            stim_amp_pA=amp, stim_delay_ms=proto.ss_pre_pad_ms,
            stim_dur_ms=proto.ss_duration_ms, tstop_ms=tstop,
            v_init_mV=v0, dt_ms=0.025)
        t_s, v_clean = _resample(t_ms, v, proto.ss_sampling_rate_Hz,
                                 0.0, tstop)               # window includes pre-pad
        # shift so t=0 at onset
        t_s = t_s - proto.ss_pre_pad_ms * 1e-3
        i_clean = np.where(
            (t_s >= 0) & (t_s <= proto.ss_duration_ms * 1e-3), amp, 0.0)
        reps = []
        for _r in range(proto.ss_n_repeats):
            vr = add_recording_noise(v_clean, noise.sigma_mV, noise.rho_lag1, rng,
                                     baseline_sigma_mV=noise.baseline_sigma_mV,
                                     drift_sigma_mV=noise.drift_sigma_mV)
            reps.append(vr)
            individuals.append(dict(
                t=t_s.copy(), v=vr, i=i_clean.copy(), polarity=pol,
                peak_pA=float(amp), stim_duration_s=proto.ss_duration_ms * 1e-3,
                sampling_rate_Hz=proto.ss_sampling_rate_Hz, sweep_number=sweep_id))
            sweep_id += 1
        bundles.append(Bundle(
            polarity=pol, amplitude_pA=float(amp), t=t_s,
            v_mV=np.mean(reps, axis=0), i_pA=i_clean, stim_onset_s=0.0,
            stim_duration_s=proto.ss_duration_ms * 1e-3,
            n_repeats_averaged=proto.ss_n_repeats,
            sweep_numbers=list(range(sweep_id - proto.ss_n_repeats, sweep_id)),
            sampling_rate_Hz=proto.ss_sampling_rate_Hz,
            stimulus_name="Square Subthreshold (synthetic)"))
    return bundles, individuals


def simulate_ls_protocol(cell, gt: GroundTruthParams, proto: ProtocolConfig,
                         noise: NoiseConfig, rng: np.random.Generator,
                         amplitudes_pA: Sequence[float], polarity: str,
                         v_init_mV: Optional[float] = None
                         ) -> List[Bundle]:
    """Long-Square steps: one bundle per amplitude (averaged over ls_n_repeats).
    `v_init_mV` (default e_pas) should be the measured resting potential when
    active/I_h mechanisms shift rest above e_pas."""
    bundles: List[Bundle] = []
    v0 = gt.e_pas_mV if v_init_mV is None else float(v_init_mV)
    tstop = proto.ls_onset_ms + proto.ls_duration_ms + proto.ls_post_ms
    sweep_id = 1000
    for amp in amplitudes_pA:
        t_ms, v = cell.simulate(
            stim_amp_pA=amp, stim_delay_ms=proto.ls_onset_ms,
            stim_dur_ms=proto.ls_duration_ms, tstop_ms=tstop,
            v_init_mV=v0, dt_ms=0.025)
        t_s, v_clean = _resample(t_ms, v, proto.ls_sampling_rate_Hz, 0.0, tstop)
        i_clean = np.where(
            (t_s >= proto.ls_onset_ms * 1e-3) &
            (t_s <= (proto.ls_onset_ms + proto.ls_duration_ms) * 1e-3), amp, 0.0)
        reps = [add_recording_noise(v_clean, noise.sigma_mV, noise.rho_lag1, rng,
                                    baseline_sigma_mV=noise.baseline_sigma_mV,
                                    drift_sigma_mV=noise.drift_sigma_mV)
                for _ in range(proto.ls_n_repeats)]
        bundles.append(Bundle(
            polarity=polarity, amplitude_pA=float(amp), t=t_s,
            v_mV=np.mean(reps, axis=0), i_pA=i_clean,
            stim_onset_s=proto.ls_onset_ms * 1e-3,
            stim_duration_s=proto.ls_duration_ms * 1e-3,
            n_repeats_averaged=proto.ls_n_repeats,
            sweep_numbers=[sweep_id], sampling_rate_Hz=proto.ls_sampling_rate_Hz,
            stimulus_name="Long Square (synthetic)"))
        sweep_id += 1
    return bundles


# ===========================================================================
#  Derived ground-truth scalars (R_in, tau_m, sag)
# ===========================================================================
def measure_rin_tau_sag(cell, gt: GroundTruthParams, probe_pA: float = -30.0
                        ) -> Tuple[float, float, float, float]:
    """Return (v_rest_mV, rin_MOhm, tau_ms, sag_ratio) by simulation.

    v_rest: settled Vm with no stimulus (= e_pas for a passive cell; may differ
            slightly with active/extra mechs).
    R_in  : (V_ss - V_rest)/I from a long probe step.
    tau_m : Rm*Cm analytically for a passive cell; for an active/I_h cell the
            single-exponential onset fit (degrades gracefully).
    sag   : (V_trough - V_ss)/(V_trough - V_rest) on the probe step
            (~0 for passive; >0 with I_h)."""
    has_dyn = bool(gt.active or gt.extra_mechs or gt.ih is not None)
    settle = 3000.0 if has_dyn else 600.0     # I_h is slow (~100s of ms); settle long
    t_ms, v = cell.simulate(stim_amp_pA=0.0, stim_delay_ms=0.0, stim_dur_ms=0.0,
                            tstop_ms=settle, v_init_mV=gt.e_pas_mV, dt_ms=0.05)
    v_rest = float(v[t_ms >= settle - 20.0].mean())

    onset, dur = 200.0, 1000.0
    t_ms, v = cell.simulate(stim_amp_pA=probe_pA, stim_delay_ms=onset,
                            stim_dur_ms=dur, tstop_ms=onset + dur,
                            v_init_mV=v_rest, dt_ms=0.05)    # init at rest (I_h-settled)
    base = float(v[(t_ms >= onset - 20.0) & (t_ms < onset)].mean())
    step = (t_ms >= onset) & (t_ms <= onset + dur)
    v_ss = float(v[(t_ms >= onset + dur - 20.0) & (t_ms <= onset + dur)].mean())
    v_trough = float(v[step].min()) if probe_pA < 0 else float(v[step].max())
    rin = (v_ss - base) / probe_pA * 1e3
    denom = (v_trough - base)
    sag = ((v_trough - v_ss) / denom) if abs(denom) > 1e-9 else 0.0
    sag = float(max(sag, 0.0))
    tau = (_fit_tau_onset(t_ms, v, base, onset) if has_dyn else gt.tau_m_ms)
    return v_rest, float(rin), float(tau), sag


def _fit_tau_onset(t_ms, v, base, onset_ms, win_ms=150.0) -> float:
    """Single-exponential charging fit V(t)=A(1-exp(-(t-onset)/tau)); analytic
    fallback if scipy is absent or the fit fails."""
    try:
        from scipy.optimize import curve_fit
        m = (t_ms >= onset_ms) & (t_ms <= onset_ms + win_ms)
        x = t_ms[m] - onset_ms
        y = v[m] - base
        A0 = y[-1] if abs(y[-1]) > 0 else (y.min() or -1.0)
        (A, tau), _ = curve_fit(lambda x, A, tau: A * (1 - np.exp(-x / tau)),
                                x, y, p0=[A0, 15.0], maxfev=20000)
        return float(tau) if np.isfinite(tau) and tau > 0 else float("nan")
    except Exception:
        return float("nan")


# ===========================================================================
#  Top-level: one cell
# ===========================================================================
def generate_synthetic_cell(
    swc_path: Path, gt: GroundTruthParams, *,
    proto: Optional[ProtocolConfig] = None, noise: Optional[NoiseConfig] = None,
    specimen_id: Optional[int] = None, passive_cell_factory: Optional[Callable] = None,
    verbose: bool = True,
) -> SyntheticCell:
    proto = proto or ProtocolConfig()
    noise = noise or NoiseConfig()
    rng = np.random.default_rng(noise.seed)
    cell = build_synthetic_cell(Path(swc_path), gt, passive_cell_factory)

    v_rest, rin, tau, sag = measure_rin_tau_sag(cell, gt)
    ss_bundles, ss_individuals = simulate_ss_protocol(cell, gt, proto, noise, rng,
                                                      v_init_mV=v_rest)
    ls_hyp = simulate_ls_protocol(cell, gt, proto, noise, rng,
                                  proto.ls_hyp_amplitudes_pA, "hyp",
                                  v_init_mV=v_rest)
    ls_dep = simulate_ls_protocol(cell, gt, proto, noise, rng,
                                  proto.ls_dep_amplitudes_pA, "dep",
                                  v_init_mV=v_rest) \
        if proto.ls_dep_amplitudes_pA else []

    sid = specimen_id if specimen_id is not None else abs(hash(str(swc_path))) % 10**9
    if verbose:
        tags = []
        if gt.ih is not None:
            tags.append(f"Ih:{gt.ih.distribution}")
        if gt.active:
            tags.append("Na/Kv")
        if gt.extra_mechs:
            tags.append("extra")
        mode = "passive+" + "+".join(tags) if tags else "passive-only"
        print(f"[synthetic] specimen {sid} ({mode}): "
              f"Cm={gt.cm_uF_cm2:.3f} Rm={gt.rm_Ohm_cm2:.0f} Ra={gt.ra_Ohm_cm:.0f} "
              f"-> Vrest={v_rest:.1f}mV Rin={rin:.1f}MOhm tau_m={tau:.1f}ms sag={sag:.3f}; "
              f"SS={len(ss_bundles)} bundles, LS_hyp={len(ls_hyp)}, LS_dep={len(ls_dep)}")
    return SyntheticCell(
        specimen_id=sid, swc_path=str(swc_path), ground_truth=gt,
        square_subthreshold=ss_bundles, long_square_subthreshold=ls_hyp,
        ss_individual_pulses=ss_individuals, rin_MOhm=rin, tau_ms=tau,
        v_rest_mV=v_rest, sag_ratio=sag, long_square_depolarising=ls_dep)


# ===========================================================================
#  Adapter: SyntheticCell -> monolith CellData (for the GP)
# ===========================================================================
def to_cell_data(syn: SyntheticCell, mono):
    """Wrap a SyntheticCell as the monolith's CellData so prepare_optimiser_inputs
    / split_train_validation_with_long_step / fit_one_cell run unchanged.
    Bundles are duck-typed (same attributes as SweepBundle)."""
    import dataclasses as _dc
    kw = dict(
        specimen_id=syn.specimen_id,
        metadata={"synthetic": True, "ground_truth": asdict(syn.ground_truth)},
        swc_path=Path(syn.swc_path),
        square_subthreshold=syn.square_subthreshold,
        long_square_subthreshold=syn.long_square_subthreshold,
        rin_MOhm=syn.rin_MOhm, tau_ms=syn.tau_ms, v_rest_mV=syn.v_rest_mV,
        ljp_correction_mV=0.0, n_avg_groups=1,
        ss_individual_pulses=syn.ss_individual_pulses)
    fields = {f.name for f in _dc.fields(mono.CellData)}
    if "sag_ratio" in fields:
        kw["sag_ratio"] = syn.sag_ratio
    return mono.CellData(**{k: v for k, v in kw.items() if k in fields})


# ===========================================================================
#  Archive writer: SyntheticCell -> Phase-0 archive (load_cell_from_archive)
# ===========================================================================
def write_archive_cell(syn: SyntheticCell, specimen_dir: Path, *,
                       layer: str = "synthetic", dendrite_type: str = "spiny",
                       verbose: bool = True) -> Path:
    """Write a SyntheticCell as a Phase-0 archive directory that
    `load_cell_from_archive` reads UNCHANGED.

    Produces, in `specimen_dir/`:
        reconstruction.swc   the morphology used (copied from syn.swc_path)
        ss_pulses.npz        stacked SS pulses  (keys: t, v, i_pA,
                             polarity_is_dep, peak_pA, stim_duration_s,
                             sampling_rate_Hz, sweep_number)
        ls_sweeps.npz        per-sweep LS arrays (keys: v_{k}, i_{k}, n_sweeps)
        metadata.json        scalars + ls_sweeps schema (+ sag_ratio + the
                             injected ground truth under full_allen_metadata)

    These keys mirror exactly what `download_allen_archive` emits and what the
    HPC reader consumes, so the synthetic data exercises the loader too -- not
    just the in-memory `to_cell_data` path. ljp_correction_mV is 0 because the
    synthetic v_mV is already true membrane voltage (no LJP)."""
    import json
    import shutil
    specimen_dir = Path(specimen_dir)
    specimen_dir.mkdir(parents=True, exist_ok=True)

    # reconstruction.swc
    src_swc = Path(syn.swc_path)
    if src_swc.exists():
        shutil.copy2(src_swc, specimen_dir / "reconstruction.swc")

    # ss_pulses.npz (stacked; all synthetic pulses share one time axis)
    pulses = syn.ss_individual_pulses
    if pulses:
        lengths = {len(p["t"]) for p in pulses}
        if len(lengths) == 1:
            ss = dict(
                t=np.asarray(pulses[0]["t"], dtype=np.float64),
                v=np.stack([p["v"] for p in pulses]).astype(np.float64),
                i_pA=np.stack([p["i"] for p in pulses]).astype(np.float64),
                polarity_is_dep=np.array([p["polarity"] == "dep" for p in pulses],
                                         dtype=bool),
                peak_pA=np.array([p["peak_pA"] for p in pulses], dtype=np.float64),
                stim_duration_s=np.array([p["stim_duration_s"] for p in pulses],
                                         dtype=np.float64),
                sampling_rate_Hz=np.array([p["sampling_rate_Hz"] for p in pulses],
                                          dtype=np.float64),
                sweep_number=np.array([p.get("sweep_number", i)
                                       for i, p in enumerate(pulses)], dtype=np.int64),
            )
            np.savez_compressed(specimen_dir / "ss_pulses.npz", **ss)

    # ls_sweeps.npz + ls_sweeps metadata (hyp bundles; dep appended if present)
    ls_all = list(syn.long_square_subthreshold) + list(syn.long_square_depolarising)
    ls_arrays: Dict[str, np.ndarray] = {}
    ls_info: List[Dict[str, Any]] = []
    for k, b in enumerate(ls_all):
        ls_arrays[f"v_{k}"] = np.asarray(b.v_mV, dtype=np.float64)
        ls_arrays[f"i_{k}"] = np.asarray(b.i_pA, dtype=np.float64)
        ls_info.append(dict(
            index=k, sweep_number=int(b.sweep_numbers[0]) if b.sweep_numbers else k,
            detected_amplitude_pA=float(b.amplitude_pA),
            sampling_rate_Hz=float(b.sampling_rate_Hz),
            stimulus_name=b.stimulus_name, n_samples=int(len(b.v_mV))))
    if ls_arrays:
        ls_arrays["n_sweeps"] = np.array([len(ls_all)], dtype=np.int64)
        np.savez_compressed(specimen_dir / "ls_sweeps.npz", **ls_arrays)

    # metadata.json
    gt = syn.ground_truth
    meta = dict(
        specimen_id=int(syn.specimen_id),
        layer=layer, dendrite_type=dendrite_type, donor_id="synthetic",
        structure_area_abbrev="synthetic",
        rin_MOhm=float(syn.rin_MOhm), tau_ms=float(syn.tau_ms),
        v_rest_mV=float(syn.v_rest_mV), sag_ratio=float(syn.sag_ratio),
        ljp_correction_mV=0.0,
        full_allen_metadata=dict(structure_layer_name=layer,
                                 dendrite_type=dendrite_type, id=int(syn.specimen_id)),
        ground_truth=asdict(gt),
        ss_extraction=dict(n_pulses_qc=len(pulses),
                           n_dep=sum(1 for p in pulses if p["polarity"] == "dep"),
                           n_hyp=sum(1 for p in pulses if p["polarity"] == "hyp"),
                           stacked=True),
        ls_sweeps=ls_info,
        smoke_test=dict(synthetic=True),
    )
    with open(specimen_dir / "metadata.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    if verbose:
        print(f"[archive] wrote {specimen_dir.name}: ss_pulses.npz "
              f"({len(pulses)} pulses), ls_sweeps.npz ({len(ls_all)} sweeps), "
              f"metadata.json (sag={syn.sag_ratio:.3f})")
    return specimen_dir


# ===========================================================================
#  Folder driver
# ===========================================================================
def generate_ground_truth_dataset(
    swc_folder: Path, gt: GroundTruthParams, *,
    proto: Optional[ProtocolConfig] = None, noise: Optional[NoiseConfig] = None,
    glob: str = "*.swc", out_dir: Optional[Path] = None,
    passive_cell_factory: Optional[Callable] = None, verbose: bool = True,
) -> Tuple[List[SyntheticCell], List[Dict[str, Any]]]:
    """Generate synthetic ground-truth data for every SWC in a folder.

    Writes (if out_dir): one pickle per cell (SyntheticCell) and a
    ground_truth.csv table of injected params + derived R_in/tau/sag, so a GP
    assessment can compare recovered vs injected per cell."""
    swc_folder = Path(swc_folder)
    swcs = sorted(swc_folder.glob(glob))
    if not swcs:
        raise FileNotFoundError(f"no SWC files matching {glob!r} in {swc_folder}")
    if out_dir is not None:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    cells: List[SyntheticCell] = []
    rows: List[Dict[str, Any]] = []
    for k, swc in enumerate(swcs):
        try:
            syn = generate_synthetic_cell(
                swc, gt, proto=proto, noise=noise, specimen_id=k,
                passive_cell_factory=passive_cell_factory, verbose=verbose)
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"[synthetic] {swc.name}: FAILED ({type(e).__name__}: {e})")
            continue
        cells.append(syn)
        rows.append(dict(
            specimen_id=syn.specimen_id, swc=swc.name,
            cm_true=gt.cm_uF_cm2, rm_true=gt.rm_Ohm_cm2, ra_true=gt.ra_Ohm_cm,
            tau_m_true_ms=gt.tau_m_ms, rin_MOhm=syn.rin_MOhm,
            tau_ms=syn.tau_ms, sag_ratio=syn.sag_ratio, v_rest_mV=syn.v_rest_mV))
        if out_dir is not None:
            with open(out_dir / f"synthetic_{syn.specimen_id}.pkl", "wb") as fh:
                pickle.dump(syn, fh)
    if out_dir is not None:
        import csv
        with open(out_dir / "ground_truth.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        if verbose:
            print(f"[synthetic] wrote {len(cells)} cell(s) + ground_truth.csv "
                  f"to {out_dir}")
    return cells, rows


# ===========================================================================
#  GP-recovery assessment harness
# ===========================================================================
def _clear_neuron_sections() -> None:
    """Delete every NEURON section (global singleton: cells must not leak across
    the generate->fit boundary, nor between cells in the loop)."""
    try:
        from neuron import h
        for s in list(h.allsec()):
            h.delete_section(sec=s)
    except Exception:
        pass


def _recovery_stats(ratios: List[float], tol: float) -> Dict[str, float]:
    """Bias (median ratio), scatter (IQR of ratio), and fraction within +/-tol."""
    a = np.asarray([r for r in ratios if np.isfinite(r)], dtype=float)
    if a.size == 0:
        return dict(n=0, median_ratio=float("nan"), iqr_ratio=float("nan"),
                    frac_within=float("nan"))
    q25, q50, q75 = np.percentile(a, [25, 50, 75])
    within = np.mean((a >= 1 - tol) & (a <= 1 + tol))
    return dict(n=int(a.size), median_ratio=float(q50),
                iqr_ratio=float(q75 - q25), frac_within=float(within))


def assess_gp_recovery(
    swcs: Sequence[Path], gt: GroundTruthParams, fit_fn: Callable, *,
    proto: Optional[ProtocolConfig] = None, noise: Optional[NoiseConfig] = None,
    passive_cell_factory: Optional[Callable] = None,
    tol: float = 0.20, out_csv: Optional[Path] = None, verbose: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, float]]]:
    """Quantify how well a GP recovers KNOWN (Cm, Rm, Ra) from synthetic data.

    For each SWC: clear NEURON -> generate synthetic data with the injected `gt`
    -> clear NEURON -> call `fit_fn(syn) -> {cm_uF_cm2, rm_Ohm_cm2, ra_Ohm_cm,
    [rin_MOhm], [tau_ms]}` (your GP) -> record recovered/true ratios. Returns
    (rows, summary) where `summary[param]` has median ratio (bias), IQR of ratio
    (scatter), and fraction within +/-`tol`.

    For PASSIVE ground truth the recovery should be tight (the model matches the
    generator). For the I_h ground truth (gt.ih set) the recovered passive
    parameters are EXPECTED to be biased -- that bias is the I_h contamination,
    and comparing the summary with vs without your sag-gated long-step guard
    (two calls, different fit_fn) is exactly the experiment that tells you
    whether the guard recovers the true passive params despite I_h.

    `fit_fn` must build its OWN (passive) fit cell from `syn.swc_path`; the
    harness clears NEURON before calling it so set_passive only touches that
    cell. Pass `passive_cell_factory` (e.g. the monolith's build_neuron_model)
    so the generating cell matches what your fitter builds, for passive ground
    truth; for I_h/active gt the bundled SyntheticPassiveCell is used."""
    proto = proto or ProtocolConfig()
    noise = noise or NoiseConfig()
    rows: List[Dict[str, Any]] = []

    for k, swc in enumerate(swcs):
        _clear_neuron_sections()
        try:
            syn = generate_synthetic_cell(
                swc, gt, proto=proto, noise=noise, specimen_id=k,
                passive_cell_factory=passive_cell_factory, verbose=False)
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"[assess] {Path(swc).name}: GENERATE failed "
                      f"({type(e).__name__}: {e})")
            continue
        _clear_neuron_sections()
        try:
            rec = fit_fn(syn)
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"[assess] {Path(swc).name}: FIT failed "
                      f"({type(e).__name__}: {e})")
            continue
        _clear_neuron_sections()

        cm, rm, ra = (float(rec["cm_uF_cm2"]), float(rec["rm_Ohm_cm2"]),
                      float(rec["ra_Ohm_cm"]))
        tau_fit = rm * cm * 1e-3
        row = dict(
            specimen_id=syn.specimen_id, swc=Path(swc).name,
            cm_true=gt.cm_uF_cm2, rm_true=gt.rm_Ohm_cm2, ra_true=gt.ra_Ohm_cm,
            tau_m_true_ms=gt.tau_m_ms, rin_true_MOhm=syn.rin_MOhm,
            cm_fit=cm, rm_fit=rm, ra_fit=ra, tau_m_fit_ms=tau_fit,
            cm_ratio=cm / gt.cm_uF_cm2, rm_ratio=rm / gt.rm_Ohm_cm2,
            ra_ratio=ra / gt.ra_Ohm_cm, tau_m_ratio=tau_fit / gt.tau_m_ms,
            sag_ratio=syn.sag_ratio)
        if "rin_MOhm" in rec and np.isfinite(rec["rin_MOhm"]) and syn.rin_MOhm:
            row["rin_fit_MOhm"] = float(rec["rin_MOhm"])
            row["rin_ratio"] = float(rec["rin_MOhm"]) / syn.rin_MOhm
        rows.append(row)
        if verbose:
            print(f"[assess] {Path(swc).name}: "
                  f"Cm x{row['cm_ratio']:.2f}  Rm x{row['rm_ratio']:.2f}  "
                  f"Ra x{row['ra_ratio']:.2f}  tau_m x{row['tau_m_ratio']:.2f}"
                  + (f"  R_in x{row['rin_ratio']:.2f}" if "rin_ratio" in row else "")
                  + f"   (sag={syn.sag_ratio:.3f})")

    keys = ["cm_ratio", "rm_ratio", "ra_ratio", "tau_m_ratio", "rin_ratio"]
    summary = {k: _recovery_stats([r[k] for r in rows if k in r], tol)
               for k in keys}
    if verbose and rows:
        print(f"\n[assess] recovery over n={len(rows)} cell(s) "
              f"(tol +/-{int(100*tol)}%):")
        for k in keys:
            s = summary[k]
            if s["n"]:
                print(f"  {k:12s}: bias(med)={s['median_ratio']:.3f}  "
                      f"scatter(IQR)={s['iqr_ratio']:.3f}  "
                      f"within={100*s['frac_within']:.0f}%  (n={s['n']})")
    if out_csv is not None and rows:
        import csv
        out_csv = Path(out_csv)
        cols = sorted({c for r in rows for c in r.keys()})
        with open(out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader(); w.writerows(rows)
        if verbose:
            print(f"[assess] wrote {out_csv}")
    return rows, summary
def write_ball_and_stick_swc(path: Path, *, soma_r_um: float = 10.0,
                             dend_len_um: float = 400.0, dend_r_um: float = 1.0,
                             apic_len_um: float = 600.0, apic_r_um: float = 1.2,
                             step_um: float = 20.0) -> Path:
    """Write a minimal soma + basal + apical SWC (single root) for Import3d."""
    path = Path(path)
    lines = ["# synthetic ball-and-stick", f"1 1 0 0 0 {soma_r_um:.3f} -1"]
    nid = 2
    parent = 1
    for d in np.arange(step_um, dend_len_um + step_um, step_um):   # basal, +x, type 3
        lines.append(f"{nid} 3 {d:.3f} 0 0 {dend_r_um:.3f} {parent}")
        parent = nid; nid += 1
    parent = 1
    for d in np.arange(step_um, apic_len_um + step_um, step_um):   # apical, +y, type 4
        lines.append(f"{nid} 4 0 {d:.3f} 0 {apic_r_um:.3f} {parent}")
        parent = nid; nid += 1
    path.write_text("\n".join(lines) + "\n")
    return path
