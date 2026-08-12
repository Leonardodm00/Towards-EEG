"""
eyal_reference_scalars.py -- NEURON-side companion to eyal_archive_builder.py

Purpose
-------
Two jobs, both requiring NEURON, both deliberately kept OUT of the pure-I/O
builder:

  1. Verify that the Neurolucida .asc morphologies import and instantiate
     (soma / apical / basal / axon present, sensible section counts).
  2. Compute the simulation-derived reference scalars requested for
     metadata.json -- principally the input resistance R_in* that the
     published triplet theta* = (Cm*, Rm*, Ra*) implies on that cell's own
     morphology.

This module builds a REFERENCE model that mirrors Eyal's own
PassiveModels/*.hoc recipe as closely as possible:

    * import via Import3d_Neurolucida3
    * geom_nseg():   nseg = 1 + 2 * int(L / 40)   on every section
    * delete_axon(): axonal sections deleted, NO replacement stub
    * biophys():     cm = CM, g_pas = 1/RM, Ra = RA, e_pas = E_PAS everywhere,
                     then cm *= F_Spines and g_pas *= F_Spines on basal and
                     apical segments whose path distance from the soma
                     exceeds StepDist = 60 um

It is NOT the pipeline's PassiveCell and must not be confused with it. Its
purpose is to give an independent, Eyal-faithful yardstick.

The spine-origin discrepancy this module measures
-------------------------------------------------
Eyal's biophys() calls ``soma distance()``, which sets the path-distance
origin at the soma's 0 end. The pipeline's ``_precompute_F_per_segment``
calls ``h.distance(0, self.soma[0](0.5))``, i.e. the soma CENTRE. The rule
is identical but the origin differs by half a soma length, so the two
codebases apply the F multiplier to slightly different sets of proximal
segments. ``compute_reference_scalars`` measures the resulting difference in
total effective membrane area so that the size of the effect is on record
before it is (or is not) acted on.

R_in convention
---------------
R_in is measured from a long hyperpolarising step:

    R_in [MOhm] = 1000 * (V_steady - V_base) [mV] / I [pA]

with I = -100 pA by default, a 500 ms step, and V_steady taken as the mean
over the final 20 ms. For a purely passive model there is no sag, so the
'peak' and 'steady' conventions coincide; this is stated because the
pipeline exposes --r-in-target peak|steady and the distinction is
meaningless here.

ASCII-only, LF-only by construction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "EyalReferenceModel",
    "build_reference_model",
    "measure_input_resistance",
    "compute_reference_scalars",
]


def _import_neuron():
    from neuron import h
    h.load_file("stdlib.hoc")
    h.load_file("import3d.hoc")
    h.load_file("stdrun.hoc")
    return h


class EyalReferenceModel(object):
    """A passive model built to Eyal's own .hoc recipe. One per process."""

    def __init__(self, morph_path: "str | Path", *,
                 cm_uF_per_cm2: float,
                 rm_Ohm_cm2: float,
                 ra_Ohm_cm: float,
                 e_pas_mV: float = -86.0,
                 F_spines: float = 1.9,
                 spine_cutoff_um: float = 60.0,
                 delete_axon: bool = True,
                 nseg_rule: str = "eyal",
                 spine_origin: str = "soma_zero"):
        self.h = _import_neuron()
        h = self.h
        self.morph_path = Path(morph_path)
        self.cm = float(cm_uF_per_cm2)
        self.rm = float(rm_Ohm_cm2)
        self.ra = float(ra_Ohm_cm)
        self.e_pas = float(e_pas_mV)
        self.F = float(F_spines)
        self.cutoff = float(spine_cutoff_um)
        self.nseg_rule = str(nseg_rule)
        self.spine_origin = str(spine_origin)

        self.soma: List[Any] = []
        self.dend: List[Any] = []
        self.apic: List[Any] = []
        self.axon: List[Any] = []

        self._import_morphology()
        self._categorise()
        self._geom_nseg()
        if delete_axon:
            self._delete_axon()
        self._biophys()

        self._iclamp = h.IClamp(self.soma[0](0.5))
        self._iclamp.delay = 0.0
        self._iclamp.dur = 0.0
        self._iclamp.amp = 0.0
        self._t_vec = h.Vector()
        self._v_vec = h.Vector()
        self._t_vec.record(h._ref_t)
        self._v_vec.record(self.soma[0](0.5)._ref_v)

    # -- construction -------------------------------------------------------
    def _import_morphology(self) -> None:
        h = self.h
        suffix = self.morph_path.suffix.lower()
        if suffix == ".asc":
            reader = h.Import3d_Neurolucida3()
        elif suffix == ".swc":
            reader = h.Import3d_SWC_read()
        else:
            raise ValueError("unsupported morphology suffix %r" % suffix)
        try:
            reader.quiet = 1
        except Exception:
            pass
        reader.input(str(self.morph_path))
        gui = h.Import3d_GUI(reader, 0)
        gui.instantiate(None)

    def _categorise(self) -> None:
        for sec in self.h.allsec():
            name = sec.name().lower()
            if "soma" in name:
                self.soma.append(sec)
            elif "apic" in name:
                self.apic.append(sec)
            elif "dend" in name:
                self.dend.append(sec)
            elif "axon" in name:
                self.axon.append(sec)
        if not self.soma:
            raise RuntimeError("no soma section after importing %s"
                               % self.morph_path.name)

    def _geom_nseg(self) -> None:
        if self.nseg_rule == "none":
            return
        if self.nseg_rule != "eyal":
            raise ValueError("nseg_rule must be 'eyal' or 'none'")
        for sec in self.h.allsec():
            sec.nseg = 1 + 2 * int(sec.L / 40.0)

    def _delete_axon(self) -> None:
        for sec in list(self.axon):
            self.h.delete_section(sec=sec)
        self.axon = []

    def _set_distance_origin(self, origin: str) -> None:
        h = self.h
        if origin == "soma_zero":
            h.distance(0, self.soma[0](0.0))     # matches 'soma distance()'
        elif origin == "soma_centre":
            h.distance(0, self.soma[0](0.5))     # matches PassiveCell
        else:
            raise ValueError("origin must be 'soma_zero' or 'soma_centre'")

    def _biophys(self) -> None:
        for sec in self.h.allsec():
            sec.insert("pas")
            sec.cm = self.cm
            sec.Ra = self.ra
            for seg in sec:
                seg.pas.g = 1.0 / self.rm
                seg.pas.e = self.e_pas
        self._set_distance_origin(self.spine_origin)
        for sec in self.dend + self.apic:
            for seg in sec:
                if self.h.distance(seg.x, sec=sec) > self.cutoff:
                    seg.cm = self.cm * self.F
                    seg.pas.g = (1.0 / self.rm) * self.F

    # -- measurements -------------------------------------------------------
    def section_counts(self) -> Dict[str, int]:
        return {"soma": len(self.soma), "dend": len(self.dend),
                "apic": len(self.apic), "axon": len(self.axon)}

    def nseg_histogram(self) -> Dict[int, int]:
        hist: Dict[int, int] = {}
        for sec in self.h.allsec():
            hist[int(sec.nseg)] = hist.get(int(sec.nseg), 0) + 1
        return dict(sorted(hist.items()))

    def total_segments(self) -> int:
        return int(sum(sec.nseg for sec in self.h.allsec()))

    def geometric_area_um2(self) -> float:
        return float(sum(seg.area() for sec in self.h.allsec() for seg in sec))

    def effective_area_um2(self, origin: str) -> float:
        """Area weighted by the per-segment spine multiplier for `origin`.

        This is the quantity that actually enters the model: multiplying cm
        by F on a segment is equivalent, for the purpose of total membrane
        capacitance, to multiplying that segment's area by F.
        """
        self._set_distance_origin(origin)
        # NOTE: identity must be compared on sec.name(), NOT id(sec).
        # h.allsec() yields a FRESH Python wrapper object for each section on
        # every iteration, so id() never matches a previously stored wrapper
        # and every membership test would silently return False -- which
        # would make this function return the geometric area unchanged and
        # report a spine-origin difference of exactly zero.
        dend_apic = set(s.name() for s in (self.dend + self.apic))
        total = 0.0
        for sec in self.h.allsec():
            in_da = sec.name() in dend_apic
            for seg in sec:
                m = 1.0
                if in_da and self.h.distance(seg.x, sec=sec) > self.cutoff:
                    m = self.F
                total += seg.area() * m
        self._set_distance_origin(self.spine_origin)   # restore
        return float(total)

    def simulate_step(self, *, amp_pA: float, delay_ms: float,
                      dur_ms: float, tstop_ms: float,
                      dt_ms: float = 0.025) -> Tuple[np.ndarray, np.ndarray]:
        h = self.h
        self._iclamp.delay = float(delay_ms)
        self._iclamp.dur = float(dur_ms)
        self._iclamp.amp = float(amp_pA) * 1e-3        # pA -> nA
        h.dt = float(dt_ms)
        h.steps_per_ms = 1.0 / float(dt_ms)
        h.tstop = float(tstop_ms)
        h.v_init = self.e_pas
        h.finitialize(self.e_pas)
        h.continuerun(float(tstop_ms))
        return (np.array(self._t_vec, dtype=np.float64),
                np.array(self._v_vec, dtype=np.float64))


def build_reference_model(morph_path, **kwargs) -> EyalReferenceModel:
    """Thin factory kept for symmetry with the pipeline's build_neuron_model."""
    return EyalReferenceModel(morph_path, **kwargs)


def measure_input_resistance(model: EyalReferenceModel, *,
                             amp_pA: float = -100.0,
                             delay_ms: float = 100.0,
                             dur_ms: float = 500.0,
                             steady_window_ms: float = 20.0,
                             dt_ms: float = 0.025) -> Dict[str, float]:
    """Somatic input resistance from a long subthreshold step.

    Returns {'rin_MOhm', 'deflection_mV', 'v_base_mV', 'v_steady_mV'}.
    """
    tstop = delay_ms + dur_ms + 50.0
    t_ms, v_mV = model.simulate_step(amp_pA=amp_pA, delay_ms=delay_ms,
                                     dur_ms=dur_ms, tstop_ms=tstop,
                                     dt_ms=dt_ms)
    base_mask = (t_ms >= delay_ms - 20.0) & (t_ms < delay_ms)
    end = delay_ms + dur_ms
    steady_mask = (t_ms >= end - steady_window_ms) & (t_ms < end)
    if base_mask.sum() < 5 or steady_mask.sum() < 5:
        raise RuntimeError("input-resistance windows are too short")
    v_base = float(v_mV[base_mask].mean())
    v_steady = float(v_mV[steady_mask].mean())
    dv = v_steady - v_base
    rin = 1000.0 * dv / float(amp_pA)      # mV / pA -> MOhm
    return {"rin_MOhm": float(rin), "deflection_mV": float(dv),
            "v_base_mV": v_base, "v_steady_mV": v_steady,
            "probe_amp_pA": float(amp_pA), "probe_dur_ms": float(dur_ms)}


def compute_reference_scalars(morph_path: "str | Path",
                              reference_published: Dict[str, float],
                              *,
                              e_pas_mV: float = -86.0,
                              F_spines: float = 1.9,
                              spine_cutoff_um: float = 60.0,
                              probe_amp_pA: float = -100.0,
                              verbose: bool = True) -> Dict[str, Any]:
    """Build the Eyal-faithful reference model and measure its scalars.

    Runs in a FRESH process ideally, because NEURON's section namespace is
    global: building two morphologies in one process leaves the first one's
    sections in h.allsec(). See run_reference_scalars_isolated().
    """
    model = build_reference_model(
        morph_path,
        cm_uF_per_cm2=float(reference_published["cm_uF_per_cm2"]),
        rm_Ohm_cm2=float(reference_published["rm_Ohm_cm2"]),
        ra_Ohm_cm=float(reference_published["ra_Ohm_cm"]),
        e_pas_mV=e_pas_mV,
        F_spines=F_spines,
        spine_cutoff_um=spine_cutoff_um,
        delete_axon=True,
        nseg_rule="eyal",
        spine_origin="soma_zero",
    )
    counts = model.section_counts()
    rin = measure_input_resistance(model, amp_pA=probe_amp_pA)

    area_geom = model.geometric_area_um2()
    area_zero = model.effective_area_um2("soma_zero")
    area_centre = model.effective_area_um2("soma_centre")
    delta_pct = 100.0 * (area_centre - area_zero) / area_zero

    out: Dict[str, Any] = {
        "rin_MOhm": rin["rin_MOhm"],
        "rin_MOhm_simulated": rin,
        "area_um2_geometric": area_geom,
        "area_um2_effective_soma_zero_origin": area_zero,
        "area_um2_effective_soma_centre_origin": area_centre,
        "spine_origin_area_delta_pct": delta_pct,
        "n_sections": counts,
        "n_segments": model.total_segments(),
        "nseg_rule": "eyal: 1 + 2*int(L/40)",
        "smoke_test": {
            "import3d_ok": True,
            "morphology_format": Path(morph_path).suffix.lower().lstrip("."),
            "n_sections": counts,
            "nseg_histogram": {str(k): v
                               for k, v in model.nseg_histogram().items()},
            "total_segments": model.total_segments(),
            "axon_deleted": counts["axon"] == 0,
        },
    }
    if verbose:
        print("[eyal_ref] %s: soma=%d dend=%d apic=%d axon=%d, segments=%d"
              % (Path(morph_path).name, counts["soma"], counts["dend"],
                 counts["apic"], counts["axon"], model.total_segments()))
        print("[eyal_ref]   Rin* = %.1f MOhm (probe %.0f pA), "
              "geometric area = %.0f um^2" % (rin["rin_MOhm"], probe_amp_pA,
                                              area_geom))
        print("[eyal_ref]   effective area: soma(0) origin = %.0f um^2, "
              "soma(0.5) origin = %.0f um^2  -> %+.3f%%"
              % (area_zero, area_centre, delta_pct))
    return out


def _scalars_worker(q, morph_path, reference_published, kwargs):
    """Module-level so it survives pickling under the 'spawn' start method."""
    try:
        q.put(("ok", compute_reference_scalars(
            morph_path, reference_published, **kwargs)))
    except Exception as exc:                           # pragma: no cover
        q.put(("err", "%s: %s" % (type(exc).__name__, exc)))


def run_reference_scalars_isolated(morph_path: "str | Path",
                                   reference_published: Dict[str, float],
                                   **kwargs) -> Dict[str, Any]:
    """Run compute_reference_scalars in a child process.

    NEURON's section list is process-global and Import3d_GUI.instantiate does
    not clear it, so building six morphologies sequentially in one process
    silently accumulates sections from every earlier cell -- which would
    corrupt every area and Rin measurement after the first. Isolating each
    cell in its own process is the only robust fix that does not depend on
    NEURON's section-deletion semantics.

    Note the worker is a module-level function and the payload is plain
    dicts/floats, so nothing NEURON-owned ever crosses the process boundary
    (Hoc objects are not picklable).
    """
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_scalars_worker,
                    args=(q, str(morph_path),
                          dict(reference_published), dict(kwargs)))
    p.start()
    status, payload = q.get()
    p.join()
    if status != "ok":
        raise RuntimeError(payload)
    return payload
