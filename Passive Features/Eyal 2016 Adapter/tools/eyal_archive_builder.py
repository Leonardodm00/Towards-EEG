"""
eyal_archive_builder.py -- Phase 0 archive builder for the Eyal et al. (2016)
human L2/3 dataset (ModelDB accession 195667).

Purpose
-------
Convert the ModelDB release of Eyal et al. (2016) into the SAME on-disk
archive format that ``download_allen_archive`` produces for the Allen Cell
Types Database, so that the existing HPC passive-fitting pipeline
(``load_cell_from_archive`` -> ``prepare_optimiser_inputs`` -> Phase 2)
can consume it with the smallest possible set of pipeline patches.

Scope / separation of concerns (deliberate)
-------------------------------------------
This module does file I/O and array arithmetic ONLY.
    * It does NOT import NEURON.
    * It does NOT fit anything.
    * It does NOT plot anything.
Reference scalars that require a simulation (input resistance) are computed
by ``eyal_reference_scalars.py`` and written back through the single
pure-I/O entry point ``backfill_reference_scalars``.

Archive layout produced
-----------------------
    <out_root>/
        manifest.json                 provenance for the whole build
        comparison_targets.csv        published (Cm*, Rm*, Ra*) per cell
        specimen_<id>/
            metadata.json
            morphology.asc            verbatim copy of the Neurolucida file
            ss_pulses.npz             Format A (stacked 2-D arrays)
    (no ls_sweeps.npz -- the Eyal release contains no Long Square data;
     its absence is deliberate and must NOT be papered over with a fake file)

Unit and time conventions (see also the project handoff, Sec. 1.1)
------------------------------------------------------------------
Raw Eyal .dat files:  t in ms, starting at 0, pulse onset at t_inj > 0,
                      V in mV, ALREADY corrected for a 16 mV liquid
                      junction potential.
Pipeline SweepBundle: t in SECONDS with t = 0 at pulse onset, so
                      pre-stimulus samples carry NEGATIVE t.

For cell c and trace file f the adapter applies, for every sample index k:

    t_k        = (t_raw_k - t_inj_c) * 1e-3                     [s]   (Eq. 1)
    i_pA_k     = A_f  if  -tol <= (t_raw_k - t_inj_c) < D - tol       (Eq. 2)
                 0    otherwise                                 [pA]
    v_mV_k     = V_k  (unchanged; NO second LJP shift)          [mV]  (Eq. 3)

with D = 2.0 ms, tol = dt/2, and A_f signed (positive = depolarising).
Equation (2) is a MODELLING ASSUMPTION (ideal rectangular pulse), not a
measurement. It is the same assumption Eyal's own Fig1cd.py makes
(``inj.amp = INJ_AMP; inj.dur = DUR``), so it introduces no discrepancy
relative to the published comparison target, but it must be reported as an
assumption.

ASCII-only, LF-only by construction (HPC transfer safety).
"""

from __future__ import annotations

import csv
import json
import re
import shutil
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "MANIFEST",
    "ARCHIVE_FORMAT_VERSION",
    "validate_manifest",
    "parse_amplitude_from_filename",
    "parse_eyal_trace",
    "build_ss_pulse_record",
    "group_pulses_by_amplitude",
    "write_specimen",
    "build_eyal_archive",
    "backfill_reference_scalars",
    "make_tarball",
]

# ---------------------------------------------------------------------------
#  Dataset constants -- every one verified byte-level against ModelDB 195667
# ---------------------------------------------------------------------------

ARCHIVE_FORMAT_VERSION = "eyal-1.0"

EYAL_MODELDB_ACCESSION = 195667
EYAL_DOI = "10.7554/eLife.16553"

EYAL_LJP_MV = 16.0            # already applied to the distributed traces
EYAL_PULSE_DURATION_MS = 2.0  # Fig1cd.py: DUR = 2
EYAL_E_PAS_MV = -86.0         # all six .hoc declare E_PAS = -86
EYAL_F_SPINES = 1.9           # all six .hoc declare F_Spines = 1.9
EYAL_SPINE_CUTOFF_UM = 60.0   # all six .hoc declare StepDist = 60
EYAL_N_SWEEPS_AVERAGED = 50   # README: "Each stimulus was repeated 50 times"

EYAL_HEADER_LINES = 2         # 'label:v(.5)' then a '<n_samples> <n_cols>' line
EYAL_EXPECTED_DT_MS = 0.02    # 50 kHz
EYAL_RECORD_END_POST_ONSET_MS = 102.0  # == DUR + 100 ms, for ALL eleven traces

# Grouping metadata written into every metadata.json so that Phase 2's
# aggregate_population() places all six cells in one population.
EYAL_LAYER = "2/3"
EYAL_DENDRITE_TYPE = "spiny"
EYAL_STRUCTURE_AREA = "TCx"   # human temporal cortex

# ---------------------------------------------------------------------------
#  MANIFEST -- the trace <-> morphology <-> onset <-> published-theta mapping.
#
#  This is DATA, not logic: swapping in a different dataset should require
#  editing this list and nothing else. Every field below was read directly
#  from the ModelDB release (Fig1/Fig1cd.py for the onsets, PassiveModels/*.hoc
#  for the published parameters), NOT transcribed from the paper.
#
#  TRACE ORDERING MATTERS. Within a cell, traces are ordered depolarising
#  first then hyperpolarising, each ascending in |A|. This makes the
#  unpatched loader's np.array_split partition amplitude-homogeneously when
#  it is called with n_avg_groups = 3 (see group_pulses_by_amplitude).
# ---------------------------------------------------------------------------

MANIFEST: List[Dict[str, Any]] = [
    {
        "cell_tag": "0603_cell08",
        "specimen_id": 60308,
        "asc": "morphs/2013_03_06_cell08_876_H41_05_Cell2.ASC",
        "hoc": "PassiveModels/model_0603_cell08.hoc",
        "t_inj_ms": 27.06,
        "depth_from_pia_um": 876,
        "paper_panel": "Fig 1a,b",
        "traces": [
            ("Fig1/Voltage_traces_1AB/p050pA_average_e86.dat", 50.0),
            ("Fig1/Voltage_traces_1AB/p100pA_average_e86.dat", 100.0),
            ("Fig1/Voltage_traces_1AB/p200pA_average_e86.dat", 200.0),
            ("Fig1/Voltage_traces_1AB/m050pA_average_e86.dat", -50.0),
            ("Fig1/Voltage_traces_1AB/m100pA_average_e86.dat", -100.0),
            ("Fig1/Voltage_traces_1AB/m200pA_average_e86.dat", -200.0),
        ],
        "reference_published": {
            "cm_uF_per_cm2": 0.45234, "rm_Ohm_cm2": 38907.0, "ra_Ohm_cm": 203.23,
        },
    },
    {
        "cell_tag": "0603_cell03",
        "specimen_id": 60303,
        "asc": "morphs/2013_03_06_cell03_789_H41_03.ASC",
        "hoc": "PassiveModels/model_0603_cell03.hoc",
        "t_inj_ms": 25.50,          # NOT a typo: INJ_0306_cell03 in Fig1cd.py
        "depth_from_pia_um": 789,
        "paper_panel": "Fig 1c2",
        "traces": [
            ("Fig1/Voltage_traces_1CD/0306_cell03_200pA_average_e86.dat", 200.0),
        ],
        "reference_published": {
            "cm_uF_per_cm2": 0.488, "rm_Ohm_cm2": 21406.0, "ra_Ohm_cm": 281.78,
        },
    },
    {
        "cell_tag": "0603_cell11",
        "specimen_id": 60311,
        "asc": "morphs/2013_03_06_cell11_1125_H41_06.ASC",
        "hoc": "PassiveModels/model_0603_cell11.hoc",
        "t_inj_ms": 27.06,
        "depth_from_pia_um": 1125,
        "paper_panel": "Fig 1c3",
        "traces": [
            ("Fig1/Voltage_traces_1CD/0306_cell11_200pA_average_e86.dat", 200.0),
        ],
        "reference_published": {
            "cm_uF_per_cm2": 0.44, "rm_Ohm_cm2": 48730.0, "ra_Ohm_cm": 261.97,
        },
    },
    {
        "cell_tag": "1303_cell03",
        "specimen_id": 130303,
        "asc": "morphs/2013_03_13_cell03_1204_H42_02.ASC",
        "hoc": "PassiveModels/model_1303_cell03.hoc",
        "t_inj_ms": 27.06,
        "depth_from_pia_um": 1204,
        "paper_panel": "Fig 1c1",
        "traces": [
            ("Fig1/Voltage_traces_1CD/0313_cell03_200pA_average_e86.dat", 200.0),
        ],
        "reference_published": {
            "cm_uF_per_cm2": 0.43, "rm_Ohm_cm2": 39360.0, "ra_Ohm_cm": 262.54,
        },
    },
    {
        "cell_tag": "1303_cell05",
        "specimen_id": 130305,
        "asc": "morphs/2013_03_13_cell05_675_H42_04.ASC",
        "hoc": "PassiveModels/model_1303_cell05.hoc",
        "t_inj_ms": 27.06,
        "depth_from_pia_um": 675,
        "paper_panel": "Fig 1c5",
        "traces": [
            ("Fig1/Voltage_traces_1CD/0313_cell05_200pA_average_e86.dat", 200.0),
        ],
        "reference_published": {
            "cm_uF_per_cm2": 0.49675, "rm_Ohm_cm2": 31314.0, "ra_Ohm_cm": 292.95,
        },
    },
    {
        "cell_tag": "1303_cell06",
        "specimen_id": 130306,
        "asc": "morphs/2013_03_13_cell06_945_H42_05.ASC",
        "hoc": "PassiveModels/model_1303_cell06.hoc",
        "t_inj_ms": 27.06,
        "depth_from_pia_um": 945,
        "paper_panel": "Fig 1c4",
        "traces": [
            ("Fig1/Voltage_traces_1CD/0313_cell06_200pA_average_e86.dat", 200.0),
        ],
        "reference_published": {
            "cm_uF_per_cm2": 0.52, "rm_Ohm_cm2": 36519.0, "ra_Ohm_cm": 290.0,
        },
    },
]


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder handling numpy scalars; NaN -> null."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            v = float(obj)
            return None if np.isnan(v) else v
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


# ---------------------------------------------------------------------------
#  Manifest validation
# ---------------------------------------------------------------------------

def validate_manifest(eyal_root: Path,
                      manifest: Sequence[Dict[str, Any]] = MANIFEST) -> None:
    """Fail loudly if the manifest and the on-disk release disagree.

    Checks structural invariants that a silent transcription error in the
    manifest would violate. A wrong trace-to-morphology pairing produces a
    plausible-looking but meaningless fit, so this runs before anything else.
    """
    eyal_root = Path(eyal_root)
    if len(manifest) != 6:
        raise ValueError("expected 6 cells in MANIFEST, got %d" % len(manifest))

    ids = [e["specimen_id"] for e in manifest]
    if len(set(ids)) != len(ids):
        raise ValueError("specimen_id values are not unique: %r" % (ids,))

    tags = [e["cell_tag"] for e in manifest]
    if len(set(tags)) != len(tags):
        raise ValueError("cell_tag values are not unique: %r" % (tags,))

    onsets = [e["t_inj_ms"] for e in manifest]
    n_late = sum(1 for t in onsets if abs(t - 27.06) < 1e-9)
    n_early = sum(1 for t in onsets if abs(t - 25.50) < 1e-9)
    if n_late != 5 or n_early != 1:
        raise ValueError(
            "expected exactly five cells at t_inj = 27.06 ms and one at "
            "25.50 ms, got %d and %d" % (n_late, n_early))
    early = [e["cell_tag"] for e in manifest if abs(e["t_inj_ms"] - 25.50) < 1e-9]
    if early != ["0603_cell03"]:
        raise ValueError(
            "the 25.50 ms onset belongs to 0603_cell03, not %r" % (early,))

    n_traces = 0
    for e in manifest:
        for rel in (e["asc"], e["hoc"]):
            if not (eyal_root / rel).is_file():
                raise FileNotFoundError(str(eyal_root / rel))
        if not e["traces"]:
            raise ValueError("cell %s has no traces" % e["cell_tag"])
        for rel, amp in e["traces"]:
            p = eyal_root / rel
            if not p.is_file():
                raise FileNotFoundError(str(p))
            parsed = parse_amplitude_from_filename(p.name)
            if parsed is not None and abs(parsed - amp) > 1e-9:
                raise ValueError(
                    "amplitude mismatch for %s: manifest says %+.1f pA, "
                    "filename implies %+.1f pA" % (p.name, amp, parsed))
            n_traces += 1
        ref = e["reference_published"]
        for key in ("cm_uF_per_cm2", "rm_Ohm_cm2", "ra_Ohm_cm"):
            if not (float(ref[key]) > 0.0):
                raise ValueError("non-positive %s for %s" % (key, e["cell_tag"]))

    if n_traces != 11:
        raise ValueError("expected 11 trace files in total, got %d" % n_traces)


# ---------------------------------------------------------------------------
#  Parsing
# ---------------------------------------------------------------------------

_AMP_PREFIXED = re.compile(r"^([pm])(\d{3})pA")
_AMP_PLAIN = re.compile(r"_(\d{2,4})pA")


def parse_amplitude_from_filename(name: str) -> Optional[float]:
    """Signed pulse amplitude in pA implied by an Eyal trace filename.

    Two naming conventions exist in the release:
      'p200pA_average_e86.dat'              -> +200 pA (Voltage_traces_1AB)
      '0306_cell03_200pA_average_e86.dat'   -> +200 pA (Voltage_traces_1CD;
          no polarity prefix, and the README states these are the 200 pA
          DEPOLARISING traces)

    Returns None if the name matches neither convention. Used only as a
    cross-check on the manifest, which is authoritative.
    """
    m = _AMP_PREFIXED.match(name)
    if m is not None:
        sign = 1.0 if m.group(1) == "p" else -1.0
        return sign * float(int(m.group(2)))
    m = _AMP_PLAIN.search(name)
    if m is not None:
        return float(int(m.group(1)))
    return None


def parse_eyal_trace(path: Path) -> Dict[str, Any]:
    """Read one Eyal .dat voltage trace.

    Returns
    -------
    dict with keys 't_ms', 'v_mV', 'dt_ms', 'n_samples', 'source'.

    The sampling interval is derived as (t[-1] - t[0]) / (N - 1) rather than
    from np.diff, so that a single duplicated or dropped sample cannot be
    masked by a median.
    """
    path = Path(path)
    arr = np.loadtxt(path, skiprows=EYAL_HEADER_LINES)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("%s: expected 2 columns, got shape %r"
                         % (path.name, arr.shape))
    t_ms = np.ascontiguousarray(arr[:, 0], dtype=np.float64)
    v_mV = np.ascontiguousarray(arr[:, 1], dtype=np.float64)
    n = int(t_ms.size)
    if n < 100:
        raise ValueError("%s: only %d samples" % (path.name, n))
    if not np.all(np.diff(t_ms) > 0):
        raise ValueError("%s: time axis is not strictly increasing" % path.name)

    dt_ms = float((t_ms[-1] - t_ms[0]) / (n - 1))
    if abs(dt_ms - EYAL_EXPECTED_DT_MS) > 1e-6:
        raise ValueError("%s: dt = %.6f ms, expected %.6f ms"
                         % (path.name, dt_ms, EYAL_EXPECTED_DT_MS))
    return {
        "t_ms": t_ms,
        "v_mV": v_mV,
        "dt_ms": dt_ms,
        "n_samples": n,
        "source": path.name,
    }


def build_ss_pulse_record(trace: Dict[str, Any], *,
                          t_inj_ms: float,
                          amp_pA: float,
                          dur_ms: float = EYAL_PULSE_DURATION_MS,
                          sweep_number: int = 0) -> Dict[str, Any]:
    """Turn one parsed trace into one pipeline-shaped SS pulse record.

    Applies Eq. (1) re-zero + unit conversion, Eq. (2) current synthesis and
    Eq. (3) LJP pass-through (no second shift). The returned dict has exactly
    the keys that ``load_cell_from_archive`` reconstructs per pulse.
    """
    t_ms = trace["t_ms"]
    v_mV = trace["v_mV"]
    dt_ms = float(trace["dt_ms"])

    # -- Eq. (1): re-zero on the ms axis, then convert to seconds ----------
    t_rel_ms = t_ms - float(t_inj_ms)
    t_s = t_rel_ms * 1e-3

    # -- Eq. (2): ideal rectangular pulse on the SAME time base ------------
    # Half-sample tolerance so that floating-point noise in t_rel_ms cannot
    # add or drop a boundary sample. Selects exactly the samples whose
    # nominal time lies in {0, dt, ..., dur - dt}.
    tol = 0.5 * dt_ms
    on = (t_rel_ms >= -tol) & (t_rel_ms < float(dur_ms) - tol)
    i_pA = np.where(on, float(amp_pA), 0.0).astype(np.float64)

    n_expected = int(round(float(dur_ms) / dt_ms))
    if int(on.sum()) != n_expected:
        raise ValueError(
            "%s: synthesised current has %d active samples, expected %d"
            % (trace["source"], int(on.sum()), n_expected))

    return {
        "t": t_s,
        "v": np.array(v_mV, dtype=np.float64, copy=True),   # Eq. (3)
        "i": i_pA,
        "polarity": "dep" if amp_pA > 0 else "hyp",
        "peak_pA": float(amp_pA),
        "stim_duration_s": float(dur_ms) * 1e-3,
        "sampling_rate_Hz": float(1.0 / (dt_ms * 1e-3)),
        "sweep_number": int(sweep_number),
        "source": trace["source"],
    }


def group_pulses_by_amplitude(pulses: Sequence[Dict[str, Any]], *,
                              tol_pA: float = 1.0
                              ) -> List[List[int]]:
    """Partition a pulse pool into amplitude-homogeneous groups.

    REFERENCE IMPLEMENTATION for the loader patch. ``load_cell_from_archive``
    currently groups Square-Subthreshold pulses by POLARITY only and averages
    within a polarity. That is safe for Allen (every SS pulse is QC-filtered
    to |A| within 30 pA of 200 pA) but WRONG for cell 0603_cell08, whose
    depolarising pool contains +50, +100 and +200 pA: averaging them yields a
    bundle labelled +116.7 pA that corresponds to no experiment.

    Returns a list of index lists, ordered by polarity then ascending |A|.
    """
    order = sorted(range(len(pulses)),
                   key=lambda k: (pulses[k]["polarity"],
                                  abs(float(pulses[k]["peak_pA"]))))
    groups: List[List[int]] = []
    for k in order:
        placed = False
        for g in groups:
            ref = pulses[g[0]]
            if (ref["polarity"] == pulses[k]["polarity"]
                    and abs(float(ref["peak_pA"])
                            - float(pulses[k]["peak_pA"])) <= tol_pA):
                g.append(k)
                placed = True
                break
        if not placed:
            groups.append([k])
    return groups


# ---------------------------------------------------------------------------
#  Writing
# ---------------------------------------------------------------------------

def _tau_m_analytic_ms(cm_uF_per_cm2: float, rm_Ohm_cm2: float) -> float:
    """tau_m = Cm * Rm, in ms.

    1 uF/cm^2 * 1 Ohm.cm^2 = 1 us, hence the 1e-3 factor.

    NOTE: tau_m is INVARIANT under the global spine correction. The spine
    factor F multiplies cm and divides Rm on the same segments, so
    (F * Cm) * (Rm / F) = Cm * Rm. The value is therefore well defined
    without reference to F.
    """
    return float(cm_uF_per_cm2) * float(rm_Ohm_cm2) * 1e-3


def write_specimen(out_root: Path,
                   entry: Dict[str, Any],
                   pulses: Sequence[Dict[str, Any]],
                   eyal_root: Path,
                   *,
                   write_reference_scalars: bool = True,
                   verbose: bool = True) -> Path:
    """Write one ``specimen_<id>/`` directory. Pure I/O."""
    out_root = Path(out_root)
    eyal_root = Path(eyal_root)
    sid = int(entry["specimen_id"])
    cell_dir = out_root / ("specimen_%d" % sid)
    cell_dir.mkdir(parents=True, exist_ok=True)

    if not pulses:
        raise ValueError("cell %s: no pulses to write" % entry["cell_tag"])

    # -- all pulses of one specimen must share the time base (Format A) ----
    t_ref = pulses[0]["t"]
    for p in pulses[1:]:
        if p["t"].shape != t_ref.shape or not np.allclose(p["t"], t_ref,
                                                          rtol=0, atol=1e-12):
            raise ValueError(
                "cell %s: trace %s does not share the time base of %s; "
                "Format A requires a single shared t vector"
                % (entry["cell_tag"], p["source"], pulses[0]["source"]))

    # -- morphology --------------------------------------------------------
    morph_src = eyal_root / entry["asc"]
    morph_dst = cell_dir / "morphology.asc"
    shutil.copy2(morph_src, morph_dst)

    # -- ss_pulses.npz (Format A: stacked 2-D) ------------------------------
    ss_data = {
        "t": t_ref.astype(np.float64),
        "v": np.stack([p["v"] for p in pulses]).astype(np.float64),
        "i_pA": np.stack([p["i"] for p in pulses]).astype(np.float64),
        "polarity_is_dep": np.array([p["polarity"] == "dep" for p in pulses],
                                    dtype=bool),
        "peak_pA": np.array([p["peak_pA"] for p in pulses], dtype=np.float64),
        "stim_duration_s": np.array([p["stim_duration_s"] for p in pulses],
                                    dtype=np.float64),
        "sampling_rate_Hz": np.array([p["sampling_rate_Hz"] for p in pulses],
                                     dtype=np.float64),
        "sweep_number": np.array([p["sweep_number"] for p in pulses],
                                 dtype=np.int64),
    }
    np.savez_compressed(cell_dir / "ss_pulses.npz", **ss_data)

    # -- reference scalars --------------------------------------------------
    ref = entry["reference_published"]
    tau_star_ms = _tau_m_analytic_ms(ref["cm_uF_per_cm2"], ref["rm_Ohm_cm2"])

    # WARNING, recorded in the file itself: tau_ms below is DERIVED from the
    # comparison target theta*, not measured independently. Any validation
    # gate that compares a fitted tau against it is CIRCULAR. It is written
    # because it was explicitly requested as a surrogate; the
    # 'reference_scalars_are_derived' flag exists so downstream code and
    # write-ups can see that at a glance.
    rin_MOhm = None
    tau_ms = tau_star_ms if write_reference_scalars else None

    amps = sorted(set(round(float(p["peak_pA"]), 3) for p in pulses))
    n_dep = sum(1 for p in pulses if p["polarity"] == "dep")

    metadata: Dict[str, Any] = {
        "specimen_id": sid,
        "cell_tag": entry["cell_tag"],
        "layer": EYAL_LAYER,
        "dendrite_type": EYAL_DENDRITE_TYPE,
        "donor_id": "eyal2016_not_disclosed",
        "structure_area_abbrev": EYAL_STRUCTURE_AREA,
        "species": "Homo Sapiens",

        # -- morphology: NOT an SWC. The loader must read this key rather
        #    than hard-coding 'reconstruction.swc'. Allen archives lack the
        #    key, so a .get(..., 'reconstruction.swc') default keeps Allen
        #    behaviour bit-identical.
        "morphology_file": "morphology.asc",
        "morphology_format": "neurolucida_asc",

        # -- reference scalars consumed by CellData / the +-20% gate --------
        "rin_MOhm": rin_MOhm,
        "tau_ms": tau_ms,
        "v_rest_mV": EYAL_E_PAS_MV,
        "ljp_correction_mV": EYAL_LJP_MV,

        "reference_scalars_are_derived": bool(write_reference_scalars),
        "reference_scalar_provenance": {
            "tau_ms": ("derived as Cm* x Rm* from the published triplet; "
                       "NOT an independent measurement; using it as a "
                       "validation gate is circular"),
            "rin_MOhm": ("null until backfilled by "
                         "eyal_reference_scalars.compute_reference_scalars, "
                         "which simulates the published triplet on this "
                         "cell's own morphology; also NOT independent"),
            "v_rest_mV": ("measured from the trace baselines and equal to "
                          "E_PAS in the published .hoc; this one IS "
                          "independent of the fitted parameters"),
        },

        "reference_published": {
            "cm_uF_per_cm2": float(ref["cm_uF_per_cm2"]),
            "rm_Ohm_cm2": float(ref["rm_Ohm_cm2"]),
            "ra_Ohm_cm": float(ref["ra_Ohm_cm"]),
            "tau_m_analytic_ms": tau_star_ms,
            "source_hoc": entry["hoc"],
            "paper_panel": entry["paper_panel"],
            "F_spines": EYAL_F_SPINES,
            "spine_cutoff_um": EYAL_SPINE_CUTOFF_UM,
            "spine_distance_origin": "soma(0) -- 'soma distance()' in the .hoc",
        },

        # -- what fit_one_cell / aggregate_population read -------------------
        "full_allen_metadata": {
            "id": sid,
            "structure_layer_name": EYAL_LAYER,
            "dendrite_type": EYAL_DENDRITE_TYPE,
            "structure_area_abbrev": EYAL_STRUCTURE_AREA,
            "donor__species": "Homo Sapiens",
        },

        "ss_extraction": {
            "n_pulses_qc": len(pulses),
            "n_dep": n_dep,
            "n_hyp": len(pulses) - n_dep,
            "distinct_amplitudes_pA": amps,
            "multi_amplitude": len(amps) > 2,
            "stacked": True,
            "sources": [p["source"] for p in pulses],
            "n_sweeps_averaged_by_authors": EYAL_N_SWEEPS_AVERAGED,
            "individual_sweeps_available": False,
        },

        # Deliberately empty: the release has no Long Square data. Loaders
        # must be called with require_long_square=False rather than being
        # fed a synthetic ls_sweeps.npz.
        "ls_sweeps": [],

        "eyal_provenance": {
            "doi": EYAL_DOI,
            "modeldb_accession": EYAL_MODELDB_ACCESSION,
            "t_inj_ms": float(entry["t_inj_ms"]),
            "pulse_duration_ms": EYAL_PULSE_DURATION_MS,
            "depth_from_pia_um": entry.get("depth_from_pia_um"),
            "ljp_note": ("traces are distributed ALREADY corrected for a "
                         "16 mV LJP; ljp_correction_mV records what was "
                         "applied and must NOT be applied again"),
            "current_waveform_note": ("i_pA is SYNTHESISED as an ideal "
                                      "rectangular pulse; the release "
                                      "distributes voltage only. Same "
                                      "assumption as Fig1cd.py."),
            "bridge_blanking_ms_post_onset": (
                4.0 if entry["cell_tag"] == "0603_cell03" else 3.0),
            "record_end_post_onset_ms": EYAL_RECORD_END_POST_ONSET_MS,
            "eyal_fit_window_ms_post_onset": [3.0, 102.0],
        },

        "smoke_test": {},
    }

    with open(cell_dir / "metadata.json", "w", encoding="ascii") as f:
        json.dump(metadata, f, indent=2, cls=_NumpyEncoder)

    if verbose:
        print("[eyal_archive] %-12s -> specimen_%d  (%d pulse(s), amps %s pA, "
              "t_inj=%.2f ms)" % (entry["cell_tag"], sid, len(pulses),
                                  amps, entry["t_inj_ms"]))
    return cell_dir


def build_eyal_archive(eyal_root: "str | Path",
                       out_root: "str | Path",
                       *,
                       manifest: Sequence[Dict[str, Any]] = MANIFEST,
                       write_reference_scalars: bool = True,
                       verbose: bool = True) -> List[Path]:
    """Build the whole archive. Returns the list of specimen directories."""
    eyal_root = Path(eyal_root)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    validate_manifest(eyal_root, manifest)
    if verbose:
        print("[eyal_archive] manifest validated against %s" % eyal_root)

    dirs: List[Path] = []
    results: List[Dict[str, Any]] = []
    for entry in manifest:
        pulses = []
        for k, (rel, amp) in enumerate(entry["traces"]):
            trace = parse_eyal_trace(eyal_root / rel)
            pulses.append(build_ss_pulse_record(
                trace,
                t_inj_ms=entry["t_inj_ms"],
                amp_pA=amp,
                dur_ms=EYAL_PULSE_DURATION_MS,
                sweep_number=k,
            ))
        d = write_specimen(out_root, entry, pulses, eyal_root,
                           write_reference_scalars=write_reference_scalars,
                           verbose=verbose)
        dirs.append(d)
        results.append({
            "specimen_id": int(entry["specimen_id"]),
            "cell_tag": entry["cell_tag"],
            "n_pulses": len(pulses),
            "status": "ok",
        })

    # -- top-level manifest -------------------------------------------------
    manifest_json = {
        "created": datetime.now().isoformat(),
        "archive_format_version": ARCHIVE_FORMAT_VERSION,
        "dataset": "Eyal et al. 2016, human L2/3 pyramidal cells",
        "doi": EYAL_DOI,
        "modeldb_accession": EYAL_MODELDB_ACCESSION,
        "parameters": {
            "pulse_duration_ms": EYAL_PULSE_DURATION_MS,
            "ljp_correction_mV": EYAL_LJP_MV,
            "v_rest_mV": EYAL_E_PAS_MV,
            "F_spines": EYAL_F_SPINES,
            "spine_cutoff_um": EYAL_SPINE_CUTOFF_UM,
            "reference_scalars_written": bool(write_reference_scalars),
        },
        "known_gaps": [
            "no Long Square sweeps: load with require_long_square=False and "
            "run with --n-long-train 0",
            "no individual (pre-average) sweeps: the Phase 3 nonparametric "
            "bootstrap is degenerate; use --phase3-subset none",
            "no independent Rin / tau_m: the values written are derived from "
            "the comparison target and are therefore circular as gates",
            "morphology is Neurolucida .asc, not .swc",
        ],
        "cells": results,
    }
    with open(out_root / "manifest.json", "w", encoding="ascii") as f:
        json.dump(manifest_json, f, indent=2, cls=_NumpyEncoder)

    # -- comparison targets, as a flat CSV for the eventual write-up --------
    with open(out_root / "comparison_targets.csv", "w",
              encoding="ascii", newline="") as f:
        w = csv.writer(f)
        w.writerow(["specimen_id", "cell_tag", "paper_panel",
                    "cm_star_uF_per_cm2", "rm_star_Ohm_cm2", "ra_star_Ohm_cm",
                    "tau_m_star_ms", "depth_from_pia_um"])
        for e in manifest:
            r = e["reference_published"]
            w.writerow([e["specimen_id"], e["cell_tag"], e["paper_panel"],
                        r["cm_uF_per_cm2"], r["rm_Ohm_cm2"], r["ra_Ohm_cm"],
                        round(_tau_m_analytic_ms(r["cm_uF_per_cm2"],
                                                 r["rm_Ohm_cm2"]), 4),
                        e.get("depth_from_pia_um")])

    if verbose:
        print("[eyal_archive] DONE -- %d specimen(s) written to %s"
              % (len(dirs), out_root))
    return dirs


def backfill_reference_scalars(specimen_dir: "str | Path",
                               scalars: Dict[str, Any],
                               *, verbose: bool = True) -> None:
    """Merge simulation-derived reference scalars into an existing metadata.json.

    Pure I/O: the caller (eyal_reference_scalars.py) does the NEURON work.
    """
    specimen_dir = Path(specimen_dir)
    meta_path = specimen_dir / "metadata.json"
    with open(meta_path, "r", encoding="ascii") as f:
        meta = json.load(f)

    if "rin_MOhm" in scalars and scalars["rin_MOhm"] is not None:
        meta["rin_MOhm"] = float(scalars["rin_MOhm"])
    meta.setdefault("reference_published", {})
    for key in ("rin_MOhm_simulated", "area_um2_geometric",
                "area_um2_effective_soma_centre_origin",
                "area_um2_effective_soma_zero_origin",
                "spine_origin_area_delta_pct",
                "n_sections", "n_segments", "nseg_rule"):
        if key in scalars:
            meta["reference_published"][key] = scalars[key]
    meta.setdefault("smoke_test", {})
    meta["smoke_test"].update(scalars.get("smoke_test", {}))

    with open(meta_path, "w", encoding="ascii") as f:
        json.dump(meta, f, indent=2, cls=_NumpyEncoder)
    if verbose:
        print("[eyal_archive] backfilled reference scalars into %s"
              % meta_path.name)


def make_tarball(archive_root: "str | Path",
                 out_path: "str | Path",
                 *, verbose: bool = True) -> Path:
    """Pack the archive as a binary tar.gz for transfer to the cluster.

    Multi-file deliveries must travel as a binary archive, never as pasted
    text, so that no Windows/MobaXterm boundary can rewrite line endings or
    re-encode bytes.
    """
    archive_root = Path(archive_root)
    out_path = Path(out_path)
    with tarfile.open(out_path, "w:gz") as tar:
        tar.add(archive_root, arcname=archive_root.name)
    if verbose:
        print("[eyal_archive] tarball -> %s (%.1f MB)"
              % (out_path, out_path.stat().st_size / 1e6))
    return out_path
