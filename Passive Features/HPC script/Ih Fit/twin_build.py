# -*- coding: utf-8 -*-
"""
twin_build.py -- the twin build: the fitter's own model at the truth, clean responses plus noise, an archive cell (Block 12)
==========================================================================================================================

specs/SPEC.md, section 2.7 (c), (d), (f), (h) and the entry of Block 12
(D-183, D-186, D-187, D-190, D-192); the twin without noise of 2.8 (e) and
the entry of Block 18 (D-207). NEURON side. Three separate concerns,
three functions: the cell (build_twin_cell), the traces (simulate_twin), the
archive (write_twin_archive); build_twin chains them for one twin.

What it does
    build_twin_cell     (TC1): passive_fitting_hpc_fixed.build_neuron_model,
                        ih_mechanism.attach_ih (twins with I_h), the fit's
                        ParamSpec.apply at theta(g) and V_rest(g); then the
                        rest guard ih_mechanism.rest_drift <= rest_drift_max_mV
    simulate_twin       (TW1) for every stored SS window: Block 11's
                        simulate_ss_clean, one run per polarity at A_p, d_p,
                        plus Block 10's ss_sweep_noise; (TW2) for every LS
                        sweep with in_twin True: simulate_ls_clean, one run per
                        distinct (amplitude, onset, duration, length), plus
                        ls_sweep_noise
    write_twin_archive  the archive cell in the layout the fitter's loader
                        reads (as synthetic_ground_truth.write_archive_cell)
    build_twin          the whole build of one twin; with_noise=False (Block 18,
                        SPEC 2.8 (e)) writes the clean responses alone

This module imports no NEURON at module scope (twin_manifest imports
TwinSpec from it): the fitter's monolith, ih_mechanism and
synthetic_ground_truth are imported inside the functions that need them.

Units: mV, pA, s, Hz; NEURON's step and the rest-drift run in ms.
Custom code: none (compositions and a writer).
"""
from __future__ import annotations

import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

import twin_noise
import twin_table
from twin_config import TwinConfig

CONFIGURATIONS = ("full", "passive_fit", "active_fit", "full_no_noise")
HAS_IH = {"full": True, "passive_fit": True, "active_fit": False, "full_no_noise": True}
PASSIVE_KEYS = ("Cm", "Rm", "Ra")
IH_KEYS = ("gbar", "dv_h", "kappa_tau")

SS_NPZ_KEYS = ("t", "v", "i_pA", "polarity_is_dep", "peak_pA", "stim_duration_s", "sampling_rate_Hz",
               "sweep_number")
_PROTOCOLS = ("SS", "LS")
# The `twin` block's record of where tau_ms comes from (amendment 6 (iii), D-193).
TAU_MS_SOURCE = "Rm*Cm of the truth"


class TwinBuildError(RuntimeError):
    """A twin the build refuses (SPEC, Block 12, "Refused")."""


@dataclass(frozen=True)
class TwinSpec:
    """One twin (one manifest row, resolved).

    twin_id             : the twin's own id (int), distinct from every real id
    twin_of_specimen_id : the real cell it is the twin of (int)
    configuration       : 'full' | 'passive_fit' | 'active_fit' |
                          'full_no_noise' (I_h as 'full'; Block 18)
    theta               : the truth in the fitter's names -- Cm (uF/cm^2),
                          Rm (Ohm cm^2), Ra (Ohm cm), and when
                          HAS_IH[configuration] gbar (S/cm^2), dv_h (mV),
                          kappa_tau (dimensionless); no other key
    v_rest_mV           : V_rest(g), the real cell's resting potential as the
                          fitter's loader returns it (mV)
    F                   : the spine factor of the fit
    """
    twin_id: int
    twin_of_specimen_id: int
    configuration: str
    theta: Mapping[str, float]
    v_rest_mV: float
    F: float


@dataclass(frozen=True)
class TwinTraces:
    """The traces of one twin (mV, pA, s, Hz; float64 unless stated).

    SS, per stored window of the real cell in the archive's order, (K,):
        ss_v_mV (K, n_w), ss_i_pA (K, n_w), ss_polarity_is_dep (bool),
        ss_peak_pA (A_p of the window's polarity), ss_stim_duration_s (d_p),
        ss_sweep_number (int64, the real sweep numbers); ss_t_s (n_w,) s, 0 at
        the pulse onset; ss_fs_Hz.
    LS, per LS sweep with in_twin True in CellProtocol's order, (M,):
        ls_sweep_number (int64, real), ls_amplitude_pA; ls_v_mV and ls_i_pA
        tuples of M arrays (n(c, s),); ls_fs_Hz.
    """
    ss_fs_Hz: float
    ss_t_s: np.ndarray
    ss_v_mV: np.ndarray
    ss_i_pA: np.ndarray
    ss_polarity_is_dep: np.ndarray
    ss_peak_pA: np.ndarray
    ss_stim_duration_s: np.ndarray
    ss_sweep_number: np.ndarray
    ls_fs_Hz: float
    ls_sweep_number: np.ndarray
    ls_amplitude_pA: np.ndarray
    ls_v_mV: Tuple[np.ndarray, ...]
    ls_i_pA: Tuple[np.ndarray, ...]


# ===========================================================================
#  Checks shared by the public functions
# ===========================================================================
def _check_spec(spec: TwinSpec) -> Dict[str, float]:
    """The refusals that concern the spec alone; returns theta as floats."""
    if spec.configuration not in CONFIGURATIONS:
        raise TwinBuildError("twin %r: unknown configuration %r (expected one of %s)"
                             % (spec.twin_id, spec.configuration, CONFIGURATIONS))
    if int(spec.twin_id) == int(spec.twin_of_specimen_id):
        raise TwinBuildError("twin %d: its id equals its real cell's id" % int(spec.twin_id))
    want = set(PASSIVE_KEYS) | (set(IH_KEYS) if HAS_IH[spec.configuration] else set())
    got = set(spec.theta.keys())
    if got != want:
        raise TwinBuildError("twin %d (%s): theta keys %s, expected exactly %s (missing %s, extra %s)"
                             % (int(spec.twin_id), spec.configuration, sorted(got), sorted(want),
                                sorted(want - got), sorted(got - want)))
    theta: Dict[str, float] = {}
    for k in sorted(want):
        try:
            val = float(spec.theta[k])
        except (TypeError, ValueError):
            raise TwinBuildError("twin %d: theta[%r] = %r is not a number" % (int(spec.twin_id), k, spec.theta[k]))
        if not math.isfinite(val):
            raise TwinBuildError("twin %d: theta[%r] = %r is not finite" % (int(spec.twin_id), k, val))
        theta[k] = val
    for name in ("v_rest_mV", "F"):
        val = float(getattr(spec, name))
        if not math.isfinite(val):
            raise TwinBuildError("twin %d: %s = %r is not finite" % (int(spec.twin_id), name, val))
    return theta


def _check_with_noise(with_noise, twin_id) -> bool:
    """with_noise as a Python bool; a value that is not a bool (bool or
    numpy.bool_) is refused, so that a string such as "False" read from a
    CSV cannot build a noisy twin."""
    if not isinstance(with_noise, (bool, np.bool_)):
        raise TwinBuildError("twin %d: with_noise = %r is not a bool" % (int(twin_id), with_noise))
    return bool(with_noise)


def _check_table(table: pd.DataFrame, protocol, specimen: int) -> pd.DataFrame:
    """The real cell's rows of the twins' table, checked against the
    CellProtocol read from its archive: the same SS sweeps with the same
    window counts, the same LS sweeps with the same amplitude, onset,
    duration, length and rate, the same real_spike_reason, and in_twin equal
    to (real_spike_reason == ''). Returns the rows."""
    if int(protocol.specimen_id) != int(specimen):
        raise TwinBuildError("the CellProtocol is of cell %d, the twin is of cell %d"
                             % (int(protocol.specimen_id), int(specimen)))
    rows = table[table["specimen_id"].astype(np.int64) == int(specimen)]
    if len(rows) == 0:
        raise TwinBuildError("cell %d: no row in the twins' table" % int(specimen))
    bad_pr = sorted(set(str(p) for p in rows["protocol"]) - set(_PROTOCOLS))
    if bad_pr:
        raise TwinBuildError("cell %d: protocol(s) %s in the twins' table, expected SS / LS" % (specimen, bad_pr))
    for pr in _PROTOCOLS:
        nums = [int(s) for s in rows.loc[rows["protocol"] == pr, "sweep_number"]]
        dup = sorted({s for s in nums if nums.count(s) > 1})
        if dup:
            raise TwinBuildError("cell %d %s sweep(s) %s: more than one row in the twins' table" % (specimen, pr, dup))

    ss_s, ss_n = np.unique(np.asarray(protocol.ss_sweep_number, dtype=np.int64), return_counts=True)
    ss_rows = rows[rows["protocol"] == "SS"]
    t_ss = {int(r["sweep_number"]): r for r in ss_rows.to_dict("records")}
    if set(t_ss) != set(int(s) for s in ss_s):
        raise TwinBuildError("cell %d: SS sweeps %s in the twins' table, %s stored in the archive"
                             % (specimen, sorted(t_ss), sorted(int(s) for s in ss_s)))
    n_w = int(np.asarray(protocol.ss_t_s).size)
    for s, k in zip(ss_s, ss_n):
        r = t_ss[int(s)]
        if int(r["n_windows"]) != int(k):
            raise TwinBuildError("cell %d SS sweep %d: n_windows %d in the twins' table, %d stored"
                                 % (specimen, int(s), int(r["n_windows"]), int(k)))
        if int(r["n_samples"]) != n_w or float(r["fs_Hz"]) != float(protocol.ss_fs_Hz):
            raise TwinBuildError("cell %d SS sweep %d: n_samples %d / fs_Hz %r in the twins' table, %d / %r stored"
                                 % (specimen, int(s), int(r["n_samples"]), float(r["fs_Hz"]), n_w,
                                    float(protocol.ss_fs_Hz)))

    ls_rows = rows[rows["protocol"] == "LS"]
    t_ls = {int(r["sweep_number"]): r for r in ls_rows.to_dict("records")}
    p_ls = [int(s) for s in protocol.ls_sweep_number]
    if set(t_ls) != set(p_ls):
        raise TwinBuildError("cell %d: LS sweeps %s in the twins' table, %s stored in the archive"
                             % (specimen, sorted(t_ls), sorted(p_ls)))
    for i, s in enumerate(p_ls):
        r = t_ls[s]
        checks = (("amplitude_pA", float(r["amplitude_pA"]), float(protocol.ls_amplitude_pA[i])),
                  ("onset_s", float(r["onset_s"]), float(protocol.ls_onset_s[i])),
                  ("duration_s", float(r["duration_s"]), float(protocol.ls_duration_s[i])),
                  ("n_samples", int(r["n_samples"]), int(protocol.ls_n_samples[i])),
                  ("fs_Hz", float(r["fs_Hz"]), float(protocol.ls_fs_Hz)),
                  ("real_spike_reason", str(r["real_spike_reason"]), str(protocol.ls_spike_reason[i])))
        for name, a, b in checks:
            if a != b:
                raise TwinBuildError("cell %d LS sweep %d: %s %r in the twins' table, %r in the archive"
                                     % (specimen, s, name, a, b))
        if bool(r["in_twin"]) != (str(r["real_spike_reason"]) == ""):
            raise TwinBuildError("cell %d LS sweep %d: in_twin %r disagrees with real_spike_reason %r"
                                 % (specimen, s, bool(r["in_twin"]), str(r["real_spike_reason"])))
    return rows


# ===========================================================================
#  The cell, (TC1)
# ===========================================================================
def build_twin_cell(swc_path, spec: TwinSpec, *, ih_spec, apply_theta, cfg: TwinConfig = TwinConfig()):
    """The twin's cell, (TC1) of SPEC 2.7 (h): the fitter's forward model at
    the truth -- passive_fitting_hpc_fixed.build_neuron_model(swc_path,
    F=spec.F); ih_mechanism.attach_ih(cell, ih_spec) when
    HAS_IH[spec.configuration]; apply_theta(cell, dict(spec.theta),
    spec.v_rest_mV), the fit's own ParamSpec.apply for this twin's model
    (passed by the caller). Refused (TwinBuildError): SPEC, Block 12 -- among
    them a rest drift ih_mechanism.rest_drift(cell, v_rest, t_ms=
    cfg.rest_drift_ms) above cfg.rest_drift_max_mV.

    The measured drift (mV) is left on the cell as `twin_rest_drift_mV`."""
    theta = _check_spec(spec)
    has_ih = HAS_IH[spec.configuration]
    if has_ih and ih_spec is None:
        raise TwinBuildError("twin %d (%s): simulated with I_h but no ih_spec given"
                             % (int(spec.twin_id), spec.configuration))
    swc_path = Path(swc_path)
    if not swc_path.is_file():
        raise TwinBuildError("twin %d: morphology %s not found" % (int(spec.twin_id), swc_path))
    import ih_mechanism as IM                     # local: NEURON is loaded by the monolith
    import passive_fitting_hpc_fixed as mono

    cell = mono.build_neuron_model(swc_path, F=float(spec.F))
    if has_ih:
        IM.attach_ih(cell, ih_spec)
    apply_theta(cell, dict(theta), float(spec.v_rest_mV))

    drift = float(IM.rest_drift(cell, float(spec.v_rest_mV), t_ms=float(cfg.rest_drift_ms)))
    if not (math.isfinite(drift) and drift <= float(cfg.rest_drift_max_mV)):
        raise TwinBuildError("twin %d (%s): unstimulated drift %.3g mV from V_rest %.6g mV over %g ms exceeds "
                             "rest_drift_max_mV %g (the rest is not a uniform equilibrium, D-192)"
                             % (int(spec.twin_id), spec.configuration, drift, float(spec.v_rest_mV),
                                float(cfg.rest_drift_ms), float(cfg.rest_drift_max_mV)))
    cell.twin_rest_drift_mV = drift
    return cell


# ===========================================================================
#  The traces, (TW1) and (TW2)
# ===========================================================================
def _polarity_mean(values: np.ndarray, mask: np.ndarray) -> float:
    """The loader's mean over the windows of one polarity, in archive order
    (passive_fitting_hpc_fixed.load_cell_from_archive: float(np.mean([...])))."""
    return float(np.mean([float(x) for x in np.asarray(values)[mask]]))


def simulate_twin(cell, spec: TwinSpec, protocol, table: pd.DataFrame, *, cfg: TwinConfig = TwinConfig(),
                  with_noise: bool = True) -> TwinTraces:
    """(TW1) and (TW2) of SPEC 2.7 for every stored SS window and every LS
    sweep with in_twin True of the real cell: Block 11's clean responses from
    V_rest (one SS simulation per polarity at A_p, d_p; one LS simulation per
    distinct (amplitude, onset, duration, length)) plus, when `with_noise`,
    Block 10's noise with each sweep's law and tilt from the twins' table
    (twin_table.sweep_law on spec.twin_of_specimen_id) and the seeds of
    (cfg.base_seed, spec.twin_id). `protocol`: twin_table.CellProtocol of the
    real cell; `table`: the twins' table (Block 5). Refused (TwinBuildError):
    SPEC, Block 12."""
    _check_spec(spec)
    specimen = int(spec.twin_of_specimen_id)
    _check_table(table, protocol, specimen)
    import synthetic_ground_truth as sgt          # local: the NEURON side

    v0 = float(spec.v_rest_mV)
    t_s = np.asarray(protocol.ss_t_s, dtype=np.float64)
    n_w = int(t_s.size)
    pol = np.asarray(protocol.ss_polarity_is_dep, dtype=bool)
    sweeps = np.asarray(protocol.ss_sweep_number, dtype=np.int64)
    pos = np.asarray(protocol.ss_position, dtype=np.int64)
    K = int(pol.size)

    # SS: one clean pulse per polarity present, at the loader's A_p and d_p (2.7 (f))
    ss_v = np.empty((K, n_w), dtype=np.float64)
    ss_i = np.empty((K, n_w), dtype=np.float64)
    ss_peak = np.empty(K, dtype=np.float64)
    ss_dur = np.empty(K, dtype=np.float64)
    for p in (True, False):
        m = pol == p
        if not m.any():
            continue
        a_p = _polarity_mean(protocol.ss_peak_pA, m)
        d_p = _polarity_mean(protocol.ss_stim_duration_s, m)
        r, i = sgt.simulate_ss_clean(cell, amplitude_pA=a_p, duration_s=d_p, t_s=t_s, v_init_mV=v0,
                                     dt_ms=float(cfg.neuron_dt_ms), settle_ms=float(cfg.settle_ms))
        ss_v[m] = r
        ss_i[m] = i
        ss_peak[m] = a_p
        ss_dur[m] = d_p
    if with_noise:
        dt_ss = 1.0 / float(protocol.ss_fs_Hz)
        for s in np.unique(sweeps):
            m = sweeps == s
            law, tilt = twin_table.sweep_law(table, specimen, "SS", int(s))
            ss_v[m] += twin_noise.ss_sweep_noise(law, tilt, dt_ss, n_w, pos[m], int(protocol.ss_period_samples),
                                                 base_seed=int(cfg.base_seed), twin_id=int(spec.twin_id),
                                                 sweep_number=int(s))

    # LS: every sweep with in_twin True, one simulation per distinct protocol (2.7 (d), D-190)
    fs_ls = float(protocol.ls_fs_Hz)
    cache: Dict[Tuple[float, float, float, int], Tuple[np.ndarray, np.ndarray]] = {}
    ls_num: List[int] = []
    ls_amp: List[float] = []
    ls_v: List[np.ndarray] = []
    ls_i: List[np.ndarray] = []
    for k, s in enumerate(np.asarray(protocol.ls_sweep_number, dtype=np.int64)):
        if str(protocol.ls_spike_reason[k]) != "":
            continue
        key = (float(protocol.ls_amplitude_pA[k]), float(protocol.ls_onset_s[k]),
               float(protocol.ls_duration_s[k]), int(protocol.ls_n_samples[k]))
        if key not in cache:
            cache[key] = sgt.simulate_ls_clean(cell, amplitude_pA=key[0], onset_s=key[1], duration_s=key[2],
                                               n_samples=key[3], fs_Hz=fs_ls, v_init_mV=v0,
                                               dt_ms=float(cfg.neuron_dt_ms), settle_ms=float(cfg.settle_ms))
        r, i = cache[key]
        v = np.array(r, dtype=np.float64, copy=True)
        if with_noise:
            law, tilt = twin_table.sweep_law(table, specimen, "LS", int(s))
            v += twin_noise.ls_sweep_noise(law, tilt, 1.0 / fs_ls, key[3], base_seed=int(cfg.base_seed),
                                           twin_id=int(spec.twin_id), sweep_number=int(s))
        ls_num.append(int(s))
        ls_amp.append(key[0])
        ls_v.append(v)
        ls_i.append(np.array(i, dtype=np.float64, copy=True))

    return TwinTraces(
        ss_fs_Hz=float(protocol.ss_fs_Hz), ss_t_s=t_s.copy(), ss_v_mV=ss_v, ss_i_pA=ss_i,
        ss_polarity_is_dep=pol.copy(), ss_peak_pA=ss_peak, ss_stim_duration_s=ss_dur,
        ss_sweep_number=sweeps.copy(), ls_fs_Hz=fs_ls,
        ls_sweep_number=np.asarray(ls_num, dtype=np.int64), ls_amplitude_pA=np.asarray(ls_amp, dtype=np.float64),
        ls_v_mV=tuple(ls_v), ls_i_pA=tuple(ls_i))


# ===========================================================================
#  The archive
# ===========================================================================
def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError("not JSON serialisable: %r (%s)" % (obj, type(obj).__name__))


def write_twin_archive(group_dir, spec: TwinSpec, traces: TwinTraces, *, swc_path, real_metadata: Mapping,
                       features: Mapping[str, float], provenance: Optional[Mapping] = None,
                       overwrite: bool = False, cfg: TwinConfig = TwinConfig(),
                       rest_drift_mV: float = float("nan"),
                       ls_sweeps_left_out: Optional[Mapping[int, str]] = None,
                       with_noise: bool = True) -> Path:
    """Write <group_dir>/specimen_<twin_id>/ in the layout the fitter's loader
    reads (SPEC, Block 12, "Outputs": ss_pulses.npz, ls_sweeps.npz,
    reconstruction.swc, metadata.json with its `twin` block) and return that
    directory. `features`: rin_MOhm, tau_ms, sag_ratio of the twin's cell;
    `real_metadata`: the real cell's metadata.json (for the layer and the
    dendrite type). Keyword-only (amendment 6 (ii), D-193): `cfg` (the
    `twin` block's base_seed, neuron_dt_ms, settle_ms), `rest_drift_mV` (mV,
    the drift build_twin_cell measured) and `ls_sweeps_left_out` (real LS
    sweep number -> the real trace's spike reason). `with_noise` (Block 18):
    whether `traces` carry the noise; recorded as `twin.with_noise`, and with
    False also `twin.noise_laws_applied` = False (the twins' table's laws and
    tilts were read for its checks only, not added). Refused (TwinBuildError):
    SPEC, Block 12."""
    theta = _check_spec(spec)
    with_noise = _check_with_noise(with_noise, spec.twin_id)
    swc_path = Path(swc_path)
    if not swc_path.is_file():
        raise TwinBuildError("twin %d: the real cell's reconstruction.swc %s not found" % (int(spec.twin_id),
                                                                                         swc_path))
    for key in ("rin_MOhm", "tau_ms", "sag_ratio"):
        if key not in features:
            raise TwinBuildError("twin %d: features lack %r" % (int(spec.twin_id), key))
    group_dir = Path(group_dir)
    final = group_dir / ("specimen_%d" % int(spec.twin_id))
    if final.exists() and not overwrite:
        raise TwinBuildError("twin %d: %s exists (overwrite=False)" % (int(spec.twin_id), final))
    group_dir.mkdir(parents=True, exist_ok=True)
    tmp = group_dir / (".specimen_%d.tmp.%d" % (int(spec.twin_id), os.getpid()))
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()

    K = int(np.asarray(traces.ss_polarity_is_dep).size)
    pol = np.asarray(traces.ss_polarity_is_dep, dtype=bool)
    np.savez_compressed(
        tmp / "ss_pulses.npz",
        t=np.asarray(traces.ss_t_s, dtype=np.float64),
        v=np.asarray(traces.ss_v_mV, dtype=np.float64),
        i_pA=np.asarray(traces.ss_i_pA, dtype=np.float64),
        polarity_is_dep=pol,
        peak_pA=np.asarray(traces.ss_peak_pA, dtype=np.float64),
        stim_duration_s=np.asarray(traces.ss_stim_duration_s, dtype=np.float64),
        sampling_rate_Hz=np.full(K, float(traces.ss_fs_Hz), dtype=np.float64),
        sweep_number=np.asarray(traces.ss_sweep_number, dtype=np.int64))

    M = len(traces.ls_v_mV)
    ls_arrays: Dict[str, np.ndarray] = {}
    ls_info: List[Dict[str, object]] = []
    for m in range(M):
        v = np.asarray(traces.ls_v_mV[m], dtype=np.float64)
        ls_arrays["v_%d" % m] = v
        ls_arrays["i_%d" % m] = np.asarray(traces.ls_i_pA[m], dtype=np.float64)
        ls_info.append(dict(index=m, sweep_number=int(traces.ls_sweep_number[m]),
                            detected_amplitude_pA=float(traces.ls_amplitude_pA[m]),
                            sampling_rate_Hz=float(traces.ls_fs_Hz), stimulus_name="Long Square",
                            n_samples=int(v.size)))
    ls_arrays["n_sweeps"] = np.array([M], dtype=np.int64)
    np.savez_compressed(tmp / "ls_sweeps.npz", **ls_arrays)

    shutil.copy2(swc_path, tmp / "reconstruction.swc")

    real_allen = dict(real_metadata.get("full_allen_metadata") or {})
    layer = real_metadata.get("layer", "")
    dendrite = real_metadata.get("dendrite_type", "")
    meta = dict(
        specimen_id=int(spec.twin_id),
        layer=layer,
        dendrite_type=dendrite,
        rin_MOhm=float(features["rin_MOhm"]),
        tau_ms=float(features["tau_ms"]),
        sag_ratio=float(features["sag_ratio"]),
        v_rest_mV=float(spec.v_rest_mV),
        ljp_correction_mV=0.0,
        full_allen_metadata=dict(structure_layer_name=real_allen.get("structure_layer_name", layer),
                                 dendrite_type=real_allen.get("dendrite_type", dendrite)),
        ss_extraction=dict(n_pulses_qc=K, n_dep=int(pol.sum()), n_hyp=int(K - pol.sum())),
        ls_sweeps=ls_info,
        twin=dict(twin_of_specimen_id=int(spec.twin_of_specimen_id), configuration=str(spec.configuration),
                  theta={k: float(theta[k]) for k in spec.theta}, F=float(spec.F),
                  has_ih=bool(HAS_IH[spec.configuration]), base_seed=int(cfg.base_seed),
                  neuron_dt_ms=float(cfg.neuron_dt_ms), settle_ms=float(cfg.settle_ms),
                  rest_drift_mV=float(rest_drift_mV), tau_ms_source=TAU_MS_SOURCE,
                  ls_sweeps_left_out={str(int(k)): str(v) for k, v in dict(ls_sweeps_left_out or {}).items()},
                  provenance=(dict(provenance) if provenance is not None else None)),
    )
    meta["twin"]["with_noise"] = with_noise
    if not with_noise:
        meta["twin"]["noise_laws_applied"] = False
    try:
        text = json.dumps(meta, indent=2, default=_json_default)
    except TypeError as exc:
        shutil.rmtree(tmp, ignore_errors=True)
        raise TwinBuildError("twin %d: metadata not serialisable: %s" % (int(spec.twin_id), exc)) from None
    with open(tmp / "metadata.json", "w", encoding="ascii", newline="\n") as fh:
        fh.write(text)

    if final.exists():
        shutil.rmtree(final)
    os.replace(tmp, final)
    return final


# ===========================================================================
#  The whole build of one twin
# ===========================================================================
def build_twin(spec: TwinSpec, real_spec_dir, table: pd.DataFrame, group_dir, *, ih_spec, apply_theta,
               spike_reason_fn=None, cfg: TwinConfig = TwinConfig(), overwrite: bool = False,
               provenance: Optional[Mapping] = None, with_noise: bool = True) -> Path:
    """The whole build of one twin from its real cell's archive directory:
    twin_table.load_cell_protocol (spike_reason_fn None ->
    passive_fitting_hpc_fixed.sweep_spike_reason), build_twin_cell on
    <real_spec_dir>/reconstruction.swc, simulate_twin, the features
    (synthetic_ground_truth.measure_rin_tau_sag from rest), write_twin_archive.
    Returns the twin's specimen directory. The caller clears NEURON's
    sections between twins. `provenance` is passed on to write_twin_archive
    (amendment 6 (ii)). The features (amendment 6 (iii), D-193):
    measure_rin_tau_sag on GroundTruthParams(cm_uF_cm2=Cm, rm_Ohm_cm2=Rm,
    e_pas_mV=V_rest) with the truth's Cm and Rm, so tau_ms = Rm Cm 1e-3 (the
    truth's passive membrane time constant); rin_MOhm and sag_ratio
    simulated. `with_noise` (Block 18, SPEC 2.8 (e)): False writes the clean
    responses alone (simulate_twin(..., with_noise=False)) and records it in
    the archive's `twin` block."""
    theta = _check_spec(spec)
    with_noise = _check_with_noise(with_noise, spec.twin_id)
    real_spec_dir = Path(real_spec_dir)
    swc = real_spec_dir / "reconstruction.swc"
    if not swc.is_file():
        raise TwinBuildError("twin %d: the real cell %s has no reconstruction.swc" % (int(spec.twin_id),
                                                                                    real_spec_dir))
    final = Path(group_dir) / ("specimen_%d" % int(spec.twin_id))
    if final.exists() and not overwrite:
        raise TwinBuildError("twin %d: %s exists (overwrite=False)" % (int(spec.twin_id), final))
    if spike_reason_fn is None:
        import passive_fitting_hpc_fixed as mono
        spike_reason_fn = mono.sweep_spike_reason
    protocol = twin_table.load_cell_protocol(real_spec_dir, spike_reason_fn=spike_reason_fn, twin_cfg=cfg)
    _check_table(table, protocol, int(spec.twin_of_specimen_id))
    with open(real_spec_dir / "metadata.json") as fh:
        real_meta = json.load(fh)

    cell = build_twin_cell(swc, spec, ih_spec=ih_spec, apply_theta=apply_theta, cfg=cfg)
    traces = simulate_twin(cell, spec, protocol, table, cfg=cfg, with_noise=with_noise)

    import synthetic_ground_truth as sgt
    _v, rin, tau, sag = sgt.measure_rin_tau_sag(cell, sgt.GroundTruthParams(cm_uF_cm2=theta["Cm"], rm_Ohm_cm2=theta["Rm"],
                                                                 e_pas_mV=float(spec.v_rest_mV)))
    left_out = {int(s): str(r) for s, r in zip(protocol.ls_sweep_number, protocol.ls_spike_reason) if str(r)}
    return write_twin_archive(group_dir, spec, traces, swc_path=swc, real_metadata=real_meta,
                              features=dict(rin_MOhm=rin, tau_ms=tau, sag_ratio=sag), provenance=provenance,
                              overwrite=overwrite, cfg=cfg, rest_drift_mV=float(cell.twin_rest_drift_mV),
                              ls_sweeps_left_out=left_out, with_noise=with_noise)
