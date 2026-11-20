# -*- coding: utf-8 -*-
"""
twin_build.py -- the twin build: the fitter's own model at the truth, clean responses plus noise, an archive cell (Block 12)
==========================================================================================================================

specs/SPEC.md, section 2.7 (c), (d), (f), (h) and the entry of Block 12
(D-183, D-186, D-187, D-190, D-192). NEURON side. Three separate concerns,
three functions: the cell (build_twin_cell), the traces (simulate_twin), the
archive (write_twin_archive); build_twin chains them for one twin.

CONTRACT STUB (contract: wave 2). The signatures, shapes, dtypes and units
below are the contract; the bodies are written by Block 12's implementer.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from twin_config import TwinConfig

CONFIGURATIONS = ("full", "passive_fit", "active_fit")
HAS_IH = {"full": True, "passive_fit": True, "active_fit": False}
PASSIVE_KEYS = ("Cm", "Rm", "Ra")
IH_KEYS = ("gbar", "dv_h", "kappa_tau")


class TwinBuildError(RuntimeError):
    """A twin the build refuses (SPEC, Block 12, "Refused")."""


@dataclass(frozen=True)
class TwinSpec:
    """One twin (one manifest row, resolved).

    twin_id             : the twin's own id (int), distinct from every real id
    twin_of_specimen_id : the real cell it is the twin of (int)
    configuration       : 'full' | 'passive_fit' | 'active_fit'
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


def build_twin_cell(swc_path, spec: TwinSpec, *, ih_spec, apply_theta, cfg: TwinConfig = TwinConfig()):
    """The twin's cell, (TC1) of SPEC 2.7 (h): the fitter's forward model at
    the truth -- passive_fitting_hpc_fixed.build_neuron_model(swc_path,
    F=spec.F); ih_mechanism.attach_ih(cell, ih_spec) when
    HAS_IH[spec.configuration]; apply_theta(cell, dict(spec.theta),
    spec.v_rest_mV), the fit's own ParamSpec.apply for this twin's model
    (passed by the caller). Refused (TwinBuildError): SPEC, Block 12 -- among
    them a rest drift ih_mechanism.rest_drift(cell, v_rest, t_ms=
    cfg.rest_drift_ms) above cfg.rest_drift_max_mV."""
    raise NotImplementedError("Block 12")


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
    raise NotImplementedError("Block 12")


def write_twin_archive(group_dir, spec: TwinSpec, traces: TwinTraces, *, swc_path, real_metadata: Mapping,
                       features: Mapping[str, float], provenance: Optional[Mapping] = None,
                       overwrite: bool = False) -> Path:
    """Write <group_dir>/specimen_<twin_id>/ in the layout the fitter's loader
    reads (SPEC, Block 12, "Outputs": ss_pulses.npz, ls_sweeps.npz,
    reconstruction.swc, metadata.json with its `twin` block) and return that
    directory. `features`: rin_MOhm, tau_ms, sag_ratio of the twin's cell;
    `real_metadata`: the real cell's metadata.json (for the layer and the
    dendrite type). Refused (TwinBuildError): SPEC, Block 12."""
    raise NotImplementedError("Block 12")


def build_twin(spec: TwinSpec, real_spec_dir, table: pd.DataFrame, group_dir, *, ih_spec, apply_theta,
               spike_reason_fn=None, cfg: TwinConfig = TwinConfig(), overwrite: bool = False) -> Path:
    """The whole build of one twin from its real cell's archive directory:
    twin_table.load_cell_protocol (spike_reason_fn None ->
    passive_fitting_hpc_fixed.sweep_spike_reason), build_twin_cell on
    <real_spec_dir>/reconstruction.swc, simulate_twin, the features
    (synthetic_ground_truth.measure_rin_tau_sag from rest), write_twin_archive.
    Returns the twin's specimen directory. The caller clears NEURON's
    sections between twins."""
    raise NotImplementedError("Block 12")
