# -*- coding: utf-8 -*-
"""
twin_table.py -- the twins' table: each twin sweep's law, tilt and protocol (Block 5)
===================================================================================

specs/SPEC.md, section 2.7 (a), (b) and the entry of Block 5 (D-073, D-076
(iii), D-122.3, D-182.3, D-186). Loading layer: the only twin module that
reads the real archive and the noise-fit tables. No NEURON, no plotting.

CONTRACT STUB (contract: wave 1). The signatures, shapes, dtypes and units
below are the contract; the bodies are written by Block 5's implementer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
import pandas as pd

from noise_law import NoiseLaw
from noise_windows import WindowConfig
from twin_config import TwinConfig

# Columns of the twins' table, in this order (SPEC, Block 5, "Outputs").
TWIN_SWEEP_COLUMNS = (
    "specimen_id", "protocol", "sweep_number", "n_windows", "amplitude_pA", "onset_s", "duration_s", "n_samples",
    "fs_Hz", "sigma_f_mV", "sigma_m_mV", "tau_m_s", "sigma_s_mV", "tau_s_s", "tilt_mV_per_s", "law_source",
    "noise_flags", "cell_tau_s_above_window")


class TwinTableError(RuntimeError):
    """A cell or a sweep the twins' table refuses (SPEC, Block 5, "Refused")."""


@dataclass(frozen=True)
class CellProtocol:
    """The stimulation protocol of one real cell, read from its archive.

    SS, per stored window in the archive's order, shape (K,):
        ss_sweep_number (int64), ss_position (int64, 0-based place in the
        sweep's train), ss_polarity_is_dep (bool), ss_peak_pA (float64,
        signed pA), ss_stim_duration_s (float64, s);
        ss_t_s float64 (n_w,), s, 0 at the pulse onset; ss_fs_Hz (Hz);
        ss_pre_n (samples before the onset); ss_period_samples (onset to
        onset); ss_train_length (pulses per sweep).
    LS, per stored sweep in the order of metadata.json['ls_sweeps'], (K_LS,):
        ls_index (int64, the <index> of v_<index>), ls_sweep_number (int64),
        ls_amplitude_pA (float64, signed pA), ls_onset_s (float64, s),
        ls_duration_s (float64, s: step end - onset), ls_n_samples (int64);
        ls_fs_Hz (Hz).
    """
    specimen_id: int
    ss_fs_Hz: float
    ss_t_s: np.ndarray
    ss_pre_n: int
    ss_period_samples: int
    ss_train_length: int
    ss_sweep_number: np.ndarray
    ss_position: np.ndarray
    ss_polarity_is_dep: np.ndarray
    ss_peak_pA: np.ndarray
    ss_stim_duration_s: np.ndarray
    ls_fs_Hz: float
    ls_index: np.ndarray
    ls_sweep_number: np.ndarray
    ls_amplitude_pA: np.ndarray
    ls_onset_s: np.ndarray
    ls_duration_s: np.ndarray
    ls_n_samples: np.ndarray


def load_cell_protocol(spec_dir, cfg: WindowConfig = WindowConfig()) -> CellProtocol:
    """The protocol of the real cell in `spec_dir` (an archive cell directory,
    SPEC section 3), through noise_windows.load_ss_windows and
    load_ls_prewindows, plus `peak_pA` and `stim_duration_s` of ss_pulses.npz
    and `index` of metadata.json['ls_sweeps']. Raises TwinTableError when a
    sweep of metadata.json['ls_sweeps'] is not returned by load_ls_prewindows."""
    raise NotImplementedError("Block 5")


def read_noise_tables(cells_csv, sweeps_csv) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """(cells, sweeps): the per-cell and the per-sweep table written by
    `run_noise_fit.py merge` (SPEC section 3), as DataFrames; an empty CSV
    cell is a missing value."""
    raise NotImplementedError("Block 5")


def sweep_laws_from_tables(cells: pd.DataFrame, sweeps: pd.DataFrame,
                           cfg: TwinConfig = TwinConfig()) -> pd.DataFrame:
    """(P0') of SPEC 2.7 (b): one row per row of `sweeps`, with columns
    specimen_id, protocol ('SS' | 'LS'), sweep_number, sigma_f_mV,
    sigma_m_mV, tau_m_s, sigma_s_mV, tau_s_s, tilt_mV_per_s, law_source
    ('sweep' | 'cell'), noise_flags (str, '' when none),
    cell_tau_s_above_window (bool: tau_s_s > cfg.slow_flag_tau_s). mV, s,
    mV/s. Refusals: SPEC, Block 5."""
    raise NotImplementedError("Block 5")


def build_twin_sweep_table(protocols: Sequence[CellProtocol], cells: pd.DataFrame, sweeps: pd.DataFrame,
                           cfg: TwinConfig = TwinConfig()) -> pd.DataFrame:
    """The twins' table for the cells in `protocols`: one row per real sweep
    (SS and LS), columns TWIN_SWEEP_COLUMNS, dtypes and sorting as SPEC,
    Block 5, "Outputs" states. Refusals: SPEC, Block 5."""
    raise NotImplementedError("Block 5")


def write_twin_sweep_table(table: pd.DataFrame, path) -> None:
    """Write the table as CSV, floats with 17 significant digits ('%.17g')."""
    raise NotImplementedError("Block 5")


def read_twin_sweep_table(path) -> pd.DataFrame:
    """Read a table written by write_twin_sweep_table: the same values,
    dtypes and column order."""
    raise NotImplementedError("Block 5")


def sweep_law(table: pd.DataFrame, specimen_id: int, protocol: str, sweep_number: int) -> Tuple[NoiseLaw, float]:
    """(law, tilt_mV_per_s) of one row: NoiseLaw(sigma_f, sigma_m, tau_m,
    sigma_s, tau_s) in mV, mV, s, mV, s and the tilt in mV/s. KeyError when
    the row is absent, TwinTableError when it is not unique."""
    raise NotImplementedError("Block 5")
