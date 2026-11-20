# -*- coding: utf-8 -*-
"""
twin_manifest.py -- the twins' manifest: three configurations of n twins on real cells drawn at random (Block 13)
================================================================================================================

specs/SPEC.md, section 2.7 (i) and the entry of Block 13 (D-184, D-188).
Pure: numpy and pandas; no NEURON, no archive I/O.

CONTRACT STUB (contract: wave 2). The signatures, shapes, dtypes and units
below are the contract; the bodies are written by Block 13's implementer.
"""
from __future__ import annotations

from typing import Dict, Mapping, Optional

import pandas as pd

CONFIGURATIONS = ("full", "passive_fit", "active_fit")
ARM_OF = {"full": "ih6", "passive_fit": "passive_fullstep", "active_fit": "ih6"}
HAS_IH = {"full": True, "passive_fit": True, "active_fit": False}


def draw_twin_manifest(real_cells: pd.DataFrame, *, n_per_configuration: int = 10, seed: int = 0,
                       id_base: int = 900_000_000, truth_kwargs: Optional[Mapping] = None) -> pd.DataFrame:
    """The manifest of SPEC 2.7 (i): 3 * n_per_configuration rows, one per
    twin, in the order of CONFIGURATIONS and, within one, the drawn order.

    real_cells : one row per real cell available -- specimen_id (int64),
                 swc_path (str), v_rest_mV (float64), cell_tau_s_above_window
                 (bool)
    truth_kwargs : keyword arguments passed unchanged to
                 synth_gt_grid.draw_manifest (it must not name
                 draws_per_morph, use_ih, fp_control_frac, seed, id_base or a
                 noise table: this function sets them)
    Columns, dtypes, the id rule, the seeds and the refusals (ValueError):
    SPEC, Block 13."""
    raise NotImplementedError("Block 13")


def save_twin_manifest(df: pd.DataFrame, path) -> None:
    """Write the manifest as CSV (ASCII, LF, floats with 17 significant
    digits)."""
    raise NotImplementedError("Block 13")


def load_twin_manifest(path) -> pd.DataFrame:
    """Read a manifest written by save_twin_manifest: the same values, dtypes
    and column order."""
    raise NotImplementedError("Block 13")


def twin_theta(row) -> Dict[str, float]:
    """The truth of one manifest row in the fitter's names: Cm (uF/cm^2), Rm
    (Ohm cm^2), Ra (Ohm cm), and for a configuration with I_h gbar (S/cm^2),
    dv_h (mV), kappa_tau."""
    raise NotImplementedError("Block 13")


def twin_spec_of(row, *, F: Optional[float] = None):
    """twin_build.TwinSpec of one manifest row (F from the row's spine-factor
    column when not given)."""
    raise NotImplementedError("Block 13")
