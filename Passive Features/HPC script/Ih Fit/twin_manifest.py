# -*- coding: utf-8 -*-
"""
twin_manifest.py -- the twins' manifest: three configurations of n twins on real cells drawn at random (Block 13)
================================================================================================================

specs/SPEC.md, section 2.7 (i) and the entry of Block 13 (D-184, D-188).
Pure: numpy and pandas; no NEURON, no archive I/O (synth_gt_grid is imported
lazily inside draw_twin_manifest; it does not import NEURON).

What it does
------------
Stage 7 has three configurations of n twins each, q = 0, 1, 2 in the order of
CONFIGURATIONS. For configuration q:

  cells(q) = default_rng(SeedSequence([seed, q])).choice(sorted real ids, n,
             replace=False), kept in the drawn order;
  twin i   = id_base + 100 q + i, the twin of cells(q)[i];
  truths   = synth_gt_grid.draw_manifest(<swc paths of cells(q), drawn order>,
             draws_per_morph=1, seed=SeedSequence([seed, q, 1]).generate_state(1)[0],
             id_base=id_base + 100 q, use_ih=True, fp_control_frac=0.0,
             **truth_kwargs);
  active_fit: the three I_h truth columns take draw_manifest's false-positive
             values (gbar at the floor, dv_h 0, kappa_tau 1) and is_fp_control
             is True.

Columns of draw_manifest whose name starts with `noise_` or `acq_`, and its
leak-reversal column `e_pas_mV`, are dropped (D-192; the twins' noise lives in
the twins' table). Every other draw_manifest column is kept under its own name.

Units: the truth columns as synth_gt_grid writes them -- cm_true uF/cm^2,
rm_true Ohm cm^2, ra_true Ohm cm, ih_gihbar_S_cm2 S/cm^2, ih_dvh_mV mV,
ih_kappa_tau dimensionless; v_rest_mV mV.
"""
from __future__ import annotations

import inspect
from pathlib import Path
from typing import Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

CONFIGURATIONS = ("full", "passive_fit", "active_fit")
ARM_OF = {"full": "ih6", "passive_fit": "passive_fullstep", "active_fit": "ih6"}
HAS_IH = {"full": True, "passive_fit": True, "active_fit": False}

#: the fitter's parameter names -> the manifest's truth columns (the same map
#: as ih_recovery_report.TRUTH_COLUMN; checked in the smoke test, TM10)
PASSIVE_TRUTH = (("Cm", "cm_true"), ("Rm", "rm_true"), ("Ra", "ra_true"))
IH_TRUTH = (("gbar", "ih_gihbar_S_cm2"), ("dv_h", "ih_dvh_mV"), ("kappa_tau", "ih_kappa_tau"))

#: the columns this module writes ahead of draw_manifest's, in this order
LEAD_COLUMNS = ("specimen_id", "configuration", "arm", "twin_of_specimen_id", "v_rest_mV",
                "real_swc_path", "cell_tau_s_above_window", "manifest_seed", "is_fp_control")
#: the columns of `real_cells` required on input
REAL_CELL_COLUMNS = ("specimen_id", "swc_path", "v_rest_mV", "cell_tau_s_above_window")
#: draw_manifest's arguments this function sets itself (refused in truth_kwargs)
RESERVED_KWARGS = ("draws_per_morph", "use_ih", "fp_control_frac", "seed", "id_base", "noise_table")
#: draw_manifest columns dropped by name (besides the noise_ / acq_ prefixes)
DROPPED_COLUMNS = ("e_pas_mV",)
DROPPED_PREFIXES = ("noise_", "acq_")

#: dtypes enforced on reading (columns absent from a file are skipped)
_INT_COLUMNS = ("specimen_id", "twin_of_specimen_id", "manifest_seed", "cohort", "draw_idx")
_BOOL_COLUMNS = ("cell_tau_s_above_window", "is_fp_control", "use_ih")
_STR_COLUMNS = ("configuration", "arm", "real_swc_path", "group", "morph_name", "swc",
                "ra_mode", "ih_dist", "ih_kinetics", "ih_regions")

MAX_PER_CONFIGURATION = 100   # the id rule id_base + 100 q + i needs i < 100


# ===========================================================================
#  Validation of the inputs
# ===========================================================================
def _real_ids(real_cells: pd.DataFrame) -> np.ndarray:
    """The real specimen ids as int64, refusing non-integers and duplicates."""
    col = real_cells["specimen_id"]
    if pd.api.types.is_bool_dtype(col):
        raise ValueError("real_cells.specimen_id is boolean, not an integer id")
    if pd.api.types.is_integer_dtype(col):
        ids = col.to_numpy(dtype=np.int64)
    else:
        vals = []
        for v in col.tolist():
            if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating)):
                raise ValueError("real_cells.specimen_id holds a non-integer value: {!r}".format(v))
            if not (np.isfinite(float(v)) and float(v).is_integer()):
                raise ValueError("real_cells.specimen_id holds a non-integer value: {!r}".format(v))
            vals.append(int(v))
        ids = np.asarray(vals, dtype=np.int64)
    if len(np.unique(ids)) != len(ids):
        dup = sorted(int(x) for x in pd.Series(ids)[pd.Series(ids).duplicated()].unique())
        raise ValueError("real_cells.specimen_id has duplicates: {}".format(dup))
    return ids


def _check_inputs(real_cells: pd.DataFrame, n: int, truth_kwargs: Mapping) -> np.ndarray:
    missing = [c for c in REAL_CELL_COLUMNS if c not in real_cells.columns]
    if missing:
        raise ValueError("real_cells misses column(s) {}".format(missing))
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)):
        raise ValueError("n_per_configuration must be an int, got {!r}".format(n))
    if n < 1 or n > MAX_PER_CONFIGURATION:
        raise ValueError("n_per_configuration must be in [1, {}], got {}".format(MAX_PER_CONFIGURATION, n))
    if n > len(real_cells):
        raise ValueError("n_per_configuration = {} exceeds the {} real cells available"
                         .format(n, len(real_cells)))
    ids = _real_ids(real_cells)
    v = pd.to_numeric(real_cells["v_rest_mV"], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(v)):
        bad = [int(i) for i, x in zip(ids, v) if not np.isfinite(x)]
        raise ValueError("real_cells.v_rest_mV is not finite for specimen(s) {}".format(bad))
    named = sorted(k for k in truth_kwargs if k in RESERVED_KWARGS)
    if named:
        raise ValueError("truth_kwargs names {} -- set by draw_twin_manifest itself".format(named))
    return ids


def _twin_ids(n: int, id_base: int) -> np.ndarray:
    """(3, n) int64: id_base + 100 q + i."""
    q = np.arange(len(CONFIGURATIONS), dtype=np.int64)[:, None]
    i = np.arange(n, dtype=np.int64)[None, :]
    return np.int64(id_base) + 100 * q + i


def _fp_values(truth_kwargs: Mapping) -> Dict[str, float]:
    """The I_h truth values draw_manifest gives a false-positive control: gbar at
    `ih_gbar_floor_S_cm2` (the caller's, else draw_manifest's default), dv_h 0,
    kappa_tau 1 (synth_gt_grid.draw_manifest, the false-positive block)."""
    import synth_gt_grid as G
    floor = truth_kwargs.get(
        "ih_gbar_floor_S_cm2",
        inspect.signature(G.draw_manifest).parameters["ih_gbar_floor_S_cm2"].default)
    return {"ih_gihbar_S_cm2": float(floor), "ih_dvh_mV": 0.0, "ih_kappa_tau": 1.0}


def _kept(col: str) -> bool:
    return col not in DROPPED_COLUMNS and not col.startswith(DROPPED_PREFIXES)


# ===========================================================================
#  The manifest
# ===========================================================================
def configuration_cells(sorted_ids: np.ndarray, n: int, seed: int, q: int) -> np.ndarray:
    """The n real ids of configuration q, in the drawn order (SPEC 2.7 (i))."""
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(q)]))
    return np.asarray(rng.choice(sorted_ids, size=int(n), replace=False), dtype=np.int64)


def truth_seed(seed: int, q: int) -> int:
    """The seed draw_manifest receives for configuration q (SPEC, Block 13)."""
    return int(np.random.SeedSequence([int(seed), int(q), 1]).generate_state(1)[0])


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
    import synth_gt_grid as G

    truth_kwargs = dict(truth_kwargs or {})
    n = n_per_configuration
    ids = _check_inputs(real_cells, n, truth_kwargs)
    n = int(n)
    twin_ids = _twin_ids(n, int(id_base))
    clash = sorted(set(int(x) for x in twin_ids.ravel()) & set(int(x) for x in ids))
    if clash:
        raise ValueError("twin id(s) {} equal real specimen id(s); choose another id_base".format(clash))

    by_id = real_cells.assign(_sid=ids).set_index("_sid")
    sorted_ids = np.sort(ids)
    fp_vals = _fp_values(truth_kwargs)

    frames: List[pd.DataFrame] = []
    for q, conf in enumerate(CONFIGURATIONS):
        cells = configuration_cells(sorted_ids, n, seed, q)
        swcs = [str(by_id.loc[int(c), "swc_path"]) for c in cells]
        truths = G.draw_manifest(swcs, draws_per_morph=1, seed=truth_seed(seed, q),
                                 id_base=int(id_base) + 100 * q, use_ih=True,
                                 fp_control_frac=0.0, **truth_kwargs)
        if len(truths) != n:
            raise ValueError("draw_manifest returned {} rows for {} cells (truth_kwargs {!r} truncates the draw?)"
                             .format(len(truths), n, sorted(truth_kwargs)))
        got = truths["specimen_id"].to_numpy(dtype=np.int64)
        if not np.array_equal(got, twin_ids[q]):
            raise RuntimeError("draw_manifest's ids {} differ from the twin ids {}".format(got, twin_ids[q]))
        truths = truths[[c for c in truths.columns if _kept(c)]].copy()
        if not HAS_IH[conf]:
            for col, val in fp_vals.items():
                truths[col] = val
            truths["is_fp_control"] = True
        else:
            truths["is_fp_control"] = False
        lead = pd.DataFrame({
            "specimen_id": twin_ids[q].astype(np.int64),
            "configuration": conf,
            "arm": ARM_OF[conf],
            "twin_of_specimen_id": cells.astype(np.int64),
            "v_rest_mV": [float(by_id.loc[int(c), "v_rest_mV"]) for c in cells],
            "real_swc_path": swcs,
            "cell_tau_s_above_window": [bool(by_id.loc[int(c), "cell_tau_s_above_window"]) for c in cells],
            "manifest_seed": np.full(n, int(seed), dtype=np.int64),
            "is_fp_control": truths["is_fp_control"].to_numpy(dtype=bool),
        })
        rest = truths.drop(columns=["specimen_id", "is_fp_control"]).reset_index(drop=True)
        frames.append(pd.concat([lead, rest], axis=1))

    df = pd.concat(frames, ignore_index=True)
    return _coerce(df)


def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce the manifest's dtypes on the columns present."""
    df = df.copy()
    for c in _INT_COLUMNS:
        if c in df.columns:
            df[c] = df[c].astype(np.int64)
    for c in _BOOL_COLUMNS:
        if c in df.columns:
            df[c] = df[c].astype(bool)
    for c in _STR_COLUMNS:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str).astype(object)
    # every other column is a float64 (a float written as "0" reads back as int)
    for c in df.columns:
        if c not in _INT_COLUMNS and c not in _BOOL_COLUMNS and c not in _STR_COLUMNS:
            df[c] = df[c].astype(np.float64)
    return df


# ===========================================================================
#  I/O
# ===========================================================================
def save_twin_manifest(df: pd.DataFrame, path) -> None:
    """Write the manifest as CSV (ASCII, LF, floats with 17 significant
    digits)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.17g", lineterminator="\n")


def load_twin_manifest(path) -> pd.DataFrame:
    """Read a manifest written by save_twin_manifest: the same values, dtypes
    and column order."""
    head = pd.read_csv(path, nrows=0).columns
    dtype = {c: str for c in _STR_COLUMNS if c in head}
    df = pd.read_csv(path, float_precision="round_trip", dtype=dtype)
    return _coerce(df)


# ===========================================================================
#  One row
# ===========================================================================
def twin_theta(row) -> Dict[str, float]:
    """The truth of one manifest row in the fitter's names: Cm (uF/cm^2), Rm
    (Ohm cm^2), Ra (Ohm cm), and for a configuration with I_h gbar (S/cm^2),
    dv_h (mV), kappa_tau."""
    conf = str(row["configuration"])
    if conf not in HAS_IH:
        raise ValueError("unknown configuration {!r}".format(conf))
    pairs = PASSIVE_TRUTH + (IH_TRUTH if HAS_IH[conf] else ())
    theta = {name: float(row[col]) for name, col in pairs}
    bad = [k for k, v in theta.items() if not np.isfinite(v)]
    if bad:
        raise ValueError("twin {}: non-finite truth for {}".format(row["specimen_id"], bad))
    return theta


def twin_spec_of(row, *, F: Optional[float] = None):
    """twin_build.TwinSpec of one manifest row (F from the row's spine-factor
    column when not given)."""
    from twin_build import TwinSpec
    return TwinSpec(
        twin_id=int(row["specimen_id"]),
        twin_of_specimen_id=int(row["twin_of_specimen_id"]),
        configuration=str(row["configuration"]),
        theta=twin_theta(row),
        v_rest_mV=float(row["v_rest_mV"]),
        F=float(row["F"] if F is None else F),
    )
