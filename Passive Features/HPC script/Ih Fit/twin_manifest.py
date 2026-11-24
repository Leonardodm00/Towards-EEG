# -*- coding: utf-8 -*-
"""
twin_manifest.py -- the twins' manifest: three configurations of n twins on real cells drawn at random (Block 13),
the noise-free configuration (Block 16) and the run's I_h arm (Block 21)
================================================================================================================

specs/SPEC.md, section 2.7 (i) and the entry of Block 13 (D-184, D-188);
section 2.8 (e) and the entry of Block 16 (D-207, D-209);
section 2.9 (d) and the entry of Block 21 (D-234).
Pure: numpy and pandas; no NEURON, no archive I/O (synth_gt_grid is imported
lazily inside draw_twin_manifest; it does not import NEURON).

What it does
------------
Stage 7 draws three configurations of n twins each, q = 0, 1, 2 in the order of
DRAWN_CONFIGURATIONS (the first three of CONFIGURATIONS). For configuration q:

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

Block 16 (SPEC 2.8 (e)): after the three drawn configurations, ih_dvh_mV is
set to `dvh_truth_mV` in every row whose configuration has I_h (HAS_IH); None
(the default since Block 21) keeps the drawn values -- drawn uniformly when
truth_kwargs carries `ih_dvh_range_mV`, else 0. Then the fourth configuration
`full_no_noise` (q = 3): its i-th row, i < n_no_noise, is a copy of the i-th
`full` row with specimen_id = id_base + 300 + i, configuration
`full_no_noise` and arm ARM_OF["full_no_noise"]; every other column (truths,
real cell, manifest_seed, flags, paths) is the `full` row's. Two columns are
appended after draw_manifest's, in every row: `with_noise` (bool,
WITH_NOISE[configuration]) and `paired_with` (int64: the `full` twin's id for
a `full_no_noise` row, -1 otherwise).

Block 21 (SPEC 2.9 (d)): the run's I_h arm `ih_arm` (IH_ARMS; default
`ih6_d207`) sets the `arm` column: arm_of(configuration, ih_arm) is
`passive_fullstep` for `passive_fit` and `ih_arm` for every other
configuration. ARM_OF stays the `ih5` table. With ih_arm = "ih5" a row with
I_h whose dv_h truth is not 0 is refused (ValueError): the arm's model has no
dv_h. A row without I_h (active_fit) has dv_h 0 whatever the options.

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

CONFIGURATIONS = ("full", "passive_fit", "active_fit", "full_no_noise")
ARM_OF = {"full": "ih5", "passive_fit": "passive_fullstep", "active_fit": "ih5", "full_no_noise": "ih5"}
HAS_IH = {"full": True, "passive_fit": True, "active_fit": False, "full_no_noise": True}
WITH_NOISE = {"full": True, "passive_fit": True, "active_fit": True, "full_no_noise": False}

#: the run's I_h arms (Block 21, SPEC 2.9 (d)); the arm of passive_fit is
#: always PASSIVE_ARM
IH_ARMS = ("ih5", "ih6_d207")
DEFAULT_IH_ARM = "ih6_d207"
PASSIVE_ARM = "passive_fullstep"
PASSIVE_CONFIGURATION = "passive_fit"

#: the configurations drawn by draw_manifest (q = 0, 1, 2); full_no_noise
#: (q = 3) copies rows of `full`
DRAWN_CONFIGURATIONS = ("full", "passive_fit", "active_fit")
NO_NOISE_CONFIGURATION = "full_no_noise"
NO_NOISE_PARTNER = "full"

#: the fitter's parameter names -> the manifest's truth columns (the same map
#: as ih_recovery_report.TRUTH_COLUMN; checked in the smoke test, TM10)
PASSIVE_TRUTH = (("Cm", "cm_true"), ("Rm", "rm_true"), ("Ra", "ra_true"))
IH_TRUTH = (("gbar", "ih_gihbar_S_cm2"), ("dv_h", "ih_dvh_mV"), ("kappa_tau", "ih_kappa_tau"))

#: the columns this module writes ahead of draw_manifest's, in this order
LEAD_COLUMNS = ("specimen_id", "configuration", "arm", "twin_of_specimen_id", "v_rest_mV",
                "real_swc_path", "cell_tau_s_above_window", "manifest_seed", "is_fp_control")
#: the columns appended after draw_manifest's (Block 16)
TAIL_COLUMNS = ("with_noise", "paired_with")
PAIRED_WITH_NONE = -1
#: the columns of `real_cells` required on input
REAL_CELL_COLUMNS = ("specimen_id", "swc_path", "v_rest_mV", "cell_tau_s_above_window")
#: draw_manifest's arguments this function sets itself (refused in truth_kwargs)
RESERVED_KWARGS = ("draws_per_morph", "use_ih", "fp_control_frac", "seed", "id_base", "noise_table")
#: draw_manifest columns dropped by name (besides the noise_ / acq_ prefixes)
DROPPED_COLUMNS = ("e_pas_mV",)
DROPPED_PREFIXES = ("noise_", "acq_")

#: dtypes enforced on reading (columns absent from a file are skipped)
_INT_COLUMNS = ("specimen_id", "twin_of_specimen_id", "manifest_seed", "cohort", "draw_idx", "paired_with")
_BOOL_COLUMNS = ("cell_tau_s_above_window", "is_fp_control", "use_ih", "with_noise")
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


def _check_no_noise(n_no_noise, n: int) -> int:
    """0 <= n_no_noise <= n_per_configuration (an int), else ValueError."""
    if isinstance(n_no_noise, (bool, np.bool_)) or not isinstance(n_no_noise, (int, np.integer)):
        raise ValueError("n_no_noise must be an int, got {!r}".format(n_no_noise))
    if n_no_noise < 0 or n_no_noise > n:
        raise ValueError("n_no_noise must be in [0, n_per_configuration = {}], got {}".format(n, n_no_noise))
    return int(n_no_noise)


def _check_dvh(dvh_truth_mV) -> Optional[float]:
    """None, or a finite number (mV), else ValueError."""
    if dvh_truth_mV is None:
        return None
    if isinstance(dvh_truth_mV, (bool, np.bool_)) or not isinstance(dvh_truth_mV, (int, float, np.integer,
                                                                                   np.floating)):
        raise ValueError("dvh_truth_mV must be a number (mV) or None, got {!r}".format(dvh_truth_mV))
    if not np.isfinite(float(dvh_truth_mV)):
        raise ValueError("dvh_truth_mV must be finite, got {!r}".format(dvh_truth_mV))
    return float(dvh_truth_mV)


def _check_ih_arm(ih_arm) -> str:
    """One of IH_ARMS, else ValueError."""
    if not isinstance(ih_arm, str) or ih_arm not in IH_ARMS:
        raise ValueError("ih_arm must be one of {}, got {!r}".format(IH_ARMS, ih_arm))
    return ih_arm


def arm_of(configuration: str, ih_arm: str) -> str:
    """The arm a twin of `configuration` is fitted with when the run's I_h arm
    is `ih_arm` (SPEC, Block 21): passive_fit -> passive_fullstep, every other
    configuration -> ih_arm. Unknown names: ValueError."""
    ih_arm = _check_ih_arm(ih_arm)
    if not isinstance(configuration, str) or configuration not in CONFIGURATIONS:
        raise ValueError("configuration must be one of {}, got {!r}".format(CONFIGURATIONS, configuration))
    return PASSIVE_ARM if configuration == PASSIVE_CONFIGURATION else ih_arm


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
    """(4, n) int64: id_base + 100 q + i, q in the order of CONFIGURATIONS
    (row 3, full_no_noise, is id_base + 300 + i; only its first n_no_noise
    entries are used)."""
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


def draw_twin_manifest(real_cells: pd.DataFrame, *, n_per_configuration: int = 10, n_no_noise: int = 5,
                       seed: int = 0, id_base: int = 900_000_000, ih_arm: str = DEFAULT_IH_ARM,
                       dvh_truth_mV: Optional[float] = None,
                       truth_kwargs: Optional[Mapping] = None) -> pd.DataFrame:
    """The manifest of SPEC 2.7 (i) and 2.8 (e): 3 * n_per_configuration +
    n_no_noise rows, one per twin, in the order of CONFIGURATIONS and, within
    one, the drawn order (full_no_noise: the order of `full`).

    n_no_noise : the number of full_no_noise twins, 0 <= n_no_noise <=
                 n_per_configuration; the i-th copies the i-th `full` twin
    ih_arm     : the run's I_h arm, one of IH_ARMS (Block 21); the `arm`
                 column is arm_of(configuration, ih_arm)
    dvh_truth_mV : the truth of dv_h (mV) set in every row with I_h after the
                 draw; None (default) keeps draw_manifest's values (drawn when
                 truth_kwargs carries ih_dvh_range_mV, else 0). With
                 ih_arm = "ih5" a row with I_h whose dv_h truth is not 0 is
                 refused (ValueError).

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
    n0 = _check_no_noise(n_no_noise, n)
    dvh = _check_dvh(dvh_truth_mV)
    ih_arm = _check_ih_arm(ih_arm)
    twin_ids = _twin_ids(n, int(id_base))
    used = [int(x) for x in twin_ids[:len(DRAWN_CONFIGURATIONS)].ravel()] + \
        [int(x) for x in twin_ids[CONFIGURATIONS.index(NO_NOISE_CONFIGURATION), :n0]]
    clash = sorted(set(used) & set(int(x) for x in ids))
    if clash:
        raise ValueError("twin id(s) {} equal real specimen id(s); choose another id_base".format(clash))

    by_id = real_cells.assign(_sid=ids).set_index("_sid")
    sorted_ids = np.sort(ids)
    fp_vals = _fp_values(truth_kwargs)

    frames: List[pd.DataFrame] = []
    for q, conf in enumerate(DRAWN_CONFIGURATIONS):
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
            "arm": arm_of(conf, ih_arm),
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
    if dvh is not None:
        has_ih = df["configuration"].map(HAS_IH).to_numpy(dtype=bool)
        df.loc[has_ih, "ih_dvh_mV"] = dvh
    if ih_arm == "ih5":
        has_ih = df["configuration"].map(HAS_IH).to_numpy(dtype=bool)
        dv = df["ih_dvh_mV"].to_numpy(dtype=np.float64)
        bad = has_ih & ~(dv == 0.0)
        if bad.any():
            raise ValueError(
                "ih_arm='ih5' cannot hold a dv_h truth other than 0 (the arm does not fit dv_h): twin(s) {} have "
                "ih_dvh_mV {}; pass dvh_truth_mV=0.0, or drop ih_dvh_range_mV from truth_kwargs, or use "
                "ih_arm='ih6_d207'".format([int(x) for x in df.loc[bad, "specimen_id"]][:5],
                                           [float(x) for x in dv[bad]][:5]))
    df["with_noise"] = df["configuration"].map(WITH_NOISE).to_numpy(dtype=bool)
    df["paired_with"] = np.full(len(df), PAIRED_WITH_NONE, dtype=np.int64)

    partner = df[df["configuration"] == NO_NOISE_PARTNER].iloc[:n0].copy()
    q0 = CONFIGURATIONS.index(NO_NOISE_CONFIGURATION)
    partner["paired_with"] = partner["specimen_id"].to_numpy(dtype=np.int64)
    partner["specimen_id"] = twin_ids[q0, :n0].astype(np.int64)
    partner["configuration"] = NO_NOISE_CONFIGURATION
    partner["arm"] = arm_of(NO_NOISE_CONFIGURATION, ih_arm)
    partner["with_noise"] = WITH_NOISE[NO_NOISE_CONFIGURATION]
    if n0:
        df = pd.concat([df, partner], ignore_index=True)
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
