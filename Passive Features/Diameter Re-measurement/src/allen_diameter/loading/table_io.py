"""Files of the bias table -- Blocks 6 and 14 in specs/SPEC.md.

Replicate rows: CSV, one file per chunk of draws (rows_<hash>_<start>_<stop>.csv).
Table (Block 14.4): <stem>.npz -- the three axes (axis_0, axis_1, axis_2), the
grid arrays of table.ARRAYS and the support mask -- and <stem>.json, the record
(BiasTable.meta: the estimator, the bandwidths and multiplier, the
cross-validation, q_loo, the counts, the signature, the estimator hash); the
interpolants are rebuilt on load (deterministic). The per-draw table
(table.per_draw_table, D-060.2) is a CSV of its own (draws_<hash>.csv).

Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import csv
import json
import os

import numpy as np

from ..analysis.table import ARRAYS, BiasTable

TABLE_FORMAT = "grid-3d-2026-10-21"      # Block 14.4; files without it predate the grid (Block 6's spline)


def _parse(value):
    if value in ("True", "False"):
        return value == "True"
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def write_rows(rows, path):
    """CSV with the union of the rows' keys (first-seen order)."""
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    os.replace(tmp, path)


def read_rows(paths):
    """Rows of one or more CSV files, values parsed to bool / int / float / str."""
    if isinstance(paths, str):
        paths = [paths]
    out = []
    for p in paths:
        with open(p, newline="") as f:
            out += [{k: _parse(v) for k, v in r.items()} for r in csv.DictReader(f)]
    return out


# the columns a merge reads (Block 14.6), by type; a file without an optional one (older runs) gives "" or NaN
INT_COLUMNS = ("draw_index", "realization_index")
FLOAT_COLUMNS = ("d_um", "phi_rad", "meas_phi_rad", "theta_rad", "mu_per_um", "d_hat_um", "mu_hat_per_um", "ratio")
BOOL_COLUMNS = ("in_S",)
STR_COLUMNS = ("sim_hash", "reject")
OPTIONAL_COLUMNS = ("sim_hash", "reject", "theta_rad", "meas_phi_rad")
MERGE_COLUMNS = INT_COLUMNS + FLOAT_COLUMNS + BOOL_COLUMNS + STR_COLUMNS


def read_columns(paths, columns=MERGE_COLUMNS):
    """{column: numpy array} of `columns` over one or more row CSVs, typed by INT_COLUMNS, FLOAT_COLUMNS
    (an empty cell is NaN), BOOL_COLUMNS ("True") and STR_COLUMNS; a file without one of OPTIONAL_COLUMNS gives ""
    (strings) or NaN; any other missing column is a ValueError. Row by row, so that 10^6 rows hold a few arrays,
    not 10^6 dicts (the merge of Block 14.6)."""
    if isinstance(paths, str):
        paths = [paths]
    out = {c: [] for c in columns}
    for p in paths:
        with open(p, newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None) or []
            where = {c: header.index(c) for c in columns if c in header}
            missing = [c for c in columns if c not in where and c not in OPTIONAL_COLUMNS]
            if missing:
                raise ValueError("%s has no column %s" % (p, ", ".join(missing)))
            for row in reader:
                for c in columns:
                    out[c].append(row[where[c]] if c in where else "")
    typed = {}
    for c, v in out.items():
        if c in INT_COLUMNS:
            typed[c] = np.array([int(x) for x in v], dtype=np.int64)
        elif c in FLOAT_COLUMNS:
            typed[c] = np.array([float(x) if x != "" else np.nan for x in v], dtype=float)
        elif c in BOOL_COLUMNS:
            typed[c] = np.array([x == "True" for x in v], dtype=bool)
        else:
            typed[c] = np.array(v, dtype=object)
    return typed


def save_table(table, stem):
    """<stem>.npz (axes, arrays, support) and <stem>.json (the record, with the file format)."""
    np.savez(stem + ".npz", axis_0=table.axes[0], axis_1=table.axes[1], axis_2=table.axes[2],
             supported=table.supported, **{k: table.arrays[k] for k in ARRAYS})
    meta = dict(table.meta, format=TABLE_FORMAT)
    tmp = stem + ".json.tmp"
    with open(tmp, "w") as f:
        json.dump(meta, f, sort_keys=True, indent=1)
    os.replace(tmp, stem + ".json")


def load_table(stem):
    """The BiasTable of save_table; ValueError for a table file written before Block 14.4 (no grid)."""
    with open(stem + ".json") as f:
        meta = json.load(f)
    if meta.get("format") != TABLE_FORMAT:
        raise ValueError("%s.json is a table of format %r, not %r (Block 14.4's grid): rebuild it with "
                         "scripts/build_table.py merge" % (stem, meta.get("format"), TABLE_FORMAT))
    a = np.load(stem + ".npz")
    meta.pop("format")
    return BiasTable((a["axis_0"], a["axis_1"], a["axis_2"]), {k: a[k] for k in ARRAYS}, a["supported"], meta)
