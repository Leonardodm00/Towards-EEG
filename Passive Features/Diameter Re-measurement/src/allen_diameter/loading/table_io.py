"""Files of the bias table -- Block 6 in specs/SPEC.md.

Replicate rows: CSV, one file per chunk (rows_<hash>_<start>_<stop>.csv).
Table: <stem>.npz (arrays) + <stem>.json (signature, ranges, smoothing,
counts); the spline is rebuilt on load from the stored inputs (deterministic).

Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import csv
import json
import os

import numpy as np

from ..analysis.table import BiasTable


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


def save_table(table, stem):
    np.savez(stem + ".npz", X_kept=table.X_kept, ratio_kept=table.ratio_kept, X_all=table.X_all,
             ok_all=table.ok_all, cv_grid=table.cv_grid, cv_scores=table.cv_scores)
    meta = dict(smoothing=table.smoothing, bandwidth=list(table.bandwidth), d_range=list(table.d_range),
                phi_range_rad=list(table.phi_range_rad), estimator_hash=table.estimator_hash,
                n_all=int(table.n_rows), n_failure_counted=int(table.X_all.shape[0]),
                n_kept=int(table.X_kept.shape[0]), signature=table.signature)
    with open(stem + ".json", "w") as f:
        json.dump(meta, f, sort_keys=True, indent=1)


def load_table(stem):
    with open(stem + ".json") as f:
        meta = json.load(f)
    a = np.load(stem + ".npz")
    return BiasTable(a["X_kept"], a["ratio_kept"], a["X_all"], a["ok_all"], meta["smoothing"], meta["bandwidth"],
                     meta["d_range"], meta["phi_range_rad"], meta["signature"], meta["estimator_hash"],
                     a["cv_grid"], a["cv_scores"], n_rows=meta.get("n_all"))
