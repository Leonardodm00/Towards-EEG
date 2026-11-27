"""The screening export's files -- Block 61 in specs/SPEC.md (section 5, "The screening export"). File reading
and writing only.

Implemented 2026-11-27 (wave 10) against the contract of commit 500426e. The views file: one compressed .npz,
no pickled objects; per view q the keys v<q>_<field> (scalars as 0-d arrays, status as a str array), plus
n_views and format. CSV files: UTF-8, LF line endings, floats written with repr (exact round trip, nan for NaN).
Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import csv
import dataclasses
import os

import numpy as np

from ..analysis.screening import NodeView

INDEX_COLUMNS = ("specimen_id", "node_id", "set", "allen_diameter_um", "k_star", "phi_deg", "n_planes", "n_ok",
                 "B_focus_gl", "omega_focus_um2")

_FORMAT = "screening_views_v1"
_INT_FIELDS = ("cell_id", "node_id", "k_star", "left", "top")
_FLOAT_FIELDS = ("allen_diameter_um", "res0_um", "cx_um", "cy_um", "theta_rad", "phi_rad")
_STR_FIELDS = ("set",)
_TUPLE_FIELDS = ("status",)
_FIELDS = tuple(f.name for f in dataclasses.fields(NodeView))
_ARRAY_FIELDS = tuple(n for n in _FIELDS if n not in _INT_FIELDS + _FLOAT_FIELDS + _STR_FIELDS + _TUPLE_FIELDS)


def _replace_atomically(path, write):
    tmp = str(path) + ".part"
    write(tmp)
    os.replace(tmp, path)


def save_views(path, views) -> None:
    """One compressed .npz with every field of every NodeView of a cell; load_views returns them exactly."""
    out = {"format": np.array(_FORMAT), "n_views": np.array(len(views), dtype=np.int64)}
    for q, view in enumerate(views):
        pre = "v%d_" % q
        for name in _FIELDS:
            x = getattr(view, name)
            if name in _INT_FIELDS:
                out[pre + name] = np.array(int(x), dtype=np.int64)
            elif name in _FLOAT_FIELDS:
                out[pre + name] = np.array(float(x), dtype=np.float64)
            elif name in _STR_FIELDS:
                out[pre + name] = np.array(str(x))
            elif name in _TUPLE_FIELDS:
                out[pre + name] = np.array([str(s) for s in x], dtype=str)
            else:
                out[pre + name] = np.asarray(x)

    def write(tmp):
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **out)
    _replace_atomically(path, write)


def load_views(path):
    """list[NodeView] as saved."""
    with np.load(path, allow_pickle=False) as z:
        if "format" not in z.files or str(z["format"][()]) != _FORMAT:
            raise ValueError("%s: not a screening views file (no format %r)" % (path, _FORMAT))
        views = []
        for q in range(int(z["n_views"][()])):
            pre = "v%d_" % q
            kw = {}
            for name in _FIELDS:
                a = z[pre + name]
                if name in _INT_FIELDS:
                    kw[name] = int(a[()])
                elif name in _FLOAT_FIELDS:
                    kw[name] = float(a[()])
                elif name in _STR_FIELDS:
                    kw[name] = str(a[()])
                elif name in _TUPLE_FIELDS:
                    kw[name] = tuple(str(s) for s in a.tolist())
                else:
                    kw[name] = np.array(a)
            views.append(NodeView(**kw))
    return views


def _cell_text(x):
    if isinstance(x, (bool, np.bool_)):
        raise ValueError("a boolean is not an index value: %r" % (x,))
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        return repr(float(x))
    return str(x)


def write_index(path, rows) -> None:
    """CSV with INDEX_COLUMNS in that order, one row per node (rows: screening.index_row's dicts)."""
    rows = list(rows)
    for i, row in enumerate(rows):
        keys = set(row)
        if keys != set(INDEX_COLUMNS):
            raise ValueError("index row %d: columns %s missing, %s unexpected"
                             % (i, sorted(set(INDEX_COLUMNS) - keys), sorted(keys - set(INDEX_COLUMNS))))

    def write(tmp):
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(INDEX_COLUMNS)
            for row in rows:
                w.writerow([_cell_text(row[c]) for c in INDEX_COLUMNS])
    _replace_atomically(path, write)


def write_candidates(path, views) -> None:
    """The header "specimen_id,node_id" and one row per view, near set first then by node id: Block 52's
    chosen-nodes format (step2_io.read_chosen_nodes reads it back), for the user to cut down (D-278.2)."""
    order = sorted(views, key=lambda v: (0 if v.set == "near" else 1, int(v.node_id)))

    def write(tmp):
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            f.write("specimen_id,node_id\n")
            for v in order:
                f.write("%d,%d\n" % (int(v.cell_id), int(v.node_id)))
    _replace_atomically(path, write)
