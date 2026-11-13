"""CSV files of the per-node features -- Block 23 in specs/SPEC.md (wave 1; D-129, D-130).

One header line, then one row per node in the table's order, columns COLUMNS:
node_id, swc_type (integers); d_um, mu_per_um, dx_um, dy_um, dz_um (floats
written with repr of a Python float, the shortest string that reads back to
the same double, so the round trip is bit for bit); source, flags (strings,
csv quoting where needed); round_index (the same integer in every row). LF
line endings, UTF-8. A table with no node is refused (its round_index would
not survive the round trip).

Library calls: csv (standard library), numpy. Pure ASCII, for the cluster.
"""
from __future__ import annotations

import csv

import numpy as np

from ..analysis.node_features import NodeFeatures

COLUMNS = ("node_id", "swc_type", "d_um", "mu_per_um", "dx_um", "dy_um", "dz_um",
           "source", "flags", "round_index")


def write_features(path, features):
    """One row per node, floats written with repr; round_index in every row. ValueError on an
    empty table."""
    n = len(features)
    if n == 0:
        raise ValueError("write_features: the table holds no node")
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(COLUMNS)
        for k in range(n):
            o = features.offset_um[k]
            w.writerow([str(int(features.node_id[k])), str(int(features.swc_type[k])),
                        repr(float(features.d_um[k])), repr(float(features.mu_per_um[k])),
                        repr(float(o[0])), repr(float(o[1])), repr(float(o[2])),
                        str(features.source[k]), str(features.flags[k]),
                        str(int(features.round_index))])


def read_features(path):
    """-> NodeFeatures, every array bit for bit as written. ValueError on a wrong header, a
    malformed row, round_index differing between rows, no row, or values NodeFeatures refuses."""
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    if not rows or tuple(rows[0]) != COLUMNS:
        raise ValueError("read_features: %s: header must be %s" % (path, ",".join(COLUMNS)))
    body = rows[1:]
    if not body:
        raise ValueError("read_features: %s: no node" % path)
    ids, typ, d, mu, off, src, flg, rnd = [], [], [], [], [], [], [], set()
    for k, r in enumerate(body):
        if len(r) != len(COLUMNS):
            raise ValueError("read_features: %s line %d: expected %d fields, got %d"
                             % (path, k + 2, len(COLUMNS), len(r)))
        try:
            ids.append(int(r[0]))
            typ.append(int(r[1]))
            d.append(float(r[2]))
            mu.append(float(r[3]))
            off.append((float(r[4]), float(r[5]), float(r[6])))
            rnd.add(int(r[9]))
        except ValueError as exc:
            raise ValueError("read_features: %s line %d: %s" % (path, k + 2, exc)) from None
        src.append(r[7])
        flg.append(r[8])
    if len(rnd) != 1:
        raise ValueError("read_features: %s: round_index differs between rows" % path)
    return NodeFeatures(node_id=np.asarray(ids, dtype=np.int64), swc_type=np.asarray(typ, dtype=np.int64),
                        d_um=np.asarray(d, dtype=np.float64), mu_per_um=np.asarray(mu, dtype=np.float64),
                        offset_um=np.asarray(off, dtype=np.float64).reshape(-1, 3),
                        source=np.asarray(src, dtype=str), flags=np.asarray(flg, dtype=str),
                        round_index=rnd.pop())
