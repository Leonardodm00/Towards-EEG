"""morph_inflate.py -- an SWC copy with the radii of the proximal arbour
multiplied by a factor (SPEC section 2.11, Block 29; D-268).

Pure module: numpy only, no NEURON. Used by the twins' run to build a twin
whose "real" cell is thicker near the soma than the reconstruction the fit
reads.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np

__all__ = ["read_swc", "path_length_from_root", "inflate_swc"]

SWC_SOMA, SWC_AXON, SWC_BASAL, SWC_APICAL = 1, 2, 3, 4


def read_swc(path) -> Tuple[np.ndarray, list]:
    """(rows, header): rows float64 (n, 7) = id, type, x, y, z, radius,
    parent (um; parent -1 for a root); header the comment lines, kept."""
    header, rows = [], []
    with open(path, "r") as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                header.append(line.rstrip("\n"))
                continue
            parts = s.split()
            if len(parts) < 7:
                raise ValueError("%s: an SWC row with %d fields: %r" % (path, len(parts), s))
            rows.append([float(x) for x in parts[:7]])
    if not rows:
        raise ValueError("%s: no SWC row" % (path,))
    arr = np.asarray(rows, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError("%s: a non-finite SWC value" % (path,))
    return arr, header


def path_length_from_root(rows: np.ndarray) -> np.ndarray:
    """float64 (n,), um: for each node the length of the path to its root
    along the parent links (sum of the Euclidean distances between
    consecutive nodes); 0 for a root. ValueError: a duplicated id, a parent
    that is not a node, a cycle."""
    ids = rows[:, 0].astype(np.int64)
    par = rows[:, 6].astype(np.int64)
    if np.unique(ids).size != ids.size:
        raise ValueError("duplicated SWC node id")
    index = {int(i): k for k, i in enumerate(ids)}
    xyz = rows[:, 2:5]
    dist = np.full(ids.size, np.nan)
    for k in range(ids.size):
        chain, j = [], k
        while np.isnan(dist[j]):
            chain.append(j)
            p = int(par[j])
            if p < 0:
                dist[j] = 0.0
                break
            if p not in index:
                raise ValueError("SWC node %d: its parent %d is not a node" % (int(ids[j]), p))
            j = index[p]
            if len(chain) > ids.size:
                raise ValueError("SWC parent links form a cycle")
        for c in reversed(chain):
            if np.isnan(dist[c]):
                q = index[int(par[c])]
                dist[c] = dist[q] + float(np.linalg.norm(xyz[c] - xyz[q]))
    return dist


def inflate_swc(src, dst, *, factor: float, max_path_um: float,
                types: Sequence[int] = (SWC_SOMA, SWC_BASAL, SWC_APICAL)) -> Dict[str, object]:
    """Write `dst`: `src` with radius * factor on every node whose type is in
    `types` and whose path length from its root is <= max_path_um (a hard
    step: * 1 beyond; the soma's root node is at 0). Everything else is
    copied (ids, types, coordinates, parents, the header plus one comment
    line). Returns counts: n_nodes, n_inflated, n_by_type (type -> inflated
    nodes), max_path_um, factor. ValueError: factor not finite and > 0,
    max_path_um not finite and >= 0, src == dst."""
    factor, max_path_um = float(factor), float(max_path_um)
    if not (np.isfinite(factor) and factor > 0.0):
        raise ValueError("factor must be finite and > 0, got %r" % (factor,))
    if not (np.isfinite(max_path_um) and max_path_um >= 0.0):
        raise ValueError("max_path_um must be finite and >= 0, got %r" % (max_path_um,))
    src, dst = Path(src), Path(dst)
    if src.resolve() == dst.resolve():
        raise ValueError("inflate_swc: dst is src (%s)" % (src,))
    rows, header = read_swc(src)
    dist = path_length_from_root(rows)
    typ = rows[:, 1].astype(np.int64)
    sel = np.isin(typ, np.asarray(list(types), dtype=np.int64)) & (dist <= max_path_um)
    out = rows.copy()
    out[sel, 5] = rows[sel, 5] * factor
    lines = list(header)
    lines.append("# morph_inflate: radius * %.10g for types %s within %.10g um of path length from the root"
                 % (factor, ",".join(str(int(t)) for t in types), max_path_um))
    for r in out:
        lines.append("%d %d %.10g %.10g %.10g %.10g %d" % (int(r[0]), int(r[1]), r[2], r[3], r[4], r[5], int(r[6])))
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    return dict(n_nodes=int(rows.shape[0]), n_inflated=int(sel.sum()),
                n_by_type={int(t): int((sel & (typ == t)).sum()) for t in np.unique(typ[sel])},
                max_path_um=max_path_um, factor=factor)
