"""SWC reading and format-preserving writing -- Block 8 (loading half) in specs/SPEC.md.

Reads an SWC file into arrays and writes it back with new radii while keeping
every other byte of the file unchanged (decision D-013: the corrected
morphology is "a new swc file exactly in the same format as the original").

[2026-11-22, wave 6 (D-222.5): `edited_lines` and `write_swc`'s `xyz`, `parent`
and `keep` also move nodes, re-parent them and drop rows (SPEC Block 47); with
`radius` alone the writer is unchanged.] Only the radius field (the 6th
whitespace-separated token of a data line) is rewritten by `rewrite_radius_lines`; comment lines, blank lines, line endings, field separators
and the other six fields are copied verbatim. A radius is formatted with the
same number of decimals as the token it replaces (at least
``MIN_DECIMALS``), so the file keeps its own precision convention.

Library calls: numpy only. Custom code: the tokeniser that keeps the
separators (``_split_keep``), because ``str.split`` discards them and a
pandas round trip cannot promise byte identity.

Units: SWC coordinates and radii are in um (Allen convention; handoff
"Coordinate transform"). Node ids are the file's own (1-based in Allen files;
not assumed here).

Pure ASCII (hpc-python-compat).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

MIN_DECIMALS = 4
_TOKEN = re.compile(r"(\s*)(\S+)")


@dataclass
class SWC:
    """One SWC file: arrays for computation, raw lines for writing.

    Attributes
        ids, types, parent : int arrays, shape (N,)
        xyz                : float array, shape (N, 3), um
        radius             : float array, shape (N,), um
        lines              : every line of the file, verbatim (with its own
                             line ending); ``data_rows[i]`` is the index in
                             ``lines`` of node i
        path               : where it was read from ("" if built in memory)
    """

    ids: np.ndarray
    types: np.ndarray
    xyz: np.ndarray
    radius: np.ndarray
    parent: np.ndarray
    lines: List[str]
    data_rows: List[int]
    path: str = ""

    def __len__(self) -> int:
        return int(self.ids.size)

    @property
    def diameter(self) -> np.ndarray:
        return 2.0 * self.radius

    def dendrite_mask(self, types: Sequence[int] = (3, 4)) -> np.ndarray:
        """Boolean mask of the nodes whose SWC type is in ``types``
        (3 basal, 4 apical, handoff step 1)."""
        return np.isin(self.types, np.asarray(list(types), dtype=int))

    def index_of(self) -> dict:
        """Map node id -> row index."""
        return {int(i): k for k, i in enumerate(self.ids)}

    def parent_index(self) -> np.ndarray:
        """Row index of each node's parent; -1 for a root."""
        idx = self.index_of()
        return np.array([idx.get(int(p), -1) for p in self.parent], dtype=int)


def read_swc(path: str) -> SWC:
    """Parse an SWC file. Data lines have 7 fields: id type x y z radius parent.

    Raises ValueError on a data line with fewer than 7 tokens or a
    non-numeric field, naming the line number.
    """
    with open(path, "r", encoding="ascii", errors="strict", newline="") as f:
        lines = f.read().splitlines(keepends=True)
    ids, types, xyz, radius, parent, rows = [], [], [], [], [], []
    for k, line in enumerate(lines):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        tok = s.split()
        if len(tok) < 7:
            raise ValueError("%s line %d: expected 7 fields, got %d" % (path, k + 1, len(tok)))
        try:
            ids.append(int(tok[0]))
            types.append(int(tok[1]))
            xyz.append((float(tok[2]), float(tok[3]), float(tok[4])))
            radius.append(float(tok[5]))
            parent.append(int(tok[6]))
        except ValueError as exc:
            raise ValueError("%s line %d: %s" % (path, k + 1, exc)) from None
        rows.append(k)
    return SWC(ids=np.asarray(ids, dtype=int), types=np.asarray(types, dtype=int),
               xyz=np.asarray(xyz, dtype=float).reshape(-1, 3), radius=np.asarray(radius, dtype=float),
               parent=np.asarray(parent, dtype=int), lines=lines, data_rows=rows, path=str(path))


def _split_keep(s: str) -> List[Tuple[str, str]]:
    """Tokenise ``s`` into (leading_whitespace, token) pairs; the trailing
    whitespace (including the line ending) is returned as a final
    ('', '') pair's leading part so that ''.join restores the line."""
    out = []
    pos = 0
    for m in _TOKEN.finditer(s):
        out.append((m.group(1), m.group(2)))
        pos = m.end()
    out.append((s[pos:], ""))
    return out


def _format_like(value: float, template: str) -> str:
    """Format ``value`` with the decimals of ``template`` (>= MIN_DECIMALS)."""
    if "." in template and "e" not in template.lower():
        decimals = max(len(template.split(".")[1]), MIN_DECIMALS)
    else:
        decimals = MIN_DECIMALS
    return "%.*f" % (decimals, value)


def rewrite_radius_lines(swc: SWC, radius: np.ndarray) -> List[str]:
    """Return the file's lines with the radius field of each data line
    replaced by ``radius[i]`` and nothing else changed."""
    radius = np.asarray(radius, dtype=float)
    if radius.shape != (len(swc),):
        raise ValueError("radius must have shape (%d,), got %s" % (len(swc), radius.shape))
    if not np.all(np.isfinite(radius)) or np.any(radius <= 0):
        raise ValueError("every radius must be finite and > 0")
    lines = list(swc.lines)
    for i, row in enumerate(swc.data_rows):
        parts = _split_keep(lines[row])
        # parts[5] is the 6th token (radius); the final element is trailing whitespace
        lead, tok = parts[5]
        parts[5] = (lead, _format_like(float(radius[i]), tok))
        lines[row] = "".join(a + b for a, b in parts)
    return lines


def _parent_ids(parent, n: int) -> np.ndarray:
    """``parent`` as an int64 array of shape (n,): integers, or floats with integral values (not bools)."""
    a = np.asarray(parent)
    if a.shape != (n,):
        raise ValueError("parent must have shape (%d,), got %s" % (n, a.shape))
    if a.dtype.kind in "iu":
        return a.astype(np.int64)
    if a.dtype.kind == "f" and np.all(np.isfinite(a)) and np.all(a == np.floor(a)):
        return a.astype(np.int64)
    raise ValueError("parent must hold integer node ids, got dtype %s" % a.dtype)


def edited_lines(swc: SWC, xyz: Optional[np.ndarray] = None, radius: Optional[np.ndarray] = None,
                 parent: Optional[np.ndarray] = None, keep: Optional[np.ndarray] = None) -> List[str]:
    """The file's lines (SPEC Block 47, wave 6; D-222.5) with, on the kept data lines, x, y, z (tokens 3-5), the
    radius (6) and the parent (7) replaced where given (floats with _format_like, the parent as an integer), the removed
    rows' lines dropped and every other line verbatim (comments, blank lines, separators, line endings, other tokens).
    xyz (N, 3) um, radius (N,) um, parent (N,) ids, keep (N,) bool; None leaves that field (or every row) as read.
    With radius alone the result is rewrite_radius_lines(swc, radius). ValueError for shapes that disagree, a
    non-finite coordinate, a radius <= 0 or not finite, parent ids that are not integers, keep not bool, a kept row
    whose parent is removed, a parent id absent from the kept rows (-1 allowed)."""
    n = len(swc)
    X = R = PA = None
    if xyz is not None:
        X = np.asarray(xyz, dtype=float)
        if X.shape != (n, 3):
            raise ValueError("xyz must have shape (%d, 3), got %s" % (n, X.shape))
        if not np.all(np.isfinite(X)):
            raise ValueError("every coordinate must be finite")
    if radius is not None:
        R = np.asarray(radius, dtype=float)
        if R.shape != (n,):
            raise ValueError("radius must have shape (%d,), got %s" % (n, R.shape))
        if not np.all(np.isfinite(R)) or np.any(R <= 0):
            raise ValueError("every radius must be finite and > 0")
    if parent is not None:
        PA = _parent_ids(parent, n)
    if keep is None:
        K = np.ones(n, dtype=bool)
    else:
        K = np.asarray(keep)
        if K.shape != (n,) or K.dtype != np.bool_:
            raise ValueError("keep must be a bool array of shape (%d,), got %s %s" % (n, K.dtype, K.shape))
    ids = np.asarray(swc.ids, dtype=np.int64)
    par = PA if PA is not None else np.asarray(swc.parent, dtype=np.int64)
    kept_ids = set(int(i) for i in ids[K])
    removed_ids = set(int(i) for i in ids[~K])
    for i in np.flatnonzero(K):
        p = int(par[i])
        if p == -1 or p in kept_ids:
            continue
        if p in removed_ids:
            raise ValueError("node %d (row %d) is kept but its parent %d is removed" % (ids[i], i, p))
        raise ValueError("node %d (row %d): parent id %d is not a kept node" % (ids[i], i, p))
    # Custom: an SWC editor that keeps every byte it does not edit; no SWC library promises byte identity (checked:
    # str.split, a pandas round trip). It reuses Block 8's separator-keeping tokeniser, _split_keep.
    node_of_line = {line_no: i for i, line_no in enumerate(swc.data_rows)}
    out = []
    for k, line in enumerate(swc.lines):
        i = node_of_line.get(k)
        if i is None:
            out.append(line)                               # a comment or blank line: verbatim
            continue
        if not K[i]:
            continue                                       # a removed row: its line dropped
        if X is None and R is None and PA is None:
            out.append(line)
            continue
        parts = _split_keep(line)
        # parts[2:5] are x, y, z, parts[5] the radius, parts[6] the parent; the final element is trailing whitespace
        if X is not None:
            for j in range(3):
                lead, tok = parts[2 + j]
                parts[2 + j] = (lead, _format_like(float(X[i, j]), tok))
        if R is not None:
            lead, tok = parts[5]
            parts[5] = (lead, _format_like(float(R[i]), tok))
        if PA is not None:
            lead, _tok = parts[6]
            parts[6] = (lead, "%d" % int(PA[i]))
        out.append("".join(a + b for a, b in parts))
    return out


def write_swc(swc: SWC, path: str, radius: Optional[np.ndarray] = None, xyz: Optional[np.ndarray] = None,
              parent: Optional[np.ndarray] = None, keep: Optional[np.ndarray] = None) -> None:
    """Write ``swc`` to ``path``; with ``radius`` alone, only the radius
    field of every data line changes (D-013); with ``xyz``, ``parent`` or
    ``keep`` the lines of ``edited_lines`` (SPEC Block 47, wave 6)."""
    if xyz is not None or parent is not None or keep is not None:
        lines = edited_lines(swc, xyz=xyz, radius=radius, parent=parent, keep=keep)
    else:
        lines = swc.lines if radius is None else rewrite_radius_lines(swc, radius)
    with open(path, "w", encoding="ascii", newline="") as f:
        f.write("".join(lines))


def segment_lengths_um(swc: SWC) -> np.ndarray:
    """Length (um) of the segment from each node to its parent; 0 for roots."""
    pidx = swc.parent_index()
    seg = np.zeros(len(swc), dtype=float)
    has = pidx >= 0
    seg[has] = np.linalg.norm(swc.xyz[has] - swc.xyz[pidx[has]], axis=1)
    return seg


def frustum_areas_um2(swc: SWC, radius: Optional[np.ndarray] = None) -> np.ndarray:
    """Lateral area (um^2) of the frustum from each node to its parent, with
    the node's and the parent's radii as the two end radii (textbook:
    pi (r1 + r2) sqrt(h^2 + (r1 - r2)^2)); 0 for roots. ``radius``
    overrides the file's radii (e.g. the corrected ones)."""
    r = swc.radius if radius is None else np.asarray(radius, dtype=float)
    pidx = swc.parent_index()
    h = segment_lengths_um(swc)
    area = np.zeros(len(swc), dtype=float)
    has = pidx >= 0
    r1, r2 = r[has], r[pidx[has]]
    area[has] = np.pi * (r1 + r2) * np.sqrt(h[has] ** 2 + (r1 - r2) ** 2)
    return area
