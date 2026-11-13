"""Independent references for the spec-derived tests of Blocks 16-18 (tester
run D5, 2026-11-13). Written from specs/SPEC.md only: nothing here calls the
code under test.

  ref_select            16.1 (d): sort the ids, permute with
                        default_rng([seed, crc32(key)]), take the first n
  ref_search_score      16.2 (c): best sum of D under a mask moved by every
                        integer shift |dx|, |dy| <= M (zero outside D), both
                        masks, divided by the mask's pixel count (brute force)
  ref_fine_search       16.2 (d): S(a, b) = mean of D over the mask moved by a
                        columns and b rows, pixels outside D left out (brute force)
  ref_place_window      16.3 (b): x0 = min(max(floor(c + 1/2) - h, s), X - 2h - s)
  ref_robust_sd         18.4: 1.4826 median |x - median x|
  ref_allen_bin         17.1: the bin of 2 r among [0, e1), [e1, e2), ..., [e_last, inf)
  ref_stratified        17.1, the natural reading: per bin one Generator
                        default_rng([seed, sid, b]); first the permutation of
                        the bin's stretches, then, stretch by stretch in that
                        permuted order, the permutation of its nodes; round-robin
  write_swc_text        a small SWC file from rows (id, type, x, y, z, r, parent)
  star_swc              a soma with straight unbranched dendrites (one stretch each)

Pure ASCII.
"""
from __future__ import annotations

import math
import os
import sys
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
for _p in (WS / "src", WS / "scripts", WS / "tests" / "smoke"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


# ------------------------------------------------------------ Block 16.1 ----

def ref_select(ids, seed, key, n):
    """[(specimen_id, rank)] of the group `key` (16.1 (d))."""
    ids = sorted(int(i) for i in ids)
    rng = np.random.default_rng([int(seed), zlib.crc32(key.encode("ascii"))])
    perm = rng.permutation(len(ids))
    return [(ids[int(j)], r) for r, j in enumerate(perm[:min(int(n), len(ids))])]


# ------------------------------------------------------------ Block 16.2 ----

def ref_search_score(D, masks, M):
    D = np.asarray(D, dtype=float)
    H, W = D.shape
    best = -np.inf
    for S in masks:
        r, c = np.nonzero(S)
        if r.size == 0:
            continue
        for dy in range(-M, M + 1):
            for dx in range(-M, M + 1):
                rr, cc = r + dy, c + dx
                ok = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
                best = max(best, float(D[rr[ok], cc[ok]].sum()) / r.size)
    return best


def ref_fine_search(D, mask, A):
    """(a*, b*, S*, S00, S) with S[b + A, a + A]; ties broken by the first in
    row-major order of S (not fixed by the spec: only S* and S00 are compared
    when a tie exists)."""
    D = np.asarray(D, dtype=float)
    H, W = D.shape
    r, c = np.nonzero(mask)
    S = np.full((2 * A + 1, 2 * A + 1), -np.inf)
    for b in range(-A, A + 1):
        for a in range(-A, A + 1):
            rr, cc = r + b, c + a
            ok = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
            if ok.any():
                S[b + A, a + A] = D[rr[ok], cc[ok]].mean()
    j = np.unravel_index(int(np.argmax(S)), S.shape)
    return int(j[1] - A), int(j[0] - A), float(S[j]), float(S[A, A]), S


# ------------------------------------------------------------ Block 16.3 ----

def ref_place_window(cx, cy, h, s, X, Y):
    if X < 2 * (h + s) or Y < 2 * (h + s):
        return None
    x0 = min(max(int(math.floor(cx + 0.5)) - h, s), X - 2 * h - s)
    y0 = min(max(int(math.floor(cy + 0.5)) - h, s), Y - 2 * h - s)
    return x0, y0


def ref_shift_search(P_big, O, s):
    """Every r(a, b) by numpy.corrcoef and the first maximum in the order
    b = -s..s, then a = -s..s."""
    H, W = O.shape
    best = None
    for b in range(-s, s + 1):
        for a in range(-s, s + 1):
            p = P_big[s + b:s + b + H, s + a:s + a + W].ravel().astype(float)
            if np.ptp(p) == 0 or np.ptp(O) == 0:
                continue
            r = float(np.corrcoef(p, O.ravel().astype(float))[0, 1])
            if best is None or r > best[0]:
                best = (r, a, b)
    return best


# ------------------------------------------------------------ Block 18 ------

def ref_robust_sd(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return 1.4826 * float(np.median(np.abs(x - np.median(x))))


# ------------------------------------------------------------ Block 17 ------

def ref_allen_bin(radius, inner_edges):
    r = float(radius)
    if not math.isfinite(r) or r < 0:
        return None
    D = 2.0 * r
    edges = [0.0] + [float(e) for e in inner_edges] + [math.inf]
    for b in range(len(edges) - 1):
        if edges[b] <= D < edges[b + 1]:
            return b
    return None


def ref_stratified(stretches, radius_of, inner_edges, seed, sid, N):
    """stretches: [[node ids, proximal to distal]] in the order the cell lists
    them; radius_of: {id: Allen radius}. Returns (ids, {b: (available, taken)})."""
    by_bin = {}
    for run in stretches:
        groups = {}
        for nid in run:
            b = ref_allen_bin(radius_of[nid], inner_edges)
            if b is not None:
                groups.setdefault(b, []).append(nid)
        for b, ids in groups.items():
            by_bin.setdefault(b, []).append(ids)
    out, counts = [], {}
    for b in sorted(by_bin):
        rng = np.random.default_rng([int(seed), int(sid), int(b)])
        runs = [by_bin[b][int(i)] for i in rng.permutation(len(by_bin[b]))]
        runs = [[ids[int(j)] for j in rng.permutation(len(ids))] for ids in runs]
        n_av = sum(len(x) for x in runs)
        taken, depth = [], 0
        while len(taken) < min(N, n_av):
            for ids in runs:
                if depth < len(ids) and len(taken) < N:
                    taken.append(ids[depth])
            depth += 1
        out += taken
        counts[b] = (n_av, len(taken))
    return out, counts


# -------------------------------------------------------------- SWC files ---

def write_swc_text(path, rows):
    with open(path, "w") as f:
        f.write("# synthetic (tester D5)\n")
        for (i, t, x, y, z, r, p) in rows:
            f.write("%d %d %.4f %.4f %.4f %.4f %d\n" % (i, t, x, y, z, r, p))
    return path


def star_swc(path, radii_per_branch, step_um=1.0, types=None):
    """A soma (id 1, type 1) and one straight dendrite per entry of
    radii_per_branch (its nodes' Allen radii, proximal to distal), each
    attached to the soma, so each dendrite is one unbranched stretch.
    Returns (path, [[ids of branch 0], [ids of branch 1], ...])."""
    rows = [(1, 1, 0.0, 0.0, 0.0, 5.0, -1)]
    nid, branches = 2, []
    nb = len(radii_per_branch)
    for b, radii in enumerate(radii_per_branch):
        ang = 2 * math.pi * b / max(nb, 1)
        par, ids = 1, []
        t = 3 if types is None else types[b]
        for k, r in enumerate(radii):
            d = 8.0 + step_um * k
            rows.append((nid, t, d * math.cos(ang), d * math.sin(ang), 0.0, float(r), par))
            ids.append(nid)
            par = nid
            nid += 1
        branches.append(ids)
    write_swc_text(path, rows)
    return path, branches
