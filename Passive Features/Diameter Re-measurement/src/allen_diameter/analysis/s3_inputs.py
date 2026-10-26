"""S3b's pooled suggestions for the simulator's inputs -- Block 18.4 in
specs/SPEC.md (D-065.2-3, D-085.8-9, D-086). Pure functions of per-cell
records; nothing here reads or writes a file. Every value is a suggestion:
the production configuration takes it only from the user (Cell 8, D-065).

  robust_sd  1.4826 x the median absolute deviation from the median
  d_max      the largest Allen dendrite diameter over the cells (D-086)
  jitter     the robust SDs of the ON nodes' offsets, pooled (D-085.8)
  camera     the common table set, the median background and noise SD (D-085.9)

Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import math

import numpy as np


def robust_sd(x):
    """1.4826 median|x - median(x)| over the finite x; NaN when none."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def _finite(v):
    v = float(v)
    return v if math.isfinite(v) else None


def d_max(cells):
    """D-086: cells = [{specimen_id, node_ids, D_allen}] (D_allen um, the
    dendrite nodes' 2r). Returns the maximum over every cell with its cell
    and node, each cell's maximum, the pooled p99 and p99.5 (comparisons)."""
    per, pooled = [], []
    for c in cells:
        D = np.asarray(c["D_allen"], dtype=float)
        ok = np.isfinite(D)
        if not ok.any():
            continue
        k = int(np.argmax(np.where(ok, D, -np.inf)))
        per.append(dict(specimen_id=int(c["specimen_id"]), max_um=float(D[k]), node_id=int(c["node_ids"][k]),
                        n=int(ok.sum())))
        pooled.append(D[ok])
    if not per:
        return dict(value_um=None, note="no cell with Allen diameters")
    allD = np.concatenate(pooled)
    top = max(per, key=lambda r: r["max_um"])
    return dict(value_um=top["max_um"], specimen_id=top["specimen_id"], node_id=top["node_id"],
                rule="max over the cells of Allen's dendrite 2r, SWC types 3 and 4 (D-086)",
                p99_um=float(np.percentile(allD, 99.0)), p99_5_um=float(np.percentile(allD, 99.5)),
                n_nodes=int(allD.size), per_cell=sorted(per, key=lambda r: r["specimen_id"]))


def jitter(cells, z0_flag_um):
    """D-085.8: cells = [{specimen_id, s_star, dz_star}] -- the ON nodes'
    lateral and depth offsets (um). Pooled robust SDs, and per cell the
    robust SDs, the medians, the count; a cell with |median dz*| above
    z0_flag_um is flagged z0."""
    per, s_all, z_all = [], [], []
    for c in cells:
        s = np.asarray(c["s_star"], dtype=float)
        z = np.asarray(c["dz_star"], dtype=float)
        s, z = s[np.isfinite(s)], z[np.isfinite(z)]
        mz = float(np.median(z)) if z.size else float("nan")
        per.append(dict(specimen_id=int(c["specimen_id"]), n_on=int(max(s.size, z.size)), sd_xy_um=_finite(robust_sd(s)),
                        sd_z_um=_finite(robust_sd(z)), median_s_um=_finite(np.median(s)) if s.size else None,
                        median_dz_um=_finite(mz), z0_flag=bool(math.isfinite(mz) and abs(mz) > z0_flag_um)))
        s_all.append(s)
        z_all.append(z)
    s_all = np.concatenate(s_all) if s_all else np.zeros(0)
    z_all = np.concatenate(z_all) if z_all else np.zeros(0)
    return dict(jitter_xy_um=_finite(robust_sd(s_all)), jitter_z_um=_finite(robust_sd(z_all)),
                n_on_xy=int(s_all.size), n_on_z=int(z_all.size),
                rule="1.4826 MAD of the ON nodes' s* and dz*, pooled across the cells (D-085.8)",
                z0_flagged=[r["specimen_id"] for r in per if r["z0_flag"]],
                per_cell=sorted(per, key=lambda r: r["specimen_id"]))


def _key(tables):
    return None if not tables else tuple(tuple(int(v) for v in t) for t in tables)


def camera(cells):
    """D-085.9: cells = [{specimen_id, background_B_gl, noise_sd_gl,
    jpeg_qtables}] (None where a cell has no value). The most common table
    set (its count), the median background and the median noise SD."""
    groups = {}
    for c in cells:
        k = _key(c.get("jpeg_qtables"))
        if k is not None:
            groups.setdefault(k, []).append(int(c["specimen_id"]))
    B = [float(c["background_B_gl"]) for c in cells if c.get("background_B_gl") is not None]
    N = [float(c["noise_sd_gl"]) for c in cells if c.get("noise_sd_gl") is not None]
    best = max(groups.items(), key=lambda kv: (len(kv[1]), -min(kv[1]))) if groups else None
    return dict(background_B_gl=_finite(np.median(B)) if B else None, noise_sd_gl=_finite(np.median(N)) if N else None,
                jpeg_qtables=[list(t) for t in best[0]] if best else None,
                jpeg_cells=sorted(best[1]) if best else [], n_table_sets=len(groups),
                n_cells=len(cells), n_background=len(B), n_noise=len(N),
                rule="one setting for every cell: the most common table set, the medians over the cells (D-085.9); "
                     "the noise SD an upper bound")
