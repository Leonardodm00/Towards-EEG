"""Background of the fit -- Block 5 in specs/SPEC.md (D-018.1, D-023).

B_bar = median of the node's block in plane k* over the pixels whose centre
lies farther than (radius + bbar_mask_margin_um) from every segment of the
branch (distance in the image plane; a segment uses the larger radius of its
two ends). Pixel (row i, column j) of a block with frame (left, top) is
centred at x = (left + j) p_x, y = (top + i) p_x.

Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import numpy as np


def segment_distance(px, py, a, b):
    """Distance (um) from points (px, py) to the segment a-b in the plane.
    Custom: a two-line closed form; no library call for point-segment distance."""
    ax, ay = float(a[0]), float(a[1])
    dx, dy = float(b[0]) - ax, float(b[1]) - ay
    L2 = dx * dx + dy * dy
    if L2 == 0.0:
        return np.hypot(px - ax, py - ay)
    t = np.clip(((px - ax) * dx + (py - ay) * dy) / L2, 0.0, 1.0)
    return np.hypot(px - (ax + t * dx), py - (ay + t * dy))


def mask_near_branch(shape, left, top, res_um, xyz_um, radius_um, margin_um):
    """Boolean (H, W): True where a pixel centre -- (left + column, top + row)
    x res_um -- is within radius + margin of a segment between consecutive
    nodes (or of the node, for one node)."""
    H, W = int(shape[0]), int(shape[1])
    p = float(res_um)
    rows, cols = np.mgrid[0:H, 0:W]
    px, py = (left + cols) * p, (top + rows) * p
    P = np.asarray(xyz_um, dtype=float)
    r = np.asarray(radius_um, dtype=float)
    mask = np.zeros((H, W), dtype=bool)
    if P.shape[0] == 1:
        return np.hypot(px - P[0, 0], py - P[0, 1]) <= r[0] + margin_um
    for j in range(P.shape[0] - 1):
        mask |= segment_distance(px, py, P[j, :2], P[j + 1, :2]) <= max(r[j], r[j + 1]) + margin_um
    return mask


def masked_median(plane, mask, min_frac):
    """(B_bar, unmasked fraction, ok): median over pixels where mask is False
    (B_bar is NaN when none); ok = unmasked fraction >= min_frac."""
    a = np.asarray(plane, dtype=float)
    keep = ~np.asarray(mask, dtype=bool)
    frac = float(keep.mean())
    if not keep.any():
        return float("nan"), 0.0, False
    return float(np.median(a[keep])), frac, frac >= min_frac
