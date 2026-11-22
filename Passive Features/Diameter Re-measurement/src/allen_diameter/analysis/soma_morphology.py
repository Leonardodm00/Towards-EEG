"""The soma in the morphology -- Block 47 in specs/SPEC.md (D-215.2, D-216.3, D-222.5). CONTRACT STUB (wave 6).

The fitted ellipsoid (Block 46) enters the reconstruction as the sphere of equal membrane area, r_eq = sqrt(S / 4 pi),
at the fitted centre (the SWC's first type-1 node moved there; any other type-1 node removed, its children re-parented);
each stem's first node moves along the stem's own direction onto the fitted surface; when the stem's next nodes lie
inside the surface they are removed and the first node is placed where the stem leaves it (D-222.5). Frames: the image
frame of Blocks 44-46 and the SWC frame (analysis.cell.to_image_um and its inverse here). See SPEC Block 47.

No file access, no plotting. Pure ASCII, LF, for the cluster.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

Q_SURFACE_TOL = 1e-12        # SPEC Block 47: |Q| <= this counts as on the surface


@dataclass(frozen=True)
class SomaEdit:
    """The corrected soma (SPEC Block 47). xyz (N, 3) SWC frame um, radius (N,) um, parent (N,) ids, keep (N,) bool --
    every row of the input SWC, edited where moved or re-parented; soma_row the kept type-1 row; stems a tuple of dicts
    (row, id, type, flag in {"moved", "on surface", "missed", "branch inside", "ends inside"}, old_um (3,), new_um (3,),
    moved_um, removed_ids); removed_soma_ids the other type-1 nodes' ids."""
    xyz: np.ndarray
    radius: np.ndarray
    parent: np.ndarray
    keep: np.ndarray
    soma_row: int
    stems: Tuple[dict, ...]
    removed_soma_ids: Tuple[int, ...]


def image_to_swc_um(xyz_img, res0_um, shift_full_px=(0.0, 0.0), flip_y_full_h=None, z0_um=0.0):
    """-> (n, 3) float64 um in the SWC frame: the inverse of analysis.cell.to_image_um with the same arguments."""
    raise NotImplementedError("Block 47 (wave 6)")


def fit_frame(P_img, theta_g):
    """-> (n, 3) um: the points P_img (n, 3) in the fitted ellipsoid's frame, (u, v, w) = Block 45's rotated in-plane
    coordinates about (x0, y0) and w = z - z0. theta_g (8,) as Block 45."""
    raise NotImplementedError("Block 47 (wave 6)")


def ellipsoid_q(P_img, theta_g):
    """-> (n,): Q = (u/a)^2 + (v/b)^2 + (w/c)^2 - 1 at the points P_img (n, 3) um (negative inside, 0 on the surface)."""
    raise NotImplementedError("Block 47 (wave 6)")


def line_crossings(P_img, d, theta_g):
    """-> (t_lo, t_hi) (n,) each, t_lo <= t_hi: the roots of Q(P + t d) = 0, NaN where the line misses. P_img (n, 3) um,
    d (n, 3) or (3,) nonzero directions."""
    raise NotImplementedError("Block 47 (wave 6)")


def ellipse_outline(theta_g, z_um=None, n=256):
    """-> (x (n,), y (n,)) um image frame: the fitted silhouette (z_um None) or the section at depth z_um (the silhouette
    scaled by sqrt(1 - ((z - z0)/c)^2); empty (0,) arrays when |z - z0| >= c)."""
    raise NotImplementedError("Block 47 (wave 6)")


def side_outline(theta_g, n=256):
    """-> (u (n,), w (n,)) um: the ellipse (a, c) in the ellipsoid's (u, w) plane."""
    raise NotImplementedError("Block 47 (wave 6)")


def stem_rows(swc):
    """-> ndarray[int]: the rows of the stems' first nodes (parent a type-1 node, own type not 1), ascending."""
    raise NotImplementedError("Block 47 (wave 6)")


def correct_soma(swc, theta_g, r_eq_um, res0_um, shift_full_px=(0.0, 0.0), flip_y_full_h=None, z0_um=0.0):
    """-> SomaEdit: the SWC (loading.swc_io.SWC, not mutated) with its soma replaced by the sphere of radius r_eq_um at
    the fitted centre and its stems' first nodes on the fitted surface (SPEC Block 47); theta_g (8,) in the image frame;
    res0_um and the alignment (shift_full_px, flip_y_full_h, z0_um) as analysis.cell.to_image_um. ValueError for no
    type-1 node, a bad theta_g, r_eq_um not finite and > 0."""
    raise NotImplementedError("Block 47 (wave 6)")
