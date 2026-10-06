"""Profiles along a measuring line -- Block 5 in specs/SPEC.md (handoff Eq. 9).

For an origin o = (x, y) in um (global image frame), a unit measuring axis
y_hat and the branch axis e_u (both unit vectors in the image plane), the
profile of a plane I_k is

    I(v_n) = 1/(2A + 1) sum_{a=-A..A} I_k(o + v_n y_hat + a * step * e_u),

v_n = n * step, |n| <= round(profile_half_um / step), step = profile_step_um,
A = round(along_branch_avg_um / step) (A = 0: no averaging). Each sample is
bilinear (scipy.ndimage.map_coordinates, order 1) at the array coordinates
(col, row) = (x / p_x - left, y / p_x - top) of the block's frame (integer
full-resolution coordinates are pixel centres). A sample outside the block is
NaN, never clamped to the edge.

Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def profile_offsets(cfg):
    """v_n (um), n = -N..N, N = round(profile_half_um / profile_step_um)."""
    step = float(cfg.profile_step_um)
    n = int(round(cfg.profile_half_um / step))
    return np.arange(-n, n + 1) * step


def n_along(cfg):
    """A = round(along_branch_avg_um / profile_step_um): lines on each side of the main one."""
    return int(round(cfg.along_branch_avg_um / cfg.profile_step_um))


def sample_profile(plane, frame, origin_xy, y_hat, e_u, v, n_avg=0, step=0.0):
    """Mean of 2 n_avg + 1 parallel bilinear profiles (handoff Eq. 9).

    plane: 2-D array (rows = y); frame: CropFrame-like (left, top, res_um_px
    at downsample 0); origin_xy (um); y_hat, e_u: unit 2-vectors; v (n,) um;
    n_avg lines on each side, step um apart along e_u. Returns float64 (n,),
    NaN where any averaged sample falls outside the plane.
    """
    p = float(frame.res_um_px)
    img = np.asarray(plane, dtype=float)
    v = np.asarray(v, dtype=float)
    yh = np.asarray(y_hat, dtype=float)
    eu = np.asarray(e_u, dtype=float)
    total = np.zeros(v.shape)
    for a in range(-int(n_avg), int(n_avg) + 1):
        x = origin_xy[0] + v * yh[0] + a * step * eu[0]
        y = origin_xy[1] + v * yh[1] + a * step * eu[1]
        cols, rows = x / p - frame.left, y / p - frame.top
        total += ndimage.map_coordinates(img, [rows, cols], order=1, mode="constant", cval=np.nan)
    return total / (2 * int(n_avg) + 1)
