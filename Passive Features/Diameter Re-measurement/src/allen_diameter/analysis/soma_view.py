"""The soma's view in the image stack -- Block 44 in specs/SPEC.md (D-210).

The traced soma (the SWC's type-1 nodes) placed in the image frame, the square crop about it, the recorded planes
about its depth, and a focus measure per plane over a disk about it:

    F_k = Var over the disk { (G_sigma * L)(I_k) }       (L the Laplacian, G_sigma the Gaussian: their product the
                                                          Laplacian of Gaussian, scipy.ndimage.gaussian_laplace)

large where the plane holds sharp structure (the soma's outline and its interior at focus), small where it is
blurred. The sharpest plane is the one of largest F_k. Image frame: pixel (row, col) of a crop with corner
(left, top) is centred on ((left + col) p, (top + row) p) um, p = res0 (plotting.figures.image_extent_um).

No file access, no plotting. Pure ASCII, LF, for the cluster.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import gaussian_laplace


def soma_geometry(xyz_um, radius_um, types):
    """-> (centre (3,) um, radius um): the mean position of the type-1 nodes (image frame, um) and the largest of
    their radii. xyz_um (n, 3); radius_um (n,); types (n,). ValueError without a type-1 node or with a non-finite
    one."""
    P = np.asarray(xyz_um, dtype=float)
    r = np.asarray(radius_um, dtype=float).reshape(-1)
    t = np.asarray(types).reshape(-1)
    if P.ndim != 2 or P.shape[1] != 3 or r.shape[0] != P.shape[0] or t.shape[0] != P.shape[0]:
        raise ValueError("xyz_um must be (n, 3) with radius_um and types (n,)")
    som = t == 1
    if not som.any():
        raise ValueError("no soma node (type 1) in the trace")
    c, rad = P[som].mean(axis=0), float(r[som].max())
    if not (np.all(np.isfinite(c)) and math.isfinite(rad) and rad > 0):
        raise ValueError("the soma node's position or radius is not finite and > 0")
    return c, rad


def crop_box(centre_xy_um, half_um, res0_um):
    """-> (left, top, width, height) full-resolution px: the square of half-size half_um about the centre, its
    corner rounded down and its side rounded up to whole pixels."""
    cx, cy = (float(v) for v in np.asarray(centre_xy_um, dtype=float).reshape(-1)[:2])
    h, p = float(half_um), float(res0_um)
    if not (math.isfinite(cx) and math.isfinite(cy) and h > 0 and p > 0):
        raise ValueError("crop_box needs a finite centre and half_um, res0_um > 0")
    left, top = int(math.floor((cx - h) / p)), int(math.floor((cy - h) / p))
    side = int(math.ceil(2.0 * h / p)) + 1
    return left, top, side, side


def plane_range(z_um, half_um, dz_um):
    """-> (k_lo, k_hi): the plane indices k with |k dz - z| <= half_um (k dz the plane's depth, um)."""
    z, h, dz = float(z_um), float(half_um), float(dz_um)
    if not (math.isfinite(z) and h >= 0 and dz > 0):
        raise ValueError("plane_range needs a finite z, half_um >= 0 and dz_um > 0")
    return int(math.ceil((z - h) / dz - 1e-9)), int(math.floor((z + h) / dz + 1e-9))


def disk_mask(shape, left, top, res0_um, centre_xy_um, radius_um):
    """(H, W) bool: the pixels whose centre lies within radius_um of the centre (image frame)."""
    H, W = int(shape[0]), int(shape[1])
    p = float(res0_um)
    x = (int(left) + np.arange(W)) * p
    y = (int(top) + np.arange(H)) * p
    cx, cy = (float(v) for v in np.asarray(centre_xy_um, dtype=float).reshape(-1)[:2])
    return (x[None, :] - cx) ** 2 + (y[:, None] - cy) ** 2 <= float(radius_um) ** 2


def focus_measure(block, valid, left, top, res0_um, centre_xy_um, radius_um, sigma_px):
    """-> F (n_planes,): per plane the variance, over the disk of radius_um about the centre, of the Laplacian of
    Gaussian (sigma_px) of the plane (grey levels^2 per px^4); NaN for a plane not valid or an empty disk.
    block (n_planes, H, W) grey levels (any real dtype, not modified); valid (n_planes,) bool."""
    B = np.asarray(block)
    v = np.asarray(valid, dtype=bool).reshape(-1)
    if B.ndim != 3 or v.shape != (B.shape[0],):
        raise ValueError("block must be (n_planes, H, W) and valid (n_planes,)")
    if not (float(sigma_px) > 0 and float(radius_um) > 0):
        raise ValueError("sigma_px and radius_um must be > 0")
    mask = disk_mask(B.shape[1:], left, top, res0_um, centre_xy_um, radius_um)
    F = np.full(B.shape[0], np.nan)
    if not mask.any():
        return F
    for j in range(B.shape[0]):
        if not v[j]:
            continue
        L = gaussian_laplace(np.asarray(B[j], dtype=float), float(sigma_px))
        F[j] = float(np.var(L[mask]))
    return F


def sharpest(F):
    """-> the index of the largest finite F (the first on a tie); ValueError when none is finite."""
    f = np.asarray(F, dtype=float).reshape(-1)
    if not np.any(np.isfinite(f)):
        raise ValueError("no finite focus value")
    return int(np.nanargmax(f))
