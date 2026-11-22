"""The soma's view in the image stack -- Block 44 in specs/SPEC.md (D-210, D-211).

The traced soma (the SWC's type-1 nodes) placed in the image frame, the square crop about it, the recorded planes
about its depth, the plane to show and the planes of the montage, and a diagnostic per plane over a disk about it,
the fine-detail energy:

    E_k = Var over the disk { (G_sigma * L)(I_k) }       (L the Laplacian, G_sigma the Gaussian: their product the
                                                          Laplacian of Gaussian, scipy.ndimage.gaussian_laplace)

large where the plane holds sharp structure anywhere in the disk -- the soma's outline, or a dendrite crossing it --
and small where all of it is blurred. It is not a focus estimate of the soma: on the pilot its maximum was a basal
dendrite's plane, and the soma's plane is chosen by eye (D-211). Image frame: pixel (row, col) of a crop with corner
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


def detail_energy(block, valid, left, top, res0_um, centre_xy_um, radius_um, sigma_px):
    """-> E (n_planes,): per plane the variance, over the disk of radius_um about the centre, of the Laplacian of
    Gaussian (sigma_px) of the plane (grey levels^2 per px^4); NaN for a plane not valid or an empty disk. Every
    structure in the disk counts: a diagnostic, not a focus estimate of the soma (D-211.3). block (n_planes, H, W)
    grey levels (any real dtype, not modified); valid (n_planes,) bool."""
    B = np.asarray(block)
    v = np.asarray(valid, dtype=bool).reshape(-1)
    if B.ndim != 3 or v.shape != (B.shape[0],):
        raise ValueError("block must be (n_planes, H, W) and valid (n_planes,)")
    if not (float(sigma_px) > 0 and float(radius_um) > 0):
        raise ValueError("sigma_px and radius_um must be > 0")
    mask = disk_mask(B.shape[1:], left, top, res0_um, centre_xy_um, radius_um)
    E = np.full(B.shape[0], np.nan)
    if not mask.any():
        return E
    for j in range(B.shape[0]):
        if not v[j]:
            continue
        L = gaussian_laplace(np.asarray(B[j], dtype=float), float(sigma_px))
        E[j] = float(np.var(L[mask]))
    return E


def reference_plane(ks, valid, z_um, dz_um, k_chosen=None):
    """-> (index into ks, label). With k_chosen: that plane, label "chosen by eye" (ValueError when it is outside the
    window ks or not recorded). Without: the recorded plane whose depth k dz_um is nearest z_um (the first on a tie),
    label "traced depth" (ValueError when no plane is recorded). ks (n,) int, ascending; valid (n,) bool."""
    k = np.asarray(ks).astype(int).reshape(-1)
    v = np.asarray(valid, dtype=bool).reshape(-1)
    if k.shape != v.shape or k.size == 0:
        raise ValueError("ks and valid must be (n,) with n >= 1")
    if k_chosen is not None:
        hit = np.flatnonzero(k == int(k_chosen))
        if hit.size == 0:
            raise ValueError("plane %d is outside the window %d..%d" % (int(k_chosen), k.min(), k.max()))
        if not v[hit[0]]:
            raise ValueError("plane %d is not recorded" % int(k_chosen))
        return int(hit[0]), "chosen by eye"
    if not v.any():
        raise ValueError("no recorded plane in the window")
    d = np.abs(k * float(dz_um) - float(z_um))
    d[~v] = np.inf
    return int(np.argmin(d)), "traced depth"


def montage_planes(ks, valid, k_ref, step, k_lo=None, k_hi=None):
    """-> the indices into ks of the recorded planes k = k_ref + j step (j any integer) with k_lo <= k <= k_hi
    (defaults: the window's first and last plane), in the order of ks. ValueError for step < 1 or k_lo > k_hi."""
    k = np.asarray(ks).astype(int).reshape(-1)
    v = np.asarray(valid, dtype=bool).reshape(-1)
    s = int(step)
    if k.shape != v.shape or k.size == 0:
        raise ValueError("ks and valid must be (n,) with n >= 1")
    if s < 1 or s != step:
        raise ValueError("step must be an integer >= 1")
    lo = int(k.min()) if k_lo is None else int(k_lo)
    hi = int(k.max()) if k_hi is None else int(k_hi)
    if lo > hi:
        raise ValueError("k_lo %d > k_hi %d" % (lo, hi))
    r = int(k_ref)
    return [i for i in range(k.size) if v[i] and lo <= k[i] <= hi and (int(k[i]) - r) % s == 0]
