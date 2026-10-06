"""Ray world: an independent generator of synthetic planes -- Block 9 in
specs/SPEC.md (impl-handoff (S6), (S10); procedure s.3.10, last row).

Geometric optics with an evenly filled NA condenser (sine condition,
index-matched): for each plane k and lateral point x,

    I_k(x) / B = < exp(-mu L(x, z_k, s)) >_s ,                         (S6)

L the length inside the tube of the line through (x, z_k) with unit
direction s, the average over directions whose transverse part (s_x, s_y) is
uniform on the disc of radius s_m = NA / n_oil. Quadrature: Gauss-Legendre
in rho^2 on [0, s_m^2] (the uniform measure of the disc in rho^2) times
n_psi equally spaced azimuths. The ray world has no diffraction: a Gaussian
of ray_post_sigma_um is applied afterwards on the fine grid as a stand-in for
the in-focus blur (as checks/optics_points_check.py does), and its geometric
blur near focus is too wide, so it serves to test the TREATMENT OF ABSORPTION
of the partition renderer, never to give an absolute b (impl-handoff (S6)).

The light direction does not enter: absorption alone is reciprocal, so
exp(-mu L) is the same for both senses of travel along a line. For a tube
that is mirror-symmetric about z = c_z (phi = 0), the mirror maps the line
through (x, z_k) with direction (s_x, s_y, s_z) to the line through
(x, 2 c_z - z_k) with direction (-s_x, -s_y, s_z), i.e. the azimuth
psi -> psi + pi; the quadrature set is invariant under that map when n_psi
is even (psi_m = (m + 1/2) 2 pi / n_psi), so the planes at c_z -+ delta are
then equal to roundoff (for odd n_psi, to quadrature accuracy).

Chords come from Block 2's line_interval (squashed sections and both end cuts
included). Points that no ray of the cone can bring into the tube are skipped
(I = 1): their lateral distance to the rectangle that holds the tube's
footprint (half-extents from Block 2's lateral_half_extent along e_u, e_v
about c) exceeds max |z_k - z| over the tube times tan(theta_max),
theta_max = arcsin(s_m). The rectangle is analytic, so parts of the tube
outside the grid still count (an oblique ray from a block pixel can reach
them).

Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import math

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import ndimage

from . import geometry

_CHUNK_ENTRIES = 1 << 20      # points x directions per chunk (memory tile, ~8 MB per float64 array)


def directions(n_rho, n_psi, na, n_oil):
    """(S (N, 3) unit directions with s_z > 0, W (N,) weights summing to 1)."""
    s_m = float(na) / float(n_oil)
    if not (0 < s_m < 1) or int(n_rho) < 1 or int(n_psi) < 1:
        raise ValueError("directions: need 0 < NA / n_oil < 1 and n_rho, n_psi >= 1")
    x, wx = leggauss(int(n_rho))
    rho2 = 0.5 * (x + 1.0) * s_m ** 2
    psi = (np.arange(int(n_psi)) + 0.5) * 2.0 * math.pi / int(n_psi)
    R2, P = np.meshgrid(rho2, psi, indexing="ij")
    W = np.repeat((0.5 * wx)[:, None], int(n_psi), axis=1) / int(n_psi)
    rho = np.sqrt(R2)
    S = np.stack([(rho * np.cos(P)).ravel(), (rho * np.sin(P)).ravel(), np.sqrt(1.0 - R2).ravel()], axis=1)
    return S, W.ravel() / W.sum()


def mean_inverse_cos(S, W):
    """<1 / cos(theta)> = sum W / s_z; analytic (2 / s_m^2)(1 - sqrt(1 - s_m^2)) (S10)."""
    return float(np.sum(W / S[:, 2]))


def ray_transmittance(points, tube, mu, S, W):
    """<exp(-mu L)> over the directions, for points (n, 3) um; returns (n,)."""
    P = np.asarray(points, dtype=float).reshape(-1, 3)
    out = np.empty(P.shape[0])
    step = max(1, _CHUNK_ENTRIES // S.shape[0])
    for a in range(0, P.shape[0], step):
        p = P[a:a + step, None, :]
        t1, t2, hit = geometry.line_interval(p, S[None, :, :], tube)   # |S| = 1: t is arc length
        L = np.where(hit, t2 - t1, 0.0)
        out[a:a + step] = np.exp(-mu * L) @ W
    return out


def render_planes(tube, mu, z_planes, grid, rcfg, reduce=None):
    """I_k / B on a FineGrid for every depth in z_planes (S6), post-blurred by
    rcfg.ray_post_sigma_um; reduce as in render.render_transmittance.
    Returns (n_planes, ...) stacked planes."""
    reduce = (lambda plane: plane) if reduce is None else reduce
    S, W = directions(rcfg.ray_n_rho, rcfg.ray_n_psi, rcfg.ray_na, rcfg.ray_n_oil)
    tan_max = (rcfg.ray_na / rcfg.ray_n_oil) / math.sqrt(1.0 - (rcfg.ray_na / rcfg.ray_n_oil) ** 2)
    X, Y = grid.mesh()
    z_min, z_max = geometry.depth_extent(tube)
    if not (math.isfinite(z_min) and math.isfinite(z_max)):
        raise ValueError("the ray world needs a tube with a finite depth extent (an end cut)")
    # lateral distance from each grid point to the rectangle |u| <= E_u, |v| <= E_v that
    # holds the footprint (0 inside it); a lower bound of the distance to the footprint
    E_u, E_v = geometry.lateral_half_extent(tube)
    dx, dy = X - tube.c[0], Y - tube.c[1]
    u = dx * math.cos(tube.theta) + dy * math.sin(tube.theta)
    v = -dx * math.sin(tube.theta) + dy * math.cos(tube.theta)
    dist = np.hypot(np.clip(np.abs(u) - E_u, 0.0, None), np.clip(np.abs(v) - E_v, 0.0, None))
    out = []
    for zk in np.atleast_1d(np.asarray(z_planes, dtype=float)):
        reach = max(abs(zk - z_min), abs(zk - z_max)) * tan_max
        sel = dist <= reach
        I = np.ones(X.shape)
        if mu > 0 and sel.any():
            pts = np.column_stack([X[sel], Y[sel], np.full(int(sel.sum()), zk)])
            I[sel] = ray_transmittance(pts, tube, mu, S, W)
        if rcfg.ray_post_sigma_um > 0:
            I = 1.0 - ndimage.gaussian_filter(1.0 - I, rcfg.ray_post_sigma_um / grid.h, mode="constant", cval=0.0,
                                              truncate=rcfg.direct_truncate)
        out.append(reduce(I))
    return np.stack(out)
