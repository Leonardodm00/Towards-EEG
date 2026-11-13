"""The forward model on the measuring line only -- Block 26 in specs/SPEC.md (wave 1; D-129.1, D-130.O10).

Section 2.3 of SPEC.md for one tube (Block 2 Tube), evaluated only where the
measuring line of Block 5 reads it. For plane k at depth z_k = k dz_um (stage
units) the plane is

    tau_k(x, y) = 1 - sum_j (Delta A_j * G_{sigma_r(zeta_j - z_k)})(x, y)      (Eqs. 5-6 / B)

with Delta A_j by name (Block 4, Block 21):

    linear              Delta A_j = a_j
    partition_vertical  Delta A_j = T_<j (1 - exp(-a_j))
    hybrid              Delta A_j = a_j h(x, y, zbar_j)        (section 2.3 (H))

The value returned for plane k and offset v_n is what profiles.sample_profile
(Block 5) reads from the plane of pixel means of tau_k, before the camera
chain: bilinear between pixel centres, the mean over the 2 n_avg + 1 parallel
lines. Only the pixels that read touches are computed.

Discretisation (the same as Block 4's where Block 4 has one): the object is
sampled on the fine lattice of line_fine_factor samples per pixel side,
aligned with the pixels (sample centres x = (n + 1/2) h - p/2, h = p / f, the
lattice of render.block_fine_grid); the slab lattice tiles the depth range of
the tube's columns on that lattice (Block 2 slab_centres, thickness dzeta_um;
the tube's whole footprint, so the lattice depends on the tube only). The
pixel mean of each sample's blurred mass h^2 Delta A_j is integrated exactly:
the circular Gaussian is separable, and its mean over a pixel
[x_c - p/2, x_c + p/2] x [y_c - p/2, y_c + p/2] is the product of
(Phi((x_c + p/2 - x_s) / sigma) - Phi((x_c - p/2 - x_s) / sigma)) / p and the
same in y (scipy.special.ndtr). Block 4 instead point-samples the blurred
plane on the same lattice and averages; the two agree to the midpoint rule's
error (the smoke test compares them).

Reach: for slab j only samples within line_reach_sigmas x max_k sigma_r(zeta_j
- z_k) (Euclidean, um) of the needed pixels' area are drawn, and a (slab,
plane) pair whose drawn samples all lie farther than line_reach_sigmas x
sigma_r(zeta_j - z_k) contributes 0 (a Gaussian's mass beyond 6 SD in one
direction is 1e-9).

The hybrid's slabs are Block 21's slab_absorbed on the drawn samples, with the
condenser of Block 9 (ray_world.directions of the RendererConfig's ray_*
fields), each slab computed once and served to every plane.

Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import math
from collections import namedtuple

import numpy as np
from scipy import special

from ..analysis.profiles import sample_profile        # Block 5's reading rule
from . import geometry
from . import hybrid as _hybrid
from . import ray_world
from .kernel import sigma_r

ABSORPTIONS_LINE = ("hybrid", "partition_vertical", "linear")
_UNIT_TOL = 1e-9          # |norm - 1| allowed for y_hat and e_u

_Frame = namedtuple("_Frame", ["left", "top", "res_um_px"])   # what profiles.sample_profile reads of a frame


# ------------------------------------------------------------ validation ---

def _unit2(a, name):
    a = np.asarray(a, dtype=float).reshape(-1)
    if a.shape != (2,) or not np.all(np.isfinite(a)):
        raise ValueError("%s must be a finite 2-vector" % name)
    if abs(math.hypot(a[0], a[1]) - 1.0) > _UNIT_TOL:
        raise ValueError("%s must be a unit vector (norm %r)" % (name, math.hypot(a[0], a[1])))
    return a


def _line_args(origin_xy, y_hat, e_u, v, n_avg, step):
    o = np.asarray(origin_xy, dtype=float).reshape(-1)
    if o.shape != (2,) or not np.all(np.isfinite(o)):
        raise ValueError("origin_xy must be a finite 2-vector (um)")
    yh, eu = _unit2(y_hat, "y_hat"), _unit2(e_u, "e_u")
    v = np.asarray(v, dtype=float)
    if v.ndim != 1 or v.size < 1 or not np.all(np.isfinite(v)):
        raise ValueError("v must be a non-empty 1-D array of finite offsets (um)")
    if int(n_avg) != n_avg or n_avg < 0:
        raise ValueError("n_avg must be an integer >= 0")
    step = float(step)
    if not math.isfinite(step):
        raise ValueError("step must be finite (um)")
    return o, yh, eu, v, int(n_avg), step


def _sample_points(o, yh, eu, v, n_avg, step):
    """(x, y) um of every sample read, (2 n_avg + 1, n_v) each, in sample_profile's arithmetic."""
    a = np.arange(-n_avg, n_avg + 1, dtype=float)[:, None]
    x = o[0] + v[None, :] * yh[0] + a * step * eu[0]
    y = o[1] + v[None, :] * yh[1] + a * step * eu[1]
    return x, y


# ------------------------------------------------------------ the pixels ---

def pixels_needed(origin_xy, y_hat, e_u, v, n_avg, step, res0_um):
    """(cols, rows) int64 arrays of equal length: the full-resolution pixels
    (integer coordinates = pixel centres, at x = col res0_um, y = row res0_um)
    that profiles.sample_profile's bilinear read touches -- for each sample at
    (x, y) um, the pixels floor(x / p) and floor(x / p) + 1 by floor(y / p) and
    floor(y / p) + 1 (p = res0_um), every one of the 2 n_avg + 1 lines. Unique
    pairs, sorted by row then column."""
    o, yh, eu, v, n_avg, step = _line_args(origin_xy, y_hat, e_u, v, n_avg, step)
    p = float(res0_um)
    if not (math.isfinite(p) and p > 0):
        raise ValueError("res0_um must be finite and > 0")
    x, y = _sample_points(o, yh, eu, v, n_avg, step)
    c0 = np.floor(x / p).astype(np.int64).ravel()
    r0 = np.floor(y / p).astype(np.int64).ravel()
    cols = np.concatenate([c0, c0 + 1, c0, c0 + 1])
    rows = np.concatenate([r0, r0, r0 + 1, r0 + 1])
    pairs = np.unique(np.stack([rows, cols], axis=1), axis=0)
    return pairs[:, 1].copy(), pairs[:, 0].copy()


def _pixel_kernel(edges, xs, sigma, h, p):
    """(n_px, n_s): h times the mean over pixel i = [edges[i], edges[i+1]] (width p)
    of a 1-D Gaussian of SD sigma centred at each sample xs,
    (h / p) (Phi((edges[i+1] - xs) / sigma) - Phi((edges[i] - xs) / sigma)).
    Where both edges lie in the upper tail the difference is taken there,
    Phi(-lo) - Phi(-hi), so that it keeps its relative accuracy."""
    t = (edges[:, None] - xs[None, :]) / sigma            # (n_px + 1, n_s)
    lo, hi = t[:-1], t[1:]
    diff = np.where(lo > 0, special.ndtr(-lo) - special.ndtr(-hi), special.ndtr(hi) - special.ndtr(lo))
    return (h / p) * diff


# ------------------------------------------------------------- the model ---

def line_profiles(tube, mu, ks, origin_xy, y_hat, e_u, v, n_avg, step, cfg, absorption=None):
    """tau (n_planes, n_v) float64, dimensionless: for each plane index of ks
    (depth k dz_um; rows in the order of ks) what profiles.sample_profile reads,
    along the line (origin_xy um, unit y_hat, unit e_u, offsets v (n_v,) um,
    n_avg lines on each side step um apart along e_u), from the plane of pixel
    means of tau_k of Eqs. 5-6, before the camera chain.

    tube: Block 2 Tube with a half-length (a capped tube); mu (1/um) >= 0;
    cfg: DiameterConfig (renderer kernel, dzeta_um and ray_* fields;
    acquisition res0_um, dz_um, light_direction; inference line_fine_factor,
    line_reach_sigmas); absorption: "hybrid", "partition_vertical" or "linear"
    (None: cfg.inference.absorption). ValueError for mu < 0 or not finite, a
    non-unit axis, an unknown absorption, an uncapped tube."""
    acq, rcfg, icfg = cfg.acquisition, cfg.renderer, cfg.inference
    absorption = icfg.absorption if absorption is None else absorption
    if absorption not in ABSORPTIONS_LINE:
        raise ValueError("unknown absorption %r for the line renderer (one of %r)" % (absorption, ABSORPTIONS_LINE))
    mu = float(mu)
    if not (math.isfinite(mu) and mu >= 0):
        raise ValueError("mu must be finite and >= 0 (1/um)")
    if tube.half_length is None:
        raise ValueError("the line renderer needs a capped tube (half_length set): its footprint must be finite")
    o, yh, eu, v, n_avg, step = _line_args(origin_xy, y_hat, e_u, v, n_avg, step)
    ks_arr = np.asarray(ks).reshape(-1)
    if ks_arr.size and (not np.all(np.isfinite(ks_arr.astype(float))) or np.any(ks_arr.astype(float) % 1 != 0)):
        raise ValueError("ks must be integer plane indices")
    ks_arr = ks_arr.astype(np.int64)
    light = acq.light_direction
    if light not in (+1, -1):
        raise ValueError("light_direction must be +1 or -1")
    f = int(icfg.line_fine_factor)
    if f != icfg.line_fine_factor or f < 1:
        raise ValueError("inference.line_fine_factor must be an integer >= 1")
    reach = float(icfg.line_reach_sigmas)
    if not (reach > 0):
        raise ValueError("inference.line_reach_sigmas must be > 0")
    p, dz, dzeta = float(acq.res0_um), float(acq.dz_um), float(rcfg.dzeta_um)
    h = p / f
    n_planes, n_v = ks_arr.size, v.size
    if n_planes == 0:
        return np.ones((0, n_v))
    if mu == 0.0:
        return np.ones((n_planes, n_v))

    # the pixel box the read touches (one pixel of margin against floor's roundoff)
    cols, rows = pixels_needed(o, yh, eu, v, n_avg, step, p)
    c_lo, c_hi = int(cols.min()) - 1, int(cols.max()) + 1
    r_lo, r_hi = int(rows.min()) - 1, int(rows.max()) + 1
    ex = (np.arange(c_lo, c_hi + 2) - 0.5) * p           # pixel edges, um
    ey = (np.arange(r_lo, r_hi + 2) - 0.5) * p
    bx0, bx1, by0, by1 = ex[0], ex[-1], ey[0], ey[-1]     # the box's area, um

    # the tube's footprint on the fine lattice x = (n + 1/2) h - p/2
    eu3, ev3 = tube.e_u, tube.e_v
    Eu, Ev = geometry.lateral_half_extent(tube)
    cx, cy = float(tube.c[0]), float(tube.c[1])
    hx = abs(Eu * eu3[0]) + abs(Ev * ev3[0])
    hy = abs(Eu * eu3[1]) + abs(Ev * ev3[1])
    ix = np.arange(int(math.floor((cx - hx + 0.5 * p) / h - 0.5)) - 1,
                   int(math.ceil((cx + hx + 0.5 * p) / h - 0.5)) + 2)
    iy = np.arange(int(math.floor((cy - hy + 0.5 * p) / h - 0.5)) - 1,
                   int(math.ceil((cy + hy + 0.5 * p) / h - 0.5)) + 2)
    xs_all = (ix + 0.5) * h - 0.5 * p
    ys_all = (iy + 0.5) * h - 0.5 * p
    X, Y = np.meshgrid(xs_all, ys_all, indexing="xy")
    z_lo, z_hi, inside = geometry.column_interval(X, Y, tube)
    if not inside.any():
        return np.ones((n_planes, n_v))
    zeta = geometry.slab_centres(float(z_lo[inside].min()), float(z_hi[inside].max()), dzeta)
    z_planes = ks_arr * dz
    sig = sigma_r(zeta[:, None] - z_planes[None, :], rcfg)        # (J, n_planes), um
    R_jk = reach * sig
    R_j = R_jk.max(axis=1)

    # distance of every column to the pixel box's area; keep what any slab can reach
    dxb = np.maximum(np.maximum(bx0 - xs_all, xs_all - bx1), 0.0)
    dyb = np.maximum(np.maximum(by0 - ys_all, ys_all - by1), 0.0)
    dist = np.hypot(dxb[None, :], dyb[:, None])
    keep = inside & (dist <= float(R_j.max()))
    if not keep.any():
        return np.ones((n_planes, n_v))
    kr = np.flatnonzero(keep.any(axis=1))
    kc = np.flatnonzero(keep.any(axis=0))
    sl = (slice(int(kr[0]), int(kr[-1]) + 1), slice(int(kc[0]), int(kc[-1]) + 1))
    xs, ys = xs_all[sl[1]], ys_all[sl[0]]
    X, Y, z_lo, z_hi, inside, dist = X[sl], Y[sl], z_lo[sl], z_hi[sl], inside[sl], dist[sl]

    S = W = None
    if absorption == "hybrid":
        S, W = ray_world.directions(rcfg.ray_n_rho, rcfg.ray_n_psi, rcfg.ray_na, rcfg.ray_n_oil)

    deficit = np.zeros((n_planes, r_hi - r_lo + 1, c_hi - c_lo + 1))
    for j in range(zeta.size):
        zj = float(zeta[j])
        hit = inside & (z_lo < zj + 0.5 * dzeta) & (z_hi > zj - 0.5 * dzeta) & (dist <= R_j[j])
        if not hit.any():
            continue
        hr = np.flatnonzero(hit.any(axis=1))
        hc = np.flatnonzero(hit.any(axis=0))
        rs, cs = slice(int(hr[0]), int(hr[-1]) + 1), slice(int(hc[0]), int(hc[-1]) + 1)
        m = hit[rs, cs]
        zl, zh = z_lo[rs, cs], z_hi[rs, cs]
        if absorption == "hybrid":
            dA = np.zeros(m.shape)
            dA[m] = _hybrid.slab_absorbed(X[rs, cs][m], Y[rs, cs][m], zl[m], zh[m], zj, dzeta, mu, tube, S, W,
                                          light)
        else:
            a = geometry.slab_absorbance(zl, zh, zj, dzeta, mu)
            if absorption == "partition_vertical":
                dA = geometry.transmitted_before(zl, zh, zj, dzeta, mu, light) * (-np.expm1(-a))
            else:
                dA = a
            dA = np.where(m, dA, 0.0)
        dmin = float(dist[rs, cs][m].min())
        for k in range(n_planes):
            if dmin > R_jk[j, k]:
                continue
            s = float(sig[j, k])
            Kx = _pixel_kernel(ex, xs[cs], s, h, p)        # (n_cols, nx_box)
            Ky = _pixel_kernel(ey, ys[rs], s, h, p)        # (n_rows, ny_box)
            deficit[k] += Ky @ (dA @ Kx.T)

    frame = _Frame(c_lo, r_lo, p)
    out = np.empty((n_planes, n_v))
    for k in range(n_planes):
        out[k] = 1.0 - sample_profile(deficit[k], frame, o, yh, eu, v, n_avg, step)
    return out
