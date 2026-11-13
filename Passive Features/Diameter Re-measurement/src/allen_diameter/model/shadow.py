"""The shadow image of a node's neighbours (tier 1) -- Block 25 in specs/SPEC.md
with its amendments 2 and 3 (D-129, D-130.O1, O4, O5; D-132.1; D-134, D-135).

The shadow model (amendment 3). A segment q with ends (p0, d0, mu0),
(p1, d1, mu1) is a straight capped cylinder on the line p0 p1 (Block 2 Tube,
round section, axial caps at the two ends) of diameter dbar = (d0 + d1) / 2
and coefficient mubar = (mu0 + mu1) / 2. Its shadow in plane k is equations
(5-6) of section 2.3 for the cylinder rendered alone,

    s_qk(x, y) = sum_j (DeltaA_qj * G_{sigma_r(zeta_j - z_k)})(x, y),

DeltaA_qj the light absorbed in slab j (centre zeta_j, thickness dzeta_um, the
lattice of render.slab_grid) under CleaningConfig.shadow_absorption:
"hybrid" -- Block 21's slab_absorbed with the directions
ray_world.directions(shadow_ray_n_rho, shadow_ray_n_psi, ray_na, ray_n_oil);
"partition_vertical" -- Block 4's T_<j (1 - exp(-a_j)). The column total
A_q = sum_j DeltaA_qj is what segment_absorbed returns, with z_axis. A soma
disc (x_s, y_s, r_s) is a sphere centred at depth soma_z_um whose column
absorbs 1 - exp(-soma_mu_per_um l_s), split over the slabs by the vertical
partition under both absorptions (PROVISIONAL, D-135.3). The shadow image of
node i is S_ik = sum of s_qk over the kept, included neighbours, averaged
over each pixel.

Discretisation. DeltaA_qj is sampled on a fine grid of shadow_fine_factor
samples per pixel side, on the lattice of the patch's own fine grid (Block 4
block_fine_grid), over the whole footprint of the segment. The custom code
(amendment 3): (5-6) evaluated from the footprint to the patch's pixels
without a common padded grid -- each slab's Gaussian convolution written as
two matrix products, rows then columns, of the continuous Gaussian sampled at
the fine offsets and averaged over each pixel's fine samples (Block 4's FFT
blur followed by pixel_integrate, evaluated only on the patch). A (slab,
plane) pair whose nearest absorbing column lies farther than
shadow_reach_sigmas x sigma from the patch rectangle contributes exactly 0.

Units: um, 1/um, dimensionless shadows; float64. Pure ASCII, for the cluster.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import geometry
from . import hybrid as _hybrid
from . import kernel as _kernel
from . import ray_world as _ray_world
from .render import FineGrid, slab_grid

SHADOW_ABSORPTIONS = ("hybrid", "partition_vertical")
_SQRT_2PI = math.sqrt(2.0 * math.pi)
_MIN_HORIZONTAL_UM = 1e-9          # a segment whose ends differ in xy by less than this is taken as vertical
_PHI_VERTICAL = 0.5 * math.pi - 1e-9   # Block 2's Tube needs phi < pi/2


@dataclass(frozen=True)
class ShadowResult:
    """S (n_planes, height, width) float64, dimensionless: the summed shadow of the kept and included
    neighbours, pixel means; per_neighbour_peak (n_q + n_s,): each neighbour's largest pixel value over
    the planes (segments in neighbour_indices' order, then the soma discs); kept (n_q + n_s,) bool: in S;
    overlap (height, width) bool: pixels where a kept neighbour's footprint (A_q > 0) meets the own
    stretch's footprint in projection."""

    S: np.ndarray
    per_neighbour_peak: np.ndarray
    kept: np.ndarray
    overlap: np.ndarray


# ----------------------------------------------------------- the geometry --

def _check_scene(fscene):
    xyz = np.asarray(fscene.seg_xyz, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 6:
        raise ValueError("fscene.seg_xyz must have shape (n, 6)")
    return xyz


def _segment_tube(fscene, q):
    """(Tube, mubar) of segment q: the capped round cylinder on p0 p1 of diameter (d0 + d1)/2, axial caps
    at the two ends, and mubar = (mu0 + mu1)/2; (None, mubar) for a segment of zero length."""
    xyz = _check_scene(fscene)
    q = int(q)
    p0, p1 = xyz[q, :3], xyz[q, 3:]
    d = np.asarray(fscene.seg_d_um, dtype=float)[q]
    mu = np.asarray(fscene.seg_mu_per_um, dtype=float)[q]
    if not (np.all(np.isfinite(xyz[q])) and np.all(np.isfinite(d)) and np.all(np.isfinite(mu))):
        raise ValueError("segment %d: non-finite position, diameter or coefficient" % q)
    dbar, mubar = 0.5 * float(d[0] + d[1]), 0.5 * float(mu[0] + mu[1])
    if not (dbar > 0) or not (mubar >= 0):
        raise ValueError("segment %d: need mean diameter > 0 and mean coefficient >= 0" % q)
    t = p1 - p0
    length = float(np.sqrt(np.sum(t * t)))
    if length == 0.0:
        return None, mubar
    if t[2] < 0:
        t = -t
    horiz = math.hypot(float(t[0]), float(t[1]))
    if horiz < _MIN_HORIZONTAL_UM:
        theta, phi = 0.0, _PHI_VERTICAL
    else:
        theta = math.atan2(float(t[1]), float(t[0]))
        phi = min(math.atan2(float(t[2]), horiz), _PHI_VERTICAL)
    c = 0.5 * (p0 + p1)
    tube = geometry.Tube(c=(float(c[0]), float(c[1]), float(c[2])), r=0.5 * dbar, phi=phi, theta=theta,
                         aspect=1.0, half_length=0.5 * length, end_cut="axial")
    return tube, mubar


def _directions(cfg):
    cl, rc = cfg.cleaning, cfg.renderer
    return _ray_world.directions(cl.shadow_ray_n_rho, cl.shadow_ray_n_psi, rc.ray_na, rc.ray_n_oil)


def _absorption(cfg):
    a = cfg.cleaning.shadow_absorption
    if a not in SHADOW_ABSORPTIONS:
        raise ValueError("cleaning.shadow_absorption must be one of %r, got %r" % (SHADOW_ABSORPTIONS, a))
    return a


def _z_axis(tube, X, Y):
    u = (X - tube.c[0]) * math.cos(tube.theta) + (Y - tube.c[1]) * math.sin(tube.theta)
    u_end = tube.half_length * math.cos(tube.phi0)
    return tube.c[2] + np.clip(u, -u_end, u_end) * math.tan(tube.phi)


def _tube_slabs(tube, mubar, grid, cfg, directions=None):
    """Yield (zeta_j, row slice, column slice, DeltaA_j on that box) for every slab of the tube with
    absorbed light on grid, in increasing j (the slab lattice of render.slab_grid)."""
    if mubar == 0.0:
        return
    X, Y = grid.mesh()
    z_lo, z_hi, inside = geometry.column_interval(X, Y, tube)
    if not inside.any():
        return
    dz = float(cfg.renderer.dzeta_um)
    light = int(cfg.acquisition.light_direction)
    hyb = _absorption(cfg) == "hybrid"
    if hyb:
        S, W = _directions(cfg) if directions is None else directions
    xs, ys = grid.xs, grid.ys
    for zj in slab_grid(tube, grid, dz):
        hit = inside & (z_lo < zj + 0.5 * dz) & (z_hi > zj - 0.5 * dz)
        if not hit.any():
            continue
        rows = np.flatnonzero(hit.any(axis=1))
        cols = np.flatnonzero(hit.any(axis=0))
        rs, cs = slice(int(rows[0]), int(rows[-1]) + 1), slice(int(cols[0]), int(cols[-1]) + 1)
        zl, zh = z_lo[rs, cs], z_hi[rs, cs]
        if hyb:
            dA = _hybrid.slab_absorbed(xs[cs][None, :], ys[rs][:, None], zl, zh, zj, dz, mubar, tube, S, W, light)
        else:
            a = geometry.slab_absorbance(zl, zh, zj, dz, mubar)
            dA = geometry.transmitted_before(zl, zh, zj, dz, mubar, light) * (-np.expm1(-a))
        yield float(zj), rs, cs, dA


def _soma_slabs(xs, ys, r, grid, cfg):
    """Yield the slabs of the soma sphere (vertical partition of its Beer-Lambert column)."""
    mu = float(cfg.cleaning.soma_mu_per_um)
    if mu == 0.0:
        return
    zc = float(cfg.cleaning.soma_z_um)
    X, Y = grid.mesh()
    half = np.sqrt(np.clip(r * r - (X - xs) ** 2 - (Y - ys) ** 2, 0.0, None))
    inside = half > 0
    if not inside.any():
        return
    z_lo, z_hi = np.where(inside, zc - half, zc), np.where(inside, zc + half, zc)
    dz = float(cfg.renderer.dzeta_um)
    light = int(cfg.acquisition.light_direction)
    for zj in geometry.slab_centres(float(z_lo[inside].min()), float(z_hi[inside].max()), dz):
        hit = inside & (z_lo < zj + 0.5 * dz) & (z_hi > zj - 0.5 * dz)
        if not hit.any():
            continue
        rows = np.flatnonzero(hit.any(axis=1))
        cols = np.flatnonzero(hit.any(axis=0))
        rs, cs = slice(int(rows[0]), int(rows[-1]) + 1), slice(int(cols[0]), int(cols[-1]) + 1)
        zl, zh = z_lo[rs, cs], z_hi[rs, cs]
        a = geometry.slab_absorbance(zl, zh, zj, dz, mu)
        yield float(zj), rs, cs, geometry.transmitted_before(zl, zh, zj, dz, mu, light) * (-np.expm1(-a))


def _column_total(slabs, shape):
    A = np.zeros(shape)
    for _zj, rs, cs, dA in slabs:
        A[rs, cs] += dA
    return A


def segment_absorbed(fscene, q, grid, cfg):
    """(A (ny, nx), z_axis (ny, nx)) of segment q on the render.FineGrid grid, under
    cfg.cleaning.shadow_absorption: A the column total of the absorbed light of the segment's capped
    cylinder rendered alone (dimensionless), z_axis (um) the depth of the cylinder's axis under each
    column (the axis line c_z + u tan(phi), u clipped to the axis' own extent)."""
    tube, mubar = _segment_tube(fscene, q)
    if tube is None:
        return np.zeros((grid.ny, grid.nx)), np.full((grid.ny, grid.nx), float(fscene.seg_xyz[int(q)][2]))
    X, Y = grid.mesh()
    return _column_total(_tube_slabs(tube, mubar, grid, cfg), X.shape), _z_axis(tube, X, Y)


def neighbour_indices(fscene, own_ids):
    """(segments (n_q,), somas (n_s,)) int64: every segment with at least one end outside own_ids
    (D-130.O4), in scene order; all soma discs."""
    ids = np.asarray(fscene.seg_ids).astype(np.int64).reshape(-1, 2)
    own = np.asarray(sorted(int(i) for i in own_ids), dtype=np.int64)
    out_any = ~(np.isin(ids[:, 0], own) & np.isin(ids[:, 1], own))
    soma = np.asarray(fscene.soma, dtype=float).reshape(-1, 3)
    return np.flatnonzero(out_any).astype(np.int64), np.arange(soma.shape[0], dtype=np.int64)


def _own_segments(fscene, own_ids):
    ids = np.asarray(fscene.seg_ids).astype(np.int64).reshape(-1, 2)
    own = np.asarray(sorted(int(i) for i in own_ids), dtype=np.int64)
    return np.flatnonzero(np.isin(ids[:, 0], own) & np.isin(ids[:, 1], own))


# --------------------------------------------------------- the patch blur --

class _Patch:
    """The pixel block (left, top, width, height), its fine lattice and the blur onto its pixel means."""

    def __init__(self, left, top, width, height, cfg):
        for name, val in (("left", left), ("top", top), ("width", width), ("height", height)):
            if int(val) != val:
                raise ValueError("%s must be an integer" % name)
        if int(width) < 1 or int(height) < 1:
            raise ValueError("width and height must be >= 1")
        self.left, self.top, self.W, self.H = int(left), int(top), int(width), int(height)
        self.p = float(cfg.acquisition.res0_um)
        self.f = int(cfg.cleaning.shadow_fine_factor)
        if self.f < 1 or self.f != cfg.cleaning.shadow_fine_factor:
            raise ValueError("cleaning.shadow_fine_factor must be an integer >= 1")
        self.h = self.p / self.f
        self.x0 = self.left * self.p - 0.5 * self.p + 0.5 * self.h     # first fine sample (Block 4 block_fine_grid)
        self.y0 = self.top * self.p - 0.5 * self.p + 0.5 * self.h
        sub = np.arange(self.f) * self.h
        self.tx = (self.x0 + np.arange(self.W)[:, None] * self.p + sub[None, :])   # (W, f) fine targets per pixel
        self.ty = (self.y0 + np.arange(self.H)[:, None] * self.p + sub[None, :])   # (H, f)
        self.rect = ((self.left - 0.5) * self.p, (self.left + self.W - 0.5) * self.p,
                     (self.top - 0.5) * self.p, (self.top + self.H - 0.5) * self.p)
        self.reach = float(cfg.cleaning.shadow_reach_sigmas)

    def source_grid(self, xa, xb, ya, yb):
        """FineGrid on the patch's lattice covering [xa, xb] x [ya, yb] (um)."""
        h = self.h
        n0 = int(math.floor((xa - self.x0) / h)) - 1
        n1 = int(math.ceil((xb - self.x0) / h)) + 1
        m0 = int(math.floor((ya - self.y0) / h)) - 1
        m1 = int(math.ceil((yb - self.y0) / h)) + 1
        return FineGrid(x0=self.x0 + n0 * h, y0=self.y0 + m0 * h, h=h, nx=n1 - n0 + 1, ny=m1 - m0 + 1), (n0, m0)

    def _weights(self, targets, src, sigma):
        """(n_target_pixels, n_src): the pixel mean over the fine targets of h g_sigma(target - src)."""
        dlt = (targets[:, :, None] - src[None, None, :]) / sigma
        g = np.exp(-0.5 * dlt * dlt) * (self.h / (_SQRT_2PI * sigma))
        return g.mean(axis=1)

    def dist_to_rect(self, X, Y):
        x_lo, x_hi, y_lo, y_hi = self.rect
        dx = np.maximum(np.maximum(x_lo - X, X - x_hi), 0.0)
        dy = np.maximum(np.maximum(y_lo - Y, Y - y_hi), 0.0)
        return np.sqrt(dx * dx + dy * dy)

    def blur_slabs(self, slabs, grid, planes_z, rcfg):
        """(n_planes, H, W): equations (5-6) of the slabs (zeta_j, rows, cols, DeltaA_j on grid), each
        slab blurred at sigma_r(zeta_j - z_k), onto the patch's pixel means; and the union footprint."""
        out = np.zeros((len(planes_z), self.H, self.W))
        foot = np.zeros((grid.ny, grid.nx), dtype=bool)
        xs, ys = grid.xs, grid.ys
        z = np.asarray(planes_z, dtype=float)
        for zj, rs, cs, dA in slabs:
            nz = dA != 0.0
            if not nz.any():
                continue
            foot[rs, cs] |= nz
            Xb, Yb = np.meshgrid(xs[cs], ys[rs])
            dmin = float(self.dist_to_rect(Xb[nz], Yb[nz]).min())
            sig = np.atleast_1d(_kernel.sigma_r(zj - z, rcfg))
            for k, s in enumerate(sig):
                s = float(s)
                if dmin > self.reach * s:
                    continue
                Gy = self._weights(self.ty, ys[rs], s)          # (H, ny_box)
                Gx = self._weights(self.tx, xs[cs], s)          # (W, nx_box)
                out[k] += Gy @ dA @ Gx.T
        return out, foot


def _segment_box(tube):
    """Lateral bounding box (xa, xb, ya, yb) um of a capped tube's footprint (a superset)."""
    t = tube.t_hat0
    e0 = np.asarray(tube.c) - tube.half_length * t
    e1 = np.asarray(tube.c) + tube.half_length * t
    r = tube.r
    return (min(e0[0], e1[0]) - r, max(e0[0], e1[0]) + r, min(e0[1], e1[1]) - r, max(e0[1], e1[1]) + r)


def _neighbour_image(fscene, kind, idx, patch, planes_z, cfg, directions):
    """(image (n_planes, H, W), footprint on its grid (bool, absorbed light > 0), grid offset (n0, m0))."""
    if kind == "seg":
        tube, mubar = _segment_tube(fscene, idx)
        if tube is None:
            return np.zeros((len(planes_z), patch.H, patch.W)), None, None
        grid, off = patch.source_grid(*_segment_box(tube))
        slabs = _tube_slabs(tube, mubar, grid, cfg, directions)
    else:
        xs, ys, r = (float(v) for v in np.asarray(fscene.soma, dtype=float).reshape(-1, 3)[idx])
        if not (r > 0 and math.isfinite(xs) and math.isfinite(ys)):
            raise ValueError("soma disc %d: need finite centre and radius > 0" % idx)
        grid, off = patch.source_grid(xs - r, xs + r, ys - r, ys + r)
        slabs = _soma_slabs(xs, ys, r, grid, cfg)
    img, foot = patch.blur_slabs(slabs, grid, planes_z, cfg.renderer)
    return img, foot, off


def _paste(mask, off, patch):
    """A footprint (on a lattice grid at offset off) as a bool mask on the patch's fine grid."""
    out = np.zeros((patch.H * patch.f, patch.W * patch.f), dtype=bool)
    if mask is None:
        return out
    n0, m0 = off
    ny, nx = mask.shape
    r0, r1 = max(0, m0), min(out.shape[0], m0 + ny)
    c0, c1 = max(0, n0), min(out.shape[1], n0 + nx)
    if r1 > r0 and c1 > c0:
        out[r0:r1, c0:c1] = mask[r0 - m0:r1 - m0, c0 - n0:c1 - n0]
    return out


def _check_planes(planes_z):
    z = np.atleast_1d(np.asarray(planes_z, dtype=float)).reshape(-1)
    if z.size == 0 or not np.all(np.isfinite(z)):
        raise ValueError("planes_z must be a non-empty array of finite depths (um)")
    return z


def shadow_image(fscene, own_ids, left, top, width, height, planes_z, cfg, B_gl=None, include=None):
    """ShadowResult of node i's neighbours on the pixel block (left, top, width, height) (full-resolution
    pixel coordinates, integer = pixel centre) at the plane depths planes_z (um).

    own_ids: the node ids of the own stretch (its segments: both ends in own_ids). B_gl: background (gl)
    for the floor of D-130.O5 (a neighbour with B_gl x peak < shadow_floor_gl is not kept), None = keep
    every neighbour. include: bool (n_q + n_s,) in neighbour_indices' order, False = left out (tier 2,
    D-132.2); a left-out neighbour's peak is still computed, and it is not kept.
    ValueError: an own stretch with no segment in the scene, non-finite or empty planes, include of the
    wrong length, B_gl not finite and > 0."""
    _check_scene(fscene)
    _absorption(cfg)
    z = _check_planes(planes_z)
    own_segs = _own_segments(fscene, own_ids)
    if own_segs.size == 0:
        raise ValueError("shadow_image: the own stretch has no segment in the scene")
    segs, somas = neighbour_indices(fscene, own_ids)
    n_nb = segs.size + somas.size
    if include is None:
        inc = np.ones(n_nb, dtype=bool)
    else:
        inc = np.asarray(include, dtype=bool).reshape(-1)
        if inc.size != n_nb:
            raise ValueError("include must have one entry per neighbour (%d), got %d" % (n_nb, inc.size))
    if B_gl is not None:
        B = float(B_gl)
        if not (math.isfinite(B) and B > 0):
            raise ValueError("B_gl must be finite and > 0")
    patch = _Patch(left, top, width, height, cfg)
    directions = _directions(cfg) if cfg.cleaning.shadow_absorption == "hybrid" else None

    own_fp = np.zeros((patch.H * patch.f, patch.W * patch.f), dtype=bool)
    for q in own_segs:
        tube, _ = _segment_tube(fscene, q)
        if tube is None:
            continue
        g, off = patch.source_grid(*_segment_box(tube))
        X, Y = g.mesh()
        own_fp |= _paste(geometry.column_interval(X, Y, tube)[2], off, patch)

    S = np.zeros((z.size, patch.H, patch.W))
    peak = np.zeros(n_nb)
    kept = np.zeros(n_nb, dtype=bool)
    nb_fp = np.zeros_like(own_fp)
    items = [("seg", int(q)) for q in segs] + [("soma", int(s)) for s in somas]
    for i, (kind, idx) in enumerate(items):
        img, fp, off = _neighbour_image(fscene, kind, idx, patch, z, cfg, directions)
        peak[i] = float(img.max()) if img.size else 0.0
        kept[i] = bool(inc[i]) and (B_gl is None or float(B_gl) * peak[i] >= float(cfg.cleaning.shadow_floor_gl))
        if kept[i]:
            S += img
            nb_fp |= _paste(fp, off, patch)
    both = (own_fp & nb_fp).reshape(patch.H, patch.f, patch.W, patch.f)
    overlap = both.any(axis=(1, 3))
    return ShadowResult(S=S, per_neighbour_peak=peak, kept=kept, overlap=overlap)


def stretch_profiles(fscene, stretch_ids, ks, origin_xy, y_hat, e_u, v, n_avg, step, cfg):
    """(n_planes, n_v) float64, dimensionless: the shadow model's image of the segments whose two ends
    are in stretch_ids, at the planes ks (plane indices, depth k dz_um), read along the measuring line as
    profiles.sample_profile reads a plane of pixel means (bilinear between pixel centres, mean over the
    2 n_avg + 1 lines step um apart along e_u). ValueError: no segment with both ends in stretch_ids."""
    from allen_image_io import CropFrame     # the contract's frame type (flat module in src/)
    from ..analysis.profiles import sample_profile

    _check_scene(fscene)
    _absorption(cfg)
    segs = _own_segments(fscene, stretch_ids)
    if segs.size == 0:
        raise ValueError("stretch_profiles: no segment has both ends in stretch_ids")
    ks = np.atleast_1d(np.asarray(ks)).reshape(-1)
    if ks.size == 0:
        raise ValueError("ks must not be empty")
    z = _check_planes(ks.astype(float) * float(cfg.acquisition.dz_um))
    v = np.asarray(v, dtype=float).reshape(-1)
    yh = np.asarray(y_hat, dtype=float).reshape(2)
    eu = np.asarray(e_u, dtype=float).reshape(2)
    o = np.asarray(origin_xy, dtype=float).reshape(2)
    if not (np.all(np.isfinite(v)) and np.all(np.isfinite(yh)) and np.all(np.isfinite(eu))
            and np.all(np.isfinite(o)) and math.isfinite(float(step))):
        raise ValueError("stretch_profiles: non-finite line geometry")
    p = float(cfg.acquisition.res0_um)
    a = np.arange(-int(n_avg), int(n_avg) + 1) * float(step)
    xs = o[0] + v[None, :] * yh[0] + a[:, None] * eu[0]
    ys = o[1] + v[None, :] * yh[1] + a[:, None] * eu[1]
    left = int(math.floor(float(xs.min()) / p)) - 1
    top = int(math.floor(float(ys.min()) / p)) - 1
    width = int(math.floor(float(xs.max()) / p)) + 2 - left + 1
    height = int(math.floor(float(ys.max()) / p)) + 2 - top + 1
    patch = _Patch(left, top, width, height, cfg)
    directions = _directions(cfg) if cfg.cleaning.shadow_absorption == "hybrid" else None
    planes = np.zeros((z.size, height, width))
    for q in segs:
        img, _, _ = _neighbour_image(fscene, "seg", int(q), patch, z, cfg, directions)
        planes += img
    frame = CropFrame(left, top, 0, p)
    return np.stack([sample_profile(planes[k], frame, o, yh, eu, v, int(n_avg), float(step))
                     for k in range(z.size)])
