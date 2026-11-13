"""The fast line renderer: one cross-section carried along the tube -- Block 35 in specs/SPEC.md (wave 3; D-137, D-138).

The quantity is Block 26's (model/line_render.py): for a capped straight tube
(Block 2 Tube), mu, plane indices ks and Block 5's measuring line, the value
tau^px_k(v_n) that profiles.sample_profile reads from the pixel means of the
plane of Eqs. 5-6 (SPEC section 2.3),

    tau_k(x, y) = 1 - sum_j (Delta A_j * G_{sigma_r(zeta_j - z_k)})(x, y).

What is fast. Away from its caps the tube is invariant under translation
along its axis a_hat = (cos phi cos theta, cos phi sin theta, sin phi), and so
is the light it absorbs per unit volume: rho(P) = rho_perp(P - ((P - c).a_hat)
a_hat). For the hybrid (section 2.3 (H)) rho = mu h, with h the cone's average
of exp(-mu l_back) / s_z (Block 21 ray_history): it is computed once, by
Block 21 on the infinite tube, on a square lattice of the cross-section's
plane through c (basis e1 = e_v, e2 = a_hat x e_v), and read at any point by
the bicubic interpolant of that lattice (scipy.interpolate.RectBivariateSpline).
Slab j's absorbed light at a column is then Block 21's slab rule with the
section's h,

    Delta A_j(x, y) = a_j(x, y) h_perp(proj(x, y, zbar_j(x, y)))        (H)

(a_j Block 2's slab absorbance, zbar_j the midpoint of the column's part
inside the slab): the integral of rho over the slab's part of the column by
the midpoint rule of Block 21 (21.3), the rule Block 26 applies. For
partition_vertical and linear the cross-section is in closed form (Block 2's
slab_absorbance and transmitted_before: T_<j (1 - exp(-a_j)), resp. a_j), so
no lattice is built for them.

Everything else is Block 26's discretisation, unchanged so that the two
renderers differ only in how h is evaluated: the fine lattice of
line_fine_factor samples per pixel side aligned with the pixels; the slab
lattice tiling the depth range of the whole tube's columns (Block 2
slab_centres, dzeta_um); the reach (a slab's samples are drawn only within
line_reach_sigmas x its widest kernel of the pixels' box, a (slab, plane)
pair beyond line_reach_sigmas x its kernel contributes 0); the exact pixel
integral of each sample's Gaussian (Block 26's _pixel_kernel, ndtr); the read
by profiles.sample_profile.

The section is cached (SectionCache) by (d, mu, phi, theta, aspect,
absorption, the directions and their weights, light_direction, the lattice
size): a shift of c, lateral or in depth, reuses it. theta is in the key
because the condenser's quadrature (Block 9: n_psi azimuths) is not invariant
under a rotation about z, so h depends on theta at the quadrature's
resolution (see the block's report).

Validity (D-138.2): extrusion_valid is True when every point of the axis whose
lateral (x, y) distance to the read pixels' box is at most
line_reach_sigmas x max_{j,k} sigma_r(zeta_j - z_k) + d (1 + tan(theta_max))
(sin(theta_max) = ray_na / ray_n_oil; j over the slab lattice above, k over
ks) lies inside the tube's own length; line_profiles raises ExtrusionInvalid
otherwise.

Directions: ray_world.directions(inference.ray_n_rho, inference.ray_n_psi,
renderer.ray_na, renderer.ray_n_oil) (D-137.2).

Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import hashlib
import math
from collections import OrderedDict

import numpy as np
from scipy import interpolate, special

from ..analysis.profiles import sample_profile        # Block 5's reading rule
from . import geometry
from . import hybrid as _hybrid
from . import line_render as _lr                      # Block 26: line arguments, pixels, pixel kernel
from . import ray_world
from .kernel import sigma_r

ABSORPTIONS_LINE = _lr.ABSORPTIONS_LINE
SECTION_NODES = 256       # lattice intervals across the section's larger diameter (the read's accuracy; report)
SECTION_PAD = 4           # lattice nodes beyond the section on each side (the bicubic's support at the rim)


class ExtrusionInvalid(ValueError):
    """A cap of the tube is within reach of the measuring line: the extrusion does not hold (D-138.2)."""


# ------------------------------------------------------------ the section ---

class _Section(object):
    """h_perp on the lattice of the cross-section's plane through c (offsets q1 along e1, q2 along e2, um)."""

    def __init__(self, e1, e2, spline, n1, n2):
        self.e1, self.e2, self.spline, self.shape = e1, e2, spline, (n1, n2)

    def h(self, q):
        """h at offsets q (n, 3) um from the tube's centre c."""
        return self.spline.ev(q @ self.e1, q @ self.e2)


class SectionCache(object):
    """Cross-sections by (d, mu, phi, theta, aspect, absorption, directions,
    light_direction, lattice); least recently used dropped beyond max_entries.
    Counters: hits, misses."""

    def __init__(self, max_entries=64):
        if int(max_entries) != max_entries or max_entries < 1:
            raise ValueError("max_entries must be an integer >= 1")
        self.max_entries = int(max_entries)
        self._store = OrderedDict()
        self.hits = 0
        self.misses = 0

    def __len__(self):
        return len(self._store)

    def get(self, key, build):
        if key in self._store:
            self._store.move_to_end(key)
            self.hits += 1
            return self._store[key]
        self.misses += 1
        value = build()
        self._store[key] = value
        while len(self._store) > self.max_entries:
            self._store.popitem(last=False)
        return value


def _section_basis(tube):
    """e1 = e_v, e2 = a_hat x e_v (unit, orthogonal to the observed axis) and the
    section's half-widths along them (um): r along e1; along e2 the r for which
    S^-1 (q2 e2) is r from the unsquashed axis t0 (S = diag(1, 1, k))."""
    a = tube.t_hat
    e1 = tube.e_v
    e2 = np.cross(a, e1)
    e2 = e2 / np.linalg.norm(e2)
    w = e2 * np.array([1.0, 1.0, 1.0 / tube.aspect])
    t0 = tube.t_hat0
    perp = np.linalg.norm(w - (w @ t0) * t0)
    return e1, e2, tube.r, tube.r / perp


def _build_section(tube, mu, S, W, light):
    """h_perp of the infinite tube (Block 21 ray_history) on a square lattice
    of spacing 2 max(b1, b2) / SECTION_NODES covering the section and
    SECTION_PAD nodes beyond it; the bicubic interpolant (s = 0)."""
    e1, e2, b1, b2 = _section_basis(tube)
    hq = 2.0 * max(b1, b2) / SECTION_NODES
    n1 = int(math.ceil(b1 / hq)) + SECTION_PAD
    n2 = int(math.ceil(b2 / hq)) + SECTION_PAD
    g1 = np.arange(-n1, n1 + 1) * hq
    g2 = np.arange(-n2, n2 + 1) * hq
    Q1, Q2 = np.meshgrid(g1, g2, indexing="ij")
    inf = geometry.Tube((0.0, 0.0, 0.0), tube.r, tube.phi, tube.theta, tube.aspect, None)
    P = Q1.reshape(-1, 1) * e1[None, :] + Q2.reshape(-1, 1) * e2[None, :]
    hv = _hybrid.ray_history(P, inf, mu, S, W, light).reshape(Q1.shape)
    spline = interpolate.RectBivariateSpline(g1, g2, hv, kx=3, ky=3, s=0)
    return _Section(e1, e2, spline, g1.size, g2.size)


def _directions_digest(S, W):
    m = hashlib.sha1()
    m.update(np.ascontiguousarray(S, dtype=np.float64).tobytes())
    m.update(np.ascontiguousarray(W, dtype=np.float64).tobytes())
    return m.hexdigest()


def _section_for(tube, mu, S, W, light, cache):
    key = ("hybrid", float(tube.d), float(mu), float(tube.phi), float(tube.theta), float(tube.aspect),
           _directions_digest(S, W), int(light), SECTION_NODES, SECTION_PAD)
    return cache.get(key, lambda: _build_section(tube, mu, S, W, light))


# ------------------------------------------------------- the pixel kernel ---

def _pixel_kernel_lattice(n_lo, n_px, i_lo, n_s, f, sigma, h, p):
    """Block 26's _pixel_kernel for pixel edges at (n_lo + i - 1/2) p,
    i = 0 .. n_px, and lattice samples at (i_lo + s + 1/2) h - p/2,
    s = 0 .. n_s - 1 (h = p / f): (n_px, n_s), h times the mean over each
    pixel of the sample's Gaussian of SD sigma. An edge's offset from a sample
    is (m - 1/2) h with the integer m = (n_lo + i) f - (i_lo + s), and a
    pixel's upper edge has m + f, so entry (i, s) is a function D of
    m = (n_lo + i) f - i_lo - s alone: D is evaluated once on the range of m
    and gathered (the matrix is Toeplitz). The same tail rule as Block 26."""
    m_min = n_lo * f - i_lo - (n_s - 1)
    m_max = (n_lo + n_px - 1) * f - i_lo
    m = np.arange(m_min, m_max + f + 1)
    t = (m - 0.5) * (h / sigma)
    lo, hi = t[:m.size - f], t[f:]
    D = (h / p) * np.where(lo > 0, special.ndtr(-lo) - special.ndtr(-hi), special.ndtr(hi) - special.ndtr(lo))
    idx = (np.arange(n_px)[:, None] * f) - np.arange(n_s)[None, :] + (n_lo * f - i_lo - m_min)
    return D[idx]


# ----------------------------------------------------------- the geometry ---

def _check_tube(tube):
    if tube.half_length is None:
        raise ValueError("the line renderer needs a capped tube (half_length set): its footprint must be finite")


def _pixel_box(o, yh, eu, v, n_avg, step, p):
    """Block 26's box: the pixels the read touches, one pixel of margin; its
    pixel edges ex, ey (um) and its area [bx0, bx1] x [by0, by1]."""
    cols, rows = _lr.pixels_needed(o, yh, eu, v, n_avg, step, p)
    c_lo, c_hi = int(cols.min()) - 1, int(cols.max()) + 1
    r_lo, r_hi = int(rows.min()) - 1, int(rows.max()) + 1
    ex = (np.arange(c_lo, c_hi + 2) - 0.5) * p
    ey = (np.arange(r_lo, r_hi + 2) - 0.5) * p
    return c_lo, c_hi, r_lo, r_hi, ex, ey


def _footprint(tube, h, p):
    """Block 26's lattice over the tube's whole footprint: sample centres
    xs_all, ys_all (um), the columns' X, Y and Block 2's intervals."""
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
    return int(ix[0]), int(iy[0]), xs_all, ys_all, X, Y, z_lo, z_hi, inside


def _axis_length(tube):
    """Half-range of the observed arc length s (P = c + s a_hat) inside the tube's own length."""
    U = float(tube.half_length)
    if tube.end_cut == "vertical":
        return U / math.cos(tube.phi)                        # |u| = |s| cos(phi) <= U
    a = tube.t_hat * np.array([1.0, 1.0, 1.0 / tube.aspect])
    return U / float(np.linalg.norm(a))                     # |s0| = |s| |S^-1 a_hat| <= U


def _line_box_interval(px, py, dx, dy, box, M):
    """The parameters t of the points (px + t dx, py + t dy) whose Euclidean
    distance to the rectangle box = (x0, x1, y0, y1) is at most M: the line
    against the rectangle grown by M (two crossed rectangles and four corner
    discs, a convex set), -> (t_lo, t_hi), or None when the line misses it.
    (dx, dy) != 0."""
    x0, x1, y0, y1 = box
    lo, hi = math.inf, -math.inf

    def rect(ax0, ax1, ay0, ay1):
        t0, t1 = -math.inf, math.inf
        for p0, d0, a0, a1 in ((px, dx, ax0, ax1), (py, dy, ay0, ay1)):
            if d0 == 0.0:
                if not (a0 <= p0 <= a1):
                    return None
            else:
                ta, tb = (a0 - p0) / d0, (a1 - p0) / d0
                t0, t1 = max(t0, min(ta, tb)), min(t1, max(ta, tb))
        return (t0, t1) if t1 >= t0 else None

    pieces = [rect(x0 - M, x1 + M, y0, y1), rect(x0, x1, y0 - M, y1 + M)]
    A = dx * dx + dy * dy
    for qx, qy in ((x0, y0), (x0, y1), (x1, y0), (x1, y1)):
        ex_, ey_ = px - qx, py - qy
        B = ex_ * dx + ey_ * dy
        C = ex_ * ex_ + ey_ * ey_ - M * M
        disc = B * B - A * C
        if disc >= 0.0:
            sq = math.sqrt(disc)
            pieces.append(((-B - sq) / A, (-B + sq) / A))
    for piece in pieces:
        if piece is not None:
            lo, hi = min(lo, piece[0]), max(hi, piece[1])
    return (lo, hi) if hi >= lo else None


def _validity(tube, ks_arr, ex, ey, cfg, zeta):
    """(valid, margin um, axis interval reached (s_lo, s_hi) or None, half-range inside)."""
    rcfg, icfg, acq = cfg.renderer, cfg.inference, cfg.acquisition
    z_planes = ks_arr * float(acq.dz_um)
    sig_max = float(np.max(sigma_r(zeta[:, None] - z_planes[None, :], rcfg)))
    s_m = float(rcfg.ray_na) / float(rcfg.ray_n_oil)
    tan_max = s_m / math.sqrt(1.0 - s_m * s_m)
    M = float(icfg.line_reach_sigmas) * sig_max + tube.d * (1.0 + tan_max)
    a = tube.t_hat
    S_in = _axis_length(tube)
    box = (float(ex[0]), float(ex[-1]), float(ey[0]), float(ey[-1]))
    if math.hypot(a[0], a[1]) == 0.0:          # phi < pi/2 always (Block 2), kept for safety
        return False, M, None, S_in
    iv = _line_box_interval(float(tube.c[0]), float(tube.c[1]), float(a[0]), float(a[1]), box, M)
    if iv is None:
        return True, M, None, S_in
    return (iv[0] >= -S_in and iv[1] <= S_in), M, iv, S_in


def _prepare(tube, ks, origin_xy, y_hat, e_u, v, n_avg, step, cfg):
    _check_tube(tube)
    o, yh, eu, v, n_avg, step = _lr._line_args(origin_xy, y_hat, e_u, v, n_avg, step)
    ks_arr = np.asarray(ks).reshape(-1)
    if ks_arr.size and (not np.all(np.isfinite(ks_arr.astype(float))) or np.any(ks_arr.astype(float) % 1 != 0)):
        raise ValueError("ks must be integer plane indices")
    ks_arr = ks_arr.astype(np.int64)
    icfg = cfg.inference
    f = int(icfg.line_fine_factor)
    if f != icfg.line_fine_factor or f < 1:
        raise ValueError("inference.line_fine_factor must be an integer >= 1")
    if not (float(icfg.line_reach_sigmas) > 0):
        raise ValueError("inference.line_reach_sigmas must be > 0")
    return o, yh, eu, v, n_avg, step, ks_arr, f


def extrusion_valid(tube, ks, origin_xy, y_hat, e_u, v, n_avg, step, cfg):
    """-> bool: no cap within reach of the pixels the line reads (D-138.2; the
    margin in the module docstring). An empty ks or a tube whose columns miss
    the lattice is valid (nothing is drawn)."""
    o, yh, eu, v, n_avg, step, ks_arr, f = _prepare(tube, ks, origin_xy, y_hat, e_u, v, n_avg, step, cfg)
    if ks_arr.size == 0:
        return True
    p = float(cfg.acquisition.res0_um)
    _c0, _c1, _r0, _r1, ex, ey = _pixel_box(o, yh, eu, v, n_avg, step, p)
    _ix0, _iy0, _xs, _ys, _X, _Y, z_lo, z_hi, inside = _footprint(tube, p / f, p)
    if not inside.any():
        return True
    zeta = geometry.slab_centres(float(z_lo[inside].min()), float(z_hi[inside].max()), float(cfg.renderer.dzeta_um))
    return bool(_validity(tube, ks_arr, ex, ey, cfg, zeta)[0])


# ------------------------------------------------------------- the model ---

def line_profiles(tube, mu, ks, origin_xy, y_hat, e_u, v, n_avg, step, cfg, absorption=None, cache=None):
    """tau (n_planes, n_v) float64, dimensionless: Block 26's line_profiles
    (rows in the order of ks; what profiles.sample_profile reads from the
    pixel means of tau_k of Eqs. 5-6, before the camera chain), with the
    hybrid's h read off one cached cross-section. cache: a SectionCache to
    reuse across calls (None: a fresh one). ExtrusionInvalid (a ValueError)
    when a cap is within reach (extrusion_valid False); ValueError for mu < 0
    or not finite, a non-unit axis, an unknown absorption, an uncapped tube."""
    return _profiles(tube, mu, ks, origin_xy, y_hat, e_u, v, n_avg, step, cfg, absorption, cache, check=True)


def _profiles(tube, mu, ks, origin_xy, y_hat, e_u, v, n_avg, step, cfg, absorption, cache, check=True, S=None, W=None):
    """line_profiles; check=False skips the validity test (the smoke test's
    measurements only); S, W override the inference's directions (the same)."""
    acq, rcfg, icfg = cfg.acquisition, cfg.renderer, cfg.inference
    absorption = icfg.absorption if absorption is None else absorption
    if absorption not in ABSORPTIONS_LINE:
        raise ValueError("unknown absorption %r for the line renderer (one of %r)" % (absorption, ABSORPTIONS_LINE))
    mu = float(mu)
    if not (math.isfinite(mu) and mu >= 0):
        raise ValueError("mu must be finite and >= 0 (1/um)")
    o, yh, eu, v, n_avg, step, ks_arr, f = _prepare(tube, ks, origin_xy, y_hat, e_u, v, n_avg, step, cfg)
    light = acq.light_direction
    if light not in (+1, -1):
        raise ValueError("light_direction must be +1 or -1")
    if cache is None:
        cache = SectionCache()
    elif not isinstance(cache, SectionCache):
        raise ValueError("cache must be a line_fast.SectionCache or None")
    reach = float(icfg.line_reach_sigmas)
    p, dz, dzeta = float(acq.res0_um), float(acq.dz_um), float(rcfg.dzeta_um)
    h = p / f
    n_planes, n_v = ks_arr.size, v.size
    if n_planes == 0:
        return np.ones((0, n_v))

    c_lo, c_hi, r_lo, r_hi, ex, ey = _pixel_box(o, yh, eu, v, n_avg, step, p)
    bx0, bx1, by0, by1 = ex[0], ex[-1], ey[0], ey[-1]
    ix0, iy0, xs_all, ys_all, X, Y, z_lo, z_hi, inside = _footprint(tube, h, p)
    if not inside.any():
        return np.ones((n_planes, n_v))
    zeta = geometry.slab_centres(float(z_lo[inside].min()), float(z_hi[inside].max()), dzeta)
    if check and not _validity(tube, ks_arr, ex, ey, cfg, zeta)[0]:
        raise ExtrusionInvalid("a cap of the tube is within reach of the line's pixels: the extrusion does not hold "
                               "(D-138.2); flag the node")
    if mu == 0.0:
        return np.ones((n_planes, n_v))
    z_planes = ks_arr * dz
    sig = sigma_r(zeta[:, None] - z_planes[None, :], rcfg)        # (J, n_planes), um
    R_jk = reach * sig
    R_j = R_jk.max(axis=1)

    dxb = np.maximum(np.maximum(bx0 - xs_all, xs_all - bx1), 0.0)
    dyb = np.maximum(np.maximum(by0 - ys_all, ys_all - by1), 0.0)
    dist = np.hypot(dxb[None, :], dyb[:, None])
    keep = inside & (dist <= float(R_j.max()))
    if not keep.any():
        return np.ones((n_planes, n_v))
    kr = np.flatnonzero(keep.any(axis=1))
    kc = np.flatnonzero(keep.any(axis=0))
    sl = (slice(int(kr[0]), int(kr[-1]) + 1), slice(int(kc[0]), int(kc[-1]) + 1))
    ix0, iy0 = ix0 + int(kc[0]), iy0 + int(kr[0])       # lattice index of the kept box's first sample
    X, Y, z_lo, z_hi, inside, dist = X[sl], Y[sl], z_lo[sl], z_hi[sl], inside[sl], dist[sl]

    section = None
    if absorption == "hybrid":
        if S is None:
            S, W = ray_world.directions(icfg.ray_n_rho, icfg.ray_n_psi, rcfg.ray_na, rcfg.ray_n_oil)
        section = _section_for(tube, mu, S, W, light, cache)
        c3 = np.asarray(tube.c, dtype=float)

    n_rows, n_cols = r_hi - r_lo + 1, c_hi - c_lo + 1
    deficit = np.zeros((n_planes, n_rows, n_cols))
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
        a = geometry.slab_absorbance(zl, zh, zj, dzeta, mu)
        if absorption == "hybrid":
            mm = m & (a > 0)
            dA = np.zeros(m.shape)
            if mm.any():
                zbar = 0.5 * (np.maximum(zl[mm], zj - 0.5 * dzeta) + np.minimum(zh[mm], zj + 0.5 * dzeta))
                q = np.column_stack([X[rs, cs][mm] - c3[0], Y[rs, cs][mm] - c3[1], zbar - c3[2]])
                dA[mm] = a[mm] * section.h(q)
        elif absorption == "partition_vertical":
            dA = np.where(m, geometry.transmitted_before(zl, zh, zj, dzeta, mu, light) * (-np.expm1(-a)), 0.0)
        else:
            dA = np.where(m, a, 0.0)
        dmin = float(dist[rs, cs][m].min())
        for k in range(n_planes):
            if dmin > R_jk[j, k]:
                continue
            s = float(sig[j, k])
            Kx = _pixel_kernel_lattice(c_lo, n_cols, ix0 + cs.start, cs.stop - cs.start, f, s, h, p)
            Ky = _pixel_kernel_lattice(r_lo, n_rows, iy0 + rs.start, rs.stop - rs.start, f, s, h, p)
            deficit[k] += Ky @ (dA @ Kx.T)

    frame = _lr._Frame(c_lo, r_lo, p)
    out = np.empty((n_planes, n_v))
    for k in range(n_planes):
        out[k] = 1.0 - sample_profile(deficit[k], frame, o, yh, eu, v, n_avg, step)
    return out
