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
(amendment 1 (A)): the condenser's quadrature (Block 9: n_psi azimuths) is not
invariant under a rotation about z, so h depends on theta at the quadrature's
resolution.

Validity (D-138.2; SPEC Block 35, amendment 1 (C), D-140): the block
computes the tube without caps, read on the capped tube's lattice.
extrusion_valid is True when the point of the axis nearest the line's origin
(nearest in the image plane: the origin has no depth) is at least
inference.extrusion_cap_margin_um along the axis (observed arc length) from
each cap; line_profiles raises ExtrusionInvalid otherwise, before the mu = 0
shortcut.

Directions: ray_world.directions(inference.ray_n_rho, inference.ray_n_psi,
renderer.ray_na, renderer.ray_n_oil) (D-137.2).

The own shadow split by depth (SPEC Block 35, amendment 2; D-151, D-156):
line_shadow_split runs line_profiles' slab loop with a sink that receives
each (slab j, plane k) pair's pixel-mean term s_jk and books it twice more,
w_jk s_jk to the band's image and (1 - w_jk) s_jk to the rest, w_jk the
fraction of slab j's thickness inside [z_k - beta, z_k + beta]; with layers,
u_jm s_jk to layer m's image, u_jm the fraction inside the layer
[z_0 + m h -+ h/2]. Each image is then read as line_profiles reads its
deficit (profiles.sample_profile, linear in the image), so S_in + S_out and
sum_m L_m equal 1 - tau to rounding. line_profiles itself is unchanged.

Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import dataclasses
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


def _cap_distance(tube, origin_xy):
    """(distance um from the nearer cap, along the axis, of the axis point
    nearest the origin in the image plane; that point's arc length s; the
    half-range inside). Arc length is the observed one, P = c + s a_hat."""
    a = tube.t_hat
    axy = math.hypot(a[0], a[1])                 # cos(phi) > 0 (phi < pi/2, Block 2)
    dxy = np.asarray(origin_xy, dtype=float) - np.asarray(tube.c[:2], dtype=float)
    s = float(dxy[0] * a[0] + dxy[1] * a[1]) / (axy * axy)
    S_in = _axis_length(tube)
    return S_in - abs(s), s, S_in


def _is_valid(tube, origin_xy, cfg):
    return _cap_distance(tube, origin_xy)[0] >= float(cfg.inference.extrusion_cap_margin_um)


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
    """-> bool: the axis point nearest the line's origin lies at least
    inference.extrusion_cap_margin_um along the axis from each cap (SPEC Block
    35, amendment 1 (C)). The other arguments are validated as line_profiles'."""
    o = _prepare(tube, ks, origin_xy, y_hat, e_u, v, n_avg, step, cfg)[0]
    return bool(_is_valid(tube, o, cfg))


# ------------------------------------------------------------- the model ---

def line_profiles(tube, mu, ks, origin_xy, y_hat, e_u, v, n_avg, step, cfg, absorption=None, cache=None,
                  kernel=None):
    """tau (n_planes, n_v) float64, dimensionless: Block 26's line_profiles
    (rows in the order of ks; what profiles.sample_profile reads from the
    pixel means of tau_k of Eqs. 5-6, before the camera chain), with the
    hybrid's h read off one cached cross-section. cache: a SectionCache to
    reuse across calls (None: a fresh one). kernel (Block 49, wave 7): None =
    Block 4's Gaussian law (this module's path, unchanged); a
    line_sgt.SGTKernel = the SGT point-spread function of Block 49; ValueError
    for anything else. ExtrusionInvalid (a ValueError) when extrusion_valid is
    False (checked before the mu = 0 shortcut); ValueError for mu < 0
    or not finite, a non-unit axis, an unknown absorption, an uncapped tube."""
    if kernel is None:
        return _profiles(tube, mu, ks, origin_xy, y_hat, e_u, v, n_avg, step, cfg, absorption, cache, check=True)
    from . import line_sgt
    if not isinstance(kernel, line_sgt.SGTKernel):
        raise ValueError("kernel must be None (the Gaussian law) or a line_sgt.SGTKernel, got %r" % (kernel,))
    return line_sgt.profiles(tube, mu, ks, origin_xy, y_hat, e_u, v, n_avg, step, cfg, absorption, cache, kernel)


def _check_args(tube, mu, ks, origin_xy, y_hat, e_u, v, n_avg, step, cfg, absorption, cache):
    """line_profiles' argument checks, in its order: -> (absorption, mu, _prepare's tuple, light, cache)."""
    absorption = cfg.inference.absorption if absorption is None else absorption
    if absorption not in ABSORPTIONS_LINE:
        raise ValueError("unknown absorption %r for the line renderer (one of %r)" % (absorption, ABSORPTIONS_LINE))
    mu = float(mu)
    if not (math.isfinite(mu) and mu >= 0):
        raise ValueError("mu must be finite and >= 0 (1/um)")
    prepared = _prepare(tube, ks, origin_xy, y_hat, e_u, v, n_avg, step, cfg)
    light = cfg.acquisition.light_direction
    if light not in (+1, -1):
        raise ValueError("light_direction must be +1 or -1")
    if cache is None:
        cache = SectionCache()
    elif not isinstance(cache, SectionCache):
        raise ValueError("cache must be a line_fast.SectionCache or None")
    return absorption, mu, prepared, light, cache


class _Setup(object):
    """What the slab loop needs (_render_setup): the line, the pixel box, the kept part of the fine lattice, the
    slabs, the kernel widths and reaches, the absorption. ones: True when the result is exactly 1 (no planes,
    mu = 0, no column inside the tube, no sample within reach)."""


def _render_setup(tube, mu, ks, origin_xy, y_hat, e_u, v, n_avg, step, cfg, absorption, cache, check, reach=None,
                  S=None, W=None, extra_checks=None):
    """The slab loop's setup, shared by the Gaussian path (_profiles) and Block 49's SGT path (line_sgt.profiles):
    line_profiles' argument checks in its order, then extra_checks() (a callable or None: the SGT path's own
    ValueErrors), the validity test (check), the mu = 0 shortcut, the pixel box, the footprint, the slabs,
    sig = sigma_r(zeta_j - z_k) and R_jk = reach x sig (reach None: inference.line_reach_sigmas), the samples
    within max R_jk of the box, the hybrid's section. -> _Setup (the arithmetic is the one Block 35's loop
    performed inline)."""
    acq, rcfg, icfg = cfg.acquisition, cfg.renderer, cfg.inference
    absorption, mu, (o, yh, eu, v, n_avg, step, ks_arr, f), light, cache = _check_args(
        tube, mu, ks, origin_xy, y_hat, e_u, v, n_avg, step, cfg, absorption, cache)
    if extra_checks is not None:
        extra_checks()
    reach = float(icfg.line_reach_sigmas) if reach is None else float(reach() if callable(reach) else reach)
    st = _Setup()
    p, dz, dzeta = float(acq.res0_um), float(acq.dz_um), float(rcfg.dzeta_um)
    h = p / f
    n_planes, n_v = ks_arr.size, v.size
    st.absorption, st.mu, st.light, st.cache = absorption, mu, light, cache
    st.o, st.yh, st.eu, st.v, st.n_avg, st.step, st.ks_arr, st.f = o, yh, eu, v, n_avg, step, ks_arr, f
    st.p, st.dz, st.dzeta, st.h, st.n_planes, st.n_v = p, dz, dzeta, h, n_planes, n_v
    st.ones = True
    if check and not _is_valid(tube, o, cfg):
        raise ExtrusionInvalid("the line's origin is within inference.extrusion_cap_margin_um of a cap along the "
                               "axis: the extrusion does not hold (D-138.2, D-140); flag the node")
    if n_planes == 0 or mu == 0.0:
        return st

    c_lo, c_hi, r_lo, r_hi, ex, ey = _pixel_box(o, yh, eu, v, n_avg, step, p)
    bx0, bx1, by0, by1 = ex[0], ex[-1], ey[0], ey[-1]
    ix0, iy0, xs_all, ys_all, X, Y, z_lo, z_hi, inside = _footprint(tube, h, p)
    if not inside.any():
        return st
    zeta = geometry.slab_centres(float(z_lo[inside].min()), float(z_hi[inside].max()), dzeta)
    z_planes = ks_arr * dz
    sig = sigma_r(zeta[:, None] - z_planes[None, :], rcfg)        # (J, n_planes), um
    R_jk = reach * sig
    R_j = R_jk.max(axis=1)

    dxb = np.maximum(np.maximum(bx0 - xs_all, xs_all - bx1), 0.0)
    dyb = np.maximum(np.maximum(by0 - ys_all, ys_all - by1), 0.0)
    dist = np.hypot(dxb[None, :], dyb[:, None])
    keep = inside & (dist <= float(R_j.max()))
    if not keep.any():
        return st
    kr = np.flatnonzero(keep.any(axis=1))
    kc = np.flatnonzero(keep.any(axis=0))
    sl = (slice(int(kr[0]), int(kr[-1]) + 1), slice(int(kc[0]), int(kc[-1]) + 1))
    ix0, iy0 = ix0 + int(kc[0]), iy0 + int(kr[0])       # lattice index of the kept box's first sample
    X, Y, z_lo, z_hi, inside, dist = X[sl], Y[sl], z_lo[sl], z_hi[sl], inside[sl], dist[sl]

    section = None
    c3 = None
    if absorption == "hybrid":
        if S is None:
            S, W = ray_world.directions(icfg.ray_n_rho, icfg.ray_n_psi, rcfg.ray_na, rcfg.ray_n_oil)
        section = _section_for(tube, mu, S, W, light, cache)
        c3 = np.asarray(tube.c, dtype=float)

    st.ones = False
    st.c_lo, st.c_hi, st.r_lo, st.r_hi, st.bx0, st.bx1, st.by0, st.by1 = c_lo, c_hi, r_lo, r_hi, bx0, bx1, by0, by1
    st.ix0, st.iy0, st.X, st.Y, st.z_lo, st.z_hi, st.inside, st.dist = ix0, iy0, X, Y, z_lo, z_hi, inside, dist
    st.zeta, st.z_planes, st.sig, st.R_jk, st.R_j = zeta, z_planes, sig, R_jk, R_j
    st.section, st.c3 = section, c3
    st.n_rows, st.n_cols = r_hi - r_lo + 1, c_hi - c_lo + 1
    return st


def _slab_maps(st, order=None):
    """For each slab j whose drawn samples (within R_j of the box) meet the tube: (j, zeta_j, rs, cs, dA, dmin),
    rs, cs the slices of the kept lattice holding the slab's samples, dA (rows, cols) its absorbed light under the
    absorption named (0 outside the slab), dmin the distance (um) of its nearest drawn sample from the box. order:
    the slabs' indices in the order wanted (None: increasing; the Gaussian path's)."""
    zeta, dzeta, mu, light = st.zeta, st.dzeta, st.mu, st.light
    inside, z_lo, z_hi, dist, R_j = st.inside, st.z_lo, st.z_hi, st.dist, st.R_j
    for j in (range(zeta.size) if order is None else order):
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
        if st.absorption == "hybrid":
            c3 = st.c3
            mm = m & (a > 0)
            dA = np.zeros(m.shape)
            if mm.any():
                zbar = 0.5 * (np.maximum(zl[mm], zj - 0.5 * dzeta) + np.minimum(zh[mm], zj + 0.5 * dzeta))
                q = np.column_stack([st.X[rs, cs][mm] - c3[0], st.Y[rs, cs][mm] - c3[1], zbar - c3[2]])
                dA[mm] = a[mm] * st.section.h(q)
        elif st.absorption == "partition_vertical":
            dA = np.where(m, geometry.transmitted_before(zl, zh, zj, dzeta, mu, light) * (-np.expm1(-a)), 0.0)
        else:
            dA = np.where(m, a, 0.0)
        dmin = float(dist[rs, cs][m].min())
        yield j, zj, rs, cs, dA, dmin


def _read(st, deficit):
    """tau (n_planes, n_v) = 1 - profiles.sample_profile of each plane's pixel-mean deficit (n_rows, n_cols)."""
    frame = _lr._Frame(st.c_lo, st.r_lo, st.p)
    out = np.empty((st.n_planes, st.n_v))
    for k in range(st.n_planes):
        out[k] = 1.0 - sample_profile(deficit[k], frame, st.o, st.yh, st.eu, st.v, st.n_avg, st.step)
    return out


def _profiles(tube, mu, ks, origin_xy, y_hat, e_u, v, n_avg, step, cfg, absorption, cache, check=True, S=None, W=None,
              sink=None):
    """line_profiles; check=False skips the validity test (the smoke test's
    measurements only); S, W override the inference's directions (the same).
    sink: None, or a callable sink(j, zeta_j, k, term) called with every
    (slab, plane) pair's pixel-mean term (n_rows, n_cols) as it is added to
    plane k's deficit (amendment 2: the split by depth); the result is the
    same with or without it."""
    st = _render_setup(tube, mu, ks, origin_xy, y_hat, e_u, v, n_avg, step, cfg, absorption, cache, check,
                       None, S, W)
    if st.ones:
        return np.ones((st.n_planes, st.n_v))
    f, h, p, c_lo, r_lo, n_rows, n_cols = st.f, st.h, st.p, st.c_lo, st.r_lo, st.n_rows, st.n_cols
    sig, R_jk = st.sig, st.R_jk
    deficit = np.zeros((st.n_planes, n_rows, n_cols))
    for j, zj, rs, cs, dA, dmin in _slab_maps(st):
        for k in range(st.n_planes):
            if dmin > R_jk[j, k]:
                continue
            s = float(sig[j, k])
            Kx = _pixel_kernel_lattice(c_lo, n_cols, st.ix0 + cs.start, cs.stop - cs.start, f, s, h, p)
            Ky = _pixel_kernel_lattice(r_lo, n_rows, st.iy0 + rs.start, rs.stop - rs.start, f, s, h, p)
            term = Ky @ (dA @ Kx.T)
            deficit[k] += term
            if sink is not None:
                sink(j, zj, k, term)
    return _read(st, deficit)


# ------------------------------------- the own shadow split by depth (amendment 2) ---

@dataclasses.dataclass(frozen=True)
class OwnSplit:
    """The own shadow 1 - tau of line_profiles split by depth (SPEC Block 35, amendment 2; D-151, D-156).

    s_in, s_out   (n_planes, n_v) float64, dimensionless, rows in the order of ks: the parts of plane k's
                  deficit from the slabs inside / outside the band |zeta - z_k| <= band_half_um, a slab's
                  share being the fraction of its thickness inside the band; s_in + s_out = 1 - tau
    layers        (n_layers, n_planes, n_v) float64, dimensionless, or None: the deficit of each depth layer
                  of thickness layer_um (their sum over layers = 1 - tau)
    layer_z       (n_layers,) um, the layers' centres layer_origin + m layer_um, increasing; or None
    band_half_um  float, um
    layer_um      float, um, or None
    """

    s_in: np.ndarray
    s_out: np.ndarray
    layers: object
    layer_z: object
    band_half_um: float
    layer_um: object


def line_shadow_split(tube, mu, ks, origin_xy, y_hat, e_u, v, n_avg, step, cfg, band_half_um, absorption=None,
                      cache=None, layer_um=None, layer_origin_um=None):
    """-> OwnSplit: line_profiles' deficit 1 - tau_k(v_n) split by depth (SPEC Block 35, amendment 2).

    Arguments as line_profiles, plus band_half_um (um, >= 0): the half-thickness beta of plane k's band;
    layer_um (um, > 0, or None): the thickness h of the depth layers to return; layer_origin_um (um, or
    None: tube.c[2]): the depth z_0 the layers are centred on (centres z_0 + m h). Validity and errors as
    line_profiles (ExtrusionInvalid first); mu = 0 or no planes gives zeros; ValueError for band_half_um < 0
    or not finite, layer_um <= 0 or not finite, layer_origin_um not finite."""
    s_in, s_out, layers, layer_z, _whole = _split_reads(tube, mu, ks, origin_xy, y_hat, e_u, v, n_avg, step, cfg,
                                                        band_half_um, absorption, cache, layer_um, layer_origin_um)
    return OwnSplit(s_in=s_in, s_out=s_out, layers=layers, layer_z=layer_z, band_half_um=float(band_half_um),
                    layer_um=None if layer_um is None else float(layer_um))


def _share(lo, hi, a, b, width):
    """The fraction of the slab [lo, hi] (thickness width, um) inside the interval [a, b]: exactly 1 when the
    slab lies wholly inside, exactly 0 when it does not meet it, else the overlap / width clipped to [0, 1]."""
    if lo >= a and hi <= b:
        return 1.0
    return min(1.0, max(0.0, min(hi, b) - max(lo, a)) / width)


def _layer_range(zeta, dzeta, h, z0):
    """(m_min, centres (M,) um, shares (J, M)): the layers m (centre z0 + m h, thickness h) that meet the slabs
    [zeta_j -+ dzeta/2] (a positive overlap), and u_{j,m}, the fraction of slab j's thickness inside layer m."""
    if zeta.size == 0:
        return 0, np.zeros(0), np.zeros((0, 0))
    lo, hi = float(zeta[0]) - 0.5 * dzeta, float(zeta[-1]) + 0.5 * dzeta
    m_lo = int(math.floor((lo - z0) / h - 0.5)) - 1          # one layer of margin each side; trimmed below
    m_hi = int(math.ceil((hi - z0) / h + 0.5)) + 1
    ms = np.arange(m_lo, m_hi + 1)
    centres = z0 + ms * h
    u = np.zeros((zeta.size, ms.size))
    for j, zj in enumerate(zeta):
        a, b = float(zj) - 0.5 * dzeta, float(zj) + 0.5 * dzeta
        for i, lam in enumerate(centres):
            u[j, i] = _share(a, b, float(lam) - 0.5 * h, float(lam) + 0.5 * h, dzeta)
    meet = np.flatnonzero(u.max(axis=0) > 0.0)
    keep = slice(int(meet[0]), int(meet[-1]) + 1)
    return int(ms[keep][0]), centres[keep], u[:, keep]


def _split_reads(tube, mu, ks, origin_xy, y_hat, e_u, v, n_avg, step, cfg, band_half_um, absorption=None,
                 cache=None, layer_um=None, layer_origin_um=None):
    """line_shadow_split's arrays and, last, the read of the unsplit deficit (n_planes, n_v), the whole own
    shadow as the split books it (the smoke test's exact limits). See line_shadow_split.

    Custom: the two shares w_{j,k} (band) and u_{j,m} (layers) applied to each (slab, plane) pair's term in
    line_profiles' slab loop (SPEC Block 35, amendment 2); the read on the line is profiles.sample_profile's,
    linear in the deficit image, so splitting the terms before the read is exact. Checked: no library splits a
    depth-dependent blur by source depth."""
    beta = float(band_half_um)
    if not (math.isfinite(beta) and beta >= 0):
        raise ValueError("band_half_um must be finite and >= 0 (um)")
    if layer_um is not None:
        h_l = float(layer_um)
        if not (math.isfinite(h_l) and h_l > 0):
            raise ValueError("layer_um must be finite and > 0 (um), or None")
    if layer_origin_um is not None and not math.isfinite(float(layer_origin_um)):
        raise ValueError("layer_origin_um must be finite (um), or None")
    absorption, mu, (o, yh, eu, v, n_avg, step, ks_arr, f), light, cache = _check_args(
        tube, mu, ks, origin_xy, y_hat, e_u, v, n_avg, step, cfg, absorption, cache)
    if not _is_valid(tube, o, cfg):
        raise ExtrusionInvalid("the line's origin is within inference.extrusion_cap_margin_um of a cap along the "
                               "axis: the extrusion does not hold (D-138.2, D-140); flag the node")
    acq = cfg.acquisition
    p, dz, dzeta = float(acq.res0_um), float(acq.dz_um), float(cfg.renderer.dzeta_um)
    n_planes, n_v = ks_arr.size, v.size

    # the tube's slabs, as _profiles lays them (Block 2 slab_centres over the whole footprint's columns)
    z_lo, z_hi, inside = _footprint(tube, p / f, p)[6:9]
    zeta = (geometry.slab_centres(float(z_lo[inside].min()), float(z_hi[inside].max()), dzeta) if inside.any()
            else np.zeros(0))
    if layer_um is not None:
        z0 = float(tube.c[2]) if layer_origin_um is None else float(layer_origin_um)
        _m_min, layer_z, u = _layer_range(zeta, dzeta, h_l, z0)
        n_layers = layer_z.size
    else:
        layer_z, n_layers = None, 0

    zeros = np.zeros((n_planes, n_v))
    if n_planes == 0 or mu == 0.0 or zeta.size == 0:
        layers = None if layer_um is None else np.zeros((n_layers, n_planes, n_v))
        return zeros, zeros.copy(), layers, layer_z, zeros.copy()

    z_planes = ks_arr * dz
    c_lo, c_hi, r_lo, r_hi = _pixel_box(o, yh, eu, v, n_avg, step, p)[:4]
    shape = (n_planes, r_hi - r_lo + 1, c_hi - c_lo + 1)
    acc_in, acc_out, acc_all = np.zeros(shape), np.zeros(shape), np.zeros(shape)
    acc_layer = np.zeros((n_layers,) + shape) if layer_um is not None else None
    layer_of = {}                                   # slab j -> [(layer row, u_{j,m} > 0)]
    seen = {}

    def sink(j, zj, k, term):
        if seen.setdefault(j, zj) != zj or zj != zeta[j]:
            raise RuntimeError("the split's slab grid differs from the slab loop's")   # internal consistency
        acc_all[k] += term
        w = _share(zj - 0.5 * dzeta, zj + 0.5 * dzeta, float(z_planes[k]) - beta, float(z_planes[k]) + beta, dzeta)
        if w > 0.0:
            acc_in[k] += w * term
        if w < 1.0:
            acc_out[k] += (1.0 - w) * term
        if acc_layer is not None:
            if j not in layer_of:
                layer_of[j] = [(i, float(u[j, i])) for i in np.flatnonzero(u[j] > 0.0)]
            for i, uj in layer_of[j]:
                acc_layer[i, k] += uj * term

    _profiles(tube, mu, ks_arr, o, yh, eu, v, n_avg, step, cfg, absorption, cache, check=False, sink=sink)

    frame = _lr._Frame(c_lo, r_lo, p)

    def read(img):
        if not img.any():
            return np.zeros(n_v)
        return sample_profile(img, frame, o, yh, eu, v, n_avg, step)

    s_in = np.array([read(acc_in[k]) for k in range(n_planes)]).reshape(n_planes, n_v)
    s_out = np.array([read(acc_out[k]) for k in range(n_planes)]).reshape(n_planes, n_v)
    whole = np.array([read(acc_all[k]) for k in range(n_planes)]).reshape(n_planes, n_v)
    layers = None
    if acc_layer is not None:
        layers = np.zeros((n_layers, n_planes, n_v))
        for i in range(n_layers):
            for k in range(n_planes):
                layers[i, k] = read(acc_layer[i, k])
    return s_in, s_out, layers, layer_z, whole
