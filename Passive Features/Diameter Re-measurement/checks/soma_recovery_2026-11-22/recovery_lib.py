"""The soma fit's recovery on simulated stacks (2026-11-22; D-216.2) -- the simulation and the analysis of the check.

Question. Does analysis.soma_fit.fit_soma (Block 46), with model.soma_render.SomaRenderer (Block 45) as its forward
model at the production settings (the SomaFitConfig defaults: fine_factor 1, dzeta_um 0.14, far_slope "fit"),
recover a known soma -- its height c above all, then a, b, the membrane area S and the equivalent radius
r_eq = sqrt(S / 4 pi) -- from simulated stacks like the pilot's? How large are the errors from (i) the fit's own
discretisation (the truth rendered finer), (ii) a far slope other than the configured one, (iii) a soma that is not an
ellipsoid, (iv) stems leaving the soma?

This module holds the check's configuration (SETTINGS, CASES) and its parts, kept apart from the entry point
(run_case.py), the checks of the check (validate_check.py) and the figures (recovery_figures.py):
  - the truth's solid (Solid): an ellipsoid; the ellipsoid united with a right circular cone (case C); the same with
    an apical trunk and two basal stems (case D). Every part is symmetric about the same depth z0, so a column's depth
    interval is the longest of the parts' intervals, [z0 - H, z0 + H];
  - the check renderer (render_columns): the column intervals through Block 45's computation -- slabs
    (geometry.slab_centres), partition absorption (geometry.slab_absorbance, transmitted_before), the normalised sampled
    Gaussian's transfer (soma_render.sampled_gaussian_transfer), the per-(slab, plane) accumulation on a zero-padded
    FFT box sized by Block 4's rule, pixel integration -- with the planes processed in groups to bound the memory;
  - the soma's membrane area by marching cubes (skimage.measure.marching_cubes, mesh_surface_area);
  - the simulated stack (per-plane background with a smooth drift, Gaussian noise, rounding, clipping), the fit's crop,
    planes and start, the call of fit_soma, and the errors against the truth.

A check, not pipeline code (scientific-coding s.6); it changes nothing in src/. Pure ASCII.
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from scipy import fft as sfft

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(os.path.dirname(HERE))           # "Passive Features/Diameter Re-measurement"
sys.path.insert(0, os.path.join(WS, "src"))

from allen_diameter.analysis import soma_fit as SF                              # noqa: E402
from allen_diameter.analysis.soma_view import plane_range, segment_mask       # noqa: E402
from allen_diameter.config import (DiameterConfig, RendererConfig, SomaFitConfig,  # noqa: E402
                                   SomaViewConfig)
from allen_diameter.model import geometry                                      # noqa: E402
from allen_diameter.model.kernel import sigma_r                                # noqa: E402
from allen_diameter.model.render import block_fine_grid, pixel_integrate       # noqa: E402
from allen_diameter.model.soma_render import (SomaRenderer, ellipsoid_columns,  # noqa: E402
                                              sampled_gaussian_transfer)


# ---------------------------------------------------------------- settings --

@dataclass(frozen=True)
class Settings:
    """The check's settings (the brief of D-216.2; the choices it left to the check marked "check's choice")."""

    # the pilot's acquisition (AcquisitionConfig) and the soma's place
    p_um: float = 0.1144                      # pixel (um)
    dz_um: float = 0.28                       # plane spacing (um, stage units; plane k at depth k dz)
    k_centre: int = 117                       # the true centre on plane 117 (depth 32.76 um)
    x0_um: float = 503.317                    # check's choice: the true centre in the image frame, off the pixel grid
    y0_um: float = 287.641
    light_direction: int = 1                  # acquisition.light_direction
    # the ellipsoid of cases A-D
    a_um: float = 6.5
    b_um: float = 5.0
    phi_rad: float = 0.6
    mu_per_um: float = 0.15
    # the kernel: the pilot's cleaning configuration, the linear continuation
    kernel_table_delta_um: Tuple[float, ...] = (0.0, 0.28, 0.56, 0.84)
    kernel_table_sigma_um: Tuple[float, ...] = (0.08, 0.15360355675214307, 0.3924328247895506, 0.5620226934149679)
    sigma_r0_um: float = 0.08
    slope_cfg: float = 0.79                   # the configured far slope (um of SD per um of defocus); slope0 of the fit
    # the truth's discretisation (finer than the fit's f = 1, dzeta 0.14 um)
    truth_fine_factor: int = 2
    truth_dzeta_um: float = 0.07
    truth_acc_mb: float = 450.0               # check's choice: memory bound of a truth render's accumulators (MB)
    # the camera
    B_gl: float = 110.0                       # background (grey levels)
    drift_rel: float = 0.05                   # B_k = B (1 + 0.05 sin(1.5 pi t_k)), t_k in [0, 1] across the planes
    noise_sd_gl: float = 3.0
    seed: int = 20261122                      # the noise's seed (one data set per case; common to all cases)
    # the start, as the pipeline takes it from Allen's trace
    start_offset_um: Tuple[float, float, float] = (0.8, -0.6, 1.5)
    start_radius_factor: float = 0.9          # traced radius = 0.9 r_eq(truth)
    # the crop: the fit region's bounding box with a margin (Block 46's data flow)
    crop_margin_um: float = 1.0               # check's choice
    # case C: the cone (right circular; axis in the image plane at depth z0 along phi + pi/2)
    cone_apex_um: float = 11.0                # apex 11 um from the centre
    # case D: the stems (one absorption: the soma's mu)
    trunk_radius_um: float = 1.0
    trunk_length_um: float = 15.0
    basal_radius_um: float = 0.6
    basal_length_um: float = 12.0
    basal_angles_deg: Tuple[float, float] = (125.0, -130.0)  # check's choice: from the apical direction (in-plane)
    stem_inset_um: float = 1.0                # check's choice: the basal cylinders start 1 um inside the ellipsoid
    # the soma's area by marching cubes
    mc_spacing_um: float = 0.05
    # the figure's planes: offsets from the true z0 (um)
    figure_offsets_um: Tuple[float, ...] = (-8.0, -3.0, 0.0, 3.0, 8.0)

    @property
    def z0_um(self):
        return self.k_centre * self.dz_um


SETTINGS = Settings()

# case -> (shape, c, the truth's far slope, far_slope of the fit)
CASES = {
    "A_c4.5": dict(shape="ellipsoid", c_um=4.5, slope_true=0.79, far_slope="fit"),
    "B_fit": dict(shape="ellipsoid", c_um=4.5, slope_true=0.95, far_slope="fit"),
    "B_fixed": dict(shape="ellipsoid", c_um=4.5, slope_true=0.95, far_slope="fixed"),
    "C": dict(shape="pyramidal", c_um=4.5, slope_true=0.79, far_slope="fit"),
    "D": dict(shape="pyramidal_stems", c_um=4.5, slope_true=0.79, far_slope="fit"),
    "A_c3.0": dict(shape="ellipsoid", c_um=3.0, slope_true=0.79, far_slope="fit"),
    "A_c6.0": dict(shape="ellipsoid", c_um=6.0, slope_true=0.79, far_slope="fit"),
}
CASE_ORDER = ("A_c4.5", "B_fit", "B_fixed", "C", "D", "A_c3.0", "A_c6.0")


def renderer_config(s: Settings = SETTINGS) -> RendererConfig:
    """The renderer configuration of the pilot's cleaning: its kernel table, sigma_r0 0.08 um, the linear
    continuation at the configured slope; everything else at the defaults (fft_wrap_sigmas 6). Checked by
    DiameterConfig's validation."""
    rcfg = dataclasses.replace(RendererConfig(), kernel_table_delta_um=tuple(s.kernel_table_delta_um),
                               kernel_table_sigma_um=tuple(s.kernel_table_sigma_um), sigma_r0_um=s.sigma_r0_um,
                               kernel_continuation="linear", kernel_continuation_slope=s.slope_cfg)
    DiameterConfig(renderer=rcfg)                    # raises ValueError if the configuration is not legal
    return rcfg


# ------------------------------------------------------------------- solid --

def _rotate(dx, dy, phi):
    """(u, v): (dx, dy) in the ellipsoid's frame, u along (cos phi, sin phi), v along (-sin phi, cos phi)."""
    cp, sp = math.cos(phi), math.sin(phi)
    return dx * cp + dy * sp, -dx * sp + dy * cp


@dataclass(frozen=True)
class Stem:
    """A cylinder of radius r in the image plane at the soma's depth z0: axis from p (x, y) um along the unit vector d
    for s in [-inset, length] um. Its footprint seen from above, for the mask, is the segment p -> p + length d."""

    p: Tuple[float, float]
    d: Tuple[float, float]
    length: float
    radius: float
    inset: float = 0.0
    name: str = ""


@dataclass(frozen=True)
class Solid:
    """The truth's soma: the ellipsoid theta_g = (x0, y0, z0, a, b, c, phi, mu) (Block 45's parameters), optionally
    united with a right circular cone (apex at the centre + cone_apex (-sin phi, cos phi), axis toward the centre,
    base the disc of radius cone_apex tan(alpha) in the plane v = 0 through the centre) and with stems; one absorption
    coefficient mu for the whole solid. Every part is symmetric about z0."""

    theta_g: Tuple[float, ...]
    cone_apex_um: Optional[float] = None
    cone_tan_alpha: Optional[float] = None
    stems: Tuple[Stem, ...] = ()

    @property
    def x0(self):
        return self.theta_g[0]

    @property
    def y0(self):
        return self.theta_g[1]

    @property
    def z0(self):
        return self.theta_g[2]

    @property
    def mu(self):
        return self.theta_g[7]

    @property
    def half_height(self):
        """The solid's largest column half-depth (um), analytically: c for the ellipsoid; the cone's largest radius,
        cone_apex tan(alpha), at its base; the stems' radii."""
        h = [self.theta_g[5]]
        if self.cone_apex_um is not None:
            h.append(self.cone_apex_um * self.cone_tan_alpha)
        h += [st.radius for st in self.stems]
        return float(max(h))

    def cone_half_depth(self, X, Y):
        """(..) um: the cone's column half-depth sqrt(R(t)^2 - u^2), R(t) = t tan(alpha), t = apex - v in [0, apex];
        0 outside."""
        x0, y0, _z0, _a, _b, _c, phi, _mu = self.theta_g
        u, v = _rotate(np.asarray(X, float) - x0, np.asarray(Y, float) - y0, phi)
        L, ta = float(self.cone_apex_um), float(self.cone_tan_alpha)
        t = L - v
        R = t * ta
        ok = (t >= 0.0) & (t <= L) & (np.abs(u) < R)
        return np.where(ok, np.sqrt(np.maximum(R * R - u * u, 0.0)), 0.0)

    @staticmethod
    def stem_half_depth(X, Y, st: Stem):
        """(..) um: the cylinder's column half-depth sqrt(r^2 - w^2) for s in [-inset, length], |w| < r; 0 outside."""
        px, py = st.p
        dx, dy = st.d
        ex, ey = np.asarray(X, float) - px, np.asarray(Y, float) - py
        s = ex * dx + ey * dy
        w = -ex * dy + ey * dx
        ok = (s >= -st.inset) & (s <= st.length) & (np.abs(w) < st.radius)
        return np.where(ok, np.sqrt(np.maximum(st.radius ** 2 - w * w, 0.0)), 0.0)

    def other_half_depth(self, X, Y, stems=True):
        """The largest half-depth of the parts other than the ellipsoid (the cone, and the stems if stems), or None."""
        H = None
        if self.cone_apex_um is not None:
            H = self.cone_half_depth(X, Y)
        if stems:
            for st in self.stems:
                h = self.stem_half_depth(X, Y, st)
                H = h if H is None else np.maximum(H, h)
        return H

    def columns(self, X, Y, stems=True):
        """-> (z_lo, z_hi, inside) on the columns (X, Y) um: the ellipsoid's interval from Block 45's
        ellipsoid_columns, widened to [z0 - H, z0 + H] where another part's half-depth H is larger (the union of
        intervals centred on one depth is the longest). For the plain ellipsoid this returns ellipsoid_columns'
        arrays unchanged (min(z_lo, z0 - 0) = z_lo, exactly)."""
        z_lo, z_hi, inside = ellipsoid_columns(X, Y, np.asarray(self.theta_g, float))
        H = self.other_half_depth(X, Y, stems=stems)
        if H is not None:
            z_lo = np.minimum(z_lo, self.z0 - H)
            z_hi = np.maximum(z_hi, self.z0 + H)
            inside = inside | (H > 0.0)
        return z_lo, z_hi, inside

    def half_depth(self, X, Y, stems=True):
        """(..) um: the column half-depth H of the solid (0 outside)."""
        z_lo, z_hi, _inside = self.columns(X, Y, stems=stems)
        return 0.5 * (z_hi - z_lo)

    def level(self, U, V, Zeta):
        """The soma's (ellipsoid and cone, no stems) level function on points (U, V, Zeta) of its own frame (um;
        Zeta = z - z0), positive inside, about a signed distance near the surface: max(F_E, F_C) with
        F_E = min(a, b, c) (1 - sqrt((u/a)^2 + (v/b)^2 + (zeta/c)^2)) and
        F_C = min((t tan(alpha) - sqrt(u^2 + zeta^2)) cos(alpha), t, apex - t), t = apex - v."""
        _x0, _y0, _z0, a, b, c, _phi, _mu = self.theta_g
        F = min(a, b, c) * (1.0 - np.sqrt((U / a) ** 2 + (V / b) ** 2 + (Zeta / c) ** 2))
        if self.cone_apex_um is not None:
            L, ta = float(self.cone_apex_um), float(self.cone_tan_alpha)
            t = L - V
            ca = 1.0 / math.sqrt(1.0 + ta * ta)
            Fc = np.minimum(np.minimum((t * ta - np.sqrt(U * U + Zeta * Zeta)) * ca, t), L - t)
            F = np.maximum(F, Fc)
        return F

    def soma_extent_uv(self):
        """(u_lo, u_hi, v_lo, v_hi) um: the soma's (ellipsoid and cone) bounding box in its own frame."""
        _x0, _y0, _z0, a, b, _c, _phi, _mu = self.theta_g
        v_hi = b if self.cone_apex_um is None else max(b, float(self.cone_apex_um))
        return -a, a, -b, v_hi

    def footprint_box_xy(self, stems=True):
        """(x_lo, x_hi, y_lo, y_hi) um: a bounding box of the solid's footprint in the image frame."""
        x0, y0, _z0, a, b, _c, phi, _mu = self.theta_g
        cp, sp = math.cos(phi), math.sin(phi)
        ex = math.sqrt((a * cp) ** 2 + (b * sp) ** 2)
        ey = math.sqrt((a * sp) ** 2 + (b * cp) ** 2)
        xs, ys = [x0 - ex, x0 + ex], [y0 - ey, y0 + ey]
        if self.cone_apex_um is not None:
            xs.append(x0 - sp * self.cone_apex_um)
            ys.append(y0 + cp * self.cone_apex_um)
        if stems:
            for st in self.stems:
                for s_ in (-st.inset, st.length):
                    px, py = st.p[0] + s_ * st.d[0], st.p[1] + s_ * st.d[1]
                    xs += [px - st.radius, px + st.radius]
                    ys += [py - st.radius, py + st.radius]
        return min(xs), max(xs), min(ys), max(ys)


def make_solid(case: dict, s: Settings = SETTINGS) -> Solid:
    """The truth's solid of a case. Cone (C, D): tan(alpha) = c / apex, so that the base -- the disc of radius c in
    the ellipsoid's central section perpendicular to the cone's axis, its widest section -- touches the ellipsoid's
    surface at that section's top and bottom (z0 +- c) and lies inside it elsewhere: the cone meets the ellipsoid at
    its widest section, and the union's depth profile stays continuous. Stems (D): the trunk from the apex outward
    along the cone's axis; each basal stem radial from the centre, starting on the ellipsoid's in-plane outline at
    depth z0 (plus a hidden 1 um inset), at the configured angles from the apical direction."""
    c = float(case["c_um"])
    th = (s.x0_um, s.y0_um, s.z0_um, s.a_um, s.b_um, c, s.phi_rad, s.mu_per_um)
    if case["shape"] == "ellipsoid":
        return Solid(theta_g=th)
    L = s.cone_apex_um
    ta = c / L
    if case["shape"] == "pyramidal":
        return Solid(theta_g=th, cone_apex_um=L, cone_tan_alpha=ta)
    if case["shape"] != "pyramidal_stems":
        raise ValueError("unknown shape %r" % (case["shape"],))
    phi = s.phi_rad
    e = (-math.sin(phi), math.cos(phi))                               # the cone's axis, apex side (phi + pi/2)
    apex = (s.x0_um + L * e[0], s.y0_um + L * e[1])
    stems = [Stem(p=apex, d=e, length=s.trunk_length_um, radius=s.trunk_radius_um, inset=0.0, name="apical trunk")]
    for ang in s.basal_angles_deg:
        b_ = math.radians(ang)
        d = (math.cos(b_) * e[0] - math.sin(b_) * e[1], math.sin(b_) * e[0] + math.cos(b_) * e[1])
        du, dv = _rotate(d[0], d[1], phi)
        r_out = 1.0 / math.sqrt((du / s.a_um) ** 2 + (dv / s.b_um) ** 2)  # the ellipse's radius along d
        p = (s.x0_um + r_out * d[0], s.y0_um + r_out * d[1])
        stems.append(Stem(p=p, d=d, length=s.basal_length_um, radius=s.basal_radius_um, inset=s.stem_inset_um,
                          name="basal %+.0f deg" % ang))
    return Solid(theta_g=th, cone_apex_um=L, cone_tan_alpha=ta, stems=tuple(stems))


def stem_mask(solid: Solid, shape, left, top, p_um, widen_um):
    """(H, W) bool: the stems' footprints (segment p -> p + length d, radius r) widened by widen_um, by Block 44's
    segment_mask -- the mask the pipeline builds from Allen's trace (SomaFitConfig.mask_widen_um)."""
    if not solid.stems:
        return np.zeros(tuple(shape), dtype=bool)
    p1 = np.array([st.p for st in solid.stems], float)
    p2 = np.array([(st.p[0] + st.length * st.d[0], st.p[1] + st.length * st.d[1]) for st in solid.stems], float)
    r = np.array([st.radius + widen_um for st in solid.stems], float)
    return segment_mask(shape, left, top, p_um, p1, p2, r, r)


# ------------------------------------------------------------ the renderer --

def _trim(wy, wx):
    """The rows (fftfreq order: the head [0, ra) and the tail [rb, n)) and the column count cx where the per-axis
    transfers are not exactly zero (their products elsewhere are exactly zero; Block 45's skipping, rewritten)."""
    n = wy.size
    half = (n + 1) // 2
    pos = np.flatnonzero(wy[:half])
    neg = np.flatnonzero(wy[half:])
    ra = int(pos[-1]) + 1 if pos.size else 0
    rb = half + int(neg[0]) if neg.size else n
    if ra >= rb:
        rows = (slice(0, n),)
    else:
        rows = tuple(r for r in (slice(0, ra), slice(rb, n)) if r.stop > r.start)
    nzx = np.flatnonzero(wx)
    cx = int(nzx[-1]) + 1 if nzx.size else 0
    return rows, cx


def box_shape(ny, nx, s_max, h, wrap):
    """Block 4's rule, per axis: next_fast_len(n + ceil(wrap s_max / h))."""
    pad = int(math.ceil(wrap * s_max / h))
    return sfft.next_fast_len(ny + pad), sfft.next_fast_len(nx + pad)


def render_columns(z_lo, z_hi, z0, half_height, h, f, z_planes, rcfg, dzeta, mu, light, slope=None,
                   acc_mb=None, log=None):
    """The check renderer: -> tau (K, ny/f, nx/f) for the column intervals (z_lo, z_hi) (ny, nx) on samples h apart
    (f per pixel side), by Block 45's computation (SPEC Block 45; model/soma_render.py):
        slabs zeta_j = geometry.slab_centres(z0 - half_height, z0 + half_height, dzeta);
        Delta A_j = geometry.transmitted_before(.., light) (1 - exp(-geometry.slab_absorbance(..)));
        tau_k = pixel_integrate(1 - sum_j g_{s_jk} (*) Delta A_j, f), s_jk = sigma_r(zeta_j - z_k; slope),
    g_s the normalised sampled Gaussian applied by its transfer (sampled_gaussian_transfer) on a zero-padded box of
    next_fast_len(n + ceil(fft_wrap_sigmas s_max / h)) per axis. The planes are processed in groups whose
    accumulators stay below acc_mb (None: one group, the box of all the planes -- SomaRenderer's); each group's box
    follows its own s_max (Block 4's rule holds per group).
    Custom: SomaRenderer renders ellipsoids only (its columns come from its parameters); this is its computation
    rewritten on given column intervals. Checked against SomaRenderer on the plain ellipsoid (validate_check.py)."""
    rc = rcfg if slope is None else dataclasses.replace(rcfg, kernel_continuation_slope=float(slope))
    z = np.atleast_1d(np.asarray(z_planes, dtype=float))
    K = z.size
    ny, nx = z_lo.shape
    zeta = geometry.slab_centres(z0 - half_height, z0 + half_height, dzeta)
    sig_all = sigma_r(zeta[:, None] - z[None, :], rc)                     # (J, K) um
    wrap = float(rcfg.fft_wrap_sigmas)
    if acc_mb is None:
        groups = [np.arange(K)]
    else:
        Ny, Nx = box_shape(ny, nx, float(sig_all.max()), h, wrap)
        per_plane_mb = Ny * (Nx // 2 + 1) * 16 / 2.0 ** 20
        G = max(1, int(acc_mb // per_plane_mb))
        groups = [np.arange(i, min(K, i + G)) for i in range(0, K, G)]
    t0 = time.perf_counter()
    out = np.empty((K, ny // f, nx // f))
    for gi, grp in enumerate(groups):
        sig = sig_all[:, grp]
        shape = box_shape(ny, nx, float(sig.max()), h, wrap)
        nu_y = sfft.fftfreq(shape[0], d=h)
        nu_x = sfft.rfftfreq(shape[1], d=h)
        acc = np.zeros((grp.size, shape[0], shape[1] // 2 + 1), dtype=complex)
        transfers = {}
        for j in range(zeta.size):
            # the slab's partition absorption on the samples (recomputed per group: J arrays would not fit)
            zj = float(zeta[j])
            a_j = geometry.slab_absorbance(z_lo, z_hi, zj, dzeta, mu)
            dA = geometry.transmitted_before(z_lo, z_hi, zj, dzeta, mu, light) * (-np.expm1(-a_j))
            if not dA.any():
                continue
            spec = sfft.rfft2(dA, s=shape)
            for g in range(grp.size):
                sv = float(sig[j, g])
                tr = transfers.get(sv)
                if tr is None:
                    wy = sampled_gaussian_transfer(sv, nu_y, h)
                    wx = sampled_gaussian_transfer(sv, nu_x, h)
                    tr = transfers[sv] = (wy, wx) + _trim(wy, wx)
                wy, wx, rows, cx = tr
                for rs in rows:
                    t = spec[rs, :cx] * wy[rs, None]
                    t *= wx[None, :cx]
                    acc[g, rs, :cx] += t
        for g, k in enumerate(grp):
            plane = 1.0 - sfft.irfft2(acc[g], s=shape)[:ny, :nx]
            out[k] = pixel_integrate(plane, f)
        del acc
        if log is not None:
            log("    check renderer: group %d/%d (%d planes, box %s) done, %.0f s"
                % (gi + 1, len(groups), grp.size, shape, time.perf_counter() - t0))
    return out


def render_solid(solid: Solid, crop, p_um, z_planes, rcfg, f, dzeta, light, slope=None, acc_mb=None, log=None):
    """tau (K, height, width) of a solid on a crop (left, top, width, height) px, by the check renderer: the solid's
    column intervals on render.block_fine_grid's samples (no padding), then render_columns."""
    left, top, width, height = (int(v) for v in crop)
    grid, _inner = block_fine_grid(left, top, width, height, p_um, f)
    X, Y = grid.mesh()
    z_lo, z_hi, _inside = solid.columns(X, Y)
    return render_columns(z_lo, z_hi, solid.z0, solid.half_height, float(grid.h), f, z_planes, rcfg, dzeta,
                          solid.mu, light, slope=slope, acc_mb=acc_mb, log=log)


def render_ellipsoid_grouped(theta_g, crop, p_um, z_planes, rcfg, scfg, light, slope=None, acc_mb=None, log=None):
    """tau (K, height, width) by SomaRenderer itself, the planes in groups whose accumulators stay below acc_mb
    (one SomaRenderer per group: each group's box follows its own widest kernel, Block 45's rule)."""
    z = np.atleast_1d(np.asarray(z_planes, dtype=float))
    K = z.size
    left, top, width, height = (int(v) for v in crop)
    f = int(scfg.fine_factor)
    if acc_mb is None:
        groups = [np.arange(K)]
    else:
        x0, y0, z0, a, b, c, phi, mu = (float(t) for t in theta_g)
        rc = rcfg if slope is None else dataclasses.replace(rcfg, kernel_continuation_slope=float(slope))
        zeta = geometry.slab_centres(z0 - c, z0 + c, float(scfg.dzeta_um))
        s_max = float(sigma_r(zeta[:, None] - z[None, :], rc).max())
        Ny, Nx = box_shape(height * f, width * f, s_max, p_um / f, float(rcfg.fft_wrap_sigmas))
        G = max(1, int(acc_mb // (Ny * (Nx // 2 + 1) * 16 / 2.0 ** 20)))
        groups = [np.arange(i, min(K, i + G)) for i in range(0, K, G)]
    out = np.empty((K, height, width))
    t0 = time.perf_counter()
    for gi, grp in enumerate(groups):
        rend = SomaRenderer(crop, p_um, z[grp], rcfg, scfg, light_direction=light)
        out[grp] = rend.render(np.asarray(theta_g, float), slope=slope)
        if log is not None:
            log("    SomaRenderer: group %d/%d (%d planes) done, %.0f s"
                % (gi + 1, len(groups), grp.size, time.perf_counter() - t0))
    return out


# ------------------------------------------------------------------ area --

def mesh_area(solid: Solid, spacing_um, chunk=64):
    """(area um^2, n_faces): the soma's (ellipsoid and cone; no stems) membrane area as the area of the marching-cubes
    mesh (skimage.measure.marching_cubes at level 0, mesh_surface_area) of Solid.level on a grid of spacing_um in the
    soma's frame, two cells beyond its bounding box. The level function is filled in chunks of u to bound the
    temporaries."""
    from skimage.measure import marching_cubes, mesh_surface_area
    h = float(spacing_um)
    u_lo, u_hi, v_lo, v_hi = solid.soma_extent_uv()
    c = solid.theta_g[5]
    zh = max(c, solid.half_height)
    us = np.arange(u_lo - 2 * h, u_hi + 2.5 * h, h)
    vs = np.arange(v_lo - 2 * h, v_hi + 2.5 * h, h)
    zs = np.arange(-zh - 2 * h, zh + 2.5 * h, h)
    F = np.empty((us.size, vs.size, zs.size), dtype=np.float64)
    V, Z = np.meshgrid(vs, zs, indexing="ij")
    for i in range(0, us.size, chunk):
        U = us[i:i + chunk, None, None]
        F[i:i + chunk] = solid.level(U, V[None], Z[None])
    verts, faces, _n, _v = marching_cubes(F, level=0.0, spacing=(h, h, h))
    return float(mesh_surface_area(verts, faces)), int(faces.shape[0])


def soma_reference(solid: Solid, s: Settings = SETTINGS):
    """The truth's reference values: c_ref the soma's half-height (the union's for C and D), S_ref its membrane area
    (ellipsoid_area for the ellipsoid; the marching-cubes mesh at s.mc_spacing_um for the union), r_eq_ref, and the
    ellipsoid's a, b, phi."""
    x0, y0, z0, a, b, c, phi, mu = solid.theta_g
    out = dict(a_um=a, b_um=b, c_ellipsoid_um=c, phi_rad=phi, mu_per_um=mu, x0_um=x0, y0_um=y0, z0_um=z0)
    out["S_ellipsoid_um2"] = SF.ellipsoid_area(a, b, c)
    if solid.cone_apex_um is None:
        out["c_ref_um"] = c
        out["S_ref_um2"] = out["S_ellipsoid_um2"]
        out["S_ref_method"] = "ellipsoid_area (Legendre's form)"
    else:
        # the soma's half-height: the largest column half-depth of the ellipsoid and the cone (stems are not soma)
        out["c_ref_um"] = max(c, solid.cone_apex_um * solid.cone_tan_alpha)
        area, nf = mesh_area(solid, s.mc_spacing_um)
        out["S_ref_um2"] = area
        out["S_ref_method"] = "marching cubes, spacing %.3f um (%d faces)" % (s.mc_spacing_um, nf)
        out["cone_apex_um"] = solid.cone_apex_um
        out["cone_tan_alpha"] = solid.cone_tan_alpha
        out["cone_half_angle_deg"] = math.degrees(math.atan(solid.cone_tan_alpha))
    out["r_eq_ref_um"] = math.sqrt(out["S_ref_um2"] / (4.0 * math.pi))
    return out


# ------------------------------------------------------------------ stack --

def traced_start(solid: Solid, ref: dict, s: Settings = SETTINGS):
    """The start as the pipeline takes it from Allen's trace: the traced centre = the truth's centre + the offset,
    the traced radius = 0.9 r_eq of the truth (the union's for C and D)."""
    cx = solid.x0 + s.start_offset_um[0]
    cy = solid.y0 + s.start_offset_um[1]
    cz = solid.z0 + s.start_offset_um[2]
    return np.array([cx, cy, cz]), s.start_radius_factor * ref["r_eq_ref_um"]


def fit_crop(centre_xy, scfg: SomaFitConfig, s: Settings = SETTINGS):
    """(left, top, width, height) px: the bounding box of the fit's region (the region_radius_um disc about the
    traced centre) with s.crop_margin_um on each side, in full-resolution pixels (pixel j centred on j p)."""
    half = float(scfg.region_radius_um) + s.crop_margin_um
    left = int(math.floor((centre_xy[0] - half) / s.p_um))
    right = int(math.ceil((centre_xy[0] + half) / s.p_um))
    top = int(math.floor((centre_xy[1] - half) / s.p_um))
    bottom = int(math.ceil((centre_xy[1] + half) / s.p_um))
    return left, top, right - left + 1, bottom - top + 1


def box_holding(crop, box_xy, p_um, margin_um=1.0):
    """The crop (left, top, width, height) px holding the crop and the box (x_lo, x_hi, y_lo, y_hi) um plus a margin."""
    left, top, width, height = crop
    x_lo = min(left * p_um, box_xy[0] - margin_um)
    x_hi = max((left + width - 1) * p_um, box_xy[1] + margin_um)
    y_lo = min(top * p_um, box_xy[2] - margin_um)
    y_hi = max((top + height - 1) * p_um, box_xy[3] + margin_um)
    L, R = int(math.floor(x_lo / p_um)), int(math.ceil(x_hi / p_um))
    T, B = int(math.floor(y_lo / p_um)), int(math.ceil(y_hi / p_um))
    return L, T, R - L + 1, B - T + 1


def stack_planes(centre_z, radius, scfg: SomaFitConfig, s: Settings = SETTINGS):
    """(ks_window, ks): the planes of Block 44's window (|k dz - z_s| <= r_s + SomaViewConfig.z_margin_um) and the
    planes select_planes takes from it (all recorded). The stack is rendered on ks: the fit uses no other plane."""
    k_lo, k_hi = plane_range(centre_z, radius + SomaViewConfig().z_margin_um, s.dz_um)
    ks_window = np.arange(k_lo, k_hi + 1)
    idx = SF.select_planes(ks_window, np.ones(ks_window.size, bool), centre_z, s.dz_um, scfg.plane_half_um,
                           scfg.plane_step)
    return ks_window, ks_window[idx]


def simulate_stack(tau, ks, s: Settings = SETTINGS, seed=None):
    """(data (K, H, W) uint8, B_k (K,) gl): each plane multiplied by B_k = B (1 + drift sin(1.5 pi t_k)),
    t_k = (k - k_first) / (k_last - k_first), Gaussian noise of SD noise_sd added (numpy default_rng(seed)), rounded
    to integers and clipped to [0, 255]."""
    rng = np.random.default_rng(s.seed if seed is None else seed)
    k = np.asarray(ks, dtype=float)
    t = (k - k[0]) / (k[-1] - k[0])
    B_k = s.B_gl * (1.0 + s.drift_rel * np.sin(1.5 * math.pi * t))
    I = B_k[:, None, None] * tau + s.noise_sd_gl * rng.standard_normal(tau.shape)
    return np.clip(np.rint(I), 0, 255).astype(np.uint8), B_k


def region_of(shape, crop, centre_xy, scfg, mask):
    """The fit's region (Block 46): the pixels whose centre lies within region_radius_um of the traced centre and whose
    mask is False."""
    left, top, width, height = crop
    x = (left + np.arange(width)) * SETTINGS.p_um
    y = (top + np.arange(height)) * SETTINGS.p_um
    disk = (x[None, :] - centre_xy[0]) ** 2 + (y[:, None] - centre_xy[1]) ** 2 <= float(scfg.region_radius_um) ** 2
    return disk & ~mask


# ----------------------------------------------------------------- errors --

def canonical_axes(theta, se, phi_true):
    """(a, b, phi, se_a, se_b, swapped): the fitted ellipse labelled like the truth -- (a, b, phi) and
    (b, a, phi + pi/2) are one ellipsoid, and phi is defined modulo pi; the labelling whose phi is nearer the truth's
    modulo pi is kept; phi returned within pi/2 of the truth."""
    a, b, phi = float(theta[3]), float(theta[4]), float(theta[6])
    sa, sb = float(se[3]), float(se[4])

    def dmod(x):
        return (x - phi_true + 0.5 * math.pi) % math.pi - 0.5 * math.pi

    d0 = dmod(phi)
    d1 = dmod(phi + 0.5 * math.pi)
    if abs(d1) < abs(d0):
        return b, a, phi_true + d1, sb, sa, True
    return a, b, phi_true + d0, sa, sb, False


def error_entry(value, truth, se):
    err = float(value) - float(truth)
    ok_se = se is not None and math.isfinite(se) and se > 0
    return dict(fitted=float(value), truth=float(truth), err=err, err_pct=100.0 * err / float(truth),
                se=float(se) if se is not None else None, err_se=(err / se) if ok_se else None,
                within_2se=bool(abs(err) <= 2.0 * se) if ok_se else None)


def fit_errors(fit, ref, slope_true):
    """The errors of the fit against the truth: c (against c_ref), a, b, S (against S_ref), r_eq, and the other
    parameters, in their units, % and SE; r_eq's SE from SE(S) by the delta method, SE(S) / (8 pi r_eq)."""
    a, b, phi, sa, sb, swapped = canonical_axes(fit.theta, fit.se, ref["phi_rad"])
    se_req = fit.se_area_um2 / (8.0 * math.pi * fit.r_eq_um) if math.isfinite(fit.se_area_um2) else float("nan")
    e = dict(
        c=error_entry(fit.theta[5], ref["c_ref_um"], fit.se[5]),
        a=error_entry(a, ref["a_um"], sa),
        b=error_entry(b, ref["b_um"], sb),
        S=error_entry(fit.area_um2, ref["S_ref_um2"], fit.se_area_um2),
        r_eq=error_entry(fit.r_eq_um, ref["r_eq_ref_um"], se_req),
        x0=error_entry(fit.theta[0], ref["x0_um"], fit.se[0]),
        y0=error_entry(fit.theta[1], ref["y0_um"], fit.se[1]),
        z0=error_entry(fit.theta[2], ref["z0_um"], fit.se[2]),
        phi=error_entry(phi, ref["phi_rad"], fit.se[6]),
        mu=error_entry(fit.theta[7], ref["mu_per_um"], fit.se[7]),
        slope=error_entry(fit.theta[8], slope_true, fit.se[8] if math.isfinite(fit.se[8]) else None),
    )
    e["axes_swapped"] = swapped
    return e


# ------------------------------------------------------------------- misc --

def jsonable(x):
    """x with numpy types turned into Python ones and non-finite floats into None (strict JSON)."""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if math.isfinite(v) else None
    return x


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(jsonable(obj), fh, indent=1, allow_nan=False)
        fh.write("\n")
    os.replace(tmp, path)


def environment():
    import scipy
    import skimage
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                skimage=skimage.__version__, machine=platform.machine(),
                omp_num_threads=os.environ.get("OMP_NUM_THREADS"))


def git_head():
    try:
        import subprocess
        return subprocess.check_output(["git", "-C", HERE, "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:                                   # noqa: BLE001 -- a label only
        return None
