"""The soma's forward model -- Block 45 in specs/SPEC.md (D-215, D-216, D-218, D-219).

An ellipsoid soma E(theta_g), theta_g = (x0, y0, z0, a, b, c, phi, mu), rendered with the partition absorption
(procedure Eq. 5: Delta A_j = T_<j (1 - exp(-a_j)) per column and slab, model.geometry) and the pipeline's kernel
(model.kernel.sigma_r; the continuation slope replaceable by a parameter, D-218):

    tau_k = pixel_integrate(1 - sum_j g_{s_jk} (*) Delta A_j, f),   s_jk = sigma_r(zeta_j - z_k; gamma)

on the crop's samples (render.block_fine_grid, f = soma_fit.fine_factor, no padding: the part of the ellipsoid
outside the crop is not rendered), the slabs zeta_j = geometry.slab_centres(z0 - c, z0 + c, soma_fit.dzeta_um),
g_s the normalised sampled Gaussian (a positive kernel of sum 1; D-219.2), applied by FFT on a zero-padded box of
next_fast_len(n + ceil(renderer.fft_wrap_sigmas s_max / h)) samples per axis (Block 4's rule) with its transfer by
Poisson summation (sampled_gaussian_transfer).
Image frame: pixel (row, col) of a crop with corner (left, top) is centred on ((left + col) p, (top + row) p) um,
p = res0; depth in stage units; light toward +z when light_direction = +1. See SPEC Block 45 for the full
statement, the parameters and the oracles (tests/smoke/test_smoke_soma_render.py).

Library calls: scipy.fft (rfft2, irfft2, next_fast_len, fftfreq, rfftfreq); model.render.block_fine_grid,
pixel_integrate; model.geometry.slab_centres, slab_absorbance, transmitted_before; model.kernel.sigma_r.
Custom code (marked below): the ellipsoid's column interval, the sampled Gaussian's transfer, the accumulation.

No file access, no plotting. Pure ASCII, LF, for the cluster.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
from scipy import fft as sfft

from . import geometry
from .kernel import sigma_r
from .render import block_fine_grid, pixel_integrate

_THETA_NAMES = ("x0_um", "y0_um", "z0_um", "a_um", "b_um", "c_um", "phi_rad", "mu_per_um")


# ------------------------------------------------------------------ checks --

def _integer(x, name):
    """x as an int when it is a finite whole number (an int, or a float with an integral value; not a bool)."""
    if isinstance(x, (bool, np.bool_)):
        raise ValueError("%s must be an integer, got %r" % (name, x))
    if isinstance(x, (int, np.integer)):
        return int(x)
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise ValueError("%s must be an integer, got %r" % (name, x)) from None
    if not (math.isfinite(v) and v == math.floor(v)):
        raise ValueError("%s must be an integer, got %r" % (name, x))
    return int(v)


def _light(light_direction):
    """light_direction as the int +1 or -1 (a scalar number of that value; not a bool or a string)."""
    bad = isinstance(light_direction, (bool, np.bool_, str, bytes)) or np.ndim(light_direction) != 0
    try:
        v = float(light_direction) if not bad else 0.0
    except (TypeError, ValueError):
        v = 0.0
    if bad or v not in (1.0, -1.0):
        raise ValueError("light_direction must be +1 or -1, got %r" % (light_direction,))
    return int(v)


def _positive(x, name):
    """x as a float when it is one finite number > 0."""
    try:
        v = np.asarray(x, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("%s must be a finite number > 0, got %r" % (name, x)) from None
    if v.ndim != 0 or not (math.isfinite(float(v)) and float(v) > 0.0):
        raise ValueError("%s must be a finite number > 0, got %r" % (name, x))
    return float(v)


def _theta(theta_g):
    """theta_g (8,) as eight floats (x0, y0, z0, a, b, c, phi, mu), checked as SPEC Block 45 states."""
    try:
        th = np.asarray(theta_g, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("theta_g must be 8 numbers %s" % (_THETA_NAMES,)) from None
    if th.shape != (8,):
        raise ValueError("theta_g must have shape (8,) %s, got shape %s" % (_THETA_NAMES, th.shape))
    x0, y0, z0, a, b, c, phi, mu = (float(t) for t in th)
    if not all(math.isfinite(t) for t in (x0, y0, z0, phi)):
        raise ValueError("theta_g: x0, y0, z0 and phi must be finite, got %r" % ((x0, y0, z0, phi),))
    if not all(math.isfinite(t) and t > 0.0 for t in (a, b, c)):
        raise ValueError("theta_g: the semi-axes a, b, c must be finite and > 0 (um), got %r" % ((a, b, c),))
    if not (math.isfinite(mu) and mu >= 0.0):
        raise ValueError("theta_g: mu must be finite and >= 0 (1/um), got %r" % (mu,))
    return x0, y0, z0, a, b, c, phi, mu


# ---------------------------------------------------------------- geometry --

def ellipsoid_columns(x_um, y_um, theta_g):
    """-> (z_lo, z_hi, inside): for columns at (x_um, y_um) (arrays, broadcast together; um, image frame) the depth
    interval [z0 - c sqrt(q), z0 + c sqrt(q)] (um, stage units) where q = 1 - (u/a)^2 - (v/b)^2 > 0 (inside, bool),
    u = (x - x0) cos(phi) + (y - y0) sin(phi), v = -(x - x0) sin(phi) + (y - y0) cos(phi); outside
    z_lo = z_hi = z0. theta_g (8,): x0, y0, z0, a, b, c (um), phi (rad), mu (1/um). ValueError for a, b or c not
    finite and > 0, mu not finite and >= 0, x0, y0, z0 or phi not finite, theta_g not of shape (8,)."""
    x0, y0, z0, a, b, c, phi, _mu = _theta(theta_g)
    x, y = np.broadcast_arrays(np.asarray(x_um, dtype=float), np.asarray(y_um, dtype=float))
    dx, dy = x - x0, y - y0
    cp, sp = math.cos(phi), math.sin(phi)
    # Custom: no library gives an ellipsoid's column interval. Checked: geometry.column_interval (takes a Tube).
    # q = 1 - A u^2 - B v^2 (A = 1/a^2, B = 1/b^2) is evaluated as 1 - min(A, B) rho^2 - |A - B| w^2 with
    # rho^2 = (x - x0)^2 + (y - y0)^2 and w = u (a <= b) or v (a > b): the same number, both subtracted terms
    # non-negative (no cancellation between them), and for a = b the rotation drops out exactly (SPEC T1).
    inv_a2, inv_b2 = 1.0 / (a * a), 1.0 / (b * b)
    rho2 = dx * dx + dy * dy
    if inv_a2 >= inv_b2:                      # a <= b: the extra curvature lies along u
        w = dx * cp + dy * sp
        q = 1.0 - inv_b2 * rho2 - (inv_a2 - inv_b2) * (w * w)
    else:                                     # a > b: along v
        w = -dx * sp + dy * cp
        q = 1.0 - inv_a2 * rho2 - (inv_b2 - inv_a2) * (w * w)
    inside = q > 0.0
    half = c * np.sqrt(np.where(inside, q, 0.0))
    z_lo = np.where(inside, z0 - half, z0)
    z_hi = np.where(inside, z0 + half, z0)
    return z_lo, z_hi, inside


# ------------------------------------------------------------------ kernel --

def sampled_gaussian_transfer(s_um, nu_per_um, h_um):
    """-> the transfer (DFT) of the normalised sampled Gaussian of SD s_um on samples h_um apart, at the frequencies
    nu_per_um (cycles/um; any shape), dimensionless, nu's shape:

        w_s(nu) = sum_{m=-M..M} exp(-2 pi^2 s^2 (nu - m/h)^2) / sum_{m=-M..M} exp(-2 pi^2 s^2 m^2 / h^2),
        M = max(1, ceil(1.2 h / s))   (the aliases left out below 1e-12).

    The kernel w_m = exp(-m^2 h^2 / (2 s^2)) / sum_m' exp(-m'^2 h^2 / (2 s^2)) is positive with sum 1, so
    w_s(0) = 1. ValueError unless s_um and h_um are (scalar) finite and > 0."""
    s = _positive(s_um, "s_um")
    h = _positive(h_um, "h_um")
    nu = np.asarray(nu_per_um, dtype=float)
    # Custom: scipy.ndimage.fourier_gaussian multiplies by the continuous transfer exp(-2 pi^2 s^2 nu^2), whose kernel
    # on the sample grid has negative lobes below about one sample (D-219.2; checks/soma_fit_2026-11-22/
    # check_transfer.py); the normalised sampled Gaussian's DFT is its Poisson sum, written out here.
    # Checked: scipy.ndimage.fourier_gaussian, scipy.ndimage.gaussian_filter (real space, truncated).
    M = max(1, int(math.ceil(1.2 * h / s)))
    mh = np.arange(-M, M + 1) / h                     # the aliases' frequencies m / h (cycles/um)
    cst = -2.0 * math.pi ** 2 * s * s                 # um^2
    num = np.exp(cst * (nu[..., None] - mh) ** 2).sum(axis=-1)
    den = np.exp(cst * mh ** 2).sum()                 # the same terms at nu = 0, so w_s(0) = 1
    return np.asarray(num / den)                      # an ndarray of nu's shape (0-d for a scalar nu)


def _trimmed_transfer(s, nu_y, nu_x, h):
    """The per-axis transfers at SD s on the box's frequencies, with the part where each is not exactly zero:
    the row slices (fftfreq order: the head [0, ra) and the tail [rb, N_y)) and the column count cx (rfftfreq)."""
    wy = sampled_gaussian_transfer(s, nu_y, h)
    wx = sampled_gaussian_transfer(s, nu_x, h)
    n = wy.size
    half = (n + 1) // 2                               # fftfreq: [0, half) non-negative, [half, n) negative
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
    return wy, wx, rows, cx


# ---------------------------------------------------------------- renderer --

class SomaRenderer:
    """The planes of an ellipsoid soma on a crop (SPEC Block 45).

    SomaRenderer(crop, res0_um, z_planes_um, rcfg, scfg, light_direction=+1): crop = (left, top, width, height),
    integers, px of the full-resolution image; res0_um the pixel size (um); z_planes_um (K,) the planes' depths (um,
    stage units); rcfg a RendererConfig (the kernel: its table and continuation; fft_wrap_sigmas); scfg the
    SomaFitConfig (fine_factor, dzeta_um); light_direction +1 or -1 (the caller passes acquisition.light_direction).
    Holds the sample grid, and the box's frequencies for every box size it has rendered on (the box follows the
    widest kernel of a render, so it is sized per render and its frequencies kept for the next).
    ValueError for an empty or non-integer crop, a pixel size not finite and > 0, empty or non-finite depths,
    light_direction not +-1, a fine_factor that is not a whole number >= 1, a slab thickness or
    rcfg.fft_wrap_sigmas not finite and > 0.
    """

    def __init__(self, crop, res0_um, z_planes_um, rcfg, scfg, light_direction=+1):
        try:
            vals = tuple(crop)
        except TypeError:
            raise ValueError("crop must be (left, top, width, height), got %r" % (crop,)) from None
        if len(vals) != 4:
            raise ValueError("crop must be (left, top, width, height), got %r" % (crop,))
        left, top, width, height = (_integer(v, n) for v, n in zip(vals, ("left", "top", "width", "height")))
        if width < 1 or height < 1:
            raise ValueError("empty crop: width and height must be >= 1, got %r" % (crop,))
        p = _positive(res0_um, "res0_um")
        try:
            z = np.atleast_1d(np.asarray(z_planes_um, dtype=float))
        except (TypeError, ValueError):
            raise ValueError("z_planes_um must be a 1-D array of finite depths (um)") from None
        if z.ndim != 1 or z.size == 0 or not np.all(np.isfinite(z)):
            raise ValueError("z_planes_um must be a non-empty 1-D array of finite depths (um)")
        light = _light(light_direction)
        f = _integer(scfg.fine_factor, "scfg.fine_factor")
        if f < 1:
            raise ValueError("scfg.fine_factor must be >= 1, got %r" % (scfg.fine_factor,))
        self._dzeta = _positive(scfg.dzeta_um, "scfg.dzeta_um")
        self._wrap = _positive(rcfg.fft_wrap_sigmas, "rcfg.fft_wrap_sigmas")
        grid, _inner = block_fine_grid(left, top, width, height, p, f)          # no padding (SPEC Block 45)
        self._X, self._Y = grid.mesh()                                           # (ny, nx) um, row = y
        self._h = float(grid.h)                                                  # p / f, um
        self._f = f
        self._shape_px = (height, width)
        self._z = z.copy()
        self._rcfg = rcfg
        self._light = light
        self._freq = {}                               # box shape (N_y, N_x) -> (fftfreq (N_y,), rfftfreq (N_x//2+1,))

    def _kernel_config(self, slope):
        """The renderer configuration whose sigma_r the render uses: rcfg, or rcfg with its continuation slope
        replaced by slope (D-218; the continuation must then be "linear")."""
        if slope is None:
            return self._rcfg
        try:
            g = float(slope)
        except (TypeError, ValueError):
            raise ValueError("slope must be a finite number >= 0 (um/um), got %r" % (slope,)) from None
        if not (math.isfinite(g) and g >= 0.0):
            raise ValueError("slope must be a finite number >= 0 (um/um), got %r" % (slope,))
        if self._rcfg.kernel_continuation != "linear":
            raise ValueError("a slope replaces the continuation's slope only for kernel_continuation 'linear', got %r"
                             % (self._rcfg.kernel_continuation,))
        return dataclasses.replace(self._rcfg, kernel_continuation_slope=g)

    def _frequencies(self, shape):
        freq = self._freq.get(shape)
        if freq is None:
            freq = (sfft.fftfreq(shape[0], d=self._h), sfft.rfftfreq(shape[1], d=self._h))
            self._freq[shape] = freq
        return freq

    def render(self, theta_g, slope=None):
        """-> tau (K, height, width) float64, dimensionless, the planes in the order of z_planes_um; tau <= 1, no
        lower bound imposed (the model does not clip). theta_g (8,) as ellipsoid_columns; slope (um of SD per um of
        defocus, finite, >= 0) replaces the kernel's continuation slope (the continuation must be "linear";
        ValueError otherwise). mu = 0 gives tau = 1 exactly."""
        th = _theta(theta_g)
        rc = self._kernel_config(slope)
        z0, c, mu = th[2], th[5], th[7]
        K = self._z.size
        H, W = self._shape_px
        if mu == 0.0:
            return np.ones((K, H, W))
        z_lo, z_hi, inside = ellipsoid_columns(self._X, self._Y, th)
        if not inside.any():                          # no sample of the crop meets the soma: nothing absorbed
            return np.ones((K, H, W))
        h, dzeta = self._h, self._dzeta
        ny, nx = self._X.shape
        zeta = geometry.slab_centres(z0 - c, z0 + c, dzeta)                     # (J,) um
        sig = sigma_r(zeta[:, None] - self._z[None, :], rc)                      # (J, K) um: s_jk
        s_max = float(sig.max())
        pad = int(math.ceil(self._wrap * s_max / h))
        shape = (sfft.next_fast_len(ny + pad), sfft.next_fast_len(nx + pad))   # Block 4's rule, per axis
        nu_y, nu_x = self._frequencies(shape)
        nfx = shape[1] // 2 + 1
        acc = np.zeros((K, shape[0], nfx), dtype=complex)                        # per plane: sum_j spec_j w_jk
        transfers = {}                                                           # s -> _trimmed_transfer(s)
        # Custom: the per-(slab, plane) accumulation, as Block 4's "fft" backend (no library renders a
        # depth-dependent blur): one rfft2 per slab, the separable transfer w_s(nu_y) w_s(nu_x) per (slab, plane),
        # one irfft2 per plane. Rows and columns where a transfer is exactly 0.0 in float64 are skipped: their
        # products are exactly zero, so the sum is the full product's.
        for j in range(zeta.size):
            zj = float(zeta[j])
            a_j = geometry.slab_absorbance(z_lo, z_hi, zj, dzeta, mu)
            dA = geometry.transmitted_before(z_lo, z_hi, zj, dzeta, mu, self._light) * (-np.expm1(-a_j))
            if not dA.any():                          # the slab absorbs nothing on the crop's samples
                continue
            spec = sfft.rfft2(dA, s=shape)            # zero-padded to the box
            for k in range(K):
                s = float(sig[j, k])
                tr = transfers.get(s)
                if tr is None:
                    tr = transfers[s] = _trimmed_transfer(s, nu_y, nu_x, h)
                wy, wx, rows, cx = tr
                for rs in rows:
                    t = spec[rs, :cx] * wy[rs, None]
                    t *= wx[None, :cx]
                    acc[k, rs, :cx] += t
        out = np.empty((K, H, W))
        for k in range(K):
            plane = 1.0 - sfft.irfft2(acc[k], s=shape)[:ny, :nx]
            out[k] = pixel_integrate(plane, self._f)
        return out
