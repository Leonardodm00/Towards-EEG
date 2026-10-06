"""Synthetic stack renderer -- Block 4 in specs/SPEC.md.

Procedure s.3.6 (Eqs. 5-6) with the exact slab absorbances of impl-handoff
(S4) and the FFT form of Eq. 6. For a straight tube (Block 2) of absorption
coefficient mu (1/um), cut into slabs of thickness dzeta at centres zeta_j,
and for each plane k at depth z_k:

    Delta A_j = T_<j (1 - exp(-a_j))                (partition_vertical, Eq. 5)
    Delta A_j = a_j                                 (linear, mathematics Eq. 11)
    tau_k = I_k / B = 1 - sum_j (Delta A_j * G_{sigma_r(zeta_j - z_k)})     (Eq. 6 / B)

on a fine grid of sample centres. The object is drawn on every sample of the
grid and nowhere else: the caller sizes the grid (Block 6 uses the block, a
padding and the tube's footprint). Depth z is in stage units; with
light_direction = +1 the light travels toward increasing z (and increasing
plane index, z_k = k dz).

Backends:
  "fft"     (slab, plane) pairs with sigma_r <= fft_split_sigma_um: one rfft2
            per slab, the Gaussian transfer function per pair, one irfft2 per
            plane, zero padding of at least fft_wrap_sigmas * sigma / h
            samples (a source's periodic images stay that far from every
            output sample). Wider pairs (far defocus, steep or long tubes):
            the sampled object summed against the continuous Gaussian on a
            coarse output grid (spacing <= smallest far sigma /
            far_grid_per_sigma), then bicubic-spline interpolated to the
            fine grid. Without the split the padding would grow with the
            widest kernel (several um for steep tubes).
  "direct"  the reference: scipy.ndimage.gaussian_filter per (slab, plane),
            mode="constant", as procedure s.3.6 step 4 writes it.

Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import fft as sfft
from scipy import ndimage
from scipy.interpolate import RectBivariateSpline

from . import camera, geometry
from .kernel import sigma_r

ABSORPTIONS_RENDERED = ("partition_vertical", "linear")   # "ray_world" is Block 9
_SQRT_2PI = math.sqrt(2.0 * math.pi)
BACKENDS = ("fft", "direct")


@dataclass(frozen=True)
class FineGrid:
    """Sample centres x_n = x0 + n h (n < nx), y_m = y0 + m h (m < ny), in um.
    Arrays on the grid have shape (ny, nx): row = y, column = x."""

    x0: float
    y0: float
    h: float
    nx: int
    ny: int

    def __post_init__(self):
        if not (math.isfinite(self.x0) and math.isfinite(self.y0) and math.isfinite(self.h) and self.h > 0):
            raise ValueError("FineGrid needs finite x0, y0 and h > 0")
        if int(self.nx) != self.nx or int(self.ny) != self.ny or self.nx < 1 or self.ny < 1:
            raise ValueError("FineGrid needs integer nx, ny >= 1")

    @property
    def xs(self):
        return self.x0 + self.h * np.arange(self.nx)

    @property
    def ys(self):
        return self.y0 + self.h * np.arange(self.ny)

    def mesh(self):
        """(X, Y), each (ny, nx)."""
        return np.meshgrid(self.xs, self.ys, indexing="xy")


@dataclass(frozen=True)
class RenderResult:
    tau: np.ndarray                 # (n_planes, ny, nx) transmittance I_k / B on the grid
    zeta: np.ndarray                # (J,) slab centres, um
    n_slabs_used: int               # slabs with some absorbed light on the grid
    sigma_max_um: float             # largest kernel width used (0 when nothing was drawn)
    fft_shape: Optional[Tuple[int, int]]  # padded FFT shape of the near-field path, else None
    backend: str
    n_near_pairs: int               # (slab, plane) pairs through the FFT (all of them for "direct")
    n_far_pairs: int                # (slab, plane) pairs through the far-field path
    far_shape: Optional[Tuple[int, int]]  # coarse output grid of the far-field path, else None


def block_fine_grid(left, top, width, height, p_x, factor, pad_px=0):
    """Fine grid of a pixel block and the slices of the block's own samples.

    Pixel (i, j) of the block -- row i, column j -- is centred at
    x = (left + j) p_x, y = (top + i) p_x (integer full-resolution coordinates
    are pixel centres) and is split into factor x factor samples; pad_px whole
    pixels are added on each side. Returns (grid, (row slice, column slice)).
    """
    f, pad = int(factor), int(pad_px)
    if f != factor or f < 1 or pad != pad_px or pad < 0 or int(width) < 1 or int(height) < 1:
        raise ValueError("block_fine_grid: need integer factor >= 1, pad_px >= 0, width, height >= 1")
    if not (p_x > 0):
        raise ValueError("p_x must be > 0")
    h = p_x / f
    grid = FineGrid(x0=(left - pad) * p_x - 0.5 * p_x + 0.5 * h,
                    y0=(top - pad) * p_x - 0.5 * p_x + 0.5 * h,
                    h=h, nx=(int(width) + 2 * pad) * f, ny=(int(height) + 2 * pad) * f)
    inner = (slice(pad * f, (pad + int(height)) * f), slice(pad * f, (pad + int(width)) * f))
    return grid, inner


def fine_factor(d, rcfg, p_x):
    """Samples per pixel side: p_x / h_g with h_g = h_g_um_thin for d <= h_g_switch_d_um,
    else h_g_um_thick (impl-handoff Findings, grid convergence)."""
    h = rcfg.h_g_um_thin if d <= rcfg.h_g_switch_d_um else rcfg.h_g_um_thick
    ratio = p_x / h
    f = int(round(ratio))
    if f < 1 or abs(ratio - f) > 1e-9 * ratio:
        raise ValueError("h_g = %r does not divide p_x = %r into whole samples" % (h, p_x))
    return f


def slab_grid(tube, grid, dzeta):
    """Slab centres covering the depth range of the tube's columns on the grid
    (empty when no column of the grid meets the tube)."""
    X, Y = grid.mesh()
    z_lo, z_hi, inside = geometry.column_interval(X, Y, tube)
    if not inside.any():
        return np.empty(0)
    return geometry.slab_centres(float(z_lo[inside].min()), float(z_hi[inside].max()), dzeta)


def _check_absorption(absorption):
    if absorption == "ray_world":
        raise NotImplementedError("absorption 'ray_world' is the Block 9 generator")
    if absorption not in ABSORPTIONS_RENDERED:
        raise ValueError("unknown absorption %r" % (absorption,))


def _slabs(z_lo, z_hi, inside, zeta, dzeta, mu, light_direction, absorption):
    """Yield (j, row slice, column slice, Delta A_j on that box) for every slab
    with absorbed light on the grid."""
    for j, zj in enumerate(zeta):
        hit = inside & (z_lo < zj + 0.5 * dzeta) & (z_hi > zj - 0.5 * dzeta)
        if not hit.any():
            continue
        rows = np.flatnonzero(hit.any(axis=1))
        cols = np.flatnonzero(hit.any(axis=0))
        rs, cs = slice(int(rows[0]), int(rows[-1]) + 1), slice(int(cols[0]), int(cols[-1]) + 1)
        zl, zh = z_lo[rs, cs], z_hi[rs, cs]
        a = geometry.slab_absorbance(zl, zh, zj, dzeta, mu)
        if absorption == "partition_vertical":
            dA = geometry.transmitted_before(zl, zh, zj, dzeta, mu, light_direction) * (-np.expm1(-a))
        else:
            dA = a
        yield j, rs, cs, dA


def absorbed_fractions(tube, mu, grid, zeta, dzeta, light_direction=+1, absorption="partition_vertical"):
    """Delta A_j on the whole grid, (J, ny, nx) -- for tests and small cases."""
    _check_absorption(absorption)
    X, Y = grid.mesh()
    z_lo, z_hi, inside = geometry.column_interval(X, Y, tube)
    out = np.zeros((len(zeta), grid.ny, grid.nx))
    for j, rs, cs, dA in _slabs(z_lo, z_hi, inside, np.asarray(zeta, dtype=float), dzeta, mu,
                                light_direction, absorption):
        out[j, rs, cs] = dA
    return out


def render_transmittance(tube, mu, z_planes, grid, rcfg, light_direction=+1, backend=None):
    """tau_k = I_k / B on the grid for every plane depth in z_planes (procedure Eq. 6).

    tube: geometry.Tube; mu (1/um) >= 0; z_planes: depths (um, stage units);
    grid: FineGrid; rcfg: RendererConfig (kernel, absorption, dzeta_um,
    fft_wrap_sigmas, fft_split_sigma_um, far_grid_per_sigma,
    direct_truncate); light_direction +1 or -1; backend "fft" or "direct"
    (default rcfg.backend). The grid must resolve the narrowest kernel,
    h <= sigma_min / 2 (spectral truncation below exp(-(pi^2/2) 4) = 3e-9);
    ValueError otherwise.
    """
    backend = rcfg.backend if backend is None else backend
    if backend not in BACKENDS:
        raise ValueError("unknown backend %r" % (backend,))
    _check_absorption(rcfg.absorption)
    mu = float(mu)
    if not (math.isfinite(mu) and mu >= 0):
        raise ValueError("mu must be finite and >= 0")
    if light_direction not in (+1, -1):
        raise ValueError("light_direction must be +1 or -1")
    z_planes = np.atleast_1d(np.asarray(z_planes, dtype=float))
    if z_planes.ndim != 1 or not np.all(np.isfinite(z_planes)):
        raise ValueError("z_planes must be a 1-D array of finite depths")
    n_planes, ny, nx, h = z_planes.size, grid.ny, grid.nx, grid.h
    dzeta = float(rcfg.dzeta_um)

    X, Y = grid.mesh()
    z_lo, z_hi, inside = geometry.column_interval(X, Y, tube)
    if mu == 0 or not inside.any():
        return RenderResult(np.ones((n_planes, ny, nx)), np.empty(0), 0, 0.0, None, backend, 0, 0, None)
    zeta = geometry.slab_centres(float(z_lo[inside].min()), float(z_hi[inside].max()), dzeta)
    sig = sigma_r(zeta[:, None] - z_planes[None, :], rcfg)          # (J, n_planes), um
    sigma_max = float(sig.max())
    if float(sig.min()) < 2.0 * h:
        raise ValueError("grid step h = %g um does not resolve the narrowest kernel (sigma = %g um): need h <= sigma / 2"
                         % (h, float(sig.min())))
    work = np.zeros((ny, nx))
    used = 0

    if backend == "fft":
        near = sig <= rcfg.fft_split_sigma_um               # (J, n_planes): FFT path
        far = ~near                                         # coarse-output path
        tau = np.ones((n_planes, ny, nx))
        shape = far_shape = None
        if near.any():
            pad = int(math.ceil(rcfg.fft_wrap_sigmas * float(sig[near].max()) / h))
            shape = (sfft.next_fast_len(ny + pad, real=True), sfft.next_fast_len(nx + pad, real=True))
            fy2 = 2.0 * math.pi ** 2 * sfft.fftfreq(shape[0], d=h) ** 2
            fx2 = 2.0 * math.pi ** 2 * sfft.rfftfreq(shape[1], d=h) ** 2
            acc = np.zeros((n_planes, shape[0], shape[1] // 2 + 1), dtype=complex)
        if far.any():
            if nx < 4 or ny < 4:
                raise ValueError("the far-field path needs a grid of at least 4 x 4 samples")
            spacing = float(sig[far].min()) / rcfg.far_grid_per_sigma
            xc = np.linspace(grid.x0, grid.x0 + (nx - 1) * h, max(4, int(math.ceil((nx - 1) * h / spacing)) + 1))
            yc = np.linspace(grid.y0, grid.y0 + (ny - 1) * h, max(4, int(math.ceil((ny - 1) * h / spacing)) + 1))
            far_shape = (yc.size, xc.size)
            far_acc = np.zeros((n_planes,) + far_shape)
        xs, ys = grid.xs, grid.ys
        for j, rs, cs, dA in _slabs(z_lo, z_hi, inside, zeta, dzeta, mu, light_direction, rcfg.absorption):
            used += 1
            near_k, far_k = np.flatnonzero(near[j]), np.flatnonzero(far[j])
            if near_k.size:
                work[rs, cs] = dA
                spec = sfft.rfft2(work, s=shape)
                work[rs, cs] = 0.0
                for k in near_k:
                    s2 = sig[j, k] ** 2
                    acc[k] += (spec * np.exp(-s2 * fy2)[:, None]) * np.exp(-s2 * fx2)[None, :]
            if far_k.size:
                # Custom: separable sum of the sampled object against the continuous Gaussian,
                # evaluated on a coarse output grid (the result is smooth at scale sigma).
                dy = yc[:, None] - ys[rs][None, :]
                dx = xc[:, None] - xs[cs][None, :]
                for k in far_k:
                    sk = sig[j, k]
                    gy = (h / (sk * _SQRT_2PI)) * np.exp(-0.5 * (dy / sk) ** 2)
                    gx = (h / (sk * _SQRT_2PI)) * np.exp(-0.5 * (dx / sk) ** 2)
                    far_acc[k] += gy @ dA @ gx.T
        for k in range(n_planes):
            if shape is not None:
                tau[k] -= sfft.irfft2(acc[k], s=shape)[:ny, :nx]
            if far_shape is not None and far[:, k].any():
                tau[k] -= RectBivariateSpline(yc, xc, far_acc[k], kx=3, ky=3, s=0)(ys, xs)
        return RenderResult(tau, zeta, used, sigma_max, shape, backend, int(near.sum()), int(far.sum()), far_shape)

    total = np.zeros((n_planes, ny, nx))
    for j, rs, cs, dA in _slabs(z_lo, z_hi, inside, zeta, dzeta, mu, light_direction, rcfg.absorption):
        used += 1
        work[rs, cs] = dA
        for k in range(n_planes):
            total[k] += ndimage.gaussian_filter(work, sig[j, k] / h, mode="constant", cval=0.0,
                                                truncate=rcfg.direct_truncate)
        work[rs, cs] = 0.0
    return RenderResult(1.0 - total, zeta, used, sigma_max, None, backend, int(sig.size), 0, None)


def pixel_integrate(fine, factor):
    """Mean over factor x factor samples: (..., ny, nx) -> (..., ny/factor, nx/factor)."""
    a = np.asarray(fine, dtype=float)
    f = int(factor)
    if f != factor or f < 1 or a.ndim < 2 or a.shape[-1] % f or a.shape[-2] % f:
        raise ValueError("pixel_integrate: the last two axes must be multiples of an integer factor >= 1")
    ny, nx = a.shape[-2], a.shape[-1]
    return a.reshape(a.shape[:-2] + (ny // f, f, nx // f, f)).mean(axis=(-3, -1))


def synthetic_block(tube, mu, ks, left, top, width, height, cfg, rng, pad_px, qtables=None, backend=None):
    """A synthetic image block in the contract of allen_image_io.fetch_zblock.

    Renders the planes ks (plane indices; depth z_k = k * dz_um) on the fine
    grid of the block padded by pad_px pixels, crops the block, pixel-integrates
    and runs the camera chain. cfg: DiameterConfig; rng: numpy.random.Generator
    (camera noise). Returns (block uint8 (n_planes, height, width), ks, valid
    (all True), frame CropFrame(left, top, 0, res0_um)).
    """
    from allen_image_io import CropFrame   # the contract's frame type (flat module in src/)

    acq, rcfg = cfg.acquisition, cfg.renderer
    ks = np.asarray(ks, dtype=int).reshape(-1)
    f = fine_factor(tube.d, rcfg, acq.res0_um)
    grid, inner = block_fine_grid(left, top, width, height, acq.res0_um, f, pad_px)
    res = render_transmittance(tube, mu, ks * acq.dz_um, grid, rcfg, acq.light_direction, backend)
    tau_px = pixel_integrate(res.tau[:, inner[0], inner[1]], f)
    block = camera.camera_chain(tau_px, rcfg, rng, qtables)
    return block, ks, np.ones(ks.size, dtype=bool), CropFrame(int(left), int(top), 0, float(acq.res0_um))
