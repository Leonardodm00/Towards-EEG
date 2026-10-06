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
            output sample); planes are processed in depth-ordered chunks
            whose accumulators fit max_fft_accumulator_mb. Wider pairs (far
            defocus, steep or long tubes): the sampled object summed against
            the continuous Gaussian on coarse output grids, one per octave of
            sigma above the split (spacing <= the octave's smallest sigma /
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

ABSORPTIONS_RENDERED = ("partition_vertical", "linear")   # "ray_world": model/ray_world.py (Block 9)
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
    tau: np.ndarray                 # (n_planes, ...) transmittance I_k / B per plane: (ny, nx) on the grid, or reduced
    zeta: np.ndarray                # (J,) slab centres, um
    n_slabs_used: int               # slabs with some absorbed light on the grid (distinct)
    sigma_max_um: float             # largest kernel width used (0 when nothing was drawn)
    fft_shape: Optional[Tuple[int, int]]  # padded FFT shape of the near-field path, else None
    backend: str
    n_near_pairs: int               # (slab, plane) pairs through the FFT (all of them for "direct")
    n_far_pairs: int                # (slab, plane) pairs through the far-field path
    far_shape: Optional[Tuple[int, int]]  # largest coarse output grid of the far-field path, else None


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


def _check_absorption(absorption, allow_ray_world=False):
    if absorption == "ray_world" and allow_ray_world:
        return
    if absorption == "ray_world":
        raise ValueError("absorbed fractions are not defined in the ray world (Block 9)")
    if absorption not in ABSORPTIONS_RENDERED:
        raise ValueError("unknown absorption %r" % (absorption,))


def _slabs(z_lo, z_hi, inside, zeta, dzeta, mu, light_direction, absorption, subset=None):
    """Yield (j, row slice, column slice, Delta A_j on that box) for every slab
    (of subset, when given) with absorbed light on the grid."""
    for j in (range(len(zeta)) if subset is None else subset):
        j = int(j)
        zj = zeta[j]
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


def render_transmittance(tube, mu, z_planes, grid, rcfg, light_direction=+1, backend=None, reduce=None):
    """tau_k = I_k / B for every plane depth in z_planes (procedure Eq. 6).

    tube: geometry.Tube; mu (1/um) >= 0; z_planes: depths (um, stage units);
    grid: FineGrid; rcfg: RendererConfig (kernel, absorption, dzeta_um,
    fft_wrap_sigmas, fft_split_sigma_um, far_grid_per_sigma,
    max_fft_accumulator_mb, direct_truncate); light_direction +1 or -1;
    backend "fft" or "direct" (default rcfg.backend); reduce: an optional
    function applied to each finished plane (ny, nx) -- e.g. crop and
    pixel-integrate -- so that only reduced planes are kept (memory).
    RenderResult.tau stacks the (reduced) planes in the order of z_planes.
    The grid must resolve the narrowest kernel, h <= sigma_min / 2 (spectral
    truncation below exp(-(pi^2/2) 4) = 3e-9); ValueError otherwise.
    """
    backend = rcfg.backend if backend is None else backend
    if backend not in BACKENDS:
        raise ValueError("unknown backend %r" % (backend,))
    _check_absorption(rcfg.absorption, allow_ray_world=True)
    mu = float(mu)
    if not (math.isfinite(mu) and mu >= 0):
        raise ValueError("mu must be finite and >= 0")
    if light_direction not in (+1, -1):
        raise ValueError("light_direction must be +1 or -1")
    z_planes = np.atleast_1d(np.asarray(z_planes, dtype=float))
    if z_planes.ndim != 1 or not np.all(np.isfinite(z_planes)):
        raise ValueError("z_planes must be a 1-D array of finite depths")
    reduce = (lambda plane: plane) if reduce is None else reduce
    if rcfg.absorption == "ray_world":          # the independent generator of Block 9 (no slabs, no kernel)
        from . import ray_world
        return RenderResult(ray_world.render_planes(tube, mu, z_planes, grid, rcfg, reduce), np.empty(0), 0, 0.0,
                            None, "ray_world", 0, 0, None)
    n_planes, ny, nx, h = z_planes.size, grid.ny, grid.nx, grid.h
    dzeta = float(rcfg.dzeta_um)

    X, Y = grid.mesh()
    z_lo, z_hi, inside = geometry.column_interval(X, Y, tube)
    del X, Y
    if mu == 0 or not inside.any():
        one = reduce(np.ones((ny, nx)))
        return RenderResult(np.stack([one] * n_planes), np.empty(0), 0, 0.0, None, backend, 0, 0, None)
    zeta = geometry.slab_centres(float(z_lo[inside].min()), float(z_hi[inside].max()), dzeta)
    sig = sigma_r(zeta[:, None] - z_planes[None, :], rcfg)          # (J, n_planes), um
    sigma_max = float(sig.max())
    if float(sig.min()) < 2.0 * h:
        raise ValueError("grid step h = %g um does not resolve the narrowest kernel (sigma = %g um): need h <= sigma / 2"
                         % (h, float(sig.min())))
    args = (z_lo, z_hi, inside, zeta, dzeta, mu, light_direction, rcfg.absorption)
    work = np.zeros((ny, nx))
    used = set()
    out = [None] * n_planes

    if backend == "direct":
        for k in range(n_planes):
            total = np.zeros((ny, nx))
            for j, rs, cs, dA in _slabs(*args):
                used.add(j)
                work[rs, cs] = dA
                total += ndimage.gaussian_filter(work, sig[j, k] / h, mode="constant", cval=0.0,
                                                 truncate=rcfg.direct_truncate)
                work[rs, cs] = 0.0
            out[k] = reduce(1.0 - total)
        return RenderResult(np.stack(out), zeta, len(used), sigma_max, None, backend, int(sig.size), 0, None)

    near = sig <= rcfg.fft_split_sigma_um               # (J, n_planes): FFT path
    far = ~near                                         # coarse-output path
    xs, ys = grid.xs, grid.ys
    far_shape = None
    levels = {}
    if far.any():
        # Custom: separable sum of the sampled object against the continuous Gaussian on coarse
        # output grids (the result is smooth at scale sigma), one grid per octave of sigma above
        # the split (spacing <= the octave's smallest sigma / far_grid_per_sigma), one pass.
        if nx < 4 or ny < 4:
            raise ValueError("the far-field path needs a grid of at least 4 x 4 samples")
        octave = np.where(far, np.floor(np.log2(np.maximum(sig, 1e-300) / rcfg.fft_split_sigma_um)), -1).astype(int)
        for L in np.unique(octave[far]):
            spacing = rcfg.fft_split_sigma_um * 2.0 ** int(L) / rcfg.far_grid_per_sigma
            xc = np.linspace(grid.x0, grid.x0 + (nx - 1) * h, max(4, int(math.ceil((nx - 1) * h / spacing)) + 1))
            yc = np.linspace(grid.y0, grid.y0 + (ny - 1) * h, max(4, int(math.ceil((ny - 1) * h / spacing)) + 1))
            levels[int(L)] = (xc, yc, np.zeros((n_planes, yc.size, xc.size)))
        far_shape = max((v[1].size, v[0].size) for v in levels.values())
        for j, rs, cs, dA in _slabs(*args, subset=np.flatnonzero(far.any(axis=1))):
            used.add(j)
            for k in np.flatnonzero(far[j]):
                xc, yc, acc_L = levels[int(octave[j, k])]
                sk = sig[j, k]
                gy = (h / (sk * _SQRT_2PI)) * np.exp(-0.5 * ((yc[:, None] - ys[rs][None, :]) / sk) ** 2)
                gx = (h / (sk * _SQRT_2PI)) * np.exp(-0.5 * ((xc[:, None] - xs[cs][None, :]) / sk) ** 2)
                acc_L[k] += gy @ dA @ gx.T

    def finish(k, tau_k):
        for L, (xc, yc, acc_L) in levels.items():
            if np.any(octave[:, k] == L):
                tau_k -= RectBivariateSpline(yc, xc, acc_L[k], kx=3, ky=3, s=0)(ys, xs)
        out[k] = reduce(tau_k)

    shape = None
    if not near.any():
        for k in range(n_planes):
            finish(k, np.ones((ny, nx)))
    else:
        pad = int(math.ceil(rcfg.fft_wrap_sigmas * float(sig[near].max()) / h))
        shape = (sfft.next_fast_len(ny + pad, real=True), sfft.next_fast_len(nx + pad, real=True))
        fy2 = 2.0 * math.pi ** 2 * sfft.fftfreq(shape[0], d=h) ** 2
        fx2 = 2.0 * math.pi ** 2 * sfft.rfftfreq(shape[1], d=h) ** 2
        spec_mb = shape[0] * (shape[1] // 2 + 1) * 16 / 2.0 ** 20
        per_chunk = max(1, int(rcfg.max_fft_accumulator_mb // spec_mb))
        order = np.argsort(z_planes, kind="stable")     # depth order: a slab is near few, adjacent planes
        for c0 in range(0, n_planes, per_chunk):
            ks = order[c0:c0 + per_chunk]
            acc = np.zeros((ks.size, shape[0], shape[1] // 2 + 1), dtype=complex)
            for j, rs, cs, dA in _slabs(*args, subset=np.flatnonzero(near[:, ks].any(axis=1))):
                used.add(j)
                work[rs, cs] = dA
                spec = sfft.rfft2(work, s=shape)
                work[rs, cs] = 0.0
                for ci, k in enumerate(ks):
                    if near[j, k]:
                        s2 = sig[j, k] ** 2
                        acc[ci] += (spec * np.exp(-s2 * fy2)[:, None]) * np.exp(-s2 * fx2)[None, :]
            for ci, k in enumerate(ks):
                finish(k, 1.0 - sfft.irfft2(acc[ci], s=shape)[:ny, :nx])
            del acc
    return RenderResult(np.stack(out), zeta, len(used), sigma_max, shape, backend, int(near.sum()), int(far.sum()),
                        far_shape)


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
    res = render_transmittance(tube, mu, ks * acq.dz_um, grid, rcfg, acq.light_direction, backend,
                               reduce=lambda plane: pixel_integrate(plane[inner[0], inner[1]], f))
    block = camera.camera_chain(res.tau, rcfg, rng, qtables)
    return block, ks, np.ones(ks.size, dtype=bool), CropFrame(int(left), int(top), 0, float(acq.res0_um))
