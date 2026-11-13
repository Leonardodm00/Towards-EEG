"""Several tubes rendered in one stack (Eqs. 5-6 applied jointly) -- Block 24 in specs/SPEC.md (wave 1; D-129, D-130.O11).

Section 2.3 of SPEC.md with the tubes t = 1 .. T of absorption coefficients
mu_t (1/um) in one slab lattice: slab centres zeta_j (um, stage units) tile
the union of the tubes' depth ranges on the grid (Block 2 slab_centres, slab
thickness dzeta_um). For slab j and column (x, y), with a_{t,j}(x, y) tube t's
slab absorbance (S4) and a_j = sum_t a_{t,j}:

    linear              Delta A_j = a_j
    partition_vertical  Delta A_j = T_<j (1 - exp(-a_j)),
                        T_<j = exp(-sum_{j' < j} a_{j'}), slabs numbered in
                        the light's direction
    hybrid              Delta A_j = sum_t a_{t,j} h(x, y, zbar_{t,j}),
                        h(p) = < exp(-sum_t' mu_t' l_back,t'(p; s)) / s_z >_W

with l_back,t' Block 21's back length through tube t' (hybrid.back_lengths)
and zbar_{t,j} the midpoint of tube t's part of the column inside slab j.
The planes are (5-6): tau_k = 1 - sum_j Delta A_j * G_{sigma_r(zeta_j - z_k)}.
Tubes are assumed not to intersect in space (a column may cross several).

T_<j is evaluated as the product over tubes of Block 2's transmitted_before
(the length of each tube's column before the slab): equal to
exp(-sum_{j' < j} a_{j'}) because the lattice starts below (light +1), resp.
ends above (light -1), every column of every tube; with one tube this is
Block 4's own expression, so one tube renders as render.render_transmittance.

Tubes with mu = 0 are dropped before the lattice is built: they absorb
nothing and leave the lattice, hence every plane, unchanged.

The plane loop (backends "direct" and "fft", the FFT split and the far-field
path, the grid-resolution check) is Block 4's render_transmittance applied to
the scene's slabs; Block 4 exposes no blur of given slabs, so it is written
here again, operation for operation (one tube: render_transmittance to
roundoff). The hybrid's slabs are computed once per call and kept as their
boxes' nonzero entries, as Block 21 does.

Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import math

import numpy as np
from scipy import fft as sfft
from scipy import ndimage
from scipy.interpolate import RectBivariateSpline

from . import camera, geometry
from . import hybrid as _hybrid
from .kernel import sigma_r
from .render import (ABSORPTIONS_RENDERED, BACKENDS, RenderResult, block_fine_grid, fine_factor,
                     pixel_integrate)

_SQRT_2PI = math.sqrt(2.0 * math.pi)
_CHUNK_ENTRIES = 1 << 20      # points x directions per chunk of the joint history (as Block 21)


# ------------------------------------------------------------ validation ---

def _check_scene(tubes, mus):
    """(tubes list, mus float64 array) after the contract's checks."""
    tubes = list(tubes)
    mus = np.asarray(list(mus) if not isinstance(mus, np.ndarray) else mus, dtype=float).reshape(-1)
    if len(tubes) == 0:
        raise ValueError("render_scene: the scene is empty (no tubes)")
    if mus.size != len(tubes):
        raise ValueError("render_scene: %d tubes but %d absorption coefficients" % (len(tubes), mus.size))
    if not np.all(np.isfinite(mus)) or np.any(mus < 0):
        raise ValueError("render_scene: every mu must be finite and >= 0 (1/um)")
    return tubes, mus


def _active(tubes, mus):
    """The tubes that absorb (mu > 0) and their coefficients."""
    keep = [i for i in range(len(tubes)) if mus[i] > 0]
    return [tubes[i] for i in keep], [float(mus[i]) for i in keep]


# ------------------------------------------------------------- the slabs ---

def _columns(tubes, grid):
    """Per tube: (z_lo, z_hi, inside) on the grid (Block 2 column_interval), (ny, nx) each."""
    X, Y = grid.mesh()
    return [geometry.column_interval(X, Y, t) for t in tubes]


def _lattice(cols, dzeta):
    """Slab centres over the union of the tubes' depth ranges on the grid (empty when no column meets a tube)."""
    lo, hi = [], []
    for z_lo, z_hi, inside in cols:
        if inside.any():
            lo.append(float(z_lo[inside].min()))
            hi.append(float(z_hi[inside].max()))
    if not lo:
        return np.empty(0)
    return geometry.slab_centres(min(lo), max(hi), dzeta)


def _joint_history(points, tubes, mus, S, W, light_direction):
    """h(p) = < exp(-sum_t mu_t l_back,t(p; s)) / s_z >_W at points (n, 3) um, (n,).
    Every mu_t > 0 (a tube with mu = 0 never enters, so 0 * inf never occurs)."""
    P = np.asarray(points, dtype=float).reshape(-1, 3)
    w_path = W / S[:, 2]                                 # weight x path per unit depth
    out = np.empty(P.shape[0])
    step = max(1, _CHUNK_ENTRIES // S.shape[0])
    for a in range(0, P.shape[0], step):
        chunk = P[a:a + step]
        acc = None
        for tube, mu in zip(tubes, mus):
            ell = _hybrid.back_lengths(chunk, tube, S, light_direction)
            ell *= mu
            acc = ell if acc is None else acc + ell
        acc *= -1.0
        out[a:a + step] = np.exp(acc, out=acc) @ w_path
    return out


def _scene_slabs(cols, tubes, mus, xs, ys, zeta, dzeta, light_direction, absorption, subset=None, dirs=None):
    """Yield (j, row slice, column slice, Delta A_j on that box) for every slab (of
    subset, when given) with absorbed light on the grid; the box bounds the union of
    the tubes' hits. dirs = (S, W) for the hybrid."""
    if absorption == "hybrid" and dirs is None:
        raise ValueError("the hybrid absorption needs the condenser's directions (S, W)")
    ranges = []
    for z_lo, z_hi, inside in cols:
        ranges.append((float(z_lo[inside].min()), float(z_hi[inside].max())) if inside.any() else None)
    for j in (range(len(zeta)) if subset is None else subset):
        j = int(j)
        zj = zeta[j]
        hit = None
        for (z_lo, z_hi, inside), rg in zip(cols, ranges):
            if rg is None or rg[0] >= zj + 0.5 * dzeta or rg[1] <= zj - 0.5 * dzeta:
                continue
            h_t = inside & (z_lo < zj + 0.5 * dzeta) & (z_hi > zj - 0.5 * dzeta)
            hit = h_t if hit is None else (hit | h_t)
        if hit is None or not hit.any():
            continue
        rows = np.flatnonzero(hit.any(axis=1))
        cs_ = np.flatnonzero(hit.any(axis=0))
        rs, cs = slice(int(rows[0]), int(rows[-1]) + 1), slice(int(cs_[0]), int(cs_[-1]) + 1)
        a = None
        parts = []
        for (z_lo, z_hi, inside), mu in zip(cols, mus):
            zl, zh = z_lo[rs, cs], z_hi[rs, cs]
            a_t = geometry.slab_absorbance(zl, zh, float(zj), float(dzeta), float(mu))
            parts.append((zl, zh, a_t))
            a = a_t if a is None else a + a_t
        if absorption == "linear":
            yield j, rs, cs, a
        elif absorption == "partition_vertical":
            T = None
            for (zl, zh, _a_t), mu in zip(parts, mus):
                T_t = geometry.transmitted_before(zl, zh, zj, dzeta, mu, light_direction)
                T = T_t if T is None else T * T_t
            yield j, rs, cs, T * (-np.expm1(-a))
        else:                                            # hybrid
            S, W = dirs
            X = np.broadcast_to(xs[cs][None, :], a.shape)
            Y = np.broadcast_to(ys[rs][:, None], a.shape)
            dA = np.zeros(a.shape)
            for zl, zh, a_t in parts:
                m = a_t > 0
                if not m.any():
                    continue
                lo = np.maximum(zl[m], zj - 0.5 * dzeta)
                hi = np.minimum(zh[m], zj + 0.5 * dzeta)
                pts = np.column_stack([X[m], Y[m], 0.5 * (lo + hi)])
                dA[m] += a_t[m] * _joint_history(pts, tubes, mus, S, W, light_direction)
            yield j, rs, cs, dA


class _SlabStore:
    """Every slab's Delta A_j computed once, kept as its box and the box's nonzero
    entries, served again for any subset in increasing j (the hybrid's slabs cost
    far more than the partition's, and the plane loop asks for them repeatedly)."""

    def __init__(self, slabs):
        self._items = {}
        for j, rs, cs, dA in slabs:
            flat = dA.ravel()
            nz = np.flatnonzero(flat)
            idx = nz.astype(np.int32) if flat.size < 2 ** 31 else nz
            self._items[int(j)] = (rs, cs, dA.shape, idx, flat[nz].copy())

    def __call__(self, subset=None):
        js = sorted(self._items) if subset is None else [int(j) for j in subset if int(j) in self._items]
        for j in js:
            rs, cs, shape, idx, vals = self._items[j]
            flat = np.zeros(int(shape[0]) * int(shape[1]))
            flat[idx] = vals
            yield j, rs, cs, flat.reshape(shape)


def _directions(rcfg):
    from . import ray_world
    return ray_world.directions(rcfg.ray_n_rho, rcfg.ray_n_psi, rcfg.ray_na, rcfg.ray_n_oil)


def _absorbed_fractions(tubes, mus, grid, dzeta, light_direction=+1, absorption="partition_vertical",
                        directions=None, zeta=None):
    """(zeta (J,), Delta A_j (J, ny, nx)) of the scene on the whole grid -- for tests
    and small cases. zeta: the slab lattice (default: the scene's own, over the
    absorbing tubes' depth ranges on the grid); directions = (S, W) for "hybrid"."""
    tubes, mus = _check_scene(tubes, mus)
    if absorption not in ABSORPTIONS_RENDERED:
        raise ValueError("unknown absorption %r" % (absorption,))
    if light_direction not in (+1, -1):
        raise ValueError("light_direction must be +1 or -1")
    tubes, mus = _active(tubes, mus)
    cols = _columns(tubes, grid)
    zeta = _lattice(cols, dzeta) if zeta is None else np.asarray(zeta, dtype=float).reshape(-1)
    out = np.zeros((zeta.size, grid.ny, grid.nx))
    if not tubes:
        return zeta, out
    dirs = None
    if absorption == "hybrid":
        if directions is None:
            raise ValueError("_absorbed_fractions: the hybrid needs directions=(S, W)")
        S, W = np.asarray(directions[0], dtype=float), np.asarray(directions[1], dtype=float)
        if W.shape != (S.shape[0],) or not np.all(np.isfinite(W)) or np.any(W < 0) or abs(W.sum() - 1.0) > 1e-12:
            raise ValueError("weights W must be (N,), finite, >= 0 and sum to 1")
        dirs = (S, W)                                    # S is checked by hybrid.back_lengths
    for j, rs, cs, dA in _scene_slabs(cols, tubes, mus, grid.xs, grid.ys, zeta, dzeta, light_direction,
                                      absorption, dirs=dirs):
        out[j, rs, cs] = dA
    return zeta, out


# ------------------------------------------------------------ the planes ---

def render_scene(tubes, mus, z_planes, grid, rcfg, light_direction=+1, reduce=None):
    """tau_k = I_k / B of a scene of tubes for every plane depth in z_planes (Eqs. 5-6, jointly).

    tubes: sequence of Block 2 Tubes (um, rad; assumed not to intersect);
    mus: sequence of absorption coefficients (1/um, finite, >= 0), one per tube;
    z_planes: depths (um, stage units); grid: render.FineGrid; rcfg:
    RendererConfig (absorption in render.ABSORPTIONS_RENDERED, the kernel,
    dzeta_um, backend and its settings, and for "hybrid" ray_na, ray_n_oil,
    ray_n_rho, ray_n_psi); light_direction +1 or -1; reduce: an optional
    function applied to each finished plane (ny, nx). Returns
    render.RenderResult, tau (n_planes, ny, nx) or the reduced planes, in the
    order of z_planes. ValueError for unequal lengths, a negative or
    non-finite mu, an empty scene, an unknown absorption or backend, and a
    grid that does not resolve the narrowest kernel (h > sigma_min / 2).
    """
    tubes, mus = _check_scene(tubes, mus)
    backend = rcfg.backend
    if backend not in BACKENDS:
        raise ValueError("unknown backend %r" % (backend,))
    if rcfg.absorption not in ABSORPTIONS_RENDERED:
        raise ValueError("render_scene: absorption %r is not a rendered absorption %r"
                         % (rcfg.absorption, ABSORPTIONS_RENDERED))
    if light_direction not in (+1, -1):
        raise ValueError("light_direction must be +1 or -1")
    z_planes = np.atleast_1d(np.asarray(z_planes, dtype=float))
    if z_planes.ndim != 1 or not np.all(np.isfinite(z_planes)):
        raise ValueError("z_planes must be a 1-D array of finite depths")
    reduce = (lambda plane: plane) if reduce is None else reduce
    n_planes, ny, nx, h = z_planes.size, grid.ny, grid.nx, grid.h
    dzeta = float(rcfg.dzeta_um)

    tubes, mus = _active(tubes, mus)
    cols = _columns(tubes, grid)
    zeta = _lattice(cols, dzeta)
    if zeta.size == 0:
        one = reduce(np.ones((ny, nx)))
        return RenderResult(np.stack([one] * n_planes), np.empty(0), 0, 0.0, None, backend, 0, 0, None)
    sig = sigma_r(zeta[:, None] - z_planes[None, :], rcfg)          # (J, n_planes), um
    sigma_max = float(sig.max())
    if float(sig.min()) < 2.0 * h:
        raise ValueError("grid step h = %g um does not resolve the narrowest kernel (sigma = %g um): need h <= sigma / 2"
                         % (h, float(sig.min())))
    xs, ys = grid.xs, grid.ys
    if rcfg.absorption == "hybrid":
        S, W = _directions(rcfg)
        slabs = _SlabStore(_scene_slabs(cols, tubes, mus, xs, ys, zeta, dzeta, light_direction, "hybrid",
                                        dirs=(S, W)))
    else:
        def slabs(subset=None):
            return _scene_slabs(cols, tubes, mus, xs, ys, zeta, dzeta, light_direction, rcfg.absorption,
                                subset=subset)
    work = np.zeros((ny, nx))
    used = set()
    out = [None] * n_planes

    if backend == "direct":
        for k in range(n_planes):
            total = np.zeros((ny, nx))
            for j, rs, cs, dA in slabs():
                used.add(j)
                work[rs, cs] = dA
                total += ndimage.gaussian_filter(work, sig[j, k] / h, mode="constant", cval=0.0,
                                                 truncate=rcfg.direct_truncate)
                work[rs, cs] = 0.0
            out[k] = reduce(1.0 - total)
        return RenderResult(np.stack(out), zeta, len(used), sigma_max, None, backend, int(sig.size), 0, None)

    near = sig <= rcfg.fft_split_sigma_um               # (J, n_planes): FFT path
    far = ~near                                         # coarse-output path (Block 4's far field)
    far_shape = None
    levels = {}
    if far.any():
        if nx < 4 or ny < 4:
            raise ValueError("the far-field path needs a grid of at least 4 x 4 samples")
        octave = np.where(far, np.floor(np.log2(np.maximum(sig, 1e-300) / rcfg.fft_split_sigma_um)), -1).astype(int)
        for L in np.unique(octave[far]):
            spacing = rcfg.fft_split_sigma_um * 2.0 ** int(L) / rcfg.far_grid_per_sigma
            xc = np.linspace(grid.x0, grid.x0 + (nx - 1) * h, max(4, int(math.ceil((nx - 1) * h / spacing)) + 1))
            yc = np.linspace(grid.y0, grid.y0 + (ny - 1) * h, max(4, int(math.ceil((ny - 1) * h / spacing)) + 1))
            levels[int(L)] = (xc, yc, np.zeros((n_planes, yc.size, xc.size)))
        far_shape = max((v[1].size, v[0].size) for v in levels.values())
        for j, rs, cs, dA in slabs(subset=np.flatnonzero(far.any(axis=1))):
            used.add(j)
            for k in np.flatnonzero(far[j]):
                xc, yc, acc_L = levels[int(octave[j, k])]
                sk = sig[j, k]
                gy = (h / (sk * _SQRT_2PI)) * np.exp(-0.5 * ((yc[:, None] - ys[rs][None, :]) / sk) ** 2)
                gx = (h / (sk * _SQRT_2PI)) * np.exp(-0.5 * ((xc[:, None] - xs[cs][None, :]) / sk) ** 2)
                acc_L[k] += gy @ dA @ gx.T
    else:
        octave = None

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
            for j, rs, cs, dA in slabs(subset=np.flatnonzero(near[:, ks].any(axis=1))):
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


def synthetic_scene_block(tubes, mus, ks, left, top, width, height, cfg, rng, pad_px, qtables=None):
    """A synthetic image block of a scene of tubes in the contract of allen_image_io.fetch_zblock.

    Renders the planes ks (plane indices; depth z_k = k * dz_um) on the fine
    grid of the block padded by pad_px pixels, crops the block, pixel-integrates
    (render.pixel_integrate) and runs the camera chain (camera.camera_chain).
    The fine factor is Block 4's fine_factor at the smallest diameter among the
    absorbing tubes (among all tubes when none absorbs). cfg: DiameterConfig;
    rng: numpy.random.Generator (camera noise). Returns (block uint8
    (n_planes, height, width), ks, valid (all True), frame CropFrame(left,
    top, 0, res0_um)).
    """
    from allen_image_io import CropFrame   # the contract's frame type (flat module in src/)

    tubes, mus = _check_scene(tubes, mus)
    acq, rcfg = cfg.acquisition, cfg.renderer
    ks = np.asarray(ks, dtype=int).reshape(-1)
    active, _ = _active(tubes, mus)
    f = fine_factor(min(t.d for t in (active if active else tubes)), rcfg, acq.res0_um)
    grid, inner = block_fine_grid(left, top, width, height, acq.res0_um, f, pad_px)
    res = render_scene(tubes, mus, ks * acq.dz_um, grid, rcfg, acq.light_direction,
                       reduce=lambda plane: pixel_integrate(plane[inner[0], inner[1]], f))
    block = camera.camera_chain(res.tau, rcfg, rng, qtables)
    return block, ks, np.ones(ks.size, dtype=bool), CropFrame(int(left), int(top), 0, float(acq.res0_um))
