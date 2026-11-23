"""Smoke test for the SGT point-spread function in the fast line renderer -- Block 49 in specs/SPEC.md (wave 7;
D-221, D-222, D-224, D-225).

line_fast.line_profiles(..., kernel=line_sgt.SGTKernel(beta, xi, zeta)) is Block 35's quantity with the SGT PSF
(isotropic for zeta = 1, the half-plane two-piece stretch across y_hat otherwise), truncated at sgt_reach_sigmas law
SDs (radially, before the stretch; the mass beyond dropped). Oracles (i) to (vii) of the entry:

    test_gaussian_member    (i) SGTKernel(2, 0, 1) against kernel=None (Block 35's Gaussian path) [5e-5]
    test_known_answer       (ii) a flat tube along the pixel grid read at pixel centres against the identifiability
                            check's 1-D model (closed-form SGT CDF pixel means, untruncated), the reach raised until
                            truncated_mass <= 1e-7 [5e-5]; reported: (2, 1, 1) at the default reach
    test_reference          (iii) tilted tubes against a brute-force FFT reference written here (one grid on the fine
                            lattice padded beyond the largest support, the transfer times sinc(p nu_x) sinc(p nu_y),
                            the half-sample phase) [5e-5]; (2, 0.5, 1) and the skews at the default reach, the
                            reference truncated as the module is (the raised reach's grids: see the test)
    test_reference_raised   (iii) the skews at the raised reach on the tilt 0.3 rad geometries (long run only) [5e-5]
    test_transfer           (iv) along y_hat the untruncated transfer equals the 1-D characteristic function of
                            g_{w,kappa} by quadrature [1e-7]; zeta = 1 gives the isotropic transfer [exact]; zeta ->
                            1/zeta mirrors the profile [1e-10]; sgt_tail_mass against its closed forms [1e-8]
    test_smoothness         (v) central differences of tau in log10 beta, xi, ln zeta at steps 1e-3 and 5e-4 agree
                            within 1 % of the derivative's largest magnitude
    test_truncation         (vi) (2, 1, 1) at the default reach against the reference with the truncated bivariate t
                            [5e-5]; truncated_mass against (1 + xi r^2)^(-1/xi) [1e-8]
    test_invariants         (vii) mu = 0 gives 1 [exact]; mirror symmetry in v for zeta = 1 [1e-10]; the order of ks;
                            a cached call equals an uncached one [1e-12]
    test_contract           (vii) shape, dtype, finiteness; the constants and the API
    test_determinism        (vii) bit-identical repeat
    test_edge_cases         (vii) the ValueErrors, ExtrusionInvalid
    test_block35            (vii) Block 35's quick smoke tests pass unchanged; its Gaussian path is bit-identical to
                            the contract commit's (git show 9ea1abe) on 6 cases
    test_timing             reported: time per warm call, Gaussian and SGT (zeta = 1 and != 1), at line_fine_factor 8
                            and 32, hybrid at 8 x 16 directions; truncated_mass of the shapes tested
    test_convergence        skipped: the entry names no convergence oracle; the scheme's own discretisation (bands at
                            sgt_samples_per_fwhm, levels, Chebyshev nodes, the unit table) is checked at its settings
                            through the references above

Quick and long runs. By default the quick subset runs (about 10 min on 2 cores): every oracle on a reduced set of
geometries. Set LINE_SGT_LONG=1 for the full sets the entry names (about 2.5 h; test_reference_raised and
test_timing only in the long run). Test names given on the command line select checks; a leading '-' excludes one.

Run (from the code folder "Passive Features/Diameter Re-measurement")
    python3 tests/smoke/test_smoke_line_sgt.py
    LINE_SGT_LONG=1 python3 tests/smoke/test_smoke_line_sgt.py
    LINE_SGT_LONG=1 python3 tests/smoke/test_smoke_line_sgt.py test_timing
    python3 -m pytest tests/smoke/test_smoke_line_sgt.py -q

Debug
    A failed check_close() saves both arrays to tests/smoke/artifacts/<check name>.npz

Keep this file pure ASCII so it survives transfer to the cluster.
"""
from __future__ import annotations

import dataclasses
import importlib.metadata
import importlib.util
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
import traceback
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal  # noqa: F401
from scipy import fft as sfft
from scipy import integrate, interpolate, special

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
SRC = WS / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from allen_diameter.analysis import profiles as PR  # noqa: E402
from allen_diameter.config import default_config  # noqa: E402
from allen_diameter.model import blur_kernels as BK  # noqa: E402
from allen_diameter.model import geometry as G  # noqa: E402
from allen_diameter.model import line_fast as LF  # noqa: E402
from allen_diameter.model import line_render as LR  # noqa: E402
from allen_diameter.model import line_sgt as LS  # noqa: E402
from allen_diameter.model.kernel import sigma_r  # noqa: E402

SEED = 12345  # no randomness in the block; kept for the runner's header
REPORT_PACKAGES = ("numpy", "scipy")
LONG = os.environ.get("LINE_SGT_LONG", "") not in ("", "0")
TOL = 5e-5            # the entry's bound in tau: 0.007 of the pilot's noise in tau units (1.1 / 150)
TM_TARGET = 1e-7      # (ii), (iii): the reach is raised until truncated_mass <= this
CONTRACT = "9ea1abe"  # the contract commit (Block 35's Gaussian path there is the bit-identity reference)
SHAPES_II = ((2.0, 0.5, 1.0), (1.3, 0.0, 1.0), (4.0, 0.0, 1.0), (2.0, 0.3, 1.3), (2.0, 0.3, 0.77))
FW = LS.FWHM_PER_SD
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
_SAVED_ARTIFACTS = []  # files written by check_close during this run
REPORT = {}           # numbers gathered for the summary


# --------------------------------------------------------------- helpers ---

def make_rng(seed: int = SEED) -> np.random.Generator:
    """Fresh generator, to be passed explicitly to the code under test."""
    return np.random.default_rng(seed)


def check_close(name, actual, expected, rtol=1e-12, atol=0.0):
    """numpy.testing.assert_allclose that keeps the evidence on failure."""
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    try:
        assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=name)
    except AssertionError:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
        path = ARTIFACT_DIR / f"{safe_name}.npz"
        np.savez(path, actual=actual, expected=expected)
        _SAVED_ARTIFACTS.append(path)
        raise


def observed_order(h, err):
    """Least-squares slope p of log e(h) = log C + p * log h."""
    h = np.asarray(h, dtype=float)
    err = np.asarray(err, dtype=float)
    if np.any(err <= 0.0):
        raise ValueError("observed_order needs strictly positive errors")
    slope, _intercept = np.polyfit(np.log(h), np.log(err), 1)
    return float(slope)


def make_cfg(f=None, reach=None, n_rho=None, n_psi=None, **renderer):
    """The default configuration with inference.line_fine_factor f, sgt_reach_sigmas reach, the hybrid's directions
    n_rho x n_psi (inference.ray_*), renderer fields replaced."""
    c = default_config()
    inf = {}
    if f is not None:
        inf["line_fine_factor"] = int(f)
    if reach is not None:
        inf["sgt_reach_sigmas"] = float(reach)
    if n_rho is not None:
        inf["ray_n_rho"], inf["ray_n_psi"] = int(n_rho), int(n_psi)
    if inf:
        c = dataclasses.replace(c, inference=dataclasses.replace(c.inference, **inf))
    if renderer:
        c = dataclasses.replace(c, renderer=dataclasses.replace(c.renderer, **renderer))
    return c


def line_geometry(theta, origin):
    o = np.asarray(origin, dtype=float)
    return o, np.array([-math.sin(theta), math.cos(theta)]), np.array([math.cos(theta), math.sin(theta)])


def reach_for(shape, target=TM_TARGET):
    """The smallest reach (law SDs) on the grid 12 + 0.1 n with truncated_mass <= target (the tail mass decreases
    with the reach: doubling, then bisection on the grid)."""
    W = BK.sgt_unit_fwhm(*shape)

    def ok(n):
        return LS.sgt_tail_mass(shape[0], shape[1], (12.0 + 0.1 * n) * W / FW) <= target
    if ok(0):
        return 12.0
    lo, hi = 0, 1
    while not ok(hi):
        lo, hi = hi, 2 * hi
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if ok(mid) else (mid, hi)
    return round(12.0 + 0.1 * hi, 1)


# ------------------------------------------------- independent transfers ---

def cf_table(beta, xi, L=200.0, n=2 ** 23):
    """The untruncated unit core's characteristic function hhat(t), t >= 0, by a 1-D FFT of blur_kernels.sgt_unit_pdf
    sampled on [-L, L) (fft_line.unit_cf_table's construction; independent of the module's table): (dt, values) on
    t = k dt. Aliasing below 1e-12 for the shapes tested (dx = 2L / n = 4.8e-5)."""
    dx = 2.0 * L / n
    x = -L + dx * np.arange(n)
    g = BK.sgt_unit_pdf(x, beta, xi, 1.0)
    vals = dx * np.fft.rfft(np.fft.ifftshift(g)).real
    return 1.0 / (2.0 * L), vals


def hankel_bivariate_t(xi, rho_c, t_max=60.0, dt=0.0025, n_nodes=6000):
    """The truncated bivariate t's transfer in core units (beta = 2; P_1(r) = (1/pi)(1 + xi r^2)^-(1/xi + 1), the
    isotropic PSF whose projection is the core of (H1) at beta = 2, cut at rho_c): 2 int_0^rho_c (1 + xi r^2)^-(1/xi
    + 1) J_0(2 pi t r) r dr by Gauss-Legendre on [0, rho_c] (closed-form profile; independent of the module's Abel
    construction): (dt, values) on t = k dt."""
    rn, wn = np.polynomial.legendre.leggauss(n_nodes)
    r = 0.5 * rho_c * (rn + 1.0)
    w = 0.5 * rho_c * wn * 2.0 * (1.0 + xi * r * r) ** (-(1.0 / xi + 1.0)) * r
    t = np.arange(int(round(t_max / dt)) + 1) * dt
    out = np.empty(t.size)
    for i0 in range(0, t.size, 512):
        tt = t[i0:i0 + 512]
        out[i0:i0 + 512] = special.j0(2.0 * math.pi * np.outer(tt, r)) @ w
    return dt, out


class Table1D(object):
    """A function of t >= 0 on a uniform table, read by a C2 cubic spline; 0 beyond the table."""

    def __init__(self, dt, values):
        self.t_max = dt * (values.size - 1)
        self.spline = interpolate.CubicSpline(dt * np.arange(values.size), values)

    def __call__(self, t):
        t = np.asarray(t, dtype=float)
        out = np.zeros(t.shape)
        m = t <= self.t_max
        out[m] = self.spline(t[m])
        return out


def iso_transfer(table, W):
    """transfer(nux, nuy, sigma) of an isotropic kernel whose unit transfer is table(t): table(a |nu|),
    a = FW sigma / W."""
    def tr(nux, nuy, s):
        return table((FW * s / W) * np.hypot(nux, nuy))
    return tr


def cell_skew_transfer(xi, zeta, W, rho_c, y_hat, nsub=2):
    """make_transfer(Nx, Ny, h) for the brute force: the skewed kernel built independently of the module's (SPEC
    Block 49 (iii)): for beta = 2 the isotropic PSF is the bivariate t, P_1(r) = (1/pi) (1 + xi r^2)^-(1/xi + 1) in
    core units, cut at rho_c (the radial variable before the stretch), stretched as the entry writes (c P_1 of
    sqrt((u/zeta)^2 + s^2) for u >= 0, of sqrt((zeta u)^2 + s^2) for u < 0), cell-averaged on the grid (nsub x nsub
    midpoints per cell), transformed (rfft2) and divided by the average's own transfer, sin(pi h nu) / (nsub sin(pi h
    nu / nsub)) per axis (sinc(h nu) / sinc(h nu / nsub); 1 for nsub = 1, point samples). Accuracy on (iii)'s
    geometry (d 0.4, tilt 0.3, f 4, the default reach): at zeta = 1 against the Hankel reference (iso_t_factory)
    2.5e-9, 2.1e-9, 1.2e-9 in tau for nsub 1, 2, 3; for (2, 0.3, 1.3) and (2, 0.3, 0.77) the module differs from it by
    1.6e-8 (nsub 1) and 7.7e-8 (nsub 2, 3). (Dividing by the exact cell's sinc(h nu) instead, as a first version did,
    leaves the midpoint rule's own transfer 1 / sinc(h nu / nsub) in the reference: 7.2e-5 at nsub 2 on that geometry,
    1.8e-5 at nsub 4.)"""
    yh = np.asarray(y_hat, dtype=float)
    sh = np.array([yh[1], -yh[0]])
    c = 2.0 / (zeta + 1.0 / zeta)
    off = (np.arange(nsub) + 0.5) / nsub - 0.5

    def make(Nx, Ny, h):
        def avg(nu):                    # the nsub-point midpoint average's own transfer, sin(pi h nu) / (nsub sin(pi h nu / nsub))
            return np.sinc(h * nu) / np.sinc(h * nu / nsub)
        cell = avg(sfft.fftfreq(Ny, h))[:, None] * avg(sfft.rfftfreq(Nx, h))[None, :]

        def tr(nux, nuy, s):
            a = FW * s / W
            nr = int(math.ceil(rho_c * a * max(zeta, 1.0 / zeta) / h)) + 2
            if 2 * nr + 1 > min(Nx, Ny):
                raise AssertionError("the reference grid does not hold the kernel's support")
            loc = np.arange(-nr, nr + 1) * h
            sub = np.zeros((loc.size, loc.size))
            for oy in off:
                for ox in off:
                    X = loc[None, :] + ox * h
                    Y = loc[:, None] + oy * h
                    u = X * yh[0] + Y * yh[1]
                    w = X * sh[0] + Y * sh[1]
                    rho = np.where(u >= 0.0, np.hypot(u / zeta, w), np.hypot(zeta * u, w)) / a
                    sub += np.where(rho <= rho_c, (c / (math.pi * a * a)) * (1.0 + xi * rho * rho) ** (-(1.0 / xi + 1.0)),
                                    0.0)
            img = np.zeros((Ny, Nx))
            img[np.ix_(np.arange(-nr, nr + 1) % Ny, np.arange(-nr, nr + 1) % Nx)] = sub / (nsub * nsub)
            return sfft.rfft2(img) * (h * h) / cell
        return tr
    return make


# ---------------------------------------------------- the brute force ---

def brute_reference(tube, mu, ks, o, yh, eu, v, n_avg, step, cfg, make_transfer, support_um):
    """tau (n_planes, n_v), Block 35's quantity on one zero-padded FFT grid of the fine lattice x = (n + 1/2) h - p/2
    (h = p / f, f even), padded by more than support_um (the largest kernel support) around the union of the tube's
    whole footprint and the read's pixel box: the partition absorption of every slab (Block 2) as point masses
    h^2 Delta A_j, every (slab, plane) pair's transfer (make_transfer(Nx, Ny, h) -> transfer(nux, nuy, sigma)) times
    the pixel box's sinc(p nu_x) sinc(p nu_y) and the half-sample phase exp(i pi (nu_x + nu_y) h) that puts the
    pixel centres on the output lattice (fft_line.py's construction); the read by profiles.sample_profile. All
    samples of the tube enter (no reach rule)."""
    acq, rcfg = cfg.acquisition, cfg.renderer
    f = int(cfg.inference.line_fine_factor)
    assert f % 2 == 0
    p, dz, dzeta, light = float(acq.res0_um), float(acq.dz_um), float(rcfg.dzeta_um), acq.light_direction
    h = p / f
    o, yh, eu, v, n_avg, step = LR._line_args(o, yh, eu, v, n_avg, step)
    ks = np.asarray(ks, dtype=np.int64)
    c_lo, c_hi, r_lo, r_hi = LF._pixel_box(o, yh, eu, v, n_avg, step, p)[:4]
    ix0, iy0, xs_all, ys_all, X, Y, z_lo, z_hi, inside = LF._footprint(tube, h, p)
    zeta = G.slab_centres(float(z_lo[inside].min()), float(z_hi[inside].max()), dzeta)
    sig = sigma_r(zeta[:, None] - (ks * dz)[None, :], rcfg)
    out_cx = np.arange(c_lo, c_hi + 1) * f + f // 2 - 1
    out_ry = np.arange(r_lo, r_hi + 1) * f + f // 2 - 1
    npad = int(math.ceil(support_um / h)) + 8
    gx0 = min(ix0, int(out_cx[0])) - npad
    gx1 = max(ix0 + xs_all.size, int(out_cx[-1]) + 1) + npad
    gy0 = min(iy0, int(out_ry[0])) - npad
    gy1 = max(iy0 + ys_all.size, int(out_ry[-1]) + 1) + npad
    Nx = sfft.next_fast_len(gx1 - gx0, real=True)
    Ny = sfft.next_fast_len(gy1 - gy0)
    nux = sfft.rfftfreq(Nx, h)[None, :]
    nuy = sfft.fftfreq(Ny, h)[:, None]
    common = np.sinc(p * nux) * np.sinc(p * nuy) * np.exp(1j * math.pi * (nux + nuy) * h)
    transfer = make_transfer(Nx, Ny, h)
    out = np.empty((ks.size, v.size))
    frame = LR._Frame(c_lo, r_lo, p)
    for k in range(ks.size):                            # plane by plane: one accumulator in memory
        acc = np.zeros((Ny, Nx // 2 + 1), dtype=complex)
        for j in range(zeta.size):
            zj = float(zeta[j])
            m = inside & (z_lo < zj + 0.5 * dzeta) & (z_hi > zj - 0.5 * dzeta)
            if not m.any():
                continue
            a = G.slab_absorbance(z_lo, z_hi, zj, dzeta, mu)
            dA = np.where(m, G.transmitted_before(z_lo, z_hi, zj, dzeta, mu, light) * (-np.expm1(-a)), 0.0)
            grid = np.zeros((Ny, Nx))
            grid[iy0 - gy0:iy0 - gy0 + ys_all.size, ix0 - gx0:ix0 - gx0 + xs_all.size] = dA
            acc += sfft.rfft2(grid) * transfer(nux, nuy, float(sig[j, k]))
        field = sfft.irfft2(acc * common, s=(Ny, Nx))
        deficit = field[np.ix_(out_ry - gy0, out_cx - gx0)]
        out[k] = 1.0 - PR.sample_profile(deficit, frame, o, yh, eu, v, n_avg, step)
    return out, (Ny, Nx)


def max_support_um(tube, ks, cfg, reach, zeta=1.0):
    """The largest support radius (um) of the tube's (slab, plane) pairs: reach x max sigma x max(zeta, 1/zeta)."""
    zl, zh = G.depth_extent(tube)
    dz = cfg.acquisition.dz_um
    dmax = max(abs(zh - min(ks) * dz), abs(zl - max(ks) * dz)) + cfg.renderer.dzeta_um
    return reach * float(sigma_r(dmax, cfg.renderer)) * max(zeta, 1.0 / zeta)


def load_1d_model():
    """The identifiability check's 1-D model (checks/sgt_identifiability_2026-11-23/sgt_line_model.py)."""
    path = WS / "checks" / "sgt_identifiability_2026-11-23" / "sgt_line_model.py"
    name = "sgt_line_model_check"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod                              # registered first: its dataclasses look themselves up there
    spec.loader.exec_module(mod)
    return mod


def cf_1d(t, beta, xi, zeta):
    """The 1-D characteristic function of f( . | kappa) of (H2) at t (core units) by scipy.integrate.quad (QAWF)."""
    f = lambda x: float(BK.sgt_unit_pdf(x, beta, xi, zeta))      # noqa: E731
    g = lambda x: float(BK.sgt_unit_pdf(-x, beta, xi, zeta))     # noqa: E731
    if t == 0.0:
        return 1.0 + 0.0j
    w = 2.0 * math.pi * t
    kw = dict(limlst=500)
    re_ = integrate.quad(f, 0, np.inf, weight="cos", wvar=w, **kw)[0] + integrate.quad(g, 0, np.inf, weight="cos",
                                                                                       wvar=w, **kw)[0]
    im_ = -(integrate.quad(f, 0, np.inf, weight="sin", wvar=w, **kw)[0]
            - integrate.quad(g, 0, np.inf, weight="sin", wvar=w, **kw)[0])
    return re_ + 1j * im_


def iso_cf_factory(shape):
    """make_transfer for the brute force: the untruncated isotropic transfer hhat(a |nu|), hhat by cf_table."""
    tab = Table1D(*cf_table(shape[0], shape[1]))
    W = BK.sgt_unit_fwhm(*shape)
    return lambda Nx, Ny, h: iso_transfer(tab, W)


def iso_t_factory(shape, reach):
    """make_transfer for the brute force: the truncated bivariate t (beta = 2, zeta = 1) by its Hankel transform."""
    W = BK.sgt_unit_fwhm(*shape)
    tab = Table1D(*hankel_bivariate_t(shape[1], reach * W / FW))
    return lambda Nx, Ny, h: iso_transfer(tab, W)


def tilted_case(d, tilt, theta=0.52, U=3.5, cz=0.1):
    """(iii)'s geometry: a tube of diameter d, tilt and heading theta, half-length U, the line through its middle."""
    o, yh, eu = line_geometry(theta, (50.0, 40.0))
    tube = G.Tube((o[0] + 0.02, o[1] - 0.01, cz), 0.5 * d, tilt, theta, 1.0, U, "axial")
    return tube, o, yh, eu


def i_cases():
    """(i)'s geometries: d, tilt (rad), heading (rad), axis depth (um); the quick run takes a subset."""
    full = [(d, t, th, cz) for d in (0.3, 1.0) for t in (0.0, 0.3) for th in (0.0, 0.52) for cz in (0.0, 0.14)]
    return full if LONG else [(d, t, 0.52, 0.14) for d in (0.3, 1.0) for t in (0.0, 0.3)]


def i_tube(d, tilt, th, cz, U=8.0):
    o, yh, eu = line_geometry(th, (50.0, 40.0))
    return G.Tube((o[0] + 0.02, o[1] - 0.01, cz), 0.5 * d, tilt, th, 1.0, U, "axial"), o, yh, eu


# ---------------------------------------------------------------- checks ---

def test_gaussian_member():
    # (i) SGTKernel(2, 0, 1) is the Gaussian law's PSF (the isotropic Gaussian of SD sigma_r, FWHM_PER_SD sigma_r):
    # against kernel=None (Block 35's exact separable pixel integral), 2 x 4 directions for the hybrid (time),
    # half-length 8 um, n_avg 4, v = n p, n = -26..26, planes -4..4. Bound 5e-5 (the entry); the scheme's own
    # discretisation (band at 16 samples per FWHM, the spreading's aliases) is far below it for this smooth kernel.
    cfg = make_cfg(n_rho=2, n_psi=4)
    p = cfg.acquisition.res0_um
    v, ks = np.arange(-26, 27) * p, np.arange(-4, 5)
    worst, rows = 0.0, []
    for d, tilt, th, cz in i_cases():
        tube, o, yh, eu = i_tube(d, tilt, th, cz)
        for mode in ("partition_vertical", "hybrid"):
            cache = LF.SectionCache()
            g = LF.line_profiles(tube, 1.0, ks, o, yh, eu, v, 4, p, cfg, absorption=mode, cache=cache)
            s = LF.line_profiles(tube, 1.0, ks, o, yh, eu, v, 4, p, cfg, absorption=mode, cache=cache,
                                 kernel=LS.SGTKernel(2.0, 0.0, 1.0))
            err = float(np.max(np.abs(s - g)))
            worst = max(worst, err)
            rows.append((d, tilt, th, cz, mode, err))
            check_close("gaussian_member_d%.1f_t%.1f_h%.2f_z%.2f_%s" % (d, tilt, th, cz, mode), s, g, rtol=0, atol=TOL)
    print("  (i) Gaussian member against kernel=None, %d cases: max |dtau| %.2e (bound %.0e)" % (len(rows), worst, TOL))
    REPORT["i"] = worst


def test_known_answer():
    # (ii) a flat tube along x (heading 0) through the read's axis, the line along y through a pixel centre, read at
    # v = n p (pixel centres; n_avg 0), half-length 10 um, partition_vertical, mu 1 /um, axis 0.04 um aside and 0.1 um
    # deep; against the identifiability check's 1-D model (the SGT's pixel means by its closed-form CDF, exact for this
    # geometry and untruncated), the reach raised per shape until truncated_mass <= 1e-7. Bound 5e-5. Reported, no
    # bound: (2, 1, 1) at the default reach.
    M1 = load_1d_model()
    shapes = SHAPES_II if LONG else ((1.3, 0.0, 1.0), (2.0, 0.3, 1.3))
    diams = (0.4, 1.5) if LONG else (0.4,)
    plane_half = 4 if LONG else 2
    v0, z_ax, mu = 0.04, 0.1, 1.0
    worst, rows = 0.0, []
    for shape in shapes + ((2.0, 1.0, 1.0),):
        reported = shape == (2.0, 1.0, 1.0)
        reach = 12.0 if reported else reach_for(shape)
        cfg = make_cfg(reach=reach)
        st = M1.make_setup(cfg, plane_half=plane_half)
        p = st.p
        o = np.array([437 * p, 211 * p])
        for d in diams:
            tube = G.Tube((o[0], o[1] + v0, z_ax), 0.5 * d, 0.0, 0.0, 1.0, 10.0, "axial")
            got = LF.line_profiles(tube, mu, st.ks, o, (0.0, 1.0), (1.0, 0.0), st.v, 0, p, cfg,
                                   absorption="partition_vertical", kernel=LS.SGTKernel(*shape))
            ref = M1.tau(d, mu, v0, z_ax, st, M1.Kernel(shape, method="exact", gauss_exact=False))
            err = float(np.max(np.abs(got - ref)))
            tm = LS.truncated_mass(LS.SGTKernel(*shape), cfg)
            rows.append((shape, d, reach, tm, err, reported))
            if not reported:
                worst = max(worst, err)
    print("  (ii) flat tube against the 1-D model (planes -%d..%d):" % (plane_half, plane_half))
    for shape, d, reach, tm, err, reported in rows:
        print("    shape %-16s d %.1f  reach %6.1f  truncated_mass %.1e  max |dtau| %.2e%s" % (
            shape, d, reach, tm, err, "  (reported, default reach)" if reported else ""))
    REPORT["ii"] = rows
    bad = [r for r in rows if not r[5] and r[4] > TOL]
    assert not bad, "above %.0e: %s" % (TOL, bad)


def test_reference():
    # (iii) tilted tubes (heading 0.52 rad, half-length 3.5 um, planes -1..1, the line through the middle, partition,
    # n_avg 4, v = n p, n = -26..26) at line_fine_factor 4 (both the module and the reference: the reference grid is
    # the fine lattice padded by more than the largest support), against brute_reference. References: (1.3, 0, 1) and
    # (4, 0, 1) the untruncated isotropic transfer (cf_table) at the reach raised to truncated_mass <= 1e-7 (13.9 and
    # 12 law SDs); (2, 0.5, 1) the truncated bivariate t (Hankel) and the skews (2, 0.3, 1.3), (2, 0.3, 0.77) the
    # cell-averaged stretched truncated bivariate t, at the default reach, the reference truncated as the module is
    # (the issue reported): the raised reach (117.2 and 28.6 law SDs) needs fine-lattice grids of 10400^2 to 18000^2
    # for (2, 0.5, 1) (0.9 to 2.6 GB per complex array; this machine has 7 GB) and 3600^2 to 5900^2 for the skews
    # (about 5 to 60 min per case with the cell-averaged kernel). The long run adds the skews at the raised reach on
    # the tilt 0.3 rad geometries (test_reference_raised). Bound 5e-5.
    geoms = [(d, t) for t in (0.3, 0.6) for d in (0.4, 1.0)] if LONG else [(0.4, 0.3)]
    shapes = ([((1.3, 0.0, 1.0), "cf"), ((4.0, 0.0, 1.0), "cf"), ((2.0, 0.5, 1.0), "t"), ((2.0, 0.3, 1.3), "cell"),
               ((2.0, 0.3, 0.77), "cell")] if LONG else [((1.3, 0.0, 1.0), "cf"), ((2.0, 0.3, 1.3), "cell")])
    ks = np.arange(-1, 2)
    worst, rows = 0.0, []
    for shape, kind in shapes:
        reach = reach_for(shape) if kind == "cf" else 12.0
        cfg = make_cfg(f=4, reach=reach)
        p = cfg.acquisition.res0_um
        v = np.arange(-26, 27) * p
        for d, tilt in geoms:
            tube, o, yh, eu = tilted_case(d, tilt)
            t0 = time.perf_counter()
            got = LF.line_profiles(tube, 1.0, ks, o, yh, eu, v, 4, p, cfg, absorption="partition_vertical",
                                   kernel=LS.SGTKernel(*shape))
            t_mod = time.perf_counter() - t0
            if kind == "cf":
                make = iso_cf_factory(shape)
            elif kind == "t":
                make = iso_t_factory(shape, reach)
            else:
                W = BK.sgt_unit_fwhm(*shape)
                make = cell_skew_transfer(shape[1], shape[2], W, reach * W / FW, yh)
            t0 = time.perf_counter()
            ref, grid = brute_reference(tube, 1.0, ks, o, yh, eu, v, 4, p, cfg, make,
                                        max_support_um(tube, ks, cfg, reach, shape[2]))
            t_ref = time.perf_counter() - t0
            err = float(np.max(np.abs(got - ref)))
            worst = max(worst, err)
            rows.append((shape, kind, reach, d, tilt, err, grid, t_mod, t_ref))
    print("  (iii) tilted tubes against the brute force (f 4, planes -1..1):")
    for shape, kind, reach, d, tilt, err, grid, t_mod, t_ref in rows:
        print("    shape %-16s ref %-4s reach %5.1f d %.1f tilt %.1f  max |dtau| %.2e  grid %dx%d  module %.1f s,"
              " reference %.1f s" % (shape, kind, reach, d, tilt, err, grid[0], grid[1], t_mod, t_ref))
    REPORT["iii"] = rows
    bad = [r[:6] for r in rows if r[5] > TOL]
    assert not bad, "above %.0e: %s" % (TOL, bad)


def test_reference_raised():
    # (iii) at the raised reach as the entry writes it, on the cases affordable here: both skews, (2, 0.3, 1.3) and
    # (2, 0.3, 0.77), on the tilt 0.3 rad geometries (d 0.4 and 1.0 um), sgt_reach_sigmas 28.6 (truncated_mass
    # <= 1e-7), f 4, against the cell-averaged reference cut at the same reach on a grid padded by more than its
    # support (3400^2 to 4200^2). Long run only (about 40 min). Bound 5e-5. Not run: the tilt 0.6 rad geometries
    # (grids 5400^2 to 5900^2, about 1 h per case here) and (2, 0.5, 1) at 117.2 law SDs (grids 10400^2 to 18000^2).
    if not LONG:
        raise unittest.SkipTest("long run only (LINE_SGT_LONG=1): the raised reach's grids take about 40 min")
    v_n = np.arange(-26, 27)
    ks = np.arange(-1, 2)
    rows = []
    for shape in ((2.0, 0.3, 1.3), (2.0, 0.3, 0.77)):
        reach = reach_for(shape)
        cfg = make_cfg(f=4, reach=reach)
        p = cfg.acquisition.res0_um
        v = v_n * p
        W = BK.sgt_unit_fwhm(*shape)
        for d in (0.4, 1.0):
            tube, o, yh, eu = tilted_case(d, 0.3)
            t0 = time.perf_counter()
            got = LF.line_profiles(tube, 1.0, ks, o, yh, eu, v, 4, p, cfg, absorption="partition_vertical",
                                   kernel=LS.SGTKernel(*shape))
            t_mod = time.perf_counter() - t0
            t0 = time.perf_counter()
            ref, grid = brute_reference(tube, 1.0, ks, o, yh, eu, v, 4, p, cfg,
                                        cell_skew_transfer(shape[1], shape[2], W, reach * W / FW, yh),
                                        max_support_um(tube, ks, cfg, reach, shape[2]))
            t_ref = time.perf_counter() - t0
            err = float(np.max(np.abs(got - ref)))
            rows.append((shape, reach, d, err, grid, t_mod, t_ref))
            print("  (iii) raised reach: shape %-16s reach %.1f (truncated_mass %.1e) d %.1f tilt 0.3: max |dtau| %.2e"
                  "  grid %dx%d  module %.1f s, reference %.1f s" % (shape, reach, LS.truncated_mass(
                      LS.SGTKernel(*shape), cfg), d, err, grid[0], grid[1], t_mod, t_ref), flush=True)
    REPORT["iii_raised"] = rows
    bad = [r[:4] for r in rows if r[3] > TOL]
    assert not bad, "above %.0e: %s" % (TOL, bad)


def test_transfer():
    # (iv) (a) along y_hat (nu = t y_hat) the untruncated kernel's transfer equals the 1-D characteristic function of
    # g_{w,kappa} by quadrature of (H2)-(H3) [1e-7 absolute]: the module's construction (its table H and, for the
    # skew, its odd part _odd_transfer) in core units (a = 1) on a lattice aligned with y_hat at H/a = 0.02 (the
    # construction converges with order about 4.5: 2e-5, 6.4e-7, 3.4e-8 at 0.08, 0.04, 0.02 for (2, 0.3, 1.3)), the
    # reach large enough that the truncated mass is below 1e-9. (b) zeta = 1 gives the isotropic transfer [exact].
    # (c) zeta -> 1/zeta mirrors the profile of a tube symmetric about the line's origin [1e-10]. (d) sgt_tail_mass
    # against its closed forms: exp(-r^2), r K_1(r), (1 + xi r^2)^(-1/xi) [1e-8].
    frame = LS._Frame(np.array([0.0, 1.0]))
    worst_a = 0.0
    for shape in ((2.0, 0.5, 1.0), (1.3, 0.0, 1.0), (4.0, 0.0, 1.0), (2.0, 0.3, 1.3), (2.0, 0.3, 0.77)):
        b, x, z = shape
        rho_c = 12.0
        while LS.sgt_tail_mass(b, x, rho_c) > 1e-9:
            rho_c *= 1.25
        table = LS._build_table(b, x, rho_c)
        if z == 1.0:
            ts = np.linspace(0.0, 6.0, 25)
            mod = table(ts) + 0j
        else:
            H = 0.02
            N = sfft.next_fast_len(int(math.ceil((2.0 * max(z, 1.0 / z) * rho_c + 4.0) / H)))
            O = LS._odd_transfer(table, z, 1.0, N, N, H, frame)
            fy = sfft.fftfreq(N, H)
            idx = np.flatnonzero((fy >= 0.0) & (fy <= 6.0))
            idx = idx[::max(1, idx.size // 24)]
            ts = fy[idx]
            mod = LS._even(table, 1.0, ts, np.zeros_like(ts), z) + O[idx, 0]
        ref = np.array([cf_1d(float(t), b, x, z) for t in ts])
        err = float(np.max(np.abs(mod - ref)))
        worst_a = max(worst_a, err)
        print("  (iv)(a) shape %-16s rho_c %6.1f: max |K(t y_hat) - CF_1D(t)| %.2e over %d t in [0, 6]"
              % (shape, rho_c, err, ts.size))
        check_close("transfer_along_yhat_%s" % (shape,), mod, ref, rtol=0, atol=1e-7)
    # (b) the skew construction at zeta = 1: the even part is table(a |nu|) and the odd part 0, exactly
    table = LS._build_table(2.0, 0.3, 8.0)
    nu = np.linspace(-3.0, 3.0, 41)
    nu_u, nu_s = nu[:, None] * 0.6 + nu[None, :] * 0.8, nu[:, None] * 0.8 - nu[None, :] * 0.6
    assert_array_equal(LS._even(table, 0.7, nu_u, nu_s, 1.0), table(0.7 * np.hypot(nu_u, nu_s)))
    assert not np.any(LS._odd_transfer(table, 1.0, 0.7, 64, 60, 0.05, frame)), "the odd part at zeta = 1 is not 0"
    # (c) mirror: a flat tube along x through the line's origin on a pixel centre, read across, planes -2..2
    cfg = make_cfg()
    p = cfg.acquisition.res0_um
    v, ks = np.arange(-26, 27) * p, np.arange(-2, 3)
    o = np.array([437 * p, 211 * p])
    tube = G.Tube((o[0], o[1], 0.1), 0.3, 0.0, 0.0, 1.0, 8.0, "axial")
    t1 = LF.line_profiles(tube, 1.0, ks, o, (0.0, 1.0), (1.0, 0.0), v, 4, p, cfg, absorption="partition_vertical",
                          kernel=LS.SGTKernel(2.0, 0.3, 1.3))
    t2 = LF.line_profiles(tube, 1.0, ks, o, (0.0, 1.0), (1.0, 0.0), v, 4, p, cfg, absorption="partition_vertical",
                          kernel=LS.SGTKernel(2.0, 0.3, 1.0 / 1.3))
    mirror = float(np.max(np.abs(t1 - t2[:, ::-1])))
    skew = float(np.max(np.abs(t1 - t1[:, ::-1])))
    print("  (iv)(c) zeta 1.3 against 1/1.3 mirrored: max |dtau| %.2e (the profile's own asymmetry %.2e)"
          % (mirror, skew))
    check_close("mirror_zeta", t1, t2[:, ::-1], rtol=0, atol=1e-10)
    assert skew > 1e-4, "the skew does not show in the profile (%.2e)" % skew
    # (d) the 2-D tail mass against closed forms
    worst_d = 0.0
    for b, x, form in ((2.0, 0.0, lambda r: math.exp(-r * r)), (1.0, 0.0, lambda r: r * special.k1(r)),
                       (2.0, 0.5, lambda r: (1.0 + 0.5 * r * r) ** -2.0), (2.0, 1.0, lambda r: 1.0 / (1.0 + r * r)),
                       (2.0, 2.0, lambda r: (1.0 + 2.0 * r * r) ** -0.5)):
        for r in (0.0, 0.3, 1.0, 2.5, 7.8, 20.0):
            got, exp = LS.sgt_tail_mass(b, x, r), (1.0 if r == 0.0 else form(r))
            worst_d = max(worst_d, abs(got - exp))
            check_close("tail_mass_%g_%g_%g" % (b, x, r), got, exp, rtol=0, atol=1e-8)
    print("  (iv)(d) sgt_tail_mass against the closed forms: max |diff| %.2e" % worst_d)
    REPORT["iv"] = (worst_a, mirror, worst_d)


def test_smoothness():
    # (v) central differences of tau in log10 beta, xi and ln zeta at steps 1e-3 and 5e-4 agree within 1 % of the
    # derivative's largest magnitude, on two of (iii)'s geometries (f 8, the default reach); the base shapes
    # (1.3, 0.2, 1) (isotropic; log10 beta, xi, and ln zeta about zeta = 1) and (2, 0.3, 1.3) (skewed). Quick run:
    # one geometry, (1.3, 0.2, 1), log10 beta and xi only.
    cfg = make_cfg()
    p = cfg.acquisition.res0_um
    v, ks = np.arange(-26, 27) * p, np.arange(-1, 2)
    geoms = [(0.4, 0.3), (1.0, 0.6)] if LONG else [(0.4, 0.3)]
    bases = [(1.3, 0.2, 1.0), (2.0, 0.3, 1.3)] if LONG else [(1.3, 0.2, 1.0)]
    rows = []
    for d, tilt in geoms:
        tube, o, yh, eu = tilted_case(d, tilt)
        cache = LF.SectionCache()

        def tau(lb, xi, lz):
            return LF.line_profiles(tube, 1.0, ks, o, yh, eu, v, 4, p, cfg, absorption="partition_vertical",
                                    cache=cache, kernel=LS.SGTKernel(10.0 ** lb, xi, math.exp(lz)))
        for b, x, z in bases:
            q0 = np.array([math.log10(b), x, math.log(z)])
            for i, name in enumerate(("log10_beta", "xi", "ln_zeta")):
                if name == "ln_zeta" and not LONG:
                    continue                            # the quick run: the isotropic path's coordinates only
                der = []
                for hstep in (1e-3, 5e-4):
                    e = np.zeros(3)
                    e[i] = hstep
                    der.append((tau(*(q0 + e)) - tau(*(q0 - e))) / (2.0 * hstep))
                scale = float(np.max(np.abs(der[0])))
                diff = float(np.max(np.abs(der[0] - der[1])))
                rows.append((d, tilt, (b, x, z), name, scale, diff))
    print("  (v) smoothness: central differences at 1e-3 and 5e-4")
    for d, tilt, base, name, scale, diff in rows:
        print("    d %.1f tilt %.1f base %-16s %-10s  max |dtau/dq| %.3e  max |D(1e-3) - D(5e-4)| %.2e (%.3f %%)"
              % (d, tilt, base, name, scale, diff, 100.0 * diff / scale))
    REPORT["v"] = rows
    bad = [r for r in rows if r[5] > 0.01 * r[4]]
    assert not bad, "central differences differ by more than 1 %%: %s" % (bad,)


def test_truncation():
    # (vi) the truncation at the default reach (12 law SDs) on (2, 1, 1), the bivariate t with nu = 2 (its line
    # projection decays as |u|^-3: 12 law SDs drop 1.6 % of it): line_profiles against the brute force with the
    # truncated P_1 (Hankel transform of the closed form), on (iii)'s tilted geometries [5e-5]; truncated_mass against
    # (1 + xi r^2)^(-1/xi), r = 12 W / FWHM_PER_SD [1e-8].
    shape = (2.0, 1.0, 1.0)
    cfg = make_cfg(f=4)
    p = cfg.acquisition.res0_um
    v, ks = np.arange(-26, 27) * p, np.arange(-1, 2)
    W = BK.sgt_unit_fwhm(*shape)
    r = 12.0 * W / FW
    tm = LS.truncated_mass(LS.SGTKernel(*shape), cfg)
    print("  (vi) truncated_mass(2, 1, 1) at 12 law SDs: %.10f, closed form %.10f" % (tm, 1.0 / (1.0 + r * r)))
    check_close("truncated_mass_211", tm, 1.0 / (1.0 + r * r), rtol=0, atol=1e-8)
    geoms = [(d, t) for t in (0.3, 0.6) for d in (0.4, 1.0)] if LONG else [(0.4, 0.3)]
    worst = 0.0
    for d, tilt in geoms:
        tube, o, yh, eu = tilted_case(d, tilt)
        got = LF.line_profiles(tube, 1.0, ks, o, yh, eu, v, 4, p, cfg, absorption="partition_vertical",
                               kernel=LS.SGTKernel(*shape))
        ref = brute_reference(tube, 1.0, ks, o, yh, eu, v, 4, p, cfg, iso_t_factory(shape, 12.0),
                              max_support_um(tube, ks, cfg, 12.0))[0]
        err = float(np.max(np.abs(got - ref)))
        worst = max(worst, err)
        print("  (vi) d %.1f tilt %.1f: max |dtau| against the truncated-P_1 reference %.2e" % (d, tilt, err))
        check_close("truncation_211_d%.1f_t%.1f" % (d, tilt), got, ref, rtol=0, atol=TOL)
    REPORT["vi"] = (tm, worst)


def _small_case():
    cfg = make_cfg(n_rho=2, n_psi=4)
    p = cfg.acquisition.res0_um
    tube, o, yh, eu = i_tube(0.5, 0.3, 0.52, 0.05)
    return cfg, p, tube, o, yh, eu, np.arange(-26, 27) * p, np.arange(-2, 3)


def test_invariants():
    # (vii) mu = 0 gives 1 [exact]; mirror symmetry in v for zeta = 1 and a tube through the line's origin on a pixel
    # centre [1e-10]; the order of ks (rows follow ks) [exact]; a cached call equals an uncached one [1e-12: a cached
    # table is the same arrays, so the results are bit-identical].
    cfg, p, tube, o, yh, eu, v, ks = _small_case()
    kern = LS.SGTKernel(1.3, 0.2, 1.0)
    for mode in ("partition_vertical", "hybrid", "linear"):
        one = LF.line_profiles(tube, 0.0, ks, o, yh, eu, v, 4, p, cfg, absorption=mode, kernel=kern)
        assert_array_equal(one, np.ones((ks.size, v.size)))
    oo = np.array([437 * p, 211 * p])
    flat = G.Tube((oo[0], oo[1], 0.1), 0.35, 0.0, 0.0, 1.0, 8.0, "axial")
    worst = 0.0
    for mode in ("partition_vertical", "hybrid"):
        t = LF.line_profiles(flat, 1.0, ks, oo, (0.0, 1.0), (1.0, 0.0), v, 4, p, cfg, absorption=mode, kernel=kern)
        worst = max(worst, float(np.max(np.abs(t - t[:, ::-1]))))
        check_close("mirror_v_%s" % mode, t, t[:, ::-1], rtol=0, atol=1e-10)
    perm = np.array([2, 0, 4, 1, 3])
    a = LF.line_profiles(tube, 1.0, ks, o, yh, eu, v, 4, p, cfg, absorption="partition_vertical", kernel=kern)
    b = LF.line_profiles(tube, 1.0, ks[perm], o, yh, eu, v, 4, p, cfg, absorption="partition_vertical", kernel=kern)
    check_close("order_of_ks", b, a[perm], rtol=0, atol=1e-12)
    cache = LF.SectionCache()
    skew = LS.SGTKernel(2.0, 0.3, 1.3)
    for kk in (kern, skew):
        c1 = LF.line_profiles(tube, 1.0, ks, o, yh, eu, v, 4, p, cfg, absorption="hybrid", cache=cache, kernel=kk)
        c2 = LF.line_profiles(tube, 1.0, ks, o, yh, eu, v, 4, p, cfg, absorption="hybrid", cache=cache, kernel=kk)
        c0 = LF.line_profiles(tube, 1.0, ks, o, yh, eu, v, 4, p, cfg, absorption="hybrid", kernel=kk)
        check_close("cached_equals_uncached_%s" % (kk,), c2, c0, rtol=0, atol=1e-12)
        check_close("cached_repeat_%s" % (kk,), c1, c0, rtol=0, atol=1e-12)
    assert cache.hits >= 2, cache.hits
    print("  (vii) mirror in v %.2e; mu = 0 exact; order of ks; cached = uncached (cache hits %d)" % (worst, cache.hits))


def test_contract():
    # (vii) shape (n_planes, n_v), float64, finite, values in (0, 1] for a faint tube; the constants and the API.
    cfg, p, tube, o, yh, eu, v, ks = _small_case()
    for kk in (LS.SGTKernel(2.0, 0.5), LS.SGTKernel(1.3, 0.0, 0.8)):
        t = LF.line_profiles(tube, 0.5, ks, o, yh, eu, v, 4, p, cfg, absorption="partition_vertical", kernel=kk)
        assert t.shape == (ks.size, v.size), t.shape
        assert t.dtype == np.float64, t.dtype
        assert np.all(np.isfinite(t))
        assert np.all(t <= 1.0 + 1e-12) and np.all(t > 0.0), (t.min(), t.max())
    assert LS.FWHM_PER_SD == 2.0 * math.sqrt(2.0 * math.log(2.0))
    k = LS.SGTKernel(2, 0, 1)
    assert (k.beta, k.xi, k.zeta) == (2.0, 0.0, 1.0) and all(isinstance(q, float) for q in (k.beta, k.xi, k.zeta))
    assert LS.SGTKernel(1.5, 0.2).zeta == 1.0
    try:
        k.beta = 3.0
    except dataclasses.FrozenInstanceError:
        pass
    else:
        raise AssertionError("SGTKernel is not frozen")
    tm = LS.truncated_mass(LS.SGTKernel(1.3, 0.0, 1.0), cfg)
    assert isinstance(tm, float) and 0.0 <= tm <= 1.0
    assert cfg.inference.sgt_reach_sigmas == 12.0 and cfg.inference.sgt_samples_per_fwhm == 16


def test_determinism():
    # (vii) no randomness: two calls are bit-identical (fresh caches).
    cfg, p, tube, o, yh, eu, v, ks = _small_case()
    for kk in (LS.SGTKernel(1.3, 0.2, 1.0), LS.SGTKernel(2.0, 0.3, 1.3)):
        a = LF.line_profiles(tube, 1.0, ks, o, yh, eu, v, 4, p, cfg, absorption="hybrid", kernel=kk)
        b = LF.line_profiles(tube, 1.0, ks, o, yh, eu, v, 4, p, cfg, absorption="hybrid", kernel=kk)
        assert_array_equal(a, b)


def test_edge_cases():
    # (vii) the ValueErrors: a kernel neither None nor an SGTKernel; an SGT shape outside the family's domain;
    # sgt_reach_sigmas <= 0 or not finite; sgt_samples_per_fwhm not an integer >= 1; Block 35's own (mu < 0, an
    # unknown absorption, a non-unit axis, an uncapped tube) with an SGT kernel; ExtrusionInvalid 0.5 um from a cap
    # (also at mu = 0), checked before the mu = 0 shortcut; sgt_tail_mass and truncated_mass refusals.
    cfg, p, tube, o, yh, eu, v, ks = _small_case()
    kern = LS.SGTKernel(1.3, 0.2, 1.0)

    def raises(exc, fn, *a, **kw):
        try:
            fn(*a, **kw)
        except exc:
            return
        raise AssertionError("%s not raised by %s %s %s" % (exc.__name__, getattr(fn, "__name__", fn), a, kw))

    call = LF.line_profiles
    args = (tube, 1.0, ks, o, yh, eu, v, 4, p)
    for bad in ((2.0, 0.0), "sgt", 3, (2.0, 0.0, 1.0)):
        raises(ValueError, call, *args, cfg, kernel=bad)
    for b, x, z in ((0.0, 0.0, 1.0), (-1.0, 0.0, 1.0), (2.0, -0.1, 1.0), (2.0, 0.0, 0.0), (2.0, 0.0, -1.0),
                    (float("nan"), 0.0, 1.0), (2.0, float("inf"), 1.0), (2.0, 0.0, float("inf"))):
        raises(ValueError, LS.SGTKernel, b, x, z)
    for reach in (0.0, -1.0, float("nan"), float("inf")):
        raises(ValueError, call, *args, make_cfg(reach=reach), kernel=kern)
    for spf in (0, -3, 2.5):
        c = dataclasses.replace(cfg, inference=dataclasses.replace(cfg.inference, sgt_samples_per_fwhm=spf))
        raises(ValueError, call, *args, c, kernel=kern)
    raises(ValueError, call, tube, -0.1, ks, o, yh, eu, v, 4, p, cfg, kernel=kern)
    raises(ValueError, call, *args, cfg, absorption="ray_world", kernel=kern)
    raises(ValueError, call, tube, 1.0, ks, o, (0.0, 2.0), eu, v, 4, p, cfg, kernel=kern)
    raises(ValueError, call, G.Tube(tube.c, tube.r, tube.phi, tube.theta, 1.0, None), 1.0, ks, o, yh, eu, v, 4, p,
           cfg, kernel=kern)
    # a cap 0.5 um from the line's origin along the axis
    a3 = G.axis_direction(tube.phi, tube.theta)
    c = np.array([o[0], o[1], 0.0]) - (8.0 - 0.5) * a3
    near = G.Tube((c[0], c[1], c[2]), tube.r, tube.phi, tube.theta, 1.0, 8.0, "axial")
    for mu in (1.0, 0.0):
        raises(LF.ExtrusionInvalid, call, near, mu, ks, o, yh, eu, v, 4, p, cfg, kernel=kern)
    assert issubclass(LF.ExtrusionInvalid, ValueError)
    for bad in (-1.0, float("nan"), float("inf")):
        raises(ValueError, LS.sgt_tail_mass, 2.0, 0.0, bad)
    raises(ValueError, LS.sgt_tail_mass, 0.0, 0.0, 1.0)
    raises(ValueError, LS.truncated_mass, (2.0, 0.0, 1.0), cfg)
    raises(ValueError, LS.truncated_mass, kern, make_cfg(reach=-1.0))
    print("  (vii) ValueErrors and ExtrusionInvalid as specified")


def _contract_line_fast():
    """Block 35's line_fast.py at the contract commit (git show), loaded inside allen_diameter.model."""
    rel = "Passive Features/Diameter Re-measurement/src/allen_diameter/model/line_fast.py"
    try:
        top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=str(WS), capture_output=True, text=True,
                             check=True).stdout.strip()
        text = subprocess.run(["git", "show", "%s:%s" % (CONTRACT, rel)], cwd=top, capture_output=True, text=True,
                              check=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise unittest.SkipTest("the contract commit's line_fast.py is not reachable by git (%s)" % (exc,))
    tmp = Path(tempfile.mkdtemp()) / "line_fast_contract.py"
    tmp.write_text(text)
    name = "allen_diameter.model._line_fast_contract"
    spec = importlib.util.spec_from_file_location(name, str(tmp))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod                              # registered first: its dataclasses look themselves up there
    spec.loader.exec_module(mod)
    return mod


def test_block35():
    # (vii) Block 35's quick smoke tests pass unchanged (tests/smoke/test_smoke_line_fast.py: known answer,
    # convergence, invariants, contract, determinism, edge cases and the eight split tests but its timing), and its
    # Gaussian path (kernel=None) is bit-identical to the contract commit's on 6 cases (3 absorptions, flat and
    # tilted, heading 0 and 0.52, axis depth 0 and 0.14 um).
    old = _contract_line_fast()
    cfg = make_cfg(n_rho=2, n_psi=4)
    p = cfg.acquisition.res0_um
    v, ks = np.arange(-26, 27) * p, np.arange(-4, 5)
    n_same = 0
    for d, tilt, th, cz, mode in ((0.3, 0.0, 0.0, 0.0, "partition_vertical"), (1.0, 0.3, 0.52, 0.14,
                                                                             "partition_vertical"),
                                  (0.3, 0.3, 0.52, 0.0, "hybrid"), (1.0, 0.0, 0.0, 0.14, "hybrid"),
                                  (1.0, 0.3, 0.0, 0.0, "linear"), (0.3, 0.0, 0.52, 0.14, "linear")):
        tube, o, yh, eu = i_tube(d, tilt, th, cz)
        new = LF.line_profiles(tube, 1.0, ks, o, yh, eu, v, 4, p, cfg, absorption=mode)
        ref = old.line_profiles(tube, 1.0, ks, o, yh, eu, v, 4, p, cfg, absorption=mode)
        assert_array_equal(new, ref)
        n_same += 1
    spec = importlib.util.spec_from_file_location("test_smoke_line_fast_b35", str(HERE / "test_smoke_line_fast.py"))
    b35 = importlib.util.module_from_spec(spec)
    sys.modules["test_smoke_line_fast_b35"] = b35
    spec.loader.exec_module(b35)
    names = ("test_known_answer", "test_convergence", "test_invariants", "test_contract", "test_determinism",
             "test_edge_cases", "test_split_sum", "test_split_known_answer", "test_split_limits",
             "test_split_slab_grid", "test_split_symmetry", "test_split_tiling", "test_split_contract")
    for name in names:
        getattr(b35, name)()
    print("  (vii) Gaussian path bit-identical to %s on %d cases; Block 35's %d quick tests pass" % (CONTRACT, n_same,
                                                                                                  len(names)))


def test_convergence():
    raise unittest.SkipTest("the entry names no convergence oracle; the scheme's discretisation is checked at its "
                            "settings through the references of (i)-(iii), (vi)")


def test_timing():
    # Reported, no bound: the time per warm call (the second call with the same SectionCache) of the Gaussian path
    # and of the SGT path, zeta = 1 (1.3, 0.2, 1) and zeta != 1 (2, 0.3, 1.3), at the default reach, hybrid at 8 x 16
    # directions (the inference's), planes -4..4, on (i)'s 16 geometries at line_fine_factor 8, and at 32 (the error
    # bars' lattice, D-224.2) on the 8 with the axis in the plane (d, tilt, heading; the axis depth changes little:
    # about 1 h saved); truncated_mass of the shapes tested. Long run only.
    print("  truncated_mass at the default reach (12 law SDs) and at the raised reach of (ii):")
    for shape in SHAPES_II + ((2.0, 1.0, 1.0), (1.3, 0.2, 1.0)):
        k = LS.SGTKernel(*shape)
        rr = reach_for(shape)
        print("    shape %-16s %.3e   (reach %5.1f: %.3e)" % (shape, LS.truncated_mass(k, make_cfg()), rr,
                                                             LS.truncated_mass(k, make_cfg(reach=rr))))
    if not LONG:
        raise unittest.SkipTest("timing runs in the long run only (LINE_SGT_LONG=1); truncated masses printed")
    rows = []
    for f, cases in ((8, i_cases()), (32, [c for c in i_cases() if c[3] == 0.0])):
        cfg = make_cfg(f=f)
        p = cfg.acquisition.res0_um
        v, ks = np.arange(-26, 27) * p, np.arange(-4, 5)
        for d, tilt, th, cz in cases:
            tube, o, yh, eu = i_tube(d, tilt, th, cz)
            times = []
            for kern in (None, LS.SGTKernel(1.3, 0.2, 1.0), LS.SGTKernel(2.0, 0.3, 1.3)):
                cache = LF.SectionCache()
                LF.line_profiles(tube, 1.0, ks, o, yh, eu, v, 4, p, cfg, absorption="hybrid", cache=cache, kernel=kern)
                t0 = time.perf_counter()
                LF.line_profiles(tube, 1.0, ks, o, yh, eu, v, 4, p, cfg, absorption="hybrid", cache=cache, kernel=kern)
                times.append(time.perf_counter() - t0)
            rows.append((f, d, tilt, th, cz, times))
            print("    f %2d d %.1f tilt %.1f heading %.2f cz %.2f: Gaussian %7.2f s, SGT zeta=1 %7.2f s, zeta!=1 %7.2f s"
                  % ((f, d, tilt, th, cz) + tuple(times)), flush=True)
    REPORT["timing"] = rows


# ---------------------------------------------------------------- runner ---

def _environment():
    parts = [f"python {platform.python_version()}"]
    for package in REPORT_PACKAGES:
        try:
            parts.append(f"{package} {importlib.metadata.version(package)}")
        except importlib.metadata.PackageNotFoundError:
            parts.append(f"{package} (not installed)")
    parts.append(platform.platform())
    parts.append(f"seed {SEED}")
    parts.append("LONG run" if LONG else "quick run (LINE_SGT_LONG=1 for the long one)")
    return " | ".join(parts)


def main(argv=()):
    """Run the checks (all, or those named in argv, e.g. test_timing; a leading '-' excludes a name)."""
    checks = [(name, obj) for name, obj in globals().items()
              if name.startswith("test_") and callable(obj)]
    keep = [a for a in argv if not a.startswith("-")]
    drop = [a[1:] for a in argv if a.startswith("-")]
    checks = [(n, f) for n, f in checks if (not keep or n in keep) and n not in drop]
    print(f"== {Path(__file__).name}")
    print(_environment())

    results = []
    for name, func in checks:
        start = time.perf_counter()
        try:
            func()
            status, detail = "PASS", ""
        except unittest.SkipTest as exc:
            status, detail = "SKIP", str(exc)
        except NotImplementedError as exc:
            status, detail = "TODO", str(exc)
        except AssertionError as exc:
            status, detail = "FAIL", str(exc)
        except Exception:  # unexpected errors are reported, never hidden
            status, detail = "ERROR", traceback.format_exc()
        results.append((name, status, time.perf_counter() - start, detail))
        sys.stdout.flush()

    width = max([len(name) for name, _, _, _ in results] + [4])
    for name, status, seconds, detail in results:
        lines = detail.strip().splitlines()
        headline = lines[0] if lines else ""
        print(f"{status:<5}  {name:<{width}}  {seconds:8.3f}s  {headline}".rstrip())

    for name, status, _seconds, detail in results:
        if status in ("FAIL", "ERROR"):
            print(f"\n---- {status}: {name}\n{detail.strip()}")

    counts = {s: sum(1 for r in results if r[1] == s)
              for s in ("PASS", "FAIL", "ERROR", "TODO", "SKIP")}
    print("\n-- " + ", ".join(f"{n} {s.lower()}" for s, n in counts.items()))
    for path in _SAVED_ARTIFACTS:
        print(f"-- arrays from a failed comparison: {path}")
    failed = counts["FAIL"] + counts["ERROR"] + counts["TODO"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
