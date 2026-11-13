"""Helpers of the integration tester's run D2 (2026-11-13): Block 11 (Phase II
tools: the pilot, the fit variants of D-046, the variant iteration of D-047)
and its boundaries with Blocks 0, 4-8, 10, 12, 13.

Everything here is written from specs/SPEC.md, independently of the code
under test: the kernels are rebuilt from scipy's own densities, the tube
model is integrated by scipy.integrate.quad in the tube's own coordinate u
(not the code's Gauss-Legendre rule in psi), and the synthetic cell is the
fixture SPEC.md section 7 names as the substitute for Allen's server
(tests/smoke/fixtures_cell.py).

Pure ASCII.
"""
from __future__ import annotations

import functools
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy import integrate, optimize, special, stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent                      # Passive Features/Diameter Re-measurement
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
SMOKE = ROOT / "tests" / "smoke"
for _p in (SRC, SCRIPTS, SMOKE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

os.environ.setdefault("MPLBACKEND", "Agg")

FWHM_GAUSS = 2.0 * math.sqrt(2.0 * math.log(2.0))     # SPEC Block 11: W_gauss = 2 sqrt(2 ln 2)


def default_cfg():
    """The default configuration, validated (SPEC section 4: parameters only in config.py)."""
    from allen_diameter.config import default_config
    cfg = default_config()
    cfg.validate()
    return cfg


# ------------------------------------------------------------------ kernels ---
# SPEC Block 11 (D-046): g(z) = f_F(z / a + m_F(s); s) / a, a = w / W_F(s), with
# f_F the family's density at unit scale, W_F its FWHM, m_F its mode.

def scipy_unit_density(family, shape):
    """A density of the family at SOME scale, straight from scipy (the scale
    is irrelevant once the kernel is rescaled to its FWHM). Voigt: Gaussian
    SD 1 and Lorentzian half-width gamma chosen so that the Lorentzian FWHM /
    Gaussian FWHM = r (2 gamma / (2 sqrt(2 ln 2)) = r)."""
    if family == "gauss":
        return stats.norm.pdf
    if family == "lorentz":
        return stats.cauchy.pdf
    if family == "student":
        return lambda z: stats.t.pdf(z, shape)
    if family == "gennorm":
        return lambda z: stats.gennorm.pdf(z, shape)
    if family == "skewnorm":
        return lambda z: stats.skewnorm.pdf(z, shape)
    if family == "voigt":
        gamma = shape * math.sqrt(2.0 * math.log(2.0))
        return lambda z: special.voigt_profile(z, 1.0, gamma)
    raise ValueError(family)


def density_mode(family, shape, f):
    """The mode: 0 for the even families; for the skew-normal the root of
    d/dz ln f = -z + a phi(a z) / Phi(a z) (SPEC), by brentq."""
    if family != "skewnorm" or shape == 0:
        return 0.0
    a = float(shape)

    def dlog(z):
        return -z + a * math.exp(stats.norm.logpdf(a * z) - special.log_ndtr(a * z))
    # the mode of skewnorm(a) lies between 0 and sign(a) * sqrt(2/pi) (it tends to that as |a| grows)
    lo, hi = (0.0, 1.0) if a > 0 else (-1.0, 0.0)
    return optimize.brentq(dlog, lo, hi, xtol=1e-15)


def half_max_points(f, mode):
    """(left, right) half-maximum points of a unimodal density f about its mode, by brentq."""
    half = 0.5 * float(f(mode))
    span = 0.5
    while float(f(mode - span)) > half or float(f(mode + span)) > half:
        span *= 2.0
    left = optimize.brentq(lambda x: float(f(x)) - half, mode - span, mode, xtol=1e-15)
    right = optimize.brentq(lambda x: float(f(x)) - half, mode, mode + span, xtol=1e-15)
    return left, right


def reference_kernel(family, fwhm_um, shape=None):
    """The kernel of SPEC Block 11 rebuilt here: the scipy density rescaled
    to FWHM fwhm_um with its mode moved to 0, unit integral."""
    f = scipy_unit_density(family, shape)
    m = density_mode(family, shape, f)
    lo, hi = half_max_points(f, m)
    a = float(fwhm_um) / (hi - lo)

    def g(z):
        return f(np.asarray(z, dtype=float) / a + m) / a
    return g


def closed_form_unit_fwhm(family, shape=None):
    """SPEC Block 11's closed forms of W_F(s)."""
    if family == "gauss":
        return FWHM_GAUSS
    if family == "lorentz":
        return 2.0
    if family == "student":
        nu = float(shape)
        return 2.0 * math.sqrt(nu * (2.0 ** (2.0 / (nu + 1.0)) - 1.0))
    if family == "gennorm":
        return 2.0 * math.log(2.0) ** (1.0 / float(shape))
    raise ValueError("no closed form for %s" % family)


# -------------------------------------------------------------- tube model ---

def quad_model(v, d, alpha, v0, g, B_bar, peak_breakpoint=True):
    """SPEC section 2.2 (3'') / Block 11 (D-046), integrated by quad in the
    tube's coordinate u: I(v) = B [1 - int_{|u| <= d/2} (1 - exp(-alpha
    sqrt(1 - (2u/d)^2))) g(v - v0 - u) du]. The substitution u = (d/2) sin psi
    gives the spec's form; this is a different quadrature on a different
    variable, so it does not share the code's rule."""
    out = []
    r = 0.5 * float(d)
    for vi in np.atleast_1d(np.asarray(v, dtype=float)):
        def f(u, vi=vi):
            s = math.sqrt(max(0.0, 1.0 - (u / r) ** 2))
            return -math.expm1(-alpha * s) * float(g(vi - v0 - u))
        pts = None
        c = vi - v0                       # the kernel's peak at u = v - v0
        if peak_breakpoint and -r < c < r:
            pts = [c]
        val, _err = integrate.quad(f, -r, r, points=pts, limit=800, epsabs=1e-13, epsrel=1e-12)
        out.append(B_bar * (1.0 - val))
    return np.asarray(out)


def model_profile_reference(v, d, alpha, v0, sigma, B_bar):
    """Block 3's (3) with the Gaussian of SD sigma, by quad (the same integral as quad_model)."""
    return quad_model(v, d, alpha, v0, lambda z: math.exp(-0.5 * (z / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi)),
                      B_bar)


# ------------------------------------------------------------ the synthetic cell ---

@functools.lru_cache(maxsize=4)
def synthetic_cell(d_true=0.8, mu=1.0, n_nodes=6, allen_radius=0.3, seed=20261006):
    """(cfg, swc, provider, fetcher, planes, swc path) of tests/smoke/fixtures_cell.synthetic_cell
    (SPEC section 7's substitute for Allen's server), rendered once per session and parameters."""
    import fixtures_cell
    import run_cell
    cfg = default_cfg()
    tmp = tempfile.mkdtemp(prefix="tester_D2_cell_")
    swc, fetcher, planes, path = fixtures_cell.synthetic_cell(tmp, cfg, d_true=d_true, mu=mu, n_nodes=n_nodes,
                                                              allen_radius=allen_radius, seed=seed)
    provider = run_cell.real_provider(fetcher, planes, cfg.acquisition.res0_um)
    return cfg, swc, provider, fetcher, planes, path


class CountingProvider:
    """A provider wrapper that records every request (left, top, width, height, k_lo, k_hi)."""

    def __init__(self, provider):
        self.provider, self.calls = provider, []

    def __call__(self, *args):
        self.calls.append(tuple(args))
        return self.provider(*args)


def stretch_of(swc, cfg, node_id, transform=None):
    """(Branch, index) of a node on its own stretch (SPEC Block 8: cell.stretches, cell.stretch_branch)."""
    from allen_diameter.analysis import cell
    types = cfg.acquisition.dendrite_swc_types
    xyz_img = cell.to_image_um(swc.xyz, cfg.acquisition.res0_um, **(transform or {}))
    row = int(np.flatnonzero(swc.ids == int(node_id))[0])
    for run in cell.stretches(swc, types):
        hit = np.flatnonzero(run == row)
        if hit.size:
            branch, off = cell.stretch_branch(swc, run, xyz_img)
            return branch, off + int(hit[0])
    raise ValueError("node %d is on no dendrite stretch" % node_id)


@functools.lru_cache(maxsize=8)
def measured_node(node_id=4, **cell_kw):
    """(NodeResult of Block 13's measure_node with the cell's scene, Allen's radius, Branch, index) of one
    node of the synthetic cell, computed once per session."""
    from allen_diameter.analysis import cell
    from allen_diameter.analysis.node_pipeline import measure_node
    cfg, swc, provider, _f, _p, _path = synthetic_cell(**cell_kw)
    branch, i = stretch_of(swc, cfg, node_id)
    scene = cell.cell_scene(swc, cfg.acquisition.dendrite_swc_types, cfg.acquisition.res0_um, None)
    res = measure_node(branch, i, provider, cfg, None, scene)
    r_allen = float(swc.radius[int(np.flatnonzero(swc.ids == node_id)[0])])
    return res, r_allen, branch, i


@functools.lru_cache(maxsize=8)
def measured_pilot(node_id=4, **cell_kw):
    """Block 5's chain (node_pipeline.measure_pilot) on the same node, computed once per session."""
    from allen_diameter.analysis.node_pipeline import measure_pilot
    cfg, swc, provider, _f, _p, _path = synthetic_cell(**cell_kw)
    branch, i = stretch_of(swc, cfg, node_id)
    return measure_pilot(branch, i, provider, cfg, None), branch, i


def noisy_profile(cfg, d=0.8, mu=0.8, v0=0.05, phi=0.2, B=180.0, noise_sd=1.5, seed=7, kernel=None, n_nodes=None):
    """(v, I, phi, B) of a synthetic profile on Block 5's offsets: Block 3's Gaussian model at sigma_fit
    (kernel None) or model_profile_kernel with the kernel g, plus white noise of SD noise_sd (grey levels)
    from default_rng(seed)."""
    from allen_diameter.analysis import profiles
    from allen_diameter.model import tube_model
    v = profiles.profile_offsets(cfg.measure)
    alpha = mu * d / math.cos(phi)
    if kernel is None:
        n = n_nodes or 364
        I = tube_model.model_profile(v, d, alpha, v0, cfg.measure.sigma_fit_um, B, n)
    else:
        I = tube_model.model_profile_kernel(v, d, alpha, v0, kernel, B, n_nodes or 848)
    if noise_sd:
        I = I + np.random.default_rng(seed).normal(0.0, noise_sd, I.shape)
    return v, I, phi, B
