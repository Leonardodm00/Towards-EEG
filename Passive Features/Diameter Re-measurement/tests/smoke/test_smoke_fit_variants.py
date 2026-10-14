"""Smoke test for the fit variants at the iteration's plane (Cell 4l) --
Block 11 in specs/SPEC.md (D-046, the user's request of 2026-10-13, 14:50).

A kernel is g(z) = f_F(z / a + m_F(s); s) / a, a = w / W_F(s): unit integral,
FWHM w, maximum at z = 0 (model/blur_kernels.py). The model is (B3.1) with g
in place of the Gaussian (tube_model.model_profile_kernel). A variant fits
theta = (d, mu, v0[, w][, t(s)]) by least squares, unweighted or with the
centre weights omega_n = exp(-|v_n - v0| / lambda) re-centred on the fit's own
v0 by IRLS (analysis/fit_variants.py); Cell 4l runs every variant at Cell 4j's
k_iter (scripts/fit_variants.py).

Checks
    test_known_answer   per family and shape (bounds included): unit integral
                        (scipy.integrate.quad, 1e-9), FWHM w (brentq on the
                        kernel itself, 1e-9 relative), maximum at 0; the closed
                        forms of W_F against the numeric FWHM (1e-10); the
                        skew-normal's mode a root of d/dz ln f, m(-a) = -m(a);
                        noise-free profiles made with each fixed kernel are
                        fitted back by that kernel to 1e-6 um in d (unweighted
                        and lambda 0.5 um); with width or shape fitted, w, the
                        shape and d come back (1e-4)
    test_reference      the Gaussian kernel model equals tube_model.model_profile
                        (1e-11 grey levels); every kernel's model equals a
                        direct convolution by scipy.integrate.quad (1e-7 grey
                        levels for the smooth kernels, 0.5 for the cusped
                        generalized normals); the variant gauss, fixed width,
                        unweighted equals fit.fit_profile (d, mu, v0 to 1e-6);
                        a weighted fit is a fixed point of one more IRLS step
                        (1e-4 um); variant_curve at v equals the fit's model
    test_convergence    the kernel model converges with the Gauss-Legendre
                        nodes for a smooth kernel (N to 4N); for the cusped
                        gennorm beta = 1, tripling the nodes moves a fitted d
                        by <= 0.005 um (the claim of the module docstring)
    test_invariants     a gain on I and B_bar leaves every variant's d, mu, v0,
                        w, s (1e-6); shifting v by 0.15 um shifts v0 and keeps
                        d (1e-6); mirroring the profile mirrors v0 and the skew
                        and keeps d (1e-5); the weights peak at v0 with 1 and
                        fall symmetrically; the baseline's ratio is 1
    test_contract       check_spec, kernel_pdf, model_profile_kernel,
                        fit_weighted and select_specs refuse what they should;
                        variant_grid's size and order (100 by default); a start
                        without a finite d gives a failed fit with NaNs; flags;
                        IRLS stops only when d and v0 have both settled
                        (scripted runs)
    test_wiring         the starts of each unweighted variant (Block 5's 3, then
                        the warm ones it lists); every weighted fit is
                        fit_weighted from the unweighted fit of the same kernel
                        and width mode (identical numbers); PlaneFits keeps two
                        weightings of one kernel apart
    test_determinism    two fit_grid runs: identical numbers
    test_edge_cases     run() end to end on the 0.8 um synthetic cell from
                        Cell 4j's JSON (a reduced grid): one figure per node,
                        the summary, the CSV (one row per node and variant)
                        and the JSON (keys, the soma skipped, the baseline
                        equal to Cell 4j's d_iter); the node figure of the full
                        grid and the summary with every text inside; the CLI
                        passes its options to run (server stubbed) and refuses
                        a bad width mode, lambda and family

Run
    cd "Passive Features/Diameter Re-measurement"
    python tests/smoke/test_smoke_fit_variants.py

Keep this file pure ASCII (hpc-python-compat).
"""
from __future__ import annotations

import contextlib
import csv
import dataclasses
import importlib.metadata
import inspect
import io
import json
import math
import os
import platform
import sys
import tempfile
import time
import traceback
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
SRC = WS / "src"
for p in (SRC, HERE, WS / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from allen_diameter.config import default_config  # noqa: E402

SEED = 20261013
REPORT_PACKAGES = ("numpy", "scipy", "matplotlib")
NAN = float("nan")
CASES = [("gauss", None), ("lorentz", None), ("voigt", 0.01), ("voigt", 1.0), ("voigt", 100.0), ("student", 1.0),
         ("student", 3.0), ("student", 100.0), ("gennorm", 0.5), ("gennorm", 1.0), ("gennorm", 4.0),
         ("gennorm", 8.0), ("skewnorm", -10.0), ("skewnorm", -1.0), ("skewnorm", 0.0), ("skewnorm", 3.0),
         ("skewnorm", 10.0)]


def _measure():
    return default_config().measure


def _profile(cfg, d, mu, v0, g, B=180.0, phi=0.2, noise=0.0, rng=None, n_nodes=2000):
    """Block 5's offsets and a profile made with kernel g (noise-free unless noise > 0)."""
    from allen_diameter.analysis import profiles
    from allen_diameter.model import tube_model as tm
    v = profiles.profile_offsets(cfg)
    I = tm.model_profile_kernel(v, d, mu * d / math.cos(phi), v0, g, B, n_nodes)
    if noise > 0:
        I = I + rng.normal(0.0, noise, v.size)
    return v, I, phi, B


def _quad_model(v, d, alpha, v0, g, B):
    """B (1 - int (1 - T(u)) g(v - v0 - u) du) by scipy.integrate.quad, the kernel's peak as a break point."""
    from scipy import integrate
    from allen_diameter.model import tube_model as tm
    out = []
    for vi in np.atleast_1d(v):
        f = lambda u: (1.0 - float(tm.transmittance(u, d, alpha, 0.0))) * float(g(vi - v0 - u))  # noqa: E731
        pts = [x for x in (vi - v0,) if -0.5 * d < x < 0.5 * d]
        val = integrate.quad(f, -0.5 * d, 0.5 * d, points=pts or None, limit=400, epsabs=1e-13, epsrel=1e-12)[0]
        out.append(B * (1.0 - val))
    return np.array(out)


def _texts_inside(fig):
    """Every figure-level text, legend entry, axes title and axes text lies inside the figure, to the pixel."""
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    texts = list(fig.texts) + [ax.title for ax in fig.axes]
    texts += [t for leg in fig.legends for t in leg.get_texts()]
    for ax in fig.axes:
        texts += list(ax.texts)
    for t in texts:
        if t.get_text():
            bb = t.get_window_extent(renderer=ren)
            assert bb.x0 >= -1.0 and bb.x1 <= fig.bbox.width + 1.0 and bb.y0 >= -1.0 and \
                bb.y1 <= fig.bbox.height + 1.0, (t.get_text()[:60], bb.x0, bb.x1, bb.y0, bb.y1)


def _cell4j(tmp, d_true=0.8, nodes=(3, 4, 1)):
    """The synthetic cell (fixtures_cell) and Cell 4j's records at its default m = 3, read back from its JSON."""
    import iteration_fits as IF
    import plane_differences as PD
    import run_cell
    from fixtures_cell import synthetic_cell
    base = default_config()
    cfg = dataclasses.replace(base, measure=dataclasses.replace(base.measure, block_half_um=3.5))
    swc, fetcher, planes, swc_path = synthetic_cell(tmp, cfg, d_true=d_true, mu=1.0)
    prov = run_cell.real_provider(fetcher, planes, cfg.acquisition.res0_um)
    out = os.path.join(tmp, "iter")
    PD.run(swc, prov, cfg, list(nodes), out, "999", planes_half=6, evaluation="iterate", log=lambda m: None)
    path = os.path.join(out, "planeiter_999.json")
    return cfg, swc, prov, swc_path, IF.read_records(path), path


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    from scipy import integrate, optimize, special, stats
    from allen_diameter.analysis import fit_variants as FV
    from allen_diameter.model import blur_kernels as BK
    w = 0.2331
    for fam, s in CASES:
        g = BK.kernel_pdf(fam, w, s)
        total = sum(integrate.quad(g, lo, hi, limit=500, epsabs=1e-14, epsrel=1e-12)[0]
                    for lo, hi in ((-np.inf, -w), (-w, 0.0), (0.0, w), (w, np.inf)))
        assert abs(total - 1.0) <= 1e-9, (fam, s, total)
        g0 = float(g(0.0))
        for h in (1e-4, 1e-3, 1e-2):                 # the maximum at 0
            assert float(g(-h)) <= g0 and float(g(h)) <= g0, (fam, s, h)
        # the half-maximum points found on g itself (brentq), independently of the module's routine
        left = optimize.brentq(lambda z: float(g(z)) - 0.5 * g0, -50 * w, 0.0, xtol=1e-15)
        right = optimize.brentq(lambda z: float(g(z)) - 0.5 * g0, 0.0, 50 * w, xtol=1e-15)
        assert abs((right - left) - w) <= 1e-9 * w, (fam, s, right - left)
    # closed forms against the numeric FWHM of the unit densities
    for fam, s in [("gauss", None), ("lorentz", None), ("student", 1.0), ("student", 7.5), ("gennorm", 0.5),
                   ("gennorm", 3.3)]:
        num = BK._numeric_fwhm(lambda x: float(BK.unit_pdf(fam, s, x)), 0.0)
        assert abs(num - BK.unit_fwhm(fam, s)) <= 1e-10, (fam, s, num, BK.unit_fwhm(fam, s))
    # the Voigt's r is the ratio of the Lorentzian's FWHM to the Gaussian's (FWHM 1 at unit scale): the unit density
    # is the convolution of those two, done here by quad
    sig = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    for r in (0.3, 1.0, 4.0):
        for z in (0.0, 0.37, -1.4):
            ref = integrate.quad(lambda x: stats.norm.pdf(z - x, scale=sig) * stats.cauchy.pdf(x, scale=0.5 * r),
                                 -np.inf, np.inf, limit=500, epsabs=1e-14)[0]
            assert abs(float(BK.unit_pdf("voigt", r, z)) - ref) <= 1e-9, (r, z, float(BK.unit_pdf("voigt", r, z)), ref)
    # the centre weights: exp(-|v - v0| / lambda)
    wts = FV.centre_weights([-0.45, 0.05, 0.55, 1.05], 0.05, 0.25)
    assert np.allclose(wts, [math.exp(-2.0), 1.0, math.exp(-2.0), math.exp(-4.0)], rtol=1e-15, atol=0), wts
    # the skew-normal's mode: a root of d/dz ln f; odd in a
    for a in (0.3, 1.0, 4.0, 10.0):
        m = BK.unit_mode("skewnorm", a)
        dl = -m + a * math.exp(stats.norm.logpdf(a * m) - special.log_ndtr(a * m))
        assert abs(dl) <= 1e-12 and BK.unit_mode("skewnorm", -a) == -m and 0.0 < m < 1.0, (a, m, dl)
    # fitted back: noise-free profiles made with each fixed kernel (the fit's own quadrature, so exact)
    cfg = _measure()
    wf = FV.fixed_fwhm_um(cfg)
    for kern in FV.KERNEL_CONFIGS[:6]:
        fam, s, _ = kern
        spec = FV.VariantSpec(fam, s, False, False)
        v, I, phi, B = _profile(cfg, 0.9, 1.2, 0.07, BK.kernel_pdf(fam, wf, s), n_nodes=FV.n_nodes_for(spec, cfg))
        u = FV.fit_unweighted(v, I, phi, B, cfg, spec)
        assert abs(u.d_um - 0.9) <= 1e-6 and abs(u.v0_um - 0.07) <= 1e-6 and u.status == "converged", (kern, u)
        f = FV.fit_weighted(v, I, phi, B, cfg, FV.VariantSpec(fam, s, False, False, 0.5), u)
        assert abs(f.d_um - 0.9) <= 1e-6 and f.irls_ok and f.n_irls <= 3, (kern, f.d_um, f.n_irls)
    # width and shape fitted: a Student kernel of nu = 6 and FWHM 0.45 um, noise-free, comes back
    spec = FV.VariantSpec("student", None, True, True)
    v, I, phi, B = _profile(cfg, 1.1, 0.9, -0.05, BK.kernel_pdf("student", 0.45, 6.0),
                            n_nodes=FV.n_nodes_for(spec, cfg))
    f = FV.fit_unweighted(v, I, phi, B, cfg, spec)
    assert abs(f.d_um - 1.1) <= 1e-4 and abs(f.fwhm_um - 0.45) <= 1e-4 and abs(f.shape - 6.0) <= 1e-2, \
        (f.d_um, f.fwhm_um, f.shape, f.status)
    spec = FV.VariantSpec("skewnorm", None, True, False)
    v, I, phi, B = _profile(cfg, 0.6, 1.5, 0.1, BK.kernel_pdf("gauss", 0.5))
    f = FV.fit_unweighted(v, I, phi, B, cfg, spec)       # a profile blurred twice as wide: the skew runs to a bound
    assert f.status == "at_bound" and f.at_bound in (("s_lo",), ("s_hi",)) and abs(abs(f.shape) - 10.0) <= 1e-6, \
        (f.status, f.at_bound, f.shape)
    spec = FV.VariantSpec("skewnorm", None, True, False)            # a skew at the fixed width
    v, I, phi, B = _profile(cfg, 0.7, 1.5, 0.1, BK.kernel_pdf("skewnorm", wf, -2.5), n_nodes=FV.n_nodes_for(spec, cfg))
    f = FV.fit_unweighted(v, I, phi, B, cfg, spec)
    assert abs(f.d_um - 0.7) <= 1e-4 and abs(f.shape + 2.5) <= 1e-3, (f.d_um, f.shape, f.status)


def test_reference():
    from allen_diameter.analysis import fit as block5
    from allen_diameter.analysis import fit_variants as FV
    from allen_diameter.model import blur_kernels as BK
    from allen_diameter.model import tube_model as tm
    cfg = _measure()
    v = np.linspace(-3.0, 3.0, 53)
    for d, al in ((0.3, 0.2), (0.8, 1.5), (3.0, 6.0), (6.0, 0.5)):
        a = tm.model_profile(v, d, al, 0.13, 0.099, 200.0, 364)
        b = tm.model_profile_kernel(v, d, al, 0.13, BK.kernel_pdf("gauss", BK.gauss_fwhm(0.099)), 200.0, 364)
        assert np.max(np.abs(a - b)) <= 1e-11, (d, np.max(np.abs(a - b)))
    vv = np.array([-1.3, -0.41, 0.0, 0.17, 0.9])
    for fam, s in [("gauss", None), ("lorentz", None), ("voigt", 1.0), ("student", 3.0), ("gennorm", 4.0),
                   ("skewnorm", 3.0), ("gennorm", 1.0), ("gennorm", 0.5)]:
        for wd in (0.1, 0.2331, 2.0):
            g = BK.kernel_pdf(fam, wd, s)
            n = tm.n_quadrature_nodes(6.0, min(wd, 0.2331) / BK.gauss_fwhm(1.0), 64, 6.0)
            for d, al in ((0.3, 0.5), (2.0, 4.0)):
                got = tm.model_profile_kernel(vv, d, al, 0.07, g, 200.0, n)
                ref = _quad_model(vv, d, al, 0.07, g, 200.0)
                tol = 0.5 if (fam == "gennorm" and s <= 1.0) else 1e-7      # the cusp: slow convergence, gl
                assert np.max(np.abs(got - ref)) <= tol, (fam, s, wd, d, np.max(np.abs(got - ref)))
    rng = np.random.default_rng(SEED)
    for d_true in (0.6, 2.5):
        v, I, phi, B = _profile(cfg, d_true, 1.0, 0.1, BK.kernel_pdf("gauss", 0.5), noise=2.0, rng=rng)
        b5 = block5.fit_profile(v, I, phi, B, cfg)
        fits, base = FV.fit_grid(v, I, phi, B, cfg, FV.variant_grid(FV.KERNEL_CONFIGS[:1], ("fixed",), (0.5,)))
        assert fits[0] is base and base.spec.key == "gauss|w fixed|none"
        for a, b in ((base.d_um, b5.d_hat_um), (base.mu_per_um, b5.mu_hat_per_um), (base.v0_um, b5.v0_hat_um)):
            assert abs(a - b) <= 1e-6, (a, b)
        assert base.status == b5.status and abs(base.rss_gl2 - b5.rss_gl2) <= 1e-6 * b5.rss_gl2
        # IRLS fixed point: one more re-weighted step from the result moves d and v0 by <= the tolerance
        f = fits[1]
        again = FV.fit_weighted(v, I, phi, B, cfg, f.spec, f, max_iter=1)
        assert abs(again.d_um - f.d_um) <= FV.IRLS_TOL_UM and abs(again.v0_um - f.v0_um) <= FV.IRLS_TOL_UM
        assert np.array_equal(FV.variant_curve(f, v, phi, B), f.model_gl)
    # a darker neighbour 1 um away merges with the tube: the unweighted fit takes both as one (v0 near 0.6 um) and the
    # weights re-centred on the fit's own v0 move it (to the neighbour at lambda 0.25 um). The weighted fit is a
    # fixed point of its own weights, checked by a weighted least squares written here (weights
    # exp(-|v - v0| / lambda) at its v0, from its theta); weights left at the start's v0 would fail it
    from scipy.optimize import least_squares
    gk = BK.kernel_pdf("gauss", FV.fixed_fwhm_um(cfg))
    v, I1, phi, B = _profile(cfg, 0.8, 1.2, 0.0, gk, noise=0.5, rng=rng)
    _v, I2, _p, _b = _profile(cfg, 1.0, 1.5, 1.0, gk)
    I = I1 + I2 - B                                       # two dips on one background
    for lam in (0.25, 0.5):
        spec = FV.VariantSpec("gauss", None, False, False, lam)
        fits, base = FV.fit_grid(v, I, phi, B, cfg, [spec])
        f = fits[0]
        assert f.irls_ok and abs(f.v0_um - base.v0_um) >= 0.02, (lam, f.v0_um, base.v0_um, f.irls_ok)
        sw = np.sqrt(np.exp(-np.abs(v - f.v0_um) / lam))

        def res(p):
            al = p[1] * p[0] / math.cos(phi)
            return sw * (tm.model_profile_kernel(v, p[0], al, p[2], gk, B, f.n_nodes) - I) / B
        r = least_squares(res, [f.d_um, f.mu_per_um, f.v0_um], method="trf", x_scale="jac", ftol=1e-12, xtol=1e-12,
                          gtol=1e-12, bounds=([0.05, 0.0, -1.0], [6.0, 20.0, 1.0]))
        assert abs(r.x[0] - f.d_um) <= FV.IRLS_TOL_UM and abs(r.x[2] - f.v0_um) <= FV.IRLS_TOL_UM, (lam, r.x, f.d_um,
                                                                                                    f.v0_um)
        r2 = (f.model_gl - I) ** 2                         # rss_w with the last iteration's weights (its v0 to 1e-4)
        assert abs(f.rss_gl2 - np.sum(r2)) <= 1e-9 * np.sum(r2)
        om = np.exp(-np.abs(v - f.v0_um) / lam)
        assert abs(f.rss_w_gl2 - np.sum(om * r2)) <= 1e-3 * np.sum(om * r2), (f.rss_w_gl2, np.sum(om * r2))


def test_convergence():
    from allen_diameter.analysis import fit_variants as FV
    from allen_diameter.model import blur_kernels as BK
    from allen_diameter.model import tube_model as tm
    v = np.linspace(-3.0, 3.0, 53)
    g = BK.kernel_pdf("student", 0.2331, 3.0)
    ref = tm.model_profile_kernel(v, 2.0, 1.5, 0.05, g, 200.0, 4 * 364)
    errs = [np.max(np.abs(tm.model_profile_kernel(v, 2.0, 1.5, 0.05, g, 200.0, n) - ref)) for n in (64, 128, 364)]
    assert errs[0] > errs[1] > errs[2] and errs[2] <= 1e-9, errs
    cfg = _measure()
    rng = np.random.default_rng(SEED + 1)
    spec = FV.VariantSpec("gennorm", 1.0, False, False)
    for d_true in (0.6, 2.0):
        v, I, phi, B = _profile(cfg, d_true, 1.0, 0.1, BK.kernel_pdf("gennorm", 0.4, 1.0), noise=2.0, rng=rng,
                                n_nodes=4000)
        f1 = FV.fit_unweighted(v, I, phi, B, cfg, spec)
        saved = FV.n_nodes_for
        try:
            FV.n_nodes_for = lambda s, c: 3 * saved(s, c)
            f3 = FV.fit_unweighted(v, I, phi, B, cfg, spec)
        finally:
            FV.n_nodes_for = saved
        assert f3.n_nodes == 3 * f1.n_nodes and abs(f3.d_um - f1.d_um) <= 0.005, (d_true, f1.d_um, f3.d_um)


def test_invariants():
    from allen_diameter.analysis import fit_variants as FV
    from allen_diameter.model import blur_kernels as BK
    cfg = _measure()
    rng = np.random.default_rng(SEED + 2)
    v, I, phi, B = _profile(cfg, 0.8, 1.2, 0.05, BK.kernel_pdf("voigt", 0.4, 1.0), noise=1.5, rng=rng)
    specs = [FV.VariantSpec("gauss", None, False, False, None), FV.VariantSpec("gauss", None, False, False, 0.5),
             FV.VariantSpec("lorentz", None, False, True, None), FV.VariantSpec("skewnorm", None, True, False, None),
             FV.VariantSpec("student", None, True, True, 1.0)]
    fits, base = FV.fit_grid(v, I, phi, B, cfg, specs)
    gain, _b = FV.fit_grid(v, I * 0.6, phi, B * 0.6, cfg, specs)
    for a, b in zip(fits, gain):
        for x, y in ((a.d_um, b.d_um), (a.mu_per_um, b.mu_per_um), (a.v0_um, b.v0_um), (a.fwhm_um, b.fwhm_um)):
            assert abs(x - y) <= 1e-6, (a.spec.key, x, y)
        assert (a.shape is None and b.shape is None) or abs(a.shape - b.shape) <= 1e-5 * max(1.0, abs(a.shape))
    shifted, _b = FV.fit_grid(v + 0.15, I, phi, B, cfg, specs)
    for a, b in zip(fits, shifted):
        assert abs(b.v0_um - (a.v0_um + 0.15)) <= 1e-6 and abs(b.d_um - a.d_um) <= 1e-6, (a.spec.key, a.d_um, b.d_um)
    mirrored, _b = FV.fit_grid(-v[::-1], I[::-1], phi, B, cfg, specs)
    for a, b in zip(fits, mirrored):
        assert abs(b.v0_um + a.v0_um) <= 1e-5 and abs(b.d_um - a.d_um) <= 1e-5, (a.spec.key, a.d_um, b.d_um)
        if a.spec.family == "skewnorm":
            assert abs(b.shape + a.shape) <= 1e-3, (a.shape, b.shape)
    w = FV.centre_weights(v, 0.05, 0.5)
    assert abs(FV.centre_weights([0.05], 0.05, 0.5)[0] - 1.0) == 0.0 and np.all(w <= 1.0)
    assert np.allclose(FV.centre_weights(0.05 + v, 0.05, 0.5), FV.centre_weights(0.05 - v, 0.05, 0.5), rtol=0,
                       atol=1e-15)                       # |(0.05 +- v) - 0.05| differ by roundoff only
    assert FV.variant_row(base, base)["ratio"] == 1.0


def test_contract():
    import fit_variants as FVS
    from allen_diameter.analysis import fit_variants as FV
    from allen_diameter.model import blur_kernels as BK
    from allen_diameter.model import tube_model as tm
    bad_specs = [FV.VariantSpec("cosine", None, False, False), FV.VariantSpec("skewnorm", 2.0, False, False),
                 FV.VariantSpec("gauss", 1.0, False, False), FV.VariantSpec("voigt", 1.0, True, False),
                 FV.VariantSpec("lorentz", None, True, False), FV.VariantSpec("gennorm", 9.0, False, False),
                 FV.VariantSpec("gauss", None, False, False, 0.0), FV.VariantSpec("gauss", None, False, False, NAN)]
    for s in bad_specs:
        _refuses(FV.check_spec, s)
    for args in (("gauss", 0.0), ("gauss", NAN), ("voigt", 0.2, None), ("student", 0.2, 0.5), ("cosine", 0.2)):
        _refuses(BK.kernel_pdf, *args)
    g = BK.kernel_pdf("gauss", 0.2331)
    _refuses(tm.model_profile_kernel, [0.0], 1.0, 1.0, 0.0, "not a kernel", 100.0, 64, exc=TypeError)
    _refuses(tm.model_profile_kernel, [0.0], 1.0, 1.0, 0.0, g, 0.0, 64)
    _refuses(tm.model_profile_kernel, [NAN], 1.0, 1.0, 0.0, g, 100.0, 64)
    grid = FV.variant_grid()
    assert len(grid) == 100 and len({s.key for s in grid}) == 100
    assert grid[0].key == "gauss|w fixed|none" and grid[1].key == "gauss|w fixed|lam=0.25"
    assert grid[5].key == "gauss|w fitted|none" and grid[-1].key == "skewnorm a fit|w fitted|lam=2"
    assert len(FV.variant_grid(lambdas_um=())) == 20
    cfg = _measure()
    v, I, phi, B = _profile(cfg, 0.8, 1.0, 0.0, g)
    u = FV.fit_unweighted(v, I, phi, B, cfg, grid[0])
    _refuses(FV.fit_weighted, v, I, phi, B, cfg, grid[0], u)
    nan_start = dataclasses.replace(u, d_um=NAN)
    f = FV.fit_weighted(v, I, phi, B, cfg, grid[1], nan_start)
    assert f.status == "failed" and math.isnan(f.d_um) and f.irls_ok is False and np.all(np.isnan(f.model_gl))
    assert FVS.flag_of(f) == "x" and FVS.flag_of(u) == ""
    assert FVS.flag_of(dataclasses.replace(u, status="at_bound")) == "*"
    assert FVS.flag_of(dataclasses.replace(u, irls_ok=False)) == "~"
    assert FVS.flag_of(dataclasses.replace(u, status="at_bound", irls_ok=False)) == "*~"
    specs = FVS.select_specs(["lorentz"], [0.5], ["fixed"])
    assert [s.key for s in specs] == ["gauss|w fixed|none", "gauss|w fixed|lam=0.5", "lorentz|w fixed|none",
                                      "lorentz|w fixed|lam=0.5"], [s.key for s in specs]
    _refuses(FVS.select_specs, ["lorentzian"])
    assert FV.n_nodes_for(grid[0], cfg) == tm.n_quadrature_nodes(6.0, cfg.sigma_fit_um, 64, 6.0) == 364
    assert FV.n_nodes_for(grid[5], cfg) == tm.n_quadrature_nodes(6.0, 0.1 / BK.gauss_fwhm(1.0), 64, 6.0) == 848
    other = dataclasses.replace(u, d_um=0.6)
    assert FV.variant_row(other, u)["ratio"] == 0.6 / u.d_um
    # IRLS stops when d AND v0 both moved by <= 1e-4 um: runs scripted so that v0 settles first, d two steps later
    class _Run:
        def __init__(self, x):
            self.x, self.status, self.cost, self.nfev, self.message = np.array(x, dtype=float), 1, 0.0, 1, "scripted"
    seq = [(1.0, 1.0, 0.10), (0.9, 1.0, 0.20), (0.8, 1.0, 0.20), (0.75, 1.0, 0.20), (0.75, 1.0, 0.20),
           (0.75, 1.0, 0.20)]
    calls = []

    def scripted(*a, **k):                                # the start is seq[0]; run t returns seq[t]
        calls.append(1)
        return [_Run(seq[len(calls)])]
    saved = FV._runs
    try:
        FV._runs = scripted
        f = FV.fit_weighted(v, I, phi, B, cfg, grid[1], dataclasses.replace(u, d_um=1.0, mu_per_um=1.0, v0_um=0.1))
    finally:
        FV._runs = saved
    assert f.n_irls == 4 and f.irls_ok and f.d_um == 0.75, (f.n_irls, f.irls_ok, f.d_um)
    for x in (u.d_um, u.mu_per_um, u.v0_um, u.fwhm_um, u.alpha_hat, u.rss_gl2):
        assert math.isfinite(x)
    assert u.model_gl.shape == v.shape and u.n_irls == 0 and u.irls_ok is None and u.shape is None


def _refuses(func, *args, exc=ValueError):
    try:
        func(*args)
    except exc:
        return
    raise AssertionError("%s accepted %r" % (func.__name__, args))


def test_wiring():
    from allen_diameter.analysis import fit_variants as FV
    from allen_diameter.model import blur_kernels as BK
    cfg = _measure()
    v, I, phi, B = _profile(cfg, 0.7, 1.2, 0.0, BK.kernel_pdf("student", 0.4, 4.0), noise=1.5,
                            rng=np.random.default_rng(SEED + 4))
    specs = [FV.VariantSpec("gennorm", None, True, False, None), FV.VariantSpec("gennorm", None, True, True, 0.5),
             FV.VariantSpec("lorentz", None, False, True, 1.0)]
    fits, base = FV.fit_grid(v, I, phi, B, cfg, specs)
    # starts: Block 5's 3, then the warm ones (a fitted shape at the fixed width: the family's fixed members and the
    # Gaussian; a fitted width: the same kernel at the fixed width)
    assert fits[0].n_starts == 3 + 2 + 1, fits[0].n_starts
    unw = {}
    FV.fit_grid(v, I, phi, B, cfg, specs, log=lambda s, f: unw.setdefault(s.key, f))
    assert unw["gennorm beta fit|w fitted|none"].n_starts == 3 + 1 and unw["lorentz|w fitted|none"].n_starts == 4
    assert unw["gennorm beta=1|w fixed|none"].n_starts == 3 and base.n_starts == 3
    # two weightings of one kernel are kept apart: each is fit_weighted from the kernel's unweighted fit
    pf = FV.PlaneFits(v, I, phi, B, cfg)
    lam1, lam2 = FV.VariantSpec("student", 3.0, False, True, 0.25), FV.VariantSpec("student", 3.0, False, True, 2.0)
    f1, f2 = pf.fit(lam1), pf.fit(lam2)
    u = pf.unweighted(lam1.kernel, True)
    for spec, f in ((lam1, f1), (lam2, f2)):
        ref = FV.fit_weighted(v, I, phi, B, cfg, spec, u)
        assert FV.variant_row(ref, u) == FV.variant_row(f, u) and pf.fit(spec) is f, spec.key
    # a weighted fit is fit_weighted from the unweighted fit of the same kernel and width mode
    for f in fits[1:]:
        ref = FV.fit_weighted(v, I, phi, B, cfg, f.spec, unw[f.spec.unweighted().key])
        assert FV.variant_row(ref, base) == FV.variant_row(f, base), f.spec.key


def test_determinism():
    from allen_diameter.analysis import fit_variants as FV
    from allen_diameter.model import blur_kernels as BK
    cfg = _measure()
    v, I, phi, B = _profile(cfg, 1.2, 0.8, -0.1, BK.kernel_pdf("lorentz", 0.35), noise=2.0,
                            rng=np.random.default_rng(SEED + 3))
    specs = FV.variant_grid(FV.KERNEL_CONFIGS[1:2] + FV.KERNEL_CONFIGS[9:], ("fixed", "fitted"), (0.5,))
    a, _ = FV.fit_grid(v, I, phi, B, cfg, specs)
    b, _ = FV.fit_grid(v, I, phi, B, cfg, specs)
    for x, y in zip(a, b):
        assert FV.variant_row(x, a[0]) == FV.variant_row(y, b[0]) and np.array_equal(x.model_gl, y.model_gl)


def test_edge_cases():
    import matplotlib.pyplot as plt
    import fit_variants as FVS
    from allen_diameter.analysis import fit_variants as FV
    from allen_diameter.plotting import figures as FG
    with tempfile.TemporaryDirectory() as tmp:
        cfg, swc, prov, swc_path, recs, path = _cell4j(tmp, 0.8, nodes=(3, 4, 1))
        out_dir = os.path.join(tmp, "fitvar")
        logs = []
        specs = FVS.select_specs(["lorentz", "skewnorm"], [0.5], ["fixed", "fitted"])
        out = FVS.run(swc, prov, cfg, recs, out_dir, "999", specs=specs, log=logs.append)
        assert [x["node_id"] for x in out] == [3, 4, 1] and out[2]["skipped"].startswith("Cell 4j: "), out[2]
        keys = {"node_id", "k_iter", "k_star", "k_swc", "d_iter_um", "d_block5_um", "d_base_um", "phi_rad", "B_bar",
                "v_um", "profile", "variants", "notes", "png", "seconds"}
        for x, r in zip(out[:2], recs[:2]):
            assert set(x) == keys, set(x) ^ keys
            assert x["d_base_um"] == x["variants"][0]["d_um"] and abs(x["d_base_um"] - r["d_iter_um"]) <= 1e-6
            assert x["d_block5_um"] == r["d_iter_um"] and x["notes"] == [], x["notes"]
            assert len(x["variants"]) == len(specs) == 12 and os.path.exists(x["png"])
        with open(os.path.join(out_dir, "fitvar_999.csv")) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2 * len(specs) and rows[0]["variant"] == "gauss|w fixed|none" and rows[0]["ratio"] == "1.0"
        with open(os.path.join(out_dir, "fitvar_999.json")) as f:
            assert len(json.load(f)) == 3
        assert os.path.exists(os.path.join(out_dir, "fitvar_summary_999.png"))
        assert any("the baseline equals Cell 4j's d_iter on 2 of 2" in m for m in logs), logs[-1:]
        assert sum(1 for m in logs if m.startswith("[fitvar] node 3: k_iter")) == 1
        # the full grid's node figure and a summary of many nodes: every text inside
        st, fit5, notes = FVS.node_inputs(swc, prov, cfg, recs[1])
        _s, _f, notes2 = FVS.node_inputs(swc, prov, cfg, dict(recs[1], d_iter_um=recs[1]["d_iter_um"] + 0.01))
        assert notes == [] and len(notes2) == 1 and notes2[0].startswith("Block 5 at k_iter gives d"), notes2
        for bad in (dict(recs[1], k_iter=None), dict(recs[1], k_iter=99)):
            _refuses(FVS.node_inputs, swc, prov, cfg, bad)
        v, I = np.asarray(fit5["v_um"]), np.asarray(fit5["profile"])
        grid = FV.variant_grid()
        fits, base = FV.fit_grid(v, I, fit5["phi_rad"], fit5["B_bar"], cfg.measure, grid)
        blk = FVS.node_block("node 4: " + "a long label " * 12, "the profile " * 40, fit5, fits, base, grid, cfg)
        assert len(blk["table"]["rows"]) == 10 and len(blk["table"]["cols"]) == 10 and blk["table"]["base"] == (0, 0)
        for mode, c0 in (("fixed", 0), ("fitted", 5)):          # the least residual among the unweighted fits
            unw = [f for f in fits if f.spec.lam_um is None and f.spec.width_mode == mode and f.status != "failed"]
            want = min(unw, key=lambda f: f.rss_gl2)
            r = [x[1] for x in FVS.kernel_rows(grid)].index(want.spec.kernel_label)
            assert (r, c0) in blk["table"]["best"] and FVS.least_rss(fits)[mode] is want, (mode, blk["table"]["best"])
        gfix = [f for f in fits if f.spec.kernel == FV.BASE_KERNEL and not f.spec.width_fitted]
        assert [x["lam"] for x in blk["lam"]] == [None, 0.25, 0.5, 1.0, 2.0]
        assert [x["d"] for x in blk["lam"]] == [f.d_um for f in gfix]
        fig = FG.fit_variants_node_figure(blk, "a test title")
        _texts_inside(fig)
        segs = sorted(tuple(ln.get_xdata()) for ln in fig.axes[0].get_lines()
                      if len(ln.get_xdata()) == 2 and ln.get_ydata()[0] == ln.get_ydata()[1])
        want = sorted((x["v0"] - 0.5 * x["d"], x["v0"] + 0.5 * x["d"]) for x in blk["fixed"])
        assert np.allclose(segs, want, rtol=0, atol=1e-12), (segs[:2], want[:2])
        plt.close(fig)
        rec = dict(node_id=4, d_base=base.d_um, ratio={(f.spec.kernel_label, f.spec.width_mode, f.spec.weight_label):
                                                       FV.variant_row(f, base)["ratio"] for f in fits},
                   flag={(f.spec.kernel_label, f.spec.width_mode, f.spec.weight_label): FVS.flag_of(f) for f in fits})
        many = [dict(rec, node_id=i, d_base=base.d_um * (1.3 ** i)) for i in range(8)]
        fig = FG.fit_variants_summary_figure(many, [x[1] for x in FVS.kernel_rows(grid)], "a summary title")
        assert len([a for a in fig.axes if a.axison]) == 20
        _texts_inside(fig)
        plt.close(fig)
        saved = FVS.FV.fit_grid
        try:
            def shifted(*a, **k):
                fits, base = saved(*a, **k)
                return fits, dataclasses.replace(base, d_um=base.d_um + 1e-3)
            FVS.FV.fit_grid = shifted
            out2 = FVS.run(swc, prov, cfg, recs, os.path.join(tmp, "f2"), "999", nodes=[4], specs=specs[:2],
                           log=lambda m: None)
        finally:
            FVS.FV.fit_grid = saved
        assert any(n.startswith("the variants' baseline d") for n in out2[0]["notes"]), out2[0]["notes"]
        _check_cli(FVS, swc_path, path, tmp)


def _check_cli(FVS, swc_path, iter_json, tmp):
    import allen_image_io as aio
    import run_cell

    class _Fetcher:
        def __init__(self, cache_dir="", verbose=True):
            self.n_cache_hits, self.n_requests, self.bytes_downloaded = 0, 0, 0
    saved = (aio.HttpFetcher, aio.list_images, aio.plane_table, run_cell.real_provider, FVS.run)
    calls = []
    try:
        aio.HttpFetcher, aio.list_images, aio.plane_table = _Fetcher, (lambda specimen: None), (lambda images: None)
        run_cell.real_provider = lambda *x, **k: None
        FVS.run = lambda *x, **k: calls.append((x, k)) or []
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            assert FVS.main(["--specimen", "999", "--iter-json", iter_json, "--out-dir", os.path.join(tmp, "cli"),
                             "--swc", swc_path, "--nodes", "4,3", "--families", "voigt,gennorm", "--lambdas", "0.5,2",
                             "--width-modes", "fitted", "--half-um", "3.5", "--shift-x", "2.5", "--z0", "1.0"]) == 0
            for bad in (["--width-modes", "free"], ["--lambdas", "-1"], ["--families", "cosine"], ["--nodes", "a"]):
                try:
                    FVS.main(["--specimen", "999", "--iter-json", iter_json, "--out-dir", tmp, "--swc", swc_path]
                             + bad)
                except SystemExit as e:
                    assert e.code == 2, bad
                else:
                    raise AssertionError("%r accepted" % (bad,))
    finally:
        aio.HttpFetcher, aio.list_images, aio.plane_table, run_cell.real_provider, FVS.run = saved
    (x, k), = calls
    got = inspect.signature(FVS.run).bind(*x, **k).arguments
    assert got["nodes"] == [4, 3] and got["half_um"] == 3.5 and got["specimen"] == "999", got
    assert got["transform"]["shift_full_px"] == (2.5, 0.0) and got["transform"]["z0_um"] == 1.0, got["transform"]
    fams = sorted({s.family for s in got["specs"]})
    assert fams == ["gauss", "gennorm", "voigt"] and {s.width_mode for s in got["specs"]} == {"fitted"}
    assert sorted({s.lam_um for s in got["specs"] if s.lam_um is not None}) == [0.5, 2.0]
    assert len(got["specs"]) == 6 * 3, len(got["specs"])        # gauss, voigt x 2, gennorm x 3; 3 weightings


# ---------------------------------------------------------------- runner ---

def _environment():
    parts = ["python %s" % platform.python_version()]
    for package in REPORT_PACKAGES:
        try:
            parts.append("%s %s" % (package, importlib.metadata.version(package)))
        except importlib.metadata.PackageNotFoundError:
            parts.append("%s (not installed)" % package)
    parts.append(platform.platform())
    parts.append("seed %d" % SEED)
    return " | ".join(parts)


def main():
    checks = [(name, obj) for name, obj in globals().items()
              if name.startswith("test_") and callable(obj)]
    print("== %s" % Path(__file__).name)
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
        except Exception:
            status, detail = "ERROR", traceback.format_exc()
        results.append((name, status, time.perf_counter() - start, detail))
    width = max([len(n) for n, _, _, _ in results] + [4])
    for name, status, seconds, detail in results:
        lines = detail.strip().splitlines()
        headline = lines[0] if lines else ""
        print(("%-5s  %-" + str(width) + "s  %8.3fs  %s") % (status, name, seconds, headline))
    for name, status, _s, detail in results:
        if status in ("FAIL", "ERROR"):
            print("\n---- %s: %s\n%s" % (status, name, detail.strip()))
    counts = {s: sum(1 for r in results if r[1] == s) for s in ("PASS", "FAIL", "ERROR", "TODO", "SKIP")}
    print("\n-- " + ", ".join("%d %s" % (n, s.lower()) for s, n in counts.items()))
    return 1 if (counts["FAIL"] + counts["ERROR"] + counts["TODO"]) else 0


if __name__ == "__main__":
    sys.exit(main())
