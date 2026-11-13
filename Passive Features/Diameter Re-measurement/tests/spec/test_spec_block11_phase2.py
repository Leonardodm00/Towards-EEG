"""Spec-derived tests of Block 11 (Phase II tools), integration tester run D2,
2026-11-13, commit 1555036. Written from specs/SPEC.md (Block 11, sections
2.2, 4, 6-8 and the neighbour entries), not from the code: the oracle IDs
(K, M, F, I, U, V, P) are those of the report
specs/test_reports/2026-11-13_0830_incremental_D2.md.

Run from `Passive Features/Diameter Re-measurement/`:
    python -m pytest -q tests/spec/test_spec_block11_phase2.py
Pure ASCII.
"""
from __future__ import annotations

import dataclasses
import math
import types

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import integrate, ndimage

import spec_helpers_D2 as H

from allen_diameter.analysis import camera_fit, cell, focus, profiles, registration, survey  # noqa: E402
from allen_diameter.analysis import fit as block5  # noqa: E402
from allen_diameter.analysis import fit_variants as FV  # noqa: E402
from allen_diameter.analysis import plane_iteration as PI  # noqa: E402
from allen_diameter.model import blur_kernels as BK  # noqa: E402
from allen_diameter.model import tube_model  # noqa: E402

CFG = H.default_cfg()
M = CFG.measure

SHAPES = {"voigt": (0.01, 0.05, 1.0, 100.0), "student": (1.0, 3.0, 30.0, 100.0),
          "gennorm": (0.5, 1.0, 2.0, 4.0, 8.0), "skewnorm": (-10.0, -2.5, 0.0, 2.5, 10.0)}


def _cases():
    out = [("gauss", None), ("lorentz", None)]
    for fam, ss in SHAPES.items():
        out += [(fam, s) for s in ss]
    return out


def _nan_equal(a, b):
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)
    return a == b


# =============================================================== K: blur kernels ===

@pytest.mark.parametrize("family,shape", _cases())
@pytest.mark.parametrize("w", (0.1, 0.2331, 2.0))
def test_K1_K2_K3_unit_integral_fwhm_mode(family, shape, w):
    """K1-K3: unit integral, FWHM w, maximum at z = 0, for every family at its bounds and inside."""
    g = BK.kernel_pdf(family, w, shape)
    # K1: quad on each half-line; tolerance 1e-8 is 100x quad's requested accuracy on the heavy
    # Lorentzian / Student nu = 1 tails (epsabs 1e-12 per half, observed errors below 1e-10)
    total = sum(integrate.quad(lambda z: float(g(z)), a, b, limit=1000, epsabs=1e-12, epsrel=1e-12)[0]
                for a, b in ((-np.inf, 0.0), (0.0, np.inf)))
    assert abs(total - 1.0) <= 1e-8, (family, shape, w, total)
    # K3: g(0) is the maximum over a fine grid (no tolerance beyond 1e-12 relative: the mode is moved to 0
    # by construction and must be the grid's largest value up to rounding)
    z = np.linspace(-4 * w, 4 * w, 8001)
    gz = g(z)
    assert float(g(0.0)) >= gz.max() * (1 - 1e-12), (family, shape, w)
    # K2: the half-maximum points about 0 are w apart (brentq to 1e-15 on each side; 1e-9 relative)
    lo, hi = H.half_max_points(lambda x: float(g(x)), 0.0)
    assert abs((hi - lo) - w) <= 1e-9 * w, (family, shape, w, hi - lo)


@pytest.mark.parametrize("family,shape", [("gauss", None), ("lorentz", None)]
                         + [("student", s) for s in SHAPES["student"]] + [("gennorm", s) for s in SHAPES["gennorm"]])
def test_K4_closed_form_unit_fwhm(family, shape):
    """K4: unit_fwhm equals the spec's closed form, and the closed form equals the numeric FWHM of
    scipy's own density (a check of the spec's formulas too). 1e-12 / 1e-10 relative: closed forms vs
    double rounding / brentq at xtol 1e-15."""
    W = BK.unit_fwhm(family, shape)
    assert abs(W - H.closed_form_unit_fwhm(family, shape)) <= 1e-12 * W
    f = H.scipy_unit_density(family, shape)
    lo, hi = H.half_max_points(lambda x: float(f(x)), 0.0)
    assert abs((hi - lo) - H.closed_form_unit_fwhm(family, shape)) <= 1e-10 * W


@pytest.mark.parametrize("family,shape", _cases())
def test_K5_kernel_equals_scipy_density_rescaled(family, shape):
    """K5: g equals the scipy density rescaled independently in the test (another unit scale for the
    Voigt: Gaussian SD 1). 1e-9 relative to g's peak: brentq FWHM and mode at xtol 1e-15 on both sides."""
    w = 0.47
    g = BK.kernel_pdf(family, w, shape)
    ref = H.reference_kernel(family, w, shape)
    z = np.linspace(-6 * w, 6 * w, 1201)
    peak = float(ref(0.0))
    assert_allclose(g(z), ref(z), rtol=0, atol=1e-9 * peak)


@pytest.mark.parametrize("family,shape", [(f, s) for f, s in _cases() if f != "skewnorm"])
def test_K6_even_kernels(family, shape):
    """K6: every family but the skew-normal is even (1e-13 relative: the same arithmetic on +z and -z)."""
    g = BK.kernel_pdf(family, 0.3, shape)
    z = np.linspace(0, 2.0, 401)
    assert_allclose(g(z), g(-z), rtol=1e-13, atol=0)


@pytest.mark.parametrize("a", (0.5, 2.5, 10.0))
def test_K6_skewnorm_mirror(a):
    """K6: g_{-a}(z) = g_a(-z) (1e-12 relative: the mode is found by brentq at xtol 1e-15 on each sign)."""
    gp, gm = BK.kernel_pdf("skewnorm", 0.3, a), BK.kernel_pdf("skewnorm", 0.3, -a)
    z = np.linspace(-1.5, 1.5, 601)
    assert_allclose(gm(z), gp(-z), rtol=1e-12, atol=1e-14)


def test_K7_gauss_fwhm():
    """K7: gauss_fwhm(sigma) = 2 sqrt(2 ln 2) sigma; at sigma_fit 0.099 um, 0.2331 um (the spec's 0.233)."""
    assert BK.gauss_fwhm(0.099) == pytest.approx(H.FWHM_GAUSS * 0.099, rel=1e-15)
    assert abs(BK.gauss_fwhm(M.sigma_fit_um) - 0.2331) < 5e-5


@pytest.mark.parametrize("args", [("gauss", 0.0), ("gauss", -1.0), ("gauss", float("nan")), ("gauss", float("inf")),
                                  ("box", 1.0), ("gauss", 1.0, 2.0), ("student", 1.0, None),
                                  ("student", 1.0, 0.5), ("gennorm", 1.0, 9.0), ("voigt", 1.0, 1000.0),
                                  ("skewnorm", 1.0, 11.0)])
def test_K8_refusals(args):
    """K8: a FWHM not finite and > 0, an unknown family, a shape given to a shapeless family, a missing
    shape, and a shape outside the spec's ranges (log10 r in [-2, 2], log10 nu in [0, 2], beta in
    [0.5, 8], a in [-10, 10]) are refused."""
    with pytest.raises(ValueError):
        BK.kernel_pdf(*args)


# ============================================================ M: model_profile_kernel ===

V25 = np.linspace(-3.0, 3.0, 25)


@pytest.mark.parametrize("family,shape,w,n_nodes,tol", [
    ("gauss", None, 0.2331, 364, 1e-7), ("lorentz", None, 0.2331, 364, 1e-7), ("student", 3.0, 0.2331, 364, 1e-7),
    ("voigt", 1.0, 0.2331, 364, 1e-7), ("gennorm", 4.0, 0.2331, 364, 1e-7), ("skewnorm", 3.0, 0.5, 848, 1e-7),
    ("student", 6.0, 0.45, 848, 1e-7),
    # the cusped generalized normal: the spec's own tolerance (0.5 gl) for the Gauss-Legendre rule at a cusp
    ("gennorm", 1.0, 0.2331, 364, 0.5)])
@pytest.mark.parametrize("d,alpha", [(0.6, 0.4), (2.0, 1.5)])
def test_M1_model_kernel_vs_quad(family, shape, w, n_nodes, tol, d, alpha):
    """M1: model_profile_kernel equals the tube's shadow convolved with the kernel, integrated by quad in u
    (tolerances: the spec's, 1e-7 grey levels for smooth kernels, 0.5 for the cusped generalized normal;
    quad's own error is below 1e-10 grey levels here)."""
    g = BK.kernel_pdf(family, w, shape)
    B, v0 = 150.0, 0.07
    got = tube_model.model_profile_kernel(V25, d, alpha, v0, g, B, n_nodes)
    ref = H.quad_model(V25, d, alpha, v0, g, B)
    assert np.max(np.abs(got - ref)) <= tol, np.max(np.abs(got - ref))


@pytest.mark.parametrize("d,alpha,v0", [(0.3, 0.2, 0.0), (1.2, 1.0, -0.3), (4.0, 2.5, 0.5)])
def test_M2_gauss_kernel_equals_model_profile(d, alpha, v0):
    """M2: with the Gaussian of SD sigma_fit the kernel model is Block 3's model_profile (the spec's
    1e-11 grey levels: the same rule, the kernel evaluated by another formula), and both equal quad."""
    s = M.sigma_fit_um
    g = BK.kernel_pdf("gauss", BK.gauss_fwhm(s))
    a = tube_model.model_profile_kernel(V25, d, alpha, v0, g, 120.0, 364)
    b = tube_model.model_profile(V25, d, alpha, v0, s, 120.0, 364)
    assert np.max(np.abs(a - b)) <= 1e-11
    # the GL rule at N = 364 against an independent quad of (3): 1e-7 grey levels (the spec's tolerance class)
    assert np.max(np.abs(b - H.model_profile_reference(V25, d, alpha, v0, s, 120.0))) <= 1e-7


def test_M3_M4_M6_bounds_linearity_equivariance_dtype():
    """M3: I <= B exactly for non-negative kernels, -> B far away; M4: linear in B, shift- and
    mirror-equivariant (1e-12 relative: identical arithmetic up to the order of float operations);
    M6: float64 of v's shape."""
    rng = np.random.default_rng(11)
    for fam, shape in [("gauss", None), ("lorentz", None), ("student", 3.0), ("gennorm", 4.0), ("voigt", 1.0)]:
        g = BK.kernel_pdf(fam, 0.3, shape)
        for _ in range(5):
            d, alpha, v0, B = rng.uniform(0.1, 5.0), rng.uniform(0.0, 10.0), rng.uniform(-1, 1), rng.uniform(50, 250)
            I = tube_model.model_profile_kernel(V25, d, alpha, v0, g, B, 364)
            assert np.all(I <= B), (fam, d, alpha)
            assert_allclose(tube_model.model_profile_kernel(V25, d, alpha, v0, g, 2 * B, 364), 2 * I, rtol=1e-12)
            c = 0.37
            assert_allclose(tube_model.model_profile_kernel(V25 + c, d, alpha, v0 + c, g, B, 364), I,
                            rtol=1e-12, atol=1e-10)
            assert_allclose(tube_model.model_profile_kernel(-V25, d, alpha, -v0, g, B, 364), I, rtol=1e-12, atol=1e-10)
    # far from the tube the light is the background (light-tailed kernels only): 1e-9 relative
    for fam, shape in [("gauss", None), ("gennorm", 4.0)]:
        g = BK.kernel_pdf(fam, 0.3, shape)
        far = tube_model.model_profile_kernel(np.array([-12.0, 12.0]), 1.0, 1.0, 0.0, g, 100.0, 364)
        assert_allclose(far, 100.0, rtol=1e-9)
    v2 = np.linspace(-2, 2, 35).reshape(5, 7)
    out = tube_model.model_profile_kernel(v2, 1.0, 0.5, 0.0, BK.kernel_pdf("gauss", 0.3), 100.0, 64)
    assert out.shape == (5, 7) and out.dtype == np.float64


def test_M5_converges_with_nodes():
    """M5: the error against quad does not grow as the Gauss-Legendre nodes double (a smooth kernel), and is
    below 1e-9 grey levels at the largest N (analytic integrand: geometric convergence)."""
    g = BK.kernel_pdf("student", 0.2331, 3.0)
    ref = H.quad_model(V25, 2.0, 1.0, 0.1, g, 150.0)
    errs = [np.max(np.abs(tube_model.model_profile_kernel(V25, 2.0, 1.0, 0.1, g, 150.0, n) - ref))
            for n in (64, 128, 256, 512)]
    assert all(b <= a * (1 + 1e-6) + 1e-11 for a, b in zip(errs, errs[1:])), errs
    assert errs[-1] <= 1e-9, errs


# ============================================================== F: fit variants ===

def test_F1_grid_size_and_order():
    """F1: 10 kernels x 2 width modes x 5 weightings = 100, the kernels in the spec's order."""
    grid = FV.variant_grid()
    assert len(grid) == 100
    kernels = []
    for s in grid:
        if not kernels or kernels[-1] != s.kernel:
            kernels.append(s.kernel)
    assert kernels == [("gauss", None, False), ("lorentz", None, False), ("voigt", 1.0, False),
                       ("student", 3.0, False), ("gennorm", 1.0, False), ("gennorm", 4.0, False),
                       ("voigt", None, True), ("student", None, True), ("gennorm", None, True),
                       ("skewnorm", None, True)]
    for k in kernels:
        sub = [s for s in grid if s.kernel == k]
        assert sorted((s.width_fitted, -1 if s.lam_um is None else s.lam_um) for s in sub) == sorted(
            (wf, lam) for wf in (False, True) for lam in (-1, 0.25, 0.5, 1.0, 2.0))


def test_F2_F3_fixed_width_and_quadrature_nodes():
    """F2: the fixed width is the FWHM of Block 5's Gaussian; F3: N = max(min, ceil(k d_hi / sigma_min)),
    364 at the fixed width and 848 when the width is fitted (SPEC Block 11 and Block 12's rule)."""
    assert FV.fixed_fwhm_um(M) == pytest.approx(H.FWHM_GAUSS * M.sigma_fit_um, rel=1e-15)
    fixed = FV.VariantSpec("lorentz", None, False, False)
    fitted = FV.VariantSpec("lorentz", None, False, True)
    exp_fixed = max(M.fit_quad_min_nodes, math.ceil(M.fit_quad_nodes_per_sigma * M.fit_d_bounds_um[1] / M.sigma_fit_um))
    exp_fitted = max(M.fit_quad_min_nodes, math.ceil(M.fit_quad_nodes_per_sigma * M.fit_d_bounds_um[1]
                                                     / (0.1 / H.FWHM_GAUSS)))
    assert FV.n_nodes_for(fixed, M) == exp_fixed == 364
    assert FV.n_nodes_for(fitted, M) == exp_fitted == 848


def test_F4_centre_weights():
    """F4: omega_n = exp(-|v_n - v0| / lambda) (1e-15 relative)."""
    v = np.linspace(-3, 3, 53)
    for v0, lam in [(0.0, 0.25), (0.31, 0.5), (-0.7, 2.0)]:
        assert_allclose(FV.centre_weights(v, v0, lam), np.exp(-np.abs(v - v0) / lam), rtol=1e-15)


@pytest.fixture(scope="module")
def noisy():
    """A noisy Block 3 profile (d 0.8 um, mu 0.8 /um, v0 0.05 um, phi 0.2 rad, B 180, noise SD 1.5 gl)."""
    return H.noisy_profile(CFG)


def test_F5_baseline_equals_block5_fit(noisy):
    """F5: gauss, fixed width, unweighted equals fit.fit_profile (the spec's 1e-6 um / 1e-6 in d, mu, v0;
    the same solver from the same starts)."""
    v, I, phi, B = noisy
    base = FV.fit_unweighted(v, I, phi, B, M, FV.VariantSpec("gauss", None, False, False))
    f5 = block5.fit_profile(v, I, phi, B, M)
    assert abs(base.d_um - f5.d_hat_um) <= 1e-6
    assert abs(base.mu_per_um - f5.mu_hat_per_um) <= 1e-6
    assert abs(base.v0_um - f5.v0_hat_um) <= 1e-6
    assert base.status == f5.status
    # and fit_grid's base is that fit
    _fits, base2 = FV.fit_grid(v, I, phi, B, M, specs=[FV.VariantSpec("gauss", None, False, False)])
    assert base2.d_um == base.d_um


@pytest.mark.parametrize("family,shape", [("lorentz", None), ("student", 3.0), ("gennorm", 4.0), ("voigt", 1.0)])
def test_F6_noise_free_recovery_fixed_width(family, shape):
    """F6: a noise-free profile of the kernel (rebuilt in the test from scipy) at the fixed width is fitted
    back by that variant: d to 1e-6 um (the spec's), mu and v0 to 1e-5."""
    w = FV.fixed_fwhm_um(M)
    spec = FV.VariantSpec(family, shape, False, False)
    n = FV.n_nodes_for(spec, M)
    v, I, phi, B = H.noisy_profile(CFG, d=1.1, mu=0.6, v0=0.08, phi=0.3, B=170.0, noise_sd=0.0,
                                   kernel=H.reference_kernel(family, w, shape), n_nodes=n)
    f = FV.fit_unweighted(v, I, phi, B, M, spec)
    assert f.status == "converged", f
    assert abs(f.d_um - 1.1) <= 1e-6 and abs(f.mu_per_um - 0.6) <= 1e-5 and abs(f.v0_um - 0.08) <= 1e-5


def _neighbour_profile():
    """A profile whose weighted fit moves: a 0.7 um tube (alpha 0.6) whose shadow merges with a 0.5 um
    neighbour 0.6 um away (alpha 0.5), noise SD 1 gl. The unweighted fit takes both as one tube (d about
    1.31 um); the centre weights at lambda 0.5 um move it by about 0.03 um (checked before use)."""
    from allen_diameter.model import tube_model as tm
    v = profiles.profile_offsets(M)
    s = M.sigma_fit_um
    B = 170.0
    I = tm.model_profile(v, 0.7, 0.6, 0.0, s, B, 364) + tm.model_profile(v, 0.5, 0.5, 0.6, s, B, 364) - B
    I = I + np.random.default_rng(3).normal(0, 1.0, v.size)
    return v, I, 0.1, B


def test_F7_F14_F8_F9_weighted_fit_fixed_point_rss_alpha():
    """F7: the IRLS result is the minimiser of the least squares with the weights frozen at its own v0
    (re-solved here with least_squares and the kernel model): d, v0 within 2e-4 um -- the stopping rule
    leaves at most 1e-4 um per step, and the frozen-weight minimiser of the next step differs from the last
    iterate by at most about that much again. F14: PlaneFits' weighted fit is fit_weighted from the
    unweighted fit of its kernel (bit-equal: the same calls). F8: rss by their definitions (1e-9 relative);
    F9: alpha_hat = mu d / cos phi, ratio = d / d_base."""
    from scipy.optimize import least_squares
    v, I, phi, B = _neighbour_profile()
    spec_u = FV.VariantSpec("gauss", None, False, False)
    spec_w = FV.VariantSpec("gauss", None, False, False, 0.5)
    pf = FV.PlaneFits(v, I, phi, B, M)
    fw = pf.fit(spec_w)
    fu = pf.unweighted(spec_u.kernel, False)
    again = FV.fit_weighted(v, I, phi, B, M, spec_w, fu)
    assert again.d_um == fw.d_um and again.v0_um == fw.v0_um and again.n_irls == fw.n_irls        # F14
    assert fw.irls_ok is True and fw.status != "failed"
    g = BK.kernel_pdf("gauss", FV.fixed_fwhm_um(M))
    om = np.exp(-np.abs(v - fw.v0_um) / 0.5)

    def res(p):
        d, mu, v0 = p
        return np.sqrt(om) * (tube_model.model_profile_kernel(v, d, mu * d / math.cos(phi), v0, g, B, fw.n_nodes)
                              - I) / B
    lo = [M.fit_d_bounds_um[0], M.fit_mu_bounds_per_um[0], M.fit_v0_bounds_um[0]]
    hi = [M.fit_d_bounds_um[1], M.fit_mu_bounds_per_um[1], M.fit_v0_bounds_um[1]]
    for x0 in ([fw.d_um, fw.mu_per_um, fw.v0_um], [fu.d_um, fu.mu_per_um, fu.v0_um]):
        r = least_squares(res, x0, bounds=(lo, hi), method="trf", x_scale="jac", ftol=1e-12, xtol=1e-12, gtol=1e-12)
        assert abs(r.x[0] - fw.d_um) <= 2e-4 and abs(r.x[2] - fw.v0_um) <= 2e-4, (x0, r.x, fw)
    # the re-centring moved the fit (otherwise this profile would not test the weights)
    assert abs(fw.v0_um - fu.v0_um) > 1e-3 or abs(fw.d_um - fu.d_um) > 1e-3
    # F8 / F9
    model = tube_model.model_profile_kernel(v, fw.d_um, fw.mu_per_um * fw.d_um / math.cos(phi), fw.v0_um, g, B,
                                            fw.n_nodes)
    assert_allclose(fw.model_gl, model, rtol=1e-12)
    assert fw.rss_gl2 == pytest.approx(float(np.sum((model - I) ** 2)), rel=1e-9)
    # rss_w with "the last iteration's weights": the weights of the last weighted solve, i.e. centred on the
    # previous iterate's v0, which the stopping rule puts within 1e-4 um of the reported v0
    om_prev_lo = np.exp(-np.abs(v - fw.v0_um) / 0.5) * math.exp(-1e-4 / 0.5)
    om_prev_hi = np.exp(-np.abs(v - fw.v0_um) / 0.5) * math.exp(1e-4 / 0.5)
    r2 = (model - I) ** 2
    assert float(np.sum(om_prev_lo * r2)) * (1 - 1e-9) <= fw.rss_w_gl2 <= float(np.sum(om_prev_hi * r2)) * (1 + 1e-9)
    assert fu.rss_w_gl2 == pytest.approx(fu.rss_gl2, rel=1e-12)
    assert fw.alpha_hat == pytest.approx(fw.mu_per_um * fw.d_um / math.cos(phi), rel=1e-12)
    row = FV.variant_row(fw, fu)
    assert row["ratio"] == pytest.approx(fw.d_um / fu.d_um, rel=1e-15)
    assert set(row) >= {"variant", "family", "kernel", "shape_mode", "width_mode", "weight", "lam_um", "d_um",
                        "mu_per_um", "v0_um", "fwhm_um", "shape", "alpha_hat", "status", "at_bound", "rss_w_gl2",
                        "rss_gl2", "n_irls", "irls_ok", "nfev", "n_starts", "best_start", "n_nodes", "d_base_um",
                        "ratio"}


def test_F10_F11_bounds_status_cap(noisy):
    """F10: every fitted parameter inside its box; F11: the IRLS cap reached gives irls_ok False (the spec
    makes only a non-converging solve 'failed'); a start without a finite d gives a failed fit of NaNs."""
    v, I, phi, B = noisy
    specs = [FV.VariantSpec("gauss", None, False, True), FV.VariantSpec("student", None, True, False),
             FV.VariantSpec("skewnorm", None, True, False), FV.VariantSpec("gennorm", None, True, True)]
    fits, _base = FV.fit_grid(v, I, phi, B, M, specs=specs)
    box = {"voigt": (0.01, 100.0), "student": (1.0, 100.0), "gennorm": (0.5, 8.0), "skewnorm": (-10.0, 10.0)}
    for f in fits:
        assert M.fit_d_bounds_um[0] <= f.d_um <= M.fit_d_bounds_um[1]
        assert M.fit_mu_bounds_per_um[0] <= f.mu_per_um <= M.fit_mu_bounds_per_um[1]
        assert M.fit_v0_bounds_um[0] <= f.v0_um <= M.fit_v0_bounds_um[1]
        if f.spec.width_fitted:
            assert 0.1 <= f.fwhm_um <= 2.0
        else:
            assert f.fwhm_um == pytest.approx(FV.fixed_fwhm_um(M), rel=1e-15)
        if f.spec.shape_fitted:
            lo, hi = box[f.spec.family]
            assert lo <= f.shape <= hi
    v2, I2, phi2, B2 = _neighbour_profile()
    spec_w = FV.VariantSpec("gauss", None, False, False, 0.25)
    fu = FV.fit_unweighted(v2, I2, phi2, B2, M, spec_w.unweighted())
    capped = FV.fit_weighted(v2, I2, phi2, B2, M, spec_w, fu, max_iter=1)
    assert capped.irls_ok is False and capped.n_irls == 1
    nan_start = dataclasses.replace(fu, d_um=float("nan"))
    bad = FV.fit_weighted(v2, I2, phi2, B2, M, spec_w, nan_start)
    assert bad.status == "failed" and bad.irls_ok is False and math.isnan(bad.d_um)


def test_F11_stop_needs_both_d_and_v0(monkeypatch):
    """F11: the re-weighting stops only when d AND v0 have both moved by at most the tolerance (scripted
    solves: d settles from the first step, v0 only at the fourth)."""
    v, I, phi, B = _neighbour_profile()
    spec_w = FV.VariantSpec("gauss", None, False, False, 0.5)
    fu = FV.fit_unweighted(v, I, phi, B, M, spec_w.unweighted())

    class Run:
        def __init__(self, x):
            self.x, self.status, self.cost, self.nfev, self.message = np.array(x, float), 1, 0.0, 1, "scripted"

    d0, mu0, v00 = fu.d_um, fu.mu_per_um, fu.v0_um
    # (a) d never moves, v0 settles (a step of 5e-5 um) only at the third solve
    path = iter([(d0, mu0, v00 + 0.01), (d0, mu0, v00 + 0.02), (d0, mu0, v00 + 0.02005), (d0, mu0, v00 + 0.03)])
    monkeypatch.setattr(FV, "_runs", lambda lay, v, I, phi, B_bar, cfg, n_nodes, starts, omega: [Run(next(path))])
    f = FV.fit_weighted(v, I, phi, B, M, spec_w, fu)
    assert f.n_irls == 3 and f.irls_ok is True
    # (b) v0 never moves, d settles only at the third solve
    path2 = iter([(d0 + 0.01, mu0, v00), (d0 + 0.02, mu0, v00), (d0 + 0.02005, mu0, v00), (d0 + 0.03, mu0, v00)])
    monkeypatch.setattr(FV, "_runs", lambda lay, v, I, phi, B_bar, cfg, n_nodes, starts, omega: [Run(next(path2))])
    f = FV.fit_weighted(v, I, phi, B, M, spec_w, fu)
    assert f.n_irls == 3 and f.irls_ok is True


@pytest.mark.parametrize("spec", [FV.VariantSpec("gauss", None, False, False),
                                  FV.VariantSpec("skewnorm", None, True, False)])
def test_F12_F13_invariance_equivariance_determinism(noisy, spec):
    """F12: a gain on I and B leaves (d, mu, v0); a shift of v moves v0 by the shift; the mirror v -> -v maps
    v0 -> -v0 (and the skew a -> -a). Tolerances 1e-6 um (the solver's tolerance 1e-10 leaves the optimum to
    about 1e-6 of the parameters' scale on a noisy profile). F13: a repeat is bit-identical."""
    v, I, phi, B = noisy
    f = FV.fit_unweighted(v, I, phi, B, M, spec)
    f2 = FV.fit_unweighted(v, I, phi, B, M, spec)
    assert (f2.d_um, f2.mu_per_um, f2.v0_um, f2.shape) == (f.d_um, f.mu_per_um, f.v0_um, f.shape)       # F13
    g = FV.fit_unweighted(v, 1.7 * I, phi, 1.7 * B, M, spec)
    assert abs(g.d_um - f.d_um) <= 1e-6 and abs(g.v0_um - f.v0_um) <= 1e-6 and abs(g.mu_per_um - f.mu_per_um) <= 1e-6
    c = 0.2
    s = FV.fit_unweighted(v + c, I, phi, B, M, spec)
    assert abs(s.d_um - f.d_um) <= 1e-6 and abs(s.v0_um - (f.v0_um + c)) <= 1e-6
    m = FV.fit_unweighted(-v[::-1], I[::-1], phi, B, M, spec)
    assert abs(m.d_um - f.d_um) <= 1e-6 and abs(m.v0_um + f.v0_um) <= 1e-6
    if spec.family == "skewnorm":
        assert abs(m.shape + f.shape) <= 1e-4                # the skew changes sign (shape to 1e-4, a scale 10)


def test_swap_sigma_fit_and_quadrature_through_the_configuration(noisy):
    """Directive audit, swappability: sigma_fit_um and fit_quad_nodes_per_sigma changed in the configuration
    reach the variants -- the fixed width and the node counts follow the spec's rule (the code's own operation
    order, the formula being the spec's), the baseline is still Block 5's fit under that configuration (1e-6),
    and other variants run to finite fits."""
    m2 = dataclasses.replace(M, sigma_fit_um=0.12, fit_quad_nodes_per_sigma=5.0)
    assert FV.fixed_fwhm_um(m2) == pytest.approx(H.FWHM_GAUSS * 0.12, rel=1e-15)
    fixed, fitted = FV.VariantSpec("gauss", None, False, False), FV.VariantSpec("gauss", None, False, True)
    assert FV.n_nodes_for(fixed, m2) == max(m2.fit_quad_min_nodes, math.ceil(5.0 * 6.0 / 0.12))
    assert FV.n_nodes_for(fitted, m2) == max(m2.fit_quad_min_nodes, math.ceil(5.0 * 6.0 / (0.1 / H.FWHM_GAUSS)))
    v, I, phi, B = noisy
    base = FV.fit_unweighted(v, I, phi, B, m2, fixed)
    f5 = block5.fit_profile(v, I, phi, B, m2)
    assert abs(base.d_um - f5.d_hat_um) <= 1e-6 and base.n_nodes == f5.n_nodes
    fits, _b = FV.fit_grid(v, I, phi, B, m2, specs=[FV.VariantSpec("lorentz", None, False, False),
                                                     FV.VariantSpec("gauss", None, False, True, 0.5)])
    assert all(math.isfinite(f.d_um) and f.status != "failed" for f in fits)
    assert fits[0].fwhm_um == pytest.approx(H.FWHM_GAUSS * 0.12, rel=1e-15)


def test_swap_summary_threshold_through_the_configuration():
    """Directive audit, swappability: the dark-flag threshold changed in the configuration moves the summary's
    dark share by its definition, and the summary reports the configured focus rule."""
    rows = _rows()
    cfg2 = dataclasses.replace(CFG, measure=dataclasses.replace(M, alpha_dark_flag=0.55, focus_rule="dip_depth"))
    s = survey.summarize(rows, cfg2)
    conv = [r for r in rows if r["fit_status"] == "converged"]
    assert s["dark_share_of_converged"] == pytest.approx(sum(r["alpha_hat"] > 0.55 for r in conv) / len(conv))
    assert s["focus_rule"] == "dip_depth"


# ====================================================== I: the moved iteration (13.1) ===

def test_I1_script_names_are_the_module_objects():
    """I1: the script imports the moved definitions under the same names (one implementation)."""
    import plane_differences as PD
    for name in ("ITER_LINE_MULT", "ITER_STATUSES", "check_iteration", "iterate_planes", "far_from_dendrites",
                 "plane_background", "block_background", "line_profiles", "refit_at_plane", "gradient_iteration",
                 "_argmax_plane"):
        assert getattr(PD, name) is getattr(PI, name), name
    assert PI.ITER_LINE_MULT == 3.0 and PD.D_LINE_MULT == 2.0


def _table_iteration(planes, fits, d0=1.0, m=2.0, R=5):
    calls = []

    def plane_of(h):
        k = planes(h)
        return (k, {}) if not isinstance(k, tuple) else k

    def fit_at(k):
        calls.append(k)
        return fits(k)
    return PI.iterate_planes(d0, m, plane_of, fit_at, R), calls


def test_I2_iterate_planes_by_hand():
    """I2: h_1 = m d0 / 2 and h_i = m d_(i-1) / 2; converged / cycle / max_rounds / fit_failed / no_plane;
    each new plane fitted once and a repeated plane's fit reused; k_final the last plane picked, d_final its fit."""
    table = {5: 1.5, 6: 1.6, 7: 1.2}
    # converged: 5 (d 1.5) -> 6 (d 1.6) -> 6
    res, calls = _table_iteration(lambda h: 5 if h < 1.2 else 6, lambda k: {"d_hat_um": table[k]})
    assert res["status"] == "converged" and [r["k"] for r in res["rounds"]] == [5, 6, 6]
    assert_allclose([r["half_um"] for r in res["rounds"]], [1.0, 1.5, 1.6], rtol=1e-15)
    assert calls == [5, 6] and res["k_first"] == 5 and res["k_final"] == 6 and res["d_final_um"] == 1.6
    # cycle: 5 -> 6 -> 5
    res, calls = _table_iteration(lambda h: 5 if h < 1.2 or h > 1.55 else 6, lambda k: {"d_hat_um": table[k]})
    assert res["status"] == "cycle" and [r["k"] for r in res["rounds"]] == [5, 6, 5]
    assert calls == [5, 6] and res["k_final"] == 5 and res["d_final_um"] == 1.5
    # max_rounds at R = 2: 5 -> 6
    res, _ = _table_iteration(lambda h: 5 if h < 1.2 else 6, lambda k: {"d_hat_um": table[k]}, R=2)
    assert res["status"] == "max_rounds" and res["n_rounds"] == 2 and res["k_final"] == 6
    # m = 3: h_1 = 1.5 d0
    res, _ = _table_iteration(lambda h: 7, lambda k: {"d_hat_um": table[k]}, d0=0.6, m=3.0)
    assert res["rounds"][0]["half_um"] == pytest.approx(0.9, rel=1e-15)
    # fit_failed: NaN, absent
    for bad in ({"d_hat_um": float("nan")}, {}):
        res, _ = _table_iteration(lambda h: 5, lambda k, bad=bad: bad)
        assert res["status"] == "fit_failed" and res["k_final"] == 5 and math.isnan(res["d_final_um"])
    # no plane in round 1, its reason kept
    for why in ("no_plane", "line_short", "square_small"):
        res, calls = _table_iteration(lambda h, why=why: (None, {"stop": why}), lambda k: {"d_hat_um": 1.0})
        assert res["status"] == why and res["k_final"] is None and res["k_first"] is None and calls == []
        assert math.isnan(res["d_final_um"])


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_I2_fit_failed_nonpositive_d_reports_nan(bad):
    """I2 (spec: 'd_iter_um, the fit there (NaN when it failed)'; 'fit_failed when that fit returns no finite
    positive d'): a non-positive d ends the iteration fit_failed and is reported as NaN."""
    res, _ = _table_iteration(lambda h: 5, lambda k: {"d_hat_um": bad})
    assert res["status"] == "fit_failed"
    assert math.isnan(res["d_final_um"]), res["d_final_um"]


@pytest.mark.parametrize("kw", [dict(d0=0.0), dict(d0=float("nan")), dict(d0=-1.0), dict(m=0.0), dict(m=float("nan")),
                                dict(R=0), dict(R=2.5), dict(R=True), dict(R="5"), dict(R=None), dict(R=float("nan"))])
def test_I2_refusals(kw):
    """I2: refusals of d0, m and R (spec list: 0, 2.5, True, "5", None, NaN, where each applies)."""
    args = dict(d0=1.0, m=2.0, R=5)
    args.update(kw)
    with pytest.raises(ValueError):
        PI.iterate_planes(args["d0"], args["m"], lambda h: (1, {}), lambda k: {"d_hat_um": 1.0}, args["R"])


def test_I3_gradient_energy_definition():
    """I3: G_k = B_k^-2 trapezoid((d/dv gaussian_filter1d(I_k))^2) over the whole line (scipy and numpy in the
    test; 1e-12 relative), NaN for an invalid plane, a non-finite sample, B <= 0; and a linear ramp of slope a
    with smoothing off gives a^2 2 N Delta / B^2 exactly (the spec's closed form; 1e-12 relative)."""
    rng = np.random.default_rng(5)
    v = profiles.profile_offsets(M)
    prof = 100 + rng.normal(0, 5, (6, v.size))
    Bk = np.array([150.0, 160.0, 0.0, 170.0, 140.0, 155.0])
    prof[3, 7] = np.nan
    valid = np.array([True, True, True, True, False, True])
    G = focus.plane_gradient_energies(prof, v, Bk, M, valid)
    for k in range(6):
        if k in (2, 3, 4):
            assert math.isnan(G[k])
            continue
        It = ndimage.gaussian_filter1d(prof[k], sigma=M.focus_smooth_px, mode="nearest")
        ref = integrate.trapezoid(np.gradient(It, v) ** 2, v) / Bk[k] ** 2
        assert G[k] == pytest.approx(ref, rel=1e-12)
    m0 = dataclasses.replace(M, focus_smooth_px=0.0)
    a, N = 2.5, (v.size - 1) // 2
    G0 = focus.plane_gradient_energies((7.0 + a * v)[None, :], v, np.array([120.0]), m0)
    assert G0[0] == pytest.approx(a * a * 2 * N * M.profile_step_um / 120.0 ** 2, rel=1e-12)


# ============================================================= U: survey, by hand ===

def test_U8_spread_ranks():
    """U8: floor(j (m - 1)/(k - 1) + 1/2), j = 0..k-1; all of them when m <= k; floor((m - 1)/2) when k = 1."""
    for m in range(1, 40):
        for k in range(1, 12):
            if m <= k:
                exp = list(range(m))
            elif k == 1:
                exp = [(m - 1) // 2]
            else:
                exp = [int(math.floor(j * (m - 1) / (k - 1) + 0.5)) for j in range(k)]
            assert list(survey.spread_ranks(m, k)) == exp, (m, k)
    assert list(survey.spread_ranks(6, 3)) == [0, 3, 5]          # the spec's half-way case


def test_U9_registration_category():
    """U9: the category is the sentence's prefix; '' without a verdict; UNKNOWN otherwise; bare names accepted."""
    rc = registration.registration_category
    assert rc("ON THE PROCESS (lateral offset +0.00 um)") == "ON"
    assert rc("ALONGSIDE: the trace runs 0.8 um beside a process") == "ALONGSIDE"
    assert rc("FAR: 3.1 um from the nearest process") == "FAR"
    assert rc("NOT ON A VISIBLE PROCESS (contrast 0.1)") == "NOT_ON"
    assert rc("ON THE PROCESS (lateral offset +0.10 um) [the ridge is wider: probably a DIFFERENT process]") == "ON"
    assert rc(None) == "" and rc("") == ""
    assert rc("something else") == "UNKNOWN"
    for c in ("ON", "ALONGSIDE", "FAR", "NOT_ON"):
        assert rc(c) == c
    assert registration.different_process("ON THE PROCESS (x) [probably a DIFFERENT process]") is True
    assert registration.different_process("ON THE PROCESS (lateral offset +0.00 um)") is False


def test_U10_clipped_sd_definition():
    """U10: SD (ddof 1) of the values within 5 robust SDs of the median, the robust SD (1.4826 MAD) floored
    at 1 grey level (1e-12 relative)."""
    rng = np.random.default_rng(9)
    x = np.r_[rng.normal(100, 3, 2000), [400.0, -200.0, 160.0]]
    med = np.median(x)
    rsd = max(1.0, 1.4826 * np.median(np.abs(x - med)))
    assert camera_fit.clipped_sd(x) == pytest.approx(np.std(x[np.abs(x - med) <= 5 * rsd], ddof=1), rel=1e-12)
    y = np.r_[np.full(95, 100.0), [99.0, 101.0, 104.0, 106.0, 130.0]]      # MAD 0: the floor of 1 keeps |y-100|<=5
    assert camera_fit.clipped_sd(y) == pytest.approx(np.std(y[np.abs(y - 100.0) <= 5.0], ddof=1), rel=1e-12)


def _rows():
    """Hand-made pilot rows (the columns summarize and focus_agreement read)."""
    base = dict(reject="", fit_status="converged", alpha_hat=0.5, mu_hat_per_um=1.0, d_hat_um=1.0,
                allen_radius_um=0.3, phi_rad=0.1, calibration_node=False, calibration_set="", z_sub_um=0.5,
                k_star=10, k_star_depth=10, in_S=True)
    spec = [dict(d_hat_um=0.5, mu_hat_per_um=0.4, alpha_hat=0.2, calibration_node=True, calibration_set="near"),
            dict(d_hat_um=0.9, mu_hat_per_um=1.4, alpha_hat=1.3, k_star_depth=12),
            dict(d_hat_um=1.6, mu_hat_per_um=0.7, alpha_hat=1.1, calibration_node=True, calibration_set="far",
                 k_star_depth=7),
            dict(d_hat_um=2.4, mu_hat_per_um=2.2, alpha_hat=5.0, fit_status="at_bound", in_S=False,
                 reject="status:at_bound"),
            dict(d_hat_um=0.7, mu_hat_per_um=0.9, alpha_hat=0.6, in_S=False, reject="crossing;stack_edge",
                 z_sub_um=float("nan"), k_star_depth=3),
            dict(d_hat_um=1.2, mu_hat_per_um=0.5, alpha_hat=0.5, allen_radius_um=0.0, calibration_node=True),
            dict(d_hat_um=3.1, mu_hat_per_um=0.3, alpha_hat=0.9, phi_rad=0.8, k_star_depth=9)]
    rows = []
    for i, s in enumerate(spec):
        r = dict(base, node_id=100 + i)
        r.update(s)
        rows.append(r)
    del rows[5]["calibration_set"]                   # a row without the column counts under ""
    return rows


def test_U3_U4_summarize_and_focus_agreement_by_hand():
    """U3: numpy.percentile over the rows in S; mu range [p10, p90]; the dark share over the converged rows;
    the calibration counts in all and per set; the focus rule; U4: the rule/dip agreement over rows with a
    finite z_sub."""
    rows = _rows()
    s = survey.summarize(rows, CFG)
    inS = [r for r in rows if r["in_S"]]
    for key, col in (("d_hat_um", "d_hat_um"), ("mu_hat_per_um", "mu_hat_per_um"), ("alpha_hat", "alpha_hat")):
        x = np.array([r[col] for r in inS])
        for q in (5, 10, 25, 50, 75, 90, 95):
            assert s[key]["p%d" % q] == pytest.approx(np.percentile(x, q), rel=1e-15)
    ratio = np.array([r["d_hat_um"] / (2 * r["allen_radius_um"]) for r in inS if r["allen_radius_um"] > 0])
    assert s["d_hat_over_allen_d"]["p50"] == pytest.approx(np.percentile(ratio, 50), rel=1e-15)
    assert s["phi_deg"]["p90"] == pytest.approx(np.percentile(np.degrees([r["phi_rad"] for r in inS]), 90), rel=1e-15)
    mu = np.array([r["mu_hat_per_um"] for r in inS])
    assert s["suggested_phantom_mu_range_per_um"] == pytest.approx([np.percentile(mu, 10), np.percentile(mu, 90)])
    conv = [r for r in rows if r["fit_status"] == "converged"]
    assert s["dark_share_of_converged"] == pytest.approx(sum(r["alpha_hat"] > 1.0 for r in conv) / len(conv))
    assert s["n_nodes"] == 7 and s["n_in_S"] == 5
    assert s["reject_counts"] == {"status:at_bound": 1, "crossing": 1, "stack_edge": 1}
    assert s["n_calibration_nodes"] == 3 and s["n_calibration_by_set"] == {"near": 1, "far": 1, "": 1}
    assert s["focus_rule"] == M.focus_rule
    fa = survey.focus_agreement(rows)
    assert fa["n_nodes"] == 6 and fa["n_differ"] == 3 and fa["max_abs_planes"] == 3
    assert fa["counts_by_planes"] == {"2": 1, "3": 1, "1": 1}
    assert s["k_star_vs_dip_depth"] == fa


def test_breadth_unchanged_block11_functions(tmp_path):
    """Breadth over Block 11's unchanged public API, by the spec's closed forms: path_distance_um (sum of the
    segment lengths from the root; ValueError for a node not connected to a root), image_extent_um
    ((l - 1/2)p, (l + W - 1/2)p, (t + H - 1/2)p, (t - 1/2)p), depth_alpha (1 within solid_um, linear to the
    floor at fade_um), the plug-in entropy (3 bits for 8 levels, 0.811278 for 1:3), the sigmoid weight
    (1/2 at d0, 1/4 and 3/4 at d0 +- s ln 3) and the blend J = w g - (1 - w) h by hand."""
    from allen_diameter.loading import swc_io
    from allen_diameter.plotting import figures
    p = tmp_path / "t.swc"
    p.write_text("# t\n1 1 0 0 0 2 -1\n2 3 3 4 0 0.5 1\n3 3 3 4 12 0.5 2\n4 3 6 8 12 0.4 3\n5 4 0 -2 0 0.5 1\n")
    swc = swc_io.read_swc(str(p))
    assert_allclose(survey.path_distance_um(swc), [0.0, 5.0, 17.0, 22.0, 2.0], rtol=1e-15)
    q = tmp_path / "u.swc"
    q.write_text("# u\n1 1 0 0 0 2 -1\n2 3 3 4 0 0.5 3\n3 3 3 4 12 0.5 2\n")       # 2 and 3: a loop, no root
    try:
        loop = swc_io.read_swc(str(q))
    except ValueError:
        loop = None                                                                 # refused earlier, by Block 8
    if loop is not None:
        with pytest.raises(ValueError):
            survey.path_distance_um(loop)
    fr = types.SimpleNamespace(left=100, top=-20, downsample=0, res_um_px=0.1144)
    ext = figures.image_extent_um(fr, (30, 50))
    assert_allclose(ext, [(100 - 0.5) * 0.1144, (150 - 0.5) * 0.1144, (-20 + 30 - 0.5) * 0.1144, (-20 - 0.5) * 0.1144],
                    rtol=1e-15)
    assert figures.depth_alpha(0.0, 1.0, 1.5, 1.0, 3.0) == 1.0              # 0.5 um from the range: solid
    assert figures.depth_alpha(0.0, 1.0, 4.0, 1.0, 3.0) == pytest.approx(0.2)  # 3 um: the floor
    assert figures.depth_alpha(0.0, 1.0, 3.0, 1.0, 3.0) == pytest.approx(1.0 - 0.8 * 0.5)   # half way: 0.6
    H8, N8, m8 = focus.histogram_entropy(np.arange(8.0))
    assert H8 == pytest.approx(3.0, abs=1e-12) and N8 == 8 and m8 == 8
    assert focus.histogram_entropy(np.array([1.0, 2.0, 2.0, 2.0]))[0] == pytest.approx(0.811278, abs=1e-6)
    d0, s = 1.5, 0.3
    assert focus.sigmoid_weight(d0, d0, s) == pytest.approx(0.5, abs=1e-15)
    assert focus.sigmoid_weight(d0 + s * math.log(3), d0, s) == pytest.approx(0.25, abs=1e-12)
    assert focus.sigmoid_weight(d0 - s * math.log(3), d0, s) == pytest.approx(0.75, abs=1e-12)
    out = focus.blend_scores(np.array([1.0, 3.0, 2.0]), np.array([5.0, 4.0, 3.0]), 0.5)
    assert_allclose(out["g"], [0, 1, 0.5], atol=1e-15)
    assert_allclose(out["h"], [1, 0.5, 0], atol=1e-15)
    assert_allclose(out["J"], [-0.5, 0.25, 0.25], atol=1e-15)


# ======================================================= U: survey on the synthetic cell ===

@pytest.fixture(scope="module")
def cell_case():
    return H.synthetic_cell()


def test_U1_measure_nodes_orchestration(cell_case, monkeypatch):
    """U1: every node of the stretches holding the requested ids (or all), each measured by Block 13's
    measure_node with the cell's scene; (NodeResult, Allen radius, Branch, index) per node (stubbed measure)."""
    cfg, swc, provider, *_ = cell_case
    seen = []

    def fake(branch, i, prov, cfg_, reg=None, scene=None):
        seen.append((int(branch.ids[i]), scene))
        return ("R", int(branch.ids[i]))
    monkeypatch.setattr(survey, "measure_node", fake)
    out = survey.measure_nodes(swc, provider, cfg, None, None, {4})
    assert [o[0][1] for o in out] == [2, 3, 4, 5, 6, 7]                       # the whole stretch of node 4
    assert all(o[1] == pytest.approx(0.3) for o in out)                       # Allen's radius
    assert all(int(o[2].ids[o[3]]) == o[0][1] for o in out)                   # Branch and index of the node
    ref = cell.cell_scene(swc, cfg.acquisition.dendrite_swc_types, cfg.acquisition.res0_um, None)
    for _nid, scene in seen:
        assert scene is not None and np.array_equal(scene.soma, ref.soma)
    assert survey.measure_nodes(swc, provider, cfg, None, None, {1}) == []     # the soma's id: no dendrite stretch
    assert len(survey.measure_nodes(swc, provider, cfg)) == 6                 # None: every dendrite node


def test_U1_U2_measured_node_and_pilot_row(cell_case):
    """U1: measure_nodes' result for a node equals measure_node(branch, i, provider, cfg, None, cell_scene);
    U2: pilot_row's columns by the spec (in_S, reject, the pilot's S, the calibration columns with
    D_A = 2 r_Allen, ITER_COLUMNS with tuples joined by ';', sigma_fit, focus columns)."""
    from allen_diameter.analysis import calibration, phantoms
    from allen_diameter.analysis.node_pipeline import pilot_view
    cfg, swc, provider, *_ = cell_case
    res, r_allen, branch, i = H.measured_node(4)
    out = survey.measure_nodes(swc, provider, cfg, None, None, {4}, None, whole_stretches=False)
    assert len(out) == 1
    got = out[0][0]
    for f in dataclasses.fields(res):
        if f.name in ("fit", "plane_fit", "iteration"):
            continue
        assert _nan_equal(getattr(got, f.name), getattr(res, f.name)), f.name
    row = survey.pilot_row(res, r_allen, cfg)
    reasons = phantoms.reject_reasons(res)
    assert row["in_S"] == (not reasons) and row["reject"] == ";".join(reasons)
    preasons = phantoms.reject_reasons(pilot_view(res))
    assert row["in_S_pilot"] == (not preasons) and row["reject_pilot"] == ";".join(preasons)
    cal = calibration.calibration_reasons(res, cfg, 2 * r_allen)
    assert row["calibration_node"] == (not cal)
    assert row["calibration_set"] == ("" if cal else calibration.calibration_set(2 * r_allen, cfg))
    assert row["allen_radius_um"] == r_allen and row["sigma_fit_um"] == M.sigma_fit_um
    assert row["focus_rule"] == res.focus_rule and row["k_star_depth"] == res.k_star_depth
    for c in cell.ITER_COLUMNS:
        assert c in row, c
    assert row["at_bound"] == ";".join(res.at_bound) and row["pilot_flags"] == ";".join(res.pilot_flags)
    assert row["flags"] == ";".join(res.flags)
    nr_cols = [c for c in cell.CSV_COLUMNS if hasattr(res, c) and c not in ("flags",) + tuple(cell.ITER_COLUMNS)]
    missing = [c for c in nr_cols if c not in row]
    assert not missing, "per-node CSV columns of the NodeResult missing from the pilot row: %s" % missing


def test_U5_profile_at_node_draws_the_pilot(cell_case):
    """U5: the profile is Eq. 9 through the pilot's centre in plane k* (bilinear by map_coordinates in the
    test, averaged over 2A+1 lines; 1e-9 grey levels), the model Block 3's (3) at the pilot's fit and B_bar
    (quad in the test; 1e-7 grey levels)."""
    from allen_diameter.analysis.node_pipeline import pilot_view
    cfg, swc, provider, *_ = cell_case
    res, _r, _b, _i = H.measured_node(4)
    p = pilot_view(res)
    v, I, model = survey.profile_at_node(provider, res, cfg)
    assert_allclose(v, profiles.profile_offsets(M), rtol=0, atol=0)
    px = cfg.acquisition.res0_um
    left, top = int(math.floor(p.cx_um / px)) - 60, int(math.floor(p.cy_um / px)) - 60
    blk, _ks, _valid, _frame = provider(left, top, 121, 121, p.k_star, p.k_star)
    th = p.theta_rad
    y_hat, e_u = np.array([-math.sin(th), math.cos(th)]), np.array([math.cos(th), math.sin(th)])
    A = int(round(M.along_branch_avg_um / M.profile_step_um))
    acc = np.zeros(v.size)
    for a in range(-A, A + 1):
        pts = np.array([p.cx_um, p.cy_um])[None, :] + v[:, None] * y_hat[None, :] + a * M.profile_step_um * e_u[None, :]
        col, row = pts[:, 0] / px - left, pts[:, 1] / px - top
        acc += ndimage.map_coordinates(blk[0].astype(float), [row, col], order=1, mode="constant", cval=np.nan)
    assert_allclose(I, acc / (2 * A + 1), rtol=0, atol=1e-9)
    f = p.fit
    ref = H.model_profile_reference(v, f.d_hat_um, f.alpha_hat, f.v0_hat_um, M.sigma_fit_um, p.B_bar)
    assert np.max(np.abs(model - ref)) <= 1e-7
    assert p.B_bar == res.B_bar_pilot


def _seg_dist(px_, py_, a, b):
    ab = b - a
    L2 = float(ab @ ab)
    t = np.zeros_like(px_) if L2 == 0 else np.clip(((px_ - a[0]) * ab[0] + (py_ - a[1]) * ab[1]) / L2, 0, 1)
    return np.hypot(px_ - (a[0] + t * ab[0]), py_ - (a[1] + t * ab[1]))


def test_U6_background_stats_definition(cell_case):
    """U6: median, 1.4826 MAD, clipped SD and unmasked fraction of the pixels of the +-block_half_um block about
    the centre (pixel centres at integer coordinates x p) in plane k*, outside (radius + margin) of every
    segment of the branch (each with the larger radius of its ends); 1e-9 relative."""
    cfg, swc, provider, *_ = cell_case
    res, _r, branch, _i = H.measured_node(4)
    med, rsd, csd, frac = survey.background_stats(provider, res, branch, cfg)
    px, h = cfg.acquisition.res0_um, M.block_half_um
    c0, c1 = math.ceil((res.cx_um - h) / px), math.floor((res.cx_um + h) / px)
    r0, r1 = math.ceil((res.cy_um - h) / px), math.floor((res.cy_um + h) / px)
    blk, _ks, _valid, _fr = provider(c0, r0, c1 - c0 + 1, r1 - r0 + 1, res.k_star, res.k_star)
    plane = blk[0].astype(float)
    rows, cols = np.mgrid[r0:r1 + 1, c0:c1 + 1]
    X, Y = cols * px, rows * px
    near = np.zeros(plane.shape, bool)
    xyz, rad = branch.xyz_um, np.asarray(branch.radius_um)
    for j in range(len(xyz) - 1):
        near |= _seg_dist(X, Y, xyz[j, :2], xyz[j + 1, :2]) <= max(rad[j], rad[j + 1]) + M.bbar_mask_margin_um
    vals = plane[~near]
    m_ = np.median(vals)
    assert med == pytest.approx(m_, rel=1e-12)
    assert rsd == pytest.approx(1.4826 * np.median(np.abs(vals - m_)), rel=1e-12)
    assert csd == pytest.approx(camera_fit.clipped_sd(vals), rel=1e-12)
    assert frac == pytest.approx(vals.size / plane.size, rel=1e-12)


def test_U7_node_planes(cell_case):
    """U7: node_planes measures the pilot (equal to measure_pilot on the node's own stretch, one provider
    request), its segments are the SWC segments with a dendrite child whose xy bounding box meets the block
    (rows rebuilt here: ends, Allen radii -- r0 = r1 at the soma link --, in_stretch), k_swc = round(z/dz);
    ids not in the SWC and the soma refused."""
    cfg, swc, provider, *_ = cell_case
    cp = H.CountingProvider(provider)
    pl = survey.node_planes(swc, cp, cfg, 4)
    assert len(cp.calls) == 1
    ref, _b, _i = H.measured_pilot(4)
    for f in dataclasses.fields(ref):
        if f.name in ("fit", "plane_fit", "iteration"):
            continue
        assert _nan_equal(getattr(pl["result"], f.name), getattr(ref, f.name)), f.name
    fr, (Hh, W) = pl["frame"], pl["block"].shape[1:]
    p = fr.res_um_px
    xlo, xhi = (fr.left - 0.5) * p, (fr.left + W - 0.5) * p
    ylo, yhi = (fr.top - 0.5) * p, (fr.top + Hh - 0.5) * p
    xyz = cell.to_image_um(swc.xyz, cfg.acquisition.res0_um)
    ids = list(map(int, swc.ids))
    types_ = list(map(int, swc.types))
    stretch_ids = set(int(x) for x in pl["branch"].ids)
    exp = []
    for c in range(len(ids)):
        par = int(swc.parent[c])
        if types_[c] not in (3, 4) or par < 0 or par not in ids:
            continue
        pr = ids.index(par)
        a, b = xyz[pr], xyz[c]
        if max(a[0], b[0]) < xlo or min(a[0], b[0]) > xhi or max(a[1], b[1]) < ylo or min(a[1], b[1]) > yhi:
            continue
        r1 = float(swc.radius[c])
        r0 = float(swc.radius[pr]) if types_[pr] in (3, 4) else r1
        exp.append(list(a) + list(b) + [r0, r1, 1.0 if (par in stretch_ids and ids[c] in stretch_ids) else 0.0])
    got = np.asarray(pl["segments"])
    exp = np.asarray(exp)
    assert got.shape == exp.shape, (got.shape, exp.shape)
    assert_allclose(got[np.lexsort(got.T[::-1])], exp[np.lexsort(exp.T[::-1])], rtol=0, atol=1e-12)
    assert pl["k_swc"] == int(round(float(pl["branch"].xyz_um[pl["index"], 2]) / cfg.acquisition.dz_um))
    for bad in (999, 1):
        with pytest.raises(ValueError):
            survey.node_planes(swc, provider, cfg, bad)


def test_U7_node_planes_record_reports_the_pilots_values(cell_case, tmp_path):
    """U7 (scripts/node_planes.py): the record's `pilot` is `same` for the pilot run's own row, and otherwise
    'the fields ... that differ, each with the pilot's and the new value' -- the pilot's value is the row read
    as the pilot's (cell.pilot_row_view), since Block 13 the row's d_hat_um is the deliverable's."""
    import node_planes as NP
    cfg, swc, provider, *_ = cell_case
    res, r_allen, _b, _i = H.measured_node(4)
    row = survey.pilot_row(res, r_allen, cfg)
    recs = NP.run(swc, provider, cfg, [4], str(tmp_path / "a"), "999", pilot_rows=[row], log=lambda m: None)
    assert recs[0]["pilot"] == "same", recs[0]["pilot"]
    changed = dict(row, d_pilot_um=row["d_pilot_um"] + 0.1)        # a pilot run whose pilot fit differed
    recs = NP.run(swc, provider, cfg, [4], str(tmp_path / "b"), "999", pilot_rows=[changed], log=lambda m: None)
    diff = recs[0]["pilot"]
    assert isinstance(diff, dict) and set(diff) == {"d_hat_um"}, diff
    pilot_value, new_value = diff["d_hat_um"]
    assert new_value == pytest.approx(res.d_pilot_um, abs=0)        # the re-measured pilot's d
    assert pilot_value == changed["d_pilot_um"], ("reported as the pilot's value", pilot_value,
                                                  "the pilot's d (d_pilot_um)", changed["d_pilot_um"],
                                                  "the deliverable's d (d_hat_um)", changed["d_hat_um"])


# ========================================================= V / P: variant iteration, CLI ===

def test_V1_V7_iteration_params_and_specs():
    """V1: (m, R) from the record; V7: 50 variants at the fitted width by default; P3: the Gaussian always
    kept and unknown families refused (the Cell 4l script)."""
    import fit_variants as FVS
    import variant_iteration as VI
    assert VI.iteration_params({"line_mult": 3, "max_rounds": 5}) == (3.0, 5)
    with pytest.raises(ValueError):
        VI.iteration_params({"line_mult": 0, "max_rounds": 5})
    specs = VI.select_specs()
    assert len(specs) == 50 and all(s.width_fitted for s in specs)
    assert len({s.kernel for s in specs}) == 10 and {s.lam_um for s in specs} == {None, 0.25, 0.5, 1.0, 2.0}
    fams = {s.family for s in FVS.select_specs(families=["voigt"])}
    assert fams == {"gauss", "voigt"}
    with pytest.raises(ValueError):
        FVS.select_specs(families=["box"])


def test_V4_V5_selection_edges_and_ratio():
    """V4: h_sel the last round that picked a plane, s = (c - o).y_hat, edges s + v0 -+ d/2 on L
    (1e-12); line L = o +- h_sel y_hat; V5: ratio_ref = d / d_ref."""
    import variant_iteration as VI
    th = 0.4
    st = {"result": types.SimpleNamespace(cx_um=10.3, cy_um=5.2), "theta": th, "o": [10.0, 5.0]}
    ev = {"rounds": [dict(round=1, k=3, half_um=0.9, d_hat_um=0.8, v0_hat_um=0.02),
                     dict(round=2, k=4, half_um=1.2, d_hat_um=0.85, v0_hat_um=0.01),
                     dict(round=3, k=4, half_um=1.275, d_hat_um=0.85, v0_hat_um=0.01)],
          "k_final": 4, "k_first": 3, "status": "converged", "n_rounds": 3, "n_restacks": 0, "d_start_um": 0.6}
    sel = VI.selection(ev, st)
    y_hat = np.array([-math.sin(th), math.cos(th)])
    s = float(np.dot(np.array([0.3, 0.2]), y_hat))
    assert sel["k_iter"] == 4 and sel["sel_round"] == 3 and sel["h_sel"] == 1.275
    assert sel["s_um"] == pytest.approx(s, abs=1e-12)
    assert_allclose(sel["edges_v"], [s + 0.01 - 0.425, s + 0.01 + 0.425], atol=1e-12)
    assert_allclose(sel["line_xy"], [np.array([10.0, 5.0]) - 1.275 * y_hat, np.array([10.0, 5.0]) + 1.275 * y_hat],
                    atol=1e-12)
    assert_allclose(sel["edges_xy"], np.array([10.0, 5.0])[None, :] + np.asarray(sel["edges_v"])[:, None] * y_hat,
                    atol=1e-12)
    rec = VI.variant_record(None, ev, sel, 0.8)
    assert rec["ratio_ref"] == pytest.approx(0.85 / 0.8, rel=1e-15) and rec["d_iter_um"] == 0.85


@pytest.mark.parametrize("script,argv", [
    ("fit_variants", ["--width-modes", "bogus"]), ("fit_variants", ["--lambdas", "-1"]),
    ("fit_variants", ["--families", "box"]), ("fit_variants", ["--nodes", "a,b"]),
    ("variant_iteration", ["--width-modes", "bogus"]), ("variant_iteration", ["--lambdas", "0"]),
    ("variant_iteration", ["--families", "box"]), ("variant_iteration", ["--nodes", "x"])])
def test_P3_cli_refusals(script, argv):
    """P3: the Cell 4l / 4m command lines refuse a bad width mode, lambda, family or node list (argparse exit 2)
    before any other work (no configuration, no network)."""
    mod = __import__(script)
    with pytest.raises(SystemExit) as e:
        mod.main(["--specimen", "1", "--iter-json", "/nonexistent.json", "--out-dir", "/nonexistent"] + argv)
    assert e.value.code == 2
