"""Spec-derived tests of Block 12 (the skewed generalized-t kernel and the
seven-parameter fit at a plane) -- integration tester, run D3, 2026-11-13.
Expected values come from specs/SPEC.md (section 2.2, Block 12), from scipy
(quad, betainc, gammainc, brentq, scipy.stats) or by hand; the code under test
only produces the outputs that are checked.

Run (from the workstream directory):
    python -m pytest tests/spec/test_spec_block12_sgt_fit.py -q
The fitted-profile checks run about eight fit_plane calls (seconds each on one
core; longer on a shared machine).
"""
from __future__ import annotations

import dataclasses
import math
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import integrate, optimize, stats

import spec_helpers_D3 as H

from allen_diameter.analysis import fit_sgt
from allen_diameter.config import default_config, with_overrides
from allen_diameter.model import blur_kernels as bk
from allen_diameter.model import tube_model

CFG = default_config()
M = CFG.measure
B_BAR = 200.0
PHI = 0.2
V = np.arange(-26, 27) * M.profile_step_um          # Block 5's offsets at the defaults (+-3 um, 53 samples)

SHAPES = [(b, x, z) for b in (1.0, 1.5, 2.0, 4.0, 10.0) for x in (0.0, 0.05, 1.0, 2.0)
          for z in (math.exp(-0.7), 1.0, 1.3, math.exp(0.7))]


# ----------------------------------------------------------- the kernel ----

def test_kernel_half_masses_match_closed_forms():
    # O12.1: quad on a finite smooth range [0, X] (break point at the mode only) is accurate to ~1e-14; the
    # closed forms are exact (betainc / gammainc to ~1e-15); 1e-10 leaves room for quad on the cusped members.
    w = 0.5
    worst = 0.0
    for beta, xi, zeta in SHAPES:
        g = bk.sgt_kernel_pdf(w, beta, xi, zeta)
        for X in (0.3 * w, w, 3.0 * w, 10.0 * w):
            right = integrate.quad(lambda z: float(g(z)), 0.0, X, epsabs=1e-14, epsrel=1e-13, limit=400)[0]
            left = integrate.quad(lambda z: float(g(z)), -X, 0.0, epsabs=1e-14, epsrel=1e-13, limit=400)[0]
            r_ref, l_ref = H.ref_half_mass(X, w, beta, xi, zeta)
            worst = max(worst, abs(right - r_ref), abs(left - l_ref))
    assert worst < 1e-10, worst


def test_kernel_unit_mass_and_side_masses():
    # O12.1: the closed forms at X -> infinity are the side masses zeta^2/(1+zeta^2) and 1/(1+zeta^2) (by hand),
    # so the kernel integrates to one; checked directly by quad over each half-line for the lighter tails.
    w = 0.5
    for beta, xi, zeta in SHAPES:
        r_ref, l_ref = H.ref_half_mass(np.inf, w, beta, xi, zeta)
        assert_allclose([r_ref, l_ref], [zeta ** 2 / (1 + zeta ** 2), 1 / (1 + zeta ** 2)], rtol=0, atol=1e-14)
        if xi <= 1.0 or beta >= 2.0:          # tails |z|^-(1 + beta/xi) at most |z|^-2: quad on inf converges
            g = bk.sgt_kernel_pdf(w, beta, xi, zeta)
            tot = (integrate.quad(lambda z: float(g(z)), 0.0, np.inf, epsabs=1e-13, limit=500)[0]
                   + integrate.quad(lambda z: float(g(z)), -np.inf, 0.0, epsabs=1e-13, limit=500)[0])
            assert abs(tot - 1.0) < 1e-8, (beta, xi, zeta, tot)


def test_kernel_fwhm_mode_and_half_width_ratio():
    # O12.1: FWHM = w by brentq on the kernel itself (1e-12 um), right/left half-width ratio zeta^2 (1e-10
    # relative), maximum at 0 and monotone on each side (checked on a grid of 2001 offsets).
    w = 0.37
    for beta, xi, zeta in SHAPES:
        g = bk.sgt_kernel_pdf(w, beta, xi, zeta)
        peak = float(g(0.0))
        zr = optimize.brentq(lambda z: float(g(z)) - 0.5 * peak, 1e-12, 50 * w, xtol=1e-15, rtol=1e-15)
        zl = optimize.brentq(lambda z: float(g(z)) - 0.5 * peak, -50 * w, -1e-12, xtol=1e-15, rtol=1e-15)
        assert abs((zr - zl) - w) < 1e-12, (beta, xi, zeta, zr - zl)
        assert abs(zr / (-zl) / zeta ** 2 - 1) < 1e-10
        z = np.linspace(-5 * w, 5 * w, 2001)
        gz = g(z)
        assert np.all(gz <= peak + 1e-15)
        assert np.all(np.diff(gz[z >= 0]) <= 1e-15) and np.all(np.diff(gz[z <= 0]) >= -1e-15)


def test_kernel_matches_written_formula():
    # O12.1: the code against (H1)-(H3) written out independently (spec_helpers_D3.ref_sgt_pdf). The two evaluate
    # ln g = ln c + ln(2/s) - ln a + core in another order; the core's rounding is amplified by its exponent
    # (1/xi + 1/beta <= 21 here), so ln g agrees to a few ulps of 21 |ln g|: |d ln g| / (1 + |ln g|) < 1e-12.
    z = np.linspace(-3.0, 3.0, 301)
    for beta, xi, zeta in SHAPES:
        for w in (0.1, 0.5, 2.0):
            got, ref = bk.sgt_kernel_pdf(w, beta, xi, zeta)(z), H.ref_sgt_pdf(z, w, beta, xi, zeta)
            pos = ref > 1e-300          # normal doubles only: below ~2e-308 values are subnormal (few significant bits)
            lg, lr = np.log(got[pos]), np.log(ref[pos])
            assert np.max(np.abs(lg - lr) / (1 + np.abs(lr))) < 1e-12, (beta, xi, zeta, w)


def test_kernel_members_against_scipy_stats():
    # O12.2: (2,0,1) normal, (1,0,1) Laplace, (2,2/nu,1) Student t_nu, (2,2,1) Cauchy, (beta,0,1) generalized
    # normal, each rescaled to FWHM w (brentq on its own pdf); 1e-12 absolute (values of a few per um).
    w = 0.3
    z = np.linspace(-2.0, 2.0, 401)
    members = [((2, 0, 1), stats.norm), ((1, 0, 1), stats.laplace), ((2, 0.5, 1), stats.t(4)),
               ((2, 2.0 / 3.0, 1), stats.t(3)), ((2, 2, 1), stats.cauchy), ((3, 0, 1), stats.gennorm(3)),
               ((1.5, 0, 1), stats.gennorm(1.5))]
    for (beta, xi, zeta), dist in members:
        ref = H.scipy_member_pdf(dist, w)(z)
        got = bk.sgt_kernel_pdf(w, beta, xi, zeta)(z)
        assert np.max(np.abs(got - ref)) < 1e-12, (beta, xi, np.max(np.abs(got - ref)))


def test_kernel_mirror_and_tail_exponent():
    # O12.2: g(z | zeta) = g(-z | 1/zeta) (by hand from (H2)); tails ~ |z|^-(1 + beta/xi) for xi > 0: the log-log
    # slope between z = 1e5 w and 1e6 w is the exponent. For xi >= 1 there xi x^beta >= 8e4, so the slope differs
    # from the exponent by about (1/xi + 1/beta) / (xi x^beta ln 10) < 1e-5 (by hand): tolerance 1e-4.
    w = 0.5
    z = np.linspace(-4, 4, 801)
    for beta, xi, zeta in SHAPES:
        a = bk.sgt_kernel_pdf(w, beta, xi, zeta)(z)
        b = bk.sgt_kernel_pdf(w, beta, xi, 1.0 / zeta)(-z)
        assert np.max(np.abs(a - b)) <= 1e-14 * np.max(a)
        if xi >= 1.0:
            g = bk.sgt_kernel_pdf(w, beta, xi, 1.0)
            z1, z2 = 1e5 * w, 1e6 * w
            slope = (math.log(float(g(z2))) - math.log(float(g(z1)))) / math.log(10.0)
            assert abs(slope + (1 + beta / xi)) < 1e-4, (beta, xi, slope)


def test_kernel_xi_zero_limit():
    # O12.2: SGT_XI_ZERO = 1e-12; W continuous at xi -> 0 through the expm1 branch: xi = 1e-10 within 1e-9 relative
    # of the xi = 0 value; the kernel at xi = 1e-9 within 1e-6 of the xi = 0 kernel; just above the switch
    # (xi = 2e-12, the beta-function form with 1/xi = 5e11) the kernel still has unit mass (quad, 1e-9).
    assert bk.SGT_XI_ZERO == 1e-12
    z = np.linspace(-2, 2, 401)
    for beta in (1.0, 2.0, 5.0, 10.0):
        for zeta in (0.6, 1.0, 1.7):
            W0, W1 = bk.sgt_unit_fwhm(beta, 0.0, zeta), bk.sgt_unit_fwhm(beta, 1e-10, zeta)
            assert abs(W1 / W0 - 1) < 1e-9
            k0 = bk.sgt_kernel_pdf(0.5, beta, 0.0, zeta)(z)
            k1 = bk.sgt_kernel_pdf(0.5, beta, 1e-9, zeta)(z)
            assert np.max(np.abs(k1 - k0)) < 1e-6
            for xi in (2e-12, 1e-11, 1e-12):
                g = bk.sgt_kernel_pdf(0.5, beta, xi, zeta)
                tot = (integrate.quad(lambda t: float(g(t)), 0, np.inf, epsabs=1e-13, limit=400)[0]
                       + integrate.quad(lambda t: float(g(t)), -np.inf, 0, epsabs=1e-13, limit=400)[0])
                assert abs(tot - 1) < 1e-9, (beta, zeta, xi, tot)
    # the core's half width at half maximum and W (H3), by hand
    for beta, xi, zeta in SHAPES:
        assert abs(bk.sgt_core_hwhm(beta, xi) / H.ref_core_hwhm(beta, xi) - 1) < 1e-13
        assert abs(bk.sgt_unit_fwhm(beta, xi, zeta) / ((zeta + 1 / zeta) * H.ref_core_hwhm(beta, xi)) - 1) < 1e-13


def test_kernel_contracts():
    # O12.3
    for bad in ((0, 0, 1), (-1, 0, 1), (2, -1e-9, 1), (2, 0, 0), (2, 0, -1), (np.nan, 0, 1), (2, np.inf, 1),
                (2, 0, np.nan)):
        with pytest.raises(ValueError):
            bk.check_sgt_shape(*bad)
    assert bk.check_sgt_shape(2, 0, 1) == (2.0, 0.0, 1.0)
    for w in (0.0, -0.1, np.nan, np.inf):
        with pytest.raises(ValueError):
            bk.sgt_kernel_pdf(w, 2, 0, 1)
    x = np.zeros((2, 3))
    assert bk.sgt_unit_pdf(x, 2, 0.5, 1.2).shape == (2, 3)
    assert bk.sgt_kernel_pdf(0.4, 2, 0.5, 1.2)(x).shape == (2, 3)
    assert bk.SGT not in bk.FAMILIES and "sgt" not in bk.FAMILIES
    assert bk.FAMILIES == ("gauss", "lorentz", "voigt", "student", "gennorm", "skewnorm")


# ------------------------------------------------------------ the model ----

@pytest.mark.parametrize("shape,tol_gl", [((2.0, 0.0, 1.0), 1e-7), ((4.0, 0.5, 1.0), 1e-7),
                                          ((2.0, 0.3, 1.3), 1e-4), ((1.0, 0.0, 1.0), 2e-3)])
def test_model_against_direct_convolution(shape, tol_gl):
    # O12.4: (3'') at the fit's N against a direct convolution by quad (the spec's tolerances by member class at
    # w = 0.5 um, d <= 1.5 um: analytic 1e-7, skewed 1e-4, Laplace 2e-3 grey levels).
    beta, xi, zeta = shape
    N = fit_sgt.n_nodes_for(M)
    v = np.linspace(-2.0, 2.0, 17)
    for d, mu, v0 in ((1.5, 1.0, 0.1), (0.4, 2.5, -0.05)):
        q = fit_sgt.to_q(d, mu, v0, 0.5, beta, xi, zeta)
        got = fit_sgt.sgt_model(v, q, PHI, B_BAR, N)
        ref = H.ref_model_quad(v, d, mu, v0, PHI, B_BAR, bk.sgt_kernel_pdf(0.5, beta, xi, zeta))
        assert np.max(np.abs(got - ref)) < tol_gl, (shape, d, np.max(np.abs(got - ref)))


def test_model_dip_area_is_kernel_independent():
    # O12.4: the kernel has unit mass, so the dip's area is Block 3's B pi d/2 (I_1(alpha) - L_1(alpha)) whatever
    # the kernel; only the analytic psi-integrand is quadratured (N = 848), and quad over v is accurate to ~1e-9
    # relative on these tails (at most |z|^-3): 1e-7 relative.
    N = fit_sgt.n_nodes_for(M)
    for d, mu, v0, shape, w in ((1.5, 0.8, 0.1, (2.0, 0.0, 1.0), 0.5), (0.6, 3.0, -0.2, (1.0, 0.0, 1.4), 0.3),
                                (2.5, 0.3, 0.0, (2.0, 1.0, 0.8), 1.0), (1.0, 1.0, 0.0, (10.0, 0.4, 1.0), 0.2)):
        q = fit_sgt.to_q(d, mu, v0, w, *shape)
        dip = (lambda t: B_BAR - float(fit_sgt.sgt_model(np.array([t]), q, PHI, B_BAR, N)[0]))
        area = sum(integrate.quad(dip, a, b, epsabs=1e-10, epsrel=1e-12, limit=800)[0]
                   for a, b in ((-np.inf, v0 - d), (v0 - d, v0 + d), (v0 + d, np.inf)))
        ref = H.dip_area_exact(d, mu * d / math.cos(PHI), B_BAR)
        assert abs(area / ref - 1) < 1e-7, (d, shape, area, ref)


def test_model_bounded_by_background_and_gaussian_member():
    # O12.4: I_model <= B_bar exactly at N (every quadrature term is >= 0) over the box's corners and random
    # interior points; at (2, 0, 1) the model is Block 3's Gaussian model with sigma = w / (2 sqrt(2 ln 2))
    # (1e-10 grey levels, the spec's oracle).
    N = fit_sgt.n_nodes_for(M)
    lb, ub = fit_sgt.box(M)
    rng = np.random.default_rng(12)
    v = np.linspace(-3, 3, 61)
    corners = [np.where(np.array(list(np.binary_repr(k, 7)), dtype=int) == 1, ub, lb) for k in range(128)]
    for q in corners + [lb + rng.random(7) * (ub - lb) for _ in range(40)]:
        assert np.all(fit_sgt.sgt_model(v, q, PHI, B_BAR, N) <= B_BAR)
    for d, mu, v0, w in ((1.5, 0.8, 0.1, 0.5), (0.3, 4.0, -0.3, 0.25), (5.0, 0.2, 0.0, 1.5)):
        got = fit_sgt.sgt_model(v, fit_sgt.to_q(d, mu, v0, w, 2.0, 0.0, 1.0), PHI, B_BAR, N)
        ref = tube_model.model_profile(v, d, mu * d / math.cos(PHI), v0, w / H.FWHM_GAUSS, B_BAR, N)
        assert np.max(np.abs(got - ref)) < 1e-10


def test_quadrature_orders_as_documented():
    # Block 12's quadrature oracles, in terms free of the background (the spec's grey-level figures 0.05, 0.18,
    # 0.70, 2.1 gl do not state the B_bar they were taken at; they scale with it): at the box corner (d 6 um,
    # w 0.1 um, mu 20) the observed order between N/2 and N against a 16 N reference is >= 8 for the analytic
    # member (error <= 1e-6 gl), in [2.6, 3.8] for the skew (zeta 1.3), [2.1, 3.0] for beta 1.5, [1.6, 2.5] for the
    # Laplace; the classes' errors at N stand to the Laplace's as the documented figures do (0.05 : 0.18 : 2.1 to
    # 0.70, within 15 %), and the fitted regime's worst (Laplace, w 0.3 um, d <= 5.5 um) to the corner's as
    # 0.07 : 0.70 (within 15 %); I_model >= -0.02 gl at 16 N.
    N = fit_sgt.n_nodes_for(M)
    v = np.linspace(-4, 4, 161)

    def errs(q):
        ref = fit_sgt.sgt_model(v, q, 0.0, B_BAR, 16 * N)
        e = [np.max(np.abs(fit_sgt.sgt_model(v, q, 0.0, B_BAR, n) - ref)) for n in (N // 2, N)]
        return e, ref.min()
    (e2, e1), mn = errs(fit_sgt.to_q(6.0, 20.0, 0.0, 0.1, 2.0, 0.0, 1.0))
    # analytic: at the rounding floor already at N/2 (~1e-9 gl, 5e-12 of B_bar) no order is measurable: converged
    assert e1 <= 1e-6 and (e2 <= 1e-8 or math.log2(e2 / e1) >= 8), (e2, e1)
    ranges = {(2.0, 0.0, 1.3): (2.6, 3.8, 0.05), (1.5, 0.0, 1.0): (2.1, 3.0, 0.18), (1.0, 0.0, 1.0): (1.6, 2.5, 0.70),
              (10.0, 0.0, 1.0): (None, None, 2.1)}
    at_N = {}
    for shape, (lo, hi, _) in ranges.items():
        (e2, e1), mn = errs(fit_sgt.to_q(6.0, 20.0, 0.0, 0.1, *shape))
        at_N[shape] = e1
        assert mn >= -0.02
        if lo is not None:
            assert lo <= math.log2(e2 / e1) <= hi, (shape, math.log2(e2 / e1))
    lap = at_N[(1.0, 0.0, 1.0)]
    for shape, (_, _, doc) in ranges.items():
        assert abs((at_N[shape] / lap) / (doc / 0.70) - 1) < 0.15, (shape, at_N[shape] / lap, doc / 0.70)
    fitted = max(np.max(np.abs(fit_sgt.sgt_model(v, q, 0.0, B_BAR, N) - fit_sgt.sgt_model(v, q, 0.0, B_BAR, 16 * N)))
                 for q in (fit_sgt.to_q(d, mu, 0.0, 0.3, 1.0, 0.0, 1.0) for d in (1.0, 3.0, 5.5) for mu in (0.3, 3, 20)))
    assert abs((fitted / lap) / (0.07 / 0.70) - 1) < 0.15, (fitted, lap)


def test_mirror_of_the_model():
    # O12.4: I_model(-v | -v0, 1/zeta) = I_model(v | v0, zeta) (symmetric Gauss-Legendre nodes): 1e-11 gl.
    N = fit_sgt.n_nodes_for(M)
    v = np.linspace(-2.5, 2.5, 51)
    q = fit_sgt.to_q(1.2, 1.1, 0.13, 0.45, 3.0, 0.4, 1.35)
    qm = fit_sgt.to_q(1.2, 1.1, -0.13, 0.45, 3.0, 0.4, 1 / 1.35)
    assert np.max(np.abs(fit_sgt.sgt_model(-v, qm, PHI, B_BAR, N) - fit_sgt.sgt_model(v, q, PHI, B_BAR, N))) < 1e-11


def test_objective_by_hand():
    # O12.5: Q = 1/2 sum omega_n ((I_model - I_n)/B_bar)^2, recomputed from sgt_model (exact arithmetic up to
    # the summation order: 1e-15 relative).
    N = fit_sgt.n_nodes_for(M)
    rng = np.random.default_rng(5)
    q = fit_sgt.to_q(1.0, 0.9, 0.05, 0.4, 2.0, 0.2, 1.1)
    I = fit_sgt.sgt_model(V, q, PHI, B_BAR, N) + rng.normal(0, 3, V.size)
    q2 = fit_sgt.to_q(1.1, 0.8, 0.0, 0.5, 2.5, 0.1, 1.0)
    om = np.exp(-np.abs(V - 0.02) / 0.5)
    r = (fit_sgt.sgt_model(V, q2, PHI, B_BAR, N) - I) / B_BAR
    assert abs(fit_sgt.objective(V, I, PHI, B_BAR, q2, om, N) / (0.5 * np.sum(om * r * r)) - 1) < 1e-13
    assert abs(fit_sgt.objective(V, I, PHI, B_BAR, q2, None, N) / (0.5 * np.sum(r * r)) - 1) < 1e-13


# ------------------------------------------------- coordinates and box ----

def test_box_coordinates_and_quadrature_rule():
    # O12.6: the box of the defaults (D-052.2), q's transforms, N = 848 by hand.
    lb, ub = fit_sgt.box(M)
    assert lb.dtype == np.float64 and lb.shape == (7,) and ub.shape == (7,)
    assert_allclose(lb, [0.05, 0.0, -1.0, 0.1, 0.0, 0.0, -0.7], rtol=0, atol=1e-15)
    assert_allclose(ub, [6.0, 20.0, 1.0, 2.0, 1.0, 2.0, 0.7], rtol=0, atol=1e-15)
    assert fit_sgt.PARAM_NAMES == ("d", "mu", "v0", "w", "log10_beta", "xi", "ln_zeta")
    theta = (1.3, 0.7, -0.2, 0.45, 3.0, 0.25, 1.4)
    q = fit_sgt.to_q(*theta)
    assert_allclose(q, [1.3, 0.7, -0.2, 0.45, math.log10(3.0), 0.25, math.log(1.4)], rtol=1e-15)
    assert_allclose(fit_sgt.from_q(q), theta, rtol=1e-14)
    sigma_min = 0.1 / H.FWHM_GAUSS
    assert fit_sgt.n_nodes_for(M) == max(M.fit_quad_min_nodes, math.ceil(M.fit_quad_nodes_per_sigma * 6.0 / sigma_min))
    assert fit_sgt.n_nodes_for(M) == 848


def test_start_points():
    # O12.7: one start per sgt_start_shapes from gauss_fitted's (d, mu, v0, w), then gauss_fixed's (d, mu, v0) at
    # sgt_start_w_um with sgt_base_start_shape; a start with a non-finite entry is left out.
    gf = SimpleNamespace(d_um=1.2, mu_per_um=0.9, v0_um=0.03, fwhm_um=0.42)
    gx = SimpleNamespace(d_um=1.25, mu_per_um=0.85, v0_um=0.02, fwhm_um=0.233)
    st = fit_sgt.start_points(gf, gx, M)
    assert len(st) == len(M.sgt_start_shapes) + 1 == 5
    for (label, q), shape in zip(st[:4], M.sgt_start_shapes):
        assert_allclose(q, fit_sgt.to_q(1.2, 0.9, 0.03, 0.42, *shape), rtol=0, atol=0)
    assert_allclose(st[4][1], fit_sgt.to_q(1.25, 0.85, 0.02, M.sgt_start_w_um, *M.sgt_base_start_shape), rtol=0,
                    atol=0)
    assert len({lab for lab, _ in st}) == 5
    bad = SimpleNamespace(d_um=float("nan"), mu_per_um=0.9, v0_um=0.03, fwhm_um=0.42)
    st2 = fit_sgt.start_points(bad, gx, M)
    assert len(st2) == 1 and np.all(np.isfinite(st2[0][1]))


# -------------------------------------------------------------- the fit ----

def _status_by_hand(fit, cfg=M):
    """O12.10 recomputed from the fit's own q and rule: (status, at_bound names) by the spec's words."""
    lb, ub = fit_sgt.box(cfg)
    names = ("d", "mu", "v0", "w", "beta", "xi", "zeta")
    hits = []
    for n, x, lo, hi in zip(names, fit.q, lb, ub):
        tol = cfg.fit_at_bound_rel_tol * (hi - lo)
        if x - lo <= tol:
            hits.append(n + "_lo")
        if hi - x <= tol:
            hits.append(n + "_hi")
    rule = fit.solve_ok if fit.weighting == "none" else bool(fit.irls_ok)
    if not rule:
        return "failed", tuple(hits)
    return ("at_bound" if any(h.rsplit("_", 1)[0] in ("d", "mu", "v0", "w") for h in hits) else "converged"), \
        tuple(hits)


def _check_fit_consistency(fit, v, I, phi, B):
    # O12.10, O12.11: the fields agree with q, the status rule and the model by their definitions.
    N = fit.n_nodes
    if not np.all(np.isfinite(fit.q)):
        assert fit.status == "failed"
        return
    d, mu, v0, w, beta, xi, zeta = fit_sgt.from_q(fit.q)
    assert_allclose([fit.d_um, fit.mu_per_um, fit.v0_um, fit.w_um, fit.beta, fit.xi, fit.zeta],
                    [d, mu, v0, w, beta, xi, zeta], rtol=1e-15, atol=0)
    assert abs(fit.alpha_hat - mu * d / math.cos(phi)) <= 1e-14 * max(1.0, fit.alpha_hat)
    model = fit_sgt.sgt_model(v, np.asarray(fit.q), phi, B, N)
    assert_array_equal(fit.model_gl, model)
    assert abs(fit.rss_gl2 - np.sum((model - I) ** 2)) <= 1e-12 * max(1.0, fit.rss_gl2)
    assert_array_equal(fit_sgt.fit_curve(fit, v, B), model)
    status, hits = _status_by_hand(fit)
    assert fit.status == status, (fit.status, status, fit.at_bound)
    assert tuple(fit.at_bound) == hits


@pytest.fixture(scope="module")
def noisy():
    """A noisy profile of a 1.2 um tube blurred by a skewed kernel, and its fit_plane."""
    N = fit_sgt.n_nodes_for(M)
    q = fit_sgt.to_q(1.2, 0.9, 0.06, 0.45, 2.5, 0.3, 1.2)
    rng = np.random.default_rng(20261113)
    I = fit_sgt.sgt_model(V, q, PHI, B_BAR, N) + rng.normal(0.0, 2.0, V.size)
    return I, fit_sgt.fit_plane(V, I, PHI, B_BAR, M)


def test_fit_plane_record_and_cost(noisy):
    # O12.8-O12.11 on a noisy profile: the record's fields, the IRLS's stop rule and its cost at the weights of
    # the last solve, the unweighted fit's start count, the status rule by hand.
    I, pf = noisy
    f, u = pf.final, pf.unweighted
    assert f.weighting == "irls" and u.weighting == "none" and fit_sgt.WEIGHTINGS == ("irls", "none")
    assert u.irls_ok is None and u.n_irls == 0 and u.n_starts == 5
    for fit in (f, u):
        _check_fit_consistency(fit, V, I, PHI, B_BAR)
    assert f.irls_ok is True and f.status in ("converged", "at_bound")
    path = [u.q[0:3:2]] + [tuple(p) for p in f.irls_path]          # (d, v0) of theta^(0), then of each solve
    steps = [(abs(b[0] - a[0]), abs(b[1] - a[1])) for a, b in zip(path[:-1], path[1:])]
    tol = M.sgt_irls_tol_um
    assert len(steps) == f.n_irls >= 1
    assert steps[-1][0] <= tol and steps[-1][1] <= tol                # the last step within the tolerance
    assert all(s[0] > tol or s[1] > tol for s in steps[:-1])           # the earlier ones not
    assert_allclose(f.irls_path[-1], (f.d_um, f.v0_um), rtol=0, atol=0)
    v0_prev = path[-2][1]                                              # the weights of the last solve
    omega = np.exp(-np.abs(V - v0_prev) / M.sgt_lambda_um)
    assert abs(f.cost - fit_sgt.objective(V, I, PHI, B_BAR, np.asarray(f.q), omega, f.n_nodes)) <= 1e-12 * f.cost
    assert abs(u.cost - fit_sgt.objective(V, I, PHI, B_BAR, np.asarray(u.q), None, u.n_nodes)) <= 1e-12 * u.cost
    # nesting: the SGT family holds the Gaussian, and a start is gauss_fitted's own fit (1e-6 relative slack)
    assert u.rss_gl2 <= pf.gauss_fitted.rss_gl2 * (1 + 1e-6)


def test_gaussian_comparisons_are_block5s(noisy):
    # Boundary 3/11 -> 12: gauss_fixed is Block 5's Gaussian at sigma_fit (Block 11: equal to fit.fit_profile to
    # 1e-6 um); gauss_fitted's FWHM lies in fit_variants' box (0.1, 2) um, equal to the default sgt_w_bounds_um.
    from allen_diameter.analysis import fit as block3
    from allen_diameter.analysis import fit_variants
    I, pf = noisy
    ref = block3.fit_profile(V, I, PHI, B_BAR, M)
    assert abs(pf.gauss_fixed.d_um - ref.d_hat_um) < 1e-6
    assert fit_variants.FITTED_FWHM_BOUNDS_UM == M.sgt_w_bounds_um
    assert 0.1 <= pf.gauss_fitted.fwhm_um <= 2.0


def test_fit_plane_deterministic(noisy):
    # O12.9: two runs are bit-identical.
    I, pf = noisy
    pf2 = fit_sgt.fit_plane(V, I, PHI, B_BAR, M)
    for a, b in ((pf.final, pf2.final), (pf.unweighted, pf2.unweighted)):
        assert a.q == b.q and a.irls_path == b.irls_path and a.status == b.status and a.cost == b.cost


def test_fit_plane_invariances(noisy):
    # O12.13: a gain on I and B_bar, a shift of v (v0 moves with it), the mirror v -> -v (v0 -> -v0, zeta ->
    # 1/zeta). Tolerance 2e-4 um on d, v0, w: the IRLS stops on steps of sgt_irls_tol_um = 1e-4 um and each
    # solve at ftol = 1e-10 on the cost (about 1e-5 relative on the flat directions), so two runs whose paths
    # differ by rounding (the mirror reverses every sum) may end up to two tolerance steps apart; 1e-3 on the
    # shapes (flatter directions of the objective).
    I, pf = noisy
    f = pf.final
    tol = 2 * M.sgt_irls_tol_um
    g = fit_sgt.fit_plane(V, 1.7 * I, PHI, 1.7 * B_BAR, M).final
    assert_allclose([g.d_um, g.v0_um, g.w_um, g.mu_per_um], [f.d_um, f.v0_um, f.w_um, f.mu_per_um], atol=tol)
    s = fit_sgt.fit_plane(V + 0.1, I, PHI, B_BAR, M).final
    assert_allclose([s.d_um, s.v0_um - 0.1, s.w_um], [f.d_um, f.v0_um, f.w_um], atol=tol)
    m = fit_sgt.fit_plane(-V[::-1], I[::-1], PHI, B_BAR, M).final
    assert_allclose([m.d_um, -m.v0_um, m.w_um], [f.d_um, f.v0_um, f.w_um], atol=tol)
    assert_allclose([m.beta, m.xi], [f.beta, f.xi], atol=1e-3)
    assert abs(math.log(m.zeta) + math.log(f.zeta)) < 1e-3


def test_noise_free_recovery():
    # O12.13: noise-free recovery of theta on a 1.5 um tube (the spec's tolerances: d, v0, w 1e-6; mu 1e-5;
    # shapes 1e-3; rss <= 1e-12 gl^2).
    N = fit_sgt.n_nodes_for(M)
    theta = (1.5, 0.7, 0.04, 0.5, 2.5, 0.2, 1.15)
    I = fit_sgt.sgt_model(V, fit_sgt.to_q(*theta), PHI, B_BAR, N)
    f = fit_sgt.fit_plane(V, I, PHI, B_BAR, M).final
    _check_fit_consistency(f, V, I, PHI, B_BAR)
    assert f.status == "converged" and f.irls_ok
    assert_allclose([f.d_um, f.v0_um, f.w_um], [theta[0], theta[2], theta[3]], atol=1e-6)
    assert abs(f.mu_per_um - theta[1]) < 1e-5
    assert_allclose([f.beta, f.xi, f.zeta], theta[4:], atol=1e-3)
    assert f.rss_gl2 <= 1e-12


def test_shape_at_its_edge_does_not_change_status():
    # O12.10: a noise-free profile blurred by a Gaussian (xi = 0, the box's edge): the fit ends at xi_lo, which
    # at_bound names, and the status stays converged.
    N = fit_sgt.n_nodes_for(M)
    I = fit_sgt.sgt_model(V, fit_sgt.to_q(0.9, 1.1, -0.03, 0.35, 2.0, 0.0, 1.0), PHI, B_BAR, N)
    f = fit_sgt.fit_plane(V, I, PHI, B_BAR, M).final
    _check_fit_consistency(f, V, I, PHI, B_BAR)
    assert "xi_lo" in f.at_bound and f.status == "converged", (f.at_bound, f.status)


def test_irls_cap_and_budget(noisy):
    # O12.9: at a cap of 1 weighted solve whose step exceeds the tolerance, and at a budget of 5 evaluations per
    # solve, irls_ok is False and the status failed.
    I, pf = noisy
    start = pf.unweighted
    first = np.array(pf.final.irls_path[0]) - np.array([start.d_um, start.v0_um])
    assert np.any(np.abs(first) > M.sgt_irls_tol_um)               # the case the oracle needs
    cap = dataclasses.replace(M, sgt_irls_max_iter=1)
    f1 = fit_sgt.fit_irls(V, I, PHI, B_BAR, cap, start)
    assert f1.n_irls == 1 and f1.irls_ok is False and f1.status == "failed"
    bud = dataclasses.replace(M, sgt_max_nfev=5)
    f2 = fit_sgt.fit_irls(V, I, PHI, B_BAR, bud, start)
    assert f2.irls_ok is False and f2.status == "failed" and f2.solve_ok is False
    _check_fit_consistency(f1, V, I, PHI, B_BAR)
    _check_fit_consistency(f2, V, I, PHI, B_BAR)


def test_fit_inputs_refused():
    # O12.12: Block 3's input rule.
    I = B_BAR - 20 * np.exp(-V ** 2)
    bad = [(V[::-1], I, PHI, B_BAR), (V, np.where(V == 0, np.nan, I), PHI, B_BAR), (V, I, math.pi / 2, B_BAR),
           (V, I, -0.1, B_BAR), (V, I, PHI, 0.0), (V, I, PHI, np.nan), (V, I, PHI, -5.0), (V[:3], I[:3], PHI, B_BAR),
           (V, I[:-1], PHI, B_BAR), (V, np.where(V == 0, np.inf, I), PHI, B_BAR)]
    for args in bad:
        with pytest.raises(ValueError):
            fit_sgt.fit_plane(*args, M)
        with pytest.raises(ValueError):
            fit_sgt.fit_unweighted(*args, M, [("x", fit_sgt.to_q(1, 1, 0, 0.5, 2, 0.1, 1))])


# ------------------------------------------------------- configuration ----

def test_config_defaults_and_signature():
    # O12.14: the defaults of the entry; every sgt_* field in the estimator signature.
    assert M.sgt_w_bounds_um == (0.1, 2.0) and M.sgt_beta_bounds == (1.0, 10.0) and M.sgt_xi_bounds == (0.0, 2.0)
    assert M.sgt_ln_zeta_bounds == (-0.7, 0.7) and M.sgt_lambda_um == 0.5 and M.sgt_irls_tol_um == 1e-4
    assert M.sgt_irls_max_iter == 30 and M.sgt_max_nfev == 3000 and M.sgt_start_w_um == 0.7
    assert M.sgt_start_shapes == ((2, 0, 1), (1, 0, 1), (5, 0, 1), (2, 1, 1))
    assert M.sgt_base_start_shape == (2, 0, 1)
    h0 = CFG.signature_hash("estimator")
    changes = {"sgt_w_bounds_um": (0.1, 2.5), "sgt_beta_bounds": (1.0, 9.0), "sgt_xi_bounds": (0.0, 1.5),
               "sgt_ln_zeta_bounds": (-0.6, 0.7), "sgt_lambda_um": 0.6, "sgt_irls_tol_um": 2e-4,
               "sgt_irls_max_iter": 20, "sgt_max_nfev": 2000, "sgt_start_shapes": ((2.0, 0.0, 1.0),),
               "sgt_start_w_um": 0.6, "sgt_base_start_shape": (1.0, 0.0, 1.0)}
    for k, v in changes.items():
        c = dataclasses.replace(CFG, measure=dataclasses.replace(M, **{k: v}))
        c.validate()
        assert c.signature_hash("estimator") != h0, k


def test_config_validate_refusals():
    # O12.14: validate's refusals.
    bad = {"sgt_w_bounds_um": [(0.0, 2.0), (0.5, 0.5), (2.0, 0.1), (0.1, np.inf)],
           "sgt_beta_bounds": [(0.0, 10.0), (3.0, 2.0)], "sgt_xi_bounds": [(-0.1, 2.0), (1.0, 1.0)],
           "sgt_ln_zeta_bounds": [(0.7, -0.7), (0.1, 0.1)], "sgt_lambda_um": [0.0, -1.0, np.nan],
           "sgt_irls_tol_um": [0.0, -1e-4], "sgt_irls_max_iter": [0], "sgt_max_nfev": [0],
           "sgt_start_shapes": [(), ((20.0, 0.0, 1.0),), ((2.0, 3.0, 1.0),), ((2.0, 0.0, 3.0),)],
           "sgt_start_w_um": [0.05, 2.5], "sgt_base_start_shape": [(0.5, 0.0, 1.0)]}
    for k, vals in bad.items():
        for v in vals:
            c = dataclasses.replace(CFG, measure=dataclasses.replace(M, **{k: v}))
            with pytest.raises(ValueError):
                c.validate()


def test_override_coerces_nested_start_shapes():
    # O12.14: overrides of the nested start shapes are coerced to floats (one configuration, one hash).
    a = with_overrides(CFG, {"measure.sgt_start_shapes": [[2, 0, 1], [1, 0, 1], [5, 0, 1], [2, 1, 1]]})
    assert a.signature_hash("estimator") == CFG.signature_hash("estimator")
    assert all(isinstance(x, float) for s in a.measure.sgt_start_shapes for x in s)
    b = with_overrides(CFG, {"measure.sgt_base_start_shape": [2, 0, 1], "measure.sgt_start_w_um": 1})
    assert b.measure.sgt_start_w_um == 1.0 and isinstance(b.measure.sgt_start_w_um, float)
    assert b.signature_hash("estimator") != CFG.signature_hash("estimator")      # the width changed
