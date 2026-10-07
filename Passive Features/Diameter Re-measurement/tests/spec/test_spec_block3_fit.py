"""Block 3 (forward model and fitter): oracles from SPEC.md section 2.2 Eq. 3 and Block 3.
Expected values come from scipy.integrate.quad, scipy.special and hand formulas."""
import dataclasses
import math

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import integrate, special

from allen_diameter.config import default_config
from allen_diameter.model import tube_model as TM
from allen_diameter.analysis import fit as F

CFG = default_config().measure
N = TM.n_quadrature_nodes(CFG.fit_d_bounds_um[1], CFG.sigma_fit_um, CFG.fit_quad_min_nodes,
                          CFG.fit_quad_nodes_per_sigma)


def _ref_profile(v, d, alpha, v0, sigma, B):
    """Eq. 3 unsubstituted: B (T * g)(v), T = exp(-alpha sqrt(1 - (2u/d)^2)) inside |u| <= d/2."""
    def integrand(u):
        s = math.sqrt(max(0.0, 1 - (2 * u / d) ** 2))
        return (1 - math.exp(-alpha * s)) * math.exp(-0.5 * ((v - v0 - u) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))
    val, _ = integrate.quad(integrand, -d / 2, d / 2, epsabs=1e-14, epsrel=1e-13, limit=400,
                            points=[v - v0] if abs(v - v0) < d / 2 else None)
    return B * (1 - val)


def test_n_nodes_rule():
    assert N == max(CFG.fit_quad_min_nodes, math.ceil(CFG.fit_quad_nodes_per_sigma * CFG.fit_d_bounds_um[1] / CFG.sigma_fit_um))


@pytest.mark.parametrize("d", [0.05, 0.5, 2.0, 6.0])
@pytest.mark.parametrize("alpha", [0.3, 3.0, 30.0])
def test_model_vs_quad(d, alpha):
    B, sigma, v0 = 200.0, 0.099, 0.07
    v = np.linspace(-d - 0.5, d + 0.5, 7)
    got = TM.model_profile(v, d, alpha, v0, sigma, B, N)
    ref = np.array([_ref_profile(x, d, alpha, v0, sigma, B) for x in v])
    # 1e-9 B: quad's own accuracy on a kink-free but steep integrand, plus the rule's error (spec claims 1e-12 B)
    assert_allclose(got, ref, atol=1e-9 * B, rtol=0)


@pytest.mark.parametrize("alpha", [1e-3, 0.3, 3.0, 10.0])
def test_dip_area_bessel_struve(alpha):
    B, d, sigma = 1.0, 1.3, 0.1
    v = np.linspace(-4, 4, 160001)
    I = TM.model_profile(v, d, alpha, 0.0, sigma, B, N)
    area = np.trapezoid(B - I, v)
    exact = B * math.pi * d / 2 * (special.iv(1, alpha) - special.modstruve(1, alpha))
    # trapezoid on 5e-5 um samples of a smooth function, Gaussian tails < 1e-30 at 4 um: 1e-8 relative
    assert_allclose(area, exact, rtol=1e-8)


def test_faint_limit_moments():
    d, sigma, v0, alpha = 0.8, 0.1, 0.13, 1e-6
    v = np.linspace(-3, 3, 120001)
    w = 1.0 - TM.model_profile(v, d, alpha, v0, sigma, 1.0, N)
    m0 = np.trapezoid(w, v)
    m1 = np.trapezoid(w * v, v) / m0
    var = np.trapezoid(w * (v - m1) ** 2, v) / m0
    assert_allclose(m1, v0, rtol=1e-6)
    # first order in alpha the weight is the semicircle: variance d^2/16 + sigma^2; O(alpha) correction ~1e-6
    assert_allclose(var, sigma ** 2 + d ** 2 / 16, rtol=1e-5)


def test_alpha_from_mu():
    assert TM.alpha_from_mu(0.7, 1.2, 0.4) == pytest.approx(0.7 * 1.2 / math.cos(0.4), rel=1e-15)
    with pytest.raises(ValueError):
        TM.alpha_from_mu(0.7, 1.2, math.pi / 2)


@pytest.mark.parametrize("d", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("phi_deg", [0, 30, 60])
@pytest.mark.parametrize("alpha", [0.3, 3.0])
def test_noise_free_recovery(d, phi_deg, alpha):
    phi = math.radians(phi_deg)
    mu = alpha * math.cos(phi) / d
    B = 205.0
    v = np.arange(-26, 27) * 0.1144
    I = TM.model_profile(v, d, TM.alpha_from_mu(mu, d, phi), 0.05, CFG.sigma_fit_um, B, N)
    r = F.fit_profile(v, I, phi, B, CFG)
    assert r.status == "converged"
    assert_allclose([r.d_hat_um, r.mu_hat_per_um, r.v0_hat_um], [d, mu, 0.05], rtol=1e-6, atol=1e-8)
    assert_allclose(r.alpha_hat, r.mu_hat_per_um * r.d_hat_um / math.cos(phi), rtol=1e-12)
    # rss in grey levels squared, recomputed from the reported parameters
    model = TM.model_profile(v, r.d_hat_um, r.alpha_hat, r.v0_hat_um, CFG.sigma_fit_um, B, r.n_nodes)
    assert_allclose(r.rss_gl2, np.sum((I - model) ** 2), atol=1e-9)


def test_mu_and_alpha_forms_agree():
    sigma_true = 0.10
    v = np.arange(-26, 27) * 0.1144
    phi = math.radians(15)
    cfg = dataclasses.replace(CFG, sigma_fit_um=0.125)
    for mu in (0.6,):
        for d in (0.5, 1.0, 2.0):
            I = TM.model_profile(v, d, TM.alpha_from_mu(mu, d, phi), 0.0, sigma_true, 200.0, N)
            a = F.fit_profile(v, I, phi, 200.0, cfg)
            b = F.fit_profile_alpha(v, I, phi, 200.0, cfg)
            if a.status == "converged" and b.status == "converged":
                assert abs(a.d_hat_um - b.d_hat_um) <= 2e-3 * d


def test_half_depth_width_triangle_and_start_rule():
    v = np.arange(-30, 31) * 0.1
    B = 100.0
    I = B - np.clip(40.0 * (1 - np.abs(v) / 1.0), 0, None)   # triangle dip depth 40, half width 1 um at base
    n = int(np.argmin(I))
    assert_allclose(F.half_depth_width(v, I, B, n), 1.0, rtol=1e-12)    # FWHM of a triangle = base half-width x 1
    st = F.start_points(v, I, 0.0, B, CFG)
    d0 = math.sqrt(1.0 - 8 * math.log(2) * CFG.sigma_fit_um ** 2)
    assert_allclose(st[:, 0], np.clip(d0 * np.array(CFG.fit_multistart_factors), *CFG.fit_d_bounds_um), rtol=1e-12)
    alpha0 = -math.log(60.0 / 100.0)
    assert_allclose(st[:, 1], alpha0 / st[:, 0], rtol=1e-12)
    assert np.all(st[:, 2] == 0.0)


def test_flat_and_wide_profiles():
    v = np.arange(-26, 27) * 0.1144
    r = F.fit_profile(v, np.full(v.size, 200.0), 0.1, 200.0, CFG)
    assert r.status == "at_bound" and "mu_lo" in r.at_bound
    I = TM.model_profile(v, 8.0, 1.0, 0.0, 0.099, 200.0, N)
    r = F.fit_profile(v, I, 0.0, 200.0, CFG)
    assert "d_hi" in r.at_bound and r.status == "at_bound"


def test_determinism_and_invalid_inputs():
    v = np.arange(-26, 27) * 0.1144
    I = TM.model_profile(v, 1.0, 0.8, 0.0, 0.099, 200.0, N) + np.random.default_rng(0).normal(0, 2, v.size)
    a, b = F.fit_profile(v, I, 0.2, 200.0, CFG), F.fit_profile(v, I, 0.2, 200.0, CFG)
    assert a == b
    bad = [(v[:3], I[:3], 0.2, 200.0), (v[::-1], I, 0.2, 200.0), (v, np.where(v > 0, np.nan, I), 0.2, 200.0),
           (v, I, 0.2, 0.0), (v, I, 0.2, float("nan")), (v, I, math.pi / 2, 200.0), (v, I, float("nan"), 200.0),
           (v + 10.0, I, 0.2, 200.0)]
    for args in bad:
        with pytest.raises(ValueError):
            F.fit_profile(*args, CFG)
    with pytest.raises(NotImplementedError):
        F.fit_profile(v, I, 0.2, 200.0, dataclasses.replace(CFG, mu_mode="shared_branch"))
