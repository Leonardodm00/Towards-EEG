"""Independent references for the spec-derived tests of Blocks 12, 13 and 14
(integration tester, run D3, 2026-11-13). Written from specs/SPEC.md only:
nothing here calls the code under test, except grid_table, which hands
known grid values to the BiasTable constructor (the only way to obtain a
table object whose values are known).

  ref_sgt_pdf          (H1)-(H3) of Block 12, written from the spec's formulas
                       (scipy.special.betaln / gammaln / expm1)
  ref_half_mass        the kernel's mass on [0, X] and [-X, 0] by the closed
                       forms of Block 12's test oracles (betainc / gammainc)
  scipy_member_pdf     a scipy.stats member rescaled to FWHM w (FWHM by brentq)
  ref_model_quad       (3'') as a direct convolution of the transmittance
                       with the kernel, scipy.integrate.quad (no substitution)
  dip_area_exact       Block 3's exact dip area B pi d / 2 (I_1 - L_1)
  np_features / np_nuisances / np_radius
                       Block 14.2's streams recomputed from NumPy alone
  brute_local_linear   the local linear fit of section 2.4 at one point by
                       numpy.linalg.lstsq on explicitly weighted rows
  grid_table           a BiasTable from known grid arrays

Pure ASCII.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from scipy import integrate, optimize, special, stats

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
for _p in (WS / "src", WS / "scripts", WS / "tests" / "smoke"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

FWHM_GAUSS = 2.0 * math.sqrt(2.0 * math.log(2.0))


# ------------------------------------------------------------ Block 12 ----

def ref_core_hwhm(beta, xi):
    """x_h: (1 + xi x^beta)^-(1/xi + 1/beta) = 1/2  <=>  x^beta = (2^e - 1)/xi, e = xi beta/(beta + xi)
    (derived by hand from (H1)); (ln 2)^(1/beta) at xi = 0."""
    if xi <= 1e-12:
        return math.log(2.0) ** (1.0 / beta)
    e = xi * beta / (beta + xi)
    return (math.expm1(e * math.log(2.0)) / xi) ** (1.0 / beta)


def ref_sgt_pdf(z, w, beta, xi, zeta):
    """g_{w,kappa}(z) of (H1)-(H3)."""
    z = np.asarray(z, dtype=float)
    s = zeta + 1.0 / zeta
    W = s * ref_core_hwhm(beta, xi)
    x = W * z / w
    u = np.where(x >= 0, x / zeta, x * zeta)
    if xi <= 1e-12:
        logc = math.log(beta) - math.log(2.0) - special.gammaln(1.0 / beta)
        core = -np.abs(u) ** beta
    else:
        logc = math.log(beta) + math.log(xi) / beta - math.log(2.0) - special.betaln(1.0 / beta, 1.0 / xi)
        core = -(1.0 / xi + 1.0 / beta) * np.log1p(xi * np.abs(u) ** beta)
    return (W / w) * (2.0 / s) * np.exp(logc + core)


def ref_half_mass(X, w, beta, xi, zeta):
    """(mass on [0, X], mass on [-X, 0]) of g_{w,kappa}: (2 zeta / s) H(X/(a zeta)) and (2/(s zeta)) H(zeta X/a),
    s = zeta + 1/zeta, a = w / W, H(y) = I_t(1/beta, 1/xi)/2 with t = xi y^beta/(1 + xi y^beta), or
    P(1/beta, y^beta)/2 at xi = 0 (Block 12's test oracles)."""
    s = zeta + 1.0 / zeta
    a = w / (s * ref_core_hwhm(beta, xi))

    def H(y):
        if xi <= 1e-12:
            return 0.5 * special.gammainc(1.0 / beta, y ** beta)
        t = 1.0 / (1.0 + 1.0 / (xi * y ** beta)) if y > 0 else 0.0     # = xi y^b / (1 + xi y^b), exact at y = inf
        return 0.5 * special.betainc(1.0 / beta, 1.0 / xi, t)
    return (2.0 * zeta / s) * H(X / (a * zeta)), (2.0 / (s * zeta)) * H(zeta * X / a)


def scipy_member_pdf(dist, w):
    """The density of a symmetric scipy.stats distribution (unit scale, mode 0) rescaled to FWHM w: its FWHM
    W_s found by brentq on pdf(x) = pdf(0)/2, then g(z) = (W_s/w) pdf(W_s z / w)."""
    p0 = dist.pdf(0.0)
    xh = optimize.brentq(lambda x: dist.pdf(x) - 0.5 * p0, 1e-12, 1e3, xtol=1e-15, rtol=1e-15)
    Ws = 2.0 * xh

    def g(z):
        return (Ws / w) * dist.pdf(Ws * np.asarray(z, dtype=float) / w)
    return g


def ref_model_quad(v, d, mu, v0, phi, B, g):
    """(3'') by a direct convolution: B [1 - int_{v0-d/2}^{v0+d/2} (1 - exp(-alpha s_d(u - v0))) g(v - u) du],
    alpha = mu d / cos(phi), s_d(u) = sqrt(1 - (2u/d)^2) (section 2.2); quad with the kernel's peak u = v as a
    break point."""
    alpha = mu * d / math.cos(phi)
    lo, hi = v0 - 0.5 * d, v0 + 0.5 * d
    out = []
    for vv in np.atleast_1d(v):
        def f(u):
            s = math.sqrt(max(0.0, 1.0 - (2.0 * (u - v0) / d) ** 2))
            return -math.expm1(-alpha * s) * float(g(vv - u))
        pts = [vv] if lo < vv < hi else None
        val, _ = integrate.quad(f, lo, hi, points=pts, epsabs=1e-14, epsrel=1e-13, limit=500)
        out.append(B * (1.0 - val))
    return np.array(out)


def dip_area_exact(d, alpha, B):
    """Block 3's exact dip area: B (pi d / 2) (I_1(alpha) - L_1(alpha))."""
    return B * 0.5 * math.pi * d * (special.iv(1, alpha) - special.modstruve(1, alpha))


# ------------------------------------------------------------ Block 14 ----

def np_features(seed, g, ph):
    """(d, phi, theta, mu) of draw g from numpy.random.default_rng([seed, g]) in that order, the random design
    with log-uniform d and mu (section 2.4)."""
    rng = np.random.default_rng([seed, g])
    d = math.exp(rng.uniform(math.log(ph.d_range_um[0]), math.log(ph.d_range_um[1])))
    phi = math.radians(rng.uniform(ph.phi_range_deg[0], ph.phi_range_deg[1]))
    theta = rng.uniform(ph.theta_range_rad[0], ph.theta_range_rad[1])
    mu = math.exp(rng.uniform(math.log(ph.mu_range_per_um[0]), math.log(ph.mu_range_per_um[1])))
    return d, phi, theta, mu


def np_nuisances(seed, g, r, p, dz):
    """(cx, cy, cz, the generator after them) from default_rng([seed, g, r]): cx, cy uniform within one pixel,
    cz uniform within +-dz/2 (Block 6's laws, Block 14.2's order)."""
    rng = np.random.default_rng([seed, g, r])
    cx = rng.uniform(-0.5 * p, 0.5 * p)
    cy = rng.uniform(-0.5 * p, 0.5 * p)
    cz = rng.uniform(-0.5 * dz, 0.5 * dz)
    return cx, cy, cz, rng


def np_radius(seed, g, r, d, sd):
    """(r_br, rejections): eps ~ N(0, sd^2) from default_rng([seed, g, r, 1]) until d/2 + eps > 0."""
    rng = np.random.default_rng([seed, g, r, 1])
    k = 0
    while True:
        eps = rng.normal(0.0, sd)
        if 0.5 * d + eps > 0:
            return 0.5 * d + eps, k
        k += 1


def brute_local_linear(X, y, x, h):
    """At one point x (3,): Gaussian weights w_n = exp(-1/2 sum_k ((X_nk - x_k)/h_k)^2), the weighted least
    squares of y on z_n = (1, (X_n - x)/h). Returns (b, ell (n,), w (n,), A (4, 4)) with ell_n = e1' A^-1 z_n w_n."""
    X = np.asarray(X, dtype=float)
    U = (X - np.asarray(x, dtype=float)[None, :]) / np.asarray(h, dtype=float)[None, :]
    w = np.exp(-0.5 * np.sum(U * U, axis=1))
    Z = np.column_stack([np.ones(X.shape[0]), U])
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(Z * sw[:, None], y * sw, rcond=None)
    A = (Z * w[:, None]).T @ Z
    ell = np.linalg.solve(A, Z.T * w[None, :])[0]
    return float(beta[0]), ell, w, A


def grid_table(axes, b, mass=None, p_fail=None, supported=None, q_loo=0.0, estimator_hash="test", se=None):
    """A table.BiasTable whose arrays are known: b (n1, n2, n3) or a function b(lnd, phi, lnmd) on the grid."""
    from allen_diameter.analysis.table import BiasTable
    axes = tuple(np.asarray(a, dtype=float) for a in axes)
    L1, L2, L3 = np.meshgrid(*axes, indexing="ij")
    B = b(L1, L2, L3) if callable(b) else np.asarray(b, dtype=float)
    shape = B.shape
    ones = np.ones(shape)
    arrays = dict(b_hat=B, se_hat=0.01 * ones if se is None else se, se_indep=0.01 * ones, tau_hat=0.03 * ones,
                  n_eff=100.0 * ones, mass=1000.0 * ones if mass is None else mass,
                  p_fail=0.0 * ones if p_fail is None else p_fail)
    sup = np.ones(shape, dtype=bool) if supported is None else supported
    return BiasTable(axes, arrays, sup, dict(estimator_hash=estimator_hash, q_loo=q_loo))
