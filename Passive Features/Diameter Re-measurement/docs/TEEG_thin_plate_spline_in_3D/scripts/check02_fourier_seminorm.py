"""check02_fourier_seminorm.py -- the Fourier side of the polyharmonic kernels and the identity
"penalty of the fitted function = c^T K c" (2026-10-18; standalone). Document sections 3.4-3.7.

Convention: f_hat(nu) = int f(x) exp(-i nu.x) dx, f(x) = (2 pi)^-d int f_hat(nu) exp(i nu.x) dnu.
(a) sympy, exact: FT of |x|^-2 on R^3 is 2 pi^2 / |nu|, from (4 pi / nu) int_0^inf sin(nu r)/r dr.
(b) sympy, exact: the generalised FTs that follow from check01 (FT(Delta^2 g) = |nu|^4 g_hat, nu != 0):
      R^3, r^2 ln r: 6 * 2 pi^2 / |nu|^5 = 12 pi^2 |nu|^-5;  R^3, -r: 8 pi |nu|^-4;  R^2, r^2 ln r: 8 pi |nu|^-4;
    against Wendland (2005) Thms 8.16-8.17 as recalled (textbook, from memory), converted to this convention by
    (2 pi)^(d/2):  |x|^(2k) ln|x| -> (-1)^(k+1) 2^(2k-1+d/2) Gamma(k+d/2) k! |w|^(-d-2k);
                   |x|^b         -> 2^(b+d/2) Gamma((d+b)/2) / Gamma(-b/2) |w|^(-b-d).
(c) numerics: for random centres u_n and coefficients c with P^T c = 0 (P = [1, u], degree 1),
      (2 pi)^-d int |c_hat(nu)|^2 phi_hat(nu) dnu  ==  c^T K c,   c_hat(nu) = sum_n c_n exp(-i nu.u_n),
    by the radial reduction (angular averages: sinc(nu r) on R^3, J0(nu r) on R^2):
      R^3, r^2 ln r: 6 int_0^inf S3(nu) nu^-3 dnu;  R^3, -r: (4/pi) int_0^inf S3(nu) nu^-2 dnu;
      R^2, r^2 ln r: 4 int_0^inf S2(nu) nu^-3 dnu;   S3 = sum c_n c_m sinc(nu r_nm), S2 = sum c_n c_m J0(nu r_nm).
    Consequences (Parseval with f_hat = c_hat phi_hat at nu != 0): bending energy on R^2 = 8 pi c^T K c;
    bending energy on R^3 of a '-r' fit = 8 pi c^T K c; the order-5/2 penalty of an r^2 ln r fit on R^3,
    (2 pi)^-3 int |nu|^5 |f_hat|^2, = 12 pi^2 c^T K c.  Tolerance: relative 1e-6.
(d) numerics: r^2 ln r on R^3 with only sum c = 0 (degree 0): the integral from nu_min diverges like
    ln(1/nu_min) -- the reason the affine side conditions are needed (conditional positive definiteness of order 2).
    For -r, sum c = 0 is enough (the integrand stays bounded at nu -> 0).
(e) numerics: the real-space form of the order-1/2 seminorm on R^3,
      int int |g(x) - g(y)|^2 / |x - y|^4 dx dy = C (2 pi)^-3 int |nu| |g_hat|^2 dnu,   C = 2 pi^2,
    C from 2 int (1 - cos xi_1) / |xi|^4 dxi, and both sides for g(x) = exp(-|x|^2 / 2) by 1-D quadratures.
Run:  python3 check02_fourier_seminorm.py      (about 1 min)
"""
import math

import numpy as np
import sympy as sp
from scipy import integrate, special

from tps_common import KERNELS, affine_null_basis, pairwise

CFG = dict(seed=20261022, n_centres=7, n_trials=3, nu_lo=1e-3, nu_hi=400.0, quad_limit=4000, rtol=1e-6,
           nu_min_list=(1e-1, 1e-2, 1e-3, 1e-4))


# ------------------------------------------------------------- (a), (b) ----
def symbolic():
    r, nu = sp.symbols("r nu", positive=True)
    ft_inv_r2 = sp.simplify(4 * sp.pi / nu * sp.integrate(sp.sin(nu * r) / r, (r, 0, sp.oo)))
    ok = sp.simplify(ft_inv_r2 - 2 * sp.pi ** 2 / nu) == 0
    print("(a) FT of |x|^-2 on R^3 = %s  (predicted 2 pi^2/nu) %s" % (ft_inv_r2, "PASS" if ok else "FAIL"))
    derived = {"R3 r^2 ln r": 6 * ft_inv_r2 / nu ** 4, "R3 -r": 8 * sp.pi / nu ** 4, "R2 r^2 ln r": 8 * sp.pi / nu ** 4}

    def wend_log(k, d):
        return (2 * sp.pi) ** sp.Rational(d, 2) * (-1) ** (k + 1) * 2 ** (2 * k - 1 + sp.Rational(d, 2)) \
            * sp.gamma(k + sp.Rational(d, 2)) * sp.factorial(k) / nu ** (d + 2 * k)

    def wend_pow(b, d):
        return (2 * sp.pi) ** sp.Rational(d, 2) * 2 ** (b + sp.Rational(d, 2)) * sp.gamma(sp.Rational(d + b, 2)) \
            / sp.gamma(-sp.Rational(b, 2)) / nu ** (b + d)

    recalled = {"R3 r^2 ln r": wend_log(1, 3), "R3 -r": -wend_pow(1, 3), "R2 r^2 ln r": wend_log(1, 2)}
    for key in derived:
        same = sp.simplify(derived[key] - recalled[key]) == 0
        ok &= same
        print("(b) %-12s derived %-16s recalled (Wendland) %-16s %s"
              % (key, sp.simplify(derived[key]), sp.simplify(recalled[key]), "PASS" if same else "FAIL"))
    return ok


# ------------------------------------------------------------------ (c) ----
def radial_integral(S, p, lo, hi):
    """int_0^inf S(nu) nu^-p dnu: quad on [lo, hi]; below lo S ~ a nu^4 is negligible (reported);
    above hi S -> S(inf) = sum c_n^2 (oscillating parts decay), tail S(inf) hi^(1-p)/(p-1)."""
    val, err = integrate.quad(lambda v: S(v) / v ** p, lo, hi, limit=CFG["quad_limit"], epsabs=1e-13, epsrel=1e-11)
    return val, err


def S3_factory(c, R):
    def S(v):
        x = v * R
        return float(np.sum(np.outer(c, c) * np.sinc(x / math.pi)))   # numpy sinc(t) = sin(pi t)/(pi t)
    return S


def S2_factory(c, R):
    def S(v):
        return float(np.sum(np.outer(c, c) * special.j0(v * R)))
    return S


def seminorm_identity(rng):
    ok = True
    for trial in range(CFG["n_trials"]):
        for d, kernel, p, pref, label in ((3, "thin_plate_spline", 3, 6.0, "R3 r^2 ln r"),
                                          (3, "linear", 2, 4.0 / math.pi, "R3 -r"),
                                          (2, "thin_plate_spline", 3, 4.0, "R2 r^2 ln r")):
            U = rng.uniform(0, 1, (CFG["n_centres"], d))
            Z = affine_null_basis(U, 1)
            c = Z @ rng.standard_normal(Z.shape[1])
            R = pairwise(U, U)
            K = KERNELS[kernel](R)
            quad_form = float(c @ K @ c)
            S = S3_factory(c, R) if d == 3 else S2_factory(c, R)
            val, err = radial_integral(S, p, CFG["nu_lo"], CFG["nu_hi"])
            tail = float(np.sum(c * c)) * CFG["nu_hi"] ** (1 - p) / (p - 1)
            fourier = pref * (val + tail)
            rel = abs(fourier - quad_form) / abs(quad_form)
            good = rel <= CFG["rtol"]
            ok &= good
            print("(c) trial %d %-12s c^T K c = %+.8f   Fourier integral = %+.8f   rel diff %.1e %s"
                  % (trial, label, quad_form, fourier, rel, "PASS" if good else "FAIL"))
    return ok


# ------------------------------------------------------------------ (d) ----
def divergence_demo(rng):
    U = rng.uniform(0, 1, (CFG["n_centres"], 3))
    R = pairwise(U, U)
    Z0 = affine_null_basis(U, 0)
    c = Z0 @ rng.standard_normal(Z0.shape[1])          # only sum c = 0
    S = S3_factory(c, R)
    print("(d) only sum c = 0 (affine moments not zero): sum c_n u_n = %s" % np.round(c @ U, 3))
    prev = None
    for nu_min in CFG["nu_min_list"]:
        v_tps, _ = integrate.quad(lambda v: S(v) / v ** 3, nu_min, 1.0, limit=CFG["quad_limit"])
        v_lin, _ = integrate.quad(lambda v: S(v) / v ** 2, nu_min, 1.0, limit=CFG["quad_limit"])
        step = "" if prev is None else "  increment per decade %.4f" % (v_tps - prev)
        prev = v_tps
        print("    nu_min %.0e: int S nu^-3 (r^2 ln r) = %+.4f%s | int S nu^-2 (-r) = %+.6f"
              % (nu_min, v_tps, step, v_lin))
    a2 = float(np.sum(np.outer(c, c) * R ** 2)) / 6.0       # S ~ -a2 nu^2 at small nu
    print("    predicted increment per decade for r^2 ln r: -(sum c c r^2 / 6) ln 10 = %+.4f" % (-a2 * math.log(10)))


# ------------------------------------------------------------------ (e) ----
def gagliardo():
    def inner(t):
        if t == 0:
            return 0.0
        b = 50.0 / t
        main, _ = integrate.quad(lambda p: (1 - math.cos(p * t)) / p ** 2, 0, b, limit=2000)
        tail_cos, _ = integrate.quad(lambda p: 1.0 / p ** 2, b, np.inf, weight="cos", wvar=t)
        return main + 1.0 / b - tail_cos
    val, _ = integrate.quad(lambda t: inner(abs(t)), -1, 1, limit=200)
    C = 2 * 2 * math.pi * val
    okc = abs(C / (2 * math.pi ** 2) - 1) <= 1e-6
    print("(e) C(3, 1/2) = 2 int (1 - cos xi_1)/|xi|^4 dxi = %.8f ; 2 pi^2 = %.8f %s"
          % (C, 2 * math.pi ** 2, "PASS" if okc else "FAIL"))
    # Gaussian test function g(x) = exp(-|x|^2/2): separable 1-D quadratures, isotropy for the radial integrals
    def I1(a):
        return integrate.quad(lambda x: math.exp(-0.5 * x * x - 0.5 * (x + a) ** 2), -np.inf, np.inf)[0]
    i0 = I1(0.0)

    def D(rho):                                  # int |g(x+z) - g(x)|^2 dx at |z| = rho
        return 2 * i0 ** 3 - 2 * I1(rho) * i0 ** 2
    lhs = 4 * math.pi * integrate.quad(lambda rho: D(rho) / rho ** 2, 0, np.inf, limit=400)[0]

    g0 = integrate.quad(lambda x: math.exp(-0.5 * x * x), -np.inf, np.inf)[0]     # 1-D FT at 0

    def g1hat_full(s):                          # 1-D FT of exp(-x^2/2) (even): 2 int_0^inf cos(s x) g(x) dx
        return 2 * integrate.quad(lambda x: math.exp(-0.5 * x * x), 0, np.inf, weight="cos", wvar=s)[0] if s > 0 else g0
    rhs_int = (2 * math.pi) ** -3 * 4 * math.pi * integrate.quad(
        lambda s: s ** 3 * (g1hat_full(s) * g0 ** 2) ** 2, 0, 40, limit=400)[0]
    ratio = lhs / rhs_int
    okg = abs(ratio / (2 * math.pi ** 2) - 1) <= 1e-6
    print("    Gaussian g: LHS = %.8f (4 pi^3 = %.8f), (2 pi)^-3 int |nu| |g_hat|^2 = %.8f (2 pi = %.8f), "
          "ratio %.8f vs 2 pi^2 %.8f %s" % (lhs, 4 * math.pi ** 3, rhs_int, 2 * math.pi, ratio, 2 * math.pi ** 2,
                                          "PASS" if okg else "FAIL"))
    return okc and okg


def main():
    rng = np.random.default_rng(CFG["seed"])
    ok = symbolic()
    ok &= seminorm_identity(rng)
    divergence_demo(rng)
    ok &= gagliardo()
    print("overall:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
