#!/usr/bin/env python3
"""A blur kernel with three shape parameters: the skewed generalized t (SGT),
two-piece form, parameterised by its FWHM (2026-10-15). Standalone check code,
not pipeline code.

Symmetric core (generalized t), unit area, for x real:
    g(x; p, r) = c(p, r) (1 + r |x|^p)^-(1/r + 1/p)            r > 0
    g(x; p, 0) = c(p, 0) exp(-|x|^p)                           r = 0 (the limit r -> 0)
    c(p, r) = p r^(1/p) / (2 B(1/p, 1/r)),   c(p, 0) = p / (2 Gamma(1/p))
Two-piece skew (gamma > 0; gamma = 1 symmetric; mode stays at 0):
    f(x) = 2/(gamma + 1/gamma) * g(x/gamma)   x >= 0
           2/(gamma + 1/gamma) * g(gamma x)   x <  0
Kernel of FWHM w:  k(z) = f(z/a)/a,  a = w / (h(p, r) (gamma + 1/gamma)),
h(p, r) the half-width at half maximum of g:
    h = ((2^(1/(1/r + 1/p)) - 1) / r)^(1/p)   (r > 0),   h = (ln 2)^(1/p)   (r = 0).

Members (checked in check()): p 2, r 0, gamma 1 Gaussian; p 1, r 0 Laplace;
p 2, r 2/nu Student t with nu degrees of freedom (r 2: Lorentzian); p beta,
r 0 generalized normal (p large: box-like); gamma != 1 the skewed versions.
p shapes the peak and shoulders, r the tail weight (power-law tails of
exponent 1 + p/r when r > 0), gamma the asymmetry.

Custom: scipy.stats has no generalized t or SGT; checked: t, gennorm,
genhyperbolic (no platykurtic members), jf_skew_t, johnsonsu (leptokurtic
only), pearson3 (one Pearson type).

    python sgt_kernel.py      # runs check(), prints PASS/FAIL per item
Pure ASCII.
"""
import math

import numpy as np
from scipy import integrate, optimize, special, stats

R_ZERO = 1e-12          # r at or below this is the generalized-normal limit r = 0


def _log_c(p, r):
    """log of the core's normaliser c(p, r)."""
    if r <= R_ZERO:
        return math.log(p) - math.log(2.0) - special.gammaln(1.0 / p)
    return math.log(p) + math.log(r) / p - math.log(2.0) - special.betaln(1.0 / p, 1.0 / r)


def _log_core(x, p, r):
    """log of (1 + r|x|^p)^-(1/r + 1/p), or -|x|^p at r = 0."""
    ax = np.abs(x)
    if r <= R_ZERO:
        return -(ax ** p)
    return -(1.0 / r + 1.0 / p) * np.log1p(r * ax ** p)


def hwhm_core(p, r):
    """Half-width at half maximum of the symmetric core g(.; p, r) of unit scale."""
    if r <= R_ZERO:
        return math.log(2.0) ** (1.0 / p)
    e = 1.0 / (1.0 / r + 1.0 / p)
    return (math.expm1(e * math.log(2.0)) / r) ** (1.0 / p)     # expm1: 2^e - 1 without cancellation at small r


def sgt_pdf(fwhm, p, r, gamma):
    """The kernel k(z) (1/um) of FWHM fwhm (um), mode at z = 0, unit area: a
    vectorised callable of z (um)."""
    if not (fwhm > 0 and p > 0 and r >= 0 and gamma > 0):
        raise ValueError("need fwhm > 0, p > 0, r >= 0, gamma > 0; got %r" % ((fwhm, p, r, gamma),))
    s = gamma + 1.0 / gamma
    a = fwhm / (hwhm_core(p, r) * s)
    log_lead = _log_c(p, r) + math.log(2.0 / s) - math.log(a)

    def k(z):
        x = np.asarray(z, dtype=float) / a
        u = np.where(x >= 0.0, x / gamma, x * gamma)
        return np.exp(log_lead + _log_core(u, p, r))
    return k


def check(bk=None):
    """Numerical checks of the family; bk is the pipeline's blur_kernels module
    (optional: the comparison with its kernels is skipped without it)."""
    res = []
    w = 0.5
    grid = [(p, r, g) for p in (1.0, 2.0, 4.0, 10.0) for r in (0.0, 1e-9, 1e-3, 0.5, 2.0) for g in (1.0, 0.6, 1.8)]
    worst_area, worst_fwhm = 0.0, 0.0
    for p, r, g in grid:
        k = sgt_pdf(w, p, r, g)
        area = (integrate.quad(k, -np.inf, 0.0, epsabs=1e-13, epsrel=1e-12, limit=400)[0]
                + integrate.quad(k, 0.0, np.inf, epsabs=1e-13, epsrel=1e-12, limit=400)[0])
        worst_area = max(worst_area, abs(area - 1.0))
        k0 = float(k(0.0))
        right = optimize.brentq(lambda x: float(k(x)) - k0 / 2.0, 1e-9, 50.0, xtol=1e-14)
        left = optimize.brentq(lambda x: float(k(-x)) - k0 / 2.0, 1e-9, 50.0, xtol=1e-14)
        worst_fwhm = max(worst_fwhm, abs(right + left - w))
        if abs(g - 1.0) > 0 and abs(right / left - g * g) > 1e-9:
            res.append(("skew ratio p=%g r=%g g=%g" % (p, r, g), False, right / left))
    # heavy tails (r = 2, p = 1: tail exponent 1.5) converge slowly under quad: 1e-6 tolerance
    res.append(("unit area (worst |area - 1| over %d members)" % len(grid), worst_area < 1e-6, worst_area))
    res.append(("FWHM = w (worst |FWHM - w|, um)", worst_fwhm < 1e-10, worst_fwhm))
    z = np.linspace(-3.0, 3.0, 6001)
    sig = w / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    d_gauss = np.max(np.abs(sgt_pdf(w, 2.0, 0.0, 1.0)(z) - stats.norm.pdf(z, scale=sig)))
    res.append(("p 2, r 0 = Gaussian (max abs diff, 1/um)", d_gauss < 1e-12, d_gauss))
    d_lap = np.max(np.abs(sgt_pdf(w, 1.0, 0.0, 1.0)(z) - stats.laplace.pdf(z, scale=w / (2.0 * math.log(2.0)))))
    res.append(("p 1, r 0 = Laplace", d_lap < 1e-12, d_lap))
    d_lim = np.max(np.abs(sgt_pdf(w, 2.0, 1e-9, 1.0)(z) - sgt_pdf(w, 2.0, 0.0, 1.0)(z)))
    res.append(("continuity at r -> 0 (r 1e-9 vs 0)", d_lim < 1e-6, d_lim))
    if bk is not None:
        for nu in (1.0, 1.5, 4.0, 100.0):
            dd = np.max(np.abs(sgt_pdf(w, 2.0, 2.0 / nu, 1.0)(z) - bk.kernel_pdf("student", w, nu)(z)))
            res.append(("p 2, r 2/nu = pipeline student nu=%g" % nu, dd < 1e-10, dd))
        dd = np.max(np.abs(sgt_pdf(w, 2.0, 2.0, 1.0)(z) - bk.kernel_pdf("lorentz", w, None)(z)))
        res.append(("p 2, r 2 = pipeline lorentz", dd < 1e-10, dd))
        for beta in (1.0, 4.0):
            dd = np.max(np.abs(sgt_pdf(w, beta, 0.0, 1.0)(z) - bk.kernel_pdf("gennorm", w, beta)(z)))
            res.append(("p %g, r 0 = pipeline gennorm beta=%g" % (beta, beta), dd < 1e-10, dd))
    ok = all(r[1] for r in res)
    for name, passed, val in res:
        print("%-48s %s  (%.3e)" % (name, "PASS" if passed else "FAIL", val))
    print("ALL PASS" if ok else "SOME FAIL")
    return ok


if __name__ == "__main__":
    import sys
    bk = None
    if len(sys.argv) > 1:
        import check_objective as CO
        bk = CO.load(sys.argv[1]).BK
    check(bk)
