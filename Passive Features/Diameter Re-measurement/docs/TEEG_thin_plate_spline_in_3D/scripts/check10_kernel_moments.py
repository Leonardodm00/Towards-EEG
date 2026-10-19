"""check10_kernel_moments.py -- why the weights of a high-order smoother must go negative: the moments of the
equivalent kernel E of eq. (25) on R^3 (2026-10-18; standalone). Document section 3.9.

Claim (reasoning, eqs. (24)-(25), fixed before running): T(nu) = 1 / (1 + (|nu| / nu_c)^(2m)) = 1 - (|nu| / nu_c)^(2m) + ...
For 2m > 2 the expansion has no |nu|^2 term, so the second moment M2 = int |u|^2 E(u) du = -(Laplacian of T)(0) is
zero, while the zeroth moment M0 = int E(u) du = T(0) = 1. A kernel with M2 = 0 and M0 = 1 cannot be non-negative.
For 2m = 2 (order 1; on R^3 it gives no spline for point data, it enters only as the comparison filter) E is the
positive kernel nu_c^2 exp(-nu_c r) / (4 pi r), with M2 = 6 / nu_c^2.
Computed at nu_c = 1 (scale-free): E(r) = (2 pi^2 r)^-1 int_0^inf t sin(t r) T(t) dt by QAWF quadrature on a grid
of r in [dr, R], dr = 0.005, R = 60, with the radial density 4 pi r^2 E = 0 at r = 0 added exactly;
M0 = int 4 pi r^2 E dr and M2 = int 4 pi r^4 E dr by the trapezoid rule; min E over the grid.
The tail beyond R is left out: for order 5/2 E decays like r^-8, which bounds the omitted part of M2 by about
1e-3; the other orders decay exponentially.
Orders m = 1, 2, 5/2, 3. Pass rules (fixed before running): |M0 - 1| <= 1e-3 for every order; order 1:
|M2 / 6 - 1| <= 1e-2 and min E >= -1e-6 max E; 2m > 2: |M2| <= 0.01 (against 6 for order 1) and min E < 0.
Added after the first run, reported without a tolerance: for order 5/2, r^8 E(r) at large r against the coefficient
-c of the |nu|^5 term of T, c = 2^5 Gamma(4) / (pi^(3/2) Gamma(-5/2)) (inverse FT of |nu|^(2a) on R^3, a = 5/2), and the
omitted tail of M2 beyond R, 4 pi (-c) / (3 R^3), added to the computed M2.
Run (same folder):  python3 check10_kernel_moments.py      (about 15 s)
"""
import math

import numpy as np
from scipy import integrate

CFG = dict(orders=(1.0, 2.0, 2.5, 3.0), r_max=60.0, dr=0.005, tol_m0=1e-3, tol_m2_rel=1e-2,
           tol_m2_abs=0.01, tol_neg=1e-6, far_r=(30.0, 40.0, 50.0))


def E_of_r(r, m):
    """Equivalent kernel on R^3 at distance r, nu_c = 1: (2 pi^2 r)^-1 int_0^inf t sin(t r) / (1 + t^(2m)) dt."""
    val = integrate.quad(lambda t: t / (1.0 + t ** (2 * m)), 0, np.inf, weight="sin", wvar=r, limlst=400)[0]
    return val / (2 * math.pi ** 2 * r)


def moments(m):
    r = np.arange(CFG["dr"], CFG["r_max"] + 1e-12, CFG["dr"])
    E = np.array([E_of_r(x, m) for x in r])
    w = 4 * math.pi * r ** 2 * E
    # the radial density w vanishes at r = 0 for every order (E is at most ~ 1/r there), so the grid starts at
    # r = dr and the point (0, 0) is added exactly; QAWF is not reliable at very small r (very long first cycle)
    r0 = np.concatenate([[0.0], r])
    w0 = np.concatenate([[0.0], w])
    m0 = float(np.trapezoid(w0, r0))
    m2 = float(np.trapezoid(w0 * r0 ** 2, r0))
    return m0, m2, float(E.min()), float(E.max()), float(r[np.argmin(E)])


def main():
    ok = True
    for m in CFG["orders"]:
        m0, m2, emin, emax, rmin = moments(m)
        good0 = abs(m0 - 1) <= CFG["tol_m0"]
        if 2 * m <= 2:
            good2 = abs(m2 / 6.0 - 1) <= CFG["tol_m2_rel"]
            goodn = emin >= -CFG["tol_neg"] * emax
            rule = "M2 = 6 within 1 %, E >= 0"
        else:
            good2 = abs(m2) <= CFG["tol_m2_abs"]
            goodn = emin < 0
            rule = "M2 = 0 within 0.01, min E < 0"
        good = good0 and good2 and goodn
        ok &= good
        print("order %-3g (2m = %g): M0 = %.5f; M2 = %+.5f; min E = %+.3e at r = %.2f (max E %.3e); rule: %s -> %s"
              % (m, 2 * m, m0, m2, emin, rmin, emax, rule, "PASS" if good else "FAIL"))
    a = 2.5                                         # far field of order 5/2 (reported, no tolerance)
    coef = -(2 ** (2 * a) * math.gamma(a + 1.5) / (math.pi ** 1.5 * math.gamma(-a)))
    far = ", ".join("%.0f: %.2f" % (r, r ** 8 * E_of_r(r, a)) for r in CFG["far_r"])
    tail = 4 * math.pi * coef / (3 * CFG["r_max"] ** 3)
    m2_52 = moments(a)[1]
    print("order 5/2 far field: r^8 E(r) at r = %s (predicted %.2f); omitted tail of M2 beyond R = %.1e; "
          "M2 + tail = %+.1e" % (far, coef, tail, m2_52 + tail))
    print("overall:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
