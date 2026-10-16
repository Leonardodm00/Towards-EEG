#!/usr/bin/env python3
"""Checks of the formulas in the identifiability equations (2026-10-15).

(i)   W_nu = 2 sqrt(nu (2^(2/(nu+1)) - 1)) is the FWHM of the standard Student
      t density: f_nu(W_nu/2) / f_nu(0) = 1/2 (scipy.stats.t).
(ii)  The Student excess kurtosis is 6/(nu - 4) for nu > 4 (scipy.stats.t).
(iii) The pipeline's kernel blur_kernels.kernel_pdf("student", w, nu) has FWHM w.
(iv)  kappa_4 / kappa_2^2 of the exact noise-free dips D = 1 - I_model/B_bar
      (identifiability_check.dip_cumulants, +-40 um) for d_true 0.6 and 1.5 um,
      nu_true 100, w_true 0.5 um, mu_true 1 /um, phi 0.2 rad; with the faint-limit
      formula (kappa_4(D) = -d^4/256 + kappa_4(g), kappa_2(D) = d^2/16 + kappa_2(g))
      beside it.
Tolerance: (i)-(iii) to 1e-9 relative (float64 roundoff on a 2e5-point grid
for (iii), whose step 1e-5 um sets its resolution); (iv) is reported, not tested.

    python kernel_formula_check.py --repo "<repo>/Passive Features/Diameter Re-measurement"
Pure ASCII.
"""
import argparse
import math

import numpy as np
from scipy import stats

import check_objective as CO
import identifiability_check as IC

NUS = (1.0, 1.5, 4.0, 6.03, 100.0)


def w_nu(nu):
    return 2.0 * math.sqrt(nu * (2.0 ** (2.0 / (nu + 1.0)) - 1.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    a = ap.parse_args()
    P = CO.load(a.repo)
    c = IC.CONFIG
    ok = True
    for nu in NUS:
        r = stats.t.pdf(w_nu(nu) / 2.0, nu) / stats.t.pdf(0.0, nu)
        line = "nu %6.2f  W_nu %.6f  f(W/2)/f(0) %.12f" % (nu, w_nu(nu), r)
        ok &= abs(r - 0.5) < 1e-9
        if nu > 4:
            k = float(stats.t.stats(nu, moments="k"))
            line += "  excess kurtosis %.6f vs 6/(nu-4) %.6f" % (k, 6.0 / (nu - 4.0))
            ok &= abs(k - 6.0 / (nu - 4.0)) < 1e-9 * max(1.0, k)
        print(line)
    print("Gaussian limit 2 sqrt(2 ln 2) = %.6f" % (2.0 * math.sqrt(2.0 * math.log(2.0))))
    z = np.linspace(0.0, 2.0, 200001)
    for nu in (1.5, 4.0, 100.0):
        gz = P.BK.kernel_pdf("student", c["w"], nu)(z)
        fwhm = 2.0 * z[np.argmin(np.abs(gz - gz[0] / 2.0))]
        print("pipeline kernel nu %g: FWHM %.5f um (w %.2f)" % (nu, fwhm, c["w"]))
        ok &= abs(fwhm - c["w"]) <= 2e-5
    s2 = (c["w"] / w_nu(100.0)) ** 2 * 100.0 / 98.0
    k4g = 6.0 * s2 ** 2 / 96.0
    for d in (0.6, 1.5):
        D = IC.dip_cumulants(P, d, c["mu"], c["v0"], c["w"], 100.0)
        faint = (-(d ** 4) / 256.0 + k4g) / (d * d / 16.0 + s2) ** 2
        print("d_true %.1f um: dip area %.4f um, kappa_2 %.5f um^2, kappa_4 %.4e um^4, excess kurtosis %.3f "
              "(faint-limit formula %.3f); alpha = %.3f" % (d, D["area"], D["k2"], D["k4"], D["k4"] / D["k2"] ** 2,
                                                            faint, c["mu"] * d / math.cos(c["phi"])))
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
