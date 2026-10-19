"""check06_equivalent_kernel.py -- the order of the penalty sets the shape of the weights with which the
spline averages the data (2026-10-18; standalone). Document section 3.9, Figures 1-2.

Continuum prediction (a reasoning step of the document, eqs. (24)-(25)): for data of density rho (points per
unit volume) far from the boundary, a smoothing spline with kernel phi, phi_hat(nu) = A |nu|^-2m, and smoothing
lambda acts as the filter  H(nu) = 1 / (1 + lambda |nu|^2m / (A rho)),  so the weight of point n at x is
l_n(x) ~ k(|x - u_n|) / rho, with k the inverse FT of H:  k(r) = (2 pi^2 r)^-1 int_0^inf nu sin(nu r) H(nu) dnu.
  'thin_plate_spline' on R^3: A = 12 pi^2, 2m = 5;   'linear' (degree 1) on R^3: A = 8 pi, 2m = 4 (check02).
Experiment: N points uniform in [0, 1]^3 (rho = N), the query at the centre; lambda set so that both kernels
have the same half-power frequency nu_c (H(nu_c) = 1/2): lambda = A rho / nu_c^2m.
 (a) N l_n(x0) binned by distance against k averaged over the same points: RMS difference / k(0) <= 0.05 over
     the bins inside 0.4 (tolerance fixed before running; the rest is scatter of the random design);
 (b) the first zero r0 and the depth min k / k(0): theory and empirical;
 (c) widths: r0 ~ lambda^(1/2m) exactly in the continuum (dimensional analysis) -- empirical r0 at lambda x
     {0.1, 1, 10} against the factor 10^(1/2m);
 (d) df = trace of the hat matrix at the same three lambdas, the exponent of df - (p + 1) against lambda (the
     quantity eq. (26) predicts, continuum -3/2m; the exponent of df itself is printed too) and the continuum value
     df - (p + 1) ~ k(0) (unit volume).
Writes check06_kernels.npz for Figures 1-2.
Run:  python3 check06_equivalent_kernel.py      (about 2 min)
"""
import math

import numpy as np
from scipy import integrate, optimize

from tps_common import hat_trace, smoother_weights

CFG = dict(seed=20261026, n=3000, nu_c=15.0, bin_w=0.02, r_max=0.44, fit_r=0.40, tol=0.05,
           lam_factors=(0.1, 1.0, 10.0), r_curve=np.linspace(1e-3, 0.6, 300))
KER = {"thin_plate_spline": dict(A=12 * math.pi ** 2, two_m=5), "linear": dict(A=8 * math.pi, two_m=4)}


def H(nu, lam, A, two_m, rho):
    return 1.0 / (1.0 + lam * nu ** two_m / (A * rho))


def k_theory(r, lam, A, two_m, rho):
    r = float(r)
    f = lambda v: v * H(v, lam, A, two_m, rho)
    val, _ = integrate.quad(f, 0, np.inf, weight="sin", wvar=r, limlst=200)
    return val / (2 * math.pi ** 2 * r)


def k0_theory(lam, A, two_m, rho):
    return integrate.quad(lambda v: v * v * H(v, lam, A, two_m, rho), 0, np.inf, limit=400)[0] / (2 * math.pi ** 2)


def first_zero(rs, ks):
    idx = np.where(np.diff(np.sign(ks)) != 0)[0]
    if idx.size == 0:
        return float("nan")
    i = idx[0]
    return float(rs[i] - ks[i] * (rs[i + 1] - rs[i]) / (ks[i + 1] - ks[i]))


def binned(r, v, w, r_max):
    edges = np.arange(0.0, r_max + 1e-12, w)
    mid, mean, se, sel = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (r >= a) & (r < b)
        if m.sum() >= 3:
            mid.append(0.5 * (a + b))
            mean.append(v[m].mean())
            se.append(v[m].std(ddof=1) / math.sqrt(m.sum()))
            sel.append(m)
    return np.array(mid), np.array(mean), np.array(se), sel


def main():
    rng = np.random.default_rng(CFG["seed"])
    X = rng.uniform(0, 1, (CFG["n"], 3))
    x0 = np.full((1, 3), 0.5)
    rn = np.linalg.norm(X - x0, axis=1)
    rho = float(CFG["n"])
    out = dict(r=rn, r_curve=CFG["r_curve"])
    ok = True
    for name, p in KER.items():
        lam0 = p["A"] * rho / CFG["nu_c"] ** p["two_m"]
        print("%s: A = %.4f, 2m = %d, lambda at nu_c = %g: %.5f" % (name, p["A"], p["two_m"], CFG["nu_c"], lam0))
        r0s, dfs = [], []
        for fac in CFG["lam_factors"]:
            lam = lam0 * fac
            L = smoother_weights(X, lam, x0, kernel=name, degree=1)[0]
            emp = CFG["n"] * L
            kt_pts = np.array([k_theory(r, lam, p["A"], p["two_m"], rho) for r in rn])
            mid, mean, se, sel = binned(rn, emp, CFG["bin_w"], CFG["r_max"])
            th_bin = np.array([kt_pts[m].mean() for m in sel])
            k0 = k0_theory(lam, p["A"], p["two_m"], rho)
            inside = mid <= CFG["fit_r"]
            rms = float(np.sqrt(np.mean((mean[inside] - th_bin[inside]) ** 2))) / k0
            curve = np.array([k_theory(r, lam, p["A"], p["two_m"], rho) for r in CFG["r_curve"]])
            r0_th = first_zero(CFG["r_curve"], curve)
            r0_emp = first_zero(mid, mean)
            depth_th = curve.min() / k0
            depth_emp = mean.min() / k0
            df = hat_trace(X, lam, kernel=name, degree=1)
            r0s.append((r0_th, r0_emp))
            dfs.append(df)
            good = rms <= CFG["tol"] if fac == 1.0 else True
            ok &= good
            print("   lambda x%-4g: sum l = %.4f; RMS(binned N l - k)/k(0) = %.3f%s; first zero theory %.3f, "
                  "empirical %.3f; depth min/k(0) theory %+.3f, empirical %+.3f; k(0) = %.1f; df = %.1f"
                  % (fac, float(L.sum()), rms, (" %s" % ("PASS" if good else "FAIL")) if fac == 1.0 else "",
                     r0_th, r0_emp, depth_th, depth_emp, k0, df))
            if fac == 1.0:
                rr = np.linspace(1e-3, 3.0, 3000)
                kk = np.array([k_theory(r, lam, p["A"], p["two_m"], rho) for r in rr])
                neg_th = float(4 * math.pi * np.trapezoid(rr ** 2 * np.minimum(kk, 0.0), rr))
                inner = rn <= 0.5                      # the sphere that fits in the cube
                neg_emp = float(L[inner & (L < 0)].sum())
                neg_th_05 = float(4 * math.pi * np.trapezoid((rr ** 2 * np.minimum(kk, 0.0))[rr <= 0.5], rr[rr <= 0.5]))
                print("      negative mass (sum of negative weights): theory, all r: %+.4f; theory r <= 0.5: %+.4f; "
                      "empirical r <= 0.5: %+.4f" % (neg_th, neg_th_05, neg_emp))
                short = "tps" if name == "thin_plate_spline" else "lin"
                out[short + "_emp"] = emp
                out[short + "_curve"] = curve
                out[short + "_k0"] = k0
                out[short + "_lam"] = lam
                out[short + "_bins"] = np.vstack([mid, mean, se, th_bin])
                out[short + "_H_nu"] = np.logspace(-0.5, 2.5, 300)
                out[short + "_H"] = H(out[short + "_H_nu"], lam, p["A"], p["two_m"], rho)
        f10 = 10 ** (1.0 / p["two_m"])
        ratios_th = [r0s[i + 1][0] / r0s[i][0] for i in range(2)]
        ratios_emp = [r0s[i + 1][1] / r0s[i][1] for i in range(2)]
        expo = math.log(dfs[2] / dfs[0]) / math.log(100.0)
        expo4 = math.log((dfs[2] - 4) / (dfs[0] - 4)) / math.log(100.0)
        print("   widths: 10^(1/2m) = %.3f; theory ratios %s; empirical ratios %s"
              % (f10, ["%.3f" % v for v in ratios_th], ["%.3f" % v for v in ratios_emp]))
        print("   df: %s; exponent of df - 4 against lambda = %.3f (continuum -3/2m = %.3f); of df itself %.3f"
              % (["%.1f" % v for v in dfs], expo4, -3.0 / p["two_m"], expo))
    np.savez("check06_kernels.npz", **out)
    print("overall:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
