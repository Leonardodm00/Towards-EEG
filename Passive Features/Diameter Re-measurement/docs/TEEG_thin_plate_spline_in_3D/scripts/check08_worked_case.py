"""check08_worked_case.py -- the worked case of section 3.10: does eq. (24)-(26) predict where the weights of
the bias table's spline change sign, at the point x = (1 um, 30 deg, alpha_hat 1.2) of Figure 5?
(2026-10-18; standalone). Uses the batch-0 table of check_table_se_coverage.py (imported; same folder):
N_S = 1500, range-normalised inputs u = s(x), smoothing by its CV (10 in that run).

Prediction (fixed before running): the local density rho(u_x) of the design around u_x, counted in balls of
radius h; nu_c = (12 pi^2 rho / lambda_s)^(1/5); the first zero of the order-5/2 equivalent kernel at
rho_0 = s_0 / nu_c, s_0 the first zero of the scale-free kernel (int_0^inf t sin(t s) / (1 + t^5) dt = 0).
Empirical: the first sign change of the radial weight density of l_n(x) (bins of 0.025). The local-density
approximation assumes rho constant over the kernel's width, which this design does not satisfy (the kernel spans
about half the cube), so agreement within 25 % is the tolerance, fixed before running.
Also: the width ratio between lambda_s = 1 and 10 (prediction 10^(1/5) = 1.585) and, from the df column of
check_tps_deep.py T4 (copied below), the exponent of (df - 4) against lambda_s (continuum -3/5).
Run:  python3 check08_worked_case.py      (about 30 s)
"""
import math

import numpy as np
from scipy import integrate, optimize

import check_table_se_coverage as cov

CFG = dict(query=(1.0, 30.0, 1.2), radii=(0.10, 0.15, 0.20), lams=(1.0, 10.0), bin_w=0.025, r_max=0.9, tol=0.25,
           t4_df={0.1: 131.3, 1.0: 40.2, 10.0: 13.4, 100.0: 5.8})


def s0_order52():
    f = lambda s: integrate.quad(lambda t: t / (1 + t ** 5), 0, np.inf, weight="sin", wvar=s, limlst=200)[0]
    return optimize.brentq(f, 2.0, 6.0)


def first_sign_change(r, l):
    edges = np.arange(0.0, CFG["r_max"] + 1e-12, CFG["bin_w"])
    dens = np.array([l[(r >= a) & (r < b)].sum() for a, b in zip(edges[:-1], edges[1:])])
    mids = 0.5 * (edges[:-1] + edges[1:])
    idx = np.where((dens[:-1] > 0) & (dens[1:] <= 0))[0]
    if idx.size == 0:
        return float("nan")
    i = idx[0]
    return float(mids[i] + (mids[i + 1] - mids[i]) * dens[i] / (dens[i] - dens[i + 1]))


def main():
    cov.CFG["tau_mode"] = "hetero"
    T = cov.one_table(0)
    q = CFG["query"]
    qraw = np.array([[math.log(q[0]), math.radians(q[1]), math.log(q[2])]])
    qs = T["sc"](qraw)
    rn = np.linalg.norm(T["Xs"] - qs, axis=1)
    n = T["Xs"].shape[0]
    s0 = s0_order52()
    print("batch-0 table: N_S = %d, CV smoothing %g; u_x = %s; s_0 (order-5/2 first zero, scale-free) = %.4f"
          % (n, T["lam"], np.round(qs[0], 3), s0))
    rhos = []
    for h in CFG["radii"]:
        k = int(np.sum(rn <= h))
        rho = k / (4.0 / 3.0 * math.pi * h ** 3)
        rhos.append(rho)
        print("  local density, ball radius %.2f: %d points -> rho = %.0f per unit volume (cube average %d)" % (h, k, rho, n))
    rho = float(np.median(rhos))
    zeros = {}
    ok = True
    for lam in CFG["lams"]:
        L = cov.smoother_weights(T["Xs"], lam, qs)[0]
        z_emp = first_sign_change(rn, L)
        nu_c = (12 * math.pi ** 2 * rho / lam) ** 0.2
        z_pred = s0 / nu_c
        zeros[lam] = (z_pred, z_emp)
        rel = abs(z_emp / z_pred - 1)
        good = rel <= CFG["tol"]
        ok &= good
        print("  lambda_s = %-4g: nu_c = %.2f; first zero predicted %.3f, empirical %.3f (rel diff %.2f) %s; "
              "negative sum %.3f" % (lam, nu_c, z_pred, z_emp, rel, "PASS" if good else "FAIL", float(L[L < 0].sum())))
    print("  (rho above is the median of the three ball estimates)")
    for h, rho_k in zip(CFG["radii"], rhos):          # sensitivity to the density estimate; no tolerance
        parts = []
        for lam in CFG["lams"]:
            z = s0 / (12 * math.pi ** 2 * rho_k / lam) ** 0.2
            parts.append("lambda %g: predicted %.3f (empirical %.3f, rel diff %+.2f)"
                         % (lam, z, zeros[lam][1], z / zeros[lam][1] - 1))
        print("  sensitivity, rho from the ball of radius %.2f (%.0f): %s" % (h, rho_k, "; ".join(parts)))
    ratio_emp = zeros[10.0][1] / zeros[1.0][1]
    print("  width ratio lambda 10 / lambda 1: predicted %.3f, empirical %.3f" % (10 ** 0.2, ratio_emp))
    d = CFG["t4_df"]
    for a, b in ((0.1, 1.0), (1.0, 10.0), (10.0, 100.0)):
        e = math.log((d[b] - 4) / (d[a] - 4)) / math.log(b / a)
        print("  T4 df: exponent of (df - 4) between lambda %g and %g: %.3f (continuum -0.6)" % (a, b, e))
    print("overall:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
