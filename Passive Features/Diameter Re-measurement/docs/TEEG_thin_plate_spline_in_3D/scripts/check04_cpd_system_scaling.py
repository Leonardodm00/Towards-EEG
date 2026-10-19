"""check04_cpd_system_scaling.py -- conditional positive definiteness, the saddle-point system as the
minimiser of the penalised problem (scalar and per-point smoothing), the scaling rule, solvability, and
polynomial reproduction (2026-10-18; standalone). Document section 3.8.

(a) CPD: lambda_min of Z^T K Z, Z an orthonormal basis of {c : P^T c = 0}, for P of degree 0 (sum c = 0) and
    degree 1 (affine moments), on N random points in [0, s]^d, s = 0.1, 1, 10. Prediction (textbook): r^2 ln r
    (2-D and 3-D) and r^3 positive with degree 1, not with degree 0 (some configuration negative); -r positive
    with degree 0 already. Also the unconstrained K: indefinite.
(b) The coefficients of SciPy's system minimise
      Q(c, q) = (y - K c - P q)^T W (y - K c - P q) + c^T K c   over P^T c = 0,  W = diag(1 / lambda_n),
    for a scalar smoothing (W = I / lambda) and for a per-point smoothing array: RBFInterpolator reproduced,
    Q rises under 200 random feasible perturbations, projected gradient / |y| <= 1e-8.
(c) Scaling: fitting at s X with smoothing lambda equals fitting at X with lambda / s^2 ('thin_plate_spline')
    or lambda / s ('linear', degree 1), to 1e-9 relative; a scaling of ONE axis is not absorbed (reported).
(d) Solvability: coplanar points in 3-D (degree 1) -> SciPy raises; duplicated points at smoothing 0 -> raises,
    at smoothing > 0 -> solves.
(e) Affine data are reproduced exactly at every smoothing (both kernels, degree 1), max error <= 1e-8.
Run:  python3 check04_cpd_system_scaling.py      (about 30 s)
"""
import math

import numpy as np
from scipy.interpolate import RBFInterpolator

from tps_common import KERNELS, affine_null_basis, fit_coeffs, evaluate, pairwise, poly, saddle_matrix

CFG = dict(seed=20261024, n_cpd=30, scales=(0.1, 1.0, 10.0), n_fit=200, lam=0.3, n_pert=200, s_list=(0.1, 3.7),
           grid=(1e-6, 1e-3, 1e-1, 1.0, 10.0, 1e3))


def cpd(rng):
    print("(a) lambda_min of Z^T K Z  (N = %d random points in [0, s]^d)" % CFG["n_cpd"])
    for d, kernel in ((2, "thin_plate_spline"), (3, "thin_plate_spline"), (3, "linear"), (3, "cubic")):
        for s in CFG["scales"]:
            X = s * rng.uniform(0, 1, (CFG["n_cpd"], d))
            K = KERNELS[kernel](pairwise(X, X))
            ev = np.linalg.eigvalsh(K)
            row = []
            for deg in (0, 1):
                Z = affine_null_basis(X, deg)
                row.append(np.linalg.eigvalsh(Z.T @ K @ Z).min())
            print("    d=%d %-18s s=%-5g K: [%+.2e, %+.2e]   degree 0: %+.3e   degree 1: %+.3e"
                  % (d, kernel, s, ev.min(), ev.max(), row[0], row[1]))


def objective_check(rng, lam, label):
    X = rng.uniform(0, 1, (CFG["n_fit"], 3))
    y = np.sin(3 * X[:, 0]) + X[:, 1] ** 2 + 0.05 * rng.standard_normal(CFG["n_fit"])
    c, q = fit_coeffs(X, y, lam)
    ref = RBFInterpolator(X, y, kernel="thin_plate_spline", smoothing=lam)
    Qx = rng.uniform(0, 1, (50, 3))
    rep = float(np.max(np.abs(evaluate(Qx, X, c, q) - ref(Qx))))
    K = KERNELS["thin_plate_spline"](pairwise(X, X))
    P = poly(X, 1)
    w = 1.0 / np.broadcast_to(np.asarray(lam, dtype=float), (CFG["n_fit"],))
    Qf = lambda c_, q_: float(np.sum(w * (y - K @ c_ - P @ q_) ** 2) + c_ @ K @ c_)
    q0 = Qf(c, q)
    Qp, _ = np.linalg.qr(P)
    proj = lambda v: v - Qp @ (Qp.T @ v)
    rises = 0
    for _ in range(CFG["n_pert"]):
        dc = proj(rng.standard_normal(CFG["n_fit"]))
        dc *= 1e-3 * np.linalg.norm(c) / np.linalg.norm(dc)
        dq = 1e-3 * np.linalg.norm(q) * rng.standard_normal(4) / 2.0
        rises += int(Qf(c + dc, q + dq) > q0)
    res = y - K @ c - P @ q
    g_c = -2 * K @ (w * res) + 2 * K @ c
    g_q = -2 * P.T @ (w * res)
    pg = max(float(np.linalg.norm(proj(g_c))), float(np.linalg.norm(g_q))) / float(np.linalg.norm(y))
    resid_rule = float(np.max(np.abs(res - c / w)))
    good = rep <= 1e-8 and rises == CFG["n_pert"] and pg <= 1e-8
    print("(b) %-22s RBFInterpolator reproduced to %.1e; Q rose under %d/%d perturbations; projected gradient/|y| "
          "%.1e; max |residual - lambda_n c_n| %.1e %s" % (label, rep, rises, CFG["n_pert"], pg, resid_rule,
                                                          "PASS" if good else "FAIL"))
    return good


def scaling(rng):
    X = rng.uniform(0, 1, (CFG["n_fit"], 3))
    y = np.sin(3 * X[:, 0]) + X[:, 1] ** 2 + 0.05 * rng.standard_normal(CFG["n_fit"])
    Qx = rng.uniform(0, 1, (100, 3))
    ok = True
    for kernel, power in (("thin_plate_spline", 2), ("linear", 1)):
        base = RBFInterpolator(X, y, kernel=kernel, degree=1, smoothing=CFG["lam"])
        for s in CFG["s_list"]:
            f_s = RBFInterpolator(s * X, y, kernel=kernel, degree=1, smoothing=CFG["lam"] * s ** power)
            diff = float(np.max(np.abs(f_s(s * Qx) - base(Qx)))) / float(np.ptp(y))
            good = diff <= 1e-9
            ok &= good
            print("(c) %-18s isotropic s=%-4g with lambda*s^%d: max |difference| / range(y) = %.1e %s"
                  % (kernel, s, power, diff, "PASS" if good else "FAIL"))
        S = np.array([1.0, 1.0, 3.7])
        best = min(float(np.max(np.abs(RBFInterpolator(X * S, y, kernel=kernel, degree=1,
                                                       smoothing=CFG["lam"] * f)(Qx * S) - base(Qx))))
                   for f in np.logspace(-3, 3, 61)) / float(np.ptp(y))
        print("    %-18s one axis x3.7: smallest max |difference| / range(y) over 61 smoothings = %.2e (not absorbed)"
              % (kernel, best))
    return ok


def solvability(rng):
    X = rng.uniform(0, 1, (30, 3))
    X[:, 2] = 0.3
    y = rng.standard_normal(30)
    for lam in (0.0, 1.0):
        try:
            RBFInterpolator(X, y, kernel="thin_plate_spline", smoothing=lam)
            print("(d) coplanar points, smoothing %g: solved (unexpected)" % lam)
        except Exception as e:                   # numpy LinAlgError
            print("(d) coplanar points, smoothing %g: %s: %s" % (lam, type(e).__name__, str(e)[:110]))
    X = rng.uniform(0, 1, (30, 3))
    X[1] = X[0]
    for lam in (0.0, 1e-3):
        try:
            f = RBFInterpolator(X, y, kernel="thin_plate_spline", smoothing=lam)
            print("    duplicated point, smoothing %g: solved; f at the duplicate %.4f (data %.4f, %.4f)"
                  % (lam, float(f(X[:1])[0]), y[0], y[1]))
        except Exception as e:
            print("    duplicated point, smoothing %g: %s: %s" % (lam, type(e).__name__, str(e)[:110]))


def affine(rng):
    X = rng.uniform(0, 1, (CFG["n_fit"], 3))
    a = np.array([0.3, -1.2, 0.7, 2.0])
    ya = a[0] + X @ a[1:]
    Qx = rng.uniform(0, 1, (100, 3))
    err = max(float(np.max(np.abs(RBFInterpolator(X, ya, kernel=k, degree=1, smoothing=l)(Qx) - (a[0] + Qx @ a[1:]))))
              for k in ("thin_plate_spline", "linear") for l in CFG["grid"])
    print("(e) affine data reproduced, both kernels, smoothing %s: max error %.1e %s"
          % (list(CFG["grid"]), err, "PASS" if err <= 1e-8 else "FAIL"))
    return err <= 1e-8


def main():
    rng = np.random.default_rng(CFG["seed"])
    cpd(rng)
    ok = objective_check(rng, CFG["lam"], "scalar smoothing 0.3")
    lam_vec = CFG["lam"] * np.exp(rng.uniform(-2, 2, CFG["n_fit"]))
    ok &= objective_check(rng, lam_vec, "per-point smoothing")
    ok &= scaling(rng)
    solvability(rng)
    ok &= affine(rng)
    print("overall:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
