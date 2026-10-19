"""check_tps_deep.py -- claims of the deepened explanation of Step 4 (the thin-plate spline), tested on
simulated data with a known truth (2026-10-18; standalone, not pipeline code). Uses the design, fixture,
tau(d), range normalisation and saddle-point weights of check_table_se_coverage.py (imported; same folder).

T1 The name: r^2 ln r is the biharmonic Green's function in 2-D (Delta^2 of it vanishes for r > 0) and is
   not in 3-D, where Delta^2 (r^2 ln r) = 6 / r^2 and the biharmonic Green's function is r (sympy, exact).
T2 The two limits of the smoothing on the batch-0 design (hetero): lambda = 1e-10 interpolates the ratios
   (max |f(x_n) - y_n| <= 1e-6); lambda = 1e10 gives the least-squares affine fit in s(x)
   (numpy.linalg.lstsq; max difference at the query points <= 1e-6).
T3 The variational form: the coefficients (c, g) that reproduce RBFInterpolator minimise
   Q = ||y - K c - P g||^2 + lambda c^T K c over P^T c = 0: Q rises under 200 random feasible
   perturbations (relative size 1e-3), and the gradient projected on the feasible set is <= 1e-8 of |y|.
T4 Effective degrees of freedom df(lambda) = trace(H), H_nm = l_m(x_n) the hat matrix, over the grid.
T5 Affine reproduction: y affine in s(x) is reproduced at every lambda of the grid (max error <= 1e-8).
T6 Cost of one fit: wall time and the dense matrix at N = 1500, 3000, 6000 (smoothing 10).
T7 CV folds under D-059/D-060: G = 300 draws x k = 5 realizations; realization r of draw g sits at
   ln alpha_hat = ln alpha_g + 0.1 eps (fit scatter) and has y = b(x_g) + delta_g + tau(d_g) eps,
   delta_g ~ N(0, (0.5 tau(d_g))^2) shared by the draw; lambda chosen by plain 5-fold CV (a draw's
   realizations spread over folds) and by folds by draw; RMS error of each table against b at the
   realizations' x, 6 seeds. Tolerance-free: the comparison is reported.

Run:  python3 check_tps_deep.py        (about 6 min on one core)
"""
import math
import time

import numpy as np
from scipy.interpolate import RBFInterpolator

import check_table_se_coverage as cov

CFG = dict(t7_seeds=6, t7_groups=300, t7_reps=5, t7_sd_lnalpha=0.1, t7_offset=0.5, folds=5,
           timing_n=(1500, 3000, 6000), seed=20261020)


def t1():
    import sympy as sp
    r = sp.symbols("r", positive=True)

    def lap(f, p):
        return sp.simplify(sp.diff(f, r, 2) + (p - 1) / r * sp.diff(f, r))
    f = r ** 2 * sp.log(r)
    print("T1 Delta^2 (r^2 ln r): 2-D ->", sp.simplify(lap(lap(f, 2), 2)), "| 3-D ->", sp.simplify(lap(lap(f, 3), 3)),
          "| Delta^2 r in 3-D (r > 0) ->", sp.simplify(lap(lap(r, 3), 3)))


def saddle(Xs, lam):
    n = Xs.shape[0]
    K = cov.tps(np.linalg.norm(Xs[:, None, :] - Xs[None, :, :], axis=2))
    P = np.column_stack([np.ones(n), Xs])
    M = np.block([[K + lam * np.eye(n), P], [P.T, np.zeros((4, 4))]])
    return K, P, M


def t2_t5(T):
    Xs, y = T["Xs"], T["y"]
    n = Xs.shape[0]
    f0 = RBFInterpolator(Xs, y, kernel="thin_plate_spline", smoothing=1e-10)
    e0 = float(np.max(np.abs(f0(Xs) - y)))
    finf = RBFInterpolator(Xs, y, kernel="thin_plate_spline", smoothing=1e10)
    A = np.column_stack([np.ones(n), Xs])
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    Qs = T["sc"](cov.query_raw())
    einf = float(np.max(np.abs(finf(Qs) - np.column_stack([np.ones(len(Qs)), Qs]) @ coef)))
    print("T2 lambda 1e-10: max |f(x_n) - y_n| = %.2e (%s) | lambda 1e10 vs least-squares affine: max diff %.2e (%s)"
          % (e0, "PASS" if e0 <= 1e-6 else "FAIL", einf, "PASS" if einf <= 1e-6 else "FAIL"))
    # T3 variational form at the chosen smoothing
    lam = T["lam"]
    K, P, M = saddle(Xs, lam)
    sol = np.linalg.solve(M, np.concatenate([y, np.zeros(4)]))
    c, g = sol[:n], sol[n:]
    ref = RBFInterpolator(Xs, y, kernel="thin_plate_spline", smoothing=lam)(Qs)
    kq = cov.tps(np.linalg.norm(Qs[:, None, :] - Xs[None, :, :], axis=2))
    mine = kq @ c + np.column_stack([np.ones(len(Qs)), Qs]) @ g
    Qf = lambda c_, g_: float(np.sum((y - K @ c_ - P @ g_) ** 2) + lam * c_ @ K @ c_)
    q0 = Qf(c, g)
    rng = np.random.default_rng(CFG["seed"])
    # projector onto {c : P^T c = 0}
    Qp, _ = np.linalg.qr(P)
    proj = lambda v: v - Qp @ (Qp.T @ v)
    rises = 0
    for _ in range(200):
        dc = proj(rng.standard_normal(n))
        dc *= 1e-3 * np.linalg.norm(c) / np.linalg.norm(dc)
        dg = 1e-3 * np.linalg.norm(g) * rng.standard_normal(4) / 2.0
        rises += int(Qf(c + dc, g + dg) > q0)
    grad_c = -2 * K @ (y - K @ c - P @ g) + 2 * lam * K @ c
    grad_g = -2 * P.T @ (y - K @ c - P @ g)
    pg = max(float(np.linalg.norm(proj(grad_c))), float(np.linalg.norm(grad_g))) / float(np.linalg.norm(y))
    print("T3 lambda %g: coefficients reproduce RBFInterpolator to %.1e; Q rose under %d of 200 feasible perturbations; "
          "projected gradient / |y| = %.1e (%s)" % (lam, float(np.max(np.abs(mine - ref))), rises, pg,
                                                   "PASS" if rises == 200 and pg <= 1e-8 else "FAIL"))
    # T4 effective degrees of freedom
    rows = []
    for lam_ in cov.CFG["grid"] + (1e3, 1e4, 1e6):
        L = cov.smoother_weights(Xs, lam_, Xs)
        rows.append((lam_, float(np.trace(L))))
    print("T4 df(lambda) = trace(H), N = %d: " % n + ", ".join("%g: %.1f" % r_ for r_ in rows))
    # T5 affine reproduction
    a = np.array([0.3, -1.2, 0.7, 2.0])
    ya = a[0] + Xs @ a[1:]
    errs = []
    for lam_ in cov.CFG["grid"]:
        fa = RBFInterpolator(Xs, ya, kernel="thin_plate_spline", smoothing=lam_)
        errs.append(float(np.max(np.abs(fa(Qs) - (a[0] + Qs @ a[1:])))))
    print("T5 affine y reproduced: max error over the grid %.1e (%s)" % (max(errs), "PASS" if max(errs) <= 1e-8 else "FAIL"))


def t6():
    rng = np.random.default_rng(CFG["seed"] + 1)
    out = []
    for n in CFG["timing_n"]:
        X = cov.design(rng, n)
        y = cov.truth(X[:, 0], X[:, 1], X[:, 2]) + 0.03 * rng.standard_normal(n)
        Xs = cov.Scaler(X)(X)
        t0 = time.time()
        RBFInterpolator(Xs, y, kernel="thin_plate_spline", smoothing=10.0)
        dt = time.time() - t0
        out.append((n, dt, (n + 4) ** 2 * 8 / 1e9))
    print("T6 one fit (smoothing 10): " + "; ".join("N %d: %.2f s, dense matrix %.2f GB" % o for o in out))
    n1, t1_, _ = out[-2]
    n2, t2_, _ = out[-1]
    expo = math.log(t2_ / t1_) / math.log(n2 / n1)
    print("   observed time exponent between N %d and %d: %.2f" % (n1, n2, expo))


def cv_choice(Xs, y, groups):
    n = Xs.shape[0]
    rng = np.random.default_rng(cov.CFG["fold_seed"])
    if groups is None:
        fold = np.empty(n, dtype=int)
        fold[rng.permutation(n)] = np.arange(n) % CFG["folds"]
    else:
        ug = np.unique(groups)
        gf = np.empty(ug.size, dtype=int)
        gf[rng.permutation(ug.size)] = np.arange(ug.size) % CFG["folds"]
        fold = gf[np.searchsorted(ug, groups)]
    scores = []
    for lam in cov.CFG["grid"]:
        err = 0.0
        try:
            for f in range(CFG["folds"]):
                tr = fold != f
                err += float(np.sum((RBFInterpolator(Xs[tr], y[tr], kernel="thin_plate_spline", smoothing=lam)(Xs[~tr])
                                     - y[~tr]) ** 2))
        except np.linalg.LinAlgError:
            err = math.inf
        scores.append(err / n)
    return float(cov.CFG["grid"][int(np.argmin(scores))])


def t7():
    cov.CFG["tau_mode"] = "hetero"
    G, k = CFG["t7_groups"], CFG["t7_reps"]
    res = {"plain": [], "by draw": []}
    for s in range(CFG["t7_seeds"]):
        rng = np.random.default_rng([CFG["seed"], 7, s])
        Xg = cov.design(rng, G)
        tau_g = cov.tau_of_d(np.exp(Xg[:, 0]))
        delta = CFG["t7_offset"] * tau_g * rng.standard_normal(G)
        X = np.repeat(Xg, k, axis=0)
        X[:, 2] = X[:, 2] + CFG["t7_sd_lnalpha"] * rng.standard_normal(G * k)
        groups = np.repeat(np.arange(G), k)
        b_true = cov.truth(X[:, 0], X[:, 1], X[:, 2])
        y = b_true + np.repeat(delta, k) + np.repeat(tau_g, k) * rng.standard_normal(G * k)
        Xs = cov.Scaler(X)(X)
        for name, grp in (("plain", None), ("by draw", groups)):
            lam = cv_choice(Xs, y, grp)
            f = RBFInterpolator(Xs, y, kernel="thin_plate_spline", smoothing=lam)
            rms = float(np.sqrt(np.mean((f(Xs) - b_true) ** 2)))
            res[name].append((lam, rms))
        print("   T7 seed %d: plain CV lambda %g (RMS %.4f) | folds by draw lambda %g (RMS %.4f)"
              % (s, res["plain"][-1][0], res["plain"][-1][1], res["by draw"][-1][0], res["by draw"][-1][1]), flush=True)
    for name in res:
        lams = [r_[0] for r_ in res[name]]
        print("T7 %-8s: lambda chosen %s; mean RMS error of the table %.4f"
              % (name, lams, float(np.mean([r_[1] for r_ in res[name]]))))


def main():
    t1()
    cov.CFG["tau_mode"] = "hetero"
    T = cov.one_table(0)
    t2_t5(T)
    t6()
    t7()


if __name__ == "__main__":
    main()
