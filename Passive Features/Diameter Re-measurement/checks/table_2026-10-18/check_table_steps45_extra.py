"""check_table_steps45_extra.py -- five more claims of the Steps 4-5 explanation, tested on the
simulated tables of check_table_se_coverage.py (2026-10-18; standalone, not pipeline code).

Same design, fixture and spline as check_table_se_coverage.py (imported from the same folder):
ln d ~ U[ln 0.2, ln 4], phi ~ U[0, pi/2), ln mu ~ U[ln 0.3, ln 3], alpha_hat := alpha = mu d / cos phi,
b = 1 + 0.06/d + 0.15 sin^2 phi + 0.05 ln alpha, tau(d) = clip(0.009 (d/1.5)^-3.27, 0.009, 0.30),
N_S = 1500, TPS on range-normalised inputs, smoothing by 5-fold CV.

A  The table's error across independent tables (sd_emp) against the spline's fixed-smoothing SD
   (sd_spl): restricted to the tables that chose the modal smoothing, does the excess of sd_emp
   over sd_spl disappear?  (reads check_table_se_coverage_<mode>.npz)
B  Table term of the propagated error: a noise-free node (d_hat = m(x)) inverted with an estimated
   table at its own (phi, alpha) has (d_tilde - d)/d = -(b_hat(x) - b(x)) / [b(x) (1 + beta(x))] to
   first order, beta = d ln b / d ln d at fixed (phi, alpha) = -0.06 / (d b). Tolerance (fixed
   before running): |actual - predicted| <= 2 (b_hat - b)^2 + 1e-6 (the first-order remainder).
C  Reach of the spline's weights l_n(x) against the kernel's: the share of sum |l_n| from replicates
   farther than two kernel bandwidths (z > 2, z^2 = sum_k ((x_nk - x_k) / eta_k)^2, raw
   coordinates), against the kernel's own share; and the negative mass sum_{l_n < 0} |l_n|.
D  The ln alpha range used by the range normalisation (min to max of the retained rows) is set by
   the near-vertical draws: the 1st-99th percentile span as a share of the min-max span.
E  SD of the median of 3 iid normal values relative to one value (the fill's running median).
B2 (added after B failed, 2026-10-18: 16 of 35 within its tolerance; the misses are 1-2 % of the term
   where b_hat - b is small and 13 % at the thin corner, i.e. B's TRUE slope b (1 + beta) is not the
   slope the root moves along). Same nodes, the table's OWN slope along the inversion line,
   m_hat'(d) = dm_hat/dd at fixed (phi, alpha) by central difference (h = 1e-4 in ln d):
   predicted = -(b_hat(x) - b(x)) / m_hat'(d). Tolerance (fixed before running): |actual - predicted|
   <= 0.05 |predicted| + 1e-6 (the second-order remainder, |(d_tilde - d)/d| <= 0.07 here).

F  Noise of the failure rate (table.py's Nadaraya-Watson form, sum varpi (1 - in_S) / sum varpi over
   every counted replicate, D-057 (i)): with a constant true failure probability p in {0.1, 0.2, 0.3}
   drawn independently per replicate on the batch-0 design, the SD of p_hat over repetitions against
   the binomial prediction sqrt(p (1 - p) / N_eff_loc), and P(p_hat > 0.2) (Block 7's high_failure,
   strict). Tolerance: SD within 10 % of the prediction (2000 repetitions: MC error of an SD ~1.6 %).

Run:  python3 check_table_steps45_extra.py         (A-F, about 3 min on one core)
      python3 check_table_steps45_extra.py B2      (B2 only, about 2 min)
      python3 check_table_steps45_extra.py F       (F only, seconds)
"""
import math

import numpy as np
from scipy.optimize import brentq

import check_table_se_coverage as cov

EXTRA = dict(batches_B=5, mode_A=("hetero", "homo"), seed_E=20261018, n_E=1_000_000)


def check_A():
    print("A  across-table SD with and without the smoothing choice varying")
    for mode in EXTRA["mode_A"]:
        z = np.load("check_table_se_coverage_%s.npz" % mode)
        lam, bhat, sd_spl = z["lam"], z["bhat"], z["sd_exact"]
        vals, counts = np.unique(lam, return_counts=True)
        modal = float(vals[np.argmax(counts)])
        keep = lam == modal
        n_all, n_mod = lam.size, int(keep.sum())
        print("   %s: smoothing chosen %s (counts %s); modal %g in %d of %d tables"
              % (mode, vals.tolist(), counts.tolist(), modal, n_mod, n_all))
        print("   %-17s %9s %9s %9s %9s %9s" % ("query", "sd_emp", "sd_spl", "sd_emp|m", "sd_spl|m", "bias|m"))
        for i, q in enumerate(cov.CFG["queries"]):
            bq = float(cov.truth(math.log(q[0]), math.radians(q[1]), math.log(q[2])))
            print("   %-17s %9.4f %9.4f %9.4f %9.4f %+9.4f" % (
                "(%g, %g, %g)" % q, bhat[:, i].std(ddof=1), sd_spl[:, i].mean(),
                bhat[keep, i].std(ddof=1), sd_spl[keep, i].mean(), bhat[keep, i].mean() - bq))
        print("   relative MC uncertainty of an SD from n tables, 1/sqrt(2(n-1)): %.2f (n = %d), %.2f (n = %d)"
              % (1 / math.sqrt(2 * (n_all - 1)), n_all, 1 / math.sqrt(2 * (n_mod - 1)), n_mod))


def invert(T, d_hat, phi, ln_a):
    lo, hi = cov.CFG["d_range"]
    grid = np.exp(np.linspace(math.log(lo), math.log(hi), cov.CFG["n_grid"]))

    def m_hat(d):
        x = np.column_stack([np.log(np.atleast_1d(d)), np.full(np.size(d), phi), np.full(np.size(d), ln_a)])
        return np.atleast_1d(d) * T["f"](T["sc"](x))
    g = m_hat(grid) - d_hat
    ch = np.flatnonzero(np.sign(g[:-1]) != np.sign(g[1:]))
    if ch.size != 1:
        return float("nan"), int(ch.size)
    k = int(ch[0])
    return float(brentq(lambda d: float(m_hat(d)[0]) - d_hat, grid[k], grid[k + 1], xtol=1e-13)), 1


def check_B_C():
    cov.CFG["tau_mode"] = "hetero"
    Q = cov.query_raw()
    bw = np.array(cov.CFG["bw"])
    worst, n_ok, n_all = 0.0, 0, 0
    print("\nB  table term: noise-free nodes inverted with %d estimated tables (hetero)" % EXTRA["batches_B"])
    print("   %-5s %-17s %10s %10s %10s %10s %6s" % ("table", "query", "b_hat-b", "actual", "predicted", "|diff|", "pass"))
    C_rows = []
    for b in range(EXTRA["batches_B"]):
        T = cov.one_table(b)
        Qs = T["sc"](Q)
        bhat_q = T["f"](Qs)
        L = cov.smoother_weights(T["Xs"], T["lam"], Qs)
        for i, q in enumerate(cov.CFG["queries"]):
            d, phi, ln_a = q[0], math.radians(q[1]), math.log(q[2])
            bt = float(cov.truth(math.log(d), phi, ln_a))
            beta = (-0.06 / d) / bt
            root, nc = invert(T, d * bt, phi, ln_a)
            act = (root - d) / d
            pred = -(float(bhat_q[i]) - bt) / (bt * (1.0 + beta))
            tol = 2.0 * (float(bhat_q[i]) - bt) ** 2 + 1e-6
            ok = nc == 1 and abs(act - pred) <= tol
            n_ok += int(ok)
            n_all += 1
            worst = max(worst, abs(act - pred))
            print("   %-5d %-17s %+10.5f %+10.5f %+10.5f %10.2e %6s" % (b, "(%g, %g, %g)" % q, float(bhat_q[i]) - bt,
                                                                     act, pred, abs(act - pred), "PASS" if ok else "FAIL"))
            # C: reach of the weights
            z = np.sqrt((((T["X"] - Q[i][None, :]) / bw[None, :]) ** 2).sum(1))
            l = L[i]
            w = np.exp(-0.5 * z ** 2)
            C_rows.append((b, q, float(np.abs(l[z > 2]).sum() / np.abs(l).sum()), float(w[z > 2].sum() / w.sum()),
                           float(-l[l < 0].sum()), float(1.0 / np.sum(l ** 2))))
    print("   B: %d of %d pass; largest |actual - predicted| = %.2e" % (n_ok, n_all, worst))
    print("\nC  reach of the spline's weights l_n(x) vs the kernel's (hetero, smoothing as chosen per table)")
    print("   %-5s %-17s %12s %12s %12s %10s" % ("table", "query", "spl |l|, z>2", "kern w, z>2", "neg mass", "1/sum l^2"))
    for b, q, s_spl, s_ker, neg, nspl in C_rows:
        print("   %-5d %-17s %12.2f %12.2f %12.3f %10.1f" % (b, "(%g, %g, %g)" % q, s_spl, s_ker, neg, nspl))
    arr = np.array([r[2:] for r in C_rows])
    print("   C: over %d (table, query) pairs: spline share beyond z > 2 median %.2f (range %.2f-%.2f); kernel share "
          "median %.2f (range %.2f-%.2f); negative mass median %.3f"
          % (len(C_rows), np.median(arr[:, 0]), arr[:, 0].min(), arr[:, 0].max(), np.median(arr[:, 1]),
             arr[:, 1].min(), arr[:, 1].max(), np.median(arr[:, 2])))


def check_D():
    z = np.load("check_table_se_coverage_hetero.npz")
    X = z["X0"]
    la = X[:, 2]
    p = np.percentile(la, [1, 50, 99])
    span = la.max() - la.min()
    print("\nD  ln alpha of the batch-0 replicates (N = %d): min %.2f, 1st pct %.2f, median %.2f, 99th pct %.2f, max %.2f"
          % (la.size, la.min(), p[0], p[1], p[2], la.max()))
    print("   the 1st-99th percentile span is %.0f %% of the min-max span the normalisation divides by; the replicate "
          "with the largest ln alpha has phi = %.2f deg; %d replicates have phi > 85 deg"
          % (100 * (p[2] - p[0]) / span, math.degrees(X[np.argmax(la), 1]), int((X[:, 1] > math.radians(85)).sum())))


def check_E():
    rng = np.random.default_rng(EXTRA["seed_E"])
    x = rng.standard_normal((EXTRA["n_E"], 3))
    r = float(np.median(x, axis=1).std(ddof=1))
    print("\nE  SD of the median of 3 iid N(0, 1) values: %.3f (n = %d draws, seed %d)" % (r, EXTRA["n_E"], EXTRA["seed_E"]))


def check_B2():
    cov.CFG["tau_mode"] = "hetero"
    Q = cov.query_raw()
    h = 1e-4
    n_ok, n_all, rel = 0, 0, []
    print("\nB2 table term with the table's own slope along the inversion line (hetero, %d tables)" % EXTRA["batches_B"])
    print("   %-5s %-17s %10s %10s %10s %10s %10s %6s" % ("table", "query", "b_hat-b", "m_hat'", "actual", "predicted",
                                                         "rel diff", "pass"))
    for b in range(EXTRA["batches_B"]):
        T = cov.one_table(b)
        bhat_q = T["f"](T["sc"](Q))
        for i, q in enumerate(cov.CFG["queries"]):
            d, phi, ln_a = q[0], math.radians(q[1]), math.log(q[2])
            bt = float(cov.truth(math.log(d), phi, ln_a))
            root, nc = invert(T, d * bt, phi, ln_a)
            dp, dm = d * math.exp(h), d * math.exp(-h)
            x = np.array([[math.log(dp), phi, ln_a], [math.log(dm), phi, ln_a]])
            mp, mm = (np.array([dp, dm]) * T["f"](T["sc"](x)))
            slope = float((mp - mm) / (dp - dm))
            act = (root - d) / d
            pred = -(float(bhat_q[i]) - bt) / slope
            ok = nc == 1 and abs(act - pred) <= 0.05 * abs(pred) + 1e-6
            n_ok += int(ok)
            n_all += 1
            rel.append(abs(act - pred) / max(abs(pred), 1e-12))
            print("   %-5d %-17s %+10.5f %10.4f %+10.5f %+10.5f %10.4f %6s" % (
                b, "(%g, %g, %g)" % q, float(bhat_q[i]) - bt, slope, act, pred, rel[-1], "PASS" if ok else "FAIL"))
    rel = np.array(rel)
    print("   B2: %d of %d pass; relative |actual - predicted| median %.4f, max %.4f" % (n_ok, n_all, np.median(rel),
                                                                                       rel.max()))


def check_F():
    z = np.load("check_table_se_coverage_hetero.npz")
    X = z["X0"]
    Q = cov.query_raw()
    W = cov.kernel_w(Q, X)                                    # (7, N): the weights do not depend on the draws
    sw = W.sum(1)
    neff = sw ** 2 / (W ** 2).sum(1)
    rng = np.random.default_rng(EXTRA["seed_E"] + 1)
    reps = 2000
    print("\nF  failure-rate noise on the batch-0 design (N = %d counted replicates, %d repetitions)" % (X.shape[0], reps))
    print("   %-17s %10s %6s %10s %10s %8s %14s" % ("query", "N_eff_loc", "p", "SD p_hat", "binomial", "pass",
                                                   "P(p_hat > 0.2)"))
    for p in (0.1, 0.2, 0.3):
        fail = rng.random((reps, X.shape[0])) < p             # True = outside S
        ph = (fail.astype(float) @ W.T) / sw[None, :]          # (reps, 7)
        for i, q in enumerate(cov.CFG["queries"]):
            sd = float(ph[:, i].std(ddof=1))
            pred = math.sqrt(p * (1 - p) / neff[i])
            print("   %-17s %10.1f %6.2f %10.4f %10.4f %8s %14.3f" % ("(%g, %g, %g)" % q, neff[i], p, sd, pred,
                                                                   "PASS" if abs(sd / pred - 1) <= 0.10 else "FAIL",
                                                                   float((ph[:, i] > 0.2).mean())))


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "B2":
        check_B2()
    elif len(sys.argv) > 1 and sys.argv[1] == "F":
        check_F()
    else:
        check_A()
        check_B_C()
        check_D()
        check_E()
        check_B2()
        check_F()
