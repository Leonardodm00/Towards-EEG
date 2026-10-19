"""check_table_se_coverage.py -- two claims about Steps 4-5 of the bias table, tested on
simulated data with a known truth (2026-10-18; standalone, not pipeline code).

Design: the production defaults (D-024 (iii)): ln d ~ U[ln 0.2, ln 4], phi ~ U[0, pi/2),
ln mu ~ U[ln 0.3, ln 3]; alpha = mu d / cos(phi). The table's third axis is the FITTED
alpha_hat (D-031, D-053 (iii)); here alpha_hat := alpha (the true value), a stand-in that
holds where the fit is accurate and understates the spread on thin nodes.
Truth (the handoff s.8.3 fixture): b = 1 + 0.06/d + 0.15 sin^2(phi) + 0.05 ln(alpha).
Scatter of the ratio (heteroscedastic, modelled on the handoff s.4.1 Monte Carlo, simulation
only): tau(d) = clip(0.009 (d / 1.5)^(-3.27), 0.009, 0.30), so 0.009 at 1.5 um and 0.18 at 0.6 um.
Replicates: y_n = b(x_n) + tau(d_n) eps_n, eps ~ N(0, 1), N_S = 1500, all retained.
Spline (D-024 (v), D-054 (i)): RBFInterpolator thin_plate_spline on the range-normalised inputs
(ln d, phi, ln alpha), smoothing by 5-fold CV over the configured grid (table.choose_smoothing's rule).
Local statistics (table.py, D-054 (ii)): Gaussian weights with bandwidths (0.15 in ln d,
5 deg in phi, 0.3 in ln alpha) in raw coordinates; tau_hat, N_eff = (sum w)^2 / sum w^2,
SE = tau_hat / sqrt(N_eff).

C1  At fixed smoothing the spline is a linear smoother, b_hat(x) = sum_n l_n(x) y_n, with
    sum_n l_n(x) = 1: the weights l(x) from the saddle-point system reproduce
    RBFInterpolator to 1e-8 and sum to 1 to 1e-8.
C2  Per query point, the table's reported SE (the kernel's) against the spline's own
    sampling SD at the chosen smoothing, sqrt(sum_n l_n(x)^2 tau(d_n)^2), and the two
    effective counts, N_eff of the kernel and 1 / sum l_n^2 of the spline.
C3  Monte Carlo over BATCHES independent tables: the empirical SD of b_hat(x) across tables
    (the table's real Monte Carlo error, CV included) and its mean error (smoothing bias),
    against the mean reported SE; tau_hat against tau(d).
C4  The inversion slice: at fixed (phi, alpha_hat) the inversion searches d over [0.2, 4] um
    (Block 7's bracket). The phantoms cover only the band alpha cos(phi) / mu_max <= d <=
    alpha cos(phi) / mu_min of that slice. Along the slice: |b_hat - b| inside and outside
    the band; the kernel's N_eff (scale-free: it stays large off the data) and the kernel
    mass sum_n w_n (it falls off the data), against the mass at the replicates themselves
    (leave-one-out: the interior's 5th percentile and median).
C5  A node darker than every phantom of its size (true d 0.5 um, mu 4 /um, phi 10 deg,
    noise-free d_hat = m(d)) inverted at its own alpha_hat with the batch-0 table: the root,
    N_eff, mass and SE there.
Contrast: the same tables with a constant tau = 0.03 (TAU_MODE = "homo"), for the smoothing CV picks.

Run:  python3 check_table_se_coverage.py [homo]    (about 2 min per mode on one core)
"""
import math
import time

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.optimize import brentq

CFG = dict(seed=20261018, batches=20, n=1500, folds=5, fold_seed=20261006,
           d_range=(0.2, 4.0), mu_range=(0.3, 3.0), phi_hi=0.5 * math.pi,
           grid=(1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0),
           bw=(0.15, math.radians(5.0), 0.3), neff_support=5.0,
           queries=((0.25, 5.0, 0.10), (0.3, 20.0, 0.25), (0.6, 10.0, 0.6), (1.0, 30.0, 1.2),
                    (2.0, 10.0, 2.0), (3.0, 45.0, 4.0), (1.0, 70.0, 3.0)),
           slice_phi_deg=10.0, slice_alphas=(0.1, 0.3, 1.0, 3.0, 10.0), n_grid=256,
           dark_node=(0.5, 4.0, 10.0), tau_mode="hetero")


def truth(ln_d, phi, ln_a):
    return 1.0 + 0.06 / np.exp(ln_d) + 0.15 * np.sin(phi) ** 2 + 0.05 * ln_a


def tau_of_d(d):
    if CFG["tau_mode"] == "homo":
        return np.full(np.shape(d), 0.03)
    return np.clip(0.009 * (np.asarray(d) / 1.5) ** (-3.27), 0.009, 0.30)


def design(rng, n):
    lo, hi = CFG["d_range"]
    ln_d = rng.uniform(math.log(lo), math.log(hi), n)
    phi = rng.uniform(0.0, CFG["phi_hi"], n)
    mlo, mhi = CFG["mu_range"]
    ln_mu = rng.uniform(math.log(mlo), math.log(mhi), n)
    ln_a = ln_mu + ln_d - np.log(np.cos(phi))
    return np.column_stack([ln_d, phi, ln_a])


class Scaler:
    """Range normalisation (D-054 (i)): ln d and phi by the design ranges, ln alpha by the retained rows'."""

    def __init__(self, X):
        lo, hi = CFG["d_range"]
        self.lo = np.array([math.log(lo), 0.0, X[:, 2].min()])
        self.span = np.array([math.log(hi) - math.log(lo), CFG["phi_hi"], X[:, 2].max() - X[:, 2].min()])

    def __call__(self, X):
        return (np.asarray(X) - self.lo) / self.span


def spline(Xs, y, lam):
    return RBFInterpolator(Xs, y, kernel="thin_plate_spline", smoothing=float(lam))


def choose_smoothing(Xs, y):
    n = Xs.shape[0]
    fold = np.empty(n, dtype=int)
    fold[np.random.default_rng(CFG["fold_seed"]).permutation(n)] = np.arange(n) % CFG["folds"]
    scores = []
    for lam in CFG["grid"]:
        err = 0.0
        for f in range(CFG["folds"]):
            tr = fold != f
            err += float(np.sum((spline(Xs[tr], y[tr], lam)(Xs[~tr]) - y[~tr]) ** 2))
        scores.append(err / n)
    return float(CFG["grid"][int(np.argmin(scores))])


def tps(r):
    with np.errstate(divide="ignore", invalid="ignore"):
        out = r * r * np.log(r)
    return np.where(r > 0, out, 0.0)


def smoother_weights(Xs, lam, Qs):
    """l(x) for the query rows Qs: f(Qs) = L @ y, L (m, n), from the saddle-point system
    [[K + lam I, P], [P^T, 0]] [a; c] = [y; 0], P = [1, x] (degree 1, SciPy's TPS default)."""
    n = Xs.shape[0]
    K = tps(np.linalg.norm(Xs[:, None, :] - Xs[None, :, :], axis=2)) + lam * np.eye(n)
    P = np.column_stack([np.ones(n), Xs])
    M = np.block([[K, P], [P.T, np.zeros((4, 4))]])
    kq = tps(np.linalg.norm(Qs[:, None, :] - Xs[None, :, :], axis=2))
    pq = np.column_stack([np.ones(Qs.shape[0]), Qs])
    Z = np.linalg.solve(M, np.hstack([kq, pq]).T)          # M symmetric: rows of [kq, pq] M^-1
    return Z[:n, :].T


def kernel_w(Qraw, Xraw):
    z = (Qraw[:, None, :] - Xraw[None, :, :]) / np.array(CFG["bw"])
    return np.exp(-0.5 * np.sum(z * z, axis=2))


def local_stats(Qraw, Xraw, resid, mass=False):
    w = kernel_w(Qraw, Xraw)
    sw = w.sum(1)
    neff = sw ** 2 / np.maximum((w ** 2).sum(1), 1e-300)
    var = (w * resid[None, :] ** 2).sum(1) / np.maximum(sw, 1e-300)
    with np.errstate(divide="ignore", invalid="ignore"):
        var = np.where(neff > 1, var * neff / (neff - 1), np.nan)
    tau_hat = np.sqrt(var)
    if mass:
        return neff, tau_hat, tau_hat / np.sqrt(neff), sw
    return neff, tau_hat, tau_hat / np.sqrt(neff)


def query_raw():
    q = np.array(CFG["queries"], dtype=float)
    return np.column_stack([np.log(q[:, 0]), np.radians(q[:, 1]), np.log(q[:, 2])])


def one_table(b):
    rng = np.random.default_rng([CFG["seed"], b])
    X = design(rng, CFG["n"])
    tau_n = tau_of_d(np.exp(X[:, 0]))
    y = truth(X[:, 0], X[:, 1], X[:, 2]) + tau_n * rng.standard_normal(CFG["n"])
    sc = Scaler(X)
    Xs = sc(X)
    lam = choose_smoothing(Xs, y)
    f = spline(Xs, y, lam)
    return dict(X=X, y=y, tau_n=tau_n, sc=sc, Xs=Xs, lam=lam, f=f, resid=y - f(Xs))


def main():
    t0 = time.time()
    Q = query_raw()
    bq = truth(Q[:, 0], Q[:, 1], Q[:, 2])
    tq = tau_of_d(np.exp(Q[:, 0]))
    rec = dict(bhat=[], se=[], neff=[], tauh=[], neff_spl=[], sd_exact=[], lam=[])
    tab0 = None
    for b in range(CFG["batches"]):
        T = one_table(b)
        Qs = T["sc"](Q)
        L = smoother_weights(T["Xs"], T["lam"], Qs)
        if b == 0:
            tab0 = T
            dev = float(np.max(np.abs(L @ T["y"] - T["f"](Qs))))
            ssum = float(np.max(np.abs(L.sum(1) - 1.0)))
            print("C1 linear smoother: |L y - RBFInterpolator| = %.1e, |sum l - 1| = %.1e  -> %s"
                  % (dev, ssum, "PASS" if dev <= 1e-8 and ssum <= 1e-8 else "FAIL"))
        neff, tauh, se = local_stats(Q, T["X"], T["resid"])
        rec["bhat"].append(T["f"](Qs))
        rec["se"].append(se)
        rec["neff"].append(neff)
        rec["tauh"].append(tauh)
        rec["neff_spl"].append(1.0 / np.sum(L ** 2, axis=1))
        rec["sd_exact"].append(np.sqrt(np.sum(L ** 2 * T["tau_n"][None, :] ** 2, axis=1)))
        rec["lam"].append(T["lam"])
        print("  batch %2d: smoothing %g  (%.0f s)" % (b, T["lam"], time.time() - t0), flush=True)
    R = {k: np.array(v) for k, v in rec.items()}
    sd_emp = R["bhat"].std(axis=0, ddof=1)
    bias = R["bhat"].mean(axis=0) - bq
    print("\nC2/C3 per query point (d um, phi deg, alpha), %d tables of N_S = %d; smoothing chosen: %s"
          % (CFG["batches"], CFG["n"], sorted(set(R["lam"].tolist()))))
    hdr = ("query", "b", "tau(d)", "tau_hat", "N_eff kern", "N_eff spl", "SE rep", "sd spl", "sd emp", "bias",
           "sd emp/SE")
    print(("%-17s" + " %9s" * (len(hdr) - 1)) % hdr)
    for i, q in enumerate(CFG["queries"]):
        print(("%-17s" + " %9.4f %9.4f %9.4f %9.1f %9.1f %9.4f %9.4f %9.4f %+9.4f %9.2f") % (
            "(%g, %g, %g)" % q, bq[i], tq[i], np.nanmean(R["tauh"][:, i]), R["neff"][:, i].mean(),
            R["neff_spl"][:, i].mean(), np.nanmean(R["se"][:, i]), R["sd_exact"][:, i].mean(), sd_emp[i], bias[i],
            sd_emp[i] / np.nanmean(R["se"][:, i])))
    # C4: the inversion slice, batch-0 table
    T = tab0
    W = kernel_w(T["X"], T["X"])
    loo = W.sum(1) - 1.0                                      # kernel mass at each replicate, itself excluded
    m05, m50 = float(np.percentile(loo, 5)), float(np.median(loo))
    print("\nC4 kernel mass at the replicates (leave-one-out): 5th percentile %.2f, median %.2f" % (m05, m50))
    phi = math.radians(CFG["slice_phi_deg"])
    lo, hi = CFG["d_range"]
    dg = np.exp(np.linspace(math.log(lo), math.log(hi), CFG["n_grid"]))
    print("   slices at phi = %g deg (the inversion searches d over [%g, %g] um at fixed alpha_hat):"
          % (CFG["slice_phi_deg"], lo, hi))
    for a in CFG["slice_alphas"]:
        S = np.column_stack([np.log(dg), np.full(dg.size, phi), np.full(dg.size, math.log(a))])
        neff, _, _, mass = local_stats(S, T["X"], T["resid"], mass=True)
        err = np.abs(T["f"](T["sc"](S)) - truth(S[:, 0], S[:, 1], S[:, 2]))
        b_lo, b_hi = a * math.cos(phi) / CFG["mu_range"][1], a * math.cos(phi) / CFG["mu_range"][0]
        inb = (dg >= b_lo) & (dg <= b_hi)
        out = ~inb
        def fmt(x):
            return "%.4f" % x if np.isfinite(x) else "-"
        print("   alpha_hat %5.2f: band d in [%.2f, %.2f] um = %3.0f %% of the bracket | max |b_hat - b| in %s, out %s"
              " | N_eff min in %.1f, min out %s | mass min in %.2f, max out %s"
              % (a, max(b_lo, lo), min(b_hi, hi), 100.0 * inb.mean(),
                 fmt(float(err[inb].max())) if inb.any() else "-", fmt(float(err[out].max())) if out.any() else "-",
                 float(neff[inb].min()) if inb.any() else float("nan"),
                 ("%.1f" % float(neff[out].min())) if out.any() else "-",
                 float(mass[inb].min()) if inb.any() else float("nan"),
                 ("%.3f" % float(mass[out].max())) if out.any() else "-"))
    # C5: the dark thin node
    d_t, mu_t, phi_deg = CFG["dark_node"]
    ph = math.radians(phi_deg)
    a_t = mu_t * d_t / math.cos(ph)
    d_hat = d_t * float(truth(math.log(d_t), ph, math.log(a_t)))

    def mhat(d):
        x = np.array([[math.log(d), ph, math.log(a_t)]])
        return d * float(T["f"](T["sc"](x))[0])
    g = np.array([mhat(d) for d in dg]) - d_hat
    ch = np.flatnonzero(np.sign(g[:-1]) != np.sign(g[1:]))
    root = brentq(lambda d: mhat(d) - d_hat, dg[ch[0]], dg[ch[0] + 1]) if ch.size == 1 else float("nan")
    x_r = np.array([[math.log(root), ph, math.log(a_t)]])
    neff_r, tau_r, se_r, mass_r = local_stats(x_r, T["X"], T["resid"], mass=True)
    print("\nC5 dark thin node: true d %.2f um, mu %.1f /um, alpha %.2f, d_hat %.4f um; crossings %d; root %.4f um "
          "(error %+.4f um); at the root N_eff %.1f, SE %.4f, kernel mass %.3f (replicates' 5th percentile %.2f); "
          "the band at this alpha starts at d = %.2f um"
          % (d_t, mu_t, a_t, d_hat, ch.size, root, root - d_t, float(neff_r[0]), float(se_r[0]), float(mass_r[0]),
             m05, a_t * math.cos(ph) / CFG["mu_range"][1]))
    np.savez("check_table_se_coverage_%s.npz" % CFG["tau_mode"], X0=T["X"], y0=T["y"], lam0=T["lam"], **{k: v for k, v in R.items()})
    print("\n(%.0f s)" % (time.time() - t0))


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "homo":
        CFG["tau_mode"] = "homo"
    main()
