"""check07_kernel_comparison.py -- does the order of the penalty matter for the bias table? The table's
fixture fitted with SciPy's 'thin_plate_spline' (order 5/2 on R^3), with 'linear' at degree 1 (the 3-D
bending-energy spline, order 2) and with 'linear' at SciPy's default degree 0 (2026-10-18; standalone).
Document section 3.10.

Fixture (check_table_se_coverage.py, copied in tps_common.py): design of D-024 (iii), truth
b = 1 + 0.06/d + 0.15 sin^2 phi + 0.05 ln alpha, alpha_hat := alpha, heteroscedastic tau(d), N_S = 1500,
range normalisation of D-054 (i); tables b = 0..TABLES-1 drawn exactly as there (default_rng([20261018, b])).
Each kernel picks its own smoothing by 5-fold CV over one wide grid (folds from default_rng(20261006)).
Metrics on 3000 evaluation points drawn from the design (default_rng(20261027)), in raw coordinates:
RMS(b_hat - b) over all points, thin (d < 0.5 um) and thick (d >= 1 um); RMS error of the slope
d b_hat / d ln d at fixed (phi, ln alpha_hat) -- the derivative Step 5's inversion and error propagation use --
against the truth's -0.06/d, by central differences of step 1e-3 in ln d; the smoothing chosen and df.
Reported: per-kernel mean over tables with its SE, and the paired difference to 'thin_plate_spline'.
No tolerance: the comparison is the result.
Run:  python3 check07_kernel_comparison.py      (about 5 min on two cores)
"""
import math
from multiprocessing import Pool

import numpy as np
from scipy.interpolate import RBFInterpolator

from tps_common import Scaler, design, hat_trace, tau_of_d, truth

CFG = dict(table_seed=20261018, fold_seed=20261006, eval_seed=20261027, tables=10, n=1500, folds=5, n_eval=3000,
           grid=tuple(10.0 ** k for k in range(-5, 6)), h_lnd=1e-3, workers=2,
           variants=(("thin_plate_spline", 1), ("linear", 1), ("linear", 0)))


def cv_choice(Xs, y, kernel, degree):
    n = Xs.shape[0]
    fold = np.empty(n, dtype=int)
    fold[np.random.default_rng(CFG["fold_seed"]).permutation(n)] = np.arange(n) % CFG["folds"]
    scores = []
    for lam in CFG["grid"]:
        err = 0.0
        try:
            for f in range(CFG["folds"]):
                tr = fold != f
                pred = RBFInterpolator(Xs[tr], y[tr], kernel=kernel, degree=degree, smoothing=lam)(Xs[~tr])
                err += float(np.sum((pred - y[~tr]) ** 2))
        except np.linalg.LinAlgError:
            err = math.inf
        scores.append(err / n)
    return float(CFG["grid"][int(np.argmin(scores))])


def one_table(b):
    rng = np.random.default_rng([CFG["table_seed"], b])
    X = design(rng, CFG["n"])
    y = truth(X[:, 0], X[:, 1], X[:, 2]) + tau_of_d(np.exp(X[:, 0])) * rng.standard_normal(CFG["n"])
    sc = Scaler(X)
    Xs = sc(X)
    E = design(np.random.default_rng(CFG["eval_seed"]), CFG["n_eval"])
    bE = truth(E[:, 0], E[:, 1], E[:, 2])
    slope_true = -0.06 / np.exp(E[:, 0])
    dE = np.exp(E[:, 0])
    h = np.zeros(3)
    h[0] = CFG["h_lnd"]
    rows = []
    for kernel, degree in CFG["variants"]:
        lam = cv_choice(Xs, y, kernel, degree)
        f = RBFInterpolator(Xs, y, kernel=kernel, degree=degree, smoothing=lam)
        err = f(sc(E)) - bE
        slope = (f(sc(E + h)) - f(sc(E - h))) / (2 * CFG["h_lnd"])
        rows.append(dict(lam=lam, df=hat_trace(Xs, lam, kernel=kernel, degree=degree),
                         rms=float(np.sqrt(np.mean(err ** 2))),
                         rms_thin=float(np.sqrt(np.mean(err[dE < 0.5] ** 2))),
                         rms_thick=float(np.sqrt(np.mean(err[dE >= 1.0] ** 2))),
                         slope_rms=float(np.sqrt(np.mean((slope - slope_true) ** 2))),
                         slope_rms_thick=float(np.sqrt(np.mean((slope - slope_true)[dE >= 1.0] ** 2)))))
    return b, rows


def main():
    with Pool(CFG["workers"]) as pool:
        res = dict(pool.map(one_table, range(CFG["tables"])))
    keys = ("rms", "rms_thin", "rms_thick", "slope_rms", "slope_rms_thick", "df")
    names = ["%s/deg%d" % v for v in CFG["variants"]]
    for b in range(CFG["tables"]):
        print("table %d: " % b + " | ".join("%s lam %g rms %.4f slope %.3f" % (nm, r["lam"], r["rms"], r["slope_rms"])
                                           for nm, r in zip(names, res[b])))
    A = {k: np.array([[res[b][i][k] for i in range(len(names))] for b in range(CFG["tables"])]) for k in keys}
    lams = np.array([[res[b][i]["lam"] for i in range(len(names))] for b in range(CFG["tables"])])
    print("\nmean over %d tables (SE):" % CFG["tables"])
    for i, nm in enumerate(names):
        print("  %-22s lambda %s | df %.1f | RMS all %.4f (%.4f) thin %.4f thick %.4f | slope RMS %.3f (%.3f) thick %.4f"
              % (nm, sorted(set(lams[:, i].tolist())), A["df"][:, i].mean(), A["rms"][:, i].mean(),
                 A["rms"][:, i].std(ddof=1) / math.sqrt(CFG["tables"]), A["rms_thin"][:, i].mean(),
                 A["rms_thick"][:, i].mean(), A["slope_rms"][:, i].mean(),
                 A["slope_rms"][:, i].std(ddof=1) / math.sqrt(CFG["tables"]), A["slope_rms_thick"][:, i].mean()))
    print("paired difference to thin_plate_spline (mean, SE):")
    for i, nm in enumerate(names[1:], start=1):
        for k in ("rms", "rms_thick", "slope_rms", "slope_rms_thick"):
            dlt = A[k][:, i] - A[k][:, 0]
            print("  %-22s %-16s %+.4f (%.4f)" % (nm, k, dlt.mean(), dlt.std(ddof=1) / math.sqrt(CFG["tables"])))


if __name__ == "__main__":
    main()
