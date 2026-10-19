"""check_grouped_design.py -- at a fixed number of measured phantoms, does drawing R nuisance
realizations per cylinder (fixed d, phi, theta, mu) estimate the table better than R x more
independent cylinders? (2026-10-18; standalone, not pipeline code).

Same design, fixture, tau(d) and query points as check_table_se_coverage.py (imported).
Budget N = 1500 measured replicates in both designs:
  independent: 1500 cylinders, one nuisance draw each (the current design, D-024 (iii));
  grouped:     300 cylinders x R = 5 nuisance draws, the replicates of a cylinder at the same x
               (alpha_hat := alpha stand-in, so they coincide).
Ratio model: y = b(x) + delta_g + tau(d) eps, eps ~ N(0, 1) per replicate (the nuisances xi);
delta_g ~ N(0, (c tau(d))^2) per CYLINDER: what the cylinder's own features that are not table axes
(heading theta, mu at a given alpha_hat) add to its mean ratio. c = 0 (they add nothing) or c = 0.5.
Smoothing fixed at 10 (the modal CV choice of check_table_se_coverage.py, hetero) so that only the
design differs.

Prediction (design effect, textbook): Var_grouped / Var_independent = (1 + R c^2) / (1 + c^2), i.e. an
SD ratio of 1.00 at c = 0 and sqrt(2.25 / 1.25) = 1.34 at c = 0.5. Tolerance (fixed before running):
the SD ratio pooled over the 7 query points within 12 % of the prediction (60 tables per arm: the MC
error of each pooled SD is about 3-4 %).

Run:  python3 check_grouped_design.py      (a few minutes on one core)
"""
import math

import numpy as np
from scipy.interpolate import RBFInterpolator

import check_table_se_coverage as cov

CFG = dict(seed=20261019, tables=60, n=1500, groups=300, reps=5, cs=(0.0, 0.5), smoothing=10.0)


def one_table(rng, grouped, c):
    if grouped:
        Xg = cov.design(rng, CFG["groups"])
        X = np.repeat(Xg, CFG["reps"], axis=0)
        tau = cov.tau_of_d(np.exp(X[:, 0]))
        delta = np.repeat(c * cov.tau_of_d(np.exp(Xg[:, 0])) * rng.standard_normal(CFG["groups"]), CFG["reps"])
    else:
        X = cov.design(rng, CFG["n"])
        tau = cov.tau_of_d(np.exp(X[:, 0]))
        delta = c * tau * rng.standard_normal(CFG["n"])
    y = cov.truth(X[:, 0], X[:, 1], X[:, 2]) + delta + tau * rng.standard_normal(X.shape[0])
    sc = cov.Scaler(X)
    f = RBFInterpolator(sc(X), y, kernel="thin_plate_spline", smoothing=CFG["smoothing"])
    return f(sc(cov.query_raw()))


def main():
    cov.CFG["tau_mode"] = "hetero"
    bq = cov.truth(*cov.query_raw().T)
    print("%-6s %-17s %10s %10s %8s" % ("c", "query", "SD indep", "SD group", "ratio"))
    for c in CFG["cs"]:
        sd = {}
        for grouped in (False, True):
            rng = np.random.default_rng([CFG["seed"], int(grouped), int(10 * c)])
            B = np.array([one_table(rng, grouped, c) for _ in range(CFG["tables"])])
            sd[grouped] = B.std(axis=0, ddof=1)
            if grouped:
                bias_g = B.mean(axis=0) - bq
            else:
                bias_i = B.mean(axis=0) - bq
        for i, q in enumerate(cov.CFG["queries"]):
            print("%-6.1f %-17s %10.4f %10.4f %8.2f" % (c, "(%g, %g, %g)" % q, sd[False][i], sd[True][i],
                                                       sd[True][i] / sd[False][i]))
        pooled = math.sqrt(np.mean(sd[True] ** 2) / np.mean(sd[False] ** 2))
        pred = math.sqrt((1 + CFG["reps"] * c ** 2) / (1 + c ** 2))
        print("   c = %.1f: pooled SD ratio %.3f, predicted %.3f -> %s; max |mean error| independent %.4f, "
              "grouped %.4f" % (c, pooled, pred, "PASS" if abs(pooled / pred - 1) <= 0.12 else "FAIL",
                                float(np.abs(bias_i).max()), float(np.abs(bias_g).max())))


if __name__ == "__main__":
    main()
