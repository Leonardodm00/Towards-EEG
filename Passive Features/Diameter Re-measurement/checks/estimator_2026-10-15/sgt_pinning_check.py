#!/usr/bin/env python3
"""Thin d with the three-shape-parameter kernel PINNED from a thick node
(2026-10-15). Standalone check; reads identifiability_sgt.json (its Monte
Carlo) and regenerates the same thin noise draws.

For draw n of the thin truth (d 0.6 um, kernel G or S4), the kernel (w, p, r,
gamma) is fixed at the values the all-free fit returned on draw n of the thick
truth with the same kernel (d 1.5 um, its own noise), and only d, mu, v0 are
fitted: unweighted, and IRLS with lambda 0.5 um. Starts: the pipeline's Block 5
fit and Gaussian fitted-width fit of the same draw.

Criterion, fixed before the run: IDENTIFIES d if |median(d_hat) - 0.6| <= 0.05 um
and SD(d_hat) <= 0.05 um. Prediction: passes for G; S4 uncertain (the thick
fits' shape parameters scatter).

    python sgt_pinning_check.py --repo "<repo>/..." --res <dir>/identifiability_sgt.json
Pure ASCII.
"""
import argparse
import json
import math
import os

import numpy as np
from scipy.optimize import least_squares

import check_objective as CO
import identifiability_sgt as IS

TOL = dict(bias=0.05, sd=0.05)


def fit_pinned(P, v, I, kern, starts, lam=None):
    c = IS.CONFIG
    lb, ub = np.array([IS.D_BOUNDS[0], IS.LB_Q[0], IS.LB_Q[1]]), np.array([IS.D_BOUNDS[1], IS.UB_Q[0], IS.UB_Q[1]])

    def run(th0, omega):
        sw = 1.0 if omega is None else np.sqrt(omega)

        def res(th):
            return sw * (IS.model(P, v, th[0], th[1], th[2], *kern) - I) / c["B"]
        return least_squares(res, np.clip(th0, lb + 1e-9, ub - 1e-9), bounds=(lb, ub), method="trf", jac="2-point",
                             x_scale="jac", ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=2000)
    best = min((run(np.asarray(s, dtype=float), None) for s in starts), key=lambda r: r.cost)
    if lam is None:
        return best
    th = np.array(best.x)
    for _ in range(c["irls_max"]):
        r = run(th, np.exp(-np.abs(v - th[2]) / lam))
        done = abs(r.x[0] - th[0]) <= c["irls_tol"] and abs(r.x[2] - th[2]) <= c["irls_tol"]
        th = np.array(r.x)
        best = r
        if done:
            break
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--res", required=True)
    a = ap.parse_args()
    P = CO.load(a.repo)
    c = IS.CONFIG
    R = json.load(open(a.res))
    v = P.profiles.profile_offsets(P.cfg)
    out = {}
    for kname in ("G", "S4"):
        thick = [m for m in R["mc"] if abs(m["d_true"] - 1.5) < 1e-9 and m["kernel"] == kname][0]["raw"]["sgt"]
        shape_t = c["kernels"][kname]
        I0 = IS.model(P, v, 0.6, c["mu"], c["v0"], c["w"], *shape_t)
        kidx = sorted(c["kernels"]).index(kname)
        rows = dict(none=[], lam=[])
        for n, tk in enumerate(thick):
            rng = np.random.default_rng([c["seed"], 60, kidx, n])          # = identifiability_sgt.run_mc
            I = I0 + rng.normal(0.0, c["noise"], v.size)
            pf = P.FV.PlaneFits(v, I, c["phi"], c["B"], P.cfg)
            g = pf.fit(P.FV.VariantSpec("gauss", None, False, True, None))
            starts = [(pf.base.d_um, pf.base.mu_per_um, pf.base.v0_um), (g.d_um, g.mu_per_um, g.v0_um)]
            kern = (tk["w"], tk["p"], tk["r"], tk["gamma"])
            rows["none"].append(float(fit_pinned(P, v, I, kern, starts).x[0]))
            rows["lam"].append(float(fit_pinned(P, v, I, kern, starts, lam=c["lam"]).x[0]))
        out[kname] = {}
        for k, d in rows.items():
            d = np.array(d)
            med, sd = float(np.median(d)), float(d.std(ddof=1))
            out[kname][k] = dict(median=med, sd=sd, q10=float(np.quantile(d, 0.1)), q90=float(np.quantile(d, 0.9)),
                                 identifies=bool(abs(med - 0.6) <= TOL["bias"] and sd <= TOL["sd"]), d=d.tolist())
            print("thin %-2s pinned from thick, %-4s: median %.3f sd %.3f q10 %.3f q90 %.3f identifies %s"
                  % (kname, k, med, sd, out[kname][k]["q10"], out[kname][k]["q90"], out[kname][k]["identifies"]))
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(a.res)), "pinning_sgt.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
