#!/usr/bin/env python3
"""Does pinning the kernel make a thin tube's d identifiable? (2026-10-15)

A standalone check, not pipeline code; it imports allen_diameter (9e3296c),
check_objective.py and identifiability_check.py, and reads the Monte Carlo of
identifiability_check.py (identifiability.json, part 3).

Same thin truths and the same noise draws as that Monte Carlo (d_true 0.6 um,
nu_true in {100, 4}, w 0.5 um FWHM, mu 1 /um, v0 0.1 um, phi 0.2 rad, B_bar
180 gl, noise SD 2 gl, 40 draws, seeds [20261016, 60, nu, n]). Each draw is
refitted by the pipeline (fit_variants.fit_unweighted / fit_weighted) with
the Student kernel PINNED -- w and nu fixed, d, mu, v0 fitted:
  oracle      the true kernel (w 0.5 um, nu_true);
  from_thick  the kernel the free Student fit (w, nu fitted, unweighted)
              returned on draw n of the thick truth (d 1.5 um, same nu_true,
              same w) in that Monte Carlo: a one-node calibration with its
              own noise; unweighted, and weighted at lambda 0.5 and 0.25 um.
The width is pinned through the pipeline's fixed-width mode, by setting
sigma_fit_um = w_pin / (2 sqrt(2 ln 2)) in a copy of the config.

Criterion, fixed before the run (the same as identifiability_check.py's):
  IDENTIFIES d  if |median(d_hat) - 0.6| <= 0.05 um and SD(d_hat) <= 0.05 um.
Prediction written before the run: the oracle passes; from_thick passes with
a larger SD, the calibration's width error carried into d roughly along
sigma_w^2 + d^2/16 = const.

    python kernel_pinning_check.py --repo "<repo>/Passive Features/Diameter Re-measurement" --res <dir>/identifiability.json
Pure ASCII.
"""
import argparse
import dataclasses
import json
import os
import time
from multiprocessing import Pool

import numpy as np

import check_objective as CO
import identifiability_check as IC

CONFIG = dict(thin=((0.6, 100.0), (0.6, 4.0)), d_thick=1.5, lams=(0.5, 0.25))
TOL = dict(bias=0.05, sd=0.05)
REPO = None
RES = None


def pinned_fits(P, v, I, w_pin, nu_pin, lams):
    """Unweighted and weighted Student fits with (w, nu) pinned; dict label -> VariantFit."""
    c = IC.CONFIG
    cfg = dataclasses.replace(P.cfg, sigma_fit_um=float(w_pin) / P.BK.gauss_fwhm(1.0))
    nu = float(min(max(nu_pin, 1.0), 100.0))
    spec = P.FV.VariantSpec("student", nu, False, False, None)
    u = P.FV.fit_unweighted(v, I, c["phi"], c["B"], cfg, spec)
    out = {"none": u}
    for lam in lams:
        out["lam=%g" % lam] = P.FV.fit_weighted(v, I, c["phi"], c["B"], cfg,
                                                P.FV.VariantSpec("student", nu, False, False, lam), u)
    return out


def run_truth(truth):
    P = CO.load(REPO)
    c = IC.CONFIG
    d_t, nu_t = truth
    R = json.load(open(RES))
    thick = [m for m in R["mc"] if abs(m["d_true"] - CONFIG["d_thick"]) < 1e-9 and m["nu_true"] == nu_t][0]
    w_cal = np.array(thick["raw"]["student"]["w"], dtype=float)
    nu_cal = np.array(thick["raw"]["student"]["nu"], dtype=float)
    v = P.profiles.profile_offsets(P.cfg)
    I0 = IC.model(P, v, d_t, c["mu"], c["v0"], c["w"], nu_t)
    rows = []
    for n in range(w_cal.size):
        rng = np.random.default_rng([c["seed"], int(d_t * 100), int(nu_t), n])   # = identifiability_check.run_mc
        I = I0 + rng.normal(0.0, c["noise"], v.size)
        row = dict(n=n, w_cal=float(w_cal[n]), nu_cal=float(nu_cal[n]))
        for tag, (wp, nup, lams) in (("oracle", (c["w"], nu_t, ())),
                                     ("from_thick", (w_cal[n], nu_cal[n], CONFIG["lams"]))):
            for lab, f in pinned_fits(P, v, I, wp, nup, lams).items():
                row["%s|%s" % (tag, lab)] = dict(d=f.d_um, mu=f.mu_per_um, v0=f.v0_um, status=f.status,
                                                 at_bound=list(f.at_bound))
        rows.append(row)
    keys = [k for k in rows[0] if "|" in k]
    summ = {}
    for k in keys:
        d = np.array([r[k]["d"] for r in rows], dtype=float)
        nb = sum(1 for r in rows if r[k]["at_bound"])
        med, sd = float(np.median(d)), float(d.std(ddof=1))
        summ[k] = dict(median=med, sd=sd, q10=float(np.quantile(d, 0.1)), q90=float(np.quantile(d, 0.9)),
                       n_bound=nb, bounds=sorted(set(b for r in rows for b in r[k]["at_bound"])),
                       identifies=bool(abs(med - d_t) <= TOL["bias"] and sd <= TOL["sd"]))
    cal = dict(w_median=float(np.median(w_cal)), w_sd=float(w_cal.std(ddof=1)),
               nu_median=float(np.median(nu_cal)), nu_q10=float(np.quantile(nu_cal, 0.1)),
               nu_q90=float(np.quantile(nu_cal, 0.9)))
    return dict(d_true=d_t, nu_true=nu_t, calibrator=cal, rows=rows, summary=summ)


def _init(repo, res):
    global REPO, RES
    REPO, RES = repo, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--res", required=True)
    ap.add_argument("--procs", type=int, default=2)
    a = ap.parse_args()
    t0 = time.time()
    with Pool(a.procs, initializer=_init, initargs=(a.repo, a.res)) as pool:
        out = pool.map(run_truth, CONFIG["thin"])
    res = dict(config=CONFIG, tol=TOL, truths=out, seconds=time.time() - t0)
    path = os.path.join(os.path.dirname(os.path.abspath(a.res)), "pinning.json")
    json.dump(res, open(path, "w"), indent=1, default=float)
    for t in out:
        cal = t["calibrator"]
        print("truth d %.2f nu %g | calibrator w %.3f +- %.3f, nu median %.2f (q10 %.2f, q90 %.2f)"
              % (t["d_true"], t["nu_true"], cal["w_median"], cal["w_sd"], cal["nu_median"], cal["nu_q10"], cal["nu_q90"]))
        for k, s in t["summary"].items():
            print("   %-18s median %.3f sd %.3f q10 %.3f q90 %.3f n_bound %2d %s identifies %s"
                  % (k, s["median"], s["sd"], s["q10"], s["q90"], s["n_bound"], s["bounds"], s["identifies"]))
    print("wrote %s (%.0f s)" % (path, res["seconds"]))


if __name__ == "__main__":
    main()
