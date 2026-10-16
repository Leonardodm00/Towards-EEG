#!/usr/bin/env python3
"""Can different parameter sets of I_model draw the same shadow when the blur
kernel has THREE shape parameters? (2026-10-15, the user's clarification)

A standalone check, not pipeline code; it imports allen_diameter (9e3296c),
check_objective.py, identifiability_check.py (for its Student run, to compare)
and sgt_kernel.py (the kernel family: skewed generalized t, two-piece form,
FWHM w, shape p (peak and shoulders), r (tail weight), gamma (skew); Gaussian,
Laplace, Student t / Lorentzian, generalized normal / box-like and their skewed
versions are members).

Part A, impostors. Noise-free truth profiles on Block 5's 53 offsets (mu 1 /um,
v0 0.1 um, phi 0.2 rad, B_bar 180 gl, w 0.5 um), d_true in {0.6, 1.5} um, the
kernel shapes (p, r, gamma): G (2, 0, 1) Gaussian, S4 (2, 0.5, 1) Student nu 4,
L (1, 0, 1) Laplace, B4 (4, 0, 1) platykurtic generalized normal, K (2, 0, 1.5)
skewed Gaussian. For each d = f d_true (f in REL_D) the best impostor minimises
sum_n (I_imp(v_n) - I_true(v_n))^2 over (mu, v0, w, log10 p, r, ln gamma) in
the box below, by trf from 10 starts (warm, area-scaled truth, 4 shapes x 2
widths). Reported: Delta chi^2 at sigma 1 and 2 gl, max |I_imp - I_true|.

Part B, the approach (Monte Carlo): noisy draws (SD 2 gl) of five truths
(0.6 G, 0.6 S4, 1.5 G, 1.5 S4, 1.5 B4) fitted with all seven parameters free
(d, mu, v0, w, p, r, gamma), unweighted and centre-weighted (IRLS as D-046,
lambda 0.5 um); starts from the pipeline's Gaussian fit at fitted width.

Criteria, fixed before the first run:
  REFUTED IN PRACTICE  (thin, thick separately) if, for some truth, an impostor
                       with |d/d_true - 1| >= 0.2 has Delta chi^2 < 1 at 1 gl.
  REFUTED EXACTLY      if such an impostor has max |I_imp - I_true| < 1e-6 gl.
  THICK IDENTIFIABLE   if, for every thick truth, every impostor with
                       |d/d_true - 1| >= 0.1 has Delta chi^2 >= 4 at 1 gl.
  APPROACH IDENTIFIES d  per truth and weighting, if |median(d_hat) - d_true|
                       <= 0.05 um and SD(d_hat) <= 0.05 um.
  SEARCH CHECK         for the S4 truths (the same profiles as the Student run's
                       nu 4 truths, whose impostors are members of this family),
                       Delta chi^2_SGT(d) <= Delta chi^2_Student(d) + 1e-6 at every d;
                       a failure means the search missed minima (values are then
                       upper bounds).
Predictions written before the run: thin refuted for every kernel; thick
loses Delta chi^2 against the Student-only family on the thinner side
(platykurtic kernels mimic the chord's flat top), size unknown.

    python identifiability_sgt.py --repo "<repo>/Passive Features/Diameter Re-measurement" \
        --student <dir>/identifiability.json --out <dir> [--n-mc 30] [--procs 2]
Pure ASCII.
"""
import argparse
import json
import math
import os
import time
from multiprocessing import Pool

import numpy as np
from scipy.optimize import least_squares

import check_objective as CO
import sgt_kernel as SK

CONFIG = dict(seed=20261017, phi=0.2, B=180.0, mu=1.0, v0=0.1, w=0.5, n_nodes=848,
              d_truths=(0.6, 1.5),
              kernels=dict(G=(2.0, 0.0, 1.0), S4=(2.0, 0.5, 1.0), L=(1.0, 0.0, 1.0), B4=(4.0, 0.0, 1.0),
                           K=(2.0, 0.0, 1.5)),
              rel_d=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.05, 1.1, 1.2, 1.35, 1.5),
              shape_starts=((2.0, 0.0, 1.0), (1.0, 0.0, 1.0), (5.0, 0.0, 1.0), (2.0, 1.0, 1.0)),
              w_starts=(0.4, 0.7), sigmas=(1.0, 2.0),
              mc_truths=((0.6, "G"), (0.6, "S4"), (1.5, "G"), (1.5, "S4"), (1.5, "B4")),
              noise=2.0, lam=0.5, irls_tol=1e-4, irls_max=30)
# box: mu (1/um), v0 (um), w (um), log10 p, r, ln gamma; d (um) for the Monte Carlo
LB_Q = np.array([0.0, -1.0, 0.1, 0.0, 0.0, -0.7])
UB_Q = np.array([20.0, 1.0, 2.0, 1.0, 2.0, 0.7])
D_BOUNDS = (0.05, 6.0)
TOL = dict(refute_rel=0.2, refute_dchi2=1.0, exact_gl=1e-6, thick_rel=0.1, thick_dchi2=4.0,
           approach_bias=0.05, approach_sd=0.05, search=1e-6)
REPO = None
STUDENT = None


def model(P, v, d, mu, v0, w, p, r, gamma):
    c = CONFIG
    return P.TM.model_profile_kernel(v, d, mu * d / math.cos(c["phi"]), v0, SK.sgt_pdf(w, p, r, gamma),
                                     c["B"], c["n_nodes"])


def model_q(P, v, d, q):
    """q = (mu, v0, w, log10 p, r, ln gamma)."""
    return model(P, v, d, q[0], q[1], q[2], 10.0 ** q[3], q[4], math.exp(q[5]))


def q_of(mu, v0, w, shape):
    p, r, g = shape
    return np.clip(np.array([mu, v0, w, math.log10(p), r, math.log(g)]), LB_Q + 1e-9, UB_Q - 1e-9)


def _flags(x, lb, ub, names):
    return [nm + ("_lo" if xi - lo <= 1e-6 * (hi - lo) else "_hi") for nm, xi, lo, hi in zip(names, x, lb, ub)
            if xi - lo <= 1e-6 * (hi - lo) or hi - xi <= 1e-6 * (hi - lo)]


def impostor(P, v, I_true, d, starts):
    c = CONFIG

    def res(q):
        return (model_q(P, v, d, q) - I_true) / c["B"]
    best = None
    for q0 in starts:
        q0 = np.clip(np.asarray(q0, dtype=float), LB_Q + 1e-9, UB_Q - 1e-9)
        r = least_squares(res, q0, bounds=(LB_Q, UB_Q), method="trf", jac="2-point", x_scale="jac",
                          ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=3000)
        if best is None or r.cost < best.cost:
            best = r
    q = best.x
    diff = model_q(P, v, d, q) - I_true
    return dict(d=float(d), mu=float(q[0]), v0=float(q[1]), w=float(q[2]), p=float(10.0 ** q[3]), r=float(q[4]),
                gamma=float(math.exp(q[5])), rss_gl2=float(np.sum(diff ** 2)), max_abs_gl=float(np.max(np.abs(diff))),
                dchi2={("%g" % s): float(np.sum(diff ** 2) / s ** 2) for s in c["sigmas"]},
                at_bound=_flags(q, LB_Q, UB_Q, ("mu", "v0", "w", "log10p", "r", "lngamma")))


def run_truth(task):
    d_t, kname = task
    P = CO.load(REPO)
    c = CONFIG
    shape_t = c["kernels"][kname]
    v = P.profiles.profile_offsets(P.cfg)
    I_true = model(P, v, d_t, c["mu"], c["v0"], c["w"], *shape_t)
    n_star = int(np.argmin(I_true))
    alpha0 = -math.log(I_true[n_star] / c["B"])
    rows = []
    sides = ([f for f in c["rel_d"] if f < 1][::-1], [f for f in c["rel_d"] if f > 1])
    for side in sides:
        prev = q_of(c["mu"], c["v0"], c["w"], shape_t)
        for f in side:
            d = f * d_t
            starts = [prev, q_of(c["mu"] * (d_t / d) ** 2, c["v0"], c["w"], shape_t)]
            for shp in c["shape_starts"]:
                for w0 in c["w_starts"]:
                    starts.append(q_of(alpha0 * math.cos(c["phi"]) / d, v[n_star], w0, shp))
            r = impostor(P, v, I_true, d, starts)
            r["rel"] = f
            rows.append(r)
            prev = q_of(r["mu"], r["v0"], r["w"], (r["p"], r["r"], r["gamma"]))
    rows.sort(key=lambda x: x["rel"])
    return dict(d_true=d_t, kernel=kname, shape=list(shape_t), rows=rows, v=v.tolist(), I_true=I_true.tolist())


# ---------------------------------------------------------------- Part B
NAMES_T = ("d", "mu", "v0", "w", "log10p", "r", "lngamma")


def _box_theta():
    lb = np.concatenate(([D_BOUNDS[0]], LB_Q))
    ub = np.concatenate(([D_BOUNDS[1]], UB_Q))
    return lb, ub


def _theta_model(P, v, th):
    return model_q(P, v, th[0], th[1:])


def _lsq(P, v, I, th0, omega=None):
    c = CONFIG
    lb, ub = _box_theta()
    sw = 1.0 if omega is None else np.sqrt(omega)

    def res(th):
        return sw * (_theta_model(P, v, th) - I) / c["B"]
    th0 = np.clip(np.asarray(th0, dtype=float), lb + 1e-9, ub - 1e-9)
    return least_squares(res, th0, bounds=(lb, ub), method="trf", jac="2-point", x_scale="jac",
                         ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=3000)


def fit_sgt(P, v, I, gauss_fit, base_fit):
    """Unweighted SGT fit, all seven free; starts: the Gaussian fitted-width fit with each
    start shape, and Block 5's (d, mu, v0) at w 0.7 um (a generic width, not the truth's), Gaussian."""
    c = CONFIG
    starts = [np.concatenate(([gauss_fit.d_um], q_of(gauss_fit.mu_per_um, gauss_fit.v0_um, gauss_fit.fwhm_um, s)))
              for s in c["shape_starts"]]
    starts.append(np.concatenate(([base_fit.d_um], q_of(base_fit.mu_per_um, base_fit.v0_um, 0.7, (2.0, 0.0, 1.0)))))
    runs = [_lsq(P, v, I, s) for s in starts]
    ok = [r for r in runs if r.status > 0 and np.all(np.isfinite(r.x))]
    best = min(ok if ok else runs, key=lambda r: r.cost)
    return best, bool(ok)


def fit_sgt_weighted(P, v, I, start):
    """IRLS with omega_n = exp(-|v_n - v0| / lambda), re-centred, from the unweighted fit."""
    c = CONFIG
    th = np.array(start.x, dtype=float)
    ok, run = False, start
    for _ in range(c["irls_max"]):
        omega = np.exp(-np.abs(v - th[2]) / c["lam"])
        run = _lsq(P, v, I, th, omega)
        if not (run.status > 0 and np.all(np.isfinite(run.x))):
            break
        done = abs(run.x[0] - th[0]) <= c["irls_tol"] and abs(run.x[2] - th[2]) <= c["irls_tol"]
        th = np.array(run.x, dtype=float)
        if done:
            ok = True
            break
    return run, ok


def _row(run, conv):
    lb, ub = _box_theta()
    th = run.x
    return dict(d=float(th[0]), mu=float(th[1]), v0=float(th[2]), w=float(th[3]), p=float(10.0 ** th[4]),
                r=float(th[5]), gamma=float(math.exp(th[6])), converged=bool(conv), at_bound=_flags(th, lb, ub, NAMES_T))


def run_mc(task):
    (d_t, kname), n_mc = task
    P = CO.load(REPO)
    c = CONFIG
    FV = P.FV
    v = P.profiles.profile_offsets(P.cfg)
    shape_t = c["kernels"][kname]
    I0 = model(P, v, d_t, c["mu"], c["v0"], c["w"], *shape_t)
    kidx = sorted(c["kernels"]).index(kname)
    out = dict(gauss_w=[], sgt=[], sgt_l050=[])
    for n in range(n_mc):
        rng = np.random.default_rng([c["seed"], int(round(d_t * 100)), kidx, n])
        I = I0 + rng.normal(0.0, c["noise"], v.size)
        pf = FV.PlaneFits(v, I, c["phi"], c["B"], P.cfg)
        g = pf.fit(FV.VariantSpec("gauss", None, False, True, None))
        out["gauss_w"].append(dict(d=g.d_um, w=g.fwhm_um, at_bound=list(g.at_bound)))
        u, conv = fit_sgt(P, v, I, g, pf.base)
        out["sgt"].append(_row(u, conv))
        wt, ok = fit_sgt_weighted(P, v, I, u)
        out["sgt_l050"].append(_row(wt, ok))
    summ = {}
    for k, rows in out.items():
        d = np.array([r["d"] for r in rows], dtype=float)
        med, sd = float(np.median(d)), float(d.std(ddof=1))
        summ[k] = dict(median=med, sd=sd, q10=float(np.quantile(d, 0.1)), q90=float(np.quantile(d, 0.9)),
                       n_mu_hi=int(sum(1 for r in rows if "mu_hi" in r["at_bound"])),
                       n_bound=int(sum(1 for r in rows if r["at_bound"])),
                       identifies=bool(abs(med - d_t) <= TOL["approach_bias"] and sd <= TOL["approach_sd"]))
        if k != "gauss_w":
            summ[k].update(w_median=float(np.median([r["w"] for r in rows])),
                           p_median=float(np.median([r["p"] for r in rows])),
                           r_median=float(np.median([r["r"] for r in rows])),
                           gamma_median=float(np.median([r["gamma"] for r in rows])))
    return dict(d_true=d_t, kernel=kname, raw=out, summary=summ)


def _init(repo):
    global REPO
    REPO = repo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--student", required=True, help="identifiability.json of the Student run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-mc", type=int, default=30)
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--part", choices=("A", "B", "AB"), default="AB")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "identifiability_sgt.json")
    res = json.load(open(path)) if os.path.exists(path) else dict(config=CONFIG, tol=TOL)
    t0 = time.time()
    with Pool(a.procs, initializer=_init, initargs=(a.repo,)) as pool:
        if "A" in a.part:
            tasks = [(d, k) for d in CONFIG["d_truths"] for k in CONFIG["kernels"]]
            res["impostors"] = pool.map(run_truth, tasks)
            print("part A done (%.0f s)" % (time.time() - t0), flush=True)
            json.dump(res, open(path, "w"), indent=1, default=float)
        if "B" in a.part:
            res["mc"] = pool.map(run_mc, [(t, a.n_mc) for t in CONFIG["mc_truths"]])
            print("part B done (%.0f s)" % (time.time() - t0), flush=True)
    # verdicts
    v = {}
    if "impostors" in res:
        for tag, dsel in (("thin", 0.6), ("thick", 1.5)):
            cand = [(t["kernel"], r) for t in res["impostors"] if abs(t["d_true"] - dsel) < 1e-9 for r in t["rows"]
                    if abs(r["rel"] - 1) >= TOL["refute_rel"] - 1e-9]
            prac = [x for x in cand if x[1]["dchi2"]["1"] < TOL["refute_dchi2"]]
            v["refuted_in_practice_" + tag] = bool(prac)
            v["n_practical_" + tag] = len(prac)
            v["refuted_exactly_" + tag] = bool([x for x in prac if x[1]["max_abs_gl"] < TOL["exact_gl"]])
        thick_rows = [r for t in res["impostors"] if abs(t["d_true"] - 1.5) < 1e-9 for r in t["rows"]
                      if abs(r["rel"] - 1) >= TOL["thick_rel"] - 1e-9]
        v["thick_identifiable"] = all(r["dchi2"]["1"] >= TOL["thick_dchi2"] for r in thick_rows)
        S = json.load(open(a.student))
        worst = -np.inf
        for t in res["impostors"]:
            if t["kernel"] != "S4":
                continue
            st = [s for s in S["impostors"] if abs(s["d_true"] - t["d_true"]) < 1e-9 and s["nu_true"] == 4.0][0]
            for r in t["rows"]:
                rs = [x for x in st["rows"] if abs(x["rel"] - r["rel"]) < 1e-9][0]
                worst = max(worst, r["dchi2"]["1"] - rs["dchi2"]["1"])
        v["search_check_worst_excess"] = float(worst)
        v["search_check_pass"] = bool(worst <= TOL["search"])
    if "mc" in res:
        v["approach"] = {"%g %s" % (m["d_true"], m["kernel"]): {k: m["summary"][k]["identifies"] for k in m["summary"]}
                         for m in res["mc"]}
    res["verdict"] = v
    res["seconds_last_run"] = time.time() - t0
    json.dump(res, open(path, "w"), indent=1, default=float)
    print("verdict:", json.dumps(v, indent=1))


if __name__ == "__main__":
    main()
