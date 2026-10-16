#!/usr/bin/env python3
"""Can different parameter sets of I_model draw the same shadow? (2026-10-15)

A standalone check, not pipeline code; it imports allen_diameter (9e3296c)
and check_objective.py. The user's thesis: with a flexible kernel whose shape
parameter runs from the Gaussian to heavy tails, fitted with the width and the
diameter, the diameter, width and kernel shape are not so correlated, so a
different parameter set cannot reproduce the same shadow. This script tries
to refute it.

Kernel family: Student's t with nu degrees of freedom (blur_kernels "student":
unit area, FWHM w, maximum at 0; nu in [1, 100], nu = 1 Lorentzian, nu -> inf
Gaussian, nu = 100 used as the Gaussian end, as the pipeline's bound).

Part 1, impostors. Noise-free truth profiles on Block 5's 53 offsets (mu 1/um,
v0 0.1 um, phi 0.2 rad, B_bar 180 gl, w 0.5 um), for d_true in {0.6, 1.5} um
and nu_true in {100, 4, 1.5}. For each truth and each d = f * d_true with f in
REL_D, the best impostor minimises sum_n (I_imp(v_n) - I_true(v_n))^2 over
(mu, v0, w, nu) in the pipeline's box, by trf from several starts. Reported:
Delta chi^2(d) = sum_n (I_imp - I_true)^2 / sigma^2 for sigma = 1 and 2 grey
levels (the noise SD of one profile), max |I_imp - I_true|, the impostor.

Part 2, mechanism (faint limit, kernels with finite fourth moment): the
chord's cumulants are kappa_2 = d^2/16 and kappa_4 = -d^4/256 (checked
numerically), so mu, v0, w and nu can match the dip's area, centre, kappa_2
and kappa_4 for a range of d. The dips' kappa_2 and kappa_4 of truth and
best impostor are compared where nu > 6 for both.

Part 3, the approach (Monte Carlo): noisy draws (SD 2 gl) of four truths,
d_true in {0.6, 1.5} um x nu_true in {100, 4}, fitted with the pipeline
(fit_variants.PlaneFits): Block 5's fit; the Gaussian at a fitted width; the
Student kernel with nu and w fitted, unweighted and with lambda 0.5 and
0.25 um.

Criteria, fixed before the first run:
  REFUTED IN PRACTICE  if, for some truth, an impostor with |d/d_true - 1| >= 0.2
                       has Delta chi^2 < 1 at sigma = 1 gl (one profile cannot
                       tell them apart even at the lower noise).
  REFUTED EXACTLY      if such an impostor has max |I_imp - I_true| < 1e-6 gl.
  APPROACH IDENTIFIES d  if, for every truth, the unweighted Student fit has
                       |median(d_hat) - d_true| <= 0.05 um and SD(d_hat) <= 0.05 um.

    python identifiability_check.py --repo "<repo>/Passive Features/Diameter Re-measurement" --out <dir> [--n-mc 40]
Pure ASCII.
"""
import argparse
import json
import math
import os
import time
from multiprocessing import Pool

import numpy as np
from scipy import integrate
from scipy.optimize import least_squares

import check_objective as CO

CONFIG = dict(seed=20261016, phi=0.2, B=180.0, mu=1.0, v0=0.1, w=0.5, n_nodes=848,
              truths=((0.6, 100.0), (0.6, 4.0), (0.6, 1.5), (1.5, 100.0), (1.5, 4.0), (1.5, 1.5)),
              rel_d=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.05, 1.1, 1.2, 1.35, 1.5),
              w_starts=(0.3, 0.6, 1.0), nu_starts=(2.0, 10.0), sigmas=(1.0, 2.0),
              mc_truths=((0.6, 100.0), (0.6, 4.0), (1.5, 100.0), (1.5, 4.0)), noise=2.0)
TOL = dict(refute_rel=0.2, refute_dchi2=1.0, exact_gl=1e-6, approach_bias=0.05, approach_sd=0.05)
REPO = None


def model(P, v, d, mu, v0, w, nu):
    c = CONFIG
    g = P.BK.kernel_pdf("student", w, nu)
    return P.TM.model_profile_kernel(v, d, mu * d / math.cos(c["phi"]), v0, g, c["B"], c["n_nodes"])


def impostor(P, v, I_true, d, starts):
    """min over q = (mu, v0, w, log10 nu) of 0.5 sum ((I_imp - I_true)/B)^2 at fixed d."""
    c = CONFIG
    lb, ub = np.array([0.0, -1.0, 0.1, 0.0]), np.array([20.0, 1.0, 2.0, 2.0])

    def res(q):
        return (model(P, v, d, q[0], q[1], q[2], 10.0 ** q[3]) - I_true) / c["B"]
    best = None
    for q0 in starts:
        q0 = np.clip(np.asarray(q0, dtype=float), lb + 1e-9, ub - 1e-9)
        r = least_squares(res, q0, bounds=(lb, ub), method="trf", jac="2-point", x_scale="jac",
                          ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=2000)
        if best is None or r.cost < best.cost:
            best = r
    q = best.x
    diff = model(P, v, d, q[0], q[1], q[2], 10.0 ** q[3]) - I_true
    return dict(d=float(d), mu=float(q[0]), v0=float(q[1]), w=float(q[2]), nu=float(10.0 ** q[3]),
                rss_gl2=float(np.sum(diff ** 2)), max_abs_gl=float(np.max(np.abs(diff))),
                dchi2={("%g" % s): float(np.sum(diff ** 2) / s ** 2) for s in c["sigmas"]},
                at_bound=[nm for nm, x, lo, hi in zip(("mu", "v0", "w", "lognu"), q, lb, ub)
                          if x - lo < 1e-6 * (hi - lo) or hi - x < 1e-6 * (hi - lo)])


def run_truth(truth):
    P = CO.load(REPO)
    c = CONFIG
    d_t, nu_t = truth
    v = P.profiles.profile_offsets(P.cfg)
    I_true = model(P, v, d_t, c["mu"], c["v0"], c["w"], nu_t)
    n_star = int(np.argmin(I_true))
    alpha0 = -math.log(I_true[n_star] / c["B"])
    rows, prev = [], None
    # walk outwards from the truth on each side, so the warm start follows the valley
    sides = ([f for f in c["rel_d"] if f < 1][::-1], [f for f in c["rel_d"] if f > 1])
    for side in sides:
        prev = np.array([c["mu"], c["v0"], c["w"], math.log10(nu_t)])
        for f in side:
            d = f * d_t
            starts = [prev, [c["mu"] * (d_t / d) ** 2, c["v0"], c["w"], math.log10(nu_t)]]
            for w0 in c["w_starts"]:
                for nu0 in c["nu_starts"]:
                    starts.append([alpha0 * math.cos(c["phi"]) / d, v[n_star], w0, math.log10(nu0)])
            r = impostor(P, v, I_true, d, starts)
            r["rel"] = f
            rows.append(r)
            prev = np.array([r["mu"], r["v0"], r["w"], math.log10(r["nu"])])
    rows.sort(key=lambda x: x["rel"])
    return dict(d_true=d_t, nu_true=nu_t, rows=rows, v=v.tolist(), I_true=I_true.tolist())


def chord_cumulants(d, n=200001):
    """kappa_2, kappa_4 of the density proportional to s_d(u) (numerical)."""
    u = np.linspace(-d / 2, d / 2, n)
    f = np.sqrt(np.clip(1 - (2 * u / d) ** 2, 0, None))
    m0 = integrate.trapezoid(f, u)
    m2 = integrate.trapezoid(u ** 2 * f, u) / m0
    m4 = integrate.trapezoid(u ** 4 * f, u) / m0
    return float(m2), float(m4 - 3 * m2 ** 2)


def dip_cumulants(P, d, mu, v0, w, nu, half=40.0, n=16001):
    """kappa_2, kappa_4 (um^2, um^4) of the noise-free dip 1 - I/B on +-half um (the tails of a
    Student kernel with nu > 6 beyond 40 um change kappa_4 by far less than the differences compared)."""
    c = CONFIG
    vv = np.linspace(-half, half, n)
    g = P.BK.kernel_pdf("student", w, nu)
    D = 1.0 - P.TM.model_profile_kernel(vv, d, mu * d / math.cos(c["phi"]), v0, g, 1.0, 2000)
    m0 = integrate.trapezoid(D, vv)
    m1 = integrate.trapezoid(vv * D, vv) / m0
    m2 = integrate.trapezoid((vv - m1) ** 2 * D, vv) / m0
    m4 = integrate.trapezoid((vv - m1) ** 4 * D, vv) / m0
    return dict(area=float(m0), k2=float(m2), k4=float(m4 - 3 * m2 ** 2))


def run_mc(args):
    truth, n_mc = args
    P = CO.load(REPO)
    c = CONFIG
    FV = P.FV
    d_t, nu_t = truth
    v = P.profiles.profile_offsets(P.cfg)
    I0 = model(P, v, d_t, c["mu"], c["v0"], c["w"], nu_t)
    keys = ("block5", "gauss_w", "student", "student_l050", "student_l025")
    specs = dict(gauss_w=FV.VariantSpec("gauss", None, False, True, None),
                 student=FV.VariantSpec("student", None, True, True, None),
                 student_l050=FV.VariantSpec("student", None, True, True, 0.5),
                 student_l025=FV.VariantSpec("student", None, True, True, 0.25))
    out = {k: dict(d=[], w=[], nu=[], at_bound=[]) for k in keys}
    for n in range(n_mc):
        rng = np.random.default_rng([c["seed"], int(d_t * 100), int(nu_t), n])
        I = I0 + rng.normal(0.0, c["noise"], v.size)
        pf = FV.PlaneFits(v, I, c["phi"], c["B"], P.cfg)
        fits = dict(block5=pf.base, **{k: pf.fit(s) for k, s in specs.items()})
        for k in keys:
            f = fits[k]
            out[k]["d"].append(f.d_um)
            out[k]["w"].append(f.fwhm_um)
            out[k]["nu"].append(f.shape if f.shape is not None else float("nan"))
            out[k]["at_bound"].append(";".join(f.at_bound))
    summ = {}
    for k in keys:
        d = np.array(out[k]["d"], dtype=float)
        summ[k] = dict(median=float(np.median(d)), mean=float(d.mean()), sd=float(d.std(ddof=1)),
                       q10=float(np.quantile(d, 0.1)), q90=float(np.quantile(d, 0.9)),
                       n_bound=int(sum(1 for s in out[k]["at_bound"] if s)),
                       bounds=sorted(set(x for s in out[k]["at_bound"] for x in s.split(";") if x)),
                       nu_median=float(np.nanmedian(np.array(out[k]["nu"], dtype=float))) if k.startswith("student") else None,
                       w_median=float(np.median(np.array(out[k]["w"], dtype=float))))
    return dict(d_true=d_t, nu_true=nu_t, raw=out, summary=summ)


def main():
    global REPO
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-mc", type=int, default=40)
    ap.add_argument("--procs", type=int, default=2)
    a = ap.parse_args()
    REPO = a.repo
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    res = dict(config=CONFIG, tol=TOL)
    with Pool(a.procs, initializer=_init, initargs=(a.repo,)) as pool:
        res["impostors"] = pool.map(run_truth, CONFIG["truths"])
        print("part 1 done (%.0f s)" % (time.time() - t0), flush=True)
        res["mc"] = pool.map(run_mc, [(t, a.n_mc) for t in CONFIG["mc_truths"]])
        print("part 3 done (%.0f s)" % (time.time() - t0), flush=True)
    P = CO.load(REPO)
    res["chord"] = {("%g" % d): dict(zip(("k2", "k4"), chord_cumulants(d)), k2_pred=d * d / 16, k4_pred=-d ** 4 / 256)
                    for d in (0.3, 0.6, 1.5)}
    pairs = []
    for t in res["impostors"]:
        if t["nu_true"] <= 6:
            continue
        for r in t["rows"]:
            if r["nu"] > 6 and r["dchi2"]["1"] < 4:
                tr = dip_cumulants(P, t["d_true"], CONFIG["mu"], CONFIG["v0"], CONFIG["w"], t["nu_true"])
                im = dip_cumulants(P, r["d"], r["mu"], r["v0"], r["w"], r["nu"])
                pairs.append(dict(d_true=t["d_true"], nu_true=t["nu_true"], d_imp=r["d"], nu_imp=r["nu"],
                                  w_imp=r["w"], dchi2_1=r["dchi2"]["1"], truth=tr, impostor=im))
    res["cumulant_pairs"] = pairs
    # verdicts
    cand = [(t["d_true"], t["nu_true"], r) for t in res["impostors"] for r in t["rows"]
            if abs(r["rel"] - 1) >= TOL["refute_rel"] - 1e-9]
    prac = [x for x in cand if x[2]["dchi2"]["1"] < TOL["refute_dchi2"]]
    exact = [x for x in prac if x[2]["max_abs_gl"] < TOL["exact_gl"]]
    ident = all(abs(m["summary"]["student"]["median"] - m["d_true"]) <= TOL["approach_bias"]
                and m["summary"]["student"]["sd"] <= TOL["approach_sd"] for m in res["mc"])
    res["verdict"] = dict(refuted_in_practice=bool(prac), n_practical=len(prac), refuted_exactly=bool(exact),
                          approach_identifies_d=bool(ident))
    res["seconds"] = time.time() - t0
    json.dump(res, open(os.path.join(a.out, "identifiability.json"), "w"), indent=1, default=float)
    print("verdict:", res["verdict"], "(%.0f s)" % res["seconds"])


def _init(repo):
    global REPO
    REPO = repo


if __name__ == "__main__":
    main()
