#!/usr/bin/env python3
"""Standalone check (NOT pipeline code), 2026-11-23 (D-221, D-222): part (c) of the SGT identifiability check,
recovery on simulated noisy profiles.

Per geometry and kernel truth (cases.py): noise-free profiles B tau_k(v_n; truth) (the exact path), B = 150 gl;
n_sets data sets with white Gaussian noise of SD sigma = 1.1 gl per line sample, seeds
numpy.random.default_rng([cases.SEED, cases.case_index(g, truth), set]); the planes node_inference.select_planes
keeps at the truth. Per data set, fitted with fitting.fit (scipy.optimize.least_squares, trf, within Block 30's
bounds and Block 12's sgt_* bounds; per-plane backgrounds solved exactly):
  1. "gauss": today's model (Gaussian kernel), started at the truth's (d, mu, v0, z_ax);
  2. each SGT model of cases.models_for(truth) ("z1": (beta, xi) free, zeta = 1; "zf": (beta, xi, zeta) free),
     warm-started from the Gaussian fit's (d, mu, v0, z_ax) with the Gaussian shape (2, 0, 1) -- how the option
     would start from today's fit; the fast CDF path (validate_model.py: within 5e-13 of the exact model);
  3. on the first n_robust data sets, each SGT model also from Block 12's other start shapes (1, 0, 1), (5, 0, 1),
     (2, 1, 1) (zeta = 1): whether the warm start missed a lower minimum.
Aggregates per (case, model): the error of d (mean, SD, RMSE, and relative to d), the coverage of d's SE
(|d_hat - d| <= SE and <= 1.96 SE; SE by Block 30's rule, and the conditional SE with the parameters at a bound
held), the errors of the shape, the share of fits at a bound (any, and per bound), chi2 per degree of freedom,
nfev and seconds per fit.

Run one batch per diameter, as separate processes (at most two at a time):
    cd "Passive Features/Diameter Re-measurement/checks/sgt_identifiability_2026-11-23"
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 recovery.py --d 0.4 > recovery_d0.4.out
(--mu, --z, --truths to restrict; --n-sets 30, --n-robust 2.) Writes recovery_d<d>.json after every case.
Pure ASCII, LF.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

import cases as C
import fitting as FT
import sgt_line_model as M
from allen_diameter.analysis import node_inference

ROBUST_SHAPES = ((1.0, 0.0, 1.0), (5.0, 0.0, 1.0), (2.0, 1.0, 1.0))
SPEC_GAUSS = M.ModelSpec(free=(), fixed_shape=M.GAUSS, method="exact")
SPEC_SGT = {"z1": M.ModelSpec(free=("beta", "xi"), fixed_shape=(2.0, 0.0, 1.0), method="fast"),
            "zf": M.ModelSpec(free=("beta", "xi", "zeta"), method="fast")}


def record(r, spec):
    """The fit's numbers kept in the JSON (natural parameters, Block 30's SEs, the conditional SE of d)."""
    return dict(phi=[float(x) for x in r["phi"]], se=[float(x) for x in r["se"]], se_d_cond=float(r["se_d_cond"]),
                chi2=float(r["chi2"]), n_res=int(r["n_res"]), status=r["status"], at_bound=list(r["at_bound"]),
                nfev=int(r["nfev"]), njev=int(r["njev"]), seconds=float(r["seconds"]), names=list(spec.names))


def summarise(recs, phi_t, spec, n_planes):
    names = spec.names
    p = len(names) + n_planes                      # fitted parameters: the model's and one scale per plane
    truth = {"d": phi_t[0], "mu": phi_t[1], "v0": phi_t[2], "z_ax": phi_t[3], "beta": phi_t[4], "xi": phi_t[5],
             "zeta": phi_t[6]}
    ok = [r for r in recs if r["status"] != "failed"]
    out = dict(n=len(recs), n_failed=len(recs) - len(ok))
    if not ok:
        return out
    d_hat = np.array([r["phi"][0] for r in ok])
    se_d = np.array([r["se"][0] for r in ok])
    err = d_hat - truth["d"]
    out.update(d_err_mean=float(err.mean()), d_err_sd=float(err.std(ddof=1)) if err.size > 1 else float("nan"),
               d_rmse=float(np.sqrt(np.mean(err ** 2))), d_err_mean_rel=float(err.mean() / truth["d"]),
               d_rmse_rel=float(np.sqrt(np.mean(err ** 2)) / truth["d"]),
               se_d_median=float(np.nanmedian(se_d)))
    fin = np.isfinite(se_d)
    out["cover_1se"] = float(np.mean(np.abs(err[fin]) <= se_d[fin])) if fin.any() else float("nan")
    out["cover_196se"] = float(np.mean(np.abs(err[fin]) <= 1.96 * se_d[fin])) if fin.any() else float("nan")
    out["n_se_finite"] = int(fin.sum())
    se_c = np.array([r["se_d_cond"] for r in ok])
    fc = np.isfinite(se_c)
    out["cover_1se_cond"] = float(np.mean(np.abs(err[fc]) <= se_c[fc])) if fc.any() else float("nan")
    out["cover_196se_cond"] = float(np.mean(np.abs(err[fc]) <= 1.96 * se_c[fc])) if fc.any() else float("nan")
    # the spread of d_hat against the median SE: does the reported SE describe the scatter?
    out["sd_over_se"] = float(out["d_err_sd"] / out["se_d_median"]) if out["se_d_median"] > 0 else float("nan")
    for i, n in enumerate(names):
        if i < 4 and n != "d":
            e = np.array([r["phi"][i] for r in ok]) - truth[n]
            out["%s_err_mean" % n] = float(e.mean())
            out["%s_err_sd" % n] = float(e.std(ddof=1)) if e.size > 1 else float("nan")
        if i >= 4:
            vals = np.array([r["phi"][i] for r in ok])
            e = vals - truth[n]
            out["%s_err_mean" % n] = float(e.mean())
            out["%s_err_sd" % n] = float(e.std(ddof=1)) if e.size > 1 else float("nan")
            out["%s_median" % n] = float(np.median(vals))
            ses = np.array([r["se"][i] for r in ok])
            out["%s_se_median" % n] = float(np.nanmedian(ses)) if np.isfinite(ses).any() else float("nan")
    bounds = {}
    for r in ok:
        for b in r["at_bound"]:
            bounds[b] = bounds.get(b, 0) + 1
    out["share_any_bound"] = float(np.mean([len(r["at_bound"]) > 0 for r in ok]))
    out["bounds"] = {k: v / len(ok) for k, v in sorted(bounds.items())}
    out["share_status"] = {s: float(np.mean([r["status"] == s for r in recs])) for s in FT_STATUSES}
    out["chi2_per_dof_mean"] = float(np.mean([r["chi2"] / (r["n_res"] - p) for r in ok]))
    out["nfev_mean"] = float(np.mean([r["nfev"] for r in recs]))
    out["njev_mean"] = float(np.mean([r["njev"] for r in recs]))
    out["seconds_mean"] = float(np.mean([r["seconds"] for r in recs]))
    return out


FT_STATUSES = ("converged", "at_bound", "failed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=float, required=True)
    ap.add_argument("--mu", type=float, default=None)
    ap.add_argument("--z", type=float, default=None)
    ap.add_argument("--truths", default=",".join(C.TRUTH_ORDER))
    ap.add_argument("--n-sets", type=int, default=30)
    ap.add_argument("--n-robust", type=int, default=2)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    st = M.make_setup()
    out_path = M.HERE / (a.out or ("recovery_d%.1f.json" % a.d))
    truths = [t for t in a.truths.split(",") if t]
    geoms = [g for g in C.geometries() if abs(g["d"] - a.d) < 1e-9 and (a.mu is None or abs(g["mu"] - a.mu) < 1e-9)
             and (a.z is None or abs(g["z_ax"] - a.z) < 1e-9)]
    results = dict(meta=dict(B_gl=st.B_gl, sigma_gl=st.sigma_gl, n_sets=a.n_sets, n_robust=a.n_robust,
                             seed=C.SEED, started=time.strftime("%Y-%m-%d %H:%M:%S")), cases=[])
    t_all = time.perf_counter()
    for g in geoms:
        for truth in truths:
            t_case = time.perf_counter()
            phi_t = C.phi_true(g, truth)
            truth_spec = M.ModelSpec(free=("beta", "xi", "zeta"), method="exact")
            t_true = M.tau_phi(phi_t, truth_spec, st)
            planes = node_inference.select_planes(t_true, st.B_gl, st.sigma_gl, st.cfg)
            y = st.B_gl * t_true
            ci = C.case_index(g, truth)
            recs = {"gauss": []}
            for m in C.models_for(truth):
                recs[m] = []
            robust = []
            for s in range(a.n_sets):
                rng = np.random.default_rng([C.SEED, ci, s])
                I = y + st.sigma_gl * rng.standard_normal(y.shape)
                rg = FT.fit(I, SPEC_GAUSS, st, phi_t[:4], planes)
                recs["gauss"].append(record(rg, SPEC_GAUSS))
                geo0 = rg["phi"][:4] if rg["status"] != "failed" else phi_t[:4]
                for m in C.models_for(truth):
                    spec = SPEC_SGT[m]
                    start = np.r_[geo0, 2.0, 0.0] if m == "z1" else np.r_[geo0, 2.0, 0.0, 1.0]
                    r = FT.fit(I, spec, st, start, planes)
                    recs[m].append(record(r, spec))
                    if s < a.n_robust:
                        for shp in ROBUST_SHAPES:
                            st0 = np.r_[geo0, shp[0], shp[1]] if m == "z1" else np.r_[geo0, list(shp)]
                            ra = FT.fit(I, spec, st, st0, planes)
                            robust.append(dict(set=s, model=m, start=list(shp), chi2_warm=float(r["chi2"]),
                                               chi2_alt=float(ra["chi2"]), d_warm=float(r["phi"][0]),
                                               d_alt=float(ra["phi"][0]), status_alt=ra["status"],
                                               shape_alt=[float(x) for x in ra["phi"][4:]]))
            summ = {m: summarise(recs[m], phi_t, SPEC_GAUSS if m == "gauss" else SPEC_SGT[m], int(planes.sum()))
                    for m in recs}
            better = [rb for rb in robust if rb["chi2_alt"] < rb["chi2_warm"] - 1e-6 * max(1.0, rb["chi2_warm"])]
            rob = dict(n=len(robust), n_better=len(better),
                       max_abs_d_diff_better=float(max([abs(rb["d_alt"] - rb["d_warm"]) for rb in better], default=0.0)),
                       max_chi2_gain=float(max([rb["chi2_warm"] - rb["chi2_alt"] for rb in better], default=0.0)))
            results["cases"].append(dict(geom=C.geom_id(g), d=g["d"], mu=g["mu"], z_ax=g["z_ax"], v0=g["v0"],
                                         truth=truth, shape=list(C.TRUTHS[truth]), n_planes=int(planes.sum()),
                                         summary=summ, robust=rob, robust_rows=robust, fits=recs,
                                         seconds=time.perf_counter() - t_case))
            line = "; ".join("%s d err %+.4f (rmse %.4f) cover %.2f/%.2f bound %.2f nfev %.0f %.2fs"
                             % (m, summ[m].get("d_err_mean", np.nan), summ[m].get("d_rmse", np.nan),
                                summ[m].get("cover_1se", np.nan), summ[m].get("cover_196se", np.nan),
                                summ[m].get("share_any_bound", np.nan), summ[m].get("nfev_mean", np.nan),
                                summ[m].get("seconds_mean", np.nan)) for m in recs)
            print("%s %s [%d planes] %s | robust %d/%d better | %.0f s"
                  % (C.geom_id(g), truth, int(planes.sum()), line, rob["n_better"], rob["n"],
                     time.perf_counter() - t_case))
            sys.stdout.flush()
            results["meta"]["seconds"] = time.perf_counter() - t_all
            out_path.write_text(json.dumps(results))
    results["meta"]["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out_path.write_text(json.dumps(results))
    print("wrote %s (%.0f s)" % (out_path.name, time.perf_counter() - t_all))


if __name__ == "__main__":
    main()
