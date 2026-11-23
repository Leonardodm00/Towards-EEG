#!/usr/bin/env python3
"""Standalone check (NOT pipeline code), 2026-11-23 (D-222): the compact tables of the README, generated from
validation.json, fisher_bias.json and recovery_d*.json (no number is typed by hand). Each cell of a compact
table spans the four geometries of one diameter (mu 0.5, 1.5 /um x the axis on k* and 0.5 um off): "a..b" is
their range.

Run:  python3 tables.py > tables.md
Pure ASCII, LF.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

import cases as C

HERE = Path(__file__).resolve().parent


def rng(vals, fmt="%.2f"):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    if not v:
        return "n/a"
    lo, hi = min(v), max(v)
    return (fmt % lo) if fmt % lo == fmt % hi else ("%s..%s" % (fmt % lo, fmt % hi))


def amax(vals, fmt="%.2f"):
    v = [abs(x) for x in vals if x is not None and np.isfinite(x)]
    return fmt % max(v) if v else "n/a"


def table_validation(val):
    print("### Validation (validate_model.py)\n")
    v1 = val["v1"]
    print("| check | result |")
    print("|---|---|")
    print("| V1: 1-D model vs Block 35 `line_profiles` (partition, Gaussian), 12 tubes | max abs diff in tau %.1e "
          "(min tau of the cases %.3f) |" % (max(r["max_abs_diff"] for r in v1), min(r["min_tau"] for r in v1)))
    print("| V2: SGT CDF vs quad of `blur_kernels.sgt_kernel_pdf`, 11 shapes x 3 widths | max abs %.1e |"
          % val["v2"]["worst_cdf_vs_quad"])
    print("| V2: dF/dt vs the density (central difference) | max rel %.1e |"
          % max(r["max_rel_dcdf_vs_pdf"] for r in val["v2"]["rows"]))
    print("| V2: SGT (2, 0, 1) vs ndtr; xi = 1e-10 vs xi = 0 | %.1e; %.1e |"
          % (val["v2"]["gauss_member_vs_ndtr"], val["v2"]["xi_1e10_vs_0"]))
    print("| V3: fast (Hermite) H vs exact: CDF; model tau over the 64 truth cases | %.1e; %.1e |"
          % (max(r["max_abs_cdf"] for r in val["v3"]["cdf"]), val["v3"]["model_max_abs"]))
    print("| V4: grouped FFT correlation vs the per-pair sum | %.1e |" % val["v4"]["max_abs"])
    print()


def table_a(fb):
    print("### (a) Fisher information at the truth\n")
    print("Per diameter, over its four geometries. SE(d)/d in %; infl = SE(d) / SE(d | shape known); "
          "|corr| the largest; cond = condition number of the correlation matrix; w = Block 39's centre weights "
          "(sandwich); bk = backgrounds known (Block 30's fit_node) instead of solved.\n")
    print("| truth | model | d (um) | SE(d)/d % | infl | infl w | infl bk | max abs corr(d,beta) | (d,xi) | (d,zeta) "
          "| SE(beta) | SE(xi) | SE(zeta) | cond |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for truth in C.TRUTH_ORDER:
        for model in ("known",) + C.models_for(truth):
            for d in C.D_LIST:
                rr = [r for r in fb["a"] if r["truth"] == truth and r["model"] == model and r["d"] == d]
                print("| %s | %s | %.1f | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                    truth, model, d, rng([100 * r["se_d"] / d for r in rr]), rng([r["infl_d"] for r in rr]),
                    rng([r["w_infl_d"] for r in rr]), rng([r.get("bk_infl_d", np.nan) for r in rr]),
                    amax([r["corr_d_beta"] for r in rr]), amax([r["corr_d_xi"] for r in rr]),
                    amax([r["corr_d_zeta"] for r in rr]),
                    rng([r["se_beta"] for r in rr]), rng([r["se_xi"] for r in rr]),
                    rng([r["se_zeta"] for r in rr], "%.3f"), rng([r["cond_corr"] for r in rr], "%.0f")))
    print()
    if fb.get("a_step_check") and "factor" in fb["a_step_check"][0]:
        print("Step check (every finite-difference step x 0.5 and x 2; heavy truth, axis 0.5 um off, zf model):\n")
        print("| geometry | step factor | SE(d) rel change | SE(v0) rel change | shape SEs, max rel change | "
              "inflation (ref -> step) | corr(d, shape) max abs change |")
        print("|---|---|---|---|---|---|---|")
        for c in fb["a_step_check"]:
            print("| %s | %.1f | %+.3f | %+.3f | %.1e | %.3f -> %.3f | %.3f |" % (
                c["geom"], c["factor"], c["rel_se_d"], c["rel_se_v0"], c["rel_se_shape_max"], c["infl_ref"],
                c["infl_step"], c["max_abs_corr_d_shape_change"]))
        print()


def table_b(fb):
    print("### (b) Bias in d when the truth is an SGT and the fit assumes the Gaussian kernel (noise-free)\n")
    print("Per diameter, over its four geometries. fit = the actual least-squares fit; lin = the first Gauss-Newton "
          "step from the truth; bias/SE in units of the Gaussian model's SE(d) at the truth; chi2 = the fit's "
          "noise-free chi2 in noise units (the misfit's visibility: 477 residuals, 4 + 9 fitted parameters, 464 degrees of freedom, so a chi2 SD of 30).\n")
    print("| truth | fit model | d (um) | bias fit % | bias lin % | bias/SE | chi2 noise-free |")
    print("|---|---|---|---|---|---|---|")
    for truth in ("heavy", "b13", "skew"):
        for model in (("gauss", "z1") if truth == "skew" else ("gauss",)):
            for d in C.D_LIST:
                rr = [r for r in fb["b"] if r["truth"] == truth and r["fit_model"] == model and r["d"] == d]
                print("| %s | %s | %.1f | %s | %s | %s | %s |" % (
                    truth, model, d, rng([100 * r["fit_bias_d_rel"] for r in rr], "%+.2f"),
                    rng([100 * r["lin_bias_d_rel"] for r in rr], "%+.2f"),
                    rng([r["bias_over_se"] for r in rr], "%+.1f"), rng([r["chi2_noise_free"] for r in rr], "%.0f")))
    print()


def load_rec():
    rec = {}
    for path in sorted(glob.glob(str(HERE / "recovery_d*.json"))):
        data = json.loads(Path(path).read_text())
        for c in data["cases"]:
            c["_n_sets"] = data["meta"]["n_sets"]
            rec[(c["geom"], c["truth"])] = c
    return rec


def table_c(rec):
    print("### (c) Recovery on noisy profiles (B 150 gl, noise 1.1 gl per line sample)\n")
    print("Per diameter, over its geometries run. err = mean (d_hat - d)/d %; rmse %; cover = share of "
          "|d_hat - d| <= SE (nominal 0.68) and <= 1.96 SE (0.95), SE by Block 30's rule; SD/SE = the scatter of "
          "d_hat over the median reported SE; bound = share of fits with any parameter at a bound (the bounds "
          "met); robust = data sets where one of Block 12's other start shapes found a lower chi2 / sets tried.\n")
    print("| truth | fit | d (um) | sets | err % | rmse % | cover 1 SE | cover 1.96 SE | SD/SE | bound | bounds met "
          "| median beta | median xi | median zeta | SD beta | SD xi | SD zeta | robust | chi2/dof |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for truth in C.TRUTH_ORDER:
        for model in ("gauss",) + C.models_for(truth):
            for d in C.D_LIST:
                cs = [c for (gid, t), c in rec.items() if t == truth and c["d"] == d]
                ss = [c["summary"][model] for c in cs if model in c["summary"] and "d_err_mean" in c["summary"][model]]
                if not ss:
                    continue
                nsets = sorted(set(c["_n_sets"] for c in cs))
                bounds = {}
                for s in ss:
                    for k, v in s["bounds"].items():
                        bounds[k] = bounds.get(k, 0.0) + v / len(ss)
                bstr = ", ".join("%s %.2f" % (k, v) for k, v in sorted(bounds.items(), key=lambda kv: -kv[1])[:3])
                rob = [rb for c in cs for rb in c["robust_rows"] if rb["model"] == model]
                better = [rb for rb in rob if rb["chi2_alt"] < rb["chi2_warm"] - 1e-6 * max(1.0, rb["chi2_warm"])]
                rstr = ("%d/%d" % (len(better), len(rob))) if rob else "-"
                if better:
                    rstr += " (max chi2 gain %.2f, max abs d change %.4f um)" % (
                        max(rb["chi2_warm"] - rb["chi2_alt"] for rb in better),
                        max(abs(rb["d_alt"] - rb["d_warm"]) for rb in better))
                print("| %s | %s | %.1f | %s x %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s "
                      "| %s |" % (
                          truth, model, d, "/".join(str(n) for n in nsets), len(ss),
                          rng([100 * s["d_err_mean_rel"] for s in ss], "%+.2f"),
                          rng([100 * s["d_rmse_rel"] for s in ss]),
                          rng([s["cover_1se"] for s in ss]), rng([s["cover_196se"] for s in ss]),
                          rng([s["sd_over_se"] for s in ss]), rng([s["share_any_bound"] for s in ss]), bstr or "-",
                          rng([s.get("beta_median", np.nan) for s in ss]),
                          rng([s.get("xi_median", np.nan) for s in ss]),
                          rng([s.get("zeta_median", np.nan) for s in ss], "%.3f"),
                          rng([s.get("beta_err_sd", np.nan) for s in ss]),
                          rng([s.get("xi_err_sd", np.nan) for s in ss]),
                          rng([s.get("zeta_err_sd", np.nan) for s in ss], "%.3f"), rstr,
                          rng([s["chi2_per_dof_mean"] for s in ss])))
    print()
    # pooled coverage per model (all cases run): the binomial SE of a share over N sets is about 0.47 / sqrt(N)
    print("Pooled over every case run (each data set counted once):\n")
    print("| fit | truth | data sets | failed | SE(d) not finite | cover 1 SE | cover 1.96 SE | share at a bound |")
    print("|---|---|---|---|---|---|---|---|")
    for truth in C.TRUTH_ORDER:
        for model in ("gauss",) + C.models_for(truth):
            n, c1, c2, nb, nf, nn, tot = 0, 0, 0, 0, 0, 0, 0
            for (gid, t), c in rec.items():
                if t != truth or model not in c["fits"]:
                    continue
                for f in c["fits"][model]:
                    tot += 1
                    if f["status"] == "failed":
                        nf += 1
                        continue
                    if not np.isfinite(f["se"][0]):
                        nn += 1
                        continue
                    e = abs(f["phi"][0] - c["d"])
                    n += 1
                    c1 += e <= f["se"][0]
                    c2 += e <= 1.96 * f["se"][0]
                    nb += len(f["at_bound"]) > 0
            if n:
                print("| %s | %s | %d | %d | %d | %.3f | %.3f | %.3f |" % (model, truth, tot, nf, nn, c1 / n, c2 / n,
                                                                         nb / n))
    print()


def table_d(val, rec):
    print("### (d) Cost\n")
    print("| d (um) | forward, Gaussian (ndtr) ms | SGT exact ms | SGT fast ms | fit: gauss s (nfev) | z1 s (nfev) | "
          "zf s (nfev) |")
    print("|---|---|---|---|---|---|---|")
    for d in C.D_LIST:
        t = {r["kernel"]: r["ms"] for r in val["v5"] if r["d"] == d}
        cs = [c for c in rec.values() if c["d"] == d]
        cells = []
        for m in ("gauss", "z1", "zf"):
            ss = [c["summary"][m] for c in cs if m in c["summary"] and "seconds_mean" in c["summary"][m]]
            cells.append("%.2f (%.0f)" % (np.mean([s["seconds_mean"] for s in ss]), np.mean([s["nfev_mean"] for s in ss]))
                         if ss else "n/a")
        print("| %.1f | %.1f | %.1f | %.1f | %s | %s | %s |" % (d, t.get("gauss ndtr", np.nan), t.get("sgt exact", np.nan),
                                                            t.get("sgt fast", np.nan), *cells))
    print()


def main():
    val = json.loads((HERE / "validation.json").read_text())
    fb = json.loads((HERE / "fisher_bias.json").read_text())
    rec = load_rec()
    table_validation(val)
    table_a(fb)
    table_b(fb)
    table_c(rec)
    table_d(val, rec)


if __name__ == "__main__":
    main()
