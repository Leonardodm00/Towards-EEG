#!/usr/bin/env python3
"""Standalone check (NOT pipeline code), 2026-11-23 (D-221, D-222): parts (a) and (b) of the SGT identifiability
check, on the 1-D model of sgt_line_model.py (exact CDF path).

(a) Fisher information at the truth, per geometry and kernel truth (cases.py), the per-plane backgrounds
    profiled out (Schur complement), noise SD setup.sigma_gl per line sample, the planes Block 30's
    node_inference.select_planes keeps at the truth. Models: "known" (d, mu, v0, z_ax with the true shape held:
    SE(d)'s reference), "z1" ((beta, xi) free, zeta = 1; symmetric truths) and "zf" ((beta, xi, zeta) free).
    Reported: the SE of every parameter, corr(d, beta), corr(d, xi), corr(d, zeta), the condition number of
    the correlation matrix, and SE(d)'s inflation over "known". Beside it, labelled, the same with Block 39's
    centre weights (w_n = exp(-|v_n - v0| / lambda), lambda = weight_kappa x allen_fwhm_um(d, sigma_r(0)) with
    Allen's diameter = the true d), the covariance being the sandwich of Block 30's amendment 2.
    A step check: the SEs and correlations with every finite-difference step halved, on 8 cases.
(b) The bias in d when the data come from an SGT truth and the fit assumes the Gaussian kernel: linearised (the
    first Gauss-Newton step from the truth, backgrounds free) and by an actual fit of the noise-free data
    (fitting.fit, the Gaussian model, started at the truth); the chi2 of that fit in noise units (the
    misspecification's visibility); the bias over the Gaussian model's SE(d) at the truth. For the skewed truth
    also the symmetric option (z1) fitted to it.

Run (a few minutes):
    cd "Passive Features/Diameter Re-measurement/checks/sgt_identifiability_2026-11-23"
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 fisher_bias.py > fisher_bias.out
Writes fisher_a.csv, bias_b.csv, fisher_bias.json. Pure ASCII, LF.
"""
from __future__ import annotations

import csv
import json
import sys
import time

import numpy as np

import cases as C
import fitting as FT
import sgt_line_model as M
from allen_diameter.analysis import line_weights, node_inference

SPECS = {"known": None, "gauss": M.ModelSpec(free=(), fixed_shape=M.GAUSS, method="exact"),
         "z1": M.ModelSpec(free=("beta", "xi"), fixed_shape=(2.0, 0.0, 1.0), method="exact"),
         "zf": M.ModelSpec(free=("beta", "xi", "zeta"), method="exact")}
ALL_NAMES = ("d", "mu", "v0", "z_ax", "beta", "xi", "zeta")


def spec_known(truth):
    return M.ModelSpec(free=(), fixed_shape=C.TRUTHS[truth], method="exact")


def planes_at_truth(st, phi, truth_spec):
    t0 = M.tau_phi(phi, truth_spec, st)
    return node_inference.select_planes(t0, st.B_gl, st.sigma_gl, st.cfg), t0


def weights_for(st, g):
    lam = float(st.cfg.inference.weight_kappa) * line_weights.allen_fwhm_um(g["d"], float(st.rcfg.sigma_r0_um))
    return line_weights.centre_weights(st.v, g["v0"], lam), lam


def row_from(fi, prefix=""):
    out = {}
    se = dict(zip(fi["names"], fi["se"]))
    ix = {n: i for i, n in enumerate(fi["names"])}
    for n in ALL_NAMES:
        out[prefix + "se_" + n] = float(se[n]) if n in se else float("nan")
    for n in ("mu", "v0", "z_ax", "beta", "xi", "zeta"):
        out[prefix + "corr_d_" + n] = float(fi["corr"][0, ix[n]]) if n in ix else float("nan")
    out[prefix + "cond_corr"] = float(fi["cond_corr"])
    return out


def part_a(st, step_check=True):
    rows = []
    t_start = time.perf_counter()
    for g in C.geometries():
        for truth in C.TRUTH_ORDER:
            phi_t = C.phi_true(g, truth)
            truth_spec = M.ModelSpec(free=("beta", "xi", "zeta"), method="exact")
            planes, _t0 = planes_at_truth(st, phi_t, truth_spec)
            w, lam = weights_for(st, g)
            ref = None
            for model in ("known",) + C.models_for(truth):
                spec = spec_known(truth) if model == "known" else SPECS[model]
                phi = phi_t[:4] if model == "known" else np.r_[phi_t[:4], [phi_t[4 + ALL_NAMES[4:].index(n)]
                                                                         for n in spec.free]]
                fi = M.fisher(phi, spec, st, planes=planes)
                fw = M.fisher(phi, spec, st, planes=planes, weights=w, jac=fi["jac_all"])
                row = dict(geom=C.geom_id(g), d=g["d"], mu=g["mu"], z_ax=g["z_ax"], truth=truth, model=model,
                           n_planes=int(planes.sum()), lambda_um=lam)
                row.update(row_from(fi))
                row.update(row_from(fw, prefix="w_"))
                row.update(row_from(fi["bk"], prefix="bk_"))
                if model == "known":
                    ref = (row["se_d"], row["w_se_d"], row["bk_se_d"])
                row["infl_d"] = row["se_d"] / ref[0]
                row["w_infl_d"] = row["w_se_d"] / ref[1]
                row["bk_infl_d"] = row["bk_se_d"] / ref[2]
                row["vif_d"] = float(fi["vif"][0])
                rows.append(row)
        print("(a) %s done (%.0f s)" % (C.geom_id(g), time.perf_counter() - t_start))
        sys.stdout.flush()
    checks = []
    if step_check:
        # every finite-difference step x 0.5 and x 2 (heavy truth, axis 0.5 um off, the 8 (d, mu)): the relative
        # change of SE(d), SE(v0), the largest over the shape's SEs, SE(d)'s inflation zf / known, and the
        # largest change of corr(d, shape)
        truth = "heavy"
        truth_spec = M.ModelSpec(free=("beta", "xi", "zeta"), method="exact")
        for g in [x for x in C.geometries() if x["z_ax"] == 0.5]:
            phi_t = C.phi_true(g, truth)
            planes, _ = planes_at_truth(st, phi_t, truth_spec)
            sk, sf = spec_known(truth), SPECS["zf"]
            ref_k = M.fisher(phi_t[:4], sk, st, planes=planes)
            ref_f = M.fisher(phi_t, sf, st, planes=planes)
            for fac in (0.5, 2.0):
                fk = M.fisher(phi_t[:4], sk, st, planes=planes, steps=fac * M.default_steps(sk))
                ff = M.fisher(phi_t, sf, st, planes=planes, steps=fac * M.default_steps(sf))
                rel = ff["se"] / ref_f["se"] - 1.0
                infl0 = ref_f["se"][0] / ref_k["se"][0]
                infl1 = ff["se"][0] / fk["se"][0]
                checks.append(dict(geom=C.geom_id(g), truth=truth, factor=fac, rel_se_d=float(rel[0]),
                                   rel_se_v0=float(rel[2]), rel_se_shape_max=float(np.max(np.abs(rel[4:]))),
                                   infl_ref=float(infl0), infl_step=float(infl1),
                                   max_abs_corr_d_shape_change=float(np.max(np.abs(ff["corr"][0, 4:]
                                                                                   - ref_f["corr"][0, 4:])))))
                c = checks[-1]
                print("(a) steps x%.1f %s: SE(d) %+.3f, SE(v0) %+.3f, shape SEs <= %.1e (relative); inflation "
                      "%.3f -> %.3f; corr(d, shape) moves <= %.3f" % (fac, c["geom"], c["rel_se_d"], c["rel_se_v0"],
                                                                     c["rel_se_shape_max"], c["infl_ref"],
                                                                     c["infl_step"], c["max_abs_corr_d_shape_change"]))
    return rows, checks


def part_b(st):
    rows = []
    gspec = SPECS["gauss"]
    z1 = SPECS["z1"]
    t_start = time.perf_counter()
    for g in C.geometries():
        for truth in C.TRUTH_ORDER:
            phi_t = C.phi_true(g, truth)
            truth_spec = M.ModelSpec(free=("beta", "xi", "zeta"), method="exact")
            planes, t_true = planes_at_truth(st, phi_t, truth_spec)
            y = st.B_gl * t_true
            fits = [("gauss", gspec, phi_t[:4])]
            if truth == "skew":
                fits.append(("z1", z1, np.r_[phi_t[:4], 2.0, 0.0]))
            for label, spec, phi0 in fits:
                if label == "gauss":
                    lin, _dB = FT.linearised_shift(y, spec, st, phi_t[:4], st.B_gl, planes)
                    se_d = float(M.fisher(phi_t[:4], spec, st, planes=planes)["se"][0])
                else:
                    # linearise about the symmetric option's nearest member: (beta, xi) of the truth, zeta = 1
                    phi_lin = np.r_[phi_t[:4], phi_t[4], phi_t[5]]
                    lin, _dB = FT.linearised_shift(y, spec, st, phi_lin, st.B_gl, planes)
                    se_d = float(M.fisher(phi_lin, spec, st, planes=planes)["se"][0])
                r = FT.fit(y, spec, st, phi0, planes)
                rows.append(dict(geom=C.geom_id(g), d=g["d"], mu=g["mu"], z_ax=g["z_ax"], truth=truth,
                                 fit_model=label, n_planes=int(planes.sum()),
                                 lin_bias_d=float(lin[0]), lin_bias_d_rel=float(lin[0] / g["d"]),
                                 lin_bias_mu=float(lin[1]), lin_bias_v0=float(lin[2]), lin_bias_z=float(lin[3]),
                                 fit_bias_d=float(r["phi"][0] - g["d"]),
                                 fit_bias_d_rel=float((r["phi"][0] - g["d"]) / g["d"]),
                                 fit_bias_mu=float(r["phi"][1] - g["mu"]), fit_bias_v0=float(r["phi"][2] - g["v0"]),
                                 fit_bias_z=float(r["phi"][3] - g["z_ax"]),
                                 fit_shape=[float(x) for x in r["phi"][4:]],
                                 chi2_noise_free=r["chi2"], n_res=r["n_res"], fit_status=r["status"],
                                 fit_at_bound=list(r["at_bound"]), se_d_model=se_d,
                                 bias_over_se=float((r["phi"][0] - g["d"]) / se_d)))
        print("(b) %s done (%.0f s)" % (C.geom_id(g), time.perf_counter() - t_start))
        sys.stdout.flush()
    return rows


def write_csv(path, rows):
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (";".join(str(x) for x in v) if isinstance(v, (list, tuple)) else v)
                         for k, v in r.items()})


def main():
    st = M.make_setup()
    print("setup: B %.0f gl, sigma %.2f gl per line sample, planes %d..%d, n_v %d"
          % (st.B_gl, st.sigma_gl, st.ks.min(), st.ks.max(), st.v.size))
    a_rows, checks = part_a(st)
    write_csv(M.HERE / "fisher_a.csv", a_rows)
    b_rows = part_b(st)
    write_csv(M.HERE / "bias_b.csv", b_rows)
    (M.HERE / "fisher_bias.json").write_text(json.dumps(dict(a=a_rows, a_step_check=checks, b=b_rows,
                                                             B_gl=st.B_gl, sigma_gl=st.sigma_gl), indent=1))
    print("wrote fisher_a.csv, bias_b.csv, fisher_bias.json")


if __name__ == "__main__":
    main()
