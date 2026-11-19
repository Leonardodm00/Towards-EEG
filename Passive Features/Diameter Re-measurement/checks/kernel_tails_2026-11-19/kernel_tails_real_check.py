#!/usr/bin/env python3
"""Standalone check (NOT pipeline code), 2026-11-19: on a REAL clear node, which change of the blur kernel lets
Allen's diameter fit the recorded profiles -- heavier tails at the configured in-focus width, or a wider in-focus
width (core)?

Data: a run of clean_cell.py clean --no-clean (its cleaning_<S>.json and cleaning_<S>_node<id>.npz) and the fit
of that run (inference_<S>.json), for the start, the planes and the weights' width actually used.
Model: the pipeline's own forward (node_inputs.forward_factory -> line_fast.line_profiles, hybrid absorption,
the run's rays), one tube, theta = (d, mu, v0, z_ax), the residual weights exp(-|v - v_c| / lambda) re-centred
on the fitted centre (node_inference.fit_node_recentred), lambda FIXED at the run's value for every variant so
that the weighted chi2/N are comparable.
Kernels (the image is linear in the kernel, so a mixture of two renderings is the rendering of a mixture kernel):
  T(s0)          the run's Gaussian table re-anchored at the in-focus SD s0 (kernel.with_in_focus_width)
  halo(f, h)     (1 - f) T(0.080) + f T_h, T_h the table with every SD (and the slope) multiplied by h
Each kernel is fitted with d free and with d held at Allen's diameter (bounds +-0.0005 um).
With --unweighted every line sample has weight 1 (fit_node), the same planes.
Run:  python3 -I kernel_tails_real_check.py <code folder> <cleaning dir> <inference dir> <node id> [variants] [--unweighted]
"""
import dataclasses
import json
import math
import os
import sys
import time
import types

import numpy as np

WS = os.path.abspath(sys.argv[1])
sys.path.insert(0, os.path.join(WS, "src"))
sys.path.insert(0, os.path.join(WS, "scripts"))
import infer_nodes as INF                                              # noqa: E402
from allen_diameter.analysis import node_inference as NI, node_inputs  # noqa: E402
from allen_diameter.model import kernel, line_fast                     # noqa: E402

CLEAN_DIR, INFER_DIR, NID = sys.argv[2], sys.argv[3], int(sys.argv[4])
UNWEIGHTED = "--unweighted" in sys.argv[5:]
ARGS = [a for a in sys.argv[5:] if a != "--unweighted"]
WANT = ARGS[0].split(",") if ARGS else None
SPEC = "529878215"
HALF_FIXED = 0.0005                    # um: d held at Allen's within +-this


def load():
    with open(os.path.join(CLEAN_DIR, "cleaning_%s.json" % SPEC)) as f:
        rec = json.load(f)
    with open(os.path.join(INFER_DIR, "inference_%s.json" % SPEC)) as f:
        inf = json.load(f)
    info, run = rec["nodes"][str(NID)], inf["nodes"][str(NID)]
    cfg = INF.run_config(rec, "", "")
    pil = info["pilot"]
    pilot = types.SimpleNamespace(cx_um=float(pil["cx_um"]), cy_um=float(pil["cy_um"]),
                                  theta_rad=float(pil["theta_rad"]), phi_rad=float(pil["phi_rad"]),
                                  z_sub_um=float(pil["z_sub_um"]), d_hat_um=float(pil["d_hat_um"]),
                                  mu_hat_per_um=float(pil["mu_hat_per_um"]))
    with np.load(os.path.join(CLEAN_DIR, "cleaning_%s_node%d.npz" % (SPEC, NID)), allow_pickle=False) as z:
        node = INF.node_from_npz(NID, z, None)
        prof = np.array(z["profiles_after"], dtype=float)
    return rec, info, run, cfg, pilot, node, prof


def renderer_scaled(ren, h):
    return dataclasses.replace(ren, sigma_r0_um=h * ren.sigma_r0_um,
                               kernel_table_sigma_um=tuple(h * s for s in ren.kernel_table_sigma_um),
                               kernel_continuation_slope=h * ren.kernel_continuation_slope)


def main():
    rec, info, run, cfg, pilot, node, prof = load()
    d_allen = float(info["d_allen_um"])
    sigma = float(info["sigma_gl"])
    U = tuple(float(x) for x in info["U_axis_um"])
    theta0 = np.array(run["theta0"], dtype=float)
    lam = float(run["lambda_um"])
    planes = np.array(run["planes"], dtype=bool) if "planes" in run else None
    v_t = node_inputs.trace_offset(pilot, node.node_xyz)
    print("node %d: Allen %.4f um; run: d %.4f um, chi2/N %s; lambda %.4f um (fixed below); sigma %.3f gl; "
          "rays %d x %d, fine %d" % (NID, d_allen, run["theta"][0], "n/a", lam, sigma, cfg.inference.ray_n_rho,
                                     cfg.inference.ray_n_psi, cfg.inference.line_fine_factor), flush=True)

    def fwd_for(ren):
        c = dataclasses.replace(cfg, renderer=ren)
        return node_inputs.forward_factory(pilot, U, node, c, line_fast.SectionCache())

    def mixture(f_main, f_halo, f):
        def fwd(d, mu, v0, z):
            return (1.0 - f) * f_main(d, mu, v0, z) + f * f_halo(d, mu, v0, z)
        return fwd

    def fit(fwd, fixed):
        icfg = cfg.inference
        if fixed:
            icfg = dataclasses.replace(icfg, d_bounds_um=(d_allen - HALF_FIXED, d_allen + HALF_FIXED),
                                       multistart_factors=(1.0,))
            th0 = theta0.copy()
            th0[0] = d_allen
        else:
            icfg = dataclasses.replace(icfg, multistart_factors=(1.0, 1.6))
            th0 = theta0.copy()
        c = dataclasses.replace(cfg, inference=icfg)
        t0 = time.time()
        if UNWEIGHTED:                 # every line sample at weight 1, the same planes
            res = NI.fit_node(prof, node.B_gl, sigma, fwd, th0, c, planes=planes)
            inf_ = dict(n_solves=1)
        else:
            res, inf_ = NI.fit_node_recentred(prof, node.B_gl, sigma, fwd, th0, c, node.v, lam,
                                              centre_offset_um=-v_t, keep=None, planes=planes)
        chi = res.chi2_per_plane
        used = np.isfinite(chi)
        return dict(d=res.theta[0], mu=res.theta[1], v0=res.theta[2], z=res.theta[3], status=res.status,
                    chi2N=float(np.mean(chi[used])), cost=float(res.cost), solves=inf_["n_solves"],
                    sec=time.time() - t0, se_d=float(res.se[0]) if np.isfinite(res.se[0]) else float("nan"))

    ren0 = cfg.renderer
    variants = [("gauss s0 0.080", lambda: fwd_for(ren0))]
    for s0 in (0.15, 0.25):
        variants.append(("gauss s0 %.3f" % s0, (lambda s=s0: fwd_for(kernel.with_in_focus_width(ren0, s)))))
    for f, h in ((0.3, 3.0), (0.3, 6.0), (0.5, 6.0)):
        variants.append(("halo f %.1f h %.0f (core 0.080)" % (f, h),
                         (lambda f=f, h=h: mixture(fwd_for(ren0), fwd_for(renderer_scaled(ren0, h)), f))))
    for name, make in variants:
        for fixed in (False, True):
            tag = "%s, d %s" % (name, "= Allen" if fixed else "free")
            if WANT and not any(w in tag for w in WANT):
                continue
            r = fit(make(), fixed)
            print("%-44s d %.4f (se %.4f) mu %.3f v0 %+.3f z %.3f  chi2/N %.3f  cost %.1f  %s, %d solve(s), %.0f s"
                  % (tag, r["d"], r["se_d"], r["mu"], r["v0"], r["z"], r["chi2N"], r["cost"], r["status"],
                     r["solves"], r["sec"]), flush=True)


if __name__ == "__main__":
    main()
