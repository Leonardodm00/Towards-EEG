#!/usr/bin/env python3
"""Standalone check (NOT pipeline code), 2026-11-20: the identifiability check of SPEC Block 42 (D-201's
precondition; the clear-node handoff's section 11.3 target, sigma_r(0) to about +-0.02 um). Reported, not
pass/fail.

All with the production hybrid of the per-node fit (inference.absorption "hybrid", 8 x 16 directions, the
configuration's defaults otherwise), through Block 42's own code (sigma_calibration: forward_with_kernel,
start_theta, fit_sigma_node, and its Jacobian and covariance helpers), on the synthetic node of the smoke test
(a flat tube along x traced at Allen's diameter d, the node at the trace's middle, measure_pilot and build_node),
prepared as notebook Cell 23A prepares the thick nodes (D-201.6): the planes k* - 2 .. k* + 2 and the measuring
line +-4.5 um. The geometry's image is a noise-free partition_vertical rendering (Block 4); only the node's
pilot, planes and line are taken from it. Profiles (1)-(2) are the forward's own output on a straight
background b0 + b1 v, b0 = 150 gl, b1 = -2 gl/um (the T3 values of the smoke test; the pilot cell's 5353 sits on
about 140 gl).

  (1) For d in {1.0, 1.5, 2.5} um and sigma0 in {0.10, 0.20} um, at theta_true = (d, 0.25 /um, v0_start + 0.04 um,
      z_start + 0.1 um, sigma0, 1): the SEs from J at the truth -- the fit's own forward-difference Jacobian of the
      projected residual (Golub-Pereyra) of noise-free profiles, residuals in units of 1.1 gl, Cov = (J^T J)^-1 --
      and the correlations of sigma0 with d and with c (and, beside them, of d with mu and with c).
  (2) One noisy fit per d at sigma0 0.15 um (c 1, mu 0.25 /um), Gaussian noise 1.1 gl per sample, from
      start_theta (its rule of amendment 4, D-203: the k* dip against the straight line through |v| >= 3 um):
      its errors, SE_s, number of forward evaluations (the Jacobian's included) and seconds.
  (3) The cross-renderer case: a noise-free block of a 1.5 um tube (mu 0.25 /um, on the trace, axis at z 0.05 um)
      rendered by the 2-D renderer render.render_transmittance with absorption "hybrid" at 8 x 16 directions (the
      line renderer's) and the kernel kernel.with_kernel(renderer, 0.15 um, 1.3), 210 gl x tau; the node prepared
      as test_noise_free_fit does (measure_pilot, build_node, the profiles read from the block by
      clean_iteration._profiles) and fitted by Block 42 from start_theta (sigma 1.1 gl nominal): the errors of
      every parameter, sigma0_hat, d_hat and c_hat first -- whether the hybrid line renderer's own approximations
      bias sigma0_hat. Beside it, the two renderers' profiles at the truth before any fit.

Run (about 20 to 40 minutes on one core):
    cd "Passive Features/Diameter Re-measurement"
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 checks/sigma_calibration_2026-11-20/identifiability.py \
        > checks/sigma_calibration_2026-11-20/identifiability.out
    (--parts 1,2,3 to run some parts only.)

Pure ASCII, LF.
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.metadata
import math
import os
import platform
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
for _p in (WS / "src", WS / "scripts", WS / "tests" / "smoke"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import test_smoke_pilot_run as PR  # noqa: E402  (write_swc, block_provider, node_of, fscene_of, FAST)
from allen_diameter.analysis import cell, clean_iteration  # noqa: E402
from allen_diameter.analysis import node_inputs as NI  # noqa: E402
from allen_diameter.analysis import sigma_calibration as SC  # noqa: E402
from allen_diameter.analysis.node_pipeline import measure_pilot  # noqa: E402
from allen_diameter.config import default_config  # noqa: E402
from allen_diameter.model import geometry, kernel, line_fast, render  # noqa: E402

SEED = 20261120
D_LIST = (1.0, 1.5, 2.5)          # um
S0_LIST = (0.10, 0.20)            # um, part (1)
S0_FIT = 0.15                     # um, part (2)
MU = 0.25                         # 1/um
C_TRUE = 1.0                      # parts (1), (2)
NOISE = 1.1                       # gl per sample
B0, B1 = 150.0, -2.0              # gl, gl/um
DV0, DZ = 0.04, 0.1               # um: the truth's axis off the start (as the smoke test's T3)
CROSS = dict(d=1.5, mu=0.25, sigma0=0.15, c=1.3)
AXIS_Z = 0.05                     # um: the rendered axis' depth (on the trace)


def prod_cfg():
    """The defaults (inference: hybrid, 8 x 16), the smoke tests' cheap shadow settings for the background of
    build_node (FAST's cleaning section: it does not enter the fit), Cell 23A's planes k* +- 2 and line +-4.5 um."""
    cfg = default_config()
    cl = dataclasses.replace(cfg.cleaning, **{k: (tuple(v) if isinstance(v, list) else v)
                                              for k, v in PR.FAST["cleaning"].items()})
    cfg = dataclasses.replace(cfg, cleaning=dataclasses.replace(cl, plane_half_range=2),
                              measure=dataclasses.replace(cfg.measure, profile_half_um=4.5))
    cfg.validate()
    assert cfg.inference.absorption == "hybrid" and (cfg.inference.ray_n_rho, cfg.inference.ray_n_psi) == (8, 16)
    return cfg


def build(cfg, d, mu, rcfg_render):
    """(pilot, node, info, prof_gl, seconds): a flat tube of diameter d along x at y 0, z AXIS_Z, half-length 14 um,
    rendered noise-free by render.render_transmittance under rcfg_render (210 gl x tau), traced at Allen's
    diameter d; the pilot, the NodeData (5 planes) and the block's profiles on the node's line."""
    t0 = time.time()
    res0, dz = cfg.acquisition.res0_um, cfg.acquisition.dz_um
    tmp = tempfile.mkdtemp(prefix="b42_ident_")
    swc = PR.write_swc(os.path.join(tmp, "one.swc"), [((0.0, 0.0, AXIS_Z), (20.0, 0.0, AXIS_Z), 11, 0.5 * d)])
    tube = geometry.Tube((10.0, 0.0, AXIS_Z), 0.5 * d, 0.0, 0.0, 1.0, 14.0, "axial")
    left, top = int(math.floor(-0.5 / res0)), int(math.floor(-10.5 / res0))
    width = height = int(math.ceil(21.0 / res0))
    ks = np.arange(-6, 7)
    f = render.fine_factor(d, rcfg_render, res0)
    grid, inner = render.block_fine_grid(left, top, width, height, res0, f, 8)
    out = render.render_transmittance(tube, mu, ks * dz, grid, rcfg_render, cfg.acquisition.light_direction,
                                      reduce=lambda pl: render.pixel_integrate(pl[inner[0], inner[1]], f))
    block = rcfg_render.background_B_gl * np.asarray(out.tau, dtype=float)
    provider = PR.block_provider(block, ks, left, top, res0)
    full, i_full, ob, t = PR.node_of(swc, cfg, 7)
    pilot = measure_pilot(full, i_full, provider, cfg)
    scene = cell.cell_scene(swc, (2, 3, 4), res0)
    node, info = NI.build_node(ob, t, pilot, provider, scene, cfg, NI.own_ids_within(ob, t, 5.0)[0],
                               PR.fscene_of(swc, scene, d=d))
    if node is None:
        raise RuntimeError("build_node refused the node: %r" % (info,))
    prof = clean_iteration._profiles(np.asarray(node.block, dtype=float), node, cfg)
    return pilot, node, info, prof, time.time() - t0


def k_index_of(node, pilot):
    return int(np.flatnonzero(np.asarray(node.ks) == int(pilot.k_star))[0])


def describe(node, pilot, info, seconds, label):
    print("  %s: pilot k* %d, z_sub %.4f um, cx %.4f cy %.4f um, theta %.2e phi %.2e rad; planes %s; %d samples "
          "(+-%.2f um); tube extents %s um; prepared in %.1f s"
          % (label, int(pilot.k_star), float(pilot.z_sub_um), float(pilot.cx_um), float(pilot.cy_um),
             float(pilot.theta_rad), float(pilot.phi_rad), list(int(k) for k in node.ks), len(node.v),
             float(np.max(np.abs(node.v))), tuple(round(float(x), 3) for x in info["U_axis_um"]), seconds))


def corr(cov, i, j):
    return float(cov[i, j] / math.sqrt(cov[i, i] * cov[j, j]))


def part1_and_2(cfg, parts, cache):
    scfg = cfg.sigma_calibration
    rows1, rows2 = [], []
    for n_d, d in enumerate(D_LIST):
        pilot, node, info, _prof, sec = build(cfg, d, MU, cfg.renderer)
        describe(node, pilot, info, sec, "d %.1f um" % d)
        f6 = SC.forward_with_kernel(pilot, info["U_axis_um"], node, cfg, cache)
        v = np.asarray(node.v, dtype=float)
        st = NI.pilot_start(pilot, node.node_xyz)
        v0s, zs = float(st[2]), float(node.node_xyz[2]) + float(st[3])
        k_index = k_index_of(node, pilot)
        if 1 in parts:
            for s0 in S0_LIST:
                th = np.array([d, MU, v0s + DV0, zs + DZ, s0, C_TRUE])
                t0 = time.time()
                I = (B0 + B1 * v)[None, :] * f6(*th)
                lo, hi = SC._bounds(scfg, th[2], th[3])
                prob = SC._Problem(I, NOISE, v, f6, hi, scfg.diff_steps)
                J = prob.jac(th)
                cov, how = SC._covariance(J)
                se = np.sqrt(np.diag(cov))
                rows1.append((d, s0, se, corr(cov, 4, 0), corr(cov, 4, 5), corr(cov, 0, 1), corr(cov, 0, 5), how,
                              prob.n_eval, time.time() - t0, float(np.min(I[k_index] / (B0 + B1 * v)))))
                print("    (1) d %.1f sigma0 %.2f: SE d %.4f mu %.5f v0 %.4f z %.4f sigma0 %.4f c %.4f; corr(s0,d) "
                      "%+.3f corr(s0,c) %+.3f; %d evaluations, %.1f s" % (d, s0, se[0], se[1], se[2], se[3], se[4],
                                                                          se[5], rows1[-1][3], rows1[-1][4],
                                                                          prob.n_eval, rows1[-1][9]))
                sys.stdout.flush()
        if 2 in parts:
            th = np.array([d, MU, v0s + DV0, zs + DZ, S0_FIT, C_TRUE])
            rng = np.random.default_rng([SEED, n_d])
            I = (B0 + B1 * v)[None, :] * f6(*th) + rng.normal(0.0, NOISE, (len(node.ks), v.size))
            th0 = SC.start_theta(I, np.full(len(node.ks), B0), v, k_index, pilot, node.node_xyz, scfg)
            fit = SC.fit_sigma_node(I, NOISE, v, f6, th0, scfg)
            rows2.append((d, th, th0, fit))
            err = fit.theta - th
            print("    (2) d %.1f: %s, %d evaluations, %.1f s, chi2/dof %.3f; errors d %+.4f sigma0 %+.4f c %+.4f; "
                  "SE_s %.4f %.4f %.4f" % (d, fit.status, fit.n_eval, fit.seconds, fit.chi2 / fit.dof, err[0],
                                           err[4], err[5], fit.se_scaled[0], fit.se_scaled[4], fit.se_scaled[5]))
            sys.stdout.flush()
    return rows1, rows2


def part3(cfg, cache):
    scfg = cfg.sigma_calibration
    rr = dataclasses.replace(cfg.renderer, absorption="hybrid", ray_n_rho=int(cfg.inference.ray_n_rho),
                             ray_n_psi=int(cfg.inference.ray_n_psi))
    rr = kernel.with_kernel(rr, CROSS["sigma0"], CROSS["c"])
    pilot, node, info, prof, sec = build(cfg, CROSS["d"], CROSS["mu"], rr)
    describe(node, pilot, info, sec, "cross-renderer, d %.1f um (2-D hybrid render included)" % CROSS["d"])
    f6 = SC.forward_with_kernel(pilot, info["U_axis_um"], node, cfg, cache)
    v = np.asarray(node.v, dtype=float)
    k_index = k_index_of(node, pilot)
    # the truth in the fit's parameters: v0 about the trace (the tube lies on it: 0), z_ax absolute
    th = np.array([CROSS["d"], CROSS["mu"], 0.0, AXIS_Z, CROSS["sigma0"], CROSS["c"]])
    line = rr.background_B_gl * f6(*th)
    diff = prof - line
    th0 = SC.start_theta(prof, node.B_gl, v, k_index, pilot, node.node_xyz, scfg)
    fit = SC.fit_sigma_node(prof, NOISE, v, f6, th0, scfg)
    return dict(node=node, pilot=pilot, th=th, th0=th0, fit=fit, diff=diff, line=line, prof=prof, B=node.B_gl,
                k_index=k_index)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--parts", default="1,2,3", help="the parts to run, comma-separated (default 1,2,3)")
    a = ap.parse_args(argv)
    parts = set(int(x) for x in a.parts.split(","))
    t_all = time.time()
    cfg = prod_cfg()
    scfg = cfg.sigma_calibration
    vers = ", ".join("%s %s" % (p, importlib.metadata.version(p)) for p in ("numpy", "scipy"))
    print("Block 42 identifiability check, %s (python %s, %s, %s; OMP_NUM_THREADS=%s)"
          % (time.strftime("%Y-%m-%d %H:%M"), platform.python_version(), vers, platform.platform(),
             os.environ.get("OMP_NUM_THREADS", "unset")))
    ic, rc = cfg.inference, cfg.renderer
    print("forward: line_fast under inference.absorption %r, %d x %d directions, line_fine_factor %d, reach %.1f "
          "sigmas; the configured kernel table delta %s um, sigma %s um (%s continuation, slope %.2f); the fit: "
          "max_nfev %d, ftol %g, xtol %g, steps %s, bounds d %s mu %s v0/z about the start %s %s sigma0 %s c %s"
          % (ic.absorption, ic.ray_n_rho, ic.ray_n_psi, ic.line_fine_factor, ic.line_reach_sigmas,
             rc.kernel_table_delta_um, rc.kernel_table_sigma_um, rc.kernel_continuation, rc.kernel_continuation_slope,
             scfg.max_nfev, scfg.ftol, scfg.xtol, scfg.diff_steps, scfg.d_bounds_um, scfg.mu_bounds_per_um,
             scfg.v0_bounds_um, scfg.z_bounds_um, scfg.sigma0_bounds_um, scfg.growth_scale_bounds))
    print("profiles (1)-(2): (%.0f %+.0f v) gl x tau(theta_true), noise %.1f gl per sample; theta_true = (d, %.2f /um, "
          "v0_start + %.2f um, z_start + %.2f um, sigma0, c %.1f)" % (B0, B1, NOISE, MU, DV0, DZ, C_TRUE))
    sys.stdout.flush()
    cache = line_fast.SectionCache()
    rows1, rows2 = [], []
    if parts & {1, 2}:
        print("\nbuilding the nodes and running parts %s" % sorted(parts & {1, 2}))
        rows1, rows2 = part1_and_2(cfg, parts, cache)
    if rows1:
        print("\n(1) SEs from J at the truth (noise %.1f gl, background %.0f gl), Cov = (J^T J)^-1" % (NOISE, B0))
        print("  d um  sigma0 um | SE d     SE mu     SE v0    SE z_ax  SE sigma0  SE c   | corr(s0,d) corr(s0,c) "
              "corr(d,mu) corr(d,c) | cov    evals  s     | min tau at k*")
        for d, s0, se, c_sd, c_sc, c_dm, c_dc, how, ne, sec, tmin in rows1:
            print("  %4.1f  %5.2f     | %.5f  %.6f  %.5f  %.5f  %.5f    %.4f | %+.3f     %+.3f     %+.3f     %+.3f    "
                  "| %-5s  %5d  %5.1f | %.4f" % (d, s0, se[0], se[1], se[2], se[3], se[4], se[5], c_sd, c_sc, c_dm,
                                                c_dc, how, ne, sec, tmin))
    if rows2:
        print("\n(2) one noisy fit per d at sigma0 %.2f um, c %.1f, mu %.2f /um (seed [%d, index of d])"
              % (S0_FIT, C_TRUE, MU, SEED))
        print("  d um | status     evals  seconds  chi2/dof | error d    mu        v0        z_ax      sigma0    c       "
              "| SE_s d   sigma0   c       | at_bound")
        for d, th, th0, fit in rows2:
            e = fit.theta - th
            print("  %4.1f | %-9s  %5d  %7.1f  %7.3f  | %+.5f  %+.6f  %+.5f  %+.5f  %+.5f  %+.4f | %.5f  %.5f  %.4f | %s"
                  % (d, fit.status, fit.n_eval, fit.seconds, fit.chi2 / fit.dof, e[0], e[1], e[2], e[3], e[4], e[5],
                     fit.se_scaled[0], fit.se_scaled[4], fit.se_scaled[5], ",".join(fit.at_bound) or "-"))
        for d, th, th0, fit in rows2:
            print("  d %.1f: truth %s; start %s; fit %s; %s" % (d, np.array2string(th, precision=4),
                                                                np.array2string(th0, precision=4),
                                                                np.array2string(fit.theta, precision=5), fit.message))
    if 3 in parts:
        print("\n(3) the cross-renderer case: 2-D hybrid render (8 x 16 directions, kernel with_kernel(renderer, %.2f, "
              "%.1f)) of a %.1f um tube, mu %.2f /um, fitted by Block 42 with the line renderer"
              % (CROSS["sigma0"], CROSS["c"], CROSS["d"], CROSS["mu"]))
        sys.stdout.flush()
        r = part3(cfg, cache)
        fit, th, diff = r["fit"], r["th"], r["diff"]
        e = fit.theta - th
        print("  the two renderers at the truth, before any fit: profile (2-D) - 210 gl x tau_line, per plane max |.| "
              "%s gl; dip depth at k* (2-D) %.2f gl" % (np.array2string(np.max(np.abs(diff), axis=1), precision=3),
                                                        float(np.max(r["B"][r["k_index"]] - r["prof"][r["k_index"]]))))
        print("  build_node's backgrounds B_k %s gl" % np.array2string(np.asarray(r["B"]), precision=3))
        print("  start %s" % np.array2string(r["th0"], precision=4))
        print("  truth %s" % np.array2string(th, precision=4))
        print("  fit   %s; %s, %d evaluations, %.1f s; %s" % (np.array2string(fit.theta, precision=5), fit.status,
                                                           fit.n_eval, fit.seconds, fit.message))
        print("  errors: sigma0 %+.2e um, d %+.2e um (%+.2e of d), c %+.2e (%+.2e of c); mu %+.2e /um, v0 %+.2e um, "
              "z_ax %+.2e um" % (e[4], e[0], e[0] / th[0], e[5], e[5] / th[5], e[1], e[2], e[3]))
        print("  the fit's SE at noise %.1f gl (for scale): d %.5f sigma0 %.5f c %.4f; chi2/N per plane %s (in units "
              "of %.1f gl: the model mismatch alone); fitted background b0 %s gl, b1 %s gl/um"
              % (NOISE, fit.se[0], fit.se[4], fit.se[5], np.array2string(fit.chi2_per_plane, precision=4), NOISE,
                 np.array2string(fit.b[:, 0], precision=3), np.array2string(fit.b[:, 1], precision=4)))
    print("\ntotal %.1f min" % ((time.time() - t_all) / 60.0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
