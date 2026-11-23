#!/usr/bin/env python3
"""Standalone check (NOT pipeline code), 2026-11-23 (D-221, D-222): verification of sgt_line_model.py.

(V1) The 1-D model with the Gaussian kernel against the pipeline's own fast line renderer (Block 35
     line_fast.line_profiles, absorption "partition_vertical", the default configuration) on a flat tube along
     the pixel grid (heading 0, the line through a pixel centre across it, n_avg = 4 lines along the tube, half
     length 12 um, axial caps): d in {0.4, 1.5, 2.5} um, mu in {0.5, 1.5} /um, the axis 0.04 um aside and at depth
     0 or 0.5 um, planes -4..4.
(V2) The SGT's CDF (closed form) against scipy.integrate.quad of blur_kernels.sgt_kernel_pdf, the pipeline's
     density, for the four truths of the check and the box's corners; its derivative against the density; the
     (2, 0, 1) member against ndtr.
(V3) The fast H (cubic Hermite) against the exact one: on the CDF, and on the model tau for every truth case.
(V4) The grouped slab-plane sum (one 2-D correlation, FFT) against the explicit per-pair sum.
(V5) The time of one forward evaluation per diameter and kernel path.

Run:
    cd "Passive Features/Diameter Re-measurement/checks/sgt_identifiability_2026-11-23"
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 validate_model.py > validate_model.out
Writes validation.json. Pure ASCII, LF.
"""
from __future__ import annotations

import json
import math
import sys
import time

import numpy as np
from scipy import integrate, special

import sgt_line_model as M
from allen_diameter.model import blur_kernels, geometry, line_fast

OUT_JSON = M.HERE / "validation.json"


def v1_line_renderer(st, half_length=12.0):
    cfg = st.cfg
    rows = []
    for d in (0.4, 1.5, 2.5):
        for mu in (0.5, 1.5):
            for z_ax in (0.0, 0.5):
                v0 = 0.04
                tube = geometry.Tube((0.0, v0, z_ax), 0.5 * d, 0.0, 0.0, 1.0, half_length, "axial")
                t0 = time.perf_counter()
                ref = line_fast.line_profiles(tube, mu, st.ks, (0.0, 0.0), (0.0, 1.0), (1.0, 0.0), st.v, 4,
                                              float(cfg.measure.profile_step_um), cfg,
                                              absorption="partition_vertical")
                t_ref = time.perf_counter() - t0
                t0 = time.perf_counter()
                mine = M.tau(d, mu, v0, z_ax, st, M.Kernel(M.GAUSS, "exact"))
                t_mine = time.perf_counter() - t0
                diff = float(np.abs(mine - ref).max())
                rows.append(dict(d=d, mu=mu, z_ax=z_ax, v0=v0, max_abs_diff=diff, min_tau=float(ref.min()),
                                 seconds_line_fast=t_ref, seconds_1d=t_mine))
                print("V1 d %.1f mu %.1f z_ax %.1f: max |tau_1D - tau_Block35| = %.2e (min tau %.4f); "
                      "%.2f s vs %.4f s" % (d, mu, z_ax, diff, ref.min(), t_ref, t_mine))
                sys.stdout.flush()
    return rows


SHAPES = [M.GAUSS, (2.0, 0.5, 1.0), (1.3, 0.0, 1.0), (2.0, 0.3, 1.3),
          (1.0, 0.0, 1.0), (10.0, 0.0, 1.0), (1.0, 2.0, 1.0), (10.0, 2.0, 1.0),
          (1.0, 2.0, math.exp(0.7)), (10.0, 0.0, math.exp(-0.7)), (2.0, 1e-10, 1.0)]


def v2_cdf():
    out = []
    worst = 0.0
    for shape in SHAPES:
        for w in (0.188, 0.6, 2.5):
            g = blur_kernels.sgt_kernel_pdf(w, *shape)
            kern = M.Kernel(shape, "exact", gauss_exact=False)
            sd = w / M.FWHM_PER_SD
            err = 0.0
            for t in (-3.0 * w, -w, -0.3 * w, -1e-3, 0.0, 0.2 * w, 0.7 * w, 2.0 * w, 5.0 * w):
                # quad from 0 (where F is known only through the closed form's F(0)) is avoided: integrate the
                # density from -inf with the break point at 0 (the two-piece kernel's kink)
                if t <= 0.0:
                    val, _e = integrate.quad(g, -np.inf, t, epsabs=1e-14, epsrel=1e-12, limit=400)
                else:
                    a, _e1 = integrate.quad(g, -np.inf, 0.0, epsabs=1e-14, epsrel=1e-12, limit=400)
                    b, _e2 = integrate.quad(g, 0.0, t, epsabs=1e-14, epsrel=1e-12, limit=400)
                    val = a + b
                mine = float(kern.cdf(np.array([t]), sd)[0])
                err = max(err, abs(mine - val))
            # derivative of the CDF against the density (central difference, step 1e-5 w)
            tt = np.linspace(-3 * w, 3 * w, 41) + 0.0123 * w
            hh = 1e-5 * w
            dF = (kern.cdf(tt + hh, sd) - kern.cdf(tt - hh, sd)) / (2 * hh)
            derr = float(np.max(np.abs(dF - g(tt)) / np.max(g(tt))))
            worst = max(worst, err)
            out.append(dict(shape=list(shape), fwhm_um=w, max_abs_cdf_vs_quad=err, max_rel_dcdf_vs_pdf=derr))
            print("V2 shape (%.3g, %.3g, %.3g) w %.3f: |F - quad| <= %.1e, |F' - g| / max g <= %.1e"
                  % (shape + (w, err, derr)))
    # the Gaussian member against ndtr, and xi -> 0 continuity
    t = np.linspace(-2.0, 2.0, 2001)
    sd = 0.3
    g_sgt = M.Kernel(M.GAUSS, "exact", gauss_exact=False).cdf(t, sd)
    g_nd = special.ndtr(t / sd)
    gauss_diff = float(np.abs(g_sgt - g_nd).max())
    x_small = M.Kernel((2.0, 1e-10, 1.0), "exact", gauss_exact=False).cdf(t, sd)
    xi_cont = float(np.abs(x_small - g_sgt).max())
    x_fd = M.Kernel((2.0, 1e-6, 1.0), "exact", gauss_exact=False).cdf(t, sd)
    xi_slope = float(np.abs(x_fd - g_sgt).max() / 1e-6)
    print("V2 SGT (2, 0, 1) vs ndtr: max |diff| = %.2e; xi = 1e-10 vs 0: %.2e; |dF/dxi| at 0 about %.3f"
          % (gauss_diff, xi_cont, xi_slope))
    return dict(rows=out, worst_cdf_vs_quad=worst, gauss_member_vs_ndtr=gauss_diff, xi_1e10_vs_0=xi_cont,
                dF_dxi_at_0=xi_slope)


def v3_fast(st):
    out = []
    t = np.linspace(-30.0, 30.0, 200001)
    for shape in SHAPES[:4] + [(1.0, 2.0, math.exp(0.7)), (10.0, 0.0, math.exp(-0.7))]:
        ex = M.Kernel(shape, "exact", gauss_exact=False).cdf(t, 0.08)
        fa = M.Kernel(shape, "fast", gauss_exact=False).cdf(t, 0.08)
        out.append(dict(shape=list(shape), max_abs_cdf=float(np.abs(ex - fa).max())))
        print("V3 CDF fast vs exact, shape (%.3g, %.3g, %.3g): %.2e" % (tuple(shape) + (out[-1]["max_abs_cdf"],)))
    model = []
    for d in (0.4, 0.8, 1.5, 2.5):
        for mu in (0.5, 1.5):
            for z_ax in (0.0, 0.5):
                for shape in SHAPES[:4]:
                    a = M.tau(d, mu, 0.04, z_ax, st, M.Kernel(shape, "exact", gauss_exact=False))
                    b = M.tau(d, mu, 0.04, z_ax, st, M.Kernel(shape, "fast", gauss_exact=False))
                    model.append(float(np.abs(a - b).max()))
    print("V3 model tau fast vs exact over 64 truth cases: max %.2e (noise SD in tau units %.2e)"
          % (max(model), st.sigma_gl / st.B_gl))
    return dict(cdf=out, model_max_abs=max(model))


def v4_grouping(st):
    worst = 0.0
    for d, mu, z_ax in ((0.4, 0.5, 0.0), (0.8, 1.5, 0.5), (2.5, 1.5, 0.5)):
        for shape in (M.GAUSS, (2.0, 0.3, 1.3)):
            k = M.Kernel(shape, "exact", gauss_exact=False)
            a = M.deficit(d, mu, 0.04, z_ax, st, k)
            b = M.deficit_direct(d, mu, 0.04, z_ax, st, k)
            worst = max(worst, float(np.abs(a - b).max()))
    print("V4 grouped FFT correlation vs the per-pair sum: max %.2e" % worst)
    return dict(max_abs=worst)


def v5_timing(st, reps=5):
    rows = []
    for d in (0.4, 0.8, 1.5, 2.5):
        for label, kern in (("gauss ndtr", M.Kernel(M.GAUSS, "exact")),
                            ("sgt exact", M.Kernel((2.0, 0.5, 1.0), "exact")),
                            ("sgt fast", M.Kernel((2.0, 0.5, 1.0), "fast"))):
            M.tau(d, 1.5, 0.04, 0.5, st, kern)
            t0 = time.perf_counter()
            for _ in range(reps):
                M.tau(d, 1.5, 0.04, 0.5, st, kern)
            dt = (time.perf_counter() - t0) / reps
            t0 = time.perf_counter()
            for _ in range(reps):
                M.tau(d, 1.5, 0.04, 0.5, st, M.Kernel((2.0, 0.5, 1.0), "fast"))
            dt_build = (time.perf_counter() - t0) / reps
            rows.append(dict(d=d, kernel=label, ms=1e3 * dt))
            print("V5 d %.1f %-10s %.1f ms per call%s" % (d, label, 1e3 * dt,
                  ("" if label != "sgt fast" else "  (%.1f ms with the H table built)" % (1e3 * dt_build))))
    return rows


def main():
    st = M.make_setup()
    print("setup: p %.4f um, f %d, dzeta %.3f um, planes %s, n_v %d, B %.0f gl, sigma %.2f gl"
          % (st.p, st.f, st.dzeta, list(st.ks), st.v.size, st.B_gl, st.sigma_gl))
    res = dict(v1=v1_line_renderer(st), v2=v2_cdf(), v3=v3_fast(st), v4=v4_grouping(st), v5=v5_timing(st))
    OUT_JSON.write_text(json.dumps(res, indent=1))
    print("wrote", OUT_JSON.name)


if __name__ == "__main__":
    main()
