#!/usr/bin/env python3
"""Standalone check (NOT pipeline code), 2026-11-18: how much does a wrong defocus kernel move the fitted diameter
of a clear node (one flat tube, nothing else on the line)?

Truth: one flat tube rendered with the fast line renderer (SPEC Block 35), hybrid absorption, the pilot's kernel
table T (sigma 0.080, 0.1536, 0.3924, 0.5620 um at 0, 0.28, 0.56, 0.84 um, slope 0.79 beyond); 9 planes
k = -4 .. 4, 53 line samples, noise-free. The fit is always the pipeline's model: one tube, Gaussian kernel, table T,
theta = (d, mu, v0, z_ax), unweighted least squares.

  A  shape   the TRUE kernel has a halo: K = (1 - f) G(sigma) + f G(HALO x sigma); the image is linear in the
             kernel, so the truth is (1 - f) x render(T) + f x render(HALO x T).
  B  growth  the TRUE kernel grows with defocus g times as fast in variance as the table says:
             sigma_true(delta)^2 = sigma_0^2 + g (sigma_T(delta)^2 - sigma_0^2)   (g < 1: the table overstates the growth).

Reported: fitted d / true d, fitted mu, and the rms misfit (gl at B = 200).
Run:  python3 kernel_sensitivity_check.py <src>
"""
import dataclasses
import sys
import time

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, sys.argv[1])
from allen_diameter.config import default_config                      # noqa: E402
from allen_diameter.model import geometry as G, line_fast as LF        # noqa: E402

B_GL = 200.0
TRUTHS = ((0.5, 0.8), (1.0, 0.25))             # (d um, mu 1/um): thin and dark, thick and faint (about the same dip)
DELTA = (0.0, 0.28, 0.56, 0.84)
SIGMA_T = (0.0800, 0.1536, 0.3924, 0.5620)
SLOPE = 0.79
HALO, FRACTIONS = 3.0, (0.1, 0.2, 0.3)
GROWTH = (0.5, 0.75, 1.5, 2.0)
base = default_config()
P = base.acquisition.res0_um
V = np.arange(-26, 27) * P
KS = np.arange(-4, 5)
O0, YH, EU = np.array([437 * P, 211 * P]), np.array([0.0, 1.0]), np.array([1.0, 0.0])


def cfg_for(sig, slope):
    return dataclasses.replace(base, renderer=dataclasses.replace(
        base.renderer, kernel_table_delta_um=DELTA, kernel_table_sigma_um=tuple(float(s) for s in sig),
        sigma_r0_um=float(sig[0]), kernel_continuation_slope=float(slope)),
        inference=dataclasses.replace(base.inference, line_fine_factor=4))


CFG_T = cfg_for(SIGMA_T, SLOPE)
CACHES = {}


def tau(theta, cfg, key):
    d, mu, v0, z = theta
    tube = G.Tube((O0[0], O0[1] + v0, z), 0.5 * d, 0.0, 0.0, 1.0, 10.0, "axial")
    return LF.line_profiles(tube, mu, KS, O0, YH, EU, V, 0, P, cfg, "hybrid", CACHES.setdefault(key, LF.SectionCache()))


def fit(data, start):
    res = least_squares(lambda th: (B_GL * tau(th, CFG_T, "T") - data).ravel(), start,
                        bounds=([0.1, 0.01, -1.0, -1.5], [6.0, 10.0, 1.0, 1.5]), x_scale="jac", diff_step=1e-2,
                        max_nfev=80)
    return res.x, float(np.sqrt(np.mean(res.fun ** 2)))


def grown(g):
    s0 = SIGMA_T[0]
    sig = [np.sqrt(s0 ** 2 + g * (s ** 2 - s0 ** 2)) for s in SIGMA_T]
    return cfg_for(sig, SLOPE * np.sqrt(g))


if __name__ == "__main__":
    t0 = time.time()
    for d, mu in TRUTHS:
        th = np.array([d, mu, 0.0, 0.0])
        ref = B_GL * tau(th, CFG_T, "T")
        x, rms = fit(ref, th * [1.3, 0.8, 1, 1] + [0, 0, 0.03, 0.05])
        print("\ntruth d %.2f um, mu %.2f /um (dip %.1f gl in focus)\n  control, true kernel = table: d/d_true %.3f, "
              "mu %.3f, rms %.2f gl" % (d, mu, B_GL - ref.min(), x[0] / d, x[1], rms), flush=True)
        halo = B_GL * tau(th, cfg_for(np.array(SIGMA_T) * HALO, SLOPE * HALO), "halo")
        for f in FRACTIONS:
            x, rms = fit((1 - f) * ref + f * halo, th)
            print("  A  halo %.0f %% at %.0f x sigma:  d/d_true %.3f, mu %.3f, z_ax %+.3f, rms %.2f gl"
                  % (100 * f, HALO, x[0] / d, x[1], x[3], rms), flush=True)
        for g in GROWTH:
            x, rms = fit(B_GL * tau(th, grown(g), "g%.2f" % g), th)
            print("  B  true growth %.2f x the table's:  d/d_true %.3f, mu %.3f, z_ax %+.3f, rms %.2f gl"
                  % (g, x[0] / d, x[1], x[3], rms), flush=True)
    print("\n%.0f s" % (time.time() - t0))
