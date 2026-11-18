#!/usr/bin/env python3
"""Standalone check (NOT pipeline code), 2026-11-18, for choice A1 of the joint-fit handoff (section 8b):
the half-depth width (FWHM) of the model's own in-focus dip for a tube of Allen's diameter, against the
quadrature formula FWHM_A = [(0.866 D)^2 + (2.355 sigma_r0)^2]^(1/2) of D-169.10.

Model: the pipeline's fast line renderer (SPEC Block 35), hybrid absorption, one flat tube (heading along x,
tilt 0), axis at the depth of plane k = 0, one line (n_avg = 0) of 53 samples at 0.1144 um, read across the tube.
Kernel: the pilot's table (sigma 0.080, 0.1536, 0.3924, 0.5620 um at defocus 0, 0.28, 0.56, 0.84 um, slope 0.79
beyond) and, for a larger in-focus width s0, the same table with s0^2 - 0.080^2 added to every variance (as
checks/kernel_2026-11-18/kernel_infocus_check.py). FWHM of the dip: the width at half its depth, by linear
interpolation between samples on a 4x finer line (the renderer evaluated at 0.0286 um steps).

Run:  python3 a1_infocus_width.py "<repo>/Passive Features/Diameter Re-measurement/src"
"""
import dataclasses
import sys
import time

import numpy as np

sys.path.insert(0, sys.argv[1])
from allen_diameter.config import default_config                      # noqa: E402
from allen_diameter.model import geometry as G, line_fast as LF        # noqa: E402

DELTA = (0.0, 0.28, 0.56, 0.84)
SIGMA_T = (0.0800, 0.1536, 0.3924, 0.5620)
SLOPE = 0.79
base = default_config()
P = base.acquisition.res0_um
V = np.arange(-26 * 4, 26 * 4 + 1) * (P / 4.0)      # fine line, +-2.97 um
KS = np.array([0])
O0, YH, EU = np.array([437 * P, 211 * P]), np.array([0.0, 1.0]), np.array([1.0, 0.0])


def cfg_for(s0):
    extra = s0 ** 2 - SIGMA_T[0] ** 2
    sig = tuple(float(np.sqrt(s ** 2 + extra)) for s in SIGMA_T)
    return dataclasses.replace(base, renderer=dataclasses.replace(
        base.renderer, kernel_table_delta_um=DELTA, kernel_table_sigma_um=sig, sigma_r0_um=sig[0],
        kernel_continuation_slope=SLOPE), inference=dataclasses.replace(base.inference, line_fine_factor=4))


def dip(d, mu, cfg, cache):
    tube = G.Tube((O0[0], O0[1], 0.0), 0.5 * d, 0.0, 0.0, 1.0, 10.0, "axial")
    return 1.0 - LF.line_profiles(tube, mu, KS, O0, YH, EU, V, 0, P / 4.0, cfg, "hybrid", cache)[0]


def fwhm(v, s):
    """Width at half the maximum of s (s >= 0, one peak), linear interpolation on each flank."""
    i = int(np.argmax(s))
    h = 0.5 * s[i]
    left = np.flatnonzero(s[:i] < h)[-1]
    right = i + np.flatnonzero(s[i:] < h)[0]
    vl = v[left] + (h - s[left]) * (v[left + 1] - v[left]) / (s[left + 1] - s[left])
    vr = v[right - 1] + (h - s[right - 1]) * (v[right] - v[right - 1]) / (s[right] - s[right - 1])
    return vr - vl


if __name__ == "__main__":
    t0 = time.time()
    print("FWHM of the in-focus dip (um): model (hybrid, pilot table) vs formula [(0.866 D)^2 + (2.355 s0)^2]^0.5")
    print("%5s %5s %6s | %8s %8s %8s | %7s" % ("s0", "D", "mu", "depth", "model", "formula", "ratio"))
    for s0 in (0.080, 0.18, 0.28):
        cfg = cfg_for(s0)
        for d in (0.45, 0.8, 1.2):
            for mu in (0.2, 0.9):
                s = dip(d, mu, cfg, LF.SectionCache())
                w_model = fwhm(V, s)
                w_form = np.hypot(0.866 * d, 2.355 * s0)
                print("%5.3f %5.2f %6.2f | %7.1f%% %8.3f %8.3f | %7.3f"
                      % (s0, d, mu, 100 * s.max(), w_model, w_form, w_model / w_form), flush=True)
    print("%.0f s" % (time.time() - t0))
