#!/usr/bin/env python3
"""Standalone check (NOT pipeline code), 2026-11-18, for choices A1, A2 and A5 of the joint-fit handoff (section 8b):
what the centre weights of D-169.4 and the cut of D-169.5 do to a clear node's fitted diameter.

Truth (noise-free, so fitted minus true is the bias): the node, one flat tube of diameter D and absorption MU, axis at
the depth of plane 0, centred on the line's origin; optionally a "bump": a second, untraced flat tube parallel to it,
diameter DB, the same absorption, its axis BUMP_V um away across the line, at depth BUMP_Z. Rendered with the
pipeline's fast line renderer (SPEC Block 35), hybrid absorption, the pilot's kernel table (sigma 0.080, 0.1536,
0.3924, 0.5620 um at 0, 0.28, 0.56, 0.84 um, slope 0.79), 9 planes (k = -4 .. 4, 0.28 um apart), one line of 53
samples at 0.1144 um (n_avg = 0), B = 200 gl. Two tubes are rendered as the product of their transmittances (the
renderer draws one tube; the product is exact only for tubes whose columns do not overlap, as here at 1.3 um).

Fit: always the one-tube model, theta = (d, mu, v0, z_ax), scipy least_squares (trf, x_scale jac), start
d = Allen's = D, mu = 0.5, v0 = 0, z = 0. Weightings, on the squared residuals:
  none   w_n = 1
  R1     w_n = exp(-|v_n - v_c| / lambda), lambda = 1.25 FWHM_A  (the handing-over chat's reading of D-169.4)
  R2     lambda = 1.25 FWHM_A / (2 ln 2): the weight's OWN FWHM = 1.25 FWHM_A
with FWHM_A = [(0.866 D)^2 + (2.355 x 0.080)^2]^0.5 and v_c re-centred by IRLS (weights fixed within a solve, then
v_c = the solve's v0; until d and v0 move by < 1e-4 um, at most 15 solves) -- D-052's scheme.
  +cut   the samples from the saddle between the node's dip and the bump outwards (the bump's side) get weight 0 in
         every plane; the saddle = the highest sample between the two minima of the noise-free line in plane 0.
Also: SE of d for the clean node at sigma = 1 gl, by the sandwich (J'WJ)^-1 J'W^2J (J'WJ)^-1 sigma^2 at the truth.

Run:  python3 a1_a5_weights_bump.py "<repo>/Passive Features/Diameter Re-measurement/src"
"""
import dataclasses
import sys
import time

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, sys.argv[1])
from allen_diameter.config import default_config                      # noqa: E402
from allen_diameter.model import geometry as G, line_fast as LF        # noqa: E402

B_GL, S0 = 200.0, 0.080
DELTA, SIGMA_T, SLOPE = (0.0, 0.28, 0.56, 0.84), (0.0800, 0.1536, 0.3924, 0.5620), 0.79
base = default_config()
P = base.acquisition.res0_um
V = np.arange(-26, 27) * P
KS = np.arange(-4, 5)
O0, YH, EU = np.array([437 * P, 211 * P]), np.array([0.0, 1.0]), np.array([1.0, 0.0])
CFG = dataclasses.replace(base, renderer=dataclasses.replace(
    base.renderer, kernel_table_delta_um=DELTA, kernel_table_sigma_um=SIGMA_T, sigma_r0_um=SIGMA_T[0],
    kernel_continuation_slope=SLOPE), inference=dataclasses.replace(base.inference, line_fine_factor=4))
CACHE = LF.SectionCache()
LO, HI = np.array([0.1, 0.01, -1.0, -1.0]), np.array([6.0, 10.0, 1.0, 1.0])


def tau(d, mu, v0, z):
    tube = G.Tube((O0[0], O0[1] + v0, z), 0.5 * d, 0.0, 0.0, 1.0, 10.0, "axial")
    return LF.line_profiles(tube, mu, KS, O0, YH, EU, V, 0, P, CFG, "hybrid", CACHE)


def model(th):
    return B_GL * tau(*th)


def solve(data, w, start):
    sw = np.sqrt(w)[None, :]
    res = least_squares(lambda th: (sw * (model(th) - data)).ravel(), start, bounds=(LO, HI), method="trf",
                        x_scale="jac", diff_step=1e-3, max_nfev=200)
    return res.x


def fit(data, D, rule, cut=None):
    fw = np.hypot(0.866 * D, 2.355 * S0)
    lam = {"none": None, "R1": 1.25 * fw, "R2": 1.25 * fw / (2 * np.log(2))}[rule]
    mask = np.ones_like(V) if cut is None else cut
    th = np.array([D, 0.5, 0.0, 0.0])
    if lam is None:
        return solve(data, mask, th), 1
    for it in range(1, 16):
        w = np.exp(-np.abs(V - th[2]) / lam) * mask
        new = solve(data, w, th)
        done = abs(new[0] - th[0]) < 1e-4 and abs(new[2] - th[2]) < 1e-4
        th = new
        if done:
            break
    return th, it


def se_d(D, MU, rule):
    fw = np.hypot(0.866 * D, 2.355 * S0)
    lam = {"none": None, "R1": 1.25 * fw, "R2": 1.25 * fw / (2 * np.log(2))}[rule]
    w = np.ones_like(V) if lam is None else np.exp(-np.abs(V) / lam)
    th = np.array([D, MU, 0.0, 0.0])
    h = np.array([1e-3, 1e-3, 1e-3, 1e-3])
    J = np.stack([((model(th + h[i] * np.eye(4)[i]) - model(th - h[i] * np.eye(4)[i])) / (2 * h[i])).ravel()
                  for i in range(4)], axis=1)
    W = np.tile(w, len(KS))
    A = np.linalg.inv(J.T @ (W[:, None] * J))
    C = A @ (J.T @ ((W ** 2)[:, None] * J)) @ A          # sigma = 1 gl
    return float(np.sqrt(C[0, 0]))


if __name__ == "__main__":
    t0 = time.time()
    MU = 0.5
    for D, DB, BV, BZ in ((0.8, 0.6, -1.3, 0.0), (0.45, 0.6, -1.3, 0.0), (0.8, 0.6, -1.3, 0.84)):
        node = model(np.array([D, MU, 0.0, 0.0]))
        bump = tau(DB, MU, BV, BZ)
        data = node * bump
        line0 = data[4]
        i_core = int(np.argmin(np.where(V > BV / 2, line0, np.inf)))
        i_bump = int(np.argmin(np.where(V < BV / 2, line0, np.inf)))
        lo, hi = sorted((i_core, i_bump))
        i_sad = lo + int(np.argmax(line0[lo:hi + 1]))
        cut = (V > V[i_sad]).astype(float) if BV < 0 else (V < V[i_sad]).astype(float)
        print("\nnode D %.2f um, mu %.2f; bump DB %.2f um at v %+.2f um, depth %+.2f um: dips in plane 0 "
              "core %.1f gl at v %+.2f, bump %.1f gl at v %+.2f, saddle %.1f gl at v %+.2f"
              % (D, MU, DB, BV, BZ, B_GL - line0[i_core], V[i_core], B_GL - line0[i_bump], V[i_bump],
                 B_GL - line0[i_sad], V[i_sad]), flush=True)
        for rule in ("none", "R1", "R2"):
            fw = np.hypot(0.866 * D, 2.355 * S0)
            lam = {"none": float("nan"), "R1": 1.25 * fw, "R2": 1.25 * fw / (2 * np.log(2))}[rule]
            x0, n0 = fit(node, D, rule)
            x1, n1 = fit(data, D, rule)
            x2, n2 = fit(data, D, rule, cut)
            print("  %-4s lambda %5.3f um (w at the bump %.2f) | clean d/D %.3f | bump d/D %.3f (mu %.2f, v0 %+.3f; "
                  "%d solves) | bump+cut d/D %.3f (mu %.2f, v0 %+.3f; %d solves)"
                  % (rule, lam, np.exp(-abs(BV) / lam) if rule != "none" else 1.0, x0[0] / D, x1[0] / D, x1[1],
                     x1[2], n1, x2[0] / D, x2[1], x2[2], n2), flush=True)
    print("\nSE of d at sigma = 1 gl, clean node, sandwich, mu %.2f:" % MU)
    for D in (0.45, 0.8, 1.2):
        print("  D %.2f um: none %.4f, R1 %.4f, R2 %.4f um" % (D, se_d(D, MU, "none"), se_d(D, MU, "R1"),
                                                             se_d(D, MU, "R2")), flush=True)
    print("%.0f s" % (time.time() - t0))
