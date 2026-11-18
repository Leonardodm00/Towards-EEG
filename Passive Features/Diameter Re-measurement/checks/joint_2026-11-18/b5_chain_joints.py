#!/usr/bin/env python3
"""Standalone check (NOT pipeline code), 2026-11-18, for choice B5 of the joint-fit handoff (section 8b): a chain
of pieces between traced nodes (D-164.1) has a joint AT the working node, i.e. on its measuring line. How must the
pieces be rendered for a straight chain to reproduce the straight tube?

One straight flat tube along x at depth 0, 9.44 um long (capped, axial), against the SAME solid cut into 8 pieces
of 1.18 um (the trace's mean node spacing, config phantom_node_step_um), one cut on the line (x = 0):
  T  one tube                                       scene_render.render_scene, one Tube
  J  the 8 pieces rendered together                 render_scene, 8 Tubes (the hybrid's back lengths run through
                                                    every piece, SPEC Block 24)
  S  the 8 pieces each rendered ALONE, shadows summed  (how SPEC Block 25's shadow model renders a stretch:
                                                    each segment a capped cylinder on its own)
Hybrid absorption, the pilot's kernel table, 8 x 16 ray directions, 9 planes, 53 samples of one pixel on the line
x = 0, B = 200 gl, noise-free. Reported: max |B (tau_T - tau_X)| (gl) and, for S, the one-tube fit (d, mu, v0, z)
of the S profiles: the bias the summed-alone chain would carry against a continuous dendrite.

Run:  python3 b5_chain_joints.py "<repo>/Passive Features/Diameter Re-measurement/src"
"""
import dataclasses
import sys
import time

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, sys.argv[1])
from allen_diameter.config import default_config                      # noqa: E402
from allen_diameter.model import geometry as G                         # noqa: E402
from allen_diameter.model import render as R                           # noqa: E402
from allen_diameter.model import scene_render as SR                    # noqa: E402

B_GL, PIECE, N_PIECES = 200.0, 1.18, 8
N_V, KS, PAD_PX, FACTOR = 53, np.arange(-4, 5), 24, 4
base = default_config()
PX, DZ = base.acquisition.res0_um, base.acquisition.dz_um
RC = dataclasses.replace(base.renderer, absorption="hybrid", kernel_table_delta_um=(0.0, 0.28, 0.56, 0.84),
                         kernel_table_sigma_um=(0.0800, 0.1536, 0.3924, 0.5620), sigma_r0_um=0.0800,
                         kernel_continuation_slope=0.79, ray_n_rho=8, ray_n_psi=16)
GRID, INNER = R.block_fine_grid(0, -(N_V // 2), 1, N_V, PX, FACTOR, PAD_PX)
ZP = KS * DZ
HALF = 0.5 * PIECE * N_PIECES


def tau(tubes, mus):
    fine = SR.render_scene(tubes, mus, ZP, GRID, RC).tau
    return R.pixel_integrate(fine[:, INNER[0], INNER[1]], FACTOR)[:, :, 0]


def whole(d, v0=0.0, z=0.0):
    return G.Tube((0.0, v0, z), 0.5 * d, 0.0, 0.0, 1.0, HALF, "axial")


def pieces(d):
    xs = -HALF + PIECE * (np.arange(N_PIECES) + 0.5)
    return [G.Tube((float(x), 0.0, 0.0), 0.5 * d, 0.0, 0.0, 1.0, 0.5 * PIECE, "axial") for x in xs]


if __name__ == "__main__":
    t0 = time.time()
    for d, mu in ((0.45, 0.5), (0.8, 0.5), (1.2, 0.5), (0.8, 0.2)):
        t_T = tau([whole(d)], [mu])
        ps = pieces(d)
        t_J = tau(ps, [mu] * N_PIECES)
        t_S = 1.0 - sum(1.0 - tau([p], [mu]) for p in ps)
        eJ, eS = B_GL * (t_T - t_J), B_GL * (t_T - t_S)
        print("\nd %.2f um, mu %.2f /um: one tube's dip %.1f gl in focus" % (d, mu, B_GL * (1 - t_T).max()))
        print("  T - J (pieces together): max |.| %.3f gl" % np.abs(eJ).max())
        print("  T - S (pieces alone, summed): max |.| %.2f gl; S's dip %.1f gl in focus (%+.1f %%)"
              % (np.abs(eS).max(), B_GL * (1 - t_S).max(), 100 * ((1 - t_S).max() / (1 - t_T).max() - 1)), flush=True)
        data = B_GL * t_S
        res = least_squares(lambda th: (B_GL * tau([whole(th[0], th[2], th[3])], [th[1]]) - data).ravel(),
                            np.array([d, mu, 0.0, 0.0]), bounds=([0.1, 0.01, -1, -1], [6, 5, 1, 1]), method="trf",
                            x_scale="jac", diff_step=1e-3, max_nfev=40)
        print("  one-tube fit of S: d %.3f um (%+.1f %%), mu %.3f /um (%+.1f %%), rms %.2f gl"
              % (res.x[0], 100 * (res.x[0] / d - 1), res.x[1], 100 * (res.x[1] / mu - 1),
                 np.sqrt(np.mean(res.fun ** 2))), flush=True)
    print("\n%.0f s" % (time.time() - t0))
