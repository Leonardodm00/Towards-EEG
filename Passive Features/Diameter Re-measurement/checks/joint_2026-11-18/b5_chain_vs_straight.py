#!/usr/bin/env python3
"""Standalone check (NOT pipeline code), 2026-11-18, for the chain geometry of the joint fit (D-164.1, D-169.1):
how far is a node's fitted diameter off when the dendrite's axis and the model's axis differ in shape near the node?

Truth: a flat chain of 8 pieces of 1.15 um (the pilot trace's median node spacing), d 0.6 um, mu 0.5 /um, one joint
at the node, rendered together under the hybrid (scene_render.render_scene, SPEC Block 24: the ray history runs
through every piece; exact for a straight chain, checks/b5_chain_joints.py). Axial caps: at a joint turning by
beta the pieces overlap on the inside and leave a gap on the outside, each of width r tan(beta/2) at the tube's
edge (0.03 um at 10 deg, 0.12 um at 45 deg, for r 0.3 um), so the kinked truths are approximate there.
  arc R    the nodes on a circle of radius R tangent to the x axis at the node (constant turn per joint)
  kink b   two straight arms meeting at the node, the heading turning by b there
Model: one straight tube along x (the tangent of the arc; the bisector of the kink), theta = (d, mu, v0, z_ax) free,
unweighted least squares. Read as in the pipeline: 9 planes (k = -4..4), 53 samples across, the mean of 9 parallel
lines one pixel apart (+-0.46 um along the branch; measure.along_branch_avg_um = 0.5), the pilot's kernel table,
8 x 16 ray directions, B = 200 gl, noise-free. The same numbers measure the reverse case: a straight dendrite
modelled by a chain that follows a trace with that shape error.

Run:  python3 b5_chain_vs_straight.py "<repo>/Passive Features/Diameter Re-measurement/src"
"""
import dataclasses
import math
import sys
import time

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, sys.argv[1])
from allen_diameter.config import default_config                      # noqa: E402
from allen_diameter.model import geometry as G                         # noqa: E402
from allen_diameter.model import render as R                           # noqa: E402
from allen_diameter.model import scene_render as SR                    # noqa: E402

B_GL, D, MU, PIECE, N_HALF = 200.0, 0.6, 0.5, 1.15, 4
N_V, N_U, KS, PAD_PX, FACTOR = 53, 9, np.arange(-4, 5), 16, 4
base = default_config()
PX, DZ = base.acquisition.res0_um, base.acquisition.dz_um
RC = dataclasses.replace(base.renderer, absorption="hybrid", kernel_table_delta_um=(0.0, 0.28, 0.56, 0.84),
                         kernel_table_sigma_um=(0.0800, 0.1536, 0.3924, 0.5620), sigma_r0_um=0.0800,
                         kernel_continuation_slope=0.79, ray_n_rho=8, ray_n_psi=16)
GRID, INNER = R.block_fine_grid(-(N_U // 2), -(N_V // 2), N_U, N_V, PX, FACTOR, PAD_PX)
ZP = KS * DZ


def tau(tubes, mus):
    fine = SR.render_scene(tubes, mus, ZP, GRID, RC).tau
    return R.pixel_integrate(fine[:, INNER[0], INNER[1]], FACTOR).mean(axis=2)      # (planes, n_v): mean over the lines


def chain(points, d):
    out = []
    for a, b in zip(points[:-1], points[1:]):
        c, t = 0.5 * (a + b), b - a
        out.append(G.Tube((float(c[0]), float(c[1]), 0.0), 0.5 * d, 0.0, math.atan2(t[1], t[0]), 1.0,
                          0.5 * float(np.hypot(*t)), "axial"))
    return out


def arc(radius):
    s = PIECE * np.arange(-N_HALF, N_HALF + 1)
    return np.column_stack([radius * np.sin(s / radius), radius * (1.0 - np.cos(s / radius))])


def kink(beta_deg):
    h = math.radians(beta_deg) / 2.0
    L = PIECE * N_HALF
    return np.array([[-L * math.cos(h), L * math.sin(h)], [0.0, 0.0], [L * math.cos(h), L * math.sin(h)]])


def straight(d, v0, z):
    return G.Tube((0.0, v0, z), 0.5 * d, 0.0, 0.0)


if __name__ == "__main__":
    t0 = time.time()
    cases = [("straight chain (control)", np.column_stack([PIECE * np.arange(-N_HALF, N_HALF + 1), np.zeros(9)]))]
    cases += [("arc R %.0f um (turn %.1f deg per joint)" % (r, math.degrees(PIECE / r)), arc(r)) for r in (25.0, 12.0, 6.0)]
    cases += [("kink %d deg at the node" % b, kink(b)) for b in (10, 20, 45)]
    for name, pts in cases:
        tubes = chain(pts, D)
        data = B_GL * tau(tubes, [MU] * len(tubes))
        res = least_squares(lambda th: (B_GL * tau([straight(th[0], th[2], th[3])], [th[1]]) - data).ravel(),
                            np.array([D, MU, 0.0, 0.0]), bounds=([0.1, 0.01, -1, -1], [6, 5, 1, 1]), method="trf",
                            x_scale="jac", diff_step=1e-3, max_nfev=40)
        print("%-34s straight-tube fit: d %.3f um (%+.1f %%), mu %.3f /um (%+.1f %%), v0 %+.3f um, rms %.2f gl "
              "(dip %.1f gl)" % (name, res.x[0], 100 * (res.x[0] / D - 1), res.x[1], 100 * (res.x[1] / MU - 1),
                                 res.x[2], np.sqrt(np.mean(res.fun ** 2)), B_GL - data.min()), flush=True)
    print("%.0f s" % (time.time() - t0))
