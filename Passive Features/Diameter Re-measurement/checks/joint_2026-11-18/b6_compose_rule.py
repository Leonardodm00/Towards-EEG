#!/usr/bin/env python3
"""Standalone check (NOT pipeline code), 2026-11-18, for choices B5 and B6 of the joint-fit handoff (section 8b):
can the joint model of a node and a close neighbour be COMPOSED from each object rendered alone, or must the two
be rendered together?

Truth: node and neighbour rendered JOINTLY under the hybrid (scene_render.render_scene, SPEC Block 24), set-up of
checks/tier2_2026-11-18/tier2_neighbour_check.py: flat infinite tubes, node along x at depth 0 (d 0.6 um, mu 0.5
/um), the line along y (53 samples of one pixel), 9 planes k = -4..4 (0.28 um apart), the pilot's kernel table,
8 x 16 ray directions, B = 200 gl, noise-free. Neighbour d 0.6 um, mu 0.5 /um; scene = (lateral offset um, depth um,
heading deg); depth > 0 and < 0 are both run (which of the two shades the other depends on the light's direction).

Composed models, from tau_i (node alone) and tau_q (neighbour alone):
  sum      tau = tau_i + tau_q - 1       (shadows add; what subtracting a neighbour's shadow assumes)
  product  tau = tau_i * tau_q
Reported: max over planes and samples of |B (tau_joint - tau_composed)| (gl), and where it is.
With --fit: the node's four parameters and the neighbour's diameter fitted to the joint truth with the composed
model (neighbour's absorption and position held at the truth, as D-169.3 and the user's lean on B1): the bias the
composition alone puts on the two diameters.

Run:  python3 b6_compose_rule.py "<repo>/Passive Features/Diameter Re-measurement/src" [--fit]
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

B_GL = 200.0
NODE = (0.60, 0.50, 0.0, 0.0)
NB_D, NB_MU = 0.60, 0.50
SCENES = (("parallel, 1.0 um aside, same depth", (1.0, 0.0, 0.0)),
          ("parallel, 0.7 um aside, same depth", (0.7, 0.0, 0.0)),
          ("crossing 45 deg, 0.84 um deeper", (0.0, 0.84, 45.0)),
          ("crossing 45 deg, 0.84 um shallower", (0.0, -0.84, 45.0)),
          ("parallel on the line, 1.68 um deeper", (0.0, 1.68, 0.0)),
          ("parallel on the line, 1.68 um shallower", (0.0, -1.68, 0.0)),
          ("crossing 45 deg, 2.8 um deeper", (0.0, 2.8, 45.0)),
          ("parallel, 2.0 um aside, 1.68 um deeper", (2.0, 1.68, 0.0)))
N_V, KS, PAD_PX, FACTOR = 53, np.arange(-4, 5), 14, 4
base = default_config()
PX, DZ = base.acquisition.res0_um, base.acquisition.dz_um
RC = dataclasses.replace(base.renderer, absorption="hybrid", kernel_table_delta_um=(0.0, 0.28, 0.56, 0.84),
                         kernel_table_sigma_um=(0.0800, 0.1536, 0.3924, 0.5620), sigma_r0_um=0.0800,
                         kernel_continuation_slope=0.79, ray_n_rho=8, ray_n_psi=16)
GRID, INNER = R.block_fine_grid(0, -(N_V // 2), 1, N_V, PX, FACTOR, PAD_PX)
ZP = KS * DZ
V = (np.arange(N_V) - N_V // 2) * PX


def node_tube(d, v0, z):
    return G.Tube((0.0, v0, z), 0.5 * d, 0.0, 0.0)


def nb_tube(d, scene):
    off, depth, heading = scene
    return G.Tube((0.0, off, depth), 0.5 * d, 0.0, math.radians(heading))


def tau(tubes, mus):
    fine = SR.render_scene(tubes, mus, ZP, GRID, RC).tau
    return R.pixel_integrate(fine[:, INNER[0], INNER[1]], FACTOR)[:, :, 0]


if __name__ == "__main__":
    t0 = time.time()
    do_fit = "--fit" in sys.argv
    t_i = tau([node_tube(NODE[0], NODE[2], NODE[3])], [NODE[1]])
    print("light_direction %+d; node alone: dip %.1f gl in focus" % (base.acquisition.light_direction,
                                                                     B_GL * (1 - t_i).max()))
    for name, scene in SCENES:
        t_q = tau([nb_tube(NB_D, scene)], [NB_MU])
        t_j = tau([node_tube(NODE[0], NODE[2], NODE[3]), nb_tube(NB_D, scene)], [NODE[1], NB_MU])
        e_sum = B_GL * (t_j - (t_i + t_q - 1.0))
        e_prod = B_GL * (t_j - t_i * t_q)
        ks, ns = np.unravel_index(np.abs(e_sum).argmax(), e_sum.shape)
        kp, npp = np.unravel_index(np.abs(e_prod).argmax(), e_prod.shape)
        print("\n%s (neighbour's shadow on the line: peak %.1f gl)" % (name, B_GL * (1 - t_q).max()))
        print("  joint - sum:     max |.| %5.2f gl (%+.2f at k %+d, v %+.2f um); rms %.2f gl"
              % (np.abs(e_sum).max(), e_sum[ks, ns], KS[ks], V[ns], np.sqrt(np.mean(e_sum ** 2))))
        print("  joint - product: max |.| %5.2f gl (%+.2f at k %+d, v %+.2f um); rms %.2f gl"
              % (np.abs(e_prod).max(), e_prod[kp, npp], KS[kp], V[npp], np.sqrt(np.mean(e_prod ** 2))), flush=True)
        if do_fit and max(np.abs(e_sum).max(), np.abs(e_prod).max()) >= 0.5:
            data = B_GL * t_j
            x0 = np.array([NODE[0] * 1.2, NODE[1] * 0.9, 0.02, 0.05, NB_D * 0.8])
            lo, hi = np.array([0.1, 0.01, -1.0, -1.0, 0.1]), np.array([6.0, 5.0, 1.0, 1.0, 6.0])
            for label, combine in (("sum", lambda a, b: a + b - 1.0), ("product", lambda a, b: a * b)):
                def r(th, combine=combine):
                    a = tau([node_tube(th[0], th[2], th[3])], [th[1]])
                    b = tau([nb_tube(th[4], scene)], [NB_MU])
                    return (B_GL * combine(a, b) - data).ravel()
                res = least_squares(r, x0, bounds=(lo, hi), method="trf", x_scale="jac", diff_step=1e-3, max_nfev=40)
                print("  fit with the %-7s model: node d %.3f um (%+.1f %%), mu %.3f, z %+.3f; neighbour d %.3f um "
                      "(%+.1f %%); rms %.2f gl" % (label, res.x[0], 100 * (res.x[0] / NODE[0] - 1), res.x[1], res.x[3],
                                                    res.x[4], 100 * (res.x[4] / NB_D - 1),
                                                    np.sqrt(np.mean(res.fun ** 2))), flush=True)
    print("\n%.0f s" % (time.time() - t0))
