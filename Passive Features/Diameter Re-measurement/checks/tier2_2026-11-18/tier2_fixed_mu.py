#!/usr/bin/env python3
"""Standalone check (NOT pipeline code), 2026-11-18: the neighbour's absorption mu_q FIXED instead of free.
Set-up of tier2_sloppiness.py (Jacobian at the truth, central differences, noise 1 gl). Free: the node's four
parameters and the neighbour's d_q, v0_q, z_q (7). Reported: the condition number of the scaled Jacobian, the SE
of the node's d and of d_q, and the first-order shift of both when mu_q is held MU_ERR (relative) too high:
dtheta = -(J7^T J7)^{-1} J7^T (J_muq * MU_ERR * mu_q).
Run:  python3 tier2_fixed_mu.py <src>"""
import importlib.util
import math
import os
import sys

import numpy as np

spec = importlib.util.spec_from_file_location("chk", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                  "tier2_neighbour_check.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
STEP, MU_ERR = 0.02, 0.30
SCENES = dict(m.SCENES)
SCENES["parallel, directly above, 1.68 um deeper"] = (0.0, 1.68, 0.0)
SCENES["crossing at 45 deg over the node, 2.8 um deeper"] = (0.0, 2.8, 45.0)
SCENES["parallel, 2.0 um aside, 1.68 um deeper"] = (2.0, 1.68, 0.0)
x = np.array([m.NODE["d"], m.NODE["mu"], m.NODE["v0"], m.NODE["z"], m.NB_D, m.NB_MU, 0.0, 0.0])
FREE = [0, 1, 2, 3, 4, 6, 7]                     # everything but mu_q (index 5)


def nb(t, scene):
    off, depth, heading = scene
    h = math.radians(heading)
    return m.G.Tube((-math.sin(h) * t[6], off + math.cos(h) * t[6], depth + t[7]), 0.5 * t[4], 0.0, h)


for name, scene in SCENES.items():
    model = lambda t: m.B_GL * m.tau([m.node_tube(t[0], t[2], t[3]), nb(t, scene)], [t[1], t[5]])
    J = np.array([(model(x + STEP * e) - model(x - STEP * e)).ravel() / (2 * STEP) for e in np.eye(8)]).T / m.SIGMA_GL
    J7 = J[:, FREE]
    se = np.sqrt(np.diag(np.linalg.inv(J7.T @ J7)))
    cond = np.linalg.cond(J7 / np.linalg.norm(J7, axis=0))
    shift = -np.linalg.solve(J7.T @ J7, J7.T @ (J[:, 5] * MU_ERR * x[5]))
    print("%s:\n  mu_q fixed: condition number %.1f; SE(d) %.4f um, SE(d_q) %.4f um (%.1f %%); mu_q held %.0f %% too "
          "high: node's d %+.2f %%, d_q %+.1f %%"
          % (name, cond, se[0], se[4], 100 * se[4] / x[4], 100 * MU_ERR, 100 * shift[0] / x[0], 100 * shift[4] / x[4]),
          flush=True)
