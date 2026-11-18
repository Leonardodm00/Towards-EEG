#!/usr/bin/env python3
"""Standalone check (NOT pipeline code), 2026-11-18: does freeing the neighbour's position add sloppy directions?
Set-up of tier2_neighbour_position.py (Jacobian at the truth, central differences, noise 1 gl). For the node's
four parameters alone with the neighbour held (4), plus the neighbour's d and mu (6), plus its shift and depth (8):
the condition number of the Jacobian with columns scaled to unit norm, and the standard error of every parameter.
Run:  python3 tier2_sloppiness.py <src>"""
import importlib.util
import math
import os
import sys

import numpy as np

spec = importlib.util.spec_from_file_location("chk", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                  "tier2_neighbour_check.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
STEP = 0.02
SCENES = dict(m.SCENES)
SCENES["parallel, directly above, 1.68 um deeper"] = (0.0, 1.68, 0.0)
SCENES["crossing at 45 deg over the node, 2.8 um deeper"] = (0.0, 2.8, 45.0)
SCENES["parallel, 2.0 um aside, 1.68 um deeper"] = (2.0, 1.68, 0.0)
x = np.array([m.NODE["d"], m.NODE["mu"], m.NODE["v0"], m.NODE["z"], m.NB_D, m.NB_MU, 0.0, 0.0])
NAMES = ("d", "mu", "v0", "z", "d_q", "mu_q", "v0_q", "z_q")


def nb(t, scene):
    off, depth, heading = scene
    h = math.radians(heading)
    return m.G.Tube((-math.sin(h) * t[6], off + math.cos(h) * t[6], depth + t[7]), 0.5 * t[4], 0.0, h)


for name, scene in SCENES.items():
    model = lambda t: m.B_GL * m.tau([m.node_tube(t[0], t[2], t[3]), nb(t, scene)], [t[1], t[5]])
    J = np.array([(model(x + STEP * e) - model(x - STEP * e)).ravel() / (2 * STEP) for e in np.eye(8)]).T / m.SIGMA_GL
    print(name + ":", flush=True)
    for n in (4, 6, 8):
        Jn = J[:, :n]
        se = np.sqrt(np.diag(np.linalg.inv(Jn.T @ Jn)))
        cond = np.linalg.cond(Jn / np.linalg.norm(Jn, axis=0))
        print("  %d free: condition number %.1f; SE " % (n, cond) + ", ".join(
            "%s %.4f" % (NAMES[a], se[a]) for a in range(n)), flush=True)
