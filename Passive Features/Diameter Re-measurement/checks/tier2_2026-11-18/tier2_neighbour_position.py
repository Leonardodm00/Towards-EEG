#!/usr/bin/env python3
"""Standalone check (NOT pipeline code), 2026-11-18: should the neighbour's position be free in the joint fit?

Set-up of tier2_neighbour_check.py. Parameters: the node's (d, mu, v0, z_ax) and the neighbour's
(d_q, mu_q, v0_q, z_q), v0_q its lateral shift across its own axis (um), z_q its depth (um). From the Jacobian at
the truth (central differences, step 0.02), noise SD 1 gl:
  SE(d) with 6 free parameters (the neighbour's position held) and with 8 (its position free too);
  the first-order shift of the node's fitted d when the neighbour's position is HELD at a wrong value:
      dtheta_6 = -(J6^T J6)^{-1} J6^T (J_pos * err),   err = ERR_UM in v0_q, resp. in z_q.

Run:  python3 tier2_neighbour_position.py <src>
"""
import importlib.util
import math
import os
import sys

import numpy as np

spec = importlib.util.spec_from_file_location("chk", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                  "tier2_neighbour_check.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
STEP, ERR_UM = 0.02, 0.2
SCENES = dict(m.SCENES)
SCENES["parallel, directly above, 1.68 um deeper"] = (0.0, 1.68, 0.0)
SCENES["crossing at 45 deg over the node, 2.1 um deeper"] = (0.0, 2.1, 45.0)
x = np.array([m.NODE["d"], m.NODE["mu"], m.NODE["v0"], m.NODE["z"], m.NB_D, m.NB_MU, 0.0, 0.0])
NAMES = ("d", "mu", "v0", "z", "d_q", "mu_q", "v0_q", "z_q")


def nb(t, scene):
    off, depth, heading = scene
    h = math.radians(heading)
    c = (-math.sin(h) * t[6], off + math.cos(h) * t[6], depth + t[7])      # shifted across its own axis, and in depth
    return m.G.Tube(c, 0.5 * t[4], 0.0, h)


for name, scene in SCENES.items():
    model = lambda t: m.B_GL * m.tau([m.node_tube(t[0], t[2], t[3]), nb(t, scene)], [t[1], t[5]])
    J = np.array([(model(x + STEP * e) - model(x - STEP * e)).ravel() / (2 * STEP) for e in np.eye(8)]).T / m.SIGMA_GL
    out = []
    for n in (6, 8):
        C = np.linalg.inv(J[:, :n].T @ J[:, :n])
        se = np.sqrt(np.diag(C))
        out.append((se, C / np.outer(se, se)))
    (se6, _), (se8, r8) = out
    P6 = np.linalg.solve(J[:, :6].T @ J[:, :6], J[:, :6].T)
    shift_v = -(P6 @ (J[:, 6] * ERR_UM))[0]
    shift_z = -(P6 @ (J[:, 7] * ERR_UM))[0]
    print("%s:\n  SE(d): position held %.2f %%, position free %.2f %%; SE(v0_q) %.3f um, SE(z_q) %.3f um; "
          "largest |corr| of d with the neighbour's four: %.2f (%s)\n  neighbour's position held %.1f um wrong "
          "(first order): d shifts %+.1f %% (lateral error), %+.1f %% (depth error)"
          % (name, 100 * se6[0] / x[0], 100 * se8[0] / x[0], se8[6], se8[7], np.abs(r8[0, 4:]).max(),
             NAMES[4 + int(np.abs(r8[0, 4:]).argmax())], ERR_UM, 100 * shift_v / x[0], 100 * shift_z / x[0]), flush=True)
