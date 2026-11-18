#!/usr/bin/env python3
"""Standalone check (NOT pipeline code), 2026-11-18: how far must a neighbour be for SUBTRACTING its shadow (the
cleaning, tier 1) to be good enough, when the diameter used for it is wrong?

Set-up of tier2_neighbour_check.py (node and neighbour 0.6 um, mu 0.5 /um, flat, hybrid, the pilot's kernel table,
noise-free). The neighbour is PARALLEL to the node, at lateral offset L (um) and depth offset Z (um). For each (L, Z):
  ignored   one-tube fit, the neighbour left in the data
  cleared   the neighbour's shadow rendered alone with its diameter x NB_D_ERROR added back, then the one-tube fit
The bias of the node's fitted diameter is reported in % of the truth; SE(d) at 1 gl noise is 0.77 % (tier2_neighbour_se.py).

Run:  python3 tier2_distance_sweep.py <src> [first|second]      (the two halves of the list, for two processes)
"""
import importlib.util
import os
import sys

import numpy as np

spec = importlib.util.spec_from_file_location("chk", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                  "tier2_neighbour_check.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

POINTS = [(0.8, 0.0), (1.0, 0.0), (1.5, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0),      # (L um, Z um): same depth
          (0.0, 0.84), (0.0, 1.68), (0.0, 2.8), (0.0, 4.2),                              # directly above or below
          (1.0, 1.68), (2.0, 1.68)]                                                       # aside and out of focus
half = sys.argv[2] if len(sys.argv) > 2 else "all"
pts = POINTS[0::2] if half == "first" else POINTS[1::2] if half == "second" else POINTS
truth = np.array([m.NODE["d"], m.NODE["mu"], m.NODE["v0"], m.NODE["z"]])
x0 = truth * np.array([1.3, 0.8, 1.0, 1.0]) + np.array([0.0, 0.0, 0.05, 0.1])
lo, hi = np.array([0.1, 0.01, -1.5, -1.5]), np.array([6.0, 5.0, 1.5, 1.5])
print("# L_um Z_um  nb_peak_gl nb_range_gl  wrong_minus_true_peak_gl  d_ignored_um bias_ignored_%  d_cleared_um bias_cleared_%",
      flush=True)
for L, Z in pts:
    scene = (L, Z, 0.0)
    node = m.node_tube(m.NODE["d"], m.NODE["v0"], m.NODE["z"])
    data = m.B_GL * m.tau([node, m.nb_tube(m.NB_D, scene)], [m.NODE["mu"], m.NB_MU])
    s_true = m.B_GL * (1.0 - m.tau([m.nb_tube(m.NB_D, scene)], [m.NB_MU]))
    s_wrong = m.B_GL * (1.0 - m.tau([m.nb_tube(m.NB_D * m.NB_D_ERROR, scene)], [m.NB_MU]))

    def one(th, target):
        return ((m.B_GL * m.tau([m.node_tube(th[0], th[2], th[3])], [th[1]]) - target) / m.SIGMA_GL).ravel()

    xa, _ = m.fit(lambda th: one(th, data), x0, lo, hi)
    xc, _ = m.fit(lambda th: one(th, data + s_wrong), x0, lo, hi)
    print("%4.1f %4.2f  %8.2f %8.2f  %8.2f  %8.3f %+8.1f  %8.3f %+8.1f" % (
        L, Z, s_true.max(), s_true.max() - s_true.min(), np.abs(s_true - s_wrong).max(),
        xa[0], 100 * (xa[0] / truth[0] - 1), xc[0], 100 * (xc[0] / truth[0] - 1)), flush=True)
