#!/usr/bin/env python3
"""Standalone check (NOT pipeline code), 2026-11-18: does a LINEARISED predictor reproduce the sweep's biases?

Predicted shift of the node's features when an error delta (gl, on the 9 planes x 53 samples) is left on its line:
    dtheta = (J^T J)^{-1} J^T delta,   J = d(B tau_node)/d theta at the truth (central differences, step 0.02).
Cleared with a wrong neighbour diameter: delta = B (S_wrong - S_true), the neighbour rendered alone.
Neighbour ignored: delta = -B S_true. Compared with the nonlinear fits of tier2_distance_sweep.py (its .out files).

Run:  python3 tier2_bias_predictor.py <src> <sweep .out> [<sweep .out> ...]
"""
import importlib.util
import os
import sys

import numpy as np

spec = importlib.util.spec_from_file_location("chk", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                  "tier2_neighbour_check.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
STEP = 0.02
x = np.array([m.NODE["d"], m.NODE["mu"], m.NODE["v0"], m.NODE["z"]])
alone = lambda t: m.B_GL * m.tau([m.node_tube(t[0], t[2], t[3])], [t[1]])
J = np.array([(alone(x + STEP * e) - alone(x - STEP * e)).ravel() / (2 * STEP) for e in np.eye(4)]).T
P = np.linalg.solve(J.T @ J, J.T)
rows = []
for path in sys.argv[2:]:
    with open(path) as f:
        rows += [[float(w) for w in line.split()] for line in f if line.strip() and not line.startswith("#")]
print("# L_um Z_um | cleared, neighbour d x %.2f: predicted %%  fitted %% | ignored: predicted %%  fitted %%" % m.NB_D_ERROR)
for r in sorted(rows):
    L, Z = r[0], r[1]
    scene = (L, Z, 0.0)
    s_true = m.B_GL * (1.0 - m.tau([m.nb_tube(m.NB_D, scene)], [m.NB_MU]))
    s_wrong = m.B_GL * (1.0 - m.tau([m.nb_tube(m.NB_D * m.NB_D_ERROR, scene)], [m.NB_MU]))
    pc = 100 * (P @ (s_wrong - s_true).ravel())[0] / x[0]
    pi = 100 * (P @ (-s_true).ravel())[0] / x[0]
    print("%4.1f %4.2f | %+8.1f %+8.1f | %+8.1f %+8.1f" % (L, Z, pc, r[8], pi, r[6]))
