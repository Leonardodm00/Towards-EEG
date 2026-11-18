#!/usr/bin/env python3
"""Standalone (NOT pipeline code): Gauss-Newton standard error of the node's diameter at the truth, noise SD 1 gl,
for the one-tube model without a neighbour, the joint model with the neighbour held, and the joint model with the
neighbour's d and mu free. Central differences, steps 0.02 (um or 1/um). Uses tier2_neighbour_check.py's set-up.
Run:  python3 tier2_neighbour_se.py <src>"""
import importlib.util
import os
import sys

import numpy as np

spec = importlib.util.spec_from_file_location("chk", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                  "tier2_neighbour_check.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
STEP = 0.02
x = np.array([m.NODE["d"], m.NODE["mu"], m.NODE["v0"], m.NODE["z"], m.NB_D, m.NB_MU])


def jac(model, n):
    cols = []
    for a in range(n):
        e = np.zeros(6)
        e[a] = STEP
        cols.append((model(x + e) - model(x - e)).ravel() / (2 * STEP))
    return np.array(cols).T / m.SIGMA_GL


def se(J):
    C = np.linalg.inv(J.T @ J)
    return np.sqrt(np.diag(C)), C


alone = lambda t: m.B_GL * m.tau([m.node_tube(t[0], t[2], t[3])], [t[1]])
s0, _ = se(jac(alone, 4))
print("one tube, no neighbour in the scene: SE(d) %.4f um (%.2f %%), SE(mu) %.4f /um" % (s0[0], 100 * s0[0] / x[0], s0[1]))
for name, scene in m.SCENES.items():
    joint = lambda t: m.B_GL * m.tau([m.node_tube(t[0], t[2], t[3]), m.nb_tube(t[4], scene)], [t[1], t[5]])
    J = jac(joint, 6)
    s4, _ = se(J[:, :4])
    s6, C6 = se(J)
    r = C6 / np.outer(s6, s6)
    print("%s:\n  joint, neighbour held: SE(d) %.4f um (%.2f %%)\n  joint, neighbour d and mu free: SE(d) %.4f um (%.2f %%); "
          "neighbour SE(d_q) %.4f um; corr(d, d_q) %+.2f, corr(d, mu_q) %+.2f, corr(d, mu) %+.2f"
          % (name, s4[0], 100 * s4[0] / x[0], s6[0], 100 * s6[0] / x[0], s6[4], r[0, 4], r[0, 5], r[0, 1]))
