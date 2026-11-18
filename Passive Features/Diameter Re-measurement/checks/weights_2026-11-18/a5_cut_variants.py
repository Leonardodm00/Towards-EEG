#!/usr/bin/env python3
"""Standalone check (NOT pipeline code), 2026-11-18, for choice A5 of the joint-fit handoff (section 8b): why the
cut at the saddle (a1_a5_weights_bump.py) leaves a bias, and what removes it. Same truths, renderer, kernel, line and
start as a1_a5_weights_bump.py (imported from it); no centre weights (they add little once the cut is applied there).

  tail     the bump's shadow on the node's side of the saddle, per plane: max over v beyond the saddle of
           B (1 - tau_bump,k(v)), gl
  cut|k|<=K  the cut of a1_a5 (samples from the saddle outwards get weight 0) and only the planes |k| <= K
  2 tubes  no cut, all 9 planes; the model is two parallel flat tubes (product of transmittances), the bump's
           (d_b, mu_b, v_b, z_b) free beside the node's four, started at d_b = 0.5, mu_b = 0.5, v_b at the bump's
           minimum in plane 0, z_b = 0

Run:  python3 a5_cut_variants.py "<repo>/Passive Features/Diameter Re-measurement/src"
"""
import importlib.util
import os
import sys
import time

import numpy as np
from scipy.optimize import least_squares

spec = importlib.util.spec_from_file_location("wb", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                 "a1_a5_weights_bump.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def solve_planes(data, mask, start, kmax):
    rows = np.abs(m.KS) <= kmax
    sw = np.sqrt(mask)[None, :]
    res = least_squares(lambda th: (sw * (m.model(th)[rows] - data[rows])).ravel(), start, bounds=(m.LO, m.HI),
                        method="trf", x_scale="jac", diff_step=1e-3, max_nfev=200)
    return res.x


def two_tubes(data, D, vb0):
    lo = np.concatenate([m.LO, [0.1, 0.01, -2.5, -1.5]])
    hi = np.concatenate([m.HI, [3.0, 10.0, 2.5, 1.5]])

    def r(th):
        return (m.B_GL * m.tau(*th[:4]) * m.tau(*th[4:]) - data).ravel()
    res = least_squares(r, np.array([D, 0.5, 0.0, 0.0, 0.5, 0.5, vb0, 0.0]), bounds=(lo, hi), method="trf",
                        x_scale="jac", diff_step=1e-3, max_nfev=400)
    return res.x


if __name__ == "__main__":
    t0 = time.time()
    MU = 0.5
    for D, DB, BV, BZ in ((0.8, 0.6, -1.3, 0.0), (0.45, 0.6, -1.3, 0.0), (0.8, 0.6, -1.3, 0.84)):
        node = m.model(np.array([D, MU, 0.0, 0.0]))
        bump = m.tau(DB, MU, BV, BZ)
        data = node * bump
        line0 = data[4]
        i_core = int(np.argmin(np.where(m.V > BV / 2, line0, np.inf)))
        i_bump = int(np.argmin(np.where(m.V < BV / 2, line0, np.inf)))
        lo, hi = sorted((i_core, i_bump))
        i_sad = lo + int(np.argmax(line0[lo:hi + 1]))
        keep = m.V > m.V[i_sad]
        tail = (m.B_GL * (1.0 - bump[:, keep])).max(axis=1)
        print("\nnode D %.2f, bump DB %.2f at v %+.2f, depth %+.2f; saddle at v %+.2f. Bump's tail beyond the saddle "
              "by plane k=-4..4 (gl): %s" % (D, DB, BV, BZ, m.V[i_sad], " ".join("%.1f" % t for t in tail)), flush=True)
        th = np.array([D, 0.5, 0.0, 0.0])
        for K in (4, 2, 1):
            x = solve_planes(data, keep.astype(float), th, K)
            print("  cut, planes |k| <= %d: d/D %.3f (mu %.2f, v0 %+.3f, z %+.3f)" % (K, x[0] / D, x[1], x[2], x[3]),
                  flush=True)
            xc = solve_planes(node, keep.astype(float), th, K)
            print("     control without bump: d/D %.3f" % (xc[0] / D), flush=True)
        x = two_tubes(data, D, float(m.V[i_bump]))
        print("  2 tubes, all planes: d/D %.3f (mu %.2f, v0 %+.3f, z %+.3f); bump d_b %.3f (true %.2f), mu_b %.2f, "
              "v_b %+.3f, z_b %+.3f" % (x[0] / D, x[1], x[2], x[3], x[4], DB, x[5], x[6], x[7]), flush=True)
    print("%.0f s" % (time.time() - t0))
