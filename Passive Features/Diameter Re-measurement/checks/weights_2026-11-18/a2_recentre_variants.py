#!/usr/bin/env python3
"""Standalone check (NOT pipeline code), 2026-11-18, for D-171.2 ("v_c re-centred while fitting"): two readings.
  IRLS        the weights are held fixed within each least-squares solve and re-centred on the solve's v0 between
              solves (D-052's scheme), until d and v0 move by < 1e-4 um (at most 15 solves)
  in-solve    v_c = v0 of the current iterate inside the objective, so the weights move with v0 during one solve
Weights exp(-|v - v_c| / lambda), lambda = 1.25 FWHM_A (D-171.7). Same renderer, kernel table, line and start as
a1_a5_weights_bump.py (imported). Cases: (a) a clean node (the model exact); (b) the node with a bump (a parallel tube
0.6 um, 1.3 um to the left, in focus); (c) a clean node whose TRUE in-focus kernel is 0.18 um while the model keeps
0.080 um (a misfit at the core, as on the real clear nodes); noise-free.

Run:  python3 a2_recentre_variants.py "<repo>/Passive Features/Diameter Re-measurement/src"
"""
import dataclasses
import importlib.util
import os
import time

import numpy as np
from scipy.optimize import least_squares

spec = importlib.util.spec_from_file_location("wb", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                 "a1_a5_weights_bump.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
from allen_diameter.model import geometry as G, line_fast as LF        # noqa: E402  (path set by the import above)


def in_solve(data, D, lam):
    def r(th):
        w = np.exp(-np.abs(m.V - th[2]) / lam)
        return (np.sqrt(w)[None, :] * (m.model(th) - data)).ravel()
    res = least_squares(r, np.array([D, 0.5, 0.0, 0.0]), bounds=(m.LO, m.HI), method="trf", x_scale="jac",
                        diff_step=1e-3, max_nfev=300)
    return res.x


def wide_kernel_data(D, MU, s0):
    extra = s0 ** 2 - m.SIGMA_T[0] ** 2
    sig = tuple(float(np.sqrt(s ** 2 + extra)) for s in m.SIGMA_T)
    cfg = dataclasses.replace(m.CFG, renderer=dataclasses.replace(m.CFG.renderer, kernel_table_sigma_um=sig,
                                                                  sigma_r0_um=sig[0]))
    tube = G.Tube((m.O0[0], m.O0[1], 0.0), 0.5 * D, 0.0, 0.0, 1.0, 10.0, "axial")
    return m.B_GL * LF.line_profiles(tube, MU, m.KS, m.O0, m.YH, m.EU, m.V, 0, m.P, cfg, "hybrid", LF.SectionCache())


if __name__ == "__main__":
    t0 = time.time()
    MU = 0.5
    for D in (0.45, 0.8):
        lam = 1.25 * np.hypot(0.866 * D, 2.355 * m.S0)
        clean = m.model(np.array([D, MU, 0.0, 0.0]))
        cases = (("(a) clean", clean), ("(b) bump", clean * m.tau(0.6, MU, -1.3, 0.0)),
                 ("(c) kernel 0.18", wide_kernel_data(D, MU, 0.18)))
        print("\nnode D %.2f um, mu %.2f, lambda %.3f um" % (D, MU, lam), flush=True)
        for name, data in cases:
            xi, ni = m.fit(data, D, "R1")
            xs = in_solve(data, D, lam)
            print("  %-16s IRLS: d/D %.3f, v0 %+.3f um (%d solves) | in-solve: d/D %.3f, mu %.2f, v0 %+.3f um"
                  % (name, xi[0] / D, xi[2], ni, xs[0] / D, xs[1], xs[2]), flush=True)
    print("%.0f s" % (time.time() - t0))
