#!/usr/bin/env python3
"""Standalone check (NOT pipeline code), 2026-11-18, for D-171.3 (a near bump = a second dip separated from the
central one by a rise of at least m noise SDs, m = 2, tunable; searched in every used plane, D-171.7): how often does
the rule flag a bump on a CLEAN node from noise alone, and how often does it find a real one?

Profiles: one flat tube (D, mu 0.5 /um) rendered by the pipeline's fast line renderer (SPEC Block 35; hybrid, the
pilot's kernel table, 9 planes k = -4..4, 53 samples at 0.1144 um, one line), B = 200 gl, plus independent Gaussian
noise of SD sigma per sample (real profiles average parallel lines and their samples are correlated: not modelled).
Bump case: a parallel tube DB 0.6 um or 0.3 um, mu 0.5, 1.3 um to the left, at the node's depth, multiplied in.

Rule tested ("rise" read as the topographic prominence of the second minimum, scipy.signal.find_peaks on -I with
prominence >= m sigma, which is the depth of that minimum below the highest point it must cross to reach a deeper
one): per plane, the profile optionally smoothed with a Gaussian of 1 sample (the first stage's focus_smooth_px);
the central dip = the minimum nearest v_c = 0; a bump = any OTHER minimum with prominence >= m sigma and, optionally,
at least s_min from v_c. A node is flagged when any of its 9 planes has a bump.

Run:  python3 a4_bump_rule_noise.py "<repo>/Passive Features/Diameter Re-measurement/src"
"""
import importlib.util
import os
import time

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

spec = importlib.util.spec_from_file_location("wb", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                 "a1_a5_weights_bump.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
RNG = np.random.default_rng(20261118)
N_DRAWS = 400


def flagged(I, sigma, mult, smooth_px, s_min):
    for row in I:
        x = gaussian_filter1d(row, smooth_px, mode="nearest") if smooth_px > 0 else row
        idx, props = find_peaks(-x, prominence=0.0)
        if idx.size == 0:
            continue
        c = int(np.argmin(np.abs(m.V[idx])))
        others = [j for j in range(idx.size) if j != c and props["prominences"][j] >= mult * sigma
                  and abs(m.V[idx[j]]) >= s_min]
        if others:
            return True
    return False


if __name__ == "__main__":
    t0 = time.time()
    MU = 0.5
    for D in (0.45, 0.8, 1.2):
        fw = np.hypot(0.866 * D, 2.355 * m.S0)
        clean = m.model(np.array([D, MU, 0.0, 0.0]))
        bumps = {DB: clean * m.tau(DB, MU, -1.3, 0.0) for DB in (0.6, 0.3)}
        print("\nnode D %.2f um (FWHM_A %.2f um), in-focus dip %.1f gl; bump 0.6 / 0.3 um at -1.3 um: dips %.1f / %.1f gl"
              % (D, fw, 200 - clean[4].min(), 200 - (clean[4] * 0 + bumps[0.6][4])[m.V < -0.65].min(),
                 200 - bumps[0.3][4][m.V < -0.65].min()))
        for sigma in (1.0, 2.0):
            noise = RNG.normal(0.0, sigma, (N_DRAWS,) + clean.shape)
            for mult in (2.0, 3.0, 4.0):
                for smooth_px, s_min in ((0, 0.0), (1, 0.0), (1, fw)):
                    fp = np.mean([flagged(clean + e, sigma, mult, smooth_px, s_min) for e in noise])
                    tp6 = np.mean([flagged(bumps[0.6] + e, sigma, mult, smooth_px, s_min) for e in noise[:100]])
                    tp3 = np.mean([flagged(bumps[0.3] + e, sigma, mult, smooth_px, s_min) for e in noise[:100]])
                    print("  sigma %.1f gl, m %.0f, smooth %d px, s_min %.2f um: clean node flagged %5.1f %%;"
                          " bump 0.6 found %5.1f %%, bump 0.3 found %5.1f %%"
                          % (sigma, mult, smooth_px, s_min, 100 * fp, 100 * tp6, 100 * tp3), flush=True)
    print("%.0f s" % (time.time() - t0))
