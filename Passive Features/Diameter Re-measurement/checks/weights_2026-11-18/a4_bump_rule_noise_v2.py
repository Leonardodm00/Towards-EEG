#!/usr/bin/env python3
"""Standalone check (NOT pipeline code), 2026-11-18, follow-up of a4_bump_rule_noise.py (same profiles, noise and
renderer): the central dip taken as the DEEPEST smoothed minimum within FWHM_A of v_c = 0 (not the nearest
minimum), a bump = another minimum with prominence >= m sigma at least FWHM_A from v_c; smoothing 1 sample.
Absorption 0.2 /um (the clear nodes' fitted values, D-169's data) and 0.5 /um.

Run:  python3 a4_bump_rule_noise_v2.py "<repo>/Passive Features/Diameter Re-measurement/src"
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
RNG = np.random.default_rng(20261119)
N_DRAWS = 400


def flagged(I, sigma, mult, fw):
    for row in I:
        x = gaussian_filter1d(row, 1.0, mode="nearest")
        idx, props = find_peaks(-x, prominence=0.0)
        if idx.size == 0:
            continue
        near = np.flatnonzero(np.abs(m.V[idx]) <= fw)
        c = int(near[np.argmin(x[idx[near]])]) if near.size else int(np.argmin(x[idx]))
        if any(j != c and props["prominences"][j] >= mult * sigma and abs(m.V[idx[j]]) >= fw for j in range(idx.size)):
            return True
    return False


if __name__ == "__main__":
    t0 = time.time()
    for MU in (0.2, 0.5):
        for D in (0.45, 0.8, 1.2):
            fw = np.hypot(0.866 * D, 2.355 * m.S0)
            clean = m.model(np.array([D, MU, 0.0, 0.0]))
            bump = clean * m.tau(0.3, MU, -1.3, 0.0)
            depths = " ".join("%.0f" % (200 - r.min()) for r in clean)
            print("\nmu %.1f, node D %.2f um: dip by plane k=-4..4 (gl) %s; bump 0.3 um at -1.3 um: %.1f gl in focus"
                  % (MU, D, depths, 200 - bump[4][m.V < -0.65].min()))
            for sigma in (1.0, 2.0):
                noise = RNG.normal(0.0, sigma, (N_DRAWS,) + clean.shape)
                for mult in (2.0, 3.0):
                    fp = np.mean([flagged(clean + e, sigma, mult, fw) for e in noise])
                    tp = np.mean([flagged(bump + e, sigma, mult, fw) for e in noise[:100]])
                    print("  sigma %.1f gl, m %.0f: clean node flagged %5.1f %%, bump 0.3 um found %5.1f %%"
                          % (sigma, mult, 100 * fp, 100 * tp), flush=True)
    print("%.0f s" % (time.time() - t0))
