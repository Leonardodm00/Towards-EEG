"""How often the near-bump rule of Block 39 cuts a clean profile, and how often it cuts a real bump (D-176.1).

Standalone check, not pipeline code. A clean central dip (Gaussian, SD 0.25 um, 40 gl in the sharpest plane,
fading with defocus) in K planes, white noise of SD SIGMA gl per sample, optionally a second dip 1.3 um aside
(Gaussian, SD 0.3 um) of the given depth in every plane. Rules compared, each with m = 2 noise SDs and the
central dip ending at half its depth:
  per plane   bump_cut on each plane's raw samples, any plane cutting
  mean, s     bump_cut on the mean dip over the planes, smoothed by a Gaussian of SD s line samples
Run:  python3 bump_false_positive_check.py > bump_false_positive_check.out
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from allen_diameter.analysis import line_weights as L  # noqa: E402

SEED, N_DRAWS, SIGMA, M_SD, HALF = 20261118, 500, 2.5, 2.0, 0.5
V = (np.arange(53) - 26) * 0.1144            # the pilot's line: 53 samples, pixel 0.1144 um
WINDOW = L.allen_fwhm_um(0.45, 0.080)
BUMPS_SD = (0.0, 2.0, 3.0, 4.0)              # the second dip's depth, in noise SDs


def scene(K, depth_gl):
    central = 40.0 * np.exp(-0.5 * ((np.arange(K) - K // 2) / 2.0) ** 2)
    return (200.0 - central[:, None] * np.exp(-0.5 * (V / 0.25) ** 2)[None]
            - depth_gl * np.exp(-0.5 * ((V - 1.3) / 0.3) ** 2)[None])


def main():
    rng = np.random.default_rng(SEED)
    print("cut rate over %d draws; noise SD %.1f gl; m = %.1f; columns: second dip of depth %s noise SDs"
          % (N_DRAWS, SIGMA, M_SD, ", ".join("%g" % b for b in BUMPS_SD)))
    for K in (1, 3, 5, 9):
        rates = {"per plane": [], "mean, s=0": [], "mean, s=1": [], "mean, s=1.5": []}
        for b in BUMPS_SD:
            count = dict.fromkeys(rates, 0)
            for _ in range(N_DRAWS):
                P = scene(K, b * SIGMA) + rng.normal(0.0, SIGMA, (K, V.size))
                cut = False
                for k in range(K):
                    lo, hi = L.bump_cut(200.0 - P[k], SIGMA, V, 0.0, WINDOW, M_SD, HALF)
                    cut = cut or np.isfinite(lo) or np.isfinite(hi)
                count["per plane"] += cut
                for s in (0.0, 1.0, 1.5):
                    keep, _lo, _hi = L.bump_mask(P, 200.0, SIGMA, V, 0.0, WINDOW, M_SD, HALF, s)
                    count["mean, s=%g" % s] += not keep.all()
            for name in rates:
                rates[name].append(count[name] / N_DRAWS)
        for name, r in rates.items():
            print("K = %d planes  %-12s %s" % (K, name, "  ".join("%.2f" % x for x in r)))


if __name__ == "__main__":
    main()
