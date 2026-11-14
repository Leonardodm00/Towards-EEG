"""Reproducer and check (testers, 2026-11-14; rewritten at 18:30 against Block 36 amendment 2, F2, D-142).
The earlier version showed finding F2 of the 11:00 report: the fits were given the pixel's noise SD while
their residuals are Block 5's profile samples (bilinear, mean over 2 n_along + 1 lines), so chi2/N was about
0.07. Amendment 2 hands on sigma_gl = sigma_pixel x profile_noise_factor. Here:
Part 1 (independent, white noise): the profile SD of 400 seeded white-noise images of pixel SD 3 gl, read
with nc_helpers' bilinear read (scipy.ndimage), against 3 x node_inputs.profile_noise_factor (the
amendment's oracle: within 3 %).
Part 2: an inference CSV of infer_nodes.py (if given): its mean chi2/N per node (white noise -> about 1;
JPEG noise is not white, so the factor is a lower bound of the profile's SD and chi2/N may exceed 1).

Run: python3 tests/spec/repro_block36_profile_sigma.py [inference_S.csv]   (from the code folder)
Pure ASCII.
"""
import csv
import math
import sys

import numpy as np

import nc_helpers
from allen_image_io import CropFrame
from allen_diameter.config import default_config
from allen_diameter.analysis import node_inputs, profiles


def main():
    cfg = default_config()
    m = cfg.measure
    v, n_avg, step = profiles.profile_offsets(m), profiles.n_along(m), float(m.profile_step_um)
    p = float(cfg.acquisition.res0_um)
    rng = np.random.default_rng(0)
    th = 0.3
    yh, eu = np.array([-math.sin(th), math.cos(th)]), np.array([math.cos(th), math.sin(th)])
    o = np.array([60 * p, 60 * p]) + np.array([0.037, 0.071])
    fac = node_inputs.profile_noise_factor(CropFrame(0, 0, 0, p), (120, 120), o, yh, eu, v, n_avg, step)
    prof = np.array([nc_helpers.bilinear_read(rng.normal(0.0, 3.0, (120, 120)), 0, 0, p, o, yh, eu, v, n_avg, step)
                     for _ in range(400)])
    s = float(np.sqrt(np.mean(prof ** 2)))
    print("n_along %d, step %.4f um: pixel SD 3.00 gl -> profile SD %.4f gl (MC, 400 images); 3 x factor %.4f gl "
          "(%+.2f %%); chi2/N with sigma = 3 x factor: %.3f; with the pixel SD (before F2): %.3f"
          % (n_avg, step, s, 3.0 * fac, 100.0 * (s / (3.0 * fac) - 1.0), (s / (3.0 * fac)) ** 2, (s / 3.0) ** 2))
    if len(sys.argv) > 1:
        for r in csv.DictReader(open(sys.argv[1])):
            print("node %s: chi2_mean %s, status %s" % (r["node_id"], r.get("chi2_mean"), r["status"]))


if __name__ == "__main__":
    main()
