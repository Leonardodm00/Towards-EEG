"""Reproducer (tester, 2026-11-14), finding F2: the per-node fit (Block 30,
through scripts/infer_nodes.py) and the stop rule of the cleaning (Block 32,
through scripts/clean_cell.py) take as sigma the noise SD of single pixels
(Block 36: sigma_gl = the median of the planes' noise SDs,
calibration.patch_planes), while the profiles they are applied to are Block
5's reads: bilinear interpolation and the mean over 2 n_along + 1 parallel
lines. The profile's noise SD is about 0.26 x the pixel's at the defaults, so
chi2/N is about 0.07 instead of 1 and every SE is about 3.9 times too large.
Part 1 (independent, white noise): pixel SD 3 gl -> profile SD.
Part 2: the synthetic dry run's inference CSV (if given): its mean chi2/N.

Run: python3 tests/spec/repro_block36_profile_sigma.py [inference_S.csv]
Pure ASCII.
"""
import csv
import math
import sys

import numpy as np

import nc_helpers
from allen_diameter.config import default_config
from allen_diameter.analysis import profiles


def main():
    cfg = default_config()
    m = cfg.measure
    v, n_avg, step = profiles.profile_offsets(m), profiles.n_along(m), float(m.profile_step_um)
    p = float(cfg.acquisition.res0_um)
    rng = np.random.default_rng(0)
    th = 0.3
    yh, eu = np.array([-math.sin(th), math.cos(th)]), np.array([math.cos(th), math.sin(th)])
    sds = []
    for _ in range(200):
        img = rng.normal(0.0, 3.0, (120, 120))
        o = np.array([60 * p, 60 * p]) + rng.uniform(0, p, 2)
        sds.append(nc_helpers.bilinear_read(img, 0, 0, p, o, yh, eu, v, n_avg, step, ).std())
    s = float(np.sqrt(np.mean(np.square(sds))))
    print("n_along %d, step %.4f um: pixel SD 3.00 gl -> profile SD %.3f gl; chi2/N with sigma = 3 gl: %.3f; "
          "SE inflation %.2f" % (n_avg, step, s, (s / 3.0) ** 2, 3.0 / s))
    if len(sys.argv) > 1:
        for r in csv.DictReader(open(sys.argv[1])):
            print("node %s: chi2_mean %s, status %s" % (r["node_id"], r["chi2_mean"], r["status"]))


if __name__ == "__main__":
    main()
