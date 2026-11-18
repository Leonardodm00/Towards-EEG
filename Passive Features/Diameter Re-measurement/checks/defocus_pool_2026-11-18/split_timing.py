"""Timing (2026-11-18, D-175): which fit dominates the pooled bootstrap's cost at C cells -- one near growth fit
(Block 10, sparse Jacobian, D-119) and one far fit (Block 15.4, dense Jacobian) on the synthetic pooled scans of
pool_timing.py (30 near and 2 far scans per cell). Run from the workstream folder:
    python3 checks/defocus_pool_2026-11-18/split_timing.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pool_timing as P  # noqa: E402  (puts src, scripts and tests/smoke on the path)
from allen_diameter.analysis import calibration as CAL, calibration_pool as CP, far_field as FF  # noqa: E402
from allen_diameter.config import default_config  # noqa: E402

cfg = default_config()
for C in (51, 100, 158):
    rng = np.random.default_rng(1)
    per_cell, sig0 = P.per_cell_scans(cfg, C, 30, 2, rng)
    pooled, meta = CP.pool_scans(per_cell, cfg)
    near = [s for s in pooled if s.set == "near"]
    far = [s for s in pooled if s.set == "far"]
    t0 = time.perf_counter()
    CAL.fit_growth(near, cfg)
    tn = time.perf_counter() - t0
    sel = FF.select_far_planes(far, cfg)
    t0 = time.perf_counter()
    FF.fit_far_slope(sel, sig0, cfg)
    tf = time.perf_counter() - t0
    print("C = %3d: near %4d nodes, one growth fit %.2f s; far %3d nodes, one far fit %.2f s"
          % (C, len(near), tn, len(far), tf))
