"""Timing (2026-11-04, D-118.3): the far fit (15.4) and its cell bootstrap (19.6) at the davinci pool's size, 51
cells x 2 far nodes, on exact synthetic far scans (test_smoke_s3c.far_scan: omega = c + (sigma0 + gamma x)^2, one
slope per cell drawn around 0.79 with SD 0.1). Run from the workstream folder:
    python3 checks/d118_2026-11-04/far_fit_timing.py
"""
import dataclasses
import sys
import time

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "tests/smoke")
sys.argv = ["x"]
import test_smoke_s3c as T  # noqa: E402
from allen_diameter.analysis import calibration_pool as CP, far_field as FF  # noqa: E402
from allen_diameter.config import default_config  # noqa: E402

d = default_config()
cfg = dataclasses.replace(d, calibration=dataclasses.replace(d.calibration, calib_min_nodes_per_set=2))
rng = np.random.default_rng(20261104)
gam = {s: 0.79 + 0.1 * rng.normal() for s in range(1, 52)}
far = [dataclasses.replace(T.far_scan(1, cfg, gam[s], 0.6, rng), node_id=s * CP.QUALIFIER + n)
       for s in range(1, 52) for n in range(2)]
sel = FF.select_far_planes(far, cfg)
t0 = time.perf_counter()
fit = FF.fit_far_slope(sel, 0.6, cfg)
t_fit = time.perf_counter() - t0
t0 = time.perf_counter()
boot = FF.gamma_se(sel, 0.6, cfg, np.random.default_rng(1), n_resamples=20)
t_boot = (time.perf_counter() - t0) / 20
print("far set: %d nodes, %d cells, %d planes; one far fit %.2f s; one cell-bootstrap resample %.2f s "
      "(x 1000 ~ %.1f min); rule %s, SE %.4f" % (sel.node_ids.size, boot["n_cells"], sel.omega_um2.size, t_fit, t_boot,
                                                  t_boot * 1000 / 60, boot["rule"], boot["se"]))
