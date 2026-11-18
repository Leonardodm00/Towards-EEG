"""Timing check (2026-11-18, D-170): how long survey_pool.pbs's calibration analysis
(survey_cells.pooled_fits: Block 10's near fit with its anchor bootstrap, Block 15's
far fit with its bootstrap and anchor sensitivity, Block 19.8's cell test) takes when
the pool holds C cells instead of 51 -- to set the pool job's walltime (2 h in its
header) before the run over all the eligible cells.

Synthetic pooled scans, made with the smoke suite's generators (test_smoke_s3c):
  near: per cell n_near nodes, omega = c + a_s 0.3 (z_ax - z_k)^2 + noise (SD 0.003
        um^2), 7 planes, a_s ~ N(1, 0.1^2) per cell (test_smoke_s3c.near_cells);
  far:  per cell n_far nodes, omega = c + (sigma0 + gamma_s x)^2 exactly on 33
        planes (test_smoke_s3c.far_scan), sigma0 the near law's anchor, gamma_s ~
        N(0.40, 0.05^2) per cell.
The pooled path is the pool job's own (calibration_pool.pool_scans, then
survey_cells.pooled_fits with figures off), at B bootstrap resamples; the time at
B = 1000 is extrapolated from two values of B (the cost is affine in B).

Run from the workstream folder (Passive Features/Diameter Re-measurement):
    python3 /path/to/pool_timing.py --cells 158 --near 30 --far 2
"""
import argparse
import contextlib
import dataclasses
import io
import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
sys.path.insert(0, "tests/smoke")
_argv, sys.argv = sys.argv, ["x"]
import test_smoke_s3c as T  # noqa: E402
sys.argv = _argv
import survey_cells as SC  # noqa: E402
from allen_diameter.analysis import calibration as CAL, calibration_pool as CP  # noqa: E402
from allen_diameter.loading import calibration_io  # noqa: E402
from allen_diameter.config import default_config  # noqa: E402


def per_cell_scans(cfg, n_cells, n_near, n_far, rng):
    near = T.near_cells(n_cells, n_near, 0.1, rng)            # qualified ids s x 10^6 + n
    sig0 = float(np.sqrt(cfg.renderer.sigma_r0_um ** 2 + 0.3 * cfg.calibration.table_max_delta_um ** 2))
    cells = {}
    for sc in near:
        s, n = CP.cell_of(sc.node_id)
        cells.setdefault(s, []).append(dataclasses.replace(sc, node_id=n))
    for s in range(1, n_cells + 1):
        g = 0.40 + 0.05 * rng.normal()
        for n in range(n_far):
            cells[s].append(T.far_scan(10000 + n, cfg, g, sig0, rng))
    return sorted(cells.items()), sig0


def run_once(cfg, per_cell, B):
    cfg_b = dataclasses.replace(cfg, calibration=dataclasses.replace(cfg.calibration, bootstrap_resamples=B))
    with tempfile.TemporaryDirectory() as tmp:
        pooled, meta = CP.pool_scans(per_cell, cfg_b)
        s3c = os.path.join(tmp, "s3c")
        os.makedirs(s3c)
        calibration_io.save_scans(pooled, os.path.join(s3c, "scans_pooled.json"), meta)
        buf = io.StringIO()
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(buf):
            done, outliers = SC.pooled_fits(cfg_b, pooled, meta, s3c, log=lambda m: print(m), figures=False)
        dt = time.perf_counter() - t0
        removed = {w: outliers[w]["removed"] for w in ("near", "far")} if outliers else None
        return dt, done, removed, buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=int, default=158)
    ap.add_argument("--near", type=int, default=30)
    ap.add_argument("--far", type=int, default=2)
    ap.add_argument("--b", type=int, nargs=2, default=(20, 60))
    ap.add_argument("--seed", type=int, default=20261118)
    a = ap.parse_args()
    cfg = default_config()
    rng = np.random.default_rng(a.seed)
    per_cell, sig0 = per_cell_scans(cfg, a.cells, a.near, a.far, rng)
    print("pool: %d cells x (%d near, %d far) scans; synthetic anchor %.4f um" % (a.cells, a.near, a.far, sig0))
    times = []
    for B in a.b:
        dt, done, removed, out = run_once(cfg, per_cell, B)
        times.append(dt)
        print("  B = %4d resamples: %.1f s; done %s; removed %s" % (B, dt, done, removed))
    slope = (times[1] - times[0]) / (a.b[1] - a.b[0])
    fixed = times[0] - slope * a.b[0]
    t1000 = fixed + slope * 1000
    print("  affine in B: fixed %.1f s + %.3f s per resample -> B = 1000: %.0f s = %.2f h (one pass; a removal of "
          "cells by Block 19.8 runs both fits again)" % (fixed, slope, t1000, t1000 / 3600))
    print("  last run's log, first lines:")
    for line in out.splitlines()[:12]:
        print("    " + line[:200])


if __name__ == "__main__":
    main()
