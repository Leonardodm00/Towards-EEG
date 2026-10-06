#!/usr/bin/env python3
"""End-to-end check of the correction (procedure s.3.10, last row; gate 2).

  phantoms  render and measure test phantoms at fixed diameters, with the
            other draws of the design (tilt, heading, mu, offsets) from a seed
            independent of the table's; rows to --out (CSV)
  evaluate  invert every retained test phantom with a bias table (its
            line-fit tilt, as at a real node) and report d_tilde / d per
            diameter; the gate passes when every mean is within --tol of 1

The test phantoms come from the same renderer as the table unless another
generator is chosen (the ray world, Block 9); with the same generator the
check tests the Monte Carlo, the estimator and the inversion, not the
simulator (procedure s.3.10).

Examples
    python scripts/build_table.py run --start 0 --stop 300 --workers 2 --out-dir T --config-json gate2.json
    python scripts/build_table.py merge --out-dir T --config-json gate2.json
    python scripts/end_to_end.py phantoms --d 0.5,1,2,3 --reps 8 --seed 777 --out T/test.csv --config-json gate2.json
    python scripts/end_to_end.py evaluate --table T/bias_table_<hash> --rows T/test.csv --config-json gate2.json

Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from allen_diameter.analysis import invert, phantoms  # noqa: E402
from allen_diameter.loading import table_io  # noqa: E402
from build_table import load_config  # noqa: E402


def _one(args):
    cfg, seed, index, d = args
    row = phantoms.run_replicate(cfg, seed, index, d_override_um=d)
    row["d_target_um"] = d
    return row


def run_phantoms(cfg, diameters, reps, seed, out, workers=1):
    jobs = [(cfg, seed, k * reps + r, d) for k, d in enumerate(diameters) for r in range(reps)]
    if workers > 1:
        import multiprocessing
        with multiprocessing.Pool(workers) as pool:
            rows = pool.map(_one, jobs, chunksize=1)
    else:
        rows = [_one(j) for j in jobs]
    table_io.write_rows(rows, out)
    print("%d test phantoms -> %s" % (len(rows), out))


def evaluate(cfg, table_stem, rows_path, tol):
    table = table_io.load_table(table_stem)
    table.check_estimator(cfg)
    rows = table_io.read_rows(rows_path)
    ok_all = True
    print("d_um   n  in_S  inverted  mean(d_tilde/d)  sd     mean(d_hat/d)  verdict")
    for d in sorted({float(r["d_target_um"]) for r in rows}):
        sub = [r for r in rows if float(r["d_target_um"]) == d]
        kept = [r for r in sub if r["in_S"]]
        inv = [invert.invert_node(r["d_hat_um"], r["meas_phi_rad"], table, cfg) for r in kept]
        ratios = np.array([x.d_tilde_um / d for x in inv if math.isfinite(x.d_tilde_um)])
        raw = np.array([r["d_hat_um"] / d for r in kept])
        mean = float(ratios.mean()) if ratios.size else float("nan")
        good = ratios.size > 0 and abs(mean - 1) <= tol
        ok_all &= good
        print("%-5.2f %3d %5d %9d %16.4f %6.4f %14.4f  %s" % (d, len(sub), len(kept), ratios.size, mean,
              float(ratios.std()) if ratios.size else float("nan"), float(raw.mean()) if raw.size else float("nan"),
              "PASS" if good else "FAIL"))
    print("gate 2: %s (tolerance %g)" % ("PASS" if ok_all else "FAIL", tol))
    return ok_all


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("phantoms")
    p.add_argument("--d", required=True, help="comma-separated diameters (um)")
    p.add_argument("--reps", type=int, default=8)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=1)
    e = sub.add_parser("evaluate")
    e.add_argument("--table", required=True, help="table stem (without .npz/.json)")
    e.add_argument("--rows", required=True)
    e.add_argument("--tol", type=float, default=0.10)
    for q in (p, e):
        q.add_argument("--config-json", default="")
    a = ap.parse_args(argv)
    cfg = load_config(a.config_json)
    if a.cmd == "phantoms":
        run_phantoms(cfg, [float(x) for x in a.d.split(",")], a.reps, a.seed, a.out, a.workers)
        return 0
    return 0 if evaluate(cfg, a.table, a.rows, a.tol) else 1


if __name__ == "__main__":
    sys.exit(main())
