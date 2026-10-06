#!/usr/bin/env python3
"""Build the bias table b(d, phi | C) -- Block 6 in specs/SPEC.md.

Two steps:

  run    render and measure replicates [start, stop) of the random design and
         write their rows to <out-dir>/rows_<hash>_<start>_<stop>.csv; a PBS
         array runs one chunk per job (replicate n always draws from
         numpy.random.default_rng([seed, n]), so the chunking does not matter)
  merge  read every rows_<hash>_*.csv of <out-dir>, fit the table and write
         <out-dir>/bias_table_<hash>.npz/.json

<hash> = DiameterConfig.signature_hash("estimator"). The configuration is the
default one unless --config-json gives a full_signature() JSON (a labelled
variant: reduced ranges for the sandbox, another sigma_fit, another kernel).

Examples
    # sandbox, 40 replicates on 2 cores, then the table
    python scripts/build_table.py run --start 0 --stop 40 --workers 2 --out-dir tables/reduced
    python scripts/build_table.py merge --out-dir tables/reduced
    # davinci, one PBS array job per 20 replicates
    python scripts/build_table.py run --start $((20*PBS_ARRAY_INDEX)) --stop $((20*PBS_ARRAY_INDEX+20)) --out-dir "$OUT"

Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from allen_diameter.analysis import phantoms  # noqa: E402
from allen_diameter.analysis.table import fit_table  # noqa: E402
from allen_diameter.config import config_from_dict, default_config  # noqa: E402
from allen_diameter.loading import table_io  # noqa: E402


def load_config(path):
    if not path:
        cfg = default_config()
    else:
        with open(path) as f:
            cfg = config_from_dict(json.load(f))
    cfg.validate()
    return cfg


def _one(args):
    cfg, seed, index, backend = args
    return phantoms.run_replicate(cfg, seed, index, backend)


def run(cfg, start, stop, out_dir, workers=1, seed=None, backend=None):
    seed = cfg.phantom.seed if seed is None else int(seed)
    os.makedirs(out_dir, exist_ok=True)
    jobs = [(cfg, seed, n, backend) for n in range(int(start), int(stop))]
    t0 = time.perf_counter()
    if workers > 1:
        import multiprocessing
        with multiprocessing.Pool(workers) as pool:
            rows = pool.map(_one, jobs, chunksize=1)
    else:
        rows = [_one(j) for j in jobs]
    path = os.path.join(out_dir, "rows_%s_%06d_%06d.csv" % (cfg.signature_hash("estimator"), start, stop))
    table_io.write_rows(rows, path)
    kept = sum(1 for r in rows if r["in_S"])
    print("%d replicates (%d in S) in %.1f s -> %s" % (len(rows), kept, time.perf_counter() - t0, path))
    return path


def merge(cfg, out_dir):
    h = cfg.signature_hash("estimator")
    paths = sorted(glob.glob(os.path.join(out_dir, "rows_%s_*.csv" % h)))
    if not paths:
        sys.exit("no rows_%s_*.csv in %s" % (h, out_dir))
    rows = table_io.read_rows(paths)
    table = fit_table(rows, cfg)
    stem = os.path.join(out_dir, "bias_table_%s" % h)
    table_io.save_table(table, stem)
    print("%d rows from %d files, %d in S; smoothing %g -> %s.npz/.json"
          % (len(rows), len(paths), table.X_kept.shape[0], table.smoothing, stem))
    return stem


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--start", type=int, required=True)
    r.add_argument("--stop", type=int, required=True)
    r.add_argument("--workers", type=int, default=1)
    r.add_argument("--seed", type=int, default=None)
    r.add_argument("--backend", default=None, choices=(None, "fft", "direct"))
    for p in (r, sub.add_parser("merge")):
        p.add_argument("--out-dir", required=True)
        p.add_argument("--config-json", default="")
    a = ap.parse_args(argv)
    cfg = load_config(a.config_json)
    if a.cmd == "run":
        run(cfg, a.start, a.stop, a.out_dir, a.workers, a.seed, a.backend)
    else:
        merge(cfg, a.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
