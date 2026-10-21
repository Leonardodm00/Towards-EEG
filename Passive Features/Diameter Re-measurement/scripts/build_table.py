#!/usr/bin/env python3
"""Build the bias table b(x | C, S) -- Blocks 6 and 14 in specs/SPEC.md.

Two steps:

  run    render and measure draws g = start .. stop - 1 (start >= 1), every
         realization r = 1 .. n_realizations of each (Block 14.2, D-059,
         D-062), and write their rows to
         <out-dir>/rows_<hash>_<start>_<stop>.csv; a PBS array runs one chunk
         of draws per task (a draw's features come from
         numpy.random.default_rng([seed, g]) and a realization's nuisances from
         default_rng([seed, g, r]), so the chunking does not matter)
  merge  read every rows_<hash>_*.csv of <out-dir> (only the columns the
         table reads), refuse rows of another simulation configuration and a
         (draw, realization) seen twice, report the draws with fewer than
         n_realizations rows, fit the table and write
         <out-dir>/bias_table_<hash>.npz/.json and the per-draw table
         <out-dir>/draws_<hash>.csv (Block 14.6; D-060.2). At 10^6 rows the
         fit takes of order 20 min and a few hundred MB (SPEC Block 14.4):
         run it in a job, not on a login node

<hash> = DiameterConfig.signature_hash("estimator"). The configuration is the
default one unless --config-json gives a full_signature() JSON (a labelled
variant: reduced ranges for the sandbox, another sigma_fit, another kernel).

Examples
    # sandbox, draws 1-8 (each with its n_realizations realizations) on 2 cores, then the table
    python scripts/build_table.py run --draw-start 1 --draw-stop 9 --workers 2 --out-dir tables/reduced
    python scripts/build_table.py merge --out-dir tables/reduced
    # davinci: scripts/pbs/build_table.pbs runs one chunk of draws per array task

Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from allen_diameter.analysis import phantoms  # noqa: E402
from allen_diameter.analysis.table import check_rows_config, fit_table, per_draw_table, row_provenance  # noqa: E402
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
    cfg, seed, g, r, backend = args
    return phantoms.run_replicate(cfg, seed, g, r, backend)


def run(cfg, start, stop, out_dir, workers=1, seed=None, backend=None):
    """Draws g = start .. stop - 1 (start >= 1), each with its realizations r = 1 .. n_realizations."""
    seed = cfg.phantom.seed if seed is None else int(seed)
    start, stop = phantoms.check_index("--draw-start", start), int(stop)
    if stop <= start:
        raise ValueError("--draw-stop (%d) must exceed --draw-start (%d)" % (stop, start))
    os.makedirs(out_dir, exist_ok=True)
    jobs = [(cfg, seed, g, r, backend) for g in range(start, stop)
            for r in range(1, int(cfg.phantom.n_realizations) + 1)]
    t0 = time.perf_counter()
    rows = []

    provenance = row_provenance(cfg)

    def note(row):
        row["sim_hash"] = provenance             # the merge refuses rows rendered under another configuration
        rows.append(row)
        print("draw %d realization %d: d %.3f um, phi %.1f deg -> d_hat/d %.4f, %s, %.1f s (%d/%d)"
              % (row["draw_index"], row["realization_index"], row["d_um"], row["phi_rad"] * 57.29577951308232,
                 row["ratio"],
                 "in S" if row["in_S"] else "out (%s)" % row["reject"], row["seconds"], len(rows), len(jobs)),
              flush=True)

    if workers > 1:
        import multiprocessing
        with multiprocessing.Pool(workers) as pool:
            for row in pool.imap(_one, jobs, chunksize=1):
                note(row)
    else:
        for j in jobs:
            note(_one(j))
    path = os.path.join(out_dir, "rows_%s_%06d_%06d.csv" % (cfg.signature_hash("estimator"), start, stop))
    table_io.write_rows(rows, path)
    kept = sum(1 for r in rows if r["in_S"])
    print("%d draws x %d realizations = %d rows (%d in S) in %.1f s -> %s"
          % (stop - start, int(cfg.phantom.n_realizations), len(rows), kept, time.perf_counter() - t0, path))
    return path


def merge(cfg, out_dir):
    """The table and the per-draw table of every rows_<hash>_*.csv of out_dir (see the module docstring)."""
    t0 = time.perf_counter()
    h = cfg.signature_hash("estimator")
    paths = sorted(glob.glob(os.path.join(out_dir, "rows_%s_*.csv" % h)))
    if not paths:
        sys.exit("no rows_%s_*.csv in %s" % (h, out_dir))
    cols = table_io.read_columns(paths)
    n = cols["draw_index"].size
    try:
        unchecked = check_rows_config(cols, cfg)
    except ValueError as exc:
        sys.exit("merge refused: %s" % exc)
    if unchecked:
        print("note: %d of %d rows carry no sim_hash (written before 2026-10-07): their configuration is not checked"
              % (unchecked, n))
    pairs, counts = np.unique(np.column_stack([cols["draw_index"], cols["realization_index"]]), axis=0,
                              return_counts=True)
    if np.any(counts > 1):
        dup = pairs[counts > 1]
        sys.exit("merge refused: %d (draw, realization) pairs occur more than once, e.g. %s: a chunk was run twice "
                 "or the rows of two runs share %s" % (dup.shape[0], [tuple(int(v) for v in x) for x in dup[:5]],
                                                       out_dir))
    draws, per = np.unique(cols["draw_index"], return_counts=True)
    n_real = int(cfg.phantom.n_realizations)
    short = draws[per < n_real]
    print("%d rows of %d draws from %d files (phantom.n_draws records %d)" % (n, draws.size, len(paths),
                                                                              int(cfg.phantom.n_draws)))
    if short.size:
        print("note: %d draws have fewer than n_realizations = %d rows, e.g. %s" % (short.size, n_real,
                                                                                  [int(g) for g in short[:5]]))
    table = fit_table(cols, cfg)
    stem = os.path.join(out_dir, "bias_table_%s" % h)
    table_io.save_table(table, stem)
    table_io.write_rows(per_draw_table(cols, cfg), os.path.join(out_dir, "draws_%s.csv" % h))
    m = table.meta
    print("%d in S; %s, multiplier %g%s, grid %s (%d supported), q_loo %.3g -> %s.npz/.json and draws_%s.csv "
          "(%.0f s)" % (m["n_kept"], m["estimator"], m["multiplier"],
                        "" if m["smoothing"] is None else ", smoothing %g" % m["smoothing"],
                        "x".join(map(str, m["grid_shape"])), m["n_supported"], m["q_loo"], stem, h,
                        time.perf_counter() - t0))
    if m["cv_at_grid_edge"]:
        print("WARNING: the cross-validated %s (%g) is at an end of its grid %s: widen the grid"
              % (m["cv_parameter"], m["multiplier"] if m["cv_parameter"] == "bandwidth_multiplier" else m["smoothing"],
                 m["cv_grid"]))
    return stem

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--draw-start", type=int, required=True, help="first draw g (>= 1)")
    r.add_argument("--draw-stop", type=int, required=True, help="one past the last draw")
    r.add_argument("--workers", type=int, default=1)
    r.add_argument("--seed", type=int, default=None)
    r.add_argument("--backend", default=None, choices=(None, "fft", "direct"))
    for p in (r, sub.add_parser("merge")):
        p.add_argument("--out-dir", required=True)
        p.add_argument("--config-json", default="")
    a = ap.parse_args(argv)
    cfg = load_config(a.config_json)
    if a.cmd == "run":
        run(cfg, a.draw_start, a.draw_stop, a.out_dir, a.workers, a.seed, a.backend)
    else:
        merge(cfg, a.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
