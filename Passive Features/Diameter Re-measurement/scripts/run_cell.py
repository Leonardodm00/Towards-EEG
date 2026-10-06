#!/usr/bin/env python3
"""Re-measure the dendrite diameters of one Allen cell -- Block 8 in specs/SPEC.md.

Real data (Colab: api.brain-map.org is blocked from the sandbox): the SWC and
the 63x planes come through allen_image_io (HttpFetcher with a cache), the
SWC -> image transform from the global alignment (cells 1-12: shift and the
flip), the registration verdicts optionally from cell 13 (a JSON
{node_id: registration_check dict}). Every dendrite node is measured
(Block 5), corrected with the bias table (Block 7, refused when its estimator
signature differs) and filled per stretch; outputs in --out-dir:

    nodes_<specimen>.csv                  the per-node table (Block 8 columns)
    specimen_<specimen>/reconstruction.swc  radius = d_final / 2 on dendrite
                                          nodes, every other byte unchanged (D-013)
    summary_<specimen>.json               area ratio, counts, configuration hash

Example (Colab, after the repo bootstrap)
    python scripts/run_cell.py --specimen 529878215 --table /content/drive/MyDrive/tables/bias_table_<hash> \
        --shift-x 12.3 --shift-y -4.1 --flip-h 23456 --cache-dir /content/drive/MyDrive/allen_cache \
        --out-dir /content/drive/MyDrive/diameters --nodes 4505,4506

Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from allen_diameter.analysis import cell  # noqa: E402
from allen_diameter.loading import swc_io, table_io  # noqa: E402
from build_table import load_config  # noqa: E402


def real_provider(fetcher, planes, res0_um):
    """The fetch_zblock contract over Allen's planes (allen_image_io)."""
    import allen_image_io as aio

    def provider(left, top, width, height, k_lo, k_hi):
        return aio.fetch_zblock(fetcher, planes, k_lo, k_hi, left, top, width, height, res0_um)
    return provider


def write_outputs(out_dir, specimen, swc, rows, radius_new, ratio, cfg):
    os.makedirs(os.path.join(out_dir, "specimen_%s" % specimen), exist_ok=True)
    table_io.write_rows(rows, os.path.join(out_dir, "nodes_%s.csv" % specimen))
    swc_io.write_swc(swc, os.path.join(out_dir, "specimen_%s" % specimen, "reconstruction.swc"), radius_new)
    summary = dict(specimen=str(specimen), area_ratio=ratio, n_nodes=len(rows),
                   n_corrected=sum(1 for r in rows if r["filled_from"] == "self"),
                   n_filled_neighbours=sum(1 for r in rows if r["filled_from"] == "neighbours"),
                   n_allen=sum(1 for r in rows if r["filled_from"] == "allen"),
                   estimator_hash=cfg.signature_hash("estimator"), sigma_fit_um=cfg.measure.sigma_fit_um)
    with open(os.path.join(out_dir, "summary_%s.json" % specimen), "w") as f:
        json.dump(summary, f, indent=1, sort_keys=True)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--specimen", required=True)
    ap.add_argument("--table", required=True, help="bias table stem (without .npz/.json)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--swc", default="", help="an SWC on disk instead of fetching it")
    ap.add_argument("--cache-dir", default="allen_cache")
    ap.add_argument("--shift-x", type=float, default=0.0, help="full-res px (global alignment)")
    ap.add_argument("--shift-y", type=float, default=0.0)
    ap.add_argument("--flip-h", type=float, default=None, help="full-res image height when y is flipped")
    ap.add_argument("--z0", type=float, default=0.0, help="um: depth of plane index 0")
    ap.add_argument("--registration-json", default="")
    ap.add_argument("--nodes", default="", help="comma-separated node ids: only their stretches")
    ap.add_argument("--config-json", default="")
    a = ap.parse_args(argv)
    cfg = load_config(a.config_json)

    import allen_image_io as aio
    swc_path = a.swc or aio.fetch_swc(int(a.specimen), a.cache_dir)
    swc = swc_io.read_swc(swc_path)
    planes = aio.plane_table(aio.list_images(int(a.specimen)))
    provider = real_provider(aio.HttpFetcher(cache_dir=a.cache_dir), planes, cfg.acquisition.res0_um)
    table = table_io.load_table(a.table)
    table.check_estimator(cfg)
    regs = {}
    if a.registration_json:
        with open(a.registration_json) as f:
            regs = {int(k): v for k, v in json.load(f).items()}
    only = {int(x) for x in a.nodes.split(",")} if a.nodes else None
    t0 = time.perf_counter()
    rows, radius_new, ratio = cell.measure_cell(
        swc, provider, table, cfg, dict(shift_full_px=(a.shift_x, a.shift_y), flip_y_full_h=a.flip_h, z0_um=a.z0),
        regs, only, log=lambda m: print(m, flush=True))
    summary = write_outputs(a.out_dir, a.specimen, swc, rows, radius_new, ratio, cfg)
    print(json.dumps(summary, indent=1, sort_keys=True))
    print("%.1f min" % ((time.perf_counter() - t0) / 60))
    return 0


if __name__ == "__main__":
    sys.exit(main())
