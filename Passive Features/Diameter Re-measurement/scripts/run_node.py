#!/usr/bin/env python3
"""Pilot measurement of real dendrite nodes, no bias table -- Block 11 in
specs/SPEC.md (design handoff Next actions 2-3, 7; procedure s.3.5, s.3.10).

Measures every node of the dendrite stretches that hold the given node ids
(Block 5, the same chain the table is built with) and writes, in --out-dir:

    pilot_<specimen>.csv            NodeResult columns, Allen's radius, sigma_fit,
                                    in_S, reject, calibration_node (and, with
                                    --background, the block's masked median and
                                    robust SD in plane k*)
    pilot_summary_<specimen>.json   counts, rejection reasons, percentiles over
                                    the nodes in S of d_hat, mu_hat, alpha_hat,
                                    d_hat / (2 r_Allen), phi; the phantom mu range
                                    they suggest (10th-90th percentile of mu_hat,
                                    procedure s.3.10); the dark-flag share; the
                                    number of calibration nodes
    figures/node_<id>.png           with --figures: focus score and fitted profile

Example (Colab, after the bootstrap cell)
    python scripts/run_node.py --specimen 529878215 --nodes 4505,4506 --cache-dir /content/drive/MyDrive/allen_cache \
        --out-dir /content/drive/MyDrive/diameters/pilot --figures --background

Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
for p in (SRC, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from allen_diameter.analysis import survey  # noqa: E402
from allen_diameter.loading import table_io  # noqa: E402
from build_table import load_config  # noqa: E402


def run(swc, provider, cfg, out_dir, specimen, transform=None, regs=None, only=None, figures=False, background=False,
        log=print):
    """The pilot on an SWC and a block provider (real: run_cell.real_provider). Returns (rows, summary)."""
    os.makedirs(out_dir, exist_ok=True)
    meas = survey.measure_nodes(swc, provider, cfg, transform, regs, only, log)
    rows = [survey.pilot_row(res, r_allen, cfg) for res, r_allen, _, _ in meas]
    summary = survey.summarize(rows, cfg)
    if background:
        stats = [survey.background_stats(provider, res, br, cfg) for res, _, br, _ in meas]
        for row, st in zip(rows, stats):
            row.update(bg_median_gl=st[0], bg_robust_sd_gl=st[1], bg_clipped_sd_gl=st[2], bg_unmasked_frac=st[3])
        summary["background"] = survey.background_summary(stats)
    summary.update(specimen=str(specimen), estimator_hash=cfg.signature_hash("estimator"),
                   sigma_fit_um=cfg.measure.sigma_fit_um)
    table_io.write_rows(rows, os.path.join(out_dir, "pilot_%s.csv" % specimen))
    with open(os.path.join(out_dir, "pilot_summary_%s.json" % specimen), "w") as f:
        json.dump(summary, f, indent=1, sort_keys=True, allow_nan=True)
    if figures:
        from allen_diameter.plotting import figures as fg
        os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)
        import matplotlib.pyplot as plt
        for res, _, _, _ in meas:
            if not math.isfinite(res.z_sub_um):
                continue
            v, I, model = survey.profile_at_node(provider, res, cfg)
            fig = fg.node_figure(res, v, I, model)
            fig.savefig(os.path.join(out_dir, "figures", "node_%d.png" % res.node_id), dpi=110)
            plt.close(fig)
    return rows, summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--specimen", required=True)
    ap.add_argument("--nodes", required=True, help="comma-separated node ids: their stretches are measured; 'all' for every dendrite node")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--swc", default="")
    ap.add_argument("--cache-dir", default="allen_cache")
    ap.add_argument("--shift-x", type=float, default=0.0, help="full-res px (global alignment)")
    ap.add_argument("--shift-y", type=float, default=0.0)
    ap.add_argument("--flip-h", type=float, default=None)
    ap.add_argument("--z0", type=float, default=0.0)
    ap.add_argument("--registration-json", default="")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--background", action="store_true")
    ap.add_argument("--config-json", default="")
    a = ap.parse_args(argv)
    cfg = load_config(a.config_json)
    import allen_image_io as aio
    import run_cell
    from allen_diameter.loading import swc_io
    swc = swc_io.read_swc(a.swc or aio.fetch_swc(int(a.specimen), a.cache_dir))
    planes = aio.plane_table(aio.list_images(int(a.specimen)))
    provider = run_cell.real_provider(aio.HttpFetcher(cache_dir=a.cache_dir), planes, cfg.acquisition.res0_um)
    regs = {}
    if a.registration_json:
        with open(a.registration_json) as f:
            regs = {int(k): v for k, v in json.load(f).items()}
    only = None if a.nodes.strip().lower() == "all" else {int(x) for x in a.nodes.split(",")}
    t0 = time.perf_counter()
    _, summary = run(swc, provider, cfg, a.out_dir, a.specimen,
                     dict(shift_full_px=(a.shift_x, a.shift_y), flip_y_full_h=a.flip_h, z0_um=a.z0), regs, only,
                     a.figures, a.background, log=lambda m: print(m, flush=True))
    print(json.dumps(summary, indent=1, sort_keys=True))
    print("%.1f min" % ((time.perf_counter() - t0) / 60))
    return 0


if __name__ == "__main__":
    sys.exit(main())
