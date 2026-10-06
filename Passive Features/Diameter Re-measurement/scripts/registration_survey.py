#!/usr/bin/env python3
"""Local registration on a list of nodes ("cell 13") -- Block 11 in specs/SPEC.md
(design handoff Next actions 2-3; the 2026-09-23 module allen_image_align).

For every node id: the unbranched stretch through it (allen_image_align.
path_through_node), the block that holds every shift the check tries
(plan_path_block, fetch_zblock), and registration_check (verdict, lateral
offset s*, depth offset dz*, p, coverage). Writes, in --out-dir:

    registration_<specimen>.json          {node_id: scalar results}, the file
                                          run_cell.py / run_node.py read with
                                          --registration-json (keys verdict,
                                          lateral_offset_um, z_offset_um)
    registration_summary_<specimen>.json  verdict counts; percentiles of |s*|
                                          and |dz*| over the ON nodes; their
                                          robust SDs as candidate jitter
                                          amplitudes (D-024 (vii)) -- a
                                          suggestion, not a decision
    figures/registration_<id>.png         with --figures (allen_image_plot)

Example (Colab; SHIFT and FLIP from the global alignment, cells 1-12)
    python scripts/registration_survey.py --specimen 529878215 --nodes 4505,1203,2210 \
        --cache-dir /content/drive/MyDrive/allen_cache --out-dir /content/drive/MyDrive/diameters/registration --figures

Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
for p in (SRC, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

SCALAR_KEYS = ("verdict", "z_verdict", "lateral_offset_um", "p_value", "coverage", "z_null", "contrast", "flank_snr",
               "n_null", "passes_p", "passes_coverage", "peak_width_um", "z_offset_um", "z_snr", "path_nodes",
               "path_len_um", "best_k")
PERCENTILES = (10, 25, 50, 75, 90)


def _scalar(v):
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v) if math.isfinite(float(v)) else None
    return v if isinstance(v, str) else str(v)


def summarize(results):
    """Counts by verdict category (analysis.registration); percentiles and robust SD (1.4826 MAD) of s*
    and dz* over the ON nodes."""
    from allen_diameter.analysis.registration import registration_category
    counts = {}
    for r in results.values():
        cat = registration_category(r["verdict"]) or "NONE"
        counts[cat] = counts.get(cat, 0) + 1
    on = [r for r in results.values() if registration_category(r["verdict"]) == "ON"]
    out = dict(n_nodes=len(results), verdicts=counts)
    for key in ("lateral_offset_um", "z_offset_um"):
        x = np.array([r[key] for r in on if r[key] is not None], dtype=float)
        if x.size:
            out[key] = {"p%d_abs" % q: float(np.percentile(np.abs(x), q)) for q in PERCENTILES}
            out[key]["robust_sd"] = float(1.4826 * np.median(np.abs(x - np.median(x))))
            out[key]["median"] = float(np.median(x))
            out[key]["n"] = int(x.size)
    if "lateral_offset_um" in out and "z_offset_um" in out:
        out["suggested_jitter_um"] = dict(jitter_xy_um=out["lateral_offset_um"]["robust_sd"],
                                          jitter_z_um=out["z_offset_um"]["robust_sd"],
                                          note="robust SDs over the ON nodes; D-024 (vii) keeps the knobs at 0 "
                                               "until the user decides")
    return out


def run(swc_df, fetcher, planes, node_ids, out_dir, specimen, res0_um, dz_um, shift_full_px=(0.0, 0.0),
        flip_y_full_h=None, z0_um=0.0, n_each_way=15, dz_max_um=4.0, max_width_um=4.0, margin_um=20.0,
        figures=False, log=print):
    """Registration on every node of node_ids. swc_df: allen_image_plot.read_swc's DataFrame; fetcher and
    planes as for allen_image_io.fetch_zblock. Returns (results {node_id: dict}, summary)."""
    import allen_image_align as aia
    import allen_image_io as aio
    os.makedirs(out_dir, exist_ok=True)
    k_min, k_max = int(planes["plane_index"].min()), int(planes["plane_index"].max())
    results = {}
    for nid in node_ids:
        path = aia.path_through_node(swc_df, int(nid), n_each_way)
        plan = aia.plan_path_block(path, res0_um, dz_um, shift_full_px, flip_y_full_h, margin_um=margin_um,
                                   dz_max_um=dz_max_um, k_min=k_min, k_max=k_max, z0_um=z0_um)
        block, ks, valid, frame = aio.fetch_zblock(fetcher, planes, plan["k_lo"], plan["k_hi"], plan["left"],
                                                   plan["top"], plan["width"], plan["height"], res0_um)
        chk = aia.registration_check(path, block, ks, valid, frame, dz_um, node_id=int(nid),
                                     shift_full_px=shift_full_px, flip_y_full_h=flip_y_full_h, z0_um=z0_um,
                                     dz_max_um=dz_max_um, max_width_um=max_width_um)
        results[int(nid)] = {k: _scalar(chk.get(k)) for k in SCALAR_KEYS}
        log("node %d: %s, s* %s um, dz* %s um, p %.3f, coverage %.2f"
            % (int(nid), chk["verdict"], results[int(nid)]["lateral_offset_um"], results[int(nid)]["z_offset_um"],
               chk["p_value"], chk["coverage"]))
        if figures:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import allen_image_plot as aip
            os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)
            fig = aip.plot_registration_check(chk, res0_um)
            fig = fig if hasattr(fig, "savefig") else plt.gcf()
            fig.savefig(os.path.join(out_dir, "figures", "registration_%d.png" % int(nid)), dpi=100)
            plt.close("all")
    summary = summarize(results)
    summary["specimen"] = str(specimen)
    with open(os.path.join(out_dir, "registration_%s.json" % specimen), "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=1, sort_keys=True)
    with open(os.path.join(out_dir, "registration_summary_%s.json" % specimen), "w") as f:
        json.dump(summary, f, indent=1, sort_keys=True)
    return results, summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--specimen", required=True)
    ap.add_argument("--nodes", required=True, help="comma-separated node ids")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--swc", default="")
    ap.add_argument("--cache-dir", default="allen_cache")
    ap.add_argument("--shift-x", type=float, default=0.0, help="full-res px (global alignment)")
    ap.add_argument("--shift-y", type=float, default=0.0)
    ap.add_argument("--flip-h", type=float, default=None)
    ap.add_argument("--z0", type=float, default=0.0)
    ap.add_argument("--n-each-way", type=int, default=15)
    ap.add_argument("--dz-max", type=float, default=4.0, help="um")
    ap.add_argument("--max-width", type=float, default=4.0, help="um")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--config-json", default="")
    a = ap.parse_args(argv)
    from build_table import load_config
    import allen_image_io as aio
    import allen_image_plot as aip
    cfg = load_config(a.config_json)
    swc_df = aip.read_swc(a.swc or aio.fetch_swc(int(a.specimen), a.cache_dir))
    planes = aio.plane_table(aio.list_images(int(a.specimen)))
    _, summary = run(swc_df, aio.HttpFetcher(cache_dir=a.cache_dir), planes, [int(x) for x in a.nodes.split(",")],
                     a.out_dir, a.specimen, cfg.acquisition.res0_um, cfg.acquisition.dz_um, (a.shift_x, a.shift_y),
                     a.flip_h, a.z0, a.n_each_way, a.dz_max, a.max_width, figures=a.figures,
                     log=lambda m: print(m, flush=True))
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
