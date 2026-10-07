#!/usr/bin/env python3
"""Plane montages of measured nodes: every plane the focus rule scored, with
Allen's reconstruction drawn on it -- Block 11 in specs/SPEC.md (2026-10-07).

Each node is measured again by Block 5 exactly as the pilot (run_node.py)
measures it: the same stretch, hence the same single block request, so with the
pilot's image cache every crop comes from disk (the last line counts the crops
read from the cache and those downloaded). Writes, in --out-dir:

    planes_<id>.png            one panel per plane of the node's focus curves
                               (k_swc +- planes_half, widened for tilt): the
                               image, Allen's traced centre lines and +-radius
                               (cyan: the measured stretch; amber: the other
                               dendrites), the SWC node, the fit's profile line,
                               the fitted edges in plane k*
                               (plotting.figures.plane_montage)
    planes_<specimen>.json     per node: k*, k_star_depth, k_swc, d_hat, Allen's
                               2r, the number of segments drawn, and, with
                               --pilot-csv, the fields where the re-measurement
                               differs from the pilot row (none expected for the
                               pilot's configuration and code)

Node selection, --nodes: comma-separated node ids, or 'differ' for the nodes of
--pilot-csv where the configured focus rule and the dip depth pick different
planes (k_star != k_star_depth, finite z_sub_um, D-030); at most --max-nodes.

Example (Colab, after Cell 4b)
    python scripts/node_planes.py --specimen 529878215 --nodes differ \
        --pilot-csv /content/drive/MyDrive/diameters/pilot/pilot_529878215.csv \
        --cache-dir /content/drive/MyDrive/allen_cache --out-dir /content/drive/MyDrive/diameters/pilot/planes

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

COMPARED = ("k_star", "k_star_depth", "z_sub_um", "d_hat_um", "mu_hat_per_um", "focus_rule")


def select_nodes(spec, pilot_rows=None, max_nodes=6):
    """Node ids from --nodes: explicit ids in the given order, or 'differ' (needs
    the pilot rows): the rows with a finite z_sub_um and k_star != k_star_depth,
    in the CSV's order. At most max_nodes ids."""
    spec = str(spec).strip()
    if spec.lower() == "differ":
        if pilot_rows is None:
            raise ValueError("--nodes differ needs --pilot-csv")
        if pilot_rows and "k_star_depth" not in pilot_rows[0]:
            raise ValueError("the pilot CSV has no k_star_depth column (written before D-030): re-run the pilot")
        ids = [int(r["node_id"]) for r in pilot_rows
               if math.isfinite(float(r["z_sub_um"])) and int(r["k_star"]) != int(r["k_star_depth"])]
    else:
        ids = [int(x) for x in spec.split(",") if x.strip()]
    return ids[:int(max_nodes)]


def _same(x, y):
    if isinstance(x, str) or isinstance(y, str):
        return str(x) == str(y)
    x, y = float(x), float(y)
    if math.isnan(x) or math.isnan(y):
        return math.isnan(x) and math.isnan(y)
    return abs(x - y) <= 1e-9 * max(1.0, abs(x), abs(y))


def compare_with_pilot(result, row):
    """The COMPARED fields where the re-measured NodeResult differs from the
    pilot CSV row (a CSV float is written by repr, so it reads back exactly;
    1e-9 relative leaves room for nothing but a change of code, configuration
    or library). Fields the row lacks are skipped."""
    return [c for c in COMPARED if c in row and not _same(getattr(result, c), row[c])]


def run(swc, provider, cfg, node_ids, out_dir, specimen, transform=None, pilot_rows=None, view_half_um=None,
        dpi=110, log=print):
    """Montage per node id; returns the list of per-node records (also written to planes_<specimen>.json)."""
    from allen_diameter.plotting import figures as fg
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    m, dz = cfg.measure, cfg.acquisition.dz_um
    by_id = {int(r["node_id"]): r for r in (pilot_rows or [])}
    records = []
    for nid in node_ids:
        try:
            pl = survey.node_planes(swc, provider, cfg, nid, transform)
        except ValueError as e:
            log("[planes] node %d skipped: %s" % (nid, e))
            records.append(dict(node_id=int(nid), skipped=str(e)))
            continue
        res = pl["result"]
        fig = fg.plane_montage(pl, dz, m.profile_half_um, view_half_um=view_half_um)
        path = os.path.join(out_dir, "planes_%d.png" % nid)
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        rec = dict(node_id=int(nid), png=path, k_star=int(res.k_star), k_star_depth=int(res.k_star_depth),
                   k_swc=int(pl["k_swc"]), z_sub_um=float(res.z_sub_um), d_hat_um=float(res.d_hat_um),
                   allen_2r_um=2.0 * float(pl["allen_radius_um"]), n_planes=int(len(res.focus_planes)),
                   n_segments=int(len(pl["segments"])), n_segments_stretch=int((pl["segments"][:, 8] > 0.5).sum()))
        line = ("[planes] node %d: %d planes, k* %d, dip-depth plane %d, SWC plane %d, d_hat %.2f um, Allen 2r %.2f um"
                % (nid, rec["n_planes"], rec["k_star"], rec["k_star_depth"], rec["k_swc"], rec["d_hat_um"],
                   rec["allen_2r_um"]))
        if pilot_rows is not None:
            row = by_id.get(int(nid))
            if row is None:
                rec["pilot"] = "not in the pilot CSV"
            else:
                diff = compare_with_pilot(res, row)
                rec["pilot"] = "same" if not diff else {c: [row[c], getattr(res, c)] for c in diff}
                if diff:
                    was = ", ".join(str(row[c]) for c in diff)
                    line += (" | DIFFERS from the pilot in %s (pilot %s): another configuration or code version; the "
                             "figure shows the re-measurement" % (", ".join(diff), was))
                else:
                    line += " | same as the pilot"
        log(line)
        records.append(rec)
    with open(os.path.join(out_dir, "planes_%s.json" % specimen), "w") as f:
        json.dump(records, f, indent=1, sort_keys=True, allow_nan=True)
    return records


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--specimen", required=True)
    ap.add_argument("--nodes", required=True, help="comma-separated node ids, or 'differ' (needs --pilot-csv)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pilot-csv", default="", help="the pilot's CSV: 'differ' selection and the consistency check")
    ap.add_argument("--max-nodes", type=int, default=6)
    ap.add_argument("--view-half-um", type=float, default=None,
                    help="half-width of each panel, um (default: the largest square about the node inside its block)")
    ap.add_argument("--swc", default="")
    ap.add_argument("--cache-dir", default="allen_cache")
    ap.add_argument("--shift-x", type=float, default=0.0, help="full-res px (global alignment)")
    ap.add_argument("--shift-y", type=float, default=0.0)
    ap.add_argument("--flip-h", type=float, default=None)
    ap.add_argument("--z0", type=float, default=0.0)
    ap.add_argument("--config-json", default="")
    a = ap.parse_args(argv)
    cfg = load_config(a.config_json)
    pilot_rows = table_io.read_rows([a.pilot_csv]) if a.pilot_csv else None
    node_ids = select_nodes(a.nodes, pilot_rows, a.max_nodes)
    if a.pilot_csv:
        summ = os.path.join(os.path.dirname(a.pilot_csv), "pilot_summary_%s.json" % a.specimen)
        if os.path.exists(summ):
            with open(summ) as f:
                h = json.load(f).get("estimator_hash")
            if h != cfg.signature_hash("estimator"):
                print("[planes] the pilot was measured with estimator hash %s, this configuration has %s: expect "
                      "differences" % (h, cfg.signature_hash("estimator")), flush=True)
    if not node_ids:
        print("[planes] no node selected (--nodes %s): nothing to draw" % a.nodes, flush=True)
        return 0
    import allen_image_io as aio
    import run_cell
    from allen_diameter.loading import swc_io
    swc = swc_io.read_swc(a.swc or aio.fetch_swc(int(a.specimen), a.cache_dir))
    fetcher = aio.HttpFetcher(cache_dir=a.cache_dir)
    planes = aio.plane_table(aio.list_images(int(a.specimen)))
    provider = run_cell.real_provider(fetcher, planes, cfg.acquisition.res0_um)
    t0 = time.perf_counter()
    recs = run(swc, provider, cfg, node_ids, a.out_dir, a.specimen,
               dict(shift_full_px=(a.shift_x, a.shift_y), flip_y_full_h=a.flip_h, z0_um=a.z0), pilot_rows,
               a.view_half_um, log=lambda m: print(m, flush=True))
    n_ok = sum(1 for r in recs if "png" in r)
    print("[planes] wrote %d montages (%d skipped) in %s; crops: %d from the cache, %d downloaded (%.1f MB); %.1f min"
          % (n_ok, len(recs) - n_ok, a.out_dir, fetcher.n_cache_hits, fetcher.n_requests,
             fetcher.bytes_downloaded / 1e6, (time.perf_counter() - t0) / 60), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
