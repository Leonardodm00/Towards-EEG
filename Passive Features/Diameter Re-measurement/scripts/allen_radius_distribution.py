#!/usr/bin/env python3
"""Distribution of the diameters Allen annotated in SWC files (D-024, iii).

Sets the upper end of the phantom diameter range (config PhantomConfig.d_range_um,
temporary (0.2, 4.0) um until this script has been run) by reporting, per SWC
type and over all dendrite nodes, the percentiles and the maximum of the
annotated diameter 2 r, plus the share of nodes above a few thresholds.

Inputs, one of
    --swc FILE [FILE ...]        SWC files on disk
    --archive-root DIR           the davinci archive: every
                                 <DIR>/**/specimen_*/reconstruction.swc
                                 (D-013 layout; the cluster path is
                                 /davinci-1/home/ldellamea/Human Neurons Fitting)
    --fetch SPECIMEN_ID [...]    download the SWC through the Allen API
                                 (Colab only: api.brain-map.org is blocked
                                 from the sandbox); uses allen_image_io.fetch_swc
                                 and a cache directory (--cache-dir)

Outputs
    a per-node CSV (--out-nodes), a per-file summary CSV (--out), and a
    printed summary with a text histogram of log10(diameter).

Examples
    # Colab, after the repo bootstrap of the implementation handoff:
    python scripts/allen_radius_distribution.py --fetch 529878215 --cache-dir /content/drive/MyDrive/allen_cache
    # davinci, all cells of one group:
    python scripts/allen_radius_distribution.py --archive-root "/davinci-1/home/ldellamea/Human Neurons Fitting/L3_exc"
    # a file:
    python scripts/allen_radius_distribution.py --swc path/to/reconstruction.swc

Pure ASCII, no dependency beyond numpy (hpc-python-compat).
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from allen_diameter.loading import swc_io  # noqa: E402

PERCENTILES = (0.0, 1.0, 5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0, 99.0, 99.9, 100.0)
THRESHOLDS_UM = (1.0, 2.0, 3.0, 4.0, 6.0)
TYPE_NAMES = {1: "soma", 2: "axon", 3: "basal", 4: "apical"}


def specimen_id_from_path(path: str) -> str:
    """'.../specimen_529878215/reconstruction.swc' -> '529878215'; else the stem."""
    for part in reversed(path.replace("\\", "/").split("/")):
        if part.startswith("specimen_"):
            return part[len("specimen_"):]
    return os.path.splitext(os.path.basename(path))[0]


def summarise(diam: np.ndarray) -> dict:
    """Percentiles (numpy default linear interpolation), mean, and the share
    above each threshold, for one array of diameters (um)."""
    d = np.asarray(diam, dtype=float)
    out = {"n": int(d.size)}
    if d.size == 0:
        return out
    pct = np.percentile(d, PERCENTILES)
    for p, v in zip(PERCENTILES, pct):
        out["p%g" % p] = float(v)
    out["mean"] = float(d.mean())
    for t in THRESHOLDS_UM:
        out["share_above_%gum" % t] = float(np.mean(d > t))
    return out


def text_histogram(diam: np.ndarray, n_bins: int = 24, width: int = 50) -> str:
    """A log10 histogram of the diameters, as text."""
    d = np.asarray(diam, dtype=float)
    d = d[d > 0]
    if d.size == 0:
        return "(no positive diameters)"
    lo, hi = np.log10(d.min()), np.log10(d.max())
    if hi - lo < 1e-9:
        hi = lo + 1e-9
    edges = np.linspace(lo, hi, n_bins + 1)
    counts, _ = np.histogram(np.log10(d), bins=edges)
    top = max(int(counts.max()), 1)
    lines = []
    for c, a, b in zip(counts, edges[:-1], edges[1:]):
        bar = "#" * int(round(width * c / top))
        lines.append("%7.3f-%7.3f um %7d %s" % (10 ** a, 10 ** b, c, bar))
    return "\n".join(lines)


def collect(paths, types):
    rows_nodes, rows_files = [], []
    for p in paths:
        s = swc_io.read_swc(p)
        sid = specimen_id_from_path(p)
        mask = s.dendrite_mask(types)
        for i in np.flatnonzero(mask):
            rows_nodes.append({"specimen": sid, "node_id": int(s.ids[i]), "type": int(s.types[i]),
                               "radius_um": float(s.radius[i]), "diameter_um": float(2 * s.radius[i])})
        rec = {"specimen": sid, "file": p, "n_nodes_total": len(s)}
        rec.update({"dend_" + k: v for k, v in summarise(s.diameter[mask]).items()})
        for t in sorted(set(types)):
            m = s.types == t
            rec.update({"%s_%s" % (TYPE_NAMES.get(t, "type%d" % t), k): v for k, v in summarise(s.diameter[m]).items()})
        rows_files.append(rec)
    return rows_nodes, rows_files


def write_csv(path, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--swc", nargs="+", help="SWC files")
    src.add_argument("--archive-root", help="root of the davinci archive (specimen_*/reconstruction.swc)")
    src.add_argument("--fetch", nargs="+", type=int, metavar="SPECIMEN_ID", help="download via the Allen API (Colab)")
    ap.add_argument("--cache-dir", default="allen_swc_cache", help="where --fetch stores the SWC files")
    ap.add_argument("--types", default="3,4", help="SWC types counted as dendrite (default 3,4)")
    ap.add_argument("--out", default="allen_radius_summary.csv", help="per-file summary CSV")
    ap.add_argument("--out-nodes", default="allen_radius_nodes.csv", help="per-node CSV")
    args = ap.parse_args(argv)
    types = tuple(int(t) for t in args.types.split(","))

    if args.swc:
        paths = list(args.swc)
    elif args.archive_root:
        paths = sorted(glob.glob(os.path.join(args.archive_root, "**", "specimen_*", "reconstruction.swc"), recursive=True))
        if not paths:
            sys.exit("no specimen_*/reconstruction.swc under %s" % args.archive_root)
    else:
        import allen_image_io as aio  # flat module in src/; needs the network
        paths = [aio.fetch_swc(sid, args.cache_dir) for sid in args.fetch]

    nodes, files = collect(paths, types)
    write_csv(args.out, files)
    write_csv(args.out_nodes, nodes)

    diam = np.array([r["diameter_um"] for r in nodes], dtype=float)
    print("%d files, %d dendrite nodes (types %s)" % (len(files), diam.size, ",".join(map(str, types))))
    pooled = summarise(diam)
    print("pooled dendrite diameter (um): " + ", ".join(
        "p%g=%.3f" % (p, pooled["p%g" % p]) for p in PERCENTILES if ("p%g" % p) in pooled))
    print("mean %.3f um; share above " % pooled.get("mean", float("nan")) +
          ", ".join("%g um: %.4f" % (t, pooled["share_above_%gum" % t]) for t in THRESHOLDS_UM if ("share_above_%gum" % t) in pooled))
    print("per file (dendrites): specimen  n  p50  p99  max")
    for r in files:
        print("   %-12s %6d %7.3f %7.3f %7.3f" % (r["specimen"], r["dend_n"], r.get("dend_p50", float("nan")),
                                                   r.get("dend_p99", float("nan")), r.get("dend_p100", float("nan"))))
    print("log10 histogram of the pooled dendrite diameters:")
    print(text_histogram(diam))
    print("wrote %s and %s" % (args.out, args.out_nodes))
    print("NOTE: PhantomConfig.d_range_um is (0.2, 4.0) um until this output sets d_max (D-024).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
