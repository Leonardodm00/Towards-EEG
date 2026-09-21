#!/usr/bin/env python3
"""Soma-radius census over the campaign skeletons, WITHOUT running P1.

    python3 soma_census.py --neurons-dir ../h01/neurons --out ../h01/soma_census.csv
    python3 soma_census.py --neurons-dir ../h01/neurons --ids-file "../Save nids/p23_nids.txt"

Answers one question before ten populations are launched: how many H01
skeletons carry a soma that passes the completeness gate of
`soma_enforce.py`, and how many are truncated arbours with a promoted root.
The answer decides whether "select the high-confidence cells downstream" is a
filter or a decimation.

It re-uses the pipeline's own test rather than restating it: the radius floor
and the outlier ratio come from `soma_enforce.DEFAULT_MIN_SOMA_RADIUS_NM` and
`DEFAULT_MIN_RADIUS_RATIO`, and the geometric identification is
`soma_enforce.identify_soma_by_geometry` itself. If that module's thresholds
change, this census changes with it, which is the point.

Reads six columns per skeleton (id, p, r, x, y, z -- the coordinates are
needed by identify_soma_by_geometry for its distance-to-origin field), so it
is I/O bound and needs no
NEURON, no network and no alignment bank.

Columns written, one row per cell:
    cell_id, n_nodes, root_id, root_r_nm, root_diam_um, median_r_nm,
    root_ratio, root_above_floor, geom_candidate_id, geom_candidate_r_nm,
    geom_ratio, geom_above_floor, geom_is_outlier, name_geometry_agree,
    soma_area_um2, verdict
where verdict is one of
    ok                    the root passes the floor and is an outlier
    below_floor           the root is a soma by name but too thin
    geometry_disagrees    a thicker candidate exists elsewhere in the cell
    both                  below the floor AND the geometry disagrees
    no_root / no_r_column / unreadable

Pure ASCII, LF only.
"""
import argparse
import os
import sys

import pandas as pd

CENSUS_VERSION = "soma_census v1.0"
COLUMNS = ("cell_id", "n_nodes", "root_id", "root_r_nm", "root_diam_um",
           "median_r_nm", "root_ratio", "root_above_floor",
           "geom_candidate_id", "geom_candidate_r_nm", "geom_ratio",
           "geom_above_floor", "geom_is_outlier", "name_geometry_agree",
           "soma_area_um2", "verdict")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--neurons-dir", required=True,
                   help="folder of neuron_<id>.csv")
    p.add_argument("--stage1-dir", default=None,
                   help="the symlink farm holding soma_enforce.py "
                        "(default: <this file's dir>/stage1)")
    p.add_argument("--ids-file", default=None,
                   help="restrict to these ids, one per line (default: every "
                        "neuron_*.csv in --neurons-dir)")
    p.add_argument("--out", default=None, help="CSV to write (default: stdout summary only)")
    p.add_argument("--limit", type=int, default=None, help="stop after N cells (a smoke run)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def load_soma_enforce(stage1_dir):
    here = os.path.dirname(os.path.abspath(__file__))
    stage1_dir = stage1_dir or os.path.join(here, "stage1")
    for d in (stage1_dir, here):
        if d not in sys.path:
            sys.path.insert(0, d)
    try:
        import soma_enforce as se
    except ImportError as exc:
        raise SystemExit(
            "cannot import soma_enforce (%s). Give --stage1-dir <h01_code/stage1>."
            % exc)
    return se


def cell_ids(neurons_dir, ids_file=None):
    if ids_file:
        if not os.path.isfile(ids_file):
            raise SystemExit("--ids-file does not exist: %s" % ids_file)
        out = []
        for ln in open(ids_file):
            ln = ln.strip()
            if ln and not ln.startswith("#"):
                out.append(int(float(ln)))
        return sorted(set(out))
    out = []
    for f in os.listdir(neurons_dir):
        if f.startswith("neuron_") and f.endswith(".csv"):
            stem = f[len("neuron_"):-len(".csv")]
            if stem.isdigit():
                out.append(int(stem))
    return sorted(out)


def census_one(path, cell_id, se):
    """One row. Never raises: an unreadable skeleton is a verdict, not a crash."""
    row = {k: None for k in COLUMNS}
    row["cell_id"] = cell_id
    try:
        df = pd.read_csv(path, usecols=["id", "p", "r", "x", "y", "z"],
                         low_memory=False)
    except ValueError:
        # the column set differs; read the header to say so precisely
        try:
            cols = list(pd.read_csv(path, nrows=0).columns)
        except Exception:
            cols = []
        missing = [c for c in ("id", "p", "r", "x", "y", "z") if c not in cols]
        row["verdict"] = ("no_r_column" if "r" in missing
                          else ("missing:%s" % ",".join(missing) if missing
                                else "unreadable"))
        return row
    except Exception:
        row["verdict"] = "unreadable"
        return row

    row["n_nodes"] = int(len(df))
    roots = df[df["p"] == -1]
    if roots.empty:
        row["verdict"] = "no_root"
        return row

    r = df["r"].astype(float)
    root = roots.iloc[0]
    root_r = float(root["r"])
    # the median over every OTHER node, matching identify_soma_by_geometry's
    # median_other_r_nm rather than the median over all nodes
    med = float(r.drop(index=roots.index[:1]).median())
    row.update({"root_id": int(root["id"]), "root_r_nm": root_r,
                "root_diam_um": 2.0 * root_r / 1000.0, "median_r_nm": med,
                "root_ratio": (root_r / med) if med > 0 else float("nan"),
                "root_above_floor": bool(root_r >= se.DEFAULT_MIN_SOMA_RADIUS_NM),
                "soma_area_um2": float(se.soma_area_um2(root_r))})

    # the pipeline's own geometric identification. It wants a class column and
    # excludes spine/glia; this census has no labels, so every node is a
    # candidate -- which is the conservative reading (it can only find a
    # THICKER candidate than the labelled pipeline would).
    work = df.copy()
    work["compartment_class"] = ""
    try:
        # thresholds passed EXPLICITLY: identify_soma_by_geometry binds them
        # as default arguments, which Python evaluates once at import, so a
        # later change to soma_enforce's module constants would not reach it.
        geom = se.identify_soma_by_geometry(
            work, min_soma_radius_nm=se.DEFAULT_MIN_SOMA_RADIUS_NM,
            min_radius_ratio=se.DEFAULT_MIN_RADIUS_RATIO)
    except Exception as exc:
        row["verdict"] = "geometry_failed:%s" % type(exc).__name__
        return row
    row.update({"geom_candidate_id": geom.get("candidate_id"),
                "geom_candidate_r_nm": geom.get("candidate_r_nm"),
                "geom_ratio": geom.get("radius_ratio"),
                "geom_above_floor": bool(geom.get("radius_above_floor")),
                "geom_is_outlier": bool(geom.get("radius_is_outlier"))})
    agree = (geom.get("candidate_id") is not None
             and int(geom["candidate_id"]) == int(root["id"]))
    row["name_geometry_agree"] = bool(agree)

    below = not row["root_above_floor"]
    if below and not agree:
        row["verdict"] = "both"
    elif below:
        row["verdict"] = "below_floor"
    elif not agree:
        row["verdict"] = "geometry_disagrees"
    else:
        row["verdict"] = "ok"
    return row


def summarise(df, se):
    n = len(df)
    print()
    print("%s | %d cells | floor %.0f nm radius (%.1f um diameter), outlier ratio %.1f"
          % (CENSUS_VERSION, n, se.DEFAULT_MIN_SOMA_RADIUS_NM,
             2 * se.DEFAULT_MIN_SOMA_RADIUS_NM / 1000.0,
             se.DEFAULT_MIN_RADIUS_RATIO))
    print("\nverdict:")
    vc = df["verdict"].value_counts()
    for k, v in vc.items():
        print("  %-22s %6d  (%5.1f%%)" % (k, v, 100.0 * v / n))
    good = df[df["verdict"] == "ok"]
    print("\nHIGH CONFIDENCE (root passes floor AND is the geometric soma): "
          "%d of %d = %.1f%%" % (len(good), n, 100.0 * len(good) / n))
    ok_r = pd.to_numeric(df["root_r_nm"], errors="coerce").dropna()
    if len(ok_r):
        q = ok_r.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
        print("\nroot radius (nm), all cells: "
              "p5 %.0f | p25 %.0f | median %.0f | p75 %.0f | p95 %.0f"
              % tuple(q.values))
        print("root diameter (um), same:   "
              "p5 %.2f | p25 %.2f | median %.2f | p75 %.2f | p95 %.2f"
              % tuple(2 * q.values / 1000.0))
    return vc


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not os.path.isdir(args.neurons_dir):
        raise SystemExit("--neurons-dir does not exist: %s" % args.neurons_dir)
    se = load_soma_enforce(args.stage1_dir)
    ids = cell_ids(args.neurons_dir, args.ids_file)
    if args.limit:
        ids = ids[:args.limit]
    if not ids:
        raise SystemExit("no neuron_*.csv found in %s" % args.neurons_dir)
    print("%s: %d skeletons in %s" % (CENSUS_VERSION, len(ids), args.neurons_dir))

    rows = []
    for i, cid in enumerate(ids, 1):
        path = os.path.join(args.neurons_dir, "neuron_%d.csv" % cid)
        if not os.path.isfile(path):
            rows.append({"cell_id": cid, "verdict": "missing_file"})
            continue
        rows.append(census_one(path, cid, se))
        if args.verbose or i % 200 == 0:
            r = rows[-1]
            print("  [%4d/%d] %d %-20s root_r %s" %
                  (i, len(ids), cid, r.get("verdict"), r.get("root_r_nm")))
    df = pd.DataFrame(rows, columns=COLUMNS)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        tmp = args.out + ".tmp"
        df.to_csv(tmp, index=False, lineterminator="\n")
        os.replace(tmp, args.out)
        print("\nwrote %s" % args.out)
    summarise(df, se)
    return 0


if __name__ == "__main__":
    sys.exit(main())
