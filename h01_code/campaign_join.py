#!/usr/bin/env python3
"""One campaign table per cell (and per passive tree), with BOTH F values
under names that cannot be confused (decision D-004, 2026-09-22).

  python3 campaign_join.py --root ../h01                      # every p1*/ tree
  python3 campaign_join.py --root ../h01 --p1-dir p1 --p1-dir p1_inh_SST
  python3 campaign_join.py --root ../h01 --require-complete   # exit 1 if any
                                                              # P1 cell has no P3

Reads <root>/<tree>/p1_summary.csv for every P1 tree (default: every
directory under <root> named p1 or p1_* that holds one) and
<root>/out/spine_area_F_summary.csv (P3), and writes
<root>/out/campaign_F.csv, one row per (cell_id, p1_tree).

The standing trap this closes: p1_summary.csv carries F_lit, which is P1's
SKELETON-frustum bracket and is never updated by P3, so any consumer reading
it as "the F" reports the fallback. Here that column is renamed
F_lit_skel_p1, the deliverable is F_lit_deliverable (P3, mesh_beyond), and a
cell without a P3 row keeps NaN there -- it is NEVER filled from the
skeleton. p3_present says which case a row is.

Refusals (exit 1, nothing written): a P3 row whose deliverable_variant is not
the expected one (--deliverable, default mesh_beyond); no P1 tree found; a
p1_summary.csv without cell_id. --require-complete additionally exits 1,
after writing, if any P1 cell lacks a P3 row, listing them.

Pure ASCII, LF only.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

JOIN_VERSION = "campaign_join v1.0"
DELIVERABLE_DEFAULT = "mesh_beyond"

# From P1's summary, renamed where the name would otherwise mislead.
P1_COLUMNS = {
    "layer": "layer", "cell_type": "cell_type", "layer_source": "layer_source",
    "status": "p1_status", "qc_status": "p1_qc_status", "reasons": "p1_reasons",
    "gate_status": "p1_gate_status", "hoc_verdict": "p1_hoc_verdict",
    "quarantined": "p1_quarantined", "n_spines": "n_spines_p1",
    "cm": "cm", "Ra": "Ra", "passive_table": "passive_table",
    "F_lit": "F_lit_skel_p1", "F_lit_nocap": "F_lit_skel_nocap_p1",
    "fingerprint": "p1_fingerprint",
}
# From P3's summary. F_lit_deliverable is THE F (D-004).
P3_COLUMNS = {
    "deliverable_variant": "deliverable_variant",
    "F_lit_deliverable": "F_lit_deliverable",
    "F_whole_deliverable": "F_whole_deliverable",
    "F_lit_skel": "F_lit_skel_p3",
    "F_lit_mesh": "F_lit_mesh_p3",
    "qc_status": "p3_qc_status", "qc_reason": "p3_qc_reason",
    "coverage_count_deliverable": "coverage_count_deliverable",
    "fallback_area_frac_deliverable": "fallback_area_frac_deliverable",
    "min_coverage": "min_coverage",
    "n_spines": "n_spines_p3", "n_measured": "n_measured_p3",
    "module_version": "p3_module_version",
}
OUT_COLUMNS = (["cell_id", "p1_tree"] + list(P1_COLUMNS.values())
               + ["p3_present"] + list(P3_COLUMNS.values()))


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", required=True, help="campaign root (has out/ and p1*/)")
    p.add_argument("--p1-dir", action="append", default=None,
                   help="P1 tree, relative to --root or absolute; repeatable. "
                        "Default: every <root>/p1 and <root>/p1_* with a p1_summary.csv")
    p.add_argument("--p3-summary", default=None,
                   help="default <root>/out/spine_area_F_summary.csv")
    p.add_argument("--out", default=None, help="default <root>/out/campaign_F.csv")
    p.add_argument("--deliverable", default=DELIVERABLE_DEFAULT,
                   help="the variant every P3 row must carry (default %s)" % DELIVERABLE_DEFAULT)
    p.add_argument("--require-complete", action="store_true",
                   help="exit 1 (after writing) if any P1 cell has no P3 row")
    return p


def find_p1_trees(root, p1_dirs=None):
    """(tree name, p1_summary.csv path) for every P1 tree."""
    if p1_dirs:
        cands = [d if os.path.isabs(d) else os.path.join(root, d) for d in p1_dirs]
    else:
        cands = [os.path.join(root, d) for d in sorted(os.listdir(root))
                 if d == "p1" or d.startswith("p1_")]
    trees = []
    for d in cands:
        f = os.path.join(d, "p1_summary.csv")
        if os.path.isfile(f):
            trees.append((os.path.basename(os.path.normpath(d)), f))
        elif p1_dirs:
            raise SystemExit("--p1-dir %s has no p1_summary.csv (run "
                             "run_p1_export.py --summarise --out-dir %s first)" % (d, d))
    if not trees:
        raise SystemExit("no P1 tree with a p1_summary.csv under %s" % root)
    return trees


def load_p1(trees):
    frames = []
    for name, f in trees:
        df = pd.read_csv(f)
        if "cell_id" not in df.columns:
            raise SystemExit("%s has no cell_id column" % f)
        out = pd.DataFrame({"cell_id": df["cell_id"].astype("int64"), "p1_tree": name})
        for src, dst in P1_COLUMNS.items():
            out[dst] = df[src].values if src in df.columns else np.nan
        frames.append(out)
    p1 = pd.concat(frames, ignore_index=True)
    dup = p1.duplicated(["cell_id", "p1_tree"], keep=False)
    if dup.any():
        raise SystemExit("duplicate (cell_id, p1_tree) rows in P1 summaries: %s"
                         % sorted(set(p1.loc[dup, "cell_id"].tolist()))[:10])
    return p1


def load_p3(path, deliverable):
    """P3 summary as one row per cell_id; refuses a row whose deliverable
    variant is not `deliverable` (a table mixing variants under one column
    name is exactly the confusion D-004 forbids)."""
    if not os.path.isfile(path):
        return pd.DataFrame(columns=["cell_id"] + list(P3_COLUMNS.values()))
    df = pd.read_csv(path)
    if "cell_id" not in df.columns:
        raise SystemExit("%s has no cell_id column" % path)
    if "deliverable_variant" in df.columns:
        bad = df[df["deliverable_variant"].astype(str) != deliverable]
        if len(bad):
            raise SystemExit(
                "P3 rows whose deliverable_variant is not %s: cells %s -- re-merge them "
                "with --deliverable %s; a mixed table is refused"
                % (deliverable, bad["cell_id"].astype("int64").tolist()[:10], deliverable))
    else:
        raise SystemExit("%s has no deliverable_variant column (pre-v1.7 merge?)" % path)
    out = pd.DataFrame({"cell_id": df["cell_id"].astype("int64")})
    for src, dst in P3_COLUMNS.items():
        out[dst] = df[src].values if src in df.columns else np.nan
    dup = out.duplicated("cell_id", keep=False)
    if dup.any():
        raise SystemExit("duplicate cell_id rows in %s: %s"
                         % (path, sorted(set(out.loc[dup, "cell_id"].tolist()))[:10]))
    return out


def join(p1, p3):
    """Left join on cell_id: every P1 row survives; P3 columns are NaN where
    no merge exists and p3_present says so. Nothing is ever back-filled."""
    m = p1.merge(p3, on="cell_id", how="left", indicator=True)
    m["p3_present"] = m["_merge"] == "both"
    m = m.drop(columns=["_merge"])
    m = m.sort_values(["cell_id", "p1_tree"]).reset_index(drop=True)
    return m[OUT_COLUMNS]


def report(m, p3):
    n_cells = m["cell_id"].nunique()
    n_p3 = int(m.drop_duplicates("cell_id")["p3_present"].sum())
    p3_only = sorted(set(p3["cell_id"].tolist()) - set(m["cell_id"].tolist()))
    print("%s | %d rows, %d cells, %d trees (%s)"
          % (JOIN_VERSION, len(m), n_cells, m["p1_tree"].nunique(),
             ", ".join(sorted(m["p1_tree"].unique()))))
    print("  cells with a P3 (mesh) F: %d / %d; P3-only cells (no P1 row): %d"
          % (n_p3, n_cells, len(p3_only)))
    if p3_only:
        print("  P3-only cell ids: %s" % p3_only[:20])
    have = m[m["p3_present"]]
    if len(have):
        d = have.drop_duplicates("cell_id")
        print("  F_lit_deliverable (%s): n=%d finite=%d mean %.4f sd %.4f"
              % (d["deliverable_variant"].iloc[0], len(d),
                 int(np.isfinite(d["F_lit_deliverable"].astype(float)).sum()),
                 float(np.nanmean(d["F_lit_deliverable"].astype(float))),
                 float(np.nanstd(d["F_lit_deliverable"].astype(float), ddof=1))
                 if len(d) > 1 else float("nan")))
        print("  F_lit_skel_p1 on the same cells: mean %.4f  (comparison bracket, NOT the F)"
              % float(np.nanmean(d["F_lit_skel_p1"].astype(float))))
        q = d["p3_qc_status"].value_counts(dropna=False).to_dict()
        print("  P3 qc_status: %s" % q)
    return p3_only


def main(argv=None):
    args = build_parser().parse_args(argv)
    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        raise SystemExit("--root does not exist: %s" % root)
    p3_path = args.p3_summary or os.path.join(root, "out", "spine_area_F_summary.csv")
    out_path = args.out or os.path.join(root, "out", "campaign_F.csv")
    trees = find_p1_trees(root, args.p1_dir)
    p1 = load_p1(trees)
    p3 = load_p3(p3_path, args.deliverable)
    if not os.path.isfile(p3_path):
        print("NOTE: no P3 summary at %s -- every F_lit_deliverable is NaN" % p3_path)
    m = join(p1, p3)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    m.to_csv(out_path, index=False, lineterminator="\n")
    report(m, p3)
    print("wrote %s" % out_path)
    missing = sorted(m.loc[~m["p3_present"], "cell_id"].unique().tolist())
    if missing:
        print("  %d P1 cell(s) without a P3 row (F_lit_deliverable NaN): %s%s"
              % (len(missing), missing[:20], " ..." if len(missing) > 20 else ""))
        if args.require_complete:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
