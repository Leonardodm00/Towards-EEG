#!/usr/bin/env python3
"""How much of each cell's reported spine area is mesh, and how much is fill.

  python3 campaign_policy_report.py --root ../h01
  python3 campaign_policy_report.py --root ../h01 --cell 1302789404 -v
  python3 campaign_policy_report.py --root ../h01 --tol 0.02   # bracket labels

DECISION D-008 (2026-09-22) closed both policies D-004 had left open:

  (1) COVERAGE. The kappa fill STAYS. No cell is dropped and no threshold is
      set; `--min-coverage` (0.99) remains a flag on qc_status and nothing
      else. What is required instead is that the fraction of the reported
      spine area which did NOT come from the mesh is REPORTED -- which is
      what this script exists for.
  (2) shaft_terminates. Those components are KEPT AS SPINES, carrying their
      kappa-filled skeleton area, which is what the code already does. They
      are not demoted.

So the deliverable F is `as_reported` and nothing here proposes changing it.
The other policies below are kept as labelled COMPARISON BRACKETS: each says
how much of F rests on the class it removes. A bracket is never the reported
F (D-004), and none of them licenses a re-merge -- they are counterfactuals
on the recorded areas, not re-measurements.

The numbers are computed exactly, post-hoc, from the two CSVs P3 writes --

  <root>/out/cell<id>_spines.csv       one row per spine, with seg_from/seg_to
  <root>/out/neuron_<id>_phi_mesh.csv  one row per segment, d_from_um, areas

-- by the SAME attribution the assembler uses: group the per-spine area on
(seg_from, seg_to), map it onto the phi rows, and take
F = 1 + sum(A_spine)/sum(A_shaft) over segments with d_from_um >= 60
(`h01_spine_area_F.phi_with_spine_areas` + `cell_F`, read 2026-09-22).
Nothing is re-measured and no measurement code is touched or imported.

Every cell is first CHECKED: the F this script reconstructs with all spines
kept must equal the `F_lit_deliverable` P3 recorded. A cell that fails that
reproduction is reported and excluded -- its other numbers would be
meaningless.

Output: <root>/out/campaign_policy_report.csv, one row per cell, plus the
printed campaign figure: per cell and POOLED (a ratio of sums, never a mean
of ratios), the fraction of reported spine area that is kappa fill rather
than mesh measurement -- whole-cell, as P3 records it, and restricted to the
segments beyond the 60 um cutoff, which are the only ones F_lit sees. The
two are different quantities and are named apart.

Pure ASCII, LF only.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

REPORT_VERSION = "campaign_policy_report v1.0"
F_CUTOFF_UM = 60.0                     # h01_spine_area_F.F_CUTOFF_UM
DELIVERABLE_AREA_COL = "A_used_beyond_um2"      # the mesh_beyond track (D-004)
DELIVERABLE_MEASURED_COL = "measured_beyond"

# name -> what the F under it means. Only `as_reported` is ever the F
# (D-004); the rest are comparison brackets that size what a class is worth.
POLICIES = (
    ("as_reported",
     "THE DELIVERABLE: every spine, measured or kappa-filled (D-004, D-008)"),
    ("drop_unmeasured",
     "bracket: the kappa fill removed -- mesh-measured spines only"),
    ("drop_shaft_terminates",
     "bracket: shaft_terminates demoted -- NOT the policy (D-008 keeps them)"),
    ("drop_not_reaching_box",
     "bracket: components whose shaft never crosses the ROI box removed"),
)


# --------------------------------------------------------------------- #
# loading                                                               #
# --------------------------------------------------------------------- #
def cell_ids_in(out_dir, summary):
    if summary is not None and len(summary):
        return [int(c) for c in summary["cell_id"].tolist()]
    ids = []
    for f in sorted(os.listdir(out_dir)):
        if f.startswith("cell") and f.endswith("_spines.csv"):
            try:
                ids.append(int(f[len("cell"):-len("_spines.csv")]))
            except ValueError:
                pass
    return ids


def load_summary(out_dir):
    p = os.path.join(out_dir, "spine_area_F_summary.csv")
    if not os.path.isfile(p):
        raise SystemExit(
            "no P3 summary at %s -- this script reports on FINISHED cells; run "
            "the campaign first (campaign.pbs), then come back." % p)
    return pd.read_csv(p)


def load_cell(out_dir, cell_id):
    """(spines, phi) for one cell, or (None, None) with a reason."""
    sp = os.path.join(out_dir, "cell%d_spines.csv" % int(cell_id))
    ph = os.path.join(out_dir, "neuron_%d_phi_mesh.csv" % int(cell_id))
    for p in (sp, ph):
        if not os.path.isfile(p):
            return None, None, "missing %s" % os.path.basename(p)
    return pd.read_csv(sp), pd.read_csv(ph), ""


# --------------------------------------------------------------------- #
# computation -- no I/O, no printing                                    #
# --------------------------------------------------------------------- #
def f_from_phi(phi, spine_area, cutoff_um=F_CUTOFF_UM):
    """F = 1 + sum A_spine / sum A_shaft over rows with d_from_um >= cutoff.

    `spine_area` is an array aligned with phi's rows. Mirrors
    h01_spine_area_F.cell_F, which is the function that produced the recorded
    F; it is reimplemented here (two lines) rather than imported so this
    script never pulls in the measurement stack."""
    sel = phi["d_from_um"].to_numpy(dtype=float) >= float(cutoff_um)
    a_sh = float(phi["shaft_area_um2"].to_numpy(dtype=float)[sel].sum())
    a_sp = float(np.asarray(spine_area, dtype=float)[sel].sum())
    return {"F": (1.0 + a_sp / a_sh) if a_sh > 0 else float("nan"),
            "A_spine_um2": a_sp, "A_shaft_um2": a_sh,
            "n_segments": int(sel.sum())}


def attribute(spines, phi, area_col, keep):
    """Per-segment spine area for the kept spines, in phi's row order.

    The attribution of h01_spine_area_F.phi_with_spine_areas: sum `area_col`
    over (seg_from, seg_to), look each phi row's (node_from, node_to) up in
    that sum, 0.0 where no spine maps. Spines with seg_from < 0 are unmapped
    and are excluded there too."""
    m = spines[(spines["seg_from"] >= 0) & np.asarray(keep, dtype=bool)]
    sums = m.groupby(["seg_from", "seg_to"])[area_col].sum()
    keys = list(zip(phi["node_from"].astype(np.int64),
                    phi["node_to"].astype(np.int64)))
    return np.array([float(sums.get(k, 0.0)) for k in keys])


def keep_masks(spines):
    """One boolean mask per policy, aligned with `spines`."""
    n = len(spines)
    all_true = np.ones(n, dtype=bool)

    def col(name, default):
        if name not in spines.columns:
            return pd.Series([default] * n, index=spines.index)
        return spines[name]

    measured = col(DELIVERABLE_MEASURED_COL, False).astype("boolean").fillna(False)
    verdict = col("base_verdict", "").astype(str)
    reaches = col("shaft_reaches_box", True).astype("boolean").fillna(True)
    return {
        "as_reported": all_true,
        "drop_unmeasured": measured.to_numpy(dtype=bool),
        "drop_shaft_terminates": (verdict != "shaft_terminates").to_numpy(dtype=bool),
        "drop_not_reaching_box": reaches.to_numpy(dtype=bool),
    }


def cell_row(cell_id, spines, phi, summary_row, cutoff_um=F_CUTOFF_UM,
             area_col=DELIVERABLE_AREA_COL):
    """One row: F under each policy, its shift from as_reported, the counts."""
    row = {"cell_id": int(cell_id), "n_spines": int(len(spines))}
    if area_col not in spines.columns:
        row["reproduces"] = False
        row["note"] = "no %s column (deliverable track absent)" % area_col
        return row
    masks = keep_masks(spines)
    areas = {}
    for name, _ in POLICIES:
        f = f_from_phi(phi, attribute(spines, phi, area_col, masks[name]), cutoff_um)
        row["F_" + name] = f["F"]
        row["n_kept_" + name] = int(np.asarray(masks[name], dtype=bool).sum())
        areas[name] = f["A_spine_um2"]
        row["A_shaft_beyond_um2"] = f["A_shaft_um2"]
    base = row["F_as_reported"]
    for name, _ in POLICIES:
        if name != "as_reported":
            row["dF_" + name] = row["F_" + name] - base

    # D-008's reported quantity, restricted to what F_lit actually sees: the
    # spine area beyond the cutoff that came from the kappa fill rather than
    # from the mesh. Kept as an area as well as a fraction, so the campaign
    # figure can be a ratio of sums rather than a mean of per-cell ratios.
    tot = float(areas["as_reported"])
    filled = tot - float(areas["drop_unmeasured"])
    row["A_spine_beyond_um2"] = tot
    row["A_spine_beyond_filled_um2"] = filled
    row["frac_area_filled_beyond"] = (filled / tot) if tot > 0 else float("nan")
    st = tot - float(areas["drop_shaft_terminates"])
    row["A_spine_beyond_shaft_terminates_um2"] = st
    row["frac_area_shaft_terminates_beyond"] = (st / tot) if tot > 0 else float("nan")

    # reproduction check, twice over: the phi column P3 wrote, and this
    # script's own re-attribution, must both give the recorded F.
    f_phi = f_from_phi(phi, phi["spine_area_um2"].to_numpy(dtype=float), cutoff_um)
    rec = float(summary_row["F_lit_deliverable"]) if summary_row is not None else float("nan")
    row["F_recorded"] = rec
    row["F_from_phi_column"] = f_phi["F"]
    ok = (np.isfinite(rec) and np.isfinite(base)
          and abs(base - rec) < 1e-6 and abs(f_phi["F"] - rec) < 1e-6)
    row["reproduces"] = bool(ok)
    row["note"] = "" if ok else "reconstruction %.6f / phi %.6f vs recorded %.6f" % (
        base, f_phi["F"], rec)

    if summary_row is not None:
        for k in ("coverage_count_deliverable", "fallback_area_frac_deliverable",
                  "qc_status", "n_shaft_terminates", "n_base_by_area_drop",
                  "n_shaft_not_reaching_box", "n_measured_mesh_beyond",
                  "n_fallback_mesh_beyond", "min_coverage"):
            if k in summary_row:
                row[k] = summary_row[k]
    return row


def population_view(df, tol):
    """What the rows say, as numbers -- no prose, no printing.

    The campaign fractions are POOLED: sum of filled area over sum of total
    area, not the mean of the per-cell fractions. A mean of ratios would let
    a tiny cell with one unmeasured spine count as much as a large one, which
    is the same reason kappa itself is a ratio of sums
    (`h01_spine_area_F.kappa_function`)."""
    ok = df[df["reproduces"].astype(bool)] if "reproduces" in df else df
    out = {"n_cells": int(len(df)), "n_reproduced": int(len(ok)),
           "tol": float(tol)}
    if not len(ok):
        return out
    F = ok["F_as_reported"].astype(float)
    out["F_mean"] = float(np.nanmean(F))
    out["F_sd"] = float(np.nanstd(F, ddof=1)) if len(ok) > 1 else float("nan")
    out["F_min"], out["F_max"] = float(np.nanmin(F)), float(np.nanmax(F))

    # ---- D-008's figure: how much of the reported spine area is not mesh
    def pooled(num_col, den_col):
        num = float(np.nansum(ok[num_col].astype(float))) if num_col in ok else float("nan")
        den = float(np.nansum(ok[den_col].astype(float))) if den_col in ok else float("nan")
        return (num / den) if den and np.isfinite(den) and den > 0 else float("nan")

    out["frac_area_filled_beyond_pooled"] = pooled("A_spine_beyond_filled_um2",
                                                   "A_spine_beyond_um2")
    out["frac_area_shaft_terminates_beyond_pooled"] = pooled(
        "A_spine_beyond_shaft_terminates_um2", "A_spine_beyond_um2")
    for col in ("frac_area_filled_beyond", "fallback_area_frac_deliverable"):
        if col in ok:
            v = ok[col].astype(float)
            out[col + "_median"] = float(np.nanmedian(v))
            out[col + "_max"] = float(np.nanmax(v))
            out[col + "_worst_cell"] = (int(ok.loc[v.idxmax(), "cell_id"])
                                        if np.isfinite(v).any() else None)
    if "coverage_count_deliverable" in ok:
        c = ok["coverage_count_deliverable"].astype(float)
        out["coverage_count_min"] = float(np.nanmin(c))
        out["n_below_min_coverage"] = int((c < ok.get(
            "min_coverage", pd.Series(0.99, index=ok.index)).astype(float)).sum())

    # ---- the brackets, in units of F
    for name, _ in POLICIES:
        if name == "as_reported":
            continue
        d = ok["dF_" + name].astype(float).abs()
        out["n_material_" + name] = int((d > tol).sum())
        out["max_abs_dF_" + name] = float(np.nanmax(d)) if len(d) else float("nan")
        out["median_abs_dF_" + name] = float(np.nanmedian(d)) if len(d) else float("nan")
    return out


# --------------------------------------------------------------------- #
# reporting                                                             #
# --------------------------------------------------------------------- #
def print_report(df, view, tol):
    print("%s | %d cells, %d reproduce the recorded F"
          % (REPORT_VERSION, view["n_cells"], view["n_reproduced"]))
    bad = df[~df["reproduces"].astype(bool)] if "reproduces" in df else df.iloc[:0]
    if len(bad):
        print("  EXCLUDED, F not reproduced (their numbers are void):")
        for _, r in bad.iterrows():
            print("    cell %s: %s" % (r["cell_id"], r.get("note", "")))
    if not view["n_reproduced"]:
        print("  nothing to report on.")
        return
    print("  F (mesh_beyond, as reported): mean %.4f  sd %.4f  range %.4f--%.4f"
          % (view["F_mean"], view["F_sd"], view["F_min"], view["F_max"]))
    print()

    # ---- the reported figure (D-008) ---------------------------------
    print("  MESH COVERAGE OF THE REPORTED SPINE AREA (decision D-008)")
    print("  The kappa fill stays and no cell is dropped; this is the number")
    print("  that travels with F.")
    pb = view.get("frac_area_filled_beyond_pooled", float("nan"))
    print("    beyond the %.0f um cutoff, i.e. the area F_lit actually sees:"
          % F_CUTOFF_UM)
    print("      campaign pooled (sum of filled / sum of total): %.4f  (%.2f %%)"
          % (pb, 100.0 * pb))
    if "frac_area_filled_beyond_median" in view:
        print("      per cell: median %.4f, worst %.4f (cell %s)"
              % (view["frac_area_filled_beyond_median"],
                 view["frac_area_filled_beyond_max"],
                 view["frac_area_filled_beyond_worst_cell"]))
    if "fallback_area_frac_deliverable_median" in view:
        print("    whole cell, as P3 records it in "
              "fallback_area_frac_deliverable:")
        print("      per cell: median %.4f, worst %.4f (cell %s)"
              % (view["fallback_area_frac_deliverable_median"],
                 view["fallback_area_frac_deliverable_max"],
                 view["fallback_area_frac_deliverable_worst_cell"]))
    print("    the two differ on purpose: F_lit is computed only beyond the")
    print("    cutoff, while P3's column covers every spine of the cell.")
    if "coverage_count_min" in view:
        print("    spine COUNT coverage: worst cell %.4f; %d cell(s) below their"
              " own min_coverage (a flag on qc_status, not an exclusion)"
              % (view["coverage_count_min"], view["n_below_min_coverage"]))
    print()

    # ---- the brackets -------------------------------------------------
    print("  COMPARISON BRACKETS -- what F would be if a class were removed.")
    print("  None of these is the reported F, and none licenses a re-merge.")
    print("  %-24s %8s %8s %10s   %s" % ("bracket", "median", "max", ">tol", "meaning"))
    for name, meaning in POLICIES:
        if name == "as_reported":
            continue
        print("  %-24s %8.4f %8.4f %5d/%-4d   %s"
              % (name, view["median_abs_dF_" + name], view["max_abs_dF_" + name],
                 view["n_material_" + name], view["n_reproduced"], meaning))
    print("  (tol = %.3f, against a between-cell F sd of %.4f)"
          % (tol, view["F_sd"]))
    ps = view.get("frac_area_shaft_terminates_beyond_pooled", float("nan"))
    print("    shaft_terminates components hold %.4f (%.2f %%) of the pooled "
          "beyond-cutoff spine area." % (ps, 100.0 * ps))
    print("    D-008 keeps them as spines, so that area is IN the reported F;")
    print("    the bracket above says what dropping them would cost.")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", required=True, help="campaign root (holds out/)")
    p.add_argument("--out-dir", default=None, help="default <root>/out")
    p.add_argument("--out", default=None,
                   help="default <out-dir>/campaign_policy_report.csv")
    p.add_argument("--cell", type=int, action="append", default=None,
                   help="restrict to this cell; repeatable")
    p.add_argument("--tol", type=float, default=0.01,
                   help="|dF| above which a policy is material (default 0.01)")
    p.add_argument("--cutoff-um", type=float, default=F_CUTOFF_UM,
                   help="the F cutoff (default 60, as the deliverable uses)")
    p.add_argument("-v", "--verbose", action="store_true", help="per-cell lines")
    args = p.parse_args(argv)

    root = os.path.abspath(args.root)
    out_dir = args.out_dir or os.path.join(root, "out")
    if not os.path.isdir(out_dir):
        raise SystemExit("no %s -- P2/P3 have not run yet" % out_dir)
    summary = load_summary(out_dir)
    ids = args.cell or cell_ids_in(out_dir, summary)
    if not ids:
        raise SystemExit("no finished cells under %s" % out_dir)

    rows, skipped = [], []
    by_id = {int(r["cell_id"]): r for _, r in summary.iterrows()}
    for cid in ids:
        spines, phi, why = load_cell(out_dir, cid)
        if spines is None:
            skipped.append((cid, why))
            continue
        rows.append(cell_row(cid, spines, phi, by_id.get(int(cid)),
                             args.cutoff_um))
        if args.verbose:
            r = rows[-1]
            print("  cell %d: F %.4f | dF fill %+.4f | dF shaft_term %+.4f | %s"
                  % (cid, r.get("F_as_reported", float("nan")),
                     r.get("dF_drop_unmeasured", float("nan")),
                     r.get("dF_drop_shaft_terminates", float("nan")),
                     "ok" if r.get("reproduces") else r.get("note", "")))
    if skipped:
        print("  %d cell(s) skipped: %s" % (len(skipped), skipped[:5]))
    df = pd.DataFrame(rows).sort_values("cell_id").reset_index(drop=True)
    out_path = args.out or os.path.join(out_dir, "campaign_policy_report.csv")
    df.to_csv(out_path, index=False, lineterminator="\n")
    view = population_view(df, args.tol)
    print_report(df, view, args.tol)
    print("wrote %s" % out_path)
    return 0 if view["n_reproduced"] == view["n_cells"] else 1


if __name__ == "__main__":
    sys.exit(main())
