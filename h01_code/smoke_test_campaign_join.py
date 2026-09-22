#!/usr/bin/env python3
"""Smoke test for campaign_join.py (decision D-004).

  python3 smoke_test_campaign_join.py        (quiet)
  python3 smoke_test_campaign_join.py -v     (every check)

Fixture: a campaign root with three P1 trees (p1 for exc, p1_inh_SST and
p1_inh_PVVIP for the same two interneurons) and a P3 summary that covers some
cells and not others. Asserts the join is per (cell, tree), that the mesh F
lands in F_lit_deliverable ONLY from P3 and is NaN -- never the skeleton --
where P3 is absent, that the interneuron's one P3 row joins to both trees,
that a foreign deliverable variant is refused, --require-complete exits 1
naming the cells, the run is idempotent, and every refusal fires.

Pure ASCII, LF only.
"""

import os
import shutil
import sys
import tempfile
import traceback

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import campaign_join as J

VERBOSE = "-v" in sys.argv
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    if VERBOSE or not ok:
        print("  [%s] %-60s %s" % ("PASS" if ok else "FAIL", name, detail))
    return ok


EXC = (1001, 1002, 1003)          # exc cells, tree p1
INH = (2001, 2002)                # inh cells, trees p1_inh_SST and p1_inh_PVVIP


def p1_row(cid, layer, ctype, tree, F):
    return {"cell_id": cid, "layer": layer, "cell_type": ctype, "layer_source": "bank",
            "status": "ok", "error": "", "qc_status": "pass", "reasons": "",
            "gate_status": "pass", "hoc_verdict": "pass", "quarantined": False,
            "n_spines": 100 + cid % 7, "cm": 0.5 if ctype == "exc" else (1.0 if "SST" in tree else 2.0),
            "Ra": 268.5 if ctype == "exc" else 100.0,
            "passive_table": "passive_params.csv" if tree == "p1" else "passive_params_inh_%s.csv" % tree.split("_")[-1],
            "F_lit": F, "F_lit_nocap": F - 0.01, "fingerprint": "abc%d" % cid}


def p3_row(cid, F_deliv, F_skel, variant="mesh_beyond", qc="pass"):
    return {"cell_id": cid, "module_version": "h01_spine_area_F v1.7",
            "n_spines": 100 + cid % 7, "n_measured": 100 + cid % 7 - 1,
            "deliverable_variant": variant, "F_lit_deliverable": F_deliv,
            "F_whole_deliverable": F_deliv - 0.1, "F_lit_skel": F_skel,
            "F_lit_mesh": F_deliv + 0.02, "qc_status": qc, "qc_reason": "",
            "coverage_count_deliverable": 0.995, "fallback_area_frac_deliverable": 0.003,
            "min_coverage": 0.99}


def build_root(tmp, p3_cells=(1001, 1002, 2001), variant_of=None):
    root = os.path.join(tmp, "h01")
    shutil.rmtree(root, ignore_errors=True)
    for d in ("p1", "p1_inh_SST", "p1_inh_PVVIP", "out", "neurons"):
        os.makedirs(os.path.join(root, d))
    pd.DataFrame([p1_row(c, "L3", "exc", "p1", 1.50 + 0.01 * (c % 10)) for c in EXC]).to_csv(
        os.path.join(root, "p1", "p1_summary.csv"), index=False, lineterminator="\n")
    for tree in ("p1_inh_SST", "p1_inh_PVVIP"):
        pd.DataFrame([p1_row(c, "L2", "inh", tree, 1.10 + 0.01 * (c % 10)) for c in INH]).to_csv(
            os.path.join(root, tree, "p1_summary.csv"), index=False, lineterminator="\n")
    if p3_cells:
        rows = []
        for c in p3_cells:
            v = (variant_of or {}).get(c, "mesh_beyond")
            rows.append(p3_row(c, 1.80 + 0.01 * (c % 10), 1.50 + 0.01 * (c % 10), v))
        pd.DataFrame(rows).to_csv(os.path.join(root, "out", "spine_area_F_summary.csv"),
                                  index=False, lineterminator="\n")
    return root


def main():
    tmp = tempfile.mkdtemp()
    try:
        root = build_root(tmp)
        out = os.path.join(root, "out", "campaign_F.csv")

        # ---- A the join
        rc = J.main(["--root", root])
        check("A1 join exits 0 and writes out/campaign_F.csv", rc == 0 and os.path.isfile(out))
        m = pd.read_csv(out)
        check("A2 one row per (cell, tree): 3 exc + 2 inh x 2 trees = 7",
              len(m) == 7 and m.duplicated(["cell_id", "p1_tree"]).sum() == 0
              and set(m["p1_tree"]) == {"p1", "p1_inh_SST", "p1_inh_PVVIP"}, str(len(m)))
        check("A3 column order: cell_id, p1_tree, P1 block, p3_present, P3 block",
              list(m.columns) == J.OUT_COLUMNS)
        check("A3' no raw 'F_lit' column survives (the trap is closed by renaming)",
              "F_lit" not in m.columns and "F_lit_skel_p1" in m.columns
              and "F_lit_deliverable" in m.columns)
        r = m[(m["cell_id"] == 1001) & (m["p1_tree"] == "p1")].iloc[0]
        check("A4 cell with P3: p3_present True, F_lit_deliverable is P3's value, F_lit_skel_p1 is P1's",
              bool(r["p3_present"]) and abs(r["F_lit_deliverable"] - 1.81) < 1e-9
              and abs(r["F_lit_skel_p1"] - 1.51) < 1e-9 and r["deliverable_variant"] == "mesh_beyond"
              and r["p3_qc_status"] == "pass" and abs(r["F_lit_skel_p3"] - 1.51) < 1e-9,
              "deliv %.3f skel %.3f" % (r["F_lit_deliverable"], r["F_lit_skel_p1"]))
        r = m[(m["cell_id"] == 1003) & (m["p1_tree"] == "p1")].iloc[0]
        check("A5 cell WITHOUT P3: p3_present False, F_lit_deliverable NaN -- NOT the skeleton value",
              not bool(r["p3_present"]) and not np.isfinite(r["F_lit_deliverable"])
              and np.isfinite(r["F_lit_skel_p1"]) and pd.isna(r["deliverable_variant"]),
              "deliv %s skel %.3f" % (r["F_lit_deliverable"], r["F_lit_skel_p1"]))
        inh = m[m["cell_id"] == 2001]
        check("A6 interneuron 2001: its ONE P3 row joins to BOTH trees with the same mesh F (D-004 item 2)",
              len(inh) == 2 and set(inh["p1_tree"]) == {"p1_inh_SST", "p1_inh_PVVIP"}
              and inh["p3_present"].all() and inh["F_lit_deliverable"].nunique() == 1
              and inh["cm"].nunique() == 2 and inh["F_lit_skel_p1"].nunique() == 1,
              str(inh[["p1_tree", "cm", "F_lit_deliverable"]].values.tolist()))
        inh2 = m[m["cell_id"] == 2002]
        check("A6' interneuron 2002 without P3: two rows, both NaN deliverable",
              len(inh2) == 2 and not inh2["p3_present"].any()
              and not np.isfinite(inh2["F_lit_deliverable"]).any())
        check("A7 P1 provenance carried per tree: passive_table and cm differ between the inh trees",
              set(inh["passive_table"]) == {"passive_params_inh_SST.csv", "passive_params_inh_PVVIP.csv"})

        # ---- B idempotent, explicit trees, no P3 at all
        m1 = pd.read_csv(out)
        J.main(["--root", root])
        m2 = pd.read_csv(out)
        check("B1 re-running writes an identical table", m1.equals(m2))
        rc = J.main(["--root", root, "--p1-dir", "p1", "--out", os.path.join(tmp, "exc_only.csv")])
        e = pd.read_csv(os.path.join(tmp, "exc_only.csv"))
        check("B2 --p1-dir p1 restricts to that tree", rc == 0 and len(e) == 3 and set(e["p1_tree"]) == {"p1"})
        root0 = build_root(tmp, p3_cells=())
        rc = J.main(["--root", root0])
        m0 = pd.read_csv(os.path.join(root0, "out", "campaign_F.csv"))
        check("B3 no P3 summary yet: table still written, every deliverable NaN, p3_present all False",
              rc == 0 and len(m0) == 7 and not m0["p3_present"].any()
              and not np.isfinite(m0["F_lit_deliverable"]).any()
              and np.isfinite(m0["F_lit_skel_p1"]).all())

        # ---- C --require-complete
        root = build_root(tmp)
        rc = J.main(["--root", root, "--require-complete"])
        check("C1 --require-complete exits 1 when a P1 cell has no P3 row (table still written)",
              rc == 1 and os.path.isfile(os.path.join(root, "out", "campaign_F.csv")))
        rootc = build_root(tmp, p3_cells=EXC + INH)
        rc = J.main(["--root", rootc, "--require-complete"])
        mc = pd.read_csv(os.path.join(rootc, "out", "campaign_F.csv"))
        check("C2 --require-complete exits 0 when every cell has a P3 row",
              rc == 0 and mc["p3_present"].all() and len(mc) == 7)

        # ---- D refusals
        rootv = build_root(tmp, variant_of={1002: "mesh"})
        try:
            J.main(["--root", rootv])
            check("D1 a P3 row with deliverable_variant != mesh_beyond is REFUSED", False, "accepted")
        except SystemExit as ex:
            check("D1 a P3 row with deliverable_variant != mesh_beyond is REFUSED",
                  "deliverable_variant is not mesh_beyond" in str(ex) and "1002" in str(ex), str(ex)[:80])
        check("D1' ...and nothing was written", not os.path.isfile(os.path.join(rootv, "out", "campaign_F.csv")))
        try:
            J.main(["--root", rootv, "--deliverable", "mesh"])
            check("D1'' --deliverable mesh: still refused, for the mesh_beyond rows (a MIXED table is the error)",
                  False, "accepted")
        except SystemExit as ex:
            check("D1'' --deliverable mesh: still refused, for the mesh_beyond rows (a MIXED table is the error)",
                  "1001" in str(ex) and "1002" not in str(ex), str(ex)[:80])
        try:
            J.main(["--root", os.path.join(tmp, "nope")])
            check("D2 missing root refused", False)
        except SystemExit:
            check("D2 missing root refused", True)
        roote = os.path.join(tmp, "empty"); os.makedirs(roote, exist_ok=True)
        try:
            J.main(["--root", roote])
            check("D3 root with no P1 tree refused", False)
        except SystemExit as ex:
            check("D3 root with no P1 tree refused", "no P1 tree" in str(ex))
        try:
            J.main(["--root", root, "--p1-dir", "p1_missing"])
            check("D4 --p1-dir without a p1_summary.csv refused", False)
        except SystemExit as ex:
            check("D4 --p1-dir without a p1_summary.csv refused", "p1_summary.csv" in str(ex))
        rootd = build_root(tmp)
        s = pd.read_csv(os.path.join(rootd, "p1", "p1_summary.csv"))
        pd.concat([s, s.iloc[[0]]]).to_csv(os.path.join(rootd, "p1", "p1_summary.csv"), index=False)
        try:
            J.main(["--root", rootd])
            check("D5 duplicate cell in a P1 summary refused", False)
        except SystemExit as ex:
            check("D5 duplicate cell in a P1 summary refused", "duplicate" in str(ex))
        rootn = build_root(tmp)
        s = pd.read_csv(os.path.join(rootn, "out", "spine_area_F_summary.csv"))
        s.drop(columns=["deliverable_variant"]).to_csv(
            os.path.join(rootn, "out", "spine_area_F_summary.csv"), index=False)
        try:
            J.main(["--root", rootn])
            check("D6 P3 summary without deliverable_variant refused (never guess the variant)", False)
        except SystemExit as ex:
            check("D6 P3 summary without deliverable_variant refused (never guess the variant)",
                  "deliverable_variant" in str(ex))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    n_fail = sum(1 for _, ok in RESULTS if not ok)
    print("\n%d checks passed, %d failed" % (len(RESULTS) - n_fail, n_fail))
    print("ALL GREEN" if n_fail == 0 else "FAILURES")
    return n_fail


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
