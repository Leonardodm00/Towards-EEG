#!/usr/bin/env python3
"""Smoke test for soma_census.py.

    python3 smoke_test_soma_census.py [-v]

Auto-discovered by run_smoke_tests.sh. Offline, no NEURON, no network: it
builds synthetic skeletons whose verdict is known by construction and runs
the real census over them.

Two of the fixtures are the cells soma_enforce.py documents by name, so this
suite fails if that module's own examples ever stop reproducing:
  neuron_606394351  soma radius  331.9 nm -- "a truncated arbour fragment"
  neuron_794820508  soma radius 5325.5 nm -- intact
and a third mimics H01 cell 1302789404 (root 958.4 nm, a thicker 1642.5 nm
node elsewhere), whose soma_area_um2 the pipeline recorded as 11.54336752 on
2026-09-21 -- pinned below, so a change to the area convention is caught here
rather than in a campaign summary.

Sections
  A  verdict classification, the four cases
  B  the area convention, 4*pi*r^2, against the recorded value
  C  the thresholds come from soma_enforce, not from a copy in the census
  D  degenerate skeletons: no root, no r column, unreadable
  E  CLI: --ids-file, --limit, the written schema, atomic rewrite
  F  refusals

Pure ASCII, LF only.
"""
import math
import os
import shutil
import sys
import tempfile

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
STAGE1 = os.environ.get("STAGE1_DIR", os.path.join(HERE, "stage1"))
for _d in (HERE, STAGE1):
    if _d not in sys.path:
        sys.path.insert(0, _d)

VERBOSE = "-v" in sys.argv
PASS = FAIL = 0

# the pipeline's own record for cell 1302789404, 2026-09-21 [run]
RECORDED_SOMA_R_NM = 958.4320528863796
RECORDED_SOMA_AREA_UM2 = 11.54336751538542


def check(label, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print("  [PASS] %-62s %s" % (label, detail if VERBOSE else ""))
    else:
        FAIL += 1
        print("  [FAIL] %-62s %s" % (label, detail))


def skeleton(root_r, other_r=300.0, n=60, fat_elsewhere=None, no_root=False,
             drop_r=False):
    rows = [{"id": 0, "p": (0 if no_root else -1), "x": 0.0, "y": 0.0,
             "z": 0.0, "r": root_r, "annotated_type": "Soma",
             "file_source": "promoted_root"}]
    for i in range(1, n):
        rows.append({"id": i, "p": i - 1, "x": 1000.0 * i, "y": 0.0, "z": 0.0,
                     "r": other_r, "annotated_type": "dendrite",
                     "file_source": "x"})
    if fat_elsewhere is not None:
        rows[n // 2]["r"] = fat_elsewhere
    df = pd.DataFrame(rows)
    return df.drop(columns=["r"]) if drop_r else df


def tree(tmp, cells):
    d = os.path.join(tmp, "neurons")
    os.makedirs(d, exist_ok=True)
    for cid, frame in cells.items():
        frame.to_csv(os.path.join(d, "neuron_%d.csv" % cid), index=False)
    return d


def main():
    import soma_census as SC
    import soma_enforce as se

    tmp = tempfile.mkdtemp(prefix="somacensus_")
    try:
        cells = {
            794820508: skeleton(5325.5),                               # intact
            606394351: skeleton(331.9),                                # fragment
            1302789404: skeleton(958.4320528863796, fat_elsewhere=1642.457886),
            4004: skeleton(6000.0, fat_elsewhere=9000.0),              # thick, but not thickest
        }
        nd = tree(tmp, cells)

        print("== A verdicts")
        rows = {cid: SC.census_one(os.path.join(nd, "neuron_%d.csv" % cid), cid, se)
                for cid in cells}
        for cid, want in ((794820508, "ok"), (606394351, "below_floor"),
                          (1302789404, "both"), (4004, "geometry_disagrees")):
            check("A%d %d -> %s" % (list(cells).index(cid) + 1, cid, want),
                  rows[cid]["verdict"] == want,
                  "%s (root_r %.1f)" % (rows[cid]["verdict"], rows[cid]["root_r_nm"]))
        check("A5 the intact cell is the only one above the floor",
              [r["root_above_floor"] for r in rows.values()].count(True) == 2,
              "4004 is also above it, by construction")
        check("A6 name_geometry_agree is False exactly where a thicker node exists",
              rows[1302789404]["name_geometry_agree"] is False
              and rows[4004]["name_geometry_agree"] is False
              and rows[794820508]["name_geometry_agree"] is True)

        print("== B the area convention")
        a = rows[1302789404]["soma_area_um2"]
        check("B1 soma_area_um2 == 4*pi*r^2 in um",
              abs(a - 4.0 * math.pi * (RECORDED_SOMA_R_NM / 1000.0) ** 2) < 1e-9,
              "%.9f" % a)
        check("B2 matches what the pipeline recorded for cell 1302789404",
              abs(a - RECORDED_SOMA_AREA_UM2) < 1e-6,
              "census %.8f vs record %.8f" % (a, RECORDED_SOMA_AREA_UM2))
        check("B3 the intact example's area is ~31x this one",
              abs(rows[794820508]["soma_area_um2"] / a - 30.876) < 0.01,
              "%.3f" % (rows[794820508]["soma_area_um2"] / a))

        print("== C the thresholds are soma_enforce's, not a copy")
        old = se.DEFAULT_MIN_SOMA_RADIUS_NM
        try:
            se.DEFAULT_MIN_SOMA_RADIUS_NM = 300.0
            r = SC.census_one(os.path.join(nd, "neuron_606394351.csv"),
                              606394351, se)
            check("C1 lowering the floor in soma_enforce flips the verdict",
                  r["verdict"] == "ok", "%s at floor 300" % r["verdict"])
            check("C2 the GEOMETRIC test follows it too -- the census passes the "
                  "threshold explicitly, because identify_soma_by_geometry binds "
                  "its default at import",
                  r["geom_above_floor"] is True,
                  "geom_above_floor=%s" % r["geom_above_floor"])
        finally:
            se.DEFAULT_MIN_SOMA_RADIUS_NM = old
        r = SC.census_one(os.path.join(nd, "neuron_606394351.csv"), 606394351, se)
        check("C1' and restoring it restores the verdict", r["verdict"] == "below_floor")

        print("== D degenerate skeletons")
        dd = tree(os.path.join(tmp, "deg"), {
            5001: skeleton(3000.0, no_root=True),
            5002: skeleton(3000.0, drop_r=True)})
        check("D1 no p == -1 node -> no_root",
              SC.census_one(os.path.join(dd, "neuron_5001.csv"), 5001, se)["verdict"]
              == "no_root")
        check("D2 no r column -> no_r_column, not a crash",
              SC.census_one(os.path.join(dd, "neuron_5002.csv"), 5002, se)["verdict"]
              == "no_r_column")
        check("D3 an unreadable path is a verdict, not an exception",
              SC.census_one(os.path.join(dd, "nope.csv"), 5003, se)["verdict"]
              in ("unreadable", "no_r_column"))

        print("== E the CLI")
        out = os.path.join(tmp, "census.csv")
        rc = SC.main(["--neurons-dir", nd, "--stage1-dir", STAGE1, "--out", out])
        check("E1 exit 0 and the file exists", rc == 0 and os.path.isfile(out))
        df = pd.read_csv(out)
        check("E2 schema is COLUMNS, in order", list(df.columns) == list(SC.COLUMNS),
              str(list(df.columns)[:4]))
        check("E3 one row per skeleton", len(df) == len(cells))
        check("E4 high-confidence count is 1 of 4 here",
              int((df["verdict"] == "ok").sum()) == 1)
        idf = os.path.join(tmp, "ids.txt")
        with open(idf, "w") as fh:
            fh.write("# comment\n606394351\n1302789404\n\n")
        out2 = os.path.join(tmp, "subset.csv")
        SC.main(["--neurons-dir", nd, "--stage1-dir", STAGE1, "--ids-file", idf,
                 "--out", out2])
        d2 = pd.read_csv(out2)
        check("E5 --ids-file restricts, comments and blanks skipped",
              sorted(d2["cell_id"]) == [606394351, 1302789404], str(list(d2["cell_id"])))
        out3 = os.path.join(tmp, "lim.csv")
        SC.main(["--neurons-dir", nd, "--stage1-dir", STAGE1, "--limit", "2",
                 "--out", out3])
        check("E6 --limit stops early", len(pd.read_csv(out3)) == 2)
        SC.main(["--neurons-dir", nd, "--stage1-dir", STAGE1, "--out", out])
        check("E7 rewriting is idempotent and leaves no .tmp",
              len(pd.read_csv(out)) == len(cells)
              and not [f for f in os.listdir(tmp) if f.endswith(".tmp")])

        print("== F refusals")
        for argv, why in ((["--neurons-dir", os.path.join(tmp, "nope")], "missing dir"),
                          (["--neurons-dir", nd, "--ids-file",
                            os.path.join(tmp, "nope.txt")], "missing ids file")):
            try:
                SC.main(argv + ["--stage1-dir", STAGE1])
                check("F refuses: %s" % why, False, "accepted")
            except SystemExit as e:
                check("F refuses: %s" % why, "does not exist" in str(e), str(e)[:60])
        empty = os.path.join(tmp, "empty", "neurons")
        os.makedirs(empty, exist_ok=True)
        try:
            SC.main(["--neurons-dir", empty, "--stage1-dir", STAGE1])
            check("F refuses: no skeletons at all", False, "accepted")
        except SystemExit as e:
            check("F refuses: no skeletons at all", "no neuron_" in str(e), str(e)[:60])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    print("%d checks passed, %d failed" % (PASS, FAIL))
    print("ALL GREEN" if FAIL == 0 else "FAILURES ABOVE")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
