#!/usr/bin/env python3
"""Smoke test for campaign_policy_report.py and campaign_queue_check.py.

  python3 smoke_test_campaign_policy.py        (quiet)
  python3 smoke_test_campaign_policy.py -v     (every check)

Section A -- the policy report, against a hand-computed fixture. Four
segments, three of them beyond the 60 um cutoff with shaft areas 2 + 3 + 4 =
9 um2, and six spines whose areas are chosen so every policy's F can be
written down by hand:

  spine  seg    d_from  A_used_beyond  measured  base_verdict       reaches
  1      (1,2)  10      5.0            yes       ok                 yes
  2      (2,3)  70      1.0            yes       ok                 yes
  3      (2,3)  70      2.0            NO        ok                 yes
  4      (3,4)  80      3.0            yes       shaft_terminates   yes
  5      (4,5)  90      0.5            yes       ok                 NO
  6      unmapped (seg_from = -1)      100.0     yes       ok       yes

Spine 1 is below the cutoff and spine 6 is unmapped, so neither may ever
reach F -- spine 6 is the tripwire: at 100 um2 any leak is unmissable.

  as_reported            1 + (1+2+3+0.5)/9 = 1 + 6.5/9
  drop_unmeasured        1 + (1+3+0.5)/9   = 1 + 4.5/9 = 1.5
  drop_shaft_terminates  1 + (1+2+0.5)/9   = 1 + 3.5/9
  drop_not_reaching_box  1 + (1+2+3)/9     = 1 + 6.0/9

A second cell carries a deliberately wrong recorded F, to prove a cell whose
F cannot be reproduced is excluded rather than reported on.

Decision D-008 (2026-09-22) closed both policies, so section A also asserts
what the script must now do instead of proposing a threshold: report the
fraction of reported spine area that is kappa fill rather than mesh, pooled
as a ratio of sums, with shaft_terminates components counted INSIDE F.

Section B -- the queue check, against captured qstat text: PBS line folding,
the `[u:PBS_GENERIC=N]` limit spelling, the fit/no-fit verdict, the
chunk-size advice, and the case where no limit can be read at all (which
must be UNKNOWN and exit 2, never a silent yes).

Pure ASCII, LF only.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import traceback

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import campaign_policy_report as P
import campaign_queue_check as Q

VERBOSE = "-v" in sys.argv
RESULTS = []
CELL_OK, CELL_BAD = 111, 222

F_AS_REPORTED = 1.0 + 6.5 / 9.0
F_DROP_UNMEASURED = 1.0 + 4.5 / 9.0
F_DROP_SHAFT_TERM = 1.0 + 3.5 / 9.0
F_DROP_NOT_BOX = 1.0 + 6.0 / 9.0


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    if VERBOSE or not ok:
        print("  [%s] %-62s %s" % ("PASS" if ok else "FAIL", name, detail))
    return ok


def spines_frame():
    rows = [
        # sigma, seg_from, seg_to, A_used_beyond, measured, verdict, reaches
        (1, 1, 2, 5.0, True, "ok", True),
        (2, 2, 3, 1.0, True, "ok", True),
        (3, 2, 3, 2.0, False, "ok", True),
        (4, 3, 4, 3.0, True, "shaft_terminates", True),
        (5, 4, 5, 0.5, True, "ok", False),
        (6, -1, -1, 100.0, True, "ok", True),
    ]
    return pd.DataFrame(
        [{"sigma_id": s, "seg_from": f, "seg_to": t, "A_used_beyond_um2": a,
          "A_skel_um2": a, "measured_beyond": m, "base_verdict": v,
          "shaft_reaches_box": r, "ok": True, "clipped": False}
         for s, f, t, a, m, v, r in rows])


def phi_frame():
    seg = [(1, 2, 10.0, 1.0, 5.0), (2, 3, 70.0, 2.0, 3.0),
           (3, 4, 80.0, 3.0, 3.0), (4, 5, 90.0, 4.0, 0.5)]
    return pd.DataFrame(
        [{"node_from": a, "node_to": b, "d_from_um": d, "shaft_area_um2": sh,
          "spine_area_um2": sp, "seg_len_um": 1.0, "shaft_diam_um": 1.0}
         for a, b, d, sh, sp in seg])


def build_root(tmp, bad_cell=True):
    root = os.path.join(tmp, "h01")
    out = os.path.join(root, "out")
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(out)
    ids = [CELL_OK] + ([CELL_BAD] if bad_cell else [])
    for cid in ids:
        spines_frame().to_csv(os.path.join(out, "cell%d_spines.csv" % cid),
                              index=False, lineterminator="\n")
        phi_frame().to_csv(os.path.join(out, "neuron_%d_phi_mesh.csv" % cid),
                           index=False, lineterminator="\n")
    rows = [{"cell_id": CELL_OK, "deliverable_variant": "mesh_beyond",
             "F_lit_deliverable": F_AS_REPORTED, "qc_status": "pass",
             "coverage_count_deliverable": 5.0 / 6.0,
             "fallback_area_frac_deliverable": 2.0 / 111.5,
             "min_coverage": 0.99, "n_shaft_terminates": 1,
             "n_base_by_area_drop": 0, "n_shaft_not_reaching_box": 1,
             "n_measured_mesh_beyond": 5, "n_fallback_mesh_beyond": 1}]
    if bad_cell:
        r = dict(rows[0])
        r["cell_id"] = CELL_BAD
        r["F_lit_deliverable"] = 9.9999            # deliberately not reproducible
        rows.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(out, "spine_area_F_summary.csv"),
                              index=False, lineterminator="\n")
    return root, out


QSTAT_Q = """Queue: cpu
    queue_type = Execution
    max_queued = [u:PBS_GENERIC=100]
    max_run = [u:PBS_GENERIC=40]
    resources_max.walltime = 24:00:00
    comment = a very long comment value that PBS folds across the eighty col
\tumn boundary and continues here
    enabled = True
"""
QSTAT_B = """Server: dvlogin02
    server_state = Active
    max_array_size = 1000
    default_queue = cpu
"""
QSTAT_U = """dvlogin02:
                                                            Req'd  Req'd   Elap
Job ID          Username Queue    Jobname    SessID NDS TSK Memory Time  S Time
--------------- -------- -------- ---------- ------ --- --- ------ ----- - -----
1575001.dvlogin ldellame cpu      p1_L3exc     1234   1   2   16gb 02:00 R 00:10
1575002.dvlogin ldellame cpu      p1_L2inh     1235   1   2   16gb 02:00 Q   --
"""


def write(tmp, name, text):
    p = os.path.join(tmp, name)
    with open(p, "w") as fh:
        fh.write(text)
    return p


def main():
    tmp = tempfile.mkdtemp()
    try:
        # ================= Section A: the policy report =================
        root, out = build_root(tmp)
        sp, ph = spines_frame(), phi_frame()

        # A1-A2 the two primitives, in isolation
        area = P.attribute(sp, ph, "A_used_beyond_um2", np.ones(len(sp), dtype=bool))
        check("A1 attribute() sums per segment and drops the unmapped spine",
              list(np.round(area, 6)) == [5.0, 3.0, 3.0, 0.5],
              str(list(np.round(area, 6))))
        f = P.f_from_phi(ph, area)
        check("A2 f_from_phi uses only segments at or beyond 60 um",
              abs(f["F"] - F_AS_REPORTED) < 1e-12 and abs(f["A_shaft_um2"] - 9.0) < 1e-12
              and f["n_segments"] == 3,
              "F %.6f A_shaft %.1f" % (f["F"], f["A_shaft_um2"]))

        # A3 every policy, against the hand-computed value
        masks = P.keep_masks(sp)
        want = {"as_reported": F_AS_REPORTED, "drop_unmeasured": F_DROP_UNMEASURED,
                "drop_shaft_terminates": F_DROP_SHAFT_TERM,
                "drop_not_reaching_box": F_DROP_NOT_BOX}
        got = {k: P.f_from_phi(ph, P.attribute(sp, ph, "A_used_beyond_um2", masks[k]))["F"]
               for k in want}
        check("A3 every policy reproduces its hand-computed F",
              all(abs(got[k] - want[k]) < 1e-12 for k in want),
              ", ".join("%s %.5f" % (k, got[k]) for k in sorted(got)))
        check("A3a drop_unmeasured is exactly 1.5 on this fixture",
              abs(got["drop_unmeasured"] - 1.5) < 1e-12)

        # A4 the 100 um2 unmapped spine never reaches any policy
        check("A4 the unmapped spine (100 um2) leaks into no policy",
              all(v < 2.0 for v in got.values()), str(sorted(got.values())))

        # A5 the whole run
        rc = P.main(["--root", root])
        rep = os.path.join(out, "campaign_policy_report.csv")
        check("A5 report runs, writes the CSV, exits 1 because one cell fails "
              "to reproduce", rc == 1 and os.path.isfile(rep), "rc=%d" % rc)
        df = pd.read_csv(rep)
        r_ok = df[df["cell_id"] == CELL_OK].iloc[0]
        r_bad = df[df["cell_id"] == CELL_BAD].iloc[0]
        check("A6 the good cell reproduces the recorded F, by BOTH routes",
              bool(r_ok["reproduces"])
              and abs(r_ok["F_as_reported"] - F_AS_REPORTED) < 1e-9
              and abs(r_ok["F_from_phi_column"] - F_AS_REPORTED) < 1e-9)
        check("A6a the cell with a wrong recorded F is marked NOT reproduced",
              not bool(r_bad["reproduces"]) and "recorded" in str(r_bad["note"]))
        check("A7 the dF columns carry the policy shifts",
              abs(r_ok["dF_drop_unmeasured"] - (F_DROP_UNMEASURED - F_AS_REPORTED)) < 1e-9
              and abs(r_ok["dF_drop_shaft_terminates"] - (F_DROP_SHAFT_TERM - F_AS_REPORTED)) < 1e-9
              and abs(r_ok["dF_drop_not_reaching_box"] - (F_DROP_NOT_BOX - F_AS_REPORTED)) < 1e-9)
        check("A7a the summary columns are carried through for the filter",
              "fallback_area_frac_deliverable" in df.columns
              and "n_shaft_terminates" in df.columns and r_ok["n_shaft_terminates"] == 1)

        # A8 the population view excludes the unreproduced cell
        view = P.population_view(df, 0.01)
        check("A8 population_view counts 2 cells but reports on 1",
              view["n_cells"] == 2 and view["n_reproduced"] == 1)
        check("A8a both policies are MATERIAL at tol 0.01 on this fixture",
              view["n_material_drop_unmeasured"] == 1
              and view["n_material_drop_shaft_terminates"] == 1,
              "fill %d, shaft_term %d" % (view["n_material_drop_unmeasured"],
                                          view["n_material_drop_shaft_terminates"]))
        view_loose = P.population_view(df, 1.0)
        check("A8b at tol 1.0 neither is material -- the verdict follows tol, "
              "not the code",
              view_loose["n_material_drop_unmeasured"] == 0
              and view_loose["n_material_drop_shaft_terminates"] == 0)
        # ---- D-008: the reported figure is the fill fraction, not a threshold
        check("A8c the reported figure exists per cell: beyond-cutoff spine area "
              "2.0 of 6.5 came from the fill",
              abs(r_ok["A_spine_beyond_um2"] - 6.5) < 1e-9
              and abs(r_ok["A_spine_beyond_filled_um2"] - 2.0) < 1e-9
              and abs(r_ok["frac_area_filled_beyond"] - 2.0 / 6.5) < 1e-9,
              "%.4f of %.4f" % (r_ok["A_spine_beyond_filled_um2"],
                                r_ok["A_spine_beyond_um2"]))
        check("A8d shaft_terminates area is REPORTED but stays inside F (D-008): "
              "3.0 of 6.5, and F_as_reported is unchanged by it",
              abs(r_ok["A_spine_beyond_shaft_terminates_um2"] - 3.0) < 1e-9
              and abs(r_ok["frac_area_shaft_terminates_beyond"] - 3.0 / 6.5) < 1e-9
              and abs(r_ok["F_as_reported"] - F_AS_REPORTED) < 1e-9)
        check("A8e the pooled campaign fraction is on the view",
              abs(view["frac_area_filled_beyond_pooled"] - 2.0 / 6.5) < 1e-9
              and abs(view["frac_area_shaft_terminates_beyond_pooled"] - 3.0 / 6.5) < 1e-9,
              "%.5f" % view["frac_area_filled_beyond_pooled"])
        check("A8f no threshold is derived any more -- D-008 sets none",
              "fallback_frac_min_unsafe" not in view
              and "fallback_frac_max_safe" not in view)

        # A8g pooled must be a RATIO OF SUMS, not a mean of ratios: one large
        # well-covered cell and one tiny poorly-covered one have mean 0.30 and
        # pooled 1.5/11 = 0.13636..., which tells them apart.
        two = pd.DataFrame([
            {"cell_id": 1, "reproduces": True, "F_as_reported": 1.5,
             "A_spine_beyond_um2": 10.0, "A_spine_beyond_filled_um2": 1.0,
             "frac_area_filled_beyond": 0.1,
             "A_spine_beyond_shaft_terminates_um2": 0.0,
             "dF_drop_unmeasured": 0.0, "dF_drop_shaft_terminates": 0.0,
             "dF_drop_not_reaching_box": 0.0},
            {"cell_id": 2, "reproduces": True, "F_as_reported": 1.6,
             "A_spine_beyond_um2": 1.0, "A_spine_beyond_filled_um2": 0.5,
             "frac_area_filled_beyond": 0.5,
             "A_spine_beyond_shaft_terminates_um2": 0.0,
             "dF_drop_unmeasured": 0.0, "dF_drop_shaft_terminates": 0.0,
             "dF_drop_not_reaching_box": 0.0}])
        v2 = P.population_view(two, 0.01)
        check("A8g the campaign fraction is a ratio of sums (0.1364), not a mean "
              "of per-cell ratios (0.30)",
              abs(v2["frac_area_filled_beyond_pooled"] - 1.5 / 11.0) < 1e-12
              and abs(v2["frac_area_filled_beyond_pooled"] - 0.3) > 0.1,
              "%.5f" % v2["frac_area_filled_beyond_pooled"])
        check("A8h the per-cell median and worst cell are reported alongside it",
              abs(v2["frac_area_filled_beyond_median"] - 0.3) < 1e-12
              and v2["frac_area_filled_beyond_worst_cell"] == 2)

        # A9 a clean population exits 0
        root2, out2 = build_root(tmp, bad_cell=False)
        check("A9 every cell reproducing -> exit 0", P.main(["--root", root2]) == 0)

        # A10 refusals
        try:
            P.main(["--root", os.path.join(tmp, "nope")])
            check("A10 missing root refused", False, "accepted")
        except SystemExit as ex:
            check("A10 missing root refused", "P2/P3 have not run" in str(ex)
                  or "no " in str(ex), str(ex)[:60])
        root3 = os.path.join(tmp, "empty")
        os.makedirs(os.path.join(root3, "out"), exist_ok=True)
        try:
            P.main(["--root", root3])
            check("A10a root with an out/ but no P3 summary refused", False)
        except SystemExit as ex:
            check("A10a root with an out/ but no P3 summary refused",
                  "no P3 summary" in str(ex), str(ex)[:60])

        # ================= Section B: the queue check ==================
        fq = write(tmp, "qf.txt", QSTAT_Q)
        fb = write(tmp, "bf.txt", QSTAT_B)
        fu = write(tmp, "qu.txt", QSTAT_U)

        check("B1 unwrap rejoins PBS newline+TAB folding",
              "eighty column boundary" in Q.unwrap(QSTAT_Q))
        attrs = Q.parse_attrs(QSTAT_Q)
        check("B1a the folded attribute is parsed whole, not truncated",
              "continues here" in attrs.get("comment", ""), attrs.get("comment", "")[:40])
        check("B2 as_int reads the [u:PBS_GENERIC=N] spelling",
              Q.as_int("[u:PBS_GENERIC=100]") == 100 and Q.as_int("40") == 40
              and Q.as_int(None) is None and Q.as_int("none") is None)
        check("B3 count_user_jobs counts jobs, not the header or rule rows",
              Q.count_user_jobs(QSTAT_U) == 2, str(Q.count_user_jobs(QSTAT_U)))
        lim = Q.limits_from(Q.parse_attrs(QSTAT_Q), Q.parse_attrs(QSTAT_B))
        check("B4 the three limit kinds are found, from the right sources",
              lim["array_size"] == ("max_array_size", 1000, "server")
              and lim["queued"] == ("max_queued", 100, "queue")
              and lim["run"] == ("max_run", 40, "queue"), str(lim))

        v = Q.verdict(537, 1, lim, 2)
        check("B5 537 cells x 1 shard breaches max_queued (100) and max_run (40)",
              not v["ok"] and v["subjobs"] == 537
              and {b[1] for b in v["breaches"]} == {"max_queued", "max_run"},
              str([b[1] for b in v["breaches"]]))
        check("B5a the advice is a submission size, not a complaint",
              v["max_cells_per_submission"] == 38,
              str(v.get("max_cells_per_submission")))
        v2 = Q.verdict(20, 1, lim, 2)
        check("B6 20 cells x 1 shard fits every limit", v2["ok"] and v2["subjobs"] == 20)
        v3 = Q.verdict(20, 4, lim, 2)
        check("B6a the same 20 cells at SHARDS=4 do NOT fit (80 + 2 > 40 running)",
              not v3["ok"] and v3["subjobs"] == 80)

        # B7 end to end through main(), with captured qstat text
        rc = Q.main(["--rows", "20", "--shards", "1", "--from-queue", fq,
                     "--from-server", fb, "--from-user", fu])
        check("B7 main() exits 0 when the array fits", rc == 0, "rc=%d" % rc)
        rc = Q.main(["--rows", "537", "--shards", "1", "--from-queue", fq,
                     "--from-server", fb, "--from-user", fu])
        check("B7a main() exits 1 when it does not", rc == 1, "rc=%d" % rc)
        empty = write(tmp, "empty.txt", "Queue: cpu\n    enabled = True\n")
        rc = Q.main(["--rows", "537", "--from-queue", empty, "--from-server", empty,
                     "--from-user", fu])
        check("B7b no limit readable -> exit 2, UNKNOWN, never a silent yes",
              rc == 2, "rc=%d" % rc)

        # B8 the manifest row count matches campaign.pbs's own awk rule
        man = write(tmp, "m.csv",
                    "cell_id,layer\r\n111,L3\r\n\r\n222,L3\r\n333,L3\r\n")
        check("B8 manifest_rows skips the header and blank lines, CRLF included",
              Q.manifest_rows(man) == 3, str(Q.manifest_rows(man)))
        rc = Q.main(["--manifest", man, "--from-queue", fq, "--from-server", fb,
                     "--from-user", fu])
        check("B8a a manifest drives the check the same way --rows does", rc == 0)

        # ---- B10-B12 the CPU budget (D-008), which cannot refuse a job
        check("B10 the allocation defaults are the decided ones (200 CPUs, 300 h)",
              Q.ALLOC_MAX_CPUS == 200 and abs(Q.ALLOC_MAX_WALLTIME_H - 300.0) < 1e-9,
              "%s cpus, %s h" % (Q.ALLOC_MAX_CPUS, Q.ALLOC_MAX_WALLTIME_H))
        pbs = write(tmp, "fake.pbs",
                    "#!/bin/bash\n#PBS -N x\n#PBS -l select=1:ncpus=2:mem=16gb\n"
                    "#PBS -l walltime=12:00:00\nset -e\necho ncpus=99\n")
        spec = Q.parse_pbs_script(pbs)
        check("B10a ncpus, mem and walltime are READ from the job script, and "
              "a non-directive line is ignored",
              spec["ncpus"] == 2 and spec["mem"] == "16gb"
              and abs(spec["walltime_h"] - 12.0) < 1e-9, str(spec))
        check("B10b a missing script yields no spec rather than a crash",
              Q.parse_pbs_script(os.path.join(tmp, "nope.pbs")) == {})
        b = Q.budget_view(537, 2, 200, 12.0, 300.0, hours_per_task=3.5)
        check("B11 200 CPUs at 2 per subjob -> 100 at once; 537 subjobs is 6 waves",
              b["concurrency"] == 100 and b["waves"] == 6,
              "conc %d waves %d" % (b["concurrency"], b["waves"]))
        check("B11a the wall-clock and CPU-hour estimates follow from the waves",
              abs(b["elapsed_h_estimate"] - 21.0) < 1e-9
              and abs(b["cpu_hours_estimate"] - 537 * 3.5 * 2) < 1e-6
              and b["walltime_ok"] and b["fits_task_walltime"],
              "%.1f h, %.0f cpu-h" % (b["elapsed_h_estimate"], b["cpu_hours_estimate"]))
        b2 = Q.budget_view(537, 2, 200, 12.0, 300.0, hours_per_task=20.0)
        check("B11b a per-cell time above the script's request is flagged, not "
              "silently accepted", not b2["fits_task_walltime"])
        b3 = Q.budget_view(10, 2, 200, 400.0, 300.0)
        check("B11c a walltime request above the 300 h ceiling is flagged",
              not b3["walltime_ok"])
        b4 = Q.budget_view(537, 1, 200, 12.0, 300.0)
        check("B12 at 1 CPU per subjob the same allocation runs 200 at once "
              "(3 waves), so ncpus is what sets throughput",
              b4["concurrency"] == 200 and b4["waves"] == 3)
        rc = Q.main(["--rows", "537", "--from-queue", fq, "--from-server", fb,
                     "--from-user", fu, "--hours-per-task", "3.5"])
        check("B12a main() prints the budget alongside the limits and keeps its "
              "limit-based exit status", rc == 1, "rc=%d" % rc)

        # B9 the scripts are syntactically clean and byte-safe for the cluster
        for name in ("campaign_policy_report.py", "campaign_queue_check.py"):
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
            r = subprocess.run([sys.executable, "-m", "py_compile", path],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               universal_newlines=True)
            raw = open(path, "rb").read()
            check("B9 %s compiles, LF only, pure ASCII" % name,
                  r.returncode == 0 and b"\r" not in raw and all(b < 128 for b in raw),
                  r.stdout.strip()[:70])
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
