#!/usr/bin/env python3
"""Smoke test for P3 assembly (h01_spine_area_F v1.6) against the REAL
spine_density 1.3.0 -- no network, no mesh: the per-spine mesh areas are
synthetic records, so the test isolates assemble_cell's bookkeeping from P2.

What it establishes, on a synthetic cell with four spines:

  CAP     phi_skel is built with the tip cap (spine_density 1.3.0). The shaft
          side of the cap survives in the denominator of every mesh variant;
          the spine side is discarded when the mesh replaces the spine column.
          The attribution gate compares against the UNCAPPED spine column, so
          the gate passes with the cap on.
  MESH    with every spine measured, F_mesh_beyond == 1 + sum A_beyond /
          sum A_shaft(capped) exactly, and the deliverable is fully covered.
  FALL    one failed + one clipped spine -> both take kappa_hat * A_skel, the
          summary counts them, and qc is pass_low_confidence.
  EMPTY   no base measurement at all -> F_mesh_beyond is NaN and qc is fail,
          never a silent copy of the skeleton value.
  GUARD   a spine_density that cannot cap raises instead of returning an
          uncapped table.

    cd h01_code && python3 smoke_test_p3_assemble.py

Expected: "14 checks passed, 0 failed" and "ALL GREEN" on the last two lines.
"""
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
_stage1 = os.path.join(HERE, "stage1")
if os.path.isfile(os.path.join(_stage1, "spine_density.py")):
    _paths = [HERE, _stage1]
else:
    _paths = [HERE, os.path.join(REPO, "towards_eeg", "structure"),
              os.path.join(REPO, "Stage 1")]
for _p in _paths:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                    # noqa: E402
import pandas as pd                                   # noqa: E402
import spine_density as sd                            # noqa: E402
import sma_run as sr                                  # noqa: E402
import h01_spine_area_F as SAF                        # noqa: E402

FAILURES = []
CELL = 4242


def check(name, ok, detail=""):
    print("  [%s] %-60s %s" % ("PASS" if ok else "FAIL", name, detail))
    if not ok:
        FAILURES.append(name)
    return ok


def synthetic_cell(n_spines=4):
    """Shaft along +x (21 nodes, r=300 nm) with n_spines identical spines
    (thin neck, bulged head) on interior nodes. Returns nodes, labelled, comp."""
    rows = []
    n_shaft, step, r_shaft = 21, 250.0, 300.0
    for i in range(n_shaft):
        rows.append(dict(id=i, p=i - 1 if i else -1, x=step * i, y=0.0, z=0.0,
                         r=r_shaft, annotated_type="dendrite"))
    nid = 1000
    for k in range(n_spines):
        base = 4 + 4 * k
        par = base
        for j, r in enumerate([70, 70, 70, 70, 250, 250, 250], start=1):
            rows.append(dict(id=nid, p=par, x=step * base, y=step * j, z=0.0,
                             r=float(r), annotated_type="spine"))
            par = nid
            nid += 1
    nodes = pd.DataFrame(rows)
    labelled = nodes.copy()
    nodes["spine_label"] = labelled["annotated_type"].str.lower().to_numpy()
    comp = sr.spine_components(nodes, sr.spine_mask(nodes, spine_values=sd.SPINE_LABELS))
    return nodes, labelled, comp


def synthetic_recs(sk, kappa=1.30, k_raw=1.25, k_nr=1.27, k_by=1.20):
    """One 'measured' record per spine: mesh areas as fixed multiples of the
    skeleton area, so every expected F is computable in closed form."""
    recs = {}
    for r in sk.itertuples(index=False):
        sid = int(r.sigma_id)
        a = float(r.A_skel_um2)
        recs[sid] = {"sigma_id": sid, "ok": True, "clipped": False,
                     "A_mesh_um2": kappa * a, "A_mesh_raw_um2": k_raw * a,
                     "A_mesh_norind_um2": k_nr * a, "A_beyond_um2": k_by * a,
                     "A_rind_um2": (kappa - k_nr) * a, "rind_tol_nm": 8.0,
                     "frac_rind": (kappa - k_nr) / kappa,
                     "frac_cut": 0.05, "frac_bridge": 0.02, "frac_box": 0.0,
                     "frac_unresolved": 0.0, "s_base_nm": 320.0,
                     "s_base_minus_r_shaft_nm": 20.0}
    return recs


def main():
    print("smoke_test_p3_assemble  (%s / %s)" % (SAF.MODULE_VERSION, sd.MODULE_VERSION))
    check("real spine_density 1.3.0 with a cap default",
          "stub" not in sd.MODULE_VERSION and hasattr(sd, "CAP_H_UM_DEFAULT"),
          "%s CAP_H_UM_DEFAULT=%s" % (sd.MODULE_VERSION, getattr(sd, "CAP_H_UM_DEFAULT", None)))

    nodes, labelled, comp = synthetic_cell()
    sk = SAF.skeleton_spine_table(sd, labelled, nodes, comp)
    recs = synthetic_recs(sk)
    n = len(sk)

    # ---- CAP -------------------------------------------------------------
    out = SAF.assemble_cell(sd, labelled, nodes, comp, recs, CELL, sk=sk,
                            cap_tips=True, min_per_bin=1)
    s = out["summary"]
    ph = out["phi"]
    check("gate passes with the cap on (compared against the uncapped column)",
          s["attribution_gate"]["pass"],
          "max|diff| %.2e" % s["attribution_gate"]["max_abs_diff_um2"])
    shaft_cap = float(ph["skel"]["shaft_cap_um2"].sum())
    spine_cap = float(ph["skel"]["spine_cap_um2"].sum())
    check("both caps are non-zero on the real module",
          shaft_cap > 0 and spine_cap > 0,
          "shaft_cap %.4f spine_cap %.4f um2" % (shaft_cap, spine_cap))
    sh_cap = float(ph["skel"]["shaft_area_um2"].sum())
    sh_nocap = float(ph["skel_nocap"]["shaft_area_um2"].sum())
    check("skel vs skel_nocap differ by exactly the caps",
          abs(sh_cap - sh_nocap - shaft_cap) < 1e-9
          and abs(float(ph["skel"]["spine_area_um2"].sum())
                  - float(ph["skel_nocap"]["spine_area_um2"].sum()) - spine_cap) < 1e-9)
    check("every mesh variant keeps the CAPPED shaft area",
          all(abs(float(ph[v]["shaft_area_um2"].sum()) - sh_cap) < 1e-9
              for v in SAF.MESH_VARIANTS))

    # ---- MESH ------------------------------------------------------------
    A_by = sum(r["A_beyond_um2"] for r in recs.values())
    F_expect = 1.0 + A_by / sh_cap
    check("F_whole_mesh_beyond == 1 + sum A_beyond / A_shaft(capped)",
          abs(s["F_whole_mesh_beyond"] - F_expect) < 1e-9,
          "%.6f vs %.6f" % (s["F_whole_mesh_beyond"], F_expect))
    check("spine cap is discarded on the mesh side",
          abs(float(ph["mesh_beyond"]["spine_area_um2"].sum()) - A_by) < 1e-9)
    check("deliverable fully covered, qc pass, zero fallback",
          s["deliverable_variant"] == "mesh_beyond" and s["qc_status"] == "pass"
          and s["coverage_count_deliverable"] == 1.0
          and s["n_fallback_mesh_beyond"] == 0
          and abs(s["F_lit_deliverable"] - s["F_lit_mesh_beyond"]) < 1e-12
          or (np.isnan(s["F_lit_deliverable"]) and np.isnan(s["F_lit_mesh_beyond"])),
          "qc=%s cov=%.3f" % (s["qc_status"], s["coverage_count_deliverable"]))

    # ---- FALL ------------------------------------------------------------
    sids = sorted(recs)
    recs2 = {k: dict(v) for k, v in recs.items()}
    recs2[sids[0]]["ok"] = False
    recs2[sids[-1]]["clipped"] = True
    out2 = SAF.assemble_cell(sd, labelled, nodes, comp, recs2, CELL, sk=sk,
                             cap_tips=True, min_per_bin=1)
    s2 = out2["summary"]
    sp2 = out2["spines"]
    meas = sp2["measured_beyond"].to_numpy(bool)
    check("one failed + one clipped -> two on kappa fallback, none raw",
          s2["n_measured_mesh_beyond"] == n - 2 and s2["n_fallback_mesh_beyond"] == 2
          and s2["n_fallback_kappa_mesh_beyond"] == 2 and s2["n_fallback_raw_mesh_beyond"] == 0,
          "meas=%s fb=%s kappa=%s raw=%s" % (s2["n_measured_mesh_beyond"],
                                            s2["n_fallback_mesh_beyond"],
                                            s2["n_fallback_kappa_mesh_beyond"],
                                            s2["n_fallback_raw_mesh_beyond"]))
    # kappa_hat on the beyond track is the ratio of sums over the measured
    # spines (= 1.20 here), so the fallback reproduces the same F as full coverage
    k_by = float(sp2.loc[meas, "A_beyond_um2"].sum() / sp2.loc[meas, "A_skel_um2"].sum())
    A_used = float(sp2["A_used_beyond_um2"].sum())
    A_expect = float(sp2.loc[meas, "A_beyond_um2"].sum()
                     + k_by * sp2.loc[~meas, "A_skel_um2"].sum())
    check("fallback uses kappa_hat(A_skel) * A_skel, kappa_hat from the measured spines",
          abs(A_used - A_expect) < 1e-9 and abs(k_by - 1.20) < 1e-9,
          "kappa_hat %.4f" % k_by)
    check("qc pass_low_confidence with a reason naming the fallback",
          s2["qc_status"] == "pass_low_confidence" and "fallback" in s2["qc_reason"]
          and abs(s2["coverage_count_deliverable"] - (n - 2) / n) < 1e-12
          and 0.0 < s2["fallback_area_frac_deliverable"] < 1.0,
          "%.3f of spine area on fallback" % s2["fallback_area_frac_deliverable"])

    # ---- EMPTY -----------------------------------------------------------
    recs3 = {k: {kk: vv for kk, vv in v.items()
                 if kk not in ("A_beyond_um2", "s_base_nm", "s_base_minus_r_shaft_nm")}
             for k, v in recs.items()}
    s3 = SAF.assemble_cell(sd, labelled, nodes, comp, recs3, CELL, sk=sk,
                           cap_tips=True, min_per_bin=1)["summary"]
    check("no base measurement -> F_mesh_beyond NaN, qc fail, mesh track intact",
          s3["qc_status"] == "fail" and np.isnan(s3["F_whole_mesh_beyond"])
          and np.isnan(s3["F_lit_deliverable"]) and np.isfinite(s3["F_whole_mesh"]),
          s3["qc_reason"][:60])

    # ---- cap off ---------------------------------------------------------
    s4 = SAF.assemble_cell(sd, labelled, nodes, comp, recs, CELL, sk=sk,
                           cap_tips=False, min_per_bin=1)["summary"]
    check("cap_tips=False: no cap columns, skel == skel_nocap",
          s4["shaft_cap_total_um2"] == 0.0 and s4["cap_tips"] is False
          and abs(s4["F_whole_skel"] - s4["F_whole_skel_nocap"]) < 1e-12)

    # ---- GUARD -----------------------------------------------------------
    fake = types.SimpleNamespace(**{k: getattr(sd, k) for k in dir(sd)
                                    if not k.startswith("__") and k != "CAP_H_UM_DEFAULT"})
    fake.MODULE_VERSION = "spine_density-1.2.0 (guard test)"
    try:
        SAF.assemble_cell(fake, labelled, nodes, comp, recs, CELL, sk=sk, cap_tips=True)
        check("spine_density without a cap default is REFUSED", False, "accepted")
    except SAF.SpineAreaError as exc:
        check("spine_density without a cap default is REFUSED",
              "CAP_H_UM_DEFAULT" in str(exc), str(exc)[:60])

    n_checks = 14
    print("\n%d checks passed, %d failed" % (n_checks - len(FAILURES), len(FAILURES)))
    print("ALL GREEN" if not FAILURES else "FAILURES: %s" % FAILURES)
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    sys.exit(main())
