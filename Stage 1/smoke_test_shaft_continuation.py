"""Smoke test for shaft_continuation. Run: python3 smoke_test_shaft_continuation.py

Offline; no labeller needed -- the labeller's output is simulated so the
corrector is tested in isolation against constructed answers. Cases marked
BROKEN or AMBIGUOUS must NOT be silently corrected.

Exit 0 and a final 'ALL GREEN' mean pass.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shaft_continuation as sc  # noqa: E402

CHECKS = [0, 0]


def check(cond, msg):
    CHECKS[0 if cond else 1] += 1
    print(("  ok    " if cond else "  FAIL  ") + msg)


def must_raise(fn, exc, label):
    try:
        fn()
    except exc as e:
        check(True, "%s: raised (%s)" % (label, str(e)[:60]))
        return
    except Exception as e:  # noqa: BLE001
        check(False, "%s: wrong exception %s" % (label, type(e).__name__))
        return
    check(False, "%s: did NOT raise" % label)


def build(shaft_n=12, stub_n=4, spine_at=5, spine_n=3, step=300.0,
          r_shaft=400.0, r_stub=400.0, r_neck=120.0, spacing=300.0):
    """Shaft along +x with a spine at `spine_at` (+y) and a terminal stub.

    The stub continues along +x past the last branch point; a real spine
    leaves along +y. Returns (df_before, df_after, expected_stub_ids,
    expected_spine_ids).
    """
    rows, nid = [], 0
    for i in range(shaft_n):
        rows.append({"id": i, "p": i - 1 if i > 0 else -1,
                     "x": 1.0e5 + i * spacing, "y": 2.0e5, "z": 3.0e5,
                     "r": r_shaft, "annotated_type": "dendrite"})
    rows[0]["annotated_type"] = "soma"
    nid = shaft_n

    spine_ids = []
    prev = spine_at
    for k in range(spine_n):
        rows.append({"id": nid, "p": prev, "x": 1.0e5 + spine_at * spacing,
                     "y": 2.0e5 + (k + 1) * step, "z": 3.0e5,
                     "r": r_neck, "annotated_type": "dendrite"})
        spine_ids.append(nid); prev = nid; nid += 1

    # branch at the last shaft node: a long daughter (-y) plus the +x stub
    bp = shaft_n - 1
    stub_ids, prev = [], bp
    for k in range(stub_n):
        rows.append({"id": nid, "p": prev, "x": 1.0e5 + (shaft_n + k) * spacing,
                     "y": 2.0e5, "z": 3.0e5, "r": r_stub,
                     "annotated_type": "dendrite"})
        stub_ids.append(nid); prev = nid; nid += 1

    prev = bp
    for k in range(25):                       # long daughter, never a spine
        rows.append({"id": nid, "p": prev, "x": 1.0e5 + bp * spacing,
                     "y": 2.0e5 - (k + 1) * step, "z": 3.0e5, "r": 350.0,
                     "annotated_type": "dendrite"})
        prev = nid; nid += 1

    before = pd.DataFrame(rows)
    after = before.copy()
    # simulate the labeller: both short subtrees relabelled head/neck
    after.loc[after["id"].isin(spine_ids), "annotated_type"] = "neck"
    after.loc[after["id"].isin(spine_ids[-1:]), "annotated_type"] = "head"
    after.loc[after["id"].isin(stub_ids), "annotated_type"] = "neck"
    after.loc[after["id"].isin(stub_ids[-1:]), "annotated_type"] = "head"
    return before, after, set(stub_ids), set(spine_ids)


def main():
    # ---- 1. calibre route: stub is shaft-calibre, spine neck is thin -------
    before, after, stub, spine = build()
    rt = sc.radius_trustworthy(before)
    check(rt["usable"], "T1 radii usable on this cell (%.2f real, %d distinct)"
          % (rt["fraction_real"], rt["n_distinct"]))

    fixed, rep = sc.demote_shaft_continuations(before, after)
    got_spine = set(fixed.loc[fixed["annotated_type"].str.lower()
                              .isin(sc.DEFAULT_SPINE_LABELS), "id"])
    check(rep["method"] == "calibre+collinearity", "T1 both observables used when radii are real")
    check(rep["n_components_demoted"] == 1 and rep["n_nodes_demoted"] == len(stub),
          "T1 exactly the stub demoted (%d comp, %d nodes)"
          % (rep["n_components_demoted"], rep["n_nodes_demoted"]))
    check(got_spine == spine, "T1 the real spine survives untouched")
    check(all(fixed.loc[fixed["id"].isin(stub), "annotated_type"] == "dendrite"),
          "T1 stub restored to its PRE-LABEL annotated_type, not a guess")
    check(rep["demoted"][0]["restored_to"] == "dendrite", "T1 restored_to recorded")
    check(abs(rep["demoted"][0]["rho"] - 1.0) < 1e-9 and abs(rep["demoted"][0]["cos"] - 1.0) < 1e-9,
          "T1 stub scores rho=1.0 cos=1.0; the real spine root scores rho=0.30 cos=0.00")
    check(abs(rep["demoted"][0]["length_nm"] - 4 * 300.0) < 1e-6,
          "T1 demoted length %.0f nm reported" % rep["demoted"][0]["length_nm"])

    # ---- 2. collinearity route: flat radii, geometry must carry it ---------
    before2, after2, stub2, spine2 = build(r_shaft=50.0, r_stub=50.0, r_neck=50.0)
    rt2 = sc.radius_trustworthy(before2)
    check(not rt2["usable"], "T2 flat fallback radii correctly judged unusable")
    fixed2, rep2 = sc.demote_shaft_continuations(before2, after2)
    check(rep2["method"] == "collinearity only", "T2 falls back to collinearity alone")
    check(rep2["n_nodes_demoted"] == len(stub2),
          "T2 stub still found by geometry alone (%d nodes)" % rep2["n_nodes_demoted"])
    got2 = set(fixed2.loc[fixed2["annotated_type"].str.lower()
                          .isin(sc.DEFAULT_SPINE_LABELS), "id"])
    check(got2 == spine2, "T2 real spine survives on the collinearity route")

    # ---- 3. a mid-shaft spine must never be demoted ------------------------
    # The continuation at the spine's branch point is the long shaft, which was
    # never labelled, so nothing there is demotable.
    tbl = sc.score_spine_roots(after)[0]
    sp_row = tbl.loc[tbl["root"] == 12].iloc[0]
    check(not sp_row["is_shaft"] and not sp_row["rho_ok"],
          "T3 the mid-shaft spine root is NOT shaft-like (rho %.2f, cos %.2f)"
          % (sp_row["rho"], sp_row["cos"]))
    st_row = tbl.loc[tbl["root"] == 15].iloc[0]
    check(st_row["is_shaft"] and st_row["rho_ok"] and st_row["cos_ok"],
          "T3 the stub root passes BOTH tests")

    # ---- 4. AMBIGUOUS: two identical short branches, nothing demoted -------
    amb = before.copy()
    amb.loc[amb["id"].isin(stub), "r"] = 400.0
    # make the long daughter short and identical in calibre and angle mirror
    amb_after = after.copy()
    tie = amb.copy()
    tie_after = amb_after.copy()
    # duplicate the stub geometry mirrored in y so the two children tie exactly
    bp = 11
    extra = []
    nid = int(tie["id"].max()) + 1
    prev = bp
    for k in range(4):
        extra.append({"id": nid, "p": prev, "x": 1.0e5 + (12 + k) * 300.0,
                      "y": 2.0e5, "z": 3.0e5, "r": 400.0,
                      "annotated_type": "dendrite"})
        prev = nid; nid += 1
    tie = pd.concat([tie, pd.DataFrame(extra)], ignore_index=True)
    tie_after = pd.concat([tie_after, pd.DataFrame(extra)], ignore_index=True)
    tie_after.loc[tie_after["id"] >= extra[0]["id"], "annotated_type"] = "neck"
    _, rep_tie = sc.demote_shaft_continuations(tie, tie_after)
    check(11 in rep_tie["ambiguous_branch_points"],
          "T4 AMBIGUOUS exact tie flagged, not resolved arbitrarily")
    check(all(d["bp"] != 11 for d in rep_tie["demoted"]),
          "T4 nothing demoted at the ambiguous branch point")

    # ---- 5. margins are the control ---------------------------------------
    _, rep_strict = sc.demote_shaft_continuations(before, after, rho_shaft_min=1.5)
    check(rep_strict["n_components_demoted"] == 0,
          "T5 an unreachable calibre threshold demotes nothing (fails safe)")
    _, rep_cos = sc.demote_shaft_continuations(before, after, cos_shaft_min=1.5)
    check(rep_cos["n_components_demoted"] == 0,
          "T5 an unreachable collinearity threshold demotes nothing either")
    _, rep_loose = sc.demote_shaft_continuations(before, after, rho_shaft_min=0.0,
                                                 cos_shaft_min=-1.0)
    check(rep_loose["n_components_demoted"] >= 1,
          "T5 permissive thresholds demote (both tests are live controls)")

    # ---- 6. idempotence ----------------------------------------------------
    twice, rep_twice = sc.demote_shaft_continuations(before, fixed)
    check(rep_twice["n_components_demoted"] == 0,
          "T6 re-running on a corrected frame changes nothing")
    check(twice["annotated_type"].equals(fixed["annotated_type"]),
          "T6 idempotent on labels")

    # ---- 7. BROKEN inputs --------------------------------------------------
    must_raise(lambda: sc.demote_shaft_continuations(before, after.iloc[:-1]),
               ValueError, "T7 BROKEN row-count mismatch")
    shuffled = after.sample(frac=1.0, random_state=0).reset_index(drop=True)
    must_raise(lambda: sc.demote_shaft_continuations(before, shuffled),
               ValueError, "T7 BROKEN row order differs")

    # ---- 8. no spines at all -> no-op --------------------------------------
    _, rep_none = sc.demote_shaft_continuations(before, before)
    check(rep_none["n_components_demoted"] == 0 and rep_none["n_nodes_demoted"] == 0,
          "T8 an unlabelled frame is a no-op")

    # ---- 9. vocabulary sourced from spine_density when supplied ------------
    class _SD:
        SHAFT_REGEX = r"dendrite|apical|^1$"
        SPINE_LABELS = ("spine", "head", "neck")
        DEFAULT_RADIUS_NM = 50.0
        MODULE_VERSION = "spine_density-1.3.0"

    _, rep_voc = sc.demote_shaft_continuations(before, after, spine_density=_SD)
    check(rep_voc["vocabulary"]["spine_density_version"] == "spine_density-1.3.0",
          "T9 vocabulary and version taken from spine_density")

    # ---- 10. the wrapper requires an explicit threshold --------------------
    must_raise(lambda: sc.label_dendritic_spines_corrected([1], input_dir="/tmp"),
               TypeError, "T10 BROKEN wrapper refuses the implicit 5000 nm default")

    # ---- 11. the sibling-continuation observable ---------------------------
    # In `build` the stub at bp 11 has a long (25-node) sibling, so the parent
    # process continues there: category short_terminal_branch, not shaft_ending.
    tbl11 = sc.score_spine_roots(after)[0]
    st = tbl11.loc[tbl11["root"] == 15].iloc[0]
    check(st["sibling_continues"] and st["category"] == "short_terminal_branch",
          "T11 stub with a long sibling is classed short_terminal_branch")
    sp = tbl11.loc[tbl11["root"] == 12].iloc[0]
    check(sp["sibling_continues"], "T11 the mid-shaft spine also has a continuing sibling")

    _, rep_open = sc.demote_shaft_continuations(before, after)
    check(rep_open["demoted_by_category"]["short_terminal_branch"] == 1
          and rep_open["demoted_by_category"]["shaft_ending"] == 0,
          "T11 default mode demotes it and says which category")

    # ---- 12. TERMINAL ZONE: shaft ends just past a spine -------------------
    # Spine and stub share the SAME branch point (11) and the long daughter is
    # removed, so bp 11's children are the 3-node spine and the 4-node stub --
    # both labelled, nothing unlabelled. A shaft ending is present among them.
    tb0, ta0, stub2b, spine2b = build(spine_at=11)
    keep = ~tb0["id"].isin(range(19, 60))          # drop the long daughter
    tb, ta = tb0[keep].reset_index(drop=True), ta0[keep].reset_index(drop=True)
    tt = sc.score_spine_roots(ta)[0]
    st2 = tt.loc[tt["root"] == 15].iloc[0]
    check(int(st2["n_unlabelled_siblings"]) == 0 and st2["all_children_labelled"],
          "T12 terminal zone: every child of bp 11 is labelled")
    check(st2["category"] == "shaft_ending",
          "T12 the stub is classed shaft_ending, not short_terminal_branch")
    fixed_t, rep_t = sc.demote_shaft_continuations(tb, ta)
    check(rep_t["demoted_by_category"]["shaft_ending"] == 1,
          "T12 the stub is demoted and reported under shaft_ending")
    check(set(fixed_t.loc[fixed_t["annotated_type"].str.lower()
                          .isin(sc.DEFAULT_SPINE_LABELS), "id"]) == spine2b,
          "T12 the tip spine survives; only the shaft ending is demoted")

    # ---- 13. out-degree at bp is ALWAYS >= 2 in labeller output ------------
    # The labeller iterates branch_points = [n for n in G.nodes()
    # if G.out_degree(n) > 1], so a one-child parent is never examined and its
    # child is never labelled. An out_degree(bp) == 1 case therefore cannot
    # exist in labeller output, and a 1-vs-2 test can never fire.
    for frame in (after, ta):
        t = sc.score_spine_roots(frame)[0]
        if len(t):
            check(int(t["n_children_of_bp"].min()) >= 2,
                  "T13 every labelled root sits at a bp with >= 2 children "
                  "(min %d)" % int(t["n_children_of_bp"].min()))

    # The unlabelled-sibling count is what actually discriminates.
    t_term = sc.score_spine_roots(ta)[0]
    check(int(t_term.loc[t_term["root"] == 15, "n_unlabelled_siblings"].iloc[0]) == 0,
          "T13 terminal zone: the stub has ZERO unlabelled siblings")
    t_mid = sc.score_spine_roots(after)[0]
    check(int(t_mid.loc[t_mid["root"] == 12, "n_unlabelled_siblings"].iloc[0]) >= 1,
          "T13 mid-shaft spine: the shaft continues as an unlabelled sibling")
    check(int(t_mid["category_disagreement"].sum()) == 0,
          "T13 the labeller-native signal agrees with the length-based one")

    # ---- 14. the topology fields are DESCRIPTIVE, never decisive -----------
    # T11 (short_terminal_branch) and T12 (shaft_ending) sit in opposite
    # topological categories, and both are demoted. If category ever gated the
    # decision, one of them would survive -- this is the regression test.
    check(rep_open["demoted_by_category"]["short_terminal_branch"] == 1
          and rep_t["demoted_by_category"]["shaft_ending"] == 1,
          "T14 both topological categories are demoted: topology labels, "
          "it does not decide")
    check("require_terminal_bp" not in
          str(__import__("inspect").signature(sc.score_spine_roots)),
          "T14 no topological gate remains in the decision path")

    print("\n%d checks passed, %d failed" % tuple(CHECKS))
    print("ALL GREEN" if CHECKS[1] == 0 else "FAILURES PRESENT")
    return 0 if CHECKS[1] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
