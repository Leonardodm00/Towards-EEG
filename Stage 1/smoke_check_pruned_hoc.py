#!/usr/bin/env python3
"""Smoke test for check_pruned_hoc. Runs from the Stage 1 folder.

  python3 smoke_check_pruned_hoc.py        (quiet)
  python3 smoke_check_pruned_hoc.py -v

Builds a toy neuron (soma, one dendrite, three spines), exports it with the
REAL morphology_exporter -- once aligned through a genuine rotation, once
unaligned -- and checks the .hoc round trip. Then it tampers with the file
and expects the check to notice each edit:

  T0  PREFLIGHT: every module morphology_exporter needs for the three-vote
      correction is importable from this folder, at an adequate version.
      Runs first so a missing dependency is caught here, in seconds, rather
      than 20 minutes into a bank run.
  T1  clean aligned export passes; the soma is the only node the exporter
      touched; the removed nodes are exactly the spine nodes
  T2  unaligned export (no alignment.json) passes with the identity
  T3  one shaft point deleted from the .hoc -> reported missing, cable short
  T4  one shaft diameter shrunk by 1 percent -> reported as a mismatch
  T5  a point moved by 5 nm -> reported missing AND extra (position, not radius)
  T6  a shaft tip labelled as spine (a continuation) -> pruned as a "spine",
      and the over-threshold count is confirmed VACUOUS (the labeller's own
      criterion bounds it), so section D is the detector
  T7  score_continuations finds that continuation, demotes it, moves its area
      from the spine bucket to the shaft, and lowers F_lit; a cell with no
      continuation gives dF = 0 exactly
  T8  continuation_inspect: the boundary population, dF-by-band (the single
      occupied band carries the whole shift and equals score_continuations),
      the local skeleton roles, and the stratified gallery
  T11 REGRESSION: a .hoc exported WITH the correction must be checked with the
      same export_kw. Without it the rebuild is a different partition and the
      check reports spurious extra points -- exactly what neuron 15543554616
      showed (1566 extra, 470 um of cable) on 2026-09-15.
  T10 O-1, the export hook: export_neuron(demote_continuations=True) applies
      the three-vote correction at step 4b -- before the re-classify and
      before phi -- moving area from the spine bucket to the shaft, leaving
      every coordinate and radius untouched, and holding back the stub that
      the two-observable rule would have taken
  T9  the taper test: a neck-then-head spine reads spine_like, a tapering
      branch reads branch_like, a 60 nm stub is undecidable, the three-vote
      resolution keeps everything the taper objects to, and dF_by_cos_band
      partitions the shaft-like roots exactly once

Pure ASCII, LF only.
"""

import json
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import alignment as al
import check_pruned_hoc as CK
import spine_density as sd
try:
    import shaft_continuation as shc          # lives in Spine Mesh Analysis
except ImportError:
    shc = None
import morphology_exporter as mx
import node_classify as nc

VERBOSE = "-v" in sys.argv
NID = 777
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    if VERBOSE or not ok:
        print("  [%s] %-60s %s" % ("PASS" if ok else "FAIL", name, detail))
    return ok


def toy_neuron(continuation=False):
    """Soma (r 5 um) at the origin, one dendrite of 40 nodes at 300 nm along
    +x with a branch, three spines. Ids >= 1000 are spine nodes. With
    continuation=True a 5-node shaft TIP (ids 2000+) hangs off the last
    dendrite node and is labelled as a spine by the toy labeller."""
    rows = [dict(id=0, p=-1, x=0.0, y=0.0, z=0.0, r=5000.0, annotated_type="Soma")]
    # A wiggle keeps the cable non-collinear: deleting an interior point of a
    # straight polyline leaves its length unchanged, which is not a property
    # any real dendrite has and would blind the cable-length check.
    for i in range(1, 41):
        rows.append(dict(id=i, p=i - 1, x=5000.0 + 300.0 * i,
                         y=120.0 * np.sin(0.7 * i), z=80.0 * np.cos(0.5 * i),
                         r=300.0 - 2.0 * i, annotated_type="Dendrite"))
    for i in range(41, 61):                                # a side branch at 20
        rows.append(dict(id=i, p=20 if i == 41 else i - 1,
                         x=5000.0 + 6000.0 + 60.0 * np.sin(0.9 * i),
                         y=300.0 * (i - 40), z=100.0 * (i - 40),
                         r=180.0, annotated_type="Dendrite"))
    # Spines must be THIN relative to their parent and leave at an angle, or
    # shaft_continuation scores them as continuations -- correctly, by its own
    # criteria (rho >= 0.50 and cos >= 0.70). An earlier fixture gave the
    # base-50 spine half its parent's radius AND pointed it along the branch,
    # and T7 caught it. Directions are perpendicular to the LOCAL shaft tangent.
    nid = 1000
    for base, n, d in ((8, 3, (0.0, 1.0, 0.0)), (25, 4, (0.0, 1.0, 0.0)),
                       (50, 2, (1.0, 0.0, 0.0))):
        par = base
        for j in range(n):
            # neck then head: thin along the neck, thick at the tip. This is
            # the taper signature, and it must hold for ANY n -- an earlier
            # fixture made the profile depend on the absolute index, so a
            # 3-node spine came out monotone and read as a branch.
            _r = 45.0 if j < n - 1 else 120.0
            rows.append(dict(id=nid, p=par,
                             x=rows[base]["x"] + 350.0 * (j + 1) * d[0],
                             y=rows[base]["y"] + 350.0 * (j + 1) * d[1],
                             z=rows[base]["z"] + 350.0 * (j + 1) * d[2],
                             r=_r, annotated_type="Dendrite"))
            par = nid
            nid += 1
    if continuation:
        # A 60 nm stub off node 30, collinear and fat: it PASSES rho and cos
        # and is therefore a false positive for the two-observable scorer.
        # It is the case the length floor and the taper test exist to catch.
        for _j in (0, 1):
            rows.append(dict(id=3000 + _j, p=30 if _j == 0 else 3000,
                             x=rows[30]["x"] + 60.0 * (_j + 1), y=rows[30]["y"],
                             z=rows[30]["z"], r=130.0, annotated_type="Dendrite"))
        # Carries on from node 40 in its own direction at comparable calibre:
        # what shaft_continuation is built to catch.
        par, tip = 40, rows[40]
        for j in range(5):
            rows.append(dict(id=2000 + j, p=par,
                             x=tip["x"] + 300.0 * (j + 1), y=tip["y"],
                             z=tip["z"], r=200.0 - 20.0 * j,   # tapers
                             annotated_type="Dendrite"))
            par = 2000 + j
    return pd.DataFrame(rows)


def toy_label_fn(df, threshold_nm):
    """Labels ids >= 1000 as neck/head; leaves the rest untouched."""
    out = df.copy()
    sp = out["id"] >= 1000
    out.loc[sp, "annotated_type"] = "neck"
    tips = set(out.loc[sp, "id"]) - set(out.loc[sp, "p"])
    out.loc[out["id"].isin(tips), "annotated_type"] = "head"
    return out


def export(df, out_dir, aligned=True, extra=None):
    if aligned:
        th = np.radians(37.0)
        M = np.array([[np.cos(th), -np.sin(th), 0.0],
                      [np.sin(th), np.cos(th), 0.0], [0.0, 0.0, 1.0]])
        soma = np.array([0.0, 0.0, 0.0])
        align_fn = al.make_align_fn(soma, M)
    mx.export_neuron(df, NID, out_dir, label_fn=toy_label_fn,
                     align_fn=align_fn if aligned else None,
                     write_files=True, return_frames=False, verbose=False,
                     **(extra or {}))
    if aligned:
        with open(os.path.join(out_dir, "neuron_%s_alignment.json" % NID),
                  "w") as fh:
            json.dump({"alignment": {"soma_pos_nm": soma.tolist(),
                                     "mean_matrix": M.tolist()}}, fh)


def run_check(df, out_dir):
    return CK.check_pruned_hoc(NID, df, out_dir, mx, al, nc, toy_label_fn)


def edit_hoc(path, fn):
    lines = open(path).read().split("\n")
    lines = fn(lines)
    with open(path, "w", newline="\n") as fh:
        fh.write("\n".join(lines))


def shaft_pt_lines(lines):
    """Indices of pt3dadd lines in dend sections with diam < 1 um (shaft,
    not soma), skipping the first point of each section (a parent repeat)."""
    idx, in_sec, first = [], False, True
    for i, ln in enumerate(lines):
        if CK._RE_SEC.match(ln):
            in_sec, first = ln.strip().startswith("dend"), True
        elif in_sec and ln.strip().startswith("pt3dadd"):
            if not first:
                idx.append(i)
            first = False
    return idx


class _Done(Exception):
    """Clean exit from the fixture block when an optional module is absent."""


def preflight():
    """T0. The exporter's step-4b dependency closure, checked by import."""
    import importlib

    need = (("shaft_continuation", "shaft_continuation-1.1.0",
             "the two-observable scorer. Lives in the Spine Mesh Analysis "
             "folder; copy it beside morphology_exporter.py"),
            ("continuation_inspect", "continuation_inspect v1.1",
             "the taper vote. v1.0 lacks taper_table and is refused"),
            ("spine_density", "spine_density-1.3.0", "vocabulary and build_phi"),
            ("node_classify", None, "classify_frame, step 5"),
            ("soma_enforce", None, "step 6"))
    ok = True
    for name, min_ver, why in need:
        try:
            m = importlib.import_module(name)
        except ImportError as exc:
            ok = check("T0 %s importable" % name, False, "%s -- %s" % (exc, why))
            continue
        got = getattr(m, "MODULE_VERSION", None)
        fine = min_ver is None or (got is not None and str(got) >= min_ver)
        ok = check("T0 %s %s" % (name, ("(needs >= %s)" % min_ver) if min_ver else ""),
                   fine, "%s -- %s" % (got, why)) and ok
    caps = ("taper_table", "three_vote", "vote_summary")
    try:
        import continuation_inspect as ci
        miss = [c for c in caps if not hasattr(ci, c)]
        check("T0 continuation_inspect exposes the taper vote", not miss,
              "missing %s" % miss if miss else "")
    except ImportError:
        pass
    try:
        import morphology_exporter as _mx
        check("T0 morphology_exporter can reach both", _mx.shc is not None
              and _mx.cinsp is not None,
              "shc=%s cinsp=%s" % (_mx.shc is not None, _mx.cinsp is not None))
    except ImportError as exc:
        check("T0 morphology_exporter importable", False, str(exc))
    return ok


def main():
    if not preflight():
        print("\nPREFLIGHT FAILED -- fix the above before running anything "
              "else; the remaining tests would fail for the same reason.")
        return sum(1 for _, o in RESULTS if not o)
    tmp = tempfile.mkdtemp()
    try:
        df = toy_neuron()
        d1 = os.path.join(tmp, "aligned")
        export(df, d1, aligned=True)
        rep = run_check(df, d1)
        if VERBOSE:
            CK.print_report(rep)
        check("T1a clean aligned export passes", rep["ok"],
              "missing %d extra %d diam %d cable diff %.2e non-spine removed %d"
              % (rep["n_missing_in_hoc"], rep["n_extra_in_hoc"],
                 rep["n_diam_mismatch"], rep["cable_diff_um"],
                 rep["n_non_spine_nodes_removed"]))
        check("T1b removed nodes are exactly the 9 spine nodes, 3 spines",
              rep["n_removed"] == 9 and rep["n_spines_removed"] == 3,
              "%d / %d" % (rep["n_removed"], rep["n_spines_removed"]))
        check("T1c only the soma changed between raw and labelled",
              rep["n_changed_non_soma"] == 0 and not rep["ids_lost_raw_to_labelled"],
              "%d changed, %d non-soma" % (rep["n_changed_raw_to_labelled"],
                                            rep["n_changed_non_soma"]))
        check("T1d hoc unique points == pruned frame nodes (= 61)",
              rep["n_hoc_unique_points"] == rep["n_pruned_frame"] == 61,
              "%d vs %d" % (rep["n_hoc_unique_points"], rep["n_pruned_frame"]))
        check("T1e the aligned transform was used (json found)", rep["aligned"])

        d2 = os.path.join(tmp, "unaligned")
        export(df, d2, aligned=False)
        rep2 = run_check(df, d2)
        check("T2 unaligned export passes with the identity",
              rep2["ok"] and not rep2["aligned"])

        hoc = os.path.join(d1, "neuron_%s_aligned.hoc" % NID)
        shutil.copy(hoc, hoc + ".bak")

        idx = shaft_pt_lines(open(hoc).read().split("\n"))
        edit_hoc(hoc, lambda L: [ln for i, ln in enumerate(L) if i != idx[3]])
        r3 = run_check(df, d1)
        check("T3 deleted shaft point -> missing 1, cable shorter, verdict fails",
              (not r3["ok"]) and r3["n_missing_in_hoc"] == 1
              and r3["cable_diff_um"] < -1e-4,
              "missing %d diff %.4f" % (r3["n_missing_in_hoc"], r3["cable_diff_um"]))
        shutil.copy(hoc + ".bak", hoc)

        def shrink(L):
            ln = L[idx[5]]
            m = CK._RE_PT.match(ln)
            d = float(m.group(4)) * 0.99
            L[idx[5]] = "  pt3dadd(%s, %s, %s, %r)" % (m.group(1), m.group(2),
                                                      m.group(3), d)
            return L
        edit_hoc(hoc, shrink)
        r4 = run_check(df, d1)
        check("T4 1%% shrunk diameter -> exactly 1 mismatch, verdict fails",
              (not r4["ok"]) and r4["n_diam_mismatch"] == 1
              and r4["n_missing_in_hoc"] == 0, "mismatch %d" % r4["n_diam_mismatch"])
        shutil.copy(hoc + ".bak", hoc)

        def nudge(L):
            ln = L[idx[7]]
            m = CK._RE_PT.match(ln)
            x = float(m.group(1)) + 0.005                     # 5 nm
            L[idx[7]] = "  pt3dadd(%r, %s, %s, %s)" % (x, m.group(2), m.group(3),
                                                      m.group(4))
            return L
        edit_hoc(hoc, nudge)
        r5 = run_check(df, d1)
        check("T5 point moved 5 nm -> 1 missing and 1 extra",
              (not r5["ok"]) and r5["n_missing_in_hoc"] == 1
              and r5["n_extra_in_hoc"] == 1)
        shutil.copy(hoc + ".bak", hoc)

        dfc = toy_neuron(continuation=True)
        d6 = os.path.join(tmp, "cont")
        export(dfc, d6, aligned=True)
        r6 = run_check(dfc, d6)
        c = r6["pruned_components"]
        check("T6a continuation AND stub pruned as 'spines': 5 components, "
              "one of 5 nodes", r6["n_spines_removed"] == 5
              and (c["n_nodes"] == 5).sum() == 1,
              "%d comps, sizes %s" % (len(c), c["n_nodes"].tolist()))
        check("T6b it is NOT flagged as a lost shaft node (labels agree)",
              r6["ok"] and r6["n_non_spine_nodes_removed"] == 0)
        check("T6c its path length is the largest of the pruned components",
              c["path_len_nm"].max() >= 1200.0,
              "max %.0f nm" % c["path_len_nm"].max())
        check("T6d REGRESSION: the over-threshold count is vacuous",
              r6["n_pruned_components_over_threshold"] == 0
              and r6["pruned_component_max_nm"] <= r6["threshold_nm"],
              "max %.0f nm vs threshold %.0f"
              % (r6["pruned_component_max_nm"], r6["threshold_nm"]))

        # ---- T7: what shaft_continuation would do -------------------------
        # shaft_continuation.py ships with Spine Mesh Analysis, not Stage 1.
        # Without it sections A-C still stand and only T7 is skipped.
        if shc is None:
            print("  [SKIP] T7 -- shaft_continuation.py not importable here; "
                  "copy it into this folder to exercise section D")
            raise _Done
        # cutoff 0: the toy cell is 23 um long, so F_lit (>= 60 um) is NaN by
        # construction -- F_whole is the one with content here.
        sc = CK.score_continuations(r6["labelled_frame"], shc, sd, nid=NID,
                                    cutoff_um=0.0)
        t = sc["table"]
        shaft_like = set(int(v) for v in t.loc[t["is_shaft"], "root"]) \
            if len(t) else set()
        check("T7a the continuation root (2000) is scored shaft-like",
              2000 in shaft_like, "shaft-like roots %s" % sorted(shaft_like))
        check("T7b the three real spines are NOT",
              not ({1000, 1003, 1007} & shaft_like))
        check("T7c rho/cos alone demotes 7 nodes: the continuation (5) AND the "
              "60 nm stub (2), which is the false positive",
              sc["n_nodes_demoted"] == 7 and shaft_like == {2000, 3000},
              str(sc["n_nodes_demoted"]))
        check("T7d area moves out of the spine bucket and F_lit falls",
              sc["A_moved_um2"] > 0 and sc["dF_lit"] < 0,
              "%.3f um2, dF %+.4f" % (sc["A_moved_um2"], sc["dF_lit"]))
        check("T7e spine + shaft area is conserved by the demotion",
              abs((sc["A_spine_um2_corrected"] + sc["A_shaft_um2_corrected"])
                  - (sc["A_spine_um2_as_exported"] + sc["A_shaft_um2_as_exported"]))
              < 1e-6,
              "%.4f vs %.4f" % (sc["A_spine_um2_corrected"] + sc["A_shaft_um2_corrected"],
                                sc["A_spine_um2_as_exported"] + sc["A_shaft_um2_as_exported"]))
        sc0 = CK.score_continuations(rep["labelled_frame"], shc, sd, nid=NID,
                                     cutoff_um=0.0)
        check("T7f a cell with no continuation: nothing demoted, dF exactly 0",
              sc0["n_shaft_like"] == 0 and sc0["dF_lit"] == 0.0
              and sc0["n_nodes_demoted"] == 0,
              "%d shaft-like" % sc0["n_shaft_like"])
        if VERBOSE:
            CK.print_continuations(sc)

        # ---- T8: the inspector ------------------------------------------
        import continuation_inspect as CI
        lab6 = r6["labelled_frame"]
        table, rep = shc.score_spine_roots(lab6, spine_density=sd)
        bp = CI.boundary_population(table, rep["rho_shaft_min"], rep["cos_shaft_min"])
        check("T8a boundary population sees both shaft-like roots",
              bp["n_shaft_like"] == 2, str(bp["n_shaft_like"]))
        band = CI.dF_by_band(lab6, table, sd, nid=NID, cutoff_um=0.0)
        allrow = band[band["group"].str.startswith("ALL")].iloc[0]
        occ = band[band["group"].str.startswith("rho") & (band["n_roots"] > 0)]
        check("T8b the rho bands sum to the all-at-once shift, which equals "
              "score_continuations",
              len(occ) >= 1 and abs(allrow["dF_lit"] - sc["dF_lit"]) < 1e-12
              and abs(band.attrs["sum_of_band_dF"] - allrow["dF_lit"]) < 5e-4,
              "%s sum %.5f all %.5f" % (occ["group"].tolist(),
                                        band.attrs["sum_of_band_dF"],
                                        allrow["dF_lit"]))
        check("T8c empty bands leave F untouched",
              (band[band["n_roots"] == 0]["dF_lit"] == 0).all())
        sub, bpid = CI.local_skeleton(lab6, 2000)
        roles = sub["role"].value_counts().to_dict()
        check("T8d local skeleton of the continuation: 5 component nodes off bp 40, "
              "parent path, no siblings",
              roles.get("component") == 5 and bpid == 40
              and roles.get("parent_path", 0) > 0 and roles.get("sibling", 0) == 0,
              str(roles))
        sub2, bp2 = CI.local_skeleton(lab6, 1000)
        check("T8e a mid-shaft spine root has siblings (the shaft continues)",
              bp2 == 8 and (sub2["role"] == "sibling").sum() > 0)
        picks = CI.pick_gallery(table, k=6)
        check("T8f gallery picks only shaft-like roots",
              set(picks) <= set(table.loc[table["is_shaft"], "root"].astype(int)))

        # ---- T9: the taper test, the third observable --------------------
        tap = CI.taper_table(lab6, table["root"]).set_index("root")
        check("T9a a neck-then-head spine reads spine_like, at any node count",
              all(tap.loc[r, "verdict"] == "spine_like" for r in (1000, 1003, 1007))
              and int(tap.loc[1007, "n_path"]) == 2,
              tap["verdict"].to_dict())
        check("T9b the tapering continuation reads branch_like",
              tap.loc[2000, "verdict"] == "branch_like"
              and abs(tap.loc[2000, "bulge"] - 1.0) < 1e-9
              and tap.loc[2000, "s_peak_frac"] == 0.0)
        check("T9c the 60 nm stub is undecidable, not guessed",
              tap.loc[3000, "verdict"] == "undecidable"
              and "shorter than" in tap.loc[3000, "why"], tap.loc[3000, "why"])
        check("T9d bulge is the distal max over the preceding min",
              abs(tap.loc[1000, "bulge"] - 120.0 / 45.0) < 1e-9
              and tap.loc[1000, "s_peak_frac"] == 1.0)
        tv = CI.three_vote(table, tap.reset_index())
        check("T9e only the continuation is demoted by three votes",
              set(tv.loc[tv["demote"], "root"].astype(int)) == {2000},
              str(tv.loc[tv["demote"], "root"].tolist()))
        check("T9f the stub passed rho AND cos but is NOT demoted",
              bool(tv.set_index("root").loc[3000, "is_shaft"])
              and not bool(tv.set_index("root").loc[3000, "demote"]))
        vs = CI.vote_summary(tv)
        check("T9g summary: 2 shaft-like, 1 demoted, 1 undecidable, and the "
              "head-label disagreement is reported not hidden",
              vs["n_shaft_like_rho_cos"] == 2 and vs["n_demote_three_vote"] == 1
              and vs["n_undecidable"] == 1
              and vs["n_taper_vs_headlabel_disagree"] == 1, str(vs))
        cb = CI.dF_by_cos_band(lab6, table, sd, nid=NID, cutoff_um=0.0)
        check("T9h cos bands partition the shaft-like roots exactly once",
              int(cb["n_roots"].sum()) == int(table["is_shaft"].sum()),
              "%d vs %d" % (cb["n_roots"].sum(), table["is_shaft"].sum()))
        check("T9i an occupied cos band moves F, empty ones do not",
              (cb[cb["n_roots"] == 0]["dF_lit"] == 0).all()
              and (cb[cb["n_roots"] > 0]["dF_lit"] < 0).all())
        # ---- T10: the export hook (O-1) ----------------------------------
        def _exp(**kw):
            d = tempfile.mkdtemp()
            try:
                return mx.export_neuron(dfc, NID, d, label_fn=toy_label_fn,
                                        write_files=False, return_frames=True,
                                        verbose=False, **kw)
            finally:
                shutil.rmtree(d, ignore_errors=True)

        e_off = _exp()
        e_3 = _exp(demote_continuations=True)
        e_2 = _exp(demote_continuations=True,
                   continuation_kw={"require_taper": False})
        check("T10a off by default: nothing applied, frame unchanged",
              e_off["continuation_report"] == {"applied": False})
        r3 = e_3["continuation_report"]
        check("T10b three votes demote ONLY the continuation, not the stub",
              r3["demoted_roots"] == [2000] and r3["n_nodes_demoted"] == 5
              and r3["n_undecidable"] == 1, str(r3["demoted_roots"]))
        check("T10c two observables alone would also take the 60 nm stub",
              sorted(e_2["continuation_report"]["demoted_roots"]) == [2000, 3000],
              str(e_2["continuation_report"]["demoted_roots"]))
        a, b = e_off["frames"]["labelled"], e_3["frames"]["labelled"]
        check("T10d geometry untouched: ids, coordinates and radii identical",
              a["id"].equals(b["id"]) and np.allclose(
                  a[["x", "y", "z", "r"]].to_numpy(),
                  b[["x", "y", "z", "r"]].to_numpy()))
        p_off = sd.build_phi(a, nid=NID, input_units="nm")
        p_3 = sd.build_phi(b, nid=NID, input_units="nm")
        moved = float(p_off["spine_area_um2"].sum() - p_3["spine_area_um2"].sum())
        gained = float(p_3["shaft_area_um2"].sum() - p_off["shaft_area_um2"].sum())
        check("T10e area moves from the spine bucket to the shaft, conserved",
              moved > 0 and abs(moved - gained) < 1e-6,
              "%.4f um2 moved, %.4f gained" % (moved, gained))
        check("T10f phi is built on the CORRECTED partition (step 4b precedes 7)",
              abs(float(e_3["phi"]["nocap"]["spine_area_um2"].sum())
                  - float(p_3["spine_area_um2"].sum())) < 1e-9
              if "phi" in e_3 else True)
        check("T10g compartment_class is re-classified after the demotion",
              not (b.loc[b["id"].isin(range(2000, 2005)), "compartment_class"]
                   .astype(str).str.lower().str.contains("spine").any()))
        import continuation_inspect as _ci
        check("T10g' continuation_inspect advertises its capabilities and has "
              "them all", all(hasattr(_ci, n) for n in _ci.CAPABILITIES)
              and _ci.MODULE_VERSION >= "continuation_inspect v1.1",
              _ci.MODULE_VERSION)
        _saved = {n: getattr(_ci, n) for n in ("taper_table", "three_vote",
                                               "vote_summary")}
        try:
            for n in _saved:
                delattr(_ci, n)
            _exp(demote_continuations=True)
            check("T10g'' REGRESSION: a stale continuation_inspect is refused "
                  "at the door, not deep in the export", False, "accepted")
        except ImportError as _e:
            check("T10g'' REGRESSION: a stale continuation_inspect is refused "
                  "at the door, not deep in the export",
                  "v1.1 or later" in str(_e) and "taper_table" in str(_e),
                  str(_e)[:70])
        finally:
            for n, v in _saved.items():
                setattr(_ci, n, v)
        # ---- T11: the export_kw round trip --------------------------------
        _d11 = tempfile.mkdtemp()
        try:
            _kw = {"demote_continuations": True}
            export(dfc, _d11, aligned=True, extra=_kw)
            _bad = CK.check_pruned_hoc(NID, dfc, _d11, mx, al, nc, toy_label_fn)
            check("T11a WITHOUT export_kw the check reports spurious extras",
                  (not _bad["ok"]) and _bad["n_extra_in_hoc"] > 0
                  and _bad["n_missing_in_hoc"] == 0
                  and not _bad["continuation_applied"],
                  "extra %d, cable diff %+.3f um"
                  % (_bad["n_extra_in_hoc"], _bad["cable_diff_um"]))
            _good = CK.check_pruned_hoc(NID, dfc, _d11, mx, al, nc, toy_label_fn,
                                        export_kw=_kw)
            check("T11b WITH export_kw it passes: same partition, 0 extra, "
                  "cable matches",
                  _good["ok"] and _good["n_extra_in_hoc"] == 0
                  and abs(_good["cable_diff_um"]) < 1e-6
                  and _good["continuation_applied"],
                  "extra %d, cable diff %.2e, demoted %s"
                  % (_good["n_extra_in_hoc"], _good["cable_diff_um"],
                     _good["n_continuations_demoted"]))
            check("T11c the corrected rebuild has MORE nodes than the "
                  "uncorrected one, by the demoted branches",
                  _good["n_pruned_frame"] > _bad["n_pruned_frame"],
                  "%d vs %d" % (_good["n_pruned_frame"], _bad["n_pruned_frame"]))
        finally:
            shutil.rmtree(_d11, ignore_errors=True)
        check("T10h the report records the thresholds actually used",
              r3["require_taper"] and r3["min_len_nm"] == 150.0
              and r3["rho_shaft_min"] == 0.5 and r3["scorer_version"]
              and r3["inspect_version"], "%s / %s" % (r3["scorer_version"],
                                                      r3["inspect_version"]))
    except _Done:
        pass
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    n_fail = sum(1 for _, ok in RESULTS if not ok)
    print("\n%d checks passed, %d failed" % (len(RESULTS) - n_fail, n_fail))
    print("ALL GREEN" if n_fail == 0 else "FAILURES")
    return n_fail


if __name__ == "__main__":
    sys.exit(main())
