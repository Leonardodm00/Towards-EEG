"""Smoke test for morphology_exporter (stage S1.0).

Runs standalone (python3 test_morphology_exporter.py) or under pytest. Every
expected quantity is recomputed here from the raw synthetic geometry with a
local frustum function; the test never asks the module to confirm its own
arithmetic. The NEURON checks are skipped gracefully if the package is absent.

Coverage
  E1  prune_spines removes spine subtrees and records every base BEFORE pruning
  E2  section decomposition does NOT split at a synapse (the O8 regression) and
      DOES split at class change and branch point
  E3  phi branches and emitted dend sections are the SAME partition
  E4  section map: every emitted segment joins to a phi row on (from, to)
  E5  spine_base_section matches spine_density's own base attribution
  E6  .hoc is ASCII, LF, has no synapse arrays, and area is conserved against
      an independent frustum sum
  E7  end to end: files written, provenance JSON valid and ASCII
  E8  NEURON loads the file and I-13a / I-15 / I-16 / I-18 pass
  E9  a cell with no root fails softly and writes nothing
  E10 REGRESSION: a relabelled glia node stays visible to phi
  E11 mislabels resolve BEFORE labelling; re-classify is mandatory
"""

import json
import math
import os
import shutil
import tempfile

import numpy as np
import pandas as pd

import spine_density as sd
import node_classify as nc
import soma_enforce as se
import morphology_exporter as mx


COLS = ["id", "p", "x", "y", "z", "r", "annotated_type"]


def _frus(r1, r2, L):
    """Independent frustum lateral area, nm in -> nm^2 out."""
    return math.pi * (r1 + r2) * math.sqrt((r1 - r2) ** 2 + L ** 2)


def _mk(rows):
    return pd.DataFrame(rows, columns=COLS)


def _cell():
    """Soma, one dendrite that branches, two spines, an AIS and an axon.

        0 soma
        |-- 1 -- 2 -- 3 (branch pt) -- 4 -- 5 tip
        |                           \\- 6 -- 7 tip
        |    spine on 2  : 20 (neck) -- 21 (head)
        |    spine on 6  : 22 (neck) -- 23 (head)
        |-- 10 AIS -- 11 AIS -- 12 axon
    """
    return _mk([
        (0,  -1,      0.0,    0.0,   0.0, 5000.0, "Soma"),
        (1,   0,   1000.0,    0.0,   0.0,  400.0, "Dendrite"),
        (2,   1,   2000.0,    0.0,   0.0,  380.0, "Dendrite"),
        (3,   2,   3000.0,    0.0,   0.0,  360.0, "Dendrite"),
        (4,   3,   4000.0,    0.0,   0.0,  300.0, "Dendrite"),
        (5,   4,   5000.0,    0.0,   0.0,  280.0, "Dendrite"),
        (6,   3,   3000.0, 1000.0,   0.0,  300.0, "Dendrite"),
        (7,   6,   3000.0, 2000.0,   0.0,  260.0, "Dendrite"),
        (20,  2,   2000.0,    0.0, 200.0,  100.0, "neck"),
        (21, 20,   2000.0,    0.0, 500.0,  300.0, "head"),
        (22,  6,   3000.0, 1000.0, 200.0,  100.0, "neck"),
        (23, 22,   3000.0, 1000.0, 500.0,  300.0, "head"),
        (10,  0,  -1000.0,    0.0,   0.0,  250.0, "AIS"),
        (11, 10,  -2000.0,    0.0,   0.0,  200.0, "AIS"),
        (12, 11,  -3000.0,    0.0,   0.0,  150.0, "Axon"),
    ])


def _prepared():
    df = nc.classify_frame(_cell())
    df, _ = se.enforce_soma(df, nid="T", verbose=False)
    return df


# --------------------------------------------------------------------------- #
def test_E1_prune_records_bases():
    df = _prepared()
    pruned, info = mx.prune_spines(df)
    assert info["n_spines"] == 2, info
    assert info["n_spine_nodes"] == 4, info
    assert info["n_nodes_removed"] == 4, info
    assert len(pruned) == len(df) - 4
    bases = {b["spine_root_id"]: b["base_node_id"] for b in info["spine_bases"]}
    assert bases == {20: 2, 22: 6}, bases
    # every spine node is gone, every shaft node survives
    assert not set(pruned["id"]) & {20, 21, 22, 23}
    assert {0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12} <= set(pruned["id"])


def test_E2_sections_split_correctly():
    df = _prepared()
    poisoned = df.copy()
    poisoned["synapse_label"] = "exc_syn"          # every node
    pruned_a, _ = mx.prune_spines(mx.assign_domain(df))
    pruned_b, _ = mx.prune_spines(mx.assign_domain(poisoned))
    sa = mx.decompose_sections(pruned_a)
    sb = mx.decompose_sections(pruned_b)
    assert len(sa) == len(sb), "a synapse label changed the section count (O8)"
    arrays = sorted({s["array"] for s in sa})
    assert arrays == ["ais", "axon", "dend", "soma"], arrays
    nc.assert_no_synapse_leakage(arrays, "sections")

    # split at class change: AIS -> axon are different sections
    ais = [s for s in sa if s["array"] == "ais"]
    axon = [s for s in sa if s["array"] == "axon"]
    assert len(ais) == 1 and len(axon) == 1, (len(ais), len(axon))
    # the axon section starts with the last AIS node (continuity, no gap)
    assert axon[0]["nodes"][0] == 11, axon[0]["nodes"]

    # split at the branch point: node 3 has two dendrite children
    dend = [s for s in sa if s["array"] == "dend"]
    assert len(dend) == 3, [d["nodes"] for d in dend]
    assert all(d["nodes"][0] in (0, 3) for d in dend), [d["nodes"] for d in dend]


def test_E3_phi_and_sections_same_partition():
    df = _prepared()
    phi = sd.build_phi(df, nid="T", input_units="nm")
    pruned, _ = mx.prune_spines(mx.assign_domain(df))
    secs = mx.decompose_sections(pruned)

    phi_edges = {(int(r.node_from), int(r.node_to)) for r in
                 phi.itertuples(index=False)}
    dend_edges = set()
    for s in secs:
        if s["array"] != "dend":
            continue
        n = s["nodes"]
        dend_edges |= {(n[i - 1], n[i]) for i in range(1, len(n))}
    assert phi_edges == dend_edges, (
        sorted(phi_edges - dend_edges), sorted(dend_edges - phi_edges))

    # and the branch count matches the dend section count
    n_dend = sum(1 for s in secs if s["array"] == "dend")
    assert phi["branch_id"].nunique() == n_dend, (
        phi["branch_id"].nunique(), n_dend)


def test_E4_section_map_joins():
    df = _prepared()
    phi = sd.build_phi(df, nid="T", input_units="nm")
    pruned, info = mx.prune_spines(mx.assign_domain(df))
    secs = mx.decompose_sections(pruned)
    seg_map, sec_tab = mx.build_section_map(secs, phi)

    dend_rows = seg_map[seg_map["array"] == "dend"]
    assert (dend_rows["branch_id"] >= 0).all(), "a dend segment failed to join phi"
    # non-dendritic segments have no phi row, by construction
    other = seg_map[seg_map["array"] != "dend"]
    assert (other["branch_id"] == -1).all()

    # spine area recovered through the map equals phi's total
    assert abs(dend_rows["spine_area_um2"].sum()
               - phi["spine_area_um2"].sum()) < 1e-12
    # every dend section maps to exactly one branch
    dsec = sec_tab[sec_tab["array"] == "dend"]
    assert (dsec["n_branch_ids"] == 1).all(), dsec
    assert sec_tab["n_segments"].sum() == sum(len(s["nodes"]) - 1 for s in secs)


def test_E5_spine_base_matches_spine_density():
    df = _prepared()
    phi = sd.build_phi(df, nid="T", input_units="nm")
    pruned, info = mx.prune_spines(mx.assign_domain(df))
    secs = mx.decompose_sections(pruned)
    base_tab = mx.resolve_spine_base_sections(info["spine_bases"], secs, pruned)
    assert len(base_tab) == 2
    assert (base_tab["section_id"] >= 0).all(), base_tab

    # spine_density attributes each spine to the segment whose DISTAL node is
    # the base; the exporter must land in the section containing that segment
    seg_map, _ = mx.build_section_map(secs, phi)
    for r in base_tab.itertuples(index=False):
        row = phi[phi["node_to"] == r.base_node_id]
        assert len(row) == 1, r.base_node_id
        assert float(row["spine_area_um2"].iloc[0]) > 0.0
        owner = seg_map[(seg_map["node_to"] == r.base_node_id)
                        & (seg_map["array"] == "dend")]
        assert int(owner["section_id"].iloc[0]) == r.section_id, (r, owner)


def test_E6_hoc_geometry_and_encoding():
    tmp = tempfile.mkdtemp()
    try:
        df = _prepared()
        pruned, _ = mx.prune_spines(mx.assign_domain(df))
        secs = mx.decompose_sections(pruned)
        p = os.path.join(tmp, "t.hoc")
        info = mx.write_hoc(secs, pruned, p, input_units="nm")

        raw = open(p, "rb").read()
        assert all(b < 128 for b in raw), "hoc must be pure ASCII"
        assert b"\r\n" not in raw, "hoc must be LF only"
        assert b"exc_syn" not in raw and b"inh_syn" not in raw
        assert b"create soma[1]" in raw, raw[:200]
        assert b"create ais[1]" in raw

        # shaft area, recomputed independently from the raw geometry
        expect = (_frus(5000, 400, 1000) + _frus(400, 380, 1000)
                  + _frus(380, 360, 1000)                       # 0-1,1-2,2-3
                  + _frus(360, 300, 1000) + _frus(300, 280, 1000)  # 3-4,4-5
                  + _frus(360, 300, 1000) + _frus(300, 260, 1000)  # 3-6,6-7
                  ) / 1e6
        phi = sd.build_phi(df, nid="T", input_units="nm")
        assert abs(float(phi["shaft_area_um2"].sum()) - expect) < 1e-9, (
            float(phi["shaft_area_um2"].sum()), expect)
        assert info["arrays"]["dend"] == 3, info
    finally:
        shutil.rmtree(tmp)


def test_E7_end_to_end():
    tmp = tempfile.mkdtemp()
    try:
        df = _cell()
        df["synapse_label"] = [None] * len(df)
        df.loc[df["id"] == 4, "synapse_label"] = "exc_syn"
        df.loc[df["id"] == 21, "synapse_label"] = "exc_syn"   # on a spine head
        res = mx.export_neuron(df, "T1", tmp, label_fn=None, verbose=False)

        assert res["qc_status"] == se.QC_PASS, res["reasons"]
        assert res["n_sections"] == 6, res["n_sections"]      # 1+3+1+1
        assert res["spine_report"]["n_spines"] == 2
        assert res["n_spines_unattached"] == 0
        assert res["n_sections_multi_branch"] == 0
        for k in ("hoc", "phi", "segment_map", "section_table",
                  "spine_bases", "synapses", "provenance"):
            assert k in res["files"], k

        syn = pd.read_csv(res["files"]["synapses"])
        assert len(syn) == 2, syn
        assert set(syn["node_id"]) == {4, 21}
        # coordinates converted nm -> um
        assert abs(float(syn[syn.node_id == 4]["x"].iloc[0]) - 4.0) < 1e-12

        raw = open(res["files"]["provenance"], "rb").read()
        assert all(b < 128 for b in raw), "provenance must be ASCII"
        prov = json.loads(raw.decode("ascii"))
        assert "exporter_id" in prov
        assert "thr:4000nm" in prov["exporter_id"], prov["exporter_id"]
        assert prov["record"]["spine_length_threshold_nm"] == 4000.0

        # f_implied recomputed independently
        a_sp = (_frus(380, 100, 200) + _frus(100, 300, 300)
                + _frus(300, 100, 200) + _frus(100, 300, 300)) / 1e6
        assert abs(res["A_spine_um2"] - a_sp) < 1e-9, (res["A_spine_um2"], a_sp)
        assert abs(res["f_implied"]
                   - (1.0 + res["A_spine_um2"] / res["A_shaft_um2"])) < 1e-12
    finally:
        shutil.rmtree(tmp)


def test_E8_neuron_validation():
    try:
        import neuron                                        # noqa: F401
    except Exception:
        print("      (NEURON not installed -- E8 skipped)")
        return
    tmp = tempfile.mkdtemp()
    try:
        res = mx.export_neuron(_cell(), "T2", tmp, verbose=False)
        rep = mx.validate_hoc(res["files"]["hoc"]["path"])
        assert rep["neuron_available"] is True
        assert rep["I16_no_synapse_sections"] is True, rep
        assert rep["I13a_vocabulary_admissible"] is True, rep
        assert rep["I15_single_soma_is_root"] is True, rep
        assert rep["I18_name_geometry_agree"] is True, rep
        assert rep["ok"] is True, rep["violations"]
        assert rep["n_sections"] == 6, rep
        assert rep["arrays"] == {"soma": 1, "dend": 3, "ais": 1, "axon": 1}, rep
    finally:
        shutil.rmtree(tmp)


def test_E9_no_root_fails_softly():
    tmp = tempfile.mkdtemp()
    try:
        df = _cell()
        df.loc[df["id"] == 0, "p"] = 999
        res = mx.export_neuron(df, "T3", tmp, verbose=False)
        assert res["qc_status"] == se.QC_FAIL, res
        assert "no_root" in res["reasons"], res
        assert res["files"] == {}, "nothing must be written for a failed cell"
    finally:
        shutil.rmtree(tmp)


def test_E10_glia_relabel_keeps_partitions_aligned():
    """REGRESSION, found on neuron_794820508 (1760 dend sections vs 1759 phi
    branches). A glia node relabelled to dend must be visible to BOTH phi and
    the section decomposition, or the section map has an unjoinable row."""
    df = _cell()
    # an astrocyte node spliced onto the dendrite, 1.2 um away so the k-NN
    # vote resolves it to dend
    df.loc[len(df)] = (30, 4, 4300.0, 0.0, 0.0, 200.0, "Astrocyte")
    df = nc.classify_frame(df)
    df, mis = nc.resolve_mislabelled_nodes(df, policy="knn_relabel", k=3)
    assert mis["n_relabelled"] == 1, mis
    assert mis["subtrees"][0]["assigned_class"] == nc.CLS_DEND, mis

    df, _ = se.enforce_soma(df, nid="G", verbose=False)
    phi = sd.build_phi(df, nid="G", input_units="nm")
    pruned, _ = mx.prune_spines(mx.assign_domain(df))
    secs = mx.decompose_sections(pruned)
    seg_map, sec_tab = mx.build_section_map(secs, phi)

    dend = sec_tab[sec_tab["array"] == "dend"]
    assert int((dend["branch_id"] < 0).sum()) == 0, (
        "a dend section did not join phi:\n%s" % dend[dend["branch_id"] < 0])
    assert phi["branch_id"].nunique() == len(dend), (
        phi["branch_id"].nunique(), len(dend))
    dseg = seg_map[seg_map["array"] == "dend"]
    assert (dseg["branch_id"] >= 0).all()


def test_E11_resolution_precedes_labelling():
    """The reorder that makes the spine bias unreachable, and the re-classify
    that the reorder makes mandatory.

    (a) at resolution time no node is head/neck, so 'spine' cannot be voted;
    (b) the labeller then rewrites annotated_type and leaves compartment_class
        stale, so classify_frame MUST run again;
    (c) the step-3 resolution survives the re-classify only because it was
        written into annotated_type (rewrite_annotation), which is what couples
        the two fixes.
    """
    seen = {}

    def spy_labeller(df, thr):
        seen["classes_at_labelling"] = set(
            df["compartment_class"].astype(str).unique())
        seen["annots_at_labelling"] = set(df["annotated_type"].astype(str).unique())
        seen["threshold"] = thr
        out = df.copy()
        # relabel a real terminal twig, exactly as the labeller would
        out.loc[out["id"].isin([20, 21, 22, 23]), "annotated_type"] = "head"
        return out

    # a RAW frame, as the H01 CSV actually arrives: the future spine nodes are
    # annotated 'Dendrite' and only the labeller turns them into head / neck
    df = _cell()
    df.loc[df["id"].isin([20, 21, 22, 23]), "annotated_type"] = "Dendrite"
    df.loc[len(df)] = (30, 4, 4300.0, 0.0, 0.0, 200.0, "Astrocyte")
    tmp = tempfile.mkdtemp()
    try:
        res = mx.export_neuron(df, "R", tmp, label_fn=spy_labeller, verbose=False)

        # (a) the labeller saw a frame with no spine class at all
        assert nc.CLS_SPINE not in seen["classes_at_labelling"], seen
        assert not (seen["annots_at_labelling"] & set(sd.SPINE_LABELS)), seen
        assert seen["threshold"] == mx.SPINE_LENGTH_THRESHOLD_NM

        # (a) so the vote could not have produced 'spine'
        st = res["mislabel_report"]["subtrees"][0]
        assert st["assigned_class"] == nc.CLS_DEND, st
        assert nc.CLS_SPINE not in st["vote"], st

        # (b) the re-classify happened: spines exist in the post-label counts
        assert res["classes_labelled"][nc.CLS_SPINE] == 4, res["classes_labelled"]
        assert nc.CLS_SPINE not in res["classes_raw"], res["classes_raw"]
        assert res["spine_report"]["n_spine_nodes"] == 4

        # (c) the resolved node survived the re-classify as dendrite, and is
        #     visible to phi -- partitions still aligned
        assert res["n_sections_multi_branch"] == 0, res
        sec = pd.read_csv(res["files"]["section_table"])
        d = sec[sec["array"] == "dend"]
        assert int((d["branch_id"] < 0).sum()) == 0, d[d["branch_id"] < 0]
        assert res["n_branches"] == len(d), (res["n_branches"], len(d))
    finally:
        shutil.rmtree(tmp)


# --------------------------------------------------------------------------- #
def _run_all():
    tests = [
        ("E1 prune records bases before pruning", test_E1_prune_records_bases),
        ("E2 sections split on class, not synapse", test_E2_sections_split_correctly),
        ("E3 phi branches == dend sections", test_E3_phi_and_sections_same_partition),
        ("E4 section map joins on (from,to)", test_E4_section_map_joins),
        ("E5 spine base matches spine_density", test_E5_spine_base_matches_spine_density),
        ("E6 hoc encoding + area conservation", test_E6_hoc_geometry_and_encoding),
        ("E7 end to end + provenance", test_E7_end_to_end),
        ("E8 NEURON: I-13a/I-15/I-16/I-18", test_E8_neuron_validation),
        ("E9 no root fails softly, writes nothing", test_E9_no_root_fails_softly),
        ("E10 glia relabel keeps partitions aligned", test_E10_glia_relabel_keeps_partitions_aligned),
        ("E11 resolution precedes labelling", test_E11_resolution_precedes_labelling),
    ]
    n_pass = 0
    for name, fn in tests:
        try:
            fn()
            print("[PASS] %s" % name)
            n_pass += 1
        except AssertionError as e:
            print("[FAIL] %s -- %s" % (name, e))
        except Exception as e:                     # noqa: BLE001
            print("[ERROR] %s -- %r" % (name, e))
    print("-" * 62)
    print("%d / %d passed" % (n_pass, len(tests)))
    return n_pass == len(tests)


if __name__ == "__main__":
    import sys
    sys.exit(0 if _run_all() else 1)
