"""Smoke test for node_classify (stage S1.1).

Runs standalone (python3 test_node_classify.py) or under pytest. Expectations
are stated independently of the module: the O8 regression test in particular
does not ask the module what it thinks a synapse is, it constructs a frame in
which EVERY node carries a synapse label and asserts the classification is
bit-identical to the frame with no synapse column at all.

Coverage
  N1  the measured H01 vocabulary maps to the intended classes
  N2  AIS and Astrocyte do NOT fall through to dend (the inherited defect)
  N3  O8 REGRESSION: synapse labels cannot influence classification
  N4  assert_no_synapse_leakage rejects a synapse label anywhere (I-16)
  N5  section_vocabulary drops spines; section_array_name maps and raises
  N6  extract_synapse_frame / strip_synapse_column round trip, nm -> um
  N7  resolve_mislabelled_nodes: subtree grouping, vote, guard, drop cascade
  N8  the dend rule IS spine_density.SHAFT_REGEX -- one source of truth (O7)
  N9  REGRESSION: a relabel moves annotated_type too, or phi and the
      exporter silently partition the tree differently
  N10 a spine node votes as dendrite, and 'spine' can never win
  N11 assert_domain_collapsed: DOM_NONE passes; DOM_APICAL/DOM_BASAL and
      apic_dend/basal_dend all raise; section_array_name is UNCHANGED by it
"""

import re

import numpy as np
import pandas as pd

import spine_density as sd
import node_classify as nc


COLS = ["id", "p", "x", "y", "z", "r", "annotated_type"]


def _mk(rows):
    return pd.DataFrame(rows, columns=COLS)


def _cell():
    """Soma at origin, one dendrite with a spine, an axon, an AIS, an
    astrocyte process hanging off the soma 6 um away (the measured topology)."""
    return _mk([
        (0,  -1,     0.0,    0.0, 0.0, 5000.0, "Soma"),
        (1,   0,  1000.0,    0.0, 0.0,  300.0, "Dendrite"),
        (2,   1,  2000.0,    0.0, 0.0,  280.0, "Dendrite"),
        (3,   2,  2000.0,    0.0, 200.0, 100.0, "neck"),
        (4,   3,  2000.0,    0.0, 500.0, 300.0, "head"),
        (5,   0, -1000.0,    0.0, 0.0,  200.0, "AIS"),
        (6,   5, -2000.0,    0.0, 0.0,  150.0, "Axon"),
        (7,   0,     0.0, 6000.0, 0.0,   72.0, "Astrocyte"),
        (8,   7,     0.0, 7000.0, 0.0,   70.0, "Astrocyte"),
        (9,   8,     0.0, 8000.0, 0.0,   68.0, "Astrocyte"),
    ])


# --------------------------------------------------------------------------- #
def test_N1_vocabulary():
    got = {a: nc.classify_node(a) for a in
           ("Soma", "Dendrite", "Axon", "AIS", "Astrocyte",
            "apical", "head", "neck", "spine", "Mitochondrion", "")}
    assert got["Soma"] == nc.CLS_SOMA, got
    assert got["Dendrite"] == nc.CLS_DEND, got
    assert got["Axon"] == nc.CLS_AXON, got
    assert got["AIS"] == nc.CLS_AIS, got
    assert got["Astrocyte"] == nc.CLS_GLIA, got
    assert got["apical"] == nc.CLS_DEND, got
    for s in ("head", "neck", "spine"):
        assert got[s] == nc.CLS_SPINE, (s, got[s])
    # total function: nothing falls through to dend
    assert got["Mitochondrion"] == nc.CLS_UNKNOWN, got
    assert got[""] == nc.CLS_UNKNOWN, got


def test_N2_ais_astrocyte_not_dend():
    """The inherited get_hoc_type sent both of these to 'dend'."""
    for label in ("AIS", "ais", "axon_initial_segment", "Axon Initial Segment"):
        assert nc.classify_node(label) == nc.CLS_AIS, label
    for label in ("Astrocyte", "astrocyte", "glia"):
        assert nc.classify_node(label) == nc.CLS_GLIA, label
    # and AIS must not be swallowed by the axon rule
    assert nc.classify_node("AIS") != nc.CLS_AXON


def test_N3_o8_regression_synapse_cannot_leak():
    base = _cell()
    poisoned = base.copy()
    poisoned["synapse_label"] = "exc_syn"          # every single node
    a = nc.classify_frame(base)["compartment_class"].tolist()
    b = nc.classify_frame(poisoned)["compartment_class"].tolist()
    assert a == b, "synapse label changed the classification -- O8 has returned"
    assert nc.CLS_SOMA in a, a
    nc.assert_no_synapse_leakage(b)
    # the input frame must not be mutated
    assert "compartment_class" not in base.columns


def test_N4_i16_leakage_guard():
    nc.assert_no_synapse_leakage(["soma", "dend", "axon"])
    for bad in (["dend", "exc_syn"], ["inh_syn"], ["weird_syn"]):
        try:
            nc.assert_no_synapse_leakage(bad)
        except ValueError:
            continue
        raise AssertionError("did not reject %r" % bad)


def test_N5_vocabulary_and_arrays():
    df = nc.classify_frame(_cell())
    voc = nc.section_vocabulary(df)
    classes = {c for c, _ in voc}
    assert nc.CLS_SPINE not in classes, "spines are pruned, they emit no section"
    assert classes == {nc.CLS_SOMA, nc.CLS_DEND, nc.CLS_AXON,
                       nc.CLS_AIS, nc.CLS_GLIA}, classes
    assert all(d == nc.DOM_NONE for _, d in voc), voc

    assert nc.section_array_name(nc.CLS_SOMA) == "soma"
    assert nc.section_array_name(nc.CLS_AIS) == "ais"
    assert nc.section_array_name(nc.CLS_DEND, nc.DOM_APICAL) == "apic_dend"
    assert nc.section_array_name(nc.CLS_DEND, nc.DOM_BASAL) == "basal_dend"
    try:
        nc.section_array_name(nc.CLS_GLIA)
    except ValueError:
        pass
    else:
        raise AssertionError("glia must have no section array")


def test_N6_synapse_frame_roundtrip():
    df = _cell()
    df["synapse_label"] = [None, "exc_syn", None, None, None,
                           "inh_syn", None, None, None, None]
    syn = nc.extract_synapse_frame(df, nid=42, input_units="nm")
    assert len(syn) == 2, syn
    assert set(syn["synapse_label"]) == {"exc_syn", "inh_syn"}
    # nm -> um, recomputed independently
    row = syn[syn["node_id"] == 1].iloc[0]
    assert abs(float(row["x"]) - 1000.0 / 1000.0) < 1e-12
    assert list(syn.columns) == ["nid", "node_id", "x", "y", "z", "synapse_label"]

    stripped = nc.strip_synapse_column(df)
    assert "synapse_label" not in stripped.columns
    assert "synapse_label" in df.columns          # original untouched
    assert len(nc.extract_synapse_frame(stripped, nid=42)) == 0


def test_N7_mislabel_resolution():
    df = nc.classify_frame(_cell())
    n_glia = int((df["compartment_class"] == nc.CLS_GLIA).sum())
    assert n_glia == 3

    # (a) the three astrocyte nodes are ONE subtree, not three decisions
    out, rep = nc.resolve_mislabelled_nodes(
        df, offending_classes=(nc.CLS_GLIA,), policy="knn_relabel", k=3)
    assert rep["n_subtrees"] == 1, rep
    assert rep["subtrees"][0]["n_nodes"] == 3, rep
    assert rep["n_relabelled"] == 3, rep
    assert (out["compartment_class"] == nc.CLS_GLIA).sum() == 0
    # the distance is reported so the user can judge the vote
    assert rep["subtrees"][0]["median_nn_distance_nm"] > 0.0

    # (b) guard: these nodes sit >= 6 um from anything, so a 1 um guard
    #     must refuse to relabel them
    out2, rep2 = nc.resolve_mislabelled_nodes(
        df, policy="knn_relabel", k=3, max_distance_nm=1000.0,
        unresolved_action="drop", offending_classes=(nc.CLS_GLIA,))
    assert rep2["subtrees"][0]["guard_failed"] is True, rep2
    assert rep2["subtrees"][0]["decision"] == "dropped", rep2
    assert len(out2) == len(df) - 3, (len(out2), len(df))

    # (c) drop policy removes the subtree AND anything orphaned below it
    df_deep = df.copy()
    df_deep.loc[len(df_deep)] = (10, 9, 0.0, 9000.0, 0.0, 60.0,
                                 "Dendrite", nc.CLS_DEND)
    out3, rep3 = nc.resolve_mislabelled_nodes(
        df_deep, policy="drop", offending_classes=(nc.CLS_GLIA,))
    assert 10 not in set(out3["id"]), "orphan below a dropped subtree survived"
    assert rep3["n_dropped"] == 4, rep3

    # (d) keep policy is a no-op
    out4, rep4 = nc.resolve_mislabelled_nodes(df, policy="keep")
    assert (out4["compartment_class"] == nc.CLS_GLIA).sum() == 3
    assert rep4["n_relabelled"] == 0


def test_N9_relabel_rewrites_annotation():
    """REGRESSION, found on neuron_794820508.

    build_phi decides shaft membership from 'annotated_type', not from the
    class column. If a relabel moves only the class, the exporter emits a dend
    section that phi never sees: 1760 dend sections against 1759 phi branches.
    A relabelled node must be consistent under BOTH keys.
    """
    df = nc.classify_frame(_cell())
    out, rep = nc.resolve_mislabelled_nodes(
        df, offending_classes=(nc.CLS_GLIA,), policy="knn_relabel", k=3)

    moved = out.loc[df["compartment_class"].values == nc.CLS_GLIA]
    assert len(moved) == 3
    for r in moved.itertuples(index=False):
        # the annotation now classifies back to the assigned class -- the
        # round trip is what keeps phi and the exporter on the same partition
        assert nc.classify_node(r.annotated_type) == r.compartment_class, r
        assert r.annotated_type != "Astrocyte", r

    rx = re.compile(sd.SHAFT_REGEX, re.IGNORECASE)
    for r in moved.itertuples(index=False):
        if r.compartment_class == nc.CLS_DEND:
            assert rx.search(str(r.annotated_type)), (
                "a node relabelled to dend must match SHAFT_REGEX, or phi "
                "will not see it: %r" % (r.annotated_type,))

    # opting out reproduces the defect, which is the point of the flag
    out2, _ = nc.resolve_mislabelled_nodes(
        df, offending_classes=(nc.CLS_GLIA,), policy="knn_relabel", k=3,
        rewrite_annotation=False)
    stale = out2.loc[df["compartment_class"].values == nc.CLS_GLIA]
    assert (stale["annotated_type"] == "Astrocyte").all()


def test_N8_dend_rule_is_shaft_regex():
    """O7 hygiene: the dendrite rule is not a second copy of SHAFT_REGEX."""
    assert any(p is sd.SHAFT_REGEX or p == sd.SHAFT_REGEX
               for p, c in nc.CLASS_RULES if c == nc.CLS_DEND), \
        "the dend rule must BE spine_density.SHAFT_REGEX, not a copy"
    # and the two agree node-for-node on a frame with no spines
    rx = re.compile(sd.SHAFT_REGEX, re.IGNORECASE)
    df = _cell()
    df = df[~df["annotated_type"].isin(["head", "neck", "spine"])]
    cl = nc.classify_frame(df)
    by_regex = {i for i, a in zip(df["id"], df["annotated_type"])
                if rx.search(str(a))}
    by_class = set(cl.loc[cl["compartment_class"] == nc.CLS_DEND, "id"])
    assert by_regex == by_class, (by_regex, by_class)


def test_N10_spine_votes_as_dendrite():
    """A spine is dendritic membrane, so a nearby spine node is evidence of
    dendritic territory and its vote is COUNTED AS 'dend'. What must not happen
    is 'spine' winning: spine membership is topological, not spatial, and a
    mid-cable node voted into 'spine' would be removed by prune_spines together
    with everything below it.
    """
    df = _cell()
    # an astrocyte node sitting right next to the spine head (node 4)
    df.loc[len(df)] = (30, 2, 2000.0, 100.0, 500.0, 90.0, "Astrocyte")
    df = nc.classify_frame(df)

    out, rep = nc.resolve_mislabelled_nodes(df, policy="knn_relabel", k=3)
    st = [t for t in rep["subtrees"] if t["n_nodes"] == 1][0]
    # the spine's evidence is kept, but under the dendrite's name
    assert nc.CLS_SPINE not in st["vote"], st
    assert st["assigned_class"] == nc.CLS_DEND, st
    assert rep["vote_class_alias"] == {nc.CLS_SPINE: nc.CLS_DEND}, rep

    # the relabelled node is now visible to phi as well as to the exporter
    row = out[out["id"] == 30].iloc[0]
    assert nc.classify_node(row.annotated_type) == nc.CLS_DEND, row
    assert re.compile(sd.SHAFT_REGEX, re.IGNORECASE).search(str(row.annotated_type))

    # spine nodes really are in the reference set: disabling the alias lets
    # 'spine' win, which is the pathology the alias removes
    out2, rep2 = nc.resolve_mislabelled_nodes(
        df, policy="knn_relabel", k=3, vote_class_alias={})
    st2 = [t for t in rep2["subtrees"] if t["n_nodes"] == 1][0]
    assert st2["assigned_class"] == nc.CLS_SPINE, st2

    # and the canonical map round-trips for every class it defines
    for cls, token in nc.CANONICAL_ANNOTATION.items():
        assert nc.classify_node(token) == cls, (cls, token)


def test_N11_domain_collapsed_guard():
    """The apical/basal split is retired but its vocabulary is kept.

    section_array_name / SECTION_ARRAY must keep working exactly as N5 checks
    -- this test does not touch them. assert_domain_collapsed is a SEPARATE,
    additional guard: it must accept DOM_NONE and 'dend' (the only values the
    collapsed pipeline ever actually produces) and raise on every value that
    would mean the split is running again, from either value space (a node's
    domain label, or a section's array name).
    """
    # the only values a collapsed pipeline ever produces: silent pass
    assert nc.assert_domain_collapsed([nc.DOM_NONE], "domain") is True
    assert nc.assert_domain_collapsed(["dend", "soma", "axon", "ais"],
                                      "section arrays") is True
    assert nc.assert_domain_collapsed([], "empty") is True

    # domain-label space
    for bad in (nc.DOM_APICAL, nc.DOM_BASAL):
        try:
            nc.assert_domain_collapsed([nc.DOM_NONE, bad], "domain")
        except ValueError as e:
            assert bad in str(e), e
        else:
            raise AssertionError("%r should have raised" % bad)

    # section-array-name space
    for bad in ("apic_dend", "basal_dend"):
        try:
            nc.assert_domain_collapsed(["dend", bad], "section arrays")
        except ValueError as e:
            assert bad in str(e), e
        else:
            raise AssertionError("%r should have raised" % bad)

    # both retired values reported together, sorted, when both are present
    try:
        nc.assert_domain_collapsed([nc.DOM_APICAL, "basal_dend"], "mixed")
    except ValueError as e:
        assert "apical" in str(e) and "basal_dend" in str(e), e
    else:
        raise AssertionError("mixed retired values should have raised")

    # the guard is ADDITIVE: the lookup table itself is untouched (same as N5)
    assert nc.section_array_name(nc.CLS_DEND, nc.DOM_APICAL) == "apic_dend"
    assert nc.section_array_name(nc.CLS_DEND, nc.DOM_BASAL) == "basal_dend"
    assert nc.RETIRED_DOMAIN_ARRAYS == frozenset({"apic_dend", "basal_dend"})
    assert nc.RETIRED_DOMAIN_VALUES == (
        frozenset({nc.DOM_APICAL, nc.DOM_BASAL}) | nc.RETIRED_DOMAIN_ARRAYS)


# --------------------------------------------------------------------------- #
def _run_all():
    tests = [
        ("N1 measured H01 vocabulary", test_N1_vocabulary),
        ("N2 AIS/Astrocyte are not dend", test_N2_ais_astrocyte_not_dend),
        ("N3 O8 regression: no synapse leak", test_N3_o8_regression_synapse_cannot_leak),
        ("N4 I-16 leakage guard", test_N4_i16_leakage_guard),
        ("N5 vocabulary and section arrays", test_N5_vocabulary_and_arrays),
        ("N6 synapse frame roundtrip", test_N6_synapse_frame_roundtrip),
        ("N7 mislabel resolution", test_N7_mislabel_resolution),
        ("N8 dend rule is SHAFT_REGEX", test_N8_dend_rule_is_shaft_regex),
        ("N9 relabel rewrites annotation", test_N9_relabel_rewrites_annotation),
        ("N10 spine votes as dendrite", test_N10_spine_votes_as_dendrite),
        ("N11 domain-collapsed guard", test_N11_domain_collapsed_guard),
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
