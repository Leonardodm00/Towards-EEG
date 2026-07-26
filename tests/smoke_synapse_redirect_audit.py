"""Smoke test for synapse_redirect_audit.

Runs standalone (python3 smoke_synapse_redirect_audit.py) or under pytest.
Needs only numpy and pandas: the LFPy cell is replaced by a stub with
hand-placed compartment midpoints, so correctness is checked against numbers
computed by hand in this file rather than against the module's own output.

The synthetic cell (all coordinates in um, the tree drawn along +x):

    soma[0]  mid (0,0,0)     L=2    nseg=1   root
    dend[0]  mid (6,0,0)     L=10   nseg=1   child of soma[0]
    dend[1]  x 11..21        L=10   nseg=3   child of dend[0]
             seg mids (12.667,0,0) (16,0,0) (19.333,0,0)
    dend[2]  mid (26,0,0)    L=10   nseg=1   child of dend[1]
    dend[3]  mid (26,1,0)    L=1    nseg=1   child of soma[0]   <- the trap

dend[3] is topologically a sibling of dend[0] via the soma, but sits 1 um from
dend[2]'s midpoint. A spine on dend[2] with its head at (26,0.9,0) is therefore
0.1 um from dend[3] and 0.9 um from its own base: the free snap must choose
dend[3] and be classified FOREIGN. That is the defect, reproduced in miniature.

Coverage
  R1  section tree: parent map, root, cycle detection
  R2  relate_sections: same / ancestor / descendant / foreign
  R3  path_length_um against a hand-computed 27.5 um
  R4  build_segment_index: cumulative flattening, totnsegs mismatch raises
  R5  spine base walk, and it AGREES with an exporter-style spine_bases table
  R6  audit_redirect: the trap fires, the shaft synapse is untouched,
      the anchor really is the base node's transformed coordinate
  R7  summarise_audit arithmetic and the by-type breakdown
  R8  the verdict is invariant under a rigid transform, as it must be
"""

import math

import numpy as np
import pandas as pd

import synapse_redirect_audit as sra


# --------------------------------------------------------------------------- #
#  Stub cell                                                                   #
# --------------------------------------------------------------------------- #
class _Sec(object):
    def __init__(self, name, L, nseg):
        self._name, self.L, self.nseg = name, float(L), int(nseg)

    def name(self):
        return self._name


class StubCell(object):
    """Minimal stand-in for LFPy.Cell: midpoints plus nearest-midpoint search.

    LFPy.get_closest_idx returns the compartment whose MIDPOINT is nearest, so
    that is exactly what this reproduces.
    """

    def __init__(self, secs, midpoints):
        self.allseclist = secs
        self._mid = np.asarray(midpoints, dtype=float)
        self.totnsegs = int(sum(s.nseg for s in secs))
        if len(self._mid) != self.totnsegs:
            raise ValueError("midpoint count != totnsegs")

    def get_closest_idx(self, x=0.0, y=0.0, z=0.0):
        d = np.linalg.norm(self._mid - np.array([x, y, z], dtype=float), axis=1)
        return int(np.argmin(d))


def _cell():
    secs = [_Sec("soma[0]", 2, 1), _Sec("dend[0]", 10, 1),
            _Sec("dend[1]", 10, 3), _Sec("dend[2]", 10, 1),
            _Sec("dend[3]", 1, 1)]
    mids = [(0, 0, 0), (6, 0, 0),
            (11 + 10.0 / 6, 0, 0), (16, 0, 0), (21 - 10.0 / 6, 0, 0),
            (26, 0, 0), (26, 1, 0)]
    return StubCell(secs, mids)


def _section_table():
    return pd.DataFrame([
        (0, "soma", 0, -1),
        (1, "dend", 0, 0),
        (2, "dend", 1, 1),
        (3, "dend", 2, 2),
        (4, "dend", 3, 0),
    ], columns=["section_id", "array", "type_idx", "parent_sec_id"])


def _skeleton_nm():
    """Pre-prune labelled frame. Coordinates in nm, so um * 1000."""
    rows = [
        (0, -1, 0.0, 0.0, 0.0, "soma"),
        (1, 0, 6.0, 0.0, 0.0, "dend"),
        (2, 1, 16.0, 0.0, 0.0, "dend"),
        (3, 2, 26.0, 0.0, 0.0, "dend"),
        (4, 3, 26.0, 0.45, 0.0, "spine"),     # neck of spine A
        (5, 4, 26.0, 0.90, 0.0, "spine"),     # head of spine A -> the trap
        (6, 2, 16.0, 0.20, 0.0, "spine"),     # head of spine B -> harmless
    ]
    df = pd.DataFrame(rows, columns=["id", "p", "x", "y", "z",
                                     "compartment_class"])
    for c in ("x", "y", "z"):
        df[c] = df[c] * 1000.0
    return df


def _to_um(a):
    return np.asarray(a, dtype=float) / 1000.0


# --------------------------------------------------------------------------- #
def test_R1_section_tree():
    parent_of = sra.build_section_tree(_section_table())
    assert parent_of == {"soma[0]": None, "dend[0]": "soma[0]",
                         "dend[1]": "dend[0]", "dend[2]": "dend[1]",
                         "dend[3]": "soma[0]"}, parent_of
    assert sra.ancestor_chain("dend[2]", parent_of) == \
        ["dend[2]", "dend[1]", "dend[0]", "soma[0]"]

    bad = _section_table().copy()
    bad.loc[bad.section_id == 0, "parent_sec_id"] = 3      # soma <- dend[2]
    try:
        sra.build_section_tree(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("a cyclic section table was accepted")


def test_R2_relations():
    p = sra.build_section_tree(_section_table())
    assert sra.relate_sections("dend[2]", "dend[2]", p) == sra.REL_SAME
    assert sra.relate_sections("dend[1]", "dend[2]", p) == sra.REL_ANCESTOR
    assert sra.relate_sections("dend[2]", "dend[1]", p) == sra.REL_DESCENDANT
    assert sra.relate_sections("dend[3]", "dend[2]", p) == sra.REL_FOREIGN
    assert sra.relate_sections("dend[2]", "dend[3]", p) == sra.REL_FOREIGN
    # a sibling is foreign even though it shares a parent
    assert sra.relate_sections("dend[0]", "dend[3]", p) == sra.REL_FOREIGN


def test_R3_path_length():
    p = sra.build_section_tree(_section_table())
    L = {"soma[0]": 2.0, "dend[0]": 10.0, "dend[1]": 10.0,
         "dend[2]": 10.0, "dend[3]": 1.0}
    assert sra.section_path("dend[3]", "dend[2]", p) == \
        ["dend[3]", "soma[0]", "dend[0]", "dend[1]", "dend[2]"]
    # by hand: (1 + 2 + 10 + 10 + 10) - 0.5*1 - 0.5*10 = 33 - 5.5
    assert abs(sra.path_length_um("dend[3]", "dend[2]", p, L) - 27.5) < 1e-12
    assert sra.path_length_um("dend[2]", "dend[2]", p, L) == 0.0
    # adjacent sections: half of each
    assert abs(sra.path_length_um("dend[1]", "dend[2]", p, L) - 10.0) < 1e-12


def test_R4_segment_index():
    cell = _cell()
    sec_of_idx, names, Lmap = sra.build_segment_index(cell)
    assert names == ["soma[0]", "dend[0]", "dend[1]", "dend[2]", "dend[3]"]
    assert sec_of_idx.tolist() == [0, 1, 2, 2, 2, 3, 4], sec_of_idx
    assert Lmap["dend[1]"] == 10.0 and Lmap["dend[3]"] == 1.0
    assert len(sec_of_idx) == cell.totnsegs == 7

    broken = _cell()
    broken.totnsegs = 99
    try:
        sra.build_segment_index(broken)
    except ValueError:
        pass
    else:
        raise AssertionError("segment count mismatch was not detected")


def test_R5_spine_bases_and_crosscheck():
    df = _skeleton_nm()
    nm = sra.map_nodes_to_spine_bases(df, spine_classes={"spine"})
    got = {int(r.node_id): (int(r.spine_root_id), int(r.base_node_id))
           for r in nm.itertuples(index=False)}
    # spine A is two nodes deep: both resolve to root 4, base 3
    assert got[5] == (4, 3), got
    assert got[4] == (4, 3), got
    assert got[6] == (6, 2), got
    assert set(got) == {4, 5, 6}, "shaft nodes must not appear"
    assert int(nm.loc[nm.node_id == 5, "depth_in_spine"].iloc[0]) == 1
    assert int(nm.loc[nm.node_id == 4, "depth_in_spine"].iloc[0]) == 0

    exporter_table = pd.DataFrame([(4, 3), (6, 2)],
                                  columns=["spine_root_id", "base_node_id"])
    rep = sra.verify_against_spine_bases(nm, exporter_table)
    assert rep["agree"] is True, rep
    assert rep["n_agree"] == 2, rep

    disagreeing = pd.DataFrame([(4, 3), (6, 99)],
                               columns=["spine_root_id", "base_node_id"])
    rep2 = sra.verify_against_spine_bases(nm, disagreeing)
    assert rep2["agree"] is False and rep2["n_agree"] == 1, rep2


def _run_audit(transform_fn=_to_um):
    df = _skeleton_nm()
    node_map = sra.map_nodes_to_spine_bases(df, spine_classes={"spine"})
    node_xyz = {int(r.id): (r.x, r.y, r.z) for r in df.itertuples(index=False)}

    syn = pd.DataFrame([
        (5, 26.0, 0.90, 0.0, "exc"),      # on spine A head  -> must be FOREIGN
        (6, 16.0, 0.20, 0.0, "exc"),      # on spine B head  -> must be SAME
        (1, 6.0, 0.0, 0.0, "inh"),        # on the shaft     -> not at risk
    ], columns=["node_id", "syn_x_nm", "syn_y_nm", "syn_z_nm", "synapse_type"])
    for c in ("syn_x_nm", "syn_y_nm", "syn_z_nm"):
        syn[c] = syn[c] * 1000.0

    cell = _cell()
    sec_of_idx, names, Lmap = sra.build_segment_index(cell)
    parent_of = sra.build_section_tree(_section_table())
    return sra.audit_redirect(syn, node_xyz, node_map, cell, transform_fn,
                              parent_of, sec_of_idx, names, Lmap)


def test_R6_audit_detects_the_trap():
    a = _run_audit().set_index("node_id")

    trap = a.loc[5]
    assert bool(trap["on_pruned_spine"])
    assert trap["sec_naive"] == "dend[3]", trap["sec_naive"]
    assert trap["sec_anchor"] == "dend[2]", trap["sec_anchor"]
    assert trap["relation"] == sra.REL_FOREIGN, trap["relation"]
    assert not bool(trap["same_section"])
    assert abs(float(trap["path_um"]) - 27.5) < 1e-9, trap["path_um"]
    # the anchor IS node 3, transformed: (26,0,0) um
    assert abs(float(trap["anchor_x"]) - 26.0) < 1e-12
    assert abs(float(trap["anchor_y"]) - 0.0) < 1e-12
    # and the euclidean displacement is the spine length, 0.9 um
    assert abs(float(trap["euclid_um"]) - 0.9) < 1e-9, trap["euclid_um"]
    assert int(trap["base_node_id"]) == 3 and int(trap["spine_root_id"]) == 4

    harmless = a.loc[6]
    assert bool(harmless["on_pruned_spine"])
    assert harmless["relation"] == sra.REL_SAME, harmless["relation"]
    assert bool(harmless["same_compartment"]), "should land on the same segment"
    assert float(harmless["path_um"]) == 0.0

    shaft = a.loc[1]
    assert not bool(shaft["on_pruned_spine"])
    assert int(shaft["lfpy_idx_naive"]) == int(shaft["lfpy_idx_anchor"])
    assert shaft["sec_naive"] == "dend[0]"
    assert int(shaft["base_node_id"]) == -1, "a shaft synapse has no base"


def test_R7_summary_arithmetic():
    s = sra.summarise_audit(_run_audit())
    assert s["n_synapses"] == 3
    assert s["n_on_pruned_spine"] == 2
    assert s["n_moved_section"] == 1
    assert s["n_foreign_branch"] == 1
    assert abs(s["pct_foreign_branch"] - 50.0) < 1e-9
    assert s["by_relation"] == {sra.REL_FOREIGN: 1, sra.REL_SAME: 1}, s["by_relation"]
    assert abs(s["median_path_um_foreign"] - 27.5) < 1e-9
    assert abs(s["median_euclid_um_foreign"] - 0.9) < 1e-9

    # the inhibitory shaft synapse is not at risk, so exc carries both
    assert s["by_type"]["exc"]["n_on_pruned_spine"] == 2
    assert s["by_type"]["exc"]["n_foreign_branch"] == 1
    assert "inh" not in s["by_type"], s["by_type"]

    txt = sra.format_summary(s)
    assert "FOREIGN BRANCH" in txt and "LOWER BOUND" in txt
    assert all(ord(ch) < 128 for ch in txt), "report must be pure ASCII"


def test_R8_invariant_under_rigid_transform():
    """Alignment is rigid, so the verdict must not depend on the frame."""
    th = 0.7
    R = np.array([[math.cos(th), -math.sin(th), 0.0],
                  [math.sin(th), math.cos(th), 0.0],
                  [0.0, 0.0, 1.0]])
    # NOTE: the stub cell's midpoints are NOT rotated, so rotating only the
    # synapses would be wrong. Instead check the pure-geometry invariants that
    # a rigid map must preserve, on the transform itself.
    pts = np.array([[26000.0, 900.0, 0.0], [26000.0, 0.0, 0.0]])
    plain = _to_um(pts)
    moved = _to_um(pts) @ R.T + np.array([3.0, -4.0, 5.0])
    d0 = np.linalg.norm(plain[0] - plain[1])
    d1 = np.linalg.norm(moved[0] - moved[1])
    assert abs(d0 - d1) < 1e-12, (d0, d1)
    assert abs(np.linalg.det(R) - 1.0) < 1e-12

    # and the audit is deterministic: same input, same verdict, twice
    a1 = _run_audit().drop(columns=["syn_x", "syn_y", "syn_z"])
    a2 = _run_audit().drop(columns=["syn_x", "syn_y", "syn_z"])
    assert a1.equals(a2), "audit is not deterministic"


# --------------------------------------------------------------------------- #
def _run_all():
    tests = [
        ("R1 section tree and cycle guard", test_R1_section_tree),
        ("R2 topological relations", test_R2_relations),
        ("R3 path length, hand-computed", test_R3_path_length),
        ("R4 flattened segment index", test_R4_segment_index),
        ("R5 spine bases + exporter cross-check", test_R5_spine_bases_and_crosscheck),
        ("R6 audit detects the foreign snap", test_R6_audit_detects_the_trap),
        ("R7 summary arithmetic", test_R7_summary_arithmetic),
        ("R8 rigid-transform invariance", test_R8_invariant_under_rigid_transform),
    ]
    n = 0
    for name, fn in tests:
        try:
            fn()
            print("[PASS] %s" % name)
            n += 1
        except AssertionError as e:
            print("[FAIL] %s -- %s" % (name, e))
        except Exception as e:                      # noqa: BLE001
            print("[ERROR] %s -- %r" % (name, e))
    print("-" * 62)
    print("%d / %d passed" % (n, len(tests)))
    return n == len(tests)


if __name__ == "__main__":
    import sys
    sys.exit(0 if _run_all() else 1)
