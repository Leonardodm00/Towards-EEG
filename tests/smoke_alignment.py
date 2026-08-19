"""Smoke test for alignment (the merged S1 alignment module).

Runs standalone (python3 smoke_alignment.py) or under pytest. Needs only numpy,
pandas, scipy and the S1 modules; the LFPy cell is a stub, so no NEURON.

Expectations are stated independently of the module wherever possible: the
rigidity tests recompute pairwise distances from first principles, the anchor
test recomputes the transform by hand, and the redirect test uses the same
geometric trap as smoke_synapse_redirect_audit -- a foreign branch physically
nearer the spine head than the head's own base.

Coverage
  A1  metadata loading: literal_eval, det==1 validation, bad bank rejected
  A2  neighbourhood_rotation reproduces Alignment.py L244-251 and yields a
      proper rotation; diagnostics report the spread rather than gating on it
  A3  the transform is RIGID: all pairwise distances preserved exactly
  A4  aligned_nm stays in nm and aligned_um converts EXACTLY ONCE
  A5  align_fn puts the soma at the origin and leaves the frame in nm
  A6  spine_node_to_base agrees with morphology_exporter.prune_spines
  A7  resolve_synapse_anchors: spine synapses get the base coordinate, shaft
      synapses are untouched, and the anchor equals the hand-computed transform
  A8  snap_synapses redirects only the flagged rows, and records the naive index
  A9  write_mapped_synapses emits the C-09 columns and the downstream vocabulary
  A10 regression_check is exact by default and catches a one-ulp move
"""

import json
import math
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

import node_classify as nc
import morphology_exporter as mx
import alignment as al


# --------------------------------------------------------------------------- #
#  Fixtures                                                                    #
# --------------------------------------------------------------------------- #
def _metadata(n=5, seed=0):
    """A small reference bank with known rotations."""
    rng = np.random.default_rng(seed)
    rots = R.from_rotvec(rng.normal(scale=0.25, size=(n, 3)))
    return pd.DataFrame({
        "neuron_id": np.arange(1000, 1000 + n),
        "soma_x": np.linspace(0.0, 4.0e5, n),
        "soma_y": np.zeros(n),
        "soma_z": np.zeros(n),
        "rotation_matrix": [m.tolist() for m in rots.as_matrix()],
        "FA_2D": np.linspace(0.80, 0.95, n),
        "angle_from_mean": np.linspace(1.0, 40.0, n),
    })


def _cell_nm():
    """Soma, a dendrite chain, a two-node spine on the distal node, an axon.
    Coordinates in nm. Mirrors the real H01 vocabulary.
    """
    rows = [
        (0, -1,      0.0,     0.0, 0.0, 5325.5, "Soma"),
        (1,  0,  10000.0,     0.0, 0.0,  300.0, "Dendrite"),
        (2,  1,  20000.0,     0.0, 0.0,  280.0, "Dendrite"),
        (3,  2,  30000.0,     0.0, 0.0,  260.0, "Dendrite"),
        (4,  3,  30000.0,   900.0, 0.0,  100.0, "neck"),
        (5,  4,  30000.0,  1800.0, 0.0,  300.0, "head"),
        (6,  0, -10000.0,     0.0, 0.0,  150.0, "Axon"),
    ]
    return pd.DataFrame(rows, columns=["id", "p", "x", "y", "z", "r",
                                       "annotated_type"])


class StubCell(object):
    """Nearest-midpoint search, which is what LFPy.get_closest_idx does."""

    def __init__(self, midpoints, totnsegs=None):
        self._mid = np.asarray(midpoints, dtype=float)
        self.totnsegs = int(totnsegs if totnsegs is not None else len(self._mid))

    def get_closest_idx(self, x=0.0, y=0.0, z=0.0):
        d = np.linalg.norm(self._mid - np.array([x, y, z], float), axis=1)
        return int(np.argmin(d))


# --------------------------------------------------------------------------- #
def test_A1_metadata_loading():
    tmp = tempfile.mkdtemp()
    try:
        md = _metadata()
        p = os.path.join(tmp, "meta.csv")
        md.assign(rotation_matrix=md["rotation_matrix"].map(repr)).to_csv(
            p, index=False)
        got = al.load_alignment_metadata(p)
        assert isinstance(got["rotation_matrix"].iloc[0], list)
        M = np.array(got["rotation_matrix"].tolist())
        assert M.shape == (5, 3, 3)
        assert np.allclose(np.linalg.det(M), 1.0)

        bad = md.copy()
        bad.loc[0, "rotation_matrix"] = repr((2.0 * np.eye(3)).tolist())
        bad["rotation_matrix"] = bad["rotation_matrix"].map(
            lambda v: v if isinstance(v, str) else repr(v))
        pb = os.path.join(tmp, "bad.csv")
        bad.to_csv(pb, index=False)
        try:
            al.load_alignment_metadata(pb)
        except ValueError:
            pass
        else:
            raise AssertionError("a non-rotation reference matrix was accepted")

        missing = md.drop(columns=["soma_z"])
        pm = os.path.join(tmp, "missing.csv")
        missing.assign(rotation_matrix=missing["rotation_matrix"].map(repr)) \
               .to_csv(pm, index=False)
        try:
            al.load_alignment_metadata(pm)
        except ValueError:
            pass
        else:
            raise AssertionError("a bank missing soma_z was accepted")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_A2_neighbourhood_rotation():
    md = _metadata()
    soma = np.array([0.0, 0.0, 0.0])
    M, diag = al.neighbourhood_rotation(soma, md, k_neighbors=3)

    # recomputed independently, exactly as Alignment.py L244-251 does it
    d = np.linalg.norm(md[["soma_x", "soma_y", "soma_z"]].to_numpy(float) - soma,
                       axis=1)
    near = np.argsort(d)[:3]
    expect = R.from_matrix(
        np.array(md.iloc[near]["rotation_matrix"].tolist())).mean().as_matrix()
    assert np.allclose(M, expect, atol=0, rtol=0), "not the scipy group mean"

    assert abs(np.linalg.det(M) - 1.0) < 1e-12
    assert np.abs(M @ M.T - np.eye(3)).max() < 1e-12
    assert diag["k_actual"] == 3 and diag["k_requested"] == 3
    assert diag["neighbour_nids"] == [1000, 1001, 1002], diag["neighbour_nids"]
    assert diag["neighbour_distance_um"][0] == 0.0
    assert diag["pairwise_angle_deg_max"] > 0.0, "spread must be reported"

    # k larger than the bank clamps rather than failing
    M2, diag2 = al.neighbourhood_rotation(soma, md, k_neighbors=99)
    assert diag2["k_actual"] == len(md)

    # an elementwise mean would NOT be a rotation -- confirm they differ, i.e.
    # the "simplification" section 3.1 forbids is genuinely a different answer
    naive = np.array(md.iloc[near]["rotation_matrix"].tolist()).mean(axis=0)
    assert abs(np.linalg.det(naive) - 1.0) > 1e-6
    assert not np.allclose(M, naive, atol=1e-9)


def test_A3_transform_is_rigid():
    md = _metadata()
    df = _cell_nm()
    soma = al.soma_position_nm(df)
    assert np.allclose(soma, [0.0, 0.0, 0.0])
    M, _ = al.neighbourhood_rotation(soma, md, 3)

    P = df[["x", "y", "z"]].to_numpy(float)
    Q = al.aligned_nm(P, soma, M)

    for i in range(len(P)):
        for j in range(i + 1, len(P)):
            a = np.linalg.norm(P[i] - P[j])
            b = np.linalg.norm(Q[i] - Q[j])
            assert abs(a - b) < 1e-6, (i, j, a, b)

    # a shifted soma changes the offset but not the distances
    df2 = df.copy()
    for c, off in zip("xyz", (2.7e6, 5.3e5, 5.7e4)):
        df2[c] = df2[c] + off
    Q2 = al.aligned_nm(df2[["x", "y", "z"]].to_numpy(float),
                       al.soma_position_nm(df2), M)
    assert np.allclose(Q, Q2, atol=1e-6), "translation leaked into the result"


def test_A4_units_converted_exactly_once():
    md = _metadata()
    df = _cell_nm()
    soma = al.soma_position_nm(df)
    M, _ = al.neighbourhood_rotation(soma, md, 3)
    P = df[["x", "y", "z"]].to_numpy(float)

    nm = al.aligned_nm(P, soma, M)
    um = al.aligned_um(P, soma, M)
    assert np.allclose(um, nm / 1000.0, atol=0, rtol=0)
    # the dendrite tip is 30 um from the soma in nm space; it must still be
    # 30 um after alignment, and 30000 in the nm form -- NOT 0.03
    tip = np.linalg.norm(um[3])
    assert abs(tip - 30.0) < 1e-9, tip
    assert abs(np.linalg.norm(nm[3]) - 30000.0) < 1e-6

    # a single point works too
    one = al.aligned_um(P[3], soma, M)
    assert one.shape == (1, 3) and np.allclose(one[0], um[3])


def test_A5_align_fn_contract():
    md = _metadata()
    df = _cell_nm()
    soma = al.soma_position_nm(df)
    M, _ = al.neighbourhood_rotation(soma, md, 3)
    fn = al.make_align_fn(soma, M)
    out = fn(df, "T1")

    assert list(out.columns) == list(df.columns), "columns changed"
    assert len(out) == len(df)
    root = out[out["p"] == -1].iloc[0]
    assert abs(root.x) < 1e-9 and abs(root.y) < 1e-9 and abs(root.z) < 1e-9
    # STILL nm: the tip is 30000, not 30
    assert abs(float(np.linalg.norm(out.loc[3, ["x", "y", "z"]].to_numpy(float)))
               - 30000.0) < 1e-6
    assert not out.equals(df), "align_fn was a no-op"
    assert abs(float(df.loc[3, "x"]) - 30000.0) < 1e-9, "input frame mutated"


def test_A6_spine_base_agrees_with_prune_spines():
    df = nc.classify_frame(_cell_nm())
    mine = al.spine_node_to_base(df)
    assert set(mine) == {4, 5}, mine
    assert mine[5] == (4, 3), mine
    assert mine[4] == (4, 3), mine

    _, info = mx.prune_spines(df)
    theirs = {(b["spine_root_id"], b["base_node_id"])
              for b in info["spine_bases"]}
    assert theirs == {(4, 3)}, theirs
    assert set(mine.values()) == theirs, (mine, theirs)
    assert info["n_spines"] == 1 and info["n_spine_nodes"] == 2


def test_A7_anchor_resolution():
    md = _metadata()
    df = nc.classify_frame(_cell_nm())
    soma = al.soma_position_nm(df)
    M, _ = al.neighbourhood_rotation(soma, md, 3)

    syn = pd.DataFrame({
        "node_id": [5, 1, 4],
        "syn_x_nm": [30000.0, 10000.0, 30000.0],
        "syn_y_nm": [1800.0, 0.0, 900.0],
        "syn_z_nm": [0.0, 0.0, 0.0],
        "synapse_label": ["exc_syn", "inh_syn", "exc_syn"],
    })
    a = al.resolve_synapse_anchors(syn, df, soma, M).set_index("node_id")

    assert bool(a.loc[5, "on_pruned_spine"]) and bool(a.loc[4, "on_pruned_spine"])
    assert not bool(a.loc[1, "on_pruned_spine"]), "a shaft synapse was flagged"
    assert int(a.loc[5, "base_node_id"]) == 3
    assert int(a.loc[5, "spine_root_id"]) == 4
    assert int(a.loc[1, "base_node_id"]) == -1

    # the anchor is node 3's coordinate through the SAME transform, by hand
    expect = al.aligned_um(np.array([[30000.0, 0.0, 0.0]]), soma, M)[0]
    for k, v in zip("xyz", expect):
        assert abs(float(a.loc[5, "anchor_" + k]) - v) < 1e-12, k
    # the shaft synapse anchors to itself
    for k in "xyz":
        assert abs(float(a.loc[1, "anchor_" + k]) - float(a.loc[1, k])) < 1e-12

    # aligned coordinates are um: the head sits ~30.05 um out
    assert abs(float(np.linalg.norm(a.loc[5, ["x", "y", "z"]].to_numpy(float)))
               - math.hypot(30.0, 1.8)) < 1e-9


def test_A8_snap_redirects_only_the_flagged():
    md = _metadata()
    df = nc.classify_frame(_cell_nm())
    soma = al.soma_position_nm(df)
    M, _ = al.neighbourhood_rotation(soma, md, 3)

    syn = pd.DataFrame({
        "node_id": [5, 1],
        "syn_x_nm": [30000.0, 10000.0],
        "syn_y_nm": [1800.0, 0.0],
        "syn_z_nm": [0.0, 0.0],
        "synapse_label": ["exc_syn", "inh_syn"],
    })
    a = al.resolve_synapse_anchors(syn, df, soma, M)

    # compartments: index 0 at the head's aligned position (the trap), index 1
    # at the base's, index 2 at the shaft synapse's
    head = a.loc[0, ["x", "y", "z"]].to_numpy(float)
    base = a.loc[0, ["anchor_x", "anchor_y", "anchor_z"]].to_numpy(float)
    shaft = a.loc[1, ["x", "y", "z"]].to_numpy(float)
    cell = StubCell([head, base, shaft])

    s = al.snap_synapses(a, cell)
    assert int(s.loc[0, "lfpy_idx_naive"]) == 0, "the naive snap must hit the trap"
    assert int(s.loc[0, "lfpy_idx"]) == 1, "the redirect must hit the base"
    assert bool(s.loc[0, "redirected"])
    assert int(s.loc[1, "lfpy_idx"]) == int(s.loc[1, "lfpy_idx_naive"]) == 2
    assert not bool(s.loc[1, "redirected"]), "a shaft synapse must not move"
    assert int(s["redirected"].sum()) == 1


def test_A9_c09_emission():
    md = _metadata()
    df = nc.classify_frame(_cell_nm())
    soma = al.soma_position_nm(df)
    M, _ = al.neighbourhood_rotation(soma, md, 3)
    syn = pd.DataFrame({
        "node_id": [5, 1],
        "syn_x_nm": [30000.0, 10000.0],
        "syn_y_nm": [1800.0, 0.0],
        "syn_z_nm": [0.0, 0.0],
        "synapse_label": ["exc_syn", "inh_syn"],
    })
    a = al.resolve_synapse_anchors(syn, df, soma, M)
    cell = StubCell([a.loc[0, ["x", "y", "z"]].to_numpy(float),
                     a.loc[0, ["anchor_x", "anchor_y", "anchor_z"]].to_numpy(float),
                     a.loc[1, ["x", "y", "z"]].to_numpy(float)])
    s = al.snap_synapses(a, cell)

    tmp = tempfile.mkdtemp()
    try:
        p = os.path.join(tmp, "mapped.csv")
        rep = al.write_mapped_synapses(s, p, cm=0.5, Ra=268.5, lambda_f=100.0,
                                       d_lambda=0.1,
                                       spine_base_section={3: "dend[2]"})
        assert rep["n_rows"] == 2 and rep["n_redirected"] == 1
        got = pd.read_csv(p)

        for c in ("x", "y", "z", "synapse_label", "synapse_type", "lfpy_idx",
                  "spine_base_section", "lambda_f", "nsegs_method",
                  "cm", "Ra", "d_lambda", "on_pruned_spine",
                  "anchor_x", "anchor_y", "anchor_z", "base_node_id"):
            assert c in got.columns, c
        assert list(got.columns)[:6] == al.C09_COLUMNS[:6], list(got.columns)[:6]

        # BOTH vocabularies, and the downstream one is what the consumer filters
        assert set(got["synapse_label"]) == {"exc_syn", "inh_syn"}
        assert set(got["synapse_type"]) == {"exc", "inh"}
        assert all(v in ("exc", "inh", "unknown") for v in got["synapse_type"])

        assert (got["cm"] == 0.5).all() and (got["Ra"] == 268.5).all()
        assert (got["lambda_f"] == 100.0).all()
        assert (got["d_lambda"] == 0.1).all()
        assert (got["nsegs_method"] == "lambda_f").all()

        sbs = got.loc[got["base_node_id"] == 3, "spine_base_section"].iloc[0]
        assert sbs == "dend[2]", sbs
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    assert al.to_downstream_type("exc_syn") == "exc"
    assert al.to_downstream_type("inh_syn") == "inh"
    assert al.to_downstream_type("Type 2 asymmetric") == "exc"
    assert al.to_downstream_type("banana") == "unknown"


def test_A11_orientation_convention():
    """The transform must apply mean_matrix in the sense that sends v_com to +z.

    This is NOT covered by the rigidity tests: writing `coords @ M` instead of
    `coords @ M.T` applies the INVERSE rotation, which is still a perfectly
    rigid transform, so every distance is preserved and every regression
    quantity is bit-identical -- while the cell is oriented wrong. It would
    silently break D-4, which defines apical as the subtree maximising z-extent
    AFTER alignment.

    The convention is fixed by the metadata itself and verified on the real
    delivered bank: for every reference row, M @ v_com == [0, 0, 1] to machine
    precision, whereas M.T @ v_com points essentially the opposite way. For an
    Nx3 array of row vectors the equivalent is `coords @ M.T`, which is what
    Alignment.py L254 does and what aligned_nm must reproduce.
    """
    u = np.array([1.0, 1.0, 1.0]) / math.sqrt(3.0)
    rot, _ = R.align_vectors([[0.0, 0.0, 1.0]], [u])
    M = rot.as_matrix()
    assert np.allclose(M @ u, [0.0, 0.0, 1.0], atol=1e-12), "fixture is wrong"

    md = pd.DataFrame({
        "neuron_id": [1],
        "soma_x": [0.0], "soma_y": [0.0], "soma_z": [0.0],
        "v_com_x": [u[0]], "v_com_y": [u[1]], "v_com_z": [u[2]],
        "rotation_matrix": [M.tolist()],
    })
    df = pd.DataFrame([
        (0, -1, 0.0, 0.0, 0.0, 5325.5, "Soma"),
        (1, 0, 30000.0 * u[0], 30000.0 * u[1], 30000.0 * u[2], 300.0, "Dendrite"),
    ], columns=["id", "p", "x", "y", "z", "r", "annotated_type"])

    soma = al.soma_position_nm(df)
    mean_matrix, _ = al.neighbourhood_rotation(soma, md, k_neighbors=1)
    assert np.allclose(mean_matrix, M, atol=1e-12)

    out = al.make_align_fn(soma, mean_matrix)(df, "T")
    tip = out.loc[1, ["x", "y", "z"]].to_numpy(float)
    assert np.allclose(tip, [0.0, 0.0, 30000.0], atol=1e-6), (
        "the dendrite must land on +z; got %r. Using `coords @ M` instead of "
        "`coords @ M.T` produces exactly this failure." % (tip,))
    assert tip[2] > 0, "arbour is upside down -- the inverse rotation was applied"

    # and the same convention in the um form used for synapses and anchors
    tip_um = al.aligned_um(df.loc[[1], ["x", "y", "z"]].to_numpy(float),
                           soma, mean_matrix)[0]
    assert np.allclose(tip_um, [0.0, 0.0, 30.0], atol=1e-9), tip_um


def test_A10_regression_check_is_exact():
    base = {"qc_status": "pass", "n_sections": 1785, "n_branches": 1760,
            "n_sections_multi_branch": 0, "n_spines_unattached": 0,
            "f_implied": 1.5285241084552337, "F_lit": 1.8948,
            "A_shaft_um2": 20360.622221659312,
            "A_spine_um2": 10761.079707296305}
    assert al.regression_check(base, dict(base))["identical"]

    moved = dict(base)
    moved["A_shaft_um2"] = np.nextafter(base["A_shaft_um2"], np.inf)
    rep = al.regression_check(base, moved)
    assert not rep["identical"], "a one-ulp move slipped through"
    assert "A_shaft_um2" in rep["diffs"]

    other = dict(base)
    other["n_sections"] = 1786
    assert "n_sections" in al.regression_check(base, other)["diffs"]


# --------------------------------------------------------------------------- #
def _run_all():
    tests = [
        ("A1  metadata loading and validation", test_A1_metadata_loading),
        ("A2  neighbourhood rotation is the group mean", test_A2_neighbourhood_rotation),
        ("A3  transform is rigid", test_A3_transform_is_rigid),
        ("A4  units converted exactly once", test_A4_units_converted_exactly_once),
        ("A5  align_fn contract (origin, nm, no mutation)", test_A5_align_fn_contract),
        ("A6  spine base agrees with prune_spines", test_A6_spine_base_agrees_with_prune_spines),
        ("A7  anchor resolution", test_A7_anchor_resolution),
        ("A8  snap redirects only the flagged", test_A8_snap_redirects_only_the_flagged),
        ("A9  C-09 emission and vocabulary", test_A9_c09_emission),
        ("A10 regression check is exact", test_A10_regression_check_is_exact),
        ("A11 orientation convention (v_com -> +z)", test_A11_orientation_convention),
    ]
    n = 0
    for name, fn in tests:
        try:
            fn()
            print("[PASS] %s" % name)
            n += 1
        except AssertionError as e:
            print("[FAIL] %s -- %s" % (name, e))
        except Exception as e:                          # noqa: BLE001
            print("[ERROR] %s -- %r" % (name, e))
    print("-" * 62)
    print("%d / %d passed" % (n, len(tests)))
    return n == len(tests)


if __name__ == "__main__":
    import sys
    sys.exit(0 if _run_all() else 1)
