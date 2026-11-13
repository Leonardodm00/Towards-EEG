"""Spec-derived tests, Block 23 (the features table) -- tester 2026-11-13.

Oracles from SPEC.md Block 23: round 0 = Allen's radii doubled and the trace
[exact]; CSV round trip bit for bit; featured_scene with zero offsets and d =
2 r reproduces the scene [exact]; an offset moves exactly the ends that carry
its id; with_rows leaves its input unchanged; max_relative_change against a
hand loop [1e-15]; the ValueErrors listed. Boundary with Block 13/8:
initial_features' nodes = cell_scene's segment children for the same types,
so featured_scene never meets an unknown child.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_allclose

import nc_helpers  # noqa: F401  (paths)
from allen_diameter.analysis import node_features as NF
from allen_diameter.analysis.cell import cell_scene
from allen_diameter.loading import features_io as FIO
from allen_diameter.loading.swc_io import read_swc

SWC_TEXT = """# test cell
1 1 10.0 10.0 5.0 6.0 -1
2 3 20.0 10.0 5.0 1.0 1
3 3 25.0 12.0 5.5 0.9 2
4 3 30.0 15.0 6.0 0.7 3
5 3 28.0 5.0 5.0 0.6 3
6 4 10.0 25.0 4.0 1.5 1
7 4 10.0 32.0 4.2 1.2 6
8 2 3.0 10.0 5.0 0.4 1
9 2 -4.0 11.0 5.2 0.3 8
10 3 50.0 50.0 1.0 0.5 99
"""


@pytest.fixture()
def swc(tmp_path):
    p = tmp_path / "cell.swc"
    p.write_text(SWC_TEXT)
    return read_swc(str(p))


def test_initial_features_round0(swc):
    f = NF.initial_features(swc, (2, 3, 4), 0.7)
    # hand: nodes of types 2,3,4 with a parent in the file (10's parent 99 absent)
    exp_ids = [2, 3, 4, 5, 6, 7, 8, 9]
    rad = {2: 1.0, 3: 0.9, 4: 0.7, 5: 0.6, 6: 1.5, 7: 1.2, 8: 0.4, 9: 0.3}
    typ = {2: 3, 3: 3, 4: 3, 5: 3, 6: 4, 7: 4, 8: 2, 9: 2}
    assert_array_equal(f.node_id, exp_ids)
    assert f.node_id.dtype == np.int64 and f.swc_type.dtype == np.int64
    assert_array_equal(f.d_um, [2.0 * rad[i] for i in exp_ids])          # exact doubling
    assert_array_equal(f.swc_type, [typ[i] for i in exp_ids])
    assert_array_equal(f.mu_per_um, 0.7)
    assert_array_equal(f.offset_um, np.zeros((8, 3)))
    assert set(f.source.tolist()) == {"allen"} and f.round_index == 0


def test_boundary_scene_children_equal_feature_nodes(swc):
    # Block 23 consumes Block 13's Scene built by cell_scene with the same types
    f = NF.initial_features(swc, (2, 3, 4), 0.5)
    sc = cell_scene(swc, (2, 3, 4), 0.1144)
    assert sorted(sc.seg_ids[:, 0].tolist()) == sorted(f.node_id.tolist())
    fs = NF.featured_scene(sc, f)                     # must not raise
    assert fs.seg_xyz.shape == sc.seg_xyz.shape


def test_featured_scene_identity_and_offsets(swc):
    f = NF.initial_features(swc, (2, 3, 4), 0.5)
    sc = cell_scene(swc, (2, 3, 4), 0.1144)
    fs = NF.featured_scene(sc, f)
    assert_array_equal(fs.seg_xyz, sc.seg_xyz)               # exact
    assert_array_equal(fs.seg_d_um, 2.0 * sc.seg_radius)     # exact
    assert_array_equal(fs.seg_ids, sc.seg_ids)
    # root segments (parent = soma, not in features): parent end takes the child's mu
    f2 = NF.with_rows(f, [{"node_id": 2, "mu_per_um": 1.3, "d_um": 5.0}], 1)
    fs2 = NF.featured_scene(sc, f2)
    q = int(np.flatnonzero(sc.seg_ids[:, 0] == 2)[0])
    assert fs2.seg_mu_per_um[q, 0] == 1.3 and fs2.seg_mu_per_um[q, 1] == 1.3
    assert fs2.seg_d_um[q, 0] == 2.0 * sc.seg_radius[q, 0]   # keeps Allen's radius at the soma end
    assert fs2.seg_d_um[q, 1] == 5.0
    # offset of node 3 moves exactly the ends carrying id 3
    off = np.array([0.25, -0.5, 0.125])
    f3 = NF.with_rows(f, [{"node_id": 3, "offset_um": off}], 1)
    fs3 = NF.featured_scene(sc, f3)
    for qq in range(sc.seg_ids.shape[0]):
        child, parent = sc.seg_ids[qq]
        e1 = sc.seg_xyz[qq, 3:] + (off if child == 3 else 0.0)
        e0 = sc.seg_xyz[qq, :3] + (off if parent == 3 else 0.0)
        assert_array_equal(fs3.seg_xyz[qq, 3:], e1)
        assert_array_equal(fs3.seg_xyz[qq, :3], e0)


def test_with_rows_no_mutation(swc):
    f = NF.initial_features(swc, (2, 3, 4), 0.5)
    snap = {k: np.array(getattr(f, k), copy=True) for k in ("node_id", "d_um", "mu_per_um", "offset_um", "source", "flags")}
    g = NF.with_rows(f, [{"node_id": 4, "d_um": 3.3, "source": "refined", "flags": "x;y"}], 2)
    for k, v in snap.items():
        assert_array_equal(getattr(f, k), v)
    assert g.round_index == 2 and f.round_index == 0
    k4 = int(np.flatnonzero(g.node_id == 4)[0])
    others = np.flatnonzero(g.node_id != 4)
    assert g.d_um[k4] == 3.3 and g.source[k4] == "refined" and g.flags[k4] == "x;y"
    assert_array_equal(g.d_um[others], f.d_um[others])
    assert_array_equal(g.offset_um, f.offset_um)


def test_csv_round_trip_bitwise(swc, tmp_path):
    f = NF.initial_features(swc, (2, 3, 4), 0.5)
    n = len(f)
    rng = np.random.default_rng(23)
    rows = []
    weird = [1.0 / 3.0, 5e-324, 1e300, 0.1 + 0.2, math.pi, 2.0 ** -1074 * 3, 7.0, 1e-17]
    for k, nid in enumerate(f.node_id):
        rows.append({"node_id": int(nid), "d_um": weird[k % len(weird)] if weird[k % len(weird)] > 0 else 1.0,
                     "mu_per_um": float(rng.random()), "offset_um": rng.normal(size=3),
                     "source": NF.SOURCES[k % 4], "flags": ["", "a;b", 'q"uote,comma', "x\ny"][k % 4]})
    g = NF.with_rows(f, rows, 3)
    p = tmp_path / "f.csv"
    FIO.write_features(str(p), g)
    h = FIO.read_features(str(p))
    for k in ("node_id", "swc_type", "d_um", "mu_per_um", "offset_um"):
        a, b = getattr(g, k), getattr(h, k)
        assert a.dtype == b.dtype
        assert_array_equal(a.view(np.uint8) if a.dtype.kind == "f" else a, b.view(np.uint8) if b.dtype.kind == "f" else b)
    assert h.source.tolist() == g.source.tolist() and h.flags.tolist() == g.flags.tolist()
    assert h.round_index == 3 and len(h) == n


def test_max_relative_change_hand_loop(swc):
    f = NF.initial_features(swc, (2, 3, 4), 0.5)
    rng = np.random.default_rng(5)
    rows = [{"node_id": int(i), "d_um": float(d * (1 + 0.1 * rng.normal())), "mu_per_um": float(rng.random() + 0.1),
             "offset_um": rng.normal(size=3) * 0.1} for i, d in zip(f.node_id, f.d_um)]
    g = NF.with_rows(f, rows, 1)
    # permuted copy of g (matched by node_id)
    perm = rng.permutation(len(g))
    gp = NF.NodeFeatures(g.node_id[perm], g.swc_type[perm], g.d_um[perm], g.mu_per_um[perm], g.offset_um[perm],
                         g.source[perm], g.flags[perm], 1)
    out = NF.max_relative_change(f, gp)
    rd = rmu = ro = 0.0
    for k in range(len(f)):
        j = int(np.flatnonzero(gp.node_id == f.node_id[k])[0])
        rd = max(rd, abs(gp.d_um[j] - f.d_um[k]) / f.d_um[k])
        rmu = max(rmu, abs(gp.mu_per_um[j] - f.mu_per_um[k]) / f.mu_per_um[k])
        ro = max(ro, math.sqrt(sum((gp.offset_um[j, i] - f.offset_um[k, i]) ** 2 for i in range(3))))
    assert_allclose([out["d"], out["mu"], out["offset_um"]], [rd, rmu, ro], rtol=1e-15, atol=0)


def test_value_errors(swc):
    f = NF.initial_features(swc, (2, 3, 4), 0.5)
    n = len(f)
    base = dict(node_id=f.node_id, swc_type=f.swc_type, d_um=f.d_um, mu_per_um=f.mu_per_um,
                offset_um=f.offset_um, source=f.source, flags=f.flags)

    def mk(**kw):
        d = dict(base)
        d.update(kw)
        return NF.NodeFeatures(**d)

    bad_d = f.d_um.copy(); bad_d[0] = np.nan
    with pytest.raises(ValueError):
        mk(d_um=bad_d)
    bad_d = f.d_um.copy(); bad_d[0] = 0.0
    with pytest.raises(ValueError):
        mk(d_um=bad_d)
    bad_mu = f.mu_per_um.copy(); bad_mu[1] = -1e-12
    with pytest.raises(ValueError):
        mk(mu_per_um=bad_mu)
    bad_off = f.offset_um.copy(); bad_off[2, 1] = np.inf
    with pytest.raises(ValueError):
        mk(offset_um=bad_off)
    with pytest.raises(ValueError):
        mk(source=np.array(["allen"] * (n - 1) + ["bogus"]))
    with pytest.raises(ValueError):
        mk(d_um=f.d_um[:-1])
    with pytest.raises(ValueError):
        NF.with_rows(f, [{"node_id": 12345, "d_um": 1.0}], 1)
    with pytest.raises(ValueError):
        NF.with_rows(f, [{"node_id": 2, "d_um": float("nan")}], 1)
