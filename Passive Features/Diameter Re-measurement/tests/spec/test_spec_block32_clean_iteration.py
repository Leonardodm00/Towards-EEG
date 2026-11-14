"""Spec-derived tests of Block 32 (the cleaning iteration; D-130.O6, D-136.2, D-139) -- integration tester,
2026-11-14. Oracles from specs/SPEC.md Block 32 and its amendment 1. The loop is tested with a scripted
refine_node (monkeypatched in clean_iteration's namespace) so that every number the loop must produce is
known by hand; the cleaning and the profile read are checked against nc_helpers' independent bilinear read.

Run: python3 -m pytest -q tests/spec/test_spec_block32_clean_iteration.py
Pure ASCII.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import nc_helpers
from allen_diameter.config import default_config
from allen_diameter.analysis import clean_iteration as CI
from allen_diameter.analysis.node_features import NodeFeatures
from allen_diameter.analysis.plane_iteration import Scene

P = 0.1144


def scene_two():
    """Two parallel flat chains along x, 3 um apart in y: ids 1-2-3 (y = 1.0) and 11-12-13 (y = 4.0);
    segments child -> parent: (2, 1), (3, 2), (12, 11), (13, 12)."""
    xyz = np.array([[0.0, 1.0, 0.5, 1.5, 1.0, 0.5], [1.5, 1.0, 0.5, 3.0, 1.0, 0.5],
                    [0.0, 4.0, 0.5, 1.5, 4.0, 0.5], [1.5, 4.0, 0.5, 3.0, 4.0, 0.5]])
    return Scene(seg_xyz=xyz, seg_radius=np.full((4, 2), 0.3), seg_ids=np.array([[2, 1], [3, 2], [12, 11], [13, 12]]),
                 soma=np.zeros((0, 3)))


def features0():
    ids = np.array([1, 2, 3, 11, 12, 13])
    n = ids.size
    return NodeFeatures(node_id=ids, swc_type=np.full(n, 3), d_um=np.full(n, 0.6), mu_per_um=np.full(n, 1.0),
                        offset_um=np.zeros((n, 3)), source=np.array(["allen"] * n), flags=np.array([""] * n),
                        round_index=0)


def node(nid, y0, theta0=None, seed=0):
    rng = np.random.default_rng(seed)
    ks = np.array([1, 2, 3])
    H, W = 80, 30
    blk = 200.0 + rng.normal(0, 1.0, size=(3, H, W))
    v = (np.arange(21) - 10) * P
    return CI.NodeData(node_id=nid, own_ids=(nid - 1, nid, nid + 1), node_xyz=np.array([1.5, y0, 0.5]), block=blk,
                       B_gl=np.array([205.0, 210.0, 207.0]), ks=ks, planes_z=ks * 0.28, left=0, top=-20,
                       origin_xy=np.array([1.5, y0]), y_hat=np.array([0.0, 1.0]), e_u=np.array([1.0, 0.0]), v=v,
                       n_avg=1, step=P, theta0=theta0)


class Script:
    """A scripted refine_node: per node id a list of results by round; records each call's start."""

    def __init__(self, plan):
        self.plan, self.calls = plan, []

    def __call__(self, prof, B, v, predict, start, cfg):
        nid = predict.node_id
        r = sum(1 for c in self.calls if c[0] == nid)
        self.calls.append((nid, np.array(start, dtype=float), np.array(prof, dtype=float)))
        d, mu, v0, dz, status = self.plan[nid][min(r, len(self.plan[nid]) - 1)]
        return {"d_um": d, "mu_per_um": mu, "v0_um": v0, "dz_um": dz, "cost": 0.0, "rms_gl": 0.0,
                "status": status, "at_bound": ("d_um",) if status == "at_bound" else (), "nfev": 1}


def factory(node_):
    def predict(d, mu, v0, dz):
        return np.zeros((len(node_.ks), len(node_.v)))
    predict.node_id = node_.node_id
    return predict


def zero_shadow(fscene, own_ids, left, top, width, height, planes_z, cfg, B_gl, include):
    return np.zeros((len(planes_z), height, width))


def cfg_rounds(max_rounds=4, stop_rel_d=0.01):
    cfg = default_config()
    return dataclasses.replace(cfg, cleaning=dataclasses.replace(cfg.cleaning, max_rounds=max_rounds,
                                                                stop_rel_d=stop_rel_d))


def test_loop_by_hand(monkeypatch):
    """Block 32 (f), the rows and the stop rule, all by hand:
    round 1 starts at theta0 (node 2) and at (d0, mu0, 0, 0) where theta0 is None (node 12); from round 2 a
    refined node starts at (d, mu) of its row and (v0, dz) read back from offset_um (amendment 1 (iii));
    offset_um = offset_xyz(v0, dz, y_hat) = (v0 y_x, v0 y_y, dz); source 'refined'; round_index t;
    failed: row unchanged, flag refine_failed, counted; at_bound: row updated, flag refine_at_bound;
    max_rel_d = max over the refined nodes of |d_t - d_{t-1}| / d_{t-1}; stop when max_rel_d < stop_rel_d
    and the example line change < sigma_gl."""
    plan = {2: [(0.70, 1.2, 0.10, -0.2, "converged"), (0.703, 1.2, 0.11, -0.2, "converged"),
                (0.7031, 1.2, 0.11, -0.2, "converged")],
            12: [(0.0, 0.0, 0.0, 0.0, "failed"), (0.50, 0.8, -0.3, 0.1, "at_bound"),
                 (0.5001, 0.8, -0.3, 0.1, "converged")]}
    sc = Script(plan)
    monkeypatch.setattr(CI.coarse_fit, "refine_node", sc)
    nodes = [node(2, 1.0, theta0=np.array([0.65, 1.1, 0.05, 0.02])), node(12, 4.0)]
    cfg = cfg_rounds(4, 0.01)
    res = CI.run_rounds(features0(), scene_two(), nodes, cfg, 3.0, factory, example_ids=(2,), shadow_fn=zero_shadow)
    starts = {(nid, k): s for k, (nid, s, _p) in enumerate(sc.calls)}
    calls2 = [c for c in sc.calls if c[0] == 2]
    calls12 = [c for c in sc.calls if c[0] == 12]
    assert_allclose(calls2[0][1], [0.65, 1.1, 0.05, 0.02], rtol=0, atol=0)
    assert_allclose(calls12[0][1], [0.6, 1.0, 0.0, 0.0], rtol=0, atol=0)
    assert_allclose(calls2[1][1], [0.70, 1.2, 0.10, -0.2], rtol=0, atol=1e-15)
    assert_allclose(calls12[1][1], [0.6, 1.0, 0.0, 0.0], rtol=0, atol=0)      # failed in round 1: not refined
    assert_allclose(calls12[2][1], [0.50, 0.8, -0.3, 0.1], rtol=0, atol=1e-15)
    # round 1: rel d = |0.70 - 0.6| / 0.6; round 2: max(|0.703 - 0.70|/0.70, |0.50 - 0.6|/0.6);
    # round 3: max(1e-4/0.703, 1e-4/0.5) = 2e-4 < 0.01 and the line change 0 (zero shadow) -> converged
    exp = [abs(0.70 - 0.6) / 0.6, max(0.003 / 0.70, 0.1 / 0.6), max(0.0001 / 0.703, 0.0001 / 0.5)]
    assert [h["round"] for h in res.history] == [1, 2, 3]
    assert_allclose([h["max_rel_d"] for h in res.history], exp, rtol=1e-12)
    assert [h["n_refined"] for h in res.history] == [1, 2, 2]
    assert [h["n_failed"] for h in res.history] == [1, 0, 0]
    assert [h["n_at_bound"] for h in res.history] == [0, 1, 0]
    assert all(h["max_line_change_gl"] == 0.0 for h in res.history)
    assert res.stopped == "converged" and res.rounds == 3
    f = res.features
    k = f.index_of()
    assert f.round_index == 3
    assert f.d_um[k[2]] == pytest.approx(0.7031) and f.d_um[k[12]] == pytest.approx(0.5001)
    assert_allclose(f.offset_um[k[12]], [0.0, -0.3, 0.1], atol=1e-15)
    assert f.source[k[2]] == "refined" and f.source[k[1]] == "allen"
    assert f.flags[k[12]] == ""                                  # the flags reflect the latest fit only
    assert set(res.history[0]) == set(CI.HISTORY_KEYS)
    del starts


def test_failed_keeps_row_and_flags(monkeypatch):
    """Block 32: a failed fit leaves the row unchanged and adds refine_failed; max_rounds reached."""
    sc = Script({2: [(0, 0, 0, 0, "failed")], 12: [(0.61, 1.0, 0.0, 0.0, "converged")]})
    monkeypatch.setattr(CI.coarse_fit, "refine_node", sc)
    cfg = cfg_rounds(1)
    f0 = features0()
    res = CI.run_rounds(f0, scene_two(), [node(2, 1.0), node(12, 4.0)], cfg, 3.0, factory, shadow_fn=zero_shadow)
    k = res.features.index_of()
    assert res.features.flags[k[2]] == "refine_failed"
    assert res.features.d_um[k[2]] == 0.6 and res.features.source[k[2]] == "allen"
    assert_array_equal(res.features.offset_um[k[2]], [0, 0, 0])
    assert res.history[0]["n_failed"] == 1
    assert res.stopped in ("max_rounds", "converged") and res.rounds == 1


def test_shadow_fn_arguments_and_cleaning(monkeypatch):
    """Amendment 1 (i), (ii) and steps (c)-(e): shadow_fn gets B_gl = max_k B_k (a scalar) and include =
    (tiers == 1); the cleaned block is I + B_k S with the per-plane B_k; IterationResult.shadow = B_k S (gl);
    profiles_before / profiles_after = the bilinear read (nc_helpers, independent) of the uncleaned and the
    cleaned planes [1e-12 gl]; cleaned and shadow kept for the example nodes when keep_ids is None."""
    seen = []
    rng = np.random.default_rng(3)
    Smaps = {}

    def shadow_fn(fscene, own_ids, left, top, width, height, planes_z, cfg, B_gl, include):
        seen.append((tuple(own_ids), B_gl, np.array(include)))
        key = tuple(own_ids)
        if key not in Smaps:
            Smaps[key] = rng.uniform(0, 0.05, size=(len(planes_z), height, width))
        return Smaps[key]

    sc = Script({2: [(0.6, 1.0, 0.0, 0.0, "converged")], 12: [(0.6, 1.0, 0.0, 0.0, "converged")]})
    monkeypatch.setattr(CI.coarse_fit, "refine_node", sc)
    nodes = [node(2, 1.0), node(12, 4.0, seed=1)]
    cfg = cfg_rounds(1)
    res = CI.run_rounds(features0(), scene_two(), nodes, cfg, 3.0, factory, example_ids=(12,), shadow_fn=shadow_fn)
    for own, B, inc in seen:
        assert np.isscalar(B) or np.ndim(B) == 0
        assert float(B) == 210.0
    for nd in nodes:
        tiers = res.tiers[nd.node_id]
        inc = [s[2] for s in seen if s[0] == nd.own_ids][0]
        assert_array_equal(inc, tiers == 1)
    nd = nodes[1]
    S = Smaps[nd.own_ids]
    exp_clean = nd.block + nd.B_gl[:, None, None] * S
    assert set(res.cleaned) == {12} and set(res.shadow) == {12}
    assert_allclose(res.cleaned[12], exp_clean, rtol=0, atol=1e-12)
    assert_allclose(res.shadow[12], nd.B_gl[:, None, None] * S, rtol=0, atol=1e-12)
    for nd in nodes:
        before = np.stack([nc_helpers.bilinear_read(nd.block[q], 0, -20, P, nd.origin_xy, nd.y_hat, nd.e_u, nd.v,
                                                    nd.n_avg, nd.step) for q in range(3)])
        after_blk = nd.block + nd.B_gl[:, None, None] * Smaps[nd.own_ids]
        after = np.stack([nc_helpers.bilinear_read(after_blk[q], 0, -20, P, nd.origin_xy, nd.y_hat, nd.e_u, nd.v,
                                                   nd.n_avg, nd.step) for q in range(3)])
        assert_allclose(res.profiles_before[nd.node_id], before, rtol=0, atol=1e-9)
        assert_allclose(res.profiles_after[nd.node_id], after, rtol=0, atol=1e-9)
        # the profiles handed to the refinement are the cleaned ones
        prof = [c[2] for c in sc.calls if c[0] == nd.node_id][0]
        assert_allclose(prof, after, rtol=0, atol=1e-9)
    # the line change of round 1 is against the uncleaned profiles (Block 32: "for t = 1")
    exp_line = float(np.max(np.abs(res.profiles_after[12] - res.profiles_before[12])))
    assert res.history[0]["max_line_change_gl"] == pytest.approx(exp_line, rel=1e-12)


def test_order_independence_and_inputs_unchanged(monkeypatch):
    """(c) Jacobi: the result does not depend on the order of nodes [exact]; (f) inputs unchanged."""
    plan = {2: [(0.7, 1.2, 0.1, 0.0, "converged"), (0.72, 1.2, 0.1, 0.0, "converged")],
            12: [(0.5, 0.9, -0.1, 0.0, "converged"), (0.52, 0.9, -0.1, 0.0, "converged")]}
    cfg = cfg_rounds(2)
    out = []
    for order in ((0, 1), (1, 0)):
        monkeypatch.setattr(CI.coarse_fit, "refine_node", Script(plan))
        nodes = [node(2, 1.0), node(12, 4.0, seed=1)]
        blocks = [n.block.copy() for n in nodes]
        f0 = features0()
        d0 = f0.d_um.copy()
        res = CI.run_rounds(f0, scene_two(), [nodes[i] for i in order], cfg, 3.0, factory, example_ids=(2, 12))
        for n, b in zip(nodes, blocks):
            assert_array_equal(n.block, b)
        assert_array_equal(f0.d_um, d0)
        out.append(res)
    a, b = out
    assert_array_equal(a.features.d_um, b.features.d_um)
    assert_array_equal(a.features.offset_um, b.features.offset_um)
    for nid in (2, 12):
        assert_array_equal(a.profiles_after[nid], b.profiles_after[nid])
        assert_array_equal(a.tiers[nid], b.tiers[nid])
    assert a.history == b.history


def test_value_errors():
    """ValueError for an empty work list, duplicate ids, sigma_gl <= 0, an example id not in the work list,
    NodeData whose shapes disagree."""
    cfg = cfg_rounds(1)
    with pytest.raises(ValueError):
        CI.run_rounds(features0(), scene_two(), [], cfg, 3.0, factory)
    with pytest.raises(ValueError):
        CI.run_rounds(features0(), scene_two(), [node(2, 1.0), node(2, 1.0)], cfg, 3.0, factory)
    for s in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError):
            CI.run_rounds(features0(), scene_two(), [node(2, 1.0)], cfg, s, factory)
    with pytest.raises(ValueError):
        CI.run_rounds(features0(), scene_two(), [node(2, 1.0)], cfg, 3.0, factory, example_ids=(12,))
    nd = node(2, 1.0)
    with pytest.raises(ValueError):
        CI.NodeData(node_id=2, own_ids=(1, 2, 3), node_xyz=nd.node_xyz, block=nd.block, B_gl=nd.B_gl[:2],
                    ks=nd.ks, planes_z=nd.planes_z, left=0, top=-20, origin_xy=nd.origin_xy, y_hat=nd.y_hat,
                    e_u=nd.e_u, v=nd.v, n_avg=1, step=P)


# ------------------------------------------- amendment 2 (D-144.2), 2026-11-14 18:30 ---

class ScriptB(Script):
    """A scripted refine_node that also records the backgrounds it is given."""

    def __init__(self, plan):
        super().__init__(plan)
        self.B = []

    def __call__(self, prof, B, v, predict, start, cfg):
        self.B.append((predict.node_id, np.array(B, dtype=float)))
        return super().__call__(prof, B, v, predict, start, cfg)


def const_shadow(value):
    seen = []

    def fn(fscene, own_ids, left, top, width, height, planes_z, cfg, B_gl, include):
        seen.append(float(B_gl))
        return np.full((len(planes_z), height, width), value)
    fn.seen = seen
    return fn


def test_background_fn_used_everywhere(monkeypatch):
    """Block 32, amendment 2: background_fn(node, F) -> (n_planes,) gl is that round's B_k for the floor
    (shadow_fn's scalar = max_k B_k), the cleaning (I + B_k S) and the refinement (refine_node's B); returned
    as per_node B_gl and IterationResult.B_gl; called once per node and round with the featured scene of the
    round before (compared with featured_scene of the features after round 1, recomputed by hand from the
    scripted fits). None keeps NodeData.B_gl (regression of amendment 1's tests)."""
    from allen_diameter.analysis import node_features
    plan = {2: [(0.70, 1.2, 0.10, -0.2, "converged"), (0.71, 1.2, 0.10, -0.2, "converged")],
            12: [(0.55, 0.9, -0.1, 0.1, "converged"), (0.56, 0.9, -0.1, 0.1, "converged")]}
    sc = ScriptB(plan)
    monkeypatch.setattr(CI.coarse_fit, "refine_node", sc)
    Bs = {2: np.array([190.0, 191.0, 189.5]), 12: np.array([180.0, 182.0, 181.0])}
    calls = []

    def bfn(nd, F):
        calls.append((nd.node_id, F))
        return Bs[nd.node_id] + 0.5 * (len([c for c in calls if c[0] == nd.node_id]) - 1)
    shp = const_shadow(0.01)
    nodes = [node(2, 1.0), node(12, 4.0, seed=1)]
    cfg = cfg_rounds(2, 1e-12)
    f0 = features0()
    sc_ = scene_two()
    res = CI.run_rounds(f0, sc_, nodes, cfg, 3.0, factory, example_ids=(2,), shadow_fn=shp, background_fn=bfn,
                        keep_ids=(2, 12))
    assert res.rounds == 2 and len(calls) == 4
    # round 1 sees the features0 scene, round 2 the scene of the round-1 features
    F0 = node_features.featured_scene(sc_, f0)
    rows1 = [dict(node_id=2, d_um=0.70, mu_per_um=1.2, offset_um=np.array([0.0, 0.10, -0.2]), source="refined"),
             dict(node_id=12, d_um=0.55, mu_per_um=0.9, offset_um=np.array([0.0, -0.1, 0.1]), source="refined")]
    F1 = node_features.featured_scene(sc_, node_features.with_rows(f0, rows1, 1))
    for nid, F in calls[:2]:
        assert_array_equal(F.seg_xyz, F0.seg_xyz)
        assert_array_equal(F.seg_d_um, F0.seg_d_um)
    for nid, F in calls[2:]:
        assert_allclose(F.seg_xyz, F1.seg_xyz, rtol=0, atol=1e-15)
        assert_array_equal(F.seg_d_um, F1.seg_d_um)
    # the floor's scalar: max_k of this round's B_k (not NodeData.B_gl's 210)
    assert shp.seen == [191.0, 182.0, 191.5, 182.5]
    # refine_node got this round's B_k
    assert [nid for nid, _B in sc.B] == [2, 12, 2, 12]
    assert_array_equal(sc.B[0][1], Bs[2])
    assert_array_equal(sc.B[1][1], Bs[12])
    assert_array_equal(sc.B[2][1], Bs[2] + 0.5)
    # the cleaning: cleaned = I + B_k S with the last round's B_k [exact], shadow = B_k S
    for nd in nodes:
        Bl = Bs[nd.node_id] + 0.5
        assert_array_equal(res.B_gl[nd.node_id], Bl)
        assert_allclose(res.cleaned[nd.node_id], nd.block + Bl[:, None, None] * 0.01, rtol=0, atol=1e-12)
        assert_allclose(res.shadow[nd.node_id], np.broadcast_to(Bl[:, None, None] * 0.01, nd.block.shape),
                        rtol=0, atol=1e-15)
        # the profiles are Block 5's read of the cleaned planes (independent bilinear read)
        for q in range(3):
            ref = nc_helpers.bilinear_read(res.cleaned[nd.node_id][q], nd.left, nd.top, P, nd.origin_xy, nd.y_hat,
                                           nd.e_u, nd.v, nd.n_avg, nd.step)
            assert_allclose(res.profiles_after[nd.node_id][q], ref, rtol=0, atol=1e-9)
    # None: NodeData.B_gl every round
    sc2 = ScriptB(plan)
    monkeypatch.setattr(CI.coarse_fit, "refine_node", sc2)
    shp2 = const_shadow(0.0)
    res2 = CI.run_rounds(f0, sc_, nodes, cfg_rounds(1), 3.0, factory, shadow_fn=shp2)
    assert shp2.seen == [210.0, 210.0]
    assert_array_equal(sc2.B[0][1], nodes[0].B_gl)
    assert_array_equal(res2.B_gl[2], nodes[0].B_gl)


def test_background_fn_bad_values(monkeypatch):
    """Block 32, amendment 2: a background_fn that returns a wrong shape, a NaN or a non-positive value is
    refused (ValueError) rather than cleaning with it (the contract: (n_planes,) gl, a background)."""
    monkeypatch.setattr(CI.coarse_fit, "refine_node", Script({2: [(0.7, 1.0, 0.0, 0.0, "converged")]}))
    for bad in (np.array([200.0, 200.0]), np.array([200.0, np.nan, 200.0]), np.array([200.0, 0.0, 200.0])):
        with pytest.raises(ValueError):
            CI.run_rounds(features0(), scene_two(), [node(2, 1.0)], cfg_rounds(1), 3.0, factory,
                          shadow_fn=zero_shadow, background_fn=lambda nd, F, b=bad: b)
