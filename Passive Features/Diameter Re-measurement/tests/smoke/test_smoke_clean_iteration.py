"""Smoke test for the cleaning iteration -- Block 32 in specs/SPEC.md (plan step C5; D-129.2,
D-130.O3, D-130.O6, D-132).

Closed forms (the entry's oracles): the shadow model and the predictor are built here from
calibration_scene.shadow_depths (a flat tube, one Gaussian kernel at the defocus sigma_r(z_tube - z_k)),
and the observed blocks are synthesised from the same closed forms at the pixel centres, so the
fixed point of the iteration is the truth.

Checks
    test_known_answer   (a) two parallel flat tubes 26 px (2.97 um) apart, each the other's tier-1
                        neighbour, started 30 % off in d: converges within max_rounds and both
                        diameters return to the truth within 1e-3 relative (stop_rel_d 1e-4); the
                        trajectory per round is printed
    test_reference      (b) shadow_fn returning zeros: the first round equals refine_node called by
                        hand on the uncleaned profiles [exact]; the real shadow.shadow_image
                        (partition_vertical) on Block 24's two-tube scene: one round's cleaned
                        profiles equal the node rendered alone [1e-9 gl]
    test_convergence    SKIP (no discretisation; the iteration's contraction is test_known_answer)
    test_invariants     (c) order of nodes irrelevant [exact]; (d) the stop rule against the two
                        thresholds written by hand on a recorded history (a loop of round_once and
                        with_rows written here), both branches, max_rounds = 1; (e) a NaN predict:
                        row kept, flag refine_failed, counted
    test_contract       NodeData / IterationResult shapes, dtypes, keys; history feeds
                        cleaning_figures.figure_rounds
    test_determinism    (f) features0, scene and every NodeData array unchanged; two runs bit-identical
    test_edge_cases     ValueError for an empty work list, duplicate node ids, a NodeData whose shapes
                        disagree, sigma_gl <= 0, an example id not in the work list

Run
    cd "Passive Features/Diameter Re-measurement"
    python3 tests/smoke/test_smoke_clean_iteration.py

Keep this file pure ASCII (hpc-python-compat).
"""
from __future__ import annotations

import dataclasses
import importlib.metadata
import math
import platform
import re
import sys
import time
import traceback
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal  # noqa: F401

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
SRC = WS / "src"
for p in (str(SRC), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from allen_image_io import CropFrame  # noqa: E402
from allen_diameter.analysis import clean_iteration as CI  # noqa: E402
from allen_diameter.analysis import coarse_fit  # noqa: E402
from allen_diameter.analysis.calibration_scene import shadow_depths  # noqa: E402
from allen_diameter.analysis.node_features import NodeFeatures, with_rows  # noqa: E402
from allen_diameter.analysis.plane_iteration import Scene  # noqa: E402
from allen_diameter.analysis.profiles import sample_profile  # noqa: E402
from allen_diameter.config import default_config  # noqa: E402
from allen_diameter.model import geometry as G  # noqa: E402
from allen_diameter.model import kernel as K  # noqa: E402
from allen_diameter.model import render as R  # noqa: E402
from allen_diameter.model import scene_render as SR  # noqa: E402

SEED = 20261113
REPORT_PACKAGES = ("numpy", "scipy", "matplotlib")
ARTIFACT_DIR = HERE / "artifacts"
_SAVED_ARTIFACTS = []
MEASURED = {}
PX = 0.1144
B_REF = 210.0
KS = np.arange(-3, 4)


# --------------------------------------------------------------- helpers ---

def make_rng(seed: int = SEED) -> np.random.Generator:
    return np.random.default_rng(seed)


def check_close(name, actual, expected, rtol=1e-12, atol=0.0):
    """numpy.testing.assert_allclose that keeps the evidence on failure."""
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    try:
        assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=name)
    except AssertionError:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
        path = ARTIFACT_DIR / f"{safe_name}.npz"
        np.savez(path, actual=actual, expected=expected)
        _SAVED_ARTIFACTS.append(path)
        raise


def observed_order(h, err):
    h = np.asarray(h, dtype=float)
    err = np.asarray(err, dtype=float)
    if np.any(err <= 0.0):
        raise ValueError("observed_order needs strictly positive errors")
    slope, _intercept = np.polyfit(np.log(h), np.log(err), 1)
    return float(slope)


def cfg_with(**cleaning):
    cfg = default_config()
    assert cfg.acquisition.res0_um == PX and cfg.measure.profile_step_um == PX
    return dataclasses.replace(cfg, cleaning=dataclasses.replace(cfg.cleaning, **cleaning))


CFG = cfg_with(stop_rel_d=1e-4)
DZ = CFG.acquisition.dz_um
PLANES_Z = KS * DZ


def sig(delta):
    return float(K.sigma_r(float(delta), CFG.renderer))


# ------------------------------------------------- the closed-form world ---
# Two parallel flat tubes along x, at y = 0 (nodes 1 -> 2) and y = SEP (nodes 11 -> 12), depth 0, each
# one segment. The work list is the two child nodes 2 and 12 (the features table's rows). The shadow
# model of a segment: shadow_depths at the lateral distance |y - y_child| with the child end's d and mu,
# kernel sigma_r(z_child - z_k) (a flat tube, one kernel; independent of x).

SEP = 26 * PX                                   # 2.9744 um: on a pixel centre, > close_lateral_um (2.0)
TRUTH = {2: (1.0, 1.0), 12: (0.8, 1.5)}         # node -> (d um, mu 1/um)
Y_OF = {2: 0.0, 12: SEP}
OWN = {2: (1, 2), 12: (11, 12)}


def make_scene(y12=SEP, half=8.0, r=0.5):
    xyz = np.array([[-half, 0.0, 0.0, half, 0.0, 0.0], [-half, y12, 0.0, half, y12, 0.0]])
    return Scene(seg_xyz=xyz, seg_radius=np.full((2, 2), r), seg_ids=np.array([[2, 1], [12, 11]]),
                 soma=np.zeros((0, 3)))


def make_features(d, mu, source=("allen", "allen")):
    return NodeFeatures(node_id=np.array([2, 12]), swc_type=np.array([3, 3]), d_um=np.asarray(d, float),
                        mu_per_um=np.asarray(mu, float), offset_um=np.zeros((2, 3)),
                        source=np.asarray(source), flags=np.array(["", ""]), round_index=0)


def closed_shadow_fn(fscene, own_ids, left, top, width, height, planes_z, cfg, B_gl, include):
    """S (n_planes, H, W): the included neighbours' closed-form shadows at the pixel centres."""
    from allen_diameter.model.shadow import neighbour_indices
    segs, somas = neighbour_indices(fscene, own_ids)
    assert somas.size == 0
    rows_y = (top + np.arange(height)) * PX
    S = np.zeros((len(planes_z), height, width))
    for i, q in enumerate(segs):
        if not include[i]:
            continue
        yq, zq = fscene.seg_xyz[q, 4], fscene.seg_xyz[q, 5]
        d, mu = fscene.seg_d_um[q, 1], fscene.seg_mu_per_um[q, 1]
        for k, zk in enumerate(planes_z):
            col = shadow_depths(np.abs(rows_y - yq), 0.5 * d, mu, sig(zq - zk), 1.0)
            S[k] += col[:, None]
    return S


def closed_predict_factory(node):
    """predict(d, mu, v0, dz) -> (n_planes, n_v): the node's own flat tube at v0, depth node z + dz."""
    v = np.asarray(node.v)
    zn = float(node.node_xyz[2])

    def predict(d, mu, v0, dz):
        return np.stack([shadow_depths(np.abs(v - v0), 0.5 * d, mu, sig(zn + dz - zk), 1.0)
                         for zk in node.planes_z])
    return predict


def observed_block(nid, left, top, width, height):
    """B (1 - s_own - s_other) at the pixel centres, from the truth."""
    rows_y = (top + np.arange(height)) * PX
    tot = np.zeros((KS.size, height))
    for m, (d, mu) in TRUTH.items():
        for k, zk in enumerate(PLANES_Z):
            tot[k] += shadow_depths(np.abs(rows_y - Y_OF[m]), 0.5 * d, mu, sig(0.0 - zk), 1.0)
    return np.repeat((B_REF * (1.0 - tot))[:, :, None], width, axis=2)


def make_node(nid, theta0=None, B=None):
    left, width, height = -8, 17, 60
    top = int(round(Y_OF[nid] / PX)) - 30
    n_avg = int(round(CFG.measure.along_branch_avg_um / CFG.measure.profile_step_um))
    return CI.NodeData(node_id=nid, own_ids=OWN[nid], node_xyz=np.array([0.0, Y_OF[nid], 0.0]),
                       block=observed_block(nid, left, top, width, height),
                       B_gl=np.full(KS.size, B_REF) if B is None else B, ks=KS.copy(), planes_z=PLANES_Z.copy(),
                       left=left, top=top, origin_xy=np.array([0.0, Y_OF[nid]]), y_hat=np.array([0.0, 1.0]),
                       e_u=np.array([1.0, 0.0]), v=np.arange(-26, 27) * PX, n_avg=n_avg, step=PX, theta0=theta0)


def start_features():
    """30 % off in d (node 2 high, node 12 low), mu at the truth."""
    return make_features([1.3 * TRUTH[2][0], 0.7 * TRUTH[12][0]], [TRUTH[2][1], TRUTH[12][1]])


def run(nodes=None, cfg=CFG, sigma=0.01, examples=(2, 12), features0=None, **kw):
    nodes = [make_node(2), make_node(12)] if nodes is None else nodes
    f0 = start_features() if features0 is None else features0
    return CI.run_rounds(f0, make_scene(), nodes, cfg, sigma, closed_predict_factory,
                         example_ids=examples, shadow_fn=kw.pop("shadow_fn", closed_shadow_fn), **kw)


def hand_loop(cfg, sigma, examples, nodes, features0, shadow_fn=closed_shadow_fn, predict_factory=closed_predict_factory):
    """The iteration written out here from round_once, with_rows and the stop rule of D-130.O6."""
    feats = features0
    frame = {n.node_id: CropFrame(n.left, n.top, 0, PX) for n in nodes}
    prev = {n.node_id: np.stack([sample_profile(n.block[k], frame[n.node_id], n.origin_xy, n.y_hat, n.e_u,
                                                n.v, n.n_avg, n.step) for k in range(n.block.shape[0])])
            for n in nodes}
    hist, traj, stopped = [], [], "max_rounds"
    for t in range(1, cfg.cleaning.max_rounds + 1):
        rows, per = CI.round_once(feats, make_scene(), nodes, cfg, predict_factory, shadow_fn)
        new = with_rows(feats, rows, t)
        ok = [i for i in per if per[i]["fit"]["status"] != "failed"]
        rel = [abs(new.d_um[new.index_of()[i]] - feats.d_um[feats.index_of()[i]]) / feats.d_um[feats.index_of()[i]]
               for i in ok]
        max_rel = max(rel) if rel else float("nan")
        line = max(np.max(np.abs(per[i]["profiles"] - prev[i])) for i in examples) if examples else float("nan")
        hist.append((t, max_rel, line))
        traj.append((t, dict(zip(new.node_id.tolist(), new.d_um.tolist())), line))
        feats = new
        prev = {i: per[i]["profiles"] for i in per}
        if max_rel < cfg.cleaning.stop_rel_d and (examples == () or line < sigma):
            stopped = "converged"
            break
    return feats, hist, traj, stopped


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    # (a) both diameters return to the truth within 1e-3 relative (the closed loop's contraction; the
    # test's stop_rel_d 1e-4 keeps the stop rule's own slack below the tolerance)
    res = run()
    _f, _h, traj, _s = hand_loop(CFG, 0.01, (2, 12), [make_node(2), make_node(12)], start_features())
    MEASURED["(a) trajectory: round, d(2), d(12) um, line change gl"] = [
        "start: d(2) %.6f, d(12) %.6f" % (1.3 * TRUTH[2][0], 0.7 * TRUTH[12][0])] + [
        "round %d: d(2) %.9f, d(12) %.9f, max line change %.3e" % (t, d[2], d[12], ln) for t, d, ln in traj]
    MEASURED["(a) stopped"] = "%s after %d rounds (max_rounds %d); history %s" % (
        res.stopped, res.rounds, CFG.cleaning.max_rounds,
        [(h["round"], "%.2e" % h["max_rel_d"], "%.2e" % h["max_line_change_gl"]) for h in res.history])
    assert res.stopped == "converged" and res.rounds <= CFG.cleaning.max_rounds, (res.stopped, res.rounds)
    pos = res.features.index_of()
    for nid, (d, mu) in TRUTH.items():
        rel = abs(res.features.d_um[pos[nid]] - d) / d
        MEASURED["(a) node %d" % nid] = "d %.9f (truth %.3f, rel err %.2e), mu %.6f (truth %.3f), offset %s" % (
            res.features.d_um[pos[nid]], d, rel, res.features.mu_per_um[pos[nid]], mu,
            np.array2string(res.features.offset_um[pos[nid]], precision=2))
        assert rel <= 1e-3, "node %d: relative error %.3e" % (nid, rel)
        assert res.features.source[pos[nid]] == "refined"
    # the test is not vacuous: the neighbour's shadow is on the line before cleaning (gl)
    dip = float(np.max(np.abs(res.profiles_before[2] - res.profiles_after[2])))
    MEASURED["(a) largest cleaning on node 2's line"] = "%.2f gl" % dip
    assert dip > 1.0


def test_reference():
    # (b) shadow_fn returning zeros: round 1 is refine_node on the uncleaned profiles [exact]
    nodes = [make_node(2), make_node(12, theta0=(0.9, 1.4, 0.05, 0.02))]
    feats = start_features()
    zero = lambda fs, own, left, top, w, h, z, cfg, B, inc: np.zeros((len(z), h, w))  # noqa: E731
    rows, per = CI.round_once(feats, make_scene(), nodes, CFG, closed_predict_factory, zero)
    res1 = CI.run_rounds(feats, make_scene(), nodes, dataclasses.replace(CFG, cleaning=dataclasses.replace(
        CFG.cleaning, max_rounds=1)), 0.01, closed_predict_factory, example_ids=(2,), shadow_fn=zero)
    for n in nodes:
        fr = CropFrame(n.left, n.top, 0, PX)
        prof = np.stack([sample_profile(n.block[k].astype(float), fr, n.origin_xy, n.y_hat, n.e_u, n.v,
                                        n.n_avg, n.step) for k in range(KS.size)])
        k = feats.index_of()[n.node_id]
        start = (feats.d_um[k], feats.mu_per_um[k], 0.0, 0.0) if n.theta0 is None else tuple(n.theta0)
        ref = coarse_fit.refine_node(prof, n.B_gl, n.v, closed_predict_factory(n), start, CFG)
        assert_array_equal(per[n.node_id]["profiles"], prof)
        assert_array_equal(res1.profiles_after[n.node_id], prof)
        for key in ("d_um", "mu_per_um", "v0_um", "dz_um", "cost", "rms_gl", "status", "at_bound", "nfev"):
            assert per[n.node_id]["fit"][key] == ref[key], (n.node_id, key)
            assert res1.fits[n.node_id][key] == ref[key], (n.node_id, key)
        row = [r for r in rows if r["node_id"] == n.node_id][0]
        assert row["d_um"] == ref["d_um"] and row["mu_per_um"] == ref["mu_per_um"]
        assert_array_equal(row["offset_um"], coarse_fit.offset_xyz(ref["v0_um"], ref["dz_um"], n.y_hat))
    # the real shadow.shadow_image (partition_vertical) on Block 24's two-tube scene [1e-9 gl]
    _real_shadow_oracle()


def _real_shadow_oracle():
    """Two flat capped tubes (d 0.6 um, mu 2 /um, half-length 2 um) at y = 0 and y = 21 px (2.40 um),
    depth 0, so that each alone and the pair share one slab lattice; the pixel means of Block 24's
    partition render (no far-field split) on the patch, B = 210. One round of round_once with the real
    shadow model: node 2's cleaned profiles equal sample_profile of B tau(node alone)."""
    cfg = cfg_with(shadow_absorption="partition_vertical")
    d, mu, U, y2 = 0.6, 2.0, 2.0, 21 * PX
    scene = Scene(seg_xyz=np.array([[-U, 0.0, 0.0, U, 0.0, 0.0], [-U, y2, 0.0, U, y2, 0.0]]),
                  seg_radius=np.full((2, 2), 0.5 * d), seg_ids=np.array([[2, 1], [12, 11]]), soma=np.zeros((0, 3)))
    feats = make_features([d, d], [mu, mu])
    tA = G.Tube(c=(0.0, 0.0, 0.0), r=0.5 * d, phi=0.0, theta=0.0, half_length=U, end_cut="axial")
    tB = G.Tube(c=(0.0, y2, 0.0), r=0.5 * d, phi=0.0, theta=0.0, half_length=U, end_cut="axial")
    left, top, W, H = -60, -60, 121, 141           # x -6.9..6.9 um, y -6.9..9.2 um: tubes + 6 sigma of blur
    grid, inner = R.block_fine_grid(left, top, W, H, PX, cfg.cleaning.shadow_fine_factor, 0)
    rc = dataclasses.replace(cfg.renderer, absorption="partition_vertical", fft_split_sigma_um=1e9)
    red = lambda pl: R.pixel_integrate(pl[inner[0], inner[1]], cfg.cleaning.shadow_fine_factor)  # noqa: E731
    both = SR.render_scene([tA, tB], [mu, mu], PLANES_Z, grid, rc, cfg.acquisition.light_direction, reduce=red)
    alone = SR.render_scene([tA], [mu], PLANES_Z, grid, rc, cfg.acquisition.light_direction, reduce=red)
    assert both.zeta.size == alone.zeta.size and np.max(np.abs(both.zeta - alone.zeta)) <= 1e-12
    n_avg = int(round(cfg.measure.along_branch_avg_um / cfg.measure.profile_step_um))
    v = np.arange(-26, 27) * PX
    node = CI.NodeData(node_id=2, own_ids=(1, 2), node_xyz=np.zeros(3), block=B_REF * both.tau,
                       B_gl=np.full(KS.size, B_REF), ks=KS.copy(), planes_z=PLANES_Z.copy(), left=left, top=top,
                       origin_xy=np.zeros(2), y_hat=np.array([0.0, 1.0]), e_u=np.array([1.0, 0.0]), v=v,
                       n_avg=n_avg, step=PX, theta0=None)
    t0 = time.perf_counter()
    rows, per = CI.round_once(feats, scene, [node], cfg, closed_predict_factory)
    dt = time.perf_counter() - t0
    fr = CropFrame(left, top, 0, PX)
    ref = np.stack([sample_profile(B_REF * alone.tau[k], fr, node.origin_xy, node.y_hat, node.e_u, v, n_avg, PX)
                    for k in range(KS.size)])
    before = np.stack([sample_profile(B_REF * both.tau[k], fr, node.origin_xy, node.y_hat, node.e_u, v, n_avg, PX)
                       for k in range(KS.size)])
    resid = float(np.max(np.abs(per[2]["profiles"] - ref)))
    MEASURED["real shadow_image oracle"] = ("max |cleaned - alone| %.3e gl (tol 1e-9); neighbour's dip on the line "
                                            "%.2f gl; tiers %s; round_once %.1f s" % (
                                                resid, float(np.max(np.abs(before - ref))), per[2]["tiers"].tolist(), dt))
    assert per[2]["tiers"].tolist() == [1]
    assert float(np.max(np.abs(before - ref))) > 1.0, "the neighbour's shadow must reach the line"
    assert resid <= 1e-9, "real-shadow oracle: %.3e gl" % resid


def test_convergence():
    raise unittest.SkipTest("no discretisation in the iteration; its contraction is test_known_answer (a)")


def test_invariants():
    # (c) the order of the nodes does not matter [exact]
    a = run(nodes=[make_node(2), make_node(12)])
    b = run(nodes=[make_node(12), make_node(2)])
    for name in ("node_id", "d_um", "mu_per_um", "offset_um", "source", "flags"):
        assert_array_equal(getattr(a.features, name), getattr(b.features, name))
    assert a.history == b.history and a.stopped == b.stopped and a.rounds == b.rounds
    for i in (2, 12):
        assert_array_equal(a.profiles_after[i], b.profiles_after[i])

    # (d) the stop rule against the thresholds written by hand on a recorded history
    nodes = [make_node(2), make_node(12)]
    feats, hist, _traj, stopped = hand_loop(CFG, 0.01, (2, 12), nodes, start_features())
    assert [(h["round"], h["max_rel_d"], h["max_line_change_gl"]) for h in a.history] == hist
    assert stopped == a.stopped
    for t, rel, line in hist[:-1]:
        assert not (rel < CFG.cleaning.stop_rel_d and line < 0.01), t
    # branch: d below its threshold but the line change not (sigma_gl tiny) -> runs to max_rounds
    c = run(sigma=1e-300)
    assert c.stopped == "max_rounds" and c.rounds == CFG.cleaning.max_rounds
    assert any(h["max_rel_d"] < CFG.cleaning.stop_rel_d for h in c.history)
    # branch: the line below sigma_gl but d not (stop_rel_d 0) -> runs to max_rounds
    cfg0 = cfg_with(stop_rel_d=0.0)
    e = run(cfg=cfg0, sigma=1e9)
    assert e.stopped == "max_rounds" and all(h["max_line_change_gl"] < 1e9 for h in e.history)
    # max_rounds = 1
    cfg1 = cfg_with(stop_rel_d=1e-4, max_rounds=1)
    m1 = run(cfg=cfg1)
    assert m1.rounds == 1 and m1.stopped == "max_rounds" and len(m1.history) == 1
    m1c = run(cfg=cfg1, sigma=1e9, features0=make_features([1.0, 0.8], [1.0, 1.5]))
    MEASURED["(d) max_rounds 1 from the truth"] = "stopped %s, max_rel_d %.2e" % (
        m1c.stopped, m1c.history[0]["max_rel_d"])
    assert m1c.stopped == "converged" and m1c.rounds == 1
    # the rule itself, both thresholds strict, NaN cases
    rule = CI._stop_reached
    assert rule(0.5e-4, 0.005, 1e-4, 0.01) and not rule(1e-4, 0.005, 1e-4, 0.01)
    assert not rule(0.5e-4, 0.01, 1e-4, 0.01) and not rule(float("nan"), 0.0, 1e-4, 0.01)
    assert rule(0.0, float("nan"), 1e-4, 0.01)

    # (e) a node whose predict returns NaN keeps its row, is flagged refine_failed and counted
    def nan_factory(node):
        p = closed_predict_factory(node)
        if node.node_id == 12:
            return lambda d, mu, v0, dz: np.full(p(d, mu, v0, dz).shape, np.nan)
        return p
    f0 = start_features()
    r = CI.run_rounds(f0, make_scene(), [make_node(2), make_node(12)], cfg1, 0.01, nan_factory, example_ids=(2,),
                      shadow_fn=closed_shadow_fn)
    k = r.features.index_of()[12]
    assert r.features.d_um[k] == f0.d_um[1] and r.features.mu_per_um[k] == f0.mu_per_um[1]
    assert_array_equal(r.features.offset_um[k], f0.offset_um[1])
    assert r.features.source[k] == "allen" and "refine_failed" in r.features.flags[k].split(";")
    assert r.history[0]["n_failed"] == 1 and r.history[0]["n_refined"] == 1
    assert r.fits[12]["status"] == "failed" and r.features.source[r.features.index_of()[2]] == "refined"


def test_contract():
    res = run(keep_ids=None)
    assert isinstance(res, CI.IterationResult) and res.stopped in CI.STOPPED
    assert set(res.cleaned) == {2, 12} == set(res.shadow) == set(res.tiers) == set(res.fits)
    for i in (2, 12):
        assert res.profiles_before[i].shape == res.profiles_after[i].shape == (KS.size, 53)
        assert res.cleaned[i].shape == res.shadow[i].shape == (KS.size, 60, 17)
        assert res.cleaned[i].dtype == np.float64 and res.tiers[i].dtype == np.int64
        assert res.tiers[i].tolist() == [1]
    for h in res.history:
        assert set(h) == set(CI.HISTORY_KEYS), sorted(h)
    assert res.features.round_index == res.rounds
    # B S (gl) and the cleaned block are consistent: cleaned - observed = shadow
    n = make_node(2)
    check_close("cleaned minus observed", res.cleaned[2] - n.block, res.shadow[2], rtol=0, atol=1e-12)
    # keep_ids restricts the kept arrays
    r2 = run(examples=(2,), keep_ids=(12,))
    assert set(r2.cleaned) == {12} and set(r2.profiles_after) == {2, 12}
    r3 = run(examples=(12,))
    assert set(r3.cleaned) == {12}
    # history feeds figure_rounds
    from allen_diameter.plotting import cleaning_figures as CF
    import matplotlib.pyplot as plt
    fig = CF.figure_rounds(res.history, CFG)
    plt.close(fig)
    fig = CF.figure_rounds(run(examples=()).history, CFG)     # no example nodes: NaN line changes
    plt.close(fig)


def test_determinism():
    nodes = [make_node(2), make_node(12, theta0=(0.7, 1.2, 0.1, -0.1))]
    copies = [{f.name: np.array(getattr(n, f.name), copy=True) for f in dataclasses.fields(n)
               if isinstance(getattr(n, f.name), np.ndarray)} for n in nodes]
    f0 = start_features()
    f0c = {f.name: np.array(getattr(f0, f.name), copy=True) for f in dataclasses.fields(f0)}
    scene = make_scene()
    sc = {f.name: np.array(getattr(scene, f.name), copy=True) for f in dataclasses.fields(scene)}
    a = CI.run_rounds(f0, scene, nodes, CFG, 0.01, closed_predict_factory, (2, 12), closed_shadow_fn)
    b = CI.run_rounds(f0, scene, nodes, CFG, 0.01, closed_predict_factory, (2, 12), closed_shadow_fn)
    for n, c in zip(nodes, copies):
        for name, arr in c.items():
            assert_array_equal(getattr(n, name), arr)
    for name, arr in f0c.items():
        assert_array_equal(getattr(f0, name), arr)
    for name, arr in sc.items():
        assert_array_equal(getattr(scene, name), arr)
    # the caller's own array is copied by NodeData: changing it later changes nothing
    blk = observed_block(2, -8, -30, 17, 60)
    nd = dataclasses.replace(make_node(2), block=blk)
    blk[:] = 0.0
    assert nd.block.max() > 0
    for name in ("d_um", "mu_per_um", "offset_um", "flags", "source"):
        assert_array_equal(getattr(a.features, name), getattr(b.features, name))
    assert a.history == b.history
    for i in (2, 12):
        assert_array_equal(a.profiles_after[i], b.profiles_after[i])
        assert_array_equal(a.cleaned[i], b.cleaned[i])


def _raises(func, *args, **kw):
    try:
        func(*args, **kw)
    except ValueError:
        return
    raise AssertionError("no ValueError from %r" % (func,))


def test_edge_cases():
    f0, sc = start_features(), make_scene()
    n2, n12 = make_node(2), make_node(12)
    run_ = lambda nodes, sigma=0.01, ex=(): CI.run_rounds(f0, sc, nodes, CFG, sigma, closed_predict_factory,  # noqa: E731
                                                          ex, closed_shadow_fn)
    _raises(run_, [])
    _raises(run_, [n2, make_node(2)])
    _raises(run_, [n2, n12], 0.0)
    _raises(run_, [n2, n12], -1.0)
    _raises(run_, [n2, n12], float("nan"))
    _raises(run_, [n2, n12], 0.01, (2, 99))
    _raises(CI.run_rounds, f0, sc, [n2], CFG, 0.01, closed_predict_factory, (), closed_shadow_fn, keep_ids=(12,))
    _raises(CI.round_once, f0, sc, [], CFG, closed_predict_factory, closed_shadow_fn)
    # NodeData whose shapes disagree
    bad = dict(B_gl=np.full(KS.size + 1, B_REF))
    _raises(dataclasses.replace, n2, **bad)
    _raises(dataclasses.replace, n2, ks=np.arange(3))
    _raises(dataclasses.replace, n2, planes_z=np.zeros(2))
    _raises(dataclasses.replace, n2, block=np.zeros((KS.size, 4)))
    _raises(dataclasses.replace, n2, origin_xy=np.zeros(3))
    _raises(dataclasses.replace, n2, theta0=np.zeros(3))
    _raises(dataclasses.replace, n2, B_gl=np.zeros(KS.size))
    _raises(dataclasses.replace, n2, own_ids=())
    # a node that is not in the features; an S of the wrong shape; a line that leaves the patch
    _raises(CI.round_once, f0, sc, [dataclasses.replace(n2, node_id=5)], CFG, closed_predict_factory,
            closed_shadow_fn)
    _raises(CI.round_once, f0, sc, [n2], CFG, closed_predict_factory,
            lambda *a: np.zeros((KS.size, 3, 3)))
    _raises(CI.round_once, f0, sc, [dataclasses.replace(n2, origin_xy=np.array([0.0, 2.0]))], CFG,
            closed_predict_factory, closed_shadow_fn)


# ---------------------------------------------------------------- runner ---

def _environment():
    parts = [f"python {platform.python_version()}"]
    for package in REPORT_PACKAGES:
        try:
            parts.append(f"{package} {importlib.metadata.version(package)}")
        except importlib.metadata.PackageNotFoundError:
            parts.append(f"{package} (not installed)")
    parts.append(platform.platform())
    parts.append(f"seed {SEED}")
    return " | ".join(parts)


def main():
    checks = [(name, obj) for name, obj in globals().items()
              if name.startswith("test_") and callable(obj)]
    print(f"== {Path(__file__).name}")
    print(_environment())

    results = []
    for name, func in checks:
        start = time.perf_counter()
        try:
            func()
            status, detail = "PASS", ""
        except unittest.SkipTest as exc:
            status, detail = "SKIP", str(exc)
        except NotImplementedError as exc:
            status, detail = "TODO", str(exc)
        except AssertionError as exc:
            status, detail = "FAIL", str(exc)
        except Exception:  # unexpected errors are reported, never hidden
            status, detail = "ERROR", traceback.format_exc()
        results.append((name, status, time.perf_counter() - start, detail))

    width = max([len(name) for name, _, _, _ in results] + [4])
    for name, status, seconds, detail in results:
        lines = detail.strip().splitlines()
        headline = lines[0] if lines else ""
        print(f"{status:<5}  {name:<{width}}  {seconds:8.3f}s  {headline}".rstrip())

    for name, status, _seconds, detail in results:
        if status in ("FAIL", "ERROR"):
            print(f"\n---- {status}: {name}\n{detail.strip()}")

    counts = {s: sum(1 for r in results if r[1] == s)
              for s in ("PASS", "FAIL", "ERROR", "TODO", "SKIP")}
    print("\n-- " + ", ".join(f"{n} {s.lower()}" for s, n in counts.items()))
    for key, val in MEASURED.items():
        if isinstance(val, list):
            print("-- %s:" % key)
            for line in val:
                print("     " + line)
        else:
            print("-- %s: %s" % (key, val))
    for path in _SAVED_ARTIFACTS:
        print(f"-- arrays from a failed comparison: {path}")
    failed = counts["FAIL"] + counts["ERROR"] + counts["TODO"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
