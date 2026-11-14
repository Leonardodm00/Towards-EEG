"""Spec-derived tests of Block 36's node inputs (analysis/node_inputs.py) and its boundaries with Blocks 30,
32 and 35 -- integration tester, 2026-11-14. Oracles from specs/SPEC.md, Block 36 and its amendment 1
(D-141): the geometry by hand, the factories against line_fast called by hand (the spec's own oracle).

Run: python3 -m pytest -q tests/spec/test_spec_block36_node_inputs.py
Pure ASCII.
"""
from __future__ import annotations

import dataclasses
import math
import types

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import nc_helpers  # noqa: F401  (sys.path)
from allen_diameter.config import default_config
from allen_diameter.analysis import node_inputs as NI
from allen_diameter.analysis.clean_iteration import NodeData
from allen_diameter.analysis.node_pipeline import Branch
from allen_diameter.model import line_fast

P = 0.1144


def pilot(**kw):
    base = dict(cx_um=10.3, cy_um=-4.2, theta_rad=0.6, phi_rad=0.35, z_sub_um=1.13, k_star=4,
                d_hat_um=0.82, mu_hat_per_um=0.9, B_bar=210.0)
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_local_tube_by_hand():
    """Amendment 1 (2): centre (c_x, c_y) + (v0 - v_t) y_hat, v_t = ((c_x, c_y) - (x_t, y_t)) . y_hat,
    depth z_t + dz, radius d/2, heading theta, tilt phi, aspect 1, axial caps at U [1e-12]."""
    p = pilot()
    trace = np.array([10.1, -4.0, 0.6])
    yh = np.array([-math.sin(p.theta_rad), math.cos(p.theta_rad)])
    v_t = float((np.array([p.cx_um, p.cy_um]) - trace[:2]) @ yh)
    d, v0, dz, U = 0.7, 0.25, -0.3, 4.5
    t = NI.local_tube(p, trace, d, v0, dz, U)
    exp_xy = np.array([p.cx_um, p.cy_um]) + (v0 - v_t) * yh
    assert_allclose(np.asarray(t.c), [exp_xy[0], exp_xy[1], trace[2] + dz], rtol=0, atol=1e-12)
    assert t.r == pytest.approx(d / 2, abs=1e-15)
    assert t.theta == p.theta_rad and t.phi == p.phi_rad and t.aspect == 1.0
    assert t.half_length == U and t.end_cut == "axial"


def test_first_stage_start_is_the_pilot_tube():
    """Amendment 1 (2): theta0 = (d_hat, mu_hat, v_t, z_sub - z_t); the tube at theta0 is the pilot's own
    (centre (c_x, c_y, z_sub)) [1e-12]. The trace's offset along the branch does not move the tube."""
    p = pilot()
    e_u = np.array([math.cos(p.theta_rad), math.sin(p.theta_rad)])
    yh = np.array([-math.sin(p.theta_rad), math.cos(p.theta_rad)])
    trace = np.array([p.cx_um, p.cy_um, 0.4]) + np.r_[0.31 * yh + 0.8 * e_u, 0.0]
    th0 = NI.pilot_start(p, trace)
    assert_allclose(th0, [p.d_hat_um, p.mu_hat_per_um, -0.31, p.z_sub_um - 0.4], rtol=0, atol=1e-12)
    t = NI.local_tube(p, trace, th0[0], th0[2], th0[3], 5.0)
    assert_allclose(np.asarray(t.c), [p.cx_um, p.cy_um, p.z_sub_um], rtol=0, atol=1e-12)


def test_node_planes_and_half_length():
    """node_planes = k* - 4 .. k* + 4 (plane_half_range 4); tube_half_length = min(arc to first node, arc to
    last node, 5.0); near_end below extrusion_cap_margin_um (3.0)."""
    cfg = default_config()
    assert cfg.cleaning.plane_half_range == 4 and cfg.inference.tube_half_length_um == 5.0
    assert_array_equal(NI.node_planes(pilot(k_star=7), cfg), np.arange(3, 12))
    xyz = np.array([[0, 0, 0], [1.0, 0, 0], [3.0, 0, 0], [3.0, 4.0, 0], [3.0, 10.0, 0], [3.0, 11.0, 0]])
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))])
    br = Branch(np.arange(6), np.full(6, 3), xyz, np.full(6, 0.3), s)
    exp = [min(s[i] - s[0], s[-1] - s[i], 5.0) for i in range(6)]
    got = [NI.tube_half_length(br, i, cfg) for i in range(6)]
    assert_allclose(got, exp, rtol=0, atol=1e-12)
    assert [NI.is_near_end(br, i, cfg) for i in range(6)] == [e < 3.0 for e in exp]
    # at the margin exactly: 3.0 is not near_end ("at least" the margin)
    xyz2 = np.array([[0, 0, 0], [3.0, 0, 0], [6.0, 0, 0]])
    br2 = Branch(np.arange(3), np.full(3, 3), xyz2, np.full(3, 0.3), np.array([0.0, 3.0, 6.0]))
    assert not NI.is_near_end(br2, 1, cfg)
    with pytest.raises(ValueError):
        NI.tube_half_length(br, 6, cfg)


def test_own_branch_has_no_parent():
    """own_branch: the stretch's own nodes only (D-136.1); None for fewer than 2 nodes."""
    swc = types.SimpleNamespace(ids=np.array([10, 11, 12, 13, 14]), types=np.array([1, 3, 3, 3, 3]),
                                radius=np.array([5.0, 0.4, 0.3, 0.3, 0.2]))
    xyz = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0.0]])
    br, off = NI.own_branch(swc, [2, 3, 4], xyz)
    assert off == 0
    assert_array_equal(br.ids, [12, 13, 14])
    assert_allclose(br.s_um, [0, 1, 2])
    assert NI.own_branch(swc, [3], xyz) is None


def make_node(p, trace, ks, n_v=31):
    yh = np.array([-math.sin(p.theta_rad), math.cos(p.theta_rad)])
    eu = np.array([math.cos(p.theta_rad), math.sin(p.theta_rad)])
    v = (np.arange(n_v) - n_v // 2) * P
    H = W = 40
    return NodeData(node_id=5, own_ids=(4, 5, 6), node_xyz=trace, block=np.full((len(ks), H, W), 210.0),
                    B_gl=np.full(len(ks), 210.0), ks=np.asarray(ks), planes_z=np.asarray(ks) * 0.28,
                    left=0, top=0, origin_xy=np.array([p.cx_um, p.cy_um]), y_hat=yh, e_u=eu, v=v,
                    n_avg=2, step=P, theta0=None)


def fast_cfg(absorption="linear"):
    cfg = default_config()
    return dataclasses.replace(cfg, inference=dataclasses.replace(cfg.inference, absorption=absorption,
                                                                  ray_n_rho=4, ray_n_psi=8))


@pytest.mark.parametrize("absorption", ["linear", "hybrid"])
def test_factories_against_line_fast(absorption):
    """predict = 1 - line_fast.line_profiles(local_tube(pilot, trace, d, v0, dz, U), mu, ...) and
    forward(d, mu, v0, z_t + dz) = line_fast.line_profiles(same) [exact]; so forward = 1 - predict at
    z_ax = z_t + dz (the boundary Block 36 -> Blocks 29/32 and 30)."""
    cfg = fast_cfg(absorption)
    p = pilot(cx_um=2.0, cy_um=2.0, z_sub_um=0.84, k_star=3)
    trace = np.array([2.05, 1.95, 0.7])
    node = make_node(p, trace, np.arange(1, 6))
    U = 4.0
    pred = NI.predict_factory({5: p}, {5: U}, cfg, line_fast.SectionCache())(node)
    fwd = NI.forward_factory(p, U, node, cfg, line_fast.SectionCache())
    d, mu, v0, dz = 0.6, 1.1, 0.1, 0.2
    ref = line_fast.line_profiles(NI.local_tube(p, trace, d, v0, dz, U), mu, node.ks, node.origin_xy, node.y_hat,
                                  node.e_u, node.v, node.n_avg, node.step, cfg)
    assert_array_equal(pred(d, mu, v0, dz), 1.0 - ref)
    assert_array_equal(fwd(d, mu, v0, trace[2] + dz), ref)
    assert pred(d, mu, v0, dz).shape == (5, node.v.size)


def test_factories_nan_on_extrusion_invalid():
    """An ExtrusionInvalid gives NaN of the right shape (Block 36; Block 29/30 then fail the fit). A tube of
    half-length 2.5 um < margin 3.0 um is invalid wherever the line is."""
    cfg = fast_cfg()
    p = pilot(cx_um=2.0, cy_um=2.0, z_sub_um=0.84, k_star=3)
    trace = np.array([2.0, 2.0, 0.84])
    node = make_node(p, trace, np.arange(1, 4))
    pred = NI.predict_factory({5: p}, {5: 2.5}, cfg, None)(node)
    fwd = NI.forward_factory(p, 2.5, node, cfg, None)
    a, b = pred(0.6, 1.0, 0.0, 0.0), fwd(0.6, 1.0, 0.0, 0.84)
    assert a.shape == b.shape == (3, node.v.size)
    assert np.all(np.isnan(a)) and np.all(np.isnan(b))


# ------------------------------------------------------------ work list ---

def two_branch_swc(tmp_path, gap_um=2.6):
    """A soma (id 1) and two straight dendrites along x, 16 nodes 1 um apart each, gap_um apart in y and at
    the same depth: ids 2..17 (y = 0) and 18..33 (y = gap_um), both children of the soma."""
    lines = ["1 1 -3.0 %.3f 0.0 1.5 -1" % (gap_um / 2)]
    nid = 2
    for y in (0.0, gap_um):
        parent = 1
        for i in range(16):
            lines.append("%d 3 %.3f %.3f 0.0 0.3 %d" % (nid, float(i), y, parent))
            parent = nid
            nid += 1
    path = tmp_path / "two.swc"
    path.write_text("\n".join(lines) + "\n")
    from allen_diameter.loading import swc_io
    return swc_io.read_swc(str(path))


def test_work_list_examples_cap_threshold(tmp_path):
    """Block 36 work_list: the example nodes first; then end nodes of tier-1 neighbour segments whose peak
    x B reaches peak_min_gl, by decreasing peak; capped at work_max_nodes; no near_end node added; a huge
    threshold leaves the examples alone."""
    from allen_diameter.analysis import cell, node_features
    swc = two_branch_swc(tmp_path)
    cfg = default_config()
    types = tuple(cfg.cleaning.neighbour_swc_types)
    scene = cell.cell_scene(swc, types, cfg.acquisition.res0_um)
    feats = node_features.initial_features(swc, types, 1.0)
    fs = node_features.featured_scene(scene, feats)
    ex = [9]                                                    # x = 7 um on the first branch
    full = NI.work_list(ex, fs, swc, cfg, 3.0, 210.0)
    assert full[:1] == ex and len(set(full)) == len(full)
    extra = full[1:]
    assert extra, "a parallel dendrite 2.6 um away in focus must reach 3 gl"
    assert all(18 <= e <= 33 for e in extra), extra        # the other branch only (own stretch excluded)
    # ends of the other branch within 3 um of its ends are near_end (U < 3) and never added
    assert all(3.0 <= (e - 18) <= 12.0 for e in extra), extra
    capped = NI.work_list(ex, fs, swc, dataclasses.replace(cfg, cleaning=dataclasses.replace(
        cfg.cleaning, work_max_nodes=3)), 3.0, 210.0)
    assert capped == full[:3]
    assert NI.work_list(ex, fs, swc, cfg, 1e9, 210.0) == ex
