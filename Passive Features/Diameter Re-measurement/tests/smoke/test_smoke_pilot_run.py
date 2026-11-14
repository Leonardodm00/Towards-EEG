"""Smoke test for the pilot run's node inputs and its two entry points -- Block 36 in specs/SPEC.md
(node_inputs.py; scripts/clean_cell.py survey / clean; scripts/infer_nodes.py real).

Checks
    test_known_answer       local_tube against the pilot's geometry by hand [1e-12]; tube_half_length on
                            a branch written by hand, the near_end flag on either side of the margin;
                            own_branch holds no parent node (D-136.1), None for one node; node_planes
    test_reference          predict_factory and forward_factory equal line_fast.line_profiles called by
                            hand [exact], and agree with each other at z_ax = z_sub + dz [exact]
    test_convergence        not a discretisation: skipped
    test_invariants         build_node on a rendered two-tube scene (Block 24 behind a provider): B_k
                            within 0.5 gl of the scene's background, the profiles Block 32 reads equal
                            sample_profile on the provider's block along the pilot's line [exact], node_xyz
                            and theta0 as stated; work_list contains the examples first, follows the
                            threshold (a strong in-focus neighbour in, a weak defocused one out) and the cap
    test_trace_offset       amendment 1 (2): a noise-free rendered tube whose trace is displaced by
                            (0.3 um lateral, 0.5 um deep): one round of Block 32 with this block's
                            predict recovers the displacement in the features row's offset [0.02 um
                            lateral, 0.05 um depth], and featured_scene draws the node on the rendered
                            tube within the same tolerance
    test_noise_free_fit     on a noise-free rendered isolated tube (Block 4, no camera chain), the pilot,
                            build_node and fit_node of forward_factory: the fitted diameter within 2 % of
                            the rendered one [the pilot's and the model's discretisation]
    test_contract           end to end with --synthetic: survey, clean (1 round) and infer_nodes.py real
                            run; every listed file exists and reloads; features_<S>.csv round-trips
                            (Block 23); the two JSONs carry the cleaning and inference sections; the inputs
                            (the examples file) unchanged by clean
    test_determinism        the three commands run twice give identical files (the seconds column of
                            inference_<S>.csv aside)
    test_edge_cases         build_node's refusals with their reasons (near_end, no sharpest plane, no
                            usable pilot fit, a missing plane, a plane without background); an
                            ExtrusionInvalid gives NaN in predict and forward; work_list's ValueErrors;
                            the inputs of build_node unchanged

The fits run under partition_vertical with 4 x 8 directions, one start and few evaluations, the shadow
model under partition_vertical at fine factor 2 with the soma transparent (the entry allows it for time).
The background runs under cleaning.background_support at its default "square" (Block 36, amendment 1,
D-141); the shifts v0, dz are about the trace (amendment 1 (2)).

Run
    cd "Passive Features/Diameter Re-measurement"
    python3 tests/smoke/test_smoke_pilot_run.py

Keep this file pure ASCII (hpc-python-compat).
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
import traceback
import types
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal  # noqa: F401

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
for _p in (WS / "src", WS / "scripts", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from allen_image_io import CropFrame  # noqa: E402
from allen_diameter.analysis import calibration_scene, cell, clean_iteration, node_features  # noqa: E402
from allen_diameter.analysis import cleaning, node_inference, profiles  # noqa: E402
from allen_diameter.analysis import node_inputs as NI  # noqa: E402
from allen_diameter.analysis.node_pipeline import Branch, measure_pilot  # noqa: E402
from allen_diameter.config import default_config  # noqa: E402
from allen_diameter.loading import features_io, swc_io  # noqa: E402
from allen_diameter.model import geometry, line_fast, render, scene_render, shadow  # noqa: E402

SEED = 20261113
REPORT_PACKAGES = ("numpy", "scipy", "matplotlib")
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
_SAVED_ARTIFACTS = []
_CACHE = {}


# --------------------------------------------------------------- helpers ---

def check_close(name, actual, expected, rtol=1e-12, atol=0.0):
    """numpy.testing.assert_allclose that keeps the evidence on failure (ARTIFACT_DIR/<name>.npz)."""
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    try:
        assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=name)
    except AssertionError:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        path = ARTIFACT_DIR / ("%s.npz" % re.sub(r"[^A-Za-z0-9_.-]+", "_", name))
        np.savez(path, actual=actual, expected=expected)
        _SAVED_ARTIFACTS.append(path)
        raise


FAST = {"cleaning": {"shadow_absorption": "partition_vertical", "shadow_fine_factor": 2, "soma_mu_per_um": 0.0,
                     "example_counts": [1, 0, 0, 0, 0], "max_rounds": 1,
                     "refine_max_nfev": 40},
        "inference": {"absorption": "partition_vertical", "ray_n_rho": 4, "ray_n_psi": 8,
                      "multistart_factors": [1.0], "max_nfev": 40}}


def fast_cfg():
    """The test's configuration: FAST over the defaults."""
    cfg = default_config()
    cl = dataclasses.replace(cfg.cleaning, **{k: (tuple(v) if isinstance(v, list) else v)
                                              for k, v in FAST["cleaning"].items()})
    inf = dataclasses.replace(cfg.inference, **{k: (tuple(v) if isinstance(v, list) else v)
                                                for k, v in FAST["inference"].items()})
    cfg = dataclasses.replace(cfg, cleaning=cl, inference=inf)
    cfg.validate()
    return cfg


def fake_pilot(**kw):
    base = dict(cx_um=1.3, cy_um=-0.7, theta_rad=0.4, phi_rad=0.25, z_sub_um=0.31, k_star=1, d_hat_um=0.7,
                mu_hat_per_um=0.9, B_bar=210.0, fit_status="converged")
    base.update(kw)
    return types.SimpleNamespace(**base)


def write_swc(path, dendrites, soma=(-20.0, 0.0, 0.0, 2.0)):
    """A soma (id 1) and straight dendrites [(start xyz, end xyz, n nodes, radius)], each a child chain of the
    soma. Returns the SWC object."""
    lines = ["# block 36 test", "1 1 %.4f %.4f %.4f %.4f -1" % soma]
    nid = 2
    for (a, b, n, r) in dendrites:
        a, b = np.asarray(a, float), np.asarray(b, float)
        parent = 1
        for t in np.linspace(0.0, 1.0, n):
            q = a + t * (b - a)
            lines.append("%d 3 %.4f %.4f %.4f %.4f %d" % (nid, q[0], q[1], q[2], r, parent))
            parent = nid
            nid += 1
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return swc_io.read_swc(path)


def block_provider(block, ks, left, top, res0):
    """The fetch_zblock contract over a precomputed block (pixels outside it 255, planes outside invalid)."""
    ks = np.asarray(ks)
    H, W = block.shape[1:]

    def provider(l_, t_, w_, h_, k_lo, k_hi):
        kk = np.arange(int(k_lo), int(k_hi) + 1)
        out = np.full((kk.size, h_, w_), 255.0 if block.dtype.kind == "f" else 255, dtype=block.dtype)
        valid = np.zeros(kk.size, dtype=bool)
        x0, y0 = max(l_, left), max(t_, top)
        x1, y1 = min(l_ + w_, left + W), min(t_ + h_, top + H)
        for n, k in enumerate(kk):
            if k in ks:
                valid[n] = True
                i = int(np.flatnonzero(ks == k)[0])
                if x1 > x0 and y1 > y0:
                    out[n, y0 - t_:y1 - t_, x0 - l_:x1 - l_] = block[i, y0 - top:y1 - top, x0 - left:x1 - left]
        return out, kk, valid, CropFrame(int(l_), int(t_), 0, float(res0))
    return provider


def two_tube_case():
    """A rendered two-tube scene behind a provider and its SWC: dendrite A along x at y = 0 (d 0.6 um), its
    neighbour B parallel at y = 3 um (d 0.6 um), both in focus at z = 0, mu 1 /um, camera chain with noise
    (Block 24); and the SWC of a cell that also traces a third dendrite C (not rendered) 4 um deep at y = -4 um, d 0.2 um
    (_CACHE['three'], for work_list)."""
    if "two" in _CACHE:
        return _CACHE["two"]
    cfg = fast_cfg()            # cleaning.background_support at its default, "square" (amendment 1, D-141)
    tmp = tempfile.mkdtemp(prefix="b36_two_")
    ab = [((0.0, 0.0, 0.0), (20.0, 0.0, 0.0), 11, 0.3), ((0.0, 3.0, 0.0), (20.0, 3.0, 0.0), 11, 0.3)]
    swc = write_swc(os.path.join(tmp, "two.swc"), ab)                     # what is rendered
    swc_c = write_swc(os.path.join(tmp, "three.swc"), ab + [((0.0, -4.0, 4.0), (20.0, -4.0, 4.0), 11, 0.1)])
    res0 = cfg.acquisition.res0_um
    tubes = [geometry.Tube((10.0, 0.0, 0.0), 0.3, 0.0, 0.0, 1.0, 14.0, "axial"),
             geometry.Tube((10.0, 3.0, 0.0), 0.3, 0.0, 0.0, 1.0, 14.0, "axial")]
    ks = np.arange(-5, 6)
    left, top = int(math.floor(-0.5 / res0)), int(math.floor(-10.5 / res0))
    width = height = int(math.ceil(21.0 / res0))
    block, ks, _valid, frame = scene_render.synthetic_scene_block(tubes, [1.0, 1.0], ks, left, top, width, height, cfg,
                                                                  np.random.default_rng(SEED), 8)
    provider = block_provider(block, ks, left, top, res0)
    _CACHE["two"] = (cfg, swc, provider, (block, ks, left, top))
    _CACHE["three"] = swc_c
    return _CACHE["two"]


def fscene_of(swc, scene, mu=1.0, d=None):
    """The round-0 featured scene: Allen's diameters (or d um everywhere) and mu everywhere."""
    feats = node_features.initial_features(swc, (2, 3, 4), mu)
    if d is not None:
        feats = node_features.with_rows(feats, [dict(node_id=int(i), d_um=float(d)) for i in feats.node_id], 0)
    return node_features.featured_scene(scene, feats)


def node_of(swc, cfg, nid, transform=None):
    """(full stretch branch, offset, own branch, index) of node nid on its dendrite stretch."""
    xyz = cell.to_image_um(swc.xyz, cfg.acquisition.res0_um, **(transform or {}))
    for run in cell.stretches(swc, cfg.acquisition.dendrite_swc_types):
        ids = [int(swc.ids[r]) for r in run]
        if nid in ids:
            t = ids.index(nid)
            full, off = cell.stretch_branch(swc, run, xyz)
            ob, _ = NI.own_branch(swc, run, xyz)
            return full, off + t, ob, t
    raise KeyError(nid)


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    cfg = default_config()
    # local_tube against the pilot's geometry and the trace by hand (amendment 1, D-141): the foot of the
    # trace on the measuring line plus v0 y_hat, depth z_t + dz
    p = fake_pilot()
    trace = np.array([1.1, -0.4, -0.2])
    d, v0, dz, U = 0.83, 0.21, -0.47, 4.2
    tube = NI.local_tube(p, trace, d, v0, dz, U, U)
    yh = np.array([-math.sin(p.theta_rad), math.cos(p.theta_rad)])
    c = np.array([p.cx_um, p.cy_um])
    foot = c + np.dot(trace[:2] - c, yh) * yh
    check_close("local_tube_centre", np.array(tube.c), [foot[0] + v0 * yh[0], foot[1] + v0 * yh[1], trace[2] + dz],
                rtol=0, atol=1e-12)
    v_t = np.dot(c - trace[:2], yh)
    check_close("trace_offset", NI.trace_offset(p, trace), v_t, rtol=0, atol=1e-12)
    check_close("pilot_start", NI.pilot_start(p, trace), [p.d_hat_um, p.mu_hat_per_um, v_t, p.z_sub_um - trace[2]],
                rtol=0, atol=1e-12)
    # at the pilot's own start the tube sits on the pilot's centre and depth
    t0 = NI.local_tube(p, trace, d, v_t, p.z_sub_um - trace[2], U, U)
    check_close("start_on_pilot", np.array(t0.c), [p.cx_um, p.cy_um, p.z_sub_um], rtol=0, atol=1e-12)
    check_close("local_tube_r", tube.r, d / 2.0, rtol=0, atol=1e-12)
    assert (tube.theta, tube.phi, tube.aspect, tube.half_length, tube.end_cut) == \
        (p.theta_rad, p.phi_rad, 1.0, U, "axial"), tube
    # asymmetric extents (amendment 3): centre moved by (U+ - U-)/2 along a_hat, half-length (U+ + U-)/2
    ta = NI.local_tube(p, trace, d, v0, dz, 2.0, 7.0)
    a_hat = np.array([math.cos(p.phi_rad) * math.cos(p.theta_rad), math.cos(p.phi_rad) * math.sin(p.theta_rad),
                      math.sin(p.phi_rad)])
    check_close("local_tube_asym", np.array(ta.c), np.array(tube.c) + 2.5 * a_hat, rtol=0, atol=1e-12)
    assert ta.half_length == 4.5
    # own_ids_within and tube_extent on a branch written by hand: arc lengths 0, 1, 3, 6, 10, 15 um, T = 5:
    # each side up to and including the first node beyond 5 um, or the end
    s = np.array([0.0, 1.0, 3.0, 6.0, 10.0, 15.0])
    P = np.column_stack([s, np.zeros(6), np.zeros(6)])
    br = Branch(np.arange(10, 16), np.full(6, 3), P, np.full(6, 0.3), s)
    want = {0: ((10, 11, 12, 13), 0.0, 6.0), 2: ((10, 11, 12, 13, 14), 3.0, 7.0), 3: ((10, 11, 12, 13, 14, 15), 6.0, 9.0),
            5: ((13, 14, 15), 9.0, 0.0)}
    for i, w in want.items():
        got = NI.own_ids_within(br, i, cfg.inference.tube_half_length_um)
        assert got[0] == w[0], (i, got)
        check_close("own_extent_%d" % i, got[1:], w[1:], rtol=0, atol=1e-12)
        check_close("tube_extent_%d" % i, NI.tube_extent(br, i, cfg), w[1:], rtol=0, atol=1e-12)
    # near_end at the margin (3.0 um): an extent exactly 3.0 um is valid, 3.0 - 1e-9 um is not
    m = cfg.inference.extrusion_cap_margin_um
    for arc, expect in ((m, False), (m - 1e-9, True), (m + 1e-9, False)):
        s2 = np.array([0.0, arc, 20.0])
        b2 = Branch(np.arange(3), np.full(3, 3), np.column_stack([s2, np.zeros(3), np.zeros(3)]), np.full(3, 0.3), s2)
        assert NI.is_near_end(b2, 1, cfg) is expect, (arc, expect)
    # own_branch: no parent node (D-136.1); cell.stretch_branch prepends one
    tmp = tempfile.mkdtemp(prefix="b36_own_")
    swc = write_swc(os.path.join(tmp, "y.swc"), [((0, 0, 0), (10, 0, 0), 6, 0.3)])
    # a branch point: two children of the last node
    with open(os.path.join(tmp, "y.swc"), "a") as f:
        f.write("8 3 12.0 1.0 0.0 0.3 7\n9 3 14.0 2.0 0.0 0.3 8\n10 3 12.0 -1.0 0.0 0.3 7\n11 3 14.0 -2.0 0.0 0.3 10\n")
    swc = swc_io.read_swc(os.path.join(tmp, "y.swc"))
    xyz = cell.to_image_um(swc.xyz, cfg.acquisition.res0_um)
    runs = cell.stretches(swc)
    child = [r for r in runs if int(swc.ids[r[0]]) == 8][0]
    full, off = cell.stretch_branch(swc, child, xyz)
    ob, k0 = NI.own_branch(swc, child, xyz)
    assert off == 1 and int(full.ids[0]) == 7, (off, full.ids)
    assert k0 == 0 and [int(i) for i in ob.ids] == [8, 9], ob.ids
    check_close("own_branch_xyz", ob.xyz_um, xyz[child], rtol=0, atol=0)
    assert NI.own_branch(swc, child[:1], xyz) is None
    # node_planes
    assert_array_equal(NI.node_planes(fake_pilot(k_star=7), cfg),
                       np.arange(7 - cfg.cleaning.plane_half_range, 8 + cfg.cleaning.plane_half_range))


def _line_node(cfg):
    p = fake_pilot(phi_rad=0.1, theta_rad=0.2, cx_um=10.0, cy_um=0.0, z_sub_um=0.05, k_star=0)
    ks = np.arange(-2, 3)
    origin, yh, eu = np.array([p.cx_um, p.cy_um]), np.array([-math.sin(0.2), math.cos(0.2)]), \
        np.array([math.cos(0.2), math.sin(0.2)])
    v = profiles.profile_offsets(cfg.measure)
    node = clean_iteration.NodeData(node_id=5, own_ids=(4, 5, 6), node_xyz=[9.9, -0.15, -0.3],
                                    block=np.full((ks.size, 4, 4), 210.0), B_gl=np.full(ks.size, 210.0), ks=ks,
                                    planes_z=ks * cfg.acquisition.dz_um, left=0, top=0, origin_xy=origin, y_hat=yh,
                                    e_u=eu, v=v, n_avg=profiles.n_along(cfg.measure),
                                    step=cfg.measure.profile_step_um, theta0=None)
    return p, node


def test_reference():
    cfg = fast_cfg()
    p, node = _line_node(cfg)
    U = (5.0, 5.0)
    cache = line_fast.SectionCache()
    pred = NI.predict_factory({5: p}, {5: U}, cfg, cache)(node)
    fwd = NI.forward_factory(p, U, node, cfg, cache)
    d, mu, v0, dz = 0.7, 1.1, 0.12, -0.2
    z_t = float(node.node_xyz[2])
    by_hand = line_fast.line_profiles(NI.local_tube(p, node.node_xyz, d, v0, dz, *U), mu, node.ks, node.origin_xy,
                                      node.y_hat, node.e_u, node.v, node.n_avg, node.step, cfg)
    assert_array_equal(pred(d, mu, v0, dz), 1.0 - by_hand)
    assert_array_equal(fwd(d, mu, v0, z_t + dz), by_hand)
    assert_array_equal(1.0 - pred(d, mu, v0, dz), fwd(d, mu, v0, z_t + dz))
    assert np.min(by_hand) < 0.9, "the tube must cast a dip"
    # the hybrid through the cache: equal to an uncached call
    hcfg = dataclasses.replace(cfg, inference=dataclasses.replace(cfg.inference, absorption="hybrid"))
    fh = NI.forward_factory(p, U, node, hcfg, line_fast.SectionCache())
    ref = line_fast.line_profiles(NI.local_tube(p, node.node_xyz, d, v0, 0.1 - z_t, *U), mu, node.ks, node.origin_xy, node.y_hat,
                                  node.e_u, node.v, node.n_avg, node.step, hcfg)
    assert_array_equal(fh(d, mu, v0, 0.1), ref)


def test_convergence():
    raise unittest.SkipTest("not a discretisation: the block wires merged blocks")


def test_invariants():
    cfg, swc, provider, (blk, ks_all, left, top) = two_tube_case()
    full, i_full, ob, t = node_of(swc, cfg, 7)            # dendrite A, the middle node (x = 10 um)
    pilot = measure_pilot(full, i_full, provider, cfg)
    scene = cell.cell_scene(swc, (2, 3, 4), cfg.acquisition.res0_um)
    own = NI.own_ids_within(ob, t, cfg.inference.tube_half_length_um)[0]
    node, info = NI.build_node(ob, t, pilot, provider, scene, cfg, own, fscene_of(swc, scene))   # true features
    assert node is not None, info
    bg = cfg.renderer.background_B_gl
    assert np.max(np.abs(node.B_gl - bg)) <= 0.5, ("B_k within 0.5 gl of the background", node.B_gl)
    assert_array_equal(node.ks, NI.node_planes(pilot, cfg))
    assert cfg.cleaning.background_support == "model", "the default support (amendment 4)"
    check_close("node_xyz", node.node_xyz, ob.xyz_um[t], rtol=0, atol=0)          # the traced position
    check_close("theta0", node.theta0, NI.pilot_start(pilot, ob.xyz_um[t]), rtol=0, atol=0)
    print("    two tubes 3 um apart, default support %r: B_k %s gl, support share %s, profile noise SD %.2f gl"
          % (cfg.cleaning.background_support, np.round(node.B_gl, 2).tolist(),
             np.round(info["support_frac"], 3).tolist(), info["sigma_gl"]))
    assert node.own_ids == own and 1 not in own
    assert math.isfinite(info["sigma_gl"]) and info["sigma_gl"] > 0
    # the profiles Block 32 reads equal sample_profile on the provider's block along the pilot's line
    patch = calibration_scene.fetch_patch(provider, pilot.cx_um, pilot.cy_um, cfg.cleaning.patch_half_um,
                                          node.ks[0], node.ks[-1], cfg.acquisition.res0_um)
    th = pilot.theta_rad
    o, yh, eu = np.array([pilot.cx_um, pilot.cy_um]), np.array([-math.sin(th), math.cos(th)]), \
        np.array([math.cos(th), math.sin(th)])
    v = profiles.profile_offsets(cfg.measure)
    want = np.stack([profiles.sample_profile(patch[0][q], patch[3], o, yh, eu, v, profiles.n_along(cfg.measure),
                                             cfg.measure.profile_step_um) for q in range(node.ks.size)])
    got = clean_iteration._profiles(np.asarray(node.block, dtype=float), node, cfg)
    assert_array_equal(got, want)
    # work_list: the examples first, the threshold, the cap (the traced cell adds dendrite C)
    swc = _CACHE["three"]
    scene = cell.cell_scene(swc, (2, 3, 4), cfg.acquisition.res0_um)
    feats = node_features.initial_features(swc, (2, 3, 4), 1.0)
    fs = node_features.featured_scene(scene, feats)
    ex = [7, 18]                                            # A's and B's middle nodes
    B = 210.0
    peaks = {}
    for e in ex:
        _f, _i, obe, _t = node_of(swc, cfg, e)
        owne = tuple(int(i) for i in obe.ids)
        xyz = NI._node_position(fs, e)
        l_, t_, w_, h_ = calibration_scene.patch_bounds(xyz[0], xyz[1], cfg.cleaning.patch_half_um,
                                                        cfg.acquisition.res0_um)
        k0 = int(round(xyz[2] / cfg.acquisition.dz_um))
        pz = np.arange(k0 - cfg.cleaning.plane_half_range, k0 + cfg.cleaning.plane_half_range + 1) * \
            cfg.acquisition.dz_um
        r = shadow.shadow_image(fs, owne, l_, t_, w_, h_, pz, cfg, include=None)
        segs, _ = shadow.neighbour_indices(fs, owne)
        for m, q in enumerate(segs):
            peaks[(e, int(q))] = B * r.per_neighbour_peak[m]
    seg_of = {int(q): tuple(int(x) for x in fs.seg_ids[q]) for q in range(fs.seg_ids.shape[0])}
    peak_B = max(pk for (e, q), pk in peaks.items() if e == 7 and seg_of[q][0] in range(13, 24))
    peak_C = max(pk for (e, q), pk in peaks.items() if seg_of[q][0] in range(24, 35))
    assert peak_C < peak_B, (peak_C, peak_B)
    G = 0.5 * (peak_B + peak_C)
    w = NI.work_list(ex, fs, swc, cfg, G, B)
    assert w[:2] == ex, w
    assert all(13 <= n <= 23 or 2 <= n <= 12 for n in w), ("no node of the weak, defocused dendrite C", w)
    assert any(13 <= n <= 23 for n in w[2:]), ("B's nodes join for A", w)
    assert len(set(w)) == len(w)
    assert not set(w) & {2, 3, 11, 12, 13, 14, 22, 23}, ("no near_end node (D-138.2)", w)
    lo = NI.work_list(ex, fs, swc, cfg, 0.0, B)
    assert any(24 <= n <= 34 for n in lo), ("with the threshold at 0 C's non-near_end nodes join", lo)
    capped = NI.work_list(ex, fs, swc, dataclasses.replace(cfg, cleaning=dataclasses.replace(
        cfg.cleaning, work_max_nodes=5)), G, B)
    assert capped[:2] == ex and len(capped) == min(5, len(w)) and capped == w[:len(capped)], (capped, w)
    _CACHE["two_node"] = (node, info, pilot, ob, t, scene, own)


def _noise_free_tube(cfg, centre, d_true, mu_true, ks):
    """(provider, ks) of a noise-free rendering of one flat tube along x (Block 4, no camera chain): 210 tau."""
    res0, dz = cfg.acquisition.res0_um, cfg.acquisition.dz_um
    tube = geometry.Tube(tuple(centre), 0.5 * d_true, 0.0, 0.0, 1.0, 14.0, "axial")
    left, top = int(math.floor(-0.5 / res0)), int(math.floor(-10.5 / res0))
    width = height = int(math.ceil(21.0 / res0))
    f = render.fine_factor(d_true, cfg.renderer, res0)
    grid, inner = render.block_fine_grid(left, top, width, height, res0, f, 8)
    out = render.render_transmittance(tube, mu_true, ks * dz, grid, cfg.renderer, +1,
                                      reduce=lambda pl: render.pixel_integrate(pl[inner[0], inner[1]], f))
    block = cfg.renderer.background_B_gl * np.asarray(out.tau, dtype=float)
    return block_provider(block, ks, left, top, res0)


def test_trace_offset():
    cfg = fast_cfg()
    cfg = dataclasses.replace(cfg, cleaning=dataclasses.replace(cfg.cleaning, refine_max_nfev=200, max_rounds=1))
    true_off = np.array([0.0, 0.3, 0.5])                    # the rendered tube minus the trace (um)
    trace_z = 0.05
    provider = _noise_free_tube(cfg, (10.0, true_off[1], trace_z + true_off[2]), 0.8, 1.0, np.arange(-5, 9))
    tmp = tempfile.mkdtemp(prefix="b36_off_")
    swc = write_swc(os.path.join(tmp, "off.swc"), [((0.0, 0.0, trace_z), (20.0, 0.0, trace_z), 11, 0.3)])
    full, i_full, ob, t = node_of(swc, cfg, 7)
    pilot = measure_pilot(full, i_full, provider, cfg)
    scene = cell.cell_scene(swc, (2, 3, 4), cfg.acquisition.res0_um)
    own = NI.own_ids_within(ob, t, cfg.inference.tube_half_length_um)[0]
    # the first background from the true features (d 0.8 um, the displacement): on a noise-free image the
    # residuals elsewhere are exactly 0, so their MAD is 0 and the trim would keep only the exact pixels
    tf = node_features.initial_features(swc, (2, 3, 4), 1.0)
    tf = node_features.with_rows(tf, [dict(node_id=int(i), d_um=0.8, offset_um=true_off) for i in tf.node_id], 0)
    node, info = NI.build_node(ob, t, pilot, provider, scene, cfg, own, node_features.featured_scene(scene, tf))
    assert node is not None, info
    feats = node_features.initial_features(swc, (2, 3, 4), 1.0)
    feats = node_features.with_rows(feats, [dict(node_id=7, d_um=pilot.d_hat_um, mu_per_um=pilot.mu_hat_per_um,
                                                 source="pilot")], 0)
    pf = NI.predict_factory({7: pilot}, {7: info["U_axis_um"]}, cfg, None)
    t0 = time.perf_counter()
    res = clean_iteration.run_rounds(feats, scene, [node], cfg, 1.0, pf)                       # fixed background
    t_fixed = time.perf_counter() - t0
    t0 = time.perf_counter()
    res = clean_iteration.run_rounds(feats, scene, [node], cfg, 1.0, pf,
                                     background_fn=NI.background_factory(cfg))                  # refitted (amendment 4)
    t_refit = time.perf_counter() - t0
    print("    one node, one round: %.1f s with the background fixed, %.1f s refitted through the model"
          % (t_fixed, t_refit))
    _CACHE["offset"] = (cfg, feats, scene, node, pilot, info, res)
    k = res.features.index_of()[7]
    off = res.features.offset_um[k]
    lat = float(np.hypot(off[0] - true_off[0], off[1] - true_off[1]))
    dep = abs(float(off[2]) - true_off[2])
    fs = node_features.featured_scene(scene, res.features)
    q = int(np.flatnonzero(np.asarray(fs.seg_ids)[:, 0] == 7)[0])
    drawn = fs.seg_xyz[q, 3:]
    lat_d = abs(float(drawn[1]) - true_off[1])
    dep_d = abs(float(drawn[2]) - (trace_z + true_off[2]))
    print("    displaced trace (0.3 um lateral, 0.5 um deep): pilot start v_t %.4f, dz %.4f; refined offset %s "
          "(d %.4f um, %s): errors %.4f um lateral, %.4f um depth; drawn as a neighbour at y %.4f, z %.4f "
          "(rendered 0.3, 0.55)" % (node.theta0[2], node.theta0[3], np.round(off, 4).tolist(),
                                    res.features.d_um[k], res.fits[7]["status"], lat, dep, drawn[1], drawn[2]))
    assert lat <= 0.02 and dep <= 0.05, (off, true_off)
    assert lat_d <= 0.02 and dep_d <= 0.05, drawn
    # the offset is v0 y_hat: along x only v0 sin(theta_pilot), theta_pilot ~ 0 on this flat x-aligned tube
    check_close("drawn_x_on_trace", drawn[0], ob.xyz_um[t][0], rtol=0, atol=1e-6)


def test_noise_free_fit():
    cfg = fast_cfg()
    res0, dz = cfg.acquisition.res0_um, cfg.acquisition.dz_um
    d_true, mu_true = 0.8, 1.0
    tmp = tempfile.mkdtemp(prefix="b36_nf_")
    swc = write_swc(os.path.join(tmp, "one.swc"), [((0.0, 0.0, 0.05), (20.0, 0.0, 0.05), 11, 0.3)])
    tube = geometry.Tube((10.0, 0.0, 0.05), 0.5 * d_true, 0.0, 0.0, 1.0, 14.0, "axial")
    left, top = int(math.floor(-0.5 / res0)), int(math.floor(-10.5 / res0))
    width = height = int(math.ceil(21.0 / res0))
    ks = np.arange(-6, 7)
    f = render.fine_factor(d_true, cfg.renderer, res0)
    grid, inner = render.block_fine_grid(left, top, width, height, res0, f, 8)
    out = render.render_transmittance(tube, mu_true, ks * dz, grid, cfg.renderer, +1,
                                      reduce=lambda pl: render.pixel_integrate(pl[inner[0], inner[1]], f))
    block = cfg.renderer.background_B_gl * np.asarray(out.tau, dtype=float)       # noise-free, no camera chain
    provider = block_provider(block, ks, left, top, res0)
    full, i_full, ob, t = node_of(swc, cfg, 7)
    pilot = measure_pilot(full, i_full, provider, cfg)
    scene = cell.cell_scene(swc, (2, 3, 4), res0)
    node, info = NI.build_node(ob, t, pilot, provider, scene, cfg, NI.own_ids_within(ob, t, 5.0)[0],
                               fscene_of(swc, scene, d=d_true))
    assert node is not None, info
    prof = clean_iteration._profiles(np.asarray(node.block, dtype=float), node, cfg)
    fwd = NI.forward_factory(pilot, info["U_axis_um"], node, cfg, line_fast.SectionCache())
    th0 = [pilot.d_hat_um, pilot.mu_hat_per_um, 0.0, pilot.z_sub_um]
    icfg = dataclasses.replace(cfg, inference=dataclasses.replace(cfg.inference, max_nfev=200))
    fit = node_inference.fit_node(prof, node.B_gl, 1.0, fwd, th0, icfg)    # noise-free: sigma only scales chi2
    rel = abs(fit.theta[0] - d_true) / d_true
    print("    noise-free isolated tube: rendered d %.4f um, pilot %.4f um, fitted %.4f um (%.2f %%), mu %.4f, "
          "z_ax %.4f um (rendered 0.05), %s" % (d_true, pilot.d_hat_um, fit.theta[0], 100 * rel, fit.theta[1],
                                                fit.theta[3], fit.status))
    assert fit.status == "converged", fit.status
    assert rel <= 0.02, ("within 2 %% of the rendered diameter", fit.theta[0])



# ------------------------------------------------- amendments 2 to 4 ---

def _chain_case(spacing, cfg):
    """A straight tube of half-length 20 um at 30 deg (d 1 um, mu 1 /um) rendered noise-free (B = 210 gl) by
    Block 24 as a chain of segments `spacing` um long, over x in +-22 um so that every segment's blur is in
    the image; its SWC (a transparent soma on the axis 1 spacing beyond the first node); the middle node."""
    res0, dz = cfg.acquisition.res0_um, cfg.acquisition.dz_um
    phi = math.radians(30.0)
    a = np.array([math.cos(phi), 0.0, math.sin(phi)])
    L = 20.0
    s_nodes = np.linspace(-L, L, int(round(2 * L / spacing)) + 1)
    s_soma = -L - spacing
    tubes = [geometry.Tube(tuple(0.5 * (s0 + s1) * a), 0.5, phi, 0.0, 1.0, 0.5 * (s1 - s0), "axial")
             for s0, s1 in zip(np.r_[s_soma, s_nodes[:-1]], s_nodes)]
    left, top = int(math.floor(-22.0 / res0)), int(math.floor(-10.5 / res0))
    width, height = int(math.ceil(44.0 / res0)), int(math.ceil(21.0 / res0))
    f = render.fine_factor(1.0, cfg.renderer, res0)
    grid, inner = render.block_fine_grid(left, top, width, height, res0, f, 8)
    ks = np.arange(-4, 5)
    out = scene_render.render_scene(tubes, [1.0] * len(tubes), ks * dz, grid, cfg.renderer, +1,
                                    reduce=lambda pl: render.pixel_integrate(pl[inner[0], inner[1]], f))
    provider = block_provider(210.0 * np.asarray(out.tau, dtype=float), ks, left, top, res0)
    tmp = tempfile.mkdtemp(prefix="b36_chain_")
    lines = ["# chain", "1 1 %.6f %.6f %.6f 0.5 -1" % tuple(s_soma * a)]
    for j, sj in enumerate(s_nodes):
        q = sj * a
        lines.append("%d 3 %.6f %.6f %.6f 0.5 %d" % (j + 2, q[0], q[1], q[2], j + 1))
    with open(os.path.join(tmp, "c.swc"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    swc = swc_io.read_swc(os.path.join(tmp, "c.swc"))
    mid = int(np.argmin(np.abs(s_nodes))) + 2
    pilot = types.SimpleNamespace(cx_um=0.0, cy_um=0.0, theta_rad=0.0, phi_rad=phi, z_sub_um=0.0, k_star=0,
                                  d_hat_um=1.0, mu_hat_per_um=1.0, B_bar=210.0, fit_status="converged")
    return swc, provider, mid, pilot


def test_chain_background_and_tube_cut():
    """Amendments 2 (F1), 3 (1) and 4 (1): the 30 deg chain at nodes 1 and 2 um apart."""
    cfg = default_config()
    cfg = dataclasses.replace(
        cfg, renderer=dataclasses.replace(cfg.renderer, absorption="partition_vertical", dzeta_um=0.05),
        cleaning=dataclasses.replace(cfg.cleaning, shadow_absorption="partition_vertical", soma_mu_per_um=0.0),
        inference=dataclasses.replace(cfg.inference, absorption="partition_vertical"))
    assert cfg.cleaning.background_support == "model"
    for spacing in (1.0, 2.0):
        swc, provider, mid, pilot = _chain_case(spacing, cfg)
        scene = cell.cell_scene(swc, (2, 3, 4), cfg.acquisition.res0_um)
        full, i_full, ob, t = node_of(swc, cfg, mid)
        own = NI.own_ids_within(ob, t, cfg.inference.tube_half_length_um)[0]
        fs = fscene_of(swc, scene)                             # true features: d 1 um (Allen radius 0.5), mu 1
        t0 = time.perf_counter()
        node, info = NI.build_node(ob, t, pilot, provider, scene, cfg, own, fs)
        t_build = time.perf_counter() - t0
        assert node is not None, ("the 30 deg chain is not refused (amendment 4)", info)
        err = node.B_gl - 210.0
        big, info25 = NI.build_node(ob, t, pilot, provider, scene, cfg, own, fscene_of(swc, scene, d=1.25))
        err25 = (big.B_gl - 210.0) if big is not None else None
        # one cleaning round of the tier-1 neighbours (the stretch beyond the own nodes included)
        tiers = cleaning.neighbour_tiers(fs, own, node.node_xyz, cfg)
        S = shadow.shadow_image(fs, own, node.left, node.top, node.block.shape[2], node.block.shape[1],
                                node.planes_z, cfg, B_gl=float(np.max(node.B_gl)), include=(tiers == 1)).S
        cr = cleaning.clean_block(node.block, S, node.B_gl, cfg)
        before = clean_iteration._profiles(np.asarray(node.block, dtype=float), node, cfg)
        after = clean_iteration._profiles(cr.block, node, cfg)
        Ua = info["U_axis_um"]
        tau = line_fast.line_profiles(NI.local_tube(pilot, node.node_xyz, 1.0, 0.0, 0.0, Ua[0], Ua[1]), 1.0,
                                      node.ks, node.origin_xy, node.y_hat, node.e_u, node.v, node.n_avg, node.step,
                                      cfg)
        model = node.B_gl[:, None] * tau
        dip = np.max(node.B_gl[:, None] * (1.0 - tau), axis=1)
        short_before = np.max(np.abs(before - model), axis=1) / dip
        short_after = np.max(np.abs(after - model), axis=1) / dip
        print("    chain, nodes %.0f um apart: %d own nodes, tube -%.2f..+%.2f um, build %.1f s" %
              (spacing, len(own), Ua[0], Ua[1], t_build))
        print("      B_k - 210 (true features)  %s; support share %s" % (np.round(err, 3).tolist(),
                                                                         np.round(info["support_frac"], 3).tolist()))
        print("      B_k - 210 (d 25 %% large)   %s; support share %s" % (
            None if err25 is None else np.round(err25, 3).tolist(), np.round(info25.get("support_frac"), 3).tolist()))
        print("      shortfall / dip, uncleaned %s" % np.round(short_before, 4).tolist())
        print("      shortfall / dip, cleaned   %s" % np.round(short_after, 4).tolist())
        assert np.max(np.abs(err)) <= 0.5, ("B_k within 0.5 gl of 210", err)
        assert np.all(short_before > 0.01), ("the tester's shortfall reproduced (> 1 %)", short_before)
        assert np.all(short_after <= 0.015), ("the cleaned profiles within 1.5 % of the dip", short_after)


def test_noise_factor_and_sd():
    """Amendment 2 (F2): profile_noise_factor against 400 white-noise images [3 %]; amendment 4: the plane's
    noise SD from model_background on white noise of SD 3 gl within 10 % of 3."""
    cfg = default_config()
    m, p = cfg.measure, cfg.acquisition.res0_um
    v, n_avg, step = profiles.profile_offsets(m), profiles.n_along(m), float(m.profile_step_um)
    th = 0.3
    yh, eu = np.array([-math.sin(th), math.cos(th)]), np.array([math.cos(th), math.sin(th)])
    frame = CropFrame(0, 0, 0, p)
    o = np.array([60 * p + 0.031, 60 * p + 0.047])
    fac = NI.profile_noise_factor(frame, (120, 120), o, yh, eu, v, n_avg, step)
    rng = np.random.default_rng(SEED)
    prof = np.array([profiles.sample_profile(rng.normal(0.0, 3.0, (120, 120)), frame, o, yh, eu, v, n_avg, step)
                     for _ in range(400)])
    emp = float(np.sqrt(np.mean(prof ** 2)))
    print("    noise factor %.4f: profile SD on 400 white-noise images %.4f gl, predicted %.4f gl (%+.2f %%)"
          % (fac, emp, 3.0 * fac, 100 * (emp / (3.0 * fac) - 1)))
    assert abs(emp / (3.0 * fac) - 1) <= 0.03       # 400 x 61 samples: the SD's sampling error is ~0.5 %
    # the model background's noise SD on white noise of SD 3 gl over a flat 210 with a tube in the patch
    S = np.zeros((3, 120, 120))
    S[:, 55:65, :] = 0.4                             # a band of shadow, outside the support
    img = 210.0 * (1.0 - S) + rng.normal(0.0, 3.0, S.shape)
    B, sd, frac, ok = NI.model_background(img, S, frame, o, cfg)
    print("    model_background on white noise SD 3 gl: B %s, SD %s, support %s"
          % (np.round(B, 3).tolist(), np.round(sd, 3).tolist(), np.round(frac, 3).tolist()))
    assert np.all(ok) and np.all(np.abs(sd - 3.0) <= 0.3), sd
    assert np.all(np.abs(B - 210.0) <= 0.5), B


def test_chi2_coverage():
    """Amendment 2 (F2): a rendered tube with seeded white noise of SD 3 gl (no camera chain): mean chi2_k/N
    within 0.8..1.25 with the profile's SD, d_hat +- SE covers the rendered d; the SEs with the pixel's SD
    (before F2) reported beside them."""
    cfg = fast_cfg()
    cfg = dataclasses.replace(cfg, inference=dataclasses.replace(cfg.inference, max_nfev=200))
    res0, dz = cfg.acquisition.res0_um, cfg.acquisition.dz_um
    d_true = 0.8
    ks = np.arange(-6, 7)
    clean = _noise_free_tube(cfg, (10.0, 0.0, 0.05), d_true, 1.0, ks)
    rng = np.random.default_rng(SEED + 1)
    cache_img = {}

    def noisy(*args):
        b, k, v, fr = clean(*args)
        key = args
        if key not in cache_img:
            cache_img[key] = np.asarray(b, dtype=float) + rng.normal(0.0, 3.0, np.shape(b))
        return cache_img[key], k, v, fr
    tmp = tempfile.mkdtemp(prefix="b36_chi_")
    swc = write_swc(os.path.join(tmp, "one.swc"), [((0.0, 0.0, 0.05), (20.0, 0.0, 0.05), 11, 0.3)])
    full, i_full, ob, t = node_of(swc, cfg, 7)
    pilot = measure_pilot(full, i_full, noisy, cfg)
    scene = cell.cell_scene(swc, (2, 3, 4), res0)
    node, info = NI.build_node(ob, t, pilot, noisy, scene, cfg, NI.own_ids_within(ob, t, 5.0)[0],
                               fscene_of(swc, scene, d=d_true))
    assert node is not None, info
    prof = clean_iteration._profiles(np.asarray(node.block, dtype=float), node, cfg)
    fwd = NI.forward_factory(pilot, info["U_axis_um"], node, cfg, line_fast.SectionCache())
    th0 = [pilot.d_hat_um, pilot.mu_hat_per_um, 0.0, pilot.z_sub_um]
    fit = node_inference.fit_node(prof, node.B_gl, info["sigma_gl"], fwd, th0, cfg)
    old = node_inference.fit_node(prof, node.B_gl, info["sigma_pixel_gl"], fwd, th0, cfg)
    chi = float(np.nanmean(fit.chi2_per_plane))
    print("    white noise SD 3 gl: pixel SD %.3f gl, factor %.4f, profile SD %.3f gl; fitted d %.4f +- %.4f um "
          "(rendered %.1f), mean chi2/N %.3f; with the pixel SD (before F2): SE %.4f um, chi2/N %.3f"
          % (info["sigma_pixel_gl"], info["noise_factor"], info["sigma_gl"], fit.theta[0], fit.se[0], d_true, chi,
             old.se[0], float(np.nanmean(old.chi2_per_plane))))
    assert 0.8 <= chi <= 1.25, chi
    assert abs(fit.theta[0] - d_true) <= fit.se[0], (fit.theta[0], fit.se[0])


def test_background_spy_f4_f5():
    """Amendment 4: background_factory through run_rounds on two rounds -- the second round's B_k is computed
    from the first round's features [spy]; amendment 2 F4 (infer's start) and F5 (the cap) by construction."""
    if "offset" not in _CACHE:
        test_trace_offset()
    cfg, feats, scene, node, pilot, info, res1 = _CACHE["offset"]
    cfg2 = dataclasses.replace(cfg, cleaning=dataclasses.replace(cfg.cleaning, max_rounds=2, stop_rel_d=1e-12))
    pf = NI.predict_factory({7: pilot}, {7: info["U_axis_um"]}, cfg2, None)
    bg = NI.background_factory(cfg2)
    seen = []

    def spy(nd, F):
        seen.append(F)
        return bg(nd, F)
    res2 = clean_iteration.run_rounds(feats, scene, [node], cfg2, 1.0, pf, background_fn=spy)
    assert res2.rounds == 2 and len(seen) == 2
    F1 = node_features.featured_scene(scene, res1.features)       # the features after one round
    assert_array_equal(seen[1].seg_xyz, F1.seg_xyz)
    assert_array_equal(seen[1].seg_d_um, F1.seg_d_um)
    from allen_image_io import CropFrame as _CF
    want = NI.model_background(node.block, NI.all_shadow(F1, node, cfg2),
                               _CF(node.left, node.top, 0, cfg2.acquisition.res0_um), node.origin_xy, cfg2)[0]
    h = bg.history[7]
    assert_array_equal(h[1]["B_fit_gl"], want)                     # round 2's fit is the round-1 features'
    assert_array_equal(res2.B_gl[7], h[1]["B_gl"])                 # and it is what the round used
    if h[1]["kept_previous"]:
        assert_array_equal(h[1]["B_gl"], h[0]["B_gl"])
    else:
        assert_array_equal(h[1]["B_gl"], want)
    print("    two rounds: B_k round 1 %s (kept previous: %s), round 2 fit %s (kept previous: %s)"
          % (np.round(h[0]["B_gl"], 3).tolist(), h[0]["kept_previous"], np.round(h[1]["B_fit_gl"], 3).tolist(),
             h[1]["kept_previous"]))
    # F4: a row not refined starts at pilot_start (z_ax = z_t + (z_sub - z_t)); a refined row at its features
    import infer_nodes
    p = fake_pilot()
    trace = np.array([1.1, -0.4, -0.2])
    st, which = infer_nodes.fit_start("pilot", 0.6, 0.9, np.zeros(3), p, trace)
    check_close("f4_pilot", st, [p.d_hat_um, p.mu_hat_per_um, NI.trace_offset(p, trace), p.z_sub_um], rtol=0,
                atol=1e-12)
    assert which == "pilot_start"
    yh = np.array([-math.sin(p.theta_rad), math.cos(p.theta_rad)])
    st, which = infer_nodes.fit_start("refined", 0.6, 0.9, np.array([0.2 * yh[0], 0.2 * yh[1], 0.3]), p, trace)
    check_close("f4_refined", st, [0.6, 0.9, 0.2, trace[2] + 0.3], rtol=0, atol=1e-12)
    assert which == "features"
    # F5: the work list holds at most work_max_nodes nodes, the examples first
    cfg3, swc3, _prov, _b = two_tube_case()
    scene3 = cell.cell_scene(_CACHE["three"], (2, 3, 4), cfg3.acquisition.res0_um)
    fs3 = fscene_of(_CACHE["three"], scene3)
    one = dataclasses.replace(cfg3, cleaning=dataclasses.replace(cfg3.cleaning, work_max_nodes=1))
    assert NI.work_list([7, 18], fs3, _CACHE["three"], one, 3.0, 210.0) == [7]

# ----------------------------------------------------------- end to end ---

def _run(args, cwd):
    cmd = [sys.executable] + args
    r = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    if r.returncode != 0:
        raise AssertionError("%s failed (%d):\n%s\n%s" % (" ".join(args), r.returncode, r.stdout[-3000:],
                                                           r.stderr[-3000:]))
    return r.stdout


def _pipeline(out):
    """The three commands with --synthetic into out; their printed output."""
    os.makedirs(out, exist_ok=True)
    cfgj = os.path.join(out, "fast_config.json")
    with open(cfgj, "w") as f:
        json.dump(FAST, f)
    sc = str(WS / "scripts" / "clean_cell.py")
    inf = str(WS / "scripts" / "infer_nodes.py")
    o1 = _run([sc, "survey", "--synthetic", "--specimen", "7", "--B", "210", "--mu", "1.0", "--out", out,
               "--config-json", cfgj], WS)
    ex = os.path.join(out, "examples_7.json")
    h0 = hashlib.sha256(open(ex, "rb").read()).hexdigest()
    o2 = _run([sc, "clean", "--synthetic", "--specimen", "7", "--examples", ex, "--out", out, "--rounds", "1",
               "--config-json", cfgj], WS)
    h1 = hashlib.sha256(open(ex, "rb").read()).hexdigest()
    assert h0 == h1, "clean changed its input"
    o3 = _run([inf, "real", "--synthetic", "--specimen", "7", "--cleaning-dir", out, "--out",
               os.path.join(out, "inf")], WS)
    return o1, o2, o3


def _e2e():
    if "e2e" not in _CACHE:
        base = tempfile.mkdtemp(prefix="b36_e2e_")
        a, b = os.path.join(base, "a"), os.path.join(base, "b")
        outs = _pipeline(a)
        _CACHE["e2e"] = (a, b, outs)
    return _CACHE["e2e"]


def test_contract():
    a, _b, outs = _e2e()
    for o in outs:
        for line in o.strip().splitlines()[-4:]:
            print("    | " + line)
    ex = json.load(open(os.path.join(a, "examples_7.json")))
    ids = ex["example_ids"]
    assert len(ids) >= 1
    rows = open(os.path.join(a, "contamination_7.csv")).read().splitlines()
    assert rows[0].split(",")[-1] == "near_end" and len(rows) > 1
    rec = json.load(open(os.path.join(a, "cleaning_7.json")))
    for key in ("cleaning", "inference", "history", "stopped", "nodes", "rounds"):
        assert key in rec, key
    assert rec["cleaning"]["max_rounds"] == 1 and rec["inference"]["absorption"] == "partition_vertical"
    feats = features_io.read_features(os.path.join(a, "features_7.csv"))
    tmpf = os.path.join(tempfile.mkdtemp(prefix="b36_rt_"), "features_rt.csv")
    features_io.write_features(tmpf, feats)
    assert open(tmpf, "rb").read() == open(os.path.join(a, "features_7.csv"), "rb").read(), "round trip"
    for nid in rec["example_ids"]:
        z = np.load(os.path.join(a, "cleaning_7_node%d.npz" % nid))
        for key in ("block", "shadow_gl", "cleaned", "ks", "B_gl", "v", "profiles_before", "profiles_after",
                    "origin_xy", "line_xy", "left", "top"):
            assert key in z.files, key
        assert os.path.getsize(os.path.join(a, "cleaning_node%d.png" % nid)) > 0
        assert os.path.getsize(os.path.join(a, "inf", "fit_node%d.png" % nid)) > 0
    assert os.path.getsize(os.path.join(a, "cleaning_rounds.png")) > 0
    inf = json.load(open(os.path.join(a, "inf", "inference_7.json")))
    assert "cleaning" in inf and "inference" in inf and set(inf["nodes"]) == {str(i) for i in rec["example_ids"]}
    lines = open(os.path.join(a, "inf", "inference_7.csv")).read().splitlines()
    assert len(lines) == 1 + len(rec["example_ids"])
    for name in ("diameters.png", "chi2.png"):
        assert os.path.getsize(os.path.join(a, "inf", name)) > 0
    import csv as _csv
    for r in _csv.DictReader(open(os.path.join(a, "inf", "inference_7.csv"))):
        print("    node %s: Allen %.3f, pilot %.3f, refined %.3f, fitted %.4f um (rendered 0.8 um), %s"
              % (r["node_id"], float(r["d_allen_um"]), float(r["d_pilot_um"]), float(r["d_refined_um"]),
                 float(r["d_hat_um"]), r["status"]))


def test_determinism():
    a, b, _outs = _e2e()
    _pipeline(b)
    for root, _dirs, files in os.walk(a):
        for name in files:
            pa = os.path.join(root, name)
            pb = os.path.join(b, os.path.relpath(pa, a))
            if name.endswith(".png") or name == "fast_config.json":
                continue
            if name.endswith(".npz"):
                za, zb = np.load(pa), np.load(pb)
                assert za.files == zb.files
                for k in za.files:
                    assert_array_equal(za[k], zb[k], err_msg="%s %s" % (name, k))
                continue
            ta, tb = open(pa).read(), open(pb).read()
            if name.startswith("inference_") and name.endswith(".csv"):
                ta = "\n".join(",".join(r.split(",")[:-1]) for r in ta.splitlines())   # the seconds column
                tb = "\n".join(",".join(r.split(",")[:-1]) for r in tb.splitlines())
            if name.endswith(".json"):
                ja, jb = json.loads(ta), json.loads(tb)
                def strip(x):                      # times (background_rounds' seconds) are not deterministic
                    if isinstance(x, dict):
                        return {k: strip(v) for k, v in x.items() if k != "seconds"}
                    if isinstance(x, list):
                        return [strip(v) for v in x]
                    return x
                ja, jb = strip(ja), strip(jb)
                for j in (ja, jb):
                    j.pop("examples_file", None)
                    j.pop("cleaning_dir", None)
                assert ja == jb, name
                continue
            assert ta == tb, name


def test_edge_cases():
    cfg = fast_cfg()
    # near_end: refused before any fetch
    s = np.array([0.0, 1.0, 2.0])
    br = Branch(np.arange(3), np.full(3, 3), np.column_stack([s, np.zeros(3), np.zeros(3)]), np.full(3, 0.3), s)

    def no_fetch(*_a):
        raise AssertionError("no fetch for a refused node")
    node, info = NI.build_node(br, 1, fake_pilot(), no_fetch, None, cfg, (0, 1, 2), None)
    assert node is None and info["reason"] == NI.REASON_NEAR_END and info["near_end"], info
    long = Branch(np.arange(3), np.full(3, 3), np.column_stack([[0.0, 10.0, 20.0], np.zeros(3), np.zeros(3)]),
                  np.full(3, 0.3), np.array([0.0, 10.0, 20.0]))
    node, info = NI.build_node(long, 1, fake_pilot(z_sub_um=float("nan")), no_fetch, None, cfg, (0, 1, 2), None)
    assert node is None and info["reason"] == NI.REASON_NO_PLANE, info
    node, info = NI.build_node(long, 1, fake_pilot(d_hat_um=float("nan")), no_fetch, None, cfg, (0, 1, 2), None)
    assert node is None and info["reason"] == NI.REASON_NO_FIT, info
    # a missing plane and a plane without background, on the two-tube scene
    cfg2, swc, provider, (blk, ks_all, left, top) = two_tube_case()
    full, i_full, ob, t = node_of(swc, cfg2, 7)
    pilot = measure_pilot(full, i_full, provider, cfg2)
    scene = cell.cell_scene(swc, (2, 3, 4), cfg2.acquisition.res0_um)
    own = NI.own_ids_within(ob, t, cfg2.inference.tube_half_length_um)[0]
    fs2 = fscene_of(swc, scene)
    kk = NI.node_planes(pilot, cfg2)

    def missing(*args):
        b, k, v, fr = provider(*args)
        v = v.copy()
        v[k == kk[-1]] = False
        return b, k, v, fr
    node, info = NI.build_node(ob, t, pilot, missing, scene, cfg2, own, fs2)
    assert node is None and info["reason"] == NI.REASON_MISSING and str(int(kk[-1])) in info["detail"], info

    def blank(*args):
        b, k, v, fr = provider(*args)
        b = np.array(b, dtype=float)
        b[k == kk[0]] = np.nan
        return b, k, v, fr
    node, info = NI.build_node(ob, t, pilot, blank, scene, cfg2, own, fs2)
    assert node is None and info["reason"] == NI.REASON_NO_BG, info
    # the inputs of build_node unchanged
    names = ("cx_um", "cy_um", "theta_rad", "phi_rad", "z_sub_um", "k_star", "d_hat_um", "mu_hat_per_um")
    before = (ob.xyz_um.copy(), ob.s_um.copy(), tuple(getattr(pilot, n) for n in names))
    NI.build_node(ob, t, pilot, provider, scene, cfg2, own, fs2)
    assert_array_equal(before[0], ob.xyz_um)
    assert_array_equal(before[1], ob.s_um)
    assert tuple(getattr(pilot, n) for n in names) == before[2]
    # ExtrusionInvalid -> NaN in predict and forward (the line 1 um from a cap of a 1.5 um tube)
    p, nd = _line_node(cfg)
    pred = NI.predict_factory({5: p}, {5: (1.5, 1.5)}, cfg, None)(nd)
    assert np.all(np.isnan(pred(0.6, 1.0, 0.0, 0.0)))
    assert np.all(np.isnan(NI.forward_factory(p, (1.5, 1.5), nd, cfg, None)(0.6, 1.0, 0.0, 0.0)))
    # work_list refusals
    feats = node_features.initial_features(swc, (2, 3, 4), 1.0)
    fs = node_features.featured_scene(scene, feats)
    for bad in (dict(example_ids=[7, 7], B_gl=210.0), dict(example_ids=[7], B_gl=0.0),
                dict(example_ids=[1], B_gl=210.0)):
        try:
            NI.work_list(bad["example_ids"], fs, swc, cfg2, 3.0, bad["B_gl"])
        except ValueError:
            continue
        raise AssertionError("work_list accepted %r" % bad)
    try:
        NI.tube_extent(long, 5, cfg)
    except ValueError:
        pass
    else:
        raise AssertionError("tube_extent accepted an index outside the branch")


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
    for path in _SAVED_ARTIFACTS:
        print(f"-- arrays from a failed comparison: {path}")
    failed = counts["FAIL"] + counts["ERROR"] + counts["TODO"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
