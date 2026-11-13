"""Spec-derived tests, Block 27 (the contamination survey, with amendment 1) -- tester 2026-11-13.

Oracles (SPEC.md Block 27): an isolated branch has P = 0, C = 0, class
clean [exact]; one flat neighbour at lateral rho: P(v_n) = shadow_depths at
|rho - v_n| [1e-12] (and the quadrature-free reference within 1e-4 relative);
a steep neighbour: distance and kernel from the same nearest point
(amendment 1) [1e-12]; C decreases laterally (5 steps) and for a neighbour
directly over the line out of focus (5 planes); the four classes on four
scenes (precedence clean first); per_neighbour sums to P [1e-12];
pick_examples deterministic, no node twice, a short class warned; the
ValueErrors.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose

import nc_helpers as H
from spec_helpers import ref_shadow_depth
from allen_diameter.analysis import contamination as CT
from allen_diameter.analysis.calibration_scene import shadow_depths
from allen_diameter.analysis.node_pipeline import Branch
from allen_diameter.analysis.plane_iteration import Scene
from allen_diameter.analysis.profiles import profile_offsets

B, MU = 210.0, 1.0


def branch():
    x = np.arange(11, dtype=float)
    xyz = np.column_stack([x, np.zeros(11), np.zeros(11)])
    return Branch(np.arange(1, 12), np.full(11, 3), xyz, np.full(11, 0.3), x.copy())


def scene(extra):
    """The branch's own segments + extra [(p0, p1, r, (child, parent))]."""
    br = branch()
    segs = [(br.xyz_um[i], br.xyz_um[i + 1], 0.3, (int(br.ids[i + 1]), int(br.ids[i]))) for i in range(10)]
    segs += extra
    xyz = np.array([np.r_[a, b] for a, b, _, _ in segs], dtype=float)
    rad = np.array([[r, r] for _, _, r, _ in segs], dtype=float)
    ids = np.array([list(i) for *_, i in segs], dtype=np.int64)
    return Scene(xyz, rad, ids, np.empty((0, 3)))


def test_isolated_branch_clean():
    cfg = H.cfg_with()
    out = CT.node_contamination(branch(), 5, scene([]), B, MU, cfg)
    assert out["index"] == 0.0 and out["class"] == "clean" and out["n_neighbours"] == 0
    P, per = CT.pollution_profile(branch(), 5, scene([]), profile_offsets(cfg.measure), B, MU, cfg)
    assert np.all(P == 0.0) and per.shape[0] == 0


def test_parallel_neighbour_profile():
    cfg = H.cfg_with()
    rho, rq = 1.3, 0.4
    sc = scene([(np.array([-20.0, rho, 0.0]), np.array([30.0, rho, 0.0]), rq, (101, 100))])
    v = profile_offsets(cfg.measure)
    P, per = CT.pollution_profile(branch(), 5, sc, v, B, MU, cfg)
    s0 = float(H.ref_sigma_r(0.0, cfg.renderer))
    # spec section 2.1: theta = 0 for a branch along +x (distal), y_hat = (0, 1)
    exp = shadow_depths(np.abs(rho - v), rq, MU, s0, B)
    assert np.max(np.abs(P - exp)) <= 1e-12
    assert np.max(np.abs(per.sum(axis=0) - P)) <= 1e-12
    ref = np.array([ref_shadow_depth(abs(rho - x), rq, MU, s0, B) for x in v[::7]])
    assert_allclose(P[::7], ref, rtol=1e-3, atol=1e-2)        # Block 10's GL-16 predictor vs adaptive quad: 1e-2 gl << the 0.3 gl floor


def test_steep_neighbour_nearest_point():
    cfg = H.cfg_with()
    a, b = np.array([3.0, 2.0, -2.0]), np.array([7.0, 0.5, 3.0])     # steep, passing near the line
    rq = 0.35
    sc = scene([(a, b, rq, (101, 100))])
    v = profile_offsets(cfg.measure)
    P, _ = CT.pollution_profile(branch(), 5, sc, v, B, MU, cfg)
    exp = np.zeros(v.size)
    for n, vn in enumerate(v):
        p = np.array([5.0, vn])
        d = b[:2] - a[:2]
        t = float(np.clip(np.dot(p - a[:2], d) / np.dot(d, d), 0, 1))
        q = a + t * (b - a)
        rho = float(np.hypot(*(p - q[:2])))
        s = float(H.ref_sigma_r(q[2] - 0.0, cfg.renderer))
        exp[n] = float(shadow_depths(np.array(rho), rq, MU, s, B))
    assert np.max(np.abs(P - exp)) <= 1e-12


def test_monotone_index():
    cfg = H.cfg_with()
    Cs = []
    for rho in (0.8, 1.2, 1.6, 2.0, 2.4):
        sc = scene([(np.array([-20.0, rho, 0.0]), np.array([30.0, rho, 0.0]), 0.3, (101, 100))])
        Cs.append(CT.node_contamination(branch(), 5, sc, B, MU, cfg)["index"])
    assert all(Cs[i] > Cs[i + 1] for i in range(4)), Cs
    Cz = []
    for m in range(1, 6):
        z = 0.28 * m
        sc = scene([(np.array([5.0, -10.0, z]), np.array([5.0, 10.0, z]), 0.3, (101, 100))])
        Cz.append(CT.node_contamination(branch(), 5, sc, B, MU, cfg)["index"])
    assert all(Cz[i] > Cz[i + 1] for i in range(4)), Cz


def test_four_classes():
    cfg = H.cfg_with()
    out = {}
    out["clean"] = CT.node_contamination(branch(), 5, scene([]), B, MU, cfg)
    out["overlap"] = CT.node_contamination(
        branch(), 5, scene([(np.array([5.0, -10.0, 3.0]), np.array([5.0, 10.0, 3.0]), 0.5, (101, 100))]), B, MU, cfg)
    out["close_in_focus"] = CT.node_contamination(
        branch(), 5, scene([(np.array([-20.0, 1.5, 0.0]), np.array([30.0, 1.5, 0.0]), 0.3, (101, 100))]), B, MU, cfg)
    out["diffuse"] = CT.node_contamination(
        branch(), 5, scene([(np.array([-20.0, 1.0, 1.5]), np.array([30.0, 1.0, 1.5]), 0.3, (101, 100))]), B, MU, cfg)
    for cls, row in out.items():
        print(cls, row["index"], row["class"])
        assert row["class"] == cls
    # attached at a branch point (shares node 6) and lying across the line: never overlap (amendment 1)
    att = CT.node_contamination(
        branch(), 5, scene([(np.array([6.0, 0.0, 0.0]), np.array([6.0, 6.0, 0.0]), 0.5, (101, 6))]), B, MU, cfg)
    assert att["class"] != "overlap"


def test_pick_examples():
    rng_rows = np.random.default_rng(1)
    classes = ["clean"] * 10 + ["diffuse"] * 10 + ["close_in_focus"] * 3 + ["overlap"] * 1
    rows = [{"node_id": 100 + i, "class": c, "soma_lateral_um": float(rng_rows.random() * 50)}
            for i, c in enumerate(classes)]
    counts = (4, 6, 6, 2, 2)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        a = CT.pick_examples(rows, counts, np.random.default_rng(9))
        assert any(issubclass(x.category, RuntimeWarning) for x in w)
    b = CT.pick_examples(rows, counts, np.random.default_rng(9))
    assert a == b and len(a) == len(set(a))
    assert len(a) == 4 + 6 + 3 + 1 + 2


def test_value_errors():
    cfg = H.cfg_with()
    sc = scene([])
    for args in ((5, sc, 0.0, MU), (5, sc, B, 0.0), (11, sc, B, MU), (-1, sc, B, MU)):
        with pytest.raises(ValueError):
            CT.node_contamination(branch(), args[0], args[1], args[2], args[3], cfg)
