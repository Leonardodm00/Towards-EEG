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


# ---------------------------------------------------------------------------------------------------------------
# Tester 2026-11-22 (incremental at a749cc9): Block 27, amendments 3 and 4 -- pick_clear_examples. Oracles: the
# amendments' rule (not near_end, n_tier2 = 0 and worst2_peak_gl = 0, soma_lateral_um >= min_soma_um; increasing
# worst1_peak_gl, ties by decreasing own_dip_gl then node id; skipped within min_separation_um (3-D) of a node
# taken in this call or of taken_xyz; until n), written out by hand below on rows built by hand.
# ---------------------------------------------------------------------------------------------------------------

def _clear_row(nid, x, peak, dip=50.0, near_end=False, n2=0, p2=0.0, soma=30.0, y=0.0, z=0.0):
    return dict(node_id=nid, x_um=float(x), y_um=float(y), z_um=float(z), own_dip_gl=float(dip), n_tier2=n2,
                n_evaluated=3, worst1_range_gl=0.0, worst1_peak_gl=float(peak), worst1_child_id=-1,
                worst1_lateral_um=float("nan"), worst1_defocus_um=float("nan"), worst2_peak_gl=float(p2),
                soma_lateral_um=float(soma), near_end=near_end)


def _ref_pick(rows, n, sep, min_soma=0.0, taken=()):
    ok = [r for r in rows if not r["near_end"] and r["n_tier2"] == 0 and r["worst2_peak_gl"] == 0
          and r["soma_lateral_um"] >= min_soma]
    ok = sorted(ok, key=lambda r: (r["worst1_peak_gl"], -r["own_dip_gl"], r["node_id"]))
    pos = [np.asarray(t, float) for t in taken]
    out = []
    for r in ok:
        if len(out) >= n:
            break
        p = np.array([r["x_um"], r["y_um"], r["z_um"]])
        if any(np.linalg.norm(p - q) < sep for q in pos):
            continue
        out.append(r["node_id"])
        pos.append(p)
    return out


def test_pick_clear_examples_rule_by_hand():
    rows = [_clear_row(1, 0, 0.30), _clear_row(2, 20, 0.10), _clear_row(3, 40, 0.10, dip=70.0),
            _clear_row(4, 60, 0.05, near_end=True), _clear_row(5, 80, 0.01, n2=1), _clear_row(6, 100, 0.02, p2=0.4),
            _clear_row(7, 120, 0.00, soma=5.0), _clear_row(8, 3, 0.00), _clear_row(9, 140, 0.10, dip=70.0),
            _clear_row(10, 160, 0.20)]
    before = [dict(r) for r in rows]
    got = CT.pick_clear_examples(rows, 10, 10.0, min_soma_um=10.0)
    # 8 first (peak 0); 3 and 9 tie at 0.10 with dip 70 -> id order; then 2; 10 at 0.20; 1 at 0.30 is within
    # 10 um of 8 (3 um apart) -> skipped; 4 near_end, 5 a tier-2 neighbour, 6 a tier-2 peak, 7 the soma: out
    assert got == [8, 3, 9, 2, 10] == _ref_pick(rows, 10, 10.0, 10.0)
    assert rows == before
    assert CT.pick_clear_examples(rows, 0, 10.0) == []
    assert CT.pick_clear_examples(rows, 2, 10.0, min_soma_um=10.0) == [8, 3]
    # taken_xyz: a node within the distance of a position taken before is skipped
    got = CT.pick_clear_examples(rows, 10, 10.0, min_soma_um=10.0, taken_xyz=[(42.0, 0.0, 0.0)])
    assert got == _ref_pick(rows, 10, 10.0, 10.0, taken=[(42.0, 0.0, 0.0)]) and 3 not in got
    assert CT.pick_clear_examples(rows, 10, 10.0, 10.0, taken_xyz=()) == CT.pick_clear_examples(rows, 10, 10.0, 10.0)
    for bad in (dict(n=-1), dict(min_separation_um=-1.0), dict(min_separation_um=float("nan")),
                dict(min_soma_um=float("inf")), dict(taken_xyz=[(1.0, 2.0)]), dict(taken_xyz=[(1.0, np.nan, 0.0)])):
        kw = dict(rows=rows, n=3, min_separation_um=10.0)
        kw.update(bad)
        with pytest.raises(ValueError):
            CT.pick_clear_examples(**kw)


def test_pick_clear_examples_on_random_rows_against_reference():
    rng = np.random.default_rng(9)
    for _ in range(50):
        rows = [_clear_row(i, rng.uniform(0, 60), round(rng.uniform(0, 1), 1), dip=round(rng.uniform(10, 90), -1),
                           near_end=bool(rng.random() < 0.2), n2=int(rng.random() < 0.2),
                           p2=float(rng.random() < 0.1), soma=rng.uniform(0, 40), y=rng.uniform(0, 20))
                for i in rng.permutation(40)]
        n, sep, ms = int(rng.integers(0, 12)), float(rng.uniform(0, 15)), float(rng.uniform(0, 20))
        assert CT.pick_clear_examples(rows, n, sep, ms) == _ref_pick(rows, n, sep, ms)


@pytest.mark.xfail(strict=True, reason="2026-11-22 report F2 (spec gap): a NaN soma_lateral_um fails the "
                   "comparison with min_soma_um, so a cell without a soma disc gets no clear node")
def test_pick_clear_examples_soma_distance_nan_without_a_soma():
    """Block 27: soma_lateral_um is NaN without a soma. The rule "at least min_soma_um from the soma" is silent
    on NaN; with min_soma_um = 0 (the function's default) a cell without a soma disc gets no clear node."""
    rows = [_clear_row(1, 0, 0.0, soma=float("nan")), _clear_row(2, 20, 0.0, soma=float("nan"))]
    assert CT.pick_clear_examples(rows, 2, 10.0, min_soma_um=0.0) == [1, 2]

# Tester, 2026-11-22 (incremental run at 97f5104): Block 27 amendment 5 (D-212.2, F2 and S2 of the final check of
# wave 4): in pick_clear_examples a NaN soma_lateral_um (no soma disc) passes any min_soma_um; the other rules of
# amendments 3-4 unchanged. Rows written by hand.

def _row(nid, peak1, soma, x=0.0, own=10.0, tier2=0, peak2=0.0, near_end=False):
    return {"node_id": nid, "worst1_peak_gl": peak1, "own_dip_gl": own, "n_tier2": tier2, "worst2_peak_gl": peak2,
            "soma_lateral_um": soma, "near_end": near_end, "x_um": x, "y_um": 0.0, "z_um": 0.0}


def test_nan_soma_distance_passes_any_min_soma():
    nan = float("nan")
    rows = [_row(1, 0.3, nan, x=0.0), _row(2, 0.1, 3.0, x=50.0), _row(3, 0.2, nan, x=100.0),
            _row(4, 0.0, nan, x=150.0, near_end=True), _row(5, 0.05, nan, x=200.0, tier2=1)]
    before = [dict(r) for r in rows]
    for m in (0.0, 5.0, 1e9):
        got = CT.pick_clear_examples(rows, 10, 10.0, min_soma_um=m)
        want = [3, 1] if m > 3.0 else [2, 3, 1]          # the finite 3.0 um row falls below 5 um; NaN rows never do
        assert got == want, (m, got)
    assert CT.pick_clear_examples(rows, 10, 10.0) == [2, 3, 1]                     # the default min_soma_um 0
    assert CT.pick_clear_examples(rows, 10, 10.0, 5.0, taken_xyz=[(100.0, 0.0, 0.0)]) == [1]   # spacing unchanged
    assert [dict(r) for r in rows] == [dict(b) for b in before] or all(
        (a.keys() == b.keys()) for a, b in zip(rows, before))                     # rows unchanged (NaN != NaN)
    for bad in (float("nan"), -1.0, float("inf")):
        with pytest.raises(ValueError):
            CT.pick_clear_examples(rows, 1, 10.0, min_soma_um=bad)
