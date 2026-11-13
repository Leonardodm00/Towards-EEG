"""Spec-derived tests of Block 13 (the Allen-guided plane and the SGT fit in
the per-node chain) -- integration tester, run D3, 2026-11-13. Expected values
come from specs/SPEC.md (Block 13, Block 5's rules it reuses) or by hand; the
code under test only produces the outputs that are checked.

Run (from the workstream directory):
    python -m pytest tests/spec/test_spec_block13_plane_iteration.py -q
The measure_node checks run the synthetic cell of tests/smoke/fixtures_cell.py
(minutes on a shared machine).
"""
from __future__ import annotations

import dataclasses
import math
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import ndimage

import spec_helpers_D3 as H  # noqa: F401  (sys.path)

from allen_image_io import CropFrame
from allen_diameter.analysis import cell, fit_sgt, node_pipeline as NP, phantoms, survey
from allen_diameter.analysis import plane_iteration as PI
from allen_diameter.config import default_config
from allen_diameter.loading import swc_io

CFG = default_config()
M = CFG.measure
NAN = float("nan")


# ------------------------------------------------- the loop (13.1, 13.6) --

class Calls:
    def __init__(self, fn):
        self.fn, self.args = fn, []

    def __call__(self, *a):
        self.args.append(a)
        return self.fn(*a)


def test_iterate_planes_converged_and_records():
    # O13.1: round 1 at m d0/2; a repeat of the previous plane ends "converged", its fit reused (no refit).
    plane_of = Calls(lambda h: (5, {"G": h}))
    fit_at = Calls(lambda k: {"d_hat_um": 1.0, "fit_status": "converged"})
    r = PI.iterate_planes(0.6, 3.0, plane_of, fit_at, max_rounds=5)
    assert r["status"] == "converged" and r["n_rounds"] == 2 and r["k_first"] == 5 and r["k_final"] == 5
    assert r["d_final_um"] == 1.0 and len(fit_at.args) == 1
    assert_allclose([x["half_um"] for x in r["rounds"]], [0.9, 1.5], rtol=1e-15)
    assert r["rounds"][0]["d_in_k"] is None and r["rounds"][1]["d_in_k"] == 5
    assert r["rounds"][0]["refit"] is True and r["rounds"][1]["refit"] is False


def test_iterate_planes_cycle_max_rounds_and_failures():
    # O13.1: "cycle" for an earlier plane; "max_rounds" after R rounds without a repeat; "fit_failed" for a fit
    # without a finite positive d; "no_plane" (or the plane_of's stop) when no plane is picked.
    d_of = {5: 2.0, 6: 1.0}
    r = PI.iterate_planes(1.0, 3.0, lambda h: (5 if h < 2 else 6, {}), lambda k: {"d_hat_um": d_of[k]}, 5)
    assert r["status"] == "cycle" and r["n_rounds"] == 3 and r["k_final"] == 5 and r["d_final_um"] == 2.0
    r = PI.iterate_planes(1.0, 3.0, lambda h: (int(round(100 * h)), {}), lambda k: {"d_hat_um": k / 100.0}, 4)
    assert r["status"] == "max_rounds" and r["n_rounds"] == 4
    # by hand: k = round(100 h) -> 150, 225, 338 (337.5 rounds to even), so h = 1.5, 2.25, 3.375, 1.5 x 3.38
    assert_allclose([x["half_um"] for x in r["rounds"]], [1.5, 2.25, 3.375, 5.07], rtol=1e-12)
    for bad in (NAN, 0.0, -1.0, float("inf")):
        r = PI.iterate_planes(1.0, 3.0, lambda h: (7, {}), lambda k: {"d_hat_um": bad}, 5)
        assert r["status"] == "fit_failed" and r["k_final"] == 7 and r["n_rounds"] == 1
    r = PI.iterate_planes(1.0, 3.0, lambda h: (7, {}), lambda k: {"d_hat_um": NAN}, 5)
    assert math.isnan(r["d_final_um"])
    r = PI.iterate_planes(1.0, 3.0, lambda h: (None, {}), lambda k: {"d_hat_um": 1.0}, 5)
    assert r["status"] == "no_plane" and r["k_first"] is None and r["k_final"] is None and math.isnan(r["d_final_um"])
    r = PI.iterate_planes(1.0, 3.0, lambda h: (None, {"stop": "line_short"}), lambda k: {"d_hat_um": 1.0}, 5)
    assert r["status"] == "line_short"


def test_iterate_planes_failed_fit_reports_nan():
    # O13.1 (Block 11's record: "d_iter_um, the fit there (NaN when it failed)"; iterate_planes' docstring:
    # "d_final_um (the fit there; NaN when it failed or there is none)"): a fit without a finite positive d ends
    # the iteration "fit_failed" and its d is reported as NaN.
    for bad in (0.0, -1.0, float("inf")):
        r = PI.iterate_planes(1.0, 3.0, lambda h: (7, {}), lambda k: {"d_hat_um": bad}, 5)
        assert r["status"] == "fit_failed" and math.isnan(r["d_final_um"]), (bad, r["d_final_um"])


def test_iterate_planes_floor_and_status_free_sizing():
    # O13.1 (D-069): round 1's half-length is max(m d0/2, floor); later rounds are never floored. A fit at a bound
    # sizes the next line like any other (D-052: d_hat_um whatever the status).
    seq = iter([3, 4, 5, 6])
    r = PI.iterate_planes(0.1, 3.0, lambda h: (next(seq), {}), lambda k: {"d_hat_um": 0.1}, 3, min_first_half_um=0.5)
    assert_allclose([x["half_um"] for x in r["rounds"]], [0.5, 0.15, 0.15], rtol=1e-15)
    r = PI.iterate_planes(1.0, 3.0, lambda h: (int(h * 10), {}),
                          lambda k: {"d_hat_um": 2.0, "fit_status": "at_bound"}, 2)
    assert_allclose(r["rounds"][1]["half_um"], 3.0, rtol=1e-15)
    r = PI.iterate_planes(1.0, 3.0, lambda h: (2, {}), lambda k: {"d_hat_um": 1.0}, 2, min_first_half_um=0.0)
    assert_allclose(r["rounds"][0]["half_um"], 1.5, rtol=1e-15)


def test_iterate_planes_refusals():
    # O13.1: ValueError for a floor not finite or < 0, a start not finite and > 0, a multiplier not finite and > 0,
    # a round limit that is not an integer >= 1.
    po, fa = (lambda h: (1, {})), (lambda k: {"d_hat_um": 1.0})
    for kw in (dict(min_first_half_um=NAN), dict(min_first_half_um=-0.1), dict(min_first_half_um=float("inf"))):
        with pytest.raises(ValueError):
            PI.iterate_planes(1.0, 3.0, po, fa, 5, **kw)
    for d0, m, R in ((0.0, 3.0, 5), (NAN, 3.0, 5), (-1.0, 3.0, 5), (1.0, 0.0, 5), (1.0, NAN, 5), (1.0, -2.0, 5),
                     (1.0, 3.0, 0), (1.0, 3.0, 1.5), (1.0, 3.0, True), (1.0, 3.0, "5")):
        with pytest.raises(ValueError):
            PI.iterate_planes(d0, m, po, fa, R)


def test_gradient_iteration_sample_floor_refusals():
    # O13.1: min_first_samples an odd integer >= 3, else ValueError (checked before the square is read).
    st = {"allen_radius_um": 0.3}
    for n in (2, 4, 1, 0, 3.5, True, "3", -3):
        with pytest.raises(ValueError):
            PI.gradient_iteration(st, CFG, min_first_samples=n)


# --------------------------------------------------- square and scene ----

def _block(n=3, H_=40, W=50, left=100, top=200, p=0.1144):
    k, r, c = np.meshgrid(np.arange(n), np.arange(H_), np.arange(W), indexing="ij")
    return (10000.0 * k + 100.0 * r + c), np.arange(-1, n - 1), CropFrame(left, top, 0, p)


def test_cut_square_by_hand():
    # O13.2: the columns c0 = ceil((o_x - h)/p - left), c1 = floor((o_x + h)/p - left) + 1, rows alike; cut from
    # the block when inside; else one provider call over the block's planes, valid ANDed; ValueErrors.
    block, ks, fr = _block()
    p, valid = fr.res_um_px, np.array([True, True, False])
    o, h = np.array([(100 + 25.3) * p, (200 + 20.6) * p]), 1.0
    c0, c1 = math.ceil((o[0] - h) / p - 100), math.floor((o[0] + h) / p - 100) + 1
    r0, r1 = math.ceil((o[1] - h) / p - 200), math.floor((o[1] + h) / p - 200) + 1
    stack, v, sub, fetched = PI.cut_square(block, ks, valid, fr, o, h, None)
    assert not fetched and (c0, c1, r0, r1) == (17, 35, 12, 30)
    assert_array_equal(stack, block[:, r0:r1, c0:c1])
    assert_array_equal(v, valid) and (sub.left, sub.top) == (100 + c0, 200 + r0) and sub.res_um_px == p
    calls = []

    def provider(left, top, width, height, k_lo, k_hi):
        calls.append((left, top, width, height, k_lo, k_hi))
        return (np.full((3, height, width), 7.0), np.arange(k_lo, k_hi + 1), np.array([True, False, True]),
                CropFrame(left, top, 0, p))
    o2 = np.array([(100 + 47.2) * p, (200 + 20.6) * p])                  # the square runs past the right edge
    c0, c1 = math.ceil((o2[0] - h) / p - 100), math.floor((o2[0] + h) / p - 100) + 1
    stack, v, sub, fetched = PI.cut_square(block, ks, valid, fr, o2, h, provider)
    assert fetched and calls == [(100 + c0, 200 + r0, c1 - c0, r1 - r0, -1, 1)]
    assert_array_equal(v, [True, False, False]) and np.all(stack == 7.0)

    def other_planes(left, top, width, height, k_lo, k_hi):
        return np.zeros((3, height, width)), np.arange(k_lo + 1, k_hi + 2), np.ones(3, bool), CropFrame(left, top, 0, p)
    with pytest.raises(ValueError):
        PI.cut_square(block, ks, valid, fr, o2, h, other_planes)
    with pytest.raises(ValueError):
        PI.cut_square(block, ks, valid, fr, o, 0.04, None)              # under 2 x 2 pixels


def test_scene_segments_and_from_branch():
    # O13.3: the bounding-box rule and the row layout, by hand; Scene.from_branch.
    fr = CropFrame(0, 0, 0, 0.1)
    shape = (50, 40)                                   # x in [-0.05, 3.95], y in [-0.05, 4.95] um
    seg = np.array([[1, 1, 0, 2, 2, 0], [5, 5, 0, 6, 6, 0], [3.9, 1, 0, 5, 1, 0], [3.96, 2, 0, 4.5, 2, 0],
                    [-1, -1, 0, -0.06, 5, 0], [-1, 4, 0, 1, 6, 1.0]], dtype=float)
    rad = np.array([[0.1, 0.2], [0.3, 0.3], [0.4, 0.5], [0.6, 0.6], [0.7, 0.7], [0.8, 0.9]])
    ids = np.array([[2, 1], [3, 2], [4, 3], [5, 4], [6, 5], [7, 6]], dtype=np.int64)
    sc = PI.Scene(seg, rad, ids, np.empty((0, 3)))
    branch = NP.Branch.from_points(np.array([[0, 0, 0], [1, 0, 0.0]]), 0.3, ids=np.array([1, 2]))
    out = PI.scene_segments(sc, fr, shape, branch)
    keep = [0, 2, 5]
    assert out.shape == (3, 9) and out.dtype == np.float64
    assert_array_equal(out[:, :6], seg[keep])
    assert_array_equal(out[:, 6:8], rad[keep])
    assert_array_equal(out[:, 8], [1.0, 0.0, 0.0])
    xyz = np.array([[0, 0, 0], [1, 0.5, 0.1], [2, 0.7, 0.3], [3, 1.0, 0.2]])
    br = NP.Branch.from_points(xyz, np.array([0.1, 0.2, 0.3, 0.4]), ids=np.array([10, 11, 12, 13]))
    s = PI.Scene.from_branch(br)
    assert_array_equal(s.seg_xyz, np.column_stack([xyz[:-1], xyz[1:]]))
    assert_array_equal(s.seg_radius, [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]])
    assert_array_equal(s.seg_ids, [[11, 10], [12, 11], [13, 12]]) and s.soma.shape == (0, 3)


def test_cell_scene_by_hand():
    # O13.3: every segment whose child is a dendrite node of `types`; r0 = r1 when the parent is not a dendrite
    # node; the soma discs from the type-1 nodes.
    text = "\n".join(["# hand-made", "1 1 0 0 0 5 -1", "2 3 10 0 0 0.5 1", "3 3 20 0 0 0.4 2", "4 2 0 10 0 0.3 1",
                      "5 3 0 20 0 0.2 4", "6 4 30 0 0 0.35 3", "7 2 0 -10 0 0.3 1"]) + "\n"
    with tempfile.TemporaryDirectory() as tmp:
        path = tmp + "/c.swc"
        with open(path, "w") as f:
            f.write(text)
        swc = swc_io.read_swc(path)
    sc = cell.cell_scene(swc, (3, 4), 0.1144)
    rows = {int(c): (int(p), tuple(r)) for (c, p), r in zip(sc.seg_ids, sc.seg_radius)}
    assert rows == {2: (1, (0.5, 0.5)), 3: (2, (0.5, 0.4)), 5: (4, (0.2, 0.2)), 6: (3, (0.4, 0.35))}
    assert_allclose(sc.soma, [[0.0, 0.0, 5.0]], atol=1e-12)
    k = [int(c) for c, _ in sc.seg_ids].index(3)
    assert_allclose(sc.seg_xyz[k], [10, 0, 0, 20, 0, 0], atol=1e-12)


# --------------------------------------------------- the fit at a plane ----

def _analytic_state(cx_off=0.05, theta=0.4):
    """An analytic block: background 200 minus a Gaussian valley along the line through the node at heading
    theta; planes k = -2..2 with k = 1 invalid and k = 2 all zero."""
    p = M.profile_step_um
    o = np.array([500.3 * p, 400.6 * p])
    left, top = int(round(o[0] / p)) - 62, int(round(o[1] / p)) - 62
    fr = CropFrame(left, top, 0, p)
    rows, cols = np.mgrid[0:125, 0:125]
    x, y = (left + cols) * p, (top + rows) * p
    dist = (x - o[0]) * (-math.sin(theta)) + (y - o[1]) * math.cos(theta)
    plane = 200.0 - 60.0 * np.exp(-dist ** 2 / (2 * 0.35 ** 2))
    block = np.stack([plane - 2, plane, plane + 1, plane, np.zeros_like(plane)])
    ks, valid = np.arange(-2, 3), np.array([True, True, True, False, True])
    t = np.array([math.cos(theta), math.sin(theta), 0.0])
    xyz = np.array([o[0], o[1], 0.0])[None, :] + np.arange(-2, 3)[:, None] * 1.18 * t[None, :]
    br = NP.Branch.from_points(xyz, 0.4)
    yh = np.array([-math.sin(theta), math.cos(theta)])
    res = SimpleNamespace(cx_um=o[0] + cx_off * yh[0], cy_um=o[1] + cx_off * yh[1], theta_rad=theta, phi_rad=0.1)
    seg = PI.scene_segments(PI.Scene.from_branch(br), fr, block.shape[1:], br)
    st = PI.iteration_state(block, ks, valid, fr, br, 2, theta, res, seg, np.empty((0, 3)), M.block_half_um, None, 0)
    return st, block, fr, br, res, o


def _ref_profile(plane, fr, centre, theta, m):
    """Block 5's Eq. 9 written out: the mean over a = -A..A of bilinear samples at centre + v y_hat + a d e_u."""
    p, step = fr.res_um_px, m.profile_step_um
    n = int(round(m.profile_half_um / step))
    v = np.arange(-n, n + 1) * step
    A = int(round(m.along_branch_avg_um / step))
    yh, eu = np.array([-math.sin(theta), math.cos(theta)]), np.array([math.cos(theta), math.sin(theta)])
    acc = np.zeros(v.size)
    for a in range(-A, A + 1):
        P = centre[None, :] + v[:, None] * yh[None, :] + a * step * eu[None, :]
        acc += ndimage.map_coordinates(plane, [P[:, 1] / p - fr.top, P[:, 0] / p - fr.left], order=1,
                                       mode="constant", cval=np.nan)
    return v, acc / (2 * A + 1)


def _ref_bbar(plane, fr, o, br, m):
    """D-018.1: the median of the block +-block_half_um about o over pixels farther than radius + margin from
    every branch segment (2-D, the larger radius of its ends)."""
    p = fr.res_um_px
    rows, cols = np.mgrid[0:plane.shape[0], 0:plane.shape[1]]
    x, y = (fr.left + cols) * p, (fr.top + rows) * p
    inside = (np.abs(x - o[0]) <= m.block_half_um) & (np.abs(y - o[1]) <= m.block_half_um)
    far = np.ones(plane.shape, bool)
    for j in range(len(br.ids) - 1):
        a, b = br.xyz_um[j, :2], br.xyz_um[j + 1, :2]
        ab = b - a
        t = np.clip(((x - a[0]) * ab[0] + (y - a[1]) * ab[1]) / (ab @ ab), 0, 1)
        dd = np.hypot(x - a[0] - t * ab[0], y - a[1] - t * ab[1])
        far &= dd > max(br.radius_um[j], br.radius_um[j + 1]) + m.bbar_mask_margin_um
    sel = plane[inside & far]
    return float(np.median(sel)), sel.size / float(inside.sum())


def test_plane_inputs_by_hand():
    # O13.4: the four statuses; at a valid plane the profile is Block 5's Eq. 9 through the pilot's centre across
    # its heading (bilinear, 2A + 1 lines; 1e-12 gl) and B_bar its masked median (exact).
    st, block, fr, br, res, o = _analytic_state()
    for k in (5, -3, 1):                                   # not among the planes, and the invalid plane
        out = PI.plane_inputs(st, CFG, k)
        assert out["status"] == "no_plane" and out["v_um"] is None and out["profile"] is None
        assert math.isnan(out["B_bar"]) and out["bbar_ok"] is False
    out = PI.plane_inputs(st, CFG, 0)
    v, I = _ref_profile(block[2], fr, np.array([res.cx_um, res.cy_um]), res.theta_rad, M)
    assert out["status"] == "ok" and out["phi_rad"] == res.phi_rad
    assert_allclose(out["v_um"], v, rtol=0, atol=1e-12)
    # 1e-9 gl: the sample coordinates may be formed in another order (~1e-13 px) on gradients up to ~100 gl/px
    assert_allclose(out["profile"], I, rtol=0, atol=1e-9)
    B, frac = _ref_bbar(block[2], fr, o, br, M)
    assert out["B_bar"] == B and out["bbar_ok"] == (frac >= M.bbar_min_unmasked_frac)
    assert PI.plane_inputs(st, CFG, 2)["status"] == "bbar_nonpositive"
    st2 = _analytic_state(cx_off=6.0)[0]
    assert PI.plane_inputs(st2, CFG, 0)["status"] == "profile_nan"


def test_sgt_at_plane_record():
    # O13.4 and boundary 12 -> 13: the record's keys; its numbers are fit_plane(...).final on plane_inputs; NaN
    # numbers and the inputs' status without inputs.
    st = _analytic_state()[0]
    keys = {"d_hat_um", "fit_status", "v0_hat_um", "mu_hat_per_um", "B_bar", "bbar_ok", "w_hat_um", "beta_hat",
            "xi_hat", "zeta_hat", "alpha_hat", "irls_ok", "at_bound", "plane_fit", "v_um", "profile"}
    none = PI.sgt_at_plane(st, CFG, 1)
    assert set(none) == keys and none["fit_status"] == "no_plane" and none["plane_fit"] is None
    assert all(math.isnan(none[k]) for k in ("d_hat_um", "v0_hat_um", "mu_hat_per_um", "w_hat_um", "beta_hat",
                                               "xi_hat", "zeta_hat", "alpha_hat"))
    assert PI.sgt_at_plane(st, CFG, 2)["fit_status"] == "bbar_nonpositive"
    rec = PI.sgt_at_plane(st, CFG, 0)
    assert set(rec) == keys
    inp = PI.plane_inputs(st, CFG, 0)
    f = fit_sgt.fit_plane(inp["v_um"], inp["profile"], inp["phi_rad"], inp["B_bar"], M).final
    got = [rec[k] for k in ("d_hat_um", "v0_hat_um", "mu_hat_per_um", "w_hat_um", "beta_hat", "xi_hat", "zeta_hat",
                            "alpha_hat")]
    assert got == [f.d_um, f.v0_um, f.mu_per_um, f.w_um, f.beta, f.xi, f.zeta, f.alpha_hat]
    assert rec["fit_status"] == f.status and rec["irls_ok"] == f.irls_ok and tuple(rec["at_bound"]) == f.at_bound
    assert isinstance(rec["plane_fit"], fit_sgt.PlaneSGT) and rec["B_bar"] == inp["B_bar"]


# ------------------------------------------------------------ the flags ----

def _faint_crossing(I, v, B, m):
    """Block 5's faint and crossing rules written out from the spec."""
    s = ndimage.gaussian_filter1d(I, m.focus_smooth_px, mode="nearest") if m.focus_smooth_px > 0 else I
    n0 = int(np.argmin(s))
    depth = B - s[n0]
    out = ["faint"] if depth < m.faint_min_dip_gl else []
    mins = [n for n in range(1, s.size - 1) if s[n] < s[n - 1] and s[n] <= s[n + 1]
            and abs(v[n] - v[n0]) >= m.second_dip_min_sep_um and B - s[n] >= m.second_dip_rel * depth]
    if depth > 0 and mins:
        out.append("crossing")
    return out


def ref_flags(vertical, steep, rec, ks, m, ending):
    """D-061.3 and D-068 by the spec's words: Block 5's rules and order at k_iter, then iter_<ending>."""
    f = ["vertical"] if vertical else []
    fitted = rec is not None and rec.get("plane_fit") is not None
    if rec is not None and rec.get("profile") is not None:
        I = np.asarray(rec["profile"], float)
        if not np.all(np.isfinite(I)):
            f.append("profile_nan")
        elif fitted:
            f += _faint_crossing(I, np.asarray(rec["v_um"], float), rec["B_bar"], m)
    if rec is not None:
        G = np.asarray(rec["G"], float)
        fin = np.flatnonzero(np.isfinite(G))
        n = int(rec["k"]) - int(ks[0])
        if fin.size and (n - fin[0] < m.edge_margin_planes or fin[-1] - n < m.edge_margin_planes):
            f.append("stack_edge")
    if rec is None or not rec["bbar_ok"]:
        f.append("bbar_few")
    if steep:
        f.append("steep")
    if fitted and rec["alpha_hat"] > m.alpha_dark_flag:
        f.append("dark")
    if fitted and rec["fit_status"] != "converged":
        f.append(rec["fit_status"])
    if ending is not None and ending != "converged":
        f.append("iter_" + ending)
    return tuple(f)


def test_deliverable_flags_by_hand():
    # O13.5: hand-made records through every rule.
    v = np.linspace(-3, 3, 53)
    deep = 200 - 50 * np.exp(-v ** 2 / 0.1)
    two = deep - 40 * np.exp(-(v - 1.8) ** 2 / 0.1)
    G = np.array([NAN, 1.0, 2.0, 3.0, 2.5, NAN])
    ks = np.arange(-2, 4)
    base = dict(profile=deep, v_um=v, B_bar=200.0, bbar_ok=True, plane_fit=object(), alpha_hat=0.5,
                fit_status="converged", k=0, G=G)
    cases = [(False, False, base, "converged"), (True, True, dict(base, profile=two, alpha_hat=1.5), "cycle"),
             (False, False, dict(base, k=-1, bbar_ok=False, fit_status="at_bound"), "max_rounds"),
             (False, True, dict(base, k=2, profile=np.full(53, 199.0)), "converged"),
             (False, False, dict(base, profile=np.where(v > 2.9, NAN, deep)), "converged"),
             (False, False, dict(base, plane_fit=None, fit_status="bbar_nonpositive"), "fit_failed"),
             (True, False, None, "no_plane"), (False, False, None, None)]
    for vertical, steep, rec, ending in cases:
        pilot = SimpleNamespace(vertical=vertical, steep=steep)
        got = NP.deliverable_flags(pilot, rec, ks, M, ending)
        assert got == ref_flags(vertical, steep, rec, ks, M, ending), (got, ending)
    assert "iter_cycle" in NP.deliverable_flags(SimpleNamespace(vertical=False, steep=False), base, ks, M, "cycle")
    assert not any(f.startswith("iter_") for f in NP.deliverable_flags(SimpleNamespace(vertical=False, steep=False),
                                                                        base, ks, M, "converged"))
    assert not any(f.startswith("iter_") for f in phantoms.SELECTION_FLAGS)


# ---------------------------------------------------------- the chain ----

def _cell_case(tmp):
    import run_cell
    from fixtures_cell import synthetic_cell
    cfg = dataclasses.replace(CFG, measure=dataclasses.replace(M, block_half_um=3.5))
    swc, fetcher, planes, _ = synthetic_cell(tmp, cfg, d_true=0.8, mu=1.0)
    prov = run_cell.real_provider(fetcher, planes, cfg.acquisition.res0_um)
    types = cfg.acquisition.dendrite_swc_types
    xyz = cell.to_image_um(swc.xyz, cfg.acquisition.res0_um)
    run = cell.stretches(swc, types)[0]
    br, off = cell.stretch_branch(swc, run, xyz)
    return cfg, swc, prov, br, off + 2


def _same_result(a, b):
    for f in dataclasses.fields(NP.NodeResult):
        if f.name in ("plane_fit", "iteration"):
            continue
        x, y = getattr(a, f.name), getattr(b, f.name)
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            assert np.array_equal(np.asarray(x), np.asarray(y), equal_nan=True), f.name
        elif isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y):
            continue
        else:
            assert x == y, (f.name, x, y)


@pytest.fixture(scope="module")
def chain():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, swc, prov, br, i = _cell_case(tmp)
        calls = Calls(prov)
        scene = cell.cell_scene(swc, cfg.acquisition.dendrite_swc_types, cfg.acquisition.res0_um)
        res = NP.measure_node(br, i, calls, cfg, scene=scene)
        n_calls = len(calls.args)
        pil = NP.measure_pilot(br, i, prov, cfg)
        r_none = NP.measure_node(br, i, prov, cfg, scene=None)
        r_own = NP.measure_node(br, i, prov, cfg, scene=PI.Scene.from_branch(br))
        yield dict(cfg=cfg, br=br, i=i, res=res, n_calls=n_calls, req=calls.args[0], pil=pil, r_none=r_none,
                   r_own=r_own)


def test_measure_node_pilot_is_block5(chain):
    # O13.5: pilot_view(measure_node(...)) equals measure_pilot(...) field by field; the request is node_request's.
    _same_result(NP.pilot_view(chain["res"]), chain["pil"])
    assert tuple(chain["req"]) == tuple(NP.node_request(chain["br"], chain["i"], chain["cfg"]))
    assert NP.pilot_view(chain["pil"]) is chain["pil"]


def test_measure_node_deliverable_and_flags(chain):
    # O13.5: the deliverable columns are the SGT record of the last round that picked a plane; the flags are Block
    # 5's rules at k_iter (recomputed here from that record) with iter_<ending>; one provider call unless a restack
    # left the block.
    res, cfg = chain["res"], chain["cfg"]
    it = res.iteration
    picked = [r for r in it["rounds"] if r["k"] is not None]
    assert res.iter_status == it["status"] in PI.ITER_STATUSES and res.n_rounds == len(it["rounds"])
    assert 1 <= chain["n_calls"] <= 1 + res.n_restacks
    rec = picked[-1]
    assert res.k_iter == rec["k"]
    for col, key in (("d_hat_um", "d_hat_um"), ("mu_hat_per_um", "mu_hat_per_um"), ("v0_hat_um", "v0_hat_um"),
                     ("alpha_hat", "alpha_hat"), ("B_bar", "B_bar"), ("w_hat_um", "w_hat_um"), ("beta_hat", "beta_hat"),
                     ("xi_hat", "xi_hat"), ("zeta_hat", "zeta_hat")):
        assert getattr(res, col) == rec[key], col
    assert res.fit_status == rec["fit_status"] and res.irls_ok == rec["irls_ok"]
    assert tuple(res.at_bound) == tuple(rec["at_bound"])
    k_swc = int(round(float(chain["br"].xyz_um[chain["i"], 2]) / cfg.acquisition.dz_um))
    ks = np.arange(k_swc - cfg.measure.iter_planes_half, k_swc + cfg.measure.iter_planes_half + 1)
    assert res.flags == ref_flags(res.vertical, res.steep, rec, ks, cfg.measure, it["status"])
    pil = chain["pil"]
    assert (res.d_pilot_um, res.mu_pilot_per_um, res.v0_pilot_um, res.alpha_pilot, res.B_bar_pilot) == \
        (pil.d_hat_um, pil.mu_hat_per_um, pil.v0_hat_um, pil.alpha_hat, pil.B_bar)
    assert res.pilot_status == pil.fit_status and res.pilot_flags == pil.flags and res.fit is not None
    for col in ("k_star", "z_sub_um", "cx_um", "cy_um", "cz_um", "theta_rad", "phi_rad", "steep", "vertical"):
        assert getattr(res, col) == getattr(pil, col), col


def test_measure_node_scene_none_is_own_branch(chain):
    # O13.5: scene=None equals Scene.from_branch(branch).
    _same_result(chain["r_none"], chain["r_own"])


def test_one_function_object():
    # O13.5 (D-057 (iii)): phantoms, cell and survey call node_pipeline.measure_node.
    assert phantoms.measure_node is NP.measure_node and cell.measure_node is NP.measure_node
    assert survey.measure_node is NP.measure_node


def test_pilot_view_defaults():
    # O13.8: NodeResult's Block 13 fields default to a pilot's; pilot_view is the identity when pilot_status is "".
    defaults = {f.name: f.default for f in dataclasses.fields(NP.NodeResult) if f.default is not dataclasses.MISSING}
    assert defaults["k_iter"] == -1 and defaults["iter_status"] == "" and defaults["n_rounds"] == 0
    assert defaults["n_restacks"] == 0 and defaults["irls_ok"] is None and defaults["at_bound"] == ()
    assert defaults["pilot_status"] == "" and defaults["pilot_flags"] == () and defaults["plane_fit"] is None
    for k in ("w_hat_um", "beta_hat", "xi_hat", "zeta_hat", "d_pilot_um", "mu_pilot_per_um", "v0_pilot_um",
              "alpha_pilot", "B_bar_pilot"):
        assert math.isnan(defaults[k]), k


def test_node_request_planes_by_hand():
    # O13.6: the planes from the lowest to the highest of Block 5's ranges (k_j +- (planes_half + widening)) of the
    # window nodes and of k_SWC +- iter_planes_half; the xy block holds +-block_half_um about every window node.
    dz, p = CFG.acquisition.dz_um, CFG.acquisition.res0_um
    for phi_deg in (0.0, 30.0, 70.0):
        phi = math.radians(phi_deg)
        t = np.array([math.cos(phi) * math.cos(0.3), math.cos(phi) * math.sin(0.3), math.sin(phi)])
        xyz = np.array([20.0, 15.0, 3.0])[None, :] + np.arange(-6, 7)[:, None] * 1.18 * t[None, :]
        br = NP.Branch.from_points(xyz, 0.3)
        i = 6
        left, top, width, height, k_lo, k_hi = NP.node_request(br, i, CFG)
        R, L = M.direction_redraws, M.line_fit_window_um
        win = np.flatnonzero(np.abs(br.s_um - br.s_um[i]) <= 0.5 * (R + 1) * L + 1e-9)
        P = M.planes_half + (math.ceil(0.5 * L * math.sin(phi) / dz - 1e-9) if M.planes_widen_for_tilt else 0)
        lo5 = min(int(round(xyz[j, 2] / dz)) - P for j in win)
        hi5 = max(int(round(xyz[j, 2] / dz)) + P for j in win)
        k_swc = int(round(xyz[i, 2] / dz))
        assert (k_lo, k_hi) == (min(lo5, k_swc - M.iter_planes_half), max(hi5, k_swc + M.iter_planes_half)), phi_deg
        assert left * p <= xyz[win, 0].min() - M.block_half_um and (left + width - 1) * p >= xyz[win, 0].max() + M.block_half_um
        assert top * p <= xyz[win, 1].min() - M.block_half_um and (top + height - 1) * p >= xyz[win, 1].max() + M.block_half_um


def test_first_line_floor_on_a_thin_branch():
    # 13.6's oracle (D-069): on the fixture a branch of radius 0.01 um ends line_short without the floor and
    # picks a plane with it, round 1 holding 3 samples on a half-length of one profile step; measure_node passes
    # iter_min_first_samples = 3.
    import run_cell
    from fixtures_cell import synthetic_cell
    with tempfile.TemporaryDirectory() as tmp:
        cfg = dataclasses.replace(CFG, measure=dataclasses.replace(M, block_half_um=3.5))
        swc, fetcher, planes, _ = synthetic_cell(tmp, cfg, d_true=0.8, mu=1.0, allen_radius=0.01)
        prov = run_cell.real_provider(fetcher, planes, cfg.acquisition.res0_um)
        xyz = cell.to_image_um(swc.xyz, cfg.acquisition.res0_um)
        br, off = cell.stretch_branch(swc, cell.stretches(swc, cfg.acquisition.dendrite_swc_types)[0], xyz)
        i = off + 2
        assert br.radius_um[i] == 0.01
        pil = NP.measure_pilot(br, i, prov, cfg)
        blk = prov(*NP.node_request(br, i, cfg))
        k_swc = int(round(float(br.xyz_um[i, 2]) / cfg.acquisition.dz_um))
        P = cfg.measure.iter_planes_half
        ks = np.asarray(blk[1])
        sel = (ks >= k_swc - P) & (ks <= k_swc + P)
        bb, kb, vb, fr = np.asarray(blk[0])[sel], ks[sel], np.asarray(blk[2])[sel], blk[3]
        seg = PI.scene_segments(PI.Scene.from_branch(br), fr, bb.shape[1:], br)
        st = PI.iteration_state(bb, kb, vb, fr, br, i, pil.theta_rad, pil, seg, np.empty((0, 3)),
                                cfg.measure.block_half_um, prov, k_swc)
        no_floor = PI.gradient_iteration(st, cfg, d_start_um=0.02, line_mult=3.0, max_rounds=5, bg_margin_um=4.0)
        assert no_floor["status"] == "line_short" and no_floor["k_final"] is None
        floor = PI.gradient_iteration(st, cfg, d_start_um=0.02, line_mult=3.0, max_rounds=5, bg_margin_um=4.0,
                                      min_first_samples=3)
        r1 = floor["rounds"][0]
        assert r1["k"] is not None and r1["n_samples"] == 3 and abs(r1["half_um"] - cfg.measure.profile_step_um) < 1e-15
        res = NP.measure_node(br, i, prov, cfg)
        assert res.iteration["rounds"][0]["k"] is not None and res.iteration["rounds"][0]["n_samples"] == 3


# ------------------------------------------------------- configuration ----

def test_iteration_config_defaults_validate_and_hash():
    # O13.7
    assert (M.iter_min_first_samples, M.iter_line_mult, M.iter_max_rounds, M.iter_planes_half, M.iter_bg_margin_um) \
        == (3, 3.0, 5, 6, 4.0)
    bad = {"iter_line_mult": [0.0, -1.0, NAN, float("inf"), True], "iter_max_rounds": [0, 1.5, True, "5"],
           "iter_planes_half": [0], "iter_bg_margin_um": [-0.5, NAN, float("inf")],
           "iter_min_first_samples": [2, 4, 1, 3.5, True]}
    for k, vals in bad.items():
        for v in vals:
            c = dataclasses.replace(CFG, measure=dataclasses.replace(M, **{k: v}))
            with pytest.raises(ValueError):
                c.validate()
    h0 = CFG.signature_hash("estimator")
    assert h0 not in ("1bb3585b4a4f1373", "535847535e83cf7d", "1654924e11020dd9")    # tables of those are refused
    for k, v in (("iter_line_mult", 2.0), ("iter_max_rounds", 4), ("iter_planes_half", 5), ("iter_bg_margin_um", 3.0),
                 ("iter_min_first_samples", 5)):
        c = dataclasses.replace(CFG, measure=dataclasses.replace(M, **{k: v}))
        c.validate()
        assert c.signature_hash("estimator") != h0, k
