"""Smoke test for Block 13 in specs/SPEC.md: the Allen-guided plane and the
SGT fit inside the per-node chain (D-052.3, D-057 (iii), D-061, D-064).

13.2 (this version): the scene the iteration's background masks (D-064) and
the square about the node, shared by the diagnostics and the measurement:
plane_iteration.Scene (from_branch), scene_segments, cut_square,
iteration_state, analysis.cell.cell_scene, and survey.node_planes and
scripts/plane_differences.node_stack rebuilt on them. 13.3-13.4 extend this
file with measure_node's checks.

Checks
    test_known_answer   Scene.from_branch on a hand-made branch (the rows, radii
                        and ids by hand); scene_segments on a hand-made block
                        frame: a segment inside kept, one outside dropped, one
                        touching the block's bounding box at its edge kept and
                        one 1e-4 um beyond it dropped, one crossing the block
                        without an end inside kept, in_stretch by hand;
                        cut_square on a hand-made block: the slice by hand when
                        inside (no provider call), one provider call with the
                        arguments by hand when outside, valid ANDed
    test_reference      on the synthetic cell (fixtures_cell, d 0.8 um):
                        survey.node_planes' segments equal the rule of
                        84ea91f written out here (bit for bit), and
                        node_stack's dict equals the construction of 84ea91f
                        written out here, key by key, for a square inside the
                        block and one fetched
    test_convergence    not a discretisation (skipped)
    test_invariants     every row of a lone branch's scene is in_stretch; a
                        permuted scene gives the permuted rows; a missing
                        scene and Scene.from_branch give equal rows
    test_contract       shapes and dtypes ((n, 9) float64, (0, 9) for an empty
                        scene); Scene refuses inconsistent shapes and
                        non-finite entries; cut_square refuses a square below
                        2 x 2 pixels and a provider returning other planes,
                        with the label in the message
    test_determinism    no randomness in 13.2's functions (skipped)
    test_edge_cases     a square whose last column is the block's last is cut,
                        one pixel more is fetched; a two-node branch's scene;
                        a cell without soma gives soma (0, 3)

Run
    cd "Passive Features/Diameter Re-measurement"
    python tests/smoke/test_smoke_block13.py

Keep this file pure ASCII (hpc-python-compat).
"""
from __future__ import annotations

import dataclasses
import importlib.metadata
import math
import platform
import sys
import tempfile
import time
import traceback
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
for p in (WS / "src", HERE, WS / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from allen_diameter.analysis import plane_iteration as PI  # noqa: E402
from allen_diameter.analysis.node_pipeline import Branch  # noqa: E402
from allen_diameter.config import default_config  # noqa: E402

SEED = 20261020
REPORT_PACKAGES = ("numpy", "scipy")


def _frame(left, top, p):
    from allen_image_io import CropFrame
    return CropFrame(int(left), int(top), 0, float(p))


class _NoProvider:
    """A provider that must not be called."""
    calls = 0

    def __call__(self, *args):
        _NoProvider.calls += 1
        raise AssertionError("the provider was called: %r" % (args,))


class _RecordingProvider:
    """Serves a fixed square for any request and records the requests."""

    def __init__(self, ks, valid, value=7.0):
        self.ks, self.valid, self.value, self.calls = np.asarray(ks), np.asarray(valid, bool), value, []

    def __call__(self, left, top, width, height, k_lo, k_hi):
        self.calls.append((left, top, width, height, k_lo, k_hi))
        ks = np.arange(int(k_lo), int(k_hi) + 1)
        return np.full((ks.size, height, width), self.value), ks, self.valid.copy(), _frame(left, top, 0.1)


def _cell(tmp, d_true=0.8):
    import run_cell
    from fixtures_cell import synthetic_cell
    base = default_config()
    cfg = dataclasses.replace(base, measure=dataclasses.replace(base.measure, block_half_um=3.5))
    swc, fetcher, planes, _ = synthetic_cell(tmp, cfg, d_true=d_true, mu=1.0)
    return cfg, swc, run_cell.real_provider(fetcher, planes, cfg.acquisition.res0_um)


def _segments_84ea91f(swc, cfg, transform, branch, frame, block):
    """survey.node_planes' segments as written at 84ea91f (the reference)."""
    from allen_diameter.analysis import cell
    types = cfg.acquisition.dendrite_swc_types
    xyz_img = cell.to_image_um(swc.xyz, cfg.acquisition.res0_um, **(transform or {}))
    pidx = swc.parent_index()
    child = np.flatnonzero(swc.dendrite_mask(types) & (pidx >= 0))
    par = pidx[child]
    a, b = xyz_img[par], xyz_img[child]
    p = float(frame.res_um_px)
    x_lo, x_hi = (frame.left - 0.5) * p, (frame.left + block.shape[2] - 0.5) * p
    y_lo, y_hi = (frame.top - 0.5) * p, (frame.top + block.shape[1] - 0.5) * p
    meets = ((np.maximum(a[:, 0], b[:, 0]) >= x_lo) & (np.minimum(a[:, 0], b[:, 0]) <= x_hi)
             & (np.maximum(a[:, 1], b[:, 1]) >= y_lo) & (np.minimum(a[:, 1], b[:, 1]) <= y_hi))
    in_branch = {int(x) for x in branch.ids}
    own = np.array([float(int(swc.ids[c]) in in_branch and int(swc.ids[q]) in in_branch) for c, q in zip(child, par)])
    r1 = np.asarray(swc.radius[child], dtype=float)
    r0 = np.where(swc.dendrite_mask(types)[par], np.asarray(swc.radius[par], dtype=float), r1)
    return np.column_stack([a, b, r0, r1, own])[meets] if child.size else np.empty((0, 9))


def _node_stack_84ea91f(swc, provider, cfg, node_id, planes_half, half_um, line_half_um, d_line_mult,
                        allen_line_mult):
    """scripts/plane_differences.node_stack as written at 84ea91f (band_um 0, no transform), on the current
    survey.node_planes (whose segments test_reference checks on their own)."""
    from allen_image_io import CropFrame
    from allen_diameter.analysis import cell, survey
    from allen_diameter.plotting.figures import image_extent_um
    pl = survey.node_planes(swc, provider, cfg, node_id, None)
    frame, (H, W) = pl["frame"], pl["block"].shape[1:]
    k_swc = int(pl["k_swc"])
    k_lo, k_hi = k_swc - int(planes_half), k_swc + int(planes_half)
    block, ks, valid, _ = provider(frame.left, frame.top, W, H, k_lo, k_hi)
    p = float(frame.res_um_px)
    half = float(cfg.measure.block_half_um if half_um is None else half_um)
    if line_half_um is not None:
        half = max(half, float(line_half_um) + 2.0 * p)
    if d_line_mult is not None:
        d = float(pl["result"].d_hat_um)
        if math.isfinite(d) and d > 0:
            half = max(half, 0.5 * float(d_line_mult) * d + 2.0 * p)
    if allen_line_mult is not None:
        ra = float(pl["allen_radius_um"])
        if math.isfinite(ra) and ra > 0:
            half = max(half, float(allen_line_mult) * ra + 2.0 * p)
    o = np.asarray(pl["branch"].xyz_um[pl["index"], :2], dtype=float)
    c0 = int(math.ceil((o[0] - half) / p - frame.left))
    c1 = int(math.floor((o[0] + half) / p - frame.left)) + 1
    r0 = int(math.ceil((o[1] - half) / p - frame.top))
    r1 = int(math.floor((o[1] + half) / p - frame.top)) + 1
    fetched = not (c0 >= 0 and r0 >= 0 and c1 <= W and r1 <= H)
    if fetched:
        square, ks_sq, valid_sq, _ = provider(int(frame.left + c0), int(frame.top + r0), c1 - c0, r1 - r0, k_lo, k_hi)
        stack = np.asarray(square, dtype=float)
        valid = np.asarray(valid, dtype=bool) & np.asarray(valid_sq, dtype=bool)
    else:
        stack = np.asarray(block[:, r0:r1, c0:c1], dtype=float)
    sub = CropFrame(int(frame.left + c0), int(frame.top + r0), 0, p)
    xyz, radius = pl["branch"].xyz_um, np.asarray(pl["branch"].radius_um, dtype=float)
    som = np.asarray(swc.types) == 1
    soma = np.column_stack([cell.to_image_um(swc.xyz[som], cfg.acquisition.res0_um)[:, :2],
                            np.asarray(swc.radius, dtype=float)[som]]) if som.any() else np.empty((0, 3))
    mask = np.ones(stack.shape[1:], dtype=bool)
    return dict(stack=stack, ks=np.asarray(ks), valid=np.asarray(valid, dtype=bool), mask=mask, frame=sub,
                extent=image_extent_um(sub, stack.shape[1:]), k_swc=k_swc, result=pl["result"], o=o,
                theta=float(pl["result"].theta_rad), xyz=xyz, radius=radius, block=np.asarray(block, dtype=float),
                block_frame=frame, segments=np.asarray(pl["segments"], dtype=float), soma=soma,
                square_half_um=half, square_fetched=bool(fetched), allen_radius_um=float(pl["allen_radius_um"]),
                node_type=int(np.asarray(swc.types)[np.flatnonzero(np.asarray(swc.ids) == int(node_id))[0]]),
                branch=pl["branch"])


def _same(a, b, where):
    """AssertionError unless a and b are equal (arrays bit for bit, NaN = NaN)."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        A, B = np.asarray(a), np.asarray(b)
        assert A.shape == B.shape and A.dtype == B.dtype, (where, A.shape, A.dtype, B.shape, B.dtype)
        assert np.array_equal(A, B, equal_nan=A.dtype.kind in "fc"), where
    elif isinstance(a, float) and isinstance(b, float):
        assert a == b or (math.isnan(a) and math.isnan(b)), (where, a, b)
    elif isinstance(a, tuple) and isinstance(b, tuple):
        assert len(a) == len(b), where
        for i, (x, y) in enumerate(zip(a, b)):
            _same(x, y, "%s[%d]" % (where, i))
    else:
        assert a == b, (where, a, b)


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    # Scene.from_branch: one segment per consecutive pair, the proximal node the parent
    br = Branch.from_points([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 1.0, 0.5]], [0.5, 0.4, 0.3], ids=[10, 11, 12])
    sc = PI.Scene.from_branch(br)
    np.testing.assert_array_equal(sc.seg_xyz, [[0, 0, 0, 1, 0, 0], [1, 0, 0, 2, 1, 0.5]])
    np.testing.assert_array_equal(sc.seg_radius, [[0.5, 0.4], [0.4, 0.3]])
    np.testing.assert_array_equal(sc.seg_ids, [[11, 10], [12, 11]])
    assert sc.soma.shape == (0, 3)
    # scene_segments on a block of W = 6, H = 8 pixels at left 10, top 20, p 0.5 um: its bounding box is
    # x [4.75, 7.75], y [9.75, 13.75] um (pixel centres at k p, edges half a pixel out)
    fr = _frame(10, 20, 0.5)
    seg = np.array([[5.0, 10.0, 0, 6.0, 11.0, 0],        # inside
                    [0.0, 0.0, 0, 1.0, 1.0, 0],          # outside
                    [7.75, 12.0, 0, 9.0, 12.0, 0],       # touches the right edge exactly: kept (closed box)
                    [7.7501, 12.0, 0, 9.0, 12.0, 0],     # 1e-4 um beyond it: dropped
                    [3.0, 11.0, 0, 9.0, 11.0, 0]])       # crosses the block, no end inside: kept
    rad = np.array([[0.3, 0.2], [0.1, 0.1], [0.4, 0.4], [0.5, 0.5], [0.6, 0.7]])
    ids = np.array([[2, 1], [3, 2], [4, 3], [5, 4], [6, 5]])
    own_branch = Branch.from_points([[0, 0, 0], [1, 0, 0]], 0.1, ids=[1, 2])
    rows = PI.scene_segments(PI.Scene(seg, rad, ids, np.empty((0, 3))), fr, (8, 6), own_branch)
    expect = np.column_stack([seg, rad, [1.0, 0.0, 0.0, 0.0, 0.0]])[[0, 2, 4]]
    np.testing.assert_array_equal(rows, expect)
    # cut_square inside: block 3 planes x 10 rows x 12 columns, left 100, top 200, p 0.1 um; o (10.53, 20.42) um,
    # half 0.25 um: c0 = ceil(102.8 - 100) = 3, c1 = floor(107.8 - 100) + 1 = 8, r0 = ceil(201.7 - 200) = 2,
    # r1 = floor(206.7 - 200) + 1 = 7
    blk = np.arange(3 * 10 * 12, dtype=float).reshape(3, 10, 12)
    ks, valid = np.arange(-1, 2), np.array([True, True, False])
    fr = _frame(100, 200, 0.1)
    stack, v, sub, fetched = PI.cut_square(blk, ks, valid, fr, np.array([10.53, 20.42]), 0.25, _NoProvider())
    np.testing.assert_array_equal(stack, blk[:, 2:7, 3:8])
    assert not fetched and (sub.left, sub.top) == (103, 202) and float(sub.res_um_px) == 0.1
    np.testing.assert_array_equal(v, valid)
    # cut_square outside (half 1 um: c0 = ceil(95.3 - 100) = -4): one request with the arguments by hand
    prov = _RecordingProvider(ks, [True, False, True])
    stack, v, sub, fetched = PI.cut_square(blk, ks, valid, fr, np.array([10.53, 20.42]), 1.0, prov)
    c0, c1 = math.ceil(95.3 - 100), math.floor(115.3 - 100) + 1          # -4, 16
    r0, r1 = math.ceil(194.2 - 200), math.floor(214.2 - 200) + 1         # -5, 15
    assert prov.calls == [(100 + c0, 200 + r0, c1 - c0, r1 - r0, -1, 1)], prov.calls
    assert fetched and stack.shape == (3, r1 - r0, c1 - c0) and (sub.left, sub.top) == (100 + c0, 200 + r0)
    np.testing.assert_array_equal(v, [True, False, False])
    # iteration_state on a hand-made branch with distinct radii: node 1's xy, radius and plane, the square of
    # cut_square about it, the inputs passed through
    br = Branch.from_points([[10.4, 20.3, 0.0], [10.53, 20.42, 0.1], [10.7, 20.5, 0.2]], [0.1, 0.2, 0.3],
                            ids=[7, 8, 9])
    seg, som = np.ones((2, 9)), np.full((1, 3), 2.0)
    st = PI.iteration_state(blk, ks, valid, fr, br, 1, 0.75, "pilot", seg, som, 0.25, _NoProvider(), 3)
    np.testing.assert_array_equal(st["o"], [10.53, 20.42])
    assert st["allen_radius_um"] == 0.2 and st["index"] == 1 and st["k_swc"] == 3 and st["theta"] == 0.75
    assert st["result"] == "pilot" and st["branch"] is br and st["block_frame"] is fr
    np.testing.assert_array_equal(st["stack"], blk[:, 2:7, 3:8])
    np.testing.assert_array_equal(st["block"], blk)
    np.testing.assert_array_equal(st["segments"], seg)
    np.testing.assert_array_equal(st["soma"], som)
    assert st["square_half_um"] == 0.25 and st["square_fetched"] is False and (st["frame"].left, st["frame"].top) == (103, 202)
    # cell.cell_scene by hand: a soma off the origin (id 1, r 4 um), a dendrite node on it (its r0 is its own r1:
    # the parent is not a dendrite node) and one more; a transform that shifts x and y and moves z0
    from allen_diameter.analysis import cell
    from allen_diameter.loading import swc_io
    tf = dict(shift_full_px=(2.0, 3.0), flip_y_full_h=None, z0_um=1.0)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "hand.swc"
        path.write_text("1 1 1.0 2.0 3.0 4.0 -1\n2 3 5.0 2.0 3.0 0.5 1\n3 3 6.0 2.5 3.5 0.4 2\n")
        sc = cell.cell_scene(swc_io.read_swc(str(path)), (3, 4), 0.1144, tf)
    xyz = cell.to_image_um(np.array([[1.0, 2.0, 3.0], [5.0, 2.0, 3.0], [6.0, 2.5, 3.5]]), 0.1144, **tf)
    np.testing.assert_array_equal(sc.soma, [[xyz[0, 0], xyz[0, 1], 4.0]])
    np.testing.assert_array_equal(sc.seg_xyz, [np.concatenate([xyz[0], xyz[1]]), np.concatenate([xyz[1], xyz[2]])])
    np.testing.assert_array_equal(sc.seg_radius, [[0.5, 0.5], [0.5, 0.4]])
    np.testing.assert_array_equal(sc.seg_ids, [[2, 1], [3, 2]])
    assert abs(xyz[0, 0] - (1.0 + 2.0 * 0.1144)) < 1e-12 and abs(xyz[0, 2] - 2.0) < 1e-12, xyz[0]


def test_reference():
    import plane_differences as PD
    from allen_diameter.analysis import survey
    with tempfile.TemporaryDirectory() as tmp:
        cfg, swc, prov = _cell(tmp, 0.8)
        for nid in (3, 5):
            pl = survey.node_planes(swc, prov, cfg, nid)
            ref = _segments_84ea91f(swc, cfg, None, pl["branch"], pl["frame"], pl["block"])
            _same(pl["segments"], ref, "node_planes segments, node %d" % nid)
            assert ref.shape[0] >= 2, ref.shape
        # node_stack: a square inside the block (default half) and one fetched (half 5 um > block_half_um 3.5)
        for half, want_fetched in ((None, False), (5.0, True)):
            new = PD.node_stack(swc, prov, cfg, 4, None, 6, half, 0.0, None, 3.0, 3.0)
            old = _node_stack_84ea91f(swc, prov, cfg, 4, 6, half, None, 3.0, 3.0)
            assert new["square_fetched"] is want_fetched, (half, new["square_fetched"])
            missing = set(old) - set(new)
            assert not missing, missing
            for k in old:
                if k in ("result", "branch", "frame", "block_frame"):
                    continue
                _same(new[k], old[k], "node_stack[%r], half %r" % (k, half))
            for k in ("frame", "block_frame"):
                f0, f1 = old[k], new[k]
                assert (f0.left, f0.top, float(f0.res_um_px)) == (f1.left, f1.top, float(f1.res_um_px)), k
            for f in ("node_id", "k_star", "d_hat_um", "cx_um", "cy_um", "theta_rad", "phi_rad", "B_bar"):
                _same(getattr(new["result"], f), getattr(old["result"], f), "result.%s, half %r" % (f, half))
            np.testing.assert_array_equal(new["branch"].xyz_um, old["branch"].xyz_um)


def test_convergence():
    raise unittest.SkipTest("13.2 holds no discretisation")


def test_invariants():
    br = Branch.from_points([[1.0, 2.0, 0.0], [2.0, 2.5, 0.1], [3.0, 2.0, 0.2], [4.0, 2.4, 0.2]],
                            [0.3, 0.3, 0.25, 0.2], ids=[5, 6, 7, 8])
    fr, shape = _frame(0, 0, 0.1144), (80, 80)
    sc = PI.Scene.from_branch(br)
    rows = PI.scene_segments(sc, fr, shape, br)
    assert rows.shape == (3, 9) and np.all(rows[:, 8] == 1.0), rows
    perm = np.array([2, 0, 1])
    rows_p = PI.scene_segments(PI.Scene(sc.seg_xyz[perm], sc.seg_radius[perm], sc.seg_ids[perm], sc.soma),
                               fr, shape, br)
    np.testing.assert_array_equal(rows_p, rows[perm])
    # another branch: none of the rows is in its stretch
    other = Branch.from_points([[0, 0, 0], [1, 1, 0]], 0.2, ids=[100, 101])
    assert np.all(PI.scene_segments(sc, fr, shape, other)[:, 8] == 0.0)


def test_contract():
    br = Branch.from_points([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], 0.2)
    rows = PI.scene_segments(PI.Scene.from_branch(br), _frame(0, 0, 0.1), (40, 40), br)
    assert rows.shape == (1, 9) and rows.dtype == np.float64
    empty = PI.Scene(np.empty((0, 6)), np.empty((0, 2)), np.empty((0, 2), dtype=np.int64), np.empty((0, 3)))
    e = PI.scene_segments(empty, _frame(0, 0, 0.1), (40, 40), br)
    assert e.shape == (0, 9) and e.dtype == np.float64
    for args in ((np.zeros((2, 5)), np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((0, 3))),
                 (np.zeros((2, 6)), np.zeros((3, 2)), np.zeros((2, 2)), np.zeros((0, 3))),
                 (np.zeros((2, 6)), np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((1, 2))),
                 (np.zeros((1, 6)), np.array([[0.3, np.nan]]), np.zeros((1, 2)), np.zeros((0, 3))),
                 (np.array([[0, 0, 0, np.inf, 0, 0]]), np.zeros((1, 2)), np.zeros((1, 2)), np.zeros((0, 3)))):
        try:
            PI.Scene(*args)
        except ValueError:
            continue
        raise AssertionError("Scene accepted %r" % (args,))
    blk, ks = np.zeros((2, 10, 10)), np.arange(2)
    try:
        PI.cut_square(blk, ks, [True, True], _frame(0, 0, 0.1), np.array([0.5, 0.5]), 0.01, _NoProvider(), "node 9")
        raise AssertionError("a square below 2 x 2 pixels was accepted")
    except ValueError as e:
        assert str(e).startswith("node 9: the square") and "fewer than 2 x 2" in str(e), e

    def other_planes(left, top, width, height, k_lo, k_hi):
        return np.zeros((3, height, width)), np.arange(5, 8), np.ones(3, bool), _frame(left, top, 0.1)
    try:
        PI.cut_square(blk, ks, [True, True], _frame(0, 0, 0.1), np.array([0.5, 0.5]), 2.0, other_planes, "node 9")
        raise AssertionError("a provider returning other planes was accepted")
    except ValueError as e:
        assert str(e) == "node 9: the square's planes differ from the block's", e


def test_determinism():
    raise unittest.SkipTest("13.2's functions use no randomness")


def test_edge_cases():
    # the square's last column on the block's last column (c1 = W) is cut; one pixel more is fetched
    blk, ks = np.arange(2 * 10 * 12, dtype=float).reshape(2, 10, 12), np.arange(2)
    fr = _frame(0, 0, 0.1)
    o = np.array([0.93, 0.47])                    # c1 = floor((0.93 + h)/0.1) + 1 = 12 at h = 0.25
    stack, _v, _s, fetched = PI.cut_square(blk, ks, [True, True], fr, o, 0.25, _NoProvider())
    assert not fetched and stack.shape[2] == 12 - math.ceil((0.93 - 0.25) / 0.1), stack.shape
    prov = _RecordingProvider(ks, [True, True])
    _st, _v, _s, fetched = PI.cut_square(blk, ks, [True, True], fr, o + np.array([0.1, 0.0]), 0.25, prov)
    assert fetched and len(prov.calls) == 1
    # a two-node branch: one segment
    br = Branch.from_points([[0, 0, 0], [0.5, 0, 0]], [0.2, 0.3], ids=[1, 2])
    assert PI.Scene.from_branch(br).seg_xyz.shape == (1, 6)
    # a cell without soma
    from allen_diameter.analysis import cell
    from allen_diameter.loading import swc_io
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "nosoma.swc"
        path.write_text("1 3 0 0 0 0.5 -1\n2 3 1 0 0 0.4 1\n3 3 2 0 0 0.3 2\n")
        sc = cell.cell_scene(swc_io.read_swc(str(path)), (3, 4), 0.1144)
    assert sc.soma.shape == (0, 3) and sc.seg_xyz.shape == (2, 6), (sc.soma.shape, sc.seg_xyz.shape)
    np.testing.assert_array_equal(sc.seg_ids, [[2, 1], [3, 2]])
    np.testing.assert_array_equal(sc.seg_radius, [[0.5, 0.4], [0.4, 0.3]])


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
    checks = [(name, obj) for name, obj in globals().items() if name.startswith("test_") and callable(obj)]
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
            status, detail = "FAIL", str(exc) or traceback.format_exc()
        except Exception:  # unexpected errors are reported, never hidden
            status, detail = "ERROR", traceback.format_exc()
        results.append((name, status, time.perf_counter() - start, detail))
    width = max([len(name) for name, _, _, _ in results] + [4])
    for name, status, seconds, detail in results:
        lines = detail.strip().splitlines()
        print(f"{status:<5}  {name:<{width}}  {seconds:8.3f}s  {lines[0] if lines else ''}".rstrip())
    for name, status, _seconds, detail in results:
        if status in ("FAIL", "ERROR"):
            print(f"\n---- {status}: {name}\n{detail.strip()}")
    counts = {s: sum(1 for r in results if r[1] == s) for s in ("PASS", "FAIL", "ERROR", "TODO", "SKIP")}
    print("\n-- " + ", ".join(f"{n} {s.lower()}" for s, n in counts.items()))
    return 1 if counts["FAIL"] + counts["ERROR"] + counts["TODO"] else 0


if __name__ == "__main__":
    sys.exit(main())
