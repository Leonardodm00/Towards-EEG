"""Smoke test for Block 13 in specs/SPEC.md: the Allen-guided plane and the
SGT fit inside the per-node chain (D-052.3, D-057 (iii), D-061, D-064).

13.2: the scene the iteration's background masks (D-064) and the square
about the node, shared by the diagnostics and the measurement:
plane_iteration.Scene (from_branch), scene_segments, cut_square,
iteration_state, analysis.cell.cell_scene, and survey.node_planes and
scripts/plane_differences.node_stack rebuilt on them. 13.3a: the fit at a
plane, plane_iteration.plane_inputs (refit_at_plane's inputs, factored out)
and sgt_at_plane (Block 12's fit on them, D-052.2). 13.3b: the chain,
node_pipeline.measure_node (the pilot, the iteration, the SGT fit at k_iter)
with measure_pilot, node_request, pilot_view and deliverable_flags; the
iteration's configuration; the CSV columns (cell.ITER_COLUMNS,
pilot_row_view). 13.6: D-068 (a non-converged iteration keeps its fit,
flagged iter_<ending>) and D-069 (round 1's line at least
iter_min_first_samples = 3 samples). The pilot against Block 5 at 84ea91f is
checks/block13_2026-10-20/check_13_3_pilot.py (it needs the history).

Checks
    test_known_answer   Scene.from_branch on a hand-made branch (the rows, radii
                        and ids by hand); scene_segments on a hand-made block
                        frame: a segment inside kept, one outside dropped, one
                        touching the block's bounding box at its edge kept and
                        one 1e-4 um beyond it dropped, one crossing the block
                        without an end inside kept, in_stretch by hand;
                        cut_square on a hand-made block: the slice by hand when
                        inside (no provider call), one provider call with the
                        arguments by hand when outside, valid ANDed;
                        plane_inputs on a hand-made block: the offsets, the
                        flat profile and B_bar of a constant plane by hand;
                        on the synthetic cell's 3.0 um tube the pilot's plane
                        is the first of its planes (stack_edge), and the
                        iteration with fit_at = sgt_at_plane leaves it, ends
                        converged and reports the fit of its last plane; the
                        edge rule on the iteration's curve and the flags of
                        deliverable_flags on hand-made records; the first
                        line's floor on iterate_planes' callables (round 1
                        only) and its refusals
    test_reference      on the synthetic cell (fixtures_cell, d 0.8 um):
                        survey.node_planes' segments equal the rule of
                        84ea91f written out here (bit for bit), and
                        node_stack's dict equals the construction of 84ea91f
                        written out here, key by key, for a square inside the
                        block and one fetched; at k* refit_at_plane (through
                        plane_inputs) returns the pilot's own fit bit for bit
                        and plane_inputs the pilot's B_bar and tilt; sgt_at_plane
                        equals fit_sgt.fit_plane on those inputs, and a fit
                        whose IRLS rule is not met is reported failed with its
                        numbers; pilot_view(measure_node) equals measure_pilot
                        (the cell, and a rendered phantom with noise and JPEG);
                        with Block 5's fit in its place measure_node's iteration
                        is Cell 4j's on the 3.0 um tube (planes, B_k and fits
                        bit for bit, G to 1e-12 relative: the squares differ);
                        a start too thin for 3 samples ends line_short without
                        the floor and picks a plane with it
    test_convergence    not a discretisation (skipped)
    test_invariants     every row of a lone branch's scene is in_stretch; a
                        permuted scene gives the permuted rows; a missing
                        scene and Scene.from_branch give equal rows;
                        plane_inputs reads the block, not the square: a square
                        inside the block and a fetched one give equal inputs
                        in every plane; phantoms, cells and the pilot run call
                        one measure_node, whose inputs are the node's, the
                        provider, the configuration, the registration and the
                        scene; scene None measures as Scene.from_branch
    test_contract       shapes and dtypes ((n, 9) float64, (0, 9) for an empty
                        scene); Scene refuses inconsistent shapes and
                        non-finite entries; cut_square refuses a square below
                        2 x 2 pixels and a provider returning other planes,
                        with the label in the message; sgt_at_plane's keys
                        and types; a NodeResult built from Block 5's fields
                        carries a pilot's defaults; measure_node's fields and
                        types, its deliverable and flags read from the record
                        of the round that picked k_iter; the iteration's
                        defaults and validate's refusals (13.6's floor too);
                        the estimator hash moved and a table of the older
                        hashes is refused; the CSV columns and pilot_row_view
    test_determinism    sgt_at_plane twice on one node: bit-identical fits; a
                        rendered phantom through run_replicate twice: the same
                        row (its d_hat / d printed)
    test_edge_cases     a square whose last column is the block's last is cut,
                        one pixel more is fetched; a two-node branch's scene;
                        a cell without soma gives soma (0, 3); a plane outside
                        the block's planes or invalid (no_plane), a profile
                        leaving the block (profile_nan) and a black plane
                        (bbar_nonpositive): sgt_at_plane returns NaN numbers,
                        no fit and that status, as refit_at_plane does;
                        measure_node makes one provider call (node_request's)
                        plus one per restack leaving the block; the flags are
                        k_iter's (the pilot's stack_edge gone); every section
                        missing: no plane, k_iter -1, fit_status none; a
                        profile leaving the block at k_iter: fit_failed, none;
                        a branch of radius 0.01 um measured from a 3-sample
                        first line; an iteration ending max_rounds kept in S
                        and flagged iter_max_rounds, a converged one unflagged

Run
    cd "Passive Features/Diameter Re-measurement"
    python tests/smoke/test_smoke_block13.py

Keep this file pure ASCII (hpc-python-compat).
"""
from __future__ import annotations

import dataclasses
import importlib.metadata
import math
import types
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


def _hand_state(cfg, planes, ks, valid, o=(5.0, 4.5), p=0.1144, theta=0.3, phi=0.2):
    """A state dict for plane_inputs: the block `planes` (n, H, W) at left 0, top 0, a pilot whose centre is the
    SWC node o (um) with heading theta and tilt phi, and a two-node branch through o along the heading."""
    o = np.asarray(o, dtype=float)
    t = np.array([math.cos(theta), math.sin(theta), 0.0])
    br = Branch.from_points(np.array([np.r_[o, 0.0] - 0.5 * t, np.r_[o, 0.0] + 0.5 * t]), 0.3)
    pilot = types.SimpleNamespace(cx_um=float(o[0]), cy_um=float(o[1]), theta_rad=float(theta), phi_rad=float(phi))
    return dict(ks=np.asarray(ks), valid=np.asarray(valid, bool), block=np.asarray(planes, float),
                block_frame=_frame(0, 0, p), o=o, branch=br, result=pilot)


def _fixture_state(cfg, swc, prov, nid, half=None):
    """node_stack's state of node nid (planes k_SWC +- 6, the Allen and d-lines at m = 3), and its restack."""
    import plane_differences as PD
    st = PD.node_stack(swc, prov, cfg, nid, None, 6, half, 0.0, None, 3.0, 3.0)

    def restack(h):
        return PD.node_stack(swc, prov, cfg, nid, None, 6, half, 0.0, float(h), 3.0, 3.0)
    return st, restack


_FIXTURE = {}


def _fixture_cell(d_true):
    """(cfg, swc, provider, branch, offset, scene) of the synthetic cell with a d_true tube, its first stretch as a
    Branch in the image frame and the cell's scene; cached (the provider serves the planes from memory)."""
    if d_true not in _FIXTURE:
        from allen_diameter.analysis import cell
        with tempfile.TemporaryDirectory() as tmp:
            cfg, swc, prov = _cell(tmp, d_true)
        types_ = cfg.acquisition.dendrite_swc_types
        branch, off = cell.stretch_branch(swc, cell.stretches(swc, types_)[0],
                                          cell.to_image_um(swc.xyz, cfg.acquisition.res0_um))
        _FIXTURE[d_true] = (cfg, swc, prov, branch, off, cell.cell_scene(swc, types_, cfg.acquisition.res0_um))
    return _FIXTURE[d_true]


def _recording(provider):
    """(provider that records its requests, the list of requests)."""
    calls = []

    def rec(*args):
        calls.append(args)
        return provider(*args)
    return rec, calls


def _fields_equal(a, b, names=None):
    """The fields (all, or `names`) where two NodeResults differ (arrays bit for bit, NaN = NaN)."""
    bad = []
    for f in dataclasses.fields(a):
        if names is not None and f.name not in names:
            continue
        try:
            _same(getattr(a, f.name), getattr(b, f.name), f.name)
        except AssertionError:
            bad.append(f.name)
    return bad


SGT_KEYS = ("d_hat_um", "fit_status", "v0_hat_um", "mu_hat_per_um", "B_bar", "bbar_ok", "w_hat_um", "beta_hat",
            "xi_hat", "zeta_hat", "alpha_hat", "irls_ok", "at_bound", "plane_fit", "v_um", "profile")


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
    # plane_inputs on a hand-made block: a constant plane gives the measurement's offsets, a flat profile at its
    # level (bilinear weights sum to 1: exact up to roundoff) and B_bar = that level (the masked median of a
    # constant block), status ok, the pilot's tilt
    from allen_diameter.analysis import profiles
    cfg = default_config()
    planes = np.stack([np.full((90, 90), 180.0), np.zeros((90, 90))])
    inp = PI.plane_inputs(_hand_state(cfg, planes, [3, 4], [True, True]), cfg, 3)
    np.testing.assert_array_equal(inp["v_um"], profiles.profile_offsets(cfg.measure))
    assert inp["status"] == "ok" and inp["B_bar"] == 180.0 and inp["bbar_ok"] is True and inp["phi_rad"] == 0.2, inp
    assert np.max(np.abs(inp["profile"] - 180.0)) <= 1e-12, inp["profile"]
    # the synthetic cell's 3.0 um tube: the pilot's plane k* lies at the end of its plane range, and the iteration
    # with fit_at = sgt_at_plane leaves it, ends converged, and reports the SGT fit of its last plane
    from allen_diameter.analysis import fit_sgt
    with tempfile.TemporaryDirectory() as tmp:
        cfgc, swc, prov = _cell(tmp, 3.0)
        st, restack = _fixture_state(cfgc, swc, prov, 4)
        res = PI.gradient_iteration(st, cfgc, line_mult=3.0, max_rounds=5, bg_margin_um=4.0, restack=restack,
                                    fit_at=lambda cur, k: PI.sgt_at_plane(cur, cfgc, k))
    k_star = int(st["result"].k_star)
    assert k_star == int(np.min(st["result"].focus_planes)) and "stack_edge" in st["result"].flags, st["result"].flags
    picked = [r for r in res["rounds"] if r["k"] is not None]
    assert res["status"] == "converged" and res["k_final"] != k_star, (res["status"], res["k_final"], k_star)
    pf = picked[-1]["plane_fit"]
    assert isinstance(pf, fit_sgt.PlaneSGT) and res["d_final_um"] == pf.final.d_um, res["d_final_um"]
    for r in res["rounds"]:
        if r.get("refit"):
            assert r["d_hat_um"] == r["plane_fit"].final.d_um and r["fit_status"] == r["plane_fit"].final.status
    # 13.3b: the edge rule on the iteration's curve (Block 5's: within edge_margin_planes = 1 of the first or last
    # plane with a finite G) and the flags at the deliverable's plane, by hand (deliverable_flags)
    from allen_diameter.analysis import node_pipeline as NP
    m = default_config().measure
    G = np.full(13, np.nan)
    G[2:11] = 1.0
    ks13 = np.arange(-6, 7)
    assert NP._iter_at_edge(dict(G=G), ks13, -4, 1) and NP._iter_at_edge(dict(G=G), ks13, 4, 1)
    assert not NP._iter_at_edge(dict(G=G), ks13, -3, 1) and not NP._iter_at_edge(dict(G=G), ks13, 3, 1)
    v = profiles.profile_offsets(m)
    dip = 200.0 - 100.0 * np.exp(-0.5 * (v / 0.3) ** 2)          # deep (not faint), one minimum (no crossing)
    rec = dict(k=-4, G=G, profile=dip, v_um=v, B_bar=200.0, bbar_ok=False, plane_fit=object(), alpha_hat=1.5,
               fit_status="at_bound")
    tilted, flat = types.SimpleNamespace(vertical=True, steep=True), types.SimpleNamespace(vertical=False, steep=False)
    got = NP.deliverable_flags(tilted, rec, ks13, m)
    assert got == ("vertical", "stack_edge", "bbar_few", "steep", "dark", "at_bound"), got
    ok = dict(rec, k=0, bbar_ok=True, alpha_hat=0.5, fit_status="converged")
    assert NP.deliverable_flags(flat, ok, ks13, m) == ()
    faint = dict(ok, profile=200.0 - 3.0 * np.exp(-0.5 * (v / 0.3) ** 2))      # a 3 gl dip: below faint_min_dip_gl
    assert NP.deliverable_flags(flat, faint, ks13, m) == ("faint",), NP.deliverable_flags(flat, faint, ks13, m)
    holed = dict(ok, profile=np.where(v > 2.5, np.nan, dip), plane_fit=None, fit_status="profile_nan")
    assert NP.deliverable_flags(flat, holed, ks13, m) == ("profile_nan",)
    assert NP.deliverable_flags(flat, None, ks13, m) == ("bbar_few",)     # no plane: as Block 5 without k*
    # 13.6 (D-069): the first line's floor, by hand on iterate_planes' callables: round 1 at max(m d0 / 2, floor),
    # the later rounds never floored; without the floor, round 1 at m d0 / 2
    for floor, first in ((0.1144, 0.1144), (None, 0.015)):
        seen = []

        def plane_of(h, seen=seen):
            seen.append(h)
            return 0, {}
        out = PI.iterate_planes(0.01, 3.0, plane_of, lambda k: dict(d_hat_um=0.02), 5, floor)
        assert seen == [first, 0.5 * 3.0 * 0.02] and out["status"] == "converged", (floor, seen, out["status"])
    for bad in (-0.1, float("nan"), float("inf")):
        try:
            PI.iterate_planes(0.01, 3.0, lambda h: (0, {}), lambda k: dict(d_hat_um=1.0), 5, bad)
        except ValueError:
            continue
        raise AssertionError("a first-line floor of %r was accepted" % bad)


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
    # 13.3a: at k* the inputs are the pilot's own (Block 5's final fit), so refit_at_plane through plane_inputs
    # returns the pilot's fit bit for bit; sgt_at_plane is fit_sgt.fit_plane on those inputs
    from allen_diameter.analysis import fit_sgt
    with tempfile.TemporaryDirectory() as tmp:
        cfg, swc, prov = _cell(tmp, 0.8)
        st, _restack = _fixture_state(cfg, swc, prov, 4)
        pilot, k = st["result"], int(st["result"].k_star)
        rf = PI.refit_at_plane(st, cfg, k)
        inp = PI.plane_inputs(st, cfg, k)
        sg = PI.sgt_at_plane(st, cfg, k)
    for key in ("d_hat_um", "mu_hat_per_um", "v0_hat_um", "B_bar"):
        _same(float(rf[key]), float(getattr(pilot, key)), "refit_at_plane at k*: " + key)
    assert rf["fit_status"] == pilot.fit_status and inp["status"] == "ok", (rf["fit_status"], inp["status"])
    assert inp["phi_rad"] == pilot.phi_rad and inp["B_bar"] == pilot.B_bar, (inp["phi_rad"], inp["B_bar"])
    ref = fit_sgt.fit_plane(inp["v_um"], inp["profile"], inp["phi_rad"], inp["B_bar"], cfg.measure).final
    for key, name in (("d_hat_um", "d_um"), ("mu_hat_per_um", "mu_per_um"), ("v0_hat_um", "v0_um"), ("w_hat_um", "w_um"),
                      ("beta_hat", "beta"), ("xi_hat", "xi"), ("zeta_hat", "zeta"), ("alpha_hat", "alpha_hat")):
        _same(float(sg[key]), float(getattr(ref, name)), "sgt_at_plane " + key)
    assert sg["fit_status"] == ref.status and sg["irls_ok"] == ref.irls_ok and sg["at_bound"] == tuple(ref.at_bound)
    # the wiring, not the accuracy: on this 0.8 um tube the fit reads about 0.85 um (the halo the table corrects);
    # a swapped tilt or background sends d to a bound (0.05 or 6 um)
    assert 0.7 <= sg["d_hat_um"] <= 1.0 and sg["fit_status"] == "converged", (sg["d_hat_um"], sg["fit_status"])
    # a fit whose IRLS rule is not met (one weighted solve allowed) is reported failed, with its numbers, so that
    # it still sizes the next line (D-052, D-047)
    one = dataclasses.replace(cfg, measure=dataclasses.replace(cfg.measure, sgt_irls_max_iter=1))
    sg1 = PI.sgt_at_plane(st, one, k)
    ref1 = fit_sgt.fit_plane(inp["v_um"], inp["profile"], inp["phi_rad"], inp["B_bar"], one.measure).final
    assert sg1["irls_ok"] is False and sg1["fit_status"] == "failed" == ref1.status, (sg1["irls_ok"], sg1["fit_status"])
    assert sg1["d_hat_um"] == ref1.d_um and math.isfinite(sg1["d_hat_um"]), sg1["d_hat_um"]
    # 13.3b: (i) the pilot inside measure_node is measure_pilot's; (ii) with Block 5's fit in place of Block 12's,
    # measure_node's iteration is Cell 4j's (gradient_iteration on node_stack's square): the same planes, G and
    # fits bit for bit -- on the 3.0 um tube, whose rounds restack
    from allen_diameter.analysis import node_pipeline as NP
    cfg, swc, prov, branch, off, scene = _fixture_cell(0.8)
    full = NP.measure_node(branch, off + 3, prov, cfg, None, scene)
    bad = _fields_equal(NP.measure_pilot(branch, off + 3, prov, cfg), NP.pilot_view(full))
    assert not bad and full.pilot_status == "converged" and full.iter_status == "converged", bad
    cfg3, swc3, prov3, branch3, off3, scene3 = _fixture_cell(3.0)
    i3 = off3 + 2
    saved = NP.sgt_at_plane
    NP.sgt_at_plane = PI.refit_at_plane
    try:
        old = NP.measure_node(branch3, i3, prov3, cfg3, None, scene3)
    finally:
        NP.sgt_at_plane = saved
    st, restack = _fixture_state(cfg3, swc3, prov3, int(branch3.ids[i3]))
    ref = PI.gradient_iteration(st, cfg3, line_mult=3.0, max_rounds=5, bg_margin_um=4.0, restack=restack)
    it = old.iteration
    assert [r["k"] for r in it["rounds"]] == [r["k"] for r in ref["rounds"]], ([r["k"] for r in it["rounds"]],
                                                                             [r["k"] for r in ref["rounds"]])
    # the two squares differ (measure_node's starts at +-block_half_um, node_stack's is sized for the pilot's d-line),
    # and bilinear samples of the same pixels taken from sub-arrays at other offsets agree to roundoff (5e-14 grey
    # levels on this node), so G agrees to 1e-12 relative; the fits read the block, not the square: d bit for bit
    for a, b in zip(it["rounds"], ref["rounds"]):
        Ga, Gb = np.asarray(a["G"], dtype=float), np.asarray(b["G"], dtype=float)
        assert np.array_equal(np.isfinite(Ga), np.isfinite(Gb)), ("G finite", a["round"])
        assert np.allclose(Ga[np.isfinite(Ga)], Gb[np.isfinite(Gb)], rtol=1e-12, atol=0.0), ("G", a["round"], Ga, Gb)
        _same(float(a.get("d_hat_um", np.nan)), float(b.get("d_hat_um", np.nan)), "d of round %d" % a["round"])
    _same(np.asarray(it["background"]), np.asarray(ref["background"]), "B_k")
    assert it["status"] == ref["status"] and it["k_final"] == ref["k_final"] == old.k_iter, (it["status"], ref["status"])
    _same(float(it["d_final_um"]), float(ref["d_final_um"]), "d_final")
    assert it["n_restacks"] >= 1, it["n_restacks"]
    # (iii) on a rendered phantom (noise and JPEG), one seed: the pilot inside measure_node is measure_pilot's
    import test_smoke_node_pipeline as T5
    from allen_diameter.model import geometry
    pcfg = T5.small_cfg()
    c = (0.03, -0.02, 0.05)
    tube = geometry.Tube(c, 0.4, 0.0, 0.6, 1.0, 6.0, "axial")
    br = T5.phantom_branch(c, 0.0, 0.6, 0.4)
    a = NP.measure_pilot(br, 4, T5.make_provider([tube], [1.0], pcfg, 5), pcfg)
    b = NP.measure_node(br, 4, T5.make_provider([tube], [1.0], pcfg, 5), pcfg)
    bad = _fields_equal(a, NP.pilot_view(b))
    assert not bad and b.iter_status in PI.ITER_STATUSES, (bad, b.iter_status)
    # 13.6 (D-069): on the fixture, a start too thin for 3 samples (d0 = 0.02 um: round 1's line +-0.03 um) ends
    # line_short without the floor; with min_first_samples = 3 round 1's line is one profile step, 3 samples, and
    # picks a plane
    st8, restack8 = _fixture_state(cfg, swc, prov, 4)
    step = cfg.measure.profile_step_um
    bare = PI.gradient_iteration(st8, cfg, d_start_um=0.02, line_mult=3.0, restack=restack8)
    floored = PI.gradient_iteration(st8, cfg, d_start_um=0.02, line_mult=3.0, restack=restack8, min_first_samples=3)
    assert bare["status"] == "line_short" and bare["k_final"] is None, (bare["status"], bare["rounds"])
    r1 = floored["rounds"][0]
    assert r1["half_um"] == step and r1["n_samples"] == 3 and r1["k"] is not None, r1
    assert floored["rounds"][1]["half_um"] == 0.5 * 3.0 * floored["rounds"][0]["d_hat_um"], floored["rounds"][1]
    for bad in (2, 4, 1, 3.5, True, "3"):
        try:
            PI.gradient_iteration(st8, cfg, d_start_um=0.02, line_mult=3.0, min_first_samples=bad)
        except ValueError:
            continue
        raise AssertionError("min_first_samples %r accepted" % (bad,))


def test_convergence():
    raise unittest.SkipTest("Block 13 holds no discretisation (the fit's quadrature is Block 12's)")


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
    # plane_inputs reads the block, not the square: a square inside the block and a fetched one give equal inputs
    # in every plane (so a restack never changes a fit)
    with tempfile.TemporaryDirectory() as tmp:
        cfg, swc, prov = _cell(tmp, 0.8)
        a, _r = _fixture_state(cfg, swc, prov, 4)
        b, _r = _fixture_state(cfg, swc, prov, 4, half=5.0)
        assert not a["square_fetched"] and b["square_fetched"]
        for k in a["ks"]:
            ia, ib = PI.plane_inputs(a, cfg, int(k)), PI.plane_inputs(b, cfg, int(k))
            for key in ia:
                _same(ia[key], ib[key], "plane_inputs[%r] at plane %d" % (key, int(k)))
    # 13.3b: phantoms and real nodes call one function object (D-057 (iii)), whose only inputs beyond the node's are
    # the provider, the configuration, the registration and the scene; a missing scene is the branch's own tube
    import inspect
    from allen_diameter.analysis import cell, node_pipeline as NP, phantoms, survey
    assert phantoms.measure_node is NP.measure_node and cell.measure_node is NP.measure_node \
        and survey.measure_node is NP.measure_node
    assert list(inspect.signature(NP.measure_node).parameters) == ["branch", "i", "provider", "cfg", "reg", "scene"]
    cfg, swc, prov, branch, off, scene = _fixture_cell(0.8)
    a = NP.measure_node(branch, off + 3, prov, cfg)
    b = NP.measure_node(branch, off + 3, prov, cfg, None, PI.Scene.from_branch(branch))
    bad = _fields_equal(a, b, [f.name for f in dataclasses.fields(a) if f.name not in ("plane_fit", "iteration")])
    assert not bad, bad
    # the scene's other segments are masked (D-064): one more traced segment through the block, 3.4 um across
    # from the stretch, leaves fewer pixels far from the traced structures (a smaller margin, or a smaller share)
    extra = PI.Scene(np.vstack([scene.seg_xyz, [[8.0, 4.0, 0.05, 14.0, 4.6, 0.05]]]),
                     np.vstack([scene.seg_radius, [[0.3, 0.3]]]), np.vstack([scene.seg_ids, [[100, 101]]]),
                     scene.soma)
    with_x = NP.measure_node(branch, off + 3, prov, cfg, None, extra).iteration
    plain = NP.measure_node(branch, off + 3, prov, cfg, None, scene).iteration
    assert (with_x["bg_margin_used"], with_x["bg_frac"]) < (plain["bg_margin_used"], plain["bg_frac"]), \
        ((with_x["bg_margin_used"], with_x["bg_frac"]), (plain["bg_margin_used"], plain["bg_frac"]))


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
    # sgt_at_plane's record: its keys in order, and the types of a fit
    from allen_diameter.analysis import fit_sgt
    with tempfile.TemporaryDirectory() as tmp:
        cfg, swc, prov = _cell(tmp, 0.8)
        st, _r = _fixture_state(cfg, swc, prov, 4)
        sg = PI.sgt_at_plane(st, cfg, int(st["result"].k_star))
    assert tuple(sg) == SGT_KEYS, tuple(sg)
    for key in ("d_hat_um", "v0_hat_um", "mu_hat_per_um", "B_bar", "w_hat_um", "beta_hat", "xi_hat", "zeta_hat",
                "alpha_hat"):
        assert isinstance(sg[key], float) and math.isfinite(sg[key]), key
    assert isinstance(sg["fit_status"], str) and isinstance(sg["bbar_ok"], bool) and isinstance(sg["irls_ok"], bool)
    assert isinstance(sg["at_bound"], tuple) and isinstance(sg["plane_fit"], fit_sgt.PlaneSGT)
    assert sg["v_um"].dtype == np.float64 and sg["profile"].shape == sg["v_um"].shape
    # 13.3b: a NodeResult built from Block 5's fields only (as before Block 13) carries a pilot's defaults
    from allen_diameter import config as C
    from allen_diameter.analysis import cell, node_pipeline as NP, table
    cfg, swc, prov, branch, off, scene = _fixture_cell(0.8)
    pil = NP.measure_pilot(branch, off + 3, prov, cfg)
    legacy = NP.NodeResult(*[getattr(pil, f.name) for f in dataclasses.fields(NP.NodeResult)
                             if f.name not in NP.ITER_FIELDS])
    assert legacy.k_iter == -1 and legacy.iter_status == "" and legacy.pilot_status == "" and legacy.plane_fit is None
    assert NP.pilot_view(legacy) is legacy and NP.pilot_view(pil) is pil
    assert tuple(f.name for f in dataclasses.fields(NP.NodeResult))[-len(NP.ITER_FIELDS):] == NP.ITER_FIELDS
    # measure_node's fields, and the deliverable read from the record of the round that picked k_iter
    res = NP.measure_node(branch, off + 3, prov, cfg, None, scene)
    assert isinstance(res.k_iter, int) and res.iter_status in PI.ITER_STATUSES and isinstance(res.n_rounds, int) \
        and isinstance(res.n_restacks, int) and isinstance(res.irls_ok, bool) and isinstance(res.at_bound, tuple)
    for name in ("w_hat_um", "beta_hat", "xi_hat", "zeta_hat", "d_pilot_um", "mu_pilot_per_um", "v0_pilot_um",
                 "alpha_pilot", "B_bar_pilot", "d_hat_um", "B_bar"):
        assert isinstance(getattr(res, name), float) and math.isfinite(getattr(res, name)), name
    assert res.pilot_status == pil.fit_status and res.pilot_flags == pil.flags and res.d_pilot_um == pil.d_hat_um
    assert isinstance(res.plane_fit, fit_sgt.PlaneSGT) and "st" not in res.iteration
    rec = [r for r in res.iteration["rounds"] if r["k"] is not None][-1]
    k_swc = int(round(float(branch.xyz_um[off + 3, 2]) / cfg.acquisition.dz_um))
    assert res.k_iter == rec["k"] == res.iteration["k_final"] and res.B_bar == rec["B_bar"]
    assert res.d_hat_um == rec["d_hat_um"] == res.plane_fit.final.d_um and res.fit_status == rec["fit_status"]
    assert res.flags == NP.deliverable_flags(res, rec, np.arange(k_swc - 6, k_swc + 7), cfg.measure), res.flags
    # the iteration's configuration: the decided values (D-044, D-042, D-052.3, D-036), and validate's refusals
    base = C.default_config()
    m = base.measure
    assert (m.iter_line_mult, m.iter_max_rounds, m.iter_planes_half, m.iter_bg_margin_um) == (3.0, 5, 6, 4.0)
    for name, value in (("iter_line_mult", 0.0), ("iter_line_mult", -1.0), ("iter_line_mult", float("inf")),
                        ("iter_line_mult", float("nan")), ("iter_line_mult", True), ("iter_max_rounds", 0),
                        ("iter_max_rounds", 5.5), ("iter_max_rounds", True), ("iter_max_rounds", "5"),
                        ("iter_planes_half", 0), ("iter_planes_half", 2.5), ("iter_bg_margin_um", -0.1),
                        ("iter_bg_margin_um", float("nan"))):
        try:
            dataclasses.replace(base, measure=dataclasses.replace(m, **{name: value})).validate()
        except ValueError:
            continue
        raise AssertionError("validate accepted %s = %r" % (name, value))
    dataclasses.replace(base, measure=dataclasses.replace(m, iter_max_rounds=5.0)).validate()
    # the estimator hash moved, and a table of the hashes before Block 13 is refused (procedure s.3.11)
    h = base.signature_hash("estimator")
    assert h not in ("1bb3585b4a4f1373", "535847535e83cf7d"), h
    for old in ("1bb3585b4a4f1373", "535847535e83cf7d"):
        try:
            table.BiasTable.check_estimator(types.SimpleNamespace(estimator_hash=old), base)
        except ValueError:
            continue
        raise AssertionError("a table of estimator %s was not refused" % old)
    table.BiasTable.check_estimator(types.SimpleNamespace(estimator_hash=h), base)
    # the CSV columns: Block 13's after Block 8's, tuples joined; a row read as the pilot's
    assert cell.CSV_COLUMNS[-len(cell.ITER_COLUMNS):] == cell.ITER_COLUMNS
    assert len(set(cell.CSV_COLUMNS)) == len(cell.CSV_COLUMNS)
    cols = cell.iter_columns(res)
    assert tuple(cols) == cell.ITER_COLUMNS and cols["at_bound"] == ";".join(res.at_bound)
    assert cols["pilot_flags"] == ";".join(res.pilot_flags)
    row = dict(d_hat_um=1.0, mu_hat_per_um=1.1, v0_hat_um=0.1, alpha_hat=0.5, B_bar=200.0, fit_status="at_bound",
               flags="dark", in_S=False, reject="status:at_bound;dark", d_pilot_um=2.0, mu_pilot_per_um=2.1,
               v0_pilot_um=0.2, alpha_pilot=0.6, B_bar_pilot=201.0, pilot_status="converged", pilot_flags="",
               in_S_pilot=True, reject_pilot="", k_star=3)
    view = cell.pilot_row_view(row)
    assert (view["d_hat_um"], view["mu_hat_per_um"], view["v0_hat_um"], view["alpha_hat"], view["B_bar"]) == \
        (2.0, 2.1, 0.2, 0.6, 201.0) and view["fit_status"] == "converged" and view["flags"] == ""
    assert view["in_S"] is True and view["reject"] == "" and view["k_star"] == 3 and row["d_hat_um"] == 1.0
    for legacy_row in (dict(d_hat_um=1.0, fit_status="converged"), dict(row, pilot_status="")):
        back = cell.pilot_row_view(legacy_row)
        assert back == legacy_row and back is not legacy_row
    # 13.6: the first line's floor in the configuration (D-069), validate's refusals, the hash moved again
    assert m.iter_min_first_samples == 3, m.iter_min_first_samples
    for value in (1, 2, 4, 3.5, True, "3", float("nan")):
        try:
            dataclasses.replace(base, measure=dataclasses.replace(m, iter_min_first_samples=value)).validate()
        except ValueError:
            continue
        raise AssertionError("validate accepted iter_min_first_samples = %r" % (value,))
    dataclasses.replace(base, measure=dataclasses.replace(m, iter_min_first_samples=5)).validate()
    assert h != "1654924e11020dd9", h


def test_determinism():
    # 13.2's functions and sgt_at_plane hold no randomness: twice on one node, the same fit bit for bit
    with tempfile.TemporaryDirectory() as tmp:
        cfg, swc, prov = _cell(tmp, 0.8)
        st, _r = _fixture_state(cfg, swc, prov, 3)
        k = int(st["result"].k_star)
        a, b = PI.sgt_at_plane(st, cfg, k), PI.sgt_at_plane(st, cfg, k)
    for key in SGT_KEYS:
        if key != "plane_fit":
            _same(a[key], b[key], key)
    assert a["plane_fit"].final.q == b["plane_fit"].final.q and a["plane_fit"].final.cost == b["plane_fit"].final.cost
    # 13.3b: a rendered phantom end to end (Block 6's run_replicate: the chain the table is built with) twice, the
    # same row; its d_hat / d reported, the bias the table will correct (not a check)
    from allen_diameter.analysis import phantoms
    base = default_config()
    pcfg = dataclasses.replace(base, measure=dataclasses.replace(base.measure, block_half_um=3.5),
                               renderer=dataclasses.replace(base.renderer, h_g_um_thin=0.1144 / 8),
                               phantom=dataclasses.replace(base.phantom, d_range_um=(0.6, 1.2), phi_range_deg=(0.0, 10.0)))
    a, b = phantoms.run_replicate(pcfg, SEED, 0), phantoms.run_replicate(pcfg, SEED, 0)
    for key in a:
        if key != "seconds":
            _same(a[key], b[key], "run_replicate[%r]" % key)
    new_cols = {"k_iter", "iter_status", "n_rounds", "n_restacks", "w_hat_um", "beta_hat", "xi_hat", "zeta_hat",
                "irls_ok", "at_bound", "d_pilot_um", "alpha_pilot", "pilot_status"}
    assert new_cols <= set(a) and a["iter_status"] in PI.ITER_STATUSES and a["pilot_status"], sorted(a)
    print("    phantom d %.3f um, phi %.1f deg, mu %.2f /um: d_hat / d %.3f (the pilot's %.3f), k* %d, k_iter %d (%s, "
          "%d rounds, %d restacks), %s, in S %s, %.1f s"
          % (a["d_um"], math.degrees(a["phi_rad"]), a["mu_per_um"], a["ratio"], a["d_pilot_um"] / a["d_um"],
             a["k_star"], a["k_iter"], a["iter_status"], a["n_rounds"], a["n_restacks"], a["fit_status"], a["in_S"],
             a["seconds"]))


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
    # no inputs at the plane: outside the block's planes or invalid (no_plane), a profile leaving the block
    # (profile_nan), a black plane (bbar_nonpositive): plane_inputs says why, sgt_at_plane returns NaN numbers
    # and no fit with that status, as refit_at_plane does
    cfg = default_config()
    planes = np.stack([np.full((90, 90), 180.0), np.zeros((90, 90)), np.full((90, 90), 180.0)])
    st = _hand_state(cfg, planes, [3, 4, 5], [True, True, False])
    corner = _hand_state(cfg, planes, [3, 4, 5], [True, True, False], o=(0.5, 0.5))
    for state, k, want in ((st, 9, "no_plane"), (st, 5, "no_plane"), (st, 4, "bbar_nonpositive"),
                           (corner, 3, "profile_nan")):
        inp, sg, rf = PI.plane_inputs(state, cfg, k), PI.sgt_at_plane(state, cfg, k), PI.refit_at_plane(state, cfg, k)
        assert inp["status"] == sg["fit_status"] == rf["fit_status"] == want, (k, want, inp["status"], sg["fit_status"])
        assert sg["plane_fit"] is None and sg["irls_ok"] is None and sg["at_bound"] == (), (k, want)
        assert all(math.isnan(sg[x]) for x in ("d_hat_um", "v0_hat_um", "mu_hat_per_um", "w_hat_um", "beta_hat",
                                               "xi_hat", "zeta_hat", "alpha_hat")), (k, want)
        if want == "no_plane":
            assert inp["v_um"] is None and inp["profile"] is None and math.isnan(inp["B_bar"]) and not inp["bbar_ok"]
        else:
            assert inp["profile"] is not None and math.isfinite(inp["B_bar"]), (k, want)
    assert np.isnan(PI.plane_inputs(corner, cfg, 3)["profile"]).any()
    # 13.3b: one provider call, node_request's, plus one per restack that leaves the block (D-061.1)
    from allen_diameter.analysis import node_pipeline as NP
    cfg, swc, prov, branch, off, scene = _fixture_cell(0.8)
    rec_p, calls = _recording(prov)
    r = NP.measure_node(branch, off + 3, rec_p, cfg, None, scene)
    assert len(calls) == 1 and r.n_restacks == 0 and calls[0] == NP.node_request(branch, off + 3, cfg), calls
    cfg3, swc3, prov3, branch3, off3, scene3 = _fixture_cell(3.0)
    rec_p, calls = _recording(prov3)
    r3 = NP.measure_node(branch3, off3 + 2, rec_p, cfg3, None, scene3)
    k_swc = int(round(float(branch3.xyz_um[off3 + 2, 2]) / cfg3.acquisition.dz_um))
    assert r3.n_restacks >= 1 and len(calls) == 1 + r3.n_restacks, (len(calls), r3.n_restacks)
    assert all(c[4:] == (k_swc - 6, k_swc + 6) for c in calls[1:]), calls
    # the squares by hand: the first +-block_half_um about the SWC node, then one of half-width h + 2 pixels for
    # each round whose line of half-length h comes within 2 pixels of the current square's edge
    halves, state = [], NP.iteration_state

    def spy(*args, **kw):
        halves.append(float(args[10]))
        return state(*args, **kw)
    NP.iteration_state = spy
    try:
        again = NP.measure_node(branch3, off3 + 2, prov3, cfg3, None, scene3)
    finally:
        NP.iteration_state = state
    px = cfg3.acquisition.res0_um
    want, cur = [float(cfg3.measure.block_half_um)], float(cfg3.measure.block_half_um)
    for rnd in again.iteration["rounds"]:
        if rnd["half_um"] + 2.0 * px > cur + 1e-9:
            cur = rnd["half_um"] + 2.0 * px
            want.append(cur)
    assert halves == want and len(want) == 1 + again.n_restacks, (halves, want)
    # the flags are those of k_iter: the pilot's plane is the first of its planes, the iteration's is inside K
    assert "stack_edge" in r3.pilot_flags and "stack_edge" not in r3.flags and r3.k_iter != r3.k_star, \
        (r3.pilot_flags, r3.flags, r3.k_star, r3.k_iter)
    # where k* and k_iter differ, the deliverable's B_bar and fit are k_iter's, and the pilot's readers read the
    # pilot: pilot_view, the pilot's S in the pilot row, Block 10's reasons, the drawn profile's model
    from allen_diameter.analysis import calibration, survey
    rec3 = [r for r in r3.iteration["rounds"] if r["k"] is not None][-1]
    pil3 = NP.measure_pilot(branch3, off3 + 2, prov3, cfg3)
    assert r3.B_bar == rec3["B_bar"] != pil3.B_bar == r3.B_bar_pilot and r3.d_hat_um == rec3["d_hat_um"], \
        (r3.B_bar, pil3.B_bar)
    assert not _fields_equal(pil3, NP.pilot_view(r3)), _fields_equal(pil3, NP.pilot_view(r3))
    assert pil3.k_star == -4 and int(np.min(pil3.focus_planes)) == -4, (pil3.k_star, pil3.focus_planes)  # Block 5's
    row = survey.pilot_row(r3, 0.3, cfg3)
    assert "stack_edge" in row["reject_pilot"] and "stack_edge" not in row["reject"], (row["reject"], row["reject_pilot"])
    assert calibration.calibration_reasons(r3, cfg3) == calibration.calibration_reasons(pil3, cfg3)
    _same(survey.profile_at_node(prov3, r3, cfg3)[2], survey.profile_at_node(prov3, pil3, cfg3)[2], "drawn model")
    # no plane at all (every section missing): no fit and k_iter -1, the flags of Block 5 without k*
    def missing(*args):
        blk, ks, valid, fr = prov(*args)
        return np.full_like(blk, 255), ks, np.zeros_like(valid), fr
    r0 = NP.measure_node(branch, off + 3, missing, cfg, None, scene)
    assert r0.k_iter == -1 and r0.iter_status == "no_plane" and r0.fit_status == "none" and r0.plane_fit is None, r0
    assert math.isnan(r0.d_hat_um) and math.isnan(r0.B_bar) and "bbar_few" in r0.flags, (r0.d_hat_um, r0.flags)
    # a plane without inputs: with block_half_um 1.5 um the +-3 um profile across the node leaves the block (which
    # reaches about 2.7 um across it), the line of round 1 (+-0.9 um) does not -- the iteration ends fit_failed on
    # its plane, which carries no fit
    small = dataclasses.replace(cfg, measure=dataclasses.replace(cfg.measure, block_half_um=1.5))
    rs = NP.measure_node(branch, off + 3, prov, small, None, scene)
    assert rs.iter_status == "fit_failed" and rs.k_iter != -1 and rs.fit_status == "none", (rs.iter_status, rs.k_iter)
    assert math.isnan(rs.d_hat_um) and "profile_nan" in rs.flags and math.isfinite(rs.B_bar), (rs.flags, rs.B_bar)
    # 13.6 (D-069): a branch whose radius is too thin for a 3-sample first line (0.01 um) measures: round 1's line
    # is one profile step, 3 samples, and the node gets a plane
    thin = dataclasses.replace(branch, radius_um=np.full_like(np.asarray(branch.radius_um, dtype=float), 0.01))
    rt = NP.measure_node(thin, off + 3, prov, cfg, None, scene)
    first = rt.iteration["rounds"][0]
    assert first["half_um"] == cfg.measure.profile_step_um and first["n_samples"] == 3, first
    assert rt.k_iter != -1 and not any(f.startswith("iter_line_short") for f in rt.flags), (rt.k_iter, rt.flags)
    # 13.6 (D-068): an iteration that ends without converging keeps its fit, stays in S and is flagged; a converged
    # one carries no label
    from allen_diameter.analysis import phantoms
    one = dataclasses.replace(cfg, measure=dataclasses.replace(cfg.measure, iter_max_rounds=1))
    rm = NP.measure_node(branch, off + 3, prov, one, None, scene)
    assert rm.iter_status == "max_rounds" and rm.flags[-1] == "iter_max_rounds", (rm.iter_status, rm.flags)
    assert rm.fit_status == "converged" and math.isfinite(rm.d_hat_um) and phantoms.reject_reasons(rm) == [], \
        (rm.fit_status, rm.d_hat_um, phantoms.reject_reasons(rm))
    assert "iter_max_rounds" not in phantoms.SELECTION_FLAGS
    assert not any(f.startswith("iter_") for f in r.flags) and r.iter_status == "converged", r.flags
    assert "iter_no_plane" in r0.flags and "iter_fit_failed" in rs.flags, (r0.flags, rs.flags)


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
