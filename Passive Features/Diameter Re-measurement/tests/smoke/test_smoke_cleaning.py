"""Smoke test for the cleaning operator and the two tiers -- Block 28 in
specs/SPEC.md (plan step C3; D-130.O1, D-132).

Checks
    test_known_answer   S = 0 returns the block as float64 [exact]; the exact
                        inverse under the renderer's own rule: with Block 24
                        (partition_vertical, then linear), a node tube and a
                        neighbour tube whose projections do not overlap,
                        B tau_scene cleaned with S = the neighbour's deficit
                        rendered alone equals B tau_node [1e-9 gl at B = 210];
                        the tiers on constructed scenes (far lateral: 1; far
                        in depth: 1; close and in focus: 2; directly above,
                        unattached: 2; attached at a branch point and not
                        close: 1, with its unattached twin: 2; soma: 1)
    test_reference      added_gl equals the mean of B S [1e-12]; the tier
                        rule against a brute force written independently
                        (dense random points of the reach disc, nearest
                        points by sampling the axis) on random scenes
    test_convergence    SKIP (no discretisation in the cleaning operator)
    test_invariants     the hybrid: the residual of the same inverse, a
                        reported measurement with no bound (amendment 1,
                        D-135), for the draft's three cases and for two equal
                        tubes (r 0.25 um, mu 3 /um) at tilt 0 and 45 deg,
                        centres 1.0, 2.0, 3.0 um apart, printed as a table
                        with the partition's residual as its floor; cleaning
                        is linear in S and the cleaned block minus the block
                        is B S
    test_contract       CleanResult's shapes and dtypes; the three forms of
                        B_gl agree [exact]; any real dtype of block; tiers
                        (n_q + n_s,) int64 in {1, 2} in neighbour order
    test_determinism    both functions twice give bit-identical output; the
                        inputs are not changed
    test_edge_cases     ValueError for unequal shapes, a non-finite or
                        negative S, B <= 0 (and non-finite B), a bad B shape,
                        a node position that is not 3 finite numbers, an own
                        stretch with no segment; a scene without neighbours
                        gives only the soma tiers

Block 25 (model/shadow.py) is not imported: S is built from Block 24
(model/scene_render.py) as the neighbour's deficit 1 - tau rendered alone.

Run
    cd "Passive Features/Diameter Re-measurement"
    python3 tests/smoke/test_smoke_cleaning.py

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

from allen_diameter.analysis import background  # noqa: E402
from allen_diameter.analysis import cleaning as CL  # noqa: E402
from allen_diameter.analysis.node_features import FeaturedScene  # noqa: E402
from allen_diameter.config import default_config  # noqa: E402
from allen_diameter.model import geometry as G  # noqa: E402
from allen_diameter.model import render as R  # noqa: E402
from allen_diameter.model import scene_render as SR  # noqa: E402

SEED = 20261113
REPORT_PACKAGES = ("numpy", "scipy")
ARTIFACT_DIR = HERE / "artifacts"
_SAVED_ARTIFACTS = []
PX = 0.1144
B_REF = 210.0
MEASURED = {}          # numbers printed by main() after the table
HYBRID_TABLE = []      # rows of the hybrid residual table (Block 28, amendment 1)
CFG = default_config()


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


def rcfg(absorption, **changes):
    """Default RendererConfig with the absorption chosen; the hybrid at 8 x 16 directions (cost; the
    cleaning's own shadow model uses 8 x 16 too, CleaningConfig.shadow_ray_n_rho / _psi)."""
    base = dict(absorption=absorption, ray_n_rho=8, ray_n_psi=16)
    base.update(changes)
    return dataclasses.replace(default_config().renderer, **base)


# A 24 x 24 pixel patch about the origin (pixel centres at integer multiples of PX), 8 px padding.
LEFT, TOP, WIDTH, HEIGHT, PAD = -12, -12, 24, 24, 8
KS = np.arange(-3, 4)                                     # planes -3 .. 3, z_k = k dz


def render_pixels(tubes, mus, rc, frame=None):
    """Block 24 planes of a scene, pixel means of the fine planes (before the camera chain):
    (n_planes, height, width) and the slab lattice. frame = (left, top, width, height) in pixels,
    default the 24 x 24 patch about the origin."""
    acq = default_config().acquisition
    left, top, width, height = (LEFT, TOP, WIDTH, HEIGHT) if frame is None else frame
    f = R.fine_factor(min(t.d for t in tubes), rc, acq.res0_um)
    grid, inner = R.block_fine_grid(left, top, width, height, acq.res0_um, f, PAD)
    res = SR.render_scene(tubes, mus, KS * acq.dz_um, grid, rc, acq.light_direction,
                          reduce=lambda plane: R.pixel_integrate(plane[inner[0], inner[1]], f))
    return res.tau, res.zeta


def pair(kind):
    """A node tube and a neighbour tube with disjoint projections and equal depth ranges on the grid
    (same r, tilt, half-length and centre depth, centres on pixel centres, so the two tubes sample the
    grid alike and the scene, each alone, share one slab lattice). The blurred shadows overlap."""
    if kind == "flat":
        node = G.Tube((0.0, 0.0, 0.0), 0.25, 0.0, 0.0, 1.0, 1.0, "axial")
        nb = G.Tube((0.0, 9 * PX, 0.0), 0.25, 0.0, 0.0, 1.0, 1.0, "axial")      # 1.03 um centre to centre
    elif kind == "tilted":
        node = G.Tube((0.0, 0.0, 0.0), 0.3, math.radians(30.0), 0.0, 1.0, 1.0, "axial")
        nb = G.Tube((0.0, 8 * PX, 0.0), 0.3, math.radians(30.0), 0.0, 1.0, 1.0, "axial")  # 0.92 um apart
    else:
        raise KeyError(kind)
    return node, nb


def inverse_residual(kind, absorption, mus, tubes=None, frame=None, same_lattice=True):
    """max |B tau_scene cleaned with S = 1 - tau_nb  -  B tau_node| (gl), B max (1 - tau_node)(1 - tau_nb)
    (gl, the withdrawn bound, printed for comparison), and the peaks of B S and of the node's dip (gl).
    tubes = (node, nb) overrides pair(kind); same_lattice: assert the three renders share one slab lattice."""
    node, nb = pair(kind) if tubes is None else tubes
    rc = rcfg(absorption)
    t_scene, z_s = render_pixels([node, nb], list(mus), rc, frame)
    t_node, z_n = render_pixels([node], [mus[0]], rc, frame)
    t_nb, z_b = render_pixels([nb], [mus[1]], rc, frame)
    if same_lattice:
        assert z_s.size == z_n.size == z_b.size, "the three renders must share one slab lattice"
        # one lattice up to roundoff of z_lo.min() (the two tubes sit at different y): 1e-12 um
        check_close("lattice node", z_n, z_s, rtol=0, atol=1e-12)
        check_close("lattice neighbour", z_b, z_s, rtol=0, atol=1e-12)
    S = 1.0 - t_nb
    assert np.all(S >= -1e-12), "a neighbour's deficit is >= 0 up to roundoff"
    S = np.clip(S, 0.0, None)                            # roundoff of 1 - tau where tau = 1 to the last bit
    got = CL.clean_block(B_REF * t_scene, S, B_REF, CFG)
    resid = float(np.max(np.abs(got.block - B_REF * t_node)))
    bound = float(B_REF * np.max((1.0 - t_node) * (1.0 - t_nb)))
    # the test is not vacuous: the neighbour's shadow reaches the node's own footprint
    return resid, bound, float(np.max(B_REF * S)), float(np.max(B_REF * (1.0 - t_node)))


def fscene(segs, soma=()):
    """FeaturedScene from rows (child_id, parent_id, (x0, y0, z0), (x1, y1, z1), d0, d1); the
    parent end first in seg_xyz, as Block 23 writes it; mu 1/um 1.0 (not read by the tiers)."""
    n = len(segs)
    xyz = np.array([list(s[2]) + list(s[3]) for s in segs], dtype=float).reshape(n, 6)
    ids = np.array([[s[0], s[1]] for s in segs], dtype=np.int64).reshape(n, 2)
    d = np.array([[s[4], s[5]] for s in segs], dtype=float).reshape(n, 2)
    return FeaturedScene(seg_xyz=xyz, seg_d_um=d, seg_mu_per_um=np.ones((n, 2)), seg_ids=ids,
                         seg_type=np.full(n, 3, dtype=np.int64),
                         soma=np.array(list(soma), dtype=float).reshape(-1, 3))


# The own stretch: nodes 1 (-3, 0, 0) -> 2 (0, 0, 0) -> 3 (2.5, 0, 0), d = 1 um; the node is 2.
OWN_SEGS = [(2, 1, (-3.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1.0, 1.0),
            (3, 2, (0.0, 0.0, 0.0), (2.5, 0.0, 0.0), 1.0, 1.0)]
OWN_IDS = (1, 2, 3)
NODE = (0.0, 0.0, 0.0)
CASES = {   # name: (segment row, expected tier)
    "far_lateral": ((11, 10, (-1.0, 5.0, 0.0), (1.0, 5.0, 0.0), 1.0, 1.0), 1),
    "far_in_depth": ((21, 20, (-1.0, 1.5, 3.0), (1.0, 1.5, 3.0), 1.0, 1.0), 1),
    "close_in_focus": ((31, 30, (-1.0, 1.5, 0.3), (1.0, 1.5, 0.3), 1.0, 1.0), 2),
    "directly_above_unattached": ((41, 40, (0.0, -3.0, 4.0), (0.0, 3.0, 4.0), 0.6, 0.6), 2),
    "attached_not_close": ((51, 3, (2.5, 0.0, 0.0), (2.5, 3.0, 1.0), 1.0, 1.0), 1),
    "unattached_twin": ((61, 60, (2.5, 0.0, 0.0), (2.5, 3.0, 1.0), 1.0, 1.0), 2),
}


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    # S = 0 returns the block as float64, exactly
    rng = make_rng()
    blk = rng.integers(0, 256, size=(3, 5, 4)).astype(np.uint8)
    out = CL.clean_block(blk, np.zeros(blk.shape), 200.0, CFG)
    assert out.block.dtype == np.float64
    assert_array_equal(out.block, blk.astype(np.float64))
    assert_array_equal(out.added_gl, np.zeros(3))

    # the exact inverse under the renderer's own rule (partition, then linear); 1e-9 gl at B = 210 (SPEC):
    # with disjoint projections every column holds one tube, so Delta A_j of the scene is the sum of each
    # tube's, and Eqs. 5-6 are linear in Delta A_j: B tau_scene + B (1 - tau_nb) = B tau_node to roundoff.
    for kind in ("flat", "tilted"):
        for absorption in ("partition_vertical", "linear"):
            resid, bound, s_peak, own_peak = inverse_residual(kind, absorption, (3.0, 2.0))
            MEASURED["exact inverse %s %s" % (absorption, kind)] = \
                "residual %.3e gl (tol 1e-9); neighbour's peak B S %.2f gl, node's peak dip %.2f gl" % (
                    resid, s_peak, own_peak)
            assert s_peak > 1.0 and own_peak > 1.0, "the shadows must be visible for the check to mean anything"
            assert resid <= 1e-9, "exact inverse %s %s: %.3e gl" % (absorption, kind, resid)

    # tiers on constructed scenes, one neighbour each (D-132; defaults: close 2.0 um / 0.84 um, reach 3.0 um)
    cfg = default_config()
    for name, (row, expected) in CASES.items():
        t = CL.neighbour_tiers(fscene(OWN_SEGS + [row]), OWN_IDS, NODE, cfg)
        assert t.tolist() == [expected], "%s: tiers %r, expected [%d]" % (name, t.tolist(), expected)
    # all together, with a soma disc (tier 1) and the own segments interleaved: the neighbours' order
    rows = [CASES["far_lateral"][0], OWN_SEGS[0], CASES["close_in_focus"][0], OWN_SEGS[1],
            CASES["attached_not_close"][0], CASES["directly_above_unattached"][0]]
    t = CL.neighbour_tiers(fscene(rows, soma=[(0.5, 0.5, 4.0)]), OWN_IDS, NODE, cfg)
    assert t.tolist() == [1, 2, 1, 2, 1], t.tolist()
    # the reach matters: the unattached twin's footprint (x in [2.0, 3.0]) meets the own footprint only at
    # 2.0 um or more from the node, so at a reach of 1.5 um it is tier 1
    cfg_r = dataclasses.replace(cfg, cleaning=dataclasses.replace(cfg.cleaning, tier_reach_um=1.5))
    t = CL.neighbour_tiers(fscene(OWN_SEGS + [CASES["unattached_twin"][0]]), OWN_IDS, NODE, cfg_r)
    assert t.tolist() == [1], t.tolist()


def _seg_dist(px, py, a, b):
    """Point-segment distance in projection by the vector form |p - a - clip(t) (b - a)|, written here
    independently of background.segment_distance (the code under test uses that one)."""
    A = np.array([a[0], a[1]], dtype=float)
    D = np.array([b[0] - a[0], b[1] - a[1]], dtype=float)
    P = np.stack([px - A[0], py - A[1]], axis=-1)
    L2 = float(D @ D)
    t = np.zeros(P.shape[:-1]) if L2 == 0.0 else np.clip(P @ D / L2, 0.0, 1.0)
    return np.linalg.norm(P - t[..., None] * D, axis=-1)


def _brute_tiers(fs, own_ids, node, cfg, spacing=0.01):
    """Independent statement of the rule: nearest point by dense sampling of the axis (20001 points), the
    footprints' meeting on a grid of spacing 0.01 um over the reach disc (about 11 times finer than the
    code's). Returns the tiers and, per neighbour segment, the distance of its decision from a threshold
    (um): the close test's nearer margin and, for the overlap, |min over the own footprint of (distance
    to the axis - radius)| -- small when the footprints only just meet or miss."""
    c = cfg.cleaning
    own = set(int(i) for i in own_ids)
    xyz, ids, d = fs.seg_xyz, fs.seg_ids, fs.seg_d_um
    s = np.linspace(0.0, 1.0, 20001)
    off = np.arange(-c.tier_reach_um, c.tier_reach_um + 0.5 * spacing, spacing)
    X, Y = np.meshgrid(node[0] + off, node[1] + off)
    inside = np.hypot(X - node[0], Y - node[1]) <= c.tier_reach_um
    px, py = X[inside], Y[inside]
    own_q = [q for q in range(ids.shape[0]) if ids[q, 0] in own and ids[q, 1] in own]
    own_fp = np.zeros(px.shape, dtype=bool)
    for q in own_q:
        own_fp |= _seg_dist(px, py, xyz[q, :2], xyz[q, 3:5]) <= 0.5 * d[q].max()
    tiers, margins = [], []
    for q in range(ids.shape[0]):
        if q in own_q:
            continue
        line = xyz[q, :3][None, :] + s[:, None] * (xyz[q, 3:] - xyz[q, :3])[None, :]
        dd = np.hypot(line[:, 0] - node[0], line[:, 1] - node[1])
        m = int(np.argmin(dd))
        dz = abs(line[m, 2] - node[2])
        close = dd[m] <= c.close_lateral_um and dz <= c.close_defocus_um
        margin = min(abs(dd[m] - c.close_lateral_um), abs(dz - c.close_defocus_um))
        attached = ids[q, 0] in own or ids[q, 1] in own
        over = False
        if not attached and own_fp.any():
            gap = _seg_dist(px[own_fp], py[own_fp], xyz[q, :2], xyz[q, 3:5]) - 0.5 * d[q].max()
            over = bool(np.any(gap <= 0.0))
            if not close:
                margin = min(margin, abs(float(gap.min())))
        tiers.append(2 if (close or over) else 1)
        margins.append(margin)
    return np.array(tiers + [1] * fs.soma.shape[0], dtype=np.int64), np.array(margins)


def test_reference():
    # added_gl equals the mean of B S; 1e-12 relative (float64 means of ~1e2 numbers of size ~1e2)
    rng = make_rng()
    blk = 200.0 + 10.0 * rng.standard_normal((4, 6, 7))
    S = 0.1 * rng.random((4, 6, 7))
    for B in (210.0, 205.0 + rng.random(4), 200.0 + 20.0 * rng.random((4, 6, 7))):
        out = CL.clean_block(blk, S, B, CFG)
        Bf = np.broadcast_to(np.asarray(B, dtype=float).reshape((-1, 1, 1) if np.ndim(B) == 1 else np.shape(B)),
                             blk.shape)
        check_close("added_gl", out.added_gl, np.array([np.mean(Bf[k] * S[k]) for k in range(4)]),
                    rtol=1e-12, atol=0.0)
        exp_over = np.array([np.mean((blk[k] + Bf[k] * S[k]) - Bf[k] > 3.0) for k in range(4)])
        assert_array_equal(out.over_background, exp_over)

    # _nearest_t's point is the nearest: its distance equals background.segment_distance [1e-12 um]
    for _ in range(50):
        a, b, p = rng.uniform(-3, 3, 2), rng.uniform(-3, 3, 2), rng.uniform(-3, 3, 2)
        t = CL._nearest_t(p[0], p[1], a, b)
        check_close("nearest point", math.hypot(p[0] - (a[0] + t * (b[0] - a[0])), p[1] - (a[1] + t * (b[1] - a[1]))),
                    float(background.segment_distance(p[0], p[1], a, b)), rtol=0, atol=1e-12)

    # tiers on random scenes against the brute force; decisions within 0.03 um of a threshold (where the code's
    # grid of 0.1144 um and the brute force's 0.01 um may legitimately disagree) are not counted
    cfg = default_config()
    agree = counted = n_two = 0
    for trial in range(12):
        segs = list(OWN_SEGS)
        for k in range(6):
            p0 = rng.uniform(-4, 4, 3) * np.array([1, 1, 0.6])
            p1 = p0 + rng.uniform(-3, 3, 3) * np.array([1, 1, 0.5])
            parent = int(rng.choice([1, 2, 3])) if k == 0 else 100 + 10 * trial + k
            if k == 0:
                p0 = np.array([[-3.0, 0, 0], [0.0, 0, 0], [2.5, 0, 0]])[parent - 1]
            segs.append((200 + 10 * trial + k, parent, tuple(p0), tuple(p1), 0.3 + rng.random(), 0.3 + rng.random()))
        fs = fscene(segs, soma=[(5.0, 5.0, 2.0)])
        got = CL.neighbour_tiers(fs, OWN_IDS, NODE, cfg)
        ref, margins = _brute_tiers(fs, OWN_IDS, NODE, cfg)
        keep = np.concatenate([margins > 0.03, [True]])
        counted += int(keep.sum())
        agree += int(np.sum(got[keep] == ref[keep]))
        n_two += int(np.sum(ref[keep] == 2))
        assert np.array_equal(got[keep], ref[keep]), "trial %d: got %r, brute force %r, margins %r" % (
            trial, got.tolist(), ref.tolist(), np.round(margins, 4).tolist())
    MEASURED["tiers vs brute force"] = "%d of %d neighbours agree (12 random scenes; %d of them tier 2)" % (
        agree, counted, n_two)


def test_convergence():
    raise unittest.SkipTest("the cleaning operator is pointwise algebra, the tiers a geometric predicate: "
                            "no discretisation whose order could be measured")


def test_invariants():
    # hybrid (Block 28, amendment 1, D-135): the residual of the same inverse is a reported measurement, no
    # bound (the bound B max (1 - tau_1)(1 - tau_2) was withdrawn: oblique rays couple two tubes away from where
    # their blurred shadows overlap). (i) The three cases of the draft's test:
    for kind, mus in (("flat", (3.0, 2.0)), ("tilted", (3.0, 2.0)), ("flat", (6.0, 6.0))):
        resid, bound, s_peak, own_peak = inverse_residual(kind, "hybrid", mus)
        HYBRID_TABLE.append(("draft " + kind, {"flat": 0.0, "tilted": 30.0}[kind], 0.25 if kind == "flat" else 0.3,
                             mus, 9 * PX if kind == "flat" else 8 * PX, resid, float("nan"), bound, s_peak, own_peak))
        assert np.isfinite(resid) and resid > 0.0, "the hybrid couples the two tubes: the residual should not vanish"
    # (ii) two equal tubes, r 0.25 um, mu 3 /um, tilt 0 and 45 deg, centres 1.0, 2.0, 3.0 um apart (along y,
    # heading 0). The patch spans y from -1.83 to 4.80 um (58 px) so that a 3 um separation holds both tubes and
    # their blurred shadows. The separations are not whole pixels, so the node and the neighbour sample the
    # grid differently and their slab lattices differ slightly (z_lo.min() is read on different samples): the
    # partition's residual on the same scenes, which would be exactly 0 on one lattice, is printed beside the
    # hybrid's as the floor of the measurement (no tolerance: a reported number, like the hybrid's).
    frame = (-14, -16, 28, 58)
    for tilt in (0.0, 45.0):
        for sep in (1.0, 2.0, 3.0):
            node = G.Tube((0.0, 0.0, 0.0), 0.25, math.radians(tilt), 0.0, 1.0, 1.0, "axial")
            nb = G.Tube((0.0, sep, 0.0), 0.25, math.radians(tilt), 0.0, 1.0, 1.0, "axial")
            resid, bound, s_peak, own_peak = inverse_residual(None, "hybrid", (3.0, 3.0), (node, nb), frame, False)
            floor = inverse_residual(None, "partition_vertical", (3.0, 3.0), (node, nb), frame, False)[0]
            HYBRID_TABLE.append(("pair", tilt, 0.25, (3.0, 3.0), sep, resid, floor, bound, s_peak, own_peak))
            assert np.isfinite(resid), "hybrid residual not finite"
            assert np.isfinite(floor), "partition floor not finite"

    # linearity in S and the added light: clean(S1 + S2) - clean(0) = B (S1 + S2); 1e-12 relative to B
    rng = make_rng(SEED + 1)
    blk = rng.integers(100, 255, size=(2, 9, 8)).astype(np.uint8)
    S1, S2 = 0.2 * rng.random(blk.shape), 0.1 * rng.random(blk.shape)
    B = 190.0 + 20.0 * rng.random(2)
    c12 = CL.clean_block(blk, S1 + S2, B, CFG).block
    c1 = CL.clean_block(blk, S1, B, CFG).block
    c2 = CL.clean_block(blk, S2, B, CFG).block
    c0 = CL.clean_block(blk, np.zeros(blk.shape), B, CFG).block
    check_close("linear in S", c12 - c0, (c1 - c0) + (c2 - c0), rtol=0, atol=1e-12 * 255)
    check_close("added is B S", c12 - blk, B[:, None, None] * (S1 + S2), rtol=0, atol=1e-12 * 255)


def test_contract():
    rng = make_rng(SEED + 2)
    n, H, W = 3, 5, 6
    S = 0.1 * rng.random((n, H, W))
    for dtype in (np.uint8, np.int16, np.float32, np.float64):
        blk = (200 * rng.random((n, H, W))).astype(dtype)
        out = CL.clean_block(blk, S, 210.0, CFG)
        assert isinstance(out, CL.CleanResult)
        assert out.block.shape == (n, H, W) and out.block.dtype == np.float64
        assert out.added_gl.shape == (n,) and out.added_gl.dtype == np.float64
        assert out.over_background.shape == (n,)
        assert np.all((out.over_background >= 0) & (out.over_background <= 1))
        assert np.all(np.isfinite(out.block))
    # the three forms of B_gl agree, exactly
    blk = 200 * rng.random((n, H, W))
    a = CL.clean_block(blk, S, 207.5, CFG)
    b = CL.clean_block(blk, S, np.full(n, 207.5), CFG)
    c = CL.clean_block(blk, S, np.full((n, H, W), 207.5), CFG)
    for r in (b, c):
        assert_array_equal(r.block, a.block)
        assert_array_equal(r.added_gl, a.added_gl)
        assert_array_equal(r.over_background, a.over_background)
    # per-plane B: (n,) equals the (n, H, W) array of the same values
    Bn = np.array([190.0, 205.0, 220.0])
    assert_array_equal(CL.clean_block(blk, S, Bn, CFG).block,
                       CL.clean_block(blk, S, np.broadcast_to(Bn[:, None, None], (n, H, W)).copy(), CFG).block)
    # over_background counts pixels strictly above B + over_background_gl
    blk2 = np.full((1, 2, 2), 200.0)
    S2 = np.array([[[0.0, 3.0 / 200.0], [4.0 / 200.0, 0.01]]])
    out = CL.clean_block(blk2, S2, 200.0, CFG)                # cleaned - B = 0, 3, 4, 2 gl; the threshold 3 gl
    assert_array_equal(out.over_background, [0.25])
    # the threshold is read from cfg (Block 28, amendment 1): at 1.5 gl three of the four pixels count
    cfg_lo = dataclasses.replace(CFG, cleaning=dataclasses.replace(CFG.cleaning, over_background_gl=1.5))
    assert_array_equal(CL.clean_block(blk2, S2, 200.0, cfg_lo).over_background, [0.75])
    # cfg is required: no default configuration inside the function
    try:
        CL.clean_block(blk2, S2, 200.0)
    except TypeError:
        pass
    else:
        raise AssertionError("clean_block without cfg should raise TypeError")
    # tiers: (n_q + n_s,) int64 in {1, 2}
    fs = fscene(OWN_SEGS + [CASES["close_in_focus"][0], CASES["far_lateral"][0]], soma=[(9.0, 9.0, 3.0)] * 2)
    t = CL.neighbour_tiers(fs, OWN_IDS, NODE, default_config())
    assert t.shape == (4,) and t.dtype == np.int64 and set(t.tolist()) <= {1, 2}
    assert t.tolist() == [2, 1, 1, 1]


def test_determinism():
    rng = make_rng(SEED + 3)
    blk = rng.integers(0, 256, size=(2, 7, 7)).astype(np.uint8)
    S = 0.05 * rng.random(blk.shape)
    B = np.array([200.0, 210.0])
    blk0, S0, B0 = blk.copy(), S.copy(), B.copy()
    a, b = CL.clean_block(blk, S, B, CFG), CL.clean_block(blk, S, B, CFG)
    assert_array_equal(a.block, b.block)
    assert_array_equal(a.added_gl, b.added_gl)
    assert_array_equal(a.over_background, b.over_background)
    assert_array_equal(blk, blk0)
    assert_array_equal(S, S0)
    assert_array_equal(B, B0)
    assert a.block is not blk
    rows = OWN_SEGS + [r for r, _ in CASES.values()]
    fs = fscene(rows, soma=[(1.0, 1.0, 2.0)])
    before = [np.array(getattr(fs, f), copy=True) for f in ("seg_xyz", "seg_d_um", "seg_ids", "soma")]
    t1 = CL.neighbour_tiers(fs, OWN_IDS, NODE, default_config())
    t2 = CL.neighbour_tiers(fs, OWN_IDS, NODE, default_config())
    assert_array_equal(t1, t2)
    for arr, f in zip(before, ("seg_xyz", "seg_d_um", "seg_ids", "soma")):
        assert_array_equal(getattr(fs, f), arr)


def _raises(func, *args):
    try:
        func(*args)
    except ValueError:
        return
    raise AssertionError("no ValueError for %r" % (args[1:] if len(args) > 1 else args,))


def test_edge_cases():
    blk = np.full((2, 3, 3), 200.0)
    S = np.zeros((2, 3, 3))
    _raises(CL.clean_block, blk, np.zeros((2, 3, 4)), 200.0, CFG)            # unequal shapes
    _raises(CL.clean_block, blk[0], S[0], 200.0, CFG)                        # not (n_planes, H, W)
    bad = S.copy(); bad[0, 1, 1] = np.nan
    _raises(CL.clean_block, blk, bad, 200.0, CFG)                            # non-finite S
    bad = S.copy(); bad[1, 0, 0] = np.inf
    _raises(CL.clean_block, blk, bad, 200.0, CFG)
    bad = S.copy(); bad[1, 2, 2] = -1e-9
    _raises(CL.clean_block, blk, bad, 200.0, CFG)                            # negative S
    for B in (0.0, -5.0, np.nan, np.inf, np.array([200.0, 0.0]), np.full((2, 3, 3), -1.0)):
        _raises(CL.clean_block, blk, S, B, CFG)                              # B <= 0, non-finite B
    _raises(CL.clean_block, blk, S, np.full(3, 200.0), CFG)                  # B of a wrong length
    _raises(CL.clean_block, blk, S, np.full((2, 1, 1), 200.0), CFG)          # not one of the three forms
    _raises(CL.clean_block, blk.astype(complex), S, 200.0, CFG)              # not a real dtype
    cfg = default_config()
    fs = fscene(OWN_SEGS + [CASES["far_lateral"][0]])
    _raises(CL.neighbour_tiers, fs, OWN_IDS, (0.0, 0.0), cfg)           # node not 3 numbers
    _raises(CL.neighbour_tiers, fs, OWN_IDS, (0.0, np.nan, 0.0), cfg)
    _raises(CL.neighbour_tiers, fs, (77, 78), NODE, cfg)                # no own segment in the scene
    # no neighbour segment: only the soma discs, tier 1
    t = CL.neighbour_tiers(fscene(OWN_SEGS, soma=[(0.0, 0.0, 5.0)]), OWN_IDS, NODE, cfg)
    assert t.tolist() == [1]
    t = CL.neighbour_tiers(fscene(OWN_SEGS), OWN_IDS, NODE, cfg)
    assert t.shape == (0,) and t.dtype == np.int64
    # a degenerate (zero-length) neighbour at the node in focus is close: tier 2
    t = CL.neighbour_tiers(fscene(OWN_SEGS + [(71, 70, (0.5, 0.5, 0.1), (0.5, 0.5, 0.1), 0.4, 0.4)]),
                           OWN_IDS, NODE, cfg)
    assert t.tolist() == [2]


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

    if HYBRID_TABLE:
        print("\n-- hybrid: residual of cleaning a joint scene with the neighbour rendered alone (B = 210 gl, "
              "planes -3..3, 8 x 16 directions; no bound, Block 28 amendment 1)")
        print("   %-13s %6s %6s %-11s %8s %12s %14s %12s %10s %10s" % (
            "case", "tilt", "r_um", "mu", "sep_um", "resid_gl", "floor_part_gl", "old_bnd_gl", "peakBS_gl",
            "nodedip_gl"))
        for row in HYBRID_TABLE:
            print("   %-13s %6.1f %6.2f %-11s %8.3f %12.4e %14.2e %12.4f %10.2f %10.2f" % (
                row[0], row[1], row[2], "%g,%g" % row[3], row[4], row[5], row[6], row[7], row[8], row[9]))
    if MEASURED:
        print("\n-- measured")
        for key, val in MEASURED.items():
            print("   %s: %s" % (key, val))

    counts = {s: sum(1 for r in results if r[1] == s)
              for s in ("PASS", "FAIL", "ERROR", "TODO", "SKIP")}
    print("\n-- " + ", ".join(f"{n} {s.lower()}" for s, n in counts.items()))
    for path in _SAVED_ARTIFACTS:
        print(f"-- arrays from a failed comparison: {path}")
    failed = counts["FAIL"] + counts["ERROR"] + counts["TODO"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
