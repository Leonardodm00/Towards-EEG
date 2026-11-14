"""Spec-derived tests of Block 36's node inputs (analysis/node_inputs.py) and its boundaries with Blocks 25,
30, 32 and 35 -- integration testers, 2026-11-14 (first written against amendment 1; rewritten at 18:30
against amendments 2 to 4, D-142 to D-144). Oracles from specs/SPEC.md, Block 36 and its amendments: the
geometry by hand, the profile's noise factor from an independent bilinear-weight sum and a Monte Carlo,
the model background on planes built by hand, S_all from Block 25's additivity, the factories against
line_fast called by hand (the spec's own oracle), the work list on hand-built cells, and build_node on a
noise-free scene rendered by Block 24.

Rewritten tests (amendments 2 to 4 changed their contracts): test_local_tube_by_hand (U_minus, U_plus),
test_first_stage_start_is_the_pilot_tube (the extents), test_node_planes_and_extent (was
test_node_planes_and_half_length: tube_half_length is replaced by tube_extent), test_factories_against_line_fast
and test_factories_nan_on_extrusion_invalid ((U_minus, U_plus) pairs), test_work_list_examples_cap_threshold
(the cap now cuts the examples, F5).

Run: python3 -m pytest -q tests/spec/test_spec_block36_node_inputs.py   (from the code folder)
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
from allen_image_io import CropFrame
from allen_diameter.config import default_config
from allen_diameter.analysis import node_inputs as NI
from allen_diameter.analysis.clean_iteration import NodeData
from allen_diameter.analysis.node_features import FeaturedScene
from allen_diameter.analysis.node_pipeline import Branch
from allen_diameter.model import line_fast, shadow

P = 0.1144


def pilot(**kw):
    base = dict(cx_um=10.3, cy_um=-4.2, theta_rad=0.6, phi_rad=0.35, z_sub_um=1.13, k_star=4,
                d_hat_um=0.82, mu_hat_per_um=0.9, B_bar=210.0)
    base.update(kw)
    return types.SimpleNamespace(**base)


def branch(xyz, ids=None):
    xyz = np.asarray(xyz, dtype=float)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))])
    n = len(xyz)
    return Branch(np.arange(n) + 100 if ids is None else np.asarray(ids), np.full(n, 3), xyz, np.full(n, 0.3), s)


# ------------------------------------------------------------- geometry ---

def ref_own(s, i, U):
    """Amendment 3 (1) by hand: on each side the nodes up to and including the first whose arc length from
    the node exceeds U, or to the stretch's end."""
    lo = i
    for j in range(i - 1, -1, -1):
        lo = j
        if s[i] - s[j] > U:
            break
    hi = i
    for j in range(i + 1, len(s)):
        hi = j
        if s[j] - s[i] > U:
            break
    return lo, hi


def test_own_ids_within_by_hand():
    """own_ids_within(branch, index, U) -> (ids, U_minus, U_plus): irregular spacing, a node exactly at U
    (not 'beyond': the next one is the first beyond), both stretch ends [exact]."""
    xs = [0.0, 0.7, 2.0, 5.0, 6.0, 9.5, 11.0, 13.0, 14.0, 20.0]
    br = branch([[x, 0.0, 0.0] for x in xs])
    s = np.asarray(br.s_um)
    for U in (0.5, 3.0, 5.0, 7.0, 30.0):
        for i in range(len(xs)):
            ids, um, up = NI.own_ids_within(br, i, U)
            lo, hi = ref_own(s, i, U)
            assert ids == tuple(int(x) for x in br.ids[lo:hi + 1]), (U, i)
            assert um == pytest.approx(s[i] - s[lo], abs=0) and up == pytest.approx(s[hi] - s[i], abs=0)
    # node 3 (x 5) with U 4: x 9.5 is 4.5 > 4 (first beyond, included); x 0.7 is 4.3 > 4 (included)
    ids, um, up = NI.own_ids_within(br, 3, 4.0)
    assert ids == (101, 102, 103, 104, 105) and um == pytest.approx(4.3) and up == pytest.approx(4.5)
    # a node exactly at U is not beyond it: node 0 with U 2.0 -> 0.7, 2.0 (= U), 5.0 (first beyond)
    ids, um, up = NI.own_ids_within(br, 0, 2.0)
    assert ids == (100, 101, 102, 103) and um == 0.0 and up == 5.0
    with pytest.raises(ValueError):
        NI.own_ids_within(br, len(xs), 5.0)


def test_node_planes_and_extent():
    """node_planes = k* - 4 .. k* + 4; tube_extent = own_ids_within's (U_minus, U_plus) at
    tube_half_length_um (5.0); near_end when min(U_minus, U_plus) < extrusion_cap_margin_um (3.0), 'at least'
    the margin is valid (amendment 3 (1))."""
    cfg = default_config()
    assert cfg.cleaning.plane_half_range == 4 and cfg.inference.tube_half_length_um == 5.0
    assert cfg.inference.extrusion_cap_margin_um == 3.0
    assert_array_equal(NI.node_planes(pilot(k_star=7), cfg), np.arange(3, 12))
    xs = [0.0, 1.0, 3.0, 7.0, 13.0, 14.0, 21.0]
    br = branch([[x, 0.0, 0.0] for x in xs])
    s = np.asarray(br.s_um)
    for i in range(len(xs)):
        lo, hi = ref_own(s, i, 5.0)
        um, up = NI.tube_extent(br, i, cfg)
        assert (um, up) == (s[i] - s[lo], s[hi] - s[i])
        assert NI.is_near_end(br, i, cfg) == (min(um, up) < 3.0)
    br2 = branch([[0.0, 0, 0], [3.0, 0, 0], [6.0, 0, 0]])
    assert NI.tube_extent(br2, 1, cfg) == (3.0, 3.0) and not NI.is_near_end(br2, 1, cfg)


def test_local_tube_by_hand():
    """Amendments 1 (2) and 3 (1): the node's point (c_x, c_y) + (v0 - v_t) y_hat at depth z_t + dz, the tube
    from -U_minus to +U_plus along a_hat: centre moved by (U_plus - U_minus)/2 a_hat, half-length
    (U_plus + U_minus)/2; radius d/2, heading, tilt, aspect 1, axial caps [1e-12]."""
    p = pilot()
    trace = np.array([10.1, -4.0, 0.6])
    th, ph = p.theta_rad, p.phi_rad
    yh = np.array([-math.sin(th), math.cos(th)])
    a = np.array([math.cos(ph) * math.cos(th), math.cos(ph) * math.sin(th), math.sin(ph)])
    v_t = float((np.array([p.cx_um, p.cy_um]) - trace[:2]) @ yh)
    d, v0, dz, Um, Up = 0.7, 0.25, -0.3, 3.5, 6.25
    t = NI.local_tube(p, trace, d, v0, dz, Um, Up)
    node_pt = np.r_[np.array([p.cx_um, p.cy_um]) + (v0 - v_t) * yh, trace[2] + dz]
    exp_c = node_pt + 0.5 * (Up - Um) * a
    assert_allclose(np.asarray(t.c), exp_c, rtol=0, atol=1e-12)
    assert t.half_length == pytest.approx(0.5 * (Up + Um), abs=1e-15)
    # the two caps are the node's point -U_minus and +U_plus along a_hat
    c = np.asarray(t.c)
    assert_allclose(c - t.half_length * a, node_pt - Um * a, rtol=0, atol=1e-12)
    assert_allclose(c + t.half_length * a, node_pt + Up * a, rtol=0, atol=1e-12)
    assert t.r == pytest.approx(d / 2, abs=1e-15)
    assert t.theta == th and t.phi == ph and t.aspect == 1.0 and t.end_cut == "axial"


def test_first_stage_start_is_the_pilot_tube():
    """Amendment 1 (2): theta0 = (d_hat, mu_hat, v_t, z_sub - z_t); at theta0 with symmetric extents the
    tube's centre is the pilot's (c_x, c_y, z_sub) [1e-12]; the trace's offset along the branch does not move
    it."""
    p = pilot()
    e_u = np.array([math.cos(p.theta_rad), math.sin(p.theta_rad)])
    yh = np.array([-math.sin(p.theta_rad), math.cos(p.theta_rad)])
    trace = np.array([p.cx_um, p.cy_um, 0.4]) + np.r_[0.31 * yh + 0.8 * e_u, 0.0]
    th0 = NI.pilot_start(p, trace)
    assert_allclose(th0, [p.d_hat_um, p.mu_hat_per_um, -0.31, p.z_sub_um - 0.4], rtol=0, atol=1e-12)
    t = NI.local_tube(p, trace, th0[0], th0[2], th0[3], 5.0, 5.0)
    assert_allclose(np.asarray(t.c), [p.cx_um, p.cy_um, p.z_sub_um], rtol=0, atol=1e-12)


@pytest.mark.parametrize("rising", [True, False])
@pytest.mark.parametrize("reverse", [False, True])
def test_axis_extent_tube_covers_own_segments(rising, reverse):
    """Amendment 3 (1) with amendment 4 (3): on a straight tilted stretch, the pilot on the node, the model
    tube from local_tube(..., *axis_extent(...)) runs between the outermost own nodes when the trace rises
    along +e_u, and between their mirror images across the measuring line (u -> -u) when it falls; the
    stretch listed in either direction [1e-9]."""
    cfg = default_config()
    phi = math.radians(30.0)
    sgn = 1.0 if rising else -1.0
    u = np.arange(0.0, 16.0, 1.5)                       # 11 nodes along +x, 1.5 um apart (arc 1.5/cos phi)
    xyz = np.column_stack([u * math.cos(phi), np.zeros_like(u), sgn * u * math.sin(phi)])
    if reverse:
        xyz = xyz[::-1].copy()
    br = branch(xyz)
    i = 4 if not reverse else 6                          # asymmetric extents on the two sides
    ids, Um, Up = NI.own_ids_within(br, i, cfg.inference.tube_half_length_um)
    assert (Um, Up) == NI.tube_extent(br, i, cfg)
    P0 = xyz[i]
    pl = types.SimpleNamespace(cx_um=P0[0], cy_um=P0[1], theta_rad=0.0, phi_rad=phi, z_sub_um=P0[2])
    Ub, Uf = NI.axis_extent(pl, br, i, Um, Up)
    t = NI.local_tube(pl, P0, 1.0, 0.0, 0.0, Ub, Uf)
    a = np.array([math.cos(phi), 0.0, math.sin(phi)])
    ends = sorted([np.asarray(t.c) - t.half_length * a, np.asarray(t.c) + t.half_length * a], key=lambda q: q[0])
    pos = {int(n): xyz[k] for k, n in enumerate(br.ids)}
    own_pts = sorted([pos[ids[0]], pos[ids[-1]]], key=lambda q: q[0])
    if not rising:                                       # mirror across the measuring line: u -> -u about the node
        own_pts = sorted([np.array([2 * P0[0] - q[0], q[1], q[2]]) for q in own_pts], key=lambda q: q[0])
    assert_allclose(ends[0], own_pts[0], rtol=0, atol=1e-9)
    assert_allclose(ends[1], own_pts[1], rtol=0, atol=1e-9)


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


# --------------------------------------------------- profile noise factor ---

def ref_noise_factor(left, top, p, origin, yh, eu, v, n_avg, step):
    """Amendment 2 (F2) by hand: the bilinear weights of every sample (pixel centres at integer pixel
    coordinates, col = x / p - left), the mean over 2 n_avg + 1 lines; sqrt(mean_n sum_px w^2)."""
    tot = 0.0
    for vn in v:
        w = {}
        for a in range(-n_avg, n_avg + 1):
            x = origin[0] + vn * yh[0] + a * step * eu[0]
            y = origin[1] + vn * yh[1] + a * step * eu[1]
            cx, cy = x / p - left, y / p - top
            i0, j0 = int(math.floor(cx)), int(math.floor(cy))
            fx, fy = cx - i0, cy - j0
            for di, wx in ((0, 1 - fx), (1, fx)):
                for dj, wy in ((0, 1 - fy), (1, fy)):
                    key = (i0 + di, j0 + dj)
                    w[key] = w.get(key, 0.0) + wx * wy / (2 * n_avg + 1)
        tot += sum(q * q for q in w.values())
    return math.sqrt(tot / len(v))


@pytest.mark.parametrize("theta,n_avg", [(0.3, None), (0.0, None), (1.1, 0), (0.0, 0)])
def test_profile_noise_factor_by_hand(theta, n_avg):
    """profile_noise_factor = sqrt((1/N_v) sum_n sum_px w_{n,px}^2) against the bilinear weights summed by
    hand [1e-12, round-off of sums of products]; n_avg None = Block 5's (profiles.n_along)."""
    from allen_diameter.analysis import profiles
    cfg = default_config()
    m = cfg.measure
    v, step = profiles.profile_offsets(m), float(m.profile_step_um)
    A = profiles.n_along(m) if n_avg is None else n_avg
    yh, eu = np.array([-math.sin(theta), math.cos(theta)]), np.array([math.cos(theta), math.sin(theta)])
    left, top = 13, -7
    frame = CropFrame(left, top, 0, P)
    o = np.array([(left + 60) * P + 0.031, (top + 60) * P + 0.047])
    got = NI.profile_noise_factor(frame, (120, 120), o, yh, eu, v, A, step)
    exp = ref_noise_factor(left, top, P, o, yh, eu, v, A, step)
    assert got == pytest.approx(exp, rel=1e-12, abs=0)
    if A == 0 and theta == 0.0:
        # one line along y through fractional column offset fx: sum w^2 = ((1-fx)^2 + fx^2) x mean over the
        # samples of ((1-fy)^2 + fy^2); on pixel centres exactly (fx = fy = 0) the factor is 1
        o2 = np.array([(left + 60) * P, (top + 60) * P])
        assert NI.profile_noise_factor(frame, (120, 120), o2, yh, eu, v, 0, step) == pytest.approx(1.0, abs=1e-12)
    # a line that reads outside the patch
    with pytest.raises(ValueError):
        NI.profile_noise_factor(frame, (40, 40), o, yh, eu, v, A, step)


def test_profile_noise_factor_monte_carlo():
    """Amendment 2 oracle: the factor against the SD of the profiles of 400 seeded white-noise images read
    with nc_helpers' independent bilinear read (scipy.ndimage) [3 %: 400 x 61 samples, sampling error of the
    SD about 0.5 %, correlations along the profile raise it to about 1 %]."""
    from allen_diameter.analysis import profiles
    cfg = default_config()
    m = cfg.measure
    v, A, step = profiles.profile_offsets(m), profiles.n_along(m), float(m.profile_step_um)
    th = 0.7
    yh, eu = np.array([-math.sin(th), math.cos(th)]), np.array([math.cos(th), math.sin(th)])
    frame = CropFrame(0, 0, 0, P)
    o = np.array([50 * P + 0.021, 50 * P + 0.083])
    fac = NI.profile_noise_factor(frame, (100, 100), o, yh, eu, v, A, step)
    rng = np.random.default_rng(36)
    prof = np.array([nc_helpers.bilinear_read(rng.normal(0.0, 3.0, (100, 100)), 0, 0, P, o, yh, eu, v, A, step)
                     for _ in range(400)])
    emp = float(np.sqrt(np.mean(prof ** 2)))
    print("noise factor %.4f; MC profile SD / pixel SD %.4f" % (fac, emp / 3.0))
    assert abs(emp / (3.0 * fac) - 1.0) <= 0.03


# ------------------------------------------------------- model background ---

def plane_bg(b0, bx, by, x, y):
    return b0 + bx * (x - 3.0) + by * (y - 2.0)


def synthetic_planes(rng, H=60, W=80, noise=0.0):
    """Three planes I = b (1 - S) (+ noise): b planar per plane, S in [0, 0.09] on the support and 0.6 on a
    band of 10 rows (left out by background_shadow_max 0.10), the band's pixels set to garbage."""
    frame = CropFrame(0, 0, 0, P)
    xs, ys = np.meshgrid(np.arange(W) * P, np.arange(H) * P)
    coefs = [(210.0, 0.8, -0.5), (205.0, -1.1, 0.3), (190.0, 0.0, 1.7)]
    S = np.empty((3, H, W))
    I = np.empty((3, H, W))
    for k, (b0, bx, by) in enumerate(coefs):
        S[k] = 0.09 * rng.uniform(size=(H, W))
        S[k, 20:30, :] = 0.6
        I[k] = plane_bg(b0, bx, by, xs, ys) * (1.0 - S[k]) + noise * rng.normal(size=(H, W))
        I[k, 20:30, :] = 17.0                              # garbage under the band: must not matter
    return frame, coefs, S, I


def test_model_background_noise_free():
    """Amendment 4 (1): b fitted through (1 - S_all) on the pixels with S_all <= 0.10; B = b at the line's
    origin [1e-9 gl: linear least squares on exact data]; frac <= the support's share; ok; inputs unchanged."""
    cfg = default_config()
    assert cfg.cleaning.background_shadow_max == 0.10
    rng = np.random.default_rng(1)
    frame, coefs, S, I = synthetic_planes(rng)
    I0, S0 = I.copy(), S.copy()
    o = np.array([4.1, 3.3])
    B, sd, frac, ok = NI.model_background(I, S, frame, o, cfg)
    assert_array_equal(I, I0)
    assert_array_equal(S, S0)
    exp = [plane_bg(b0, bx, by, o[0], o[1]) for (b0, bx, by) in coefs]
    assert_allclose(B, exp, rtol=0, atol=1e-9)
    share = np.mean(S <= 0.10, axis=(1, 2))
    assert np.all(frac <= share + 1e-15) and np.all(frac > 0) and np.all(ok)
    assert np.all(sd < 1e-9)
    assert B.shape == sd.shape == frac.shape == ok.shape == (3,)


def test_model_background_noise_trim_and_validity():
    """Amendment 4 (1): white noise SD 3 gl -> the plane's noise SD within 10 % of 3 (spec oracle); 2 % of
    the support at +60 gl is trimmed (B within 0.4 gl of b(origin): without the trim the bias would be about
    1.2 gl; 0.4 gl is about 5 SEs of the plane fit on ~4000 pixels); a support under background_min_far_frac
    -> ok False; a non-positive background -> ok False; a NaN pixel is ignored; ValueError for unequal
    shapes."""
    cfg = default_config()
    rng = np.random.default_rng(2)
    frame, coefs, S, I = synthetic_planes(rng, noise=3.0)
    o = np.array([4.1, 3.3])
    exp = np.array([plane_bg(b0, bx, by, o[0], o[1]) for (b0, bx, by) in coefs])
    B, sd, frac, ok = NI.model_background(I, S, frame, o, cfg)
    print("white noise SD 3: sd %s, B - b %s" % (np.round(sd, 3), np.round(B - exp, 3)))
    assert np.all(np.abs(sd - 3.0) <= 0.3) and np.all(ok)
    assert np.all(np.abs(B - exp) <= 0.4)
    J = I.copy()
    sup = np.argwhere(S[0] <= 0.10)
    pick = sup[rng.choice(len(sup), size=int(0.02 * len(sup)), replace=False)]
    J[0, pick[:, 0], pick[:, 1]] += 60.0
    J[1, 5, 5] = np.nan
    B2, sd2, frac2, ok2 = NI.model_background(J, S, frame, o, cfg)
    print("with 2 %% outliers: B - b %.3f, sd %.3f" % (B2[0] - exp[0], sd2[0]))
    assert abs(B2[0] - exp[0]) <= 0.4 and ok2[0]
    assert np.isfinite(B2[1]) and ok2[1]
    # validity: support below background_min_far_frac (0.05)
    S3 = S.copy()
    S3[2] = 0.5
    S3[2, :2, :] = 0.0                                     # 2 rows of 60: 3.3 % of the patch
    _B, _sd, frac3, ok3 = NI.model_background(I, S3, frame, o, cfg)
    assert frac3[2] < cfg.calibration.background_min_far_frac and not ok3[2] and ok3[0]
    _B, _sd, _f, ok4 = NI.model_background(-np.abs(I), S, frame, o, cfg)
    assert not np.any(ok4)
    with pytest.raises(ValueError):
        NI.model_background(I[:, :10], S, frame, o, cfg)


# -------------------------------------------------------------- S_all ---

def small_fscene(with_soma=False):
    """Two flat segments in focus, far apart laterally: A (ids 2-1) along x at y 1.5, B (ids 4-3) at y 5.0;
    optional soma disc."""
    xyz = np.array([[0.5, 1.5, 0.0, 3.5, 1.5, 0.0], [0.5, 5.0, 0.2, 3.5, 5.0, 0.2]])
    soma = np.array([[2.0, 3.2, 0.6]]) if with_soma else np.zeros((0, 3))
    return FeaturedScene(seg_xyz=xyz, seg_d_um=np.array([[0.5, 0.5], [0.4, 0.6]]),
                         seg_mu_per_um=np.array([[1.0, 1.0], [0.8, 1.2]]), seg_ids=np.array([[2, 1], [4, 3]]),
                         seg_type=np.array([3, 3]), soma=soma)


def fast_shadow_cfg(soma_mu=0.3):
    cfg = default_config()
    return dataclasses.replace(cfg, cleaning=dataclasses.replace(
        cfg.cleaning, shadow_absorption="partition_vertical", shadow_fine_factor=2, soma_mu_per_um=soma_mu))


@pytest.mark.parametrize("with_soma", [False, True])
def test_all_shadow_is_every_segment(with_soma):
    """Amendment 4 (1): S_all = Block 25's shadow of every traced segment and soma disc, own stretch included,
    no floor: by Block 25's additivity, S_all = S(own A) + S(own B) - S(own A and B) (the last is the soma
    alone, zero without one) [1e-12, additivity of float sums]."""
    cfg = fast_shadow_cfg()
    fs = small_fscene(with_soma)
    left, top, W, H = 0, 0, 36, 60
    pz = np.array([-0.28, 0.0, 0.28])
    like = types.SimpleNamespace(left=left, top=top, block=np.zeros((3, H, W)), planes_z=pz)
    got = NI.all_shadow(fs, like, cfg)
    sA = shadow.shadow_image(fs, (1, 2), left, top, W, H, pz, cfg).S       # B (+ soma)
    sB = shadow.shadow_image(fs, (3, 4), left, top, W, H, pz, cfg).S       # A (+ soma)
    s_soma = shadow.shadow_image(fs, (1, 2, 3, 4), left, top, W, H, pz, cfg).S
    assert got.shape == (3, H, W) and got.dtype == np.float64
    assert_allclose(got, sA + sB - s_soma, rtol=0, atol=1e-12)
    assert got.max() > 0.05
    if with_soma:
        assert s_soma.max() > 0


# ----------------------------------------------------------- factories ---

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
    """predict = 1 - line_fast.line_profiles(local_tube(pilot, trace, d, v0, dz, U_minus, U_plus), mu, ...) and
    forward(d, mu, v0, z_t + dz) = line_fast.line_profiles(same) [exact] (D-138.4; amendments 1, 3)."""
    cfg = fast_cfg(absorption)
    p = pilot(cx_um=2.0, cy_um=2.0, z_sub_um=0.84, k_star=3)
    trace = np.array([2.05, 1.95, 0.7])
    node = make_node(p, trace, np.arange(1, 6))
    Um, Up = 4.0, 5.5
    pred = NI.predict_factory({5: p}, {5: (Um, Up)}, cfg, line_fast.SectionCache())(node)
    fwd = NI.forward_factory(p, (Um, Up), node, cfg, line_fast.SectionCache())
    d, mu, v0, dz = 0.6, 1.1, 0.1, 0.2
    ref = line_fast.line_profiles(NI.local_tube(p, trace, d, v0, dz, Um, Up), mu, node.ks, node.origin_xy,
                                  node.y_hat, node.e_u, node.v, node.n_avg, node.step, cfg)
    assert_array_equal(pred(d, mu, v0, dz), 1.0 - ref)
    assert_array_equal(fwd(d, mu, v0, trace[2] + dz), ref)
    assert pred(d, mu, v0, dz).shape == (5, node.v.size)


def test_factories_nan_on_extrusion_invalid():
    """An ExtrusionInvalid gives NaN of the right shape (Block 36; Blocks 29/30 then fail the fit): extents
    of 2.5 um (< margin 3.0) on one side."""
    cfg = fast_cfg()
    p = pilot(cx_um=2.0, cy_um=2.0, z_sub_um=0.84, k_star=3)
    trace = np.array([2.0, 2.0, 0.84])
    node = make_node(p, trace, np.arange(1, 4))
    pred = NI.predict_factory({5: p}, {5: (2.5, 6.0)}, cfg, None)(node)
    fwd = NI.forward_factory(p, (6.0, 2.5), node, cfg, None)
    a, b = pred(0.6, 1.0, 0.0, 0.0), fwd(0.6, 1.0, 0.0, 0.84)
    assert a.shape == b.shape == (3, node.v.size)
    assert np.all(np.isnan(a)) and np.all(np.isnan(b))


# ------------------------------------------------------------ work list ---

def chain_swc(tmp_path, chains, name="c.swc"):
    """A soma (id 1, far away) and straight dendrites along x, given as (y, n nodes): nodes 1 um apart from
    x = 0, all at z 0, radius 0.3; ids consecutive from 2."""
    lines = ["1 1 -30.0 0.0 0.0 1.5 -1"]
    nid = 2
    for (y, n) in chains:
        parent = 1
        for i in range(n):
            lines.append("%d 3 %.3f %.3f 0.0 0.3 %d" % (nid, float(i), y, parent))
            parent = nid
            nid += 1
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n")
    from allen_diameter.loading import swc_io
    return swc_io.read_swc(str(path))


def featured(swc, cfg):
    from allen_diameter.analysis import cell, node_features
    types_ = tuple(cfg.cleaning.neighbour_swc_types)
    scene = cell.cell_scene(swc, types_, cfg.acquisition.res0_um)
    return node_features.featured_scene(scene, node_features.initial_features(swc, types_, 1.0))


def test_work_list_examples_cap_threshold(tmp_path):
    """Block 36 work_list: the example nodes first; then end nodes of tier-1 neighbour segments whose peak
    x B reaches peak_min_gl, by decreasing peak; capped at work_max_nodes with the examples first and, when
    there are more examples than the cap, the examples cut (amendment 2, F5); no near_end node added; a huge
    threshold leaves the examples alone."""
    swc = chain_swc(tmp_path, [(0.0, 16), (2.6, 16)])         # ids 2..17 (y 0), 18..33 (y 2.6)
    cfg = default_config()
    fs = featured(swc, cfg)
    ex = [9]                                                    # x = 7 um on the first branch
    full = NI.work_list(ex, fs, swc, cfg, 3.0, 210.0)
    assert full[:1] == ex and len(set(full)) == len(full)
    extra = full[1:]
    assert extra, "a parallel dendrite 2.6 um away in focus must reach 3 gl"
    assert all(18 <= e <= 33 for e in extra), extra
    assert all(3.0 <= (e - 18) <= 12.0 for e in extra), extra   # never near_end (U < 3 within 3 um of an end)
    capped = NI.work_list(ex, fs, swc, dataclasses.replace(cfg, cleaning=dataclasses.replace(
        cfg.cleaning, work_max_nodes=3)), 3.0, 210.0)
    assert capped == full[:3]
    assert NI.work_list(ex, fs, swc, cfg, 1e9, 210.0) == ex
    two = dataclasses.replace(cfg, cleaning=dataclasses.replace(cfg.cleaning, work_max_nodes=2))
    assert NI.work_list([9, 25, 12], fs, swc, two, 3.0, 210.0) == [9, 25]      # F5: examples cut at the cap
    with pytest.raises(ValueError):
        NI.work_list([9, 9], fs, swc, cfg, 3.0, 210.0)
    with pytest.raises(ValueError):
        NI.work_list(ex, fs, swc, cfg, 3.0, 0.0)


def test_work_list_own_stretch_beyond_U(tmp_path):
    """Amendment 2 (F1) and 3 (3): the example's own nodes (own_ids_within at U = 5 um) are never candidates,
    but the stretch's segments beyond them are tier-1 neighbours (> close_lateral_um from the node) whose
    in-focus shadow on the patch (half-size 10 um) reaches 3 gl, so their non-near_end end nodes join."""
    swc = chain_swc(tmp_path, [(0.0, 31)])                      # ids 2..32 at x = 0..30
    cfg = default_config()
    fs = featured(swc, cfg)
    ex = [17]                                                   # x = 15
    work = NI.work_list(ex, fs, swc, cfg, 3.0, 210.0)
    own = set(range(17 - 6, 17 + 7))                            # x 9..21 (6 um > 5 on each side included)
    extra = set(work[1:])
    assert not (extra & own), sorted(extra & own)
    assert {8, 9, 10, 24, 25, 26} <= extra, sorted(extra)       # x 6..8 and 22..24: in the patch, in focus
    assert all(5 <= e <= 29 for e in extra), sorted(extra)      # x 3..27: not near_end


# ---------------------------------------------------------- build_node ---

def render_two_tubes(cfg, ks, noise_sd=0.0, seed=36):
    """Two flat tubes d 0.6 um, mu 1 /um, along x over [-20, 20] um at y 0 and y 3 um, axis at z 0, rendered
    by Block 24 (partition_vertical) and pixel-integrated, times B = 210 gl, over x [-1, 21], y [-11, 11] um,
    plus seeded white noise of SD noise_sd gl (none by default); a provider in the fetch_zblock contract."""
    from allen_diameter.model import geometry, render, scene_render
    res0, dz = cfg.acquisition.res0_um, cfg.acquisition.dz_um
    tubes = [geometry.Tube((0.0, 0.0, 0.0), 0.3, 0.0, 0.0, 1.0, 20.0, "axial"),
             geometry.Tube((0.0, 3.0, 0.0), 0.3, 0.0, 0.0, 1.0, 20.0, "axial")]
    left, top = int(math.floor(-1.0 / res0)), int(math.floor(-11.0 / res0))
    width, height = int(math.ceil(22.0 / res0)), int(math.ceil(22.0 / res0))
    f = render.fine_factor(0.6, cfg.renderer, res0)
    grid, inner = render.block_fine_grid(left, top, width, height, res0, f, 8)
    out = scene_render.render_scene(tubes, [1.0, 1.0], ks * dz, grid, cfg.renderer, +1,
                                    reduce=lambda pl: render.pixel_integrate(pl[inner[0], inner[1]], f))
    block = 210.0 * np.asarray(out.tau, dtype=float)
    return _provider_of(block, ks, left, top, width, height, res0), (block, left, top)


def _provider_of(block, ks, left, top, width, height, res0):

    def provider(l_, t_, w_, h_, k_lo, k_hi, valid_mask=None):
        kk = np.arange(int(k_lo), int(k_hi) + 1)
        o = np.full((kk.size, h_, w_), 255.0)
        valid = np.isin(kk, ks)
        x0, y0 = max(l_, left), max(t_, top)
        x1, y1 = min(l_ + w_, left + width), min(t_ + h_, top + height)
        for n, k in enumerate(kk):
            if k in ks:
                i = int(np.flatnonzero(ks == k)[0])
                o[n, y0 - t_:y1 - t_, x0 - l_:x1 - l_] = block[i, y0 - top:y1 - top, x0 - left:x1 - left]
        return o, kk, valid, CropFrame(int(l_), int(t_), 0, float(res0))
    return provider


_BN = {}


def build_case(tmp_path_factory):
    if "case" in _BN:
        return _BN["case"]
    from allen_diameter.analysis import cell, node_features
    from allen_diameter.loading import swc_io
    cfg = default_config()
    cfg = dataclasses.replace(
        cfg, renderer=dataclasses.replace(cfg.renderer, absorption="partition_vertical"),
        cleaning=dataclasses.replace(cfg.cleaning, shadow_absorption="partition_vertical", soma_mu_per_um=0.0,
                                     shadow_fine_factor=4),
        inference=dataclasses.replace(cfg.inference, absorption="partition_vertical"))
    ks = np.arange(-4, 5)
    provider, (blk, left, top) = render_two_tubes(cfg, ks)
    noisy = blk + np.random.default_rng(36).normal(0.0, 3.0, blk.shape)
    _BN["noisy"] = _provider_of(noisy, ks, left, top, blk.shape[2], blk.shape[1], cfg.acquisition.res0_um)
    tmp = tmp_path_factory.mktemp("b36_bn")
    lines = ["1 1 -30.0 1.5 0.0 1.0 -1"]
    nid = 2
    for y in (0.0, 3.0):                                        # nodes at x = -20, -18, ..., 20 (2 um apart)
        parent = 1
        for x in np.arange(-20.0, 20.01, 2.0):
            lines.append("%d 3 %.3f %.3f 0.0 0.3 %d" % (nid, x, y, parent))
            parent = nid
            nid += 1
    (tmp / "two.swc").write_text("\n".join(lines) + "\n")
    swc = swc_io.read_swc(str(tmp / "two.swc"))
    res0 = cfg.acquisition.res0_um
    scene = cell.cell_scene(swc, (2, 3, 4), res0)
    feats = node_features.initial_features(swc, (2, 3, 4), 1.0)
    fs = node_features.featured_scene(scene, feats)
    xyz = cell.to_image_um(swc.xyz, res0)
    run = [r for r in cell.stretches(swc, cfg.acquisition.dendrite_swc_types) if int(swc.ids[r[0]]) == 2][0]
    ob, _ = NI.own_branch(swc, run, xyz)
    _BN["case"] = (cfg, provider, scene, fs, ob, swc, feats)
    return _BN["case"]


def test_build_node_model_background(tmp_path_factory):
    """Amendments 2 and 4 on the rendered two-tube scene (true features, partition_vertical in both the scene
    and the shadow model) with seeded white noise of SD 3 gl: the node at x = 10 um is not refused; B_k within
    0.5 gl of 210 in every plane (the two-tube oracle); the planes' noise SD within 10 % of 3 (the white-noise
    oracle); own_ids are own_ids_within's; node_xyz the trace; theta0 pilot_start; the noise SD handed on is
    sigma_pixel x noise_factor, the factor equal to the bilinear weights' by hand [1e-12], the pixel SD the
    median of the planes'."""
    from allen_diameter.analysis import calibration, profiles
    cfg, _clean, scene, fs, ob, swc, feats = build_case(tmp_path_factory)
    provider = _BN["noisy"]
    i = int(np.flatnonzero(np.asarray(ob.ids) == 17)[0])       # x = 10 on the first dendrite (ids 2..22)
    assert ob.xyz_um[i][0] == pytest.approx(10.0)
    own = NI.own_ids_within(ob, i, cfg.inference.tube_half_length_um)[0]
    pl = types.SimpleNamespace(cx_um=10.0, cy_um=0.0, theta_rad=0.0, phi_rad=0.0, z_sub_um=0.0, k_star=0,
                               d_hat_um=0.6, mu_hat_per_um=1.0, B_bar=210.0, fit_status="converged")
    node, info = NI.build_node(ob, i, pl, provider, scene, cfg, own, fs)
    assert node is not None, info
    print("B_k - 210: %s; support %s; sigma pixel %.3g x factor %.4f" % (
        np.round(node.B_gl - 210.0, 4).tolist(), np.round(info["support_frac"], 3).tolist(),
        info["sigma_pixel_gl"], info["noise_factor"]))
    assert np.max(np.abs(node.B_gl - 210.0)) <= 0.5
    assert np.all(np.abs(np.asarray(info["sd_gl"]) - 3.0) <= 0.3), info["sd_gl"]
    assert node.own_ids == own and len(own) == 7               # x 4..16 (2 um apart, 6 > 5 included)
    assert_array_equal(node.node_xyz, ob.xyz_um[i])
    assert_allclose(node.theta0, NI.pilot_start(pl, ob.xyz_um[i]), rtol=0, atol=0)
    assert_array_equal(node.ks, np.arange(-4, 5))
    assert_allclose(node.planes_z, node.ks * cfg.acquisition.dz_um, rtol=0, atol=0)
    m = cfg.measure
    o, yh, eu = calibration.line_geometry(pl)
    fac = ref_noise_factor(node.left, node.top, cfg.acquisition.res0_um, o, yh, eu, profiles.profile_offsets(m),
                           profiles.n_along(m), float(m.profile_step_um))
    assert info["noise_factor"] == pytest.approx(fac, rel=1e-12)
    assert info["sigma_pixel_gl"] == float(np.median(info["sd_gl"]))
    assert info["sigma_gl"] == info["sigma_pixel_gl"] * info["noise_factor"]
    assert info["U_stretch_um"] == (6.0, 6.0)
    # the profiles read on the node's block are Block 5's read (independent bilinear read) [1e-9 gl]
    for q in range(node.ks.size):
        ref = nc_helpers.bilinear_read(node.block[q], node.left, node.top, cfg.acquisition.res0_um, node.origin_xy,
                                       node.y_hat, node.e_u, node.v, node.n_avg, node.step)
        from allen_diameter.analysis.clean_iteration import _profiles
        got = _profiles(np.asarray(node.block, dtype=float), node, cfg)[q]
        assert_allclose(got, ref, rtol=0, atol=1e-9)


def test_build_node_refusals(tmp_path_factory):
    """build_node's refusals under 'model': near_end before any fetch; no sharpest plane; a missing plane;
    no valid background (a black patch: B = 0)."""
    cfg, provider, scene, fs, ob, swc, feats = build_case(tmp_path_factory)
    i = int(np.flatnonzero(np.asarray(ob.ids) == 17)[0])
    own = NI.own_ids_within(ob, i, 5.0)[0]
    pl = types.SimpleNamespace(cx_um=10.0, cy_um=0.0, theta_rad=0.0, phi_rad=0.0, z_sub_um=0.0, k_star=0,
                               d_hat_um=0.6, mu_hat_per_um=1.0, B_bar=210.0, fit_status="converged")

    def no_fetch(*_a):
        raise AssertionError("fetched")
    node, info = NI.build_node(ob, 1, pl, no_fetch, scene, cfg, own, fs)      # x = -18: U_minus 2 < 3
    assert node is None and info["reason"] == NI.REASON_NEAR_END
    no_plane = types.SimpleNamespace(**dict(vars(pl), z_sub_um=float("nan")))
    node, info = NI.build_node(ob, i, no_plane, no_fetch, scene, cfg, own, fs)
    assert node is None and info["reason"] == NI.REASON_NO_PLANE

    def missing(l_, t_, w_, h_, k_lo, k_hi):
        b, kk, valid, fr = provider(l_, t_, w_, h_, k_lo, k_hi)
        valid = valid.copy()
        valid[2] = False
        return b, kk, valid, fr
    node, info = NI.build_node(ob, i, pl, missing, scene, cfg, own, fs)
    assert node is None and info["reason"] == NI.REASON_MISSING
    def dark(l_, t_, w_, h_, k_lo, k_hi):                     # a black patch: B = 0, not a valid background
        b, kk, valid, fr = provider(l_, t_, w_, h_, k_lo, k_hi)
        return np.zeros_like(b), kk, valid, fr
    node, info = NI.build_node(ob, i, pl, dark, scene, cfg, own, fs)
    assert node is None and info["reason"] == NI.REASON_NO_BG, info


def test_background_factory_fallback_and_log():
    """Amendment 4 (2): background_fn(node, F) is model_background on all_shadow(F, node); a round in which a
    plane has no valid background keeps the node's previous backgrounds (NodeData.B_gl at first) and logs."""
    cfg = fast_shadow_cfg(0.0)
    rng = np.random.default_rng(5)
    H, W = 60, 36
    fs = small_fscene()
    pz = np.array([-0.28, 0.0, 0.28])
    like = types.SimpleNamespace(left=0, top=0, block=np.zeros((3, H, W)), planes_z=pz)
    S = NI.all_shadow(fs, like, cfg)
    blk = 200.0 * (1.0 - S) + rng.normal(0.0, 1.0, S.shape)
    node = NodeData(node_id=7, own_ids=(1, 2), node_xyz=np.array([2.0, 1.5, 0.0]), block=blk,
                    B_gl=np.array([150.0, 151.0, 152.0]), ks=np.array([-1, 0, 1]), planes_z=pz, left=0, top=0,
                    origin_xy=np.array([2.0, 1.5]), y_hat=np.array([0.0, 1.0]), e_u=np.array([1.0, 0.0]),
                    v=(np.arange(11) - 5) * P, n_avg=1, step=P, theta0=None)
    logs = []
    fn = NI.background_factory(cfg, log=logs.append)
    out = fn(node, fs)
    want = NI.model_background(blk, S, CropFrame(0, 0, 0, cfg.acquisition.res0_um), node.origin_xy, cfg)
    assert np.all(want[3]) and not logs
    assert_array_equal(out, want[0])
    assert np.all(np.abs(out - 200.0) < 0.5)
    thick = dataclasses.replace(fs, seg_d_um=np.full((2, 2), 8.0))
    out2 = fn(node, thick)
    assert len(logs) == 1 and fn.history[7][1]["kept_previous"]
    assert_array_equal(out2, out)                                # the previous round's, not NodeData.B_gl
    fn2 = NI.background_factory(cfg, log=logs.append)
    assert_array_equal(fn2(node, thick), node.B_gl)             # first call: NodeData.B_gl


# --------------------------------------------- amendment 2 (F1) without an image ---

@pytest.mark.parametrize("spacing", [1.0, 2.0])
def test_tube_cut_decomposition_30deg(spacing):
    """Amendments 2 (F1) and 3 (1): on a straight 20 um tube at 30 deg (d 1 um, mu 1 /um), nodes `spacing` um
    apart, the node at its middle: the whole tube's profile plus Block 25's shadow of the stretch beyond the
    own nodes (what one cleaning round with true features adds back) equals the model tube over the own
    nodes within 1.5 % of the dip in every plane of k* +- 4 (the amendment's bound); without the cleaning the
    shortfall exceeds 1 % (the earlier F1, reproduced). partition_vertical throughout, no image rendered
    (repro_block36_tube_cut.case)."""
    import repro_block36_tube_cut as R
    after, before, n_own, U = R.case(30.0, spacing)
    print("30 deg, %.0f um: after %s; before %s" % (spacing, np.round(after, 4), np.round(before, 4)))
    assert np.all(after <= 0.015)
    assert np.all(before > 0.01)


@pytest.mark.xfail(strict=False, reason="tester's finding F2 (2026-11-14 18:30): on noise-free data with a small "
                   "shadow-model mismatch the trim of model_background (|res| against 3 x 1.4826 x MAD, not "
                   "centred on the median) rejects almost the whole support of the in-focus plane and build_node "
                   "refuses the node")
def test_build_node_noise_free_flat_scene(tmp_path_factory):
    """Not a stated oracle (amendment 4 states 'not refused' for the noise-free 30 deg chain only): the same
    two-tube scene noise-free, true features, matching absorptions, the shadow model at shadow_fine_factor 4
    (this file's speed setting; the default 8 passes: the mismatch at the in-focus edges is what triggers it)
    -- the node is expected to pass, with B_k within 0.5 gl of 210. It is refused (no_background in plane k*):
    reproducer of finding F2."""
    cfg, provider, scene, fs, ob, swc, feats = build_case(tmp_path_factory)
    i = int(np.flatnonzero(np.asarray(ob.ids) == 17)[0])
    own = NI.own_ids_within(ob, i, cfg.inference.tube_half_length_um)[0]
    pl = types.SimpleNamespace(cx_um=10.0, cy_um=0.0, theta_rad=0.0, phi_rad=0.0, z_sub_um=0.0, k_star=0,
                               d_hat_um=0.6, mu_hat_per_um=1.0, B_bar=210.0, fit_status="converged")
    node, info = NI.build_node(ob, i, pl, provider, scene, cfg, own, fs)
    print("noise-free two-tube scene: reason %r, support share %s" % (
        info.get("reason"), np.round(info.get("support_frac"), 4).tolist()))
    assert node is not None, info
    assert np.max(np.abs(node.B_gl - 210.0)) <= 0.5


def test_infer_fit_start_f4():
    """Amendment 2 (F4) and 1 (2): infer_nodes.fit_start -- a row not "refined" starts at pilot_start about
    the trace, theta0 = (d_hat, mu_hat, v_t, z_t + (z_sub - z_t)) = (.., .., v_t, z_sub); a refined row at its
    (d, mu), v0 = offset . y_hat, z_ax = z_t + offset_z [1e-12, by hand]."""
    import sys
    sys.path.insert(0, str(nc_helpers.Path(__file__).resolve().parent.parent.parent / "scripts"))
    import infer_nodes
    p = pilot()
    trace = np.array([10.0, -4.5, 0.4])
    yh = np.array([-math.sin(p.theta_rad), math.cos(p.theta_rad)])
    v_t = float((np.array([p.cx_um, p.cy_um]) - trace[:2]) @ yh)
    for src in ("pilot", "allen"):
        st, which = infer_nodes.fit_start(src, 0.5, 0.7, np.array([0.3, 0.2, 0.1]), p, trace)
        assert which == "pilot_start"
        assert_allclose(st, [p.d_hat_um, p.mu_hat_per_um, v_t, p.z_sub_um], rtol=0, atol=1e-12)
    off = np.array([0.25 * yh[0], 0.25 * yh[1], -0.15])
    st, which = infer_nodes.fit_start("refined", 0.5, 0.7, off, p, trace)
    assert which == "features"
    assert_allclose(st, [0.5, 0.7, 0.25, trace[2] - 0.15], rtol=0, atol=1e-12)
