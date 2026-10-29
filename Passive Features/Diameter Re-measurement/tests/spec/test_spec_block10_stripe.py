"""Spec-derived tests of Block 10.S (the stripe) and D-101.1 (flatness per
set) -- written by the integration tester from specs/SPEC.md (2026-10-28),
not from the code. Every expected value comes from spec_helpers (scipy quad /
brentq, closed-form geometry) or by hand.

Run (from the workstream directory):
    python -m pytest tests/spec/test_spec_block10_stripe.py -q
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import tempfile

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import spec_helpers as H  # noqa: F401  (puts src/, scripts/ and tests/smoke on sys.path)
from spec_helpers import (ref_margin, ref_patch_bounds, ref_shadow_depth, ref_sigma_r, ref_stripe, seg_dist,
                          pixel_centres_ref)

from allen_diameter.analysis import calibration as CAL
from allen_diameter.analysis import calibration_scene as CS
from allen_diameter.analysis.node_pipeline import Branch, NodeResult
from allen_diameter.analysis.plane_iteration import Scene
from allen_diameter.config import config_from_dict, default_config
from allen_diameter.loading import calibration_io

NAN = float("nan")
B0, MU0 = 200.0, 1.0


# ---------------------------------------------------------------- helpers --

def cfg_with(**cal):
    cfg = default_config()
    return dataclasses.replace(cfg, calibration=dataclasses.replace(cfg.calibration, **cal))


def pilot(cx=0.0, cy=0.0, theta=0.0, d=0.4, mu=MU0, B_bar=B0, z_sub=0.0, k_star=0, phi=0.0, alpha=None,
          status="converged", flags=()):
    """A pilot NodeResult (Block 5's columns) with the fields the calibration reads."""
    return NodeResult(node_id=7, type=3, x_um=cx, y_um=cy, z_um=z_sub, path_um=0.0, reg_verdict="",
                      s_star_um=NAN, dz_star_um=NAN, k_star=k_star, z_sub_um=z_sub, cx_um=cx, cy_um=cy, cz_um=z_sub,
                      theta_rad=theta, phi_rad=phi, steep=False, vertical=False, B_bar=B_bar,
                      B_bar_region="block_masked", d_hat_um=d, mu_hat_per_um=mu, v0_hat_um=0.0,
                      alpha_hat=(mu * d if alpha is None else alpha), fit_status=status, flags=tuple(flags), fit=None,
                      focus_rule="gradient_energy", focus_planes=np.empty(0, dtype=int), focus_score=np.empty(0),
                      focus_depth=np.empty(0), k_star_depth=-1)


def line_branch(theta=0.0, r=0.2, z=0.0, ids0=100, half=14.0, n=29, offset=0.0):
    """A straight flat branch through (offset along the normal) at heading theta."""
    s = np.linspace(-half, half, n)
    t = np.array([math.cos(theta), math.sin(theta)])
    nv = np.array([-math.sin(theta), math.cos(theta)])
    xy = s[:, None] * t[None, :] + offset * nv[None, :]
    return Branch.from_points(np.c_[xy, np.full(n, z)], r, ids=np.arange(ids0, ids0 + n))


def scene_from(branches, soma=None):
    parts = [Scene.from_branch(b) for b in branches]
    return Scene(np.vstack([q.seg_xyz for q in parts]), np.vstack([q.seg_radius for q in parts]),
                 np.vstack([q.seg_ids for q in parts]),
                 np.empty((0, 3)) if soma is None else np.atleast_2d(np.asarray(soma, dtype=float)))


def frame_shape(cfg, cx=0.0, cy=0.0, half=None):
    half = cfg.calibration.patch_half_um if half is None else half
    return CS.patch_frame(cx, cy, half, cfg.acquisition.res0_um)


def ref_masks(shape, frame, scene, own_ids, planes_z, B, mu, cfg, d_own, sigma_soma):
    """The spec's shaded pixels (10.S): neighbour q shades pixel p in plane k when its 2-D distance to the segment
    is <= rho_tau(max end radius, sigma_r(max_ends |z_end - z_k|)); the own branch the same with radius
    max(d_own/2, end radius); a soma disc as a tube of radius r_s at sigma_soma (distance to its centre).
    Returns (nb, own, dist_to_margin) where dist_to_margin is the smallest |dist - margin| per pixel (for ties)."""
    p, tau = float(frame.res_um_px), float(cfg.calibration.isolation_depth_gl)
    px, py = pixel_centres_ref(shape, frame.left, frame.top, p)
    z = np.asarray(planes_z, dtype=float)
    nb = np.zeros((z.size,) + tuple(shape), dtype=bool)
    ow = np.zeros_like(nb)
    gap = np.full((z.size,) + tuple(shape), np.inf)
    for q in range(scene.seg_xyz.shape[0]):
        a, b = scene.seg_xyz[q, :3], scene.seg_xyz[q, 3:]
        c_id, p_id = (int(x) for x in scene.seg_ids[q])
        is_own = c_id in own_ids and p_id in own_ids
        r = max(float(scene.seg_radius[q, 0]), float(scene.seg_radius[q, 1]))
        if is_own and d_own is not None and math.isfinite(d_own):
            r = max(r, 0.5 * d_own)
        dist = seg_dist(px, py, a[:2], b[:2])
        for k in range(z.size):
            s = float(ref_sigma_r(max(abs(a[2] - z[k]), abs(b[2] - z[k])), cfg.renderer))
            m = ref_margin(r, mu, s, B, tau)
            if m < 0:
                continue
            (ow if is_own else nb)[k] |= dist <= m
            gap[k] = np.minimum(gap[k], np.abs(dist - m))
    for x, y, r in np.atleast_2d(scene.soma).reshape(-1, 3):
        m = ref_margin(float(r), mu, float(sigma_soma), B, tau)
        if m >= 0:
            d = np.hypot(px - x, py - y)
            nb |= (d <= m)[None]
            gap = np.minimum(gap, np.abs(d - m)[None])
    return nb, ow, gap


# ------------------------------------------------------- (b) references ----

@pytest.mark.parametrize("r", [0.1, 0.3, 1.0, 2.0, 12.0])          # 12 um: a large soma disc (shaded as a tube)
@pytest.mark.parametrize("sigma", [0.08, 0.3, 1.0, 3.5])
def test_shadow_depths_against_quad(r, sigma):
    # 10.I's integral by quad (spec_helpers). Tolerance: 2e-3 of the dip at rho = 0 -- the spec allows the
    # Gauss-Legendre rule a ripple "at the 1e-3 level" (10.S); quad itself is at 1e-12.
    for mu in (0.3, 1.0, 3.0):
        rho = np.array([0.0, 0.5 * r, r, r + sigma, r + 3.0 * sigma])
        got = CS.shadow_depths(rho, r, mu, sigma, B0)
        ref = np.array([ref_shadow_depth(x, r, mu, sigma, B0) for x in rho])
        assert got.shape == rho.shape and got.dtype == np.float64
        assert np.all(np.abs(got - ref) <= 2e-3 * ref[0] + 1e-9), (r, sigma, mu, got, ref)


def test_shadow_depth_limits():
    # sigma -> 0: the unblurred deficit B (1 - exp(-2 mu sqrt(r^2 - rho^2))) inside the tube (spec 10.I);
    # mu -> infinity: B (Phi((r - rho)/s) - Phi((-r - rho)/s)) (an opaque strip blurred). Tolerances: the blur of
    # 1e-3 um against r = 1 um changes the value by O((s/r)^2) ~ 1e-6 relative away from the edge; mu = 200/um
    # leaves exp(-2 mu sqrt(...)) < 1e-9 except within ~1e-5 um of the edge, whose share of the integral is
    # below 1e-4 relative.
    # The spec caps the rule at 512 nodes ("max(16, ceil(6 r/sigma)), at most 512, so that a blur narrow against
    # the tube is resolved"), so the sharp limit is checked at the narrowest blur the cap resolves for r = 1 um,
    # sigma = 6 r / 400 = 0.015 um (400 nodes); the blur's own correction there is (sigma^2/2) |f''| B ~ 0.008 gl.
    # The expected value carries that second-order term exactly: B [f(rho) + (s^2/2) f''(rho)] with
    # f(v) = 1 - exp(-2 mu g), g = sqrt(r^2 - v^2): f'' = 2 mu e^{-2 mu g} (g'' - 2 mu g'^2), g' = -v/g,
    # g'' = -r^2/g^3; the next term, (s^4/8) f'''', is below 1e-5 gl here.
    from scipy import stats
    r, mu, s = 1.0, 0.7, 0.015
    for rho in (0.0, 0.3, 0.6):
        got = CS.shadow_depth(rho, r, mu, s, B0)
        g = math.sqrt(r * r - rho * rho)
        g1, g2 = -rho / g, -r * r / g ** 3
        f = 1.0 - math.exp(-2.0 * mu * g)
        f2 = 2.0 * mu * math.exp(-2.0 * mu * g) * (g2 - 2.0 * mu * g1 * g1)
        exp = B0 * (f + 0.5 * s * s * f2)
        assert abs(got - exp) <= 1e-3, (rho, got, exp)
    for rho, s in ((0.0, 0.5), (1.0, 0.5), (2.0, 1.0)):
        got = CS.shadow_depth(rho, r, 200.0, s, B0)
        exp = B0 * (stats.norm.cdf((r - rho) / s) - stats.norm.cdf((-r - rho) / s))
        assert abs(got - exp) <= 1e-3 * B0, (rho, s, got, exp)


@pytest.mark.parametrize("r,mu,sigma", [(0.1, 1.0, 0.08), (0.2, 1.0, 0.6), (0.35, 0.5, 3.5), (0.3, 3.0, 1.2),
                                        (2.0, 1.0, 0.08), (2.0, 1.0, 3.5), (5.0, 0.8, 0.6)])
def test_shadow_margin_against_brentq(r, mu, sigma):
    # rho_tau = the root of D(rho) = tau (10.S). The scan of 257 points and the bisection reach ~1e-13 um; the
    # quadrature's ripple (2e-3 of the dip, test above) moves the root by at most a few 1e-3 um where D is steep,
    # so 5e-3 um is the tolerance (the pixel is 0.1144 um).
    tau = default_config().calibration.isolation_depth_gl
    got = CS.shadow_margin(r, mu, sigma, B0, tau)
    ref = ref_margin(r, mu, sigma, B0, tau)
    assert ref > 0
    assert abs(got - ref) <= 5e-3, (got, ref)
    # and at the returned root the predicted shadow is tau within 2 % (the root's own check)
    assert abs(ref_shadow_depth(got, r, mu, sigma, B0) - tau) <= 0.02 * tau + 1e-6


def test_shadow_margin_nothing_shaded():
    # D(0) <= tau: -1.0 exactly (10.S: "no shading when D_{q,k}(0) <= tau")
    tau = 0.03
    assert ref_shadow_depth(0.0, 0.05, 1e-3, 1.0, B0) < tau
    assert CS.shadow_margin(0.05, 1e-3, 1.0, B0, tau) == -1.0
    assert CS.shadow_margin(0.2, 1.0, 0.3, B0, 1e6) == -1.0


def test_shadow_margin_monotone():
    # deeper shadow -> wider margin: increasing in B, mu and r; decreasing in tau (by the definition)
    base = dict(r=0.3, mu=1.0, sigma=0.6, B=B0, tau=0.03)

    def m(**kw):
        a = dict(base, **kw)
        return CS.shadow_margin(a["r"], a["mu"], a["sigma"], a["B"], a["tau"])
    assert m(B=100.0) < m() < m(B=250.0)
    assert m(mu=0.3) < m() < m(mu=3.0)
    assert m(r=0.1) < m() < m(r=1.0)
    assert m(tau=0.3) < m() < m(tau=0.003)


def test_shadow_masks_reference_with_soma():
    # shadow_masks against the per-pixel rule of 10.S with every quantity from spec_helpers: two neighbours (one
    # 0.5 um deeper), the own branch (d_hat 0.6 um > Allen's 0.4 um) and a soma disc, over the near stage's seven
    # planes (k* +- 3). The soma's blur is "the stage's largest blur" (10.I's convention: the stage's largest
    # offset, 3 dz). Pixels may disagree only within 2e-3 um of a margin (the margin's tolerance above).
    cfg = cfg_with()
    own = line_branch(r=0.2)
    n1 = line_branch(r=0.3, offset=2.5, ids0=300, half=6.0, n=7)
    n2 = line_branch(r=0.5, offset=-3.5, ids0=400, half=6.0, n=7, z=0.5)
    scene = scene_from([own, n1, n2], soma=[[1.0, -7.0, 1.5]])
    shape, frame = frame_shape(cfg, half=8.0)
    dz = cfg.acquisition.dz_um
    z = np.arange(-3, 4) * dz
    rows = CS.segment_rows(scene, frame, shape, own, 100.0)
    nb, ow = CS.shadow_masks(shape, frame, rows, scene.soma, z, B0, MU0, cfg, 0.6)
    s_soma = float(ref_sigma_r(3 * dz, cfg.renderer))
    nb_r, ow_r, gap = ref_masks(shape, frame, scene, set(own.ids.tolist()), z, B0, MU0, cfg, 0.6, s_soma)
    assert nb.shape == nb_r.shape and nb.dtype == bool and ow.dtype == bool
    tie = gap <= 2e-3
    assert not np.any((nb != nb_r) & ~tie), int(np.sum((nb != nb_r) & ~tie))
    assert not np.any((ow != ow_r) & ~tie), int(np.sum((ow != ow_r) & ~tie))
    assert nb.any() and ow.any()


@pytest.mark.parametrize("B_bar,mu", [(NAN, 1.0), (200.0, NAN), (200.0, 0.0), (0.0, 1.0), (-5.0, 1.0),
                                      (float("inf"), 1.0)])
def test_shadow_masks_geometric_fallback(B_bar, mu):
    # "Without a finite positive B_bar_j and mu_hat_j no shadow can be predicted and every margin is 10.P's
    # geometric r_q + n_sigma sigma_r (the own branch's too)" (10.S); the soma r_s + n_sigma sigma_soma.
    cfg = cfg_with()
    own = line_branch(r=0.2)
    n1 = line_branch(r=0.3, offset=2.5, ids0=300, half=6.0, n=7, z=0.3)
    scene = scene_from([own, n1], soma=[[0.0, -6.0, 1.0]])
    shape, frame = frame_shape(cfg, half=6.0)
    z = np.array([0.0, 0.84])
    rows = CS.segment_rows(scene, frame, shape, own, 100.0)
    nb, ow = CS.shadow_masks(shape, frame, rows, scene.soma, z, B_bar, mu, cfg, 0.6)
    p = cfg.acquisition.res0_um
    px, py = pixel_centres_ref(shape, frame.left, frame.top, p)
    ns = cfg.calibration.far_pixel_sigmas
    nb_r = np.zeros_like(nb)
    ow_r = np.zeros_like(ow)
    gap = np.full(nb.shape, np.inf)
    own_ids = set(own.ids.tolist())
    for q in range(scene.seg_xyz.shape[0]):
        a, b = scene.seg_xyz[q, :3], scene.seg_xyz[q, 3:]
        is_own = all(int(x) in own_ids for x in scene.seg_ids[q])
        r = max(scene.seg_radius[q])
        if is_own:
            r = max(r, 0.3)
        d = seg_dist(px, py, a[:2], b[:2])
        for k in range(z.size):
            m = r + ns * float(ref_sigma_r(max(abs(a[2] - z[k]), abs(b[2] - z[k])), cfg.renderer))
            (ow_r if is_own else nb_r)[k] |= d <= m
            gap[k] = np.minimum(gap[k], np.abs(d - m))
    s_soma = float(ref_sigma_r(0.5 * (z.max() - z.min()), cfg.renderer))
    d = np.hypot(px - 0.0, py + 6.0)
    nb_r |= (d <= 1.0 + ns * s_soma)[None]
    gap = np.minimum(gap, np.abs(d - (1.0 + ns * s_soma))[None])
    tie = gap <= 1e-9
    assert not np.any((nb != nb_r) & ~tie) and not np.any((ow != ow_r) & ~tie)
    assert ow.any(), "the own cylinder must be removed with the geometric margin too"


# ------------------------------------------------------ (a) known answer ---

def _stripe_of(cfg, scene, own, node, half, kk=(0,)):
    shape, frame = frame_shape(cfg, node.cx_um, node.cy_um, half)
    support, stripe = CAL.support_masks(shape, frame, np.array(kk), node, own, scene, cfg)
    return support, stripe, shape, frame


def test_stripe_known_answer_parallel_neighbour():
    # own tube along x, one parallel neighbour (r 0.3 um) at lateral distance D = 6 um on +v, nothing on -v:
    # L+ = D - rho_tau(nb), v_own = rho_tau(own) within one pixel, L- = the patch's edge, W* = the widest (12 um).
    cfg = cfg_with()
    p = cfg.acquisition.res0_um
    tau = cfg.calibration.isolation_depth_gl
    D, half = 6.0, 10.0
    own = line_branch(r=0.2)
    nbr = line_branch(r=0.3, offset=D, ids0=300)
    scene = scene_from([own, nbr])
    node = pilot(d=0.4)
    support, stripe, shape, frame = _stripe_of(cfg, scene, own, node, half)
    info = stripe[0]
    s0 = float(ref_sigma_r(0.0, cfg.renderer))
    rho_nb, rho_own = ref_margin(0.3, MU0, s0, B0, tau), ref_margin(0.2, MU0, s0, B0, tau)
    assert abs(info["l_plus_um"] - (D - rho_nb)) <= p, (info, D - rho_nb)
    assert abs(info["v_own_plus_um"] - rho_own) <= p and abs(info["v_own_minus_um"] - rho_own) <= p, (info, rho_own)
    assert abs(info["l_minus_um"] - half) <= p, info
    assert info["w_um"] == max(cfg.calibration.stripe_half_widths_um)
    assert_allclose(info["side_plus_um"], info["l_plus_um"] - info["v_own_plus_um"], atol=1e-12)
    assert_allclose(info["side_minus_um"], info["l_minus_um"] - info["v_own_minus_um"], atol=1e-12)
    assert info["ok"] and info["reason"] == ""
    # the pixel count: every band pixel (|u| <= 12, inside the +-10 um patch: the whole width) in the run columns
    px, py = pixel_centres_ref(shape, frame.left, frame.top, p)
    v = py
    in_run = ((v >= info["v_own_plus_um"] - 0.5 * p) & (v < info["l_plus_um"] - 0.5 * p)) | \
             ((v <= -info["v_own_minus_um"] + 0.5 * p) & (v > -info["l_minus_um"] + 0.5 * p))
    assert abs(info["n_pixels"] - int(in_run.sum())) <= shape[1], (info["n_pixels"], int(in_run.sum()))
    assert int(support[0].sum()) == info["n_pixels"]


@pytest.mark.parametrize("theta_deg", [35.0, 70.0, 135.0])
def test_stripe_rotation_keeps_count(theta_deg):
    # (a) "the same scene rotated by theta gives N within 3 %". The patch is a square in the image axes, so a run
    # that ends at the patch's edge changes length with the rotation; here both runs end at a neighbour (+6 um and
    # -8 um) and the band's rectangle (|u| <= 12, -8 <= v <= 6: corners within 14.4 um of the centre) lies inside
    # the +-25 um patch at every angle, so N changes only by the pixelisation of its edges.
    cfg = cfg_with()
    D = 6.0
    base = None
    for th in (0.0, math.radians(theta_deg)):
        own = line_branch(theta=th, r=0.2, half=30.0, n=61)
        nbr = line_branch(theta=th, r=0.3, offset=D, ids0=300, half=30.0, n=61)
        nbr2 = line_branch(theta=th, r=0.3, offset=-8.0, ids0=500, half=30.0, n=61)
        scene = scene_from([own, nbr, nbr2])
        node = pilot(theta=th, d=0.4)
        _s, stripe, _sh, _fr = _stripe_of(cfg, scene, own, node, 25.0)
        if base is None:
            base = stripe[0]
            continue
        rel = abs(stripe[0]["n_pixels"] - base["n_pixels"]) / base["n_pixels"]
        assert rel <= 0.03, (theta_deg, stripe[0], base)
        assert abs(stripe[0]["l_plus_um"] - base["l_plus_um"]) <= 2 * cfg.acquisition.res0_um


def test_stripe_mirror_symmetry():
    # mirroring the scene across the measuring line swaps the two sides
    cfg = cfg_with()
    own = line_branch(r=0.2)
    out = []
    for sgn in (+1, -1):
        nbr = line_branch(r=0.3, offset=sgn * 5.0, ids0=300)
        scene = scene_from([own, nbr])
        _s, stripe, _sh, _fr = _stripe_of(cfg, scene, own, pilot(d=0.4), 10.0)
        out.append(stripe[0])
    a, b = out
    p = cfg.acquisition.res0_um
    assert abs(a["l_plus_um"] - b["l_minus_um"]) <= p and abs(a["l_minus_um"] - b["l_plus_um"]) <= p
    assert abs(a["n_pixels"] - b["n_pixels"]) <= 0.01 * a["n_pixels"]


# ---------------------------------------------------------- (c) invariants --

def _random_scene(rng, cfg, half):
    th = float(rng.uniform(0, math.pi))
    own = line_branch(theta=th, r=float(rng.uniform(0.1, 0.4)), half=30.0, n=61)
    parts = [own]
    for m in range(int(rng.integers(0, 4))):
        th2 = th + float(rng.uniform(-0.6, 0.6))
        off = float(rng.uniform(-half, half))
        parts.append(line_branch(theta=th2, r=float(rng.uniform(0.1, 0.8)), offset=off, ids0=300 + 100 * m,
                                 half=30.0, n=31, z=float(rng.uniform(-1.5, 1.5))))
    soma = None
    if rng.uniform() < 0.4:
        soma = [[float(rng.uniform(-half, half)), float(rng.uniform(-half, half)), float(rng.uniform(1, 4))]]
    return th, own, scene_from(parts, soma)


def test_stripe_invariants_random_scenes():
    # On 15 seeded random scenes and the near stage's planes: the stripe lies in the band of W*, holds no shaded
    # pixel, n_pixels = its pixel count, sides = L - v_own, ok/reason by the thresholds, N(W*) >= N(W) for every W
    # of the grid; and the whole record equals spec_helpers.ref_stripe (the spec's walk, written independently)
    # on the same masks.
    rng = np.random.default_rng(20261028)
    cfg = cfg_with()
    cal = cfg.calibration
    half = 12.0
    p = cfg.acquisition.res0_um
    for trial in range(15):
        th, own, scene = _random_scene(rng, cfg, half)
        node = pilot(theta=th, d=float(rng.uniform(0.2, 1.0)))
        shape, frame = frame_shape(cfg, half=half)
        kk = np.arange(-3, 4)
        support, stripe = CAL.support_masks(shape, frame, kk, node, own, scene, cfg)
        rows = CS.segment_rows(scene, frame, shape, own, 100.0)
        nb, ow = CS.shadow_masks(shape, frame, rows, scene.soma, kk * cfg.acquisition.dz_um, node.B_bar,
                                 node.mu_hat_per_um, cfg, node.d_hat_um)
        px, py = pixel_centres_ref(shape, frame.left, frame.top, p)
        u = math.cos(th) * px + math.sin(th) * py
        for q in range(kk.size):
            info, mask = stripe[q], support[q]
            assert set(info) == {"w_um", "v_own_minus_um", "v_own_plus_um", "l_minus_um", "l_plus_um",
                                 "side_minus_um", "side_plus_um", "n_pixels", "ok", "reason"}
            assert not np.any(mask & (nb[q] | ow[q])), (trial, q)
            assert np.all(np.abs(u[mask]) <= info["w_um"] + 1e-9), (trial, q)
            assert info["n_pixels"] == int(mask.sum())
            assert_allclose(info["side_plus_um"], info["l_plus_um"] - info["v_own_plus_um"], atol=1e-9)
            assert_allclose(info["side_minus_um"], info["l_minus_um"] - info["v_own_minus_um"], atol=1e-9)
            exp_reason = ("too_few" if info["n_pixels"] < cal.stripe_min_pixels else
                          "side_short" if min(info["side_plus_um"], info["side_minus_um"]) < cal.stripe_min_side_um
                          else "")
            assert info["reason"] == exp_reason and info["ok"] == (exp_reason == ""), info
            for w in cal.stripe_half_widths_um:
                c1 = cfg_with(stripe_half_widths_um=(float(w),))
                _m1, i1 = CS.stripe_support(nb[q], ow[q], frame, 0.0, 0.0, th, c1)
                assert i1["n_pixels"] <= info["n_pixels"], (trial, q, w)
            ref = ref_stripe(nb[q], ow[q], frame.left, frame.top, p, 0.0, 0.0, th, cal.stripe_half_widths_um,
                             cal.stripe_min_pixels, cal.stripe_min_side_um)
            assert ref["w"] == info["w_um"] and ref["n"] == info["n_pixels"], (trial, q, ref["n"], info)
            for k_ref, k_code in (("l_plus", "l_plus_um"), ("l_minus", "l_minus_um"),
                                  ("v_own_plus", "v_own_plus_um"), ("v_own_minus", "v_own_minus_um")):
                assert abs(ref[k_ref] - info[k_code]) <= 1e-9, (trial, q, k_ref, ref[k_ref], info[k_code])
            assert_array_equal(ref["mask"], mask)


def test_stripe_tie_takes_smaller_width_whatever_the_order():
    # "W* = argmax_W N_k(W) (the smaller W on a tie)": on a +-3 um patch every band of W >= 4 um holds the whole
    # patch, so N(4) = N(8) = N(12) and W* = 4, also when the grid is given in decreasing order.
    for widths in ((1.0, 2.0, 4.0, 8.0, 12.0), (12.0, 8.0, 4.0, 2.0, 1.0), (8.0, 4.0, 12.0)):
        cfg = cfg_with(stripe_half_widths_um=widths, stripe_min_pixels=6, stripe_min_side_um=0.5)
        own = line_branch(r=0.2)
        _s, stripe, _sh, _fr = _stripe_of(cfg, scene_from([own]), own, pilot(d=0.4), 3.0)
        assert stripe[0]["w_um"] == 4.0, (widths, stripe[0])


def test_stripe_side_short_and_too_few():
    # (c) "a neighbour hugging both sides within 3 um gives side_short"; a run under the pixel floor: too_few
    cfg = cfg_with()
    own = line_branch(r=0.2)
    scene = scene_from([own, line_branch(r=0.5, offset=2.6, ids0=300), line_branch(r=0.5, offset=-2.6, ids0=400)])
    _s, stripe, _sh, _fr = _stripe_of(cfg, scene, own, pilot(d=0.4), 10.0)
    assert not stripe[0]["ok"] and stripe[0]["reason"] == "side_short", stripe[0]
    scene2 = scene_from([own, line_branch(r=0.5, offset=1.1, ids0=300), line_branch(r=0.5, offset=-1.1, ids0=400)])
    _s, stripe2, _sh, _fr = _stripe_of(cfg, scene2, own, pilot(d=0.4), 10.0)
    assert not stripe2[0]["ok"] and stripe2[0]["reason"] == "too_few", stripe2[0]


def test_stripe_no_segment_but_own():
    # (f) no segment: the band to the patch's edges minus the own cylinder
    cfg = cfg_with()
    p = cfg.acquisition.res0_um
    own = line_branch(r=0.2)
    support, stripe, shape, frame = _stripe_of(cfg, scene_from([own]), own, pilot(d=0.4), 10.0)
    info = stripe[0]
    assert abs(info["l_plus_um"] - 10.0) <= p and abs(info["l_minus_um"] - 10.0) <= p
    px, py = pixel_centres_ref(shape, frame.left, frame.top, p)
    expect = np.abs(py) >= info["v_own_plus_um"] - 0.5 * p          # the band (|u| <= 12 > 10) less the own columns
    assert_array_equal(support[0], expect)


# ------------------------------------------------------ square (10.P) -------

def test_square_support_equals_far_pixel_rule():
    # background_support = "square": 10.P's far pixels, bit for bit: a pixel is far when farther than
    # max(r0, r1) + n_sigma sigma_r(delta_q,k) from every segment (the own included) and r_s + n_sigma
    # sigma_r(the stage's largest offset) from every soma; no stripe record.
    cfg = cfg_with(background_support="square")
    own = line_branch(r=0.2)
    n1 = line_branch(r=0.3, offset=3.0, ids0=300, z=0.6)
    scene = scene_from([own, n1], soma=[[2.0, -7.0, 2.0]])
    node = pilot(d=0.4)
    shape, frame = frame_shape(cfg, half=9.0)
    kk = np.arange(-3, 4)
    z = kk * cfg.acquisition.dz_um
    support, stripe = CAL.support_masks(shape, frame, kk, node, own, scene, cfg)
    assert stripe is None
    p, ns = cfg.acquisition.res0_um, cfg.calibration.far_pixel_sigmas
    px, py = pixel_centres_ref(shape, frame.left, frame.top, p)
    near = np.zeros(support.shape, dtype=bool)
    gap = np.full(support.shape, np.inf)
    for q in range(scene.seg_xyz.shape[0]):
        a, b = scene.seg_xyz[q, :3], scene.seg_xyz[q, 3:]
        d = seg_dist(px, py, a[:2], b[:2])
        for k in range(z.size):
            m = max(scene.seg_radius[q]) + ns * float(ref_sigma_r(max(abs(a[2] - z[k]), abs(b[2] - z[k])),
                                                                  cfg.renderer))
            near[k] |= d <= m
            gap[k] = np.minimum(gap[k], np.abs(d - m))
    d = np.hypot(px - 2.0, py + 7.0)
    m = 2.0 + ns * float(ref_sigma_r(3 * cfg.acquisition.dz_um, cfg.renderer))
    near |= (d <= m)[None]
    gap = np.minimum(gap, np.abs(d - m)[None])
    tie = gap <= 1e-9
    assert not np.any((support != ~near) & ~tie)


# -------------------------------------------- the patch and the verdict -----

def test_patch_frame_equals_fetch_patch_request():
    # the verdict before any fetch uses patch_frame; the scan uses fetch_patch: both must address the same pixels,
    # the square of pixel centres within +-patch_half_um of the centre (10.P)
    rng = np.random.default_rng(7)
    p = default_config().acquisition.res0_um
    seen = []

    def provider(left, top, width, height, k_lo, k_hi):
        seen.append((left, top, width, height))
        from allen_image_io import CropFrame
        ks = np.arange(k_lo, k_hi + 1)
        return np.zeros((ks.size, height, width), dtype=np.uint8), ks, np.ones(ks.size, bool), CropFrame(left, top, 0, p)
    for _ in range(40):
        cx, cy = rng.uniform(-500, 2500, 2)
        half = float(rng.choice([3.0, 12.0, 25.0]))
        shape, frame = CS.patch_frame(cx, cy, half, p)
        CS.fetch_patch(provider, cx, cy, half, 0, 0, p)
        left, top, width, height = seen[-1]
        assert (frame.left, frame.top, shape) == (left, top, (height, width))
        assert (left, top, width, height) == ref_patch_bounds(cx, cy, half, p)
        px, py = pixel_centres_ref(shape, left, top, p)
        assert np.all(np.abs(px - cx) <= half + 1e-9) and np.all(np.abs(py - cy) <= half + 1e-9)


def _flat_patch(cfg, node, kk, half, level=150.0):
    shape, frame = CS.patch_frame(node.cx_um, node.cy_um, half, cfg.acquisition.res0_um)
    block = np.full((len(kk),) + shape, level)
    return (block, np.asarray(kk), np.ones(len(kk), dtype=bool), frame)


@pytest.mark.parametrize("which,soma_v", [("near", -6.4), ("far", -16.0)])
def test_verdict_equals_scan_stripe_at_the_pilot_plane(which, soma_v):
    # 10.S: "The node's isolation verdict before any fetch (10.I (vi)) is the same test at the plane nearest the
    # pilot's depth" -- and a soma disc is shaded "as a tube of radius r_s at the stage's largest blur (10.I's
    # convention)". So the verdict (calibration_reasons) and the scan's own stripe at that plane (patch_planes over
    # the stage) must agree. Scene: the own branch along x, a soma disc of radius 2 um on the -v side, nothing
    # else; flat image so b0 > 0 everywhere.
    cfg = cfg_with()
    cal, dz = cfg.calibration, cfg.acquisition.dz_um
    own = line_branch(r=0.2 if which == "near" else 0.35, half=30.0, n=61)
    scene = scene_from([own], soma=[[0.0, soma_v, 2.0]])
    node = pilot(d=0.4 if which == "near" else 0.7, mu=1.0)
    allen_d = 0.4 if which == "near" else 0.7
    kk = CAL.stage_planes(node, cfg, which)
    planes_z = kk * dz
    verdict = CAL.calibration_reasons(node, cfg, allen_d, own, 30, scene, planes_z)
    patch = _flat_patch(cfg, node, kk, cal.patch_half_um)
    idx, far, b, ok, frame, stripe = CAL.patch_planes(patch, kk, node, own, scene, cfg)
    q = int(np.argmin(np.abs(planes_z - node.z_sub_um)))
    scan_ok = bool(stripe[q]["ok"])
    # the spec's expectation at that plane, independently: the soma shades to rho_tau(2 um, sigma_r(P dz))
    P = cal.offsets_planes if which == "near" else cal.far_offsets_planes
    rho_soma = ref_margin(2.0, 1.0, float(ref_sigma_r(P * dz, cfg.renderer)), B0, cal.isolation_depth_gl)
    rho_own = ref_margin(max(0.5 * node.d_hat_um, own.radius_um[0]), 1.0, float(ref_sigma_r(0.0, cfg.renderer)), B0,
                         cal.isolation_depth_gl)
    side_minus_spec = (-soma_v - rho_soma) - rho_own
    spec_ok = side_minus_spec >= cal.stripe_min_side_um
    assert scan_ok == spec_ok, (stripe[q], side_minus_spec)
    assert (verdict == []) == scan_ok, ("verdict %s but the scan's stripe at the pilot's plane is %s"
                                        % (verdict, stripe[q]))


def test_patch_planes_ok_follows_the_stripe_not_the_far_fraction():
    # under "stripe" a plane is usable with >= stripe_min_pixels clear pixels, both runs >= 3 um and b0 > 0
    # (10.S); 10.P's 5 % far fraction is the square's rule. Two neighbours at +-4.6 um and half-widths up to
    # 4 um leave a stripe of about 2 % of a +-25 um patch: usable under the stripe.
    cfg = cfg_with(stripe_half_widths_um=(1.0, 2.0, 4.0))
    own = line_branch(r=0.2, half=30.0, n=61)
    scene = scene_from([own, line_branch(r=0.3, offset=4.6, ids0=300, half=30.0, n=61),
                        line_branch(r=0.3, offset=-4.6, ids0=400, half=30.0, n=61)])
    node = pilot(d=0.4)
    kk = CAL.stage_planes(node, cfg, "near")
    patch = _flat_patch(cfg, node, kk, cfg.calibration.patch_half_um)
    idx, far, b, ok, frame, stripe = CAL.patch_planes(patch, kk, node, own, scene, cfg)
    frac = far.reshape(kk.size, -1).mean(axis=1)
    q0 = int(np.flatnonzero(kk == 0)[0])
    assert frac[q0] < cfg.calibration.background_min_far_frac, frac
    assert stripe[q0]["ok"] and ok[q0]
    for q in range(kk.size):
        assert ok[q] == (stripe[q]["ok"] and b[q, 0] > 0), (q, stripe[q], b[q])
    # and a dark plane (b0 <= 0) makes the scan's plane bg_few
    block = patch[0].copy()
    block[q0] = 0.0
    scan = CAL.plane_scan((block, patch[1], patch[2], patch[3]), node, own, scene, cfg)
    assert scan.status[q0] == "bg_few", scan.status


# ---------------------------------------------- D-101.1: flatness per set ---

def test_flatness_per_set_boundaries():
    cfg = cfg_with()
    cal = cfg.calibration
    assert cal.node_phi_max_deg == 10.0 and cal.node_phi_max_deg_far == 15.0
    near, far, out = 0.4, 0.7, 1.0
    R = math.radians
    assert CAL.criteria_reasons(R(10.0), 0.3, near, cfg) == []            # <= bound
    assert CAL.criteria_reasons(R(10.0) * (1 + 1e-9), 0.3, near, cfg) == ["not_flat"]
    assert CAL.criteria_reasons(R(12.0), 0.3, near, cfg) == ["not_flat"]
    assert CAL.criteria_reasons(R(12.0), 0.3, far, cfg) == []
    assert CAL.criteria_reasons(R(15.0), 0.3, far, cfg) == []
    assert CAL.criteria_reasons(R(15.0) * (1 + 1e-9), 0.3, far, cfg) == ["not_flat"]
    assert CAL.criteria_reasons(R(16.0), 0.3, far, cfg) == ["not_flat"]
    # a node in neither set is judged by the near bound, after allen_diameter_out (D-101.1)
    assert CAL.criteria_reasons(R(12.0), 0.3, out, cfg) == ["allen_diameter_out", "not_flat"]
    assert CAL.criteria_reasons(R(9.0), 0.3, out, cfg) == ["allen_diameter_out"]
    assert CAL.criteria_reasons(R(12.0), 0.3, NAN, cfg) == ["allen_diameter_missing", "not_flat"]
    # set edges: 0.6 near (bound 10), 0.6 + 1e-9 far (bound 15), 0.8 far, 0.8 + 1e-9 neither
    assert CAL.criteria_reasons(R(12.0), 0.3, 0.6, cfg) == ["not_flat"]
    assert CAL.criteria_reasons(R(12.0), 0.3, 0.6 + 1e-9, cfg) == []
    assert CAL.criteria_reasons(R(12.0), 0.3, 0.8, cfg) == []
    assert CAL.criteria_reasons(R(12.0), 0.3, 0.8 + 1e-9, cfg) == ["allen_diameter_out", "not_flat"]
    # a NaN tilt is not flat ("phi <= bound" fails), for both sets
    assert CAL.criteria_reasons(NAN, 0.3, near, cfg) == ["not_flat"]
    assert CAL.criteria_reasons(NAN, 0.3, far, cfg) == ["not_flat"]
    # the bounds are read from the configuration (swappable)
    c2 = cfg_with(node_phi_max_deg=4.0, node_phi_max_deg_far=6.0)
    assert CAL.criteria_reasons(R(5.0), 0.3, near, c2) == ["not_flat"]
    assert CAL.criteria_reasons(R(5.0), 0.3, far, c2) == []


def test_calibration_reasons_tail_and_csv_rule_follow_the_set():
    import calibrate_kernel as CK
    cfg = cfg_with()
    R = math.radians
    for phi in (R(9.0), R(12.0), R(16.0)):
        for allen_d in (0.4, 0.7, 1.0):
            node = pilot(phi=phi, d=0.3)
            reasons = CAL.calibration_reasons(node, cfg, allen_d)
            assert reasons == CAL.criteria_reasons(phi, node.alpha_hat, allen_d, cfg)
    rows = []
    for nid, (phi_deg, r_allen) in enumerate([(9, 0.2), (12, 0.2), (12, 0.35), (16, 0.35), (12, 0.5)], start=1):
        rows.append(dict(node_id=nid, fit_status="converged", flags="", phi_rad=R(phi_deg), alpha_hat=0.3,
                         allen_radius_um=r_allen))
    assert CK.candidates_from_rows(rows, cfg) == {1, 3}


# ---------------------------------------------- configuration (Block 1) -----

def test_new_calibration_fields_defaults_validate_and_signatures():
    cfg = default_config()
    cal = cfg.calibration
    assert cal.background_support == "stripe"
    assert tuple(cal.stripe_half_widths_um) == (1.0, 2.0, 4.0, 8.0, 12.0)
    assert cal.stripe_min_pixels == 500 and cal.stripe_min_side_um == 3.0
    # D-108 (the user, 2026-10-29): tau = 0.3 gl, the spec's value since then (was 0.03, D-081.1 / D-101.5)
    assert cal.isolation_depth_gl == 0.3
    # Block 1's pinned default hashes (SPEC Block 1, D-096): estimator 59678d4e2aedf2b6, simulation 52aa73e479eb77cf
    assert cfg.signature_hash("estimator") == "59678d4e2aedf2b6"
    assert cfg.signature_hash("simulation") == "52aa73e479eb77cf"
    # the new fields are in no signature (10.S: "none in a signature")
    changed = cfg_with(background_support="square", stripe_half_widths_um=(2.0, 5.0), stripe_min_pixels=900,
                       stripe_min_side_um=1.0, node_phi_max_deg_far=20.0, node_phi_max_deg=7.0)
    changed.validate()
    for which in ("estimator", "simulation"):
        assert changed.signature_hash(which) == cfg.signature_hash(which)
    # illegal values refused (Block 1: validate() refuses illegal values)
    for bad in (dict(background_support="disc"), dict(stripe_half_widths_um=()), dict(stripe_half_widths_um=(0.0,)),
                dict(stripe_half_widths_um=(NAN,)), dict(stripe_half_widths_um=(-1.0, 2.0)),
                dict(node_phi_max_deg_far=NAN), dict(node_phi_max_deg_far=-1.0), dict(node_phi_max_deg_far=91.0),
                dict(stripe_min_side_um=NAN), dict(stripe_min_side_um=-0.5), dict(stripe_min_pixels=0)):
        with pytest.raises(ValueError):
            cfg_with(**bad).validate()


def test_config_json_calibration_section_is_read():
    # the entry point's --config-json carries a "calibration" section (SPEC section 6: reduce patch_half_um and
    # far_offsets_planes through --config-json)
    c = config_from_dict({"calibration": {"background_support": "square", "stripe_half_widths_um": [3, 6],
                                          "stripe_min_pixels": 50, "node_phi_max_deg_far": 12.5,
                                          "patch_half_um": 8.0, "far_offsets_planes": 6}})
    assert c.calibration.background_support == "square"
    assert tuple(c.calibration.stripe_half_widths_um) == (3, 6)
    assert c.calibration.stripe_min_pixels == 50 and c.calibration.node_phi_max_deg_far == 12.5


# ------------------------------------------------ the scans file (10.S) -----

def test_scans_json_carries_the_stripe_and_reads_old_files():
    cfg = cfg_with()
    own = line_branch(r=0.2)
    scene = scene_from([own, line_branch(r=0.3, offset=5.0, ids0=300)])
    node = pilot(d=0.4)
    kk = CAL.stage_planes(node, cfg, "near")
    patch = _flat_patch(cfg, node, kk, 10.0)
    scan = CAL.plane_scan(patch, node, own, scene, cfg)
    assert scan.stripe is not None and len(scan.stripe) == kk.size
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "scans.json")
        calibration_io.save_scans([scan], path, {"t": 1})
        text = open(path).read()
        json.loads(text)                                   # strict JSON
        assert "NaN" not in text and "Infinity" not in text
        back, _m = calibration_io.load_scans(path)
        assert back[0].stripe == scan.stripe
        # a file written before 10.S (no "stripe") loads with stripe None
        d = json.loads(text)
        for sc in d["scans"] if "scans" in d else []:
            sc.pop("stripe", None)
        with open(path, "w") as f:
            json.dump(d, f)
        old, _m = calibration_io.load_scans(path)
        assert old[0].stripe is None and np.array_equal(old[0].ks, scan.ks)


def test_stripe_determinism():
    cfg = cfg_with()
    rng = np.random.default_rng(3)
    th, own, scene = _random_scene(rng, cfg, 12.0)
    node = pilot(theta=th)
    shape, frame = frame_shape(cfg, half=12.0)
    a = CAL.support_masks(shape, frame, np.arange(-3, 4), node, own, scene, cfg)
    b = CAL.support_masks(shape, frame, np.arange(-3, 4), node, own, scene, cfg)
    assert_array_equal(a[0], b[0]) and a[1] == b[1]
    assert a[1] == b[1]


def test_square_restores_the_predicted_depth_verdict():
    # background_support = "square" restores 10.I's node verdict: not_isolated exactly when some traced neighbour's
    # predicted shadow in the +-patch_half_um patch (at its lateral distance rho_q to the square, 0 when it crosses
    # it) exceeds isolation_depth_gl in a plane of the stage (both ends' depths tried; spec 10.I (vi)).
    cfg = cfg_with(background_support="square")
    cal, dz = cfg.calibration, cfg.acquisition.dz_um
    node = pilot(d=0.4)
    planes_z = CAL.stage_planes(node, cfg, "near") * dz
    own = line_branch(r=0.2, half=40.0, n=81)
    for off, expect in ((3.0, True), (3 * 2 * cal.patch_half_um, False)):
        nbr = line_branch(r=0.3, offset=off, ids0=300, half=40.0, n=81, z=0.4)
        scene = scene_from([own, nbr])
        rho = max(0.0, off - cal.patch_half_um)
        depth = max(ref_shadow_depth(rho, 0.3, MU0, float(ref_sigma_r(abs(0.4 - z), cfg.renderer)), B0)
                    for z in planes_z)
        assert (depth > cal.isolation_depth_gl) == expect, (off, depth)
        reasons = CAL.calibration_reasons(node, cfg, 0.4, own, 40, scene, planes_z)
        assert reasons == (["not_isolated"] if expect else []), (off, reasons, depth)
