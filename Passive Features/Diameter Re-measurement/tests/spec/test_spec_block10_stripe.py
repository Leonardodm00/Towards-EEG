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


# =============================================================================
# Run D4 (integration tester, 2026-11-13, commit 1555036): Block 10's calibration
# sets, dip statistics, the growth fit of Eq. 4 with D-119's sparse Jacobian,
# kernel_from_growth, the anchor (10.A, D-117.4), 10.G's background SD and
# goodness of fit, 10.C's trace direction, typical background and choice, and
# 10.C.R's digest, key and reuse. Expected values come from spec_helpers_D4
# (written from SPEC.md) or by hand; every tolerance is justified where used.
# Run: python -m pytest tests/spec/test_spec_block10_stripe.py -q -k d4
# =============================================================================
import spec_helpers_D4 as D4  # noqa: E402

from allen_diameter.analysis import candidate_search as D4_CSR  # noqa: E402
from allen_diameter.analysis.node_pipeline import Branch as D4_Branch  # noqa: E402
from allen_diameter.loading import table_io as D4_table_io  # noqa: E402
from allen_image_io import CropFrame as D4_CropFrame  # noqa: E402

D4_G_SYM = np.array([0.0, 0.010, 0.060, 0.150, 0.280])                       # um^2 at 0, 0.28 .. 1.12 um
D4_G_MIN = np.array([0.30, 0.17, 0.07, 0.012, 0.0, 0.008, 0.05, 0.13, 0.25])  # um^2 at -1.12 .. 1.12 um


def d4_design(n, seed, convention="symmetric"):
    """c (um^2), true axes z_ax (um), starts z_sub (um), k* for n nodes; axes off-plane on both sides."""
    rng = np.random.default_rng(seed)
    k = rng.integers(-5, 15, n)
    u = rng.uniform(0.03, 0.11, n) * np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    z_ax = k * D4.DZ + u
    z_sub = z_ax + rng.uniform(-0.04, 0.04, n)
    c = rng.uniform(0.005, 0.05, n)
    return c, z_ax, z_sub, k


def d4_cal(**kw):
    return D4.cfg_with(calibration=kw)


def test_d4_calibration_set_edges():
    """O10.1: the two sets by Allen's diameter, edges included."""
    cfg = D4.cfg_with()
    cases = [(0.6, "near"), (0.6 + 1e-9, "far"), (0.8, "far"), (0.8 + 1e-9, ""), (0.0, ""), (-0.1, ""),
             (NAN, ""), (float("inf"), ""), (2 * 0.3, "near"), (2 * 0.4, "far"), (1e-12, "near")]
    for d, want in cases:
        assert CAL.calibration_set(d, cfg) == want, (d, want)


@pytest.mark.parametrize("thin, band", [(0.0, (0.6, 0.8)), (-0.1, (0.6, 0.8)), (0.7, (0.6, 0.8)),
                                        (0.6, (0.8, 0.8)), (0.6, (0.8, 0.7)), (0.6, (0.6,)),
                                        (0.6, (0.6, 0.7, 0.8)), (0.6, (0.6, float("inf"))),
                                        (float("nan"), (0.6, 0.8))])
def test_d4_validate_refuses_bad_set_bands(thin, band):
    """O10.2: validate() keeps 0 < D_thin <= a < b, finite, two band values."""
    with pytest.raises((ValueError, TypeError)):
        D4.cfg_with(calibration=dict(node_allen_diameter_max_um=thin, far_field_allen_diameter_um=band))


def test_d4_validate_accepts_touching_sets():
    D4.cfg_with(calibration=dict(node_allen_diameter_max_um=0.6, far_field_allen_diameter_um=(0.6, 0.8)))
    D4.cfg_with(calibration=dict(node_allen_diameter_max_um=0.5, far_field_allen_diameter_um=(0.6, 0.8)))


def test_d4_criteria_reasons_order_bounds_and_nan():
    """O10.3: Allen's diameter, then not_flat by the set's bound (neither set: the near bound), then not_faint
    only with node_alpha_check; a NaN fails its criterion; bounds inclusive."""
    on = d4_cal(node_alpha_check=True)
    r = math.radians
    assert CAL.criteria_reasons(r(20), 0.9, NAN, on) == ["allen_diameter_missing", "not_flat", "not_faint"]
    assert CAL.criteria_reasons(r(20), 0.9, 1.0, on) == ["allen_diameter_out", "not_flat", "not_faint"]
    assert CAL.criteria_reasons(r(12), 0.1, 1.0, on) == ["allen_diameter_out", "not_flat"]   # near bound 10 deg
    assert CAL.criteria_reasons(r(12), 0.1, 0.7, on) == []                                    # far bound 15 deg
    assert CAL.criteria_reasons(r(12), 0.1, 0.4, on) == ["not_flat"]
    assert CAL.criteria_reasons(r(16), 0.1, 0.7, on) == ["not_flat"]
    assert CAL.criteria_reasons(r(10), 0.5, 0.4, on) == []                                    # both bounds inclusive
    assert CAL.criteria_reasons(r(15), 0.1, 0.7, on) == []
    assert CAL.criteria_reasons(NAN, 0.1, 0.4, on) == ["not_flat"]
    assert CAL.criteria_reasons(0.0, NAN, 0.4, on) == ["not_faint"]
    off = D4.cfg_with()
    assert CAL.criteria_reasons(0.0, 5.0, 0.4, off) == []                                     # D-094: no alpha test
    assert CAL.criteria_reasons(0.0, NAN, 0.4, off) == []


def test_d4_selection_flags_default_and_switch():
    """O10.4 (D-094, D-096)."""
    base = ("crossing", "stack_edge", "bbar_few", "profile_nan")
    assert tuple(CAL.selection_flags(D4.cfg_with())) == base
    assert tuple(CAL.selection_flags(d4_cal(node_faint_flag_check=True))) == ("faint",) + base


def test_d4_dip_statistics_sampled_gaussian():
    """O10.5: omega = s^2 to 1e-12 relative (the spec's figure: the core fit runs on the unsmoothed
    exact Gaussian, so only the solver's round-off remains); A_W = a s sqrt(2 pi) and V_W = s^2 to 1e-9
    with a window of 10 s (the Riemann sums of a Gaussian sampled at s / step = 1.75 err by
    ~exp(-2 pi^2 1.75^2) ~ 1e-26, the window's truncation by exp(-50))."""
    s, a, vc, B = 0.2, 30.0, 0.05, 200.0
    cfg = d4_cal(window_half_um=10 * s)
    step = cfg.measure.profile_step_um
    n = int(round(3.0 / step))
    v = np.arange(-n, n + 1) * step
    I = B - D4.gauss_dip(v, a, vc, s)
    st = CAL.dip_statistics(v, I, B, cfg.measure, cfg.calibration)
    assert st["status"] == "ok"
    assert_allclose(st["omega_um2"], s ** 2, rtol=1e-12)
    assert_allclose(st["area_gl_um"], a * s * math.sqrt(2 * math.pi), rtol=1e-9)
    assert_allclose(st["vw_um2"], s ** 2, rtol=1e-9)


def test_d4_dip_statistics_refusals():
    """O10.5: a flat profile is no_dip; a dip wider than the profile has no half-depth width (no_fwhm)."""
    cfg = D4.cfg_with()
    step = cfg.measure.profile_step_um
    n = int(round(3.0 / step))
    v = np.arange(-n, n + 1) * step
    assert CAL.dip_statistics(v, np.full(v.size, 200.0), 200.0, cfg.measure, cfg.calibration)["status"] == "no_dip"
    wide = 200.0 - D4.gauss_dip(v, 30.0, 0.0, 6.0)        # half depth reached at +-7 um, outside +-3 um
    assert CAL.dip_statistics(v, wide, 200.0, cfg.measure, cfg.calibration)["status"] == "no_fwhm"


@pytest.mark.parametrize("convention, interp", [("symmetric", "cubic"), ("symmetric", "linear"),
                                                ("min_origin", "cubic"), ("min_origin", "linear")])
def test_d4_fit_growth_recovers_exact_eq4_data(convention, interp):
    """O10.6: exact Eq. 4 data of the fitted family (knots m dz, M = 4) -> c_i, z_ax,i, the knots and the
    curve to 1e-9 (the spec's tolerance); the knots are m dz (symmetric) or (m - 4) dz (min_origin)."""
    G = D4_G_SYM if convention == "symmetric" else D4_G_MIN
    c, z_ax, z_sub, k = d4_design(8, 11)
    scans = D4.eq4_scans(G, c, z_ax, z_sub, k, interp, convention)
    cfg = d4_cal(origin_convention=convention, growth_interp=interp)
    g = CAL.fit_growth(scans, cfg)
    want_knots = np.arange(5) * D4.DZ if convention == "symmetric" else (np.arange(9) - 4) * D4.DZ
    assert_allclose(g.knots_um, want_knots, atol=1e-12)
    assert_allclose(g.g_um2, G, atol=1e-9)
    assert_allclose(g.c_um2, c, atol=1e-9)
    assert_allclose(g.z_ax_um, z_ax, atol=1e-9)
    d = np.linspace(-1.0, 1.0, 41)
    assert_allclose(g.growth(d), D4.ref_growth(d, G, D4.DZ, interp, convention), atol=1e-9)


@pytest.mark.parametrize("convention, interp", [("symmetric", "cubic"), ("symmetric", "linear"),
                                                ("min_origin", "linear")])
def test_d4_fit_growth_matches_independent_minimum(convention, interp):
    """O10.7: on noisy data (SD 0.003 um^2, 10 % NaN planes) the fit is the minimum of Eq. 4: its cost is no
    larger than an independent minimisation's (numerical Jacobian, tolerances 1e-15) by more than 1e-9
    relative, and the parameters agree to 2e-6 (both solvers stop on relative cost changes of 1e-12..1e-15,
    which bound the parameters to ~sqrt(1e-12) x their scale, 1e-6 for um^2 and um)."""
    G = D4_G_SYM if convention == "symmetric" else D4_G_MIN
    c, z_ax, z_sub, k = d4_design(8, 12)
    scans = D4.eq4_scans(G, c, z_ax, z_sub, k, interp, convention, noise_sd=0.003, nan_frac=0.1,
                         rng=np.random.default_rng(5))
    cfg = d4_cal(origin_convention=convention, growth_interp=interp)
    g = CAL.fit_growth(scans, cfg)
    ref = D4.ref_fit_eq4(scans, 4, D4.DZ, interp, convention, cfg.calibration.z_ax_bound_um)
    assert list(g.node_ids) == ref["node_ids"]
    assert g.cost <= ref["cost"] * (1 + 1e-9) + 1e-18
    ok = np.isfinite(g.g_um2)
    assert_allclose(g.g_um2[ok], ref["G"][ok], atol=2e-6)
    assert_allclose(g.c_um2, ref["c"], atol=2e-6)
    assert_allclose(g.z_ax_um, ref["z"], atol=2e-6)


def test_d4_fit_growth_invariances():
    """O10.8: a constant on every omega moves only the c_i; a common depth shift u of the planes and the
    starts moves only the z_ax,i (by u); symmetric data under min_origin give a symmetric curve. 1e-8: the
    two fits are the same problem in shifted coordinates, solved to the 1e-12 tolerances (1e-9 on exact
    data, O10.6), with the noise's conditioning."""
    c, z_ax, z_sub, k = d4_design(8, 13)
    rng = np.random.default_rng(6)
    scans = D4.eq4_scans(D4_G_SYM, c, z_ax, z_sub, k, noise_sd=0.002, rng=rng)
    cfg = D4.cfg_with()
    g0 = CAL.fit_growth(scans, cfg)
    shifted = [dataclasses.replace(s, omega_um2=s.omega_um2 + 0.05) for s in scans]
    g1 = CAL.fit_growth(shifted, cfg)
    assert_allclose(g1.c_um2, g0.c_um2 + 0.05, atol=1e-8)
    assert_allclose(g1.g_um2, g0.g_um2, atol=1e-8)
    assert_allclose(g1.z_ax_um, g0.z_ax_um, atol=1e-8)
    u = 0.1
    moved = [dataclasses.replace(s, z_um=s.z_um + u, z_sub_um=s.z_sub_um + u) for s in scans]
    g2 = CAL.fit_growth(moved, cfg)
    assert_allclose(g2.z_ax_um, g0.z_ax_um + u, atol=1e-8)
    assert_allclose(g2.g_um2, g0.g_um2, atol=1e-8)
    assert_allclose(g2.c_um2, g0.c_um2, atol=1e-8)
    # symmetric data under min_origin: an even law, axes in +-u pairs
    G_even = np.r_[D4_G_SYM[::-1], D4_G_SYM[1:]]
    n = 8
    kk = np.repeat(np.arange(4) * 5, 2)
    uu = np.tile([0.07, -0.07], 4)
    za = kk * D4.DZ + uu
    sym = D4.eq4_scans(G_even, np.full(n, 0.02), za, za, kk, "linear", "min_origin")
    gm = CAL.fit_growth(sym, d4_cal(origin_convention="min_origin", growth_interp="linear"))
    assert_allclose(gm.g_um2, gm.g_um2[::-1], atol=1e-8)


def test_d4_fit_growth_knot_count_follows_offsets_planes():
    """O10.6: M = ceil((P + 1/2) dz / h) with P = offsets_planes (3) -> five knots 0 .. 1.12 um, whatever
    the planes that happen to hold a finite width; a knot without an observation in either adjacent
    interval is NaN. Scans of +-3 planes whose +-3 planes are all NaN (axes on the planes) leave the
    interval [0.84, 1.12] empty, so the 1.12 knot is NaN."""
    n = 6
    c = np.linspace(0.01, 0.03, n)
    k = np.arange(n) * 4
    z = k * D4.DZ
    scans = D4.eq4_scans(D4_G_SYM, c, z, z, k)
    scans = [dataclasses.replace(s, omega_um2=np.where(np.abs(s.ks - s.k_star) == 3, np.nan, s.omega_um2))
             for s in scans]
    g = CAL.fit_growth(scans, D4.cfg_with())
    assert_allclose(g.knots_um, np.arange(5) * D4.DZ, atol=1e-12)
    assert np.isnan(g.g_um2[4])


def test_d4_fit_growth_refuses_one_node():
    c, z_ax, z_sub, k = d4_design(2, 14)
    scans = D4.eq4_scans(D4_G_SYM, c, z_ax, z_sub, k)
    scans[1] = dataclasses.replace(scans[1], omega_um2=np.where(np.arange(7) < 5, np.nan, scans[1].omega_um2))
    with pytest.raises(ValueError):
        CAL.fit_growth(scans, D4.cfg_with())


@pytest.mark.parametrize("convention, interp, n", [("symmetric", "cubic", 30), ("symmetric", "linear", 30),
                                                   ("symmetric", "cubic", 100), ("min_origin", "linear", 30)])
def test_d4_sparse_and_dense_jacobians_agree(convention, interp, n):
    """O10.9 (D-119): the sparse default and the dense path solve the same problem -- symmetric: knots and
    c_i within 1e-9 um^2, z_ax within 1e-8 um, the anchor within 1e-9 um; min_origin: knots within 1e-5
    um^2 and costs within 1e-10 relative (the spec's figures)."""
    G = D4_G_SYM if convention == "symmetric" else D4_G_MIN
    c, z_ax, z_sub, k = d4_design(n, 20 + n)
    scans = D4.eq4_scans(G, c, z_ax, z_sub, k, interp, convention, noise_sd=0.003, nan_frac=0.1,
                         rng=np.random.default_rng(n))
    sp = d4_cal(origin_convention=convention, growth_interp=interp)
    de = d4_cal(origin_convention=convention, growth_interp=interp, growth_jacobian="dense")
    assert sp.calibration.growth_jacobian == "sparse"
    gs, gd = CAL.fit_growth(scans, sp), CAL.fit_growth(scans, de)
    ok = np.isfinite(gs.g_um2)
    assert_array_equal(ok, np.isfinite(gd.g_um2))
    if convention == "symmetric":
        assert_allclose(gs.g_um2[ok], gd.g_um2[ok], atol=1e-9)
        assert_allclose(gs.c_um2, gd.c_um2, atol=1e-9)
        assert_allclose(gs.z_ax_um, gd.z_ax_um, atol=1e-8)
        assert_allclose(CAL.anchor_value(gs, sp), CAL.anchor_value(gd, de), atol=1e-9)
    else:
        assert_allclose(gs.g_um2[ok], gd.g_um2[ok], atol=1e-5)
        assert_allclose(gs.cost, gd.cost, rtol=1e-10)


@pytest.mark.parametrize("field, value", [("growth_jacobian", "lsmr"), ("knot_step_um", 0.14)])
def test_d4_validate_refuses_growth_settings(field, value):
    with pytest.raises(ValueError):
        d4_cal(**{field: value})


def d4_table_growth(cfg, interp="cubic"):
    """A growth fitted to exact Eq. 4 data generated from the default table's sigma at the knots."""
    s0 = cfg.renderer.sigma_r0_um
    sig = D4.ref_sigma_r(np.arange(5) * D4.DZ, cfg.renderer.kernel_table_delta_um, cfg.renderer.kernel_table_sigma_um,
                         cfg.renderer.kernel_continuation_slope)
    G = sig ** 2 - s0 ** 2
    c, z_ax, z_sub, k = d4_design(8, 30)
    return CAL.fit_growth(D4.eq4_scans(G, c, z_ax, z_sub, k, interp), cfg), sig


def test_d4_kernel_from_growth_round_trip_and_cut():
    """O10.10: a table on the knots -> its growth -> exact data -> fit -> the table to 1e-10; sigma(0) is the
    configured sigma_r0_um; the cut keeps the knots <= max_delta_um; the result validates."""
    cfg = D4.cfg_with()
    g, sig = d4_table_growth(cfg)
    k_all = CAL.kernel_from_growth(g, cfg.renderer)
    assert_allclose(k_all.kernel_table_delta_um, np.arange(5) * D4.DZ, atol=1e-12)
    assert_allclose(k_all.kernel_table_sigma_um, sig, atol=1e-10)
    assert k_all.kernel_table_sigma_um[0] == cfg.renderer.sigma_r0_um == k_all.sigma_r0_um
    assert k_all.kernel_family == "gaussian_table"
    assert k_all.kernel_continuation == cfg.renderer.kernel_continuation
    k_cut = CAL.kernel_from_growth(g, cfg.renderer, max_delta_um=cfg.calibration.table_max_delta_um)
    assert_allclose(k_cut.kernel_table_delta_um, [0.0, 0.28, 0.56, 0.84], atol=1e-12)
    assert_allclose(k_cut.kernel_table_sigma_um, sig[:4], atol=1e-10)
    dataclasses.replace(cfg, renderer=k_cut).validate()
    k_s0 = CAL.kernel_from_growth(g, cfg.renderer, sigma_r0_um=0.1)
    assert_allclose(k_s0.kernel_table_sigma_um, np.sqrt(0.1 ** 2 + g.g_um2), atol=1e-12)
    assert k_s0.sigma_r0_um == 0.1


def test_d4_kernel_from_growth_refusals():
    """O10.10: min_origin refused; sigma_r(0)^2 + G <= 0 or NaN at a kept knot refused; a NaN knot beyond the
    cut does not matter."""
    cfg = D4.cfg_with()
    g, _ = d4_table_growth(cfg)
    s0 = cfg.renderer.sigma_r0_um
    with pytest.raises(ValueError):
        CAL.kernel_from_growth(dataclasses.replace(g, convention="min_origin"), cfg.renderer)
    low = g.g_um2.copy()
    low[1] = -s0 ** 2
    with pytest.raises(ValueError):
        CAL.kernel_from_growth(dataclasses.replace(g, g_um2=low), cfg.renderer)
    nan2 = g.g_um2.copy()
    nan2[2] = np.nan
    with pytest.raises(ValueError):
        CAL.kernel_from_growth(dataclasses.replace(g, g_um2=nan2), cfg.renderer)
    nan4 = g.g_um2.copy()
    nan4[4] = np.nan
    k = CAL.kernel_from_growth(dataclasses.replace(g, g_um2=nan4), cfg.renderer, max_delta_um=0.84)
    assert len(k.kernel_table_sigma_um) == 4


def test_d4_anchor_value_and_anchor_from_growth():
    """O10.11 (D-117.4)."""
    cfg = D4.cfg_with()
    g, sig = d4_table_growth(cfg)
    s0 = cfg.renderer.sigma_r0_um
    assert_allclose(CAL.anchor_value(g, cfg), CAL.anchor_from_growth(g, cfg), rtol=1e-15)
    assert_allclose(CAL.anchor_value(g, cfg), sig[3], atol=1e-10)
    low = g.g_um2.copy()
    low[1] = -s0 ** 2 - 0.001
    gl = dataclasses.replace(g, g_um2=low)
    with pytest.raises(ValueError):
        CAL.anchor_from_growth(gl, cfg)
    assert_allclose(CAL.anchor_value(gl, cfg), math.sqrt(s0 ** 2 + low[3]), rtol=1e-15)
    bad = g.g_um2.copy()
    bad[3] = -s0 ** 2 - 0.001
    with pytest.raises(ValueError):
        CAL.anchor_value(dataclasses.replace(g, g_um2=bad), cfg)


def d4_plane(n=64, p=0.1144, b=(150.0, 2.0, -1.5), noise=0.0, seed=0, debris=0.0):
    frame = D4_CropFrame(100, 50, 0, p)
    rows, cols = np.mgrid[0:n, 0:n]
    x, y = (100 + cols) * p, (50 + rows) * p
    cx, cy = x.mean(), y.mean()
    img = b[0] + b[1] * (x - cx) + b[2] * (y - cy)
    rng = np.random.default_rng(seed)
    if noise:
        img = img + rng.normal(0.0, noise, img.shape)
    if debris:
        hit = rng.random(img.shape) < debris
        img = np.where(hit, img + 80.0, img)
    return img, frame, cx, cy


def test_d4_planar_background_known_answer_and_sd():
    """O10.12-13: exact plane recovered (1e-9 gl: one lstsq of 4096 exact rows); sd within 5 % of 2 gl on
    ~4000 pixels of SD 2 (the MAD's own SE is ~1.6 % there), b the same with and without return_sd; 1 %
    debris trimmed (b0 within 5 x its SE 2/sqrt(N) = 0.16 gl); too few far pixels: not ok, sd NaN."""
    img, frame, cx, cy = d4_plane()
    far = np.ones(img.shape, dtype=bool)
    far[30:34, :] = False
    b, n, ok = CS.planar_background(img, far, frame, cx, cy, 3.0, 0.05)
    assert ok and n == int(far.sum())
    assert_allclose(b, [150.0, 2.0, -1.5], atol=1e-9)
    img2, frame, cx, cy = d4_plane(noise=2.0, seed=3)
    b1 = CS.planar_background(img2, far, frame, cx, cy, 3.0, 0.05)
    b2 = CS.planar_background(img2, far, frame, cx, cy, 3.0, 0.05, return_sd=True)
    assert len(b1) == 3 and len(b2) == 4
    assert_array_equal(b1[0], b2[0])
    assert abs(b2[3] - 2.0) <= 0.05 * 2.0
    img3, frame, cx, cy = d4_plane(noise=2.0, seed=4, debris=0.01)
    b3 = CS.planar_background(img3, far, frame, cx, cy, 3.0, 0.05, return_sd=True)
    assert abs(b3[0][0] - 150.0) <= 5 * 2.0 / math.sqrt(far.sum())
    few = np.zeros(img.shape, dtype=bool)
    few[:2, :10] = True                                  # 20 px of 4096 < 5 %
    b4 = CS.planar_background(img, few, frame, cx, cy, 3.0, 0.05, return_sd=True)
    assert (not b4[2]) and np.all(np.isnan(b4[0])) and math.isnan(b4[3])


def test_d4_gof_poor_rule():
    """O10.13: RMS > kappa_g x SD with both finite and SD > 0, and the switch on."""
    cfg = D4.cfg_with()
    kg = cfg.calibration.gof_kappa
    assert kg == 1.2
    assert CAL.gof_poor(kg * 2.0 + 1e-6, 2.0, cfg) is True
    assert CAL.gof_poor(kg * 2.0 - 1e-6, 2.0, cfg) is False
    assert CAL.gof_poor(kg * 2.0, 2.0, cfg) is False           # strictly above
    assert CAL.gof_poor(NAN, 2.0, cfg) is False
    assert CAL.gof_poor(10.0, NAN, cfg) is False
    assert CAL.gof_poor(10.0, 0.0, cfg) is False
    assert CAL.gof_poor(10.0, 2.0, d4_cal(gof_check=False)) is False


def d4_line_branch(theta_deg, phi_deg, n=41, step=0.5, reverse=False):
    th, ph = math.radians(theta_deg), math.radians(phi_deg)
    t = np.array([math.cos(ph) * math.cos(th), math.cos(ph) * math.sin(th), math.sin(ph)])
    s = np.arange(n) * step
    xyz = np.array([10.0, 10.0, 5.0]) + s[:, None] * t[None, :]
    if reverse:
        xyz = xyz[::-1]
    return D4_Branch(np.arange(1, n + 1), np.full(n, 3), xyz, np.full(n, 0.25), s)


@pytest.mark.parametrize("theta, phi", [(30.0, 20.0), (-120.0, 5.0), (90.0, 0.0), (170.0, 60.0)])
def test_d4_trace_direction_known_line(theta, phi):
    """O10.14: on a straight trace the TLS direction is the line's, oriented first -> last; reversing the
    trace turns the heading by pi (1e-9 rad: an eigenvector of an exact rank-one scatter)."""
    b = d4_line_branch(theta, phi)
    th, ph = D4_CSR.trace_direction(b, 20, 4.0)
    assert_allclose(ph, math.radians(phi), atol=1e-9)
    assert_allclose(math.cos(th - math.radians(theta)), 1.0, atol=1e-9)
    th2, ph2 = D4_CSR.trace_direction(d4_line_branch(theta, phi, reverse=True), 20, 4.0)
    assert_allclose(math.cos(th2 - math.radians(theta)), -1.0, atol=1e-9)
    assert_allclose(ph2, math.radians(phi), atol=1e-9)


def test_d4_trace_direction_coinciding_points():
    n = 5
    b = D4_Branch(np.arange(1, n + 1), np.full(n, 3), np.tile([1.0, 2.0, 3.0], (n, 1)), np.full(n, 0.2),
                  np.zeros(n))
    with pytest.raises(ValueError):
        D4_CSR.trace_direction(b, 2, 4.0)


def test_d4_typical_background_medians_over_rows_in_S():
    """O10.14: the medians of B_bar and mu_hat over the rows in S (converged, no flag of S), finite positive
    values only; NaN without such rows."""
    cfg = D4.cfg_with()
    rows = [dict(fit_status="converged", flags="", B_bar="150", mu_hat_per_um="0.2"),
            dict(fit_status="converged", flags="dark;faint", B_bar="170", mu_hat_per_um="0.4"),   # labels only
            dict(fit_status="converged", flags="crossing", B_bar="999", mu_hat_per_um="9"),        # out of S
            dict(fit_status="at_bound", flags="", B_bar="999", mu_hat_per_um="9"),                 # out of S
            dict(fit_status="converged", flags="", B_bar="nan", mu_hat_per_um="0.3"),
            dict(fit_status="converged", flags="", B_bar="160", mu_hat_per_um="-1")]
    B, mu = D4_CSR.typical_background(rows, cfg)
    assert_allclose(B, np.median([150.0, 170.0, 160.0]))
    assert_allclose(mu, np.median([0.2, 0.4, 0.3]))
    B2, mu2 = D4_CSR.typical_background(rows[2:4], cfg)
    assert math.isnan(B2) and math.isnan(mu2)


def d4_ref_choose(records, which, sp):
    ok = [r for r in records if r["set"] == which and not r["reason"]]
    if which == "far":
        key = lambda r: (-r["n_fit_usable"], -r["n_pixels_k"], r["node_id"])   # noqa: E731
    else:
        key = lambda r: (-r["n_pixels_k"], r["node_id"])                        # noqa: E731
    kept = []
    for r in sorted(ok, key=key):
        if all(not (q["stretch"] == r["stretch"] and abs(q["s_um"] - r["s_um"]) < sp) for q in kept):
            kept.append(r)
    return [r["node_id"] for r in kept]


@pytest.mark.parametrize("which", ["near", "far"])
def test_d4_choose_greedy_spacing_and_rank(which):
    """O10.14: choose equals the greedy rule written from the spec on random records; invariants: only
    passing records of the set, spacing >= the set's along each stretch, every passing record left out
    lies within the spacing of a kept one ranked before it."""
    cfg = D4.cfg_with()
    sp = getattr(cfg.calibration, which + "_search_spacing_um")
    rng = np.random.default_rng(7)
    recs = []
    for i in range(120):
        recs.append(dict(set=which if rng.random() < 0.9 else ("far" if which == "near" else "near"),
                         node_id=int(1000 + i), stretch=int(rng.integers(0, 4)), s_um=float(rng.uniform(0, 40)),
                         reason="" if rng.random() < 0.7 else "few_planes", n_pixels_k=int(rng.integers(500, 900)),
                         n_fit_usable=int(rng.integers(6, 10))))
    got = D4_CSR.choose(recs, cfg, which)
    assert got == d4_ref_choose(recs, which, sp)
    by_id = {r["node_id"]: r for r in recs}
    for a in got:
        assert by_id[a]["set"] == which and by_id[a]["reason"] == ""
    for i, a in enumerate(got):
        for b in got[:i]:
            if by_id[a]["stretch"] == by_id[b]["stretch"]:
                assert abs(by_id[a]["s_um"] - by_id[b]["s_um"]) >= sp
    with pytest.raises(ValueError):
        D4_CSR.choose(recs, cfg, "middle")


def d4_swc(n=6):
    from allen_diameter.loading.swc_io import SWC
    ids = np.arange(1, n + 1)
    return SWC(ids, np.r_[1, np.full(n - 1, 3)], np.column_stack([np.arange(n) * 1.0, np.zeros(n), np.zeros(n)]),
               np.full(n, 0.3), np.r_[-1, ids[:-1]], ["# a comment\n"], list(range(n)), "")


def test_d4_swc_digest_and_search_config():
    """O10.15: the digest changes with each of the five arrays and not with lines or path; the key's settings
    are the four sections minus the fit-only fields, strict JSON, NaN as 'nan'."""
    a = d4_swc()
    d0 = D4_CSR.swc_digest(a)
    for name in ("ids", "types", "parent", "xyz", "radius"):
        b = dataclasses.replace(a)
        arr = np.array(getattr(a, name), copy=True)
        arr.flat[-1] = arr.flat[-1] + 1
        setattr(b, name, arr)
        assert D4_CSR.swc_digest(b) != d0, name
    b = dataclasses.replace(a, lines=["# other\n"], path="/elsewhere.swc")
    assert D4_CSR.swc_digest(b) == d0
    cfg = D4.cfg_with()
    sc = D4_CSR.search_config(cfg)
    assert set(sc) == {"acquisition", "calibration", "measure", "renderer"}
    for f in ("growth_jacobian", "bootstrap_unit", "bootstrap_cell_min", "pool_weighting", "cell_outlier_removal",
              "cell_outlier_alpha"):
        assert f not in sc["calibration"], f
    assert "isolation_depth_gl" in sc["calibration"] and "far_offsets_planes" in sc["calibration"]
    json.dumps(sc, allow_nan=False)
    assert isinstance(sc["calibration"]["stripe_half_widths_um"], list)
    sc_nan = D4_CSR.search_config(dataclasses.replace(cfg, renderer=dataclasses.replace(cfg.renderer,
                                                                                        ray_post_sigma_um=NAN)))
    assert sc_nan["renderer"]["ray_post_sigma_um"] == "nan"
    for f, v in (("growth_jacobian", "dense"), ("bootstrap_unit", "node"), ("bootstrap_cell_min", 5),
                 ("pool_weighting", "node"), ("cell_outlier_removal", False), ("cell_outlier_alpha", 0.01)):
        assert D4_CSR.search_config(d4_cal(**{f: v})) == sc, f
    assert D4_CSR.search_config(d4_cal(isolation_depth_gl=1.0)) != sc


def test_d4_reuse_search_rules(tmp_path):
    """O10.15: reuse only with the CSV and its record, the CSV unchanged and the key equal; the why strings."""
    import calibrate_kernel as CK
    cfg = D4.cfg_with()
    swc = d4_swc()
    key = CK.search_key(swc, cfg)
    d = str(tmp_path)
    ids, rec, why = CK.reuse_search(d, "far", 7, key)
    assert ids is None and why.startswith("no search file")
    rows = [dict(set="far", node_id=3, chosen=True), dict(set="far", node_id=4, chosen=False),
            dict(set="far", node_id=5, chosen=True)]
    D4_table_io.write_rows(rows, CK.search_name(d, "far", 7))
    ids, rec, why = CK.reuse_search(d, "far", 7, key)
    assert ids is None and why.startswith("no record")
    CK.write_search_record(d, "far", 7, key, 150.0, 0.25, None, 3, 2)
    ids, rec, why = CK.reuse_search(d, "far", 7, key)
    assert why == "" and sorted(ids) == [3, 5]
    other = CK.search_key(dataclasses.replace(swc, radius=swc.radius * 1.01), cfg)
    assert CK.reuse_search(d, "far", 7, other)[2] == "another trace"
    other = CK.search_key(swc, d4_cal(isolation_depth_gl=1.0))
    assert CK.reuse_search(d, "far", 7, other)[2] == "other settings: calibration"
    with open(CK.search_name(d, "far", 7), "a") as f:
        f.write("far,6,,,,,,,,,,,,,,False\n")
    assert CK.reuse_search(d, "far", 7, key)[2] == "the CSV changed since its record"


def d4_painted_patch(cfg, k0, a=30.0, B=180.0, noise=2.0, seed=0):
    """A patch of planes k0 - 3 .. k0 + 3 about (cx, cy) = (50, 40) um: a dendrite along x painted as a Gaussian
    dip of width 0.25 + 0.3 |delta| um and depth a under B, white pixel noise; its pixel centres within
    +-patch_half_um of the centre (10.P), frame CropFrame(left, top, 0, p)."""
    p, half = cfg.acquisition.res0_um, cfg.calibration.patch_half_um
    cx, cy = 50.0, 40.0
    left, right = math.ceil((cx - half) / p - 1e-9), math.floor((cx + half) / p + 1e-9)
    top, bottom = math.ceil((cy - half) / p - 1e-9), math.floor((cy + half) / p + 1e-9)
    rows, cols = np.mgrid[0:bottom - top + 1, 0:right - left + 1]
    y = (top + rows) * p
    rng = np.random.default_rng(seed)
    ks = k0 + np.arange(-3, 4)
    block = np.stack([B - a * np.exp(-0.5 * ((y - cy) / (0.25 + 0.3 * abs(k - k0) * D4.DZ)) ** 2)
                      + rng.normal(0.0, noise, y.shape) for k in ks])
    frame = D4_CropFrame(int(left), int(top), 0, p)
    return (block, ks, np.ones(ks.size, dtype=bool), frame), cx, cy


def test_d4_goodness_of_fit_drops_one_near_plane(monkeypatch):
    """10.G oracle (c): a near scan whose core fit is made to report an RMS far above the bound on one plane --
    that plane poor_fit with NaN omega, SE and V_W, its depth, amplitude, centre, RMS and area kept, the others
    ok; bg_sd_gl holds the pixels' noise SD (2 gl, within 10 %: the far pixels' residuals about the fitted
    plane over ~8000 pixels); fit_growth keeps the node on its other planes."""
    from allen_diameter.analysis.plane_iteration import Scene
    cfg = D4.cfg_with(calibration=dict(background_support="square", patch_half_um=6.0))
    k0 = 10
    patch, cx, cy = d4_painted_patch(cfg, k0)
    n = 41
    xs = cx - 10.0 + 0.5 * np.arange(n)
    branch = D4_Branch(np.arange(1, n + 1), np.full(n, 3), np.column_stack([xs, np.full(n, cy), np.full(n, k0 * D4.DZ)]),
                       np.full(n, 0.15), 0.5 * np.arange(n))
    node = D4.node_result(node_id=7, k_star=k0, z_sub_um=k0 * D4.DZ, cx_um=cx, cy_um=cy, theta_rad=0.0)
    clean = CAL.plane_scan(patch, node, branch, Scene.from_branch(branch), cfg)
    assert all(s == "ok" for s in clean.status), clean.status
    assert np.all(np.abs(clean.bg_sd_gl - 2.0) <= 0.2)
    real = CAL.dip_statistics
    calls = []

    def inflated(*args, **kw):
        out = dict(real(*args, **kw))
        calls.append(1)
        if len(calls) == 5:                                   # the plane k0 + 1 (planes in order)
            out["rms_gl"] = 100.0
        return out
    monkeypatch.setattr(CAL, "dip_statistics", inflated)
    sc = CAL.plane_scan(patch, node, branch, Scene.from_branch(branch), cfg)
    q = list(sc.ks).index(k0 + 1)
    assert sc.status[q] == "poor_fit"
    assert all(s == "ok" for i, s in enumerate(sc.status) if i != q)
    assert np.isnan(sc.omega_um2[q]) and np.isnan(sc.se_omega_um2[q]) and np.isnan(sc.vw_um2[q])
    for arr in (sc.depth_gl, sc.amp_gl, sc.v_c_um, sc.rms_gl, sc.area_gl_um):
        assert np.isfinite(arr[q])
    other = dataclasses.replace(clean, node_id=8)
    g = CAL.fit_growth([sc, other, dataclasses.replace(clean, node_id=9)], D4.cfg_with())
    assert 7 in list(g.node_ids) and int(np.sum(g.obs_node == list(g.node_ids).index(7))) == 6


def d4_two_dendrites_swc(n=12, step=0.6):
    """Two straight dendrite trees without a soma (roots with parent -1), 40 um apart, flat at z = 1.4 um:
    A (ids 1..n) radius 0.25 (Allen's diameter 0.5: near), B (ids n+1..2n) radius 0.35 (0.7: far)."""
    from allen_diameter.loading.swc_io import SWC
    ids = np.arange(1, 2 * n + 1)
    t = np.arange(n) * step
    xyz = np.vstack([np.column_stack([60 + t, np.zeros(n), np.full(n, 1.4)]),
                     np.column_stack([60 + t, np.full(n, 40.0), np.full(n, 1.4)])])
    parent = np.r_[-1, np.arange(1, n), -1, np.arange(n + 1, 2 * n)]
    radius = np.r_[np.full(n, 0.25), np.full(n, 0.35)]
    return SWC(ids, np.full(2 * n, 3), xyz, radius, parent, ["%d\n" % i for i in ids], list(range(2 * n)), "")


def test_d4_search_known_answer_isolated_flat_dendrites():
    """10.C oracle (a): a flat near-range dendrite with nothing beside it passes after the trace's plane and the
    pair +-1 (3 planes evaluated, 3 usable); a flat far-range one passes with every far-fit plane usable (24 of
    24); each search lists its own range's nodes only; the trace's plane is round(z / dz) = 5. The patch is cut
    to +-12 um for time: at the farthest far plane (|delta| 4.48 um, sigma_r 3.48 um) the own branch's tau-margin is
    about 6.8 um (B 150 gl, mu 0.3 /um, r 0.35 um, tau 0.3 gl: 1.98 exp(-rho^2 / 24.2) = 0.3), leaving > 3 um."""
    cfg = d4_cal(patch_half_um=12.0)
    swc = d4_two_dendrites_swc()
    near = D4_CSR.search(swc, cfg, "near", B_bar=150.0, mu=0.3)
    far = D4_CSR.search(swc, cfg, "far", B_bar=150.0, mu=0.3)
    assert [r["node_id"] for r in near] == list(range(1, 13))
    assert [r["node_id"] for r in far] == list(range(13, 25))
    for r in near:
        assert (r["reason"], r["n_fit"], r["n_fit_usable"], r["k_trace"]) == ("", 3, 3, 5), r
        assert r["n_pixels_k"] >= cfg.calibration.stripe_min_pixels and abs(r["phi_deg"]) < 1e-9
    for r in far:
        assert (r["reason"], r["n_fit"], r["n_fit_usable"], r["k_trace"]) == ("", 24, 24, 5), r
    assert set(near[0]) == set(D4_CSR.RECORD_KEYS)
    chosen = D4_CSR.choose(near, cfg, "near")
    s = {r["node_id"]: r["s_um"] for r in near}
    assert all(abs(s[a] - s[b]) >= 5.0 for i, a in enumerate(chosen) for b in chosen[:i])


def test_d4_anchor_bootstrap_se_matches_regenerated_sd():
    """10.A: on Eq. 4 data with noise (12 nodes, SD 0.01 um^2) the node bootstrap's SE of the anchor lies within
    30 % of the SD of the anchor over 40 regenerated data sets (the spec's oracle; fixed seeds; 200 resamples: the
    SD over 40 sets carries ~11 % relative error, the bootstrap's own ~5 %, so 30 % is about 2.5 combined SEs)."""
    cfg = D4.cfg_with()
    sig = D4.ref_sigma_r(np.arange(5) * D4.DZ, cfg.renderer.kernel_table_delta_um, cfg.renderer.kernel_table_sigma_um,
                         cfg.renderer.kernel_continuation_slope)
    G = sig ** 2 - cfg.renderer.sigma_r0_um ** 2
    c, z_ax, z_sub, k = d4_design(12, 70)

    def data(seed):
        return D4.eq4_scans(G, c, z_ax, z_sub, k, noise_sd=0.01, rng=np.random.default_rng(seed))
    anchors = [CAL.anchor_value(CAL.fit_growth(data(1000 + s), cfg), cfg) for s in range(40)]
    sd = float(np.std(anchors, ddof=1))
    se, n_ok, draws = CAL.bootstrap_sigma0(data(77), cfg, np.random.default_rng(78), n_resamples=200)
    assert n_ok == 200 and draws.size == 200
    assert_allclose(se, np.std(draws, ddof=1), rtol=1e-12)
    assert 0.7 * sd <= se <= 1.3 * sd, (se, sd)
