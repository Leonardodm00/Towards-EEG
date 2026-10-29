"""Smoke test for the stripe -- Block 10.S in specs/SPEC.md (D-101, 2026-10-28):
the background's support of a calibration plane is the band along the
measuring line with most pixels that no traced neighbour shades (predicted
shadow above isolation_depth_gl) and that the node's own fitted cylinder does
not shade, walked outward on each side to the first shaded column; a plane
is usable with at least stripe_min_pixels of them and stripe_min_side_um of
clear run on each side. Flatness is per set (node_phi_max_deg for the near
set, node_phi_max_deg_far for the far set; D-101.1).

Checks
    test_known_answer   the own tube along x with one parallel neighbour at
                        lateral distance D on the +v side, nothing on the
                        other: L+ = D - rho_tau(neighbour) and v_own =
                        rho_tau(own) within one pixel, L- the patch's edge,
                        W* the widest half-width; the scene rotated by 35 deg
                        gives the same pixel count within 3 %
    test_reference      shadow_margin against a brute-force scan of
                        shadow_depth on a fine rho grid (the first rho below
                        tau); shadow_masks against the per-pixel evaluation
                        of the deepest shadow over the segments on a small
                        patch; shadow_depths equals shadow_depth element-wise
    test_convergence    not a discretisation: skipped
    test_invariants     the stripe lies in the band, holds no shaded pixel,
                        N(W*) >= N(W) for every W; neighbours hugging both
                        sides within 3 um give side_short, within 1.2 um
                        too_few; "square"
                        reproduces far_pixel_mask bit for bit and the old
                        node verdict
    test_contract       shapes and dtypes; the info keys; criteria_reasons at
                        12 deg (near fails, far passes) and 16 deg (far
                        fails); patch_planes' six-tuple and PlaneScan.stripe
                        through the scans JSON; stripe_reasons' names
    test_determinism    two computations are identical
    test_edge_cases     no segment but the own branch: the band to the
                        patch's edges minus the own cylinder; a non-finite
                        d_hat falls back to Allen's radius; nothing shaded at
                        a huge tau: the whole band
    test_recheck        2026-10-28, after the integration tester's report and
                        the run on the pilot's reconstruction: (F1) a soma
                        beside the node -- the verdict before the fetch equals
                        the scan's stripe at k*, the soma at the stage's blur;
                        (F2) the trace 0.5 um off the fitted centre -- the own
                        cylinder covers the fitted axis (column 0 own-shaded,
                        no stripe pixel within d_hat / 2 of the fitted axis);
                        the vectorized margins equal the scalar search of
                        2026-10-28 (verbatim copy) to 1e-9 um, and the windowed
                        masks equal the full-patch masks of the same day
                        (verbatim copy) pixel for pixel on random scenes with
                        deep segments whose margins cover the whole patch;
                        2026-10-29, the pre-cull of rows that cannot reach the
                        crop: at a tiny tau each margin is its bracket's end r
                        + 6 sigma, and segments 0.5 sigma inside that reach
                        from outside the crop still shade its edge (equal to
                        the full-patch masks), while those 0.5 sigma beyond it
                        shade nothing

Run
    cd "Passive Features/Diameter Re-measurement"
    python tests/smoke/test_smoke_stripe.py

Keep this file pure ASCII (hpc-python-compat).
"""
from __future__ import annotations

import dataclasses
import importlib.metadata
import json
import math
import os
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
SRC = WS / "src"
for _p in (SRC, HERE, WS / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from allen_diameter.analysis import calibration as CAL  # noqa: E402
from allen_diameter.analysis import calibration_scene as CS  # noqa: E402
from allen_diameter.analysis.node_pipeline import Branch, NodeResult  # noqa: E402
from allen_diameter.analysis.plane_iteration import Scene  # noqa: E402
from allen_diameter.config import default_config  # noqa: E402
from allen_diameter.loading import calibration_io  # noqa: E402
from allen_diameter.model.kernel import sigma_r  # noqa: E402
from allen_image_io import CropFrame  # noqa: E402

SEED = 20261028
REPORT_PACKAGES = ("numpy", "scipy")
NAN = float("nan")
HALF = 10.0                      # um: the patch of these checks
B_BAR, MU = 200.0, 1.0


# --------------------------------------------------------------- helpers ---

def cfg_with(**cal):
    cfg = default_config()
    return dataclasses.replace(cfg, calibration=dataclasses.replace(cfg.calibration, **cal))


def pilot(cx=0.0, cy=0.0, z=0.0, theta=0.0, d=0.4, mu=MU, B_bar=B_BAR, z_sub=0.0, k_star=0):
    """A pilot NodeResult with a centre, a heading, a sharpest plane and finite B_bar and mu."""
    return NodeResult(12, 3, cx, cy, z, 0.0, "", NAN, NAN, k_star, z_sub, cx, cy, z, theta, 0.0, False, False,
                      B_bar, "block_masked", d, mu, 0.0, mu * d, "converged", (),
                      None, "gradient_energy", np.empty(0, dtype=int), np.empty(0), np.empty(0), -1)


def rot(xy, theta):
    c, s = math.cos(theta), math.sin(theta)
    xy = np.asarray(xy, dtype=float)
    return np.c_[c * xy[:, 0] - s * xy[:, 1], s * xy[:, 0] + c * xy[:, 1]]


def scene_of(neighbours=((3.0, 0.3),), theta=0.0, r_own=0.2, z_other=0.0, with_soma=False):
    """(own Branch, Scene): the own branch through the origin along the heading theta (ids 100..124), one parallel
    neighbour per (lateral offset along +v, radius) (ids 200 + 100 m ...); every segment at depth 0 but z_other."""
    x = np.linspace(-14, 14, 29)
    own_xy = rot(np.c_[x, np.zeros(x.size)], theta)
    own = Branch.from_points(np.c_[own_xy, np.zeros(x.size)], r_own, ids=np.arange(100, 100 + x.size))
    parts = [Scene.from_branch(own)]
    for m, (v_off, r) in enumerate(neighbours):
        xy = rot(np.c_[np.linspace(-10, 10, 5), np.full(5, v_off)], theta)
        br = Branch.from_points(np.c_[xy, np.full(5, z_other)], r, ids=np.arange(200 + 100 * m, 205 + 100 * m))
        parts.append(Scene.from_branch(br))
    soma = np.array([[0.0, -9.0, 2.0]]) if with_soma else np.empty((0, 3))
    return own, Scene(np.vstack([q.seg_xyz for q in parts]), np.vstack([q.seg_radius for q in parts]),
                      np.vstack([q.seg_ids for q in parts]), soma)


def frame_of(cfg, half=HALF):
    return CS.patch_frame(0.0, 0.0, half, cfg.acquisition.res0_um)


def one_plane(cfg, own, scene, node, half=HALF):
    """(support (H, W), info, nb (H, W), ow (H, W), frame) of plane 0 for a node at the origin."""
    shape, frame = frame_of(cfg, half)
    support, stripe = CAL.support_masks(shape, frame, np.array([0]), node, own, scene, cfg)
    rows = CS.segment_rows(scene, frame, shape, own, 100.0)
    nb, ow = CS.shadow_masks(shape, frame, rows, scene.soma, np.array([0.0]), node.B_bar, node.mu_hat_per_um, cfg,
                             node.d_hat_um)
    return support[0], stripe[0], nb[0], ow[0], frame


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    cfg = cfg_with(background_support="stripe")
    p = cfg.acquisition.res0_um
    tau = cfg.calibration.isolation_depth_gl
    D, r_nb = 6.0, 0.3
    own, scene = scene_of(((D, r_nb),))
    node = pilot(d=0.4)
    support, info, nb, ow, frame = one_plane(cfg, own, scene, node)
    s0 = float(sigma_r(0.0, cfg.renderer))
    rho_nb = CS.shadow_margin(r_nb, MU, s0, B_BAR, tau)
    rho_own = CS.shadow_margin(max(0.2, 0.2), MU, s0, B_BAR, tau)     # the own radius max(d/2, r_seg) = 0.2 um
    assert rho_nb > 0 and rho_own > 0, (rho_nb, rho_own)
    # the + side stops where the neighbour's shade begins: L+ = D - rho_nb; the own cylinder ends at rho_own
    assert abs(info["l_plus_um"] - (D - rho_nb)) <= 1.5 * p, (info["l_plus_um"], D - rho_nb)
    assert abs(info["v_own_plus_um"] - rho_own) <= 1.5 * p and abs(info["v_own_minus_um"] - rho_own) <= 1.5 * p, info
    # the - side runs to the patch's edge (HALF, less the pixel grid)
    assert info["l_minus_um"] >= HALF - 1.5 * p, info["l_minus_um"]
    assert info["w_um"] == max(cfg.calibration.stripe_half_widths_um), info["w_um"]
    assert info["ok"] and info["reason"] == "", info
    # the same scene rotated by 35 deg: the same count within 3 %
    th = math.radians(35.0)
    own2, scene2 = scene_of(((D, r_nb),), theta=th)
    support2, info2, _nb2, _ow2, _fr = one_plane(cfg, own2, scene2, pilot(theta=th, d=0.4))
    rel = abs(info2["n_pixels"] - info["n_pixels"]) / float(info["n_pixels"])
    assert rel <= 0.03, (info["n_pixels"], info2["n_pixels"], rel)
    assert abs(info2["l_plus_um"] - info["l_plus_um"]) <= 2.0 * p, (info2["l_plus_um"], info["l_plus_um"])


def test_reference():
    cfg = cfg_with(background_support="stripe")
    tau = cfg.calibration.isolation_depth_gl
    rng = np.random.default_rng(SEED)
    # (1) shadow_margin against a brute-force scan
    for _ in range(12):
        r = float(rng.uniform(0.1, 2.0))
        s = float(rng.uniform(0.1, 3.0))
        rho_grid = np.arange(0.0, r + 6.0 * s + 1e-9, 0.0005)
        depths = CS.shadow_depths(rho_grid, r, MU, s, B_BAR)
        assert np.all(np.diff(depths) <= 1e-6 * B_BAR), "the shadow is not decreasing in rho (beyond the quadrature's ripple)"
        m = CS.shadow_margin(r, MU, s, B_BAR, tau)
        below = np.flatnonzero(depths <= tau)
        if m < 0:
            assert depths[0] <= tau
        else:
            brute = rho_grid[below[0]] if below.size else rho_grid[-1]
            assert abs(m - brute) <= 0.001, (m, brute)
    # (2) shadow_depths equals shadow_depth element-wise
    rho = rng.uniform(0.0, 3.0, 20)
    a = CS.shadow_depths(rho, 0.3, MU, 0.7, B_BAR)
    b = np.array([CS.shadow_depth(x, 0.3, MU, 0.7, B_BAR) for x in rho])
    assert np.allclose(a, b, rtol=0, atol=1e-12), np.max(np.abs(a - b))
    # (3) shadow_masks against the per-pixel evaluation on a small patch, two neighbours and the own branch
    own, scene = scene_of(((2.5, 0.3), (-3.5, 0.5)), z_other=0.5)
    node = pilot(d=0.6)
    shape, frame = frame_of(cfg, 4.0)
    z = np.array([0.0, 0.84])
    rows = CS.segment_rows(scene, frame, shape, own, 100.0)
    nb, ow = CS.shadow_masks(shape, frame, rows, scene.soma, z, B_BAR, MU, cfg, node.d_hat_um)
    px, py = CS.pixel_centres(shape, frame)
    from allen_diameter.analysis import background as BG
    for k in range(z.size):
        nb_ref = np.zeros(shape, dtype=bool)
        ow_ref = np.zeros(shape, dtype=bool)
        for sg in rows:
            is_own = sg[8] >= 0.5
            r = max(sg[6], sg[7], 0.5 * node.d_hat_um if is_own else 0.0)
            delta = max(abs(sg[2] - z[k]), abs(sg[5] - z[k]))
            dist = BG.segment_distance(px, py, sg[0:2], sg[3:5])
            depth = CS.shadow_depths(dist, r, MU, float(sigma_r(delta, cfg.renderer)), B_BAR)
            (ow_ref if is_own else nb_ref)[depth > tau] = True
        # the masks agree but at the pixels within one bisection tolerance of the margin
        for mine, ref in ((nb[k], nb_ref), (ow[k], ow_ref)):
            disagree = mine != ref
            assert disagree.mean() < 0.002, disagree.mean()
            assert mine.sum() > 0


def test_convergence():
    raise unittest.SkipTest("the stripe is a pixel-set rule, not a discretisation")


def test_invariants():
    cfg = cfg_with(background_support="stripe")
    p = cfg.acquisition.res0_um
    own, scene = scene_of(((4.0, 0.3), (-7.0, 0.4)), with_soma=True)
    node = pilot(d=0.4, theta=0.3)
    support, info, nb, ow, frame = one_plane(cfg, own, scene, node)
    assert not np.any(support & (nb | ow)), "a shaded pixel in the stripe"
    # the heading is oblique to the branch here (0.3 rad): the own cylinder is the contiguous block of own-shaded
    # columns, and the - side, free of neighbours to -7 um, keeps its run
    assert info["side_minus_um"] >= 3.0, info
    px, py = CS.pixel_centres(support.shape, frame)
    u = px * math.cos(0.3) + py * math.sin(0.3)
    assert np.all(np.abs(u[support]) <= info["w_um"] + 1e-9), "a stripe pixel outside the band"
    # N(W*) >= N(W) for every W of the grid
    best = info["n_pixels"]
    for w in cfg.calibration.stripe_half_widths_um:
        c1 = cfg_with(background_support="stripe", stripe_half_widths_um=(float(w),))
        _s, i1 = CS.stripe_support(nb, ow, frame, 0.0, 0.0, 0.3, c1)
        assert i1["n_pixels"] <= best, (w, i1["n_pixels"], best)
    # a neighbour hugging both sides within 3 um (clear runs of about 1 um): side_short
    own3, scene3 = scene_of(((2.6, 0.5), (-2.6, 0.5)))
    _s3, i3, _n3, _o3, _f3 = one_plane(cfg, own3, scene3, pilot(d=0.4))
    assert not i3["ok"] and i3["reason"] == "side_short", i3
    # the soma shades like a tube of its radius: alone at -6 um along v it ends the - run before 6 um
    own_s, scene_s = scene_of(())
    scene_s = Scene(scene_s.seg_xyz, scene_s.seg_radius, scene_s.seg_ids, np.array([[0.0, -6.0, 2.0]]))
    _ss, i_s, n_s, _o_s, _f_s = one_plane(cfg, own_s, scene_s, pilot(d=0.4))
    s0 = float(sigma_r(0.0, cfg.renderer))
    assert n_s.any() and i_s["l_minus_um"] <= 6.0 - CS.shadow_margin(2.0, MU, s0, B_BAR, cfg.calibration.isolation_depth_gl) + 1.5 * p, i_s
    assert i_s["l_plus_um"] >= HALF - 1.5 * p
    # "square" reproduces far_pixel_mask bit for bit and the old node verdict
    cs = cfg_with(background_support="square")
    shape, frame = frame_of(cs)
    sup_sq, stripe_sq = CAL.support_masks(shape, frame, np.array([0, 1]), node, own, scene, cs)
    assert stripe_sq is None
    z = np.array([0.0, cs.acquisition.dz_um])
    d_max = float(np.max(np.abs(scene.seg_xyz[:, [2, 5]].ravel()[:, None] - z[None, :])))
    reach = float(scene.seg_radius.max()) + cs.calibration.far_pixel_sigmas * float(sigma_r(d_max, cs.renderer))
    rows = CS.segment_rows(scene, frame, shape, own, reach)
    far_ref = CS.far_pixel_mask(shape, frame, rows, scene.soma, z, cs)
    assert np.array_equal(sup_sq, far_ref)
    planes_z = CAL.stage_planes(node, cs, "near") * cs.acquisition.dz_um
    reasons_sq = CAL.calibration_reasons(node, cs, 0.4, own, 14, scene, planes_z)
    assert reasons_sq == ["not_isolated"], reasons_sq


def test_contract():
    cfg = cfg_with(background_support="stripe")
    own, scene = scene_of(((9.0, 0.3),))
    node = pilot(d=0.4)
    support, info, nb, ow, frame = one_plane(cfg, own, scene, node)
    assert support.dtype == bool and nb.dtype == bool and ow.dtype == bool and support.shape == nb.shape
    keys = {"w_um", "v_own_minus_um", "v_own_plus_um", "l_minus_um", "l_plus_um", "side_minus_um", "side_plus_um",
            "n_pixels", "ok", "reason"}
    assert set(info) == keys, set(info) ^ keys
    assert isinstance(info["n_pixels"], int) and info["n_pixels"] == int(support.sum())
    # flatness per set: 12 deg fails the near set (10) and passes the far set (15); 16 deg fails the far set
    assert CAL.criteria_reasons(math.radians(12.0), 0.3, 0.4, cfg) == ["not_flat"]
    assert CAL.criteria_reasons(math.radians(12.0), 0.3, 0.7, cfg) == []
    assert CAL.criteria_reasons(math.radians(16.0), 0.3, 0.7, cfg) == ["not_flat"]
    assert CAL.criteria_reasons(math.radians(9.0), 0.3, 0.4, cfg) == []
    assert cfg.calibration.node_phi_max_deg == 10.0 and cfg.calibration.node_phi_max_deg_far == 15.0
    # patch_planes: the six-tuple on a flat image, the stripe carried by the PlaneScan through the scans JSON
    shape, frame = frame_of(cfg)
    kk = CAL.stage_planes(node, cfg, "near")
    block = np.full((kk.size,) + shape, 150.0)
    patch = (block, kk, np.ones(kk.size, dtype=bool), frame)
    out = CAL.patch_planes(patch, kk, node, own, scene, cfg)
    assert len(out) == 6 and len(out[5]) == kk.size and all(isinstance(x, dict) for x in out[5])
    assert bool(np.all(out[3])), out[3]
    assert np.allclose(out[2][:, 0], 150.0) and np.allclose(out[2][:, 1:], 0.0, atol=1e-9)
    scan = CAL.plane_scan(patch, node, own, scene, cfg)
    assert scan.stripe is not None and len(scan.stripe) == kk.size
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "scans.json")
        calibration_io.save_scans([scan], path, {"check": 1})
        back, _meta = calibration_io.load_scans(path)
        assert back[0].stripe is not None and back[0].stripe[0]["n_pixels"] == scan.stripe[0]["n_pixels"]
        json.load(open(path))                              # plain JSON
    # stripe_reasons' names
    planes_z = kk * cfg.acquisition.dz_um
    assert CAL.calibration_reasons(node, cfg, 0.4, own, 14, scene, planes_z) == []
    own3, scene3 = scene_of(((2.6, 0.5), (-2.6, 0.5)))
    r3 = CAL.calibration_reasons(pilot(d=0.4), cfg, 0.4, own3, 14, scene3, planes_z)
    assert r3 == ["not_isolated:side_short"], r3
    own5, scene5 = scene_of(((1.2, 0.5), (-1.2, 0.5)))                      # no clear run at all: too_few
    r5 = CAL.calibration_reasons(pilot(d=0.4), cfg, 0.4, own5, 14, scene5, planes_z)
    assert r5 == ["not_isolated:too_few"], r5
    c_big = cfg_with(background_support="stripe", stripe_min_pixels=10 ** 6)
    r4 = CAL.calibration_reasons(node, c_big, 0.4, own, 14, scene, planes_z)
    assert r4 == ["not_isolated:too_few"], r4
    out_big = CAL.patch_planes(patch, kk, node, own, scene, c_big)       # the planes follow the stripe's verdict
    assert not np.any(out_big[3]) and all(x["reason"] == "too_few" for x in out_big[5])


def test_determinism():
    cfg = cfg_with(background_support="stripe")
    own, scene = scene_of(((3.0, 0.3),), with_soma=True)
    node = pilot(d=0.4, theta=1.1)
    a = one_plane(cfg, own, scene, node)
    b = one_plane(cfg, own, scene, node)
    assert np.array_equal(a[0], b[0]) and a[1] == b[1]


def test_edge_cases():
    cfg = cfg_with(background_support="stripe")
    p = cfg.acquisition.res0_um
    # no segment but the own branch: both runs reach the patch's edge
    own, scene = scene_of(())
    node = pilot(d=0.4)
    support, info, nb, ow, frame = one_plane(cfg, own, scene, node)
    assert not nb.any() and ow.any()
    assert info["l_plus_um"] >= HALF - 1.5 * p and info["l_minus_um"] >= HALF - 1.5 * p, info
    assert info["ok"]
    # a non-finite d_hat: the own radius is Allen's (0.2 um), so the own cylinder is narrower than with d = 1.0
    _s1, i1, _n1, o1, _f1 = one_plane(cfg, own, scene, pilot(d=NAN))
    _s2, i2, _n2, o2, _f2 = one_plane(cfg, own, scene, pilot(d=1.0))
    assert o1.sum() < o2.sum() and i1["v_own_plus_um"] < i2["v_own_plus_um"]
    # a thick own cylinder (d 1.2 um) shades with its own radius, a thin neighbour (r 0.15 um) with its own
    own_t, scene_t = scene_of(((6.0, 0.15),))
    _st, i_t, _nt, _ot, _ft = one_plane(cfg, own_t, scene_t, pilot(d=1.2))
    s0 = float(sigma_r(0.0, cfg.renderer))
    tau = cfg.calibration.isolation_depth_gl
    assert abs(i_t["l_plus_um"] - (6.0 - CS.shadow_margin(0.15, MU, s0, B_BAR, tau))) <= 1.5 * p, i_t
    assert abs(i_t["v_own_plus_um"] - CS.shadow_margin(0.6, MU, s0, B_BAR, tau)) <= 1.5 * p, i_t
    # nothing shaded at a huge tau: the whole band
    c_huge = cfg_with(background_support="stripe", isolation_depth_gl=1e6)
    s3, i3, n3, o3, _f3 = one_plane(c_huge, own, scene_of(((2.0, 0.5),))[1], node)
    assert not n3.any() and not o3.any() and i3["v_own_plus_um"] == 0.0
    px, py = CS.pixel_centres(s3.shape, _f3)
    assert np.array_equal(s3, np.abs(py) <= i3["w_um"])


def _old_shadow_margin(r, mu, sigma, B, tau, n_iter=40):
    """calibration_scene.shadow_margin as committed on 2026-10-28 (03c43b2), verbatim but for the name: the
    reference of the vectorized search."""
    r, s, tau = float(r), max(float(sigma), 1e-6), float(tau)
    if not (r > 0 and B > 0 and mu > 0):
        return -1.0
    hi = r + 6.0 * s
    grid = np.linspace(0.0, hi, 257)
    depth = CS.shadow_depths(grid, r, mu, s, B)
    below = np.flatnonzero(depth <= tau)
    if below.size and below[0] == 0:
        return -1.0
    if not below.size:
        return hi
    lo, hi = float(grid[below[0] - 1]), float(grid[below[0]])
    for _ in range(int(n_iter)):
        mid = 0.5 * (lo + hi)
        if CS.shadow_depth(mid, r, mu, s, B) > tau:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _old_shadow_masks(shape, frame, rows, soma, planes_z, B_bar, mu, cfg, d_own_um=None, sigma_soma_um=None):
    """calibration_scene.shadow_masks as committed on 2026-10-28 (03c43b2), verbatim but for the names: the
    full-patch reference of the windowed masks."""
    from allen_diameter.analysis import background
    H, W = int(shape[0]), int(shape[1])
    px, py = CS.pixel_centres(shape, frame)
    z = np.asarray(planes_z, dtype=float).reshape(-1)
    rc, tau = cfg.renderer, float(cfg.calibration.isolation_depth_gl)
    n_sigma = float(cfg.calibration.far_pixel_sigmas)
    predicted = all(math.isfinite(float(x)) and float(x) > 0 for x in (B_bar, mu))
    nb = np.zeros((z.size, H, W), dtype=bool)
    own = np.zeros((z.size, H, W), dtype=bool)
    r_own = 0.5 * float(d_own_um) if d_own_um is not None and math.isfinite(float(d_own_um)) and d_own_um > 0 else 0.0
    rows = np.atleast_2d(np.asarray(rows, dtype=float)).reshape(-1, 9)

    def margin_of(r, sigma):
        return _old_shadow_margin(r, float(mu), sigma, float(B_bar), tau) if predicted else r + n_sigma * sigma

    for q in range(rows.shape[0]):
        sg = rows[q]
        is_own = sg[8] >= 0.5
        r = max(float(sg[6]), float(sg[7]), r_own if is_own else 0.0)
        dist = None
        for k in range(z.size):
            delta = max(abs(sg[2] - z[k]), abs(sg[5] - z[k]))
            margin = margin_of(r, float(sigma_r(delta, rc)))
            if margin < 0:
                continue
            if dist is None:
                dist = background.segment_distance(px, py, sg[0:2], sg[3:5])
            (own if is_own else nb)[k] |= dist <= margin
    soma = np.atleast_2d(np.asarray(soma, dtype=float)).reshape(-1, 3)
    if soma.shape[0]:
        if sigma_soma_um is None:
            span = 0.5 * (z.max() - z.min()) if z.size > 1 else 0.0
            sigma_soma_um = float(sigma_r(span, rc))
        for x, y, r in soma:
            margin = margin_of(float(r), float(sigma_soma_um))
            if margin >= 0:
                nb |= (np.hypot(px - x, py - y) <= margin)[None]
    return nb, own


def test_recheck():
    cfg = cfg_with(background_support="stripe")
    p, dz = cfg.acquisition.res0_um, cfg.acquisition.dz_um
    # (F1) a soma beside the node: the verdict equals the scan's stripe at k*, the soma at the stage's blur
    own, scene = scene_of(())
    for which, d_a, P in (("near", 0.4, cfg.calibration.offsets_planes), ("far", 0.7, cfg.calibration.far_offsets_planes)):
        node = pilot(d=0.4)
        sig_stage = float(sigma_r(P * dz, cfg.renderer))
        # the soma's edge where the in-focus margin leaves a 3-um side but the stage's does not
        m0 = CS.shadow_margin(2.0, MU, float(sigma_r(0.0, cfg.renderer)), B_BAR, cfg.calibration.isolation_depth_gl)
        ms = CS.shadow_margin(2.0, MU, sig_stage, B_BAR, cfg.calibration.isolation_depth_gl)
        y_soma = -(0.5 * (m0 + ms) + 3.2)
        sc = Scene(scene.seg_xyz, scene.seg_radius, scene.seg_ids, np.array([[0.0, y_soma, 2.0]]))
        kk = CAL.stage_planes(node, cfg, which)
        planes_z = kk * dz
        shape, frame = frame_of(cfg)
        _sup, st = CAL.support_masks(shape, frame, kk, node, own, sc, cfg)
        at_k = st[int(np.flatnonzero(kk == node.k_star)[0])]
        verdict = CAL.calibration_reasons(node, cfg, d_a, own, 14, sc, planes_z)
        want = [] if at_k["ok"] else ["not_isolated:" + at_k["reason"]]
        assert verdict == want, (which, verdict, at_k)
        assert verdict != [], (which, at_k, y_soma, m0, ms)
        # discriminating: with the soma at the in-focus blur (the verdict before F1) the stripe is usable
        _sup0, st0 = CAL.support_masks(shape, frame, kk, node, own, sc, cfg,
                                       sigma_soma_um=float(sigma_r(0.0, cfg.renderer)))
        assert st0[int(np.flatnonzero(kk == node.k_star)[0])]["ok"], (which, st0)
    # (F2) the trace 0.5 um across from the fitted centre: the fitted axis is own-shaded, no stripe pixel lies
    # within d_hat / 2 of it, and the trace's own tube is removed too
    own2, scene2 = scene_of((), r_own=0.2)
    node = pilot(cy=0.5, d=0.4)                     # the pilot's centre 0.5 um across (+v) from the trace (y = 0)
    support, info, nb, ow, frame = one_plane(cfg, own2, scene2, node)
    px, py = CS.pixel_centres(support.shape, frame)
    assert not np.any(support & (np.abs(py - 0.5) <= 0.2)), "stripe pixels on the fitted cylinder"
    assert not np.any(support & (np.abs(py) <= 0.2)), "stripe pixels on the trace's tube"
    assert info["v_own_plus_um"] > 0 and info["v_own_minus_um"] > 0.5, info
    moved = CS.fitted_own_rows(CS.segment_rows(scene2, frame, support.shape, own2, 100.0), 0.0, 0.5)
    assert moved.shape[0] > 0 and np.allclose(moved[:, [1, 4]], 0.5) and np.all(moved[:, 8] == 1.0)
    assert CS.fitted_own_rows(np.empty((0, 9)), 0.0, 0.0).shape == (0, 9)
    # the vectorized margins against the scalar search of 2026-10-28, to 1e-9 um (both bisect the same monotone
    # quadrature on the same rule; the brackets differ, so the last digits may)
    rng = np.random.default_rng(SEED + 1)
    r = rng.uniform(0.05, 6.0, 300)
    sg = np.exp(rng.uniform(np.log(0.08), np.log(40.0), 300))
    for mu, B, tau in ((1.0, 200.0, 0.03), (2.5, 120.0, 0.3), (0.3, 230.0, 0.03)):
        new = CS.shadow_margins(r, mu, sg, B, tau)
        old = np.array([_old_shadow_margin(a, mu, b, B, tau) for a, b in zip(r, sg)])
        assert np.max(np.abs(new - old)) <= 1e-9, np.max(np.abs(new - old))
    assert np.all(CS.shadow_margins(r[:5], float("nan"), sg[:5], 200.0, 0.03) == -1.0)
    # the windowed masks against the full-patch masks of 2026-10-28, pixel for pixel: random scenes with near,
    # oblique and deep segments (whose margins cover the whole patch), a soma, the own branch; predicted and
    # geometric margins; several planes
    for trial in range(6):
        g = np.random.default_rng([SEED, trial])
        segs, rad, sid = [], [], []
        for q in range(25):
            a = g.uniform(-14, 14, 2)
            b = a + g.normal(0, 4, 2)
            zz = g.choice([0.0, 0.5, 3.0, 25.0, 60.0])
            segs.append([a[0], a[1], zz, b[0], b[1], zz + g.normal(0, 1.0)])
            rad.append([g.uniform(0.1, 1.5)] * 2)
            sid.append([300 + 2 * q, 301 + 2 * q])
        base = scene_of(())[1]
        sc = Scene(np.vstack([base.seg_xyz, np.array(segs)]), np.vstack([base.seg_radius, np.array(rad)]),
                   np.vstack([base.seg_ids, np.array(sid)]), np.array([[g.uniform(-8, 8), g.uniform(-8, 8), 3.0]]))
        own3 = scene_of(())[0]
        shape, frame = frame_of(cfg, 6.0)
        rows = CS.segment_rows(sc, frame, shape, own3, 500.0)
        for B, mu in ((B_BAR, MU), (float("nan"), MU)):
            planes = np.array([0.0, 0.84, 4.48])
            a1 = CS.shadow_masks(shape, frame, rows, sc.soma, planes, B, mu, cfg, 0.5)
            a0 = _old_shadow_masks(shape, frame, rows, sc.soma, planes, B, mu, cfg, 0.5)
            assert np.array_equal(a1[0], a0[0]) and np.array_equal(a1[1], a0[1]), (trial, B, int((a1[0] != a0[0]).sum()),
                                                                                   int((a1[1] != a0[1]).sum()))
            assert a1[0].any(), trial
    # 2026-10-29, the pre-cull (shadow_masks drops the rows whose window is empty even at r + 6 sigma, the largest
    # margin shadow_margins can return): at tau 1e-12 gl every margin is that bracket's end (the shadow there is
    # still ~1e-8 of the dip), so a segment outside the crop at r + 5.5 sigma from its edge pixels shades them and one
    # at r + 6.5 sigma shades nothing; the full-patch masks of 2026-10-28 are the oracle
    cfg_t = cfg_with(background_support="stripe", isolation_depth_gl=1e-12)
    shape, frame = frame_of(cfg_t, 4.0)
    px, py = CS.pixel_centres(shape, frame)
    edge = float(py.max())                                   # the crop's last pixel row (um)
    r_q, delta = 0.3, 2.0
    sig = float(sigma_r(delta, cfg_t.renderer))
    assert abs(CS.shadow_margin(r_q, MU, sig, B_BAR, 1e-12) - (r_q + 6.0 * sig)) < 1e-12
    own4 = scene_of(())[0]
    for gap, shades in ((r_q + 5.5 * sig, True), (r_q + 6.5 * sig, False)):
        y = edge + gap
        sc = Scene(np.array([[-3.0, y, delta, 3.0, y, delta]]), np.array([[r_q, r_q]]), np.array([[900, 901]]),
                   np.empty((0, 3)))
        rows = CS.segment_rows(sc, frame, shape, own4, 500.0)
        a1 = CS.shadow_masks(shape, frame, rows, sc.soma, np.array([0.0]), B_BAR, MU, cfg_t, 0.5)
        a0 = _old_shadow_masks(shape, frame, rows, sc.soma, np.array([0.0]), B_BAR, MU, cfg_t, 0.5)
        assert np.array_equal(a1[0], a0[0]) and np.array_equal(a1[1], a0[1]), (gap, int((a1[0] != a0[0]).sum()))
        assert bool(a1[0][0, -1].any()) is shades and bool(a1[0].any()) is shades, (gap, int(a1[0].sum()))


# ----------------------------------------------------------------- runner --

def _environment():
    parts = ["python %s" % platform.python_version()]
    for package in REPORT_PACKAGES:
        try:
            parts.append("%s %s" % (package, importlib.metadata.version(package)))
        except importlib.metadata.PackageNotFoundError:
            parts.append("%s (not installed)" % package)
    parts.append(platform.platform())
    parts.append("seed %d" % SEED)
    return " | ".join(parts)


def main():
    checks = [(name, obj) for name, obj in globals().items() if name.startswith("test_") and callable(obj)]
    print("== %s" % Path(__file__).name)
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
        except Exception:
            status, detail = "ERROR", traceback.format_exc()
        results.append((name, status, time.perf_counter() - start, detail))
    width = max([len(n) for n, _, _, _ in results] + [4])
    for name, status, seconds, detail in results:
        lines = detail.strip().splitlines()
        print(("%-5s  %-" + str(width) + "s  %8.3fs  %s") % (status, name, seconds, lines[0] if lines else ""))
    for name, status, _s, detail in results:
        if status in ("FAIL", "ERROR"):
            print("\n---- %s: %s\n%s" % (status, name, detail.strip()))
    counts = {s: sum(1 for r in results if r[1] == s) for s in ("PASS", "FAIL", "ERROR", "TODO", "SKIP")}
    print("\n-- " + ", ".join("%d %s" % (n, s.lower()) for s, n in counts.items()))
    return 1 if (counts["FAIL"] + counts["ERROR"] + counts["TODO"]) else 0


if __name__ == "__main__":
    sys.exit(main())
