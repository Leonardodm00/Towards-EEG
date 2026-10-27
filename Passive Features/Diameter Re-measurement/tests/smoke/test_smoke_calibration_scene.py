"""Smoke test for the calibration's patch, planar background, isolation and
straightness -- Block 10.P and 10.I in specs/SPEC.md (D-078.1, D-079,
D-081.1-3; 2026-10-23).

Checks
    test_known_answer   the planar background recovers b0 + bx x + by y
                        exactly on a synthetic plane with a masked tube and
                        to 0.01 gl with 2 gl of noise and 1 % of debris
                        pixels; background_on_line equals the plane at the
                        line's points; the far-pixel mask leaves out exactly
                        the pixels within r + 3 sigma_r(delta) of a segment
                        and the margin grows with the plane's defocus
    test_reference      shadow_depth of a flat faint tube at distance rho
                        equals the renderer's own dip there (the tube's
                        transmittance deficit convolved with the Gaussian on
                        a fine grid) within 5 %; _rect_distance against a
                        brute-force minimum over sampled points
    test_convergence    no discretisation step: skipped
    test_invariants     a tube inside the patch fails isolation whatever its
                        depth; one three patch widths away never does; the
                        node's own branch never counts; a deeper neighbour
                        throws a wider, shallower shadow (the depth at the
                        patch edge rises as the neighbour moves to a depth of
                        a few um and falls beyond); a straight branch has
                        zero excess and a bend of 2 r fails the tolerance
    test_contract       fetch_patch's request covers +-half_um of pixel
                        centres; shapes and dtypes of the mask, the plane and
                        the rows
    test_determinism    the same inputs give bit-identical outputs
    test_edge_cases     too few far pixels -> ok False; an empty scene -> all
                        pixels far, isolation depth 0; a two-point branch ->
                        excess 0

Run
    cd "Passive Features/Diameter Re-measurement"
    python tests/smoke/test_smoke_calibration_scene.py

Keep this file pure ASCII (hpc-python-compat).
"""
from __future__ import annotations

import dataclasses
import importlib.metadata
import math
import platform
import sys
import time
import traceback
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
SRC = WS / "src"
for _p in (SRC, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from allen_diameter.analysis import calibration_scene as CS  # noqa: E402
from allen_diameter.analysis.node_pipeline import Branch  # noqa: E402
from allen_diameter.analysis.plane_iteration import Scene  # noqa: E402
from allen_diameter.config import default_config  # noqa: E402
from allen_diameter.model import kernel as K  # noqa: E402
from allen_image_io import CropFrame  # noqa: E402

SEED = 20261023
REPORT_PACKAGES = ("numpy", "scipy")


# --------------------------------------------------------------- helpers ---

def two_tube_scene(y_other=3.0, z_other=0.0, r_other=0.3, with_soma=False):
    """A branch along x (ids 100..114, r 0.1 um, z 0) and a second one at y = y_other (ids 200..204)."""
    br = Branch.from_points(np.c_[np.linspace(-8, 8, 15), np.zeros(15), np.zeros(15)], 0.1, ids=np.arange(100, 115))
    other = Branch.from_points(np.c_[np.linspace(-8, 8, 5), np.full(5, y_other), np.full(5, z_other)], r_other,
                               ids=np.arange(200, 205))
    s1, s2 = Scene.from_branch(br), Scene.from_branch(other)
    soma = np.array([[-30.0, 0.0, 5.0]]) if with_soma else np.empty((0, 3))
    return br, Scene(np.vstack([s1.seg_xyz, s2.seg_xyz]), np.vstack([s1.seg_radius, s2.seg_radius]),
                     np.vstack([s1.seg_ids, s2.seg_ids]), soma)


def patch_frame(cfg, cx=0.02, cy=-0.03, half=5.0):
    p = cfg.acquisition.res0_um
    left, top, w, h = CS.patch_bounds(cx, cy, half, p)
    return CropFrame(left, top, 0, p), (h, w)


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    cfg = default_config()
    cx, cy = 0.02, -0.03
    frame, shape = patch_frame(cfg, cx, cy, 5.0)
    br, scene = two_tube_scene()
    z = np.arange(-3, 4) * cfg.acquisition.dz_um
    rows = CS.segment_rows(scene, frame, shape, br, 20.0)
    far = CS.far_pixel_mask(shape, frame, rows, scene.soma, z, cfg)
    px, py = CS.pixel_centres(shape, frame)
    # exact plane recovery with the tube's pixels masked (b to 1e-9: float64 lstsq on an exact plane)
    truth = np.array([200.0, 0.04, -0.03])
    img = truth[0] + truth[1] * (px - cx) + truth[2] * (py - cy)
    b, n, ok = CS.planar_background(img, far[3], frame, cx, cy, 3.0, 0.05)
    assert ok and np.max(np.abs(b - truth)) <= 1e-9, b
    # with 2 gl of white noise and 1 % of bright debris: b0 to 0.01 gl (the plane averages ~6000 pixels: 2 / sqrt(6000)
    # = 0.026 gl SE, 3 SE = 0.08; the trim removes the debris), the slopes to 0.005 gl/um
    rng = np.random.default_rng(SEED)
    noisy = img + rng.normal(0.0, 2.0, img.shape)
    debris = rng.random(img.shape) < 0.01
    noisy[debris] += 60.0
    b2, n2, ok2 = CS.planar_background(noisy, far[3], frame, cx, cy, 3.0, 0.05)
    assert ok2 and abs(b2[0] - truth[0]) <= 0.08 and np.max(np.abs(b2[1:] - truth[1:])) <= 0.01, b2
    assert n2 < int(far[3].sum()), "the trim must drop the debris"
    # the plane along a line
    xy = np.c_[cx + np.linspace(-3, 3, 7) * 0.6, cy + np.linspace(-3, 3, 7) * 0.8]
    line = CS.background_on_line(truth, cx, cy, xy)
    assert np.max(np.abs(line - (truth[0] + truth[1] * (xy[:, 0] - cx) + truth[2] * (xy[:, 1] - cy)))) <= 1e-12
    # the mask: a pixel is far iff farther than max(r0, r1) + 3 sigma_r(delta) from every segment; the margin grows
    # with |delta| (planes -3 and +3 mask more than plane 0)
    for q, zk in enumerate(z):
        margin_own = 0.1 + 3.0 * float(K.sigma_r(abs(zk), cfg.renderer))
        margin_other = 0.3 + 3.0 * float(K.sigma_r(abs(zk), cfg.renderer))
        d_own = np.abs(py - 0.0)                                # the own branch runs along x at y = 0
        d_other = np.abs(py - 3.0)
        expect = (d_own > margin_own) & (d_other > margin_other)
        # the x extent: both branches span -8..8 um, beyond the +-5 um patch, so the distance is the y distance
        assert np.array_equal(far[q], expect), ("mask mismatch in plane", q, int(np.sum(far[q] != expect)))
    assert far[3].mean() > far[0].mean() > 0 and far[3].mean() > far[6].mean(), far.mean(axis=(1, 2))


def test_reference():
    cfg = default_config()
    # shadow_depth against a fine-grid convolution of the tube's transmittance deficit
    B, mu, r = 210.0, 1.0, 0.35
    for sigma, rho in ((0.3, 0.0), (0.6, 1.2), (1.5, 3.0), (3.0, 4.0)):
        vp = np.linspace(-r, r, 4001)
        deficit = 1.0 - np.exp(-2.0 * mu * np.sqrt(np.maximum(r * r - vp * vp, 0.0)))
        g = np.exp(-0.5 * ((rho - vp) / sigma) ** 2) / (math.sqrt(2.0 * math.pi) * sigma)
        ref = B * np.trapezoid(deficit * g, vp) if hasattr(np, "trapezoid") else B * np.trapz(deficit * g, vp)
        got = CS.shadow_depth(rho, r, mu, sigma, B)
        # 16-node Gauss-Legendre on a sqrt-edged integrand: within 5 % (the SPEC's tolerance); the fine grid is the
        # reference
        assert abs(got - ref) <= 0.05 * ref + 1e-6, (sigma, rho, got, ref)
    # _rect_distance against brute force over sampled points of the segment and the rectangle
    rng = np.random.default_rng(SEED)
    lo, hi = np.array([-1.0, -2.0]), np.array([3.0, 1.0])
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], 81), np.linspace(lo[1], hi[1], 61))
    rect = np.c_[gx.ravel(), gy.ravel()]
    for _ in range(40):
        a, b = rng.uniform(-6, 8, 2), rng.uniform(-6, 8, 2)
        t = np.linspace(0, 1, 401)[:, None]
        seg = a[None, :] + t * (b - a)[None, :]
        brute = float(np.min(np.linalg.norm(seg[:, None, :] - rect[None, :, :], axis=2)))
        got = CS._rect_distance(a, b, lo, hi)
        # the brute force samples both sets: within one sample step (0.05 um) plus a little
        assert abs(got - brute) <= 0.06, (a, b, got, brute)


def test_convergence():
    raise unittest.SkipTest("no discretisation step in Block 10.P/10.I: the shadow quadrature is checked in test_reference")


def test_invariants():
    cfg = default_config()
    cx, cy, half = 0.02, -0.03, 5.0
    z = np.arange(-3, 4) * cfg.acquisition.dz_um
    B, mu = 200.0, 1.0
    # a neighbour inside the patch fails at every depth; one 3 patch widths away never does; own branch never counts
    for z_other in (0.0, 2.0, 10.0, 30.0):
        br, scene = two_tube_scene(y_other=3.0, z_other=z_other)
        depth, who = CS.isolation_depth(br, 7, scene, cx, cy, half, z, B, mu, cfg)
        assert depth > cfg.calibration.isolation_depth_gl and who == 201, (z_other, depth, who)
    br, scene = two_tube_scene(y_other=3 * 2 * half + 5.0)
    depth, who = CS.isolation_depth(br, 7, scene, cx, cy, half, z, B, mu, cfg)
    assert depth <= cfg.calibration.isolation_depth_gl, (depth, who)
    depth_own, who_own = CS.isolation_depth(br, 7, Scene.from_branch(br), cx, cy, half, z, B, mu, cfg)
    assert depth_own == 0.0 and who_own is None
    # a neighbour just outside the patch (1 um beyond the edge): its shadow at the edge first grows with its depth
    # (wider blur reaches the edge) and then fades (the same absorption spread thinner)
    depths = []
    for z_other in (0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 40.0):
        br, scene = two_tube_scene(y_other=cy + half + 1.0 + 0.3, z_other=z_other)
        depths.append(CS.isolation_depth(br, 7, scene, cx, cy, half, z, B, mu, cfg)[0])
    i_max = int(np.argmax(depths))
    assert 0 < i_max < len(depths) - 1 and depths[-1] < depths[i_max], depths
    # the soma counts, as a disc of its radius at the stage's largest blur
    br, scene = two_tube_scene(y_other=100.0, with_soma=True)
    scene_near = Scene(scene.seg_xyz, scene.seg_radius, scene.seg_ids, np.array([[cx - half - 2.0, cy, 5.0]]))
    assert CS.isolation_depth(br, 7, scene_near, cx, cy, half, z, B, mu, cfg)[1] == "soma"
    # straightness: a straight branch has zero excess; a bend of 2 r fails a tolerance of 1 radius (D-081.3's; the
    # test is off by default and its tolerance 2 radii since D-093, tests/smoke/test_smoke_calibration.py)
    straight = Branch.from_points(np.c_[np.linspace(-12, 12, 25), 0.3 * np.linspace(-12, 12, 25), np.zeros(25)], 0.2)
    assert CS.straightness_excess(straight, 12, 10.0) <= 1e-9
    x = np.linspace(-12, 12, 25)
    bent = Branch.from_points(np.c_[x, 0.4 * (1 - np.cos(np.pi * x / 10.0)) * (np.abs(x) <= 10), np.zeros(25)], 0.2)
    exc = CS.straightness_excess(bent, 12, 10.0)
    assert exc > 1.0 * 0.2, exc


def test_contract():
    cfg = default_config()
    p = cfg.acquisition.res0_um
    left, top, w, h = CS.patch_bounds(0.02, -0.03, 5.0, p)
    xs = (left + np.arange(w)) * p
    assert xs[0] >= 0.02 - 5.0 - 1e-9 and xs[-1] <= 0.02 + 5.0 + 1e-9 and xs[0] - p < 0.02 - 5.0 and xs[-1] + p > 0.02 + 5.0
    calls = []

    def provider(left_, top_, width_, height_, k_lo, k_hi):
        calls.append((left_, top_, width_, height_, k_lo, k_hi))
        return None
    CS.fetch_patch(provider, 0.02, -0.03, 5.0, -3, 3, p)
    assert calls == [(left, top, w, h, -3, 3)], calls
    frame, shape = patch_frame(cfg)
    br, scene = two_tube_scene()
    rows = CS.segment_rows(scene, frame, shape, br, 20.0)
    assert rows.shape == (18, 9) and rows.dtype == float and set(rows[:, 8].tolist()) == {0.0, 1.0}
    z = np.arange(-3, 4) * cfg.acquisition.dz_um
    far = CS.far_pixel_mask(shape, frame, rows, scene.soma, z, cfg)
    assert far.shape == (7,) + shape and far.dtype == bool
    b, n, ok = CS.planar_background(np.full(shape, 200.0), far[3], frame, 0.02, -0.03, 3.0, 0.05)
    assert b.shape == (3,) and isinstance(n, int) and ok and abs(b[0] - 200.0) <= 1e-9 and abs(b[1]) + abs(b[2]) <= 1e-9
    assert CS.segment_margins(rows, z, 3.0, cfg.renderer).shape == (18, 7)


def test_determinism():
    cfg = default_config()
    frame, shape = patch_frame(cfg)
    br, scene = two_tube_scene()
    z = np.arange(-3, 4) * cfg.acquisition.dz_um
    rows = CS.segment_rows(scene, frame, shape, br, 20.0)
    a = CS.far_pixel_mask(shape, frame, rows, scene.soma, z, cfg)
    b = CS.far_pixel_mask(shape, frame, rows, scene.soma, z, cfg)
    assert np.array_equal(a, b)
    rng = np.random.default_rng(SEED)
    img = 200.0 + rng.normal(0.0, 2.0, shape)
    p1 = CS.planar_background(img, a[3], frame, 0.02, -0.03, 3.0, 0.05)[0]
    p2 = CS.planar_background(img, a[3], frame, 0.02, -0.03, 3.0, 0.05)[0]
    assert np.array_equal(p1, p2)
    d1 = CS.isolation_depth(br, 7, scene, 0.02, -0.03, 5.0, z, 200.0, 1.0, cfg)
    d2 = CS.isolation_depth(br, 7, scene, 0.02, -0.03, 5.0, z, 200.0, 1.0, cfg)
    assert d1 == d2


def test_edge_cases():
    cfg = default_config()
    frame, shape = patch_frame(cfg)
    img = np.full(shape, 200.0)
    few = np.zeros(shape, dtype=bool)
    few[0, :3] = True
    b, n, ok = CS.planar_background(img, few, frame, 0.02, -0.03, 3.0, 0.05)
    assert not ok and n == 0 and np.all(np.isnan(b))
    empty = Scene(np.empty((0, 6)), np.empty((0, 2)), np.empty((0, 2), dtype=np.int64), np.empty((0, 3)))
    br = Branch.from_points(np.c_[np.linspace(-8, 8, 15), np.zeros(15), np.zeros(15)], 0.1)
    z = np.arange(-3, 4) * cfg.acquisition.dz_um
    rows = CS.segment_rows(empty, frame, shape, br, 20.0)
    assert rows.shape == (0, 9)
    far = CS.far_pixel_mask(shape, frame, rows, empty.soma, z, cfg)
    assert far.all()
    assert CS.isolation_depth(br, 7, empty, 0.02, -0.03, 5.0, z, 200.0, 1.0, cfg) == (0.0, None)
    two = Branch.from_points(np.array([[0.0, 0.0, 0.0], [1.0, 0.5, 0.0]]), 0.1)
    assert CS.straightness_excess(two, 0, 10.0) == 0.0


# ---------------------------------------------------------------- runner ---

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
