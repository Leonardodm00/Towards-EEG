"""Block 5 (per-node chain): oracles from SPEC.md section 2.2 and Block 5."""
import dataclasses
import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from allen_image_io import CropFrame
from allen_diameter.config import default_config
from allen_diameter.analysis import background, focus, node_pipeline as NP, path, profiles
from allen_diameter.model import geometry as G, render

CFG = default_config()
M = CFG.measure
P = CFG.acquisition.res0_um


def test_profile_offsets():
    v = profiles.profile_offsets(M)
    n = round(M.profile_half_um / M.profile_step_um)
    assert_allclose(v, np.arange(-n, n + 1) * M.profile_step_um, rtol=0, atol=0)
    assert v[n] == 0.0


def test_bilinear_exact_on_ramp_and_nan_outside():
    H, W = 40, 50
    rows, cols = np.mgrid[0:H, 0:W]
    plane = 3.0 + 0.7 * cols - 0.4 * rows                         # linear in (col, row)
    frame = CropFrame(100, 200, 0, P)
    o = np.array([(100 + 25.3) * P, (200 + 18.6) * P])
    th = 0.37
    yh, eu = np.array([-math.sin(th), math.cos(th)]), np.array([math.cos(th), math.sin(th)])
    v = profiles.profile_offsets(M)[10:-10]
    I = profiles.sample_profile(plane, frame, o, yh, eu, v, n_avg=2, step=M.profile_step_um)
    # expected: the ramp evaluated at (col, row) = (x / p - left, y / p - top), averaged over 5 along-branch
    # offsets symmetric about 0 (the mean of a linear function = its value at the centre)
    x, y = o[0] + v * yh[0], o[1] + v * yh[1]
    col, row = x / P - 100, y / P - 200
    assert_allclose(I, 3.0 + 0.7 * col - 0.4 * row, rtol=1e-12)
    far = profiles.sample_profile(plane, frame, o, yh, eu, np.array([-100.0, 0.0]), 0, 0.0)
    assert np.isnan(far[0]) and np.isfinite(far[1])


def test_focus_score_definition():
    v = profiles.profile_offsets(M)
    I = 200.0 - 80.0 * np.exp(-0.5 * (v / 0.3) ** 2)
    I[np.abs(v) > 2.5] = 190.0                       # ends differ from the centre background
    F, Imin, B = focus.focus_score(I, v, M)
    from scipy.ndimage import gaussian_filter1d
    sm = gaussian_filter1d(I, M.focus_smooth_px, mode="nearest")
    B_ref = np.median(I[np.abs(v) > M.focus_bg_ends_um])
    assert B == pytest.approx(B_ref) and Imin == pytest.approx(sm.min())
    assert F == pytest.approx(-math.log(sm.min() / B_ref), rel=1e-14)
    I2 = I.copy()
    I2[3] = np.nan
    assert all(math.isnan(x) for x in focus.focus_score(I2, v, M))


def test_best_plane_vertex_plateau_edge():
    dz = 0.28
    z = np.arange(9) * dz
    F = -(z - 1.0) ** 2 + 5.0                       # parabola, vertex at 1.0 um
    k, zs, plateau, edge = focus.best_plane(F, z, M, dz)
    assert k == int(np.argmax(F)) and zs == pytest.approx(1.0, abs=1e-12) and not plateau and not edge
    Ff = np.array([1, 2, 5, 5, 5, 5, 2, 1, 0.5], float)
    k, zs, plateau, edge = focus.best_plane(Ff, z, M, dz)
    assert plateau and zs == pytest.approx(np.mean(z[2:6]))
    Fe = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], float)
    assert focus.best_plane(Fe, z, M, dz)[3]
    Fn = np.array([np.nan, np.nan, 3, 4, 9, 4, 3, 2, 1.0])
    Fn[1] = np.nan
    Fn2 = np.array([np.nan, np.nan, 9, 4, 3, 2, 1, 0.5, 0.2])
    assert focus.best_plane(Fn2, z, M, dz)[3]           # first usable plane = stack edge


def test_tls_and_angles():
    rng = np.random.default_rng(4)
    t = G.axis_direction(math.radians(35), 2.0)
    s = np.linspace(-2, 2, 9)
    pts = s[:, None] * t[None, :] + rng.normal(0, 0.02, (9, 3))
    est = path.orient(path.tls_direction(pts), pts[0], pts[-1])
    # reference: leading right singular vector of the centred points (numpy SVD)
    ref = np.linalg.svd(pts - pts.mean(0))[2][0]
    ref = ref if ref @ (pts[-1] - pts[0]) >= 0 else -ref
    assert_allclose(est, ref, atol=1e-10)
    assert math.degrees(math.acos(min(1, abs(est @ t)))) < 0.5
    th, ph, yh, eu = path.angles(est)
    assert th == pytest.approx(math.atan2(est[1], est[0])) and ph == pytest.approx(math.asin(abs(est[2])))
    assert abs(yh @ eu) < 1e-15 and 0 <= ph <= math.pi / 2


def test_segment_distance_and_mask():
    d = background.segment_distance(np.array([0.0, 3.0, -1.0]), np.array([1.0, 0.0, 0.0]), (0, 0), (2, 0))
    assert_allclose(d, [1.0, 1.0, 1.0], rtol=1e-15)
    plane = np.arange(100.0).reshape(10, 10)
    mask = background.mask_near_branch((10, 10), 0, 0, 1.0, np.array([[2.0, 2, 0], [7, 2, 0]]),
                                       np.array([0.5, 1.0]), 1.0)
    # pixel centres within max radius (1.0) + margin (1.0) of the segment (2,2)-(7,2)
    rows, cols = np.mgrid[0:10, 0:10]
    ref = background.segment_distance(cols.astype(float), rows.astype(float), (2, 2), (7, 2)) <= 2.0
    assert np.array_equal(mask, ref)
    B, frac, ok = background.masked_median(plane, mask, 0.2)
    assert B == np.median(plane[~mask]) and frac == pytest.approx((~mask).mean()) and ok


def _phantom(d=1.0, phi_deg=0.0, theta=0.6, mu=0.6, seed=3):
    cfg = CFG
    tube = G.Tube((0.04, -0.03, 0.05), d / 2, math.radians(phi_deg), theta, 1.0, cfg.renderer.U_um, "axial")
    t = tube.t_hat
    s = np.arange(-6, 7) * 1.18
    xyz = np.array(tube.c)[None, :] + s[:, None] * t[None, :]
    br = NP.Branch.from_points(xyz, d / 2)
    rng = np.random.default_rng(seed)

    def provider(left, top, width, height, k_lo, k_hi):
        return render.synthetic_block(tube, mu, np.arange(k_lo, k_hi + 1), left, top, width, height, cfg, rng,
                                      int(math.ceil(cfg.renderer.pad_um / P)))
    return br, provider


def test_measure_node_on_phantom():
    br, prov = _phantom()
    r = NP.measure_node(br, 6, prov, CFG)
    assert r.fit_status == "converged"
    dth = (r.theta_rad - 0.6 + math.pi / 2) % math.pi - math.pi / 2      # heading defined modulo pi
    assert abs(math.degrees(dth)) < 2 and math.degrees(r.phi_rad) < 4
    assert 0.95 <= r.d_hat_um / 1.0 <= 1.25
    assert np.isnan(r.s_star_um) and r.reg_verdict == ""
    # null registration values (a NOT ON node) are NaN, not TypeError
    r2 = NP.measure_node(br, 6, prov, CFG, reg={"verdict": "NOT ON A VISIBLE PROCESS (x)", "lateral_offset_um": None,
                                                 "z_offset_um": None})
    assert np.isnan(r2.s_star_um) and np.isnan(r2.dz_star_um)


def test_measure_node_empty_tissue_is_faint():
    br, _ = _phantom()

    def empty(left, top, width, height, k_lo, k_hi):
        ks = np.arange(k_lo, k_hi + 1)
        rng = np.random.default_rng(0)
        blk = np.clip(np.rint(210 + rng.normal(0, 2, (ks.size, height, width))), 0, 255).astype(np.uint8)
        return blk, ks, np.ones(ks.size, bool), CropFrame(left, top, 0, P)
    r = NP.measure_node(br, 6, empty, CFG)
    assert "faint" in r.flags


def test_exactly_vertical_window_does_not_crash():
    """SPEC Block 5: phi = arcsin|t_z| in [0, pi/2]; 'vertical' (phi > phi_vertical_deg) is a flag and a
    reported column, never a reason to stop. A window whose nodes share x, y (a purely vertical piece of
    a trace) gives phi = pi/2 exactly; measure_node must return a flagged NodeResult, not raise."""
    def prov(left, top, width, height, k_lo, k_hi):
        ks = np.arange(k_lo, k_hi + 1)
        blk = np.full((ks.size, height, width), 210, np.uint8)
        rr, cc = np.mgrid[0:height, 0:width]
        x, y = (cc + left) * P, (rr + top) * P
        blk[:, (x - 2) ** 2 + (y - 2) ** 2 < 0.25] = 120
        return blk, ks, np.ones(ks.size, bool), CropFrame(left, top, 0, P)
    xyz = np.array([[2.0, 2.0, z] for z in np.arange(0, 6, 1.18)])
    br = NP.Branch.from_points(xyz, 0.5)
    r = NP.measure_node(br, 2, prov, CFG)
    assert r.vertical and "vertical" in r.flags
