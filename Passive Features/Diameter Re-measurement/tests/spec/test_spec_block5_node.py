"""Spec-derived tests of Block 5 (the shared per-node chain: profiles, focus
scores, sub-plane depth, direction, background, the request) -- integration
tester, run D1, 2026-11-13. Expected values: SPEC Block 5 / section 2.2
written out with the library calls the spec names (scipy.ndimage
gaussian_filter1d, numpy.gradient, scipy.integrate.trapezoid, numpy.median,
numpy.linalg) or closed forms; the code under test only produces the outputs
that are checked.

Run (from the workstream directory):
    python -m pytest tests/spec/test_spec_block5_node.py -q
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import integrate, ndimage, special

import spec_helpers_D1 as H

import allen_image_io as aio
from allen_diameter.analysis import background as BG
from allen_diameter.analysis import focus as F
from allen_diameter.analysis import node_pipeline as NP
from allen_diameter.analysis import path as PA
from allen_diameter.analysis import profiles as PR
from allen_diameter.config import default_config

CFG = default_config()
M = CFG.measure
PX = CFG.acquisition.res0_um
DZ = CFG.acquisition.dz_um


def mcfg(**kw):
    return dataclasses.replace(M, **kw)


# ----------------------------------------------------------------- O5.1 ---

def test_profile_offsets():
    v = PR.profile_offsets(M)
    N = int(round(M.profile_half_um / M.profile_step_um))
    assert_allclose(v, np.arange(-N, N + 1) * M.profile_step_um, rtol=0, atol=1e-15)
    assert v.dtype == np.float64 and v[N] == 0.0


# ----------------------------------------------------------------- O5.2 ---

def ramp(H_, W_):
    rr, cc = np.mgrid[0:H_, 0:W_]
    return 3.0 + 0.5 * cc + 0.25 * rr


@pytest.mark.parametrize("theta", [0.0, 0.37, 1.9])
def test_sample_profile_bilinear_exact_and_nan_outside(theta):
    plane = ramp(80, 90)
    frame = aio.CropFrame(100, 200, 0, PX)
    o = np.array([(100 + 45.3) * PX, (200 + 40.7) * PX])
    y_hat = np.array([-math.sin(theta), math.cos(theta)])
    e_u = np.array([math.cos(theta), math.sin(theta)])
    v = PR.profile_offsets(M)
    I = PR.sample_profile(plane, frame, o, y_hat, e_u, v, n_avg=3, step=M.profile_step_um)
    col = (o[0] + v * y_hat[0]) / PX - 100
    row = (o[1] + v * y_hat[1]) / PX - 200
    assert_allclose(I, 3.0 + 0.5 * col + 0.25 * row, rtol=0, atol=1e-10)   # bilinear exact on a ramp
    # a profile that leaves the block: NaN there, finite where all 7 lines are inside
    o2 = np.array([(100 + 3.0) * PX, (200 + 40.0) * PX])
    I2 = PR.sample_profile(plane, frame, o2, np.array([1.0, 0.0]), np.array([0.0, 1.0]), v,
                           n_avg=0, step=M.profile_step_um)
    c2 = (o2[0] + v) / PX - 100
    inside = (c2 >= 0) & (c2 <= 89)
    assert np.all(np.isnan(I2[~inside])) and np.all(np.isfinite(I2[inside]))
    assert_allclose(I2[inside], 3.0 + 0.5 * c2[inside] + 0.25 * 40.0, atol=1e-10)


def test_sample_profile_averages_along_the_branch():
    rng = np.random.default_rng(2)
    plane = rng.uniform(50, 200, size=(60, 60))
    frame = aio.CropFrame(0, 0, 0, PX)
    o = np.array([30.2 * PX, 29.6 * PX])
    y_hat, e_u = np.array([0.0, 1.0]), np.array([1.0, 0.0])
    v = np.arange(-5, 6) * PX
    A, step = 2, M.profile_step_um
    I = PR.sample_profile(plane, frame, o, y_hat, e_u, v, n_avg=A, step=step)
    exp = np.zeros_like(v)
    for a in range(-A, A + 1):
        cols = np.full_like(v, (o[0] + a * step) / PX)
        rows = (o[1] + v) / PX
        exp += ndimage.map_coordinates(plane, [rows, cols], order=1, mode="constant", cval=np.nan)
    assert_allclose(I, exp / (2 * A + 1), rtol=0, atol=1e-10)


# ------------------------------------------------------------ O5.3, O5.4 --

def hand_G(I, v, m, radius):
    It = ndimage.gaussian_filter1d(I, m.focus_smooth_px, mode="nearest") if m.focus_smooth_px > 0 else I
    B = np.median(I[np.abs(v) > m.focus_bg_ends_um])
    g = np.gradient(It, v)
    h = radius + m.focus_grad_margin_um if m.focus_grad_window == "radius_margin" else np.max(np.abs(v))
    w = np.abs(v) <= h + 1e-9
    return integrate.trapezoid(g[w] ** 2, v[w]) / B ** 2, B, It


def test_gradient_energy_and_dip_depth_by_hand():
    rng = np.random.default_rng(7)
    v = PR.profile_offsets(M)
    I = 180.0 * (1 - 0.35 * np.exp(-v ** 2 / (2 * 0.3 ** 2))) + rng.normal(0, 2.0, v.size)
    G, B = F.gradient_energy(I, v, M, radius_um=0.4)
    Gh, Bh, It = hand_G(I, v, M, 0.4)
    assert abs(G - Gh) <= 1e-12 * Gh and B == Bh
    mw = mcfg(focus_grad_window="whole_profile")
    G2, _ = F.gradient_energy(I, v, mw, radius_um=None)
    assert abs(G2 - hand_G(I, v, mw, 0.0)[0]) <= 1e-12 * G2
    Fv, Imin, B3 = F.dip_depth(I, v, M)
    assert abs(Imin - It.min()) < 1e-12 and B3 == Bh
    assert abs(Fv - (-math.log(It.min() / Bh))) < 1e-12


def test_gradient_energy_analytic_gaussian_dip():
    """section 2.2: whole line sqrt(pi) D^2/(2w); over |v| <= h the erf form;
    smoothing off, within the central-difference error h_v^2/(2 w^2)."""
    m = mcfg(focus_smooth_px=0.0)
    D, w, B = 0.4, 0.4, 200.0
    v = np.arange(-300, 301) * 0.01
    I = B * (1 - D * np.exp(-v ** 2 / (2 * w * w)))
    G, Bo = F.gradient_energy(I, v, dataclasses.replace(m, focus_grad_window="whole_profile"))
    rel = 0.01 ** 2 / (2 * w * w)
    assert abs(G - math.sqrt(math.pi) * D * D / (2 * w)) <= 2 * rel * G
    r = 0.3
    h = r + m.focus_grad_margin_um
    Gh, _ = F.gradient_energy(I, v, m, radius_um=r)
    exp = (D * D / w) * (math.sqrt(math.pi) / 2 * special.erf(h / w) - (h / w) * math.exp(-h * h / (w * w)))
    assert abs(Gh - exp) <= 2 * rel * exp + 1e-6 * exp
    # fixed dip area A = sqrt(2 pi) D w: G = A^2/(4 sqrt(pi) w^3)
    A = math.sqrt(2 * math.pi) * D * w
    assert abs(math.sqrt(math.pi) * D * D / (2 * w) - A * A / (4 * math.sqrt(math.pi) * w ** 3)) < 1e-14


def test_scores_nan_and_refusals():
    v = PR.profile_offsets(M)
    I = 150.0 - 30.0 * np.exp(-v ** 2 / 0.2)
    bad = I.copy()
    bad[5] = np.nan
    assert np.isnan(F.gradient_energy(bad, v, M, radius_um=0.3)[0])
    assert np.isnan(F.dip_depth(bad, v, M)[0])
    neg = I - 1000.0
    assert np.isnan(F.gradient_energy(neg, v, M, radius_um=0.3)[0])     # B <= 0
    assert np.isnan(F.dip_depth(neg, v, M)[0])
    for rad in (None, float("nan"), -0.1):
        with pytest.raises(ValueError):
            F.gradient_energy(I, v, M, radius_um=rad)
    with pytest.raises(ValueError):
        F.gradient_energy(I, v[::-1], M, radius_um=0.3)                  # not increasing
    with pytest.raises(ValueError):
        F.gradient_energy(I, v[:-1], M, radius_um=0.3)                   # wrong length
    with pytest.raises(ValueError):
        F.gradient_energy(I[:2], v[:2], M, radius_um=0.3)                # < 3 samples


# ----------------------------------------------------------------- O5.5 ---

def test_best_plane_vertex_plateau_edges():
    z = np.arange(7) * DZ
    z0 = 3 * DZ + 0.1
    S = 10 - 100 * (z - z0) ** 2
    k, zs, plateau, edge = F.best_plane(S, z, M, DZ)
    assert k == 3 and not plateau and not edge
    assert abs(zs - z0) < 1e-12                       # the vertex is exact on a parabola
    S2 = np.array([1.0, 2.0, 5.0, 5.0, 5.0, 2.0, 1.0])
    k, zs, plateau, edge = F.best_plane(S2, z, M, DZ)
    assert plateau and abs(zs - 3 * DZ) < 1e-12       # mean depth of the run 2..4
    S3 = np.array([1.0, 2.0, np.nan, 9.0, 3.0, 2.0, 1.0])
    k, zs, plateau, edge = F.best_plane(S3, z, M, DZ)
    assert k == 3 and zs == z[3]                       # a missing neighbour: z_k*
    S4 = 10 - 100 * (z - (3 * DZ + 0.2)) ** 2          # vertex beyond +dz/2: clipped
    S4[4] = S4[3] - 1e-3
    k, zs, plateau, edge = F.best_plane(S4, z, M, DZ)
    assert zs <= z[k] + DZ / 2 + 1e-15 and zs >= z[k] - DZ / 2 - 1e-15
    # at_edge: k* on the first / last usable plane -- the end of the searched range,
    # or of the stack where planes are missing -- in both cases alike
    assert F.best_plane(np.array([9.0, 3, 2, 1, 1, 1, 1]), z, M, DZ)[3]
    assert F.best_plane(np.array([1.0, 1, 1, 1, 2, 3, 9]), z, M, DZ)[3]
    assert F.best_plane(np.array([np.nan, np.nan, 9.0, 3, 2, 1, 1]), z, M, DZ)[3]
    assert F.best_plane(np.array([1.0, 2, 3, 9, np.nan, np.nan, np.nan]), z, M, DZ)[3]
    S6 = np.array([1.0, 2.0, 3.0, 9.0, 3.0, 2.0, 1.0])
    assert not F.best_plane(S6, z, M, DZ)[3]
    assert F.best_plane(np.full(7, np.nan), z, M, DZ)[0] is None


@pytest.mark.xfail(strict=True, reason="D1-F1: the code flags only k* ON the first/last usable plane; "
                   "'within edge_margin_planes (1)' read as distance <= 1 would also flag the next one")
def test_best_plane_edge_margin_reading_distance_le_margin():
    z = np.arange(7) * DZ
    assert M.edge_margin_planes == 1
    assert F.best_plane(np.array([1.0, 9, 3, 2, 1, 1, 1]), z, M, DZ)[3]          # 1 from the range end
    assert F.best_plane(np.array([np.nan, np.nan, 1.0, 5, 2, 1, 1]), z, M, DZ)[3]  # 1 from the first usable


# ----------------------------------------------------------------- O5.6 ---

def test_direction_tools():
    rng = np.random.default_rng(1)
    t_true = np.array([0.6, -0.48, 0.64])
    t_true /= np.linalg.norm(t_true)
    pts = np.array([1.0, 2.0, 3.0]) + np.outer(np.linspace(-2, 2, 9), t_true)
    t = PA.tls_direction(pts)
    assert abs(abs(t @ t_true) - 1) < 1e-12
    to = PA.orient(t, pts[0], pts[-1])
    assert to @ (pts[-1] - pts[0]) >= 0 and abs(to @ t_true - 1) < 1e-12
    to2 = PA.orient(t, pts[-1], pts[0])
    assert to2 @ (pts[0] - pts[-1]) >= 0
    theta, phi, y_hat, e_u = PA.angles(to)
    assert abs(theta - math.atan2(to[1], to[0])) < 1e-15
    assert abs(phi - math.asin(abs(to[2]))) < 1e-15
    assert_allclose(y_hat, [-math.sin(theta), math.cos(theta)], atol=1e-15)
    assert_allclose(e_u, [math.cos(theta), math.sin(theta)], atol=1e-15)
    s = np.array([0.0, 1.0, 2.0, 2.5, 4.0, 4.1])
    assert list(PA.window(s, 2, 1.0)) == [1, 2, 3]
    assert list(PA.window(s, 4, 0.1)) == [4, 5]
    assert list(PA.window(s, 0, 2.0)) == [0, 1, 2]      # closed at |s_j - s_i| = half


# ----------------------------------------------------------------- O5.7 ---

def seg_dist(px, py, a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = b - a
    L2 = d @ d
    t = np.clip(((px - a[0]) * d[0] + (py - a[1]) * d[1]) / L2, 0, 1) if L2 > 0 else 0.0
    return np.hypot(px - (a[0] + t * d[0]), py - (a[1] + t * d[1]))


def test_background_mask_and_median():
    rng = np.random.default_rng(4)
    px, py = rng.uniform(-3, 3, 500), rng.uniform(-3, 3, 500)
    a, b = (0.3, -0.2), (1.7, 1.1)
    assert_allclose(BG.segment_distance(px, py, a, b), seg_dist(px, py, a, b), atol=1e-12)
    xyz = np.array([[10.0, 10.0, 0.0], [12.0, 11.0, 0.3], [13.0, 14.0, 0.6]])
    rad = np.array([0.3, 0.6, 0.2])
    left, top, shape = 70, 75, (60, 70)
    mask = BG.mask_near_branch(shape, left, top, PX, xyz, rad, 1.0)
    rows, cols = np.mgrid[0:shape[0], 0:shape[1]]
    X, Y = (left + cols) * PX, (top + rows) * PX
    exp = np.zeros(shape, bool)
    for j in range(2):
        R_ = max(rad[j], rad[j + 1]) + 1.0
        exp |= seg_dist(X, Y, xyz[j, :2], xyz[j + 1, :2]) <= R_
    # pixels within 1e-9 um of the mask's edge are not judged
    D = np.min([seg_dist(X, Y, xyz[j, :2], xyz[j + 1, :2]) - (max(rad[j], rad[j + 1]) + 1.0)
                for j in range(2)], axis=0)
    firm = np.abs(D) > 1e-9
    assert_array_equal(mask[firm], exp[firm])
    plane = rng.uniform(0, 255, shape)
    Bv, frac, ok = BG.masked_median(plane, exp, 0.2)
    assert Bv == np.median(plane[~exp]) and abs(frac - (~exp).mean()) < 1e-15
    assert ok == ((~exp).mean() >= 0.2)
    Bn, fr, ok2 = BG.masked_median(plane, np.ones(shape, bool), 0.2)
    assert np.isnan(Bn) and fr == 0 and not ok2


# ----------------------------------------------------------------- O5.8 ---

def test_node_request_covers_the_pilot_block_and_K():
    n = 9
    xyz = np.column_stack([20.0 + 1.18 * np.arange(n), 30.0 + 0.2 * np.arange(n), 0.05 * np.arange(n)])
    br = NP.Branch.from_points(xyz, np.full(n, 0.4))
    for i in (0, 4, 8):
        left, top, width, height, k_lo, k_hi = NP.node_request(br, i, CFG)
        k_swc = int(round(xyz[i, 2] / DZ))
        assert k_lo <= k_swc - M.iter_planes_half and k_hi >= k_swc + M.iter_planes_half
        x0, y0 = xyz[i, :2]
        half = M.block_half_um
        assert left * PX <= x0 - half + PX and (left + width - 1) * PX >= x0 + half - PX
        assert top * PX <= y0 - half + PX and (top + height - 1) * PX >= y0 + half - PX
        assert width > 0 and height > 0 and k_hi >= k_lo
