"""Spec-derived tests of Block 40 (the image-based clearness screen; tester, 2026-11-22).

Written from specs/SPEC.md (Block 40 and its amendment 1, D-194, D-198) before the code was read. Expected
values are closed forms (Gaussian and half-Gaussian dips), hand-made samples, or the independent reference
below; the code under test is only checked.

  ref_features   the features (a)-(e) of one smoothed dip, written from Block 40's text: the central extreme
                 in the window, the outward walk to f D_c with linear interpolation between the last sample
                 above and the first at or below, W, c, the skew S over skew_levels, the excess outside E.
  bilinear_read  Block 5's reading rule (an earlier tester's, tests/spec/nc_helpers.py).

Pure ASCII.
"""
from __future__ import annotations

import dataclasses
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.ndimage import gaussian_filter1d

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
for _p in (WS / "src", WS / "scripts", WS / "tests" / "smoke", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from nc_helpers import bilinear_read  # noqa: E402  (independent Block 5 read, earlier tester run)

from allen_diameter import config as C  # noqa: E402
from allen_diameter.analysis import clearness as CL  # noqa: E402

FWHM_SD = 2.0 * math.sqrt(2.0 * math.log(2.0))
P = 0.1144                    # um per pixel (section 2.1)


def ccfg(**kw):
    q = C.default_config().clearness
    return dataclasses.replace(q, **kw) if kw else q


# ------------------------------------------------------------- reference ---

def _cross(D, v, n, step, level):
    while 0 <= n + step < D.size and D[n + step] > level:
        n += step
    if not 0 <= n + step < D.size:
        return math.nan
    return float(v[n] + (v[n + step] - v[n]) * (D[n] - level) / (D[n] - D[n + step]))


def ref_features(D, v, c0=0.0, window=1.0, levels=(0.75, 0.25), a=1.0):
    D, v = np.asarray(D, float), np.asarray(v, float)
    out = dict(depth_gl=math.nan, fwhm_um=math.nan, centre_um=math.nan, skew=math.nan, outside=math.nan)
    if not np.all(np.isfinite(D)):
        return out
    near = np.flatnonzero(np.abs(v - c0) <= window)
    n_c = int(near[np.argmax(D[near])])
    Dc = float(D[n_c])
    out["depth_gl"] = Dc
    if Dc <= 0:
        return out
    xl, xr = _cross(D, v, n_c, -1, 0.5 * Dc), _cross(D, v, n_c, +1, 0.5 * Dc)
    if not (math.isfinite(xl) and math.isfinite(xr)):
        return out
    W, c = xr - xl, 0.5 * (xl + xr)
    out.update(fwhm_um=W, centre_um=c)
    mids = [0.5 * (_cross(D, v, n_c, -1, f * Dc) + _cross(D, v, n_c, +1, f * Dc)) for f in levels]
    if all(math.isfinite(m) for m in mids):
        out["skew"] = max(abs(m - c) for m in mids) / W
    left, right = v - c <= -a * W, v - c >= a * W
    if left.any() and right.any():
        out["outside"] = float(np.max(np.abs(D[left | right]))) / Dc
    return out


# ------------------------------------------------------- dip_features ---

def test_gaussian_dip():
    v = np.round(np.arange(-5000, 5001) * 0.001, 12)
    s, c0, A = 0.4, 0.13, 50.0
    D = A * np.exp(-0.5 * ((v - c0) / s) ** 2)
    f = CL.dip_features(D, v)
    # step 0.001 um: linear interpolation of a Gaussian at the half depth errs by O(h^2) = 1e-6 um; 1e-5 is the spec's
    assert_allclose(f["fwhm_um"], FWHM_SD * s, atol=1e-5)
    assert_allclose(f["centre_um"], c0, atol=1e-5)
    assert f["skew"] <= 1e-5
    assert_allclose(f["outside"], 1.0 / 16.0, rtol=1e-2)     # 2^(-4 a^2) at a = 1, the first sample outside
    assert_allclose(f["depth_gl"], A, rtol=1e-6)


def test_half_gaussians_by_hand():
    v = np.round(np.arange(-6000, 6001) * 0.001, 12)
    sL, sR, c0, A = 0.3, 0.6, -0.2, 40.0
    D = A * np.where(v < c0, np.exp(-0.5 * ((v - c0) / sL) ** 2), np.exp(-0.5 * ((v - c0) / sR) ** 2))
    f = CL.dip_features(D, v)
    q = lambda frac: math.sqrt(2.0 * math.log(1.0 / frac))    # noqa: E731
    xl, xr = c0 - sL * q(0.5), c0 + sR * q(0.5)
    W, c = xr - xl, 0.5 * (xl + xr)
    S = max(abs(0.5 * ((c0 - sL * q(fr)) + (c0 + sR * q(fr))) - c) for fr in (0.75, 0.25)) / W
    assert_allclose([f["fwhm_um"], f["centre_um"], f["skew"]], [W, c, S], atol=1e-5)
    assert_allclose([f["x_lo_um"], f["x_hi_um"]], [xl, xr], atol=1e-5)


def test_second_dip_outside_gives_its_share():
    v = np.round(np.arange(-5000, 5001) * 0.001, 12)
    D = 50.0 * np.exp(-0.5 * (v / 0.3) ** 2) + 20.0 * np.exp(-0.5 * ((v - 2.0) / 0.2) ** 2)
    f = CL.dip_features(D, v)
    # the second dip's peak is 0.4 of the depth; the main dip's own tail there is e^-22 of it
    assert_allclose(f["outside"], 0.4, atol=1e-6)


def test_width_error_order_two():
    s = 0.37
    errs = []
    for h in (0.08, 0.04, 0.02, 0.01, 0.005):
        v = (np.arange(-int(4.0 / h), int(4.0 / h) + 1) + 0.37) * h        # a grid not aligned with the dip
        D = 30.0 * np.exp(-0.5 * (v / s) ** 2)
        errs.append(abs(CL.dip_features(D, v, window_um=1.0)["fwhm_um"] - FWHM_SD * s))
    orders = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    slope = np.polyfit(np.log2([0.08, 0.04, 0.02, 0.01, 0.005]), np.log2(errs), 1)[0]   # err ~ h^order
    print("W errors", errs, "orders", orders, "fitted order", slope)
    assert 1.5 <= slope <= 2.5


def test_crossing_rule_on_hand_samples():
    v = np.arange(-3.0, 4.0)
    f = CL.dip_features(np.array([0, 1, 3, 10, 3, 1, 0.0]), v)
    assert_allclose([f["x_lo_um"], f["x_hi_um"], f["fwhm_um"]], [-5 / 7, 5 / 7, 10 / 7], rtol=1e-15)
    # a sample exactly at the level is "at or below": the crossing is on it
    f = CL.dip_features(np.array([0, 0, 5, 10, 5, 0, 0.0]), v)
    assert f["x_lo_um"] == -1.0 and f["x_hi_um"] == 1.0
    # never falls to the level on the right before the line ends: no width, the depth stays
    f = CL.dip_features(np.array([0, 1, 3, 10, 9, 8, 7.0]), v)
    assert math.isnan(f["fwhm_um"]) and math.isnan(f["x_hi_um"]) and f["depth_gl"] == 10.0
    # one NaN anywhere: everything NaN
    D = np.array([0, 1, 3, 10, 3, 1, np.nan])
    f = CL.dip_features(D, v)
    assert all(math.isnan(f[k]) for k in ("depth_gl", "fwhm_um", "centre_um", "skew", "outside"))
    # no sample outside on one side: E NaN
    f = CL.dip_features(np.array([0, 1, 3, 10, 3, 1, 0.0]), v, outside_fwhm=3.0)
    assert math.isnan(f["outside"]) and math.isfinite(f["fwhm_um"])


def test_against_reference_on_random_dips():
    rng = np.random.default_rng(3)
    v = np.round(np.arange(-26, 27) * 0.1144, 10)
    for _ in range(300):
        D = rng.uniform(5, 60) * np.exp(-0.5 * ((v - rng.uniform(-0.6, 0.6)) / rng.uniform(0.15, 0.8)) ** 2)
        D += rng.uniform(0, 30) * np.exp(-0.5 * ((v - rng.uniform(-3, 3)) / rng.uniform(0.1, 0.6)) ** 2)
        D += rng.normal(0, 1.0, v.size)
        D = gaussian_filter1d(D, 1.0, mode="nearest")
        got, want = CL.dip_features(D, v), ref_features(D, v)
        for k in want:
            a, b = got[k], want[k]
            assert (math.isnan(a) and math.isnan(b)) or abs(a - b) <= 1e-12 * max(1.0, abs(b)), (k, a, b)


def test_mirror_and_scale():
    v = np.round(np.arange(-40, 41) * 0.05, 10)
    D = 30.0 * np.exp(-0.5 * ((v - 0.2) / 0.35) ** 2) + 6.0 * np.exp(-0.5 * ((v + 1.4) / 0.3) ** 2)
    f, m, s = CL.dip_features(D, v), CL.dip_features(D[::-1], v), CL.dip_features(3.0 * D, v)
    assert_allclose([m["fwhm_um"], m["skew"], m["outside"]], [f["fwhm_um"], f["skew"], f["outside"]], atol=1e-12)
    assert_allclose(m["centre_um"], -f["centre_um"], atol=1e-12)
    assert_allclose([s["fwhm_um"], s["centre_um"], s["skew"], s["outside"]],
                    [f["fwhm_um"], f["centre_um"], f["skew"], f["outside"]], atol=1e-12)
    assert_allclose(s["depth_gl"], 3.0 * f["depth_gl"], rtol=1e-15)


def test_dip_features_refusals():
    v = np.arange(-3.0, 4.0)
    D = np.array([0, 1, 3, 10, 3, 1, 0.0])
    for kw in (dict(window_um=0.0), dict(window_um=np.nan), dict(skew_levels=(0.25, 0.75)),
               dict(skew_levels=(0.75,)), dict(outside_fwhm=0.0), dict(centre_um=np.inf)):
        with pytest.raises(ValueError):
            CL.dip_features(D, v, **kw)
    with pytest.raises(ValueError):
        CL.dip_features(D, v[::-1])
    with pytest.raises(ValueError):
        CL.dip_features(D[:5], v)


# ------------------------------------------------- strips and offsets ---

def test_strip_profiles_and_offsets():
    rng = np.random.default_rng(0)
    plane = rng.uniform(100, 200, (90, 110))
    left, top = 37, -12
    th = 0.35
    y_hat, e_u = np.array([-math.sin(th), math.cos(th)]), np.array([math.cos(th), math.sin(th)])
    origin = np.array([(left + 55) * P, (top + 45) * P])
    v = np.arange(-20, 21) * P
    u = CL.along_offsets(2.0, 0.5)
    assert_allclose(u, np.arange(-4, 5) * 0.5, rtol=0, atol=0)
    got = CL.strip_profiles(plane, left, top, P, origin, y_hat, e_u, v, 4, P, u)
    want = np.stack([bilinear_read(plane, left, top, P, origin + uu * e_u, y_hat, e_u, v, 4, P) for uu in u])
    assert got.shape == (9, v.size)
    assert_allclose(got, want, rtol=0, atol=1e-12)    # the same rule (map_coordinates, order 1): roundoff
    far = CL.strip_profiles(plane, left, top, P, origin, y_hat, e_u, np.arange(-60, 61) * P, 0, P, [0.0])
    assert np.isnan(far).any() and np.isfinite(far).any()          # a read that leaves the patch is NaN
    assert list(CL.along_offsets(0.0, 0.5)) == [0.0]
    assert_allclose(CL.along_offsets(1.2, 0.5), [-1.0, -0.5, 0.0, 0.5, 1.0])
    for bad in ((-1.0, 0.5), (1.0, 0.0), (np.nan, 0.5)):
        with pytest.raises(ValueError):
            CL.along_offsets(*bad)


# ------------------------------------------------------------ screen_node ---

def band_patch(ks, depth=50.0, B=200.0, s=0.35, extra=None, H=120, W=120):
    """(block (n_planes, H, W), left, top): a straight band along x (a Gaussian dip across y of SD s um and
    the given depth, wider and fainter off focus), and optionally `extra(X, Y, k)` gl subtracted."""
    left, top = -60, -60
    X, Y = np.meshgrid((left + np.arange(W)) * P, (top + np.arange(H)) * P)
    out = []
    for k in ks:
        sk = math.sqrt(s ** 2 + (0.15 * k) ** 2)
        img = B - depth * (s / sk) * np.exp(-0.5 * (Y / sk) ** 2)
        if extra is not None:
            img = img - extra(X, Y, k)
        out.append(img)
    return np.array(out), left, top


def line_geometry():
    return (np.array([0.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0]), np.arange(-26, 27) * P, 4, P)


def ref_verdict(block, ks, k_star, B, sigma, left, top, q):
    o, y_hat, e_u, v, n_avg, step = line_geometry()
    j = int(np.argmin(np.abs(np.asarray(ks) - k_star)))
    E, S = [], []
    for i, k in enumerate(ks):
        if abs(k - ks[j]) > q.focus_half_planes:
            continue
        prof = bilinear_read(block[i], left, top, P, o, y_hat, e_u, v, n_avg, step)
        D = gaussian_filter1d(B[i] - prof, q.smooth_samples, mode="nearest")
        f = ref_features(D, v, 0.0, q.window_um, q.skew_levels, q.outside_fwhm)
        E.append(f["outside"])
        S.append(f["skew"])
    prof = bilinear_read(block[j], left, top, P, o, y_hat, e_u, v, n_avg, step)
    Dc = ref_features(gaussian_filter1d(B[j] - prof, q.smooth_samples, mode="nearest"), v, 0.0, q.window_um,
                      q.skew_levels, q.outside_fwhm)["depth_gl"]
    Q = max(max(E) / q.outside_max, max(S) / q.skew_max)
    return Q, Q <= 1.0 and Dc >= q.min_depth_sd * sigma, Dc / sigma


def _screen(block, ks, k_star, B, sigma, left, top, q, profiles=None):
    o, y_hat, e_u, v, n_avg, step = line_geometry()
    return CL.screen_node(block, ks, k_star, B, sigma, left, top, P, o, y_hat, e_u, v, n_avg, step, q,
                          profiles_gl=profiles)


def test_screen_clean_band_by_hand():
    ks = np.arange(-4, 5)
    block, left, top = band_patch(ks)
    B = np.full(ks.size, 200.0)
    q = ccfg()
    res = _screen(block, ks, 0, B, 1.0, left, top, q)
    Q, clean, dsd = ref_verdict(block, ks, 0, B, 1.0, left, top, q)
    assert res.clean is True and clean is True and res.reasons == ()
    assert_allclose(res.score, Q, rtol=1e-12)
    assert_allclose(res.depth_sd, dsd, rtol=1e-12)
    assert set(res.ratios) == set(CL.LINE_CRITERIA)
    assert list(res.focus) == [abs(k) <= 1 for k in ks]
    assert res.strips_gl.shape == (9, 53) and res.u_um.shape == (9,)
    assert res.width_range < 0.02 and res.depth_range < 0.02 and res.centre_range < 0.02
    # the recorded line given equals the line read from the block
    o, y_hat, e_u, v, n_avg, step = line_geometry()
    prof = np.stack([bilinear_read(block[i], left, top, P, o, y_hat, e_u, v, n_avg, step) for i in range(ks.size)])
    res2 = _screen(block, ks, 0, B, 1.0, left, top, q, profiles=prof)
    assert_allclose(res2.score, res.score, rtol=1e-12)
    # k* not recorded: the nearest recorded plane
    ks_odd = np.array([-4, -2, 0, 2, 4])
    res3 = _screen(block[::2], ks_odd, 1.2, B[::2], 1.0, left, top, q)
    assert res3.k_index == 3                    # 2 is nearest to 1.2
    assert list(res3.focus) == [False, False, False, True, False]


def test_screen_rejections_and_the_floor():
    ks = np.arange(-4, 5)
    B = np.full(ks.size, 200.0)
    q = ccfg()
    nb = lambda X, Y, k: 25.0 * np.exp(-0.5 * ((Y - 1.4) / 0.35) ** 2)    # noqa: E731  half the depth, 1.4 um aside
    block, left, top = band_patch(ks, extra=nb)
    res = _screen(block, ks, 0, B, 1.0, left, top, q)
    Q, clean, _ = ref_verdict(block, ks, 0, B, 1.0, left, top, q)
    assert res.clean is False and clean is False and "outside" in res.reasons
    assert_allclose(res.score, Q, rtol=1e-12)
    # the depth floor: sigma large enough that the central depth is below 10 SDs -> "faint" alone
    block, left, top = band_patch(ks)
    res = _screen(block, ks, 0, B, 6.0, left, top, q)
    assert res.reasons == ("faint",) and res.clean is False
    # judging along: CRITERIA keys, and judging more never lowers the score
    blob = lambda X, Y, k: 0.83 * 50.0 * np.exp(-0.5 * (((X - 1.5) / 0.3) ** 2 + ((Y - 0.35) / 0.3) ** 2))  # noqa: E731
    block, left, top = band_patch(ks, extra=blob)
    a = _screen(block, ks, 0, B, 1.0, left, top, q)
    b = _screen(block, ks, 0, B, 1.0, left, top, ccfg(judge_along=True))
    assert set(b.ratios) == set(CL.CRITERIA) and b.score >= a.score
    assert np.array_equal(a.strips_gl, b.strips_gl)          # the lines along are computed either way
    # a dip that never returns to half depth on one side: score infinite, not_finite
    ramp = lambda X, Y, k: np.where(Y > 0, 40.0, 0.0)      # noqa: E731
    block, left, top = band_patch(ks, extra=ramp)
    r = _screen(block, ks, 0, B, 1.0, left, top, q)
    assert math.isinf(r.score) and "not_finite" in r.reasons


def test_screen_inputs_unchanged_determinism_and_refusals():
    ks = np.arange(-2, 3)
    block, left, top = band_patch(ks)
    B = np.full(ks.size, 200.0)
    b0, B0 = block.copy(), B.copy()
    q = ccfg()
    r1 = _screen(block, ks, 0, B, 1.0, left, top, q)
    r2 = _screen(block, ks, 0, B, 1.0, left, top, q)
    assert np.array_equal(block, b0) and np.array_equal(B, B0)
    assert r1.score == r2.score and np.array_equal(r1.strips_gl, r2.strips_gl)
    for bad in (dict(B=np.full(5, -1.0)), dict(B=np.full(4, 200.0)), dict(sigma=0.0), dict(sigma=np.nan),
                dict(block=block[:4])):
        kw = dict(block=block, ks=ks, k_star=0, B=B, sigma=1.0, left=left, top=top, q=q)
        kw.update(bad)
        with pytest.raises(ValueError):
            _screen(**kw)


def test_lines_leaving_the_patch_matter_only_when_judged():
    """Amendment 1: the lines along the dendrite enter the verdict only with judge_along; lines that leave the
    patch are "not_finite" only then. A patch 40 px wide (+-2.3 um along the axis): the lines at u = +-2 um with
    the read's +-0.46 um leave it, the measuring line does not."""
    ks = np.arange(-2, 3)
    block, left, top = band_patch(ks, W=40)
    left = -20
    B = np.full(ks.size, 200.0)
    a = _screen(block, ks, 0, B, 1.0, left, top, ccfg())
    b = _screen(block, ks, 0, B, 1.0, left, top, ccfg(judge_along=True))
    assert np.isnan(a.strips_gl).any() and np.isfinite(a.strips_gl[a.u_um.size // 2]).all()
    assert a.clean is True and "not_finite" not in a.reasons
    assert b.clean is False and "not_finite" in b.reasons and math.isinf(b.score)
