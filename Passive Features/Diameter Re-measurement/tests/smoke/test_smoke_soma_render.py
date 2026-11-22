"""Smoke test for the soma's forward model (model/soma_render.py) -- Block 45 in specs/SPEC.md (D-215, D-216, D-218,
D-219).

Oracles (SPEC Block 45): (T1) the ellipsoid's column interval -- a sphere's chord, no dependence on phi when a = b,
the chords along the rotated axes; (T2) the sampled Gaussian's transfer against the DFT of the normalised sampled
Gaussian on a periodic grid, and 1 at nu = 0; (T3) the light of every plane equals the light the soma absorbs;
(T4) a direct reference with scipy.ndimage.gaussian_filter per (slab, plane); (T5) convergence in the samples per
pixel and in the slab thickness, the differences at the defaults reported; (T6) the far slope: the configured slope
gives the result of slope None, a larger one keeps the light and widens the far planes; (T7) tau <= 1, mu = 0,
shapes, dtypes, the planes' order, the light direction, determinism, the ValueErrors; (T8) the speed and peak memory
of the fit's render, measured in a fresh process.

Checks, from the strongest oracle to the weakest
    test_known_answer   analytic or tabulated solution
    test_reference      independent implementation (library function, or
                        brute force on a tiny input)
    test_convergence    discretisations only: observed order p in
                        e(h) ~ C * h**p, where e(h) is the error at step h
    test_invariants     conservation laws, symmetries, bounds, normalisation
    test_contract       output shapes, dtypes and finiteness
    test_determinism    identical output for an identical seed
    test_edge_cases     degenerate or boundary inputs named in the spec

Run
    python tests/smoke/test_smoke_soma_render.py
        Prints the environment and a PASS / FAIL / ERROR / TODO / SKIP table,
        then the full message of every failure. Exit code 1 if any check
        failed, errored or is still TODO.
    python -m pytest tests/smoke/test_smoke_soma_render.py -q
        The same checks through pytest (TODO checks show up as failures).

Debug
    A failed check_close() saves both arrays to
        tests/smoke/artifacts/<check name>.npz
    Inspect it with
        d = np.load("tests/smoke/artifacts/<check name>.npz")
        diff = d["actual"] - d["expected"]
        np.unravel_index(np.argmax(np.abs(diff)), diff.shape)

Add a check
    Define a function test_<something>() with no arguments that raises
    AssertionError on failure; the runner below and pytest both pick it up.

Keep this file pure ASCII (no Greek letters, dashes or arrows outside ASCII)
so it survives transfer to the cluster.
"""
from __future__ import annotations

import dataclasses
import importlib.metadata
import math
import os
import platform
import re
import subprocess
import sys
import time
import traceback
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal  # noqa: F401

WS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(WS, "src")
sys.path.insert(0, SRC)
from allen_diameter.config import RendererConfig, SomaFitConfig   # noqa: E402
from allen_diameter.model import geometry                         # noqa: E402
from allen_diameter.model import soma_render as SR                # noqa: E402
from allen_diameter.model.kernel import sigma_r                    # noqa: E402
from allen_diameter.model.render import block_fine_grid, pixel_integrate   # noqa: E402

SEED = 20261122  # no randomness in the block; kept for the runner's header
REPORT_PACKAGES = ("numpy", "scipy")
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
_SAVED_ARTIFACTS = []  # files written by check_close during this run

P = 0.1144  # um: the pilot's pixel (AcquisitionConfig.res0_um)


# --------------------------------------------------------------- helpers ---

def make_rng(seed: int = SEED) -> np.random.Generator:
    """Fresh generator, to be passed explicitly to the code under test."""
    return np.random.default_rng(seed)


def check_close(name, actual, expected, rtol=1e-12, atol=0.0):
    """numpy.testing.assert_allclose that keeps the evidence on failure.

    On failure both arrays are saved to ARTIFACT_DIR/<name>.npz before the
    AssertionError propagates. Justify rtol and atol at the call site
    (float64 roundoff, discretisation-error estimate, statistical spread).
    """
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
    """Least-squares slope p of log e(h) = log C + p * log h.

    h: step sizes (any order); err: positive errors e(h) at those steps.
    """
    h = np.asarray(h, dtype=float)
    err = np.asarray(err, dtype=float)
    if np.any(err <= 0.0):
        raise ValueError("observed_order needs strictly positive errors")
    slope, _intercept = np.polyfit(np.log(h), np.log(err), 1)
    return float(slope)


def _cfgs(f=1, dzeta=0.14, **renderer):
    """(RendererConfig, SomaFitConfig): the defaults, the soma's samples per pixel and slab thickness replaced."""
    return (dataclasses.replace(RendererConfig(), **renderer),
            dataclasses.replace(SomaFitConfig(), fine_factor=f, dzeta_um=dzeta))


def _square_crop(half_px):
    """A crop of 2 half_px + 1 px a side whose middle pixel is centred on (0, 0) um."""
    return (-half_px, -half_px, 2 * half_px + 1, 2 * half_px + 1)


def _columns_plain(X, Y, th):
    """The column interval as SPEC Block 45 writes it (u, v rotated; q = 1 - (u/a)^2 - (v/b)^2): the reference."""
    x0, y0, z0, a, b, c, phi, _mu = th
    u = (X - x0) * math.cos(phi) + (Y - y0) * math.sin(phi)
    v = -(X - x0) * math.sin(phi) + (Y - y0) * math.cos(phi)
    q = 1.0 - (u / a) ** 2 - (v / b) ** 2
    inside = q > 0
    half = c * np.sqrt(np.where(inside, q, 0.0))
    return np.where(inside, z0 - half, z0), np.where(inside, z0 + half, z0), inside


def _samples(crop, f):
    """The crop's samples (render.block_fine_grid, no padding): (X, Y) (ny, nx) um and the spacing h um."""
    grid, _ = block_fine_grid(crop[0], crop[1], crop[2], crop[3], P, f)
    X, Y = grid.mesh()
    return X, Y, grid.h


def _s_jk(th, z_planes, rcfg, dzeta, slope=None):
    """The kernel widths s_jk (J, K) um of SPEC Block 45, from Block 2's slabs and Block 4's sigma_r."""
    z0, c = th[2], th[5]
    zeta = geometry.slab_centres(z0 - c, z0 + c, dzeta)
    rk = rcfg if slope is None else dataclasses.replace(rcfg, kernel_continuation_slope=float(slope))
    return zeta, sigma_r(zeta[:, None] - np.asarray(z_planes, dtype=float)[None, :], rk)


def _crop_holding_blur(th, z_planes, rcfg, dzeta, slope=None, n_sigma=7.0):
    """A square crop extending at least n_sigma s_max beyond the soma's widest semi-axis (T3, T6)."""
    _zeta, s = _s_jk(th, z_planes, rcfg, dzeta, slope)
    reach = max(th[3], th[4]) + n_sigma * float(s.max()) + math.hypot(th[0], th[1])
    return _square_crop(int(math.ceil(reach / P)) + 1), float(s.max())


def _light_absorbed(crop, f, th):
    """h^2 sum over the crop's samples of 1 - exp(-mu (z_hi - z_lo)): the light the soma absorbs on the crop (um^2)."""
    X, Y, h = _samples(crop, f)
    zlo, zhi, _ins = _columns_plain(X, Y, th)
    return h * h * float(np.sum(1.0 - np.exp(-th[7] * (zhi - zlo))))


def _light_per_plane(tau):
    """p^2 sum over the pixels of 1 - tau_k, per plane (um^2)."""
    return P * P * (1.0 - tau).sum(axis=(1, 2))


def _second_moment(tau, crop, x0, y0):
    """The light-weighted second moment of 1 - tau_k about (x0, y0), per plane (um^2), at the pixel centres."""
    left, top, width, height = crop
    x = (left + np.arange(width)) * P
    y = (top + np.arange(height)) * P
    r2 = (x[None, :] - x0) ** 2 + (y[:, None] - y0) ** 2
    w = 1.0 - tau
    return (w * r2[None]).sum(axis=(1, 2)) / w.sum(axis=(1, 2))


def _expect_value_error(label, fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("no ValueError: %s" % label)


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    """(T1) ellipsoid_columns: a sphere's chord 2 sqrt(R^2 - rho^2) and midpoint z0 [1e-12 um]; for a = b the result
    does not depend on phi [exact]; along (cos phi, sin phi) at distance r the chord 2 c sqrt(1 - r^2/a^2), across it
    2 c sqrt(1 - r^2/b^2) [1e-12 um], for a > b and a < b; outside, inside False and z_lo = z_hi = z0 [exact];
    broadcasting, dtypes."""
    R, x0, y0, z0 = 3.7, 1.3, -0.4, 2.9
    th = [x0, y0, z0, R, R, R, 0.83, 0.4]
    rho = np.linspace(0.0, 0.99 * R, 41)           # away from the rim, where sqrt amplifies roundoff of q
    ang = np.linspace(-math.pi, math.pi, 13)
    RHO, ANG = np.meshgrid(rho, ang)
    zlo, zhi, ins = SR.ellipsoid_columns(x0 + RHO * np.cos(ANG), y0 + RHO * np.sin(ANG), th)
    assert ins.all()
    # 1e-12 um: a few ulp of q (~1e-16) amplified by 1/(2 sqrt(q)) <= 3.6 at rho = 0.99 R, times 2 R
    check_close("T1 sphere chord", zhi - zlo, 2.0 * np.sqrt(R * R - RHO * RHO), rtol=0, atol=1e-12)
    check_close("T1 sphere midpoint", 0.5 * (zlo + zhi), np.full(zlo.shape, z0), rtol=0, atol=1e-12)
    out_r = np.linspace(1.001 * R, 3.0 * R, 17)
    zlo, zhi, ins = SR.ellipsoid_columns(x0 + out_r * math.cos(0.4), y0 + out_r * math.sin(0.4), th)
    assert not ins.any() and np.all(zlo == z0) and np.all(zhi == z0), (zlo, zhi)

    pts = np.linspace(-3.0, 3.0, 23)
    X, Y = np.meshgrid(pts + 0.3, pts - 0.2)
    ref = SR.ellipsoid_columns(X, Y, [0.1, -0.15, 1.0, 2.5, 2.5, 1.7, 0.0, 0.3])
    for phi in (0.3, 1.1, -2.0, 3.0, 17.3):
        got = SR.ellipsoid_columns(X, Y, [0.1, -0.15, 1.0, 2.5, 2.5, 1.7, phi, 0.3])
        for name, g, r in zip(("z_lo", "z_hi", "inside"), got, ref):
            assert_array_equal(g, r, err_msg="T1 a = b, phi %g: %s" % (phi, name))

    phi, c, xc, yc, zc = 0.7, 2.2, -0.6, 0.9, -1.4
    e1 = np.array([math.cos(phi), math.sin(phi)])
    e2 = np.array([-math.sin(phi), math.cos(phi)])
    for a, b in ((3.0, 1.8), (1.8, 3.0)):
        th = [xc, yc, zc, a, b, c, phi, 1.0]
        for e, axis in ((e1, a), (e2, b)):
            r = np.linspace(0.0, 0.99 * axis, 31)
            zlo, zhi, ins = SR.ellipsoid_columns(xc + r * e[0], yc + r * e[1], th)
            assert ins.all()
            check_close("T1 chord along an axis (a %g, b %g, semi-axis %g)" % (a, b, axis), zhi - zlo,
                        2.0 * c * np.sqrt(1.0 - (r / axis) ** 2), rtol=0, atol=1e-12)   # as above, 2 c instead of 2 R

    zlo, zhi, ins = SR.ellipsoid_columns(np.linspace(-2, 2, 5)[:, None], np.linspace(-2, 2, 7)[None, :],
                                         [0.0, 0.0, 0.0, 1.5, 1.0, 1.0, 0.2, 0.5])
    assert zlo.shape == zhi.shape == ins.shape == (5, 7), (zlo.shape, zhi.shape, ins.shape)
    assert zlo.dtype == np.float64 and zhi.dtype == np.float64 and ins.dtype == np.bool_
    zlo, zhi, ins = SR.ellipsoid_columns(0.0, 0.0, [0.0, 0.0, 0.5, 1.0, 1.0, 2.0, 0.0, 0.5])
    assert np.ndim(zlo) == 0 and bool(ins) and float(zlo) == -1.5 and float(zhi) == 2.5   # the centre: z0 -+ c


def test_known_answer_transfer():
    """(T2) sampled_gaussian_transfer equals the DFT (numpy.fft.fft) of the normalised sampled Gaussian built on a
    periodic grid of 4096 samples, for s/h from 0.2 to 20 [1e-12: the aliases left out are below 4.5e-13], and is 1
    at nu = 0 [1e-15]; nu's shape is kept."""
    N, h = 4096, P
    nu = np.fft.fftfreq(N, d=h)
    n = np.arange(N)
    n = np.minimum(n, N - n)                          # periodic distance from sample 0
    worst = 0.0
    for r in (0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.833, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0):
        s = r * h
        w = np.exp(-(n * h) ** 2 / (2.0 * s * s))
        w /= w.sum()
        dft = np.fft.fft(w)
        assert np.max(np.abs(dft.imag)) < 1e-14, "the kernel is even: a real DFT"
        got = SR.sampled_gaussian_transfer(s, nu, h)
        check_close("T2 transfer vs DFT, s/h = %g" % r, got, dft.real, rtol=0, atol=1e-12)
        worst = max(worst, float(np.max(np.abs(got - dft.real))))
        at0 = float(SR.sampled_gaussian_transfer(s, 0.0, h))
        assert abs(at0 - 1.0) <= 1e-15, ("T2 transfer at nu = 0", r, at0)
    print("  T2: max |transfer - DFT| over s/h in [0.2, 20]: %.1e" % worst)
    assert SR.sampled_gaussian_transfer(0.1, nu.reshape(64, 64), h).shape == (64, 64)
    w0 = SR.sampled_gaussian_transfer(0.1, 0.3, h)
    assert isinstance(w0, np.ndarray) and w0.shape == () and w0.dtype == np.float64, (type(w0), np.shape(w0))


def test_reference():
    """(T4) against a direct reference on small cases: on the same samples, per (slab, plane),
    scipy.ndimage.gaussian_filter of Delta A_j (written out here from Block 2's functions and the plain column
    interval) with sigma s_jk / h, mode "constant", truncate rcfg.direct_truncate; summed, 1 minus, pixel-integrated.
    Case 1: f = 1 with SDs below one sample among the pairs (the in-focus 0.08 um on 0.1144 um samples); case 2:
    f = 2, light toward -z, a slope 1.1. [1e-9 absolute in tau: the wrap below e^-18 of a slab's peak, the
    truncation's tail below 1e-15, the aliases below 1e-12]."""
    from scipy.ndimage import gaussian_filter
    th = [0.04, -0.03, 2.3, 1.3, 0.9, 0.8, 0.6, 0.7]
    zk = 2.3 + np.array([-1.1, -0.25, 0.07, 0.6])
    crop = (-22, -19, 44, 38)
    for f, light, slope in ((1, +1, None), (2, -1, 1.1)):
        rcfg, scfg = _cfgs(f=f)
        tau = SR.SomaRenderer(crop, P, zk, rcfg, scfg, light).render(th, slope)
        X, Y, h = _samples(crop, f)
        zlo, zhi, _ins = _columns_plain(X, Y, th)
        zeta, s = _s_jk(th, zk, rcfg, scfg.dzeta_um, slope)
        dz, mu = scfg.dzeta_um, th[7]
        dA = [geometry.transmitted_before(zlo, zhi, zj, dz, mu, light)
              * (1.0 - np.exp(-geometry.slab_absorbance(zlo, zhi, zj, dz, mu))) for zj in zeta]
        ref = np.stack([pixel_integrate(1.0 - sum(gaussian_filter(dA[j], s[j, k] / h, mode="constant", cval=0.0,
                                                                  truncate=rcfg.direct_truncate)
                                                  for j in range(zeta.size)), f)
                        for k in range(zk.size)])
        if f == 1:
            assert np.any(s < h), "the case must hold SDs below one sample"
        err = float(np.max(np.abs(tau - ref)))
        print("  T4: f %d, light %+d, slope %s: %d slabs x %d planes, s/h %.2f..%.1f, max |tau - direct| %.1e"
              % (f, light, slope, zeta.size, zk.size, float(s.min()) / h, float(s.max()) / h, err))
        check_close("T4 direct reference (f %d)" % f, tau, ref, rtol=0, atol=1e-9)


def test_reference_full_product():
    """The accumulation skips the rows and columns where a transfer is exactly 0.0 in float64: with every row and
    column kept instead (the full product w_s(nu_y) w_s(nu_x) on the whole box), the planes are bit-identical. The
    case has planes far from the soma, so that wide kernels (s from 0.08 to 8.87 um at slope 0.9) leave most of the
    box at exactly 0."""
    th = [0.02, -0.04, 0.0, 1.6, 1.2, 1.0, 0.7, 0.9]
    zk = np.array([-9.0, -4.0, -0.3, 0.2, 5.5])
    rcfg, scfg = _cfgs()
    crop = (-30, -26, 61, 53)
    trimmed = SR.SomaRenderer(crop, P, zk, rcfg, scfg, +1).render(th, 0.9)
    original = SR._trimmed_transfer

    def full(s, nu_y, nu_x, h):
        wy, wx, _rows, _cx = original(s, nu_y, nu_x, h)
        return wy, wx, (slice(0, wy.size),), wx.size

    n_trimmed = []

    def counting(s, nu_y, nu_x, h):
        out = original(s, nu_y, nu_x, h)
        n_trimmed.append(out[3] < out[1].size)
        return out

    try:
        SR._trimmed_transfer = full
        whole = SR.SomaRenderer(crop, P, zk, rcfg, scfg, +1).render(th, 0.9)
        SR._trimmed_transfer = counting
        SR.SomaRenderer(crop, P, zk, rcfg, scfg, +1).render(th, 0.9)
    finally:
        SR._trimmed_transfer = original
    assert any(n_trimmed), "the case must exercise the skipped zeros"
    assert_array_equal(trimmed, whole, err_msg="the skipped zeros must not change the planes")


def test_convergence():
    """(T5) a small soma: the largest difference in tau against f = 8 decreases over f = 1, 2, 4 (at the default
    slab thickness 0.14 um); against dzeta = 0.035 um it decreases over dzeta = 0.56, 0.28, 0.14 um (at f = 1). The
    differences at the defaults (f = 1, dzeta = 0.14 um) are reported: the discretisation error of D-219.1."""
    th = [0.01, -0.02, 0.3, 1.1, 0.8, 0.7, 0.4, 0.6]
    zk = 0.3 + np.array([-1.0, -0.3, 0.0, 0.45, 1.2])
    crop = _square_crop(15)

    def tau(f, dzeta):
        rcfg, scfg = _cfgs(f=f, dzeta=dzeta)
        return SR.SomaRenderer(crop, P, zk, rcfg, scfg, +1).render(th)

    fine = tau(8, 0.14)
    e_f = [float(np.max(np.abs(tau(f, 0.14) - fine))) for f in (1, 2, 4)]
    thin = tau(1, 0.035)
    e_dz = [float(np.max(np.abs(tau(1, dz) - thin))) for dz in (0.56, 0.28, 0.14)]
    print("  T5: max |tau(f) - tau(8)| at f = 1, 2, 4: %.2e, %.2e, %.2e (dzeta 0.14 um)" % tuple(e_f))
    print("  T5: max |tau(dzeta) - tau(0.035 um)| at dzeta = 0.56, 0.28, 0.14 um: %.2e, %.2e, %.2e (f 1)" % tuple(e_dz))
    print("  T5: at the defaults (f 1, dzeta 0.14 um): %.2e against f 8, %.2e against dzeta 0.035 um" % (e_f[0], e_dz[2]))
    assert e_f[0] > e_f[1] > e_f[2] > 0.0, e_f
    assert e_dz[0] > e_dz[1] > e_dz[2] > 0.0, e_dz


def test_invariants():
    """(T3) light per plane: p^2 sum_pixels (1 - tau_k) = h^2 sum_samples (1 - exp(-mu (z_hi - z_lo))) for every
    plane, at f = 1 and f = 2, on a crop extending at least 7 s_max beyond the soma [1e-9 relative: the light outside
    the crop below 1e-11]. (T6) slope = the configured kernel_continuation_slope gives the result of slope None
    [exact]; a larger slope keeps each plane's light (T3) and increases the light-weighted second moment of 1 - tau_k
    about (x0, y0) for every plane farther than c + 1 um from z0. Also: light toward -z with the planes mirrored about
    z0 gives the same planes [1e-12: roundoff of the reordered slab sum], and the light direction matters."""
    th = [0.03, -0.05, 1.9, 1.5, 1.1, 1.0, 0.5, 0.8]
    zk = 1.9 + np.array([-2.0, -0.6, 0.0, 0.35, 1.7])
    for f in (1, 2):
        rcfg, scfg = _cfgs(f=f)
        crop, s_max = _crop_holding_blur(th, zk, rcfg, scfg.dzeta_um)
        tau = SR.SomaRenderer(crop, P, zk, rcfg, scfg, +1).render(th)
        light, ref = _light_per_plane(tau), _light_absorbed(crop, f, th)
        print("  T3: f %d, crop %d px, s_max %.2f um: light %.6f um^2, max relative error %.1e"
              % (f, crop[2], s_max, ref, float(np.max(np.abs(light / ref - 1.0)))))
        check_close("T3 light per plane (f %d)" % f, light, np.full(zk.size, ref), rtol=1e-9)

    th = [0.02, -0.03, 0.0, 1.4, 1.0, 0.9, 0.35, 0.6]
    zk = np.array([-3.2, -2.4, -0.5, 0.0, 2.2, 3.4])
    rcfg, scfg = _cfgs()
    big = 1.3
    crop, s_max = _crop_holding_blur(th, zk, rcfg, scfg.dzeta_um, slope=big)
    R = SR.SomaRenderer(crop, P, zk, rcfg, scfg, +1)
    t_none = R.render(th)
    t_cfg = R.render(th, slope=rcfg.kernel_continuation_slope)
    assert_array_equal(t_cfg, t_none, err_msg="T6 the configured slope must give the result of slope None")
    t_big = R.render(th, slope=big)
    ref = _light_absorbed(crop, 1, th)
    check_close("T6 light per plane at slope %g" % big, _light_per_plane(t_big), np.full(zk.size, ref), rtol=1e-9)
    m_none, m_big = _second_moment(t_none, crop, th[0], th[1]), _second_moment(t_big, crop, th[0], th[1])
    far = np.abs(zk - th[2]) > th[5] + 1.0
    print("  T6: crop %d px, s_max %.2f um at slope %g; second moments (um^2) at slope 0.79 / %g, planes %s: %s / %s"
          % (crop[2], s_max, big, big, zk.tolist(), np.round(m_none, 3).tolist(), np.round(m_big, 3).tolist()))
    assert far.sum() == 4 and np.all(m_big[far] > m_none[far]), (m_none, m_big)

    th = [0.0, 0.1, 0.4, 1.2, 0.9, 0.7, 0.3, 1.5]     # 2c / dzeta = 10: the slabs tile [z0 - c, z0 + c] both ways
    zk = 0.4 + np.array([-0.9, -0.2, 0.5, 1.3])
    rcfg, scfg = _cfgs()
    crop = _square_crop(25)
    up = SR.SomaRenderer(crop, P, zk, rcfg, scfg, +1).render(th)
    down = SR.SomaRenderer(crop, P, 2.0 * th[2] - zk, rcfg, scfg, -1).render(th)
    check_close("light reversed = depth mirrored", down, up, rtol=0, atol=1e-12)
    other = SR.SomaRenderer(crop, P, zk, rcfg, scfg, -1).render(th)
    assert float(np.max(np.abs(other - up))) > 1e-3, "the light direction must matter for a dark soma"


def test_contract():
    """(T7) shapes (K, height, width) and dtype float64, finite; tau <= 1 + 1e-12 everywhere (a pale and a dark soma,
    with and without a slope); mu = 0 gives 1 exactly; a soma outside the crop is not rendered (1 exactly); the planes
    in the order of z_planes_um [exact]."""
    rcfg, scfg = _cfgs()
    crop = (-20, -17, 41, 35)
    zk = np.array([0.1, -0.4, 0.9])
    R = SR.SomaRenderer(crop, P, zk, rcfg, scfg, +1)
    for th, slope in (([0.0, 0.0, 0.0, 1.5, 1.0, 1.2, 0.4, 0.3], None),
                      ([0.1, -0.1, 0.2, 1.4, 1.1, 1.0, -0.8, 3.0], None),
                      ([0.1, -0.1, 0.2, 1.4, 1.1, 1.0, -0.8, 3.0], 2.5)):
        tau = R.render(th, slope)
        assert tau.shape == (3, 35, 41) and tau.dtype == np.float64, (tau.shape, tau.dtype)
        assert np.all(np.isfinite(tau)) and float(tau.max()) <= 1.0 + 1e-12, float(tau.max())
        assert float(tau.min()) < 0.9, "the soma absorbs"
    one = R.render([0.0, 0.0, 0.0, 1.5, 1.0, 1.2, 0.4, 0.0])
    assert one.shape == (3, 35, 41) and one.dtype == np.float64 and np.array_equal(one, np.ones((3, 35, 41)))
    one = R.render([0.0, 0.0, 0.0, 1.5, 1.0, 1.2, 0.4, 0.0], slope=1.0)
    assert np.array_equal(one, np.ones((3, 35, 41)))
    away = R.render([100.0, 0.0, 0.0, 1.5, 1.0, 1.2, 0.4, 0.5])
    assert np.array_equal(away, np.ones((3, 35, 41))), "a soma outside the crop is not rendered"
    th = [0.05, 0.02, 0.0, 1.5, 1.0, 1.2, 0.4, 0.5]
    fwd = R.render(th)
    rev = SR.SomaRenderer(crop, P, zk[::-1], rcfg, scfg, +1).render(th)
    assert_array_equal(rev, fwd[::-1], err_msg="the planes in the order of z_planes_um")
    single = SR.SomaRenderer(crop, P, 0.1, rcfg, scfg, +1).render(th)
    assert single.shape == (1, 35, 41)
    rcfg2, scfg2 = _cfgs(f=2)
    tau2 = SR.SomaRenderer(crop, P, zk, rcfg2, scfg2, +1).render(th)
    assert tau2.shape == (3, 35, 41) and tau2.dtype == np.float64   # pixels, not samples, at f = 2


def test_determinism():
    """No randomness: repeated renders, and a second renderer on the same inputs, are bit-identical."""
    rcfg, scfg = _cfgs()
    crop, zk = (-15, -15, 31, 29), np.array([-0.8, 0.0, 0.6])
    th = [0.02, 0.01, 0.1, 1.2, 0.9, 0.8, 1.0, 0.9]
    R = SR.SomaRenderer(crop, P, zk, rcfg, scfg, +1)
    a = R.render(th, 1.1)
    b = R.render(th, 1.1)
    c = SR.SomaRenderer(crop, P, zk, rcfg, scfg, +1).render(th, 1.1)
    assert_array_equal(a, b)
    assert_array_equal(a, c)


def test_edge_cases():
    """(T7) the ValueErrors of SPEC Block 45: a bad theta_g (ellipsoid_columns and render); s_um or h_um not finite
    and > 0; an empty or non-integer crop; empty or non-finite depths; light_direction not +-1; a slope not finite
    and >= 0, or given with a continuation other than "linear" (slope None is accepted there). Also refused: a pixel
    size, slab thickness or fine_factor out of range."""
    good = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.5]
    bad_thetas = []
    for i, v in ((3, 0.0), (3, -1.0), (4, 0.0), (5, math.inf), (3, math.nan), (7, -0.1), (7, math.nan),
                 (7, math.inf), (0, math.nan), (1, math.inf), (2, math.nan), (6, math.inf)):
        t = list(good)
        t[i] = v
        bad_thetas.append(t)
    bad_thetas += [good[:7], good + [1.0], np.reshape(good, (2, 4)), ["x"] * 8]
    for t in bad_thetas:
        _expect_value_error("ellipsoid_columns %r" % (t,), lambda t=t: SR.ellipsoid_columns(0.0, 0.0, t))
    for s, h in ((0.0, 0.1), (-1.0, 0.1), (math.nan, 0.1), (math.inf, 0.1), (0.1, 0.0), (0.1, math.nan),
                 (0.1, -0.1), (np.array([0.1, 0.2]), 0.1)):
        _expect_value_error("transfer s %r h %r" % (s, h), lambda s=s, h=h: SR.sampled_gaussian_transfer(s, 0.0, h))

    rcfg, scfg = _cfgs()
    zk = [0.0, 0.5]
    for crop in ((0, 0, 0, 5), (0, 0, 5, 0), (0, 0, -3, 5), (0.5, 0, 5, 5), (0, 0, 5.5, 5), (0, 0, math.nan, 5),
                 (0, 0, 5), (0, 0, 5, 5, 1), None, (True, 0, 5, 5), ("a", 0, 5, 5)):
        _expect_value_error("crop %r" % (crop,), lambda crop=crop: SR.SomaRenderer(crop, P, zk, rcfg, scfg, +1))
    crop = (-5, -5, 11, 11)
    assert SR.SomaRenderer((-5.0, -5.0, 11.0, 11.0), P, zk, rcfg, scfg, +1).render(good).shape == (2, 11, 11)
    for z in ([], [math.nan], [0.0, math.inf], [[0.0, 0.5]]):
        _expect_value_error("depths %r" % (z,), lambda z=z: SR.SomaRenderer(crop, P, z, rcfg, scfg, +1))
    for ld in (0, 2, 0.5, "1", None, True, np.array([1])):
        _expect_value_error("light %r" % (ld,), lambda ld=ld: SR.SomaRenderer(crop, P, zk, rcfg, scfg, ld))
    for p in (0.0, -0.1, math.nan, math.inf):
        _expect_value_error("res0 %r" % (p,), lambda p=p: SR.SomaRenderer(crop, p, zk, rcfg, scfg, +1))
    for kw in (dict(fine_factor=0), dict(fine_factor=1.5), dict(dzeta_um=0.0), dict(dzeta_um=math.nan)):
        sc = dataclasses.replace(scfg, **kw)
        _expect_value_error("scfg %r" % (kw,), lambda sc=sc: SR.SomaRenderer(crop, P, zk, rcfg, sc, +1))
    assert SR.SomaRenderer(crop, P, zk, rcfg, scfg, -1).render(good).shape == (2, 11, 11)
    assert SR.SomaRenderer(crop, P, zk, rcfg, scfg, np.int64(-1)).render(good).shape == (2, 11, 11)

    R = SR.SomaRenderer(crop, P, zk, rcfg, scfg, +1)
    for t in bad_thetas:
        _expect_value_error("render theta %r" % (t,), lambda t=t: R.render(t))
    for slope in (math.nan, math.inf, -0.1, "x", [0.5, 0.6]):
        _expect_value_error("slope %r" % (slope,), lambda slope=slope: R.render(good, slope))
    assert R.render(good, 0.0).shape == (2, 11, 11)                  # slope 0 is allowed (>= 0)
    for cont in ("frozen", "proportional"):
        Rc = SR.SomaRenderer(crop, P, zk, dataclasses.replace(rcfg, kernel_continuation=cont), scfg, +1)
        _expect_value_error("slope with %s" % cont, lambda Rc=Rc: Rc.render(good, 0.79))
        _expect_value_error("slope with %s, mu 0" % cont, lambda Rc=Rc: Rc.render(good[:7] + [0.0], 0.79))
        assert Rc.render(good).shape == (2, 11, 11)                  # slope None: any continuation


def test_speed_memory():
    """(T8) one render of 36 planes 0.56 um apart about the centre of a 280 x 280 px crop (p = 0.1144 um, a soma of
    radius 6 um, mu 0.3/um, the default kernel, f = 1, dzeta = 0.14 um) within 30 s wall time and below 2 GB of
    peak memory, measured in a fresh Python process (its peak resident set, ru_maxrss, imports included); both
    reported, with the machine's load average."""
    code = "\n".join([
        "import resource, sys, time",
        "sys.path.insert(0, %r)" % SRC,
        "import numpy as np",
        "from allen_diameter.config import RendererConfig, SomaFitConfig",
        "from allen_diameter.model.soma_render import SomaRenderer",
        "p, n = 0.1144, 280",
        "xc = yc = 0.5 * (n - 1) * p",
        "z = 0.56 * (np.arange(36) - 17.5)",
        "t0 = time.perf_counter()",
        "R = SomaRenderer((0, 0, n, n), p, z, RendererConfig(), SomaFitConfig(), +1)",
        "t1 = time.perf_counter()",
        "tau = R.render([xc, yc, 0.0, 6.0, 6.0, 6.0, 0.0, 0.3])",
        "t2 = time.perf_counter()",
        "ok = tau.shape == (36, n, n) and bool(np.all(np.isfinite(tau)))",
        "print('T8 %.4f %.4f %.1f %d %.6f %.3e' % (t1 - t0, t2 - t1,"
        " resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, ok, float(tau.min()), float(tau.max()) - 1.0))",
    ])
    load0 = os.getloadavg()
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    load1 = os.getloadavg()
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("T8 ")]
    assert proc.returncode == 0 and lines, (proc.returncode, proc.stdout, proc.stderr)
    t_init, t_render, rss_mb, ok, tmin, tmax1 = lines[-1].split()[1:]
    t_render, rss_mb = float(t_render), float(rss_mb)
    print("  T8: render %.2f s (construction %.3f s), peak memory %.0f MB; tau min %s, max - 1 %s; load average "
          "(1 min) %.2f before, %.2f after on %d CPUs" % (t_render, float(t_init), rss_mb, tmin, tmax1, load0[0],
                                                          load1[0], os.cpu_count() or 0))
    assert ok == "1", "shape or finiteness"
    assert t_render <= 30.0, "T8 render %.2f s > 30 s" % t_render
    assert rss_mb < 2048.0, "T8 peak memory %.0f MB >= 2 GB" % rss_mb


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

    counts = {s: sum(1 for r in results if r[1] == s)
              for s in ("PASS", "FAIL", "ERROR", "TODO", "SKIP")}
    print("\n-- " + ", ".join(f"{n} {s.lower()}" for s, n in counts.items()))
    for path in _SAVED_ARTIFACTS:
        print(f"-- arrays from a failed comparison: {path}")
    failed = counts["FAIL"] + counts["ERROR"] + counts["TODO"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
