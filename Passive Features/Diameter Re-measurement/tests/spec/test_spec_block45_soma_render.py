"""Spec-derived tests of Block 45 (the soma's forward model: an ellipsoid rendered with the partition absorption;
D-215, D-216, D-218, D-219). Tester, 2026-11-22 (incremental run at 97f5104).

The reference renderer `ref_render` is written from SPEC.md's text: the ellipsoid's columns (Block 45), the
partition (S4) and (5-6) of section 2.3, Block 4's kernel law (`spec_helpers.ref_sigma_r`, an earlier tester's
independent copy), the normalised sampled Gaussian by `scipy.ndimage.gaussian_filter` (mode "constant", a long
truncation), the f x f samples of each pixel and their mean. Only the slab centres are taken from Block 2
(`geometry.slab_centres`, outside this run's scope), because the spec names that function as the slab rule without
restating it; every test that must not depend on the slab grid (light conservation, a constant kernel, the mirror) is
written so that the slabs drop out.

    python3 -m pytest -q -p no:cacheprovider tests/spec/test_spec_block45_soma_render.py

Pure ASCII.
"""
from __future__ import annotations

import dataclasses
import math
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.ndimage import gaussian_filter

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
for _p in (WS / "src", WS / "scripts", WS / "tests" / "smoke", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from spec_helpers import ref_sigma_r                             # noqa: E402
from allen_diameter import config as C                           # noqa: E402
from allen_diameter.model import geometry                        # noqa: E402  (Block 2: the slab rule only)
from allen_diameter.model import soma_render as SR               # noqa: E402

P = 0.1144


def cfgs(**soma):
    cfg = C.default_config()
    return cfg.renderer, dataclasses.replace(cfg.soma_fit, **soma)


# ------------------------------------------------------------ references ---

def ref_columns(X, Y, th):
    x0, y0, z0, a, b, c, phi, mu = th
    u = (X - x0) * math.cos(phi) + (Y - y0) * math.sin(phi)
    v = -(X - x0) * math.sin(phi) + (Y - y0) * math.cos(phi)
    q = 1.0 - (u / a) ** 2 - (v / b) ** 2
    ins = q > 0
    half = c * np.sqrt(np.where(ins, q, 0.0))
    return np.where(ins, z0 - half, z0), np.where(ins, z0 + half, z0), ins


def ref_samples(crop, p, f):
    """the f x f samples of every pixel: pixel (row, col) centred on ((left + col) p, (top + row) p), its samples at
    offsets ((i + 1/2)/f - 1/2) p"""
    left, top, W, H = crop
    xs = (left + (np.arange(W * f) + 0.5) / f - 0.5) * p
    ys = (top + (np.arange(H * f) + 0.5) / f - 0.5) * p
    return np.meshgrid(xs, ys)


def ref_render(crop, p, z_planes, rcfg, f, dzeta, th, light=+1, slope=None, zeta=None):
    """tau (K, H, W) by SPEC Block 45 written out: (S4) per slab, Delta A = T_<j (1 - e^-a_j), each blurred by the
    normalised sampled Gaussian of SD sigma_r(zeta_j - z_k) on the samples (zero outside: linear convolution), summed,
    1 minus, pixel mean."""
    left, top, W, H = crop
    X, Y = ref_samples(crop, p, f)
    h = p / f
    x0, y0, z0, a, b, c, phi, mu = th
    zlo, zhi, _ = ref_columns(X, Y, th)
    if zeta is None:
        zeta = geometry.slab_centres(z0 - c, z0 + c, dzeta)
    rc = rcfg if slope is None else dataclasses.replace(rcfg, kernel_continuation_slope=float(slope))
    out = np.empty((len(z_planes), H, W))
    dA = []
    for zj in zeta:
        lo, hi = zj - dzeta / 2, zj + dzeta / 2
        aj = mu * np.clip(np.minimum(zhi, hi) - np.maximum(zlo, lo), 0, None)
        if light == +1:
            before = np.clip(np.minimum(zhi, lo) - zlo, 0, None)       # the column below the slab
        else:
            before = np.clip(zhi - np.maximum(zlo, hi), 0, None)       # the column above the slab
        dA.append(np.exp(-mu * before) * (1 - np.exp(-aj)))
    for k, zk in enumerate(z_planes):
        acc = np.zeros_like(X)
        for zj, d in zip(zeta, dA):
            s = float(ref_sigma_r(np.array([zj - zk]), rc)[0])
            acc += gaussian_filter(d, s / h, mode="constant", cval=0.0, truncate=14.0)
        t = 1.0 - acc
        out[k] = t.reshape(H, f, W, f).mean(axis=(1, 3))
    return out


def ref_light(crop, p, f, th):
    """h^2 sum over the samples of 1 - exp(-mu (z_hi - z_lo)): the light the soma absorbs (um^2)"""
    X, Y = ref_samples(crop, p, f)
    zlo, zhi, _ = ref_columns(X, Y, th)
    return (p / f) ** 2 * float(np.sum(1 - np.exp(-th[7] * (zhi - zlo))))


# ---------------------------------------------------- ellipsoid_columns ---

def test_columns_sphere_rotation_axes_and_boundary():
    rng = np.random.default_rng(0)
    x, y = rng.uniform(-5, 5, 2000), rng.uniform(-5, 5, 2000)
    th = (0.3, -0.2, 7.0, 3.0, 3.0, 3.0, 0.0, 0.5)
    zlo, zhi, ins = SR.ellipsoid_columns(x, y, th)
    rho2 = (x - 0.3) ** 2 + (y + 0.2) ** 2
    chord = 2 * np.sqrt(np.clip(9.0 - rho2, 0, None))
    assert_allclose(zhi - zlo, chord, atol=1e-12, rtol=0)                    # sphere chord [1e-12 um]
    assert_array_equal(ins, rho2 < 9.0)
    assert np.all(zlo[~ins] == 7.0) and np.all(zhi[~ins] == 7.0)          # outside: z_lo = z_hi = z0
    # a = b: no dependence on phi [exact]
    th1 = (0.3, -0.2, 7.0, 2.5, 2.5, 4.0, 0.0, 0.5)
    th2 = (0.3, -0.2, 7.0, 2.5, 2.5, 4.0, 1.234, 0.5)
    for u, v in zip(SR.ellipsoid_columns(x, y, th1), SR.ellipsoid_columns(x, y, th2)):
        assert_array_equal(u, v)
    # the rotated axes: a point at distance r along (cos phi, sin phi) has the chord 2 c sqrt(1 - r^2/a^2)
    phi, a, b, c = 0.7, 4.0, 1.5, 2.0
    r = np.linspace(0, 3.9, 40)
    th3 = (1.0, 2.0, 0.0, a, b, c, phi, 1.0)
    zl, zh, _ = SR.ellipsoid_columns(1.0 + r * math.cos(phi), 2.0 + r * math.sin(phi), th3)
    assert_allclose(zh - zl, 2 * c * np.sqrt(1 - r ** 2 / a ** 2), atol=1e-12, rtol=0)
    zl, zh, _ = SR.ellipsoid_columns(1.0 - r * math.sin(phi) * b / 4.0, 2.0 + r * math.cos(phi) * b / 4.0, th3)
    rb = r * b / 4.0
    assert_allclose(zh - zl, 2 * c * np.sqrt(1 - rb ** 2 / b ** 2), atol=1e-12, rtol=0)
    # against the reference on random ellipsoids and points, broadcast shapes kept
    for _ in range(20):
        t = (rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(-5, 5), rng.uniform(1, 5), rng.uniform(1, 5),
             rng.uniform(1, 5), rng.uniform(-math.pi, math.pi), rng.uniform(0, 2))
        X, Y = np.meshgrid(np.linspace(-6, 6, 37), np.linspace(-6, 6, 29))
        got = SR.ellipsoid_columns(X, Y[:1, :1] * 0 + Y, t)
        ref = ref_columns(X, Y, t)
        assert got[0].shape == X.shape
        assert_allclose(got[0], ref[0], atol=1e-12, rtol=0)
        assert_allclose(got[1], ref[1], atol=1e-12, rtol=0)
    zl, zh, ins = SR.ellipsoid_columns(np.zeros((3, 1)), np.zeros((1, 4)), th)
    assert zl.shape == (3, 4)
    # a point exactly on the rim (q = 0) is outside (q > 0 is the rule)
    zl, zh, ins = SR.ellipsoid_columns(np.array([2.0]), np.array([0.0]), (0, 0, 0, 2.0, 1.0, 1.0, 0.0, 1.0))
    assert not ins[0] and zl[0] == zh[0] == 0.0


@pytest.mark.parametrize("bad", [(0, 0, 0, 0.0, 1, 1, 0, 1), (0, 0, 0, 1, -1, 1, 0, 1), (0, 0, 0, 1, 1, np.inf, 0, 1),
                                 (0, 0, 0, 1, 1, 1, 0, -0.1), (0, 0, 0, 1, 1, 1, 0, np.nan), (np.nan, 0, 0, 1, 1, 1, 0, 1),
                                 (0, 0, np.inf, 1, 1, 1, 0, 1), (0, 0, 0, 1, 1, 1, np.nan, 1), (0, 0, 0, 1, 1, 1, 0)])
def test_columns_refusals(bad):
    with pytest.raises(ValueError):
        SR.ellipsoid_columns(0.0, 0.0, bad)


# ------------------------------------------------- the sampled Gaussian ---

def ref_dtft(s, nu, h):
    """sum_m w_m exp(-2 pi i nu m h), w the normalised sampled Gaussian, |m| up to 40 s/h + 40 (the tail below 1e-300)"""
    M = int(40 * s / h) + 40
    m = np.arange(-M, M + 1)
    w = np.exp(-(m * h) ** 2 / (2 * s * s))
    w /= w.sum()
    return np.real(np.exp(-2j * np.pi * np.multiply.outer(nu, m * h)) @ w)


@pytest.mark.parametrize("ratio", [0.2, 0.35, 0.5, 0.71, 1.0, 1.7, 3.0, 7.0, 20.0])
def test_transfer_equals_the_sampled_kernels_dft(ratio):
    h = 0.05
    s = ratio * h
    nu = np.linspace(-1 / (2 * h), 1 / (2 * h), 257)
    got = SR.sampled_gaussian_transfer(s, nu, h)
    assert got.shape == nu.shape
    assert_allclose(got, ref_dtft(s, nu, h), atol=1e-12, rtol=0)            # [1e-12]
    assert abs(SR.sampled_gaussian_transfer(s, 0.0, h) - 1.0) <= 1e-15      # 1 at nu = 0 [1e-15]
    assert_allclose(got, got[::-1], atol=1e-15, rtol=0)                     # even in nu
    # the periodic grid of 4096 samples (SPEC T2): numpy.fft.fft of the wrapped kernel at the DFT's frequencies
    N = 4096
    m = np.arange(N)
    mm = np.where(m < N // 2, m, m - N)
    w = np.exp(-(mm * h) ** 2 / (2 * s * s))
    w /= w.sum()
    F = np.real(np.fft.fft(w))
    nuf = np.fft.fftfreq(N, d=h)
    assert_allclose(SR.sampled_gaussian_transfer(s, nuf, h), F, atol=1e-12, rtol=0)


@pytest.mark.parametrize("args", [(0.0, 0.1, 0.1), (-1, 0.1, 0.1), (np.nan, 0.1, 0.1), (0.1, 0.1, 0.0),
                                  (0.1, 0.1, np.inf), (np.array([0.1, 0.2]), 0.1, 0.1)])
def test_transfer_refusals(args):
    with pytest.raises(ValueError):
        SR.sampled_gaussian_transfer(*args)


# ------------------------------------------------------------ the render ---

CASES = [
    # crop, z planes, f, theta_g, light, slope
    ((-30, -26, 60, 52), (-1.4, -0.56, 0.0, 0.84, 2.24, 4.2), 1, (0.17, -0.31, 0.3, 1.6, 1.1, 1.2, 0.6, 0.35), +1, None),
    ((-30, -26, 60, 52), (-1.4, 0.0, 0.84, 3.1), 2, (0.17, -0.31, 0.3, 1.6, 1.1, 1.2, 0.6, 0.35), -1, None),
    ((-24, -24, 48, 48), (1.0, 2.7, 6.0), 1, (0.05, 0.02, 2.0, 1.2, 1.2, 0.9, 0.0, 1.5), +1, 1.3),
]


@pytest.mark.parametrize("case", range(len(CASES)))
def test_render_against_direct_reference(case):
    crop, z, f, th, light, slope = CASES[case]
    rcfg, scfg = cfgs(fine_factor=f)
    ren = SR.SomaRenderer(crop, P, np.array(z), rcfg, scfg, light_direction=light)
    tau = ren.render(np.array(th), slope)
    ref = ref_render(crop, P, z, rcfg, f, scfg.dzeta_um, th, light, slope)
    assert tau.shape == (len(z), crop[3], crop[2]) and tau.dtype == np.float64
    # tolerance: SPEC T4's 1e-9 absolute (the wrap of the FFT box, the truncation, the aliases)
    assert np.max(np.abs(tau - ref)) <= 1e-9, np.max(np.abs(tau - ref))
    assert np.max(tau) <= 1 + 1e-12


def test_light_conserved_per_plane():
    # a crop extending more than 7 s_max beyond the soma; f = 1 and 2; planes up to 3 um away (s_max ~ 1.9 um)
    th = (0.1, -0.05, 0.0, 1.3, 1.0, 1.1, 0.4, 0.6)
    for f in (1, 2):
        rcfg, scfg = cfgs(fine_factor=f)
        z = np.array([-3.0, -1.0, 0.0, 0.5, 2.0, 3.0])
        smax = float(ref_sigma_r(np.array([1.1 + 3.0]), rcfg)[0])
        half_px = int(math.ceil((1.3 + 7 * smax) / P))
        crop = (-half_px, -half_px, 2 * half_px + 1, 2 * half_px + 1)
        tau = SR.SomaRenderer(crop, P, z, rcfg, scfg).render(np.array(th))
        light = ref_light(crop, P, f, th)
        got = P * P * np.sum(1 - tau, axis=(1, 2))
        assert_allclose(got, light, rtol=1e-9, atol=0)                       # [1e-9 relative]


def test_constant_kernel_does_not_depend_on_the_slabs():
    # a flat table and a frozen continuation: every slab is blurred alike, so tau_k = 1 - g_s (*) (1 - exp(-mu L))
    # for every plane, whatever the slab grid [1e-12: one FFT path against gaussian_filter]
    cfg = C.with_overrides(C.default_config(), {"renderer.kernel_table_delta_um": [0.0, 1.0],
                                                "renderer.kernel_table_sigma_um": [0.23, 0.23],
                                                "renderer.sigma_r0_um": 0.23, "renderer.kernel_continuation": "frozen"})
    crop = (-28, -28, 57, 57)
    th = (0.0, 0.1, 0.0, 1.4, 0.9, 1.0, -0.5, 0.8)
    X, Y = ref_samples(crop, P, 1)
    zlo, zhi, _ = ref_columns(X, Y, th)
    ref = 1 - gaussian_filter(1 - np.exp(-0.8 * (zhi - zlo)), 0.23 / P, mode="constant", truncate=14.0)
    for dz in (0.14, 0.05, 0.6):
        scfg = dataclasses.replace(cfg.soma_fit, dzeta_um=dz)
        tau = SR.SomaRenderer(crop, P, np.array([-4.0, 0.0, 2.5]), cfg.renderer, scfg).render(np.array(th))
        for k in range(3):
            assert np.max(np.abs(tau[k] - ref)) <= 1e-12


def test_light_direction_mirror():
    # 2c a multiple of dzeta, so the slabs are symmetric about z0: tau(z_k; -1) = tau(2 z0 - z_k; +1) [1e-12]
    rcfg, scfg = cfgs()
    th = np.array([0.0, 0.0, 1.0, 1.5, 1.2, 0.7, 0.3, 1.2])      # 2c = 1.4 = 10 dzeta
    z = np.array([-0.4, 0.3, 1.0, 1.84, 2.6])
    crop = (-25, -25, 51, 51)
    tp = SR.SomaRenderer(crop, P, 2 * 1.0 - z, rcfg, scfg, light_direction=+1).render(th)
    tm = SR.SomaRenderer(crop, P, z, rcfg, scfg, light_direction=-1).render(th)
    assert np.max(np.abs(tp - tm)) <= 1e-12
    assert np.max(np.abs(tp - SR.SomaRenderer(crop, P, z, rcfg, scfg, +1).render(th))) > 1e-4   # the light matters


def test_pixel_convention_and_translation():
    rcfg, scfg = cfgs()
    crop = (-20, -14, 41, 33)
    # a soma centred on the centre of pixel (row 14, col 20) = image point ((left+20) p, (top+14) p) = (0, 0)
    th = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.7])
    tau = SR.SomaRenderer(crop, P, np.array([0.0, 1.2]), rcfg, scfg).render(th)
    c = tau[:, 14 - 12:14 + 13, 20 - 12:20 + 13]
    assert np.max(np.abs(c - c[:, ::-1, :])) <= 1e-12 and np.max(np.abs(c - c[:, :, ::-1])) <= 1e-12
    # f = 2: the samples are symmetric about each pixel's centre too
    tau2 = SR.SomaRenderer(crop, P, np.array([0.0]), rcfg, dataclasses.replace(scfg, fine_factor=2)).render(th)
    c2 = tau2[:, 2:27, 8:33]
    assert np.max(np.abs(c2 - c2[:, ::-1, ::-1])) <= 1e-12
    # a shift of x0 by 3 pixels shifts the image by 3 columns (every source inside the crop)
    th3 = th.copy()
    th3[0] += 3 * P
    t3 = SR.SomaRenderer(crop, P, np.array([0.0, 1.2]), rcfg, scfg).render(th3)
    assert np.max(np.abs(t3[:, :, 3:] - tau[:, :, :-3])) <= 1e-12


def test_slope_mu_zero_order_and_bounds():
    rcfg, scfg = cfgs()
    crop = (-40, -40, 81, 81)
    th = np.array([0.0, 0.0, 0.0, 1.2, 1.0, 1.1, 0.2, 0.5])
    z = np.array([0.0, 2.5, -3.0, 4.0])
    ren = SR.SomaRenderer(crop, P, z, rcfg, scfg)
    base = ren.render(th)
    assert_array_equal(ren.render(th, rcfg.kernel_continuation_slope), base)       # the configured slope [exact]
    assert_array_equal(SR.SomaRenderer(crop, P, z[[2, 0, 3, 1]], rcfg, scfg).render(th), base[[2, 0, 3, 1]])
    z0 = th.copy()
    z0[7] = 0.0
    assert_array_equal(ren.render(z0), np.ones((4, 81, 81)))                         # mu = 0: 1 exactly
    assert np.all(base <= 1 + 1e-12)
    assert_array_equal(ren.render(th), base)                                          # determinism
    # a larger slope: each plane's light unchanged; the light-weighted second moment larger for planes farther than
    # c + 1 um from z0 (here the planes at 2.5, -3.0 and 4.0 um), on a crop that holds the blur: 7 s_max beyond the
    # soma, s_max the kernel at the largest defocus (4 + 1.1 um) under the larger slope
    wide_rc = dataclasses.replace(rcfg, kernel_continuation_slope=1.4)
    smax = float(ref_sigma_r(np.array([4.0 + 1.1]), wide_rc)[0])
    half_px = int(math.ceil((1.2 + 7 * smax) / P))
    big = (-half_px, -half_px, 2 * half_px + 1, 2 * half_px + 1)
    renb = SR.SomaRenderer(big, P, z, rcfg, scfg)
    b0, b1 = renb.render(th), renb.render(th, 1.4)
    X, Y = ref_samples(big, P, 1)
    light = ref_light(big, P, 1, th)
    for k in range(4):
        l0, l1 = P * P * np.sum(1 - b0[k]), P * P * np.sum(1 - b1[k])
        assert abs(l0 - light) <= 1e-9 * light and abs(l1 - light) <= 1e-9 * light     # T3 at both slopes
        m0 = np.sum((1 - b0[k]) * (X ** 2 + Y ** 2)) / np.sum(1 - b0[k])
        m1 = np.sum((1 - b1[k]) * (X ** 2 + Y ** 2)) / np.sum(1 - b1[k])
        if abs(z[k]) > 1.1 + 1.0:
            assert m1 > m0
    assert np.all(b1 <= 1 + 1e-12)


def test_refusals():
    rcfg, scfg = cfgs()
    th = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.5])
    for crop in ((0, 0, 0, 5), (0, 0, 5, -1), (0.5, 0, 5, 5), (0, 0, 5), (0, 0, 5.5, 5), (True, 0, 5, 5)):
        with pytest.raises(ValueError):
            SR.SomaRenderer(crop, P, [0.0], rcfg, scfg)
    for z in ([], [np.nan], [0.0, np.inf]):
        with pytest.raises(ValueError):
            SR.SomaRenderer((0, 0, 5, 5), P, z, rcfg, scfg)
    for light in (0, 2, True, "1", 0.5):
        with pytest.raises(ValueError):
            SR.SomaRenderer((0, 0, 5, 5), P, [0.0], rcfg, scfg, light_direction=light)
    ren = SR.SomaRenderer((-5, -5, 11, 11), P, [0.0], rcfg, scfg)
    for s in (-0.1, np.nan, np.inf):
        with pytest.raises(ValueError):
            ren.render(th, s)
    frozen = dataclasses.replace(rcfg, kernel_continuation="frozen")
    with pytest.raises(ValueError):
        SR.SomaRenderer((-5, -5, 11, 11), P, [0.0], frozen, scfg).render(th, 0.5)    # a slope needs "linear"
    with pytest.raises(ValueError):
        ren.render(th[:7])


def test_convergence_in_f_and_dzeta():
    # SPEC T5: the largest difference against a fine reference decreases with f and with dzeta (reported)
    rcfg, _ = cfgs()
    th = np.array([0.03, -0.02, 0.0, 1.2, 1.0, 0.9, 0.5, 0.4])
    crop = (-22, -22, 45, 45)
    z = np.array([-1.12, 0.0, 0.56, 1.68])
    tf8 = SR.SomaRenderer(crop, P, z, rcfg, cfgs(fine_factor=8)[1]).render(th)
    errs = [np.max(np.abs(SR.SomaRenderer(crop, P, z, rcfg, cfgs(fine_factor=f)[1]).render(th) - tf8))
            for f in (1, 2, 4)]
    assert errs[0] > errs[1] > errs[2]
    tz = SR.SomaRenderer(crop, P, z, rcfg, cfgs(dzeta_um=0.035)[1]).render(th)
    ez = [np.max(np.abs(SR.SomaRenderer(crop, P, z, rcfg, cfgs(dzeta_um=d)[1]).render(th) - tz))
          for d in (0.56, 0.28, 0.14)]
    assert ez[0] > ez[1] > ez[2]
    print("T5 (tester): f 1, 2, 4 vs 8: %s; dzeta 0.56, 0.28, 0.14 vs 0.035: %s" % (errs, ez))


def test_speed_and_memory_t8():
    # SPEC T8: 36 planes 0.56 um apart about the centre of a 280 x 280 crop, a soma of radius 6 um, mu 0.3/um, the
    # default kernel, f = 1, dzeta 0.14 um: within 30 s and below 2 GB (both reported)
    rcfg, scfg = cfgs()
    z = (np.arange(36) - 17.5) * 0.56
    th = np.array([140 * P, 140 * P, 0.0, 6.0, 6.0, 6.0, 0.0, 0.3])
    ren = SR.SomaRenderer((0, 0, 280, 280), P, z, rcfg, scfg)
    t0 = time.perf_counter()
    tau = ren.render(th)
    dt = time.perf_counter() - t0
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    print("T8 (tester): %.1f s, peak RSS of the test process %.0f MB" % (dt, peak_mb))
    assert tau.shape == (36, 280, 280)
    assert dt <= 30.0 and peak_mb <= 2048.0
