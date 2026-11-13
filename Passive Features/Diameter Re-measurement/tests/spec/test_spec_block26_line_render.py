"""Spec-derived tests, Block 26 (the line renderer) -- tester 2026-11-13.

Oracles (SPEC.md Block 26): against Block 4 (render_transmittance on the
block's fine grid, pixel_integrate, sample_profile) <= 1e-3 in each mode;
an independent semi-analytic statement of the linear mode for a long flat
tube (the column lengths per slab, exact pixel means of the Gaussian by erf,
Block 5's bilinear read) <= 1e-3; mu = 0 gives 1 [exact]; mirror symmetry
in v [1e-10]; order of ks; determinism; pixels_needed; the ValueErrors.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import special

import nc_helpers as H
from allen_diameter.analysis.profiles import sample_profile
from allen_diameter.model import geometry, render
from allen_diameter.model import line_render as LR
from allen_image_io import CropFrame

P = 0.1144
NOFAR = {"renderer.fft_split_sigma_um": 50.0, "renderer.ray_n_rho": 4, "renderer.ray_n_psi": 8}


def block4_profiles(tube, mu, ks, o, yh, eu, v, n_avg, step, cfg, absorption):
    rc = cfg.renderer.__class__(**{**cfg.renderer.__dict__, "absorption": absorption})
    left, top, W, Hh = -30, -30, 60, 60
    g, inner = render.block_fine_grid(left, top, W, Hh, P, 8, 30)
    tau = render.render_transmittance(tube, mu, np.asarray(ks) * cfg.acquisition.dz_um, g, rc,
                                      reduce=lambda pl: render.pixel_integrate(pl[inner[0], inner[1]], 8)).tau
    fr = CropFrame(left, top, 0, P)
    return np.stack([sample_profile(tau[i], fr, o, yh, eu, v, n_avg, step) for i in range(len(ks))])


CASES = [(0.3, 0.0, 0.0), (0.3, 30.0, 0.14), (1.0, 30.0, 0.0), (1.0, 60.0, 0.14)]


@pytest.mark.parametrize("absorption", ["linear", "partition_vertical"])
@pytest.mark.parametrize("d,tilt,zoff", CASES)
def test_against_block4(d, tilt, zoff, absorption):
    cfg = H.cfg_with(NOFAR)
    th = 0.35
    tube = geometry.Tube(c=(0.02, -0.03, zoff), r=0.5 * d, phi=math.radians(tilt), theta=th, half_length=2.5)
    yh, eu = np.array([-math.sin(th), math.cos(th)]), np.array([math.cos(th), math.sin(th)])
    o = np.array([0.02, -0.03])
    v = np.arange(-15, 16) * P
    ks = np.array([-1, 0, 1, 3])
    errs = []
    for n_avg in (0, 4):
        a = LR.line_profiles(tube, 1.2, ks, o, yh, eu, v, n_avg, P, cfg, absorption=absorption)
        b = block4_profiles(tube, 1.2, ks, o, yh, eu, v, n_avg, P, cfg, absorption)
        errs.append(float(np.max(np.abs(a - b))))
    print(absorption, d, tilt, zoff, "max |line - Block 4| =", errs)
    assert max(errs) <= 1e-3


def test_hybrid_against_block4_one_case():
    cfg = H.cfg_with(NOFAR)
    th = 0.2
    tube = geometry.Tube(c=(0.0, 0.0, 0.0), r=0.15, phi=math.radians(30), theta=th, half_length=1.5)
    yh, eu = np.array([-math.sin(th), math.cos(th)]), np.array([math.cos(th), math.sin(th)])
    v = np.arange(-10, 11) * P
    ks = np.array([0, 1])
    a = LR.line_profiles(tube, 2.0, ks, (0.0, 0.0), yh, eu, v, 0, P, cfg, absorption="hybrid")
    b = block4_profiles(tube, 2.0, ks, np.zeros(2), yh, eu, v, 0, P, cfg, "hybrid")
    print("hybrid max |line - Block 4| =", float(np.max(np.abs(a - b))))
    assert np.max(np.abs(a - b)) <= 1e-3


def test_linear_semi_analytic_long_flat_tube():
    """Linear mode, flat tube along x through (0, 0, zc), long (no cap effect near x = 0).
    deficit(row) = sum_j mu * int lambda_j(y') [pixel mean of g_sigma_j(y - y')] dy', the
    x-direction integrating to 1. lambda_j(y') = |column inside the tube and slab j|."""
    cfg = H.cfg_with(NOFAR)
    rc = cfg.renderer
    r, mu, zc = 0.4, 1.0, 0.07
    tube = geometry.Tube(c=(0.0, 0.0, zc), r=r, phi=0.0, theta=0.0, half_length=6.0)
    ks = np.array([-2, 0, 1])
    v = np.arange(-12, 13) * P
    o, yh, eu = np.array([0.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0])
    got = LR.line_profiles(tube, mu, ks, o, yh, eu, v, 0, P, cfg, absorption="linear")
    # independent: fine y' quadrature (midpoint, h/16), slab lattice per Block 2's rule over the column range
    h = P / 8
    yq = (np.arange(-int(r / (h / 16)) - 2, int(r / (h / 16)) + 3) + 0.5) * (h / 16)
    half = np.sqrt(np.clip(r * r - yq ** 2, 0, None))
    # Block 26 builds its lattice on the fine sample lattice (y = (n + 1/2) h - p/2): the column range there
    ys_lat = (np.arange(-200, 200) + 0.5) * h - 0.5 * P
    hl = np.sqrt(np.clip(r * r - ys_lat ** 2, 0, None))
    ins = np.abs(ys_lat) <= r
    zeta = H.ref_lattice(float((zc - hl)[ins].min()), float((zc + hl)[ins].max()), rc.dzeta_um)
    rows = np.arange(-20, 21)
    ref_rows = np.zeros((ks.size, rows.size))
    for j, zj in enumerate(zeta):
        lam = np.clip(np.minimum(zc + half, zj + rc.dzeta_um / 2) - np.maximum(zc - half, zj - rc.dzeta_um / 2), 0, None)
        for i, k in enumerate(ks):
            s = float(H.ref_sigma_r(zj - k * cfg.acquisition.dz_um, rc))
            lo = ((rows[:, None] - 0.5) * P - yq[None, :]) / s
            hi = ((rows[:, None] + 0.5) * P - yq[None, :]) / s
            ref_rows[i] += mu * ((special.ndtr(hi) - special.ndtr(lo)) / P) @ lam * (h / 16)
    # bilinear read along y of the row means (columns identical along x)
    ref = np.stack([1.0 - np.interp(v / P, rows, ref_rows[i]) for i in range(ks.size)])
    err = float(np.max(np.abs(got - ref)))
    print("linear, long flat tube: max |line - semi-analytic| =", err)
    assert err <= 1e-3


def test_mu_zero_symmetry_order_determinism():
    cfg = H.cfg_with(NOFAR)
    tube = geometry.Tube(c=(0.0, 0.0, 0.0), r=0.3, phi=0.0, theta=0.0, half_length=2.0)
    o, yh, eu = np.array([0.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0])
    v = np.arange(-10, 11) * P
    ks = np.array([0, -1, 2])
    assert_array_equal(LR.line_profiles(tube, 0.0, ks, o, yh, eu, v, 2, P, cfg, absorption="partition_vertical"), 1.0)
    a = LR.line_profiles(tube, 1.5, ks, o, yh, eu, v, 2, P, cfg, absorption="partition_vertical")
    assert np.max(np.abs(a - a[:, ::-1])) <= 1e-10                   # mirror in v
    perm = np.array([2, 0, 1])
    b = LR.line_profiles(tube, 1.5, ks[perm], o, yh, eu, v, 2, P, cfg, absorption="partition_vertical")
    assert np.max(np.abs(b - a[perm])) <= 1e-15
    c = LR.line_profiles(tube, 1.5, ks, o, yh, eu, v, 2, P, cfg, absorption="partition_vertical")
    assert_array_equal(a, c)


def test_pixels_needed_by_hand():
    o, yh, eu = np.array([0.37, -0.21]), np.array([-math.sin(0.4), math.cos(0.4)]), np.array([math.cos(0.4), math.sin(0.4)])
    v = np.arange(-5, 6) * P
    cols, rows = LR.pixels_needed(o, yh, eu, v, 2, 0.1, P)
    want = set()
    for a in range(-2, 3):
        for vn in v:
            x = o[0] + vn * yh[0] + a * 0.1 * eu[0]
            y = o[1] + vn * yh[1] + a * 0.1 * eu[1]
            c0, r0 = math.floor(x / P), math.floor(y / P)
            want |= {(r0, c0), (r0, c0 + 1), (r0 + 1, c0), (r0 + 1, c0 + 1)}
    assert set(zip(rows.tolist(), cols.tolist())) == want


def test_value_errors():
    cfg = H.cfg_with(NOFAR)
    tube = geometry.Tube(c=(0.0, 0.0, 0.0), r=0.3, phi=0.0, theta=0.0, half_length=2.0)
    o, v = (0.0, 0.0), np.arange(-3, 4) * P
    with pytest.raises(ValueError):
        LR.line_profiles(tube, -0.1, [0], o, (0, 1), (1, 0), v, 0, P, cfg)
    with pytest.raises(ValueError):
        LR.line_profiles(tube, 1.0, [0], o, (0, 1.1), (1, 0), v, 0, P, cfg)
    with pytest.raises(ValueError):
        LR.line_profiles(tube, 1.0, [0], o, (0, 1), (1, 0), v, 0, P, cfg, absorption="ray_world")
    uncapped = geometry.Tube(c=(0.0, 0.0, 0.0), r=0.3, phi=0.0, theta=0.0)
    with pytest.raises(ValueError):
        LR.line_profiles(uncapped, 1.0, [0], o, (0, 1), (1, 0), v, 0, P, cfg)
