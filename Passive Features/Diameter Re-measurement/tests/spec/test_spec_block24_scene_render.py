"""Spec-derived tests, Block 24 (the scene renderer) -- tester 2026-11-13.

Oracles (SPEC.md Block 24, section 2.3): one tube = Block 4 in the three
modes [1e-10]; an independent numpy/scipy statement of Eqs. 5-6 for flat tubes
(linear, partition) [1e-6: scipy's sampled-and-normalised Gaussian against
the FFT's analytic transfer function at sigma >= 2 h]; additivity of
non-overlapping tubes on one lattice [1e-10]; C1 per column [1e-12]; the tube
later along the light books exp(-mu_1 l_1) of its lone value [1e-10]; hybrid
faint limit = linear at <1/s_z>_W mu [5e-3]; equal totals per plane [1e-10
relative]; light reversed = depth mirrored [1e-10]; mu = 0 changes nothing
[exact]; the product of lone transmittances within the bound; determinism;
the ValueErrors.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import nc_helpers as H
from allen_diameter.model import geometry, render, ray_world
from allen_diameter.model import scene_render as SR

P = 0.1144
F = 8


def grid_block(left, top, w, hgt):
    g, _ = render.block_fine_grid(left, top, w, hgt, P, F, 0)
    return g


def flat(c, r, theta, U):
    return geometry.Tube(c=tuple(c), r=r, phi=0.0, theta=theta, half_length=U, end_cut="axial")


CFG_SMALL_RAYS = {"renderer.ray_n_rho": 4, "renderer.ray_n_psi": 8}


@pytest.mark.parametrize("absorption", ["linear", "partition_vertical", "hybrid"])
def test_one_tube_equals_block4(absorption):
    cfg = H.cfg_with(dict(CFG_SMALL_RAYS, **{"renderer.absorption": absorption}))
    t = geometry.Tube(c=(1.0, 1.1, 0.1), r=0.25, phi=math.radians(30), theta=0.4, half_length=0.8, end_cut="axial")
    g = grid_block(-12, -12, 44, 44)
    z = np.array([-0.28, 0.0, 0.28, 0.56])
    a = SR.render_scene([t], [1.5], z, g, cfg.renderer).tau
    b = render.render_transmittance(t, 1.5, z, g, cfg.renderer).tau
    assert np.max(np.abs(a - b)) <= 1e-10


@pytest.mark.parametrize("backend", ["fft", "direct"])
@pytest.mark.parametrize("absorption", ["linear", "partition_vertical"])
def test_two_flat_tubes_independent_reference(absorption, backend):
    # crossing headings, stacked in depth (a column crosses both): the joint partition matters
    cfg = H.cfg_with({"renderer.absorption": absorption, "renderer.backend": backend})
    tubes = [((1.0, 1.0, 0.0), 0.3, 0.0, 1.0), ((1.0, 1.0, 0.75), 0.3, math.pi / 2, 1.0)]
    mus = [2.0, 1.2]
    g = grid_block(-8, -8, 34, 34)
    z = np.array([0.0, 0.28, 0.84])
    tau = SR.render_scene([flat(*t) for t in tubes], mus, z, g, cfg.renderer).tau
    ref, _, _ = H.ref_render_flat(tubes, mus, g.xs, g.ys, z, cfg.renderer, absorption)
    err = float(np.max(np.abs(tau - ref)))
    print("max |scene - independent ref| =", err)
    assert err <= 1e-6


def test_additivity_non_overlapping_same_lattice():
    for absorption in ("linear", "partition_vertical"):
        cfg = H.cfg_with({"renderer.absorption": absorption})
        t1 = flat((0.0, 0.0, 0.0), 0.3, 0.0, 1.0)
        t2 = flat((0.0, 20 * P, 0.0), 0.3, 0.0, 1.0)         # shifted by whole pixels: same columns, same lattice alone and together
        g = grid_block(-25, -15, 50, 70)
        z = np.array([-0.28, 0.0, 0.56])
        sc = SR.render_scene([t1, t2], [1.0, 2.0], z, g, cfg.renderer).tau
        a = SR.render_scene([t1], [1.0], z, g, cfg.renderer).tau
        b = SR.render_scene([t2], [2.0], z, g, cfg.renderer).tau
        assert np.max(np.abs((1 - sc) - ((1 - a) + (1 - b)))) <= 1e-10


@pytest.mark.parametrize("light", [+1, -1])
def test_partition_c1_and_attenuation_order(light):
    lower = flat((0.0, 0.0, 0.0), 0.3, 0.0, 1.0)
    upper = flat((0.0, 0.0, 1.0), 0.3, math.pi / 2, 1.0)
    mu_l, mu_u = 1.5, 2.5
    g = grid_block(-12, -12, 24, 24)
    zeta, dA = SR._absorbed_fractions([lower, upper], [mu_l, mu_u], g, 0.02, light, "partition_vertical")
    X, Y = g.mesh()
    zl1, zh1, in1 = geometry.column_interval(X, Y, lower)
    zl2, zh2, in2 = geometry.column_interval(X, Y, upper)
    a_tot = mu_l * (zh1 - zl1) + mu_u * (zh2 - zl2)
    # C1: per column, sum_j DeltaA_j = 1 - exp(-sum_j a_j)
    assert np.max(np.abs(dA.sum(axis=0) - (-np.expm1(-a_tot)))) <= 1e-12
    # the tube later along the light books exp(-mu_first l_first) times its lone value
    both = in1 & in2
    assert both.any()
    books_lower = dA[zeta < 0.5].sum(axis=0)
    books_upper = dA[zeta >= 0.5].sum(axis=0)
    lone_l = -np.expm1(-mu_l * (zh1 - zl1))
    lone_u = -np.expm1(-mu_u * (zh2 - zl2))
    if light == +1:   # light toward +z: the lower tube first
        assert_allclose(books_lower[both], lone_l[both], atol=1e-10)
        assert_allclose(books_upper[both], np.exp(-mu_l * (zh1 - zl1))[both] * lone_u[both], atol=1e-10)
    else:
        assert_allclose(books_upper[both], lone_u[both], atol=1e-10)
        assert_allclose(books_lower[both], np.exp(-mu_u * (zh2 - zl2))[both] * lone_l[both], atol=1e-10)


def test_hybrid_faint_limit():
    cfg_h = H.cfg_with({"renderer.absorption": "hybrid"})
    cfg_l = H.cfg_with({"renderer.absorption": "linear"})
    t1 = flat((0.0, 0.0, 0.0), 0.25, 0.0, 0.8)
    t2 = flat((0.0, 1.8, 0.1), 0.25, 0.3, 0.8)
    mu = 1e-3 / 0.5                        # mu d = 1e-3
    g = grid_block(-15, -12, 30, 50)
    z = np.array([0.0, 0.28])
    th = SR.render_scene([t1, t2], [mu, mu], z, g, cfg_h.renderer).tau
    k = H.mean_inv_sz(cfg_h.renderer.ray_na, cfg_h.renderer.ray_n_oil)
    assert abs(k - 1.4470) < 5e-4                 # section 2.3's constant from the condenser's disc
    tl = SR.render_scene([t1, t2], [k * mu, k * mu], z, g, cfg_l.renderer).tau
    dh, dl = 1 - th, 1 - tl
    rel = np.max(np.abs(dh - dl)) / np.max(dl)
    print("hybrid faint-limit relative difference", rel)
    assert rel <= 5e-3


@pytest.mark.parametrize("absorption", ["linear", "partition_vertical"])
def test_equal_totals_per_plane(absorption):
    cfg = H.cfg_with({"renderer.absorption": absorption})
    t1 = geometry.Tube(c=(0.0, 0.0, 0.0), r=0.3, phi=math.radians(20), theta=0.2, half_length=1.0)
    t2 = flat((0.2, 2.0, 0.3), 0.2, 1.0, 0.8)
    g = grid_block(-75, -75, 150, 170)                  # margin >= 7 um, > 8 sigma at |delta| <= 1.15 um
    z = np.array([-0.56, 0.0, 0.28, 0.84])
    tau = SR.render_scene([t1, t2], [1.0, 2.0], z, g, cfg.renderer).tau
    tot = (1 - tau).sum(axis=(1, 2))
    _, dA = SR._absorbed_fractions([t1, t2], [1.0, 2.0], g, cfg.renderer.dzeta_um, 1, absorption)
    assert_allclose(tot, dA.sum(), rtol=1e-10)


@pytest.mark.parametrize("absorption", ["partition_vertical", "hybrid"])
def test_light_reversed_equals_depth_mirrored(absorption):
    # SPEC [1e-10] holds on a mirrored slab lattice: dzeta is chosen to divide the union's depth range
    # (on another lattice the two renders differ by the depth quadrature, measured and printed below)
    t = geometry.Tube(c=(0.0, 0.0, 0.13), r=0.3, phi=math.radians(35), theta=0.3, half_length=0.9)
    tm = geometry.Tube(c=(0.0, 0.0, -0.13), r=0.3, phi=math.radians(35), theta=0.3 + math.pi, half_length=0.9)
    u = flat((0.1, 1.9, -0.2), 0.25, 1.2, 0.7)
    um = flat((0.1, 1.9, 0.2), 0.25, 1.2 + math.pi, 0.7)
    g = grid_block(-15, -15, 30, 50)
    X, Y = g.mesh()
    lo = min(float(geometry.column_interval(X, Y, q)[0][geometry.column_interval(X, Y, q)[2]].min()) for q in (t, u))
    hi = max(float(geometry.column_interval(X, Y, q)[1][geometry.column_interval(X, Y, q)[2]].max()) for q in (t, u))
    dzeta = (hi - lo) / math.ceil((hi - lo) / 0.02)
    z = np.array([-0.28, 0.0, 0.28])
    errs = []
    for dz in (0.02, dzeta):
        cfg = H.cfg_with(dict(CFG_SMALL_RAYS, **{"renderer.absorption": absorption, "renderer.dzeta_um": dz}))
        a = SR.render_scene([t, u], [2.0, 1.0], z, g, cfg.renderer, light_direction=-1)
        b = SR.render_scene([tm, um], [2.0, 1.0], -z, g, cfg.renderer, light_direction=+1)
        errs.append(float(np.max(np.abs(a.tau - b.tau))))
    print("light reversed vs mirrored: default lattice %.3g, mirrored lattice %.3g" % tuple(errs))
    assert errs[1] <= 1e-10


def test_mu_zero_changes_nothing_and_product_bound():
    cfg = H.cfg_with({"renderer.absorption": "partition_vertical"})
    t1 = flat((0.0, 0.0, 0.0), 0.3, 0.0, 1.0)
    t2 = flat((0.0, 1.5, 0.0), 0.3, 0.4, 1.0)
    g = grid_block(-20, -15, 40, 50)
    z = np.array([0.0, 0.56])
    a = SR.render_scene([t1], [1.5], z, g, cfg.renderer).tau
    b = SR.render_scene([t1, t2], [1.5, 0.0], z, g, cfg.renderer).tau
    assert_array_equal(a, b)
    sc = SR.render_scene([t1, t2], [1.5, 2.0], z, g, cfg.renderer).tau
    t_b = SR.render_scene([t2], [2.0], z, g, cfg.renderer).tau
    prod = a * t_b
    bound = np.max((1 - a) * (1 - t_b))
    assert np.max(np.abs(prod - sc)) <= bound + 1e-12


def test_synthetic_scene_block_contract_and_determinism():
    cfg = H.cfg_with({"renderer.absorption": "partition_vertical"})
    t1 = flat((3.0, 3.0, 0.0), 0.3, 0.0, 1.0)
    t2 = flat((3.0, 4.2, 0.28), 0.25, 0.5, 1.0)
    ks = np.array([-1, 0, 1])
    out1 = SR.synthetic_scene_block([t1, t2], [1.0, 1.0], ks, 15, 15, 24, 24, cfg, np.random.default_rng(7), 6)
    out2 = SR.synthetic_scene_block([t1, t2], [1.0, 1.0], ks, 15, 15, 24, 24, cfg, np.random.default_rng(7), 6)
    blk, kk, valid, frame = out1
    assert blk.dtype == np.uint8 and blk.shape == (3, 24, 24)
    assert_array_equal(kk, ks) and valid.all() and valid.shape == (3,)
    assert (frame.left, frame.top, frame.downsample) == (15, 15, 0) and frame.res_um_px == P
    assert_array_equal(blk, out2[0])


def test_value_errors():
    cfg = H.cfg_with()
    t = flat((0.0, 0.0, 0.0), 0.3, 0.0, 1.0)
    g = grid_block(-5, -5, 10, 10)
    with pytest.raises(ValueError):
        SR.render_scene([t, t], [1.0], [0.0], g, cfg.renderer)
    with pytest.raises(ValueError):
        SR.render_scene([t], [-0.1], [0.0], g, cfg.renderer)
    with pytest.raises(ValueError):
        SR.render_scene([], [], [0.0], g, cfg.renderer)
    with pytest.raises(ValueError):
        SR.render_scene([t], [float("nan")], [0.0], g, cfg.renderer)
