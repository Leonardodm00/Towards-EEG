"""Spec-derived tests, Block 25 (the shadow image, with amendments 2 and 3) -- tester 2026-11-13.

Oracles (SPEC.md Block 25 + amendments): s_{q,k} = Eqs. 5-6 of the segment's
capped cylinder alone -- against an independent numpy/scipy statement for a
flat segment [1e-6: sampled vs analytic Gaussian] and against Block 4
(render_transmittance + pixel_integrate) for a tilted one under both
absorptions [1e-9]; totals per plane = iint A_q [relative 1e-8]; reach: a
segment beyond shadow_reach_sigmas SDs contributes exactly 0; additivity
[1e-12]; `kept` follows the floor; `include`; `overlap`; the hybrid's A_q =
column sum of render.absorbed_fractions [1e-10]; faint limit [5e-3];
stretch_profiles = sample_profile on shadow_image's planes [1e-10]; the soma
sphere's column total (vertical Beer-Lambert) against its analytic integral;
pixel-centre convention (symmetry about a pixel row); the ValueErrors.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import integrate

import nc_helpers as H
from allen_diameter.analysis.node_features import FeaturedScene
from allen_diameter.analysis.profiles import sample_profile
from allen_diameter.model import geometry, render, ray_world
from allen_diameter.model import shadow as SH
from allen_image_io import CropFrame

P = 0.1144
SMALL_RAYS = {"cleaning.shadow_ray_n_rho": 4, "cleaning.shadow_ray_n_psi": 8,
              "renderer.ray_n_rho": 4, "renderer.ray_n_psi": 8}


def fscene(segs, soma=None):
    """segs: list of (p0 (3,), p1 (3,), d, mu, (child, parent))."""
    n = len(segs)
    xyz = np.array([list(s[0]) + list(s[1]) for s in segs], dtype=float).reshape(n, 6)
    d = np.array([[s[2], s[2]] for s in segs], dtype=float).reshape(n, 2)
    mu = np.array([[s[3], s[3]] for s in segs], dtype=float).reshape(n, 2)
    ids = np.array([list(s[4]) for s in segs], dtype=np.int64).reshape(n, 2)
    soma = np.empty((0, 3)) if soma is None else np.asarray(soma, dtype=float).reshape(-1, 3)
    return FeaturedScene(seg_xyz=xyz, seg_d_um=d, seg_mu_per_um=mu, seg_ids=ids,
                         seg_type=np.full(n, 3, dtype=np.int64), soma=soma)


OWN = ((0.0, -20.0, 0.0), (0.5, -20.0, 0.0), 0.5, 1.0, (2, 1))     # own stretch, far away


def test_flat_segment_against_independent_reference():
    cfg = H.cfg_with({"cleaning.shadow_absorption": "partition_vertical"})
    # flat neighbour along x at y = 0, z = 0.1, length 2 um
    seg = ((-1.0, 0.0, 0.1), (1.0, 0.0, 0.1), 0.6, 1.5, (4, 3))
    fs = fscene([OWN, seg])
    left, top, W, Hh = -12, -8, 24, 16
    z = np.array([0.0, 0.28, 0.84])
    S = SH.shadow_image(fs, {1, 2}, left, top, W, Hh, z, cfg).S
    # reference on a larger block of the same lattice (holds the whole footprint and blur), cropped
    pad = 40
    xs, ys = H.fine_lattice(left - pad, top - pad, W + 2 * pad, Hh + 2 * pad, P, 8)
    tau, _, _ = H.ref_render_flat([((0.0, 0.0, 0.1), 0.3, 0.0, 1.0)], [1.5], xs, ys, z, cfg.renderer,
                                  "partition_vertical")
    ref = H.pixel_mean(1.0 - tau, 8)[:, pad:pad + Hh, pad:pad + W]
    err = float(np.max(np.abs(S - ref)))
    print("flat segment vs independent reference:", err, "peak", float(ref.max()))
    assert err <= 1e-6


@pytest.mark.parametrize("absorption", ["partition_vertical", "hybrid"])
def test_tilted_segment_against_block4(absorption):
    # Block 4 reference without its far-field split (its coarse-grid path is an approximation of its own)
    cfg = H.cfg_with(dict(SMALL_RAYS, **{"cleaning.shadow_absorption": absorption, "renderer.absorption": absorption,
                                          "renderer.fft_split_sigma_um": 50.0}))
    phi, theta, L = math.radians(30), 0.7, 2.0
    c = np.array([0.2, 0.1, 0.05])
    t3 = np.array([math.cos(phi) * math.cos(theta), math.cos(phi) * math.sin(theta), math.sin(phi)])
    p0, p1 = c - 0.5 * L * t3, c + 0.5 * L * t3
    seg = (tuple(p0), tuple(p1), 0.6, 1.5, (4, 3))
    fs = fscene([OWN, seg])
    left, top, W, Hh = -14, -14, 28, 28
    z = np.array([-0.28, 0.0, 0.56])
    S = SH.shadow_image(fs, {1, 2}, left, top, W, Hh, z, cfg).S
    tube = geometry.Tube(c=tuple(c), r=0.3, phi=phi, theta=theta, half_length=0.5 * L, end_cut="axial")
    pad = 45
    g, inner = render.block_fine_grid(left, top, W, Hh, P, 8, pad)
    rc = cfg.renderer
    tau = render.render_transmittance(tube, 1.5, z, g, rc,
                                      reduce=lambda pl: render.pixel_integrate(pl[inner[0], inner[1]], 8)).tau
    err = float(np.max(np.abs(S - (1 - tau))))
    print(absorption, "tilted segment vs Block 4:", err)
    assert err <= 1e-9


def test_totals_reach_additivity_kept_include():
    cfg = H.cfg_with({"cleaning.shadow_absorption": "partition_vertical"})
    a = ((-1.0, 0.5, 0.0), (1.0, 0.8, 0.3), 0.5, 1.0, (4, 3))
    b = ((0.0, -1.0, -0.3), (0.3, 1.0, 0.2), 0.3, 0.2, (6, 5))
    far = ((60.0, 60.0, 0.0), (61.0, 60.0, 0.0), 0.5, 1.0, (8, 7))   # 60 um away: beyond 6 sigma at every plane
    left, top, W, Hh = -60, -60, 120, 120
    z = np.array([0.0, 0.28, 0.84])
    fs = fscene([OWN, a, b, far])
    r = SH.shadow_image(fs, {1, 2}, left, top, W, Hh, z, cfg)
    # totals per plane equal and = iint A (A on a fine grid of the patch lattice)
    tot = r.S.sum(axis=(1, 2)) * P * P
    xs, ys = H.fine_lattice(left, top, W, Hh, P, 8)
    g = render.FineGrid(x0=xs[0], y0=ys[0], h=P / 8, nx=xs.size, ny=ys.size)
    A = sum(SH.segment_absorbed(fs, q, g, cfg)[0] for q in (1, 2))
    assert_allclose(tot, A.sum() * (P / 8) ** 2, rtol=1e-8)
    # reach: the far segment contributes exactly 0
    assert r.per_neighbour_peak[2] == 0.0
    # additivity
    ra = SH.shadow_image(fscene([OWN, a]), {1, 2}, left, top, W, Hh, z, cfg).S
    rb = SH.shadow_image(fscene([OWN, b]), {1, 2}, left, top, W, Hh, z, cfg).S
    assert np.max(np.abs(r.S - (ra + rb))) <= 1e-12
    # kept follows the floor: B x peak >= floor
    B = 210.0
    rk = SH.shadow_image(fs, {1, 2}, left, top, W, Hh, z, cfg, B_gl=B)
    exp_kept = B * rk.per_neighbour_peak >= cfg.cleaning.shadow_floor_gl
    assert_array_equal(rk.kept, exp_kept)
    S_exp = (ra if exp_kept[0] else 0) + (rb if exp_kept[1] else 0)
    assert np.max(np.abs(rk.S - S_exp)) <= 1e-12
    # include=False leaves a neighbour out, its peak still computed
    ri = SH.shadow_image(fs, {1, 2}, left, top, W, Hh, z, cfg, include=[True, False, True])
    assert not ri.kept[1] and ri.per_neighbour_peak[1] == r.per_neighbour_peak[1]
    assert np.max(np.abs(ri.S - ra)) <= 1e-12


def test_overlap_above_and_aside():
    cfg = H.cfg_with({"cleaning.shadow_absorption": "partition_vertical"})
    own = ((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), 0.5, 1.0, (2, 1))
    above = ((-1.0, 0.0, 2.0), (1.0, 0.0, 2.0), 0.5, 1.0, (4, 3))     # directly above, unattached
    aside = ((-1.0, 3.0, 0.0), (1.0, 3.0, 0.0), 0.5, 1.0, (4, 3))
    z = np.array([0.0])
    r1 = SH.shadow_image(fscene([own, above]), {1, 2}, -15, -15, 30, 30, z, cfg)
    r2 = SH.shadow_image(fscene([own, aside]), {1, 2}, -15, -40, 30, 80, z, cfg)
    assert r1.overlap.any() and not r2.overlap.any()
    assert r1.overlap.shape == (30, 30)


def test_hybrid_column_sum_and_faint_limit():
    cfg = H.cfg_with(dict(SMALL_RAYS, **{"cleaning.shadow_absorption": "hybrid"}))
    phi, theta, L = math.radians(25), 0.3, 1.6
    c = np.array([0.0, 0.0, 0.0])
    t3 = np.array([math.cos(phi) * math.cos(theta), math.cos(phi) * math.sin(theta), math.sin(phi)])
    for mu, d in ((1.5, 0.6), (1e-3 / 0.4, 0.4)):
        seg = (tuple(c - 0.5 * L * t3), tuple(c + 0.5 * L * t3), d, mu, (4, 3))
        fs = fscene([OWN, seg])
        xs, ys = H.fine_lattice(-12, -10, 24, 20, P, 8)
        g = render.FineGrid(x0=xs[0], y0=ys[0], h=P / 8, nx=xs.size, ny=ys.size)
        A, _ = SH.segment_absorbed(fs, 1, g, cfg)
        tube = geometry.Tube(c=tuple(c), r=0.5 * d, phi=phi, theta=theta, half_length=0.5 * L, end_cut="axial")
        S_, W_ = ray_world.directions(4, 8, cfg.renderer.ray_na, cfg.renderer.ray_n_oil)
        zeta = render.slab_grid(tube, g, cfg.renderer.dzeta_um)
        ref = render.absorbed_fractions(tube, mu, g, zeta, cfg.renderer.dzeta_um, 1, "hybrid", (S_, W_)).sum(axis=0)
        assert np.max(np.abs(A - ref)) <= 1e-10
        if mu * d < 0.01:
            X, Y = g.mesh()
            zl, zh, ins = geometry.column_interval(X, Y, tube)
            k_disc = float(np.sum(W_ / S_[:, 2]))
            k = H.mean_inv_sz(cfg.renderer.ray_na, cfg.renderer.ray_n_oil)
            ratio = A[ins].sum() / (k * mu * (zh - zl)[ins].sum())
            print("faint limit A/(1.4470 mu l) =", ratio, "quadrature <1/s_z> =", k_disc, "analytic", k)
            assert abs(ratio - 1.0) <= 5e-3


def test_stretch_profiles_equal_sample_profile():
    cfg = H.cfg_with({"cleaning.shadow_absorption": "partition_vertical"})
    seg = ((-1.0, 0.2, 0.0), (1.0, 0.4, 0.28), 0.5, 1.0, (4, 3))
    fs = fscene([OWN, seg])
    ks = np.array([-1, 0, 2])
    o, yh, eu = np.array([0.03, 0.31]), np.array([-math.sin(0.1), math.cos(0.1)]), np.array([math.cos(0.1), math.sin(0.1)])
    v = np.arange(-15, 16) * P
    prof = SH.stretch_profiles(fs, {3, 4}, ks, o, yh, eu, v, 2, 0.1144, cfg)
    left, top, W, Hh = -30, -30, 60, 60
    S = SH.shadow_image(fs, {1, 2}, left, top, W, Hh, ks * cfg.acquisition.dz_um, cfg).S
    fr = CropFrame(left, top, 0, P)
    ref = np.stack([sample_profile(S[i], fr, o, yh, eu, v, 2, 0.1144) for i in range(3)])
    ref2 = np.stack([H.bilinear_read(S[i], left, top, P, o, yh, eu, v, 2, 0.1144) for i in range(3)])
    assert prof.shape == (3, v.size)
    assert np.max(np.abs(prof - ref)) <= 1e-10
    assert np.max(np.abs(ref - ref2)) <= 1e-12


@pytest.mark.parametrize("absorption", ["partition_vertical", "hybrid"])
def test_soma_total_against_analytic(absorption):
    cfg = H.cfg_with(dict(SMALL_RAYS, **{"cleaning.shadow_absorption": absorption, "cleaning.soma_mu_per_um": 0.3,
                                          "cleaning.soma_z_um": 0.0}))
    rs = 1.0                                           # slabs within 1 um of the plane: sigma <= 0.73 um
    fs = fscene([OWN], soma=[(0.0, 0.0, rs)])
    z = np.array([0.0])
    left, top, W, Hh = -80, -80, 160, 160              # margin 8 um > 10 sigma
    r = SH.shadow_image(fs, {1, 2}, left, top, W, Hh, z, cfg)
    tot = float(r.S.sum() * P * P)
    mu = 0.3
    ana = integrate.quad(lambda q: 2 * math.pi * q * (-math.expm1(-2 * mu * math.sqrt(max(rs * rs - q * q, 0.0)))),
                         0, rs, limit=200)[0]
    rel = abs(tot - ana) / ana
    print(absorption, "soma total", tot, "analytic", ana, "rel", rel)
    # tolerance: the disc sampled on a p/8 lattice; edge error ~ (h / r)^(3/2) ~ 1e-4
    assert rel <= 1e-3


def test_vertical_segment_total():
    cfg = H.cfg_with({"cleaning.shadow_absorption": "partition_vertical"})
    seg = ((0.0, 0.0, -1.0), (0.0, 0.0, 1.0), 0.8, 0.7, (4, 3))
    fs = fscene([OWN, seg])
    r = SH.shadow_image(fs, {1, 2}, -40, -40, 80, 80, np.array([0.0]), cfg)
    tot = float(r.S.sum() * P * P)
    ana = math.pi * 0.4 ** 2 * (-math.expm1(-0.7 * 2.0))
    print("vertical segment total", tot, "analytic", ana)
    assert abs(tot - ana) / ana <= 5e-3


def test_pixel_centre_convention():
    cfg = H.cfg_with({"cleaning.shadow_absorption": "partition_vertical"})
    top, m = -10, 10                                    # row m of the patch is centred at y = (top + m) p = 0
    seg = ((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), 0.5, 1.0, (4, 3))
    S = SH.shadow_image(fscene([OWN, seg]), {1, 2}, -15, top, 30, 21, np.array([0.0, 0.28]), cfg).S
    assert_allclose(S[:, m - 5:m][:, ::-1], S[:, m + 1:m + 6], atol=1e-12)


def test_value_errors():
    cfg = H.cfg_with({"cleaning.shadow_absorption": "partition_vertical"})
    seg = ((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), 0.5, 1.0, (4, 3))
    fs = fscene([OWN, seg])
    with pytest.raises(ValueError):
        SH.shadow_image(fs, {77, 78}, 0, 0, 5, 5, [0.0], cfg)
    with pytest.raises(ValueError):
        SH.shadow_image(fs, {1, 2}, 0, 0, 5, 5, [np.nan], cfg)
    with pytest.raises(ValueError):
        SH.shadow_image(fs, {1, 2}, 0, 0, 5, 5, [0.0], cfg, include=[True, True])
    with pytest.raises(ValueError):
        SH.shadow_image(fs, {1, 2}, 0, 0, 5, 5, [0.0], cfg, B_gl=0.0)
    with pytest.raises(ValueError):
        SH.shadow_image(fs, {1, 2}, 0, 0, 5, 5, [0.0], H.cfg_with({"cleaning.shadow_absorption": "linear"}))
