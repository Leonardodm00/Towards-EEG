"""Spec-derived tests, Block 28 (the cleaning operator and the two tiers, amendment 1) -- tester 2026-11-13.

Oracles (SPEC.md Block 28): S = 0 returns the block as float64 [exact];
exact inverse with Block 24 (partition, linear): the scene cleaned with the
neighbour's deficit rendered alone equals the node alone [1e-9 gl at B =
210]; the same through the real producer, Block 25's S (the boundary
24 -> 25 -> 28) [1e-9 x B]; added_gl = mean of B S [1e-12]; the three forms
of B_gl agree [exact]; over_background by hand; tiers on the constructed
scenes; the order of neighbour_tiers = shadow.neighbour_indices; the
ValueErrors; the hybrid residual reported (no bound, amendment 1 (ii)).
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_allclose

import nc_helpers as H
from allen_diameter.analysis import cleaning as CL
from allen_diameter.analysis.node_features import FeaturedScene
from allen_diameter.model import geometry, render
from allen_diameter.model import scene_render as SR
from allen_diameter.model import shadow as SH

P = 0.1144
B = 210.0


def fscene(segs, soma=None):
    n = len(segs)
    xyz = np.array([list(s[0]) + list(s[1]) for s in segs], dtype=float).reshape(n, 6)
    d = np.array([[s[2], s[2]] for s in segs], dtype=float).reshape(n, 2)
    mu = np.array([[s[3], s[3]] for s in segs], dtype=float).reshape(n, 2)
    ids = np.array([list(s[4]) for s in segs], dtype=np.int64).reshape(n, 2)
    soma = np.empty((0, 3)) if soma is None else np.asarray(soma, dtype=float).reshape(-1, 3)
    return FeaturedScene(xyz, d, mu, ids, np.full(n, 3, dtype=np.int64), soma)


def render_block(tubes, mus, ks, cfg, left, top, W, Hh, pad=40):
    g, inner = render.block_fine_grid(left, top, W, Hh, P, 8, pad)
    return SR.render_scene(tubes, mus, np.asarray(ks) * cfg.acquisition.dz_um, g, cfg.renderer,
                           reduce=lambda pl: render.pixel_integrate(pl[inner[0], inner[1]], 8)).tau


Y_NB = 16 * P                                   # whole pixels: the two tubes' columns, hence lattices, coincide


@pytest.mark.parametrize("absorption", ["partition_vertical", "linear"])
def test_exact_inverse_block24(absorption):
    cfg = H.cfg_with({"renderer.absorption": absorption, "renderer.fft_split_sigma_um": 50.0})
    node = geometry.Tube(c=(0.0, 0.0, 0.0), r=0.3, phi=0.0, theta=0.0, half_length=1.0)
    nb = geometry.Tube(c=(0.0, Y_NB, 0.0), r=0.3, phi=0.0, theta=0.0, half_length=1.0)
    ks = [-1, 0, 2]
    left, top, W, Hh = -20, -15, 40, 40
    sc = render_block([node, nb], [1.5, 2.0], ks, cfg, left, top, W, Hh)
    alone_nb = render_block([nb], [2.0], ks, cfg, left, top, W, Hh)
    alone_node = render_block([node], [1.5], ks, cfg, left, top, W, Hh)
    res = CL.clean_block(B * sc, np.clip(1 - alone_nb, 0, None), B, cfg)
    err = float(np.max(np.abs(res.block - B * alone_node)))
    print(absorption, "exact inverse residual (gl):", err)
    assert err <= 1e-9 * B


@pytest.mark.parametrize("absorption", ["partition_vertical", "hybrid"])
def test_boundary_block24_to_25_to_28(absorption):
    """The real producer chain: a two-tube scene by Block 24, the neighbour's S by Block 25, cleaned by Block 28."""
    ov = {"renderer.absorption": absorption, "cleaning.shadow_absorption": absorption,
          "renderer.fft_split_sigma_um": 50.0, "renderer.ray_n_rho": 4, "renderer.ray_n_psi": 8,
          "cleaning.shadow_ray_n_rho": 4, "cleaning.shadow_ray_n_psi": 8}
    cfg = H.cfg_with(ov)
    node = geometry.Tube(c=(0.0, 0.0, 0.0), r=0.3, phi=0.0, theta=0.0, half_length=1.0)
    nb = geometry.Tube(c=(0.0, Y_NB, 0.0), r=0.3, phi=0.0, theta=0.0, half_length=1.0)
    ks = np.array([-1, 0, 2])
    left, top, W, Hh = -20, -15, 40, 40
    I_scene = B * render_block([node, nb], [1.5, 2.0], ks, cfg, left, top, W, Hh)
    I_node = B * render_block([node], [1.5], ks, cfg, left, top, W, Hh)
    fs = fscene([((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), 0.6, 1.5, (2, 1)),
                 ((-1.0, Y_NB, 0.0), (1.0, Y_NB, 0.0), 0.6, 2.0, (4, 3))])
    S = SH.shadow_image(fs, {1, 2}, left, top, W, Hh, ks * cfg.acquisition.dz_um, cfg).S
    res = CL.clean_block(I_scene, S, B, cfg)
    err = float(np.max(np.abs(res.block - I_node)))
    print(absorption, "Block 24 scene cleaned with Block 25's S, residual vs node alone (gl):", err)
    if absorption == "partition_vertical":
        assert err <= 1e-9 * B
    # hybrid: amendment 1 (ii) -- reported, no bound


def test_operator_algebra():
    cfg = H.cfg_with()
    rng = np.random.default_rng(28)
    blk = rng.integers(0, 256, size=(3, 7, 9)).astype(np.uint8)
    res0 = CL.clean_block(blk, np.zeros((3, 7, 9)), 200.0, cfg)
    assert res0.block.dtype == np.float64
    assert_array_equal(res0.block, blk.astype(np.float64))
    S = rng.random((3, 7, 9)) * 0.2
    Bp = np.array([190.0, 200.0, 210.0])
    r_scalar_like = CL.clean_block(blk, S, np.full((3, 7, 9), 200.0), cfg)
    r_scalar = CL.clean_block(blk, S, 200.0, cfg)
    r_plane = CL.clean_block(blk, S, np.full(3, 200.0), cfg)
    assert_array_equal(r_scalar.block, r_scalar_like.block)
    assert_array_equal(r_scalar.block, r_plane.block)
    r = CL.clean_block(blk, S, Bp, cfg)
    BS = Bp[:, None, None] * S
    assert_allclose(r.added_gl, BS.mean(axis=(1, 2)), rtol=0, atol=1e-12)
    assert_array_equal(r.block, blk.astype(float) + BS)
    thr = cfg.cleaning.over_background_gl
    assert_array_equal(r.over_background, ((blk + BS - Bp[:, None, None]) > thr).mean(axis=(1, 2)))
    assert_array_equal(blk, blk)        # input unchanged (uint8 copy semantics)


def test_value_errors():
    cfg = H.cfg_with()
    blk = np.full((2, 3, 3), 100.0)
    with pytest.raises(ValueError):
        CL.clean_block(blk, np.zeros((2, 3, 4)), 200.0, cfg)
    bad = np.zeros((2, 3, 3)); bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        CL.clean_block(blk, bad, 200.0, cfg)
    bad = np.zeros((2, 3, 3)); bad[0, 0, 0] = -1e-3
    with pytest.raises(ValueError):
        CL.clean_block(blk, bad, 200.0, cfg)
    for Bb in (0.0, -1.0, np.nan):
        with pytest.raises(ValueError):
            CL.clean_block(blk, np.zeros((2, 3, 3)), Bb, cfg)


OWN = ((-3.0, 0.0, 0.0), (3.0, 0.0, 0.0), 0.5, 1.0, (2, 1))     # own stretch, nodes 1 and 2; the node at the origin


def tiers_of(extra):
    cfg = H.cfg_with()
    fs = fscene([OWN, extra])
    t = CL.neighbour_tiers(fs, [1, 2], (0.0, 0.0, 0.0), cfg)
    segs, somas = SH.neighbour_indices(fs, {1, 2})
    assert t.shape == (segs.size + somas.size,)
    return int(t[0])


def test_tiers_constructed_scenes():
    assert tiers_of(((-1.0, 5.0, 0.0), (1.0, 5.0, 0.0), 0.5, 1.0, (4, 3))) == 1      # far lateral
    assert tiers_of(((-1.0, 1.0, 3.0), (1.0, 1.0, 3.0), 0.3, 1.0, (4, 3))) == 1      # far in depth, no overlap
    assert tiers_of(((-1.0, 1.0, 0.2), (1.0, 1.0, 0.2), 0.5, 1.0, (4, 3))) == 2      # close and in focus
    assert tiers_of(((0.0, -3.0, 2.5), (0.0, 3.0, 2.5), 0.5, 1.0, (4, 3))) == 2      # directly above, unattached
    assert tiers_of(((3.0, 0.0, 0.0), (6.0, 3.0, 0.0), 0.5, 1.0, (4, 2))) == 1       # attached at node 2, 3 um away
    # attached and lying over the own stretch near the node: rule (b) excludes attached segments
    assert tiers_of(((3.0, 0.0, 0.0), (-2.0, 0.1, 2.5), 0.5, 1.0, (4, 2))) == 1


def test_tiers_order_with_soma_and_mixed():
    cfg = H.cfg_with()
    segs = [OWN,
            ((-1.0, 5.0, 0.0), (1.0, 5.0, 0.0), 0.5, 1.0, (4, 3)),
            ((-1.0, 1.0, 0.2), (1.0, 1.0, 0.2), 0.5, 1.0, (6, 5)),
            ((-3.0, 0.0, 0.0), (-1.0, 0.0, 0.0), 0.5, 1.0, (1, 7))]
    fs = fscene(segs, soma=[(0.0, 0.0, 3.0)])
    t = CL.neighbour_tiers(fs, [1, 2], (0.0, 0.0, 0.0), cfg)
    s, so = SH.neighbour_indices(fs, {1, 2})
    assert t.size == s.size + so.size == 4
    assert t[-1] == 1                     # soma discs always tier 1
    assert t[0] == 1 and t[1] == 2


def test_own_ids_as_set_accepted_like_block25():
    """Finding T-own-ids: Block 25 (shadow_image, neighbour_indices) accepts own_ids as a set;
    Block 28's neighbour_tiers, documented to share that order and input, raises TypeError on one."""
    cfg = H.cfg_with()
    fs = fscene([OWN, ((-1.0, 5.0, 0.0), (1.0, 5.0, 0.0), 0.5, 1.0, (4, 3))])
    SH.neighbour_indices(fs, {1, 2})                       # accepted
    t = CL.neighbour_tiers(fs, {1, 2}, (0.0, 0.0, 0.0), cfg)
    assert t.tolist() == [1]
