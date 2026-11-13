"""Smoke test for the scene renderer, several tubes in one stack -- Block 24 in
specs/SPEC.md (section 2.3 applied jointly; D-130.O11).

Checks
    test_known_answer   C1 for a scene: sum_j Delta A_j = 1 - exp(-sum_j a_j) per
                        column under the partition, a column through two stacked
                        tubes included (both light directions); the tube second
                        in the light's path books what it books alone times
                        exp(-mu_1 l_1) of the first tube's column, the first
                        exactly what it books alone; hybrid faint limit: at
                        mu d = 1e-3 the scene is `linear` at <1/s_z> mu_t
    test_reference      one tube equals Block 4's render_transmittance in the
                        three modes, fft (with far pairs) and direct; two tubes
                        with disjoint projections: 1 - tau = sum of each alone
                        on the same slab lattice (linear, partition); the
                        partition against T_<j = exp(-sum_{j'<j} a_j') written
                        literally; the hybrid's joint Delta A_j against a brute
                        force from Block 2's line_interval
    test_convergence    SKIP (see the check)
    test_invariants     every plane carries the same total (three modes); light
                        reversed = depth mirrored (three modes); the labelled
                        comparison: the product of separately rendered
                        transmittances (and make_provider's block) within
                        max (1 - tau_1)(1 - tau_2) of the scene
    test_contract       render_scene's RenderResult and the reduce path;
                        synthetic_scene_block in the fetch_zblock contract
    test_determinism    render_scene twice and synthetic_scene_block under one
                        seed bit-identical
    test_edge_cases     a tube with mu = 0 changes nothing (exact); all mu = 0
                        and a tube off the grid give tau = 1; ValueError for
                        unequal lengths, a negative or NaN mu, an empty scene,
                        the ray world

Sizes: fine grids at h = p_x / 4 = 0.0286 um (the narrowest kernel, sigma_r(0)
= 0.08 um, needs h <= 0.04 um), 105 x 105 to 295 x 295 samples; tubes d = 0.3
to 0.5 um, half-lengths 0.5 to 0.8 um; the hybrid at 8 x 16 directions. About
half a minute on one core.

Run
    cd "Passive Features/Diameter Re-measurement"
    python tests/smoke/test_smoke_scene_render.py

Keep this file pure ASCII (hpc-python-compat).
"""
from __future__ import annotations

import dataclasses
import importlib.metadata
import math
import platform
import re
import sys
import time
import traceback
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal  # noqa: F401

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
SRC = WS / "src"
for p in (str(SRC), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from allen_diameter.config import default_config  # noqa: E402
from allen_diameter.model import geometry as G  # noqa: E402
from allen_diameter.model import ray_world as RW  # noqa: E402
from allen_diameter.model import render as R  # noqa: E402
from allen_diameter.model import scene_render as SR  # noqa: E402

SEED = 20261113
REPORT_PACKAGES = ("numpy", "scipy")
ARTIFACT_DIR = HERE / "artifacts"
_SAVED_ARTIFACTS = []
PX = 0.1144
H = PX / 4.0
MODES = ("linear", "partition_vertical", "hybrid")


# --------------------------------------------------------------- helpers ---

def make_rng(seed: int = SEED) -> np.random.Generator:
    return np.random.default_rng(seed)


def check_close(name, actual, expected, rtol=1e-12, atol=0.0):
    """numpy.testing.assert_allclose that keeps the evidence on failure."""
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


def rcfg(absorption="partition_vertical", **changes):
    """Default RendererConfig with the absorption chosen; the hybrid at 8 x 16 directions (cost)."""
    base = dict(absorption=absorption, ray_n_rho=8, ray_n_psi=16)
    base.update(changes)
    return dataclasses.replace(default_config().renderer, **base)


def centred_grid(half_um, h=H):
    """Square grid with a sample at the origin; tube centres at multiples of h sample alike."""
    m = int(round(half_um / h))
    return R.FineGrid(-m * h, -m * h, h, 2 * m + 1, 2 * m + 1)


def pair_side_by_side():
    """Two equal tilted tubes whose projections are disjoint (v-distance 1.0 um > 2 r);
    centres at multiples of h, so both depth ranges on a centred grid are equal."""
    phi, theta = math.radians(20.0), 1.3
    a = G.Tube((-18 * H, 0.0, 0.0), 0.25, phi, theta, 1.0, 0.7, "axial")
    b = G.Tube((20 * H, 4 * H, 0.0), 0.25, phi, theta, 1.0, 0.7, "axial")
    return a, b


def pair_stacked(U=0.5, r=0.15):
    """Two tubes crossing at the origin in projection, separated in depth:
    `lower` (smaller z, along x) and `upper` (larger z, along y)."""
    lower = G.Tube((0.0, 0.0, -0.3), r, math.radians(10.0), 0.0, 1.0, U, "axial")
    upper = G.Tube((0.0, 0.0, 0.3), r, math.radians(5.0), 0.5 * math.pi, 1.0, U, "axial")
    return lower, upper


def column_length(tube, grid):
    X, Y = grid.mesh()
    z_lo, z_hi, _ = G.column_interval(X, Y, tube)
    return z_hi - z_lo                                   # 0 where the column misses


def hybrid_reference(points, tubes, mus, S, W, light):
    """Brute force of the joint h from Block 2's line_interval (Block 21's reference form)."""
    total = np.zeros((points.shape[0], S.shape[0]))
    for tube, mu in zip(tubes, mus):
        t1, t2, hit = G.line_interval(points[:, None, :], S[None, :, :], tube)
        with np.errstate(invalid="ignore"):
            ell = np.minimum(t2, 0.0) - t1 if light == +1 else t2 - np.maximum(t1, 0.0)
        total += mu * np.where(hit, np.clip(ell, 0.0, None), 0.0)
    return (np.exp(-total) / S[None, :, 2]) @ W


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    lower, upper = pair_stacked()
    mus = (2.0, 3.0)
    g = centred_grid(1.5)
    dz = default_config().renderer.dzeta_um
    l_low, l_up = column_length(lower, g), column_length(upper, g)
    both = (l_low > 0) & (l_up > 0)
    assert both[g.ny // 2, g.nx // 2] and both.sum() > 20, "the stacked scene must have columns through both tubes"
    for light in (+1, -1):
        zeta, dA = SR._absorbed_fractions([lower, upper], mus, g, dz, light, "partition_vertical")
        _, a = SR._absorbed_fractions([lower, upper], mus, g, dz, light, "linear", zeta=zeta)
        # C1: the partition telescopes per column; sum_j a_j = mu_1 l_1 + mu_2 l_2 when the lattice tiles the
        # columns (checked against the closed-form lengths). 1e-12: roundoff of ~70 summed slabs.
        check_close("C1 sum a_j light %+d" % light, a.sum(axis=0), mus[0] * l_low + mus[1] * l_up, rtol=0, atol=1e-12)
        check_close("C1 partition light %+d" % light, dA.sum(axis=0), -np.expm1(-a.sum(axis=0)), rtol=0, atol=1e-12)
        # the tube second in the light's path books its own booking times exp(-mu_1 l_1) of the first one's column
        first, second = ((lower, mus[0]), (upper, mus[1])) if light == +1 else ((upper, mus[1]), (lower, mus[0]))
        _, alone_first = SR._absorbed_fractions([first[0]], [first[1]], g, dz, light, "partition_vertical", zeta=zeta)
        _, alone_second = SR._absorbed_fractions([second[0]], [second[1]], g, dz, light, "partition_vertical",
                                                 zeta=zeta)
        assert not np.any((alone_first > 0) & (alone_second > 0)), "the stacked tubes share a slab"
        on_first = alone_first > 0
        check_close("first tube unchanged light %+d" % light, dA[on_first], alone_first[on_first], rtol=0, atol=1e-10)
        factor = np.exp(-first[1] * column_length(first[0], g))
        on_second = alone_second > 0
        expected = (alone_second * factor[None, :, :])[on_second]
        check_close("second tube shadowed light %+d" % light, dA[on_second], expected, rtol=0, atol=1e-10)
        assert np.all(dA[on_second] <= alone_second[on_second]), "the second tube booked more than alone"
        assert (dA[on_second] < alone_second[on_second] - 1e-3).any(), "no column of the second tube was shadowed"

    # hybrid faint limit: mu d = 1e-3 (d = 0.3 um); the scene is `linear` at <1/s_z>_W mu_t.
    # Relative 5e-3 on the planes' dips (SPEC); the O(mu L) correction is ~1e-3 here.
    rc_h, rc_l = rcfg("hybrid"), rcfg("linear")
    S, W = RW.directions(rc_h.ray_n_rho, rc_h.ray_n_psi, rc_h.ray_na, rc_h.ray_n_oil)
    m_inv = RW.mean_inverse_cos(S, W)
    assert abs(m_inv - 1.4470) < 1e-3, m_inv
    mu = 1e-3 / (2.0 * lower.r)
    zp = [-0.28, 0.0, 0.28]
    th = SR.render_scene([lower, upper], [mu, mu], zp, g, rc_h).tau
    tl = SR.render_scene([lower, upper], [m_inv * mu, m_inv * mu], zp, g, rc_l).tau
    for k in range(len(zp)):
        dip_h, dip_l = 1.0 - th[k], 1.0 - tl[k]
        rel = float(np.abs(dip_h - dip_l).max() / dip_l.max())
        assert rel < 5e-3, "faint limit plane %d: relative difference %.3e" % (k, rel)


def test_reference():
    # one tube equals Block 4 in the three modes; 1e-10 (SPEC). Planes up to 1.0 um away -> far pairs too.
    tube = G.Tube((0.02, -0.01, 0.1), 0.25, math.radians(25.0), 0.4, 1.0, 0.8, "axial")
    g = centred_grid(1.7)
    zp = np.array([-1.0, -0.28, 0.0, 0.28, 0.56])
    for absorption in MODES:
        for backend in ("fft", "direct"):
            rc = rcfg(absorption, backend=backend)
            ref = R.render_transmittance(tube, 1.5, zp, g, rc, -1)
            got = SR.render_scene([tube], [1.5], zp, g, rc, -1)
            check_close("one tube %s %s" % (absorption, backend), got.tau, ref.tau, rtol=0, atol=1e-10)
            assert_array_equal(got.zeta, ref.zeta)
            assert (got.n_slabs_used, got.n_near_pairs, got.n_far_pairs) == \
                   (ref.n_slabs_used, ref.n_near_pairs, ref.n_far_pairs)
            if backend == "fft":
                assert ref.n_far_pairs > 0, "the one-tube comparison should exercise the far-field path"

    # disjoint projections, linear and partition: 1 - tau_scene = (1 - tau_1) + (1 - tau_2) on one lattice; 1e-10
    a, b = pair_side_by_side()
    g2 = centred_grid(2.0)
    zp2 = [-0.56, -0.28, 0.0, 0.28]
    for absorption in ("linear", "partition_vertical"):
        rc = rcfg(absorption)
        sc = SR.render_scene([a, b], [3.0, 1.2], zp2, g2, rc)
        ta = R.render_transmittance(a, 3.0, zp2, g2, rc)
        tb = R.render_transmittance(b, 1.2, zp2, g2, rc)
        assert sc.zeta.size == ta.zeta.size == tb.zeta.size
        check_close("same lattice a", ta.zeta, sc.zeta, rtol=0, atol=1e-12)
        check_close("same lattice b", tb.zeta, sc.zeta, rtol=0, atol=1e-12)
        check_close("disjoint sum %s" % absorption, 1.0 - sc.tau, (1.0 - ta.tau) + (1.0 - tb.tau), rtol=0, atol=1e-10)

    # the partition against T_<j = exp(-sum_{j' < j} a_j') written literally (cumulative sum in the light's order)
    lower, upper = pair_stacked()
    g3 = centred_grid(1.5)
    dz = default_config().renderer.dzeta_um
    for light in (+1, -1):
        zeta, dA = SR._absorbed_fractions([lower, upper], [2.0, 3.0], g3, dz, light, "partition_vertical")
        _, a_j = SR._absorbed_fractions([lower, upper], [2.0, 3.0], g3, dz, light, "linear", zeta=zeta)
        order = a_j if light == +1 else a_j[::-1]
        before = np.cumsum(order, axis=0) - order       # exclusive cumulative sum
        ref = np.exp(-before) * (-np.expm1(-order))
        ref = ref if light == +1 else ref[::-1]
        check_close("partition literal light %+d" % light, dA, ref, rtol=0, atol=1e-12)

    # hybrid: the joint Delta A_j = sum_t a_tj h(zbar_tj) against a brute force from line_interval; 1e-12
    rc = rcfg("hybrid")
    S, W = RW.directions(rc.ray_n_rho, rc.ray_n_psi, rc.ray_na, rc.ray_n_oil)
    mus = (2.0, 3.0)
    X, Y = g3.mesh()
    cols = [G.column_interval(X, Y, t) for t in (lower, upper)]
    for light in (+1, -1):
        zeta, dA = SR._absorbed_fractions([lower, upper], mus, g3, dz, light, "hybrid", directions=(S, W))
        js = [j for j in range(zeta.size) if j % 7 == 3]
        for j in js:
            expected = np.zeros((g3.ny, g3.nx))
            for (z_lo, z_hi, _), mu in zip(cols, mus):
                a_t = G.slab_absorbance(z_lo, z_hi, zeta[j], dz, mu)
                m = a_t > 0
                if not m.any():
                    continue
                zbar = 0.5 * (np.maximum(z_lo[m], zeta[j] - 0.5 * dz) + np.minimum(z_hi[m], zeta[j] + 0.5 * dz))
                pts = np.column_stack([X[m], Y[m], zbar])
                expected[m] += a_t[m] * hybrid_reference(pts, [lower, upper], mus, S, W, light)
            check_close("hybrid joint slab %d light %+d" % (j, light), dA[j], expected, rtol=0, atol=1e-12)
        # two tubes in one slab of one column (a gap of 0.002 um < dzeta at the crossing): the slab books both
        # (lower on [-0.291, -0.001], upper on [0.001, 0.301]: slab boundaries at -0.291 + 0.02 j, so the slab
        # [-0.011, 0.009] holds the top of one and the bottom of the other)
        lo_t = G.Tube((0.0, 0.0, -0.146), 0.145, 0.0, 0.0, 1.0, 0.5, "axial")
        up_t = G.Tube((0.0, 0.0, 0.151), 0.15, 0.0, 0.5 * math.pi, 1.0, 0.5, "axial")
        zt, dAt = SR._absorbed_fractions([lo_t, up_t], mus, g3, dz, light, "hybrid", directions=(S, W))
        colt = [G.column_interval(X, Y, t) for t in (lo_t, up_t)]
        shared = [j for j in range(zt.size)
                  if all((G.slab_absorbance(c[0], c[1], zt[j], dz, 1.0) > 0).any() for c in colt)]
        both_col = [j for j in shared if ((G.slab_absorbance(colt[0][0], colt[0][1], zt[j], dz, 1.0) > 0)
                                          & (G.slab_absorbance(colt[1][0], colt[1][1], zt[j], dz, 1.0) > 0)).any()]
        assert both_col, "no column holds both touching tubes in one slab"
        for j in shared:
            expected = np.zeros((g3.ny, g3.nx))
            for (z_lo, z_hi, _), mu in zip(colt, mus):
                a_t = G.slab_absorbance(z_lo, z_hi, zt[j], dz, mu)
                m = a_t > 0
                zbar = 0.5 * (np.maximum(z_lo[m], zt[j] - 0.5 * dz) + np.minimum(z_hi[m], zt[j] + 0.5 * dz))
                expected[m] += a_t[m] * hybrid_reference(np.column_stack([X[m], Y[m], zbar]), [lo_t, up_t], mus,
                                                         S, W, light)
            check_close("hybrid shared slab %d light %+d" % (j, light), dAt[j], expected, rtol=0, atol=1e-12)
        # the coupling is active: the tube second in the light's path books less than alone
        second = upper if light == +1 else lower
        mu2 = mus[1] if light == +1 else mus[0]
        _, alone = SR._absorbed_fractions([second], [mu2], g3, dz, light, "hybrid", directions=(S, W), zeta=zeta)
        on = alone > 0
        assert np.all(dA[on] <= alone[on] + 1e-15) and (dA[on] < alone[on] - 1e-3).any()


def test_convergence():
    raise unittest.SkipTest("the slab and direction discretisations are Block 4's and Block 21's, whose tests "
                            "check their convergence; the one-tube oracle ties this renderer to them to roundoff "
                            "and SPEC lists no convergence oracle for Block 24")


def test_invariants():
    # every plane carries the same total: h^2 sum (1 - tau_k) = h^2 sum_j sum Delta A_j. Relative 1e-10 (SPEC).
    # Requires no mass lost off the grid (margin 3.6 um > 7.5 sigma_max, tail < 1e-13) and no far pairs (the
    # far path interpolates a coarse grid and does not conserve mass to 1e-10).
    lower, upper = pair_stacked()
    g = centred_grid(4.2)
    zp = [-0.1, 0.0, 0.1]
    dz = default_config().renderer.dzeta_um
    for absorption in MODES:
        rc = rcfg(absorption)
        res = SR.render_scene([lower, upper], [2.0, 3.0], zp, g, rc)
        assert res.n_far_pairs == 0 and res.sigma_max_um < 0.5, (res.n_far_pairs, res.sigma_max_um)
        dirs = RW.directions(rc.ray_n_rho, rc.ray_n_psi, rc.ray_na, rc.ray_n_oil) if absorption == "hybrid" else None
        zeta, dA = SR._absorbed_fractions([lower, upper], [2.0, 3.0], g, dz, +1, absorption, directions=dirs)
        assert_array_equal(zeta, res.zeta)
        booked = float(dA.sum())
        for k in range(len(zp)):
            check_close("plane total %s k=%d" % (absorption, k), float((1.0 - res.tau[k]).sum()), booked, rtol=1e-10)

    # light reversed = depth mirrored: tau(z_k; +1, scene) = tau(-z_k; -1, mirrored scene: c_z -> -c_z,
    # theta -> theta + pi). The lattice must span the union of depth ranges exactly, so dzeta is set to divide
    # it; even ray_n_psi (16) for the hybrid. 1e-10 (SPEC).
    g = centred_grid(1.5)
    X, Y = g.mesh()
    tubes = [lower, upper]
    lo = min(float(G.column_interval(X, Y, t)[0][G.column_interval(X, Y, t)[2]].min()) for t in tubes)
    hi = max(float(G.column_interval(X, Y, t)[1][G.column_interval(X, Y, t)[2]].max()) for t in tubes)
    dzeta = (hi - lo) / round((hi - lo) / 0.02)
    mirrored = [G.Tube((t.c[0], t.c[1], -t.c[2]), t.r, t.phi, t.theta + math.pi, t.aspect, t.half_length, t.end_cut)
                for t in tubes]
    zp = np.array([-0.56, -0.2, 0.0, 0.28])
    for absorption in MODES:
        rc = rcfg(absorption, dzeta_um=dzeta)
        fwd = SR.render_scene(tubes, [2.0, 3.0], zp, g, rc, +1)
        rev = SR.render_scene(mirrored, [2.0, 3.0], -zp, g, rc, -1)
        check_close("mirror lattice %s" % absorption, -rev.zeta[::-1], fwd.zeta, rtol=0, atol=1e-12)
        check_close("light reversed %s" % absorption, rev.tau, fwd.tau, rtol=0, atol=1e-10)
    # and the light direction matters for these dark stacked tubes (the check is not vacuous)
    rc = rcfg("partition_vertical", dzeta_um=dzeta)
    d_light = np.abs(SR.render_scene(tubes, [2.0, 3.0], zp, g, rc, +1).tau
                     - SR.render_scene(tubes, [2.0, 3.0], zp, g, rc, -1).tau).max()
    assert d_light > 1e-3, d_light

    # labelled comparison: the product of separately rendered transmittances (make_provider's model) differs from
    # the scene by at most max (1 - tau_1)(1 - tau_2) for tubes with disjoint projections [a bound]. For linear and
    # the partition the difference equals (1 - tau_1)(1 - tau_2) pointwise; 1e-12 slack for roundoff.
    a, b = pair_side_by_side()
    g = centred_grid(2.0)
    zp = [-0.56, -0.28, 0.0, 0.28]
    for absorption in MODES:
        rc = rcfg(absorption)
        ta = R.render_transmittance(a, 3.0, zp, g, rc).tau
        tb = R.render_transmittance(b, 3.0, zp, g, rc).tau
        sc = SR.render_scene([a, b], [3.0, 3.0], zp, g, rc).tau
        bound = float(((1.0 - ta) * (1.0 - tb)).max())
        diff = float(np.abs(ta * tb - sc).max())
        assert bound > 1e-4, "the bound is vacuous (%.3e)" % bound
        assert diff <= bound + 1e-12, "%s: product - scene %.3e > bound %.3e" % (absorption, diff, bound)
    # the same through make_provider (tests/smoke/test_smoke_node_pipeline) at block level: no noise, no JPEG, so
    # the grey levels differ by at most B x bound + 1 (two roundings of half a level)
    from test_smoke_node_pipeline import make_provider
    cfg = default_config()
    cfg = dataclasses.replace(cfg, renderer=dataclasses.replace(cfg.renderer, noise_sd_gl=0.0, jpeg=False))
    left, top, width, height, pad = -14, -12, 29, 25, 6
    prov = make_provider([a, b], [3.0, 3.0], cfg, SEED, pad_px=pad)
    pb = prov(left, top, width, height, -2, 1)[0].astype(float)
    sb = SR.synthetic_scene_block([a, b], [3.0, 3.0], np.arange(-2, 2), left, top, width, height, cfg, make_rng(),
                                  pad)[0].astype(float)
    f = R.fine_factor(min(a.d, b.d), cfg.renderer, PX)
    grid, _inner = R.block_fine_grid(left, top, width, height, PX, f, pad)
    zk = np.arange(-2, 2) * cfg.acquisition.dz_um
    t1 = R.render_transmittance(a, 3.0, zk, grid, cfg.renderer).tau
    t2 = R.render_transmittance(b, 3.0, zk, grid, cfg.renderer).tau
    bound_gl = cfg.renderer.background_B_gl * float(((1.0 - t1) * (1.0 - t2)).max()) + 1.0
    assert np.abs(pb - sb).max() <= bound_gl, (np.abs(pb - sb).max(), bound_gl)


def test_contract():
    a, b = pair_side_by_side()
    g = centred_grid(1.5)
    for absorption in MODES:
        rc = rcfg(absorption)
        res = SR.render_scene((a, b), np.array([1.0, 2.0]), [-0.28, 0.0, 0.28], g, rc)
        assert isinstance(res, R.RenderResult)
        assert res.tau.shape == (3, g.ny, g.nx) and res.tau.dtype == np.float64, (res.tau.shape, res.tau.dtype)
        assert np.all(np.isfinite(res.tau)) and res.tau.min() < 0.9 and res.backend == rc.backend
        assert res.zeta.ndim == 1 and np.all(np.diff(res.zeta) > 0) and res.n_slabs_used > 0
        red = SR.render_scene([a, b], [1.0, 2.0], [-0.28, 0.0, 0.28], g, rc, reduce=lambda p: p[::3, 1::2])
        assert_array_equal(red.tau, res.tau[:, ::3, 1::2])
    cfg = default_config()
    ks = np.arange(-1, 2)
    block, ks_out, valid, frame = SR.synthetic_scene_block([a, b], [1.0, 2.0], ks, -12, -10, 25, 21, cfg, make_rng(),
                                                           pad_px=3)
    assert block.shape == (3, 21, 25) and block.dtype == np.uint8, (block.shape, block.dtype)
    assert_array_equal(ks_out, ks)
    assert valid.dtype == bool and valid.shape == (3,) and valid.all()
    assert (frame.left, frame.top, frame.downsample, frame.res0_um_px) == (-12, -10, 0, cfg.acquisition.res0_um)
    assert block.min() < 150 and block.max() > 200, (block.min(), block.max())   # dark tubes on B = 210
    # one tube: Block 4's synthetic_block, bit for bit (same grid, same planes, same camera draws)
    one = SR.synthetic_scene_block([a], [1.0], ks, -12, -10, 25, 21, cfg, make_rng(), pad_px=3)[0]
    assert_array_equal(one, R.synthetic_block(a, 1.0, ks, -12, -10, 25, 21, cfg, make_rng(), pad_px=3)[0])
    # the fine factor is fine_factor at the smallest absorbing diameter: with a fine thin-tube step (p_x / 16)
    # and a coarse thick-tube step (p_x / 3, still resolving sigma_r(0)) a scene of a 0.3 um and a 0.8 um tube
    # renders on the thin tube's grid
    fcfg = dataclasses.replace(cfg, renderer=dataclasses.replace(cfg.renderer, h_g_um_thin=PX / 16.0,
                                                                 h_g_um_thick=PX / 3.0))
    thin = G.Tube((0.0, 0.0, 0.0), 0.15, 0.2, 0.3, 1.0, 0.5, "axial")
    thick = G.Tube((0.3, 0.4, 0.1), 0.4, 0.1, 1.9, 1.0, 0.5, "axial")
    ks2 = np.array([0, 1])
    got = SR.synthetic_scene_block([thick, thin], [1.0, 2.0], ks2, -4, -4, 9, 9, fcfg, make_rng(), pad_px=2)[0]
    f = R.fine_factor(thin.d, fcfg.renderer, PX)
    assert f == 16 and R.fine_factor(thick.d, fcfg.renderer, PX) == 3
    grid, inner = R.block_fine_grid(-4, -4, 9, 9, PX, f, 2)
    tau = SR.render_scene([thick, thin], [1.0, 2.0], ks2 * cfg.acquisition.dz_um, grid, fcfg.renderer).tau
    from allen_diameter.model import camera as CAM
    want = CAM.camera_chain(R.pixel_integrate(tau[:, inner[0], inner[1]], f), fcfg.renderer, make_rng())
    assert_array_equal(got, want)
    src = (SRC / "allen_diameter" / "model" / "scene_render.py").read_bytes()
    assert all(c < 128 for c in src) and b"\r" not in src, "scene_render.py must be pure ASCII with LF"


def test_determinism():
    lower, upper = pair_stacked()
    g = centred_grid(1.0)
    rc = rcfg("hybrid")
    r1 = SR.render_scene([lower, upper], [2.0, 3.0], [0.0, 0.28], g, rc).tau
    r2 = SR.render_scene([lower, upper], [2.0, 3.0], [0.0, 0.28], g, rc).tau
    assert_array_equal(r1, r2)
    cfg = default_config()
    runs = [SR.synthetic_scene_block([lower, upper], [2.0, 3.0], [-1, 0, 1], -9, -9, 19, 19, cfg, make_rng(),
                                     pad_px=2)[0] for _ in range(2)]
    assert_array_equal(runs[0], runs[1])
    other = SR.synthetic_scene_block([lower, upper], [2.0, 3.0], [-1, 0, 1], -9, -9, 19, 19, cfg, make_rng(SEED + 1),
                                     pad_px=2)[0]
    assert not np.array_equal(runs[0], other), "the camera noise does not follow the generator"


def test_edge_cases():
    lower, upper = pair_stacked()
    g = centred_grid(1.2)
    zp = [-0.28, 0.0]
    # a tube with mu = 0 changes nothing, exactly -- also when its depth range would widen the lattice and it is
    # the thinnest tube of a synthetic block (the fine factor)
    ghost = G.Tube((0.1, 0.05, -0.9), 0.1, math.radians(30.0), 0.7, 1.0, 0.6, "axial")
    for absorption in MODES:
        rc = rcfg(absorption)
        base = SR.render_scene([lower, upper], [2.0, 3.0], zp, g, rc)
        more = SR.render_scene([lower, ghost, upper], [2.0, 0.0, 3.0], zp, g, rc)
        assert_array_equal(more.tau, base.tau)
        assert_array_equal(more.zeta, base.zeta)
        none = SR.render_scene([lower, upper], [0.0, 0.0], zp, g, rc)
        assert np.all(none.tau == 1.0) and none.zeta.size == 0 and none.n_slabs_used == 0
        off = G.Tube((30.0, 30.0, 0.0), 0.2, 0.1, 0.3, 1.0, 1.0, "axial")
        assert np.all(SR.render_scene([off], [3.0], zp, g, rc).tau == 1.0)
    cfg = default_config()
    b1 = SR.synthetic_scene_block([lower, upper], [2.0, 3.0], [0, 1], -9, -9, 19, 19, cfg, make_rng(), pad_px=2)[0]
    b2 = SR.synthetic_scene_block([lower, ghost, upper], [2.0, 0.0, 3.0], [0, 1], -9, -9, 19, 19, cfg, make_rng(),
                                  pad_px=2)[0]
    assert_array_equal(b1, b2)
    rc = rcfg("partition_vertical")
    bad = [
        ("unequal lengths", lambda: SR.render_scene([lower, upper], [1.0], zp, g, rc)),
        ("negative mu", lambda: SR.render_scene([lower, upper], [1.0, -0.1], zp, g, rc)),
        ("nan mu", lambda: SR.render_scene([lower, upper], [1.0, float("nan")], zp, g, rc)),
        ("empty scene", lambda: SR.render_scene([], [], zp, g, rc)),
        ("empty block", lambda: SR.synthetic_scene_block([], [], [0], 0, 0, 5, 5, cfg, make_rng(), 1)),
        ("unequal block", lambda: SR.synthetic_scene_block([lower], [1.0, 2.0], [0], 0, 0, 5, 5, cfg, make_rng(), 1)),
        ("ray world", lambda: SR.render_scene([lower], [1.0], zp, g, rcfg("ray_world"))),
        ("light 0", lambda: SR.render_scene([lower], [1.0], zp, g, rc, light_direction=0)),
        ("coarse grid", lambda: SR.render_scene([lower], [1.0], zp, centred_grid(1.0, PX), rc)),
    ]
    for name, call in bad:
        try:
            call()
        except ValueError:
            continue
        raise AssertionError("%s: no ValueError" % name)


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
