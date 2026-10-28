"""Smoke test for the hybrid absorption -- Block 21 in specs/SPEC.md (D-084,
D-092; the renderers document of 2026-10-09, s.3.8-3.9, Eqs. 13-14).

Checks
    test_known_answer   vertical illumination: Delta A_j = a_j exp(-mu (zbar - z_lo))
                        (light +1), a_j exp(-mu (z_hi - zbar)) (-1), i.e. the
                        partition's Delta A_j times (a_j/2)/sinh(a_j/2); faint
                        limit: h = <1/s_z> = 1.4470 at mu = 0 and Delta A_j /
                        (1.447 a_j) in [exp(-mu L_max), 1] at mu = 1e-3, so the
                        rendered dip is the linear renderer's at 1.447 mu; the
                        prototype's centre dips (checks/hybrid_absorption_check.out,
                        column H) from ray_history; the centre column of a dark
                        flat tube books more than 1, equal to the column
                        integral of mu h within the midpoint rule's error
    test_reference      back_lengths against Block 2's line_interval (interior
                        points to 1e-12, all points to 1e-9, and over several
                        chunks) and against a bisection of Block 2's
                        inside_tube; ray_history against the prototype's
                        two-dimensional closed form (A_density)
    test_convergence    the node plane's centre dip when dzeta halves and when
                        the direction counts double
    test_invariants     per-direction conservation: the booking over a flat
                        tube's cross-section per unit length equals the ray
                        world's total shadow (closed form); a rendered plane's
                        dip volume equals the ray world's; light reversed =
                        depth mirrored; fft (no split) = direct; tau <= 1
    test_contract       render_transmittance and synthetic_block with "hybrid";
                        the configuration and its hashes
    test_determinism    two renders bit-identical; synthetic_block under one seed
    test_edge_cases     mu = 0; a tube off the grid; missing or invalid
                        directions, weights, light direction; a point on the
                        lower skin; an uncapped tube along a direction parallel
                        to its axis

Run
    cd "Passive Features/Diameter Re-measurement"
    python tests/smoke/test_smoke_hybrid.py          # about 1 min on one core

Keep this file pure ASCII (hpc-python-compat).
"""
from __future__ import annotations

import contextlib
import dataclasses
import importlib.metadata
import importlib.util
import io
import math
import platform
import sys
import time
import traceback
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
SRC = WS / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from allen_diameter.config import default_config  # noqa: E402
from allen_diameter.model import geometry as G  # noqa: E402
from allen_diameter.model import hybrid as H  # noqa: E402
from allen_diameter.model import ray_world as RW  # noqa: E402
from allen_diameter.model import render as R  # noqa: E402

SEED = 20261027
REPORT_PACKAGES = ("numpy", "scipy")
PX = 0.1144
MEAN_INV_COS = (2.0 / (1.4 / 1.515) ** 2) * (1.0 - math.sqrt(1.0 - (1.4 / 1.515) ** 2))   # (S10): 1.4470
PROTOTYPE_H = {0.05: 0.057, 0.5: 0.450, 1.5: 0.845, 3.0: 0.961, 10.0: 0.811, 50.0: 0.732}  # hybrid_absorption_check.out


def rcfg(**changes):
    base = dict(absorption="hybrid")
    base.update(changes)
    return dataclasses.replace(default_config().renderer, **base)


def centred_grid(half_um, h, cx=0.0, cy=0.0):
    m = int(round(half_um / h))
    n = 2 * m + 1
    return R.FineGrid(cx - m * h, cy - m * h, h, n, n)


def load_optics_check():
    spec = importlib.util.spec_from_file_location("optics_points_check", str(WS / "checks" / "optics_points_check.py"))
    mod = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    return mod


def prototype_history(v, z, r, mu, S, W):
    """h of an infinite flat tube along x from the prototype's closed form
    (checks/hybrid_absorption_check.py, A_density / mu; renderers document
    s.3.8): l_back = l2D / p, l2D the in-plane distance back to the circle."""
    sv, sz = S[:, 1], S[:, 2]
    p = np.sqrt(sv ** 2 + sz ** 2)
    ev, ez = -sv / p, -sz / p
    qe = v[:, None] * ev[None, :] + z[:, None] * ez[None, :]
    c = v[:, None] ** 2 + z[:, None] ** 2 - r ** 2
    ell = (-qe + np.sqrt(np.clip(qe ** 2 - c, 0.0, None))) / p[None, :]
    return (np.exp(-mu * ell) / sz[None, :]) @ W


def line_interval_back(P, S, tube, light):
    """The reference: l_back from Block 2's line_interval."""
    t1, t2, hit = G.line_interval(P[:, None, :], S[None, :, :], tube)
    with np.errstate(invalid="ignore"):
        ell = np.minimum(t2, 0.0) - t1 if light == +1 else t2 - np.maximum(t1, 0.0)
    return np.where(hit, np.clip(ell, 0.0, None), 0.0)


def bisection_back(p, s, tube, light, step=0.01):
    """l_back at an interior point by marching back along the ray with Block 2's
    inside_tube (closed membership) and bisecting the exit."""
    back = -np.asarray(s) if light == +1 else np.asarray(s)
    inside = lambda t: bool(G.inside_tube(p[0] + t * back[0], p[1] + t * back[1], p[2] + t * back[2], tube))
    assert inside(0.0)
    lo = 0.0
    while inside(lo + step):
        lo += step
    hi = lo + step
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if inside(mid) else (lo, mid)
    return 0.5 * (lo + hi)


def interior_points(tube, n, rng, box):
    pts = []
    while len(pts) < n:
        p = rng.uniform(-box, box, 3) + np.asarray(tube.c)
        if G.inside_tube(p[0], p[1], p[2], tube):
            pts.append(p)
    return np.array(pts)


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    # vertical illumination: the ray's history is the column's, and the midpoint rule's
    S1, W1 = np.array([[0.0, 0.0, 1.0]]), np.array([1.0])
    tube = G.Tube((0.01, -0.02, 0.03), 0.45, math.radians(35.0), math.radians(110.0), 1.0, 1.2, "axial")
    grid = centred_grid(1.6, PX / 4)
    dz, mu = 0.02, 2.0
    zeta = R.slab_grid(tube, grid, dz)
    X, Y = grid.mesh()
    lo, hi, ins = G.column_interval(X, Y, tube)
    for light in (+1, -1):
        dA = R.absorbed_fractions(tube, mu, grid, zeta, dz, light, "hybrid", directions=(S1, W1))
        part = R.absorbed_fractions(tube, mu, grid, zeta, dz, light, "partition_vertical")
        worst = worst_rel = 0.0
        for j, zj in enumerate(zeta):
            a = G.slab_absorbance(lo, hi, zj, dz, mu)
            zbar = 0.5 * (np.maximum(lo, zj - 0.5 * dz) + np.minimum(hi, zj + 0.5 * dz))
            hist = np.exp(-mu * (zbar - lo)) if light == +1 else np.exp(-mu * (hi - zbar))
            expect = np.where(a > 0, a * hist, 0.0)
            worst = max(worst, float(np.max(np.abs(dA[j] - expect))))
            m = a > 0
            if m.any():
                ratio = dA[j][m] / part[j][m]
                worst_rel = max(worst_rel, float(np.max(np.abs(ratio / ((a[m] / 2) / np.sinh(a[m] / 2)) - 1))))
        assert worst <= 1e-12, ("vertical illumination", light, worst)
        assert worst_rel <= 1e-12, ("vertical vs partition (a/2)/sinh(a/2)", light, worst_rel)
    # faint limit
    S, W = RW.directions(16, 32, 1.4, 1.515)
    q = RW.mean_inverse_cos(S, W)                      # the quadrature's <1/s_z>
    assert abs(q - MEAN_INV_COS) <= 1e-10, ("<1/s_z>: quadrature vs (S10)", q, MEAN_INV_COS)
    pts = interior_points(tube, 50, np.random.default_rng(SEED), 1.0)
    assert np.max(np.abs(H.ray_history(pts, tube, 0.0, S, W) - q)) <= 1e-12, "h(mu = 0) = <1/s_z>"
    mu = 1e-3
    tf = G.Tube((0.0, 0.0, 0.0), 0.3, math.radians(20.0), 0.2, 1.0, 1.0, "axial")
    g = centred_grid(1.2, PX / 4)
    zf = R.slab_grid(tf, g, dz)
    dA = R.absorbed_fractions(tf, mu, g, zf, dz, +1, "hybrid", directions=(S, W))
    lin = R.absorbed_fractions(tf, mu, g, zf, dz, +1, "linear")
    m = lin > 0
    L_max = 2.0 * 1.0 + 2 * 0.3       # a chord can be no longer than the capped tube's diagonal
    ratio = dA[m] / (q * lin[m])
    assert ratio.max() <= 1 + 1e-12 and ratio.min() >= math.exp(-mu * L_max), ("faint booking", ratio.min(), ratio.max())
    t_h = R.render_transmittance(tf, mu, [0.0], g, rcfg()).tau
    t_l = R.render_transmittance(tf, q * mu, [0.0], g, rcfg(absorption="linear")).tau
    rel = (1 - t_h).max() / (1 - t_l).max() - 1
    assert -mu * L_max <= rel <= 1e-12, ("faint limit: hybrid = linear at 1.447 mu", rel)
    # the prototype's centre dips: a flat 1 um tube, the node plane's centre, the cone kernel
    # weighted W* (lines through the centre), 16 x 32 directions (its column H, 3 decimals)
    opc = load_optics_check()
    r = 0.5
    flat = G.Tube((0.0, 0.0, 0.0), r, 0.0, 0.0, 1.0, None)
    Wst = W / S[:, 2]
    Wst = Wst / Wst.sum()
    t1, t2, good = opc.chord_interval(np.zeros(3), S, r, 0.0, 6.0)
    k = np.flatnonzero(good)
    x, wx = np.polynomial.legendre.leggauss(200)       # 200 nodes reproduce the prototype's 600 to its print
    tt = 0.5 * (t1[k] + t2[k])[:, None] + 0.5 * (t2[k] - t1[k])[:, None] * x[None, :]
    P = (tt[:, :, None] * S[k][:, None, :]).reshape(-1, 3)
    got = {}
    for mud, want in PROTOTYPE_H.items():
        mu = mud / (2 * r)
        hh = H.ray_history(P, flat, mu, S, W).reshape(tt.shape)
        got[mud] = float(np.sum(Wst[k] * (wx[None, :] * mu * hh).sum(axis=1) * S[k, 2] * 0.5 * (t2[k] - t1[k])))
        assert abs(got[mud] - want) <= 5e-4, ("prototype centre dip", mud, got[mud], want)
    # a dark flat tube's centre column books more than its own light: sum_j Delta A_j against
    # the column integral of mu h (Gauss-Legendre); the midpoint rule's error ~ (mu dz / s_z)^2 / 24
    mu = 3.0
    zc = r * np.polynomial.legendre.leggauss(400)[0]
    wc = r * np.polynomial.legendre.leggauss(400)[1]
    col = float(np.sum(wc * mu * prototype_history(np.zeros_like(zc), zc, r, mu, S, W)))
    g1 = R.FineGrid(0.0, 0.0, PX / 8, 1, 1)
    zeta1 = R.slab_grid(G.Tube((0.0, 0.0, 0.0), r, 0.0, 0.0, 1.0, 6.0, "axial"), g1, dz)
    booked = float(R.absorbed_fractions(G.Tube((0.0, 0.0, 0.0), r, 0.0, 0.0, 1.0, 6.0, "axial"), mu, g1, zeta1, dz,
                                        +1, "hybrid", directions=(S, W)).sum())
    assert col > 1.1 and abs(booked / col - 1) <= 2e-3, ("centre column", booked, col)


def test_reference():
    rng = np.random.default_rng(SEED)
    S, W = RW.directions(16, 32, 1.4, 1.515)
    worst_in = worst_all = 0.0
    for cut in ("axial", "vertical"):
        for phi in (0.0, 30.0, 75.0):
            for k in (1.0, 0.7):
                tube = G.Tube((0.1, -0.2, 0.05), 0.45, math.radians(phi), 0.6, k, 1.5, cut)
                P = rng.uniform(-2.0, 2.0, (1500, 3))
                ins = G.inside_tube(P[:, 0], P[:, 1], P[:, 2], tube)
                for light in (+1, -1):
                    d = np.abs(H.back_lengths(P, tube, S, light) - line_interval_back(P, tube=tube, S=S, light=light))
                    worst_all = max(worst_all, float(d.max()))
                    if ins.any():
                        worst_in = max(worst_in, float(d[ins].max()))
    assert worst_in <= 1e-12 and worst_all <= 1e-9, ("back_lengths vs line_interval", worst_in, worst_all)
    # several chunks: 5000 points x 512 directions span three chunks of 2^20 point-directions
    tube = G.Tube((0.0, 0.1, -0.1), 0.5, math.radians(20.0), 1.3, 1.0, 2.0, "axial")
    P = interior_points(tube, 5000, rng, 2.2)
    for light in (+1, -1):
        d = np.abs(H.back_lengths(P, tube, S, light) - line_interval_back(P, S, tube, light))
        assert d.max() <= 1e-12, ("back_lengths over several chunks", light, d.max())
    # bisection of the membership test, independent of line_interval's algebra
    tube = G.Tube((0.1, -0.2, 0.05), 0.4, math.radians(50.0), 0.9, 0.8, 1.0, "axial")
    pts = interior_points(tube, 12, rng, 1.0)
    dirs = S[rng.choice(S.shape[0], 6, replace=False)]
    for light in (+1, -1):
        ell = H.back_lengths(pts, tube, dirs, light)
        for i in range(pts.shape[0]):
            for n in range(dirs.shape[0]):
                ref = bisection_back(pts[i], dirs[n], tube, light)
                assert abs(ell[i, n] - ref) <= 1e-9, ("bisection", light, ell[i, n], ref)
    # the prototype's two-dimensional closed form, a flat tube along x away from its caps
    r, mu = 0.5, 3.0
    flat = G.Tube((0.0, 0.0, 0.0), r, 0.0, 0.0, 1.0, 6.0, "axial")
    v, z = rng.uniform(-r, r, 4000), rng.uniform(-r, r, 4000)
    m = v ** 2 + z ** 2 < r ** 2
    v, z = v[m], z[m]
    P = np.column_stack([rng.uniform(-1.0, 1.0, v.size), v, z])
    err = np.max(np.abs(H.ray_history(P, flat, mu, S, W) - prototype_history(v, z, r, mu, S, W)))
    assert err <= 1e-12, ("ray_history vs the prototype's closed form", err)


def test_convergence():
    tube = G.Tube((0.0, 0.0, 0.0), 0.3, math.radians(10.0), 0.3, 1.0, 1.0, "axial")
    g = centred_grid(1.2, PX / 4)
    c = g.nx // 2

    def dip(**kw):
        return 1.0 - R.render_transmittance(tube, 2.0, [0.0], g, rcfg(**kw)).tau[0, c, c]

    base = dip()
    d_dz = abs(dip(dzeta_um=0.01) - base)
    d_dir = abs(dip(ray_n_rho=32, ray_n_psi=64) - base)
    assert d_dz <= 2e-4 and d_dir <= 2e-4, ("centre dip moves", base, d_dz, d_dir)   # 3.5e-5, 4.5e-7 (2026-10-27)


def test_invariants():
    # per-direction conservation, density level: the booking over a flat tube's cross-section per
    # unit length equals the ray world's total shadow <(p/s_z) int (1 - exp(-2 mu sqrt(r^2 - w^2)/p)) dw>
    S, W = RW.directions(16, 32, 1.4, 1.515)
    r = 0.5
    flat = G.Tube((0.0, 0.0, 0.0), r, 0.0, 0.0, 1.0, None)
    p = np.sqrt(S[:, 1] ** 2 + S[:, 2] ** 2)
    xa, wa = np.polynomial.legendre.leggauss(160)
    alpha, w_alpha = 0.5 * math.pi * xa, 0.5 * math.pi * wa          # w = r sin(alpha): smooth integrand
    xt, wt = np.polynomial.legendre.leggauss(160)
    for mud, tol in ((3.0, 1e-8), (50.0, 1e-6)):        # 160 x 160 nodes: 6e-10 and 1.7e-7 (2026-10-27)
        mu = mud / (2 * r)
        chord = 2 * r * np.cos(alpha)
        shadow = float(np.sum(W * (p / S[:, 2]) * ((1 - np.exp(-mu * chord[None, :] / p[:, None])) @ (w_alpha * r * np.cos(alpha)))))
        V = r * np.sin(alpha)[:, None] * np.ones_like(xt)[None, :]
        half = r * np.cos(alpha)[:, None]
        Z = half * xt[None, :]
        jac = (w_alpha * r * np.cos(alpha))[:, None] * (half * wt[None, :])
        hh = H.ray_history(np.column_stack([np.zeros(V.size), V.ravel(), Z.ravel()]), flat, mu, S, W).reshape(V.shape)
        booked = float(np.sum(jac * mu * hh))
        assert abs(booked / shadow - 1) <= tol, ("booking = shadow per unit length", mud, booked, shadow)
        assert abs(shadow - {3.0: 1.1137, 50.0: 1.2383}[mud]) <= 5e-5, ("the prototype's ray-world shadow", mud, shadow)
    # renderer level: a plane's dip volume equals the ray world's (no post blur) on a grid that holds both
    S8, W8 = RW.directions(8, 16, 1.4, 1.515)
    tube = G.Tube((0.0, 0.0, 0.0), 0.3, math.radians(20.0), 0.4, 1.0, 0.8, "axial")
    g = centred_grid(3.0, PX / 4)
    rc8 = rcfg(ray_n_rho=8, ray_n_psi=16, ray_post_sigma_um=0.0)
    hyb = R.render_transmittance(tube, 1.5, [0.0], g, rc8).tau[0]
    ray = R.render_transmittance(tube, 1.5, [0.0], g, dataclasses.replace(rc8, absorption="ray_world")).tau[0]
    vol_h, vol_r = (1 - hyb).sum(), (1 - ray).sum()
    assert abs(vol_h / vol_r - 1) <= 1e-3, ("dip volume vs the ray world", vol_h / vol_r)   # -3.2e-4: the grid's sampling
    assert hyb.max() <= 1.0 + 1e-12
    # light reversed = depth mirrored about c_z with theta -> theta + pi (even n_psi; a slab lattice
    # that spans the depth range exactly mirrors onto itself)
    cz = 0.11
    g2 = centred_grid(1.5, PX / 4)
    tp = G.Tube((0.0, 0.0, cz), 0.3, math.radians(35.0), 0.7, 1.0, 1.0, "axial")
    tm = G.Tube((0.0, 0.0, cz), 0.3, math.radians(35.0), 0.7 + math.pi, 1.0, 1.0, "axial")
    X, Y = g2.mesh()
    lo, hi, ins = G.column_interval(X, Y, tp)
    dz = (hi[ins].max() - lo[ins].min()) / 40.0
    zk = np.array([cz - 0.3, cz, cz + 0.45])
    rcm = rcfg(dzeta_um=dz, ray_n_rho=8, ray_n_psi=16)
    up = R.render_transmittance(tp, 2.0, zk, g2, rcm, +1).tau
    down = R.render_transmittance(tm, 2.0, 2 * cz - zk, g2, rcm, -1).tau
    assert np.max(np.abs(up - down)) <= 1e-12, ("light/mirror", np.max(np.abs(up - down)))
    assert np.max(np.abs(up - R.render_transmittance(tp, 2.0, zk, g2, rcm, -1).tau)) > 1e-3, \
        "the light direction must matter for a dark tube"
    # one cache serves both backends: fft without the split = direct
    ex = rcfg(ray_n_rho=8, ray_n_psi=16, fft_split_sigma_um=1e9, fft_wrap_sigmas=10.0)
    tube3 = G.Tube((0.0, 0.0, 0.0), 0.4, math.radians(30.0), 0.5, 1.0, 1.2, "axial")
    g3 = centred_grid(1.6, PX / 4)
    zk3 = np.array([-0.28, 0.0, 0.28])
    a = R.render_transmittance(tube3, 2.0, zk3, g3, ex, +1, "fft").tau
    b = R.render_transmittance(tube3, 2.0, zk3, g3, ex, +1, "direct").tau
    assert np.max(np.abs(a - b)) <= 1e-10, ("fft vs direct", np.max(np.abs(a - b)))


def test_contract():
    cfg = default_config()
    # the default's hashes (Block 21 adds no field); since D-096 (2026-10-28) the flags of S are in the estimator
    # signature: 0508e7ff12f48728 / 0ea5a6d2ec069c68 before it
    assert (cfg.signature_hash("estimator"), cfg.signature_hash("simulation")) == ("59678d4e2aedf2b6", "52aa73e479eb77cf")
    hcfg = dataclasses.replace(cfg, renderer=dataclasses.replace(cfg.renderer, absorption="hybrid"))
    hcfg.validate()
    assert hcfg.signature_hash("estimator") == cfg.signature_hash("estimator")
    assert hcfg.signature_hash("simulation") != cfg.signature_hash("simulation")
    tube = G.Tube((0.3 * PX, -0.2 * PX, 0.05), 0.15, math.radians(10.0), 1.1, 1.0, 10.0, "axial")
    g = centred_grid(1.0, PX / 4)
    res = R.render_transmittance(tube, 1.0, [-0.28, 0.0], g, hcfg.renderer)
    assert res.tau.shape == (2, g.ny, g.nx) and res.backend == "fft" and res.n_slabs_used > 0
    assert res.tau.max() <= 1.0 + 1e-12 and res.tau.min() < 0.99
    red = R.render_transmittance(tube, 1.0, [-0.28, 0.0], g, hcfg.renderer, reduce=lambda p: p[::2, ::2])
    assert np.array_equal(red.tau, res.tau[:, ::2, ::2])
    ks = np.arange(-1, 2)
    block, ks_out, valid, frame = R.synthetic_block(tube, 1.0, ks, -12, -10, 25, 21, hcfg,
                                                    np.random.default_rng(SEED), pad_px=3)
    assert block.shape == (3, 21, 25) and block.dtype == np.uint8 and np.array_equal(ks_out, ks) and valid.all()
    assert (frame.left, frame.top, frame.downsample, frame.res0_um_px) == (-12, -10, 0, cfg.acquisition.res0_um)


def test_determinism():
    rc = rcfg(ray_n_rho=8, ray_n_psi=16)
    tube = G.Tube((0.0, 0.0, 0.0), 0.3, math.radians(15.0), 0.5, 1.0, 1.0, "axial")
    g = centred_grid(1.0, PX / 4)
    a = R.render_transmittance(tube, 1.5, [0.0, 0.28], g, rc).tau
    b = R.render_transmittance(tube, 1.5, [0.0, 0.28], g, rc).tau
    assert np.array_equal(a, b)
    cfg = default_config()
    hcfg = dataclasses.replace(cfg, renderer=dataclasses.replace(cfg.renderer, absorption="hybrid", ray_n_rho=8,
                                                                 ray_n_psi=16))
    runs = [R.synthetic_block(tube, 1.0, [0, 1], -9, -9, 19, 19, hcfg, np.random.default_rng(SEED), pad_px=2)[0]
            for _ in range(2)]
    assert np.array_equal(runs[0], runs[1])


def test_edge_cases():
    S, W = RW.directions(8, 16, 1.4, 1.515)
    rc = rcfg(ray_n_rho=8, ray_n_psi=16)
    g = centred_grid(1.0, PX / 4)
    tube = G.Tube((0.0, 0.0, 0.0), 0.3, 0.2, 0.1, 1.0, 1.0, "axial")
    assert np.array_equal(R.render_transmittance(tube, 0.0, [0.0], g, rc).tau, np.ones((1, g.ny, g.nx)))
    off = G.Tube((30.0, 30.0, 0.0), 0.3, 0.2, 0.1, 1.0, 1.0, "axial")
    res = R.render_transmittance(off, 1.0, [0.0], g, rc)
    assert np.array_equal(res.tau, np.ones((1, g.ny, g.nx))) and res.zeta.size == 0
    zeta = R.slab_grid(tube, g, 0.02)
    bad = [lambda: R.absorbed_fractions(tube, 1.0, g, zeta, 0.02, +1, "hybrid"),
           lambda: H.ray_history(np.zeros((1, 3)), tube, 1.0, np.array([[0.0, 0.6, -0.8]]), np.array([1.0])),
           lambda: H.ray_history(np.zeros((1, 3)), tube, 1.0, np.array([[0.0, 0.0, 2.0]]), np.array([1.0])),
           lambda: H.ray_history(np.zeros((1, 3)), tube, 1.0, S, 2 * W),
           lambda: H.ray_history(np.zeros((1, 3)), tube, -1.0, S, W),
           lambda: H.back_lengths(np.zeros((1, 3)), tube, S, 0),
           lambda: H.back_lengths(np.zeros((2, 2)), tube, S),
           lambda: R.render_transmittance(tube, 1.0, [0.0], g, rc, light_direction=0)]
    for k, call in enumerate(bad):
        try:
            call()
        except ValueError:
            continue
        raise AssertionError("invalid input %d accepted" % k)
    # a point on the lower skin (light +1): every ray enters there
    flat = G.Tube((0.0, 0.0, 0.0), 0.5, 0.0, 0.0, 1.0, 3.0, "axial")
    ell = H.back_lengths(np.array([[0.2, 0.0, -0.5]]), flat, S, +1)
    assert np.max(ell) <= 1e-12, ("lower skin", np.max(ell))
    # an uncapped tube and a direction along its axis: infinite path, no NaN
    inf_tube = G.Tube((0.0, 0.0, 0.0), 0.5, math.radians(30.0), 0.5, 1.0, None)
    Sa = np.vstack([inf_tube.t_hat, [0.0, 0.0, 1.0]])
    Wa = np.array([0.5, 0.5])
    ell = H.back_lengths(np.zeros((1, 3)), inf_tube, Sa, +1)
    assert math.isinf(ell[0, 0]) and math.isfinite(ell[0, 1]), ell
    h1 = H.ray_history(np.zeros((1, 3)), inf_tube, 1.0, Sa, Wa)
    assert np.isfinite(h1).all() and abs(h1[0] - 0.5 * math.exp(-ell[0, 1])) <= 1e-15, h1
    h0 = H.ray_history(np.zeros((1, 3)), inf_tube, 0.0, Sa, Wa)
    assert abs(h0[0] - (0.5 / Sa[0, 2] + 0.5)) <= 1e-15, h0
    try:
        list(R._slabs(np.zeros((1, 1)), np.ones((1, 1)), np.ones((1, 1), bool), np.array([0.5]), 0.02, 1.0, +1,
                      "hybrid"))
    except ValueError:
        pass
    else:
        raise AssertionError("_slabs must refuse the hybrid without its inputs")


# ---------------------------------------------------------------- runner ---

def _environment():
    parts = ["python %s" % platform.python_version()]
    for package in REPORT_PACKAGES:
        try:
            parts.append("%s %s" % (package, importlib.metadata.version(package)))
        except importlib.metadata.PackageNotFoundError:
            parts.append("%s (not installed)" % package)
    parts.append(platform.platform())
    parts.append("seed %d" % SEED)
    return " | ".join(parts)


def main():
    checks = [(name, obj) for name, obj in globals().items()
              if name.startswith("test_") and callable(obj)]
    print("== %s" % Path(__file__).name)
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
        except Exception:
            status, detail = "ERROR", traceback.format_exc()
        results.append((name, status, time.perf_counter() - start, detail))
    width = max([len(n) for n, _, _, _ in results] + [4])
    for name, status, seconds, detail in results:
        lines = detail.strip().splitlines()
        headline = lines[0] if lines else ""
        print(("%-5s  %-" + str(width) + "s  %8.3fs  %s") % (status, name, seconds, headline))
    for name, status, _s, detail in results:
        if status in ("FAIL", "ERROR"):
            print("\n---- %s: %s\n%s" % (status, name, detail.strip()))
    counts = {s: sum(1 for r in results if r[1] == s) for s in ("PASS", "FAIL", "ERROR", "TODO", "SKIP")}
    print("\n-- " + ", ".join("%d %s" % (n, s.lower()) for s, n in counts.items()))
    return 1 if (counts["FAIL"] + counts["ERROR"] + counts["TODO"]) else 0


if __name__ == "__main__":
    sys.exit(main())
