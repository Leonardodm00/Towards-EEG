"""Spec-derived tests of Block 21 (the hybrid absorption renderer) --
integration tester, run D1, 2026-11-13. Expected values: section 2.3's (H)
written out with the chord of a round capped tube from its quadratic
(spec_helpers_D1), its closed forms (vertical illumination, the faint limit,
the per-direction shadow by scipy.integrate.quad), and the symmetries the
spec states; the code under test only produces the outputs that are checked.

Run (from the workstream directory):
    python -m pytest tests/spec/test_spec_block21_hybrid.py -q
About two minutes on one free core (the rendering checks dominate).
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import spec_helpers_D1 as H

from allen_diameter.config import default_config
from allen_diameter.model import hybrid as HY
from allen_diameter.model import ray_world as RW
from allen_diameter.model import render as R
from allen_diameter.model.geometry import Tube

CFG = default_config()
RC = CFG.renderer
RCH = dataclasses.replace(RC, absorption="hybrid")
S16, W16 = RW.directions(RC.ray_n_rho, RC.ray_n_psi, RC.ray_na, RC.ray_n_oil)
S8, W8 = RW.directions(8, 16, RC.ray_na, RC.ray_n_oil)
UP = np.array([[0.0, 0.0, 1.0]])
ONE = np.array([1.0])


def slab_parts(z_lo, z_hi, zeta, dz):
    lo = np.maximum(z_lo, zeta - dz / 2)
    hi = np.minimum(z_hi, zeta + dz / 2)
    lam = np.maximum(0.0, hi - lo)
    return lo, hi, lam, 0.5 * (lo + hi)


# -------------------------------------------------- O21.1 vertical light --

@pytest.mark.parametrize("light", [+1, -1])
def test_vertical_illumination_known_answer(light):
    phi, theta, r, mu = math.radians(25.0), 0.4, 0.5, 2.3
    c = np.array([0.1, -0.05, 0.2])
    tube = Tube(tuple(c), r, phi, theta, 1.0, 10.0, "axial")
    rng = np.random.default_rng(5)
    X = rng.uniform(-0.6, 0.8, 300)
    Y = rng.uniform(-0.6, 0.5, 300)
    z_lo, z_hi, inside = H.ref_column(X, Y, c, r, phi, theta, 10.0)
    assert inside.sum() > 50
    dz = 0.02
    zetas = np.arange(-1.2, 1.6, dz)
    worst, worst_rel = 0.0, 0.0
    for zeta in zetas:
        got = HY.slab_absorbed(X, Y, z_lo, z_hi, zeta, dz, mu, tube, UP, ONE, light)
        lo, hi, lam, zbar = slab_parts(z_lo, z_hi, zeta, dz)
        a = mu * lam
        if light > 0:
            exp = a * np.exp(-mu * (zbar - z_lo))
            Tpre = np.exp(-mu * np.maximum(0.0, np.minimum(z_hi, zeta - dz / 2) - z_lo))
        else:
            exp = a * np.exp(-mu * (z_hi - zbar))
            Tpre = np.exp(-mu * np.maximum(0.0, z_hi - np.maximum(z_lo, zeta + dz / 2)))
        exp = np.where(lam > 0, exp, 0.0)
        worst = max(worst, float(np.max(np.abs(got - exp))))
        part = Tpre * (-np.expm1(-a))     # expm1: no cancellation at small a
        m = a > 1e-12
        if m.any():
            ratio = (a[m] / 2) / np.sinh(a[m] / 2)
            worst_rel = max(worst_rel, float(np.max(np.abs(got[m] - part[m] * ratio) / part[m])))
    assert worst < 1e-12           # closed form, roundoff only
    assert worst_rel < 1e-12       # = the partition times (a/2)/sinh(a/2)


# ----------------------------------------------------- O21.2 faint limit --

def test_faint_limit_h_is_mean_inverse_cos():
    tube = Tube((0.0, 0.0, 0.0), 0.5, math.radians(30), 0.2, 1.0, 10.0, "axial")
    P = np.random.default_rng(1).uniform(-0.3, 0.3, size=(200, 3))
    h0 = HY.ray_history(P, tube, 0.0, S16, W16, +1)
    exact = float(np.sum(W16 / S16[:, 2]))
    assert_allclose(h0, exact, rtol=0, atol=1e-12)
    assert abs(exact - H.ref_mean_inverse_cos(RC.ray_na, RC.ray_n_oil)) < 1e-10


def test_faint_booking_bounds():
    """At mu = 1e-3/um: DeltaA_hyb/(<1/s_z> a) in [exp(-mu L_max), 1]."""
    phi, theta, r, mu = math.radians(15.0), 0.1, 0.5, 1e-3
    c = np.zeros(3)
    U = 10.0
    tube = Tube(tuple(c), r, phi, theta, 1.0, U, "axial")
    X, Y = np.meshgrid(np.linspace(-0.6, 0.6, 25), np.linspace(-0.6, 0.6, 25))
    X, Y = X.ravel(), Y.ravel()
    z_lo, z_hi, inside = H.ref_column(X, Y, c, r, phi, theta, U)
    k = float(np.sum(W16 / S16[:, 2]))
    L_max = 2 * U + 2 * r    # longer than any chord of this tube
    lo_b = math.exp(-mu * L_max)
    for zeta in np.arange(-0.8, 0.8, 0.02):
        got = HY.slab_absorbed(X, Y, z_lo, z_hi, zeta, 0.02, mu, tube, S16, W16, +1)
        _lo, _hi, lam, _zb = slab_parts(z_lo, z_hi, zeta, 0.02)
        a = mu * lam
        m = a > 0
        if m.any():
            q = got[m] / (k * a[m])
            assert np.all(q <= 1.0 + 1e-12) and np.all(q >= lo_b - 1e-12)
        assert np.all(got[~m] == 0.0)


# ------------------------------------------- O21.3 back lengths, history --

@pytest.mark.parametrize("phi_deg", [0.0, 30.0, 75.0])
@pytest.mark.parametrize("light", [+1, -1])
def test_back_lengths_equal_the_chord_reference(phi_deg, light):
    phi, theta, r, U = math.radians(phi_deg), 0.9, 0.5, 1.5
    c = np.array([0.2, 0.1, -0.3])
    tube = Tube(tuple(c), r, phi, theta, 1.0, U, "axial")
    rng = np.random.default_rng(int(phi_deg) * 7 + light + 2)
    a = H.axis_dir(phi, theta)
    # interior points: along the axis within the caps, inside the radius
    n = 5000
    s0 = rng.uniform(-0.9 * U, 0.9 * U, n)
    perp1 = np.cross(a, [0.0, 0.0, 1.0]) if phi_deg < 89 else np.array([1.0, 0, 0])
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(a, perp1)
    rr = r * np.sqrt(rng.uniform(0, 0.95, n))
    ang = rng.uniform(0, 2 * np.pi, n)
    P = c + s0[:, None] * a + (rr * np.cos(ang))[:, None] * perp1 + (rr * np.sin(ang))[:, None] * perp2
    got = HY.back_lengths(P, tube, S16, light)       # 5000 x 512 > 2^20: several chunks
    exp = H.ref_back_length(P, S16, c, r, phi, theta, U, light)
    assert got.shape == (n, S16.shape[0])
    assert np.max(np.abs(got - exp)) < 1e-12        # the spec's interior tolerance
    # all points, inside and outside
    Q = c + rng.uniform(-2.5, 2.5, size=(800, 3))
    got = HY.back_lengths(Q, tube, S8, light)
    exp = H.ref_back_length(Q, S8, c, r, phi, theta, U, light)
    assert np.max(np.abs(got - exp)) < 1e-9          # near-tangent chords: sqrt of roundoff


def test_ray_history_equals_reference_and_2d_closed_form():
    r, mu = 0.5, 2.0
    tube = Tube((0.0, 0.0, 0.0), r, 0.0, 0.0, 1.0, 50.0, "axial")   # flat, along x
    rng = np.random.default_rng(9)
    rho = r * np.sqrt(rng.uniform(0, 0.97, 300))
    ang = rng.uniform(0, 2 * np.pi, 300)
    P = np.column_stack([rng.uniform(-1, 1, 300), rho * np.cos(ang), rho * np.sin(ang)])
    for light in (+1, -1):
        got = HY.ray_history(P, tube, mu, S16, W16, light)
        assert_allclose(got, H.ref_h(P, S16, W16, mu, np.zeros(3), r, 0.0, 0.0, 50.0, light),
                        rtol=1e-12, atol=0)
    # the 2-D closed form (renderers document s.3.8): l_back = l2D_back / p
    sv, sz = S16[:, 1], S16[:, 2]
    p = np.hypot(sv, sz)
    ev, ew = sv / p, sz / p
    v, w = P[:, 1:2], P[:, 2:3]
    proj = v * ev + w * ew
    l2d = proj + np.sqrt(proj ** 2 - (v ** 2 + w ** 2 - r * r))    # backwards to the circle (light +1)
    h2d = np.exp(-mu * l2d / p) @ (W16 / sz)
    assert_allclose(HY.ray_history(P, tube, mu, S16, W16, +1), h2d, rtol=1e-12, atol=0)


# ----------------------------------- O21.4 per-direction conservation -----

@pytest.mark.parametrize("mud", [3.0, 50.0])
def test_per_direction_booking_equals_the_shadow(mud):
    """Block 21 (c): for one direction the total booked over a flat tube's
    cross-section per unit length, int int mu h dA, equals that direction's
    total shadow (p/s_z) int (1 - exp(-2 mu sqrt(r^2 - b^2)/p)) db."""
    r = 0.5
    mu = mud / (2 * r)
    tube = Tube((0.0, 0.0, 0.0), r, 0.0, 0.0, 1.0, None, "axial")   # infinite flat tube along x
    s = np.array([0.35, 0.55, 0.0])
    s[2] = math.sqrt(1.0 - s[0] ** 2 - s[1] ** 2)
    xa, wa = np.polynomial.legendre.leggauss(160)
    xz, wz = np.polynomial.legendre.leggauss(160)
    alpha = xa * math.pi / 2
    wal = wa * math.pi / 2
    y = r * np.sin(alpha)
    half = r * np.cos(alpha)
    Y = np.repeat(y, 160)
    Z = (half[:, None] * xz[None, :]).ravel()
    Wt = (wal[:, None] * (r * np.cos(alpha))[:, None] * half[:, None] * wz[None, :]).ravel()
    P = np.column_stack([np.zeros_like(Y), Y, Z])
    hval = HY.ray_history(P, tube, mu, s[None, :], ONE, +1)
    booked = float(np.sum(mu * hval * Wt))
    shadow = H.ref_shadow_per_length(mu, r, s)
    tol = 1e-8 if mud <= 3 else 1e-6        # the spec's own tolerances (160 x 160 nodes)
    assert abs(booked - shadow) <= tol * shadow, (booked, shadow)


# ----------------------------------------------- O21.5 renders: symmetry --

def _grid(n=70, h=0.0286):
    return R.FineGrid(-n * h / 2, -n * h / 2, h, n, n)


def _exact_lattice_rc(grid, c, r, phi, theta, U, rc):
    """dzeta_um chosen so that the slabs tile [min z_lo, max z_hi] of the
    grid's columns exactly (the spec's condition for the mirror identity);
    the range from the independent column reference."""
    X, Y = np.meshgrid(grid.x0 + grid.h * np.arange(grid.nx), grid.y0 + grid.h * np.arange(grid.ny))
    z_lo, z_hi, ins = H.ref_column(X, Y, np.asarray(c, float), r, phi, theta, U)
    span = float(z_hi[ins].max() - z_lo[ins].min())
    return dataclasses.replace(rc, dzeta_um=span / round(span / 0.02))


def test_light_reversed_equals_depth_mirrored():
    phi, theta, mu = math.radians(20.0), 0.4, 2.0
    zs = np.array([-2, -1, 0, 1, 2]) * 0.28
    t1 = Tube((0.0, 0.0, 0.0), 0.3, phi, theta, 1.0, 2.0, "axial")
    t2 = Tube((0.0, 0.0, 0.0), 0.3, phi, theta + math.pi, 1.0, 2.0, "axial")
    rc = _exact_lattice_rc(_grid(), (0.0, 0.0, 0.0), 0.3, phi, theta, 2.0, RCH)
    a = R.render_transmittance(t1, mu, zs, _grid(), rc, light_direction=+1).tau
    b = R.render_transmittance(t2, mu, -zs, _grid(), rc, light_direction=-1).tau
    assert np.max(np.abs(a - b)) < 1e-12
    c = R.render_transmittance(t1, mu, zs, _grid(), rc, light_direction=-1).tau
    assert np.max(np.abs(a - c)) > 1e-3     # the light direction matters for a dark tube
    assert np.all(a <= 1.0 + 1e-12)


def test_fft_without_split_equals_direct():
    tube = Tube((0.0, 0.0, 0.0), 0.3, math.radians(10), 0.2, 1.0, 1.5, "axial")
    zs = np.array([-0.56, 0.0, 0.28])
    rc_nosplit = dataclasses.replace(RCH, fft_split_sigma_um=1e9)
    f = R.render_transmittance(tube, 1.5, zs, _grid(60), rc_nosplit, backend="fft").tau
    d = R.render_transmittance(tube, 1.5, zs, _grid(60), rc_nosplit, backend="direct").tau
    assert np.max(np.abs(f - d)) < 1e-10


def test_mu_zero_and_off_grid_give_tau_one():
    tube = Tube((0.0, 0.0, 0.0), 0.3, 0.1, 0.0, 1.0, 1.0, "axial")
    res = R.render_transmittance(tube, 0.0, [0.0, 0.28], _grid(30), RCH)
    assert_array_equal(res.tau, 1.0)
    far = Tube((40.0, 40.0, 0.0), 0.3, 0.1, 0.0, 1.0, 1.0, "axial")
    res = R.render_transmittance(far, 2.0, [0.0], _grid(30), RCH)
    assert_array_equal(res.tau, 1.0)
    assert res.n_slabs_used == 0


def test_determinism_and_contract():
    tube = Tube((0.0, 0.0, 0.0), 0.3, 0.2, 0.3, 1.0, 1.0, "axial")
    a = R.render_transmittance(tube, 1.0, [0.0, 0.28], _grid(40), RCH)
    b = R.render_transmittance(tube, 1.0, [0.0, 0.28], _grid(40), RCH)
    assert_array_equal(a.tau, b.tau)
    assert a.tau.shape == (2, 40, 40) and a.tau.dtype == np.float64


# ------------------------------------------------- O21.6 refusals ---------

def _bad_cases():
    P = np.zeros((3, 3))
    tube = Tube((0.0, 0.0, 0.0), 0.3, 0.1, 0.0, 1.0, 1.0, "axial")
    Sd = np.array([[0.0, 0.6, -0.8]])
    Snu = np.array([[0.0, 0.0, 1.1]])
    return [
        lambda: HY.ray_history(P, tube, 1.0, Sd, ONE, +1),                 # s_z <= 0
        lambda: HY.ray_history(P, tube, 1.0, Snu, ONE, +1),                # not unit
        lambda: HY.ray_history(P, tube, 1.0, S8, W8 * 1.01, +1),           # weights sum != 1
        lambda: HY.ray_history(P, tube, 1.0, UP.repeat(2, 0), np.array([1.5, -0.5]), +1),  # negative
        lambda: HY.ray_history(P, tube, 1.0, UP.repeat(2, 0), np.array([np.nan, 1.0]), +1),  # non-finite
        lambda: HY.ray_history(P, tube, 1.0, S8, W8, 0),                   # light 0
        lambda: HY.ray_history(P, tube, -1.0, S8, W8, +1),                 # negative mu
        lambda: HY.ray_history(np.zeros((3, 2)), tube, 1.0, S8, W8, +1),   # points not (n, 3)
        lambda: HY.back_lengths(np.zeros(3), tube, S8, +1),                # points not (n, 3)
        lambda: R.absorbed_fractions(tube, 1.0, _grid(10), np.array([0.0]), 0.02,
                                     +1, "hybrid", None),                  # no directions
    ]


@pytest.mark.parametrize("i", [k for k in range(10) if k != 8])
def test_refusals(i):
    with pytest.raises(ValueError):
        _bad_cases()[i]()


@pytest.mark.xfail(strict=True, reason="D1-F3: a 1-D point (3,) is taken as one point (1, 3), not refused")
def test_refusal_of_a_1d_point():
    with pytest.raises(ValueError):
        _bad_cases()[8]()


# ---------------------------------------------- O21.7 edge geometry -------

def test_lower_skin_and_uncapped_axis():
    r = 0.4
    tube = Tube((0.0, 0.0, 0.0), r, 0.0, 0.0, 1.0, 5.0, "axial")
    p = np.array([[0.0, 0.0, -r]])                     # the bottom of the skin
    lb = HY.back_lengths(p, tube, S16, +1)
    assert np.max(np.abs(lb)) < 1e-12
    phi = math.radians(40.0)
    up = Tube((0.0, 0.0, 0.0), r, phi, 0.3, 1.0, None, "axial")
    a = H.axis_dir(phi, 0.3)
    S = np.array([a, [0.0, 0.0, 1.0]])
    W = np.array([0.5, 0.5])
    q = np.array([[0.0, 0.0, 0.0]])
    lb = HY.back_lengths(q, up, S, +1)
    assert np.isinf(lb[0, 0]) and np.isfinite(lb[0, 1])
    h = HY.ray_history(q, up, 1.0, S, W, +1)
    assert np.all(np.isfinite(h))
    assert_allclose(h, 0.5 * math.exp(-1.0 * lb[0, 1]) / 1.0, rtol=1e-12)
    h0 = HY.ray_history(q, up, 0.0, S, W, +1)
    assert_allclose(h0, 0.5 / a[2] + 0.5, rtol=1e-14)


# --------------------------------------------------- convergence (spec) ---

def _centre_dip(theta, rc):
    """1 - tau at the grid sample on the axis of the node plane (an odd grid
    centred on the tube's node point), d = 0.6 um, 10 deg, mu = 2/um."""
    tube = Tube((0.0, 0.0, 0.0), 0.3, math.radians(10.0), theta, 1.0, 1.0, "axial")
    h = 0.1144 / 4
    n = 2 * int(round(1.2 / h)) + 1
    g = R.FineGrid(-(n // 2) * h, -(n // 2) * h, h, n, n)
    return 1.0 - float(R.render_transmittance(tube, 2.0, [0.0], g, rc).tau[0, n // 2, n // 2])


def test_centre_dip_converges_in_directions():
    """Block 21: the centre dip moves by less than 2e-4 when both direction
    counts double."""
    base = _centre_dip(0.0, RCH)
    dbl = _centre_dip(0.0, dataclasses.replace(RCH, ray_n_rho=32, ray_n_psi=64))
    assert abs(dbl - base) < 2e-4


@pytest.mark.xfail(strict=True, reason="D1-F2: at heading 0 the centre dip moves 2.4e-4 (> 2e-4) when "
                   "dzeta_um halves from 0.02 um; 3.5e-5 at heading 0.3 (the smoke test's case)")
def test_centre_dip_converges_in_dzeta_heading_0():
    """Block 21: the node plane's centre dip (d = 0.6 um, 10 deg, mu = 2/um)
    moves by less than 2e-4 when dzeta_um halves from 0.02 um (the spec
    names no heading)."""
    base = _centre_dip(0.0, RCH)
    half = _centre_dip(0.0, dataclasses.replace(RCH, dzeta_um=0.01))
    assert abs(half - base) < 2e-4
