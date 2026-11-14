"""Spec-derived tests of Block 35 (the fast line renderer, D-137, D-138, D-140)
-- integration tester, 2026-11-14. Oracles from specs/SPEC.md (section 2.3,
Block 35 and its amendment 1); expected values from closed forms, scipy
quadrature, Block 9's quadrature (outside the scope) or Block 26 (the
reference the spec names), never from line_fast itself.

Run: python3 -m pytest -q tests/spec/test_spec_block35_line_fast.py
Pure ASCII.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import integrate

import nc_helpers  # noqa: F401  (sys.path)
from allen_diameter.config import default_config
from allen_diameter.model import geometry, line_fast, line_render, ray_world

P = 0.1144


def cfg_fast(absorption="hybrid", n_rho=4, n_psi=8, **inf):
    """Default configuration with the inference's directions reduced (Block 35's own test allows 4 x 8)."""
    cfg = default_config()
    i = dataclasses.replace(cfg.inference, absorption=absorption, ray_n_rho=n_rho, ray_n_psi=n_psi, **inf)
    r = dataclasses.replace(cfg.renderer, ray_n_rho=n_rho, ray_n_psi=n_psi)
    return dataclasses.replace(cfg, inference=i, renderer=r)


def flat_line(n_half=60):
    """A tube along x (theta = 0) through the origin, read along y on pixel centres: o = (0, 0)
    (pixel centre of pixel (0, 0), section 2.1), y_hat = (0, 1), e_u = (1, 0), v = n p."""
    v = np.arange(-n_half, n_half + 1) * P
    return np.array([0.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0]), v


# ---------------------------------------------------------- mass oracle ---

def chord_mass(r, mu, q=1.0):
    """int_{-r}^{r} (1 - exp(-mu 2 sqrt(r^2 - b^2) / q)) db (um): the shadow of an infinite tube per unit
    length, for rays whose component normal to the tube's axis has norm q."""
    val, _err = integrate.quad(lambda b: -math.expm1(-mu * 2.0 * math.sqrt(max(r * r - b * b, 0.0)) / q), -r, r,
                               epsabs=1e-13, epsrel=1e-12)
    return val


@pytest.mark.parametrize("absorption", ["linear", "partition_vertical", "hybrid"])
def test_mass_per_unit_length_flat(absorption):
    """Section 2.3: the blur and the pixel mean conserve the absorbed light; for a flat tube along x read
    across on pixel centres (bilinear = the pixel values), p sum_n (1 - tau_k(v_n)) is, in every plane,
    linear: mu pi r^2; partition: int (1 - e^{-mu l(b)}) db; hybrid: sum_W w (q/s_z) int (1 - e^{-mu l(b)/q}) db,
    q = |(s_y, s_z)| (each direction's shadow, section 2.3 'Conserved'; the projection of the
    cross-section along s onto z = const widens by q/s_z).
    Tolerance 3e-3 relative: the fine lattice (8 samples per pixel) samples the chord's square-root edge
    (error ~ (h/r)^1.5 ~ 1e-3 at r 0.3 um), the hybrid's midpoint rule in depth (section 2.3: <= 1e-3) and the
    section's bicubic read."""
    cfg = cfg_fast(absorption)
    r, mu = 0.3, 1.5
    tube = geometry.Tube(c=(0.0, 0.0, 0.0), r=r, phi=0.0, theta=0.0, half_length=12.0)
    o, yh, eu, v = flat_line()
    ks = np.array([-2, 0, 1])
    tau = line_fast.line_profiles(tube, mu, ks, o, yh, eu, v, 0, P, cfg, absorption=absorption)
    got = P * np.sum(1.0 - tau, axis=1)
    if absorption == "linear":
        exp = mu * math.pi * r * r
    elif absorption == "partition_vertical":
        exp = chord_mass(r, mu)
    else:
        S, W = ray_world.directions(4, 8, cfg.renderer.ray_na, cfg.renderer.ray_n_oil)
        exp = 0.0
        for s, w in zip(S, W):
            q = math.hypot(s[1], s[2])
            exp += w * (q / s[2]) * chord_mass(r, mu, q)
    assert_allclose(got, exp, rtol=3e-3)


# ------------------------------------------------------- Block 26 reference ---

@pytest.mark.parametrize("absorption", ["partition_vertical", "hybrid"])
def test_against_block26_flat_thin(absorption):
    """Block 35 oracle (D-138.1): max |tau_fast - tau_Block26| <= 2e-4, same tube and directions (4 x 8 in
    both), half-length 10 um, line through the middle. One cheap case of the entry's grid: d 0.3 um, flat,
    axis 0.14 um off the plane, planes -3..3, n_avg 0 and 2."""
    cfg = cfg_fast(absorption)
    tube = geometry.Tube(c=(0.0, 0.0, 0.14), r=0.15, phi=0.0, theta=0.0, half_length=10.0)
    o, yh, eu, v = flat_line(30)
    ks = np.arange(-3, 4)
    for n_avg in (0, 2):
        a = line_fast.line_profiles(tube, 1.0, ks, o, yh, eu, v, n_avg, P, cfg)
        b = line_render.line_profiles(tube, 1.0, ks, o, yh, eu, v, n_avg, P, cfg)
        assert np.max(np.abs(a - b)) <= 2e-4, np.max(np.abs(a - b))


# ------------------------------------------------------------ invariances ---

def test_axial_translation_invariance_tilted_hybrid():
    """Amendment 1 (B): the block computes the tube without caps, so translating the capped tube along its
    own axis (caps far from the line) leaves tau unchanged; the cross-section is reused (a change of c alone
    hits the cache, amendment 1 (A)). Tolerance 2e-4, the accuracy budget of D-138.1: the slab lattice is
    re-tiled under the shift (midpoint rule, section 2.3)."""
    cfg = cfg_fast("hybrid")
    phi, th = math.radians(30.0), math.radians(20.0)
    a_hat = np.array([math.cos(phi) * math.cos(th), math.cos(phi) * math.sin(th), math.sin(phi)])
    yh = np.array([-math.sin(th), math.cos(th)])
    eu = np.array([math.cos(th), math.sin(th)])
    o = np.array([0.0, 0.0])
    v = np.arange(-25, 26) * P
    ks = np.array([-1, 0, 1])
    cache = line_fast.SectionCache()
    t0 = geometry.Tube(c=(0.0, 0.0, 0.0), r=0.4, phi=phi, theta=th, half_length=20.0)
    c1 = tuple(0.37 * a_hat)
    t1 = geometry.Tube(c=c1, r=0.4, phi=phi, theta=th, half_length=20.0)
    a = line_fast.line_profiles(t0, 1.0, ks, o, yh, eu, v, 0, P, cfg, cache=cache)
    m0 = cache.misses
    b = line_fast.line_profiles(t1, 1.0, ks, o, yh, eu, v, 0, P, cfg, cache=cache)
    assert cache.misses == m0 and cache.hits >= 1, (cache.hits, cache.misses)
    assert np.max(np.abs(a - b)) <= 2e-4, np.max(np.abs(a - b))


@pytest.mark.parametrize("absorption", ["linear", "hybrid"])
def test_whole_pixel_shift(absorption):
    """Block 35 oracle: a lateral shift of the tube by whole pixels, with the line shifted alike, gives the
    same samples [1e-12: the lattices are pixel-aligned, so only the float order of the sums changes]."""
    cfg = cfg_fast(absorption)
    th = math.radians(35.0)
    yh = np.array([-math.sin(th), math.cos(th)])
    eu = np.array([math.cos(th), math.sin(th)])
    v = np.arange(-20, 21) * 0.05
    ks = np.array([-1, 1])
    sx, sy = 3 * P, -2 * P
    t0 = geometry.Tube(c=(0.02, 0.01, 0.05), r=0.35, phi=math.radians(10), theta=th, half_length=10.0)
    t1 = geometry.Tube(c=(0.02 + sx, 0.01 + sy, 0.05), r=0.35, phi=math.radians(10), theta=th, half_length=10.0)
    a = line_fast.line_profiles(t0, 1.0, ks, np.array([0.0, 0.0]), yh, eu, v, 1, P, cfg)
    b = line_fast.line_profiles(t1, 1.0, ks, np.array([sx, sy]), yh, eu, v, 1, P, cfg)
    assert_allclose(a, b, rtol=0, atol=1e-12)


@pytest.mark.parametrize("absorption", ["linear", "partition_vertical", "hybrid"])
def test_mirror_symmetry(absorption):
    """Block 35 oracle: a flat tube through the origin on a pixel centre, v symmetric: tau(v) = tau(-v)
    [1e-10]. The condenser's azimuths psi_m = (m + 1/2) 2 pi / n_psi are symmetric under y -> -y (Block 9),
    so the hybrid is included."""
    cfg = cfg_fast(absorption)
    tube = geometry.Tube(c=(0.0, 0.0, 0.1), r=0.4, phi=0.0, theta=0.0, half_length=10.0)
    o, yh, eu, v = flat_line(25)
    tau = line_fast.line_profiles(tube, 1.2, np.array([-2, 0, 2]), o, yh, eu, v, 2, P, cfg)
    assert_allclose(tau, tau[:, ::-1], rtol=0, atol=1e-10)


def test_hybrid_faint_limit():
    """Section 2.3: as mu -> 0 the hybrid is linear at <1/s_z>_W mu, <1/s_z>_W over the quadrature in use
    (Block 9's directions at 4 x 8). Relative 5e-3 at mu d = 1e-3 (Block 24's tolerance for the same limit)."""
    cfg = cfg_fast("hybrid")
    S, W = ray_world.directions(4, 8, cfg.renderer.ray_na, cfg.renderer.ray_n_oil)
    g = float(np.sum(W / S[:, 2]))
    d = 0.5
    mu = 1e-3 / d
    tube = geometry.Tube(c=(0.0, 0.0, 0.0), r=d / 2, phi=0.0, theta=0.0, half_length=10.0)
    o, yh, eu, v = flat_line(20)
    ks = np.array([-1, 0, 1])
    hyb = 1.0 - line_fast.line_profiles(tube, mu, ks, o, yh, eu, v, 0, P, cfg)
    lin = 1.0 - line_fast.line_profiles(tube, g * mu, ks, o, yh, eu, v, 0, P, cfg, absorption="linear")
    core = lin > 0.2 * lin.max()
    assert_allclose(hyb[core], lin[core], rtol=5e-3)


# ------------------------------------------------------ cache and contract ---

def test_cache_key():
    """Amendment 1 (A): the section is cached by (d, mu, phi, theta), aspect, absorption, directions and light;
    a change of c alone reuses it. A change of theta, mu or d is a miss; a cached call equals an uncached one
    [1e-12]."""
    cfg = cfg_fast("hybrid")
    o, yh, eu, v = flat_line(10)
    ks = np.array([0])
    cache = line_fast.SectionCache()
    base = dict(r=0.3, phi=0.2, theta=0.0, half_length=10.0)
    t = geometry.Tube(c=(0.0, 0.0, 0.0), **base)
    first = line_fast.line_profiles(t, 1.0, ks, o, yh, eu, v, 0, P, cfg, cache=cache)
    assert (cache.hits, cache.misses) == (0, 1)
    again = line_fast.line_profiles(geometry.Tube(c=(0.0, 0.0, 0.3), **base), 1.0, ks, o, yh, eu, v, 0, P, cfg,
                                    cache=cache)
    assert (cache.hits, cache.misses) == (1, 1)
    line_fast.line_profiles(geometry.Tube(c=(0.0, 0.0, 0.0), r=0.3, phi=0.2, theta=0.05, half_length=10.0),
                            1.0, ks, o, yh, eu, v, 0, P, cfg, cache=cache)
    assert cache.misses == 2
    line_fast.line_profiles(t, 1.1, ks, o, yh, eu, v, 0, P, cfg, cache=cache)
    assert cache.misses == 3
    cold = line_fast.line_profiles(geometry.Tube(c=(0.0, 0.0, 0.3), **base), 1.0, ks, o, yh, eu, v, 0, P, cfg)
    assert_allclose(again, cold, rtol=0, atol=1e-12)
    assert first.shape == (1, v.size)


def test_contract_and_errors():
    """Shape (n_planes, n_v) float64; absorption None = cfg.inference.absorption; mu = 0 gives 1 exactly;
    the order of ks only permutes rows [exact]; determinism [exact]; ValueError for mu < 0, a non-unit
    axis, an unknown absorption (Block 26's)."""
    cfg = cfg_fast("linear")
    tube = geometry.Tube(c=(0.0, 0.0, 0.0), r=0.3, phi=0.3, theta=0.4, half_length=10.0)
    o, yh, eu, v = np.array([0.0, 0.0]), np.array([-math.sin(0.4), math.cos(0.4)]), \
        np.array([math.cos(0.4), math.sin(0.4)]), np.arange(-15, 16) * P
    ks = np.array([-2, 0, 3])
    a = line_fast.line_profiles(tube, 1.0, ks, o, yh, eu, v, 1, P, cfg)
    assert a.shape == (3, v.size) and a.dtype == np.float64 and np.all(np.isfinite(a))
    b = line_fast.line_profiles(tube, 1.0, ks, o, yh, eu, v, 1, P, cfg, absorption="linear")
    assert_array_equal(a, b)
    rev = line_fast.line_profiles(tube, 1.0, ks[::-1], o, yh, eu, v, 1, P, cfg)
    assert_array_equal(rev, a[::-1])
    assert_array_equal(line_fast.line_profiles(tube, 1.0, ks, o, yh, eu, v, 1, P, cfg), a)
    one = line_fast.line_profiles(tube, 0.0, ks, o, yh, eu, v, 1, P, cfg)
    assert_array_equal(one, np.ones((3, v.size)))
    with pytest.raises(ValueError):
        line_fast.line_profiles(tube, -0.1, ks, o, yh, eu, v, 1, P, cfg)
    with pytest.raises(ValueError):
        line_fast.line_profiles(tube, 1.0, ks, o, 2.0 * yh, eu, v, 1, P, cfg)
    with pytest.raises(ValueError):
        line_fast.line_profiles(tube, 1.0, ks, o, yh, eu, v, 1, P, cfg, absorption="ray_world")
    with pytest.raises(ValueError):
        line_fast.line_profiles(tube, float("nan"), ks, o, yh, eu, v, 1, P, cfg)


@pytest.mark.parametrize("phi_deg", [0.0, 30.0])
def test_extrusion_valid_boundary(phi_deg):
    """Amendment 1 (C): valid when the axis point nearest the line's origin is at least
    extrusion_cap_margin_um (3.0) along the axis from each cap. Axial caps: the axis inside is the arc length
    |s| <= U. Origin at arc length s from the centre, in the image plane at s cos(phi) along the heading.
    Checked 0.01 um either side of the margin; ExtrusionInvalid (a ValueError) raised also at mu = 0
    ('checked before the mu = 0 shortcut')."""
    cfg = cfg_fast("linear")
    phi, th, U = math.radians(phi_deg), 0.7, 10.0
    tube = geometry.Tube(c=(1.0, 2.0, 0.0), r=0.3, phi=phi, theta=th, half_length=U)
    yh = np.array([-math.sin(th), math.cos(th)])
    eu = np.array([math.cos(th), math.sin(th)])
    v = np.arange(-5, 6) * P
    ks = np.array([0])
    margin = cfg.inference.extrusion_cap_margin_um
    assert margin == 3.0
    for sign in (+1, -1):
        for s, ok in ((U - margin - 0.01, True), (U - margin + 0.01, False)):
            o = np.array([1.0, 2.0]) + sign * s * math.cos(phi) * eu
            assert line_fast.extrusion_valid(tube, ks, o, yh, eu, v, 0, P, cfg) is ok
            if not ok:
                with pytest.raises(line_fast.ExtrusionInvalid):
                    line_fast.line_profiles(tube, 0.0, ks, o, yh, eu, v, 0, P, cfg)
                assert issubclass(line_fast.ExtrusionInvalid, ValueError)
    assert line_fast.extrusion_valid(tube, ks, np.array([1.0, 2.0]), yh, eu, v, 0, P, cfg)
    near = np.array([1.0, 2.0]) + (U - 0.5) * math.cos(phi) * eu
    assert not line_fast.extrusion_valid(tube, ks, near, yh, eu, v, 0, P, cfg)
