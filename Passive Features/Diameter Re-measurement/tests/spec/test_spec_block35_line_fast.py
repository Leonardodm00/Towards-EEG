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


# ===================================================================================================
# Block 35, amendment 2 (D-151, D-156; D-157): the own shadow split by depth -- integration tester,
# 2026-11-17. Oracles from the amendment's text; the expected split from an independent 1-D reference
# (below), never from line_fast.
# ===================================================================================================

from scipy.special import ndtr  # noqa: E402

from spec_helpers import ref_sigma_r  # noqa: E402


def ref_split_flat(r, mu, zc, ks, n_lines, beta, cfg, absorption, layer_h=None, layer_z0=None):
    """Independent reference of the split for a flat infinite tube along x through (y = 0, z = zc), read along
    y on pixel centres v = n p (n in n_lines), n_avg = 0. Section 2.3 / Block 26 / Block 35 amendment 2:
      * fine lattice in y: samples (i + 1/2) h - p/2, h = p / line_fine_factor (section 2.1, Block 4);
      * a column is inside when |y| <= r; z_lo/hi = zc -+ sqrt(r^2 - y^2) (S3, flat);
      * slab lattice: zeta_j = z_min + (j + 1/2) dzeta over [min z_lo, max z_hi] of the sampled columns
        (Block 2 slab_centres as stated in section 2.3 / Block 24);
      * a_j (S4); linear: dA = a; partition (light +1, slabs numbered upward): T_<j (1 - e^{-a_j});
      * the pixel mean of the blurred plane at pixel n, for a tube infinite along x, is
        sum_s dA(y_s) (h / p) [Phi((y_hi - y_s)/sigma) - Phi((y_lo - y_s)/sigma)] (the x sum telescopes to 1);
      * w_{j,k} = |[zeta_j -+ dzeta/2] cap [z_k -+ beta]| / dzeta; u_{j,m} likewise with the layer.
    Returns (s_in, s_out, whole, layers, layer_centres)."""
    p = cfg.acquisition.res0_um
    f = cfg.inference.line_fine_factor
    h = p / f
    dzeta = cfg.renderer.dzeta_um
    nmax = int(math.ceil(r / h)) + f + 2
    ys = (np.arange(-nmax, nmax + 1) + 0.5) * h - 0.5 * p
    ys = ys[np.abs(ys) <= r]
    half = np.sqrt(np.clip(r * r - ys * ys, 0.0, None))
    z_lo, z_hi = zc - half, zc + half
    zmin, zmax = float(z_lo.min()), float(z_hi.max())
    J = max(1, int(math.ceil((zmax - zmin) / dzeta - 1e-12)))
    zeta = zmin + (np.arange(J) + 0.5) * dzeta
    a = mu * np.clip(np.minimum(z_hi[None, :], zeta[:, None] + dzeta / 2)
                     - np.maximum(z_lo[None, :], zeta[:, None] - dzeta / 2), 0.0, None)
    if absorption == "linear":
        dA = a
    elif absorption == "partition_vertical":
        before = np.vstack([np.zeros((1, ys.size)), np.cumsum(a, axis=0)[:-1]])
        dA = np.exp(-before) * (-np.expm1(-a))
    else:
        raise ValueError(absorption)
    yc = np.asarray(n_lines) * p                       # pixel centres read (v = n p)
    zk = np.asarray(ks) * cfg.acquisition.dz_um
    terms = np.zeros((J, len(ks), yc.size))
    for j in range(J):
        for k, z in enumerate(zk):
            s = float(ref_sigma_r(zeta[j] - z, cfg.renderer))
            K = ndtr((yc[:, None] + p / 2 - ys[None, :]) / s) - ndtr((yc[:, None] - p / 2 - ys[None, :]) / s)
            terms[j, k] = (h / p) * (K @ dA[j])
    lo, hi = zeta - dzeta / 2, zeta + dzeta / 2
    w = np.clip(np.minimum(hi[:, None], zk[None, :] + beta) - np.maximum(lo[:, None], zk[None, :] - beta),
                0.0, None) / dzeta                     # (J, n_planes)
    s_in = np.einsum("jk,jkn->kn", w, terms)
    s_out = np.einsum("jk,jkn->kn", 1.0 - w, terms)
    whole = terms.sum(axis=0)
    layers = cen = None
    if layer_h is not None:
        z0 = zc if layer_z0 is None else layer_z0
        m = np.arange(int(math.floor((zmin - z0) / layer_h)) - 2, int(math.ceil((zmax - z0) / layer_h)) + 3)
        cen = z0 + m * layer_h
        u = np.clip(np.minimum(hi[:, None], cen[None, :] + layer_h / 2)
                    - np.maximum(lo[:, None], cen[None, :] - layer_h / 2), 0.0, None) / dzeta
        keep = u.max(axis=0) > 0
        cen, u = cen[keep], u[:, keep]
        layers = np.einsum("jm,jkn->mkn", u, terms)
    return s_in, s_out, whole, layers, cen


def _flat_args(n_half=26):
    o, yh, eu, _v = flat_line(n_half)
    n = np.arange(-n_half, n_half + 1)
    return o, yh, eu, n * P, n


@pytest.mark.parametrize("absorption", ["linear", "partition_vertical"])
@pytest.mark.parametrize("r,zc,beta", [(0.3, 0.05, 0.14), (0.5, -0.11, 0.05), (0.5, 0.0, 0.3)])
def test_split_against_independent_reference(absorption, r, zc, beta):
    """Amendment 2: S_in = sum_j w_jk s_jk, S_out = sum_j (1 - w_jk) s_jk with w the fraction of the slab's
    thickness inside the band, and the layers sum_j u_jm s_jk, against ref_split_flat. Tolerance 1e-8 (absolute,
    on a dimensionless deficit of order 0.1-0.5): the reference is the same continuous formula; the only
    expected differences are the reach rule's truncation at 6 sigma (2 Phi(-6) = 2e-9 relative) and rounding."""
    cfg = cfg_fast(absorption)
    mu = 1.3
    tube = geometry.Tube(c=(0.0, 0.0, zc), r=r, phi=0.0, theta=0.0, half_length=10.0)
    o, yh, eu, v, n = _flat_args()
    ks = np.arange(-4, 5)
    got = line_fast.line_shadow_split(tube, mu, ks, o, yh, eu, v, 0, P, cfg, beta, absorption=absorption,
                                      layer_um=0.1)
    s_in, s_out, whole, layers, cen = ref_split_flat(r, mu, zc, ks, n, beta, cfg, absorption, layer_h=0.1)
    assert_allclose(1.0 - line_fast.line_profiles(tube, mu, ks, o, yh, eu, v, 0, P, cfg, absorption=absorption),
                    whole, rtol=0, atol=1e-8)        # the reference itself, against the unsplit renderer
    assert_allclose(got.s_in, s_in, rtol=0, atol=1e-8)
    assert_allclose(got.s_out, s_out, rtol=0, atol=1e-8)
    assert_allclose(got.layer_z, cen, rtol=0, atol=1e-12)
    assert_allclose(got.layers, layers, rtol=0, atol=1e-8)


@pytest.mark.parametrize("absorption", ["linear", "partition_vertical", "hybrid"])
def test_split_monotone_in_band_and_bounds(absorption):
    """w_{j,k} is non-decreasing in beta and every term s_jk >= 0 (absorbed light, positive kernel), so
    S_in(beta1) <= S_in(beta2) for beta1 < beta2 pointwise, S_in, S_out >= 0, and S_in <= 1 - tau.
    Tolerance 1e-15 (rounding of sums of non-negative terms). Tilted tube (30 deg), off-plane axis."""
    cfg = cfg_fast(absorption, n_rho=2, n_psi=4)
    th = 0.4
    tube = geometry.Tube(c=(0.0, 0.0, 0.07), r=0.35, phi=math.radians(30), theta=th, half_length=5.0)
    o, yh, eu = np.array([0.0, 0.0]), np.array([-math.sin(th), math.cos(th)]), np.array([math.cos(th), math.sin(th)])
    v = np.arange(-20, 21) * P
    ks = np.arange(-4, 5)
    prev = None
    for beta in (0.0, 0.03, 0.1, 0.14, 0.3, 0.9):
        sp = line_fast.line_shadow_split(tube, 1.0, ks, o, yh, eu, v, 1, P, cfg, beta)
        assert np.all(sp.s_in >= -1e-15) and np.all(sp.s_out >= -1e-15)
        if prev is not None:
            assert np.all(sp.s_in >= prev - 1e-15), beta
        prev = sp.s_in


def test_split_layer_equals_band():
    """A layer of thickness h = 2 beta centred on z_k (layer_origin_um = z_k) is the band of plane k, so
    L_{0,k} = S_in_k [1e-12]; and the layers' centres are layer_origin + m h (spacing h, increasing)."""
    cfg = cfg_fast("partition_vertical")
    tube = geometry.Tube(c=(0.0, 0.0, 0.09), r=0.4, phi=math.radians(20), theta=0.0, half_length=8.0)
    o, yh, eu, v, _n = _flat_args(20)
    beta = 0.07
    ks = np.arange(-3, 4)
    sp = line_fast.line_shadow_split(tube, 1.0, ks, o, yh, eu, v, 0, P, cfg, beta)
    for i, k in enumerate(ks):
        zk = k * cfg.acquisition.dz_um
        lay = line_fast.line_shadow_split(tube, 1.0, ks, o, yh, eu, v, 0, P, cfg, beta, layer_um=2 * beta,
                                          layer_origin_um=zk)
        m = (lay.layer_z - zk) / (2 * beta)
        assert_allclose(m, np.round(m), rtol=0, atol=1e-9)
        assert np.all(np.diff(lay.layer_z) > 0)
        i0 = int(np.argmin(np.abs(lay.layer_z - zk)))
        assert abs(lay.layer_z[i0] - zk) < 1e-12
        assert_allclose(lay.layers[i0, i], sp.s_in[i], rtol=0, atol=1e-12)


def test_split_layer_origin_default_and_depth_shift():
    """layer_origin_um None = the tube's centre depth tube.c[2]: layer_z - c_z are multiples of h [1e-9];
    a depth shift of the tube by dz with every plane index raised by one leaves s_in, s_out unchanged [1e-9]
    (the amendment's own tolerance)."""
    cfg = cfg_fast("linear")
    o, yh, eu, v, _n = _flat_args(20)
    dz = cfg.acquisition.dz_um
    ks = np.arange(-3, 4)
    t0 = geometry.Tube(c=(0.0, 0.0, 0.123), r=0.4, phi=0.0, theta=0.0, half_length=8.0)
    t1 = geometry.Tube(c=(0.0, 0.0, 0.123 + dz), r=0.4, phi=0.0, theta=0.0, half_length=8.0)
    a = line_fast.line_shadow_split(t0, 0.8, ks, o, yh, eu, v, 0, P, cfg, 0.14, layer_um=0.1)
    b = line_fast.line_shadow_split(t1, 0.8, ks + 1, o, yh, eu, v, 0, P, cfg, 0.14, layer_um=0.1)
    m = (a.layer_z - 0.123) / 0.1
    assert_allclose(m, np.round(m), rtol=0, atol=1e-9)
    assert_allclose(b.s_in, a.s_in, rtol=0, atol=1e-9)
    assert_allclose(b.s_out, a.s_out, rtol=0, atol=1e-9)


def test_split_linear_in_mu_and_known_mass():
    """Under `linear` each term is proportional to mu: S_in(2 mu) = 2 S_in(mu) [1e-12 relative]. With
    beta = dz/2 and a flat tube whose depth extent lies inside the planes' range the bands tile the depth, so
    sum_k p sum_n S_in_k(v_n) = mu pi r^2 up to the lattice's point sampling of the circle (rtol 3e-3, as the
    earlier mass oracle) -- the amendment's oracle (vi), here against the continuum value."""
    cfg = cfg_fast("linear")
    o, yh, eu, v, _n = _flat_args(40)
    r, mu = 0.45, 0.9
    tube = geometry.Tube(c=(0.0, 0.0, 0.03), r=r, phi=0.0, theta=0.0, half_length=10.0)
    ks = np.arange(-5, 6)
    beta = cfg.acquisition.dz_um / 2
    a = line_fast.line_shadow_split(tube, mu, ks, o, yh, eu, v, 0, P, cfg, beta)
    b = line_fast.line_shadow_split(tube, 2 * mu, ks, o, yh, eu, v, 0, P, cfg, beta)
    assert_allclose(b.s_in, 2 * a.s_in, rtol=1e-12, atol=1e-15)
    assert_allclose(P * a.s_in.sum(), mu * math.pi * r * r, rtol=3e-3)


def test_split_contract_and_errors():
    """Contract: OwnSplit with s_in, s_out (n_planes, n_v) float64 finite, rows in the order of ks (a
    permutation of ks permutes the rows exactly); band_half_um, layer_um stored; layers None without
    layer_um; mu = 0 gives zeros [exact]; no planes gives (0, n_v); determinism [exact]; ValueError for
    band_half_um < 0 / nan / inf, layer_um <= 0 / nan, layer_origin_um nan; ExtrusionInvalid (checked first,
    also at mu = 0) for a line 0.5 um from a cap."""
    cfg = cfg_fast("linear")
    th = 0.3
    tube = geometry.Tube(c=(0.0, 0.0, 0.0), r=0.3, phi=0.2, theta=th, half_length=6.0)
    o, yh, eu = np.array([0.0, 0.0]), np.array([-math.sin(th), math.cos(th)]), np.array([math.cos(th), math.sin(th)])
    v = np.arange(-12, 13) * P
    ks = np.array([-2, 0, 3, 1])
    sp = line_fast.line_shadow_split(tube, 1.0, ks, o, yh, eu, v, 1, P, cfg, 0.14)
    assert isinstance(sp, line_fast.OwnSplit)
    assert sp.s_in.shape == sp.s_out.shape == (4, v.size)
    assert sp.s_in.dtype == np.float64 and np.all(np.isfinite(sp.s_in)) and np.all(np.isfinite(sp.s_out))
    assert sp.layers is None and sp.layer_z is None and sp.layer_um is None and sp.band_half_um == 0.14
    perm = np.array([2, 0, 3, 1])
    sp2 = line_fast.line_shadow_split(tube, 1.0, ks[perm], o, yh, eu, v, 1, P, cfg, 0.14)
    assert_array_equal(sp2.s_in, sp.s_in[perm])
    assert_array_equal(sp2.s_out, sp.s_out[perm])
    again = line_fast.line_shadow_split(tube, 1.0, ks, o, yh, eu, v, 1, P, cfg, 0.14)
    assert_array_equal(again.s_in, sp.s_in)
    z = line_fast.line_shadow_split(tube, 0.0, ks, o, yh, eu, v, 1, P, cfg, 0.14, layer_um=0.1)
    assert_array_equal(z.s_in, 0.0)
    assert_array_equal(z.s_out, 0.0)
    assert np.all(z.layers == 0.0)
    e = line_fast.line_shadow_split(tube, 1.0, np.array([], dtype=int), o, yh, eu, v, 1, P, cfg, 0.14)
    assert e.s_in.shape == (0, v.size)
    for bad in (-0.01, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            line_fast.line_shadow_split(tube, 1.0, ks, o, yh, eu, v, 1, P, cfg, bad)
    for bad in (0.0, -0.1, float("nan")):
        with pytest.raises(ValueError):
            line_fast.line_shadow_split(tube, 1.0, ks, o, yh, eu, v, 1, P, cfg, 0.14, layer_um=bad)
    with pytest.raises(ValueError):
        line_fast.line_shadow_split(tube, 1.0, ks, o, yh, eu, v, 1, P, cfg, 0.14, layer_um=0.1,
                                    layer_origin_um=float("nan"))
    near = np.array([math.cos(th), math.sin(th)]) * (6.0 - 0.5) * math.cos(0.2)
    with pytest.raises(line_fast.ExtrusionInvalid):
        line_fast.line_shadow_split(tube, 0.0, ks, near, yh, eu, v, 1, P, cfg, 0.14)


@pytest.mark.parametrize("phi_deg", [0.0, 60.0])
def test_split_tiling_hybrid_long_tube(phi_deg):
    """S_in + S_out = 1 - line_profiles and sum_m L_m = the same [1e-12] under the hybrid, on a steep (60 deg)
    tube of half-length 10 um (the reach rule crops the footprint; the split must lay the same slabs as the
    slab loop) and a flat one. 2 x 4 directions for time."""
    cfg = cfg_fast("hybrid", n_rho=2, n_psi=4)
    th = 1.1
    tube = geometry.Tube(c=(0.2, -0.1, 0.05), r=0.4, phi=math.radians(phi_deg), theta=th, half_length=10.0)
    o, yh, eu = np.array([0.2, -0.1]), np.array([-math.sin(th), math.cos(th)]), np.array([math.cos(th), math.sin(th)])
    v = np.arange(-15, 16) * P
    ks = np.arange(-4, 5)
    sp = line_fast.line_shadow_split(tube, 1.0, ks, o, yh, eu, v, 1, P, cfg, 0.14, layer_um=0.1)
    whole = 1.0 - line_fast.line_profiles(tube, 1.0, ks, o, yh, eu, v, 1, P, cfg)
    assert_allclose(sp.s_in + sp.s_out, whole, rtol=0, atol=1e-12)
    assert_allclose(sp.layers.sum(axis=0), whole, rtol=0, atol=1e-12)


# ------------------------------------------- amendment 3 (wave 7; tester, 2026-11-23 13:55) ---

_B35_CASES = r'''
import dataclasses, math, os
import numpy as np
from allen_diameter.config import default_config
from allen_diameter.model import geometry, line_fast

P = 0.1144


def b35_cases():
    """The Gaussian path's outputs on cases spanning the three absorptions, tilts, headings, depth offsets,
    n_avg, a shared cache, line_shadow_split (amendment 2) with layers, and extrusion_valid."""
    out = {}
    base = default_config()
    for absorption in ("linear", "partition_vertical", "hybrid"):
        i = dataclasses.replace(base.inference, absorption=absorption, ray_n_rho=2, ray_n_psi=4)
        cfg = dataclasses.replace(base, inference=i)
        cache = line_fast.SectionCache()
        for g, (d, tilt, th, zoff, n_avg, mu) in enumerate([(0.6, 0.0, 0.0, 0.0, 0, 1.0),
                                                            (1.0, 0.4, 0.52, 0.14, 4, 1.7),
                                                            (0.3, 0.2, 1.1, 0.05, 2, 0.6)]):
            yh = np.array([-math.sin(th), math.cos(th)])
            eu = np.array([math.cos(th), math.sin(th)])
            o = np.array([0.03, -0.02])
            tube = geometry.Tube((0.03, -0.02, zoff), d / 2.0, tilt, th, 1.0, 8.0, "axial")
            v = np.arange(-26, 27) * P
            ks = list(range(-3, 4))
            out["lp_%s_%d" % (absorption, g)] = line_fast.line_profiles(tube, mu, ks, o, yh, eu, v, n_avg, P, cfg)
            out["lpc_%s_%d" % (absorption, g)] = line_fast.line_profiles(tube, mu, ks, o, yh, eu, v, n_avg, P, cfg,
                                                                         cache=cache)
            out["ev_%s_%d" % (absorption, g)] = np.array(line_fast.extrusion_valid(tube, ks, o, yh, eu, v, n_avg, P,
                                                                                   cfg))
            if g == 1:
                sp = line_fast.line_shadow_split(tube, mu, ks, o, yh, eu, v, n_avg, P, cfg, 0.14, layer_um=0.1)
                out["sin_%s" % absorption] = sp.s_in
                out["sout_%s" % absorption] = sp.s_out
                out["lay_%s" % absorption] = sp.layers
                out["layz_%s" % absorption] = sp.layer_z
    return out


if __name__ == "__main__":
    np.savez(os.path.join(os.environ["OUT"], "b35_base.npz"), **b35_cases())
'''


def test_amendment3_gaussian_path_bit_identical_to_base(tmp_path):
    """Block 35 amendment 3 / Block 49 (vii) (B49.2): kernel=None is Block 35's path "unchanged bit for bit", and
    line_shadow_split keeps the Gaussian (D-222.4): the same cases computed with the base commit's package (git
    archive of f1bc7cb, before wave 7, in a subprocess) and with the commit under test are equal bit for bit
    (line_profiles with and without a shared cache, extrusion_valid, line_shadow_split's s_in, s_out, layers,
    layer_z), in the three absorptions."""
    import wave7_refs
    wave7_refs.run_at_commit(_B35_CASES, tmp_path)
    ns = {}
    exec(compile(_B35_CASES, "b35_cases", "exec"), ns)
    head = ns["b35_cases"]()
    with np.load(tmp_path / "b35_base.npz") as z:
        assert sorted(z.files) == sorted(head)
        for key in z.files:
            assert z[key].shape == np.asarray(head[key]).shape, key
            assert_array_equal(np.asarray(head[key]), z[key], err_msg=key)
