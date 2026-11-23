"""Spec-derived tests of Block 49 (the SGT point-spread function in the fast line renderer; wave 7, D-221, D-222,
D-224, D-225, D-232) and of Block 35 amendment 3 (the `kernel` argument) -- integration tester, 2026-11-23 13:55.

Oracles from specs/SPEC.md (section 2.3, Block 12's (H1)-(H3), Block 35, Block 49); expected values from the
closed forms and scipy (tests/spec/wave7_refs.py, which imports nothing under test) or from the Gaussian path
(kernel=None) where the spec names it as the reference of the Gaussian member, never from line_sgt itself.

Run: python3 -m pytest -q tests/spec/test_spec_block49_line_sgt.py
Pure ASCII.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import special

import nc_helpers as H  # noqa: F401  (sys.path, bilinear_read)
import wave7_refs as R
from allen_diameter.config import default_config
from allen_diameter.model import geometry, kernel as kernel_mod, line_fast, line_sgt

P = 0.1144
O = np.array([0.0, 0.0])
YH = np.array([0.0, 1.0])
EU = np.array([1.0, 0.0])
V26 = np.arange(-26, 27) * P          # Block 49 (i): v = n p, n = -26..26


def cfg_const(sigma, reach=12.0, absorption="linear", **inf):
    """The default configuration with a law of constant SD sigma ("frozen" beyond the table): every (slab, plane)
    pair has the same kernel, so that "linear" absorption sums to mu l(y) whatever the slab lattice."""
    c = default_config()
    n = len(c.renderer.kernel_table_sigma_um)
    r = dataclasses.replace(c.renderer, kernel_table_sigma_um=(sigma,) * n, sigma_r0_um=sigma,
                            kernel_continuation="frozen")
    i = dataclasses.replace(c.inference, absorption=absorption, sgt_reach_sigmas=float(reach), **inf)
    c = dataclasses.replace(c, renderer=r, inference=i)
    c.validate()
    return c


def cfg_dirs(absorption="partition_vertical", n_rho=2, n_psi=4, **inf):
    """The default law; the inference's directions reduced for time (Block 49 (i) allows 2 x 4 in the test)."""
    c = default_config()
    i = dataclasses.replace(c.inference, absorption=absorption, ray_n_rho=n_rho, ray_n_psi=n_psi, **inf)
    c = dataclasses.replace(c, inference=i)
    c.validate()
    return c


def reach_for(shape, tail=1e-7, cap=400):
    """The smallest whole number of law SDs (>= 12) whose 2-D tail mass (independent quadrature) is <= tail, or
    None above cap (bisection on the monotone tail)."""
    W = R.unit_fwhm(*shape)

    def ok(s):
        return R.tail_mass_2d(shape[0], shape[1], s * W / R.FWHM_PER_SD) <= tail
    if ok(12):
        return 12
    if not ok(cap):
        return None
    lo, hi = 12, cap
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if ok(mid):
            hi = mid
        else:
            lo = mid
    return hi


# ---------------------------------------------------------------- the API ---

def test_shape_domain_and_constants():
    """SGTKernel's domain (Block 12's check_sgt_shape: finite, beta > 0, xi >= 0, zeta > 0); frozen; FWHM_PER_SD."""
    k = line_sgt.SGTKernel(2.0, 0.0)
    assert k.zeta == 1.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        k.beta = 3.0
    for bad in [(0.0, 0.1, 1.0), (-1.0, 0.1, 1.0), (2.0, -1e-9, 1.0), (2.0, 0.1, 0.0), (2.0, 0.1, -1.0),
                (math.nan, 0.1, 1.0), (2.0, math.inf, 1.0), (2.0, 0.1, math.nan)]:
        with pytest.raises(ValueError):
            line_sgt.SGTKernel(*bad)
    assert_allclose(line_sgt.FWHM_PER_SD, 2.0 * math.sqrt(2.0 * math.log(2.0)), rtol=0, atol=1e-15)


@pytest.mark.parametrize("beta,xi,r", [(2.0, 0.0, 0.7), (2.0, 0.0, 2.3), (1.0, 0.0, 0.4), (1.0, 0.0, 5.0),
                                       (2.0, 0.5, 3.0), (2.0, 1.0, 7.81), (2.0, 2.0, 1.5), (2.0, 0.3, 8.6),
                                       (1.3, 0.0, 3.0), (4.0, 0.0, 1.2), (1.5, 0.7, 4.0), (7.0, 1.9, 2.0)])
def test_sgt_tail_mass(beta, xi, r):
    """B49.7: the isotropic unit kernel's 2-D mass beyond r against the closed forms the spec names (Gaussian
    exp(-r^2), Laplace r K_1(r), bivariate t (1 + xi r^2)^(-1/xi)) and, for the other members, against the Abel
    identity by an independent quadrature [1e-8 absolute, the spec's bound; quad's error is below 1e-13]."""
    got = line_sgt.sgt_tail_mass(beta, xi, r)
    if (beta, xi) == (2.0, 0.0):
        ref = math.exp(-r * r)
    elif (beta, xi) == (1.0, 0.0):
        ref = r * special.k1(r)
    elif beta == 2.0:
        ref = (1.0 + xi * r * r) ** (-1.0 / xi)
    else:
        ref = R.tail_mass_2d(beta, xi, r)
    assert abs(got - ref) <= 1e-8, (got, ref)
    assert line_sgt.sgt_tail_mass(beta, xi, 0.0) == 1.0


@pytest.mark.parametrize("shape,reach", [((2.0, 1.0, 1.0), 12.0), ((2.0, 0.5, 1.0), 12.0), ((2.0, 0.3, 1.3), 12.0),
                                         ((2.0, 0.3, 0.77), 12.0), ((1.3, 0.2, 1.0), 12.0), ((2.0, 2.0, 1.0), 12.0),
                                         ((2.0, 0.5, 1.0), 117.0)])
def test_truncated_mass(shape, reach):
    """B49.7: truncated_mass = the tail at R/a = sgt_reach_sigmas W(kappa) / FWHM_PER_SD, W the full (H3) FWHM,
    zeta included (D-225.4) [1e-8]. (2, 1, 1) at 12 law SDs is 0.016125 (the spec's "0.01612 as the closed form")."""
    cfg = cfg_dirs(sgt_reach_sigmas=reach)
    ra = reach * R.unit_fwhm(*shape) / R.FWHM_PER_SD
    ref = R.tail_mass_2d(shape[0], shape[1], ra)
    got = line_sgt.truncated_mass(line_sgt.SGTKernel(*shape), cfg)
    assert abs(got - ref) <= 1e-8, (got, ref)
    if shape == (2.0, 1.0, 1.0) and reach == 12.0:
        assert abs(got - 1.0 / (1.0 + ra * ra)) <= 1e-12 and abs(got - 0.016125) < 1e-6
    with pytest.raises(ValueError):
        line_sgt.truncated_mass((2.0, 1.0, 1.0), cfg)


# ------------------------------------------- the 1-D reduction (spec (ii)) ---

SHAPES_1D = [(2.0, 0.5, 1.0), (1.3, 0.0, 1.0), (4.0, 0.0, 1.0), (2.0, 0.3, 1.3), (2.0, 0.3, 0.77),
             (2.0, 0.0, 1.0), (1.5, 0.1, 1.0), (3.0, 0.2, 1.6)]


@pytest.mark.parametrize("shape", SHAPES_1D)
@pytest.mark.parametrize("sigma", [0.08, 0.30, 0.45])
def test_one_dimensional_reduction(shape, sigma):
    """B49.5 (spec (ii) in a geometry where the reference is exact): a tube along x read at pixel centres across it,
    a law of constant SD (all pairs one kernel; sigma 0.08, 0.30, 0.45 um put the pairs on Block 49's level 0, 1
    and 2 at line_fine_factor 8), "linear" absorption, the reach raised until the dropped mass is <= 1e-7: the
    deficit equals the 1-D model with the two-piece SGT's CDF (wave7_refs.flat_line_ref), the right side towards +v.
    Bound 5e-5 (Block 49's, 0.007 of the pilot's noise in tau units); measured differences printed. Two tubes: r 0.2
    um through the origin and r 0.75 um with its axis 0.37 um off the line's origin (so the profile is not even);
    the tilt 0.3 rad on the second (the column scales by 1/cos(phi) only, the kernel being depth-independent)."""
    reach = reach_for(shape)
    if reach is None or sigma * reach * max(2.1, shape[2], 1.0 / shape[2]) > 30.0:
        pytest.skip("the raised reach exceeds the tube's length here")
    cfg = cfg_const(sigma, reach)
    worst = 0.0
    for r, y_c, phi in [(0.2, 0.0, 0.0), (0.75, 0.37, 0.3)]:
        U = 3.5 + sigma * reach * max(2.1, shape[2], 1.0 / shape[2]) + 2.0
        tube = geometry.Tube((0.0, y_c, 0.0), r, phi, 0.0, 1.0, U, "axial")
        tau = line_fast.line_profiles(tube, 1.3, [-1, 0, 2], O, YH, EU, V26, 0, P, cfg, absorption="linear",
                                      kernel=line_sgt.SGTKernel(*shape))
        ref = R.flat_line_ref(r, 1.3, y_c, phi, V26, P, 8, sigma, *shape)
        for k in range(3):
            d = float(np.max(np.abs((1.0 - tau[k]) - ref)))
            worst = max(worst, d)
    print("shape %s sigma %.2f reach %d: max |delta tau| %.2e" % (shape, sigma, reach, worst))
    assert worst <= 5e-5, worst


@pytest.mark.parametrize("shape", [(1.0, 0.0, 1.0), (2.0, 2.0, 1.0), (2.0, 1.0, 1.0), (1.0, 0.0, math.exp(0.7)),
                                   (2.0, 2.0, math.exp(-0.7)), (2.0, 0.0, math.exp(0.7))])
@pytest.mark.parametrize("sigma", [0.08, 0.45])
def test_one_dimensional_reduction_truncated(shape, sigma):
    """B49.5 at the default reach (12 law SDs) against the 1-D model of the truncated kernel (spec (vi)'s check,
    D-225.4: the isotropic PSF truncated at R in its radial variable before the stretch, its mass dropped), for the
    members whose 2-D profile is known in closed form (beta = 2: the bivariate t / the Gaussian; the Laplace (1, 0):
    K_0/(2 pi)) -- the corners of the fit's box among them: beta = 1 (sgt_log10_beta_bounds' lower end, the cusp),
    xi = 2 (sgt_xi_bounds' upper end, Cauchy tails: about 10 % of the kernel dropped), zeta = e^+-0.7
    (sgt_ln_zeta_bounds' ends). (2, 1, 1) is the spec's (vi) member. Bound 5e-5 (Block 49's)."""
    cfg = cfg_const(sigma, 12.0)
    kern = line_sgt.SGTKernel(*shape)
    Z = 1.0 if shape[2] == 1.0 else max(2.1, shape[2], 1.0 / shape[2])
    U = 3.5 + sigma * 12.0 * Z + 2.0
    worst = 0.0
    for r, y_c in [(0.2, 0.0), (0.6, 0.23)]:
        tube = geometry.Tube((0.0, y_c, 0.0), r, 0.0, 0.0, 1.0, U, "axial")
        tau = line_fast.line_profiles(tube, 1.3, [0], O, YH, EU, V26, 0, P, cfg, absorption="linear", kernel=kern)
        ref = R.flat_line_ref(r, 1.3, y_c, 0.0, V26, P, 8, sigma, *shape, reach_sigmas=12.0)
        worst = max(worst, float(np.max(np.abs((1.0 - tau[0]) - ref))))
    print("shape %s sigma %.2f: max |delta tau| %.2e, truncated_mass %.2e"
          % (shape, sigma, worst, line_sgt.truncated_mass(kern, cfg)))
    assert worst <= 5e-5, worst


@pytest.mark.parametrize("shape", [(2.0, 1.0, 1.0), (2.0, 0.5, 1.0), (2.0, 0.3, 1.3), (1.0, 0.0, 1.0),
                                   (2.0, 2.0, 1.0)])
def test_mass_not_renormalised(shape):
    """B49.6: at the default reach the kernel's mass beyond R is dropped, not renormalised (D-225.4), and the stretch
    keeps the fraction: p sum_n (1 - tau(v_n)) = (1 - truncated_mass) sum_m h mu l(y_m) on a line that holds the
    truncated support (the blur and the pixel mean conserve mass; the read at pixel centres is the pixel). The dropped
    fraction is computed independently (Abel quadrature). Tolerance 1e-6 relative: a renormalised kernel would miss by
    the dropped fraction itself (1e-3 to 1e-1 for these shapes), while the module's tabulated transfer carries its
    own error at nu = 0 (its documented table accuracy, 1e-9 to 1e-8; measured 1.4e-9 to 2.0e-8 relative here)."""
    sigma, r, mu = 0.08, 0.2, 1.3
    cfg = cfg_const(sigma, 12.0)
    ra = 12.0 * R.unit_fwhm(*shape) / R.FWHM_PER_SD
    tm = R.tail_mass_2d(shape[0], shape[1], ra)
    v = np.arange(-60, 61) * P                          # +-6.9 um >> r + max(zeta, 1/zeta) 12 sigma
    tube = geometry.Tube((0.0, 0.0, 0.0), r, 0.0, 0.0, 1.0, 8.0, "axial")
    tau = line_fast.line_profiles(tube, mu, [0], O, YH, EU, v, 0, P, cfg, absorption="linear",
                                  kernel=line_sgt.SGTKernel(*shape))
    got = P * float(np.sum(1.0 - tau[0]))
    ref = (1.0 - tm) * R.flat_line_mass(r, mu, 0.0, 0.0, P, 8)
    print("shape %s: mass %.12f, (1 - tm) x lattice mass %.12f, tm %.3e" % (shape, got, ref, tm))
    assert_allclose(got, ref, rtol=1e-6, atol=0)


# ---------------------------------- the Gaussian member (spec (i)) and the law ---

GEOMS_I = [(0.3, 0.3, 0.52, 0.0), (1.0, 0.3, 0.0, 0.14), (1.0, 0.0, 0.52, 0.14), (0.3, 0.0, 0.0, 0.0),
           (1.0, 0.3, 0.52, 0.0)]


@pytest.mark.parametrize("absorption", ["partition_vertical", "hybrid"])
@pytest.mark.parametrize("geom", GEOMS_I)
def test_gaussian_member_equals_gaussian_path(absorption, geom):
    """B49.4 (spec (i)): (2, 0, 1) is the Gaussian of SD sigma (its core e^{-x^2}/sqrt(pi) at a = w/W = sqrt(2)
    sigma), so SGTKernel(2, 0, 1) and kernel=None render the same tube within 5e-5; the default law (depth-dependent),
    tilted tubes, the axis in the plane and 0.14 um off, planes -4..4, n_avg 4, half-length 8 um, 2 x 4 directions."""
    d, tilt, heading, zoff = geom
    cfg = cfg_dirs(absorption)
    yh = np.array([-math.sin(heading), math.cos(heading)])
    eu = np.array([math.cos(heading), math.sin(heading)])
    tube = geometry.Tube((0.0, 0.0, zoff), d / 2.0, tilt, heading, 1.0, 8.0, "axial")
    ks = list(range(-4, 5))
    g = line_fast.line_profiles(tube, 1.5, ks, O, yh, eu, V26, 4, P, cfg)
    s = line_fast.line_profiles(tube, 1.5, ks, O, yh, eu, V26, 4, P, cfg, kernel=line_sgt.SGTKernel(2.0, 0.0, 1.0))
    diff = float(np.max(np.abs(s - g)))
    print("geom %s %s: max |SGT(2,0,1) - Gaussian| %.2e (min tau %.3f)" % (geom, absorption, diff, g.min()))
    assert diff <= 5e-5


def test_width_switch_reaches_the_sgt():
    """B49.13 (Block 36 amendment 22: the switch acts on the law, so the SGT's FWHM follows it): with a table
    re-anchored by kernel.with_kernel (sigma_0 0.15 um, c 1.4), SGTKernel(2, 0, 1) equals kernel=None under the
    same configuration [5e-5], and differs from the default law's render (the switch reached it)."""
    base = cfg_dirs("partition_vertical")
    ren = kernel_mod.with_kernel(base.renderer, 0.15, 1.4)
    cfg = dataclasses.replace(base, renderer=ren)
    cfg.validate()
    tube = geometry.Tube((0.0, 0.0, 0.07), 0.35, 0.3, 0.52, 1.0, 8.0, "axial")
    yh = np.array([-math.sin(0.52), math.cos(0.52)])
    eu = np.array([math.cos(0.52), math.sin(0.52)])
    ks = [-2, 0, 2]
    g = line_fast.line_profiles(tube, 1.5, ks, O, yh, eu, V26, 4, P, cfg)
    s = line_fast.line_profiles(tube, 1.5, ks, O, yh, eu, V26, 4, P, cfg, kernel=line_sgt.SGTKernel(2.0, 0.0))
    s0 = line_fast.line_profiles(tube, 1.5, ks, O, yh, eu, V26, 4, P, base, kernel=line_sgt.SGTKernel(2.0, 0.0))
    assert float(np.max(np.abs(s - g))) <= 5e-5
    assert float(np.max(np.abs(s - s0))) > 1e-3


# ------------------------------------------------------------ symmetries ---

@pytest.mark.parametrize("absorption", ["linear", "partition_vertical"])
def test_zeta_mirror(absorption):
    """B49.8 (spec (iv)): zeta -> 1/zeta mirrors the profile, v -> -v, of a tube symmetric about the line's origin
    (a flat tube along x through a pixel centre: the reflection y -> -y maps the scene and the lattices onto
    themselves) [1e-10]; the default (depth-dependent) law, planes -2..2, n_avg 4."""
    cfg = cfg_dirs(absorption)
    tube = geometry.Tube((0.0, 0.0, 0.05), 0.3, 0.0, 0.0, 1.0, 8.0, "axial")
    ks = [-2, -1, 0, 1, 2]
    a = line_fast.line_profiles(tube, 1.2, ks, O, YH, EU, V26, 4, P, cfg, kernel=line_sgt.SGTKernel(2.0, 0.3, 1.3))
    b = line_fast.line_profiles(tube, 1.2, ks, O, YH, EU, V26, 4, P, cfg,
                                kernel=line_sgt.SGTKernel(2.0, 0.3, 1.0 / 1.3))
    assert_allclose(a, b[:, ::-1], rtol=0, atol=1e-10)
    assert float(np.max(np.abs(a - a[:, ::-1]))) > 1e-4      # the skew does act


def test_mirror_symmetry_zeta_one():
    """B49.9 / spec (vii): at zeta = 1 a tube through the line's origin gives a profile even in v [1e-10]."""
    cfg = cfg_dirs("partition_vertical")
    tube = geometry.Tube((0.0, 0.0, -0.1), 0.45, 0.0, 0.0, 1.0, 8.0, "axial")
    t = line_fast.line_profiles(tube, 1.0, [-1, 0, 1], O, YH, EU, V26, 4, P, cfg,
                                kernel=line_sgt.SGTKernel(1.4, 0.6, 1.0))
    assert_allclose(t, t[:, ::-1], rtol=0, atol=1e-10)


# ------------------------------------------------------------- invariants ---

def test_invariants_mu_zero_order_cache_determinism():
    """B49.9: mu = 0 gives 1 exactly; the order of ks only permutes the rows [1e-12]; a SectionCache reused across
    calls gives what fresh calls give [1e-12]; two calls are bit-identical. Hybrid, 2 x 4 directions."""
    cfg = cfg_dirs("hybrid")
    tube = geometry.Tube((0.0, 0.0, 0.0), 0.3, 0.3, 0.52, 1.0, 8.0, "axial")
    yh = np.array([-math.sin(0.52), math.cos(0.52)])
    eu = np.array([math.cos(0.52), math.sin(0.52)])
    kern = line_sgt.SGTKernel(1.6, 0.4, 1.2)
    one = line_fast.line_profiles(tube, 0.0, [-1, 0, 1], O, yh, eu, V26, 4, P, cfg, kernel=kern)
    assert one.shape == (3, V26.size) and one.dtype == np.float64
    assert np.all(one == 1.0)
    ks = [-2, -1, 0, 1, 2]
    perm = [1, -2, 2, 0, -1]
    a = line_fast.line_profiles(tube, 1.5, ks, O, yh, eu, V26, 4, P, cfg, kernel=kern)
    b = line_fast.line_profiles(tube, 1.5, perm, O, yh, eu, V26, 4, P, cfg, kernel=kern)
    assert_allclose(b, a[[ks.index(k) for k in perm]], rtol=0, atol=1e-12)
    cache = line_fast.SectionCache()
    c1 = line_fast.line_profiles(tube, 1.5, ks, O, yh, eu, V26, 4, P, cfg, cache=cache, kernel=kern)
    c2 = line_fast.line_profiles(tube, 1.5, ks, O, yh, eu, V26, 4, P, cfg, cache=cache, kernel=kern)
    assert_allclose(c1, a, rtol=0, atol=1e-12)
    assert_allclose(c2, a, rtol=0, atol=1e-12)
    a2 = line_fast.line_profiles(tube, 1.5, ks, O, yh, eu, V26, 4, P, cfg, kernel=kern)
    assert_array_equal(a, a2)
    assert np.all(np.isfinite(a)) and np.all(a <= 1.0 + 1e-12)


def _raises(fn):
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 -- the type is what is compared
        return type(exc), str(exc)
    return None, ""


def test_errors_as_block35_in_its_order():
    """B49.3, B49.10: a kernel that is neither None nor an SGTKernel is refused (ValueError); ExtrusionInvalid for a
    line 0.5 um from a cap with the SGT kernel as with the Gaussian; Block 35's other ValueErrors raised by the SGT
    path as by the Gaussian path, with the same type (mu < 0 near a cap is the argument error, not the extrusion);
    the SGT's own settings refused (a reach or a samples-per-FWHM out of range in a configuration built without
    validate)."""
    cfg = cfg_dirs("partition_vertical")
    tube = geometry.Tube((0.0, 0.0, 0.0), 0.3, 0.0, 0.0, 1.0, 8.0, "axial")
    k = line_sgt.SGTKernel(2.0, 0.5, 1.0)
    for bad in [(2.0, 0.0, 1.0), "sgt", object(), 2.0]:
        with pytest.raises(ValueError):
            line_fast.line_profiles(tube, 1.0, [0], O, YH, EU, V26, 0, P, cfg, kernel=bad)
    near = np.array([7.5, 0.0])                         # 0.5 um from the cap at x = 8
    cases = [
        dict(tube=tube, mu=1.0, o=near, yh=YH, absorption=None),            # extrusion invalid
        dict(tube=tube, mu=-1.0, o=near, yh=YH, absorption=None),           # mu < 0 first
        dict(tube=tube, mu=math.nan, o=O, yh=YH, absorption=None),
        dict(tube=tube, mu=1.0, o=O, yh=np.array([0.0, 2.0]), absorption=None),
        dict(tube=tube, mu=1.0, o=O, yh=YH, absorption="ray_world"),
        dict(tube=tube, mu=1.0, o=O, yh=YH, absorption="nonsense"),
    ]
    for c in cases:
        eg = _raises(lambda: line_fast.line_profiles(c["tube"], c["mu"], [0], c["o"], c["yh"], EU, V26, 0, P, cfg,
                                                     absorption=c["absorption"]))
        es = _raises(lambda: line_fast.line_profiles(c["tube"], c["mu"], [0], c["o"], c["yh"], EU, V26, 0, P, cfg,
                                                     absorption=c["absorption"], kernel=k))
        assert eg[0] is not None and es[0] is eg[0], (c, eg, es)
    assert _raises(lambda: line_fast.line_profiles(tube, 1.0, [0], near, YH, EU, V26, 0, P, cfg, kernel=k))[0] \
        is line_fast.ExtrusionInvalid
    assert _raises(lambda: line_fast.line_profiles(tube, -1.0, [0], near, YH, EU, V26, 0, P, cfg, kernel=k))[0] \
        is ValueError
    for repl in (dict(sgt_reach_sigmas=0.0), dict(sgt_reach_sigmas=math.nan), dict(sgt_samples_per_fwhm=0),
                 dict(sgt_samples_per_fwhm=2.5)):
        bad = dataclasses.replace(cfg, inference=dataclasses.replace(cfg.inference, **repl))
        with pytest.raises(ValueError):
            line_fast.line_profiles(tube, 1.0, [0], O, YH, EU, V26, 0, P, bad, kernel=k)


# ------------------------------------------------- smoothness (spec (v)) ---

def test_smoothness_in_the_shape():
    """B49.11 (spec (v)): the central differences of tau in log10(beta), xi and ln(zeta) at steps 1e-3 and 5e-4 agree
    within 1 % of the derivative's largest magnitude (no lattice or level switches with the shape). A tilted tube
    (0.3 rad, heading 0.52 rad, d 0.4 um), partition, planes -1..1, n_avg 4; bases (1.3, 0.2, 1) for beta and xi,
    (2, 0.3, 1.3) for zeta."""
    cfg = cfg_dirs("partition_vertical")
    tube = geometry.Tube((0.0, 0.0, 0.0), 0.2, 0.3, 0.52, 1.0, 8.0, "axial")
    yh = np.array([-math.sin(0.52), math.cos(0.52)])
    eu = np.array([math.cos(0.52), math.sin(0.52)])
    ks = [-1, 0, 1]

    def tau(b, xi, lz):
        return line_fast.line_profiles(tube, 1.5, ks, O, yh, eu, V26, 4, P, cfg,
                                       kernel=line_sgt.SGTKernel(10.0 ** b, xi, math.exp(lz)))

    for name, base in (("log10_beta", (math.log10(1.3), 0.2, 0.0)), ("xi", (math.log10(1.3), 0.2, 0.0)),
                       ("ln_zeta", (math.log10(2.0), 0.3, math.log(1.3)))):
        i = ("log10_beta", "xi", "ln_zeta").index(name)
        ds = []
        for h in (1e-3, 5e-4):
            up, dn = list(base), list(base)
            up[i] += h
            dn[i] -= h
            ds.append((tau(*up) - tau(*dn)) / (2.0 * h))
        big = float(np.max(np.abs(ds[0])))
        rel = float(np.max(np.abs(ds[0] - ds[1]))) / big
        print("d tau / d %s: largest %.3e, steps 1e-3 vs 5e-4 differ by %.2e of it" % (name, big, rel))
        assert big > 0 and rel <= 0.01


# ------------------------- spec (iii): an oblique heading against a brute-force reference ---

@pytest.mark.parametrize("shape", [(2.0, 0.0, 1.0), (2.0, 0.5, 1.0), (2.0, 0.3, 1.3), (2.0, 0.3, 0.77), (2.0, 1.0, 1.0),
                                   (2.0, 2.0, 1.0)])
@pytest.mark.parametrize("d", [0.4, 1.0])
def test_oblique_heading_against_brute_force(shape, d):
    """B49.5 extended (spec (iii)'s class of geometry: a tilted tube, heading 0.52 rad, so that y-hat is not a lattice
    axis and the skew's half-plane is oblique to the pixels): a constant law of SD 0.15 um, "linear" absorption, the
    default reach (12 law SDs), n_avg 4, v = n p (n = -26..26), against wave7_refs.brute_force_line -- each fine
    sample a point mass, the pixel-box integral of the stretched, truncated kernel (the bivariate t's closed form,
    beta = 2) by a 32 x 32 midpoint rule, one FFT convolution on the lattice, Block 5's bilinear read written
    independently (nc_helpers.bilinear_read). The reference's own quadrature error is 5e-6 at 32 x 32 (falling as
    sub^-2: 8.2e-5, 2.0e-5, 5.1e-6, 1.3e-6 at 8, 16, 32, 64, measured on the Gaussian path). Bound 5e-5."""
    sigma, th, tilt = 0.15, 0.52, 0.3
    cfg = cfg_const(sigma, 12.0)
    yh = np.array([-math.sin(th), math.cos(th)])
    eu = np.array([math.cos(th), math.sin(th)])
    tube = geometry.Tube((0.0, 0.0, 0.0), d / 2.0, tilt, th, 1.0, 12.0, "axial")
    tau = line_fast.line_profiles(tube, 1.3, [0], O, yh, eu, V26, 4, P, cfg, absorption="linear",
                                  kernel=line_sgt.SGTKernel(*shape))[0]
    ref = R.brute_force_line(d / 2.0, 1.3, tilt, th, (0.0, 0.0), V26, 4, P, P, 8, sigma, shape[1], shape[2], 12.0,
                             H.bilinear_read, sub=32)
    diff = float(np.max(np.abs(tau - ref)))
    print("shape %s d %.1f: max |SGT - brute force| %.2e (dip %.3f)" % (shape, d, diff, 1.0 - ref.min()))
    assert diff <= 5e-5


@pytest.mark.parametrize("shape", [(2.0, 0.5, 1.0), (2.0, 1.0, 1.0)])
def test_oblique_heading_spread_level(shape):
    """As test_oblique_heading_against_brute_force on Block 49's spread levels (a law SD of 0.42 um: FWHM 0.99 um,
    level 2 at line_fine_factor 8, the Gaussian spreading of the samples onto the coarse lattice); the reference's
    pixel-box rule 16 x 16 (the wider kernel is smoother at the pixel's scale: its error about (p/16)^2 K''/24, below
    3e-6 here). Bound 5e-5."""
    sigma, th, tilt, d = 0.42, 0.52, 0.3, 0.6
    cfg = cfg_const(sigma, 12.0)
    yh = np.array([-math.sin(th), math.cos(th)])
    eu = np.array([math.cos(th), math.sin(th)])
    tube = geometry.Tube((0.0, 0.0, 0.0), d / 2.0, tilt, th, 1.0, 14.0, "axial")
    tau = line_fast.line_profiles(tube, 1.3, [0], O, yh, eu, V26, 4, P, cfg, absorption="linear",
                                  kernel=line_sgt.SGTKernel(*shape))[0]
    ref = R.brute_force_line(d / 2.0, 1.3, tilt, th, (0.0, 0.0), V26, 4, P, P, 8, sigma, shape[1], shape[2], 12.0,
                             H.bilinear_read, sub=16)
    diff = float(np.max(np.abs(tau - ref)))
    print("spread level, shape %s: max |SGT - brute force| %.2e (dip %.3f)" % (shape, diff, 1.0 - ref.min()))
    assert diff <= 5e-5
