"""Spec-derived tests, Block 46 (the soma fit) with its amendment 1 (D-220) -- tester 2026-11-23.

Oracles (SPEC Block 46 "Mathematics", "Public API", T1-T8, amendment 1):
  amendment 1 (i)  start_theta reads start_far_radii, start_core_radii, start_core_percentile, start_ratio_clip and
                   start_ratio_none from scfg (None: the defaults) -- against the start written out from the spec with
                   numpy [1e-12]; fit_soma passes its own scfg (the first point the forward sees is the start clipped
                   into the bounds);
  amendment 1 (ii) varrho = start_ratio_none when the core holds no region pixel: mu_0 = -ln(start_ratio_none)/(2 r_s);
  amendment 1 (iii) select_planes keeps the planes of Block 44's closed window (|k dz - z| <= half with a tolerance of
                   1e-9 of a plane): the edge planes at exact multiples are kept (exact rationals as the reference);
  amendment 1 (iv) under a continuation other than "linear", fit_soma with Block 45's renderer raises ValueError at the
                   first render, before the solver;
  T1 ellipsoid_area against closed forms and dblquad [1e-8 relative], permutations [1e-12];
  T2 select_planes on a window with invalid planes, steps 1-3; the ValueErrors;
  T3 vp_scale: D = B tau gives B [1e-12 relative] and a zero residual;
  SE: cov = (J^T J)^-1 chi2 / dof with J the forward differences of the projected residual at theta_hat (absolute steps
      diff_steps), dof = |K||R| - n_fitted - |K| -- against an independent J [1e-6 relative];
  T4(a) noise-free recovery with an injected, identifiable stand-in forward model [1e-4 in each unit];
  T5 far_slope "fixed": theta[8] = slope0 exactly, se[8] NaN, eight names; T6 masked pixels do not enter;
  T7 a truth beyond a bound puts that parameter in at_bound; T8 a NaN forward gives "failed" without a raise.
Boundary Block 45 -> Block 46/44: the renderer's pixel (row, col) is centred on ((left + col) p, (top + row) p), the
convention of the fit's region and of Block 44's disk_mask (a sphere centred on a pixel renders symmetric about it).
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import soma_spec_helpers as H
from allen_diameter.analysis import soma_fit as SF
from allen_diameter.analysis import soma_view as SV
from allen_diameter.config import SomaFitConfig, default_config
from allen_diameter.model.soma_render import SomaRenderer

P = 0.25          # um: a coarse pixel for the stand-in (speed)
DZ = 0.28


# --------------------------------------------------------- the stand-in model --

def _softplus(x, w=0.5):
    return w * np.logaddexp(0.0, x / w)


def standin_factory(crop, p):
    """An injected forward model of the same parameters, smooth and with every parameter identifiable (T4): a Gaussian
    blob of in-plane semi-widths (a, b) at phi, darkness 1 - exp(-2 mu c), whose spread grows as 1 + (gamma s_k)^2 with
    s_k the smoothed distance from the plane to the soma's nearest surface along z (c enters there), light-conserving."""
    left, top, W, Hh = crop
    x = (left + np.arange(W)) * p
    y = (top + np.arange(Hh)) * p
    X, Y = np.meshgrid(x, y)

    def factory(z_planes):
        z = np.asarray(z_planes, dtype=float)

        def forward(th, slope):
            x0, y0, z0, a, b, c, phi, mu = (float(t) for t in th)
            u = (X - x0) * math.cos(phi) + (Y - y0) * math.sin(phi)
            v = -(X - x0) * math.sin(phi) + (Y - y0) * math.cos(phi)
            r2 = (u / a) ** 2 + (v / b) ** 2
            out = np.empty((z.size,) + X.shape)
            amp0 = 1.0 - math.exp(-2.0 * mu * c)
            for k, zk in enumerate(z):
                s = _softplus(math.sqrt((zk - z0) ** 2 + 0.25) - c)
                g = 1.0 + (float(slope) * s) ** 2
                out[k] = 1.0 - (amp0 / g) * np.exp(-0.5 * r2 / g)
            return out
        return forward
    return factory


TRUTH = np.array([0.4, -0.3, 0.35, 4.2, 3.0, 2.6, math.radians(46.0), 0.25, 0.9])   # a/b = 1.4, phi 46 deg


def _data(truth=TRUTH, noise=0.0, seed=0, crop=(-40, -40, 81, 81), ks=np.arange(-20, 21)):
    fwd = standin_factory(crop, P)(ks * DZ)
    tau = fwd(truth[:8], truth[8])
    B = 100.0 + 2.0 * np.arange(ks.size)
    D = B[:, None, None] * tau
    if noise:
        D = D + np.random.default_rng(seed).normal(0.0, noise, D.shape)
    return D, ks


def _scfg(**kw):
    # [2026-11-24, D-240] far_slope "fit": the fitted-slope path, an opt-in comparison since D-240 ("fixed" the default)
    base = dict(plane_step=2, plane_half_um=5.0, region_radius_um=9.0, ftol=1e-10, xtol=1e-10, max_nfev=200,
                far_slope="fit")
    base.update(kw)
    return dataclasses.replace(SomaFitConfig(), **base)


def _fit(D, ks, scfg, mask=None, centre=(-0.6, 0.7, -0.65), radius=3.0 * 1.15, slope0=0.7, factory=None,
         crop=(-40, -40, 81, 81)):
    left, top, W, Hh = crop
    m = np.zeros((Hh, W), bool) if mask is None else mask
    return SF.fit_soma(D, ks, np.ones(ks.size, bool), left, top, P, DZ, centre, radius, m,
                       factory or standin_factory(crop, P), scfg, slope0)


def _projected_residual(D, ks_sel, region, theta, crop=(-40, -40, 81, 81)):
    """Variable projection written out: per plane B = sum D tau / sum tau^2 over R, r = D - B tau (C order of R)."""
    tau = standin_factory(crop, P)(ks_sel * DZ)(theta[:8], theta[8])
    out = []
    for k in range(ks_sel.size):
        d, t = D[k][region], tau[k][region]
        Bk = float(np.dot(d, t) / np.dot(t, t))
        out.append(d - Bk * t)
    return np.concatenate(out)


# ------------------------------------------------------------- amendment 1 --

def _start_plane(seed=3):
    crop = (-40, -40, 81, 81)
    fwd = standin_factory(crop, P)(np.array([0.0]))
    tau = fwd(np.array([0.4, -0.3, 0.0, 4.2, 3.0, 2.6, math.radians(46), 0.25]), 0.9)[0]
    plane = 120.0 * tau + np.random.default_rng(seed).normal(0, 1.0, tau.shape)
    x = (crop[0] + np.arange(81)) * P
    X, Y = np.meshgrid(x, x)
    region = (X ** 2 + Y ** 2 <= 9.0 ** 2) & ~((X > 4.0) & (np.abs(Y) < 1.0))   # a disk minus a masked stem
    return plane, region, crop


@pytest.mark.parametrize("fields", [
    {},
    dict(start_far_radii=2.0, start_core_radii=0.3, start_core_percentile=20.0, start_ratio_clip=(0.2, 0.6),
         start_ratio_none=0.3),
    dict(start_far_radii=1.2, start_core_radii=0.8, start_core_percentile=50.0, start_ratio_clip=(0.01, 0.99)),
])
def test_start_theta_reads_its_settings(fields):
    plane, region, crop = _start_plane()
    q = dataclasses.replace(SomaFitConfig(), **fields)
    centre, rs = (-0.2, 0.3, 0.1), 3.1
    got = SF.start_theta(plane, crop[0], crop[1], P, centre, rs, region, 0.79, q)
    exp = H.ref_start_theta(plane, crop[0], crop[1], P, centre, rs, region, 0.79, far_r=q.start_far_radii,
                            core_r=q.start_core_radii, pct=q.start_core_percentile, clip=q.start_ratio_clip,
                            none=q.start_ratio_none)
    assert got.shape == (9,) and got.dtype == np.float64
    assert_allclose(got, exp, rtol=1e-12, atol=1e-12)
    if not fields:
        assert_array_equal(SF.start_theta(plane, crop[0], crop[1], P, centre, rs, region, 0.79), got)


def test_start_ratio_none_when_core_empty():
    plane, region, crop = _start_plane()
    x = (crop[0] + np.arange(81)) * P
    X, Y = np.meshgrid(x, x)
    rs = 3.1
    hole = region & ~(np.hypot(X + 0.2, Y - 0.3) <= 0.5 * rs + 1e-9)       # no region pixel in the core
    q = dataclasses.replace(SomaFitConfig(), start_ratio_none=0.37)
    th = SF.start_theta(plane, crop[0], crop[1], P, (-0.2, 0.3, 0.1), rs, hole, 0.79, q)
    assert_allclose(th[7], -math.log(0.37) / (2 * rs), rtol=1e-15)


def test_fit_soma_starts_from_its_own_scfg_start():
    D, ks = _data()
    seen = []
    base = standin_factory((-40, -40, 81, 81), P)

    def factory(z):
        f = base(z)

        def fwd(th, slope):
            seen.append(np.append(np.asarray(th, float), slope))
            return f(th, slope)
        return fwd
    q = _scfg(start_core_percentile=30.0, start_ratio_clip=(0.3, 0.9), start_far_radii=1.8, max_nfev=1)
    SF.fit_soma(D, ks, np.ones(ks.size, bool), -40, -40, P, DZ, (-0.6, 0.7, -0.65), 3.45,
                np.zeros((81, 81), bool), factory, q, 0.7)
    # the start on plane k_0 (the valid plane nearest z_s), on the region (the disk, nothing masked), clipped
    k0 = int(np.argmin(np.abs(ks * DZ - (-0.65))))
    x = (-40 + np.arange(81)) * P
    X, Y = np.meshgrid(x, x)
    region = (X + 0.6) ** 2 + (Y - 0.7) ** 2 <= q.region_radius_um ** 2
    exp = H.ref_start_theta(D[k0], -40, -40, P, (-0.6, 0.7, -0.65), 3.45, region, 0.7, far_r=1.8, core_r=0.5,
                            pct=30.0, clip=(0.3, 0.9), none=0.5)
    lo = np.array([-0.6 - 4, 0.7 - 4, -0.65 - 6, 1, 1, 1, -math.pi, 0.001, 0.2])
    hi = np.array([-0.6 + 4, 0.7 + 4, -0.65 + 6, 15, 15, 15, math.pi, 3.0, 3.0])
    assert_allclose(seen[0], np.clip(exp, lo, hi), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("z, half, dz", [(0.0, 10.08, 0.28), (0.0, 2.8, 0.28), (0.1, 2.8, 0.28), (-3.36, 1.12, 0.28),
                                         (5.6, 0.0, 0.28), (1.7, 3.3, 0.3), (0.84, 0.84, 0.28)])
def test_select_planes_window_is_block44s(z, half, dz):
    ks = np.arange(-60, 61)
    got = ks[SF.select_planes(ks, np.ones(ks.size, bool), z, dz, half, 1)] if \
        len(H.ref_window(z, half, dz, ks)) >= 3 else None
    exp = H.ref_window(z, half, dz, ks)
    if got is None:
        with pytest.raises(ValueError):
            SF.select_planes(ks, np.ones(ks.size, bool), z, dz, half, 1)
    else:
        assert_array_equal(got, exp)
        assert (exp[0], exp[-1]) == SV.plane_range(z, half, dz)


def test_select_planes_steps_and_invalid():
    ks = np.arange(100, 141)
    valid = np.ones(ks.size, bool)
    valid[[19, 20, 23]] = False                      # k = 119, 120, 123 not recorded
    z = 120 * 0.28 + 0.01                            # nearest valid: 119 and 120 invalid -> 121 (0.27) vs 118 (0.57)
    for step in (1, 2, 3):
        got = ks[SF.select_planes(ks, valid, z, 0.28, 3.0, step)]
        k0 = 121
        exp = [k for k, ok in zip(ks, valid) if ok and abs(k * 0.28 - z) <= 3.0 and (k - k0) % step == 0]
        assert_array_equal(got, exp)
    with pytest.raises(ValueError):
        SF.select_planes(ks, valid, z, 0.28, 0.2, 1)        # fewer than 3
    with pytest.raises(ValueError):
        SF.select_planes(ks, valid, z, 0.28, 3.0, 0)
    with pytest.raises(ValueError):
        SF.select_planes(ks, valid, z, 0.28, -0.1, 1)


# ----------------------------------------------------------------------- T1 --

@pytest.mark.parametrize("axes", [(3.0, 3.0, 3.0), (5.0, 2.0, 2.0), (4.0, 4.0, 1.5), (5.0, 3.5, 2.0),
                                  (7.3, 1.1, 4.4)])
def test_t1_area(axes):
    exp = H.ref_area(*axes)
    got = SF.ellipsoid_area(*axes)
    assert_allclose(got, exp, rtol=1e-8)
    import itertools
    for perm in itertools.permutations(axes):
        assert_allclose(SF.ellipsoid_area(*perm), got, rtol=1e-12)
    for bad in ((0.0, 1, 1), (1, -1, 1), (1, 1, np.nan), (np.inf, 1, 1)):
        with pytest.raises(ValueError):
            SF.ellipsoid_area(*bad)


# ----------------------------------------------------------------------- T3 --

def test_t3_vp_scale_exact():
    rng = np.random.default_rng(5)
    tau = rng.uniform(0.2, 1.0, (4, 9, 11))
    B = np.array([90.0, 101.5, 120.25, 77.0])
    region = rng.uniform(size=(9, 11)) < 0.6
    Bh, r = SF.vp_scale(B[:, None, None] * tau, tau, region)
    assert_allclose(Bh, B, rtol=1e-12)
    assert r.shape == (4, int(region.sum())) and np.max(np.abs(r)) <= 1e-9


# -------------------------------------------------------- T4(a), SE, T5-T8 --

def test_t4a_noise_free_recovery_and_shapes():
    D, ks = _data()
    fit = _fit(D, ks, _scfg())
    assert fit.status == "converged", fit.message
    assert fit.theta.shape == (9,) and fit.se.shape == (9,) and fit.cov.shape == (9, 9)
    assert fit.fitted == SF.PARAMS
    assert_allclose(fit.theta, TRUTH, rtol=0, atol=1e-4)
    assert_allclose(fit.B_gl, 100.0 + 2.0 * np.arange(ks.size)[np.isin(ks, fit.ks)], rtol=1e-6)
    a, b, c = TRUTH[3:6]
    assert_allclose(fit.area_um2, H.ref_area(*fit.theta[3:6]), rtol=1e-8)
    assert_allclose(fit.volume_um3, 4 / 3 * math.pi * np.prod(fit.theta[3:6]), rtol=1e-12)
    assert_allclose(fit.r_eq_um, math.sqrt(fit.area_um2 / (4 * math.pi)), rtol=1e-12)
    del a, b, c


def test_se_formula_against_independent_jacobian():
    D, ks = _data(noise=1.0, seed=11)
    q = _scfg(ftol=1e-8, xtol=1e-8)
    fit = _fit(D, ks, q)
    assert fit.status == "converged", fit.message
    idx = np.isin(ks, fit.ks)
    Ds = D[idx]
    x = (-40 + np.arange(81)) * P
    X, Y = np.meshgrid(x, x)
    region = (X + 0.6) ** 2 + (Y - 0.7) ** 2 <= q.region_radius_um ** 2
    th = fit.theta
    r0 = _projected_residual(Ds, fit.ks, region, th)
    assert_allclose(fit.chi2, float(np.dot(r0, r0)), rtol=1e-9)
    K, nR = int(fit.ks.size), int(region.sum())
    assert fit.n_samples == K * nR and fit.dof == K * nR - 9 - K
    hi = np.array([-0.6 + 4, 0.7 + 4, -0.65 + 6, 15, 15, 15, math.pi, 3.0, 3.0])
    J = np.empty((r0.size, 9))
    for j in range(9):
        h = q.diff_steps[j]
        tp = th.copy()
        if th[j] + h <= hi[j]:
            tp[j] = th[j] + h
            J[:, j] = (_projected_residual(Ds, fit.ks, region, tp) - r0) / (tp[j] - th[j])
        else:
            tp[j] = th[j] - h
            J[:, j] = (r0 - _projected_residual(Ds, fit.ks, region, tp)) / (th[j] - tp[j])
    cov = np.linalg.inv(J.T @ J) * fit.chi2 / fit.dof
    # the solver's final Jacobian is taken at the returned point: one matrix, up to the round-off of the differences
    assert_allclose(fit.cov, cov, rtol=1e-6, atol=0)
    assert_allclose(fit.se, np.sqrt(np.diag(cov)), rtol=1e-6)
    # SE(S) by the delta method on the (a, b, c) block with a central-difference gradient
    g = np.array([(H.ref_area(*(th[3:6] + e)) - H.ref_area(*(th[3:6] - e))) / (2 * e.sum())
                  for e in np.eye(3) * 1e-5 * th[3:6]])
    assert_allclose(fit.se_area_um2, math.sqrt(g @ cov[3:6, 3:6] @ g), rtol=1e-4)


def test_t5_fixed_slope():
    D, ks = _data(truth=np.r_[TRUTH[:8], 0.7])
    fit = _fit(D, ks, _scfg(far_slope="fixed"), slope0=0.7)
    assert fit.theta[8] == 0.7 and math.isnan(fit.se[8])
    assert np.all(np.isnan(fit.cov[8])) and np.all(np.isnan(fit.cov[:, 8]))
    assert fit.fitted == SF.PARAMS[:8]
    assert_allclose(fit.theta[:8], TRUTH[:8], atol=1e-4)


def test_t6_masked_pixels_do_not_enter():
    D, ks = _data(noise=1.0, seed=2)
    x = (-40 + np.arange(81)) * P
    X, Y = np.meshgrid(x, x)
    mask = (X > 3.0) & (np.abs(Y - 1.0) < 0.8)
    q = _scfg(ftol=1e-8, xtol=1e-8)
    f1 = _fit(D, ks, q, mask=mask)
    D2 = D.copy()
    D2[:, mask] = 0.0                                   # a black stem on the masked pixels only
    f2 = _fit(D2, ks, q, mask=mask)
    assert_array_equal(f2.theta, f1.theta)


def test_t7_truth_beyond_a_bound():
    truth = TRUTH.copy()
    truth[0] = -0.6 + 5.0                               # x0 beyond the traced centre's x + shift_xy_um (4 um)
    D, ks = _data(truth=truth)
    fit = _fit(D, ks, _scfg(ftol=1e-8, xtol=1e-8))
    assert "x0_um" in fit.at_bound, (fit.at_bound, fit.theta)
    assert_allclose(fit.theta[0], -0.6 + 4.0, atol=4e-6 * 8.0)    # within 1e-6 of the bound interval's width


def test_t8_nan_forward_failed_without_raise():
    D, ks = _data()

    def factory(z):
        return lambda th, s: np.full((np.size(z), 81, 81), np.nan)
    fit = _fit(D, ks, _scfg(), factory=factory)
    assert fit.status == "failed"
    assert np.all(np.isnan(fit.theta)) and np.all(np.isnan(fit.se)) and math.isnan(fit.area_um2)
    assert fit.at_bound == ()


# ---------------------------------------- amendment 1 (iv) and Block 45's grid --

def test_nonlinear_continuation_raises_at_first_render():
    cfg = default_config()
    rcfg = dataclasses.replace(cfg.renderer, kernel_continuation="frozen")
    q = dataclasses.replace(cfg.soma_fit, plane_half_um=1.0, plane_step=1, region_radius_um=2.0)
    crop = (-20, -20, 41, 41)
    calls = []

    def factory(z):
        r = SomaRenderer(crop, 0.1144, z, rcfg, q, +1)

        def fwd(th, slope):
            calls.append(slope)
            return r.render(th, slope)
        return fwd
    ks = np.arange(-5, 6)
    D = np.full((ks.size, 41, 41), 100.0)
    with pytest.raises(ValueError):
        SF.fit_soma(D, ks, np.ones(ks.size, bool), -20, -20, 0.1144, 0.28, (0.0, 0.0, 0.0), 1.5,
                    np.zeros((41, 41), bool), factory, q, 0.79)
    assert len(calls) == 1                              # the first render, before the solver


@pytest.mark.parametrize("f", [1, 2])
def test_block45_pixel_centres_match_the_region_convention(f):
    cfg = default_config()
    q = dataclasses.replace(cfg.soma_fit, fine_factor=f)
    p, left, top, W, Hh = 0.1144, -30, -25, 61, 51
    r0, c0 = 25, 30                                     # the pixel at (row 25, col 30) is centred on (0, 0)
    assert (left + c0) * p == 0.0 and (top + r0) * p == 0.0
    tau = SomaRenderer((left, top, W, Hh), p, np.array([0.0, 0.6]), cfg.renderer, q, +1).render(
        np.array([0.0, 0.0, 0.0, 1.5, 1.5, 1.5, 0.0, 0.4]))
    for k in range(2):
        t = tau[k]
        assert np.unravel_index(np.argmin(t), t.shape) == (r0, c0)
        core = t[r0 - 20:r0 + 21, c0 - 20:c0 + 21]
        assert_allclose(core, core[::-1, ::-1], rtol=0, atol=1e-12)   # point-symmetric about that pixel
        assert_allclose(core, core.T, rtol=0, atol=1e-12)
    # Block 44's disk about (0, 0) is symmetric about the same pixel
    disk = SV.disk_mask((Hh, W), left, top, p, (0.0, 0.0), 1.0)
    assert disk[r0, c0] and np.array_equal(disk[r0 - 20:r0 + 21, c0 - 20:c0 + 21],
                                           disk[r0 - 20:r0 + 21, c0 - 20:c0 + 21][::-1, ::-1])
