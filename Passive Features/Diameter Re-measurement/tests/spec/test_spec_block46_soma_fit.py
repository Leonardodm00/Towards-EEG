"""Spec-derived tests of Block 46 (the soma fit: an ellipsoid fitted to the planes about the soma; D-215, D-216,
D-218, D-219) and of its boundaries with Blocks 44 and 45. Tester, 2026-11-22 (incremental run at 97f5104).

Independent references, written from SPEC.md's text:
  ref_area          the ellipsoid's area by Carlson's symmetric integral, 4 pi abc R_G(a^-2, b^-2, c^-2)
                    (scipy.special.elliprg), and by scipy.integrate.dblquad of the parametric area element
  ref_select        select_planes' rule written out
  ref_start         start_theta's rule written out (B_far, varrho, the weighted moments)
  ToyForward        an injected forward model of the same nine parameters, smooth and identifiable (SPEC T4): the
                    columns' absorption 1 - exp(-mu L) of a (softened) ellipsoid, blurred by a Gaussian whose SD grows
                    at gamma with the distance from the plane to the soma's nearest surface along z
  ref_resid         the variable projection of the spec (B_k = sum D tau / sum tau^2) on the region, for chi2, B_gl
                    and an independent central-difference Jacobian (the SE check)

    python3 -m pytest -q -p no:cacheprovider tests/spec/test_spec_block46_soma_fit.py

Pure ASCII.
"""
from __future__ import annotations

import dataclasses
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import fft as sfft
from scipy import integrate, special

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
for _p in (WS / "src", WS / "scripts", WS / "tests" / "smoke", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from allen_diameter import config as C                           # noqa: E402
from allen_diameter.analysis import soma_fit as SF               # noqa: E402

NAMES = ("x0_um", "y0_um", "z0_um", "a_um", "b_um", "c_um", "phi_rad", "mu_per_um", "slope_um_per_um")


# ------------------------------------------------------------------ area ---

def ref_area(a, b, c):
    return 4 * math.pi * a * b * c * float(special.elliprg(1 / a ** 2, 1 / b ** 2, 1 / c ** 2))


def dblquad_area(a, b, c):
    def dS(v, u):           # r(u, v) = (a cos u sin v, b sin u sin v, c cos v): |r_u x r_v|
        su, cu, sv, cv = math.sin(u), math.cos(u), math.sin(v), math.cos(v)
        return sv * math.sqrt((b * c * cu * sv) ** 2 + (a * c * su * sv) ** 2 + (a * b * cv) ** 2)
    val, _ = integrate.dblquad(dS, 0, 2 * math.pi, 0, math.pi, epsabs=0, epsrel=1e-12)
    return val


def test_area_closed_forms_and_references():
    r = 2.7
    assert abs(SF.ellipsoid_area(r, r, r) - 4 * math.pi * r * r) <= 1e-12 * 4 * math.pi * r * r
    # oblate a = b > c: 2 pi a^2 (1 + (1 - e^2)/e artanh e), e^2 = 1 - c^2/a^2
    a, c = 5.0, 2.0
    e = math.sqrt(1 - c * c / (a * a))
    oblate = 2 * math.pi * a * a * (1 + (1 - e * e) / e * math.atanh(e))
    assert abs(SF.ellipsoid_area(a, a, c) - oblate) <= 1e-12 * oblate
    # prolate a > b = c: 2 pi b^2 (1 + a/(b e) arcsin e), e^2 = 1 - b^2/a^2
    a, b = 5.0, 2.0
    e = math.sqrt(1 - b * b / (a * a))
    prolate = 2 * math.pi * b * b * (1 + a / (b * e) * math.asin(e))
    assert abs(SF.ellipsoid_area(a, b, b) - prolate) <= 1e-12 * prolate
    # general ellipsoids: Carlson [1e-12 rel] and dblquad of the area element (epsrel 1e-12) [1e-8 rel]
    rng = np.random.default_rng(5)
    for _ in range(12):
        abc = rng.uniform(0.5, 12.0, 3)
        S = SF.ellipsoid_area(*abc)
        assert abs(S - ref_area(*abc)) <= 1e-12 * S
    for abc in ((4.3, 3.9, 3.5), (9.0, 2.0, 1.0), (1.0, 1.1, 7.0)):
        S = SF.ellipsoid_area(*abc)
        assert abs(S - dblquad_area(*abc)) <= 1e-8 * S
    # symmetric in its arguments [1e-12 rel], continuous with the sphere and the spheroids [1e-8 rel]
    abc = (4.3, 3.9, 3.5)
    S = SF.ellipsoid_area(*abc)
    import itertools
    for perm in itertools.permutations(abc):
        assert abs(SF.ellipsoid_area(*perm) - S) <= 1e-12 * S
    for eps in (1e-9, 1e-7):
        for near in ((r * (1 + eps), r, r), (r, r * (1 + eps), r * (1 - eps)), (5.0, 5.0 * (1 + eps), 2.0),
                     (5.0, 2.0 * (1 + eps), 2.0)):
            assert abs(SF.ellipsoid_area(*near) - ref_area(*near)) <= 1e-8 * ref_area(*near)
    for bad in ((0, 1, 1), (-1, 1, 1), (1, np.nan, 1), (1, 1, np.inf)):
        with pytest.raises(ValueError):
            SF.ellipsoid_area(*bad)


# ---------------------------------------------------------------- planes ---

def ref_select(ks, valid, z, dz, half, step):
    ks = np.asarray(ks)
    vk = [i for i in range(ks.size) if valid[i]]
    d = [abs(ks[i] * dz - z) for i in vk]
    best = min(d)
    k0 = min(ks[i] for i, di in zip(vk, d) if di == best)          # ties: the lower k
    return [i for i in range(ks.size) if valid[i] and abs(ks[i] * dz - z) <= half and (ks[i] - k0) % step == 0]


def test_select_planes_rule():
    ks = np.arange(40, 90)
    rng = np.random.default_rng(2)
    for trial in range(60):
        valid = rng.uniform(size=ks.size) > 0.25
        z = rng.uniform(50, 80) * 0.25
        half = rng.uniform(0.5, 6.0)
        step = int(rng.integers(1, 4))
        want = ref_select(ks, valid, z, 0.25, half, step)
        if len(want) < 3:
            with pytest.raises(ValueError):
                SF.select_planes(ks, valid, z, 0.25, half, step)
            continue
        got = SF.select_planes(ks, valid, z, 0.25, half, step)
        assert list(got) == want, (trial, list(got), want)
    # a tie for k_0 (z exactly between two valid planes, dyadic): the lower k
    ks2 = np.arange(0, 20)
    got = SF.select_planes(ks2, np.ones(20, bool), 10.5 * 0.25, 0.25, 2.0, 2)
    assert list(ks2[got]) == [4, 6, 8, 10, 12, 14, 16, 18]  # k_0 = 10 (not 11); |k dz - z| <= 2 um: k in 3..18
    for bad in (dict(step=0), dict(step=1.5), dict(half_um=-0.1)):
        kw = dict(ks=ks2, valid=np.ones(20, bool), z_start_um=2.5, dz_um=0.25, half_um=2.0, step=1)
        kw.update(bad)
        with pytest.raises(ValueError):
            SF.select_planes(**kw)


def test_vp_scale_exact_and_lstsq():
    rng = np.random.default_rng(3)
    tau = rng.uniform(0.3, 1.0, (4, 9, 11))
    region = rng.uniform(size=(9, 11)) > 0.4
    Btrue = np.array([90.0, 110.0, 130.0, 70.0])
    B, res = SF.vp_scale(Btrue[:, None, None] * tau, tau, region)
    assert_allclose(B, Btrue, rtol=1e-12, atol=0)
    assert res.shape == (4, int(region.sum())) and np.max(np.abs(res)) <= 1e-9
    D = rng.normal(100, 10, tau.shape)
    B, res = SF.vp_scale(D, tau, region)
    for k in range(4):
        sol = np.linalg.lstsq(tau[k][region][:, None], D[k][region], rcond=None)[0][0]
        assert abs(B[k] - sol) <= 1e-12 * abs(sol)
        assert_allclose(res[k], D[k][region] - sol * tau[k][region], atol=1e-9, rtol=0)
    with pytest.raises(ValueError):
        SF.vp_scale(D, tau, np.zeros((9, 11), bool))


# ----------------------------------------------------------------- start ---

def ref_start(P, left, top, p, centre, rs, region, slope0):
    H, W = P.shape
    x = (left + np.arange(W)) * p
    y = (top + np.arange(H)) * p
    X, Y = np.meshgrid(x, y)
    X, Y, V = X[region], Y[region], P[region]
    rho = np.hypot(X - centre[0], Y - centre[1])
    far = rho > 1.5 * rs
    Bf = np.median(V[far]) if far.any() else np.median(V)
    core = rho <= 0.5 * rs
    vr = np.clip(np.percentile(V[core], 5) / Bf, 0.05, 0.95) if core.any() else 0.5
    mu0 = -math.log(vr) / (2 * rs)
    near = rho <= 1.5 * rs
    w = np.maximum(0, 1 - V[near] / Bf)
    a0 = b0 = rs
    phi0 = 0.0
    if w.sum() > 0:
        xm, ym = np.sum(w * X[near]) / w.sum(), np.sum(w * Y[near]) / w.sum()
        cxx = np.sum(w * (X[near] - xm) ** 2) / w.sum()
        cyy = np.sum(w * (Y[near] - ym) ** 2) / w.sum()
        cxy = np.sum(w * (X[near] - xm) * (Y[near] - ym)) / w.sum()
        tr, det = cxx + cyy, cxx * cyy - cxy * cxy
        l1 = tr / 2 + math.sqrt(max(tr * tr / 4 - det, 0))
        l2 = tr / 2 - math.sqrt(max(tr * tr / 4 - det, 0))
        if l2 > 0:
            phi0 = 0.5 * math.atan2(2 * cxy, cxx - cyy)
            a0, b0 = rs * (l1 / l2) ** 0.25, rs * (l2 / l1) ** 0.25
    return np.array([centre[0], centre[1], centre[2], a0, b0, rs, phi0, mu0, slope0])


def _shadow_plane(shape, left, top, p, x0, y0, a, b, phi, depth, B=120.0):
    H, W = shape
    X, Y = np.meshgrid((left + np.arange(W)) * p, (top + np.arange(H)) * p)
    u = (X - x0) * math.cos(phi) + (Y - y0) * math.sin(phi)
    v = -(X - x0) * math.sin(phi) + (Y - y0) * math.cos(phi)
    q = 1 - (u / a) ** 2 - (v / b) ** 2
    return B * (1 - depth * np.sqrt(np.clip(q, 0, None)))


def test_start_theta_rule():
    shape, left, top, p = (64, 70), -35, -32, 0.2
    img = _shadow_plane(shape, left, top, p, 0.4, -0.3, 3.0, 1.8, 0.7, 0.6)
    rng = np.random.default_rng(1)
    img = img + rng.normal(0, 0.5, shape)
    X, Y = np.meshgrid((left + np.arange(70)) * p, (top + np.arange(64)) * p)
    region = (X ** 2 + Y ** 2 <= 6.0 ** 2) & ~((X > 3.0) & (np.abs(Y) < 0.5))      # a masked stem
    for centre, rs in (((0.0, 0.0, 1.0), 2.3), ((0.5, -0.5, 0.0), 1.1), ((0.0, 0.0, 0.0), 5.5)):
        got = SF.start_theta(img, left, top, p, centre, rs, region, 0.79)
        ref = ref_start(img, left, top, p, centre, rs, region, 0.79)
        assert_allclose(got, ref, rtol=1e-12, atol=1e-12)
    th = SF.start_theta(img, left, top, p, (0.0, 0.0, 1.0), 2.3, region, 0.79)
    assert abs(th[6] - 0.7) < 0.1 and th[3] > th[4]          # the shadow's orientation and elongation
    assert abs(math.sqrt(th[3] * th[4]) - 2.3) <= 1e-12        # an aspect of the same mean radius
    # no pixel in the core: varrho = 0.5, mu0 = ln 2 / (2 r_s); a flat plane: a0 = b0 = r_s, phi0 = 0
    reg2 = region & ((X ** 2 + Y ** 2) > 1.0)
    th2 = SF.start_theta(img, left, top, p, (0.0, 0.0, 0.0), 1.9, reg2, 0.5)
    assert abs(th2[7] - math.log(2) / (2 * 1.9)) <= 1e-15
    flat = np.full(shape, 100.0)
    th3 = SF.start_theta(flat, left, top, p, (0.0, 0.0, 0.0), 2.0, region, 0.5)
    assert (th3[3], th3[4], th3[6]) == (2.0, 2.0, 0.0)


# ------------------------------------------------------- the toy forward ---

class ToyForward:
    """SPEC T4's injected forward model: tau_k = 1 - G_{s_k} (*) (1 - exp(-mu L)), L = 2 c sqrt(softplus(q)) the
    column chord of a slightly softened ellipsoid (smooth in every parameter), s_k = s0 + gamma sp(|z_k - z0|~ - c)
    the blur growing with the distance from the plane to the soma's nearest surface along z (|.|~ and sp smooth),
    the Gaussian by its continuous transfer on a zero-padded box."""

    def __init__(self, crop, p, z, s0=0.35, pad=64):
        left, top, W, H = crop
        self.X, self.Y = np.meshgrid((left + np.arange(W)) * p, (top + np.arange(H)) * p)
        self.z = np.asarray(z, dtype=float)
        self.H, self.W, self.s0 = H, W, s0
        self.N = (H + pad, W + pad)
        fy = sfft.fftfreq(self.N[0], d=p)
        fx = sfft.rfftfreq(self.N[1], d=p)
        self.nu2 = fy[:, None] ** 2 + fx[None, :] ** 2
        self.calls = 0

    def __call__(self, th8, slope):
        self.calls += 1
        x0, y0, z0, a, b, c, phi, mu = (float(t) for t in th8)
        u = (self.X - x0) * math.cos(phi) + (self.Y - y0) * math.sin(phi)
        v = -(self.X - x0) * math.sin(phi) + (self.Y - y0) * math.cos(phi)
        q = 1 - (u / a) ** 2 - (v / b) ** 2
        eps = 0.02
        sq = np.sqrt(eps * np.logaddexp(0.0, q / eps))
        A = 1 - np.exp(-mu * 2 * c * sq)
        FA = sfft.rfft2(A, s=self.N)
        out = np.empty((self.z.size, self.H, self.W))
        for k, zk in enumerate(self.z):
            d = math.sqrt((zk - z0) ** 2 + 0.25) - c
            s = self.s0 + float(slope) * 0.5 * math.log1p(math.exp(2 * d)) if d < 30 else self.s0 + float(slope) * d
            out[k] = 1 - sfft.irfft2(FA * np.exp(-2 * math.pi ** 2 * s * s * self.nu2), s=self.N)[:self.H, :self.W]
        return out


DZ_TOY = 0.5
P_TOY = 0.25
CROP = (-28, -28, 57, 57)                       # 14 um square about (0, 0)
KS = np.arange(-16, 17)                         # planes at k * 0.5 um
TRUTH = np.array([0.3, -0.2, 0.4, 3.0, 2.2, 2.0, 0.78, 0.35, 0.9])     # a/b = 1.36, phi = 44.7 deg
B_TRUE = 100.0 + 2.0 * np.sin(KS)


def toy_scfg(**kw):
    base = dict(plane_half_um=5.0, plane_step=2, region_radius_um=6.0)
    base.update(kw)
    return dataclasses.replace(C.default_config().soma_fit, **base)


def toy_data(truth=TRUTH, noise_frac=0.0, seed=None):
    fwd = ToyForward(CROP, P_TOY, KS * DZ_TOY)
    D = B_TRUE[:, None, None] * fwd(truth[:8], truth[8])
    if noise_frac:
        D = D + np.random.default_rng(seed).normal(0.0, noise_frac * 100.0, D.shape)
    return D


def factory(z):
    return ToyForward(CROP, P_TOY, z)


def traced(truth=TRUTH):
    """the traced centre off by 1 um in each coordinate and the radius by 15 % (SPEC T4 (a))"""
    return (truth[0] + 1.0, truth[1] + 1.0, truth[2] + 1.0), 1.15 * float(np.cbrt(truth[3] * truth[4] * truth[5]))


def run_fit(D, scfg, mask=None, slope0=0.6, valid=None, truth=TRUTH, fwd_factory=factory):
    centre, rs = traced(truth)
    valid = np.ones(KS.size, bool) if valid is None else valid
    mask = np.zeros(D.shape[1:], bool) if mask is None else mask
    return SF.fit_soma(D, KS, valid, CROP[0], CROP[1], P_TOY, DZ_TOY, centre, rs, mask, fwd_factory, scfg, slope0)


def region_of(scfg, mask=None, truth=TRUTH):
    centre, _ = traced(truth)
    X, Y = np.meshgrid((CROP[0] + np.arange(CROP[2])) * P_TOY, (CROP[1] + np.arange(CROP[3])) * P_TOY)
    reg = (X - centre[0]) ** 2 + (Y - centre[1]) ** 2 <= scfg.region_radius_um ** 2
    return reg if mask is None else reg & ~mask


def ref_resid(theta, D, idx, region, scfg_far="fit", slope0=0.6):
    fwd = ToyForward(CROP, P_TOY, KS[idx] * DZ_TOY)
    g = theta[8] if scfg_far == "fit" else slope0
    tau = fwd(theta[:8], g)
    DR, TR = D[idx][:, region], tau[:, region]
    B = np.sum(DR * TR, axis=1) / np.sum(TR * TR, axis=1)
    return (DR - B[:, None] * TR).ravel(), B


@pytest.fixture(scope="module")
def noise_free_fit():
    scfg = toy_scfg(ftol=1e-10, xtol=1e-10, max_nfev=200)
    return run_fit(toy_data(), scfg), scfg


def test_noise_free_recovery_t4a(noise_free_fit):
    fit, scfg = noise_free_fit
    assert fit.status == "converged", fit.message
    err = np.abs(fit.theta - TRUTH)
    print("T4(a) (tester): max error %.2e, %d evaluations, %s" % (err.max(), fit.n_eval, fit.message))
    assert np.all(err <= 1e-4), dict(zip(NAMES, err))          # every parameter within 1e-4 of the truth [T4 (a)]
    assert fit.fitted == NAMES and fit.at_bound == ()


def test_contract_of_the_result(noise_free_fit):
    fit, scfg = noise_free_fit
    centre, rs = traced()
    idx = [i for i in range(KS.size) if abs(KS[i] * DZ_TOY - centre[2]) <= scfg.plane_half_um
           and (KS[i] - round(centre[2] / DZ_TOY)) % scfg.plane_step == 0]
    assert_array_equal(fit.ks, KS[idx])
    reg = region_of(scfg)
    K, nR = len(idx), int(reg.sum())
    assert fit.n_samples == K * nR and fit.dof == K * nR - 9 - K          # dof = |K||R| - fitted - |K|
    r, B = ref_resid(fit.theta, toy_data(), idx, reg)
    assert_allclose(fit.B_gl, B, rtol=1e-10)
    assert abs(fit.chi2 - float(r @ r)) <= 1e-9 * max(1.0, float(r @ r))
    assert fit.theta.shape == (9,) and fit.se.shape == (9,) and fit.cov.shape == (9, 9)
    a, b, c = fit.theta[3:6]
    S = ref_area(a, b, c)
    assert abs(fit.area_um2 - S) <= 1e-10 * S
    assert abs(fit.volume_um3 - 4 / 3 * math.pi * a * b * c) <= 1e-12 * fit.volume_um3
    assert abs(fit.r_eq_um - math.sqrt(S / (4 * math.pi))) <= 1e-10 * fit.r_eq_um
    assert fit.status in ("converged", "max_nfev", "failed") and isinstance(fit.message, str)


@pytest.fixture(scope="module")
def noisy_fit():
    scfg = toy_scfg()
    D = toy_data(noise_frac=0.01, seed=17)
    return run_fit(D, scfg), scfg, D


def test_noisy_recovery_and_se_t4b(noisy_fit):
    fit, scfg, D = noisy_fit
    assert fit.status == "converged", fit.message
    z = np.abs(fit.theta - TRUTH) / fit.se
    print("T4(b) (tester): |error|/SE = %s" % np.round(z, 2))
    assert np.all(z <= 4.0)                                    # every fitted parameter within 4 SE [T4 (b)]
    # the SE: sqrt(diag((J^T J)^-1) chi2/dof) with J an independent central-difference Jacobian of the projected
    # residual at theta_hat [5 %: the fit's J is its solver's forward-difference Jacobian]
    centre, _ = traced()
    idx = [i for i in range(KS.size) if KS[i] in set(fit.ks)]
    reg = region_of(scfg)
    h = np.array([2e-3, 2e-3, 2e-3, 2e-3, 2e-3, 2e-3, 4e-4, 1e-5, 5e-4])
    J = np.empty((fit.n_samples, 9))
    for j in range(9):
        tp, tm = fit.theta.copy(), fit.theta.copy()
        tp[j] += h[j]
        tm[j] -= h[j]
        J[:, j] = (ref_resid(tp, D, idx, reg)[0] - ref_resid(tm, D, idx, reg)[0]) / (2 * h[j])
    cov = np.linalg.inv(J.T @ J) * fit.chi2 / fit.dof
    se = np.sqrt(np.diag(cov))
    assert_allclose(fit.se, se, rtol=0.05)
    assert_allclose(fit.cov, (fit.cov + fit.cov.T) / 2, rtol=0, atol=1e-15 * np.max(np.abs(fit.cov)))
    assert_allclose(np.sqrt(np.diag(fit.cov)), fit.se, rtol=1e-12)
    # SE(S) by the delta method on the (a, b, c) block, grad S by central differences of the reference area
    a, b, c = fit.theta[3:6]
    g = np.array([(ref_area(*(np.array([a, b, c]) + d)) - ref_area(*(np.array([a, b, c]) - d))) / (2 * 1e-5 * x)
                  for d, x in ((np.array([1e-5 * a, 0, 0]), a), (np.array([0, 1e-5 * b, 0]), b),
                               (np.array([0, 0, 1e-5 * c]), c))])
    assert abs(fit.se_area_um2 - math.sqrt(g @ fit.cov[3:6, 3:6] @ g)) <= 1e-6 * fit.se_area_um2
    print("T4(b) (tester): chi2/dof %.4f (noise SD 1 gl -> expected about 1)" % (fit.chi2 / fit.dof))
    assert 0.9 <= fit.chi2 / fit.dof <= 1.1


def test_coverage_of_c_t4c():
    # 40 seeded noisy data sets: the fraction with |c_hat - c| <= SE(c_hat) in [0.45, 0.88] (SPEC T4 (c))
    scfg = toy_scfg()
    clean = toy_data()
    hits, n = {k: 0 for k in NAMES}, 0
    for seed in range(40):
        D = clean + np.random.default_rng(1000 + seed).normal(0.0, 1.0, clean.shape)
        fit = run_fit(D, scfg)
        assert fit.status == "converged", fit.message
        n += 1
        for j, k in enumerate(NAMES):
            hits[k] += int(abs(fit.theta[j] - TRUTH[j]) <= fit.se[j])
    cov = {k: v / n for k, v in hits.items()}
    print("T4(c) (tester): coverage of theta +- SE over %d sets: %s" % (n, {k: round(v, 3) for k, v in cov.items()}))
    assert 0.45 <= cov["c_um"] <= 0.88


def test_fixed_slope_t5():
    scfg = toy_scfg(far_slope="fixed", ftol=1e-10, xtol=1e-10, max_nfev=200)
    fit = run_fit(toy_data(), scfg, slope0=0.9)                  # held at the truth
    assert fit.theta[8] == 0.9 and np.isnan(fit.se[8])
    assert np.all(np.isnan(fit.cov[8, :])) and np.all(np.isnan(fit.cov[:, 8]))
    assert fit.fitted == NAMES[:8]
    assert np.all(np.abs(fit.theta[:8] - TRUTH[:8]) <= 1e-4)
    assert fit.dof == fit.n_samples - 8 - fit.ks.size


def test_masked_pixels_do_not_enter_t6():
    scfg = toy_scfg()
    D = toy_data(noise_frac=0.01, seed=4)
    X, Y = np.meshgrid((CROP[0] + np.arange(CROP[2])) * P_TOY, (CROP[1] + np.arange(CROP[3])) * P_TOY)
    stem = (np.abs(Y - 0.5) < 0.6) & (X > 1.5)                    # a stem leaving the soma, masked
    f0 = run_fit(D, scfg, mask=stem)
    D2 = D.copy()
    D2[:, stem] -= 40.0                                          # a dark stem on the masked pixels only
    f1 = run_fit(D2, scfg, mask=stem)
    assert np.max(np.abs(f1.theta - f0.theta)) <= 1e-10
    assert f1.n_samples == f0.n_samples == f0.ks.size * int(region_of(scfg, stem).sum())
    f2 = run_fit(D2, scfg)                                       # unmasked, the stem moves the fit
    assert np.max(np.abs(f2.theta - f0.theta)) > 1e-3


def test_start_clipped_and_truth_beyond_a_bound_t7():
    # the traced radius 15 % off would put a0 under a lower axis bound of 2.5 um: clipped and named in the message
    scfg = toy_scfg(axis_bounds_um=(2.5, 15.0), max_nfev=60)
    D = toy_data()
    centre, rs = (TRUTH[0] + 1.0, TRUTH[1] + 1.0, TRUTH[2] + 1.0), 1.6
    fit = SF.fit_soma(D, KS, np.ones(KS.size, bool), CROP[0], CROP[1], P_TOY, DZ_TOY, centre, rs,
                      np.zeros(D.shape[1:], bool), factory, scfg, 0.6)
    assert "clipped" in fit.message
    # the truth's c = 2.0 lies below that bound: c ends at the bound and is named in at_bound
    assert "c_um" in fit.at_bound and abs(fit.theta[5] - 2.5) <= 1e-6 * (15.0 - 2.5)
    # a slope truth above the slope bounds: the slope at its upper bound
    scfg2 = toy_scfg(slope_bounds=(0.2, 0.6), max_nfev=60)
    fit2 = run_fit(D, scfg2)
    assert "slope_um_per_um" in fit2.at_bound


def test_nan_forward_failure_determinism_and_refusals():
    scfg = toy_scfg(max_nfev=40)
    D = toy_data(noise_frac=0.01, seed=8)

    def nan_factory(z):
        f = ToyForward(CROP, P_TOY, z)
        return lambda th, s: np.full((len(z), CROP[3], CROP[2]), np.nan)

    fit = run_fit(D, scfg, fwd_factory=nan_factory)
    assert fit.status == "failed" and np.all(np.isnan(fit.theta)) and np.all(np.isnan(fit.se))
    assert np.isnan(fit.chi2) and np.isnan(fit.area_um2) and fit.at_bound == ()
    # NaN beyond a bound only, the truth's mu on that bound: the Jacobian's forward step would cross it, the column
    # is taken backwards, and the fit ends finite with mu at the bound
    def nan_above(z):
        f = ToyForward(CROP, P_TOY, z)
        def g(th, s):
            return np.full((len(z), CROP[3], CROP[2]), np.nan) if th[7] > 0.35 else f(th, s)
        return g
    fit_b = run_fit(D, toy_scfg(mu_bounds_per_um=(0.001, 0.35), max_nfev=60), fwd_factory=nan_above)
    assert fit_b.status in ("converged", "max_nfev") and np.all(np.isfinite(fit_b.theta)), fit_b.message
    assert np.all(np.isfinite(fit_b.se[:8]))
    # determinism
    f1, f2 = run_fit(D, scfg), run_fit(D, scfg)
    assert_array_equal(f1.theta, f2.theta)
    assert_array_equal(f1.se, f2.se)
    # the input is not modified; uint8 data accepted
    D8 = np.clip(np.round(D), 0, 255).astype(np.uint8)
    before = D8.copy()
    f8 = run_fit(D8, scfg)
    assert_array_equal(D8, before) and f8.status in ("converged", "max_nfev")
    # refusals
    centre, rs = traced()
    ok = dict(block=D, ks=KS, valid=np.ones(KS.size, bool), left=CROP[0], top=CROP[1], res0_um=P_TOY, dz_um=DZ_TOY,
              centre_um=centre, radius_um=rs, mask=np.zeros(D.shape[1:], bool), forward_factory=factory, scfg=scfg,
              slope0=0.6)
    bads = [dict(scfg=dataclasses.replace(scfg, far_slope="free")), dict(mask=np.ones(D.shape[1:], bool)),
            dict(valid=np.zeros(KS.size, bool)), dict(ks=KS[:-1]), dict(mask=np.zeros(D.shape[1:], int)),
            dict(radius_um=0.0), dict(slope0=-0.1), dict(centre_um=(0.0, 0.0)), dict(ks=KS[::-1].copy())]
    for bad in bads:
        kw = dict(ok)
        kw.update(bad)
        with pytest.raises(ValueError):
            SF.fit_soma(**kw)


def test_config_defaults_and_outside_signatures():
    cfg = C.default_config()
    q = cfg.soma_fit
    assert (q.fine_factor, q.dzeta_um, q.plane_step, q.plane_half_um, q.region_radius_um, q.mask_widen_um,
            q.far_slope, q.shift_xy_um, q.shift_z_um, q.axis_bounds_um, q.mu_bounds_per_um, q.slope_bounds,
            q.max_nfev, q.ftol, q.xtol, q.diff_steps) == (
        1, 0.14, 2, 10.0, 13.0, 2.0, "fit", 4.0, 6.0, (1.0, 15.0), (0.001, 3.0), (0.2, 3.0), 80, 1e-6, 1e-6,
        (0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.004, 1e-4, 0.005))
    assert cfg.signature_hash("estimator") == "16be09f944e45972"       # the pinned hashes (Block 1)
    assert cfg.signature_hash("simulation") == "dc84ed2e89bcfea6"
    h = {w: cfg.signature_hash(w) for w in ("estimator", "simulation", "full")}
    cfg2 = C.with_overrides(cfg, {"soma_fit.dzeta_um": 0.07, "soma_fit.far_slope": "fixed",
                                  "soma_fit.plane_step": 3, "soma_fit.axis_bounds_um": [0.5, 9.0]})
    assert {w: cfg2.signature_hash(w) for w in h} == h
    for bad in ({"soma_fit.far_slope": "free"}, {"soma_fit.plane_step": 0}, {"soma_fit.max_nfev": 2.5},
                {"soma_fit.fine_factor": True}, {"soma_fit.axis_bounds_um": [3.0, 1.0]},
                {"soma_fit.diff_steps": [0.02] * 8}, {"soma_fit.ftol": 0.0}, {"soma_fit.max_nfev": float("inf")}):
        with pytest.raises(ValueError):
            C.with_overrides(cfg, bad)
    # a configuration file without the section loads with the defaults
    d = json.loads(json.dumps(dataclasses.asdict(cfg)))
    d.pop("soma_fit")
    d.pop("soma_view")
    assert C.config_from_dict(d).soma_fit == q
