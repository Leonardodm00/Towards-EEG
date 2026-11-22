"""Spec-derived tests of Block 42 (the in-focus width calibrated on thick nodes; tester, 2026-11-22).

Written from specs/SPEC.md (Block 42, Block 4 amendment 2, Block 36 amendment 19) before the code was read.
Expected values come from the spec, from numpy / scipy, or from the independent references below; the code
under test (analysis.sigma_calibration) is only ever the thing being checked.

Independent references (nothing here calls the code under test):
  ref_tau        a six-parameter forward of the same family as the pipeline's (a round flat tube of diameter
                 d and coefficient mu, Beer-Lambert across its chord, blurred by a Gaussian whose variance is
                 sigma0^2 + c G(z_k - z_ax), G the configured table's growth), written from SPEC section 2.2 /
                 Block 3's (B3.1) with numpy's Gauss-Legendre rule. It is a stand-in for forward6, cheap enough
                 for hundreds of fits; the solver, the variable projection, the covariance and the start are
                 what is tested, not the line renderer's fidelity (the spec's T3/T4 note).
  ref_vp         per plane least squares of I on [tau, v tau] by numpy.linalg.lstsq.
  ref_dip        the half-depth width, depth and centre of one smoothed dip, from Block 40's text.

Pure ASCII.
"""
from __future__ import annotations

import dataclasses
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.ndimage import gaussian_filter1d

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
for _p in (WS / "src", WS / "scripts", WS / "tests" / "smoke", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from spec_helpers import ref_sigma_r  # noqa: E402  (an earlier tester's independent Block 4 kernel law)

from allen_diameter import config as C  # noqa: E402
from allen_diameter.analysis import sigma_calibration as SC  # noqa: E402

DZ = 0.28                     # um, the plane spacing (section 2.1)
B0, B1 = 150.0, -2.0          # gl, gl/um: the straight background of T3
NOISE = 1.1                   # gl per line sample (T4)
TRUE = np.array([1.5, 0.25, 0.04, 0.10, 0.15, 1.3])   # (d, mu, v0, z_ax, sigma0, c), T3's values about the start
KS = np.arange(-2, 3)                                   # the 5 planes k* +- 2, k* = 0
V = np.round(np.arange(-39, 40) * 0.1144, 10)          # +-4.5 um at the pixel pitch (Cell 23A's line)


def _growth(delta, rcfg):
    """G(delta) = sigma_r(delta)^2 - sigma_r(0)^2 of the configured table (um^2), by the earlier tester's law."""
    s0 = float(ref_sigma_r(0.0, rcfg))
    return np.asarray(ref_sigma_r(delta, rcfg), dtype=float) ** 2 - s0 ** 2


_PSI_X, _PSI_W = np.polynomial.legendre.leggauss(256)
_PSI = 0.5 * math.pi * _PSI_X
_PSI_W = 0.5 * math.pi * _PSI_W


def ref_tau(theta, ks=KS, v=V, rcfg=None):
    """(n_planes, n_v): 1 - (d/2) int (1 - exp(-mu d cos psi)) cos psi g_s(v - v0 - (d/2) sin psi) dpsi with
    s_k^2 = sigma0^2 + c G(z_k - z_ax), z_k = k dz (SPEC 2.2, Block 3 (B3.1) at phi = 0, Block 4 am. 2)."""
    d, mu, v0, z_ax, s0, c = (float(x) for x in theta)
    rcfg = rcfg or C.default_config().renderer
    G = _growth(np.asarray(ks, dtype=float) * DZ - z_ax, rcfg)
    var = s0 ** 2 + c * G
    if d <= 0 or mu < 0 or s0 <= 0 or c <= 0 or np.any(var <= 0):
        return np.full((len(ks), len(v)), np.nan)
    s = np.sqrt(var)[:, None, None]
    r = 0.5 * d
    u = (np.asarray(v, dtype=float)[None, :, None] - v0 - r * np.sin(_PSI)[None, None, :]) / s
    g = np.exp(-0.5 * u * u) / (math.sqrt(2.0 * math.pi) * s)
    integrand = (1.0 - np.exp(-mu * d * np.cos(_PSI))) * np.cos(_PSI)
    return 1.0 - r * np.sum(g * (integrand * _PSI_W)[None, None, :], axis=2)


def ref_forward6(*theta):
    return ref_tau(theta)


def ref_vp(I, tau, v):
    """(b (n_planes, 2), model (n_planes, n_v)) by numpy.linalg.lstsq over the finite samples of each plane."""
    b = np.full((I.shape[0], 2), np.nan)
    model = np.full(I.shape, np.nan)
    for k in range(I.shape[0]):
        m = np.isfinite(I[k]) & np.isfinite(tau[k])
        if m.sum() < 3:
            continue
        A = np.column_stack([tau[k, m], v[m] * tau[k, m]])
        b[k] = np.linalg.lstsq(A, I[k, m], rcond=None)[0]
        model[k] = (b[k, 0] + b[k, 1] * v) * tau[k]
    return b, model


def ref_resid(theta, I, sigma, v=V):
    tau = ref_tau(theta)
    _b, model = ref_vp(I, tau, v)
    return ((I - model) / sigma)[np.isfinite(I)]


def data(theta=TRUE, seed=None):
    I = (B0 + B1 * V)[None, :] * ref_tau(theta)
    if seed is not None:
        I = I + np.random.default_rng(seed).normal(0.0, NOISE, I.shape)
    return I


def scfg(**kw):
    q = C.default_config().sigma_calibration
    return dataclasses.replace(q, **kw) if kw else q


START = TRUE * np.array([1.1, 0.8, 1.0, 1.0, 1.0, 1.0]) + np.array([0, 0, 0.05, -0.1, 0.0, -0.3])
START[4] = 0.15 - 0.03                                  # sigma0 started 0.03 um off


# ----------------------------------------------------------------- planes ---

def test_select_planes_by_hand():
    ks = np.array([7, 3, 4, 5, 6, 9, 2])
    # |k - 5| <= 2 -> 7, 3, 4, 5, 6 at indices 0, 1, 2, 3, 4 (sorted indices, not sorted planes)
    assert list(SC.select_planes(ks, 5, 2)) == [0, 1, 2, 3, 4]
    assert list(SC.select_planes(ks, 3, 1)) == [1, 2, 6]
    for bad in ((ks, 9, 1), (ks, 5, 0), (ks, 5, 0.5), (np.array([1, 2]), 1, 3)):
        with pytest.raises(ValueError):
            SC.select_planes(*bad)


# ------------------------------------------------------------- background ---

def test_vp_background_against_lstsq():
    rng = np.random.default_rng(7)
    tau = 1.0 - 0.4 * np.exp(-0.5 * (V[None, :] / (0.5 + 0.1 * np.arange(3)[:, None])) ** 2)
    I_exact = (np.array([150.0, 120.0, 200.0])[:, None] + np.array([-2.0, 3.0, 0.5])[:, None] * V[None, :]) * tau
    b, resid = SC.vp_background(I_exact, tau, V)
    # exact data: b recovered to roundoff of a 2-column lstsq on O(100) values (1e-10 relative, T1)
    assert_allclose(b, [[150.0, -2.0], [120.0, 3.0], [200.0, 0.5]], rtol=1e-10, atol=1e-10)
    assert_allclose(resid, 0.0, atol=1e-9)
    I = I_exact + rng.normal(0, 3.0, I_exact.shape)
    I[0, 5] = np.nan
    I[1, :] = np.nan
    I[1, 10:12] = 7.0                       # 2 finite samples: fewer than 3 -> NaN plane
    tau2 = tau.copy()
    tau2[2, 40] = np.nan
    b, resid = SC.vp_background(I, tau2, V)
    b_ref, model_ref = ref_vp(I, tau2, V)
    assert_allclose(b[[0, 2]], b_ref[[0, 2]], rtol=1e-12, atol=1e-12)   # the same lstsq: roundoff
    assert np.all(np.isnan(b[1])) and np.all(np.isnan(resid[1]))
    assert np.isnan(resid[0, 5]) and np.isnan(resid[2, 40])
    m = np.isfinite(model_ref) & np.isfinite(I)
    assert_allclose(resid[m], (I - model_ref)[m], rtol=0, atol=1e-10)


# ---------------------------------------------------------------- summary ---

def test_summarize_by_hand():
    s = np.array([0.10, 0.12, 0.11, 0.30, 0.09])
    e = np.array([0.01, 0.02, 0.01, 0.05, 0.02])
    ok = np.array([True, True, True, False, True])
    out = SC.summarize(s, e, ok)
    x, w = s[ok], 1.0 / e[ok] ** 2
    mean = np.sum(w * x) / np.sum(w)
    assert out["n"] == 4 and out["dof"] == 3
    assert_allclose(out["median_um"], np.median(x), rtol=1e-15)
    assert_allclose(out["mad_um"], np.median(np.abs(x - np.median(x))), rtol=1e-15)   # unscaled MAD
    assert_allclose(out["ivw_mean_um"], mean, rtol=1e-14)
    assert_allclose(out["ivw_se_um"], 1.0 / math.sqrt(np.sum(w)), rtol=1e-14)
    assert_allclose(out["chi2_cons"], np.sum(w * (x - mean) ** 2), rtol=1e-12)
    one = SC.summarize(s, e, np.array([True, False, False, False, False]))
    assert one["n"] == 1 and math.isnan(one["chi2_cons"]) and one["ivw_mean_um"] == 0.10
    none = SC.summarize(s, e, np.zeros(5, dtype=bool))
    assert none["n"] == 0 and all(math.isnan(none[k]) for k in ("median_um", "mad_um", "ivw_mean_um",
                                                                 "ivw_se_um", "chi2_cons"))
    for bad in ((s, e[:4], ok), (s, e, ok.astype(int)), (s, np.where(ok, np.nan, e), ok)):
        with pytest.raises(ValueError):
            SC.summarize(*bad)


# -------------------------------------------------------------------- fit ---

def test_noise_free_recovery_and_contract():
    """T3 with an independent forward: profiles (b0 + b1 v) tau(theta_true) fitted from a start off in
    every parameter. Tolerances: T3's own (sigma0 5e-3 um, d 1 %, c 3 %), which the forward's discretisation
    allows; here the data and the model are the same function, so the observed errors are reported too."""
    I = data()
    q = scfg(max_nfev=200)
    fit = SC.fit_sigma_node(I, NOISE, V, ref_forward6, START, q)
    err = fit.theta - TRUE
    print("noise-free: errors", err, "status", fit.status, "n_eval", fit.n_eval)
    assert fit.status == "converged", fit.message
    assert abs(err[4]) <= 5e-3 and abs(err[0]) / TRUE[0] <= 1e-2 and abs(err[5]) / TRUE[5] <= 3e-2
    assert fit.at_bound == ()
    # contract: shapes, n_res, dof = n_res - (6 + 2 n_planes), chi2 = sum r^2, chi2_per_plane = mean r^2
    assert fit.theta.shape == fit.se.shape == fit.se_scaled.shape == (6,)
    assert fit.b.shape == (5, 2) and fit.model_gl.shape == I.shape and fit.chi2_per_plane.shape == (5,)
    assert fit.n_res == I.size and fit.dof == I.size - (6 + 2 * 5)
    r = ref_resid(fit.theta, I, NOISE)
    assert_allclose(fit.chi2, np.sum(r ** 2), rtol=1e-8, atol=1e-12)
    assert_allclose(fit.chi2_per_plane, (r.reshape(I.shape) ** 2).mean(axis=1), rtol=1e-8, atol=1e-12)
    b_ref, model_ref = ref_vp(I, ref_tau(fit.theta), V)
    assert_allclose(fit.b, b_ref, rtol=1e-9, atol=1e-9)
    assert_allclose(fit.model_gl, model_ref, rtol=1e-12, atol=1e-9)
    assert_allclose(fit.se_scaled, fit.se * math.sqrt(fit.chi2 / fit.dof), rtol=1e-12)
    assert fit.n_eval >= 7 and fit.seconds >= 0


def _central_jacobian(theta, I, sigma, h):
    J = np.empty((int(np.isfinite(I).sum()), 6))
    for j in range(6):
        e = np.zeros(6)
        e[j] = h[j]
        J[:, j] = (ref_resid(theta + e, I, sigma) - ref_resid(theta - e, I, sigma)) / (2 * h[j])
    return J


def test_covariance_is_inverse_jtj_of_the_projected_residual():
    """SE = sqrt(diag((J^T J)^-1)), J the Jacobian of the projected residual (Golub-Pereyra): with tiny
    absolute steps the solver's forward-difference J is the true one to O(h) = 1e-6 relative; the reference
    J is a central difference of an independent VP residual. Tolerance 1e-3 relative on SE (the two
    difference rules and the solver's last step)."""
    I = data(seed=11)
    q = scfg(max_nfev=200, diff_steps=(1e-6,) * 6)
    fit = SC.fit_sigma_node(I, NOISE, V, ref_forward6, START, q)
    assert fit.status == "converged", fit.message
    J = _central_jacobian(fit.theta, I, NOISE, np.full(6, 1e-5))
    se_ref = np.sqrt(np.diag(np.linalg.inv(J.T @ J)))
    print("se", fit.se, "ref", se_ref)
    assert_allclose(fit.se, se_ref, rtol=1e-3)


def test_se_scaled_coverage_on_seeded_noise():
    """The interval theta_hat +- SE_s covers the truth at the nominal 68.3 % rate. 150 seeded data sets,
    noise 1.1 gl, the configured steps and budget; binomial 3-SD band at n = 150: 0.683 +- 0.114."""
    n, hits, chi = 150, np.zeros(6), []
    q = scfg()
    statuses = {}
    for s in range(n):
        I = data(seed=1000 + s)
        fit = SC.fit_sigma_node(I, NOISE, V, ref_forward6, START, q)
        statuses[fit.status] = statuses.get(fit.status, 0) + 1
        hits += np.abs(fit.theta - TRUE) <= fit.se_scaled
        chi.append(fit.chi2 / fit.dof)
    rate = hits / n
    print("coverage per parameter", rate, "statuses", statuses, "mean chi2/dof", np.mean(chi))
    for j in (0, 4, 5):                                  # d, sigma0, c: the quantities the summary reports
        assert 0.683 - 0.114 <= rate[j] <= 0.683 + 0.114, (SC.PARAMS[j], rate[j])
    assert abs(np.mean(chi) - 1.0) <= 0.05              # 150 x ~375 dof: SD of the mean ~ 0.006


def test_failure_modes_never_raise():
    I = data()
    nan6 = lambda *th: np.full((5, V.size), np.nan)    # noqa: E731
    fit = SC.fit_sigma_node(I, NOISE, V, nan6, START, scfg())
    assert fit.status == "failed" and np.all(np.isnan(fit.theta)) and fit.message
    # a start outside the bounds is clipped (and the fit still runs)
    st = START.copy()
    st[4] = 5.0
    fit = SC.fit_sigma_node(I, NOISE, V, ref_forward6, st, scfg(max_nfev=200))
    assert fit.status in ("converged", "max_nfev") and np.isfinite(fit.theta[4])
    assert "clipped" in fit.message
    # a non-finite start is "failed"
    st = START.copy()
    st[0] = np.nan
    assert SC.fit_sigma_node(I, NOISE, V, ref_forward6, st, scfg()).status == "failed"
    for bad in (dict(prof=I[:2]), dict(sigma_gl=0.0), dict(sigma_gl=np.nan), dict(theta0=START[:5])):
        kw = dict(prof=I, sigma_gl=NOISE, v=V, forward6=ref_forward6, theta0=START, scfg=scfg())
        kw.update(bad)
        with pytest.raises(ValueError):
            SC.fit_sigma_node(**kw)


def test_jacobian_steps_absolute_and_backward_at_the_upper_bound():
    """The custom jac: column j from r(x + h_j e_j) with the absolute step h_j; backwards (x - h_j e_j)
    when x_j + h_j leaves the upper bound or gives a non-finite residual. A recording forward shows the
    points the solver asks for."""
    I = data()
    calls = []

    def rec(*th):
        calls.append(np.array(th, dtype=float))
        return ref_forward6(*th)
    q = scfg(max_nfev=1)
    SC.fit_sigma_node(I, NOISE, V, rec, START, q)
    pts = np.array(calls)
    x0 = pts[0]
    steps = np.asarray(q.diff_steps)
    for j in range(6):
        want = x0.copy()
        want[j] += steps[j]
        assert np.any(np.all(np.abs(pts - want) <= 1e-15, axis=1)), ("forward step", j)
    # sigma0's upper bound just above the start: the column must be taken backwards
    calls.clear()
    st = START.copy()
    hi = st[4] + 0.001
    q2 = scfg(max_nfev=1, sigma0_bounds_um=(0.02, hi))
    SC.fit_sigma_node(I, NOISE, V, rec, st, q2)
    pts = np.array(calls)
    x0 = pts[0]
    back = x0.copy()
    back[4] -= steps[4]
    fwd = x0.copy()
    fwd[4] += steps[4]
    assert np.any(np.all(np.abs(pts - back) <= 1e-15, axis=1))
    assert not np.any(np.all(np.abs(pts - fwd) <= 1e-15, axis=1))
    # a forward that is NaN only beyond d = start + 0.005 (inside the box): backward column, still finite fit
    calls.clear()
    lim = START[0] + 0.005

    def nan_beyond(*th):
        calls.append(np.array(th, dtype=float))
        return ref_forward6(*th) if th[0] <= lim else np.full((5, V.size), np.nan)
    fit = SC.fit_sigma_node(I, NOISE, V, nan_beyond, START, scfg(max_nfev=3))
    assert fit.status in ("converged", "max_nfev") and np.all(np.isfinite(fit.theta)), fit.message
    pts = np.array(calls)
    x0 = pts[0]
    back = x0.copy()
    back[0] -= steps[0]
    assert np.any(np.all(np.abs(pts - back) <= 1e-15, axis=1))


def test_at_bound_names_and_rule():
    """A truth beyond a bound ends with that parameter within 1e-6 (relative to the width) of the bound."""
    I = data()
    q = scfg(max_nfev=200, sigma0_bounds_um=(0.02, 0.12))
    st = START.copy()
    st[4] = 0.10
    fit = SC.fit_sigma_node(I, NOISE, V, ref_forward6, st, q)
    print("at_bound", fit.at_bound, fit.theta)
    assert "sigma0_um" in fit.at_bound
    assert abs(fit.theta[4] - 0.12) <= 1e-6 * (0.12 - 0.02)


def test_determinism_and_inputs_unchanged():
    I = data(seed=5)
    I0, st0 = I.copy(), START.copy()
    a = SC.fit_sigma_node(I, NOISE, V, ref_forward6, START, scfg())
    b = SC.fit_sigma_node(I, NOISE, V, ref_forward6, START, scfg())
    assert np.array_equal(a.theta, b.theta) and np.array_equal(a.se, b.se) and a.n_eval == b.n_eval
    assert np.array_equal(I, I0) and np.array_equal(START, st0)


# ------------------------------------------------------------------- start ---

def _ref_crossing(D, v, n, step, level):
    while 0 <= n + step < D.size and D[n + step] > level:
        n += step
    if not 0 <= n + step < D.size:
        return math.nan
    return v[n] + (v[n + step] - v[n]) * (D[n] - level) / (D[n] - D[n + step])


def ref_dip(D, v, window=1.0):
    near = np.flatnonzero(np.abs(v) <= window)
    n_c = int(near[np.argmax(D[near])])
    Dc = float(D[n_c])
    lo, hi = _ref_crossing(D, v, n_c, -1, Dc / 2), _ref_crossing(D, v, n_c, +1, Dc / 2)
    return hi - lo, Dc, 0.5 * (lo + hi)


class _Pilot(object):
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_start_theta_by_hand():
    """SPEC Block 42 start: L the lstsq line through |v| >= start_ends_um; W, D, c of the smoothed L - I at
    k*; d0 = W / width_to_d; mu0 = -ln max(1 - D / L(c), 1e-3) / (1.447 d0); v0, z_ax from pilot_start;
    sigma0, c the configured starts; clipped. The pilot and trace are made up so v_t is not 0."""
    q = scfg()
    th_true = np.array([1.5, 0.25, 0.04, 0.10, 0.15, 1.3])
    I = (B0 + 6.0 * V)[None, :] * ref_tau(th_true)          # a steep straight background, 6 gl/um (T8)
    pilot = _Pilot(cx_um=1.3, cy_um=-0.7, theta_rad=0.4, phi_rad=0.0, z_sub_um=0.31, d_hat_um=0.7,
                   mu_hat_per_um=0.9)
    trace = np.array([1.1, -0.9, 0.2])
    th0 = SC.start_theta(I, np.full(5, B0), V, 2, pilot, trace, q)
    m = np.abs(V) >= q.start_ends_um
    a, b = np.linalg.lstsq(np.column_stack([np.ones(m.sum()), V[m]]), I[2, m], rcond=None)[0]
    D = gaussian_filter1d((a + b * V) - I[2], 1.0, mode="nearest")
    W, Dc, cen = ref_dip(D, V)
    d0 = W / q.width_to_d
    mu0 = -math.log(max(1.0 - Dc / (a + b * cen), 1e-3)) / (q.hybrid_faint_factor * d0)
    yx, yy = -math.sin(0.4), math.cos(0.4)
    v_t = (1.3 - 1.1) * yx + (-0.7 + 0.9) * yy
    want = np.array([d0, mu0, v_t, 0.31, q.sigma0_start_um, q.growth_scale_start])
    assert_allclose(th0, want, rtol=1e-10, atol=1e-12)
    # T8: the start is blind to the straight background (within 5 %: the dip's depth is measured against L)
    th_flat = SC.start_theta((B0 + 0.0 * V)[None, :] * ref_tau(th_true), np.full(5, B0), V, 2, pilot, trace, q)
    assert abs(th0[0] / th_flat[0] - 1) <= 0.05 and abs(th0[1] / th_flat[1] - 1) <= 0.05
    # fewer than 3 finite samples at an end: L = B_k*; no dip at all: the pilot's d and mu (clipped)
    I2 = I.copy()
    I2[2, V >= q.start_ends_um] = np.nan
    I2[2, np.flatnonzero(V >= q.start_ends_um)[:2]] = 140.0
    th2 = SC.start_theta(I2, np.full(5, 160.0), V, 2, pilot, trace, q)
    D2 = gaussian_filter1d(160.0 - I2[2], 1.0, mode="nearest")
    if np.all(np.isfinite(D2)):
        W2, _Dc2, _c2 = ref_dip(D2, V)
        assert_allclose(th2[0], W2 / q.width_to_d, rtol=1e-10)
    else:                                                   # NaN reaches the dip: the pilot's diameter
        assert_allclose(th2[0], 0.7, rtol=0)
    flat = np.full((5, V.size), 150.0)
    th3 = SC.start_theta(flat, np.full(5, 150.0), V, 2, pilot, trace, q)
    assert_allclose(th3[:2], [0.7, 0.9], rtol=0)


# --------------------------------------------------- forward_with_kernel ---

def _hand_node(cfg):
    """A NodeData built by hand (no rendering): the line through (2.0, 0.3) at heading 0.2 rad."""
    from allen_diameter.analysis.clean_iteration import NodeData
    th = 0.2
    y_hat = np.array([-math.sin(th), math.cos(th)])
    e_u = np.array([math.cos(th), math.sin(th)])
    v = np.arange(-30, 31) * cfg.measure.profile_step_um
    ks = np.arange(-2, 3)
    return NodeData(node_id=5, own_ids=(4, 5, 6), node_xyz=np.array([2.0, 0.3, 0.05]),
                    block=np.zeros((5, 4, 4)), B_gl=np.full(5, 150.0), ks=ks, planes_z=ks * DZ, left=0, top=0,
                    origin_xy=np.array([2.0, 0.3]), y_hat=y_hat, e_u=e_u, v=v, n_avg=2,
                    step=float(cfg.measure.profile_step_um), theta0=None)


def test_forward_with_kernel_against_forward_factory_and_the_cache():
    """T2 exactly; NaN arrays for a refused kernel; one SectionCache shared across kernels gives what a fresh
    cache gives [1e-12] (the section's key holds no kernel; boundary Blocks 4 -> 35); the kernel is used."""
    from allen_diameter.analysis import node_inputs
    from allen_diameter.model import line_fast
    cfg = C.default_config()
    cfg = dataclasses.replace(cfg, inference=dataclasses.replace(cfg.inference, ray_n_rho=2, ray_n_psi=4))
    node = _hand_node(cfg)
    pilot = _Pilot(cx_um=2.0, cy_um=0.3, theta_rad=0.2, phi_rad=0.15, z_sub_um=0.05, d_hat_um=1.0,
                   mu_hat_per_um=0.3)
    U = (5.0, 5.0)
    th4 = (1.2, 0.3, 0.02, 0.07)
    s0 = float(cfg.renderer.sigma_r0_um)
    f6 = SC.forward_with_kernel(pilot, U, node, cfg, line_fast.SectionCache())
    ref = node_inputs.forward_factory(pilot, U, node, cfg, line_fast.SectionCache())(*th4)
    got = f6(*th4, s0, 1.0)
    assert np.array_equal(got, ref)
    shared = line_fast.SectionCache()
    f_sh = SC.forward_with_kernel(pilot, U, node, cfg, shared)
    a1 = f_sh(*th4, 0.15, 1.0)
    a2 = f_sh(*th4, 0.30, 2.5)                       # the second kernel reuses the cached section
    b2 = SC.forward_with_kernel(pilot, U, node, cfg, line_fast.SectionCache())(*th4, 0.30, 2.5)
    assert shared.hits >= 1
    assert_allclose(a2, b2, rtol=0, atol=1e-12)
    assert np.max(np.abs(a1 - a2)) > 1e-3            # the kernel does enter
    for bad in ((0.0, 1.0), (-0.1, 1.0), (0.15, 0.0), (0.15, -1.0), (np.nan, 1.0)):
        out = f6(*th4, *bad)
        assert out.shape == ref.shape and np.all(np.isnan(out)), bad
    emp = dataclasses.replace(cfg, renderer=dataclasses.replace(cfg.renderer, kernel_family="empirical"))
    with pytest.raises(ValueError):
        SC.forward_with_kernel(pilot, U, node, emp, None)


def test_unidentified_direction_gives_nan_se_only_there():
    """Block 42: "the covariance by pinv with NaN for unidentified directions" (accepted at the gate). A forward
    that ignores c: c's column of J is zero, its SE NaN; the identified parameters keep finite SEs equal to
    (J^T J)^-1 of the five others (the pinv of a block-diagonal J^T J)."""
    I = data(seed=21)
    no_c = lambda d, mu, v0, z, s0, c: ref_tau((d, mu, v0, z, s0, 1.3))    # noqa: E731
    fit = SC.fit_sigma_node(I, NOISE, V, no_c, START, scfg(max_nfev=200, diff_steps=(1e-6,) * 6))
    assert np.isnan(fit.se[5]) and np.all(np.isfinite(fit.se[:5])), fit.se
    assert "pinv" in fit.message
    th = fit.theta.copy()

    def r5(t5):
        return ref_resid(np.concatenate([t5, [1.3]]), I, NOISE)
    J = np.empty((I.size, 5))
    for j in range(5):
        e = np.zeros(5)
        e[j] = 1e-5
        J[:, j] = (r5(th[:5] + e) - r5(th[:5] - e)) / 2e-5
    assert_allclose(fit.se[:5], np.sqrt(np.diag(np.linalg.inv(J.T @ J))), rtol=1e-3)
