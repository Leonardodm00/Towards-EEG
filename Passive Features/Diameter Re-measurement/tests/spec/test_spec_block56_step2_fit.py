"""Spec-derived tests of Block 56 (analysis/step2_fit.py; tester, 2026-11-27). Oracles F1 to F8 of SPEC.md Block 56:
linear recovery; a non-linear toy against scipy.optimize.least_squares with its own 2-point Jacobian; the callable's
Jacobian against the analytic one; the steps actually taken are the absolute steps asked for; the clustered (CR0) and
classical covariances against statsmodels; a Monte Carlo of the clustered SE; bounds and `free`; S + L model calls per
Jacobian and the reversed step at an upper bound; the NaN rules; L = 0; determinism; the refusals.

Pure ASCII.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.optimize import least_squares

import step2_helpers as H  # noqa: F401  (sys.path)
from allen_diameter.analysis import step2_fit as F


def _linear_problem(seed=0, N=6, m=5, S=2, L=1, noise=0.0, corr=0.0):
    rng = np.random.default_rng(seed)
    group = np.repeat(np.arange(N), m)
    n = group.size
    A = rng.normal(size=(n, S))
    Bg = rng.normal(size=(n, L)) if L else np.zeros((n, 0))
    a = rng.normal(size=S)
    b = rng.normal(size=(N, L))
    y = A @ a + (np.sum(Bg * b[group], axis=1) if L else 0.0)
    y = y + noise * rng.normal(size=n) + corr * rng.normal(size=N)[group]

    def model(sh, lo):
        return A @ sh + (np.sum(Bg * lo[group], axis=1) if L else 0.0)
    return model, y, group, A, Bg, a, b


def test_F1_linear_recovery():
    model, y, g, A, Bg, a, b = _linear_problem()
    r = F.fit_shared_local(model, y, np.ones_like(y), g, np.zeros(2), (-10 * np.ones(2), 10 * np.ones(2)),
                           local0=np.zeros((6, 1)), local_bounds=(-10 * np.ones((6, 1)), 10 * np.ones((6, 1))),
                           diff_step_shared=[1e-3, 1e-3], diff_step_local=[1e-3])
    # 1e-8: a linear model's forward difference is exact up to roundoff, so the fit converges to the solution
    assert_allclose(r.shared, a, rtol=0, atol=1e-8)
    assert_allclose(r.local, b, rtol=0, atol=1e-8)
    assert r.jac.shape == (y.size, 2 + 6) and r.residual.shape == y.shape and r.kept.all()
    assert_allclose(r.cost, 0.5 * np.sum(r.residual ** 2), rtol=1e-12)
    assert r.success


def _parabola(seed=1, N=5, noise=0.01):
    # step 2's near stage in miniature: y_gk = c (k - m_g)^2 + b0, curvature c and offset b0 shared, centre m_g local
    rng = np.random.default_rng(seed)
    k = np.tile(np.arange(-3, 4, dtype=float), N)
    group = np.repeat(np.arange(N), 7)
    c, b0 = 0.05, 0.02
    m = rng.uniform(-0.4, 0.4, N)
    y = c * (k - m[group]) ** 2 + b0 + noise * rng.normal(size=k.size)

    def model(sh, lo):
        return sh[0] * (k - lo[group, 0]) ** 2 + sh[1]
    return model, y, group, k, (c, b0, m)


def test_F2_nonlinear_toy_against_scipy():
    model, y, g, k, _ = _parabola()
    N = 5
    hs, hl = np.array([1e-6, 1e-6]), np.array([1e-6])
    r = F.fit_shared_local(model, y, np.ones_like(y), g, np.array([0.03, 0.0]), ([1e-4, -1.0], [1.0, 1.0]),
                           local0=np.zeros((N, 1)), local_bounds=(-np.ones((N, 1)), np.ones((N, 1))),
                           diff_step_shared=hs, diff_step_local=hl)

    def fun(x):
        return y - model(x[:2], x[2:].reshape(N, 1))
    ref = least_squares(fun, np.r_[0.03, 0.0, np.zeros(N)], bounds=(np.r_[1e-4, -1.0, -np.ones(N)],
                                                                    np.r_[1.0, 1.0, np.ones(N)]),
                        method="trf", x_scale="jac", jac="2-point", xtol=1e-12, ftol=1e-12, gtol=1e-12)
    # 1e-6: the spec's tolerance; both converge to the same least-squares minimum, the steps differ (absolute
    # 1e-6 against scipy's relative ones), which moves the solution by about h times the residual (< 1e-7 here)
    assert_allclose(np.r_[r.shared, r.local.ravel()], ref.x, rtol=0, atol=1e-6)
    # the returned Jacobian against the analytic one at the solution: d r/d c = -(k - m)^2, d r/d b0 = -1,
    # d r/d m_g = 2 c (k - m_g); forward-difference truncation h |f''| / 2: h c-curvature terms are 0 for c and b0
    # (linear), 2 c h / 2 = c h for m_g: tolerance 2e-6 covers it with margin
    c, b0 = r.shared
    m = r.local[:, 0]
    Ja = np.zeros((y.size, 2 + N))
    Ja[:, 0] = -(k - m[g]) ** 2
    Ja[:, 1] = -1.0
    Ja[np.arange(y.size), 2 + g] = 2 * c * (k - m[g])
    assert_allclose(r.jac, Ja, rtol=0, atol=2e-6)


def test_F2_steps_taken_are_absolute():
    model, y, g, k, _ = _parabola()
    N = 5
    seen = []

    def spy(sh, lo):
        seen.append(np.r_[sh, lo.ravel()].copy())
        return model(sh, lo)
    hs, hl = np.array([0.005, 0.02]), np.array([0.03])
    F.fit_shared_local(spy, y, np.ones_like(y), g, np.array([0.03, 0.0]), ([1e-4, -1.0], [1.0, 1.0]),
                       local0=np.zeros((N, 1)) + 0.1, local_bounds=(-np.ones((N, 1)), np.ones((N, 1))),
                       diff_step_shared=hs, diff_step_local=hl, max_nfev=10)
    X = np.array(seen)
    # find every Jacobian: a base point followed by S + L = 3 probes that each differ from it in a known pattern
    found = 0
    for i in range(len(X) - 3):
        d1, d2, d3 = X[i + 1] - X[i], X[i + 2] - X[i], X[i + 3] - X[i]
        if np.count_nonzero(d1) == 1 and d1[0] != 0 and np.count_nonzero(d2) == 1 and d2[1] != 0 \
                and np.count_nonzero(d3) == N and np.all(d3[2:] != 0):
            assert_allclose(abs(d1[0]), 0.005, rtol=1e-9)
            assert_allclose(abs(d2[1]), 0.02, rtol=1e-9)
            assert_allclose(np.abs(d3[2:]), 0.03, rtol=1e-9)
            found += 1
    assert found >= 2                                    # at the start and at a later iterate


def _ols_reference(A, Bg, group, y, N):
    sm = pytest.importorskip("statsmodels.api", reason="statsmodels needed for F3")
    D = np.zeros((y.size, N * Bg.shape[1]))
    for gi in range(N):
        sel = group == gi
        D[sel, gi * Bg.shape[1]:(gi + 1) * Bg.shape[1]] = Bg[sel]
    X = np.hstack([A, D])
    cl = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": group, "use_correction": False})
    nr = sm.OLS(y, X).fit()
    S = A.shape[1]
    return cl.params[:S], cl.cov_params()[:S, :S], nr.cov_params()[:S, :S]


def test_F3_covariances_against_statsmodels():
    model, y, g, A, Bg, a, b = _linear_problem(seed=4, N=8, m=6, noise=0.3, corr=0.5)
    p, cov_cl, cov_nr = _ols_reference(A, Bg, g, y, 8)
    r = F.fit_shared_local(model, y, np.ones_like(y), g, np.zeros(2), (-10 * np.ones(2), 10 * np.ones(2)),
                           local0=np.zeros((8, 1)), local_bounds=(-10 * np.ones((8, 1)), 10 * np.ones((8, 1))),
                           diff_step_shared=[1e-3, 1e-3], diff_step_local=[1e-3])
    assert_allclose(r.shared, p, rtol=0, atol=1e-8)
    # 1e-8 relative (spec): a linear model's Jacobian is exact to roundoff
    assert_allclose(r.cov_cluster, cov_cl, rtol=1e-8)
    assert_allclose(r.cov_classical, cov_nr, rtol=1e-8)
    assert_allclose(r.se_cluster, np.sqrt(np.diag(cov_cl)), rtol=1e-8)
    assert_allclose(r.se_classical, np.sqrt(np.diag(cov_nr)), rtol=1e-8)


def test_F3_weighted_against_statsmodels_wls():
    sm = pytest.importorskip("statsmodels.api", reason="statsmodels needed for F3")
    model, y, g, A, Bg, a, b = _linear_problem(seed=5, N=7, m=5, L=0, noise=0.3, corr=0.4)
    w = np.random.default_rng(9).uniform(0.5, 3.0, y.size)
    r = F.fit_shared_local(model, y, w, g, np.zeros(2), (-10 * np.ones(2), 10 * np.ones(2)),
                           diff_step_shared=[1e-3, 1e-3])
    cl = sm.WLS(y, A, weights=w).fit(cov_type="cluster", cov_kwds={"groups": g, "use_correction": False})
    nr = sm.WLS(y, A, weights=w).fit()
    assert_allclose(r.shared, cl.params, rtol=0, atol=1e-8)
    assert_allclose(r.cov_cluster, cl.cov_params(), rtol=1e-8)
    assert_allclose(r.cov_classical, nr.cov_params(), rtol=1e-8)


def test_F4_monte_carlo_clustered_se():
    # y_gj = a0 + a1 x_g + u_g + e_gj: the regressor constant inside a group, the error correlated inside groups
    # (rho = su^2 / (su^2 + se^2) = 0.5), N = 40 groups of m = 5, no local unknowns. The classical SE of a1 assumes
    # independent rows and is too small by the design effect's root sqrt(1 + (m - 1) rho) = sqrt(3) = 1.73 for a
    # regressor that is constant within groups (Kish); the clustered SE estimates the true SD (CR0 is low by about
    # sqrt((G - 1) / G) = 0.987 at G = 40), so within 15 % of the Monte Carlo SD (400 datasets: the SD's own
    # relative SE is 1 / sqrt(2 x 399) = 3.5 %).
    rng = np.random.default_rng(20261127)
    N, m = 40, 5
    group = np.repeat(np.arange(N), m)
    x = rng.normal(size=N)[group]
    X = np.column_stack([np.ones(N * m), x])

    def model(sh, lo):
        return X @ sh
    est, se_cl, se_cs = [], [], []
    for _ in range(400):
        y = X @ np.array([1.0, 0.5]) + rng.normal(size=N)[group] + rng.normal(size=N * m)
        r = F.fit_shared_local(model, y, np.ones_like(y), group, np.zeros(2), ([-10, -10], [10, 10]),
                               diff_step_shared=[1e-3, 1e-3])
        est.append(r.shared[1])
        se_cl.append(r.se_cluster[1])
        se_cs.append(r.se_classical[1])
    sd = float(np.std(est, ddof=1))
    assert abs(np.mean(se_cl) / sd - 1.0) <= 0.15, (np.mean(se_cl), sd)
    ratio = sd / float(np.mean(se_cs))
    assert abs(ratio / math.sqrt(3.0) - 1.0) <= 0.15, ratio


def test_F5_bounds_and_free():
    model, y, g, A, Bg, a, b = _linear_problem(seed=2)
    hi = np.array([a[0] - 0.3, 10.0])                     # the first shared unknown's optimum lies above its bound
    r = F.fit_shared_local(model, y, np.ones_like(y), g, np.array([hi[0] - 1.0, 0.0]), (-10 * np.ones(2), hi),
                           local0=np.zeros((6, 1)), local_bounds=(-10 * np.ones((6, 1)), 10 * np.ones((6, 1))),
                           diff_step_shared=[1e-3, 1e-3], diff_step_local=[1e-3])
    assert r.shared[0] <= hi[0] + 1e-12
    assert_allclose(r.shared[0], hi[0], rtol=0, atol=1e-8)
    assert not r.free[0] and r.free[1:].all()
    assert np.all(np.isnan(r.cov_cluster[0])) and np.all(np.isnan(r.cov_cluster[:, 0]))
    assert np.isfinite(r.cov_cluster[1, 1]) and np.isnan(r.se_cluster[0]) and np.isfinite(r.se_classical[1])


def test_F6_model_calls_per_jacobian_and_reversed_step():
    model, y, g, k, _ = _parabola(N=4)
    N = 4
    calls = []

    def spy(sh, lo):
        calls.append(np.r_[sh, lo.ravel()].copy())
        return model(sh, lo)
    # start the curvature at its upper bound: the first Jacobian must step backwards there
    r = F.fit_shared_local(spy, y, np.ones_like(y), g, np.array([0.06, 0.0]), ([1e-4, -1.0], [0.06, 1.0]),
                           local0=np.zeros((N, 1)), local_bounds=(-np.ones((N, 1)), np.ones((N, 1))),
                           diff_step_shared=[0.004, 0.01], diff_step_local=[0.02], max_nfev=1)
    X = np.array(calls)
    # max_nfev 1: the start's prediction, the residual at scipy's start (trf moves a start that lies on a bound
    # strictly inside, by 1e-10 relative, so the cached start prediction is not reused: one extra call), then one
    # Jacobian = S + L = 3 probes about that point
    assert len(X) == 2 + 3, len(X)
    base = X[1]
    assert 0 < 0.06 - base[0] < 1e-9
    d = X[2:] - base
    assert_allclose(d[0], [-0.004, 0, 0, 0, 0, 0], rtol=0, atol=1e-15)   # reversed at the upper bound
    assert_allclose(d[1], [0, 0.01, 0, 0, 0, 0], rtol=0, atol=1e-15)
    assert_allclose(d[2], [0, 0, 0.02, 0.02, 0.02, 0.02], rtol=0, atol=1e-15)   # every group's local at once
    assert np.all(X[:, 0] <= 0.06)
    assert r.n_model_calls == len(X)


def test_F7_nan_rules():
    model, y, g, A, Bg, a, b = _linear_problem(seed=3)
    y = y.copy()
    y[4] = np.nan
    bad_start = {7}

    def m2(sh, lo):
        out = model(sh, lo)
        if np.allclose(sh, 0) and np.allclose(lo, 0):
            out = out.copy()
            out[list(bad_start)] = np.nan
        return out
    r = F.fit_shared_local(m2, y, np.ones_like(y), g, np.zeros(2), (-10 * np.ones(2), 10 * np.ones(2)),
                           local0=np.zeros((6, 1)), local_bounds=(-10 * np.ones((6, 1)), 10 * np.ones((6, 1))),
                           diff_step_shared=[1e-3, 1e-3], diff_step_local=[1e-3])
    assert not r.kept[4] and not r.kept[7] and r.kept.sum() == y.size - 2
    assert np.isnan(r.residual[4]) and np.isnan(r.residual[7]) and np.all(np.isfinite(r.residual[r.kept]))
    assert r.jac.shape[0] == y.size - 2
    assert r.n_nan_filled == 0
    # a NaN arising later at a kept observation: filled with its last finite prediction and counted
    state = {"n": 0}

    def m3(sh, lo):
        state["n"] += 1
        out = model(sh, lo).copy()
        if state["n"] in (3, 4):
            out[2] = np.nan
        return out
    r3 = F.fit_shared_local(m3, np.nan_to_num(y), np.ones_like(y), g, np.zeros(2), (-10 * np.ones(2), 10 * np.ones(2)),
                            local0=np.zeros((6, 1)), local_bounds=(-10 * np.ones((6, 1)), 10 * np.ones((6, 1))),
                            diff_step_shared=[1e-3, 1e-3], diff_step_local=[1e-3])
    assert r3.kept.all() and r3.n_nan_filled == 2 and np.all(np.isfinite(r3.residual))


def test_F8_no_local_unknowns_and_determinism():
    model, y, g, A, Bg, a, b = _linear_problem(seed=6, L=0)
    kw = dict(diff_step_shared=[1e-3, 1e-3])
    r1 = F.fit_shared_local(model, y, np.ones_like(y), g, np.zeros(2), (-10 * np.ones(2), 10 * np.ones(2)), **kw)
    r2 = F.fit_shared_local(model, y, np.ones_like(y), g, np.zeros(2), (-10 * np.ones(2), 10 * np.ones(2)), **kw)
    assert_allclose(r1.shared, a, rtol=0, atol=1e-8)
    assert r1.local.shape[1] == 0 and r1.jac.shape == (y.size, 2)
    for name in ("shared", "local", "residual", "jac", "cov_cluster", "cov_classical"):
        assert_array_equal(getattr(r1, name), getattr(r2, name))
    assert r1.n_model_calls == r2.n_model_calls and r1.nfev == r2.nfev


@pytest.mark.parametrize("case", ["w_neg", "w_nan", "w_shape", "group_out", "group_neg", "start_out", "local_out",
                                  "no_kept", "shared_bounds_shape", "local0_1d"])
def test_refusals(case):
    model, y, g, A, Bg, a, b = _linear_problem()
    w = np.ones_like(y)
    kw = dict(local0=np.zeros((6, 1)), local_bounds=(-np.ones((6, 1)), np.ones((6, 1))),
              diff_step_shared=[1e-3, 1e-3], diff_step_local=[1e-3])
    sb = (-np.ones(2), np.ones(2))
    s0 = np.zeros(2)
    if case == "w_neg":
        w[3] = -1
    elif case == "w_nan":
        w[3] = np.nan
    elif case == "w_shape":
        w = w[:-1]
    elif case == "group_out":
        g = g.copy()
        g[0] = 6
    elif case == "group_neg":
        g = g.copy()
        g[0] = -1
    elif case == "start_out":
        s0 = np.array([2.0, 0.0])
    elif case == "local_out":
        kw["local0"] = np.full((6, 1), 2.0)
    elif case == "no_kept":
        y = np.full_like(y, np.nan)
    elif case == "shared_bounds_shape":
        sb = (-np.ones(3), np.ones(3))
    elif case == "local0_1d":
        kw["local0"] = np.zeros(6)
    with pytest.raises(ValueError):
        F.fit_shared_local(model, y, w, g, s0, sb, **kw)


# ---------------------------------------------------------------------------------------------------------------
# Amendment 3 (tester, 2026-11-27 22:11; SPEC Block 56): "a step is taken backwards (-h) where x + h would pass the
# upper bound, and where x - h would then pass the lower bound the step is the larger of the two rooms, u - x or
# -(x - l) (no model call outside the box)". Oracles: (F9) every model call of a fit lies inside the box, for boxes
# narrower than the step, shared and local unknowns, the start at either bound or inside; (F10) the probe of the
# first Jacobian (taken about the start when the start is strictly inside, so no trf nudge) is x + h, x - h or the
# endpoint with the larger room, as hand-computed; (F11) on a linear model the Jacobian column is exact whatever
# step is used, so the narrow-box fit still recovers the optimum; (F12) the wide-box behaviour is unchanged
# (covered by F2 and F6 above).
# ---------------------------------------------------------------------------------------------------------------

def _spy_model(N):
    calls = []

    def model(sh, lo):
        calls.append(np.r_[sh, lo.ravel()].copy())
        x = np.arange(4.0)
        out = []
        for gi in range(N):
            out.append(sh[0] * x + lo[gi, 0] + 0.5 * x * sh[0] ** 2)
        return np.concatenate(out)
    return model, calls


@pytest.mark.parametrize("start", ["lo", "hi", "mid", "near_lo", "near_hi"])
def test_F9_no_model_call_outside_a_narrow_box(start):
    N = 3
    model, calls = _spy_model(N)
    group = np.repeat(np.arange(N), 4)
    y = np.linspace(0.0, 1.0, group.size)
    lo_s, hi_s = 0.0, 0.001                    # box 0.001 wide; step 0.005 (5 times the box)
    lo_l, hi_l = -0.002, 0.002                 # box 0.004 wide; step 0.02
    s0 = dict(lo=lo_s, hi=hi_s, mid=0.0005, near_lo=0.0001, near_hi=0.0009)[start]
    l0 = dict(lo=lo_l, hi=hi_l, mid=0.0, near_lo=-0.0019, near_hi=0.0019)[start]
    r = F.fit_shared_local(model, y, np.ones_like(y), group, np.array([s0]), ([lo_s], [hi_s]),
                           local0=np.full((N, 1), l0), local_bounds=(np.full((N, 1), lo_l), np.full((N, 1), hi_l)),
                           diff_step_shared=[0.005], diff_step_local=[0.02], max_nfev=5)
    X = np.array(calls)
    assert np.all(X[:, 0] >= lo_s) and np.all(X[:, 0] <= hi_s), X[:, 0]
    assert np.all(X[:, 1:] >= lo_l) and np.all(X[:, 1:] <= hi_l), X[:, 1:]
    assert np.all(np.isfinite(r.jac))


def _expected_probe(x, h, lo, hi):
    # SPEC Block 56 amendment 3, written out by hand
    if x + h <= hi:
        return x + h
    if x - h >= lo:
        return x - h
    return hi if (hi - x) >= (x - lo) else lo


@pytest.mark.parametrize("x0,h,lo,hi", [(0.0003, 0.005, 0.0, 0.001),    # larger room up: probe at hi
                                        (0.0008, 0.005, 0.0, 0.001),    # larger room down: probe at lo
                                        (0.0100, 0.005, 0.0, 0.012),    # x + h passes hi, x - h inside: -h
                                        (0.0010, 0.005, 0.0, 0.100)])   # forward step
def test_F10_first_jacobian_probe_point(x0, h, lo, hi):
    N = 2
    model, calls = _spy_model(N)
    group = np.repeat(np.arange(N), 4)
    y = np.linspace(0.0, 1.0, group.size)
    F.fit_shared_local(model, y, np.ones_like(y), group, np.array([x0]), ([lo], [hi]),
                       local0=np.zeros((N, 1)), local_bounds=(-np.ones((N, 1)), np.ones((N, 1))),
                       diff_step_shared=[h], diff_step_local=[0.01], max_nfev=1)
    X = np.array(calls)
    # calls: the start; (the residual at the start is the cached one: the start is strictly inside); the first
    # Jacobian's shared probe, then its local probe
    assert_allclose(X[0, 0], x0, rtol=0, atol=0)
    shared_probe = [c for c in X[1:] if c[0] != x0]
    assert len(shared_probe) >= 1
    assert_allclose(shared_probe[0][0], _expected_probe(x0, h, lo, hi), rtol=0, atol=1e-15)


def test_F11_linear_fit_in_a_narrow_box_recovers_the_optimum():
    # y = a x + b_g on four points per group; the optimum a* = 0.0004 lies inside the box [0, 0.001], the step
    # 0.005 is wider than the box. A linear model's difference quotient is exact for any step, so the fit must
    # follow scipy's trf with the ANALYTIC Jacobian iterate for iterate: same solution to 1e-12 (rounding of the
    # quotient only). (trf itself stops at gtol 2.9e-7 short of a* here; that is the solver's tolerance, the same
    # with the exact Jacobian, and not under test.)
    N = 3
    x = np.arange(4.0)
    group = np.repeat(np.arange(N), 4)
    a_true, b_true = 0.0004, np.array([0.1, -0.2, 0.3])
    y = a_true * np.tile(x, N) + b_true[group]

    def model(sh, lo):
        return sh[0] * np.tile(x, N) + lo[group, 0]
    r = F.fit_shared_local(model, y, np.ones_like(y), group, np.array([0.0009]), ([0.0], [0.001]),
                           local0=np.zeros((N, 1)), local_bounds=(-np.ones((N, 1)), np.ones((N, 1))),
                           diff_step_shared=[0.005], diff_step_local=[0.02])
    A = np.zeros((group.size, 1 + N))
    A[:, 0] = np.tile(x, N)
    A[np.arange(group.size), 1 + group] = 1.0
    ref = least_squares(lambda p: y - A @ p, np.r_[0.0009, np.zeros(N)], jac=lambda p: -A,
                        bounds=(np.r_[0.0, -np.ones(N)], np.r_[0.001, np.ones(N)]), method="trf", x_scale="jac",
                        tr_solver="exact")
    assert_allclose(np.r_[r.shared, r.local.ravel()], ref.x, rtol=0, atol=1e-12)
    assert abs(r.shared[0] - a_true) < 1e-6 and r.free.all()
