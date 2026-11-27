"""Smoke test for step 2's estimator -- Block 56 in specs/SPEC.md (section 5, "Kernel calibration, step 2";
D-273.3, amendment 1 of 2026-11-27): analysis.step2_fit.fit_shared_local, weighted least squares with shared and
per-group unknowns, its structured 2-point Jacobian with absolute steps and the two covariances of the shared
unknowns.

Checks (the SPEC's oracles F1 to F8)
    test_known_answer   F1: a linear model y = A a + B_g b_g without noise, shared and local to 1e-8
    test_reference      F2: a parabola in the plane offset (curvature and offset shared, centre local) against
                        least_squares with its own jac="2-point"; the callable's Jacobian against the analytic one
                        to the forward difference's truncation error; the steps taken at a second iterate equal the
                        absolute steps asked for. F3: cov_cluster and cov_classical against statsmodels OLS
                        (cov_type "cluster", use_correction False; and its non-robust covariance)
    test_convergence    not a discretisation (SkipTest); the O(h) truncation of the difference is bounded in F2
    test_invariants     F4: Monte Carlo, 400 datasets with noise correlated inside groups: SD of the estimates
                        against the mean se_cluster, and se_classical too small by the derived factor.
                        F5: bounds respected, free False at an active bound, NaN covariance rows there
    test_contract       shapes, dtypes, cost and residual definitions; F6: S + L model calls per Jacobian, the
                        step reversed at an upper bound (shared and local); a box narrower than the step: the
                        larger room taken and no model call outside the box (amendment 3)
    test_determinism    F8: two runs bit for bit
    test_edge_cases     F7: the NaN rules (y NaN, model NaN at the start, NaN arising later); F8: L = 0 against
                        numpy's lstsq; every ValueError of the entry

Run (from Passive Features/Diameter Re-measurement/)
    python tests/smoke/test_smoke_step2_fit.py
    python -m pytest tests/smoke/test_smoke_step2_fit.py -q

Pure ASCII, for the cluster.
"""
from __future__ import annotations

import importlib.metadata
import platform
import re
import sys
import time
import traceback
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal  # noqa: F401
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from allen_diameter.analysis.step2_fit import FitResult, fit_shared_local  # noqa: E402

SEED = 12345
REPORT_PACKAGES = ("numpy", "scipy", "statsmodels")
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
_SAVED_ARTIFACTS = []  # files written by check_close during this run


# --------------------------------------------------------------- helpers ---

def make_rng(seed: int = SEED) -> np.random.Generator:
    """Fresh generator, to be passed explicitly to the code under test."""
    return np.random.default_rng(seed)


def check_close(name, actual, expected, rtol=1e-12, atol=0.0):
    """numpy.testing.assert_allclose that keeps the evidence on failure.

    On failure both arrays are saved to ARTIFACT_DIR/<name>.npz before the
    AssertionError propagates. Justify rtol and atol at the call site
    (float64 roundoff, discretisation-error estimate, statistical spread).
    """
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


# ---------------------------------------------------------------- checks ---

class Logged:
    """Wraps a model(shared, local) and keeps a copy of every input it was called with."""

    def __init__(self, f):
        self.f = f
        self.log = []

    def __call__(self, a, b):
        self.log.append((np.array(a, dtype=float), np.array(b, dtype=float)))
        return self.f(a, b)


def jac_blocks(log, h_s, h_l):
    """The Jacobian blocks found in a call log: a base call followed by S + L probes, probe s differing from the
    base in shared[s] alone by +-h_s[s], probe S + l in local[:, l] alone by +-h_l[l] for every group. Returns a
    list of (index of the base, the signed steps of the probes: list of arrays). The tolerance on a step is the
    rounding of (x + h) - x: 1e-9 relative is far above it (|x| / h <= 1e4 here) and far below any other
    difference between calls."""
    S, L = len(h_s), len(h_l)
    out = []
    i = 0
    while i + S + L < len(log):
        a0, b0 = log[i]
        steps = []
        ok = True
        for s in range(S):
            a, b = log[i + 1 + s]
            d = a - a0
            others = np.delete(d, s)
            if not (np.all(others == 0) and np.all(b == b0) and abs(abs(d[s]) - h_s[s]) <= 1e-9 * h_s[s]):
                ok = False
                break
            steps.append(np.array([d[s]]))
        if ok:
            for l in range(L):
                a, b = log[i + 1 + S + l]
                d = b - b0
                others = np.delete(d, l, axis=1)
                if not (np.all(a == a0) and np.all(others == 0)
                        and np.all(np.abs(np.abs(d[:, l]) - h_l[l]) <= 1e-9 * h_l[l])):
                    ok = False
                    break
                steps.append(d[:, l].copy())
        if ok and S + L > 0:
            out.append((i, steps))
            i += 1 + S + L
        else:
            i += 1
    return out


def linear_problem(rng, N=5, m=8, S=2, L=2, noise=0.0):
    """y = A a + B_g b_g: A (n, S) dense, B the local design (n, L) whose row j multiplies b_{g(j)}."""
    n = N * m
    group = np.repeat(np.arange(N), m)
    A = rng.normal(size=(n, S))
    B = np.column_stack([np.ones(n)] + [rng.normal(size=n) for _ in range(L - 1)])[:, :L]
    a_true = rng.normal(size=S)
    b_true = rng.normal(size=(N, L))

    def model(a, b):
        return A @ a + np.sum(B * b[group], axis=1)

    y = model(a_true, b_true) + noise * rng.normal(size=n)
    return model, y, group, A, B, a_true, b_true


def test_known_answer():
    # F1. The forward difference of a linear model is exact up to rounding, eps |f| / h ~ 2e-16 * 10 / 1e-3 =
    # 2e-12 per Jacobian entry, and trf ends at the least-squares solution to about the same order relative to
    # the data; 1e-8 (the SPEC's) is a margin of 1e3 over both.
    rng = np.random.default_rng(SEED)
    model, y, group, A, B, a_true, b_true = linear_problem(rng)
    N, L = b_true.shape
    S = a_true.size
    r = fit_shared_local(model, y, np.ones(y.size), group, np.zeros(S), (np.full(S, -10.0), np.full(S, 10.0)),
                         local0=np.zeros((N, L)), local_bounds=(np.full((N, L), -10.0), np.full((N, L), 10.0)),
                         diff_step_shared=np.full(S, 1e-3), diff_step_local=np.full(L, 1e-3))
    assert r.success, r.message
    check_close("F1_shared", r.shared, a_true, rtol=0.0, atol=1e-8)
    check_close("F1_local", r.local, b_true, rtol=0.0, atol=1e-8)


def parabola_problem(rng, N=4, sigma=0.02):
    """Step 2's near stage in miniature: y_{g,k} = c0 + c1 (k - m_g)^2 on k = -3..3, shared (c0, c1), local m_g."""
    k = np.tile(np.arange(-3.0, 4.0), N)
    group = np.repeat(np.arange(N), 7)
    c_true = np.array([0.1, 0.25])
    m_true = rng.uniform(-0.8, 0.8, size=(N, 1))

    def model(a, b):
        return a[0] + a[1] * (k - b[group, 0]) ** 2

    y = model(c_true, m_true) + sigma * rng.normal(size=k.size)
    w = rng.uniform(0.5, 2.0, size=k.size)
    return model, y, w, group, k


def test_reference():
    rng = np.random.default_rng(SEED + 1)
    model, y, w, group, k = parabola_problem(rng)
    N = 4
    h_s = np.array([0.005, 0.002])
    h_l = np.array([0.02])
    lo_s, hi_s = np.array([-1.0, 0.01]), np.array([1.0, 2.0])
    lo_l, hi_l = np.full((N, 1), -2.0), np.full((N, 1), 2.0)
    a0, b0 = np.array([0.0, 0.5]), np.zeros((N, 1))
    lm = Logged(model)
    r = fit_shared_local(lm, y, w, group, a0, (lo_s, hi_s), local0=b0, local_bounds=(lo_l, hi_l),
                         diff_step_shared=h_s, diff_step_local=h_l)
    assert r.success, r.message

    # F2 (a): the same problem by least_squares with its own dense jac="2-point" (relative steps, scipy's
    # default size ~1.5e-8 |x|). Both converge to a point where (their Jacobian)' r = 0; a forward difference
    # with step h biases that point by O(h x residual): with the steps above (0.02 in m) the two solutions
    # differ by ~1e-4 (printed), so the solver comparison is made with small absolute steps, 1e-7, whose
    # truncation (c1 h ~ 3e-8) and rounding (eps |f| / h ~ 5e-9) errors in J move the solution by < 1e-8; the
    # rest is the solvers' tolerances (ftol = xtol = 1e-8 relative); 1e-6, the SPEC's.
    sw = np.sqrt(w)

    def fun(x):
        return sw * (y - model(x[:2], x[2:].reshape(N, 1)))

    ref = least_squares(fun, np.concatenate([a0, b0.ravel()]), jac="2-point", method="trf", x_scale="jac",
                        bounds=(np.concatenate([lo_s, lo_l.ravel()]), np.concatenate([hi_s, hi_l.ravel()])))
    r_small = fit_shared_local(model, y, w, group, a0, (lo_s, hi_s), local0=b0, local_bounds=(lo_l, hi_l),
                               diff_step_shared=np.full(2, 1e-7), diff_step_local=np.full(1, 1e-7))
    check_close("F2_solution", np.concatenate([r_small.shared, r_small.local.ravel()]), ref.x, rtol=0.0,
                atol=1e-6)
    print("    F2: solution with steps (0.005, 0.002; 0.02) differs from scipy's by %.2e (O(h) bias of the "
          "difference); with steps 1e-7 by %.2e"
          % (float(np.max(np.abs(np.concatenate([r.shared, r.local.ravel()]) - ref.x))),
             float(np.max(np.abs(np.concatenate([r_small.shared, r_small.local.ravel()]) - ref.x)))))

    # F2 (b): the callable's Jacobian at the solution against the analytic one of r = sqrt(w) (y - yhat):
    # d r / d c0 = -sqrt(w), d r / d c1 = -sqrt(w) (k - m)^2, d r / d m_g = sqrt(w) 2 c1 (k - m_g) on group g.
    # Forward-difference truncation: c0 and c1 enter linearly (exact up to rounding); in m the error is
    # sqrt(w) c1 h (half the second derivative, 2 c1, times h), with a sign; plus rounding eps |f| / h < 1e-12.
    c, m = r.shared, r.local[:, 0]
    J_an = np.zeros((k.size, 2 + N))
    J_an[:, 0] = -sw
    J_an[:, 1] = -sw * (k - m[group]) ** 2
    J_an[np.arange(k.size), 2 + group] = sw * 2.0 * c[1] * (k - m[group])
    bound = np.zeros_like(J_an)
    bound[:, :2] = 1e-10
    bound[np.arange(k.size), 2 + group] = sw * c[1] * h_l[0] * (1.0 + 1e-6) + 1e-10
    err = np.abs(r.jac - J_an)
    assert np.all(err <= bound), "F2 Jacobian: max excess %.3e" % float(np.max(err - bound))
    # and the truncation is actually present at about its predicted size (the difference is a forward one)
    trunc = err[np.arange(k.size), 2 + group] / (sw * c[1] * h_l[0])
    assert np.max(trunc) > 0.99, "F2: no O(h) truncation seen (max ratio %.3f)" % float(np.max(trunc))

    # F2 (c): the steps actually taken at a second iterate equal the absolute steps asked for (scipy's own
    # diff_step would scale them with |x|: amendment 1)
    blocks = jac_blocks(lm.log, h_s, h_l)
    assert len(blocks) >= 2, "fewer than two Jacobians found in the call log (%d)" % len(blocks)
    i2, steps2 = blocks[1]
    a_base, b_base = lm.log[i2]
    assert not (np.array_equal(a_base, a0) and np.array_equal(b_base, b0)), "second Jacobian at the start"
    for s in range(2):
        check_close("F2_step_shared_%d" % s, abs(steps2[s][0]), h_s[s], rtol=1e-9)
    check_close("F2_step_local", np.abs(steps2[2]), np.full(N, h_l[0]), rtol=1e-9)
    print("    second Jacobian at shared %s, local %s: steps %s, %s (asked %s, %s)"
          % (np.round(a_base, 4), np.round(b_base[:, 0], 4), [float(abs(s[0])) for s in steps2[:2]],
             np.abs(steps2[2]).round(6).tolist(), h_s.tolist(), h_l.tolist()))

    # F3: the covariances against statsmodels on a linear model with a local intercept per group
    try:
        import statsmodels.api as sm
    except ImportError:
        raise unittest.SkipTest("F3 needs statsmodels (pip install statsmodels), not installed")
    rng = np.random.default_rng(SEED + 2)
    N, m_obs = 12, 6
    n = N * m_obs
    group = np.repeat(np.arange(N), m_obs)
    A = rng.normal(size=(n, 2))
    u = rng.normal(size=N)[group]
    y3 = A @ np.array([1.0, -0.5]) + rng.normal(size=N)[group] + (0.5 + np.abs(A[:, 0])) * rng.normal(size=n) + u

    def model3(a, b):
        return A @ a + b[group, 0]

    r3 = fit_shared_local(model3, y3, np.ones(n), group, np.zeros(2), (np.full(2, -50.0), np.full(2, 50.0)),
                          local0=np.zeros((N, 1)), local_bounds=(np.full((N, 1), -50.0), np.full((N, 1), 50.0)),
                          diff_step_shared=np.full(2, 1e-3), diff_step_local=np.full(1, 1e-3))
    X = np.column_stack([A, (group[:, None] == np.arange(N)[None, :]).astype(float)])
    ols = sm.OLS(y3, X)
    f_cl = ols.fit(cov_type="cluster", cov_kwds={"groups": group, "use_correction": False})
    f_nr = ols.fit()
    # 1e-8 relative (the SPEC's): the Jacobian is exact up to ~1e-12 and the solution agrees with statsmodels'
    # QR/pinv one to ~1e-10 relative (checked first), so the covariances agree far inside it
    check_close("F3_params", r3.shared, f_cl.params[:2], rtol=1e-8, atol=1e-10)
    check_close("F3_cov_cluster", r3.cov_cluster, np.asarray(f_cl.cov_params())[:2, :2], rtol=1e-8, atol=0.0)
    check_close("F3_cov_classical", r3.cov_classical, np.asarray(f_nr.cov_params())[:2, :2], rtol=1e-8, atol=0.0)


def test_convergence():
    raise unittest.SkipTest("not a discretisation; the O(h) truncation of the forward difference is bounded in "
                            "test_reference (F2)")


def test_invariants():
    # F4. y_{g,i} = a0 + a1 x_g + u_g + e_{g,i}, x_g constant within a group, G groups of m, Var u = rho sigma^2,
    # Var e = (1 - rho) sigma^2: the errors' covariance is Sigma = sigma^2 [(1 - rho) I + rho Bd], Bd the
    # block-diagonal of m x m blocks of ones. Every column of X is constant within groups, so Bd X = m X and
    #   Var(a_hat) = (X'X)^-1 X' Sigma X (X'X)^-1 = sigma^2 [1 + (m - 1) rho] (X'X)^-1    (exact).
    # The classical estimate s^2 (X'X)^-1 has E[RSS] = tr(M Sigma), M = I - H, H the hat matrix (rank p);
    # tr(M I) = n - p, tr(M Bd) = tr(Bd) - tr(H Bd) = n - m p (H Bd = m H since Bd X = m X), so
    #   E[s^2] = sigma^2 [(1 - rho) + rho (n - m p) / (n - p)],
    # and the expected ratio of the classical SE to the true SD is
    #   f = sqrt( [(1 - rho) + rho (n - m p)/(n - p)] / [1 + (m - 1) rho] ),
    # compared with sqrt(mean se_classical^2) / SD (the mean of s^2 is what the derivation gives exactly).
    # Tolerance 15 % (the SPEC's): the Monte Carlo SD from 400 datasets has relative SE 1/sqrt(2 * 399) = 3.5 %,
    # so 15 % is about 4 SE; CR0's small-sample bias with G = 40 groups is of order 1/G = 2.5 %, inside it.
    rng = np.random.default_rng(SEED + 3)
    G, m, rho, sigma, R = 40, 8, 0.5, 1.0, 400
    n, p = G * m, 2
    group = np.repeat(np.arange(G), m)
    xg = rng.normal(size=G)[group]
    a_true = np.array([0.3, 1.2])

    def model(a, b):
        return a[0] + a[1] * xg

    est = np.empty((R, 2))
    se_cl = np.empty((R, 2))
    se_cs = np.empty((R, 2))
    for t in range(R):
        e = np.sqrt(rho) * sigma * rng.normal(size=G)[group] + np.sqrt(1 - rho) * sigma * rng.normal(size=n)
        y = model(a_true, None) + e
        r = fit_shared_local(model, y, np.ones(n), group, np.zeros(2), (np.full(2, -10.0), np.full(2, 10.0)),
                             diff_step_shared=np.full(2, 1e-3))
        est[t], se_cl[t], se_cs[t] = r.shared, r.se_cluster, r.se_classical
    sd = est.std(axis=0, ddof=1)
    ratio_cl = se_cl.mean(axis=0) / sd
    ratio_cs = np.sqrt((se_cs ** 2).mean(axis=0)) / sd
    f = np.sqrt(((1 - rho) + rho * (n - m * p) / (n - p)) / (1 + (m - 1) * rho))
    print("    F4: SD %s, mean se_cluster / SD %s, rms se_classical / SD %s, expected factor %.4f"
          % (sd.round(4).tolist(), ratio_cl.round(3).tolist(), ratio_cs.round(3).tolist(), f))
    check_close("F4_cluster", ratio_cl, np.ones(2), rtol=0.15)
    check_close("F4_classical", ratio_cs, np.full(2, f), rtol=0.15)

    # F5. The unbounded optimum has a1 = 2 > 1.5 = its upper bound, and group 2's intercept 3 > 2.5 = its bound.
    rng = np.random.default_rng(SEED + 4)
    N, mm = 5, 10
    group = np.repeat(np.arange(N), mm)
    x = rng.normal(size=N * mm)
    b_true = np.array([0.0, 1.0, 3.0, -1.0, 0.5])

    def model5(a, b):
        return a[0] * x ** 2 + a[1] * x + b[group, 0]

    y = model5(np.array([0.2, 2.0]), b_true[:, None]) + 0.05 * rng.normal(size=x.size)
    hi_l = np.full((N, 1), 2.5)
    r = fit_shared_local(model5, y, np.ones(x.size), group, np.zeros(2), (np.array([-5.0, -5.0]),
                         np.array([5.0, 1.5])), local0=np.zeros((N, 1)), local_bounds=(np.full((N, 1), -5.0), hi_l),
                         diff_step_shared=np.full(2, 1e-3), diff_step_local=np.full(1, 1e-3))
    assert np.all(r.shared <= np.array([5.0, 1.5])) and np.all(r.shared >= -5.0), r.shared
    assert np.all(r.local <= hi_l) and np.all(r.local >= -5.0), r.local
    # at an active bound trf ends on the bound up to its xtol (1e-8 relative)
    check_close("F5_at_bound", [r.shared[1], r.local[2, 0]], [1.5, 2.5], rtol=1e-6)
    expect_free = np.ones(2 + N, dtype=bool)
    expect_free[1] = False
    expect_free[2 + 2] = False
    assert_array_equal(r.free, expect_free)
    assert np.all(np.isnan(r.cov_cluster[1, :])) and np.all(np.isnan(r.cov_cluster[:, 1]))
    assert np.all(np.isnan(r.cov_classical[1, :])) and np.all(np.isnan(r.cov_classical[:, 1]))
    assert np.isfinite(r.cov_cluster[0, 0]) and np.isfinite(r.cov_classical[0, 0])
    assert np.isnan(r.se_cluster[1]) and np.isfinite(r.se_cluster[0])


def test_contract():
    rng = np.random.default_rng(SEED + 5)
    model, y, w, group, k = parabola_problem(rng)
    N, S, L = 4, 2, 1
    h_s, h_l = np.array([0.005, 0.002]), np.array([0.02])
    lm = Logged(model)
    # F6: shared[1] and group 1's centre start half a step below their upper bounds, so the first Jacobian must
    # take those steps backwards
    hi_s = np.array([1.0, 2.0])
    hi_l = np.full((N, 1), 2.0)
    a0 = np.array([0.0, hi_s[1] - h_s[1] / 2])
    b0 = np.zeros((N, 1))
    b0[1, 0] = hi_l[1, 0] - h_l[0] / 2
    r = fit_shared_local(lm, y, w, group, a0, (np.array([-1.0, 0.01]), hi_s), local0=b0,
                         local_bounds=(np.full((N, 1), -2.0), hi_l), diff_step_shared=h_s, diff_step_local=h_l)
    assert isinstance(r, FitResult)
    n = y.size
    for name, shape in (("shared", (S,)), ("local", (N, L)), ("residual", (n,)), ("kept", (n,)),
                        ("jac", (n, S + N * L)), ("free", (S + N * L,)), ("cov_cluster", (S, S)),
                        ("cov_classical", (S, S)), ("se_cluster", (S,)), ("se_classical", (S,))):
        assert getattr(r, name).shape == shape, (name, getattr(r, name).shape)
    assert r.kept.dtype == bool and r.free.dtype == bool and r.jac.dtype == np.float64
    for name in ("n_model_calls", "nfev", "n_nan_filled", "status"):
        assert isinstance(getattr(r, name), int), name
    assert isinstance(r.message, str) and isinstance(r.success, bool) and isinstance(r.cost, float)
    # the definitions: residual = sqrt(w) (y - model) at the solution, cost = 0.5 sum r^2 (rounding only)
    check_close("residual_def", r.residual, np.sqrt(w) * (y - model(r.shared, r.local)), rtol=0.0, atol=1e-13)
    check_close("cost_def", r.cost, 0.5 * np.sum(r.residual ** 2), rtol=1e-14)
    # F6: every model call is counted; the calls are the start, the residual evaluations and Jacobian blocks of
    # exactly S + L probes each
    assert r.n_model_calls == len(lm.log), (r.n_model_calls, len(lm.log))
    blocks = jac_blocks(lm.log, h_s, h_l)
    in_blocks = sum(S + L for _ in blocks)
    others = len(lm.log) - in_blocks - len(blocks)
    assert len(blocks) >= 2 and others >= 0, (len(blocks), others)
    print("    F6: %d model calls = %d Jacobians x (S + L = %d) + %d bases + %d other residual calls; nfev %d"
          % (r.n_model_calls, len(blocks), S + L, len(blocks), others, r.nfev))
    i0, steps0 = blocks[0]
    assert i0 == 0, "the first Jacobian is not at the start"
    assert steps0[0][0] > 0 and steps0[1][0] < 0, (steps0[0], steps0[1])
    expect = np.full(N, h_l[0])
    expect[1] = -h_l[0]
    check_close("F6_reversed_local", steps0[2], expect, rtol=1e-9)
    check_close("F6_reversed_shared", steps0[1], [-h_s[1]], rtol=1e-9)

    # F6, amendment 3: a box narrower than the step. shared[1] in [0.01, 0.013] with step 0.005 (rooms from the
    # start 0.011: 0.002 up, 0.001 down, so the probe goes to the upper bound 0.013), the centres in [-0.005,
    # 0.005] with step 0.02 (from 0.002: 0.003 up, 0.007 down, so the probe goes to the lower bound -0.005).
    # Every model call of the whole fit must stay inside the box (no tolerance: the probe points are clipped).
    lm2 = Logged(model)
    lo_s2, hi_s2 = np.array([-1.0, 0.01]), np.array([1.0, 0.013])
    lo_l2, hi_l2 = np.full((N, 1), -0.005), np.full((N, 1), 0.005)
    h_s2 = np.array([0.005, 0.005])
    r2 = fit_shared_local(lm2, y, w, group, np.array([0.0, 0.011]), (lo_s2, hi_s2), local0=np.full((N, 1), 0.002),
                          local_bounds=(lo_l2, hi_l2), diff_step_shared=h_s2, diff_step_local=h_l)
    outside = [i for i, (a, b) in enumerate(lm2.log)
               if np.any(a < lo_s2) or np.any(a > hi_s2) or np.any(b < lo_l2) or np.any(b > hi_l2)]
    assert not outside, "model calls outside the box: %s" % outside[:5]
    assert r2.n_model_calls == len(lm2.log)
    # the first Jacobian (calls 1 to 3, after the start at call 0): shared[0] by +h, shared[1] to its upper bound,
    # every centre to its lower bound
    assert lm2.log[1][0][0] == 0.0 + h_s2[0] and lm2.log[1][0][1] == 0.011, lm2.log[1][0]
    assert lm2.log[2][0][1] == hi_s2[1], lm2.log[2][0]
    assert_array_equal(lm2.log[3][1], lo_l2)
    print("    F6 narrow box: %d model calls, none outside the box" % len(lm2.log))


def test_determinism():
    # F8 (determinism): no randomness inside the estimator, so two runs agree bit for bit
    rng = np.random.default_rng(SEED + 6)
    model, y, w, group, k = parabola_problem(rng)
    args = dict(local0=np.zeros((4, 1)), local_bounds=(np.full((4, 1), -2.0), np.full((4, 1), 2.0)),
                diff_step_shared=np.array([0.005, 0.002]), diff_step_local=np.array([0.02]))
    r1 = fit_shared_local(model, y, w, group, np.array([0.0, 0.5]), ([-1.0, 0.01], [1.0, 2.0]), **args)
    r2 = fit_shared_local(model, y, w, group, np.array([0.0, 0.5]), ([-1.0, 0.01], [1.0, 2.0]), **args)
    for name in ("shared", "local", "residual", "jac", "cov_cluster", "cov_classical"):
        assert_array_equal(getattr(r1, name), getattr(r2, name), err_msg=name)
    assert (r1.n_model_calls, r1.nfev, r1.cost) == (r2.n_model_calls, r2.nfev, r2.cost)


def test_edge_cases():
    rng = np.random.default_rng(SEED + 7)
    N, m = 4, 6
    n = N * m
    group = np.repeat(np.arange(N), m)
    x = rng.normal(size=n)
    y = 0.7 + 1.3 * x + 0.01 * rng.normal(size=n)
    bnd = (np.full(2, -10.0), np.full(2, 10.0))
    hs = np.full(2, 1e-3)

    # F8: L = 0 against numpy's lstsq. 1e-8: the difference is exact for a linear model up to rounding and trf
    # stops within ftol/xtol 1e-8 relative of the minimiser of a well-conditioned 2-unknown problem
    def lin(a, b):
        assert b.shape == (N, 0)
        return a[0] + a[1] * x

    r = fit_shared_local(lin, y, np.ones(n), group, np.zeros(2), bnd, diff_step_shared=hs)
    ref = np.linalg.lstsq(np.column_stack([np.ones(n), x]), y, rcond=None)[0]
    check_close("F8_L0", r.shared, ref, rtol=0.0, atol=1e-8)
    assert r.local.shape == (N, 0) and r.jac.shape == (n, 2) and r.free.shape == (2,)

    # F7 (a), (b): y NaN at observation 0 and the model NaN at the start at observation 1: both dropped
    y7 = y.copy()
    y7[0] = np.nan
    j_late = 5

    def nan_model(a, b):
        out = a[0] + a[1] * x
        out[1] = np.nan
        # F7 (c): observation 5's prediction is the constant 5.0 while a1 < 0.5, NaN once a1 passes 0.5 (it does:
        # the start is 0, the optimum 1.3); its last finite prediction is therefore exactly 5.0
        out[j_late] = 5.0 if a[1] < 0.5 else np.nan
        return out

    w7 = np.ones(n)
    w7[j_late] = 1e-6  # small, so the constant 5.0 does not pull the fit
    r = fit_shared_local(nan_model, y7, w7, group, np.zeros(2), bnd, diff_step_shared=hs)
    assert not r.kept[0] and not r.kept[1] and r.kept[2:].all(), r.kept[:4]
    assert np.isnan(r.residual[0]) and np.isnan(r.residual[1]) and np.all(np.isfinite(r.residual[2:]))
    assert r.jac.shape == (n - 2, 2), r.jac.shape
    assert r.n_nan_filled > 0, r.n_nan_filled
    check_close("F7_filled", r.residual[j_late], np.sqrt(w7[j_late]) * (y7[j_late] - 5.0), rtol=1e-14)
    assert r.shared[1] > 0.5, r.shared  # the fit did go where the NaNs arise
    # F7 (d): nothing kept
    with_err(lambda: fit_shared_local(lambda a, b: np.full(n, np.nan), y, np.ones(n), group, np.zeros(2), bnd),
             "no kept")

    # the ValueErrors of the entry
    good = dict(local0=np.zeros((N, 1)), local_bounds=(np.full((N, 1), -1.0), np.full((N, 1), 1.0)))

    def lm(a, b):
        return a[0] + a[1] * x + b[group, 0]

    with_err(lambda: fit_shared_local(lm, y, np.ones(n - 1), group, np.zeros(2), bnd, **good), "w shape")
    with_err(lambda: fit_shared_local(lm, y, np.ones(n), group[:-1], np.zeros(2), bnd, **good), "group shape")
    with_err(lambda: fit_shared_local(lm, y, np.ones(n), group, np.zeros(2), bnd, local0=np.zeros((N, 1)),
                                      local_bounds=(np.zeros((N, 2)), np.ones((N, 2)))), "local bounds shape")
    with_err(lambda: fit_shared_local(lm, y, np.ones(n), group, np.zeros(3), bnd, **good), "shared bounds shape")
    w_neg = np.ones(n)
    w_neg[3] = -1.0
    with_err(lambda: fit_shared_local(lm, y, w_neg, group, np.zeros(2), bnd, **good), "negative weight")
    w_nan = np.ones(n)
    w_nan[3] = np.nan
    with_err(lambda: fit_shared_local(lm, y, w_nan, group, np.zeros(2), bnd, **good), "NaN weight")
    with_err(lambda: fit_shared_local(lm, y, np.ones(n), group, np.array([0.0, 11.0]), bnd, **good),
             "shared start outside")
    with_err(lambda: fit_shared_local(lm, y, np.ones(n), group, np.zeros(2), bnd, local0=np.full((N, 1), 2.0),
                                      local_bounds=good["local_bounds"]), "local start outside")
    g_bad = group.copy()
    g_bad[0] = N
    with_err(lambda: fit_shared_local(lm, y, np.ones(n), g_bad, np.zeros(2), bnd, **good), "group index")
    g_neg = group.copy()
    g_neg[0] = -1
    with_err(lambda: fit_shared_local(lm, y, np.ones(n), g_neg, np.zeros(2), bnd, **good), "negative group")
    with_err(lambda: fit_shared_local(lambda a, b: np.zeros(n + 1), y, np.ones(n), group, np.zeros(2), bnd,
                                      **good), "model output shape")


def with_err(f, what):
    try:
        f()
    except ValueError:
        return
    raise AssertionError("no ValueError for %s" % what)


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
