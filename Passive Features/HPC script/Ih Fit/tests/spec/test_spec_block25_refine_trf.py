"""Spec-derived tests of Block 25 (local_refine.TrfConfig, refine_trf,
wrap_minimizer_trf; SPEC 2.10 (c) and the Block 25 entry). Integration tester,
2026-11-25.

Oracles: closed-form least-squares minima (numpy.linalg.lstsq), scipy's own
least_squares run with the analytic Jacobian as the reference solver, the
bookkeeping rules of 2.10 (c) written out (estimate = recorded point of lowest
L2, ties earlier start then earlier evaluation, starts by 2.9 (b)), the
Jacobian of the Block 24 rule (one residuals call of p rows when f0 is reused).
"""
import types

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.optimize import least_squares

import local_refine as LR


class Counter:
    """residuals(Q) with a record of every call (rows asked)."""

    def __init__(self, fun):
        self.fun = fun
        self.calls = []

    def __call__(self, Q):
        Q = np.asarray(Q, dtype=float)
        self.calls.append(Q.copy())
        return np.array([self.fun(q) for q in Q], dtype=float)

    @property
    def rows(self):
        return sum(c.shape[0] for c in self.calls)


def _box(p, lo=-2.0, hi=3.0):
    return [(lo, hi)] * p


def _linear_problem(seed=0, p=4, N=30):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(N, p))
    q_true = rng.uniform(-1.0, 2.0, p)
    y = A @ q_true + 0.05 * rng.normal(size=N)
    return A, y, (lambda q: A @ np.asarray(q) - y)


def _search(fun, bounds, n=25, seed=1):
    rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
    X = lo + rng.random((n, len(bounds))) * (hi - lo)
    F = np.array([float(np.sqrt(np.sum(fun(x) ** 2))) for x in X])
    return X, F


def test_defaults_of_the_spec():
    c = LR.TrfConfig()
    assert (c.n_starts, c.min_start_dist_unit, c.max_nfev, c.fd_step_unit, c.xtol, c.ftol, c.gtol, c.penalty) == \
        (3, 0.1, 30, 1e-3, 1e-4, 1e-8, 1e-8, 1e6)


@pytest.mark.parametrize("kw", [dict(n_starts=0), dict(max_nfev=0), dict(n_starts=1.5), dict(n_starts=True),
                                dict(fd_step_unit=0.0), dict(fd_step_unit=0.6), dict(min_start_dist_unit=0.0),
                                dict(xtol=0.0), dict(ftol=-1.0), dict(gtol=np.nan), dict(penalty=np.inf)])
def test_config_refusals(kw):
    with pytest.raises(ValueError):
        LR.TrfConfig(**kw)


def test_linear_least_squares_reaches_the_closed_form_minimum():
    A, y, f = _linear_problem()
    bounds = _box(4)
    X, F = _search(f, bounds)
    res = Counter(f)
    r = LR.refine_trf(res, X, F, bounds, LR.TrfConfig(max_nfev=50))
    q_ls = np.linalg.lstsq(A, y, rcond=None)[0]
    assert np.all((q_ls > -2) & (q_ls < 3))                       # interior minimum
    u_hat = (np.array(r.x) + 2.0) / 5.0
    u_ls = (q_ls + 2.0) / 5.0
    # xtol 1e-4 in unit coordinates stops the solver once the step is below
    # 1e-4 * (1e-4 + |u|); a linear problem is solved in a step or two: 1e-6.
    assert np.max(np.abs(u_hat - u_ls)) < 1e-6
    assert_allclose(r.l2, np.sum(f(q_ls) ** 2), rtol=1e-9)
    assert all(st["converged"] for st in r.starts)


def test_bookkeeping_of_2_10_c():
    A, y, f = _linear_problem(seed=4)
    bounds = _box(4)
    X, F = _search(f, bounds, n=40, seed=5)
    res = Counter(f)
    cfg = LR.TrfConfig(n_starts=3, max_nfev=10)
    r = LR.refine_trf(res, X, F, bounds, cfg, loss_of_row=lambda row: float(np.sum(np.abs(row))))
    # n_sims == len(l2_evals) == rows computed by `residuals`
    assert r.n_sims == len(r.l2_evals) == res.rows == r.x_evals.shape[0] == len(r.start_of_eval)
    assert r.n_evals == r.n_sims
    # every recorded L2 is the squared norm of the row at that point
    for q, l2 in zip(r.x_evals, r.l2_evals):
        assert_allclose(l2, np.sum(f(q) ** 2), rtol=1e-12)
    # the estimate is the recorded point of lowest L2, the first such
    k = int(np.argmin(r.l2_evals))
    assert_array_equal(r.x, r.x_evals[k])
    assert r.l2 == r.l2_evals[k]
    assert r.fun == r.l == float(np.sum(np.abs(f(r.x_evals[k]))))
    # starts: 2.9 (b)'s rule (select_starts with the same distance and penalty)
    idx = LR.select_starts(X, F, bounds, LR.RefineConfig(n_starts=3, min_start_dist_unit=0.1, penalty=1e6))
    assert [st["index"] for st in r.starts] == list(idx)
    assert idx[0] == int(np.argmin(F))
    # the first record is the first start (the search's best point), x_search there
    assert_allclose(r.x_evals[0], X[idx[0]], rtol=0, atol=1e-12)
    assert_allclose(r.x_search, X[idx[0]], atol=1e-12)
    assert r.l2_search == r.l2_evals[0]
    assert r.improved == (r.l2 < r.l2_search)
    assert r.best_start == (int(r.start_of_eval[k]) if r.improved else -1)
    # start_of_eval non-decreasing (starts run in order); per start counts add up
    assert np.all(np.diff(r.start_of_eval) >= 0)
    assert sum(st["n_sims"] for st in r.starts) == r.n_sims
    # each start's x is its own lowest recorded point
    for s, st in enumerate(r.starts):
        own = np.flatnonzero(r.start_of_eval == s)
        j = own[int(np.argmin(r.l2_evals[own]))]
        assert_array_equal(st["x"], r.x_evals[j])
        assert st["l2"] == r.l2_evals[j]
        assert st["converged"] == (st["status"] in (1, 2, 3, 4))
        assert len(st["active_mask"]) == 4


def test_each_jacobian_is_one_call_of_p_rows():
    """2.10 (b), (c): every perturbed point of one Jacobian in ONE residuals call;
    with the start's row reused as f0 (the point is the last one computed) the call
    holds p rows. Calls of one row are the solver's trial points."""
    f = lambda q: np.array([q[0] - 1.0, 10 * (q[1] - q[0] ** 2), q[2] - 0.5, 0.3 * q[0] * q[2]])
    bounds = _box(3)
    X, F = _search(f, bounds, n=15, seed=2)
    res = Counter(f)
    r = LR.refine_trf(res, X, F, bounds, LR.TrfConfig(n_starts=1, max_nfev=20))
    sizes = [c.shape[0] for c in res.calls]
    assert set(sizes) <= {1, 3, 4}, sizes
    n_jac = sum(1 for s in sizes if s in (3, 4))
    assert n_jac == r.starts[0]["njev"]
    # the start's own row first (one row), then the first Jacobian reuses it (p rows)
    assert sizes[0] == 1 and sizes[1] == 3


def test_nonlinear_against_scipy_with_the_analytic_jacobian():
    """Rosenbrock residuals in q on a box; the reference is scipy least_squares from
    the same start in the same unit coordinates with the exact Jacobian."""
    def f(q):
        return np.array([10.0 * (q[1] - q[0] ** 2), 1.0 - q[0], 0.5 * (q[2] - q[1])])
    bounds = [(-1.5, 2.0), (-0.5, 3.0), (-1.0, 2.5)]
    lo = np.array([b[0] for b in bounds]); w = np.array([b[1] - b[0] for b in bounds])
    X = np.array([[0.2, 0.9, 1.7], [-1.0, 2.5, 0.0], [1.8, 0.0, 2.0]])
    F = np.array([float(np.linalg.norm(f(x))) for x in X])
    r = LR.refine_trf(Counter(f), X, F, bounds, LR.TrfConfig(n_starts=1, max_nfev=200))
    fu = lambda u: f(lo + u * w)

    def ju(u):
        q = lo + u * w
        Jq = np.array([[-20.0 * q[0], 10.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -0.5, 0.5]])
        return Jq * w
    ref = least_squares(fu, (X[int(np.argmin(F))] - lo) / w, jac=ju, bounds=(0, 1), method="trf",
                        x_scale=1.0, xtol=1e-10, ftol=1e-12, gtol=1e-12)
    # the global minimum (1, 1, 1) is in the box; the FD refinement stops on xtol 1e-4 (unit)
    assert_allclose(lo + ref.x * w, [1, 1, 1], atol=1e-6)
    assert np.max(np.abs((np.array(r.x) - (lo + ref.x * w)) / w)) < 1e-3
    assert r.starts[0]["converged"]


def test_minimum_outside_the_box_lies_on_the_face():
    f = lambda q: np.array([q[0] - 5.0, q[1] - 0.3, 2.0 * (q[2] + 0.1)])     # q0* = 5 > hi = 3
    bounds = _box(3)
    X, F = _search(f, bounds, n=20, seed=8)
    res = Counter(f)
    r = LR.refine_trf(res, X, F, bounds, LR.TrfConfig(max_nfev=60))
    assert abs(r.x[0] - 3.0) < 1e-6 * 5.0
    assert_allclose(r.x[1:], [0.3, -0.1], atol=5e-4)
    for c in res.calls:                          # every point asked lies in the box
        assert np.all(c >= -2.0) and np.all(c <= 3.0)
    best = r.starts[int(r.best_start) if r.best_start >= 0 else 0]
    assert best["active_mask"][0] == 1           # upper bound active on axis 0 (scipy: +1)


def test_jtj_and_singular_values_at_the_solver_endpoint():
    A, y, f = _linear_problem(seed=11, p=3)
    bounds = [(-3.0, 4.0), (-2.0, 3.0), (-5.0, 5.0)]
    X, F = _search(f, bounds, n=20, seed=12)
    r = LR.refine_trf(Counter(f), X, F, bounds, LR.TrfConfig(n_starts=1, max_nfev=20))
    st = r.starts[0]
    # linear residuals: the FD Jacobian in q is A up to roundoff of a forward difference (1e-8 rel)
    assert_allclose(st["jtj"], A.T @ A, rtol=1e-6)
    ev = np.sort(np.linalg.eigvalsh(A.T @ A))[::-1]
    assert_allclose(st["sv"], np.sqrt(ev), rtol=1e-6)
    assert np.all(np.diff(st["sv"]) <= 0)


def test_no_start_when_every_global_value_is_penalised():
    f = lambda q: np.asarray(q, dtype=float)
    X = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    F = np.array([1e6, 2e6, np.inf])
    res = Counter(f)
    r = LR.refine_trf(res, X, F, _box(2))
    assert r.n_sims == 0 and res.calls == [] and r.starts == []
    assert r.improved is False and r.best_start == -1
    assert len(r.x) == 2


def test_tie_takes_the_earlier_start_and_evaluation():
    """A residual function constant everywhere: every recorded L2 ties; the estimate is
    record 0 (the first start's own point), not improved."""
    f = lambda q: np.array([1.0, 2.0])
    bounds = _box(2)
    X = np.array([[0.0, 0.0], [2.0, 2.0], [-1.5, 2.5]])
    F = np.array([1.0, 2.0, 3.0])
    r = LR.refine_trf(Counter(f), X, F, bounds, LR.TrfConfig(n_starts=3, max_nfev=5))
    assert_array_equal(r.x, X[0])
    assert r.improved is False and r.best_start == -1


def test_failed_row_region_no_crash():
    fail = np.sqrt(1e6 / 3)

    def f(q):
        if np.linalg.norm(q - 0.5) > 1.0:
            return np.full(3, fail)
        return np.array([q[0] - 0.4, q[1] - 0.6, 0.2 * q[0]])
    bounds = _box(2)
    X, F = _search(lambda q: f(q), bounds, n=30, seed=3)
    r = LR.refine_trf(Counter(f), X, F, bounds, LR.TrfConfig(max_nfev=30))
    assert np.isfinite(r.l2) and r.l2 < 1.0


def test_determinism():
    A, y, f = _linear_problem(seed=21)
    X, F = _search(f, _box(4), seed=22)
    r1 = LR.refine_trf(Counter(f), X, F, _box(4))
    r2 = LR.refine_trf(Counter(f), X, F, _box(4))
    assert_array_equal(r1.x_evals, r2.x_evals)
    assert_array_equal(r1.l2_evals, r2.l2_evals)
    assert r1.x == r2.x


def test_wrap_minimizer_trf_contract():
    from skopt.space import Real
    f = lambda q: np.array([q[0] - 0.7, 2.0 * (q[1] + 0.2)])
    dims = [Real(-2.0, 3.0, name="a"), Real(-2.0, 3.0, name="b")]
    X, F = _search(f, [(-2.0, 3.0)] * 2, n=12, seed=4)
    order = []
    canned = types.SimpleNamespace(x=list(X[np.argmin(F)]), fun=float(F.min()), x_iters=[list(x) for x in X],
                                   func_vals=F.copy(), models=["m"], space="S")

    def fake_min(func, dimensions, **kw):
        order.append("min")
        return canned

    class Res(Counter):
        def l_and_l2(self, row):
            return float(np.sum(np.abs(row))), float(np.sum(row ** 2))

    holder = {}

    def get_res():
        order.append("get")
        holder["r"] = Res(f)
        return holder["r"]

    w = LR.wrap_minimizer_trf(fake_min, get_res, LR.TrfConfig(n_starts=2, max_nfev=20))
    out = w(lambda x: float(np.linalg.norm(f(np.asarray(x)))), dims, n_calls=12)
    assert out is canned
    assert order == ["min", "get"]
    assert_allclose(out.x, [0.7, -0.2], atol=5e-4)
    assert out.fun == out.refine.l == float(np.sum(np.abs(f(np.asarray(out.x)))))
    assert out.models == ["m"] and out.space == "S"
    assert_array_equal(out.func_vals, F)
    # a residual object without l_and_l2: fun is func(x)
    canned2 = types.SimpleNamespace(x=list(X[0]), fun=1.0, x_iters=[list(x) for x in X], func_vals=F.copy(),
                                    models=[], space=None)
    w2 = LR.wrap_minimizer_trf(lambda func, d, **k: canned2, lambda: Counter(f), LR.TrfConfig(n_starts=1))
    func = lambda x: 123.0 + x[0]
    o2 = w2(func, dims)
    assert o2.fun == func(o2.x)
    # a non-Real dimension: ValueError before the minimizer runs
    from skopt.space import Integer
    called = []
    w3 = LR.wrap_minimizer_trf(lambda *a, **k: called.append(1), get_res)
    with pytest.raises(ValueError):
        w3(func, [Integer(0, 3)])
    assert called == []
