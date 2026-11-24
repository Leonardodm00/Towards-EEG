"""Spec-derived tests of Block 20, the pure part (local_refine.py; SPEC 2.9 (b) and
the Block 20 entry). Integration tester, 2026-11-24.

Expected values come from SPEC 2.9 (b) written out in this file (a brute-force
re-implementation of the start rule, the initial simplex, the tie rules of the
estimate), from scipy.optimize.minimize called directly (the reference
implementation the spec names), and from closed-form minima of the test
functions. The code under test is never used to produce an expected value.
"""
import math

import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_allclose
from scipy.optimize import minimize

import local_refine as LR

PEN = 1e6


# --------------------------------------------------------------------------- the spec, written out

def ref_starts(X, F, B, n_starts=3, dmin=0.1, penalty=PEN):
    """SPEC 2.9 (b) 'Starts', 0-based, brute force."""
    X, F, B = np.asarray(X, float), np.asarray(F, float), np.asarray(B, float)
    U = (X - B[:, 0]) / (B[:, 1] - B[:, 0])
    order = sorted([j for j in range(len(F)) if np.isfinite(F[j]) and F[j] < penalty],
                   key=lambda j: (F[j], j))
    out = []
    for j in order:
        if len(out) == n_starts:
            break
        if all(np.max(np.abs(U[j] - U[k])) >= dmin for k in out):
            out.append(j)
    return out


def ref_simplex(u0, step=0.05):
    p = len(u0)
    s = [np.array(u0, float)]
    for i in range(p):
        v = np.array(u0, float)
        v[i] = u0[i] + step if u0[i] + step <= 1.0 else u0[i] - step
        s.append(v)
    return np.array(s)


class Counter:
    def __init__(self, f):
        self.f, self.n, self.pts = f, 0, []

    def __call__(self, x):
        self.n += 1
        self.pts.append(list(x))
        return float(self.f(np.asarray(x, float)))


def quad(c):
    c = np.asarray(c, float)
    return lambda q: float(np.sum((np.asarray(q, float) - c) ** 2))


# --------------------------------------------------------------------------- starts

def test_select_starts_against_brute_force_random():
    rng = np.random.default_rng(11)
    for trial in range(200):
        p = int(rng.integers(1, 7))
        N = int(rng.integers(1, 40))
        lo = rng.normal(size=p) * 3
        B = np.c_[lo, lo + rng.uniform(0.1, 5.0, p)]
        X = B[:, 0] + rng.random((N, p)) * (B[:, 1] - B[:, 0])
        # coarse losses so that ties occur; some penalised / non-finite
        F = np.round(rng.random(N) * 5) / 5
        F[rng.random(N) < 0.15] = PEN
        F[rng.random(N) < 0.05] = np.nan
        F[rng.random(N) < 0.05] = np.inf
        n_st = int(rng.integers(1, 5))
        dmin = float(rng.choice([0.05, 0.1, 0.3, 1.0]))
        cfg = LR.RefineConfig(n_starts=n_st, min_start_dist_unit=dmin)
        got = LR.select_starts(X.tolist(), F.tolist(), [tuple(b) for b in B], cfg)
        assert got == ref_starts(X, F, B, n_st, dmin), trial


def test_select_starts_hand_cases():
    B = [(0.0, 10.0), (-1.0, 1.0)]
    X = [[1, 0], [5, 0], [1.5, 0.05], [9, 0.9], [5, -0.5]]
    # ties by index: 0 and 4 equal loss -> 0 first; 2 is within 0.1 (unit) of 0 -> skipped
    F = [0.5, 0.7, 0.6, 0.9, 0.5]
    assert LR.select_starts(X, F, B) == [0, 4, 1]
    # >= d_min qualifies (closed): u-distance exactly 0.1 on axis 0 (1 -> 2 on a box of 10)
    assert LR.select_starts([[1, 0], [2, 0]], [1.0, 2.0], B) == [0, 1]
    # penalty: >= 1e6 is left out, just below is kept
    assert LR.select_starts([[1, 0], [5, 0]], [PEN, PEN * (1 - 1e-12)], B) == [1]
    assert LR.select_starts([[1, 0], [5, 0]], [PEN, np.nan], B) == []


# --------------------------------------------------------------------------- one start = scipy's Nelder-Mead

def test_one_start_is_scipy_nelder_mead_in_unit_coordinates():
    """With one start, refine's recorded evaluations are exactly those of
    scipy.optimize.minimize(method='Nelder-Mead') called as 2.9 (b) states,
    mapped q = lo + u (hi - lo)."""
    rng = np.random.default_rng(3)
    p = 4
    B = np.c_[np.log([0.3, 1e3, 50.0, 1e-5]), np.log([3.0, 1e5, 1e3, 1e-3])]
    c = B[:, 0] + (0.2 + 0.6 * rng.random(p)) * (B[:, 1] - B[:, 0])
    f = quad(c)
    X = B[:, 0] + rng.random((12, p)) * (B[:, 1] - B[:, 0])
    F = np.array([f(x) for x in X])
    cfg = LR.RefineConfig(n_starts=1, maxfev_per_start=60)
    res = LR.refine(Counter(f), X.tolist(), F.tolist(), [tuple(b) for b in B], cfg)

    j0 = int(np.argmin(F))
    u0 = (X[j0] - B[:, 0]) / (B[:, 1] - B[:, 0])
    ref_pts = []

    def fu(u):
        q = B[:, 0] + np.asarray(u) * (B[:, 1] - B[:, 0])
        ref_pts.append(q.copy())
        return f(q)
    r = minimize(fu, u0, method="Nelder-Mead", bounds=[(0, 1)] * p,
                 options=dict(initial_simplex=ref_simplex(u0), maxfev=60, xatol=1e-3,
                              fatol=1e-4 * F[j0], adaptive=False))
    assert res.n_evals == len(ref_pts) == r.nfev
    # q = lo + u (hi - lo) then a clip into the box: identical arithmetic except the clip, so 1e-12 absolute
    assert_allclose(res.x_evals, np.array(ref_pts), rtol=0, atol=1e-12)
    assert_allclose(res.f_evals, [f(q) for q in ref_pts], rtol=1e-13)
    assert res.starts[0]["nit"] == r.nit and res.starts[0]["status"] == r.status
    # the first p + 1 recorded points are the initial simplex of 2.9 (b)
    assert_allclose(res.x_evals[:p + 1], B[:, 0] + ref_simplex(u0) * (B[:, 1] - B[:, 0]), atol=1e-12)


def test_simplex_flips_on_upper_face_and_stays_in_box():
    B = [(0.0, 1.0)] * 3
    c = [2.0, 0.5, -1.0]                     # outside on two axes
    f = quad(c)
    X = [[1.0, 0.97, 0.0], [0.2, 0.2, 0.2]]
    F = [f(x) for x in X]
    res = LR.refine(Counter(f), X, F, B, LR.RefineConfig(n_starts=1))
    sim = ref_simplex(np.array([1.0, 0.97, 0.0]))
    assert_allclose(res.x_evals[:4], sim, atol=1e-15)
    assert sim[1, 0] == pytest.approx(0.95) and sim[2, 1] == pytest.approx(0.92)
    assert np.all(res.x_evals >= 0.0) and np.all(res.x_evals <= 1.0)
    # the minimum on the box is (1, 0.5, 0): on two faces
    assert res.x[0] == pytest.approx(1.0, abs=1e-9) and res.x[2] == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------- the estimate

def test_estimate_is_lowest_over_all_and_bookkeeping():
    rng = np.random.default_rng(0)
    c = 0.15 + 0.7 * rng.random(5)
    X = rng.random((30, 5))
    f = quad(c)
    F = np.array([f(x) for x in X])
    cnt = Counter(f)
    res = LR.refine(cnt, X.tolist(), F.tolist(), [(0.0, 1.0)] * 5)
    assert res.n_evals == cnt.n == len(res.f_evals) == sum(s["nfev"] for s in res.starts)
    assert res.x_evals.dtype == np.float64 and res.x_evals.shape == (res.n_evals, 5)
    assert res.start_of_eval.dtype == np.int64
    assert res.fun == min(float(np.min(res.f_evals)), float(np.min(F)))
    assert res.fun_search == float(np.min(F)) and res.x_search == X[int(np.argmin(F))].tolist()
    assert res.improved and res.best_start >= 0
    i = int(np.argmin(res.f_evals))
    assert res.x == res.x_evals[i].tolist() and res.best_start == int(res.start_of_eval[i])
    # each start's 'fun' is the lowest of its own recorded evaluations
    for k, s in enumerate(res.starts):
        assert s["fun"] == float(np.min(res.f_evals[res.start_of_eval == k]))
    # R1a numbers (reported): ratio and distance in unit coordinates
    print("R1a ratio %.3g, distance %.3g, n_evals %d" % (res.fun / res.fun_search,
                                                          np.max(np.abs(np.array(res.x) - c)), res.n_evals))
    assert res.fun < 1e-3 * res.fun_search
    assert np.max(np.abs(np.array(res.x) - c)) < 2e-2
    assert res.n_evals <= 3 * 100 + 3 * 5 + 3 * 5   # scipy may overshoot maxfev by one iteration (<= p+1)


def test_ties_keep_the_search_point():
    """A local evaluation EQUAL to min_j L_j does not count as improved (2.9 (b): strictly lower)."""
    f = lambda q: 1.0                                     # flat: every local value equals the search's best
    X = [[0.1, 0.1], [0.9, 0.9], [0.5, 0.5]]
    res = LR.refine(Counter(f), X, [1.0, 1.0, 1.0], [(0.0, 1.0)] * 2)
    assert res.n_evals > 0
    assert not res.improved and res.best_start == -1
    assert res.x == [0.1, 0.1] and res.fun == 1.0


def test_exact_minimum_not_improved():
    c = [0.3, 0.6]
    f = quad(c)
    X = [c, [0.9, 0.1], [0.1, 0.9]]
    res = LR.refine(Counter(f), X, [f(x) for x in X], [(0.0, 1.0)] * 2)
    assert res.fun_search == 0.0
    assert not res.improved and res.best_start == -1 and res.x == res.x_search == c


def test_two_basins_best_start_named():
    # shallow basin at a (value 1), deep at b (value 0); search best near a
    a, b = np.array([0.2, 0.2]), np.array([0.8, 0.8])
    f = lambda q: min(1.0 + 10 * np.sum((q - a) ** 2), 10 * np.sum((q - b) ** 2))
    # losses (by hand): 1.001, 1.005, 1.8 (deep basin side), 2.0; the second is within 0.1 of the first
    X = np.array([[0.21, 0.2], [0.22, 0.21], [0.5, 0.5], [0.5, 0.1]])
    F = [float(f(x)) for x in X]
    assert int(np.argmin(F)) == 0 and F[2] < F[3]
    res = LR.refine(Counter(f), X.tolist(), F, [(0.0, 1.0)] * 2)
    starts = [s["index"] for s in res.starts]
    assert 2 in starts
    assert np.max(np.abs(np.array(res.x) - b)) < 0.02
    assert res.starts[res.best_start]["index"] == 2


def test_penalty_everywhere_and_nan_objective():
    f = lambda q: 1e6 if np.any(np.abs(np.asarray(q) - 0.5) > 0.01) else 0.5
    X = [[0.5, 0.5], [0.1, 0.1]]
    res = LR.refine(Counter(f), X, [0.5, 1e6], [(0.0, 1.0)] * 2)
    assert res.x == [0.5, 0.5] and res.fun == 0.5 and not res.improved
    # an objective that returns NaN over part of the box: no crash, the estimate is finite
    g = lambda q: float("nan") if q[0] > 0.6 else float(np.sum((np.asarray(q) - 0.7) ** 2))
    X = [[0.58, 0.5], [0.2, 0.2]]
    res = LR.refine(Counter(g), X, [g(np.array(x)) for x in X], [(0.0, 1.0)] * 2)
    assert np.isfinite(res.fun) and res.x[0] <= 0.6 + 1e-12
    # every global value penalised: no start, estimate the search's (first index)
    res = LR.refine(Counter(g), [[0.1, 0.1], [0.2, 0.2]], [PEN, PEN], [(0.0, 1.0)] * 2)
    assert res.n_evals == 0 and res.starts == [] and res.x == [0.1, 0.1] and res.fun == PEN
    assert res.x_evals.shape == (0, 2)


def test_determinism():
    rng = np.random.default_rng(5)
    X = rng.random((15, 3))
    f = quad([0.3, 0.4, 0.5])
    F = [f(x) for x in X]
    r1 = LR.refine(Counter(f), X.tolist(), F, [(0.0, 1.0)] * 3)
    r2 = LR.refine(Counter(f), X.tolist(), F, [(0.0, 1.0)] * 3)
    assert_array_equal(r1.x_evals, r2.x_evals)
    assert r1.x == r2.x


def test_inputs_not_mutated():
    X = np.random.default_rng(1).random((6, 2))
    F = np.array([quad([0.5, 0.5])(x) for x in X])
    X0, F0 = X.copy(), F.copy()
    LR.refine(Counter(quad([0.5, 0.5])), X, F, np.array([[0.0, 1.0], [0.0, 1.0]]))
    assert_array_equal(X, X0)
    assert_array_equal(F, F0)


@pytest.mark.parametrize("kw", [dict(n_starts=0), dict(n_starts=1.5), dict(n_starts=True),
                                dict(maxfev_per_start=0), dict(min_start_dist_unit=0.0),
                                dict(min_start_dist_unit=1.5), dict(simplex_step_unit=1.0),
                                dict(simplex_step_unit=0.0), dict(xatol_unit=0.0), dict(fatol_rel=-1.0),
                                dict(penalty=0.0), dict(penalty=float("nan")), dict(xatol_unit=float("inf"))])
def test_config_refusals(kw):
    with pytest.raises(ValueError):
        LR.RefineConfig(**kw)


def test_config_defaults():
    c = LR.RefineConfig()
    assert (c.n_starts, c.maxfev_per_start, c.min_start_dist_unit, c.simplex_step_unit, c.xatol_unit,
            c.fatol_rel, c.penalty) == (3, 100, 0.1, 0.05, 1e-3, 1e-4, 1e6)


def test_bad_bounds_refused():
    with pytest.raises(ValueError):
        LR.refine(Counter(quad([0.5])), [[0.5]], [0.0], [(1.0, 1.0)])
    with pytest.raises(ValueError):
        LR.refine(Counter(quad([0.5])), [[0.5, 0.5]], [0.0], [(0.0, 1.0)])


# --------------------------------------------------------------------------- wrap_minimizer

class _Res:
    pass


def test_wrap_minimizer_same_object_and_untouched_fields():
    from skopt.space import Real, Integer
    rng = np.random.default_rng(2)
    f = quad([0.3, 0.7])
    X = rng.random((10, 2)).tolist()
    F = [f(np.array(x)) for x in X]
    canned = _Res()
    canned.x_iters, canned.func_vals = list(X), np.array(F)
    canned.x, canned.fun = X[int(np.argmin(F))], float(np.min(F))
    canned.models, canned.space = ["m"], "space"
    calls = []

    def fake(func, dimensions, **kw):
        calls.append(kw)
        return canned
    w = LR.wrap_minimizer(fake)
    got = w(lambda x: f(np.array(x)), [Real(0.0, 1.0), Real(0.0, 1.0)], n_calls=10, random_state=1)
    assert got is canned and calls == [dict(n_calls=10, random_state=1)]
    assert got.x_iters == X and list(got.func_vals) == F and got.models == ["m"] and got.space == "space"
    assert got.fun < min(F) and got.fun == got.refine.fun and got.x == got.refine.x
    # 'dimensions' given by keyword, as fit_one_cell passes it
    got2 = w(lambda x: f(np.array(x)), dimensions=[Real(0.0, 1.0), Real(0.0, 1.0)])
    assert got2.refine.n_evals > 0
    # a non-Real dimension: ValueError before the minimizer runs
    calls.clear()
    with pytest.raises(ValueError):
        w(lambda x: 0.0, [Real(0.0, 1.0), Integer(0, 3)])
    assert calls == []


def test_wrap_minimizer_with_gp_minimize():
    from skopt import gp_minimize
    from skopt.space import Real
    f = quad([0.15, 0.8, 0.45])
    w = LR.wrap_minimizer(gp_minimize, LR.RefineConfig(maxfev_per_start=30))
    r = w(lambda x: f(np.array(x)), [Real(0.0, 1.0)] * 3, n_calls=12, n_initial_points=6, random_state=0)
    assert r.fun <= min(r.func_vals) and len(r.x_iters) == 12
    assert r.fun == pytest.approx(f(np.array(r.x)), rel=0, abs=0)
