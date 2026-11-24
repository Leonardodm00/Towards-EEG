"""Smoke test for the local refinement -- Block 20 in specs/SPEC.md (R1-R10), SPEC 2.9 (b).

Pure: no NEURON, no file I/O by the code under test.

Checks
    test_R1_quadratic_5d      f(q) = sum (q_i - c_i)^2, p = 5, box [0,1]^5, rng = default_rng(0),
                              c = 0.15 + 0.7 rng.random(5), 30 evaluations rng.random((30, 5)).
                              R1a (defaults): fun < 1e-3 x the search's best, max_i |u_i(q_hat) - c_i|
                              < 2e-2, n_evals <= 3 x 100 + 3 x p. R1b (maxfev_per_start = 10000):
                              every start status 0, fun < 1e-5 x the search's best, distance < 2e-3.
    test_R2_face              c outside the box along one axis: the estimate lies on that face
                              (1e-9) and every recorded point is inside the box
    test_R3_select_starts     select_starts against a brute-force re-implementation on 200 random
                              draws (ties, penalised and non-finite values included) and hand cases
    test_R4_initial_simplex   the first p + 1 recorded points of each start are u(s) and
                              u(s) + delta_i e_i, the step flipping on the upper face
    test_R5_bookkeeping       n_evals == len(f_evals) == sum(nfev) == a counter around the
                              objective; fun == min(min f_evals, min func_vals); a search whose
                              best is the exact minimum is not improved (best_start -1, x == x_search)
    test_R6_two_basins        search best in the shallow basin, its third best (> 0.1 away) in the
                              deep one: the estimate is in the deep basin, best_start names it
    test_R7_penalty           an objective at 1e6 off a small region: no crash, the search's estimate
    test_R8_wrap_minimizer    a fake minimizer: same object returned, x / fun replaced, x_iters /
                              func_vals equal to the canned ones, refine attached; skopt.gp_minimize
                              (20 calls) on the R1 function: runs, fun not above the search's
    test_R9_determinism       two calls give identical x_evals and f_evals
    test_R10_refusals         RefineConfig refusals; refine's ValueErrors (shapes, empty, bounds);
                              wrap_minimizer refuses a non-Real dimension

Run (from this folder)
    python smoke_local_refine.py [--only R1,R3]
        Prints the environment and a PASS / FAIL / ERROR / TODO / SKIP table, then the full
        message of every failure. Exit code 1 if any check failed, errored or is still TODO.

Debug
    A failed check_close() saves both arrays to smoke_artifacts/<check name>.npz.

Keep this file pure ASCII (hpc-python-compat).
"""
from __future__ import annotations

import argparse
import importlib.metadata
import math
import platform
import re
import sys
import time
import traceback
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np                         # noqa: E402
from numpy.testing import assert_allclose  # noqa: E402

import local_refine as LR                  # noqa: E402

SEED = 12345
REPORT_PACKAGES = ("numpy", "scipy", "scikit-optimize")
ARTIFACT_DIR = HERE / "smoke_artifacts"
_SAVED_ARTIFACTS = []


def check_close(name, actual, expected, rtol=0.0, atol=0.0):
    """assert_allclose that keeps the evidence (smoke_artifacts/<name>.npz) on failure."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    try:
        assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=name)
    except AssertionError:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        path = ARTIFACT_DIR / ("%s.npz" % re.sub(r"[^A-Za-z0-9_.-]+", "_", name))
        np.savez(path, actual=actual, expected=expected)
        _SAVED_ARTIFACTS.append(path)
        raise


def _quad(c):
    c = np.asarray(c, dtype=float)
    return lambda x: float(np.sum((np.asarray(x, dtype=float) - c) ** 2))


def _r1_problem():
    rng = np.random.default_rng(0)
    c = 0.15 + 0.7 * rng.random(5)
    X = rng.random((30, 5))
    f = _quad(c)
    return c, X, [f(x) for x in X], f, [(0.0, 1.0)] * 5


def _unit(q, bounds):
    b = np.asarray(bounds, dtype=float)
    return (np.asarray(q, dtype=float) - b[:, 0]) / (b[:, 1] - b[:, 0])


# ---------------------------------------------------------------- checks ---
def test_R1_quadratic_5d():
    c, X, F, f, B = _r1_problem()
    p = 5
    best = min(F)
    # R1a: the default budget (thresholds of amendment 1, SPEC Block 20)
    r = LR.refine(f, X, F, B)
    ratio = r.fun / best
    dist = float(np.max(np.abs(_unit(r.x, B) - c)))
    print("  [R1a] search best %.6g; refined %.6g (ratio %.3g); max unit distance %.3g; n_evals %d; "
          "nfev %s; status %s" % (best, r.fun, ratio, dist, r.n_evals, [s["nfev"] for s in r.starts],
                                   [s["status"] for s in r.starts]))
    bad = []
    if not ratio < 1e-3:
        bad.append("R1a ratio %.3g >= 1e-3" % ratio)
    if not dist < 2e-2:
        bad.append("R1a distance %.3g >= 2e-2" % dist)
    if not r.n_evals <= 3 * 100 + 3 * p:
        bad.append("R1a n_evals %d > %d" % (r.n_evals, 3 * 100 + 3 * p))
    # R1b: convergence (every start stops on its tolerances)
    r2 = LR.refine(f, X, F, B, LR.RefineConfig(maxfev_per_start=10000))
    ratio2 = r2.fun / best
    dist2 = float(np.max(np.abs(_unit(r2.x, B) - c)))
    print("  [R1b] refined %.6g (ratio %.3g); max unit distance %.3g; nfev %s; status %s"
          % (r2.fun, ratio2, dist2, [s["nfev"] for s in r2.starts], [s["status"] for s in r2.starts]))
    if not r2.starts or any(int(s["status"]) != 0 for s in r2.starts):
        bad.append("R1b statuses %s" % [s["status"] for s in r2.starts])
    if not ratio2 < 1e-5:
        bad.append("R1b ratio %.3g >= 1e-5" % ratio2)
    if not dist2 < 2e-3:
        bad.append("R1b distance %.3g >= 2e-3" % dist2)
    assert not bad, "; ".join(bad)


def test_R2_face():
    rng = np.random.default_rng(SEED)
    p = 3
    B = [(-2.0, 1.0), (0.5, 4.0), (-1.0, 1.0)]
    b = np.asarray(B)
    c = np.array([2.5, 1.7, 0.2])          # axis 0 beyond its upper face (hi = 1.0)
    f = _quad(c)
    X = b[:, 0] + rng.random((30, p)) * (b[:, 1] - b[:, 0])
    F = [f(x) for x in X]
    r = LR.refine(f, X, F, B)
    bad = []
    if abs(r.x[0] - 1.0) > 1e-9:
        bad.append("estimate axis 0 = %.17g, not on the face 1.0" % r.x[0])
    out = (r.x_evals < b[:, 0]) | (r.x_evals > b[:, 1])
    if np.any(out):
        bad.append("%d recorded coordinates outside the box" % int(out.sum()))
    xh = np.asarray(r.x)
    if np.any(xh < b[:, 0]) or np.any(xh > b[:, 1]):
        bad.append("estimate outside the box %s" % r.x)
    print("  [R2] estimate %s (face at 1.0: |x0 - 1| = %.3g); %d recorded points, all inside: %s"
          % ([round(v, 6) for v in r.x], abs(r.x[0] - 1.0), r.n_evals, not np.any(out)))
    assert not bad, "; ".join(bad)


def _brute_starts(X, F, B, n_starts, dmin, penalty):
    """An obviously-correct re-implementation of the start rule of SPEC 2.9 (b)."""
    X, F, B = np.asarray(X, float), np.asarray(F, float), np.asarray(B, float)
    N = len(F)
    order = []
    for j in range(N):
        if math.isfinite(F[j]) and F[j] < penalty:
            order.append(j)
    # bubble the order by (L, j)
    for a in range(len(order)):
        for k in range(len(order) - 1 - a):
            j1, j2 = order[k], order[k + 1]
            if (F[j2] < F[j1]) or (F[j2] == F[j1] and j2 < j1):
                order[k], order[k + 1] = j2, j1
    U = (X - B[:, 0]) / (B[:, 1] - B[:, 0])
    chosen = []
    for j in order:
        if len(chosen) == n_starts:
            break
        far = True
        for k in chosen:
            d = max(abs(U[j][i] - U[k][i]) for i in range(X.shape[1]))
            if d < dmin:
                far = False
        if far:
            chosen.append(j)
    return chosen


def test_R3_select_starts():
    rng = np.random.default_rng(SEED)
    bad = []
    n_nontrivial = 0
    for t in range(200):
        p = int(rng.integers(1, 7))
        N = int(rng.integers(1, 40))
        lo = rng.normal(size=p)
        B = np.stack([lo, lo + 0.1 + 3 * rng.random(p)], axis=1)
        X = B[:, 0] + rng.random((N, p)) * (B[:, 1] - B[:, 0])
        F = np.round(rng.random(N), 1)                         # many ties
        F[rng.random(N) < 0.15] = 1e6                          # penalised
        F[rng.random(N) < 0.05] = 2e7
        F[rng.random(N) < 0.08] = np.nan
        F[rng.random(N) < 0.04] = np.inf
        cfg = LR.RefineConfig(n_starts=int(rng.integers(1, 5)),
                              min_start_dist_unit=float(rng.choice([0.05, 0.1, 0.3, 1.0])))
        got = LR.select_starts(X, F, B, cfg)
        want = _brute_starts(X, F, B, cfg.n_starts, cfg.min_start_dist_unit, cfg.penalty)
        n_nontrivial += len(want) > 1
        if got != want:
            bad.append("draw %d: %s != brute %s" % (t, got, want))
    # hand cases (box [0, 1]^2)
    B = [(0.0, 1.0), (0.0, 1.0)]
    X = [[0.5, 0.5], [0.1, 0.1], [0.9, 0.9], [0.15, 0.15], [0.8, 0.2]]   # 3 is 0.05 from 1
    cases = [
        ("ties by index", [0.2, 0.1, 0.1, 0.1, 0.5], [1, 2, 0]),
        ("penalised and non-finite out", [1e6, np.nan, 0.3, np.inf, 0.4], [2, 4]),
        ("all penalised", [1e6, 2e6, np.nan, np.inf, 1e6], []),
    ]
    for label, F, want in cases:
        got = LR.select_starts(X, F, B)
        if got != want:
            bad.append("%s: %s != %s" % (label, got, want))
    # closer than 0.1: fewer than three
    Xc = [[0.5, 0.5], [0.55, 0.5], [0.5, 0.58], [0.45, 0.45], [0.7, 0.5]]
    got = LR.select_starts(Xc, [0.1, 0.2, 0.3, 0.4, 0.5], B)
    if got != [0, 4]:
        bad.append("close points: %s != [0, 4]" % got)
    # exactly d_min away counts as far enough (">= d_min"); 0.25 and 0.5 are exact in binary
    got = LR.select_starts([[0.5, 0.5], [0.75, 0.5]], [0.1, 0.2], B,
                           LR.RefineConfig(min_start_dist_unit=0.25))
    if got != [0, 1]:
        bad.append("distance exactly d_min: %s != [0, 1]" % got)
    print("  [R3] 200 random draws agree with the brute force (%d with more than one start); 5 hand cases"
          % n_nontrivial)
    assert not bad, "; ".join(bad[:5])


def test_R4_initial_simplex():
    p = 4
    B = [(-1.0, 1.0), (0.0, 2.0), (10.0, 11.0), (-5.0, 5.0)]
    b = np.asarray(B)
    c = np.array([0.2, 0.3, 10.4, 1.0])
    f = _quad(c)
    # start 0 on the upper face of axis 2 and near the upper face of axis 1 (u = 0.97)
    X = np.array([[0.0, 1.94, 11.0, 0.0],           # best: u = (0.5, 0.97, 1.0, 0.5)
                  [-0.8, 0.2, 10.1, -4.0],          # far enough: a second start
                  [0.6, 1.0, 10.9, 3.0]])
    F = [f(x) for x in X]
    cfg = LR.RefineConfig(n_starts=2, maxfev_per_start=20)
    r = LR.refine(f, X, F, B, cfg)
    bad = []
    for k, st in enumerate(r.starts):
        u0 = _unit(X[st["index"]], B)
        delta = np.where(u0 + 0.05 <= 1.0, 0.05, -0.05)
        want_u = np.vstack([u0] + [u0 + delta[i] * np.eye(p)[i] for i in range(p)])
        want_q = b[:, 0] + want_u * (b[:, 1] - b[:, 0])
        rec = r.x_evals[r.start_of_eval == k][:p + 1]
        try:
            # q -> u -> q round trip: a few ulp of the box widths
            check_close("R4_simplex_start%d" % k, rec, want_q, rtol=0.0, atol=1e-12)
        except AssertionError as exc:
            bad.append(str(exc).splitlines()[0])
        if k == 0:
            print("  [R4] start 0 at u = %s; steps %s (axis 2 on the face, axis 1 at 0.97: both flip)"
                  % (np.round(u0, 3).tolist(), delta.tolist()))
            if not (delta[1] < 0 and delta[2] < 0 and delta[0] > 0 and delta[3] > 0):
                bad.append("the flipped axes are not 1 and 2: %s" % delta)
    if len(r.starts) != 2:
        bad.append("%d starts, 2 expected" % len(r.starts))
    assert not bad, "; ".join(bad)


def test_R5_bookkeeping():
    c, X, F, f, B = _r1_problem()
    count = [0]

    def counted(x):
        count[0] += 1
        return f(x)
    r = LR.refine(counted, X, F, B)
    bad = []
    nfev = sum(int(s["nfev"]) for s in r.starts)
    if not (r.n_evals == len(r.f_evals) == nfev == count[0] == r.x_evals.shape[0] == len(r.start_of_eval)):
        bad.append("counts n_evals %d, len(f_evals) %d, sum nfev %d, counter %d"
                   % (r.n_evals, len(r.f_evals), nfev, count[0]))
    if r.fun != min(float(np.min(r.f_evals)), min(F)):
        bad.append("fun %r != min(min f_evals, min func_vals)" % r.fun)
    if r.x_evals.dtype != np.float64 or r.f_evals.dtype != np.float64 or r.start_of_eval.dtype != np.int64:
        bad.append("dtypes %s %s %s" % (r.x_evals.dtype, r.f_evals.dtype, r.start_of_eval.dtype))
    for k, st in enumerate(r.starts):
        own = r.f_evals[r.start_of_eval == k]
        if st["fun"] != float(np.min(own)) or len(own) != st["nfev"]:
            bad.append("start %d: fun %r vs min %r, nfev %d vs %d" % (k, st["fun"], np.min(own), st["nfev"], len(own)))
    if not r.improved or r.best_start < 0:
        bad.append("R1 problem not improved")
    # the search's best is the exact minimum: f = 0 at an evaluated point, nothing strictly lower
    c2 = X[7].copy()
    f2 = _quad(c2)
    F2 = [f2(x) for x in X]
    r2 = LR.refine(f2, X, F2, B)
    if r2.improved or r2.best_start != -1 or r2.x != [float(v) for v in X[7]] or r2.fun != 0.0:
        bad.append("exact minimum: improved %s, best_start %d, x %s, fun %r"
                   % (r2.improved, r2.best_start, r2.x, r2.fun))
    if r2.x != r2.x_search:
        bad.append("x != x_search when not improved")
    print("  [R5] %d local evaluations = sum nfev = counter; exact minimum kept (%d local evaluations made)"
          % (r.n_evals, r2.n_evals))
    assert not bad, "; ".join(bad)


def _two_basin(x):
    """Deep basin at (0.2, 0.2), value -1; shallow at (0.8, 0.8), value -0.5 (box [0,1]^2)."""
    x = np.asarray(x, dtype=float)
    d1 = np.sum((x - 0.2) ** 2)
    d2 = np.sum((x - 0.8) ** 2)
    return float(1.0 - 1.0 * np.exp(-d1 / 0.01) - 0.5 * np.exp(-d2 / 0.01))


def test_R6_two_basins():
    B = [(0.0, 1.0), (0.0, 1.0)]
    X = np.array([[0.78, 0.81],      # 0: best of the search, shallow basin
                  [0.83, 0.80],      # 1: second best, < 0.1 from 0 (no start)
                  [0.30, 0.27],      # 2: third best, deep basin, > 0.1 away
                  [0.50, 0.50], [0.05, 0.95], [0.95, 0.05]])
    F = [_two_basin(x) for x in X]
    order = sorted(range(len(F)), key=lambda j: F[j])
    assert order[:3] == [0, 1, 2], "fixture order %s (F %s)" % (order, F)
    r = LR.refine(_two_basin, X, F, B)
    pos = [s["index"] for s in r.starts]
    bad = []
    if 2 not in pos:
        bad.append("evaluation 2 is not a start: %s" % pos)
    elif r.best_start != pos.index(2):
        bad.append("best_start %d, the deep basin's start is %d" % (r.best_start, pos.index(2)))
    if np.max(np.abs(np.asarray(r.x) - 0.2)) > 0.02 or not r.improved:
        bad.append("estimate %s (improved %s) not in the deep basin" % (r.x, r.improved))
    print("  [R6] starts %s; estimate %s, L %.6g (search best %.6g); best_start %d"
          % (pos, np.round(r.x, 4).tolist(), r.fun, min(F), r.best_start))
    assert not bad, "; ".join(bad)


def test_R7_penalty():
    B = [(0.0, 1.0)] * 3

    def pen(x):
        x = np.asarray(x, dtype=float)
        d = float(np.sum((x - 0.5) ** 2))
        return 0.3 + d if d < 0.01 ** 2 else 1e6
    rng = np.random.default_rng(SEED)
    X = rng.random((15, 3))
    X[4] = [0.5, 0.5, 0.505]                         # the one point inside the region
    F = [pen(x) for x in X]
    r = LR.refine(pen, X, F, B)
    bad = []
    if r.improved or r.x != r.x_search or r.fun != min(F):
        # an improvement inside the region would be legitimate only below min(F); report it
        if not (r.improved and r.fun < min(F)):
            bad.append("estimate %s fun %r, search %s %r" % (r.x, r.fun, r.x_search, r.fun_search))
    if [s["index"] for s in r.starts] != [4]:
        bad.append("starts %s, only evaluation 4 is below the penalty" % [s["index"] for s in r.starts])
    print("  [R7] one start (index 4); %d local evaluations, %d at the penalty; estimate L %.6g, improved %s"
          % (r.n_evals, int(np.sum(r.f_evals >= 1e6)), r.fun, r.improved))
    # all penalised: no start, the search's estimate
    r2 = LR.refine(lambda x: 1e6, X, [1e6] * len(X), B)
    if r2.n_evals or r2.starts or r2.improved or r2.x != [float(v) for v in X[0]]:
        bad.append("all penalised: n_evals %d, starts %d, improved %s" % (r2.n_evals, len(r2.starts), r2.improved))
    if r2.x_evals.shape != (0, 3):
        bad.append("empty x_evals shape %s" % (r2.x_evals.shape,))
    assert not bad, "; ".join(bad)


class _Canned:
    pass


def test_R8_wrap_minimizer():
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    c, X, F, f, B = _r1_problem()
    dims = [Real(lo, hi, name="a%d" % i) for i, (lo, hi) in enumerate(B)]
    canned = _Canned()
    canned.x_iters = [list(map(float, x)) for x in X]
    canned.func_vals = np.asarray(F)
    j = int(np.argmin(F))
    canned.x, canned.fun = list(canned.x_iters[j]), float(F[j])
    canned.models, canned.space = ["model"], "space"
    seen = {}

    def fake(func, dimensions, **kw):
        seen.update(func=func, dimensions=dimensions, kw=kw)
        return canned
    wrapped = LR.wrap_minimizer(fake)
    res = wrapped(f, dimensions=dims, n_calls=30, random_state=0)
    bad = []
    if res is not canned:
        bad.append("not the same object")
    if seen.get("func") is not f or seen.get("dimensions") is not dims or seen.get("kw") != dict(n_calls=30, random_state=0):
        bad.append("the minimizer did not receive the call as made: %s" % seen.get("kw"))
    if not hasattr(res, "refine") or not isinstance(res.refine, LR.RefineResult):
        bad.append("refine not attached")
    else:
        if res.x != res.refine.x or res.fun != res.refine.fun or not res.refine.improved:
            bad.append("x / fun not replaced by the refinement's")
        if res.fun >= float(F[j]):
            bad.append("fun not below the canned one")
    if res.x_iters != [list(map(float, x)) for x in X] or not np.array_equal(res.func_vals, np.asarray(F)):
        bad.append("x_iters / func_vals changed")
    if res.models != ["model"] or res.space != "space":
        bad.append("models / space changed")
    # positional dimensions too (skopt's own form)
    res_b = LR.wrap_minimizer(fake)(f, dims)
    if res_b.refine.x_evals.shape[0] == 0:
        bad.append("positional call made no refinement")
    # a non-Real dimension is refused
    try:
        LR.wrap_minimizer(fake)(f, [Integer(0, 3)] + dims[1:])
        bad.append("Integer dimension accepted")
    except ValueError:
        pass
    # the real gp_minimize, 20 calls
    g = LR.wrap_minimizer(gp_minimize)(f, dims, n_calls=20, n_initial_points=10, random_state=0)
    search_best = float(np.min(g.func_vals))
    if not g.fun <= search_best or len(g.x_iters) != 20:
        bad.append("gp_minimize: fun %r above the search's %r" % (g.fun, search_best))
    print("  [R8] fake: same object, refined %.4g < canned %.4g; gp_minimize(20): search %.4g -> %.4g (%d local)"
          % (res.fun, F[j], search_best, g.fun, g.refine.n_evals))
    assert not bad, "; ".join(bad)


def test_R9_determinism():
    c, X, F, f, B = _r1_problem()
    a = LR.refine(f, X, F, B)
    b = LR.refine(f, X, F, B)
    assert np.array_equal(a.x_evals, b.x_evals) and np.array_equal(a.f_evals, b.f_evals), "two calls differ"
    assert a.x == b.x and a.best_start == b.best_start


def test_R10_refusals():
    bad = []
    for kw in (dict(n_starts=0), dict(n_starts=1.5), dict(n_starts=True), dict(maxfev_per_start=0),
               dict(maxfev_per_start=2.0), dict(min_start_dist_unit=0.0), dict(min_start_dist_unit=1.5),
               dict(simplex_step_unit=0.0), dict(simplex_step_unit=1.0), dict(xatol_unit=0.0),
               dict(fatol_rel=-1e-4), dict(penalty=0.0), dict(penalty=float("nan")),
               dict(xatol_unit=float("inf")), dict(fatol_rel="1e-4")):
        try:
            LR.RefineConfig(**kw)
            bad.append("RefineConfig accepted %s" % kw)
        except ValueError:
            pass
    LR.RefineConfig(min_start_dist_unit=1.0)                  # the closed end is allowed
    f = _quad([0.5, 0.5])
    B = [(0.0, 1.0), (0.0, 1.0)]
    X = [[0.1, 0.2], [0.3, 0.4]]
    for label, args in (("empty x_iters", (np.empty((0, 2)), [], B)),
                        ("func_vals length", (X, [1.0], B)),
                        ("x_iters width", ([[0.1], [0.2]], [1.0, 2.0], B)),
                        ("hi <= lo", (X, [1.0, 2.0], [(0.0, 1.0), (1.0, 1.0)])),
                        ("hi < lo", (X, [1.0, 2.0], [(0.0, 1.0), (2.0, 1.0)])),
                        ("bounds not pairs", (X, [1.0, 2.0], [0.0, 1.0])),
                        ("non-finite bounds", (X, [1.0, 2.0], [(0.0, np.inf), (0.0, 1.0)]))):
        for fn in ("refine", "select_starts"):
            try:
                if fn == "refine":
                    LR.refine(f, *args)
                else:
                    LR.select_starts(*args)
                bad.append("%s accepted %s" % (fn, label))
            except ValueError:
                pass
    assert not bad, "; ".join(bad)


# ---------------------------------------------------------------- runner ---
def _environment():
    parts = ["python %s" % platform.python_version()]
    for package in REPORT_PACKAGES:
        try:
            parts.append("%s %s" % (package, importlib.metadata.version(package)))
        except importlib.metadata.PackageNotFoundError:
            parts.append("%s (not installed)" % package)
    parts.append(platform.platform())
    parts.append("seed %d" % SEED)
    return " | ".join(parts)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="", help="comma-separated check ids to run, e.g. R1,R3")
    args = ap.parse_args(argv)
    only = [t.strip() for t in args.only.split(",") if t.strip()]
    checks = [(name, obj) for name, obj in globals().items()
              if name.startswith("test_") and callable(obj)
              and (not only or any(name.startswith("test_%s_" % o) for o in only))]
    print("== %s" % Path(__file__).name)
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
    width = max([len(n) for n, _, _, _ in results] + [4])
    for name, status, seconds, detail in results:
        lines = detail.strip().splitlines()
        headline = lines[0] if lines else ""
        print(("%-5s  %-*s  %8.3fs  %s" % (status, width, name, seconds, headline[:200])).rstrip())
    for name, status, _s, detail in results:
        if status in ("FAIL", "ERROR"):
            print("\n---- %s: %s\n%s" % (status, name, detail.strip()))
    counts = {s: sum(1 for r in results if r[1] == s) for s in ("PASS", "FAIL", "ERROR", "TODO", "SKIP")}
    print("\n-- " + ", ".join("%d %s" % (n, s.lower()) for s, n in counts.items()))
    for path in _SAVED_ARTIFACTS:
        print("-- arrays from a failed comparison: %s" % path)
    return 1 if counts["FAIL"] + counts["ERROR"] + counts["TODO"] else 0


if __name__ == "__main__":
    sys.exit(main())
