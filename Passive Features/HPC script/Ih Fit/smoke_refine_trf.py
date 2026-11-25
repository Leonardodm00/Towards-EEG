"""Smoke test for the least-squares refinement -- Block 25 in specs/SPEC.md (L1-L11), SPEC 2.10 (c).

Pure: no NEURON, no file I/O by the code under test.

Checks
    test_L1_toy_model         the nonlinear toy model of the review's smoke_sensitivity_reference.py
                              (five parameters, noise-free, reproduced here): 20 uniformly random single
                              starts (default_rng(100 + k)); every estimate within 1e-4 (max-norm, unit box)
                              of the truth, every start converged, n_sims per start <= 180
    test_L2_bowls             the quadratic bowls of the review's check02 (condition numbers 1, 1e2, 1e4;
                              seeds 0..19; start = best of 200 random points): within 1e-4 of the minimum
    test_L3_face              a minimum outside the box along one axis: the estimate on that face (1e-6),
                              active_mask non-zero there, every recorded point inside the box
    test_L4_bookkeeping       n_sims == len(l2_evals) == rows returned by `residuals`; each Jacobian is one
                              call of p rows; the estimate is the lowest recorded L2; l2_evals[k] is the
                              squared norm of the k-th row; per-start fields and shapes
    test_L5_two_basins        search best in the shallow basin, a later start in the deep one: the estimate
                              is in the deep basin and best_start names that start
    test_L6_failed_rows       `residuals` returning the constant failed row off a region: no crash, finite
                              result; every global value penalised: no start, the search's point
    test_L7_wrap_minimizer    a fake minimizer: the same object back, x / fun replaced, refine attached,
                              get_residuals called once and after the minimizer; fun = func(x) without
                              l_and_l2; a non-Real dimension refused; skopt.gp_minimize runs wrapped
    test_L8_determinism       two calls give identical x_evals and l2_evals
    test_L9_refusals          TrfConfig refusals; refine_trf's ValueErrors
    test_L10_local_refine     smoke_local_refine.py still passes 10/10 (run as a subprocess)
    test_L11_ascii_lf         local_refine.py and this file are ASCII with LF line endings

Block 24 (fd_jacobian.py) is implemented in parallel. While its functions raise
NotImplementedError, a private reference implementation with EXACTLY the contract's
signatures (SPEC Block 24) is injected into the fd_jacobian module before the checks;
local_refine.py itself always calls the real module. The header line says which one ran.

Run (from this folder)
    python smoke_refine_trf.py [--only L1,L4]
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
import subprocess
import sys
import time
import traceback
import unittest
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np                         # noqa: E402
from numpy.testing import assert_allclose  # noqa: E402

import fd_jacobian as FD                   # noqa: E402
import local_refine as LR                  # noqa: E402

SEED = 12345
REPORT_PACKAGES = ("numpy", "scipy", "scikit-optimize")
ARTIFACT_DIR = HERE / "smoke_artifacts"
_SAVED_ARTIFACTS = []
FD_SOURCE = "fd_jacobian.py (Block 24)"


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


# ------------------------------------------- reference fd_jacobian (Block 24 contract) ---
def _ref_bounds(bounds):
    B = np.asarray(bounds, dtype=np.float64)
    if B.ndim != 2 or B.shape[1] != 2 or not np.all(B[:, 1] > B[:, 0]):
        raise ValueError("bounds")
    return B


def _ref_to_unit(q, bounds):
    B = _ref_bounds(bounds)
    return (np.asarray(q, dtype=np.float64) - B[:, 0]) / (B[:, 1] - B[:, 0])


def _ref_to_q(u, bounds):
    B = _ref_bounds(bounds)
    q = B[:, 0] + np.asarray(u, dtype=np.float64) * (B[:, 1] - B[:, 0])
    return np.minimum(np.maximum(q, B[:, 0]), B[:, 1])


def _ref_fd_points(u, *, h=1e-3, scheme="forward"):
    u = np.asarray(u, dtype=np.float64)
    if u.ndim != 1 or not np.all(np.isfinite(u)) or np.any(u < 0) or np.any(u > 1):
        raise ValueError("u")
    if not (0.0 < h <= 0.5):
        raise ValueError("h")
    if scheme not in ("forward", "central"):
        raise ValueError("scheme")
    rows, plan = [], []
    for i in range(u.size):
        if scheme == "central" and u[i] + h <= 1.0 and u[i] - h >= 0.0:
            up, dn = u.copy(), u.copy()
            up[i] += h
            dn[i] -= h
            plan.append(("central", len(rows), len(rows) + 1, h))
            rows += [up, dn]
        else:
            d = h if u[i] + h <= 1.0 else -h
            v = u.copy()
            v[i] += d
            plan.append(("forward", len(rows), d))
            rows.append(v)
    return np.asarray(rows, dtype=np.float64).reshape(len(rows), u.size), plan


def _ref_fd_jacobian(batch_fun, u, *, h=1e-3, scheme="forward", f0=None):
    u = np.asarray(u, dtype=np.float64)
    U, plan = _ref_fd_points(u, h=h, scheme=scheme)
    need_f0 = any(e[0] == "forward" for e in plan)
    if f0 is not None or not need_f0:
        R = np.asarray(batch_fun(U), dtype=np.float64)
        off = 0
    else:
        R = np.asarray(batch_fun(np.vstack([u[None, :], U])), dtype=np.float64)
        f0 = R[0]
        off = 1
    if R.ndim != 2 or R.shape[0] != U.shape[0] + off or not np.all(np.isfinite(R)):
        raise ValueError("batch_fun")
    f0 = None if f0 is None else np.asarray(f0, dtype=np.float64)
    J = np.empty((R.shape[1], u.size), dtype=np.float64)
    for i, e in enumerate(plan):
        if e[0] == "forward":
            J[:, i] = (R[off + e[1]] - f0) / e[2]
        else:
            J[:, i] = (R[off + e[1]] - R[off + e[2]]) / (2.0 * e[3])
    return FD.FdResult(J=J, f0=f0, n_points=int(R.shape[0]), plan=plan)


def _ref_jac_unit_to_q(J_u, bounds):
    B = _ref_bounds(bounds)
    return np.asarray(J_u, dtype=np.float64) / (B[:, 1] - B[:, 0])[None, :]


def _install_fd_reference():
    """Inject the reference while Block 24's module is still a stub."""
    global FD_SOURCE
    try:
        FD.to_unit(np.zeros(1), [(0.0, 1.0)])
    except NotImplementedError:
        FD.to_unit, FD.to_q, FD.fd_points = _ref_to_unit, _ref_to_q, _ref_fd_points
        FD.fd_jacobian, FD.jac_unit_to_q = _ref_fd_jacobian, _ref_jac_unit_to_q
        FD_SOURCE = "smoke's private reference (fd_jacobian.py is still a stub)"


# ------------------------------------------------------------- helpers ---
class CountingResiduals:
    """Wraps a batch residual function; counts calls, rows; keeps every row."""

    def __init__(self, fun):
        self.fun = fun
        self.calls = []        # m of each call
        self.rows = []

    def __call__(self, Q):
        Q = np.asarray(Q, dtype=np.float64)
        R = np.asarray(self.fun(Q), dtype=np.float64)
        self.calls.append(Q.shape[0])
        self.rows.extend([r.copy() for r in R])
        return R


def _batch(single):
    """Batch residual function from a single-point one."""
    return lambda Q: np.vstack([single(q) for q in np.atleast_2d(Q)])


def _unit(q, bounds):
    b = np.asarray(bounds, dtype=float)
    return (np.asarray(q, dtype=float) - b[:, 0]) / (b[:, 1] - b[:, 0])


# ---- the toy model of the review's smoke_sensitivity_reference.py (five parameters) ----
TOY_K = 8.05
TOY_T = np.linspace(0.0, 1.0, 400)
TOY_AMPS = np.array([-50.0, -100.0, -150.0, -200.0])
TOY_IS_LS = np.array([False, True, True, True])
TOY_Q_LO = np.array([np.log(0.02), np.log(0.005), np.log(0.05), np.log(0.1), np.log(0.5)])
TOY_Q_HI = np.array([np.log(0.20), np.log(0.050), np.log(1.00), np.log(10.), np.log(2.0)])
TOY_U_TRUE = np.array([0.55, 0.40, 0.60, 0.45, 0.50])


def _toy_simulate(q):
    R, tau, tau_h, g, a = (np.exp(q[0]), np.exp(q[1]), np.exp(q[2]), np.exp(q[3]), np.exp(q[4]))
    out = []
    for amp in TOY_AMPS:
        G = g * (abs(amp) / 100.0) ** a
        sag = G / (1.0 + G) * (1.0 - np.exp(-TOY_T / tau_h))
        out.append(amp * R * (1.0 - np.exp(-TOY_T / tau)) * (1.0 - sag))
    return np.array(out)


def _toy_weights(V):
    D = np.max(np.abs(V), axis=1)
    w = np.empty_like(V)
    for b in range(V.shape[0]):
        if TOY_IS_LS[b]:
            k = max(int(np.argmax(np.abs(V[b]))), 1)
            w[b, :k + 1] = TOY_T[:k + 1] / TOY_T[k]
            w[b, k + 1:] = np.exp(-(TOY_T[k + 1:] - TOY_T[k]) / 0.5)
        else:
            w[b] = np.exp(-TOY_T / 0.005)
    n_ss, n_ls = int((~TOY_IS_LS).sum()), int(TOY_IS_LS.sum())
    omega = np.abs(TOY_AMPS) * TOY_IS_LS / np.sum(np.abs(TOY_AMPS[TOY_IS_LS]))
    c = np.where(TOY_IS_LS, n_ls * omega, 1.0) / (n_ss + n_ls)
    a = c[:, None] * w / w.sum(axis=1, keepdims=True)
    return a, D, c, w


def _toy_problem():
    bounds = [(float(lo), float(hi)) for lo, hi in zip(TOY_Q_LO, TOY_Q_HI)]
    V = _toy_simulate(TOY_Q_LO + TOY_U_TRUE * (TOY_Q_HI - TOY_Q_LO))
    a, D, c, _w = _toy_weights(V)
    sa = np.sqrt(a)

    def single(q):
        return (sa * (_toy_simulate(q) - V) / D[:, None]).ravel()

    def loss_of_row(row):           # L of 2.10 (a) from the same vector
        r2 = (row.reshape(V.shape) ** 2).sum(axis=1)
        return float(np.sum(c * np.sqrt(r2 / c)))

    return bounds, _batch(single), loss_of_row


# ---- the bowls of the review's check02 ----
def _bowl(seed, cond, p=5, n_search=200):
    from scipy.stats import ortho_group
    rng = np.random.default_rng(seed)
    c = 0.15 + 0.7 * rng.random(p)
    Q = ortho_group.rvs(p, random_state=seed)
    lam = np.logspace(0, -np.log10(cond), p)
    M = np.sqrt(lam)[:, None] * Q
    X = rng.random((n_search, p))
    return c, M, X


# ---------------------------------------------------------------- checks ---
def test_L1_toy_model():
    bounds, res, lor = _toy_problem()
    cfg = LR.TrfConfig(n_starts=1)
    worst, sims, bad = 0.0, [], []
    for k in range(20):
        u0 = np.random.default_rng(100 + k).random(5)
        q0 = TOY_Q_LO + u0 * (TOY_Q_HI - TOY_Q_LO)
        f0 = lor(res(q0[None, :])[0])
        r = LR.refine_trf(res, [q0], [f0], bounds, cfg, loss_of_row=lor)
        d = float(np.max(np.abs(_unit(r.x, bounds) - TOY_U_TRUE)))
        worst = max(worst, d)
        s = r.starts[0]
        sims.append(s["n_sims"])
        if not s["converged"]:
            bad.append("start %d not converged (status %d)" % (k, s["status"]))
        if not s["n_sims"] <= 180:
            bad.append("start %d n_sims %d > 180" % (k, s["n_sims"]))
        if d >= 1e-4:
            bad.append("start %d distance %.2e" % (k, d))
    print("  [L1] fd: %s" % FD_SOURCE)
    print("  [L1] 20 starts: worst max-norm unit distance %.2e (limit 1e-4); n_sims median %d, max %d (limit 180)"
          % (worst, int(np.median(sims)), max(sims)))
    assert not bad, "; ".join(bad)


def test_L2_bowls():
    cfg = LR.TrfConfig(n_starts=1)       # the oracle's start: the best of the 200 points
    bad, worst = [], {}
    for cond in (1e0, 1e2, 1e4):
        w, ns = 0.0, []
        for seed in range(20):
            c, M, X = _bowl(seed, cond)
            res = lambda Q, M=M, c=c: (np.atleast_2d(Q) - c) @ M.T     # noqa: E731
            F = np.sum(res(X) ** 2, axis=1)
            r = LR.refine_trf(res, X, F, [(0.0, 1.0)] * 5, cfg)
            if r.starts[0]["index"] != int(np.argmin(F)):
                bad.append("cond %g seed %d: start not the best point" % (cond, seed))
            d = float(np.max(np.abs(np.asarray(r.x) - c)))
            w = max(w, d)
            ns.append(r.n_sims)
            if d >= 1e-4:
                bad.append("cond %g seed %d: distance %.2e" % (cond, seed, d))
        worst[cond] = w
        print("  [L2] cond %.0e: worst distance %.2e (limit 1e-4); n_sims median %d max %d"
              % (cond, w, int(np.median(ns)), max(ns)))
    assert not bad, "; ".join(bad)


def test_L3_face():
    bounds = [(-2.0, 3.0), (0.0, 1.0), (10.0, 12.0)]
    B = np.asarray(bounds)
    c = np.array([0.4, 0.3, 1.3])          # unit coordinates; axis 2 beyond the upper face
    scale = np.array([1.0, 2.0, 0.5])

    def single(q):
        u = (q - B[:, 0]) / (B[:, 1] - B[:, 0])
        return np.concatenate([scale * (u - c), [0.3 * (u[0] - c[0]) * (u[1] - c[1])]])

    res = CountingResiduals(_batch(single))
    rng = np.random.default_rng(SEED)
    X = B[:, 0] + rng.random((30, 3)) * (B[:, 1] - B[:, 0])
    F = [float(np.sum(single(x) ** 2)) for x in X]
    r = LR.refine_trf(res, X, F, bounds)
    u_hat = _unit(r.x, bounds)
    bad = []
    if abs(u_hat[2] - 1.0) > 1e-6:
        bad.append("u_2 %.9f not on the upper face (1e-6)" % u_hat[2])
    if np.max(np.abs(u_hat[:2] - c[:2])) > 1e-4:
        bad.append("free axes off: %s" % u_hat[:2])
    s_best = r.starts[r.best_start if r.best_start >= 0 else 0]
    if s_best["active_mask"][2] == 0:
        bad.append("active_mask %s zero on axis 2" % s_best["active_mask"])
    inside = np.all((r.x_evals >= B[:, 0]) & (r.x_evals <= B[:, 1]))
    if not inside:
        bad.append("a recorded point left the box")
    print("  [L3] u_hat %s; active_mask %s; %d recorded points, all inside: %s"
          % (np.array2string(u_hat, precision=7), s_best["active_mask"], r.n_sims, inside))
    assert not bad, "; ".join(bad)


def test_L4_bookkeeping():
    bounds, base, lor = _toy_problem()
    p = 5
    res = CountingResiduals(base)
    rng = np.random.default_rng(SEED)
    U = rng.random((25, p))
    X = TOY_Q_LO + U * (TOY_Q_HI - TOY_Q_LO)
    F = [lor(row) for row in base(X)]
    r = LR.refine_trf(res, X, F, bounds, loss_of_row=lor)
    bad = []
    n_rows = sum(res.calls)
    if not (r.n_sims == len(r.l2_evals) == n_rows == len(res.rows) == r.n_evals):
        bad.append("n_sims %d, len(l2_evals) %d, rows %d" % (r.n_sims, len(r.l2_evals), n_rows))
    njev = sum(s["njev"] for s in r.starts)
    n_p = sum(1 for m in res.calls if m == p)
    if any(m not in (1, p) for m in res.calls):
        bad.append("a call with %s rows" % sorted(set(res.calls)))
    if n_p != njev:
        bad.append("%d calls of p rows, %d Jacobians" % (n_p, njev))
    if sum(s["n_sims"] for s in r.starts) != r.n_sims:
        bad.append("per-start n_sims do not add up")
    rows = np.asarray(res.rows)
    check_close("L4 l2_evals", r.l2_evals, np.sum(rows ** 2, axis=1), rtol=1e-14)
    i = int(np.argmin(r.l2_evals))
    if r.l2 != float(r.l2_evals[i]) or r.x != [float(v) for v in r.x_evals[i]]:
        bad.append("estimate not the lowest recorded L2")
    if r.fun != r.l or not math.isfinite(r.l) or abs(r.l - lor(rows[i])) > 0.0:
        bad.append("l / fun not L of the estimate's row")
    if r.x_search != [float(v) for v in r.x_evals[0]] or r.l2_search != float(r.l2_evals[0]):
        bad.append("x_search / l2_search not the first start")
    if r.improved != (r.l2 < r.l2_search):
        bad.append("improved")
    if r.improved and r.best_start != int(r.start_of_eval[i]):
        bad.append("best_start")
    # shapes and dtypes
    if (r.x_evals.dtype != np.float64 or r.x_evals.shape != (r.n_sims, p) or r.l2_evals.dtype != np.float64
            or r.start_of_eval.dtype != np.int64 or r.start_of_eval.shape != (r.n_sims,)):
        bad.append("array shapes / dtypes")
    keys = {"index", "x0", "l2_0", "l_0", "x", "l2", "l", "x_solver", "status", "converged", "optimality",
            "active_mask", "nfev", "njev", "n_sims", "message", "jtj", "sv"}
    for k, s in enumerate(r.starts):
        if set(s) != keys:
            bad.append("start %d keys %s" % (k, sorted(set(s) ^ keys)))
            continue
        if s["jtj"].shape != (p, p) or s["sv"].shape != (p,) or np.any(np.diff(s["sv"]) > 0):
            bad.append("start %d jtj / sv shape or order" % k)
        # J^T J in q at x_solver: against a fresh forward Jacobian at x_solver (the same rule)
        u_s = _unit(s["x_solver"], bounds)
        fr = FD.fd_jacobian(lambda Uu: base(FD.to_q(Uu, bounds)), u_s, h=1e-3)
        Jq = FD.jac_unit_to_q(fr.J, bounds)
        check_close("L4 jtj start %d" % k, s["jtj"], Jq.T @ Jq, rtol=1e-8, atol=1e-12 * np.max(np.abs(Jq.T @ Jq)))
        check_close("L4 sv start %d" % k, s["sv"] ** 2, np.sort(np.linalg.eigvalsh(s["jtj"]))[::-1],
                    rtol=1e-8, atol=1e-12 * s["sv"][0] ** 2)
        own = r.l2_evals[r.start_of_eval == k]
        if s["l2"] != float(np.min(own)) or s["l2_0"] != float(own[0]):
            bad.append("start %d l2 / l2_0" % k)
        if s["converged"] != (s["status"] in (1, 2, 3, 4)):
            bad.append("start %d converged" % k)
    print("  [L4] %d starts; %d rows in %d calls (%d Jacobians of p rows); L2 %.3e -> %.3e; best_start %d"
          % (len(r.starts), n_rows, len(res.calls), njev, r.l2_search, r.l2, r.best_start))
    assert not bad, "; ".join(bad)


def test_L5_two_basins():
    a = np.array([0.25, 0.25])
    b = np.array([0.75, 0.75])

    def single(u):
        da, db = np.sum((u - a) ** 2), np.sum((u - b) ** 2)
        return np.array([40.0 * da * db, 0.3 * (1.0 - np.exp(-da / 0.02))])

    res = _batch(single)
    X = np.array([b, b + [0.02, 0.0], a + [0.15, 0.0], [0.5, 0.05], [0.05, 0.9]])
    F = np.sum(res(X) ** 2, axis=1)
    order = list(np.argsort(F, kind="stable"))
    r = LR.refine_trf(res, X, F, [(0.0, 1.0)] * 2)
    bad = []
    if order[:3] != [0, 1, 2]:
        bad.append("fixture: order %s" % order)
    if [s["index"] for s in r.starts][:2] != [0, 2]:
        bad.append("starts %s" % [s["index"] for s in r.starts])
    # "in the deep basin": near a and below the shallow basin's floor (L2 = 0.09 at b). Both residuals
    # vanish quadratically at a (zero Jacobian there), so the solver approaches a sublinearly: the
    # distance to a is not a 1e-4 quantity here, the basin is what this check asks.
    if np.max(np.abs(np.asarray(r.x) - a)) > 0.05 or not r.l2 < 1e-6:
        bad.append("estimate %s (L2 %.3g) not in the deep basin" % (r.x, r.l2))
    if r.best_start != 1 or not r.improved:
        bad.append("best_start %d" % r.best_start)
    print("  [L5] search values %s; starts %s; estimate %s, L2 %.2e; best_start %d"
          % (np.array2string(F, precision=3), [s["index"] for s in r.starts],
             np.array2string(np.asarray(r.x), precision=6), r.l2, r.best_start))
    assert not bad, "; ".join(bad)


def test_L6_failed_rows():
    c = np.array([0.5, 0.6, 0.4])
    N = 4
    fail = math.sqrt(100.0 / N)

    def single(u):
        if np.max(np.abs(u - c)) > 0.2:
            return np.full(N, fail)
        return np.array([u[0] - c[0], 2.0 * (u[1] - c[1]), 0.5 * (u[2] - c[2]), (u[0] - c[0]) * (u[2] - c[2])])

    res = _batch(single)
    rng = np.random.default_rng(SEED)
    X = rng.random((40, 3))
    X[0] = c + [0.1995, -0.15, 0.19]        # at the region's edge: its first Jacobian steps into the failed rows
    X[1] = c + [0.05, 0.05, -0.05]
    F = np.sum(res(X) ** 2, axis=1)
    F = np.where(F >= 100.0 - 1e-9, 1e6, F)  # a failed point gets the loss's penalty in the search
    r = LR.refine_trf(res, X, F, [(0.0, 1.0)] * 3)
    bad = []
    if not (np.all(np.isfinite(r.x)) and math.isfinite(r.l2)):
        bad.append("non-finite result")
    if not r.l2 < 1e-6:
        bad.append("L2 %.3g at the estimate" % r.l2)
    n_failed = int(np.sum(np.abs(r.l2_evals - 100.0) < 1e-9))
    if n_failed == 0:
        bad.append("fixture: no failed row was met")
    # every global value penalised: no start
    r0 = LR.refine_trf(res, X, np.full(len(X), 1e6), [(0.0, 1.0)] * 3)
    if r0.n_sims != 0 or r0.improved or r0.starts or r0.x != [float(v) for v in X[0]] or r0.best_start != -1:
        bad.append("no-start case: n_sims %d, x %s" % (r0.n_sims, r0.x))
    if not (math.isnan(r0.l2) and math.isnan(r0.fun)) or r0.x_evals.shape != (0, 3):
        bad.append("no-start case: l2 / fun / x_evals")
    print("  [L6] %d starts; failed rows recorded %d of %d; estimate L2 %.2e; statuses %s; no-start case n_sims %d"
          % (len(r.starts), int(np.sum(np.abs(r.l2_evals - 100.0) < 1e-9)), r.n_sims, r.l2,
             [s["status"] for s in r.starts], r0.n_sims))
    assert not bad, "; ".join(bad)


def test_L7_wrap_minimizer():
    from skopt import gp_minimize
    from skopt.space import Integer, Real
    c = np.array([0.3, 0.7])
    order = []

    class Resid:
        def __call__(self, Q):
            return np.atleast_2d(Q) - c

        def l_and_l2(self, row):
            l2 = float(np.sum(row ** 2))
            return math.sqrt(l2), l2

    def f(x):
        return float(np.sqrt(np.sum((np.asarray(x) - c) ** 2)))

    rng = np.random.default_rng(SEED)
    X = [list(map(float, x)) for x in rng.random((12, 2))]
    F = [f(x) for x in X]
    canned = SimpleNamespace(x=list(X[int(np.argmin(F))]), fun=float(min(F)), x_iters=X, func_vals=np.asarray(F),
                             models=["model"], space="space")

    def fake(func, dimensions, **kw):
        order.append("minimizer")
        return canned

    def get_res():
        order.append("get_residuals")
        return Resid()

    dims = [Real(0.0, 1.0), Real(0.0, 1.0)]
    out = LR.wrap_minimizer_trf(fake, get_res)(f, dims, n_calls=5)
    bad = []
    if out is not canned:
        bad.append("not the same object")
    if order != ["minimizer", "get_residuals"]:
        bad.append("call order %s" % order)
    if not isinstance(out.refine, LR.TrfResult):
        bad.append("refine not attached")
    if out.x != out.refine.x or out.fun != out.refine.l or np.max(np.abs(np.asarray(out.x) - c)) > 1e-6:
        bad.append("x / fun not the estimate: %s %r" % (out.x, out.fun))
    if out.x_iters is not X or out.models != ["model"] or out.space != "space":
        bad.append("x_iters / models / space changed")
    # without l_and_l2: fun = func(x)
    canned2 = SimpleNamespace(x=list(X[0]), fun=F[0], x_iters=X, func_vals=np.asarray(F), models=[], space=None)
    out2 = LR.wrap_minimizer_trf(lambda func, d, **k: canned2,
                                 lambda: (lambda Q: np.atleast_2d(Q) - c))(f, dims)
    if not (math.isnan(out2.refine.l) and out2.fun == f(out2.x)):
        bad.append("fun without l_and_l2: %r" % out2.fun)
    # a non-Real dimension refused before the minimizer runs
    order.clear()
    try:
        LR.wrap_minimizer_trf(fake, get_res)(f, [Integer(0, 3), dims[1]])
        bad.append("Integer dimension accepted")
    except ValueError:
        if order:
            bad.append("minimizer ran before the refusal")
    # the real gp_minimize, 15 calls
    g = LR.wrap_minimizer_trf(gp_minimize, get_res)(f, dims, n_calls=15, n_initial_points=8, random_state=0)
    if len(g.x_iters) != 15 or not g.fun <= float(np.min(g.func_vals)) + 1e-12:
        bad.append("gp_minimize: fun %r, search %r" % (g.fun, float(np.min(g.func_vals))))
    print("  [L7] fake: same object, fun %.2e (canned %.3g); gp_minimize(15): search %.3g -> %.2e (%d sims)"
          % (out.fun, canned.func_vals.min(), float(np.min(g.func_vals)), g.fun, g.refine.n_sims))
    assert not bad, "; ".join(bad)


def test_L8_determinism():
    bounds, res, lor = _toy_problem()
    rng = np.random.default_rng(SEED)
    X = TOY_Q_LO + rng.random((15, 5)) * (TOY_Q_HI - TOY_Q_LO)
    F = [lor(row) for row in res(X)]
    a = LR.refine_trf(res, X, F, bounds, loss_of_row=lor)
    b = LR.refine_trf(res, X, F, bounds, loss_of_row=lor)
    assert np.array_equal(a.x_evals, b.x_evals) and np.array_equal(a.l2_evals, b.l2_evals), "two calls differ"
    assert a.x == b.x and a.best_start == b.best_start and np.array_equal(a.start_of_eval, b.start_of_eval)


def test_L9_refusals():
    bad = []
    for kw in (dict(n_starts=0), dict(n_starts=1.5), dict(n_starts=True), dict(max_nfev=0), dict(max_nfev=3.0),
               dict(min_start_dist_unit=0.0), dict(min_start_dist_unit=1.5), dict(fd_step_unit=0.0),
               dict(fd_step_unit=0.6), dict(xtol=0.0), dict(ftol=-1e-8), dict(gtol=float("nan")),
               dict(penalty=0.0), dict(penalty=float("inf")), dict(xtol="1e-4")):
        try:
            LR.TrfConfig(**kw)
            bad.append("TrfConfig accepted %s" % kw)
        except ValueError:
            pass
    LR.TrfConfig(min_start_dist_unit=1.0, fd_step_unit=0.5)     # the closed ends are allowed
    res = lambda Q: np.atleast_2d(Q) - 0.5                       # noqa: E731
    B = [(0.0, 1.0), (0.0, 1.0)]
    X = [[0.1, 0.2], [0.3, 0.4]]
    for label, args in (("empty x_iters", (np.empty((0, 2)), [], B)),
                        ("func_vals length", (X, [1.0], B)),
                        ("x_iters width", ([[0.1], [0.2]], [1.0, 2.0], B)),
                        ("hi <= lo", (X, [1.0, 2.0], [(0.0, 1.0), (1.0, 1.0)])),
                        ("bounds not pairs", (X, [1.0, 2.0], [0.0, 1.0]))):
        try:
            LR.refine_trf(res, *args)
            bad.append("refine_trf accepted %s" % label)
        except ValueError:
            pass
    try:
        LR.refine_trf(lambda Q: np.zeros(3), X, [1.0, 2.0], B)
        bad.append("refine_trf accepted a 1-D residual output")
    except ValueError:
        pass
    assert not bad, "; ".join(bad)


def test_L10_local_refine():
    out = subprocess.run([sys.executable, str(HERE / "smoke_local_refine.py")], capture_output=True, text=True,
                         cwd=str(HERE), timeout=600)
    summary = [ln for ln in out.stdout.splitlines() if ln.startswith("-- ")]
    print("  [L10] smoke_local_refine.py exit %d: %s" % (out.returncode, summary[0] if summary else "?"))
    assert out.returncode == 0 and summary and summary[0].startswith("-- 10 pass"), out.stdout[-2000:]


def test_L11_ascii_lf():
    bad = []
    for name in ("local_refine.py", "smoke_refine_trf.py"):
        data = (HERE / name).read_bytes()
        if any(b > 127 for b in data):
            bad.append("%s has non-ASCII bytes" % name)
        if b"\r" in data:
            bad.append("%s has CR bytes" % name)
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
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", default="", help="comma-separated check ids to run, e.g. L1,L4")
    args = ap.parse_args(argv)
    only = [t.strip() for t in args.only.split(",") if t.strip()]
    checks = [(name, obj) for name, obj in globals().items()
              if name.startswith("test_") and callable(obj)
              and (not only or any(name.startswith("test_%s_" % o) for o in only))]
    _install_fd_reference()
    print("== %s" % Path(__file__).name)
    print(_environment())
    print("fd_jacobian in use: %s" % FD_SOURCE)
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
