"""The estimator of step 2: weighted least squares with shared and per-group unknowns, and the covariance of
the shared ones clustered by group -- Block 56 in specs/SPEC.md (section 5, "Kernel calibration, step 2";
D-273.3). Independent of rendering: the forward model arrives as a callable. No file access.

Implemented 2026-11-27 against contract amendment 1 (5962ca3). Choices SPEC.md leaves open are marked
[implementer] in the comments.
Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class FitResult:
    """shared (S,); local (N, L); residual (n,) = sqrt(w) (y - model), NaN where dropped; kept (n,) bool; cost
    = 0.5 sum residual^2; jac (n_kept, S + N L) at the solution (columns: shared, then local row by row); free
    (S + N L,) bool, False at an active bound; cov_cluster, cov_classical (S, S), NaN rows and columns for a shared
    unknown at a bound; se_cluster, se_classical (S,); n_model_calls, nfev, n_nan_filled int; status int, message
    str, success bool (scipy's)."""

    shared: np.ndarray
    local: np.ndarray
    residual: np.ndarray
    kept: np.ndarray
    cost: float
    jac: np.ndarray
    free: np.ndarray
    cov_cluster: np.ndarray
    cov_classical: np.ndarray
    se_cluster: np.ndarray
    se_classical: np.ndarray
    n_model_calls: int
    nfev: int
    n_nan_filled: int
    status: int
    message: str
    success: bool


def fit_shared_local(model, y, w, group, shared0, shared_bounds, local0=None, local_bounds=None,
                     diff_step_shared=None, diff_step_local=None, max_nfev=None) -> FitResult:
    """Minimise sum_j w_j (y_j - model(shared, local)_j)^2 within box bounds (SPEC Block 56). model(shared (S,),
    local (N, L)) -> (n,), observation j depending on local[group[j]] only; y, w (n,); group (n,) int in 0..N-1;
    shared_bounds (lo (S,), hi (S,)); local0 (N, L) or None (L = 0); local_bounds (lo (N, L), hi (N, L));
    diff_step_shared (S,), diff_step_local (L,): absolute finite-difference steps. scipy.optimize.least_squares
    (trf, x_scale "jac", tr_solver "exact") with the Jacobian as a callable: a structured 2-point forward
    difference with these absolute steps, S + L model calls, dense (amendment 1). ValueError: shapes that do not agree, a negative or non-finite
    weight, a start outside its bounds, a group index outside 0..N-1, no kept observation."""
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 1:
        raise ValueError("y must be 1-D, got shape %r" % (y.shape,))
    n = y.size
    w = np.asarray(w, dtype=np.float64)
    if w.shape != (n,):
        raise ValueError("w must have shape (%d,), got %r" % (n, w.shape))
    if not np.all(np.isfinite(w)) or np.any(w < 0.0):
        raise ValueError("weights must be finite and >= 0")
    group = np.asarray(group)
    if group.shape != (n,):
        raise ValueError("group must have shape (%d,), got %r" % (n, group.shape))
    if group.size and not np.issubdtype(group.dtype, np.integer):
        raise ValueError("group must be an integer array, got dtype %s" % group.dtype)
    group = group.astype(np.int64)

    shared0 = np.atleast_1d(np.asarray(shared0, dtype=np.float64))
    if shared0.ndim != 1:
        raise ValueError("shared0 must be 1-D")
    S = shared0.size
    s_lo, s_hi = _pair(shared_bounds, (S,), "shared_bounds")

    if local0 is None:
        # [implementer] without local unknowns N is read from the group indices (L = 0)
        N = int(group.max()) + 1 if n else 0
        local0 = np.zeros((N, 0))
    local0 = np.asarray(local0, dtype=np.float64)
    if local0.ndim != 2:
        raise ValueError("local0 must be 2-D (N, L), got shape %r" % (local0.shape,))
    N, L = local0.shape
    if local_bounds is None:
        # [implementer] no local bounds given: the local unknowns are unbounded
        l_lo = np.full((N, L), -np.inf)
        l_hi = np.full((N, L), np.inf)
    else:
        l_lo, l_hi = _pair(local_bounds, (N, L), "local_bounds")
    if n and (group.min() < 0 or group.max() > N - 1):
        raise ValueError("a group index lies outside 0..%d" % (N - 1))
    if S + N * L == 0:
        raise ValueError("no unknowns")

    lb = np.concatenate([s_lo, l_lo.ravel()])
    ub = np.concatenate([s_hi, l_hi.ravel()])
    x0 = np.concatenate([shared0, local0.ravel()])
    if np.any(np.isnan(lb)) or np.any(np.isnan(ub)) or np.any(lb >= ub):
        # scipy's least_squares needs lo < hi for every unknown
        raise ValueError("every bound pair must satisfy lo < hi")
    if not np.all(np.isfinite(x0)) or np.any(x0 < lb) or np.any(x0 > ub):
        raise ValueError("a start lies outside its bounds (or is not finite)")

    h_s = _steps(diff_step_shared, shared0, (S,), "diff_step_shared")
    h_l = _steps(diff_step_local, local0, (L,), "diff_step_local")

    calls = [0]

    def predict(x):
        calls[0] += 1
        out = np.asarray(model(x[:S].copy(), x[S:].reshape(N, L).copy()), dtype=np.float64)
        if out.shape != (n,):
            raise ValueError("model returned shape %r, expected (%d,)" % (out.shape, n))
        return out

    # the start: an observation whose y or prediction is not finite is dropped for the whole fit
    # [implementer] "NaN" read as "not finite" (an infinite value is dropped or filled the same way)
    p_start = predict(x0)
    kept = np.isfinite(y) & np.isfinite(p_start)
    if not np.any(kept):
        raise ValueError("no kept observation")
    rows = np.flatnonzero(kept)
    yk, swk, gk = y[rows], np.sqrt(w[rows]), group[rows]
    nk = rows.size
    P = S + N * L

    state = {"last": p_start[rows].copy(), "x": x0.copy(), "pred": p_start[rows].copy(), "filled": 0}

    def fun_pred(x):
        # the prediction at the kept observations, NaNs replaced by the last finite prediction of the residual
        # evaluations; a repeat of the last point reuses its prediction (the model is deterministic: common
        # random numbers, D-270.2)
        if np.array_equal(x, state["x"]):
            return state["pred"]
        p = predict(x)[rows]
        bad = ~np.isfinite(p)
        if np.any(bad):
            state["filled"] += int(bad.sum())
            p = np.where(bad, state["last"], p)
        state["last"] = p.copy()
        state["x"] = x.copy()
        state["pred"] = p
        return p

    def fun(x):
        return swk * (yk - fun_pred(x))

    def jac(x):
        # structured 2-point forward difference with absolute steps (amendment 1): one call per shared unknown,
        # one per local index l with every group's l-th local shifted at once; -h where x + h passes the upper
        # bound; the step actually used is (x + h) - x, exact in floating point
        base = fun_pred(x)
        J = np.zeros((nk, P))

        def probe(xs):
            p = predict(xs)[rows]
            bad = ~np.isfinite(p)
            if np.any(bad):
                # [implementer] a NaN in a probe is replaced by the base point's prediction (its last finite one)
                state["filled"] += int(bad.sum())
                p = np.where(bad, base, p)
            return p

        for s in range(S):
            h = h_s[s] if x[s] + h_s[s] <= ub[s] else -h_s[s]
            xs = x.copy()
            xs[s] = x[s] + h
            dx = xs[s] - x[s]
            J[:, s] = -swk * (probe(xs) - base) / dx
        for l in range(L):
            idx = S + np.arange(N) * L + l
            h = np.where(x[idx] + h_l[l] <= ub[idx], h_l[l], -h_l[l])
            xs = x.copy()
            xs[idx] = x[idx] + h
            dx = xs[idx] - x[idx]
            J[np.arange(nk), idx[gk]] = -swk * (probe(xs) - base) / dx[gk]
        return J

    res = least_squares(fun, x0, jac=jac, bounds=(lb, ub), method="trf", x_scale="jac", tr_solver="exact",
                        max_nfev=max_nfev)

    x = res.x
    r_k = np.asarray(res.fun, dtype=np.float64)
    Jk = np.asarray(res.jac, dtype=np.float64)
    residual = np.full(n, np.nan)
    residual[rows] = r_k
    free = np.asarray(res.active_mask) == 0

    cov_cl, cov_cs = _covariances(Jk, r_k, gk, free, S, N)
    with np.errstate(invalid="ignore"):
        se_cl = np.sqrt(np.diag(cov_cl))
        se_cs = np.sqrt(np.diag(cov_cs))

    ro = lambda a: _readonly(np.array(a, dtype=np.float64))  # noqa: E731
    return FitResult(shared=ro(x[:S]), local=ro(x[S:].reshape(N, L)), residual=ro(residual),
                     kept=_readonly(kept.copy()), cost=float(0.5 * np.sum(r_k ** 2)), jac=ro(Jk),
                     free=_readonly(free.copy()), cov_cluster=ro(cov_cl), cov_classical=ro(cov_cs),
                     se_cluster=ro(se_cl), se_classical=ro(se_cs), n_model_calls=int(calls[0]),
                     nfev=int(res.nfev), n_nan_filled=int(state["filled"]), status=int(res.status),
                     message=str(res.message), success=bool(res.success))


def _readonly(a):
    a.setflags(write=False)
    return a


def _pair(bounds, shape, name):
    try:
        lo, hi = bounds
    except (TypeError, ValueError):
        raise ValueError("%s must be a pair (lo, hi)" % name)
    lo = np.broadcast_to(np.asarray(lo, dtype=np.float64), shape) if np.ndim(lo) == 0 else \
        np.asarray(lo, dtype=np.float64)
    hi = np.broadcast_to(np.asarray(hi, dtype=np.float64), shape) if np.ndim(hi) == 0 else \
        np.asarray(hi, dtype=np.float64)
    if lo.shape != shape or hi.shape != shape:
        raise ValueError("%s must have shapes %r, got %r and %r" % (name, shape, lo.shape, hi.shape))
    return lo.astype(np.float64).copy(), hi.astype(np.float64).copy()


def _steps(h, start, shape, name):
    if h is None:
        # [implementer] no step given: sqrt(eps) * max(1, |start|) (scipy's default size for 2-point), taken once
        # at the start and then held absolute; for a local index the largest |start| over the groups
        scale = np.max(np.abs(start), axis=0) if start.ndim == 2 and start.shape[0] else \
            (np.abs(start) if start.ndim == 1 else np.zeros(shape))
        return np.sqrt(np.finfo(np.float64).eps) * np.maximum(1.0, scale)
    h = np.atleast_1d(np.asarray(h, dtype=np.float64))
    if h.shape != shape:
        raise ValueError("%s must have shape %r, got %r" % (name, shape, h.shape))
    if not np.all(np.isfinite(h)) or np.any(h <= 0.0):
        # [implementer] a step must be finite and > 0
        raise ValueError("%s must be finite and > 0" % name)
    return h.copy()


def _covariances(J, r, g, free, S, N):
    """CR0 sandwich clustered by group and classical s^2 (J'J)^-1, over the free unknowns, shared block (S, S)."""
    cov_cl = np.full((S, S), np.nan)
    cov_cs = np.full((S, S), np.nan)
    # [implementer] a column of J identically zero (a local unknown of a group without kept observations) carries
    # no information and is left out, which does not change the shared block of the inverse
    use = free & np.any(J != 0.0, axis=0)
    cols = np.flatnonzero(use)
    sh = cols[cols < S]
    if sh.size == 0:
        return cov_cl, cov_cs
    Jf = J[:, cols]
    try:
        bread = np.linalg.inv(Jf.T @ Jf)
    except np.linalg.LinAlgError:
        return cov_cl, cov_cs
    meat = np.zeros((cols.size, cols.size))
    for grp in np.unique(g):
        u = Jf[g == grp].T @ r[g == grp]
        meat += np.outer(u, u)
    full_cl = bread @ meat @ bread
    dof = r.size - cols.size
    s2 = float(np.sum(r ** 2) / dof) if dof > 0 else np.nan
    full_cs = s2 * bread
    pos = np.searchsorted(cols, sh)
    cov_cl[np.ix_(sh, sh)] = full_cl[np.ix_(pos, pos)]
    cov_cs[np.ix_(sh, sh)] = full_cs[np.ix_(pos, pos)]
    return cov_cl, cov_cs
