"""local_refine.py -- the local refinement after the global search.

SPEC section 2.9 (b) and Block 20 (decision D-234): from the best evaluated
points of the global search, bounded Nelder-Mead searches in the unit
coordinates of the box; the estimate is the lowest loss over every
evaluation, global and local.

Pure module: no NEURON, no file I/O, no global state.

Notation (SPEC 2.9 (b))
-----------------------
    p        number of fitted parameters
    q        optimiser coordinates, float (p,): log of a positive
             parameter or the parameter itself (param_spec)
    bounds   list of p pairs (lo, hi) in q, hi > lo
    u        unit coordinates, u_i = (q_i - lo_i) / (hi_i - lo_i) in [0, 1]
    L        the loss, dimensionless; objective(x) with x a list of p floats
             (q) returns L(q) as a float

The procedure
-------------
Starts. The global evaluations (q^(j), L_j), j = 0..N-1 (0-based here), are
ordered by increasing L_j, ties by increasing j; an evaluation with
L_j >= penalty or a non-finite L_j is left out. The first start is the first
of the order; each next start is the next of the order whose max-norm
distance in u from every start already chosen is >= min_start_dist_unit; at
most n_starts.

One start. scipy.optimize.minimize(f, u(s), method="Nelder-Mead",
bounds=[(0, 1)] * p) on f(u) = L(q(u)), initial simplex u(s) and
u(s) + delta_i e_i (delta_i = +step, or -step when u_i(s) + step > 1),
maxfev, xatol (unit coordinates), fatol = fatol_rel * |L(s)|,
adaptive=False. Every evaluation scipy makes is recorded with its point.

Estimate. The lowest loss among the global evaluations and every recorded
local one; ties: the global search's point, then the earlier start, then
the earlier evaluation. `improved` when a local evaluation is strictly
below min_j L_j.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

__all__ = ["RefineConfig", "RefineResult", "select_starts", "refine",
           "wrap_minimizer"]


def _is_int(v: Any) -> bool:
    return isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))


def _is_real(v: Any) -> bool:
    return (isinstance(v, (int, float, np.integer, np.floating))
            and not isinstance(v, (bool, np.bool_)) and math.isfinite(float(v)))


@dataclass(frozen=True)
class RefineConfig:
    """Settings of SPEC 2.9 (b). Validated in __post_init__ (ValueError):
    n_starts >= 1 and maxfev_per_start >= 1 integers; 0 < min_start_dist_unit
    <= 1; 0 < simplex_step_unit < 1; xatol_unit > 0; fatol_rel > 0;
    penalty > 0."""
    n_starts: int = 3                  # starts, at most
    maxfev_per_start: int = 100        # scipy's maxfev of one start
    min_start_dist_unit: float = 0.1   # d_min, max-norm in unit coordinates
    simplex_step_unit: float = 0.05    # |delta_i| of the initial simplex
    xatol_unit: float = 1e-3           # scipy's xatol, unit coordinates
    fatol_rel: float = 1e-4            # scipy's fatol = fatol_rel * L(start)
    penalty: float = 1e6               # evaluations with L >= penalty are no start

    def __post_init__(self) -> None:
        for name in ("n_starts", "maxfev_per_start"):
            v = getattr(self, name)
            if not _is_int(v) or int(v) < 1:
                raise ValueError("RefineConfig.%s must be an integer >= 1, got %r"
                                 % (name, v))
        for name in ("min_start_dist_unit", "simplex_step_unit", "xatol_unit",
                     "fatol_rel", "penalty"):
            v = getattr(self, name)
            if not _is_real(v):
                raise ValueError("RefineConfig.%s must be a finite real number, got %r"
                                 % (name, v))
        if not (0.0 < float(self.min_start_dist_unit) <= 1.0):
            raise ValueError("RefineConfig.min_start_dist_unit must lie in (0, 1], got %r"
                             % (self.min_start_dist_unit,))
        if not (0.0 < float(self.simplex_step_unit) < 1.0):
            raise ValueError("RefineConfig.simplex_step_unit must lie in (0, 1), got %r"
                             % (self.simplex_step_unit,))
        for name in ("xatol_unit", "fatol_rel", "penalty"):
            if not float(getattr(self, name)) > 0.0:
                raise ValueError("RefineConfig.%s must be > 0, got %r"
                                 % (name, getattr(self, name)))


@dataclass
class RefineResult:
    """What refine returns (SPEC Block 20)."""
    x: List[float]                 # q_hat, p floats
    fun: float                     # L(q_hat)
    improved: bool                 # a local evaluation strictly below fun_search
    x_search: List[float]          # the global search's best point (q)
    fun_search: float              # min_j L_j
    best_start: int                # 0-based position among the starts; -1 if not improved
    n_evals: int                   # local evaluations made, all starts
    starts: List[Dict[str, Any]]   # per start: index, x0, f0, x, fun, nfev, nit, status, message
    x_evals: np.ndarray            # float64 (n_evals, p), q of every local evaluation, in order
    f_evals: np.ndarray            # float64 (n_evals,)
    start_of_eval: np.ndarray      # int64 (n_evals,), the start's position


# ---------------------------------------------------------------------------
#  Input checks and coordinates
# ---------------------------------------------------------------------------
def _as_inputs(x_iters, func_vals, bounds) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(X (N, p), F (N,), B (p, 2)) as float64 arrays; ValueError on shapes
    that do not agree, an empty x_iters, or bounds with hi <= lo."""
    try:
        X = np.asarray(x_iters, dtype=np.float64)
        F = np.asarray(func_vals, dtype=np.float64)
        B = np.asarray(bounds, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("x_iters, func_vals and bounds must be numeric: %s" % exc)
    if B.ndim != 2 or B.shape[1] != 2 or B.shape[0] < 1:
        raise ValueError("bounds must be p >= 1 pairs (lo, hi), got shape %s" % (B.shape,))
    if not np.all(np.isfinite(B)):
        raise ValueError("bounds must be finite, got %s" % (B.tolist(),))
    if not np.all(B[:, 1] > B[:, 0]):
        raise ValueError("bounds need hi > lo on every axis, got %s" % (B.tolist(),))
    p = B.shape[0]
    if X.ndim != 2 or X.shape[0] < 1:
        raise ValueError("x_iters must be a non-empty (N, p) array, got shape %s" % (X.shape,))
    if X.shape[1] != p:
        raise ValueError("x_iters has %d coordinates per point, bounds %d" % (X.shape[1], p))
    if F.ndim != 1 or F.shape[0] != X.shape[0]:
        raise ValueError("func_vals must have shape (%d,), got %s" % (X.shape[0], F.shape))
    return X, F, B


def _to_unit(q: np.ndarray, B: np.ndarray) -> np.ndarray:
    """u = (q - lo) / (hi - lo), row-wise."""
    return (np.asarray(q, dtype=np.float64) - B[:, 0]) / (B[:, 1] - B[:, 0])


def _to_q(u: np.ndarray, B: np.ndarray) -> np.ndarray:
    """q = lo + u (hi - lo), clipped into [lo, hi] so that a rounding error
    at a face never puts a point outside the box."""
    q = B[:, 0] + np.asarray(u, dtype=np.float64) * (B[:, 1] - B[:, 0])
    return np.minimum(np.maximum(q, B[:, 0]), B[:, 1])


def _search_best(F: np.ndarray) -> int:
    """Index of min_j L_j, the earliest among equals; a non-finite L_j counts
    as +inf (index 0 when every L_j is non-finite)."""
    Fm = np.where(np.isfinite(F), F, np.inf)
    return int(np.argmin(Fm))


# ---------------------------------------------------------------------------
#  Starts
# ---------------------------------------------------------------------------
# Custom: the choice of starts from a list of evaluated points. Checked:
# scipy.optimize.minimize, basinhopping, shgo, dual_annealing,
# differential_evolution(polish=True); none starts from given evaluations.
def select_starts(x_iters: Sequence[Sequence[float]], func_vals: Sequence[float],
                  bounds: Sequence[Tuple[float, float]],
                  cfg: RefineConfig = None) -> List[int]:
    """The 0-based indices j of the starts of SPEC 2.9 (b), in the order
    chosen. x_iters (N, p) in q; func_vals (N,); bounds p pairs in q."""
    cfg = RefineConfig() if cfg is None else cfg
    X, F, B = _as_inputs(x_iters, func_vals, bounds)
    ok = np.isfinite(F) & (F < float(cfg.penalty))
    cand = [int(j) for j in np.flatnonzero(ok)]
    # increasing L_j, ties by increasing j (a stable sort of increasing j)
    cand.sort(key=lambda j: (float(F[j]), j))
    U = _to_unit(X, B)
    chosen: List[int] = []
    for j in cand:
        if len(chosen) >= int(cfg.n_starts):
            break
        if all(float(np.max(np.abs(U[j] - U[k]))) >= float(cfg.min_start_dist_unit)
               for k in chosen):
            chosen.append(j)
    return chosen


def _initial_simplex(u0: np.ndarray, step: float) -> np.ndarray:
    """(p + 1, p): u0 and u0 + delta_i e_i, delta_i = +step, or -step when
    u0_i + step > 1."""
    p = u0.shape[0]
    sim = np.empty((p + 1, p), dtype=np.float64)
    sim[0] = u0
    for i in range(p):
        v = u0.copy()
        v[i] = u0[i] + step if u0[i] + step <= 1.0 else u0[i] - step
        sim[i + 1] = v
    return sim


# ---------------------------------------------------------------------------
#  The refinement
# ---------------------------------------------------------------------------
def refine(objective: Callable[[List[float]], float],
           x_iters: Sequence[Sequence[float]], func_vals: Sequence[float],
           bounds: Sequence[Tuple[float, float]],
           cfg: RefineConfig = None) -> RefineResult:
    """The refinement of SPEC 2.9 (b). ValueError on shapes that do not
    agree, an empty x_iters, or bounds with hi <= lo."""
    cfg = RefineConfig() if cfg is None else cfg
    X, F, B = _as_inputs(x_iters, func_vals, bounds)
    p = B.shape[0]
    j_best = _search_best(F)
    x_search = [float(v) for v in X[j_best]]
    fun_search = float(F[j_best])

    starts_idx = select_starts(X, F, B, cfg)
    xs: List[np.ndarray] = []
    fs: List[float] = []
    so: List[int] = []
    starts: List[Dict[str, Any]] = []
    unit_bounds = [(0.0, 1.0)] * p

    for k, j in enumerate(starts_idx):
        u0 = _to_unit(X[j], B)
        u0 = np.minimum(np.maximum(u0, 0.0), 1.0)     # rounding guard only
        f0 = float(F[j])
        first = len(fs)

        def f_unit(u, _k=k):
            q = _to_q(u, B)
            val = float(objective([float(v) for v in q]))
            xs.append(q.copy())
            fs.append(val)
            so.append(_k)
            return val

        res = minimize(f_unit, u0, method="Nelder-Mead", bounds=unit_bounds,
                       options=dict(initial_simplex=_initial_simplex(u0, float(cfg.simplex_step_unit)),
                                    maxfev=int(cfg.maxfev_per_start),
                                    xatol=float(cfg.xatol_unit),
                                    fatol=float(cfg.fatol_rel) * abs(f0),
                                    adaptive=False))
        own = np.asarray(fs[first:], dtype=np.float64)
        if own.size:
            fin = np.where(np.isfinite(own), own, np.inf)
            i_own = first + int(np.argmin(fin))
            x_k, f_k = [float(v) for v in xs[i_own]], float(fs[i_own])
        else:                                          # pragma: no cover
            x_k, f_k = [float(v) for v in X[j]], f0
        starts.append(dict(index=int(j), x0=[float(v) for v in X[j]], f0=f0,
                           x=x_k, fun=f_k, nfev=int(getattr(res, "nfev", len(own))),
                           nit=int(getattr(res, "nit", -1)),
                           status=int(getattr(res, "status", -1)),
                           message=str(getattr(res, "message", ""))))

    n = len(fs)
    x_evals = (np.asarray(xs, dtype=np.float64).reshape(n, p) if n
               else np.empty((0, p), dtype=np.float64))
    f_evals = np.asarray(fs, dtype=np.float64).reshape(n)
    start_of_eval = np.asarray(so, dtype=np.int64).reshape(n)

    x_hat, f_hat, improved, best_start = list(x_search), fun_search, False, -1
    if n:
        fin = np.where(np.isfinite(f_evals), f_evals, np.inf)
        i_loc = int(np.argmin(fin))                    # earliest start, earliest evaluation
        f_loc = float(f_evals[i_loc])
        # strictly below the search's best (ties keep the search's point);
        # a non-finite search best is beaten by any finite local value
        if np.isfinite(f_loc) and (not np.isfinite(fun_search) or f_loc < fun_search):
            x_hat = [float(v) for v in x_evals[i_loc]]
            f_hat, improved, best_start = f_loc, True, int(start_of_eval[i_loc])

    return RefineResult(x=x_hat, fun=float(f_hat), improved=bool(improved),
                        x_search=x_search, fun_search=fun_search,
                        best_start=int(best_start), n_evals=int(n), starts=starts,
                        x_evals=x_evals, f_evals=f_evals, start_of_eval=start_of_eval)


def wrap_minimizer(minimizer: Callable[..., Any],
                   cfg: RefineConfig = None) -> Callable[..., Any]:
    """A drop-in for skopt.gp_minimize, (func, dimensions, **kwargs): calls
    `minimizer`, refines on `func`, returns the same result object with x and
    fun at the estimate and the attribute `refine` (SPEC Block 20).

    Every dimension must be a skopt.space.Real (its low, high are the box in
    q); any other type is a ValueError, raised before the minimizer runs.
    x_iters, func_vals, models and space of the result are not touched. When
    the refinement does not improve, x and fun are left as the minimizer
    returned them (they are the search's best, which is the estimate)."""
    cfg = RefineConfig() if cfg is None else cfg

    def _bounds(dimensions) -> List[Tuple[float, float]]:
        from skopt.space import Real
        out = []
        for d in list(dimensions):
            if not isinstance(d, Real):
                raise ValueError("wrap_minimizer: every dimension must be a "
                                 "skopt.space.Real in q, got %r" % (d,))
            out.append((float(d.low), float(d.high)))
        return out

    def _wrapped(func, dimensions, **kwargs):
        bounds = _bounds(dimensions)
        result = minimizer(func, dimensions, **kwargs)
        rr = refine(func, result.x_iters, result.func_vals, bounds, cfg)
        if rr.improved:
            result.x = list(rr.x)
            result.fun = float(rr.fun)
        result.refine = rr
        return result

    _wrapped.refine_config = cfg
    _wrapped.wrapped_minimizer = minimizer
    return _wrapped
