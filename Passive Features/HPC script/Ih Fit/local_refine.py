"""local_refine.py -- the local refinement after the global search.

SPEC section 2.9 (b) and Block 20 (decision D-234): from the best evaluated
points of the global search, bounded Nelder-Mead searches in the unit
coordinates of the box; the estimate is the lowest loss over every
evaluation, global and local.

CONTRACT STUB: the signatures and the docstrings are the contract; the
bodies are written by Block 20's implementer.

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
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np

__all__ = ["RefineConfig", "RefineResult", "select_starts", "refine",
           "wrap_minimizer"]


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
        raise NotImplementedError("Block 20")


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


def select_starts(x_iters: Sequence[Sequence[float]], func_vals: Sequence[float],
                  bounds: Sequence[Tuple[float, float]],
                  cfg: RefineConfig = None) -> List[int]:
    """The 0-based indices j of the starts of SPEC 2.9 (b), in the order
    chosen. x_iters (N, p) in q; func_vals (N,); bounds p pairs in q."""
    raise NotImplementedError("Block 20")


def refine(objective: Callable[[List[float]], float],
           x_iters: Sequence[Sequence[float]], func_vals: Sequence[float],
           bounds: Sequence[Tuple[float, float]],
           cfg: RefineConfig = None) -> RefineResult:
    """The refinement of SPEC 2.9 (b). ValueError on shapes that do not
    agree, an empty x_iters, or bounds with hi <= lo."""
    raise NotImplementedError("Block 20")


def wrap_minimizer(minimizer: Callable[..., Any],
                   cfg: RefineConfig = None) -> Callable[..., Any]:
    """A drop-in for skopt.gp_minimize, (func, dimensions, **kwargs): calls
    `minimizer`, refines on `func`, returns the same result object with x and
    fun at the estimate and the attribute `refine` (SPEC Block 20)."""
    raise NotImplementedError("Block 20")
