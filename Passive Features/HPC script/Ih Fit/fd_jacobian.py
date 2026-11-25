"""fd_jacobian.py -- finite-difference Jacobian with an absolute step in the
unit coordinates of the box (SPEC section 2.10 (b), Block 24).

Pure module: numpy only, no NEURON, no file I/O, no global state.

Notation (SPEC 2.9 (b), 2.10 (b)): q in R^p the optimiser's coordinates, the
box prod_i [lo_i, hi_i], u_i = (q_i - lo_i) / (hi_i - lo_i) in [0, 1], f a map
[0, 1]^p -> R^N, h the ABSOLUTE step in u (default 1e-3).

  forward  d f / d u_i ~ (f(u + delta_i e_i) - f(u)) / delta_i,
           delta_i = +h, or -h when u_i + h > 1
  central  (f(u + h e_i) - f(u - h e_i)) / (2 h), replaced for that axis by the
           forward rule when u_i + h > 1 or u_i - h < 0

All the perturbed points of one Jacobian go to the batch function in ONE call
(the fit evaluates them in parallel). In optimiser coordinates
d / d q_i = (hi_i - lo_i)^-1 d / d u_i (jac_unit_to_q).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["FdResult", "to_unit", "to_q", "fd_points", "fd_jacobian",
           "jac_unit_to_q"]

_SCHEMES = ("forward", "central")


@dataclass
class FdResult:
    J: np.ndarray                 # float64 (N, p): d f / d u
    f0: Optional[np.ndarray]      # float64 (N,) or None when not computed
    n_points: int                 # rows asked of batch_fun
    plan: list                    # fd_points' plan


# ----------------------------------------------------------------- validation
def _bounds_arrays(bounds) -> Tuple[np.ndarray, np.ndarray]:
    """(lo (p,), hi (p,)) float64 from a sequence of p pairs; ValueError unless
    every pair is finite with hi > lo."""
    try:
        B = np.asarray(bounds, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("bounds must be a sequence of (lo, hi) pairs: %s" % exc)
    if B.ndim != 2 or B.shape[1] != 2 or B.shape[0] < 1:
        raise ValueError("bounds must be a sequence of p >= 1 pairs (lo, hi); "
                         "got shape %s" % (B.shape,))
    if not np.all(np.isfinite(B)):
        raise ValueError("bounds must be finite")
    lo, hi = B[:, 0], B[:, 1]
    if np.any(hi <= lo):
        bad = [int(i) for i in np.flatnonzero(hi <= lo)]
        raise ValueError("bounds need hi > lo; violated on axes %s" % bad)
    return lo, hi


def _points_array(x, p: int, name: str) -> np.ndarray:
    """float64 copy of x, shape (p,) or (m, p); ValueError otherwise."""
    X = np.array(x, dtype=np.float64)
    if X.ndim not in (1, 2) or X.shape[-1] != p:
        raise ValueError("%s must have shape (%d,) or (m, %d); got %s"
                         % (name, p, p, X.shape))
    return X


def _check_u(u) -> np.ndarray:
    try:
        U = np.array(u, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("u must be a 1-D float array: %s" % exc)
    if U.ndim != 1:
        raise ValueError("u must be 1-D; got shape %s" % (U.shape,))
    if U.size < 1:
        raise ValueError("u must have at least one coordinate")
    if not np.all(np.isfinite(U)):
        raise ValueError("u must be finite")
    if np.any(U < 0.0) or np.any(U > 1.0):
        raise ValueError("u must lie in [0, 1]^p; got min %r, max %r"
                         % (float(U.min()), float(U.max())))
    return U


def _check_h(h) -> float:
    try:
        hf = float(h)
    except (TypeError, ValueError):
        raise ValueError("h must be a number in (0, 0.5]; got %r" % (h,))
    if not (0.0 < hf <= 0.5):          # also refuses nan
        raise ValueError("h must be in (0, 0.5]; got %r" % (h,))
    return hf


def _check_scheme(scheme) -> str:
    if scheme not in _SCHEMES:
        raise ValueError("scheme must be one of %s; got %r" % (_SCHEMES, scheme))
    return scheme


# ----------------------------------------------------------------- coordinates
def to_unit(q, bounds: Sequence[Tuple[float, float]]) -> np.ndarray:
    """u = (q - lo) / (hi - lo); q (p,) or (m, p). Not clipped."""
    lo, hi = _bounds_arrays(bounds)
    Q = _points_array(q, lo.size, "q")
    return (Q - lo) / (hi - lo)


def to_q(u, bounds: Sequence[Tuple[float, float]]) -> np.ndarray:
    """q = lo + u (hi - lo), clipped into the box; u (p,) or (m, p)."""
    lo, hi = _bounds_arrays(bounds)
    U = _points_array(u, lo.size, "u")
    return np.clip(lo + U * (hi - lo), lo, hi)


def jac_unit_to_q(J_u: np.ndarray, bounds: Sequence[Tuple[float, float]]
                  ) -> np.ndarray:
    """J_q[:, i] = J_u[:, i] / (hi_i - lo_i) (chain rule, u_i linear in q_i)."""
    lo, hi = _bounds_arrays(bounds)
    J = np.asarray(J_u, dtype=np.float64)
    if J.ndim != 2 or J.shape[1] != lo.size:
        raise ValueError("J_u must have shape (N, %d); got %s" % (lo.size, J.shape))
    return J / (hi - lo)


# ----------------------------------------------------------------- differences
def fd_points(u, *, h: float = 1e-3, scheme: str = "forward"
              ) -> Tuple[np.ndarray, list]:
    """(U (k, p), plan) of SPEC 2.10 (b).

    U lists the perturbed points axis by axis (0, 1, ...): forward, one row per
    axis (k = p); central, two rows (plus, then minus) per two-sided axis and one
    row per axis that falls back to the forward rule. plan[i] is
    ("forward", row, delta) or ("central", row_plus, row_minus, h), rows into U.
    """
    u = _check_u(u)
    h = _check_h(h)
    scheme = _check_scheme(scheme)
    p = u.size
    rows: List[np.ndarray] = []
    plan: list = []
    for i in range(p):
        two_sided = (scheme == "central"
                     and not (u[i] + h > 1.0) and not (u[i] - h < 0.0))
        if two_sided:
            up = u.copy()
            up[i] = u[i] + h
            um = u.copy()
            um[i] = u[i] - h
            rows.append(up)
            rows.append(um)
            plan.append(("central", len(rows) - 2, len(rows) - 1, h))
        else:
            delta = -h if u[i] + h > 1.0 else h
            ud = u.copy()
            ud[i] = u[i] + delta
            rows.append(ud)
            plan.append(("forward", len(rows) - 1, delta))
    U = np.array(rows, dtype=np.float64).reshape(len(rows), p)
    return U, plan


def fd_jacobian(batch_fun: Callable[[np.ndarray], np.ndarray], u, *,
                h: float = 1e-3, scheme: str = "forward",
                f0: Optional[np.ndarray] = None) -> FdResult:
    """One call of batch_fun; J (N, p) = d f / d u. SPEC Block 24.

    batch_fun maps float64 (k, p) in unit coordinates to float64 (k, N). It is
    called once: with U alone when f0 is given or not needed (central, every
    axis two-sided), else with u as the first row followed by U. The returned
    f0 is the computed f(u), the given f0 when one was passed, or None.
    """
    # Custom: batched differences with an absolute step in unit coordinates and
    # all points in one call. Checked: scipy.optimize.approx_fprime and
    # scipy.optimize._numdiff.approx_derivative evaluate one point per call and
    # take a relative step by default (SPEC Block 24).
    u = _check_u(u)
    U, plan = fd_points(u, h=h, scheme=scheme)
    p = u.size
    needs_f0 = any(entry[0] == "forward" for entry in plan)
    f0_given = None
    if f0 is not None:
        f0_given = np.array(f0, dtype=np.float64)
        if f0_given.ndim != 1:
            raise ValueError("f0 must be 1-D; got shape %s" % (f0_given.shape,))
        if not np.all(np.isfinite(f0_given)):
            raise ValueError("f0 must be finite")
    ask_u = needs_f0 and f0_given is None
    X = np.vstack([u[None, :], U]) if ask_u else U
    k = X.shape[0]

    out = batch_fun(X.copy())
    try:
        F = np.asarray(out, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("batch_fun must return a float array: %s" % exc)
    if F.ndim != 2 or F.shape[0] != k:
        raise ValueError("batch_fun returned shape %s for %d points; expected (%d, N)"
                         % (np.shape(out), k, k))
    if not np.all(np.isfinite(F)):
        bad = sorted(set(int(r) for r in np.argwhere(~np.isfinite(F))[:, 0]))
        raise ValueError("batch_fun returned non-finite values in rows %s" % bad)
    N = F.shape[1]
    if f0_given is not None and f0_given.shape[0] != N:
        raise ValueError("f0 has length %d; batch_fun returns N = %d"
                         % (f0_given.shape[0], N))

    if ask_u:
        f0_out = F[0].copy()
        FU = F[1:]
    else:
        f0_out = None if f0_given is None else f0_given
        FU = F
    base = f0_out

    J = np.empty((N, p), dtype=np.float64)
    for i, entry in enumerate(plan):
        if entry[0] == "central":
            _, rp, rm, hh = entry
            J[:, i] = (FU[rp] - FU[rm]) / (2.0 * hh)
        else:
            _, r, delta = entry
            J[:, i] = (FU[r] - base) / delta
    return FdResult(J=J, f0=f0_out, n_points=int(k), plan=plan)
