"""fd_jacobian.py -- finite-difference Jacobian with an absolute step in the
unit coordinates of the box (SPEC section 2.10 (b), Block 24).

Pure module: numpy only, no NEURON, no file I/O, no global state.
Contract stubs: signatures and shapes as the Block 24 entry of specs/SPEC.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["FdResult", "to_unit", "to_q", "fd_points", "fd_jacobian",
           "jac_unit_to_q"]


@dataclass
class FdResult:
    J: np.ndarray                 # float64 (N, p): d f / d u
    f0: Optional[np.ndarray]      # float64 (N,) or None when not computed
    n_points: int                 # rows asked of batch_fun
    plan: list                    # fd_points' plan


def to_unit(q, bounds: Sequence[Tuple[float, float]]) -> np.ndarray:
    """u = (q - lo) / (hi - lo); q (p,) or (m, p)."""
    raise NotImplementedError("Block 24")


def to_q(u, bounds: Sequence[Tuple[float, float]]) -> np.ndarray:
    """q = lo + u (hi - lo), clipped into the box; u (p,) or (m, p)."""
    raise NotImplementedError("Block 24")


def fd_points(u, *, h: float = 1e-3, scheme: str = "forward"
              ) -> Tuple[np.ndarray, list]:
    """(U (k, p), plan) of SPEC 2.10 (b)."""
    raise NotImplementedError("Block 24")


def fd_jacobian(batch_fun: Callable[[np.ndarray], np.ndarray], u, *,
                h: float = 1e-3, scheme: str = "forward",
                f0: Optional[np.ndarray] = None) -> FdResult:
    """One call of batch_fun; J (N, p) = d f / d u. SPEC Block 24."""
    raise NotImplementedError("Block 24")


def jac_unit_to_q(J_u: np.ndarray, bounds: Sequence[Tuple[float, float]]
                  ) -> np.ndarray:
    """J_q[:, i] = J_u[:, i] / (hi_i - lo_i)."""
    raise NotImplementedError("Block 24")
