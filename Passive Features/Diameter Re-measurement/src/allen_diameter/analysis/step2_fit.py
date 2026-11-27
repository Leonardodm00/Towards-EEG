"""The estimator of step 2: weighted least squares with shared and per-group unknowns, and the covariance of
the shared ones clustered by group -- Block 56 in specs/SPEC.md (section 5, "Kernel calibration, step 2";
D-273.3). Independent of rendering: the forward model arrives as a callable. No file access.

CONTRACT STUB (wave 9, 2026-11-27): signatures and docstrings are binding; the bodies are Block 56's.
Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
    (trf, 2-point Jacobian with its sparsity). ValueError: shapes that do not agree, a negative or non-finite
    weight, a start outside its bounds, a group index outside 0..N-1, no kept observation."""
    raise NotImplementedError("Block 56")
