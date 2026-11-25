"""identifiability.py -- the identifiability table of a protocol from stored
sensitivities (SPEC section 2.10 (d), Block 26).

Pure module: numpy only. `layout` is duck-typed on Block 23's ResidualLayout
(masks, pre_masks, sqrt_a, D, slices, n_residuals).
Contract stubs: signatures and shapes as the Block 26 entry of specs/SPEC.md.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

__all__ = ["jacobian_from_sensitivities", "svd_summary", "noise_residuals",
           "predicted_scatter", "relative_difference", "identifiability_row"]


def jacobian_from_sensitivities(S_list: Sequence[np.ndarray], layout) -> np.ndarray:
    """J (N, p): sqrt_a[b][:, None] * S_list[b][masks[b]] / D[b]."""
    raise NotImplementedError("Block 26")


def svd_summary(J: np.ndarray) -> Dict[str, object]:
    """sigma (p,) descending; V (p, p), row j = v_{j+1}; ratio."""
    raise NotImplementedError("Block 26")


def noise_residuals(layout, noise_list: Sequence[np.ndarray]) -> np.ndarray:
    """r_eta (N,) of SPEC 2.10 (d)."""
    raise NotImplementedError("Block 26")


def predicted_scatter(J: np.ndarray, R_eta: np.ndarray) -> Dict[str, np.ndarray]:
    """dq (K, p), sd (p,), corr (p, p) of SPEC 2.10 (d)."""
    raise NotImplementedError("Block 26")


def relative_difference(J_a: np.ndarray, J_b: np.ndarray) -> float:
    """||J_a - J_b||_F / ||J_b||_F."""
    raise NotImplementedError("Block 26")


def identifiability_row(J: np.ndarray, names: Sequence[str],
                        R_eta: Optional[np.ndarray] = None) -> Dict[str, float]:
    """One table row: sigma_k, sigma_ratio, v_p_<name>, v_pm1_<name>, and
    with R_eta sd_lin_<name>, corr_<a>_<b>."""
    raise NotImplementedError("Block 26")
