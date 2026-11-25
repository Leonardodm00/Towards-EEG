"""The soma fit over a grid of held depth semi-axes, and the far-background check -- SPEC Block 50 (D-252, D-254.3,
D-256, D-257; wave 8). CONTRACT STUB: the signatures, shapes and units are SPEC.md's; no logic yet.

No file access, no plotting; Block 46 is not imported (the fit arrives as the callable fit_one). Units: um, gl, gl/um.
Pure ASCII, LF, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np

FLAGS = ("grid_fit_failed", "best_at_grid_edge", "refinement_failed", "refinement_worse", "refined_c_at_bound")
SOURCES = ("grid", "refinement", "none")


@dataclass(frozen=True)
class CGrid:
    """fit_c_grid's result (SPEC Block 50). c_grid_um (n,) float64 um; fits: the n results in grid order; best_index
    int (-1: no usable fit); neighbourhood_um (lo, hi) um or None; refinement: its result or None; delivered: a result
    or None; delivered_source one of SOURCES; flags: names from FLAGS, in that order."""
    c_grid_um: np.ndarray
    fits: Tuple[Any, ...]
    best_index: int
    neighbourhood_um: Optional[Tuple[float, float]]
    refinement: Any
    delivered: Any
    delivered_source: str
    flags: Tuple[str, ...]


def fit_c_grid(fit_one, c_grid_um, refine, axis_bounds_um):
    """-> CGrid. fit_one(hold: dict, start: (9,) array or None, c_bounds_um: (lo, hi) or None) -> a result with the
    attributes theta (9,), chi2, status, at_bound, n_samples; c_grid_um n >= 1 values (um), finite, strictly ascending,
    inside axis_bounds_um (lo, hi); refine bool (needs n >= 2). SPEC Block 50, (1) to (3)."""
    raise NotImplementedError("Block 50, wave 8: fit_c_grid")


def grid_table(cgrid):
    """-> dict of (n,) arrays: c_um, chi2, rms_gl, x0_um, y0_um, z0_um, a_um, b_um, phi_rad, mu_per_um, area_um2,
    r_eq_um (NaN where a result lacks the attribute or is not usable), usable (bool). SPEC Block 50."""
    raise NotImplementedError("Block 50, wave 8: grid_table")


def far_mask(shape, left, top, res0_um, centre_xy_um, far_um, mask):
    """-> (H, W) bool: the pixels whose centre ((left + col) p, (top + row) p) lies at least far_um from centre_xy_um
    (x_s, y_s) um and whose mask (H, W) bool is False. SPEC Block 50, (4)."""
    raise NotImplementedError("Block 50, wave 8: far_mask")


def far_background(planes, far, left, top, res0_um, centre_xy_um, trim_sigmas, min_frac=0.0):
    """-> (b (K, 3) [gl, gl/um, gl/um], ok (K,) bool, n_used (K,) int): per plane the trimmed least-squares plane
    b0 + b1 (x - x_s) + b2 (y - y_s) through the far pixels (calibration_scene.planar_background); NaN where not ok.
    planes (K, H, W) gl; far (H, W) bool. SPEC Block 50, (4)."""
    raise NotImplementedError("Block 50, wave 8: far_background")


def background_check(beta, z_planes_um, z0_um, far_b, far_ok, near_um, tol_gl):
    """-> dict(planes, delta_gl, mean_gl, max_abs_gl, n_planes, near_um, tol_gl, flag, status). beta, far_b (K, 3);
    z_planes_um (K,) um; z0_um the fitted centre depth (um); far_ok (K,) bool; near_um, tol_gl >= 0. SPEC Block 50,
    (5)."""
    raise NotImplementedError("Block 50, wave 8: background_check")
