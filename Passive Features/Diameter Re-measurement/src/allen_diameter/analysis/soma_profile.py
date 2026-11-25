"""The soma fit over a grid of held depth semi-axes, and the far-background check -- SPEC Block 50 (D-252, D-254.3,
D-256, D-257; wave 8). Implemented 2026-11-25 (wave 8).

No file access, no plotting; Block 46 is not imported (the fit arrives as the callable fit_one). Units: um, gl, gl/um.
Pure ASCII, LF, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np

from .calibration_scene import pixel_centres, planar_background

FLAGS = ("grid_fit_failed", "best_at_grid_edge", "refinement_failed", "refinement_worse", "refined_c_at_bound")
SOURCES = ("grid", "refinement", "none")
# Block 46's PARAMS order (soma_fit.PARAMS): grid_table reads x0 to mu from theta at these indices when a result has no
# attribute of that name. Block 46 is not imported (SPEC Block 50: the fit arrives as a callable).
_THETA_INDEX = {"x0_um": 0, "y0_um": 1, "z0_um": 2, "a_um": 3, "b_um": 4, "phi_rad": 6, "mu_per_um": 7}
_TABLE_ATTRS = ("x0_um", "y0_um", "z0_um", "a_um", "b_um", "phi_rad", "mu_per_um", "area_um2", "r_eq_um")
REFINE_RTOL = 1e-9      # SPEC Block 50 (3): the refinement is refused when its chi2 exceeds the grid fit by more
NEAR_TOL_UM = 1e-9      # SPEC Block 50 (5): the near window's closed ends, with this tolerance (um)


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


def _usable(res):
    """A fit is usable when its status is not "failed" and its chi2 is finite (SPEC Block 50 (1))."""
    try:
        chi2 = float(res.chi2)
    except (TypeError, ValueError):
        return False
    return str(res.status) != "failed" and np.isfinite(chi2)


def _check_grid(c_grid_um, axis_bounds_um):
    c = np.array(c_grid_um, dtype=np.float64).ravel()
    if c.size < 1:
        raise ValueError("c_grid_um: at least one value is needed")
    if not np.all(np.isfinite(c)):
        raise ValueError(f"c_grid_um: values must be finite, got {c.tolist()}")
    if c.size > 1 and not np.all(np.diff(c) > 0):
        raise ValueError(f"c_grid_um: values must be strictly ascending, got {c.tolist()}")
    lo, hi = (float(v) for v in axis_bounds_um)
    if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
        raise ValueError(f"axis_bounds_um: (lo, hi) finite with lo < hi, got {(lo, hi)}")
    if c[0] < lo or c[-1] > hi:
        raise ValueError(f"c_grid_um {c.tolist()} is not inside axis_bounds_um {(lo, hi)}")
    return c, lo, hi


def fit_c_grid(fit_one, c_grid_um, refine, axis_bounds_um):
    """-> CGrid. fit_one(hold: dict, start: (9,) array or None, c_bounds_um: (lo, hi) or None) -> a result with the
    attributes theta (9,), chi2, status, at_bound, n_samples; c_grid_um n >= 1 values (um), finite, strictly ascending,
    inside axis_bounds_um (lo, hi); refine bool (needs n >= 2). SPEC Block 50, (1) to (3).

    The neighbourhood is reported only when the refinement is asked for (refine True and a usable best grid fit);
    exceptions raised by fit_one pass through."""
    c, lo, hi = _check_grid(c_grid_um, axis_bounds_um)
    n = c.size
    refine = bool(refine)
    if refine and n < 2:
        raise ValueError("refine needs at least two grid values (the neighbourhood of the best one)")

    fits = tuple(fit_one({"c_um": float(ci)}, None, None) for ci in c)
    usable = [_usable(f) for f in fits]

    best = -1
    for i in range(n):            # ascending c: a strict < keeps the lower c on a tie
        if usable[i] and (best < 0 or float(fits[i].chi2) < float(fits[best].chi2)):
            best = i

    neighbourhood = None
    refinement = None
    delivered = None
    source = "none"
    if best >= 0:
        delivered, source = fits[best], "grid"
        if refine:
            cb = c[best]
            c_lo = c[best - 1] if best > 0 else cb - abs(cb - c[best + 1])
            c_hi = c[best + 1] if best < n - 1 else cb + abs(cb - c[best - 1])
            neighbourhood = (float(min(max(c_lo, lo), hi)), float(min(max(c_hi, lo), hi)))
            start = np.array(fits[best].theta, dtype=np.float64)
            refinement = fit_one({}, start, neighbourhood)
            if _usable(refinement):
                chi2_g = float(fits[best].chi2)
                if float(refinement.chi2) <= chi2_g + REFINE_RTOL * abs(chi2_g):
                    delivered, source = refinement, "refinement"

    raised = {
        "grid_fit_failed": not all(usable),
        "best_at_grid_edge": n >= 2 and best in (0, n - 1),
        "refinement_failed": refinement is not None and not _usable(refinement),
        "refinement_worse": refinement is not None and _usable(refinement) and source != "refinement",
        "refined_c_at_bound": refinement is not None and "c_um" in tuple(getattr(refinement, "at_bound", ()) or ()),
    }
    return CGrid(c_grid_um=c, fits=fits, best_index=int(best), neighbourhood_um=neighbourhood,
                 refinement=refinement, delivered=delivered, delivered_source=source,
                 flags=tuple(f for f in FLAGS if raised[f]))


def _attr_float(res, name):
    v = getattr(res, name, None)
    if v is None:
        return np.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def grid_table(cgrid):
    """-> dict of (n,) arrays: c_um, chi2, rms_gl, x0_um, y0_um, z0_um, a_um, b_um, phi_rad, mu_per_um, area_um2,
    r_eq_um, usable (bool). SPEC Block 50.

    c_um is the grid's value; chi2 the result's; rms_gl = sqrt(chi2 / n_samples); each other column the result's
    attribute of that name where present, x0 to mu otherwise read from theta in Block 46's PARAMS order; NaN where
    neither is present (or not a number). Values are taken as the results hold them, usable or not: the usable column
    says which to trust (a failed Block 46 fit holds NaN everywhere)."""
    fits = cgrid.fits
    n = len(fits)
    out = {"c_um": np.array(cgrid.c_grid_um, dtype=np.float64)}
    chi2 = np.array([_attr_float(f, "chi2") for f in fits], dtype=np.float64)
    ns = np.array([_attr_float(f, "n_samples") for f in fits], dtype=np.float64)
    out["chi2"] = chi2
    with np.errstate(divide="ignore", invalid="ignore"):
        out["rms_gl"] = np.sqrt(chi2 / ns)
    for name in _TABLE_ATTRS:
        col = np.full(n, np.nan)
        for i, f in enumerate(fits):
            if getattr(f, name, None) is not None:
                col[i] = _attr_float(f, name)
            elif name in _THETA_INDEX and getattr(f, "theta", None) is not None:
                th = np.asarray(f.theta, dtype=np.float64).ravel()
                if th.size > _THETA_INDEX[name]:
                    col[i] = th[_THETA_INDEX[name]]
        out[name] = col
    out["usable"] = np.array([_usable(f) for f in fits], dtype=bool)
    order = ("c_um", "chi2", "rms_gl", "x0_um", "y0_um", "z0_um", "a_um", "b_um", "phi_rad", "mu_per_um",
             "area_um2", "r_eq_um", "usable")
    return {k: out[k] for k in order}


class _Frame:
    """The crop's frame as calibration_scene reads it (left, top px; res_um_px um)."""
    __slots__ = ("left", "top", "res_um_px")

    def __init__(self, left, top, res0_um):
        self.left, self.top, self.res_um_px = left, top, float(res0_um)


def _frame(left, top, res0_um):
    p = float(res0_um)
    if not (np.isfinite(p) and p > 0):
        raise ValueError(f"res0_um must be finite > 0, got {res0_um}")
    return _Frame(left, top, p)


def far_mask(shape, left, top, res0_um, centre_xy_um, far_um, mask):
    """-> (H, W) bool: the pixels whose centre ((left + col) p, (top + row) p) lies at least far_um from centre_xy_um
    (x_s, y_s) um and whose mask (H, W) bool is False. SPEC Block 50, (4)."""
    H, W = (int(v) for v in shape)
    m = np.asarray(mask)
    if m.shape != (H, W) or m.dtype != bool:
        raise ValueError(f"mask must be a bool array of shape {(H, W)}, got {m.dtype} {m.shape}")
    xs, ys = (float(v) for v in centre_xy_um)
    far_um = float(far_um)
    if not (np.isfinite(xs) and np.isfinite(ys) and np.isfinite(far_um) and far_um >= 0):
        raise ValueError("centre_xy_um must be finite and far_um finite >= 0")
    px, py = pixel_centres((H, W), _frame(left, top, res0_um))
    return (np.hypot(px - xs, py - ys) >= far_um) & ~m


def far_background(planes, far, left, top, res0_um, centre_xy_um, trim_sigmas, min_frac=0.0):
    """-> (b (K, 3) [gl, gl/um, gl/um], ok (K,) bool, n_used (K,) int): per plane the trimmed least-squares plane
    b0 + b1 (x - x_s) + b2 (y - y_s) through the far pixels (calibration_scene.planar_background); NaN where not ok.
    planes (K, H, W) gl; far (H, W) bool. SPEC Block 50, (4)."""
    D = np.asarray(planes)
    f = np.asarray(far)
    if D.ndim != 3 or f.shape != D.shape[1:] or f.dtype != bool:
        raise ValueError(f"planes (K, H, W) and far (H, W) bool must agree, got {D.shape} and {f.dtype} {f.shape}")
    xs, ys = (float(v) for v in centre_xy_um)
    frame = _frame(left, top, res0_um)
    K = D.shape[0]
    b = np.full((K, 3), np.nan)
    ok = np.zeros(K, dtype=bool)
    n_used = np.zeros(K, dtype=np.int64)
    for k in range(K):
        bk, nk, okk = planar_background(D[k], f, frame, xs, ys, trim_sigmas, min_frac)
        ok[k] = bool(okk)
        n_used[k] = int(nk)
        if ok[k]:
            b[k] = bk
    return b, ok, n_used


def background_check(beta, z_planes_um, z0_um, far_b, far_ok, near_um, tol_gl):
    """-> dict(planes, delta_gl, mean_gl, max_abs_gl, n_planes, near_um, tol_gl, flag, status). beta, far_b (K, 3);
    z_planes_um (K,) um; z0_um the fitted centre depth (um); far_ok (K,) bool; near_um, tol_gl >= 0. SPEC Block 50,
    (5). A plane enters when |z_k - z0| <= near_um (+ 1e-9 um), far_ok[k], and both rows beta[k], far_b[k] are
    finite."""
    beta = np.asarray(beta, dtype=np.float64)
    fb = np.asarray(far_b, dtype=np.float64)
    z = np.asarray(z_planes_um, dtype=np.float64)
    fok = np.asarray(far_ok)
    if z.ndim != 1:
        raise ValueError(f"z_planes_um must be (K,), got {z.shape}")
    K = z.size
    if beta.shape != (K, 3) or fb.shape != (K, 3) or fok.shape != (K,):
        raise ValueError(f"beta, far_b (K, 3) and far_ok (K,) must agree with z_planes_um (K = {K}), got "
                         f"{beta.shape}, {fb.shape}, {fok.shape}")
    z0, near, tol = float(z0_um), float(near_um), float(tol_gl)
    if not np.isfinite(z0):
        raise ValueError(f"z0_um must be finite, got {z0_um}")
    if not (np.isfinite(near) and near >= 0):
        raise ValueError(f"near_um must be finite >= 0, got {near_um}")
    if not (np.isfinite(tol) and tol >= 0):
        raise ValueError(f"tol_gl must be finite >= 0, got {tol_gl}")
    with np.errstate(invalid="ignore"):
        sel = ((np.abs(z - z0) <= near + NEAR_TOL_UM) & fok.astype(bool)
               & np.all(np.isfinite(beta), axis=1) & np.all(np.isfinite(fb), axis=1))
    idx = np.flatnonzero(sel).astype(np.int64)
    delta = beta[idx, 0] - fb[idx, 0]
    if idx.size == 0:
        mean, mx, flag, status = float("nan"), float("nan"), False, "no_planes"
    else:
        mean = float(np.mean(delta))
        mx = float(np.max(np.abs(delta)))
        flag, status = bool(abs(mean) > tol), "ok"
    return {"planes": idx, "delta_gl": delta, "mean_gl": mean, "max_abs_gl": mx, "n_planes": int(idx.size),
            "near_um": near, "tol_gl": tol, "flag": flag, "status": status}
