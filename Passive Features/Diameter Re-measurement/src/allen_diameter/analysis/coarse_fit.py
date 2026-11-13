"""The coarse refinement of a node's features -- Block 29 in specs/SPEC.md (plan step C4; D-130.O2, D-130.O3).

For a node with observed profiles I_k(v_n) (gl) in its planes k in K,
backgrounds B_k (gl) and a shadow model predict(d, mu, v0, dz) -> s_k(v_n)
(dimensionless; in the pipeline Block 25's stretch_profiles), the four
features are

    (d, mu, v0, dz)^ = argmin sum_k sum_n w_n [I_k(v_n) - B_k (1 - s_k(v_n))]^2

within CleaningConfig's refine_* bounds, w_n = exp(-|v_n| / lambda),
lambda = MeasureConfig.sgt_lambda_um, started at the first-stage fit's values
(D-130.O2). The solver is scipy.optimize.least_squares (trf, bounds) on the
residuals sqrt(w_n) [I_k(v_n) - B_k (1 - s_k(v_n))].

  refine_node   the fit; returns a dict (d_um, mu_per_um, v0_um, dz_um, cost,
                rms_gl, status, at_bound, nfev)
  offset_xyz    the image-frame offset (v0 y_hat_x, v0 y_hat_y, dz) a
                features row (Block 23) takes

Custom code: none (a wrapper of the library solver). Pure ASCII, for the
cluster.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

PARAM_NAMES = ("d_um", "mu_per_um", "v0_um", "dz_um")
STATUSES = ("converged", "at_bound", "failed")


def _bounds(cfg):
    """(lo (4,), hi (4,)) from CleaningConfig, in the order of PARAM_NAMES."""
    c = cfg.cleaning
    pairs = (c.refine_d_bounds_um, c.refine_mu_bounds_per_um, c.refine_shift_bounds_um, c.refine_dz_bounds_um)
    lo = np.array([float(p[0]) for p in pairs])
    hi = np.array([float(p[1]) for p in pairs])
    if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)) and np.all(lo < hi)):
        raise ValueError("cleaning.refine_*_bounds must be finite with lower < upper")
    return lo, hi


def _validate(profiles_gl, B_gl, v, theta0, lo, hi):
    """Copies as float64 arrays, after the checks of the Block 29 entry."""
    I = np.array(profiles_gl, dtype=float)
    B = np.array(B_gl, dtype=float)
    vv = np.array(v, dtype=float)
    th = np.array(theta0, dtype=float)
    if I.ndim != 2:
        raise ValueError("profiles_gl must be (n_planes, n_v), got shape %r" % (I.shape,))
    n_planes, n_v = I.shape
    if n_planes < 1 or n_v < 1:
        raise ValueError("profiles_gl must hold at least one plane and one sample")
    if B.shape != (n_planes,):
        raise ValueError("B_gl must be (n_planes,) = (%d,), got shape %r" % (n_planes, B.shape))
    if vv.shape != (n_v,):
        raise ValueError("v must be (n_v,) = (%d,), got shape %r" % (n_v, vv.shape))
    if th.shape != (4,):
        raise ValueError("theta0 must be (d, mu, v0, dz), shape (4,), got shape %r" % (th.shape,))
    if not (np.all(np.isfinite(I)) and np.all(np.isfinite(vv)) and np.all(np.isfinite(th))):
        raise ValueError("profiles_gl, v and theta0 must be finite")
    if not (np.all(np.isfinite(B)) and np.all(B > 0)):
        raise ValueError("B_gl must be finite and > 0")
    outside = [PARAM_NAMES[i] for i in range(4) if not (lo[i] <= th[i] <= hi[i])]
    if outside:
        raise ValueError("theta0 outside the refine bounds for %s" % (", ".join(outside),))
    return I, B, vv, th


def refine_node(profiles_gl, B_gl, v, predict, theta0, cfg):
    """Refine a node's (d, mu, v0, dz) against its observed profiles.

    profiles_gl (n_planes, n_v) gl; B_gl (n_planes,) gl, > 0; v (n_v,) um;
    predict(d, mu, v0, dz) -> (n_planes, n_v) dimensionless shadow s_k(v_n);
    theta0 = (d um, mu 1/um, v0 um, dz um), inside the bounds; cfg a
    DiameterConfig. Returns a dict: d_um, mu_per_um, v0_um, dz_um (floats),
    cost = sum_k sum_n w_n r_kn^2 (gl^2) at the solution, rms_gl = the
    unweighted root mean square of r_kn over all planes and samples (gl),
    status in STATUSES, at_bound (tuple of PARAM_NAMES at a bound, from the
    solver's active_mask), nfev (the solver's count of evaluations; the
    count of predict calls when the fit fails on a non-finite predict, whose
    parameters, cost and rms_gl are then NaN).
    The inputs are not changed. ValueError for shapes that disagree, B <= 0,
    a start outside the bounds, non-finite inputs; a predict whose output is
    not finite ends "failed" without raising.
    """
    lo, hi = _bounds(cfg)
    I, B, vv, th0 = _validate(profiles_gl, B_gl, v, theta0, lo, hi)
    lam = float(cfg.measure.sgt_lambda_um)
    sqrt_w = np.sqrt(np.exp(-np.abs(vv) / lam))[None, :]
    shape = I.shape

    def raw_residuals(theta):
        s = np.asarray(predict(float(theta[0]), float(theta[1]), float(theta[2]), float(theta[3])), dtype=float)
        if s.shape != shape:
            raise ValueError("predict must return shape %r, got %r" % (shape, s.shape))
        return I - B[:, None] * (1.0 - s)

    calls = {"n": 0, "nonfinite": False}

    def residuals(theta):
        calls["n"] += 1
        r = (sqrt_w * raw_residuals(theta)).ravel()
        if not np.all(np.isfinite(r)):
            calls["nonfinite"] = True
        return r

    def failed():
        out = dict(zip(PARAM_NAMES, (float("nan"),) * 4))
        out.update(cost=float("nan"), rms_gl=float("nan"), status="failed", at_bound=(), nfev=int(calls["n"]))
        return out

    if not np.all(np.isfinite(residuals(th0))):
        return failed()
    try:
        res = least_squares(residuals, th0, bounds=(lo, hi), method="trf",
                            max_nfev=int(cfg.cleaning.refine_max_nfev))
    except ValueError:
        # A non-finite predict inside a finite-difference Jacobian reaches the
        # solver's SVD, which raises; only that case is a failed fit.
        if calls["nonfinite"]:
            return failed()
        raise
    x = np.array(res.x, dtype=float)
    r_raw = raw_residuals(x)
    finite = bool(np.all(np.isfinite(r_raw)) and np.all(np.isfinite(x)))
    cost = float(2.0 * res.cost) if finite else float("nan")
    rms = float(np.sqrt(np.mean(r_raw ** 2))) if finite else float("nan")
    at_bound = tuple(PARAM_NAMES[i] for i in range(4) if res.active_mask[i] != 0)
    if not (res.success and finite):
        status = "failed"
    elif at_bound:
        status = "at_bound"
    else:
        status = "converged"
    out = dict(zip(PARAM_NAMES, (float(t) for t in x)))
    out.update(cost=cost, rms_gl=rms, status=status, at_bound=at_bound, nfev=int(res.nfev))
    return out


def offset_xyz(v0_um, dz_um, y_hat):
    """(3,) float64 image-frame offset (v0 y_hat_x, v0 y_hat_y, dz) in um.

    y_hat: the measuring axis, (2,) or (3,) (its first two components are
    used). ValueError for a non-finite input or a y_hat of another length.
    """
    y = np.asarray(y_hat, dtype=float).ravel()
    if y.size not in (2, 3):
        raise ValueError("y_hat must have 2 or 3 components, got %d" % (y.size,))
    v0, dz = float(v0_um), float(dz_um)
    if not (np.all(np.isfinite(y)) and np.isfinite(v0) and np.isfinite(dz)):
        raise ValueError("v0_um, dz_um and y_hat must be finite")
    return np.array([v0 * y[0], v0 * y[1], dz], dtype=float)
