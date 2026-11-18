"""The per-node fit of the forward model -- Block 30 in specs/SPEC.md (wave 2; D-129.1, D-130.O9, D-132.2).

For a node with observed profiles I_k(v_n) (grey levels) in the planes k of K,
backgrounds B_k (grey levels), noise SD sigma (grey levels) and a forward
model forward(d, mu, v0, z_ax) -> tau_k(v_n) (dimensionless, shape
(n_planes, n_v)):

    theta_hat = argmin_theta sum_{k in K} sum_n [I_k(v_n) - B_k tau_k(v_n; theta)]^2 / sigma^2

over theta = (d, mu, v0, z_ax) within the bounds of InferenceConfig, by
scipy.optimize.least_squares (trf) from the starts d0 x multistart_factors,
the run of lowest cost kept. Cov(theta_hat) = (J^T J)^{-1}, J the Jacobian of
the residuals already divided by sigma, at theta_hat (Gauss-Newton; sigma
known, no rescaling by the reduced chi2). Per plane chi2_k / N = the mean
squared residual over sigma^2.

With weights w_{k,n} >= 0 (amendment 2; D-169.4, D-171.2) the sum is
sum_{k in K} sum_n w_{k,n} [I_k(v_n) - B_k tau_k(v_n; theta)]^2 / sigma^2, chi2_k / N the weighted mean of the
squared residuals over sigma^2, and the covariance the sandwich A^{-1} B A^{-1}, A = J_w^T J_w,
B = (sqrt(w) J_w)^T (sqrt(w) J_w), J_w the Jacobian of sqrt(w) r / sigma: the weights are not inverse noise
variances, so (J_w^T J_w)^{-1} alone would not be the estimator's covariance. fit_node_recentred solves with
the weights exp(-|v_n - v_c| / lambda) of Block 39 and re-centres them on the fitted centre.

The forward model is a callable argument: in the pipeline Block 26's
line_profiles (lone node) or a joint hybrid scene with the tier-2 neighbours
held at their features (D-132.2). This module renders nothing itself.

Pure ASCII, LF, for the cluster. No file access, no plotting.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import dataclasses

import numpy as np
from scipy.optimize import least_squares

from . import line_weights

PARAM_NAMES = ("d", "mu", "v0", "z")
FIT_STATUSES = ("converged", "at_bound", "failed")


@dataclass(frozen=True, eq=False)
class NodeFit:
    """Result of fit_node (SPEC Block 30)."""

    theta: np.ndarray            # (4,) = (d um, mu 1/um, v0 um, z_ax um); NaN when no run was finite
    cov: np.ndarray              # (4, 4) (J^T J)^{-1}; NaN rows/cols for a parameter at a bound, all NaN if singular
    se: np.ndarray               # (4,) sqrt(diag(cov))
    cost: float                  # chi2 = sum over the planes used of (weighted) squared residuals / sigma^2 (dimensionless)
    chi2_per_plane: np.ndarray   # (n_planes,) (weighted) mean squared residual / sigma^2; NaN for planes not used or of zero weight
    planes: np.ndarray           # (n_planes,) bool, the planes used
    status: str                  # one of FIT_STATUSES
    at_bound: Tuple[str, ...]    # e.g. ("d_hi",), names <param>_lo / <param>_hi
    n_starts: int                # starts run
    nfev: int                    # residual evaluations summed over the starts


def _inference_cfg(cfg):
    """Accept a DiameterConfig (reads .inference) or an InferenceConfig."""
    return cfg.inference if hasattr(cfg, "inference") else cfg


def _backgrounds(B_gl, n_planes):
    B = np.asarray(B_gl, dtype=float)
    if B.ndim == 0:
        B = np.full(n_planes, float(B))
    if B.shape != (n_planes,):
        raise ValueError("B_gl must be a scalar or of shape (n_planes,) = (%d,), got %s" % (n_planes, B.shape))
    if not np.all(np.isfinite(B)) or np.any(B <= 0):
        raise ValueError("B_gl must be finite and > 0")
    return B.copy()


def _sigma(sigma_gl):
    s = float(sigma_gl)
    if not (np.isfinite(s) and s > 0):
        raise ValueError("sigma_gl must be finite and > 0")
    return s


def select_planes(tau0, B_gl, sigma_gl, cfg):
    """keep_k = B_k max_n (1 - tau0_k(v_n)) >= plane_snr_min x sigma (D-130.O9; rule PROVISIONAL).

    tau0: (n_planes, n_v) dimensionless, the forward model at the start;
    B_gl: scalar or (n_planes,) grey levels > 0; sigma_gl > 0 grey levels;
    cfg: DiameterConfig or InferenceConfig. Returns (n_planes,) bool.
    """
    icfg = _inference_cfg(cfg)
    tau = np.asarray(tau0, dtype=float)
    if tau.ndim != 2 or tau.shape[1] == 0:
        raise ValueError("tau0 must be (n_planes, n_v) with n_v >= 1, got %s" % (tau.shape,))
    if not np.all(np.isfinite(tau)):
        raise ValueError("tau0 must be finite")
    B = _backgrounds(B_gl, tau.shape[0])
    sigma = _sigma(sigma_gl)
    peak_dip_gl = B * np.max(1.0 - tau, axis=1)
    return peak_dip_gl >= float(icfg.plane_snr_min) * sigma


def _bounds(icfg, z0):
    """Box of (d, mu, v0, z_ax); z_ax is bounded about the start z0 (config: 'the axis depth about the start')."""
    lo = np.array([icfg.d_bounds_um[0], icfg.mu_bounds_per_um[0], icfg.v0_bounds_um[0],
                   z0 + icfg.z_bounds_um[0]], dtype=float)
    hi = np.array([icfg.d_bounds_um[1], icfg.mu_bounds_per_um[1], icfg.v0_bounds_um[1],
                   z0 + icfg.z_bounds_um[1]], dtype=float)
    if not np.all(lo < hi):
        raise ValueError("InferenceConfig bounds must satisfy lo < hi")
    return lo, hi


def _failed(n_planes, planes, n_starts, nfev):
    nan4 = np.full(4, np.nan)
    return NodeFit(theta=nan4, cov=np.full((4, 4), np.nan), se=nan4.copy(), cost=float("nan"),
                   chi2_per_plane=np.full(n_planes, np.nan), planes=planes, status="failed",
                   at_bound=(), n_starts=n_starts, nfev=nfev)


def _covariance(J, active):
    """(J^T J)^{-1}; all NaN when J is rank-deficient (numpy.linalg.matrix_rank's default tolerance);
    NaN rows and columns for the parameters at a bound."""
    cov = np.full((4, 4), np.nan)
    if np.all(np.isfinite(J)) and np.linalg.matrix_rank(J) == J.shape[1]:
        try:
            cov = np.linalg.inv(J.T @ J)
        except np.linalg.LinAlgError:
            cov = np.full((4, 4), np.nan)
    cov = np.array(cov, dtype=float)
    cov[active, :] = np.nan
    cov[:, active] = np.nan
    return cov


def _covariance_weighted(J_w, sqrt_w, active):
    """The sandwich A^{-1} B A^{-1}, A = J_w^T J_w, B = (sqrt(w) J_w)^T (sqrt(w) J_w), for the weighted fit
    (J_w the Jacobian of sqrt(w) r / sigma, sqrt_w (n_residuals,)); NaN as _covariance."""
    cov = np.full((4, 4), np.nan)
    if np.all(np.isfinite(J_w)) and np.linalg.matrix_rank(J_w) == J_w.shape[1]:
        try:
            A_inv = np.linalg.inv(J_w.T @ J_w)
            JB = sqrt_w[:, None] * J_w
            cov = A_inv @ (JB.T @ JB) @ A_inv
        except np.linalg.LinAlgError:
            cov = np.full((4, 4), np.nan)
    cov = np.array(cov, dtype=float)
    cov[active, :] = np.nan
    cov[:, active] = np.nan
    return cov


def _weights(weights, n_planes, n_v, keep):
    """(n_planes, n_v) float weights from (n_v,) or (n_planes, n_v): finite, >= 0, not all zero on the used
    planes."""
    w = np.array(weights, dtype=float, copy=True)
    if w.shape == (n_v,):
        w = np.broadcast_to(w, (n_planes, n_v)).copy()
    elif w.shape != (n_planes, n_v):
        raise ValueError("weights must be (n_v,) = (%d,) or (n_planes, n_v) = %s, got %s"
                         % (n_v, (n_planes, n_v), w.shape))
    if not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError("weights must be finite and >= 0")
    if not np.any(w[keep] > 0):
        raise ValueError("weights are all zero on the planes used")
    return w


def fit_node(profiles_gl, B_gl, sigma_gl, forward: Callable, theta0, cfg, planes=None, weights=None,
             z_box_centre_um=None):
    """Fit (d, mu, v0, z_ax) of one node to its profiles (SPEC Block 30).

    profiles_gl: (n_planes, n_v) grey levels, finite. B_gl: scalar or (n_planes,)
    grey levels > 0. sigma_gl: noise SD, grey levels > 0. forward(d, mu, v0, z_ax)
    -> (n_planes, n_v) dimensionless. theta0: (4,) = (d0 um, mu0 1/um, v0 um, z0 um),
    the start; the starts are d0 x multistart_factors with mu0, v0, z0, each clipped
    into the box. planes: (n_planes,) bool, or None for select_planes on forward(theta0).
    cfg: DiameterConfig or InferenceConfig. Returns NodeFit. The inputs are not modified.
    weights (amendment 2): None, or (n_v,) / (n_planes, n_v) finite and >= 0, not all zero on the planes
    used: the squared residuals are multiplied by them, chi2_per_plane is their weighted mean and cov the
    sandwich of the module docstring. z_box_centre_um: the centre of the z_ax box (default theta0's z_ax).
    """
    icfg = _inference_cfg(cfg)
    I_obs = np.array(profiles_gl, dtype=float, copy=True)
    if I_obs.ndim != 2 or I_obs.shape[0] == 0 or I_obs.shape[1] == 0:
        raise ValueError("profiles_gl must be (n_planes, n_v), non-empty, got %s" % (I_obs.shape,))
    if not np.all(np.isfinite(I_obs)):
        raise ValueError("profiles_gl must be finite")
    n_planes, n_v = I_obs.shape
    B = _backgrounds(B_gl, n_planes)
    sigma = _sigma(sigma_gl)
    th0 = np.array(theta0, dtype=float, copy=True).reshape(-1)
    if th0.shape != (4,) or not np.all(np.isfinite(th0)):
        raise ValueError("theta0 must be 4 finite numbers (d, mu, v0, z_ax)")
    if not th0[0] > 0:
        raise ValueError("theta0's d must be > 0")
    factors = tuple(float(f) for f in icfg.multistart_factors)
    if len(factors) == 0 or not all(np.isfinite(f) and f > 0 for f in factors):
        raise ValueError("multistart_factors must be non-empty and > 0")
    z_c = th0[3] if z_box_centre_um is None else float(z_box_centre_um)
    if not np.isfinite(z_c):
        raise ValueError("z_box_centre_um must be finite")
    lo, hi = _bounds(icfg, z_c)

    seen = {"nonfinite": False}

    def evaluate(theta):
        tau = np.asarray(forward(float(theta[0]), float(theta[1]), float(theta[2]), float(theta[3])),
                         dtype=float)
        if tau.shape != (n_planes, n_v):
            raise ValueError("forward returned shape %s, expected %s" % (tau.shape, (n_planes, n_v)))
        if not np.all(np.isfinite(tau)):
            seen["nonfinite"] = True
        return tau

    if planes is None:
        tau0 = evaluate(th0)
        if not np.all(np.isfinite(tau0)):
            # no plane can be selected on a non-finite model: the fit fails, it does not raise
            return _failed(n_planes, np.zeros(n_planes, dtype=bool), 0, 1)
        keep = select_planes(tau0, B, sigma, icfg)
    else:
        keep = np.array(planes, copy=True)
        if keep.shape != (n_planes,) or keep.dtype != bool:
            raise ValueError("planes must be a bool array of shape (n_planes,) = (%d,)" % n_planes)
    if not np.any(keep):
        raise ValueError("no plane selected: planes is all False")

    I_used = I_obs[keep]
    B_used = B[keep][:, None]
    if weights is None:
        w_used, sqrt_w = None, None

        def residuals(theta):
            tau = evaluate(theta)[keep]
            return ((I_used - B_used * tau) / sigma).ravel()
    else:
        w_used = _weights(weights, n_planes, n_v, keep)[keep]
        sqrt_w = np.sqrt(w_used)

        def residuals(theta):
            tau = evaluate(theta)[keep]
            return (sqrt_w * (I_used - B_used * tau) / sigma).ravel()

    runs = []
    nfev = 0
    for f in factors:
        x0 = np.clip(np.array([th0[0] * f, th0[1], th0[2], th0[3]]), lo, hi)
        try:
            run = least_squares(residuals, x0, bounds=(lo, hi), method="trf", jac="2-point",
                                x_scale="jac", max_nfev=int(icfg.max_nfev))
        except ValueError as exc:
            # scipy raises "Residuals are not finite in the initial point" for a NaN model:
            # that start failed; any other ValueError (shape, our own validation) propagates
            # ... and a model that turns NaN inside the finite-difference Jacobian reaches the solver's
            # linear algebra, which raises too: a start that met a non-finite forward failed (as Block 29)
            if "not finite" not in str(exc) and not seen["nonfinite"]:
                raise
            seen["nonfinite"] = False
            nfev += 1
            continue
        nfev += int(run.nfev)
        if np.all(np.isfinite(run.x)) and np.all(np.isfinite(run.fun)):
            runs.append(run)
    if not runs:
        return _failed(n_planes, keep, len(factors), nfev)

    converged = [r for r in runs if r.status > 0]
    pool = converged if converged else runs
    best = pool[int(np.argmin([2.0 * r.cost for r in pool]))]  # first in start order on ties

    theta = np.array(best.x, dtype=float)
    active = np.asarray(best.active_mask) != 0
    at_bound = tuple("%s_%s" % (PARAM_NAMES[i], "lo" if best.active_mask[i] < 0 else "hi")
                     for i in range(4) if active[i])
    if w_used is None:
        cov = _covariance(np.asarray(best.jac, dtype=float), active)
    else:
        cov = _covariance_weighted(np.asarray(best.jac, dtype=float), sqrt_w.ravel(), active)
    with np.errstate(invalid="ignore"):
        se = np.sqrt(np.diag(cov))

    r2 = (best.fun.reshape(I_used.shape)) ** 2
    chi2 = np.full(n_planes, np.nan)
    if w_used is None:
        chi2[keep] = r2.mean(axis=1)
    else:
        w_sum = w_used.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            chi2[keep] = np.where(w_sum > 0, r2.sum(axis=1) / w_sum, np.nan)
    if not converged:
        status = "failed"
    elif at_bound:
        status = "at_bound"
    else:
        status = "converged"
    return NodeFit(theta=theta, cov=cov, se=se, cost=float(np.sum(r2)), chi2_per_plane=chi2,
                   planes=keep, status=status, at_bound=at_bound, n_starts=len(factors), nfev=nfev)


def fit_node_recentred(profiles_gl, B_gl, sigma_gl, forward: Callable, theta0, cfg, v_um, lambda_um,
                       centre_offset_um=0.0, keep=None, planes=None):
    """fit_node with the centre weights of Block 39, re-centred on the fitted centre (amendment 2; D-171.2).

    The weights are w_n = exp(-|v_n - v_c| / lambda_um) x keep_n, v_c = v0 + centre_offset_um the tube's
    centre in the line's coordinate v_um ((n_v,) um, strictly increasing; centre_offset_um maps the fit's v0
    onto it). Solve 1 runs fit_node from theta0 with every configured start and v_c from theta0; each later
    solve re-centres the weights on the previous estimate's v0 and starts there with one start, on the same
    planes and the same z_ax box; it stops when v_c moved by at most weight_irls_tol_um or after
    weight_irls_max_iter solves. keep: (n_v,) bool or None, samples of zero weight where False (the bump
    mask). Returns (NodeFit of the last solve, info) with info = dict(n_solves, centre_um (the last solve's
    v_c), converged (bool), lambda_um, nfev (summed), weights ((n_v,) of the last solve)).
    """
    icfg = _inference_cfg(cfg)
    v = np.array(v_um, dtype=float, copy=True).reshape(-1)
    th0 = np.array(theta0, dtype=float, copy=True).reshape(-1)
    if th0.shape != (4,) or not np.all(np.isfinite(th0)):
        raise ValueError("theta0 must be 4 finite numbers (d, mu, v0, z_ax)")
    off = float(centre_offset_um)
    if not np.isfinite(off):
        raise ValueError("centre_offset_um must be finite")
    if keep is None:
        mask = np.ones(v.size, dtype=bool)
    else:
        mask = np.array(keep, copy=True)
        if mask.shape != v.shape or mask.dtype != bool:
            raise ValueError("keep must be a bool array of shape (%d,)" % v.size)
    max_iter, tol = int(icfg.weight_irls_max_iter), float(icfg.weight_irls_tol_um)
    one_start = dataclasses.replace(icfg, multistart_factors=(1.0,))
    centre = float(th0[2]) + off
    start, use, fit, w = th0, planes, None, None
    n_solves, nfev, converged = 0, 0, False
    for it in range(max_iter):
        w = line_weights.centre_weights(v, centre, lambda_um) * mask
        fit = fit_node(profiles_gl, B_gl, sigma_gl, forward, start, icfg if it == 0 else one_start, planes=use,
                       weights=w, z_box_centre_um=float(th0[3]))
        n_solves += 1
        nfev += int(fit.nfev)
        if fit.status == "failed" or not np.all(np.isfinite(fit.theta)):
            break
        use = fit.planes
        new_centre = float(fit.theta[2]) + off
        if abs(new_centre - centre) <= tol:
            converged = True
            break
        if it + 1 < max_iter:
            centre, start = new_centre, fit.theta
    info = dict(n_solves=n_solves, centre_um=centre, converged=converged, lambda_um=float(lambda_um), nfev=nfev,
                weights=w)
    return fit, info
