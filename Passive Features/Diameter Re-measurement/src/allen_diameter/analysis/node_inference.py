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

The forward model is a callable argument: in the pipeline Block 26's
line_profiles (lone node) or a joint hybrid scene with the tier-2 neighbours
held at their features (D-132.2). This module renders nothing itself.

Pure ASCII, LF, for the cluster. No file access, no plotting.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
from scipy.optimize import least_squares

PARAM_NAMES = ("d", "mu", "v0", "z")
FIT_STATUSES = ("converged", "at_bound", "failed")


@dataclass(frozen=True, eq=False)
class NodeFit:
    """Result of fit_node (SPEC Block 30)."""

    theta: np.ndarray            # (4,) = (d um, mu 1/um, v0 um, z_ax um); NaN when no run was finite
    cov: np.ndarray              # (4, 4) (J^T J)^{-1}; NaN rows/cols for a parameter at a bound, all NaN if singular
    se: np.ndarray               # (4,) sqrt(diag(cov))
    cost: float                  # chi2 = sum over the planes used of squared residuals / sigma^2 (dimensionless)
    chi2_per_plane: np.ndarray   # (n_planes,) mean squared residual / sigma^2; NaN for planes not used
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


def fit_node(profiles_gl, B_gl, sigma_gl, forward: Callable, theta0, cfg, planes=None):
    """Fit (d, mu, v0, z_ax) of one node to its profiles (SPEC Block 30).

    profiles_gl: (n_planes, n_v) grey levels, finite. B_gl: scalar or (n_planes,)
    grey levels > 0. sigma_gl: noise SD, grey levels > 0. forward(d, mu, v0, z_ax)
    -> (n_planes, n_v) dimensionless. theta0: (4,) = (d0 um, mu0 1/um, v0 um, z0 um),
    the start; the starts are d0 x multistart_factors with mu0, v0, z0, each clipped
    into the box. planes: (n_planes,) bool, or None for select_planes on forward(theta0).
    cfg: DiameterConfig or InferenceConfig. Returns NodeFit. The inputs are not modified.
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
    lo, hi = _bounds(icfg, th0[3])

    def evaluate(theta):
        tau = np.asarray(forward(float(theta[0]), float(theta[1]), float(theta[2]), float(theta[3])),
                         dtype=float)
        if tau.shape != (n_planes, n_v):
            raise ValueError("forward returned shape %s, expected %s" % (tau.shape, (n_planes, n_v)))
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

    def residuals(theta):
        tau = evaluate(theta)[keep]
        return ((I_used - B_used * tau) / sigma).ravel()

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
            if "not finite" not in str(exc):
                raise
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
    cov = _covariance(np.asarray(best.jac, dtype=float), active)
    with np.errstate(invalid="ignore"):
        se = np.sqrt(np.diag(cov))

    r2 = (best.fun.reshape(I_used.shape)) ** 2
    chi2 = np.full(n_planes, np.nan)
    chi2[keep] = r2.mean(axis=1)
    if not converged:
        status = "failed"
    elif at_bound:
        status = "at_bound"
    else:
        status = "converged"
    return NodeFit(theta=theta, cov=cov, se=se, cost=float(np.sum(r2)), chi2_per_plane=chi2,
                   planes=keep, status=status, at_bound=at_bound, n_starts=len(factors), nfev=nfev)
