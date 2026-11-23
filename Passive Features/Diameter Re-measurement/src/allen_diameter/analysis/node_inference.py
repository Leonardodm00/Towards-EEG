"""The per-node fit of the forward model -- Block 30 in specs/SPEC.md (wave 2; D-129.1, D-130.O9, D-132.2;
amendment 2: D-169.4, D-171.2; amendment 3, wave 7: D-221.2-3, D-222.1-2, D-224.1-2, D-225.5-6).

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

The SGT option (amendment 3 (A)-(C)). With shape0 = (beta0, xi0, zeta0) the forward model's kernel is the
skewed generalized t (Block 12's family at the law's FWHM, Block 49) and the search vector is
phi = (d, mu, v0, z_ax, b, xi[, l]), b = log10 beta, l = ln zeta; l only under inference.sgt_skew "fit", under
"off" zeta is held at zeta0. forward(d, mu, v0, z_ax, beta, xi, zeta), beta = 10^b, zeta = e^l. The box:
theta's as above, b in sgt_log10_beta_bounds, xi in sgt_xi_bounds, l in sgt_ln_zeta_bounds; every start
d0 x multistart_factors carries shape0. xi = 0 is a member of the family (the generalized normal, D-052.2): a fit
whose only active bound is xi_lo is "converged" (D-225.5); any other active bound gives "at_bound". theta, cov,
se are theta's block of the covariance in the search parameters (cov_search, n_phi x n_phi); shape =
(beta, xi, zeta) and se_shape by the delta method: se(beta) = beta ln 10 se(b), se(xi), se(zeta) = zeta se(l),
NaN for a held zeta or a parameter at a bound.

The error bars on the fine lattice (amendment 3 (D); D-224.2). With forward_cov -- forward's arity, the model on
the lattice inference.cov_fine_factor -- the covariance is computed at phi_hat from the Jacobian of the residuals
by central differences of forward_cov with the absolute steps h = inference.cov_steps (in the order d, mu, v0,
z_ax, log10 beta, xi, ln zeta; the first n_phi); a parameter within two steps of a bound takes the one-sided
three-point rule inward, (-3 f(phi) + 4 f(phi +- h) - f(phi +- 2h)) / (+-2h); unweighted (J^T J)^{-1}, weighted
the sandwich above, on the solve's planes and weights; NaN rows and columns for the parameters at a bound; all
NaN when forward_cov returns a non-finite value on the planes used at a point the differences need.
cov_source "fine"; without forward_cov "fit" (the solver's own Jacobian, scipy's 2-point).

The forward models are callable arguments: in the pipeline Block 35's (and Block 49's) line_profiles for a lone
node, or a joint hybrid scene with the tier-2 neighbours held at their features (D-132.2). This module renders
nothing itself.

Pure ASCII, LF, for the cluster. No file access, no plotting.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import dataclasses
import math

import numpy as np
from scipy.optimize import least_squares

from . import line_weights

PARAM_NAMES = ("d", "mu", "v0", "z")
SHAPE_NAMES = ("log10_beta", "xi", "ln_zeta")    # the SGT's search parameters (amendment 3 (A))
FIT_STATUSES = ("converged", "at_bound", "failed")
SGT_SKEWS = ("off", "fit")                        # inference.sgt_skew: zeta held at zeta0, or fitted
COV_SOURCES = ("fit", "fine")
MEMBER_BOUNDS = ("xi_lo",)                        # an active bound that leaves a fit "converged" (D-225.5)
# a shape start this close outside the box (times max(1, |bound|)) is taken as rounding and clipped into it: the
# re-centred fit hands a fitted shape back in natural units, and log10(10^b), ln(e^l) can move by an ulp
_BOX_ROUNDING = 1e-12
_LN10 = math.log(10.0)


@dataclass(frozen=True, eq=False)
class NodeFit:
    """Result of fit_node (SPEC Block 30)."""

    theta: np.ndarray            # (4,) = (d um, mu 1/um, v0 um, z_ax um); NaN when no run was finite
    cov: np.ndarray              # (4, 4) theta's block of cov_search; NaN rows/cols for a parameter at a bound, all NaN if singular
    se: np.ndarray               # (4,) sqrt(diag(cov))
    cost: float                  # chi2 = sum over the planes used of (weighted) squared residuals / sigma^2 (dimensionless)
    chi2_per_plane: np.ndarray   # (n_planes,) (weighted) mean squared residual / sigma^2; NaN for planes not used or of zero weight
    planes: np.ndarray           # (n_planes,) bool, the planes used
    status: str                  # one of FIT_STATUSES
    at_bound: Tuple[str, ...]    # e.g. ("d_hi",), names <param>_lo / <param>_hi of param_names
    n_starts: int                # starts run
    nfev: int                    # residual evaluations summed over the starts
    # amendment 3 (wave 7; D-222, D-224, D-225): the SGT option and the fine-lattice error bars
    shape: object = None         # (3,) (beta, xi, zeta) of an SGT fit; None for the Gaussian fit
    se_shape: object = None      # (3,) their SEs in natural units (delta method); NaN for a held zeta or a parameter at a bound
    param_names: Tuple[str, ...] = PARAM_NAMES  # the search parameters, in order (SGT: + "log10_beta", "xi"[, "ln_zeta"])
    cov_search: object = None    # (n_p, n_p) covariance in the search parameters (theta's block is cov); None: as cov
    cov_source: str = "fit"      # "fit" (the fit's own Jacobian) or "fine" (central differences on the fine lattice, D-224.2)


@dataclass(frozen=True, eq=False)
class _Search:
    """The search vector phi of one fit (amendment 3 (A)): theta, then the SGT's (b, xi[, l])."""

    names: Tuple[str, ...]       # PARAM_NAMES (+ SHAPE_NAMES[:2] or [:3])
    lo: np.ndarray               # (n_phi,) the box
    hi: np.ndarray               # (n_phi,)
    sgt: bool                    # shape0 given
    skew_fit: bool               # l = ln zeta searched (sgt_skew "fit")
    zeta_held: float             # zeta0 under "off"
    s0: np.ndarray               # (n_phi - 4,) the shape's start in the search parameters

    @property
    def n(self):
        return len(self.names)

    def args(self, phi):
        """forward's arguments at phi: (d, mu, v0, z_ax), or + (beta, xi, zeta) for the SGT."""
        th = (float(phi[0]), float(phi[1]), float(phi[2]), float(phi[3]))
        if not self.sgt:
            return th
        zeta = math.exp(float(phi[6])) if self.skew_fit else self.zeta_held
        return th + (10.0 ** float(phi[4]), float(phi[5]), zeta)


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


def _search(icfg, lo4, hi4, shape0):
    """The search vector's names, box and shape start (amendment 3 (A), (F)). ValueError for a shape0 that is not
    three finite numbers with beta > 0 and zeta > 0, or whose (log10 beta, xi, ln zeta) lies outside the box (all
    three are checked, as config.validate checks sgt_start_shape; a held zeta's box included), and for shape
    bounds that are not finite and ordered or that leave the family's domain (xi >= 0)."""
    if shape0 is None:
        return _Search(PARAM_NAMES, lo4, hi4, False, False, float("nan"), np.zeros(0))
    try:
        s = np.array(shape0, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        raise ValueError("shape0 must be three finite numbers (beta, xi, zeta), got %r" % (shape0,)) from None
    if s.shape != (3,) or not np.all(np.isfinite(s)) or not (s[0] > 0 and s[2] > 0):
        raise ValueError("shape0 must be three finite numbers (beta, xi, zeta) with beta > 0 and zeta > 0, got %r"
                         % (shape0,))
    skew = icfg.sgt_skew
    if skew not in SGT_SKEWS:
        raise ValueError("inference.sgt_skew must be one of %s, got %r" % (SGT_SKEWS, skew))
    boxes = (tuple(icfg.sgt_log10_beta_bounds), tuple(icfg.sgt_xi_bounds), tuple(icfg.sgt_ln_zeta_bounds))
    if not all(len(b) == 2 for b in boxes):
        raise ValueError("the SGT shape bounds must be pairs (lo, hi)")
    lo_s = np.array([float(b[0]) for b in boxes])
    hi_s = np.array([float(b[1]) for b in boxes])
    if not (np.all(np.isfinite(lo_s)) and np.all(np.isfinite(hi_s)) and np.all(lo_s < hi_s)):
        raise ValueError("InferenceConfig SGT shape bounds must be finite and satisfy lo < hi")
    if not lo_s[1] >= 0.0:
        raise ValueError("inference.sgt_xi_bounds must lie in xi >= 0 (the family's domain)")
    start = np.array([math.log10(s[0]), s[1], math.log(s[2])])
    tol = _BOX_ROUNDING * np.maximum(1.0, np.maximum(np.abs(lo_s), np.abs(hi_s)))
    if np.any(start < lo_s - tol) or np.any(start > hi_s + tol):
        raise ValueError("shape0 %r lies outside the box: log10 beta in %s, xi in %s, ln zeta in %s"
                         % (tuple(float(x) for x in s), boxes[0], boxes[1], boxes[2]))
    start = np.clip(start, lo_s, hi_s)
    n_s = 3 if skew == "fit" else 2
    return _Search(PARAM_NAMES + SHAPE_NAMES[:n_s], np.concatenate([lo4, lo_s[:n_s]]),
                   np.concatenate([hi4, hi_s[:n_s]]), True, skew == "fit", float(s[2]), start[:n_s].copy())


def _cov_steps(icfg, n):
    """The first n of inference.cov_steps, finite and > 0 (ValueError otherwise)."""
    steps = tuple(icfg.cov_steps)
    if len(steps) < n:
        raise ValueError("inference.cov_steps needs %d steps, got %d" % (n, len(steps)))
    h = np.array([float(x) for x in steps[:n]])
    if not np.all(np.isfinite(h)) or np.any(h <= 0):
        raise ValueError("inference.cov_steps must be finite and > 0, got %r" % (steps,))
    return h


def _shape_of(search, phi):
    """(beta, xi, zeta) at phi (natural units; zeta0 when held)."""
    _, _, _, _, beta, xi, zeta = search.args(phi)
    return np.array([beta, xi, zeta], dtype=float)


def _error_fields(search, phi, cov_search):
    """cov, se (theta's block), cov_search and se_shape (delta method) from the covariance in the search
    parameters (amendment 3 (C)). The Gaussian fit's cov_search is a copy of cov."""
    cov = np.array(cov_search[:4, :4], dtype=float)
    with np.errstate(invalid="ignore"):
        se = np.sqrt(np.diag(cov))
    if not search.sgt:
        return dict(cov=cov, se=se, cov_search=cov.copy(), se_shape=None)
    with np.errstate(invalid="ignore"):
        sd = np.sqrt(np.diag(cov_search))
    beta, _xi, zeta = _shape_of(search, phi)
    se_shape = np.array([beta * _LN10 * sd[4], sd[5], zeta * sd[6] if search.skew_fit else np.nan])
    return dict(cov=cov, se=se, cov_search=np.array(cov_search, dtype=float), se_shape=se_shape)


def _failed(n_planes, planes, n_starts, nfev, search=None, cov_source="fit"):
    nan4 = np.full(4, np.nan)
    if search is None or not search.sgt:
        return NodeFit(theta=nan4, cov=np.full((4, 4), np.nan), se=nan4.copy(), cost=float("nan"),
                       chi2_per_plane=np.full(n_planes, np.nan), planes=planes, status="failed",
                       at_bound=(), n_starts=n_starts, nfev=nfev, cov_search=np.full((4, 4), np.nan),
                       cov_source=cov_source)
    n = search.n
    return NodeFit(theta=nan4, cov=np.full((4, 4), np.nan), se=nan4.copy(), cost=float("nan"),
                   chi2_per_plane=np.full(n_planes, np.nan), planes=planes, status="failed",
                   at_bound=(), n_starts=n_starts, nfev=nfev, shape=np.full(3, np.nan),
                   se_shape=np.full(3, np.nan), param_names=search.names, cov_search=np.full((n, n), np.nan),
                   cov_source=cov_source)


def _covariance(J, active):
    """(J^T J)^{-1}; all NaN when J is rank-deficient (numpy.linalg.matrix_rank's default tolerance);
    NaN rows and columns for the parameters at a bound."""
    n = J.shape[1]
    cov = np.full((n, n), np.nan)
    if np.all(np.isfinite(J)) and np.linalg.matrix_rank(J) == n:
        try:
            cov = np.linalg.inv(J.T @ J)
        except np.linalg.LinAlgError:
            cov = np.full((n, n), np.nan)
    cov = np.array(cov, dtype=float)
    cov[active, :] = np.nan
    cov[:, active] = np.nan
    return cov


def _covariance_weighted(J_w, sqrt_w, active):
    """The sandwich A^{-1} B A^{-1}, A = J_w^T J_w, B = (sqrt(w) J_w)^T (sqrt(w) J_w), for the weighted fit
    (J_w the Jacobian of sqrt(w) r / sigma, sqrt_w (n_residuals,)); NaN as _covariance."""
    n = J_w.shape[1]
    cov = np.full((n, n), np.nan)
    if np.all(np.isfinite(J_w)) and np.linalg.matrix_rank(J_w) == n:
        try:
            A_inv = np.linalg.inv(J_w.T @ J_w)
            JB = sqrt_w[:, None] * J_w
            cov = A_inv @ (JB.T @ JB) @ A_inv
        except np.linalg.LinAlgError:
            cov = np.full((n, n), np.nan)
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


def _fine_covariance(forward_cov, ctx):
    """The covariance in the search parameters at phi_hat from central differences of forward_cov
    (amendment 3 (D)); (n_phi, n_phi), NaN as _covariance, all NaN when forward_cov returns a non-finite value
    on the planes used at a point it needs."""
    search, phi, h = ctx["search"], ctx["phi"], ctx["steps"]
    keep, B_used, sqrt_w, sigma = ctx["keep"], ctx["B_used"], ctx["sqrt_w"], ctx["sigma"]
    n = search.n
    expected = (ctx["n_planes"], ctx["n_v"])

    def tau_at(point):
        tau = np.asarray(forward_cov(*search.args(point)), dtype=float)
        if tau.shape != expected:
            raise ValueError("forward_cov returned shape %s, expected %s" % (tau.shape, expected))
        return tau[keep]

    t0 = None
    cols = []
    for i in range(n):
        e = np.zeros(n)
        e[i] = h[i]
        if phi[i] - search.lo[i] <= 2.0 * h[i]:          # within two steps of the lower bound: inward, upwards
            if t0 is None:
                t0 = tau_at(phi)
            dtau = (-3.0 * t0 + 4.0 * tau_at(phi + e) - tau_at(phi + 2.0 * e)) / (2.0 * h[i])
        elif search.hi[i] - phi[i] <= 2.0 * h[i]:        # within two steps of the upper bound: inward, downwards
            if t0 is None:
                t0 = tau_at(phi)
            dtau = (-3.0 * t0 + 4.0 * tau_at(phi - e) - tau_at(phi - 2.0 * e)) / (-2.0 * h[i])
        else:
            dtau = (tau_at(phi + e) - tau_at(phi - e)) / (2.0 * h[i])
        if not np.all(np.isfinite(dtau)):
            return np.full((n, n), np.nan)
        cols.append((-(B_used * dtau) / sigma).ravel())   # d r / d phi_i of r = (I - B tau) / sigma
    J = np.stack(cols, axis=1)
    if sqrt_w is None:
        return _covariance(J, ctx["active"])
    sw = sqrt_w.ravel()
    return _covariance_weighted(sw[:, None] * J, sw, ctx["active"])


def _with_fine_errors(fit, ctx, forward_cov):
    """fit with its error bars from forward_cov (cov_source "fine")."""
    if ctx["phi"] is None:
        return dataclasses.replace(fit, cov_source="fine")
    cov_search = _fine_covariance(forward_cov, ctx)
    return dataclasses.replace(fit, cov_source="fine", **_error_fields(ctx["search"], ctx["phi"], cov_search))


def _solve(profiles_gl, B_gl, sigma_gl, forward, theta0, cfg, planes, weights, z_box_centre_um, shape0,
           need_steps):
    """fit_node without the fine-lattice error bars: (NodeFit with cov_source "fit", ctx), ctx the solve's
    search, estimate, planes, backgrounds, weights and steps for _fine_covariance."""
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
    lo4, hi4 = _bounds(icfg, z_c)
    search = _search(icfg, lo4, hi4, shape0)
    steps = _cov_steps(icfg, search.n) if need_steps else None
    ctx = dict(search=search, phi=None, keep=None, B_used=None, sqrt_w=None, sigma=sigma, active=None,
               n_planes=n_planes, n_v=n_v, steps=steps)

    seen = {"nonfinite": False}

    def evaluate(phi):
        tau = np.asarray(forward(*search.args(phi)), dtype=float)
        if tau.shape != (n_planes, n_v):
            raise ValueError("forward returned shape %s, expected %s" % (tau.shape, (n_planes, n_v)))
        if not np.all(np.isfinite(tau)):
            seen["nonfinite"] = True
        return tau

    if planes is None:
        tau0 = evaluate(np.concatenate([th0, search.s0]) if search.sgt else th0)
        if not np.all(np.isfinite(tau0)):
            # no plane can be selected on a non-finite model: the fit fails, it does not raise
            return _failed(n_planes, np.zeros(n_planes, dtype=bool), 0, 1, search), ctx
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

        def residuals(phi):
            tau = evaluate(phi)[keep]
            return ((I_used - B_used * tau) / sigma).ravel()
    else:
        w_used = _weights(weights, n_planes, n_v, keep)[keep]
        sqrt_w = np.sqrt(w_used)

        def residuals(phi):
            tau = evaluate(phi)[keep]
            return (sqrt_w * (I_used - B_used * tau) / sigma).ravel()
    ctx.update(keep=keep, B_used=B_used, sqrt_w=sqrt_w)

    runs = []
    nfev = 0
    for f in factors:
        x0 = np.clip(np.array([th0[0] * f, th0[1], th0[2], th0[3]]), lo4, hi4)
        if search.sgt:
            x0 = np.concatenate([x0, search.s0])
        try:
            run = least_squares(residuals, x0, bounds=(search.lo, search.hi), method="trf", jac="2-point",
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
        return _failed(n_planes, keep, len(factors), nfev, search), ctx

    converged = [r for r in runs if r.status > 0]
    pool = converged if converged else runs
    best = pool[int(np.argmin([2.0 * r.cost for r in pool]))]  # first in start order on ties

    phi = np.array(best.x, dtype=float)
    theta = phi[:4].copy()
    active = np.asarray(best.active_mask) != 0
    at_bound = tuple("%s_%s" % (search.names[i], "lo" if best.active_mask[i] < 0 else "hi")
                     for i in range(search.n) if active[i])
    if w_used is None:
        cov_search = _covariance(np.asarray(best.jac, dtype=float), active)
    else:
        cov_search = _covariance_weighted(np.asarray(best.jac, dtype=float), sqrt_w.ravel(), active)
    ctx.update(phi=phi, active=active)

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
    elif at_bound and not all(name in MEMBER_BOUNDS for name in at_bound):
        status = "at_bound"
    else:
        status = "converged"
    extra = {}
    if search.sgt:
        extra = dict(shape=_shape_of(search, phi), param_names=search.names)
    fit = NodeFit(theta=theta, cost=float(np.sum(r2)), chi2_per_plane=chi2, planes=keep, status=status,
                  at_bound=at_bound, n_starts=len(factors), nfev=nfev, cov_source="fit",
                  **_error_fields(search, phi, cov_search), **extra)
    return fit, ctx


def fit_node(profiles_gl, B_gl, sigma_gl, forward: Callable, theta0, cfg, planes=None, weights=None,
             z_box_centre_um=None, shape0=None, forward_cov=None):
    """Fit (d, mu, v0, z_ax) of one node to its profiles (SPEC Block 30), with the SGT kernel's shape when
    shape0 is given (amendment 3).

    profiles_gl: (n_planes, n_v) grey levels, finite. B_gl: scalar or (n_planes,)
    grey levels > 0. sigma_gl: noise SD, grey levels > 0. forward(d, mu, v0, z_ax)
    -> (n_planes, n_v) dimensionless. theta0: (4,) = (d0 um, mu0 1/um, v0 um, z0 um),
    the start; the starts are d0 x multistart_factors with mu0, v0, z0, each clipped
    into the box. planes: (n_planes,) bool, or None for select_planes on forward(theta0).
    cfg: DiameterConfig or InferenceConfig. Returns NodeFit. The inputs are not modified.
    weights (amendment 2): None, or (n_v,) / (n_planes, n_v) finite and >= 0, not all zero on the planes
    used: the squared residuals are multiplied by them, chi2_per_plane is their weighted mean and cov the
    sandwich of the module docstring. z_box_centre_um: the centre of the z_ax box (default theta0's z_ax).
    shape0 (amendment 3): None (the Gaussian fit), or (beta0, xi0, zeta0), the SGT shape every start carries;
    then forward(d, mu, v0, z_ax, beta, xi, zeta) and the search vector of the module docstring (zeta held at
    zeta0 under inference.sgt_skew "off"). ValueError for a shape0 that is not three finite numbers with
    beta > 0 and zeta > 0, or that lies outside the box. forward_cov: None, or a callable of forward's arity (the
    model on the fine lattice): the error bars by its central differences at the estimate (cov_source "fine").
    """
    if forward_cov is not None and not callable(forward_cov):
        raise TypeError("forward_cov must be callable or None")
    fit, ctx = _solve(profiles_gl, B_gl, sigma_gl, forward, theta0, cfg, planes, weights, z_box_centre_um,
                      shape0, forward_cov is not None)
    if forward_cov is not None:
        fit = _with_fine_errors(fit, ctx, forward_cov)
    return fit


def fit_node_recentred(profiles_gl, B_gl, sigma_gl, forward: Callable, theta0, cfg, v_um, lambda_um,
                       centre_offset_um=0.0, keep=None, planes=None, shape0=None, forward_cov=None):
    """fit_node with the centre weights of Block 39, re-centred on the fitted centre (amendment 2; D-171.2).

    The weights are w_n = exp(-|v_n - v_c| / lambda_um) x keep_n, v_c = v0 + centre_offset_um the tube's
    centre in the line's coordinate v_um ((n_v,) um, strictly increasing; centre_offset_um maps the fit's v0
    onto it). Solve 1 runs fit_node from theta0 with every configured start and v_c from theta0; each later
    solve re-centres the weights on the previous estimate's v0 and starts there with one start, on the same
    planes and the same z_ax box; it stops when v_c moved by at most weight_irls_tol_um or after
    weight_irls_max_iter solves. keep: (n_v,) bool or None, samples of zero weight where False (the bump
    mask). Returns (NodeFit of the last solve, info) with info = dict(n_solves, centre_um (the last solve's
    v_c), converged (bool), lambda_um, nfev (summed), weights ((n_v,) of the last solve)).
    shape0, forward_cov (amendment 3 (E)): as fit_node; solve 1 starts from shape0, each later solve from the
    previous estimate's shape; the error bars from forward_cov are computed once, after the last solve, with
    its planes and weights.
    """
    if forward_cov is not None and not callable(forward_cov):
        raise TypeError("forward_cov must be callable or None")
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
    start, shape, use, fit, ctx, w = th0, shape0, planes, None, None, None
    n_solves, nfev, converged = 0, 0, False
    for it in range(max_iter):
        w = line_weights.centre_weights(v, centre, lambda_um) * mask
        fit, ctx = _solve(profiles_gl, B_gl, sigma_gl, forward, start, icfg if it == 0 else one_start, use, w,
                          float(th0[3]), shape, forward_cov is not None)
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
            if shape0 is not None:
                shape = fit.shape
    if forward_cov is not None and fit is not None:
        fit = _with_fine_errors(fit, ctx, forward_cov)
    info = dict(n_solves=n_solves, centre_um=centre, converged=converged, lambda_um=float(lambda_um), nfev=nfev,
                weights=w)
    return fit, info
