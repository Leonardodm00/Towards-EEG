"""The in-focus width calibrated on thick nodes -- the fit -- Block 42 in specs/SPEC.md (D-201).

For one node i with the recorded planes K'_i = {k : |k - k*_i| <= h} (h = half_planes), the recorded profiles
I_k(v_n) (grey levels, gl), the noise SD sigma_i of one line sample (gl) and the trace's node t_i (um):

    tau_k(v_n; theta)  = node_inputs.forward_factory(pilot, (U-, U+), node, cfg')(d, mu, v0, z_ax),
                         cfg' = cfg with its renderer replaced by kernel.with_kernel(renderer, sigma0, c),
    theta              = (d um, mu 1/um, v0 um, z_ax um, sigma0 um, c),

i.e. the pipeline's line renderer as it is (D-201.1), whose kernel has the variance sigma0^2 + c G(delta) at
defocus delta, G the configured table's growth (D-201.2). The background is a straight line per plane
(D-201.3): for a given theta, (b0_k, b1_k) is the linear least squares of I_k against tau_k and v tau_k over
the plane's finite samples (variable projection, numpy.linalg.lstsq), and the residuals are

    r_kn(theta) = [I_k(v_n) - (b0_k + b1_k v_n) tau_k(v_n; theta)] / sigma_i      (noise SDs)

over k in K'_i and the finite samples of I. theta_hat = argmin 1/2 sum r^2 within the bounds, from one start
(D-201.4): scipy.optimize.least_squares, method "trf", x_scale "jac", ftol, xtol and max_nfev from the
configuration, jac the callable _Problem.jac (forward differences with the absolute steps diff_steps, taken
backwards at the upper bound or at a non-finite residual). Cov = (J^T J)^-1, J the solver's final Jacobian of
the projected residual (Golub-Pereyra); SE its root diagonal; SE_s = SE sqrt(chi2 / (N - p)), p = 6 + 2 n_planes.

The start (D-201.5; amendment 4, D-203): v0 and z_ax from node_inputs.pilot_start; the k* dip measured against
a straight background L(v) = a + b v, the least squares (numpy.linalg.lstsq) through the finite samples of I_k*
with |v| >= start_ends_um (L = B_k* when either end has fewer than 3 finite samples); d0 = W / width_to_d and
mu0 = -ln max(1 - D / L(c), 1e-3) / (hybrid_faint_factor d0), W, D and c the half-depth width, the depth and the
centre of the k* dip (clearness.dip_features of clearness.smooth_dip(L - I_k*, 1); c = 0 when not finite); the
pilot's d_hat when W is not finite, its mu_hat when D is not finite or <= 0; sigma0 and c the configured starts;
each clipped into its bounds; the bounds of v0 and z_ax are about the start.

The per-cell summary over the ok nodes: the median of sigma0_hat and its MAD (unscaled), the inverse-variance
weighted mean with SE_s, its SE, and chi2_cons = sum (sigma0_i - mean)^2 / SE_s,i^2 with n - 1 degrees of freedom.

Units: um, 1/um, gl. No file access, no plotting. Pure ASCII, LF, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import dataclasses
import math
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import median_abs_deviation

from . import clearness, node_inputs
from ..model import kernel, line_fast

PARAMS = ("d_um", "mu_per_um", "v0_um", "z_ax_um", "sigma0_um", "growth_scale")
STATUSES = ("converged", "max_nfev", "failed")

MIN_PLANES = 3                 # SPEC Block 42: select_planes and the fit need at least 3 planes
MIN_SAMPLES = 3                # SPEC Block 42: vp_background's least squares needs at least 3 finite samples a plane
AT_BOUND_REL = 1e-6            # SPEC Block 42: "within 1e-6 of a bound, relative to the bound's width"
START_SMOOTH_SAMPLES = 1.0     # SPEC Block 42: the start reads clearness.smooth_dip(L - I, 1)
START_TRANSMISSION_FLOOR = 1e-3  # SPEC Block 42: mu0 = -ln max(1 - D/L(c), 1e-3) / (...)
START_END_MIN_SAMPLES = 3      # SPEC Block 42, amendment 4: L is fitted when each end holds at least 3 finite samples
NULL_COMPONENT_TOL = 1e-8      # a parameter with a component above this along J's null space is not estimable
_MEMO_SIZE = 16                # residual evaluations kept (the Jacobian's base point and its 6 + 6 neighbours)


@dataclass(frozen=True)
class SigmaFit:
    """What fit_sigma_node returns (SPEC Block 42). n_planes the fitted planes, n_v the line's samples.

    theta, se, se_scaled: (6,) in PARAMS' order and units (um, 1/um, um, um, um, 1); se from (J^T J)^-1, se_scaled
    = se sqrt(chi2 / dof); b (n_planes, 2) gl: the background (b0, b1 per um) per plane; model_gl (n_planes, n_v):
    (b0 + b1 v) tau at theta; chi2 = sum r^2 (r in noise SDs); chi2_per_plane (n_planes,): the plane's mean r^2
    over its finite samples (chi2_k / N_k, Block 30's convention); n_res the residuals; dof = n_res - (6 + 2
    n_planes); n_eval the forward evaluations (the Jacobian's included); seconds; status one of STATUSES; at_bound
    the PARAMS within 1e-6 of a bound (relative to the bound's width); message. A "failed" fit has NaN in theta,
    se, se_scaled, b, model_gl, chi2 and chi2_per_plane."""
    theta: np.ndarray
    se: np.ndarray
    se_scaled: np.ndarray
    b: np.ndarray
    model_gl: np.ndarray
    chi2: float
    chi2_per_plane: np.ndarray
    n_res: int
    dof: int
    n_eval: int
    seconds: float
    status: str
    at_bound: Tuple[str, ...]
    message: str


class _FitFailed(Exception):
    """Raised inside the solver's callables to end a fit with status "failed" (caught by fit_sigma_node)."""


# ------------------------------------------------------------------ planes ---

def select_planes(ks, k_star, half):
    """The sorted indices of ks with |k - k_star| <= half (SPEC Block 42); ValueError when fewer than 3 or
    half < 1. ks (n_planes,) plane indices; k_star the sharpest plane's index; half (planes)."""
    k = np.asarray(ks, dtype=float).reshape(-1)
    h, ks0 = float(half), float(k_star)
    if not (math.isfinite(h) and h >= 1):
        raise ValueError("half must be finite and >= 1, got %r" % (half,))
    if not (math.isfinite(ks0) and np.all(np.isfinite(k))):
        raise ValueError("ks and k_star must be finite")
    idx = np.flatnonzero(np.abs(k - ks0) <= h).astype(np.int64)
    if idx.size < MIN_PLANES:
        raise ValueError("only %d plane(s) of ks within %g of k* = %g: at least %d are needed"
                         % (idx.size, h, ks0, MIN_PLANES))
    return idx


# ------------------------------------------------------------- background ---

def vp_background(I, tau, v):
    """-> (b (n_planes, 2) gl, resid (n_planes, n_v) gl): per plane the least squares of I against the columns
    tau and v tau over the samples where both are finite (numpy.linalg.lstsq); b[k] = (b0, b1 per um);
    resid = I - (b0 + b1 v) tau, NaN where I or tau is NaN; a plane with fewer than 3 finite samples gives NaN
    for its b and resid (SPEC Block 42). I, tau (n_planes, n_v); v (n_v,) um, finite."""
    Iv = np.asarray(I, dtype=float)
    T = np.asarray(tau, dtype=float)
    vv = np.asarray(v, dtype=float).reshape(-1)
    if Iv.ndim != 2 or T.shape != Iv.shape:
        raise ValueError("I and tau must be (n_planes, n_v) of one shape, got %s and %s" % (Iv.shape, T.shape))
    if vv.shape != (Iv.shape[1],) or not np.all(np.isfinite(vv)):
        raise ValueError("v must be finite of shape (n_v,) = (%d,), got %s" % (Iv.shape[1], vv.shape))
    b = np.full((Iv.shape[0], 2), np.nan)
    resid = np.full(Iv.shape, np.nan)
    for k in range(Iv.shape[0]):
        m = np.isfinite(Iv[k]) & np.isfinite(T[k])
        if int(np.count_nonzero(m)) < MIN_SAMPLES:
            continue
        A = np.column_stack([T[k, m], vv[m] * T[k, m]])
        sol = np.linalg.lstsq(A, Iv[k, m], rcond=None)[0]
        b[k] = sol
        resid[k] = Iv[k] - (sol[0] + sol[1] * vv) * T[k]
    return b, resid


# ---------------------------------------------------------------- forward ---

def forward_with_kernel(pilot, half_length, node, cfg, cache):
    """-> f(d, mu, v0, z_ax, sigma0, c) -> (n_planes, n_v) dimensionless (SPEC Block 42):
    node_inputs.forward_factory(pilot, half_length, node, cfg', cache)(d, mu, v0, z_ax) with cfg' = cfg whose
    renderer is kernel.with_kernel(cfg.renderer, sigma0, c). A NaN array on line_fast.ExtrusionInvalid (as
    forward_factory) and on a kernel with_kernel refuses (sigma0 or c not finite and > 0, a knot whose variance
    would be <= 0). The section cache is shared across kernels (its key holds no kernel); cache None: one
    line_fast.SectionCache made here and shared by every call of f. ValueError now for a kernel family other
    than "gaussian_table" (with_kernel's only refusal that is not about the parameters)."""
    if cfg.renderer.kernel_family != "gaussian_table":
        raise ValueError("forward_with_kernel needs renderer.kernel_family 'gaussian_table', got %r"
                         % (cfg.renderer.kernel_family,))
    cache = line_fast.SectionCache() if cache is None else cache
    shape = (len(node.ks), len(node.v))

    def f(d, mu, v0, z_ax, sigma0, c):
        try:
            rcfg = kernel.with_kernel(cfg.renderer, sigma0, c)
        except ValueError:
            return np.full(shape, np.nan)
        cfg_k = dataclasses.replace(cfg, renderer=rcfg)
        return node_inputs.forward_factory(pilot, half_length, node, cfg_k, cache)(d, mu, v0, z_ax)
    return f


# ------------------------------------------------------------------- start ---

def _bounds(scfg, v0_start, z_start):
    """(lo, hi) (6,): the configured boxes of d, mu, sigma0, c; those of v0 and z_ax about the start."""
    lo = np.array([scfg.d_bounds_um[0], scfg.mu_bounds_per_um[0], v0_start + scfg.v0_bounds_um[0],
                   z_start + scfg.z_bounds_um[0], scfg.sigma0_bounds_um[0], scfg.growth_scale_bounds[0]],
                  dtype=float)
    hi = np.array([scfg.d_bounds_um[1], scfg.mu_bounds_per_um[1], v0_start + scfg.v0_bounds_um[1],
                   z_start + scfg.z_bounds_um[1], scfg.sigma0_bounds_um[1], scfg.growth_scale_bounds[1]],
                  dtype=float)
    if not np.all(lo < hi):
        raise ValueError("sigma_calibration bounds must satisfy lo < hi")
    return lo, hi


def _end_line(I_k, v, ends_um):
    """(a gl, b gl/um) of the start's straight background L(v) = a + b v (SPEC Block 42, amendment 4): the least
    squares (numpy.linalg.lstsq) through the finite samples of I_k (n_v,) with |v| >= ends_um, or None when
    either end (v <= -ends_um, v >= +ends_um) holds fewer than 3 finite samples."""
    I_k = np.asarray(I_k, dtype=float).reshape(-1)
    v = np.asarray(v, dtype=float).reshape(-1)
    e = float(ends_um)
    fin = np.isfinite(I_k)
    left, right = fin & (v <= -e), fin & (v >= e)
    if np.count_nonzero(left) < START_END_MIN_SAMPLES or np.count_nonzero(right) < START_END_MIN_SAMPLES:
        return None
    m = left | right
    A = np.column_stack([np.ones(int(np.count_nonzero(m))), v[m]])
    a, b = np.linalg.lstsq(A, I_k[m], rcond=None)[0]
    return float(a), float(b)


def start_theta(prof, B_gl, v, k_index, pilot, node_xyz, scfg):
    """-> (6,) the start of SPEC Block 42 (D-201.5; amendment 4, D-203): v0 and z_ax from node_inputs.pilot_start
    (z_ax = z_t + its depth offset). The k* dip is measured against the straight background L(v) = a + b v of
    _end_line(I_k*, v, start_ends_um), or L = B_gl[k*] when either end holds fewer than 3 finite samples:
    W, D and c = the half-depth width, the depth and the centre of clearness.dip_features of
    clearness.smooth_dip(L - I_k*, 1) (centre 0, window 1 um: the function's defaults); d0 = W / width_to_d, the
    pilot's d_hat_um when W is not finite; mu0 = -ln max(1 - D / L(c), 1e-3) / (hybrid_faint_factor d0) with c = 0
    when not finite, the pilot's mu_hat_per_um when D is not finite or <= 0 (and when L(c) is not finite and > 0:
    no background to measure the dip against); sigma0 = sigma0_start_um, c = growth_scale_start; then d, mu,
    sigma0, c each clipped into its bounds (the bounds of v0 and z_ax are about the start). Never raises on a
    profile whose dip cannot be measured (NaN samples, no dip): the pilot's values instead. prof (n_planes, n_v)
    gl; B_gl (n_planes,) gl; v (n_v,) um, finite; k_index the index of k* in the planes; pilot: cx_um, cy_um,
    theta_rad, z_sub_um, d_hat_um, mu_hat_per_um; node_xyz (3,) um, the trace's node. ValueError for inputs of
    the wrong shape, k_index outside the planes, a node_xyz or v that is not finite."""
    P = np.asarray(prof, dtype=float)
    if P.ndim != 2 or P.shape[0] == 0 or P.shape[1] == 0:
        raise ValueError("prof must be (n_planes, n_v), non-empty, got %s" % (P.shape,))
    B = np.asarray(B_gl, dtype=float).reshape(-1)
    if B.shape != (P.shape[0],):
        raise ValueError("B_gl must be (n_planes,) = (%d,), got %s" % (P.shape[0], B.shape))
    vv = np.asarray(v, dtype=float).reshape(-1)
    if vv.shape != (P.shape[1],) or not np.all(np.isfinite(vv)):
        raise ValueError("v must be finite of shape (n_v,) = (%d,), got %s" % (P.shape[1], vv.shape))
    k = int(k_index)
    if k != k_index or not (0 <= k < P.shape[0]):
        raise ValueError("k_index must be an index of the planes, got %r" % (k_index,))
    t = np.asarray(node_xyz, dtype=float).reshape(-1)
    if t.shape != (3,) or not np.all(np.isfinite(t)):
        raise ValueError("node_xyz must be 3 finite numbers (um)")
    st = node_inputs.pilot_start(pilot, t)          # (d_hat, mu_hat, v_t, z_sub - z_t)
    v0, z_ax = float(st[2]), float(t[2]) + float(st[3])
    line = _end_line(P[k], vv, scfg.start_ends_um)
    a, b = (float(B[k]), 0.0) if line is None else line
    feats = clearness.dip_features(clearness.smooth_dip((a + b * vv) - P[k], START_SMOOTH_SAMPLES), vv)
    W, D, cen = float(feats["fwhm_um"]), float(feats["depth_gl"]), float(feats["centre_um"])
    d0 = W / float(scfg.width_to_d) if math.isfinite(W) else float(st[0])
    L_c = a + b * (cen if math.isfinite(cen) else 0.0)
    if math.isfinite(D) and D > 0 and math.isfinite(L_c) and L_c > 0 and math.isfinite(d0) and d0 > 0:
        mu0 = -math.log(max(1.0 - D / L_c, START_TRANSMISSION_FLOOR)) / (float(scfg.hybrid_faint_factor) * d0)
    else:
        mu0 = float(st[1])
    theta = np.array([d0, mu0, v0, z_ax, float(scfg.sigma0_start_um), float(scfg.growth_scale_start)])
    lo, hi = _bounds(scfg, 0.0, 0.0)             # the boxes of d, mu, sigma0, c; v0 and z_ax's are about the start
    for j in (0, 1, 4, 5):                       # a NaN (a pilot without d_hat or mu_hat) stays NaN: np.clip keeps it
        theta[j] = np.clip(theta[j], lo[j], hi[j])
    return theta


# -------------------------------------------------------------------- fit ---

class _Problem(object):
    """The projected residual of one node and its Jacobian, with the forward evaluations counted.

    I (n_planes, n_v) gl, the finite samples mask; sigma gl; v (n_v,) um; forward6(d, mu, v0, z_ax, sigma0, c)
    -> (n_planes, n_v); hi (6,) the upper bounds; steps (6,) the absolute difference steps. A memo of the last
    _MEMO_SIZE points (key: the parameters' bytes) serves the solver's repeated calls at one point."""

    def __init__(self, I, sigma, v, forward6, hi, steps):
        self.I = I
        self.mask = np.isfinite(I)
        self.sigma = float(sigma)
        self.v = v
        self.forward6 = forward6
        self.hi = np.asarray(hi, dtype=float)
        self.steps = np.asarray(steps, dtype=float)
        self.n_eval = 0
        self._memo = OrderedDict()

    def evaluate(self, x):
        """(r (n_res,), b (n_planes, 2), tau (n_planes, n_v)) at x (6,)."""
        x = np.ascontiguousarray(x, dtype=np.float64)
        key = x.tobytes()
        hit = self._memo.get(key)
        if hit is not None:
            self._memo.move_to_end(key)
            return hit
        self.n_eval += 1
        tau = np.asarray(self.forward6(*(float(t) for t in x)), dtype=float)
        if tau.shape != self.I.shape:
            raise ValueError("forward6 returned shape %s, expected %s" % (tau.shape, self.I.shape))
        b, resid = vp_background(self.I, tau, self.v)
        out = (resid[self.mask] / self.sigma, b, tau)
        self._memo[key] = out
        while len(self._memo) > _MEMO_SIZE:
            self._memo.popitem(last=False)
        return out

    def residual(self, x):
        """r (n_res,) at x, a copy (the memo's array stays the Jacobian's base point)."""
        return self.evaluate(x)[0].copy()

    # Custom: forward differences with absolute steps, taken backwards at the upper bound or at a non-finite
    # residual. Checked: scipy.optimize.least_squares' "2-point" / diff_step (relative steps, useless for v0 and
    # z_ax near 0) and scipy.optimize.approx_fprime (absolute steps, but neither bounds nor a backward step).
    def jac(self, x):
        """(n_res, 6): column j = [r(x + h_j e_j) - r(x)] / h_j, or [r(x) - r(x - h_j e_j)] / h_j when
        x_j + h_j > hi_j or r(x + h_j e_j) is not finite; _FitFailed when r(x) is not finite or a column is
        not finite either way. The step is the representable difference of the two points."""
        x = np.array(x, dtype=np.float64)
        r0 = self.evaluate(x)[0]
        if not np.all(np.isfinite(r0)):
            raise _FitFailed("the residual is not finite at %s" % (np.array2string(x, precision=6),))
        J = np.empty((r0.size, x.size))
        for j in range(x.size):
            h = float(self.steps[j])
            col = None
            if x[j] + h <= self.hi[j]:
                xp = x.copy()
                xp[j] = x[j] + h
                rp = self.evaluate(xp)[0]
                if np.all(np.isfinite(rp)):
                    col = (rp - r0) / (xp[j] - x[j])
            if col is None:
                xm = x.copy()
                xm[j] = x[j] - h
                rm = self.evaluate(xm)[0]
                if np.all(np.isfinite(rm)):
                    col = (r0 - rm) / (x[j] - xm[j])
            if col is None:
                raise _FitFailed("the Jacobian's column %s is not finite either way at %s"
                                 % (PARAMS[j], np.array2string(x, precision=6)))
            J[:, j] = col
        return J


def _covariance(J):
    """(cov (p, p), how): (J^T J)^-1 when J is finite with full column rank (numpy.linalg.matrix_rank's
    tolerance), how "inv"; otherwise numpy.linalg.pinv(J^T J) with NaN rows and columns for the parameters that
    are not estimable -- a component above NULL_COMPONENT_TOL along J's null space (the SVD's right singular
    vectors of the singular values under matrix_rank's tolerance) --, how "pinv"; all NaN when J is not finite,
    how "not_finite"."""
    J = np.asarray(J, dtype=float)
    p = J.shape[1]
    if not np.all(np.isfinite(J)):
        return np.full((p, p), np.nan), "not_finite"
    JtJ = J.T @ J
    if np.linalg.matrix_rank(J) == p:
        try:
            return np.linalg.inv(JtJ), "inv"
        except np.linalg.LinAlgError:
            pass
    cov = np.array(np.linalg.pinv(JtJ), dtype=float)
    _u, s, vt = np.linalg.svd(J, full_matrices=False)    # J has more rows than columns: vt (p, p)
    tol = float(s.max()) * max(J.shape) * np.finfo(float).eps
    null = vt[s <= tol]
    unidentified = np.any(np.abs(null) > NULL_COMPONENT_TOL, axis=0) if null.size else np.zeros(p, dtype=bool)
    cov[unidentified, :] = np.nan
    cov[:, unidentified] = np.nan
    return cov, "pinv"


def _at_bound(x, lo, hi):
    w = hi - lo
    near = (np.abs(x - lo) <= AT_BOUND_REL * w) | (np.abs(hi - x) <= AT_BOUND_REL * w)
    return tuple(PARAMS[j] for j in range(len(PARAMS)) if near[j])


def _failed(n_planes, n_v, n_res, dof, n_eval, t0, message):
    nan6 = np.full(6, np.nan)
    return SigmaFit(theta=nan6, se=nan6.copy(), se_scaled=nan6.copy(), b=np.full((n_planes, 2), np.nan),
                    model_gl=np.full((n_planes, n_v), np.nan), chi2=float("nan"),
                    chi2_per_plane=np.full(n_planes, np.nan), n_res=int(n_res), dof=int(dof), n_eval=int(n_eval),
                    seconds=time.perf_counter() - t0, status="failed", at_bound=(), message=str(message))


def fit_sigma_node(prof, sigma_gl, v, forward6, theta0, scfg):
    """-> SigmaFit: the variable-projection least squares of SPEC Block 42 on the planes forward6 renders.

    prof (n_planes, n_v) gl, n_planes >= 3, each plane with at least 3 finite samples (NaN samples are left out
    of the residuals); sigma_gl > 0 gl, the noise SD of one line sample; v (n_v,) um; forward6(d, mu, v0, z_ax,
    sigma0, c) -> (n_planes, n_v) dimensionless (forward_with_kernel); theta0 (6,) the start (start_theta), clipped
    into the bounds (those of v0 and z_ax are about theta0's own); scfg the SigmaCalibrationConfig.
    scipy.optimize.least_squares (trf, x_scale "jac", scfg.ftol, xtol, max_nfev; jac _Problem.jac). Never raises
    on a NaN forward or a solver failure: status "failed" with the message (a start that is not finite, a
    non-finite residual at the start, a Jacobian column not finite either way, the solver's linear algebra).
    ValueError for inputs of the wrong shape, too few planes or samples, a sigma_gl that is not > 0, a forward
    of the wrong shape. The inputs are not modified."""
    t0 = time.perf_counter()
    I = np.array(prof, dtype=float, copy=True)
    if I.ndim != 2 or I.shape[1] == 0:
        raise ValueError("prof must be (n_planes, n_v) with n_v >= 1, got %s" % (I.shape,))
    n_planes, n_v = I.shape
    if n_planes < MIN_PLANES:
        raise ValueError("prof has %d plane(s): at least %d are needed" % (n_planes, MIN_PLANES))
    mask = np.isfinite(I)
    if np.any(np.count_nonzero(mask, axis=1) < MIN_SAMPLES):
        raise ValueError("every plane of prof needs at least %d finite samples" % MIN_SAMPLES)
    vv = np.array(v, dtype=float, copy=True).reshape(-1)
    if vv.shape != (n_v,) or not np.all(np.isfinite(vv)):
        raise ValueError("v must be finite of shape (n_v,) = (%d,), got %s" % (n_v, vv.shape))
    sigma = float(sigma_gl)
    if not (math.isfinite(sigma) and sigma > 0):
        raise ValueError("sigma_gl must be finite and > 0, got %r" % (sigma_gl,))
    th0 = np.array(theta0, dtype=float, copy=True).reshape(-1)
    if th0.shape != (6,):
        raise ValueError("theta0 must hold 6 numbers %s, got shape %s" % (PARAMS, th0.shape))
    steps = np.asarray(scfg.diff_steps, dtype=float).reshape(-1)
    if steps.shape != (6,) or not np.all(np.isfinite(steps)) or np.any(steps <= 0):
        raise ValueError("scfg.diff_steps must be 6 finite steps > 0")
    n_res = int(np.count_nonzero(mask))
    dof = n_res - (6 + 2 * n_planes)
    if not np.all(np.isfinite(th0)):
        return _failed(n_planes, n_v, n_res, dof, 0, t0,
                       "the start is not finite: %s" % (np.array2string(th0, precision=6),))
    lo, hi = _bounds(scfg, float(th0[2]), float(th0[3]))
    x0 = np.clip(th0, lo, hi)
    clipped = tuple(PARAMS[j] for j in range(6) if x0[j] != th0[j])
    note = ("; start clipped into the bounds: %s" % ", ".join(clipped)) if clipped else ""
    prob = _Problem(I, sigma, vv, forward6, hi, steps)
    try:
        res = least_squares(prob.residual, x0, jac=prob.jac, bounds=(lo, hi), method="trf", x_scale="jac",
                            ftol=float(scfg.ftol), xtol=float(scfg.xtol), max_nfev=int(scfg.max_nfev))
    except _FitFailed as exc:
        return _failed(n_planes, n_v, n_res, dof, prob.n_eval, t0, str(exc) + note)
    except ValueError as exc:
        if "not finite in the initial point" not in str(exc):    # scipy's message for a non-finite start
            raise
        return _failed(n_planes, n_v, n_res, dof, prob.n_eval, t0, str(exc) + note)
    except np.linalg.LinAlgError as exc:
        return _failed(n_planes, n_v, n_res, dof, prob.n_eval, t0, "the solver's linear algebra: %s" % exc + note)

    x = np.array(res.x, dtype=float)
    r, b, tau = prob.evaluate(x)
    if not np.all(np.isfinite(r)):                # trf never accepts a non-finite point; kept as a guard
        return _failed(n_planes, n_v, n_res, dof, prob.n_eval, t0, "the residual at the solution is not finite" + note)
    model = (b[:, 0:1] + b[:, 1:2] * vv[None, :]) * tau
    r_full = np.full(I.shape, np.nan)
    r_full[mask] = r
    chi2 = float(np.sum(r ** 2))
    chi2_k = np.nansum(r_full ** 2, axis=1) / np.count_nonzero(mask, axis=1)
    cov, how = _covariance(np.asarray(res.jac, dtype=float))
    with np.errstate(invalid="ignore"):
        se = np.sqrt(np.diag(cov))
    se_scaled = se * math.sqrt(chi2 / dof) if dof > 0 else np.full(6, np.nan)
    status = "converged" if res.status > 0 else "max_nfev"
    msg = "%s (scipy status %d, nfev %d, njev %d, covariance %s)%s" % (res.message, res.status, res.nfev,
                                                                       res.njev, how, note)
    return SigmaFit(theta=x, se=se, se_scaled=np.asarray(se_scaled, dtype=float), b=b, model_gl=model, chi2=chi2,
                    chi2_per_plane=chi2_k, n_res=n_res, dof=int(dof), n_eval=int(prob.n_eval),
                    seconds=time.perf_counter() - t0, status=status, at_bound=_at_bound(x, lo, hi), message=msg)


# ---------------------------------------------------------------- summary ---

def summarize(sigma0_um, se_scaled_um, ok):
    """-> dict(n, median_um, mad_um, ivw_mean_um, ivw_se_um, chi2_cons, dof) over the ok nodes (SPEC Block 42):
    the median of sigma0 and its MAD (median absolute deviation from the median, unscaled:
    scipy.stats.median_abs_deviation, scale 1); the inverse-variance weighted mean with weights 1 / se_scaled^2,
    its SE (sum of the weights)^-1/2, chi2_cons = sum (sigma0 - mean)^2 / se_scaled^2 with dof = n - 1. n = 0: every
    entry but n NaN; n = 1: chi2_cons NaN, dof 0. sigma0_um, se_scaled_um (n,) um; ok (n,) bool. ValueError for
    shapes that disagree, ok not bool, an ok node whose sigma0 is not finite or whose se_scaled is not finite
    and > 0."""
    s = np.asarray(sigma0_um, dtype=float).reshape(-1)
    e = np.asarray(se_scaled_um, dtype=float).reshape(-1)
    k = np.asarray(ok)
    if k.dtype != bool:
        raise ValueError("ok must be a bool array, got dtype %s" % (k.dtype,))
    k = k.reshape(-1)
    if not (s.shape == e.shape == k.shape):
        raise ValueError("sigma0_um, se_scaled_um and ok must have one shape, got %s, %s, %s"
                         % (s.shape, e.shape, k.shape))
    x, w_se = s[k], e[k]
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(w_se)) and np.all(w_se > 0)):
        raise ValueError("an ok node needs a finite sigma0 and a finite se_scaled > 0")
    n = int(x.size)
    nan = float("nan")
    if n == 0:
        return dict(n=0, median_um=nan, mad_um=nan, ivw_mean_um=nan, ivw_se_um=nan, chi2_cons=nan, dof=nan)
    w = 1.0 / w_se ** 2
    mean = float(np.sum(w * x) / np.sum(w))
    return dict(n=n, median_um=float(np.median(x)), mad_um=float(median_abs_deviation(x, scale=1.0)),
                ivw_mean_um=mean, ivw_se_um=float(1.0 / math.sqrt(float(np.sum(w)))),
                chi2_cons=float(np.sum(w * (x - mean) ** 2)) if n >= 2 else nan, dof=n - 1)
