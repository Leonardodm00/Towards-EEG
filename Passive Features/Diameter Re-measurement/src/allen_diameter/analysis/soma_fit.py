"""The soma fit -- Block 46 in specs/SPEC.md (D-215, D-216, D-218, D-219; wave 8: D-252, D-254, D-255, D-256, D-258).

An ellipsoid fitted to the planes about the soma. Data: the soma's crop D (n, H, W) grey levels (gl) on the planes
ks (n,) with valid (n,), the crop's corner (left, top) px, the pixel size p and the plane spacing dz (um); the traced
soma's centre (x_s, y_s, z_s) um (image frame; depth in stage units, plane k at depth k dz) and radius r_s um; a mask
(H, W) bool, True where a pixel is left out (Allen's traced stems, at their traced radius: D-241).

    planes K   select_planes: the valid planes with |k dz - z_s| <= plane_half_um and k - k_0 divisible by
               plane_step, k_0 the valid plane nearest z_s (ties: the lower k);
    region R   the pixels whose centre ((left + col) p, (top + row) p) lies within r_R of (x_s, y_s) and whose mask
               is False; r_R = region_radius_um when it is a number, otherwise region_radius_scale x r_s (D-255);
    theta      (x0, y0, z0, a, b, c, phi, mu, gamma): Block 45's theta_g and the kernel's far slope gamma, fitted
               (far_slope "fit") or held at slope0 ("fixed", the default since D-240); with sigma0 "fit" (D-258) the
               kernel's in-focus width sigma0_um is one more fitted parameter, after them; any of the first eight can
               be held at a given value (hold; D-252, D-256);
    model      background "tilted" (the default, D-254.1): D_k(x, y) ~ (b0_k + b1_k u + b2_k v) tau_k(theta) on R,
               u = x - x_s, v = y - y_s (um, from the traced centre, a fixed point); for each theta the three
               coefficients of each plane are the least-squares solution of the columns [tau, u tau, v tau] on R
               (numpy.linalg.lstsq per plane), and r_k the misfit left. background "constant" (the model before wave
               8, a labelled comparison): D_k ~ B_k tau_k, B_k = sum_R D_k tau_k / sum_R tau_k^2 (D-219.3), beta_k =
               (B_k, 0, 0). Both are variable projection: the backgrounds are solved exactly at every theta.

theta_hat = argmin 1/2 sum_K sum_R r^2 within the bounds -- |x0 - x_s|, |y0 - y_s| <= shift_xy_um,
|z0 - z_s| <= shift_z_um, a, b, c in axis_bounds_um (c in c_bounds_um when given), phi in [-pi, pi], mu in
mu_bounds_per_um, gamma in slope_bounds, sigma0 in sigma0_bounds_um -- by scipy.optimize.least_squares (method
"trf", x_scale "jac", max_nfev, ftol, xtol of the configuration) over the fitted parameters only, with jac the
callable _Problem.jac: forward differences with the absolute steps diff_steps (sigma0_diff_step_um for sigma0), taken
backwards where the forward step would cross an upper bound or gives a non-finite residual (as Block 42).
cov_fit = (J^T J)^-1 chi2 / dof over the fitted parameters, J the solver's final Jacobian of the projected residual at
theta_hat (Golub-Pereyra), chi2 = sum r^2 (gl^2), dof = |K| |R| - (fitted parameters) - n_bg |K|, n_bg = 3
("tilted") or 1 ("constant"); numpy.linalg.pinv when J^T J is singular, with NaN for the parameters along its null
space (Block 42's rule); SE = sqrt(diag). A held parameter has NaN in se and in its row and column of cov. A
parameter is at a bound when within 1e-3 of the bound interval's width from it (D-236).

The start (start_theta, on the plane k_0; D-219.5), or a given start (its slope entry ignored): the traced centre and
size, with the shadow's orientation (its weighted second moments) and an aspect of the same mean radius, mu_0 from the
darkest core; clipped into the bounds by fit_soma, the held entries then set to their values. Derived: the membrane
area S (ellipsoid_area, Legendre's form), the volume 4/3 pi a b c, the equivalent radius sqrt(S / 4 pi) (D-216.3)
and SE(S) by the delta method over the fitted axes among (a, b, c) (a held axis contributes no variance), the
gradient of S by central differences.

Units: um, rad, 1/um, um of SD per um of defocus, gl. No file access, no plotting. Pure ASCII, LF, for the cluster
(hpc-python-compat).
"""
from __future__ import annotations

import math
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.optimize import least_squares
from scipy.special import ellipeinc, ellipkinc

from ..config import SomaFitConfig
from .soma_view import plane_range

PARAMS = ("x0_um", "y0_um", "z0_um", "a_um", "b_um", "c_um", "phi_rad", "mu_per_um", "slope_um_per_um")
STATUSES = ("converged", "max_nfev", "failed")
FAR_SLOPES = ("fit", "fixed")
BACKGROUNDS = ("tilted", "constant")   # SPEC Block 46, wave 8 (D-254.1): "tilted" the default
SIGMA0_MODES = ("fixed", "fit")        # SPEC Block 46, wave 8 (D-258): "fixed" the default
SIGMA0_NAME = "sigma0_um"              # the fitted in-focus width's name in SomaFit.fitted (after PARAMS)

MIN_PLANES = 3                   # SPEC Block 46: select_planes and the fit need at least 3 planes
AT_BOUND_REL = 1e-3              # SPEC Block 46: "within 1e-3 of the bound interval's width from it" (D-236: the
                                 # trust-region solver stops short of a bound, by 1e-4 of the width on the pilot)
# The start's settings are SomaFitConfig fields since D-220.4 (start_far_radii, start_core_radii,
# start_core_percentile, start_ratio_clip, start_ratio_none; SPEC Block 46, amendment 1).
AREA_GRAD_REL_STEP = 1e-5        # the central differences of S (SE(S)): a step of 1e-5 of each axis (not in SPEC)
NULL_COMPONENT_TOL = 1e-8        # a parameter with a component above this along J's null space is not estimable
_MEMO_SIZE = 4                   # residual evaluations kept: the solver's points (the Jacobian's steps are not kept)


@dataclass(frozen=True)
class SomaFit:
    """The soma fit's result (SPEC Block 46 and its wave-8 amendment). K the planes fitted, n_fit = len(fitted).

    theta (9,) in the order and units of PARAMS (um, um, um, um, um, um, rad, 1/um, um/um; theta[8] = slope0 exactly
    when gamma is held, a held parameter its value exactly); se (9,) = sqrt(diag(cov)), NaN for a held parameter;
    cov (9, 9) the fitted parameters' block of cov_fit placed in PARAMS order (NaN in a held parameter's row and
    column, and for parameters along J's null space); fitted: the names fitted, in PARAMS order, then "sigma0_um"
    when the in-focus blur is fitted; B_gl (K,) = beta[:, 0], the background at the traced centre (under "constant"
    the plane's scale) (gl); ks (K,) the plane numbers fitted; chi2 = sum r^2 (gl^2); dof = n_samples - n_fit -
    n_bg K (n_bg 3 "tilted", 1 "constant"); n_samples = K |R|; status one of STATUSES; at_bound: the fitted names
    at a bound; n_eval: the forward evaluations (the Jacobian's included); seconds (wall); area_um2, se_area_um2,
    volume_um3, r_eq_um: S(a, b, c), its SE (delta method over the fitted axes), 4/3 pi a b c, sqrt(S / 4 pi);
    message. Wave 8: beta (K, 3) the planes' backgrounds (gl, gl/um, gl/um; (B_k, 0, 0) under "constant");
    background the setting ("tilted" | "constant"); sigma0_um, se_sigma0_um: the fitted in-focus width and its SE
    under sigma0 "fit" (um), under "fixed" the argument as passed (NaN when None) and NaN; cov_fit (n_fit, n_fit)
    the scaled covariance of the fitted parameters in fitted order. A "failed" fit has NaN in theta, se, cov, B_gl,
    beta, cov_fit, chi2, the fitted sigma0 and the derived quantities, and at_bound empty."""
    theta: np.ndarray
    se: np.ndarray
    cov: np.ndarray
    fitted: Tuple[str, ...]
    B_gl: np.ndarray
    ks: np.ndarray
    chi2: float
    dof: int
    n_samples: int
    status: str
    at_bound: Tuple[str, ...]
    n_eval: int
    seconds: float
    area_um2: float
    se_area_um2: float
    volume_um3: float
    r_eq_um: float
    message: str
    beta: np.ndarray
    background: str
    sigma0_um: float
    se_sigma0_um: float
    cov_fit: np.ndarray


class _FitFailed(Exception):
    """Raised inside the solver's callables to end a fit with status "failed" (caught by fit_soma)."""


class _ForwardShapeError(ValueError):
    """The forward model returned an array of the wrong shape (an input that disagrees: re-raised)."""


# ------------------------------------------------------------------ checks ---

def _finite(x, name):
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise ValueError("%s must be a finite number, got %r" % (name, x))
    if not math.isfinite(v):
        raise ValueError("%s must be finite, got %r" % (name, x))
    return v


def _positive(x, name):
    v = _finite(x, name)
    if not v > 0:
        raise ValueError("%s must be > 0, got %r" % (name, x))
    return v


def _whole(x, name):
    v = _finite(x, name)
    if v != round(v):
        raise ValueError("%s must be a whole number, got %r" % (name, x))
    return int(round(v))


def _centre(centre_um):
    c = np.asarray(centre_um, dtype=float).reshape(-1)
    if c.shape != (3,) or not np.all(np.isfinite(c)):
        raise ValueError("centre_um must hold 3 finite numbers (um), got %r" % (centre_um,))
    return float(c[0]), float(c[1]), float(c[2])


def _bool_mask(m, shape, name):
    a = np.asarray(m)
    if a.dtype != bool:
        raise ValueError("%s must be a bool array, got dtype %s" % (name, a.dtype))
    if a.shape != tuple(shape):
        raise ValueError("%s must have shape %s, got %s" % (name, tuple(shape), a.shape))
    return a


def _plane_numbers(ks):
    k = np.asarray(ks, dtype=float).reshape(-1)
    if k.size == 0 or not np.all(np.isfinite(k)) or np.any(k != np.round(k)):
        raise ValueError("ks must be a non-empty array of whole plane numbers")
    k = np.round(k).astype(np.int64)
    if np.any(np.diff(k) <= 0):
        raise ValueError("ks must be strictly ascending")
    return k


def _pixel_centres(shape, left, top, p):
    """(x (W,), y (H,)) um: pixel (row, col) of a crop with corner (left, top) is centred on
    ((left + col) p, (top + row) p) (Block 44, render.block_fine_grid)."""
    H, W = int(shape[0]), int(shape[1])
    return (int(left) + np.arange(W)) * p, (int(top) + np.arange(H)) * p


def _nearest_valid(k, v, z_um, dz_um):
    """The index of the valid plane whose depth k dz is nearest z_um (the first, i.e. the lower k, on a tie);
    None when no plane is valid."""
    if not v.any():
        return None
    d = np.abs(k * dz_um - z_um)
    d[~v] = np.inf
    return int(np.argmin(d))


# -------------------------------------------------------------------- area ---

def ellipsoid_area(a_um, b_um, c_um):
    """-> the surface area (um^2) of the ellipsoid with semi-axes a, b, c (um). The axes sorted a >= b >= c:
    S = 2 pi c^2 + (2 pi a b / sin psi) (E(psi | m) sin^2 psi + F(psi | m) cos^2 psi), cos psi = c / a,
    m = a^2 (b^2 - c^2) / (b^2 (a^2 - c^2)) (Legendre's form; scipy.special.ellipkinc, ellipeinc, parameter m);
    the sphere 4 pi a^2, the prolate spheroid (b = c, m = 0) 2 pi c^2 + 2 pi a c arcsin(e) / e and the oblate
    (a = b, m = 1) 2 pi a^2 + 2 pi c^2 artanh(e) / e, e = sin psi, in closed form. Symmetric in its arguments
    (bit for bit: the axes are sorted). ValueError unless all are finite and > 0."""
    a, b, c = sorted((_positive(a_um, "a_um"), _positive(b_um, "b_um"), _positive(c_um, "c_um")), reverse=True)
    if a == c:                                         # a = b = c
        return 4.0 * math.pi * a * a
    sin_psi = math.sqrt((a - c) * (a + c)) / a         # e = sqrt(1 - c^2/a^2), without the cancellation
    if b == c:                                         # prolate, m = 0
        return 2.0 * math.pi * c * c + 2.0 * math.pi * a * c * math.asin(sin_psi) / sin_psi
    if a == b:                                         # oblate, m = 1
        return 2.0 * math.pi * a * a + 2.0 * math.pi * c * c * math.atanh(sin_psi) / sin_psi
    psi = math.atan2(math.sqrt((a - c) * (a + c)), c)
    m = (a * a * (b - c) * (b + c)) / (b * b * (a - c) * (a + c))
    cos_psi = c / a
    F = float(ellipkinc(psi, m))
    E = float(ellipeinc(psi, m))
    return 2.0 * math.pi * c * c + (2.0 * math.pi * a * b / sin_psi) * (E * sin_psi ** 2 + F * cos_psi ** 2)


# Custom: central differences of ellipsoid_area with a relative step. Checked: scipy.differentiate.derivative
# (adaptive, many vectorised evaluations; S is smooth and cheap, and SPEC asks for central differences).
def _area_gradient(a, b, c):
    """(3,) um: dS/da, dS/db, dS/dc by central differences with the steps AREA_GRAD_REL_STEP x each axis."""
    x = [float(a), float(b), float(c)]
    g = np.empty(3)
    for j in range(3):
        h = AREA_GRAD_REL_STEP * x[j]
        xp, xm = list(x), list(x)
        xp[j] += h
        xm[j] -= h
        g[j] = (ellipsoid_area(*xp) - ellipsoid_area(*xm)) / (xp[j] - xm[j])
    return g


# ------------------------------------------------------------------ planes ---

def select_planes(ks, valid, z_start_um, dz_um, half_um, step):
    """-> the indices i into ks (ascending, int64) with valid[i], ks[i] inside Block 44's window
    soma_view.plane_range(z_start, half_um, dz) (|k dz - z_start| <= half_um, closed, with a tolerance of 1e-9 of a
    plane; D-220.5) and ks[i] - k_0 divisible by step, k_0 the k of the valid plane nearest z_start (its depth k dz nearest z_start; ties: the
    lower k). ks (n,) whole plane numbers, strictly ascending; valid (n,) bool; z_start_um, dz_um, half_um um.
    ValueError when fewer than 3 are selected, step < 1 (or not whole) or half_um < 0, and for inputs that
    disagree (shapes, ks not ascending, z_start not finite, dz not > 0)."""
    k = _plane_numbers(ks)
    v = np.asarray(valid, dtype=bool).reshape(-1)
    if v.shape != k.shape:
        raise ValueError("valid must have the shape of ks, %s, got %s" % (k.shape, v.shape))
    z = _finite(z_start_um, "z_start_um")
    dz = _positive(dz_um, "dz_um")
    h = _finite(half_um, "half_um")
    if h < 0:
        raise ValueError("half_um must be >= 0, got %r" % (half_um,))
    s = _whole(step, "step")
    if s < 1:
        raise ValueError("step must be >= 1, got %r" % (step,))
    i0 = _nearest_valid(k, v, z, dz)
    if i0 is None:
        raise ValueError("no valid plane: at least %d are needed" % MIN_PLANES)
    k_lo, k_hi = plane_range(z, h, dz)                   # Block 44's closed window, 1e-9 of a plane (D-220.5)
    keep = v & (k >= k_lo) & (k <= k_hi) & ((k - k[i0]) % s == 0)
    idx = np.flatnonzero(keep).astype(np.int64)
    if idx.size < MIN_PLANES:
        raise ValueError("only %d valid plane(s) within %g um of z = %g um at step %d from k_0 = %d: at least %d "
                         "are needed" % (idx.size, h, z, s, int(k[i0]), MIN_PLANES))
    return idx


# ------------------------------------------------------------------- scale ---

# Custom: the one-column least squares per plane in closed form (one division). Checked: numpy.linalg.lstsq per
# plane (the same solution for a single column, one call per plane and evaluation).
def _vp_columns(DR, TR):
    """(B (K,), resid (K, n)) from the data DR and the model TR (K, n) on the region's pixels:
    B_k = sum_n D tau / sum_n tau^2 (NaN where sum tau^2 is not > 0) and resid = D - B_k tau."""
    num = np.einsum("kn,kn->k", DR, TR)
    den = np.einsum("kn,kn->k", TR, TR)
    with np.errstate(divide="ignore", invalid="ignore"):
        B = np.where(den > 0, num / den, np.nan)
    return B, DR - B[:, None] * TR


def vp_scale(D, tau, region):
    """-> (B (K,) gl, resid (K, n_region) gl): per plane the least-squares scale B_k = sum_R D tau / sum_R tau^2
    over the region's pixels and the residual D - B_k tau there (the region's pixels in C order, as D[k][region]);
    B_k NaN when sum_R tau^2 is not > 0. D, tau (K, H, W) (gl, dimensionless); region (H, W) bool. ValueError for
    shapes that disagree, a region that is not bool or is empty."""
    Dv = np.asarray(D, dtype=float)
    T = np.asarray(tau, dtype=float)
    if Dv.ndim != 3 or T.shape != Dv.shape:
        raise ValueError("D and tau must be (K, H, W) of one shape, got %s and %s" % (Dv.shape, T.shape))
    reg = _bool_mask(region, Dv.shape[1:], "region")
    if not reg.any():
        raise ValueError("region is empty")
    return _vp_columns(Dv[:, reg], T[:, reg])


# Custom: the per-plane loop of the three-column projection (SPEC Block 46, wave 8: no library projects out a
# per-plane linear background of a stacked model). Library: numpy.linalg.lstsq per plane (SVD-based; its rank with
# the default rcond decides rank < 3). Checked: numpy.linalg.qr (stacked; no rank decision), scipy.linalg.lstsq (the
# same as numpy's here), the normal equations (refused by the entry).
def _vp_tilted(DR, TR, UR, VR):
    """(beta (K, 3), resid (K, n)) from the data DR and the model TR (K, n) on the region's pixels and the pixel
    coordinates UR, VR (n,) um from the traced centre: per plane beta_k = argmin ||d_k - [t, u t, v t] beta||^2
    (numpy.linalg.lstsq) and resid = d_k - M_k beta_k; NaN coefficients and residual for a plane whose M_k is not
    finite or has rank < 3."""
    K, n = DR.shape
    beta = np.full((K, 3), np.nan)
    resid = np.full((K, n), np.nan)
    for kk in range(K):
        t = TR[kk]
        M = np.column_stack((t, UR * t, VR * t))
        if not np.all(np.isfinite(M)):
            continue
        sol, _res, rank, _sv = np.linalg.lstsq(M, DR[kk], rcond=None)
        if rank < 3:
            continue
        beta[kk] = sol
        resid[kk] = DR[kk] - M @ sol
    return beta, resid


def _vp_project(DR, TR, UR, VR, background):
    """(beta (K, 3), resid (K, n)): "tilted" -> _vp_tilted; "constant" -> _vp_columns with beta = (B, 0, 0)."""
    if background == "tilted":
        return _vp_tilted(DR, TR, UR, VR)
    B, resid = _vp_columns(DR, TR)
    beta = np.zeros((B.size, 3))
    beta[:, 0] = B
    beta[~np.isfinite(B), :] = np.nan
    return beta, resid


def vp_background(D, tau, region, u_um, v_um):
    """-> (beta (K, 3) [gl, gl/um, gl/um], resid (K, n_region) gl): per plane the least-squares coefficients of the
    columns [tau, u tau, v tau] on the region's pixels (numpy.linalg.lstsq) and the residual D - M beta there (the
    region's pixels in C order, as D[k][region] and vp_scale). D, tau (K, H, W) (gl, dimensionless); region (H, W)
    bool; u_um, v_um (H, W) float, um from the traced centre (u = x - x_s, v = y - y_s). NaN coefficients and
    residual for a plane whose columns are not finite or have rank < 3. ValueError for shapes that disagree, a region
    that is not bool or is empty. SPEC Block 46, the amendment of wave 8 (D-254.1)."""
    Dv = np.asarray(D, dtype=float)
    T = np.asarray(tau, dtype=float)
    if Dv.ndim != 3 or T.shape != Dv.shape:
        raise ValueError("D and tau must be (K, H, W) of one shape, got %s and %s" % (Dv.shape, T.shape))
    reg = _bool_mask(region, Dv.shape[1:], "region")
    if not reg.any():
        raise ValueError("region is empty")
    U = np.asarray(u_um, dtype=float)
    V = np.asarray(v_um, dtype=float)
    if U.shape != reg.shape or V.shape != reg.shape:
        raise ValueError("u_um and v_um must have shape %s, got %s and %s" % (reg.shape, U.shape, V.shape))
    return _vp_tilted(Dv[:, reg], T[:, reg], U[reg], V[reg])


def background_planes(beta, shape, left, top, res0_um, centre_xy_um):
    """-> (K, H, W) gl: b0_k + b1_k u + b2_k v at the pixel centres of a crop, u = x - x_s, v = y - y_s (um), pixel
    (row, col) centred on ((left + col) p, (top + row) p). beta (K, 3) (gl, gl/um, gl/um); shape (H, W); left, top
    whole px; res0_um p um (> 0); centre_xy_um (x_s, y_s) um. A caller draws the model as
    background_planes(...) * tau. ValueError for a beta that is not (K, 3), a shape that is not two whole numbers
    >= 1, a corner not whole, a pixel size not finite > 0, a centre not two finite numbers. SPEC Block 46, the
    amendment of wave 8."""
    bt = np.asarray(beta, dtype=float)
    if bt.ndim != 2 or bt.shape[1] != 3:
        raise ValueError("beta must be (K, 3), got shape %s" % (bt.shape,))
    sh = tuple(np.asarray(shape).reshape(-1).tolist())
    if len(sh) != 2:
        raise ValueError("shape must be (H, W), got %r" % (shape,))
    H, W = _whole(sh[0], "shape[0]"), _whole(sh[1], "shape[1]")
    if H < 1 or W < 1:
        raise ValueError("shape must be >= 1 in both axes, got %r" % (shape,))
    p = _positive(res0_um, "res0_um")
    c = np.asarray(centre_xy_um, dtype=float).reshape(-1)
    if c.shape != (2,) or not np.all(np.isfinite(c)):
        raise ValueError("centre_xy_um must hold 2 finite numbers (um), got %r" % (centre_xy_um,))
    x, y = _pixel_centres((H, W), _whole(left, "left"), _whole(top, "top"), p)
    u = (x - c[0])[None, None, :]
    v = (y - c[1])[None, :, None]
    return bt[:, 0, None, None] + bt[:, 1, None, None] * u + bt[:, 2, None, None] * v


# ------------------------------------------------------------------- start ---

def start_theta(block_k0, left, top, res0_um, centre_um, radius_um, region, slope0, scfg=None):
    """-> theta0 (9,) = (x_s, y_s, z_s, a_0, b_0, r_s, phi_0, mu_0, slope0) (D-219.5), from the plane block_k0 (H, W)
    gl, the crop's corner (left, top) px, res0_um, the traced centre centre_um (3,) um (image frame; depth in stage
    units) and radius radius_um um, the region (H, W) bool; scfg the SomaFitConfig whose start_* fields are read
    (None: SomaFitConfig(); D-220.4), shown here at their defaults. rho: a pixel centre's distance from (x_s, y_s);
    B_far = the median of block_k0 over the region with rho > 1.5 r_s (start_far_radii; the whole region if none);
    mu_0 = -ln(varrho) / (2 r_s), varrho = clip(the 5th percentile (start_core_percentile; numpy.percentile) of
    block_k0 over the region with rho <= 0.5 r_s (start_core_radii), divided by B_far, 0.05, 0.95 (start_ratio_clip)),
    0.5 (start_ratio_none) if none; weights w = max(0, 1 - block_k0 / B_far) on the region with rho <= 1.5 r_s, C their weighted covariance of (x, y) about the weighted mean (numpy.cov,
    aweights w, bias True), lambda_1 >= lambda_2 its eigenvalues (numpy.linalg.eigh):
    phi_0 = atan2(2 C_xy, C_xx - C_yy) / 2, a_0 = r_s (lambda_1 / lambda_2)^(1/4),
    b_0 = r_s (lambda_2 / lambda_1)^(1/4) (a_0 = b_0 = r_s, phi_0 = 0 when sum w = 0 or lambda_2 <= 0). Not clipped
    here: fit_soma clips the start into the bounds. ValueError for shapes that disagree, a region that is not bool
    or is empty, a plane not finite on the region, a B_far that is not > 0, a centre, radius, pixel size or slope0
    that is not finite (radius and pixel size > 0), a corner that is not whole."""
    P = np.asarray(block_k0, dtype=float)
    if P.ndim != 2 or P.size == 0:
        raise ValueError("block_k0 must be a non-empty (H, W) plane, got shape %s" % (P.shape,))
    reg = _bool_mask(region, P.shape, "region")
    if not reg.any():
        raise ValueError("region is empty")
    xs, ys, zs = _centre(centre_um)
    rs = _positive(radius_um, "radius_um")
    p = _positive(res0_um, "res0_um")
    g0 = _finite(slope0, "slope0")
    q = SomaFitConfig() if scfg is None else scfg
    far_r, core_r = float(q.start_far_radii), float(q.start_core_radii)
    x, y = _pixel_centres(P.shape, _whole(left, "left"), _whole(top, "top"), p)
    X = np.broadcast_to(x[None, :], P.shape)[reg]
    Y = np.broadcast_to(y[:, None], P.shape)[reg]
    vals = P[reg]
    if not np.all(np.isfinite(vals)):
        raise ValueError("block_k0 must be finite on the region")
    rho = np.hypot(X - xs, Y - ys)
    far = rho > far_r * rs
    B_far = float(np.median(vals[far] if far.any() else vals))
    if not B_far > 0:
        raise ValueError("the start's far background B_far must be > 0, got %r" % (B_far,))
    core = rho <= core_r * rs
    if core.any():
        lo_c, hi_c = (float(t) for t in q.start_ratio_clip)
        varrho = float(np.clip(np.percentile(vals[core], float(q.start_core_percentile)) / B_far, lo_c, hi_c))
    else:
        varrho = float(q.start_ratio_none)
    mu0 = -math.log(varrho) / (2.0 * rs)
    a0 = b0 = rs
    phi0 = 0.0
    near = rho <= far_r * rs
    w = np.maximum(0.0, 1.0 - vals[near] / B_far)
    if float(np.sum(w)) > 0:
        # Custom (glue): the shadow's weighted second moments. Library: numpy.cov (aweights, bias) and eigh.
        C = np.cov(np.vstack([X[near], Y[near]]), aweights=w, bias=True)
        lam = np.linalg.eigh(C)[0]                     # ascending
        l2, l1 = float(lam[0]), float(lam[1])
        if l2 > 0:
            phi0 = 0.5 * math.atan2(2.0 * float(C[0, 1]), float(C[0, 0]) - float(C[1, 1]))
            a0 = rs * (l1 / l2) ** 0.25
            b0 = rs * (l2 / l1) ** 0.25
    return np.array([xs, ys, zs, a0, b0, rs, phi0, mu0, g0], dtype=float)


# --------------------------------------------------------------------- fit ---

def _bounds(scfg, xs, ys, zs, c_bounds_um=None):
    """(lo, hi) (10,): the boxes of SPEC Block 46 about the traced centre (x_s, y_s, z_s), in the order PARAMS then
    sigma0 (sigma0_bounds_um); c_bounds_um (lo, hi), when given, replaces axis_bounds_um for c only."""
    sxy, sz = float(scfg.shift_xy_um), float(scfg.shift_z_um)
    ax, mu, g, s0 = scfg.axis_bounds_um, scfg.mu_bounds_per_um, scfg.slope_bounds, scfg.sigma0_bounds_um
    lo = np.array([xs - sxy, ys - sxy, zs - sz, ax[0], ax[0], ax[0], -math.pi, mu[0], g[0], s0[0]], dtype=float)
    hi = np.array([xs + sxy, ys + sxy, zs + sz, ax[1], ax[1], ax[1], math.pi, mu[1], g[1], s0[1]], dtype=float)
    if c_bounds_um is not None:
        cb = np.asarray(c_bounds_um, dtype=float).reshape(-1)
        if cb.shape != (2,) or not np.all(np.isfinite(cb)) or not (0.0 < cb[0] < cb[1]):
            raise ValueError("c_bounds_um must be (lo, hi) with 0 < lo < hi finite (um), got %r" % (c_bounds_um,))
        lo[5], hi[5] = float(cb[0]), float(cb[1])
    if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)) and np.all(lo < hi)):
        raise ValueError("soma_fit bounds must be finite with lo < hi")
    return lo, hi


class _Problem(object):
    """The projected residual of the soma's planes and its Jacobian, with the forward evaluations counted.

    DR (K, n_R) the data on the region; region (H, W) bool; UR, VR (n_R,) um the region's pixel coordinates from the
    traced centre; shape (K, H, W) the forward's; forward(theta_g (8,), slope[, sigma0_um=]) -> tau (K, H, W);
    full0 (10,) the full vector (PARAMS, then sigma0) holding the held values; free (n_fit,) the indices into it of
    the fitted parameters; background "tilted" | "constant"; sigma_fit: pass sigma0_um to the forward; hi (n_fit,)
    the upper bounds; steps (n_fit,) the absolute difference steps; names (n_fit,) the fitted names. A memo of the
    last _MEMO_SIZE solver points (key: the parameters' bytes) holds (r, beta); the Jacobian's steps are evaluated
    without it (never requested again)."""

    def __init__(self, DR, region, UR, VR, shape, forward, full0, free, background, sigma_fit, hi, steps, names):
        self.DR = DR
        self.region = region
        self.UR = UR
        self.VR = VR
        self.shape = tuple(shape)
        self.forward = forward
        self.full0 = np.array(full0, dtype=float)
        self.free = np.asarray(free, dtype=np.int64)
        self.background = background
        self.sigma_fit = bool(sigma_fit)
        self.hi = np.asarray(hi, dtype=float)
        self.steps = np.asarray(steps, dtype=float)
        self.names = tuple(names)
        self.n_eval = 0
        self._memo = OrderedDict()

    def full(self, x):
        """(10,): the held values with x (n_fit,) at the free indices."""
        th = self.full0.copy()
        th[self.free] = x
        return th

    def theta_full(self, x):
        """(9,): theta in PARAMS order at x."""
        return self.full(x)[:9]

    def compute(self, x):
        """(r (K n_R,), beta (K, 3)) at x, without the memo."""
        th = self.full(x)
        self.n_eval += 1
        if self.sigma_fit:
            tau = self.forward(th[:8].copy(), float(th[8]), sigma0_um=float(th[9]))
        else:
            tau = self.forward(th[:8].copy(), float(th[8]))
        tau = np.asarray(tau, dtype=float)
        if tau.shape != self.shape:
            raise _ForwardShapeError("the forward model returned shape %s, expected %s" % (tau.shape, self.shape))
        beta, resid = _vp_project(self.DR, tau[:, self.region], self.UR, self.VR, self.background)
        return resid.reshape(-1), beta

    def evaluate(self, x):
        """(r, beta) at x (n_fit,), through the memo."""
        x = np.ascontiguousarray(x, dtype=np.float64)
        key = x.tobytes()
        hit = self._memo.get(key)
        if hit is not None:
            self._memo.move_to_end(key)
            return hit
        out = self.compute(x)
        self._memo[key] = out
        while len(self._memo) > _MEMO_SIZE:
            self._memo.popitem(last=False)
        return out

    def residual(self, x):
        """r (K n_R,) gl at x, a copy (the memo's array stays the Jacobian's base point)."""
        return self.evaluate(x)[0].copy()

    # Custom: forward differences with absolute steps, taken backwards where the forward step would cross an upper
    # bound or gives a non-finite residual (Block 42's rule). Checked: scipy.optimize.least_squares' "2-point" /
    # diff_step (relative steps: useless for x0, y0 or phi near 0, and a step of a fraction of a slab is wanted, the
    # forward being piecewise smooth in its slab grid) and scipy.optimize.approx_fprime (absolute steps, but neither
    # bounds nor a backward step).
    def jac(self, x):
        """(K n_R, n_fit): column j = [r(x + h_j e_j) - r(x)] / h_j, or [r(x) - r(x - h_j e_j)] / h_j when
        x_j + h_j > hi_j or r(x + h_j e_j) is not finite; _FitFailed when r(x) is not finite or a column is not
        finite either way. The step is the representable difference of the two points."""
        x = np.array(x, dtype=np.float64)
        r0 = self.evaluate(x)[0]
        if not np.all(np.isfinite(r0)):
            raise _FitFailed("the residual is not finite at %s" % (np.array2string(x, precision=6),))
        J = np.empty((r0.size, x.size), order="F")
        for j in range(x.size):
            h = float(self.steps[j])
            col = None
            if x[j] + h <= self.hi[j]:
                xp = x.copy()
                xp[j] = x[j] + h
                rp = self.compute(xp)[0]
                if np.all(np.isfinite(rp)):
                    col = (rp - r0) / (xp[j] - x[j])
            if col is None:
                xm = x.copy()
                xm[j] = x[j] - h
                rm = self.compute(xm)[0]
                if np.all(np.isfinite(rm)):
                    col = (r0 - rm) / (x[j] - xm[j])
            if col is None:
                raise _FitFailed("the Jacobian's column %s is not finite either way at %s"
                                 % (self.names[j], np.array2string(x, precision=6)))
            J[:, j] = col
        return J


def _covariance(J):
    """(cov (p, p) = (J^T J)^-1, how): numpy.linalg.inv when J is finite with full column rank
    (numpy.linalg.matrix_rank's tolerance), how "inv"; otherwise numpy.linalg.pinv(J^T J) with NaN rows and
    columns for the parameters that are not estimable -- a component above NULL_COMPONENT_TOL along J's null space
    (the SVD's right singular vectors of the singular values under matrix_rank's tolerance) --, how "pinv"; all NaN
    when J is not finite, how "not_finite"."""
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
    _u, s, vt = np.linalg.svd(J, full_matrices=False)
    tol = float(s.max()) * max(J.shape) * np.finfo(float).eps
    null = vt[s <= tol]
    unidentified = np.any(np.abs(null) > NULL_COMPONENT_TOL, axis=0) if null.size else np.zeros(p, dtype=bool)
    cov[unidentified, :] = np.nan
    cov[:, unidentified] = np.nan
    return cov, "pinv"


def _at_bound(x, lo, hi, names):
    w = hi - lo
    near = (np.abs(x - lo) <= AT_BOUND_REL * w) | (np.abs(hi - x) <= AT_BOUND_REL * w)
    return tuple(names[j] for j in range(len(names)) if near[j])


def _failed(fitted, ks_sel, dof, n_samples, n_eval, t0, message, background, sigma0_out):
    nan = float("nan")
    n_fit = len(fitted)
    return SomaFit(theta=np.full(9, np.nan), se=np.full(9, np.nan), cov=np.full((9, 9), np.nan),
                   fitted=tuple(fitted), B_gl=np.full(ks_sel.size, np.nan), ks=ks_sel, chi2=nan, dof=int(dof),
                   n_samples=int(n_samples), status="failed", at_bound=(), n_eval=int(n_eval),
                   seconds=time.perf_counter() - t0, area_um2=nan, se_area_um2=nan, volume_um3=nan, r_eq_um=nan,
                   message=str(message), beta=np.full((ks_sel.size, 3), np.nan), background=str(background),
                   sigma0_um=float(sigma0_out), se_sigma0_um=nan, cov_fit=np.full((n_fit, n_fit), np.nan))


def _hold_values(hold, lo, hi):
    """{index into PARAMS: value} from the mapping hold (names among PARAMS[:8]; values finite, inside the bounds in
    force, closed). ValueError for an unknown name (sigma0_um and the far slope included), a value not finite or
    outside its bounds."""
    if hold is None:
        return {}
    try:
        items = list(hold.items())
    except AttributeError:
        raise ValueError("hold must be a mapping from parameter names to values, got %r" % (hold,))
    out = {}
    for name, value in items:
        if name not in PARAMS[:8]:
            raise ValueError("hold: %r is not one of the holdable parameters %s" % (name, PARAMS[:8]))
        j = PARAMS.index(name)
        v = _finite(value, "hold[%r]" % name)
        if not lo[j] <= v <= hi[j]:
            raise ValueError("hold[%r] = %r lies outside its bounds [%r, %r]" % (name, v, lo[j], hi[j]))
        out[j] = v
    return out


def fit_soma(block, ks, valid, left, top, res0_um, dz_um, centre_um, radius_um, mask, forward_factory, scfg,
             slope0, hold=None, start=None, c_bounds_um=None, sigma0_um=None):
    """-> SomaFit: the ellipsoid of SPEC Block 46 (and its wave-8 amendment) fitted to the planes select_planes
    chooses (scfg.plane_half_um, plane_step), over the region R: the pixels whose centre lies within r_R of the
    traced centre, r_R = scfg.region_radius_um when it is a number, otherwise scfg.region_radius_scale x radius_um
    (D-255), minus mask (True: left out). Each plane's background is solved at every theta: scfg.background
    "tilted" (b0 + b1 u + b2 v, u, v um from the traced centre; D-254.1) or "constant" (one scale).

    block (n, H, W) grey levels (any real dtype; not modified), ks (n,) whole plane numbers ascending, valid (n,)
    bool; (left, top) px the crop's corner; res0_um, dz_um um; centre_um (3,) um, radius_um um: the traced soma
    (image frame; depth in stage units, plane k at depth k dz); mask (H, W) bool; forward_factory(z_planes_um (K,))
    -> forward: forward(theta_g (8,), slope) -> tau (K, H, W) dimensionless, slope a float (the fitted gamma, or
    slope0 when held), called with these two positional arguments only under scfg.sigma0 "fixed", and as
    forward(theta_g, slope, sigma0_um=value) under "fit"; scfg the SomaFitConfig (far_slope, background, sigma0,
    the region, the bounds, diff_steps, sigma0_diff_step_um, max_nfev, ftol, xtol); slope0 the kernel's configured
    slope (um/um, finite, >= 0): the start of gamma (clipped into slope_bounds) or its held value.
    hold: None or a mapping from names among PARAMS[:8] to values (finite, inside the bounds in force): held
    exactly, not fitted. start: None (start_theta on the valid plane nearest the traced depth) or a (9,) theta whose
    first eight entries are finite (the slope entry ignored). c_bounds_um: None or (lo, hi), 0 < lo < hi, replacing
    axis_bounds_um for c. sigma0_um: the kernel's in-focus width (um): the start of the fitted sigma0 under "fit"
    (required finite > 0, clipped into sigma0_bounds_um), recorded as passed under "fixed" (None or finite > 0).
    The start is clipped into the bounds (the clipped names in the message), then the held entries set.
    scipy.optimize.least_squares (trf, x_scale "jac"; jac _Problem.jac) over the fitted parameters. Never raises
    on a solver failure or a non-finite forward: status "failed" with NaN results (a non-finite residual at the
    start, a Jacobian column not finite either way, the solver's linear algebra). ValueError for inputs that
    disagree (shapes, dtypes, a forward of the wrong shape, a forward that refuses the slope -- Block 45 under a
    continuation other than "linear", D-220.2 --), an empty region, fewer than 3 planes, a far_slope, background or
    sigma0 setting not among its names, data not finite on the region of the planes fitted, a centre, radius, pixel
    size, spacing or slope0 that is not finite (radius, sizes > 0, slope0 >= 0), a bad hold, start, c_bounds_um or
    sigma0_um, and no parameter left to fit."""
    t0 = time.perf_counter()
    far = scfg.far_slope
    if far not in FAR_SLOPES:
        raise ValueError("scfg.far_slope must be one of %s, got %r" % (FAR_SLOPES, far))
    bg = scfg.background
    if bg not in BACKGROUNDS:
        raise ValueError("scfg.background must be one of %s, got %r" % (BACKGROUNDS, bg))
    s0_mode = scfg.sigma0
    if s0_mode not in SIGMA0_MODES:
        raise ValueError("scfg.sigma0 must be one of %s, got %r" % (SIGMA0_MODES, s0_mode))
    sigma_fit = s0_mode == "fit"
    data = np.asarray(block)
    if data.ndim != 3 or data.size == 0:
        raise ValueError("block must be a non-empty (n, H, W) stack, got shape %s" % (data.shape,))
    n, H, W = data.shape
    k = _plane_numbers(ks)
    if k.shape != (n,):
        raise ValueError("ks must have one plane number per plane of block, (%d,), got %s" % (n, k.shape))
    v = np.asarray(valid, dtype=bool).reshape(-1)
    if v.shape != (n,):
        raise ValueError("valid must be (%d,), got %s" % (n, v.shape))
    msk = _bool_mask(mask, (H, W), "mask")
    left_i, top_i = _whole(left, "left"), _whole(top, "top")
    p = _positive(res0_um, "res0_um")
    dz = _positive(dz_um, "dz_um")
    xs, ys, zs = _centre(centre_um)
    rs = _positive(radius_um, "radius_um")
    g0 = _finite(slope0, "slope0")
    if g0 < 0:
        raise ValueError("slope0 must be >= 0, got %r" % (slope0,))
    if sigma_fit:
        if sigma0_um is None:
            raise ValueError("scfg.sigma0 'fit' needs sigma0_um, the kernel's in-focus width (um), as its start")
        s0_arg = _positive(sigma0_um, "sigma0_um")
    else:
        s0_arg = float("nan") if sigma0_um is None else _positive(sigma0_um, "sigma0_um")
    if not callable(forward_factory):
        raise ValueError("forward_factory must be callable")
    steps9 = np.asarray(scfg.diff_steps, dtype=float).reshape(-1)
    if steps9.shape != (9,) or not np.all(np.isfinite(steps9)) or np.any(steps9 <= 0):
        raise ValueError("scfg.diff_steps must be 9 finite steps > 0")
    s0_step = _positive(scfg.sigma0_diff_step_um, "scfg.sigma0_diff_step_um")
    steps = np.r_[steps9, s0_step]
    lo, hi = _bounds(scfg, xs, ys, zs, c_bounds_um)
    held = _hold_values(hold, lo, hi)
    free = [j for j in range(8) if j not in held]
    if far == "fit":
        free.append(8)
    if sigma_fit:
        free.append(9)
    free = np.asarray(free, dtype=np.int64)
    names = tuple(PARAMS[j] if j < 9 else SIGMA0_NAME for j in free)
    n_fit = int(free.size)
    if n_fit == 0:
        raise ValueError("every parameter is held: nothing to fit")
    n_bg = 3 if bg == "tilted" else 1

    idx = select_planes(k, v, zs, dz, scfg.plane_half_um, scfg.plane_step)
    x_px, y_px = _pixel_centres((H, W), left_i, top_i, p)
    if scfg.region_radius_um is None:
        r_reg = _positive(scfg.region_radius_scale, "scfg.region_radius_scale") * rs
    else:
        r_reg = _positive(scfg.region_radius_um, "scfg.region_radius_um")
    disk = (x_px[None, :] - xs) ** 2 + (y_px[:, None] - ys) ** 2 <= r_reg ** 2
    region = disk & ~msk
    if not region.any():
        raise ValueError("the region (the %g um disk about the traced centre minus the mask) holds no pixel of the "
                         "crop" % r_reg)
    DR = np.asarray(data[idx], dtype=float)[:, region]
    if not np.all(np.isfinite(DR)):
        raise ValueError("block must be finite on the region of the planes fitted")
    UR = np.broadcast_to((x_px - xs)[None, :], (H, W))[region]
    VR = np.broadcast_to((y_px - ys)[:, None], (H, W))[region]
    ks_sel = k[idx].copy()
    K, nR = int(idx.size), int(np.count_nonzero(region))
    n_samples = K * nR
    dof = n_samples - n_fit - n_bg * K

    if start is None:
        i0 = _nearest_valid(k, v, zs, dz)
        th0 = start_theta(data[i0], left_i, top_i, p, (xs, ys, zs), rs, region, g0, scfg)
    else:
        st = np.asarray(start, dtype=float).reshape(-1)
        if st.shape != (9,) or not np.all(np.isfinite(st[:8])):
            raise ValueError("start must be a (9,) theta whose first eight entries are finite, got %r" % (start,))
        th0 = st.copy()
    full_start = np.r_[th0[:8], g0, s0_arg if sigma_fit else 0.0]
    full0 = full_start.copy()
    clip_ix = list(free)
    full0[clip_ix] = np.clip(full_start[clip_ix], lo[clip_ix], hi[clip_ix])
    clipped = tuple(PARAMS[j] if j < 9 else SIGMA0_NAME for j in clip_ix if full0[j] != full_start[j])
    note = ("; start clipped into the bounds: %s" % ", ".join(clipped)) if clipped else ""
    for j, val in held.items():
        full0[j] = val
    if not sigma_fit:
        full0[9] = s0_arg
    x0 = full0[free].copy()
    lo_f, hi_f = lo[free], hi[free]
    s0_fail = float("nan") if sigma_fit else s0_arg

    def failed(prob_n_eval, message):
        return _failed(names, ks_sel, dof, n_samples, prob_n_eval, t0, message, bg, s0_fail)

    forward = forward_factory(ks_sel.astype(float) * dz)
    prob = _Problem(DR, region, UR, VR, (K, H, W), forward, full0, free, bg, sigma_fit, hi_f, steps[free], names)
    r_start = prob.evaluate(x0)[0]                       # the shape checked; served to the solver by the memo
    if not np.all(np.isfinite(r_start)):
        return failed(prob.n_eval, "the residual is not finite at the start %s%s"
                      % (np.array2string(x0, precision=6), note))
    try:
        res = least_squares(prob.residual, x0, jac=prob.jac, bounds=(lo_f, hi_f), method="trf", x_scale="jac",
                            ftol=float(scfg.ftol), xtol=float(scfg.xtol), max_nfev=int(scfg.max_nfev))
    except _FitFailed as exc:
        return failed(prob.n_eval, str(exc) + note)
    except _ForwardShapeError:
        raise
    except ValueError as exc:
        if "not finite in the initial point" not in str(exc):     # scipy's message for a non-finite start
            raise
        return failed(prob.n_eval, str(exc) + note)
    except np.linalg.LinAlgError as exc:
        return failed(prob.n_eval, "the solver's linear algebra: %s%s" % (exc, note))

    x = np.array(res.x, dtype=float)
    r, beta = prob.evaluate(x)
    if not np.all(np.isfinite(r)):                       # trf never accepts a non-finite point; kept as a guard
        return failed(prob.n_eval, "the residual at the solution is not finite" + note)
    chi2 = float(np.dot(r, r))
    cov_u, how = _covariance(np.asarray(res.jac, dtype=float))
    s2 = chi2 / dof if dof > 0 else float("nan")
    cov_fit = np.asarray(cov_u * s2, dtype=float)
    in9 = [i for i, j in enumerate(free) if j < 9]       # positions in cov_fit of the fitted PARAMS
    j9 = free[in9]
    cov = np.full((9, 9), np.nan)
    cov[np.ix_(j9, j9)] = cov_fit[np.ix_(in9, in9)]
    with np.errstate(invalid="ignore"):
        se = np.sqrt(np.diag(cov))
        se_s0 = float(np.sqrt(cov_fit[-1, -1])) if sigma_fit else float("nan")
    full = prob.full(x)
    theta = full[:9].copy()
    s0_out = float(full[9]) if sigma_fit else s0_arg
    a, b, c = (float(t) for t in theta[3:6])
    area = ellipsoid_area(a, b, c)
    g = _area_gradient(a, b, c)
    ax_fit = [i for i in range(3) if (3 + i) in set(j9.tolist())]   # the fitted axes among (a, b, c)
    if not ax_fit:
        se_area = 0.0                                    # every axis held: no variance (conditional on them)
    else:
        ia = np.asarray(ax_fit)
        C = cov[np.ix_(3 + ia, 3 + ia)]
        gs = g[ia]
        se_area = math.sqrt(max(0.0, float(gs @ C @ gs))) if np.all(np.isfinite(C)) else float("nan")
    status = "converged" if res.status > 0 else "max_nfev"
    msg = "%s (scipy status %d, nfev %d, njev %d, covariance %s)%s" % (res.message, res.status, res.nfev, res.njev,
                                                                       how, note)
    beta = np.asarray(beta, dtype=float)
    return SomaFit(theta=theta, se=se, cov=cov, fitted=names, B_gl=beta[:, 0].copy(), ks=ks_sel,
                   chi2=chi2, dof=int(dof), n_samples=int(n_samples), status=status,
                   at_bound=_at_bound(x, lo_f, hi_f, names), n_eval=int(prob.n_eval),
                   seconds=time.perf_counter() - t0, area_um2=float(area), se_area_um2=float(se_area),
                   volume_um3=4.0 / 3.0 * math.pi * a * b * c, r_eq_um=math.sqrt(area / (4.0 * math.pi)),
                   message=msg, beta=beta, background=str(bg), sigma0_um=float(s0_out), se_sigma0_um=se_s0,
                   cov_fit=cov_fit)
