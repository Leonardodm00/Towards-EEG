"""The soma fit -- Block 46 in specs/SPEC.md (D-215, D-216, D-218, D-219).

An ellipsoid fitted to the planes about the soma. Data: the soma's crop D (n, H, W) grey levels (gl) on the planes
ks (n,) with valid (n,), the crop's corner (left, top) px, the pixel size p and the plane spacing dz (um); the traced
soma's centre (x_s, y_s, z_s) um (image frame; depth in stage units, plane k at depth k dz) and radius r_s um; a mask
(H, W) bool, True where a pixel is left out (Allen's traced stems and the neuropil, widened by the caller).

    planes K   select_planes: the valid planes with |k dz - z_s| <= plane_half_um and k - k_0 divisible by
               plane_step, k_0 the valid plane nearest z_s (ties: the lower k);
    region R   the pixels whose centre ((left + col) p, (top + row) p) lies within region_radius_um of (x_s, y_s)
               (Block 44's disk_mask convention) and whose mask is False;
    theta      (x0, y0, z0, a, b, c, phi, mu, gamma): Block 45's theta_g and the kernel's far slope gamma, fitted
               (far_slope "fit", D-218.1) or held at slope0 ("fixed", D-218.2, the fit then over the first eight);
    model      D_k ~ B_k tau_k(theta) on R, tau the forward model passed in (Block 45 in production); for each theta
               B_k = sum_R D_k tau_k / sum_R tau_k^2 and r_k = D_k - B_k tau_k on R (variable projection, D-219.3).

theta_hat = argmin 1/2 sum_K sum_R r^2 within the bounds -- |x0 - x_s|, |y0 - y_s| <= shift_xy_um,
|z0 - z_s| <= shift_z_um, a, b, c in axis_bounds_um, phi in [-pi, pi], mu in mu_bounds_per_um, gamma in
slope_bounds -- by scipy.optimize.least_squares (method "trf", x_scale "jac", max_nfev, ftol, xtol of the
configuration) with jac the callable _Problem.jac: forward differences with the absolute steps diff_steps, taken
backwards where the forward step would cross an upper bound or gives a non-finite residual (as Block 42).
Cov = (J^T J)^-1 chi2 / dof, J the solver's final Jacobian of the projected residual at theta_hat (Golub-Pereyra),
chi2 = sum r^2 (gl^2), dof = |K| |R| - (fitted parameters) - |K|; numpy.linalg.pinv when J^T J is singular, with NaN
for the parameters along its null space (Block 42's rule); SE = sqrt(diag(Cov)). A parameter is at a bound when
within 1e-6 of the bound interval's width from it.

The start (start_theta, on the plane k_0; D-219.5): the traced centre and size, with the shadow's orientation (its
weighted second moments) and an aspect of the same mean radius, mu_0 from the darkest core; clipped into the bounds
by fit_soma. Derived: the membrane area S (ellipsoid_area, Legendre's form), the volume 4/3 pi a b c, the
equivalent radius sqrt(S / 4 pi) (D-216.3) and SE(S) by the delta method on the (a, b, c) block of Cov, the
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
    """The soma fit's result (SPEC Block 46). K the planes fitted.

    theta (9,) in the order and units of PARAMS (um, um, um, um, um, um, rad, 1/um, um/um; theta[8] = slope0 exactly
    when gamma is held); se (9,) = sqrt(diag(cov)), NaN for gamma when held; cov (9, 9) = (J^T J)^-1 chi2 / dof (NaN
    in gamma's row and column when held, and for parameters along J's null space); fitted: the names fitted, in
    PARAMS order; B_gl (K,) the planes' scales (gl); ks (K,) the plane numbers fitted; chi2 = sum r^2 (gl^2);
    dof = n_samples - len(fitted) - K; n_samples = K |R|; status one of STATUSES; at_bound: the fitted names at a
    bound; n_eval: the forward evaluations (the Jacobian's included); seconds (wall); area_um2, se_area_um2,
    volume_um3, r_eq_um: S(a, b, c), its SE (delta method), 4/3 pi a b c, sqrt(S / 4 pi); message. A "failed" fit
    has NaN in theta, se, cov, B_gl, chi2 and the derived quantities, and at_bound empty."""
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


def vp_background(D, tau, region, u_um, v_um):
    """-> (beta (K, 3) [gl, gl/um, gl/um], resid (K, n_region) gl): per plane the least-squares coefficients of the
    columns [tau, u tau, v tau] on the region's pixels and the residual D - M beta there (C order, as vp_scale). D, tau
    (K, H, W); region (H, W) bool; u_um, v_um (H, W) float, um from the traced centre. NaN coefficients and residual
    for a plane whose columns are not finite or have rank < 3. SPEC Block 46, the amendment of wave 8 (D-254.1)."""
    raise NotImplementedError("Block 46, wave 8: vp_background")


def background_planes(beta, shape, left, top, res0_um, centre_xy_um):
    """-> (K, H, W) gl: b0_k + b1_k u + b2_k v at the pixel centres of a crop, u = x - x_s, v = y - y_s (um), pixel
    (row, col) centred on ((left + col) p, (top + row) p). beta (K, 3); shape (H, W); centre_xy_um (x_s, y_s). SPEC
    Block 46, the amendment of wave 8."""
    raise NotImplementedError("Block 46, wave 8: background_planes")


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

def _bounds(scfg, xs, ys, zs):
    """(lo, hi) (9,): the boxes of SPEC Block 46 about the traced centre (x_s, y_s, z_s)."""
    sxy, sz = float(scfg.shift_xy_um), float(scfg.shift_z_um)
    ax, mu, g = scfg.axis_bounds_um, scfg.mu_bounds_per_um, scfg.slope_bounds
    lo = np.array([xs - sxy, ys - sxy, zs - sz, ax[0], ax[0], ax[0], -math.pi, mu[0], g[0]], dtype=float)
    hi = np.array([xs + sxy, ys + sxy, zs + sz, ax[1], ax[1], ax[1], math.pi, mu[1], g[1]], dtype=float)
    if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)) and np.all(lo < hi)):
        raise ValueError("soma_fit bounds must be finite with lo < hi")
    return lo, hi


class _Problem(object):
    """The projected residual of the soma's planes and its Jacobian, with the forward evaluations counted.

    DR (K, n_R) the data on the region; region (H, W) bool; shape (K, H, W) the forward's; forward(theta_g (8,),
    slope) -> tau (K, H, W); slope0 the held gamma (used when n_par = 8); hi (n_par,) the upper bounds; steps (n_par,)
    the absolute difference steps. A memo of the last _MEMO_SIZE solver points (key: the parameters' bytes) holds
    (r, B); the Jacobian's steps are evaluated without it (never requested again)."""

    def __init__(self, DR, region, shape, forward, slope0, n_par, hi, steps):
        self.DR = DR
        self.region = region
        self.shape = tuple(shape)
        self.forward = forward
        self.slope0 = float(slope0)
        self.n_par = int(n_par)
        self.hi = np.asarray(hi, dtype=float)
        self.steps = np.asarray(steps, dtype=float)
        self.n_eval = 0
        self._memo = OrderedDict()

    def theta_full(self, x):
        """(9,): x (n_par,) with gamma = slope0 appended when it is held."""
        th = np.empty(9)
        th[:self.n_par] = x
        if self.n_par == 8:
            th[8] = self.slope0
        return th

    def compute(self, x):
        """(r (K n_R,), B (K,)) at x, without the memo."""
        th = self.theta_full(x)
        self.n_eval += 1
        tau = np.asarray(self.forward(th[:8].copy(), float(th[8])), dtype=float)
        if tau.shape != self.shape:
            raise _ForwardShapeError("the forward model returned shape %s, expected %s" % (tau.shape, self.shape))
        B, resid = _vp_columns(self.DR, tau[:, self.region])
        return resid.reshape(-1), B

    def evaluate(self, x):
        """(r, B) at x (n_par,), through the memo."""
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
        """(K n_R, n_par): column j = [r(x + h_j e_j) - r(x)] / h_j, or [r(x) - r(x - h_j e_j)] / h_j when
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
                                 % (PARAMS[j], np.array2string(x, precision=6)))
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


def _failed(fitted, ks_sel, dof, n_samples, n_eval, t0, message):
    nan = float("nan")
    return SomaFit(theta=np.full(9, np.nan), se=np.full(9, np.nan), cov=np.full((9, 9), np.nan),
                   fitted=tuple(fitted), B_gl=np.full(ks_sel.size, np.nan), ks=ks_sel, chi2=nan, dof=int(dof),
                   n_samples=int(n_samples), status="failed", at_bound=(), n_eval=int(n_eval),
                   seconds=time.perf_counter() - t0, area_um2=nan, se_area_um2=nan, volume_um3=nan, r_eq_um=nan,
                   message=str(message))


def fit_soma(block, ks, valid, left, top, res0_um, dz_um, centre_um, radius_um, mask, forward_factory, scfg,
             slope0, hold=None, start=None, c_bounds_um=None, sigma0_um=None):
    """-> SomaFit: the ellipsoid of SPEC Block 46 fitted to the planes select_planes chooses (scfg.plane_half_um,
    plane_step), over the region R (the pixels whose centre lies within scfg.region_radius_um of the traced centre --
    radius_um, the traced radius, when it is None (D-241) -- minus mask, True: left out).

    block (n, H, W) grey levels (any real dtype; not modified), ks (n,) whole plane numbers ascending, valid (n,)
    bool; (left, top) px the crop's corner; res0_um, dz_um um; centre_um (3,) um, radius_um um: the traced soma
    (image frame; depth in stage units, plane k at depth k dz); mask (H, W) bool; forward_factory(z_planes_um (K,))
    -> forward, forward(theta_g (8,), slope) -> tau (K, H, W) dimensionless, slope a float (the fitted gamma, or
    slope0 when held); scfg the SomaFitConfig (far_slope "fit" | "fixed", the bounds, diff_steps, max_nfev, ftol,
    xtol); slope0 the kernel's configured slope (um/um, finite, >= 0): the start of gamma (clipped into
    slope_bounds) or its held value. The start is start_theta on the valid plane nearest the traced depth, clipped
    into the bounds (the clipped names in the message). scipy.optimize.least_squares (trf, x_scale "jac"; jac
    _Problem.jac). Never raises on a solver failure or a non-finite forward: status "failed" with NaN results (a
    non-finite residual at the start, a Jacobian column not finite either way, the solver's linear algebra).
    ValueError for inputs that disagree (shapes, dtypes, a forward of the wrong shape, a forward that refuses the
    slope -- Block 45 under a continuation other than "linear", D-220.2 --), an empty region, fewer than 3 planes, a far_slope other than "fit" or "fixed", data not finite on the region of the planes fitted, a
    centre, radius, pixel size, spacing or slope0 that is not finite (radius, sizes > 0, slope0 >= 0)."""
    # CONTRACT STUB (wave 8; SPEC Block 46, the amendment of 2026-11-25): hold (name -> value, the first eight of
    # PARAMS), start ((9,) theta), c_bounds_um ((lo, hi)), sigma0_um (um); scfg.background, scfg.sigma0,
    # scfg.region_radius_scale; SomaFit's beta, background, sigma0_um, se_sigma0_um, cov_fit. Not implemented yet.
    if hold is not None or start is not None or c_bounds_um is not None or sigma0_um is not None:
        raise NotImplementedError("Block 46, wave 8: hold, start, c_bounds_um, sigma0_um")
    t0 = time.perf_counter()
    far = scfg.far_slope
    if far not in FAR_SLOPES:
        raise ValueError("scfg.far_slope must be one of %s, got %r" % (FAR_SLOPES, far))
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
    if not callable(forward_factory):
        raise ValueError("forward_factory must be callable")
    n_par = 9 if far == "fit" else 8
    fitted = PARAMS[:n_par]
    steps = np.asarray(scfg.diff_steps, dtype=float).reshape(-1)
    if steps.shape != (9,) or not np.all(np.isfinite(steps)) or np.any(steps <= 0):
        raise ValueError("scfg.diff_steps must be 9 finite steps > 0")
    lo, hi = _bounds(scfg, xs, ys, zs)
    lo, hi = lo[:n_par], hi[:n_par]

    idx = select_planes(k, v, zs, dz, scfg.plane_half_um, scfg.plane_step)
    x_px, y_px = _pixel_centres((H, W), left_i, top_i, p)
    r_reg = rs if scfg.region_radius_um is None else _positive(scfg.region_radius_um, "scfg.region_radius_um")
    disk = (x_px[None, :] - xs) ** 2 + (y_px[:, None] - ys) ** 2 <= r_reg ** 2
    region = disk & ~msk
    if not region.any():
        raise ValueError("the region (the %g um disk about the traced centre minus the mask) holds no pixel of the "
                         "crop" % r_reg)
    DR = np.asarray(data[idx], dtype=float)[:, region]
    if not np.all(np.isfinite(DR)):
        raise ValueError("block must be finite on the region of the planes fitted")
    ks_sel = k[idx].copy()
    K, nR = int(idx.size), int(np.count_nonzero(region))
    n_samples = K * nR
    dof = n_samples - n_par - K

    i0 = _nearest_valid(k, v, zs, dz)
    th0 = start_theta(data[i0], left_i, top_i, p, (xs, ys, zs), rs, region, g0, scfg)
    x0 = np.clip(th0[:n_par], lo, hi)
    clipped = tuple(fitted[j] for j in range(n_par) if x0[j] != th0[j])
    note = ("; start clipped into the bounds: %s" % ", ".join(clipped)) if clipped else ""

    forward = forward_factory(ks_sel.astype(float) * dz)
    prob = _Problem(DR, region, (K, H, W), forward, g0, n_par, hi, steps[:n_par])
    r_start = prob.evaluate(x0)[0]                       # the shape checked; served to the solver by the memo
    if not np.all(np.isfinite(r_start)):
        return _failed(fitted, ks_sel, dof, n_samples, prob.n_eval, t0,
                       "the residual is not finite at the start %s%s" % (np.array2string(x0, precision=6), note))
    try:
        res = least_squares(prob.residual, x0, jac=prob.jac, bounds=(lo, hi), method="trf", x_scale="jac",
                            ftol=float(scfg.ftol), xtol=float(scfg.xtol), max_nfev=int(scfg.max_nfev))
    except _FitFailed as exc:
        return _failed(fitted, ks_sel, dof, n_samples, prob.n_eval, t0, str(exc) + note)
    except _ForwardShapeError:
        raise
    except ValueError as exc:
        if "not finite in the initial point" not in str(exc):     # scipy's message for a non-finite start
            raise
        return _failed(fitted, ks_sel, dof, n_samples, prob.n_eval, t0, str(exc) + note)
    except np.linalg.LinAlgError as exc:
        return _failed(fitted, ks_sel, dof, n_samples, prob.n_eval, t0, "the solver's linear algebra: %s%s"
                       % (exc, note))

    x = np.array(res.x, dtype=float)
    r, B = prob.evaluate(x)
    if not np.all(np.isfinite(r)):                       # trf never accepts a non-finite point; kept as a guard
        return _failed(fitted, ks_sel, dof, n_samples, prob.n_eval, t0,
                       "the residual at the solution is not finite" + note)
    chi2 = float(np.dot(r, r))
    cov_u, how = _covariance(np.asarray(res.jac, dtype=float))
    s2 = chi2 / dof if dof > 0 else float("nan")
    cov = np.full((9, 9), np.nan)
    cov[:n_par, :n_par] = cov_u * s2
    with np.errstate(invalid="ignore"):
        se = np.sqrt(np.diag(cov))
    theta = prob.theta_full(x)
    a, b, c = (float(t) for t in theta[3:6])
    area = ellipsoid_area(a, b, c)
    g = _area_gradient(a, b, c)
    C = cov[3:6, 3:6]
    se_area = math.sqrt(max(0.0, float(g @ C @ g))) if np.all(np.isfinite(C)) else float("nan")
    status = "converged" if res.status > 0 else "max_nfev"
    msg = "%s (scipy status %d, nfev %d, njev %d, covariance %s)%s" % (res.message, res.status, res.nfev, res.njev,
                                                                       how, note)
    return SomaFit(theta=theta, se=se, cov=cov, fitted=tuple(fitted), B_gl=np.asarray(B, dtype=float), ks=ks_sel,
                   chi2=chi2, dof=int(dof), n_samples=int(n_samples), status=status,
                   at_bound=_at_bound(x, lo, hi, fitted), n_eval=int(prob.n_eval),
                   seconds=time.perf_counter() - t0, area_um2=float(area), se_area_um2=float(se_area),
                   volume_um3=4.0 / 3.0 * math.pi * a * b * c, r_eq_um=math.sqrt(area / (4.0 * math.pi)),
                   message=msg)
