"""The soma fit -- Block 46 in specs/SPEC.md (D-215, D-216, D-218, D-219). CONTRACT STUB (wave 5).

An ellipsoid fitted to the planes about the soma: D_k(x, y) ~ B_k tau_k(x, y; theta) over a disk about the traced
soma, Allen's traced stems and the neuropil masked; theta = (x0, y0, z0, a, b, c, phi, mu, gamma), gamma the kernel's
far slope, fitted ("fit", the default, D-218.1) or held ("fixed", D-218.2); each plane's B_k solved exactly (variable
projection); scipy.optimize.least_squares (trf). Derived: the membrane area S (Legendre's form), the volume, the
equivalent radius sqrt(S / 4 pi) (D-216.3) and SE(S). The forward model is passed in (Block 45's SomaRenderer in
production). See SPEC Block 46 for the full statement, the parameters and the oracles.

No file access, no plotting. Pure ASCII, LF, for the cluster.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

PARAMS = ("x0_um", "y0_um", "z0_um", "a_um", "b_um", "c_um", "phi_rad", "mu_per_um", "slope_um_per_um")


@dataclass(frozen=True)
class SomaFit:
    """The soma fit's result (SPEC Block 46). theta, se (9,) in the order of PARAMS (se NaN for gamma when it is
    held); cov (9, 9); fitted: the names fitted; B_gl (K,) the planes' scales (gl); ks (K,) the planes fitted;
    chi2 (gl^2), dof, n_samples; status "converged" | "max_nfev" | "failed"; at_bound: names at a bound; n_eval;
    seconds; area_um2, se_area_um2, volume_um3, r_eq_um; message."""
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


def ellipsoid_area(a_um, b_um, c_um):
    """-> the surface area (um^2) of the ellipsoid with semi-axes a, b, c (um): Legendre's form with
    scipy.special.ellipkinc / ellipeinc (axes sorted), the spheroid and sphere limits in closed form; symmetric in
    its arguments. ValueError unless all are finite and > 0."""
    raise NotImplementedError("Block 46 (wave 5)")


def select_planes(ks, valid, z_start_um, dz_um, half_um, step):
    """-> the indices i into ks (ascending) with valid[i], |ks[i] dz - z_start| <= half_um and ks[i] - k_0 divisible
    by step, k_0 the k of the valid plane nearest z_start / dz (ties: the lower k). ks (n,) int ascending, valid (n,)
    bool; z_start_um, dz_um, half_um um. ValueError when fewer than 3 are selected, step < 1 or half_um < 0."""
    raise NotImplementedError("Block 46 (wave 5)")


def vp_scale(D, tau, region):
    """-> (B (K,) gl, resid (K, n_region) gl): per plane the least-squares scale B_k = sum_R D tau / sum_R tau^2
    over the region's pixels and the residual D - B_k tau there. D, tau (K, H, W); region (H, W) bool."""
    raise NotImplementedError("Block 46 (wave 5)")


def start_theta(block_k0, left, top, res0_um, centre_um, radius_um, region, slope0):
    """-> theta0 (9,) = (x_s, y_s, z_s, a_0, b_0, r_s, phi_0, mu_0, slope0) (D-219.5), from the plane block_k0 (H, W)
    gl, the crop's corner (left, top) px, res0_um, the traced centre centre_um (3,) um (image frame; depth in stage
    units) and radius radius_um um, the region (H, W) bool. rho: a pixel centre's distance from (x_s, y_s);
    B_far = the median of block_k0 over the region with rho > 1.5 r_s (the whole region if none);
    mu_0 = -ln(varrho) / (2 r_s), varrho = clip(the 5th percentile of block_k0 over the region with rho <= 0.5 r_s,
    divided by B_far, 0.05, 0.95), 0.5 if none; weights w = max(0, 1 - block_k0 / B_far) on the region with
    rho <= 1.5 r_s, C their weighted covariance of (x, y) about the weighted mean, lambda_1 >= lambda_2 its
    eigenvalues: phi_0 = atan2(2 C_xy, C_xx - C_yy) / 2, a_0 = r_s (lambda_1 / lambda_2)^(1/4),
    b_0 = r_s (lambda_2 / lambda_1)^(1/4) (a_0 = b_0 = r_s, phi_0 = 0 when sum w = 0 or lambda_2 <= 0). Not clipped
    here: fit_soma clips the start into the bounds."""
    raise NotImplementedError("Block 46 (wave 5)")


def fit_soma(block, ks, valid, left, top, res0_um, dz_um, centre_um, radius_um, mask, forward_factory, scfg,
             slope0):
    """-> SomaFit: the ellipsoid fitted to the planes select_planes chooses, over the disk of region_radius_um
    about the traced centre minus mask (H, W) bool (True: left out). block (n, H, W) grey levels, ks (n,) int,
    valid (n,) bool; (left, top) px the crop's corner; res0_um, dz_um um; centre_um (3,), radius_um the traced soma;
    forward_factory(z_planes_um (K,)) -> forward, forward(theta_g (8,), slope) -> tau (K, H, W); scfg the
    SomaFitConfig (far_slope "fit" | "fixed", bounds, steps, tolerances); slope0 the kernel's configured slope (the
    start of gamma, or its held value). Never raises on a solver failure or a NaN forward (status "failed", NaN
    results); ValueError for inputs that disagree, an empty region, fewer than 3 planes."""
    raise NotImplementedError("Block 46 (wave 5)")
