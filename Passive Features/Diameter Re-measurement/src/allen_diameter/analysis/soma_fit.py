"""The soma fit -- Block 46 in specs/SPEC.md (D-215, D-216, D-218). CONTRACT STUB (wave 5).

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
    """-> indices into ks (ascending) of the recorded planes k with |k dz - z_start| <= half_um, every step-th one
    counted from the recorded plane nearest z_start (that plane included). ks (n,) int ascending, valid (n,) bool.
    ValueError when fewer than 3 are selected, step < 1 or half_um < 0."""
    raise NotImplementedError("Block 46 (wave 5)")


def vp_scale(D, tau, region):
    """-> (B (K,) gl, resid (K, n_region) gl): per plane the least-squares scale B_k = sum_R D tau / sum_R tau^2
    over the region's pixels and the residual D - B_k tau there. D, tau (K, H, W); region (H, W) bool."""
    raise NotImplementedError("Block 46 (wave 5)")


def start_theta(block_k0, left, top, res0_um, centre_um, radius_um, region, slope0):
    """-> theta0 (9,): (x_s, y_s, z_s, r_s, r_s, r_s, 0, mu_0, slope0), mu_0 = -ln(rho) / (2 r_s), rho = clip(the
    5th percentile of the plane block_k0 (H, W) gl within 0.5 r_s of (x_s, y_s) / its median over the region's pixels
    farther than 1.5 r_s, 0.05, 0.95). centre_um (3,) um (image frame; depth in stage units); radius_um um."""
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
