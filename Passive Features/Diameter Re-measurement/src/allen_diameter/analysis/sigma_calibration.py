"""The in-focus width calibrated on thick nodes -- the fit -- Block 42 in specs/SPEC.md (D-201).

For one node: the pipeline's hybrid line renderer as it is (node_inputs.forward_factory, line_fast.line_profiles),
its kernel re-anchored by kernel.with_kernel(renderer, sigma0, c) -- the kernel's variance sigma0^2 + c G(delta),
G the configured table's growth -- fitted to the recorded profiles of the planes k* - h .. k* + h with
theta = (d, mu, v0, z_ax, sigma0, c) and a straight background b0 + b1 v per plane solved exactly (variable
projection), from one start taken from the k* dip's features. The mathematics, the API and the oracles are in
SPEC Block 42; this file is the contract's stub (wave 4).

No file access, no plotting. Pure ASCII, LF, for the cluster.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

PARAMS = ("d_um", "mu_per_um", "v0_um", "z_ax_um", "sigma0_um", "growth_scale")
STATUSES = ("converged", "max_nfev", "failed")


@dataclass(frozen=True)
class SigmaFit:
    """What fit_sigma_node returns (SPEC Block 42). n_planes the fitted planes, n_v the line's samples.

    theta, se, se_scaled: (6,) in PARAMS' order and units (um, 1/um, um, um, um, 1); se from (J^T J)^-1, se_scaled
    = se sqrt(chi2 / dof); b (n_planes, 2) gl: the background (b0, b1 per um) per plane; model_gl (n_planes, n_v):
    (b0 + b1 v) tau at theta; chi2 = sum r^2 (r in noise SDs); chi2_per_plane (n_planes,); n_res the residuals;
    dof = n_res - (6 + 2 n_planes); n_eval the forward evaluations (the Jacobian's included); seconds; status one
    of STATUSES; at_bound the PARAMS within 1e-6 of a bound (relative to the bound's width); message."""
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


def select_planes(ks, k_star, half):
    """The sorted indices of ks with |k - k_star| <= half (SPEC Block 42); ValueError when fewer than 3 or
    half < 1."""
    raise NotImplementedError("Block 42: select_planes")


def vp_background(I, tau, v):
    """-> (b (n_planes, 2) gl, resid (n_planes, n_v) gl): per plane the least squares of I against tau and v tau
    over the finite samples (numpy.linalg.lstsq); resid = I - (b0 + b1 v) tau, NaN where I or tau is NaN; a plane
    with fewer than 3 finite samples gives NaN (SPEC Block 42)."""
    raise NotImplementedError("Block 42: vp_background")


def forward_with_kernel(pilot, half_length, node, cfg, cache):
    """-> f(d, mu, v0, z_ax, sigma0, c) -> (n_planes, n_v) dimensionless: node_inputs.forward_factory with cfg's
    renderer replaced by kernel.with_kernel(cfg.renderer, sigma0, c); NaN on line_fast.ExtrusionInvalid; the
    section cache shared across kernels (SPEC Block 42)."""
    raise NotImplementedError("Block 42: forward_with_kernel")


def start_theta(prof, B_gl, v, k_index, pilot, node_xyz, scfg):
    """-> (6,) the start of SPEC Block 42: v0 and z_ax from node_inputs.pilot_start; d0 = W(k*) / width_to_d;
    mu0 = -ln max(1 - D/B, 1e-3) / (hybrid_faint_factor d0); sigma0, c the configured starts; clipped."""
    raise NotImplementedError("Block 42: start_theta")


def fit_sigma_node(prof, sigma_gl, v, forward6, theta0, scfg):
    """-> SigmaFit: the variable-projection least squares of SPEC Block 42 (scipy.optimize.least_squares, trf,
    forward differences with scfg.diff_steps, scfg.max_nfev); never raises on a NaN forward or a solver failure."""
    raise NotImplementedError("Block 42: fit_sigma_node")


def summarize(sigma0_um, se_scaled_um, ok):
    """-> dict(n, median_um, mad_um, ivw_mean_um, ivw_se_um, chi2_cons, dof) over the ok nodes (SPEC Block 42)."""
    raise NotImplementedError("Block 42: summarize")
