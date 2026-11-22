"""The soma's forward model -- Block 45 in specs/SPEC.md (D-215, D-216, D-218, D-219). CONTRACT STUB (wave 5).

An ellipsoid soma E(theta_g), theta_g = (x0, y0, z0, a, b, c, phi, mu), rendered with the partition absorption
(procedure Eq. 5: Delta A_j = T_<j (1 - exp(-a_j)) per column and slab, model.geometry) and the pipeline's kernel
(model.kernel.sigma_r; the continuation slope replaceable by a parameter, D-218):

    tau_k = pixel_integrate(1 - sum_j g_{s_jk} (*) Delta A_j, f),   s_jk = sigma_r(zeta_j - z_k; gamma)

on the crop's samples (render.block_fine_grid, f = soma_fit.fine_factor), g_s the normalised sampled Gaussian (a
positive kernel of sum 1; D-219.2), applied by FFT on a zero-padded box (Block 4's rule, renderer.fft_wrap_sigmas)
with its transfer by Poisson summation (sampled_gaussian_transfer).
Image frame: pixel (row, col) of a crop with corner (left, top) is centred on ((left + col) p, (top + row) p) um,
p = res0; depth in stage units. See SPEC Block 45 for the full statement, the parameters and the oracles.

No file access, no plotting. Pure ASCII, LF, for the cluster.
"""
from __future__ import annotations

import numpy as np


def ellipsoid_columns(x_um, y_um, theta_g):
    """-> (z_lo, z_hi, inside): for columns at (x_um, y_um) (arrays, broadcast together; um, image frame) the depth
    interval [z0 - c sqrt(q), z0 + c sqrt(q)] (um, stage units) where q = 1 - (u/a)^2 - (v/b)^2 > 0 (inside, bool),
    u, v the column's coordinates rotated by phi about (x0, y0); outside z_lo = z_hi = z0. theta_g (8,):
    x0, y0, z0, a, b, c (um), phi (rad), mu (1/um). ValueError for a, b or c not finite and > 0, mu < 0 or not
    finite."""
    raise NotImplementedError("Block 45 (wave 5)")


def sampled_gaussian_transfer(s_um, nu_per_um, h_um):
    """-> the transfer (DFT) of the normalised sampled Gaussian of SD s_um on samples h_um apart, at the frequencies
    nu_per_um (cycles/um; any shape), dimensionless:

        w_s(nu) = sum_{m=-M..M} exp(-2 pi^2 s^2 (nu - m/h)^2) / sum_{m=-M..M} exp(-2 pi^2 s^2 m^2 / h^2),
        M = max(1, ceil(1.2 h / s))   (the aliases left out below 1e-12).

    ValueError unless s_um and h_um are finite and > 0."""
    raise NotImplementedError("Block 45 (wave 5)")


class SomaRenderer:
    """The planes of an ellipsoid soma on a crop (SPEC Block 45).

    SomaRenderer(crop, res0_um, z_planes_um, rcfg, scfg, light_direction=+1): crop = (left, top, width, height),
    integers, px of the full-resolution image; res0_um the pixel size (um); z_planes_um (K,) the planes' depths (um,
    stage units); rcfg a RendererConfig (the kernel: its table and continuation; fft_wrap_sigmas); scfg the
    SomaFitConfig (fine_factor, dzeta_um); light_direction +1 or -1 (the caller passes acquisition.light_direction).
    Holds the sample grid and the box's frequencies for repeated renders.
    """

    def __init__(self, crop, res0_um, z_planes_um, rcfg, scfg, light_direction=+1):
        raise NotImplementedError("Block 45 (wave 5)")

    def render(self, theta_g, slope=None):
        """-> tau (K, height, width) float64, dimensionless, the planes in the order of z_planes_um; tau <= 1, no
        lower bound imposed (the model does not clip). theta_g (8,) as ellipsoid_columns; slope (um of SD per um of
        defocus, finite, >= 0) replaces the kernel's continuation slope (the continuation must be "linear";
        ValueError otherwise). mu = 0 gives tau = 1 exactly."""
        raise NotImplementedError("Block 45 (wave 5)")
