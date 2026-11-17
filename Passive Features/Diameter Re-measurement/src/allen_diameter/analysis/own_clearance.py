"""The same-node clearance as a product -- SPEC Block 37 (D-151, D-156).

The node's features are estimated by the joint fit of all its planes with the whole own shadow (Blocks 29,
30); no fit reads a cleared profile (D-156.1, D-156.3). After the fit, the own shadow at the estimate, split
by depth (line_fast.line_shadow_split, Block 35 amendment 2), gives the cleared profiles, the in-band model,
the residual and the tube's chord at each plane.

Keep this file pure ASCII so it survives transfer to the cluster.
"""
from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass(frozen=True)
class OwnClearance:
    """cleared_gl, model_in_gl, model_gl, residual_gl  (n_planes, n_v) float64, gl:
           I + B S_out;  B (1 - S_in);  B (1 - S_in - S_out);  I - model_gl (= cleared_gl - model_in_gl)
    chord_um         (n_planes,) um: the tube's chord on the line at each plane's depth
    in_peak_gl       (n_planes,) gl: B_k max_n S_in
    out_peak_gl      (n_planes,) gl: B_k max_n S_out
    in_share         (n_planes,): S_in / (S_in + S_out) at the whole shadow's maximum; NaN where that
                     maximum is not positive
    residual_rms_gl  (n_planes,) gl: the root mean square of residual_gl over the line
    layers_gl        (n_layers, n_planes, n_v) gl, B_k times each layer's deficit, or None
    """

    cleared_gl: np.ndarray
    model_in_gl: np.ndarray
    model_gl: np.ndarray
    residual_gl: np.ndarray
    chord_um: np.ndarray
    in_peak_gl: np.ndarray
    out_peak_gl: np.ndarray
    in_share: np.ndarray
    residual_rms_gl: np.ndarray
    layers_gl: object


def chord_um(d_um, dz_um, phi_rad=0.0):
    """c = 2 sqrt(max(d^2 / 4 - (dz cos phi)^2, 0)) (um), the shape of dz_um: the part of the measuring line
    inside a round tube of diameter d_um (um, > 0) and tilt phi_rad at the depth dz_um = z_k - z_ax (um)
    from the axis' depth where it crosses the line (SPEC Block 37)."""
    raise NotImplementedError("SPEC Block 37")


def clear_profiles(profiles_gl, B_gl, s_in, s_out, planes_z, d_um, z_ax_um, phi_rad=0.0, layers=None):
    """-> OwnClearance (SPEC Block 37). profiles_gl, s_in, s_out (n_planes, n_v); B_gl (n_planes,) gl > 0;
    planes_z (n_planes,) um; d_um > 0, z_ax_um, phi_rad the estimate's diameter, axis depth at the line and
    tilt; layers (n_layers, n_planes, n_v) dimensionless or None. The inputs are not changed. ValueError for
    shapes that disagree, a non-finite input, B <= 0, d <= 0, a split entry below -1e-9."""
    raise NotImplementedError("SPEC Block 37")
