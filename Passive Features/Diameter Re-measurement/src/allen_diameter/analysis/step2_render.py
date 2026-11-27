"""A twin's rendered stack under a trial kernel table -- Block 54 in specs/SPEC.md (section 5, "Kernel
calibration, step 2"). The noise-free, expensive part of step 2's forward model (D-270.1). No file access.

CONTRACT STUB (wave 9, 2026-11-27): signatures and docstrings are binding; the bodies are Block 54's.
Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

STAGES = ("near", "far")


@dataclass(frozen=True)
class TwinStack:
    """tau: (n_planes, H, W) float64, the pixel means of the transmittance (dimensionless), before the camera;
    ks: (n_planes,) int64 plane indices; left, top: the patch's corner (px of the full image); res0_um: the pixel
    pitch (um)."""

    tau: np.ndarray
    ks: np.ndarray
    left: int
    top: int
    res0_um: float


def trial_renderer(rcfg, sigma_r0_um, theta_um, gamma, s2cfg, stage):
    """rcfg (a RendererConfig) carrying a trial table: kernel_family = s2cfg.kernel_family, knots (0,) +
    s2cfg.knots_um with the values (sigma_r0_um,) + theta_um (um), sigma_r0_um, absorption = s2cfg.absorption;
    stage "near": kernel_continuation "tangent" (gamma ignored, may be None; D-265); "far": "linear" with slope
    gamma. ValueError: another stage; theta_um not of the knots' length or not finite and > 0; sigma_r0_um not
    finite and > 0; "far" with gamma None, not finite or < 0."""
    raise NotImplementedError("Block 54")


def twin_tube(twin, z_ax_um, cfg, s2cfg):
    """The twin's geometry.Tube: centre (cx_um, cy_um, z_ax_um), radius d_um / 2, tilt phi_rad, heading
    theta_rad, cfg.renderer.cross_section_aspect, half-length s2cfg.tube_half_length_um, cfg.renderer.end_cut."""
    raise NotImplementedError("Block 54")


def render_twin(twin, z_ax_um, rcfg_trial, cfg, s2cfg, ks=None, backend=None) -> TwinStack:
    """The twin's planes on its node's patch (calibration_scene.patch_bounds about (cx_um, cy_um) with
    cfg.calibration.patch_half_um), rendered as render.synthetic_block does up to and without the camera chain,
    with rcfg_trial as the renderer's configuration and the twin's mu_per_um. cfg: the real scans' DiameterConfig
    (acquisition, patch); ks None: twin.ks."""
    raise NotImplementedError("Block 54")
