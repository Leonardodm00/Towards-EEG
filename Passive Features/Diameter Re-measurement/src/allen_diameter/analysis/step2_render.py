"""A twin's rendered stack under a trial kernel table -- Block 54 in specs/SPEC.md (section 5, "Kernel
calibration, step 2"). The noise-free, expensive part of step 2's forward model (D-270.1). No file access.

From a Twin (Block 53), an axis depth z_ax (um) and a trial table theta = (theta_1, theta_2, theta_3) (um) with
the slope gamma (um per um, far stage only), the pixel means of the transmittance of the twin's planes on its
node's patch, before the camera chain (Block 55 applies it plane by plane). The rendering is Block 4's
(render.synthetic_block up to and without camera.camera_chain); only the renderer's configuration carries the
trial table (trial_renderer), the acquisition and the patch are the real scans' configuration.

Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import numpy as np

from ..model import geometry, render
from . import calibration_scene

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


def _finite_positive(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v) and v > 0.0


def trial_renderer(rcfg, sigma_r0_um, theta_um, gamma, s2cfg, stage):
    """rcfg (a RendererConfig) carrying a trial table: kernel_family = s2cfg.kernel_family, knots (0,) +
    s2cfg.knots_um with the values (sigma_r0_um,) + theta_um (um), sigma_r0_um, absorption = s2cfg.absorption;
    stage "near": kernel_continuation "tangent" (gamma ignored, may be None; D-265); "far": "linear" with slope
    gamma. ValueError: another stage; theta_um not of the knots' length or not finite and > 0; sigma_r0_um not
    finite and > 0; "far" with gamma None, not finite or < 0."""
    if stage not in STAGES:
        raise ValueError("stage must be one of %r, got %r" % (STAGES, stage))
    knots = tuple(float(x) for x in s2cfg.knots_um)
    theta = tuple(np.asarray(theta_um, dtype=float).reshape(-1).tolist())
    if len(theta) != len(knots):
        raise ValueError("theta_um must have %d values (one per knot %r), got %d" % (len(knots), knots, len(theta)))
    if not all(_finite_positive(t) for t in theta):
        raise ValueError("theta_um must be finite and > 0 (um), got %r" % (theta,))
    if not _finite_positive(sigma_r0_um):
        raise ValueError("sigma_r0_um must be finite and > 0 (um), got %r" % (sigma_r0_um,))
    s0 = float(sigma_r0_um)
    fields = dict(kernel_family=s2cfg.kernel_family, kernel_table_delta_um=(0.0,) + knots,
                  kernel_table_sigma_um=(s0,) + theta, sigma_r0_um=s0, absorption=s2cfg.absorption)
    if stage == "near":
        fields["kernel_continuation"] = "tangent"
    else:
        if gamma is None:
            raise ValueError("the far stage needs a slope gamma, got None")
        try:
            g = float(gamma)
        except (TypeError, ValueError):
            raise ValueError("gamma must be a finite number >= 0, got %r" % (gamma,))
        if not (math.isfinite(g) and g >= 0.0):
            raise ValueError("gamma must be finite and >= 0 (um per um), got %r" % (gamma,))
        fields["kernel_continuation"] = "linear"
        fields["kernel_continuation_slope"] = g
    return dataclasses.replace(rcfg, **fields)


def twin_tube(twin, z_ax_um, cfg, s2cfg):
    """The twin's geometry.Tube: centre (cx_um, cy_um, z_ax_um), radius d_um / 2, tilt phi_rad, heading
    theta_rad, cfg.renderer.cross_section_aspect, half-length s2cfg.tube_half_length_um, cfg.renderer.end_cut."""
    z = float(z_ax_um)
    if not math.isfinite(z):
        raise ValueError("z_ax_um must be finite (um), got %r" % (z_ax_um,))
    return geometry.Tube(c=(float(twin.cx_um), float(twin.cy_um), z), r=0.5 * float(twin.d_um),
                         phi=float(twin.phi_rad), theta=float(twin.theta_rad),
                         aspect=float(cfg.renderer.cross_section_aspect),
                         half_length=float(s2cfg.tube_half_length_um), end_cut=cfg.renderer.end_cut)


def render_twin(twin, z_ax_um, rcfg_trial, cfg, s2cfg, ks=None, backend=None) -> TwinStack:
    """The twin's planes on its node's patch (calibration_scene.patch_bounds about (cx_um, cy_um) with
    cfg.calibration.patch_half_um), rendered as render.synthetic_block does up to and without the camera chain,
    with rcfg_trial as the renderer's configuration and the twin's mu_per_um. cfg: the real scans' DiameterConfig
    (acquisition, patch); ks None: twin.ks."""
    acq = cfg.acquisition
    p_x = float(acq.res0_um)
    ks = np.asarray(twin.ks if ks is None else ks).reshape(-1).astype(np.int64)
    tube = twin_tube(twin, z_ax_um, cfg, s2cfg)
    left, top, width, height = calibration_scene.patch_bounds(twin.cx_um, twin.cy_um,
                                                              cfg.calibration.patch_half_um, p_x)
    pad_px = int(math.ceil(cfg.renderer.pad_um / p_x))
    f = render.fine_factor(tube.d, rcfg_trial, p_x)          # as synthetic_block, on the rendering configuration
    grid, inner = render.block_fine_grid(left, top, width, height, p_x, f, pad_px)
    res = render.render_transmittance(tube, float(twin.mu_per_um), ks * acq.dz_um, grid, rcfg_trial,
                                      acq.light_direction, backend,
                                      reduce=lambda plane: render.pixel_integrate(plane[inner[0], inner[1]], f))
    tau = np.asarray(res.tau, dtype=np.float64)
    return TwinStack(tau=tau, ks=ks, left=int(left), top=int(top), res0_um=p_x)
