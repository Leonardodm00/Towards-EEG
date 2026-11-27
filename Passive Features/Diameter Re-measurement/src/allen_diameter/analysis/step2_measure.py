"""A twin's measurement: the camera plane by plane with the real plane's background and noise, then the real
scan -- Block 55 in specs/SPEC.md (section 5, "Kernel calibration, step 2"; D-270.1, D-270.2). No file access.

CONTRACT STUB (wave 9, 2026-11-27): signatures and docstrings are binding; the bodies are Block 55's.
Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations


def camera_planes(stack, B_gl, noise_sd_gl, rcfg, rngs):
    """(block uint8 (n, H, W), valid (n,) bool): plane q of stack.tau through camera.camera_chain with
    background_B_gl = (B_gl[q] - rcfg.black_level_gl) / rcfg.gain and noise_sd_gl = noise_sd_gl[q], generator
    rngs[q]; a plane whose B_gl[q] is not finite or whose background_B_gl would be <= 0 is zeros, valid False."""
    raise NotImplementedError("Block 55")


def twin_scene(twin, cfg, s2cfg):
    """(node, branch, scene) for the scans: node carries the real pilot's node_id, k_star, z_sub_um, cx_um, cy_um,
    theta_rad, phi_rad, d_hat_um = d_um and alpha_hat = mu_per_um d_um / cos(phi_rad); branch a straight Branch
    along the twin's axis through (cx_um, cy_um, z_sub_um), nodes every cfg.phantom.phantom_node_step_um over
    +-s2cfg.tube_half_length_um, radius d_um / 2; scene = Scene.from_branch(branch)."""
    raise NotImplementedError("Block 55")


def noise_rngs(twin, n_planes, s2cfg, replicate):
    """n_planes generators, plane q from numpy.random.default_rng([s2cfg.seed, cell_id, node_id, replicate, q]):
    the same for every trial table (common random numbers)."""
    raise NotImplementedError("Block 55")


def measure_twin(twin, stack, scan_cfg, s2cfg, replicate=0, noise=True):
    """The PlaneScan of the twin's stack: camera_planes with twin.B_gl and the SDs s2cfg.noise_scale x
    twin.noise_sd_gl (zeros when noise is False), then calibration.plane_scan (set "near") or
    far_field.far_plane_scan (set "far") under scan_cfg, the real scans' configuration. ValueError when stack.ks is
    not the scan's stage_planes."""
    raise NotImplementedError("Block 55")


def twin_statistics(twin, stack, scan_cfg, s2cfg, noise=True):
    """dict of (n,) arrays over stack.ks from R = s2cfg.noise_replicates replicates (1 when noise is False):
    "omega_um2" the mean of the finite values, NaN where fewer than ceil(R / 2) are finite; "omega_sd_um2" their
    sample SD (NaN for fewer than two); "area_gl_um", "bg_sd_gl" means; "n_ok" int."""
    raise NotImplementedError("Block 55")
