"""A twin's measurement: the camera plane by plane with the real plane's background and noise, then the real
scan -- Block 55 in specs/SPEC.md (section 5, "Kernel calibration, step 2"; D-270.1, D-270.2). No file access.

  camera_planes    plane q of a TwinStack through Block 4's camera chain with the real plane's background
                   (background_B_gl = (B_gl[q] - black_level_gl) / gain, so that the chain's background grey level
                   black_level_gl + gain * background_B_gl is the real B_gl[q]) and noise SD (PROVISIONAL in D-270:
                   applied before quantisation and JPEG, the real SD was measured after them)
  twin_scene       the node, the straight branch and the scene the scans read
  noise_rngs       the per-plane generators default_rng([seed, cell_id, node_id, replicate, q]) (common random
                   numbers: the same draws for every trial table)
  measure_twin     the PlaneScan of one replicate: calibration.plane_scan (near) or far_field.far_plane_scan (far)
                   under the real scans' configuration
  twin_statistics  the replicates' mean and SD per plane

Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import dataclasses
import math
import types

import numpy as np

from ..model import camera, geometry
from .node_pipeline import Branch
from .plane_iteration import Scene


def camera_planes(stack, B_gl, noise_sd_gl, rcfg, rngs):
    """(block uint8 (n, H, W), valid (n,) bool): plane q of stack.tau through camera.camera_chain with
    background_B_gl = (B_gl[q] - rcfg.black_level_gl) / rcfg.gain and noise_sd_gl = noise_sd_gl[q], generator
    rngs[q]; a plane whose B_gl[q] is not finite or whose background_B_gl would be <= 0 is zeros, valid False."""
    tau = np.asarray(stack.tau, dtype=float)
    if tau.ndim != 3:
        raise ValueError("camera_planes: stack.tau must be (n, H, W), got %r" % (tau.shape,))
    n = tau.shape[0]
    B = np.asarray(B_gl, dtype=float).reshape(-1)
    sd = np.asarray(noise_sd_gl, dtype=float).reshape(-1)
    if B.size != n or sd.size != n or len(rngs) != n:
        raise ValueError("camera_planes: B_gl, noise_sd_gl and rngs need one entry per plane (%d)" % n)
    block = None
    valid = np.zeros(n, dtype=bool)
    for q in range(n):
        bq = float(B[q])
        bg = (bq - float(rcfg.black_level_gl)) / float(rcfg.gain) if math.isfinite(bq) else float("nan")
        if not (math.isfinite(bg) and bg > 0):
            continue
        sdq = float(sd[q])
        if not (math.isfinite(sdq) and sdq >= 0):
            raise ValueError("camera_planes: plane %d has a noise SD %r (finite, >= 0 needed)" % (q, sdq))
        rq = dataclasses.replace(rcfg, background_B_gl=bg, noise_sd_gl=sdq)
        img = camera.camera_chain(tau[q], rq, rngs[q])
        if block is None:
            block = np.zeros((n,) + img.shape, dtype=img.dtype)
        block[q] = img
        valid[q] = True
    if block is None:
        block = np.zeros(tau.shape, dtype=np.uint8 if int(rcfg.bit_depth) == 8 else np.uint16)
    return block, valid


def twin_scene(twin, cfg, s2cfg):
    """(node, branch, scene) for the scans: node carries the real pilot's node_id, k_star, z_sub_um, cx_um, cy_um,
    theta_rad, phi_rad, d_hat_um = d_um and alpha_hat = mu_per_um d_um / cos(phi_rad); branch a straight Branch
    along the twin's axis through (cx_um, cy_um, z_sub_um), nodes every cfg.phantom.phantom_node_step_um over
    +-s2cfg.tube_half_length_um, radius d_um / 2; scene = Scene.from_branch(branch)."""
    phi = float(twin.phi_rad)
    d = float(twin.d_um)
    node = types.SimpleNamespace(
        node_id=int(twin.node_id), d_hat_um=d, phi_rad=phi, alpha_hat=float(twin.mu_per_um) * d / math.cos(phi),
        k_star=int(twin.k_star), z_sub_um=float(twin.z_sub_um), cx_um=float(twin.cx_um), cy_um=float(twin.cy_um),
        theta_rad=float(twin.theta_rad))
    step = float(cfg.phantom.phantom_node_step_um)
    m = int(math.floor(float(s2cfg.tube_half_length_um) / step + 1e-9))
    s = np.arange(-m, m + 1) * step
    t = geometry.axis_direction(phi, float(twin.theta_rad))
    c = np.array([float(twin.cx_um), float(twin.cy_um), float(twin.z_sub_um)])
    branch = Branch.from_points(c[None, :] + s[:, None] * t[None, :], 0.5 * d)
    return node, branch, Scene.from_branch(branch)


def noise_rngs(twin, n_planes, s2cfg, replicate):
    """n_planes generators, plane q from numpy.random.default_rng([s2cfg.seed, cell_id, node_id, replicate, q]):
    the same for every trial table (common random numbers)."""
    return [np.random.default_rng([int(s2cfg.seed), int(twin.cell_id), int(twin.node_id), int(replicate), q])
            for q in range(int(n_planes))]


def measure_twin(twin, stack, scan_cfg, s2cfg, replicate=0, noise=True):
    """The PlaneScan of the twin's stack: camera_planes with twin.B_gl and the SDs s2cfg.noise_scale x
    twin.noise_sd_gl (zeros when noise is False), then calibration.plane_scan (set "near") or
    far_field.far_plane_scan (set "far") under scan_cfg, the real scans' configuration. ValueError when stack.ks is
    not the scan's stage_planes."""
    from allen_image_io import CropFrame
    from . import calibration, far_field

    if twin.set not in ("near", "far"):
        raise ValueError("measure_twin: twin.set must be 'near' or 'far', got %r" % (twin.set,))
    node, branch, scene = twin_scene(twin, scan_cfg, s2cfg)
    kk = calibration.stage_planes(node, scan_cfg, twin.set)
    ks = np.asarray(stack.ks)
    if ks.shape != kk.shape or not np.array_equal(ks, kk):
        raise ValueError("measure_twin: stack.ks %s is not the %s scan's planes %s" % (ks.tolist(), twin.set,
                                                                                     kk.tolist()))
    n = ks.size
    B = np.asarray(twin.B_gl, dtype=float)
    sd0 = np.asarray(twin.noise_sd_gl, dtype=float)
    if B.size != n or sd0.size != n:
        raise ValueError("measure_twin: twin.B_gl and twin.noise_sd_gl need one entry per plane (%d)" % n)
    sd = float(s2cfg.noise_scale) * sd0 if noise else np.zeros(n)
    # a non-finite SD (a real plane without a background) is an invalid plane only through B_gl; here it would
    # make the chain fail, so it is passed as is and camera_planes refuses it where the plane is otherwise valid
    block, valid = camera_planes(stack, B, sd, scan_cfg.renderer, noise_rngs(twin, n, s2cfg, replicate))
    patch = (block, ks, valid, CropFrame(int(stack.left), int(stack.top), 0, float(stack.res0_um)))
    if twin.set == "near":
        return calibration.plane_scan(patch, node, branch, scene, scan_cfg)
    return far_field.far_plane_scan(patch, node, branch, scene, scan_cfg)


def twin_statistics(twin, stack, scan_cfg, s2cfg, noise=True):
    """dict of (n,) arrays over stack.ks from R = s2cfg.noise_replicates replicates (1 when noise is False):
    "omega_um2" the mean of the finite values, NaN where fewer than ceil(R / 2) are finite; "omega_sd_um2" their
    sample SD (NaN for fewer than two); "area_gl_um", "bg_sd_gl" means; "n_ok" int."""
    R = int(s2cfg.noise_replicates) if noise else 1
    if R < 1:
        raise ValueError("twin_statistics: noise_replicates must be >= 1")
    scans = [measure_twin(twin, stack, scan_cfg, s2cfg, replicate=r, noise=noise) for r in range(R)]
    om = np.vstack([np.asarray(s.omega_um2, dtype=float) for s in scans])      # (R, n)
    area = np.vstack([np.asarray(s.area_gl_um, dtype=float) for s in scans])
    bsd = np.vstack([np.full(om.shape[1], np.nan) if s.bg_sd_gl is None else np.asarray(s.bg_sd_gl, dtype=float)
                     for s in scans])
    fin = np.isfinite(om)
    n_ok = fin.sum(axis=0).astype(int)
    with np.errstate(invalid="ignore", divide="ignore"):
        tot = np.where(fin, om, 0.0).sum(axis=0)
        mean = np.where(n_ok > 0, tot / np.maximum(n_ok, 1), np.nan)
        ss = np.where(fin, (om - mean[None, :]) ** 2, 0.0).sum(axis=0)
        sd = np.where(n_ok >= 2, np.sqrt(ss / np.maximum(n_ok - 1, 1)), np.nan)
    mean = np.where(n_ok >= int(math.ceil(R / 2.0)), mean, np.nan)
    return {"omega_um2": mean, "omega_sd_um2": sd,
            "area_gl_um": area.mean(axis=0), "bg_sd_gl": bsd.mean(axis=0), "n_ok": n_ok}
