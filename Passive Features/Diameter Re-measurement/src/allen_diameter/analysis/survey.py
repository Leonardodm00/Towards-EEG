"""Pilot measurements on real cells, before a bias table exists -- Block 11 in
specs/SPEC.md (design handoff Next actions 2-3 and 7; procedure s.3.5, s.3.10).

  measure_nodes      the per-node chain (Block 5) on the dendrite stretches
                     that hold the requested node ids; no correction
  pilot_row          one CSV row: the NodeResult columns, Allen's radius,
                     sigma_fit, membership of the selection S and its reasons
  summarize          what the pilot is for: percentiles of d_hat, mu_hat,
                     alpha_hat and d_hat / (2 r_Allen) over the nodes in S,
                     the phantom mu range they suggest (procedure s.3.10: the
                     10th and 90th percentiles of the real mu_hat), the share
                     the dark flag removes, and the calibration candidates
  profile_at_node    the fitted profile of a measured node, re-sampled from a
                     fresh block (for figures)
  background_stats   per node: the masked median, the robust SD (1.4826 x MAD)
                     and the clipped SD (analysis.camera_fit) of the unmasked
                     pixels of its block in plane k*

Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np

from . import background, calibration, camera_fit, cell, phantoms, profiles
from .node_pipeline import measure_node
from ..model import tube_model

PERCENTILES = (5, 10, 25, 50, 75, 90, 95)


def measure_nodes(swc, provider, cfg, transform=None, regs=None, only=None, log=None):
    """Every node of the dendrite stretches holding a node id of `only` (None:
    every dendrite node), measured (Block 5) and not corrected. Returns a list
    of (NodeResult, Allen radius um, Branch, index in the branch)."""
    types = cfg.acquisition.dendrite_swc_types
    xyz_img = cell.to_image_um(swc.xyz, cfg.acquisition.res0_um, **(transform or {}))
    out = []
    for run in cell.stretches(swc, types):
        if only is not None and not any(int(swc.ids[r]) in only for r in run):
            continue
        branch, off = cell.stretch_branch(swc, run, xyz_img)
        for t, r in enumerate(run):
            nid = int(swc.ids[r])
            out.append((measure_node(branch, off + t, provider, cfg, (regs or {}).get(nid)),
                        float(swc.radius[r]), branch, off + t))
        if log is not None:
            log("stretch of %d nodes from node %d measured" % (len(run), int(swc.ids[run[0]])))
    return out


def pilot_row(result, allen_radius_um, cfg):
    """The NodeResult columns of the per-node CSV (Block 8, before the
    correction), plus allen_radius_um, sigma_fit_um, in_S and reject."""
    row = {c: getattr(result, c) for c in cell.CSV_COLUMNS[:25]}
    row["flags"] = ";".join(result.flags)
    reasons = phantoms.reject_reasons(result)
    row.update(allen_radius_um=float(allen_radius_um), sigma_fit_um=float(cfg.measure.sigma_fit_um),
               in_S=not reasons, reject=";".join(reasons),
               calibration_node=not calibration.calibration_reasons(result, cfg))
    return row


def _pct(x):
    x = np.asarray([v for v in x if math.isfinite(v)], dtype=float)
    if x.size == 0:
        return {}
    out = {"p%d" % q: float(np.percentile(x, q)) for q in PERCENTILES}
    out["n"] = int(x.size)
    return out


def summarize(rows, cfg):
    """Counts, rejection reasons, percentiles over the nodes in S, the
    suggested phantom mu range and the dark-flag share (see the module doc)."""
    n = len(rows)
    inS = [r for r in rows if r["in_S"]]
    reasons: Dict[str, int] = {}
    for r in rows:
        for x in [x for x in str(r["reject"]).split(";") if x]:
            reasons[x] = reasons.get(x, 0) + 1
    converged = [r for r in rows if r["fit_status"] == "converged"]
    dark = sum(1 for r in converged if float(r["alpha_hat"]) > cfg.measure.alpha_dark_flag)
    mu = _pct(float(r["mu_hat_per_um"]) for r in inS)
    out = dict(n_nodes=n, n_in_S=len(inS), reject_counts=reasons,
               d_hat_um=_pct(float(r["d_hat_um"]) for r in inS),
               mu_hat_per_um=mu,
               alpha_hat=_pct(float(r["alpha_hat"]) for r in inS),
               d_hat_over_allen_d=_pct(float(r["d_hat_um"]) / (2.0 * float(r["allen_radius_um"])) for r in inS
                                       if float(r["allen_radius_um"]) > 0),
               phi_deg=_pct(math.degrees(float(r["phi_rad"])) for r in inS),
               dark_share_of_converged=(dark / len(converged)) if converged else float("nan"),
               n_calibration_nodes=sum(1 for r in rows if r.get("calibration_node")))
    if mu:
        out["suggested_phantom_mu_range_per_um"] = [mu["p10"], mu["p90"]]
    return out


def profile_at_node(provider, result, cfg):
    """(v (n,) um, I (n,) grey levels, model (n,) grey levels or None) of a
    measured node: the Eq. 9 profile through its centre along its final line
    in plane k*, from a fresh one-plane block (cached for real data), and the
    fitted model at the same offsets."""
    m, p = cfg.measure, cfg.acquisition.res0_um
    if not math.isfinite(result.z_sub_um):
        raise ValueError("profile_at_node: the node has no sharpest plane")
    left = int(math.floor((result.cx_um - m.block_half_um) / p)) - 1
    top = int(math.floor((result.cy_um - m.block_half_um) / p)) - 1
    width = int(math.ceil((result.cx_um + m.block_half_um) / p)) + 2 - left
    height = int(math.ceil((result.cy_um + m.block_half_um) / p)) + 2 - top
    block, ks, valid, frame = provider(left, top, width, height, result.k_star, result.k_star)
    th = float(result.theta_rad)
    v = profiles.profile_offsets(m)
    I = profiles.sample_profile(block[0], frame, np.array([result.cx_um, result.cy_um]),
                                np.array([-math.sin(th), math.cos(th)]), np.array([math.cos(th), math.sin(th)]), v,
                                profiles.n_along(m), m.profile_step_um)
    model = None
    if result.fit is not None:          # v0_hat is relative to the sampling origin (cx, cy), as in measure_node
        f = result.fit
        model = tube_model.model_profile(v, f.d_hat_um, f.alpha_hat, f.v0_hat_um, m.sigma_fit_um, result.B_bar,
                                         f.n_nodes)
    return v, I, model


def background_stats(provider, result, branch, cfg):
    """(masked median, robust SD = 1.4826 MAD, clipped SD (camera_fit), unmasked
    fraction) of the node's block (block_half_um around its centre) in plane k*,
    with the branch's path masked as in D-018.1. NaN when the node has no
    sharpest plane."""
    m, p = cfg.measure, cfg.acquisition.res0_um
    nan = float("nan")
    if not math.isfinite(result.z_sub_um):
        return nan, nan, nan, nan
    left = int(math.ceil((result.cx_um - m.block_half_um) / p))
    top = int(math.ceil((result.cy_um - m.block_half_um) / p))
    width = int(math.floor((result.cx_um + m.block_half_um) / p)) + 1 - left
    height = int(math.floor((result.cy_um + m.block_half_um) / p)) + 1 - top
    block, ks, valid, frame = provider(left, top, width, height, result.k_star, result.k_star)
    if not bool(valid[0]):
        return nan, nan, nan, nan
    plane = np.asarray(block[0], dtype=float)
    mask = background.mask_near_branch(plane.shape, left, top, p, branch.xyz_um, branch.radius_um,
                                       m.bbar_mask_margin_um)
    vals = plane[~mask]
    if vals.size == 0:
        return nan, nan, nan, 0.0
    med = float(np.median(vals))
    return (med, float(1.4826 * np.median(np.abs(vals - med))), camera_fit.clipped_sd(vals),
            float(vals.size) / plane.size)


def background_summary(stats: List[tuple]):
    """Percentiles over nodes of the masked median, the robust SD, the clipped SD and the unmasked fraction."""
    return dict(B_bar_gl=_pct(s[0] for s in stats), robust_sd_gl=_pct(s[1] for s in stats),
                clipped_sd_gl=_pct(s[2] for s in stats), unmasked_frac=_pct(s[3] for s in stats))
