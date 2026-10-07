"""Pilot measurements on real cells, before a bias table exists -- Block 11 in
specs/SPEC.md (design handoff Next actions 2-3 and 7; procedure s.3.5, s.3.10).

  measure_nodes      the per-node chain (Block 5) on the dendrite stretches
                     that hold the requested node ids; no correction
  pilot_row          one CSV row: the NodeResult columns, Allen's radius,
                     sigma_fit, membership of the selection S and its reasons,
                     the focus rule and the plane the dip depth would have
                     chosen (k_star_depth, D-030 diagnostic)
  summarize          what the pilot is for: percentiles of d_hat, mu_hat,
                     alpha_hat and d_hat / (2 r_Allen) over the nodes in S,
                     the phantom mu range they suggest (procedure s.3.10: the
                     10th and 90th percentiles of the real mu_hat), the share
                     the dark flag removes, the calibration candidates, and
                     how often the focus rule and the dip depth disagree
  profile_at_node    the fitted profile of a measured node, re-sampled from a
                     fresh block (for figures)
  background_stats   per node: the masked median, the robust SD (1.4826 x MAD)
                     and the clipped SD (analysis.camera_fit) of the unmasked
                     pixels of its block in plane k*
  stretch_table      one row per unbranched dendrite stretch (type, nodes,
                     length, path distance, order, Allen's diameter, node ids)
  suggest_nodes      a deterministic node list for the registration survey and
                     the pilot: per SWC type, stretches spread over path
                     distance (2026-10-07)

Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np

from . import background, calibration, camera_fit, cell, phantoms, profiles
from .node_pipeline import measure_node
from ..loading import swc_io
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
               calibration_node=not calibration.calibration_reasons(result, cfg),
               focus_rule=str(result.focus_rule), k_star_depth=int(result.k_star_depth))
    return row


def focus_agreement(rows):
    """Over the rows with a sharpest plane (finite z_sub_um; both scores are
    then finite on the same planes, so k_star_depth is defined): how many, how
    many have k_star != k_star_depth, the largest |k_star - k_star_depth| and
    the count per difference in planes (D-030 diagnostic). Rows without the
    D-030 columns (a pilot written before 2026-10-07) are not counted."""
    diffs = [abs(int(r["k_star"]) - int(r["k_star_depth"])) for r in rows
             if "k_star_depth" in r and math.isfinite(float(r.get("z_sub_um", float("nan"))))]
    counts: Dict[str, int] = {}
    for d in diffs:
        if d:
            counts[str(d)] = counts.get(str(d), 0) + 1
    return dict(n_nodes=len(diffs), n_differ=sum(1 for d in diffs if d), max_abs_planes=max(diffs) if diffs else 0,
                counts_by_planes=counts)


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
               n_calibration_nodes=sum(1 for r in rows if r.get("calibration_node")),
               focus_rule=str(cfg.measure.focus_rule), k_star_vs_dip_depth=focus_agreement(rows))
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


def path_distance_um(swc):
    """(N,) path distance (um) along the SWC from each node's tree root (roots 0)."""
    pidx = swc.parent_index()
    seg = swc_io.segment_lengths_um(swc)
    children: Dict[int, List[int]] = {}
    for i, p in enumerate(pidx):
        children.setdefault(int(p), []).append(i)
    dist = np.full(len(swc), np.nan)
    stack = list(children.get(-1, []))
    for i in stack:
        dist[i] = 0.0
    while stack:
        i = stack.pop()
        for c in children.get(i, []):
            if not np.isnan(dist[c]):
                raise ValueError("SWC node %d is reached twice: the parent links contain a cycle" % int(swc.ids[c]))
            dist[c] = dist[i] + seg[c]
            stack.append(c)
    if np.isnan(dist).any():
        raise ValueError("SWC nodes %s are not connected to a root (a cycle in the parent links)"
                         % [int(x) for x in swc.ids[np.isnan(dist)][:5]])
    return dist


def stretch_table(swc, types=(3, 4)):
    """One dict per unbranched dendrite stretch, in the order of cell.stretches:
    stretch, type (of its first node), n_nodes, first_node, mid_node (index
    n // 2), last_node, length_um (the stretch's node-to-parent segments, the
    first node's link included), path_start_um (path distance from the root to
    the first node), order (ancestor stretches), allen_diameter_um (median 2 r),
    terminal (the last node has no dendrite child)."""
    runs = cell.stretches(swc, types)
    pidx = swc.parent_index()
    dend = swc.dendrite_mask(types)
    seg = swc_io.segment_lengths_um(swc)
    dist = path_distance_um(swc)
    n_dend_children = np.zeros(len(swc), dtype=int)
    for i in np.flatnonzero(dend):
        if pidx[i] >= 0:
            n_dend_children[pidx[i]] += 1
    owner = {int(r): s for s, run in enumerate(runs) for r in run}
    parent_of = []
    for run in runs:
        p = int(pidx[int(run[0])])
        parent_of.append(owner.get(p) if p >= 0 and dend[p] else None)
    rows = []
    for s, run in enumerate(runs):
        order, q = 0, parent_of[s]
        while q is not None:
            order, q = order + 1, parent_of[q]
            if order > len(runs):
                raise ValueError("stretch %d: the stretch tree contains a cycle" % s)
        first, last = int(run[0]), int(run[-1])
        rows.append(dict(stretch=s, type=int(swc.types[first]), n_nodes=int(len(run)), first_node=int(swc.ids[first]),
                         mid_node=int(swc.ids[int(run[len(run) // 2])]), last_node=int(swc.ids[last]),
                         length_um=float(seg[run].sum()), path_start_um=float(dist[first]), order=int(order),
                         allen_diameter_um=float(np.median(2.0 * swc.radius[run])),
                         terminal=bool(n_dend_children[last] == 0)))
    return rows


def suggest_nodes(swc, types=(3, 4), per_type=3, min_nodes=10, include=()):
    """Node ids for the registration survey and the pilot (SPEC Block 11, stretch
    list): the ids of `include` first, each standing for its own stretch; then,
    per SWC type in ascending order, among the stretches with at least
    min_nodes nodes sorted by path_start_um (ties by index), those at ranks
    floor(j (m - 1) / (per_type - 1) + 1/2), j = 0 .. per_type - 1 (all when
    m <= per_type; rank (m - 1) // 2 when per_type = 1), each by its mid node,
    skipping a stretch already represented. ValueError for an include id that
    is not a dendrite node."""
    if int(per_type) < 1 or int(min_nodes) < 1:
        raise ValueError("per_type and min_nodes must be >= 1")
    per_type = int(per_type)
    rows = stretch_table(swc, types)
    of_id = {int(swc.ids[r]): s for s, run in enumerate(cell.stretches(swc, types)) for r in run}
    out, taken = [], set()
    for nid in include:
        nid = int(nid)
        if nid not in of_id:
            raise ValueError("node %d is not a dendrite node (SWC types %s) of this file" % (nid, list(types)))
        if nid not in out:
            out.append(nid)
        taken.add(of_id[nid])
    for t in sorted({r["type"] for r in rows}):
        elig = sorted((r for r in rows if r["type"] == t and r["n_nodes"] >= int(min_nodes)),
                      key=lambda r: (r["path_start_um"], r["stretch"]))
        for k in spread_ranks(len(elig), per_type):
            if elig[k]["stretch"] not in taken:
                taken.add(elig[k]["stretch"])
                out.append(elig[k]["mid_node"])
    return out


def spread_ranks(m, k):
    """k ranks spread evenly over 0 .. m - 1: all of them when m <= k, (m - 1) // 2
    when k = 1, else floor(j (m - 1) / (k - 1) + 1/2) for j = 0 .. k - 1 (distinct,
    since the step exceeds 1)."""
    if m <= k:
        return list(range(m))
    if k == 1:
        return [(m - 1) // 2]
    return sorted({int(math.floor(j * (m - 1) / (k - 1) + 0.5)) for j in range(k)})
