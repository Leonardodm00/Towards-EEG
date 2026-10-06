"""Cell level: stretches, frame transform, per-node table, membrane area --
Block 8 in specs/SPEC.md (handoff step 7; D-013).

Stretches: maximal unbranched runs of dendrite nodes (SWC types 3, 4),
proximal -> distal; a stretch starts at a dendrite node whose parent is not a
dendrite node or has two or more dendrite children, and follows the single
dendrite child. Each stretch, with its parent dendrite node prepended when it
has one (so the first node's line fit has a neighbour), is one Block 5 Branch.

Frame: SWC um -> global image frame um, the transform of
allen_image_align.swc_to_full_px (its equations (1)-(2)) times the pixel
pitch, and z - z0 for the depth (plane k at k * dz):
    x = (x_swc / p + shift_x) p,  y = ((flip_h - y_swc / p if flipped else y_swc / p) + shift_y) p.

Membrane area: lateral frustum areas between each dendrite node and its parent
(swc_io.frustum_areas_um2's formula); a dendrite node whose parent is not a
dendrite node (the soma) contributes a cylinder of its own radius.

Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import math

import numpy as np

from ..loading import swc_io
from . import fill, invert
from .node_pipeline import Branch, measure_node


def stretches(swc, types=(3, 4)):
    """List of row-index arrays, one per unbranched dendrite stretch (proximal -> distal)."""
    dend = swc.dendrite_mask(types)
    pidx = swc.parent_index()
    children = {}
    for i in np.flatnonzero(dend):
        children.setdefault(int(pidx[i]), []).append(int(i))
    out = []
    for i in np.flatnonzero(dend):
        p = int(pidx[i])
        if p >= 0 and dend[p] and len(children.get(p, [])) == 1:
            continue                                    # i continues its parent's stretch
        run = [int(i)]
        while len(children.get(run[-1], [])) == 1:
            run.append(children[run[-1]][0])
        out.append(np.array(run, dtype=int))
    return out


def to_image_um(xyz_swc, res0_um, shift_full_px=(0.0, 0.0), flip_y_full_h=None, z0_um=0.0):
    """(n, 3) SWC um -> global image frame um (see the module docstring)."""
    P = np.asarray(xyz_swc, dtype=float)
    p = float(res0_um)
    x = P[:, 0] / p + shift_full_px[0]
    y = P[:, 1] / p
    if flip_y_full_h is not None:
        y = flip_y_full_h - y
    y = y + shift_full_px[1]
    return np.column_stack([x * p, y * p, P[:, 2] - z0_um])


def stretch_branch(swc, rows, xyz_img):
    """(Branch, offset): the stretch's nodes with the parent dendrite node prepended
    when there is one (a one-node stretch takes its parent, whatever its type);
    offset = index of the stretch's first node in the Branch."""
    pidx = swc.parent_index()
    dend = swc.dendrite_mask()
    first = int(rows[0])
    parent = int(pidx[first])
    lead = [parent] if parent >= 0 and dend[parent] else []
    idx = lead + [int(r) for r in rows]
    if len(idx) < 2 and parent >= 0:
        idx = [parent, first]
    if len(idx) < 2:
        raise ValueError("node %d: a lone root node has no direction" % int(swc.ids[first]))
    idx = np.array(idx, dtype=int)
    P = xyz_img[idx]
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    return Branch(swc.ids[idx], swc.types[idx], P, swc.radius[idx], s), int(np.flatnonzero(idx == first)[0])


def dendrite_area_um2(swc, radius=None, types=(3, 4)):
    """Membrane area (um^2) of the dendrites (see the module docstring)."""
    r = swc.radius if radius is None else np.asarray(radius, dtype=float)
    dend = swc.dendrite_mask(types)
    pidx = swc.parent_index()
    h = swc_io.segment_lengths_um(swc)
    total = 0.0
    for i in np.flatnonzero(dend):
        p = int(pidx[i])
        if p < 0:
            continue
        rp = r[p] if dend[p] else r[i]
        total += math.pi * (r[i] + rp) * math.sqrt(h[i] ** 2 + (r[i] - rp) ** 2)
    return total


def area_ratio(swc, radius_new, types=(3, 4)):
    """Dendritic membrane area with the new radii / with Allen's radii (handoff step 7)."""
    return dendrite_area_um2(swc, radius_new, types) / dendrite_area_um2(swc, None, types)


CSV_COLUMNS = ("node_id", "type", "x_um", "y_um", "z_um", "path_um", "reg_verdict", "s_star_um", "dz_star_um",
               "k_star", "z_sub_um", "cx_um", "cy_um", "cz_um", "theta_rad", "phi_rad", "steep", "vertical", "B_bar",
               "B_bar_region", "d_hat_um", "mu_hat_per_um", "v0_hat_um", "alpha_hat", "fit_status", "b_hat",
               "d_tilde_um", "flags", "filled_from", "d_final_um", "allen_radius_um", "sigma_fit_um",
               "d_tilde_sigma_spread_um")


def node_row(result, inversion, d_final, filled_from, allen_radius, sigma_fit, spread=float("nan")):
    """One per-node CSV row (Block 8 columns) from a NodeResult and an Inversion."""
    row = {c: getattr(result, c) for c in CSV_COLUMNS[:25]}
    flags = list(result.flags) + [f for f in inversion.flags if f not in result.flags]
    row.update(b_hat=inversion.b_hat, d_tilde_um=inversion.d_tilde_um, flags=";".join(flags),
               filled_from=str(filled_from), d_final_um=float(d_final), allen_radius_um=float(allen_radius),
               sigma_fit_um=float(sigma_fit), d_tilde_sigma_spread_um=float(spread))
    return row


def measure_cell(swc, provider, table, cfg, transform=None, regs=None, only=None, log=None):
    """Every dendrite node of the cell (or the stretches holding the node ids in
    `only`): measure (Block 5), correct (Block 7), fill per stretch. transform:
    keyword arguments of to_image_um (shift_full_px, flip_y_full_h, z0_um);
    regs: {node_id: registration dict}. Returns (CSV rows, new radii (N,) --
    Allen's where nothing was measured or the fill left NaN --, area ratio)."""
    types = cfg.acquisition.dendrite_swc_types
    xyz_img = to_image_um(swc.xyz, cfg.acquisition.res0_um, **(transform or {}))
    radius_new = swc.radius.astype(float).copy()
    rows = []
    for run in stretches(swc, types):
        if only is not None and not any(int(swc.ids[r]) in only for r in run):
            continue
        branch, off = stretch_branch(swc, run, xyz_img)
        results = [measure_node(branch, off + t, provider, cfg, (regs or {}).get(int(swc.ids[r])))
                   for t, r in enumerate(run)]
        invs = invert.correct_nodes(results, table, cfg)
        d_final, src = fill.fill_stretch([x.d_tilde_um for x in invs], 2.0 * swc.radius[run], cfg)
        radius_new[run] = np.where(np.isfinite(d_final), 0.5 * d_final, swc.radius[run])
        rows += [node_row(res, inv, dfin, sf, swc.radius[r], cfg.measure.sigma_fit_um)
                 for res, inv, dfin, sf, r in zip(results, invs, d_final, src, run)]
        if log is not None:
            log("stretch of %d nodes from node %d done" % (len(run), int(swc.ids[run[0]])))
    return rows, radius_new, area_ratio(swc, radius_new, types)
