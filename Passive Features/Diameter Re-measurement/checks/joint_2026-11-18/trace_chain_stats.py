#!/usr/bin/env python3
"""Standalone check (NOT pipeline code), 2026-11-18, for the joint fit's chain geometry (D-164) and its open
points O6 to O8 (handoff TEEG_joint_fit_handoff_2026-11-18.md, section 12: "the node spacing of the pilot's trace";
"whether the traced depth is good enough for the tilt"). Reads a cell's SWC (Allen's, in um; the image-frame
transform is a shift and a flip, so every length and angle here is unchanged by it) and reports, with the
pipeline's own functions and default configuration:

  1  spacing   distance between consecutive traced nodes on the dendrite stretches (cell.stretches, types 3, 4)
  2  chain     per node not near an end (node_inputs.is_near_end): the number of pieces of its own chain within
               tube_half_length_um each way (own_ids_within); the turn angle at the node between its two pieces
               (3-D, and projected on the image plane)
  3  straight  per such node: the chain's largest distance from the straight line the pipeline's tube lies on --
               the total-least-squares line through the traced nodes within +-line_fit_window_um/2 of arc length
               (numpy.linalg.svd), taken through the node -- lateral (in the image plane, across the line) and in
               depth, over the nodes within +-2 um and within the whole own chain
  4  depth     the traced depth's scatter: SD of the z residual about that line over the +-2 um window, and the
               depth step |z_j+1 - z_j| between consecutive nodes, against the plane spacing dz_um
               (with the trace's spacing that window holds three nodes, so this SD says little; the offset of a
               node from the chord of its two neighbours is reported as the estimate of the trace's scatter)
  5  tiers     per such node: today's rule (cleaning.neighbour_tiers), split into its rule (a) and rule (b), with
               the depth distance of the segments rule (b) alone makes tier 2; and the nodes with an attached
               neighbour (on the own stretch beyond the own chain, or on a stretch sharing a branch point with it)
               inside rule (a). Which nodes have a neighbour ON THEIR MEASURING LINE: close_neighbour_nodes.py

Run:  python3 trace_chain_stats.py "<repo>/Passive Features/Diameter Re-measurement/src" <file.swc> [n_tier_nodes]
"""
import math
import sys
import time

import numpy as np

sys.path.insert(0, sys.argv[1])
from allen_diameter.config import default_config                               # noqa: E402
from allen_diameter.loading import swc_io                                       # noqa: E402
from allen_diameter.analysis import background, cell, cleaning, node_features, node_inputs   # noqa: E402


def pct(x, qs=(5, 25, 50, 75, 95)):
    return " ".join("%.2f" % v for v in np.percentile(np.asarray(x, dtype=float), qs))


def tls_dir(P):
    c = P.mean(axis=0)
    return np.linalg.svd(P - c, full_matrices=False)[2][0]


if __name__ == "__main__":
    t0 = time.time()
    cfg = default_config()
    swc = swc_io.read_swc(sys.argv[2])
    n_tier = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    res0, dz = float(cfg.acquisition.res0_um), float(cfg.acquisition.dz_um)
    xyz = cell.to_image_um(swc.xyz, res0)
    ids = np.asarray(swc.ids)
    runs = cell.stretches(swc, tuple(cfg.acquisition.dendrite_swc_types))
    W = 0.5 * float(cfg.measure.line_fit_window_um)
    print("%d SWC nodes; %d dendrite stretches (types %s); tube_half_length %.1f um, near-end margin %.1f um, "
          "line-fit window +-%.1f um, dz %.2f um" % (len(ids), len(runs), tuple(cfg.acquisition.dendrite_swc_types),
                                                    cfg.inference.tube_half_length_um,
                                                    cfg.inference.extrusion_cap_margin_um, W, dz))

    spacing, zstep = [], []
    stretch_of, branch_of = {}, {}
    for si, run in enumerate(runs):
        ob = node_inputs.own_branch(swc, run, xyz)
        for r in run:
            stretch_of[int(ids[r])] = si
        if ob is None:
            continue
        b = ob[0]
        branch_of[si] = b
        spacing.extend(np.diff(b.s_um))
        zstep.extend(np.abs(np.diff(b.xyz_um[:, 2])))
    spacing, zstep = np.array(spacing), np.array(zstep)
    print("\n1 spacing (um), %d pieces: percentiles 5 25 50 75 95 = %s; mean %.2f; max %.2f"
          % (spacing.size, pct(spacing), spacing.mean(), spacing.max()))

    rows = []          # (stretch, index) of the nodes not near an end
    n_pieces, turn3, turn2, lat2, lat5, dep2, dep5, zsd, xysd = ([] for _ in range(9))
    n_all = 0
    for si, b in branch_of.items():
        P, s = b.xyz_um, b.s_um
        for i in range(len(b.ids)):
            n_all += 1
            if node_inputs.is_near_end(b, i, cfg):
                continue
            rows.append((si, i))
            own, um, up = node_inputs.own_ids_within(b, i, cfg.inference.tube_half_length_um)
            n_pieces.append(len(own) - 1)
            a, c = P[i] - P[i - 1], P[i + 1] - P[i]
            turn3.append(math.degrees(math.acos(np.clip(a @ c / (np.linalg.norm(a) * np.linalg.norm(c)), -1, 1))))
            na, nc = np.linalg.norm(a[:2]), np.linalg.norm(c[:2])
            turn2.append(math.degrees(math.acos(np.clip(a[:2] @ c[:2] / (na * nc), -1, 1))) if na > 0 and nc > 0
                         else float("nan"))
            win = np.abs(s - s[i]) <= W
            t = tls_dir(P[win])
            h = t[:2] / max(np.linalg.norm(t[:2]), 1e-12)
            n2 = np.array([-h[1], h[0]])
            chain = (s >= s[i] - um - 1e-9) & (s <= s[i] + up + 1e-9)
            for sel, L, Dz in ((win, lat2, dep2), (chain, lat5, dep5)):
                D = P[sel] - P[i]
                u = D[:, :2] @ h
                L.append(np.abs(D[:, :2] @ n2).max())
                slope = t[2] / max(np.linalg.norm(t[:2]), 1e-12) * (1.0 if t[:2] @ h >= 0 else -1.0)
                Dz.append(np.abs(D[:, 2] - slope * u).max())
            D = P[win] - P[win].mean(axis=0)
            perp = D - np.outer(D @ t, t)
            zsd.append(perp[:, 2].std())
            xysd.append(np.linalg.norm(perp[:, :2], axis=1).std() if win.sum() > 2 else 0.0)
    print("\n2 chain: %d of %d stretch nodes are not near an end. Pieces in the own chain: percentiles %s"
          % (len(rows), n_all, pct(n_pieces)))
    print("  turn angle at the node (deg), 3-D: %s; in the image plane: %s; share of nodes turning > 20 deg in the "
          "plane: %.1f %%, > 45 deg: %.1f %%" % (pct(turn3), pct(np.array(turn2)[np.isfinite(turn2)]),
                                                  100 * np.nanmean(np.array(turn2) > 20),
                                                  100 * np.nanmean(np.array(turn2) > 45)))
    print("\n3 straight: the chain's largest distance from the tube's line through the node (um), percentiles 5 25 "
          "50 75 95")
    print("  lateral, nodes within +-%.0f um: %s; whole own chain: %s" % (W, pct(lat2), pct(lat5)))
    print("  depth,   nodes within +-%.0f um: %s; whole own chain: %s" % (W, pct(dep2), pct(dep5)))
    for thr in (0.11, 0.23, 0.5):
        print("  share of nodes whose chain leaves the line laterally by > %.2f um within +-%.0f um: %.1f %%; over the "
              "whole chain: %.1f %%" % (thr, W, 100 * np.mean(np.array(lat2) > thr), 100 * np.mean(np.array(lat5) > thr)))
    print("\n4 depth: SD about the line within +-%.0f um, of z: %s um; of the lateral position: %s um"
          % (W, pct(zsd), pct(xysd)))
    print("  depth step between consecutive nodes |dz| (um): %s; share > one plane (%.2f um): %.1f %%, > 3 planes: "
          "%.1f %%" % (pct(zstep), dz, 100 * np.mean(zstep > dz), 100 * np.mean(zstep > 3 * dz)))
    print("  tilt of a piece, atan(|dz| / horizontal length) (deg): %s"
          % pct(np.degrees(np.arctan2(zstep, np.sqrt(np.maximum(spacing ** 2 - zstep ** 2, 0.0))))))

    # trace jitter: the middle node of three consecutive ones against the chord of its two neighbours
    lat_e, z_e = [], []
    for b in branch_of.values():
        P = b.xyz_um
        if len(P) < 3:
            continue
        e = P[1:-1] - 0.5 * (P[:-2] + P[2:])
        ch = P[2:, :2] - P[:-2, :2]
        nrm = np.linalg.norm(ch, axis=1)
        ok = nrm > 0
        nvec = np.column_stack([-ch[ok, 1], ch[ok, 0]]) / nrm[ok, None]
        lat_e.extend(np.sum(e[ok, :2] * nvec, axis=1))
        z_e.extend(e[:, 2])
    lat_e, z_e = np.array(lat_e), np.array(z_e)
    print("  middle node against the chord of its two neighbours (um): lateral SD %.3f (|.| percentiles %s); depth SD "
          "%.3f (|.| percentiles %s). For independent trace errors of SD s per node the SD of this offset is 1.22 s, "
          "so s <= %.2f um laterally and <= %.2f um in depth (the dendrite's own curvature included)"
          % (lat_e.std(), pct(np.abs(lat_e)), z_e.std(), pct(np.abs(z_e)), lat_e.std() / math.sqrt(1.5),
             z_e.std() / math.sqrt(1.5)))

    if n_tier:
        types = tuple(cfg.cleaning.neighbour_swc_types)
        scene = cell.cell_scene(swc, types, res0, None)
        fscene = node_features.featured_scene(scene, node_features.initial_features(swc, types, 0.5))
        seg_ids = np.asarray(fscene.seg_ids).astype(np.int64)
        seg_xyz = np.asarray(fscene.seg_xyz, dtype=float)
        r_max = 0.5 * np.asarray(fscene.seg_d_um, dtype=float).max(axis=1)
        seg_stretch = np.array([stretch_of.get(int(c), -1) for c in seg_ids[:, 0]])    # -1: an axon segment
        pidx = swc.parent_index()
        parent_of = {int(ids[k]): (int(ids[pidx[k]]) if pidx[k] >= 0 else -1) for k in range(len(ids))}
        first_parent = np.array([parent_of[int(ids[run[0]])] for run in runs])
        last_node = np.array([int(ids[run[-1]]) for run in runs])
        rng = np.random.default_rng(20261118)
        pick = rng.choice(len(rows), size=min(n_tier, len(rows)), replace=False)
        reach = float(cfg.cleaning.tier_reach_um)
        a, b_ = seg_xyz[:, :3], seg_xyz[:, 3:]
        dxy = b_[:, :2] - a[:, :2]
        L2 = np.sum(dxy * dxy, axis=1)
        cnt = {"today2": 0, "today_a": 0, "today_b_only": 0, "att_a": 0}
        b_defocus = []
        for k in pick:
            si, i = rows[k]
            br = branch_of[si]
            own, um, up = node_inputs.own_ids_within(br, i, cfg.inference.tube_half_length_um)
            node = br.xyz_um[i]
            own_arr = np.asarray(sorted(own))
            is_own = np.isin(seg_ids[:, 0], own_arr) & np.isin(seg_ids[:, 1], own_arr)
            nbm = ~is_own
            tiers = cleaning.neighbour_tiers(fscene, own, node, cfg)
            t2 = np.zeros(len(seg_ids), dtype=bool)
            t2[np.flatnonzero(nbm)] = tiers[:int(nbm.sum())] == 2
            with np.errstate(invalid="ignore", divide="ignore"):
                t = np.where(L2 == 0, 0.0, np.clip(np.sum((node[:2] - a[:, :2]) * dxy, axis=1) / L2, 0.0, 1.0))
            lat = np.hypot(*(node[:2] - (a[:, :2] + t[:, None] * dxy)).T)
            dfc = np.abs(a[:, 2] + t * (b_[:, 2] - a[:, 2]) - node[2])
            fp_own = first_parent[si]
            att_st = set(np.flatnonzero((first_parent == fp_own) | (first_parent == last_node[si])).tolist()) | {si}
            if fp_own in stretch_of:
                att_st.add(stretch_of[fp_own])
            attached = np.isin(seg_stretch, np.asarray(sorted(att_st)))
            rule_a = (lat <= float(cfg.cleaning.close_lateral_um)) & (dfc <= float(cfg.cleaning.close_defocus_um))
            cnt["today2"] += bool(np.any(t2))
            cnt["today_a"] += bool(np.any(t2 & rule_a))
            cnt["today_b_only"] += bool(np.any(t2) and not np.any(t2 & rule_a))
            cnt["att_a"] += bool(np.any(nbm & attached & rule_a))
            b_defocus.extend(dfc[t2 & ~rule_a])
        n = len(pick)
        print("\n5 tiers, %d nodes not near an end (neighbour types %s)" % (n, types))
        print("  TODAY's rule (cleaning.neighbour_tiers): nodes with a tier-2 neighbour %.1f %%; with one by rule (a) "
              "(within %.1f um laterally and %.2f um in depth) %.1f %%; by rule (b) only (crossing the own stretch "
              "within %.1f um, any depth) %.1f %%. Defocus of the rule-(b)-only tier-2 segments (um): percentiles %s"
              % (100 * cnt["today2"] / n, cfg.cleaning.close_lateral_um, cfg.cleaning.close_defocus_um,
                 100 * cnt["today_a"] / n, reach, 100 * cnt["today_b_only"] / n, pct(b_defocus)))
        print("  nodes with an ATTACHED neighbour (same stretch beyond the chain, parent, sibling, daughter) inside "
              "rule (a): %.1f %%" % (100 * cnt["att_a"] / n))
    print("\n%.0f s" % (time.time() - t0))
