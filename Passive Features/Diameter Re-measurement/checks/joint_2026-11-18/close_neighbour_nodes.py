#!/usr/bin/env python3
"""Standalone check (NOT pipeline code), 2026-11-18: which nodes of a cell have a TRACED neighbour on their
measuring line, close enough to matter in the joint fit (plan step F3), by geometry alone.

For every dendrite node not near an end (node_inputs.is_near_end), in the node's frame -- u along its heading in the
image plane (the total-least-squares line through the traced nodes within +-line_fit_window_um/2 of arc length),
v across it, both from the node -- a neighbour segment (any segment not of the node's own chain,
node_inputs.own_ids_within) is ON THE LINE when a point of it has |u| <= U_BAND_UM (the band the profile averages
along the branch, along_branch_avg_um, plus a pixel) and |v| <= close_lateral_um. Reported per node: the on-line
neighbour nearest in depth -- v where it passes (the smallest |v| inside the band), its depth minus the node's
there, whether it lies OVER the node (|v| <= the two radii) or BESIDE it, and whether it is ATTACHED (on the node's
own stretch beyond the own chain, or on a stretch sharing a branch point with it) or FREE.

Writes a CSV of the nodes whose nearest on-line neighbour is within MAX_DEPTH_UM in depth: node_id, swc_type,
allen_d_um, tier2_today (cleaning.neighbour_tiers), nb_child_id, nb_type, nb_allen_d_um, v_um, depth_um (neighbour
minus node), over (0/1), attached (0/1), n_chains (stretches, or the axon, on the line within MAX_DEPTH_UM).
Prints the counts, and the same description for the node ids given on the command line at any depth.

Run:  python3 close_neighbour_nodes.py "<repo>/Passive Features/Diameter Re-measurement/src" <file.swc> <out.csv> [node ids ...]
"""
import csv
import sys

import numpy as np

sys.path.insert(0, sys.argv[1])
from allen_diameter.config import default_config                               # noqa: E402
from allen_diameter.loading import swc_io                                       # noqa: E402
from allen_diameter.analysis import cell, cleaning, node_features, node_inputs  # noqa: E402

MAX_DEPTH_UM = 4.2
N_SAMPLES = 25          # points per neighbour segment (median length 1.15 um: one every 0.05 um)

if __name__ == "__main__":
    cfg = default_config()
    swc = swc_io.read_swc(sys.argv[2])
    ask = [int(x) for x in sys.argv[4:]]
    res0 = float(cfg.acquisition.res0_um)
    U_BAND_UM = float(cfg.measure.along_branch_avg_um) + res0
    close = float(cfg.cleaning.close_lateral_um)
    W = 0.5 * float(cfg.measure.line_fit_window_um)
    xyz = cell.to_image_um(swc.xyz, res0)
    ids = np.asarray(swc.ids)
    typ = dict(zip(ids.tolist(), np.asarray(swc.types).tolist()))
    runs = cell.stretches(swc, tuple(cfg.acquisition.dendrite_swc_types))
    stretch_of = {int(ids[r]): si for si, run in enumerate(runs) for r in run}
    types = tuple(cfg.cleaning.neighbour_swc_types)
    fscene = node_features.featured_scene(cell.cell_scene(swc, types, res0, None),
                                          node_features.initial_features(swc, types, 0.5))
    seg_ids = np.asarray(fscene.seg_ids).astype(np.int64)
    seg_xyz = np.asarray(fscene.seg_xyz, dtype=float)
    seg_d = np.asarray(fscene.seg_d_um, dtype=float).mean(axis=1)
    seg_stretch = np.array([stretch_of.get(int(c), -1) for c in seg_ids[:, 0]])
    pidx = swc.parent_index()
    parent_of = {int(ids[k]): (int(ids[pidx[k]]) if pidx[k] >= 0 else -1) for k in range(len(ids))}
    first_parent = np.array([parent_of[int(ids[run[0]])] for run in runs])
    last_node = np.array([int(ids[run[-1]]) for run in runs])
    a, b_ = seg_xyz[:, :3], seg_xyz[:, 3:]
    mid = 0.5 * (a + b_)
    half = 0.5 * np.linalg.norm(b_[:, :2] - a[:, :2], axis=1)
    lam = np.linspace(0.0, 1.0, N_SAMPLES)[None, :, None]
    out, asked, n_nodes = [], {}, 0
    for si, run in enumerate(runs):
        ob = node_inputs.own_branch(swc, run, xyz)
        if ob is None:
            continue
        br = ob[0]
        fp_own = first_parent[si]
        att_st = set(np.flatnonzero((first_parent == fp_own) | (first_parent == last_node[si])).tolist()) | {si}
        if fp_own in stretch_of:
            att_st.add(stretch_of[fp_own])
        attached = np.isin(seg_stretch, np.asarray(sorted(att_st)))
        P, s = br.xyz_um, br.s_um
        for i in range(len(br.ids)):
            nid = int(br.ids[i])
            near_end = node_inputs.is_near_end(br, i, cfg)
            if near_end and nid not in ask:
                continue
            n_nodes += 0 if near_end else 1
            own, um, up = node_inputs.own_ids_within(br, i, cfg.inference.tube_half_length_um)
            node = P[i]
            win = np.abs(s - s[i]) <= W
            Q = P[win, :2] - P[win, :2].mean(axis=0)
            h = np.linalg.svd(Q, full_matrices=False)[2][0]
            nv = np.array([-h[1], h[0]])
            own_arr = np.asarray(sorted(own))
            nb = ~(np.isin(seg_ids[:, 0], own_arr) & np.isin(seg_ids[:, 1], own_arr))
            cand = np.flatnonzero(nb & (np.hypot(*(mid[:, :2] - node[:2]).T) <= half + close + U_BAND_UM))
            r_own = float(np.asarray(br.radius_um)[i])
            hits = []                                     # (|depth|, q, v, depth)
            if cand.size:
                pts = a[cand][:, None, :] + lam * (b_[cand] - a[cand])[:, None, :]      # (n, N_SAMPLES, 3)
                du = (pts[:, :, :2] - node[:2]) @ h
                dv = (pts[:, :, :2] - node[:2]) @ nv
                inside = (np.abs(du) <= U_BAND_UM) & (np.abs(dv) <= close)
                for j in np.flatnonzero(inside.any(axis=1)):
                    m = np.flatnonzero(inside[j])
                    k = m[np.argmin(np.abs(dv[j, m]))]
                    hits.append((abs(float(pts[j, k, 2] - node[2])), int(cand[j]), float(dv[j, k]),
                                 float(pts[j, k, 2] - node[2])))
            if not hits:
                if nid in ask:
                    asked[nid] = "node %d (Allen d %.3f um%s): no traced neighbour on the line at any depth" % (
                        nid, 2 * r_own, ", near an end" if near_end else "")
                continue
            hits.sort()
            _, q, v, dzq = hits[0]
            today = -1 if near_end else int(np.any(cleaning.neighbour_tiers(fscene, own, node, cfg) == 2))
            inl = [hq for (ad, hq, _, _) in hits if ad <= MAX_DEPTH_UM]
            n_ch = len(set(seg_stretch[inl].tolist()) - {-1}) + int(np.any(seg_stretch[inl] == -1)) if inl else 0
            over = int(abs(v) <= r_own + 0.5 * seg_d[q])
            row = (nid, typ[nid], round(2 * r_own, 3), today, int(seg_ids[q, 0]), typ[int(seg_ids[q, 0])],
                   round(float(seg_d[q]), 3), round(v, 2), round(dzq, 2), over, int(attached[q]), n_ch)
            if abs(dzq) <= MAX_DEPTH_UM and not near_end:
                out.append(row)
            if nid in ask:
                asked[nid] = ("node %d (Allen d %.3f um%s): the on-line neighbour nearest in depth is a segment of node %d "
                              "(type %d, Allen d %.2f um, %s), passing at v %+.2f um (%s), %+.2f um in depth; tier 2 today: %s"
                              % (nid, 2 * r_own, ", near an end" if near_end else "", row[4], row[5], row[6],
                                 "attached" if row[10] else "free", row[7], "over the node" if over else "beside it",
                                 row[8], {1: "yes", 0: "no", -1: "n/a"}[today]))
    out.sort(key=lambda r: abs(r[8]))
    with open(sys.argv[3], "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["node_id", "swc_type", "allen_d_um", "tier2_today", "nb_child_id", "nb_type", "nb_allen_d_um",
                    "v_um", "depth_um", "over", "attached", "n_chains"])
        w.writerows(out)
    d = np.array([abs(r[8]) for r in out])
    att = np.array([r[10] for r in out], dtype=bool)
    ovr = np.array([r[9] for r in out], dtype=bool)
    nch = np.array([r[11] for r in out])
    print("%d dendrite nodes not near an end; band |u| <= %.2f um, |v| <= %.1f um" % (n_nodes, U_BAND_UM, close))
    print("nodes with a traced neighbour on the line, by the depth distance of the nearest one:")
    for z in (0.84, 1.68, 2.8, 4.2):
        m = d <= z
        print("  within %.2f um: %d nodes (%.1f %%): free %d, attached %d; over the node %d, beside it %d"
              % (z, int(m.sum()), 100.0 * m.sum() / n_nodes, int((m & ~att).sum()), int((m & att).sum()),
                 int((m & ovr).sum()), int((m & ~ovr).sum())))
    print("  chains on the line within %.1f um in depth, per such node: 1: %d, 2: %d, 3 or more: %d"
          % (MAX_DEPTH_UM, int((nch == 1).sum()), int((nch == 2).sum()), int((nch >= 3).sum())))
    print("  type of the nearest one (2 axon, 3 basal, 4 apical): %s; tier 2 under today's rule: %d of %d"
          % ({k: int(sum(1 for r in out if r[5] == k)) for k in (2, 3, 4)}, int(sum(r[3] == 1 for r in out)), len(out)))
    for nid in ask:
        print(asked.get(nid, "node %d: not a dendrite node on a stretch" % nid))
