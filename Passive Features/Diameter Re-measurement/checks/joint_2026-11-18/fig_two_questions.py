#!/usr/bin/env python3
"""Standalone figure script (NOT pipeline code), 2026-11-18, for two open choices of the joint fit (D-177.5, D-177.10):
  fig_attached_vs_free.png   a chain attached to the node's own stretch and a free chain, each on a node's measuring
                             line (nodes 8089 and 7081 of cell 529878215), drawn in the node's frame
  fig_own_chain_options.png  the three candidate axes of a node's own tube at a node where the trace kinks
                             (node 2423): the straight tube, the chain through the raw trace, a smoothed chain
Plotted from the cell's SWC (Allen's trace, um) with the pipeline's cell.stretches and node_inputs.own_ids_within;
nothing is read from the image. The node's frame: origin at the traced node, u along the total-least-squares line
through the traced nodes within +-line_fit_window_um/2 of arc length (image plane), v across it. The smoothed chain
is ONE EXAMPLE of a smoother (not a decided one): at each arc length s0 the weighted least-squares line of x and of
y against s, Gaussian weights of SD SMOOTH_SD_UM in arc length (numpy.polyfit), evaluated at s0.

Run:  python3 fig_two_questions.py "<repo>/Passive Features/Diameter Re-measurement/src" <reconstruction.raw.swc> <out dir>
"""
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                             # noqa: E402
import numpy as np                                                          # noqa: E402
from matplotlib.lines import Line2D                                         # noqa: E402
from matplotlib.patches import Polygon, Rectangle                           # noqa: E402

sys.path.insert(0, sys.argv[1])
from allen_diameter.config import default_config                            # noqa: E402
from allen_diameter.loading import swc_io                                    # noqa: E402
from allen_diameter.analysis import cell, node_inputs                        # noqa: E402

# ---- config (one block) ----
NODE_ATTACHED, NB_ATTACHED_CHILD = 8089, 8231      # node and a node of the attached chain on its line
NODE_FREE, NB_FREE_CHILD = 5724, 7085              # node and a node of the free chain on its line
NODE_KINK = 2423
SMOOTH_SD_UM = 1.5
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"   # validated: dataviz validate_palette.js, light surface
SURFACE, INK, INK2, GREY = "#fcfcfb", "#1f1f1e", "#5f5e58", "#dddcd5"
LINE_HALF_UM = 3.0                                  # measure.profile_half_um

plt.rcParams.update({"font.size": 9.5, "font.family": "DejaVu Sans", "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.linewidth": 0.6,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def tube(ax, a, b, d, color, alpha, z):
    t = b - a
    n = np.array([-t[1], t[0]]) / max(np.hypot(*t), 1e-12) * 0.5 * d
    ax.add_patch(Polygon([a + n, b + n, b - n, a - n], closed=True, facecolor=color, edgecolor="none", alpha=alpha,
                         zorder=z))


def frame(P, s, i, W):
    win = np.abs(s - s[i]) <= W
    Q = P[win, :2] - P[win, :2].mean(axis=0)
    h = np.linalg.svd(Q, full_matrices=False)[2][0]
    if h @ (P[min(i + 1, len(P) - 1), :2] - P[max(i - 1, 0), :2]) < 0:
        h = -h
    return h, np.array([-h[1], h[0]])


def to_uv(xy, o, h, n):
    D = np.asarray(xy, dtype=float)[..., :2] - o[:2]
    return np.stack([D @ h, D @ n], axis=-1)


def measuring_line(ax, band):
    ax.add_patch(Rectangle((-band, -2.0), 2 * band, 4.0, facecolor="none", edgecolor=INK2, linewidth=0.8,
                           linestyle=(0, (3, 2)), zorder=5))
    ax.plot([0, 0], [-LINE_HALF_UM, LINE_HALF_UM], color=INK, linewidth=1.6, zorder=6)


if __name__ == "__main__":
    cfg = default_config()
    swc = swc_io.read_swc(sys.argv[2])
    out = sys.argv[3]
    os.makedirs(out, exist_ok=True)
    res0 = float(cfg.acquisition.res0_um)
    W = 0.5 * float(cfg.measure.line_fit_window_um)
    band = float(cfg.measure.along_branch_avg_um) + res0
    xyz = cell.to_image_um(swc.xyz, res0)
    ids = np.asarray(swc.ids)
    row = {int(v): k for k, v in enumerate(ids)}
    rad = np.asarray(swc.radius, dtype=float)
    pidx = swc.parent_index()
    runs = cell.stretches(swc, tuple(cfg.acquisition.dendrite_swc_types))
    where = {int(ids[r]): (si, k) for si, run in enumerate(runs) for k, r in enumerate(run)}

    def node_setup(nid):
        si, k = where[nid]
        br = node_inputs.own_branch(swc, runs[si], xyz)[0]
        own, um, up = node_inputs.own_ids_within(br, k, cfg.inference.tube_half_length_um)
        h, n = frame(br.xyz_um, br.s_um, k, W)
        return si, k, br, set(own), h, n

    # ------------------------------------------------------------ figure 1 ---
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.0), sharex="col", gridspec_kw=dict(height_ratios=[1.55, 1.0]))
    for col, (nid, nbc) in enumerate(((NODE_ATTACHED, NB_ATTACHED_CHILD), (NODE_FREE, NB_FREE_CHILD))):
        ax, az = axes[0, col], axes[1, col]
        si, k, br, own, h, n = node_setup(nid)
        o = br.xyz_um[k]
        sj = where[nbc][0]
        sibling = pidx[runs[sj][0]] == pidx[runs[si][0]]
        for c in np.flatnonzero((pidx >= 0)):
            p = pidx[c]
            a, b = to_uv(xyz[p], o, h, n), to_uv(xyz[c], o, h, n)
            if max(abs(a[0]), abs(b[0])) > 9.5 or max(abs(a[1]), abs(b[1])) > 6.5:
                continue
            cid, pid_ = int(ids[c]), int(ids[p])
            d = 2.0 * rad[c]
            za, zb = xyz[p, 2] - o[2], xyz[c, 2] - o[2]
            if cid in own and pid_ in own:
                tube(ax, a, b, d, BLUE, 0.85, 3)
                tube(az, np.array([a[0], za]), np.array([b[0], zb]), d, BLUE, 0.85, 3)
            elif where.get(cid, (-1,))[0] == sj:
                tube(ax, a, b, d, ORANGE, 0.85, 4)
                tube(az, np.array([a[0], za]), np.array([b[0], zb]), d, ORANGE, 0.85, 4)
            else:
                tube(ax, a, b, d, GREY, 0.9, 2)
        measuring_line(ax, band)
        ax.plot(0, 0, "o", color=INK, markersize=4.5, zorder=7)
        az.plot(0, 0, "o", color=INK, markersize=4.5, zorder=7)
        az.axvspan(-band, band, facecolor="#ecebe6", edgecolor="none", zorder=0)
        # the other chain's passage: of its points inside the window, the one nearest the node in depth
        run = runs[sj]
        best = None
        for r0, r1 in zip(run[:-1], run[1:]):
            pts = xyz[r0] + np.linspace(0, 1, 40)[:, None] * (xyz[r1] - xyz[r0])
            uv = to_uv(pts, o, h, n)
            m = (np.abs(uv[:, 0]) <= band) & (np.abs(uv[:, 1]) <= 2.0)
            if m.any():
                j = np.flatnonzero(m)[np.argmin(np.abs(pts[m, 2] - o[2]))]
                if best is None or abs(pts[j, 2] - o[2]) < abs(best[2]):
                    best = (uv[j, 0], uv[j, 1], pts[j, 2] - o[2])
        ax.annotate("on the line at v = %+.2f um" % best[1], xy=(best[0], best[1]), xytext=(2.4, -4.5), fontsize=9,
                    color=INK, ha="left", arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.7))
        az.annotate("$\\Delta z$ = %+.2f um there" % best[2], xy=(best[0], best[2]), xytext=(2.4, -3.6 if best[2] > -2.6 else 1.2),
                    fontsize=9, color=INK, ha="left", arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.7))
        if sibling:
            bp = xyz[pidx[run[0]]]
            buv = to_uv(bp, o, h, n)
            ax.plot(buv[0], buv[1], "s", markerfacecolor=SURFACE, markeredgecolor=INK, markersize=7, zorder=8)
            ax.annotate("branch point shared by\nthe two stretches", xy=buv, xytext=(buv[0] - 0.2, buv[1] - 3.0), fontsize=9,
                        ha="center", arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.7))
            az.plot(buv[0], bp[2] - o[2], "s", markerfacecolor=SURFACE, markeredgecolor=INK, markersize=7, zorder=8)
        kind = ("ATTACHED: the other chain is a sibling branch, leaving the same branch point" if sibling
                else "FREE: the other chain is a stretch that shares no branch point with the node's")
        ax.set_title("node %d  (Allen's diameter %.2f um)\n%s\ntop view (image plane)" % (nid, 2 * rad[row[nid]], kind),
                     fontsize=9.5, loc="left", color=INK)
        az.set_title("side view of the same two chains (depth against u)", fontsize=9.5, loc="left", color=INK)
        ax.set_xlim(-9, 9)
        ax.set_ylim(-5.2, 5.2)
        az.set_ylim(-4.6, 2.6)
        for a_ in (ax, az):
            a_.set_aspect("equal")
            for sp in ("top", "right"):
                a_.spines[sp].set_visible(False)
        az.set_xlabel("u, along the node's heading (um)")
    axes[0, 0].set_ylabel("v, along the measuring line (um)")
    axes[1, 0].set_ylabel("$\\Delta z$, stage z minus the node's (um)")
    handles = [Line2D([], [], color=BLUE, linewidth=6, alpha=0.85, label="node's own chain (+-5 um of trace)"),
               Line2D([], [], color=ORANGE, linewidth=6, alpha=0.85, label="the other chain (its whole stretch)"),
               Line2D([], [], color=GREY, linewidth=6, label="rest of the traced tree (top view only)"),
               Line2D([], [], color=INK, linewidth=1.6, label="measuring line, v in [-3, 3] um"),
               Line2D([], [], color=INK2, linewidth=0.8, linestyle=(0, (3, 2)),
                      label="'on the line' window: |u| <= %.2f um, |v| <= 2 um" % band)]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.06))
    fig.suptitle("Attached or free, the other chain lies on the node's measuring line in the same way",
                 fontsize=11.5, x=0.01, ha="left", color=INK)
    fig.text(0.01, 0.01, "Plotted from Allen's trace of cell 529878215 (reconstruction.raw.swc) in each node's frame; widths = "
             "Allen's diameters; $\\Delta z$ in stage z (the trace's), no claim on which sign is 'deeper'.\n"
             "Trace geometry only: nothing here is read from the image, so untraced structures are absent and the trace's "
             "own errors are included.  [data inspected: the SWC]", fontsize=8, color=INK2, ha="left", va="bottom")
    fig.subplots_adjust(left=0.06, right=0.99, top=0.86, bottom=0.2, wspace=0.08, hspace=0.22)
    fig.savefig(os.path.join(out, "fig_attached_vs_free.png"), dpi=170)
    plt.close(fig)

    # ------------------------------------------------------------ figure 2 ---
    si, k, br, own, h, n = node_setup(NODE_KINK)
    P, s = br.xyz_um, br.s_um
    o = P[k]
    sel = np.flatnonzero([int(i) in own for i in br.ids])
    uv = to_uv(P, o, h, n)
    a_, c_ = uv[k] - uv[k - 1], uv[k + 1] - uv[k]
    turn = math.degrees(math.acos(np.clip(a_ @ c_ / (np.linalg.norm(a_) * np.linalg.norm(c_)), -1, 1)))
    grid = np.linspace(s[sel[0]], s[sel[-1]], 400)
    sm = np.empty((grid.size, 2))
    for j, s0 in enumerate(grid):
        w = np.exp(-0.5 * ((s - s0) / SMOOTH_SD_UM) ** 2)
        for c in (0, 1):
            sm[j, c] = np.polyfit(s - s0, uv[:, c], 1, w=np.sqrt(w))[1]
    d_allen = 2.0 * float(br.radius_um[k])
    fig, axes = plt.subplots(2, 1, figsize=(11.6, 7.4), gridspec_kw=dict(height_ratios=[1.0, 1.55]))
    for ax, (ulim, vlim, zoom) in zip(axes, (((-5.6, 5.6), (-1.5, 1.5), False), ((-1.75, 1.75), (-0.62, 0.62), True))):
        lw = 2.4 if zoom else 1.8
        ax.add_patch(Rectangle((-band, vlim[0]), 2 * band, vlim[1] - vlim[0], facecolor="#ecebe6", edgecolor="none",
                               zorder=0))
        ax.plot([0, 0], vlim, color=INK, linewidth=1.4, zorder=2)
        ax.plot([uv[sel[0], 0], uv[sel[-1], 0]], [0, 0], color=BLUE, linewidth=lw, zorder=3)
        ax.plot(uv[sel, 0], uv[sel, 1], color=ORANGE, linewidth=lw, zorder=4)
        ax.plot(sm[:, 0], sm[:, 1], color=AQUA, linewidth=lw, zorder=5)
        ax.plot(uv[sel, 0], uv[sel, 1], "o", markerfacecolor=SURFACE, markeredgecolor=INK, markersize=6.5 if zoom else 5,
                markeredgewidth=1.0, zorder=6)
        ax.plot(0, 0, "o", color=INK, markersize=6.5 if zoom else 5, zorder=7)
        ax.set_xlim(*ulim)
        ax.set_ylim(*vlim)
        ax.set_aspect("equal")
        ax.set_ylabel("v (um)")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_title("the whole own chain (+-5 um of trace)", fontsize=9.5, loc="left")
    axes[1].set_title("zoom on the measuring line: the three axes differ most exactly where the profile is read",
                      fontsize=9.5, loc="left")
    axes[1].set_xlabel("u, along the straight tube's heading (um)")
    ax = axes[1]
    ax.annotate("the trace turns by %.0f deg\nat the node" % turn, xy=(0.03, -0.01), xytext=(0.75, -0.25), fontsize=9,
                va="center", arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.7))
    ax.text(band + 0.05, -0.59, "shaded: the band the profile averages\nalong the branch, |u| <= %.2f um" % band,
            fontsize=8.5, color=INK2, va="bottom")
    ax.plot([-1.6, -1.6 + d_allen], [-0.45, -0.45], color=INK, linewidth=1.2)
    ax.text(-1.6 + 0.5 * d_allen, -0.42, "Allen's diameter\n%.2f um, to scale" % d_allen, fontsize=8.5, ha="center",
            va="bottom")
    handles = [Line2D([], [], marker="o", linestyle="none", markerfacecolor=SURFACE, markeredgecolor=INK, markersize=6,
                      label="traced nodes (filled: the working node)"),
               Line2D([], [], color=INK, linewidth=1.4, label="measuring line (v)"),
               Line2D([], [], color=BLUE, linewidth=2.4, label="straight tube (today): one line through the node"),
               Line2D([], [], color=ORANGE, linewidth=2.4, label="chain through the raw trace (D-164.1)"),
               Line2D([], [], color=AQUA, linewidth=2.4,
                      label="chain through the smoothed trace (example smoother, SD %.1f um)" % SMOOTH_SD_UM)]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.085))
    fig.suptitle("Node %d: three candidate axes for the node's own tube -- they part at the node, where the trace kinks"
                 % NODE_KINK, fontsize=11.5, x=0.01, ha="left", color=INK)
    fig.text(0.01, 0.01, "Plotted from Allen's trace of cell 529878215 (reconstruction.raw.swc), top view in the node's "
             "frame, equal scales. The straight tube's heading is drawn from the trace\n(the line through the traced "
             "nodes within +-2 um); in the pipeline it comes from the first stage's fit to the image. In the fit each axis is "
             "also shifted rigidly\nacross (the free lateral shift $v_0$), so it is their SHAPE on the line that differs, "
             "not their offset. The smoother is an example, not a decided one.\nWhich axis is the dendrite's, the trace "
             "cannot say.  [data inspected: the SWC]", fontsize=8, color=INK2, ha="left", va="bottom")
    fig.subplots_adjust(left=0.06, right=0.99, top=0.9, bottom=0.22, hspace=0.3)
    fig.savefig(os.path.join(out, "fig_own_chain_options.png"), dpi=170)
    print("node %d: turn %.1f deg in the image plane; raw chain's |v| at |u| = %.2f um: %.3f, %.3f um; smoothed chain at "
          "u = 0: v = %.3f um" % (NODE_KINK, turn, band, abs(np.interp(-band, uv[sel, 0], uv[sel, 1])),
                                  abs(np.interp(band, uv[sel, 0], uv[sel, 1])),
                                  float(np.interp(0.0, sm[:, 0], sm[:, 1]))))
