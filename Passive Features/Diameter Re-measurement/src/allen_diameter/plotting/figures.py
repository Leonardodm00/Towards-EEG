"""Figures -- plotting only, nothing scientific is computed here (Block 11 in
specs/SPEC.md).

Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import math

import numpy as np


_SCORE_LABELS = {"gradient_energy": "gradient energy G (1/um)", "dip_depth": "dip depth F"}


def node_figure(result, v, I, model, title="", k_swc=None):
    """Two panels for one measured node. Left: the focus curves of the final
    pass over the node's planes -- the configured rule (D-030; k* is its
    maximum, red) and, on a second axis, the dip depth of handoff Eq. 1 with
    the plane it would have chosen (grey) -- and the SWC's own plane k_swc
    (dotted) when given. Right: the fitted profile in plane k* with the model
    and the background B_bar. Returns the matplotlib Figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.4))
    ks = np.asarray(result.focus_planes, dtype=float)
    S = np.asarray(result.focus_score, dtype=float)
    Dp = np.asarray(result.focus_depth, dtype=float)
    if S.size:
        a1.plot(ks, S, "o-", color="tab:red", ms=4, label=_SCORE_LABELS.get(result.focus_rule, result.focus_rule))
        if math.isfinite(result.z_sub_um):
            a1.axvline(result.k_star, color="tab:red", lw=0.9)
        if result.focus_rule != "dip_depth":
            a1b = a1.twinx()
            a1b.plot(ks, Dp, "s--", color="0.55", ms=3, lw=0.9, label=_SCORE_LABELS["dip_depth"])
            a1b.set_ylabel("dip depth F", color="0.45")
            if result.k_star_depth != result.k_star and np.any(np.isfinite(Dp)):
                a1b.axvline(result.k_star_depth, color="0.55", lw=0.9, ls="--")
            h1, l1 = a1.get_legend_handles_labels()
            h2, l2 = a1b.get_legend_handles_labels()
            a1.legend(h1 + h2, l1 + l2, fontsize=7, frameon=False, loc="upper left")
    if k_swc is not None:
        a1.axvline(k_swc, color="k", lw=0.8, ls=":")
    a1.set_xlabel("plane k (red: k*; grey: dip-depth choice; dotted: SWC)")
    a1.set_ylabel(_SCORE_LABELS.get(result.focus_rule, result.focus_rule))
    a2.plot(v, I, ".", color="0.2", ms=4, label="profile")
    if model is not None:
        a2.plot(v, model, "-", color="tab:blue", lw=1.2, label="model")
    if math.isfinite(result.B_bar):
        a2.axhline(result.B_bar, color="0.6", lw=0.8, ls="--", label="B_bar")
    a2.set_xlabel("v (um)")
    a2.set_ylabel("grey level")
    a2.legend(fontsize=7, frameon=False)
    fig.suptitle(title or "node %d: d_hat %.3f um, mu_hat %.2f /um, phi %.1f deg, %s"
                 % (result.node_id, result.d_hat_um, result.mu_hat_per_um, math.degrees(result.phi_rad),
                    result.fit_status), fontsize=9)
    fig.tight_layout()
    return fig


def image_extent_um(frame, shape):
    """(x_left, x_right, y_bottom, y_top) in um for imshow(origin="upper") of an
    array of `shape` (..., rows, cols) cut in `frame`: pixel (row, col) is
    centred on ((left + col) p, (top + row) p), p = frame.res_um_px, the
    convention of analysis.profiles.sample_profile. Downsample-0 frames only
    (as fetch_zblock returns): left and top are full-resolution pixels, so for a
    downsampled crop this formula would not hold, and it is refused."""
    if int(getattr(frame, "downsample", 0)) != 0:
        raise ValueError("image_extent_um needs a downsample-0 frame (got downsample %d)" % frame.downsample)
    p = float(frame.res_um_px)
    H, W = int(shape[-2]), int(shape[-1])
    return ((frame.left - 0.5) * p, (frame.left + W - 0.5) * p, (frame.top + H - 0.5) * p, (frame.top - 0.5) * p)




_OWN, _OTHER, _FIT = "#00e5ff", "#ffb300", "#ff1744"


def depth_alpha(z0, z1, z_plane, solid_um, fade_um, floor=0.2):
    """Opacity of a segment from depth z0 to z1 (um) drawn on the plane at depth
    z_plane (um): 1 while the plane is within solid_um of the segment's depth
    range, then falling linearly to `floor` at fade_um and beyond. Presentation
    only: no measurement reads it."""
    lo, hi = min(z0, z1), max(z0, z1)
    gap = 0.0 if lo <= z_plane <= hi else min(abs(z_plane - lo), abs(z_plane - hi))
    if gap <= solid_um:
        return 1.0
    return max(floor, 1.0 - (1.0 - floor) * (gap - solid_um) / max(fade_um - solid_um, 1e-12))


def _sides(seg):
    """The two side lines, shape (2, 2, 2), of a segment's frustum silhouette in
    xy: (x0, y0) +- r0 n to (x1, y1) +- r1 n, n the unit xy normal of the
    segment (a tilted frustum projects to this width, whatever its tilt); None
    for a segment with no xy extent."""
    a, b = seg[0:2], seg[3:5]
    L = math.hypot(b[0] - a[0], b[1] - a[1])
    if L <= 1e-9:
        return None
    n = np.array([a[1] - b[1], b[0] - a[0]]) / L
    return np.array([[a + seg[6] * n, b + seg[7] * n], [a - seg[6] * n, b - seg[7] * n]])


def plane_montage(planes, dz_um, profile_half_um, view_half_um=None, ncol=6, depth_solid_um=1.0,
                  depth_fade_um=3.0, title=""):
    """Every plane one node's focus rule scored (its focus_planes), each with
    Allen's reconstruction drawn on it. `planes` is the dict returned by
    analysis.survey.node_planes; nothing is computed here but the drawing.

    In every panel, in um of the image frame (image_extent_um): Allen's traced
    centre lines, cyan for the measured stretch and amber for the other
    dendrites in the block, each with its frustum silhouette dashed (+-r0 at the
    parent end, +-r1 at the child end), fainter the farther the segment's depth
    is from the plane's (depth_alpha); the SWC node (black circle) and the
    final fit's profile line (black, through (cx, cy) along y_hat over
    +-profile_half_um). In plane k*: the fitted edges at v0_hat +- d_hat / 2
    along that line (red ticks). Frames: k* red; the dip depth's plane, when it
    is another, grey dashed. One grey scale for all panels (1st to 99.5th
    percentile of the shown windows), so that a change of focus shows as a
    change in the picture and not in the scaling. Each panel is a square of
    half-width view_half_um (um) about the SWC node; None: the largest such square
    inside the block, so that no panel shows ground outside the image. Returns
    the Figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import to_rgba
    from matplotlib import patheffects
    from matplotlib.patches import Circle
    halo = [patheffects.Stroke(linewidth=2.6, foreground="white"), patheffects.Normal()]   # visible on dark and light
    res = planes["result"]
    block, frame = planes["block"], planes["frame"]
    ks, valid = np.asarray(planes["ks"], dtype=int), np.asarray(planes["valid"], dtype=bool)
    seg = np.asarray(planes["segments"], dtype=float).reshape(-1, 9)
    focus_k = [int(k) for k in np.asarray(res.focus_planes)]
    show = focus_k or [int(k) for k in ks]
    ext = image_extent_um(frame, block.shape)
    p = float(frame.res_um_px)
    xc, yc = float(res.x_um), float(res.y_um)
    h = float(view_half_um) if view_half_um is not None else \
        max(min(xc - ext[0], ext[1] - xc, yc - ext[3], ext[2] - yc), 2.0 * p)
    imgs = {k: np.asarray(block[k - ks[0]], dtype=float) for k in show
            if 0 <= k - ks[0] < len(ks) and valid[k - ks[0]]}
    c0, r0 = max(0, int(math.floor((xc - h) / p - frame.left))), max(0, int(math.floor((yc - h) / p - frame.top)))
    c1, r1 = int(math.ceil((xc + h) / p - frame.left)) + 1, int(math.ceil((yc + h) / p - frame.top)) + 1
    pix = np.concatenate([im[r0:r1, c0:c1].ravel() for im in imgs.values()]) if imgs else np.empty(0)
    lo, hi = np.percentile(pix, [1.0, 99.5]) if pix.size else (0.0, 255.0)
    if not hi > lo:
        lo, hi = lo - 1.0, hi + 1.0
    th = float(res.theta_rad)
    y_hat, e_u = np.array([-math.sin(th), math.cos(th)]), np.array([math.cos(th), math.sin(th)])
    centre = np.array([float(res.cx_um), float(res.cy_um)])
    score = dict(zip(focus_k, np.asarray(res.focus_score, dtype=float)))
    depth = dict(zip(focus_k, np.asarray(res.focus_depth, dtype=float)))
    has_k = math.isfinite(res.z_sub_um)
    k_swc = int(planes["k_swc"])
    n = len(show)
    nc = max(1, min(int(ncol), n))
    nr = int(math.ceil(n / float(nc)))
    # fixed layout (inches): square panels of side s, a title band t above each row, gaps g, the
    # figure title in the top band; tight_layout does not reserve the titles of square (aspect-equal) axes
    side, band, gap, top_band, bottom = 2.0, 0.42, 0.12, 0.8, 0.08
    grid_w = nc * side + (nc - 1) * gap
    W, H = max(grid_w + 2 * gap, 8.0), top_band + nr * (side + band) + bottom
    fig, axes = plt.subplots(nr, nc, figsize=(W, H), squeeze=False)
    x_left = 0.5 * (W - grid_w)
    fig.subplots_adjust(left=x_left / W, right=(x_left + grid_w) / W, bottom=bottom / H,
                        top=1.0 - (top_band + band) / H, wspace=gap / side, hspace=band / side)
    own = seg[:, 8] > 0.5
    centre_lines = seg[:, [0, 1, 3, 4]].reshape(-1, 2, 2)
    sides = [_sides(row) for row in seg]
    for a, k in zip(axes.ravel(), show):
        a.set_facecolor("0.8")                       # outside the block: no image
        if k in imgs:
            a.imshow(imgs[k], cmap="gray", vmin=lo, vmax=hi, extent=ext, origin="upper", interpolation="nearest")
        else:
            a.text(0.5, 0.5, "missing", transform=a.transAxes, ha="center", va="center", fontsize=8)
        z_k = k * float(dz_um)
        rgba = [to_rgba(_OWN if o else _OTHER, depth_alpha(row[2], row[5], z_k, depth_solid_um, depth_fade_um))
                for row, o in zip(seg, own)]
        if len(seg):
            a.add_collection(LineCollection(centre_lines, colors=rgba, linewidths=np.where(own, 1.1, 0.8)))
            lines = [ln for sd in sides if sd is not None for ln in sd]
            cols = [c for sd, c in zip(sides, rgba) if sd is not None for _ in range(2)]
            if lines:
                a.add_collection(LineCollection(lines, colors=cols, linewidths=0.6, linestyles="--"))
            for row, sd, c in zip(seg, sides, rgba):
                if sd is None:                       # a segment along z: its cross-section
                    a.add_patch(Circle((row[3], row[4]), max(row[6], row[7]), fill=False, ec=c, lw=0.6, ls="--"))
        a.plot(*np.column_stack([centre - profile_half_um * y_hat, centre + profile_half_um * y_hat]),
               "-", color="black", lw=0.7, path_effects=halo)
        a.plot([xc], [yc], "o", mfc="none", mec="black", ms=5, mew=1.0, path_effects=halo)
        tags = [name for name, hit in (("k*", has_k and k == int(res.k_star)),
                                       ("dip", has_k and k == int(res.k_star_depth)), ("SWC", k == k_swc)) if hit]
        if "k*" in tags:
            if res.fit is not None and math.isfinite(res.d_hat_um):
                for e in (res.v0_hat_um - 0.5 * res.d_hat_um, res.v0_hat_um + 0.5 * res.d_hat_um):
                    q = centre + e * y_hat
                    a.plot([q[0] - 0.4 * e_u[0], q[0] + 0.4 * e_u[0]], [q[1] - 0.4 * e_u[1], q[1] + 0.4 * e_u[1]],
                           "-", color=_FIT, lw=1.6)
            for sp in a.spines.values():
                sp.set_edgecolor(_FIT)
                sp.set_linewidth(2.5)
        elif "dip" in tags:
            for sp in a.spines.values():
                sp.set_edgecolor("0.45")
                sp.set_linewidth(2.0)
                sp.set_linestyle("--")
        line2 = ("F %.2f" % depth.get(k, float("nan"))) if res.focus_rule == "dip_depth" else \
            ("G %.3f  F %.2f" % (score.get(k, float("nan")), depth.get(k, float("nan"))))
        tag = ("  " + " ".join(tags)) if tags else ""
        a.set_title("k %d (%+.2f um)%s\n%s" % (k, z_k - float(res.z_um), tag, line2), fontsize=7.5,
                    color=_FIT if "k*" in tags else "black")
        a.set_xlim(xc - h, xc + h)
        a.set_ylim(yc + h, yc - h)                   # image rows run down
        a.set_aspect("equal")
        a.set_xticks([])
        a.set_yticks([])
    for a in axes.ravel()[n:]:
        a.axis("off")
    a0 = axes.ravel()[0]
    bar = 2.0 if h >= 2.5 else 1.0                   # scale bar (um), inside the panel
    x0, y0 = xc - h + 0.15 * h, yc + h - 0.15 * h
    a0.plot([x0, x0 + bar], [y0, y0], "-", color="black", lw=2.0, path_effects=halo)
    a0.text(x0 + 0.5 * bar, y0 - 0.06 * h, "%g um" % bar, color="black", ha="center", va="bottom", fontsize=7,
            path_effects=[patheffects.withStroke(linewidth=2.0, foreground="white")])
    head = title or ("node %d: k* %s (%s), dip-depth plane %s, SWC plane %d | d_hat %.2f um, Allen 2r %.2f um, "
                     "phi %.1f deg, %s" % (res.node_id, res.k_star if has_k else "none",
                                           res.focus_rule.replace("_", " "), res.k_star_depth if has_k else "none",
                                           k_swc, res.d_hat_um, 2.0 * planes["allen_radius_um"],
                                           math.degrees(res.phi_rad), res.fit_status))
    fig.suptitle(head + "\npanels: plane k (its depth minus the node's), then G and F of the focus curves; frame red: "
                 "k*, grey dashed: the dip depth's plane\ncyan: Allen's trace of the measured stretch, dashed +-r (SWC "
                 "frustum); amber: other traced dendrites; fainter: farther in depth from the plane\nblack: the SWC "
                 "node and the fit's profile line; red ticks: the fitted edges (v0_hat +- d_hat/2)",
                 fontsize=7.5, y=1.0 - 0.06 / H, va="top")
    return fig


_SPOS, _SNEG = "#2a78d6", "#7a7a7a"     # S+ blue (the proposal's measure), S- grey dashed


def plane_difference_figure(blocks, title=""):
    """Consecutive-plane differences (focus.plane_differences), one block of
    three rows per stack: the planes (one grey scale), the positive part of
    D_n = I_{k+1} - I_k between each pair (one blue scale from 0), and the
    curves S+ (the mean of that positive part over the pixels) and S- (the
    mean of the negative part: S+ taken from the other end of the stack) at the
    pair midpoints, aligned under the pairs.

    blocks: list of dicts with
      label   heading of the block
      stack   (n, H, W) the planes differenced (grey levels)
      ks      (n,) consecutive plane indices
      valid   (n,) bool
      res     the dict of focus.plane_differences(stack, valid, ...)
      dip     focus.difference_dip(res["pos"]), or None
      frames  {k: (colour, linestyle, tag)}: a frame on plane k's panel, the short
              tag in its title
      lines   [(k, colour, linestyle, label)]: a vertical line on the curve; the
              legend names each style once, for the frame and the line
      extent  imshow extent of a plane in um (image_extent_um), or None
    Presentation parameters only; nothing scientific is computed. Returns the Figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    side, gap, left, right = 0.95, 0.08, 0.75, 0.25
    col = side + gap
    n_max = max(len(b["ks"]) for b in blocks)
    W = left + n_max * col - gap + right
    heights = (0.34, 0.32, side, 0.24, side, 0.30, 1.55, 0.50)
    top_band = 0.95
    H = top_band + len(blocks) * sum(heights)
    fig = plt.figure(figsize=(W, H))

    def ax_at(x_in, y_top_in, w_in, h_in):
        return fig.add_axes([x_in / W, 1.0 - (y_top_in + h_in) / H, w_in / W, h_in / H])

    lines_seen = {}
    y = top_band
    for b in blocks:
        ks = np.asarray(b["ks"])
        n = ks.size
        stack = np.asarray(b["stack"], dtype=float)
        valid = np.asarray(b["valid"], dtype=bool)
        res = b["res"]
        fig.text(left / W, 1.0 - (y + 0.26) / H, b["label"], fontsize=10, fontweight="bold", ha="left")
        y += heights[0]
        shown = stack[valid] if valid.any() else stack
        lo, hi = np.percentile(shown, [1.0, 99.5])
        for j in range(n):
            ax = ax_at(left + j * col, y + heights[1], side, side)
            ax.set_xticks([])
            ax.set_yticks([])
            k = int(ks[j])
            if valid[j]:
                ax.imshow(stack[j], cmap="gray", vmin=lo, vmax=hi, extent=b.get("extent"), origin="upper",
                          interpolation="nearest")
            else:
                ax.set_facecolor("0.85")
                ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=8, transform=ax.transAxes)
            tag = ""
            if k in b.get("frames", {}):
                colour, ls, tag = b["frames"][k]
                ax.add_patch(plt.Rectangle((0.01, 0.01), 0.98, 0.98, transform=ax.transAxes, fill=False,
                                           edgecolor=colour, lw=2.6, ls=ls, clip_on=False))
            ax.set_title("k %d%s" % (k, ("\n" + tag) if tag else ""), fontsize=7.5, pad=2)
        y += heights[1] + heights[2]
        dpos = np.clip(res["diff"], 0.0, None)
        vmax = np.nanpercentile(dpos, 99.5) if np.isfinite(dpos).any() else 1.0
        vmax = vmax if vmax > 0 else 1.0
        for j in range(n - 1):
            ax = ax_at(left + j * col + col / 2.0, y + heights[3], side, side)
            ax.set_xticks([])
            ax.set_yticks([])
            if np.isfinite(res["pos"][j]):
                ax.imshow(dpos[j], cmap="Blues", vmin=0.0, vmax=vmax, extent=b.get("extent"), origin="upper",
                          interpolation="nearest")
            else:
                ax.set_facecolor("0.85")
                ax.text(0.5, 0.5, "n/a", ha="center", va="center", fontsize=8, transform=ax.transAxes)
            ax.set_title("%d->%d" % (int(ks[j]), int(ks[j + 1])), fontsize=7.5, pad=2)
        y += heights[3] + heights[4] + heights[5]
        ax = ax_at(left - gap / 2.0, y, n * col, heights[6])
        mid = ks[:-1] + 0.5
        for k, colour, ls, label in b.get("lines", []):
            ax.axvline(k, color=colour, ls=ls, lw=1.2, zorder=1)
            lines_seen[label] = (colour, ls)
        ax.plot(mid, res["pos"], "o-", color=_SPOS, ms=4.5, lw=2.0, zorder=3)
        ax.plot(mid, res["neg"], "s--", color=_SNEG, ms=3.5, lw=1.3, zorder=2)
        if b.get("dip") is not None:
            d = int(b["dip"])
            ax.plot([mid[d]], [res["pos"][d]], "o", ms=12, mfc=_SPOS, mec="white", mew=1.5, zorder=4)
            ax.annotate("dip %d->%d" % (int(ks[d]), int(ks[d + 1])), (mid[d], res["pos"][d]), textcoords="offset points",
                        xytext=(0, -16), ha="center", fontsize=8)
        ax.set_xlim(ks[0] - 0.5, ks[-1] + 0.5)
        ax.set_xticks(ks)
        ax.tick_params(labelsize=7.5)
        ax.grid(axis="y", color="#e6e5e0", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_ylabel("grey levels\nper pixel", fontsize=8.5)
        ax.set_xlabel("plane k (the curve's points sit between the two planes they compare)", fontsize=8.5)
        y += heights[6] + heights[7]
    handles = [plt.Line2D([], [], color=_SPOS, marker="o", lw=2.0, label="S+  mean over the pixels of max(I_k+1 - I_k, 0)"),
               plt.Line2D([], [], color=_SNEG, marker="s", ls="--", lw=1.3,
                          label="S-  mean of max(I_k - I_k+1, 0): S+ taken from the other end of the stack"),
               plt.Line2D([], [], color=_SPOS, marker="o", ms=10, ls="none", mec="white",
                          label="dip between the two largest maxima of S+")]
    handles += [plt.Line2D([], [], color=c, ls=ls, lw=2.0, label="frame and line: " + t)
                for t, (c, ls) in lines_seen.items()]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, 1.0 - 0.02 / H))
    if title:
        fig.suptitle(title, y=1.0 - 0.66 / H, fontsize=10)
    return fig
