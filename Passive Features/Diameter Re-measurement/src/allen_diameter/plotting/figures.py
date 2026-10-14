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


_ARAW, _ANORM = "#2a78d6", "#4a3aa7"     # area under the profile: as measured (blue), background-normalised (violet)


def _no_line_note(plt, what):
    """A text-only legend entry for planes drawn with show_line=False: the
    square is cut about the SWC node (plane_differences.node_stack), so the
    node sits at the centre of every panel, to within half a pixel."""
    return plt.Line2D([], [], color="none", label="planes drawn without %s: the node is at the centre of each" % what)


def _plane_colour(dk):
    """Blue below the reference plane, red above, darker nearer; dark at it; light grey beyond 4 planes."""
    blue, red = ("#1c5cab", "#2a78d6", "#5598e7", "#86b6ef"), ("#9e3432", "#c74845", "#dd716a", "#ea9a93")
    if dk == 0:
        return "#1a1a19"
    if abs(dk) > 4:
        return "#d6d6d1"
    return blue[abs(dk) - 1] if dk < 0 else red[dk - 1]


def profile_area_figure(blocks, title="", show_line=True):
    """The area under the profile along the measuring line, plane by plane
    (focus.profile_areas). Per block (one node): the planes on one grey scale
    with the measuring line drawn and frames; below, the profiles I_k(v)
    coloured by plane (blue below the reference plane, red above), the areas
    A_k as measured and background-normalised on one axis with their minima,
    and their differences between consecutive planes. show_line=False draws the
    planes without the line (the frames stay), so that the focus can be judged
    by eye (the user's request of 2026-10-09).

    blocks: list of dicts with
      label, stack (n, H, W), ks (n,), valid (n,), extent (image_extent_um), line ((x0, y0), (x1, y1)) in um
      (not read when show_line is False), frames {k: (colour, linestyle, tag)}, lines [(k, colour, linestyle,
      label)], v (M,), prof (n, M), area (n,), area_norm (n,) or None, background (n,) or None (each plane's
      background level, drawn in a fourth panel), k_ref (the colour reference plane)
    Presentation only. Returns the Figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    halo = [pe.Stroke(linewidth=3.0, foreground="white"), pe.Normal()]
    side, gap, left, right = 0.95, 0.08, 0.75, 0.25
    col = side + gap
    n_max = max(len(b["ks"]) for b in blocks)
    W = max(left + n_max * col - gap + right, 13.0)
    heights = (0.34, 0.34, side, 0.62, 2.3, 0.62)
    top_band = 1.25
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
        fig.text(left / W, 1.0 - (y + 0.26) / H, b["label"], fontsize=10, fontweight="bold", ha="left")
        y += heights[0]
        shown = stack[valid] if valid.any() else stack
        lo, hi = np.percentile(shown, [1.0, 99.5])
        if show_line:
            (x0, y0), (x1, y1) = b["line"]
        for j in range(n):
            ax = ax_at(left + j * col, y + heights[1], side, side)
            ax.set_xticks([])
            ax.set_yticks([])
            k = int(ks[j])
            if valid[j]:
                ax.imshow(stack[j], cmap="gray", vmin=lo, vmax=hi, extent=b.get("extent"), origin="upper",
                          interpolation="nearest")
                if show_line:
                    ax.plot([x0, x1], [y0, y1], color="black", lw=1.0, path_effects=halo)
            else:
                ax.set_facecolor("0.85")
                ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=8, transform=ax.transAxes)
            tag = ""
            if k in b.get("frames", {}):
                colour, ls, tag = b["frames"][k]
                ax.add_patch(plt.Rectangle((0.01, 0.01), 0.98, 0.98, transform=ax.transAxes, fill=False,
                                           edgecolor=colour, lw=2.6, ls=ls, clip_on=False))
            ax.set_title("k %d%s" % (k, ("\n" + tag) if tag else ""), fontsize=7.5, pad=2)
        y += heights[1] + heights[2] + heights[3]
        npan = 4 if b.get("background") is not None else 3
        w3 = (W - left - right - (npan - 1) * 0.7) / float(npan)
        a1 = ax_at(left, y, w3, heights[4])
        a2 = ax_at(left + w3 + 0.7, y, w3, heights[4])
        a3 = ax_at(left + 2 * (w3 + 0.7), y, w3, heights[4])
        a4 = ax_at(left + 3 * (w3 + 0.7), y, w3, heights[4]) if npan == 4 else None
        v, prof = np.asarray(b["v"]), np.asarray(b["prof"], dtype=float)
        order = sorted(range(n), key=lambda j: -abs(int(ks[j]) - int(b["k_ref"])))
        for j in order:
            if valid[j]:
                dk = int(ks[j]) - int(b["k_ref"])
                a1.plot(v, prof[j], color=_plane_colour(dk), lw=2.2 if dk == 0 else 1.1)
        a1.set_xlim(v[0], v[-1])
        a1.set_title("profiles along the measuring line\n(dark: plane %d; blue below, red above; darker = nearer)"
                     % int(b["k_ref"]), fontsize=8.5)
        a1.set_xlabel("v (um)", fontsize=8.5)
        a1.set_ylabel("grey level", fontsize=8.5)
        mid = ks[:-1] + 0.5
        for ax in (a2, a3):
            for k, colour, ls, label in b.get("lines", []):
                ax.axvline(k, color=colour, ls=ls, lw=1.1, zorder=1)
                lines_seen[label] = (colour, ls)
        curves = [(np.asarray(b["area"], dtype=float), _ARAW, "o-")]
        if b.get("area_norm") is not None:
            curves.append((np.asarray(b["area_norm"], dtype=float), _ANORM, "s--"))
        for A, colour, style in curves:
            a2.plot(ks, A, style, color=colour, ms=4.5, lw=1.8, zorder=3)
            if np.isfinite(A).any():
                j = int(np.nanargmin(A))
                a2.plot([ks[j]], [A[j]], "o", ms=11, mfc=colour, mec="white", mew=1.5, zorder=4)
            a3.plot(mid, np.diff(A), style, color=colour, ms=4.5, lw=1.8, zorder=3)
        a3.axhline(0.0, color="#1a1a19", lw=0.8)
        if a4 is not None:
            for k, colour, ls, label in b.get("lines", []):
                a4.axvline(k, color=colour, ls=ls, lw=1.1, zorder=1)
            bk = np.asarray(b["background"], dtype=float)
            a4.plot(ks, bk, "D-", color="#7a7a7a", ms=4, lw=1.5, zorder=3)
            if not np.isfinite(bk).any():
                a4.text(0.5, 0.5, "no pixel far from the traced dendrites", ha="center", va="center", fontsize=8,
                        transform=a4.transAxes)
            a4.set_title("B_k: each plane's background level\n(a step here is a change of the whole plane's brightness)",
                         fontsize=8.5)
            a4.set_xlabel("plane k", fontsize=8.5)
            a4.set_ylabel("grey level", fontsize=8.5)
            a4.set_xlim(ks[0] - 0.5, ks[-1] + 0.5)
        a2.set_title("A_k: area under the profile (minimum marked)", fontsize=8.5)
        a3.set_title("A_(k+1) - A_k between consecutive planes", fontsize=8.5)
        a2.set_xlabel("plane k", fontsize=8.5)
        a3.set_xlabel("pair midpoint (k + 0.5)", fontsize=8.5)
        a2.set_ylabel("grey level x um", fontsize=8.5)
        for ax in (a1, a2, a3) + ((a4,) if a4 is not None else ()):
            ax.tick_params(labelsize=7.5)
            ax.grid(axis="y", color="#e6e5e0", lw=0.6)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        for ax in (a2, a3):
            ax.set_xlim(ks[0] - 0.5, ks[-1] + 0.5)
        y += heights[4] + heights[5]
    handles = [plt.Line2D([], [], color=_ARAW, marker="o", lw=1.8, label="A_k as measured (the proposal); blue frame: its minimum"),
               plt.Line2D([], [], color=_ANORM, marker="s", ls="--", lw=1.8,
                          label="A_k with each plane divided by its own background (x their mean); violet frame: its minimum")]
    handles += [plt.Line2D([], [], color=c, ls=ls, lw=2.0, label="frame and line: " + t)
                for t, (c, ls) in lines_seen.items() if c not in (_ARAW, _ANORM)]
    if not show_line:
        handles.append(_no_line_note(plt, "the measuring line"))
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.0 - 0.02 / H))
    if title:
        fig.suptitle(title, y=1.0 - 0.80 / H, fontsize=10)
    return fig


def entropy_summary_figure(records, title=""):
    """Small multiples of the entropy evaluation over several nodes: one panel per node, the line's and the
    strip's entropy against the plane offset k - k_SWC, each rescaled to 0 (its minimum) .. 1 (its maximum) so
    that both share the axis, with k* (red), the dip depth's plane (grey dashed), the SWC plane (dotted) and
    the planes the two curves picked (large markers at 0).

    records: the per-node dicts of scripts/plane_differences.run (entropy evaluation): node_id, node_type,
    allen_radius_um, d_hat_um, ks, h_line, h_strip, k_swc, k_star, k_star_depth, k_h_line, k_h_strip.
    Presentation only. Returns the Figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(records)
    ncol = max(1, min(4, n))
    nrow = int(math.ceil(n / float(ncol)))
    H = 2.75 * nrow + 1.7
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.7 * ncol, H), squeeze=False)
    names = {3: "basal", 4: "apical"}

    def unit(h):
        h = np.asarray(h, dtype=float)
        if not np.isfinite(h).any():
            return h
        lo, hi = np.nanmin(h), np.nanmax(h)
        return (h - lo) / (hi - lo) if hi > lo else np.zeros_like(h)

    for i, r in enumerate(records):
        ax = axes[i // ncol, i % ncol]
        off = np.asarray(r["ks"]) - int(r["k_swc"])
        for kk, colour, ls in ((r.get("k_star"), "#ff1744", "-"), (r.get("k_star_depth"), "#7a7a7a", "--"),
                               (r.get("k_swc"), "#1a1a19", ":")):
            if kk is not None:
                ax.axvline(int(kk) - int(r["k_swc"]), color=colour, ls=ls, lw=1.2, zorder=1)
        for key, pick, colour, style in (("h_line", "k_h_line", _HLINE, "o-"), ("h_strip", "k_h_strip", _HSTRIP, "s--")):
            mec = _EDGE if colour == _HSTRIP else colour
            ax.plot(off, unit(r[key]), style, color=colour, mec=mec, mew=0.7, ms=3.5, lw=1.5, zorder=3)
            if r.get(pick) is not None:
                ax.plot([int(r[pick]) - int(r["k_swc"])], [0.0], style[0], ms=10, mfc=colour,
                        mec=_EDGE if colour == _HSTRIP else "white", mew=1.3, zorder=4)
        dl = None if r.get("k_h_line") is None else int(r["k_h_line"]) - int(r["k_star"])
        dsx = None if r.get("k_h_strip") is None else int(r["k_h_strip"]) - int(r["k_star"])
        fmt = lambda x: "n/a" if x is None else "%+d" % x  # noqa: E731
        ax.set_title("node %d (%s), Allen 2r %.2f um, d %.2f um\npick minus k*: line %s, strip %s"
                     % (int(r["node_id"]), names.get(r.get("node_type"), "type %s" % r.get("node_type")),
                        2.0 * float(r.get("allen_radius_um", float("nan"))), float(r.get("d_hat_um", float("nan"))),
                        fmt(dl), fmt(dsx)), fontsize=8.5)
        ax.set_ylim(-0.12, 1.12)
        ax.set_xlim(off.min() - 0.5, off.max() + 0.5)
        ax.tick_params(labelsize=7.5)
        ax.grid(axis="y", color="#e6e5e0", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if i % ncol == 0:
            ax.set_ylabel("H, rescaled 0-1", fontsize=8.5)
        if i // ncol == nrow - 1 or i + ncol >= n:
            ax.set_xlabel("plane k - k_SWC", fontsize=8.5)
    for j in range(n, nrow * ncol):
        axes[j // ncol, j % ncol].axis("off")
    handles = [plt.Line2D([], [], color=_HLINE, marker="o", lw=1.5, label="H along the measuring line"),
               plt.Line2D([], [], color=_HSTRIP, marker="s", mec=_EDGE, ls="--", lw=1.5, label="H of the strip's pixels"),
               plt.Line2D([], [], color=_HLINE, marker="o", ms=9, ls="none", mec="white", label="the plane each curve picks"),
               plt.Line2D([], [], color="#ff1744", lw=1.2, label="k* (gradient energy, D-030)"),
               plt.Line2D([], [], color="#7a7a7a", ls="--", lw=1.2, label="plane of the dip depth"),
               plt.Line2D([], [], color="#1a1a19", ls=":", lw=1.2, label="the SWC node's plane")]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.995))
    if title:
        fig.suptitle(title, y=1.0 - 0.72 / H, fontsize=10)
    fig.subplots_adjust(top=1.0 - 1.55 / H, bottom=0.55 / H, left=0.06, right=0.99, hspace=0.75, wspace=0.18)
    return fig


# entropy along the measuring line (blue) and in its strip (amber: light on white, so its markers carry a dark edge
# and its line is dashed; blue, amber and the red of k* pass the dataviz palette checks, all pairs, light surface)
_HLINE, _HSTRIP, _EDGE = "#2a78d6", "#eda100", "#1a1a19"


def entropy_figure(blocks, title="", show_line=True):
    """Histogram entropies around a node, plane by plane
    (focus.plane_entropies). Per block (one node): the planes on one grey
    scale with the measuring line, the outline of its strip and frames;
    below, the grey-level histograms of the line's samples and of the strip's
    pixels, coloured by plane (blue below the reference plane, red above,
    darker nearer), and the two entropies on one axis with their dips.
    show_line=False draws the planes without the line and the strip's outline
    (the frames stay), so that the focus can be judged by eye (the user's
    request of 2026-10-09); the legend says so.

    blocks: list of dicts with
      label, stack (n, H, W), ks (n,), valid (n,), extent (image_extent_um), line ((x0, y0), (x1, y1)) in um,
      outline (q, 2) the strip's corners in um, closed (line and outline not read when show_line is False),
      frames {k: (colour, linestyle, tag)},
      lines [(k, colour, linestyle, label)], hist_line and hist_strip: per plane (centres, counts) or None,
      h_line (n,), h_strip (n,) in bits, dip_line and dip_strip (the picked plane of each curve: an index into
      ks, or None), pick_word (optional: what the pick is, e.g. "lowest entropy"), n_line and n_strip (the
      samples behind a finite entropy), k_ref (the colour reference plane)
    Presentation only. Returns the Figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    halo = [pe.Stroke(linewidth=3.0, foreground="white"), pe.Normal()]
    side, gap, left, right = 0.95, 0.08, 0.75, 0.25
    col = side + gap
    n_max = max(len(b["ks"]) for b in blocks)
    W = max(left + n_max * col - gap + right, 13.0)
    heights = (0.34, 0.34, side, 0.62, 2.3, 0.62)
    top_band = 1.25
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
        fig.text(left / W, 1.0 - (y + 0.26) / H, b["label"], fontsize=10, fontweight="bold", ha="left")
        y += heights[0]
        shown = stack[valid] if valid.any() else stack
        lo, hi = np.percentile(shown, [1.0, 99.5])
        if show_line:
            (x0, y0), (x1, y1) = b["line"]
            out = np.asarray(b["outline"], dtype=float)
        for j in range(n):
            ax = ax_at(left + j * col, y + heights[1], side, side)
            ax.set_xticks([])
            ax.set_yticks([])
            k = int(ks[j])
            if valid[j]:
                ax.imshow(stack[j], cmap="gray", vmin=lo, vmax=hi, extent=b.get("extent"), origin="upper",
                          interpolation="nearest")
                if show_line:
                    ax.plot(out[:, 0], out[:, 1], color=_HSTRIP, lw=1.0, ls="--", path_effects=halo)
                    ax.plot([x0, x1], [y0, y1], color="black", lw=1.0, path_effects=halo)
            else:
                ax.set_facecolor("0.85")
                ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=8, transform=ax.transAxes)
            tag = ""
            if k in b.get("frames", {}):
                colour, ls, tag = b["frames"][k]
                ax.add_patch(plt.Rectangle((0.01, 0.01), 0.98, 0.98, transform=ax.transAxes, fill=False,
                                           edgecolor=colour, lw=2.6, ls=ls, clip_on=False))
            ax.set_title("k %d%s" % (k, ("\n" + tag) if tag else ""), fontsize=7.5, pad=2)
        y += heights[1] + heights[2] + heights[3]
        w3 = (W - left - right - 2 * 0.7) / 3.0
        a1 = ax_at(left, y, w3, heights[4])
        a2 = ax_at(left + w3 + 0.7, y, w3, heights[4])
        a3 = ax_at(left + 2 * (w3 + 0.7), y, w3, heights[4])
        order = sorted(range(n), key=lambda j: -abs(int(ks[j]) - int(b["k_ref"])))
        for ax, key, what in ((a1, "hist_line", "the %d samples along the measuring line" % int(b["n_line"])),
                              (a2, "hist_strip", "the %d pixels of the strip" % int(b["n_strip"]))):
            hists = b.get(key) or [None] * n
            for j in order:
                if valid[j] and hists[j] is not None and len(hists[j][1]):
                    c, cnt = np.asarray(hists[j][0], dtype=float), np.asarray(hists[j][1], dtype=float)
                    dk = int(ks[j]) - int(b["k_ref"])
                    ax.step(c, cnt / cnt.sum(), where="mid", color=_plane_colour(dk), lw=2.0 if dk == 0 else 1.0)
            ax.set_title("grey-level histogram of %s\n(dark: plane %d; blue below, red above; darker = nearer)"
                         % (what, int(b["k_ref"])), fontsize=8.5)
            ax.set_xlabel("grey level", fontsize=8.5)
            ax.set_ylabel("fraction per grey level", fontsize=8.5)
        for k, colour, ls, label in b.get("lines", []):
            a3.axvline(k, color=colour, ls=ls, lw=1.1, zorder=1)
            lines_seen[label] = (colour, ls)
        for key, dkey, colour, style in (("h_line", "dip_line", _HLINE, "o-"), ("h_strip", "dip_strip", _HSTRIP, "s--")):
            h = np.asarray(b[key], dtype=float)
            a3.plot(ks, h, style, color=colour, ms=4.5, lw=1.8, mec=_EDGE if colour == _HSTRIP else colour, mew=0.8,
                    zorder=3)
            if b.get(dkey) is not None:
                d = int(b[dkey])
                a3.plot([ks[d]], [h[d]], style[0], ms=11, mfc=colour, mec=_EDGE if colour == _HSTRIP else "white",
                        mew=1.5, zorder=4)
        a3.set_title("entropy of each histogram\n(marked: %s)" % b.get("pick_word", "the picked plane"), fontsize=8.5)
        a3.set_xlabel("plane k", fontsize=8.5)
        a3.set_ylabel("H (bits)", fontsize=8.5)
        a3.set_xlim(ks[0] - 0.5, ks[-1] + 0.5)
        for ax in (a1, a2, a3):
            ax.tick_params(labelsize=7.5)
            ax.grid(axis="y", color="#e6e5e0", lw=0.6)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        y += heights[4] + heights[5]
    pick_word = blocks[0].get("pick_word", "the picked plane") if blocks else "the picked plane"
    handles = [plt.Line2D([], [], color=_HLINE, marker="o", lw=1.8,
                          label="H along the measuring line (bilinear samples); blue frame Hl: %s" % pick_word),
               plt.Line2D([], [], color=_HSTRIP, marker="s", mec=_EDGE, ls="--", lw=1.8,
                          label="H of the strip's pixels%s; amber frame Hs: %s"
                          % (" (outlined on the planes)" if show_line else "", pick_word))]
    handles += [plt.Line2D([], [], color=c, ls=ls, lw=2.0, label="frame and line: " + t)
                for t, (c, ls) in lines_seen.items() if c not in (_HLINE, _HSTRIP)]
    if not show_line:
        handles.append(_no_line_note(plt, "the measuring line and the strip's outline"))
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.0 - 0.02 / H))
    if title:
        fig.suptitle(title, y=1.0 - 0.80 / H, fontsize=10)
    return fig


# ---- the gradient energy along lines of three lengths, and the blend with the strip entropy (D-040) ----
# colours (dataviz validator, light surface, all pairs with the red of k*: hard checks pass; red-aqua deutan 7.5 sits
# in the 6-8 band, so every series also has its own marker, line style and frame tag): G on fixed lines blue, violet
# (green for a third), G on the d-line aqua, the blend green, the strip entropy yellow; aqua and yellow markers
# carry a dark edge (below 3:1 on white)
_GD, _BLEND = "#1baf7a", "#008300"
_LIGHT = ("#1baf7a", "#eda100")


def _planes_row(plt, ax_at, b, y_top, left, col, side):
    """The planes of block b on one grey scale (1st to 99.5th percentile of the
    valid planes), drawn without any line, with b's frames and the plane index
    (and the frame's tags) as titles."""
    ks = np.asarray(b["ks"])
    stack = np.asarray(b["stack"], dtype=float)
    valid = np.asarray(b["valid"], dtype=bool)
    shown = stack[valid] if valid.any() else stack
    lo, hi = np.percentile(shown, [1.0, 99.5])
    for j in range(ks.size):
        ax = ax_at(left + j * col, y_top, side, side)
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


def _profiles_panel(ax, v, prof, ks, valid, k_ref, spans, title):
    """Every valid plane's profile along the line, coloured by plane (blue below
    k_ref, red above, darker nearer, dark at k_ref), and dashed verticals at
    +-half for each (half, colour, linestyle) of spans."""
    v, prof = np.asarray(v, dtype=float), np.asarray(prof, dtype=float)
    order = sorted(range(len(ks)), key=lambda j: -abs(int(ks[j]) - int(k_ref)))
    for j in order:
        if valid[j] and np.isfinite(prof[j]).all():
            dk = int(ks[j]) - int(k_ref)
            ax.plot(v, prof[j], color=_plane_colour(dk), lw=2.2 if dk == 0 else 1.1)
    for half, colour, ls in spans:
        for s in (-1.0, 1.0):
            ax.axvline(s * float(half), color=colour, ls=ls, lw=1.4, zorder=1)
    ax.set_xlim(v[0], v[-1])
    ax.set_title(title, fontsize=8.5)
    ax.set_xlabel("v (um)", fontsize=8.5)
    ax.set_ylabel("grey level", fontsize=8.5)


def _style_axes(axes):
    for ax in axes:
        ax.tick_params(labelsize=7.5)
        ax.grid(axis="y", color="#e6e5e0", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)


def _bare_layout(plt, blocks, top_band=1.25):
    side, gap, left, right = 0.95, 0.08, 0.75, 0.25
    col = side + gap
    n_max = max(len(b["ks"]) for b in blocks)
    W = max(left + n_max * col - gap + right, 13.0)
    heights = (0.62, 0.34, side, 0.62, 2.3, 0.62)            # the label row holds two lines (_block_label)
    top_band = float(top_band)                                # the legend's rows and the suptitle (0.45 in above)
    H = top_band + len(blocks) * sum(heights)
    fig = plt.figure(figsize=(W, H))

    def ax_at(x_in, y_top_in, w_in, h_in):
        return fig.add_axes([x_in / W, 1.0 - (y_top_in + h_in) / H, w_in / W, h_in / H])
    return fig, ax_at, dict(side=side, gap=gap, left=left, right=right, col=col, W=W, H=H, heights=heights,
                            top_band=top_band)


def _block_label(fig, L, y, label):
    """A block's label, bold, wrapped to the figure's width so that it never runs off the right edge: two lines at
    10 pt, or three at 8.5 pt when two do not hold it (the label row is 0.62 in high)."""
    import textwrap
    usable = L["W"] - L["left"] - L["right"]
    size, lines = 10.0, textwrap.wrap(label, max(40, int(usable * 10.5)))
    if len(lines) > 2:
        size, lines = 8.5, textwrap.wrap(label, max(40, int(usable * 12.5)))
    fig.text(L["left"] / L["W"], 1.0 - (y + 0.06) / L["H"], "\n".join(lines), fontsize=size, fontweight="bold",
             ha="left", va="top", linespacing=1.15)


def _pipeline_lines(ax, b, exclude_colours):
    for k, colour, ls, label in b.get("lines", []):
        if colour not in exclude_colours:
            ax.axvline(k, color=colour, ls=ls, lw=1.1, zorder=1)


def _pipeline_handles(plt, blocks, exclude_colours):
    seen = {}
    for b in blocks:
        for k, colour, ls, label in b.get("lines", []):
            if colour not in exclude_colours:
                seen[label] = (colour, ls)
    return [plt.Line2D([], [], color=c, ls=ls, lw=2.0, label="frame and line: " + t) for t, (c, ls) in seen.items()]


def gradient_lines_figure(blocks, title=""):
    """The gradient energy over the whole line, for lines of several lengths
    (D-040; focus.plane_gradient_energies). Per block (one node): the planes on
    one grey scale, without the lines, framed at each line's pick and at the
    pipeline's planes; below, every plane's profile along the longest line
    with each line's extent dashed, each line's G relative to its own maximum
    with its pick, and each plane's background B_k.

    blocks: list of dicts with
      label, stack (n, H, W), ks (n,), valid (n,), extent, frames {k: (colour, linestyle, tag)},
      lines [(k, colour, linestyle, label)] (the variants' picks and the pipeline's planes),
      variants [dict(key, label, colour, style, half_um, G (n,), pick: an index into ks or None)],
      v (M,), prof (n, M) of the longest line, background (n,), k_ref
    Presentation only. Returns the Figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax_at, L = _bare_layout(plt, blocks)
    y = L["top_band"]
    vcols = set()
    for b in blocks:
        ks = np.asarray(b["ks"])
        valid = np.asarray(b["valid"], dtype=bool)
        _block_label(fig, L, y, b["label"])
        y += L["heights"][0]
        _planes_row(plt, ax_at, b, y + L["heights"][1], L["left"], L["col"], L["side"])
        y += L["heights"][1] + L["heights"][2] + L["heights"][3]
        w3 = (L["W"] - L["left"] - L["right"] - 2 * 0.7) / 3.0
        a1 = ax_at(L["left"], y, w3, L["heights"][4])
        a2 = ax_at(L["left"] + w3 + 0.7, y, w3, L["heights"][4])
        a3 = ax_at(L["left"] + 2 * (w3 + 0.7), y, w3, L["heights"][4])
        spans = [(x["half_um"], x["colour"], "--") for x in b["variants"]]
        _profiles_panel(a1, b["v"], b["prof"], ks, valid, b["k_ref"], spans,
                        "profiles along the longest line, each line's extent dashed\n(dark: plane %d; blue below, red "
                        "above; darker = nearer)" % int(b["k_ref"]))
        vcols |= {x["colour"] for x in b["variants"]}
        _pipeline_lines(a2, b, vcols)
        for x in b["variants"]:
            G = np.asarray(x["G"], dtype=float)
            rel = G / np.nanmax(G) if np.isfinite(G).any() and np.nanmax(G) > 0 else G
            edge = "#1a1a19" if x["colour"] in _LIGHT else x["colour"]
            a2.plot(ks, rel, x["style"], color=x["colour"], mec=edge, mew=0.8, ms=4.5, lw=1.8, zorder=3)
            if x.get("pick") is not None:
                d = int(x["pick"])
                a2.plot([ks[d]], [rel[d]], x["style"][0], ms=11, mfc=x["colour"],
                        mec="#1a1a19" if x["colour"] in _LIGHT else "white", mew=1.5, zorder=4)
        a2.set_title("G of each line relative to its own maximum\n(marked: each line's maximum)", fontsize=8.5)
        a2.set_xlabel("plane k", fontsize=8.5)
        a2.set_ylabel("G / max G", fontsize=8.5)
        a2.set_xlim(ks[0] - 0.5, ks[-1] + 0.5)
        _pipeline_lines(a3, b, vcols)
        bk = np.asarray(b["background"], dtype=float)
        a3.plot(ks, bk, "D-", color="#7a7a7a", ms=4, lw=1.5, zorder=3)
        if not np.isfinite(bk).any():
            a3.text(0.5, 0.5, "no pixel far from the traced dendrites", ha="center", va="center", fontsize=8,
                    transform=a3.transAxes)
        a3.set_title("B_k: each plane's background, the same for every line\n(far from every traced dendrite)",
                     fontsize=8.5)
        a3.set_xlabel("plane k", fontsize=8.5)
        a3.set_ylabel("grey level", fontsize=8.5)
        a3.set_xlim(ks[0] - 0.5, ks[-1] + 0.5)
        _style_axes((a1, a2, a3))
        y += L["heights"][4] + L["heights"][5]
    handles, seen = [], set()
    for b in blocks:
        for x in b["variants"]:
            if x["key"] not in seen:
                seen.add(x["key"])
                handles.append(plt.Line2D([], [], color=x["colour"], marker=x["style"][0], ls=x["style"][1:] or "-",
                                          mec="#1a1a19" if x["colour"] in _LIGHT else x["colour"], lw=1.8,
                                          label="%s; frame %s" % (x["label"], x["key"])))
    handles += _pipeline_handles(plt, blocks, vcols)
    handles.append(_no_line_note(plt, "any line"))
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, 1.0 - 0.02 / L["H"]))
    if title:
        fig.suptitle(title, y=1.0 - 0.80 / L["H"], fontsize=10)
    return fig


def blend_figure(blocks, title=""):
    """G on the line sized from d_hat, the entropy of a wider strip, and their
    min-max blend J = w g - (1 - w) h (D-040, D-041; focus.blend_scores). Per
    block (one node): the planes on one grey scale, without any line, framed
    at the three picks and at the pipeline's planes; below, every plane's
    profile along the strip's line with the d-line's extent dashed; g (1 at
    the largest G) and h (0 at the lowest entropy, drawn as it is, not
    inverted) with their picks; and J with its pick and the weight w, on the
    axis -(1 - w) .. w that J spans.

    blocks: list of dicts with
      label, stack, ks, valid, extent, frames, lines (as gradient_lines_figure), v (M,), prof (n, M) along the
      strip's line of half-length half_um, d_half_um (the d-line's), g (n,), h (n,), J (n,), w, d_hat_um, d0_um,
      s_um, pick_g, pick_h, pick_J (indices into ks or None), k_ref
    Presentation only. Returns the Figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax_at, L = _bare_layout(plt, blocks)
    y = L["top_band"]
    mine = {_GD, _BLEND, "#eda100"}
    for b in blocks:
        ks = np.asarray(b["ks"])
        valid = np.asarray(b["valid"], dtype=bool)
        _block_label(fig, L, y, b["label"])
        y += L["heights"][0]
        _planes_row(plt, ax_at, b, y + L["heights"][1], L["left"], L["col"], L["side"])
        y += L["heights"][1] + L["heights"][2] + L["heights"][3]
        w3 = (L["W"] - L["left"] - L["right"] - 2 * 0.7) / 3.0
        a1 = ax_at(L["left"], y, w3, L["heights"][4])
        a2 = ax_at(L["left"] + w3 + 0.7, y, w3, L["heights"][4])
        a3 = ax_at(L["left"] + 2 * (w3 + 0.7), y, w3, L["heights"][4])
        _profiles_panel(a1, b["v"], b["prof"], ks, valid, b["k_ref"], [(b["d_half_um"], _GD, "--")],
                        "profiles along the +-%.1f um line (the entropy strip's width);\nthe d-line +-%.2f um dashed "
                        "(dark: plane %d; blue below, red above)" % (float(b["half_um"]), float(b["d_half_um"]),
                                                                    int(b["k_ref"])))
        w = float(b["w"])
        for ax, curves, lims in ((a2, (("g", "pick_g", _GD, "^-"), ("h", "pick_h", "#eda100", "s--")), (-0.08, 1.08)),
                                 (a3, (("J", "pick_J", _BLEND, "D-"),), (-(1.0 - w) - 0.08, w + 0.08))):
            _pipeline_lines(ax, b, mine)
            for key, pick, colour, style in curves:
                c = np.asarray(b[key], dtype=float)
                edge = "#1a1a19" if colour in _LIGHT else colour
                ax.plot(ks, c, style, color=colour, mec=edge, mew=0.8, ms=4.5, lw=1.8, zorder=3)
                if b.get(pick) is not None:
                    d = int(b[pick])
                    ax.plot([ks[d]], [c[d]], style[0], ms=11, mfc=colour,
                            mec="#1a1a19" if colour in _LIGHT else "white", mew=1.5, zorder=4)
            ax.set_xlim(ks[0] - 0.5, ks[-1] + 0.5)
            ax.set_ylim(*lims)
            ax.set_xlabel("plane k", fontsize=8.5)
        a3.axhline(0.0, color="#c9c8c2", lw=0.8, zorder=1)
        a2.set_title("min-max rescaled: g from G (pick: its largest),\nh from the strip's entropy (pick: its lowest)",
                     fontsize=8.5)
        a2.set_ylabel("rescaled score", fontsize=8.5)
        a3.set_title("J = w g - (1 - w) h, w = %.2f at d_hat %.2f um\n(w = 1 / (1 + exp((d_hat - %.2g) / %.2g)); "
                     "marked: its largest)" % (w, float(b["d_hat_um"]), float(b["d0_um"]), float(b["s_um"])),
                     fontsize=8.5)
        a3.set_ylabel("J", fontsize=8.5)
        _style_axes((a1, a2, a3))
        y += L["heights"][4] + L["heights"][5]
    handles = [plt.Line2D([], [], color=_GD, marker="^", mec="#1a1a19", lw=1.8,
                          label="g: G over the whole d-line, rescaled; frame Gd: its largest"),
               plt.Line2D([], [], color="#eda100", marker="s", mec="#1a1a19", ls="--", lw=1.8,
                          label="h: the strip's entropy, rescaled (not inverted); frame Hs: its lowest"),
               plt.Line2D([], [], color=_BLEND, marker="D", lw=1.8, label="J = w g - (1 - w) h; frame J: its largest")]
    handles += _pipeline_handles(plt, blocks, mine)
    handles.append(_no_line_note(plt, "the line"))
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, 1.0 - 0.02 / L["H"]))
    if title:
        fig.suptitle(title, y=1.0 - 0.80 / L["H"], fontsize=10)
    return fig


# ---- the Allen-guided iteration of the gradient energy (D-042) ----
# colours (dataviz validator, light surface): round 1, on the line sized from Allen's 2r, yellow; the later rounds a
# blue ordinal ramp, darker = later (--ordinal: monotone lightness, one hue, light end 2.50:1 on white); the line
# sized from the pilot's d_hat aqua (at the same multiplier); the last plane's frame blue. Every cross pair of the
# ramp with yellow, aqua, the red of k* and the grey of the dip passes (normal vision >= 16.3, CVD >= 15.0);
# yellow-red and yellow-aqua pass as well, red-aqua stays at deutan 7.5 (D-040), so every series keeps its own marker,
# line style and frame tag. Yellow, aqua and the lightest blue (below 3:1 on white) carry a dark marker edge.
_ITER_ALLEN = "#eda100"
_ITER_RAMP = ("#6da7ec", "#2a78d6", "#1c5cab", "#0d366b")
_ITER_MARKERS = ("o", "s", "D", "^", "v")
_ITER_LAST = "#2a78d6"
_ITER_LIGHT = ("#eda100", "#1baf7a", "#6da7ec")


def iteration_style(i):
    """(colour, curve style) of round i >= 1 of the Allen-guided iteration
    (D-042): round 1 yellow, dashed, circles; round i >= 2 the blue ramp's
    step i - 2 (its darkest from round 5 on), solid, a marker of its own."""
    i = int(i)
    if i <= 1:
        return _ITER_ALLEN, "o--"
    return _ITER_RAMP[min(i - 2, len(_ITER_RAMP) - 1)], _ITER_MARKERS[min(i - 1, len(_ITER_MARKERS) - 1)] + "-"


def _iter_edge(colour):
    return "#1a1a19" if colour in _ITER_LIGHT else colour


def iteration_figure(blocks, title=""):
    """The Allen-guided iteration of the gradient energy (D-042). Per block
    (one node): the planes on one grey scale, without any line, framed at the
    iteration's last plane, round 1's plane, the pick of the line sized from
    the pilot's d_hat and the pipeline's planes; below, every plane's profile
    along the longest line used, with round 1's, the last round's and the
    pilot d-line's extents dashed; each round's G relative to its own maximum,
    with its pick (and the pilot d-line's G, aqua); and the trajectory: the
    diameter sizing each round's line -- Allen's 2r, then the fit at the plane
    each round picked, labelled with that plane (hollow: a plane picked again,
    its fit reused) -- against the pilot's d_hat (the fit at k*, red dashed).

    blocks: list of dicts with
      label, stack, ks, valid, extent, frames, lines (as gradient_lines_figure), v (M,), prof (n, M) along the
      longest line, rounds [dict(round, half_um, G (n,), pick: an index into ks or None)] (the rounds that
      evaluated G), gd (dict(half_um, G, pick) or None), d_start_um, fits [(round, k, d_hat_um, refit)] (the
      rounds that picked a plane), status, n_rounds, d_pilot_um, k_star, k_ref
    Presentation only. Returns the Figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n_rounds_max = max([x["round"] for b in blocks for x in b["rounds"]] + [1])
    n_entries = n_rounds_max + 1 + 1 + 1 + 3 + 1          # rounds, Gd, frame it, trajectory, pipeline, note
    rows = int(math.ceil(n_entries / 3.0))
    fig, ax_at, L = _bare_layout(plt, blocks, top_band=1.25 + 0.17 * max(0, rows - 3))
    y = L["top_band"]
    mine = {_ITER_ALLEN, _ITER_LAST, _GD}
    for b in blocks:
        ks = np.asarray(b["ks"])
        valid = np.asarray(b["valid"], dtype=bool)
        _block_label(fig, L, y, b["label"])
        y += L["heights"][0]
        _planes_row(plt, ax_at, b, y + L["heights"][1], L["left"], L["col"], L["side"])
        y += L["heights"][1] + L["heights"][2] + L["heights"][3]
        w3 = (L["W"] - L["left"] - L["right"] - 2 * 0.7) / 3.0
        a1 = ax_at(L["left"], y, w3, L["heights"][4])
        a2 = ax_at(L["left"] + w3 + 0.7, y, w3, L["heights"][4])
        a3 = ax_at(L["left"] + 2 * (w3 + 0.7), y, w3, L["heights"][4])
        rounds = list(b["rounds"])
        spans = []
        if rounds:
            spans.append((rounds[0]["half_um"], iteration_style(rounds[0]["round"])[0], "--"))
        if len(rounds) > 1:
            spans.append((rounds[-1]["half_um"], iteration_style(rounds[-1]["round"])[0], "--"))
        if b.get("gd") is not None:
            spans.append((b["gd"]["half_um"], _GD, "-."))
        if np.asarray(b["v"]).size >= 2:
            _profiles_panel(a1, b["v"], b["prof"], ks, valid, b["k_ref"], spans,
                            "profiles along the longest line; dashed: the\nextents of round 1's, the last round's and "
                            "the pilot's d-line")
            a1.set_xlim(-1.04 * max(h for h, _c, _l in spans), 1.04 * max(h for h, _c, _l in spans))
        else:
            a1.text(0.5, 0.5, "no line evaluated", ha="center", va="center", fontsize=8, transform=a1.transAxes)
        # each round's G relative to its own maximum, and the pilot d-line's
        _pipeline_lines(a2, b, mine)
        curves = [(iteration_style(x["round"]), x["G"], x.get("pick")) for x in rounds]
        if b.get("gd") is not None:
            curves.append(((_GD, "^-."), b["gd"]["G"], b["gd"].get("pick")))
        for (colour, style), G, pick in curves:
            G = np.asarray(G, dtype=float)
            rel = G / np.nanmax(G) if np.isfinite(G).any() and np.nanmax(G) > 0 else G
            a2.plot(ks, rel, style, color=colour, mec=_iter_edge(colour), mew=0.8, ms=4.5, lw=1.8, zorder=3)
            if pick is not None:
                d = int(pick)
                a2.plot([ks[d]], [rel[d]], style[0], ms=11, mfc=colour,
                        mec="#1a1a19" if colour in _ITER_LIGHT else "white", mew=1.5, zorder=4)
        a2.set_title("G over each round's whole line, relative to its\nmaximum (marked: each round's pick)",
                     fontsize=8.5)
        a2.set_xlabel("plane k", fontsize=8.5)
        a2.set_ylabel("G / max G", fontsize=8.5)
        a2.set_xlim(ks[0] - 0.5, ks[-1] + 0.5)
        # the trajectory of the diameter that sizes the lines
        xs, ds = [0], [float(b["d_start_um"])]
        a3.plot([0], [ds[0]], "s", ms=8, mfc=_ITER_ALLEN, mec="#1a1a19", mew=1.0, zorder=4)
        a3.annotate("Allen", (0, ds[0]), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7.5)
        for rnd, k, d, refit in b["fits"]:
            colour, style = iteration_style(rnd)
            if math.isfinite(d):
                xs.append(rnd)
                ds.append(d)
                a3.plot([rnd], [d], style[0], ms=8, mfc=colour if refit else "white",
                        mec=_iter_edge(colour) if refit else colour, mew=1.4, zorder=4)
                a3.annotate("k %d" % k, (rnd, d), textcoords="offset points", xytext=(0, 7), ha="center",
                            fontsize=7.5)
            else:
                a3.annotate("k %d:\nno fit" % k, (rnd, 0.0), textcoords="offset points", xytext=(0, 4), ha="center",
                            va="bottom", fontsize=7.5)
        a3.plot(xs, ds, "-", color="#9a9994", lw=1.0, zorder=2)
        dp = float(b["d_pilot_um"])
        if math.isfinite(dp):
            a3.axhline(dp, color="#ff1744", ls="--", lw=1.2, zorder=1)
        top = max([x for x in ds + [dp] if math.isfinite(x)] + [0.1])
        a3.set_ylim(0.0, 1.3 * top)
        n = max([int(b.get("n_rounds", 0))] + xs)
        a3.set_xlim(-0.5, n + 0.5)
        a3.set_xticks(list(range(n + 1)))
        a3.set_xticklabels(["2r"] + [str(i) for i in range(1, n + 1)])
        a3.set_xlabel("round whose plane was fitted (0: Allen's 2r)", fontsize=8.5)
        a3.set_ylabel("d (um)", fontsize=8.5)
        a3.set_title("d sizing each line: Allen's 2r, then the fit at\neach round's plane; red dashed: the pilot's "
                     "d_hat", fontsize=8.5)
        nr = int(b.get("n_rounds", 0))
        a3.text(0.98, 0.04, "%s after %d round%s" % (b.get("status", ""), nr, "" if nr == 1 else "s"),
                transform=a3.transAxes, ha="right", va="bottom", fontsize=7.5)
        _style_axes((a1, a2, a3))
        y += L["heights"][4] + L["heights"][5]
    handles = []
    for i in range(1, n_rounds_max + 1):
        colour, style = iteration_style(i)
        what = ("G on the line sized from Allen's 2r; frame A: its pick" if i == 1 else
                "G on the line sized from round %d's fit" % (i - 1))
        handles.append(plt.Line2D([], [], color=colour, marker=style[0], ls=style[1:], mec=_iter_edge(colour), lw=1.8,
                                  label="round %d: %s" % (i, what)))
    handles += [plt.Line2D([], [], color=_GD, marker="^", ls="-.", mec="#1a1a19", lw=1.8,
                           label="G on the line sized from the pilot's d_hat, same multiplier; frame Gd: its pick"),
                plt.Line2D([], [], color=_ITER_LAST, lw=2.6, label="frame it: the iteration's last plane"),
                plt.Line2D([], [], color="#ff1744", ls="--", lw=1.2, marker="o", mfc="white", mec="#7a7a7a",
                           label="right panel: the pilot's d_hat (dashed); hollow: a plane picked again")]
    handles += _pipeline_handles(plt, blocks, mine)
    handles.append(_no_line_note(plt, "any line"))
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, 1.0 - 0.02 / L["H"]))
    if title:
        fig.suptitle(title, y=1.0 - (L["top_band"] - 0.45) / L["H"], fontsize=10)
    return fig


def picks_summary_figure(records, series, title="", weight_key=None, note_key=None):
    """Small multiples over several nodes (D-040): one panel per node (in the
    records' order), each series' curve rescaled to 0 .. 1 (min-max) and drawn
    as it is (since D-041 never inverted: a series that picks its lowest value
    has its pick at the bottom) against the plane offset k - k_SWC, with k*
    (red), the dip depth's plane (grey dashed), the SWC plane (dotted) and each
    series' pick (a large marker).

    records: per-node dicts with node_id, node_type, allen_radius_um, d_hat_um, ks, k_swc, k_star, k_star_depth,
      and the series' curves and picks; series: [dict(curve, pick, label, colour, style, tag=None)]; weight_key: a
      record field shown as w in each panel's title, or None; note_key: a record field holding a short text added
      as a third title line (D-042), or None. A series whose curve the record lacks, or holds as None, is not drawn.
    Presentation only. Returns the Figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(records)
    ncol = max(1, min(4, n))
    nrow = int(math.ceil(n / float(ncol)))
    H = 2.75 * nrow + 1.9
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.7 * ncol, H), squeeze=False)
    names = {3: "basal", 4: "apical"}

    def unit(c):
        c = np.asarray(c, dtype=float)
        if not np.isfinite(c).any():
            return c
        lo, hi = np.nanmin(c), np.nanmax(c)
        if not hi > lo:
            return np.zeros_like(c)
        return (c - lo) / (hi - lo)

    for i, r in enumerate(records):
        ax = axes[i // ncol, i % ncol]
        ks = np.asarray(r["ks"])
        off = ks - int(r["k_swc"])
        for kk, colour, ls in ((r.get("k_star"), "#ff1744", "-"), (r.get("k_star_depth"), "#7a7a7a", "--"),
                               (r.get("k_swc"), "#1a1a19", ":")):
            if kk is not None:
                ax.axvline(int(kk) - int(r["k_swc"]), color=colour, ls=ls, lw=1.2, zorder=1)
        parts = []
        for s in series:
            if r.get(s["curve"]) is None:
                continue
            u = unit(r[s["curve"]])
            edge = "#1a1a19" if s["colour"] in _LIGHT else s["colour"]
            ax.plot(off, u, s["style"], color=s["colour"], mec=edge, mew=0.7, ms=3.5, lw=1.5, zorder=3)
            k = r.get(s["pick"])
            tag = s.get("tag") or s["pick"][2:]
            if k is not None:
                j = int(k) - int(ks[0])
                ax.plot([off[j]], [u[j]], s["style"][0], ms=10, mfc=s["colour"],
                        mec="#1a1a19" if s["colour"] in _LIGHT else "white", mew=1.3, zorder=4)
                parts.append("%s %+d" % (tag, int(k) - int(r["k_star"])))
            else:
                parts.append("%s n/a" % tag)
        head = "node %d (%s), d_hat %.2f um, Allen 2r %.2f um" % (
            int(r["node_id"]), names.get(r.get("node_type"), "type %s" % r.get("node_type")),
            float(r.get("d_hat_um", float("nan"))), 2.0 * float(r.get("allen_radius_um", float("nan"))))
        if weight_key is not None and r.get(weight_key) is not None:
            head += ", w %.2f" % float(r[weight_key])
        note = ("\n" + str(r[note_key])) if note_key is not None and r.get(note_key) else ""
        ax.set_title(head + "\npick minus k*: " + ", ".join(parts) + note, fontsize=8)
        ax.set_ylim(-0.12, 1.12)
        ax.set_xlim(off.min() - 0.5, off.max() + 0.5)
        ax.tick_params(labelsize=7.5)
        ax.grid(axis="y", color="#e6e5e0", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if i % ncol == 0:
            ax.set_ylabel("rescaled 0-1 (picks marked)", fontsize=8.5)
        if i // ncol == nrow - 1 or i + ncol >= n:
            ax.set_xlabel("plane k - k_SWC", fontsize=8.5)
    for j in range(n, nrow * ncol):
        axes[j // ncol, j % ncol].axis("off")
    handles = [plt.Line2D([], [], color=s["colour"], marker=s["style"][0], ls=s["style"][1:] or "-",
                          mec="#1a1a19" if s["colour"] in _LIGHT else s["colour"], lw=1.5, label=s["label"])
               for s in series]
    handles += [plt.Line2D([], [], color="#ff1744", lw=1.2, label="k* (gradient energy, D-030)"),
                plt.Line2D([], [], color="#7a7a7a", ls="--", lw=1.2, label="plane of the dip depth"),
                plt.Line2D([], [], color="#1a1a19", ls=":", lw=1.2, label="the SWC node's plane")]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.995))
    if title:
        fig.suptitle(title, y=1.0 - 0.80 / H, fontsize=10)
    fig.subplots_adjust(top=1.0 - 1.70 / H, bottom=0.55 / H, left=0.06, right=0.99, hspace=0.75, wspace=0.18)
    return fig


# ---- Cell 4k: the iteration's planes with the line that chose them and the diameter fitted there (D-045) ----
# One accent hue: blue (the iteration's last-plane blue) is the fit -- the caliper on the image, the model and the
# edges on the profile; the data are neutral ink: the line that chose the plane black (white halo, as the measuring
# lines of Cells 4e-4g), its profile grey, the fit's samples near-black dots. The validator's categorical checks pass
# for the blue against both neutrals (CVD >= 15.1, normal vision >= 16.3, contrast >= 3:1); its lightness and chroma
# checks do not apply to neutral ink, which differs from the fit by mark (line, dots) as well as colour.
_FIT = "#2a78d6"
_SEL_GREY = "#7a7a7a"
_INK = "#1a1a19"


def _scale_bar(ax, extent, pe):
    """A white bar of 1, 2, 5 or 10 um (the longest within 30 % of the image's width) at its lower left."""
    x0, x1, yb, yt = (float(x) for x in extent)
    width = x1 - x0
    L = max([x for x in (1.0, 2.0, 5.0, 10.0) if x <= 0.3 * width] or [0.5])
    xa, ya = x0 + 0.06 * width, yb - 0.07 * (yb - yt)
    dark = [pe.Stroke(linewidth=4.5, foreground="#1a1a19"), pe.Normal()]
    ax.plot([xa, xa + L], [ya, ya], color="white", lw=2.5, solid_capstyle="butt", path_effects=dark, zorder=5)
    ax.text(xa + 0.5 * L, ya - 0.035 * (yb - yt), "%g um" % L, color="white", fontsize=7.5, ha="center", va="bottom",
            path_effects=[pe.Stroke(linewidth=2.2, foreground="#1a1a19"), pe.Normal()], zorder=5)


def _draw_fit_on_plane(ax, b, pe, tick_um, lw_line=1.2, lw_cal=2.8):
    """The plane on its own grey scale (1st-99.5th percentile), the line that chose it, the caliper (when there is
    a diameter) with end ticks along the branch, and the SWC node."""
    img = np.asarray(b["plane"], dtype=float)
    lo, hi = np.nanpercentile(img, [1.0, 99.5])
    ax.imshow(img, cmap="gray", vmin=lo, vmax=hi, extent=b["extent"], origin="upper", interpolation="nearest")
    halo = [pe.Stroke(linewidth=lw_line + 2.2, foreground="white"), pe.Normal()]
    L = np.asarray(b["line_xy"], dtype=float)
    ax.plot(L[:, 0], L[:, 1], color="black", lw=lw_line, path_effects=halo, zorder=3)
    if b.get("edges_xy") is not None:
        E = np.asarray(b["edges_xy"], dtype=float)
        e_u = np.asarray(b["e_u"], dtype=float)
        cal = [pe.Stroke(linewidth=lw_cal + 2.0, foreground="white"), pe.Normal()]
        ax.plot(E[:, 0], E[:, 1], color=_FIT, lw=lw_cal, solid_capstyle="butt", path_effects=cal, zorder=4)
        for P in E:
            T = np.array([P - 0.5 * tick_um * e_u, P + 0.5 * tick_um * e_u])
            ax.plot(T[:, 0], T[:, 1], color=_FIT, lw=0.75 * lw_cal, solid_capstyle="butt", path_effects=cal, zorder=4)
    o = np.asarray(b["o"], dtype=float)
    ax.plot([o[0]], [o[1]], marker="+", ms=9, mew=1.6, color="black", zorder=6,
            path_effects=[pe.Stroke(linewidth=3.4, foreground="white"), pe.Normal()])
    x0, x1, yb, yt = (float(x) for x in b["extent"])
    ax.set_xlim(x0, x1)
    ax.set_ylim(yb, yt)
    ax.set_xticks([])
    ax.set_yticks([])


def iteration_fit_figure(block, title=""):
    """One node of Cell 4k (D-045). Left: the plane the iteration chose, with
    the line that chose it (black), the diameter fitted there as a caliper
    between its two edges (blue) and the SWC node (+). Right: against v along
    that line (0 at the SWC node), the plane's profile along it (grey), the
    fit's profile (black dots, the average of parallel lines through the
    fitted centre, drawn at s + v), the fitted model (blue), B_bar (dashed),
    the edges (blue verticals, with d between them) and the fit's window
    (shaded).

    block: dict with label, sublabel, plane (H, W), extent (image_extent_um), o (2,), e_u (2,), line_xy (2, 2) um,
    edges_xy (2, 2) or None, edges_v (2,) or None, d (um, NaN without a fit), v_sel (M,), prof_sel (M,), s_um,
    v_fit (N,) or None (already shifted by s), prof_fit (N,) or None, model (N,) or None, B_bar, fit_half_um.
    Presentation only. Returns the Figure."""
    import textwrap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    import matplotlib.transforms as mtransforms
    W, H = 12.0, 6.7
    fig = plt.figure(figsize=(W, H))

    def ax_at(x_in, y_top_in, w_in, h_in):
        return fig.add_axes([x_in / W, 1.0 - (y_top_in + h_in) / H, w_in / W, h_in / H])

    if title:
        fig.text(0.5, 1.0 - 0.12 / H, title, fontsize=10, ha="center", va="top")
    fig.text(0.4 / W, 1.0 - 0.45 / H, "\n".join(textwrap.wrap(block["label"], 118)), fontsize=9.5,
             fontweight="bold", ha="left", va="top", linespacing=1.15)
    fig.text(0.4 / W, 1.0 - 0.92 / H, "\n".join(textwrap.wrap(block["sublabel"], 150)), fontsize=8.5, ha="left",
             va="top", linespacing=1.15, color="#3d3d3a")
    side = 4.1
    a_img = ax_at(0.4, 1.65, side, side)
    x0, x1 = (float(x) for x in block["extent"][:2])
    _draw_fit_on_plane(a_img, block, pe, tick_um=max(0.25, 0.045 * (x1 - x0)))
    _scale_bar(a_img, block["extent"], pe)
    a_img.set_title("the plane, the line that chose it, the diameter (blue)", fontsize=8.5, pad=4)
    ax = ax_at(5.55, 1.65, 6.15, 3.45)
    h_fit, s = float(block["fit_half_um"]), float(block["s_um"])
    span = max(float(np.max(np.abs(block["v_sel"]))) if np.size(block["v_sel"]) else 0.0, abs(s) + h_fit)
    ax.axvspan(s - h_fit, s + h_fit, color="#efeeea", zorder=0, lw=0)
    ax.plot(block["v_sel"], block["prof_sel"], color=_SEL_GREY, lw=1.4, zorder=2)
    if block.get("prof_fit") is not None:
        ax.plot(block["v_fit"], block["prof_fit"], "o", ms=3.2, color=_INK, zorder=3)
    if block.get("model") is not None:
        ax.plot(block["v_fit"], block["model"], color=_FIT, lw=2.0, zorder=4)
    if block.get("B_bar") is not None and math.isfinite(float(block["B_bar"])):
        ax.axhline(float(block["B_bar"]), color=_SEL_GREY, ls="--", lw=0.9, zorder=1)
    if block.get("edges_v") is not None:
        ev = [float(x) for x in block["edges_v"]]
        for x in ev:
            ax.axvline(x, color=_FIT, lw=1.1, zorder=4)
        tr = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.annotate("", xy=(ev[0], 1.04), xytext=(ev[1], 1.04), xycoords=tr, textcoords=tr,
                    arrowprops=dict(arrowstyle="<->", color=_FIT, lw=1.2, shrinkA=0, shrinkB=0), annotation_clip=False)
        ax.text(0.5 * (ev[0] + ev[1]), 1.06, "d = %.2f um" % float(block["d"]), transform=tr, fontsize=8.5,
                ha="center", va="bottom", color=_INK)
    ax.set_xlim(-1.04 * span, 1.04 * span)
    ax.set_xlabel("v along the line that chose the plane (um; 0 = the SWC node)", fontsize=8.5)
    ax.set_ylabel("grey level", fontsize=8.5)
    _style_axes([ax])
    handles = [plt.Line2D([], [], color=_SEL_GREY, lw=1.4, label="right: the plane along the line that chose it"),
               plt.Line2D([], [], color=_INK, marker="o", ms=3.2, ls="none",
                          label="right: the fit's profile (parallel lines averaged along the branch, through the "
                                "fitted centre)"),
               plt.Line2D([], [], color=_FIT, lw=2.0, label="right: the fitted model; blue verticals: its edges, d apart"),
               plt.Line2D([], [], color=_SEL_GREY, ls="--", lw=0.9, label="right: B_bar, the fit's background"),
               plt.Rectangle((0, 0), 1, 1, color="#efeeea",
                             label="right: the fit's window (+-%g um about the fitted centre)" % h_fit),
               plt.Line2D([], [], color="black", marker="+", ms=8, mew=1.5, ls="none", label="left: the SWC node")]
    fig.legend(handles=handles, loc="upper left", ncol=2, fontsize=7.5, frameon=False,
               bbox_to_anchor=(0.35 / W, 1.0 - 5.93 / H))
    return fig


def iteration_fits_overview(blocks, title="", ncol=5):
    """Every node of Cell 4k (D-045) at a glance: per node the plane the
    iteration chose with the line that chose it (black), the diameter fitted
    there as a caliper (blue) and the SWC node (+), in the order given.

    blocks: list of dicts with label (two short lines), plane, extent, o, e_u, line_xy, edges_xy or None.
    Presentation only. Returns the Figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    ncol = max(1, min(int(ncol), len(blocks)))
    rows = int(math.ceil(len(blocks) / float(ncol)))
    side, gap, head, left, top = 2.35, 0.25, 0.55, 0.35, 0.85
    W = max(2 * left + ncol * side + (ncol - 1) * gap, 8.0)
    H = top + rows * (head + side) + 0.25
    fig = plt.figure(figsize=(W, H))
    for i, b in enumerate(blocks):
        r, c = divmod(i, ncol)
        y_top = top + r * (head + side) + head
        ax = fig.add_axes([(left + c * (side + gap)) / W, 1.0 - (y_top + side) / H, side / W, side / H])
        x0, x1 = (float(x) for x in b["extent"][:2])
        _draw_fit_on_plane(ax, b, pe, tick_um=max(0.25, 0.06 * (x1 - x0)), lw_line=1.0, lw_cal=2.4)
        _scale_bar(ax, b["extent"], pe)
        ax.set_title(b["label"], fontsize=8, pad=3, linespacing=1.1)
    handles = [plt.Line2D([], [], color="black", lw=1.2, label="the line that chose the plane"),
               plt.Line2D([], [], color=_FIT, lw=2.8, label="the diameter fitted there (its two edges)"),
               plt.Line2D([], [], color="black", marker="+", ms=9, mew=1.6, ls="none", label="the SWC node")]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, 1.0 - 0.5 / H))
    if title:
        fig.text(0.5, 1.0 - 0.12 / H, title, fontsize=10, ha="center", va="top")
    return fig


# ---- Cell 4l: fit variants at the iteration's plane (D-046) ----
# Kernel families take the categorical slots in fixed order (validated 2026-10-13: adjacent CVD >= 9.1, normal vision
# >= 19.6; aqua, yellow and magenta below 3:1 on the surface, so every curve is also named by its line style, the
# caliper rows' direct labels and the table). Shapes are told apart by line style (fixed solid, a second fixed member
# dash-dot, fitted dotted). The weighting is an ordinal blue ramp from step 250 (lambda = 2 um, the weakest) to 700
# (0.25 um, the strongest), validated --ordinal; unweighted is ink. The table's ratio is diverging blue <-> red about
# a neutral grey (blue: smaller than the baseline).
FAMILY_COLOURS = {"gauss": "#2a78d6", "lorentz": "#eb6834", "voigt": "#1baf7a", "student": "#eda100",
                  "gennorm": "#e87ba4", "skewnorm": "#008300"}
_LAM_RAMP = {0.25: "#0d366b", 0.5: "#1c5cab", 1.0: "#3987e5", 2.0: "#86b6ef"}
_DIV = ("#1c5cab", "#f0efec", "#e34948")
_FLAG_NOTE = ("* a parameter at its bound; x failed; ~ IRLS not settled in 30 iterations; dashed box: the least residual "
              "among the unweighted fits at that width (more fitted parameters can only lower it)")


def kernel_style(family, shape_fitted, second_fixed=False):
    """(colour, line style) of a kernel configuration in the Cell 4l figures."""
    ls = ":" if shape_fitted else ("-." if second_fixed else "-")
    return FAMILY_COLOURS[family], ls


def lam_colour(lam_um):
    """The weighting's colour: ink unweighted, else the ramp's step (grey for a lambda off the scan)."""
    if lam_um is None:
        return _INK
    return _LAM_RAMP.get(float(lam_um), _SEL_GREY)


def _caliper_rows(ax, rows, title):
    """Horizontal segments v0 -+ d/2, one row per item (top to bottom), each
    labelled at its left with its text; rows: dicts label, d, v0, colour, ls."""
    n = len(rows)
    lo, hi = [], []
    for r in rows:
        if r["d"] is not None and math.isfinite(r["d"]):
            lo.append(r["v0"] - 0.5 * r["d"])
            hi.append(r["v0"] + 0.5 * r["d"])
    if lo:
        a, b = min(lo), max(hi)
        pad = 0.35 * (b - a) + 0.05
        ax.set_xlim(a - 1.6 * pad, b + 0.4 * pad)
    for i, r in enumerate(rows):
        y = n - 1 - i
        if r["d"] is not None and math.isfinite(r["d"]):
            x0, x1 = r["v0"] - 0.5 * r["d"], r["v0"] + 0.5 * r["d"]
            ax.plot([x0, x1], [y, y], color=r["colour"], ls=r["ls"], lw=2.2, solid_capstyle="butt")
            ax.plot([x0, x0], [y - 0.28, y + 0.28], color=r["colour"], lw=1.2)
            ax.plot([x1, x1], [y - 0.28, y + 0.28], color=r["colour"], lw=1.2)
        ax.text(0.005, y, r["label"], transform=ax.get_yaxis_transform(), fontsize=6.6, ha="left", va="center",
                color=_INK)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=6.8, top=True, labeltop=True, bottom=False, labelbottom=False, pad=1)
    for s in ("left", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_title(title, fontsize=8, pad=14)


def fit_variants_node_figure(block, title=""):
    """One node of Cell 4l (D-046). Three columns of the profile at k_iter
    against v (um from the fitted centre): every kernel unweighted at the
    fixed width; every kernel unweighted at a fitted width; the Gaussian at
    the fixed width, unweighted and for each lambda, with the weights below.
    Above each, the fits' edges v0 -+ d/2 as stacked calipers (zoomed, with
    their d). Below, d of every variant as a table coloured by d / d_base on a
    log scale (the baseline outlined).

    block: dict with label, sublabel, v (N,), I (N,), B_bar, v_fine (M,), fixed and fitted (lists of dicts label,
    family, ls, curve (M,) or None, d, v0, flag, w), lam (list of dicts lam (None or um), curve, d, v0, flag,
    weights (N,) or None), table dict(rows (R labels), row_colours, row_ls, cols (C (width mode, weight label)),
    d (R, C), ratio (R, C), flags (R, C) str, base (r, c) or None), fixed_fwhm_um.
    Presentation only. Returns the Figure."""
    import textwrap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    t = block["table"]
    R, C = len(t["rows"]), len(t["cols"])
    cal_top, cal_h, prof_top, prof_h = 2.0, 1.55, 3.8, 2.55
    tab_top, cell_h, lab_w = 8.6, 0.25, 1.75
    tab_h = max(cell_h * R, 1.3)                         # room for the colour bar under a short table
    W, H = 15.6, tab_top + tab_h + 1.55
    fig = plt.figure(figsize=(W, H))

    def ax_at(x_in, y_top_in, w_in, h_in):
        return fig.add_axes([x_in / W, 1.0 - (y_top_in + h_in) / H, w_in / W, h_in / H])

    if title:
        fig.text(0.5, 1.0 - 0.12 / H, title, fontsize=10, ha="center", va="top")
    fig.text(0.45 / W, 1.0 - 0.45 / H, "\n".join(textwrap.wrap(block["label"], 150)), fontsize=9.5,
             fontweight="bold", ha="left", va="top", linespacing=1.15)
    fig.text(0.45 / W, 1.0 - 0.80 / H, "\n".join(textwrap.wrap(block["sublabel"], 190)), fontsize=8.3, ha="left",
             va="top", linespacing=1.15, color="#3d3d3a")
    v, I, B = np.asarray(block["v"]), np.asarray(block["I"]), float(block["B_bar"])
    vf = np.asarray(block["v_fine"])
    colw, gap, left = 4.55, 0.62, 0.62
    w_fix = float(block["fixed_fwhm_um"])
    titles = ("every kernel, fixed width (FWHM %.3f um), unweighted" % w_fix,
              "every kernel, width fitted, unweighted",
              "Gaussian, fixed width: unweighted and centre-weighted")
    prof_axes = []
    for c, key in enumerate(("fixed", "fitted", "lam")):
        x = left + c * (colw + gap)
        a_cal = ax_at(x, cal_top, colw, cal_h)
        a_pro = ax_at(x, prof_top, colw, prof_h)
        prof_axes.append(a_pro)
        a_pro.plot(v, I, "o", ms=3.0, color=_INK, zorder=3)
        a_pro.axhline(B, color=_SEL_GREY, ls="--", lw=0.9, zorder=1)
        rows = []
        for item in block[key]:
            if key == "lam":
                colour, ls = lam_colour(item["lam"]), "-"
                lab = "none" if item["lam"] is None else "lambda %g um" % item["lam"]
            else:
                colour, ls = FAMILY_COLOURS[item["family"]], item["ls"]
                lab = item["label"]
                if key == "fitted" and item.get("w") is not None and math.isfinite(item["w"]):
                    lab += ", w %.2f" % item["w"]
            if item.get("curve") is not None:
                a_pro.plot(vf, item["curve"], color=colour, ls=ls, lw=1.3, zorder=4)
            d = item.get("d")
            text = "%s: %s%s" % (lab, "n/a" if d is None or not math.isfinite(d) else "%.2f" % d, item.get("flag", ""))
            rows.append(dict(label=text, d=d, v0=item.get("v0"), colour=colour, ls=ls))
        _caliper_rows(a_cal, rows, titles[c] + "\nedges v0 -+ d/2 (um, zoomed), d in um")
        a_pro.set_xlim(v[0] - 0.1, v[-1] + 0.1)
        a_pro.set_xlabel("v (um from the fitted centre, across the branch)", fontsize=8)
        if c == 0:
            a_pro.set_ylabel("grey level", fontsize=8.5)
        _style_axes([a_pro])
    # the weights under the lambda column
    a_w = ax_at(left + 2 * (colw + gap), prof_top + prof_h + 0.62, colw, 0.75)
    for item in block["lam"]:
        if item["lam"] is not None and item.get("weights") is not None:
            a_w.plot(v, item["weights"], "-", color=lam_colour(item["lam"]), lw=1.3)
    a_w.set_xlim(prof_axes[2].get_xlim())
    a_w.set_ylim(0, 1.05)
    a_w.set_ylabel("weight", fontsize=8)
    a_w.set_xlabel("v (um); the weights exp(-|v - v0| / lambda) at the fit's v0", fontsize=7.5)
    _style_axes([a_w])
    # the table
    tab_w = W - left - lab_w - 2.9
    a_t = ax_at(left + lab_w, tab_top, tab_w, cell_h * R)
    ratio = np.asarray(t["ratio"], dtype=float)
    lim = math.log2(1.5)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.clip(np.log2(ratio), -lim, lim)
    cmap = LinearSegmentedColormap.from_list("div", list(_DIV))
    cmap.set_bad("#ffffff")
    a_t.imshow(np.ma.masked_invalid(z), cmap=cmap, vmin=-lim, vmax=lim, aspect="auto", interpolation="nearest")
    for r in range(R):
        for c in range(C):
            d = t["d"][r][c]
            txt = "n/a" if d is None or not math.isfinite(d) else "%.2f" % d
            dark = np.isfinite(z[r, c]) and abs(z[r, c]) > 0.75 * lim
            a_t.text(c, r, txt + t["flags"][r][c], ha="center", va="center", fontsize=7,
                     color="white" if dark else _INK)
    if t.get("base") is not None:
        rb, cb = t["base"]
        a_t.add_patch(plt.Rectangle((cb - 0.5, rb - 0.5), 1, 1, fill=False, ec=_INK, lw=1.8))
    for rb, cb in t.get("best") or []:                    # the least residual among the unweighted fits, per width
        a_t.add_patch(plt.Rectangle((cb - 0.42, rb - 0.4), 0.84, 0.8, fill=False, ec=_INK, lw=1.2, ls="--"))
    a_t.set_yticks(range(R))
    a_t.set_yticklabels(t["rows"], fontsize=7.5, color=_INK)
    for r in range(R):                                   # a swatch of each kernel's curve, between its name and the row
        a_t.plot([-0.5 - 0.5, -0.5 - 0.1], [r, r], color=t["row_colours"][r], ls=t["row_ls"][r], lw=1.6,
                 clip_on=False)
    a_t.set_xlim(-0.5, C - 0.5)
    a_t.set_xticks(range(C))
    a_t.set_xticklabels([w for _, w in t["cols"]], fontsize=7.2)
    a_t.tick_params(length=0, labelsize=7.2)
    a_t.tick_params(axis="y", pad=42)
    for s in a_t.spines.values():
        s.set_visible(False)
    groups = []
    for c, (mode, _) in enumerate(t["cols"]):
        if not groups or groups[-1][0] != mode:
            groups.append([mode, c, c])
        else:
            groups[-1][2] = c
    for mode, c0, c1 in groups:
        a_t.text(0.5 * (c0 + c1), -0.75, "%s width" % mode, ha="center", va="bottom", fontsize=8, color=_INK,
                 fontweight="bold")
        if c0 > 0:
            a_t.axvline(c0 - 0.5, color="white", lw=3)
    a_t.set_title("d (um) of every variant; colour d / d_base (the Gaussian, fixed width, unweighted, outlined = "
                  "Block 5's fit)", fontsize=8.5, pad=20)
    # colour bar, ratio ticks
    a_cb = ax_at(left + lab_w + tab_w + 0.45, tab_top + 0.3, 0.18, tab_h - 0.5)
    ramp = np.linspace(-lim, lim, 101)[:, None]
    a_cb.imshow(ramp[::-1], cmap=cmap, vmin=-lim, vmax=lim, aspect="auto",
                extent=(0, 1, -lim, lim))
    ticks = [2.0 / 3.0, 0.8, 1.0, 1.25, 1.5]
    a_cb.set_yticks([math.log2(x) for x in ticks])
    a_cb.set_yticklabels(["%.2f" % x for x in ticks], fontsize=7)
    a_cb.yaxis.tick_right()
    a_cb.set_xticks([])
    a_cb.set_title("d / d_base", fontsize=7.5, pad=4)
    # legend
    handles = [plt.Line2D([], [], color=_INK, marker="o", ms=3.0, ls="none", label="the profile at k_iter (Block 5's)"),
               plt.Line2D([], [], color=_SEL_GREY, ls="--", lw=0.9, label="B_bar")]
    seen = set()
    for item in block["fixed"]:
        if item["family"] not in seen:
            seen.add(item["family"])
            handles.append(plt.Line2D([], [], color=FAMILY_COLOURS[item["family"]], lw=1.6, label=item["family"]))
    handles += [plt.Line2D([], [], color=_SEL_GREY, ls="-", lw=1.4, label="fixed shape"),
                plt.Line2D([], [], color=_SEL_GREY, ls="-.", lw=1.4, label="second fixed shape"),
                plt.Line2D([], [], color=_SEL_GREY, ls=":", lw=1.4, label="shape fitted")]
    handles += [plt.Line2D([], [], color=lam_colour(item["lam"]), lw=1.6,
                           label="unweighted" if item["lam"] is None else "lambda %g um" % item["lam"])
                for item in block["lam"]]
    fig.legend(handles=handles, loc="upper left", ncol=8, fontsize=7.3, frameon=False,
               bbox_to_anchor=(0.4 / W, 1.0 - (tab_top + tab_h + 0.55) / H))
    fig.text(0.45 / W, 1.0 - (tab_top + tab_h + 1.12) / H, _FLAG_NOTE, fontsize=7.5, ha="left", va="top",
             color="#3d3d3a")
    return fig


def fit_variants_summary_figure(records, kernels, title=""):
    """Every node of Cell 4l (D-046) at a glance: one panel per kernel and
    width mode (fixed width in the upper half, fitted below), d / d_base
    against d_base (both log), one series per weighting (unweighted ink,
    lambda the blue ramp), hollow where the fit is flagged.

    records: per node dicts node_id, d_base, and ratio / flag dicts keyed by (kernel label, width mode, weight
    label); kernels: kernel labels in order; weights are read from the records' keys, unweighted first.
    Presentation only. Returns the Figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, NullLocator
    modes = ("fixed", "fitted")
    weights = []
    for r in records:
        for (_, _, wl) in r["ratio"]:
            if wl not in weights:
                weights.append(wl)
    weights.sort(key=lambda s: (s != "none", float(s.split("=")[1]) if s != "none" else 0.0))
    ncol = min(5, max(1, len(kernels)))
    nrow_half = int(math.ceil(len(kernels) / float(ncol)))
    nrow = 2 * nrow_half
    H = 2.35 * nrow + 1.6
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.25 * ncol, H), squeeze=False, sharex=True, sharey=True)
    xs = [float(r["d_base"]) for r in records if math.isfinite(float(r["d_base"]))]
    ys = [float(x) for r in records for x in r["ratio"].values() if x is not None and math.isfinite(float(x))]
    order = sorted(range(len(records)), key=lambda i: float(records[i]["d_base"]))
    for m, mode in enumerate(modes):
        for k, kern in enumerate(kernels):
            ax = axes[m * nrow_half + k // ncol, k % ncol]
            ax.axhline(1.0, color=_SEL_GREY, lw=0.9, zorder=1)
            for wl in weights:
                lam = None if wl == "none" else float(wl.split("=")[1])
                colour = lam_colour(lam)
                xx, yy, hollow = [], [], []
                for i in order:
                    r = records[i]
                    y = r["ratio"].get((kern, mode, wl))
                    if y is None or not math.isfinite(float(y)):
                        continue
                    xx.append(float(r["d_base"]))
                    yy.append(float(y))
                    hollow.append(bool(r["flag"].get((kern, mode, wl))))
                if not xx:
                    continue
                ax.plot(xx, yy, "-", color=colour, lw=1.0, zorder=2)
                xx, yy, hollow = np.array(xx), np.array(yy), np.array(hollow)
                ax.plot(xx[~hollow], yy[~hollow], "o", ms=4.2, color=colour, mec="white", mew=0.5, zorder=3)
                ax.plot(xx[hollow], yy[hollow], "o", ms=4.2, mfc="white", mec=colour, mew=1.1, zorder=3)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title("%s, %s width" % (kern, mode), fontsize=8)
            _style_axes([ax])
            ax.grid(axis="x", color="#e6e5e0", lw=0.6)
            if k % ncol == 0:
                ax.set_ylabel("d / d_base", fontsize=8)
            if m == 1 and k // ncol == nrow_half - 1:
                ax.set_xlabel("d_base (um, Block 5 at k_iter)", fontsize=8)
        for k in range(len(kernels), nrow_half * ncol):
            axes[m * nrow_half + k // ncol, k % ncol].axis("off")
    if xs:
        ax0 = axes[0, 0]
        ax0.set_xlim(min(xs) / 1.25, max(xs) * 1.25)
        xt = [x for x in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0) if min(xs) / 1.25 <= x <= max(xs) * 1.25]
        ax0.xaxis.set_major_locator(FixedLocator(xt))
        ax0.xaxis.set_minor_locator(NullLocator())
        ax0.set_xticklabels(["%g" % x for x in xt])
    if ys:
        lo, hi = max(min(ys), 0.2), min(max(ys), 5.0)
        lo, hi = min(lo, 0.8), max(hi, 1.25)
        ax0 = axes[0, 0]
        ax0.set_ylim(lo / 1.1, hi * 1.1)
        yt = [y for y in (0.25, 0.5, 0.67, 0.8, 1.0, 1.25, 1.5, 2.0, 4.0) if lo / 1.1 <= y <= hi * 1.1]
        ax0.yaxis.set_major_locator(FixedLocator(yt))
        ax0.yaxis.set_minor_locator(NullLocator())
        ax0.set_yticklabels(["%g" % y for y in yt])
    handles = [plt.Line2D([], [], color=lam_colour(None if wl == "none" else float(wl.split("=")[1])), marker="o",
                          ms=4.2, lw=1.0, label="unweighted" if wl == "none" else "lambda %s um" % wl.split("=")[1])
               for wl in weights]
    handles.append(plt.Line2D([], [], color=_SEL_GREY, marker="o", mfc="white", ms=4.2, ls="none",
                              label="hollow: at a bound, failed, or IRLS not settled"))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, 1.0 - 0.42 / H))
    if title:
        fig.text(0.5, 1.0 - 0.12 / H, title, fontsize=10, ha="center", va="top")
    fig.subplots_adjust(top=1.0 - 1.05 / H, bottom=0.6 / H, left=0.06, right=0.99, hspace=0.42, wspace=0.12)
    return fig


# ---- Cell 4m: each fit variant's own iteration (D-047) ----
# The kernels keep Cell 4l's colours and line styles; the reference (Cell 4j's own iteration, Block 5's fit) is ink.
_KSTAR = "#ff1744"


def variant_stack_figure(block, title=""):
    """One node of Cell 4m (D-047): one row per variant (the reference, Cell
    4j's own iteration with Block 5's fit, first), each with its label and d,
    its fitted edges v0 -+ d/2 at the plane its own iteration chose (um from
    the pilot's fitted centre), and the planes its rounds picked (grey, in
    order) ending on k_iter (filled), against k* (red) and the SWC plane
    (dotted).

    block: dict with label, sublabel, rows (dicts label, d, v0, colour, ls, planes (list of k or None), k_iter, flag,
    reference), ks, k_star, k_swc. Presentation only. Returns the Figure."""
    import textwrap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = block["rows"]
    n = len(rows)
    row_h, top, left = 0.172, 1.55, 0.35
    W = 14.0
    H = top + n * row_h + 1.25
    fig = plt.figure(figsize=(W, H))

    def ax_at(x_in, y_top_in, w_in, h_in):
        return fig.add_axes([x_in / W, 1.0 - (y_top_in + h_in) / H, w_in / W, h_in / H])

    if title:
        fig.text(0.5, 1.0 - 0.12 / H, title, fontsize=10, ha="center", va="top")
    fig.text(left / W, 1.0 - 0.45 / H, "\n".join(textwrap.wrap(block["label"], 160)), fontsize=9.2,
             fontweight="bold", ha="left", va="top", linespacing=1.15)
    fig.text(left / W, 1.0 - 0.85 / H, "\n".join(textwrap.wrap(block["sublabel"], 190)), fontsize=8.2, ha="left",
             va="top", linespacing=1.15, color="#3d3d3a")
    h_axes = n * row_h
    a_lab = ax_at(left, top, 3.1, h_axes)
    a_cal = ax_at(left + 3.25, top, 6.0, h_axes)
    a_pl = ax_at(left + 9.55, top, 3.9, h_axes)
    ys = [n - 1 - i for i in range(n)]
    lo, hi = [], []
    for r in rows:
        if r["d"] is not None and math.isfinite(r["d"]) and math.isfinite(r["v0"]):
            lo.append(r["v0"] - 0.5 * r["d"])
            hi.append(r["v0"] + 0.5 * r["d"])
    span = (max(hi) - min(lo)) if lo else 1.0
    for a in (a_lab, a_cal, a_pl):
        a.set_ylim(-0.7, n - 0.3)
        a.set_yticks([])
        for sp in ("left", "right", "top"):
            a.spines[sp].set_visible(False)
    for r, y in zip(rows, ys):
        dtxt = "n/a" if r["d"] is None or not math.isfinite(r["d"]) else "%.2f" % r["d"]
        a_lab.text(0.0, y, r["label"], fontsize=6.9, ha="left", va="center", color=_INK,
                   fontweight="bold" if r.get("reference") else "normal")
        a_lab.text(1.0, y, "d %s%s" % (dtxt, r.get("flag", "")), fontsize=6.9, ha="right", va="center", color=_INK,
                   fontweight="bold" if r.get("reference") else "normal")
        if r["d"] is not None and math.isfinite(r["d"]) and math.isfinite(r["v0"]):
            x0, x1 = r["v0"] - 0.5 * r["d"], r["v0"] + 0.5 * r["d"]
            a_cal.plot([x0, x1], [y, y], color=r["colour"], ls=r["ls"], lw=2.0, solid_capstyle="butt")
            a_cal.plot([x0, x0], [y - 0.3, y + 0.3], color=r["colour"], lw=1.1)
            a_cal.plot([x1, x1], [y - 0.3, y + 0.3], color=r["colour"], lw=1.1)
        ks = [k for k in r["planes"] if k is not None]
        if ks:
            xs = np.arange(len(ks), dtype=float)
            jitter = 0.18 * (xs - 0.5 * (len(ks) - 1)) / max(len(ks) - 1, 1)
            a_pl.plot(np.asarray(ks, dtype=float), y + jitter, "-", color="#c9c8c3", lw=0.8, zorder=1)
            a_pl.plot(ks[:-1], (y + jitter)[:-1], "o", ms=3.0, color=_SEL_GREY, zorder=2)
        if r["k_iter"] is not None:
            a_pl.plot([r["k_iter"]], [y], "o", ms=5.5, mfc=r["colour"], mec=_INK if r["colour"] in _LIGHT + (
                "#e87ba4",) else r["colour"], mew=0.7, zorder=3)
    # kernel groups: a rule between groups, a heavier one under the reference
    prev = None
    for i, r in enumerate(rows):
        key = "ref" if r.get("reference") else r.get("kernel")
        if prev is not None and key != prev:
            for a in (a_lab, a_cal, a_pl):
                a.axhline(ys[i] + 0.5, color="#1a1a19" if prev == "ref" else "#d9d8d3",
                          lw=1.0 if prev == "ref" else 0.6, zorder=0)
        prev = key
    a_lab.set_xlim(0, 1)
    a_lab.set_xticks([])
    a_lab.spines["bottom"].set_visible(False)
    a_cal.axvline(0.0, color="#d9d8d3", lw=0.8, zorder=0)
    if lo:
        pad = 0.08 * span + 0.05
        a_cal.set_xlim(min(lo) - pad, max(hi) + pad)
    a_cal.set_xlabel("edges v0 -+ d/2 (um from the pilot's fitted centre, across the branch), at each variant's "
                     "plane", fontsize=8)
    a_cal.tick_params(labelsize=7.5)
    kk = [int(k) for k in block["ks"]]
    a_pl.axvline(block["k_star"], color=_KSTAR, lw=1.1, zorder=0)
    a_pl.axvline(block["k_swc"], color=_INK, ls=":", lw=1.0, zorder=0)
    used = [k for r in rows for k in r["planes"] if k is not None] + [block["k_star"], block["k_swc"]]
    a_pl.set_xlim(min(used) - 1.0, max(used) + 1.0)
    a_pl.set_xticks([k for k in kk if min(used) - 1 <= k <= max(used) + 1])
    a_pl.tick_params(labelsize=7.5)
    a_pl.set_xlabel("plane k: each round's pick (grey), k_iter (filled)", fontsize=8)
    for a in (a_cal, a_pl):
        a.grid(axis="x", color="#eeede9", lw=0.6)
    a_lab.set_title("variant (kernel, weighting): d_iter (um)", fontsize=8, loc="left", pad=4)
    present = {r["colour"] for r in rows if not r.get("reference")}
    handles = [plt.Line2D([], [], color=_INK, lw=2.0, label="Cell 4j's own iteration (Block 5's fit)")]
    handles += [plt.Line2D([], [], color=c, lw=2.0, label=f) for f, c in FAMILY_COLOURS.items() if c in present]
    handles += [plt.Line2D([], [], color=_SEL_GREY, ls="-", lw=1.4, label="fixed shape"),
                plt.Line2D([], [], color=_SEL_GREY, ls="-.", lw=1.4, label="second fixed shape"),
                plt.Line2D([], [], color=_SEL_GREY, ls=":", lw=1.4, label="shape fitted"),
                plt.Line2D([], [], color=_KSTAR, lw=1.1, label="k* (Block 5's focus)"),
                plt.Line2D([], [], color=_INK, ls=":", lw=1.0, label="SWC plane")]
    fig.legend(handles=handles, loc="upper left", ncol=6, fontsize=7.3, frameon=False,
               bbox_to_anchor=(left / W, 1.0 - (top + h_axes + 0.5) / H))
    fig.text(left / W, 1.0 - (top + h_axes + 1.0) / H, "* a fit parameter at its bound; x no fit or failed; ~ IRLS "
             "not settled in 30 iterations", fontsize=7.3, ha="left", va="top", color="#3d3d3a")
    return fig


def _view_limits(img, extent, view):
    """The 1st and 99.5th percentiles of the plane's pixels inside view (x0, x1, y_bottom, y_top), for its grey scale."""
    x0e, x1e, ybe, yte = (float(x) for x in extent)
    H, W = img.shape
    cols = x0e + (np.arange(W) + 0.5) * (x1e - x0e) / W
    rows = yte + (np.arange(H) + 0.5) * (ybe - yte) / H
    inx = (cols >= min(view[0], view[1])) & (cols <= max(view[0], view[1]))
    iny = (rows >= min(view[2], view[3])) & (rows <= max(view[2], view[3]))
    sel = img[np.ix_(iny, inx)] if (inx.any() and iny.any()) else img
    return np.nanpercentile(sel, [1.0, 99.5])


def _draw_fitted_dendrite(ax, b, pe, colour, view, edge_half_um, lw_line=0.9, lw_edge=1.5):
    """A Cell 4m panel: the plane on the grey scale of the pixels in view, the line that chose it (black), the
    fitted dendrite's two edges as lines along the branch (e_u), of half-length edge_half_um about the points where
    they cross the line, and the SWC node (+); the axes limited to the view."""
    img = np.asarray(b["plane"], dtype=float)
    lo, hi = _view_limits(img, b["extent"], view)
    ax.imshow(img, cmap="gray", vmin=lo, vmax=hi, extent=b["extent"], origin="upper", interpolation="nearest")
    if b.get("line_xy") is not None:
        L = np.asarray(b["line_xy"], dtype=float)
        ax.plot(L[:, 0], L[:, 1], color="black", lw=lw_line, zorder=3,
                path_effects=[pe.Stroke(linewidth=lw_line + 1.8, foreground="white"), pe.Normal()])
    if b.get("edges_xy") is not None:
        e_u = np.asarray(b["e_u"], dtype=float)
        halo = [pe.Stroke(linewidth=lw_edge + 1.8, foreground="white"), pe.Normal()]
        for P in np.asarray(b["edges_xy"], dtype=float):
            T = np.array([P - edge_half_um * e_u, P + edge_half_um * e_u])
            ax.plot(T[:, 0], T[:, 1], color=colour, lw=lw_edge, solid_capstyle="butt", path_effects=halo, zorder=4)
    o = np.asarray(b["o"], dtype=float)
    ax.plot([o[0]], [o[1]], marker="+", ms=6, mew=1.2, color="black", zorder=6,
            path_effects=[pe.Stroke(linewidth=2.8, foreground="white"), pe.Normal()])
    ax.set_xlim(float(view[0]), float(view[1]))
    ax.set_ylim(float(view[2]), float(view[3]))
    ax.set_xticks([])
    ax.set_yticks([])


def variant_planes_figure(block, title=""):
    """One node of Cell 4m (D-047): for every variant, the plane its own
    iteration chose, with the line that chose it (black, through the SWC node
    +) and the fitted dendrite: its two edges, v = s + v0 -+ d / 2 on that line,
    drawn as lines along the branch (the kernel's colour); each panel titled
    with k_iter and d. The reference (Cell 4j's own iteration, Block 5's fit)
    on top, then one row per kernel and one column per weighting, every panel
    on the same view.

    block: dict with label, sublabel, panels (dicts as variant_stack_figure's rows plus plane (H, W) or None,
    line_xy, edges_xy, title), extent, o, e_u, view (x0, x1, y_bottom, y_top), kernels (row labels in order),
    weights (column labels in order). Presentation only. Returns the Figure."""
    import textwrap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    panels = block["panels"]
    kernels, weights = list(block["kernels"]), list(block["weights"])
    side, gap, head, lab_w, left, top, hdr = 1.55, 0.12, 0.30, 1.45, 0.3, 1.45, 0.42
    ncol = max(len(weights), 1)
    W = max(left + lab_w + ncol * side + (ncol - 1) * gap + 0.3, 9.0)
    H = top + (head + side) + hdr + len(kernels) * (head + side) + 0.95
    fig = plt.figure(figsize=(W, H))

    def ax_at(x_in, y_top_in, w_in, h_in):
        return fig.add_axes([x_in / W, 1.0 - (y_top_in + h_in) / H, w_in / W, h_in / H])

    def panel_top(r):                                   # row 0: the reference; rows 1..: the kernels, under the header
        return top + head + (0.0 if r == 0 else (head + side) + hdr + (r - 1) * (head + side))

    if title:
        fig.text(0.5, 1.0 - 0.12 / H, title, fontsize=10, ha="center", va="top")
    fig.text(left / W, 1.0 - 0.45 / H, "\n".join(textwrap.wrap(block["label"], 118)), fontsize=8.6,
             fontweight="bold", ha="left", va="top", linespacing=1.15)
    fig.text(left / W, 1.0 - 0.95 / H, "\n".join(textwrap.wrap(block["sublabel"], 140)), fontsize=7.6, ha="left",
             va="top", linespacing=1.15, color="#3d3d3a")
    view = block["view"]
    edge_half = 0.42 * abs(float(view[1]) - float(view[0])) / 2.0
    ref_ax = None
    for p in panels:
        if p.get("reference"):
            r, c = 0, 0
        else:
            r, c = 1 + kernels.index(p["kernel"]), weights.index(p["weight"])
        ax = ax_at(left + lab_w + c * (side + gap), panel_top(r), side, side)
        if p.get("plane") is None:
            ax.text(0.5, 0.5, "no plane", ha="center", va="center", fontsize=7, transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            b = dict(plane=p["plane"], extent=block["extent"], o=block["o"], e_u=block["e_u"],
                     line_xy=p.get("line_xy"), edges_xy=p.get("edges_xy"))
            _draw_fitted_dendrite(ax, b, pe, _INK if p.get("reference") else p["colour"], view, edge_half)
        ax.set_title(p["title"], fontsize=7.2, pad=2)
        if p.get("reference"):
            ref_ax = ax
            ax.text(1.08, 0.5, "Cell 4j's own iteration\n(Block 5's fit: Gaussian at its\nfixed width, unweighted)",
                    transform=ax.transAxes, fontsize=7.5, ha="left", va="center", color=_INK)
    if ref_ax is not None:
        _scale_bar(ref_ax, view, pe)                     # every panel has the same view
    y_hdr = panel_top(1) - head - 0.08
    for c, wl in enumerate(weights):
        fig.text((left + lab_w + c * (side + gap) + 0.5 * side) / W, 1.0 - y_hdr / H,
                 "unweighted" if wl == "none" else wl.replace("lam=", "lambda ") + " um", fontsize=8, ha="center",
                 va="bottom", fontweight="bold", color=_INK)
    for r, kl in enumerate(kernels):
        y_mid = panel_top(1 + r) + 0.5 * side
        fig.text((left + lab_w - 0.08) / W, 1.0 - y_mid / H, kl, fontsize=7.8, ha="right", va="center", color=_INK)
    note = ("Every panel: the plane k_iter of that variant's own iteration; the line that chose it (black, +-m d / 2 "
            "through the SWC node, +); the fitted dendrite's edges, v = s + v0 -+ d / 2 on that line, drawn along the "
            "branch (the kernel's colour); titled k_iter and d (um). * a fit parameter at its bound; x no fit or "
            "failed; ~ IRLS not settled.")
    fig.text(left / W, 1.0 - (H - 0.85) / H, "\n".join(textwrap.wrap(note, int(13.5 * (W - 2 * left)))), fontsize=7.3,
             ha="left", va="top", color="#3d3d3a", linespacing=1.2)
    return fig
