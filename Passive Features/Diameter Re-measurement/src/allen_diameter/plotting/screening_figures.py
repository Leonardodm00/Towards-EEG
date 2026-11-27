"""The screening sheets: each plane's image beside its intensity profile -- Block 62 in specs/SPEC.md
(section 5, "The screening export"; D-226.3, D-227, D-261.1). Figures only: every number drawn is passed in
(a NodeView of Block 61 and the values of its diameter_marks and image_extent); nothing scientific is computed
here (section 4). The grey scale (1st and 99th percentiles of the valid planes' pixels) and the shared y-range
are display choices of the entry.

  node_sheet        one pair of panels per plane, in the order of view.ks: the plane's image (shared grey
                    scale, equal aspect) and its profile I_k(v) (solid) with the scan's background (dashed),
                    one y-range for all the node's planes. A near view as one pair per row (7 rows), a far view
                    as three pairs per row (11 rows), row-major. On the plane k* only: Allen's diameter as two
                    marks on the image and two vertical lines on the profile, and an orange frame.
  write_node_sheet  node_sheet saved as a PNG, the figure closed.

Artists carry gids for the tests: "image", "not_fetched", "profile", "background", "allen_marks" (the image's
two points, one Line2D), "allen_v" (the profile's two vertical lines).
Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import math

import numpy as np

INK, INK2 = "#0b0b0b", "#52514e"
BLUE, ORANGE, SURFACE, GRID = "#2a78d6", "#eb6834", "#fcfcfb", "#e4e3df"
GREY_PERCENTILES = (1.0, 99.0)    # SPEC Block 62: the grey scale of all the node's planes
PAIRS_PER_ROW = {"near": 1, "far": 3}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _style_profile(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=7)


def _grey_scale(stack, valid):
    """(vmin, vmax): the 1st and 99th percentiles of the valid planes' pixels; (0, 255) when none is valid."""
    if not np.any(valid):
        return 0.0, 255.0
    lo, hi = np.percentile(np.asarray(stack)[valid].astype(np.float64), GREY_PERCENTILES)
    lo, hi = float(lo), float(hi)
    return lo, (hi if hi > lo else lo + 1.0)


def _y_range(profile, background):
    """One y-range for every profile panel: the finite values of both curves with a 5 % margin (display)."""
    vals = np.concatenate([np.ravel(profile), np.ravel(background)]).astype(np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    lo, hi = float(vals.min()), float(vals.max())
    pad = 0.05 * (hi - lo) if hi > lo else 1.0
    return lo - pad, hi + pad


def _pair_title(k, k_star, dz_um, status):
    o = int(k) - int(k_star)
    text = "k* %+d (%+.2f um)" % (o, o * float(dz_um))
    if str(status) != "ok":
        text += "  [%s]" % status
    return text


def node_sheet(view, marks, extent, dz_um, title=""):
    """The Figure of one node (SPEC Block 62): one pair of panels per plane in the order of view.ks -- the image
    (shared grey scale, extent = (x0, x1, y0, y1) um from screening.image_extent) and the profile I_k(v) with the
    background dashed (shared y-range); a near view as 7 rows of one pair, a far view as 11 rows of three pairs;
    Allen's diameter marks (marks = screening.diameter_marks(view)) on the plane k* only. ValueError when view.ks
    does not contain view.k_star."""
    ks = np.asarray(view.ks)
    k_star = int(view.k_star)
    hits = np.flatnonzero(ks == k_star)
    if hits.size == 0:
        raise ValueError("node %s: view.ks %s does not contain k_star %d" % (view.node_id, ks.tolist(), k_star))
    if view.set not in PAIRS_PER_ROW:
        raise ValueError("node %s: view.set must be 'near' or 'far', got %r" % (view.node_id, view.set))
    q_star = int(hits[0])
    n = int(ks.size)
    stack = np.asarray(view.stack)
    valid = np.asarray(view.valid, dtype=bool)
    v = np.asarray(view.v_um, dtype=np.float64)
    profile = np.asarray(view.profile_gl, dtype=np.float64)
    background = np.asarray(view.background_gl, dtype=np.float64)
    lo, hi = _grey_scale(stack, valid)
    y_lo, y_hi = _y_range(profile, background)
    x0, x1, ya, yb = (float(e) for e in extent)
    y_top, y_bot = min(ya, yb), max(ya, yb)           # y increases downwards (image frame)
    im_extent = (x0, x1, y_bot, y_top)                 # imshow(origin="upper"): row 0 at y_top
    xy = np.asarray(marks["xy_um"], dtype=np.float64)  # (2, 2): one row per point, (x, y) um
    v_marks = [float(t) for t in marks["v_um"]]

    per_row = PAIRS_PER_ROW[view.set]
    n_rows = int(math.ceil(n / float(per_row)))
    plt = _plt()
    fig = plt.figure(figsize=(3.9 * per_row + 0.6, 1.75 * n_rows + 0.9), facecolor=SURFACE)
    gs = fig.add_gridspec(n_rows, 2 * per_row, width_ratios=[1.0, 1.45] * per_row,
                          left=0.07 if per_row == 1 else 0.035, right=0.985,
                          top=1.0 - 0.75 / (1.75 * n_rows + 0.9), bottom=0.45 / (1.75 * n_rows + 0.9),
                          wspace=0.32, hspace=0.62)
    for q in range(n):
        r, c = divmod(q, per_row)
        a_im = fig.add_subplot(gs[r, 2 * c])
        a_pr = fig.add_subplot(gs[r, 2 * c + 1])
        star = q == q_star
        # image
        if valid[q]:
            a_im.imshow(stack[q], cmap="gray", vmin=lo, vmax=hi, extent=im_extent, origin="upper",
                        interpolation="nearest").set_gid("image")
        else:
            a_im.set_facecolor("#efeeea")
            a_im.text(0.5, 0.5, "not fetched", transform=a_im.transAxes, ha="center", va="center",
                      fontsize=8, color=INK2, gid="not_fetched")
        a_im.set_xlim(x0, x1)
        a_im.set_ylim(y_bot, y_top)
        a_im.set_aspect("equal")
        a_im.tick_params(colors=INK2, labelsize=6)
        # profile
        _style_profile(a_pr)
        a_pr.plot(v, profile[q], color=BLUE, lw=1.1, ls="-", gid="profile")
        a_pr.plot(v, background[q], color=INK2, lw=0.9, ls="--", gid="background")
        a_pr.set_ylim(y_lo, y_hi)
        if v.size:
            a_pr.set_xlim(float(np.nanmin(v)), float(np.nanmax(v)))
        if r == n_rows - 1 or q + per_row >= n:
            a_pr.set_xlabel("v (um)", fontsize=7, color=INK2)
        if c == 0:
            a_pr.set_ylabel("I (gl)", fontsize=7, color=INK2)
        # the pair's title, over the image, left-aligned, running over the profile panel
        a_im.set_title(_pair_title(ks[q], k_star, dz_um, view.status[q]), loc="left", fontsize=8,
                       color=ORANGE if star else INK, fontweight="bold" if star else "normal")
        if star:
            a_im.plot(xy[:, 0], xy[:, 1], ls="none", marker="o", ms=5, mfc="none", mec=ORANGE, mew=1.4,
                      gid="allen_marks")
            for vm in v_marks:
                a_pr.axvline(vm, color=ORANGE, lw=1.0, ls=":", gid="allen_v")
            for ax in (a_im, a_pr):
                for side in ("left", "bottom", "top", "right"):
                    ax.spines[side].set_visible(True)
                    ax.spines[side].set_color(ORANGE)
                    ax.spines[side].set_linewidth(1.8)
    if not title:
        title = "cell %s, node %s (%s), Allen's diameter %.3g um" % (view.cell_id, view.node_id, view.set,
                                                                     float(view.allen_diameter_um))
    fig.suptitle(title, fontsize=10, color=INK, y=1.0 - 0.2 / (1.75 * n_rows + 0.9), va="top")
    return fig


def write_node_sheet(view, marks, extent, dz_um, path, dpi, title="") -> None:
    """node_sheet saved to path (PNG) at dpi, the figure closed."""
    plt = _plt()
    fig = node_sheet(view, marks, extent, dz_um, title=title)
    try:
        fig.savefig(str(path), dpi=dpi, format="png", facecolor=fig.get_facecolor())
    finally:
        plt.close(fig)
