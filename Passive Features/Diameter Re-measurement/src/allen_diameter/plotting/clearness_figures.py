"""Figures of the clearness screen -- Block 41 in specs/SPEC.md (D-192 [orchestrator]; the user, 2026-11-19:
"a cell that shows me a set of judged clean nodes (both images at focus and across planes as well as the
relative measuring lines across planes)").

Plotting only: every array arrives prepared (Block 36's files and Block 40's result by contract); nothing
scientific is computed here and no input array is written to.

  figure_node    one node on one page: the patch at the sharpest plane with the measuring line and the lines
                 along the dendrite drawn on it; the profiles on those lines; the measuring line's profiles on
                 every recorded plane, overlaid; the screen's numbers; then one row of the patch at every
                 recorded plane on one grey scale and one row of the measuring line's profile at every plane on
                 one grey-level axis, with the half-depth width drawn
  figure_sheet   a contact sheet: one small panel per node, the patch at its sharpest plane with the line
  save           the PNG of a figure (cleaning_figures.save)

Conventions as in cleaning_figures: images (n_planes, H, W), row = y, column = x; extents in um
(plotting.figures.image_extent_um); profiles in grey levels on the offsets v (um). Coordinates in the image
panels are um relative to the line's origin (x to the right, y downwards as in the image).
Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from allen_diameter.plotting.cleaning_figures import (BLUE, GRID, IMAGE_PERCENTILES, INK, INK2, ORANGE, SURFACE,
                                                      _plt, _row_scale, _style, save)

__all__ = ["figure_node", "figure_sheet", "save"]

LINE_COLOUR, ALLEN_COLOUR = "#ffd23f", "#35d0ff"
NODE_REQUIRED = ("images", "extent", "ks", "k_index", "origin_xy", "y_hat", "e_u", "v_um", "profiles_gl", "B_gl",
                 "u_um", "strips_gl")
SHEET_MAX_COLS = 6


def _node(panel):
    """Validated arrays of one node (figure_node, figure_sheet)."""
    if not isinstance(panel, Mapping):
        raise ValueError("a node must be a mapping, got %r" % type(panel).__name__)
    for key in NODE_REQUIRED:
        if key not in panel:
            raise ValueError("the node lacks %r" % key)
    images = np.asarray(panel["images"], dtype=float)
    ks = np.asarray(panel["ks"]).reshape(-1)
    v = np.asarray(panel["v_um"], dtype=float).reshape(-1)
    P = np.asarray(panel["profiles_gl"], dtype=float)
    B = np.asarray(panel["B_gl"], dtype=float).reshape(-1)
    u = np.asarray(panel["u_um"], dtype=float).reshape(-1)
    strips = np.asarray(panel["strips_gl"], dtype=float)
    extent = np.asarray(panel["extent"], dtype=float)
    vecs = [np.asarray(panel[k], dtype=float).reshape(-1) for k in ("origin_xy", "y_hat", "e_u")]
    j = int(panel["k_index"])
    if images.ndim != 3 or images.shape[0] != ks.size or ks.size == 0:
        raise ValueError("images must be (n_planes, H, W) with one plane per entry of ks")
    if P.shape != (ks.size, v.size) or B.shape != (ks.size,) or strips.shape != (u.size, v.size) or u.size == 0:
        raise ValueError("profiles_gl must be (n_planes, n_v), B_gl (n_planes,), strips_gl (n_u, n_v)")
    if extent.shape != (4,) or any(x.shape != (2,) for x in vecs) or not np.all(np.isfinite(extent)) \
            or not all(np.all(np.isfinite(x)) for x in vecs):
        raise ValueError("extent must be 4 finite numbers; origin_xy, y_hat and e_u finite 2-vectors")
    if not 0 <= j < ks.size:
        raise ValueError("k_index %d is not a plane of ks" % j)
    out = dict(images=images, ks=ks, v=v, P=P, B=B, u=u, strips=strips, extent=extent, origin=vecs[0], y_hat=vecs[1],
               e_u=vecs[2], j=j, title=str(panel.get("title", "")), text=str(panel.get("text", "")),
               d_allen=None, x_lo=None, x_hi=None, focus=None)
    if panel.get("d_allen_um") is not None:
        d = float(panel["d_allen_um"])
        if not (np.isfinite(d) and d > 0):
            raise ValueError("d_allen_um must be positive and finite, got %r" % (panel["d_allen_um"],))
        out["d_allen"] = d
    for key in ("x_lo_um", "x_hi_um"):
        if panel.get(key) is not None:
            x = np.asarray(panel[key], dtype=float).reshape(-1)
            if x.shape != (ks.size,):
                raise ValueError("%s must be (n_planes,)" % key)
            out[key[:-3]] = x
    if panel.get("focus") is not None:
        f = np.asarray(panel["focus"]).reshape(-1)
        if f.shape != (ks.size,) or f.dtype != bool:
            raise ValueError("focus must be (n_planes,) bool")
        out["focus"] = f
    return out


def _spacing(u):
    """The spacing of the offsets along the axis (um); 0 for a single line."""
    return float(np.median(np.diff(u))) if u.size > 1 else 0.0


def _view(it, zoom_um):
    """(relative extent, view, vmin, vmax): the grey range from the pixels in view of every plane together."""
    ext = it["extent"] - np.array([it["origin"][0], it["origin"][0], it["origin"][1], it["origin"][1]])
    view = (-zoom_um, zoom_um, zoom_um, -zoom_um) if zoom_um is not None else tuple(ext)
    H, W = it["images"].shape[1:]
    xs, ys = np.linspace(ext[0], ext[1], W + 1), np.linspace(ext[3], ext[2], H + 1)
    cols = (xs[1:] > min(view[0], view[1])) & (xs[:-1] < max(view[0], view[1]))
    rows = (ys[1:] > min(view[2], view[3])) & (ys[:-1] < max(view[2], view[3]))
    seen = it["images"][:, rows][:, :, cols] if rows.any() and cols.any() else it["images"]
    return ext, view, seen


def _draw_image(ax, it, plane, ext, view, lo, hi, lines=True, strips=False):
    ax.imshow(it["images"][plane], cmap="gray", vmin=lo, vmax=hi, extent=tuple(ext), origin="upper",
              interpolation="nearest")
    r = float(np.max(np.abs(it["v"])))
    if lines:
        ax.plot([-r * it["y_hat"][0], r * it["y_hat"][0]], [-r * it["y_hat"][1], r * it["y_hat"][1]],
                color=LINE_COLOUR, lw=0.9, ls=":", gid="line")
    if strips:
        for uu in it["u"]:
            if uu == 0.0:
                continue
            c = uu * it["e_u"]
            ax.plot([c[0] - r * it["y_hat"][0], c[0] + r * it["y_hat"][0]],
                    [c[1] - r * it["y_hat"][1], c[1] + r * it["y_hat"][1]], color=LINE_COLOUR, lw=0.5, alpha=0.55,
                    gid="strip")
        if it["d_allen"] is not None:
            n_hat, far = it["y_hat"], float(np.max(np.abs(it["u"]))) if it["u"].size > 1 else r
            for s in (-0.5, 0.5):
                a = s * it["d_allen"] * n_hat - far * it["e_u"]
                b = s * it["d_allen"] * n_hat + far * it["e_u"]
                ax.plot([a[0], b[0]], [a[1], b[1]], color=ALLEN_COLOUR, lw=0.9, ls="--", gid="allen")
    ax.set_xlim(view[0], view[1])
    ax.set_ylim(view[2], view[3])
    ax.set_aspect("equal")


def figure_node(node, zoom_um=5.0, percentiles=IMAGE_PERCENTILES):
    """One node of the gallery (Block 41). Returns the matplotlib Figure.

    node: a mapping --
      images (n_planes, H, W) grey levels, the node's recorded patch; extent (x_left, x_right, y_bottom, y_top)
      um; ks (n_planes,) the planes; k_index: the sharpest plane's index; origin_xy (2,) um, the measuring
      line's origin; y_hat, e_u (2,): the line's and the axis' unit directions; v_um (n_v,); profiles_gl
      (n_planes, n_v): the measuring line as recorded; B_gl (n_planes,): the backgrounds; u_um (n_u,) and
      strips_gl (n_u, n_v): the lines along the dendrite on the sharpest plane (Block 40);
      optional: d_allen_um (> 0; drawn as two dashed edges about the origin and as a bar in the profile
      panels); x_lo_um, x_hi_um (n_planes,): the half-depth crossings (the bar of the half-depth width; NaN:
      none); focus (n_planes,) bool: the planes the screen judged (their titles are marked); title; text (the
      screen's numbers, drawn as given).
    zoom_um: > 0, the image panels show +-zoom_um about the origin; None = the whole patch.
    percentiles: (lo, hi) of the grey levels in view over all planes give the one grey range; None = full range.

    ValueError for a node that is not a mapping or lacks a required key, shapes that disagree, a d_allen_um
    not positive and finite, a zoom_um not positive."""
    it = _node(node)
    if zoom_um is not None and not (np.isfinite(float(zoom_um)) and float(zoom_um) > 0):
        raise ValueError("zoom_um must be > 0 or None, got %r" % (zoom_um,))
    zoom = None if zoom_um is None else float(zoom_um)
    ext, view, seen = _view(it, zoom)
    lo, hi = _row_scale((seen,), percentiles)
    n_k, j, v = it["ks"].size, it["j"], it["v"]
    plt = _plt()
    from matplotlib import colormaps
    fig = plt.figure(figsize=(max(15.0, 1.75 * n_k), 9.6), facecolor=SURFACE)
    outer = fig.add_gridspec(3, 1, height_ratios=(3.0, 1.6, 1.7), left=0.045, right=0.99, top=0.93, bottom=0.085,
                             hspace=0.24)
    top = outer[0].subgridspec(1, 4, width_ratios=(1.0, 1.05, 1.05, 0.8), wspace=0.22)
    a_img, a_str, a_all, a_txt = (fig.add_subplot(top[0, c]) for c in range(4))
    g_img = outer[1].subgridspec(1, n_k, wspace=0.04)
    g_pro = outer[2].subgridspec(1, n_k, wspace=0.06)

    _draw_image(a_img, it, j, ext, view, lo, hi, strips=True)
    a_img.set_title("sharpest plane k* = %d" % int(it["ks"][j]), fontsize=9, color=INK)
    a_img.set_xlabel("x - x_line (um)", fontsize=8, color=INK2)
    a_img.set_ylabel("y - y_line (um)", fontsize=8, color=INK2)
    a_img.tick_params(labelsize=7, colors=INK2)

    cm_u, cm_k = colormaps["viridis"], colormaps["coolwarm"]
    u_max = float(np.max(np.abs(it["u"]))) or 1.0
    for m, uu in enumerate(it["u"]):
        if uu == 0.0:
            continue
        ends = m in (0, it["u"].size - 1)                  # the legend names the two outermost lines only
        a_str.plot(v, it["strips"][m], color=cm_u(0.5 + 0.5 * uu / u_max), lw=1.0,
                   label=("u = %+.1f um" % uu) if ends else None, gid="strip%d" % m)
    zero = np.flatnonzero(it["u"] == 0.0)
    if zero.size:
        a_str.plot(v, it["strips"][int(zero[0])], color=INK, lw=1.6, label="u = 0, the measuring line", gid="strip0")
    a_str.axhline(it["B"][j], color=ORANGE, lw=0.8, ls="--", gid="background")
    a_str.set_title("k*: the %d lines along the dendrite, %.1f um apart" % (it["u"].size, _spacing(it["u"])),
                    fontsize=9, color=INK)

    k_span = float(max(j, n_k - 1 - j)) or 1.0
    for i in range(n_k):
        if i == j:
            continue
        a_all.plot(v, it["P"][i], color=cm_k(0.5 + 0.5 * (i - j) / k_span), lw=1.0,
                   label=("k* %+d" % (i - j)) if i in (0, n_k - 1) else None, gid="plane%d" % i)
    a_all.plot(v, it["P"][j], color=INK, lw=1.6, label="k*", gid="plane%d" % j)
    a_all.set_title("the measuring line on the %d planes" % n_k, fontsize=9, color=INK)
    for ax in (a_str, a_all):
        _style(ax)
        ax.legend(fontsize=7, frameon=False, loc="lower left")
        ax.set_xlabel("v (um)", fontsize=8, color=INK2)
        ax.set_ylabel("grey level", fontsize=8, color=INK2)

    a_txt.set_axis_off()
    a_txt.text(0.0, 1.0, it["text"], va="top", ha="left", fontsize=7.5, color=INK, family="monospace",
               transform=a_txt.transAxes, gid="text")

    y_lo = float(np.nanmin(it["P"])) if np.any(np.isfinite(it["P"])) else 0.0
    y_hi = float(np.nanmax(np.concatenate([it["P"].ravel(), it["B"]])))
    pad = 0.06 * (y_hi - y_lo if y_hi > y_lo else 1.0)
    for i in range(n_k):
        ax = fig.add_subplot(g_img[0, i])
        _draw_image(ax, it, i, ext, view, lo, hi)
        mark = " *" if (it["focus"] is not None and it["focus"][i]) else ""
        ax.set_title("k = %d%s" % (int(it["ks"][i]), " (k*)" if i == j else mark), fontsize=8,
                     color=ORANGE if i == j else INK)
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_color(ORANGE if i == j else INK2)
            side.set_linewidth(1.6 if i == j else 0.6)
        ax = fig.add_subplot(g_pro[0, i])
        _style(ax)
        ax.plot(v, it["P"][i], color=INK, lw=0.9, marker="o", ms=1.8, gid="profile")
        ax.axhline(it["B"][i], color=ORANGE, lw=0.8, ls="--", gid="background")
        if it["x_lo"] is not None and it["x_hi"] is not None and np.isfinite(it["x_lo"][i]) and np.isfinite(it["x_hi"][i]):
            ax.plot([it["x_lo"][i], it["x_hi"][i]], [y_lo - 0.35 * pad] * 2, color=BLUE, lw=2.2,
                    solid_capstyle="butt", gid="fwhm")
        if it["d_allen"] is not None:
            ax.plot([-0.5 * it["d_allen"], 0.5 * it["d_allen"]], [y_lo - 0.8 * pad] * 2, color=ALLEN_COLOUR, lw=2.2,
                    solid_capstyle="butt", gid="allen")
        ax.set_ylim(y_lo - 1.2 * pad, y_hi + pad)
        ax.set_xlabel("v (um)", fontsize=7, color=INK2)
        if i == 0:
            ax.set_ylabel("grey level", fontsize=7.5, color=INK2)
        else:
            ax.tick_params(labelleft=False)
    fig.suptitle(it["title"], fontsize=10.5, color=INK, y=0.985)
    fig.text(0.5, 0.012, "dashed orange: the background B_k;  blue bar: the half-depth width of the smoothed dip;  "
             "light-blue bar and dashed edges: Allen's diameter;  dotted yellow: the measuring line, thin yellow: "
             "the lines along the dendrite;  * = a plane the screen judged", ha="center", va="bottom", fontsize=7.5,
             color=INK2)
    return fig


def figure_sheet(nodes, ncols=None, zoom_um=5.0, percentiles=IMAGE_PERCENTILES):
    """A contact sheet (Block 41): one panel per node of `nodes` (mappings as figure_node's), its patch at the
    sharpest plane with the measuring line and Allen's edges, its `title` above. ncols: panels per row (None:
    at most SHEET_MAX_COLS). Returns the Figure. ValueError for an empty sequence or a bad ncols or zoom_um."""
    if isinstance(nodes, Mapping) or not hasattr(nodes, "__len__") or len(nodes) == 0:
        raise ValueError("nodes must be a non-empty sequence of mappings")
    items = [_node(n) for n in nodes]
    if zoom_um is not None and not (np.isfinite(float(zoom_um)) and float(zoom_um) > 0):
        raise ValueError("zoom_um must be > 0 or None, got %r" % (zoom_um,))
    if ncols is None:
        ncols = min(len(items), SHEET_MAX_COLS)
    if int(ncols) != ncols or int(ncols) < 1:
        raise ValueError("ncols must be a positive integer or None, got %r" % (ncols,))
    ncols = int(ncols)
    nrows = (len(items) + ncols - 1) // ncols
    plt = _plt()
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.7 * ncols + 0.3, 3.0 * nrows + 0.4), facecolor=SURFACE,
                             squeeze=False, layout="constrained")
    zoom = None if zoom_um is None else float(zoom_um)
    for i, it in enumerate(items):
        ax = axes[i // ncols, i % ncols]
        ext, view, seen = _view(it, zoom)
        lo, hi = _row_scale((seen[it["j"]],), percentiles)
        _draw_image(ax, it, it["j"], ext, view, lo, hi, strips=True)
        ax.set_title(it["title"], fontsize=7.5, color=INK)
        ax.tick_params(labelsize=6, colors=INK2)
    for i in range(len(items), nrows * ncols):
        axes[i // ncols, i % ncols].set_axis_off()
    return fig


_ = GRID   # the palette's grid colour is applied by _style
