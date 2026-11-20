"""Figures of the in-focus width calibrated on thick nodes -- Block 43 in specs/SPEC.md (D-201).

Plotting only: every array arrives prepared (Block 42's SigmaFit and summarize, and Block 36's per-node arrays,
through the entry point, by contract); nothing scientific is computed here and no input is written to. The one
transform drawn, the residual profile - model (gl), is a display of the inputs. Not to be confused with
calibration_figures (Block 10.F, the kernel calibration of Cell 7).

  figure_node_fit  one node on one page: per fitted plane a column of three panels -- the patch (one grey range
                   over the planes, the view +-zoom_um about the line's origin, the measuring line drawn, the
                   sharpest plane framed); the recorded profile (points) with the fitted model (line) and the
                   background (dashed), one grey-level range for every plane; the residual profile - model (gl)
                   with its zero line, another range shared by every plane -- and a text panel with `text`
  figure_summary   the cell's fitted in-focus widths and growth scales: sigma0 +- SE per node sorted by value with
                   the weighted mean, its +-SE band and the median; c +- SE per node in the same order; sigma0
                   against the fitted diameter (Allen's diameter marked) and against depth; not-ok nodes hollow
  save             the PNG of a figure (cleaning_figures.save)

Conventions as in clearness_figures (Block 41): images (n_planes, H, W), row = y, column = x; extents in um
(plotting.figures.image_extent_um); in the image panels the coordinates are um relative to the line's origin
(x to the right, y downwards as in the image); profiles in grey levels on the offsets v (um) of Block 5.
Colours: the palette of cleaning_figures. Identity never rests on colour alone: profile = points, model = line,
background = dashed; ok = filled, not ok = hollow; fitted diameter = circle, Allen's = cross.
Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from allen_diameter.plotting.cleaning_figures import (BLUE, IMAGE_PERCENTILES, INK, INK2, ORANGE, SURFACE, _plt,
                                                      _row_scale, _style, save)

__all__ = ["figure_node_fit", "figure_summary", "save"]

LINE_COLOUR = "#ffd23f"   # the measuring line on the images, as Block 41's
NODE_REQUIRED = ("images", "extent", "ks", "k_index", "origin_xy", "y_hat", "v_um", "profiles_gl", "model_gl")
ROW_KEYS = ("node_id", "sigma0_um", "se_sigma0_um", "growth_scale", "se_growth_scale", "d_hat_um", "d_allen_um",
            "depth_um", "ok")
SUMMARY_KEYS = ("n", "median_um", "mad_um", "ivw_mean_um", "ivw_se_um", "chi2_cons", "dof")


def _fmt(value):
    """A number as %.4g (NaN as nan), anything else as str."""
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return "%d" % int(value)
    try:
        return "%.4g" % float(value)
    except (TypeError, ValueError):
        return str(value)


def _text(value):
    return "" if value is None else str(value)


# ------------------------------------------------------------ one node ---

def _node(node):
    """Validated arrays of figure_node_fit's node (SPEC Block 43)."""
    if not isinstance(node, Mapping):
        raise ValueError("node must be a mapping, got %r" % type(node).__name__)
    missing = [key for key in NODE_REQUIRED if key not in node]
    if missing:
        raise ValueError("node lacks the key(s) %r" % (missing,))
    ks = np.asarray(node["ks"])
    if ks.ndim != 1 or ks.size == 0:
        raise ValueError("ks must be (n_planes,) with n_planes >= 1, got shape %r" % (ks.shape,))
    n_k = ks.size
    images = np.asarray(node["images"], dtype=float)
    if images.ndim != 3 or images.shape[0] != n_k or images.shape[1] == 0 or images.shape[2] == 0:
        raise ValueError("images must be (n_planes, H, W) = (%d, H, W), got shape %r" % (n_k, images.shape))
    v = np.asarray(node["v_um"], dtype=float)
    if v.ndim != 1 or v.size < 2 or not np.all(np.isfinite(v)):
        raise ValueError("v_um must be (n_v,), finite, with n_v >= 2, got shape %r" % (v.shape,))
    arrays = {}
    for key in ("profiles_gl", "model_gl", "background_gl"):
        if key == "background_gl" and node.get(key) is None:
            arrays[key] = None
            continue
        arr = np.asarray(node[key], dtype=float)
        if arr.shape != (n_k, v.size):
            raise ValueError("%s must be (n_planes, n_v) = (%d, %d), got %r" % (key, n_k, v.size, arr.shape))
        arrays[key] = arr
    extent = np.asarray(node["extent"], dtype=float)
    origin = np.asarray(node["origin_xy"], dtype=float)
    y_hat = np.asarray(node["y_hat"], dtype=float)
    if extent.shape != (4,) or origin.shape != (2,) or y_hat.shape != (2,):
        raise ValueError("extent must be (4,), origin_xy and y_hat (2,), got %r, %r, %r"
                         % (extent.shape, origin.shape, y_hat.shape))
    if not (np.all(np.isfinite(extent)) and np.all(np.isfinite(origin)) and np.all(np.isfinite(y_hat))):
        raise ValueError("extent, origin_xy and y_hat must be finite")
    k_raw = node["k_index"]
    try:
        j = int(k_raw)
        integral = bool(j == k_raw)
    except (TypeError, ValueError):
        integral = False
    if not integral or not 0 <= j < n_k:
        raise ValueError("k_index must be an integer index of the %d planes, got %r" % (n_k, k_raw))
    return dict(images=images, ks=ks, v=v, P=arrays["profiles_gl"], M=arrays["model_gl"], bg=arrays["background_gl"],
                extent=extent, origin=origin, y_hat=y_hat, j=j, title=_text(node.get("title")),
                text=_text(node.get("text")))


def _view(images, ext, zoom):
    """(view, seen): the image panels' limits (x_left, x_right, y_bottom, y_top), um relative to the line's origin
    (+-zoom, or the whole patch for None), and the pixels of every plane that overlap them (n_planes, h, w), whose
    percentiles give the one grey range (as Block 41's figure_node; all pixels when none overlaps)."""
    view = (-zoom, zoom, zoom, -zoom) if zoom is not None else tuple(ext)
    H, W = images.shape[1:]
    xs = np.linspace(ext[0], ext[1], W + 1)          # pixel edges, left to right
    ys = np.linspace(ext[3], ext[2], H + 1)          # pixel edges, top row to bottom row
    cols = (xs[1:] > min(view[0], view[1])) & (xs[:-1] < max(view[0], view[1]))
    rows = (ys[1:] > min(view[2], view[3])) & (ys[:-1] < max(view[2], view[3]))
    seen = images[:, rows][:, :, cols] if rows.any() and cols.any() else images
    return view, seen


def _finite_range(*arrays):
    """(lo, hi) over the finite values of the arrays given (None skipped), padded by 6 %; (0, 1) when none."""
    vals = np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays if a is not None])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    lo, hi = float(vals.min()), float(vals.max())
    pad = 0.06 * (hi - lo) if hi > lo else max(1.0, 0.06 * abs(hi))
    return lo - pad, hi + pad


def _symmetric_range(arr):
    """(-a, a), a = 1.1 max |finite value|; (-1, 1) when there is none above 0."""
    vals = np.abs(np.asarray(arr, dtype=float).ravel())
    vals = vals[np.isfinite(vals)]
    a = 1.1 * float(vals.max()) if vals.size else 0.0
    return (-a, a) if a > 0 else (-1.0, 1.0)


def figure_node_fit(node, zoom_um=5.0):
    """One node's fit on its planes (SPEC Block 43). Returns the matplotlib Figure.

    node: a mapping --
      images (n_planes, H, W) grey levels, the node's recorded patch on the fitted planes; extent (x_left,
      x_right, y_bottom, y_top) um (figures.image_extent_um); ks (n_planes,) the planes; k_index: the sharpest
      plane's index in ks; origin_xy (2,) um, the measuring line's origin; y_hat (2,): the line's direction;
      v_um (n_v,) the line's offsets (um); profiles_gl (n_planes, n_v): the recorded profiles; model_gl
      (n_planes, n_v) gl: the fitted model (Block 42's model_gl);
      optional: background_gl (n_planes, n_v) gl, the fitted background (drawn dashed); title; text (the fit's
      numbers, drawn as given in a monospace panel).
    zoom_um: > 0, the image panels show +-zoom_um about the line's origin; None = the whole patch.

    Layout: one column per plane -- row 1 the patch on one grey range (numpy's percentiles IMAGE_PERCENTILES of
    the pixels in view over all planes, as Block 41's), the measuring line drawn (dotted) from v_um's first to
    its last offset, the sharpest plane framed; row 2 the profile (points), the model (line) and the background
    (dashed) on one y range; row 3 the residual profile - model (gl) on one y range symmetric about 0, with the
    zero line -- and a text panel on the right. NaN samples are left as gaps.
    ValueError for a node that is not a mapping or lacks a required key, shapes that disagree, extent, origin_xy
    or y_hat not finite, k_index not an index of the planes, zoom_um not > 0."""
    it = _node(node)
    if zoom_um is not None:
        try:
            zoom = float(zoom_um)
        except (TypeError, ValueError):
            zoom = float("nan")
        if not (np.isfinite(zoom) and zoom > 0):
            raise ValueError("zoom_um must be > 0 or None, got %r" % (zoom_um,))
    else:
        zoom = None
    images, ks, v, P, M, bg, j = it["images"], it["ks"], it["v"], it["P"], it["M"], it["bg"], it["j"]
    n_k = ks.size
    ox, oy = it["origin"]
    ext = it["extent"] - np.array([ox, ox, oy, oy])
    view, seen = _view(images, ext, zoom)
    lo, hi = _row_scale((seen,), IMAGE_PERCENTILES)
    resid = P - M                                    # display of the inputs (gl), SPEC Block 43 row 3
    y_lo, y_hi = _finite_range(P, M, bg)
    r_lo, r_hi = _symmetric_range(resid)
    v_lo, v_hi = float(v.min()), float(v.max())
    line_x = [v_lo * it["y_hat"][0], v_hi * it["y_hat"][0]]
    line_y = [v_lo * it["y_hat"][1], v_hi * it["y_hat"][1]]

    plt = _plt()
    import matplotlib.patheffects as pe
    from matplotlib.lines import Line2D
    halo = [pe.Stroke(linewidth=2.8, foreground="black", alpha=0.45), pe.Normal()]   # visible on a light patch
    fig = plt.figure(figsize=(2.35 * n_k + 3.3, 8.2), facecolor=SURFACE, layout="constrained")
    gs = fig.add_gridspec(3, n_k + 1, width_ratios=[1.0] * n_k + [1.35], height_ratios=(1.25, 1.0, 0.72))
    for i in range(n_k):
        star = i == j
        a_im = fig.add_subplot(gs[0, i])
        a_im.imshow(images[i], cmap="gray", vmin=lo, vmax=hi, extent=tuple(ext), origin="upper",
                    interpolation="nearest")
        a_im.plot(line_x, line_y, color=LINE_COLOUR, lw=1.2, ls=":", path_effects=halo, gid="line")
        a_im.set_xlim(view[0], view[1])
        a_im.set_ylim(view[2], view[3])
        a_im.set_aspect("equal")
        a_im.set_title("k = %d  (%s)" % (int(ks[i]), "k*" if star else "k* %+d" % (i - j)), fontsize=8.5,
                       color=ORANGE if star else INK)
        for side in a_im.spines.values():
            side.set_color(ORANGE if star else INK2)
            side.set_linewidth(2.0 if star else 0.6)
        if i == 0:
            a_im.tick_params(labelsize=6.5, colors=INK2)
            a_im.set_xlabel("x - x_line (um)", fontsize=7.5, color=INK2)
            a_im.set_ylabel("y - y_line (um)\ngrey %.0f-%.0f gl" % (lo, hi), fontsize=7.5, color=INK2)
        else:
            a_im.set_xticks([])
            a_im.set_yticks([])

        a_p = fig.add_subplot(gs[1, i])
        _style(a_p)
        if bg is not None:
            a_p.plot(v, bg[i], color=ORANGE, lw=1.0, ls="--", gid="background")
        a_p.plot(v, P[i], ls="none", marker="o", ms=2.4, color=INK, gid="profile")
        a_p.plot(v, M[i], ls="-", lw=1.3, color=BLUE, gid="model")
        a_p.set_xlim(v_lo, v_hi)
        a_p.set_ylim(y_lo, y_hi)

        a_r = fig.add_subplot(gs[2, i])
        _style(a_r)
        a_r.axhline(0.0, color=ORANGE, lw=0.9, gid="zero")
        a_r.plot(v, resid[i], ls="-", lw=0.6, marker=".", ms=2.8, color=INK, gid="residual")
        a_r.set_xlim(v_lo, v_hi)
        a_r.set_ylim(r_lo, r_hi)
        a_r.set_xlabel("v (um)", fontsize=7.5, color=INK2)
        if i == 0:
            a_p.set_ylabel("grey level", fontsize=7.5, color=INK2)
            a_r.set_ylabel("profile - model (gl)", fontsize=7.5, color=INK2)
        else:
            a_p.tick_params(labelleft=False)
            a_r.tick_params(labelleft=False)

    a_t = fig.add_subplot(gs[:, n_k])
    a_t.set_axis_off()
    a_t.text(0.0, 1.0, it["text"], va="top", ha="left", fontsize=7.5, color=INK, family="monospace",
             transform=a_t.transAxes, gid="text")
    handles = [Line2D([], [], ls="none", marker="o", ms=3.5, color=INK, label="profile (recorded)"),
               Line2D([], [], color=BLUE, lw=1.3, label="model (fitted)")]
    if bg is not None:
        handles.append(Line2D([], [], color=ORANGE, lw=1.0, ls="--", label="background"))
    handles += [Line2D([], [], color=ORANGE, lw=0.9, label="residual's zero line"),
                Line2D([], [], color=LINE_COLOUR, lw=1.2, ls=":", path_effects=halo,
                       label="the measuring line (images)"),
                Line2D([], [], ls="none", marker="s", ms=7, mfc="none", mec=ORANGE, mew=1.6,
                       label="framed: the sharpest plane k*")]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), fontsize=7.5, frameon=False)
    if it["title"]:
        fig.suptitle(it["title"], fontsize=10, color=INK)
    return fig


# ----------------------------------------------------------- the summary ---

def _number(where, key, value):
    """float(value), None as NaN; ValueError for what is not a number (a string included)."""
    if value is None:
        return float("nan")
    if isinstance(value, (str, bytes)):
        raise ValueError("%s: %s must be a number or None, got the string %r" % (where, key, value))
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError("%s: %s must be a number or None, got %r" % (where, key, value)) from None


def _rows(rows):
    """Validated columns of figure_summary's rows (SPEC Block 43): node_id as str, ok as bool, the rest float."""
    if isinstance(rows, (Mapping, str, bytes)) or rows is None or not hasattr(rows, "__len__") or len(rows) == 0:
        raise ValueError("rows must be a non-empty sequence of mappings")
    cols = {key: [] for key in ROW_KEYS}
    for i, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError("row %d is not a mapping, got %r" % (i, type(row).__name__))
        missing = [key for key in ROW_KEYS if key not in row]
        if missing:
            raise ValueError("row %d lacks the key(s) %r" % (i, missing))
        if not isinstance(row["ok"], (bool, np.bool_)):
            raise ValueError("row %d: ok must be a bool, got %r" % (i, row["ok"]))
        cols["ok"].append(bool(row["ok"]))
        cols["node_id"].append(str(row["node_id"]))
        for key in ROW_KEYS[1:-1]:
            cols[key].append(_number("row %d" % i, key, row[key]))
        for key in ("se_sigma0_um", "se_growth_scale"):
            if cols[key][-1] < 0:
                raise ValueError("row %d: %s must be >= 0 (or NaN), got %r" % (i, key, row[key]))
    out = {key: np.asarray(cols[key], dtype=float) for key in ROW_KEYS[1:-1]}
    out["ok"] = np.asarray(cols["ok"], dtype=bool)
    out["node_id"] = cols["node_id"]
    return out


def _summary(summary):
    """Validated copy of Block 42's summarize mapping (the values as float, None as NaN)."""
    if not isinstance(summary, Mapping):
        raise ValueError("summary must be the mapping of Block 42's summarize, got %r" % type(summary).__name__)
    missing = [key for key in SUMMARY_KEYS if key not in summary]
    if missing:
        raise ValueError("summary lacks the key(s) %r" % (missing,))
    return {key: _number("summary", key, summary[key]) for key in SUMMARY_KEYS}


def _errorbars(ax, x, y, se, gid):
    """y +- se bars where the three are finite; the bars' LineCollection carries gid."""
    bar = np.isfinite(x) & np.isfinite(y) & np.isfinite(se)
    if np.any(bar):
        eb = ax.errorbar(x[bar], y[bar], yerr=se[bar], fmt="none", ecolor=INK2, elinewidth=0.9, capsize=2.0,
                         zorder=2)
        eb.lines[2][0].set_gid(gid)


def _points(ax, x, y, ok, name, colour=INK):
    """The points, filled where ok and hollow where not (gids '<name>|ok', '<name>|not_ok')."""
    for flag, face, tag in ((True, colour, "ok"), (False, "none", "not_ok")):
        sel = ok == flag
        if np.any(sel):
            ax.plot(x[sel], y[sel], ls="none", marker="o", ms=5.5, mec=colour, mew=1.2, mfc=face, zorder=3,
                    gid="%s|%s" % (name, tag))


def _no_value(ax, x, y):
    """'n/a' at the foot of a sorted panel where a node has no finite value (gid 'no_value')."""
    from matplotlib.transforms import blended_transform_factory
    foot = blended_transform_factory(ax.transData, ax.transAxes)
    for xi in x[~np.isfinite(y)]:
        ax.text(xi, 0.02, "n/a", transform=foot, rotation=90, ha="center", va="bottom", fontsize=6.5, color=INK2,
                gid="no_value")


def figure_summary(rows, summary):
    """The cell's fitted in-focus widths and growth scales (SPEC Block 43). Returns the matplotlib Figure.

    rows: a non-empty sequence of mappings, one per node, with node_id, sigma0_um and se_sigma0_um (um),
      growth_scale and se_growth_scale (1), d_hat_um (the fitted diameter, um), d_allen_um (Allen's diameter, um),
      depth_um (um) and ok (bool: the node enters the summary). A value may be NaN or None (not drawn); an SE
      must be >= 0 where finite.
    summary: the mapping of Block 42's summarize -- n, median_um, mad_um, ivw_mean_um, ivw_se_um, chi2_cons, dof.

    Four panels. Left column, the nodes sorted by sigma0 (NaN last), the node ids below: sigma0 +- se_sigma0 with
    the weighted mean (line), its +-ivw_se band and the median (dashed); under it c +- se_growth_scale in the same
    order, with a dotted line at c = 1 (the configured table's own growth); "n/a" at the foot where a node has no
    finite value. Right column, on sigma0's y axis:
    sigma0 against the fitted diameter, Allen's diameter of each node marked by a cross joined to its fitted one;
    sigma0 against depth; both with the +- se_sigma0 bars. Not-ok nodes hollow in every panel. The summary's
    numbers are written above the first panel as given; a line or band whose value is not finite is not drawn.
    ValueError for empty rows, a row that is not a mapping or lacks a key, an ok that is not a bool, a value that
    is not a number, a negative SE, a summary that is not a mapping or lacks a key."""
    t = _rows(rows)
    s = _summary(summary)
    n = len(t["node_id"])
    order = np.argsort(t["sigma0_um"], kind="stable")      # display order: by value, NaN last
    x = np.arange(n, dtype=float)
    ids = [t["node_id"][i] for i in order]
    ok_s = t["ok"][order]

    plt = _plt()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig = plt.figure(figsize=(max(11.0, 0.3 * n + 7.5), 8.0), facecolor=SURFACE, layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=(max(1.0, 0.045 * n + 0.4), 1.0))
    a_s = fig.add_subplot(gs[0, 0])
    a_c = fig.add_subplot(gs[1, 0], sharex=a_s)
    a_d = fig.add_subplot(gs[0, 1], sharey=a_s)
    a_z = fig.add_subplot(gs[1, 1], sharey=a_s)
    for ax in (a_s, a_c, a_d, a_z):
        _style(ax)

    mean, mean_se, median = s["ivw_mean_um"], s["ivw_se_um"], s["median_um"]
    if np.isfinite(mean) and np.isfinite(mean_se):
        a_s.axhspan(mean - mean_se, mean + mean_se, color=BLUE, alpha=0.15, lw=0, zorder=1, gid="ivw_band")
    if np.isfinite(mean):
        a_s.axhline(mean, color=BLUE, lw=1.3, zorder=1, gid="ivw_mean")
    if np.isfinite(median):
        a_s.axhline(median, color=ORANGE, lw=1.2, ls="--", zorder=1, gid="median")
    sig_s = t["sigma0_um"][order]
    _errorbars(a_s, x, sig_s, t["se_sigma0_um"][order], "se|sigma0")
    _points(a_s, x, sig_s, ok_s, "sigma0")
    _no_value(a_s, x, sig_s)
    a_s.set_ylabel("in-focus width sigma0 +- SE (um)", fontsize=8.5, color=INK2)
    a_s.set_title("sigma0 per node, sorted: weighted mean %s +- %s um, median %s um (MAD %s um)\n"
                  "%s of %d nodes ok; chi2_cons %s on %s dof"
                  % (_fmt(mean), _fmt(mean_se), _fmt(median), _fmt(s["mad_um"]), _fmt(summary["n"]), n,
                     _fmt(summary["chi2_cons"]), _fmt(summary["dof"])), fontsize=8.5, color=INK)

    a_c.axhline(1.0, color=INK2, lw=0.9, ls=":", zorder=1, gid="c_one")
    c_s = t["growth_scale"][order]
    _errorbars(a_c, x, c_s, t["se_growth_scale"][order], "se|growth_scale")
    _points(a_c, x, c_s, ok_s, "growth_scale")
    _no_value(a_c, x, c_s)
    a_c.set_ylabel("growth scale c +- SE", fontsize=8.5, color=INK2)
    a_c.set_xlabel("node (in the order of sigma0)", fontsize=8.5, color=INK2)
    a_c.set_title("the growth scale c per node (dotted: c = 1, the table's own growth)", fontsize=8.5, color=INK)
    for ax in (a_s, a_c):
        ax.set_xticks(x)
        ax.set_xticklabels(ids, rotation=90, fontsize=6.5)
        ax.set_xlim(-0.6, n - 0.4)

    sig, se, ok = t["sigma0_um"], t["se_sigma0_um"], t["ok"]
    d_hat, d_allen = t["d_hat_um"], t["d_allen_um"]
    link = np.isfinite(sig) & np.isfinite(d_hat) & np.isfinite(d_allen)
    if np.any(link):
        gap = np.full(int(link.sum()), np.nan)
        lx = np.column_stack([d_hat[link], d_allen[link], gap]).ravel()
        ly = np.column_stack([sig[link], sig[link], gap]).ravel()
        a_d.plot(lx, ly, color=BLUE, lw=0.7, alpha=0.6, zorder=1, gid="allen_link")
    a_d.plot(d_allen, sig, ls="none", marker="x", ms=5.0, mew=1.1, color=BLUE, zorder=2, gid="allen")
    _errorbars(a_d, d_hat, sig, se, "se|sigma0_vs_d")
    _points(a_d, d_hat, sig, ok, "sigma0_vs_d")
    a_d.set_xlabel("diameter (um): fitted d_hat (circle), Allen's (cross)", fontsize=8.5, color=INK2)
    a_d.set_title("sigma0 against the fitted diameter", fontsize=8.5, color=INK)

    _errorbars(a_z, t["depth_um"], sig, se, "se|sigma0_vs_depth")
    _points(a_z, t["depth_um"], sig, ok, "sigma0_vs_depth")
    a_z.set_xlabel("depth (um)", fontsize=8.5, color=INK2)
    a_z.set_title("sigma0 against depth", fontsize=8.5, color=INK)
    for ax in (a_d, a_z):
        ax.set_ylabel("sigma0 (um)", fontsize=8.5, color=INK2)

    handles = [Line2D([], [], ls="none", marker="o", ms=5.5, mec=INK, mfc=INK, label="ok"),
               Line2D([], [], ls="none", marker="o", ms=5.5, mec=INK, mfc="none", label="not ok (hollow)"),
               Line2D([], [], color=BLUE, lw=1.3, label="weighted mean"),
               Patch(color=BLUE, alpha=0.15, lw=0, label="+- SE of the weighted mean"),
               Line2D([], [], color=ORANGE, lw=1.2, ls="--", label="median"),
               Line2D([], [], ls="none", marker="x", ms=5.0, mew=1.1, color=BLUE, label="Allen's diameter")]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), fontsize=8, frameon=False)
    fig.suptitle("the in-focus width sigma0 and the growth scale c fitted on thick nodes", fontsize=10, color=INK)
    return fig
