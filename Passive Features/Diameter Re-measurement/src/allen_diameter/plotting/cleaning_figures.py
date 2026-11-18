"""Figures of the cleaning -- Block 31 in specs/SPEC.md (plan step C6; D-129.4).

Plotting only: every array arrives prepared (Blocks 25, 28, 32 by contract);
nothing scientific is computed here and no input array is written to.

  figure_node    one example node: per chosen plane a row of three images --
                 the patch before, B S the predicted shadow, the patch after --
                 before and after on one grey scale (the 1st to 99th
                 percentile of the two), B S on its own (0 to its largest
                 value; amendment 2, D-150.1), with the measuring line drawn
                 on each, and one panel of the line's profiles before and after
  figure_line_rounds  one example node's measuring line across the rounds:
                 per chosen plane the profiles before and after each round,
                 and the light each round added back on the line (after
                 minus before), so a neighbour's shadow left in or put back
                 too strongly shows round by round (amendment 1, D-149.4)
  figure_rounds  the record of the rounds: the largest relative diameter
                 change with the stop threshold stop_rel_d, the largest change
                 of the example lines, the number of nodes refined; the last
                 allowed round max_rounds drawn in every panel
  save           the PNG of a figure

Conventions read from SPEC.md: images in the fetch_zblock contract (n_planes,
H, W), row = y, column = x; the image frame of section 2.1 (um; pixel (row,
col) centred on ((left + col) p, (top + row) p), plotting.figures.image_extent_um);
profiles in grey levels on the offsets v (um) of Block 5.

Colours: the palette of the project's figures (calibration_figures): ink for
"before", blue for "after", orange for the measuring line and the thresholds.
Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import os
from collections.abc import Mapping

import numpy as np

from allen_diameter.plotting.figures import image_extent_um

INK, INK2, BLUE, ORANGE = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834"
SURFACE, GRID = "#fcfcfb", "#e4e3df"
IMAGE_TITLES = ("before (observed)", "B S (predicted shadow)", "after (cleaned)")
ROUND_KEYS = ("max_rel_d", "max_line_change_gl", "n_refined")


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=7)


def _fmt(value):
    if isinstance(value, (float, np.floating)):
        return "%.4g" % float(value)
    return str(value)


def _title(meta):
    """The figure title from meta's node_id, class, round and features (each optional)."""
    parts = []
    if "node_id" in meta:
        parts.append("node %s" % _fmt(meta["node_id"]))
    if "class" in meta:
        parts.append("class %s" % _fmt(meta["class"]))
    if "round" in meta:
        parts.append("round %s" % _fmt(meta["round"]))
    head = "  |  ".join(parts) if parts else "node"
    feats = meta.get("features")
    if feats is None:
        return head
    if isinstance(feats, Mapping):
        text = ", ".join("%s %s" % (k, _fmt(v)) for k, v in feats.items())
    else:
        text = str(feats)
    return head + "\nfeatures: " + text


def _plane_rows(ks, planes):
    """Positions along the block's plane axis of the planes to show (plane indices k taken from ks)."""
    ks = [int(k) for k in ks]
    if planes is None:
        return list(range(len(ks)))
    wanted = [int(k) for k in np.asarray(planes).reshape(-1)]
    if not wanted:
        raise ValueError("planes is empty")
    rows = []
    for k in wanted:
        if k not in ks:
            raise ValueError("plane %d is not in ks %r" % (k, ks))
        rows.append(ks.index(k))
    return rows


IMAGE_PERCENTILES = (1.0, 99.0)   # the grey range of before and after: these percentiles of the two together (amendment 2, D-150.1)


def _row_scale(images, percentiles=None):
    """One (vmin, vmax) for the images of a row: their joint finite range, or, with percentiles = (lo, hi),
    numpy's percentiles lo and hi of their finite values together (amendment 2, D-150.1)."""
    vals = np.concatenate([np.asarray(im, dtype=float).ravel() for im in images])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    if percentiles is None:
        lo, hi = float(vals.min()), float(vals.max())
    else:
        lo, hi = (float(x) for x in np.percentile(vals, [float(percentiles[0]), float(percentiles[1])]))
    return (lo, hi) if hi > lo else (lo, lo + 1.0)


def _shadow_scale(shadow):
    """(vmin, vmax) of a predicted-shadow image: 0 to its largest finite value (1 when there is none above 0)."""
    vals = np.asarray(shadow, dtype=float)
    vals = vals[np.isfinite(vals)]
    hi = float(vals.max()) if vals.size else 0.0
    return 0.0, (hi if hi > 0.0 else 1.0)


def figure_node(block, shadow, cleaned, ks, frame, line_xy, v, profiles_before, profiles_after, meta, planes=None,
                percentiles=IMAGE_PERCENTILES):
    """One example node of the cleaning. Returns the matplotlib Figure.

    block, shadow, cleaned: (n_planes, H, W) on the same pixels, grey levels --
      the observed patch, the predicted shadow B S (gl; Block 25's S times the
      background, Block 28), the cleaned patch; any real dtype.
    ks: (n_planes,) plane indices k of the three stacks' first axis.
    frame: the CropFrame of the stacks (fetch_zblock contract, downsample 0).
    line_xy: (2, 2) the measuring line's two ends ((x0, y0), (x1, y1)), um in
      the image frame.
    v: (n_v,) offsets along the line, um; profiles_before, profiles_after:
      (n_planes, n_v) grey levels, the line's profiles in the planes ks.
    meta: mapping for the title -- node_id, class, round, features (a mapping
      name -> value, or a string); each optional.
    planes: plane indices k (values of ks) to show, in that order; None = all.
    percentiles: (lo, hi) -- the grey range of the before and after images of a
      row is numpy's percentiles lo and hi of the two images' finite values
      together (one scale for both, so they compare); None = their joint full
      range (amendment 2, D-150.1).

    Layout: one row per shown plane -- the observed and the cleaned image on that
    shared grey scale, the predicted shadow between them on its own scale (0 to
    its largest value), the measuring line drawn on each, then the profile panel
    (before: ink, after: blue); the row's label gives both scales.
    ValueError for shapes that disagree, an empty `planes`, a plane not in ks.
    """
    block = np.asarray(block)
    shadow = np.asarray(shadow)
    cleaned = np.asarray(cleaned)
    if block.ndim != 3:
        raise ValueError("block must be (n_planes, H, W), got shape %r" % (block.shape,))
    if shadow.shape != block.shape or cleaned.shape != block.shape:
        raise ValueError("block %r, shadow %r and cleaned %r must have one shape"
                         % (block.shape, shadow.shape, cleaned.shape))
    n_planes = block.shape[0]
    ks = np.asarray(ks)
    if ks.ndim != 1 or ks.size != n_planes:
        raise ValueError("ks must be (n_planes,) = (%d,), got shape %r" % (n_planes, np.shape(ks)))
    line = np.asarray(line_xy, dtype=float)
    if line.shape != (2, 2):
        raise ValueError("line_xy must be (2, 2) = ((x0, y0), (x1, y1)) um, got shape %r" % (line.shape,))
    v = np.asarray(v, dtype=float)
    if v.ndim != 1 or v.size < 2:
        raise ValueError("v must be 1-D with at least 2 offsets, got shape %r" % (v.shape,))
    before = np.asarray(profiles_before, dtype=float)
    after = np.asarray(profiles_after, dtype=float)
    for name, prof in (("profiles_before", before), ("profiles_after", after)):
        if prof.shape != (n_planes, v.size):
            raise ValueError("%s must be (n_planes, n_v) = (%d, %d), got %r" % (name, n_planes, v.size, prof.shape))
    if not isinstance(meta, Mapping):
        raise ValueError("meta must be a mapping (node_id, class, round, features)")
    rows = _plane_rows(ks, planes)
    extent = image_extent_um(frame, block.shape)

    plt = _plt()
    import matplotlib.patheffects as pe
    halo = [pe.Stroke(linewidth=3.2, foreground="white"), pe.Normal()]
    n = len(rows)
    side = 2.3
    fig, axes = plt.subplots(n, 4, figsize=(4 * side + 1.6, side * n + 0.9), facecolor=SURFACE, squeeze=False,
                             gridspec_kw=dict(width_ratios=[1, 1, 1, 1.35]), layout="constrained")
    for r, j in enumerate(rows):
        imgs = (block[j], shadow[j], cleaned[j])
        lo, hi = _row_scale((block[j], cleaned[j]), percentiles)
        s_lo, s_hi = _shadow_scale(shadow[j])
        for c in range(3):
            ax = axes[r, c]
            vmin, vmax = (s_lo, s_hi) if c == 1 else (lo, hi)
            ax.imshow(imgs[c], cmap="gray", vmin=vmin, vmax=vmax, extent=extent, origin="upper",
                      interpolation="nearest")
            ax.plot(line[:, 0], line[:, 1], color=ORANGE, lw=1.4, path_effects=halo)
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(IMAGE_TITLES[c], fontsize=8.5, color=INK)
        axes[r, 0].set_ylabel("plane k = %d\nbefore, after %.0f-%.0f gl\nB S 0-%.1f gl" % (int(ks[j]), lo, hi, s_hi),
                              fontsize=7.5, color=INK)
        ax = axes[r, 3]
        _style(ax)
        ax.plot(v, before[j], color=INK, lw=1.2, label="before")
        ax.plot(v, after[j], color=BLUE, lw=1.4, label="after")
        ax.set_xlim(v[0], v[-1])
        ax.set_ylabel("grey level", fontsize=7.5, color=INK2)
        if r == 0:
            ax.set_title("profile along the line", fontsize=8.5, color=INK)
            ax.legend(fontsize=7, frameon=False, loc="lower right")
        if r == n - 1:
            ax.set_xlabel("v (um)", fontsize=7.5, color=INK2)
    fig.suptitle(_title(meta), fontsize=9.5, color=INK)
    return fig


ROUND_STYLES = ((ORANGE, "-"), (BLUE, "-"), (INK2, "--"), (INK2, ":"))   # rounds 1, 2, 3, then 4 and later


def figure_line_rounds(v, ks, profiles_before, profiles_after_rounds, meta, planes=None):
    """One example node's measuring line across the cleaning's rounds. Returns the matplotlib Figure.

    v: (n_v,) offsets along the line, um; ks: (n_planes,) plane indices k.
    profiles_before: (n_planes, n_v) grey levels, the observed line.
    profiles_after_rounds: mapping round t (int >= 1) -> (n_planes, n_v) grey levels, the cleaned line
      after round t (each round's cleaning used the features of the round before).
    meta: mapping for the title, as figure_node's; planes: plane indices k (values of ks) to show; None = all.

    Layout: one row per shown plane -- the profiles (before: ink; after round 1: orange, round 2: blue,
    later rounds grey, dashed) and the light added back on the line by each round, after minus before (gl),
    on the same colours with the zero line. ValueError for shapes that disagree, no rounds, a round < 1,
    an empty `planes`, a plane not in ks.
    """
    v = np.asarray(v, dtype=float)
    if v.ndim != 1 or v.size < 2:
        raise ValueError("v must be 1-D with at least 2 offsets, got shape %r" % (v.shape,))
    ks = np.asarray(ks)
    before = np.asarray(profiles_before, dtype=float)
    if ks.ndim != 1 or before.shape != (ks.size, v.size):
        raise ValueError("profiles_before must be (n_planes, n_v) = (%d, %d), got %r" % (ks.size, v.size, before.shape))
    if not isinstance(profiles_after_rounds, Mapping) or not profiles_after_rounds:
        raise ValueError("profiles_after_rounds must be a non-empty mapping round -> (n_planes, n_v)")
    after = {}
    for t, prof in profiles_after_rounds.items():
        if int(t) != t or int(t) < 1:
            raise ValueError("rounds are integers >= 1, got %r" % (t,))
        a = np.asarray(prof, dtype=float)
        if a.shape != before.shape:
            raise ValueError("the profiles of round %d must be %r, got %r" % (int(t), before.shape, a.shape))
        after[int(t)] = a
    if not isinstance(meta, Mapping):
        raise ValueError("meta must be a mapping (node_id, class, round, features)")
    rows = _plane_rows(ks, planes)
    order = sorted(after)

    plt = _plt()
    n = len(rows)
    fig, axes = plt.subplots(n, 2, figsize=(9.6, 2.3 * n + 0.9), facecolor=SURFACE, squeeze=False,
                             layout="constrained")
    for r, j in enumerate(rows):
        ax, dx = axes[r, 0], axes[r, 1]
        _style(ax)
        _style(dx)
        ax.plot(v, before[j], color=INK, lw=1.2, label="before")
        dx.axhline(0.0, color=INK2, lw=0.8)
        for t in order:
            col, ls = ROUND_STYLES[min(t, len(ROUND_STYLES)) - 1]
            ax.plot(v, after[t][j], color=col, ls=ls, lw=1.4, label="after round %d" % t)
            dx.plot(v, after[t][j] - before[j], color=col, ls=ls, lw=1.4, label="round %d" % t)
        for a_ in (ax, dx):
            a_.set_xlim(v[0], v[-1])
        ax.set_ylabel("plane k = %d\ngrey level" % int(ks[j]), fontsize=7.5, color=INK2)
        dx.set_ylabel("added back (gl)", fontsize=7.5, color=INK2)
        if r == 0:
            ax.set_title("profile along the line", fontsize=8.5, color=INK)
            dx.set_title("light added back on the line (after - before)", fontsize=8.5, color=INK)
            ax.legend(fontsize=7, frameon=False, loc="lower right")
        if r == n - 1:
            ax.set_xlabel("v (um)", fontsize=7.5, color=INK2)
            dx.set_xlabel("v (um)", fontsize=7.5, color=INK2)
    fig.suptitle(_title(meta), fontsize=9.5, color=INK)
    return fig


def figure_rounds(history, cfg):
    """The record of the cleaning's rounds. Returns the matplotlib Figure.

    history: non-empty list of mappings, one per round in order (round t = 1,
    2, ...), each with max_rel_d (the largest relative diameter change, -),
    max_line_change_gl (the largest change of an example line, gl) and
    n_refined (the number of nodes refined). Three panels sharing the round
    axis, one point per round; the stop threshold CleaningConfig.stop_rel_d is
    drawn on max_rel_d and the last allowed round CleaningConfig.max_rounds in
    every panel, both read from cfg (a DiameterConfig: the run's own values;
    Block 31, amendment 1).
    ValueError for an empty history or a round without one of the keys.
    """
    if isinstance(history, Mapping) or len(history) == 0:
        raise ValueError("history must be a non-empty list of per-round mappings")
    cols = {key: [] for key in ROUND_KEYS}
    for t, rec in enumerate(history):
        if not isinstance(rec, Mapping) or any(key not in rec for key in ROUND_KEYS):
            raise ValueError("round %d must be a mapping with keys %r" % (t + 1, ROUND_KEYS))
        for key in ROUND_KEYS:
            cols[key].append(float(rec[key]))
    ccfg = cfg.cleaning
    t = np.arange(1, len(history) + 1)
    rel = np.asarray(cols["max_rel_d"])

    plt = _plt()
    fig, axes = plt.subplots(3, 1, figsize=(6.4, 6.6), sharex=True, facecolor=SURFACE)
    for ax in axes:
        _style(ax)
        ax.axvline(ccfg.max_rounds, color=ORANGE, lw=1.0, ls=":", zorder=1)
    axes[0].plot(t, rel, "o-", color=INK, ms=4, lw=1.1, label="largest relative change of d")
    axes[0].axhline(ccfg.stop_rel_d, color=ORANGE, lw=1.1, ls="--", label="stop_rel_d = %g" % ccfg.stop_rel_d)
    finite = rel[np.isfinite(rel)]
    if finite.size and np.all(finite > 0):
        axes[0].set_yscale("log")
    axes[0].set_ylabel("max |d_t - d_(t-1)| / d_(t-1)", fontsize=8, color=INK2)
    axes[0].legend(fontsize=7, frameon=False, loc="upper right")
    axes[1].plot(t, cols["max_line_change_gl"], "o-", color=BLUE, ms=4, lw=1.1)
    axes[1].set_ylabel("largest line change (gl)", fontsize=8, color=INK2)
    axes[2].plot(t, cols["n_refined"], "o-", color=INK2, ms=4, lw=1.1)
    axes[2].set_ylabel("nodes refined", fontsize=8, color=INK2)
    axes[2].set_xlabel("round (dotted: max_rounds = %d)" % ccfg.max_rounds, fontsize=8, color=INK2)
    axes[2].set_xlim(0.5, max(len(history), ccfg.max_rounds) + 0.5)
    axes[2].set_xticks(np.arange(1, max(len(history), ccfg.max_rounds) + 1))
    fig.suptitle("cleaning rounds", fontsize=9.5, color=INK)
    fig.tight_layout()
    return fig


FOCAL_REQUIRED = ("image", "extent", "axis_xy", "e_u", "d_um")
FOCAL_MAX_COLS = 5


def _focal_panel(panel, index):
    """Validated arrays of one figure_focal panel (amendment 3)."""
    if not isinstance(panel, Mapping):
        raise ValueError("panel %d must be a mapping, got %r" % (index, type(panel).__name__))
    for key in FOCAL_REQUIRED:
        if key not in panel:
            raise ValueError("panel %d lacks %r" % (index, key))
    image = np.asarray(panel["image"])
    if image.ndim != 2 or image.size == 0:
        raise ValueError("panel %d: image must be (H, W), got shape %r" % (index, image.shape))
    extent = np.asarray(panel["extent"], dtype=float)
    axis = np.asarray(panel["axis_xy"], dtype=float)
    e_u = np.asarray(panel["e_u"], dtype=float)
    if extent.shape != (4,) or axis.shape != (2,) or e_u.shape != (2,):
        raise ValueError("panel %d: extent must be (4,), axis_xy and e_u (2,), got %r, %r, %r"
                         % (index, extent.shape, axis.shape, e_u.shape))
    if not (np.all(np.isfinite(extent)) and np.all(np.isfinite(axis)) and np.all(np.isfinite(e_u))):
        raise ValueError("panel %d: extent, axis_xy and e_u must be finite" % index)
    norm = float(np.hypot(e_u[0], e_u[1]))
    if not norm > 0:
        raise ValueError("panel %d: e_u has zero length" % index)
    e_u = e_u / norm
    d = float(panel["d_um"])
    if not (np.isfinite(d) and d > 0):
        raise ValueError("panel %d: d_um must be positive and finite, got %r" % (index, panel["d_um"]))
    out = dict(image=image, extent=extent, axis=axis, e_u=e_u, d=d, ref=None, d_ref=None, line=None, half=None,
               title=str(panel.get("title", "")), label=str(panel.get("label", "estimate")))
    if panel.get("ref_xy") is not None and panel.get("d_ref_um") is not None:
        ref = np.asarray(panel["ref_xy"], dtype=float)
        d_ref = float(panel["d_ref_um"])
        if ref.shape != (2,) or not np.all(np.isfinite(ref)):
            raise ValueError("panel %d: ref_xy must be (2,) and finite" % index)
        if not (np.isfinite(d_ref) and d_ref > 0):
            raise ValueError("panel %d: d_ref_um must be positive and finite, got %r" % (index, panel["d_ref_um"]))
        out["ref"], out["d_ref"] = ref, d_ref
    if panel.get("line_xy") is not None:
        line = np.asarray(panel["line_xy"], dtype=float)
        if line.shape != (2, 2):
            raise ValueError("panel %d: line_xy must be (2, 2), got %r" % (index, line.shape))
        out["line"] = line
    if panel.get("half_length_um") is not None:
        half = np.asarray(panel["half_length_um"], dtype=float)
        if half.shape != (2,) or not np.all(np.isfinite(half)) or np.any(half < 0):
            raise ValueError("panel %d: half_length_um must be (2,), finite and >= 0" % index)
        out["half"] = half
    return out


def focal_edges(centre_xy, e_u, d_um, half_minus_um, half_plus_um, origin_xy):
    """The two edges of a tube of diameter d_um drawn in a focal-plane panel (amendment 3): (2, 2, 2) um, the
    edge e (0: the side of +n_hat, 1: of -n_hat), its two ends, (x, y), relative to origin_xy:
    centre_xy +- (d / 2) n_hat, from -half_minus_um to +half_plus_um along the unit e_u, n_hat = (-e_u[1], e_u[0])."""
    c = np.asarray(centre_xy, dtype=float) - np.asarray(origin_xy, dtype=float)
    e = np.asarray(e_u, dtype=float)
    n_hat = np.array([-e[1], e[0]])
    ends = np.array([c - float(half_minus_um) * e, c + float(half_plus_um) * e])
    return np.array([ends + 0.5 * float(d_um) * n_hat, ends - 0.5 * float(d_um) * n_hat])


def figure_focal(panels, ncols=None, percentiles=IMAGE_PERCENTILES, zoom_um=None):
    """Nodes at one recorded plane each, with the estimated tube drawn on the image (Block 31, amendment 3).
    Returns the matplotlib Figure.

    panels: a sequence of mappings, one per node --
      image (H, W) grey levels; extent (x_left, x_right, y_bottom, y_top) um (figures.image_extent_um);
      axis_xy (2,) um, where the estimated tube's axis crosses the measuring line; e_u (2,) the direction of the
      axis' projection; d_um > 0 the estimated diameter;
      optional: half_length_um (2,) um, how far the edges run against and along e_u (None: zoom_um, or the
      patch's diagonal); ref_xy (2,) um and d_ref_um > 0, the traced position and Allen's diameter; line_xy
      (2, 2) um, the measuring line; title; label (the estimate's name in the legend).
    ncols: panels per row (None: one row up to FOCAL_MAX_COLS, then rows of FOCAL_MAX_COLS).
    percentiles: (lo, hi) of the finite grey levels in view give the panel's grey range; None = their full range.
    zoom_um: > 0, the view is +-zoom_um about axis_xy; None = the whole patch.

    Each panel's coordinates are um relative to axis_xy (x to the right, y downwards as in the image). Drawn: the
    image; the estimate's two edges (solid) at axis_xy +- (d / 2) n_hat along e_u and its axis (thin); Allen's two
    edges (dashed) about ref_xy along the same direction; the measuring line; the title. Computes nothing
    scientific. ValueError for an empty panels, a panel that is not a mapping or lacks a required key, shapes
    that disagree, d_um or d_ref_um not positive and finite, an e_u of zero length, zoom_um or ncols not positive.
    """
    if isinstance(panels, Mapping) or not hasattr(panels, "__len__") or len(panels) == 0:
        raise ValueError("panels must be a non-empty sequence of mappings")
    items = [_focal_panel(pn, i) for i, pn in enumerate(panels)]
    if zoom_um is not None and not (np.isfinite(float(zoom_um)) and float(zoom_um) > 0):
        raise ValueError("zoom_um must be > 0 or None, got %r" % (zoom_um,))
    if ncols is None:
        ncols = min(len(items), FOCAL_MAX_COLS)
    if int(ncols) != ncols or int(ncols) < 1:
        raise ValueError("ncols must be a positive integer or None, got %r" % (ncols,))
    ncols = int(ncols)
    nrows = (len(items) + ncols - 1) // ncols

    plt = _plt()
    import matplotlib.patheffects as pe
    halo = [pe.Stroke(linewidth=2.4, foreground="black", alpha=0.5), pe.Normal()]
    side = 3.1
    fig, axes = plt.subplots(nrows, ncols, figsize=(side * ncols + 0.4, side * nrows + 1.1), facecolor=SURFACE,
                             squeeze=False, layout="constrained")
    handles = {}
    for i, it in enumerate(items):
        ax = axes[i // ncols, i % ncols]
        ext = it["extent"] - np.array([it["axis"][0], it["axis"][0], it["axis"][1], it["axis"][1]])
        if zoom_um is not None:
            z = float(zoom_um)
            view = (-z, z, z, -z)
        else:
            view = tuple(ext)
        H, W = it["image"].shape
        xs = np.linspace(ext[0], ext[1], W + 1)
        ys = np.linspace(ext[3], ext[2], H + 1)
        cols = (xs[1:] > min(view[0], view[1])) & (xs[:-1] < max(view[0], view[1]))
        rows = (ys[1:] > min(view[2], view[3])) & (ys[:-1] < max(view[2], view[3]))
        seen = it["image"][np.ix_(rows, cols)] if rows.any() and cols.any() else it["image"]
        lo, hi = _row_scale((seen,), percentiles)
        ax.imshow(it["image"], cmap="gray", vmin=lo, vmax=hi, extent=tuple(ext), origin="upper",
                  interpolation="nearest")
        diag = float(np.hypot(ext[1] - ext[0], ext[2] - ext[3]))
        if it["half"] is not None:
            hm, hp = float(it["half"][0]), float(it["half"][1])
        else:
            hm = hp = float(zoom_um) * 1.5 if zoom_um is not None else diag
        if it["line"] is not None:
            ln = it["line"] - it["axis"]
            handles["measuring line"], = ax.plot(ln[:, 0], ln[:, 1], color="#ffd23f", lw=0.9, ls=":", gid="line")
        if it["ref"] is not None:
            for e, edge in enumerate(focal_edges(it["ref"], it["e_u"], it["d_ref"], hm, hp, it["axis"])):
                handles["Allen's diameter, about the trace"], = ax.plot(
                    edge[:, 0], edge[:, 1], color="#35d0ff", lw=1.1, ls="--", path_effects=halo, gid="ref_edge%d" % e)
        for e, edge in enumerate(focal_edges(it["axis"], it["e_u"], it["d"], hm, hp, it["axis"])):
            handles[it["label"]], = ax.plot(edge[:, 0], edge[:, 1], color=ORANGE, lw=1.3, path_effects=halo,
                                           gid="edge%d" % e)
        centre = focal_edges(it["axis"], it["e_u"], 0.0, hm, hp, it["axis"])[0]
        handles["estimated axis"], = ax.plot(centre[:, 0], centre[:, 1], color=ORANGE, lw=0.6, alpha=0.8, gid="axis")
        ax.set_xlim(view[0], view[1])
        ax.set_ylim(view[2], view[3])
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7, colors=INK2)
        ax.set_xlabel("x - x_axis (um)", fontsize=7.5, color=INK2)
        if i % ncols == 0:
            ax.set_ylabel("y - y_axis (um)", fontsize=7.5, color=INK2)
        ax.set_title(it["title"], fontsize=8, color=INK)
    for j in range(len(items), nrows * ncols):
        axes[j // ncols, j % ncols].set_axis_off()
    fig.legend(list(handles.values()), list(handles.keys()), loc="outside lower center", ncol=min(4, len(handles)),
               fontsize=8, frameon=False)
    return fig


def save(fig, path, dpi):
    """Write fig as PNG at dpi (> 0) to path, creating its folder. Returns the path.
    The figure stays open: the caller closes it."""
    if not (float(dpi) > 0):
        raise ValueError("dpi must be > 0, got %r" % (dpi,))
    path = os.fspath(path)
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor=SURFACE, bbox_inches="tight")
    return path
