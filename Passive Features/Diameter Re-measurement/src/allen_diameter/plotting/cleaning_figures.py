"""Figures of the cleaning -- Block 31 in specs/SPEC.md (plan step C6; D-129.4).

Plotting only: every array arrives prepared (Blocks 25, 28, 32 by contract);
nothing scientific is computed here and no input array is written to.

  figure_node    one example node: per chosen plane a row of three images on
                 one grey scale -- the patch before, B S the predicted shadow,
                 the patch after -- with the measuring line drawn on each, and
                 one panel of the line's profiles before and after
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

from allen_diameter.config import CleaningConfig
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


def _row_scale(images):
    """One (vmin, vmax) for the three images of a row: their joint finite range."""
    vals = np.concatenate([np.asarray(im, dtype=float).ravel() for im in images])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    lo, hi = float(vals.min()), float(vals.max())
    return (lo, hi) if hi > lo else (lo, lo + 1.0)


def figure_node(block, shadow, cleaned, ks, frame, line_xy, v, profiles_before, profiles_after, meta, planes=None):
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

    Layout: one row per shown plane -- three images sharing one grey scale
    (vmin/vmax = the joint range of the three), the measuring line drawn on
    each, then the profile panel (before: ink, after: blue).
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
        lo, hi = _row_scale(imgs)
        for c in range(3):
            ax = axes[r, c]
            ax.imshow(imgs[c], cmap="gray", vmin=lo, vmax=hi, extent=extent, origin="upper",
                      interpolation="nearest")
            ax.plot(line[:, 0], line[:, 1], color=ORANGE, lw=1.4, path_effects=halo)
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(IMAGE_TITLES[c], fontsize=8.5, color=INK)
        axes[r, 0].set_ylabel("plane k = %d\nscale %.0f-%.0f gl" % (int(ks[j]), lo, hi), fontsize=8, color=INK)
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


def figure_rounds(history):
    """The record of the cleaning's rounds. Returns the matplotlib Figure.

    history: non-empty list of mappings, one per round in order (round t = 1,
    2, ...), each with max_rel_d (the largest relative diameter change, -),
    max_line_change_gl (the largest change of an example line, gl) and
    n_refined (the number of nodes refined). Three panels sharing the round
    axis, one point per round; the stop threshold CleaningConfig.stop_rel_d is
    drawn on max_rel_d and the last allowed round CleaningConfig.max_rounds in
    every panel (the configuration's defaults: the signature takes no config).
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
    ccfg = CleaningConfig()
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
