"""Figures of the soma's view -- Block 44 in specs/SPEC.md (D-210). Plotting only: every array arrives prepared
(analysis.soma_view and the image block); nothing scientific is computed here.

  figure_soma_focus   the sharpest plane (the crop about the traced soma, the traced soma's circle and centre, the
                      primary dendrites' first nodes) beside the focus measure against depth
  figure_soma_planes  a montage of the planes about the sharpest one, on one grey range, with the same overlay
Coordinates in the image panels: um relative to the traced soma's centre (x right, y down as in the image).
Pure ASCII, for the cluster.
"""
from __future__ import annotations

import numpy as np

from allen_diameter.plotting.cleaning_figures import (BLUE, IMAGE_PERCENTILES, INK, INK2, ORANGE, SURFACE, _plt,
                                                      _row_scale, _style, save)

__all__ = ["figure_soma_focus", "figure_soma_planes", "save"]

SOMA_COLOUR = "#ffd23f"      # the traced soma on the images (the measuring line's yellow of Blocks 41, 43)


def _overlay(ax, ext, centre_xy, radius_um, roots_xy):
    t = np.linspace(0, 2 * np.pi, 181)
    ax.plot(radius_um * np.cos(t), radius_um * np.sin(t), ls="--", lw=1.1, color=SOMA_COLOUR, gid="soma")
    ax.plot([0.0], [0.0], marker="+", ms=8, mew=1.3, color=SOMA_COLOUR, gid="centre")
    if roots_xy is not None and len(roots_xy):
        r = np.asarray(roots_xy, dtype=float) - np.asarray(centre_xy, dtype=float)[None, :2]
        ax.plot(r[:, 0], r[:, 1], ls="none", marker="o", ms=4, mfc="none", mec=ORANGE, mew=1.1, gid="roots")


def _image(ax, img, ext, centre_xy, lo, hi):
    e = np.asarray(ext, dtype=float) - np.array([centre_xy[0], centre_xy[0], centre_xy[1], centre_xy[1]])
    ax.imshow(np.asarray(img, dtype=float), cmap="gray", vmin=lo, vmax=hi, extent=tuple(e), origin="upper",
              interpolation="nearest")
    ax.set_aspect("equal")
    return e


def figure_soma_focus(images, ks, extent, centre_xyz, radius_um, dz_um, F, j_focus, roots_xy=None, title=""):
    """images (n, H, W) grey levels; ks (n,) plane indices; extent (4,) um (figures.image_extent_um); centre_xyz (3,)
    um, the traced soma's centre (image frame); radius_um its traced radius; dz_um the plane spacing; F (n,) the
    focus measure (NaN for a missing plane); j_focus the sharpest plane's index; roots_xy (m, 2) um or None."""
    imgs = np.asarray(images)
    F = np.asarray(F, dtype=float)
    ks = np.asarray(ks).astype(int)
    if imgs.ndim != 3 or F.shape != (imgs.shape[0],) or ks.shape != F.shape:
        raise ValueError("images (n, H, W), ks (n,) and F (n,) must agree")
    j = int(j_focus)
    lo, hi = _row_scale((imgs[j:j + 1].astype(float),), IMAGE_PERCENTILES)
    plt = _plt()
    fig = plt.figure(figsize=(12.5, 6.2), facecolor=SURFACE, layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=(1.25, 1.0))
    ax = fig.add_subplot(gs[0, 0])
    _image(ax, imgs[j], extent, centre_xyz, lo, hi)
    _overlay(ax, extent, centre_xyz, radius_um, roots_xy)
    dzk = ks[j] * float(dz_um) - float(centre_xyz[2])
    ax.set_title("sharpest plane k = %d (%+.2f um from the traced soma's depth)" % (ks[j], dzk), fontsize=9, color=INK)
    ax.set_xlabel("x - x_soma (um)", fontsize=8, color=INK2)
    ax.set_ylabel("y - y_soma (um); grey %.0f-%.0f gl" % (lo, hi), fontsize=8, color=INK2)
    ax.tick_params(labelsize=7, colors=INK2)
    a2 = fig.add_subplot(gs[0, 1])
    _style(a2)
    z = ks * float(dz_um) - float(centre_xyz[2])
    f = F / np.nanmax(F)
    a2.plot(z, f, color=BLUE, lw=1.4, marker="o", ms=3, gid="focus")
    a2.axvline(0.0, color=INK2, lw=0.9, ls="--", gid="soma_depth")
    a2.axvline(z[j], color=ORANGE, lw=1.2, gid="sharpest")
    a2.axvspan(-float(radius_um), float(radius_um), color="#e1e0d9", alpha=0.6, lw=0, zorder=0)
    a2.set_xlabel("plane depth - traced soma depth (um)", fontsize=8, color=INK2)
    a2.set_ylabel("focus measure / its maximum", fontsize=8, color=INK2)
    a2.set_title("focus over the disk of the traced soma (+ margin)\ndashed: the traced depth; orange: the sharpest "
                 "plane; grey band: +- the traced radius", fontsize=8, color=INK)
    if title:
        fig.suptitle(title, fontsize=10, color=INK)
    return fig


def figure_soma_planes(images, ks, extent, centre_xyz, radius_um, dz_um, indices, j_focus, roots_xy=None, title=""):
    """The montage: the planes `indices` (into images) on one grey range, each with the traced soma's circle, the
    sharpest one framed."""
    imgs = np.asarray(images)
    ks = np.asarray(ks).astype(int)
    idx = [int(i) for i in indices]
    if imgs.ndim != 3 or not idx or min(idx) < 0 or max(idx) >= imgs.shape[0]:
        raise ValueError("indices must index the planes of images (n, H, W)")
    lo, hi = _row_scale((imgs[idx].astype(float),), IMAGE_PERCENTILES)
    n = len(idx)
    ncol = min(5, n)
    nrow = int(np.ceil(n / float(ncol)))
    plt = _plt()
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.1 * nrow + 0.6), facecolor=SURFACE,
                             layout="constrained", squeeze=False)
    for a in axes.ravel()[n:]:
        a.set_axis_off()
    for a, i in zip(axes.ravel(), idx):
        _image(a, imgs[i], extent, centre_xyz, lo, hi)
        _overlay(a, extent, centre_xyz, radius_um, roots_xy)
        dzk = ks[i] * float(dz_um) - float(centre_xyz[2])
        star = i == int(j_focus)
        a.set_title("k = %d  (%+.2f um)%s" % (ks[i], dzk, "  sharpest" if star else ""), fontsize=8,
                    color=INK)   # text stays in ink; the orange frame marks the sharpest plane
        for side in a.spines.values():
            side.set_color(ORANGE if star else INK2)
            side.set_linewidth(2.0 if star else 0.6)
        a.tick_params(labelsize=6, colors=INK2)
    fig.suptitle((title + "\n" if title else "") + "dashed: the traced soma (Allen's radius); +: its centre; circles: "
                 "the primary dendrites' first traced nodes; grey %.0f-%.0f gl" % (lo, hi), fontsize=9, color=INK)
    return fig
