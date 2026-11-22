"""Figures of the soma's view -- Block 44 in specs/SPEC.md (D-210, D-211). Plotting only: every array arrives
prepared (analysis.soma_view and the image block); nothing scientific is computed here.

  figure_soma_focus   the reference plane -- the one chosen by eye, else the one nearest the traced depth -- (the crop
                      about the traced soma, the traced soma's circle and centre, the primary dendrites' first nodes)
                      beside the fine-detail energy against the plane index, a diagnostic, not a focus estimate
  figure_soma_planes  a montage of planes on one grey range, with the same overlay; the reference plane framed orange,
                      the planes where the soma is seen (by eye) framed dark
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


def _plane_title(k, dzk, k_traced, chosen):
    tags = [t for t, on in (("traced depth", k == int(k_traced)), ("chosen", chosen)) if on]
    return "k = %d  (%+.2f um)%s" % (k, dzk, ("  " + ", ".join(tags)) if tags else "")


def figure_soma_focus(images, ks, extent, centre_xyz, radius_um, dz_um, E, j_ref, ref_label, k_traced, k_range=None,
                      roots_xy=None, title=""):
    """images (n, H, W) grey levels; ks (n,) plane indices; extent (4,) um (figures.image_extent_um); centre_xyz (3,)
    um, the traced soma's centre (image frame); radius_um its traced radius; dz_um the plane spacing; E (n,) the
    fine-detail energy (NaN for a missing plane); j_ref the index of the plane shown and ref_label its origin ("chosen
    by eye" or "traced depth"); k_traced the plane nearest the traced depth; k_range (lo, hi) the planes where the soma
    is seen by eye, or None; roots_xy (m, 2) um or None."""
    imgs = np.asarray(images)
    E = np.asarray(E, dtype=float)
    ks = np.asarray(ks).astype(int)
    if imgs.ndim != 3 or E.shape != (imgs.shape[0],) or ks.shape != E.shape:
        raise ValueError("images (n, H, W), ks (n,) and E (n,) must agree")
    j = int(j_ref)
    lo, hi = _row_scale((imgs[j:j + 1].astype(float),), IMAGE_PERCENTILES)
    plt = _plt()
    fig = plt.figure(figsize=(12.5, 6.2), facecolor=SURFACE, layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=(1.25, 1.0))
    ax = fig.add_subplot(gs[0, 0])
    _image(ax, imgs[j], extent, centre_xyz, lo, hi)
    _overlay(ax, extent, centre_xyz, radius_um, roots_xy)
    dzk = ks[j] * float(dz_um) - float(centre_xyz[2])
    what = ("the plane chosen by eye" if ref_label == "chosen by eye"
            else "the traced depth's plane -- no plane chosen yet (SOMA_K in Cell 24)")
    ax.set_title("k = %d (%+.2f um from the traced depth): %s" % (ks[j], dzk, what), fontsize=9, color=INK)
    ax.set_xlabel("x - x_soma (um)", fontsize=8, color=INK2)
    ax.set_ylabel("y - y_soma (um); grey %.0f-%.0f gl" % (lo, hi), fontsize=8, color=INK2)
    ax.tick_params(labelsize=7, colors=INK2)
    a2 = fig.add_subplot(gs[0, 1])
    _style(a2)
    if k_range is not None:
        a2.axvspan(int(k_range[0]) - 0.5, int(k_range[1]) + 0.5, color="#e1e0d9", alpha=0.8, lw=0, zorder=0,
                   gid="range_by_eye")
    a2.plot(ks, E / np.nanmax(E), color=BLUE, lw=1.4, marker="o", ms=3, gid="detail_energy")
    a2.axvline(int(k_traced), color=INK2, lw=0.9, ls="--", gid="traced_depth")
    a2.axvline(ks[j], color=ORANGE, lw=1.2, gid="shown")
    a2.set_xlabel("plane k (%.2f um apart)" % float(dz_um), fontsize=8, color=INK2)
    a2.set_ylabel("fine-detail energy / its maximum", fontsize=8, color=INK2)
    a2.set_title("fine-detail energy inside the traced disk (+ margin): every structure counts, dendrites included --\n"
                 "a diagnostic, not a focus estimate. dashed: the traced depth (k = %d); orange: the plane shown%s"
                 % (int(k_traced), "; grey: the planes where you see the soma" if k_range is not None else ""),
                 fontsize=8, color=INK)
    if title:
        fig.suptitle(title, fontsize=10, color=INK)
    return fig


def figure_soma_planes(images, ks, extent, centre_xyz, radius_um, dz_um, indices, j_ref, ref_label, k_traced,
                       k_range=None, roots_xy=None, title="", ncol=6):
    """The montage: the planes `indices` (into images) on one grey range, each with the traced soma's circle; the
    reference plane j_ref framed orange, the planes inside k_range (by eye) framed dark."""
    imgs = np.asarray(images)
    ks = np.asarray(ks).astype(int)
    idx = [int(i) for i in indices]
    if imgs.ndim != 3 or not idx or min(idx) < 0 or max(idx) >= imgs.shape[0]:
        raise ValueError("indices must index the planes of images (n, H, W)")
    lo, hi = _row_scale((imgs[idx].astype(float),), IMAGE_PERCENTILES)
    n = len(idx)
    nc = max(1, min(int(ncol), n))
    nrow = int(np.ceil(n / float(nc)))
    plt = _plt()
    fig, axes = plt.subplots(nrow, nc, figsize=(2.7 * nc, 2.85 * nrow + 0.8), facecolor=SURFACE,
                             layout="constrained", squeeze=False)
    for a in axes.ravel()[n:]:
        a.set_axis_off()
    chosen = ref_label == "chosen by eye"
    for a, i in zip(axes.ravel(), idx):
        _image(a, imgs[i], extent, centre_xyz, lo, hi)
        _overlay(a, extent, centre_xyz, radius_um, roots_xy)
        dzk = ks[i] * float(dz_um) - float(centre_xyz[2])
        ref = i == int(j_ref)
        a.set_title(_plane_title(int(ks[i]), dzk, k_traced, ref and chosen), fontsize=7.5, color=INK)
        inside = k_range is not None and int(k_range[0]) <= ks[i] <= int(k_range[1])
        for side in a.spines.values():
            side.set_color(ORANGE if ref else (INK if inside else INK2))
            side.set_linewidth(2.0 if ref else (1.4 if inside else 0.5))
        a.tick_params(labelsize=5.5, colors=INK2)
    legend = ("orange frame: %s (k = %d)" % ("the plane chosen by eye" if chosen else "the traced depth's plane",
                                             int(ks[int(j_ref)]))
              + ("; dark frames: the planes where you see the soma (k = %d to %d)" % (int(k_range[0]), int(k_range[1]))
                 if k_range is not None else ""))
    fig.suptitle((title + "\n" if title else "") + "dashed: the traced soma (Allen's radius); +: its centre; circles: "
                 "the primary dendrites' first traced nodes; grey %.0f-%.0f gl\n%s" % (lo, hi, legend), fontsize=9,
                 color=INK)
    return fig
