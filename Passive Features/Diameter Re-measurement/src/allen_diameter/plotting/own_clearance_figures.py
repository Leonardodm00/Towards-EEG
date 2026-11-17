"""The same-node clearance figures -- SPEC Block 38 (D-156.2). Computes nothing scientific.

Plotting only: every array arrives prepared (Block 37's OwnClearance, Block 35's
split by contract); no input array is written to and nothing is derived from
the inputs but the drawing range.

  figure_own     one node: per plane shown, the profile I_k(v_n), the cleared
                 profile I + B S_out, the in-band model B (1 - S_in), the whole
                 model B (1 - S_in - S_out), the background B_k as a level line
                 and the chord c_k as a horizontal bar from v0 - c_k / 2 to
                 v0 + c_k / 2 (no bar where c_k is 0); one y range for all
                 panels; one legend for the figure
  figure_layers  one plane: the image of B_k L_{m,k}(v_n) (gl) over the offset
                 v and the layer depth, a colour bar in gl, the plane's depth
                 z_k and its band z_k +- beta drawn as lines
  save           the PNG of a figure

Conventions read from SPEC.md: profiles (n_planes, n_v) in grey levels (gl) on
the offsets v (um) of Block 5; ks the plane indices k (values, not positions);
planes_z (n_planes,) um; layers_gl (n_layers, n_planes, n_v) gl with layer
centres layer_z (n_layers,) um.

meta keys read (each optional): node_id, class, round, d_um (the estimate's
diameter, um), mu_per_um (its coefficient, 1/um), band_half_um (the band
half-thickness beta, um).

Colours: the palette of the project's figures (cleaning_figures, inference_
figures). Identity never rests on colour alone: profile = markers, cleared =
solid orange, in-band model = dashed blue, whole model = solid blue,
background = dotted grey, chord = thick ink bar with end ticks.
Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import math
import os
from collections.abc import Mapping

import numpy as np

INK, INK2, BLUE, ORANGE = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834"
SURFACE, GRID = "#fcfcfb", "#e4e3df"
OWN_COLS = 4               # panels per row of figure_own before it wraps
Y_PAD = 0.06               # share of the data span added above and below in figure_own
BAR_GAP = 0.05             # the chord bar sits this share of the span below the lowest drawn value
LAYER_CMAP = "Blues"       # the layers' shadow, 0 to its largest value (as figures.py draws a deficit)


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
    """The figure title from meta: node_id, class, round; d_um, mu_per_um, band_half_um (each optional)."""
    head = []
    if "node_id" in meta:
        head.append("node %s" % _fmt(meta["node_id"]))
    if "class" in meta:
        head.append("class %s" % _fmt(meta["class"]))
    if "round" in meta:
        head.append("round %s" % _fmt(meta["round"]))
    feats = []
    if "d_um" in meta:
        feats.append("d = %s um" % _fmt(meta["d_um"]))
    if "mu_per_um" in meta:
        feats.append("mu = %s /um" % _fmt(meta["mu_per_um"]))
    if "band_half_um" in meta:
        feats.append("band +-%s um" % _fmt(meta["band_half_um"]))
    lines = ["  |  ".join(head) if head else "node"]
    if feats:
        lines.append(",  ".join(feats))
    return "\n".join(lines)


def _plane_cols(ks, planes):
    """Positions along the plane axis of the planes to show (plane indices k taken from ks)."""
    ks = [int(k) for k in ks]
    if planes is None:
        return list(range(len(ks)))
    wanted = [int(k) for k in np.asarray(planes).reshape(-1)]
    if not wanted:
        raise ValueError("planes is empty")
    out = []
    for k in wanted:
        if k not in ks:
            raise ValueError("plane %d is not in ks %r" % (k, ks))
        out.append(ks.index(k))
    return out


def _arr(name, x, shape):
    a = np.asarray(x, dtype=float)
    if a.shape != shape:
        raise ValueError("%s must have shape %r, got %r" % (name, shape, a.shape))
    return a


def figure_own(v, ks, profiles_gl, B_gl, clearance, meta, planes=None, v0_um=0.0):
    """The same-node clearance of one node. Returns the matplotlib Figure.

    v: (n_v,) offsets along the line, um; ks: (n_planes,) plane indices k.
    profiles_gl: (n_planes, n_v) gl, the profiles I_k(v_n) the clearance was made from.
    B_gl: (n_planes,) gl, the backgrounds B_k.
    clearance: a Block 37 own_clearance.OwnClearance (cleared_gl, model_in_gl,
      model_gl (n_planes, n_v) gl; chord_um, in_share (n_planes,)).
    meta: mapping for the title (module docstring: node_id, class, round,
      d_um, mu_per_um, band_half_um; each optional).
    planes: plane indices k (values of ks) to show, in that order; None = all.
    v0_um: the estimate's lateral offset v0 (um), the centre of the chord bar.

    One panel per plane shown (at most OWN_COLS per row, rows balanced), all on
    one y range: the data of the panels shown and their backgrounds, padded by
    Y_PAD, and below them the chord bar (gid "chord"). Each panel's title:
    k, c_k (um) and the in-band share. A legend once, for the figure.
    ValueError for shapes that disagree, an empty `planes`, a plane not in ks,
    meta not a mapping, v0_um not finite.
    """
    v = np.asarray(v, dtype=float)
    ks = np.asarray(ks)
    if v.ndim != 1 or v.size == 0:
        raise ValueError("v must be (n_v,) with n_v > 0, got shape %r" % (v.shape,))
    if ks.ndim != 1 or ks.size == 0:
        raise ValueError("ks must be (n_planes,) with n_planes > 0, got shape %r" % (ks.shape,))
    n_planes, n_v = ks.size, v.size
    prof = _arr("profiles_gl", profiles_gl, (n_planes, n_v))
    B = _arr("B_gl", B_gl, (n_planes,))
    cleared = _arr("clearance.cleared_gl", clearance.cleared_gl, (n_planes, n_v))
    model_in = _arr("clearance.model_in_gl", clearance.model_in_gl, (n_planes, n_v))
    model = _arr("clearance.model_gl", clearance.model_gl, (n_planes, n_v))
    chord = _arr("clearance.chord_um", clearance.chord_um, (n_planes,))
    share = _arr("clearance.in_share", clearance.in_share, (n_planes,))
    if not isinstance(meta, Mapping):
        raise ValueError("meta must be a mapping (node_id, class, round, d_um, mu_per_um, band_half_um)")
    v0 = float(v0_um)
    if not math.isfinite(v0):
        raise ValueError("v0_um must be finite, got %r" % (v0_um,))
    cols = _plane_cols(ks, planes)

    # the one y range: the finite values of every curve shown and the backgrounds (display range only)
    vals = np.concatenate([np.concatenate([prof[j], cleared[j], model_in[j], model[j], B[j:j + 1]])
                           for j in cols])
    vals = vals[np.isfinite(vals)]
    lo, hi = (float(vals.min()), float(vals.max())) if vals.size else (0.0, 1.0)
    span = hi - lo if hi > lo else 1.0
    y_bar = lo - BAR_GAP * span
    ylim = (y_bar - Y_PAD * span, hi + Y_PAD * span)

    plt = _plt()
    from matplotlib.lines import Line2D
    n = len(cols)
    n_r = int(math.ceil(n / float(OWN_COLS)))
    n_c = int(math.ceil(n / float(n_r)))
    fig, axes = plt.subplots(n_r, n_c, figsize=(3.1 * n_c + 0.6, 2.9 * n_r + 1.4), squeeze=False, sharey=True,
                             facecolor=SURFACE, layout="constrained")
    flat = list(axes.ravel())
    for i, j in enumerate(cols):
        ax = flat[i]
        _style(ax)
        ax.axhline(B[j], color=INK2, lw=1.0, ls=":", zorder=1, gid="background")
        ax.plot(v, model[j], ls="-", lw=1.4, color=BLUE, zorder=2, gid="model")
        ax.plot(v, model_in[j], ls="--", lw=1.4, color=BLUE, zorder=2, gid="model_in")
        ax.plot(v, cleared[j], ls="-", lw=1.3, color=ORANGE, zorder=3, gid="cleared")
        ax.plot(v, prof[j], ls="none", marker="o", ms=2.6, color=INK, zorder=4, gid="profile")
        c = float(chord[j])
        if c > 0.0:
            ax.plot([v0 - c / 2.0, v0 + c / 2.0], [y_bar, y_bar], ls="-", lw=3.0, color=INK,
                    marker="|", ms=9, mew=1.5, solid_capstyle="butt", zorder=5, gid="chord")
        ax.set_xlim(float(v.min()), float(v.max()))
        ax.set_ylim(*ylim)
        sh = float(share[j])
        sh_txt = "%.3f" % sh if math.isfinite(sh) else "n/a"
        ax.set_title("k = %d,  c_k = %.3g um,  in-band share %s" % (int(ks[j]), c, sh_txt), fontsize=8.0,
                     color=INK)
        r, cc = divmod(i, n_c)
        if r == n_r - 1 or i + n_c >= n:
            ax.set_xlabel("offset v (um)", fontsize=7.5, color=INK2)
        if cc == 0:
            ax.set_ylabel("grey level (gl)", fontsize=7.5, color=INK2)
    for ax in flat[n:]:
        ax.set_visible(False)
    handles = [
        Line2D([], [], ls="none", marker="o", ms=4, color=INK, label="profile I_k"),
        Line2D([], [], ls="-", lw=1.3, color=ORANGE, label="cleared I_k + B_k S_out"),
        Line2D([], [], ls="--", lw=1.4, color=BLUE, label="in-band model B_k (1 - S_in)"),
        Line2D([], [], ls="-", lw=1.4, color=BLUE, label="whole model B_k (1 - S_in - S_out)"),
        Line2D([], [], ls=":", lw=1.0, color=INK2, label="background B_k"),
        Line2D([], [], ls="-", lw=3.0, color=INK, marker="|", ms=9, mew=1.5,
               label="chord c_k, centred at v0 = %s um" % _fmt(v0)),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=7.5, frameon=False)
    fig.suptitle(_title(meta), fontsize=9.5, color=INK)
    return fig


def figure_layers(v, layer_z, layers_gl, ks, plane, planes_z, band_half_um, meta):
    """The own shadow of one plane split into depth layers. Returns the matplotlib Figure.

    v: (n_v,) offsets along the line, um; layer_z: (n_layers,) the layers'
      centre depths, um; layers_gl: (n_layers, n_planes, n_v) gl, B_k L_{m,k}(v_n)
      (Block 37's OwnClearance.layers_gl); ks: (n_planes,) plane indices k;
      plane: the plane drawn (a value of ks); planes_z: (n_planes,) um;
      band_half_um: the band half-thickness beta, um; meta: mapping for the
      title, as figure_own's.

    One axis: the image of layers_gl[:, plane, :] (matplotlib pcolormesh,
    shading "nearest": cell m, n centred on (v_n, layer_z_m); gid "layers"),
    colour scale 0 to its largest value, a colour bar in gl; the plane's depth
    z_k (solid, gid "plane_z") and its band z_k - beta, z_k + beta (dashed,
    gid "band") as horizontal lines. Depth increases upwards.
    ValueError for layers_gl None, shapes that disagree, no layer, a plane not
    in ks, band_half_um negative or not finite, meta not a mapping.
    """
    if layers_gl is None:
        raise ValueError("layers_gl is None: the split was made without layers")
    v = np.asarray(v, dtype=float)
    ks = np.asarray(ks)
    lz = np.asarray(layer_z, dtype=float)
    if v.ndim != 1 or v.size == 0:
        raise ValueError("v must be (n_v,) with n_v > 0, got shape %r" % (v.shape,))
    if ks.ndim != 1 or ks.size == 0:
        raise ValueError("ks must be (n_planes,) with n_planes > 0, got shape %r" % (ks.shape,))
    if lz.ndim != 1 or lz.size == 0:
        raise ValueError("layer_z must be (n_layers,) with n_layers > 0, got shape %r" % (lz.shape,))
    n_layers, n_planes, n_v = lz.size, ks.size, v.size
    lay = _arr("layers_gl", layers_gl, (n_layers, n_planes, n_v))
    pz = _arr("planes_z", planes_z, (n_planes,))
    beta = float(band_half_um)
    if not (math.isfinite(beta) and beta >= 0.0):
        raise ValueError("band_half_um must be finite and >= 0, got %r" % (band_half_um,))
    if not isinstance(meta, Mapping):
        raise ValueError("meta must be a mapping (node_id, class, round, d_um, mu_per_um, band_half_um)")
    j = _plane_cols(ks, [plane])[0]
    img = lay[:, j, :]
    z_k = float(pz[j])

    finite = img[np.isfinite(img)]
    vmax = float(finite.max()) if finite.size else 0.0
    vmax = vmax if vmax > 0.0 else 1.0

    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.6, 4.6), facecolor=SURFACE, layout="constrained")
    _style(ax)
    ax.grid(False)
    mesh = ax.pcolormesh(v, lz, img, shading="nearest", cmap=LAYER_CMAP, vmin=0.0, vmax=vmax, gid="layers")
    cb = fig.colorbar(mesh, ax=ax)
    cb.set_label("layer shadow B_k L_m,k (gl)", fontsize=8.0, color=INK2)
    cb.ax.tick_params(labelsize=7, colors=INK2)
    ax.axhline(z_k, color=ORANGE, lw=1.4, ls="-", gid="plane_z", label="plane depth z_k = %s um" % _fmt(z_k))
    ax.axhline(z_k - beta, color=ORANGE, lw=1.1, ls="--", gid="band",
               label="band z_k -+ beta, beta = %s um" % _fmt(beta))
    ax.axhline(z_k + beta, color=ORANGE, lw=1.1, ls="--", gid="band")
    ax.set_xlabel("offset v (um)", fontsize=8.0, color=INK2)
    ax.set_ylabel("layer depth z (um)", fontsize=8.0, color=INK2)
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.85)
    ax.set_title("plane k = %d: the own shadow by depth layer (%d layers)" % (int(ks[j]), n_layers),
                 fontsize=8.5, color=INK)
    fig.suptitle(_title(meta), fontsize=9.5, color=INK)
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
