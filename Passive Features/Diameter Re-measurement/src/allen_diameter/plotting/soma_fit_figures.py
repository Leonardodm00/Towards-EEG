"""The soma fit's figures -- Block 48 in specs/SPEC.md (D-215, D-222.5).

Figures only: every number drawn is passed in -- the entry point builds them from Block 46's fit (the model planes
B_k tau_k from Block 45 at the fitted theta), Block 47's outlines and its SomaEdit -- and nothing scientific is computed
here; no input is written to.

  figure_soma_fit_planes   one row per plane shown: the data, the model and the residual data - model (a diverging map
                           symmetric about 0, one scale for every row); the fit region's boundary on every panel, the
                           masked pixels dimmed; on the data panel the fitted silhouette (dashed), the plane's section
                           (solid), Allen's circle (dotted) and the sphere of equal area (dash-dot)
  figure_soma_fit_summary  the parameters' table (the held ones marked), the derived area, volume and r_eq with their
                           SEs beside Allen's radius and 4 pi r^2, and the planes' scales B_k and the residual RMS per
                           plane against depth (the noise SD drawn when given)
  figure_soma_morphology   the top view (x, y) and the side view (u, w): Allen's soma, the fitted outline, the sphere of
                           equal area, and per stem its old first node (open), the new one (filled), an arrow from old
                           to new and its next nodes as a line; a flag other than "moved" named in the legend
  save                     the PNG of a figure (plotting.cleaning_figures.save)

Image panels in the image frame, um, with matplotlib's extent (left, right, bottom, top) and origin "upper", as Block
44's figures (plotting.figures.image_extent_um: pixel (row, col) of a crop with corner (left, top) centred on
((left + col) p, (top + row) p); y downwards, as in the image). Styling constants below; the palette is the project's
(plotting.cleaning_figures, Block 44's yellow for Allen's soma, plotting.figures' diverging blue - grey - red).
Pure ASCII, LF, for the cluster.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from allen_diameter.plotting.cleaning_figures import (BLUE, IMAGE_PERCENTILES, INK, INK2, ORANGE, SURFACE, _plt,
                                                      _row_scale, _style, save)
from allen_diameter.plotting.figures import _DIV
from allen_diameter.plotting.soma_figures import SOMA_COLOUR

__all__ = ["figure_soma_fit_planes", "figure_soma_fit_summary", "figure_soma_morphology", "save"]

FIT_COLOUR = ORANGE              # the fitted ellipsoid: its silhouette (dashed) and the plane's section (solid)
EQ_COLOUR = BLUE                 # the sphere of equal area (dash-dot)
ALLEN_COLOUR = SOMA_COLOUR       # Allen's traced soma on the images (dotted; Block 44's yellow)
ALLEN_COLOUR_LIGHT = "#b8860b"   # the same on the white views of the morphology figure (a darker gold)
REGION_COLOUR = INK              # the fit region's boundary (with a white halo)
MASK_DIM = ("#6f6e69", 0.55)     # the masked pixels: this grey laid over them at this opacity
RESIDUAL_COLOURS = _DIV          # the residual's diverging map: blue (data darker) - neutral grey (0) - red
RESIDUAL_PERCENTILE = 99.0       # its half-range: this percentile of |data - model| over the fitted pixels (below)
STYLES = {"silhouette": "--", "section": "-", "allen": ":", "sphere": "-."}
MOVED_COLOUR = INK               # a stem flagged "moved"
FLAG_COLOURS = {"on surface": "#1baf7a", "missed": "#e34948", "branch inside": "#7b3fb5", "ends inside": "#eda100"}
OTHER_FLAG_COLOUR = INK2         # any other flag
VALUE_FMT, SE_FMT, DERIVED_FMT = "%.6g", "%.2g", "%.4g"
NO_VALUE = "--"                  # a table cell with nothing passed (an SE not given)

# the summary's dicts: required keys, then optional ones ("--" when absent or None). The fit's own fields carry
# these names (Block 46's SomaFit: area_um2, se_area_um2, volume_um3, r_eq_um); the other SEs are not among them.
DERIVED_KEYS = ("area_um2", "volume_um3", "r_eq_um")
DERIVED_SE_KEYS = ("se_area_um2", "se_volume_um3", "se_r_eq_um")
ALLEN_KEYS = ("radius_um", "area_um2")         # Allen's traced radius r and 4 pi r^2
ALLEN_OPTIONAL_KEYS = ("volume_um3",)          # 4/3 pi r^3, beside the fitted volume
VIEW_KEYS = ("allen", "silhouette", "sphere", "stems")
STEM_KEYS = ("old", "new", "next", "flag")


# ------------------------------------------------------------- the inputs ---

def _xy(a, name):
    """(2, n) float64 points (n >= 0)."""
    arr = np.asarray(a, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != 2:
        raise ValueError("%s must be (2, n) um, got shape %r" % (name, arr.shape))
    return arr


def _vector(a, n, name):
    arr = np.asarray(a, dtype=float)
    if arr.shape != (n,):
        raise ValueError("%s must be (%d,), got shape %r" % (name, n, arr.shape))
    return arr


def _bool_image(a, shape, name):
    arr = np.asarray(a)
    if arr.dtype != bool or arr.shape != shape:
        raise ValueError("%s must be a bool array of shape %r, got %s %r" % (name, shape, arr.dtype, arr.shape))
    return arr


def _plane_indices(ks, n):
    arr = np.asarray(ks)
    if arr.shape != (n,):
        raise ValueError("ks must be (%d,), got shape %r" % (n, arr.shape))
    if arr.dtype.kind not in "iu":
        if arr.dtype.kind != "f" or not np.all(np.isfinite(arr)) or not np.all(arr == np.round(arr)):
            raise ValueError("ks must hold integer plane indices, got %r" % (arr,))
    return [int(k) for k in arr]


def _circle(c, name):
    arr = np.asarray(c, dtype=float)
    if arr.shape != (3,):
        raise ValueError("%s must be (x, y, r) um, got shape %r" % (name, arr.shape))
    if arr[2] < 0.0:
        raise ValueError("%s: the radius must be >= 0 (or NaN), got %r" % (name, float(arr[2])))
    return arr


def _sections(sections, n):
    if sections is None:
        return [None] * n, False
    if isinstance(sections, (str, bytes, Mapping)):
        raise ValueError("sections_xy must be a list of %d arrays (2, n_k) or None" % n)
    items = list(sections)
    if len(items) != n:
        raise ValueError("sections_xy must hold one entry per plane shown (%d), got %d" % (n, len(items)))
    return [None if s is None else _xy(s, "sections_xy[%d]" % j) for j, s in enumerate(items)], True


def _scalar(x, name):
    try:
        return float(x)
    except (TypeError, ValueError):
        raise ValueError("%s must be a number, got %r" % (name, x)) from None


def _numbers(d, required, optional, name):
    if not isinstance(d, Mapping):
        raise ValueError("%s must be a dict with the keys %s" % (name, ", ".join(required)))
    out = {}
    for key in required:
        if key not in d or d[key] is None:
            raise ValueError("%s lacks %r (required: %s; optional: %s)" % (name, key, ", ".join(required),
                                                                           ", ".join(optional)))
        out[key] = _scalar(d[key], "%s[%r]" % (name, key))
    for key in optional:
        out[key] = None if d.get(key) is None else _scalar(d[key], "%s[%r]" % (name, key))
    return out


def _names(names):
    if isinstance(names, (str, bytes)):
        raise ValueError("names must be a sequence of parameter names, got one string")
    nm = tuple(names)
    if not nm or not all(isinstance(n, str) for n in nm) or len(set(nm)) != len(nm):
        raise ValueError("names must be distinct strings (at least one), got %r" % (nm,))
    return nm


def _held(fitted, names):
    if isinstance(fitted, (str, bytes)):
        raise ValueError("fitted must be a sequence of parameter names, got one string")
    fit = set(fitted)
    unknown = sorted(str(n) for n in fit if n not in names)
    if unknown:
        raise ValueError("fitted names parameters not in names: %s" % ", ".join(unknown))
    return [n not in fit for n in names]


def _view(view, name):
    if not isinstance(view, Mapping) or any(k not in view for k in VIEW_KEYS):
        raise ValueError("%s must be a dict with the keys %s" % (name, ", ".join(VIEW_KEYS)))
    out = {k: _xy(view[k], "%s[%r]" % (name, k)) for k in VIEW_KEYS[:3]}
    stems = view["stems"]
    if isinstance(stems, (str, bytes, Mapping)):
        raise ValueError("%s['stems'] must be a list of dicts (%s)" % (name, ", ".join(STEM_KEYS)))
    out["stems"] = []
    for i, st in enumerate(list(stems)):
        where = "%s['stems'][%d]" % (name, i)
        if not isinstance(st, Mapping) or any(k not in st for k in STEM_KEYS):
            raise ValueError("%s must be a dict with the keys %s" % (where, ", ".join(STEM_KEYS)))
        old = _vector(st["old"], 2, where + "['old']")
        new = _vector(st["new"], 2, where + "['new']")
        if not isinstance(st["flag"], str):
            raise ValueError("%s['flag'] must be a string, got %r" % (where, st["flag"]))
        out["stems"].append(dict(old=old, new=new, next=_xy(st["next"], where + "['next']"), flag=st["flag"]))
    return out


# ------------------------------------------------------------- the planes ---

def _residual_half_range(resid, fitted):
    """The half-range h of the residual's scale [-h, h]: the RESIDUAL_PERCENTILE-th percentile (numpy's, linear) of
    |data - model| over the fitted pixels (region and not masked) of every plane together; over every finite pixel when
    the fitted ones hold none; 1 when that is 0 or there is none."""
    vals = np.abs(resid[:, fitted]).ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        vals = np.abs(resid[np.isfinite(resid)])
    half = float(np.percentile(vals, RESIDUAL_PERCENTILE)) if vals.size else 0.0
    return half if half > 0.0 else 1.0


def figure_soma_fit_planes(data, model, region, mask, ks, dz_from_centre_um, extent_um, silhouette_xy, sections_xy,
                           allen_circle, eq_circle, title=""):
    """The fit's planes. Returns the matplotlib Figure (3 K image axes and the residual's colour bar).

    data, model: (K, H, W) grey levels on the K planes shown (any real dtype; model = B_k tau_k at the fitted theta);
    region, mask: (H, W) bool -- the fit region (its boundary drawn on every panel) and the pixels left out (dimmed);
    ks: (K,) plane indices; dz_from_centre_um: (K,) each plane's depth minus the fitted z0 (um);
    extent_um: (4,) (left, right, bottom, top) um, the crop's imshow extent in the image frame;
    silhouette_xy: (2, n) um; sections_xy: a list of K arrays (2, n_k) or None per plane (None: the plane misses the
    soma), or None for no sections; allen_circle, eq_circle: (x, y, r) um, Allen's traced soma and the sphere of equal
    area.
    One row per plane: data and model on one grey range (numpy's 1st and 99th percentiles of the two together), the
    residual data - model on one diverging scale symmetric about 0 for every row (half-range: the RESIDUAL_PERCENTILE-th
    percentile of |data - model| over the fitted pixels, region and not masked, of every plane; beyond it the colours
    saturate). ValueError for shapes that disagree, masks not bool, ks not integers, a non-finite extent or a radius
    < 0.
    """
    D = np.asarray(data, dtype=float)
    M = np.asarray(model, dtype=float)
    if D.ndim != 3 or D.shape[0] < 1:
        raise ValueError("data must be (K, H, W) with K >= 1, got shape %r" % (D.shape,))
    if M.shape != D.shape:
        raise ValueError("model %r and data %r must have one shape (K, H, W)" % (M.shape, D.shape))
    K, H, W = D.shape
    reg = _bool_image(region, (H, W), "region")
    msk = _bool_image(mask, (H, W), "mask")
    kk = _plane_indices(ks, K)
    dz = _vector(dz_from_centre_um, K, "dz_from_centre_um")
    ext = _vector(extent_um, 4, "extent_um")
    if not np.all(np.isfinite(ext)) or ext[0] == ext[1] or ext[2] == ext[3]:
        raise ValueError("extent_um must be finite (left, right, bottom, top) with left != right, bottom != top")
    ext = tuple(float(e) for e in ext)
    sil = _xy(silhouette_xy, "silhouette_xy")
    secs, have_secs = _sections(sections_xy, K)
    allen = _circle(allen_circle, "allen_circle")
    eq = _circle(eq_circle, "eq_circle")

    resid = D - M                                          # a new array: data - model (gl)
    half = _residual_half_range(resid, reg & ~msk)
    plt = _plt()
    import matplotlib.patheffects as pe
    from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgba
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, Patch
    cmap = LinearSegmentedColormap.from_list("soma_fit_residual", RESIDUAL_COLOURS)
    norm = Normalize(vmin=-half, vmax=half)                # one scale object shared by every residual panel
    dim = np.zeros((H, W, 4))
    dim[msk] = to_rgba(MASK_DIM[0], MASK_DIM[1])
    draw_region = bool(reg.any()) and not bool(reg.all())
    halo = [pe.Stroke(linewidth=2.6, foreground="white"), pe.Normal()]

    side = 2.6
    fig, axes = plt.subplots(K, 3, figsize=(3 * side + 1.6, (side + 0.3) * K + 1.5), facecolor=SURFACE,
                             squeeze=False, layout="constrained")
    for j in range(K):
        lo, hi = _row_scale((D[j], M[j]), IMAGE_PERCENTILES)
        panels = ((D[j], dict(cmap="gray", vmin=lo, vmax=hi), "data"),
                  (M[j], dict(cmap="gray", vmin=lo, vmax=hi), "model"),
                  (resid[j], dict(cmap=cmap, norm=norm), "residual"))
        for c, (img, kw, gid) in enumerate(panels):
            ax = axes[j, c]
            ax.set_facecolor(SURFACE)
            im = ax.imshow(img, extent=ext, origin="upper", interpolation="nearest", **kw)
            im.set_gid(gid)
            if msk.any():
                ax.imshow(dim, extent=ext, origin="upper", interpolation="nearest", zorder=2).set_gid("masked")
            if draw_region:
                cs = ax.contour(reg.astype(float), levels=[0.5], origin="upper", extent=ext, colors=[REGION_COLOUR],
                                linewidths=0.9, zorder=4)
                cs.set_gid("region")
                cs.set_path_effects(halo)
            ax.tick_params(labelsize=6, colors=INK2)
        a0 = axes[j, 0]
        a0.plot(sil[0], sil[1], ls=STYLES["silhouette"], lw=1.3, color=FIT_COLOUR, path_effects=halo, zorder=5,
                gid="silhouette")
        if secs[j] is not None and secs[j].shape[1]:
            a0.plot(secs[j][0], secs[j][1], ls=STYLES["section"], lw=1.3, color=FIT_COLOUR, path_effects=halo,
                    zorder=5, gid="section")
        for (cx, cy, r), colour, key, lw in ((allen, ALLEN_COLOUR, "allen", 1.6), (eq, EQ_COLOUR, "sphere", 1.3)):
            a0.add_patch(Circle((cx, cy), r, fill=False, ls=STYLES[key], lw=lw, edgecolor=colour, path_effects=halo,
                                zorder=5, gid=key))
        for ax in axes[j]:
            ax.set_xlim(ext[0], ext[1])
            ax.set_ylim(ext[2], ext[3])
        misses = have_secs and (secs[j] is None or secs[j].shape[1] == 0)
        a0.set_title("k = %d   z - z0 = %+.2f um   data%s" % (kk[j], dz[j], "  (misses the fitted soma)" if misses
                                                               else ""), fontsize=8, color=INK)
        axes[j, 1].set_title("model B_k tau_k", fontsize=8, color=INK)
        axes[j, 2].set_title("residual data - model", fontsize=8, color=INK)
        a0.set_ylabel("y (um)\ngrey %.0f-%.0f gl" % (lo, hi), fontsize=7, color=INK2)
    for ax in axes[-1]:
        ax.set_xlabel("x (um)", fontsize=7, color=INK2)
    cb = fig.colorbar(axes[0, 2].images[0], ax=list(axes[:, 2]), location="right", extend="both", pad=0.02,
                      shrink=0.9, aspect=max(12, 14 * K))
    cb.set_label("data - model (gl), one scale for every row", fontsize=7.5, color=INK2)
    cb.ax.tick_params(labelsize=6.5, colors=INK2)
    handles = [Line2D([], [], color=FIT_COLOUR, ls=STYLES["silhouette"], lw=1.3, label="the fitted silhouette"),
               Line2D([], [], color=FIT_COLOUR, ls=STYLES["section"], lw=1.3,
                      label="the fitted section at the plane's depth"),
               Line2D([], [], color=ALLEN_COLOUR, ls=STYLES["allen"], lw=1.6, label="Allen's traced soma"),
               Line2D([], [], color=EQ_COLOUR, ls=STYLES["sphere"], lw=1.3, label="the sphere of equal area"),
               Line2D([], [], color=REGION_COLOUR, ls="-", lw=0.9, label="the fit region's boundary"),
               Patch(facecolor=MASK_DIM[0], alpha=MASK_DIM[1], label="masked pixels (dimmed)")]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=7.5, frameon=False)
    fig.suptitle((title + "\n" if title else "")
                 + "residual: one scale for every row, +-%.3g gl, percentile %g of |data - model| over the fitted "
                   "pixels (the region, not masked)\ngrey: per row, percentiles %g to %g of the data and the model "
                   "together" % (half, RESIDUAL_PERCENTILE, IMAGE_PERCENTILES[0], IMAGE_PERCENTILES[1]),
                 fontsize=8.5, color=INK)
    return fig


# ------------------------------------------------------------ the summary ---

def _cell(x, fmt):
    return NO_VALUE if x is None else fmt % float(x)


def _table(ax, rows, columns, widths):
    ax.set_axis_off()
    tab = ax.table(cellText=rows, colLabels=columns, colWidths=widths, loc="upper center", cellLoc="left",
                   colLoc="left")
    tab.auto_set_font_size(False)
    tab.set_fontsize(8)
    tab.scale(1.0, 1.4)
    for (r, _c), cell in tab.get_celld().items():
        cell.set_edgecolor("#e4e3df")
        cell.get_text().set_color(INK)
        if r == 0:
            cell.get_text().set_fontweight("bold")
    return tab


def figure_soma_fit_summary(names, theta, se, fitted, derived, allen, z_um, B_gl, rms_gl, noise_gl=None, title=""):
    """The fit's summary. Returns the matplotlib Figure (4 axes: two tables, two panels).

    names: (P,) the parameters' names (Block 46's PARAMS); theta, se: (P,) their values and SEs in the names' units;
    fitted: the names fitted -- the others are marked held (their SE cell reads "held");
    derived: dict of the fitted area_um2, volume_um3, r_eq_um (required) and se_area_um2, se_volume_um3, se_r_eq_um
    (optional: "--" when absent or None); allen: dict of radius_um and area_um2 = 4 pi r^2 (required) and volume_um3
    (optional), Allen's traced sphere, shown beside them; z_um: (K,) the planes' depths (um, stage units); B_gl: (K,)
    the planes' scales B_k (gl); rms_gl: (K,) the residual RMS per plane (gl, >= 0); noise_gl: None or the noise SD
    (gl), a horizontal line on the RMS panel. ValueError for lengths that disagree, a name fitted that is not in names,
    a dict without a required key or with a non-number, an RMS < 0, a noise_gl not finite and >= 0.
    """
    nm = _names(names)
    P = len(nm)
    th = _vector(theta, P, "theta")
    s = _vector(se, P, "se")
    held = _held(fitted, nm)
    der = _numbers(derived, DERIVED_KEYS, DERIVED_SE_KEYS, "derived")
    al = _numbers(allen, ALLEN_KEYS, ALLEN_OPTIONAL_KEYS, "allen")
    z = np.asarray(z_um, dtype=float)
    if z.ndim != 1 or z.size < 1:
        raise ValueError("z_um must be (K,) with K >= 1, got shape %r" % (z.shape,))
    B = _vector(B_gl, z.size, "B_gl")
    rms = _vector(rms_gl, z.size, "rms_gl")
    if np.any(rms < 0.0):
        raise ValueError("rms_gl must be >= 0 (or NaN)")
    noise = None if noise_gl is None else _scalar(noise_gl, "noise_gl")
    if noise is not None and not (np.isfinite(noise) and noise >= 0.0):
        raise ValueError("noise_gl must be None or finite and >= 0, got %r" % (noise_gl,))

    plt = _plt()
    fig = plt.figure(figsize=(12.6, 7.2), facecolor=SURFACE, layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=(1.0, 1.1), height_ratios=(1.3, 1.0))
    a_par, a_der = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
    a_B = fig.add_subplot(gs[0, 1])
    a_rms = fig.add_subplot(gs[1, 1], sharex=a_B)

    rows = [[n, VALUE_FMT % v, "held" if h else SE_FMT % e] for n, v, e, h in zip(nm, th, s, held)]
    tab = _table(a_par, rows, ("parameter", "value", "SE"), (0.46, 0.3, 0.2))
    for i, h in enumerate(held):
        if h:
            for c in range(3):
                tab[(i + 1, c)].get_text().set_color(INK2)
                tab[(i + 1, c)].get_text().set_fontstyle("italic")
    tab.set_gid("parameters")
    a_par.set_title("the fitted parameters: %d fitted, %d held (italic)" % (P - sum(held), sum(held)), fontsize=9,
                    color=INK)
    drows = [["area (um^2): S | 4 pi r^2", _cell(der["area_um2"], DERIVED_FMT), _cell(der["se_area_um2"], SE_FMT),
              _cell(al["area_um2"], DERIVED_FMT)],
             ["volume (um^3): V | 4/3 pi r^3", _cell(der["volume_um3"], DERIVED_FMT),
              _cell(der["se_volume_um3"], SE_FMT), _cell(al["volume_um3"], DERIVED_FMT)],
             ["radius (um): r_eq = sqrt(S / 4 pi) | r", _cell(der["r_eq_um"], DERIVED_FMT),
              _cell(der["se_r_eq_um"], SE_FMT), _cell(al["radius_um"], DERIVED_FMT)]]
    _table(a_der, drows, ("derived | Allen's sphere", "fitted", "SE", "Allen"), (0.5, 0.17, 0.13, 0.17)).set_gid(
        "derived")
    a_der.set_title("the fitted soma beside Allen's traced sphere (radius r)", fontsize=9, color=INK)

    _style(a_B)
    a_B.plot(z, B, color=BLUE, lw=1.3, marker="o", ms=3.5, gid="B")
    a_B.set_ylabel("plane scale B_k (gl)", fontsize=8, color=INK2)
    a_B.set_title("the planes' scales B_k", fontsize=9, color=INK)
    _style(a_rms)
    a_rms.plot(z, rms, color=INK, lw=1.3, marker="o", ms=3.5, gid="rms", label="residual RMS")
    if noise is not None:
        a_rms.axhline(noise, color=ORANGE, ls="--", lw=1.2, gid="noise", label="noise SD %.3g gl" % noise)
    a_rms.set_ylim(bottom=0.0)
    a_rms.legend(fontsize=7.5, frameon=False, loc="best")
    a_rms.set_xlabel("plane depth z (um, stage units)", fontsize=8, color=INK2)
    a_rms.set_ylabel("residual RMS per plane (gl)", fontsize=8, color=INK2)
    a_rms.set_title("the residual RMS per plane", fontsize=9, color=INK)
    if title:
        fig.suptitle(title, fontsize=10, color=INK)
    return fig


# -------------------------------------------------------- the morphology ---

def _flag_colour(flag):
    return MOVED_COLOUR if flag == "moved" else FLAG_COLOURS.get(flag, OTHER_FLAG_COLOUR)


def figure_soma_morphology(top, side, title=""):
    """The soma in the morphology. Returns the matplotlib Figure (2 axes: the top view, the side view).

    top, side: dicts of (2, n) arrays `allen` (Allen's traced soma), `silhouette` (the fitted ellipsoid's outline),
    `sphere` (the sphere of equal area), and `stems`, a list of dicts -- `old` (2,), `new` (2,) the stem's first node
    before and after, `next` (2, m) its next nodes, `flag` (Block 47's: "moved", "on surface", "missed", "branch
    inside", "ends inside") -- the same stems in the same order in both views; top in the image frame (x, y) um, y
    downwards as in the image, side in the fitted ellipsoid's frame (u, w) um. The old node an open circle, the new one
    filled, an arrow from old to new where they differ, the next nodes as a line; a flag other than "moved" in the
    legend. ValueError for a missing key, shapes that disagree, stems that differ in number or flags between the views.
    """
    views = (_view(top, "top"), _view(side, "side"))
    if len(views[0]["stems"]) != len(views[1]["stems"]):
        raise ValueError("top and side must hold the same stems (%d and %d)" % (len(views[0]["stems"]),
                                                                              len(views[1]["stems"])))
    if [st["flag"] for st in views[0]["stems"]] != [st["flag"] for st in views[1]["stems"]]:
        raise ValueError("top and side must hold the same stems in the same order (their flags differ)")
    plt = _plt()
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.4), facecolor=SURFACE, squeeze=False, layout="constrained")
    labels = (("x (um, image frame)", "y (um, image frame; downwards as in the image)",
               "top view (x, y)"),
              ("u (um, along the fitted in-plane axis a)", "w = z - z0 (um, stage units)",
               "side view (u, w) in the fitted ellipsoid's frame"))
    for ax, view, (xl, yl, tt) in zip(axes[0], views, labels):
        _style(ax)
        for key, colour, lw in (("allen", ALLEN_COLOUR_LIGHT, 1.6), ("silhouette", FIT_COLOUR, 1.4),
                                ("sphere", EQ_COLOUR, 1.4)):
            ax.plot(view[key][0], view[key][1], ls=STYLES[key], lw=lw, color=colour, gid=key)
        for st in view["stems"]:
            col = _flag_colour(st["flag"])
            old, new, nxt = st["old"], st["new"], st["next"]
            if nxt.shape[1]:
                ax.plot(nxt[0], nxt[1], color=col, lw=1.0, marker=".", ms=3.5, gid="next")
            ax.plot([old[0]], [old[1]], ls="none", marker="o", ms=6.5, mfc="none", mec=col, mew=1.2, gid="old")
            ax.plot([new[0]], [new[1]], ls="none", marker="o", ms=5.0, mfc=col, mec=col, gid="new")
            if np.all(np.isfinite(old)) and np.all(np.isfinite(new)) and np.any(old != new):
                ax.annotate("", xy=tuple(new), xytext=tuple(old), arrowprops=dict(arrowstyle="->", color=col, lw=1.0,
                                                                                shrinkA=0.0, shrinkB=0.0),
                            zorder=4).set_gid("arrow")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel(xl, fontsize=8, color=INK2)
        ax.set_ylabel(yl, fontsize=8, color=INK2)
        ax.set_title(tt, fontsize=9, color=INK)
    axes[0, 0].invert_yaxis()
    handles = [Line2D([], [], color=ALLEN_COLOUR_LIGHT, ls=STYLES["allen"], lw=1.6, label="Allen's traced soma"),
               Line2D([], [], color=FIT_COLOUR, ls=STYLES["silhouette"], lw=1.4, label="the fitted ellipsoid"),
               Line2D([], [], color=EQ_COLOUR, ls=STYLES["sphere"], lw=1.4, label="the sphere of equal area"),
               Line2D([], [], ls="none", marker="o", ms=6.5, mfc="none", mec=MOVED_COLOUR, mew=1.2,
                      label="a stem's first node, before"),
               Line2D([], [], ls="none", marker="o", ms=5.0, mfc=MOVED_COLOUR, mec=MOVED_COLOUR,
                      label="a stem's first node, after (arrow: the move)"),
               Line2D([], [], color=MOVED_COLOUR, lw=1.0, marker=".", ms=3.5, label="the stem's next nodes")]
    seen = []
    for st in views[0]["stems"]:
        if st["flag"] != "moved" and st["flag"] not in seen:
            seen.append(st["flag"])
    for flag in seen:
        col = _flag_colour(flag)
        handles.append(Line2D([], [], ls="none", marker="o", ms=5.0, mfc=col, mec=col, label="flag: %s" % flag))
    fig.legend(handles=handles, loc="outside lower center", ncol=min(4, len(handles)), fontsize=7.5, frameon=False)
    if title:
        fig.suptitle(title, fontsize=10, color=INK)
    return fig
