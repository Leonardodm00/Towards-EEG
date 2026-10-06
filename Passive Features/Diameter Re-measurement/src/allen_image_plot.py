"""
allen_image_plot.py -- rendering only. No IO, no measurement.

Design choices, and why:
  - Microscopy images get a SCALE BAR, not pixel-index axes. Pixel indices are
    an artefact of the crop; microns are the quantity being judged.
  - Grayscale, displayed with the data's own polarity (dark process on bright
    field). No colormap: any hue here would encode nothing.
  - The profile plot carries one data series, so it gets no legend -- the title
    names it. The half-maximum level and the FWHM span are annotation, drawn
    recessive, and labelled directly rather than through a legend.
"""
import numpy as np

INK = "#1a1a1a"
MUTED = "#6b6b6b"
ACCENT = "#c1440e"        # one accent, used only for the measured quantity
GRID = "#e6e6e6"


def _nice_bar_um(span_um):
    """A round scale-bar length covering ~1/5 of the field."""
    target = span_um / 5.0
    for v in (0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000):
        if v >= target:
            return v
    return 5000


def robust_levels(img, lo=1.0, hi=99.5):
    """Percentile clip. Single stack planes have low contrast that a handful of
    near-black specks would otherwise wash out entirely."""
    a = np.asarray(img)
    return float(np.percentile(a, lo)), float(np.percentile(a, hi))


def show_image(img, res_um_px, ax=None, title=None, scalebar_um=None,
               vmin=None, vmax=None, robust=False, figsize=(6, 6)):
    """One image with a scale bar. res_um_px is the scale OF THIS ARRAY
    (CropFrame.res_um_px), not of the full-resolution image."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    h, w = img.shape[:2]
    if robust and vmin is None and vmax is None:
        vmin, vmax = robust_levels(img)
    ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    bar_um = scalebar_um or _nice_bar_um(w * res_um_px)
    bar_px = bar_um / res_um_px
    x0, y0 = 0.04 * w, 0.95 * h
    ax.plot([x0, x0 + bar_px], [y0, y0], color="white", lw=4, solid_capstyle="butt",
            zorder=5)
    ax.plot([x0, x0 + bar_px], [y0, y0], color=INK, lw=2, solid_capstyle="butt",
            zorder=6)
    label = ("%g um" % bar_um) if bar_um >= 1 else ("%.1f um" % bar_um)
    ax.text(x0 + bar_px / 2, y0 - 0.02 * h, label, color=INK, ha="center",
            va="bottom", fontsize=9, zorder=6,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1.5))
    if title:
        ax.set_title(title, loc="left", fontsize=10, color=INK)
    return ax


def montage(images, res_um_px, titles=None, ncols=5, figsize_per=2.4,
            suptitle=None, share_levels=True, robust=False):
    """A grid of crops -- e.g. the same xy window across a range of z planes.
    share_levels keeps one intensity scale across panels so that focus changes
    are visible as real changes rather than as per-panel autoscaling."""
    import matplotlib.pyplot as plt
    n = len(images)
    ncols = min(ncols, max(1, n))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize_per * ncols, figsize_per * nrows))
    axes = np.atleast_1d(axes).ravel()
    vmin = vmax = None
    if share_levels and n:
        if robust:
            los, his = zip(*[robust_levels(im) for im in images])
            vmin, vmax = min(los), max(his)
        else:
            vmin = int(min(im.min() for im in images))
            vmax = int(max(im.max() for im in images))
    for k, ax in enumerate(axes):
        if k >= n:
            ax.axis("off")
            continue
        show_image(images[k], res_um_px, ax=ax,
                   title=None if titles is None else titles[k],
                   vmin=vmin, vmax=vmax,
                   robust=robust and not share_levels)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    return fig, axes


def plot_profile(t_um, intensity, info=None, width_um=None, ax=None,
                 title=None, figsize=(6, 3.2)):
    """Darkness profile across a process, with the half-maximum level and the
    FWHM span annotated. One series: no legend, the title names it."""
    import matplotlib.pyplot as plt
    from allen_image_measure import to_darkness
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    d = to_darkness(intensity)
    ax.plot(t_um, d, color=INK, lw=2, solid_capstyle="round", zorder=3)
    ax.axhline(0, color=MUTED, lw=1, zorder=1)
    if info and "half_level" in info:
        ax.axhline(info["half_level"], color=MUTED, lw=1, ls=(0, (4, 3)), zorder=2)
        ax.text(t_um[0], info["half_level"], " half max", color=MUTED, fontsize=8,
                va="bottom", ha="left")
    if info and np.isfinite(info.get("t_left", np.nan)):
        tl, tr = info["t_left"], info["t_right"]
        ax.annotate("", xy=(tl, info["half_level"]), xytext=(tr, info["half_level"]),
                    arrowprops=dict(arrowstyle="<->", color=ACCENT, lw=2), zorder=4)
        if width_um is not None and np.isfinite(width_um):
            ax.text((tl + tr) / 2, info["half_level"], "  FWHM %.2f um" % width_um,
                    color=ACCENT, fontsize=10, va="bottom", ha="center")
    ax.set_xlabel("distance across the process (um)", fontsize=9, color=MUTED)
    ax.set_ylabel("darkness (8-bit levels)", fontsize=9, color=MUTED)
    ax.grid(color=GRID, lw=1)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    if title:
        ax.set_title(title, loc="left", fontsize=10, color=INK)
    return ax


def overlay_swc(ax, swc, frame, shift_px=(0.0, 0.0), flip_y_full_h=None,
                types=(3, 4), color=ACCENT, lw=0.8, alpha=0.9):
    """Draw an SWC skeleton on top of an already-displayed crop.

    swc          : DataFrame with id, type, x, y, z, r, parent (microns)
    frame        : the CropFrame the image was fetched with
    shift_px     : (dx, dy) alignment offset in FULL-RES pixels
    flip_y_full_h: if the SWC y axis is inverted relative to the image, pass the
                   full-res image height in pixels; otherwise None
    """
    res0 = frame.res0_um_px
    x_full = swc["x"].values / res0 + shift_px[0]
    y_um = swc["y"].values
    y_full = y_um / res0 + shift_px[1]
    if flip_y_full_h is not None:
        y_full = flip_y_full_h - (y_um / res0) + shift_px[1]
    col, row = frame.to_array_xy(x_full, y_full)
    idx = {int(i): k for k, i in enumerate(swc["id"].values)}
    segs = []
    for k, p in enumerate(swc["parent"].values):
        if int(p) not in idx or int(swc["type"].values[k]) not in types:
            continue
        kp = idx[int(p)]
        segs.append([(col[kp], row[kp]), (col[k], row[k])])
    if segs:
        from matplotlib.collections import LineCollection
        ax.add_collection(LineCollection(segs, colors=color, linewidths=lw,
                                         alpha=alpha, zorder=4))
    return ax


def read_swc(path):
    """Minimal SWC reader -> DataFrame. Here rather than in the IO module
    because it is a local file format, not an Allen API concern."""
    import pandas as pd
    rows = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            p = s.split()
            rows.append((int(p[0]), int(p[1]), float(p[2]), float(p[3]),
                         float(p[4]), float(p[5]), int(p[6])))
    return pd.DataFrame(rows, columns=["id", "type", "x", "y", "z", "r", "parent"])


def plot_registration_check(chk, res_um_px, figsize=(15, 4.4)):
    """Four panels for allen_image_align.registration_check output:
      a  darkness MIP of the z-band, path as traced (solid) and at s* (dashed)
      b  darkness under the path vs sideways shift, against random placements
      c  darkness under the path vs z shift
      d  the best-focus plane, path at s*
    """
    import matplotlib.pyplot as plt
    S = chk["samples"]
    c, r, tc, tr = S[:, 0], S[:, 1], S[:, 3], S[:, 4]
    nc, nr = -tr, tc
    s_px = (chk["lateral_offset_um"] if np.isfinite(chk["lateral_offset_um"]) else 0.0) / res_um_px

    fig, ax = plt.subplots(1, 4, figsize=figsize,
                           gridspec_kw=dict(width_ratios=[1.1, 1.3, 1.1, 1.1]))
    # a
    mip = chk["mip"]
    ax[0].imshow(mip, cmap="gray_r", vmin=0, vmax=np.percentile(mip, 99.7),
                 interpolation="nearest")
    ax[0].plot(c, r, color=ACCENT, lw=1.2, alpha=0.9)
    ax[0].plot(c + s_px * nc, r + s_px * nr, color=INK, lw=1.0, ls=(0, (3, 2)))
    ax[0].plot(c[len(c) // 2], r[len(r) // 2], "o", mfc="none", mec=ACCENT, ms=9, mew=1.5)
    ax[0].set_xticks([]); ax[0].set_yticks([])
    ax[0].set_title("a  darkness, z-band MIP\nsolid: as traced   dashed: at s*",
                    loc="left", fontsize=9, color=INK)

    # b
    s, L = chk["s_um"], chk["L"]
    if len(chk["L_null"]):
        lo, hi = np.nanpercentile(chk["L_null"], [5, 95], axis=0)
        ax[1].fill_between(s, lo, hi, color=GRID, lw=0, zorder=1)
        ax[1].text(s[0], hi[0], " random placements, 5-95%", color=MUTED, fontsize=8,
                   va="bottom", ha="left")
    ax[1].plot(s, L, color=INK, lw=2, zorder=3)
    ax[1].axvline(0, color=MUTED, lw=1, zorder=2)
    if np.isfinite(chk["lateral_offset_um"]):
        ax[1].axvline(chk["lateral_offset_um"], color=ACCENT, lw=1.5, ls=(0, (4, 3)), zorder=2)
        ax[1].text(0.98, 0.96, "s* = %+.2f um" % chk["lateral_offset_um"],
                   transform=ax[1].transAxes, color=ACCENT, fontsize=9,
                   ha="right", va="top")
    ax[1].set_xlabel("sideways shift of the path (um)", fontsize=9, color=MUTED)
    ax[1].set_ylabel("mean darkness under the path", fontsize=9, color=MUTED)
    ax[1].set_title("b  p = %.3f  (n=%d),  coverage %.0f%%"
                    % (chk["p_value"], chk["n_null"], 100 * chk["coverage"]),
                    loc="left", fontsize=9, color=INK)

    # c
    ax[2].plot(chk["dz_axis_um"], chk["Z"], color=INK, lw=2)
    ax[2].axvline(0, color=MUTED, lw=1)
    if np.isfinite(chk["z_offset_um"]):
        ax[2].axvline(chk["z_offset_um"], color=ACCENT, lw=1.5, ls=(0, (4, 3)))
    ax[2].set_xlabel("z shift relative to SWC z (um)", fontsize=9, color=MUTED)
    ax[2].set_title("c  focus peak at %+.2f um" % chk["z_offset_um"],
                    loc="left", fontsize=9, color=INK)

    # d
    bp = chk["best_plane"]
    lo_, hi_ = robust_levels(bp)
    ax[3].imshow(bp, cmap="gray", vmin=lo_, vmax=hi_, interpolation="nearest")
    ax[3].plot(c + s_px * nc, r + s_px * nr, color=ACCENT, lw=0.9, alpha=0.8)
    ax[3].set_xticks([]); ax[3].set_yticks([])
    ax[3].set_title("d  best-focus plane %d" % chk["best_k"], loc="left", fontsize=9, color=INK)

    for a_ in ax[1:3]:
        a_.grid(color=GRID, lw=1)
        a_.set_axisbelow(True)
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            a_.spines[sp].set_color(MUTED)
        a_.tick_params(colors=MUTED, labelsize=8)
    # scale bar on the image panels
    for a_ in (ax[0], ax[3]):
        h, w = mip.shape
        bar = 5.0 / res_um_px
        a_.plot([0.05 * w, 0.05 * w + bar], [0.94 * h, 0.94 * h], color="white", lw=4)
        a_.plot([0.05 * w, 0.05 * w + bar], [0.94 * h, 0.94 * h], color=INK, lw=2)
        a_.text(0.05 * w + bar / 2, 0.91 * h, "5 um", ha="center", va="bottom", fontsize=8,
                color=INK, bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1))
    fig.suptitle(chk["verdict"] + "   |   " + chk["z_verdict"], x=0.01, ha="left",
                 fontsize=10, color=INK)
    fig.tight_layout()
    return fig, ax
