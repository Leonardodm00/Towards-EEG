"""The recovery check's figure (D-216.2): per case, the data, the fit and the residual on a few planes.

Rows: the data (grey levels), the fit B_k tau_k(theta_hat) on the same grey range per column (each plane its own
range, so the far planes' faint shadow shows), the residual data - fit on the fit's region (a diverging map, blue -
grey - red, symmetric about 0, one range for all planes). Outlines: the truth's section at the plane's depth (solid)
and its footprint (dashed) on the data row, blue; the fitted ellipsoid's on the fit row, orange; the region's boundary
(dashed) and the stems' mask (solid) on the residual row, dark grey. Colours: the dataviz skill's reference palette
(slots 1 and 2, validated; its diverging pair with the grey midpoint). Pure ASCII.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402
import numpy as np                                                  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap              # noqa: E402
from matplotlib.lines import Line2D                                 # noqa: E402

TRUTH = "#2a78d6"          # palette slot 1 (blue)
FIT = "#eb6834"            # palette slot 2 (orange)
INK = "#3a3a38"            # outlines and text on the residual row
SURFACE = "#fcfcfb"
DIVERGING = LinearSegmentedColormap.from_list("blue_grey_red", ["#2a78d6", "#f0efec", "#e34948"])


def _contour(ax, x, y, field, level, color, style, lw):
    if np.nanmax(field) > level:
        ax.contour(x, y, field, levels=[level], colors=[color], linestyles=[style], linewidths=[lw])


def figure_case(path, title, data, model, region, mask, H_true, H_fit, dz_true, dz_fit, labels, x_um, y_um,
                region_centre, region_radius, resid_lim, subtitle=""):
    """Write the figure to path. data, model (n, H, W) gl; region, mask (H, W) bool; H_true, H_fit (H, W) um column
    half-depths (truth's solid, fitted ellipsoid); dz_true, dz_fit (n,) um the plane's distance from each centre's
    depth; labels (n,) the columns' titles; x_um (W,), y_um (H,) pixel centres relative to the true centre;
    region_centre (2,) um, region_radius um; resid_lim gl (the residual's colour range +-)."""
    n = data.shape[0]
    p = float(x_um[1] - x_um[0])
    extent = (x_um[0] - 0.5 * p, x_um[-1] + 0.5 * p, y_um[-1] + 0.5 * p, y_um[0] - 0.5 * p)
    fig, axes = plt.subplots(3, n, figsize=(3.1 * n + 1.2, 10.2), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    resid = np.where(region[None], data - model, np.nan)
    im_r = None
    t = np.linspace(0, 2 * np.pi, 361)
    for j in range(n):
        vals = np.concatenate([data[j][region], model[j][region]])
        lo, hi = np.percentile(vals, [0.5, 99.5])
        for i, img in enumerate((data[j], model[j])):
            ax = axes[i, j]
            ax.imshow(img, cmap="gray", vmin=lo, vmax=hi, extent=extent, interpolation="nearest")
            H, dz, col = (H_true, dz_true[j], TRUTH) if i == 0 else (H_fit, dz_fit[j], FIT)
            _contour(ax, x_um, y_um, H, 1e-6, col, "dashed", 0.9)
            _contour(ax, x_um, y_um, H, dz, col, "solid", 1.4)
        axes[0, j].set_title("%s\ngrey %.0f-%.0f gl" % (labels[j], lo, hi), fontsize=9, color=INK)
        ax = axes[2, j]
        ax.set_facecolor(SURFACE)
        im_r = ax.imshow(resid[j], cmap=DIVERGING, vmin=-resid_lim, vmax=resid_lim, extent=extent,
                         interpolation="nearest")
        ax.plot(region_centre[0] + region_radius * np.cos(t), region_centre[1] + region_radius * np.sin(t),
                color=INK, ls="--", lw=0.8)
        if mask.any():
            ax.contour(x_um, y_um, mask.astype(float), levels=[0.5], colors=[INK], linewidths=[0.9])
        rr = resid[j][region]
        ax.set_title("RMS %.2f gl, mean %+.2f gl" % (np.sqrt(np.mean(rr * rr)), np.mean(rr)), fontsize=9,
                     color=INK)
        for i in range(3):
            axes[i, j].set_xlim(extent[0], extent[1])
            axes[i, j].set_ylim(extent[2], extent[3])
            axes[i, j].tick_params(labelsize=7, colors=INK)
            if j > 0:
                axes[i, j].set_yticklabels([])
            if i < 2:
                axes[i, j].set_xticklabels([])
    for i, name in enumerate(("data (gl)", "fit B_k tau_k (gl)", "residual data - fit (gl)")):
        axes[i, 0].set_ylabel(name + "\ny - y0 (um)", fontsize=9, color=INK)
    for j in range(n):
        axes[2, j].set_xlabel("x - x0 (um)", fontsize=9, color=INK)
    handles = [Line2D([], [], color=TRUTH, lw=1.4, label="truth: section at the plane"),
               Line2D([], [], color=TRUTH, lw=0.9, ls="--", label="truth: footprint"),
               Line2D([], [], color=FIT, lw=1.4, label="fit: section at the plane"),
               Line2D([], [], color=FIT, lw=0.9, ls="--", label="fit: footprint"),
               Line2D([], [], color=INK, lw=0.8, ls="--", label="fit region (13 um)"),
               Line2D([], [], color=INK, lw=0.9, label="stems' mask")]
    if not mask.any():
        handles = handles[:-1]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8.5, frameon=False,
               labelcolor=INK)
    fig.subplots_adjust(left=0.06, right=0.92, top=0.86, bottom=0.08, wspace=0.06, hspace=0.16)
    cax = fig.add_axes([0.935, 0.1, 0.012, 0.22])
    cb = fig.colorbar(im_r, cax=cax)
    cb.set_label("residual (gl)", fontsize=8, color=INK)
    cb.ax.tick_params(labelsize=7, colors=INK)
    fig.suptitle(title, fontsize=11, color=INK, y=0.985)
    if subtitle:
        fig.text(0.5, 0.925, subtitle, ha="center", va="center", fontsize=8.5, color=INK)
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)
