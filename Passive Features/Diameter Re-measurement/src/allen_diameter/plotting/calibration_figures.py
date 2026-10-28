"""The calibration's figures -- Block 10.F in specs/SPEC.md (D-075, D-080.2,
D-081.5; 2026-10-23). Plotting only: every array comes prepared from
analysis.calibration_report.figure_inputs; nothing scientific is computed here.

  width_by_plane_figure  one stage: (a) every node's omega_hat against the
                         signed plane offset in light grey with the median
                         over nodes in blue; (b) the same centred by the
                         node's c_j with the fitted law; (c) the residuals
  example_node_figure    the example node's widths with its own curve, and
                         the image of each of its planes (the patch, the line,
                         the last window and the far-pixel mask)
  gaussians_figure       one node: every plane's dip with the last pass's
                         Gaussian core over its window (D-075.2)
  kernel_law_figure      both stages as kernel SDs against |delta|, the cut
                         table, the far line and the anchor (D-075.1)
  kernel_law_by_cell_figure  Block 19.5 (D-090.2): the same points by cell --
                         one panel coloured by cell up to 3 cells, one panel
                         per cell beyond
  kernel_law_residuals_by_cell_figure  the residuals from the pooled law, one
                         row per cell, with each cell's median per set
  isolation_figure       one candidate of Cell 7 (10.F (5), D-098.4): the
                         patch its background is fitted on, at plane k*, with
                         the traced segments -- its own branch, and every
                         neighbour coloured by whether its predicted shadow
                         exceeds the isolation threshold -- and the far pixels;
                         under the stripe (10.S) also (c), every plane's
                         stripe: the far fit's planes lie 5-16 planes from k*
  isolation_overview     the candidates' patches as thumbnails, pages of 36
  write_all              the PNG files of a run

Colours: the palette of the project's figures (light grey for the nodes,
blue for the median and the law, orange for the example and the far line,
green for the near table). By cell (19.5): the first three categorical slots
of the reference palette (blue, orange, aqua), validated all-pairs for a
scatter (dataviz skill's validate_palette.js, 2026-10-27: CVD dE >= 9.2,
normal-vision dE >= 24.0; aqua below 3:1 on the surface, so the cells are
also labelled and listed in kernel_law_by_cell.csv); beyond three cells one
panel per cell. Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import math
import os

import numpy as np

GREY, INK, INK2 = "#a9a8a2", "#0b0b0b", "#52514e"
BLUE, ORANGE, GREEN, SURFACE, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#fcfcfb", "#e4e3df"
CELL_HUES = (BLUE, ORANGE, GREEN)      # 19.5: the most a scatter keeps apart by colour (see the module docstring)
BACKDROP = "#d9d8d3"                   # 19.5: every cell's used points behind one cell's panel


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
    ax.tick_params(colors=INK2, labelsize=8)


def width_by_plane_figure(stage, title=""):
    """stage: calibration_report.stage_inputs. Returns the Figure."""
    plt = _plt()
    dz = stage["dz_um"]
    has_law = stage["law"] is not None
    fig, axes = plt.subplots(3 if has_law else 1, 1, figsize=(8.4, 8.4 if has_law else 3.6), sharex=True,
                             facecolor=SURFACE, gridspec_kw=dict(height_ratios=[1, 1, 0.6]) if has_law else None)
    axes = np.atleast_1d(axes)
    for ax in axes:
        _style(ax)
    ax = axes[0]
    for nd in stage["nodes"]:
        ok = np.isfinite(nd["omega"])
        ax.plot(nd["offsets"][ok], nd["omega"][ok], "-", color=GREY, lw=0.8, alpha=0.7)
        ax.plot(nd["offsets"][ok & nd["used"]], nd["omega"][ok & nd["used"]], "o", color=GREY, ms=3.5, mec="none")
        ax.plot(nd["offsets"][ok & ~nd["used"]], nd["omega"][ok & ~nd["used"]], "o", mfc="none", mec=GREY, ms=4)
    offs = sorted(stage["median"])
    ax.plot(offs, [stage["median"][o] for o in offs], "o-", color=BLUE, ms=5, lw=1.8, label="median over nodes")
    ax.set_ylabel(r"$\hat\omega_{j,k}$ ($\mu$m$^2$)", fontsize=9)
    ax.set_title("(a) squared core width by plane: %d nodes in grey, median in blue%s"
                 % (len(stage["nodes"]), "; hollow: not used by the fit" if any((~nd["used"]).any() for nd in stage["nodes"]) else ""),
                 fontsize=9.5, loc="left", color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper center")
    if has_law:
        ax = axes[1]
        for nd in stage["nodes"]:
            ok = np.isfinite(nd["centred"])
            # the plane's offset from the fitted axis, in planes: (z_k - z_ax_hat) / dz = -delta_hat / dz
            ax.plot(-nd["delta_hat"][ok] / dz, nd["centred"][ok], "o", color=GREY, ms=3.5, mec="none", alpha=0.8)
        offs = sorted(stage["median_centred"])
        ax.plot(offs, [stage["median_centred"][o] for o in offs], "o", color=BLUE, ms=5, label="median over nodes")
        grid, law = stage["law"]
        ax.plot(-grid / dz, law, "-", color=BLUE, lw=1.8, label="fitted law")
        ax.set_ylabel(r"$\hat\omega_{j,k} - \hat c_j$ ($\mu$m$^2$)", fontsize=9)
        ax.set_title("(b) centred by the node's constant, against the fitted axis offset; the fitted law in blue",
                     fontsize=9.5, loc="left", color=INK)
        ax.legend(frameon=False, fontsize=8, loc="upper center")
        ax = axes[2]
        acc = {}
        for nd in stage["nodes"]:
            ok = np.isfinite(nd["residual"])
            ax.plot(nd["offsets"][ok], nd["residual"][ok], "o", color=GREY, ms=3.5, mec="none", alpha=0.8)
            for o, r in zip(nd["offsets"][ok], nd["residual"][ok]):
                acc.setdefault(int(o), []).append(float(r))
        if acc:
            offs = sorted(acc)
            ax.plot(offs, [np.median(acc[o]) for o in offs], "o-", color=BLUE, ms=4, lw=1.4)
        ax.axhline(0.0, color=INK2, lw=0.8)
        ax.set_ylabel(r"residual ($\mu$m$^2$)", fontsize=9)
        ax.set_title("(c) residual from the law, median per offset in blue", fontsize=9.5, loc="left", color=INK)
    axes[-1].set_xlabel(r"plane offset $k - k^*_j$ (planes of %.2f $\mu$m; panel (b): $(z_k - \hat z_{{\rm ax},j})/\Delta z$, "
                        r"the offset from the fitted axis)" % dz, fontsize=9)
    fig.suptitle(title or "%s-field calibration: width by plane" % stage["which"], fontsize=10, color=INK)
    fig.tight_layout()
    return fig


def _draw_line(ax, cx, cy, theta, half_um, color, mark_plus=False):
    """The measuring line (axis (-sin theta, cos theta)); mark_plus labels its +v end (the isolation figure's panel (c)
    plots the stripe's runs against v)."""
    y_hat = np.array([-math.sin(theta), math.cos(theta)])
    ax.plot([cx - half_um * y_hat[0], cx + half_um * y_hat[0]], [cy - half_um * y_hat[1], cy + half_um * y_hat[1]],
            "-", color=color, lw=0.9)
    if mark_plus:
        ax.annotate("+v", (cx + half_um * y_hat[0], cy + half_um * y_hat[1]), color=color, fontsize=8,
                    xytext=(3, 3), textcoords="offset points")


def example_node_figure(ex, line_half_um, title=""):
    """ex: calibration_report.example_inputs. The widths with the node's own curve, and the plane images when
    present (the patch in grey levels, the far pixels dimmed, the line in orange, the last window as a bar)."""
    plt = _plt()
    nd, images, dz = ex["node"], ex["images"], ex["dz_um"]
    n_img = 0 if images is None else int(images["stack"].shape[0])
    ncol = 6 if n_img > 7 else max(n_img, 1)
    nrow = int(math.ceil(n_img / ncol)) if n_img else 0
    fig = plt.figure(figsize=(10.5, 3.6 + 1.85 * nrow), facecolor=SURFACE)
    gs = fig.add_gridspec(1 + nrow, ncol, height_ratios=[1.6] + [1.0] * nrow, hspace=0.6, wspace=0.08,
                          top=0.93, bottom=0.05)
    ax = fig.add_subplot(gs[0, :])
    _style(ax)
    ok = np.isfinite(nd["omega"])
    ax.plot(nd["offsets"][ok & nd["used"]], nd["omega"][ok & nd["used"]], "o", color=ORANGE, ms=5, label="this node")
    ax.plot(nd["offsets"][ok & ~nd["used"]], nd["omega"][ok & ~nd["used"]], "o", mfc="none", mec=ORANGE, ms=5,
            label="not used by the fit")
    if ex["curve"] is not None:
        grid, cur = ex["curve"]
        ax.plot(-grid / dz, cur, "-", color=BLUE, lw=1.6, label="its curve: $\\hat c_j$ + the law")
    ax.set_xlabel(r"plane offset $k - k^*_j$", fontsize=9)
    ax.set_ylabel(r"$\hat\omega_{j,k}$ ($\mu$m$^2$)", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(title or "%s-field example node %d (nearest the median width by plane)" % (ex["which"], nd["node_id"]),
                 fontsize=10, loc="left", color=INK)
    if n_img:
        stack, far = images["stack"], images["far"]
        p, left, top = float(images["res_um_px"]), int(images["left"]), int(images["top"])
        cx, cy, theta = float(images["cx_um"]), float(images["cy_um"]), float(images["theta_rad"])
        ks, scan_ks = np.asarray(images["ks"]), np.asarray(images["scan_ks"])
        vc, wh = np.asarray(images["v_c_um"], dtype=float), np.asarray(images["window_half_um"], dtype=float)
        status = [str(s) for s in np.asarray(images["status"])]
        k_star = int(images["k_star"])
        H, W = stack.shape[1:]
        extent = [(left - 0.5) * p, (left + W - 0.5) * p, (top + H - 0.5) * p, (top - 0.5) * p]
        lo, hi = np.percentile(stack, [0.5, 99.8])
        for i in range(n_img):
            axi = fig.add_subplot(gs[1 + i // ncol, i % ncol])
            img = stack[i].astype(float)
            axi.imshow(img, cmap="gray", vmin=lo, vmax=hi, extent=extent, interpolation="nearest")
            shade = np.zeros(img.shape + (4,))
            shade[..., 2] = 1.0
            shade[..., 3] = np.where(far[i], 0.0, 0.18)
            axi.imshow(shade, extent=extent, interpolation="nearest")
            _draw_line(axi, cx, cy, theta, line_half_um, ORANGE)
            q = int(np.flatnonzero(scan_ks == ks[i])[0]) if np.any(scan_ks == ks[i]) else -1
            if q >= 0 and np.isfinite(wh[q]) and np.isfinite(vc[q]):
                y_hat = np.array([-math.sin(theta), math.cos(theta)])
                a0, a1 = vc[q] - wh[q], vc[q] + wh[q]
                axi.plot([cx + a0 * y_hat[0], cx + a1 * y_hat[0]], [cy + a0 * y_hat[1], cy + a1 * y_hat[1]], "-",
                         color=BLUE, lw=2.2, alpha=0.9)
            axi.set_xticks([]); axi.set_yticks([])
            axi.set_title("k - k* = %+d%s" % (int(ks[i]) - k_star, "" if q < 0 or status[q] == "ok" else " (%s)" % status[q]),
                          fontsize=7.5, color=INK)
        fig.text(0.01, 0.005, "images: the patch (grey levels), masked pixels shaded blue, the line in orange, the last "
                 "core-fit window as the blue bar", fontsize=7.5, color=INK2)
    return fig


def gaussians_figure(g, title=""):
    """g: calibration_report.gaussian_inputs of one node: one panel per plane in plane order (D-075.2)."""
    plt = _plt()
    n = len(g["planes"])
    ncol = 7 if n > 9 else min(n, 7)
    nrow = int(math.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(1.9 * ncol + 0.6, 1.7 * nrow + 0.8), facecolor=SURFACE, squeeze=False,
                             sharex=True)
    v = g["v"]
    for i, pl in enumerate(g["planes"]):
        ax = axes[i // ncol, i % ncol]
        _style(ax)
        ax.plot(v, pl["w"], ".", color=GREY, ms=2.5)
        if math.isfinite(pl["sigma"]) and math.isfinite(pl["amp"]) and math.isfinite(pl["window_half"]):
            win = np.abs(v - pl["v_c"]) <= pl["window_half"]
            ax.plot(v[win], pl["amp"] * np.exp(-0.5 * ((v[win] - pl["v_c"]) / pl["sigma"]) ** 2), "-", color=BLUE, lw=1.4)
            ax.axvspan(pl["v_c"] - pl["window_half"], pl["v_c"] + pl["window_half"], color=BLUE, alpha=0.06, lw=0)
        ax.set_title("%+d: %s" % (pl["offset"], ("$\\hat\\omega$ = %.3f, %d pass%s" % (pl["omega"], pl["passes"],
                                                                                     "" if pl["passes"] == 1 else "es"))
                                  if pl["status"] == "ok" else pl["status"]), fontsize=7, color=INK)
        ax.tick_params(labelsize=6)
    for j in range(n, nrow * ncol):
        axes[j // ncol, j % ncol].axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("v ($\\mu$m)", fontsize=7)
    fig.suptitle(title or "%s-field node %d: the dip and the last pass's Gaussian core, plane by plane"
                 % (g["set"], g["node_id"]), fontsize=9.5, color=INK)
    fig.tight_layout()
    return fig


def kernel_law_figure(kl, title=""):
    """kl: calibration_report.kernel_law_inputs (D-075.1)."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8.0, 4.6), facecolor=SURFACE)
    _style(ax)
    pts = kl["points"]
    for which, color in (("near", GREEN), ("far", ORANGE)):
        for side, marker in ((1, "o"), (-1, "s")):
            for used, mfc in ((True, color), (False, "none")):
                sel = [(d, s) for d, s, sd, u, w in pts if w == which and sd == side and u == used]
                if sel:
                    d, s = zip(*sel)
                    ax.plot(d, s, marker, color=color, mfc=mfc, ms=4, alpha=0.75, lw=0,
                            label="%s planes, $\\delta$ %s 0%s" % (which, ">" if side > 0 else "<", "" if used else ", not in the fit"))
    if kl["table"] is not None:
        ax.plot(kl["table"][0], kl["table"][1], "-", color=GREEN, lw=2.0, label="near table (cut at the anchor)")
    if kl["line"] is not None:
        ax.plot(kl["line"][0], kl["line"][1], "-", color=ORANGE, lw=2.0, label="far line")
    if kl["anchor"] is not None:
        ax.plot([kl["anchor"][0]], [kl["anchor"][1]], "D", color=INK, ms=6, label="anchor $\\hat\\sigma_0$")
        ax.axvline(kl["anchor"][0], color=INK2, lw=0.8, ls=(0, (4, 3)))
    ax.set_xlabel(r"$|\hat\delta_{j,k}|$ ($\mu$m), from the fitted axes", fontsize=9)
    ax.set_ylabel(r"kernel SD ($\mu$m): near $(\hat\omega - \hat c_j + \sigma_r(0)^2)^{1/2}$, far $(\hat\omega - \hat c_j)^{1/2}$",
                  fontsize=8.5)
    ax.legend(frameon=False, fontsize=7.5, ncol=2)
    ax.set_title(title or "the kernel law from both stages", fontsize=10, loc="left", color=INK)
    fig.tight_layout()
    return fig


def _cell_label(cell):
    return "cell" if cell is None else "specimen %d" % int(cell)


def _law(ax, kc, labels=True):
    if kc["table"] is not None:
        ax.plot(kc["table"][0], kc["table"][1], "-", color=INK, lw=1.6, label="pooled near table" if labels else None,
                zorder=4)
    if kc["line"] is not None:
        ax.plot(kc["line"][0], kc["line"][1], "--", color=INK, lw=1.4, label="pooled far line" if labels else None,
                zorder=4)
    if kc["anchor"] is not None:
        ax.plot([kc["anchor"][0]], [kc["anchor"][1]], "D", color=INK, ms=5, label="anchor" if labels else None,
                zorder=5)


def _n_nodes(kc, cell, which):
    row = next((r for r in kc["summary"] if r["specimen_id"] == cell and r["set"] == which), None)
    return 0 if row is None else int(row["n_nodes"])


def kernel_law_by_cell_figure(kc, title=""):
    """kc: calibration_report.kernel_law_by_cell. Up to len(CELL_HUES) cells: one panel, colour by cell, marker by
    set (near a circle, far a triangle), hollow when the fit did not use the point. More cells: one panel per cell,
    its points in blue over every cell's used points in light grey. The pooled table and far line in ink."""
    from matplotlib.lines import Line2D
    plt = _plt()
    cells, pts = kc["cells"], kc["points"]
    xlabel = r"$|\hat\delta_{j,k}|$ ($\mu$m), from the fitted axes"
    ylabel = r"kernel SD $\hat\sigma_{j,k}$ ($\mu$m)"
    if len(cells) <= len(CELL_HUES):
        fig, ax = plt.subplots(figsize=(8.0, 4.8), facecolor=SURFACE)
        _style(ax)
        for i, c in enumerate(cells):
            for which, marker in (("near", "o"), ("far", "^")):
                for used in (True, False):
                    sel = [p for p in pts if p["cell"] == c and p["set"] == which and p["used"] == used]
                    if sel:
                        ax.plot([p["abs_delta_um"] for p in sel], [p["sigma_um"] for p in sel], marker,
                                color=CELL_HUES[i], mfc=CELL_HUES[i] if used else "none", ms=4.5, alpha=0.8, lw=0,
                                zorder=3)
        _law(ax, kc)
        handles = [Line2D([], [], marker="o", ls="", color=CELL_HUES[i], ms=6,
                          label="%s (%d near, %d far nodes)" % (_cell_label(c), _n_nodes(kc, c, "near"),
                                                                _n_nodes(kc, c, "far")))
                   for i, c in enumerate(cells)]
        handles += [Line2D([], [], marker="o", ls="", color=INK2, label="near-field planes"),
                    Line2D([], [], marker="^", ls="", color=INK2, label="far-field planes"),
                    Line2D([], [], marker="o", ls="", color=INK2, mfc="none", label="not in the fit")]
        handles += ax.get_legend_handles_labels()[0]
        ax.legend(handles=handles, frameon=False, fontsize=7.5, ncol=2)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title or "the kernel law by cell: near and far fields together, the pooled law in ink",
                     fontsize=10, loc="left", color=INK)
        fig.tight_layout()
        return fig
    ncol = min(4, len(cells))
    nrow = int(math.ceil(len(cells) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 2.8 * nrow + 0.7), sharex=True, sharey=True,
                             squeeze=False, facecolor=SURFACE)
    back = [p for p in pts if p["used"]]
    for k, c in enumerate(cells):
        ax = axes[k // ncol, k % ncol]
        _style(ax)
        ax.plot([p["abs_delta_um"] for p in back], [p["sigma_um"] for p in back], "o", color=BACKDROP, ms=2.5,
                mec="none", zorder=1)
        for which, marker in (("near", "o"), ("far", "^")):
            for used in (True, False):
                sel = [p for p in pts if p["cell"] == c and p["set"] == which and p["used"] == used]
                if sel:
                    ax.plot([p["abs_delta_um"] for p in sel], [p["sigma_um"] for p in sel], marker, color=BLUE,
                            mfc=BLUE if used else "none", ms=4, alpha=0.85, lw=0, zorder=3)
        _law(ax, kc, labels=False)
        ax.set_title("%s: %d near, %d far nodes" % (_cell_label(c), _n_nodes(kc, c, "near"), _n_nodes(kc, c, "far")),
                     fontsize=8, color=INK)
    for k in range(len(cells), nrow * ncol):
        axes[k // ncol, k % ncol].axis("off")
    fig.supxlabel(xlabel, fontsize=9)
    fig.supylabel(ylabel, fontsize=9)
    fig.suptitle(title or "the kernel law by cell: each cell's planes in blue (circles near, triangles far, hollow "
                 "not in the fit) over every cell's in grey; the pooled table (solid) and far line (dashed) in ink",
                 fontsize=9, color=INK)
    fig.tight_layout()
    return fig


def kernel_law_residuals_by_cell_figure(kc, title=""):
    """kc: calibration_report.kernel_law_by_cell. One row per cell (increasing specimen id, top to bottom): the
    residuals from the pooled law, near circles above the row's line and far triangles below, hollow when not in
    the fit; the cell's median per set (over the used points) as an ink bar; zero as a line. Colour by cell as in
    kernel_law_by_cell_figure up to len(CELL_HUES) cells, ink-grey beyond."""
    from matplotlib.lines import Line2D
    plt = _plt()
    cells, pts = kc["cells"], kc["points"]
    n = len(cells)
    fig, ax = plt.subplots(figsize=(8.0, 1.5 + 0.42 * n), facecolor=SURFACE)
    _style(ax)
    for i, c in enumerate(cells):
        y = n - 1 - i
        col = CELL_HUES[i] if n <= len(CELL_HUES) else INK2
        for which, marker, dy in (("near", "o", 0.15), ("far", "^", -0.15)):
            for used in (True, False):
                sel = [p for p in pts if p["cell"] == c and p["set"] == which and p["used"] == used
                       and math.isfinite(p["residual_um"])]
                if sel:
                    ax.plot([p["residual_um"] for p in sel], [y + dy] * len(sel), marker, color=col,
                            mfc=col if used else "none", ms=4, alpha=0.7, lw=0, zorder=3)
            row = next((r for r in kc["summary"] if r["specimen_id"] == c and r["set"] == which), None)
            if row is not None and math.isfinite(row["median_residual_um"]):
                ax.plot([row["median_residual_um"]], [y + dy], "|", color=INK, ms=12, mew=2.2, zorder=4)
    ax.axvline(0.0, color=INK2, lw=0.9, zorder=2)
    ax.set_yticks(range(n))
    ax.set_yticklabels([_cell_label(c) for c in reversed(cells)], fontsize=8)
    ax.set_ylim(-0.6, n - 0.4)
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=INK2, label="near-field planes"),
                        Line2D([], [], marker="^", ls="", color=INK2, label="far-field planes"),
                        Line2D([], [], marker="o", ls="", color=INK2, mfc="none", label="not in the fit"),
                        Line2D([], [], marker="|", ls="", color=INK, ms=10, mew=2.2, label="the cell's median")],
               frameon=False, fontsize=7.5, ncol=4, loc="lower center")
    ax.set_xlabel(r"residual from the pooled law, $\hat\sigma_{j,k} - \sigma_{\rm law}(|\hat\delta_{j,k}|)$ ($\mu$m)",
                  fontsize=9)
    ax.set_title(title or "the kernel law's residuals by cell", fontsize=10, loc="left", color=INK)
    fig.tight_layout(rect=(0, min(0.2, 0.24 / (1.5 + 0.42 * n)), 1, 1))
    return fig


def _extent(view, shape):
    p, left, top = float(view["res_um_px"]), int(view["left"]), int(view["top"])
    H, W = int(shape[0]), int(shape[1])
    return [(left - 0.5) * p, (left + W - 0.5) * p, (top + H - 0.5) * p, (top - 0.5) * p]


def _seg_style(own, depth, thr):
    """(colour, line width, line style) of a traced segment in the isolation figures."""
    if own:
        return BLUE, 1.4, "-"
    if not np.isfinite(depth):
        return GREY, 0.9, "--"
    return (ORANGE, 1.8, "-") if depth > thr else (GREY, 1.0, "-")


def _draw_scene(ax, view, thin=False):
    """The traced segments and soma discs over a patch image (isolation figures)."""
    thr = float(view["threshold_gl"])
    scale = 0.6 if thin else 1.0
    for row, depth in zip(np.atleast_2d(view["seg_rows"]).reshape(-1, 9), view["seg_depth_gl"]):
        colour, lw, ls = _seg_style(bool(row[8] >= 0.5), float(depth), thr)
        ax.plot([row[0], row[3]], [row[1], row[4]], ls, color=colour, lw=lw * scale, solid_capstyle="round")
    import matplotlib.patches as mpatches
    for (x, y, r), depth in zip(np.atleast_2d(view["soma"]).reshape(-1, 3), view["soma_depth_gl"]):
        colour = ORANGE if np.isfinite(depth) and depth > thr else GREY
        ax.add_patch(mpatches.Circle((x, y), r, fill=False, ec=colour, lw=1.4 * scale))
    _draw_line(ax, view["cx_um"], view["cy_um"], view["theta_rad"], view["line_half_um"], GREEN)
    ax.plot([view["cx_um"]], [view["cy_um"]], "+", color=GREEN, ms=6 * scale, mew=1.2)


def _gl(x):
    """A grey-level depth for a label: 3 significant figures without an exponent."""
    x = float(x)
    return "%.0f" % x if abs(x) >= 100 else ("%.1f" % x if abs(x) >= 1 else "%.2g" % x)


def fit_planes_usable(view):
    """(usable, total): the planes of the set's fit whose stripe is usable (10.S), from the view's stripe_planes;
    None without them (the square rule, or a view written before 2026-10-28)."""
    sp = view.get("stripe_planes")
    if sp is None:
        return None
    fit = np.asarray(sp["fit"], dtype=bool)
    return int(np.sum(fit & np.asarray(sp["ok"], dtype=bool))), int(fit.sum())


def _isolation_line(view):
    thr = float(view["threshold_gl"])
    if not view["evaluated"]:
        return "isolation not evaluated (the pilot has no sharpest plane, or no finite B_bar or mu)"
    d = float(view["max_depth_gl"])
    w = view["worst"]
    who = "" if w is None else (" (soma, %.1f um from the patch)" % w["rho_um"] if w["kind"] == "soma" else
                                " (segment %d-%d, %.1f um from the patch)" % (w["child_id"], w["parent_id"], w["rho_um"]))
    st = view.get("stripe")
    if st is not None:
        used = fit_planes_usable(view)
        return "stripe at k*: %s (%d px, runs -%.1f / +%.1f um)%s; deepest predicted neighbour shadow %s gl%s" % (
            "usable" if st["ok"] else st["reason"], st["n_pixels"], st["side_minus_um"], st["side_plus_um"],
            "" if used is None else ", usable in %d of the fit's %d planes" % used, _gl(d), who)
    return "isolation: deepest predicted neighbour shadow %s gl%s %s %s gl -> %s" % (
        _gl(d), who, ">" if d > thr else "<=", _gl(thr), "not_isolated" if d > thr else "isolated")


def _stripe_planes_panel(ax_n, ax_r, view):
    """(c) of the isolation figure under the stripe (10.S; 2026-10-28, the user: "given the far field nodes highlight
    the stripe used"): every plane of the stage. Above, the stripe's clear pixels (log scale; x: none) against
    stripe_min_pixels (dashed). Below, its runs along the measuring line beyond the own cylinder's columns (light
    blue), one bar per side, as wide as the plane's W relative to the grid's largest: dark grey where the plane's
    stripe is usable, orange where it is not; a black tick where a side's run reaches stripe_min_side_um, the
    measuring line's ends green and dashed. The planes the set's fit does not use are faint (the far set: the fit
    uses |k - k*| >= far_min_k_offset; the dip-area planes, far_area_offsets, half-faint)."""
    from matplotlib.ticker import MaxNLocator
    sp = view["stripe_planes"]
    off = np.asarray(sp["offsets"], dtype=int)
    ok, fit, area = (np.asarray(sp[key], dtype=bool) for key in ("ok", "fit", "area"))
    alpha = np.where(fit, 1.0, np.where(area, 0.55, 0.22))
    wmax = max(sp["half_widths_um"]) if len(sp["half_widths_um"]) else 1.0
    min_side = float(sp["min_side_um"])
    for ax in (ax_n, ax_r):
        _style(ax)
    for j in range(off.size):
        k, a = int(off[j]), float(alpha[j])
        colour = INK2 if ok[j] else ORANGE
        n = float(sp["n_pixels"][j])
        ax_n.plot([k], [max(n, 1.0)], "o" if n > 0 else "x", color=colour, alpha=a, ms=4.5, mew=1.2)
        width = 0.2 + 0.6 * float(sp["w_um"][j]) / wmax
        vm, vp = float(sp["v_own_minus_um"][j]), float(sp["v_own_plus_um"][j])
        sm, spl = float(sp["side_minus_um"][j]), float(sp["side_plus_um"][j])
        ax_r.bar(k, vm + vp, width=width, bottom=-vm, color=BLUE, alpha=0.3 * a, lw=0)
        ax_r.bar(k, sm, width=width, bottom=-(vm + sm), color=colour, alpha=a, lw=0)
        ax_r.bar(k, spl, width=width, bottom=vp, color=colour, alpha=a, lw=0)
        for y in (-(vm + min_side), vp + min_side):
            ax_r.plot([k - 0.5 * width, k + 0.5 * width], [y, y], color=INK, lw=0.9, alpha=a)
    ax_n.axhline(float(sp["min_pixels"]), color=INK2, ls="--", lw=0.9)
    ax_n.set_yscale("log")
    h = float(view["line_half_um"])
    for y in (-h, h):
        ax_r.axhline(y, color=GREEN, ls="--", lw=1.0)
    ax_r.axhline(0.0, color=GREY, lw=0.6)
    ax_r.set_xlim(off.min() - 0.8, off.max() + 0.8)
    ax_r.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_r.set_xlabel("plane offset k - k*", fontsize=8)
    ax_r.set_ylabel("v along the measuring line (um; +v marked in (b))", fontsize=8)
    ax_n.tick_params(labelbottom=False)
    ax_n.set_ylabel("clear px", fontsize=8)
    usable, total = fit_planes_usable(view)
    far = view["set"] == "far"
    ax_n.set_title("(c) every plane's stripe: usable in %d of the %s fit's %d planes%s" % (
        usable, view["set"], total, " (|k - k*| >= %d)" % int(np.abs(off[fit]).min()) if far and fit.any() else ""),
        fontsize=9, loc="left", color=INK)


def isolation_figure(view, title=""):
    """view: calibration_report.isolation_view. (a) the patch at plane k (grey
    levels) with the node's own branch (blue), each traced neighbour in orange
    when its predicted shadow in the patch exceeds the threshold over the
    stage's planes and in grey otherwise (dashed: not evaluated), the soma
    discs, the measuring line (green); (b) the same image with the pixels
    that are not far shaded blue -- the far pixels are those the planar
    background is fitted on; under the stripe (10.S), (c) every plane's
    stripe (_stripe_planes_panel). Returns the Figure."""
    plt = _plt()
    img = np.asarray(view["image"], dtype=float)
    ext = _extent(view, img.shape)
    lo, hi = np.percentile(img, [0.5, 99.8])
    by_plane = view.get("stripe_planes") is not None
    if by_plane:
        fig = plt.figure(figsize=(16.5, 6.8), facecolor=SURFACE)
        gs = fig.add_gridspec(2, 3, width_ratios=(1.0, 1.0, 0.95), height_ratios=(1.0, 2.6), hspace=0.32)
        axes = [fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[:, 1])]
        ax_r = fig.add_subplot(gs[1, 2])
        ax_n = fig.add_subplot(gs[0, 2], sharex=ax_r)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 6.8), facecolor=SURFACE)
    for ax in axes:
        ax.imshow(img, cmap="gray", vmin=lo, vmax=hi if hi > lo else lo + 1, extent=ext, interpolation="nearest")
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        ax.set_xlabel("x (um)", fontsize=8); ax.tick_params(colors=INK2, labelsize=7)
    axes[0].set_ylabel("y (um)", fontsize=8)
    _draw_scene(axes[0], view)
    w = view["worst"]
    if w is not None and w["kind"] == "segment" and view["evaluated"]:
        rows = np.atleast_2d(view["seg_rows"]).reshape(-1, 9)
        hit = [q for q, d in enumerate(view["seg_depth_gl"]) if np.isfinite(d) and d == w["depth_gl"]]
        if hit:
            r = rows[hit[0]]
            mx, my = 0.5 * (r[0] + r[3]), 0.5 * (r[1] + r[4])
            if ext[0] <= mx <= ext[1] and ext[3] <= my <= ext[2]:
                axes[0].annotate("%s gl" % _gl(w["depth_gl"]), (mx, my), color=ORANGE, fontsize=8, xytext=(4, 4),
                                 textcoords="offset points")
    axes[0].set_title("(a) the patch at plane %s = %d, with the traced segments\n(the test covers the planes "
                      "k* +- %d)" % ("k*" if view["plane_rule"] == "k_star" else "of the SWC node", view["k_view"],
                                    view["offsets_planes"]), fontsize=9, loc="left", color=INK)
    shade = np.zeros(img.shape + (4,))
    shade[..., 2] = 1.0
    shade[..., 3] = np.where(np.asarray(view["far"], dtype=bool), 0.0, 0.30)
    axes[1].imshow(shade, extent=ext, interpolation="nearest")
    _draw_line(axes[1], view["cx_um"], view["cy_um"], view["theta_rad"], view["line_half_um"], GREEN, mark_plus=by_plane)
    frac = float(np.mean(view["far"]))
    b = np.asarray(view["background"], dtype=float)
    st = view.get("stripe")
    bg = ("background plane %.1f gl at the centre, slope (%+.3f, %+.3f) gl/um" % (b[0], b[1], b[2])
          if view["background_ok"] else ("the stripe is not usable (%s): no background" % st["reason"] if st
                                         else "too few far pixels: no background"))
    if st:
        head_b = ("(b) the stripe (unshaded) the background is fitted on at k*: W +-%.0f um, runs -%.1f / +%.1f um "
                  "beyond the own cylinder (%.1f / %.1f um), %d px" % (st["w_um"], st["side_minus_um"], st["side_plus_um"],
                                                                       st["v_own_minus_um"], st["v_own_plus_um"],
                                                                       st["n_pixels"]))
        if by_plane and view["set"] == "far":
            head_b += ";\nthe far fit uses the stripes of the planes in (c)"
    else:
        head_b = "(b) the far pixels (unshaded) the background is fitted on: %.0f %%" % (100.0 * frac)
    axes[1].set_title("%s\n%s" % (head_b, bg), fontsize=9, loc="left", color=INK)
    head = title or "node %d -- %s-field set, Allen 2r %.2f um%s" % (
        view["node_id"], view["set"], view["allen_diameter_um"], "" if view["image_valid"] else " (plane missing)")
    fig.suptitle("%s\n%s\n%s" % (head, view["verdict"] or "", _isolation_line(view)), fontsize=10, color=INK,
                 x=0.01, y=0.985, ha="left", va="top")
    offenders = view["offenders"][:5]
    listed = "; ".join(("soma %.1f um away: %s gl" % (o["rho_um"], _gl(o["depth_gl"])) if o["kind"] == "soma" else
                        "segment %d-%d %.1f um away: %s gl" % (o["child_id"], o["parent_id"], o["rho_um"],
                                                               _gl(o["depth_gl"])))
                       for o in offenders)
    if by_plane:
        _stripe_planes_panel(ax_n, ax_r, view)
        listed_c = ("\n(c): per plane the runs beyond the own cylinder (light blue), bars as wide as the plane's W; dark "
                    "grey: usable, orange: not usable; ticks: the %s um a side needs; green dashed: the measuring line's "
                    "ends; faint: planes the fit does not use%s." % (
                        _gl(view["stripe_planes"]["min_side_um"]), " (half-faint: the dip-area planes)"
                        if view["set"] == "far" else ""))
    else:
        listed_c = ""
    fig.text(0.01, 0.012, "blue: the node's own branch; orange: neighbours whose predicted shadow in the patch exceeds "
             "%s gl in some plane of k* +- %d%s; grey: below it (dashed: not evaluated); green: the "
             "measuring line.%s%s" % (_gl(view["threshold_gl"]), view["offsets_planes"],
                                       " (not_isolated)" if view.get("support", "square") == "square" else
                                       " (they bound the stripe; the verdict is the stripe's)",
                                       ("\nabove the threshold: " + listed + (" ..." if len(view["offenders"]) > 5
                                                                              else "")) if listed else "", listed_c),
             fontsize=7.5, color=INK2, va="bottom")
    fig.subplots_adjust(left=0.05 if by_plane else 0.06, right=0.98, top=0.80, bottom=0.17 if by_plane else 0.15,
                        wspace=0.22 if by_plane else 0.16)
    return fig


def isolation_overview(views, title="", ncols=6, per_page=36):
    """The candidates' patches (plane k*) as thumbnails with their segments,
    titled with the node id and the deepest neighbour shadow (orange above
    the threshold); pages of per_page. Returns the list of Figures."""
    plt = _plt()
    figs = []
    for start in range(0, len(views), per_page):
        page = views[start:start + per_page]
        nrow = int(math.ceil(len(page) / float(ncols)))
        fig, axes = plt.subplots(nrow, ncols, figsize=(2.1 * ncols, 2.3 * nrow + 0.6), facecolor=SURFACE,
                                 squeeze=False)
        for ax in axes.ravel():
            ax.set_axis_off()
        for ax, view in zip(axes.ravel(), page):
            img = np.asarray(view["image"], dtype=float)
            ext = _extent(view, img.shape)
            lo, hi = np.percentile(img, [0.5, 99.8])
            ax.imshow(img, cmap="gray", vmin=lo, vmax=hi if hi > lo else lo + 1, extent=ext, interpolation="nearest")
            _draw_scene(ax, view, thin=True)
            ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
            st = view.get("stripe")
            if st is not None:
                bad = not st["ok"]
                label = "%d px" % st["n_pixels"] if st["ok"] else st["reason"]
                used = fit_planes_usable(view)
                if used is not None:
                    label += " | %d/%d" % used
            else:
                bad = view["evaluated"] and view["max_depth_gl"] > view["threshold_gl"]
                label = ("%s gl" % _gl(view["max_depth_gl"])) if view["evaluated"] else "n/e"
            ax.set_title("%d %s: %s" % (view["node_id"], view["set"], label), fontsize=7,
                         color=ORANGE if bad else INK)
        striped = views[0].get("stripe") is not None
        fig.suptitle((title or ("candidates' background patches (plane k*): orange = the stripe at k* is not usable "
                                "(too_few / side_short); the label is its pixel count and n/N, the planes of the set's "
                                "fit whose stripe is usable" if striped else
                                "candidates' background patches (plane k*): orange = a neighbour's predicted shadow "
                                "above %s gl (not_isolated)" % _gl(views[0]["threshold_gl"]))) + "  [%d-%d of %d]"
                     % (start + 1, start + len(page), len(views)), fontsize=9, color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        figs.append(fig)
    return figs


def write_all(scans, calib, far_fit, patches, cfg, out_dir, every_node=True, dpi=150, id_map=None):
    """The PNG files of a run in out_dir: width_by_plane_<set>.png, example_<set>.png, kernel_law.png, when
    the points span two cells or more kernel_law_by_cell.png and kernel_law_residuals_by_cell.png (Block 19.5;
    id_map: the pooled scans file's meta.id_map), and, when every_node, gaussians/<set>_node_<id>.png. Returns
    the paths written."""
    from ..analysis import calibration_report as CR
    inputs = CR.figure_inputs(scans, cfg, calib, far_fit, patches)
    written = []
    plt = _plt()
    for which, st in inputs["stages"].items():
        path = os.path.join(out_dir, "width_by_plane_%s.png" % which)
        fig = width_by_plane_figure(st)
        fig.savefig(path, dpi=dpi, facecolor=SURFACE, bbox_inches="tight"); plt.close(fig); written.append(path)
        ex = inputs["examples"].get(which)
        if ex is not None:
            half = cfg.measure.profile_half_um if which == "near" else cfg.calibration.far_profile_half_um
            path = os.path.join(out_dir, "example_%s.png" % which)
            fig = example_node_figure(ex, half)
            fig.savefig(path, dpi=dpi, facecolor=SURFACE, bbox_inches="tight"); plt.close(fig); written.append(path)
    if inputs["kernel_law"] is not None:
        path = os.path.join(out_dir, "kernel_law.png")
        fig = kernel_law_figure(inputs["kernel_law"])
        fig.savefig(path, dpi=dpi, facecolor=SURFACE, bbox_inches="tight"); plt.close(fig); written.append(path)
    kc = CR.kernel_law_by_cell(scans, cfg, calib, far_fit, id_map)
    if kc is not None and len(kc["cells"]) >= 2:
        for name, draw in (("kernel_law_by_cell", kernel_law_by_cell_figure),
                           ("kernel_law_residuals_by_cell", kernel_law_residuals_by_cell_figure)):
            path = os.path.join(out_dir, "%s.png" % name)
            fig = draw(kc)
            fig.savefig(path, dpi=dpi, facecolor=SURFACE, bbox_inches="tight"); plt.close(fig); written.append(path)
    if every_node:
        os.makedirs(os.path.join(out_dir, "gaussians"), exist_ok=True)
        for g in inputs["gaussians"]:
            path = os.path.join(out_dir, "gaussians", "%s_node_%d.png" % (g["set"], g["node_id"]))
            fig = gaussians_figure(g)
            fig.savefig(path, dpi=dpi, facecolor=SURFACE, bbox_inches="tight"); plt.close(fig); written.append(path)
    return written
