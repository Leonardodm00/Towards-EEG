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
  write_all              the PNG files of a run

Colours: the palette of the project's figures (light grey for the nodes,
blue for the median and the law, orange for the example and the far line,
green for the near table). Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import math
import os

import numpy as np

GREY, INK, INK2 = "#a9a8a2", "#0b0b0b", "#52514e"
BLUE, ORANGE, GREEN, SURFACE, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#fcfcfb", "#e4e3df"


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


def _draw_line(ax, cx, cy, theta, half_um, color):
    y_hat = np.array([-math.sin(theta), math.cos(theta)])
    ax.plot([cx - half_um * y_hat[0], cx + half_um * y_hat[0]], [cy - half_um * y_hat[1], cy + half_um * y_hat[1]],
            "-", color=color, lw=0.9)


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


def write_all(scans, calib, far_fit, patches, cfg, out_dir, every_node=True, dpi=150):
    """The PNG files of a run in out_dir: width_by_plane_<set>.png, example_<set>.png, kernel_law.png and, when
    every_node, gaussians/<set>_node_<id>.png. Returns the paths written."""
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
    if every_node:
        os.makedirs(os.path.join(out_dir, "gaussians"), exist_ok=True)
        for g in inputs["gaussians"]:
            path = os.path.join(out_dir, "gaussians", "%s_node_%d.png" % (g["set"], g["node_id"]))
            fig = gaussians_figure(g)
            fig.savefig(path, dpi=dpi, facecolor=SURFACE, bbox_inches="tight"); plt.close(fig); written.append(path)
    return written
