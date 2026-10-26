"""The S3 survey's figures -- Blocks 16.2 (f) and 17 in specs/SPEC.md
(D-085.2, D-085.5). Plotting only: every array, frame and transform comes
prepared; nothing scientific is computed here.

  alignment_overlay  the projection at the coarse level with the skeleton under
                     the final transform, and the fine crop with the skeleton
                     under the coarse (grey) and the final (orange) transforms
  coverage_figures   S3a (Block 17.5): ln(mu_hat d_hat) against ln d_hat with
                     the band; the share inside per bin; ln alpha_hat per bin
  diameter_figure    S3a (Block 17.5): d_hat and Allen's 2r per layer, by class
  inputs_figure      S3b (Block 18.5): Allen's 2r pooled with d_max; the ON
                     nodes' offsets with their robust SDs; each cell's camera

Colours: the palette of the project's figures. Pure ASCII, for the cluster
(hpc-python-compat).
"""
from __future__ import annotations

import math

import numpy as np

import allen_image_plot as aip

GREY, INK, BLUE, ORANGE, GREEN = "#a9a8a2", "#0b0b0b", "#2a78d6", "#eb6834", "#1baf7a"
CLASS_COLOURS = {"exc": BLUE, "inh": ORANGE, "sparsely_spiny": GREEN}


def alignment_overlay(swc, images, rec, title=""):
    """Block 16.2 (f). swc: Block 0's SWC table; images: the second return of
    cell_alignment.align_record; rec: the alignment record."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 7))
    final = (float(rec["shift_x_px"]), float(rec["shift_y_px"]))
    flip = rec.get("flip_h_px")
    cf, ff = images["coarse_frame"], images["fine_frame"]
    aip.show_image(images["coarse"], cf.res_um_px, ax=axes[0], robust=True)
    aip.overlay_swc(axes[0], swc, cf, final, flip, color=ORANGE, lw=0.6)
    axes[0].set_title("projection, level %d (%.2f um/px)" % (cf.downsample, cf.res_um_px), fontsize=10)
    aip.show_image(images["fine"], ff.res_um_px, ax=axes[1], robust=True)
    ct = images["coarse_transform"]
    aip.overlay_swc(axes[1], swc, ff, ct["shift_full_px"], ct["flip_y_full_h"], color=GREY, lw=0.6, alpha=0.8)
    aip.overlay_swc(axes[1], swc, ff, final, flip, color=ORANGE, lw=0.6)
    fine = rec["fine"]
    axes[1].set_title("fine crop, level %d (%.2f um/px): coarse grey, final orange; fine step (%+d, %+d) px"
                      % (ff.downsample, ff.res_um_px, fine["dx_px"], fine["dy_px"]), fontsize=10)
    R = rec.get("score_ratio")
    fig.suptitle("%s  shift (%.1f, %.1f) full-res px, %s, score ratio %s -- %s" % (
        title, final[0], final[1], "y-flipped (H %.0f px)" % flip if flip is not None else "no flip",
        "%.2f" % R if R is not None else "n/a",
        "SET ASIDE (%s)" % ", ".join(rec["reasons"]) if rec.get("set_aside") else "accepted"), fontsize=11)
    fig.tight_layout()
    return fig


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _usable(n):
    return n["in_S"] and math.isfinite(n["d_hat"]) and n["d_hat"] > 0 and math.isfinite(n["mu_hat"]) and n["mu_hat"] > 0


def coverage_figures(nodes, bins, cfg, phantom=None):
    """Block 17.5: {"coverage_mud", "coverage_share", "coverage_alpha"} figures
    from analysis.coverage's node table and coverage table."""
    plt = _plt()
    mu_lo, mu_hi = (float(x) for x in cfg.phantom.mu_range_per_um)
    use = [n for n in nodes if _usable(n)]
    figs = {}
    fig, ax = plt.subplots(figsize=(8, 6))
    for cls in sorted({n["neuron_class"] for n in use}):
        sel = [n for n in use if n["neuron_class"] == cls]
        ax.scatter(np.log([n["d_hat"] for n in sel]), np.log([n["mu_hat"] * n["d_hat"] for n in sel]), s=8,
                   alpha=0.6, color=CLASS_COLOURS.get(cls, GREY), label="%s (%d)" % (cls, len(sel)))
    x = np.linspace(math.log(0.1), math.log(8.0), 50)
    ax.plot(x, math.log(mu_lo) + x, color=INK, lw=1, ls="--", label="ln mu_min + ln d_hat (mu %.2g /um)" % mu_lo)
    ax.plot(x, math.log(mu_hi) + x, color=INK, lw=1, ls=":", label="ln mu_max + ln d_hat (mu %.2g /um)" % mu_hi)
    for e in cfg.survey.coverage_dhat_bins_um:
        ax.axvline(math.log(e), color=GREY, lw=0.5)
    for b in bins:
        if math.isfinite(b.get("phantom_lo", float("nan"))):
            lo, hi = max(b["lo"], 0.1), min(b["hi"], 8.0)
            ax.hlines([b["phantom_lo"], b["phantom_hi"]], math.log(lo), math.log(hi), colors=ORANGE, lw=2)
    ax.set_xlabel("ln d_hat (d_hat in um; fitted, not corrected)")
    ax.set_ylabel("ln (mu_hat d_hat)")
    ax.set_title("S3a: the real nodes in S against the phantoms' band (D-065.1)", fontsize=10)
    ax.legend(fontsize=8)
    figs["coverage_mud"] = fig
    fig, ax = plt.subplots(figsize=(8, 4.5))
    xs = np.arange(len(bins))
    for i, b in enumerate(bins):
        counted = b["status"] in ("pass", "fail")
        val = b["share_inside"] if math.isfinite(b["share_inside"]) else 0.0
        ax.bar(i, val, color=(BLUE if b["status"] == "pass" else ORANGE) if counted else "white",
               edgecolor=INK if counted else GREY, hatch=None if counted else "//")
        ax.text(i, min(val, 1.0) + 0.02, "n=%d" % b["n"], ha="center", fontsize=8)
    ax.axhline(cfg.survey.coverage_pass_share, color=INK, ls="--", lw=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([b["bin"] for b in bins])
    ax.set_ylim(0, 1.12)
    ax.set_xlabel("bin of d_hat (um)")
    ax.set_ylabel("share inside the analytic band")
    ax.set_title("pass rule: >= %.0f %% in every bin with >= %d nodes (hatched: too few)"
                 % (100 * cfg.survey.coverage_pass_share, cfg.survey.coverage_min_nodes), fontsize=10)
    figs["coverage_share"] = fig
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, b in enumerate(bins):
        q = b["q_ln_alpha"]
        if not all(math.isfinite(v) for v in q):
            continue
        ax.add_patch(plt.Rectangle((i - 0.3, q[1]), 0.6, q[3] - q[1], facecolor=BLUE, alpha=0.4, edgecolor=INK))
        ax.plot([i - 0.3, i + 0.3], [q[2], q[2]], color=INK, lw=1.5)
        ax.plot([i, i], [q[0], q[1]], color=INK, lw=0.8)
        ax.plot([i, i], [q[3], q[4]], color=INK, lw=0.8)
        centre = math.sqrt(b["lo"] * b["hi"]) if (b["lo"] > 0 and math.isfinite(b["hi"])) else (
            b["hi"] / math.sqrt(2) if b["lo"] == 0 else b["lo"] * math.sqrt(2))
        ax.plot([i - 0.35, i + 0.35], [math.log(mu_lo * centre)] * 2, color=ORANGE, lw=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([b["bin"] for b in bins])
    ax.set_xlim(-0.6, len(bins) - 0.4)
    ax.relim()
    ax.autoscale_view(scalex=False)
    ax.set_xlabel("bin of d_hat (um)")
    ax.set_ylabel("ln alpha_hat")
    ax.set_title("ln alpha_hat per bin (box 25-75 %, whiskers 5-95 %); orange: ln mu_min + ln d at the bin's "
                 "centre (no upper edge: phi -> 90 deg)", fontsize=9)
    figs["coverage_alpha"] = fig
    return figs


def diameter_figure(nodes):
    """Block 17.5: per layer, d_hat (in S; fitted, not corrected) and Allen's
    2r by class, as boxes at 5-25-50-75-95 %."""
    plt = _plt()
    layers = sorted({n["layer"] for n in nodes})
    classes = sorted({n["neuron_class"] for n in nodes})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    width = 0.8 / max(len(classes), 1)
    for ax, key, title in ((axes[0], "d_hat", "d_hat (in S; fitted, not corrected), um"),
                           (axes[1], "D_allen", "Allen's 2r, um")):
        for j, cls in enumerate(classes):
            data, pos = [], []
            for i, lay in enumerate(layers):
                v = [n[key] for n in nodes if n["layer"] == lay and n["neuron_class"] == cls and math.isfinite(n[key])
                     and n[key] > 0 and (key != "d_hat" or n["in_S"])]
                if v:
                    data.append(v)
                    pos.append(i - 0.4 + width * (j + 0.5))
            if data:
                bp = ax.boxplot(data, positions=pos, widths=width * 0.9, whis=(5, 95), showfliers=False,
                                patch_artist=True, medianprops=dict(color=INK))
                for patch in bp["boxes"]:
                    patch.set_facecolor(CLASS_COLOURS.get(cls, GREY))
                    patch.set_alpha(0.6)
                ax.plot([], [], color=CLASS_COLOURS.get(cls, GREY), lw=6, alpha=0.6, label=cls)
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(["L%s" % l for l in layers])
        ax.set_yscale("log")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
    fig.suptitle("S3a: dendrite diameters per layer and class (D-085.5)", fontsize=11)
    fig.tight_layout()
    return fig


def inputs_figure(rad, jit, cam, sug):
    """Block 18.5: rad, jit, cam the per-cell records of s3_inputs; sug the
    suggestions {d_max, jitter, camera}."""
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    ax = axes[0]
    D = np.concatenate([np.asarray(c["D_allen"], dtype=float) for c in rad]) if rad else np.zeros(0)
    D = D[np.isfinite(D) & (D > 0)]
    if D.size:
        ax.hist(np.log10(D), bins=40, color=BLUE, alpha=0.7)
        dm = sug["d_max"]
        for key, ls, lab in (("value_um", "-", "d_max (max, D-086)"), ("p99_um", "--", "p99"), ("p99_5_um", ":", "p99.5")):
            if dm.get(key):
                ax.axvline(math.log10(dm[key]), color=ORANGE if key == "value_um" else INK, ls=ls, label="%s %.2f um"
                           % (lab, dm[key]))
        for r in dm.get("per_cell", []):
            ax.plot(math.log10(r["max_um"]), 0, marker="|", color=INK, ms=12)
        ax.legend(fontsize=8)
    ax.set_xlabel("log10 Allen's dendrite 2r (um); ticks: each cell's maximum")
    ax.set_title("d_max", fontsize=10)
    ax = axes[1]
    for key, col, lab in (("s_star", BLUE, "s* lateral"), ("dz_star", ORANGE, "dz* depth")):
        x = np.concatenate([np.asarray(c[key], dtype=float) for c in jit]) if jit else np.zeros(0)
        x = x[np.isfinite(x)]
        if x.size:
            ax.hist(x, bins=30, color=col, alpha=0.5, label="%s (%d ON nodes)" % (lab, x.size))
    jt = sug["jitter"]
    for key, col in (("jitter_xy_um", BLUE), ("jitter_z_um", ORANGE)):
        if jt.get(key):
            ax.axvline(jt[key], color=col, ls="--")
            ax.axvline(-jt[key], color=col, ls="--")
    ax.legend(fontsize=8)
    ax.set_xlabel("offset (um); dashed: +- the pooled robust SD (D-085.8)")
    ax.set_title("node jitter", fontsize=10)
    ax = axes[2]
    ids = [c["specimen_id"] for c in cam]
    B = [c["background_B_gl"] if c["background_B_gl"] is not None else np.nan for c in cam]
    N = [c["noise_sd_gl"] if c["noise_sd_gl"] is not None else np.nan for c in cam]
    xs = np.arange(len(ids))
    ax.plot(xs - 0.15, B, "o", color=BLUE, label="background B (gl)")
    ax2 = ax.twinx()
    ax2.plot(xs + 0.15, N, "s", color=ORANGE, label="noise SD (gl, upper bound)")
    cm = sug["camera"]
    if cm.get("background_B_gl") is not None:
        ax.axhline(cm["background_B_gl"], color=BLUE, ls="--")
    if cm.get("noise_sd_gl") is not None:
        ax2.axhline(cm["noise_sd_gl"], color=ORANGE, ls="--")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(i) for i in ids], rotation=90, fontsize=7)
    ax.set_title("camera per cell; dashed: the medians (D-085.9)", fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    fig.suptitle("S3b: the simulator's inputs across cells -- suggestions, not decisions", fontsize=11)
    fig.tight_layout()
    return fig
