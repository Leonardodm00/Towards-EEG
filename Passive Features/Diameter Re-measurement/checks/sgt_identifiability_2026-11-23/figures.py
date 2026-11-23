#!/usr/bin/env python3
"""Standalone check (NOT pipeline code), 2026-11-23 (D-222): the figures of the SGT identifiability check, from
fisher_bias.json and recovery_d*.json (plotting only; computes nothing scientific beyond means over the
stored fits).

  fig_d_error_se.png   per kernel truth (columns): top, the recovered d's relative error (mean +- SD over the
                       data sets) for today's Gaussian fit and the two SGT fits, with the Gaussian fit's
                       noise-free bias of part (b); bottom, SE(d)/d from the Fisher information at the truth for
                       the shape known, (beta, xi) free and (beta, xi, zeta) free.
  fig_correlations.png corr(d, beta), corr(d, xi), corr(d, zeta) of the zf model at the truth per geometry
                       (filled: unweighted; open: Block 39's centre weights), one panel per truth.
  fig_bias_gauss.png   part (b): the Gaussian assumption's bias in d (%) against d, per SGT truth, and the
                       symmetric option's (z1) bias on the skewed truth.

Run:  python3 figures.py   (after fisher_bias.py and recovery.py). Pure ASCII, LF.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

import cases as C  # noqa: E402

HERE = Path(__file__).resolve().parent
# the dataviz skill's reference categorical slots 1-3 (validated all-pairs, light mode) and text inks
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
MODEL_STYLE = {"gauss": (S1, "o", "Gaussian kernel (today's fit)"), "z1": (S2, "s", "SGT, (beta, xi) free, zeta = 1"),
               "zf": (S3, "^", "SGT, (beta, xi, zeta) free"), "known": (INK2, "D", "shape known (reference)")}
TRUTH_TITLE = {"gauss": "truth Gaussian (2, 0, 1)", "heavy": "truth heavy tail (2, 0.5, 1)",
               "b13": "truth beta 1.3 (1.3, 0, 1)", "skew": "truth skewed (2, 0.3, 1.3)"}


def style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=7)


def geom_labels():
    return [C.geom_id(g) for g in C.geometries()]


def short(gid):
    # d0.4_mu0.5_z0.0 -> "0.4|0.5|0"
    d, mu, z = gid.split("_")
    return "%s\n%s\n%s" % (d[1:], mu[2:], z[1:].rstrip("0").rstrip(".") or "0")


def load():
    fb = json.loads((HERE / "fisher_bias.json").read_text())
    rec = {}
    for path in sorted(glob.glob(str(HERE / "recovery_d*.json"))):
        for c in json.loads(Path(path).read_text())["cases"]:
            rec[(c["geom"], c["truth"])] = c
    return fb, rec


def fig_d_error_se(fb, rec):
    gids = geom_labels()
    x = np.arange(len(gids))
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.5), sharex=True)
    a_rows = fb["a"]
    b_rows = {(r["geom"], r["truth"], r["fit_model"]): r for r in fb["b"]}
    for col, truth in enumerate(C.TRUTH_ORDER):
        ax = axes[0, col]
        style(ax)
        ax.axhline(0.0, color=INK2, lw=0.8)
        models = ("gauss",) + C.models_for(truth)
        for i, m in enumerate(models):
            colr, mk, lab = MODEL_STYLE[m]
            mean, sd = [], []
            for gid, g in zip(gids, C.geometries()):
                c = rec.get((gid, truth))
                s = c["summary"].get(m, {}) if c else {}
                mean.append(100.0 * s.get("d_err_mean", np.nan) / g["d"])
                sd.append(100.0 * s.get("d_err_sd", np.nan) / g["d"])
            ax.errorbar(x + (i - 1) * 0.22, mean, yerr=sd, fmt=mk, ms=5, color=colr, ecolor=colr, elinewidth=1.2,
                        capsize=0, label=lab + " (recovery mean +- SD)")
        bias = [100.0 * b_rows[(gid, truth, "gauss")]["fit_bias_d_rel"] for gid in gids]
        ax.plot(x - 0.22, bias, "o", ms=8, mfc="none", mec=INK, mew=1.0, label="Gaussian fit, noise-free bias (b)")
        if truth == "skew":
            bz = [100.0 * b_rows[(gid, truth, "z1")]["fit_bias_d_rel"] for gid in gids]
            ax.plot(x + 0.0, bz, "s", ms=8, mfc="none", mec=S2, mew=1.0, label="z1 fit, noise-free bias (b)")
        ax.set_title(TRUTH_TITLE[truth], fontsize=9, color=INK)
        if col == 0:
            ax.set_ylabel("(d_hat - d) / d  (%)", fontsize=8, color=INK)
        ax.legend(fontsize=6, loc="upper right", frameon=False)
        ax2 = axes[1, col]
        style(ax2)
        for m in ("known",) + C.models_for(truth):
            colr, mk, lab = MODEL_STYLE[m]
            vals = []
            for gid, g in zip(gids, C.geometries()):
                rr = [r for r in a_rows if r["geom"] == gid and r["truth"] == truth and r["model"] == m]
                vals.append(100.0 * rr[0]["se_d"] / g["d"] if rr else np.nan)
            ax2.plot(x, vals, mk + "-", ms=5, lw=1.2, color=colr, label=lab)
            # the recovery's empirical SD of d_hat, for the same model
            if m != "known":
                emp = []
                for gid, g in zip(gids, C.geometries()):
                    c = rec.get((gid, truth))
                    s = c["summary"].get(m, {}) if c else {}
                    emp.append(100.0 * s.get("d_err_sd", np.nan) / g["d"])
                ax2.plot(x, emp, mk, ms=7, mfc="none", mec=colr, mew=1.0, label=lab + ": SD of d_hat (recovery)")
        ax2.set_yscale("log")
        if col == 0:
            ax2.set_ylabel("SE(d) / d  (%), Fisher at the truth", fontsize=8, color=INK)
        ax2.set_xticks(x)
        ax2.set_xticklabels([short(g) for g in gids], fontsize=6)
        ax2.set_xlabel("d (um) | mu (1/um) | z_ax (um)", fontsize=7, color=INK2)
        ax2.legend(fontsize=6, loc="upper right", frameon=False)
    fig.suptitle("SGT kernel in the per-node joint fit: error of d (top) and its SE (bottom); noise 1.1 gl, B 150 gl, "
                 "planes -4..4; 30 data sets per case (15 at d = 2.5 um)", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(HERE / "fig_d_error_se.png", dpi=130)
    plt.close(fig)


def fig_correlations(fb):
    gids = geom_labels()
    x = np.arange(len(gids))
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.0), sharey=True)
    for col, truth in enumerate(C.TRUTH_ORDER):
        ax = axes[col]
        style(ax)
        for yv in (-0.5, 0.0, 0.5):
            ax.axhline(yv, color=INK2 if yv == 0 else GRID, lw=0.8)
        for i, (par, colr, mk) in enumerate((("beta", S1, "o"), ("xi", S2, "s"), ("zeta", S3, "^"))):
            vals, wv = [], []
            for gid in gids:
                rr = [r for r in fb["a"] if r["geom"] == gid and r["truth"] == truth and r["model"] == "zf"][0]
                vals.append(rr["corr_d_" + par])
                wv.append(rr["w_corr_d_" + par])
            ax.plot(x + (i - 1) * 0.2, vals, mk, ms=6, color=colr, label="corr(d, %s)" % par)
            ax.plot(x + (i - 1) * 0.2, wv, mk, ms=6, mfc="none", mec=colr, mew=1.0,
                    label="corr(d, %s), centre weights" % par)
        ax.set_ylim(-1.0, 1.0)
        ax.set_title(TRUTH_TITLE[truth] + ", zf model", fontsize=9, color=INK)
        ax.set_xticks(x)
        ax.set_xticklabels([short(g) for g in gids], fontsize=6)
        ax.set_xlabel("d (um) | mu (1/um) | z_ax (um)", fontsize=7, color=INK2)
        if col == 0:
            ax.set_ylabel("correlation at the truth (Fisher)", fontsize=8, color=INK)
            ax.legend(fontsize=6, loc="lower left", frameon=False, ncol=2)
    fig.suptitle("corr(d, shape) at the truth, the pipeline's fine lattice (line_fine_factor 8); on a converged lattice "
                 "(64) corr(d, zeta) is about 0 and |corr(d, beta)|, |corr(d, xi)| <= 0.17 (lattice_check.out)",
                 fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(HERE / "fig_correlations.png", dpi=130)
    plt.close(fig)


def fig_bias(fb):
    panels = (("heavy", "gauss"), ("b13", "gauss"), ("skew", "gauss"), ("skew", "z1"))
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), sharey=True)
    for ax, (truth, model) in zip(axes, panels):
        style(ax)
        ax.axhline(0.0, color=INK2, lw=0.8)
        for mu, colr in ((0.5, S1), (1.5, S2)):
            for z, ls, mk in ((0.0, "-", "o"), (0.5, "--", "s")):
                rr = sorted([r for r in fb["b"] if r["truth"] == truth and r["fit_model"] == model
                             and r["mu"] == mu and r["z_ax"] == z], key=lambda r: r["d"])
                ax.plot([r["d"] for r in rr], [100.0 * r["fit_bias_d_rel"] for r in rr], ls, marker=mk, ms=5,
                        lw=1.4, color=colr, label="mu %.1f /um, axis %s" % (mu, "on k*" if z == 0 else "0.5 um off"))
        ax.set_xscale("log")
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xticks(C.D_LIST)
        ax.set_xticklabels([str(d) for d in C.D_LIST], fontsize=7)
        ax.set_xlabel("d (um)", fontsize=8, color=INK2)
        ax.set_title("%s; fit: %s" % (TRUTH_TITLE[truth], "Gaussian" if model == "gauss" else "SGT zeta = 1"),
                     fontsize=8, color=INK)
    axes[0].set_ylabel("noise-free bias of d (%)", fontsize=8, color=INK)
    axes[0].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(HERE / "fig_bias_gauss.png", dpi=130)
    plt.close(fig)


def main():
    fb, rec = load()
    fig_d_error_se(fb, rec)
    fig_correlations(fb)
    fig_bias(fb)
    print("wrote fig_d_error_se.png, fig_correlations.png, fig_bias_gauss.png (%d recovery cases)" % len(rec))


if __name__ == "__main__":
    main()
