"""Figure of the diameter-reach check (2026-11-22): the shift a passive fit of the somatic responses makes in
(Cm, Rm, Ra) when every dendritic diameter beyond path distance x is wrong by a factor s, against x; one row
per parameter set (each diameter_reach_<label>.csv given), one column per fit target; the bottom panel the share
of the cell's membrane beyond x. Plotting only: reads the CSV and JSON of diameter_reach.py.
Usage: python3 plot_diameter_reach.py <out png> <csv> [<csv> ...]
Palette: the dataviz skill's reference categorical slots 1-3 (validated all-pairs, light surface); identity also
by marker and direct label; s by line style (solid 1.2, dashed 0.8). Pure ASCII.
"""
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COLOURS = {"Cm": "#2a78d6", "Rm": "#eb6834", "Ra": "#1baf7a"}
MARKERS = {"Cm": "o", "Rm": "s", "Ra": "^"}
LABELS = {"Cm": "Cm", "Rm": "Rm", "Ra": "Ra"}
FITS = (("pulse", "fit to the short pulse (the fitter's training target)"),
        ("pulse+step", "fit to the pulse and the 1 s step"))
TITLES = {"eyal2016_060308": "Eyal 2016 cell 060308 (Cm 0.452, Rm 38.9 k, Ra 203)",
          "matched": "Rm, Cm matched to this cell's R_in and tau"}


def read(path):
    with open(path) as f:
        rows = [{k: (float(v) if k not in ("label",) else v) for k, v in r.items()} for r in csv.DictReader(f)]
    js = os.path.splitext(path)[0] + ".json"
    summary = json.load(open(js)) if os.path.exists(js) else {}
    return rows, summary


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASE)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def panel(ax, rows, fit, tol, stars, title):
    style(ax)
    ax.axhspan(-100 * tol, 100 * tol, color=GRID, alpha=0.7, lw=0, zorder=0)
    ax.axhline(0.0, color=BASE, lw=0.8, zorder=1)
    xmax = max(r["x_um"] for r in rows)
    for p in ("Cm", "Rm", "Ra"):
        for s, ls in ((1.2, "-"), (0.8, "--")):
            sub = sorted((r for r in rows if abs(r["scale"] - s) < 1e-9), key=lambda r: r["x_um"])
            x = np.array([r["x_um"] for r in sub])
            y = 100 * np.array([r["%s_%s" % (p, fit)] for r in sub])
            ax.plot(x, y, ls=ls, lw=1.6, color=COLOURS[p], marker=MARKERS[p], ms=4.5,
                    mfc=COLOURS[p] if s > 1 else SURFACE, mec=COLOURS[p], zorder=3)
    lines = ["x* (um), s 1.2 / 0.8"]
    for p in ("Cm", "Rm", "Ra"):
        vals = []
        for s in (1.2, 0.8):
            xs = stars.get("%.2f|%s|%s" % (s, fit, p))
            vals.append(("%.0f" % xs) if xs == xs else "-")
        lines.append("%s  %s / %s" % (LABELS[p], vals[0], vals[1]))
    ax.text(0.985, 0.97, "\n".join(lines), transform=ax.transAxes, ha="right", va="top", fontsize=8, color=INK,
            family="monospace", bbox=dict(boxstyle="round,pad=0.35", fc=SURFACE, ec=BASE, lw=0.6))
    ax.set_xlim(-10, xmax + 20)
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    ax.set_ylabel("fitted parameter shift (%)", fontsize=8, color=INK2)


def main(argv):
    out, paths = argv[0], argv[1:]
    sets = [read(p) for p in paths]
    n = len(sets)
    fig = plt.figure(figsize=(12, 3.6 * n + 2.6), facecolor=SURFACE, layout="constrained")
    gs = fig.add_gridspec(n + 1, 2, height_ratios=[1.0] * n + [0.6])
    tol = 0.05
    for i, (rows, summ) in enumerate(sets):
        label = summ.get("label", "eyal2016_060308")
        tol = summ.get("config", {}).get("tol", tol)
        stars = summ.get("x_star_um", {})
        for j, (fit, ftitle) in enumerate(FITS):
            ax = fig.add_subplot(gs[i, j])
            cfgd = summ.get("config", {})
            head = TITLES.get(label, label)
            if label == "matched":
                head = ("Rm %.1f k, Cm %.2f: matched to this cell's R_in and tau (Cm not physical)"
                        % (cfgd.get("rm_ohm_cm2", float("nan")) / 1e3, cfgd.get("cm_uF_cm2", float("nan"))))
            panel(ax, rows, fit, tol, stars, "%s\n%s; R_in %.0f MOhm" % (head, ftitle,
                                                                       summ.get("r_in_1s_MOhm", float("nan"))))
            if i == n - 1:
                ax.set_xlabel("x: path distance from the soma beyond which every dendritic diameter is scaled (um)",
                              fontsize=8, color=INK2)
    ax = fig.add_subplot(gs[n, :])
    style(ax)
    rows = sorted((r for r in sets[0][0] if abs(r["scale"] - 1.2) < 1e-9), key=lambda r: r["x_um"])
    x = [r["x_um"] for r in rows]
    ax.plot(x, [100 * r["membrane_beyond"] for r in rows], color=INK2, lw=1.6, marker="o", ms=4,
            label="membrane beyond x, spines (F) included: % of the whole cell's")
    ax.plot(x, [100 * r["shaft_area_beyond"] for r in rows], color=MUTED, lw=1.2, ls=":", marker="o", ms=3.5,
            label="dendritic shaft area beyond x: % of all dendritic shafts")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.set_ylabel("% of membrane", fontsize=8, color=INK2)
    ax.set_xlabel("x: path distance from the soma (um)", fontsize=8, color=INK2)
    ax.set_ylim(0, 105)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Line2D([], [], color=COLOURS[p], marker=MARKERS[p], lw=1.6, label=LABELS[p]) for p in ("Cm", "Rm", "Ra")]
    handles += [Line2D([], [], color=INK2, lw=1.6, ls="-", label="diameters beyond x x 1.2 (filled markers)"),
                Line2D([], [], color=INK2, lw=1.6, ls="--", label="diameters beyond x x 0.8 (hollow markers)"),
                Patch(color=GRID, label="+-%.0f %% (the threshold)" % (100 * tol))]
    fig.legend(handles=handles, loc="outside lower center", ncol=6, fontsize=8, frameon=False)
    fig.suptitle("Cell 529878215: how far from the soma a dendritic diameter error still shifts the passive fit "
                 "(linearised, one Gauss-Newton step)", fontsize=10, color=INK)
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1:])
