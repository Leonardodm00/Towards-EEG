"""viz_style.py -- shared matplotlib styling of the figures of
TEEG_semivariogram_2026-11-13.md (plotting layer only; no data, no science).

Colours: the dataviz skill's reference palette, light mode (categorical slots
1-3 validated adjacent: blue, orange, aqua), text tokens for every label, a
neutral ink for the reference curve, hairline solid grid.

Pure ASCII, LF.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
GRID = "#e6e5e1"
NEUTRAL = "#52514e"          # reference curve (the floor): text-secondary ink
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")   # slots 1-4: blue, orange, aqua, yellow
LW = 1.8                     # pt, lines
MS = 5.5                     # pt, marker size (ringed)
RING = 1.2                   # pt, surface ring around markers


def apply() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.linewidth": 0.8,
        "axes.labelcolor": TEXT2, "axes.titlecolor": TEXT, "axes.titlesize": 10.5,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.labelsize": 9.5,
        "xtick.color": TEXT2, "ytick.color": TEXT2, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "xtick.major.size": 0, "ytick.major.size": 0, "xtick.minor.size": 0, "ytick.minor.size": 0,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.fontsize": 8.5, "legend.labelcolor": TEXT2,
        "font.family": "DejaVu Sans", "font.size": 9, "mathtext.fontset": "dejavusans",
        "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    })


def ringed(ax, x, y, yerr, color, label=None, zorder=4):
    """Estimator dots with +/- error bars, each dot ringed in the surface colour."""
    ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor=color, elinewidth=0.9, alpha=0.55, zorder=zorder - 1)
    ax.plot(x, y, "o", ms=MS, color=color, mec=SURFACE, mew=RING, label=label, zorder=zorder)
