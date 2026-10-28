"""fig_style.py -- plotting style shared by the figure scripts (plotting layer
only). Palette: the dataviz skill's reference categorical slots 1-4 (light
mode), validated with its validate_palette.js on 2026-10-26: all checks pass,
contrast WARN for slots 3-4, hence direct labels and distinct line styles.
Pure ASCII, LF."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")    # blue, orange, aqua, yellow
GREY = "#8a8985"
INK, INK2 = "#0b0b0b", "#52514e"
SURFACE = "#fcfcfb"
LS = ("-", "--", "-.", ":")
MK = ("o", "s", "^", "D")


def setup():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GREY, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2.0,
        "font.size": 10, "axes.titlesize": 10.5, "legend.frameon": False,
    })
    return plt
