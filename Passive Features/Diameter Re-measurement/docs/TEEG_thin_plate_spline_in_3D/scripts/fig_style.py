"""fig_style.py -- shared matplotlib style of the document's figures (dataviz reference palette, light surface).
Series identity is fixed across figures: order 5/2 ('thin_plate_spline' on R^3) = slot 1 blue, order 2
('linear', degree 1, the 3-D bending-energy spline) = slot 2 orange, order 3 (r^3 on R^3) = slot 3 aqua.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PAL = dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
           tps="#2a78d6", lin="#eb6834", cub="#1baf7a")
LS = dict(tps="-", lin=(0, (5, 2)), cub=(0, (1.5, 1.5)))

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "mathtext.fontset": "dejavusans",
                     "axes.edgecolor": PAL["axis"], "axes.labelcolor": PAL["ink2"], "xtick.color": PAL["muted"],
                     "ytick.color": PAL["muted"], "axes.titlecolor": PAL["ink"], "legend.frameon": False})


def style(ax):
    ax.set_facecolor(PAL["surface"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color=PAL["grid"], lw=0.7)
    ax.set_axisbelow(True)


def new(nrows=1, ncols=1, w=7.8, h=4.4):
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h), dpi=170)
    fig.patch.set_facecolor(PAL["surface"])
    for ax in (axes.ravel() if hasattr(axes, "ravel") else [axes]):
        style(ax)
    return fig, axes


def footer(fig, text, y=-0.02):
    fig.text(0.01, y, text, color=PAL["muted"], fontsize=6.6, ha="left", va="top")


def save(fig, path):
    fig.savefig(path, bbox_inches="tight", facecolor=PAL["surface"])
    plt.close(fig)
