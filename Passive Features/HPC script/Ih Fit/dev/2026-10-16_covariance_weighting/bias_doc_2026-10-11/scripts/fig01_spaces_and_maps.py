"""Figure 1 (schematic, not to scale): the spaces and maps behind a bias.

Parameter space Theta --law--> distribution of windows on R^n --recipe t--> distribution of T on R;
Theta --target g--> a value on R.  The bias compares two points of R: E[T | theta, n] and g(theta).
Drawn shapes are illustrative (a cloud and a bump), not data.
Run:  python3 fig01_spaces_and_maps.py      (writes ../figures/fig01_spaces_and_maps.png)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

CONFIG = dict(out="../figures/fig01_spaces_and_maps.png", seed=3, dpi=170)
COL = dict(blue="#2a78d6", dblue="#1b4f8f", orange="#eb6834", purple="#7b3fbf", ink="#0b0b0b",
           ink2="#52514e", surface="#fcfcfb", box="#eef2f7")


def space_box(ax, x, y, w, h, title, subtitle):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                facecolor=COL["box"], edgecolor=COL["ink2"], linewidth=1.0))
    ax.text(x + 0.12, y + h - 0.12, title, ha="left", va="top", fontsize=10.5, color=COL["ink"],
            fontweight="bold")
    ax.text(x + w / 2, y - 0.08, subtitle, ha="center", va="top", fontsize=8.2, color=COL["ink2"])


def map_arrow(ax, p, q, text, text_xy):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=14, linewidth=1.4, color=COL["ink2"]))
    ax.text(*text_xy, text, ha="center", va="top", fontsize=8.5, color=COL["ink"])


def draw(cfg):
    rng = np.random.default_rng(cfg["seed"])
    fig, ax = plt.subplots(figsize=(11.6, 5.4), dpi=cfg["dpi"], facecolor=COL["surface"])
    ax.set_facecolor(COL["surface"])
    ax.set_xlim(0, 11.6)
    ax.set_ylim(-0.95, 4.75)
    ax.axis("off")

    # parameter space
    space_box(ax, 0.25, 0.9, 2.2, 2.5, "parameter space \u0398", "\u03b8 = (\u03c3_f, \u03c1_f, \u03c3_s, \u03c4_s) \u2208 \u0398\n(the law's parameters)")
    th = (1.35, 2.05)
    ax.plot(*th, "o", color=COL["ink"], markersize=7)
    ax.text(th[0], th[1] + 0.18, "\u03b8", ha="center", va="bottom", fontsize=12, color=COL["ink"])

    # window space: a cloud of windows (the law) and one recorded window
    space_box(ax, 3.5, 0.9, 2.8, 2.5, "window space \u211d\u207f", "p(\u00b7 | \u03b8, n): the law of a window\nof n samples (the cloud's density)")
    c = rng.multivariate_normal([4.95, 2.05], [[0.13, 0.05], [0.05, 0.07]], 240)
    c = c[(c[:, 0] > 3.7) & (c[:, 0] < 6.1) & (c[:, 1] > 1.25) & (c[:, 1] < 2.85)]
    ax.scatter(c[:, 0], c[:, 1], s=6, color=COL["blue"], alpha=0.45, linewidths=0)
    xw = (4.15, 1.45)
    ax.plot(*xw, "D", color=COL["purple"], markersize=7, zorder=5)
    ax.text(xw[0] + 0.12, xw[1] - 0.12, "x: the recorded window", ha="left", va="top", fontsize=8.0,
            color=COL["purple"])

    # values: the sampling distribution of T, its mean, one realized value, the target
    space_box(ax, 7.4, 0.9, 3.95, 2.5, "values \u211d", "")
    yb, mu, sd, g, tx = 1.25, 8.55, 0.42, 10.55, 9.2
    ax.plot([7.55, 11.2], [yb, yb], color=COL["ink2"], linewidth=1.0)
    xs = np.linspace(7.6, 10.2, 200)
    ax.fill_between(xs, yb, yb + 1.15 * np.exp(-0.5 * ((xs - mu) / sd) ** 2), color=COL["blue"], alpha=0.30,
                    linewidth=0)
    ax.text(7.62, 1.62, "distribution\nof T = t(X)", ha="left", va="bottom", fontsize=7.8, color=COL["dblue"])
    ax.plot([mu, mu], [yb, 2.6], color=COL["ink"], linewidth=1.6, linestyle="--")
    ax.text(mu, 2.63, "E[T | \u03b8, n]", ha="center", va="bottom", fontsize=8.2, color=COL["ink"])
    ax.plot([g, g], [yb, 3.4], color=COL["orange"], linewidth=2.2)
    ax.text(g + 0.08, 2.55, "g(\u03b8)", ha="left", va="center", fontsize=8.8, color=COL["orange"])
    ax.plot([tx], [yb], "D", color=COL["purple"], markersize=7, clip_on=False, zorder=5)
    ax.text(tx, yb + 0.12, "t(x)", ha="center", va="bottom", fontsize=8.2, color=COL["purple"])

    # brackets below the value line
    ax.annotate("", xy=(tx, 0.62), xytext=(g, 0.62), arrowprops=dict(arrowstyle="<->", color=COL["purple"], lw=1.2))
    ax.text((tx + g) / 2, 0.55, "error of x:  t(x) \u2212 g(\u03b8)", ha="center", va="top", fontsize=8.0,
            color=COL["purple"])
    ax.annotate("", xy=(mu, 0.02), xytext=(g, 0.02), arrowprops=dict(arrowstyle="<->", color=COL["ink"], lw=1.3))
    ax.text((mu + g) / 2, -0.05, "bias(T; g | \u03b8, n) = E[T | \u03b8, n] \u2212 g(\u03b8)", ha="center", va="top", fontsize=8.2,
            color=COL["ink"])

    # maps
    map_arrow(ax, (2.5, 2.05), (3.45, 2.05), "law\n\u03b8 \u21a6 p(\u00b7 | \u03b8, n)", (2.97, 1.92))
    map_arrow(ax, (6.35, 2.05), (7.35, 2.05), "recipe\nt: \u211d\u207f \u2192 \u211d\n(demean;\nS_L / S_0)", (6.85, 1.92))
    ax.add_patch(FancyArrowPatch((1.35, 3.45), (g, 3.45), arrowstyle="-|>", mutation_scale=14, linewidth=1.6,
                                 color=COL["orange"], connectionstyle="arc3,rad=-0.12"))
    ax.text(5.9, 4.42, "target g: \u0398 \u2192 \u211d, e.g. g(\u03b8) = \u03c1(1 ms | \u03b8): a property of the law alone (no window, no recipe)",
            ha="center", va="center", fontsize=8.6, color=COL["orange"])
    ax.text(0.25, -0.75, "Schematic, not to scale. Blue: the law and what it implies (analytic level); purple: one "
            "recorded window and its value (computed level); orange: the target.", ha="left", va="center",
            fontsize=8.0, color=COL["ink2"])
    fig.tight_layout()
    fig.savefig(cfg["out"], facecolor=COL["surface"])


if __name__ == "__main__":
    draw(CONFIG)
