"""Figure 1 (schematic): the bootstrap's one move, and the types ordered by how much model goes into P_hat.

Drawing only, no data. Run:  python fig01_two_worlds.py  -> ../figures/fig01_two_worlds.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent.parent / "figures" / "fig01_two_worlds.png"
C_REAL, C_BOOT, C_TYPE = "#e8eef7", "#fbeee6", "#f2f2f2"

TYPES = [  # (title, two short lines); order = increasing model content
    ("Nonparametric (Efron)", r"$\hat P=\hat F_n$: draw units" "\n" "with replacement; units iid"),
    ("Bayesian bootstrap", "Dirichlet(1,...,1) weights\non the units; units iid"),
    ("Hierarchical", "draw cells, then windows\nwithin each drawn cell"),
    ("Block (MBB, CBB, SB)", r"draw runs of $b$ samples;" "\n" r"dependence $\ll b\ll n$"),
    ("Sieve (AR)", "fit AR(p), redraw the\ninnovations, regenerate"),
    ("Wild", r"$m(\hat\theta)+v_i\hat\varepsilon_i$, $v_i=\pm1$;" "\n" "unequal variances kept"),
    ("Residual", r"$m(\hat\theta)$ + residuals drawn" "\n" r"with replacement; iid $\varepsilon$"),
    ("Parametric", r"$\mathbf{X}^*\sim p(\cdot\mid\hat\theta)$;" "\n" "the model is the law"),
]


def box(ax, x, y, w, h, text, fc, fs=9, bold_first=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
                                fc=fc, ec="0.35", lw=0.9))
    if bold_first:
        first, rest = text.split("\n", 1)
        ax.text(x + w / 2, y + h * 0.72, first, ha="center", va="center", fontsize=fs, weight="bold")
        ax.text(x + w / 2, y + h * 0.33, rest, ha="center", va="center", fontsize=fs - 1)
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, p, q, text="", dy=0.018, style="-|>", ls="-", color="0.2", rad=0.0, fs=8):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=12, lw=1.1, color=color,
                                 linestyle=ls, connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + dy, text, ha="center", va="bottom", fontsize=fs,
                color=color)


def main():
    fig = plt.figure(figsize=(13, 7.6))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # --- the two worlds -------------------------------------------------------------------
    for x0, title, fc in ((0.015, "REAL WORLD  (analytic level: P is unknown)", C_REAL),
                          (0.515, r"BOOTSTRAP WORLD  (built from the recorded $\mathbf{x}$)", C_BOOT)):
        ax.add_patch(FancyBboxPatch((x0, 0.555), 0.47, 0.42, boxstyle="round,pad=0.004,rounding_size=0.015",
                                    fc=fc, ec="0.5", lw=1.0))
        ax.text(x0 + 0.235, 0.952, title, ha="center", va="center", fontsize=10.5, weight="bold")
    w, h, y = 0.125, 0.15, 0.685
    real = [(0.03, "$P$\nunknown law of the data\n(Stage 7: each window\n" r"$\mathbf{X}\sim p(\cdot\mid\theta,n)$)"),
            (0.19, r"$\mathbf{X}$ (random)" "\n" r"recorded: $\mathbf{x}$" "\n" r"($K$ windows of $n$ samples)"),
            (0.35, r"$T=t(\mathbf{X})$" "\n" r"value: $t(\mathbf{x})$")]
    boot = [(0.53, r"$\hat P$" "\n" r"a law built from $\mathbf{x}$" "\n" "(how: the types below)"),
            (0.69, r"$\mathbf{X}^*\sim\hat P$ given $\mathbf{x}$" "\n" r"replicates $\mathbf{x}^*_1,\dots,\mathbf{x}^*_B$"),
            (0.85, r"$T^*=t(\mathbf{X}^*)$" "\n" r"values $t^*_1,\dots,t^*_B$")]
    for (x0, s) in real:
        box(ax, x0, y, w, h, s, "white", fs=8.5)
    for (x0, s) in boot:
        box(ax, x0, y, w, h, s, "white", fs=8.5)
    arrow(ax, (0.155, y + h / 2), (0.19, y + h / 2), "draw")
    arrow(ax, (0.315, y + h / 2), (0.35, y + h / 2), "recipe $t$")
    arrow(ax, (0.655, y + h / 2), (0.69, y + h / 2), "draw $B$\ntimes", dy=0.012)
    arrow(ax, (0.815, y + h / 2), (0.85, y + h / 2), "same $t$")
    ax.text(0.25, 0.595, "wanted: the law of $T$ (its bias, SD, quantiles);\none recorded $\\mathbf{x}$ gives one draw of it",
            ha="center", va="center", fontsize=8.5)
    ax.text(0.75, 0.595, r"read off: the law of $T^*$ given $\mathbf{x}$: $\sigma_{\rm boot}$ (E&N 3.7)," "\n"
            r"bias $\bar t^{\,*}-t(\mathbf{x})$ (E&N 3.14), percentile / BCa intervals", ha="center", va="center",
            fontsize=8.5)
    # plug-in arrow x -> P_hat, and the 'stands in for' arrow back
    arrow(ax, (0.30, y), (0.555, y), rad=0.28, color="C3")
    ax.text(0.43, 0.618, r"plug-in: build $\hat P$ from $\mathbf{x}$", ha="center", va="top", fontsize=8.5,
            color="C3")
    arrow(ax, (0.91, y + h), (0.415, y + h), rad=0.10, ls="--", color="C0")
    ax.text(0.715, 0.902, "law of $T^*$ stands in for law of $T$: right only if $\\hat P\\approx P$ "
            "in every feature $t$ depends on", ha="center", va="bottom", fontsize=8.5, color="C0")

    # --- the types along the model-content axis --------------------------------------------
    ax.text(0.5, 0.505, r"How $\hat P$ is built: the types, ordered roughly by how much model goes into $\hat P$",
            ha="center", va="center", fontsize=10.5, weight="bold")
    ya = 0.255
    arrow(ax, (0.02, ya), (0.98, ya), style="-|>", color="0.25")
    ax.text(0.015, ya - 0.02, "no model:\nresample what\nwas observed", ha="left", va="top", fontsize=8.5,
            color="0.25")
    ax.text(0.985, ya + 0.02, "full model:\nsimulate the\nfitted law", ha="right", va="bottom", fontsize=8.5,
            color="0.25")
    bw, bh = 0.20, 0.13
    for k, (title, body) in enumerate(TYPES):
        xc = 0.11 + 0.11 * k
        yb = ya + 0.035 if k % 2 == 0 else ya - 0.06 - bh
        box(ax, xc - bw / 2, yb, bw, bh, title + "\n" + body, C_TYPE, fs=8.5, bold_first=True)
        ax.plot([xc, xc], [ya, yb if k % 2 == 0 else yb + bh], color="0.6", lw=0.8)
    ax.text(0.5, 0.025, "Variants of the nonparametric type for non-smooth statistics: smoothed (draw from a kernel-"
            "smoothed estimate) and m-out-of-n (draw m < n units).   SCHEMATIC, not to scale.",
            ha="center", va="center", fontsize=8, style="italic", color="0.3")
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=160)
    print("saved", OUT)


if __name__ == "__main__":
    main()
