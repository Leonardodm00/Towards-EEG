"""fig05_pooling.py -- PLOTTED from check04: the share of cells whose tilt SD
is off by more than a factor 2 (a zero counted as off), against the spread
omega of sigma_b across cells, for the four options; 2-sweep cells (left) and
10-sweep cells (right) of the 18-cell L3_exc cohort; sigma_med = 0.1 mV/s.
Run:  python check04_pooling_options.py  (first), then  python fig05_pooling.py
"""
from pathlib import Path

import numpy as np

import fig_style as fs

CFG = dict(res=Path(__file__).resolve().parent.parent / "results" / "check04_results.npz",
           scen=("omega 0", "omega 0.25", "omega 0.5", "omega 1.0"), omega=(0.0, 0.25, 0.5, 1.0),
           labels=("(a) per cell", "(b) cohort", "(c-DL) partial pooling, log scale", "(c-EB) moderated variance"),
           out=Path(__file__).resolve().parent.parent / "figures" / "fig05_pooling.png")


def plot():
    z = np.load(CFG["res"], allow_pickle=False)
    arr, scen = z["arr"], [str(s) for s in z["scenarios"]]
    idx = [scen.index(s) for s in CFG["scen"]]
    plt = fs.setup()
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2), sharey=True)
    for g, (ax, title) in enumerate(zip(axes, ("2-sweep cells (14 of 18)", "10-sweep cells (3 of 18)"))):
        for o, lab in enumerate(CFG["labels"]):
            y = 100 * arr[idx, g, o, 1]
            ax.plot(CFG["omega"], y, color=fs.SERIES[o], ls=fs.LS[o], marker=fs.MK[o], ms=6, lw=2, label=lab)
            if g == 0:
                ax.annotate("%.0f" % y[-1], (CFG["omega"][-1], y[-1]), textcoords="offset points", xytext=(6, -3),
                            fontsize=8, color=fs.INK2)
        ax.set_title(title)
        ax.set_xlabel(r"$\omega$, log-SD of $\sigma_b(c)$ across cells")
        ax.set_xticks(CFG["omega"])
        ax.set_xlim(-0.05, 1.15)
    axes[0].set_ylabel(r"cells with $\tilde\sigma_b/\sigma_b$ outside [0.5, 2] (%)")
    axes[0].legend(loc="upper left", fontsize=8.2)
    fig.tight_layout()
    CFG["out"].parent.mkdir(exist_ok=True)
    fig.savefig(CFG["out"], dpi=150, bbox_inches="tight")
    print("wrote", CFG["out"])


if __name__ == "__main__":
    plot()
