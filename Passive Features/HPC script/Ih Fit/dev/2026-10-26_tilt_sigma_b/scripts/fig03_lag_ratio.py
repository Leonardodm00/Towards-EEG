"""fig03_lag_ratio.py -- PLOTTED from formulas (check02): the ratio of the tilt
SD that (T1) reads at lag h to the tilt SD that matches the Long Square
step's baseline error, when the variance at seconds is an OU of 3, 10 or 30 s,
for 120 ms SS windows (t_SS = 80 ms, solid) and 190 ms (t_SS = 10 ms, dashed).
Run:  python fig03_lag_ratio.py  ->  ../figures/fig03_lag_ratio.png
"""
from pathlib import Path

import numpy as np

import check02_lag_choice as c2
import fig_style as fs
import tilt_common as tc

CFG = dict(h=np.linspace(0.30, 2.5, 300), T=((0.120, "-"), (0.190, "--")),
           out=Path(__file__).resolve().parent.parent / "figures" / "fig03_lag_ratio.png")


def plot():
    plt = fs.setup()
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for i, (tau, sigma) in enumerate(c2.CFG["ous"]):
        sm = c2.sigma_b_match(sigma, tau)
        for T, ls in CFG["T"]:
            r = np.array([c2.sigma_b_read(sigma, tau, T, x) for x in CFG["h"]]) / sm
            ax.plot(CFG["h"], r, color=fs.SERIES[i], ls=ls, lw=2 if ls == "-" else 1.4,
                    label=("OU %g s, %d ms windows" % (tau, round(T * 1e3))))
    for L in (2, 4, 6, 8, 10, 12):
        ax.axvline(tc.h_of(L), color="#d8d7d2", lw=0.8, zorder=0)
    ax.axhline(1.0, color=fs.INK2, lw=1.0)
    ax.axvline(tc.h_of(4), color=fs.INK2, lw=1.0, ls=":")
    ax.text(tc.h_of(4) + 0.03, 1.55, r"$h^*$ = 4 periods", color=fs.INK2, fontsize=9)
    ax.text(tc.h_of(2) + 0.02, 0.45, "2", color=fs.INK2, fontsize=8)
    ax.text(tc.h_of(6) + 0.02, 0.45, "6", color=fs.INK2, fontsize=8)
    ax.text(tc.h_of(8) + 0.02, 0.45, "8 periods", color=fs.INK2, fontsize=8)
    ax.set_xlabel(r"lag $h$ at which (T1) is read (s)")
    ax.set_ylabel(r"$\sigma_b$(read at $h$) / $\sigma_b$(LS step match)")
    ax.set_ylim(0.4, 1.8)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    CFG["out"].parent.mkdir(exist_ok=True)
    fig.savefig(CFG["out"], dpi=150, bbox_inches="tight")
    print("wrote", CFG["out"])


if __name__ == "__main__":
    plot()
