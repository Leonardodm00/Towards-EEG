"""fig01_window_means.py -- PLOTTED: the enlarged-window means of one simulated
2-sweep cell (noise law theta_cf with planted tilts, no pulse response),
against the window centre, one panel per sweep; the sweep's tilt line; three
same-polarity pairs four positions apart.

Simulation: tilt_common (the pipeline's simulator and window functions),
layout 2 sweeps x 37 positions with 11 dropped at random, tilts b = +0.10
and -0.06 mV/s planted (the unit tilt draws replaced), seed below.
Run:  python fig01_window_means.py   ->  ../figures/fig01_window_means.png
"""
from pathlib import Path

import numpy as np

import fig_style as fs
import tilt_common as tc

CFG = dict(seed=20261030, K=63, b=(0.10, -0.06), pairs_shown=((0, 4), (13, 17), (26, 30)),
           out=Path(__file__).resolve().parent.parent / "figures" / "fig01_window_means.png")


def data():
    rng = np.random.default_rng(CFG["seed"])
    lay = tc.build_layout(2, CFG["K"], rng)
    units = tc.R.nl.draw_units(rng, lay.n_sweeps, tc.L_SWEEP)
    sb = 1.0                                            # sigma_b x xi = b exactly with xi = b
    units = tc.R.nl.UnitDraws(units.fast, units.slow, np.array(CFG["b"]), units.offset)
    v, _ = tc.simulate_windows(tc.LAW_CF, lay, rng, sb, units=units)
    ss = tc.make_ss(v, lay)
    X, _ = tc.R.nw.enlarged_windows(ss, tc.CFG["t_ss_ms"])
    m = X.mean(axis=1) + 60.0                           # raw window means, mV (the -60 mV offset of make_ss removed)
    c = tc.window_centres(lay)
    t_bar = 0.5 * (tc.L_SWEEP - 1) * tc.DT
    return lay, m, c, t_bar


def plot(lay, m, c, t_bar):
    plt = fs.setup()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.9), sharey=True)
    for s, ax in enumerate(axes):
        k = np.flatnonzero(lay.sweep == s)
        dep = lay.pos[k] % 2 == 0
        tt = np.linspace(0, tc.L_SWEEP * tc.DT, 50)
        ax.plot(tt, CFG["b"][s] * (tt - t_bar), color=fs.GREY, lw=1.5, ls="--",
                label=r"tilt line $b_s(t-\bar t_s)$, $b_s$ = %+.2f mV/s" % CFG["b"][s])
        ax.plot(c[k][dep], m[k][dep], "o", ms=6, color=fs.SERIES[0], label="dep. pulse windows")
        ax.plot(c[k][~dep], m[k][~dep], "s", ms=6, mfc="none", mec=fs.SERIES[1], mew=1.5, label="hyp. pulse windows")
        posmap = {int(p): int(j) for j, p in zip(k, lay.pos[k])}
        for p0, p1 in CFG["pairs_shown"]:
            if p0 in posmap and p1 in posmap:
                j0, j1 = posmap[p0], posmap[p1]
                ax.annotate("", xy=(c[j1], m[j1]), xytext=(c[j0], m[j0]),
                            arrowprops=dict(arrowstyle="-|>", color=fs.INK2, lw=1.0, shrinkA=4, shrinkB=4))
        ax.set_title("sweep %d: %d of 37 windows stored" % (s + 1, k.size))
        ax.set_xlabel(r"window centre $t$ from the sweep start (s)")
        ax.set_xlim(0, 7.6)
    axes[0].set_ylabel(r"window mean $\bar x_k$ (mV)")
    axes[0].legend(loc="upper left", fontsize=8.5)
    axes[1].legend(loc="upper right", fontsize=8.5)
    fig.text(0.5, -0.03, r"arrows: same-sweep, same-polarity pairs 4 positions apart, $h^*$ = 4 x 200.5 ms = 0.802 s;"
             r" each differs by $b_s h^*$ plus the stationary noise of two 120 ms window means", ha="center", fontsize=8.8,
             color=fs.INK2)
    fig.tight_layout()
    CFG["out"].parent.mkdir(exist_ok=True)
    fig.savefig(CFG["out"], dpi=150, bbox_inches="tight")
    print("wrote", CFG["out"])


if __name__ == "__main__":
    plot(*data())
