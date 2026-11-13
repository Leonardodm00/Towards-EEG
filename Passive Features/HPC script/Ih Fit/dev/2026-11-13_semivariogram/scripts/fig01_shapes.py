"""fig01_shapes.py -- Figure 1 of TEEG_semivariogram_2026-11-13.md, PLOTTED: the
semivariogram's shape for four kinds of process, formula (line) against the
Matheron estimate on simulated records (dots, +/- 2 SE across records).

Processes (h in samples; the same parameters as check01_generic.py A-D):
  (a) white noise, eta2 = 0.25:                        gamma(h) = eta2 for h >= 1
  (b) OU, sigma2 = 1, tau = 20, plus that white noise: gamma(h) = eta2 + sigma2 (1 - exp(-h / tau))
  (c) random walk, q = 1 per sample:                   gamma(h) = q h / 2
  (d) random slope, SD sigma_b = 0.01 per sample, plus noise:  gamma(h) = eta2 + sigma_b^2 h^2 / 2
gamma(0) = 0 in every panel (open circle at the origin).

Layers: CFG (parameters) | simulate() (data) | estimate() (analysis) | plot().
Seed 20261115. numpy, matplotlib.
Run:  python fig01_shapes.py      ->  ../figures/fig01_shapes.png   (a few seconds)

Pure ASCII, LF.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import viz_style as vs

CFG = dict(
    seed=20261115, eta2=0.25, sigma2=1.0, tau=20.0, q=1.0, s=0.01,
    n_rec=200, n_len=2000, n_rec_slope=2000, n_len_slope=300,
    dots=(1, 5, 10, 20, 40, 60, 80, 100), h_max=100,
    out=Path(__file__).resolve().parent.parent / "figures" / "fig01_shapes.png",
)


# ---------------------------------------------------------------------------
# Data and analysis
# ---------------------------------------------------------------------------
def simulate(rng):
    c = CFG
    w = np.sqrt(c["eta2"]) * rng.standard_normal((c["n_rec"], c["n_len"]))
    rho = np.exp(-1.0 / c["tau"])
    ou = np.empty((c["n_rec"], c["n_len"]))
    ou[:, 0] = np.sqrt(c["sigma2"]) * rng.standard_normal(c["n_rec"])
    e = np.sqrt(c["sigma2"] * (1 - rho ** 2)) * rng.standard_normal((c["n_rec"], c["n_len"]))
    for j in range(1, c["n_len"]):
        ou[:, j] = rho * ou[:, j - 1] + e[:, j]
    rw = np.cumsum(np.sqrt(c["q"]) * rng.standard_normal((c["n_rec"], c["n_len"])), axis=1)
    t = np.arange(c["n_len_slope"]) - 0.5 * (c["n_len_slope"] - 1)
    b = c["s"] * rng.standard_normal((c["n_rec_slope"], 1))
    sl = b * t[None, :] + np.sqrt(c["eta2"]) * rng.standard_normal((c["n_rec_slope"], c["n_len_slope"]))
    return dict(white=w, ou=ou + w, rw=rw, slope=sl)


def estimate(x, lags):
    """Mean over records of the per-record Matheron estimate, and its SE."""
    m, se = [], []
    for h in lags:
        g = 0.5 * np.mean((x[:, h:] - x[:, :-h]) ** 2, axis=1)
        m.append(g.mean())
        se.append(g.std(ddof=1) / np.sqrt(g.size))
    return np.array(m), np.array(se)


def formulas(h):
    c = CFG
    hp = np.where(h > 0, 1.0, 0.0)
    return dict(
        white=c["eta2"] * hp,
        ou=hp * (c["eta2"] + c["sigma2"] * (1 - np.exp(-h / c["tau"]))),
        rw=c["q"] * h / 2.0,
        slope=hp * (c["eta2"] + 0.5 * c["s"] ** 2 * h ** 2),
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot(est):
    import matplotlib.pyplot as plt
    vs.apply()
    c = CFG
    col = vs.SERIES[0]
    h = np.arange(1, c["h_max"] + 1, dtype=float)
    f = formulas(h)
    fig, axes = plt.subplots(1, 4, figsize=(12.0, 3.3), constrained_layout=True)
    titles = dict(white="(a) white noise", ou="(b) OU + white noise", rw="(c) random walk",
                  slope="(d) random slope + noise")
    sub = dict(white=r"$\gamma(h)=\eta^2$: all nugget",
               ou=r"$\gamma(h)=\eta^2+\sigma^2(1-e^{-h/\tau})$",
               rw=r"$\gamma(h)=qh/2$: no sill",
               slope=r"$\gamma(h)=\eta^2+\frac{1}{2}\sigma_b^2h^2$")
    for ax, key in zip(axes, ("white", "ou", "rw", "slope")):
        ax.plot(h, f[key], color=col, lw=vs.LW, zorder=3)
        m, se = est[key]
        vs.ringed(ax, np.array(c["dots"]), m, 2 * se, col)
        ax.plot([0], [0], "o", ms=vs.MS, mfc=vs.SURFACE, mec=col, mew=1.3, zorder=5, clip_on=False)
        ax.set_title(titles[key], pad=16)
        ax.text(0.0, 1.015, sub[key], transform=ax.transAxes, fontsize=8.5, color=vs.TEXT2, va="bottom")
        ax.set_xlim(-2, c["h_max"] + 2)
        ax.set_xlabel(r"lag $h$ (samples)")
    axes[0].set_ylabel(r"semivariogram $\gamma(h)$ (a.u.$^2$)")
    for key, ax in zip(("white", "ou", "slope"), (axes[0], axes[1], axes[3])):
        ax.set_ylim(-0.03, 1.42 if key == "ou" else (0.42 if key == "white" else 0.9))
    axes[2].set_ylim(-1.5, 54)

    # anatomy of (b): nugget, sill, practical range
    ax = axes[1]
    sill = c["sigma2"] + c["eta2"]
    ax.axhline(sill, color=vs.TEXT2, lw=0.8, zorder=2)
    ax.text(c["h_max"], sill + 0.03, r"sill $C(0)=\sigma^2+\eta^2$", ha="right", va="bottom", fontsize=8, color=vs.TEXT2)
    ax.axvline(3 * c["tau"], color=vs.TEXT2, lw=0.8, zorder=2)
    ax.text(3 * c["tau"] + 1.5, 0.12, "practical range\n" + r"$3\tau$: 95 % of $\sigma^2$", fontsize=8,
            color=vs.TEXT2, va="bottom")
    ax.annotate("", xy=(4, c["eta2"]), xytext=(4, 0.0),
                arrowprops=dict(arrowstyle="<->", color=vs.TEXT2, lw=0.8, shrinkA=0, shrinkB=0))
    ax.text(6.5, 0.07, r"nugget $\eta^2$", fontsize=8, color=vs.TEXT2)
    fig.savefig(c["out"], dpi=170)
    return c["out"]


def main():
    rng = np.random.default_rng(CFG["seed"])
    data = simulate(rng)
    est = {k: estimate(v, CFG["dots"]) for k, v in data.items()}
    h = np.array(CFG["dots"], dtype=float)
    f = formulas(h)
    worst = 0.0
    for k in est:
        z = (est[k][0] - f[k]) / est[k][1]
        worst = max(worst, float(np.max(np.abs(z))))
        print("%-6s dots - formula, in SE: %s" % (k, np.array2string(z, precision=2)))
    print("all %d dots within 3 SE of their formulas: %s" % (4 * len(CFG["dots"]), "PASS" if worst < 3 else "FAIL"))
    print("wrote", plot(est))


if __name__ == "__main__":
    main()
