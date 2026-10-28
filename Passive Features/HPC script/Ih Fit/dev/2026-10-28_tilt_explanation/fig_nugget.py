"""fig_nugget.py -- the picture for the brief chat answer on the nugget
(2026-10-28): independent white noise of variance eta2 lifts a semivariogram by
the same constant at every lag h > 0 and leaves its shape unchanged.
Standalone figure script, not pipeline code.

Lines: gamma(h) from the formulas (analytic level), h > 0, gamma(0) = 0:
  stationary OU      gamma_s(h) = sigma2 (1 - exp(-h / tau))
  random walk        gamma_s(h) = q h / 2
  random slope       gamma_s(h) = s^2 h^2 / 2
  with noise         gamma(h)   = eta2 + gamma_s(h)
Dots: gamma_hat(h) = half the mean squared difference at lag h, from simulated
records of each process plus the noise (computed level). Check printed: every
dot within 3 Monte Carlo SE of its formula (prediction fixed before running).
Layers: CFG (parameters) | simulate (data) | estimate (analysis) | plot.
Seed 20261039; numpy, matplotlib. Run:  python fig_nugget.py  ->  fig_nugget.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CFG = dict(
    seed=20261039, eta2=0.25, sigma2=1.0, tau=20.0, q=0.04, s=0.02, h_max=100,
    lags=(1, 5, 10, 20, 40, 60, 80, 100),
    n_ou=(400, 2000), n_rw=(1000, 1000), n_sl=(4000, 400),
    colors=dict(base="#2a78d6", nug="#eb6834"),           # validated: validate_palette.js, light, all PASS
    ink=dict(primary="#0b0b0b", secondary="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
             surface="#fcfcfb"),
    out=Path(__file__).resolve().parent / "fig_nugget.png",
)


# ---------------------------------------------------------------- simulation (data)
def simulate(kind, rng):
    if kind == "ou":
        n_rec, n_len = CFG["n_ou"]
        rho = np.exp(-1.0 / CFG["tau"])
        x = np.empty((n_rec, n_len))
        x[:, 0] = np.sqrt(CFG["sigma2"]) * rng.standard_normal(n_rec)
        e = np.sqrt(CFG["sigma2"] * (1 - rho ** 2)) * rng.standard_normal((n_rec, n_len))
        for j in range(1, n_len):
            x[:, j] = rho * x[:, j - 1] + e[:, j]
    elif kind == "rw":
        n_rec, n_len = CFG["n_rw"]
        x = np.cumsum(np.sqrt(CFG["q"]) * rng.standard_normal((n_rec, n_len)), axis=1)
    else:
        n_rec, n_len = CFG["n_sl"]
        t = np.arange(n_len)[None, :] - (n_len - 1) / 2
        x = CFG["s"] * rng.standard_normal((n_rec, 1)) * t
    return x + np.sqrt(CFG["eta2"]) * rng.standard_normal(x.shape)


# ---------------------------------------------------------------- formulas and estimator (analysis)
def gamma_s(kind, h):
    h = np.asarray(h, dtype=float)
    if kind == "ou":
        return CFG["sigma2"] * (1 - np.exp(-h / CFG["tau"]))
    if kind == "rw":
        return CFG["q"] * h / 2
    return CFG["s"] ** 2 * h ** 2 / 2


def gamma_hat(x, h):
    per = 0.5 * np.mean((x[:, h:] - x[:, :-h]) ** 2, axis=1)
    return per.mean(), per.std(ddof=1) / np.sqrt(per.size)


# ---------------------------------------------------------------- plot
def plot(results):
    ink, col = CFG["ink"], CFG["colors"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "axes.edgecolor": ink["axis"],
                         "axes.labelcolor": ink["secondary"], "xtick.color": ink["muted"], "ytick.color": ink["muted"]})
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6), sharey=True, facecolor=ink["surface"])
    titles = {"ou": ("Stationary OU: plateau", r"$\gamma_s(h)=\sigma^2(1-e^{-h/\tau})$,  $\sigma^2$ = 1, $\tau$ = 20"),
              "rw": ("Random walk: grows $\\propto h$", r"$\gamma_s(h)=qh/2$,  $q$ = 0.04 per sample"),
              "sl": ("Random slope: grows $\\propto h^2$", r"$\gamma_s(h)=s^2h^2/2$,  $s$ = 0.02 per sample")}
    hh = np.linspace(1e-3, CFG["h_max"], 400)
    eta2 = CFG["eta2"]
    for ax, kind in zip(axes, ("ou", "rw", "sl")):
        ax.set_facecolor(ink["surface"])
        ax.grid(True, color=ink["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        g = gamma_s(kind, hh)
        ax.plot(hh, g, color=col["base"], lw=1.8, solid_capstyle="round", label="without noise: $\\gamma_s(h)$")
        ax.plot(hh, eta2 + g, color=col["nug"], lw=1.8, solid_capstyle="round",
                label="with white noise: $\\eta^2+\\gamma_s(h)$")
        # the origin: gamma(0) = 0 for both; the noisy curve starts at eta2 just after it (open circle)
        ax.plot([0], [0], "o", ms=7, color=ink["secondary"], mec=ink["surface"], mew=1.5, zorder=5)
        ax.plot([0.0], [eta2], "o", ms=7, mfc=ink["surface"], mec=col["nug"], mew=1.8, zorder=5)
        ax.annotate("", xy=(3.0, eta2), xytext=(3.0, 0.0),
                    arrowprops=dict(arrowstyle="<->", color=ink["secondary"], lw=1.0, shrinkA=0, shrinkB=0))
        # label in the free lower-right area, joined to the step by a thin leader
        ax.annotate("nugget $\\eta^2$: a step at $h\\to0^+$", xy=(3.6, eta2 / 2), xytext=(38.0, eta2 / 2),
                    va="center", ha="left", fontsize=9, color=ink["primary"],
                    arrowprops=dict(arrowstyle="-", color=ink["muted"], lw=0.8, shrinkA=2, shrinkB=1))
        # the same offset further out, labelled just under the lower curve
        hx = 50.0 if kind == "ou" else 75.0
        y0 = float(gamma_s(kind, hx))
        ax.annotate("", xy=(hx, y0 + eta2), xytext=(hx, y0),
                    arrowprops=dict(arrowstyle="<->", color=ink["secondary"], lw=1.0, shrinkA=0, shrinkB=0))
        ax.text(hx + 1.5, y0 - 0.05, "same $\\eta^2$", va="top", ha="left", fontsize=9, color=ink["primary"])
        # simulated dots on the noisy curve
        lags = np.array(CFG["lags"])
        m = np.array([results[kind][h][0] for h in lags])
        ax.plot(lags, m, "o", ms=6.5, color=col["nug"], mec=ink["surface"], mew=1.5, zorder=6,
                label="$\\hat\\gamma(h)$, simulated with noise")
        if kind == "ou":
            for level, txt in ((CFG["sigma2"], "sill $\\sigma^2$"), (CFG["sigma2"] + eta2, "sill $\\eta^2+\\sigma^2$")):
                ax.axhline(level, color=ink["muted"], lw=0.9, ls=(0, (1.5, 2.5)))
                ax.text(CFG["h_max"], level + 0.04, txt, ha="right", va="bottom", fontsize=9, color=ink["secondary"])
        title, sub = titles[kind]
        ax.set_title(title + "\n" + sub, fontsize=10.5, color=ink["primary"], loc="left")
        ax.set_xlim(-3, CFG["h_max"] + 3)
        ax.set_ylim(-0.08, 2.6)
        ax.set_xlabel("lag $h$ (samples)")
    axes[0].set_ylabel("semivariogram $\\gamma(h)$ (a.u.$^2$)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.02),
               labelcolor=ink["primary"])
    fig.text(0.01, -0.03, "Same white noise, $\\eta^2$ = 0.25 a.u.$^2$, independent of the process, in every panel. "
             "Lines: formulas (analytic level). Dots: $\\hat\\gamma(h)$ from simulated records (computed level). "
             "Filled grey dot: $\\gamma(0)=0$ for both curves.", fontsize=9, color=ink["secondary"], ha="left")
    fig.tight_layout(rect=(0, 0.0, 1, 0.93))
    fig.savefig(CFG["out"], dpi=150, bbox_inches="tight", facecolor=ink["surface"])
    print("wrote", CFG["out"])


def main():
    rng = np.random.default_rng(CFG["seed"])
    results, ok = {}, True
    for kind in ("ou", "rw", "sl"):
        x = simulate(kind, rng)
        results[kind] = {h: gamma_hat(x, h) for h in CFG["lags"]}
        for h in CFG["lags"]:
            m, se = results[kind][h]
            pred = CFG["eta2"] + float(gamma_s(kind, h))
            good = abs(m - pred) < 3 * se
            ok &= good
            print("%-2s h = %3d: gamma_hat %.4f +/- %.4f, formula %.4f, z = %+.2f %s"
                  % (kind, h, m, se, pred, (m - pred) / se, "" if good else "<- FAIL"))
    print("dots within 3 SE of their formulas:", "PASS" if ok else "FAIL")
    plot(results)


if __name__ == "__main__":
    main()
