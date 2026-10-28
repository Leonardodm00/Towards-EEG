"""fig04_sampling.py -- PLOTTED from check03 (b): the cumulative distribution of
sigma_b_hat / sigma_b from (T1) at the true law (theta_cf, sigma_b = 0.1 mV/s,
t_SS = 80 ms) for the four reference layouts, against sqrt(chi2_Seff / S_eff),
the spread of the S_eff realised tilts alone.
Run:  python check03_sampling_distribution.py b  (first), then  python fig04_sampling.py
"""
from pathlib import Path

import numpy as np
from scipy.stats import chi2

import fig_style as fs

CFG = dict(res=Path(__file__).resolve().parent.parent / "results" / "check03_ratios.npz",
           layouts=(("2sw_K40", "2 sweeps, K = 40"), ("2sw_K63", "2 sweeps, K = 63"),
                    ("6sw_K199", "6 sweeps, K = 199"), ("10sw_K348", "10 sweeps, K = 348")),
           out=Path(__file__).resolve().parent.parent / "figures" / "fig04_sampling.png")


def plot():
    z = np.load(CFG["res"])
    plt = fs.setup()
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    x = np.linspace(0, 2.6, 400)
    for i, (key, lab) in enumerate(CFG["layouts"]):
        r = np.sort(z[key + "_ratio"])
        Se = float(z[key + "_Se"])
        F = np.arange(1, r.size + 1) / r.size
        ax.step(r, F, where="post", color=fs.SERIES[i], lw=2, label="%s (S_eff %.2f): simulated (T1)" % (lab, Se))
        ax.plot(x, chi2.cdf(x ** 2 * Se, Se), color=fs.SERIES[i], lw=1.2, ls=":")
    ax.plot([], [], color=fs.INK2, ls=":", lw=1.2, label=r"dotted: $\sqrt{\chi^2_{S_{\rm eff}}/S_{\rm eff}}$ (tilt draws alone)")
    ax.axvline(1.0, color=fs.INK2, lw=1.0)
    ax.set_xlabel(r"$\hat\sigma_b(c)\,/\,\sigma_b(c)$   (estimate clipped at 0)")
    ax.set_ylabel("cumulative share of simulated cells")
    ax.set_xlim(0, 2.6)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    CFG["out"].parent.mkdir(exist_ok=True)
    fig.savefig(CFG["out"], dpi=150, bbox_inches="tight")
    print("wrote", CFG["out"])


if __name__ == "__main__":
    plot()
