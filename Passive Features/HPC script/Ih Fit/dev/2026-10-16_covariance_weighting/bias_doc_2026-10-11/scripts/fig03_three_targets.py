"""Figure 3 (plotted): one statistic, three roles; the histogram stays put, only the target moves.

Sampling distribution of T = rho_hat(1 ms) on M = 3200 simulated 265 ms LS pre-windows under theta_sim (same
windows as check01 and fig02), with the target of each role:
  (1) g(theta) = rho(1 ms | theta), topic doc eq. (6)             -> bias = E[T | theta, n] - rho(1 ms | theta)
  (2) E[T | theta, n] itself (Monte Carlo mean)                   -> bias 0 by definition
  (3) none on this axis: the fit's target is theta; it matches med(T | theta, n) (Monte Carlo median)
Run:  python3 fig03_three_targets.py      (about 5 s; writes ../figures/fig03_three_targets.png and prints numbers)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from bias_common import DT, N_LS, LAG_1MS, THETA_SIM, true_acf, monte_carlo

CONFIG = dict(dt=DT, n=N_LS, lag=LAG_1MS, M=3200, seed=20261011, bins=50,
              out="../figures/fig03_three_targets.png", dpi=160)
COL = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", ink="#0b0b0b", ink2="#52514e", surface="#fcfcfb")


def sample(cfg):
    S0, SL = monte_carlo(THETA_SIM, cfg["dt"], cfg["n"], cfg["lag"], cfg["M"], cfg["seed"])
    return SL / S0, true_acf(THETA_SIM, cfg["dt"], cfg["lag"])


def plot(T, rho, cfg):
    mean, med = T.mean(), np.median(T)
    fig, ax = plt.subplots(figsize=(8.8, 4.7), dpi=cfg["dpi"], facecolor=COL["surface"])
    ax.set_facecolor(COL["surface"])
    ax.hist(T, bins=cfg["bins"], density=True, color=COL["blue"], alpha=0.35, edgecolor=COL["surface"], linewidth=0.5)
    top = ax.get_ylim()[1] * 1.36
    ax.set_ylim(0, top)
    ax.axvline(rho, color=COL["orange"], linewidth=2.2)
    ax.text(rho + 0.006, top * 0.97, f"(1) target g(\u03b8) = \u03c1(1 ms | \u03b8) = {rho:.3f}\n     the law's autocorrelation",
            ha="left", va="top", fontsize=8.5, color=COL["ink"])
    ax.axvline(mean, color=COL["ink"], linestyle="--", linewidth=1.8)
    ax.text(mean - 0.006, top * 0.97, f"(2) target E[T | \u03b8, n] \u2248 {mean:.3f}\n     T's own mean",
            ha="right", va="top", fontsize=8.5, color=COL["ink"])
    yb = top * 0.78
    ax.annotate("", xy=(mean, yb), xytext=(rho, yb), arrowprops=dict(arrowstyle="->", color=COL["ink2"], lw=1.4))
    ax.text((rho + mean) / 2, yb + top * 0.015, f"bias in role 1 = E[T | \u03b8, n] \u2212 g(\u03b8) \u2248 {mean - rho:+.3f}\nbias in role 2 = 0",
            ha="center", va="bottom", fontsize=8.5, color=COL["ink2"],
            bbox=dict(boxstyle="round,pad=0.2", facecolor=COL["surface"], edgecolor="none"))
    ax.plot([med], [0], marker="D", markersize=9, color=COL["aqua"], clip_on=False, zorder=5,
            markeredgecolor=COL["surface"], markeredgewidth=1.5)
    ax.annotate(f"(3) no target on this axis: the fit's target is \u03b8;\n     it matches med(T | \u03b8, n) \u2248 {med:.3f}",
                xy=(med, top * 0.02), xytext=(0.12, top * 0.55), ha="left", va="center", fontsize=8.5,
                color=COL["ink"], arrowprops=dict(arrowstyle="->", color=COL["ink2"], lw=1.2))
    ax.set_xlabel(r"$T = \hat\rho(1\,{\rm ms})$ on one 265 ms window   (dimensionless)", color=COL["ink"], fontsize=9.5)
    ax.set_ylabel("density over windows", color=COL["ink"], fontsize=9.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(COL["ink2"])
    ax.tick_params(colors=COL["ink2"], labelsize=8.5)
    ax.set_title("One statistic, three roles under $\\hat\\theta_{\\rm sim}$: the histogram stays put, only the target moves",
                 color=COL["ink"], fontsize=9.8, loc="left")
    fig.tight_layout()
    fig.savefig(cfg["out"], facecolor=COL["surface"])
    return mean, med


def main(cfg=CONFIG):
    T, rho = sample(cfg)
    mean, med = plot(T, rho, cfg)
    print(f"M {T.size}; rho {rho:.4f}; mean T {mean:.4f}; median T {med:.4f}; role-1 bias {mean - rho:+.4f}")


if __name__ == "__main__":
    main()
