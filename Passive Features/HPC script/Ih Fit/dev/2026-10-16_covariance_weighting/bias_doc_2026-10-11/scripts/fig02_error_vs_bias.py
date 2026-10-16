"""Figure 2 (plotted): one window gives an error; the bias is the mean of the errors.

Errors t(x_j) - rho(1 ms | theta_sim) of T = rho_hat(1 ms) on M = 3200 simulated 265 ms LS pre-windows under
theta_sim (same windows as check01: seed 20261011, chunks of 200), with the bias (their mean), the 10-90 % band,
and the first three windows marked one by one.
Run:  python3 fig02_error_vs_bias.py      (about 5 s; writes ../figures/fig02_error_vs_bias.png and prints numbers)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from bias_common import DT, N_LS, LAG_1MS, THETA_SIM, true_acf, monte_carlo

CONFIG = dict(dt=DT, n=N_LS, lag=LAG_1MS, M=3200, seed=20261011, n_marked=3, bins=60,
              out="../figures/fig02_error_vs_bias.png", dpi=160)
COL = dict(blue="#2a78d6", orange="#eb6834", purple="#7b3fbf", ink="#0b0b0b", ink2="#52514e", band="#d9d6cf",
           surface="#fcfcfb")


def errors(cfg):
    S0, SL = monte_carlo(THETA_SIM, cfg["dt"], cfg["n"], cfg["lag"], cfg["M"], cfg["seed"])
    rho = true_acf(THETA_SIM, cfg["dt"], cfg["lag"])
    return SL / S0 - rho, rho


def plot(err, cfg):
    bias, se = err.mean(), err.std(ddof=1) / np.sqrt(err.size)
    q10, q90 = np.quantile(err, [0.1, 0.9])
    fig, ax = plt.subplots(figsize=(8.8, 4.5), dpi=cfg["dpi"], facecolor=COL["surface"])
    ax.set_facecolor(COL["surface"])
    ax.axvspan(q10, q90, color=COL["band"], alpha=0.55, linewidth=0, zorder=0)
    ax.hist(err, bins=cfg["bins"], density=True, color=COL["blue"], alpha=0.40, edgecolor=COL["surface"],
            linewidth=0.5, zorder=1)
    top = ax.get_ylim()[1] * 1.32
    ax.set_ylim(0, top)
    ax.axvline(0.0, color=COL["orange"], linewidth=2.2, zorder=3)
    ax.text(0.006, top * 0.97, "zero error:\nt(x) = \u03c1(1 ms | \u03b8)", ha="left", va="top", fontsize=8.5,
            color=COL["orange"])
    ax.axvline(bias, color=COL["ink"], linewidth=1.8, linestyle="--", zorder=3)
    ax.text(bias - 0.006, top * 0.97, f"bias = mean error = {bias:+.3f}\n(MC SE {se:.3f}, M = {err.size})",
            ha="right", va="top", fontsize=8.5, color=COL["ink"])
    ax.text((q10 + q90) / 2, top * 0.70, f"10-90 % of single-window errors:\n[{q10:+.3f}, {q90:+.3f}]",
            ha="center", va="center", fontsize=8.5, color=COL["ink2"],
            bbox=dict(boxstyle="round,pad=0.25", facecolor=COL["surface"], edgecolor="none", alpha=0.9))
    for j in range(cfg["n_marked"]):
        e = err[j]
        ax.plot([e], [0], marker="D", markersize=8, color=COL["purple"], clip_on=False, zorder=5,
                markeredgecolor=COL["surface"], markeredgewidth=1.2)
        ax.annotate(f"window {j + 1}: {e:+.3f}", xy=(e, 0), xytext=(e, top * (0.16 + 0.11 * j)),
                    ha="center", va="bottom", fontsize=8.2, color=COL["purple"],
                    arrowprops=dict(arrowstyle="-", color=COL["purple"], lw=0.8))
    ax.set_xlabel("error of one window, t(x) \u2212 \u03c1(1 ms | $\\hat\\theta_{\\rm sim}$)   (dimensionless)", color=COL["ink"], fontsize=9.5)
    ax.set_ylabel("density over windows", color=COL["ink"], fontsize=9.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(COL["ink2"])
    ax.tick_params(colors=COL["ink2"], labelsize=8.5)
    ax.set_title(r"$T = \hat\rho(1\,{\rm ms})$ on 265 ms windows under $\hat\theta_{\rm sim}$: each window has its own error; the bias is their mean",
                 color=COL["ink"], fontsize=9.8, loc="left")
    fig.tight_layout()
    fig.savefig(cfg["out"], facecolor=COL["surface"])
    return bias, se, q10, q90


def main(cfg=CONFIG):
    err, rho = errors(cfg)
    bias, se, q10, q90 = plot(err, cfg)
    first = ", ".join(f"{e:+.3f}" for e in err[:cfg["n_marked"]])
    print(f"rho(1 ms|theta_sim) {rho:.4f}; bias {bias:+.4f} (SE {se:.4f}); 10-90 % [{q10:+.3f}, {q90:+.3f}]; "
          f"share above target {np.mean(err > 0):.3f}; first windows' errors {first}")


if __name__ == "__main__":
    main()
