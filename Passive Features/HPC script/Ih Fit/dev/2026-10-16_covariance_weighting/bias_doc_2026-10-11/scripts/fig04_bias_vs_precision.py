"""Figure 4 (plotted, twin cases): more windows shrink the spread; longer windows shrink the offset.

(a) K windows of 265 ms averaged, K = 1, 6, 36: distribution of the K-window mean of T = rho_hat(1 ms).
    K = 1: the M = 3200 simulated windows of check01 themselves.  K = 6, 36: 20 000 K-window means, each averaging
    K values drawn at random (with replacement) from those 3200 (windows are independent, so this samples the
    distribution of the mean of K independent windows).
(b) One window (K = 1) of duration n*dt = 50, 132.5, 265, 530, 1060 ms: distribution of T, M = 3200 windows each
    (seed 20261011 for every duration; at 265 ms these are exactly check01's windows).
Both panels: theta_sim, lag 1 ms; orange line = the target rho(1 ms | theta_sim), the same for every row;
black tick = the row's Monte Carlo mean.  Also prints the closed-form ratio of expectations (topic doc eq. 10).
Run:  python3 fig04_bias_vs_precision.py      (about 40 s; writes ../figures/fig04_bias_vs_precision.png)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from bias_common import DT, N_LS, LAG_1MS, THETA_SIM, true_acf, monte_carlo, closed_form_acf

CONFIG = dict(dt=DT, n=N_LS, lag=LAG_1MS, M=3200, seed=20261011,
              K_list=(1, 6, 36), R=20_000, seed_resample=7,
              durations_ms=(50.0, 132.5, 265.0, 530.0, 1060.0),
              out="../figures/fig04_bias_vs_precision.png", dpi=160)
COL = dict(blue="#2a78d6", orange="#eb6834", ink="#0b0b0b", ink2="#52514e", surface="#fcfcfb")


# --- analysis -------------------------------------------------------------------------------------
def k_window_means(cfg):
    S0, SL = monte_carlo(THETA_SIM, cfg["dt"], cfg["n"], cfg["lag"], cfg["M"], cfg["seed"])
    T = SL / S0
    rng = np.random.default_rng(cfg["seed_resample"])
    means = {K: (T if K == 1 else rng.choice(T, size=(cfg["R"], K), replace=True).mean(axis=1)) for K in cfg["K_list"]}
    return means, T


def by_duration(cfg):
    out = {}
    for d in cfg["durations_ms"]:
        n = int(round(d * 1e-3 / cfg["dt"]))
        S0, SL = monte_carlo(THETA_SIM, cfg["dt"], n, cfg["lag"], cfg["M"], cfg["seed"])
        out[d] = (SL / S0, closed_form_acf(THETA_SIM, cfg["dt"], n, cfg["lag"]), n)
    return out


# --- plotting -------------------------------------------------------------------------------------
def violin_panel(ax, rows, labels, rho, title):
    pos = np.arange(len(rows))[::-1]
    parts = ax.violinplot(rows, positions=pos, orientation="horizontal", widths=0.8, showextrema=False)
    for b in parts["bodies"]:
        b.set_facecolor(COL["blue"])
        b.set_alpha(0.40)
        b.set_edgecolor("none")
    for p, r, lab in zip(pos, rows, labels):
        m, s = r.mean(), r.std(ddof=1)
        ax.plot([m, m], [p - 0.36, p + 0.36], color=COL["ink"], linewidth=2.0)
        ax.text(1.02, p, f"bias {m - rho:+.3f}\nSD {s:.3f}", ha="left", va="center", fontsize=8.0,
                color=COL["ink"], transform=ax.get_yaxis_transform())
    ax.axvline(rho, color=COL["orange"], linewidth=2.0)
    ax.set_yticks(pos)
    ax.set_yticklabels(labels, fontsize=8.6, color=COL["ink"])
    ax.set_xlim(-0.25, 1.0)
    ax.set_title(title, fontsize=9.6, color=COL["ink"], loc="left")
    ax.set_xlabel("value of the statistic   (dimensionless)", fontsize=9.0, color=COL["ink"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(COL["ink2"])
    ax.tick_params(colors=COL["ink2"], labelsize=8.3)


def plot(means, dur, rho, cfg):
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.6, 4.8), dpi=cfg["dpi"], facecolor=COL["surface"])
    for x in (a, b):
        x.set_facecolor(COL["surface"])
    violin_panel(a, [means[K] for K in cfg["K_list"]], [f"K = {K}" for K in cfg["K_list"]], rho,
                 "(a) more windows, each 265 ms: mean of T over K windows")
    violin_panel(b, [dur[d][0] for d in cfg["durations_ms"]], [f"{d:g} ms" for d in cfg["durations_ms"]], rho,
                 "(b) longer windows, K = 1: T on one window of n\u00b7\u0394t")
    fig.text(0.01, 0.015, "Orange line: the target \u03c1(1 ms | \u03b8) at \u03b8 = $\\hat\\theta_{\\rm sim}$, %.3f, the same for every row;  "
             "black tick: the row's mean.\n(a) K = 1: %d simulated windows; K = 6, 36: %d means resampled from them;  "
             "(b) %d simulated windows per row;  lag 1 ms." % (rho, cfg["M"], cfg["R"], cfg["M"]),
             fontsize=8.2, color=COL["ink2"], ha="left", va="bottom")
    fig.tight_layout(w_pad=6.0, rect=(0, 0.09, 1, 1))
    fig.savefig(cfg["out"], facecolor=COL["surface"])


def main(cfg=CONFIG):
    rho = true_acf(THETA_SIM, cfg["dt"], cfg["lag"])
    means, T = k_window_means(cfg)
    dur = by_duration(cfg)
    plot(means, dur, rho, cfg)
    print(f"rho(1 ms|theta_sim) = {rho:.4f};  SD(T) one 265 ms window = {T.std(ddof=1):.4f}")
    for K in cfg["K_list"]:
        r = means[K]
        print(f"(a) K = {K:2d}: mean {r.mean():.4f}  bias {r.mean() - rho:+.4f}  SD {r.std(ddof=1):.4f}"
              f"  (SD(T)/sqrt(K) = {T.std(ddof=1) / np.sqrt(K):.4f})")
    for d in cfg["durations_ms"]:
        Td, qcf, n = dur[d]
        se = Td.std(ddof=1) / np.sqrt(Td.size)
        print(f"(b) {d:6.1f} ms (n = {n:5d}): mean {Td.mean():.4f}  bias {Td.mean() - rho:+.4f} (MC SE {se:.4f})"
              f"  median {np.median(Td):.4f}  SD {Td.std(ddof=1):.4f}  closed form q_cf {qcf:.4f}"
              f" (q_cf - rho {qcf - rho:+.4f}; closed form's share of the bias {100 * (qcf - rho) / (Td.mean() - rho):.1f} %;"
              f" q_cf - mean {qcf - Td.mean():+.4f})")


if __name__ == "__main__":
    main()
