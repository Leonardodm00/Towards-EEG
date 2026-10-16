"""Figure 2: spread and bias of T* in the block worlds against block length, with the true law.

Plotting only: reads ../results/boot_types.npz written by check_boot_types.py (run that first).
Run:  python fig02_block_length.py   -> ../figures/fig02_block_length.png
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results" / "boot_types.npz"
OUT = HERE.parent / "figures" / "fig02_block_length.png"
STYLE = {"mbb": ("C0", "o", "moving block (MBB)"),
         "cbb": ("C1", "s", "circular block (CBB)"),
         "sb": ("C2", "^", "stationary (SB), mean block b")}
QLO, QHI = 0.1, 0.9


def load(path):
    d = np.load(path)
    labels = [s.rsplit("_", 1) for s in d["labels"]]
    return d, [(s, int(b)) for s, b in labels]


def per_scheme(d, labels, key):
    """{scheme: (b_ms, median, q_lo, q_hi)} over windows, for 'sd' or 'bias'."""
    out = {}
    ms = 1e3 * float(d["dt"])
    for scheme in STYLE:
        idx = [j for j, (s, b) in enumerate(labels) if s == scheme]
        b = np.array([labels[j][1] for j in idx]) * ms
        vals = d["Tsd"][:, idx] if key == "sd" else d["Tm"][:, idx] - d["T_obs"][:, None]
        out[scheme] = (b, np.median(vals, 0), np.quantile(vals, QLO, 0), np.quantile(vals, QHI, 0))
    return out


def join_part(d, labels):
    """Median over windows of the cbb join term (1 - L/b)(S_L + W_L)/S_0 - T(x)."""
    L = int(d["lag"]); ms = 1e3 * float(d["dt"])
    bs = np.array([b for s, b in labels if s == "cbb"])
    vals = np.array([np.median(max(0.0, 1 - L / b) * (d["SL"] + d["WL"]) / d["S0"] - d["T_obs"]) for b in bs])
    return bs * ms, vals


def plot(d, labels):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 6.4))
    offs = {"mbb": 0.94, "cbb": 1.0, "sb": 1.06}          # small horizontal offsets for legibility
    for key, ax in (("sd", a1), ("bias", a2)):
        for scheme, (b, med, lo, hi) in per_scheme(d, labels, key).items():
            c, m, lab = STYLE[scheme]
            ax.errorbar(b * offs[scheme], med, yerr=[med - lo, hi - med], color=c, marker=m, ms=5,
                        capsize=3, lw=1.2, label=lab + " (median, 10-90 % over windows)")
    a1.axhline(float(d["true_sd_sim"]), color="k", lw=1.8, label=r"true SD$(T\mid\theta_{\rm sim},n)$, Monte Carlo")
    a1.axhline(float(d["true_sd_cf"]), color="k", lw=1.2, ls="--",
               label=r"parametric world at $\hat\theta=\hat\theta_{\rm cf}$")
    iid_sd = np.median(d["Tsd"][:, 0])
    a1.axhline(iid_sd, color="C3", lw=1.2, ls=":", label="iid world (single samples): %.4f" % iid_sd)
    a1.set_ylabel(r"bootstrap SD of $T^*$ (dimensionless)")
    a1.set_title(r"(a) spread: $\mathrm{SD}^*(T^*)$ vs block length")
    a1.set_ylim(0, 0.14)
    bj, vj = join_part(d, labels)
    a2.plot(bj, vj, color="C1", ls=":", lw=1.5, label=r"CBB join term alone, $(1-L/b)(S_L+W_L)/S_0-t(\mathbf{x})$")
    a2.axhline(float(d["true_bias_sim"]), color="k", lw=1.8,
               label=r"true bias $\mathbb{E}[T\mid\theta_{\rm sim},n]-g(\theta_{\rm sim})$")
    a2.axhline(float(d["true_bias_cf"]), color="k", lw=1.2, ls="--",
               label=r"parametric world at $\hat\theta=\hat\theta_{\rm cf}$")
    a2.set_ylabel(r"bootstrap bias estimate $\bar t^{\,*}-t(\mathbf{x})$ (dimensionless)")
    a2.set_title(r"(b) bias: $\bar t^{\,*}-t(\mathbf{x})$ vs block length")
    a2.set_ylim(-0.30, 0.02)
    iid_bias = np.median(d["Tm"][:, 0] - d["T_obs"])
    sb1 = np.median(d["Tm"][:, [j for j, (s, b) in enumerate(labels) if s == "sb"][0]] - d["T_obs"])
    a2.text(0.03, 0.87, "off scale: iid world %.2f;\n1 ms blocks %.2f (MBB, CBB), %.2f (SB)"
            % (iid_bias, np.median(d["Tm"][:, 1] - d["T_obs"]), sb1),
            transform=a2.transAxes, color="C3", fontsize=8)
    for ax in (a1, a2):
        ax.set_xscale("log")
        ax.set_xlabel(r"block length $b\,\Delta t$ (ms; log scale)")
        ax.axvline(39.74, color="0.6", lw=0.8)
        ax.text(41, ax.get_ylim()[1] - 0.06 * np.ptp(ax.get_ylim()), r"$\tau_s$ = 39.7 ms", color="0.4", fontsize=8)
        ax.grid(alpha=0.3, which="both")
    for ax in (a1, a2):
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False)
    fig.text(0.01, 0.028,
             "Plotted (computed level). Windows of n = 13 250 samples (265 ms at 50 kHz) drawn from the Stage-7 law at "
             r"$\theta_{\rm sim}$; $T=\hat\rho(1$ ms$)$, lag L = 50 samples; 40 windows, B = 500 replicates each; "
             "true law from 20 000 fresh windows (seed 2026101301).", fontsize=7)
    fig.text(0.01, 0.008,
             r"Parametric world: $\hat\theta$ held at a cohort fit, not refitted per window, so a per-cell plug-in error "
             "is not included. Scripts: check_boot_types.py (run first), fig02_block_length.py.", fontsize=7)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=160)
    print("saved", OUT)


if __name__ == "__main__":
    plot(*load(RES))
