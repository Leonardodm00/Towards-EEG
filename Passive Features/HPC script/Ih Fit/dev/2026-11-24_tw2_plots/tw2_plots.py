"""Figures of the Stage-7 run tw2 (and tw1 for comparison). Usage: python3 -I plots.py <tw2 dir> <tw1 dir> <out dir>"""
import sys, glob, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, FixedLocator, FuncFormatter

tw2, tw1, out = sys.argv[1:4]
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False})

man = pd.read_csv(os.path.join(tw2, "manifest.csv")).set_index("specimen_id")
fit = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(tw2, "fits", "twin_*", "phase2_results.csv")))]).set_index("specimen_id")
ext = pd.read_csv(os.path.join(tw2, "recovery_extras.csv")).set_index("twin_id")
cfg = pd.read_json(os.path.join(tw2, "run_config.json"), typ="series")
base = float(cfg["tau_h_pk"]["base_ms"]); kb = cfg["tau_h_pk"]["bounds_ms"]
full = [t for t in man.index if man.loc[t, "configuration"] == "full"]
clean = [t for t in man.index if man.loc[t, "configuration"] == "full_no_noise"]

# ---- Figure 1: truth vs estimate, the five fitted parameters -------------------------------------------
P = [("Cm", "cm_true", r"$C_m$ ($\mu$F/cm$^2$)", (0.3, 3.0), 1.0, np.log(1.25)),
     ("Rm", "rm_true", r"$R_m$ ($\Omega\,$cm$^2$)", (1e3, 1e5), 1.0, None),
     ("Ra", "ra_true", r"$R_a$ ($\Omega\,$cm)", (50, 1000), 1.0, None),
     ("gbar", "ih_gihbar_S_cm2", r"$\bar g_h$ (S/cm$^2$)", (1e-6, 1e-2), 1.0, np.log(1.25)),
     ("kappa_tau", "ih_kappa_tau", r"peak $\tau_h$ (ms)", tuple(kb), base, np.log(1.5))]
fig, axs = plt.subplots(1, 5, figsize=(17, 3.9))
for ax, (est, tru, lab, box, scale, tol) in zip(axs, P):
    lo, hi = box
    xs = np.geomspace(lo, hi, 50)
    if tol is not None:
        ax.fill_between(xs, xs * np.exp(-tol), xs * np.exp(tol), color=GRID, lw=0, zorder=0, label="gate tolerance")
    ax.plot(xs, xs, color=MUTED, lw=1, zorder=1)
    for ids, col, mk, name in ((full, BLUE, "o", "with noise (10)"), (clean, ORANGE, "s", "noise-free (5)")):
        ax.scatter(man.loc[ids, tru] * scale, fit.loc[ids, est] * scale, s=46, color=col, marker=mk, edgecolor=SURF,
                   linewidth=1.2, zorder=3, label=name)
    for t in clean:                                   # pair lines: the same truth with and without noise
        p = int(man.loc[t, "paired_with"])
        ax.plot([man.loc[t, tru] * scale] * 2, [fit.loc[p, est] * scale, fit.loc[t, est] * scale], color=MUTED, lw=0.7, zorder=2)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo * 0.9, hi * 1.1); ax.set_ylim(lo * 0.9, hi * 1.1)
    ax.axhline(lo, color=MUTED, lw=0.6, ls=":"); ax.axhline(hi, color=MUTED, lw=0.6, ls=":")
    if est == "Cm":
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(FixedLocator([0.3, 0.5, 1, 2, 3])); axis.set_major_formatter(FuncFormatter(lambda v, _: "%g" % v))
    ax.xaxis.set_minor_formatter(NullFormatter()); ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_title(lab); ax.set_xlabel("true value"); ax.grid(color=GRID, lw=0.5, zorder=0)
axs[0].set_ylabel("fitted value")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.12))
fig.suptitle("Run tw2: fitted against true value, twins with $I_h$ (five-parameter fit). Diagonal = perfect recovery; "
             "dotted = search bounds; thin vertical lines join a noisy twin to its noise-free copy", y=1.04, fontsize=10, color=MUTED)
fig.savefig(os.path.join(out, "tw2_fig1_truth_vs_fit.png"), dpi=170, bbox_inches="tight"); plt.close(fig)

# ---- Figure 2: the loss at the truth and at the estimate, per twin -------------------------------------
ids = full + clean
fig, ax = plt.subplots(figsize=(11, 4.2))
x = np.arange(len(ids))
lt, le = ext.loc[ids, "loss_at_truth"].to_numpy(), ext.loc[ids, "loss_at_estimate"].to_numpy()
for i in range(len(ids)):
    ax.plot([x[i], x[i]], [lt[i], le[i]], color=MUTED, lw=1, zorder=1)
ax.scatter(x, lt, s=60, color=AQUA, marker="D", edgecolor=SURF, linewidth=1.2, zorder=3, label="loss at the true parameters")
ax.scatter(x, le, s=60, color=BLUE, marker="o", edgecolor=SURF, linewidth=1.2, zorder=3, label="loss at the fit's estimate")
ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels([str(t)[-3:] for t in ids])
ax.axvline(len(full) - 0.5, color=MUTED, lw=0.6, ls=":")
ymax = max(le.max(), lt.max()) * 1.6
ax.set_ylim(min(lt.min(), le.min()) / 1.6, ymax)
ax.text((len(full) - 1) / 2, ymax / 1.15, "with noise", ha="center", va="top", color=MUTED)
ax.text(len(full) + (len(clean) - 1) / 2, ymax / 1.15, "noise-free", ha="center", va="top", color=MUTED)
ax.set_xlabel("twin (last three digits of its id)"); ax.set_ylabel("training loss (relative RMSD)")
ax.set_title("Run tw2: the true parameters fit better than the fit's result in 14 of 15 twins")
ax.grid(axis="y", color=GRID, lw=0.5); ax.legend(loc="lower left")
fig.savefig(os.path.join(out, "tw2_fig2_loss_truth_vs_estimate.png"), dpi=170, bbox_inches="tight"); plt.close(fig)

# ---- Figure 3: median error per gated parameter, tw1 vs tw2 --------------------------------------------
g1 = pd.read_csv(os.path.join(tw1, "report", "gate_verdict.csv")).set_index("parameter")
g2 = pd.read_csv(os.path.join(tw2, "report", "gate_verdict.csv")).set_index("parameter")
g0 = pd.read_csv(os.path.join(tw2, "report_no_noise", "gate_verdict.csv")).set_index("parameter")
axes_ = [("Cm", r"$C_m$"), ("Rm", r"$R_m$"), ("Ra", r"$R_a$"), ("gbar", r"$\bar g_h$"), ("kappa_tau", r"peak $\tau_h$")]
fig, ax = plt.subplots(figsize=(9.5, 4.2))
w = 0.26
for k, (g, col, name) in enumerate(((g1, AQUA, "tw1: old loss, six parameters (10 twins)"), (g2, BLUE, "tw2: new loss, five parameters (10 twins)"),
                                    (g0, ORANGE, "tw2: the same, noise-free (5 twins)"))):
    v = [np.exp(float(g.loc[a, "median_abs_err"])) for a, _ in axes_]
    b = ax.bar(np.arange(len(axes_)) + (k - 1) * w, v, w * 0.92, color=col, label=name, zorder=3)
    for r, val in zip(b, v):
        ax.text(r.get_x() + r.get_width() / 2, val * 1.04, "x%.2f" % val if val < 3 else "x%.1f" % val, ha="center", va="bottom", fontsize=8, color=INK, zorder=6,
                bbox=dict(facecolor=SURF, edgecolor="none", pad=0.6))
for i, (a, _) in enumerate(axes_):
    tol = g2.loc[a, "tolerance"]
    if np.isfinite(tol):
        ax.plot([i - 1.5 * w, i + 1.5 * w], [np.exp(tol)] * 2, color=INK, lw=1.4, zorder=4)
        ax.text(i + 1.55 * w, np.exp(tol), " gate", va="center", fontsize=8, color=INK)
ax.set_yscale("log"); ax.set_ylim(1, 9); ax.set_yticks([1, 1.25, 1.5, 2, 3, 5, 8]); ax.set_yticklabels(["x1", "x1.25", "x1.5", "x2", "x3", "x5", "x8"])
ax.set_xticks(np.arange(len(axes_))); ax.set_xticklabels([l for _, l in axes_])
ax.yaxis.set_minor_formatter(NullFormatter())
ax.set_ylabel("median miss factor  exp(median |ln(fitted/true)|)")
ax.set_title("Median recovery error per parameter: run tw1 against run tw2")
ax.grid(axis="y", color=GRID, lw=0.5, zorder=0); ax.legend(loc="upper left", fontsize=9)
fig.savefig(os.path.join(out, "tw2_fig3_error_by_parameter.png"), dpi=170, bbox_inches="tight"); plt.close(fig)
print("figures written:", sorted(os.listdir(out)))
