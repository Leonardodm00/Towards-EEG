#!/usr/bin/env python3
"""Figures for the identifiability check (2026-10-15). Plotting only: reads
identifiability.json and example.npz, computes nothing scientific.

fig3_impostors.png: (a, b) Delta chi^2_min(d | theta_true, 1 gl) of the best
impostor against its diameter d, for a thin and a thick truth, one curve per
true nu; (c) the thin truth and its impostor at d = 0.8 d_true; (d) impostor
minus truth on Block 5's 53 offsets, thin and thick, against the noise SD.
fig4_student_mc.png: d-hat over the noise draws, per fit and truth; hollow
markers: mu stopped at its upper bound (20 /um).

    python fig_identifiability.py --dir <dir>
Colours: the Cell 4l/4m family colours (Gaussian #2a78d6, Student #eda100,
Lorentzian #eb6834) for the true nu (100 ~ Gaussian end, 4 Student, 1.5 ~
Lorentzian end); ink for the reference. Pure ASCII.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.transforms as mtransforms  # noqa: E402
import numpy as np  # noqa: E402

S = dict(surface="#fcfcfb", ink="#1a1a19", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
         band="#ecebe6", nu={100.0: "#2a78d6", 4.0: "#eda100", 1.5: "#eb6834"}, gauss="#2a78d6", student="#eda100")
NU_LABEL = {100.0: " (Gaussian end)", 4.0: "", 1.5: " (near the Lorentzian)"}
YLIM_MC = {0.6: (0.15, 1.0), 1.5: (1.44, 1.67)}    # shared within the thin and the thick pair


def _style(ax):
    ax.set_facecolor(S["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(S["axis"])
    ax.tick_params(colors=S["ink2"], labelsize=8.5)
    ax.grid(True, color=S["grid"], lw=0.6)
    ax.set_axisbelow(True)


def _valley(ax, R, d0, tag, legend_anchor):
    _style(ax)
    ax.axvspan(0.8 * d0, 1.2 * d0, color=S["band"], lw=0, zorder=0)
    for t in R["impostors"]:
        if abs(t["d_true"] - d0) > 1e-9:
            continue
        x = np.array([r["d"] for r in t["rows"]])
        y = np.array([r["dchi2"]["1"] for r in t["rows"]])
        o = np.argsort(x)
        lo, hi = x[o] < d0, x[o] > d0                       # the truth itself (Delta chi^2 = 0) is not drawn
        col = S["nu"][t["nu_true"]]
        lab = r"truth $\nu_{\rm true}$ = %g" % t["nu_true"] + NU_LABEL[t["nu_true"]]
        ax.plot(x[o][lo], y[o][lo], "o-", ms=5, lw=1.8, color=col, label=lab)
        ax.plot(x[o][hi], y[o][hi], "o-", ms=5, lw=1.8, color=col)
    ax.set_yscale("log")
    ax.axhline(1.0, color=S["ink"], lw=1.0, ls="--")
    ax.axhline(4.0, color=S["muted"], lw=1.0, ls=":")
    xl = ax.get_xlim()
    ax.text(xl[1], 1.08, r"$\Delta\chi^2 = 1$ at $\sigma_\varepsilon$ = 1 gl", ha="right", va="bottom", fontsize=8,
            color=S["ink2"])
    ax.text(xl[1], 4.3, r"$\Delta\chi^2 = 1$ at $\sigma_\varepsilon$ = 2 gl", ha="right", va="bottom", fontsize=8,
            color=S["ink2"])
    ax.axvline(d0, color=S["muted"], lw=1.0, ls="-.")
    tr = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(d0, 0.985, r" truth $d_{\rm true}$", transform=tr, ha="left", va="top", fontsize=8, color=S["ink2"])
    ax.set_xlabel(r"impostor's diameter $d$ ($\mu$m); shaded: $|d/d_{\rm true} - 1| < 0.2$", fontsize=9, color=S["ink2"])
    ax.set_ylabel(r"$\Delta\chi^2_{\min}(d \mid \theta_{\rm true}, \sigma_\varepsilon = 1\ \mathrm{gl})$", fontsize=9,
                  color=S["ink2"])
    ax.set_title(tag, fontsize=10, loc="left", color=S["ink"])
    ax.legend(fontsize=7.8, frameon=False, loc="lower left", bbox_to_anchor=legend_anchor)


def fig3(R, E, path):
    fig, axs = plt.subplots(2, 2, figsize=(11.6, 8.6), facecolor=S["surface"])
    _valley(axs[0, 0], R, 0.6, r"(a) thin truth: $d_{\rm true}$ = 0.6 $\mu$m, $w_{\rm true}$ = 0.5 $\mu$m", (0.0, 0.0))
    _valley(axs[0, 1], R, 1.5, r"(b) thick truth: $d_{\rm true}$ = 1.5 $\mu$m, $w_{\rm true}$ = 0.5 $\mu$m", (0.0, 0.2))
    vv, m = E["vv"], E["thin_meta"]     # [d0, nu0, d, mu, v0, w, nu, dchi2_1, max_abs, rel]
    ax = axs[1, 0]
    _style(ax)
    ax.plot(vv, E["thin_true"], color=S["ink"], lw=2.0,
            label=r"truth: $d_{\rm true}$ = %.2f $\mu$m, $\mu_{\rm true}$ = 1.00 /$\mu$m, $w_{\rm true}$ = 0.50 $\mu$m, $\nu_{\rm true}$ = %g"
                  % (m[0], m[1]))
    ax.plot(vv, E["thin_imp"], color=S["gauss"], lw=1.8, ls="--",
            label=r"impostor: $d$ = %.2f $\mu$m, $\mu$ = %.2f /$\mu$m, $w$ = %.2f $\mu$m, $\nu$ = %.3g"
                  % (m[2], m[3], m[5], m[6]))
    ax.set_xlabel(r"$v$ ($\mu$m)", fontsize=9, color=S["ink2"])
    ax.set_ylabel(r"$I_{\rm model}(v)$ (gl), noise-free", fontsize=9, color=S["ink2"])
    ax.set_title("(c) two different tubes, one shadow", fontsize=10, loc="left", color=S["ink"])
    ax.set_ylim(97.0, 184.0)                            # room under the dip for the legend
    ax.legend(fontsize=7.8, frameon=False, loc="lower center")
    ax = axs[1, 1]
    _style(ax)
    ax.axhspan(-1.0, 1.0, color=S["band"], lw=0, zorder=0)
    ax.text(3.0, 1.15, r"$\pm$1 gl: the noise SD $\sigma_\varepsilon$ = 1 gl", ha="right", fontsize=8, color=S["ink2"])
    for tag, col in (("thin", S["gauss"]), ("thick", S["ink"])):
        mm = E[tag + "_meta"]
        ax.plot(E["v_samples"], E[tag + "_diff_samples"], "o-", ms=3.5, lw=1.2, color=col,
                label=r"%s: $d_{\rm true}$ = %.1f $\to$ $d$ = %.2f $\mu$m; $\Delta\chi^2$ = %s; max$|\Delta|$ = %.2f gl"
                      % (tag, mm[0], mm[2], ("%.2f" % mm[7]) if mm[7] < 10 else ("%.0f" % mm[7]), mm[8]))
    ax.axhline(0.0, color=S["axis"], lw=1.0)
    ax.set_ylim(-12.0, 16.5)
    ax.set_xlabel(r"$v_n$ ($\mu$m), Block 5's 53 offsets", fontsize=9, color=S["ink2"])
    ax.set_ylabel(r"$I_{\rm model}(v_n \mid$ impostor$) - I_{\rm model}(v_n \mid \theta_{\rm true})$ (gl)", fontsize=9,
                  color=S["ink2"])
    ax.set_title(r"(d) the same 20 % thinner impostor, thin against thick truth", fontsize=10, loc="left",
                 color=S["ink"])
    ax.legend(fontsize=7.8, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=S["surface"])
    plt.close(fig)


def fig4(R, path):
    keys = (("block5", "Block 5\n(Gaussian,\nw fixed)", S["ink"], "o"),
            ("gauss_w", "Gaussian,\nw fitted", S["gauss"], "o"),
            ("student", "Student,\n" + r"w, $\nu$ fitted", S["student"], "o"),
            ("student_l050", "Student,\n" + r"$\lambda$ = 0.5 $\mu$m", S["student"], "^"),
            ("student_l025", "Student,\n" + r"$\lambda$ = 0.25 $\mu$m", S["student"], "s"))
    mcs = R["mc"]
    fig, axs = plt.subplots(1, len(mcs), figsize=(4.1 * len(mcs), 4.9), facecolor=S["surface"])
    rng = np.random.default_rng(0)                      # horizontal jitter only (display)
    for ax, m in zip(axs, mcs):
        _style(ax)
        for i, (k, lab, col, mk) in enumerate(keys):
            d = np.array(m["raw"][k]["d"], dtype=float)
            mu_hi = np.array(["mu_hi" in str(s).split(";") for s in m["raw"][k]["at_bound"]])
            x = i + rng.uniform(-0.18, 0.18, d.size)
            ax.plot(x[~mu_hi], d[~mu_hi], mk, ms=5, color=col, mec="white", mew=0.5)
            ax.plot(x[mu_hi], d[mu_hi], mk, ms=5, mfc="white", mec=col, mew=1.1)
            ax.plot([i - 0.3, i + 0.3], [np.median(d)] * 2, color=col, lw=2.2)
        ax.axhline(m["d_true"], color=S["muted"], lw=1.0, ls="-.")
        ax.set_ylim(*YLIM_MC[m["d_true"]])
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels([k[1] for k in keys], fontsize=7.4)
        ax.set_xlim(-0.6, len(keys) - 0.4)
        ax.set_ylabel(r"$\hat d$ ($\mu$m)", fontsize=9, color=S["ink2"])
        ax.set_title(r"truth $d_{\rm true}$ = %.1f $\mu$m, $\nu_{\rm true}$ = %g" % (m["d_true"], m["nu_true"]), fontsize=10,
                     loc="left", color=S["ink"])
    fig.suptitle(r"$\hat d$ over %d noise draws ($\sigma_\varepsilon$ = 2 gl), truth $w_{\rm true}$ = 0.5 $\mu$m; "
                 r"bar: median; hollow: $\mu$ stopped at its upper bound (20 /$\mu$m); dash-dot: $d_{\rm true}$; "
                 r"Block 5: the Gaussian at its fixed $w$ = 0.233 $\mu$m"
                 % len(mcs[0]["raw"]["block5"]["d"]), fontsize=9.5, color=S["ink2"], x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=160, facecolor=S["surface"])
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    R = json.load(open(os.path.join(a.dir, "identifiability.json")))
    E = dict(np.load(os.path.join(a.dir, "example.npz")))
    fig3(R, E, os.path.join(a.dir, "fig3_impostors.png"))
    fig4(R, os.path.join(a.dir, "fig4_student_mc.png"))
    print("wrote fig3_impostors.png and fig4_student_mc.png")


if __name__ == "__main__":
    main()
