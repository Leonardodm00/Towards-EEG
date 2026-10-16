#!/usr/bin/env python3
"""Figures for the three-shape-parameter check (2026-10-15). Plotting only:
reads identifiability_sgt.json, example_sgt.npz and the Student run's
identifiability.json; computes nothing scientific.

fig5_sgt_impostors.png: (a, b) Delta chi^2_min(d | theta_true, 1 gl) of the best
impostor against d, thin and thick truths, one curve per true kernel shape,
the Student-only curves of the previous run dashed; (c) the thin Gaussian
truth and its impostor at half the diameter; (d) impostor minus truth on the
53 offsets, thin (d = 0.5 d_true) and thick (d = 0.8 d_true).
fig6_sgt_mc.png: d-hat over the noise draws for each fit and truth; hollow:
mu stopped at its upper bound (20 /um).

    python fig_sgt.py --dir <dir> --student <dir>/identifiability.json
Colours: the reference categorical order (blue, orange, aqua, yellow,
magenta) for G, L, B4, S4, K, with a distinct marker each. Labels use the
equation image's names: beta (code p), xi (code r), zeta (code gamma). Pure ASCII.
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
         band="#ecebe6")
ORDER = ("G", "L", "B4", "S4", "K")
COL = dict(G="#2a78d6", L="#eb6834", B4="#1baf7a", S4="#eda100", K="#e87ba4")
MRK = dict(G="o", L="s", B4="^", S4="D", K="v")
LAB = dict(G=r"Gaussian ($\beta$ 2, $\xi$ 0)", L=r"Laplace ($\beta$ 1, $\xi$ 0)", B4=r"flat-topped ($\beta$ 4, $\xi$ 0)",
           S4=r"Student $\nu$ 4 ($\beta$ 2, $\xi$ 0.5)", K=r"skewed Gaussian ($\zeta$ 1.5)")
YLIM_MC = {0.6: (0.0, 1.1), 1.5: (1.40, 1.70)}


def _style(ax):
    ax.set_facecolor(S["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(S["axis"])
    ax.tick_params(colors=S["ink2"], labelsize=8.5)
    ax.grid(True, color=S["grid"], lw=0.6)
    ax.set_axisbelow(True)


def _valley(ax, R, Sd, d0, tag, legend_loc, legend_anchor):
    _style(ax)
    ax.axvspan(0.8 * d0, 1.2 * d0, color=S["band"], lw=0, zorder=0)
    for k in ORDER:
        t = [x for x in R["impostors"] if abs(x["d_true"] - d0) < 1e-9 and x["kernel"] == k][0]
        x = np.array([r["d"] for r in t["rows"]])
        y = np.array([r["dchi2"]["1"] for r in t["rows"]])
        o = np.argsort(x)
        for sel, lab in ((x[o] < d0, "truth: " + LAB[k]), (x[o] > d0, None)):
            ax.plot(x[o][sel], y[o][sel], "-", marker=MRK[k], ms=4.5, lw=1.6, color=COL[k], label=lab)
    for nu, k in ((100.0, "G"), (4.0, "S4")):
        st = [s for s in Sd["impostors"] if abs(s["d_true"] - d0) < 1e-9 and s["nu_true"] == nu][0]
        x = np.array([r["d"] for r in st["rows"]])
        y = np.array([r["dchi2"]["1"] for r in st["rows"]])
        o = np.argsort(x)
        for sel in (x[o] < d0, x[o] > d0):
            ax.plot(x[o][sel], y[o][sel], "--", lw=1.1, color=COL[k], alpha=0.85)
    ax.plot([], [], "--", lw=1.1, color=S["muted"],
            label="dashed: Student kernel only (previous run)")
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
    ax.set_xlabel(r"impostor's diameter $d$ ($\mu$m); shaded: $|d/d_{\rm true} - 1| < 0.2$", fontsize=9,
                  color=S["ink2"])
    ax.set_ylabel(r"$\Delta\chi^2_{\min}(d \mid \theta_{\rm true}, \sigma_\varepsilon = 1\ \mathrm{gl})$", fontsize=9,
                  color=S["ink2"])
    ax.set_title(tag, fontsize=10, loc="left", color=S["ink"])
    ax.legend(fontsize=7.0, frameon=False, loc=legend_loc, bbox_to_anchor=legend_anchor)


def fig5(R, Sd, E, path):
    fig, axs = plt.subplots(2, 2, figsize=(12.0, 9.2), facecolor=S["surface"])
    _valley(axs[0, 0], R, Sd, 0.6, r"(a) thin truths: $d_{\rm true}$ = 0.6 $\mu$m, $w_{\rm true}$ = 0.5 $\mu$m",
            "upper left", (0.0, 1.0))
    _valley(axs[0, 1], R, Sd, 1.5, r"(b) thick truths: $d_{\rm true}$ = 1.5 $\mu$m, $w_{\rm true}$ = 0.5 $\mu$m",
            "lower left", (0.0, 0.2))
    vv, m = E["vv"], E["thin_meta"]     # [d_true, d, mu, v0, w, p, r, gamma, dchi2_1, max_abs, rel]
    ax = axs[1, 0]
    _style(ax)
    ax.plot(vv, E["thin_true"], color=S["ink"], lw=2.0,
            label=r"truth: $d$ = %.2f $\mu$m, $\mu$ = 1.00 /$\mu$m, Gaussian $w$ = 0.50 $\mu$m" % m[0])
    ax.plot(vv, E["thin_imp"], color=COL["G"], lw=1.8, ls="--",
            label=r"impostor: $d$ = %.2f $\mu$m, $\mu$ = %.2f /$\mu$m, $w$ = %.2f $\mu$m, $\beta$ = %.2f, $\xi$ = %.2f"
                  % (m[1], m[2], m[4], m[5], m[6]))
    ax.set_ylim(97.0, 184.0)
    ax.set_xlabel(r"$v$ ($\mu$m)", fontsize=9, color=S["ink2"])
    ax.set_ylabel(r"$I_{\rm model}(v)$ (gl), noise-free", fontsize=9, color=S["ink2"])
    ax.set_title("(c) a tube half as thick, the same shadow", fontsize=10, loc="left", color=S["ink"])
    ax.legend(fontsize=7.6, frameon=False, loc="lower center")
    ax = axs[1, 1]
    _style(ax)
    ax.axhspan(-1.0, 1.0, color=S["band"], lw=0, zorder=0)
    ax.text(3.0, 1.15, r"$\pm$1 gl: the noise SD $\sigma_\varepsilon$ = 1 gl", ha="right", fontsize=8, color=S["ink2"])
    for tag, col in (("thin", COL["G"]), ("thick", S["ink"])):
        mm = E[tag + "_meta"]
        dch = ("%.3f" % mm[8]) if mm[8] < 10 else ("%.0f" % mm[8])
        ax.plot(E["v_samples"], E[tag + "_diff"], "o-", ms=3.5, lw=1.2, color=col,
                label=r"%s: $d_{\rm true}$ = %.1f $\to$ $d$ = %.2f $\mu$m; $\Delta\chi^2$ = %s; max$|\Delta|$ = %.2f gl"
                      % (tag, mm[0], mm[1], dch, mm[9]))
    ax.axhline(0.0, color=S["axis"], lw=1.0)
    ax.set_ylim(-12.0, 16.5)
    ax.set_xlabel(r"$v_n$ ($\mu$m), Block 5's 53 offsets", fontsize=9, color=S["ink2"])
    ax.set_ylabel(r"$I_{\rm model}(v_n \mid$ impostor$) - I_{\rm model}(v_n \mid \theta_{\rm true})$ (gl)", fontsize=9,
                  color=S["ink2"])
    ax.set_title("(d) impostor minus truth: thin at half the diameter, thick at 0.8", fontsize=10, loc="left",
                 color=S["ink"])
    ax.legend(fontsize=7.6, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=S["surface"])
    plt.close(fig)


def fig6(R, path):
    keys = (("gauss_w", "Gaussian,\nw fitted", COL["G"], "o"),
            ("sgt", "3-shape family,\nall fitted", S["ink"], "o"),
            ("sgt_l050", "3-shape family,\n" + r"$\lambda$ = 0.5 $\mu$m", S["ink"], "s"))
    mcs = R["mc"]
    fig, axs = plt.subplots(1, len(mcs), figsize=(3.3 * len(mcs), 4.8), facecolor=S["surface"])
    rng = np.random.default_rng(0)                      # horizontal jitter only (display)
    for ax, m in zip(axs, mcs):
        _style(ax)
        for i, (k, lab, col, mk) in enumerate(keys):
            d = np.array([r["d"] for r in m["raw"][k]], dtype=float)
            mu_hi = np.array(["mu_hi" in r["at_bound"] for r in m["raw"][k]])
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
        ax.set_title(r"truth $d_{\rm true}$ = %.1f $\mu$m, %s" % (m["d_true"], LAB[m["kernel"]].split(" (")[0]),
                     fontsize=9.5, loc="left", color=S["ink"])
    n = len(mcs[0]["raw"]["sgt"])
    fig.suptitle(r"$\hat d$ over %d noise draws ($\sigma_\varepsilon$ = 2 gl), truth $w_{\rm true}$ = 0.5 $\mu$m; "
                 r"bar: median; hollow: $\mu$ stopped at its upper bound (20 /$\mu$m); dash-dot: $d_{\rm true}$" % n,
                 fontsize=9.5, color=S["ink2"], x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=160, facecolor=S["surface"])
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--student", required=True)
    a = ap.parse_args()
    R = json.load(open(os.path.join(a.dir, "identifiability_sgt.json")))
    Sd = json.load(open(a.student))
    E = dict(np.load(os.path.join(a.dir, "example_sgt.npz")))
    fig5(R, Sd, E, os.path.join(a.dir, "fig5_sgt_impostors.png"))
    if "mc" in R:
        fig6(R, os.path.join(a.dir, "fig6_sgt_mc.png"))
    print("wrote figures")


if __name__ == "__main__":
    main()
