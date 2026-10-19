"""fig02_weights_vs_theory.py -- Figure 2 of "The thin-plate spline in three dimensions" (2026-10-18; standalone).

Claim: at the centre of a uniform design, the weights l_n(u0) of SciPy's smoothing spline follow the
equivalent kernel of the penalty's order (eqs. (24)-(25)) -- order 5/2 for 'thin_plate_spline', order 2 for 'linear'
at degree 1 -- at the same half-power frequency. Plotted from check06_kernels.npz (run check06 first):
N = 3000 points uniform in [0, 1]^3, the query at the centre, lambda = A rho / nu_c^2m with nu_c = 15.
Drawn as radial weight densities: empirical sum of l_n over a distance bin / bin width; theory 4 pi rho^2 E(rho).
Only distances <= 0.5 are drawn: beyond them the bins leave the unit cube.
Run (same folder):  python3 fig02_weights_vs_theory.py   -> ../figures/fig02_weights_vs_theory.png
"""
import math

import numpy as np

from fig_style import PAL, footer, new, save

CFG = dict(npz="check06_kernels.npz", bin_w=0.02, r_max=0.5, n=3000, out="../figures/fig02_weights_vs_theory.png")


def main():
    D = np.load(CFG["npz"])
    r = D["r"]
    rc = D["r_curve"]
    fig, axes = new(1, 2, w=8.6, h=3.8)
    edges = np.arange(0.0, CFG["r_max"] + 1e-12, CFG["bin_w"])
    mids = 0.5 * (edges[:-1] + edges[1:])
    for ax, key, title in ((axes[0], "tps", "'thin_plate_spline' (order 5/2)"),
                           (axes[1], "lin", "'linear', degree 1 (order 2)")):
        l = D[key + "_emp"] / CFG["n"]
        dens = np.array([l[(r >= a) & (r < b)].sum() for a, b in zip(edges[:-1], edges[1:])]) / CFG["bin_w"]
        ax.bar(mids, dens, width=CFG["bin_w"] * 0.86, color=PAL[key], alpha=0.85, lw=0,
               label=r"SciPy: $\sum_{n\in\mathrm{bin}} \ell_n(u_0)$ / bin width")
        sel = rc <= CFG["r_max"]
        ax.plot(rc[sel], 4 * math.pi * rc[sel] ** 2 * D[key + "_curve"][sel], color=PAL["ink"], lw=1.6,
                label=r"theory: $4\pi\varrho^2\mathcal{E}(\varrho)$, eq. (25)")
        ax.axhline(0, color=PAL["ink2"], lw=0.8)
        neg = float(l[(r <= 0.5) & (l < 0)].sum())
        ax.text(0.98, 0.32, "negative weights\n($\\varrho\\leq0.5$): %.2f\n$\\lambda_s$ = %.3f"
                % (neg, float(D[key + "_lam"])), transform=ax.transAxes, ha="right", fontsize=7.5, color=PAL["ink2"])
        ax.set_title(title, loc="left", fontsize=9)
        ax.set_xlabel(r"distance $\varrho=\Vert u_n - u_0\Vert$ (unit-cube coordinates)")
        ax.legend(loc="upper right", fontsize=7.2)
    axes[0].set_ylabel("radial weight density (per unit distance)")
    lo = min(ax.get_ylim()[0] for ax in axes)
    hi = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(lo, hi)
    footer(fig, r"plotted (computed level): check06_equivalent_kernel.py, seed 20261026, $N$ = 3000 uniform in the unit cube, "
                r"$u_0$ its centre, matched half-power frequency $\nu_c$ = 15, bins of width 0.02; weights from SciPy's saddle-point system",
           y=-0.03)
    save(fig, CFG["out"])
    print("wrote", CFG["out"])


if __name__ == "__main__":
    main()
