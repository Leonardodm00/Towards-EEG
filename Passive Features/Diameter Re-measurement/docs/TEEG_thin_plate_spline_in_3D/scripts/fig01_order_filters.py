"""fig01_order_filters.py -- Figure 1 of "The thin-plate spline in three dimensions" (2026-10-18; standalone).

Claim: the higher the order m of the penalty, the sharper the smoother's frequency cutoff and the larger
the negative ring of its weights. Plotted from formulas (no data): the continuum filter of eq. (24),
T(nu) = 1 / (1 + (|nu| / nu_c)^(2m)), at one half-power frequency nu_c, and its inverse Fourier transform on
R^3 (the equivalent kernel E of eq. (25)) written as the radial weight density w = 4 pi varrho^2 E(varrho)
(int_0^inf w dvarrho = T(0) = 1), in the scale-free variables |nu|/nu_c and nu_c varrho (varrho a distance).
The masses are integrated over nu_c varrho in [0, 60]; the panel shows [0, 14]. Orders: 2 ('linear', degree 1), 5/2 ('thin_plate_spline'), 3 (r^3).
The negative mass int min(w, 0) dvarrho is printed and put in the legend. Checked against SciPy's weights for the
orders 2 and 5/2 in check06_equivalent_kernel.py (Figure 2).
Run (same folder):  python3 fig01_order_filters.py   -> ../figures/fig01_order_filters.png
"""
import math

import numpy as np
from scipy import integrate

from fig_style import LS, PAL, footer, new, save

CFG = dict(orders=((2.0, "lin", "order 2\n('linear', degree 1)", "2"),
                   (2.5, "tps", "order 5/2\n('thin_plate_spline')", "5/2"),
                   (3.0, "cub", "order 3 ($\\varrho^3$)", "3")),
           nu=np.logspace(-1, 1, 400), s=np.linspace(0.01, 60.0, 6000), s_show=14.0, out="../figures/fig01_order_filters.png")


def k_tilde(s, m):
    f = lambda t: t / (1.0 + t ** (2 * m))
    return integrate.quad(f, 0, np.inf, weight="sin", wvar=s, limlst=200)[0] / (2 * math.pi ** 2 * s)


def main():
    fig, (a, b) = new(1, 2, w=8.6, h=3.9)
    for m, key, label, short in CFG["orders"]:
        a.plot(CFG["nu"], 1.0 / (1.0 + CFG["nu"] ** (2 * m)), color=PAL[key], ls=LS[key], lw=2, label=label)
        w = np.array([4 * math.pi * s * s * k_tilde(s, m) for s in CFG["s"]])
        neg = float(np.trapezoid(np.minimum(w, 0.0), CFG["s"]))
        tot = float(np.trapezoid(w, CFG["s"]))
        print("order %.1f: integral of w over [0, %.0f] = %.4f; negative mass %.4f" % (m, CFG["s"][-1], tot, neg))
        show = CFG["s"] <= CFG["s_show"]
        b.plot(CFG["s"][show], w[show], color=PAL[key], ls=LS[key], lw=2, label="order %s: negative mass %.2f" % (short, neg))
    a.set_xscale("log")
    a.set_xlabel(r"frequency $|\nu|\,/\,\nu_c$ (dimensionless)")
    a.set_ylabel(r"filter $T(\nu)$: fraction of a frequency kept")
    a.axhline(0.5, color=PAL["muted"], lw=0.8, ls=(0, (3, 3)))
    a.text(0.105, 0.53, r"half power at $\nu_c$", color=PAL["muted"], fontsize=7.5)
    a.set_title("A  the penalty as a low-pass filter", loc="left", fontsize=9)
    a.legend(loc="lower left", bbox_to_anchor=(0.0, 0.0), fontsize=6.8, title=r"kernels on $\mathbb{R}^3$",
             title_fontsize=6.8, handlelength=2.2)
    b.axhline(0.0, color=PAL["ink2"], lw=0.8)
    b.set_xlabel(r"distance from the point, $\nu_c\,\varrho$ (dimensionless)")
    b.set_ylabel(r"radial weight density $4\pi\varrho^2\mathcal{E}(\varrho)$, area $=1$")
    b.set_title(r"B  its weights on $\mathbb{R}^3$: positive core, negative ring", loc="left", fontsize=9)
    b.legend(loc="upper right", fontsize=7.0)
    footer(fig, r"plotted from formulas (no data): $T$ of eq. (24) at one half-power frequency $\nu_c$; $4\pi\varrho^2\mathcal{E}(\varrho)$, "
                r"$\mathcal{E}$ the inverse Fourier transform of $T$ on $\mathbb{R}^3$ (eq. (25)), by quadrature; masses over "
                r"$\nu_c\varrho \leq 60$. Script: scripts/fig01_order_filters.py",
           y=-0.03)
    save(fig, CFG["out"])
    print("wrote", CFG["out"])


if __name__ == "__main__":
    main()
