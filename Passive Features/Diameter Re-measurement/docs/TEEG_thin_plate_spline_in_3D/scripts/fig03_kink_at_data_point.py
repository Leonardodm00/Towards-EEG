"""fig03_kink_at_data_point.py -- Figure 3 of "The thin-plate spline in three dimensions" (2026-10-18; standalone).

Claim: through a data point, a 3-D 'thin_plate_spline' fit is smooth (C^1), while a 'linear' fit (degree 1,
the 3-D bending-energy spline) has a kink, at smoothing 0 and at the smoothing its CV picks. Plotted from
check05_slices.npz (run check05 first): N = 300 uniform points in [0, 1]^3, y = sin(3 u1) + u2^2 - 0.5 u3
+ 0.05 (standard normal noise); the slice u_* + t omega through the data point nearest the centre. The insets zoom on |t| <= 0.006.
Run (same folder):  python3 fig03_kink_at_data_point.py   -> ../figures/fig03_kink_at_data_point.png
"""
import numpy as np

from fig_style import PAL, footer, new, save

CFG = dict(npz="check05_slices.npz", zoom=0.006, out="../figures/fig03_kink_at_data_point.png")


def main():
    D = np.load(CFG["npz"])
    t = D["t"]
    fig, axes = new(1, 2, w=8.6, h=3.7)
    for ax, key, title in ((axes[0], "tps", "'thin_plate_spline': smooth through the point"),
                           (axes[1], "lin", "'linear', degree 1: a kink at the point")):
        for kind, ls, lab in (("interp", (0, (4, 2)), "smoothing 0"), ("cv", "-", "CV smoothing")):
            y = D["%s_%s" % (key, kind)]
            ax.plot(t, y, color=PAL[key], ls=ls, lw=1.8 if kind == "cv" else 1.3,
                    label=r"%s, $\lambda_s$ = %g" % (lab, float(D["%s_%s_lam" % (key, kind)])))
        ax.axvline(0, color=PAL["muted"], lw=0.8, ls=(0, (2, 2)))
        ax.set_title(title, loc="left", fontsize=9)
        ax.set_xlabel(r"position along the line, $t$ (unit-cube coordinates)")
        ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.52), fontsize=7.0)
        ins = ax.inset_axes([0.16, 0.07, 0.33, 0.34])
        ins.set_facecolor(PAL["surface"])
        sel = np.abs(t) <= CFG["zoom"]
        y = D["%s_cv" % key]
        ins.plot(t[sel], y[sel], color=PAL[key], lw=1.6)
        ins.axvline(0, color=PAL["muted"], lw=0.7, ls=(0, (2, 2)))
        ins.tick_params(labelsize=6)
        ins.set_title("zoom, CV smoothing", fontsize=6.5, color=PAL["ink2"])
        for s in ("top", "right"):
            ins.spines[s].set_visible(False)
    axes[0].set_ylabel(r"fitted value $\hat f(u_* + t\,\omega)$")
    footer(fig, "plotted (computed level): check05_smoothness.py, seed 20261025, N = 300 uniform points in the unit cube; "
                "SciPy RBFInterpolator, degree 1; smoothing 0 and each kernel's 5-fold CV choice", y=-0.03)
    save(fig, CFG["out"])
    print("wrote", CFG["out"])


if __name__ == "__main__":
    main()
