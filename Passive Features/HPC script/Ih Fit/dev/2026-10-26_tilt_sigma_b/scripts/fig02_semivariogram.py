"""fig02_semivariogram.py -- PLOTTED from formulas (no random numbers): the
semivariogram V(h) of two 120 ms window means h apart in one sweep, for the
stationary law theta_cf alone, plus a seconds-scale OU (3 s, 0.142 mV; 30 s,
0.294 mV: the sizes of topic doc v3 section 3.6.3), and plus the tilt that
(T1) reads from each OU at h* = 0.802 s. V of the stationary part:
noise_law.semivariogram (exact, sampled law); the OU excess: eq. (25), as in
check02; the tilt: (1/2) sigma_b^2 h^2 (exact).
Run:  python fig02_semivariogram.py  ->  ../figures/fig02_semivariogram.png
"""
from pathlib import Path

import numpy as np

import check02_lag_choice as c2
import fig_style as fs
import tilt_common as tc

CFG = dict(ous=((3.0, 0.142), (30.0, 0.294)), T=0.120, h_max_periods=36,
           out=Path(__file__).resolve().parent.parent / "figures" / "fig02_semivariogram.png")


def data():
    lags = np.arange(1, CFG["h_max_periods"] + 1)                 # periods (odd ones: opposite polarity)
    h = lags * tc.P * tc.DT
    V_stat = np.array([tc.R.nl.semivariogram(tc.LAW_CF, tc.DT, tc.M_WIN, tc.M_WIN, int(L) * tc.P) for L in lags])
    hs = tc.h_of(4)
    curves = []
    for tau, sigma in CFG["ous"]:
        V_ou = V_stat + np.array([c2.V3(sigma, tau, CFG["T"], x) for x in h])
        sb = c2.sigma_b_read(sigma, tau, CFG["T"], hs)
        V_tilt = V_stat + 0.5 * sb ** 2 * h ** 2
        curves.append((tau, sigma, sb, V_ou, V_tilt))
    return lags, h, V_stat, curves


def plot(lags, h, V_stat, curves):
    plt = fs.setup()
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    even = lags % 2 == 0
    ax.plot(h[even], V_stat[even], color=fs.GREY, lw=2, label=r"stationary law alone, $V(h\mid\theta_{\rm cf},0)$")
    for i, (tau, sigma, sb, V_ou, V_tilt) in enumerate(curves):
        col = fs.SERIES[i]
        ax.plot(h[even], V_ou[even], color=col, lw=2, marker=fs.MK[i], ms=4,
                label=r"+ OU of %g s, $\sigma_3$ = %.3f mV" % (tau, sigma))
        ax.plot(h[even], V_tilt[even], color=col, lw=1.6, ls="--",
                label=r"+ tilt read from it at $h^*$, $\sigma_b$ = %.3f mV/s" % sb)
    ax.axvline(tc.h_of(4), color=fs.INK2, lw=1.0, ls=":")
    ax.text(tc.h_of(4) * 1.04, 2.2e-3, r"$h^*$ = 0.802 s", color=fs.INK2, fontsize=9)
    ax.axvspan(0.14, 1.14, color="#f3e9d9", alpha=0.6, lw=0)
    ax.text(0.31, 1.25e-3, "shaded: lags the LS step reads\n(0.14-1.14 s from the pre-window centre)", fontsize=7.8, color=fs.INK2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"lag $h$ between window centres (s; same-polarity lags, even multiples of 200.5 ms)")
    ax.set_ylabel(r"$V(h)$ (mV$^2$)")
    ax.set_xlim(0.3, 8.0)
    ax.legend(loc="upper left", fontsize=8.2)
    fig.tight_layout()
    CFG["out"].parent.mkdir(exist_ok=True)
    fig.savefig(CFG["out"], dpi=150, bbox_inches="tight")
    for tau, sigma, sb, V_ou, V_tilt in curves:
        i4 = 3                                                   # 4 periods
        print("OU %g s: sigma_b read at h* = %.4f mV/s; V at h*: OU %.5f, tilt %.5f mV^2 (equal); at 7.218 s: OU %.4f, tilt %.4f;"
              " at 0.401 s: OU %.5f, tilt %.5f" % (tau, sb, V_ou[i4], V_tilt[i4], V_ou[-1], V_tilt[-1], V_ou[1], V_tilt[1]))
    print("wrote", CFG["out"])


if __name__ == "__main__":
    plot(*data())
