"""fig02_window_mean_semivariogram.py -- Figure 2 of TEEG_semivariogram_2026-11-13.md,
PLOTTED: the semivariogram V(h) of two 120 ms window means of one SS sweep,
same polarity, at the project's lags (h = L x 200.5 ms, L = 2, 4, ..., 36), under
the stationary law theta_cf alone (the floor) and with one seconds-scale term
added: a per-sweep tilt (sigma_b = 0.1 mV/s), or an OU of 3 s (0.142 mV) or of
30 s (0.294 mV), the two OUs sized in the topic doc (section 3.6.3) to the
cohort's 6.89 s change.

Curves: formulas, computed exactly for the sampled law by the repository's
noise_law.semivariogram (Block 2) on a dense grid of window separations.
Dots: the Monte Carlo estimates of the floor by check02_window_means.py (400
simulated sweeps without tilt, +/- 2 SE), read from
../results/check02_window_means.json; run check02 first. (The tilt case's
estimates are reported in the document, not drawn: the 400 simulated tilts
have mean square 1.17 sigma_b^2, so they sit about 2 SE above the curve drawn
for sigma_b, on the curve of their own draws -- check02 (a), (c).) (a) V(h), log y; (b) the excess over the floor, log-log, with h and h^2
guides.

Layers: CFG | curves() (formulas) | load_dots() (data) | plot().
Run:  python fig02_window_mean_semivariogram.py  ->  ../figures/fig02_window_mean_semivariogram.png
Repository path: environment variable TEEG_IH_FIT, else the Ih Fit folder three
levels above this file.

Pure ASCII, LF.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

import viz_style as vs

HERE = Path(__file__).resolve().parent
CFG = dict(
    fs=50000.0, period=10025, m=6000,
    theta_cf=(0.0380, 0.175, 0.0508, 0.0366),
    sigma_b=0.1,
    ou_seconds=((3.0, 0.142), (30.0, 0.294)),
    lag_star=4,
    grid_step=200,                     # samples between curve points (4 ms)
    dots=HERE.parent / "results" / "check02_window_means.json",
    out=HERE.parent / "figures" / "fig02_window_mean_semivariogram.png",
)
REPO_IH_FIT = Path(os.environ.get("TEEG_IH_FIT", str(HERE.parents[2])))
sys.path.insert(0, str(REPO_IH_FIT))
import noise_law as nl   # noqa: E402


def curves():
    c = CFG
    dt = 1.0 / c["fs"]
    law = nl.NoiseLaw(*c["theta_cf"])
    off = np.arange(2 * c["period"], 36 * c["period"] + 1, c["grid_step"])
    h = off * dt
    floor = np.array([nl.semivariogram(law, dt, c["m"], c["m"], int(o)) for o in off])
    out = dict(h=h, floor=floor, tilt=floor + 0.5 * c["sigma_b"] ** 2 * h ** 2)
    for tau3, sig3 in c["ou_seconds"]:
        ou = nl.NoiseLaw(0.0, 0.5, sig3, tau3)
        out["ou%g" % tau3] = floor + np.array([nl.semivariogram(ou, dt, c["m"], c["m"], int(o)) for o in off])
    return out


def load_dots():
    d = json.loads(Path(CFG["dots"]).read_text())
    rows = d["lags"]
    get = lambda k: np.array([r[k] for r in rows])   # noqa: E731
    return dict(h=get("h_s"), v0=get("V_hat_notilt"), se0=get("se_notilt"),
                v1=get("V_hat_tilt"), se1=get("se_tilt"))


def plot(cv, dots):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter
    vs.apply()
    c = CFG
    blue, orange, aqua = vs.SERIES[:3]
    hstar = c["lag_star"] * c["period"] / c["fs"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.6, 4.1), constrained_layout=True)

    # (a) V(h), log y
    series = (("tilt", blue, r"+ tilt, $\sigma_b$ = 0.1 mV/s"),
              ("ou3", orange, "+ OU 3 s, 0.142 mV"),
              ("ou30", aqua, "+ OU 30 s, 0.294 mV"))
    a.plot(cv["h"], cv["floor"], color=vs.NEUTRAL, lw=vs.LW, label=r"floor: $\theta_{\rm cf}$ alone", zorder=3)
    for key, col, lab in series:
        a.plot(cv["h"], cv[key], color=col, lw=vs.LW, label=lab, zorder=3)
    vs.ringed(a, dots["h"], dots["v0"], 2 * dots["se0"], vs.NEUTRAL, label="floor, simulated (400 sweeps)", zorder=5)
    a.set_yscale("log")
    a.set_xlim(0, 7.6)
    a.set_ylim(7e-4, 0.5)
    a.axvline(hstar, color=vs.TEXT2, lw=0.8, zorder=2)
    a.text(hstar + 0.08, 0.33, r"$h^*$ = 0.802 s (4 periods)", fontsize=8, color=vs.TEXT2, va="top")
    a.set_xlabel(r"distance between window centres $h$ (s)")
    a.set_ylabel(r"$V(h)$ (mV$^2$)")
    a.set_title("(a) Two 120 ms window means, same sweep and polarity", pad=8)
    # direct labels only where the curves do not converge (the two OUs meet at 7.2 s: legend only)
    ends = dict(floor=(cv["floor"][-1], "floor"), tilt=(cv["tilt"][-1], "tilt"))
    for key, (yv, lab) in ends.items():
        a.text(cv["h"][-1] + 0.06, yv, lab, fontsize=8, color=vs.TEXT2, va="center")
    a.legend(loc="lower right", bbox_to_anchor=(1.0, 0.1))
    a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "%g" % v))

    # (b) excess over the floor, log-log
    for key, col, lab in series:
        b.plot(cv["h"], cv[key] - cv["floor"], color=col, lw=vs.LW, label=lab, zorder=3)
    hh = np.array([0.401, 7.218])
    for p, y0, lab in ((1, 1.2e-2, r"$\propto h$"), (2, 3.2e-3, r"$\propto h^2$")):
        b.plot(hh, y0 * (hh / hh[0]) ** p, color=vs.TEXT2, lw=0.8, zorder=2)
        b.text(hh[-1] * 1.03, y0 * (hh[-1] / hh[0]) ** p, lab, fontsize=8.5, color=vs.TEXT2,
               ha="left", va="center")
    b.set_xscale("log")
    b.set_yscale("log")
    b.set_xlim(0.36, 9.0)
    b.set_ylim(5e-4, 3)
    ticks = [0.401, 0.802, 1.604, 3.208, 7.218]
    b.xaxis.set_major_locator(FixedLocator(ticks))
    b.xaxis.set_minor_locator(NullLocator())
    b.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "%.3g" % v))
    b.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "%g" % v))
    b.set_xlabel(r"$h$ (s), log scale")
    b.set_ylabel(r"excess $V(h) - V(h \mid \theta_{\rm cf}, 0)$ (mV$^2$)")
    b.set_title("(b) What the seconds scale adds, log-log", pad=8)
    b.legend(loc="lower right")
    fig.savefig(c["out"], dpi=170)
    return c["out"]


def main():
    cv = curves()
    dots = load_dots()
    print("floor at h* = %.4e mV^2; floor + tilt at h* = %.4e mV^2"
          % (np.interp(0.802, cv["h"], cv["floor"]), np.interp(0.802, cv["h"], cv["tilt"])))
    print("wrote", plot(cv, dots))


if __name__ == "__main__":
    main()
