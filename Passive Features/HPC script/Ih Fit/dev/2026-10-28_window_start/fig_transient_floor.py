"""fig_transient_floor.py -- PLOTTED: the transient test's statistic against its
threshold, on simulated cells (pipeline functions via check_findings_2026-10-23).

For the archive's layouts "2 sweeps, K 63" and "10 sweeps, K 348", theta_cf,
sigma_b = 0.1 mV/s: the 5 ms bin means of r_odd(t) (noise_windows.transient_start)
for noise alone, over N_NULL cells -> the per-bin SD of the bin mean (the
statistic's noise floor); one cell with the RC pulse response (113 MOhm,
28.1 ms, +/-200 pA x 0.5 ms) added; the noiseless response's odd part; the
threshold of D-051 (iv), kappa_tr sigma_SS (0.1 sigma_SS), and the per-bin
simulated-null threshold 3.5 x SD (check D of 2026-10-23). Everything in units
of the cell's sigma_SS (median pre-pulse SD).

Run:  python fig_transient_floor.py    (about 2 min)  ->  fig_transient_floor.png
"""
import importlib.util
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("chk", HERE / "check_findings_2026-10-23.py")
chk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chk)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CFG = dict(seed=20261031, layouts=("2 sweeps, K 63", "10 sweeps, K 348"), n_null=(200, 120), sigma_b=0.1,
           kappa=0.1, z=3.5, out=HERE / "fig_transient_floor.png")
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
INK2, GREY = "#52514e", "#8a8985"


def bins_of(a, t_ms):
    return chk.all_bin_means(a, t_ms)


def main():
    t0 = time.time()
    rng = np.random.default_rng(CFG["seed"])
    t_ms = (np.arange(chk.N_W) - chk.PRE_N) / chk.FS * 1e3
    edges = np.arange(0.0, 200.0 + 1e-9, 5.0)
    centres = edges[:-1] + 2.5
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3), sharey=True)
    for ax, name, n_null in zip(axes, CFG["layouts"], CFG["n_null"]):
        sweep, pos = chk.build_layout(chk.CFG["layouts"][name], rng)
        rw = chk.response_windows(pos)
        null, sig = [], []
        for _ in range(n_null):
            a = chk.nw.transient_start(chk.make_ss(chk.simulate_noise_windows(CFG["sigma_b"], sweep, pos, rng), sweep, pos),
                                       chk.nw.WindowConfig(t_ss_floor_ms=0.0))
            null.append(bins_of(a.r_odd, t_ms))
            sig.append(a.sigma_ss_mV)
        null = np.array(null)
        s_ss = float(np.median(sig))
        sd = null.std(axis=0, ddof=1) / s_ss
        vn = chk.simulate_noise_windows(CFG["sigma_b"], sweep, pos, rng)
        b = chk.nw.transient_start(chk.make_ss(vn + rw, sweep, pos), chk.nw.WindowConfig(t_ss_floor_ms=0.0))
        cell = np.abs(bins_of(b.r_odd, t_ms)) / b.sigma_ss_mV
        dep = pos % 2 == 0
        resp_odd = 0.5 * (rw[dep].mean(axis=0) - rw[~dep].mean(axis=0))
        resp_odd = resp_odd - resp_odd[:chk.PRE_N].mean()
        clean = np.abs(bins_of(resp_odd, t_ms)) / s_ss
        ax.fill_between(centres, 0, sd, color="#dcdbd6", step="mid", label="noise alone: SD of a 5 ms bin mean of $\\bar r_{\\rm odd}$")
        ax.step(centres, CFG["z"] * sd, where="mid", color=GREY, lw=1.5, ls="--", label="per-bin null threshold, 3.5 x SD (option b)")
        ax.axhline(CFG["kappa"], color=SERIES[1], lw=2, label=r"threshold as written, $\kappa_{\rm tr}\hat\sigma_{\rm SS}$ = 0.1 $\hat\sigma_{\rm SS}$")
        ax.plot(centres, clean, color=SERIES[0], lw=2, label="noiseless pulse response (odd part)")
        ax.plot(centres, cell, "o", ms=4, color=SERIES[2], label="one simulated cell: noise + response")
        ax.axvline(80, color=INK2, lw=1, ls=":")
        ax.text(81, 3.0, "80 ms floor\n(D-076)", fontsize=8, color=INK2)
        ax.set_yscale("log")
        ax.set_ylim(3e-3, 5.0)
        ax.set_xlim(0, 200)
        ax.set_title("%s (%d dep / %d hyp windows)" % (name, int(dep.sum()), int((~dep).sum())))
        ax.set_xlabel("time after the pulse onset (ms), 5 ms bins")
        print("%s: floor SD at 100-195 ms %.3f-%.3f sigma_SS; response crosses 0.1 sigma_SS at %.0f ms (%.0f s)"
              % (name, sd[20:].min(), sd[20:].max(), centres[np.flatnonzero(clean >= CFG["kappa"])[-1]] + 2.5, time.time() - t0))
    axes[0].set_ylabel(r"$|$bin mean of $\bar r_{\rm odd}|$ / $\hat\sigma_{\rm SS}$")
    axes[0].legend(loc="lower left", fontsize=7.6)
    fig.tight_layout()
    fig.savefig(CFG["out"], dpi=150, bbox_inches="tight")
    print("wrote", CFG["out"])


if __name__ == "__main__":
    main()
