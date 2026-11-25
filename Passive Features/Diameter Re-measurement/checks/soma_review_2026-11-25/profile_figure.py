"""figN5 of the notation figures: Q(c), a_hat(c), b_hat(c) and S over the held c, from profile_c.py's two runs.
Standalone (not pipeline code). Run: python3 profile_figure.py profile_g079.json profile_g060.json OUT_DIR"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common  # noqa: F401  (puts the repository's src/ on the path)
from allen_diameter.analysis.soma_fit import ellipsoid_area

BLUE, ORANGE, INK, MUTED = "#257de0", "#f47230", "#191919", "#666666"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "figure.dpi": 150})
TRUTH = [0.6, -0.4, 30.96, 7.0, 5.6, 4.5, 1.13, 0.87 / 9.0]


def load(path, gamma_true):
    """profile_c.py's JSON, or -- for a run stopped early -- its .out log (the points finished; no c-free fit)."""
    if path.endswith(".json"):
        return json.load(open(path))
    rows = []
    for line in open(path):
        if line.startswith("c ") and "rms" in line:
            w = line.split()
            c, rms, a, b, z0, mu, S = (float(w[i]) for i in (1, 3, 6, 8, 10, 12, 16))
            rows.append(dict(c=c, rms_gl=rms, theta=[np.nan, np.nan, z0, a, b, c, np.nan, mu], S_um2=S))
    return dict(gamma_true=gamma_true, truth=TRUTH, S_true=ellipsoid_area(*TRUTH[3:6]),
                profile=sorted(rows, key=lambda r: r["c"]), c_free=None)


runs = [load(sys.argv[1], 0.79), load(sys.argv[2], 0.60)]
OUT = sys.argv[3]; os.makedirs(OUT, exist_ok=True)
fig, axs = plt.subplots(1, 3, figsize=(12.2, 3.9))
summary = []
for run, col in zip(runs, (BLUE, ORANGE)):
    P = run["profile"]; tr = run["truth"]
    c = np.array([p["c"] for p in P]); th = np.array([p["theta"] for p in P])
    rms = np.array([p["rms_gl"] for p in P]); S = np.array([p["S_um2"] for p in P])
    cf = run["c_free"]; chat = cf["theta"][5] if cf else np.nan
    lab = "world $\\gamma$ = %.2f%s" % (run["gamma_true"], " (the held value is right)" if abs(run["gamma_true"] - 0.79) < 1e-9 else " (the held value is wrong)")
    axs[0].plot(c, rms, "-o", color=col, lw=1.8, ms=4, label=lab)
    if cf:
        axs[0].plot(chat, cf["rms_gl"], "*", color=col, ms=12, mec="white", mew=0.6)
    axs[1].plot(c, 100 * (th[:, 3] / tr[3] - 1), "-o", color=col, lw=1.8, ms=3.5)
    axs[1].plot(c, 100 * (th[:, 4] / tr[4] - 1), "--s", color=col, lw=1.3, ms=3.0)
    axs[2].plot(c, S, "-o", color=col, lw=1.8, ms=4)
    if cf:
        axs[2].plot(chat, cf["S_um2"], "*", color=col, ms=12, mec="white", mew=0.6)
    summary.append(dict(gamma_true=run["gamma_true"], c_hat=chat, S_true=run["S_true"], a_range_pct=[float(100 * (th[:, 3].min() / tr[3] - 1)), float(100 * (th[:, 3].max() / tr[3] - 1))],
                        b_range_pct=[float(100 * (th[:, 4].min() / tr[4] - 1)), float(100 * (th[:, 4].max() / tr[4] - 1))],
                        S_range=[float(S.min()), float(S.max())], c_grid=c.tolist(), rms=rms.tolist()))
for a in axs:
    a.axvline(runs[0]["truth"][5], color=MUTED, lw=0.8, ls="--")
    a.set_xlabel("held depth semi-axis $c$ (um)")
axs[0].set_ylabel("RMS misfit $\\sqrt{2Q(c)/N}$ (gl)")
axs[0].set_title("(a) the misfit left at each held $c$", fontsize=9, loc="left")
axs[0].text(0.02, 0.02, "noise-free data; noise of SD 3 gl would add\nin quadrature: $\\sqrt{9 + m^2}$ gl at every $c$",
            transform=axs[0].transAxes, fontsize=7.2, color=MUTED, va="bottom")
axs[0].legend(frameon=False, fontsize=7.2, loc="lower left", bbox_to_anchor=(0.0, 0.13))
axs[1].set_ylabel("fitted in-plane axis vs truth (%)")
axs[1].set_title("(b) $\\hat a$ (solid), $\\hat b$ (dashed)", fontsize=9, loc="left")
axs[1].axhline(0, color=MUTED, lw=0.6)
tr0 = runs[0]["truth"]; cg = np.linspace(2.4, 6.6, 85)
axs[2].plot(cg, [ellipsoid_area(tr0[3], tr0[4], x) for x in cg], color=MUTED, lw=1.1, ls="-.",
            label="$S(a, b, c)$ with the true $a$, $b$ held fixed")
axs[2].legend(frameon=False, fontsize=7.4, loc="upper left")
axs[2].axhline(runs[0]["S_true"], color=INK, lw=0.8, ls=":")
axs[2].text(5.2, runs[0]["S_true"] - 14, "true $S$ = %.0f um$^2$" % runs[0]["S_true"], fontsize=7.6, color=INK)
axs[2].set_ylabel("$S(\\hat a, \\hat b, c)$ (um$^2$)")
axs[2].set_title("(c) the area at each held $c$", fontsize=9, loc="left")
axs[2].text(0.98, 0.04, ("stars: the fit with $c$ free ($\\gamma$ held at 0.79)\n" if any(r["c_free"] for r in runs) else
              "partial run: the points finished by 14:43\n") + "dashed line: the true $c$ = 4.5 um",
            transform=axs[2].transAxes, fontsize=7.4, color=MUTED, ha="right")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figN5_profile_Q_S.png")); plt.close(fig)
json.dump(summary, open(os.path.join(OUT, "figN5_numbers.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
