"""Figures for the explanation of the revised soma approach (2026-11-25). Standalone; run:
python3 figures_revised.py sensitivity.json OUT_DIR. fig5 is a schematic; fig6 is plotted from sensitivity.json
(sensitivity.py, linearised biases); fig7 from Legendre's area (the pipeline's ellipsoid_area)."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import common as C

J = json.load(open(sys.argv[1])); OUT = sys.argv[2]; os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, GREEN, INK, MUTED = "#257de0", "#d9730d", "#2a9d6f", "#191919", "#666666"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "figure.dpi": 150})

# ---- fig5: the revised approach (schematic)
fig, ax = plt.subplots(figsize=(10, 4.3)); ax.set_xlim(0, 100); ax.set_ylim(0, 44); ax.axis("off")
def box(x, y, w, h, text, ec=INK, fc="white", ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4", fc=fc, ec=ec, lw=1.2, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8, color=INK)
def arrow(x1, y1, x2, y2, ls="-"):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.1, ls=ls))
box(1, 27, 15, 9, "the stack:\n35 planes, region R")
box(22, 27, 20, 9, "ellipsoid fit, $c$ held\nat each value of the range", ec=BLUE, fc="#eaf2fd")
box(48, 31, 21, 7, "from the data:\ncentre, $a$, $b$, $\\varphi$, $\\mu c$", ec=BLUE)
box(48, 21, 21, 7, "misfit against $c$\n(does the stack prefer a $c$?)", ec=BLUE)
box(22, 9, 20, 9, "range of $c$: stems' depths,\nz-shrinkage decision", ec=ORANGE, fc="#fdf1e6")
box(75, 27, 11, 9, "$S$ as a range\n(low, mid, high)", ec=GREEN, fc="#e9f6f0")
box(89, 27, 10, 9, "passive fit:\n$C_m$ range", ec=GREEN)
box(60, 6, 28, 8, "only if the $C_m$ range is too wide:\ncalibrate the far slope $\\gamma$, then fit $c$", ec=MUTED, ls="--")
arrow(16.5, 31.5, 21.5, 31.5); arrow(42.5, 33, 47.5, 34.5); arrow(42.5, 30, 47.5, 25.5)
arrow(32, 18.5, 32, 26.5); arrow(69.5, 34, 74.5, 32.5); arrow(86.5, 31.5, 88.5, 31.5)
arrow(94, 26.5, 80, 14.5, ls="--")
ax.text(1, 41, "schematic: blue = fixed by the stack, orange = set from outside the fit, green = carried downstream", fontsize=8, color=MUTED)
fig.savefig(os.path.join(OUT, "fig5_revised_approach.png"), bbox_inches="tight"); plt.close(fig)

# ---- fig6: what the held far slope does to c and to a (D-241 region, B_k re-solved)
cfgk = "R 5.97 um, B_k free"
fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=True)
cols = [ORANGE, BLUE, GREEN]
for (label, r), col in zip(J["results"].items(), cols):
    b = r["configs"][cfgk]["bias"]
    x = [0.60, 0.79, 0.95]
    short = "c = %.2f um (a %.2f, b %.2f)" % (r["theta"][5], r["theta"][3], r["theta"][4])
    for ax, key in zip(axs, ("dc_rel", "da_rel")):
        y = [100 * b["far slope 0.60 (model 0.79)"][key], 0.0, 100 * b["far slope 0.95 (model 0.79)"][key]]
        ax.plot(x, y, "-o", color=col, lw=2, ms=5, label=short)
for ax, t in zip(axs, ("bias of the depth semi-axis $c$", "bias of the in-plane semi-axis $a$")):
    ax.axhline(0, color=MUTED, lw=0.6); ax.axvline(0.79, color=MUTED, lw=0.6, ls="--")
    ax.set_xlabel("true far slope $\\gamma$ of the simulated world (model holds 0.79)"); ax.set_title(t, fontsize=9, loc="left")
axs[0].set_ylabel("bias after one Gauss-Newton step (%)"); axs[0].legend(frameon=False, fontsize=8, title="simulated soma", title_fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig6_slope_bias.png")); plt.close(fig)

# ---- fig7: S against c with a carried range
a, b = 7.0, 5.6
cs = np.linspace(1.5, 8, 131); S = np.array([C.soma_fit.ellipsoid_area(a, b, x) for x in cs])
allen = 4 * np.pi * 5.97 ** 2
fig, ax = plt.subplots(figsize=(6.2, 3.8))
lo, hi = 2.5, 4.5
ax.axvspan(lo, hi, color=ORANGE, alpha=0.15, lw=0)
ax.plot(cs, S, color=BLUE, lw=2)
ax.axhline(allen, color=MUTED, lw=0.8, ls="--"); ax.text(1.6, allen + 8, "Allen's sphere: %.0f um$^2$" % allen, fontsize=8, color=MUTED)
for x in (lo, 3.45, hi, b):
    v = C.soma_fit.ellipsoid_area(a, b, x); ax.plot(x, v, "o", color=BLUE, ms=5)
    ax.annotate("%.0f" % v, (x, v), (x + 0.12, v - 22), fontsize=8, color=INK)
ax.text((lo + hi) / 2, 215, "illustrative range of $c$\nin the stack", ha="center", fontsize=8, color=INK)
ax.annotate("stems' half-extent\n3.45 um", (3.45, C.soma_fit.ellipsoid_area(a, b, 3.45)), (1.7, 395), fontsize=8, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
ax.annotate("$c = b$", (b, C.soma_fit.ellipsoid_area(a, b, b)), (6.1, 395), fontsize=8, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
ax.set_xlabel("depth semi-axis $c$ (um)"); ax.set_ylabel("area $S(a, b, c)$ (um$^2$)")
ax.set_title("a = %.1f um, b = %.1f um (illustrative, silhouette-sized)" % (a, b), fontsize=9, loc="left")
ax.set_xlim(1.5, 8); ax.set_ylim(180, 620)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig7_area_range.png")); plt.close(fig)
r = J["results"]
for k, v in r.items():
    bb = v["configs"][cfgk]["bias"]
    print(k, {w: (round(100 * x["dc_rel"], 1), round(100 * x["da_rel"], 2), round(100 * x["dS_rel"], 1)) for w, x in bb.items() if "slope" in w})
