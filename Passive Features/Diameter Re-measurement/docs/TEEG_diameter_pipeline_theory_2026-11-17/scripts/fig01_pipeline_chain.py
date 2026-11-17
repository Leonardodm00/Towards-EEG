"""Figure 1 (schematic): the chain of the main route, what feeds what. python3 fig01_pipeline_chain.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from _common import FIG

# ---- CONFIG ----
BOXES = [  # key, x, y, w, h, text, kind
    ("data", 0.30, 9.2, 4.4, 0.8, "Image stack $I_k(x, y)$ and traced nodes (SWC)\nmeasured", "data"),
    ("first", 0.30, 7.9, 4.4, 0.8, "First-stage fit of every node (3.3)\n$\\theta^{(0)}_i$: computed", "step"),
    ("bg", 0.30, 6.3, 4.4, 0.8, "Background $B_{i,k}$ through the shadow model (3.4)\ncomputed, every round", "loop"),
    ("nb", 0.30, 5.1, 4.4, 0.8, "Neighbours' shadow $S^{\\rm nb}_{i,k}$, tier 1 (3.4)\ncomputed from $\\theta^{(t-1)}$ of the others", "loop"),
    ("clean", 0.30, 3.9, 4.4, 0.8, "Cleaning: $I^{(t)}_{i,k} = I_{i,k} + B_{i,k} S^{\\rm nb}_{i,k}$ (3.4)\nthen the profile on the line", "loop"),
    ("refine", 0.30, 2.7, 4.4, 0.8, "Joint refinement of $\\theta_i$ over all planes (3.5)\n$\\hat\\theta^{(t)}_i$: computed", "loop"),
    ("fit", 0.30, 1.1, 4.4, 0.8, "Per-node joint fit with covariance (3.5)\n$\\hat\\theta_i$, $\\widehat{\\rm Cov}(\\hat\\theta_i)$: computed", "step"),
    ("own", 5.60, 1.1, 4.1, 0.8, "Same-node clearance, a product (3.6)\n$S^{\\rm in}$, $S^{\\rm out}$, $I^{\\rm clr}$, chord $c_k$", "prod"),
    ("fwd", 5.60, 5.1, 4.1, 2.0, "Forward model (3.2)\nabsorbed light per slab $\\Delta A_j$,\nblurred by $G_{\\sigma_{\\rm r}(\\zeta_j - z_k)}$\nkernel $\\sigma_{\\rm r}$: measured / configured", "model"),
    ("out", 5.60, -0.3, 4.1, 0.8, "Diameters $\\hat d_i$ to the SWC and the passive refit\n(not built here)", "data"),
]
COL = {"data": "#e8e8e8", "step": "#dbe9f6", "loop": "#fde9c9", "prod": "#d9f0e3", "model": "#ece3f5"}
ARROWS = [("data", "first"), ("first", "bg"), ("bg", "nb"), ("nb", "clean"), ("clean", "refine"), ("refine", "fit"),
          ("fit", "own")]
# ----------------

fig, ax = plt.subplots(figsize=(9.6, 8.6))
pos = {}
for key, x, y, w, h, text, kind in BOXES:
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04", fc=COL[kind], ec="0.3", lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.6)
    pos[key] = (x, y, w, h)
for a, b in ARROWS:
    xa, ya, wa, ha = pos[a]
    xb, yb, wb, hb = pos[b]
    if abs(xa - xb) < 1e-9:
        ax.annotate("", xy=(xb + wb / 2, yb + hb + 0.05), xytext=(xa + wa / 2, ya - 0.05),
                    arrowprops=dict(arrowstyle="->", lw=1.0, color="0.2"))
    else:
        ax.annotate("", xy=(xb - 0.05, yb + hb / 2), xytext=(xa + wa + 0.05, ya + ha / 2),
                    arrowprops=dict(arrowstyle="->", lw=1.0, color="0.2"))
# the loop over rounds: refine -> bg
ax.annotate("", xy=(0.25, 6.7), xytext=(0.25, 3.1),
            arrowprops=dict(arrowstyle="->", lw=1.0, color="#b06a00", connectionstyle="arc3,rad=-0.55"))
ax.text(-0.95, 4.9, "next round $t$\n(all nodes at once)", fontsize=8, color="#b06a00", ha="center", va="center", rotation=90)
# forward model feeds nb, refine, fit, own
for key in ("nb", "refine"):
    x, y, w, h = pos[key]
    ax.annotate("", xy=(x + w + 0.05, y + h / 2), xytext=(5.55, 6.1 if key == "nb" else 5.4),
                arrowprops=dict(arrowstyle="->", lw=0.8, color="#6a3d9a", ls="--"))
x, y, w, h = pos["fit"]
ax.annotate("", xy=(x + w - 0.3, y + h + 0.05), xytext=(5.9, 5.05), arrowprops=dict(arrowstyle="->", lw=0.8, color="#6a3d9a", ls="--"))
x, y, w, h = pos["own"]
ax.annotate("", xy=(x + w / 2, y + h + 0.05), xytext=(x + w / 2, 5.05), arrowprops=dict(arrowstyle="->", lw=0.8, color="#6a3d9a", ls="--"))
ax.annotate("", xy=(7.65, 0.55), xytext=(2.5, 1.05), arrowprops=dict(arrowstyle="->", lw=1.0, color="0.2", connectionstyle="arc3,rad=0.15"))
ax.set_xlim(-1.5, 10.0)
ax.set_ylim(-0.6, 10.3)
ax.axis("off")
fig.savefig(FIG / "fig01_pipeline_chain.png", dpi=170, bbox_inches="tight")
print("wrote", FIG / "fig01_pipeline_chain.png")
