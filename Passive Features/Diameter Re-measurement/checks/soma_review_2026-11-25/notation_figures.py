"""Figures of the soma fit's notation (2026-11-25, 14:17): one figure per group of symbols of the notation table.
Standalone (not pipeline code). Run: python3 notation_figures.py OUT_DIR
  figN1  where each symbol comes from (schematic: structure, statuses from the code read on 2026-11-25)
  figN2  the ellipsoid's parameters (schematic to scale, the simulated soma's numbers)
  figN3  D_k, R, tau_k, B_k on three planes (plotted: a simulated stack, the pipeline's renderer, the pilot's table)
  figN4  sigma_r(delta) and gamma (plotted from the pilot's table, D-170, and the configuration)
Simulated soma (as in profile_c.py): a 7.0, b 5.6, c 4.5 um, centre (0.6, -0.4, 30.96) um, phi 1.13 rad, 2 mu c 0.87,
background 110 gl, noise SD 3 gl (figN3 only), seed 529. Region R: disc of 5.97 um about the traced centre (0, 0)."""
import os, sys, dataclasses
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle, FancyBboxPatch, Arc, Polygon
import common as C
from allen_diameter.model.kernel import sigma_r, with_kernel
from allen_diameter.analysis.soma_fit import vp_scale

OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, AQUA = "#257de0", "#f47230", "#00b894"          # dataviz reference palette, slots 1-3 (validated)
INK, MUTED, GRID = "#191919", "#666666", "#d0d0d0"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "figure.dpi": 150})
X0, Y0, Z0, A, B, Cc, PHI = 0.6, -0.4, 30.96, 7.0, 5.6, 4.5, 1.13
MU = 0.87 / (2 * Cc)
TRUTH = np.array([X0, Y0, Z0, A, B, Cc, PHI, MU])
RS = C.CONFIG["traced_radius_um"]; XS, YS, ZS = C.CONFIG["traced_centre_um"]
PILOT = dataclasses.replace(C.RCFG, kernel_table_delta_um=(0.0, 0.28, 0.56, 0.84),
                            kernel_table_sigma_um=(0.0800, 0.1536, 0.3924, 0.5620))

# ======================================================================= figN1: provenance of the symbols
fig, ax = plt.subplots(figsize=(11.5, 6.4)); ax.set_xlim(0, 100); ax.set_ylim(0, 62); ax.axis("off")
COL = {"measured": BLUE, "configured": ORANGE, "computed": AQUA, "none": MUTED}
FILL = {"measured": "#e6f0fc", "configured": "#feeee5", "computed": "#e0f6f1", "none": "white"}
boxes = {}
def node(key, x, y, w, h, text, status):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35", fc=FILL[status], ec=COL[status], lw=1.4,
                                ls="-" if status != "none" else "--"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.6, color=INK, linespacing=1.25)
    boxes[key] = (x, y, w, h)
def link(a, b, side_a="r", side_b="l", label=None, dy_a=0.0, dy_b=0.0, rad=0.0):
    xa, ya, wa, ha = boxes[a]; xb, yb, wb, hb = boxes[b]
    pa = {"r": (xa + wa + 0.4, ya + ha / 2 + dy_a), "l": (xa - 0.4, ya + ha / 2 + dy_a), "t": (xa + wa / 2, ya + ha + 0.4),
          "b": (xa + wa / 2, ya - 0.4)}[side_a]
    pb = {"r": (xb + wb + 0.4, yb + hb / 2 + dy_b), "l": (xb - 0.4, yb + hb / 2 + dy_b), "t": (xb + wb / 2, yb + hb + 0.4),
          "b": (xb + wb / 2, yb - 0.4)}[side_b]
    ax.annotate("", pb, pa, arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.0, connectionstyle="arc3,rad=%g" % rad))
    if label:
        ax.text((pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2 + 0.8, label, fontsize=6.8, color=MUTED, ha="center")
for xh, t in ((11, "inputs to the fit"), (36, "model level\n(one value of $c$)"), (61, "the fit on $R$"),
              (87.5, "estimate level")):
    ax.text(xh, 60.2, t, ha="center", va="top", fontsize=8.5, color=INK, fontweight="bold")
for xv in (23.5, 48.5, 73.5):
    ax.plot([xv, xv], [1, 57], color=GRID, lw=0.8, ls=":")
node("D", 1, 47, 20, 8, "$D_k(x,y)$: the stack's planes\n35 planes about the start depth\n(gl)  measured", "measured")
node("allen", 1, 35, 20, 8, "Allen's trace: soma centre,\nradius $r_s$ = 5.97 um, branches\n(SWC)  measured by Allen", "measured")
node("grow", 1, 23, 20, 8, "$\\sigma_r(\\delta)$ for 0 < |$\\delta$| <= 0.84 um:\nCell 7's growth table\ncomputed from the pilot's stack", "computed")
node("conf", 1, 12, 20, 7, "$\\sigma_r(0)$ = 0.080 um (D-172),\n$\\gamma$ = 0.79 (D-240)\nconfigured", "configured")
node("c", 1, 1.5, 20, 7.5, "$c$ held at each value\nof a range (stems' depths,\nz-shrinkage)  configured (proposed)", "configured")
node("R", 26, 36, 20, 7, "$R$: disc of $r_s$ about Allen's\ncentre, minus traced branches\n(D-241)  computed", "computed")
node("theta", 26, 24.5, 20, 7, "$\\theta$ = ($x_0, y_0, z_0, a, b, \\varphi, \\mu$)\nthe fit's variables:\nno value until fitted", "none")
node("tau", 26, 11, 20, 8, "$\\tau_k(x,y;\\theta,c)$: model\ntransmission of plane $k$\nrenderer, Block 45  computed", "computed")
node("Bk", 51, 36, 20, 7.5, "$B_k$ = $\\Sigma_R D_k\\tau_k$ / $\\Sigma_R \\tau_k^2$\nbrightness of plane $k$ (gl)\nclosed form  computed", "computed")
node("ls", 51, 19, 20, 9, "least squares on $R$:\nmin over $\\theta$ of\n$\\Sigma_k\\Sigma_R\\,(D_k - B_k\\tau_k)^2$\n(trf, Block 46)", "none")
node("thhat", 76, 45, 22, 8, "$\\hat\\theta(c)$: fitted centre,\n$\\hat a$, $\\hat b$, $\\hat\\varphi$, $\\hat\\mu$\ncomputed (fitted)", "computed")
node("Q", 76, 34, 22, 7.5, "$Q(c)$ = $\\frac{1}{2}\\Sigma\\,(D_k - \\hat B_k\\tau_k)^2$\nmisfit left at this $c$ (gl$^2$)\ncomputed", "computed")
node("S", 76, 18.5, 22, 9, "$S(\\hat a, \\hat b, c)$: Legendre's area\n(um$^2$), one value per $c$\n= a range over the range of $c$\ncomputed", "computed")
node("pf", 76, 3, 22, 7, "passive fit: $C_m$\nfor the low, middle, high $S$\n(downstream, not this step)", "none")
link("allen", "R", label="D-241")
link("D", "Bk", dy_a=1.5, dy_b=1.5)
link("R", "Bk")
link("theta", "tau", "b", "t")
link("grow", "tau", dy_b=1.5)
link("conf", "tau", dy_b=-0.5)
link("c", "tau", "r", "l", dy_b=-2.5)
link("tau", "Bk", "r", "l", dy_b=-2.0, rad=0.15)
link("Bk", "ls", "b", "t")
link("tau", "ls", "r", "l", dy_b=-2.0)
link("R", "ls", "r", "l", dy_b=2.0, rad=-0.1)
link("ls", "thhat", "r", "l", label="argmin", dy_a=2.5)
link("ls", "Q", "r", "l", label="min", dy_a=0.8)
link("thhat", "S", "r", "r", rad=-0.55)
ax.text(101.2, 36.5, "$\\hat a$, $\\hat b$", fontsize=7.4, color=MUTED)
link("c", "S", "r", "l", rad=0.12)
link("S", "pf", "b", "t")
for i, (st, t) in enumerate((("measured", "measured: read from data or a trace"),
                             ("configured", "configured: set by a decision"),
                             ("computed", "computed: a function of the others (incl. fitted)"),
                             ("none", "no stored value: a variable or a step"))):
    xl, yl = 4 + 48 * (i % 2), -2.8 - 3.0 * (i // 2)
    ax.add_patch(FancyBboxPatch((xl, yl - 0.8), 2.2, 1.6, boxstyle="round,pad=0.1", fc=FILL[st], ec=COL[st],
                                lw=1.2, ls="-" if st != "none" else "--"))
    ax.text(xl + 3.2, yl, t, fontsize=7.4, color=INK, va="center")
ax.set_ylim(-7.5, 62)
fig.savefig(os.path.join(OUT, "figN1_provenance.png"), bbox_inches="tight"); plt.close(fig)

# ======================================================================= figN2: the ellipsoid's parameters
fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.4, 4.8), gridspec_kw={"width_ratios": [1, 1.15]})
ax.add_patch(Ellipse((X0, Y0), 2 * A, 2 * B, angle=np.degrees(PHI), fc="#e6f0fc", ec=BLUE, lw=1.6))
ax.add_patch(Circle((XS, YS), RS, fc="none", ec=ORANGE, lw=1.4, ls="--"))
u = np.array([np.cos(PHI), np.sin(PHI)]); v = np.array([-np.sin(PHI), np.cos(PHI)])
ax.plot([X0, X0 + A * u[0]], [Y0, Y0 + A * u[1]], color=BLUE, lw=1.2)
ax.plot([X0, X0 + B * v[0]], [Y0, Y0 + B * v[1]], color=BLUE, lw=1.2)
ax.text(X0 + 0.55 * A * u[0] + 0.5, Y0 + 0.55 * A * u[1], "$a$ = %.1f" % A, color=BLUE, fontsize=9)
ax.text(X0 + 0.55 * B * v[0] - 0.3, Y0 + 0.55 * B * v[1] - 0.6, "$b$ = %.1f" % B, color=BLUE, fontsize=9, ha="right")
ax.plot([X0, X0 + 4.2], [Y0, Y0], color=MUTED, lw=0.8, ls=":")
ax.add_patch(Arc((X0, Y0), 5.0, 5.0, angle=0, theta1=0, theta2=np.degrees(PHI), color=INK, lw=0.9))
ax.text(X0 + 2.9, Y0 + 1.6, "$\\varphi$", fontsize=10, color=INK)
ax.plot(X0, Y0, "o", color=BLUE, ms=4); ax.text(X0 - 0.4, Y0 - 0.7, "($x_0, y_0$)", color=BLUE, ha="right", fontsize=8.5)
ax.plot(XS, YS, "+", color=ORANGE, ms=9)
ax.text(-9.6, 10.8, "dashed: $R$, the disc of Allen's radius $r_s$ = %.2f um\nabout Allen's centre (+)" % RS, fontsize=7.8, color=ORANGE)
ax.set_xlim(-10, 10); ax.set_ylim(12, -10); ax.set_aspect("equal")
ax.set_xlabel("$x$ (um)"); ax.set_ylabel("$y$ (um, image rows downward)")
ax.set_title("(a) top view: what one plane shows", fontsize=9, loc="left")
# side view: the section through the centre along the long axis u
bx.add_patch(Ellipse((0, Z0), 2 * A, 2 * Cc, fc="#e6f0fc", ec=BLUE, lw=1.6))
u1 = 4.2; q1 = 1 - (u1 / A) ** 2; h1 = Cc * np.sqrt(q1)
bx.add_patch(plt.Rectangle((u1 - 0.18, Z0 - h1), 0.36, 2 * h1, fc=BLUE, ec="none", alpha=0.55))
bx.annotate("", (u1 + 0.9, Z0 + h1), (u1 + 0.9, Z0 - h1), arrowprops=dict(arrowstyle="<->", color=INK, lw=0.9))
bx.text(7.5, Z0 - 0.6, "column length\n$2c\\sqrt{q}$ = %.2f um\nat $u$ = %.1f um" % (2 * h1, u1), fontsize=8, color=INK, va="center")
bx.plot([0, A], [Z0, Z0], color=BLUE, lw=1.0, ls=":"); bx.plot([0, 0], [Z0, Z0 + Cc], color=BLUE, lw=1.0, ls=":")
bx.text(A / 2 - 1.0, Z0 + 0.9, "$a$ = %.1f" % A, color=BLUE, ha="center"); bx.text(-0.3, Z0 + 0.8 * Cc, "$c$ = %.1f" % Cc, color=BLUE, ha="right")
bx.plot(0, Z0, "o", color=BLUE, ms=4); bx.text(-0.4, Z0 - 0.35, "$z_0$ = %.2f" % Z0, color=BLUE, fontsize=8.5, ha="right")
bx.axhline(ZS, color=ORANGE, lw=0.9, ls="--"); bx.text(-11.6, ZS - 0.25, "traced depth $z_s$ = %.2f (the start)" % ZS, color=ORANGE, fontsize=7.8)
bx.annotate("", (-10.6, Z0 + 6.5), (-10.6, Z0 + 2.5), arrowprops=dict(arrowstyle="->", color=MUTED))
bx.text(-10.2, Z0 + 4.6, "light", color=MUTED, fontsize=8)
bx.text(-11.6, Z0 - 6.3, "inside: absorption $\\mu$ = %.3f /um everywhere\n(uniform shading); a column at $q$ = %.2f absorbs\n"
        "1 - exp(-2$\\mu c\\sqrt{q}$) = %.2f of the light" % (MU, q1, 1 - np.exp(-2 * MU * Cc * np.sqrt(q1))),
        fontsize=7.8, color=INK, va="top")
bx.set_xlim(-12, 12); bx.set_ylim(Z0 + 8.5, Z0 - 8.5); bx.set_aspect("equal")
bx.set_xlabel("$u$, along the long axis (um)"); bx.set_ylabel("depth $z$ (um, stage units, downward)")
bx.set_title("(b) side view: what no single plane shows", fontsize=9, loc="left")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figN2_ellipsoid_parameters.png")); plt.close(fig)

# ======================================================================= figN3: D_k, R, tau_k, B_k
s = C.setup()
left, top, width, height = s["crop"]
zpl = s["ks_all"][s["idx"]] * C.DZ
tau = C.SomaRenderer((left, top, width, height), C.RES0, zpl, PILOT, C.Q, C.LIGHT).render(TRUTH, 0.79)
rng = np.random.default_rng(529)
D = 110.0 * tau + rng.normal(0.0, 3.0, tau.shape)
reg = s["region"]
Bk, _ = vp_scale(D, tau, reg)
picks = [int(np.argmin(np.abs(zpl - (Z0 + o)))) for o in (-4.0, 0.0, 4.0)]
ext = [left * C.RES0, (left + width) * C.RES0, (top + height) * C.RES0, top * C.RES0]
xg = (left + np.arange(width) + 0.5) * C.RES0
row = int(np.argmin(np.abs((top + np.arange(height) + 0.5) * C.RES0 - Y0)))
inR = reg[row]
fig, axs = plt.subplots(2, 3, figsize=(10.6, 6.6), gridspec_kw={"height_ratios": [1.15, 1]})
for j, k in enumerate(picks):
    a = axs[0, j]
    a.imshow(D[k], cmap="gray", vmin=50, vmax=120, extent=ext)
    a.add_patch(Circle((XS, YS), RS, fc="none", ec=ORANGE, lw=1.1, ls="--"))
    a.axhline((top + row + 0.5) * C.RES0, color=BLUE, lw=0.8)
    a.set_xlim(-13, 13); a.set_ylim(13, -13)
    a.set_title("$D_k$, plane at $z_k - z_0$ = %+.2f um" % (zpl[k] - Z0), fontsize=9)
    a.set_xlabel("$x$ (um)")
    b = axs[1, j]
    b.plot(xg, D[k][row], ".", color="#9a9a9a", ms=2.2, label="$D_k$ along the line (gl)")
    b.plot(xg, Bk[k] * tau[k][row], color=BLUE, lw=1.6, label="$\\hat B_k\\,\\tau_k$, the model (gl)")
    b.axhline(Bk[k], color=INK, lw=0.9, ls="--", label="$\\hat B_k$ = %.2f gl" % Bk[k])
    xr = xg[inR]
    b.axvspan(xr.min(), xr.max(), color=ORANGE, alpha=0.12, lw=0, label="pixels in $R$ (used)")
    b.set_xlim(-13, 13); b.set_ylim(45, 125); b.set_xlabel("$x$ (um), along the blue line")
    b.text(-12.5, 48, "$\\hat B_k$ = %.2f gl" % Bk[k], fontsize=8, color=INK)
axs[0, 0].set_ylabel("$y$ (um)"); axs[1, 0].set_ylabel("grey level (gl)")
h, l = axs[1, 1].get_legend_handles_labels()
fig.legend(h, l[:2] + ["$\\hat B_k$: the plane's brightness without the soma, solved on $R$", l[3]], loc="lower center", ncol=4,
           frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(os.path.join(OUT, "figN3_data_model_brightness.png")); plt.close(fig)
print("figN3: planes", [round(float(zpl[k] - Z0), 2) for k in picks], "B_hat", [round(float(Bk[k]), 3) for k in picks],
      "true B 110; R pixels", int(reg.sum()))

# ======================================================================= figN4: sigma_r and gamma
fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.4, 3.9), gridspec_kw={"width_ratios": [1, 1.3]})
dn = np.linspace(0, 1.2, 400)
kd, ks = np.array(PILOT.kernel_table_delta_um), np.array(PILOT.kernel_table_sigma_um)
cal = with_kernel(PILOT, 0.258)
ax.plot(dn, sigma_r(dn, PILOT), color=BLUE, lw=2, label="Cell 25's kernel ($\\sigma_r(0)$ = 0.080 um, 'fixed')")
ax.plot(dn, sigma_r(dn, cal), color=BLUE, lw=1.2, ls=":", label="KERNEL_WIDTH 'calibrated' ($\\sigma_r(0)$ = 0.258 um)")
ax.plot(kd, ks, "o", color=BLUE, ms=5)
for i in range(3):
    sl = (ks[i + 1] - ks[i]) / (kd[i + 1] - kd[i])
    ax.text((kd[i] + kd[i + 1]) / 2 + (0.02, 0.02, -0.16)[i], (ks[i] + ks[i + 1]) / 2 - (0.06, 0.06, 0.02)[i] - (0, 0, 0.03)[i], "slope %.2f" % sl, fontsize=7.5, color=MUTED)
ax.axvline(0.84, color=MUTED, lw=0.6, ls="--"); ax.text(0.86, 0.05, "table ends", fontsize=7.5, color=MUTED)
ax.set_xlim(0, 1.2); ax.set_ylim(0, 0.95)
ax.set_xlabel("defocus |$\\delta$| (um)"); ax.set_ylabel("blur SD $\\sigma_r(\\delta)$ (um)")
ax.set_title("(a) near focus: the pilot's table (Cell 7)", fontsize=9, loc="left"); ax.legend(frameon=False, fontsize=7.3, loc="upper left")
df = np.linspace(0, 12, 600)
for g, col, ls, lab in ((0.95, MUTED, "--", "$\\gamma$ = 0.95"), (0.79, BLUE, "-", "$\\gamma$ = 0.79, held (D-240)"),
                        (0.60, ORANGE, "--", "$\\gamma$ = 0.60 (near the table's last slope, 0.61)")):
    bx.plot(df, sigma_r(df, dataclasses.replace(PILOT, kernel_continuation_slope=g)), color=col, lw=2 if g == 0.79 else 1.3,
            ls=ls, label=lab)
bx.axvline(0.84, color=MUTED, lw=0.6, ls="--")
bx.set_xlim(0, 12); bx.set_ylim(0, 11)
bx.set_xlabel("defocus |$\\delta$| = |$\\zeta_j - z_k$| (um)"); bx.set_ylabel("blur SD $\\sigma_r(\\delta)$ (um)")
bx.set_title("(b) far from focus: a straight line of slope $\\gamma$", fontsize=9, loc="left")
bx.legend(frameon=False, fontsize=7.6, loc="upper left", title="dashed: worlds of the sensitivity check", title_fontsize=7.3)
bx.text(6.3, 1.0, "at |$\\delta$| = 8 um: %.1f, %.1f, %.1f um" % tuple(float(sigma_r(8.0, dataclasses.replace(PILOT, kernel_continuation_slope=g))) for g in (0.60, 0.79, 0.95)),
        fontsize=7.6, color=INK)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figN4_blur_law.png")); plt.close(fig)
