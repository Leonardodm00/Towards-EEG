"""Figures for the soma-fit explanation (standalone; not pipeline code). Run: python3 figures.py OUT_DIR
Everything plotted is computed here from the pipeline's own functions on a simulated soma (common.CONFIG) or
from the configuration; nothing is read from the user's data. Schematic elements are labelled schematic."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
from scipy import integrate
import common as C
from allen_diameter.model.kernel import sigma_r

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, INK, MUTED = "#257de0", "#d9730d", "#191919", "#666666"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "figure.dpi": 150})
T = C.CONFIG["truth"]
s = C.setup()
left, top, width, height = s["crop"]
ext = [left * C.RES0, (left + width) * C.RES0, (top + height) * C.RES0, top * C.RES0]
xs, ys, zs = C.CONFIG["traced_centre_um"]
rs = C.CONFIG["traced_radius_um"]
res = {}

# ---- Fig 1: the forward model (side view, schematic to scale) and the blur law (plotted from the configuration)
fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.6, 3.9), gridspec_kw={"width_ratios": [1.25, 1]})
z0, c, a = T["z0"], T["c"], T["a"]
for k in s["ks_all"][s["idx"]]:
    ax.axhline(k * C.DZ, color="#c9c9c9", lw=0.5, zorder=0)
ax.add_patch(Ellipse((0, z0), 2 * a, 2 * c, fc="#bcd6f5", ec=BLUE, lw=1.5))
zj, zk = z0 + 2.5, z0 - 4.2                                        # one slab and one plane, for the blur cone
half_chord = a * np.sqrt(1 - ((zj - z0) / c) ** 2)
ax.plot([-half_chord, half_chord], [zj, zj], color=ORANGE, lw=2.5)
ax.axhline(zk, color=INK, lw=1.2)
sg = float(sigma_r(zj - zk, C.RCFG))
ax.fill([1.5, 1.5 - 2 * sg, 1.5 + 2 * sg], [zj, zk, zk], color=ORANGE, alpha=0.25, lw=0)
ax.annotate("one slab at depth $\\zeta_j$", (half_chord, zj), (9.3, zj + 1.6), color=INK, fontsize=8,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
ax.annotate("plane $k$ at depth $z_k$", (-9, zk), (-13.5, zk - 2.2), color=INK, fontsize=8,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
ax.text(1.5, zk - 0.5, "blur: $\\pm2\\,\\sigma_r(\\zeta_j - z_k)$ = $\\pm$%.1f um" % (2 * sg), ha="center", va="bottom",
        fontsize=8, color=INK)
ax.annotate("", (-12.2, z0 + 8), (-12.2, z0 + 3), arrowprops=dict(arrowstyle="->", color=MUTED))
ax.text(-11.8, z0 + 5.5, "light\n(toward +z)", fontsize=8, color=MUTED, va="center")
ax.plot([0, a], [z0, z0], color=BLUE, lw=0.8, ls=":"); ax.plot([0, 0], [z0, z0 + c], color=BLUE, lw=0.8, ls=":")
ax.text(a / 2, z0 + 0.25, "$a$", color=BLUE, ha="center", va="top"); ax.text(0.25, z0 + c / 2, "$c$", color=BLUE)
ax.plot(0, z0, "o", color=BLUE, ms=3); ax.text(-0.4, z0 - 0.3, "$z_0$", color=BLUE, ha="right")
ax.set_xlim(-14, 14); ax.set_ylim(zs + 11.5, zs - 11.5); ax.set_aspect("equal")
ax.set_xlabel("in-plane position along the long axis $u$ (um)"); ax.set_ylabel("depth $z$ (um, stage units)")
ax.set_title("side view: the ellipsoid, its slabs, the 35 fitted planes (to scale)", fontsize=9, loc="left")
d = np.linspace(0, 12, 600)
bx.plot(d, sigma_r(d, C.RCFG), color=BLUE, lw=2)
kn = np.array(C.RCFG.kernel_table_delta_um); kv = np.array(C.RCFG.kernel_table_sigma_um)
bx.plot(kn, kv, "o", color=BLUE, ms=4)
bx.axvline(kn[-1], color=MUTED, lw=0.6, ls="--")
bx.text(kn[-1] + 0.25, 0.3, "table ends at %.2f um" % kn[-1], fontsize=8, color=MUTED)
bx.text(5.2, 3.1, "far part: straight line,\nslope $\\gamma$ = %.2f, held" % C.SLOPE0, fontsize=8, color=INK)
bx.set_xlabel("defocus $|\\delta|$ = $|\\zeta_j - z_k|$ (um)"); bx.set_ylabel("blur SD $\\sigma_r(\\delta)$ (um)")
bx.set_title("the blur law (configuration; not fitted to this stack)", fontsize=9, loc="left")
bx.set_xlim(0, 12); bx.set_ylim(0, None)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig1_forward_model.png")); plt.close(fig)

# ---- Fig 2: twin cases -- a compact and a tall soma of equal central absorbance, same planes, same grey scale
offs = np.array([-8.0, -4.0, 0.0, 4.0, 8.0])
zpl = T["z0"] + offs
render = s["make"](zpl)
th_c = C.theta_of(T)
tall = dict(T); tall["c"] = 2 * T["c"]; tall["mu"] = T["mu"] / 2          # equal 2 mu c
th_t = C.theta_of(tall)
B = C.CONFIG["B_gl"]
Ic, It = B * render(th_c, C.SLOPE0), B * render(th_t, C.SLOPE0)
vmin = float(min(Ic.min(), It.min()))
fig, axs = plt.subplots(3, 5, figsize=(10.4, 6.4), sharex=True, sharey=True)
dmax = float(np.abs(It - Ic).max())
for j in range(5):
    for i, (img, lab) in enumerate(((Ic, "compact: c = %.0f um" % T["c"]), (It, "tall: c = %.0f um" % tall["c"]))):
        axs[i, j].imshow(img[j], cmap="gray", vmin=vmin, vmax=B, extent=ext)
        if j == 0:
            axs[i, j].set_ylabel(lab + "\ny (um)")
    im = axs[2, j].imshow(It[j] - Ic[j], cmap="RdGy_r", vmin=-dmax, vmax=dmax, extent=ext)
    for i in range(3):
        axs[i, j].add_patch(Circle((xs, ys), rs, fc="none", ec=ORANGE, lw=0.9, ls="--"))
        axs[i, j].set_xticks([-10, 0, 10]); axs[i, j].set_yticks([-10, 0, 10])
    axs[0, j].set_title("$z_k - z_0$ = %+.0f um" % offs[j], fontsize=9)
    axs[2, j].set_xlabel("x (um)")
axs[2, 0].set_ylabel("tall minus compact\ny (um)")
cb = fig.colorbar(im, ax=axs[2, :], fraction=0.025, pad=0.01); cb.set_label("difference (gl)")
fig.suptitle("simulated planes, grey scale %.0f-%.0f gl in rows 1-2; dashed: the fit's region R (Allen's radius %.2f um)"
             % (vmin, B, rs), fontsize=9, x=0.02, ha="left")
fig.savefig(os.path.join(OUT, "fig2_twin_somas.png"), bbox_inches="tight"); plt.close(fig)
reg = s["region"]
res["fig2"] = dict(max_abs_diff_gl=dmax, rms_diff_on_R_gl=[float(np.sqrt(np.mean((It[j] - Ic[j])[reg] ** 2))) for j in range(5)],
                   offsets_um=offs.tolist(), central_dip_gl=float(B - Ic[2].min()))

# ---- Fig 3: the membrane area against the depth semi-axis (Legendre's form), checked by numerical quadrature
def area_quad(a, b, c):
    f = lambda th, ph: np.sin(th) * np.sqrt(b * b * c * c * np.sin(th) ** 2 * np.cos(ph) ** 2
                                             + a * a * c * c * np.sin(th) ** 2 * np.sin(ph) ** 2
                                             + a * a * b * b * np.cos(th) ** 2)
    return integrate.dblquad(f, 0, 2 * np.pi, 0, np.pi, epsabs=1e-9, epsrel=1e-10)[0]
pa, pb = 8.08, 6.52                                   # the pilot's free-slope fit (handoff 14.6), in-plane semi-axes
cs = np.linspace(1, 15, 141)
S = np.array([C.soma_fit.ellipsoid_area(pa, pb, x) for x in cs])
chk = [(x, C.soma_fit.ellipsoid_area(pa, pb, x), area_quad(pa, pb, x)) for x in (2.0, 7.26, 15.0)]
res["area_check"] = [dict(c=x, legendre=l, quadrature=q, rel=abs(l / q - 1)) for x, l, q in chk]
lin = np.polyfit(cs[cs >= 5], S[cs >= 5], 1)
res["area_slope_um2_per_um"] = float(lin[0])
res["area_lin_maxdev_rel"] = float(np.max(np.abs(np.polyval(lin, cs[cs >= 5]) / S[cs >= 5] - 1)))
allen = 4 * np.pi * rs ** 2
fig, ax = plt.subplots(figsize=(5.6, 3.6))
ax.plot(cs, S, color=BLUE, lw=2)
ax.axhline(allen, color=MUTED, lw=0.8, ls="--"); ax.text(8.2, allen - 48, "Allen's sphere, r = %.2f um: %.0f um$^2$" % (rs, allen), fontsize=8, color=MUTED)
for x in (7.26, 15.0):
    v = C.soma_fit.ellipsoid_area(pa, pb, x)
    ax.plot(x, v, "o", color=BLUE, ms=5); ax.annotate("c = %.2f um: %.0f um$^2$" % (x, v), (x, v), (x - 5.4, v + 70), fontsize=8, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
ax.set_xlabel("depth semi-axis $c$ (um)"); ax.set_ylabel("membrane area $S(a, b, c)$ (um$^2$)")
ax.set_title("a = %.2f um, b = %.2f um held (the pilot's fitted in-plane axes)" % (pa, pb), fontsize=9, loc="left", pad=12)
ax.set_xlim(0, 15.5); ax.set_ylim(0, 1300)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_area_vs_c.png")); plt.close(fig)

# ---- Fig 4: the region R and the crop on the in-focus plane (top view)
fig, ax = plt.subplots(figsize=(4.6, 4.3))
ax.imshow(Ic[2], cmap="gray", vmin=vmin, vmax=B, extent=ext)
ax.add_patch(Circle((xs, ys), rs, fc="none", ec=ORANGE, lw=1.6))
ax.add_patch(Ellipse((T["x0"], T["y0"]), 2 * T["a"], 2 * T["b"], angle=np.degrees(T["phi"]), fc="none", ec=BLUE, lw=1.2, ls="--"))
ax.plot(xs, ys, "+", color=ORANGE, ms=8)
ax.text(xs - 1, ys - 10.5, "solid: region R, the disc of Allen's radius (%.2f um)" % rs, color=ORANGE, ha="center", fontsize=8)
ax.text(xs - 1, ys + 11.5, "dashed: outline of the simulated soma (a %.1f, b %.1f um)" % (T["a"], T["b"]), color=BLUE, fontsize=8, ha="center")
ax.set_xlabel("x (um)"); ax.set_ylabel("y (um)")
ax.set_title("crop = what the model draws (half-width %.0f um)" % s["half"], fontsize=9, loc="left")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig4_region_crop.png")); plt.close(fig)
# fraction of R inside the simulated outline
X = (left + np.arange(width)) * C.RES0; Y = (top + np.arange(height)) * C.RES0
dx, dy = X[None, :] - T["x0"], Y[:, None] - T["y0"]
u = dx * np.cos(T["phi"]) + dy * np.sin(T["phi"]); v = -dx * np.sin(T["phi"]) + dy * np.cos(T["phi"])
ins = (u / T["a"]) ** 2 + (v / T["b"]) ** 2 < 1
res["fig4"] = dict(n_region_px=int(reg.sum()), frac_R_inside_outline=float((reg & ins).sum() / reg.sum()),
                   frac_outline_in_R=float((reg & ins).sum() / ins.sum()))
json.dump(res, open(os.path.join(OUT, "figures_numbers.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
