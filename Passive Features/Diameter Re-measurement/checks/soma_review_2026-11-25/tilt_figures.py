"""Figures for the explanation of the tilted background solved inside the soma fit (D-254), 2026-11-25.
Run (after tilt_check.py and tilt_cost_check.py): OMP_NUM_THREADS=1 python3 tilt_figures.py
Reads tilt_check.json, tilt_cost_check.json and taus_cost.npz (the renders of tilt_cost_check.py); writes
figT1_nested_regression.png (schematic), figT2_tilt_vs_shift.png and figT3_region_tradeoff.png (plotted), and prints the
share of the shift column that the per-plane tilts can mimic (rho^2) by region radius."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import common as C

BLUE, ORANGE, AQUA = "#257de0", "#f47230", "#00b894"          # dataviz reference palette, slots 1-3 (validated)
INK, MUTED, GRID = "#191919", "#666666", "#d0d0d0"
FILL = {"measured": "#e6f0fc", "configured": "#feeee5", "computed": "#e0f6f1"}
EDGE = {"measured": BLUE, "configured": ORANGE, "computed": AQUA}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "figure.dpi": 150})
TH = np.array([0.6, -0.4, 30.96, 7.0, 5.6, 4.5, 1.13, 0.87 / 9.0])
H_X0, B0, GX = 0.05, 110.0, 0.15


# ---------------------------------------------------------------- figure T1: the nested regression (schematic)
def box(ax, x, y, w, h, text, kind, fs=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=0.8", fc=FILL[kind], ec=EDGE[kind],
                                lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, linespacing=1.35)


def arrow(ax, p, q, text=None, off=(0, 0), rad=0.0):
    ax.annotate("", q, p, arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.1, connectionstyle="arc3,rad=%g" % rad))
    if text:
        ax.text((p[0] + q[0]) / 2 + off[0], (p[1] + q[1]) / 2 + off[1], text, fontsize=7.2, color=MUTED, ha="center")


def fig_t1(path):
    fig, ax = plt.subplots(figsize=(10.6, 5.6))
    ax.set_xlim(0, 127); ax.set_ylim(0, 58); ax.axis("off")
    box(ax, 1, 40, 21, 11, "trial soma $\\theta$\n$(x_0, y_0, z_0, a, b, c, \\varphi, \\mu)$\n8 numbers, shared by\nall 35 planes"
        "\n($c$ held on a grid: 7)", "computed")
    box(ax, 27, 40, 23, 11, "render (Block 45)\n$\\tau_k(x, y; \\theta)$, $k = 1..35$\nslabs, absorption, blur\n"
        "$\\sigma_r(\\zeta_j - z_k)$, far slope $\\gamma$ held", "computed")
    box(ax, 56, 37, 47, 17, "inner step, each plane $k$ on its own (x 35)\n"
        "$M_k(\\theta) = [\\,\\tau_k,\\; x\\,\\tau_k,\\; y\\,\\tau_k\\,]$ on R  ($n \\times 3$)\n"
        "$\\hat\\beta_k(\\theta) = \\arg\\min_\\beta \\|d_k - M_k(\\theta)\\,\\beta\\|^2$  (3 numbers, exact)\n"
        "$r_k(\\theta) = d_k - M_k(\\theta)\\,\\hat\\beta_k(\\theta)$  ($n$ residuals, gl)", "computed", fs=8.2)
    box(ax, 107, 45, 19, 8, "data $d_k$ on R, gl\n35 planes of the stack", "measured")
    box(ax, 107, 33, 19, 8, "region R: disc of\nradius $r_R$ minus the\nbranch capsules", "configured")
    box(ax, 56, 18, 47, 11, "stack the 35 planes\n$r(\\theta) = (r_1, \\dots, r_{35})$, length $35\\,n$\n"
        "$Q(\\theta) = \\sum_k \\|r_k(\\theta)\\|^2$  (gl$^2$)", "computed")
    box(ax, 13, 16, 35, 15, "outer step: trf (scipy least_squares)\n"
        "$J = \\partial r / \\partial \\theta$ by forward differences:\n"
        "each column re-renders all planes and\nre-solves every $\\hat\\beta_k$  ($35\\,n \\times 8$)\n"
        "step within the bounds, new $\\theta$", "computed")
    box(ax, 1, 2, 24, 9, "converged: $\\hat\\theta$\n$\\to S(\\hat a, \\hat b, c) \\to r_{\\rm eq}$ (Block 47)", "computed")
    box(ax, 29, 2, 31, 9, "$\\hat\\beta_k(\\hat\\theta)$: the fitted backgrounds\n$\\to$ check against the far\n"
        "background on near-focus planes", "computed")
    arrow(ax, (22.5, 45.5), (26.5, 45.5))
    arrow(ax, (50.5, 45.5), (55.5, 45.5), "$\\tau_k$", off=(0, 1.3))
    arrow(ax, (106.5, 49), (103.5, 48))
    arrow(ax, (106.5, 37), (103.5, 40.5))
    arrow(ax, (79.5, 36.5), (79.5, 29.5), "$r_k$", off=(2.5, 0))
    arrow(ax, (55.5, 23.5), (48.5, 23.5), "$r(\\theta)$", off=(0, 1.3))
    arrow(ax, (25, 31.5), (11, 39.5), "new $\\theta$", off=(-3.5, 0), rad=-0.15)
    arrow(ax, (20, 15.5), (13, 11.5))
    arrow(ax, (38, 15.5), (42, 11.5))
    ax.text(1, 56, "schematic. Fill: blue = measured, orange = configured, aqua = computed. Only $\\theta$ is the solver's;"
            " the 105 background coefficients are recomputed inside every evaluation of $r(\\theta)$.", fontsize=7.6,
            color=MUTED)
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- shared geometry
def grid():
    s = C.setup(); left, top, width, height = s["crop"]
    X = np.broadcast_to(((left + np.arange(width) + 0.5) * C.RES0)[None, :], (height, width))
    Y = np.broadcast_to(((top + np.arange(height) + 0.5) * C.RES0)[:, None], (height, width))
    return X, Y, s["ks_all"][s["idx"]] * C.DZ


def rho2(tau, dtau_x0, X, Y, reg):
    """share of the shift column's energy (after each plane's constant column is removed) that the per-plane tilt
    columns x tau_k, y tau_k can reproduce, all planes together; dimensionless in [0, 1]."""
    num = den = 0.0
    B = B0 + GX * X[reg]
    for k in range(tau.shape[0]):
        t = tau[k][reg]
        j = B * dtau_x0[k][reg]
        cols = np.stack([X[reg] * t, Y[reg] * t], axis=1)
        j = j - t * (t @ j) / (t @ t)
        cols = cols - np.outer(t, t @ cols) / (t @ t)
        coef = np.linalg.lstsq(cols, j, rcond=None)[0]
        p = cols @ coef
        num += p @ p; den += j @ j
    return num / den


def fig_t2(path, taus, se):
    X, Y, zpl = grid()
    tau, dtau = taus[0], (taus[1] - taus[2]) / (2 * H_X0)
    k = int(np.argmin(np.abs(zpl - TH[2])))
    row = int(np.argmin(np.abs(Y[:, 0] - TH[1])))
    x = X[row]
    s_shift = (B0 + GX * x) * dtau[k][row]
    s_tilt = x * tau[k][row]
    fig, axs = plt.subplots(1, 2, figsize=(10.2, 3.6), sharey=True)
    for ax, r, lab in zip(axs, (5.97, 10.0), ("(a) $r_R$ = 5.97 um, Allen's radius (D-241.1)", "(b) $r_R$ = 10 um")):
        half = np.sqrt(max(r ** 2 - Y[row, 0] ** 2, 0.0))
        inside = np.abs(x) <= half
        t_in = tau[k][row][inside]
        def drop_const(v):
            return v - tau[k][row] * (t_in @ v[inside]) / (t_in @ t_in)
        sh = drop_const(s_shift); sh = sh / np.abs(sh[inside]).max()
        ti = drop_const(s_tilt)
        alpha = (ti[inside] @ sh[inside]) / (ti[inside] @ ti[inside])
        ax.axvspan(-half, half, color="#f2f2f2", lw=0)
        ax.axhline(0, color=GRID, lw=0.8)
        ax.plot(x, tau[k][row], color=MUTED, lw=1.0, ls="--", label="$\\tau_k$ (transmittance)")
        ax.plot(x, sh, color=BLUE, lw=2.0, label="shift column $\\partial m_k / \\partial x_0$ (scaled)")
        ax.plot(x, alpha * ti, color=ORANGE, lw=2.0, label="tilt column $x\\,\\tau_k$, its best\nimitation of the shift on R")
        reg = (X ** 2 + Y ** 2) <= r * r
        ax.text(-13.6, 1.62, "share of the shift column the tilts can mimic\n(whole R, all 35 planes): $\\rho^2$ = %.2f\n"
                "SE($x_0$) with tilt / with constant = %.2f" % (rho2(tau, dtau, X, Y, reg), se["%.2f" % r]),
                fontsize=7.6, color=INK, va="top")
        ax.set_title(lab, fontsize=9, loc="left"); ax.set_xlabel("$x$ along the line $y = y_0$ (um from the traced centre)")
        ax.set_xlim(-14, 14); ax.set_ylim(-1.5, 1.7)
    axs[0].set_ylabel("dimensionless: shift column scaled to\nmax |.| = 1 on R; per-plane constant removed")
    axs[1].legend(frameon=False, fontsize=7.4, loc="lower right")
    fig.suptitle("plotted: simulated soma (a 7.0, b 5.6, c 4.5 um), plane %.2f um (centre depth %.2f um); grey band = R on this line"
                 % (zpl[k], TH[2]), fontsize=8, color=MUTED, x=0.01, ha="left")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig_t3(path, first, cost):
    radii = sorted(float(r) for r in first["results"])
    keys = ["%.2f" % r for r in radii]
    fig, axs = plt.subplots(1, 3, figsize=(10.6, 3.4))
    a, b, c = axs
    for q, col, lab in (("x0", BLUE, "$x_0$"), ("y0", ORANGE, "$y_0$")):
        v = [first["results"][k]["tilt"]["se_" + q] / first["results"][k]["const"]["se_" + q] for k in keys]
        a.plot(radii, v, "o-", color=col, lw=2, ms=6, label=lab)
    a.axhline(1, color=GRID, lw=0.8)
    a.set_ylabel("SE with tilt / SE with constant"); a.set_title("(a) the tilt's cost in precision", fontsize=9, loc="left")
    a.legend(frameon=False, fontsize=8)
    W = cost["worlds"]
    for w, col, lab in (("g095", BLUE, "world's far slope 0.95"), ("g060", ORANGE, "world's far slope 0.60")):
        b.plot(radii, [100 * W[w]["by_radius"][k]["tilt"]["bias_S_rel"] for k in keys], "o-", color=col, lw=2, ms=6, label=lab)
        c.plot(radii, [W[w]["by_radius"][k]["tilt"]["mismatch_near_mean_gl"] for k in keys], "o-", color=col, lw=2, ms=6,
               label=lab)
    c.plot(radii, [W["exact"]["by_radius"][k]["tilt"]["mismatch_near_mean_gl"] for k in keys], "o-", color=AQUA, lw=2, ms=6,
           label="world's far slope 0.79 (model exact)")
    for ax in (b, c):
        ax.axhline(0, color=GRID, lw=0.8)
    b.set_ylabel("bias of $S$ (%), far slope held at 0.79")
    b.set_title("(b) the held slope's leverage on $S$", fontsize=9, loc="left"); b.legend(frameon=False, fontsize=7.6)
    c.set_ylabel("$\\hat b_{0k}(\\hat\\theta) - b^{\\rm far}_{0k}$ (gl), mean of 4 planes")
    c.set_title("(c) the background check's signal", fontsize=9, loc="left"); c.set_ylim(-2.6, 5.6)
    c.legend(frameon=False, fontsize=7.2, loc="upper right")
    for ax in axs:
        ax.set_xlabel("radius $r_R$ of the region (um)"); ax.set_xticks(radii)
    fig.suptitle("plotted: simulated soma (a 7.0, b 5.6, c 4.5 um), background 110 + 0.15 x gl; (a) noise SD 3 gl, "
                 "linearised; (b, c) noise-free, one Gauss-Newton step; tilted projection", fontsize=8, color=MUTED,
                 x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    first = json.load(open("tilt_check.json")); cost = json.load(open("tilt_cost_check.json"))
    taus = np.load("taus_cost.npz")["taus"]
    se = {k: v["tilt"]["se_x0"] / v["const"]["se_x0"] for k, v in first["results"].items()}
    fig_t1("figT1_nested_regression.png")
    fig_t2("figT2_tilt_vs_shift.png", taus, se)
    fig_t3("figT3_region_tradeoff.png", first, cost)
    X, Y, _ = grid()
    dtau = (taus[1] - taus[2]) / (2 * H_X0)
    for r in (5.97, 8.0, 10.0, 12.0, 14.0):
        p = rho2(taus[0], dtau, X, Y, (X ** 2 + Y ** 2) <= r * r)
        print("r_R %5.2f um: rho^2 %.3f -> 1/sqrt(1 - rho^2) = %.2f (x0 alone); full check's SE ratio %.2f"
              % (r, p, 1 / np.sqrt(1 - p), se["%.2f" % r]))
