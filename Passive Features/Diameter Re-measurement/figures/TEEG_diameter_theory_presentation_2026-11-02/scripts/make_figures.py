"""make_figures.py -- Figures 1-5 of the explanation of diffraction and coherence
(TEEG, 2026-11-02). Plotting only: the physics is optics_model.py, the checked
numbers come from check_optics_claims.py (run it first; it writes
check_results.json). Run: python3 make_figures.py  -> fig1..fig5 *.png here.
Colours: the dataviz reference palette (categorical slots 1-3 in order; the
blue ordinal ramp for families ordered by S; the blue sequential ramp for maps).
"""
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import optics_model as om  # noqa: E402

CFG = dict(lam=0.55, n=1.515, na_o=1.4, n_v=8192, L_v=80.0, d_tube=1.0, dpi=200)

INK, INK2, MUTED, AXIS, SURF, WASH = "#0b0b0b", "#52514e", "#898781", "#c3c2b7", "#fcfcfb", "#f0efec"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
ORD = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("blue_seq", [SURF, "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                                                     "#256abf", "#184f95", "#0d366b"])
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2, "axes.titlesize": 9.5, "axes.titlecolor": INK, "figure.facecolor": SURF,
    "axes.facecolor": SURF, "savefig.facecolor": SURF, "lines.linewidth": 1.6, "lines.solid_capstyle": "round",
    "legend.frameon": False, "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False})


def footer(fig, text, y=0.015):
    fig.text(0.01, y, text, fontsize=7.2, color=INK2, ha="left", va="bottom", wrap=True)


# ------------------------------------------------------------------ Figure 1
def fig1(cfg, res, path):
    fig = plt.figure(figsize=(12, 4.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.12, 1.35], wspace=0.62, left=0.05, right=0.98,
                          top=0.86, bottom=0.24)
    s_o = cfg["na_o"] / cfg["n"]
    th = np.linspace(0, 2 * np.pi, 400)
    ax = fig.add_subplot(gs[0])
    ax.fill(s_o * np.cos(th), s_o * np.sin(th), color=C1, alpha=0.12, lw=0)
    ax.plot(s_o * np.cos(th), s_o * np.sin(th), color=C1)
    ax.plot(np.cos(th), np.sin(th), color=AXIS, lw=0.8)
    ax.set_aspect("equal"); ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel(r"$s_x$ (direction sine)"); ax.set_ylabel(r"$s_y$")
    ax.text(0, 0.08, r"$P(\mathbf{s})=1$" + "\non " + r"$|\mathbf{s}|\leq s_{\rm o}=0.924$", ha="center",
            va="center", color=INK, fontsize=8.5)
    ax.text(0, -s_o - 0.13, r"edge $\leftrightarrow\ |\boldsymbol{\nu}|=\mathrm{NA_o}/\lambda=2.55\ \mu$m$^{-1}$",
            ha="center", va="top", color=INK2, fontsize=7.5)
    ax.text(0.72, 0.86, r"$|\mathbf{s}|=1$" + "\n(90°)", color=MUTED, fontsize=7.5, ha="left")
    ax.set_title("(a) pupil: the directions the\nobjective accepts (domain)", loc="left")

    ax = fig.add_subplot(gs[1])
    x = np.linspace(-0.8, 0.8, 641)
    X, Y = np.meshgrid(x, x)
    I2 = om.airy_intensity(np.hypot(X, Y), cfg)
    im = ax.imshow(np.log10(np.maximum(I2, 1e-4)), extent=[-0.8, 0.8, -0.8, 0.8], origin="lower",
                   cmap=SEQ, vmin=-4, vmax=0)
    ax.add_patch(Circle((0, 0), res["first_zero_um"], fill=False, ec=C2, lw=1.0))
    ax.text(0.2, 0.62, "first dark ring\n0.24 µm", color=INK, fontsize=7.5)
    ax.set_xlabel(r"$x$ (µm)"); ax.set_ylabel(r"$y$ (µm)")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(r"$\log_{10}\, I(\mathbf{x},0)/I(\mathbf{0},0)$", color=INK2, fontsize=8)
    cb.outline.set_edgecolor(AXIS)
    ax.set_title("(b) in-focus spot in the specimen\nplane (codomain), log scale", loc="left")

    ax = fig.add_subplot(gs[2])
    r = np.linspace(-0.8, 0.8, 1601)
    b = 2 * np.pi * cfg["na_o"] * np.abs(r) / cfg["lam"]
    amp = np.where(b > 0, 2 * om.special.j1(np.maximum(b, 1e-12)) / np.maximum(b, 1e-12), 1.0)
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.plot(r, amp, color=C1, label=r"amplitude $2J_1(\beta)/\beta$")
    ax.plot(r, amp**2, color=C2, label=r"intensity $[2J_1(\beta)/\beta]^2$")
    hw = res["fwhm_um"] / 2
    ax.plot([-hw, hw], [0.5, 0.5], color=INK, lw=0.9)
    for s in (-hw, hw):
        ax.plot([s, s], [0.46, 0.54], color=INK, lw=0.9)
    ax.text(-hw - 0.03, 0.5, "FWHM\n0.20 µm", ha="right", va="center", color=INK, fontsize=7.5)
    z = res["first_zero_um"]
    ax.plot([-z, z], [0, 0], "o", ms=4.5, color=INK, mec=SURF, mew=1.2)
    ax.annotate(r"first zeros $\pm0.61\,\lambda/\mathrm{NA_o}=\pm0.24$ µm", xy=(-z, 0), xytext=(-0.78, -0.25),
                fontsize=7.5, color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))
    ax.set_xlim(-0.8, 0.8); ax.set_ylim(-0.3, 1.08)
    ax.set_xlabel(r"$x$ (µm), through the centre"); ax.set_ylabel("relative to the centre")
    ax.legend(loc="upper right")
    ax.set_title(r"(c) profile; $\beta=2\pi\,\mathrm{NA_o}\,|\mathbf{x}|/\lambda$", loc="left")

    pa, pb = fig.axes[0].get_position(), fig.axes[1].get_position()
    x0, x1, ym = pa.x1 + 0.004, pb.x0 - 0.05, 0.5 * (pa.y0 + pa.y1)
    fig.add_artist(FancyArrowPatch((x0, ym), (x1, ym), transform=fig.transFigure, arrowstyle="-|>",
                                   mutation_scale=12, color=INK2, lw=1.0))
    fig.text(0.5 * (x0 + x1), ym + 0.03, "Fourier\ntransform\n(Eq. 1),\nthen |·|²", ha="center",
             va="bottom", fontsize=7.5, color=INK2)
    footer(fig, "Plotted from Eqs. (1)-(2): scalar Debye theory, aberration-free pupil uniform in direction "
                "sines (model level); lambda = 0.55 um, NA_o = 1.4, n_oil = 1.515. The 400-node quadrature of "
                "Eq. (1) matches the closed form to 1e-14 (check_optics_claims.py, D1). Script: make_figures.py.")
    fig.savefig(path, dpi=cfg["dpi"]); plt.close(fig)


# ------------------------------------------------------------------ Figure 2
def fig2(cfg, res, path):
    s_o = cfg["na_o"] / cfg["n"]
    tan_o = s_o / np.sqrt(1 - s_o**2)
    tan50 = res["tan_theta50"]
    d_qw = res["delta_qw_um"]
    r = np.linspace(0, 1.2, 241)
    dl = np.linspace(0, 0.84, 113)
    i00 = np.abs(om.psf_field(np.array([0.0]), 0.0, cfg))[0] ** 2
    M = np.array([np.abs(om.psf_field(r, d, cfg)) ** 2 for d in dl]) / i00
    x = np.concatenate([-r[:0:-1], r])
    Mx = np.concatenate([M[:, :0:-1], M], axis=1)
    D = np.concatenate([-dl[:0:-1], dl])
    MD = np.concatenate([Mx[:0:-1], Mx], axis=0)          # I(x, -delta) = I(x, delta) for this pupil

    fig = plt.figure(figsize=(12, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1], wspace=0.42, left=0.06, right=0.98,
                          top=0.87, bottom=0.25)
    ax = fig.add_subplot(gs[0])
    pm = ax.pcolormesh(x, D, np.log10(np.maximum(MD, 1e-3)), cmap=SEQ, vmin=-3, vmax=0, shading="auto")
    yy = np.array([-0.84, 0.0, 0.84])
    for sg in (-1, 1):
        ax.plot(sg * np.abs(yy) * tan50, yy, color=C2, lw=1.2)
        ax.plot(sg * np.abs(yy) * tan_o, yy, color=INK2, lw=0.8)
    for sg in (-1, 1):
        ax.axhline(sg * d_qw, color=INK, lw=0.7)
    ax.text(-1.17, d_qw + 0.02, "quarter-wave ±0.147 µm", fontsize=7.2, color=INK, va="bottom")
    bb = dict(boxstyle="round,pad=0.15", fc=SURF, ec="none", alpha=0.85)
    ax.text(0.6, 0.62, "rays: 50 %\nradius", fontsize=7.2, color=INK, ha="left", bbox=bb)
    ax.text(0.62, 0.2, r"rays: edge" + "\n" + r"$|\delta|\tan\vartheta_{\rm o}$", fontsize=7.2, color=INK, ha="left", bbox=bb)
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-0.84, 0.84)
    ax.set_xlabel(r"$x$ (µm)"); ax.set_ylabel(r"defocus $\delta$ (µm)")
    cb = fig.colorbar(pm, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label(r"$\log_{10}\, I(\mathbf{x},\delta)/I(\mathbf{0},0)$", color=INK2, fontsize=8)
    cb.outline.set_edgecolor(AXIS)
    ax.set_title("(a) the spot through focus (wave optics)\nwith the geometric cone drawn over it", loc="left")

    deltas = np.array(res["deltas"])
    ax = fig.add_subplot(gs[1])
    ax.axvspan(0, d_qw, color=WASH, lw=0)
    ax.plot(deltas, res["r50_wave"], color=C1, label="wave: 50 %-energy radius")
    hw = np.array([np.nan if x is None else x / 2 for x in res["fwhm_d"]])
    hw[deltas > res["core_defined_up_to_um"] + 1e-9] = np.nan     # one contiguous core regime from focus
    ax.plot(deltas, hw, color=C3, label="wave: core half width at half max")
    ax.plot(deltas, res["r50_geo"], color=C2, label=r"rays: 50 % radius $|\delta|\tan\vartheta_{50}$")
    last = res["core_defined_up_to_um"]
    ax.text(last + 0.01, hw[np.isfinite(hw)][-1], "core merges with the\nfirst ring; no separate\ncore beyond",
            fontsize=7, color=INK2, va="center")
    ax.text(d_qw / 2, 0.015, "quarter-\nwave", color=INK2, fontsize=7, ha="center", va="bottom")
    ax.set_xlim(0, 0.84); ax.set_ylim(0, 0.8)
    ax.set_xlabel(r"$|\delta|$ (µm)"); ax.set_ylabel("radius (µm)")
    ax.legend(loc="upper left")
    ax.set_title("(b) how wide the spot is", loc="left")

    ax = fig.add_subplot(gs[2])
    ax.axvspan(0, d_qw, color=WASH, lw=0)
    ax.plot(deltas, res["strehl_d"], color=C1)
    ax.axhline(0.8, color=MUTED, lw=0.8)
    ax.text(0.84, 0.82, "0.8", color=INK2, fontsize=7.2, ha="right")
    ax.plot([d_qw], [res["strehl_qw"]], "o", ms=5, color=C1, mec=SURF, mew=1.2)
    ax.annotate("0.82 at 0.147 µm", xy=(d_qw, res["strehl_qw"]), xytext=(0.3, 0.9), fontsize=7.5,
                color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))
    ax.set_xlim(0, 0.84); ax.set_ylim(0, 1.05)
    ax.set_xlabel(r"$|\delta|$ (µm)"); ax.set_ylabel(r"$I(\mathbf{0},\delta)/I(\mathbf{0},0)$")
    ax.set_title("(c) how bright its centre stays", loc="left")
    footer(fig, "Computed (computed level): scalar Debye integral, Eq. (1), 400-node quadrature, pupil uniform in "
                "direction sines; lambda = 0.55 um, NA_o = 1.4, n_oil = 1.515; intensity mirrored in delta "
                "(symmetric for this pupil). Ray curves: the lambda -> 0 limit of the same pupil, rays uniform "
                "in s landing at |x| = |delta| tan(theta), tan(theta_50) = 0.863, edge tan(theta_o) = 2.42. "
                "Checks D2-D4, D3b. Script: make_figures.py.")
    fig.savefig(path, dpi=cfg["dpi"]); plt.close(fig)


# ------------------------------------------------------------------ Figure 3
def fig3(cfg, res, path):
    fig = plt.figure(figsize=(12, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1], wspace=0.18, left=0.02, right=0.98, top=0.9,
                          bottom=0.18)
    ax = fig.add_subplot(gs[0])
    y_ap, y_l, y_sp, f, half_lens = 0.0, 1.8, 3.9, 1.8, 1.7
    # aperture diaphragm, condenser, specimen plane
    ax.plot([-1.3, 1.3], [y_ap, y_ap], color=INK2, lw=0.8)
    ax.plot([-2.4, -1.3], [y_ap, y_ap], color=INK, lw=3, solid_capstyle="butt")
    ax.plot([1.3, 2.4], [y_ap, y_ap], color=INK, lw=3, solid_capstyle="butt")
    ax.add_patch(Ellipse((0, y_l), 2 * half_lens + 0.3, 0.26, fill=False, ec=INK2, lw=1.0))
    ax.plot([-2.9, 2.9], [y_sp, y_sp], color=INK2, lw=0.8)
    for h, col, lab in ((-0.8, C1, "A"), (0.0, C2, "B"), (0.8, C3, "C")):
        sl = -h / f                                   # lateral slope of this plane wave (paraxial drawing)
        xl = np.array([-1.45, 1.45])
        xs = xl + sl * (y_sp - y_l)
        ax.fill([h, xl[0], xl[1]], [y_ap, y_l, y_l], color=col, alpha=0.07, lw=0)
        ax.fill([xl[0], xl[1], xs[1], xs[0]], [y_l, y_l, y_sp, y_sp], color=col, alpha=0.10, lw=0)
        for a, b in ((xl[0], xs[0]), (xl[1], xs[1])):
            ax.plot([a, b], [y_l, y_sp], color=col, lw=0.6, alpha=0.8)
        xc = -sl * (y_sp - y_l)                       # lens point of the ray that reaches the tube
        ax.plot([h, xc, 0.0], [y_ap, y_l, y_sp], color=col, lw=1.7)
        d = np.array([sl, 1.0]) / np.hypot(sl, 1.0)
        perp = np.array([d[1], -d[0]])
        for yw in (2.45, 3.05):
            c = np.array([sl * (yw - y_l), yw])
            q1, q2 = c - 0.5 * perp, c + 0.5 * perp
            ax.plot([q1[0], q2[0]], [q1[1], q2[1]], color=col, lw=1.1)
        ax.plot([h], [y_ap], "o", ms=8, color=col, mec=SURF, mew=1.5, zorder=5)
        ax.text(h, y_ap - 0.22, lab, ha="center", va="top", color=INK, fontsize=9, fontweight="bold")
    ax.add_patch(Circle((0, y_sp), 0.17, fc=AXIS, ec=INK2, lw=0.8, zorder=6))
    ax.add_artist(FancyArrowPatch((0, y_sp + 0.25), (0, y_sp + 0.75), arrowstyle="-|>", mutation_scale=10,
                                  color=INK2, lw=1.0))
    ax.text(0, y_sp + 0.82, r"objective + camera: each emitter gives one coherent image $|U_{\boldsymbol{\nu}_c}|^2$;"
                            "\nA, B, C never interfere, so the images add as intensities (Eq. 3)",
            ha="center", va="bottom", fontsize=7.8, color=INK)
    ax.text(-3.15, y_ap + 0.1, "condenser aperture: the lamp's\nimage, independent emitters", fontsize=7.6,
            color=INK, va="bottom", bbox=dict(boxstyle="round,pad=0.2", fc=SURF, ec="none", alpha=0.85))
    ax.text(-3.15, y_l, "condenser", fontsize=7.6, color=INK2, va="center")
    ax.text(-3.15, 2.85, "each emitter becomes one\nplane wave (bars: its wave-\nfronts), coherent across\nthe whole field",
            fontsize=7.6, color=INK, va="center", bbox=dict(boxstyle="round,pad=0.2", fc=SURF, ec="none", alpha=0.85))
    ax.text(3.1, y_sp - 0.1, "specimen plane;\ntube in grey; bold:\nthe one ray of each\nwave through it",
            fontsize=7.6, color=INK2, va="top", ha="right", bbox=dict(boxstyle="round,pad=0.2", fc=SURF, ec="none", alpha=0.85))
    ax.set_aspect("equal"); ax.set_xlim(-3.2, 3.2); ax.set_ylim(-0.55, 5.6); ax.axis("off")
    ax.set_title("(a) Köhler illumination, as Abbe described the image (schematic, not to scale)", loc="left")

    ax = fig.add_subplot(gs[1])
    dx = np.linspace(0, 1.6, 801)
    ax.axhline(0, color=AXIS, lw=0.8)
    for na_c, col, lab, off in ((1.4, C1, r"$\mathrm{NA_c}=1.4$ ($S=1$)", (-0.02, 0.06, "right")),
                                (1.05, C2, r"$\mathrm{NA_c}=1.05$ ($S=0.75$)", (0.02, -0.07, "left"))):
        ax.plot(dx, om.coherence_function(dx, na_c, cfg["lam"]), color=col, label=lab)
        z = res["coh_zero_um"][str(na_c)]
        ax.plot([z], [0], "o", ms=5, color=col, mec=SURF, mew=1.2)
        ax.text(z + off[0], off[1], f"{z:.2f} µm", fontsize=7.5, color=INK, ha=off[2], va="center")
    ax.axvline(1.0, color=MUTED, lw=0.8)
    ax.text(1.02, 0.6, "tube\ndiameter\n1 µm", fontsize=7.5, color=INK2, va="top")
    ax.set_xlim(0, 1.6); ax.set_ylim(-0.2, 1.05)
    ax.set_xlabel(r"separation $|\Delta\mathbf{x}|$ of two specimen points (µm)")
    ax.set_ylabel(r"degree of coherence $\Gamma(\Delta\mathbf{x})$")
    ax.legend(loc="upper right")
    ax.set_title("(b) how alike the light is at two points", loc="left")
    footer(fig, "(a) schematic, drawn with a paraxial lens for clarity (the real map is Eq. 5 of the sine document). "
                "(b) plotted from Eq. (4), van Cittert-Zernike for illumination uniform in direction sines (the "
                "sine document's W, hypotheses H1-H3); a 32x64 source quadrature reproduces it to 1e-8 (C1); "
                "lambda = 0.55 um, n_oil = 1.515. Script: make_figures.py.")
    fig.savefig(path, dpi=cfg["dpi"]); plt.close(fig)


# ------------------------------------------------------------------ Figure 4
def fig4(cfg, res, path):
    a_o = cfg["na_o"] / cfg["lam"]
    nu = np.linspace(0, 5.6, 1121)
    fig = plt.figure(figsize=(7.6, 4.3))
    ax = fig.add_axes([0.09, 0.27, 0.86, 0.62])
    labs = (r"$S=0$ (coherent: the pupil itself)", r"$S=0.5$", r"$S=0.75$", r"$S=1$ (= incoherent OTF)")
    for S, col, lab in zip((0.0, 0.5, 0.75, 1.0), ORD, labs):
        ax.plot(nu, om.transfer_absorption(nu, S, cfg), color=col, label=lab)
    lg = ax.legend(loc="upper right", frameon=True, facecolor=SURF, edgecolor="none", framealpha=1.0)
    lg.set_zorder(5)
    ax.axvline(a_o, color=MUTED, lw=0.8); ax.axvline(2 * a_o, color=MUTED, lw=0.8)
    ax.text(a_o + 0.05, 0.78, r"$\mathrm{NA_o}/\lambda$" + "\ncoherent cutoff", fontsize=7.5, color=INK2, va="top")
    ax.text(2 * a_o - 0.05, 0.45, r"$2\,\mathrm{NA_o}/\lambda$" + "\nincoherent cutoff", fontsize=7.5,
            color=INK2, va="top", ha="right")
    ax.set_xlim(0, 5.6); ax.set_ylim(0, 1.13)
    ax.set_xlabel(r"spatial frequency $|\boldsymbol{\nu}|$ (cycles/µm)")
    ax.set_ylabel(r"$H_S(\boldsymbol{\nu})$, first-order transfer")
    ax.set_title("How much of each spatial frequency of a faint absorber reaches the image", loc="left")
    footer(fig, "Plotted from Eq. (7): overlap of the condenser disc (radius S NA_o/lambda) with the pupil shifted by "
                "nu; in focus, pupil uniform in direction sines, weak-object approximation; lambda = 0.55 um, "
                "NA_o = 1.4. H_1 equals the closed-form incoherent OTF to 1e-12, cutoffs (1+S) NA_o/lambda (C2). "
                "Script: make_figures.py.", y=0.02)
    fig.savefig(path, dpi=cfg["dpi"]); plt.close(fig)


# ------------------------------------------------------------------ Figure 5
def fig5(cfg, res, path):
    fig, axs = plt.subplots(2, 2, figsize=(11, 6.6), sharex=True,
                            gridspec_kw=dict(height_ratios=[1.6, 1], hspace=0.12, wspace=0.12, left=0.08,
                                             right=0.98, top=0.9, bottom=0.17))
    for j, md in enumerate(("1.0", "3.0")):
        p = {k: np.array(v) for k, v in res["profiles"][md].items()}
        ax = axs[0, j]
        ax.plot(p["v"], p["T"], color=MUTED, lw=0.9, label=r"shadow $T(v)$ (no blur)")
        ax.plot(p["v"], p["inc"], color=C2, label=r"incoherent model $B(T*h_0)$")
        ax.plot(p["v"], p["S1"], color=ORD[3], label=r"Köhler, $S=1$")
        ax.plot(p["v"], p["S075"], color=ORD[2], label=r"Köhler, $S=0.75$")
        ax.plot(p["v"], p["S0"], color=ORD[0], label=r"$S\to0$ (coherent)")
        ax.set_ylim(-0.02, 1.12); ax.set_xlim(-1.2, 1.2)
        ax.set_title(rf"$\mu d={float(md):g}$ (flat 1 µm tube, in focus)", loc="left")
        if j == 0:
            ax.set_ylabel(r"$I(v)/B$")
            ax.legend(loc="lower left", fontsize=7.6)
        ax = axs[1, j]
        ax.axhline(0, color=AXIS, lw=0.8)
        ax.plot(p["v"], p["inc"] - p["S1"], color=ORD[3], label=r"$I_{\rm inc}-I_{S=1}$")
        ax.plot(p["v"], p["darkfield"], color=C2, lw=1.4, ls=(0, (4, 2)),
                label="dark-field image of the directions\noutside the objective's cone")
        ax.set_ylim(-0.003, 0.05)
        ax.set_xlabel(r"across-branch position $v$ (µm)")
        if j == 0:
            ax.set_ylabel("difference / B")
            ax.legend(loc="upper left", fontsize=7.6)
    fig.suptitle("Same tube, same objective: the renderers' incoherent model against Köhler illumination at three condenser settings",
                 x=0.08, ha="left", fontsize=10, color=INK)
    footer(fig, "Computed (computed level): thin-object (projection) pure absorber, Abbe sum over 32x64 source points, "
                "lambda = 0.55 um, NA_o = 1.4, n_oil = 1.515; incoherent model via the closed-form OTF; dark-field "
                "image summed over source directions out to |nu_cx| = 40 /um (residual 2.6 % and 1.2 % of the peak "
                "difference, C5). Checks C3-C5. Script: make_figures.py.")
    fig.savefig(path, dpi=cfg["dpi"]); plt.close(fig)


def main(cfg):
    with open("check_results.json") as fh:
        res = json.load(fh)
    fig1(cfg, res, "fig1_pupil_to_spot.png")
    fig2(cfg, res, "fig2_defocus_wave_vs_rays.png")
    fig3(cfg, res, "fig3_kohler_coherence.png")
    fig4(cfg, res, "fig4_transfer_functions.png")
    fig5(cfg, res, "fig5_tube_coherence.png")
    print("wrote fig1..fig5")


if __name__ == "__main__":
    main(CFG)
