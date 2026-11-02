"""make_theory_figures.py -- Figures 7-11 of the diameter theory pack (TEEG, 2026-11-02). Plotting only.

The mathematics and every drawn constant live in theory_figs_model.py (with its repository source); the checks of
those numbers are check_theory_figures.py (run it first). Run:  python3 make_theory_figures.py [outdir]
Colours: the dataviz reference palette (categorical slots 1-6 in fixed order; the blue ramp for ordered families);
palette validated with the dataviz skill's validate_palette.js (all checks pass; slots 3-5 below 3:1 contrast,
so every series also carries a direct label or a line style).
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import theory_figs_model as tm  # noqa: E402

INK, INK2, MUTED, AXIS, SURF, WASH = "#0b0b0b", "#52514e", "#898781", "#c3c2b7", "#fcfcfb", "#f0efec"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
C1, C2, C3 = CAT[:3]
RAMP5 = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2, "axes.titlesize": 9.5, "axes.titlecolor": INK, "figure.facecolor": SURF,
    "axes.facecolor": SURF, "savefig.facecolor": SURF, "lines.linewidth": 1.6, "lines.solid_capstyle": "round",
    "legend.frameon": False, "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False})
DPI = 200


def footer(fig, text, y=0.012):
    fig.text(0.01, y, text, fontsize=7.0, color=INK2, ha="left", va="bottom", wrap=True)


# ---------------------------------------------------------------------------------------------- Figure 7 (T1)
def fig7(path):
    fig = plt.figure(figsize=(12.5, 4.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.25, left=0.07, right=0.84, top=0.83, bottom=0.2)
    r, phi = 0.5, np.radians(30.0)
    ax = fig.add_subplot(gs[0])
    u, v = tm.ellipse_section(r, phi)
    ax.fill(u, v, color=C1, alpha=0.12, lw=0)
    ax.plot(u, v, color=C1, lw=1.2)
    ax.axhline(0, color=AXIS, lw=0.8, zorder=0); ax.axvline(0, color=AXIS, lw=0.8, zorder=0)
    ax.annotate("", xy=(1.45, 0), xytext=(1.12, 0), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1))
    ax.text(1.47, 0.03, r"$\hat e_u$" + "\n(branch, projected)", fontsize=8, color=INK2, va="bottom")
    ax.plot([0, 0], [-r, r], color=C1, lw=2.6)
    ax.text(0.05, -0.62, r"measuring line $\hat y$ ($u = 0$): chord $= d$" + "\nfor every heading and tilt (Eq. 7)",
            color=INK, fontsize=8, va="top")
    t = r * tm.chord_ratio(np.radians(45.0), phi)
    a = np.radians(45.0)
    ax.plot([-t * np.cos(a), t * np.cos(a)], [-t * np.sin(a), t * np.sin(a)], color=C2, lw=2.2)
    ax.text(t * np.cos(a) + 0.04, t * np.sin(a) + 0.02, "a pixel row, branch heading θ = 45°:\nchord = %.3f d (Eq. 8)" % (2 * t),
            color=INK, fontsize=8, va="bottom")
    ax.set_aspect("equal"); ax.set_xlim(-1.3, 2.25); ax.set_ylim(-0.95, 0.95)
    ax.set_xlabel(r"$u$ along the branch ($\mu$m)"); ax.set_ylabel(r"$v$ across ($\mu$m)")
    ax.set_title("(a) the focal-plane section of a tilted tube: an ellipse\n"
                 r"$(u\sin\varphi)^2 + v^2 \leq r^2$, here $r$ = 0.5 $\mu$m, $\varphi$ = 30° (Eq. 6)", loc="left")

    ax = fig.add_subplot(gs[1])
    eps = np.linspace(0, np.radians(60), 300)
    for col, ph in zip(RAMP5[::-1], (0, 15, 30, 60, 90)):
        y = tm.chord_ratio(eps, np.radians(ph))
        ax.plot(np.degrees(eps), y, color=col, lw=1.8)
        lab = {0: "φ = 0° (flat): 1/cos ε", 90: "φ = 90° (vertical): 1"}.get(ph, "φ = %d°" % ph)
        ax.text(60.8, y[-1] + (0.012 if ph == 60 else 0.0), lab, color=INK2, fontsize=8, va="center")
    ax.plot([45], [tm.chord_ratio(np.radians(45), np.radians(30))], "o", ms=7, mfc=C2, mec=SURF, mew=1.5, zorder=5)
    ax.annotate("1.265 d\n(panel a)", xy=(45, 1.265), xytext=(36, 1.52), fontsize=8, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    for e, txt in ((10, "+1.5 %"), (20, "+6 %"), (30, "+15 %")):
        yv = float(tm.chord_ratio(np.radians(e), 0.0))
        ax.plot([e], [yv], "o", ms=5, mfc=RAMP5[-1], mec=SURF, mew=1.2, zorder=5)
        ax.text(e - 1, yv + 0.045, txt, fontsize=7.5, color=INK2, ha="right")
    ax.set_xlim(0, 60); ax.set_ylim(0.97, 2.1)
    ax.set_xlabel(r"misalignment $\varepsilon$ of the line from $\hat y$ (deg); $\varepsilon = 90° - \alpha$; a pixel row: $\varepsilon = 90° - \theta$")
    ax.set_ylabel(r"chord / $d$")
    ax.set_title("(b) any other line over-reads (Eq. 8):\nchord / d = "
                 r"$1/\sqrt{\sin^2\!\varepsilon\,\sin^2\!\varphi + \cos^2\!\varepsilon}$", loc="left")
    fig.suptitle("T1 · Why the diameter is read along ŷ, the in-plane normal to the branch's projection",
                 x=0.01, ha="left", fontsize=11, color=INK, y=0.985)
    footer(fig, "Geometry only (optics enter later). Source: handoff_diameter_remeasurement.md, Math reference Eqs. 5-8 "
                "(main); SPEC 2.1. Numbers checked in check_theory_figures.py G1-G3 (1.2649; 1.0154 / 1.0642 / 1.1547; "
                "chord along ŷ = d to 2e-16 on 2000 random tubes).")
    fig.savefig(path, dpi=DPI); plt.close(fig)


# ---------------------------------------------------------------------------------------------- Figure 8 (T2)
def fig8(path):
    fig = plt.figure(figsize=(13, 4.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.25, 1.05], wspace=0.34, left=0.05, right=0.985, top=0.8, bottom=0.2)
    ax = fig.add_subplot(gs[0])
    v = np.linspace(-1.0, 1.0, 801)
    for col, al in zip((RAMP5[0], RAMP5[2], RAMP5[4]), (0.3, 3.0, 30.0)):
        T = tm.transmission(v, 1.0, al)
        ax.plot(v, T, color=col)
    ax.text(0.0, 0.78, r"$\alpha$ = 0.3 (faint)", color=INK2, fontsize=8, ha="center")
    ax.text(0.0, 0.11, r"$\alpha$ = 3", color=INK2, fontsize=8, ha="center")
    ax.annotate(r"$\alpha$ = 30 (nearly opaque: a box)", xy=(0.42, 0.01), xytext=(0.6, 0.3), fontsize=8, color=INK2,
                ha="center", arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.set_ylim(-0.03, 1.08); ax.set_xlim(-1, 1); ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xlabel(r"$v$ across the branch ($\mu$m)"); ax.set_ylabel(r"$T(v)$")
    ax.set_title("(a) Beer-Lambert across a tube, d = 1 µm:\n"
                 r"$T(v) = \exp(-\alpha\,s_d(v))$, $\alpha = \mu d/\cos\varphi$", loc="left")

    ax = fig.add_subplot(gs[1])
    v = np.linspace(-2.0, 2.0, 801)
    kern = lambda z: tm.gauss_kernel(z, tm.SIGMA_FIT_UM)
    for col, d in zip(RAMP5[1:], (0.3, 0.5, 1.0, 2.0)):
        prof = tm.tube_profile(v, d, 1.0 * d, kern)
        ax.plot(v, prof, color=col)
        ax.text(0.0, prof[400] - 0.035, "d = %g µm" % d, color=INK2, fontsize=8, ha="center", va="top",
                bbox=dict(boxstyle="square,pad=0.1", fc=SURF, ec="none", alpha=0.85))
    vs = np.linspace(-tm.PROFILE_HALF_UM, tm.PROFILE_HALF_UM, tm.PROFILE_N)
    vs = vs[np.abs(vs) <= 2.0]
    ax.plot(vs, np.full_like(vs, 1.035), "|", color=MUTED, ms=5)
    ax.text(2.0, 1.055, "sample positions (53 over ±3 µm)", fontsize=7.5, color=INK2, ha="right")
    ax.set_xlim(-2, 2); ax.set_ylim(0.0, 1.09)
    ax.set_xlabel(r"$v$ ($\mu$m)"); ax.set_ylabel(r"$I_{\rm model}/\bar B$")
    ax.set_title("(b) the model the fit compares with the profile, (B3.1):\n"
                 r"$1 - ((1-T) * g)(v)$, Gaussian $g$, $\sigma$ = 0.099 µm, $\mu$ = 1 µm$^{-1}$, $\varphi$ = 0", loc="left")

    ax = fig.add_subplot(gs[2])
    d = np.logspace(-1, np.log10(3), 300)
    lo, hi = tm.v_tube_bounds(d)
    ax.fill_between(d, lo, hi, color=C1, alpha=0.18, lw=0)
    ax.plot(d, lo, color=C1, lw=1.4); ax.plot(d, hi, color=C1, lw=1.4, ls="--")
    s2 = tm.SIGMA_FIT_UM ** 2
    ax.axhline(s2, color=C2, lw=1.6)
    ax.text(0.105, s2 * 1.18, r"in-focus blur $\sigma^2$ = %.4f µm$^2$" % s2, color=INK, fontsize=8)
    ax.text(1.05, 0.03, r"$d^2/16$ (faint, solid)", color=INK2, fontsize=8, ha="left", va="top")
    ax.text(0.95, 0.2, r"$d^2/12$ (opaque, dashed)", color=INK2, fontsize=8, ha="right")
    for dc in (np.sqrt(12) * tm.SIGMA_FIT_UM, 4 * tm.SIGMA_FIT_UM):
        ax.axvline(dc, color=MUTED, lw=0.8, ls=":")
    ax.text(0.105, 0.75, "below d = %.2f-%.2f µm the tube's\nown squared width is smaller\nthan the blur's"
            % (np.sqrt(12) * tm.SIGMA_FIT_UM, 4 * tm.SIGMA_FIT_UM), fontsize=7.5, color=INK2, ha="left", va="top")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.1, 3); ax.set_ylim(5e-4, 1)
    ax.set_xticks([0.1, 0.2, 0.3, 0.5, 1, 2, 3]); ax.set_xticklabels(["0.1", "0.2", "0.3", "0.5", "1", "2", "3"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel(r"$d$ ($\mu$m)"); ax.set_ylabel(r"normalised squared width $V$ (µm$^2$)")
    ax.set_title("(c) squared widths add under blur (Eq. 15):\n"
                 r"$V[{\rm dip}] = V_{\rm tube} + V[g]$; thin tubes hide in the blur", loc="left")
    fig.suptitle("T2 · The per-node forward model: an absorbing tube, blurred, read along the measuring line",
                 x=0.01, ha="left", fontsize=11, color=INK, y=0.985)
    footer(fig, "Sources: mathematics §3.1 Eq. 8 and §3.3 Eq. 15, l.460-466 (main); SPEC (3) and (B3.1) (sci/diameter-pipeline); "
                "D-019 (μ fitted, φ from the line fit). σ = 0.099 µm is the pilot fit's Gaussian (σ_fit² ≈ σ_r(0)² + p_x²/4); "
                "the final fit uses the SGT kernel (Fig. 9). Checks P1b, P2b, P3: (B3.1) converged to 1e-13; dip area = "
                "(πd/2)(I₁ − L₁); V → d²/16 and d²/12.")
    fig.savefig(path, dpi=DPI); plt.close(fig)


# ---------------------------------------------------------------------------------------------- Figure 9 (T5)
def fig9(path):
    fig = plt.figure(figsize=(12, 4.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.1], wspace=0.22, left=0.06, right=0.8, top=0.83, bottom=0.17)
    w = 0.5
    styles = ["-", "-", "--", "-.", ":", (0, (5, 1.5, 1, 1.5))]
    axa, axb = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    z1 = np.linspace(-1.0, 1.0, 1601); z2 = np.linspace(-4.0, 4.0, 3201)
    handles = []
    for col, ls, (name, (b, x, zt)) in zip(CAT, styles, tm.SGT_MEMBERS):
        g1 = tm.sgt_kernel(z1, w, b, x, zt)
        h, = axa.plot(z1, g1 / tm.sgt_kernel(np.array([0.0]), w, b, x, zt)[0], color=col, ls=ls, lw=1.6)
        axb.plot(z2, tm.sgt_kernel(z2, w, b, x, zt), color=col, ls=ls, lw=1.6)
        handles.append((h, "%s\n(%g, %g, %g)" % (name, b, x, zt)))
    axa.axhline(0.5, color=MUTED, lw=0.8, ls=":")
    axa.annotate("", xy=(0.25, 0.5), xytext=(-0.25, 0.5), arrowprops=dict(arrowstyle="<->", color=INK2, lw=1))
    axa.text(-0.98, 0.78, "half maximum at ±w/2:\nFWHM = w = 0.5 µm\nfor every member", ha="left", va="top",
             fontsize=8, color=INK)
    axa.set_xlim(-1, 1); axa.set_ylim(0, 1.05)
    axa.set_xlabel(r"$z$ ($\mu$m)"); axa.set_ylabel(r"$g_{w,\kappa}(z)\,/\,g_{w,\kappa}(0)$")
    axa.set_title("(a) shapes at equal width: peak-normalised", loc="left")
    axb.set_yscale("log"); axb.set_xlim(-4, 4); axb.set_ylim(1e-6, 5)
    axb.set_xlabel(r"$z$ ($\mu$m)"); axb.set_ylabel(r"$g_{w,\kappa}(z)$ (µm$^{-1}$), unit area")
    axb.set_title(r"(b) tails: $|z|^{-(1+\beta/\xi)}$ for $\xi > 0$ ($t_4$: $|z|^{-5}$; Lorentz: $|z|^{-2}$)", loc="left")
    axb.text(2.2, 2.2e-3, r"Lorentzian $\propto |z|^{-2}$", fontsize=7.5, color=INK2)
    axb.text(2.6, 8e-5, r"$t_4 \propto |z|^{-5}$", fontsize=7.5, color=INK2)
    fig.legend([h for h, _ in handles], [l for _, l in handles], loc="center left", bbox_to_anchor=(0.81, 0.5),
               fontsize=8, title=r"member and $\kappa = (\beta, \xi, \zeta)$", title_fontsize=8, handlelength=3.2,
               labelspacing=0.9)
    fig.suptitle(r"T5 · The fit's kernel: the skewed generalized-$t$ family $g_{w,\kappa}$ (D-052.1), "
                 r"seven parameters $(d, \mu, v_0, w, \beta, \xi, \zeta)$ per plane",
                 x=0.01, ha="left", fontsize=11, color=INK, y=0.985)
    footer(fig, "(H1)-(H3) of TEEG_diameter_correction_table_handoff_2026-10-16.md §3.2 (sci/diameter-pipeline) = SPEC Block 12; "
                "β shapes the peak, ξ the tails, ζ the skew (right/left half-width ratio ζ²); every member has FWHM w by "
                "construction. Computed here and checked against the repository's model/blur_kernels.sgt_kernel_pdf "
                "(K6: 4e-16 of the peak), scipy.stats (K3b) and the closed-form FWHM (K2). The renderer's defocus kernel "
                "stays Gaussian (D-067.2).")
    fig.savefig(path, dpi=DPI); plt.close(fig)


# ---------------------------------------------------------------------------------------------- Figure 10
def fig10(path):
    fig = plt.figure(figsize=(14, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.35], wspace=0.17, left=0.05, right=0.79, top=0.84, bottom=0.19)
    kn, sg, gm = tm.CONFIGURED_KNOTS_UM, tm.CONFIGURED_SIGMA_UM, tm.CONFIGURED_GAMMA
    pk, ps = tm.PILOT_KNOTS_UM, tm.PILOT_SIGMA_UM
    # (a) near field
    ax = fig.add_subplot(gs[0])
    ax.axvspan(0, tm.QUARTER_WAVE_UM, color=WASH, lw=0)
    ax.text(0.074, 0.40, "in focus:\n|δ| ≤ 0.147 µm\n(quarter-wave\ntolerance)", fontsize=7.0, color=INK2, ha="center", va="top")
    for k in (1, 2, 3):
        ax.axvline(k * tm.PLANE_UM, color=AXIS, lw=0.8, ls=":")
        ax.text(k * tm.PLANE_UM, 0.005, "k*±%d" % k, fontsize=7.2, color=MUTED, ha="center", va="bottom")
    dd = np.linspace(0, 0.84, 300)
    h1, = ax.plot(dd, tm.sigma_r_law(dd, kn, sg, gm), color=C1, lw=1.8, marker=None)
    ax.plot(kn, sg, "o", ms=5.5, mfc=C1, mec=SURF, mew=1.2)
    h2, = ax.plot(pk, ps, color=C2, lw=1.4, marker="s", ms=6.5, mfc=C2, mec=SURF, mew=1.2, markevery=[1, 2, 3])
    ax.errorbar([0.84], [ps[-1]], yerr=[tm.PILOT_ANCHOR_SE_UM], color=C2, lw=1.2, capsize=3, zorder=4)
    h3 = []
    for mk, (d, vals) in zip(("^", "v"), tm.PHANTOM_FIRST_STAGE.items()):
        hh, = ax.plot((0.0,) + pk[1:], (0.080,) + vals, color=C3, lw=1.0, ls="--", marker=mk, ms=6, mfc=C3, mec=SURF,
                      mew=1.0, markevery=[1, 2, 3])
        h3.append(hh)
    ax.legend([h1, h2, h3[0], h3[1]],
              ["configured: ideal Debye, 550 nm (not calibrated)",
               "pilot 529878215, near fit, 30 nodes (Cell 7, 2026-10-30);\nbar: anchor ± SE 0.044 µm",
               "Block 10 on noise-free phantoms rendered with\nthe configured kernel: d = 0.3 µm",
               "the same, d = 0.5 µm"], loc="upper left", bbox_to_anchor=(0.17, 1.0), fontsize=7.4, handlelength=2.6)
    ax.set_xlim(0, 0.9); ax.set_ylim(0, 0.8)
    ax.set_xlabel(r"defocus $|\delta|$ of a slab from the image plane ($\mu$m)")
    ax.set_ylabel(r"kernel width $\sigma_{\rm r}(\delta)$ ($\mu$m)")
    ax.set_title("(a) near field (Block 10): the table to 0.84 µm (three planes)", loc="left")
    # (b) full range
    ax = fig.add_subplot(gs[1])
    ax.axvspan(tm.FAR_MIN_K_OFFSET * tm.PLANE_UM, tm.P_FAR * tm.PLANE_UM, color=WASH, lw=0)
    ax.text(2.94, 0.12, "far fit uses planes |k − k*| = 5 … 16 (S2b)", fontsize=7.6, color=INK2, ha="center")
    ax.axvline(tm.DELTA0_UM, color=AXIS, lw=0.8)
    ax.text(tm.DELTA0_UM - 0.05, 5.2, "anchor 0.84 µm\n(junction, D-080)", fontsize=7.4, color=INK2, va="top", ha="right")
    dd = np.linspace(0, 4.6, 600)
    x0, y0 = tm.DELTA0_UM, sg[-1]
    for (lab, sl), lsty in zip(tm.PRIOR_SLOPES, ((0, (6, 2)), (0, (2, 2)), (0, (4, 1, 1, 1)), (0, (1, 1.5)))):
        xx = np.linspace(x0, 4.6, 50)
        ax.plot(xx, y0 + sl * (xx - x0), color=MUTED, lw=1.0, ls=lsty)
    lab_y = {0: y0 + 1.21 * 3.76, 1: y0 + 0.90 * 3.76}
    ax.text(4.65, lab_y[0], "uniform disc: 1.21", fontsize=7.2, color=INK2, va="center")
    ax.text(4.65, lab_y[1], "isotropic cone ($W^*$): 0.90", fontsize=7.2, color=INK2, va="center")
    ax.text(4.65, y0 + 0.449 * 3.76 + 0.06, "cos-weighted cone ($W$),\nNA$_c$ 1.12 / 1.05: 0.47 / 0.43\n(ZEISS visual default)",
            fontsize=7.2, color=INK2, va="bottom")
    ax.plot(dd, tm.sigma_r_law(dd, kn, sg, gm), color=C1, lw=2.0)
    ax.text(4.65, y0 + gm * (4.6 - x0), "configured γ = 0.79 (D-024)\n= W cone at NA$_c$ 1.40", fontsize=7.4,
            color=INK, va="center")
    xx = np.linspace(x0, 4.6, 50)
    lo_s, hi_s = tm.PILOT_GAMMA_T1
    ax.fill_between(xx, ps[-1] + lo_s * (xx - x0), ps[-1] + hi_s * (xx - x0), color=C2, alpha=0.12, lw=0)
    ax.plot(xx, ps[-1] + tm.PILOT_GAMMA_HAT * (xx - x0), color=C2, lw=1.6, ls="--")
    ax.plot(pk[1:], ps[1:], "s", ms=5.5, mfc=C2, mec=SURF, mew=1.0, zorder=5)
    ax.text(4.65, ps[-1] + tm.PILOT_GAMMA_HAT * (4.6 - x0) - 0.12, "pilot far fit γ̂ = 0.408", fontsize=7.4, color=INK,
            va="top")
    ax.text(0.95, 4.95, "pilot 529878215 (orange): near points, then γ̂ = 0.408\nwith its t₁ interval [−0.01, 0.83] shaded:"
                        "\n2 far nodes only, a diagnostic; no production\nkernel written (far set insufficient, D-088.3)",
            fontsize=7.4, color=INK2, va="top")
    ax.set_xlim(0, 4.6); ax.set_ylim(0, 5.3); ax.set_ylabel(r"$\sigma_{\rm r}(\delta)$ ($\mu$m)")
    ax.set_xlabel(r"$|\delta|$ ($\mu$m); planes every 0.28 µm")
    ax.set_title(r"(b) far field (S2b): $\sigma_{\rm r}(\delta) = \hat\sigma_0 + \gamma\,(|\delta| - 0.84\ \mu{\rm m})$;"
                 " grey: slopes geometry would give (Eq. 17)", loc="left")
    fig.suptitle("T3 · T6 · T10 · The renderer's defocus kernel: configured, calibrated on the pilot, and what the far "
                 "slope could be", x=0.01, ha="left", fontsize=11, color=INK, y=0.985)
    footer(fig, "Configured: config.py RendererConfig (sci/diameter-pipeline @5015806). Pilot: decision-log row 2026-10-30 "
                "(later), l.139; anchor SE 0.0438 µm after D-115.4 (SPEC 10.A). Phantoms: checks/d094_2026-10-28/"
                "dark_node_growth.out (α = 0.1). Geometric slopes: mathematics Eq. 17 table (l.520-524) and the "
                "sine-condition doc (l.644-648); they are priors for γ, not measurements. σ_r(0) = 0.080 µm is "
                "configured (not identifiable from the stacks). All constants read back from their sources: checks L1-L5.")
    fig.savefig(path, dpi=DPI); plt.close(fig)


# ---------------------------------------------------------------------------------------------- Figure 11
def fig11(path):
    fig = plt.figure(figsize=(12.5, 4.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1], wspace=0.25, left=0.055, right=0.985, top=0.83, bottom=0.2)
    d = np.linspace(0.2, 2.0, 400)
    m = tm.m_toy(d)
    d_star = tm.TOY["d_star"]; d_hat = float(tm.m_toy(d_star)); d_s = float(tm.shortcut(d_hat))
    ax = fig.add_subplot(gs[0])
    ax.plot(d, d, color=MUTED, lw=1.0, ls="--"); ax.text(1.75, 1.62, "m = d (no bias)", color=INK2, fontsize=8, ha="right")
    ax.plot(d, m, color=C1, lw=2.0); ax.text(1.0, 1.95, r"$m(d) = d\,b(d)$", color=INK, fontsize=8.5)
    ax.plot([0.2, d_star], [d_hat, d_hat], color=C2, lw=1.5)
    ax.text(0.23, d_hat + 0.04, r"measured $\hat d_i$ = %.2f µm" % d_hat, color=INK, fontsize=8)
    ax.plot([d_star, d_star], [0, d_hat], color=C1, lw=1.0, ls=":")
    ax.plot([d_star], [d_hat], "o", ms=7, mfc=C1, mec=SURF, mew=1.5, zorder=5)
    ax.plot([d_star], [0.05], "o", ms=7, mfc=C1, mec=SURF, mew=1.5, zorder=5, clip_on=False)
    ax.text(d_star + 0.04, 0.1, r"root $\tilde d_i$ = 1.000 µm (Eq. 20)", fontsize=8, color=INK)
    ax.plot([d_s], [0.05], "o", ms=7, mfc=SURF, mec=C2, mew=1.6, zorder=5, clip_on=False)
    full, first = tm.eq21()
    ax.annotate("superseded shortcut $\\hat d_i / b(\\hat d_i)$ = %.3f µm:\n%.1f %% (Eq. 21: %.1f %%)" % (d_s, 100 * (d_s - 1), 100 * full),
                xy=(d_s, 0.05), xytext=(0.24, 0.36), fontsize=8, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.axvspan(0.0, tm.D_RANGE_UM[0], color=WASH, lw=0)
    ax.text(0.1, 1.15, "no root in the table's d range → out_of_domain", fontsize=7.3, color=INK2, va="center",
            ha="center", rotation=90)
    ax.set_xlim(0.0, 2.0); ax.set_ylim(0, 2.3)
    ax.set_xlabel(r"true diameter $d$ ($\mu$m)"); ax.set_ylabel(r"mean measured diameter $m(d)$ ($\mu$m)")
    ax.set_title("(a) read the forward curve backwards: solve $m(d \\mid \\varphi_i, x_{3,i}) = \\hat d_i$ for $d$", loc="left")

    ax = fig.add_subplot(gs[1])
    tau = tm.TOY["tau"]
    lo, hi = d_hat - tau * d_star, d_hat + tau * d_star
    dl, dh = tm.invert(lo), tm.invert(hi)
    dz = np.linspace(0.7, 1.35, 300)
    ax.axhspan(lo, hi, color=C2, alpha=0.14, lw=0)
    ax.axvspan(dl, dh, color=C1, alpha=0.14, lw=0)
    ax.plot(dz, tm.m_toy(dz), color=C1, lw=2.0)
    for yv in (lo, hi):
        xv = tm.invert(yv)
        ax.plot([0.7, xv], [yv, yv], color=C2, lw=0.9, ls=":"); ax.plot([xv, xv], [0.85, yv], color=C1, lw=0.9, ls=":")
    ax.text(0.71, hi + 0.012, r"$\hat d_i \pm \tau d$: the node's own scatter ($\tau$ = %.2f)" % tau, fontsize=8, color=INK)
    sd = tau * d_star / (tm.TOY["b_star"] * (1 + tm.TOY["beta"]))
    ax.text(dh + 0.012, 0.88, "→ ±%.3f µm in $\\tilde d$\n$= \\tau d / [b(1+\\beta)]$\n(divided by the slope $m'$)" % sd,
            fontsize=8, color=INK, va="bottom")
    ax.set_xlim(0.7, 1.35); ax.set_ylim(0.85, 1.45)
    ax.set_xlabel(r"$d$ ($\mu$m)"); ax.set_ylabel(r"$m(d)$ ($\mu$m)")
    ax.set_title("(b) the inversion removes the bias, not the scatter:\nerrors in $\\hat d$ map through the slope", loc="left")
    fig.suptitle("T4.3 · Correcting a node: invert the table's forward curve (classical calibration) — illustration",
                 x=0.01, ha="left", fontsize=11, color=INK, y=0.985)
    footer(fig, "ILLUSTRATION, not the project's table: b(d) = 1.15 (d / 1 µm)^0.3, the numbers mathematics §3.6 uses "
                "(b* = 1.15, β = 0.3, 'chosen, not measured'); τ = 0.10 chosen. In the pipeline m̂(d) = d·b̂(ln d, φ_i, "
                "ln μ̂_i d̂_i), read by tensor cubic interpolation and solved by bracketing on 256 log-spaced d and "
                "brentq (SPEC 14.5); flags out_of_domain / non_monotone / large_correction / high_failure / off_support. "
                "Checks I1-I4 (root exact; shortcut −4.11 %; Eq. 21 −4.31 %; MC SD 0.0665 vs 0.0669).")
    fig.savefig(path, dpi=DPI); plt.close(fig)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
    os.makedirs(outdir, exist_ok=True)
    jobs = [(fig7, "fig7_T1_measuring_line_geometry.png"), (fig8, "fig8_T2_tube_forward_model.png"),
            (fig9, "fig9_T5_sgt_kernel_family.png"), (fig10, "fig10_T3-T6-T10_defocus_kernel_law.png"),
            (fig11, "fig11_T4-3_inversion_illustration.png")]
    for f, name in jobs:
        f(os.path.join(outdir, name)); print("wrote", name)


if __name__ == "__main__":
    main()
