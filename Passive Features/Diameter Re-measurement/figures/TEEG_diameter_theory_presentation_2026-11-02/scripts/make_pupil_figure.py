"""make_pupil_figure.py -- what the pupil function P(s) = A(s) exp(i Phi(s)) does
(TEEG, 2026-11-02). Four pupils, same objective model, and the spot each gives.

Model (model level): scalar Debye, Eq. (1): U(x) = iint P(s) exp(i 2 pi n s.x / lam) d^2s
in focus, so the spot is |Fourier transform of P|^2 (computed level: 2-D FFT).
Cases: (a) ideal, A = 1 on |s| <= NA/n, Phi = 0, NA = 1.4; (b) smaller cone, NA 0.7;
(c) defocus phase Phi = -(2 pi n/lam) delta (1 - cos theta), delta = 0.3 um;
(d) coma phase Phi = a (3 rho^3 - 2 rho) cos(psi), a = 2 rad, rho = |s|/s_o.
Checks, tolerances fixed before running:
  T1 (a) is the Airy pattern: radial profile vs [2J1(b)/b]^2, max abs error < 5e-3.
  T2 peak of the unit-power spot (b)/(a) = (0.7/1.4)^2 = 0.25 within 0.01.
  T3 peak (c)/(a) equals the radial Debye quadrature (optics_model.psf_field) within 0.01.
  T4 (d): peak/(a) within 0.1 of exp(-sigma^2), sigma = rms of Phi over the pupil, and the
      spot's intensity centroid is displaced along x by more than 0.01 um (asymmetry).
Run: python3 make_pupil_figure.py  (writes fig6_pupil_function.png).
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, FancyArrowPatch
from scipy import special

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import optics_model as om  # noqa: E402

CFG = dict(lam=0.55, n=1.515, na=1.4, na_small=0.7, delta=0.3, coma_rad=2.0,
           N=1024, L_um=20.0, half_um=0.8, dpi=200)
INK, INK2, MUTED, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#c3c2b7", "#fcfcfb"
SEQ = LinearSegmentedColormap.from_list("blue_seq", [SURF, "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                                                     "#256abf", "#184f95", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#184f95", "#5598e7", "#f0efec", "#e66767", "#a52a2a"])
plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.edgecolor": AXIS,
                     "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
                     "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "axes.titlecolor": INK,
                     "axes.titlesize": 9, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "savefig.facecolor": SURF})


# ---------------- model: pupils and spots
def grids(cfg):
    N, L = cfg["N"], cfg["L_um"]
    nu = np.fft.fftshift(np.fft.fftfreq(N, d=L / N))          # cycles/um
    NUX, NUY = np.meshgrid(nu, nu)
    S = np.stack([NUX, NUY]) * cfg["lam"] / cfg["n"]          # direction sines s = lam nu / n
    x = (np.arange(N) - N // 2) * (L / N)
    return S, x


def pupil(case, S, cfg):
    sx, sy = S
    r = np.hypot(sx, sy)
    na = cfg["na_small"] if case == "b" else cfg["na"]
    s_o = na / cfg["n"]
    inside = r <= s_o
    phi = np.zeros_like(r)
    if case == "c":
        phi = -(2 * np.pi * cfg["n"] / cfg["lam"]) * cfg["delta"] * (1 - np.sqrt(np.clip(1 - r**2, 0, None)))
    if case == "d":
        rho, psi = r / s_o, np.arctan2(sy, sx)
        phi = cfg["coma_rad"] * (3 * rho**3 - 2 * rho) * np.cos(psi)
    return inside.astype(float) * np.exp(1j * phi), np.where(inside, phi, np.nan), inside


def spot(P):
    """Unit-power intensity spot |FT P|^2 (computed level)."""
    U = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(P)))
    I = np.abs(U) ** 2
    return I / I.sum()


# ---------------- checks
def run_checks(cfg, S, x, spots, phases, insides):
    ok = {}
    c = len(x) // 2
    prof = spots["a"][c, c:] / spots["a"][c, c]
    xr = x[c:]
    sel = xr <= 1.0
    b = 2 * np.pi * cfg["na"] * xr[sel] / cfg["lam"]
    airy = np.where(b > 0, (2 * special.j1(np.maximum(b, 1e-12)) / np.maximum(b, 1e-12)) ** 2, 1.0)
    ok["T1"] = np.max(np.abs(prof[sel] - airy)) < 5e-3
    pk = {k: v.max() / spots["a"].max() for k, v in spots.items()}
    ok["T2"] = abs(pk["b"] - 0.25) < 0.01
    deb = np.abs(om.psf_field(np.array([0.0]), cfg["delta"], dict(lam=cfg["lam"], n=cfg["n"], na_o=cfg["na"])))[0] ** 2 \
        / np.abs(om.psf_field(np.array([0.0]), 0.0, dict(lam=cfg["lam"], n=cfg["n"], na_o=cfg["na"])))[0] ** 2
    ok["T3"] = abs(pk["c"] - deb) < 0.01
    ph = phases["d"][insides["d"]]
    sigma2 = np.mean((ph - ph.mean()) ** 2)
    X, _ = np.meshgrid(x, x)
    cen = float((spots["d"] * X).sum())
    ok["T4"] = abs(pk["d"] - np.exp(-sigma2)) < 0.1 and abs(cen) > 0.01
    print({k: round(float(v), 4) for k, v in pk.items()}, f"debye(c)={deb:.4f}",
          f"exp(-sigma^2)={np.exp(-sigma2):.3f}", f"coma centroid x={cen:+.4f} um")
    for k, v in ok.items():
        print(k, "PASS" if v else "FAIL")
    return pk, cen


# ---------------- plotting
def plot(cfg, S, x, spots, phases, pk, path):
    titles = {"a": "(a) ideal\n" + r"$A=1$ on $|\mathbf{s}|\leq0.924$ (NA 1.4), $\Phi=0$",
              "b": "(b) smaller cone\n" + r"$A=1$ on $|\mathbf{s}|\leq0.462$ (NA 0.7), $\Phi=0$",
              "c": "(c) defocus phase, " + r"$\delta=0.3$ µm" + "\n"
                   + r"$\Phi=-\frac{2\pi n_{\rm oil}}{\lambda}\delta(1-\cos\vartheta)$",
              "d": "(d) coma phase (asymmetric)\n" + r"$\Phi=2\,\mathrm{rad}\,(3\rho^3-2\rho)\cos\psi$"}
    notes = {"a": "peak 1 (reference)", "b": f"peak {pk['b']:.2f} × (a): bigger spot",
             "c": f"peak {pk['c']:.2f} × (a): light spread", "d": f"peak {pk['d']:.2f} × (a): skewed"}
    fig = plt.figure(figsize=(12.5, 7.4))
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1, 0.06], height_ratios=[1, 1], hspace=0.62,
                          wspace=0.3, left=0.05, right=0.95, top=0.88, bottom=0.14)
    sx = S[0][0]
    th = np.linspace(0, 2 * np.pi, 400)
    hw = cfg["half_um"]
    sel = np.abs(x) <= hw
    for j, k in enumerate("abcd"):
        ax = fig.add_subplot(gs[0, j])
        im1 = ax.imshow(phases[k], extent=[sx[0], sx[-1], sx[0], sx[-1]], origin="lower", cmap=DIV,
                        vmin=-3.3, vmax=3.3, interpolation="nearest")
        s_cut = (cfg["na_small"] if k == "b" else cfg["na"]) / cfg["n"]
        ring_x = np.concatenate([np.cos(th), s_cut * np.cos(th[::-1])])
        ring_y = np.concatenate([np.sin(th), s_cut * np.sin(th[::-1])])
        ax.fill(ring_x, ring_y, facecolor="none", edgecolor=MUTED, hatch="////", lw=0)
        ax.plot(np.cos(th), np.sin(th), color=AXIS, lw=0.7)
        ax.plot(s_cut * np.cos(th), s_cut * np.sin(th), color=INK, lw=1.0)
        if k == "b":
            ax.text(0, -0.72, "A = 0: blocked", ha="center", va="center", fontsize=7.6, color=INK,
                    bbox=dict(boxstyle="round,pad=0.15", fc=SURF, ec="none"))
        ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05); ax.set_aspect("equal")
        ax.set_xlabel(r"$s_x$"); ax.set_ylabel(r"$s_y$" if j == 0 else "")
        ax.set_title(titles[k], loc="left", fontsize=8.6)
        ax2 = fig.add_subplot(gs[1, j])
        I = spots[k][np.ix_(sel, sel)]
        im2 = ax2.imshow(I / I.max(), extent=[-hw, hw, -hw, hw], origin="lower", cmap=SEQ, vmin=0, vmax=1)
        ax2.add_patch(Circle((0, 0), 0.2396, fill=False, ec=MUTED, lw=0.7))
        ax2.set_xlabel(r"$x$ (µm)"); ax2.set_ylabel(r"$y$ (µm)" if j == 0 else "")
        ax2.set_title(notes[k], loc="left", fontsize=8.6)
        p0, p1 = ax.get_position(), ax2.get_position()
        xm = 0.5 * (p0.x0 + p0.x1)
        fig.add_artist(FancyArrowPatch((xm, p0.y0 - 0.06), (xm, p1.y1 + 0.045), transform=fig.transFigure,
                                       arrowstyle="-|>", mutation_scale=11, color=INK2, lw=1.0))
    cax1 = fig.add_subplot(gs[0, 4]); cb1 = fig.colorbar(im1, cax=cax1)
    cb1.set_label(r"phase $\Phi(\mathbf{s})$ (rad); hatched: $A=0$", color=INK2, fontsize=8)
    cax2 = fig.add_subplot(gs[1, 4]); cb2 = fig.colorbar(im2, cax=cax2)
    cb2.set_label(r"$|U(\mathbf{x})|^2$ / its own peak", color=INK2, fontsize=8)
    for cb in (cb1, cb2):
        cb.outline.set_edgecolor(AXIS)
    fig.text(0.05, 0.955, r"Pupil (back focal plane): one point = one direction $\mathbf{s}$.   "
             r"$P(\mathbf{s})=A\,e^{i\Phi}$ sets how much of that plane wave passes ($A$) "
             r"and how late it arrives ($\Phi$).", fontsize=9.5, color=INK, ha="left")
    fig.text(0.468, 0.497, "Eq. (1): sum one plane wave per pupil point,\nweighted by P; then |·|² = the spot",
             fontsize=8.2, color=INK2, ha="center", va="center")
    fig.text(0.01, 0.012, "Plotted (computed level): scalar Debye theory in focus; the spot is |2-D Fourier "
             "transform of P|^2, by FFT; lambda = 0.55 um, n_oil = 1.515; pupil uniform in direction sines inside "
             "the cut-off (hatched: blocked).\nGrey circle: (a)'s first dark ring, 0.24 um, for scale. Peak ratios "
             "use unit-power spots; each spot is displayed to its own peak. Checks T1-T4 in this script (Airy "
             "match, NA^2 peak law,\ndefocus peak vs the radial Debye quadrature, coma vs exp(-rms phase^2) and its "
             "centroid shift). Script: make_pupil_figure.py.", fontsize=7.2, color=INK2, ha="left", va="bottom")
    fig.savefig(path, dpi=cfg["dpi"]); plt.close(fig)


def main(cfg):
    S, x = grids(cfg)
    spots, phases, insides = {}, {}, {}
    for k in "abcd":
        P, ph, ins = pupil(k, S, cfg)
        spots[k], phases[k], insides[k] = spot(P), ph, ins
    pk, _ = run_checks(cfg, S, x, spots, phases, insides)
    plot(cfg, S, x, spots, phases, pk, "fig6_pupil_function.png")
    print("wrote fig6_pupil_function.png")


if __name__ == "__main__":
    main(CFG)
