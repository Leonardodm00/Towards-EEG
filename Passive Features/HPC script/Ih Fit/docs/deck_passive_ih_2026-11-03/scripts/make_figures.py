"""make_figures.py -- the plotted figures of the deck handoff (passive + I_h optimisation), 2026-11-03.

Writes, into ../figures/ (16:9, 2000 x 1125 px):
  fig02_roadmap_board.png     the roadmap's computed statuses (from a roadmap.py status JSON)
  fig03_model_parameters.png  schematic: the cable model and the six fitted parameters
  fig04_ih_kinetics.png       m_inf(v), tau_h(v) of the three published I_h curves and the fitted knobs' boxes
  fig06_passive_results.png   run B's excitatory medians by layer and the bootstrap widths (numbers of the
                              annual report part 1, section 6; Eyal et al. 2016's L2/3 value for reference)
  fig07_noise_law.png         the recording-noise law (D-017.1) at theta_cf: autocovariance and one sweep
The schematic figures (fig01, fig05, fig08, fig09) are Mermaid sources in this folder (*.mmd).

    python make_figures.py --status /path/to/roadmap_status.json   # about 10 s

Needs numpy, matplotlib and the Ih Fit folder two levels up (human_ih_params.py, noise_law.py).
Standalone: not pipeline code. Pure ASCII, LF.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                    # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle  # noqa: E402
import numpy as np                                  # noqa: E402

HERE = Path(__file__).resolve().parent
IH_FIT = HERE.parents[2]
OUT = HERE.parent / "figures"
sys.path.insert(0, str(IH_FIT))
import human_ih_params as hip                       # noqa: E402
import noise_law as nl                              # noqa: E402

# --- style (dataviz reference palette; text in neutral inks) -----------------
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8984"
GRID = "#e4e3df"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
STATUS = {  # the roadmap tool's own colours (specs/roadmap/roadmap.py STATES), text colour second
    "planned": ("#d62828", "#ffffff"), "local": ("#f77f00", "#1a1a1a"), "pushed": ("#e9c46a", "#1a1a1a"),
    "testing": ("#3a86ff", "#ffffff"), "tested": ("#2a9d8f", "#ffffff"), "running": ("#8338ec", "#ffffff"),
    "failing": ("#b5174e", "#ffffff"), "done": ("#2d9a3d", "#ffffff"),
    "open": ("#8d99ae", "#1a1a1a"), "studying": ("#3a86ff", "#ffffff"), "closed": ("#2d9a3d", "#ffffff"),
}
W, H, DPI = 13.333, 7.5, 150
THETA_CF = nl.NoiseLaw(0.0380, 0.175, 0.0508, 0.0366)     # (mV, -, mV, s): the cohort closed-form law
V_REST = -73.5                                            # mV, the plan's reference resting potential

plt.rcParams.update({
    "font.size": 13, "axes.edgecolor": INK3, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.titlesize": 15, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white",
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=DPI, facecolor="white")
    plt.close(fig)
    print("wrote", OUT / name)


# =============================================================================
def fig02_roadmap(status_json):
    d = json.loads(Path(status_json).read_text())
    impl = d["implementation"]
    theo = d["theory"]
    impl = impl if isinstance(impl, list) else [dict(v, id=k) for k, v in impl.items()]
    theo = theo if isinstance(theo, list) else [dict(v, id=k) for k, v in theo.items()]
    fig = plt.figure(figsize=(W, H))
    fig.suptitle("Where the allen-mtg-fit roadmap stands (computed by roadmap.py at %s, %s)"
                 % (d["commit_short"], d["commit_date"][:10]), x=0.02, y=0.985, ha="left", fontsize=15,
                 fontweight="bold", color=INK)

    def column(x0, width, stages, title):
        groups = [s for s in stages if not s.get("parent")]
        rows = []
        for g in groups:
            rows.append((g, True))
            rows += [(s, False) for s in stages if s.get("parent") == g["id"]]
        n = len(rows)
        top, bottom = 0.90, 0.04
        h = (top - bottom) / n
        fig.text(x0, 0.915, title, fontsize=12, fontweight="bold", color=INK2)
        for i, (s, is_group) in enumerate(rows):
            y = top - (i + 1) * h
            st = s.get("status", "")
            bg, fg = STATUS.get(st, ("#cccccc", "#000000"))
            chip_w = 0.065
            ax_x = x0 + (0.0 if is_group else 0.012)
            fig.patches.append(FancyBboxPatch((ax_x, y + 0.12 * h), chip_w, 0.76 * h,
                                              boxstyle="round,pad=0,rounding_size=0.004",
                                              transform=fig.transFigure, fc=bg, ec="none"))
            fig.text(ax_x + chip_w / 2, y + 0.5 * h, st, ha="center", va="center", fontsize=8.5,
                     color=fg, fontweight="bold")
            title_txt = s.get("title", "")
            max_chars = int(width * 150) - (0 if is_group else 2)
            if len(title_txt) > max_chars:
                title_txt = title_txt[:max_chars - 1].rstrip() + "..."
            fig.text(ax_x + chip_w + 0.006, y + 0.5 * h, "%s  %s" % (s["id"], title_txt), va="center",
                     fontsize=9.5 if not is_group else 10.5, color=INK if is_group else INK2,
                     fontweight="bold" if is_group else "normal")

    column(0.02, 0.50, impl, "Implementation: state computed from git, SPEC.md, tester reports, HPC markers")
    column(0.555, 0.43, theo, "Theory: state set by the user")
    save(fig, "fig02_roadmap_board.png")


# =============================================================================
def fig03_model():
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0.0, 0.0, 0.47, 0.92])
    ax.set_xlim(-6, 6)
    ax.set_ylim(-5.5, 7.5)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.suptitle("The fitted model: a reconstructed cable with passive membrane and a uniform h-current",
                 x=0.02, ha="left", fontsize=16, fontweight="bold", color=INK)
    # 60 um boundary (schematic)
    ax.add_patch(Circle((0, 0), 2.2, fill=False, ls=(0, (4, 3)), ec=INK3, lw=1.2))
    ax.text(1.65, -2.05, "60 um", color=INK3, fontsize=10)
    # dendrites (schematic): apical trunk and tuft, basal branches
    ax.plot([0, 0], [0.6, 4.4], color=INK2, lw=3.0, solid_capstyle="round")
    for dx, dy in [(-1.5, 1.0), (1.4, 1.1), (-0.5, 1.3), (0.6, 1.2)]:
        ax.plot([0, dx], [4.4, 4.4 + dy], color=INK2, lw=1.8, solid_capstyle="round")
    for (x1, y1) in [(-3.6, -1.4), (3.4, -1.0), (-2.6, 1.6), (2.8, 1.9), (-1.6, -3.0)]:
        ax.plot([0, x1], [0, y1], color=INK2, lw=2.0, solid_capstyle="round")
    # spine factor region marker on the distal parts
    for (x1, y1) in [(-3.6, -1.4), (3.4, -1.0), (-2.6, 1.6), (2.8, 1.9), (-1.6, -3.0)]:
        r = math.hypot(x1, y1)
        f0 = 2.2 / r
        ax.plot([x1 * f0, x1], [y1 * f0, y1], color=BLUE, lw=4.5, alpha=0.35, solid_capstyle="round")
    ax.plot([0, 0], [2.2, 4.4], color=BLUE, lw=6, alpha=0.35, solid_capstyle="round")
    # soma and axon stub (2 cylinders, Hay convention)
    ax.add_patch(Circle((0, 0), 0.65, fc="#d9d8d3", ec=INK2, lw=1.5, zorder=3))
    ax.add_patch(Rectangle((0.15, -2.6), 0.22, 1.0, fc="#d9d8d3", ec=INK2, lw=1.0))
    ax.add_patch(Rectangle((0.15, -3.7), 0.22, 1.0, fc="#d9d8d3", ec=INK2, lw=1.0))
    ax.text(0.55, -3.3, "axon stub\n2 x (30 um x 1 um)", fontsize=10, color=INK2, va="center")
    # electrode and protocols
    ax.plot([-0.3, -4.6], [-0.3, -4.2], color=INK, lw=1.5)
    ax.text(-5.9, -4.95, "somatic electrode: SS pulses 0.5 ms, +/-200 pA;\nLS steps 1 s", fontsize=10.5,
            color=INK)
    ax.text(-5.9, 6.4, "spine membrane folded into the arbour beyond 60 um:\n"
            "c_m(s) = C_m F,  g_pas(s) = F / R_m   (F = 1.9 for L2/L3; F is an input, not fitted)",
            fontsize=10.5, color=BLUE)
    ax.text(0.35, 3.0, "R_a\n(axial)", fontsize=11, color=INK2)
    ax.text(-4.6, 2.9, "I_h: g_h uniform on\nsoma, basal, apical", fontsize=11, color=ORANGE)
    ax.text(-5.9, -5.5, "Schematic, not to scale.", fontsize=9, color=INK3)
    # table of parameters
    tx = fig.add_axes([0.49, 0.07, 0.49, 0.80])
    tx.axis("off")
    rows = [
        ("C_m", "uF/cm2", "log", "[0.3, 3.0]", "membrane capacitance"),
        ("R_m", "Ohm cm2", "log", "[1e3, 1e5]", "membrane resistance"),
        ("R_a", "Ohm cm", "log", "[50, 1000]", "axial resistivity"),
        ("g_h", "S/cm2", "log", "[1e-6, 1e-3]", "I_h max. conductance"),
        ("dv_h", "mV", "linear", "[-10, +10]", "shift of m_inf(v)"),
        ("kappa_tau", "-", "log", "[0.5, 2.0]", "scale of tau_h(v)"),
    ]
    head = ("parameter", "unit", "coordinate", "box", "meaning")
    cols = [0.0, 0.16, 0.31, 0.47, 0.65]
    y = 0.95
    for c, h_ in zip(cols, head):
        tx.text(c, y, h_, fontsize=12, fontweight="bold", color=INK, transform=tx.transAxes)
    for i, r in enumerate(rows):
        y = 0.86 - i * 0.085
        colour = INK if i < 3 else ORANGE
        if i == 3:
            tx.plot([0, 1], [y + 0.055, y + 0.055], color=GRID, lw=1.2, transform=tx.transAxes)
        for c, v in zip(cols, r):
            tx.text(c, y, v, fontsize=11.5, color=colour if c == 0.0 else INK2, transform=tx.transAxes,
                    fontweight="bold" if c == 0.0 else "normal")
    tx.text(0.0, 0.30, "Passive fit (production, run B): 3 parameters, I_h absent from the model.\n"
            "I_h fit (ih6 arm): all 6. Base kinetics Ih_human (Rich 2021); dv_h and kappa_tau\n"
            "move the published curve; the leak reversal is not a 7th parameter: per segment\n"
            "e_pas(s) = V_rest + g_h(s) m_inf(V_rest) (V_rest - E_h) / g_pas(s),\n"
            "so the model rests at the cell's V_rest for every trial g_h.",
            fontsize=11, color=INK2, transform=tx.transAxes, va="top")
    tx.text(0.0, 0.01, "Boxes: param_spec.py defaults (Ih Fit) and the passive fitter. Optimiser coordinate:\n"
            "q = log(theta) (dv_h linear), uniform prior in q.", fontsize=9.5, color=INK3,
            transform=tx.transAxes)
    save(fig, "fig03_model_parameters.png")


# =============================================================================
def fig04_ih():
    v = np.linspace(-130.0, -40.0, 451)
    curves = [
        ("Rich 2021, human (Ih_human; the base)", BLUE, dict(mechanism="Ih_human"), dict(mechanism="Ih_human")),
        ("Kole 2006, rat (Ih)", ORANGE, dict(mechanism="Ih"), dict(mechanism="Ih")),
        ("Kole + Kalmbach 2018 shift (+20 mV)", AQUA, dict(mechanism="Ih", vshift_mV=20.0),
         dict(mechanism="Ih", vshift_mV=20.0)),
    ]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, H))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.84, bottom=0.17, wspace=0.25)
    fig.suptitle("I_h kinetics: three published curves disagree at rest, so the fit moves the curve itself",
                 x=0.02, ha="left", fontsize=16, fontweight="bold", color=INK)
    # boxes of the knobs around the base curve
    lo = np.array([hip.minf(x, "Ih_human", vshift_minf_mV=-10.0) for x in v])
    hi = np.array([hip.minf(x, "Ih_human", vshift_minf_mV=+10.0) for x in v])
    a1.fill_between(v, lo, hi, color=BLUE, alpha=0.12, lw=0, label="fitted dv_h box [-10, +10] mV")
    tlo = np.array([hip.mtau_ms(x, "Ih_human", tau_scale=0.5) for x in v])
    thi = np.array([hip.mtau_ms(x, "Ih_human", tau_scale=2.0) for x in v])
    a2.fill_between(v, tlo, thi, color=BLUE, alpha=0.12, lw=0, label="fitted kappa_tau box [0.5, 2.0]")
    at_rest = []
    for name, col, km, kt in curves:
        m = np.array([hip.minf(x, **km) for x in v])
        t = np.array([hip.mtau_ms(x, **kt) for x in v])
        a1.plot(v, m, color=col, lw=2.2, label=name)
        a2.plot(v, t, color=col, lw=2.2, label=name)
        at_rest.append((name, hip.minf(V_REST, **km), hip.mtau_ms(V_REST, **kt)))
    for a in (a1, a2):
        a.axvline(V_REST, color=INK3, lw=1.0, ls=(0, (3, 3)))
        a.set_xlabel("membrane potential v (mV, LJP-corrected)")
    a1.text(V_REST + 1, 0.03, "V_rest\n-73.5 mV", color=INK3, fontsize=10)
    a1.set_ylabel("steady-state activation m_inf(v)")
    a1.set_ylim(0, 1)
    a1.set_title("activation", loc="left")
    a2.set_ylabel("time constant tau_h(v) (ms)")
    a2.set_ylim(0, 750)
    a2.set_title("speed", loc="left")
    a1.legend(loc="upper right", fontsize=10.5)
    a2.legend(loc="upper left", fontsize=10.5)
    a2.text(-129, 420, "the published human formula falls below 1 ms\nbelow about -107 mV (no floor: mTauMin = 0)",
            fontsize=9.5, color=INK3)
    txt = "At V_rest: " + ";  ".join("%s m_inf %.3f, tau_h %.0f ms" % (n.split(" (")[0].split(",")[0], m, t)
                                      for n, m, t in at_rest)
    fig.text(0.07, 0.02, txt + "\nComputed with human_ih_params.py, the fitter's own reference kinetics; "
             "the fitted knobs act on the human base: m_inf(v - dv_h), kappa_tau * tau_h(v).",
             fontsize=9.5, color=INK3)
    save(fig, "fig04_ih_kinetics.png")
    return at_rest


# =============================================================================
def fig06_passive():
    layers = ["L2", "L3", "L4", "L5", "L6"]
    n = [10, 18, 6, 8, 5]
    cm = [1.478, 1.062, 0.785, 0.797, 1.100]
    rm = [17773, 26220, 44647, 51064, 42057]
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(W, H), gridspec_kw={"width_ratios": [1.15, 1.15, 0.9]})
    fig.subplots_adjust(left=0.06, right=0.98, top=0.80, bottom=0.24, wspace=0.32)
    fig.suptitle("Passive-only fit on 68 Allen cells (run B, the estimate of record): excitatory medians by layer",
                 x=0.02, ha="left", fontsize=16, fontweight="bold", color=INK)
    x = np.arange(len(layers))
    a1.bar(x, cm, width=0.6, color=BLUE)
    for i, (c, k) in enumerate(zip(cm, n)):
        a1.text(i, c + 0.03, "%.2f" % c, ha="center", fontsize=11, color=INK)
        a1.text(i, -0.17, "n=%d" % k, ha="center", fontsize=10, color=INK3)
    a1.fill_between([-0.4, 1.4], 0.44, 0.50, color=ORANGE, alpha=0.25, lw=0)
    a1.plot([-0.4, 1.4], [0.47, 0.47], color=ORANGE, lw=2)
    a1.text(1.75, 1.40, "orange line and band:\nEyal 2016, human L2/3,\n0.47 +/- 0.03 (mean +/- SD,\nn = 6 model fits)",
            color=ORANGE, fontsize=10, va="center")
    a1.set_xticks(x, layers)
    a1.set_ylabel("C_m median (uF/cm2)")
    a1.set_ylim(0, 1.75)
    a1.set_title("C_m: L2 ~3x, L3 ~2.3x the human L2/3 value", loc="left", fontsize=13)
    a2.bar(x, np.array(rm) / 1000.0, width=0.6, color=BLUE)
    for i, r in enumerate(rm):
        a2.text(i, r / 1000.0 + 0.8, "%.1f" % (r / 1000.0), ha="center", fontsize=11, color=INK)
    a2.set_xticks(x, layers)
    a2.set_ylabel("R_m median (kOhm cm2)")
    a2.set_title("R_m by layer", loc="left", fontsize=13)
    w = [("C_m", 0.131), ("R_a", 0.186), ("R_m", 0.592)]
    a3.barh([k for k, _ in w], [v_ for _, v_ in w], color=BLUE, height=0.55)
    for i, (_, v_) in enumerate(w):
        a3.text(v_ + 0.01, i, "%.3f" % v_, va="center", fontsize=11, color=INK)
    a3.set_xlim(0, 0.75)
    a3.set_xlabel("median relative width of the\n95 % bootstrap percentile interval")
    a3.set_title("R_m is the least determined\n(31 cells, 3-D bootstrap)", loc="left", fontsize=13)
    a3.grid(axis="y", visible=False)
    fig.text(0.06, 0.025, "Source: annual report part 1, section 6 (medians computed by the assistant from the cluster's "
             "phase2_results.csv; run B = dt01ls60, free R_a;\ninhibitory groups not shown). Eyal et al. 2016, eLife "
             "5:e16553, p. 2. Passive-only values absorb the h-current into C_m and R_m:\nan effective, not a "
             "membrane, capacitance (annual report sections 4.2 and 6).", fontsize=9.5, color=INK3)
    save(fig, "fig06_passive_results.png")


# =============================================================================
def fig07_noise():
    dt = 1.0 / 50000.0
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, H), gridspec_kw={"width_ratios": [1.0, 1.35]})
    fig.subplots_adjust(left=0.07, right=0.98, top=0.82, bottom=0.18, wspace=0.25)
    fig.suptitle("The recording-noise law each twin sweep must carry: fast AR(1) + slow OU + per-sweep tilt",
                 x=0.02, ha="left", fontsize=16, fontweight="bold", color=INK)
    lags = np.unique(np.round(np.geomspace(1, 50000, 400)).astype(int))
    t_ms = lags * dt * 1e3
    fast = THETA_CF.sigma_f ** 2 * THETA_CF.rho_f ** lags
    slow = THETA_CF.sigma_s ** 2 * np.exp(-lags * dt / THETA_CF.tau_s)
    tot = nl.autocovariance(THETA_CF, dt, lags)
    a1.plot(t_ms, tot * 1e3, color=INK, lw=2.2, label="total gamma(k | theta)")
    a1.plot(t_ms, fast * 1e3, color=BLUE, lw=2.0, label="fast AR(1): sigma_f^2 rho_f^k")
    a1.plot(t_ms, slow * 1e3, color=ORANGE, lw=2.0, label="slow OU: sigma_s^2 exp(-k dt / tau_s)")
    a1.set_xscale("log")
    a1.set_xlabel("lag (ms)")
    a1.set_ylabel("autocovariance (1e-3 mV^2)")
    a1.set_title("autocovariance at theta_cf", loc="left")
    a1.legend(fontsize=10.5, loc="upper right")
    v0 = THETA_CF.sigma_f ** 2 + THETA_CF.sigma_s ** 2
    a1.text(0.03, 0.42, "lag 0 (not on a log axis): total %.2f, of which fast %.0f %%"
            % (v0 * 1e3, 100 * THETA_CF.sigma_f ** 2 / v0), transform=a1.transAxes, fontsize=10, color=INK2)
    rng = np.random.default_rng(20261103)
    n = 50000
    units = nl.draw_units(rng, 1, n)
    b = 0.1                                            # mV/s, a tilt of the size studied in the tilt document
    tot_tr = nl.simulate_traces(THETA_CF, dt, units.fast, units.slow, tilt_mV_per_s=np.array([b]))[0]
    slow_law = nl.NoiseLaw(0.0, THETA_CF.rho_f, THETA_CF.sigma_s, THETA_CF.tau_s)
    slow_tr = nl.simulate_traces(slow_law, dt, units.fast, units.slow, tilt_mV_per_s=np.array([b]))[0]
    t = np.arange(n) * dt
    a2.plot(t, tot_tr, color=INK3, lw=0.4, label="one simulated sweep (all terms)")
    a2.plot(t, slow_tr, color=ORANGE, lw=1.8, label="slow OU + tilt (same draws)")
    a2.plot(t, b * (t - t.mean()), color=VIOLET, lw=1.6, ls=(0, (5, 3)), label="tilt b (t - t_bar), b = 0.1 mV/s")
    a2.set_xlabel("time in the sweep (s)")
    a2.set_ylabel("noise (mV)")
    a2.set_title("one simulated 1 s sweep at 50 kHz", loc="left")
    a2.legend(fontsize=10.5, loc="upper left", ncol=1)
    fig.text(0.07, 0.025, "theta_cf = (sigma_f 0.0380 mV, rho_f 0.175 at 50 kHz, sigma_s 0.0508 mV, tau_s 36.6 ms): "
             "the cohort closed-form law (L3_exc).\nPlotted with the repository's noise_law.py (Block 2); every sweep "
             "carries its own real tilt, a fixed number (D-102, D-107). Seed 20261103.", fontsize=9.5, color=INK3)
    save(fig, "fig07_noise_law.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", required=True, help="a roadmap.py status --json output for allen-mtg-fit")
    args = ap.parse_args()
    fig02_roadmap(args.status)
    fig03_model()
    at_rest = fig04_ih()
    for name, m, t in at_rest:
        print("at V_rest %.1f mV: %-45s m_inf %.3f  tau_h %.1f ms" % (V_REST, name, m, t))
    fig06_passive()
    fig07_noise()


if __name__ == "__main__":
    main()
