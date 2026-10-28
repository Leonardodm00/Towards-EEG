"""draw_decision_map.py -- SCHEMATIC (not to scale, no data): where the two open
choices of 2026-10-23 sit in the noise-calibration chain of the Stage-7 gate
(Ih Fit/specs/SPEC.md Blocks 3-5; decision log D-034, D-051, D-076).

Run:  python draw_decision_map.py   ->  noise_fit_decision_map.png
Pure ASCII source; the labels use matplotlib mathtext.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ---------------------------------------------------------------- config
OUT = "noise_fit_decision_map.png"
W, H = 11.0, 13.8                       # figure size, inches
C_DATA, C_BUILT, C_TODO, C_GATE, C_DECF = "#e8eef7", "#e9f3e6", "#f4f4f4", "#f7ecec", "#fff4e8"
C_DEC = "#d9822b"                       # border of the two open-choice nodes
FS_T, FS_S, FS_L = 10.5, 8.6, 8.6       # box title, status line, edge label font sizes

# key -> (x_centre, y_centre, width, height, title, status line, fill, is_open_choice)
NODES = {
    "ss":   (2.6, 12.0, 4.9, 0.95, "Archive: stored SS windows, $(-10,+200)$ ms\n37-pulse trains; $K$ = 37-348 per cell",
             "data inspected (D-072)", C_DATA, False),
    "ls":   (8.4, 12.0, 4.1, 0.95, "Archive: Long Square sweeps\npre-window $[t_0,\\,t_{\\rm on}-5$ ms$)$, 265 ms",
             "data inspected", C_DATA, False),
    "d1":   (2.6, 10.45, 4.9, 1.15, "(1) WINDOW-START RULE\n$t_{\\rm SS}(c)=\\max(80\\ {\\rm ms},\\ t_{\\rm tr}(c))$,\n"
             "transient test on $\\bar r_{\\rm odd}(t)$", "Block 3: built, tested; open choice (a) / (b) / (c)", C_DECF, True),
    "win":  (2.6, 8.9, 4.9, 1.0, "SS noise windows $[t_{\\rm SS}(c),\\,200)$ ms,\nminus the per-polarity mean window (D-051.1)",
             "Block 3: computed (not yet run on real cells)", C_BUILT, False),
    "vh":   (1.6, 7.4, 2.7, 0.95, "window means $\\rightarrow$\n$\\hat V_c(h^*)$, $h^*$ = 0.802 s", "Block 3", C_BUILT, False),
    "qss":  (4.6, 7.4, 2.9, 0.95, "5 SS statistics\n$\\hat\\sigma$, $\\hat\\rho$ at 4 lags", "Block 3", C_BUILT, False),
    "qls":  (8.4, 7.4, 4.1, 0.95, "10 LS statistics\n(full window, first 50 ms, last 132.5 ms)", "Block 3", C_BUILT, False),
    "d2":   (2.4, 5.8, 4.4, 1.15, "(2) TILT ESTIMATOR (T1)\n"
             "$\\hat\\sigma_b(c)^2=2[\\hat V_c(h^*)-V(h^*|\\hat\\theta_c,0)]/h^{*2}$",
             "Block 4: built, tested; open choice (a) / (b) / (c)", C_DECF, True),
    "q":    (7.8, 5.8, 4.2, 0.9, "cell statistic vector $\\hat q(c)$, $d$ = 15", "Block 3: computed", C_BUILT, False),
    "fit":  (4.9, 3.75, 4.6, 1.15, "per-cell fit $\\hat\\theta(c)=(\\sigma_f,\\rho_f,\\sigma_s,\\tau_s)$\n"
             "indirect inference; its simulator cuts\nthe same windows (length set by $t_{\\rm SS}$)",
             "Block 4: built, tested", C_BUILT, False),
    "ps":   (4.9, 2.2, 4.6, 0.95, "per-sweep stage (D-076): $\\sigma_f,\\rho_f$, shrunk $\\sigma_s$\nper sweep; $\\tau_s$, $\\sigma_b$ per cell",
             "Block 4b: not built", C_TODO, False),
    "tw":   (5.6, 0.75, 8.2, 0.95, "twins: each twin sweep gets its real sweep's law, plus a tilt\n"
             "$b\\sim\\mathcal{N}(0,\\hat\\sigma_b(c)^2)$ drawn per sweep, SS and LS sweeps alike (D-034)",
             "Block 5: not built", C_TODO, False),
    "gate": (5.6, -0.6, 8.2, 0.8, "Stage 7 gate: does the 6-parameter $I_h$ fit recover the twins' known truth?",
             "built (D-011..D-016), never run", C_GATE, False),
}

# (from, side, frac, to, side, frac, label, label_dx, label_dy, dashed, curvature)
EDGES = [
    ("ss", "b", 0.5, "d1", "t", 0.5, "", 0, 0, False, 0.0),
    ("d1", "b", 0.5, "win", "t", 0.5, "", 0, 0, False, 0.0),
    ("win", "b", 0.30, "vh", "t", 0.5, "", 0, 0, False, 0.0),
    ("win", "b", 0.70, "qss", "t", 0.5, "", 0, 0, False, 0.0),
    ("ls", "b", 0.5, "qls", "t", 0.5, "", 0, 0, False, 0.0),
    ("qss", "b", 0.5, "q", "t", 0.2, "", 0, 0, False, 0.0),
    ("qls", "b", 0.5, "q", "t", 0.7, "", 0, 0, False, 0.0),
    ("vh", "b", 0.5, "d2", "t", 0.318, "", 0, 0, False, 0.0),
    ("q", "b", 0.5, "fit", "t", 0.85, "", 0, 0, False, 0.0),
    ("d2", "b", 0.70, "fit", "t", 0.18, "$\\hat\\sigma_b$ into\nits simulator", -0.95, 0.0, False, 0.0),
    ("fit", "t", 0.40, "d2", "b", 0.93, "$\\hat\\theta$ into\n$V(h^*|\\hat\\theta,0)$", 1.05, 0.0, True, -0.35),
    ("fit", "b", 0.5, "ps", "t", 0.5, "", 0, 0, False, 0.0),
    ("ps", "b", 0.5, "tw", "t", 0.48, "", 0, 0, False, 0.0),
    ("d2", "b", 0.10, "tw", "t", 0.03, "$\\hat\\sigma_b(c)$", -0.45, 0.0, False, 0.0),
    ("tw", "b", 0.5, "gate", "t", 0.5, "", 0, 0, False, 0.0),
]


def point(key, side, frac):
    x, y, w, h = NODES[key][:4]
    if side == "t":
        return (x - w / 2 + frac * w, y + h / 2)
    if side == "b":
        return (x - w / 2 + frac * w, y - h / 2)
    if side == "l":
        return (x - w / 2, y + h / 2 - frac * h)
    return (x + w / 2, y + h / 2 - frac * h)


def box(ax, key):
    x, y, w, h, title, status, fill, dec = NODES[key]
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.03,rounding_size=0.12",
                                fc=fill, ec=C_DEC if dec else "#555555", lw=2.6 if dec else 1.0, zorder=2))
    ax.text(x, y + 0.11, title, ha="center", va="center", fontsize=FS_T, zorder=3,
            fontweight="bold" if dec else "normal")
    ax.text(x, y - h / 2 + 0.14, status, ha="center", va="center", fontsize=FS_S, style="italic",
            color="#333333", zorder=3)


def edge(ax, e):
    a, sa, fa, b, sb, fb, label, ldx, ldy, dashed, rad = e
    p, q = point(a, sa, fa), point(b, sb, fb)
    ax.annotate("", xy=q, xytext=p, zorder=1,
                arrowprops=dict(arrowstyle="-|>", lw=1.2, color="#444444", ls="--" if dashed else "-",
                                connectionstyle="arc3,rad=%g" % rad, shrinkA=1, shrinkB=1))
    if label:
        mx, my = (p[0] + q[0]) / 2 + ldx, (p[1] + q[1]) / 2 + ldy
        ax.text(mx, my, label, fontsize=FS_L, ha="center", va="center", color="#333333", zorder=4)


def main():
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, 10.8)
    ax.set_ylim(-1.2, 13.55)
    ax.axis("off")
    for e in EDGES:
        edge(ax, e)
    for k in NODES:
        box(ax, k)
    ax.text(0.1, 13.45, "SCHEMATIC -- where the two open choices sit in the noise calibration of the Stage-7 twins",
            fontsize=11.5, fontweight="bold", va="top")
    ax.text(0.1, 13.05, "orange border = open choice;  blue = archive data;  green = built (Blocks 3-4, tested in the "
            "sandbox, not yet run on real cells);  grey = not built;  red = the gate", fontsize=8.8, va="top",
            color="#333333")
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
