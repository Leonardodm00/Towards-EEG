"""Figure 4 (plotted): the own shadow on the measuring line split into its in-band and out-of-band parts, at two
planes of the flat-tube test of node 8089. python3 fig04_split_profiles.py  (about 20 s)"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from _common import FIG, line_geometry, load_check

# ---- CONFIG ----
THETA = (2.441, 0.0751, 0.0, -0.472)   # d um, mu 1/um, v0 um, z_ax um (node 8089 after round 1 of the 17 Nov run)
GAMMA = 0.79                           # far slope; the table is the pilot's (the check module's TABLE_*)
PLANES = (-2, 2)                       # the two planes shown, of ks = -4..4
# beta, B and the tube's other settings: the check module's BETA_UM, B_GL (0.14 um, 92 gl; flat, hybrid)
# ----------------

chk = load_check()
from allen_diameter.config import default_config  # noqa: E402
cfg, cache = chk.make_cfg(GAMMA), chk.LF.SectionCache()
geom = line_geometry(default_config, np.arange(-4, 5))
v, ks = geom[3], geom[4]
S = chk.B_GL * chk.shadow(THETA, cfg, None, True, geom, cache)
Sin = chk.B_GL * chk.shadow(THETA, cfg, chk.BETA_UM, True, geom, cache)
Sout = chk.B_GL * chk.shadow(THETA, cfg, chk.BETA_UM, False, geom, cache)
fig, axes = plt.subplots(1, len(PLANES), figsize=(9.2, 3.6), sharey=True)
for ax, k in zip(axes, PLANES):
    i = int(np.flatnonzero(ks == k)[0])
    off = k * cfg.acquisition.dz_um - THETA[3]
    chord = 2 * np.sqrt(max((THETA[0] / 2) ** 2 - off ** 2, 0.0))
    ax.plot(v, S[i], color="k", lw=1.6, label=r"whole, $B_k S_k(v_n \mid \theta)$")
    ax.plot(v, Sout[i], color="#888780", lw=1.4, ls="--", label=r"out-of-band, $B_k S^{\rm out}_k(v_n \mid \theta, \beta)$")
    ax.plot(v, Sin[i], color="#1d9e75", lw=1.8, label=r"in-band, $B_k S^{\rm in}_k(v_n \mid \theta, \beta)$")
    ax.plot([-chord / 2, chord / 2], [-0.7, -0.7], color="#0f6e56", lw=3, solid_capstyle="butt")
    ax.text(0, -1.5, "chord %.2f um" % chord, ha="center", fontsize=7.5, color="#0f6e56")
    ax.set_title(r"plane $k$ = %+d,  $z_k - z_{\rm ax}$ = %+.3f um" % (k, off), fontsize=9)
    ax.set_xlabel("offset $v_n$ along the measuring line (um)")
    print("k %+d: centre whole %.2f, in %.2f, out %.2f gl; chord %.2f um" % (k, S[i, 26], Sin[i, 26], Sout[i, 26], chord))
axes[0].set_ylabel("light removed (gl, $B_k$ = %.0f gl)" % chk.B_GL)
axes[0].set_ylim(-2.2, 21)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, fontsize=8, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.08))
fig.savefig(FIG / "fig04_split_profiles.png", dpi=170, bbox_inches="tight")
print("wrote", FIG / "fig04_split_profiles.png")
