"""Figure 3 (schematic, drawn to scale by code): a tube's cross-section, the recorded planes, the band of one
plane, and the blur with which each depth reaches that plane. python3 fig03_band_cross_section.py"""
import dataclasses
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle
from _common import FIG
from allen_diameter.config import default_config
from allen_diameter.model.kernel import sigma_r

# ---- CONFIG ----
D_UM = 2.441            # tube diameter (node 8089's flat-tube test)
DZ_PLANE = 0.28         # um, plane spacing
OFFSET = 1.032          # um, depth of plane k from the tube's axis (plane k = +2 of the test)
BETA = 0.14             # um, band half-thickness
TABLE_DELTA = (0.0, 0.28, 0.56, 0.84)
TABLE_SIGMA = (0.0800, 0.1536, 0.3924, 0.5620)
GAMMA = 0.79
# ----------------

r = D_UM / 2
rcfg = dataclasses.replace(default_config().renderer, kernel_table_delta_um=TABLE_DELTA, kernel_table_sigma_um=TABLE_SIGMA,
                           sigma_r0_um=TABLE_SIGMA[0], kernel_continuation_slope=GAMMA)
fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.6, 4.4), gridspec_kw=dict(width_ratios=[1, 1.25]))
circ = Circle((0, 0), r, fc="#cfcfcf", ec="0.3", lw=0.8)
ax.add_patch(circ)
band = Rectangle((-r, OFFSET - BETA), 2 * r, 2 * BETA, fc="#1d9e75", ec="none")
ax.add_patch(band)
band.set_clip_path(circ)
planes = OFFSET - DZ_PLANE * np.arange(0, 9)
for z in planes:
    ax.axhline(z, color="0.45", lw=0.5, ls=":")
ax.axhline(OFFSET, color="#0f6e56", lw=1.4)
ax.set_xlim(-1.8, 1.8)
ax.set_ylim(1.45, -1.45)
ax.set_xlabel("offset $v$ across the tube (um)")
ax.set_ylabel(r"depth $z - z_{\rm ax}$ (um)")
ax.set_aspect("equal")
ax.plot([], [], "-", color="#0f6e56", lw=1.4, label="plane $k$ (depth $z_k$)")
ax.plot([], [], ":", color="0.45", lw=0.8, label="other recorded planes, 0.28 um apart")
ax.set_title("cross-section, $d$ = %.2f um" % D_UM, fontsize=9)
ax.plot([], [], "s", color="#1d9e75", label=r"in-band: $|\zeta - z_k| \leq \beta$")
ax.plot([], [], "s", color="#cfcfcf", label="out-of-band: the rest of the node")
ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False)
zz = np.linspace(-r, r, 400)
s = sigma_r(zz - OFFSET, rcfg)
bx.fill_betweenx(zz, -s, s, color="#cfcfcf")
inb = np.abs(zz - OFFSET) <= BETA
bx.fill_betweenx(zz[inb], -s[inb], s[inb], color="#1d9e75")
for z in planes:
    bx.axhline(z, color="0.45", lw=0.5, ls=":")
bx.axhline(OFFSET, color="#0f6e56", lw=1.4)
bx.set_xlim(-1.9, 1.9)
bx.set_ylim(1.45, -1.45)
bx.set_ylabel(r"depth $\zeta - z_{\rm ax}$ of the node's material (um)")
bx.set_xlabel(r"$\pm\sigma_{\rm r}(\zeta - z_k)$: blur with which depth $\zeta$ reaches plane $k$ (um)")
bx.set_title("blur onto plane $k$", fontsize=9)
fig.subplots_adjust(wspace=0.3)
fig.savefig(FIG / "fig03_band_cross_section.png", dpi=170, bbox_inches="tight")
print("wrote", FIG / "fig03_band_cross_section.png")
