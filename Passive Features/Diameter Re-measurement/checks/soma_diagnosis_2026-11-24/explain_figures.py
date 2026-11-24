"""Figures for the plain walkthrough of the soma's area (Blocks 45-47) and of the SGT kernel in the node fit (Block 49,
Block 30 amendment 3). Plotted from the package's own functions (SomaRenderer, kernel.sigma_r, ellipsoid_area,
sgt_kernel_pdf) and the pilot's results; standalone, not pipeline code.

Usage: python3 explain_figures.py SOMA_JSON INFERENCE_CSV OUT_DIR
  SOMA_JSON      soma_fit_529878215.json (Cell 25, the user's upload of 2026-11-24)
  INFERENCE_CSV  inference_529878215.csv of OUT/inference_sgt (Cell 26, same upload)
Writes fig1_soma_area.png and fig2_sgt_kernel.png.
"""
import csv
import dataclasses
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                         # noqa: E402
import numpy as np                                                      # noqa: E402
from scipy.ndimage import map_coordinates                               # noqa: E402

sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..", "src"))
from allen_diameter.config import default_config                        # noqa: E402
from allen_diameter.model import blur_kernels, kernel                   # noqa: E402
from allen_diameter.model.soma_render import SomaRenderer              # noqa: E402
from allen_diameter.analysis.soma_fit import ellipsoid_area             # noqa: E402

PARAMS = dict(plane_dz_um=0.28,
              compact=dict(c_um=7.26, z0_um=28.39, slope=0.513),   # tradeoff.py's best match to the fitted images
              profile_planes=(83, 101, 119, 137, 151),
              sgt_shapes=[("Gaussian (2, 0, 1)", 2.0, 0.0, 1.0, "-"),
                          ("flat top (10, 0, 1)", 10.0, 0.0, 1.0, "--"),
                          ("heavy tails (2, 1, 1)", 2.0, 1.0, 1.0, ":"),
                          ("skewed (2, 0, 1.5)", 2.0, 0.0, 1.5, "-.")],
              tube_defocus_um=0.56)
INK, INK2, BLUE, ORANGE, RED, GREY = "#1a1a1a", "#555555", "#2a6fdb", "#e07a1f", "#c0392b", "#9a9a9a"

soma_path, inf_path, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out_dir, exist_ok=True)
run = json.load(open(soma_path))
fit, crop, kern, s = run["fit"], run["crop"], run["kernel"], run["settings"]
th = np.array(fit["theta"], dtype=float)
x0, y0, z0, a, b, c, phi, mu, gam = th
ks = np.array(fit["ks"], dtype=int)
B = np.array(fit["B_gl"], dtype=float)
zk = PARAMS["plane_dz_um"] * ks
cfg0 = default_config()
rcfg = dataclasses.replace(cfg0.renderer, sigma_r0_um=kern["sigma_r0_um"],
                           kernel_table_delta_um=tuple(kern["table_delta_um"]),
                           kernel_table_sigma_um=tuple(kern["table_sigma_um"]),
                           kernel_continuation=kern["continuation"], kernel_continuation_slope=kern["continuation_slope"])
scfg = dataclasses.replace(cfg0.soma_fit, fine_factor=s["fine_factor"], dzeta_um=s["dzeta_um"])
traced = run["traced"]
z_lo_bound = traced["centre_um"][2] - s["shift_z_um"]
c_hi_bound = s["axis_bounds_um"][1]

# ------------------------------------------------------------------ figure 1: the soma ---
fig, ax = plt.subplots(2, 2, figsize=(12.5, 9.6), layout="constrained")

# (a) side view along the long in-plane axis
A = ax[0, 0]
t = np.linspace(0, 2 * np.pi, 400)
cp = PARAMS["compact"]
for zc, cc, col, lab in ((z0, c, ORANGE, "fitted: c = %.1f um (bound %g), z0 = %.2f um (bound %.2f)" % (c, c_hi_bound, z0, z_lo_bound)),
                         (cp["z0_um"], cp["c_um"], BLUE, "a compact soma with nearly the same images: c = %.2f um" % cp["c_um"])):
    A.plot(a * np.cos(t), zc + cc * np.sin(t), color=col, lw=2, label=lab)
A.plot(traced["radius_um"] * np.cos(t), traced["centre_um"][2] + traced["radius_um"] * np.sin(t), color="#b8860b",
       ls=":", lw=1.5, label="Allen's traced sphere, r = %.2f um" % traced["radius_um"])
for z in zk:
    A.axhline(z, color=GREY, lw=0.4, alpha=0.6)
A.axhspan(zk.min(), zk.max(), color=GREY, alpha=0.08, lw=0, label="the 35 planes fitted (z = 0.28 k um)")
A.axhline(z_lo_bound, color=RED, lw=1, ls="--", label="z0's lower bound (start - 6 um)")
A.annotate("light", xy=(-13.5, zk.min() - 3), xytext=(-13.5, zk.min() - 9), color=INK2, ha="center",
           arrowprops=dict(arrowstyle="->", color=INK2))
A.set_xlim(-15, 15); A.set_ylim(z0 - c - 2, max(zk.max(), z0 + c) + 2)
A.set_aspect("equal"); A.invert_yaxis()
A.set_xlabel("u, along the soma's long in-plane axis (um)"); A.set_ylabel("depth z (um, stage units)")
A.set_title("(a) the ellipsoid seen from the side (plotted from the fit)", fontsize=10)
A.legend(fontsize=7, loc="lower right")

# (b) the kernel's width against defocus
A = ax[0, 1]
d = np.linspace(0, 16, 400)
for g, col, lab in ((kern["continuation_slope"], INK, "configured far slope 0.79 (optics model, D-024; not calibrated on Allen's data)"),
                    (gam, ORANGE, "far slope fitted on the pilot's soma, %.3f" % gam),
                    (cp["slope"], BLUE, "far slope matching the compact soma, %.3f" % cp["slope"])):
    rc = dataclasses.replace(rcfg, kernel_continuation_slope=float(g))
    A.plot(d, [kernel.sigma_r(x, rc) for x in d], color=col, lw=1.8, label=lab)
A.axvspan(0, kern["table_delta_um"][-1], color=GREY, alpha=0.25, lw=0, label="near table: Cell 7's calibration on the pilot (0-0.84 um)")
A.set_xlabel("defocus |zeta - z_k| of a slab from the plane (um)"); A.set_ylabel("kernel width sigma_r (um)")
A.set_title("(b) how much a slab's shadow is blurred at each distance from the plane", fontsize=10)
A.legend(fontsize=7, loc="upper left")

# (c) the modelled grey level along the long axis, at five planes: twin cases
A = ax[1, 0]
p = crop["res0_um"]
Rr = SomaRenderer((crop["left"], crop["top"], crop["width"], crop["height"]), p, zk, rcfg, scfg)
img_fit = B[:, None, None] * Rr.render(th[:8], slope=gam)
img_cmp = B[:, None, None] * Rr.render([x0, y0, cp["z0_um"], a, b, cp["c_um"], phi, mu * c / cp["c_um"]],
                                       slope=cp["slope"])
u = np.linspace(-13, 13, 261)
cols = (x0 + u * math.cos(phi)) / p - crop["left"]
rows = (y0 + u * math.sin(phi)) / p - crop["top"]
cmap = plt.get_cmap("viridis")
for i, k in enumerate(PARAMS["profile_planes"]):
    j = int(np.where(ks == k)[0][0])
    col = cmap(i / (len(PARAMS["profile_planes"]) - 1))
    A.plot(u, map_coordinates(img_fit[j], [rows, cols], order=1), color=col, lw=1.8,
           label="z = %.1f um (k %d)" % (zk[j], k))
    A.plot(u, map_coordinates(img_cmp[j], [rows, cols], order=1), color=col, lw=1.2, ls="--")
A.plot([], [], color=INK2, lw=1.8, label="fitted (c 15 um)")
A.plot([], [], color=INK2, lw=1.2, ls="--", label="compact (c 7.26 um)")
A.set_xlabel("u, along the long axis through the centre (um)"); A.set_ylabel("modelled grey level B_k tau_k (gl)")
A.set_title("(c) the soma's shadow at five planes: dark at every one (model, no data)", fontsize=10)
A.legend(fontsize=7, ncol=2, loc="lower left")

# (d) the area against the depth semi-axis
A = ax[1, 1]
cc = np.linspace(1.0, 20.0, 200)
S = np.array([ellipsoid_area(a, b, x) for x in cc])
A.plot(cc, S, color=INK, lw=1.8, label="S(a = %.2f, b = %.2f, c) um^2" % (a, b))
A.axhline(4 * math.pi * traced["radius_um"] ** 2, color="#b8860b", ls=":", lw=1.5,
          label="Allen's sphere 4 pi r^2 = %.0f um^2" % (4 * math.pi * traced["radius_um"] ** 2))
A.axvline(c_hi_bound, color=RED, ls="--", lw=1, label="c's upper bound, 15 um")
A.plot([c], [ellipsoid_area(a, b, c)], "o", color=ORANGE, ms=8, label="the pilot's fit: %.0f um^2" % ellipsoid_area(a, b, c))
A.axvspan(b, a, color=BLUE, alpha=0.15, lw=0,
          label="c between b and a: %.0f-%.0f um^2" % (ellipsoid_area(a, b, b), ellipsoid_area(a, b, a)))
A.set_xlabel("depth semi-axis c (um)"); A.set_ylabel("membrane area S (um^2)")
A.set_title("(d) the area follows c, the axis the images barely see", fontsize=10)
A.legend(fontsize=7, loc="upper left")
fig.suptitle("Figure 1. The soma's area: an ellipsoid rendered like the microscope, fitted to 35 planes; S from (a, b, c)",
             fontsize=11)
fig.savefig(os.path.join(out_dir, "fig1_soma_area.png"), dpi=130)
plt.close(fig)

# ------------------------------------------------------------------ figure 2: the SGT kernel ---
row = list(csv.DictReader(open(inf_path)))[0]
beta5, xi5 = float(row["beta"]), float(row["xi"])
shapes = PARAMS["sgt_shapes"] + [("node %s's fit (%.1f, %.2f, 1)" % (row["node_id"], beta5, xi5), beta5, xi5, 1.0, "-")]
fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.6), layout="constrained")
x = np.linspace(-3, 3, 1201)
for (lab, be, xi, ze, ls), col in zip(shapes, (INK, BLUE, "#2a9d8f", "#8338ec", ORANGE)):
    g = blur_kernels.sgt_kernel_pdf(1.0, be, xi, ze)
    ax[0].plot(x, g(x), color=col, ls=ls, lw=1.8, label=lab)
    ax[1].semilogy(x, np.maximum(g(x), 1e-6), color=col, ls=ls, lw=1.8)
ax[0].set_title("(a) kernels of the same FWHM (1 unit), shape (beta, xi, zeta)", fontsize=10)
ax[0].set_xlabel("position across the measuring line (FWHM units)"); ax[0].set_ylabel("kernel g (1/unit)")
ax[0].legend(fontsize=7)
ax[1].set_ylim(1e-4, 3); ax[1].set_title("(b) the same, log scale: the tails", fontsize=10)
ax[1].set_xlabel("position (FWHM units)")
# (c) a 1-D tube profile under two kernels (schematic: one kernel for the whole tube, no slabs)
d5, mu5 = float(row["d_hat_um"]), float(row["mu_hat_per_um"])
r = d5 / 2
v = np.linspace(-2.5, 2.5, 2001)
dv = v[1] - v[0]
chord = 2 * np.sqrt(np.clip(r * r - v * v, 0, None))
T = np.exp(-mu5 * chord)
w = 2 * math.sqrt(2 * math.log(2)) * kernel.sigma_r(PARAMS["tube_defocus_um"], rcfg)
prof = {}
for lab, be, xi, col in (("Gaussian", 2.0, 0.0, INK), ("node %s's SGT" % row["node_id"], beta5, xi5, ORANGE)):
    g = blur_kernels.sgt_kernel_pdf(w, be, xi, 1.0)(v)
    g = g / (g.sum() * dv)
    prof[lab] = np.convolve(T - 1.0, g, mode="same") * dv + 1.0
    ax[2].plot(v, prof[lab], color=col, lw=1.8, label="%s kernel, FWHM %.2f um" % (lab, w))
diff = prof["node %s's SGT" % row["node_id"]] - prof["Gaussian"]
ax[2].plot(v, 1 + 10 * diff - 0.06, color=RED, lw=1, label="difference x 10 (shifted)")
ax[2].set_xlabel("v, across the tube (um)"); ax[2].set_ylabel("transmittance (fraction of the background)")
ax[2].set_title("(c) a %.2f um tube %.2f um out of focus (schematic, 1-D)" % (d5, PARAMS["tube_defocus_um"]), fontsize=10)
ax[2].legend(fontsize=7, loc="lower left")
fig.suptitle("Figure 2. The SGT kernel: same width as the Gaussian law, a different shape; the node fit frees the shape",
             fontsize=11)
fig.savefig(os.path.join(out_dir, "fig2_sgt_kernel.png"), dpi=130)
plt.close(fig)
print("wrote", out_dir, "| max |SGT - Gaussian| profile difference %.4f of the background" % np.abs(diff).max())
