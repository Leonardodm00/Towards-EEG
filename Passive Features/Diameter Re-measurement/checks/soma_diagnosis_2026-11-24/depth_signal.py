"""How much the soma's depth extent changes its brightfield images, against how much the fit misses the data.

Standalone check (not pipeline code). Renders, with Block 45's SomaRenderer and the pilot's settings (the crop, the 35
planes k = 83..151 step 2 at 0.28 um, the kernel table of the run), the ellipsoid the pilot's soma fit returned and
alternatives with the same projected absorbance 2 mu c, and reports per plane the RMS of the difference of their
modelled grey levels B_k tau_k over the fit region's disc (13 um about the centre), in gl.

Truth chosen (no data involved): the fitted theta of soma_fit_529878215.json (2026-11-24 upload), B_k of that run.
Usage: python3 depth_signal.py PATH/TO/soma_fit_529878215.json
"""
import dataclasses
import json
import sys

import numpy as np

sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..", "src"))
from allen_diameter.config import default_config                       # noqa: E402
from allen_diameter.model.soma_render import SomaRenderer             # noqa: E402

CFG = dict(plane_dz_um=0.28, region_radius_um=13.0, c_round_um=None)  # c_round None: sqrt(a b)

run = json.load(open(sys.argv[1]))
fit, crop, kern = run["fit"], run["crop"], run["kernel"]
theta = np.array(fit["theta"], dtype=float)          # x0 y0 z0 a b c phi mu slope
ks = np.array(fit["ks"], dtype=int)
B = np.array(fit["B_gl"], dtype=float)
z = CFG["plane_dz_um"] * ks
cfg0 = default_config()
rcfg = dataclasses.replace(cfg0.renderer, sigma_r0_um=kern["sigma_r0_um"],
                           kernel_table_delta_um=tuple(kern["table_delta_um"]),
                           kernel_table_sigma_um=tuple(kern["table_sigma_um"]),
                           kernel_continuation=kern["continuation"],
                           kernel_continuation_slope=kern["continuation_slope"])
s = run["settings"]
scfg = dataclasses.replace(cfg0.soma_fit, fine_factor=s["fine_factor"], dzeta_um=s["dzeta_um"])
R = SomaRenderer((crop["left"], crop["top"], crop["width"], crop["height"]), crop["res0_um"], z, rcfg, scfg)

p = crop["res0_um"]
yy, xx = np.mgrid[0:crop["height"], 0:crop["width"]]
X, Y = (crop["left"] + xx) * p, (crop["top"] + yy) * p
x0, y0, z0, a, b, c, phi, mu, gam = theta
disc = (X - x0) ** 2 + (Y - y0) ** 2 <= CFG["region_radius_um"] ** 2
c_r = CFG["c_round_um"] or float(np.sqrt(a * b))
z_tr = run["traced"]["centre_um"][2]


def grey(th8, slope):
    return B[:, None, None] * R.render(np.asarray(th8, dtype=float), slope=slope)


def rms(d):
    return np.sqrt(np.mean(d[:, disc] ** 2, axis=1))


fitted = grey(theta[:8], gam)
cases = {
    "c only (c %.2f, mu x %.2f; z0, slope as fitted)" % (c_r, c / c_r):
        grey([x0, y0, z0, a, b, c_r, phi, mu * c / c_r], gam),
    "compact soma at the traced depth (c %.2f, z0 %.2f, slope 0.79)" % (c_r, z_tr):
        grey([x0, y0, z_tr, a, b, c_r, phi, mu * c / c_r], 0.79),
    "slope only (0.79; c 15 as fitted)":
        grey(theta[:8], 0.79),
}
print("projected absorbance through the centre, 2 mu c: fitted %.3f (transmission %.3f)" % (2 * mu * c, np.exp(-2 * mu * c)))
print("modelled grey at the centre pixel, plane nearest z0 and nearest the traced depth:")
iz0, itr = int(np.argmin(abs(z - z0))), int(np.argmin(abs(z - z_tr)))
ic = np.unravel_index(np.argmin((X - x0) ** 2 + (Y - y0) ** 2), X.shape)
print("  fitted: %.1f gl (k %d), %.1f gl (k %d); B_k %.1f" % (fitted[iz0][ic], ks[iz0], fitted[itr][ic], ks[itr], B[itr]))
for name, img in cases.items():
    d = rms(img - fitted)
    print("%-66s RMS difference over the disc: median %.2f gl, range %.2f-%.2f gl (35 planes)"
          % (name, np.median(d), d.min(), d.max()))
    print("    per plane (k: gl): " + ", ".join("%d: %.1f" % (k, v) for k, v in zip(ks[::4], d[::4])))
