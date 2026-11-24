"""Can a compact soma imitate the pilot's fitted long one by moving its depth and the far slope?

Standalone check (not pipeline code). Target: the modelled grey levels B_k tau_k of the ellipsoid the pilot's soma fit
returned (c 15 um, mu 0.029 /um, slope 0.367, z0 26.76 um; soma_fit_529878215.json, 2026-11-24 upload). Candidate:
the same x0, y0, a, b, phi with c = C_CANDIDATE and mu scaled to keep 2 mu c; its z0 and slope fitted by least
squares to the target over the fit region's disc (13 um), all 35 planes. Reports the remaining RMS difference (gl)
per plane and the fitted z0 and slope, beside the fit's own residual RMS (3.5-6.3 gl per plane, the run's summary
figure). Usage: python3 tradeoff.py PATH/TO/soma_fit_529878215.json
"""
import dataclasses
import json
import sys

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..", "src"))
from allen_diameter.config import default_config                       # noqa: E402
from allen_diameter.model.soma_render import SomaRenderer             # noqa: E402

C_CANDIDATES = (7.26, 10.0)            # um; sqrt(a b) = 7.26 um, a round soma
PLANE_DZ_UM, REGION_UM, SUB = 0.28, 13.0, 3     # every 3rd pixel of the disc (speed); all planes

run = json.load(open(sys.argv[1]))
fit, crop, kern, s = run["fit"], run["crop"], run["kernel"], run["settings"]
theta = np.array(fit["theta"], dtype=float)
ks = np.array(fit["ks"], dtype=int)
B = np.array(fit["B_gl"], dtype=float)
cfg0 = default_config()
rcfg = dataclasses.replace(cfg0.renderer, sigma_r0_um=kern["sigma_r0_um"],
                           kernel_table_delta_um=tuple(kern["table_delta_um"]),
                           kernel_table_sigma_um=tuple(kern["table_sigma_um"]),
                           kernel_continuation=kern["continuation"], kernel_continuation_slope=kern["continuation_slope"])
scfg = dataclasses.replace(cfg0.soma_fit, fine_factor=s["fine_factor"], dzeta_um=s["dzeta_um"])
R = SomaRenderer((crop["left"], crop["top"], crop["width"], crop["height"]), crop["res0_um"], PLANE_DZ_UM * ks, rcfg, scfg)
p = crop["res0_um"]
yy, xx = np.mgrid[0:crop["height"], 0:crop["width"]]
X, Y = (crop["left"] + xx) * p, (crop["top"] + yy) * p
x0, y0, z0, a, b, c, phi, mu, gam = theta
disc = ((X - x0) ** 2 + (Y - y0) ** 2 <= REGION_UM ** 2) & (yy % SUB == 0) & (xx % SUB == 0)


def grey(th8, slope):
    return (B[:, None, None] * R.render(np.asarray(th8, dtype=float), slope=slope))[:, disc]


target = grey(theta[:8], gam)
for cc in C_CANDIDATES:
    def resid(q):
        return (grey([x0, y0, q[0], a, b, cc, phi, mu * c / cc], q[1]) - target).ravel()
    r0 = np.sqrt(np.mean(resid([z0, gam]) ** 2))
    sol = least_squares(resid, [z0 + (c - cc) / 2.0, gam], bounds=([z0 - 12, 0.05], [z0 + 18, 3.0]),
                        x_scale=[1.0, 0.1], max_nfev=40)
    per = np.sqrt(np.mean(sol.fun.reshape(len(ks), -1) ** 2, axis=1))
    print("c %.2f um (mu %.4f /um): RMS %.2f gl with z0 and slope as fitted; after fitting them, z0 %.2f um, slope "
          "%.3f: RMS %.2f gl, per plane %.2f-%.2f gl (median %.2f)"
          % (cc, mu * c / cc, r0, sol.x[0], sol.x[1], np.sqrt(np.mean(sol.fun ** 2)), per.min(), per.max(),
             np.median(per)))
