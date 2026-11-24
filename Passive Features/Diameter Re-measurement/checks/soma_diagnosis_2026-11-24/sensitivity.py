"""How far a few grey levels of structured background move the soma fit's depth parameters (linear response).

Standalone check (not pipeline code). At a compact soma that renders nearly the pilot's fitted planes (tradeoff.py:
c 7.26 um, z0 28.39 um, far slope 0.513, mu scaled to keep 2 mu c; x0, y0, a, b, phi as fitted), the Jacobian J of
the model B_k tau_k(theta) over the fit region R (rebuilt as region_makeup.py does: approximate) is taken by forward
differences with the fit's own steps (Block 46's diff_steps), the 35 plane scales B_k included as linear parameters
(variable projection's exact counterpart). For a background pattern e (gl) added to the data, the least-squares
estimate moves by (J^T J)^-1 J^T e to first order. Patterns, each 1 gl in size:
  wedge    +1 gl on R's pixels in the top-left quadrant outside the silhouette, every plane
  wedge_k  the same, its amplitude falling linearly from +1 (top plane) to -0.3 (bottom plane)
  tilt     a linear gradient across the crop, 1 gl per 10 um toward the top-left, every plane
Usage: python3 sensitivity.py SOMA_FIT_JSON CORRECTED_SWC
"""
import dataclasses
import json
import sys

import numpy as np

sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..", "src"))
from allen_diameter.config import default_config                       # noqa: E402
from allen_diameter.model.soma_render import SomaRenderer             # noqa: E402
from allen_diameter.loading import swc_io                             # noqa: E402
from allen_diameter.analysis import soma_view                         # noqa: E402

COMPACT = dict(c=7.26, z0=28.39, slope=0.513)
CFG = dict(plane_dz_um=0.28, region_um=13.0, widen_um=2.0, rim_um=1.0)
NAMES = ("x0", "y0", "z0", "a", "b", "c", "phi", "mu", "slope")

run = json.load(open(sys.argv[1]))
swc = swc_io.read_swc(sys.argv[2])
fit, crop, kern, s = run["fit"], run["crop"], run["kernel"], run["settings"]
th_fit = np.array(fit["theta"], dtype=float)
ks = np.array(fit["ks"], dtype=int)
B = np.array(fit["B_gl"], dtype=float)
left, top, W, H, p = crop["left"], crop["top"], crop["width"], crop["height"], crop["res0_um"]
cfg0 = default_config()
rcfg = dataclasses.replace(cfg0.renderer, sigma_r0_um=kern["sigma_r0_um"],
                           kernel_table_delta_um=tuple(kern["table_delta_um"]),
                           kernel_table_sigma_um=tuple(kern["table_sigma_um"]),
                           kernel_continuation=kern["continuation"], kernel_continuation_slope=kern["continuation_slope"])
scfg = dataclasses.replace(cfg0.soma_fit, fine_factor=s["fine_factor"], dzeta_um=s["dzeta_um"])
R = SomaRenderer((left, top, W, H), p, CFG["plane_dz_um"] * ks, rcfg, scfg)

# the region, rebuilt
xyz = np.asarray(swc.xyz, dtype=float)
index_of = {int(i): k for k, i in enumerate(swc.ids)}
parent_index = np.array([index_of.get(int(pp), -1) for pp in np.asarray(swc.parent)])
w = CFG["widen_um"]
seg = soma_view.trace_segments(xyz, np.asarray(swc.radius), parent_index, np.asarray(swc.types),
                               (left * p - w, (left + W) * p + w, top * p - w, (top + H) * p + w))
mask = soma_view.segment_mask((H, W), left, top, p, seg["p1"], seg["p2"], seg["r1"] + w, seg["r2"] + w)
cs = run["start"]["centre_um"]
region = soma_view.disk_mask((H, W), left, top, p, cs[:2], CFG["region_um"]) & ~mask

x0, y0, z0, a, b, c, phi, mu, gam = th_fit
theta = np.array([x0, y0, COMPACT["z0"], a, b, COMPACT["c"], phi, mu * c / COMPACT["c"], COMPACT["slope"]])
steps = np.array(s["diff_steps"], dtype=float)


def tau(th):
    return R.render(th[:8], slope=th[8])[:, region]                      # (K, n_R)


t0 = tau(theta)
K, n = t0.shape
J = np.zeros((K * n, 9 + K))
for i in range(9):
    th = theta.copy()
    th[i] += steps[i]
    J[:, i] = ((B[:, None] * (tau(th) - t0)) / steps[i]).ravel()
for k in range(K):                                                       # the plane scales, linear
    col = np.zeros((K, n))
    col[k] = t0[k]
    J[:, 9 + k] = col.ravel()

yy, xx = np.mgrid[0:H, 0:W]
X, Y = ((left + xx) * p)[region], ((top + yy) * p)[region]
u = (X - x0) * np.cos(phi) + (Y - y0) * np.sin(phi)
v = -(X - x0) * np.sin(phi) + (Y - y0) * np.cos(phi)
rho = np.sqrt((u / a) ** 2 + (v / b) ** 2)
outside = (rho - 1.0) * np.sqrt(u ** 2 + v ** 2) / np.maximum(rho, 1e-9) > CFG["rim_um"]
wedge = ((X < cs[0]) & (Y < cs[1]) & outside).astype(float)
ramp = np.linspace(1.0, -0.3, K)
gx, gy = -(X - cs[0]) / 10.0, -(Y - cs[1]) / 10.0                        # toward the top-left (y down)
tilt = (gx + gy) / np.sqrt(2.0)
patterns = {"wedge": np.tile(wedge, (K, 1)), "wedge_k": ramp[:, None] * wedge[None, :], "tilt": np.tile(tilt, (K, 1))}

M = J.T @ J
print("compact point: c %.2f um, z0 %.2f um, slope %.3f, mu %.4f /um; R %d px x %d planes; wedge %.0f %% of R"
      % (theta[5], theta[2], theta[8], theta[7], n, K, 100 * wedge.mean()))
for name, e in patterns.items():
    d = np.linalg.solve(M, J.T @ e.ravel())
    print("%-8s (RMS %.2f gl over R): per gl of pattern  dz0 %+.2f um, dc %+.2f um, dslope %+.3f, dmu %+.4f /um, "
          "da %+.2f um, db %+.2f um" % (name, np.sqrt(np.mean(e ** 2)), d[2], d[5], d[8], d[7], d[3], d[4]))
