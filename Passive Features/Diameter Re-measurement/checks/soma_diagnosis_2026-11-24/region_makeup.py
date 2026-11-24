"""What the pilot's soma fit actually sees: the fit region R (the 13 um disc minus the widened trace), split by where
its pixels lie relative to the fitted silhouette.

Standalone check (not pipeline code). Rebuilds fit_soma.py's mask (every traced process whose widened disc reaches the
crop, at its traced radius + 2 um; Block 44's segments) from the CORRECTED SWC of the 2026-11-24 upload
(reconstruction_soma_529878215.swc): the original SWC is not in the sandbox. The two differ only at the stems' first
segments (first nodes moved by up to 2.5 um, 4 inside nodes removed), so R is approximate there; the pixel count is
compared with the run's n_samples / |K| as a check of the rebuild.
Usage: python3 region_makeup.py SOMA_FIT_JSON CORRECTED_SWC
"""
import json
import sys

import numpy as np

sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..", "src"))
from allen_diameter.loading import swc_io                     # noqa: E402
from allen_diameter.analysis import soma_view                 # noqa: E402

CFG = dict(widen_um=2.0, region_um=13.0, rim_um=1.0)

run = json.load(open(sys.argv[1]))
swc = swc_io.read_swc(sys.argv[2])
crop, fit = run["crop"], run["fit"]
left, top, W, H, p = crop["left"], crop["top"], crop["width"], crop["height"], crop["res0_um"]
xyz = np.asarray(swc.xyz, dtype=float)            # transform is the identity here (shift 0, no flip, z0 0)
types = np.asarray(swc.types)
index_of = {int(i): k for k, i in enumerate(swc.ids)}
parent_index = np.array([index_of.get(int(pp), -1) for pp in np.asarray(swc.parent)])
w = CFG["widen_um"]
box = (left * p - w, (left + W) * p + w, top * p - w, (top + H) * p + w)
seg = soma_view.trace_segments(xyz, np.asarray(swc.radius), parent_index, types, box)
mask = soma_view.segment_mask((H, W), left, top, p, seg["p1"], seg["p2"], seg["r1"] + w, seg["r2"] + w)
cs = run["start"]["centre_um"]
region = soma_view.disk_mask((H, W), left, top, p, cs[:2], CFG["region_um"]) & ~mask
n_run = fit["n_samples"] / len(fit["ks"])
print("segments %d (run: %d); mask %.1f %% of the crop (run: %.1f %%); region %d px (run: %d px), %.1f um^2, "
      "%.1f %% of the 13 um disc" % (len(seg["r1"]), crop["n_mask_segments"], 100 * mask.mean(),
                                    100 * crop["mask_fraction"], region.sum(), n_run, region.sum() * p * p,
                                    100 * region.sum() / soma_view.disk_mask((H, W), left, top, p, cs[:2],
                                                                            CFG["region_um"]).sum()))
x0, y0, z0, a, b, c, phi, mu, gam = fit["theta"]
yy, xx = np.mgrid[0:H, 0:W]
X, Y = (left + xx) * p, (top + yy) * p
u = (X - x0) * np.cos(phi) + (Y - y0) * np.sin(phi)
v = -(X - x0) * np.sin(phi) + (Y - y0) * np.cos(phi)
rho = np.sqrt((u / a) ** 2 + (v / b) ** 2)        # 1 on the fitted silhouette
# distance to the silhouette, approximately, in um: (rho - 1) times the local radius
rloc = np.sqrt(u ** 2 + v ** 2) / np.maximum(rho, 1e-9)
dist = (rho - 1.0) * rloc
inside = dist < -CFG["rim_um"]
rim = np.abs(dist) <= CFG["rim_um"]
outside = dist > CFG["rim_um"]
sil = rho < 1.0
print("silhouette (a %.2f, b %.2f um): %.1f um^2, of which in R %.1f %%" % (a, b, sil.sum() * p * p,
                                                                       100 * (sil & region).sum() / sil.sum()))
for name, m in (("inside the silhouette, > 1 um from its edge", inside), ("within 1 um of the edge", rim),
                ("outside, > 1 um beyond the edge", outside)):
    print("  R's pixels %-46s %5.1f %%  (%.1f um^2)" % (name, 100 * (m & region).sum() / region.sum(),
                                                       (m & region).sum() * p * p))
ang = np.degrees(np.arctan2(Y - cs[1], X - cs[0]))   # image frame: y down
tl = region & (X < cs[0]) & (Y < cs[1]) & outside
print("  R's pixels outside the silhouette in the top-left quadrant (x < x_s, y < y_s): %.1f %% of R" %
      (100 * tl.sum() / region.sum()))
