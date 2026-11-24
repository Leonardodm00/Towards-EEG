"""The compact-versus-long difference on the pixels the pilot's soma fit actually uses.

Standalone check (not pipeline code). Renders with Block 45's SomaRenderer (the run's crop, 35 planes and kernel table)
the fitted soma of soma_fit_529878215.json and the two alternatives tradeoff.py found (same x0, y0, a, b, phi; mu
scaled to keep 2 mu c; z0 and the far slope re-fitted to imitate the fitted soma over the 13 um disc), and reports the
RMS difference of the modelled grey levels B_k tau_k per plane over (i) the disc and (ii) the fit region R rebuilt by
region_makeup.py's rule from the corrected SWC (approximate: 8610 px against the run's 7644).
Usage: python3 region_signal.py SOMA_FIT_JSON CORRECTED_SWC
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

ALTERNATIVES = ((7.26, 28.39, 0.513), (10.0, 28.45, 0.490))   # (c, z0, slope): tradeoff.py's fitted values
CFG = dict(plane_dz_um=0.28, region_um=13.0, widen_um=2.0)

run = json.load(open(sys.argv[1]))
swc = swc_io.read_swc(sys.argv[2])
fit, crop, kern, s = run["fit"], run["crop"], run["kernel"], run["settings"]
theta = np.array(fit["theta"], dtype=float)
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

xyz = np.asarray(swc.xyz, dtype=float)
index_of = {int(i): k for k, i in enumerate(swc.ids)}
parent_index = np.array([index_of.get(int(pp), -1) for pp in np.asarray(swc.parent)])
w = CFG["widen_um"]
seg = soma_view.trace_segments(xyz, np.asarray(swc.radius), parent_index, np.asarray(swc.types),
                               (left * p - w, (left + W) * p + w, top * p - w, (top + H) * p + w))
mask = soma_view.segment_mask((H, W), left, top, p, seg["p1"], seg["p2"], seg["r1"] + w, seg["r2"] + w)
cs = run["start"]["centre_um"]
disc = soma_view.disk_mask((H, W), left, top, p, cs[:2], CFG["region_um"])
region = disc & ~mask

x0, y0, z0, a, b, c, phi, mu, gam = theta
fitted = B[:, None, None] * R.render(theta[:8], slope=gam)
print("fitted: centre-pixel shadow depth at the plane nearest z0: %.1f gl below B" %
      (B.max() - fitted[np.argmin(abs(CFG["plane_dz_um"] * ks - z0))].min()))
for cc, zz, gg in ALTERNATIVES:
    alt = B[:, None, None] * R.render(np.array([x0, y0, zz, a, b, cc, phi, mu * c / cc]), slope=gg)
    d = alt - fitted
    on_disc = np.sqrt(np.mean(d[:, disc] ** 2, axis=1))
    on_R = np.sqrt(np.mean(d[:, region] ** 2, axis=1))
    print("c %.2f, z0 %.2f, slope %.3f: RMS difference per plane over the disc %.2f-%.2f gl (median %.2f), over R "
          "%.2f-%.2f gl (median %.2f), over R and all planes %.2f gl"
          % (cc, zz, gg, on_disc.min(), on_disc.max(), np.median(on_disc), on_R.min(), on_R.max(), np.median(on_R),
             np.sqrt(np.mean(d[:, region] ** 2))))
