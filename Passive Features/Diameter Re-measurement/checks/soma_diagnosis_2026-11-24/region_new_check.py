"""How much the soma's depth shows on the pixels of the region the user chose (D-241): a disc of Allen's traced soma
radius, masked only where the traced branches lie (no widening). Standalone check (not pipeline code).

Renders with Block 45 (the run's crop, planes and kernel) the fitted soma of soma_fit_529878215.json (c 15 um) and two
alternatives: (A) c 7.26 um with mu scaled to keep 2 mu c, z0 and the far slope as fitted (the depth signal alone);
(B) the same with z0 28.39 um and slope 0.513 (tradeoff.py's imitation). RMS differences of B_k tau_k per plane are
reported over the old region (13 um disc, branches + 2 um masked) and the new one, both rebuilt from the corrected SWC
with the moved stems' inner stretch restored (approximate near the soma).
Usage: python3 region_new_check.py SOMA_FIT_JSON CORRECTED_SWC
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
R = SomaRenderer((left, top, W, H), p, 0.28 * ks, rcfg, scfg)
xyz = np.asarray(swc.xyz, dtype=float)
index_of = {int(i): k for k, i in enumerate(swc.ids)}
parent_index = np.array([index_of.get(int(pp), -1) for pp in np.asarray(swc.parent)])
rad = {int(i): float(r) for i, r in zip(swc.ids, swc.radius)}
st = [q for q in run["stems"] if q["moved_um"] > 0]


def region(radius_um, widen):
    seg = soma_view.trace_segments(xyz, np.asarray(swc.radius), parent_index, np.asarray(swc.types),
                                   (left * p - widen, (left + W) * p + widen, top * p - widen, (top + H) * p + widen))
    m = soma_view.segment_mask((H, W), left, top, p, seg["p1"], seg["p2"], seg["r1"] + widen, seg["r2"] + widen)
    r = np.array([rad.get(int(q["id"]), 0.5) for q in st])
    m |= soma_view.segment_mask((H, W), left, top, p, np.array([q["old_um"][:2] for q in st]),
                                np.array([q["new_um"][:2] for q in st]), r + widen, r + widen)
    cs = run["start"]["centre_um"]
    return soma_view.disk_mask((H, W), left, top, p, cs[:2], radius_um) & ~m


r_allen = float(run["traced"]["radius_um"])
regions = {"old (13 um disc, +2 um mask)": region(13.0, 2.0),
           "new (%.2f um disc, branches only)" % r_allen: region(r_allen, 0.0)}
x0, y0, z0, a, b, c, phi, mu, gam = theta
fitted = B[:, None, None] * R.render(theta[:8], slope=gam)
alts = {"A: c 7.26, z0 and slope as fitted": (7.26, z0, gam), "B: c 7.26, z0 28.39, slope 0.513": (7.26, 28.39, 0.513)}
for name, (cc, zz, gg) in alts.items():
    alt = B[:, None, None] * R.render(np.array([x0, y0, zz, a, b, cc, phi, mu * c / cc]), slope=gg)
    for rname, reg in regions.items():
        F, Aa = fitted[:, reg], alt[:, reg]
        per = np.sqrt(np.mean((Aa - F) ** 2, axis=1))
        # each plane's brightness re-solved for the alternative (variable projection, as the fit does)
        s_k = np.sum(F * Aa, axis=1) / np.sum(Aa * Aa, axis=1)
        per_vp = np.sqrt(np.mean((s_k[:, None] * Aa - F) ** 2, axis=1))
        print("%-34s on %-36s (%5d px): RMS per plane %.2f-%.2f gl (median %.2f); B_k re-solved: %.2f-%.2f "
              "(median %.2f)" % (name, rname, reg.sum(), per.min(), per.max(), np.median(per), per_vp.min(),
                                 per_vp.max(), np.median(per_vp)))
