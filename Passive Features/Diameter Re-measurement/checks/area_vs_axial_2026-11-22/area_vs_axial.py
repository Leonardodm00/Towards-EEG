"""Area or diameter? -- check of 2026-11-22 (the user, 16:53: "the compartment area is what determines the quality of
the fit"). Within path distance x <= X_UM of soma(0.5), two perturbations with the same change of membrane load:
  (a) every dendritic 3-d diameter times s: membrane area x s, axial conductance (d^2 / L) x s^2;
  (b) membrane only: cm and g_pas of those segments times s (the area's load x s), the axial resistance unchanged.
If the area alone set the fit, (a) and (b) would shift the fitted (Cm, Rm, Ra) alike. The model, the targets and the
linearised shift are those of checks/diameter_reach_2026-11-22/diameter_reach.py; two parameter sets (Eyal 2016
060308; matched to the cell's R_in and tau). Pure ASCII.

    python3 area_vs_axial.py <reconstruction.swc>
"""
import json
import math
import os
import sys
from dataclasses import replace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "diameter_reach_2026-11-22"))
import diameter_reach as DR                                   # noqa: E402

X_UM, S = 250.0, 1.2
with open(os.path.join(HERE, "..", "diameter_reach_2026-11-22", "diameter_reach_matched.json")) as f:
    _m = json.load(f)["config"]
CASES = {"eyal": {}, "matched": dict(cm_uF_cm2=_m["cm_uF_cm2"], rm_ohm_cm2=_m["rm_ohm_cm2"])}


def within(cell):
    return [k for k, dist in enumerate(cell["dist_um"]) if dist <= X_UM]


def run(swc, case):
    cfg = replace(DR.ReachConfig(), **CASES[case])
    names = tuple(cfg.protocols)
    cell = DR.build_cell(swc, cfg)
    h = cell["h"]
    q0 = np.log([cfg.cm_uF_cm2, cfg.rm_ohm_cm2, cfg.ra_ohm_cm])
    data = DR.targets(cell, q0, names, cfg)
    J = DR.jacobian(cell, q0, names, cfg)
    sel = within(cell)
    out = {}
    # (a) diameters x S
    for k in sel:
        sec, d = cell["dendritic"][k], cell["d0"][k]
        for i in range(d.size):
            h.pt3dchange(i, float(d[i] * S), sec=sec)
    y = DR.targets(cell, q0, names, cfg)
    DR.restore_diameters(cell)
    out["diameters"] = {b: y[b] - data[b] for b in names}
    # (b) membrane only x S
    saved = dict(cell["mult"])
    for k in sel:
        sec = cell["dendritic"][k]
        for j, seg in enumerate(sec):
            cell["mult"][(sec.name(), j)] *= S
    y = DR.targets(cell, q0, names, cfg)
    cell["mult"].clear()
    cell["mult"].update(saved)
    DR.set_passive(cell, q0, cfg)
    out["membrane"] = {b: y[b] - data[b] for b in names}
    print("[%s] %d of %d dendritic sections within %.0f um; perturbation x %.2f" % (case, len(sel),
                                                                               len(cell["dendritic"]), X_UM, S))
    print("   perturbation  | fit on the pulse: dCm   dRm   dRa  | fit on pulse+step: dCm   dRm   dRa  | rmsd pulse, step (mV)")
    for key, dy in out.items():
        row = []
        for sub in DR.PROTOCOL_SETS.values():
            dq = DR.gauss_newton_step(J, dy, sub)
            row += [math.exp(v) - 1 for v in dq]
        print("   %-12s  |  %+6.3f %+6.3f %+6.3f |  %+6.3f %+6.3f %+6.3f | %.4f, %.4f"
              % (key, *row, float(np.sqrt(np.mean(dy["pulse"] ** 2))), float(np.sqrt(np.mean(dy["step"] ** 2)))))


if __name__ == "__main__":
    for c in ("eyal", "matched"):
        run(sys.argv[1], c)
