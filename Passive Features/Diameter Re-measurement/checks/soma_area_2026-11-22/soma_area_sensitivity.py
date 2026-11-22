"""Soma area -- how much does an error in the soma's membrane area bias the somatic passive fit? Check of 2026-11-22.

Context: D-210 (the soma first) and D-211 (its plane by eye). The soma's outline in Allen's brightfield planes is a
silhouette: on every plane it shows the blurred projection of the whole soma, not its section (a simulated sphere:
checks/soma_focus_2026-11-22), so the soma's height, volume and membrane area need a shape model, and their error
is large. This check asks how much that error matters for the fit the refined morphology serves.

Model and fit: those of checks/diameter_reach_2026-11-22/diameter_reach.py (the fitter's PassiveCell rebuilt; the
pulse and the step targets; the linearised bias by one Gauss-Newton step, dq = -(J^T W J)^{-1} J^T W [y(q0, D_s) -
y(q0, D)], q = log(Cm, Rm, Ra)), with D_s the morphology whose soma's 3-d diameters are scaled by s at a fixed length:
the soma's membrane area times s (Import3d builds the one-point soma as a cylinder of length = diameter = 2 r).
Two parameter sets: Eyal et al. 2016 cell 060308 (as D-205), and Rm, Cm matched to the cell's R_in and tau (the
reach check's "matched" run). A full Gauss-Newton refit at one scale checks the linearisation.

Usage:  python3 soma_area_sensitivity.py <reconstruction.swc> <out dir> [eyal|matched] [--refit S]
Pure ASCII, LF.
"""
import argparse
import json
import math
import os
import sys
import time
from dataclasses import replace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "diameter_reach_2026-11-22"))
import diameter_reach as DR                                   # noqa: E402

SCALES = (0.5, 0.75, 0.86, 1.26, 1.5, 2.0)    # 0.86, 1.26: the oblate spheroid and the cylinder of the 2026-11-22 notes
with open(os.path.join(HERE, "..", "diameter_reach_2026-11-22", "diameter_reach_matched.json")) as _f:
    _m = json.load(_f)["config"]                              # the reach check's matched run: R_in 54.9 MOhm, tau 20.5 ms
CASES = {"eyal": {}, "matched": dict(cm_uF_cm2=_m["cm_uF_cm2"], rm_ohm_cm2=_m["rm_ohm_cm2"])}


def soma_area(cell):
    h = cell["h"]
    return float(sum(h.area(seg.x, sec=sec) for sec in cell["soma"] for seg in sec))


def effective_membrane(cell):
    """sum over every segment of its area times its F multiplier (um2): the membrane the fit charges."""
    h = cell["h"]
    return float(sum(h.area(seg.x, sec=sec) * cell["mult"][(sec.name(), k)]
                     for sec in h.allsec() for k, seg in enumerate(sec)))


def scale_soma(cell, s):
    h = cell["h"]
    for sec in cell["soma"]:
        for i in range(int(sec.n3d())):
            h.pt3dchange(i, float(sec.diam3d(i) * s), sec=sec)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("swc")
    ap.add_argument("out_dir")
    ap.add_argument("case", nargs="?", default="eyal", choices=sorted(CASES))
    ap.add_argument("--refit", type=float, default=None, help="a full Gauss-Newton refit at this soma scale")
    a = ap.parse_args(argv)
    cfg = replace(DR.ReachConfig(), **CASES[a.case])
    names = tuple(cfg.protocols)
    t_all = time.time()
    cell = DR.build_cell(a.swc, cfg)
    q0 = np.log([cfg.cm_uF_cm2, cfg.rm_ohm_cm2, cfg.ra_ohm_cm])
    A0, M0 = soma_area(cell), effective_membrane(cell)
    print("[%s] Cm %.3f uF/cm2, Rm %.0f Ohm cm2, Ra %.1f Ohm cm; soma area %.1f um2 (4 pi r^2 with r 5.97: %.1f); "
          "membrane charged by the fit %.0f um2 (the soma %.2f %%)"
          % (a.case, cfg.cm_uF_cm2, cfg.rm_ohm_cm2, cfg.ra_ohm_cm, A0, 4 * math.pi * 5.9694 ** 2, M0, 100 * A0 / M0))
    data = DR.targets(cell, q0, names, cfg)
    J = DR.jacobian(cell, q0, names, cfg)
    rows = []
    d0 = [np.array([sec.diam3d(i) for i in range(int(sec.n3d()))]) for sec in cell["soma"]]

    def restore():
        h = cell["h"]
        for sec, d in zip(cell["soma"], d0):
            for i in range(d.size):
                h.pt3dchange(i, float(d[i]), sec=sec)

    for s in SCALES:
        scale_soma(cell, s)
        A = soma_area(cell)
        y = DR.targets(cell, q0, names, cfg)
        restore()
        dy = {b: y[b] - data[b] for b in names}
        row = dict(scale=s, soma_area_um2=A, soma_area_ratio=A / A0,
                   rmsd_pulse_mV=float(np.sqrt(np.mean(dy["pulse"] ** 2))),
                   rmsd_step_mV=float(np.sqrt(np.mean(dy["step"] ** 2))))
        for key, sub in DR.PROTOCOL_SETS.items():
            dq = DR.gauss_newton_step(J, dy, sub)
            for j, p in enumerate(DR.PARAMS):
                row["%s_%s" % (p, key)] = float(math.exp(dq[j]) - 1.0)
        rows.append(row)
    print(" area x  soma um2 | fit on the pulse: dCm   dRm   dRa  | fit on pulse+step: dCm   dRm   dRa   (relative)")
    for r in rows:
        print("  %4.2f   %6.1f  |  %+6.3f %+6.3f %+6.3f |  %+6.3f %+6.3f %+6.3f"
              % (r["scale"], r["soma_area_um2"], r["Cm_pulse"], r["Rm_pulse"], r["Ra_pulse"],
                 r["Cm_pulse+step"], r["Rm_pulse+step"], r["Ra_pulse+step"]))
    check = None
    if a.refit is not None:
        scale_soma(cell, a.refit)
        out = {}
        for key, sub in DR.PROTOCOL_SETS.items():
            qh, it = DR.refit(cell, q0, {b: data[b] for b in sub}, sub, cfg)
            out[key] = dict(iterations=it, **{p: float(math.exp(qh[j] - q0[j]) - 1.0) for j, p in enumerate(DR.PARAMS)})
        restore()
        check = dict(scale=a.refit, refit=out)
        for key, v in out.items():
            lin = [r for r in rows if r["scale"] == a.refit]
            print("refit at area x %.2f, fit on %s (%d iterations): dCm %+.3f dRm %+.3f dRa %+.3f%s"
                  % (a.refit, key, v["iterations"], v["Cm"], v["Rm"], v["Ra"],
                     ("   (linear: %+.3f %+.3f %+.3f)" % tuple(lin[0]["%s_%s" % (p, key)] for p in DR.PARAMS))
                     if lin else ""))
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "soma_area_%s.json" % a.case), "w") as f:
        json.dump(dict(case=a.case, config=dict(cm_uF_cm2=cfg.cm_uF_cm2, rm_ohm_cm2=cfg.rm_ohm_cm2,
                                                ra_ohm_cm=cfg.ra_ohm_cm, F=cfg.F),
                       soma_area_um2=A0, membrane_um2=M0, rows=rows, refit_check=check,
                       seconds=time.time() - t_all), f, indent=1)
    print("[%s] %.0f s" % (a.case, time.time() - t_all))
    return 0


if __name__ == "__main__":
    sys.exit(main())
