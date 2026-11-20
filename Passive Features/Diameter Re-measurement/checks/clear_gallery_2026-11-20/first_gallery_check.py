#!/usr/bin/env python3
"""Standalone check (NOT pipeline code), 2026-11-20: the first real run of the clearness screen (cell 529878215,
Cell 21, 33 candidates; D-194) and the evidence for D-198 (the lines along the dendrite leave the verdict).

Reads the run's gallery_<S>.json (the features the screen stored: the measuring line on every plane, the lines along
the dendrite at k*) and the preparation's cleaning_<S>.json (the first stage's tilt phi from the trace). No image,
no model. Prints:
  1. the verdict as run (D-194) and the verdict of D-198 (the measuring line on k* - 1 .. k* + 1 and the depth
     floor, with the run's thresholds), recomputed from the stored features;
  2. the tilt signature on the lines along the dendrite at k*: where along the axis the dip is narrowest, how much
     wider it is 2 um away, and the rank correlation of the width range with the trace's tilt.
Run:  python3 -I first_gallery_check.py <gallery_S.json> <cleaning_S.json>
"""
import json
import math
import sys

import numpy as np
from scipy.stats import spearmanr


def arr(x):
    return np.array([np.nan if v is None else v for v in x], dtype=float)


def main(gallery_json, cleaning_json):
    g = json.load(open(gallery_json))
    c = json.load(open(cleaning_json))
    q = g["clearness"]
    nodes = sorted(g["nodes"].items(), key=lambda kv: int(kv[0]))
    print("thresholds: outside_max %.2f, skew_max %.2f, min_depth_sd %.0f, planes k* +- %d"
          % (q["outside_max"], q["skew_max"], q["min_depth_sd"], q["focus_half_planes"]))
    print("\n1. verdicts")
    print("node   Allen  as run (D-194)        D-198: E_line  S_line  depth(SD)  verdict")
    passed = []
    for nid, n in nodes:
        foc = np.array(n["focus"], bool)
        E, S = arr(n["line"]["outside"])[foc], arr(n["line"]["skew"])[foc]
        rE = E.max() / q["outside_max"] if np.all(np.isfinite(E)) else math.inf
        rS = S.max() / q["skew_max"] if np.all(np.isfinite(S)) else math.inf
        ok = max(rE, rS) <= 1.0 and n["depth_sd"] >= q["min_depth_sd"]
        if ok:
            passed.append(int(nid))
        print("%-6s %.2f   %-20s  %6.2f  %6.2f  %8.1f   %s" % (nid, n["d_allen_um"], "clean" if n["clean"] else
              "rejected Q %s" % ("%.2f" % n["score"] if n["score"] is not None else "inf"), np.max(E), np.max(S),
              n["depth_sd"], "CLEAN" if ok else "rejected"))
    print("as run: %d judged clean; D-198: %d judged clean: %s" % (sum(n["clean"] for _, n in nodes), len(passed),
                                                                  passed))

    print("\n2. the lines along the dendrite at k* (u = -2 .. 2 um)")
    tilt, UW, at0, ratio = [], [], 0, []
    n_ok = 0
    for nid, n in nodes:
        u, W = arr(n["u_um"]), arr(n["along"]["fwhm_um"])
        phi = math.degrees(float(c["nodes"][nid]["pilot"]["phi_rad"]))
        if np.all(np.isfinite(W)):
            n_ok += 1
            i_min = int(np.argmin(W))
            at0 += abs(u[i_min]) <= 0.5 + 1e-9
            ratio.append(0.5 * (W[0] + W[-1]) / W[u == 0][0])
        if n["width_range"] is not None:
            tilt.append(phi)
            UW.append(n["width_range"])
    rho, p = spearmanr(tilt, UW)
    print("nodes with a finite width on all nine lines: %d; narrowest within 0.5 um of the node: %d" % (n_ok, at0))
    print("width at u = +-2 um over the width at the node: median %.2f, range %.2f .. %.2f"
          % (np.median(ratio), np.min(ratio), np.max(ratio)))
    print("Spearman rho (width range U_W, the trace's tilt phi): %.2f, p %.2f, n %d" % (rho, p, len(UW)))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
