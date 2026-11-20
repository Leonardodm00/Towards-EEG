#!/usr/bin/env python3
"""Standalone check (not pipeline code), 2026-11-20: two blur-free facts on the pilot's recorded data.

(1) The dip's area A_k = sum_n (B_k - I_k(v_n)) dv over the measuring line, per plane k (gl um). A normalized blur
    kernel moves light along the line but does not remove it, so for an isolated straight tube A_k should not
    depend on the plane (on defocus) nor on the in-focus width sigma_r(0).
(2) The area a tube of diameter d and absorption mu removes, per unit background, in the weak-to-moderate
    absorption model T(v) = exp(-mu c(v)), c(v) = 2 sqrt(r^2 - v^2) (the vertical chord; tilt ignored):
        a(mu, d) = int (1 - exp(-mu c(v))) dv   (um);   a ~ mu pi d^2 / 4 when mu d << 1.
    The first stage (d held at Allen's diameter) and the multi-plane fit (d free) of the same node must both
    reproduce the measured area, so a(mu_first, d_Allen) and a(mu_fit, d_fit) should agree although d differs.
Inputs: the fit folder of the pilot's run 2 (own_<S>_node<id>.npz: profiles (9, 53) gl, v (53,) um, B_gl (9,);
inference_<S>.json: theta = (d, mu, v0, z_ax)). The first stage's mu at Allen's diameter is taken from the clear-node handoff,
section 4 (the preparation's record is not in this folder).
Run: python3 -I area_check.py <inference_d169 folder>
"""
import json
import sys

import numpy as np
from scipy.integrate import quad

SPEC = "529878215"
# node: (Allen d um, first-stage mu /um at Allen's d) -- the clear-node handoff, section 4 [data via the handoff]
FIRST = {"1013": (0.449, 1.082), "5353": (0.454, 0.581), "2669": (0.788, 0.813), "2620": (0.811, 0.942),
         "639": (1.200, 1.241)}


def a_tube(mu, d):
    """int (1 - exp(-mu 2 sqrt(r^2 - v^2))) dv over |v| <= r (um)."""
    r = 0.5 * d
    val, _ = quad(lambda v: 1.0 - np.exp(-mu * 2.0 * np.sqrt(max(r * r - v * v, 0.0))), -r, r, limit=200)
    return val


def main(folder):
    rec = json.load(open("%s/inference_%s.json" % (folder, SPEC)))
    print("(1) the dip's area per plane, A_k / B_k (um): an isolated tube gives the same value at every plane")
    print("node   k*-4 .. k*+4                                                     CV     max/min")
    meas = {}
    for nid in FIRST:
        with np.load("%s/own_%s_node%s.npz" % (folder, SPEC, nid), allow_pickle=False) as z:
            P, v, B = z["profiles"].astype(float), z["v"].astype(float), z["B_gl"].astype(float)
        dv = float(np.median(np.diff(v)))
        a = ((B[:, None] - P).sum(axis=1) * dv) / B          # um, per unit background
        meas[nid] = a
        print("%-6s %s   %.3f  %.2f" % (nid, " ".join("%.3f" % x for x in a), a.std(ddof=1) / a.mean(), a.max() / a.min()))
    print("\n(2) the area each fit's tube removes, a(mu, d) (um), against the measured area at k* (the middle plane)")
    print("node   Allen d  mu_first  a(first)   fit d   mu_fit  a(fit)   measured A/B at k*   mu d^2: first  fit")
    for nid, (dA, mu1) in FIRST.items():
        n = rec["nodes"][nid]
        d2, mu2 = float(n["theta"][0]), float(n["theta"][1])          # theta = (d, mu, v0, z_ax)
        print("%-6s %.3f    %.3f     %.4f     %.3f   %.3f   %.4f   %.4f               %.4f  %.4f"
              % (nid, dA, mu1, a_tube(mu1, dA), d2, mu2, a_tube(mu2, d2), meas[nid][4], mu1 * dA ** 2, mu2 * d2 ** 2))


if __name__ == "__main__":
    main(sys.argv[1])
