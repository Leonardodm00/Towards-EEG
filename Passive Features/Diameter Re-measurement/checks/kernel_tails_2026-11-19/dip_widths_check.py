#!/usr/bin/env python3
"""Standalone check (NOT pipeline code), 2026-11-19: model-free depth and width of each node's recorded dip, plane
by plane, on the profiles of a run of clean_cell.py clean --no-clean (cleaning_<S>_node<id>.npz).

Per node and plane k: the dip D_k(v) = B_k - I_k(v) (gl); its deepest sample within |v| <= max(0.5 um, 0.6 x Allen's
diameter) of the line's centre (so that a neighbour's dip is not taken); its full width at half depth, by linear
interpolation between samples, walking outwards from that sample. No model, no fit.
Run:  python3 -I dip_widths_check.py <cleaning dir>
"""
import json
import sys

import numpy as np

SPEC = "529878215"


def main(d):
    rec = json.load(open(d + "/cleaning_%s.json" % SPEC))
    for nid in rec["example_ids"]:
        nid = str(nid)
        info = rec["nodes"][nid]
        with np.load(d + "/cleaning_%s_node%s.npz" % (SPEC, nid), allow_pickle=False) as z:
            P, v, B, ks = z["profiles_after"], z["v"], z["B_gl"], z["ks"]
        d_a = float(info["d_allen_um"])
        win = max(0.5, 0.6 * d_a)
        out = []
        for j, k in enumerate(ks):
            D = B[j] - P[j]
            near = np.flatnonzero(np.abs(v) <= win)
            n0 = int(near[np.argmax(D[near])])
            half = 0.5 * D[n0]

            def cross(step):
                n = n0
                while 0 <= n + step < v.size and D[n + step] > half:
                    n += step
                if not 0 <= n + step < v.size:
                    return np.nan
                a, b = D[n], D[n + step]
                return v[n] + (v[n + step] - v[n]) * (a - half) / (a - b)
            out.append((int(k), D[n0], cross(+1) - cross(-1)))
        narrow = min(out, key=lambda t: t[2])
        print("node %s: Allen %.3f um, noise SD %.2f gl, k* %d; narrowest dip k %d: FWHM %.3f um, depth %.1f gl"
              % (nid, d_a, float(info["sigma_gl"]), int(info["pilot"]["k_star"]), narrow[0], narrow[2], narrow[1]))
        print("   k: depth gl / FWHM um   " + "  ".join("%d: %.0f/%.2f" % t for t in out))


if __name__ == "__main__":
    main(sys.argv[1])
