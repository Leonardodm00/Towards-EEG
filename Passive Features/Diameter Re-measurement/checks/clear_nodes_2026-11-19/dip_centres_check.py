"""Standalone check (NOT pipeline code), 2026-11-19: the lateral position of each node's dip, plane by plane (the
midpoint of the half-depth crossings, as dip_widths_check.py), and its least-squares slope against the plane's depth.
The one-tube model keeps the axis at one lateral position in every plane.
Run:  python3 -I dip_centres_check.py <cleaning dir>
"""
import json, sys
import numpy as np
d = sys.argv[1]
rec = json.load(open(d + "/cleaning_529878215.json"))
for nid in ("5353", "2669", "1013", "2620", "639"):
    info = rec["nodes"][nid]
    z = np.load(d + "/cleaning_529878215_node%s.npz" % nid, allow_pickle=False)
    P, v, B, ks, pz = z["profiles_after"], z["v"], z["B_gl"], z["ks"], z["planes_z"]
    win = max(0.5, 0.6 * float(info["d_allen_um"]))
    cs = []
    for j in range(len(ks)):
        D = B[j] - P[j]
        near = np.flatnonzero(np.abs(v) <= win); n0 = int(near[np.argmax(D[near])]); half = 0.5 * D[n0]
        def cross(step):
            n = n0
            while 0 <= n + step < v.size and D[n + step] > half:
                n += step
            if not 0 <= n + step < v.size:
                return np.nan
            a, b = D[n], D[n + step]
            return v[n] + (v[n + step] - v[n]) * (a - half) / (a - b)
        cs.append(0.5 * (cross(-1) + cross(+1)))
    cs = np.array(cs)
    ok = np.isfinite(cs)
    slope = np.polyfit(pz[ok], cs[ok], 1)[0] if ok.sum() >= 3 else np.nan
    print("node %s: dip centre (um) per plane %s; slope %.2f um per um of depth" % (nid, " ".join("%+.2f" % c for c in cs), slope))
