"""Check: how much of a thick node's own dip is still on the measuring line at a distance m x r from the dip's
centre (r = Allen's radius), on the planes k* - 2 .. k* + 2 of Cell 23A's preparation (cell 529878215, the user's
upload of 2026-11-20, 18:09). The dip fraction f(v) = (L(v) - I(v)) / D_c, with L(v) = a + b v the straight
background through the line's samples at |v - c| >= ENDS_UM (c the k* dip's centre: the midpoint of its two
half-depth crossings against the straight background through |v| >= ENDS_UM, Block 40's dip_features) and
D_c = L(c) - I(c) the depth at the centre; f is read on each side, lightly smoothed, at |v - c| = m r, and the reach R_5 is where the
dip first stays below 5 % of its depth going outwards (the larger of the two sides).
Usage: python3 -I stripe_reach.py <repo src dir> <upload dir> [node ids]"""
import csv, os, sys
import numpy as np
from scipy.ndimage import gaussian_filter1d

ENDS_UM = 3.0          # um: the straight background's support (D-203's start_ends_um)
SMOOTH = 1.0           # samples: the Gaussian smoothing of the profile before reading it (as Block 40's)
MULTIPLES = (1.2, 1.5, 2.0, 2.5, 3.0)
LEVEL = 0.05           # the reach: where the dip stays below this fraction of its depth

sys.path.insert(0, sys.argv[1])
from allen_diameter.analysis import clearness as CL
up = sys.argv[2]
want = [s for s in sys.argv[3:]] or ["7", "599", "611", "629"]
rows = {r["node_id"]: r for r in csv.DictReader(open(os.path.join(up, "gallery", "gallery_529878215.csv")))}


def side_profile(f, v, c, sign):
    """f on one side of c, outwards: (distance um, f)."""
    sel = (v - c) * sign >= 0
    d = np.abs(v[sel] - c)
    o = np.argsort(d)
    return d[o], f[sel][o]


print("dip fraction f at |v - c| = m r (r = Allen's radius), each side; R5 = the reach below 5 % of the depth")
print("node  r(um)  c(um)  plane | " + "  ".join("m=%.1f (L / R)" % m for m in MULTIPLES) + " |  R5 (um)  R5/r")
for nid in want:
    r = rows[nid]
    rad = 0.5 * float(r["d_allen_um"]); kst = int(r["k_star"])
    with np.load(os.path.join(up, "cleaning_529878215_node%s.npz" % nid)) as z:
        ks, P, v = z["ks"], np.array(z["profiles_before"], float), np.array(z["v"], float)
    i0 = int(np.flatnonzero(ks == kst)[0])
    m0 = np.abs(v) >= ENDS_UM
    a0, b0 = np.linalg.lstsq(np.column_stack([np.ones(m0.sum()), v[m0]]), P[i0][m0], rcond=None)[0]
    c = float(CL.dip_features(CL.smooth_dip(a0 + b0 * v - P[i0], SMOOTH), v)["centre_um"])   # the k* dip's centre
    for i, k in enumerate(ks):
        I = gaussian_filter1d(P[i], SMOOTH)
        m_far = np.abs(v - c) >= ENDS_UM
        a, b = np.linalg.lstsq(np.column_stack([np.ones(m_far.sum()), v[m_far]]), P[i][m_far], rcond=None)[0]
        L = a + b * v
        D = L - I
        Dc = float(np.interp(c, v, D))
        f = D / Dc
        cells, reach = [], 0.0
        for sign in (-1, +1):
            d, fs = side_profile(f, v, c, sign)
            above = np.flatnonzero(fs >= LEVEL)
            reach = max(reach, float(d[above[-1] + 1]) if above.size and above[-1] + 1 < d.size else float(d[-1]))
        for m in MULTIPLES:
            pair = []
            for sign in (-1, +1):
                d, fs = side_profile(f, v, c, sign)
                pair.append(float(np.interp(m * rad, d, fs)))
            cells.append("%5.2f / %5.2f" % tuple(pair))
        print("%-5s %.3f  %+.2f  k*%+d  | %s |  %5.2f    %4.1f" % (nid, rad, c, int(k) - kst, "  ".join(cells), reach, reach / rad))
