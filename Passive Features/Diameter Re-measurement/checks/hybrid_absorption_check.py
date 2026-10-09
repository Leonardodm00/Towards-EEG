"""Hybrid renderer vs ray world: does booking absorbed light with the history along the
rays, then spreading it with a kernel, cure the partition's dark failure? (2026-10-09)

NOT pipeline code. Theory chat check for the user's proposal (theory handoff, Open
work 1): compute absorption with the ray world's geometry (NA 1.4, n_oil 1.515) and
render each image plane with a kernel. Because a kernel does not know a ray's
direction, the absorbed light at a slice point is averaged over the directions
arriving there before it is spread. Tested where the answer is known exactly: in
the ray world, with the kernel of the cone's own rays weighted W* (the faint-limit
kernel, as in optics_points_check.py `matched`). Flat d = 1 um tube, centre point of
the node plane. Reference output in hybrid_absorption_check.out. Run from inside
checks/ (imports optics_points_check):
    python hybrid_absorption_check.py      # about a minute
"""
import numpy as np
from optics_points_check import directions, chord_interval, ray_models


r = 0.5
S_line, W_line = directions(16, 32)            # lines through the image point
Wst = W_line / S_line[:, 2]; Wst /= Wst.sum()  # W* = faint-limit kernel weights
S_abs, W_abs = directions(16, 32)               # directions arriving at a slice point

def A_density(v, z, mu):
    """direction-averaged absorption density per unit depth at points (v, z) inside the
    flat tube: < (mu / cos) exp(-mu * back-traced path inside the tube) >_W."""
    out = np.zeros(v.size)
    inside = v**2 + z**2 < r**2
    vv, zz = v[inside], z[inside]
    sv, sz = S_abs[:, 1], S_abs[:, 2]
    p2 = np.sqrt(sv**2 + sz**2)                  # length of the (v, z) projection of s
    dv, dz = -sv / p2, -sz / p2                  # unit back-trace direction in (v, z)
    acc = np.zeros(vv.size)
    for a in range(0, vv.size, 4000):
        pv, pz = vv[a:a + 4000, None], zz[a:a + 4000, None]
        pd = pv * dv[None, :] + pz * dz[None, :]
        c = pv**2 + pz**2 - r**2
        t2 = -pd + np.sqrt(np.clip(pd**2 - c, 0, None))   # 2-D distance back to the circle
        ell = t2 / p2[None, :]                            # 3-D path length inside the tube
        acc[a:a + 4000] = (mu / sz[None, :] * np.exp(-mu * ell)) @ W_abs
    out[inside] = acc
    return out

def hybrid_centre_dip(mu, n_t=600):
    P = np.zeros(3)
    t1, t2, good = chord_interval(P, S_line, r, 0.0, 6.0)
    x, wx = np.polynomial.legendre.leggauss(n_t)
    dip = 0.0
    for k in np.flatnonzero(good):
        tt = (t1[k] + t2[k]) / 2 + (t2[k] - t1[k]) / 2 * x
        v = tt * S_line[k, 1]; z = tt * S_line[k, 2]
        A = A_density(v, z, mu)
        # booked absorption along the line: integral of A dz = integral of A s_z dt
        dip += Wst[k] * np.sum(wx * A * S_line[k, 2]) * (t2[k] - t1[k]) / 2
    return dip

print("flat d = 1 um, node plane, centre point; dip = 1 - I/B")
print("mu*d   ray world G (W)   hybrid H (W*)   partition P* (W*, vert. path)   true history Gv* (W*, vert. path)")
for mud in (0.05, 0.5, 1.5, 3.0, 10.0, 50.0):
    mu = mud / 1.0
    _, _, _, G = ray_models(0.0, 0.0, r, 0.0, mu, S_line, W_line)
    _, P, Gv, _ = ray_models(0.0, 0.0, r, 0.0, mu, S_line, Wst, n_t=200)
    H = hybrid_centre_dip(mu)
    print(f"{mud:5.2f}   {1 - G:.3f}             {H:.3f}           {1 - P:.3f}                          {1 - Gv:.3f}", flush=True)
