"""Like-for-like (renderers doc s.3.7, s.3.9 test 1-2 at profile level, no fit):
the hybrid and the partition blurred by the cone's own kernel weighted W*
(K^cone*), against the exact ray world without post blur, across an infinite
flat tube along x (1-D: (f * K)(y) = <f(y + delta t_y)>_W*). With that kernel
the hybrid differs from the ray world only by its direction average. Not
pipeline code."""
import sys, math
sys.path.insert(0, sys.argv[1])
import numpy as np
from allen_diameter.config import default_config
from allen_diameter.model import geometry as G, ray_world as RW, hybrid as H

rc = default_config().renderer
S, W = RW.directions(rc.ray_n_rho, rc.ray_n_psi, rc.ray_na, rc.ray_n_oil)
Wst = W / S[:, 2]; Wst = Wst / Wst.sum()
ty = S[:, 1] / S[:, 2]
PX = 0.1144
h = PX / 8
dz = rc.dzeta_um


def profiles(d, mu, zks):
    r = 0.5 * d
    tube = G.Tube((0.0, 0.0, 0.0), r, 0.0, 0.0, 1.0, None)
    half = r + 2.0 + 2.5 * (r + 0.6)
    m = int(round(half / h))
    y = h * np.arange(-m, m + 1)
    zlo, zhi, ins = G.column_interval(np.zeros_like(y), y, tube)
    zeta = G.slab_centres(float(zlo[ins].min()), float(zhi[ins].max()), dz)
    hyb = np.zeros((len(zks), y.size)); par = np.zeros_like(hyb)
    for zj in zeta:
        a = G.slab_absorbance(zlo, zhi, zj, dz, mu)
        if not (a > 0).any():
            continue
        dA_h = H.slab_absorbed(np.zeros_like(y), y, zlo, zhi, zj, dz, mu, tube, S, W, +1)
        dA_p = G.transmitted_before(zlo, zhi, zj, dz, mu, +1) * (-np.expm1(-a))
        for i, zk in enumerate(zks):
            sh = (zj - zk) * ty                       # cone displacements, one per direction
            yy = y[None, :] + sh[:, None]
            hyb[i] += Wst @ np.interp(yy, y, dA_h, left=0.0, right=0.0)
            par[i] += Wst @ np.interp(yy, y, dA_p, left=0.0, right=0.0)
    ray = np.zeros_like(hyb)
    for i, zk in enumerate(zks):
        pts = np.column_stack([np.zeros_like(y), y, np.full(y.size, zk)])
        ray[i] = RW.ray_transmittance(pts, tube, mu, S, W)
    return y, 1 - hyb, 1 - par, ray


if __name__ == "__main__":
    zks = np.array([0.0, 0.56, -0.56])
    print("cone kernel K^cone* (W*-weighted), same mu, no post blur; infinite flat tube along x; light +1; 16 x 32 directions")
    print("  d    mu d  plane | min tau: hybrid  partition  ray | max |tau - ray|: hybrid  partition | dip area / ray's: hybrid partition")
    for d in (0.5, 1.0, 2.0):
        for mud in (0.3, 1.0, 1.5, 3.0, 10.0):
            y, hy, pa, ra = profiles(d, mud / d, zks)
            for i, zk in enumerate(zks):
                ar = np.sum(1 - ra[i])
                print("%4.1f %6.2f %+5.2f | %7.3f %8.3f %7.3f | %7.3f %8.3f | %7.4f %7.4f" % (
                    d, mud, zk, hy[i].min(), pa[i].min(), ra[i].min(), np.abs(hy[i] - ra[i]).max(), np.abs(pa[i] - ra[i]).max(),
                    np.sum(1 - hy[i]) / ar, np.sum(1 - pa[i]) / ar), flush=True)
