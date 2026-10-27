"""Node-plane and off-focus profiles of an infinite flat tube along x, by the
hybrid (Block 20) with the production Gaussian kernel, the partition, and the
ray world, computed in 1-D (the tube is invariant along x, so the 2-D Gaussian
blur reduces to a 1-D one across it). Not pipeline code: a survey of where the
hybrid's transmittance goes below 0."""
import sys, math
sys.path.insert(0, sys.argv[1])
import numpy as np
from scipy.ndimage import gaussian_filter1d
from allen_diameter.config import default_config
from allen_diameter.model import geometry as G, ray_world as RW, hybrid as H, kernel as K

rc = default_config().renderer
S, W = RW.directions(rc.ray_n_rho, rc.ray_n_psi, rc.ray_na, rc.ray_n_oil)
PX = 0.1144
h = PX / 8
dz = rc.dzeta_um


def profiles(d, mu, zks, half=None):
    r = 0.5 * d
    tube = G.Tube((0.0, 0.0, 0.0), r, 0.0, 0.0, 1.0, None)
    half = half if half is not None else r + 4.0
    m = int(round(half / h))
    y = h * np.arange(-m, m + 1)
    zlo, zhi, ins = G.column_interval(np.zeros_like(y), y, tube)
    zeta = G.slab_centres(float(zlo[ins].min()), float(zhi[ins].max()), dz)
    out = {}
    hyb = np.zeros((len(zks), y.size)); par = np.zeros_like(hyb)
    for zj in zeta:
        a = G.slab_absorbance(zlo, zhi, zj, dz, mu)
        if not (a > 0).any():
            continue
        dA_h = H.slab_absorbed(np.zeros_like(y), y, zlo, zhi, zj, dz, mu, tube, S, W, +1)
        dA_p = G.transmitted_before(zlo, zhi, zj, dz, mu, +1) * (-np.expm1(-a))
        for i, zk in enumerate(zks):
            s = float(K.sigma_r(np.array([zj - zk]), rc)[0])
            hyb[i] += gaussian_filter1d(dA_h, s / h, mode="constant", cval=0.0, truncate=8.0)
            par[i] += gaussian_filter1d(dA_p, s / h, mode="constant", cval=0.0, truncate=8.0)
    out["hybrid"] = 1 - hyb
    out["partition"] = 1 - par
    ray = np.zeros_like(hyb)
    for i, zk in enumerate(zks):
        pts = np.column_stack([np.zeros_like(y), y, np.full(y.size, zk)])
        ray[i] = RW.ray_transmittance(pts, tube, mu, S, W)
    out["ray_world"] = ray
    out["ray_world_post"] = 1 - np.array([gaussian_filter1d(1 - ray[i], rc.ray_post_sigma_um / h, mode="constant", truncate=8.0) for i in range(len(zks))])
    return y, out


if __name__ == "__main__":
    zks = np.array([0.0, 0.28, 0.56, -0.28, -0.56])
    print("min over the profile of tau = I/B; infinite flat tube along x; light +1; production Gaussian kernel; 16 x 32 directions")
    print("  d    mu d  plane   hybrid  partition  ray(no post)  ray(post 0.08)")
    for d in (0.5, 1.0, 2.0, 3.0):
        for mud in (0.3, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0):
            y, o = profiles(d, mud / d, zks)
            for i, zk in enumerate(zks):
                print("%4.1f %6.2f %+5.2f  %7.3f  %8.3f  %10.3f  %10.3f" % (d, mud, zk, o["hybrid"][i].min(), o["partition"][i].min(),
                                                                      o["ray_world"][i].min(), o["ray_world_post"][i].min()), flush=True)
