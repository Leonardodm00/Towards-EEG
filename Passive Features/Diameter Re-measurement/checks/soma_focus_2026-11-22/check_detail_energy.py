"""Check (2026-11-22, D-211): can a thin dendrite in focus inside the soma's disk outweigh the soma's own focus in
Block 44's measure, the variance over the disk of the Laplacian of Gaussian (sigma 1 px)?

Synthetic brightfield stack, Beer-Lambert absorbance blurred per slab by a Gaussian whose SD grows with defocus:
    A_k(x, y) = sum_s mu * dz * chi_s(x, y)  (*)  G[sigma(|z_k - z_s|)],   I_k = I0 exp(-A_k)
chi_s the cross-section of the absorbers in slab s (a sphere of radius R, the soma, centred at depth z_soma; a
horizontal cylinder of diameter d crossing the soma's disk at depth z_dend), sigma(delta) = sqrt(s0^2 + (g delta)^2).
The measure is computed on every plane; the claim fails if its maximum sits at the soma's centre plane rather than at
the dendrite's. Standalone check, not pipeline code. Pure ASCII.

    python3 check_detail_energy.py
"""
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_laplace

CFG = dict(res_um=0.1144, half_um=12.0, dz_um=0.28, n_planes=61,            # 61 planes, +-8.4 um about the soma
           R_um=6.0, z_soma_um=0.0, d_um=0.8, z_dend_um=-6.7, dend_y_um=-1.0,  # the dendrite 6.7 um below the soma
           mu_per_um=0.258, I0=200.0, s0_um=0.1, g=0.5, noise_gl=1.0,          # mu: the pilot's median (Cell 16)
           disk_margin_um=1.0, log_sigma_px=1.0, seed=20261122)


def stack(c, with_dendrite=True):
    p, n = c["res_um"], int(round(2 * c["half_um"] / c["res_um"])) + 1
    x = (np.arange(n) - n // 2) * p
    X, Y = np.meshgrid(x, x)
    zs = (np.arange(c["n_planes"]) - c["n_planes"] // 2) * c["dz_um"]
    rng = np.random.default_rng(c["seed"])
    out = []
    for zk in zs:
        A = np.zeros_like(X)
        for z in zs:                                   # slabs on the plane grid
            chi = np.zeros_like(X)
            h2 = c["R_um"] ** 2 - (z - c["z_soma_um"]) ** 2
            if h2 > 0:
                chi += (X ** 2 + Y ** 2 <= h2)
            hd = (c["d_um"] / 2) ** 2 - (z - c["z_dend_um"]) ** 2
            if with_dendrite and hd > 0:
                chi += (np.abs(Y - c["dend_y_um"]) <= np.sqrt(hd)) & (X <= 0.0)   # runs in from the left, ends at x = 0
            if chi.any():
                s = np.sqrt(c["s0_um"] ** 2 + (c["g"] * (zk - z)) ** 2) / p
                A += gaussian_filter(c["mu_per_um"] * c["dz_um"] * chi, s)
        out.append(c["I0"] * np.exp(-A) + rng.normal(0.0, c["noise_gl"], X.shape))
    return np.array(out), X, Y, zs


def energy(B, X, Y, c):
    m = X ** 2 + Y ** 2 <= (c["R_um"] + c["disk_margin_um"]) ** 2
    return np.array([np.var(gaussian_laplace(b, c["log_sigma_px"])[m]) for b in B])


if __name__ == "__main__":
    import sys
    c = dict(CFG)
    for kv in sys.argv[1:]:                       # overrides, e.g. d_um=1.6
        k, v = kv.split("=")
        c[k] = type(CFG[k])(v)
    print("config: d_um %.2f, R_um %.1f, mu %.3f /um, z_dend %+.2f um" % (c["d_um"], c["R_um"], c["mu_per_um"],
                                                                         c["z_dend_um"]))
    for with_d in (False, True):
        B, X, Y, zs = stack(c, with_d)
        E = energy(B, X, Y, c)
        j = int(np.argmax(E))
        jd = int(np.argmin(np.abs(zs - c["z_dend_um"])))
        print("dendrite %-5s: max at z = %+.2f um (soma centre 0.00, dendrite %+.2f); E at the max %.3g, at the soma "
              "centre %.3g, at the dendrite's plane %.3g" % (with_d, zs[j], c["z_dend_um"], E[j], E[len(zs) // 2], E[jd]))
