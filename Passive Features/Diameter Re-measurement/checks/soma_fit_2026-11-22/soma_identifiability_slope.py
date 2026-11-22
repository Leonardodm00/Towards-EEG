"""Joint fit of the kernel's far slope with the soma (2026-11-22; companion of soma_identifiability.py): the Fisher
information with the continuation slope gamma as a ninth parameter, at the configured gamma (0.79 um/um). If the
height c and gamma separate, SE(c) stays small and |corr(c, gamma)| < 1 clearly; if not, the height needs gamma from
an independent calibration. Pure ASCII.

    python3 soma_identifiability_slope.py <workstream folder> <cleaning_config.json>
"""
import math
import sys
import time

import numpy as np

sys.path.insert(0, ".")
import soma_identifiability as SI                           # noqa: E402


def analyse(cc, cfg_path, c):
    rcfg = SI.kernel_cfg(cfg_path)
    g0 = rcfg.kernel_continuation_slope
    dg = 0.01
    rcfg_g = SI.kernel_cfg(cfg_path, (g0 + dg) / g0)
    box = SI.Box(c)
    mu = -math.log(c["tau_centre"]) / (2.0 * cc)
    th = np.array([0.0, 0.0, 0.0, c["a"], c["b"], cc, math.radians(c["phi_deg"]), mu])
    zk = np.arange(-c["plane_half_um"], c["plane_half_um"] + 1e-9, c["dz"] * c["plane_step"])
    region = (box.X ** 2 + box.Y ** 2) <= c["fit_radius_um"] ** 2
    t0 = time.time()
    tau = SI.render(box, th, zk, rcfg, c)
    steps = np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.004, mu * 1e-3])
    D = []
    for i in range(8):
        t2 = th.copy()
        t2[i] += steps[i]
        D.append((SI.render(box, t2, zk, rcfg, c) - tau) / steps[i] * c["B"])
    D.append((SI.render(box, th, zk, rcfg_g, c) - tau) / dg * c["B"])
    D = np.array(D)
    K, n9 = len(zk), 9
    A = np.zeros((n9 + K, n9 + K))
    for k in range(K):
        Dk = D[:, k][:, region].T
        tk = tau[k][region]
        A[:n9, :n9] += Dk.T @ Dk
        A[:n9, n9 + k] = Dk.T @ tk
        A[n9 + k, :n9] = A[:n9, n9 + k]
        A[n9 + k, n9 + k] = tk @ tk
    cov = c["noise"] ** 2 * np.linalg.inv(A)
    se = np.sqrt(np.diag(cov)[:n9])
    corr = cov[:n9, :n9] / np.outer(se, se)
    return se, corr, time.time() - t0


if __name__ == "__main__":
    c = dict(SI.CFG)
    print("joint fit with the far slope gamma (configured 0.79 um/um) free:")
    print("  c true | SE(c) um  SE(gamma)  corr(c,gamma) corr(c,mu) corr(gamma,mu)")
    for cc in (3.5, 5.0, 8.0):
        se, corr, sec = analyse(cc, sys.argv[2], c)
        print("  %4.1f   | %.4f    %.4f     %+.3f        %+.3f      %+.3f   (%.0f s)"
              % (cc, se[5], se[8], corr[5, 8], corr[5, 7], corr[8, 7], sec), flush=True)
