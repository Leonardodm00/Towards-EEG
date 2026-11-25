"""Can the soma fit determine the in-focus blur sigma_r(0)? Noise SE of sigma_r(0) and its correlations if it were a
ninth free parameter, from the residuals' Jacobian at the truth. Standalone check (not pipeline code), 2026-11-25.
Run: OMP_NUM_THREADS=1 python3 sigma0_identifiability.py
Simulated soma a 7.0, b 5.6, c 4.5 um (2 mu c 0.87), the pilot's table (D-170) anchored at sigma_r(0) = 0.080 or
0.258 um (kernel.with_kernel keeps the growth), far slope 0.79, the D-241 region (5.97 um disc), B_k re-solved per
plane (variable projection), noise SD 3 gl on 110 gl. Central differences: 0.005-0.05 um, 0.01 rad, 2 % of mu,
0.01 um of sigma_r(0)."""
import dataclasses
import numpy as np
import common as C
from allen_diameter.model.kernel import with_kernel
from allen_diameter.analysis.soma_fit import _vp_columns

PILOT = dataclasses.replace(C.RCFG, kernel_table_delta_um=(0.0, 0.28, 0.56, 0.84),
                            kernel_table_sigma_um=(0.0800, 0.1536, 0.3924, 0.5620))
TH = np.array([0.6, -0.4, 30.96, 7.0, 5.6, 4.5, 1.13, 0.87 / 9.0])
H = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.01, 0.02 * TH[7]])
NAMES = ["x0", "y0", "z0", "a", "b", "c", "phi", "mu", "sigma0"]
s = C.setup(); left, top, width, height = s["crop"]; reg = s["region"]
zpl = s["ks_all"][s["idx"]] * C.DZ
rend = lambda s0: C.SomaRenderer((left, top, width, height), C.RES0, zpl, with_kernel(PILOT, s0), C.Q, C.LIGHT).render  # noqa: E731
for s0 in (0.080, 0.258):
    r0, rp, rm = rend(s0), rend(s0 + 0.01), rend(s0 - 0.01)
    D = (110.0 * r0(TH, 0.79))[:, reg]
    model = lambda tau: _vp_columns(D, tau[:, reg])[1].ravel() * -1.0   # -(resid) = B tau - D; its derivative is the model's  # noqa: E731
    cols = []
    for i in range(8):
        e = np.zeros(8); e[i] = H[i]
        cols.append((model(r0(TH + e, 0.79)) - model(r0(TH - e, 0.79))) / (2 * H[i]))
    cols.append((model(rp(TH, 0.79)) - model(rm(TH, 0.79))) / 0.02)
    J = np.stack(cols, 1)
    cov = 9.0 * np.linalg.inv(J.T @ J); se = np.sqrt(np.diag(cov)); cr = cov / np.outer(se, se)
    print("sigma_r(0) = %.3f um: SE(sigma_r(0)) %.4f um; its largest |correlations|: %s; SE c %.4f um (with it free) "
          "vs %.4f um (held); column norm ratio sigma0/c %.3g"
          % (s0, se[8], ", ".join("%s %.2f" % (NAMES[j], cr[8, j]) for j in np.argsort(-np.abs(cr[8, :8]))[:3]),
             se[5], np.sqrt(9.0 * np.linalg.inv(J[:, :8].T @ J[:, :8])[5, 5]),
             np.linalg.norm(J[:, 8]) / np.linalg.norm(J[:, 5])), flush=True)
