"""Would a background measured around the soma, then held fixed, bias the soma fit? Standalone check (not pipeline
code), 2026-11-25. Run: OMP_NUM_THREADS=1 python3 background_check.py
Simulated soma a 7.0, b 5.6, c 4.5 um, 2 mu c 0.87, centre (0.6, -0.4, 30.96) um; the pilot's blur table (D-170), far
slope 0.79 in the world AND in the model (the model is otherwise true); background 110 gl, no noise; the D-241 region
(5.97 um disc) and Block 46's 35 planes. Per plane, B_k is "measured" as the median of the data over an annulus around
the traced centre (8-14 um, inside the crop), as a background measured beside a structure would be, then held; the
one-Gauss-Newton-step bias of the eight parameters and of S follows (linearised, as sensitivity.py)."""
import dataclasses
import numpy as np
import common as C
from allen_diameter.analysis.soma_fit import ellipsoid_area

PILOT = dataclasses.replace(C.RCFG, kernel_table_delta_um=(0.0, 0.28, 0.56, 0.84),
                            kernel_table_sigma_um=(0.0800, 0.1536, 0.3924, 0.5620))
TH = np.array([0.6, -0.4, 30.96, 7.0, 5.6, 4.5, 1.13, 0.87 / 9.0])
H = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.01, 0.02 * TH[7]])
s = C.setup(); left, top, width, height = s["crop"]; reg = s["region"]
zpl = s["ks_all"][s["idx"]] * C.DZ
render = C.SomaRenderer((left, top, width, height), C.RES0, zpl, PILOT, C.Q, C.LIGHT).render
X = (left + np.arange(width) + 0.5) * C.RES0; Y = (top + np.arange(height) + 0.5) * C.RES0
rr = np.hypot(X[None, :], Y[:, None])
ann = (rr >= 8.0) & (rr <= 14.0)
tau0 = render(TH, 0.79); D = 110.0 * tau0
Bm = np.median(D[:, ann], axis=1)
print("annulus 8-14 um: %d px; measured B_k - 110 gl by plane offset z_k - z0:" % ann.sum())
for k in range(0, zpl.size, 4):
    print("   %+6.2f um: %+6.2f gl" % (zpl[k] - TH[2], Bm[k] - 110.0))
model = lambda tau: (Bm[:, None] * tau[:, reg])   # noqa: E731
J = np.stack([((model(render(TH + e, 0.79)) - model(render(TH - e, 0.79))) / (2 * h)).ravel()
              for e, h in ((np.eye(8)[i] * H[i], H[i]) for i in range(8))], axis=1)
r0 = (D[:, reg] - model(tau0)).ravel()
d = np.linalg.solve(J.T @ J, J.T @ r0)
th1 = TH + d
S0, S1 = ellipsoid_area(*TH[3:6]), ellipsoid_area(*th1[3:6])
print("held annulus B: bias a %+.1f %%, b %+.1f %%, c %+.1f %%, z0 %+.2f um, mu*c %+.1f %%, S %+.1f %% (S %.0f vs %.0f um^2)"
      % (100 * d[3] / TH[3], 100 * d[4] / TH[4], 100 * d[5] / TH[5], d[2], 100 * ((th1[5] * th1[7]) / (TH[5] * TH[7]) - 1),
         100 * (S1 / S0 - 1), S1, S0))
print("(B_k re-solved on R at every step, as now: no bias in this model-true world)")
