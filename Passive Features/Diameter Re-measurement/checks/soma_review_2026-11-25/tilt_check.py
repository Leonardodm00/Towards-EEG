"""Tilted background solved inside the soma fit (D-254): what it costs and what it fixes, by region radius.
Standalone check (not pipeline code), 2026-11-25. Run: OMP_NUM_THREADS=1 python3 tilt_check.py OUT.json

World: simulated soma a 7.0, b 5.6, c 4.5 um, 2 mu c 0.87, centre (0.6, -0.4, 30.96) um, phi 1.13; the pilot's blur
table (D-170), far slope 0.79 in world and model (the model otherwise exact); 35 planes; background tilted along x,
B(x, y) = 110 + 0.15 x gl (x in um from the traced centre: +-2.1 gl across +-14 um); noise SD 3 gl (for the SEs only;
the data used for the biases are noise-free). Region: a disc of radius r_R about the traced centre (no branches).
Two background models, each solved exactly per plane at every trial theta (variable projection):
  'const'  B_k                         (one column: tau_k)            -- the fit as it stands
  'tilt'   b_0k + b_1k x + b_2k y      (three columns: tau_k, x tau_k, y tau_k) -- D-254.1
For each, at the truth: the noise SEs of x0, y0, c and S from the Jacobian of the projected model (central
differences), and the bias after one Gauss-Newton step in the tilted world. Custom: the 3-column projection
(numpy.linalg.lstsq per plane); nothing in the pipeline solves it yet."""
import json, sys, dataclasses
import numpy as np
import common as C
from allen_diameter.analysis.soma_fit import ellipsoid_area

PILOT = dataclasses.replace(C.RCFG, kernel_table_delta_um=(0.0, 0.28, 0.56, 0.84),
                            kernel_table_sigma_um=(0.0800, 0.1536, 0.3924, 0.5620))
TH = np.array([0.6, -0.4, 30.96, 7.0, 5.6, 4.5, 1.13, 0.87 / 9.0])
H = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.01, 0.02 * TH[7]])
SIG, B0, GX = 3.0, 110.0, 0.15
RADII = (5.97, 8.0, 10.0, 12.0, 14.0)
s = C.setup(); left, top, width, height = s["crop"]
zpl = s["ks_all"][s["idx"]] * C.DZ
render = C.SomaRenderer((left, top, width, height), C.RES0, zpl, PILOT, C.Q, C.LIGHT).render
X = np.broadcast_to(((left + np.arange(width) + 0.5) * C.RES0)[None, :], (height, width))
Y = np.broadcast_to(((top + np.arange(height) + 0.5) * C.RES0)[:, None], (height, width))
taus = [render(TH, 0.79)] + [render(TH + sgn * np.eye(8)[i] * H[i], 0.79) for i in range(8) for sgn in (+1, -1)]
Bxy = B0 + GX * X
D_tilt = Bxy[None] * taus[0]


def project(tau, D, reg, kind):
    """the projected model on R, (K, n): per plane the least-squares background times tau."""
    t, d = tau[:, reg], D[:, reg]
    if kind == "const":
        b = (d * t).sum(1) / (t * t).sum(1)
        return b[:, None] * t
    x, y = X[reg], Y[reg]
    out = np.empty_like(t)
    for k in range(t.shape[0]):
        M = np.stack([t[k], x * t[k], y * t[k]], axis=1)
        beta = np.linalg.lstsq(M, d[k], rcond=None)[0]
        out[k] = M @ beta
    return out


res = {}
for r in RADII:
    reg = (X ** 2 + Y ** 2) <= r * r
    row = dict(n_px=int(reg.sum()))
    for kind in ("const", "tilt"):
        J = np.stack([((project(taus[1 + 2 * i], D_tilt, reg, kind) - project(taus[2 + 2 * i], D_tilt, reg, kind))
                       / (2 * H[i])).ravel() for i in range(8)], axis=1)
        cov = SIG ** 2 * np.linalg.inv(J.T @ J)
        se = np.sqrt(np.diag(cov))
        g = np.zeros(8)
        for i in (3, 4, 5):
            e = np.zeros(8); e[i] = 1e-4
            g[i] = (ellipsoid_area(*(TH + e)[3:6]) - ellipsoid_area(*(TH - e)[3:6])) / 2e-4
        S0 = ellipsoid_area(*TH[3:6])
        r0 = (D_tilt[:, reg] - project(taus[0], D_tilt, reg, kind)).ravel()
        d = np.linalg.solve(J.T @ J, J.T @ r0)
        row[kind] = dict(se_x0=float(se[0]), se_y0=float(se[1]), se_c=float(se[5]), se_S_rel=float(np.sqrt(g @ cov @ g) / S0),
                         bias_x0=float(d[0]), bias_y0=float(d[1]), bias_c_rel=float(d[5] / TH[5]), bias_S_rel=float(g @ d / S0),
                         rms_resid_at_truth=float(np.sqrt(np.mean(r0 ** 2))))
    res["%.2f" % r] = row
    c, t = row["const"], row["tilt"]
    print("R %5.2f um (%5d px): const: bias x0 %+.3f um, c %+.2f %%, S %+.2f %% | SE x0 %.4f c %.4f | tilt: bias x0 %+.4f c %+.3f %% "
          "S %+.3f %% | SE x0 %.4f c %.4f S %.3f %%"
          % (r, row["n_px"], c["bias_x0"], 100 * c["bias_c_rel"], 100 * c["bias_S_rel"], c["se_x0"], c["se_c"],
             t["bias_x0"], 100 * t["bias_c_rel"], 100 * t["bias_S_rel"], t["se_x0"], t["se_c"], 100 * t["se_S_rel"]), flush=True)
json.dump(dict(world=dict(B0=B0, gx_gl_per_um=GX, sigma_gl=SIG), truth=TH.tolist(), results=res), open(sys.argv[1], "w"), indent=1)
