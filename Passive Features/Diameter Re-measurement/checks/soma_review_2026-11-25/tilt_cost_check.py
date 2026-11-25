"""What enlarging R costs, and what the far-background comparison can see (D-254). Standalone check, 2026-11-25.
Run: OMP_NUM_THREADS=1 python3 tilt_cost_check.py OUT.json [TAUS.npz]

Same simulated soma, blur table, planes and tilted background as tilt_check.py (a 7.0, b 5.6, c 4.5 um; the pilot's
blur table, D-170; B(x, y) = 110 + 0.15 x gl). The model holds the far slope at 0.79 (D-240). Worlds:
  'exact'  far slope 0.79   (the model exact but for the tilt)
  'g095'   far slope 0.95   (the world blurs faster with depth than the model assumes)
  'g060'   far slope 0.60   (slower)
Noise-free data. For each region radius r_R and background model ('const', one column; 'tilt', three columns), the
bias of the c-free fit after one Gauss-Newton step from the truth (linearised; the same method as
checks/soma_review_2026-11-25/sensitivity.py), in x0, y0 (um), a, b, c, S (%).

The far-background comparison (D-254.3): on the planes within 1.2 um of the soma's centre depth z0, the fitted
background at the traced centre (b0_hat_k, linearised at the fitted theta) minus the background measured far from the soma
(a plane fitted by least squares to the data at r >= 12 um, the crop's outer ring and corners, evaluated at the centre).
In the 'exact' world the fitted background is the truth, so the difference is the far measurement's own error (the soma's
halo at r >= 12 um): the floor below which a mismatch means nothing. Custom: the per-plane 3-column projection
(numpy.linalg.lstsq) and the linearised beta_hat (central differences of beta_hat over the same renders as the Jacobian)."""
import json, sys, time, dataclasses
import numpy as np
from multiprocessing import Pool
import common as C
from allen_diameter.analysis.soma_fit import ellipsoid_area

PILOT = dataclasses.replace(C.RCFG, kernel_table_delta_um=(0.0, 0.28, 0.56, 0.84),
                            kernel_table_sigma_um=(0.0800, 0.1536, 0.3924, 0.5620))
TH = np.array([0.6, -0.4, 30.96, 7.0, 5.6, 4.5, 1.13, 0.87 / 9.0])
H = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.01, 0.02 * TH[7]])
B0, GX, G_MODEL = 110.0, 0.15, 0.79
WORLDS = {"exact": 0.79, "g095": 0.95, "g060": 0.60}
RADII = (5.97, 8.0, 10.0, 12.0, 14.0)
NEAR_UM, FAR_UM = 1.2, 12.0
s = C.setup(); left, top, width, height = s["crop"]
zpl = s["ks_all"][s["idx"]] * C.DZ
X = np.broadcast_to(((left + np.arange(width) + 0.5) * C.RES0)[None, :], (height, width))
Y = np.broadcast_to(((top + np.arange(height) + 0.5) * C.RES0)[:, None], (height, width))


def _render(args):
    th, g = args
    return C.SomaRenderer((left, top, width, height), C.RES0, zpl, PILOT, C.Q, C.LIGHT).render(np.asarray(th), g)


def load_or_render(path):
    jobs = [(TH, G_MODEL)] + [(TH + sgn * np.eye(8)[i] * H[i], G_MODEL) for i in range(8) for sgn in (+1, -1)]
    jobs += [(TH, g) for name, g in WORLDS.items() if name != "exact"]
    try:
        z = np.load(path)
        if z["taus"].shape[0] == len(jobs):
            return z["taus"]
    except (OSError, KeyError):
        pass
    t0 = time.time()
    with Pool(2) as p:
        taus = np.stack(p.map(_render, jobs))
    print("rendered %d stacks in %.0f s" % (len(jobs), time.time() - t0), flush=True)
    np.savez_compressed(path, taus=taus)
    return taus


def beta_hat(tau, D, reg, kind):
    """per plane the least-squares background coefficients on R: (K, 1) or (K, 3), and the projected model (K, n)."""
    t, d = tau[:, reg], D[:, reg]
    if kind == "const":
        b = ((d * t).sum(1) / (t * t).sum(1))[:, None]
        return b, b * t
    x, y = X[reg], Y[reg]
    B, out = np.empty((t.shape[0], 3)), np.empty_like(t)
    for k in range(t.shape[0]):
        M = np.stack([t[k], x * t[k], y * t[k]], axis=1)
        B[k] = np.linalg.lstsq(M, d[k], rcond=None)[0]
        out[k] = M @ B[k]
    return B, out


def main(out, cache):
    taus = load_or_render(cache)
    tau_m, tau_pm = taus[0], taus[1:17]
    world_tau = {"exact": tau_m, "g095": taus[17], "g060": taus[18]}
    Bxy = B0 + GX * X
    near = np.abs(zpl - TH[2]) <= NEAR_UM
    far = (X ** 2 + Y ** 2) >= FAR_UM ** 2
    xf, yf = X[far], Y[far]
    Mf = np.stack([np.ones_like(xf), xf, yf], axis=1)
    S0 = ellipsoid_area(*TH[3:6])
    g = np.zeros(8)
    for i in (3, 4, 5):
        e = np.zeros(8); e[i] = 1e-4
        g[i] = (ellipsoid_area(*(TH + e)[3:6]) - ellipsoid_area(*(TH - e)[3:6])) / 2e-4
    res = dict(near_planes=int(near.sum()), far_px=int(far.sum()), worlds={})
    for wname, wg in WORLDS.items():
        D = Bxy[None] * world_tau[wname]
        far_fit = np.array([np.linalg.lstsq(Mf, D[k][far], rcond=None)[0] for k in range(D.shape[0])])
        res["worlds"][wname] = dict(slope=wg, far_b0_err_near_gl=(far_fit[near, 0] - B0).tolist(),
                                    far_b1_err_near=(far_fit[near, 1] - GX).tolist(), by_radius={})
        for r in RADII:
            reg = (X ** 2 + Y ** 2) <= r * r
            row = {}
            for kind in ("const", "tilt"):
                pm = [beta_hat(tau_pm[j], D, reg, kind) for j in range(16)]
                J = np.stack([((pm[2 * i][1] - pm[2 * i + 1][1]) / (2 * H[i])).ravel() for i in range(8)], axis=1)
                b_t, m_t = beta_hat(tau_m, D, reg, kind)
                r0 = (D[:, reg] - m_t).ravel()
                d = np.linalg.solve(J.T @ J, J.T @ r0)
                dB = sum((pm[2 * i][0] - pm[2 * i + 1][0]) / (2 * H[i]) * d[i] for i in range(8))
                b_fit = b_t + dB
                mism = b_fit[near, 0] - far_fit[near, 0]
                row[kind] = dict(bias_x0=float(d[0]), bias_y0=float(d[1]), bias_a_rel=float(d[3] / TH[3]),
                                 bias_b_rel=float(d[4] / TH[4]), bias_c_rel=float(d[5] / TH[5]), bias_S_rel=float(g @ d / S0),
                                 b0_fit_err_near_gl=(b_fit[near, 0] - B0).tolist(), mismatch_near_gl=mism.tolist(),
                                 mismatch_near_max_abs_gl=float(np.abs(mism).max()), mismatch_near_mean_gl=float(mism.mean()))
                if kind == "tilt":
                    row[kind]["b1_fit_err_near"] = (b_fit[near, 1] - GX).tolist()
            res["worlds"][wname]["by_radius"]["%.2f" % r] = row
            c, t = row["const"], row["tilt"]
            print("%-5s R %5.2f: const x0 %+.3f c %+6.1f%% S %+6.1f%% mism %+.2f gl | tilt x0 %+.3f y0 %+.3f a %+5.1f%% b %+5.1f%% "
                  "c %+6.1f%% S %+6.1f%% | mism mean %+.2f max|.| %.2f gl"
                  % (wname, r, c["bias_x0"], 100 * c["bias_c_rel"], 100 * c["bias_S_rel"], c["mismatch_near_mean_gl"],
                     t["bias_x0"], t["bias_y0"], 100 * t["bias_a_rel"], 100 * t["bias_b_rel"], 100 * t["bias_c_rel"],
                     100 * t["bias_S_rel"], t["mismatch_near_mean_gl"], t["mismatch_near_max_abs_gl"]), flush=True)
        print("%-5s far measurement's own error on the %d near planes: b0 %s gl" % (
            wname, near.sum(), np.array2string(far_fit[near, 0] - B0, precision=2)), flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "taus_cost.npz")
