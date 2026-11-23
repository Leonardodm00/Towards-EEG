"""The linearized d bias of fitting the SGT-across / Gaussian-along kernel to data made with the isotropic SGT PSF.

One Gauss-Newton step from the truth: dp = (J^T J)^-1 J^T r, r = (tau_iso - tau_sep(truth)) / noise, J the
separable model's Jacobian (central differences) over (d, mu, v0, z_ax, log10 beta, xi), the backgrounds known.
Usage: python3 bias.py <src dir> <out.json>
"""
import json
import math
import sys
import time

import numpy as np

sys.path.insert(0, sys.argv[1])
from allen_diameter.model import blur_kernels as bk  # noqa: E402

import fft_line as F  # noqa: E402
import run_compare as R  # noqa: E402

CUT, PAD = 1.3, 8.0


def model(cfg, d, mu, v0, z_ax, beta, xi, phi, theta, iso=False):
    tube, mu = R.setup(d, phi, theta, 8.0, mu=mu, v0=v0, z_ax=z_ax)
    ks, o, yh, eu, v, n_avg, step = R.common_args(cfg, theta)
    table = F.unit_cf_table(lambda xx: bk.sgt_unit_pdf(xx, beta, xi, 1.0))
    W = bk.sgt_unit_fwhm(beta, xi, 1.0)
    tr = F.iso_transfer(table, W) if iso else F.sep_transfer(table, W, yh, eu)
    return F.render(tube, mu, ks, o, yh, eu, v, n_avg, step, cfg, tr, PAD, CUT)[0]


def main(out):
    cfg = R.config()
    rows = []
    for d, phi, theta, shape in ((0.4, 0.3, 0.52, "b13"), (0.4, 0.6, 0.52, "heavy"), (0.8, 0.3, 0.52, "b13"),
                                 (0.8, 0.6, 0.52, "heavy")):
        t0 = time.perf_counter()
        beta, xi, _ = R.SHAPES[shape]
        p0 = np.array([d, 1.0, 0.04, 0.1, math.log10(beta), xi])
        steps = np.array([0.01 * d, 0.01, 0.005, 0.01, 0.005, 0.02])

        def tau(p):
            return model(cfg, p[0], p[1], p[2], p[3], 10.0 ** p[4], max(p[5], 0.0), phi, theta)
        base = tau(p0)
        data = model(cfg, d, 1.0, 0.04, 0.1, beta, xi, phi, theta, iso=True)
        cols = []
        for i in range(p0.size):
            e = np.zeros(p0.size)
            e[i] = steps[i]
            if i == 5 and xi == 0.0:                       # xi = 0 is the bound: one-sided
                cols.append(((tau(p0 + e) - base) / steps[i]).ravel())
            else:
                cols.append(((tau(p0 + e) - tau(p0 - e)) / (2 * steps[i])).ravel())
        J = np.column_stack(cols) / R.NOISE_TAU
        r = (data - base).ravel() / R.NOISE_TAU
        JtJ = J.T @ J
        cov = np.linalg.inv(JtJ)
        dp = cov @ (J.T @ r)
        se = np.sqrt(np.diag(cov))
        # with the shape held at the truth (4 parameters)
        cov4 = np.linalg.inv(JtJ[:4, :4])
        dp4 = cov4 @ (J[:, :4].T @ r)
        rows.append(dict(d=d, phi=phi, theta=theta, shape=shape, bias_d_rel=float(dp[0] / d),
                         bias_d_over_se=float(dp[0] / se[0]), se_d_rel=float(se[0] / d),
                         bias_d_rel_shape_known=float(dp4[0] / d),
                         bias_d_over_se_shape_known=float(dp4[0] / math.sqrt(cov4[0, 0])),
                         rms_r=float(np.sqrt(np.mean(r ** 2))), dp=[float(x) for x in dp], se=[float(x) for x in se],
                         seconds=time.perf_counter() - t0))
        print(json.dumps(rows[-1]), flush=True)
    with open(out, "w") as fh:
        json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main(sys.argv[2])
