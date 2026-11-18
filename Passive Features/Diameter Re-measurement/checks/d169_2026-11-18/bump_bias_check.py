"""What the near-bump cut and the centre weights recover when a second dip sits beside the tube (D-169, D-171).

Standalone check, not pipeline code. Data: a tube of d = 1 um, mu = 1 /um in 7 planes (dz 0.28 um), the blur a
Gaussian whose SD grows with defocus (0.1 um in focus, +0.4 um per um), background 210 gl, no noise; plus a
second dip (Gaussian along the line, the same in every plane) that the one-tube model does not contain. Fits:
none (unweighted), cut (zero weight beyond the bump cut), weights (exp(-|v - v_c| / lambda), lambda = 1.25 x
FWHM from the true diameter), both. Reported: the fitted diameter's relative error. Second part: how often the
cut fires inside a 2 um wide flat-bottomed dip with texture, for three settings of bump_half_depth.
Run:  python3 bump_bias_check.py > bump_bias_check.out
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from allen_diameter.analysis import line_weights as LW  # noqa: E402
from allen_diameter.analysis import node_inference as NI  # noqa: E402
from allen_diameter.config import default_config  # noqa: E402
from allen_diameter.model import tube_model as M  # noqa: E402

CFG = default_config()
SEED, B_GL, SIGMA = 20261118, 210.0, 3.0
TRUTH = np.array([1.0, 1.0, 0.1, 0.07])      # d um, mu 1/um, v0 um, z_ax um
START = np.array([1.3, 0.7, 0.0, 0.0])
Z = 0.28 * np.arange(-3, 4)
V = 0.1144 * np.arange(-26, 27)
S0, GROWTH = 0.1, 0.4
N_Q = M.n_quadrature_nodes(CFG.inference.d_bounds_um[1], S0, CFG.measure.fit_quad_min_nodes,
                           CFG.measure.fit_quad_nodes_per_sigma)
CASES = ((1.3, 25.0, 0.3), (1.3, 60.0, 0.3), (1.5, 80.0, 0.6), (1.0, 60.0, 0.3), (1.0, 60.0, 0.5), (0.8, 60.0, 0.4))


def forward(d, mu, v0, z_ax):
    out = np.empty((Z.size, V.size))
    for k, zk in enumerate(Z):
        out[k] = M.model_profile(V, d, mu * d, v0, np.sqrt(S0 ** 2 + (GROWTH * (zk - z_ax)) ** 2), 1.0, N_Q)
    return out


def main():
    B = np.full(Z.size, B_GL)
    I = B_GL * forward(*TRUTH)
    fwhm = LW.allen_fwhm_um(TRUTH[0], S0)
    lam = 1.25 * fwhm
    print("central dip %.1f gl deep in focus; FWHM from the true diameter %.3f um, lambda %.3f um; m = 2 noise SDs "
          "of %.1f gl" % (B_GL - I.min(), fwhm, lam, SIGMA))
    print("second dip: offset um, depth gl, SD um | cut at um | d error: none, cut, weights, both")
    for sep, depth, sd in CASES:
        Ib = I - depth * np.exp(-0.5 * ((V - (TRUTH[2] + sep)) / sd) ** 2)[None]
        keep, _lo, hi = LW.bump_mask(Ib, B, SIGMA, V, TRUTH[2], fwhm, 2.0, 0.5, 1.0)
        w = LW.centre_weights(V, TRUTH[2], lam)
        err = []
        for ww in (None, keep.astype(float), w, w * keep):
            f = NI.fit_node(Ib, B, SIGMA, forward, START, CFG, weights=ww)
            err.append("%+6.1f %%" % (100.0 * (f.theta[0] / TRUTH[0] - 1.0)))
        print("  %.1f  %3.0f  %.1f | %s | %s" % (sep, depth, sd, ("%.2f" % hi) if np.isfinite(hi) else "none",
                                               ", ".join(err)))
    rng = np.random.default_rng(SEED)
    print("a 2 um wide flat-bottomed dip (60 gl) with texture of SD 2 noise SDs inside, 7 planes, 300 draws:")
    for h in (0.5, 0.75, 0.9):
        n_cut = 0
        for _ in range(300):
            flat = 60.0 / (1.0 + np.exp((np.abs(V) - 1.0) / 0.08))
            tex = np.convolve(rng.normal(0.0, 2.0 * SIGMA, V.size), np.ones(3) / 3.0, mode="same") * (np.abs(V) < 0.9)
            P = B_GL - (flat + tex)[None] + rng.normal(0.0, SIGMA, (Z.size, V.size))
            keep, _lo, _hi = LW.bump_mask(P, B, SIGMA, V, 0.0, fwhm, 2.0, h, 1.0)
            n_cut += not keep.all()
        print("  bump_half_depth %.2f: cut in %.2f of the draws" % (h, n_cut / 300.0))


if __name__ == "__main__":
    main()
