#!/usr/bin/env python3
"""Standalone check (NOT pipeline code), 2026-11-18: what a close (tier-2) neighbour does to the fit of a node's
diameter, and which of the candidate treatments removes the effect.

Truth: two flat tubes rendered JOINTLY with the hybrid absorption (scene_render.render_scene, SPEC Block 24), the
pilot's kernel table. The node's axis runs along x at depth 0; the measuring line runs along y through the node
(53 samples of one pixel, 9 planes k = -4 .. 4, z_k = k * 0.28 um). Data are noise-free, so a fitted value minus
the truth is the bias of the treatment. Standard errors are in tier2_neighbour_se.py (clean central differences).

Treatments, each a weighted least-squares fit of the node's theta = (d, mu, v0, z_ax):
  a  one tube, the neighbour ignored                      (the pipeline today: flag tier2_unmodelled)
  b  the neighbour's shadow, rendered ALONE at its true features, added back; then one tube   (clearing, best case)
  c  as b, the neighbour's diameter wrong by NB_D_ERROR   (clearing with a wrong neighbour)
  d  joint render, the neighbour held at its true features (D-132.2 as written, best case)
  e  joint render, the neighbour held at the wrong diameter
  f  joint render, the neighbour's d and mu free as well  (6 parameters)

Run:  python3 tier2_neighbour_check.py <path to .../Diameter Re-measurement/src>
"""
import dataclasses
import math
import sys
import time

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, sys.argv[1])
from allen_diameter.config import default_config                      # noqa: E402
from allen_diameter.model import geometry as G                         # noqa: E402
from allen_diameter.model import render as R                           # noqa: E402
from allen_diameter.model import scene_render as SR                    # noqa: E402

# ------------------------------------------------------------------ config ---
B_GL, SIGMA_GL = 200.0, 1.0                    # background and noise SD (grey levels)
NODE = dict(d=0.60, mu=0.50, v0=0.0, z=0.0)    # um, 1/um, um, um
NB_D, NB_MU = 0.60, 0.50                       # the neighbour's true diameter (um) and absorption (1/um)
NB_D_ERROR = 0.70                              # treatments c, e: the neighbour's diameter taken as 0.70 x the truth
SCENES = {                                     # name: (lateral offset of the neighbour at the line (um), its depth (um), heading (deg))
    "parallel, 1.0 um aside, same depth": (1.0, 0.0, 0.0),
    "crossing at 45 deg over the node, 0.84 um deeper": (0.0, 0.84, 45.0),
}
N_V, KS, PAD_PX, FACTOR = 53, np.arange(-4, 5), 14, 4
KERNEL_DELTA_UM = (0.0, 0.28, 0.56, 0.84)      # the pilot's kernel table (um) ...
KERNEL_SIGMA_UM = (0.0800, 0.1536, 0.3924, 0.5620)
KERNEL_SLOPE = 0.79                            # ... and its linear continuation
RAY_N_RHO, RAY_N_PSI = 8, 16                   # the hybrid's ray directions (pipeline default 16 x 32), for speed; truth and model alike

base = default_config()
PX, DZ = base.acquisition.res0_um, base.acquisition.dz_um
RC = dataclasses.replace(base.renderer, absorption="hybrid", kernel_table_delta_um=KERNEL_DELTA_UM,
                         kernel_table_sigma_um=KERNEL_SIGMA_UM, sigma_r0_um=KERNEL_SIGMA_UM[0],
                         kernel_continuation_slope=KERNEL_SLOPE, ray_n_rho=RAY_N_RHO, ray_n_psi=RAY_N_PSI)
GRID, INNER = R.block_fine_grid(0, -(N_V // 2), 1, N_V, PX, FACTOR, PAD_PX)   # one pixel column, x = 0, y = v
ZP = KS * DZ
N_RENDER = [0]


# ------------------------------------------------------------------- model ---
def node_tube(d, v0, z):
    return G.Tube((0.0, v0, z), 0.5 * d, 0.0, 0.0)                    # axis along x, infinite


def nb_tube(d, scene):
    off, depth, heading = scene
    return G.Tube((0.0, off, depth), 0.5 * d, 0.0, math.radians(heading))


def tau(tubes, mus):
    """Transmittance on the line, (9 planes, 53 samples), pixel-integrated."""
    N_RENDER[0] += 1
    fine = SR.render_scene(tubes, mus, ZP, GRID, RC).tau
    return R.pixel_integrate(fine[:, INNER[0], INNER[1]], FACTOR)[:, :, 0]


# --------------------------------------------------------------------- fit ---
def fit(residual, x0, lo, hi):
    res = least_squares(residual, x0, bounds=(lo, hi), method="trf", x_scale="jac", diff_step=1e-3, xtol=1e-10,
                        ftol=1e-12, gtol=1e-12, max_nfev=60)
    return res.x, float(np.sqrt(np.mean((res.fun * SIGMA_GL) ** 2)))


def run(name, scene):
    truth = np.array([NODE["d"], NODE["mu"], NODE["v0"], NODE["z"]])
    nb_true = nb_tube(NB_D, scene)
    data = B_GL * tau([node_tube(NODE["d"], NODE["v0"], NODE["z"]), nb_true], [NODE["mu"], NB_MU])
    alone = B_GL * tau([node_tube(NODE["d"], NODE["v0"], NODE["z"])], [NODE["mu"]])
    s_nb_true = 1.0 - tau([nb_true], [NB_MU])
    s_nb_wrong = 1.0 - tau([nb_tube(NB_D * NB_D_ERROR, scene)], [NB_MU])
    print("\n=== %s ===" % name)
    print("the neighbour on the line: peak %.1f gl; node alone: peak dip %.1f gl; joint minus (node alone + neighbour "
          "alone): max |.| %.2f gl" % (B_GL * s_nb_true.max(), (B_GL - alone).max(),
                                       np.abs(data - (alone - B_GL * s_nb_true)).max()))
    x0 = truth * np.array([1.3, 0.8, 1.0, 1.0]) + np.array([0.0, 0.0, 0.05, 0.1])
    lo4, hi4 = np.array([0.1, 0.01, -1.5, -1.5]), np.array([6.0, 5.0, 1.5, 1.5])

    def one(th, target):
        return ((B_GL * tau([node_tube(th[0], th[2], th[3])], [th[1]]) - target) / SIGMA_GL).ravel()

    def joint(th, d_q, mu_q):
        return ((B_GL * tau([node_tube(th[0], th[2], th[3]), nb_tube(d_q, scene)], [th[1], mu_q]) - data)
                / SIGMA_GL).ravel()

    cases = [
        ("a  one tube, neighbour ignored", lambda th: one(th, data), x0, lo4, hi4),
        ("b  cleared with the true neighbour", lambda th: one(th, data + B_GL * s_nb_true), x0, lo4, hi4),
        ("c  cleared, neighbour d x %.2f" % NB_D_ERROR, lambda th: one(th, data + B_GL * s_nb_wrong), x0, lo4, hi4),
        ("d  joint, neighbour held at truth", lambda th: joint(th, NB_D, NB_MU), x0, lo4, hi4),
        ("e  joint, neighbour held at d x %.2f" % NB_D_ERROR, lambda th: joint(th, NB_D * NB_D_ERROR, NB_MU), x0, lo4, hi4),
        ("f  joint, neighbour d and mu free", lambda th: joint(th[:4], th[4], th[5]),
         np.concatenate([x0, [NB_D * NB_D_ERROR, NB_MU]]), np.concatenate([lo4, [0.1, 0.01]]),
         np.concatenate([hi4, [6.0, 5.0]])),
    ]
    print("%-38s %9s %9s %9s %9s %9s" % ("treatment", "d (um)", "bias (%)", "mu (/um)", "z_ax (um)", "rms (gl)"))
    for label, f, start, lo, hi in cases:
        t0 = time.time()
        x, rms = fit(f, start, lo, hi)
        extra = "" if len(x) == 4 else "   neighbour d %.3f um, mu %.3f /um" % (x[4], x[5])
        print("%-38s %9.3f %+9.1f %9.3f %+9.3f %9.2f   [%.0f s]%s" % (
            label, x[0], 100.0 * (x[0] / truth[0] - 1.0), x[1], x[3], rms, time.time() - t0, extra))


if __name__ == "__main__":
    t0 = time.time()
    print("truth: node d %.2f um, mu %.2f /um, v0 %.1f um, z_ax %.1f um; neighbour d %.2f um, mu %.2f /um; B %.0f gl, "
          "noise SD %.1f gl; hybrid; kernel table %s um at %s um, slope %.2f; grid h %.4f um, pad %d px, rays %d x %d"
          % (NODE["d"], NODE["mu"], NODE["v0"], NODE["z"], NB_D, NB_MU, B_GL, SIGMA_GL, KERNEL_SIGMA_UM,
             KERNEL_DELTA_UM, KERNEL_SLOPE, GRID.h, PAD_PX, RAY_N_RHO, RAY_N_PSI))
    only = sys.argv[2:] or list(SCENES)
    for nm in SCENES:
        if any(o in nm for o in only):
            run(nm, SCENES[nm])
    print("\n%d renders, %.0f s" % (N_RENDER[0], time.time() - t0))
