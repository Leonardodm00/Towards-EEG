"""Reproducer (tester, 2026-11-14), finding F1: Block 36's local tube is cut at
tube_half_length_um = 5 um, which on a tilted node is inside the blur's reach;
the fast line renderer (Block 35) then misses the shadow of the node's own
branch beyond +-5 um, a deficit of 1 to 7 % of the dip in every plane of
node_planes (k* +- 4), up to about 11 gl at 210 gl (phi 60 deg, d 1 um).
Compares U = 5 um with U = 20 um, the line and the planes of the production
configuration (Block 5's profile_offsets and n_along). Flat tubes: no change.

Run (about 1 min): python3 tests/spec/repro_block36_tube_cut.py
Pure ASCII.
"""
import dataclasses
import math

import numpy as np

import nc_helpers  # noqa: F401  (sys.path)
from allen_diameter.config import default_config
from allen_diameter.analysis import profiles
from allen_diameter.model import geometry, line_fast


def main():
    cfg = default_config()
    cfg = dataclasses.replace(cfg, inference=dataclasses.replace(cfg.inference, absorption="partition_vertical"))
    m = cfg.measure
    v, n_avg, step = profiles.profile_offsets(m), profiles.n_along(m), float(m.profile_step_um)
    ks = np.arange(-cfg.cleaning.plane_half_range, cfg.cleaning.plane_half_range + 1)
    yh, eu, o = np.array([0.0, 1.0]), np.array([1.0, 0.0]), np.array([0.0, 0.0])
    print("U = %.1f um (inference.tube_half_length_um) against U = 20 um; partition_vertical" %
          cfg.inference.tube_half_length_um)
    for phi in (0.0, 30.0, 60.0):
        for d in (0.5, 1.0):
            tau = {}
            for U in (cfg.inference.tube_half_length_um, 20.0):
                t = geometry.Tube(c=(0.0, 0.0, 0.0), r=d / 2, phi=math.radians(phi), theta=0.0, half_length=U)
                tau[U] = line_fast.line_profiles(t, 1.0, ks, o, yh, eu, v, n_avg, step, cfg)
            diff = np.abs(tau[cfg.inference.tube_half_length_um] - tau[20.0]).max(axis=1)
            dip = (1.0 - tau[20.0]).max(axis=1)
            print("phi %4.0f deg d %.1f um: max |dtau| over planes %.4f (%.2f gl at 210 gl); "
                  "relative to the plane's dip %.3f..%.3f" % (phi, d, diff.max(), 210 * diff.max(),
                                                               (diff / dip).min(), (diff / dip).max()))


if __name__ == "__main__":
    main()
