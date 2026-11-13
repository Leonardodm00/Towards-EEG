"""The forward model on the measuring line only -- Block 26 in specs/SPEC.md (contract stub, wave 1; D-129, D-130).

The entry of Block 26 in specs/SPEC.md is the specification: signatures, shapes,
axis order, dtypes, units, oracles and tolerances. Pure ASCII, for the cluster.
"""
from __future__ import annotations

def pixels_needed(origin_xy, y_hat, e_u, v, n_avg, step, res0_um):
    """-> (cols, rows) int arrays: the pixels profiles.sample_profile's bilinear read touches."""
    raise NotImplementedError("contract stub")


def line_profiles(tube, mu, ks, origin_xy, y_hat, e_u, v, n_avg, step, cfg, absorption=None):
    """-> tau (n_planes, n_v) float64, dimensionless: what profiles.sample_profile reads from the pixel means of
    tau_k (before the camera chain). absorption None: cfg.inference.absorption."""
    raise NotImplementedError("contract stub")
