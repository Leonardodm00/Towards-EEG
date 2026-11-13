"""The coarse refinement of a node's features -- Block 29 in specs/SPEC.md (contract stub, wave 2; D-129, D-130, D-132).

The entry of Block 29 in specs/SPEC.md is the specification. Pure ASCII, for the cluster.
"""
from __future__ import annotations


def refine_node(profiles_gl, B_gl, v, predict, theta0, cfg):
    """-> dict (d_um, mu_per_um, v0_um, dz_um, cost, rms_gl, status, at_bound, nfev)."""
    raise NotImplementedError("contract stub")


def offset_xyz(v0_um, dz_um, y_hat):
    """-> (3,) image-frame offset (v0 y_hat_x, v0 y_hat_y, dz)."""
    raise NotImplementedError("contract stub")
