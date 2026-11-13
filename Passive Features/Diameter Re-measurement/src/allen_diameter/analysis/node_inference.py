"""The per-node fit of the forward model -- Block 30 in specs/SPEC.md (contract stub, wave 2; D-129, D-130, D-132).

The entry of Block 30 in specs/SPEC.md is the specification. Pure ASCII, for the cluster.
"""
from __future__ import annotations


def select_planes(tau0, B_gl, sigma_gl, cfg):
    """-> keep (n_planes,) bool."""
    raise NotImplementedError("contract stub")


def fit_node(profiles_gl, B_gl, sigma_gl, forward, theta0, cfg, planes=None):
    """-> NodeFit (theta, cov, se, cost, chi2_per_plane, planes, status, at_bound, n_starts, nfev)."""
    raise NotImplementedError("contract stub")
