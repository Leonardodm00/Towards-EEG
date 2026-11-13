"""The fast line renderer: one cross-section carried along the tube -- Block 35 in specs/SPEC.md (contract stub, wave 3; D-129, D-130, D-132, D-137, D-138).

The entry of Block 35 in specs/SPEC.md is the specification. Pure ASCII, for the cluster.
"""
from __future__ import annotations


class ExtrusionInvalid(ValueError):
    """A cap of the tube is within reach of the measuring line: the extrusion does not hold (D-138.2)."""


class SectionCache:
    """Cross-sections by (d, mu, phi, aspect, absorption, directions); counters hits, misses."""

    def __init__(self, max_entries=64):
        raise NotImplementedError("contract stub")


def extrusion_valid(tube, ks, origin_xy, y_hat, e_u, v, n_avg, step, cfg):
    """-> bool: no cap within reach of the pixels the line reads."""
    raise NotImplementedError("contract stub")


def line_profiles(tube, mu, ks, origin_xy, y_hat, e_u, v, n_avg, step, cfg, absorption=None, cache=None):
    """-> tau (n_planes, n_v) float64, dimensionless; Block 26's meaning; ExtrusionInvalid where the extrusion does not hold."""
    raise NotImplementedError("contract stub")
