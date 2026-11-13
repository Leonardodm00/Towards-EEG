"""Per-node features by round, and the scene rebuilt from them -- Block 23 in specs/SPEC.md (contract stub, wave 1; D-129, D-130).

The entry of Block 23 in specs/SPEC.md is the specification: signatures, shapes,
axis order, dtypes, units, oracles and tolerances. Pure ASCII, for the cluster.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SOURCES = ("allen", "pilot", "refined", "fine")


@dataclass(frozen=True)
class NodeFeatures:
    """node_id (N,) int64; swc_type (N,) int64; d_um (N,) um > 0; mu_per_um (N,) 1/um >= 0;
    offset_um (N, 3) = (dx, dy, dz) um; source (N,) str in SOURCES; flags (N,) str, ';'-joined;
    round_index int. Validation (ValueError) as Block 23 lists."""

    node_id: np.ndarray
    swc_type: np.ndarray
    d_um: np.ndarray
    mu_per_um: np.ndarray
    offset_um: np.ndarray
    source: np.ndarray
    flags: np.ndarray
    round_index: int = 0


@dataclass(frozen=True)
class FeaturedScene:
    """seg_xyz (n, 6) um; seg_d_um (n, 2) um; seg_mu_per_um (n, 2) 1/um; seg_ids (n, 2) int64
    (child, parent); seg_type (n,) int64 (the child's SWC type); soma (m, 3) = x, y, r um."""

    seg_xyz: np.ndarray
    seg_d_um: np.ndarray
    seg_mu_per_um: np.ndarray
    seg_ids: np.ndarray
    seg_type: np.ndarray
    soma: np.ndarray


def initial_features(swc, types, mu0_per_um):
    """Round 0: d = twice Allen's radius, mu = mu0_per_um, offsets 0, source 'allen' -> NodeFeatures."""
    raise NotImplementedError("contract stub")


def with_rows(features, rows, round_index):
    """A new NodeFeatures: the nodes named in rows replaced, the others carried; the input unchanged."""
    raise NotImplementedError("contract stub")


def featured_scene(scene, features):
    """plane_iteration.Scene + NodeFeatures -> FeaturedScene (Block 23's rule for ends without features)."""
    raise NotImplementedError("contract stub")


def max_relative_change(a, b):
    """{'d': ..., 'mu': ..., 'offset_um': ...} between two NodeFeatures over the same nodes."""
    raise NotImplementedError("contract stub")
