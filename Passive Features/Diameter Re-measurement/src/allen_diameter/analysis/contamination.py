"""The contamination survey of a node's measuring line -- Block 27 in specs/SPEC.md (contract stub, wave 1; D-129, D-130).

The entry of Block 27 in specs/SPEC.md is the specification: signatures, shapes,
axis order, dtypes, units, oracles and tolerances. Pure ASCII, for the cluster.
"""
from __future__ import annotations

CLASSES = ("overlap", "close_in_focus", "diffuse", "clean")   # in this order of precedence


def pollution_profile(branch, index, scene, v, B_gl, mu_per_um, cfg):
    """-> (P (n_v,) gl, per_neighbour (n_q, n_v) gl)."""
    raise NotImplementedError("contract stub")


def node_contamination(branch, index, scene, B_gl, mu_per_um, cfg):
    """-> dict with the keys Block 27 lists."""
    raise NotImplementedError("contract stub")


def pick_examples(rows, counts, rng):
    """-> list of node ids (Block 27; D-130.O7)."""
    raise NotImplementedError("contract stub")
