"""The cleaning operator and the two tiers -- Block 28 in specs/SPEC.md (contract stub, wave 2; D-129, D-130, D-132).

The entry of Block 28 in specs/SPEC.md is the specification. Pure ASCII, for the cluster.
"""
from __future__ import annotations


def clean_block(block, S, B_gl):
    """-> CleanResult (block float64 gl, added_gl, over_background)."""
    raise NotImplementedError("contract stub")


def neighbour_tiers(fscene, own_ids, node_xyz, cfg):
    """-> tiers (n_q + n_s,) int in {1, 2} (D-132)."""
    raise NotImplementedError("contract stub")
