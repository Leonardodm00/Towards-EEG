"""The cleaning iteration -- Block 32 in specs/SPEC.md (contract stub, wave 3; D-129, D-130, D-132, D-137, D-138).

The entry of Block 32 in specs/SPEC.md is the specification. Pure ASCII, for the cluster.
"""
from __future__ import annotations


def round_once(features, scene, nodes, cfg, predict_factory, shadow_fn=None):
    """-> (rows, per_node): one round from the features of the round before."""
    raise NotImplementedError("contract stub")


def run_rounds(features0, scene, nodes, cfg, sigma_gl, predict_factory, example_ids=(), shadow_fn=None, keep_ids=None):
    """-> IterationResult (features, history, stopped, rounds, profiles_before, profiles_after, cleaned, shadow, tiers, fits). NodeData and IterationResult are the frozen dataclasses of the Block 32 entry."""
    raise NotImplementedError("contract stub")
