"""CSV files of the per-node features -- Block 23 in specs/SPEC.md (contract stub, wave 1; D-129, D-130).

The entry of Block 23 in specs/SPEC.md is the specification: signatures, shapes,
axis order, dtypes, units, oracles and tolerances. Pure ASCII, for the cluster.
"""
from __future__ import annotations

def write_features(path, features):
    """One row per node, floats written with repr; round_index in every row."""
    raise NotImplementedError("contract stub")


def read_features(path):
    """-> NodeFeatures, every array bit for bit as written."""
    raise NotImplementedError("contract stub")
