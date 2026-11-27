"""The screening export's files -- Block 61 in specs/SPEC.md (section 5, "The screening export"). File reading
and writing only.

CONTRACT STUB (wave 10, 2026-11-27): signatures and docstrings are binding; the bodies are Block 61's.
Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

INDEX_COLUMNS = ("specimen_id", "node_id", "set", "allen_diameter_um", "k_star", "phi_deg", "n_planes", "n_ok",
                 "B_focus_gl", "omega_focus_um2")


def save_views(path, views) -> None:
    """One compressed .npz with every field of every NodeView of a cell; load_views returns them exactly."""
    raise NotImplementedError("Block 61")


def load_views(path):
    """list[NodeView] as saved."""
    raise NotImplementedError("Block 61")


def write_index(path, rows) -> None:
    """CSV with INDEX_COLUMNS in that order, one row per node (rows: screening.index_row's dicts)."""
    raise NotImplementedError("Block 61")


def write_candidates(path, views) -> None:
    """The header "specimen_id,node_id" and one row per view, near set first then by node id: Block 52's
    chosen-nodes format (step2_io.read_chosen_nodes reads it back), for the user to cut down (D-278.2)."""
    raise NotImplementedError("Block 61")
