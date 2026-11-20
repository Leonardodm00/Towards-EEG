"""Figures of the in-focus width calibrated on thick nodes -- Block 43 in specs/SPEC.md (D-201).

figure_node_fit: one node's planes (images, profiles with the fitted model and background, residuals);
figure_summary: the fitted in-focus widths and growth scales of a cell. Plotting only; the API, the inputs and
the oracles are in SPEC Block 43; this file is the contract's stub (wave 4). Not to be confused with
calibration_figures (Block 10.F, the kernel calibration of Cell 7). Pure ASCII, for the cluster.
"""
from __future__ import annotations

from allen_diameter.plotting.cleaning_figures import save  # noqa: F401  (Block 43 re-exports it)

__all__ = ["figure_node_fit", "figure_summary", "save"]


def figure_node_fit(node, zoom_um=5.0):
    """One node's fit on its planes (SPEC Block 43). Returns the matplotlib Figure."""
    raise NotImplementedError("Block 43: figure_node_fit")


def figure_summary(rows, summary):
    """The cell's fitted in-focus widths and growth scales (SPEC Block 43). Returns the matplotlib Figure."""
    raise NotImplementedError("Block 43: figure_summary")
