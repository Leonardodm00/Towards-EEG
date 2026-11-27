"""The screening sheets: each plane's image beside its intensity profile -- Block 62 in specs/SPEC.md
(section 5, "The screening export"; D-226.3, D-227, D-261.1). Figures only: every number drawn is passed in.

CONTRACT STUB (wave 10, 2026-11-27): signatures and docstrings are binding; the bodies are Block 62's.
Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations


def node_sheet(view, marks, extent, dz_um, title="", zoom_half_um=None):
    """The Figure of one node (SPEC Block 62): one pair of panels per plane in the order of view.ks -- the image
    (shared grey scale, extent = (x0, x1, y0, y1) um from screening.image_extent) and the profile I_k(v) with the
    background dashed (shared y-range); a near view as 7 rows of one pair, a far view as 11 rows of three pairs;
    Allen's diameter marks (marks = screening.diameter_marks(view)) on the plane k* only. zoom_half_um (amendment 1): one
    more first row, the plane k* again limited to +-zoom_half_um about the pilot's centre, image and profile, with
    the marks. ValueError when view.ks does not contain view.k_star."""
    raise NotImplementedError("Block 62")


def write_node_sheet(view, marks, extent, dz_um, path, dpi, title="", zoom_half_um=None) -> None:
    """node_sheet saved to path (PNG) at dpi, the figure closed."""
    raise NotImplementedError("Block 62")
