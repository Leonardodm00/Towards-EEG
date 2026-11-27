"""The screening views: a calibration node's planes, profiles and Allen's-diameter marks -- Block 61 in
specs/SPEC.md (section 5, "The screening export"; D-226, D-227, D-261.1, D-278). Arrays only: no file access, no
plotting.

CONTRACT STUB (wave 10, 2026-11-27): signatures and docstrings are binding; the bodies are Block 61's.
Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class NodeView:
    """What one node's screening sheet shows. Arrays over the scan's n planes; units um, rad, grey levels (gl),
    um^2."""

    cell_id: int
    node_id: int
    set: str                    # "near" | "far"
    allen_diameter_um: float
    k_star: int
    ks: np.ndarray              # (n,) int64
    stack: np.ndarray           # (n, H, W) uint8: the crop of the raw image stack
    valid: np.ndarray           # (n,) bool: False where the plane could not be fetched (its image zeros)
    left: int                   # px of the full image
    top: int
    res0_um: float
    cx_um: float                # the pilot's centre, heading and tilt
    cy_um: float
    theta_rad: float
    phi_rad: float
    v_um: np.ndarray            # (n_v,) offsets along the measuring line
    profile_gl: np.ndarray      # (n, n_v): I_k(v) = bhat_gl - w_gl (NaN where the scan has none)
    background_gl: np.ndarray   # (n, n_v): the scan's fitted background along the line
    status: Tuple[str, ...]     # (n,)
    omega_um2: np.ndarray       # (n,)
    B_bar: np.ndarray           # (n,)


def apt_nodes(scans, nodes):
    """list[(PlaneScan, node record)]: the scans with their records, near set first, then by node id. ValueError
    naming the node: a scan without a record, a record without a pilot."""
    raise NotImplementedError("Block 61")


def crop_bounds(record, which, cfg):
    """(left, top, width, height) px: calibration_scene.patch_bounds about the pilot's (cx_um, cy_um) with
    cfg.survey.screening_near_half_um (which "near") or screening_far_half_um ("far") and acquisition.res0_um."""
    raise NotImplementedError("Block 61")


def node_view(cell_id, scan, record, patch) -> NodeView:
    """The view of one node: patch = (block, ks, valid, frame) in the fetch_zblock contract over the scan's planes.
    ValueError: the patch's ks are not the scan's ks; a scan without v_um, w_gl or bhat_gl; a non-finite
    allen_diameter_um."""
    raise NotImplementedError("Block 61")


def diameter_marks(view):
    """{"v_um": (-D/2, +D/2), "xy_um": (2, 2) array}: Allen's diameter D on the measuring line (D-261.1), the image
    points (cx_um, cy_um) -+ (D / 2) (-sin theta, cos theta)."""
    raise NotImplementedError("Block 61")


def image_extent(view):
    """(x0, x1, y0, y1) um, the crop's outer pixel edges with pixel (row, col) centred on ((left + col) p,
    (top + row) p): for imshow(extent=(x0, x1, y1, y0), origin="upper"), y increasing downwards."""
    raise NotImplementedError("Block 61")


def index_row(view):
    """dict with specimen_id, node_id, set, allen_diameter_um, k_star, phi_deg, n_planes, n_ok, B_focus_gl,
    omega_focus_um2 (the plane k*; NaN if absent)."""
    raise NotImplementedError("Block 61")
