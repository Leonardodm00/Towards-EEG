"""The twins of step 2: settings and real measurements per chosen node, and the training/validation split --
Block 53 in specs/SPEC.md (section 5, "Kernel calibration, step 2"). No file access, no rendering.

CONTRACT STUB (wave 9, 2026-11-27): signatures and docstrings are binding; the bodies are Block 53's.
Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

SETS = ("near", "far")
ROLES = ("train", "validation")


@dataclass(frozen=True)
class Twin:
    """One chosen calibration node: the held settings of its twin and the real scan's measurements. The arrays
    are (n,) over the real scan's planes, read-only. Units: um, 1/um, rad, grey levels (gl), um^2, gl x um."""

    cell_id: int
    node_id: int
    set: str                    # "near" | "far"
    role: str                   # "train" | "validation"
    d_um: float                 # Allen's diameter, taken as true (D-261.5)
    mu_per_um: float            # the pilot's absorption coefficient, held (D-274.1)
    k_star: int                 # the pilot's sharpest plane
    z_sub_um: float             # the pilot's sub-plane depth: the start of the twin's axis depth
    cx_um: float                # the pilot's lateral centre (image frame)
    cy_um: float
    theta_rad: float            # heading
    phi_rad: float              # tilt
    B_pilot_gl: float           # the pilot's B_bar (gl): read by the scan's stripe support (amendment 1)
    d_pilot_um: float           # the pilot's d_hat_um (um): the scan's own-cylinder diameter (amendment 1)
    alpha_pilot: float          # the pilot's alpha_hat (amendment 1)
    ks: np.ndarray              # (n,) int64 plane indices of the real scan
    B_gl: np.ndarray            # (n,) the real plane's background at the node (PlaneScan.B_bar)
    noise_sd_gl: np.ndarray     # (n,) the real plane's background SD (PlaneScan.bg_sd_gl)
    omega_um2: np.ndarray       # (n,) the real squared core width (NaN unless status ok)
    se_omega_um2: np.ndarray    # (n,) its SE from the plane's own fit
    area_gl_um: np.ndarray      # (n,) the real dip's windowed area
    status: Tuple[str, ...]     # (n,) the real scan's status per plane


def build_twins(cell_id, scans, nodes, chosen_ids):
    """list[Twin] in the order of chosen_ids, role "train": scans a sequence of PlaneScan, nodes {node id: node
    record} (Block 52's CellScans). ValueError naming every offending node id: no scan, no record, a record whose
    set differs from the scan's, a non-finite or non-positive allen_diameter_um, a pilot without finite
    mu_hat_per_um > 0, cx_um, cy_um, theta_rad, phi_rad, z_sub_um, alpha_hat, finite B_bar > 0 and d_hat_um > 0, or
    without k_star; a scan without bg_sd_gl; a scan whose k_star differs from the pilot's or whose z_sub_um differs
    from the pilot's by more than 1e-9 um (amendment 1)."""
    raise NotImplementedError("Block 53")


def split_twins(twins, validation_fraction, seed):
    """The twins in the same order with their roles set (D-274.4): by node, stratified by set and by cell, the
    rule of SPEC Block 53 (largest remainders, a cell never emptied of training twins, a seat a capped cell cannot
    take passed on in repeated passes, a seeded permutation of the sorted node ids). Independent of the input order. ValueError: a fraction outside [0, 1); two twins with one
    (cell_id, node_id)."""
    raise NotImplementedError("Block 53")


def fit_planes(twin, s2cfg, which):
    """(n,) bool over twin.ks: which "near": |k - k*| <= s2cfg.near_planes; "far": far_planes[0] <= |k - k*| <=
    far_planes[1]; in both, status "ok" and a finite omega_um2. ValueError for another which."""
    raise NotImplementedError("Block 53")
