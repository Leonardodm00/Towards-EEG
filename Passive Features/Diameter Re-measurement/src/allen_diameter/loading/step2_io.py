"""Step 2's files: the chosen nodes, a cell's scans, the starting kernel, the result -- Block 52 in specs/SPEC.md
(section 5, "Kernel calibration, step 2"). File reading and writing only.

CONTRACT STUB (wave 9, 2026-11-27): signatures and docstrings are binding; the bodies are Block 52's.
Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class CellScans:
    """One cell's scans file. cell_id: the specimen number; scans: tuple of calibration.PlaneScan; nodes: node id
    -> the node record of meta["nodes"]; config: the DiameterConfig the scans were made with (config_from_dict of
    meta["config"], its calibration replaced by CalibrationConfig(**meta["calibration"])); meta: the file's meta."""

    cell_id: int
    scans: Tuple[Any, ...]
    nodes: Dict[int, Dict[str, Any]]
    config: Any
    meta: Dict[str, Any]


def read_chosen_nodes(path) -> Dict[int, Tuple[int, ...]]:
    """The user's chosen calibration nodes (D-261, D-273.5): a CSV with the header line "specimen_id,node_id" and
    one row per node; blank lines and lines starting with "#" skipped. Returns {specimen_id: node ids ascending}.
    ValueError naming the line: a missing or different header, a row without exactly two integers, a node listed
    twice for one cell, no row."""
    raise NotImplementedError("Block 52")


def load_cell(cell_id, scans_path) -> CellScans:
    """A cell's scans file (calibration_io.load_scans) as a CellScans. ValueError: meta without "config",
    "calibration" or "nodes"; two scans or two node records with one node id."""
    raise NotImplementedError("Block 52")


def load_start(calibration_path, far_fit_path=None) -> Dict[str, Any]:
    """The starting kernel: {"kernel_table_delta_um": tuple (um), "kernel_table_sigma_um": tuple (um),
    "sigma_r0_um": float, "gamma": float} from the calibration JSON's "kernel"; gamma from the far-fit JSON's
    kernel.kernel_continuation_slope when far_fit_path is given, else the calibration kernel's. ValueError when the
    calibration JSON has no "kernel" (its kernel_error in the message)."""
    raise NotImplementedError("Block 52")


def save_step2(path, kernel, report, meta: Optional[Dict[str, Any]] = None) -> None:
    """Write step 2's result as JSON: "kernel" = calibration_io.kernel_to_dict(kernel) plus "kernel_family";
    "step2" = report (a JSON-ready dict; NaN written as null); "meta". kernel: a RendererConfig."""
    raise NotImplementedError("Block 52")


def load_step2(path) -> Dict[str, Any]:
    """The dict save_step2 wrote."""
    raise NotImplementedError("Block 52")
