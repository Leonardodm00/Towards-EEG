"""Step 2's files: the chosen nodes, a cell's scans, the starting kernel, the result -- Block 52 in specs/SPEC.md
(section 5, "Kernel calibration, step 2"). File reading and writing only: no fit, no rendering, no plotting.

The chosen nodes (D-261, D-273.5; the format is the orchestrator's): a CSV text file, UTF-8, header line
"specimen_id,node_id", one row per chosen node, blank lines and lines starting with "#" skipped.
A cell's scans: the scans file of calibrate_kernel.py real (calibration_io.save_scans), whose meta holds
"config" (the full signature), "calibration" (dataclasses.asdict of CalibrationConfig) and "nodes" (the node
records of calibration.node_records). The start: the "kernel" of calibration_<S>.json and, optionally, the slope of
far_fit_<S>.json. The result: one JSON file {"kernel", "step2", "meta"}.

Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import csv
import dataclasses
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from . import calibration_io
from ..config import CalibrationConfig, config_from_dict

CHOSEN_HEADER = ("specimen_id", "node_id")
_META_KEYS = ("config", "calibration", "nodes")
_START_KEYS = ("kernel_table_delta_um", "kernel_table_sigma_um", "sigma_r0_um")


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


def _int_field(text, path, lineno, line):
    try:
        return int(text)
    except ValueError:
        raise ValueError("%s, line %d: %r: a row is two integers specimen_id,node_id" % (path, lineno, line)) from None


def read_chosen_nodes(path) -> Dict[int, Tuple[int, ...]]:
    """The user's chosen calibration nodes (D-261, D-273.5): a CSV with the header line "specimen_id,node_id" and
    one row per node; blank lines and lines starting with "#" skipped. Returns {specimen_id: node ids ascending}.
    ValueError naming the line: a missing or different header, a row without exactly two integers, a node listed
    twice for one cell, no row."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        lines = f.read().splitlines()
    header_seen = False
    chosen: Dict[int, Dict[int, int]] = {}
    for lineno, line in enumerate(lines, start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        values = tuple(v.strip() for v in next(csv.reader([line])))
        if not header_seen:
            if values != CHOSEN_HEADER:
                raise ValueError("%s, line %d: %r: the header must be %r" % (path, lineno, line, ",".join(CHOSEN_HEADER)))
            header_seen = True
            continue
        if len(values) != 2:
            raise ValueError("%s, line %d: %r: a row is two integers specimen_id,node_id" % (path, lineno, line))
        sid, nid = (_int_field(v, path, lineno, line) for v in values)
        cell = chosen.setdefault(sid, {})
        if nid in cell:
            raise ValueError("%s, line %d: node %d of cell %d is already listed on line %d"
                             % (path, lineno, nid, sid, cell[nid]))
        cell[nid] = lineno
    if not header_seen:
        raise ValueError("%s, line %d: no header %r" % (path, len(lines), ",".join(CHOSEN_HEADER)))
    if not chosen:
        raise ValueError("%s, line %d: no row after the header" % (path, len(lines)))
    return {sid: tuple(sorted(cell)) for sid, cell in chosen.items()}


def _lists_to_tuples(value):
    return tuple(_lists_to_tuples(v) for v in value) if isinstance(value, list) else value


def load_cell(cell_id, scans_path) -> CellScans:
    """A cell's scans file (calibration_io.load_scans) as a CellScans. ValueError: meta without "config",
    "calibration" or "nodes"; two scans or two node records with one node id."""
    scans, meta = calibration_io.load_scans(scans_path)
    meta = meta or {}
    missing = [k for k in _META_KEYS if k not in meta]
    if missing:
        raise ValueError("%s: the scans file's meta has no %s (a scans file of calibrate_kernel.py real holds %s)"
                         % (scans_path, ", ".join(missing), ", ".join(_META_KEYS)))
    seen, twice = set(), set()
    for sc in scans:
        nid = int(sc.node_id)
        (twice if nid in seen else seen).add(nid)
    if twice:
        raise ValueError("%s: two scans with one node id: %s" % (scans_path, sorted(twice)))
    nodes: Dict[int, Dict[str, Any]] = {}
    twice = set()
    for rec in meta["nodes"]:
        nid = int(rec["node_id"])
        if nid in nodes:
            twice.add(nid)
        nodes[nid] = rec
    if twice:
        raise ValueError("%s: two node records with one node id: %s" % (scans_path, sorted(twice)))
    cfg = config_from_dict(meta["config"])
    cal = CalibrationConfig(**{k: _lists_to_tuples(v) for k, v in meta["calibration"].items()})
    cfg = dataclasses.replace(cfg, calibration=cal)
    cfg.validate()
    return CellScans(cell_id=int(cell_id), scans=tuple(scans), nodes=nodes, config=cfg, meta=meta)


def load_start(calibration_path, far_fit_path=None) -> Dict[str, Any]:
    """The starting kernel: {"kernel_table_delta_um": tuple (um), "kernel_table_sigma_um": tuple (um),
    "sigma_r0_um": float, "gamma": float} from the calibration JSON's "kernel"; gamma from the far-fit JSON's
    kernel.kernel_continuation_slope when far_fit_path is given, else the calibration kernel's. ValueError when the
    calibration JSON has no "kernel" (its kernel_error in the message)."""
    cal = calibration_io.load_calibration(calibration_path)
    k = cal.get("kernel")
    if not k:
        raise ValueError("%s has no kernel table: %s" % (calibration_path, cal.get("kernel_error", "(no kernel_error)")))
    out = {"kernel_table_delta_um": tuple(float(x) for x in k["kernel_table_delta_um"]),
           "kernel_table_sigma_um": tuple(float(x) for x in k["kernel_table_sigma_um"]),
           "sigma_r0_um": float(k["sigma_r0_um"])}
    if far_fit_path is not None:
        far_obj = calibration_io.load_calibration(far_fit_path)
        far = far_obj.get("kernel")
        if not far:
            raise ValueError("%s has no kernel: %s" % (far_fit_path, far_obj.get("kernel_error", "(no kernel_error)")))
        out["gamma"] = float(far["kernel_continuation_slope"])
    else:
        out["gamma"] = float(k["kernel_continuation_slope"])
    return out


def _nan_to_none(value):
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: _nan_to_none(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_nan_to_none(v) for v in value]
    return value


def save_step2(path, kernel, report, meta: Optional[Dict[str, Any]] = None) -> None:
    """Write step 2's result as JSON: "kernel" = calibration_io.kernel_to_dict(kernel) plus "kernel_family";
    "step2" = report (a JSON-ready dict; NaN written as null); "meta". kernel: a RendererConfig."""
    kd = calibration_io.kernel_to_dict(kernel)
    kd["kernel_family"] = str(kernel.kernel_family)
    out = {"kernel": _nan_to_none(kd), "step2": _nan_to_none(report), "meta": _nan_to_none(meta or {})}
    calibration_io._write_json(out, str(path))


def load_step2(path) -> Dict[str, Any]:
    """The dict save_step2 wrote."""
    with open(path) as f:
        return json.load(f)
