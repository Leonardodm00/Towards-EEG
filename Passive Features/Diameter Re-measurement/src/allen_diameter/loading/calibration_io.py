"""Files of the kernel calibration -- Block 10 in specs/SPEC.md.

Plane scans: one JSON file, {"meta": {...}, "scans": [...]}, every array a
list with null for NaN. Calibration result: one JSON file with the growth fit
(knots, values, per-node c and z_ax, residual summary), the first-stage kernel
table when one could be built (or the reason it could not), and the
configuration signature it was made under.

Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from ..analysis.calibration import PlaneScan

_SCAN_ARRAYS = ("ks", "z_um", "omega_um2", "vw_um2", "area_gl_um", "depth_gl", "B_bar")
_SCAN_SCALARS = ("node_id", "d_hat_um", "phi_rad", "alpha_hat", "k_star", "z_sub_um")


def _list(x):
    return [None if not math.isfinite(float(v)) else float(v) for v in np.asarray(x, dtype=float).ravel()]


def _array(x):
    return np.array([np.nan if v is None else v for v in x], dtype=float)


def _scalar(v):
    v = float(v)
    return v if math.isfinite(v) else None


def scan_to_dict(sc):
    d = {k: (int(getattr(sc, k)) if k in ("node_id", "k_star") else _scalar(getattr(sc, k))) for k in _SCAN_SCALARS}
    d.update({k: _list(getattr(sc, k)) for k in _SCAN_ARRAYS})
    d["ks"] = [int(k) for k in sc.ks]
    d["status"] = list(sc.status)
    return d


def scan_from_dict(d):
    nan = float("nan")
    return PlaneScan(int(d["node_id"]), *(nan if d[k] is None else float(d[k]) for k in ("d_hat_um", "phi_rad", "alpha_hat")),
                     int(d["k_star"]), nan if d["z_sub_um"] is None else float(d["z_sub_um"]),
                     np.asarray(d["ks"], dtype=int), *(_array(d[k]) for k in _SCAN_ARRAYS[1:]), tuple(d["status"]))


def _write_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True, allow_nan=False)
    os.replace(tmp, path)


def save_scans(scans, path, meta=None):
    _write_json({"meta": meta or {}, "scans": [scan_to_dict(sc) for sc in scans]}, path)


def load_scans(path):
    """(list of PlaneScan, meta dict)."""
    with open(path) as f:
        obj = json.load(f)
    return [scan_from_dict(d) for d in obj["scans"]], obj.get("meta", {})


def growth_to_dict(g):
    return dict(convention=g.convention, statistic=g.statistic, interp=g.interp, knot_step_um=g.knot_step_um,
                knots_um=_list(g.knots_um), g_um2=_list(g.g_um2), node_ids=[int(i) for i in g.node_ids],
                c_um2=_list(g.c_um2), z_ax_um=_list(g.z_ax_um), n_obs_per_interval=[int(n) for n in g.n_obs_per_interval],
                rms_by_offset={str(k): v for k, v in g.rms_by_offset().items()},
                dropped_nodes=[int(i) for i in g.dropped_nodes], cost=g.cost, success=g.success, message=g.message,
                nfev=g.nfev)


def save_calibration(path, growth, kernel=None, kernel_error="", area_ratio=None, meta=None):
    """growth: GrowthFit; kernel: the first-stage RendererConfig (or None, with kernel_error)."""
    out = {"growth": growth_to_dict(growth), "meta": meta or {},
           "area_ratio_by_offset": {str(k): v for k, v in (area_ratio or {}).items()}}
    if kernel is not None:
        out["kernel"] = dict(kernel_table_delta_um=list(kernel.kernel_table_delta_um),
                             kernel_table_sigma_um=list(kernel.kernel_table_sigma_um), sigma_r0_um=kernel.sigma_r0_um,
                             kernel_continuation=kernel.kernel_continuation,
                             kernel_continuation_slope=kernel.kernel_continuation_slope)
    else:
        out["kernel_error"] = str(kernel_error)
    _write_json(out, path)
