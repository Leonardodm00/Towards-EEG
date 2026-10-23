"""Files of the kernel calibration -- Blocks 10 and 15 in specs/SPEC.md.

Plane scans: one JSON file, {"meta": {...}, "scans": [...]}, every array a
list with null for NaN; both sets in one file ("set" per scan); the per-plane
profiles of D-075 (v_um, w_gl, bhat_gl) and the core fit's details are
written when the scan holds them (2026-10-23) and read as None when a file
does not. The patch images a scan may carry are never written here:
save_patches writes the example node's into an .npz. Calibration result: one
JSON file with the growth fit, the first-stage kernel table (cut at
table_max_delta_um) or the reason it could not be built, the anchor and its
SE (10.A), the area diagnostic and the configuration signature. Far fit
(15.6): one JSON file with far_field.far_fit_summary and the renderer table
for production.

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
# 2026-10-23: optional per-plane arrays (None in a PlaneScan built or saved before)
_SCAN_OPTIONAL_1D = ("se_omega_um2", "amp_gl", "v_c_um", "rms_gl", "window_half_um", "line_end_gl")
_SCAN_OPTIONAL_INT = ("n_win", "passes")
_SCAN_OPTIONAL_2D = ("b_coef", "w_gl", "bhat_gl")


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
    d["set"] = str(sc.set)
    for k in _SCAN_OPTIONAL_1D:
        a = getattr(sc, k)
        if a is not None:
            d[k] = _list(a)
    for k in _SCAN_OPTIONAL_INT:
        a = getattr(sc, k)
        if a is not None:
            d[k] = [int(x) for x in np.asarray(a).ravel()]
    for k in _SCAN_OPTIONAL_2D:
        a = getattr(sc, k)
        if a is not None:
            a = np.asarray(a, dtype=float)
            d[k] = {"shape": list(a.shape), "data": _list(a)}
    if sc.v_um is not None:
        d["v_um"] = _list(sc.v_um)
    return d


def scan_from_dict(d):
    nan = float("nan")
    kw = {}
    for k in _SCAN_OPTIONAL_1D:
        if d.get(k) is not None:
            kw[k] = _array(d[k])
    for k in _SCAN_OPTIONAL_INT:
        if d.get(k) is not None:
            kw[k] = np.asarray(d[k], dtype=int)
    for k in _SCAN_OPTIONAL_2D:
        if d.get(k) is not None:
            kw[k] = _array(d[k]["data"]).reshape(d[k]["shape"])
    if d.get("v_um") is not None:
        kw["v_um"] = _array(d["v_um"])
    return PlaneScan(int(d["node_id"]), *(nan if d[k] is None else float(d[k]) for k in ("d_hat_um", "phi_rad", "alpha_hat")),
                     int(d["k_star"]), nan if d["z_sub_um"] is None else float(d["z_sub_um"]),
                     np.asarray(d["ks"], dtype=int), *(_array(d[k]) for k in _SCAN_ARRAYS[1:]), tuple(d["status"]),
                     str(d.get("set", "near")), **kw)


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


def save_patches(path, records):
    """The example nodes' patch images (10.F): records = {set: PlaneScan with
    images}; one .npz with, per set, <set>_node_id, <set>_ks, <set>_stack
    (uint8 (n, H, W)), <set>_far (bool), <set>_left, <set>_top,
    <set>_res_um_px, <set>_v_c_um and <set>_window_half_um (the last windows),
    <set>_scan_ks, <set>_status, <set>_k_star, <set>_phi_rad, <set>_theta_rad, <set>_cx_um, <set>_cy_um (the
    line's geometry)."""
    out = {}
    for which, sc in records.items():
        if sc is None or sc.images is None:
            continue
        im = sc.images
        out.update({which + "_node_id": np.array(int(sc.node_id)), which + "_ks": np.asarray(im["ks"], dtype=int),
                    which + "_stack": np.asarray(im["stack"], dtype=np.uint8), which + "_far": np.asarray(im["far"], dtype=bool),
                    which + "_left": np.array(int(im["left"])), which + "_top": np.array(int(im["top"])),
                    which + "_res_um_px": np.array(float(im["res_um_px"])),
                    which + "_v_c_um": np.asarray(sc.v_c_um if sc.v_c_um is not None else np.full(len(sc.ks), np.nan)),
                    which + "_window_half_um": np.asarray(sc.window_half_um if sc.window_half_um is not None
                                                          else np.full(len(sc.ks), np.nan)),
                    which + "_scan_ks": np.asarray(sc.ks, dtype=int), which + "_phi_rad": np.array(float(sc.phi_rad)),
                    which + "_cx_um": np.array(float(im["cx_um"])), which + "_cy_um": np.array(float(im["cy_um"])),
                    which + "_theta_rad": np.array(float(im["theta_rad"])), which + "_k_star": np.array(int(sc.k_star)),
                    which + "_status": np.array(list(sc.status))})
    np.savez_compressed(path, **out)


def load_patches(path):
    """{set: dict of the arrays save_patches wrote (keys without the prefix)}."""
    z = np.load(path)
    out = {}
    for key in z.files:
        which, name = key.split("_", 1)
        out.setdefault(which, {})[name] = z[key]
    return out


def growth_to_dict(g):
    return dict(convention=g.convention, statistic=g.statistic, interp=g.interp, knot_step_um=g.knot_step_um,
                knots_um=_list(g.knots_um), g_um2=_list(g.g_um2), node_ids=[int(i) for i in g.node_ids],
                c_um2=_list(g.c_um2), z_ax_um=_list(g.z_ax_um), n_obs_per_interval=[int(n) for n in g.n_obs_per_interval],
                rms_by_offset={str(k): v for k, v in g.rms_by_offset().items()},
                dropped_nodes=[int(i) for i in g.dropped_nodes], cost=g.cost, success=g.success, message=g.message,
                nfev=g.nfev)


def kernel_to_dict(kernel):
    return dict(kernel_table_delta_um=list(kernel.kernel_table_delta_um),
                kernel_table_sigma_um=list(kernel.kernel_table_sigma_um), sigma_r0_um=kernel.sigma_r0_um,
                kernel_continuation=kernel.kernel_continuation, kernel_continuation_slope=kernel.kernel_continuation_slope)


def save_calibration(path, growth, kernel=None, kernel_error="", area_ratio=None, meta=None, anchor=None):
    """growth: GrowthFit; kernel: the first-stage RendererConfig (or None, with
    kernel_error); anchor: {"sigma0_um", "se_um", "n_boot_ok", "seed",
    "n_resamples"} (10.A) or None."""
    out = {"growth": growth_to_dict(growth), "meta": meta or {},
           "area_ratio_by_offset": {str(k): v for k, v in (area_ratio or {}).items()}}
    if kernel is not None:
        out["kernel"] = kernel_to_dict(kernel)
    else:
        out["kernel_error"] = str(kernel_error)
    if anchor is not None:
        out["anchor"] = {k: (_scalar(v) if isinstance(v, float) else v) for k, v in anchor.items()}
    _write_json(out, path)


def load_calibration(path):
    with open(path) as f:
        return json.load(f)


def save_far_fit(path, summary, kernel=None, kernel_error="", meta=None):
    """summary: far_field.far_fit_summary's dict; kernel: the production
    RendererConfig (the near table cut at table_max_delta_um, continuation
    linear with the fitted slope) or None with kernel_error."""
    out = {"far_fit": summary, "meta": meta or {}}
    if kernel is not None:
        out["kernel"] = kernel_to_dict(kernel)
    else:
        out["kernel_error"] = str(kernel_error)
    _write_json(out, path)
