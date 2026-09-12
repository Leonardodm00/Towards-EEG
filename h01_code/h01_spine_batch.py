"""Run the per-spine profile pipeline over MANY spines and pool the radii.

WHY THIS EXISTS
---------------
h01_radius_census classifies a profile that already exists. Producing that
profile for one spine takes the whole CELL 12 + CELL 14 sequence -- ROI cutout,
component resolve, bridge, mesh, Taubin smooth, centreline, extension, tangent
stabilisation, cross sections, despike. A histogram over a single spine is not
a distribution; it is one object. This module runs that sequence over a list of
sigmas so the histogram means something.

COST, AND WHY THE CACHE MATTERS
-------------------------------
extract_spine_roi issues three bbox cutouts per spine over the network. That is
the dominant cost and the dominant failure mode of a long batch. This driver is
CACHE-FIRST: if h01_spine_roi.roi_paths() finds a saved ROI it reloads the
masks and skips the network entirely, so a re-run after a crash costs nothing
and a partially completed batch resumes where it stopped. Set
`use_cache=False` only to deliberately re-fetch.

FAILURES ARE DATA
-----------------
A spine that will not section is usually a spine whose centreline is wrong, or
one the labeller merged with a neighbour -- which is itself a finding about the
labelling, not a nuisance. Every failure is recorded with its sigma id, the
stage it failed at, and the exception text; nothing is silently dropped, and
the pooled summary always reports how many spines contributed.

WHAT IS POOLED
--------------
The per-station equivalent radius a = sqrt(A / pi) from the CORE of each
profile (grazing ends trimmed). Stations are uniform in arc length, so a count
histogram over the pool is an arc-length-weighted histogram; area-weighted
fractions are reported alongside by h01_radius_census.band_fractions.

DEPENDENCIES: numpy, pandas, h01_spine_roi, h01_spine_geometry,
h01_radius_census. Pure ASCII, LF only.
"""

import os
import time
import traceback

import numpy as np

import h01_radius_census as RC
import h01_spine_geometry as G
import h01_spine_roi as SR

MODULE_VERSION = "h01_spine_batch v1.0"

# Defaults mirror CELL 14 so a batch reproduces the single-spine numbers.
DEFAULTS = {
    "pad_nm": 750.0,
    "bridge_gap_nm": 200.0,
    "bridge_span_window_nm": 200.0,
    "bridge_span_estimator": "equivalent",
    "step_nm": 25.0,
    "tangent_win": 3,
    "ewma_n_back": 3,
    "ewma_decay": 0.5,
    "max_angle_deg": 30.0,
    "mad_k": 5.0,
    "mad_window": 7,
    "min_ext_nm": 50.0,
    "max_offset_nm": 500.0,
    "trim_stations": 3,
    "smooth_shift_nm": None,
    # Fixed Taubin count, matching the cylinder calibration. The budget rule
    # stops on the MAXIMUM vertex shift, an order statistic over the vertex
    # count, so meshes of different size get different amounts of smoothing --
    # phantom and object then do not go through the same operator. Set to None
    # to restore the budget rule.
    "taubin_iterations": 14,
}


class BatchError(RuntimeError):
    """Raised when a batch cannot be set up at all."""


# --------------------------------------------------------------------------- #
def roi_is_cached(out_dir, cell_id, sigma_id):
    """True iff a previously saved ROI can be reloaded without the network."""
    p = SR.roi_paths(out_dir, int(cell_id), int(sigma_id))
    return os.path.isfile(p["npz"]) and os.path.isfile(p["json"])


def _load_cached(out_dir, cell_id, sigma_id):
    """Rebuild the pieces CELL 14 needs from a cached ROI."""
    arrays, meta, spine_nodes, mesh = SR.load_roi(out_dir, int(cell_id),
                                                  int(sigma_id))
    if "spine_mask" not in arrays:
        raise BatchError("cached ROI for sigma %s has no spine_mask"
                         % sigma_id)
    return {"mask": arrays["spine_mask"],
            "bridge_mask": arrays.get("bridge_mask"),
            "meta": meta, "spine_nodes": spine_nodes, "mesh": mesh,
            "_from_cache": True}


def analysis_mesh(res, opts):
    """Smoothed mesh of (spine | bridge). Mirrors CELL 14 exactly."""
    seg_m = res["meta"]["layers"]["seg"]
    bridge = res.get("bridge_mask")
    has_bridge = bridge is not None and bool(np.asarray(bridge).any())
    mask = (np.asarray(res["mask"]) | np.asarray(bridge)) if has_bridge \
        else np.asarray(res["mask"]).copy()
    am = SR.surface_from_mask(mask, seg_m["resolution_nm"], seg_m["lo_vox"])
    if not G.is_closed_mesh(am["faces"]):
        raise BatchError("analysis mesh is not closed")
    if opts.get("taubin_iterations"):
        verts, sm = G.taubin_smooth_to_budget(
            am["verts_nm"], am["faces"], max_shift_nm=np.inf,
            max_iterations=int(opts["taubin_iterations"]))
        # Also record what the budget rule WOULD have allowed on this mesh.
        # The fixed count is pinned from one spine; this builds the population
        # needed to re-pin it with evidence. Costs one extra smoothing pass,
        # negligible against the ROI cutouts.
        if opts.get("record_budget_iterations", True):
            _, sm_b = G.taubin_smooth_to_budget(
                am["verts_nm"], am["faces"],
                resolution_nm=seg_m["resolution_nm"])
            sm["budget_iterations"] = int(sm_b["iterations_used"])
            sm["budget_max_shift_nm"] = float(sm_b["max_vertex_shift_nm"])
            sm["budget_nm"] = float(sm_b["max_shift_budget_nm"])
            sm["exceeds_budget"] = bool(
                sm["max_vertex_shift_nm"] > sm_b["max_shift_budget_nm"])
    else:
        verts, sm = G.taubin_smooth_to_budget(
            am["verts_nm"], am["faces"],
            resolution_nm=seg_m["resolution_nm"],
            max_shift_nm=opts["smooth_shift_nm"])
    return verts, am["faces"], mask, seg_m, sm, has_bridge


def spine_profile(res, spine_nodes, base_node, opts):
    """Centreline -> extension -> tangents -> cross sections -> despike -> core.

    Returns (profile, core, diagnostics). Mirrors CELL 14 so a batch and a
    single-spine run give the same numbers.
    """
    verts_a, faces_a, mask, seg_m, sm, has_bridge = analysis_mesh(res, opts)

    C0, cinfo = G.ordered_centreline(spine_nodes, base_node)
    L0 = cinfo["path_length_nm"]
    C, xinfo = G.extend_centreline(mask, seg_m["lo_vox"],
                                   seg_m["resolution_nm"], C0, at="both",
                                   step_nm=opts["step_nm"], smooth_window=5,
                                   min_extension_nm=opts["min_ext_nm"])
    seg = np.linalg.norm(np.diff(C, axis=0), axis=1)
    span = float(np.linalg.norm(C.max(axis=0) - C.min(axis=0)))
    retracing = bool(seg.sum() > 2.0 * span)

    Cr, s = G.resample_polyline(C, opts["step_nm"])
    T_raw = G.polyline_tangents(Cr, smooth_window=opts["tangent_win"])
    on_skeleton = G.skeleton_station_mask(s, xinfo, L0)
    tc = G.tangent_consistency(T_raw, n_back=opts["ewma_n_back"],
                               decay=opts["ewma_decay"],
                               max_angle_deg=opts["max_angle_deg"],
                               fix_sign=True, check_mask=on_skeleton)

    profile = G.cross_section_profile(verts_a, faces_a, Cr, tc["stabilised"],
                                      arclength_nm=s, select="containing",
                                      max_offset_nm=opts["max_offset_nm"])
    profile = G.despike_profile(profile, rejected=tc["rejected"],
                                mad_k=opts["mad_k"], window=opts["mad_window"])

    a = np.array([r["area_nm2"] for r in profile], dtype=float)
    ok = np.isfinite(a)
    if not ok.any():
        raise BatchError("no station produced a finite cross section")
    first = int(np.argmax(ok))
    last = len(ok) - 1 - int(np.argmax(ok[::-1]))
    t = int(opts["trim_stations"])
    core = profile[first + t: last - t + 1]
    if len(core) < 5:
        raise BatchError("core has only %d stations after trimming %d"
                         % (len(core), t))

    diag = {"n_stations": len(profile), "n_core": len(core),
            "path_length_nm": float(s[-1] - s[0]),
            "centreline_branched": bool(cinfo.get("branched")),
            "retracing_warning": retracing,
            "has_bridge": bool(has_bridge),
            "smoothing_iterations": int(sm.get("iterations_used", 0)),
            "max_vertex_shift_nm": float(sm.get("max_vertex_shift_nm",
                                                float("nan"))),
            "budget_iterations": sm.get("budget_iterations"),
            "budget_nm": sm.get("budget_nm"),
            "exceeds_budget": sm.get("exceeds_budget"),
            "n_rejected_tangents": int(tc.get("n_rejected", 0)),
            "mesh_area_nm2": float(G.mesh_area_nm2(verts_a, faces_a)),
            "mesh_volume_nm3": float(G.mesh_volume_nm3(verts_a, faces_a))}
    return profile, core, diag


# --------------------------------------------------------------------------- #
def census_spines(sigma_ids, nodes, comp, cell_id, out_dir,
                  resolution_nm=(8.0, 8.0, 33.0), opts=None,
                  use_cache=True, extract_kwargs=None,
                  validated_sphere_radius_nm=RC.VALIDATED_SPHERE_RADIUS_NM,
                  verbose=True, on_error="record", progress_every=1):
    """Profile every sigma in `sigma_ids` and pool the equivalent radii.

    Returns (per_spine, pooled, profiles) where `profiles` maps sigma id to its
    CORE profile, so nothing has to be recomputed to re-classify or re-plot.

    use_cache : reload a saved ROI instead of re-fetching. Leave True; the
        network cutouts dominate the runtime and a resumed batch is free.
    on_error : 'record' continues and logs, 'raise' stops at the first failure.
    """
    o = dict(DEFAULTS)
    if opts:
        o.update(opts)
    ek = dict(extract_kwargs or {})

    per_spine, profiles = [], {}
    all_r, all_s, all_id = [], [], []
    t_start = time.time()

    for n_done, sid in enumerate(sigma_ids, start=1):
        sid = int(sid)
        rec = {"sigma_id": sid, "ok": False, "from_cache": False,
               "stage": "start"}
        t0 = time.time()
        try:
            rec["stage"] = "subframe"
            spine_nodes, base_node, sinfo = SR.spine_subframe(
                nodes, comp, sid, include_base=True)
            rec["n_nodes"] = int(len(spine_nodes))

            rec["stage"] = "roi"
            if use_cache and roi_is_cached(out_dir, cell_id, sid):
                res = _load_cached(out_dir, cell_id, sid)
                rec["from_cache"] = True
            else:
                res = SR.extract_spine_roi(
                    nodes, comp, sid, cell_id=cell_id, out_dir=out_dir,
                    pad_nm=o["pad_nm"],
                    bridge_gap_nm=o["bridge_gap_nm"],
                    bridge_span_window_nm=o["bridge_span_window_nm"],
                    bridge_span_estimator=o["bridge_span_estimator"],
                    resolve_components=True, verbose=False, **ek)

            rec["stage"] = "profile"
            profile, core, diag = spine_profile(res, spine_nodes, base_node, o)
            profiles[sid] = core
            rec.update(diag)

            rec["stage"] = "classify"
            rec.update(RC.band_fractions(core, resolution_nm,
                                         validated_sphere_radius_nm))
            s, a, ok = RC.profile_radii(core)
            all_r.append(a[ok])
            all_s.append(s[ok])
            all_id.append(np.full(int(ok.sum()), sid, dtype=np.int64))
            rec["ok"] = True
            rec["stage"] = "done"
        except Exception as exc:                       # noqa: BLE001
            if on_error == "raise":
                raise
            rec["error"] = "%s: %s" % (type(exc).__name__, exc)
            rec["traceback"] = traceback.format_exc(limit=3)
        rec["seconds"] = round(time.time() - t0, 2)
        per_spine.append(rec)

        if verbose and (n_done % max(1, int(progress_every)) == 0
                        or not rec["ok"]):
            if rec["ok"]:
                print("  [%3d/%3d] sigma %-5d %s  %4d core stations  "
                      "min a %6.1f nm  below 2r' %5.1f%%  (%.1fs)"
                      % (n_done, len(sigma_ids), sid,
                         "cache" if rec["from_cache"] else "fetch",
                         rec["n_core"], rec["min_radius_nm"],
                         100 * rec["length_fraction_below_sampling"],
                         rec["seconds"]))
            else:
                print("  [%3d/%3d] sigma %-5d FAILED at %s -- %s"
                      % (n_done, len(sigma_ids), sid, rec["stage"],
                         rec["error"]))

    pooled = {
        "radius_nm": (np.concatenate(all_r) if all_r
                      else np.zeros(0, dtype=float)),
        "s_nm": (np.concatenate(all_s) if all_s else np.zeros(0, dtype=float)),
        "sigma_id": (np.concatenate(all_id) if all_id
                     else np.zeros(0, dtype=np.int64)),
        "n_spines_requested": len(list(sigma_ids)),
        "n_spines_ok": int(sum(1 for r in per_spine if r["ok"])),
        "resolution_nm": tuple(float(v) for v in resolution_nm),
        "step_nm": float(o["step_nm"]),
    }
    if verbose:
        print("\n%d/%d spines profiled in %.1f s (%d from cache)"
              % (pooled["n_spines_ok"], pooled["n_spines_requested"],
                 time.time() - t_start,
                 sum(1 for r in per_spine if r.get("from_cache"))))
    return per_spine, pooled, profiles


def budget_iterations_summary(per_spine):
    """Distribution of the budget-limited iteration count across the batch.

    The fixed count used everywhere is currently pinned from a single spine.
    Once a real population has been profiled, re-pin it from THIS: a single
    fixed count that sits above some spines' budget over-smooths them, and one
    below under-smooths the rest. The conservative choice is the low
    percentile, not the mean.
    """
    v = [r["budget_iterations"] for r in per_spine
         if r.get("ok") and r.get("budget_iterations") is not None]
    if not v:
        return {"n": 0, "note": "no spine recorded a budget iteration count"}
    a = np.asarray(v, dtype=float)
    over = [r["sigma_id"] for r in per_spine if r.get("exceeds_budget")]
    return {"n": int(a.size), "min": int(a.min()), "p10": float(np.percentile(a, 10)),
            "median": float(np.median(a)), "p90": float(np.percentile(a, 90)),
            "max": int(a.max()),
            "n_exceeding_budget_at_current_setting": len(over),
            "sigmas_exceeding": over[:20]}


def failure_report(per_spine):
    """Group the failures by the stage they died at."""
    bad = [r for r in per_spine if not r["ok"]]
    by_stage = {}
    for r in bad:
        by_stage.setdefault(r["stage"], []).append(
            {"sigma_id": r["sigma_id"], "error": r["error"]})
    return {"n_failed": len(bad), "n_total": len(per_spine),
            "by_stage": by_stage}


def save_pooled(path, pooled, per_spine=None):
    """Persist the pooled radii so re-classification needs no recomputation."""
    kw = {"radius_nm": pooled["radius_nm"], "s_nm": pooled["s_nm"],
          "sigma_id": pooled["sigma_id"],
          "resolution_nm": np.asarray(pooled["resolution_nm"]),
          "step_nm": np.asarray([pooled["step_nm"]])}
    if per_spine is not None:
        import json
        kw["per_spine_json"] = np.asarray([json.dumps(
            [{k: v for k, v in r.items() if k != "traceback"}
             for r in per_spine], default=float)])
    np.savez_compressed(path, **kw)
    return path
