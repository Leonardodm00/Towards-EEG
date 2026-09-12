"""Calibrated spine membrane area -> phi^mesh -> F, cell by cell.

WHAT THIS DOES
--------------
For every spine sigma of a cell it builds the SAME smoothed analysis mesh the
cylinder calibration was measured through (h01_spine_batch.analysis_mesh:
marching cubes on (spine | bridge) at 8 x 8 x 33 nm, 14 fixed Taubin
iterations), keeps only the triangles that are real plasma membrane, and
de-biases them with the per-normal calibration g(n):

    A_mesh(sigma) = sum over t in T_mem(sigma) U T_br(sigma) of a_t / g(n_t)   (1)

It then puts A_mesh(sigma) into Stage 1's own phi table in place of the
skeleton frustum area, on exactly the shaft segment Stage 1 attributes that
spine to, and recomputes F with Stage 1's definition. Nothing about the shaft,
the path distances or the attribution rule changes -- only the spine-area
estimator -- so F_mesh and F_skel are directly comparable.

WHICH TRIANGLES ARE MEMBRANE (eq. 2 in the handoff)
---------------------------------------------------
The spine mask is a Voronoi cut of a segment shared with the shaft
(h01_spine_roi.mask_local_to_spine), so its closed mesh also carries an
artificial CUT face where it abuts the shaft. Each triangle is classified by
walking outward along its normal until the ray leaves the meshed object and
reading the first voxel it lands in:

    same-cell segmentation (shaft context, or seg == cell id)  -> CUT     (excluded)
    enclosed background (a hole in the mask)                   -> CAVITY  (excluded)
    anything else (background, another cell)                   -> MEMBRANE
    ray left the cutout before exiting                         -> BOX     (excluded)
    never exited within PROBE_MAX_NM                           -> UNRESOLVED (excluded)

MEMBRANE triangles whose inward walk first meets an interpolated bridge voxel
are relabelled BRIDGE: counted, but reported separately because that surface
is interpolated, not segmented.

FRAME CONVENTION -- MEASURED, NOT ASSUMED
-----------------------------------------
h01_spine_roi.surface_from_mask puts the centre of voxel g at g * res (marching
cubes places integer array indices at integer multiples of the spacing). The
node frame used by mask_local_to_spine puts it at (g + 0.5) * res. The meshes
are therefore offset by -res/2 relative to the node table. Probing masks from
mesh coordinates must use the MESH convention, floor(p / res + 0.5); smoke
test T1 fails if this is wrong.

DEPENDENCIES: numpy, pandas, scipy; h01_area_calibration, h01_spine_batch,
h01_spine_roi imported lazily. Stage 1's spine_density is passed in as `sd`.
Pure ASCII, LF only.
"""

import hashlib
import json
import os
import time
import traceback

import numpy as np
import pandas as pd

MODULE_VERSION = "h01_spine_area_F v1.4"

CLASS_MEMBRANE, CLASS_CUT, CLASS_BRIDGE = 0, 1, 2
CLASS_BOX, CLASS_CAVITY, CLASS_UNRESOLVED = 3, 4, 5
CLASS_NAMES = ("membrane", "cut", "bridge", "box", "cavity", "unresolved")
COUNTED = (CLASS_MEMBRANE, CLASS_BRIDGE)

PROBE_STEP_NM = 4.0       # half an in-plane voxel: cannot skip an 8 nm voxel
PROBE_MAX_NM = 48.0       # > voxel diagonal (34.9) + Taubin shift (~17)
G_MAX_TRIPWIRE = 2.0      # the v6 radial-projection artefact reached 2.76
FOLD_STEP_DEG = 2.0
F_CUTOFF_UM = 60.0        # Eyal et al. 2016 / Benavides-Piccione (KB)
NM2_PER_UM2 = 1.0e6


class SpineAreaError(RuntimeError):
    """Raised when an input fails a gate. A failed gate stops the cell."""


# --------------------------------------------------------------------------- #
# 1. Calibration table                                                         #
# --------------------------------------------------------------------------- #
def load_calibration(path, resolution_nm=(8.0, 8.0, 33.0),
                     step_deg=FOLD_STEP_DEG, g_max_tripwire=G_MAX_TRIPWIRE):
    """Load the cylinder g table and gate it. Returns (table, lookup, report).

    Gates: file present; grid equals the folded 2-degree fundamental domain;
    all finite and positive; max g below the tripwire (the v6 table's
    degenerate-triangle artefact peaked at 2.76, the v7 fix brings the worst
    bin to ~1.23); resolution in the json, if recorded, equals the grid.
    The lookup is built with clip_at_one=False: the cylinder table may go
    below 1 and clipping would bias thin structures.
    """
    import h01_area_calibration as CAL

    if not os.path.isfile(path):
        raise SpineAreaError("calibration table not found: %s" % path)
    t = CAL.load_table(path)
    g = np.asarray(t["g"], dtype=float)
    th = np.asarray(t["theta_deg"], dtype=float)
    ph = np.asarray(t["phi_deg"], dtype=float)
    problems = axis_problems(th, ph, g.shape, step_deg)
    if not np.all(np.isfinite(g)):
        problems.append("table contains non-finite values")
    elif g.min() <= 0:
        problems.append("table contains g <= 0")
    elif g.max() >= g_max_tripwire:
        problems.append("max g = %.3f >= %.2f: this is the radial-projection "
                        "degeneracy signature of the v6 sweep. Rebuild the "
                        "table from the v7 sweep (min_true_area_ratio=0.5)."
                        % (g.max(), g_max_tripwire))
    meta = t.get("meta", {}) or {}
    if "resolution_nm" in meta and not np.allclose(meta["resolution_nm"],
                                                   resolution_nm):
        problems.append("table resolution %s != %s"
                        % (meta["resolution_nm"], list(resolution_nm)))
    if problems:
        raise SpineAreaError("calibration table rejected:\n  "
                             + "\n  ".join(problems))
    with open(path, "rb") as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()[:12]
    report = {"path": path, "sha256_12": sha, "has_json": bool(meta),
              "g_min": float(g.min()), "g_median": float(np.median(g)),
              "g_max": float(g.max()),
              "json": {k: meta[k] for k in sorted(meta)
                       if not isinstance(meta[k], (list, dict))}}
    report["axes"] = {"theta": [float(th[0]), float(th[-1]), len(th)],
                      "phi": [float(ph[0]), float(ph[-1]), len(ph)]}
    look = CAL.make_g_lookup(t, clip_at_one=False)
    # The per-spine normal histogram must be binned on THIS table's axes.
    look.theta_deg, look.phi_deg = th, ph
    return t, look, report


def axis_problems(theta_deg, phi_deg, g_shape, step_deg=FOLD_STEP_DEG,
                  tol=1e-6):
    """Accept any UNIFORM grid spanning the fold domain, theta in [0, 90] and
    phi in [0, <=45]. Two conventions exist and both are valid:
    build_g_table writes phi = 0, 2, ..., 44 (uniform 2 deg, last bin centred
    at 44 collects 43..45); h01_area_calibration.fundamental_grid writes
    phi = linspace(0, 45, 23) (step 45/22). make_g_lookup interpolates on the
    table's own axes either way, so the lookup is consistent with both."""
    out = []
    th, ph = np.asarray(theta_deg, float), np.asarray(phi_deg, float)
    if tuple(g_shape) != (len(th), len(ph)):
        return ["g has shape %s but axes are %d x %d"
                % (tuple(g_shape), len(th), len(ph))]
    for name, ax, hi_lo, hi_hi in (("theta", th, 90.0, 90.0),
                                   ("phi", ph, None, 45.0)):
        d = np.diff(ax)
        if len(ax) < 3 or np.any(d <= 0):
            out.append("%s axis not strictly increasing" % name)
            continue
        if np.ptp(d) > 1e-3 * d.mean():
            out.append("%s axis is not uniform (steps %.4f..%.4f)"
                       % (name, d.min(), d.max()))
        if abs(ax[0]) > tol:
            out.append("%s axis starts at %.3f, not 0" % (name, ax[0]))
        lo = hi_lo if hi_lo is not None else 45.0 - d.mean() - tol
        if not (lo - tol <= ax[-1] <= hi_hi + tol):
            out.append("%s axis ends at %.3f, outside [%.3f, %.1f]"
                       % (name, ax[-1], lo, hi_hi))
        if abs(d.mean() - step_deg) > 0.1 * step_deg:
            out.append("%s step %.3f deg, expected ~%.1f" % (name, d.mean(), step_deg))
    return out


def normal_histogram(normals, areas, theta_deg=None, phi_deg=None):
    """Area-weighted histogram of folded normals, each triangle assigned to
    the NEAREST axis point of the table it will be divided by.

    Stored per spine so a revised g table ON THE SAME AXES can be re-applied
    without re-fetching or re-meshing: A ~= sum(H / g_table), eq. (1) at bin
    centres. Default axes are build_g_table's (theta 0..90, phi 0..44, 2 deg).
    """
    import h01_area_calibration as CAL

    tha = (np.arange(46) * 2.0 if theta_deg is None
           else np.asarray(theta_deg, dtype=float))
    pha = (np.arange(23) * 2.0 if phi_deg is None
           else np.asarray(phi_deg, dtype=float))
    H = np.zeros((len(tha), len(pha)), dtype=float)
    if len(areas) == 0:
        return H
    th, ph = CAL.fold_to_fundamental(np.asarray(normals, dtype=float))
    it = np.searchsorted(0.5 * (tha[1:] + tha[:-1]), th)
    ip = np.searchsorted(0.5 * (pha[1:] + pha[:-1]), ph)
    np.add.at(H, (it, ip), np.asarray(areas, dtype=float))
    return H


def area_from_histogram(H, table):
    """Re-apply a (possibly new) table to a stored normal histogram."""
    g = np.asarray(table["g"], dtype=float)
    if g.shape != np.asarray(H).shape:
        raise SpineAreaError("histogram grid %s != table grid %s"
                             % (np.asarray(H).shape, g.shape))
    return float((np.asarray(H) / g).sum())


# --------------------------------------------------------------------------- #
# 2. Triangle classification                                                   #
# --------------------------------------------------------------------------- #
def mesh_point_to_local_voxel(points_nm, lo_vox, resolution_nm):
    """Cutout index of the voxel containing a MESH-frame point (see header)."""
    res = np.asarray(resolution_nm, dtype=float)
    return (np.floor(np.asarray(points_nm, dtype=float) / res + 0.5)
            .astype(np.int64) - np.asarray(lo_vox, dtype=np.int64))


def triangle_geometry(verts, faces):
    """Centroids, areas and OUTWARD unit normals of a closed mesh.

    Orientation is fixed from the sign of the enclosed volume, computed on
    centred coordinates (at ~1e5 nm offsets the uncentred triple products
    cancel catastrophically against a spine-sized volume).
    """
    v = np.asarray(verts, dtype=float)
    v = v - v.mean(axis=0)
    f = np.asarray(faces, dtype=np.int64)
    p0, p1, p2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    cr = np.cross(p1 - p0, p2 - p0)
    ln = np.linalg.norm(cr, axis=1)
    n = cr / np.maximum(ln[:, None], 1e-30)
    vol = float(np.einsum("ij,ij->i", p0, np.cross(p1, p2)).sum() / 6.0)
    if vol < 0:
        n = -n
    c = (p0 + p1 + p2) / 3.0 + np.asarray(verts, dtype=float).mean(axis=0)
    return c, 0.5 * ln, n, abs(vol)


def classify_triangles(verts, faces, lo_vox, resolution_nm, object_mask,
                       context_mask, bridge_mask=None, cavity_mask=None,
                       step_nm=PROBE_STEP_NM, max_nm=PROBE_MAX_NM):
    """Label every triangle with one of CLASS_NAMES. Returns a dict.

    object_mask  : the voxels that were meshed (spine | bridge).
    context_mask : same-cell segmentation NOT meshed (shaft context). A ray
                   exiting into it marks an artificial cut face.
    bridge_mask  : interpolated voxels; decides BRIDGE vs MEMBRANE.
    cavity_mask  : enclosed background (holes). Exiting into it -> CAVITY.
    """
    U = np.asarray(object_mask, dtype=bool)
    C = np.asarray(context_mask, dtype=bool)
    shape = np.asarray(U.shape, dtype=np.int64)
    cen, area, nrm, vol = triangle_geometry(verts, faces)
    cls = np.full(len(area), CLASS_UNRESOLVED, dtype=np.int8)
    pending = np.arange(len(area))
    k_max = int(np.ceil(float(max_nm) / float(step_nm)))

    for k in range(1, k_max + 1):
        if pending.size == 0:
            break
        q = mesh_point_to_local_voxel(cen[pending] + (k * step_nm) * nrm[pending],
                                      lo_vox, resolution_nm)
        inside = np.all((q >= 0) & (q < shape), axis=1)
        cls[pending[~inside]] = CLASS_BOX
        p_in, q_in = pending[inside], q[inside]
        in_obj = U[q_in[:, 0], q_in[:, 1], q_in[:, 2]]
        p_ex, q_ex = p_in[~in_obj], q_in[~in_obj]
        lab = np.full(len(p_ex), CLASS_MEMBRANE, dtype=np.int8)
        if cavity_mask is not None:
            lab[np.asarray(cavity_mask, bool)[q_ex[:, 0], q_ex[:, 1], q_ex[:, 2]]] = CLASS_CAVITY
        lab[C[q_ex[:, 0], q_ex[:, 1], q_ex[:, 2]]] = CLASS_CUT
        cls[p_ex] = lab
        pending = p_in[in_obj]

    if bridge_mask is not None and np.asarray(bridge_mask).any():
        Bm = np.asarray(bridge_mask, dtype=bool)
        pend = np.nonzero(cls == CLASS_MEMBRANE)[0]
        for k in range(0, k_max + 1):
            if pend.size == 0:
                break
            q = mesh_point_to_local_voxel(cen[pend] - (k * step_nm) * nrm[pend],
                                          lo_vox, resolution_nm)
            ok = np.all((q >= 0) & (q < shape), axis=1)
            pend, q = pend[ok], q[ok]
            hit = U[q[:, 0], q[:, 1], q[:, 2]]
            first, qf = pend[hit], q[hit]
            cls[first[Bm[qf[:, 0], qf[:, 1], qf[:, 2]]]] = CLASS_BRIDGE
            pend = pend[~hit]

    by = {name: float(area[cls == i].sum()) for i, name in enumerate(CLASS_NAMES)}
    return {"cls": cls, "area_nm2": area, "normal": nrm, "centroid": cen,
            "volume_nm3": vol, "area_by_class_nm2": by,
            "n_by_class": {name: int((cls == i).sum())
                           for i, name in enumerate(CLASS_NAMES)}}


def mask_touches_box(mask):
    """True iff any True voxel lies on a face of the cutout (spine clipped)."""
    m = np.asarray(mask, dtype=bool)
    return bool(m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any()
                or m[:, :, 0].any() or m[:, :, -1].any())


# --------------------------------------------------------------------------- #
# 3. One spine                                                                 #
# --------------------------------------------------------------------------- #
def rind_area_um2(centroid_nm, area_nm2, g, resolution_nm, shaft_axis,
                  axial_window_nm=None):
    """Calibrated counted area lying INSIDE the local shaft envelope, eq. (7).

    A spine's own membrane is by construction outside the dendrite envelope,
    so counted triangles at radial distance rho_t <= r_shaft from the shaft
    axis are dendrite surface that the Voronoi cut handed to the spine. The
    shaft frustum of the same segment already counts that surface, so it is
    double-counted in F until removed. Returns (A_rind_um2, diagnostics).

    Centroids arrive in the MESH frame and are shifted by +res/2 into the node
    frame the skeleton axis lives in (see the module header).
    """
    res = np.asarray(resolution_nm, dtype=float)
    u = np.asarray(shaft_axis["dir"], dtype=float)
    u = u / max(float(np.linalg.norm(u)), 1e-30)
    d = (np.asarray(centroid_nm, dtype=float) + 0.5 * res) \
        - np.asarray(shaft_axis["point_nm"], dtype=float)
    along = d @ u
    rho = np.linalg.norm(d - along[:, None] * u[None, :], axis=1)
    inside = rho <= float(shaft_axis["r_nm"])
    if axial_window_nm is not None:
        inside = inside & (np.abs(along) <= float(axial_window_nm))
    a = np.asarray(area_nm2, dtype=float)
    A = float((a[inside] / np.asarray(g, dtype=float)[inside]).sum()) / NM2_PER_UM2
    return A, {"n_rind_faces": int(inside.sum()),
               "rind_axial_max_nm": (float(np.abs(along[inside]).max())
                                     if inside.any() else 0.0),
               "rho_min_nm": float(rho.min()) if len(rho) else np.nan,
               "r_shaft_nm": float(shaft_axis["r_nm"])}


def measure_spine_area(roi, g_lookup, opts, cell_id=None, return_detail=False,
                       shaft_axis=None, axial_window_nm=None):
    """Eq. (1) on one ROI. Returns (record, H_counted), plus a detail dict
    (smoothed mesh, per-triangle class, frame) when return_detail=True --
    what the per-spine figures need, so nothing is recomputed to plot it.

    roi : dict with mask, bridge_mask (or None), shaft_context_mask, meta
          (meta['layers']['seg'] carries resolution_nm and lo_vox) and,
          optionally, seg (the raw c3 cutout, used to treat any voxel of the
          cell's own id as context -- matters only for foreign spines).
    """
    from scipy import ndimage
    import h01_spine_batch as B

    verts, faces, U, seg_m, sm, has_bridge = B.analysis_mesh(roi, opts)
    res, lo = seg_m["resolution_nm"], seg_m["lo_vox"]
    S = np.asarray(roi["mask"], dtype=bool)
    ctx = roi.get("shaft_context_mask")
    C = np.zeros_like(S) if ctx is None else np.asarray(ctx, dtype=bool)
    if roi.get("seg") is not None and cell_id is not None:
        C = C | ((np.asarray(roi["seg"]) == np.uint64(cell_id)) & ~U)
    Bm = (np.asarray(roi["bridge_mask"], dtype=bool) & ~S) if has_bridge else None
    cav = ndimage.binary_fill_holes(U) & ~U

    cl = classify_triangles(verts, faces, lo, res, U, C, Bm, cav)
    keep = np.isin(cl["cls"], COUNTED)
    a = cl["area_nm2"][keep]
    g = np.asarray(g_lookup(cl["normal"][keep]), dtype=float).ravel()
    br = cl["cls"][keep] == CLASS_BRIDGE
    by = cl["area_by_class_nm2"]
    tot = max(sum(by.values()), 1e-30)
    rec = {
        "n_faces": int(len(cl["cls"])),
        "A_mesh_raw_um2": float(a.sum()) / NM2_PER_UM2,
        "A_mesh_um2": float((a / g).sum()) / NM2_PER_UM2,
        "A_bridge_um2": float((a[br] / g[br]).sum()) / NM2_PER_UM2,
        "g_aw_mean": float((a * g).sum() / max(a.sum(), 1e-30)),
        "volume_um3": cl["volume_nm3"] / 1e9,
        "has_bridge": bool(has_bridge),
        "clipped": mask_touches_box(S),
        "n_spine_voxels": int(S.sum()),
        "taubin_iterations": int(sm.get("iterations_used", -1)),
        "budget_iterations": sm.get("budget_iterations"),
        "max_vertex_shift_nm": float(sm.get("max_vertex_shift_nm", np.nan)),
        "exceeds_budget": sm.get("exceeds_budget"),
    }
    for name in CLASS_NAMES:
        rec["A_%s_raw_um2" % name] = by[name] / NM2_PER_UM2
        rec["frac_%s" % name] = by[name] / tot
    if shaft_axis is None:
        rec.update({"A_rind_um2": np.nan, "frac_rind": np.nan,
                    "A_mesh_norind_um2": np.nan, "n_rind_faces": -1,
                    "rind_axial_max_nm": np.nan, "r_shaft_nm": np.nan})
    else:
        A_r, diag = rind_area_um2(cl["centroid"][keep], a, g, res, shaft_axis,
                                  axial_window_nm)
        rec.update(diag)
        rec["A_rind_um2"] = A_r
        rec["frac_rind"] = A_r / max(rec["A_mesh_um2"], 1e-30)
        rec["A_mesh_norind_um2"] = rec["A_mesh_um2"] - A_r
    H = normal_histogram(cl["normal"][keep], a,
                         getattr(g_lookup, "theta_deg", None),
                         getattr(g_lookup, "phi_deg", None))
    if return_detail:
        return rec, H, {"verts_nm": verts, "faces": faces, "cls": cl["cls"],
                        "centroid_nm": cl["centroid"], "lo_vox": lo,
                        "resolution_nm": res, "object_mask": U}
    return rec, H


def memoized_reader_factory(base_factory=None):
    """One CloudVolume per layer for the whole run, so its LRU cache is shared
    across spines. h01_spine_roi.fetch_roi otherwise builds a new volume per
    call and every neighbouring spine re-downloads the same chunks."""
    import h01_spine_roi as SR
    base = base_factory or SR.make_cloudvolume_reader
    cache = {}

    def factory(cloudpath, mip=0, **kw):
        key = (cloudpath, int(mip))
        if key not in cache:
            cache[key] = base(cloudpath, mip=mip, **kw)
        return cache[key]
    factory.cache = cache
    return factory


def get_spine_roi(nodes, comp, sigma_id, cell_id, roi_dir=None, pad_nm=500.0,
                  reader_factory=None, max_bytes=1 << 30, use_cache=True):
    """Cache-first ROI for one spine, seg layer only. Returns a roi dict.

    A CELL 12 ROI already on Drive is reused read-only. Otherwise the seg
    cutout is fetched with save=False: the batch persists numbers, not
    arrays. If the spine mask touches the cutout face it is re-fetched once
    at twice the padding.
    """
    import h01_spine_batch as B
    import h01_spine_roi as SR

    if use_cache and roi_dir:
        p = SR.roi_paths(roi_dir, int(cell_id), int(sigma_id))
        if os.path.isfile(p["npz"]) and os.path.isfile(p["json"]):
            arrays, meta, _, _ = SR.load_roi(roi_dir, int(cell_id), int(sigma_id))
            if "spine_mask" in arrays and "shaft_context_mask" in arrays \
                    and not mask_touches_box(arrays["spine_mask"]):
                return {"mask": arrays["spine_mask"],
                        "bridge_mask": arrays.get("bridge_mask"),
                        "shaft_context_mask": arrays["shaft_context_mask"],
                        "seg": arrays.get("seg"), "meta": meta,
                        "spine_segids": meta.get("spine_segids"),
                        "from_cache": True, "pad_nm": None}

    d = B.DEFAULTS
    kw = {"layers": {"seg": SR.LAYERS["seg"]}, "bridge_gap_nm": d["bridge_gap_nm"],
          "bridge_span_window_nm": d["bridge_span_window_nm"],
          "bridge_span_estimator": d["bridge_span_estimator"],
          "resolve_components": True, "build_mesh": True, "save": False,
          "verbose": False, "max_bytes": int(max_bytes)}
    if reader_factory is not None:
        kw["reader_factory"] = reader_factory
    for pad in (float(pad_nm), 2.0 * float(pad_nm)):
        r = SR.extract_spine_roi(nodes, comp, int(sigma_id), cell_id=int(cell_id),
                                 out_dir=roi_dir or ".", pad_nm=pad, **kw)
        if "mask" not in r:
            raise SpineAreaError("no spine voxels (segids %s, %s node(s) on "
                                 "background)" % (r.get("spine_segids"),
                                                  r["meta"].get("n_nodes_on_background")))
        if not mask_touches_box(r["mask"]):
            break
    return {"mask": r["mask"], "bridge_mask": r.get("bridge_mask"),
            "shaft_context_mask": r["shaft_context_mask"],
            "seg": r["arrays"].get("seg"), "meta": r["meta"],
            "spine_segids": r.get("spine_segids"), "from_cache": False,
            "pad_nm": pad}


# --------------------------------------------------------------------------- #
# 4. Many spines, resumable                                                    #
# --------------------------------------------------------------------------- #
def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def load_ledger(path):
    """Returns ({sigma_id: record}, {sigma_id: H}). Empty if absent."""
    if not os.path.isfile(path):
        return {}, {}
    with np.load(path, allow_pickle=False) as z:
        recs = json.loads(str(z["records_json"][0]))
        hs = z["H_sigma"].astype(np.int64)
        H = z["H"]
    return ({int(r["sigma_id"]): r for r in recs},
            {int(s): H[i].astype(float) for i, s in enumerate(hs)})


def save_ledger(path, recs, H):
    """Atomic: write a temp file, then rename over the ledger."""
    ids = sorted(H)
    Harr = (np.stack([H[s] for s in ids]).astype(np.float32) if ids
            else np.zeros((0, 1, 1), np.float32))
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, records_json=np.asarray(
        [json.dumps([recs[s] for s in sorted(recs)], default=_jsonable)]),
        H_sigma=np.asarray(ids, dtype=np.int64), H=Harr,
        module_version=np.asarray([MODULE_VERSION]))
    os.replace(tmp, path)
    return path


def measure_all_spines(sigma_ids, roi_fn, g_lookup, ledger_path, opts=None,
                       cell_id=None, checkpoint_every=25, verbose=True,
                       progress_every=25, on_success=None, shaft_axes=None,
                       axial_window_nm=None, require_keys=()):
    """Measure every sigma not already in the ledger. Failures are recorded
    with the stage they died at and never stop the loop.

    on_success : optional callable(sigma_id, roi, record, detail), called while
        the ROI is still in memory (e.g. to draw the per-spine figure). An
        exception inside it is recorded as `figure_error` and does NOT fail
        the measurement. Spines already in the ledger are not revisited.
    """
    import h01_spine_batch as B

    o = dict(B.DEFAULTS)
    o.update(opts or {})
    recs, H = load_ledger(ledger_path)
    # A record written by an older module version may lack a newly added
    # column; re-measure those rather than emitting a half-filled table.
    stale = [s for s, r in recs.items()
             if r.get("ok") and any(k not in r for k in require_keys)]
    for s in stale:
        recs.pop(s, None)
        H.pop(s, None)
    if stale and verbose:
        print("  %d record(s) lack %s -- re-measuring them"
              % (len(stale), ", ".join(require_keys)))
    todo = [int(s) for s in sigma_ids if int(s) not in recs]
    if verbose:
        print("  %d spine(s) requested, %d already in the ledger, %d to do"
              % (len(sigma_ids), len(sigma_ids) - len(todo), len(todo)))
    t_start = time.time()
    for i, sid in enumerate(todo, start=1):
        rec = {"sigma_id": sid, "ok": False, "stage": "roi"}
        t0 = time.time()
        try:
            roi = roi_fn(sid)
            rec["from_cache"] = bool(roi.get("from_cache"))
            rec["pad_nm"] = roi.get("pad_nm")
            segids = [int(v) for v in (roi.get("spine_segids") or [])]
            rec["spine_segids"] = segids
            rec["on_cell_segid"] = (cell_id is not None and int(cell_id) in segids)
            rec["stage"] = "measure"
            r, h, detail = measure_spine_area(
                roi, g_lookup, o, cell_id=cell_id, return_detail=True,
                shaft_axis=(shaft_axes or {}).get(sid),
                axial_window_nm=axial_window_nm)
            rec.update(r)
            H[sid] = h
            rec["ok"], rec["stage"] = True, "done"
            if on_success is not None:
                try:
                    fig_out = on_success(sid, roi, rec, detail)
                    if fig_out:
                        rec["figure"] = str(fig_out)
                except Exception as fexc:               # noqa: BLE001
                    rec["figure_error"] = "%s: %s" % (type(fexc).__name__, fexc)
        except Exception as exc:                        # noqa: BLE001
            rec["error"] = "%s: %s" % (type(exc).__name__, exc)
            rec["traceback"] = traceback.format_exc(limit=2)
        rec["seconds"] = round(time.time() - t0, 2)
        recs[sid] = rec
        if i % max(1, int(checkpoint_every)) == 0:
            save_ledger(ledger_path, recs, H)
        if verbose and (i % max(1, int(progress_every)) == 0 or not rec["ok"]):
            el = time.time() - t_start
            print("  [%5d/%5d] sigma %-6d %s  (%.1f s/spine, ~%.1f h left)"
                  % (i, len(todo), sid,
                     ("A=%.3f um2 cut %.1f%%" % (rec["A_mesh_um2"],
                                                  100 * rec["frac_cut"]))
                     if rec["ok"] else "FAILED at %s: %s" % (rec["stage"],
                                                              rec["error"][:80]),
                     el / i, el / i * (len(todo) - i) / 3600.0))
    save_ledger(ledger_path, recs, H)
    return recs, H


# --------------------------------------------------------------------------- #
# 5. Skeleton side: Stage 1's own attribution, per spine                       #
# --------------------------------------------------------------------------- #
def skeleton_spine_table(sd, labelled, nodes, comp):
    """Per-sigma skeleton area and the phi row Stage 1 attributes it to.

    Replicates spine_density._attribute_spine_area (v1.1.0 source, the rule
    documented in TEEG_18 sec 3.1) but keeps one entry PER SPINE instead of
    summing, using sd's own _prepare_nodes / _frustum_lateral_area /
    _segment_length_um so radii, unit conversion and the radius fallback are
    Stage 1's. attribution_gate() then proves the replica reproduces
    build_phi's spine_area_um2 row by row -- if sd has changed, the gate
    fails instead of F being silently wrong.
    """
    node, children, _ = sd._prepare_nodes(labelled, sd.SHAFT_REGEX,
                                          sd.SPINE_LABELS,
                                          sd.DEFAULT_RADIUS_NM, "nm")
    ids = nodes["id"].to_numpy(dtype=np.int64)
    par = nodes["p"].to_numpy(dtype=np.int64)
    xyz = nodes[["x", "y", "z"]].to_numpy(dtype=float)
    rad = (nodes["r"].to_numpy(dtype=float) if "r" in nodes.columns
           else np.full(len(nodes), np.nan))
    pos = {int(v): i for i, v in enumerate(ids)}
    comp = np.asarray(comp)
    rows = []
    for sid in np.unique(comp[comp >= 0]):
        sel = comp == sid
        members = ids[sel]
        mset = set(int(v) for v in members)
        roots = [int(m) for m, p in zip(members, par[sel]) if int(p) not in mset]
        rho = roots[0]
        sub, stack = [rho], [rho]
        while stack:
            cur = stack.pop()
            for ch in children.get(cur, ()):
                if node[ch]["is_spine"]:
                    sub.append(ch)
                    stack.append(ch)
        area = 0.0
        for s in sub:
            sp = node[s]["p"]
            area += sd._frustum_lateral_area(node[sp]["r"], node[s]["r"],
                                             sd._segment_length_um(node, sp, s))
        base = node[rho]["p"]
        key = (-1, -1)
        if base in node and node[base]["p"] in node:
            key = (int(node[base]["p"]), int(base))
        elif base in node:
            sh = [c for c in children.get(base, ()) if node[c]["is_shaft"]]
            if sh:
                key = (int(base), int(sh[0]))
        b = xyz[pos[int(base)]] if int(base) in pos else np.full(3, np.nan)
        r_b = rad[pos[int(base)]] if int(base) in pos else np.nan
        mx = xyz[sel]
        tip = float(np.max(np.linalg.norm(mx - b, axis=1))) if len(mx) else np.nan
        # The FIRST frustum of the subtree sum, base -> root. Its wide end is
        # the SHAFT radius, so it is mostly dendrite-sized surface credited to
        # the spine (junction bias 1). Reported so kappa can be formed with
        # and without it; A_skel_um2 itself is left exactly as Stage 1 has it.
        a_base = (sd._frustum_lateral_area(node[base]["r"], node[rho]["r"],
                                           sd._segment_length_um(node, base, rho))
                  if base in node else np.nan)
        # Local shaft axis: the attributed segment (seg_from -> seg_to), which
        # has the base node as one endpoint, so it passes through the shaft
        # centreline there. Used to test whether counted mesh area lies inside
        # the dendrite envelope (junction bias 2).
        # r_shaft is only a shaft radius when the base IS a shaft node. Off a
        # soma root it is the SOMA radius (microns), and a rind cylinder that
        # wide would swallow the whole spine, so the axis is withheld instead.
        on_shaft = base in node and node[base]["is_shaft"]
        u = np.full(3, np.nan)
        if on_shaft and key[0] >= 0 and key[0] in pos and key[1] in pos:
            v = xyz[pos[key[1]]] - xyz[pos[key[0]]]
            n_v = float(np.linalg.norm(v))
            if n_v > 0:
                u = v / n_v
        rows.append({"sigma_id": int(sid), "root_id": rho, "base_id": int(base),
                     "n_nodes": int(len(members)), "n_roots": len(roots),
                     "subtree_matches_comp": set(sub) == mset,
                     "A_skel_um2": float(area), "seg_from": key[0],
                     "seg_to": key[1], "base_x": b[0], "base_y": b[1],
                     "base_z": b[2],
                     "L_skel_nm": _path_length_from_root(rho, members, par[sel],
                                                         mx),
                     "tip_dist_nm": tip, "r_base_nm": float(r_b),
                     "A_skel_base_um2": float(a_base),
                     "A_skel_nobase_um2": float(area) - float(a_base),
                     "axis_ux": u[0], "axis_uy": u[1], "axis_uz": u[2],
                     "r_shaft_nm": float(r_b) if on_shaft else np.nan,
                     # Undefined off a shaft base: the virtual soma root's
                     # radius does not describe a membrane (s0_ingest), so
                     # tip - r_b there is meaningless and would be negative.
                     "protrusion_nm": (tip - float(r_b))
                     if (base in node and node[base]["is_shaft"]) else np.nan})
    # An empty cell (no spines, or every spine demoted by the size filter)
    # must still carry the schema: downstream code selects columns by name and
    # a bare empty frame fails with an opaque KeyError instead.
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=SPINE_TABLE_COLUMNS)


SPINE_TABLE_COLUMNS = (
    "sigma_id", "root_id", "base_id", "n_nodes", "n_roots",
    "subtree_matches_comp", "A_skel_um2", "seg_from", "seg_to",
    "base_x", "base_y", "base_z", "L_skel_nm", "tip_dist_nm", "r_base_nm",
    "A_skel_base_um2", "A_skel_nobase_um2", "axis_ux", "axis_uy", "axis_uz",
    "r_shaft_nm", "protrusion_nm")


def _path_length_from_root(root, members, parents, xyz):
    """Longest skeleton path (nm) from the spine root to any node of sigma,
    along parent links inside sigma. The base segment is NOT included."""
    idx = {int(m): i for i, m in enumerate(members)}
    dist = {int(root): 0.0}
    pending = [int(m) for m in members if int(m) != int(root)]
    while pending:
        nxt = []
        for m in pending:
            p = int(parents[idx[m]])
            if p in dist:
                dist[m] = dist[p] + float(np.linalg.norm(xyz[idx[m]] - xyz[idx[p]]))
            else:
                nxt.append(m)
        if len(nxt) == len(pending):        # disconnected remainder: ignore
            break
        pending = nxt
    return float(max(dist.values()))


LENGTH_METRICS = ("protrusion_nm", "L_skel_nm", "tip_dist_nm", "n_nodes")


def shaft_axes_from_table(sk):
    """{sigma_id: {point_nm, dir, r_nm}} for every spine with a shaft axis.

    Spines with no attributed shaft segment (seg_from < 0, e.g. a spine on the
    soma root) are omitted; their rind is then NaN and they are excluded from
    the rind-corrected kappa rather than silently counted as rind-free.
    """
    out = {}
    for r in sk.itertuples(index=False):
        u = np.array([r.axis_ux, r.axis_uy, r.axis_uz], dtype=float)
        if not np.all(np.isfinite(u)) or not np.isfinite(r.r_shaft_nm):
            continue
        out[int(r.sigma_id)] = {
            "point_nm": np.array([r.base_x, r.base_y, r.base_z], dtype=float),
            "dir": u, "r_nm": float(r.r_shaft_nm)}
    return out


def threshold_report(sk, metric="protrusion_nm",
                     thresholds=(100.0, 200.0, 300.0, 400.0, 500.0, 600.0)):
    """What each candidate minimum would remove: count and share of the
    SKELETON spine area. Nothing is removed by calling this."""
    if metric not in LENGTH_METRICS:
        raise SpineAreaError("metric must be one of %s" % (LENGTH_METRICS,))
    v = sk[metric].to_numpy(dtype=float)
    a = sk["A_skel_um2"].to_numpy(dtype=float)
    rows = []
    for t in thresholds:
        rm = v < float(t)
        rows.append({"metric": metric, "min": float(t), "n_removed": int(rm.sum()),
                     "frac_spines": float(rm.mean()) if len(v) else np.nan,
                     "frac_skel_area": float(a[rm].sum() / max(a.sum(), 1e-30))})
    out = pd.DataFrame(rows)
    out.attrs["n_metric_nan"] = int(np.isnan(v).sum())
    return out


def short_spine_ids(sk, metric="protrusion_nm", min_value=None):
    """Sigma ids whose `metric` is below `min_value`. None disables the filter.
    A spine whose metric is NaN (e.g. no base radius) is KEPT, and counted by
    threshold_report's attrs['n_metric_nan']."""
    if min_value is None:
        return []
    if metric not in LENGTH_METRICS:
        raise SpineAreaError("metric must be one of %s" % (LENGTH_METRICS,))
    v = sk[metric].to_numpy(dtype=float)
    return [int(s) for s in sk["sigma_id"].to_numpy()[v < float(min_value)]]


def demote_spines(labelled, nodes, comp, sk, sigma_ids, shaft_regex,
                  fallback_label="dendrite"):
    """Relabel whole spines as shaft, BEFORE phi is built, so F_skel and
    F_mesh share one partition (the shaft_continuation mechanism).

    Each demoted node takes its spine's BASE node annotated_type when that is
    a shaft label (apical stays apical), else `fallback_label`. The demoted
    skeleton becomes a short shaft side branch in phi: its frustum area moves
    from the spine column to the shaft column rather than vanishing.

    comp is NOT renumbered -- demoted sigmas become -1 and every other sigma
    keeps its id, so ledgers written before or after a threshold change still
    refer to the same spines. Returns (labelled2, nodes2, comp2, provenance).
    """
    import re

    if not re.search(shaft_regex, str(fallback_label).lower()):
        raise SpineAreaError("fallback_label %r is not a shaft label under %r"
                             % (fallback_label, shaft_regex))
    comp = np.asarray(comp)
    ids = set(int(s) for s in sigma_ids)
    lab2, nodes2, comp2 = labelled.copy(), nodes.copy(), comp.copy()
    if not ids:
        return lab2, nodes2, comp2, {"n_spines_demoted": 0, "n_nodes_demoted": 0,
                                     "A_skel_demoted_um2": 0.0}
    col = lab2.columns.get_loc("annotated_type")
    id_pos = {int(v): i for i, v in enumerate(lab2["id"].to_numpy(dtype=np.int64))}
    base_of = dict(zip(sk["sigma_id"].astype(int), sk["base_id"].astype(int)))
    n_nodes = 0
    for sid in sorted(ids):
        rows = np.nonzero(comp == sid)[0]
        bpos = id_pos.get(base_of.get(sid, -1))
        btype = (str(lab2.iloc[bpos, col]).lower() if bpos is not None else "")
        new = btype if re.search(shaft_regex, btype) else fallback_label
        lab2.iloc[rows, col] = new
        if "spine_label" in nodes2.columns:
            nodes2.iloc[rows, nodes2.columns.get_loc("spine_label")] = new
        comp2[rows] = -1
        n_nodes += len(rows)
    a = sk.loc[sk["sigma_id"].isin(ids), "A_skel_um2"].sum()
    return lab2, nodes2, comp2, {"n_spines_demoted": len(ids),
                                 "n_nodes_demoted": int(n_nodes),
                                 "A_skel_demoted_um2": float(a)}


def attribution_gate(phi_skel, sk, rtol=1e-9, atol_um2=1e-9):
    """Row-by-row: sum of per-spine skeleton areas == build_phi's column."""
    keys = list(zip(phi_skel["node_from"].astype(np.int64),
                    phi_skel["node_to"].astype(np.int64)))
    mapped = sk[sk["seg_from"] >= 0]
    sums = mapped.groupby(["seg_from", "seg_to"])["A_skel_um2"].sum()
    got = np.array([sums.get(k, 0.0) for k in keys])
    want = phi_skel["spine_area_um2"].to_numpy(dtype=float)
    diff = np.abs(got - want)
    stray = sorted(set(sums.index) - set(keys))
    bad = diff > (atol_um2 + rtol * np.abs(want))
    return {"pass": bool(not bad.any() and not stray
                         and bool(sk["subtree_matches_comp"].all())),
            "max_abs_diff_um2": float(diff.max()) if len(diff) else 0.0,
            "n_rows_mismatched": int(bad.sum()),
            "n_spines_on_rows_not_in_phi": len(stray),
            "n_spines_unmapped": int((sk["seg_from"] < 0).sum()),
            "n_subtree_mismatch": int((~sk["subtree_matches_comp"]).sum()),
            "total_skel_um2_table": float(sk["A_skel_um2"].sum()),
            "total_skel_um2_phi": float(want.sum())}


def choose_sigmas(sk, subset=None, n_strata=5, seed=0, order_bin_nm=2000.0):
    """All sigmas, or a size-stratified random subset; returned in spatial
    order (base binned on a 2 um grid) so consecutive cutouts share chunks."""
    df = sk.copy()
    if subset is not None and int(subset) < len(df):
        rng = np.random.default_rng(int(seed))
        n_strata = int(max(1, min(int(n_strata), int(subset), len(df))))
        q = pd.qcut(df["A_skel_um2"].rank(method="first"), n_strata,
                    labels=False)
        per = int(np.ceil(int(subset) / float(n_strata)))
        pick = []
        for k in range(n_strata):
            pool = df.index[q == k].to_numpy()
            pick.extend(rng.choice(pool, size=min(per, len(pool)),
                                   replace=False).tolist())
        df = df.loc[sorted(pick)[:int(subset)]]
    b = np.floor(df[["base_x", "base_y", "base_z"]].to_numpy(float)
                 / order_bin_nm)
    order = np.lexsort((b[:, 2], b[:, 1], b[:, 0]))
    return [int(v) for v in df["sigma_id"].to_numpy()[order]]


# --------------------------------------------------------------------------- #
# 6. phi^mesh, F, kappa                                                        #
# --------------------------------------------------------------------------- #
def phi_with_spine_areas(phi_skel, sk, area_col):
    """Stage 1's phi table with spine_area_um2 replaced, eq. (4).

    Same rows, same shaft areas, same path distances. phi_um and psi are
    recomputed with build_phi's own formulas (phi = A/L, psi = phi/(pi*delta)
    with delta = shaft_diam_um = r_a + r_b)."""
    keys = list(zip(phi_skel["node_from"].astype(np.int64),
                    phi_skel["node_to"].astype(np.int64)))
    m = sk[sk["seg_from"] >= 0]
    sums = m.groupby(["seg_from", "seg_to"])[area_col].sum()
    new = np.array([float(sums.get(k, 0.0)) for k in keys])
    out = phi_skel.copy()
    out["spine_area_skel_um2"] = phi_skel["spine_area_um2"].to_numpy(dtype=float)
    out["spine_area_um2"] = new
    L = out["seg_len_um"].to_numpy(dtype=float)
    D = out["shaft_diam_um"].to_numpy(dtype=float)
    out["phi_um"] = np.where(L > 0, new / np.where(L > 0, L, 1.0), 0.0)
    out["psi"] = np.where(D > 0, out["phi_um"] / (np.pi * np.where(D > 0, D, 1.0)), 0.0)
    return out


def cell_F(phi, cutoff_um=F_CUTOFF_UM, by="d_from_um"):
    """F = 1 + sum A_spine / sum A_shaft over rows with `by` >= cutoff, eq. (5)."""
    sel = phi[by].to_numpy(dtype=float) >= float(cutoff_um)
    a_sh = float(phi["shaft_area_um2"].to_numpy(dtype=float)[sel].sum())
    a_sp = float(phi["spine_area_um2"].to_numpy(dtype=float)[sel].sum())
    return {"F": (1.0 + a_sp / a_sh) if a_sh > 0 else float("nan"),
            "A_shaft_um2": a_sh, "A_spine_um2": a_sp,
            "n_segments": int(sel.sum())}


def kappa_function(sk, n_bins=6, min_per_bin=5, num_col="A_mesh_um2",
                   den_col="A_skel_um2", measured_col="measured"):
    """kappa(A_skel) as a ratio of sums in quantile bins of A_skel, eq. (6).

    Ratio of sums, not mean of ratios: the per-spine ratio has a heavy right
    tail from tiny denominators, the pooled ratio does not. num_col/den_col
    select the junction convention (raw, rind-removed, base-removed)."""
    if num_col not in sk.columns or den_col not in sk.columns:
        return pd.DataFrame(columns=["lo_um2", "hi_um2", "n", "kappa",
                                     "kappa_median", "kappa_p25", "kappa_p75"])
    d = sk[sk[measured_col].astype("boolean").fillna(False).to_numpy(dtype=bool)
           & (sk[den_col] > 0) & np.isfinite(sk[num_col])]
    nb = int(max(1, min(int(n_bins), len(d) // int(min_per_bin))))
    if len(d) == 0:
        return pd.DataFrame(columns=["lo_um2", "hi_um2", "n", "kappa",
                                     "kappa_median", "kappa_p25", "kappa_p75"])
    # Interior quantiles only, then the open ends. np.unique on the FULL set
    # collapses to one value when every A_skel ties (or n == 1), which gave an
    # empty table and silently turned kappa-fill into skeleton-fill.
    interior = np.unique(np.quantile(d[den_col],
                                     np.linspace(0, 1, nb + 1))[1:-1])
    edges = np.concatenate([[0.0], interior, [np.inf]])
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = d[(d[den_col] >= lo) & (d[den_col] < hi)]
        if len(b) == 0:
            continue
        k = b[num_col] / b[den_col]
        rows.append({"lo_um2": lo, "hi_um2": hi, "n": int(len(b)),
                     "kappa": float(b[num_col].sum() / b[den_col].sum()),
                     "kappa_median": float(k.median()),
                     "kappa_p25": float(k.quantile(0.25)),
                     "kappa_p75": float(k.quantile(0.75))})
    return pd.DataFrame(rows)


def kappa_apply(ktab, a_skel):
    """Piecewise-constant kappa_hat(A_skel); NaN if the table is empty."""
    a = np.asarray(a_skel, dtype=float)
    if len(ktab) == 0:
        return np.full(a.shape, np.nan)
    hi = ktab["hi_um2"].to_numpy(dtype=float)
    idx = np.clip(np.searchsorted(hi, a, side="right"), 0, len(hi) - 1)
    return ktab["kappa"].to_numpy(dtype=float)[idx]


def assemble_cell(sd, labelled, nodes, comp, recs, cell_id,
                  cutoff_um=F_CUTOFF_UM, sk=None, min_per_bin=5):
    """phi_skel -> gate -> per-spine join -> phi_mesh -> F. Returns a dict.

    Unmeasured spines (pilot subset, failures, clipped) are filled two ways
    and both are reported: with their skeleton area (biased by 1/kappa) and
    with kappa_hat(A_skel) * A_skel. With full coverage the two coincide.
    """
    phi_skel = sd.build_phi(labelled, nid=cell_id, input_units="nm")
    sk = skeleton_spine_table(sd, labelled, nodes, comp) if sk is None else sk.copy()
    gate = attribution_gate(phi_skel, sk)
    if not gate["pass"]:
        raise SpineAreaError("attribution gate FAILED for cell %s: %s"
                             % (cell_id, json.dumps(gate)))
    m = pd.DataFrame([r for r in recs.values()]) if recs else pd.DataFrame(
        columns=["sigma_id", "ok"])
    cols = [c for c in m.columns if c not in ("traceback",)]
    sk = sk.merge(m[cols], on="sigma_id", how="left")
    good = (sk["ok"].astype("boolean").fillna(False).to_numpy(dtype=bool)
            & ~sk.get("clipped", pd.Series(False, index=sk.index)).astype(
                "boolean").fillna(False).to_numpy(dtype=bool))
    sk["measured"] = good
    sk["kappa"] = np.where(good, sk.get("A_mesh_um2", np.nan)
                           / sk["A_skel_um2"].where(sk["A_skel_um2"] > 0), np.nan)
    ktab = kappa_function(sk, min_per_bin=min_per_bin)
    k_hat = kappa_apply(ktab, sk["A_skel_um2"])
    # Junction-corrected track: numerator loses the Voronoi rind (bias 2).
    # Spines with no shaft axis have NaN rind and are excluded from it.
    has_rind = np.isfinite(sk.get("A_mesh_norind_um2",
                                  pd.Series(np.nan, index=sk.index))).to_numpy(bool)
    sk["measured_norind"] = good & has_rind
    ktab_nr = kappa_function(sk, min_per_bin=min_per_bin,
                             num_col="A_mesh_norind_um2",
                             measured_col="measured_norind")
    k_hat_nr = kappa_apply(ktab_nr, sk["A_skel_um2"])
    a_nr = sk.get("A_mesh_norind_um2", pd.Series(np.nan, index=sk.index)).to_numpy(float)
    a_mesh = sk.get("A_mesh_um2", pd.Series(np.nan, index=sk.index)).to_numpy(float)
    a_raw = sk.get("A_mesh_raw_um2", pd.Series(np.nan, index=sk.index)).to_numpy(float)
    a_sk = sk["A_skel_um2"].to_numpy(float)
    sk["A_used_skelfill_um2"] = np.where(good, a_mesh, a_sk)
    sk["A_used_kappafill_um2"] = np.where(good, a_mesh,
                                          np.where(np.isfinite(k_hat), k_hat * a_sk, a_sk))
    sk["A_used_raw_um2"] = np.where(good, a_raw, a_sk)
    sk["A_used_norind_um2"] = np.where(sk["measured_norind"], a_nr,
                                       np.where(np.isfinite(k_hat_nr),
                                                k_hat_nr * a_sk, a_sk))

    phis = {"skel": phi_skel,
            "mesh": phi_with_spine_areas(phi_skel, sk, "A_used_kappafill_um2"),
            "mesh_skelfill": phi_with_spine_areas(phi_skel, sk, "A_used_skelfill_um2"),
            "mesh_uncalibrated": phi_with_spine_areas(phi_skel, sk, "A_used_raw_um2"),
            "mesh_norind": phi_with_spine_areas(phi_skel, sk, "A_used_norind_um2")}
    F = {}
    for name, ph in phis.items():
        F["F_whole_" + name] = cell_F(ph, 0.0)["F"]
        F["F_lit_" + name] = cell_F(ph, cutoff_um)["F"]
    if hasattr(sd, "cell_f_beyond_cutoff"):
        ref = sd.cell_f_beyond_cutoff(phi_skel, cutoff_um=cutoff_um, by="d_from_um")
        F["F_lit_skel_stage1_fn"] = float(ref["F"])
    meas = sk[good]
    mnr = sk[sk["measured_norind"].to_numpy(dtype=bool)]

    def _pooled(d, num, den):
        if len(d) == 0 or num not in d or den not in d:
            return float("nan")
        return float(d[num].sum() / max(d[den].sum(), 1e-30))

    junction = {
        "n_no_shaft_axis": int((~has_rind & good).sum()),
        "kappa_pooled_norind": _pooled(mnr, "A_mesh_norind_um2", "A_skel_um2"),
        "kappa_pooled_nobase": _pooled(meas, "A_mesh_um2", "A_skel_nobase_um2"),
        "kappa_pooled_both": _pooled(mnr, "A_mesh_norind_um2", "A_skel_nobase_um2"),
        "frac_rind_median": (float(mnr["frac_rind"].median()) if len(mnr)
                             else float("nan")),
        "frac_rind_max": (float(mnr["frac_rind"].max()) if len(mnr)
                          else float("nan")),
        "base_frac_of_skel_median": (
            float((meas["A_skel_base_um2"] / meas["A_skel_um2"]).median())
            if len(meas) else float("nan")),
        "base_frac_of_skel_pooled": _pooled(meas, "A_skel_base_um2", "A_skel_um2"),
    }
    summary = {"cell_id": int(cell_id), "module_version": MODULE_VERSION,
               "n_spines": int(len(sk)), "n_measured": int(good.sum()),
               "n_failed": int((~sk["ok"].astype("boolean").fillna(True)
                                .to_numpy(dtype=bool)).sum()),
               "n_clipped": int(sk.get("clipped", pd.Series(False)).astype(
                   "boolean").fillna(False).sum()),
               "coverage_area": float(meas["A_skel_um2"].sum()
                                      / max(sk["A_skel_um2"].sum(), 1e-30)),
               "kappa_pooled": float(meas["A_mesh_um2"].sum()
                                     / max(meas["A_skel_um2"].sum(), 1e-30))
               if len(meas) else float("nan"),
               "median_frac_cut": float(meas["frac_cut"].median()) if len(meas) else float("nan"),
               "median_frac_bridge": float(meas["frac_bridge"].median()) if len(meas) else float("nan"),
               "max_frac_box": float(meas["frac_box"].max()) if len(meas) else float("nan"),
               "max_frac_unresolved": float(meas["frac_unresolved"].max()) if len(meas) else float("nan"),
               "attribution_gate": gate}
    summary.update(junction)
    summary.update(F)
    return {"phi": phis, "spines": sk, "kappa_table": ktab, "summary": summary}


def write_cell_outputs(out_dir, cell_id, out):
    """CSV per spine, phi^mesh in Stage 1's schema, kappa table, summary row."""
    os.makedirs(out_dir, exist_ok=True)
    cid = int(cell_id)
    paths = {
        "spines": os.path.join(out_dir, "cell%d_spines.csv" % cid),
        "phi_mesh": os.path.join(out_dir, "neuron_%d_phi_mesh.csv" % cid),
        "kappa": os.path.join(out_dir, "cell%d_kappa_function.csv" % cid),
        "summary": os.path.join(out_dir, "spine_area_F_summary.csv"),
    }
    sp = out["spines"].drop(columns=[c for c in ("traceback",)
                                     if c in out["spines"].columns])
    sp.to_csv(paths["spines"], index=False, lineterminator="\n")
    out["phi"]["mesh"].to_csv(paths["phi_mesh"], index=False, lineterminator="\n")
    out["kappa_table"].to_csv(paths["kappa"], index=False, lineterminator="\n")
    row = {k: v for k, v in out["summary"].items() if k != "attribution_gate"}
    row["gate_max_abs_diff_um2"] = out["summary"]["attribution_gate"]["max_abs_diff_um2"]
    row["written_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    new = pd.DataFrame([row])
    if os.path.isfile(paths["summary"]):
        old = pd.read_csv(paths["summary"])
        new = pd.concat([old[old["cell_id"] != cid], new], ignore_index=True)
    new.to_csv(paths["summary"], index=False, lineterminator="\n")
    return paths
