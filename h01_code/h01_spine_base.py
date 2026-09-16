"""Measured spine base from the UNION mesh: where does the spine leave the shaft?

THE PROBLEM
-----------
h01_spine_batch.spine_profile slices the analysis mesh of (spine | bridge),
which is the Voronoi-cut spine mask. The shaft was removed before the profile
was drawn, so the profile cannot see the junction: its first finite station is
the cut face, wherever the Voronoi bisector happened to put it, and everything
between the base node and that face is empty. The skeleton's base node is a
SHAFT node, so the true junction -- the shaft surface -- lies somewhere in that
gap, and both A_skel (base frustum) and A_mesh (Voronoi rind) mis-state where.

WHAT THIS MODULE DOES
---------------------
Slices the UNION mesh (spine | bridge | shaft context, the same-cell material
of the cutout) along the same centreline, starting AT the base node, which sits
on the shaft centreline. Walking outward the containing loop is:

    inside the shaft   a longitudinal SLAB of the dendrite (the section plane
                       is perpendicular to the spine, hence roughly parallel to
                       the shaft axis): width 2 sqrt(R^2 - d^2), length bounded
                       by the cutout -- huge area, and very elongated
    at the surface     the slab width -> 0 and the loop becomes neck + flare
    beyond             the neck, then the head: compact, roughly round

The discriminator is CONTACT WITH THE CUTOUT BOX. The shaft runs through the
cutout, so every slab section reaches the box faces; the spine is padded by
pad_nm on every side (a spine that touches the box is refetched with more
padding, h01_spine_area_F.get_spine_roi), so no spine section ever does. A
station is SHAFT if its containing loop comes within BOX_MARGIN_VOX voxels of
any cutout face, eq. (1); the measured base is the first station, walking
outward from the base node, whose loop does not, confirmed by the next
station. This needs no skeleton radius and no assumption about the loop's
shape. (The isoperimetric quotient q = 4 pi A / P^2 was tried first and is
recorded per station, but with a 1 um cutout the slab has q ~ 0.7, which a
tilted neck can match -- measured on the phantom, not assumed.)

Two numbers come out per spine:

    s_base_nm     arc length from the base node to the measured base. For a
                  spine leaving a cylindrical shaft radially this is r_shaft,
                  so  s_base - r_shaft  is a direct test of the rind cylinder
                  (h01_spine_area_F.rind_area_um2) against the real surface.
    A_beyond_um2  calibrated counted area of the ANALYSIS mesh on the far
                  side of the base plane, eq. (2): the spine's own membrane,
                  with the Voronoi rind removed by measurement rather than
                  by a cylinder model.

DEPENDENCIES: numpy; h01_spine_geometry, h01_spine_roi, h01_spine_batch
imported lazily. Pure ASCII, LF only.
"""

import numpy as np

MODULE_VERSION = "h01_spine_base v1.0"

BOX_MARGIN_VOX = 2.0       # a loop this close to a cutout face is the shaft
NM2_PER_UM2 = 1.0e6


class BaseError(RuntimeError):
    """Raised when no base can be measured for a spine."""


# --------------------------------------------------------------------------- #
def union_mask(roi):
    """Spine | bridge | shaft context: everything of the cell in the cutout."""
    m = np.asarray(roi["mask"], dtype=bool).copy()
    if roi.get("bridge_mask") is not None:
        m |= np.asarray(roi["bridge_mask"], dtype=bool)
    if roi.get("shaft_context_mask") is not None:
        m |= np.asarray(roi["shaft_context_mask"], dtype=bool)
    return m


def mesh_from_mask(mask, seg_m, opts):
    """The SAME operator as h01_spine_batch.analysis_mesh (marching cubes, then
    the fixed Taubin count the cylinder table was measured through), applied
    to an arbitrary mask. Returns (verts_nm, faces, smoothing_info)."""
    import h01_spine_geometry as G
    import h01_spine_roi as SR

    am = SR.surface_from_mask(mask, seg_m["resolution_nm"], seg_m["lo_vox"])
    if not G.is_closed_mesh(am["faces"]):
        raise BaseError("union mesh is not closed")
    if opts.get("taubin_iterations"):
        verts, sm = G.taubin_smooth_to_budget(
            am["verts_nm"], am["faces"], max_shift_nm=np.inf,
            max_iterations=int(opts["taubin_iterations"]))
    else:
        verts, sm = G.taubin_smooth_to_budget(
            am["verts_nm"], am["faces"], resolution_nm=seg_m["resolution_nm"],
            max_shift_nm=opts["smooth_shift_nm"])
    return verts, am["faces"], sm


def isoperimetric_quotient(area_nm2, perimeter_nm):
    a = np.asarray(area_nm2, dtype=float)
    p = np.asarray(perimeter_nm, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = 4.0 * np.pi * a / (p * p)
    return np.where(np.isfinite(q) & (p > 0), q, np.nan)


# --------------------------------------------------------------------------- #
def union_profile(roi, spine_nodes, base_node, opts, union=None):
    """Cross sections of the UNION mesh from the base node outward.

    The centreline is the skeleton's own (base node first, then the spine),
    extended past the TIP into the spine mask as spine_profile does. It is NOT
    extended past the base: the base node already lies on the shaft axis, so
    every station from d = 0 to the shaft surface is already covered.

    Returns (profile, meta). meta['union'] carries the union mesh so the caller
    can reuse it.
    """
    import h01_spine_geometry as G

    seg_m = roi["meta"]["layers"]["seg"]
    if union is None:
        U = union_mask(roi)
        uv, uf, usm = mesh_from_mask(U, seg_m, opts)
        union = {"verts_nm": uv, "faces": uf, "mask": U, "smoothing": usm}
    C0, cinfo = G.ordered_centreline(spine_nodes, base_node)
    L0 = cinfo["path_length_nm"]
    spine_mask = np.asarray(roi["mask"], dtype=bool)
    if roi.get("bridge_mask") is not None:
        spine_mask = spine_mask | np.asarray(roi["bridge_mask"], dtype=bool)
    C, xinfo = G.extend_centreline(spine_mask, seg_m["lo_vox"],
                                   seg_m["resolution_nm"], C0, at="tip",
                                   step_nm=opts["step_nm"], smooth_window=5,
                                   min_extension_nm=opts["min_ext_nm"])
    Cr, s = G.resample_polyline(C, opts["step_nm"])
    T_raw = G.polyline_tangents(Cr, smooth_window=opts["tangent_win"])
    on_skel = G.skeleton_station_mask(s, xinfo, L0)
    tc = G.tangent_consistency(T_raw, n_back=opts["ewma_n_back"],
                               decay=opts["ewma_decay"],
                               max_angle_deg=opts["max_angle_deg"],
                               fix_sign=True, check_mask=on_skel)
    prof = G.cross_section_profile(union["verts_nm"], union["faces"], Cr,
                                   tc["stabilised"], arclength_nm=s,
                                   select="containing",
                                   max_offset_nm=opts["max_offset_nm"])
    lo = np.asarray(seg_m["lo_vox"], dtype=float)
    res = np.asarray(seg_m["resolution_nm"], dtype=float)
    hi = lo + np.asarray(spine_mask.shape, dtype=float)
    margin = float(opts.get("box_margin_vox", BOX_MARGIN_VOX))
    for r, p, t in zip(prof, Cr, tc["stabilised"]):
        r["q"] = float(isoperimetric_quotient(r["area_nm2"], r["perimeter_nm"]))
        r["touches_box"] = False
        if r.get("selected") is None:
            continue
        loops = G.mesh_plane_section(union["verts_nm"], union["faces"], p, t)
        lp = np.asarray(loops[int(r["selected"])], dtype=float) + 0.5 * res
        r["touches_box"] = bool(loop_touches_box(lp, lo, hi, res, margin))
        ext = lp.max(axis=0) - lp.min(axis=0)
        r["extent_max_nm"] = float(ext.max())
    return prof, {"union": union, "centreline_nm": Cr, "tangents": tc["stabilised"],
                  "arclength_nm": s, "on_skeleton": on_skel,
                  "centreline_info": cinfo, "extension": xinfo}


def loop_touches_box(loop_nm, lo_vox, hi_vox, resolution_nm, margin_vox):
    """Eq. (1): any loop point within margin_vox voxels of a cutout face.
    loop_nm is in the NODE frame (mesh frame + res/2)."""
    v = np.asarray(loop_nm, dtype=float) / np.asarray(resolution_nm, dtype=float)
    lo = np.asarray(lo_vox, dtype=float) + float(margin_vox)
    hi = np.asarray(hi_vox, dtype=float) - float(margin_vox)
    return bool(np.any(v <= lo) or np.any(v >= hi))


def find_base(profile, min_run=2):
    """First station from the base node whose containing loop is finite and
    does NOT touch the cutout box, confirmed by the next (min_run - 1)
    stations, so a single slab station that happens to clear the box cannot
    pass. Returns (index, info)."""
    q = np.array([r["q"] for r in profile], dtype=float)
    a = np.array([r["area_nm2"] for r in profile], dtype=float)
    touch = np.array([bool(r.get("touches_box", True)) for r in profile])
    ok = np.isfinite(a) & (a > 0) & ~touch
    n = len(ok)
    for i in range(n):
        if ok[i] and all(ok[i:i + min_run]) and i + min_run <= n:
            prev = profile[i - 1] if i > 0 else None
            return i, {"s_base_nm": float(profile[i]["s_nm"]),
                       "q_base": float(q[i]),
                       "area_base_nm2": float(a[i]),
                       "q_prev": float(prev["q"]) if prev else np.nan,
                       "area_prev_nm2": float(prev["area_nm2"]) if prev else np.nan,
                       "n_slab_stations": int(i),
                       "q_slab_max": float(np.nanmax(q[:i])) if i else np.nan,
                       "area_drop_ratio": (float(prev["area_nm2"] / a[i])
                                           if prev and a[i] > 0 else np.nan),
                       "point_nm": np.asarray(profile[i]["point_nm"], float),
                       "tangent": np.asarray(profile[i]["tangent"], float)}
    raise BaseError("every finite station touches the cutout box (%d stations); "
                    "the spine itself may reach the box, or the section plane "
                    "never clears the shaft" % n)


def area_beyond_plane(centroid_nm, area_nm2, g, resolution_nm, point_nm, tangent):
    """Eq. (2): calibrated counted area on the far side of the base plane.

    Centroids are MESH-frame; shifted by +res/2 into the node frame of the
    centreline (h01_spine_area_F header). Returns (A_beyond_um2, A_before_um2).
    """
    res = np.asarray(resolution_nm, dtype=float)
    t = np.asarray(tangent, dtype=float)
    t = t / max(float(np.linalg.norm(t)), 1e-30)
    d = (np.asarray(centroid_nm, dtype=float) + 0.5 * res
         - np.asarray(point_nm, dtype=float)) @ t
    w = np.asarray(area_nm2, dtype=float) / np.asarray(g, dtype=float)
    beyond = d > 0
    return (float(w[beyond].sum()) / NM2_PER_UM2,
            float(w[~beyond].sum()) / NM2_PER_UM2)


# --------------------------------------------------------------------------- #
def measure_base(roi, spine_nodes, base_node, opts, detail=None, g_lookup=None,
                 r_shaft_nm=None):
    """One spine: union profile -> base -> area beyond the base plane.

    detail   : h01_spine_area_F.measure_spine_area(..., return_detail=True)
               output (analysis mesh, classes, centroids); needed for
               A_beyond_um2. Optional.
    g_lookup : calibration lookup, applied to the counted triangles.
    r_shaft_nm : the skeleton's shaft radius at the base, for the comparison
               s_base - r_shaft. Optional.
    Returns a flat record plus the profile.
    """
    prof, meta = union_profile(roi, spine_nodes, base_node, opts)
    i, b = find_base(prof)
    rec = {"s_base_nm": b["s_base_nm"], "q_base": b["q_base"],
           "area_drop_ratio": b["area_drop_ratio"],
           "q_prev": b["q_prev"], "q_slab_max": b["q_slab_max"],
           "area_base_nm2": b["area_base_nm2"], "area_prev_nm2": b["area_prev_nm2"],
           "n_slab_stations": b["n_slab_stations"], "base_station": int(i),
           "r_eq_base_nm": float(np.sqrt(b["area_base_nm2"] / np.pi)),
           "n_stations": len(prof),
           "union_faces": int(len(meta["union"]["faces"])),
           "r_shaft_nm": float(r_shaft_nm) if r_shaft_nm is not None else np.nan,
           "s_base_minus_r_shaft_nm": (float(b["s_base_nm"] - r_shaft_nm)
                                       if r_shaft_nm is not None else np.nan)}
    if detail is not None:
        import h01_spine_area_F as SAF
        keep = np.isin(np.asarray(detail["cls"]), SAF.COUNTED)
        c = np.asarray(detail["centroid_nm"])[keep]
        a = detail_area(detail)[keep]
        g = (np.asarray(g_lookup(np.asarray(detail["normal"])[keep]), float).ravel()
             if (g_lookup is not None and "normal" in detail) else np.ones(len(a)))
        A_b, A_in = area_beyond_plane(c, a, g, detail["resolution_nm"],
                                      b["point_nm"], b["tangent"])
        rec.update({"A_beyond_um2": A_b, "A_before_base_um2": A_in})
    return rec, prof, meta


def detail_area(detail):
    """Per-triangle areas of the analysis mesh, from the detail dict or
    recomputed from its vertices and faces."""
    if "area_nm2" in detail:
        return np.asarray(detail["area_nm2"], dtype=float)
    import h01_spine_area_F as SAF
    _, area, _, _ = SAF.triangle_geometry(detail["verts_nm"], detail["faces"])
    return area


# --------------------------------------------------------------------------- #
# Batch hooks                                                                  #
# --------------------------------------------------------------------------- #
def make_base_callback(nodes, comp, sk, opts, g_lookup, fig_dir=None,
                       show_inline=False):
    """on_success hook for h01_spine_area_F.measure_all_spines: measures the
    base for every spine while its ROI is in memory and writes the result
    INTO the spine's record, so it lands in the ledger next to the rind.
    A failure is recorded as `base_error` and never fails the spine.

    fig_dir : if given, writes fig_dir/sigmaNNNNN_union_profile.html (plotly)
              and, with show_inline, displays it in the notebook.
    """
    import os
    import h01_spine_roi as SR

    r_shaft = dict(zip(sk["sigma_id"].astype(int), sk["r_shaft_nm"].astype(float)))
    if fig_dir:
        os.makedirs(fig_dir, exist_ok=True)

    def on_success(sid, roi, rec, detail):
        try:
            sn, bn, _ = SR.spine_subframe(nodes, comp, sid)
            b, prof, meta = measure_base(roi, sn, bn, opts, detail=detail,
                                         g_lookup=g_lookup,
                                         r_shaft_nm=r_shaft.get(int(sid)))
            for k in ("s_base_nm", "s_base_minus_r_shaft_nm", "A_beyond_um2",
                      "A_before_base_um2", "area_drop_ratio", "n_slab_stations",
                      "r_eq_base_nm", "q_base", "union_faces"):
                rec[k] = b.get(k, np.nan)
            if fig_dir:
                import h01_spine_area_F_figures as FIG
                fig = FIG.union_profile_figure(
                    prof, b, title="sigma %d: union-mesh cross sections" % int(sid))
                path = os.path.join(fig_dir, "sigma%05d_union_profile.html" % int(sid))
                fig.write_html(path, include_plotlyjs="cdn")
                if show_inline:
                    fig.show()
                return path
        except Exception as exc:                        # noqa: BLE001
            rec["base_error"] = "%s: %s" % (type(exc).__name__, exc)
            rec["s_base_nm"] = np.nan
            rec["A_beyond_um2"] = np.nan
        return None
    return on_success


def chain_callbacks(*callbacks):
    """Run several on_success hooks in order; the first non-empty return
    value is what measure_all_spines stores as rec['figure']."""
    cbs = [c for c in callbacks if c is not None]

    def on_success(sid, roi, rec, detail):
        out = None
        for cb in cbs:
            r = cb(sid, roi, rec, detail)
            out = out or r
        return out
    return on_success
