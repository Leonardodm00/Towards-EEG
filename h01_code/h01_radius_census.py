"""Census of local radii across spines, to count how much membrane sits at
radii where the area calibration is not trustworthy.

WHY THIS EXISTS
---------------
The g0(theta, phi) table calibrates the marching-cubes + Taubin area inflation
on a FLAT phantom. A held-out sphere sweep (11 radii, 40-300 nm, single
realisation, run 2026-09-08) showed the corrected error is stable at +1.2 to
+2.2 percent for sphere radii R >~ 130 nm, but becomes erratic below that:
+19.3 percent at R = 40 nm, -1.9 percent at R = 49 nm, +13.8 percent at
R = 60 nm. So the calibration has a validity floor in RADIUS, not only in
orientation, and we need to know what fraction of real spine membrane sits
below it before quoting any corrected area.

This module answers exactly that question and nothing else. It does not
correct anything and does not touch the g table.

THE TWO THRESHOLDS, AND WHY THEY DIFFER BY MORE THAN A FACTOR OF TWO
--------------------------------------------------------------------
Let `a` be the equivalent-circle radius of a cross section, a = sqrt(A / pi),
which h01_spine_geometry.cross_section_profile already reports per station as
`equivalent_radius_nm`.

(1) SAMPLING FLOOR.  Stelldinger, Latecki and Siqueira (IEEE TPAMI 29(1):126,
    2007, full text) prove topology preservation for an r-regular object on a
    cubic r'-grid only when 2r' < r, where r' is the covering radius of the
    grid (half the voxel diagonal) and r is the osculating-ball radius. For a
    tube of cross-sectional radius a the inside osculating ball is bounded by
    a, so the condition is

        a > 2r' = |resolution_nm| = 34.886 nm   for (8, 8, 33) nm.

    Below this NOTHING is guaranteed -- not the area, not even the topology.
    This threshold uses `a` directly and is the more defensible of the two.
    CAVEAT: their theorem is stated for a CUBIC grid. H01 is 8x8x33, so this
    is an adaptation, not a direct application; the covering radius is
    dominated by the 33 nm axis (16.5 of the 17.443 nm).

(2) CALIBRATION-VALIDATED FLOOR.  The sphere sweep validates radii as SPHERE
    radii. A sphere of radius R has mean curvature H = 1/R; a cylinder of
    cross-sectional radius a has H = 1/(2a). Matching mean curvature gives

        R_equivalent = 2a                                            (eq. 1)

    so the validated band R >= 130 nm corresponds to a >= 65 nm. Forgetting
    the factor 2 here would apply roughly twice the curvature penalty a neck
    deserves.
    CAVEAT (my own reasoning, not from a source): eq. (1) treats the neck as a
    circular cylinder. At the head-neck and neck-shaft junctions the surface is
    saddle-shaped, where the two principal curvatures have opposite signs and H
    can be near zero or negative. Equation (1) is therefore a mid-neck
    approximation and is NOT valid at the junctions -- which are precisely the
    places flagged as critical. Treat band (2) as indicative and band (1) as
    the hard one.

ARC-LENGTH WEIGHTING
--------------------
Stations come from h01_spine_geometry.resample_polyline, which is uniform in
arc length. A plain COUNT histogram over those stations is therefore already
an arc-length-weighted histogram, and the band fractions are fractions of
centreline length. They are NOT fractions of membrane area: a station at small
radius carries proportionally less surface. `area_weighted` band fractions are
reported alongside, using the local lateral area element 2*pi*a*ds.

OVERSAMPLING CAVEAT
-------------------
Oversampling the centreline below the skeleton node spacing (H01 skeletons are
coarse) does NOT add independent centreline information -- the path between
nodes is linearly interpolated. It does add independent MESH cross sections
along that interpolated path, which is the point: the mesh is far finer than
the skeleton. But a wobble in the interpolated path tilts the cutting plane and
inflates area by 1/cos(theta), so `oversample_factor` above ~8 buys correlated
samples, not new information. The default step is one in-plane voxel (8 nm).

DEPENDENCIES: numpy, h01_spine_geometry. Pure ASCII, LF only.
"""

import numpy as np

import h01_spine_geometry as G

MODULE_VERSION = "h01_radius_census v1.0"

# Sphere radius above which the held-out sweep found the corrected error
# stable. Below this the sweep was erratic. See module docstring.
VALIDATED_SPHERE_RADIUS_NM = 130.0

# Mean-curvature mapping tube -> sphere, eq. (1).
TUBE_TO_SPHERE = 2.0

DEFAULT_STEP_NM = 8.0
DEFAULT_MAX_OFFSET_NM = 400.0


class CensusError(RuntimeError):
    """Raised when a radius census cannot be computed."""


# --------------------------------------------------------------------------- #
# thresholds
# --------------------------------------------------------------------------- #
def sampling_floor_nm(resolution_nm):
    """2r' for the grid: the Stelldinger sampling bound, in nm.

    r' is the covering radius = half the voxel diagonal, so 2r' is the full
    voxel diagonal. Cross-sectional radii below this have no reconstruction
    guarantee of any kind.
    """
    r = np.asarray(resolution_nm, dtype=float)
    if r.shape != (3,) or np.any(r <= 0):
        raise CensusError("resolution_nm must be 3 positive numbers, got %r"
                          % (resolution_nm,))
    return float(np.linalg.norm(r))


def calibration_floor_nm(validated_sphere_radius_nm=VALIDATED_SPHERE_RADIUS_NM):
    """Cross-sectional radius `a` below which the g0 table is unvalidated.

    Inverts eq. (1): a = R_equivalent / 2.
    """
    return float(validated_sphere_radius_nm) / TUBE_TO_SPHERE


def equivalent_sphere_radius_nm(equivalent_circle_radius_nm):
    """Map cross-sectional radius `a` to the sphere radius of equal mean
    curvature, eq. (1). Mid-neck approximation; see module docstring."""
    a = np.asarray(equivalent_circle_radius_nm, dtype=float)
    return TUBE_TO_SPHERE * a


# --------------------------------------------------------------------------- #
# per-spine profile
# --------------------------------------------------------------------------- #
def spine_radius_profile(verts, faces, spine_nodes=None, base_node=None,
                         centreline_nm=None, tangents=None,
                         step_nm=DEFAULT_STEP_NM,
                         max_offset_nm=DEFAULT_MAX_OFFSET_NM,
                         smooth_window=3, despike=True, select="containing"):
    """Oversampled equivalent-radius profile along one spine.

    Two ways to call it, so the census is decoupled from how the centreline
    was built:

      (a) pass spine_nodes (and optionally base_node) -- the centreline is
          built here with ordered_centreline + resample_polyline;
      (b) pass centreline_nm and tangents directly -- use this when the caller
          already ran extend_centreline() and wants the extended path.

    Returns (records, info) where records is the despiked cross-section
    profile from h01_spine_geometry with `equivalent_radius_nm` per station.

    NOTE: no centreline extension is performed here. If the caller wants the
    base and tip extensions they must build the centreline themselves and use
    form (b); silently extending would change the arc-length denominator that
    every fraction in this module is computed against.
    """
    if centreline_nm is None:
        if spine_nodes is None:
            raise CensusError("pass either spine_nodes or centreline_nm")
        pts, cinfo = G.ordered_centreline(spine_nodes, base_node=base_node)
        C, s = G.resample_polyline(pts, float(step_nm))
        T = G.polyline_tangents(C, smooth_window=smooth_window)
        built = {"source": "ordered_centreline", "centreline_info": cinfo}
    else:
        C = np.asarray(centreline_nm, dtype=float)
        if tangents is None:
            T = G.polyline_tangents(C, smooth_window=smooth_window)
        else:
            T = np.asarray(tangents, dtype=float)
        if len(T) != len(C):
            raise CensusError("centreline and tangents differ in length")
        seg = np.linalg.norm(np.diff(C, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        built = {"source": "caller-supplied centreline"}

    prof = G.cross_section_profile(verts, faces, C, T, arclength_nm=s,
                                   select=select, max_offset_nm=max_offset_nm)

    info = dict(built)
    info.update({"n_stations": len(prof),
                 "step_nm": float(step_nm),
                 "path_length_nm": float(s[-1] - s[0]) if len(s) else np.nan,
                 "despiked": bool(despike)})

    if despike:
        prof = G.despike_profile(prof)
        info["despike_report"] = G.despike_report(prof)

    return prof, info


def profile_radii(profile):
    """Extract (s_nm, equivalent_radius_nm) as finite-masked arrays."""
    s = np.array([r["s_nm"] for r in profile], dtype=float)
    a = np.array([r.get("equivalent_radius_nm", np.nan) for r in profile],
                 dtype=float)
    ok = np.isfinite(a) & np.isfinite(s) & (a > 0)
    return s, a, ok


# --------------------------------------------------------------------------- #
# classification
# --------------------------------------------------------------------------- #
def classify_radii(equivalent_radius_nm, resolution_nm,
                   validated_sphere_radius_nm=VALIDATED_SPHERE_RADIUS_NM):
    """Per-station band assignment. Returns dict of boolean arrays plus edges.

    Bands, on the cross-sectional radius `a`:
        below_sampling   a <  2r'                  no guarantee at all
        unvalidated      2r' <= a < a_cal          reconstructible, but the
                                                   g0 table was never checked
                                                   at this curvature
        validated        a >= a_cal                inside the swept band
    """
    a = np.asarray(equivalent_radius_nm, dtype=float)
    a_samp = sampling_floor_nm(resolution_nm)
    a_cal = calibration_floor_nm(validated_sphere_radius_nm)
    if a_cal <= a_samp:
        # Not an error, but the bands collapse and the caller should know.
        note = ("calibration floor %.2f nm is below the sampling floor %.2f nm;"
                " the 'unvalidated' band is empty" % (a_cal, a_samp))
    else:
        note = ""
    finite = np.isfinite(a)
    below = finite & (a < a_samp)
    unval = finite & (a >= a_samp) & (a < a_cal)
    valid = finite & (a >= a_cal)
    return {"below_sampling": below,
            "unvalidated": unval,
            "validated": valid,
            "finite": finite,
            "sampling_floor_nm": a_samp,
            "calibration_floor_nm": a_cal,
            "validated_sphere_radius_nm": float(validated_sphere_radius_nm),
            "note": note}


def band_fractions(profile, resolution_nm,
                   validated_sphere_radius_nm=VALIDATED_SPHERE_RADIUS_NM):
    """Fraction of centreline length and of lateral area in each band.

    Length fraction is a plain count over uniformly spaced stations.
    Area fraction weights each station by the lateral element 2*pi*a*ds, which
    is the right weight when asking "how much MEMBRANE is at a bad radius"
    rather than "how much CENTRELINE".
    """
    s, a, ok = profile_radii(profile)
    cls = classify_radii(a, resolution_nm, validated_sphere_radius_nm)

    n_ok = int(ok.sum())
    if n_ok == 0:
        raise CensusError("no station has a finite positive radius")

    # ds per station: midpoint rule on a uniform grid, robust to gaps.
    ds = np.gradient(s) if len(s) > 1 else np.array([1.0])
    w_area = 2.0 * np.pi * np.where(ok, a, 0.0) * np.abs(ds)

    out = {"n_stations": int(len(a)),
           "n_valid_stations": n_ok,
           "sampling_floor_nm": cls["sampling_floor_nm"],
           "calibration_floor_nm": cls["calibration_floor_nm"],
           "min_radius_nm": float(np.nanmin(a[ok])),
           "median_radius_nm": float(np.nanmedian(a[ok])),
           "max_radius_nm": float(np.nanmax(a[ok])),
           "s_at_min_radius_nm": float(s[ok][int(np.argmin(a[ok]))]),
           "total_lateral_area_nm2": float(w_area.sum()),
           "note": cls["note"]}

    for band in ("below_sampling", "unvalidated", "validated"):
        m = cls[band] & ok
        out["n_" + band] = int(m.sum())
        out["length_fraction_" + band] = float(m.sum() / n_ok)
        out["area_fraction_" + band] = (
            float(w_area[m].sum() / w_area.sum()) if w_area.sum() > 0
            else float("nan"))
    return out


# --------------------------------------------------------------------------- #
# census across spines
# --------------------------------------------------------------------------- #
def census_over_spines(sigma_ids, mesh_provider, nodes=None, comp=None,
                       resolution_nm=(8.0, 8.0, 33.0),
                       step_nm=DEFAULT_STEP_NM,
                       validated_sphere_radius_nm=VALIDATED_SPHERE_RADIUS_NM,
                       spine_nodes_provider=None,
                       centreline_provider=None,
                       max_offset_nm=DEFAULT_MAX_OFFSET_NM,
                       despike=True, on_error="record"):
    """Run the radius profile over many spines and pool the result.

    mesh_provider(sigma_id) -> (verts_nm, faces). Required. Decoupled on
        purpose: the caller decides whether that is a cached npz, a freshly
        marched mask, or a smoothed mesh. Whatever it returns must have been
        through the SAME pipeline the areas will be measured on, or the radii
        will not correspond to the mesh being corrected.

    Exactly one of:
      spine_nodes_provider(sigma_id) -> (spine_nodes, base_node)
      centreline_provider(sigma_id)  -> (centreline_nm, tangents)
    If neither is given, `nodes` and `comp` must be supplied and
    h01_spine_roi.spine_subframe is used.

    on_error : 'record' (default) logs the failure per spine and continues;
        'raise' stops. Failures are common and must be visible, not silently
        dropped -- a spine that fails to section is usually a spine whose
        centreline is wrong, which is itself a finding.

    Returns (per_spine, pooled) where per_spine is a list of dicts and pooled
    holds the concatenated radii with their spine ids.
    """
    if mesh_provider is None:
        raise CensusError("mesh_provider is required")
    if spine_nodes_provider is None and centreline_provider is None:
        if nodes is None or comp is None:
            raise CensusError("supply spine_nodes_provider, or "
                              "centreline_provider, or both nodes and comp")
        import h01_spine_roi as R

        def spine_nodes_provider(sid):
            sn, bn, _ = R.spine_subframe(nodes, comp, int(sid),
                                         include_base=True)
            return sn, bn

    per_spine = []
    all_r, all_s, all_id = [], [], []

    for sid in sigma_ids:
        rec = {"sigma_id": int(sid), "ok": False}
        try:
            verts, faces = mesh_provider(sid)
            kw = dict(step_nm=step_nm, max_offset_nm=max_offset_nm,
                      despike=despike)
            if centreline_provider is not None:
                C, T = centreline_provider(sid)
                prof, pinfo = spine_radius_profile(verts, faces,
                                                   centreline_nm=C,
                                                   tangents=T, **kw)
            else:
                sn, bn = spine_nodes_provider(sid)
                prof, pinfo = spine_radius_profile(verts, faces,
                                                   spine_nodes=sn,
                                                   base_node=bn, **kw)
            frac = band_fractions(prof, resolution_nm,
                                  validated_sphere_radius_nm)
            s, a, ok = profile_radii(prof)
            rec.update(frac)
            rec["profile_info"] = pinfo
            rec["ok"] = True
            all_r.append(a[ok])
            all_s.append(s[ok])
            all_id.append(np.full(int(ok.sum()), int(sid), dtype=np.int64))
        except Exception as exc:                      # noqa: BLE001
            if on_error == "raise":
                raise
            rec["error"] = "%s: %s" % (type(exc).__name__, exc)
        per_spine.append(rec)

    pooled = {
        "radius_nm": (np.concatenate(all_r) if all_r
                      else np.zeros(0, dtype=float)),
        "s_nm": (np.concatenate(all_s) if all_s else np.zeros(0, dtype=float)),
        "sigma_id": (np.concatenate(all_id) if all_id
                     else np.zeros(0, dtype=np.int64)),
        "n_spines_requested": len(list(sigma_ids)),
        "n_spines_ok": int(sum(1 for r in per_spine if r["ok"])),
        "resolution_nm": tuple(float(v) for v in resolution_nm),
        "step_nm": float(step_nm),
    }
    return per_spine, pooled


def pooled_summary(pooled, validated_sphere_radius_nm=VALIDATED_SPHERE_RADIUS_NM):
    """Band counts over all stations of all spines pooled together."""
    a = np.asarray(pooled["radius_nm"], dtype=float)
    if a.size == 0:
        raise CensusError("pooled census is empty; every spine failed")
    cls = classify_radii(a, pooled["resolution_nm"], validated_sphere_radius_nm)
    n = int(np.isfinite(a).sum())
    out = {"n_stations": n,
           "n_spines_ok": int(pooled["n_spines_ok"]),
           "n_spines_requested": int(pooled["n_spines_requested"]),
           "sampling_floor_nm": cls["sampling_floor_nm"],
           "calibration_floor_nm": cls["calibration_floor_nm"],
           "min_radius_nm": float(np.nanmin(a)),
           "p01_radius_nm": float(np.nanpercentile(a, 1)),
           "p05_radius_nm": float(np.nanpercentile(a, 5)),
           "median_radius_nm": float(np.nanmedian(a)),
           "p95_radius_nm": float(np.nanpercentile(a, 95)),
           "max_radius_nm": float(np.nanmax(a))}
    for band in ("below_sampling", "unvalidated", "validated"):
        m = cls[band]
        out["n_" + band] = int(m.sum())
        out["length_fraction_" + band] = float(m.sum() / n) if n else np.nan
    out["n_spines_touching_below_sampling"] = int(len(
        np.unique(np.asarray(pooled["sigma_id"])[cls["below_sampling"]])))
    out["n_spines_touching_unvalidated"] = int(len(
        np.unique(np.asarray(pooled["sigma_id"])[cls["unvalidated"]])))
    return out


def radius_histogram(pooled, bin_width_nm=5.0, max_radius_nm=None):
    """Histogram of pooled station radii. Returns edges, counts, centres.

    Uniform stations mean counts are proportional to centreline length.
    """
    a = np.asarray(pooled["radius_nm"], dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        raise CensusError("nothing to histogram")
    hi = float(max_radius_nm) if max_radius_nm else float(np.ceil(a.max()))
    edges = np.arange(0.0, hi + float(bin_width_nm), float(bin_width_nm))
    counts, edges = np.histogram(a, bins=edges)
    centres = 0.5 * (edges[:-1] + edges[1:])
    return {"edges_nm": edges, "counts": counts, "centres_nm": centres,
            "bin_width_nm": float(bin_width_nm), "n_total": int(a.size)}
