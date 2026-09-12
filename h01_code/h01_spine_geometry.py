"""Mesh smoothing and cross-sectional area along the spine's own skeleton.

SCOPE
-----
Operates on the c3 segmentation mesh at mip 0 (8 x 8 x 33 nm), i.e. the output
of h01_spine_roi.surface_from_mask. Nothing here refetches, refits to EM, or
changes the label partition -- it takes the spine mask's surface as given and
measures it.

WHAT THIS ESTABLISHES
  1. a smoothed surface that does NOT shrink the object (Taubin, not Laplacian)
  2. an ordered, resampled centreline from the sigma's skeleton nodes
  3. cross-sectional area on planes perpendicular to the local centreline
     tangent, by exact mesh-plane intersection rather than voxel counting

WHY TAUBIN AND NOT LAPLACIAN
Plain Laplacian smoothing v <- v + lam*L(v) is a low-pass filter with a
transfer function that attenuates EVERY non-zero frequency, so the surface
shrinks monotonically with iterations -- a sphere collapses toward its centre.
For a volume or area measurement that is fatal: the number you report becomes
a function of how many smoothing iterations you happened to run. Taubin's
lambda/mu scheme alternates a positive step lam and a negative step mu with
mu < -lam, giving a transfer function that is approximately unity in the
passband, so low-frequency shape is preserved while the voxel staircase (the
highest representable frequency) is removed. This module reports the volume
and area change from smoothing every time, so shrinkage is never silent.

WHAT A "PERPENDICULAR CROSS-SECTION" DOES AND DOES NOT MEAN
The plane at centreline station s with unit tangent t(s) cuts the mesh; the
area returned is that of the polygon(s) of intersection. This is a faithful
measure ONLY where the object is locally tube-like and the tangent is a good
local axis. It degrades in three ways, all of which are REPORTED rather than
hidden:
  * more than one intersection loop -- the plane has cut the object somewhere
    else as well (common where the plane, extended, re-enters a curved neck or
    clips the shaft). n_loops is returned per station.
  * obliquity -- if the true local axis is not t(s), the measured area
    overestimates the true cross-section by 1/cos(theta). No correction is
    applied, because theta is unknown; instead the angle between t(s) and the
    principal axis of the selected loop's own neighbourhood is reported as
    `obliquity_deg` so suspect stations are visible.
  * blob geometry -- at a spine HEAD the object is not a tube at all and no
    tangent defines a meaningful cross-section. Areas there are large and
    should be read as "the plane cut through a blob", not as a calibre.

DEPENDENCIES: numpy, scipy. Pure ASCII, LF only.
"""

import numpy as np

MODULE_VERSION = "h01_spine_geometry v1.0"

# Taubin's classic pair: a positive smoothing step and a slightly larger
# negative one. mu < -lam is what makes the filter volume-preserving.
DEFAULT_LAMBDA = 0.5
DEFAULT_MU = -0.53
DEFAULT_ITERATIONS = 20


class GeometryError(RuntimeError):
    """Raised when a mesh or centreline is malformed for the requested measure."""


# --------------------------------------------------------------------------- #
# 1. Mesh measures                                                             #
# --------------------------------------------------------------------------- #
def mesh_area_nm2(verts, faces):
    """Total surface area: sum of triangle areas."""
    v = np.asarray(verts, dtype=float)
    f = np.asarray(faces, dtype=np.int64)
    a = v[f[:, 1]] - v[f[:, 0]]
    b = v[f[:, 2]] - v[f[:, 0]]
    return float(0.5 * np.linalg.norm(np.cross(a, b), axis=1).sum())


def mesh_volume_nm3(verts, faces):
    """Enclosed volume by the divergence theorem (signed tetrahedron sum).

    Valid for a CLOSED, consistently oriented mesh. marching_cubes with
    padding produces one; an open surface gives a meaningless number, so the
    absolute value is returned and the caller should check closure separately.
    """
    v = np.asarray(verts, dtype=float)
    f = np.asarray(faces, dtype=np.int64)
    p0, p1, p2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return float(abs(np.einsum("ij,ij->i", p0, np.cross(p1, p2)).sum() / 6.0))


def is_closed_mesh(faces):
    """True iff every edge is shared by exactly two triangles."""
    f = np.asarray(faces, dtype=np.int64)
    e = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    e = np.sort(e, axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    return bool(np.all(counts == 2))


def vertex_adjacency(faces, n_verts):
    """Sparse 0/1 adjacency matrix over vertices, from the triangle list."""
    from scipy import sparse

    f = np.asarray(faces, dtype=np.int64)
    i = np.concatenate([f[:, 0], f[:, 1], f[:, 2], f[:, 1], f[:, 2], f[:, 0]])
    j = np.concatenate([f[:, 1], f[:, 2], f[:, 0], f[:, 0], f[:, 1], f[:, 2]])
    A = sparse.coo_matrix((np.ones(len(i)), (i, j)),
                          shape=(n_verts, n_verts)).tocsr()
    A.data[:] = 1.0
    return A


def _umbrella_step(verts, A, inv_deg):
    """One uniform-weight Laplacian displacement L(v) = mean(neighbours) - v."""
    return inv_deg[:, None] * (A @ verts) - verts


def taubin_smooth(verts, faces, iterations=DEFAULT_ITERATIONS,
                  lam=DEFAULT_LAMBDA, mu=DEFAULT_MU, report=True):
    """Volume-preserving mesh smoothing. Returns (new_verts, info).

    One iteration is a positive step of size `lam` followed by a negative step
    of size `mu`. Requires mu < -lam < 0; the function refuses otherwise rather
    than silently shrinking the object.

    `info` always carries the area and volume before and after, and their
    ratios, so the cost of smoothing is visible at the call site.
    """
    v0 = np.asarray(verts, dtype=float)
    f = np.asarray(faces, dtype=np.int64)
    if not (lam > 0 and mu < -lam):
        raise GeometryError(
            "Taubin requires mu < -lam < 0; got lam=%s mu=%s. With mu >= -lam "
            "this is plain Laplacian smoothing and WILL shrink the mesh."
            % (lam, mu))

    A = vertex_adjacency(f, len(v0))
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    inv_deg = 1.0 / deg

    v = v0.copy()
    for _ in range(int(iterations)):
        v = v + lam * _umbrella_step(v, A, inv_deg)
        v = v + mu * _umbrella_step(v, A, inv_deg)

    info = {"method": "Taubin lambda/mu", "iterations": int(iterations),
            "lambda": float(lam), "mu": float(mu),
            "n_verts": int(len(v0)), "n_faces": int(len(f))}
    if report:
        a0, a1 = mesh_area_nm2(v0, f), mesh_area_nm2(v, f)
        vol0, vol1 = mesh_volume_nm3(v0, f), mesh_volume_nm3(v, f)
        info.update({
            "area_before_nm2": a0, "area_after_nm2": a1,
            "area_ratio": a1 / a0 if a0 else float("nan"),
            "volume_before_nm3": vol0, "volume_after_nm3": vol1,
            "volume_ratio": vol1 / vol0 if vol0 else float("nan"),
            "max_vertex_shift_nm": float(np.linalg.norm(v - v0, axis=1).max()),
        })
    return v, info


def laplacian_smooth(verts, faces, iterations=DEFAULT_ITERATIONS, lam=0.5):
    """Plain Laplacian smoothing. Provided ONLY as a comparison baseline.

    This SHRINKS the mesh monotonically. Do not use it for anything from which
    a volume or an area will be reported. It exists so the shrinkage can be
    demonstrated against taubin_smooth rather than asserted.
    """
    v = np.asarray(verts, dtype=float).copy()
    f = np.asarray(faces, dtype=np.int64)
    A = vertex_adjacency(f, len(v))
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    inv_deg = 1.0 / deg
    for _ in range(int(iterations)):
        v = v + lam * _umbrella_step(v, A, inv_deg)
    return v


# --------------------------------------------------------------------------- #
# 2. Centreline from the sigma's own skeleton                                  #
# --------------------------------------------------------------------------- #
def ordered_centreline(spine_nodes, base_node=None):
    """Order the sigma's nodes into a path, base first, tip last.

    The nodes carry parent links (`p`), so the ordering is the reconstruction's
    own topology, not a geometric guess. If the sigma branches, the LONGEST
    root-to-leaf path in arc length is taken and the fact is reported --
    silently averaging over branches would mix a neck with a head.

    base_node : the shaft attachment (not part of the sigma). Prepended when
        given, so the centreline starts on the shaft and the first cross
        sections sit at the neck rather than starting mid-spine.

    Returns (points_nm, info).
    """
    df = spine_nodes
    ids = [int(v) for v in df["id"].to_numpy()]
    id_set = set(ids)
    pos = {int(r["id"]): np.array([r["x"], r["y"], r["z"]], dtype=float)
           for _, r in df.iterrows()}
    parent = {int(r["id"]): int(r["p"]) for _, r in df.iterrows()}

    children = {i: [] for i in ids}
    roots = []
    for i in ids:
        p = parent[i]
        if p in id_set:
            children[p].append(i)
        else:
            roots.append(i)
    if not roots:
        raise GeometryError("no root in the sigma -- the node set is cyclic")

    # Longest path by arc length from each root.
    best_path, best_len = None, -1.0
    for r in roots:
        stack = [(r, [r], 0.0)]
        while stack:
            node, path, length = stack.pop()
            kids = children[node]
            if not kids:
                if length > best_len:
                    best_len, best_path = length, path
                continue
            for k in kids:
                stack.append((k, path + [k],
                              length + float(np.linalg.norm(pos[k] - pos[node]))))

    pts = [pos[i] for i in best_path]
    prepended = False
    if base_node is not None and len(base_node):
        b = base_node.iloc[0]
        pts = [np.array([b["x"], b["y"], b["z"]], dtype=float)] + pts
        prepended = True

    P = np.vstack(pts)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    # NOTE: path_length_nm is the length of the polyline ACTUALLY RETURNED,
    # so when base_node is prepended it includes the base-to-first-node
    # segment. It is not the sigma's own internal length.
    info = {"n_nodes_in_sigma": len(ids), "n_roots": len(roots),
            "branched": bool(len(ids) > len(best_path)),
            "n_nodes_on_path": len(best_path),
            "path_length_nm": float(seg.sum()),
            "base_prepended": prepended,
            "node_ids": [int(i) for i in best_path]}
    if info["branched"]:
        info["note"] = ("sigma branches; the longest root-to-leaf path was "
                        "used and %d node(s) are off-path"
                        % (len(ids) - len(best_path)))
    return P, info


def resample_polyline(points_nm, step_nm):
    """Resample a polyline to uniform arc-length spacing. Returns (pts, s)."""
    P = np.asarray(points_nm, dtype=float)
    if len(P) < 2:
        raise GeometryError("need at least 2 points to resample")
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    if total <= 0:
        raise GeometryError("polyline has zero length")
    n = max(2, int(np.floor(total / float(step_nm))) + 1)
    s_new = np.linspace(0.0, total, n)
    out = np.column_stack([np.interp(s_new, s, P[:, k]) for k in range(3)])
    return out, s_new


def polyline_tangents(points_nm, smooth_window=3):
    """Unit tangents by central differences, optionally box-smoothed.

    Raw per-node tangents on an 8-node skeleton are noisy, and a noisy tangent
    tilts the cutting plane, which inflates area by 1/cos(theta). Smoothing the
    tangent field is therefore not cosmetic -- it directly reduces obliquity
    bias. smooth_window=1 disables it.
    """
    P = np.asarray(points_nm, dtype=float)
    n = len(P)
    if n < 2:
        raise GeometryError("need at least 2 points for tangents")
    T = np.zeros_like(P)
    T[1:-1] = P[2:] - P[:-2]
    T[0] = P[1] - P[0]
    T[-1] = P[-1] - P[-2]

    w = int(smooth_window)
    if w > 1:
        k = np.ones(w) / w
        T = np.column_stack([np.convolve(T[:, i], k, mode="same")
                             for i in range(3)])
        T[0] = P[1] - P[0]
        T[-1] = P[-1] - P[-2]

    norms = np.linalg.norm(T, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return T / norms


# --------------------------------------------------------------------------- #
# 3. Exact mesh-plane cross sections                                           #
# --------------------------------------------------------------------------- #
def mesh_plane_section(verts, faces, point_nm, normal, eps_nm=1e-7):
    """Intersect a triangle mesh with a plane. Returns a list of loops.

    Each loop is an (M, 3) array of points in order around the polygon.

    The intersection points are indexed BY THE EDGE they lie on, not by their
    coordinates, so shared points between adjacent triangles are identical by
    construction and loops close exactly. Matching by coordinate proximity
    instead is the usual source of spurious open contours.

    Vertices lying exactly on the plane are nudged to one side by eps_nm; this
    avoids the degenerate cases (a triangle in the plane, a vertex touched by
    the plane) that otherwise produce zero-length or duplicated segments.
    """
    v = np.asarray(verts, dtype=float)
    f = np.asarray(faces, dtype=np.int64)
    n = np.asarray(normal, dtype=float)
    nn = np.linalg.norm(n)
    if nn == 0:
        raise GeometryError("normal is the zero vector")
    n = n / nn
    p = np.asarray(point_nm, dtype=float)

    d = (v - p) @ n
    d = np.where(np.abs(d) < eps_nm, eps_nm, d)

    dv = d[f]
    pos = dv > 0
    npos = pos.sum(axis=1)
    crossing = (npos == 1) | (npos == 2)
    if not crossing.any():
        return []

    fc = f[crossing]
    dc = dv[crossing]

    edge_id, pts, segments = {}, [], []
    for tri, dd in zip(fc, dc):
        hits = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            if (dd[a] > 0) != (dd[b] > 0):
                va, vb = int(tri[a]), int(tri[b])
                key = (va, vb) if va < vb else (vb, va)
                idx = edge_id.get(key)
                if idx is None:
                    t = dd[a] / (dd[a] - dd[b])
                    pts.append(v[va] + t * (v[vb] - v[va]))
                    idx = len(pts) - 1
                    edge_id[key] = idx
                hits.append(idx)
        if len(hits) == 2 and hits[0] != hits[1]:
            segments.append((hits[0], hits[1]))

    if not segments:
        return []

    adj = {}
    for a, b in segments:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    pts = np.asarray(pts, dtype=float)
    loops, seen = [], set()
    for start in adj:
        if start in seen:
            continue
        loop, prev, cur = [start], None, start
        seen.add(start)
        while True:
            nxt = [x for x in adj[cur] if x != prev and x not in seen]
            if not nxt:
                break
            cur, prev = nxt[0], cur
            seen.add(cur)
            loop.append(cur)
        if len(loop) >= 3:
            loops.append(pts[loop])
    return loops


def polygon_area_nm2(loop_nm, normal):
    """Area of a planar polygon in 3D, via the projected shoelace formula.

    A = 0.5 * | n . sum_i (P_i x P_{i+1}) |, which is exact for a planar loop
    and needs no explicit 2D projection basis.
    """
    P = np.asarray(loop_nm, dtype=float)
    if len(P) < 3:
        return 0.0
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    cross_sum = np.cross(P, np.roll(P, -1, axis=0)).sum(axis=0)
    return float(abs(n @ cross_sum) * 0.5)


def polygon_perimeter_nm(loop_nm):
    """Closed perimeter of a polygon."""
    P = np.asarray(loop_nm, dtype=float)
    if len(P) < 2:
        return 0.0
    return float(np.linalg.norm(np.roll(P, -1, axis=0) - P, axis=1).sum())


def _point_in_loop(loop_nm, q_nm, normal):
    """Winding test for a point in a planar polygon, done in the plane basis."""
    n = np.asarray(normal, float); n = n / np.linalg.norm(n)
    a = np.array([1.0, 0.0, 0.0])
    if abs(n @ a) > 0.9:
        a = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(n, a); e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    P = np.asarray(loop_nm, float)
    xy = np.column_stack([(P - q_nm) @ e1, (P - q_nm) @ e2])
    ang = np.arctan2(xy[:, 1], xy[:, 0])
    dang = np.diff(np.concatenate([ang, ang[:1]]))
    dang = (dang + np.pi) % (2 * np.pi) - np.pi
    return abs(dang.sum()) > np.pi


def cross_section_profile(verts, faces, centreline_nm, tangents,
                          arclength_nm=None, select="containing",
                          max_offset_nm=None):
    """Cross-sectional area at every centreline station. Returns list of dicts.

    select : how to choose among multiple intersection loops at one station.
        'containing' (default) -- the loop whose polygon contains the
            centreline point. This is the only choice that is robust when the
            plane also clips the shaft: the shaft loop does not contain the
            spine's centreline point. Falls back to 'nearest' if none contains.
        'nearest'  -- the loop whose centroid is nearest the centreline point.
        'largest'  -- the largest-area loop. NOT recommended: where the plane
            clips the shaft, the shaft loop is the larger one.

    max_offset_nm : discard loops whose centroid is further than this from the
        station. Use it to suppress far-away clipped material outright.

    Every station reports n_loops and the total area of ALL loops as well as
    the selected one, so a station where the selection mattered is visible.
    """
    v = np.asarray(verts, dtype=float)
    f = np.asarray(faces, dtype=np.int64)
    C = np.asarray(centreline_nm, dtype=float)
    T = np.asarray(tangents, dtype=float)
    if len(C) != len(T):
        raise GeometryError("centreline and tangents differ in length")
    if arclength_nm is None:
        seg = np.linalg.norm(np.diff(C, axis=0), axis=1)
        arclength_nm = np.concatenate([[0.0], np.cumsum(seg)])

    out = []
    for k in range(len(C)):
        p, t = C[k], T[k]
        loops = mesh_plane_section(v, f, p, t)
        rec = {"index": int(k), "s_nm": float(arclength_nm[k]),
               "point_nm": p.tolist(), "tangent": t.tolist(),
               "n_loops": len(loops)}
        if not loops:
            rec.update({"area_nm2": float("nan"), "perimeter_nm": float("nan"),
                        "equivalent_radius_nm": float("nan"),
                        "total_area_all_loops_nm2": 0.0,
                        "selected": None,
                        "note": "plane does not intersect the mesh"})
            out.append(rec)
            continue

        cents = np.array([lp.mean(axis=0) for lp in loops])
        offs = np.linalg.norm(cents - p, axis=1)
        keep = np.arange(len(loops))
        if max_offset_nm is not None:
            keep = keep[offs[keep] <= float(max_offset_nm)]
            if len(keep) == 0:
                keep = np.array([int(np.argmin(offs))])

        areas = np.array([polygon_area_nm2(loops[i], t) for i in keep])
        if select == "largest":
            pick = keep[int(np.argmax(areas))]
        elif select == "nearest":
            pick = keep[int(np.argmin(offs[keep]))]
        else:
            inside = [i for i in keep if _point_in_loop(loops[i], p, t)]
            pick = inside[0] if inside else keep[int(np.argmin(offs[keep]))]
            rec["selected_by"] = "containing" if inside else "nearest (fallback)"

        lp = loops[pick]
        a = polygon_area_nm2(lp, t)
        rec.update({
            "area_nm2": a,
            "perimeter_nm": polygon_perimeter_nm(lp),
            "equivalent_radius_nm": float(np.sqrt(a / np.pi)) if a > 0 else 0.0,
            "total_area_all_loops_nm2": float(
                sum(polygon_area_nm2(l, t) for l in loops)),
            "selected": int(pick),
            "centroid_offset_nm": float(offs[pick]),
            "loop_n_points": int(len(lp)),
        })
        out.append(rec)
    return out


def profile_summary(profile):
    """Aggregate a cross_section_profile into reportable numbers."""
    a = np.array([r["area_nm2"] for r in profile], dtype=float)
    ok = np.isfinite(a)
    s = np.array([r["s_nm"] for r in profile], dtype=float)
    multi = int(sum(1 for r in profile if r["n_loops"] > 1))
    return {
        "n_stations": int(len(profile)),
        "n_valid": int(ok.sum()),
        "n_stations_multi_loop": multi,
        "min_area_nm2": float(np.nanmin(a)) if ok.any() else float("nan"),
        "max_area_nm2": float(np.nanmax(a)) if ok.any() else float("nan"),
        "median_area_nm2": float(np.nanmedian(a)) if ok.any() else float("nan"),
        "s_at_min_area_nm": float(s[np.nanargmin(a)]) if ok.any() else float("nan"),
        "min_equivalent_radius_nm": (float(np.sqrt(np.nanmin(a) / np.pi))
                                     if ok.any() else float("nan")),
        "path_length_nm": float(s[-1] - s[0]) if len(s) else float("nan"),
        "note": "min area is the natural neck estimate ONLY if the centreline "
                "runs base-to-tip through a neck; check n_stations_multi_loop "
                "and the profile before quoting it",
    }


# --------------------------------------------------------------------------- #
# 4. Extending the centreline into material the skeleton does not reach        #
# --------------------------------------------------------------------------- #
# WHY THIS SECTION EXISTS
# The reconstruction's skeleton is sampled on a 32 x 32 x 33 nm grid
# (h01_fetch.SKEL_GRID_NM) and its last node stops wherever the tracing stopped
# -- typically short of the true distal tip of the head, and short of the
# proximal attachment once the shaft has been relabelled away. Cross sections
# can only be cut where the centreline goes, so those regions were simply not
# measured: the profile ran out before the object did.
#
# The fix is to continue the centreline using the SEGMENTATION voxels
# (8 x 8 x 33 nm), which are four times finer in x and y than the skeleton
# grid. This is not new information about the tissue -- it is the same c3
# labelling -- but it IS finer than the skeleton, which is the resolution that
# was limiting the profile.
#
# METHOD, and why not something simpler:
#   1. Find the distal tip as the voxel of the mask at maximum GEODESIC
#      distance from the last skeleton node, travelling only inside the mask.
#      Euclidean distance is wrong here: for a curved or hooked head the
#      Euclidean-farthest voxel can sit across a gap of background, and the
#      straight line to it leaves the object.
#   2. Trace back from that tip with a cost field of 1/(EDT + eps), so the
#      path prefers voxels far from the surface -- i.e. it follows the medial
#      axis rather than hugging the inside of a bend. With uniform cost the
#      geodesic shortest path cuts corners and clings to the wall, which tilts
#      every subsequent cutting plane.
#   3. Smooth and resample, because a voxel traceback is a staircase and a
#      staircase tangent is a bad plane normal.

def _nm_to_vox(points_nm, lo_vox, resolution_nm):
    res = np.asarray(resolution_nm, dtype=float)
    lo = np.asarray(lo_vox, dtype=np.int64)
    return np.floor(np.asarray(points_nm, dtype=float) / res).astype(np.int64) - lo


def _vox_to_nm(idx_vox, lo_vox, resolution_nm):
    res = np.asarray(resolution_nm, dtype=float)
    lo = np.asarray(lo_vox, dtype=np.int64)
    return (np.asarray(idx_vox, dtype=float) + lo + 0.5) * res


def _snap_into_mask(mask, idx):
    """Nearest in-mask voxel to `idx`. Needed because the last skeleton node
    can sit just outside the segmented object."""
    m = np.asarray(mask, dtype=bool)
    idx = np.asarray(idx, dtype=np.int64)
    if np.all((idx >= 0) & (idx < np.asarray(m.shape))) and m[tuple(idx)]:
        return idx, 0.0
    occ = np.argwhere(m)
    if len(occ) == 0:
        raise GeometryError("mask is empty")
    d = np.linalg.norm(occ - idx, axis=1)
    k = int(np.argmin(d))
    return occ[k], float(d[k])


def geodesic_tip(mask, start_vox, resolution_nm, candidate_mask=None):
    """Voxel of `mask` at greatest geodesic distance from `start_vox`.

    candidate_mask : optional boolean array restricting WHERE the maximum may
        be taken. Paths are still free to run through the whole mask -- only
        the argmax is restricted. This is what stops an extension seeded at
        one end of an object from selecting the far end as its "tip" and
        retracing the entire object backwards.

    Returns (tip_vox, distance_nm, reachable_fraction). reachable_fraction < 1
    means part of the mask is not connected to the start voxel -- reported,
    never silently ignored, because it usually means the mask is still split.
    """
    from skimage import graph

    m = np.asarray(mask, dtype=bool)
    res = np.asarray(resolution_nm, dtype=float)
    start, _ = _snap_into_mask(m, start_vox)

    costs = np.where(m, 1.0, np.inf)
    mcp = graph.MCP_Geometric(costs, sampling=tuple(res))
    cum, _ = mcp.find_costs([tuple(int(v) for v in start)])
    cum = np.where(m, cum, np.inf)
    finite = np.isfinite(cum) & m
    if not finite.any():
        raise GeometryError("no reachable voxel from the start point")
    searchable = finite if candidate_mask is None else (
        finite & np.asarray(candidate_mask, dtype=bool))
    if not searchable.any():
        raise GeometryError("no reachable voxel inside candidate_mask")
    flat = np.where(searchable, cum, -np.inf)
    tip = np.unravel_index(int(np.argmax(flat)), m.shape)

    # KNOWN LIMITATION, deliberately not "fixed" here. On a DOME the geodesic-
    # farthest voxel is the pole, which is what we want. On a FLAT, truncated
    # end face -- e.g. where the ROI box cut the object -- every rim voxel is
    # geodesically further from the seed than the face centre, so the tip
    # lands on the rim and the extension bends off-axis toward it.
    #
    # A medial tie-break (argmax of the interior EDT among near-maximal
    # voxels) was tried and REJECTED: on a dome it pulls the tip back inside
    # the head, since pole voxels have small EDT by construction, so it breaks
    # the case that matters while not fixing the case that does not. Two
    # existing mechanisms cover it instead: min_extension_nm skips sub-voxel
    # stubs, and tangent_consistency rejects the off-axis planes a rim tip
    # would produce.
    return (np.array(tip, dtype=np.int64), float(cum[tip]),
            float(finite.sum()) / float(m.sum()))


def trace_medial_path(mask, start_vox, end_vox, resolution_nm, eps=1e-3):
    """Path from start to end inside `mask`, biased toward the medial axis.

    Cost per unit distance is 1/(EDT + eps*max_EDT), so travelling through the
    thick middle is cheap and hugging the surface is expensive. Returns an
    (N, 3) integer voxel path.
    """
    from scipy import ndimage
    from skimage import graph

    m = np.asarray(mask, dtype=bool)
    res = np.asarray(resolution_nm, dtype=float)
    edt = ndimage.distance_transform_edt(m, sampling=res)
    scale = float(edt.max()) if edt.max() > 0 else 1.0
    costs = np.where(m, 1.0 / (edt + eps * scale), np.inf)

    mcp = graph.MCP_Geometric(costs, sampling=tuple(res))
    mcp.find_costs([tuple(int(v) for v in start_vox)])
    path = mcp.traceback(tuple(int(v) for v in end_vox))
    return np.asarray(path, dtype=np.int64)


def _smooth_path_nm(points_nm, window):
    """Box-smooth an (N,3) path, holding the endpoints fixed."""
    P = np.asarray(points_nm, dtype=float)
    w = int(window)
    if w <= 1 or len(P) <= 2:
        return P
    k = np.ones(w) / w
    S = np.column_stack([np.convolve(P[:, i], k, mode="same") for i in range(3)])
    half = max(1, w // 2)
    S[:half] = P[:half]
    S[-half:] = P[-half:]
    return S


def extend_centreline(mask, lo_vox, resolution_nm, centreline_nm,
                      at="tip", step_nm=25.0, smooth_window=5,
                      min_extension_nm=0.0):
    """Continue a centreline into mask material the skeleton never reached.

    at : 'tip'  extend past the LAST point
         'base' extend past the FIRST point
         'both' do each in turn

    Returns (new_centreline_nm, info). info records, per end, how far the
    extension ran, how far the seed had to be snapped to land inside the mask,
    and what fraction of the mask was geodesically reachable.

    The extension is appended, never substituted: every original station stays
    where it was, so a profile computed before and after is comparable over the
    shared range.
    """
    C = np.asarray(centreline_nm, dtype=float)
    if len(C) < 2:
        raise GeometryError("need at least 2 centreline points")
    if at == "both":
        C1, i1 = extend_centreline(mask, lo_vox, resolution_nm, C, at="tip",
                                   step_nm=step_nm, smooth_window=smooth_window,
                                   min_extension_nm=min_extension_nm)
        C2, i2 = extend_centreline(mask, lo_vox, resolution_nm, C1, at="base",
                                   step_nm=step_nm, smooth_window=smooth_window,
                                   min_extension_nm=min_extension_nm)
        return C2, {"tip": i1.get("tip"), "base": i2.get("base")}

    if at not in ("tip", "base"):
        raise GeometryError("at must be 'tip', 'base' or 'both'")

    m = np.asarray(mask, dtype=bool)
    res = np.asarray(resolution_nm, dtype=float)
    seed_nm = C[-1] if at == "tip" else C[0]

    # OUTWARD CONSTRAINT. Without it, the geodesic-farthest voxel from a seed
    # at one end of the object is the OTHER end, so the "extension" retraces
    # the whole object backwards and the profile measures it twice. `inward`
    # points from the seed back along the existing centreline; candidates must
    # lie on the opposite side of the plane through the seed.
    inward = (C[-2] - C[-1]) if at == "tip" else (C[1] - C[0])
    ninw = np.linalg.norm(inward)
    if ninw == 0:
        raise GeometryError("centreline has a repeated endpoint; cannot "
                            "determine the outward direction")
    inward = inward / ninw

    occ = np.argwhere(m)
    occ_nm = _vox_to_nm(occ, lo_vox, res)
    outward_ok = ((occ_nm - seed_nm) @ inward) < 0.0
    candidate = np.zeros_like(m)
    if outward_ok.any():
        good = occ[outward_ok]
        candidate[good[:, 0], good[:, 1], good[:, 2]] = True

    seed_vox_raw = _nm_to_vox(seed_nm[None, :], lo_vox, res)[0]
    seed_vox, snap_vox = _snap_into_mask(m, seed_vox_raw)

    if not candidate.any():
        return C, {at: {"extended": False, "extension_length_nm": 0.0,
                        "n_points_added": 0, "seed_snapped_voxels": snap_vox,
                        "reason": "no mask voxel lies outward of the %s end; "
                                  "the centreline already reaches it" % at}}

    tip_vox, geo_nm, reach = geodesic_tip(m, seed_vox, res,
                                          candidate_mask=candidate)

    info_end = {
        "seed_snapped_voxels": snap_vox,
        "n_candidate_voxels_outward": int(candidate.sum()),
        "geodesic_to_tip_nm": geo_nm,
        "reachable_fraction_of_mask": reach,
        "extended": False,
        "extension_length_nm": 0.0,
        "n_points_added": 0,
    }
    if reach < 0.999:
        info_end["WARNING"] = (
            "only %.1f%% of the mask is geodesically reachable from the seed; "
            "the rest is a disconnected component and was NOT explored"
            % (100.0 * reach))

    if geo_nm <= float(min_extension_nm):
        info_end["reason"] = ("geodesic reach %.1f nm <= min_extension_nm %.1f "
                              "-- nothing added" % (geo_nm, float(min_extension_nm)))
        return C, {at: info_end}

    path_vox = trace_medial_path(m, seed_vox, tip_vox, res)
    path_nm = _vox_to_nm(path_vox, lo_vox, res)
    # traceback runs tip -> seed; orient it away from the existing centreline
    if np.linalg.norm(path_nm[0] - seed_nm) > np.linalg.norm(path_nm[-1] - seed_nm):
        path_nm = path_nm[::-1]
    path_nm = _smooth_path_nm(path_nm, smooth_window)

    if len(path_nm) < 2:
        info_end["reason"] = "traceback produced fewer than 2 points"
        return C, {at: info_end}

    ext, _ = resample_polyline(path_nm, step_nm)
    ext = ext[1:]                       # drop the seed itself, already in C
    if len(ext) == 0:
        info_end["reason"] = "extension shorter than one step"
        return C, {at: info_end}

    seg = np.linalg.norm(np.diff(np.vstack([seed_nm, ext]), axis=0), axis=1)
    info_end.update({"extended": True,
                     "extension_length_nm": float(seg.sum()),
                     "n_points_added": int(len(ext)),
                     "tip_vox": [int(v) for v in tip_vox],
                     "tip_nm": _vox_to_nm(tip_vox, lo_vox, res).tolist()})

    out = np.vstack([C, ext]) if at == "tip" else np.vstack([ext[::-1], C])
    return out, {at: info_end}


# --------------------------------------------------------------------------- #
# 5. Terminal cap                                                              #
# --------------------------------------------------------------------------- #
def terminal_cap(mask, lo_vox, resolution_nm, point_nm, normal, side=+1):
    """Account for the material BEYOND the last cross-section plane.

    A profile of cross sections measures the object between its first and last
    plane. Whatever lies distal to the last plane -- the dome of the head -- is
    otherwise silently omitted from any volume integrated from the profile.
    This returns that residual so it can be added explicitly, which is what
    "capping" the profile means here.

    Volume is counted on the VOXEL mask, not by clipping the mesh: voxel
    counting is exact for the segmentation as released and needs no mesh
    clipping, at the cost of quantisation at the 8 x 8 x 33 nm grid. The
    quantisation is reported as n_voxels so the granularity is visible.

    side : +1 counts voxels on the positive side of the plane (along `normal`),
           -1 the negative side.

    PLANE CONVENTION: voxel centres landing EXACTLY on the plane (d == 0) are
    assigned to the positive side. Excluding them from both sides -- the
    obvious first implementation -- means the two sides do not partition the
    mask, and the missing voxels are invisible unless you happen to sum them.
    With an 8 nm grid and a plane at a round coordinate this is not a rare
    edge case: a whole voxel layer can land on d == 0 at once.

    Returns dict with n_voxels, volume_nm3, max_extent_nm (how far the cap
    reaches past the plane), and equivalent_hemisphere_radius_nm -- the radius
    a hemisphere of that volume would have, for comparison against the last
    cross-section's equivalent radius. If the two disagree badly the terminal
    station is not near the tip and the profile should be extended further.
    """
    m = np.asarray(mask, dtype=bool)
    res = np.asarray(resolution_nm, dtype=float)
    n = np.asarray(normal, dtype=float)
    nn = np.linalg.norm(n)
    if nn == 0:
        raise GeometryError("normal is the zero vector")
    n = n / nn
    p = np.asarray(point_nm, dtype=float)

    idx = np.argwhere(m)
    if len(idx) == 0:
        raise GeometryError("mask is empty")
    pts = _vox_to_nm(idx, lo_vox, res)
    d = (pts - p) @ n
    # d == 0 goes to the positive side, so side=+1 and side=-1 partition the
    # mask exactly. See PLANE CONVENTION in the docstring.
    sel = (d >= 0) if side > 0 else (d < 0)

    nvox = int(sel.sum())
    vol = float(nvox * np.prod(res))
    r_hemi = float((3.0 * vol / (2.0 * np.pi)) ** (1.0 / 3.0)) if vol > 0 else 0.0
    return {
        "n_voxels": nvox,
        "volume_nm3": vol,
        "max_extent_nm": float(d[sel].max()) if nvox else 0.0,
        "equivalent_hemisphere_radius_nm": r_hemi,
        "side": int(side),
        "n_voxels_on_plane": int((d == 0).sum()),
        "method": "voxel count distal to the plane, %s nm/voxel; centres with "
                  "d == 0 assigned to the positive side" % res.tolist(),
        "note": "compare equivalent_hemisphere_radius_nm with the last "
                "station's equivalent_radius_nm; a large excess means the "
                "profile stopped well short of the tip",
    }


def profile_volume_nm3(profile, cap=None):
    """Integrate the profile into a volume (trapezoid in area vs arc length).

    Adds `cap['volume_nm3']` when a terminal cap is supplied. Returns a dict
    separating the swept part from the cap, because they are measured
    differently -- the sweep from mesh-plane polygons, the cap from voxels --
    and mixing them into one number without saying so would be misleading.

    ASSUMPTION, stated because it is not always true: a trapezoidal sweep of
    perpendicular cross sections equals the volume only where the object is
    locally tube-like and the planes do not intersect one another. Where the
    centreline curves sharply relative to the calibre, adjacent planes cross
    and the sweep double-counts. Compare against the mesh volume as a check.
    """
    s = np.array([r["s_nm"] for r in profile], dtype=float)
    a = np.array([r["area_nm2"] for r in profile], dtype=float)
    ok = np.isfinite(a)
    swept = float(np.trapezoid(a[ok], s[ok])) if ok.sum() >= 2 else float("nan")
    cap_v = float(cap["volume_nm3"]) if cap else 0.0
    return {
        "swept_volume_nm3": swept,
        "cap_volume_nm3": cap_v,
        "total_volume_nm3": swept + cap_v,
        "n_stations_used": int(ok.sum()),
        "note": "swept part from mesh-plane polygons (trapezoid rule); cap "
                "from voxel counting. Valid only where planes do not cross.",
    }


# --------------------------------------------------------------------------- #
# 6. Tangent stabilisation and profile despiking                               #
# --------------------------------------------------------------------------- #
# THE FAILURE THIS FIXES
# A cross section is only as good as the plane normal that cut it. Where the
# resampled centreline wobbles -- and a medial traceback through a fat head
# wobbles a lot, because "the middle" is barely constrained there -- one
# station's tangent can swing away from its neighbours'. The plane then slices
# the object obliquely or lengthwise and returns an area several times the
# local calibre. In an area-vs-arclength plot that appears as a single-station
# spike, and in 3D as a ring crossing its neighbours.
#
# TWO INDEPENDENT REMEDIES, usable separately or together:
#
#   tangent_consistency  PREVENTIVE. Compares each tangent against an
#       exponentially weighted average of the preceding accepted tangents and
#       flags any that turn by more than `max_angle_deg`. Acts before the mesh
#       is ever cut, so the bad plane is never used.
#
#   despike_profile      CORRECTIVE. Takes a computed profile, blanks the
#       flagged stations, and interpolates area across them from the surviving
#       neighbours. Also catches spikes that survive the angle test, via a
#       robust local median/MAD rule.
#
# Neither ever deletes a station: rejected stations stay in the profile with
# `despiked=True` and their original area preserved in `area_raw_nm2`, so the
# correction is auditable and reversible.

def tangent_consistency(tangents, n_back=3, decay=0.5, max_angle_deg=30.0,
                        fix_sign=True, check_mask=None):
    """Flag tangents that turn too sharply against a decaying-weight history.

    For station k, the reference direction is the exponentially weighted mean
    of the previous `n_back` ACCEPTED tangents,

        t_ref(k) = sum_{j=1..n_back} w_j * t_(k-j) / sum_j w_j,
        w_j = decay^(j-1),                                            (EWMA)

    so the immediately preceding tangent carries weight 1, the one before it
    `decay`, and so on. A rejected tangent is excluded from the history of
    later stations, which stops one bad plane from dragging the reference off
    and cascading.

    check_mask : optional boolean array, one entry per station. Where it is
        False the angle test is COMPUTED and reported but never used to reject.
        This exists because the test is not equally valid everywhere. Its
        reference is an exponentially weighted average of preceding tangents,
        so it assumes the direction it is defending is well established. That
        holds along the shaft-ward body of a spine, where a hundred stations
        agree. It fails on a centreline EXTENSION past the last skeleton node:
        there the medial traceback is only weakly constrained (inside a head
        "the middle" is nearly degenerate), the reference is dominated by the
        long body behind it, and a genuinely turning tip reads as a violation.
        Rejecting there discards exactly the stations the extension was built
        to add. Use the area-coherence rule (despike_profile's MAD test)
        for those stations instead: a tapering tip produces a smooth monotone
        area decline, which the MAD test accepts and a spike does not.

    fix_sign : a plane is defined by +/- its normal, so a tangent that flips
        sign describes the SAME plane while appearing to turn 180 degrees.
        When True the sign is aligned to the reference before the angle is
        measured, so a mere flip is not counted as a rejection.

    Returns dict with angle_deg, `exceeded` (angle test failed anywhere),
    `rejected` (exceeded AND inside check_mask), the reference tangents, and
    `stabilised` -- the tangents with the reference substituted at EVERY
    exceeding station, not only the rejected ones. Always cut the mesh with
    `stabilised`; use `rejected` only to decide which stations to despike.
    """
    T = np.asarray(tangents, dtype=float).copy()
    n = len(T)
    if n == 0:
        raise GeometryError("no tangents given")
    if not (0.0 < decay <= 1.0):
        raise GeometryError("decay must be in (0, 1]; got %s" % decay)

    thr = float(np.cos(np.radians(float(max_angle_deg))))
    angle = np.zeros(n)
    exceeded = np.zeros(n, dtype=bool)      # angle test failed, anywhere
    rejected = np.zeros(n, dtype=bool)      # ... AND inside check_mask
    ref = T.copy()
    hist = []                       # indices of accepted stations, newest last

    for k in range(n):
        if not hist:
            hist.append(k)
            continue
        idx = hist[::-1][:int(n_back)]
        w = np.array([float(decay) ** j for j in range(len(idx))])
        r = (w[:, None] * T[idx]).sum(axis=0)
        rn = np.linalg.norm(r)
        if rn == 0:
            hist.append(k)
            continue
        r = r / rn
        ref[k] = r

        if fix_sign and (T[k] @ r) < 0:
            T[k] = -T[k]
        c = float(np.clip(T[k] @ r, -1.0, 1.0))
        angle[k] = float(np.degrees(np.arccos(c)))
        checked = True if check_mask is None else bool(np.asarray(check_mask)[k])
        if c < thr:
            # EXCEEDING and REJECTING are separate decisions, and conflating
            # them was a real bug. A tangent that fails the angle test is
            # unreliable as a plane normal WHEREVER it occurs, so it is always
            # replaced by the reference and always kept out of the history.
            # Whether the STATION is then discarded from the profile is a
            # different question, answered by check_mask: inside the skeleton
            # span yes, past it no -- there the area-coherence rule decides.
            # Coupling the two meant that sparing a tip station from rejection
            # also silently restored its wobbling raw tangent, and the tip
            # profile collapsed into 0 -> 300k -> 0 oscillation.
            exceeded[k] = True
            rejected[k] = bool(checked)
        else:
            hist.append(k)

    stabilised = T.copy()
    stabilised[exceeded] = ref[exceeded]
    return {
        "angle_deg": angle,
        "exceeded": exceeded,
        "n_exceeded": int(exceeded.sum()),
        "rejected": rejected,
        "n_rejected": int(rejected.sum()),
        "reference": ref,
        "stabilised": stabilised,
        "tangents_sign_fixed": T,
        "n_back": int(n_back), "decay": float(decay),
        "max_angle_deg": float(max_angle_deg),
        "n_stations_checked": (int(len(T)) if check_mask is None
                               else int(np.asarray(check_mask).sum())),
        "n_exceeding_but_unchecked": int((exceeded & ~rejected).sum()),
        "note": "rejected stations are excluded from the EWMA history of "
                "later stations, so one bad plane cannot cascade",
    }


def despike_profile(profile, rejected=None, mad_k=5.0, window=7,
                    interpolate=True):
    """Blank and interpolate spiked cross-section areas. Returns a NEW profile.

    rejected : optional boolean array (e.g. tangent_consistency()['rejected'])
        marking stations to blank regardless of their area.
    mad_k : additionally blank any station whose area deviates from the local
        running median by more than mad_k * MAD. MAD, not standard deviation,
        because a single spike inflates an SD enough to hide itself. Set to
        None to disable and rely solely on `rejected`.
    window : number of stations in the running median (odd, >= 3).

    Each output record gains: despiked (bool), area_raw_nm2 (the original
    value, always preserved), and despike_reason. Interpolation is linear in
    arc length between the nearest surviving neighbours on each side; a spike
    at either end of the profile cannot be interpolated and is left as NaN
    rather than being extrapolated.
    """
    prof = [dict(r) for r in profile]
    n = len(prof)
    if n == 0:
        return prof
    a = np.array([r["area_nm2"] for r in prof], dtype=float)
    s = np.array([r["s_nm"] for r in prof], dtype=float)

    flag = np.zeros(n, dtype=bool)
    reason = [""] * n
    if rejected is not None:
        rej = np.asarray(rejected, dtype=bool)
        if len(rej) != n:
            raise GeometryError("rejected has length %d, profile has %d"
                                % (len(rej), n))
        flag |= rej
        for i in np.where(rej)[0]:
            reason[i] = "tangent turned past the angle threshold"

    if mad_k is not None:
        w = max(3, int(window) | 1)
        half = w // 2
        med = np.full(n, np.nan)
        mad = np.full(n, np.nan)
        for k in range(n):
            lo, hi = max(0, k - half), min(n, k + half + 1)
            seg = a[lo:hi]
            seg = seg[np.isfinite(seg)]
            if len(seg) >= 3:
                med[k] = np.median(seg)
                mad[k] = np.median(np.abs(seg - med[k]))
        with np.errstate(invalid="ignore"):
            scale = np.where(mad > 0, mad, np.nan)
            dev = np.abs(a - med) / scale
        hit = np.isfinite(dev) & (dev > float(mad_k))
        for i in np.where(hit & ~flag)[0]:
            reason[i] = "area %.0f nm2 is %.1f MAD from the local median %.0f" \
                        % (a[i], dev[i], med[i])
        flag |= hit

    a_out = a.copy()
    a_out[flag] = np.nan
    if interpolate:
        good = np.isfinite(a_out)
        if good.sum() >= 2:
            fill = np.interp(s, s[good], a_out[good], left=np.nan, right=np.nan)
            # only fill INTERIOR gaps; never extrapolate past the ends
            first, last = np.argmax(good), n - 1 - np.argmax(good[::-1])
            inside = np.zeros(n, dtype=bool)
            inside[first:last + 1] = True
            a_out = np.where(~good & inside, fill, a_out)

    for k in range(n):
        prof[k]["area_raw_nm2"] = float(a[k])
        prof[k]["despiked"] = bool(flag[k])
        prof[k]["despike_reason"] = reason[k]
        prof[k]["area_nm2"] = float(a_out[k])
        prof[k]["equivalent_radius_nm"] = (float(np.sqrt(a_out[k] / np.pi))
                                           if np.isfinite(a_out[k]) and a_out[k] > 0
                                           else float("nan"))
    return prof


def despike_report(profile):
    """Summarise what despike_profile changed."""
    n = len(profile)
    d = [r for r in profile if r.get("despiked")]
    return {
        "n_stations": n,
        "n_despiked": len(d),
        "fraction_despiked": (len(d) / n) if n else float("nan"),
        "stations": [{"index": r["index"], "s_nm": r["s_nm"],
                      "area_raw_nm2": r["area_raw_nm2"],
                      "area_nm2": r["area_nm2"],
                      "reason": r["despike_reason"]} for r in d],
    }


def skeleton_station_mask(arclength_nm, extend_info, original_length_nm,
                          tol_nm=0.0):
    """Boolean mask of the stations that lie on the ORIGINAL skeleton span.

    After extend_centreline(at='both'), the resampled arc length runs

        [0, Lb)            base extension
        [Lb, Lb + L0]      the original skeleton centreline
        (Lb + L0, S]       tip extension

    where Lb is the base extension length and L0 the original centreline
    length. This returns the middle band, which is where the EWMA angle test
    is trustworthy. Pass it to tangent_consistency(check_mask=...).
    """
    s_arr = np.asarray(arclength_nm, dtype=float)
    lb = float((extend_info.get("base") or {}).get("extension_length_nm", 0.0))
    lo = lb - float(tol_nm)
    hi = lb + float(original_length_nm) + float(tol_nm)
    return (s_arr >= lo) & (s_arr <= hi)


def taubin_smooth_to_budget(verts, faces, resolution_nm=None,
                            max_shift_nm=None, max_iterations=200,
                            lam=DEFAULT_LAMBDA, mu=DEFAULT_MU):
    """Smooth only as far as the voxel quantisation justifies. Returns (v, info).

    WHY A BUDGET AND NOT A PLATEAU. Smoothing until the surface area stops
    changing is the wrong stopping rule: the plateau is reached by flattening
    real structure, not only the staircase. At 8 x 8 x 33 nm the position of
    the true membrane is uncertain by roughly half a voxel, so moving a vertex
    by MORE than that is moving it further than the data's own uncertainty --
    at which point the smoother is inventing shape rather than removing a
    known artefact.

    The default budget is therefore half the voxel diagonal,

        max_shift_nm = 0.5 * ||resolution_nm||_2                       (BUDGET)

    which for c3 mip 0 is 0.5 * ||(8, 8, 33)|| = 17.5 nm. Iteration stops as
    soon as the largest vertex displacement from the ORIGINAL position would
    exceed it.

    This is a defensible ceiling, not an optimum: it says how far you may go,
    not that you should go that far. Fewer iterations are always safer for
    membrane detail; more are never justified by the segmentation's own
    resolution.
    """
    from scipy import sparse                                    # noqa: F401

    v0 = np.asarray(verts, dtype=float)
    f = np.asarray(faces, dtype=np.int64)
    if not (lam > 0 and mu < -lam):
        raise GeometryError("Taubin requires mu < -lam < 0")

    if max_shift_nm is None:
        if resolution_nm is None:
            raise GeometryError("give max_shift_nm or resolution_nm")
        max_shift_nm = 0.5 * float(np.linalg.norm(
            np.asarray(resolution_nm, dtype=float)))

    A = vertex_adjacency(f, len(v0))
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    inv_deg = 1.0 / deg

    v = v0.copy()
    used, stopped = 0, "max_iterations reached"
    for it in range(int(max_iterations)):
        trial = v + lam * _umbrella_step(v, A, inv_deg)
        trial = trial + mu * _umbrella_step(trial, A, inv_deg)
        shift = float(np.linalg.norm(trial - v0, axis=1).max())
        if shift > float(max_shift_nm):
            stopped = ("budget: next iteration would shift a vertex %.2f nm, "
                       "past the %.2f nm budget" % (shift, float(max_shift_nm)))
            break
        v = trial
        used = it + 1
    else:
        stopped = "max_iterations reached without exceeding the budget"

    a0, a1 = mesh_area_nm2(v0, f), mesh_area_nm2(v, f)
    vol0, vol1 = mesh_volume_nm3(v0, f), mesh_volume_nm3(v, f)
    return v, {
        "method": "Taubin lambda/mu, shift-budgeted",
        "iterations_used": int(used), "stopped_because": stopped,
        "max_shift_budget_nm": float(max_shift_nm),
        "max_vertex_shift_nm": float(np.linalg.norm(v - v0, axis=1).max()),
        "lambda": float(lam), "mu": float(mu),
        "area_before_nm2": a0, "area_after_nm2": a1,
        "area_ratio": a1 / a0 if a0 else float("nan"),
        "volume_before_nm3": vol0, "volume_after_nm3": vol1,
        "volume_ratio": vol1 / vol0 if vol0 else float("nan"),
        "note": "budget = half the voxel diagonal by default; it is a ceiling "
                "on what the segmentation's resolution justifies, not a target",
    }
