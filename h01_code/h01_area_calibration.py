"""Calibrate the orientation-dependent surface-area inflation g(n) and apply it.

THE PROBLEM
A surface digitised on a voxel grid is measured with a systematic excess area:
a voxel boundary can only run along grid directions, so following a slanted
membrane costs extra path. The excess depends on WHICH WAY the patch faces,
and on an anisotropic grid (8 x 8 x 33 nm for H01 c3 mip 0) it depends on
direction strongly and asymmetrically.

Writing g(n) for the measured-over-true area ratio of a flat patch with unit
normal n, the measured area of any surface S is

    A_meas(S) = integral over S^2 of g(n) dmu_S(n)                        (1)

where mu_S is the area-weighted distribution of normals of S. The object-level
inflation is therefore the mu_S-weighted MEAN of g -- a property of the shape
as well as the grid, which is why one per-object factor calibrated on cylinders
cannot transfer between shapes. This module estimates g itself and inverts (1)
triangle by triangle:

    A_corr(S) = sum over triangles of a_t / g(n_t)                        (2)

so the mesh's own triangles supply mu_S and nothing about the object's global
shape is assumed.

HOW g IS MEASURED, AND WHY THIS WAY
The phantom is a HALF-SPACE: its boundary is planar, so its normal
distribution is a single point mass and (1) collapses to
A_meas = g(n) * A_true exactly -- no averaging over a family of directions and
no curvature mixed in. That is the whole reason for preferring it to a
cylinder, whose normals span a great circle and whose measured inflation is
therefore already an average over that ring.

The difficulty is knowing A_true for the finite patch actually measured. A
window of radius rho has true area pi*rho^2 only if its boundary is handled
perfectly; selecting triangles by centroid leaves an error of order
(perimeter x edge length) / area, around 10% at usable box sizes.

The fix is an identity, not a bigger box. For a surface that is a graph over
the plane perpendicular to n -- no overhangs, true of a digitised half-space --
the projected area of the patch equals its footprint EXACTLY, staircase or not:

    A_true(patch) = sum over selected t of a_t * |n_t . n|                (3)

The numerator of the ratio and (3) are summed over the SAME triangle set, so
the window-boundary error cancels to first order. Measured here: grid-aligned
normals return g = 1.0000, and (3) reproduces pi*rho^2 to within 0.3%.

Two details that were established by measurement, not assumption:

  * Triangles must be restricted to a BAND about the interface as well as a
    disk about the axis. A padded half-space mask is a CLOSED surface, so it
    also carries the far box wall; that wall shares the same footprint and
    doubles the projected sum. Without the band the measurement is exactly a
    factor of two wrong, which is easy to miss because the ratio still looks
    plausible at grid-aligned angles.

  * The plane offset is jittered by sub-voxel amounts and the results
    averaged. A real membrane has no reason to sit exactly on a voxel
    boundary, and at grid-aligned orientations the answer is otherwise
    sensitive to that coincidence.

SMOOTHING IS PART OF THE CALIBRATION
g must be measured through the SAME pipeline the real mesh receives. Budgeted
Taubin smoothing already removes most of the staircase -- on a tilted-cylinder
sweep the raw spread 1.03..1.23 collapsed to about 1.00..1.07 -- so a table
calibrated on raw marching cubes and applied to a smoothed mesh would
over-correct by more than ten points.

SYMMETRY
A grid (rx, ry, rz) with rx == ry is invariant under sign flips of each axis
and under swapping x and y. Since g(n) = g(-n) as well (a plane equals its
reverse), the fundamental domain is

    theta in [0, 90] degrees,  phi in [0, 45] degrees                     (4)

which is 1/16 of the sphere: a 1-degree table is 91 x 46 = 4186 evaluations
rather than 64800. verify_symmetry() checks (4) numerically rather than
trusting it, and fold_is_valid() refuses the x<->y fold when rx != ry.

DEPENDENCIES: numpy, scipy, scikit-image. Pure ASCII, LF only.
"""

import json
import os
import time

import numpy as np

MODULE_VERSION = "h01_area_calibration v1.1"

DEFAULT_RESOLUTION_NM = (8.0, 8.0, 33.0)

# Phantom geometry in nm. rho is the measurement window radius; half_extent
# must exceed it by enough that neither the box walls nor the smoothing's edge
# influence reach the window. band isolates the interface from the far wall.
DEFAULT_RHO_NM = 250.0
DEFAULT_HALF_EXTENT_NM = 550.0
DEFAULT_BAND_NM = 120.0
DEFAULT_N_OFFSETS = 2


class CalibrationError(RuntimeError):
    """Raised when a calibration request is malformed or degenerate."""


# --------------------------------------------------------------------------- #
# 1. Directions                                                                #
# --------------------------------------------------------------------------- #
def direction_from_angles(theta_deg, phi_deg):
    """Unit vector at polar angle theta from +z and azimuth phi from +x."""
    th = np.radians(np.asarray(theta_deg, dtype=float))
    ph = np.radians(np.asarray(phi_deg, dtype=float))
    return np.stack([np.sin(th) * np.cos(ph),
                     np.sin(th) * np.sin(ph),
                     np.cos(th)], axis=-1)


def angles_from_direction(n):
    """(theta, phi) in degrees from a vector. theta from +z, phi from +x."""
    v = np.asarray(n, dtype=float)
    v = v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-30)
    theta = np.degrees(np.arccos(np.clip(v[..., 2], -1.0, 1.0)))
    phi = np.degrees(np.arctan2(v[..., 1], v[..., 0]))
    return theta, np.mod(phi, 360.0)


def fold_to_fundamental(n):
    """Map any direction into theta in [0,90], phi in [0,45]. Returns (th, ph).

    Uses, in order: n ~ -n (a plane equals its reverse), sign flips on each
    axis (the grid is mirror symmetric), and x <-> y (valid only when
    rx == ry -- see fold_is_valid).
    """
    v = np.asarray(n, dtype=float)
    v = v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-30)
    a = np.abs(v)                                   # sign flips and n ~ -n
    x = np.maximum(a[..., 0], a[..., 1])            # x <-> y swap
    y = np.minimum(a[..., 0], a[..., 1])
    z = np.clip(a[..., 2], 0.0, 1.0)
    theta = np.degrees(np.arccos(z))
    phi = np.degrees(np.arctan2(y, np.maximum(x, 1e-30)))
    return theta, phi


def fold_is_valid(resolution_nm, rtol=1e-9):
    """The x <-> y fold requires rx == ry. Guards against a silently wrong table."""
    r = np.asarray(resolution_nm, dtype=float)
    return bool(abs(r[0] - r[1]) <= rtol * max(r[0], r[1]))


def fundamental_grid(step_deg=1.0):
    """(theta, phi) axes over the fundamental domain, inclusive of both edges."""
    n_th = int(round(90.0 / float(step_deg))) + 1
    n_ph = int(round(45.0 / float(step_deg))) + 1
    return np.linspace(0.0, 90.0, n_th), np.linspace(0.0, 45.0, n_ph)


# --------------------------------------------------------------------------- #
# 2. The measurement                                                           #
# --------------------------------------------------------------------------- #
def _halfspace_patch(nhat, resolution_nm, half_extent_nm, offset_nm):
    """Marching-cubes surface of the half-space {p . nhat <= offset} in a box."""
    from skimage import measure

    res = np.asarray(resolution_nm, dtype=float)
    H = float(half_extent_nm)
    n = np.ceil(2 * H / res).astype(int) + 2
    grids = [(np.arange(n[i]) + 0.5) * res[i] - H for i in range(3)]
    X, Y, Z = np.meshgrid(*grids, indexing="ij")
    mask = (np.stack([X, Y, Z], axis=-1) @ nhat) <= float(offset_nm)
    if not mask.any() or mask.all():
        raise CalibrationError("degenerate half-space: offset outside the box")
    verts, faces, _, _ = measure.marching_cubes(
        np.pad(mask, 1).astype(np.float32), level=0.5, spacing=tuple(res))
    return verts - H - res, faces


def _triangle_geometry(verts, faces):
    """Per-triangle centroid, area and unit normal."""
    v, f = np.asarray(verts, dtype=float), np.asarray(faces, dtype=np.int64)
    c = v[f].mean(axis=1)
    cr = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    ln = np.linalg.norm(cr, axis=1)
    return c, ln / 2.0, cr / np.maximum(ln[:, None], 1e-30)


def measure_g(nhat, resolution_nm=DEFAULT_RESOLUTION_NM, rho_nm=DEFAULT_RHO_NM,
              half_extent_nm=DEFAULT_HALF_EXTENT_NM, band_nm=DEFAULT_BAND_NM,
              n_offsets=DEFAULT_N_OFFSETS, smooth=True, smooth_fn=None,
              rng_seed=0, mirror_average=True):
    """See _measure_g_single. mirror_average averages over the x-reflection.

    WHY MIRROR AVERAGING IS ON BY DEFAULT
    The fundamental domain assumes g is invariant under reflecting x, because
    the voxel LATTICE is. The lattice is -- but the MEASUREMENT is not: the
    marching-cubes triangulation table resolves ambiguous cell configurations
    in a way that is not reflection symmetric. Measured directly, the direction
    (0.189, -0.198, 0.962) and its x-mirror give g = 1.1087 and 1.1357, a 2.4%
    disagreement that does NOT shrink with more sub-voxel offsets (g_std is
    0.002) or a larger phantom. Both fold to the same table cell, so without
    this the table would carry a 2.4% arbitrary bias in some cells -- as large
    as the correction itself.

    Averaging over the mirror pair makes the estimator symmetric by
    construction, which is the property the fold requires. The cost is 2x, and
    `mirror_rel_dev` is returned so the size of the underlying asymmetry stays
    visible rather than being hidden by the average.
    """
    if not mirror_average:
        return _measure_g_single(nhat, resolution_nm, rho_nm, half_extent_nm,
                                 band_nm, n_offsets, smooth, smooth_fn, rng_seed)
    n = np.asarray(nhat, dtype=float)
    a = _measure_g_single(n, resolution_nm, rho_nm, half_extent_nm, band_nm,
                          n_offsets, smooth, smooth_fn, rng_seed)
    b = _measure_g_single(n * np.array([-1.0, 1.0, 1.0]), resolution_nm,
                          rho_nm, half_extent_nm, band_nm, n_offsets, smooth,
                          smooth_fn, rng_seed)
    g = 0.5 * (a["g"] + b["g"])
    out = dict(a)
    out.update({
        "g": g,
        "g_mirror_pair": [a["g"], b["g"]],
        "mirror_rel_dev": abs(a["g"] - b["g"]) / max(g, 1e-12),
        "mirror_averaged": True,
        "n_triangles": int(0.5 * (a["n_triangles"] + b["n_triangles"])),
    })
    return out


def _measure_g_single(nhat, resolution_nm=DEFAULT_RESOLUTION_NM,
                      rho_nm=DEFAULT_RHO_NM,
                      half_extent_nm=DEFAULT_HALF_EXTENT_NM,
                      band_nm=DEFAULT_BAND_NM,
                      n_offsets=DEFAULT_N_OFFSETS, smooth=True, smooth_fn=None,
                      rng_seed=0):
    """Inflation factor g for a flat patch with unit normal `nhat`.

    smooth : apply the identical smoothing the real mesh receives. Not
        optional in practice -- see SMOOTHING IS PART OF THE CALIBRATION above.
    smooth_fn : callable(verts, faces) -> verts. Defaults to
        h01_spine_geometry.taubin_smooth_to_budget at this resolution.

    Returns dict with g (mean over jittered offsets), g_std, the per-offset
    values, the triangle count, and footprint_ratio -- the ratio of eq. (3) to
    pi*rho^2, which should sit near 1 and is the self-check that the window and
    band are behaving.
    """
    nhat = np.asarray(nhat, dtype=float)
    nrm = np.linalg.norm(nhat)
    if nrm == 0:
        raise CalibrationError("nhat is the zero vector")
    nhat = nhat / nrm
    res = np.asarray(resolution_nm, dtype=float)

    if smooth and smooth_fn is None:
        import h01_spine_geometry as _sg

        def smooth_fn(v, f):
            return _sg.taubin_smooth_to_budget(v, f, resolution_nm=tuple(res))[0]

    rng = np.random.default_rng(rng_seed)
    step = float(res.min())
    offsets = (np.zeros(1) if int(n_offsets) <= 1
               else rng.uniform(-0.5 * step, 0.5 * step, int(n_offsets)))

    gs, ntris, foot = [], [], []
    for off in offsets:
        v, f = _halfspace_patch(nhat, res, half_extent_nm, off)
        if smooth:
            v = smooth_fn(v, f)
        c, a, nt = _triangle_geometry(v, f)
        along = c @ nhat
        lat = np.linalg.norm(c - np.outer(along, nhat), axis=1)
        sel = (lat <= float(rho_nm)) & (np.abs(along - off) <= float(band_nm))
        if int(sel.sum()) < 50:
            raise CalibrationError(
                "only %d triangles selected for n=%s -- raise rho_nm or check "
                "half_extent_nm" % (int(sel.sum()), nhat.tolist()))
        A_meas = float(a[sel].sum())
        A_true = float((a[sel] * np.abs(nt[sel] @ nhat)).sum())        # eq. (3)
        if A_true <= 0:
            raise CalibrationError("zero projected area for n=%s" % nhat.tolist())
        gs.append(A_meas / A_true)
        ntris.append(int(sel.sum()))
        foot.append(A_true / (np.pi * float(rho_nm) ** 2))

    gs = np.asarray(gs, dtype=float)
    return {
        "g": float(gs.mean()), "g_std": float(gs.std()),
        "g_per_offset": gs.tolist(), "n_offsets": int(len(gs)),
        "n_triangles": int(np.mean(ntris)),
        "footprint_ratio": float(np.mean(foot)),
        "nhat": nhat.tolist(), "smoothed": bool(smooth),
    }


# --------------------------------------------------------------------------- #
# 3. The sweep                                                                 #
# --------------------------------------------------------------------------- #
def calibrate_g_table(step_deg=1.0, resolution_nm=DEFAULT_RESOLUTION_NM,
                      smooth=True, checkpoint_path=None, verbose=True,
                      **measure_kw):
    """Sweep the fundamental domain and tabulate g(theta, phi).

    checkpoint_path : partial results are written after every theta row and
        reloaded on restart. A 1-degree sweep is 4186 evaluations and takes
        hours; losing it to a disconnected runtime is avoidable.

    Returns dict with theta_deg, phi_deg (1-D axes), g (2-D array), meta.
    """
    if not fold_is_valid(resolution_nm):
        raise CalibrationError(
            "resolution %s has rx != ry, so the x<->y fold behind the "
            "fundamental domain is invalid. Sweep phi over [0, 90] instead."
            % (list(resolution_nm),))

    th_axis, ph_axis = fundamental_grid(step_deg)
    G = np.full((len(th_axis), len(ph_axis)), np.nan)
    meta = {"module_version": MODULE_VERSION, "step_deg": float(step_deg),
            "resolution_nm": [float(x) for x in resolution_nm],
            "smoothed": bool(smooth),
            "measure_kw": {k: (list(v) if isinstance(v, (list, tuple)) else v)
                           for k, v in measure_kw.items()},
            "n_evaluations": int(len(th_axis) * len(ph_axis))}

    start_row = 0
    if checkpoint_path and os.path.isfile(checkpoint_path):
        with np.load(checkpoint_path) as z:
            if z["g"].shape == G.shape:
                G = z["g"]
                start_row = int(z["rows_done"])
                if verbose:
                    print("resumed from %s at theta row %d"
                          % (checkpoint_path, start_row))

    t0 = time.time()
    for i in range(start_row, len(th_axis)):
        for j, ph in enumerate(ph_axis):
            n = direction_from_angles(th_axis[i], ph)
            G[i, j] = measure_g(n, resolution_nm=resolution_nm, smooth=smooth,
                                **measure_kw)["g"]
        if checkpoint_path:
            np.savez_compressed(checkpoint_path, g=G, rows_done=i + 1,
                                theta_deg=th_axis, phi_deg=ph_axis)
        if verbose:
            el = time.time() - t0
            done = (i - start_row + 1) * len(ph_axis)
            todo = (len(th_axis) - i - 1) * len(ph_axis)
            print("  theta %5.1f  g in [%.4f, %.4f]  %d done, ~%.1f min left"
                  % (th_axis[i], np.nanmin(G[i]), np.nanmax(G[i]), done,
                     (el / max(done, 1)) * todo / 60.0))

    return {"theta_deg": th_axis, "phi_deg": ph_axis, "g": G, "meta": meta}


def verify_symmetry(resolution_nm=DEFAULT_RESOLUTION_NM, tol=0.02, smooth=True,
                    n_directions=4, rng_seed=0, **measure_kw):
    """Check the symmetries the fundamental domain relies on. Never assume them.

    Tests n ~ -n, sign flips on x and z, and the x <-> y swap, on generic
    directions. Returns the worst relative deviation per symmetry and a pass
    flag at tolerance `tol`.
    """
    rng = np.random.default_rng(rng_seed)
    base = rng.normal(size=(int(n_directions), 3))
    base /= np.linalg.norm(base, axis=1, keepdims=True)

    def g_of(n):
        return measure_g(n, resolution_nm=resolution_nm, smooth=smooth,
                         **measure_kw)["g"]

    out = {}
    for name, op in (("negate", lambda v: -v),
                     ("flip_x", lambda v: v * np.array([-1.0, 1.0, 1.0])),
                     ("flip_z", lambda v: v * np.array([1.0, 1.0, -1.0])),
                     ("swap_xy", lambda v: v[[1, 0, 2]])):
        errs = [abs(g_of(op(v)) - g_of(v)) / max(g_of(v), 1e-12) for v in base]
        out[name] = {"max_rel_dev": float(np.max(errs)),
                     "ok": bool(np.max(errs) <= tol)}
    out["all_ok"] = bool(all(v["ok"] for k, v in out.items() if k != "all_ok"))
    out["tol"] = float(tol)
    return out


# --------------------------------------------------------------------------- #
# 4. Lookup and correction                                                     #
# --------------------------------------------------------------------------- #
def make_g_lookup(table, method="linear", clip_at_one=True):
    """Return g(n) as a callable on arrays of normals, via the table.

    Query directions are FOLDED into the fundamental domain, not clamped, so
    the whole sphere is covered exactly by symmetry.

    clip_at_one : g is a ratio of a staircase length to a straight one, so it
        cannot be below 1 physically. Interpolation noise near grid-aligned
        directions can dip a hair under; clipping keeps eq. (2) from inflating
        an area rather than deflating it.
    """
    from scipy.interpolate import RegularGridInterpolator

    th, ph, G = table["theta_deg"], table["phi_deg"], table["g"]
    if np.isnan(G).any():
        raise CalibrationError("table contains NaN -- the sweep is incomplete")
    interp = RegularGridInterpolator((th, ph), G, method=method,
                                     bounds_error=False, fill_value=None)

    def lookup(n):
        t, p = fold_to_fundamental(n)
        q = np.stack([np.clip(t, th[0], th[-1]),
                      np.clip(p, ph[0], ph[-1])], axis=-1)
        g = np.asarray(interp(q), dtype=float)
        return np.clip(g, 1.0, None) if clip_at_one else g

    return lookup


def correct_mesh_area(verts, faces, g_lookup):
    """Apply eq. (2): A_corr = sum_t a_t / g(n_t). Returns a dict.

    Also reports the area-weighted distribution of g over the mesh, which is
    the practical form of mu_S: it shows whether this object's normals sit in
    well-behaved or badly-behaved parts of the table, and its mean is exactly
    the single number a per-object calibration would have had to guess.
    """
    _, a, nt = _triangle_geometry(verts, faces)
    g = np.asarray(g_lookup(nt), dtype=float).ravel()
    if g.shape != a.shape:
        raise CalibrationError("lookup returned %s values for %s triangles"
                               % (g.shape, a.shape))
    A_raw = float(a.sum())
    A_corr = float((a / g).sum())
    w = a / a.sum()
    order = np.argsort(g)
    cw = np.cumsum(w[order])
    return {
        "area_raw_nm2": A_raw,
        "area_corrected_nm2": A_corr,
        "object_inflation": A_raw / A_corr if A_corr else float("nan"),
        "n_triangles": int(len(a)),
        "g_area_weighted_mean": float((w * g).sum()),
        "g_min": float(g.min()), "g_max": float(g.max()),
        "g_area_weighted_median": float(g[order][int(np.searchsorted(cw, 0.5))]),
        "g_area_weighted_p95": float(g[order][int(np.searchsorted(cw, 0.95))]),
        "note": "object_inflation is the mu_S-weighted mean of g -- a property "
                "of THIS shape on THIS grid, not of the grid alone.",
    }


def mesh_normal_distribution(verts, faces, step_deg=2.0):
    """Area-weighted histogram of the mesh's normals over the fundamental domain.

    This IS mu_S, discretised: entry (i, j) is the total area of the mesh whose
    normal folds into that (theta, phi) cell. Pair it with the g table to see
    where an object actually lives on the calibration surface.
    """
    _, a, nt = _triangle_geometry(verts, faces)
    th, ph = fold_to_fundamental(nt)
    th_axis, ph_axis = fundamental_grid(step_deg)
    H, _, _ = np.histogram2d(th, ph, bins=[
        np.append(th_axis - step_deg / 2, th_axis[-1] + step_deg / 2),
        np.append(ph_axis - step_deg / 2, ph_axis[-1] + step_deg / 2)],
        weights=a)
    return {"theta_deg": th_axis, "phi_deg": ph_axis, "area_nm2": H,
            "total_area_nm2": float(a.sum())}


def save_table(path, table):
    """Write the table (.npz) plus its provenance (.json beside it)."""
    np.savez_compressed(path, theta_deg=table["theta_deg"],
                        phi_deg=table["phi_deg"], g=table["g"])
    with open(os.path.splitext(path)[0] + ".json", "w", newline="\n") as fh:
        json.dump(table.get("meta", {}), fh, indent=2, default=str)
    return path


def load_table(path):
    with np.load(path) as z:
        t = {"theta_deg": z["theta_deg"], "phi_deg": z["phi_deg"], "g": z["g"]}
    j = os.path.splitext(path)[0] + ".json"
    if os.path.isfile(j):
        with open(j) as fh:
            t["meta"] = json.load(fh)
    return t


def interpolation_error(table_coarse, n_test=24,
                        resolution_nm=DEFAULT_RESOLUTION_NM, smooth=True,
                        rng_seed=1, **measure_kw):
    """Hold-out check: how wrong is interpolating a coarse table?

    Draws random directions, reads them from the coarse table, and measures
    them directly. Without this, a coarse sweep presented as a 1-degree table
    is an unverified claim.
    """
    look = make_g_lookup(table_coarse)
    rng = np.random.default_rng(rng_seed)
    v = rng.normal(size=(int(n_test), 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    pred = np.asarray(look(v)).ravel()
    true = np.array([measure_g(x, resolution_nm=resolution_nm, smooth=smooth,
                               **measure_kw)["g"] for x in v])
    rel = np.abs(pred - true) / true
    return {"n_test": int(n_test),
            "max_rel_error": float(rel.max()),
            "mean_rel_error": float(rel.mean()),
            "rms_rel_error": float(np.sqrt((rel ** 2).mean()))}
