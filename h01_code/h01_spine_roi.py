"""Single-spine ROI extraction from the H01 release.

PURPOSE
-------
Answer one question cheaply: what does ONE spine look like, in EM, in the c3
segmentation, and in the 6-class subcompartment rendering?

This exists because the whole-cell census (h01_segid_census.run_census) issues
one request per distinct segmentation chunk touched by the skeleton, and a
half-millimetre-wide dendritic tree touches a great many of them. A single
spine occupies a box a few micrometres on a side. One bbox cutout per layer
replaces thousands of scattered point lookups.

It also makes the per-node segid lookup free: once the c3 cutout is in memory,
the segid of every node of the spine is an array index, not a network call.

SEPARATION OF CONCERNS (deliberate, mirrors h01_segid_census)
------------------------------------------------------------
  spine_subframe        pure: pick sigma's nodes out of the node table
  roi_bbox_nm           pure: bounding box in NANOMETRES, padded
  bbox_nm_to_voxel      pure: nm -> voxel for a given layer resolution
  estimate_bytes        pure: download size guard
  segids_at_points      pure: sample a fetched array at node positions
  mask_from_segids      pure: boolean mask of the spine's segment(s)
  surface_from_mask     pure: marching cubes -> vertices in nm
  decode_subcompartment pure: strip the +100 / +1000 label offsets
  save_roi / load_roi   pure IO, no network
  make_cloudvolume_reader / fetch_roi   the ONLY functions that touch network

UNITS
-----
Every public function that takes or returns a position uses NANOMETRES, and
says so in the argument name (_nm). Voxel coordinates appear only inside
bbox_nm_to_voxel and fetch_roi, and are always paired with the resolution that
produced them. H01 reference Sec 5.3 records nm-vs-voxel mixing as the
pipeline's standard silent-failure mode.

CLOUDPATHS
----------
Verified against https://h01-release.storage.googleapis.com/data.html on
2026-09-01. The EM layer is 4 x 4 x 33 nm at mip 0; the c3 segmentation is
8 x 8 x 33 nm at mip 0; the subcompartment rendering was produced at
64 x 64 x 66 nm (H01 supplement, "Subcompartment rendering"). Resolutions are
NEVER hardcoded for the cutout: each layer reports its own via CloudVolume,
because a hardcoded resolution that is wrong produces a plausible-looking box
in the wrong place rather than an error.

SUBCOMPARTMENT LABEL OFFSETS -- read this before interpreting that layer
-----------------------------------------------------------------------
H01 supplement, "Subcompartment rendering": the rendered subcompartment
classes were incremented by 100, because the axon class label collided with
the conventional volumetric background label 0. Separately, under "Merge error
correction", skeleton components running inside a myelin sheath for more than
about 3 um had their predicted node class labels incremented by 1000.

So the raw values in this layer are NOT 0..5. decode_subcompartment strips
both offsets and returns the residual class index plus a myelin flag.

CAVEAT, and it is not resolved here: the supplement states the +100 offset for
the volumetric rendering explicitly, and the +1000 offset for the skeleton node
predictions. Whether +1000 propagates into the volumetric layer is not stated
in the text retrieved. The decoder handles both and reports the raw values it
actually saw, so you can settle it from the data rather than from my reading.

The class index -> name mapping (axon / dendrite / soma / astrocyte / AIS /
cilium) is NOT asserted here, because the integer assigned to each class is not
stated in the supplement text retrieved. Use subcompartment_report() to list
the values present in your ROI and confirm the mapping in Neuroglancer, which
displays the raw value on hover.

DEPENDENCIES: numpy, pandas. scikit-image only inside surface_from_mask.
cloud-volume only inside make_cloudvolume_reader. Pure ASCII, LF only.
"""

import json
import os

import numpy as np
import pandas as pd

MODULE_VERSION = "h01_spine_roi v1.0"

RELEASE = "20210601"

# Verified against data.html on 2026-09-01. See module docstring.
LAYERS = {
    "em": "precomputed://gs://h01-release/data/20210601/4nm_raw",
    "seg": "precomputed://gs://h01-release/data/20210601/c3",
    "subc": "precomputed://gs://h01-release/data/20210601/c3/subcompartments",
}

# Nominal mip-0 resolutions, for the size guard and for offline tests ONLY.
# fetch_roi reads the real resolution from each volume's own info.
NOMINAL_RESOLUTION_NM = {
    "em": (4.0, 4.0, 33.0),
    "seg": (8.0, 8.0, 33.0),
    "subc": (64.0, 64.0, 66.0),
}

BACKGROUND_LABEL = 0

# H01 supplement, "Subcompartment rendering" and "Merge error correction".
SUBC_CLASS_OFFSET = 100
SUBC_MYELIN_OFFSET = 1000

# Default half-width added to the spine's own bounding box, per axis, in nm.
DEFAULT_PAD_NM = 750.0

# Refuse a cutout larger than this without an explicit override. 2 GiB is about
# what a standard Colab runtime tolerates for one array plus a copy.
DEFAULT_MAX_BYTES = 2 * 1024 ** 3


class RoiError(RuntimeError):
    """Raised when an ROI request is malformed or would be unusably large."""


# --------------------------------------------------------------------------- #
# 1. Pure: picking the spine out of the node table                             #
# --------------------------------------------------------------------------- #
def spine_subframe(nodes, comp, sigma_id, include_base=True):
    """Return (spine_nodes, base_node, info) for one spine component.

    nodes     : the node table, with columns id, p, x, y, z (x/y/z in nm).
    comp      : the component labelling from sma_run.spine_components(nodes,
                mask); non-spine nodes carry -1.
    sigma_id  : int, the component index to extract.
    include_base : if True, also locate the SHAFT node that the spine's root
                hangs off. That node is not part of sigma -- it is returned
                separately -- but the spine is meaningless without knowing
                where it attaches.

    Returns
      spine_nodes : DataFrame, the rows of `nodes` with comp == sigma_id
      base_node   : one-row DataFrame or None; the non-spine parent of the
                    component's root, i.e. the shaft attachment point
      info        : dict, diagnostics (n_nodes, extent, whether the base was
                    found and whether the component has exactly one root)

    A component with more than one root means the spine subgraph is not a tree
    hanging off a single shaft node. That is reported, not silently averaged
    over, because it usually means two spines were merged by the labeller.
    """
    comp = np.asarray(comp)
    if comp.shape != (len(nodes),):
        raise RoiError("comp has shape %s, expected (%d,)" % (comp.shape, len(nodes)))
    sel = comp == int(sigma_id)
    if not sel.any():
        raise RoiError("no nodes with comp == %s; valid range is 0..%d"
                       % (sigma_id, int(comp.max())))

    spine_nodes = nodes.loc[sel].copy()
    ids = set(int(v) for v in spine_nodes["id"].to_numpy())

    # A root of the component is a node whose parent is outside the component.
    par = spine_nodes["p"].to_numpy(dtype=np.int64)
    is_root = np.array([int(p) not in ids for p in par])
    root_ids = [int(v) for v in spine_nodes["id"].to_numpy()[is_root]]

    base_node = None
    if include_base and len(root_ids) >= 1:
        root_row = spine_nodes.loc[spine_nodes["id"] == root_ids[0]]
        base_id = int(root_row["p"].iloc[0])
        hit = nodes.loc[nodes["id"] == base_id]
        if len(hit) == 1:
            base_node = hit.copy()

    xyz = spine_nodes[["x", "y", "z"]].to_numpy(dtype=float)
    info = {
        "sigma_id": int(sigma_id),
        "n_nodes": int(len(spine_nodes)),
        "n_roots": int(len(root_ids)),
        "root_ids": root_ids,
        "single_root": bool(len(root_ids) == 1),
        "base_found": bool(base_node is not None),
        "base_id": None if base_node is None else int(base_node["id"].iloc[0]),
        # np.ptp(), not ndarray.ptp(): the method was removed in numpy 2.0 and
        # Colab may be on either side of that line.
        "extent_nm": {ax: float(np.ptp(xyz[:, i]))
                      for i, ax in enumerate(("x", "y", "z"))},
        "centroid_nm": {ax: float(xyz[:, i].mean())
                        for i, ax in enumerate(("x", "y", "z"))},
    }
    return spine_nodes, base_node, info


def pick_spine(nodes, comp, strategy="median_size", sigma_id=None,
               near_nm=None, rng_seed=0):
    """Choose one sigma index. Returns (sigma_id, why).

    strategy:
      'explicit'     use sigma_id as given
      'median_size'  the component whose node count is closest to the median
      'largest'      the component with the most nodes
      'nearest'      the component whose centroid is closest to near_nm
      'random'       uniform over components, seeded by rng_seed

    'median_size' is the default because a typical spine is the useful pilot;
    the largest component is disproportionately likely to be a labeller
    artefact (a merged pair, or a shaft stub that survived demotion).
    """
    comp = np.asarray(comp)
    valid = comp >= 0
    if not valid.any():
        raise RoiError("no spine components in comp")
    sizes = pd.Series(comp[valid]).value_counts().sort_index()

    if strategy == "explicit":
        if sigma_id is None:
            raise RoiError("strategy='explicit' requires sigma_id")
        if int(sigma_id) not in sizes.index:
            raise RoiError("sigma_id %s is not a component" % sigma_id)
        return int(sigma_id), "explicit"

    if strategy == "largest":
        sid = int(sizes.idxmax())
        return sid, "largest component (%d nodes)" % int(sizes.max())

    if strategy == "median_size":
        med = float(sizes.median())
        sid = int((sizes - med).abs().idxmin())
        return sid, "size %d closest to median %.1f" % (int(sizes.loc[sid]), med)

    if strategy == "random":
        rng = np.random.default_rng(rng_seed)
        sid = int(rng.choice(sizes.index.to_numpy()))
        return sid, "random, seed %d" % rng_seed

    if strategy == "nearest":
        if near_nm is None:
            raise RoiError("strategy='nearest' requires near_nm=(x, y, z)")
        target = np.asarray(near_nm, dtype=float)
        xyz = nodes[["x", "y", "z"]].to_numpy(dtype=float)
        best, best_d = None, np.inf
        for sid in sizes.index:
            c = xyz[comp == sid].mean(axis=0)
            d = float(np.linalg.norm(c - target))
            if d < best_d:
                best, best_d = int(sid), d
        return best, "centroid %.0f nm from target" % best_d

    raise RoiError("unknown strategy %r" % strategy)


# --------------------------------------------------------------------------- #
# 2. Pure: the ROI box, in nanometres                                          #
# --------------------------------------------------------------------------- #
def roi_bbox_nm(points_nm, pad_nm=DEFAULT_PAD_NM, isotropic=False):
    """Axis-aligned bounding box of `points_nm`, grown by pad_nm on each side.

    points_nm : (N, 3) array of positions in nm.
    pad_nm    : scalar or per-axis (3,) padding, in nm, added to BOTH sides.
    isotropic : if True, expand the shorter axes so the box is a cube. Useful
                when the spine is nearly planar and a thin box would clip the
                surrounding neuropil out of the EM view.

    Returns (lo_nm, hi_nm), each a (3,) float array.
    """
    pts = np.asarray(points_nm, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise RoiError("points_nm must be (N, 3), got %s" % (pts.shape,))
    if len(pts) == 0:
        raise RoiError("points_nm is empty")

    pad = np.broadcast_to(np.asarray(pad_nm, dtype=float), (3,)).astype(float)
    lo = pts.min(axis=0) - pad
    hi = pts.max(axis=0) + pad

    if isotropic:
        side = float((hi - lo).max())
        mid = 0.5 * (lo + hi)
        lo = mid - 0.5 * side
        hi = mid + 0.5 * side
    return lo, hi


def bbox_nm_to_voxel(bbox_nm, resolution_nm, volume_bounds_vox=None,
                     voxel_offset=(0, 0, 0)):
    """Convert an nm box to an inclusive-exclusive voxel box for one layer.

    resolution_nm     : (3,) nm per voxel for the layer AND mip being fetched.
    volume_bounds_vox : optional ((x0,y0,z0), (x1,y1,z1)) clip; pass the
                        layer's own bounds so a box near the volume edge is
                        clipped rather than returning an out-of-range request.
    voxel_offset      : the layer's voxel_offset, subtracted before flooring.

    Returns (lo_vox, hi_vox) as int arrays, with hi exclusive. lo is floored
    and hi is ceiled, so the voxel box always CONTAINS the nm box -- never
    rounds inward and silently clips the spine.
    """
    lo_nm, hi_nm = (np.asarray(b, dtype=float) for b in bbox_nm)
    res = np.asarray(resolution_nm, dtype=float)
    if res.shape != (3,) or not np.all(res > 0):
        raise RoiError("resolution_nm must be 3 positive values, got %s" % (res,))
    off = np.asarray(voxel_offset, dtype=np.int64)

    lo_v = np.floor(lo_nm / res).astype(np.int64)
    hi_v = np.ceil(hi_nm / res).astype(np.int64)
    hi_v = np.maximum(hi_v, lo_v + 1)          # never a zero-thickness box

    if volume_bounds_vox is not None:
        b_lo, b_hi = (np.asarray(b, dtype=np.int64) for b in volume_bounds_vox)
        # A true interval intersection, NOT np.clip of each endpoint into the
        # bounds. Clipping endpoints independently slides a disjoint box onto
        # the volume edge and returns a valid-looking request for the wrong
        # place; intersection collapses it, which is what must raise.
        new_lo = np.maximum(lo_v, b_lo)
        new_hi = np.minimum(hi_v, b_hi)
        if np.any(new_hi <= new_lo):
            raise RoiError(
                "ROI %s..%s does not intersect the volume bounds %s..%s "
                "(axis %s is empty). Check the units on the node table."
                % (lo_v.tolist(), hi_v.tolist(), b_lo.tolist(), b_hi.tolist(),
                   np.where(new_hi <= new_lo)[0].tolist()))
        lo_v, hi_v = new_lo, new_hi
    _ = off        # kept in the signature; offsets are applied by CloudVolume
    return lo_v, hi_v


def estimate_bytes(lo_vox, hi_vox, dtype):
    """Bytes a cutout of this voxel box would occupy in memory."""
    shape = np.asarray(hi_vox, dtype=np.int64) - np.asarray(lo_vox, dtype=np.int64)
    if np.any(shape <= 0):
        raise RoiError("non-positive cutout shape %s" % shape.tolist())
    return int(np.prod(shape.astype(object)) * np.dtype(dtype).itemsize)


def voxel_to_nm(idx_vox, resolution_nm, centre=True):
    """Voxel indices -> nm positions. centre=True returns the voxel centre."""
    idx = np.asarray(idx_vox, dtype=float)
    res = np.asarray(resolution_nm, dtype=float)
    return (idx + 0.5) * res if centre else idx * res


# --------------------------------------------------------------------------- #
# 3. Pure: reading the fetched arrays                                          #
# --------------------------------------------------------------------------- #
def points_to_local_index(points_nm, lo_vox, resolution_nm):
    """nm positions -> integer indices into an array whose [0,0,0] is lo_vox.

    Returns (idx, inside) where idx is (N, 3) int64 and `inside` is the boolean
    mask of points that actually fall in the array. Points outside are NOT
    clipped -- clipping would silently sample the wrong voxel.
    """
    pts = np.asarray(points_nm, dtype=float)
    res = np.asarray(resolution_nm, dtype=float)
    lo = np.asarray(lo_vox, dtype=np.int64)
    idx = np.floor(pts / res).astype(np.int64) - lo
    return idx, np.ones(len(pts), dtype=bool)    # bounds checked by the caller


def segids_at_points(arr, points_nm, lo_vox, resolution_nm, fill=BACKGROUND_LABEL):
    """Sample a fetched 3D array at nm positions. Returns (values, inside).

    This is the whole point of the bbox approach: once the c3 cutout is in
    memory, every node's segid is an array index, not a network round trip.
    Points falling outside the cutout get `fill` and inside=False; they are
    NOT clipped to the border.
    """
    a = np.asarray(arr)
    if a.ndim != 3:
        raise RoiError("arr must be 3D, got shape %s" % (a.shape,))
    idx, _ = points_to_local_index(points_nm, lo_vox, resolution_nm)
    inside = np.all((idx >= 0) & (idx < np.asarray(a.shape, dtype=np.int64)), axis=1)
    out = np.full(len(idx), fill, dtype=a.dtype)
    if inside.any():
        good = idx[inside]
        out[inside] = a[good[:, 0], good[:, 1], good[:, 2]]
    return out, inside


def mask_from_segids(seg_arr, segids):
    """Boolean mask of the voxels belonging to any id in `segids`."""
    a = np.asarray(seg_arr)
    wanted = np.asarray(sorted(set(int(s) for s in segids)), dtype=a.dtype)
    if wanted.size == 0:
        raise RoiError("empty segid set -- nothing to mask")
    return np.isin(a, wanted)


def spine_component_audit(mask, points_nm, lo_vox, resolution_nm,
                          connectivity=3):
    """Which components of `mask` actually contain skeleton nodes?

    A spine mask with more than one connected component has exactly two
    possible causes, and they demand opposite responses:

      (a) the nearest-node partition mis-assigned a lump of shaft. Such a
          component contains NO node of the sigma -- it was captured only
          because every one of its voxels happened to be nearer to a spine
          node than to any shaft node, which happens where the skeleton
          samples the shaft sparsely. It is not part of the spine and should
          not be drawn as one.

      (b) the segmentation genuinely splits the spine. Then TWO OR MORE
          components each contain nodes of the sigma, and joining them is a
          defensible interpolation.

    Telling them apart requires the skeleton, so this is the one place the
    node positions re-enter after mask_local_to_spine.

    Returns dict with per-component size and node count, plus which labels
    carry nodes. Nodes landing on background (label 0) are counted separately:
    that means the skeleton runs outside its own segment, which is worth
    knowing on its own.
    """
    from scipy import ndimage

    m = np.asarray(mask, dtype=bool)
    struct = ndimage.generate_binary_structure(3, connectivity)
    lab, n = ndimage.label(m, structure=struct)

    res = np.asarray(resolution_nm, dtype=float)
    lo = np.asarray(lo_vox, dtype=np.int64)
    idx = np.floor(np.asarray(points_nm, dtype=float) / res).astype(np.int64) - lo
    inside = np.all((idx >= 0) & (idx < np.asarray(m.shape, dtype=np.int64)), axis=1)

    node_label = np.full(len(idx), -1, dtype=np.int64)
    if inside.any():
        g = idx[inside]
        node_label[inside] = lab[g[:, 0], g[:, 1], g[:, 2]]

    comps = []
    for i in range(1, n + 1):
        comps.append({"label": int(i),
                      "size_voxels": int((lab == i).sum()),
                      "n_nodes": int((node_label == i).sum())})
    comps.sort(key=lambda c: c["size_voxels"], reverse=True)

    with_nodes = [c["label"] for c in comps if c["n_nodes"] > 0]
    return {
        "n_components": int(n),
        "components": comps,
        "labels_with_nodes": with_nodes,
        "labels_without_nodes": [c["label"] for c in comps if c["n_nodes"] == 0],
        "n_nodes_on_background": int((node_label == 0).sum()),
        "n_nodes_outside_roi": int((~inside).sum()),
        "verdict": ("single component" if n <= 1 else
                    ("segmentation splits the spine (%d components carry "
                     "nodes) -- bridging is defensible" % len(with_nodes))
                    if len(with_nodes) > 1 else
                    "one real component; %d component(s) carry no node and are "
                    "probably mis-assigned shaft" % (n - len(with_nodes))),
        "_labels": lab,
    }


def resolve_spine_mask(mask, points_nm, lo_vox, resolution_nm, connectivity=3):
    """Split `mask` into node-bearing components and node-free orphans.

    Returns (core_mask, orphan_mask, audit). The orphans are NOT deleted --
    they are handed back so they can be redrawn as shaft context rather than
    silently vanishing.
    """
    audit = spine_component_audit(mask, points_nm, lo_vox, resolution_nm,
                                  connectivity=connectivity)
    lab = audit.pop("_labels")
    m = np.asarray(mask, dtype=bool)
    if audit["n_components"] <= 1 or not audit["labels_with_nodes"]:
        return m.copy(), np.zeros_like(m), audit
    core = np.isin(lab, audit["labels_with_nodes"]) & m
    orphan = m & ~core
    return core, orphan, audit


def bridge_components(mask, resolution_nm, max_gap_nm, lo_vox=None,
                      connectivity=3, **span_kwargs):
    """Join the components of ONE mask to each other, largest-anchor first.

    For the case where the segmentation splits a single spine: each smaller
    component is bridged to the growing anchor with bridge_masks, so the span
    of every connector is sized from the local calibre at that particular gap
    rather than from one global number.

    Returns (bridge_mask, info). Components further than max_gap_nm from the
    anchor are left unbridged and listed in info, never force-joined.
    """
    from scipy import ndimage

    m = np.asarray(mask, dtype=bool)
    struct = ndimage.generate_binary_structure(3, connectivity)
    lab, n = ndimage.label(m, structure=struct)
    total = np.zeros_like(m)
    if n <= 1:
        return total, {"bridged": False, "reason": "mask is already one "
                       "component", "n_components": int(n), "joins": []}

    sizes = [(int((lab == i).sum()), i) for i in range(1, n + 1)]
    sizes.sort(reverse=True)
    anchor = (lab == sizes[0][1])
    joins = []
    for size, i in sizes[1:]:
        comp = (lab == i)
        br, info = bridge_masks(anchor, comp, resolution_nm, max_gap_nm,
                                lo_vox=lo_vox, **span_kwargs)
        joins.append({"component_label": int(i), "size_voxels": int(size),
                      "bridged": bool(info.get("bridged")),
                      "gap_face_nm": info.get("gap_face_nm"),
                      "radius_nm": info.get("radius_nm"),
                      "n_bridge_voxels": int(info.get("n_bridge_voxels", 0)),
                      "reason": info.get("reason")})
        if info.get("bridged"):
            total |= br
            anchor = anchor | comp | br
        else:
            anchor = anchor | comp if info.get("reason") == "already touching" \
                else anchor

    joined = component_report(m | total, connectivity=connectivity)
    return total, {
        "bridged": bool(total.any()),
        "n_components": int(n),
        "joins": joins,
        "n_bridge_voxels": int(total.sum()),
        "components_after": joined["n_components"],
        "WARNING": "these voxels are INTERPOLATED and are not present in the "
                   "released c3 segmentation",
    }


def mask_contact_report(mask_a, mask_b, resolution_nm, lo_vox=None):
    """How far apart are two masks, measured BETWEEN VOXELS, not via skeleton.

    Everything here is computed on the voxel masks themselves. The skeleton is
    used nowhere in this function: it decided which voxels belong to which
    object (mask_local_to_spine), but the separation is a property of the
    voxels, so it is measured on them.

    THREE DISTINCT NOTIONS, deliberately not conflated:

      touching              26-connectivity (the SAME test
                            split_excluded_by_contact uses, so the two can
                            never disagree about whether a component is
                            attached).
      gap_center_nm         anisotropic Euclidean distance between the nearest
                            voxel CENTRES. One voxel step (8 nm in x for c3)
                            even for face-adjacent masks -- centres of
                            touching voxels are not at zero separation.
      gap_face_nm           the EMPTY SPAN between the two voxels' surfaces,
                            i.e. how much tissue is actually missing. This is
                            the number that answers "is there a hole here".

    gap_face_nm uses the support function of an axis-aligned box: for a voxel
    of size `res` and a unit direction u, the half-extent along u is
    0.5 * sum_i |u_i| * res_i, so

        gap_face = max(0, gap_center - sum_i |u_i| * res_i)          (1)

    where u is the unit vector between the closest pair of voxel centres.
    Equation (1) returns exactly 0 for face-, edge- AND corner-adjacent voxels
    at any anisotropy, and exactly n_empty * res for n_empty voxels of clear
    space along an axis. Verified on both cases.

    lo_vox : optional cutout origin; when given, the closest-pair coordinates
             are also returned in GLOBAL nm, which is what the figures need.
    """
    from scipy import ndimage

    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    if a.shape != b.shape:
        raise RoiError("masks differ in shape: %s vs %s" % (a.shape, b.shape))
    res = np.asarray(resolution_nm, dtype=float)

    if not a.any() or not b.any():
        return {"gap_center_nm": float("inf"), "gap_face_nm": float("inf"),
                "gap_nm": float("inf"), "touching": False,
                "n_contact_voxels": 0, "closest_pair_vox": None,
                "closest_pair_nm": None, "note": "one of the masks is empty"}

    struct = ndimage.generate_binary_structure(3, 3)
    touching = bool((ndimage.binary_dilation(a, structure=struct) & b).any())

    # return_indices gives, for every voxel, the index of the nearest voxel OF
    # a -- which is how the closest PAIR is recovered, not just the distance.
    d_to_a, near_a = ndimage.distance_transform_edt(
        ~a, sampling=res, return_indices=True)
    b_idx = np.argwhere(b)
    d_at_b = d_to_a[b]
    k = int(np.argmin(d_at_b))
    b_star = b_idx[k]
    a_star = np.array([near_a[i][tuple(b_star)] for i in range(3)], dtype=np.int64)

    d_center = float(d_at_b[k])
    delta_nm = (b_star.astype(float) - a_star.astype(float)) * res
    if d_center > 0:
        u = delta_nm / np.linalg.norm(delta_nm)
        gap_face = max(0.0, d_center - float(np.abs(u) @ res))       # eq. (1)
    else:
        gap_face = 0.0
    if touching:
        gap_face = 0.0

    n_contact = int((b & (d_to_a <= d_center + 1e-9)).sum())
    out = {
        "gap_center_nm": d_center,
        "gap_face_nm": gap_face,
        "gap_nm": gap_face,          # the one to use for bridging decisions
        "touching": touching,
        "n_contact_voxels": n_contact,
        "closest_pair_vox": [a_star.tolist(), b_star.tolist()],
        "note": "measured between voxels (not the skeleton); gap_face_nm is "
                "the empty span between voxel surfaces, anisotropic sampling "
                "%s nm/voxel" % res.tolist(),
    }
    if lo_vox is not None:
        lo = np.asarray(lo_vox, dtype=float)
        out["closest_pair_nm"] = [
            (((a_star + lo) + 0.5) * res).tolist(),
            (((b_star + lo) + 0.5) * res).tolist()]
    else:
        out["closest_pair_nm"] = None
    return out


def local_calibre_nm(mask, center_vox, window_nm, resolution_nm,
                     estimator="equivalent", axis_nm=None, percentile=90.0):
    """AVERAGE local radius of `mask` within window_nm of a point. Voxel-based.

    Used to size the bridge: a connector between a spine and its shaft should
    be about as thick as the spine actually is where it runs out.

    THREE ESTIMATORS, all returned, one selected by `estimator`:

      'equivalent' (DEFAULT) -- the average calibre over the window.
          Mean cross-sectional area = (voxel volume in the window) / (axial
          length of the window), then r = sqrt(A / pi). This is an AVERAGE
          over the length sampled, which is what "average calibre over the
          last N nm" means. Validated against a cylinder of known radius:
          118.0 nm recovered for a true 120.0 nm, a 1.7% error.

      'medial' -- max of the Euclidean distance transform inside the mask,
          i.e. the largest inscribed sphere. Exact for a clean tube (120.3 nm
          on the same cylinder) but it is a single voxel, so a local bulge
          such as the spine head inflates it.

      'percentile' -- the q-th percentile of the interior EDT. NOT recommended
          and NOT the default: it is systematically biased low, because for a
          cylinder the fraction of voxels with EDT >= r is (1 - r/R)^2, so
          q=90 lands at 0.684 R. Measured 82.4 nm on the 120 nm cylinder, a
          31% underestimate. Kept only so the bias is visible rather than
          hidden.

    axis_nm : direction along which to measure the axial length. When
        bridge_masks calls this it passes the spine -> shaft direction, so the
        length is measured along the direction the bridge will run. If None,
        the first principal axis of the selected voxels is used.

    Returns dict with radius_nm (the selected estimator), all three estimates
    for comparison, the window actually used, and the voxel count.
    """
    from scipy import ndimage

    m = np.asarray(mask, dtype=bool)
    res = np.asarray(resolution_nm, dtype=float)
    if not m.any():
        raise RoiError("mask is empty -- no calibre to measure")

    idx = np.argwhere(m)
    c = np.asarray(center_vox, dtype=float)
    d = np.linalg.norm((idx.astype(float) - c) * res, axis=1)

    used_window = float(window_nm)
    sel = d <= used_window
    widened = False
    if sel.sum() < 2:
        used_window = float(d.min() + 2.0 * np.linalg.norm(res))
        sel = d <= used_window
        widened = True

    sub = idx[sel]
    pts = sub.astype(float) * res

    # -- medial and percentile, from the interior distance transform
    edt_in = ndimage.distance_transform_edt(m, sampling=res)
    vals = edt_in[tuple(sub.T)]
    r_medial = float(vals.max())
    r_pct = float(np.percentile(vals, percentile))

    # -- equivalent: mean cross-section = volume / axial length
    if axis_nm is not None:
        u = np.asarray(axis_nm, dtype=float)
        nu = float(np.linalg.norm(u))
        u = u / nu if nu > 0 else np.array([1.0, 0.0, 0.0])
    elif len(pts) >= 3:
        centred = pts - pts.mean(axis=0)
        u = np.linalg.svd(centred, full_matrices=False)[2][0]
    else:
        u = np.array([1.0, 0.0, 0.0])

    proj = (pts - pts.mean(axis=0)) @ u
    # One voxel step along u, so a single-slice selection still has length > 0.
    L = float(proj.max() - proj.min()) + float(np.abs(u) @ res)
    V = float(sel.sum()) * float(np.prod(res))
    r_eq = float(np.sqrt((V / L) / np.pi)) if L > 0 else float("nan")

    chosen = {"equivalent": r_eq, "medial": r_medial,
              "percentile": r_pct}.get(estimator)
    if chosen is None:
        raise RoiError("unknown estimator %r; use 'equivalent', 'medial' or "
                       "'percentile'" % estimator)

    return {
        "radius_nm": chosen,
        "diameter_nm": 2.0 * chosen,
        "estimator": estimator,
        "radius_equivalent_nm": r_eq,
        "radius_medial_nm": r_medial,
        "radius_percentile_nm": r_pct,
        "percentile": float(percentile),
        "axial_length_nm": L,
        "volume_nm3": V,
        "window_nm": used_window,
        "window_widened": widened,
        "n_voxels_in_window": int(sel.sum()),
        "axis_unit": [float(v) for v in u],
        "method": "equivalent = sqrt((volume/axial length)/pi), averaged over "
                  "the window; medial = max interior EDT; percentile is "
                  "biased low and kept only for comparison",
    }


def bridge_masks(mask_a, mask_b, resolution_nm, max_gap_nm,
                 span_mode="local_calibre", span_window_nm=200.0,
                 span_estimator="equivalent", span_percentile=90.0,
                 span_radius_nm=None, lo_vox=None):
    """Join two masks with a connector of realistic thickness. OPT-IN ONLY.

    READ THIS BEFORE USING IT. The voxels this returns are NOT in the released
    segmentation. They are interpolated -- an assertion by you that two objects
    the segmentation left separate are in fact one. That may well be right
    (H01's c3 agglomeration favours split errors, and the supplement reports
    roughly a third of one layer-2 pyramidal cell's spines were not part of
    the same agglomerated segment), but it is your assertion, not the data's,
    and anything measured on a bridged mask -- volume, surface area, neck
    diameter -- inherits it.

    GEOMETRY. The bridge is a CAPSULE: every voxel within `radius` of the line
    segment joining the closest pair of voxel centres, excluding voxels already
    in either mask. Its radius is chosen by `span_mode`:

      'local_calibre' (default)
          radius = local_calibre_nm(mask_a, a*, span_window_nm,
                                    percentile=span_percentile).radius_nm
          i.e. how thick mask_a (the SPINE) actually is over the last
          `span_window_nm` of itself approaching the gap. A connector between
          a spine and its shaft then has the spine's own calibre rather than
          an arbitrary width.
      'explicit'
          radius = span_radius_nm, for when you want to fix it yourself.
      'minimal'
          the old thin corridor: voxels where d(v,a) + d(v,b) is within one
          voxel of the minimum separation. Adds the least material possible.

    The gap tested against max_gap_nm is gap_face_nm -- the EMPTY SPAN between
    voxel surfaces -- not the centre-to-centre distance, because the empty span
    is the amount of tissue actually being invented.

    Returns (bridge_mask, info). bridge_mask is disjoint from both inputs and
    is empty if the masks already touch or the gap exceeds max_gap_nm; in that
    case info says so and nothing is invented.
    """
    from scipy import ndimage

    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    if a.shape != b.shape:
        raise RoiError("masks differ in shape: %s vs %s" % (a.shape, b.shape))
    res = np.asarray(resolution_nm, dtype=float)
    empty = np.zeros_like(a)

    if not a.any() or not b.any():
        return empty, {"bridged": False, "reason": "a mask is empty",
                       "gap_face_nm": float("inf"), "n_bridge_voxels": 0}

    contact = mask_contact_report(a, b, res, lo_vox=lo_vox)
    if contact["touching"]:
        return empty, {"bridged": False, "reason": "already touching",
                       "gap_face_nm": 0.0, "n_bridge_voxels": 0,
                       "contact": contact}
    if contact["gap_face_nm"] > float(max_gap_nm):
        return empty, {
            "bridged": False,
            "reason": "empty span %.1f nm exceeds max_gap_nm %.1f -- nothing "
                      "invented" % (contact["gap_face_nm"], float(max_gap_nm)),
            "gap_face_nm": contact["gap_face_nm"],
            "n_bridge_voxels": 0, "contact": contact}

    a_star, b_star = (np.asarray(p, dtype=float)
                      for p in contact["closest_pair_vox"])

    if span_mode == "minimal":
        d_a = ndimage.distance_transform_edt(~a, sampling=res)
        d_b = ndimage.distance_transform_edt(~b, sampling=res)
        gap_c = contact["gap_center_nm"]
        corridor = (d_a + d_b <= gap_c + float(res.min())) & ~(a | b)
        info = {"bridged": bool(corridor.any()), "span_mode": "minimal",
                "radius_nm": None, "calibre": None}
        bridge = corridor
    else:
        if span_mode == "explicit":
            if span_radius_nm is None:
                raise RoiError("span_mode='explicit' requires span_radius_nm")
            radius = float(span_radius_nm)
            cal = {"method": "explicit", "radius_nm": radius}
        elif span_mode == "local_calibre":
            # Measure the axial length along the direction the bridge will run,
            # so "average calibre over the last N nm" is averaged along the
            # spine's approach to the gap rather than some unrelated axis.
            cal = local_calibre_nm(a, a_star, span_window_nm, res,
                                   estimator=span_estimator,
                                   axis_nm=(b_star - a_star) * res,
                                   percentile=span_percentile)
            radius = float(cal["radius_nm"])
        else:
            raise RoiError("unknown span_mode %r" % span_mode)

        # Capsule around the segment a* -> b*, built only inside a bounding box
        # around it so the distance evaluation stays cheap on a large ROI.
        p0 = a_star * res
        p1 = b_star * res
        seg = p1 - p0
        seg_len2 = float(seg @ seg)

        margin = np.ceil((radius + float(res.max())) / res).astype(np.int64) + 1
        lo_b = np.maximum(np.floor(np.minimum(a_star, b_star)).astype(np.int64)
                          - margin, 0)
        hi_b = np.minimum(np.ceil(np.maximum(a_star, b_star)).astype(np.int64)
                          + margin + 1, np.asarray(a.shape, dtype=np.int64))

        gx = np.arange(lo_b[0], hi_b[0]); gy = np.arange(lo_b[1], hi_b[1])
        gz = np.arange(lo_b[2], hi_b[2])
        X, Y, Z = np.meshgrid(gx, gy, gz, indexing="ij")
        pts = np.stack([X * res[0], Y * res[1], Z * res[2]], axis=-1)

        w = pts - p0
        t = (w @ seg) / seg_len2 if seg_len2 > 0 else np.zeros(w.shape[:-1])
        t = np.clip(t, 0.0, 1.0)
        closest = p0 + t[..., None] * seg
        dist = np.linalg.norm(pts - closest, axis=-1)

        bridge = np.zeros_like(a)
        bridge[lo_b[0]:hi_b[0], lo_b[1]:hi_b[1], lo_b[2]:hi_b[2]] = dist <= radius
        bridge &= ~(a | b)
        info = {"bridged": bool(bridge.any()), "span_mode": span_mode,
                "radius_nm": radius, "calibre": cal}

    info.update({
        "reason": "capsule of radius %s nm along the closest-pair segment"
                  % (info.get("radius_nm")) if span_mode != "minimal"
                  else "minimal corridor along the near-shortest path",
        "gap_face_nm": contact["gap_face_nm"],
        "gap_center_nm": contact["gap_center_nm"],
        "n_bridge_voxels": int(bridge.sum()),
        "bridge_volume_nm3": float(bridge.sum() * np.prod(res)),
        "max_gap_nm": float(max_gap_nm),
        "contact": contact,
        "WARNING": "these voxels are INTERPOLATED and are not present in the "
                   "released c3 segmentation; any measurement on the bridged "
                   "mask inherits that assumption",
    })
    return bridge, info


def component_report(mask, connectivity=3):
    """Connected-component sizes of a boolean mask. Diagnostic only.

    A spine mask with more than one component is a red flag: either the pad
    caught an unrelated piece of the same segment, or the spine is genuinely
    split by the segmentation. Reported, never auto-merged or auto-pruned.
    """
    from scipy import ndimage

    m = np.asarray(mask, dtype=bool)
    struct = ndimage.generate_binary_structure(3, connectivity)
    lab, n = ndimage.label(m, structure=struct)
    sizes = [int((lab == i).sum()) for i in range(1, n + 1)]
    sizes.sort(reverse=True)
    return {
        "n_components": int(n),
        "sizes_voxels": sizes,
        "largest_fraction": (sizes[0] / sum(sizes)) if sizes else float("nan"),
    }


def split_excluded_by_contact(kept_mask, excluded_mask, connectivity=3):
    """Split the excluded voxels by whether they TOUCH the spine.

    WHY: mask_local_to_spine's excluded half is not homogeneous. Some of it is
    the parent shaft, physically continuous with the spine at its neck -- that
    is the piece worth drawing, because it shows where the spine attaches.
    The rest is a different matter entirely: disconnected blobs of the SAME
    segment id that merely fall inside the padded box. A neighbouring stretch
    of the same dendrite looping back through the ROI, or another spine on the
    same shaft, produces exactly that -- a free-floating lump with no contact
    with the spine at all.

    Lumping the two together under one label and one colour makes a detached
    lump look like part of the structure. This function separates them by
    26-connectivity: an excluded component is 'shaft-connected' iff at least
    one of its voxels is adjacent to a voxel of `kept_mask`.

    Returns (connected_mask, detached_mask, info). The two are disjoint and
    their union is exactly `excluded_mask`.

    NOTE: adjacency here is at the SEGMENTATION voxel size (8 x 8 x 33 nm for
    c3), so two objects passing within one voxel of each other read as
    touching. This is a connectivity test on the released segmentation, not a
    claim about cytoplasmic continuity in the tissue.
    """
    from scipy import ndimage

    kept = np.asarray(kept_mask, dtype=bool)
    excl = np.asarray(excluded_mask, dtype=bool)
    if kept.shape != excl.shape:
        raise RoiError("kept %s and excluded %s differ in shape"
                       % (kept.shape, excl.shape))

    empty = np.zeros_like(excl)
    if not excl.any():
        return empty, empty.copy(), {"n_excluded_components": 0,
                                     "n_connected_components": 0,
                                     "n_detached_components": 0,
                                     "n_voxels_connected": 0,
                                     "n_voxels_detached": 0,
                                     "detached_sizes_voxels": []}

    struct = ndimage.generate_binary_structure(3, connectivity)
    lab, n = ndimage.label(excl, structure=struct)

    # A component is connected iff it intersects a one-voxel halo around kept.
    halo = ndimage.binary_dilation(kept, structure=struct)
    touching = sorted(set(int(v) for v in np.unique(lab[halo & excl])) - {0})

    connected = np.isin(lab, touching) & excl if touching else empty.copy()
    detached = excl & ~connected

    det_lab, det_n = ndimage.label(detached, structure=struct)
    det_sizes = sorted((int((det_lab == i).sum()) for i in range(1, det_n + 1)),
                       reverse=True)

    info = {
        "n_excluded_components": int(n),
        "n_connected_components": int(len(touching)),
        "n_detached_components": int(det_n),
        "n_voxels_connected": int(connected.sum()),
        "n_voxels_detached": int(detached.sum()),
        "detached_sizes_voxels": det_sizes,
        "connectivity": "26-neighbour at the segmentation voxel size",
    }
    return connected, detached, info


def mask_local_to_spine(mask, lo_vox, resolution_nm, nodes, spine_node_ids):
    """Split a shared-segid mask at the spine/shaft boundary.

    WHY THIS EXISTS: a 'primary' spine (G0's term -- not foreign, not mixed)
    shares its c3 segid with the parent dendrite by construction; that is what
    'primary' means. mask_from_segids therefore returns every voxel in the ROI
    carrying that id, shaft included -- there is no id-level distinction
    between "spine" and "the shaft it sits on". Visually this shows up as a
    large extra blob glued to the spine wherever the padded box happens to
    reach far enough along the shaft to catch another chunk of it.

    THE FIX: partition mask voxel-by-voxel using the skeleton's OWN, already-
    decided per-node classification (spine_labeller + shaft_continuation,
    read at CELL 5 -- not re-decided here). Each True voxel is assigned to
    whichever node of the FULL reconstruction is nearest to it in nm; the
    voxel is kept only if that nearest node's id is one of `spine_node_ids`.
    This is a nearest-neighbour (Voronoi) split, not a new spine/shaft
    criterion -- it borrows the boundary the labeller already drew on the
    skeleton and applies it to the volumetric mask.

    nodes must be the FULL node table (every node of the reconstruction, not
    just this sigma's): the shaft's own nearby nodes are exactly what the
    voxels need to be compared against to find the cut.

    Returns (kept_mask, excluded_mask) -- both same shape as `mask`, disjoint,
    union equal to `mask`. Caller decides whether to plot the excluded part
    as context or drop it.

    LIMITATION, stated rather than hidden: this is a geometric approximation.
    The true spine/shaft boundary is not crisply defined at the voxel level;
    this convention says a voxel belongs to whichever labelled node's position
    it is closest to. Near the attachment point that is exactly the ambiguous
    region, so a thin rind of true shaft can still be kept, and a thin rind of
    true spine near the neck can still be excluded. It removes the gross
    shaft blob; it is not a substitute for the skeleton-level partition.
    """
    from scipy.spatial import cKDTree

    m = np.asarray(mask, dtype=bool)
    idx = np.argwhere(m)
    if len(idx) == 0:
        return m.copy(), np.zeros_like(m)

    all_xyz = nodes[["x", "y", "z"]].to_numpy(dtype=float)
    all_ids = nodes["id"].to_numpy(dtype=np.int64)
    if len(all_xyz) == 0:
        raise RoiError("nodes is empty -- cannot partition by nearest node")
    tree = cKDTree(all_xyz)

    res = np.asarray(resolution_nm, dtype=float)
    lo = np.asarray(lo_vox, dtype=float)
    voxel_nm = (idx.astype(float) + lo + 0.5) * res      # voxel centres, GLOBAL nm

    _, nn = tree.query(voxel_nm, k=1)
    nearest_ids = all_ids[nn]
    spine_set = np.asarray(sorted(set(int(v) for v in spine_node_ids)), dtype=np.int64)
    keep = np.isin(nearest_ids, spine_set)

    kept = np.zeros_like(m)
    excl = np.zeros_like(m)
    kept[tuple(idx[keep].T)] = True
    excl[tuple(idx[~keep].T)] = True
    return kept, excl


def decode_subcompartment(arr):
    """Strip the +100 rendering offset and the +1000 myelin offset.

    Returns dict with:
      class_index  int16 array; -1 where the raw value was background (0)
      myelinated   bool array; True where the raw value carried +1000
      raw_values   sorted list of the distinct raw values present
      note         a string recording that the class-index -> name mapping is
                   NOT asserted by this module

    See the module docstring: the +100 offset is stated for the volumetric
    rendering; the +1000 offset is stated for the skeleton node predictions and
    its propagation into this layer is NOT confirmed. Read raw_values.
    """
    a = np.asarray(arr)
    raw = np.unique(a)
    v = a.astype(np.int64)
    myelin = v >= (SUBC_MYELIN_OFFSET + SUBC_CLASS_OFFSET)
    v = np.where(myelin, v - SUBC_MYELIN_OFFSET, v)
    cls = np.where(v >= SUBC_CLASS_OFFSET, v - SUBC_CLASS_OFFSET, -1)
    cls = np.where(a == BACKGROUND_LABEL, -1, cls)
    return {
        "class_index": cls.astype(np.int16),
        "myelinated": myelin,
        "raw_values": [int(x) for x in raw],
        "note": "class index -> name mapping NOT asserted; confirm raw values "
                "in Neuroglancer (hover shows the raw label).",
    }


def subcompartment_report(arr):
    """Voxel counts per raw value and per decoded class. Reporting only."""
    dec = decode_subcompartment(arr)
    a = np.asarray(arr)
    raw_counts = {int(v): int((a == v).sum()) for v in dec["raw_values"]}
    cls = dec["class_index"]
    cls_counts = {int(v): int((cls == v).sum()) for v in np.unique(cls)}
    return {
        "raw_value_counts": raw_counts,
        "decoded_class_counts": cls_counts,
        "n_myelin_voxels": int(dec["myelinated"].sum()),
        "note": dec["note"],
    }


# --------------------------------------------------------------------------- #
# 4. Pure: surface extraction                                                  #
# --------------------------------------------------------------------------- #
def surface_from_mask(mask, resolution_nm, lo_vox, step_size=1, pad=True):
    """Marching-cubes surface of a boolean mask, with vertices in NANOMETRES.

    mask          : 3D boolean array, in the cutout's index frame.
    resolution_nm : (3,) nm per voxel, the SAME resolution the mask was fetched
                    at -- passing the wrong one scales the mesh silently.
    lo_vox        : the cutout origin, so vertices come back in the global nm
                    frame of the node table rather than the local box.
    pad           : zero-pad by one voxel first, so a spine touching the ROI
                    face produces a CLOSED surface instead of an open shell.

    Returns dict: verts_nm (V, 3), faces (F, 3), normals (V, 3), n_verts,
    n_faces, volume_nm3 (voxel count x voxel volume, NOT the mesh volume).
    """
    from skimage import measure          # imported late: optional dependency

    m = np.asarray(mask, dtype=bool)
    if m.ndim != 3:
        raise RoiError("mask must be 3D, got %s" % (m.shape,))
    if not m.any():
        raise RoiError("mask is empty -- no surface to extract")

    res = np.asarray(resolution_nm, dtype=float)
    origin_vox = np.asarray(lo_vox, dtype=float)

    if pad:
        m = np.pad(m, 1, mode="constant", constant_values=False)
        origin_vox = origin_vox - 1.0

    if np.any(np.asarray(m.shape) < 2):
        raise RoiError("mask too small for marching cubes: shape %s" % (m.shape,))

    verts, faces, normals, _ = measure.marching_cubes(
        m.astype(np.float32), level=0.5, spacing=tuple(res), step_size=step_size)
    # verts are in nm relative to the (padded) array origin; shift to global nm.
    verts_nm = verts + origin_vox * res

    return {
        "verts_nm": verts_nm,
        "faces": faces,
        "normals": normals,
        "n_verts": int(len(verts)),
        "n_faces": int(len(faces)),
        "n_voxels": int(np.asarray(mask, dtype=bool).sum()),
        "volume_nm3": float(np.asarray(mask, dtype=bool).sum() * np.prod(res)),
        "resolution_nm": [float(x) for x in res],
    }


# --------------------------------------------------------------------------- #
# 5. IO: save / load an ROI bundle. No network.                                #
# --------------------------------------------------------------------------- #
def roi_paths(out_dir, cell_id, sigma_id):
    """Canonical filenames for one (cell, spine) ROI."""
    stem = "cell%d_sigma%d" % (int(cell_id), int(sigma_id))
    return {
        "stem": stem,
        "npz": os.path.join(out_dir, stem + "_roi.npz"),
        "json": os.path.join(out_dir, stem + "_roi.json"),
        "nodes_csv": os.path.join(out_dir, stem + "_nodes.csv"),
        "mesh_npz": os.path.join(out_dir, stem + "_mesh.npz"),
        "html": os.path.join(out_dir, stem + "_spine3d.html"),
    }


def save_roi(out_dir, cell_id, sigma_id, arrays, meta, spine_nodes=None,
             mesh=None, mask=None, shaft_context_mask=None, shaft_mesh=None,
             shaft_connected_mask=None, detached_mask=None,
             detached_mesh=None, bridge_mask=None, bridge_mesh=None):
    """Write the cutouts, the provenance sidecar, the nodes and the mesh(es).

    arrays : dict layer -> 3D numpy array (may omit layers that were skipped)
    meta   : dict, everything needed to reproduce the request -- cloudpaths,
             mip, resolution, voxel box, nm box, module version
    masks  : spine, shaft-connected and detached are saved as separate boolean
             arrays alongside the raw cutouts so the three-way split can be
             re-inspected without re-fetching or re-partitioning.
    Returns the dict of paths written.
    """
    os.makedirs(out_dir, exist_ok=True)
    p = roi_paths(out_dir, cell_id, sigma_id)

    to_save = dict(arrays)
    for key, arr in (("spine_mask", mask),
                     ("shaft_context_mask", shaft_context_mask),
                     ("shaft_connected_mask", shaft_connected_mask),
                     ("detached_mask", detached_mask),
                     ("bridge_mask", bridge_mask)):
        if arr is not None:
            to_save[key] = np.asarray(arr, dtype=bool)
    np.savez_compressed(p["npz"], **{k: np.asarray(v) for k, v in to_save.items()})

    with open(p["json"], "w", newline="\n") as fh:
        json.dump(meta, fh, indent=2, default=_jsonable)
    if spine_nodes is not None:
        spine_nodes.to_csv(p["nodes_csv"], index=False, lineterminator="\n")
    if mesh is not None:
        mesh_kw = dict(verts_nm=mesh["verts_nm"], faces=mesh["faces"],
                       normals=mesh["normals"])
        if shaft_mesh is not None:
            mesh_kw.update(shaft_verts_nm=shaft_mesh["verts_nm"],
                           shaft_faces=shaft_mesh["faces"],
                           shaft_normals=shaft_mesh["normals"])
        if detached_mesh is not None:
            mesh_kw.update(detached_verts_nm=detached_mesh["verts_nm"],
                           detached_faces=detached_mesh["faces"],
                           detached_normals=detached_mesh["normals"])
        if bridge_mesh is not None:
            mesh_kw.update(bridge_verts_nm=bridge_mesh["verts_nm"],
                           bridge_faces=bridge_mesh["faces"],
                           bridge_normals=bridge_mesh["normals"])
        np.savez_compressed(p["mesh_npz"], **mesh_kw)
    return p


def load_roi(out_dir, cell_id, sigma_id):
    """Read back what save_roi wrote. Returns (arrays, meta, nodes, mesh)."""
    p = roi_paths(out_dir, cell_id, sigma_id)
    with np.load(p["npz"]) as z:
        arrays = {k: z[k] for k in z.files}
    with open(p["json"], "r") as fh:
        meta = json.load(fh)
    nodes = pd.read_csv(p["nodes_csv"]) if os.path.isfile(p["nodes_csv"]) else None
    mesh = None
    if os.path.isfile(p["mesh_npz"]):
        with np.load(p["mesh_npz"]) as z:
            mesh = {k: z[k] for k in z.files}
    return arrays, meta, nodes, mesh


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# --------------------------------------------------------------------------- #
# 6. Network. The ONLY functions here that touch the wire.                     #
# --------------------------------------------------------------------------- #
def make_cloudvolume_reader(cloudpath, mip=0, lru_bytes=2 * 1024 ** 3,
                            fill_missing=False):
    """Build a reader for one layer. Returns (reader_fn, layer_info).

    reader_fn(lo_vox, hi_vox) -> 3D numpy array.

    layer_info carries the resolution and bounds THIS layer reports, so the
    caller converts nm -> voxel with the real numbers rather than a constant.

    fill_missing=False on purpose, same reasoning as h01_segid_census: a chunk
    that merely failed to download must not come back as label 0 and be
    mistaken for genuine background.
    """
    from cloudvolume import CloudVolume     # imported late: optional dependency

    vol = CloudVolume(cloudpath, mip=mip, use_https=True, progress=False,
                      fill_missing=fill_missing, lru_bytes=lru_bytes)
    bounds = vol.bounds
    info = {
        "cloudpath": cloudpath,
        "mip": int(mip),
        "resolution_nm": [float(x) for x in vol.resolution],
        "dtype": str(vol.dtype),
        "bounds_vox": [[int(x) for x in bounds.minpt],
                       [int(x) for x in bounds.maxpt]],
        "available_mips": [int(m) for m in vol.available_mips],
    }

    def reader(lo_vox, hi_vox):
        lo = [int(v) for v in lo_vox]
        hi = [int(v) for v in hi_vox]
        cut = vol[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        a = np.asarray(cut)
        return a[..., 0] if a.ndim == 4 else a

    return reader, info


def pick_mip(layer_info_by_mip, target_nm):
    """Smallest mip whose resolution is still <= target_nm on every axis.

    layer_info_by_mip : dict mip -> resolution (3,) in nm
    Falls back to the coarsest available mip if none qualifies.
    """
    target = np.asarray(target_nm, dtype=float)
    ok = [m for m, r in sorted(layer_info_by_mip.items())
          if np.all(np.asarray(r, dtype=float) <= target)]
    return int(max(ok)) if ok else int(max(layer_info_by_mip))


def fetch_roi(bbox_nm, layers=None, mips=None, max_bytes=DEFAULT_MAX_BYTES,
              reader_factory=make_cloudvolume_reader, verbose=True):
    """Cut the same nm box out of each requested layer.

    bbox_nm        : (lo_nm, hi_nm) from roi_bbox_nm
    layers         : dict name -> cloudpath; defaults to LAYERS
    mips           : dict name -> mip; defaults to 0 for every layer
    max_bytes      : per-layer guard. Raises RoiError rather than OOM-ing the
                     runtime halfway through a download.
    reader_factory : injected so the whole function is testable offline with a
                     stub. This is the same pattern as annotate_segids.

    Returns (arrays, meta). arrays maps layer name -> 3D array. meta records
    the nm box, and per layer the cloudpath, mip, real resolution, voxel box,
    dtype and shape -- everything needed to put the array back in world space.
    """
    layers = dict(LAYERS if layers is None else layers)
    mips = dict(mips or {})
    lo_nm, hi_nm = (np.asarray(b, dtype=float) for b in bbox_nm)

    arrays, per_layer = {}, {}
    for name, cloudpath in layers.items():
        mip = int(mips.get(name, 0))
        reader, linfo = reader_factory(cloudpath, mip=mip)
        res = np.asarray(linfo["resolution_nm"], dtype=float)
        lo_v, hi_v = bbox_nm_to_voxel((lo_nm, hi_nm), res,
                                      volume_bounds_vox=linfo["bounds_vox"])
        nbytes = estimate_bytes(lo_v, hi_v, linfo["dtype"])
        if nbytes > max_bytes:
            raise RoiError(
                "layer %r at mip %d would need %.2f GiB (%s voxels of %s). "
                "Reduce pad_nm, or raise the mip for this layer, or raise "
                "max_bytes deliberately."
                % (name, mip, nbytes / 1024 ** 3,
                   (hi_v - lo_v).tolist(), linfo["dtype"]))
        if verbose:
            print("  %-5s mip %d  res %s nm  box %s..%s  %s  %.1f MiB"
                  % (name, mip, res.tolist(), lo_v.tolist(), hi_v.tolist(),
                     linfo["dtype"], nbytes / 1024 ** 2))
        arrays[name] = reader(lo_v, hi_v)
        per_layer[name] = {
            "cloudpath": cloudpath, "mip": mip,
            "resolution_nm": res.tolist(),
            "lo_vox": lo_v.tolist(), "hi_vox": hi_v.tolist(),
            "dtype": str(arrays[name].dtype),
            "shape": list(arrays[name].shape),
            "bytes": int(arrays[name].nbytes),
        }

    meta = {
        "module_version": MODULE_VERSION,
        "release": RELEASE,
        "bbox_nm": {"lo": lo_nm.tolist(), "hi": hi_nm.tolist()},
        "layers": per_layer,
    }
    return arrays, meta


# --------------------------------------------------------------------------- #
# 7. Orchestration: the one call the notebook makes                            #
# --------------------------------------------------------------------------- #
def extract_spine_roi(nodes, comp, sigma_id, cell_id, out_dir,
                      pad_nm=DEFAULT_PAD_NM, isotropic=False,
                      bridge_gap_nm=None, resolve_components=True,
                      bridge_span_mode="local_calibre",
                      bridge_span_window_nm=200.0,
                      bridge_span_estimator="equivalent",
                      bridge_span_percentile=90.0,
                      bridge_span_radius_nm=None,
                      layers=None, mips=None,
                      max_bytes=DEFAULT_MAX_BYTES,
                      reader_factory=make_cloudvolume_reader,
                      build_mesh=True, save=True, verbose=True):
    """Full single-spine pipeline. Returns a dict with everything.

    Steps, each delegated to a function above so any of them can be replaced:
      1  spine_subframe    pull sigma's nodes and its shaft attachment point
      2  roi_bbox_nm       the nm box
      3  fetch_roi         one cutout per layer  [NETWORK]
      4  segids_at_points  per-node segid, sampled from the cutout, no network
      5  mask_from_segids  the spine's voxels
      6  surface_from_mask marching cubes, vertices in global nm
      7  save_roi          arrays + provenance + nodes + mesh to disk
    """
    spine_nodes, base_node, sinfo = spine_subframe(nodes, comp, sigma_id)
    if verbose:
        print("sigma %d: %d nodes, extent %s nm, single_root=%s, base=%s"
              % (sigma_id, sinfo["n_nodes"],
                 {k: round(v) for k, v in sinfo["extent_nm"].items()},
                 sinfo["single_root"], sinfo["base_id"]))
        if not sinfo["single_root"]:
            print("  WARNING: %d roots -- this component is probably two "
                  "spines merged by the labeller." % sinfo["n_roots"])

    pts = spine_nodes[["x", "y", "z"]].to_numpy(dtype=float)
    if base_node is not None:
        pts = np.vstack([pts, base_node[["x", "y", "z"]].to_numpy(dtype=float)])
    lo_nm, hi_nm = roi_bbox_nm(pts, pad_nm=pad_nm, isotropic=isotropic)
    if verbose:
        print("ROI %s .. %s nm  (%s nm span)"
              % (np.round(lo_nm).tolist(), np.round(hi_nm).tolist(),
                 np.round(hi_nm - lo_nm).tolist()))

    arrays, meta = fetch_roi((lo_nm, hi_nm), layers=layers, mips=mips,
                             max_bytes=max_bytes,
                             reader_factory=reader_factory, verbose=verbose)

    result = {"spine_info": sinfo, "spine_nodes": spine_nodes,
              "base_node": base_node, "arrays": arrays, "meta": meta}

    if "seg" in arrays:
        seg_meta = meta["layers"]["seg"]
        node_pts = spine_nodes[["x", "y", "z"]].to_numpy(dtype=float)
        vals, inside = segids_at_points(
            arrays["seg"], node_pts, seg_meta["lo_vox"], seg_meta["resolution_nm"])
        segids = sorted(set(int(v) for v in vals[inside]) - {BACKGROUND_LABEL})
        result["node_segids"] = vals
        result["node_inside_roi"] = inside
        result["spine_segids"] = segids
        meta["spine_segids"] = segids
        meta["n_nodes_outside_roi"] = int((~inside).sum())
        meta["n_nodes_on_background"] = int((vals[inside] == BACKGROUND_LABEL).sum())
        if verbose:
            print("spine occupies %d segid(s): %s" % (len(segids), segids))
            if len(segids) > 1:
                print("  NOTE: >1 segid -- this spine straddles a segmentation "
                      "boundary ('mixed' in G0 terms).")
            if meta["n_nodes_on_background"]:
                print("  NOTE: %d node(s) sit on label 0 (outside every "
                      "segment)." % meta["n_nodes_on_background"])

        if build_mesh and segids:
            raw_mask = mask_from_segids(arrays["seg"], segids)
            spine_ids = set(int(v) for v in spine_nodes["id"].to_numpy())
            mask, shaft_context_mask = mask_local_to_spine(
                raw_mask, seg_meta["lo_vox"], seg_meta["resolution_nm"],
                nodes, spine_ids)
            n_raw, n_kept, n_excl = int(raw_mask.sum()), int(mask.sum()), \
                int(shaft_context_mask.sum())
            if verbose:
                print("segid-shared voxels %d -> spine %d (%.0f%%), "
                      "excluded as shaft-context %d"
                      % (n_raw, n_kept, 100.0 * n_kept / max(n_raw, 1), n_excl))
            if n_kept == 0:
                raise RoiError(
                    "mask_local_to_spine kept 0 voxels for sigma %d -- every "
                    "voxel of the shared segid was nearer to a non-spine node "
                    "than to this spine's own nodes. Check pad_nm and the "
                    "sigma_id." % sigma_id)

            # ORDER MATTERS HERE. The component audit must run BEFORE anything
            # is meshed, because it MOVES voxels from the spine to the shaft.
            # Meshing first and relabelling afterwards (which is what this did
            # in v1.3) rebuilt the spine mesh but not the shaft mesh, so the
            # relabelled piece disappeared from the figure entirely instead of
            # changing colour. It was never deleted from the masks -- only from
            # the render -- but that is just as misleading.
            node_pts_nm = spine_nodes[["x", "y", "z"]].to_numpy(dtype=float)
            core_mask, orphan_mask, comp_audit = resolve_spine_mask(
                mask, node_pts_nm, seg_meta["lo_vox"], seg_meta["resolution_nm"])
            if resolve_components and orphan_mask.any():
                # RELABEL, not remove: these voxels stay in the ROI and stay
                # drawn, they simply become shaft rather than spine.
                shaft_context_mask = shaft_context_mask | orphan_mask
                mask = core_mask
            spine_components = component_report(mask)

            # Now that the labels are final, split the shaft material and mesh
            # everything from the SAME set of masks.
            shaft_connected_mask, detached_mask, contact = \
                split_excluded_by_contact(mask, shaft_context_mask)

            mesh = surface_from_mask(mask, seg_meta["resolution_nm"],
                                     seg_meta["lo_vox"])
            shaft_mesh = (surface_from_mask(shaft_connected_mask,
                                            seg_meta["resolution_nm"],
                                            seg_meta["lo_vox"])
                          if shaft_connected_mask.any() else None)
            detached_mesh = (surface_from_mask(detached_mask,
                                               seg_meta["resolution_nm"],
                                               seg_meta["lo_vox"])
                             if detached_mask.any() else None)

            # Measure against ALL shaft material, not just the part already
            # classified as touching: after relabelling, the spine may no
            # longer touch anything, and the thing to attach it to is then the
            # NEAREST shaft voxel wherever it is. bridge_masks finds the
            # closest pair itself, so passing the whole shaft mask attaches to
            # the closest part by construction.
            attach_target = shaft_context_mask
            contact_geom = mask_contact_report(
                mask, attach_target, seg_meta["resolution_nm"],
                lo_vox=seg_meta["lo_vox"]) if attach_target.any() else None

            bridge_mask, bridge_info = None, {"bridged": False,
                                              "reason": "bridge_gap_nm is None "
                                                        "(bridging disabled)"}
            if bridge_gap_nm is not None and attach_target.any():
                bridge_mask, bridge_info = bridge_masks(
                    mask, attach_target, seg_meta["resolution_nm"],
                    bridge_gap_nm, span_mode=bridge_span_mode,
                    span_window_nm=bridge_span_window_nm,
                    span_estimator=bridge_span_estimator,
                    span_percentile=bridge_span_percentile,
                    span_radius_nm=bridge_span_radius_nm,
                    lo_vox=seg_meta["lo_vox"])
                if bridge_info["bridged"]:
                    # The bridge is kept as its OWN mask and its own mesh. It is
                    # deliberately NOT merged into `mask`: a spine volume that
                    # silently includes interpolated voxels is not a measurement.
                    result["bridge_mask"] = bridge_mask
                    result["bridge_mesh"] = surface_from_mask(
                        bridge_mask, seg_meta["resolution_nm"], seg_meta["lo_vox"])

            # If the spine is STILL in pieces, the gap is inside the spine
            # itself, not between spine and shaft -- bridge those too.
            comp_bridge, comp_bridge_info = None, {"bridged": False,
                                                   "reason": "not attempted"}
            if bridge_gap_nm is not None and spine_components["n_components"] > 1:
                comp_bridge, comp_bridge_info = bridge_components(
                    mask, seg_meta["resolution_nm"], bridge_gap_nm,
                    lo_vox=seg_meta["lo_vox"], span_mode=bridge_span_mode,
                    span_window_nm=bridge_span_window_nm,
                    span_estimator=bridge_span_estimator,
                    span_percentile=bridge_span_percentile,
                    span_radius_nm=bridge_span_radius_nm)
                if comp_bridge_info.get("bridged"):
                    bridge_mask = comp_bridge if bridge_mask is None \
                        else (bridge_mask | comp_bridge)
                    result["bridge_mask"] = bridge_mask
                    result["bridge_mesh"] = surface_from_mask(
                        bridge_mask, seg_meta["resolution_nm"], seg_meta["lo_vox"])

            result["component_audit"] = comp_audit
            result["orphan_mask"] = orphan_mask
            result["component_bridge_info"] = comp_bridge_info
            meta["component_audit"] = {k: v for k, v in comp_audit.items()}
            meta["component_bridge"] = comp_bridge_info
            result["raw_mask"] = raw_mask
            result["mask"] = mask
            result["shaft_context_mask"] = shaft_context_mask
            result["shaft_connected_mask"] = shaft_connected_mask
            result["detached_mask"] = detached_mask
            result["mesh"] = mesh
            result["shaft_mesh"] = shaft_mesh
            result["detached_mesh"] = detached_mesh
            result["contact"] = contact
            result["spine_components"] = spine_components

            meta["mesh"] = {k: mesh[k] for k in
                            ("n_verts", "n_faces", "n_voxels", "volume_nm3")}
            meta["mask_partition"] = {
                "method": "nearest-node Voronoi split against the full "
                          "reconstruction's skeleton labels (see "
                          "mask_local_to_spine docstring)",
                "n_voxels_shared_segid": n_raw,
                "n_voxels_kept_as_spine": n_kept,
                "n_voxels_excluded_as_shaft_context": n_excl,
                "fraction_kept": n_kept / max(n_raw, 1),
            }
            meta["excluded_split"] = contact
            meta["spine_components"] = spine_components
            meta["contact_geometry"] = contact_geom
            meta["bridge"] = bridge_info
            result["contact_geometry"] = contact_geom
            result["bridge_info"] = bridge_info

            if verbose:
                print("mesh: %d verts, %d faces, %d voxels, %.3f um^3"
                      % (mesh["n_verts"], mesh["n_faces"], mesh["n_voxels"],
                         mesh["volume_nm3"] / 1e9))
                print("excluded split: %d vox touching the spine (shaft), "
                      "%d vox detached in %d blob(s)%s"
                      % (contact["n_voxels_connected"],
                         contact["n_voxels_detached"],
                         contact["n_detached_components"],
                         (" sizes %s" % contact["detached_sizes_voxels"][:5])
                         if contact["n_detached_components"] else ""))
                if contact_geom is not None:
                    print("spine <-> NEAREST shaft material: empty span %.1f nm "
                          "(centres %.1f nm, touching=%s, %d contact voxel(s))"
                          % (contact_geom["gap_face_nm"],
                             contact_geom["gap_center_nm"],
                             contact_geom["touching"],
                             contact_geom["n_contact_voxels"]))
                if bridge_info.get("bridged"):
                    print("  BRIDGED across %.1f nm of empty span with %d "
                          "INTERPOLATED voxel(s), radius %s nm (%s) -- not in "
                          "the released segmentation"
                          % (bridge_info["gap_face_nm"],
                             bridge_info["n_bridge_voxels"],
                             ("%.1f" % bridge_info["radius_nm"])
                             if bridge_info.get("radius_nm") else "n/a",
                             bridge_info.get("span_mode")))
                elif bridge_gap_nm is not None:
                    print("  not bridged: %s" % bridge_info["reason"])
                print("component audit: %s" % comp_audit["verdict"])
                for c in comp_audit["components"][:6]:
                    print("    component %d: %d vox, %d skeleton node(s)"
                          % (c["label"], c["size_voxels"], c["n_nodes"]))
                if resolve_components and orphan_mask.any():
                    print("    -> %d orphan voxel(s) with no node RELABELLED "
                          "spine -> shaft (still drawn, in amber/grey)"
                          % int(orphan_mask.sum()))
                if comp_bridge_info.get("bridged"):
                    print("    -> intra-spine bridge: %d INTERPOLATED voxel(s), "
                          "%d component(s) -> %d"
                          % (comp_bridge_info["n_bridge_voxels"],
                             comp_bridge_info["n_components"],
                             comp_bridge_info["components_after"]))
                if spine_components["n_components"] > 1:
                    print("  WARNING: the SPINE mask itself has %d components "
                          "(sizes %s). The pad may have caught unrelated "
                          "material, or the segmentation splits this spine. "
                          "Nothing was merged or pruned automatically."
                          % (spine_components["n_components"],
                             spine_components["sizes_voxels"][:5]))

    if "subc" in arrays:
        rep = subcompartment_report(arrays["subc"])
        meta["subcompartment"] = rep
        result["subcompartment"] = rep
        if verbose:
            print("subcompartment raw values present: %s"
                  % list(rep["raw_value_counts"]))

    if save:
        result["paths"] = save_roi(
            out_dir, cell_id, sigma_id, arrays, meta,
            spine_nodes=spine_nodes, mesh=result.get("mesh"),
            mask=result.get("mask"),
            shaft_context_mask=result.get("shaft_context_mask"),
            shaft_mesh=result.get("shaft_mesh"),
            shaft_connected_mask=result.get("shaft_connected_mask"),
            detached_mask=result.get("detached_mask"),
            detached_mesh=result.get("detached_mesh"),
            bridge_mask=result.get("bridge_mask"),
            bridge_mesh=result.get("bridge_mesh"))
        if verbose:
            print("wrote", result["paths"]["npz"])
    return result
