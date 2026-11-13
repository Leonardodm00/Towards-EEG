"""The cleaning operator and the two tiers -- Block 28 in specs/SPEC.md (plan step C3; D-130.O1, D-132).

Cleaning (D-130.O1). For each plane k and pixel (x, y) of node i's patch,

    I_clean_{i,k}(x, y) = I_{i,k}(x, y) + B_{i,k}(x, y) S_{i,k}(x, y),

I the observed block (gl), B the background (gl), S the summed shadow of the
tier-1 neighbours (dimensionless, Block 25's ShadowResult.S). No clipping;
float64 out. Per plane the result also carries the mean grey level added,
mean_{x,y} B S (gl), and the share of pixels whose cleaned value exceeds B by
more than CleaningConfig.over_background_gl (an over-subtraction diagnostic).

Tiers (D-132). Neighbour q of node i -- in shadow.neighbour_indices' order:
every segment of the FeaturedScene with at least one end outside own_ids, in
the scene's order, then the soma discs -- is tier 2 when
  (a) its point nearest (in projection) to the node's position lies within
      close_lateral_um of it laterally and within close_defocus_um of it in
      depth (the depth linear along the segment), or
  (b) it shares no node with the own stretch and its footprint (the points
      within its larger end radius of its axis, in projection) meets the own
      stretch's footprint (each own segment -- both ends in own_ids -- at its
      larger end radius) within tier_reach_um of the node in projection;
tier 1 otherwise. Soma discs are tier 1. The test (b) is read on a square grid
of spacing MeasureConfig.profile_step_um centred on the node, restricted to
the disc of radius tier_reach_um (Block 27's overlap test reads its square
the same way).

No file access, no plotting. Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import background


@dataclass(frozen=True)
class CleanResult:
    """block (n_planes, H, W) float64 gl: the cleaned block; added_gl (n_planes,) gl: the mean of B S
    per plane; over_background (n_planes,): the share of pixels with cleaned value - B > over_background_gl."""

    block: np.ndarray
    added_gl: np.ndarray
    over_background: np.ndarray


# -------------------------------------------------------------- cleaning ---

def _background(B_gl, shape):
    """B broadcast to shape (n_planes, H, W), float64, from a scalar, (n_planes,) or (n_planes, H, W)."""
    B = np.asarray(B_gl)
    if B.dtype.kind not in "iuf":
        raise ValueError("B_gl must be real-valued (gl)")
    B = B.astype(np.float64)
    if B.ndim == 0:
        out = np.full(shape, float(B))
    elif B.shape == (shape[0],):
        out = np.broadcast_to(B[:, None, None], shape)
    elif B.shape == tuple(shape):
        out = B
    else:
        raise ValueError("B_gl must be a scalar, (n_planes,) = (%d,) or (n_planes, H, W) = %r; got shape %r"
                         % (shape[0], tuple(shape), B.shape))
    if not np.all(np.isfinite(out)) or np.any(out <= 0):
        raise ValueError("B_gl must be finite and > 0 (gl)")
    return out


def clean_block(block, S, B_gl, cfg=None):
    """I_clean = I + B S, plane by plane (D-130.O1); no clipping.

    block: (n_planes, H, W), any real dtype (gl); S: the same shape, float, finite and >= 0
    (dimensionless, the tier-1 shadow of Block 25); B_gl: scalar, (n_planes,) or (n_planes, H, W),
    finite and > 0 (gl). cfg: DiameterConfig for CleaningConfig.over_background_gl -- NOT in the
    contract signature of SPEC Block 28 (reported to the orchestrator); None takes the default
    configuration. Returns CleanResult (block float64, added_gl (n_planes,), over_background
    (n_planes,)). The inputs are not changed. ValueError for unequal shapes, a block that is not
    3-D, real and finite, a non-finite or negative S, B <= 0 or non-finite.
    """
    if cfg is None:
        from ..config import default_config
        cfg = default_config()
    thr = float(cfg.cleaning.over_background_gl)
    I = np.asarray(block)
    if I.ndim != 3:
        raise ValueError("block must be (n_planes, H, W); got shape %r" % (I.shape,))
    if I.dtype.kind not in "iuf":
        raise ValueError("block must have a real numeric dtype; got %s" % (I.dtype,))
    I = I.astype(np.float64)                             # a copy: the input is never changed
    if not np.all(np.isfinite(I)):
        raise ValueError("block must be finite (gl)")
    Sa = np.asarray(S)
    if Sa.shape != I.shape:
        raise ValueError("S must have the block's shape %r; got %r" % (I.shape, Sa.shape))
    if Sa.dtype.kind not in "iuf":
        raise ValueError("S must be real-valued; got %s" % (Sa.dtype,))
    Sa = Sa.astype(np.float64)
    if not np.all(np.isfinite(Sa)) or np.any(Sa < 0):
        raise ValueError("S must be finite and >= 0 (a shadow is an absorbed fraction)")
    B = _background(B_gl, I.shape)
    BS = B * Sa                                          # gl
    clean = I + BS
    added = BS.mean(axis=(1, 2))
    over = ((clean - B) > thr).mean(axis=(1, 2))
    return CleanResult(block=clean, added_gl=added, over_background=over)


# ----------------------------------------------------------------- tiers ---

def _nearest_t(px, py, a, b):
    """Parameter t in [0, 1] of the point of segment a-b nearest to (px, py) in projection.
    Custom: the closed form behind background.segment_distance, which returns only the distance and
    is used for it; numpy/scipy offer no point-segment projection."""
    ax, ay = float(a[0]), float(a[1])
    dx, dy = float(b[0]) - ax, float(b[1]) - ay
    L2 = dx * dx + dy * dy
    if L2 == 0.0:
        return 0.0
    return float(np.clip(((px - ax) * dx + (py - ay) * dy) / L2, 0.0, 1.0))


def _scene_arrays(fscene):
    xyz = np.asarray(fscene.seg_xyz, dtype=np.float64)
    ids = np.asarray(fscene.seg_ids)
    d = np.asarray(fscene.seg_d_um, dtype=np.float64)
    soma = np.asarray(fscene.soma, dtype=np.float64).reshape(-1, 3)
    n = xyz.shape[0] if xyz.ndim == 2 else -1
    if xyz.shape != (n, 6) or ids.shape != (n, 2) or d.shape != (n, 2):
        raise ValueError("fscene: seg_xyz (n, 6), seg_ids (n, 2) and seg_d_um (n, 2) must agree")
    if ids.dtype.kind not in "iu":
        raise ValueError("fscene.seg_ids must be integer node ids")
    if not (np.all(np.isfinite(xyz)) and np.all(np.isfinite(d)) and np.all(d > 0)):
        raise ValueError("fscene: segment ends must be finite and diameters finite and > 0")
    return xyz, ids.astype(np.int64), d, soma


def neighbour_tiers(fscene, own_ids, node_xyz, cfg):
    """Tier (1 or 2) of every neighbour of node i (D-132), in shadow.neighbour_indices' order.

    fscene: Block 23 FeaturedScene (um); own_ids: the node ids of node i's own stretch; node_xyz
    (3,) um: the node's position; cfg: DiameterConfig (CleaningConfig.close_lateral_um,
    close_defocus_um, tier_reach_um; MeasureConfig.profile_step_um, the spacing of the grid on
    which rule (b) is read). Returns tiers (n_q + n_s,) int64 in {1, 2}: the segments with at
    least one end outside own_ids, in the scene's order, then the soma discs (always 1).
    ValueError for a node position that is not 3 finite numbers, a scene whose arrays disagree or
    are not finite, and an own stretch with no segment in the scene (no own footprint).
    """
    xyz, ids, d, soma = _scene_arrays(fscene)
    node = np.asarray(node_xyz, dtype=np.float64).reshape(-1)
    if node.shape != (3,) or not np.all(np.isfinite(node)):
        raise ValueError("node_xyz must be 3 finite numbers (um)")
    c = cfg.cleaning
    lat_max, dz_max, reach = float(c.close_lateral_um), float(c.close_defocus_um), float(c.tier_reach_um)
    step = float(cfg.measure.profile_step_um)
    own = np.asarray(sorted({int(i) for i in np.asarray(own_ids).ravel()}), dtype=np.int64)
    in0, in1 = np.isin(ids[:, 0], own), np.isin(ids[:, 1], own)
    own_seg = np.flatnonzero(in0 & in1)
    nb = np.flatnonzero(~(in0 & in1))
    if own_seg.size == 0:
        raise ValueError("neighbour_tiers: no segment of the scene has both ends in own_ids (no own footprint)")
    r_max = 0.5 * d.max(axis=1)                          # each segment at its larger end radius (um)

    # the disc of radius tier_reach_um about the node, on a grid of spacing profile_step_um
    m = int(math.floor(reach / step + 1e-12))
    off = np.arange(-m, m + 1) * step
    OX, OY = np.meshgrid(off, off, indexing="xy")
    disc = np.hypot(OX, OY) <= reach
    px, py = node[0] + OX[disc], node[1] + OY[disc]
    own_fp = np.zeros(px.shape, dtype=bool)
    for q in own_seg:
        own_fp |= background.segment_distance(px, py, xyz[q, :3], xyz[q, 3:]) <= r_max[q]
    qx, qy = px[own_fp], py[own_fp]

    tiers = np.ones(nb.size + soma.shape[0], dtype=np.int64)
    for k, q in enumerate(nb):
        a, b = xyz[q, :3], xyz[q, 3:]
        t = _nearest_t(node[0], node[1], a, b)
        lateral = math.hypot(node[0] - (a[0] + t * (b[0] - a[0])), node[1] - (a[1] + t * (b[1] - a[1])))
        defocus = abs(a[2] + t * (b[2] - a[2]) - node[2])
        if lateral <= lat_max and defocus <= dz_max:
            tiers[k] = 2                                 # (a) close and in focus
        elif not (in0[q] or in1[q]) and qx.size:
            if np.any(background.segment_distance(qx, qy, a, b) <= r_max[q]):
                tiers[k] = 2                             # (b) unattached, overlapping within tier_reach_um
    return tiers
