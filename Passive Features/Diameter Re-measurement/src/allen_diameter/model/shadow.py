"""The shadow image of a node's neighbours (tier 1) -- Block 25 in specs/SPEC.md (contract stub, wave 2; D-129, D-130, D-132).

The entry of Block 25 in specs/SPEC.md is the specification. Pure ASCII, for the cluster.
"""
from __future__ import annotations


def segment_absorbed(fscene, q, grid, cfg):
    """-> (A (ny, nx), z_axis (ny, nx)) of one segment on a render.FineGrid, under cfg.cleaning.shadow_absorption."""
    raise NotImplementedError("contract stub")


def neighbour_indices(fscene, own_ids):
    """-> (segments (n_q,), somas (n_s,)): every segment with at least one end outside own_ids; all soma discs."""
    raise NotImplementedError("contract stub")


def shadow_image(fscene, own_ids, left, top, width, height, planes_z, cfg, B_gl=None, include=None):
    """-> ShadowResult (S, per_neighbour_peak, kept, overlap); Block 25 with amendment 2."""
    raise NotImplementedError("contract stub")


def stretch_profiles(fscene, stretch_ids, ks, origin_xy, y_hat, e_u, v, n_avg, step, cfg):
    """-> (n_planes, n_v): the model's shadow of the stretch's own segments, read as profiles.sample_profile reads."""
    raise NotImplementedError("contract stub")
