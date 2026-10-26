"""The node sample of S3a -- Block 17.1 in specs/SPEC.md (D-085.4). Pure
functions of the SWC (loading/swc_io.SWC) and the configuration.

  all_nodes         every dendrite node of the cell
  stratified_nodes  up to nodes_per_bin nodes per bin of Allen's diameter,
                    taken round-robin over the cell's stretches
  sample_nodes      the one survey.nodes_mode names

Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import math

import numpy as np

from . import cell


def all_nodes(swc, cfg):
    """The ids of every dendrite node (acquisition.dendrite_swc_types)."""
    mask = swc.dendrite_mask(cfg.acquisition.dendrite_swc_types)
    return [int(i) for i in np.asarray(swc.ids)[mask]]


def allen_bin(radius_um, inner_edges):
    """The bin index b of Allen's diameter 2 r in [0, e_1), [e_1, e_2), ...,
    [e_last, inf); None for a non-finite or negative radius."""
    r = float(radius_um)
    if not math.isfinite(r) or r < 0:
        return None
    return int(np.searchsorted(np.asarray(inner_edges, dtype=float), 2.0 * r, side="right"))


def stratified_nodes(swc, cfg, specimen_id):
    """Block 17.1: per bin b of Allen's diameter, the cell's dendrite nodes in
    it grouped by stretch (cell.stretches); the stretches and each stretch's
    nodes in the orders of permutations from
    default_rng([selection_seed, specimen_id, b]); the nodes taken
    round-robin over the stretches until nodes_per_bin or the bin runs out.
    Returns (ids, counts), counts = {b: (available, taken)}."""
    sv = cfg.survey
    edges = sv.nodes_allen_bins_um
    by_bin = {}
    for run in cell.stretches(swc, cfg.acquisition.dendrite_swc_types):
        groups = {}
        for r in run:
            b = allen_bin(swc.radius[r], edges)
            if b is not None:
                groups.setdefault(b, []).append(int(swc.ids[r]))
        for b, ids in groups.items():
            by_bin.setdefault(b, []).append(ids)
    out, counts = [], {}
    for b in sorted(by_bin):
        rng = np.random.default_rng([int(sv.selection_seed), int(specimen_id), int(b)])
        runs = by_bin[b]
        order = [runs[i] for i in rng.permutation(len(runs))]
        order = [[ids[j] for j in rng.permutation(len(ids))] for ids in order]
        taken, depth = [], 0
        n_avail = sum(len(ids) for ids in order)
        while len(taken) < min(int(sv.nodes_per_bin), n_avail):
            for ids in order:
                if depth < len(ids) and len(taken) < int(sv.nodes_per_bin):
                    taken.append(ids[depth])
            depth += 1
        out.extend(taken)
        counts[b] = (n_avail, len(taken))
    return out, counts


def sample_nodes(swc, cfg, specimen_id):
    """(ids, counts) of the mode survey.nodes_mode names (counts {} for all)."""
    if cfg.survey.nodes_mode == "all":
        return all_nodes(swc, cfg), {}
    return stratified_nodes(swc, cfg, specimen_id)
