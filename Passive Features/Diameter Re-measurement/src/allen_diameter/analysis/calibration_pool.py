"""S3c's pooled calibration set: one defocus kernel calibrated on nodes of
several cells -- Block 19.2 and 19.4 in specs/SPEC.md (D-088). Bookkeeping
on PlaneScan lists; nothing here reads or writes a file.

  QUALIFIER    the factor of a qualified node id, q = specimen_id x QUALIFIER + node_id
  qualify      a cell's scans with their node ids qualified by the specimen
  pool_scans   per set, the cells' scans taken round-robin over the cells up to the cap
  sufficiency  whether a set holds the minimum number of nodes (D-088.3)

Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import dataclasses

QUALIFIER = 10 ** 6


def qualify(scans, specimen_id):
    """The scans with node_id = specimen_id x QUALIFIER + node_id (two cells
    may share a node id; the fits carry one nuisance parameter per node)."""
    out = []
    for sc in scans:
        if not 0 <= int(sc.node_id) < QUALIFIER:
            raise ValueError("qualify: node id %d of specimen %d is outside [0, %d)" % (sc.node_id, specimen_id, QUALIFIER))
        out.append(dataclasses.replace(sc, node_id=int(specimen_id) * QUALIFIER + int(sc.node_id)))
    return out


def sufficiency(n_nodes, cfg):
    """D-088.3: {sufficient, n_nodes, min_nodes, max_nodes}."""
    cal = cfg.calibration
    return dict(sufficient=bool(int(n_nodes) >= cal.calib_min_nodes_per_set), n_nodes=int(n_nodes),
                min_nodes=int(cal.calib_min_nodes_per_set), max_nodes=int(cal.calib_max_nodes_per_set))


def pool_scans(per_cell, cfg):
    """Block 19.4: per_cell = [(specimen_id, scans in the cell's scan order)].
    Per set, the cells in increasing specimen id, round-robin -- each cell's
    first scan of the set, then each one's second -- up to
    calib_max_nodes_per_set. Returns (pooled scans with qualified ids, meta:
    per set the counts by cell, the sufficiency, and the id map)."""
    cap = int(cfg.calibration.calib_max_nodes_per_set)
    cells = sorted((int(s), list(sc)) for s, sc in per_cell)
    pooled, meta = [], dict(sets={}, id_map={})
    for which in ("near", "far"):
        lists = [(s, [x for x in sc if x.set == which]) for s, sc in cells]
        taken, depth = [], 0
        n_avail = sum(len(x) for _, x in lists)
        while len(taken) < min(cap, n_avail):
            for s, xs in lists:
                if depth < len(xs) and len(taken) < cap:
                    taken.append((s, xs[depth]))
            depth += 1
        by_cell = {}
        for s, sc in taken:
            q = qualify([sc], s)[0]
            pooled.append(q)
            meta["id_map"][str(q.node_id)] = [s, int(sc.node_id)]
            by_cell[str(s)] = by_cell.get(str(s), 0) + 1
        meta["sets"][which] = dict(available=n_avail, taken=len(taken), by_cell=by_cell,
                                   **sufficiency(len(taken), cfg))
    return pooled, meta
