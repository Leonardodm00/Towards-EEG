"""Per-node features by round, and the scene rebuilt from them -- Block 23 in specs/SPEC.md (wave 1; D-129, D-130).

A node's features at a round (D-130.O3): its diameter d (um, > 0), its
absorption coefficient mu (1/um, >= 0) and its offset (dx, dy, dz) (um) from
its traced position in the image frame of section 2.1. The table holds them
for the N traced nodes that carry a segment: the nodes whose SWC type is in
`types` and whose parent is a node of the file -- the children of the scene's
segments (analysis.cell.cell_scene with the same types), one node per segment.

featured_scene rebuilds plane_iteration.Scene's segments from the features:
end 0 of a segment is its parent end (seg_xyz[:, :3], seg_radius[:, 0], the
id seg_ids[:, 1]), end 1 its child end (seg_xyz[:, 3:], seg_radius[:, 1], the
id seg_ids[:, 0]) -- the Scene's own convention, kept in FeaturedScene. Each
end whose node is in the table is moved by the node's offset and takes its
diameter and coefficient; a parent end whose node is not in the table (the
soma end of a root segment) keeps the scene's position and diameter
(2 x seg_radius) and takes the child end's coefficient.

Library calls: numpy only. No numerical custom code (indexing and copies).
No file access here (loading.features_io writes and reads the table).
Pure ASCII, for the cluster.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SOURCES = ("allen", "pilot", "refined", "fine")
ROW_KEYS = ("node_id", "d_um", "mu_per_um", "offset_um", "source", "flags")


def _frozen(x):
    x = np.array(x, copy=True)
    x.setflags(write=False)
    return x


@dataclass(frozen=True)
class NodeFeatures:
    """node_id (N,) int64; swc_type (N,) int64; d_um (N,) um > 0; mu_per_um (N,) 1/um >= 0;
    offset_um (N, 3) = (dx, dy, dz) um; source (N,) str in SOURCES; flags (N,) str, ';'-joined;
    round_index int >= 0. The arrays are copied, cast to these dtypes and made read-only.
    ValueError: arrays of unequal length or wrong shape, a repeated node_id, a non-finite
    d_um, mu_per_um or offset_um entry, d_um <= 0, mu_per_um < 0, a source not in SOURCES,
    a round_index that is not an integer >= 0."""

    node_id: np.ndarray
    swc_type: np.ndarray
    d_um: np.ndarray
    mu_per_um: np.ndarray
    offset_um: np.ndarray
    source: np.ndarray
    flags: np.ndarray
    round_index: int = 0

    def __post_init__(self):
        def setf(name, value):
            object.__setattr__(self, name, value)

        try:
            ids = np.asarray(self.node_id)
            typ = np.asarray(self.swc_type)
            if ids.ndim != 1 or typ.ndim != 1:
                raise ValueError("NodeFeatures: node_id and swc_type must be 1-D")
            if ids.size and not np.issubdtype(ids.dtype, np.integer):
                raise ValueError("NodeFeatures: node_id must be integers")
            if typ.size and not np.issubdtype(typ.dtype, np.integer):
                raise ValueError("NodeFeatures: swc_type must be integers")
            ids = ids.astype(np.int64)
            typ = typ.astype(np.int64)
            d = np.asarray(self.d_um, dtype=np.float64)
            mu = np.asarray(self.mu_per_um, dtype=np.float64)
            off = np.asarray(self.offset_um, dtype=np.float64)
        except (TypeError, OverflowError) as exc:
            raise ValueError("NodeFeatures: %s" % exc) from None
        src = np.asarray(self.source).astype(str)
        flg = np.asarray(self.flags).astype(str)
        if src.ndim == 1 and flg.ndim == 1:  # canonical width: the longest string held
            src = np.asarray(src.tolist(), dtype=str).reshape(-1)
            flg = np.asarray(flg.tolist(), dtype=str).reshape(-1)
        n = ids.shape[0]
        if (typ.shape != (n,) or d.shape != (n,) or mu.shape != (n,) or off.shape != (n, 3)
                or src.shape != (n,) or flg.shape != (n,)):
            raise ValueError("NodeFeatures: arrays of unequal length (node_id (N,), offset_um (N, 3))")
        if np.unique(ids).size != n:
            raise ValueError("NodeFeatures: a node_id appears twice")
        if not (np.all(np.isfinite(d)) and np.all(np.isfinite(mu)) and np.all(np.isfinite(off))):
            raise ValueError("NodeFeatures: non-finite entry in d_um, mu_per_um or offset_um")
        if np.any(d <= 0.0):
            raise ValueError("NodeFeatures: d_um must be > 0")
        if np.any(mu < 0.0):
            raise ValueError("NodeFeatures: mu_per_um must be >= 0")
        bad = sorted(set(src.tolist()) - set(SOURCES))
        if bad:
            raise ValueError("NodeFeatures: unknown source %r (allowed %s)" % (bad[0], SOURCES))
        r = self.round_index
        if isinstance(r, (bool, np.bool_)) or not isinstance(r, (int, np.integer)) or int(r) < 0:
            raise ValueError("NodeFeatures: round_index must be an integer >= 0")
        setf("node_id", _frozen(ids))
        setf("swc_type", _frozen(typ))
        setf("d_um", _frozen(d))
        setf("mu_per_um", _frozen(mu))
        setf("offset_um", _frozen(off))
        setf("source", _frozen(src))
        setf("flags", _frozen(flg))
        setf("round_index", int(r))

    def __len__(self):
        return int(self.node_id.shape[0])

    def index_of(self):
        """Map node id -> row of the table."""
        return {int(i): k for k, i in enumerate(self.node_id)}


@dataclass(frozen=True)
class FeaturedScene:
    """seg_xyz (n, 6) um (parent end x0, y0, z0, child end x1, y1, z1, moved by the offsets);
    seg_d_um (n, 2) um and seg_mu_per_um (n, 2) 1/um (parent end, child end); seg_ids (n, 2)
    int64 (child, parent); seg_type (n,) int64 (the child's SWC type); soma (m, 3) = x, y, r um."""

    seg_xyz: np.ndarray
    seg_d_um: np.ndarray
    seg_mu_per_um: np.ndarray
    seg_ids: np.ndarray
    seg_type: np.ndarray
    soma: np.ndarray


def initial_features(swc, types, mu0_per_um):
    """Round 0 over the SWC nodes of `types` that have a parent in the file, in file order:
    d = twice Allen's radius, mu = mu0_per_um, offsets 0, source 'allen', flags '' -> NodeFeatures."""
    pidx = swc.parent_index()
    rows = np.flatnonzero(swc.dendrite_mask(types) & (pidx >= 0))
    n = rows.size
    mu0 = float(mu0_per_um)
    return NodeFeatures(node_id=np.asarray(swc.ids)[rows], swc_type=np.asarray(swc.types)[rows],
                        d_um=2.0 * np.asarray(swc.radius, dtype=np.float64)[rows],
                        mu_per_um=np.full(n, mu0), offset_um=np.zeros((n, 3)),
                        source=np.full(n, "allen"), flags=np.full(n, ""), round_index=0)


def with_rows(features, rows, round_index):
    """A new NodeFeatures at round_index: the nodes named in rows (dicts with 'node_id' and any of
    'd_um', 'mu_per_um', 'offset_um', 'source', 'flags') replaced, the others carried; the input
    is not changed. ValueError: an unknown node_id, a key not in ROW_KEYS, a node named twice,
    or a replaced value the table refuses."""
    pos = features.index_of()
    d = features.d_um.copy()
    mu = features.mu_per_um.copy()
    off = features.offset_um.copy()
    src = features.source.astype(object)
    flg = features.flags.astype(object)
    seen = set()
    for row in rows:
        extra = set(row) - set(ROW_KEYS)
        if extra:
            raise ValueError("with_rows: unknown key %r" % sorted(extra)[0])
        if "node_id" not in row:
            raise ValueError("with_rows: a row without node_id")
        nid = int(row["node_id"])
        if nid not in pos:
            raise ValueError("with_rows: unknown node_id %d" % nid)
        if nid in seen:
            raise ValueError("with_rows: node_id %d named twice" % nid)
        seen.add(nid)
        k = pos[nid]
        if "d_um" in row:
            d[k] = float(row["d_um"])
        if "mu_per_um" in row:
            mu[k] = float(row["mu_per_um"])
        if "offset_um" in row:
            o = np.asarray(row["offset_um"], dtype=np.float64)
            if o.shape != (3,):
                raise ValueError("with_rows: offset_um of node %d must have shape (3,)" % nid)
            off[k] = o
        if "source" in row:
            src[k] = str(row["source"])
        if "flags" in row:
            flg[k] = str(row["flags"])
    return NodeFeatures(node_id=features.node_id, swc_type=features.swc_type, d_um=d, mu_per_um=mu,
                        offset_um=off, source=src.astype(str), flags=flg.astype(str),
                        round_index=round_index)


def featured_scene(scene, features):
    """plane_iteration.Scene + NodeFeatures -> FeaturedScene. Each segment end whose node is in
    features is moved by its offset and takes its d and mu; a parent end whose node is not keeps
    the scene's position and 2 x seg_radius and takes the child end's mu. ValueError when a
    segment's child is not in features (unknown node_id)."""
    pos = features.index_of()
    xyz = np.array(scene.seg_xyz, dtype=np.float64, copy=True)
    d = 2.0 * np.asarray(scene.seg_radius, dtype=np.float64)
    ids = np.asarray(scene.seg_ids).astype(np.int64)
    n = ids.shape[0]
    mu = np.empty((n, 2))
    typ = np.empty(n, dtype=np.int64)
    for q in range(n):
        child, parent = int(ids[q, 0]), int(ids[q, 1])
        if child not in pos:
            raise ValueError("featured_scene: segment %d's child node %d is not in the features" % (q, child))
        kc = pos[child]
        xyz[q, 3:] = xyz[q, 3:] + features.offset_um[kc]
        d[q, 1] = features.d_um[kc]
        mu[q, 1] = features.mu_per_um[kc]
        typ[q] = features.swc_type[kc]
        if parent in pos:
            kp = pos[parent]
            xyz[q, :3] = xyz[q, :3] + features.offset_um[kp]
            d[q, 0] = features.d_um[kp]
            mu[q, 0] = features.mu_per_um[kp]
        else:
            mu[q, 0] = mu[q, 1]
    return FeaturedScene(seg_xyz=xyz, seg_d_um=d, seg_mu_per_um=mu, seg_ids=ids, seg_type=typ,
                         soma=np.array(scene.soma, dtype=np.float64, copy=True).reshape(-1, 3))


def max_relative_change(a, b):
    """The stop rule's inputs (D-130.O6) from round a to round b over the same nodes (matched by
    node_id): 'd' = max_n |d_b - d_a| / d_a; 'mu' = the same over the nodes with mu_a > 0 (NaN when
    there is none); 'offset_um' = max_n of the Euclidean norm of offset_b - offset_a (um). NaN for
    'd' and 'offset_um' on an empty table. ValueError when the node sets differ."""
    if set(a.node_id.tolist()) != set(b.node_id.tolist()):
        raise ValueError("max_relative_change: the two tables hold different nodes")
    pos = b.index_of()
    order = np.array([pos[int(i)] for i in a.node_id], dtype=np.int64)
    db, mub, ob = b.d_um[order], b.mu_per_um[order], b.offset_um[order]
    if a.node_id.size == 0:
        return {"d": float("nan"), "mu": float("nan"), "offset_um": float("nan")}
    rd = np.abs(db - a.d_um) / a.d_um
    pm = a.mu_per_um > 0.0
    rmu = np.abs(mub[pm] - a.mu_per_um[pm]) / a.mu_per_um[pm]
    doff = np.sqrt(np.sum((ob - a.offset_um) ** 2, axis=1))
    return {"d": float(np.max(rd)), "mu": float(np.max(rmu)) if rmu.size else float("nan"),
            "offset_um": float(np.max(doff))}
