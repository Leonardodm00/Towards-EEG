"""The cleaning iteration -- Block 32 in specs/SPEC.md (plan step C5; D-129.2, D-130.O3, D-130.O6, D-132).

Features theta^(0) given (Block 23). For each round t = 1, 2, ... and each
node i of the work list, from the features of round t - 1 only (a Jacobi
update: the order of the nodes does not matter):

  (a) F^(t-1) = node_features.featured_scene(scene, features)
  (b) tiers   = cleaning.neighbour_tiers(F, own_ids_i, node_xyz_i, cfg)
  (c) S^(t)_ik = the shadow of the tier-1 neighbours on node i's patch
      (shadow_fn; by default shadow.shadow_image(..., include = tiers == 1).S)
  (d) I^(t)_ik = I_ik + B_ik S^(t)_ik                 (cleaning.clean_block)
  (e) I^(t)_ik(v_n) = profiles.sample_profile of each cleaned plane on the
      node's measuring line
  (f) theta^(t)_i = coarse_fit.refine_node(profiles, B_k, v, predict_i, start,
      cfg), predict_i = predict_factory(node_i), started at the node's values
      of round t - 1 (round 1: theta0, or (d^(0), mu^(0), 0, 0) when None).

A failed fit leaves the node's row unchanged and adds the flag refine_failed;
at_bound updates the row and adds refine_at_bound. After all nodes:
features <- with_rows(features, rows, t), offset_um = coarse_fit.offset_xyz
(v0, dz, y_hat) (the total offset, D-136.2), source "refined".

Stop (D-130.O6): after round t when max_i |d^(t)_i - d^(t-1)_i| / d^(t-1)_i
< stop_rel_d over the refined nodes and max_{i in E} max_{k,n}
|I^(t)_ik(v_n) - I^(t-1)_ik(v_n)| < sigma_gl over the example nodes E (round
1 against the uncleaned profiles), or at max_rounds.

Custom code: none numerical (the loop and the stop rule). No file access, no
plotting. Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import dataclasses

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from . import cleaning
from . import coarse_fit
from .node_features import featured_scene, with_rows
from .profiles import sample_profile

STOPPED = ("converged", "max_rounds")
FLAG_FAILED = "refine_failed"
FLAG_AT_BOUND = "refine_at_bound"
HISTORY_KEYS = ("round", "max_rel_d", "max_line_change_gl", "n_refined", "n_failed", "n_at_bound")


def _ro(x, dtype=None):
    a = np.array(x, dtype=dtype, copy=True)
    a.setflags(write=False)
    return a


@dataclass(frozen=True)
class NodeData:
    """One node of the work list. node_id int; own_ids tuple of int (the stretch's node ids without the
    prepended parent, D-136.1); node_xyz (3,) um, the traced position; block (n_planes, H, W) gl, observed;
    B_gl (n_planes,) gl > 0; ks (n_planes,) int plane indices; planes_z (n_planes,) um; left, top int, the
    patch's corner pixel (full resolution); origin_xy (2,) um, y_hat (2,), e_u (2,), v (n_v,) um, n_avg int,
    step float um: the measuring line of Block 5; theta0 (4,) = (d um, mu 1/um, v0 um, dz um) or None.
    The arrays are copied and made read-only. ValueError for shapes that disagree or non-finite entries."""

    node_id: int
    own_ids: Tuple[int, ...]
    node_xyz: np.ndarray
    block: np.ndarray
    B_gl: np.ndarray
    ks: np.ndarray
    planes_z: np.ndarray
    left: int
    top: int
    origin_xy: np.ndarray
    y_hat: np.ndarray
    e_u: np.ndarray
    v: np.ndarray
    n_avg: int
    step: float
    theta0: Optional[np.ndarray] = None

    def __post_init__(self):
        def setf(name, value):
            object.__setattr__(self, name, value)

        def as_int(name, value):
            if isinstance(value, (bool, np.bool_)) or int(value) != value:
                raise ValueError("NodeData %s must be an integer" % name)
            return int(value)

        nid = as_int("node_id", self.node_id)
        own = tuple(as_int("own_ids", i) for i in self.own_ids)
        if not own:
            raise ValueError("NodeData %d: own_ids is empty" % nid)
        blk = np.array(self.block, copy=True)
        if blk.ndim != 3 or blk.dtype.kind not in "iuf" or min(blk.shape) < 1:
            raise ValueError("NodeData %d: block must be a non-empty real (n_planes, H, W) array" % nid)
        n_planes = blk.shape[0]
        B = np.asarray(self.B_gl, dtype=np.float64)
        ks = np.asarray(self.ks)
        z = np.asarray(self.planes_z, dtype=np.float64)
        if B.shape != (n_planes,) or ks.shape != (n_planes,) or z.shape != (n_planes,):
            raise ValueError("NodeData %d: B_gl, ks and planes_z must be (n_planes,) = (%d,)" % (nid, n_planes))
        if ks.dtype.kind not in "iu":
            raise ValueError("NodeData %d: ks must be integer plane indices" % nid)
        xyz = np.asarray(self.node_xyz, dtype=np.float64)
        o = np.asarray(self.origin_xy, dtype=np.float64)
        yh = np.asarray(self.y_hat, dtype=np.float64)
        eu = np.asarray(self.e_u, dtype=np.float64)
        v = np.asarray(self.v, dtype=np.float64)
        if xyz.shape != (3,) or o.shape != (2,) or yh.shape != (2,) or eu.shape != (2,):
            raise ValueError("NodeData %d: node_xyz (3,), origin_xy, y_hat, e_u (2,)" % nid)
        if v.ndim != 1 or v.size < 1:
            raise ValueError("NodeData %d: v must be a non-empty 1-D array (um)" % nid)
        step = float(self.step)
        n_avg = as_int("n_avg", self.n_avg)
        if n_avg < 0:
            raise ValueError("NodeData %d: n_avg must be >= 0" % nid)
        finite = [blk, B, z, xyz, o, yh, eu, v, np.array(step)]
        th = None
        if self.theta0 is not None:
            th = np.asarray(self.theta0, dtype=np.float64)
            if th.shape != (4,):
                raise ValueError("NodeData %d: theta0 must be (d, mu, v0, dz), shape (4,)" % nid)
            finite.append(th)
        if not all(np.all(np.isfinite(a)) for a in finite):
            raise ValueError("NodeData %d: non-finite entry" % nid)
        if np.any(B <= 0):
            raise ValueError("NodeData %d: B_gl must be > 0" % nid)
        setf("node_id", nid)
        setf("own_ids", own)
        setf("node_xyz", _ro(xyz))
        setf("block", _ro(blk))
        setf("B_gl", _ro(B))
        setf("ks", _ro(ks, np.int64))
        setf("planes_z", _ro(z))
        setf("left", as_int("left", self.left))
        setf("top", as_int("top", self.top))
        setf("origin_xy", _ro(o))
        setf("y_hat", _ro(yh))
        setf("e_u", _ro(eu))
        setf("v", _ro(v))
        setf("n_avg", n_avg)
        setf("step", step)
        setf("theta0", None if th is None else _ro(th))


@dataclass(frozen=True)
class IterationResult:
    """features: the last round's NodeFeatures; history: list of dicts per round (HISTORY_KEYS); stopped in
    STOPPED; rounds int; profiles_before / profiles_after: node_id -> (n_planes, n_v) gl, the uncleaned and
    the last round's cleaned profiles, every node; cleaned / shadow: node_id -> (n_planes, H, W) gl, the last
    round's cleaned block and B S, the nodes in keep_ids; tiers: node_id -> (n_q + n_s,) int; fits:
    node_id -> the last refine_node dict."""

    features: object
    history: list
    stopped: str
    rounds: int
    profiles_before: dict
    profiles_after: dict
    cleaned: dict
    shadow: dict
    tiers: dict
    fits: dict
    B_gl: dict = dataclasses.field(default_factory=dict)   # node_id -> (n_planes,) gl, the last round's backgrounds (amendment 2)


# ------------------------------------------------------------- helpers ---

def _frame(node, cfg):
    from allen_image_io import CropFrame     # the contract's frame type (flat module in src/)
    return CropFrame(node.left, node.top, 0, float(cfg.acquisition.res0_um))


def _profiles(block, node, cfg):
    """(n_planes, n_v) gl: sample_profile of every plane of block on the node's line."""
    frame = _frame(node, cfg)
    return np.stack([sample_profile(block[k], frame, node.origin_xy, node.y_hat, node.e_u, node.v,
                                    node.n_avg, node.step) for k in range(block.shape[0])])


def _default_shadow_fn(fscene, own_ids, left, top, width, height, planes_z, cfg, B_gl, include):
    from ..model import shadow
    return shadow.shadow_image(fscene, own_ids, left, top, width, height, planes_z, cfg,
                               B_gl=B_gl, include=include).S


def _check_nodes(nodes):
    nodes = list(nodes)
    if not nodes:
        raise ValueError("the work list is empty")
    for n in nodes:
        if not isinstance(n, NodeData):
            raise ValueError("every work-list entry must be a NodeData")
    ids = [n.node_id for n in nodes]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate node ids in the work list")
    return nodes


def _with_flag(flags, add):
    """';'-joined flags with refine_failed / refine_at_bound replaced by add ('' = none)."""
    kept = [f for f in str(flags).split(";") if f and f not in (FLAG_FAILED, FLAG_AT_BOUND)]
    if add:
        kept.append(add)
    return ";".join(kept)


def _start(features, k, node):
    """The fit's start: the node's values of the round before (a refined row), else theta0, else
    (d^(0), mu^(0), 0, 0)."""
    d, mu = float(features.d_um[k]), float(features.mu_per_um[k])
    if str(features.source[k]) == "refined":
        off = features.offset_um[k]
        yh = node.y_hat
        v0 = float(off[0] * yh[0] + off[1] * yh[1]) / float(yh[0] * yh[0] + yh[1] * yh[1])
        return np.array([d, mu, v0, float(off[2])])
    if node.theta0 is not None:
        return np.array(node.theta0, dtype=np.float64)
    return np.array([d, mu, 0.0, 0.0])


# ---------------------------------------------------------------- round ---

def round_once(features, scene, nodes, cfg, predict_factory, shadow_fn=None, background_fn=None):
    """One round from the features of the round before -> (rows, per_node).

    features: NodeFeatures (round t - 1); scene: plane_iteration.Scene; nodes: NodeData work list;
    cfg: DiameterConfig; predict_factory(node) -> predict(d, mu, v0, dz) -> (n_planes, n_v) shadow;
    shadow_fn(fscene, own_ids, left, top, width, height, planes_z, cfg, B_gl, include) -> S
    (n_planes, H, W), None = shadow.shadow_image(...).S. rows: the with_rows dicts, sorted by node id
    (a failed fit: node_id and flags only). per_node: node_id -> dict with tiers, S, shadow_gl (B S),
    cleaned, profiles (gl), start, fit. ValueError for an empty work list, duplicate ids, a node not in
    features, an S of the wrong shape, a measuring line that leaves the patch."""
    nodes = _check_nodes(nodes)
    shadow_fn = _default_shadow_fn if shadow_fn is None else shadow_fn
    pos = features.index_of()
    for n in nodes:
        if n.node_id not in pos:
            raise ValueError("node %d is not in the features" % n.node_id)
    F = featured_scene(scene, features)                                  # (a), once: Jacobi
    rows, per_node = [], {}
    for node in nodes:
        n_planes, H, W = node.block.shape
        # The planes' backgrounds of this round (Block 32, amendment 2, D-144): background_fn(node, F) -> (n_planes,)
        # gl, from the featured scene of the round before; None: the node's own B_gl, every round.
        B_k = node.B_gl if background_fn is None else np.asarray(background_fn(node, F), dtype=np.float64)
        if B_k.shape != (n_planes,) or not np.all(np.isfinite(B_k)) or np.any(B_k <= 0):
            raise ValueError("node %d: background_fn must return (n_planes,) finite values > 0, got %r"
                             % (node.node_id, B_k))
        tiers = cleaning.neighbour_tiers(F, node.own_ids, node.node_xyz, cfg)            # (b)
        # The floor of D-130.O5 takes one background in shadow_image: max_k B_k (Block 32, amendment 1,
        # D-139); the cleaning (d) uses the per-plane B_k.
        B_floor = float(np.max(B_k))
        S = np.asarray(shadow_fn(F, node.own_ids, node.left, node.top, W, H, node.planes_z, cfg,
                                 B_floor, tiers == 1), dtype=np.float64)                  # (c)
        if S.shape != node.block.shape:
            raise ValueError("node %d: shadow_fn returned shape %r, expected %r"
                             % (node.node_id, S.shape, node.block.shape))
        cr = cleaning.clean_block(node.block, S, B_k, cfg)                          # (d)
        prof = _profiles(cr.block, node, cfg)                                            # (e)
        if not np.all(np.isfinite(prof)):
            raise ValueError("node %d: the measuring line leaves the patch (NaN profile)" % node.node_id)
        k = pos[node.node_id]
        start = _start(features, k, node)
        fit = coarse_fit.refine_node(prof, B_k, node.v, predict_factory(node), start, cfg)   # (f)
        if fit["status"] == "failed":
            row = {"node_id": node.node_id, "flags": _with_flag(features.flags[k], FLAG_FAILED)}
        else:
            row = {"node_id": node.node_id, "d_um": fit["d_um"], "mu_per_um": fit["mu_per_um"],
                   "offset_um": coarse_fit.offset_xyz(fit["v0_um"], fit["dz_um"], node.y_hat),
                   "source": "refined",
                   "flags": _with_flag(features.flags[k], FLAG_AT_BOUND if fit["status"] == "at_bound" else "")}
        rows.append(row)
        per_node[node.node_id] = {"tiers": tiers, "S": S, "shadow_gl": B_k[:, None, None] * S, "B_gl": np.array(B_k),
                                  "cleaned": cr.block, "profiles": prof, "start": start, "fit": fit}
    rows.sort(key=lambda r: r["node_id"])
    return rows, per_node


# --------------------------------------------------------------- rounds ---

def _stop_reached(max_rel_d, max_line_change_gl, stop_rel_d, sigma_gl):
    """The stop rule of D-130.O6: both changes below their thresholds. A NaN max_rel_d (no node refined)
    never stops; a NaN line change (no example node) does not hold the stop back."""
    d_ok = bool(max_rel_d < stop_rel_d)
    line_ok = bool(math.isnan(max_line_change_gl) or max_line_change_gl < sigma_gl)
    return d_ok and line_ok


def run_rounds(features0, scene, nodes, cfg, sigma_gl, predict_factory, example_ids=(), shadow_fn=None,
               keep_ids=None, background_fn=None):
    """Rounds of round_once until the stop rule of D-130.O6 or CleaningConfig.max_rounds -> IterationResult.

    sigma_gl: the noise SD (gl, > 0), the line-change threshold; example_ids: the example nodes E (ids in
    the work list); keep_ids: the nodes whose cleaned block and B S are kept (None: example_ids). Inputs are
    not changed. ValueError for an empty work list, duplicate node ids, sigma_gl <= 0, an example or kept
    id not in the work list, max_rounds < 1."""
    nodes = _check_nodes(nodes)
    sigma = float(sigma_gl)
    if not (math.isfinite(sigma) and sigma > 0):
        raise ValueError("sigma_gl must be finite and > 0")
    max_rounds = int(cfg.cleaning.max_rounds)
    if max_rounds < 1:
        raise ValueError("cleaning.max_rounds must be >= 1")
    stop_rel_d = float(cfg.cleaning.stop_rel_d)
    work = {n.node_id for n in nodes}
    E = [int(i) for i in example_ids]
    if any(i not in work for i in E):
        raise ValueError("an example id is not in the work list")
    keep = E if keep_ids is None else [int(i) for i in keep_ids]
    if any(i not in work for i in keep):
        raise ValueError("a kept id is not in the work list")

    before = {n.node_id: _profiles(np.asarray(n.block, dtype=np.float64), n, cfg) for n in nodes}
    prev = before
    features = features0
    history = []
    stopped = "max_rounds"
    per_node = {}
    t = 0
    for t in range(1, max_rounds + 1):
        rows, per_node = round_once(features, scene, nodes, cfg, predict_factory, shadow_fn, background_fn)
        new = with_rows(features, rows, t)
        statuses = {nid: per_node[nid]["fit"]["status"] for nid in per_node}
        refined = sorted(nid for nid, s in statuses.items() if s != "failed")
        po, pn = features.index_of(), new.index_of()
        if refined:
            max_rel_d = max(abs(float(new.d_um[pn[i]]) - float(features.d_um[po[i]])) / float(features.d_um[po[i]])
                            for i in refined)
        else:
            max_rel_d = float("nan")
        if E:
            line = max(float(np.max(np.abs(per_node[i]["profiles"] - prev[i]))) for i in E)
        else:
            line = float("nan")
        history.append({"round": t, "max_rel_d": float(max_rel_d), "max_line_change_gl": line,
                        "n_refined": len(refined),
                        "n_failed": sum(1 for s in statuses.values() if s == "failed"),
                        "n_at_bound": sum(1 for s in statuses.values() if s == "at_bound")})
        features = new
        prev = {nid: per_node[nid]["profiles"] for nid in per_node}
        if _stop_reached(max_rel_d, line, stop_rel_d, sigma):
            stopped = "converged"
            break
    return IterationResult(
        features=features, history=history, stopped=stopped, rounds=t,
        profiles_before=before, profiles_after={nid: per_node[nid]["profiles"] for nid in per_node},
        cleaned={i: per_node[i]["cleaned"] for i in keep}, shadow={i: per_node[i]["shadow_gl"] for i in keep},
        tiers={nid: per_node[nid]["tiers"] for nid in per_node}, fits={nid: per_node[nid]["fit"] for nid in per_node},
        B_gl={nid: per_node[nid]["B_gl"] for nid in per_node})
