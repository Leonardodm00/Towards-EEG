"""The screening views: a calibration node's planes, profiles and Allen's-diameter marks -- Block 61 in
specs/SPEC.md (section 5, "The screening export"; D-226, D-227, D-261.1, D-278). Arrays only: no file access, no
plotting.

Implemented 2026-11-27 (wave 10) against the contract of commit 500426e.
Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .calibration_scene import patch_bounds


@dataclass(frozen=True)
class NodeView:
    """What one node's screening sheet shows. Arrays over the scan's n planes; units um, rad, grey levels (gl),
    um^2."""

    cell_id: int
    node_id: int
    set: str                    # "near" | "far"
    allen_diameter_um: float
    k_star: int
    ks: np.ndarray              # (n,) int64
    stack: np.ndarray           # (n, H, W) uint8: the crop of the raw image stack
    valid: np.ndarray           # (n,) bool: False where the plane could not be fetched (its image zeros)
    left: int                   # px of the full image
    top: int
    res0_um: float
    cx_um: float                # the pilot's centre, heading and tilt
    cy_um: float
    theta_rad: float
    phi_rad: float
    v_um: np.ndarray            # (n_v,) offsets along the measuring line
    profile_gl: np.ndarray      # (n, n_v): I_k(v) = bhat_gl - w_gl (NaN where the scan has none)
    background_gl: np.ndarray   # (n, n_v): the scan's fitted background along the line
    status: Tuple[str, ...]     # (n,)
    omega_um2: np.ndarray       # (n,)
    B_bar: np.ndarray           # (n,)


_SETS = ("near", "far")


def _records_by_id(nodes):
    """{node_id: record} from a list of node records (meta["nodes"]) or a dict of them (CellScans.nodes)."""
    recs = list(nodes.values()) if isinstance(nodes, Mapping) else list(nodes)
    out = {}
    for rec in recs:
        nid = int(rec["node_id"])
        if nid in out:
            raise ValueError("node %d: two node records with this node id" % nid)
        out[nid] = rec
    return out


def _set_rank(which):
    return 0 if which == "near" else 1


def apt_nodes(scans, nodes):
    """list[(PlaneScan, node record)]: the scans with their records, near set first, then by node id. ValueError
    naming the node: a scan without a record, a record without a pilot."""
    by_id = _records_by_id(nodes)
    pairs = []
    for sc in scans:
        nid = int(sc.node_id)
        rec = by_id.get(nid)
        if rec is None:
            raise ValueError("node %d: a calibration scan without a node record" % nid)
        if rec.get("pilot") is None:
            raise ValueError("node %d: its node record has no pilot" % nid)
        pairs.append((sc, rec))
    pairs.sort(key=lambda pr: (_set_rank(pr[0].set), int(pr[0].node_id)))
    return pairs


def _half_um(which, cfg):
    if which == "near":
        return float(cfg.survey.screening_near_half_um)
    if which == "far":
        return float(cfg.survey.screening_far_half_um)
    raise ValueError("which must be 'near' or 'far', not %r" % (which,))


def _pilot_float(pilot, name):
    v = pilot.get(name)
    return float("nan") if v is None else float(v)


def crop_bounds(record, which, cfg):
    """(left, top, width, height) px: calibration_scene.patch_bounds about the pilot's (cx_um, cy_um) with
    cfg.survey.screening_near_half_um (which "near") or screening_far_half_um ("far") and acquisition.res0_um."""
    half = _half_um(which, cfg)
    pilot = record.get("pilot")
    nid = int(record["node_id"])
    if pilot is None:
        raise ValueError("node %d: its node record has no pilot" % nid)
    cx, cy = _pilot_float(pilot, "cx_um"), _pilot_float(pilot, "cy_um")
    if not (math.isfinite(cx) and math.isfinite(cy)):
        raise ValueError("node %d: the pilot's centre (%r, %r) is not finite" % (nid, cx, cy))
    return patch_bounds(cx, cy, half, float(cfg.acquisition.res0_um))


def node_view(cell_id, scan, record, patch) -> NodeView:
    """The view of one node: patch = (block, ks, valid, frame) in the fetch_zblock contract over the scan's planes.
    ValueError: the patch's ks are not the scan's ks; a scan without v_um, w_gl or bhat_gl; a non-finite
    allen_diameter_um."""
    nid = int(scan.node_id)
    if record is None or int(record["node_id"]) != nid:
        raise ValueError("node %d: the node record is not this scan's" % nid)
    pilot = record.get("pilot")
    if pilot is None:
        raise ValueError("node %d: its node record has no pilot" % nid)
    d_a = record.get("allen_diameter_um")
    d_a = float("nan") if d_a is None else float(d_a)
    if not math.isfinite(d_a):
        raise ValueError("node %d: allen_diameter_um %r is not finite" % (nid, record.get("allen_diameter_um")))
    missing = [name for name in ("v_um", "w_gl", "bhat_gl") if getattr(scan, name) is None]
    if missing:
        raise ValueError("node %d: the scan has no %s (the per-plane profiles of D-075)" % (nid, ", ".join(missing)))
    if scan.set not in _SETS:
        raise ValueError("node %d: the scan's set %r is not 'near' or 'far'" % (nid, scan.set))

    block, pks, valid, frame = patch
    ks = np.asarray(scan.ks, dtype=np.int64)
    pks = np.asarray(pks)
    if pks.shape != ks.shape or not np.array_equal(pks.astype(np.int64), ks):
        raise ValueError("node %d: the patch's planes %s are not the scan's planes %s" % (nid, pks.tolist(), ks.tolist()))
    n = ks.size
    block = np.asarray(block)
    if block.dtype != np.uint8 or block.ndim != 3 or block.shape[0] != n:
        raise ValueError("node %d: the patch's block must be (%d, H, W) uint8, not %s %s"
                         % (nid, n, block.shape, block.dtype))
    valid = np.asarray(valid)
    if valid.shape != (n,):
        raise ValueError("node %d: the patch's valid must be (%d,), not %s" % (nid, n, valid.shape))
    valid = valid.astype(bool)
    stack = block.copy()
    stack[~valid] = 0          # NodeView: a plane that could not be fetched has its image zeros

    v_um = np.array(scan.v_um, dtype=np.float64)
    w = np.asarray(scan.w_gl, dtype=np.float64)
    bhat = np.array(scan.bhat_gl, dtype=np.float64)
    if v_um.ndim != 1 or w.shape != (n, v_um.size) or bhat.shape != (n, v_um.size):
        raise ValueError("node %d: w_gl %s and bhat_gl %s must be (n, n_v) = (%d, %d)"
                         % (nid, w.shape, bhat.shape, n, v_um.size))
    profile = bhat - w         # I_k(v) = B_hat_k(v) - w_k(v) (D-227.1); NaN where the scan has none

    return NodeView(
        cell_id=int(cell_id), node_id=nid, set=str(scan.set), allen_diameter_um=d_a, k_star=int(scan.k_star),
        ks=ks.copy(), stack=stack, valid=valid.copy(), left=int(frame.left), top=int(frame.top),
        res0_um=float(frame.res0_um_px), cx_um=_pilot_float(pilot, "cx_um"), cy_um=_pilot_float(pilot, "cy_um"),
        theta_rad=_pilot_float(pilot, "theta_rad"), phi_rad=_pilot_float(pilot, "phi_rad"), v_um=v_um,
        profile_gl=profile, background_gl=bhat, status=tuple(str(s) for s in scan.status),
        omega_um2=np.array(scan.omega_um2, dtype=np.float64), B_bar=np.array(scan.B_bar, dtype=np.float64))


def diameter_marks(view):
    """{"v_um": (-D/2, +D/2), "xy_um": (2, 2) array}: Allen's diameter D on the measuring line (D-261.1), the image
    points (cx_um, cy_um) -+ (D / 2) (-sin theta, cos theta)."""
    h = 0.5 * float(view.allen_diameter_um)
    v = np.array([-h, h], dtype=np.float64)
    u = np.array([-math.sin(view.theta_rad), math.cos(view.theta_rad)])   # the line's unit vector in (x, y)
    xy = np.array([view.cx_um, view.cy_um], dtype=np.float64)[None, :] + v[:, None] * u[None, :]
    return {"v_um": (float(v[0]), float(v[1])), "xy_um": xy}


def image_extent(view):
    """(x0, x1, y0, y1) um, the crop's outer pixel edges with pixel (row, col) centred on ((left + col) p,
    (top + row) p): for imshow(extent=(x0, x1, y1, y0), origin="upper"), y increasing downwards."""
    p = float(view.res0_um)
    H, W = int(view.stack.shape[1]), int(view.stack.shape[2])
    return ((view.left - 0.5) * p, (view.left + W - 0.5) * p, (view.top - 0.5) * p, (view.top + H - 0.5) * p)


def index_row(view):
    """dict with specimen_id, node_id, set, allen_diameter_um, k_star, phi_deg, n_planes, n_ok, B_focus_gl,
    omega_focus_um2 (the plane k*; NaN if absent)."""
    at = np.flatnonzero(np.asarray(view.ks) == int(view.k_star))
    b_focus = float(view.B_bar[at[0]]) if at.size else float("nan")
    om_focus = float(view.omega_um2[at[0]]) if at.size else float("nan")
    return {"specimen_id": int(view.cell_id), "node_id": int(view.node_id), "set": str(view.set),
            "allen_diameter_um": float(view.allen_diameter_um), "k_star": int(view.k_star),
            "phi_deg": math.degrees(float(view.phi_rad)), "n_planes": int(len(view.ks)),
            "n_ok": int(sum(1 for s in view.status if s == "ok")), "B_focus_gl": b_focus,
            "omega_focus_um2": om_focus}
