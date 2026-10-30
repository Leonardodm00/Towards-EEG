"""Block 10.C in specs/SPEC.md -- the calibration candidates found over the
whole cell, for both sets (D-109, the far set, and D-112, both sets,
2026-10-29; the user: "Just let the colab code for the pilot cell find the
nodes for the far field calibration", then "the whole cell's nodes must be
evaluated for both near and far field calibration").

Before any image is read, every dendrite node of the cell whose Allen
diameter puts it in calibration set `which` (calibration.calibration_set:
"near" or "far") is judged from the trace alone:

  (i)  the trace's direction at the node -- the oriented TLS direction of the
       branch points within +- line_fit_window_um / 2 along the path, Block
       5's raw direction -- its heading theta and tilt phi; not_flat when phi
       exceeds the set's flatness bound (node_phi_max_deg near,
       node_phi_max_deg_far far);
  (ii) the stripes of Block 10.S on the patch frame about the trace's node,
       with the trace's heading, the trace's plane k_tr = round(z / dz) as
       the sharpest plane, Allen's radius for the own cylinder, the cell's
       typical background and absorption (typical_background: the medians of
       the pilot table's rows in S) and the soma at the set's stage blur
       sigma_r(P dz) (P = offsets_planes near, far_offsets_planes far):
       first at k_tr (not_isolated:too_few / not_isolated:side_short --
       D-101.4's verdict at the trace's plane), then at the set's fit planes:
         far:  k_tr + o, far_min_k_offset <= |o| <= far_offsets_planes (24
               planes), every one evaluated; few_planes when fewer than
               far_min_planes_per_node have a usable stripe;
         near: k_tr + o, |o| <= offsets_planes (k_tr itself counted), the
               pairs +-o evaluated from |o| = 1 outward until
               min_planes_per_node planes are usable or can no longer be;
               few_planes when fewer than min_planes_per_node are (the near
               fit's own minimum of finite widths per node).

choose() then keeps the passing nodes greedily -- far: the most usable fit
planes first, then the most clear pixels at k_tr; near: the most clear pixels
at k_tr (its usable-plane count stops at the minimum, so it ranks nothing) --
then the smaller node id, each at least the set's spacing
(near_search_spacing_um, far_search_spacing_um) along the path from every
node already kept on its stretch. The chosen nodes go to calibration.scan_cell
like any candidate: their pilot is measured and they are judged again on it.
Analysis only: no file is read or written here, nothing is drawn.

  trace_direction(branch, i, window_um) -> (theta rad, phi rad)
  typical_background(rows, cfg)         -> (B_bar gl, mu_hat /um), NaN without rows in S
  search(swc, cfg, which, transform, B_bar, mu, log) -> list of records (dict), one per node of the set's range
  choose(records, cfg, which)           -> list of node ids, in the order kept
  summary_line(records, chosen, cfg, which) -> the log line of calibrate_kernel.cell_candidates
  swc_digest(swc), search_config(cfg)   -> the key a search's record carries (10.C.R, D-115.6, 2026-10-30:
                                           Cell 15 reuses Cell 7's search when both equal its own)

Pure ASCII (hpc-python-compat).
"""
from __future__ import annotations

import dataclasses
import hashlib
import math
import time
import types as _types

import numpy as np

from . import calibration, calibration_scene, cell, path
from ..model.kernel import sigma_r

RECORD_KEYS = ("set", "node_id", "allen_diameter_um", "stretch", "s_um", "x_um", "y_um", "z_um", "k_trace",
               "theta_rad", "phi_deg", "reason", "n_pixels_k", "n_fit_usable", "n_fit")
# the configuration sections the search's call tree reads (this module, calibration_scene, cell, model.kernel)
SEARCH_CONFIG_SECTIONS = ("acquisition", "calibration", "measure", "renderer")


def swc_digest(swc):
    """SHA-256 (hex) of the trace a search reads (10.C.R, D-115.6): the SWC's ids, types and parent as
    little-endian int64, then xyz and radius as little-endian float64, each preceded by its name and shape; not the
    file's text lines or its path (two copies of one reconstruction digest alike)."""
    h = hashlib.sha256()
    for name, dtype in (("ids", "<i8"), ("types", "<i8"), ("parent", "<i8"), ("xyz", "<f8"), ("radius", "<f8")):
        a = np.ascontiguousarray(np.asarray(getattr(swc, name)), dtype=dtype)
        h.update(("%s%s" % (name, a.shape)).encode("ascii"))
        h.update(a.tobytes())
    return h.hexdigest()


def _plain(x):
    """x as plain JSON values: dicts with str keys, tuples and arrays as lists, numpy scalars as Python ones, a
    non-finite float as its repr string ("nan", "inf", "-inf"), so that equal settings compare equal and the
    record stays strict JSON."""
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [_plain(v) for v in (x.tolist() if isinstance(x, np.ndarray) else x)]
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return repr(x)
    return x


def search_config(cfg):
    """The settings a search depends on (10.C.R, D-115.6): the sections SEARCH_CONFIG_SECTIONS of cfg as plain
    JSON values (dataclasses.asdict, then _plain)."""
    return {name: _plain(dataclasses.asdict(getattr(cfg, name))) for name in SEARCH_CONFIG_SECTIONS}


def trace_direction(branch, i, window_um):
    """(theta, phi) in rad of the trace at point i of branch (a Block 13 Branch in the image frame): the TLS
    direction of the points within +- window_um / 2 of point i along the path (path.window; the neighbours i - 1,
    i + 1 when fewer than two points lie there), oriented from the first point to the last -- node_pipeline's
    _raw_direction -- then path.angles. ValueError when the points coincide."""
    w = path.window(branch.s_um, i, 0.5 * float(window_um))
    if w.size < 2:
        w = np.arange(max(0, i - 1), min(len(branch.ids), i + 2))
    P = np.asarray(branch.xyz_um, dtype=float)[w]
    t = path.orient(path.tls_direction(P), P[0], P[-1])
    theta, phi, _y_hat, _e_u = path.angles(t)
    return float(theta), float(phi)


def typical_background(rows, cfg):
    """(B_bar gl, mu_hat /um): the medians over the rows of a pilot table (run_node.py or run_cell.py columns,
    read as the pilot's through cell.pilot_row_view) that are in S by fit_status and the flags of
    calibration.selection_flags -- the rule candidates_from_rows applies -- and hold finite positive values. NaN
    for a quantity no such row holds (then 10.S falls back to its geometric margins)."""
    fail = calibration.selection_flags(cfg)
    B, mu = [], []
    for r in (cell.pilot_row_view(x) for x in rows):
        flags = [f for f in str(r.get("flags", "")).split(";") if f]
        if r.get("fit_status") != "converged" or any(f in fail for f in flags):
            continue
        for key, sink in (("B_bar", B), ("mu_hat_per_um", mu)):
            try:
                x = float(r.get(key, "nan"))
            except (TypeError, ValueError):
                continue
            if math.isfinite(x) and x > 0:
                sink.append(x)
    return (float(np.median(B)) if B else float("nan"), float(np.median(mu)) if mu else float("nan"))


def _check_set(which):
    if which not in calibration.CALIBRATION_SETS:
        raise ValueError("candidate_search: the set must be one of %s, got %r" % (calibration.CALIBRATION_SETS, which))


def _judge(branch, i, stretch, scene, cfg, B_bar, mu, which):
    """The record of node i of branch, in the range of set `which` (search's rule (i)-(ii))."""
    cal, dz, res0 = cfg.calibration, cfg.acquisition.dz_um, cfg.acquisition.res0_um
    x, y, z = (float(v) for v in np.asarray(branch.xyz_um, dtype=float)[i])
    k = int(round(z / dz))
    rec = dict(set=which, node_id=int(branch.ids[i]), allen_diameter_um=2.0 * float(branch.radius_um[i]),
               stretch=int(stretch), s_um=float(branch.s_um[i]), x_um=x, y_um=y, z_um=z, k_trace=k,
               theta_rad=float("nan"), phi_deg=float("nan"), reason="", n_pixels_k=0, n_fit_usable=0, n_fit=0)
    try:
        theta, phi = trace_direction(branch, i, cfg.measure.line_fit_window_um)
    except ValueError:
        rec["reason"] = "no_direction"
        return rec
    rec.update(theta_rad=theta, phi_deg=math.degrees(phi))
    near = which == "near"
    if math.degrees(phi) > (cal.node_phi_max_deg if near else cal.node_phi_max_deg_far):
        rec["reason"] = "not_flat"
        return rec
    view = _types.SimpleNamespace(cx_um=x, cy_um=y, theta_rad=theta, z_sub_um=z, k_star=k, B_bar=float(B_bar),
                                  mu_hat_per_um=float(mu), d_hat_um=float("nan"))
    shape, frame = calibration_scene.patch_frame(x, y, cal.patch_half_um, res0)
    P = int(cal.offsets_planes if near else cal.far_offsets_planes)
    sigma_soma = float(sigma_r(P * dz, cfg.renderer))           # the stage's largest blur, as the verdict (F1)

    def stripes(offsets):
        return calibration.support_masks(shape, frame, np.array([k + o for o in offsets]), view, branch, scene, cfg,
                                         sigma_soma_um=sigma_soma)[1]

    st = stripes([0])
    rec["n_pixels_k"] = int(st[0]["n_pixels"])
    if not st[0]["ok"]:
        rec["reason"] = "not_isolated:" + st[0]["reason"]
        return rec
    if near:
        need = int(cal.min_planes_per_node)
        n_fit, n_ok = 1, 1                                      # k_tr: a near fit plane, usable
        for d in range(1, P + 1):
            if n_ok >= need or n_ok + 2 * (P - d + 1) < need:
                break
            n_fit += 2
            n_ok += sum(1 for s in stripes([-d, d]) if s["ok"])
    else:
        need = int(cal.far_min_planes_per_node)
        offsets = [o for o in range(-P, P + 1) if abs(o) >= int(cal.far_min_k_offset)]
        n_fit, n_ok = len(offsets), sum(1 for s in stripes(offsets) if s["ok"])
    rec["n_fit"], rec["n_fit_usable"] = int(n_fit), int(n_ok)
    if n_ok < need:
        rec["reason"] = "few_planes"
    return rec


def search(swc, cfg, which, transform=None, B_bar=float("nan"), mu=float("nan"), log=None, every=500):
    """One record (dict with RECORD_KEYS) per dendrite node of swc (types cfg.acquisition.dendrite_swc_types) in
    the range of calibration set `which` ("near" or "far") by Allen's diameter, in the order of cell.stretches:
    the trace in the image frame (transform: keyword arguments of cell.to_image_um, as scan_cell takes them), the
    cell's scene (cell.cell_scene); B_bar (gl) and mu (/um) the cell's typical background and absorption
    (typical_background). reason is "" for a node that passes, else "no_direction", "not_flat",
    "not_isolated:too_few", "not_isolated:side_short" or "few_planes". log, when given, gets a progress line every
    `every` records."""
    _check_set(which)
    types_ = cfg.acquisition.dendrite_swc_types
    res0 = cfg.acquisition.res0_um
    radius = np.asarray(swc.radius, dtype=float)
    runs = list(cell.stretches(swc, types_))
    member = [[calibration.calibration_set(2.0 * float(radius[r]), cfg) == which for r in run] for run in runs]
    total = sum(sum(m) for m in member)
    if not total:
        return []
    xyz_img = cell.to_image_um(swc.xyz, res0, **(transform or {}))
    scene = cell.cell_scene(swc, types_, res0, transform)
    out, t0 = [], time.perf_counter()
    for stretch, (run, inside) in enumerate(zip(runs, member)):
        if not any(inside):
            continue
        branch, off = cell.stretch_branch(swc, run, xyz_img)
        for t, is_in in enumerate(inside):
            if not is_in:
                continue
            out.append(_judge(branch, off + t, stretch, scene, cfg, B_bar, mu, which))
            if log is not None and len(out) % int(every) == 0:
                log("%s search: %d of %d %s-range nodes judged (%.1f min)" % (which, len(out), total, which,
                                                                             (time.perf_counter() - t0) / 60))
    return out


def choose(records, cfg, which):
    """Node ids of the passing records of set `which` (reason ""), kept greedily -- far: the most usable fit planes
    first, then the most clear pixels at k_trace; near: the most clear pixels at k_trace -- then the smaller node
    id, each at least the set's spacing (um; near_search_spacing_um, far_search_spacing_um) along the path from
    every node already kept on its stretch. Returns the ids in the order kept."""
    _check_set(which)
    sp = float(getattr(cfg.calibration, which + "_search_spacing_um"))
    ok = [r for r in records if r["set"] == which and not r["reason"]]
    if which == "far":
        ok.sort(key=lambda r: (-int(r["n_fit_usable"]), -int(r["n_pixels_k"]), int(r["node_id"])))
    else:
        ok.sort(key=lambda r: (-int(r["n_pixels_k"]), int(r["node_id"])))
    kept, taken = [], {}
    for r in ok:
        on_stretch = taken.setdefault(int(r["stretch"]), [])
        if any(abs(float(r["s_um"]) - s) < sp for s in on_stretch):
            continue
        kept.append(int(r["node_id"]))
        on_stretch.append(float(r["s_um"]))
    return kept


def summary_line(records, chosen, cfg, which):
    """calibrate_kernel.cell_candidates' line on the search of set `which`."""
    _check_set(which)
    cal = cfg.calibration
    flat = [r for r in records if r["reason"] not in ("not_flat", "no_direction")]
    at_k = [r for r in flat if not r["reason"].startswith("not_isolated")]
    enough = [r for r in at_k if r["reason"] != "few_planes"]
    if which == "far":
        bound, planes, sp = (cal.node_phi_max_deg_far, "with at least %d usable far-fit planes"
                             % cal.far_min_planes_per_node, cal.far_search_spacing_um)
    else:
        bound, planes, sp = (cal.node_phi_max_deg, "with at least %d usable planes of the trace's plane +- %d"
                             % (cal.min_planes_per_node, cal.offsets_planes), cal.near_search_spacing_um)
    return ("%s search over the cell: %d %s-range nodes; %d flat by the trace (phi <= %g deg); %d with a usable stripe "
            "at the trace's plane; %d %s; %d chosen, at least %g um apart"
            % (which, len(records), which, len(flat), bound, len(at_k), len(enough), planes, len(chosen), sp))
