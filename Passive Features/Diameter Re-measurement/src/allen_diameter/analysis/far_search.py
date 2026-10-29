"""Block 10.C in specs/SPEC.md -- the far set's candidates found over the cell
(D-109, 2026-10-29; the user: "Just let the colab code for the pilot cell find
the nodes for the far field calibration").

Before any image is read, every dendrite node of the cell whose Allen
diameter puts it in the far set (calibration.calibration_set) is judged from
the trace alone:

  (i)  the trace's direction at the node -- the oriented TLS direction of the
       branch points within +- line_fit_window_um / 2 along the path, Block
       5's raw direction -- its heading theta and tilt phi; not_flat when
       phi > node_phi_max_deg_far;
  (ii) the stripes of Block 10.S on the patch frame about the trace's node,
       with the trace's heading, the trace's plane k_tr = round(z / dz) as
       the sharpest plane, Allen's radius for the own cylinder and the cell's
       typical background and absorption (typical_background: the medians of
       the pilot table's rows in S) -- first at k_tr (not_isolated:too_few /
       not_isolated:side_short, D-101.4's verdict at the trace's plane), then
       at the far fit's planes k_tr + o, far_min_k_offset <= |o| <=
       far_offsets_planes (few_planes when fewer than far_min_planes_per_node
       have a usable stripe).

choose() then keeps the passing nodes greedily (most usable fit planes first,
then the most pixels at k_tr, then the node id), each at least
far_search_spacing_um along the path from those already kept on its stretch.
The chosen nodes go to calibration.scan_cell like any candidate: their pilot
is measured and they are judged again on it. Analysis only: no file is read
or written here, nothing is drawn.

  trace_direction(branch, i, window_um) -> (theta rad, phi rad)
  typical_background(rows, cfg)         -> (B_bar gl, mu_hat /um), NaN without rows in S
  search(swc, cfg, transform, B_bar, mu, log) -> list of records (dict), one per far-range node
  choose(records, cfg)                  -> list of node ids, in the order kept
  summary_line(records, chosen, cfg)    -> the log line of calibrate_kernel.py real

Pure ASCII (hpc-python-compat).
"""
from __future__ import annotations

import math
import types as _types

import numpy as np

from . import calibration, calibration_scene, cell, path
from ..model.kernel import sigma_r

RECORD_KEYS = ("node_id", "allen_diameter_um", "stretch", "s_um", "x_um", "y_um", "z_um", "k_trace", "theta_rad",
               "phi_deg", "reason", "n_pixels_k", "n_fit_usable", "n_fit")


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


def _judge(branch, i, stretch, scene, cfg, B_bar, mu):
    """The record of far-range node i of branch (search's rule (i)-(ii))."""
    cal, dz, res0 = cfg.calibration, cfg.acquisition.dz_um, cfg.acquisition.res0_um
    x, y, z = (float(v) for v in np.asarray(branch.xyz_um, dtype=float)[i])
    k = int(round(z / dz))
    rec = dict(node_id=int(branch.ids[i]), allen_diameter_um=2.0 * float(branch.radius_um[i]), stretch=int(stretch),
               s_um=float(branch.s_um[i]), x_um=x, y_um=y, z_um=z, k_trace=k, theta_rad=float("nan"),
               phi_deg=float("nan"), reason="", n_pixels_k=0, n_fit_usable=0, n_fit=0)
    try:
        theta, phi = trace_direction(branch, i, cfg.measure.line_fit_window_um)
    except ValueError:
        rec["reason"] = "no_direction"
        return rec
    rec.update(theta_rad=theta, phi_deg=math.degrees(phi))
    if math.degrees(phi) > cal.node_phi_max_deg_far:
        rec["reason"] = "not_flat"
        return rec
    view = _types.SimpleNamespace(cx_um=x, cy_um=y, theta_rad=theta, z_sub_um=z, k_star=k, B_bar=float(B_bar),
                                  mu_hat_per_um=float(mu), d_hat_um=float("nan"))
    shape, frame = calibration_scene.patch_frame(x, y, cal.patch_half_um, res0)
    sigma_soma = float(sigma_r(cal.far_offsets_planes * dz, cfg.renderer))       # the far stage's largest blur (F1)
    _sup, st = calibration.support_masks(shape, frame, np.array([k]), view, branch, scene, cfg,
                                         sigma_soma_um=sigma_soma)
    rec["n_pixels_k"] = int(st[0]["n_pixels"])
    if not st[0]["ok"]:
        rec["reason"] = "not_isolated:" + st[0]["reason"]
        return rec
    offsets = [o for o in range(-int(cal.far_offsets_planes), int(cal.far_offsets_planes) + 1)
               if abs(o) >= int(cal.far_min_k_offset)]
    _sup, st = calibration.support_masks(shape, frame, np.array([k + o for o in offsets]), view, branch, scene, cfg,
                                         sigma_soma_um=sigma_soma)
    rec["n_fit"] = len(offsets)
    rec["n_fit_usable"] = int(sum(1 for s in st if s["ok"]))
    if rec["n_fit_usable"] < int(cal.far_min_planes_per_node):
        rec["reason"] = "few_planes"
    return rec


def search(swc, cfg, transform=None, B_bar=float("nan"), mu=float("nan"), log=None):
    """One record (dict with RECORD_KEYS) per dendrite node of swc (types cfg.acquisition.dendrite_swc_types) in
    the far set by Allen's diameter, in the order of cell.stretches: the trace in the image frame (transform:
    keyword arguments of cell.to_image_um, as scan_cell takes them), the cell's scene (cell.cell_scene); B_bar
    (gl) and mu (/um) the cell's typical background and absorption (typical_background). reason is "" for a node
    that passes, else "no_direction", "not_flat", "not_isolated:too_few", "not_isolated:side_short" or
    "few_planes". log, when given, gets a progress line every 100 records."""
    types_ = cfg.acquisition.dendrite_swc_types
    res0 = cfg.acquisition.res0_um
    radius = np.asarray(swc.radius, dtype=float)
    xyz_img = cell.to_image_um(swc.xyz, res0, **(transform or {}))
    scene = None
    out = []
    for stretch, run in enumerate(cell.stretches(swc, types_)):
        far = [calibration.calibration_set(2.0 * float(radius[r]), cfg) == "far" for r in run]
        if not any(far):
            continue
        if scene is None:
            scene = cell.cell_scene(swc, types_, res0, transform)
        branch, off = cell.stretch_branch(swc, run, xyz_img)
        for t, is_far in enumerate(far):
            if not is_far:
                continue
            out.append(_judge(branch, off + t, stretch, scene, cfg, B_bar, mu))
            if log is not None and len(out) % 100 == 0:
                log("far search: %d far-range nodes judged" % len(out))
    return out


def choose(records, cfg):
    """Node ids of the passing records (reason ""), kept greedily -- the most usable far-fit planes first, then
    the most pixels at k_trace, then the smaller node id -- each at least far_search_spacing_um (um) along the
    path from every node already kept on its stretch. Returns the ids in the order kept."""
    sp = float(cfg.calibration.far_search_spacing_um)
    ok = sorted((r for r in records if not r["reason"]),
                key=lambda r: (-int(r["n_fit_usable"]), -int(r["n_pixels_k"]), int(r["node_id"])))
    kept, taken = [], {}
    for r in ok:
        on_stretch = taken.setdefault(int(r["stretch"]), [])
        if any(abs(float(r["s_um"]) - s) < sp for s in on_stretch):
            continue
        kept.append(int(r["node_id"]))
        on_stretch.append(float(r["s_um"]))
    return kept


def summary_line(records, chosen, cfg):
    """calibrate_kernel.py real's line on the search."""
    cal = cfg.calibration
    flat = [r for r in records if r["reason"] not in ("not_flat", "no_direction")]
    at_k = [r for r in flat if not r["reason"].startswith("not_isolated")]
    enough = [r for r in at_k if r["reason"] != "few_planes"]
    return ("far search over the cell: %d far-range nodes; %d flat by the trace (phi <= %g deg); %d with a usable "
            "stripe at the trace's plane; %d with at least %d usable far-fit planes; %d chosen, at least %g um apart"
            % (len(records), len(flat), cal.node_phi_max_deg_far, len(at_k), len(enough), cal.far_min_planes_per_node,
               len(chosen), cal.far_search_spacing_um))
