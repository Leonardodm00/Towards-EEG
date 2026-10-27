"""The arrays behind the calibration's figures -- Block 10.F in specs/SPEC.md
(D-075, D-080.2, D-081.5; 2026-10-23). This module prepares; the plotting
module plotting/calibration_figures.py only draws.

  growth_curve         G_hat(delta) of a growth fit read back from its JSON
  stage_inputs         one stage's width-by-plane data: per node the signed
                       offsets, omega_hat, the centred omega_hat - c_j, the
                       fitted axis offset delta_hat, which planes the stage's
                       fit used; the median over nodes by offset; the fitted
                       law on a fine delta grid; the residuals; the example
  kernel_law_inputs    both stages as kernel SDs against |delta_hat|: near
                       (omega - c_j + sigma_r(0)^2)^(1/2), far (omega -
                       c_j)^(1/2), the cut table and the far line, the anchor
  law_points           those points with the node of each (Block 19.5)
  pooled_sigma_law     sigma_law(delta): the table, then the far line
  kernel_law_by_cell   the points with their cell and their residual from the
                       pooled law, and the summary per cell and set (19.5)
  gaussian_inputs      per node, per plane: the dip, the last pass's core and
                       window, the status (D-075.2)
  figure_inputs        all of the above for the scans, the calibration file
                       and the far-fit file at hand

Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

from . import calibration
from .far_field import select_far_planes
from ..model.kernel import sigma_r


def growth_curve(gd):
    """G_hat(delta) (um^2) for the growth dict of a calibration file (keys
    convention, interp, knot_step_um, knots_um, g_um2); NaN knots are carried
    as NaN where their basis function is non-zero."""
    knots = np.array([np.nan if x is None else x for x in gd["knots_um"]], dtype=float)
    g = np.array([np.nan if x is None else x for x in gd["g_um2"]], dtype=float)
    conv, interp, h = gd["convention"], gd["interp"], float(gd["knot_step_um"])
    M = (knots.size - 1) if conv == "symmetric" else (knots.size - 1) // 2

    def f(delta):
        B, _ = calibration._basis(np.asarray(delta, dtype=float).ravel(), conv, h, M, interp)
        return (B @ np.where(np.isfinite(g), g, 0.0) + np.where(np.abs(B) @ ~np.isfinite(g) > 1e-12, np.nan, 0.0)
                ).reshape(np.shape(delta))
    return f


def _by_id(ids, values):
    return {int(i): (None if v is None else float(v)) for i, v in zip(ids, values)}


def stage_inputs(scans, which, cfg, calib=None, far_fit=None):
    """The width-by-plane inputs of one stage ("near" or "far"). calib: the
    calibration JSON (the near fit: c_j, z_ax,j, the growth); far_fit: the
    far-fit JSON (c_j, s_j, gamma, the anchor). Returns a dict (None when the
    stage has no scans): offsets (sorted), nodes [dict(node_id, offsets,
    omega, centred, delta_hat, used (bool), status)], median {offset: median
    omega}, median_centred, law (delta_um grid, law_um2 or None), residual per
    node (centred - law(delta_hat), NaN where no law), example_id, has_fit."""
    sel = [sc for sc in scans if sc.set == which]
    if not sel:
        return None
    dz, d0 = cfg.acquisition.dz_um, cfg.calibration.table_max_delta_um
    c_by, z_by, s_by, law, used_by = {}, {}, {}, None, {}
    if which == "near" and calib is not None and "growth" in calib:
        gd = calib["growth"]
        c_by, z_by = _by_id(gd["node_ids"], gd["c_um2"]), _by_id(gd["node_ids"], gd["z_ax_um"])
        curve = growth_curve(gd)
        grid = np.linspace(-(cfg.calibration.offsets_planes + 0.5) * dz, (cfg.calibration.offsets_planes + 0.5) * dz, 301)
        law = (grid, curve(grid))
    elif which == "far" and far_fit is not None and far_fit.get("far_fit", {}).get("gamma") is not None:
        ff = far_fit["far_fit"]
        c_by, s_by = _by_id(ff["node_ids"], ff["c_um2"]), _by_id(ff["node_ids"], ff["s_um"])
        gamma, s0 = float(ff["gamma"]), float(ff["anchor"]["sigma0_um"])
        reach = (cfg.calibration.far_offsets_planes + 0.5) * dz
        grid = np.concatenate([np.linspace(-reach, -d0, 150), [np.nan], np.linspace(d0, reach, 150)])
        with np.errstate(invalid="ignore"):
            law = (grid, (s0 + gamma * (np.abs(grid) - d0)) ** 2)
        try:
            fsel = select_far_planes(sel, cfg)
            for j, nid in enumerate(fsel.node_ids):
                used_by[int(nid)] = set(int(k) for k in fsel.obs_k[fsel.obs_node == j])
        except ValueError:
            used_by = {}
    nodes, acc, acc_c = [], {}, {}
    for sc in sel:
        nid = int(sc.node_id)
        off = sc.offsets()
        om = np.asarray(sc.omega_um2, dtype=float)
        if which == "near":
            z_ax = z_by.get(nid, float(sc.z_sub_um))
            c = c_by.get(nid)
        else:
            z_ax = float(sc.z_sub_um) + float(s_by.get(nid, 0.0) or 0.0)
            c = c_by.get(nid)
        delta_hat = (z_ax - np.asarray(sc.z_um, dtype=float)) if math.isfinite(z_ax) else np.full(off.size, np.nan)
        centred = om - c if c is not None else np.full(off.size, np.nan)
        if which == "near":
            used = np.isfinite(om) & (nid in c_by)
        else:
            used = np.array([(nid in used_by) and (int(k) in used_by.get(nid, ())) for k in sc.ks])
        if law is not None and c is not None:
            if which == "near":
                lv = growth_curve(calib["growth"])(delta_hat)
            else:
                x = np.abs(delta_hat) - d0
                lv = np.where(x >= 0, (float(far_fit["far_fit"]["anchor"]["sigma0_um"]) + float(far_fit["far_fit"]["gamma"]) * x) ** 2,
                              np.nan)
            resid = centred - lv
        else:
            resid = np.full(off.size, np.nan)
        nodes.append(dict(node_id=nid, offsets=off, omega=om, centred=centred, delta_hat=delta_hat, used=used,
                          residual=resid, status=tuple(sc.status)))
        for o, w, wc in zip(off, om, centred):
            if np.isfinite(w):
                acc.setdefault(int(o), []).append(float(w))
            if np.isfinite(wc):
                acc_c.setdefault(int(o), []).append(float(wc))
    return dict(which=which, offsets=sorted(acc), nodes=nodes,
                median={o: float(np.median(v)) for o, v in sorted(acc.items())},
                median_centred={o: float(np.median(v)) for o, v in sorted(acc_c.items())},
                law=law, example_id=calibration.example_node(sel, which), has_fit=bool(c_by), dz_um=dz)


def law_points(scans, cfg, calib=None, far_fit=None):
    """Block 19.5: kernel_law_inputs' points with the node of each --
    (abs_delta, sigma, side, used, which, node_id) per plane with a finite
    width and a known c_j, near then far, nodes in the stage's order; []
    without a near fit."""
    if calib is None or "growth" not in calib:
        return []
    s0_cfg = float(cfg.renderer.sigma_r0_um)
    pts = []
    near = stage_inputs(scans, "near", cfg, calib, far_fit)
    far = stage_inputs(scans, "far", cfg, calib, far_fit)
    for st, which in ((near, "near"), (far, "far")):
        if st is None or not st["has_fit"]:
            continue
        for nd in st["nodes"]:
            with np.errstate(invalid="ignore"):
                val = nd["centred"] + (s0_cfg ** 2 if which == "near" else 0.0)
                sig = np.where(val > 0, np.sqrt(np.maximum(val, 0.0)), np.nan)
            for dh, s, u in zip(nd["delta_hat"], sig, nd["used"]):
                if np.isfinite(s) and np.isfinite(dh):
                    pts.append((abs(float(dh)), float(s), 1 if dh > 0 else -1, bool(u), which, int(nd["node_id"])))
    return pts


def _law_curves(cfg, calib, far_fit):
    """(table, line, anchor) of the figures: the calibration file's table (delta, sigma), the far line on a grid
    from the anchor to the far scan's reach, the anchor (delta_0, sigma_0); None where absent."""
    table = None
    if "kernel" in calib:
        table = (np.array(calib["kernel"]["kernel_table_delta_um"], dtype=float),
                 np.array(calib["kernel"]["kernel_table_sigma_um"], dtype=float))
    line, anchor = None, None
    if calib.get("anchor", {}).get("sigma0_um") is not None:
        anchor = (float(calib["anchor"]["delta0_um"]), float(calib["anchor"]["sigma0_um"]))
    if far_fit is not None and far_fit.get("far_fit", {}).get("gamma") is not None and anchor is not None:
        reach = (cfg.calibration.far_offsets_planes + 0.5) * cfg.acquisition.dz_um
        grid = np.linspace(anchor[0], reach, 100)
        line = (grid, anchor[1] + float(far_fit["far_fit"]["gamma"]) * (grid - anchor[0]))
    return table, line, anchor


def kernel_law_inputs(scans, cfg, calib=None, far_fit=None):
    """Both stages as kernel SDs (um) against |delta_hat| (D-075.1): points
    (abs_delta, sigma, side, used, which) per plane with a finite width and a
    known c_j; the table of the calibration file (delta, sigma); the far line
    on a grid; the anchor (delta_0, sigma_0). None without a calibration."""
    if calib is None or "growth" not in calib:
        return None
    s0_cfg = float(cfg.renderer.sigma_r0_um)
    pts = [p[:5] for p in law_points(scans, cfg, calib, far_fit)]
    table, line, anchor = _law_curves(cfg, calib, far_fit)
    return dict(points=pts, table=table, line=line, anchor=anchor, sigma_r0_um=s0_cfg)


BY_CELL_COLUMNS = ("specimen_id", "set", "n_nodes", "n_points_used", "n_points", "median_residual_um",
                   "q25_residual_um", "q75_residual_um", "median_abs_delta_um")


def pooled_sigma_law(calib, far_fit=None):
    """Block 19.5: sigma_law(delta) (um) for delta >= 0 (um): the calibration
    file's table, linear between its points, up to its last delta; beyond it
    the far line sigma_0 + gamma (delta - delta_0) with the calibration's
    anchor and the far fit's gamma, NaN without a far fit (or an anchor);
    NaN below 0. None without a table."""
    if calib is None or "kernel" not in calib:
        return None
    td = np.asarray(calib["kernel"]["kernel_table_delta_um"], dtype=float)
    ts = np.asarray(calib["kernel"]["kernel_table_sigma_um"], dtype=float)
    gamma = None
    if far_fit is not None and (far_fit.get("far_fit") or {}).get("gamma") is not None:
        gamma = float(far_fit["far_fit"]["gamma"])
    anchor = calib.get("anchor") or {}
    d0, s0 = anchor.get("delta0_um"), anchor.get("sigma0_um")

    def law(delta):
        d = np.asarray(delta, dtype=float)
        out = np.full(d.shape, np.nan)
        inside = (d >= 0) & (d <= td[-1])
        out[inside] = np.interp(d[inside], td, ts)
        if gamma is not None and d0 is not None and s0 is not None:
            beyond = d > td[-1]
            out[beyond] = float(s0) + gamma * (d[beyond] - float(d0))
        return out
    return law


def kernel_law_by_cell(scans, cfg, calib=None, far_fit=None, id_map=None):
    """Block 19.5 (D-090.2): the kernel-law points with their cell
    (calibration_pool.cell_of) and their residual from the pooled law,
    e = sigma - sigma_law(|delta|) (um); the cells in increasing specimen id
    (a cell of None last); per cell and set the summary row of BY_CELL_COLUMNS
    over the used points of finite residual; the table, the far line and the
    anchor. None without a near fit."""
    if calib is None or "growth" not in calib:
        return None
    from .calibration_pool import cell_of
    raw = law_points(scans, cfg, calib, far_fit)
    law = pooled_sigma_law(calib, far_fit)
    d_all = np.array([p[0] for p in raw], dtype=float)
    lv = law(d_all) if law is not None else np.full(d_all.shape, np.nan)
    pts = []
    for (d, s, side, used, which, nid), v in zip(raw, lv):
        cell, node = cell_of(nid, id_map)
        pts.append(dict(cell=cell, node=node, node_id=int(nid), set=which, abs_delta_um=float(d), sigma_um=float(s),
                        side=int(side), used=bool(used), residual_um=float(s - v)))
    cells = sorted({p["cell"] for p in pts}, key=lambda c: (c is None, 0 if c is None else c))
    summary = []
    for c in cells:
        for which in ("near", "far"):
            sel = [p for p in pts if p["cell"] == c and p["set"] == which]
            if not sel:
                continue
            ok = [p for p in sel if p["used"] and math.isfinite(p["residual_um"])]
            e = np.array([p["residual_um"] for p in ok], dtype=float)
            ad = np.array([p["abs_delta_um"] for p in ok], dtype=float)
            q = (lambda x, k: float(np.percentile(x, k)) if x.size else float("nan"))
            summary.append(dict(specimen_id=c, set=which, n_nodes=len({p["node_id"] for p in ok}), n_points_used=len(ok),
                                n_points=len(sel), median_residual_um=q(e, 50), q25_residual_um=q(e, 25),
                                q75_residual_um=q(e, 75), median_abs_delta_um=q(ad, 50)))
    table, line, anchor = _law_curves(cfg, calib, far_fit)
    return dict(points=pts, cells=cells, summary=summary, table=table, line=line, anchor=anchor)


def gaussian_inputs(scan):
    """Per plane of one scan (D-075.2): v, the dip w(v), the last pass's core
    (amp, v_c, sigma) and window, omega_hat, the status, the offset. None when
    the scan carries no profiles (a file from before 2026-10-23)."""
    if scan.w_gl is None or scan.v_um is None:
        return None
    planes = []
    for q, k in enumerate(scan.ks):
        om = float(scan.omega_um2[q])
        planes.append(dict(offset=int(k - scan.k_star), k=int(k), w=np.asarray(scan.w_gl[q], dtype=float),
                           amp=float(scan.amp_gl[q]) if scan.amp_gl is not None else float("nan"),
                           v_c=float(scan.v_c_um[q]) if scan.v_c_um is not None else float("nan"),
                           sigma=math.sqrt(om) if math.isfinite(om) and om > 0 else float("nan"),
                           window_half=float(scan.window_half_um[q]) if scan.window_half_um is not None else float("nan"),
                           omega=om, status=scan.status[q],
                           passes=int(scan.passes[q]) if scan.passes is not None else 0))
    return dict(node_id=int(scan.node_id), set=scan.set, v=np.asarray(scan.v_um, dtype=float), planes=planes,
                k_star=int(scan.k_star))


def example_inputs(scans, which, patches, cfg, calib=None, far_fit=None):
    """The example node's inputs (10.F (2)): its stage_inputs entry, its own
    curve (c_j + law) on the plane grid, and its plane images from the patches
    dict of calibration_io.load_patches (None without them)."""
    st = stage_inputs(scans, which, cfg, calib, far_fit)
    if st is None or st["example_id"] is None:
        return None
    nd = next(n for n in st["nodes"] if n["node_id"] == st["example_id"])
    sc = next(s for s in scans if s.set == which and int(s.node_id) == st["example_id"])
    curve = None
    if st["law"] is not None and np.isfinite(nd["centred"]).any():
        c = float(np.nanmedian(nd["omega"] - nd["centred"]))
        curve = (st["law"][0], c + st["law"][1])
    images = None
    rec = (patches or {}).get(which)
    if rec is not None and int(rec["node_id"]) == st["example_id"]:
        images = rec
    return dict(node=nd, curve=curve, images=images, which=which, dz_um=cfg.acquisition.dz_um,
                gaussians=gaussian_inputs(sc), z_sub_um=float(sc.z_sub_um))


def figure_inputs(scans, cfg, calib=None, far_fit=None, patches=None):
    """Everything the figures need: stages {which: stage_inputs}, kernel_law,
    examples {which: example_inputs}, gaussians [gaussian_inputs per scan]."""
    stages = {w: stage_inputs(scans, w, cfg, calib, far_fit) for w in calibration.CALIBRATION_SETS}
    return dict(stages={w: s for w, s in stages.items() if s is not None},
                kernel_law=kernel_law_inputs(scans, cfg, calib, far_fit),
                examples={w: example_inputs(scans, w, patches, cfg, calib, far_fit) for w in stages if stages[w]},
                gaussians=[g for g in (gaussian_inputs(sc) for sc in scans) if g is not None])
