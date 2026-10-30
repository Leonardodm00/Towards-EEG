#!/usr/bin/env python3
"""S3 across cells -- the resumable per-cell runner and its steps (Block 16.2
(g) in specs/SPEC.md and the blocks that add steps; D-085, D-085.12). Colab:
api.brain-map.org is blocked from the sandbox.

  run     for each cell of selection.csv (Block 16.1) -- or the one cell of
          row --task-index (a PBS array task, D-099) -- and each named step, in
          order: skip a step already done (its marker in <out>/<sid>/done/),
          unless --force names it (which also clears the markers of the
          steps after it); skip every step after `align` of a cell the
          alignment set aside, unless --include-set-aside; a step that raises
          is recorded as failed (errors.log has the traceback), the cell's
          later steps are skipped, and the next cell starts
  status  the table of every cell's steps, rebuilt from the markers

  pool    the pooled analyses over the cells: --what coverage (Block 17.3-17.5),
          --what inputs (Block 18.4-18.5), --what calibration (Block 19.4)

Steps (in their order)
  align   the cell's global alignment (Block 16.2): the minimum-intensity
          projection at level 4, align_translation with both flips, the
          null's score ratio, the fine stage at level 1 on the cell's box;
          set aside below the ratio threshold or on a search edge.
          <sid>/align/alignment_<sid>.json and overlay_<sid>.png; a hand-made
          <sid>/align/alignment_manual.json is used as given
  nodes   S3a's node sample and Block 13's measurement of it (Block 17.1-17.2):
          survey.nodes_mode, stratified (20 per bin of Allen's 2r, round-robin
          over the stretches) or all; <sid>/nodes/pilot_<sid>.csv (the per-node
          CSV with the background statistics) and pilot_summary_<sid>.json
  radius        Allen's dendrite diameters of the cell's SWC (Block 18.1): <sid>/radius/
  registration  the registration survey on list_stretches' suggestion (Block 18.2):
                <sid>/registration/ (s*, dz* and the verdict per node)
  camera        the camera inputs from the nodes' background (Block 18.3; after
                nodes): <sid>/camera/camera_<sid>.json
  calibration   the kernel calibration's scans of the cell (Block 19.3; after
                nodes): its candidates found over the whole cell from the
                trace (Block 10.C, D-112; the per-node CSV gives the typical
                background and absorption, and the candidates of a set whose
                calibration.<set>_candidates is "table"), a seeded order, at
                most survey.calib_scans_per_cell per set (D-088):
                <sid>/calibration/scans_<sid>.json, <set>_search_<sid>.csv

Example (Colab, after Cell 11; on davinci scripts/pbs/survey_cells.pbs runs one cell per array task with
--task-index and --cache-per-cell, D-099)
    python scripts/survey_cells.py run --selection .../survey/selection.csv --out-dir .../survey \
        --cache-dir /content/drive/MyDrive/allen_cache --steps align
    python scripts/survey_cells.py status --out-dir .../survey
    python scripts/survey_cells.py run ... --steps align,nodes
    python scripts/survey_cells.py pool --out-dir .../survey --what coverage

Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
for _p in (SRC, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

import allen_image_io as aio  # noqa: E402
import allen_image_plot as aip  # noqa: E402
from allen_diameter.analysis import cell_alignment  # noqa: E402
from allen_diameter.loading import survey_io  # noqa: E402
from build_table import load_config  # noqa: E402


class AllenSource:
    """Allen's server through allen_image_io, the crops cached in cache_dir."""

    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        self.fetcher = aio.HttpFetcher(cache_dir=cache_dir)

    def swc_path(self, sid):
        return aio.fetch_swc(int(sid), self.cache_dir)

    def images(self, sid):
        return aio.list_images(int(sid))

    def subimage_rows(self, sid):
        """Every SubImage row with every field (Block 16.3)."""
        return aio.subimage_rows(int(sid))

    def planes(self, sid):
        return aio.plane_table(self.images(sid))


class Context:
    """One cell for the steps: its selection record, its folder, the source,
    and its context file (cell.json: the SWC path, the alignment)."""

    def __init__(self, record, cell, source, reuse_search=None):
        self.record, self.cell, self.source = record, cell, source
        self.sid = cell.sid
        self.reuse_search = reuse_search or None   # Block 10.C.R (D-115.6): a folder of earlier searches to reuse
        self.data = cell.context()
        self.data.setdefault("selection", {k: record[k] for k in record})

    @property
    def alignment(self):
        return self.data.get("alignment")

    def set_aside(self):
        return bool((self.alignment or {}).get("set_aside"))

    def save(self):
        self.cell.save_context(self.data)


# ------------------------------------------------------------------ steps ---

def step_align(ctx, cfg):
    """Block 16.2: the cell's global alignment (D-085.2)."""
    sv, rho0, sid = cfg.survey, float(cfg.acquisition.res0_um), ctx.sid
    swc_path = ctx.source.swc_path(sid)
    ctx.data["swc_path"] = swc_path
    swc = aip.read_swc(swc_path)
    manual = ctx.cell.path_of("align", "alignment_manual.json")
    if os.path.exists(manual):
        m = survey_io.read_json(manual)
        rec = dict(method="manual", shift_x_px=float(m["shift_x_px"]), shift_y_px=float(m["shift_y_px"]),
                   flip_h_px=None if m.get("flip_h_px") is None else float(m["flip_h_px"]),
                   z0_um=float(m.get("z0_um", sv.align_z0_um)), set_aside=False, reasons=[], source=manual)
    else:
        images = ctx.source.images(sid)
        cands = cell_alignment.xy_projections(images, sv.projection_size_rel_tol)
        if cands.empty:
            raise RuntimeError("no xy projection among the SubImages of %d (types: %s)"
                               % (sid, sorted(set(str(t) for t in images["image_type"]))))
        arrays = [aio.fetch_whole(ctx.source.fetcher, row, sv.align_downsample)[0] for _, row in cands.iterrows()]
        k, rule, meds = cell_alignment.choose_projection(list(cands["image_type"]), arrays, sv.projection_xy_type)
        row = cands.iloc[k]

        def get_crop(left, top, width, height, q):
            return aio.fetch_crop(ctx.source.fetcher, int(row["id"]), left, top, width, height, q, rho0)[0]

        rng = np.random.default_rng([int(sv.selection_seed), int(sid)])
        rec, imgs = cell_alignment.align_record(swc, arrays[k], sv.align_downsample,
                                                (int(row["width"]), int(row["height"])), get_crop, rho0, sv, rng)
        rec["projection"] = dict(id=int(row["id"]), image_type=str(row["image_type"]), rule=rule, candidates=[
            dict(id=int(r["id"]), image_type=str(r["image_type"]), width=int(r["width"]), height=int(r["height"]),
                 median_gl=meds[i]) for i, (_, r) in enumerate(cands.iterrows())])
        from allen_diameter.plotting import survey_figures
        import matplotlib.pyplot as plt
        fig = survey_figures.alignment_overlay(swc, imgs, rec, title="specimen %d (%s)" % (sid, ctx.record.get("group")))
        fig.savefig(ctx.cell.path_of("align", "overlay_%d.png" % sid), dpi=110)
        plt.close(fig)
    rec["specimen_id"] = sid
    survey_io.write_json(ctx.cell.path_of("align", "alignment_%d.json" % sid), rec)
    ctx.data["alignment"] = survey_io.clean(rec)
    ctx.save()
    return dict(method=rec["method"], shift_x_px=rec["shift_x_px"], shift_y_px=rec["shift_y_px"],
                flip_h_px=rec["flip_h_px"], score_ratio=rec.get("score_ratio"), set_aside=rec["set_aside"],
                reasons=rec["reasons"])


def step_nodes(ctx, cfg):
    """Block 17.1-17.2: the node sample and its measurement (D-065.1, D-085.4)."""
    import run_cell
    from allen_diameter.analysis import s3_nodes, survey
    from allen_diameter.loading import swc_io, table_io
    sid = ctx.sid
    swc = swc_io.read_swc(ctx.data["swc_path"])
    transform = cell_alignment.full_transform(ctx.alignment)
    ids, counts = s3_nodes.sample_nodes(swc, cfg, sid)
    if not ids:
        raise RuntimeError("no dendrite node sampled in %d (mode %s)" % (sid, cfg.survey.nodes_mode))
    planes = ctx.source.planes(sid)
    provider = run_cell.real_provider(ctx.source.fetcher, planes, cfg.acquisition.res0_um)
    meas = survey.measure_nodes(swc, provider, cfg, transform, None, ids, None, whole_stretches=False)
    rows = [survey.pilot_row(res, r_allen, cfg) for res, r_allen, _, _ in meas]
    stats = [survey.background_stats(provider, res, br, cfg) for res, _, br, _ in meas]
    for row, st in zip(rows, stats):
        row.update(bg_median_gl=st[0], bg_robust_sd_gl=st[1], bg_clipped_sd_gl=st[2], bg_unmasked_frac=st[3])
    summary = survey.summarize(rows, cfg)
    summary["background"] = survey.background_summary(stats)
    summary.update(specimen=str(sid), mode=cfg.survey.nodes_mode, n_sampled=len(ids), transform=transform,
                   sample_counts={str(b): list(v) for b, v in counts.items()},
                   estimator_hash=cfg.signature_hash("estimator"))
    out_csv = ctx.cell.path_of("nodes", "pilot_%d.csv" % sid)
    table_io.write_rows(rows, out_csv)
    survey_io.write_json(ctx.cell.path_of("nodes", "pilot_summary_%d.json" % sid), summary)
    return dict(mode=cfg.survey.nodes_mode, n_sampled=len(ids), n_measured=len(rows),
                n_in_S=sum(1 for r in rows if r["in_S"]), csv=out_csv)


def _say_nodes(s):
    return "%s: %d sampled, %d measured, %d in S" % (s["mode"], s["n_sampled"], s["n_measured"], s["n_in_S"])


def step_radius(ctx, cfg):
    """Block 18.1: Allen's dendrite diameters of the cell's SWC (D-065.2, D-086)."""
    import allen_radius_distribution as ard
    sid = ctx.sid
    nodes, files = ard.collect([ctx.data["swc_path"]], tuple(cfg.acquisition.dendrite_swc_types))
    for r in nodes + files:
        r["specimen"] = str(sid)
    ard.write_csv(ctx.cell.path_of("radius", "radius_nodes_%d.csv" % sid), nodes)
    ard.write_csv(ctx.cell.path_of("radius", "radius_summary_%d.csv" % sid), files)
    D = np.array([r["diameter_um"] for r in nodes], dtype=float)
    ok = np.isfinite(D)
    if not ok.any():
        raise RuntimeError("no dendrite node with a finite radius in %d" % sid)
    k = int(np.argmax(np.where(ok, D, -np.inf)))
    return dict(n=int(ok.sum()), max_um=float(D[k]), max_node_id=int(nodes[k]["node_id"]),
                p99_um=float(np.percentile(D[ok], 99.0)))


def step_registration(ctx, cfg):
    """Block 18.2: the registration survey on list_stretches' suggestion (D-085.8)."""
    import list_stretches
    import registration_survey
    from allen_diameter.loading import swc_io
    sid, sv = ctx.sid, cfg.survey
    out_dir = os.path.dirname(ctx.cell.path_of("registration", "x"))
    swc = swc_io.read_swc(ctx.data["swc_path"])
    _rows, info = list_stretches.run(swc, out_dir, sid, types=tuple(cfg.acquisition.dendrite_swc_types),
                                     per_type=sv.registration_per_type, min_nodes=sv.registration_min_nodes,
                                     log=lambda m: None)
    nodes = [int(n) for n in info["suggested_nodes"]]
    if not nodes:
        raise RuntimeError("no stretch of at least %d nodes to register in %d" % (sv.registration_min_nodes, sid))
    t = cell_alignment.full_transform(ctx.alignment)
    _res, summ = registration_survey.run(aip.read_swc(ctx.data["swc_path"]), ctx.source.fetcher, ctx.source.planes(sid),
                                         nodes, out_dir, sid, cfg.acquisition.res0_um, cfg.acquisition.dz_um,
                                         shift_full_px=t["shift_full_px"], flip_y_full_h=t["flip_y_full_h"],
                                         z0_um=t["z0_um"], log=lambda m: None)
    return dict(n_nodes=len(nodes), verdicts=summ.get("verdicts", {}), jitter=summ.get("suggested_jitter_um"))


def step_camera(ctx, cfg):
    """Block 18.3: the camera inputs from the nodes' background statistics (D-085.9)."""
    import camera_calibration
    sid = ctx.sid
    if not ctx.cell.done("nodes"):
        raise RuntimeError("camera needs the step nodes of %d (its background statistics)" % sid)
    out_dir = os.path.dirname(ctx.cell.path_of("camera", "x"))
    out = camera_calibration.run(ctx.source.cache_dir, out_dir, sid,
                                 pilot_summary=ctx.cell.path_of("nodes", "pilot_summary_%d.json" % sid), cfg=cfg)
    r = out.get("renderer", {})
    return dict(background_B_gl=r.get("background_B_gl"), noise_sd_gl=r.get("noise_sd_gl"),
                has_tables="jpeg_qtables" in r, notes=out.get("notes", []))


def step_calibration(ctx, cfg):
    """Block 19.3: the cell's calibration scans for the pooled kernel (D-088). The candidates are Block 10.C's
    (calibrate_kernel.cell_candidates, D-109, D-112): every node of the cell in either set judged from the trace,
    the cell's per-node CSV giving the typical background and absorption -- or that CSV's rows for a set whose
    calibration.<set>_candidates is "table"; the search's records go to <sid>/calibration/<set>_search_<sid>.csv."""
    import dataclasses
    import calibrate_kernel
    import run_cell
    from allen_diameter.analysis import calibration
    from allen_diameter.loading import calibration_io, swc_io, table_io
    sid = ctx.sid
    if not ctx.cell.done("nodes"):
        raise RuntimeError("calibration needs the step nodes of %d (its per-node CSV)" % sid)
    rows = table_io.read_rows(ctx.cell.path_of("nodes", "pilot_%d.csv" % sid))
    path = ctx.cell.path_of("calibration", "scans_%d.json" % sid)
    swc = transform = None
    if "cell" in calibrate_kernel.candidate_sources(cfg).values():
        swc = swc_io.read_swc(ctx.data["swc_path"])
        transform = cell_alignment.full_transform(ctx.alignment)
    cand = calibrate_kernel.cell_candidates(swc, cfg, transform, rows, os.path.dirname(path), sid,
                                            log=lambda m: print("    %d %s" % (sid, m), flush=True),
                                            reuse_dir=ctx.reuse_search)       # D-115.6: Cell 15 reuses Cell 7's
    meta = dict(source="survey", specimen=str(sid), n_candidates=len(cand), calibration=dataclasses.asdict(cfg.calibration),
                max_per_set=int(cfg.survey.calib_scans_per_cell))
    if not cand:
        calibration_io.save_scans([], path, dict(meta, nodes=[], reasons_count={}))
        return dict(n_candidates=0, n_near=0, n_far=0, reasons={})
    if swc is None:
        swc = swc_io.read_swc(ctx.data["swc_path"])
        transform = cell_alignment.full_transform(ctx.alignment)
    provider = run_cell.real_provider(ctx.source.fetcher, ctx.source.planes(sid), cfg.acquisition.res0_um)
    out = calibration.scan_cell(swc, provider, cfg, transform, None, cand,
                                order_rng=np.random.default_rng([int(cfg.calibration.calib_order_seed), int(sid)]),
                                max_per_set=int(cfg.survey.calib_scans_per_cell))
    scans = [sc for _, _, sc, _ in out if sc is not None]
    reasons = {}
    for _, _, _, rs in out:
        for r in rs:
            reasons[r] = reasons.get(r, 0) + 1
    nodes = calibration.node_records(out, swc, cfg)   # with each measured node's pilot values (D-115.5)
    calibration_io.save_scans(scans, path, dict(meta, nodes=nodes, reasons_count=reasons))
    return dict(n_candidates=len(cand), n_near=sum(1 for x in scans if x.set == "near"),
                n_far=sum(1 for x in scans if x.set == "far"), reasons=reasons)


def _say_calibration(s):
    return "%d candidates: %d near-field, %d far-field scans; reasons %s" % (s["n_candidates"], s["n_near"], s["n_far"],
                                                                          s["reasons"])


def _say_radius(s):
    return "%d nodes, max 2r %.2f um (node %d), p99 %.2f um" % (s["n"], s["max_um"], s["max_node_id"], s["p99_um"])


def _say_registration(s):
    j = s.get("jitter") or {}
    return "%d nodes, verdicts %s, robust SD xy %s, z %s" % (s["n_nodes"], s["verdicts"], j.get("jitter_xy_um"),
                                                              j.get("jitter_z_um"))


def _say_camera(s):
    return "B %s gl, noise SD %s gl, tables %s" % (s["background_B_gl"], s["noise_sd_gl"], "yes" if s["has_tables"] else "no")


def _say_align(s):
    return "shift (%.1f, %.1f) px, %s, ratio %s, %s%s" % (
        s["shift_x_px"], s["shift_y_px"], "flip" if s["flip_h_px"] is not None else "no flip",
        "%.2f" % s["score_ratio"] if s.get("score_ratio") is not None else "n/a", s["method"],
        "; SET ASIDE (%s)" % ",".join(s["reasons"]) if s["set_aside"] else "")


STEPS = {"align": step_align, "nodes": step_nodes, "radius": step_radius, "registration": step_registration,
         "camera": step_camera, "calibration": step_calibration}
ORDER = ("align", "nodes", "radius", "registration", "camera", "calibration")
SAY = {"align": _say_align, "nodes": _say_nodes, "radius": _say_radius, "registration": _say_registration,
       "camera": _say_camera, "calibration": _say_calibration}


# ----------------------------------------------------------------- runner ---

def _status_row(cell, record, steps):
    row = dict(specimen_id=cell.sid, group=record.get("group", ""))
    for s in steps:
        row[s] = "done" if cell.done(s) else "pending"
    return row


def run(cfg, selection_path, out_dir, source, steps, cells=None, force=(), include_set_aside=False, log=print,
        task_index=None, source_for=None, reuse_search=None):
    """Block 16.2 (g). Returns the status rows. task_index (D-099.1): the one
    cell of that 0-based row of the selection, a PBS array task's; exclusive
    with cells; beyond the selection: ValueError. source_for (D-099.2): a
    callable sid -> source used in place of `source` for each cell (a crop
    cache per cell). reuse_search (Block 10.C.R, D-115.6): a folder whose
    candidate searches the calibration step reuses when they were made from
    the cell's trace under the same settings (Cell 15: Cell 7's folder)."""
    unknown = [s for s in list(steps) + list(force) if s not in STEPS]
    if unknown:
        raise ValueError("unknown steps %s (known: %s)" % (unknown, list(ORDER)))
    steps = [s for s in ORDER if s in steps]
    sel = survey_io.read_selection(selection_path)
    if task_index is not None:
        if cells:
            raise ValueError("run: --task-index and --cells are exclusive")
        i = int(task_index)
        if i < 0 or i >= len(sel):
            raise ValueError("run: task index %d outside the selection of %d cells (rows 0-%d)"
                             % (i, len(sel), len(sel) - 1))
        sel = [sel[i]]
    if cells:
        sel = [r for r in sel if int(r["specimen_id"]) in set(int(c) for c in cells)]
    os.makedirs(out_dir, exist_ok=True)
    chash = cfg.signature_hash("full")
    rows = {}
    log("survey: %d cells, steps %s%s -> %s" % (len(sel), ",".join(steps), "; force " + ",".join(force) if force else "",
                                                 out_dir))
    for n, rec in enumerate(sel, start=1):
        cell = survey_io.CellDir(out_dir, rec["specimen_id"])
        for s in force:                               # a forced step and every step after it run again
            gone = survey_io.clear_markers(cell, ORDER[ORDER.index(s):])
            if gone:
                log("  %d: markers cleared %s" % (cell.sid, ",".join(gone)))
        ctx = Context(rec, cell, source if source_for is None else source_for(cell.sid), reuse_search)
        row = _status_row(cell, rec, steps)
        failed = False
        for s in steps:
            if failed:
                row[s] = "skipped:after_failure"
                continue
            if cell.done(s):
                row[s] = "done"
                continue
            if s != "align" and ctx.alignment is None:
                row[s] = "skipped:no_alignment"
                continue
            if s != "align" and ctx.set_aside() and not include_set_aside:
                row[s] = "skipped:set_aside"
                continue
            t0 = time.perf_counter()
            try:
                summary = STEPS[s](ctx, cfg)
            except Exception as exc:          # noqa: BLE001 -- one cell's failure must not stop the survey (D-085.12)
                failed = True
                row[s], row["error"] = "failed", "%s: %s: %s" % (s, type(exc).__name__, exc)
                cell.log_error(s, traceback.format_exc())
                log("  [%d/%d] %d %s: FAILED %s" % (n, len(sel), cell.sid, s, row["error"]))
                continue
            survey_io.write_marker(cell, s, summary, time.perf_counter() - t0, chash)
            row[s] = "done"
            log("  [%d/%d] %d %s: %s (%.0f s)" % (n, len(sel), cell.sid, s, SAY.get(s, str)(summary),
                                                  time.perf_counter() - t0))
        rows[cell.sid] = row
        survey_io.write_status(out_dir, [rows[k] for k in rows], steps)
    counts = {}
    for r in rows.values():
        for s in steps:
            counts[(s, r[s].split(":")[0])] = counts.get((s, r[s].split(":")[0]), 0) + 1
    log("survey: %s -> %s" % (", ".join("%s %s %d" % (s, st, c) for (s, st), c in sorted(counts.items())),
                              os.path.join(out_dir, "status.csv")))
    return [rows[k] for k in rows]


def status(out_dir, selection_path="", log=print):
    """The status table from the markers alone (and cell.json's set-aside)."""
    if selection_path:
        sel = survey_io.read_selection(selection_path)
    else:
        sel = [dict(specimen_id=int(d), group="") for d in sorted(os.listdir(out_dir))
               if d.isdigit() and os.path.isdir(os.path.join(out_dir, d))]
    rows = []
    for rec in sel:
        cell = survey_io.CellDir(out_dir, rec["specimen_id"])
        row = _status_row(cell, rec, ORDER)
        al = cell.context().get("alignment") or {}
        if al.get("set_aside"):
            for s in ORDER[1:]:
                if row[s] == "pending":
                    row[s] = "skipped:set_aside"
        rows.append(row)
    path = survey_io.write_status(out_dir, rows, ORDER)
    for r in rows:
        log("  %-10s %-18s %s" % (r["specimen_id"], r["group"], " ".join("%s=%s" % (s, r[s]) for s in ORDER)))
    log("status: %d cells -> %s" % (len(rows), path))
    return rows


# ------------------------------------------------------------ pooled ---

def _cells(out_dir, selection_path=""):
    """The (cell folder, selection record) of the selection's cells, or of every cell folder in out_dir."""
    if selection_path:
        recs = survey_io.read_selection(selection_path)
    else:
        recs = [dict(specimen_id=int(d)) for d in sorted(os.listdir(out_dir))
                if d.isdigit() and os.path.isdir(os.path.join(out_dir, d))]
    out = []
    for rec in recs:
        cell = survey_io.CellDir(out_dir, rec["specimen_id"])
        sel = cell.context().get("selection") or rec
        out.append((cell, dict(sel, specimen_id=cell.sid)))
    return out


def read_phantom_rows(paths):
    """The phantoms' fitted d_hat and mu_hat d_hat in S (Block 6's replicate CSVs), for Block 17.3's band."""
    from allen_diameter.loading import table_io
    c = table_io.read_columns(list(paths), ("d_hat_um", "mu_hat_per_um", "in_S"))
    d, m = c["d_hat_um"], c["mu_hat_per_um"]
    ok = c["in_S"] & np.isfinite(d) & np.isfinite(m) & (d > 0) & (m > 0)
    return dict(d_hat=d[ok], mud=(m * d)[ok])


def _flat(row, key, prefix):
    from allen_diameter.analysis.coverage import QUANTILES
    return {"q%02d_%s" % (round(100 * q), prefix): v for q, v in zip(QUANTILES, row.pop(key))}


def pool_coverage(cfg, out_dir, selection_path="", phantom_paths=(), log=print):
    """Block 17.3-17.5 over the cells with a `nodes` marker."""
    from allen_diameter.analysis import coverage
    from allen_diameter.loading import table_io
    from allen_diameter.plotting import survey_figures
    import matplotlib.pyplot as plt
    rows_by_cell, per_cell = [], {}
    for cell, rec in _cells(out_dir, selection_path):
        if not cell.done("nodes"):
            continue
        rows = table_io.read_rows(cell.path_of("nodes", "pilot_%d.csv" % cell.sid))
        rows_by_cell.append((rec, rows))
    if not rows_by_cell:
        raise RuntimeError("pool coverage: no cell under %s has finished the step nodes" % out_dir)
    nodes = coverage.node_table(rows_by_cell)          # S recomputed from the rows' columns (D-098.3)
    for rec, rows in rows_by_cell:
        mine = [n for n in nodes if n["specimen_id"] == int(rec["specimen_id"])]
        per_cell[str(int(rec["specimen_id"]))] = dict(group=rec.get("group"), n=len(rows),
                                                      n_in_S=sum(1 for n in mine if n["in_S"]),
                                                      n_in_S_stored=sum(1 for n in mine if n["in_S_stored"]))
    sel = coverage.selection_summary(nodes)
    log("coverage: S recomputed from the rows' fit_status, reg_verdict and flags (flags of S: %s) for %d of %d rows; "
        "%d now in S and %d now out of S against their stored in_S (CSVs written under another S)%s"
        % (", ".join(sel["flags"]), sel["n_recomputed"], sel["n_rows"], sel["n_now_in_S"], sel["n_now_out_of_S"],
           "; %d rows without those columns keep their stored in_S" % sel["n_stored_used"] if sel["n_stored_used"]
           else ""))
    phantom = read_phantom_rows(phantom_paths) if phantom_paths else None
    bins = coverage.coverage_table(nodes, cfg, phantom)
    ver = coverage.verdict(bins, cfg)
    dia = coverage.diameter_table(nodes)
    s3a = os.path.join(out_dir, "s3a")
    os.makedirs(s3a, exist_ok=True)
    flat = []
    for b in bins:
        b = dict(b)
        b.update(_flat(b, "q_ln_mud", "ln_mud"))
        b.update(_flat(b, "q_ln_alpha", "ln_alpha"))
        flat.append(b)
    table_io.write_rows(flat, os.path.join(s3a, "coverage.csv"))
    dflat = []
    for d in dia:
        d = dict(d)
        d.update(_flat(d, "q_d_hat", "d_hat_um"))
        d.update(_flat(d, "q_D_allen", "D_allen_um"))
        dflat.append(d)
    table_io.write_rows(dflat, os.path.join(s3a, "diameters.csv"))
    survey_io.write_json(os.path.join(s3a, "coverage.json"), dict(
        bins=bins, verdict=ver, per_cell=per_cell, outside_S=coverage.outside_counts(nodes), selection=sel,
        mu_range_per_um=list(cfg.phantom.mu_range_per_um), dhat_bins_um=list(cfg.survey.coverage_dhat_bins_um),
        phantom_rows=list(phantom_paths), n_phantom_rows=None if phantom is None else int(phantom["d_hat"].size)))
    figs = survey_figures.coverage_figures(nodes, bins, cfg, phantom)
    figs["diameters"] = survey_figures.diameter_figure(nodes)
    for name, fig in figs.items():
        fig.savefig(os.path.join(s3a, "%s.png" % name), dpi=110)
        plt.close(fig)
    log("coverage: %d cells, %d nodes, %d in S and usable" % (len(rows_by_cell), len(nodes), ver["n_nodes"]))
    for b in bins:
        log("  d_hat %-8s n %5d (%3d cells)  inside %s  below %s  above %s  %s%s" % (
            b["bin"], b["n"], b["n_cells"], *("%.3f" % v if v == v else "  nan" for v in
                                               (b["share_inside"], b["share_below"], b["share_above"])),
            b["status"], "; phantoms %s" % b["status_phantom"] if b["status_phantom"] else ""))
    log("coverage: %s -- %s%s -> %s" % ("PASS" if ver["passed"] else "FAIL in %s" % ",".join(ver["failing"]),
                                       ver["rule"], "; too few in %s" % ",".join(ver["too_few"]) if ver["too_few"] else "",
                                       s3a))
    return bins, ver


def pool_inputs(cfg, out_dir, selection_path="", log=print):
    """Block 18.4-18.5 over the cells with the markers radius, registration, camera."""
    import matplotlib.pyplot as plt
    from allen_diameter.analysis import s3_inputs
    from allen_diameter.analysis.registration import registration_category
    from allen_diameter.loading import table_io
    from allen_diameter.plotting import survey_figures
    rad, jit, cam = [], [], []
    for cell, rec in _cells(out_dir, selection_path):
        if cell.done("radius"):
            rows = table_io.read_rows(cell.path_of("radius", "radius_nodes_%d.csv" % cell.sid))
            rad.append(dict(specimen_id=cell.sid, node_ids=[int(r["node_id"]) for r in rows],
                            D_allen=[float(r["diameter_um"]) for r in rows]))
        if cell.done("registration"):
            res = survey_io.read_json(cell.path_of("registration", "registration_%d.json" % cell.sid))
            on = [v for v in res.values() if registration_category(v.get("verdict")) == "ON"]
            num = (lambda v: float("nan") if v is None else float(v))
            jit.append(dict(specimen_id=cell.sid, s_star=[num(v.get("lateral_offset_um")) for v in on],
                            dz_star=[num(v.get("z_offset_um")) for v in on]))
        if cell.done("camera"):
            r = survey_io.read_json(cell.path_of("camera", "camera_%d.json" % cell.sid)).get("renderer", {})
            cam.append(dict(specimen_id=cell.sid, background_B_gl=r.get("background_B_gl"),
                            noise_sd_gl=r.get("noise_sd_gl"), jpeg_qtables=r.get("jpeg_qtables")))
    if not (rad or jit or cam):
        raise RuntimeError("pool inputs: no cell under %s has finished radius, registration or camera" % out_dir)
    sug = dict(d_max=s3_inputs.d_max(rad) if rad else dict(value_um=None, note="no cell has finished radius"),
               jitter=s3_inputs.jitter(jit, cfg.survey.inputs_z0_flag_um) if jit else
               dict(jitter_xy_um=None, jitter_z_um=None, note="no cell has finished registration"),
               camera=s3_inputs.camera(cam) if cam else dict(background_B_gl=None, noise_sd_gl=None,
                                                             note="no cell has finished camera"))
    s3b = os.path.join(out_dir, "s3b")
    os.makedirs(s3b, exist_ok=True)
    per = {}
    for r in (sug["d_max"].get("per_cell") or []):
        per.setdefault(r["specimen_id"], {}).update(max_2r_um=r["max_um"], max_node_id=r["node_id"], n_radius=r["n"])
    for r in (sug["jitter"].get("per_cell") or []):
        per.setdefault(r["specimen_id"], {}).update(n_on=r["n_on"], sd_xy_um=r["sd_xy_um"], sd_z_um=r["sd_z_um"],
                                                    median_dz_um=r["median_dz_um"], z0_flag=r["z0_flag"])
    for c in cam:
        per.setdefault(c["specimen_id"], {}).update(background_B_gl=c["background_B_gl"], noise_sd_gl=c["noise_sd_gl"],
                                                    has_tables=c["jpeg_qtables"] is not None)
    table_io.write_rows([dict(specimen_id=k, **v) for k, v in sorted(per.items())], os.path.join(s3b, "inputs_per_cell.csv"))
    survey_io.write_json(os.path.join(s3b, "inputs.json"), dict(
        suggested=dict(phantom_d_max_um=sug["d_max"].get("value_um"), jitter_xy_um=sug["jitter"].get("jitter_xy_um"),
                       jitter_z_um=sug["jitter"].get("jitter_z_um"),
                       background_B_gl=sug["camera"].get("background_B_gl"),
                       noise_sd_gl=sug["camera"].get("noise_sd_gl"), jpeg_qtables=sug["camera"].get("jpeg_qtables")),
        d_max=sug["d_max"], jitter=sug["jitter"], camera=sug["camera"],
        note="suggestions only: Cell 8's DECIDED takes a value when the user writes it (D-065)"))
    fig = survey_figures.inputs_figure(rad, jit, cam, sug)
    fig.savefig(os.path.join(s3b, "inputs.png"), dpi=110)
    plt.close(fig)
    dm, jt, cm = sug["d_max"], sug["jitter"], sug["camera"]
    log("inputs: d_max %s um%s; jitter xy %s um, z %s um%s; background %s gl, noise SD %s gl, %s"
        % (dm.get("value_um"), " (cell %s, node %s; p99 %.2f, p99.5 %.2f um)" % (
            dm["specimen_id"], dm["node_id"], dm["p99_um"], dm["p99_5_um"]) if dm.get("value_um") is not None else "",
           jt.get("jitter_xy_um"), jt.get("jitter_z_um"),
           " (z0 flagged: %s)" % jt["z0_flagged"] if jt.get("z0_flagged") else "",
           cm.get("background_B_gl"), cm.get("noise_sd_gl"),
           "%d table set(s), the most common in %d cells" % (cm.get("n_table_sets", 0), len(cm.get("jpeg_cells", [])))))
    log("inputs: suggestions only (Cell 8 decides) -> %s" % s3b)
    return sug


def pool_calibration(cfg, out_dir, selection_path="", log=print):
    """Block 19.4: the cells' scans pooled round-robin up to the cap per set, then Block 10's fit, Block 15's
    far-fit and the figures on the pooled scans: one kernel (D-088.1)."""
    import argparse as _ap
    import calibrate_kernel
    from allen_diameter.analysis import calibration_pool
    from allen_diameter.loading import calibration_io
    per_cell = []
    for cell, rec in _cells(out_dir, selection_path):
        if cell.done("calibration"):
            per_cell.append((cell.sid, calibration_io.load_scans(cell.path_of("calibration", "scans_%d.json" % cell.sid))[0]))
    if not per_cell:
        raise RuntimeError("pool calibration: no cell under %s has finished the step calibration" % out_dir)
    pooled, meta = calibration_pool.pool_scans(per_cell, cfg)
    s3c = os.path.join(out_dir, "s3c")
    os.makedirs(s3c, exist_ok=True)
    sp, cp, fp = (os.path.join(s3c, x) for x in ("scans_pooled.json", "calibration_pooled.json", "far_fit_pooled.json"))
    calibration_io.save_scans(pooled, sp, dict(meta, source="pooled", cells=[s for s, _ in per_cell]))
    for which in ("near", "far"):
        m = meta["sets"][which]
        log("pooled %s field: %d of %d scans taken (at most %d) from %d cells %s -- %s" % (
            which, m["taken"], m["available"], m["max_nodes"], len(m["by_cell"]), m["by_cell"],
            "sufficient" if m["sufficient"] else "INSUFFICIENT (< %d, D-088.3)" % m["min_nodes"]))
    done = dict(fit=False, far_fit=False)
    for name, call in (("fit", lambda: calibrate_kernel.run_fit(cfg, sp, cp)),
                       ("far_fit", lambda: calibrate_kernel.run_far_fit(cfg, sp, cp, fp))):
        if name == "far_fit" and not done["fit"]:
            log("pooled far-fit not run: the near-field fit did not run")
            continue
        try:
            call()
            done[name] = True
        except SystemExit as exc:                     # run_fit and run_far_fit exit on a fit they cannot do
            log("pooled %s: %s" % (name, exc))
    if done["fit"]:
        calibrate_kernel.run_figures(cfg, _ap.Namespace(scans=[sp], calibration=cp, far_fit=fp if done["far_fit"] else "",
                                                        patches=[], out_dir=os.path.join(s3c, "figures"),
                                                        no_node_figures=False))
        # Block 19.5 (D-090.2): the residuals from the pooled law, per cell and set
        from allen_diameter.analysis import calibration_report as CR
        from allen_diameter.loading import cell_pool
        kc = CR.kernel_law_by_cell(pooled, cfg, calibration_io.load_calibration(cp),
                                   calibration_io.load_calibration(fp) if done["far_fit"] else None, meta["id_map"])
        if kc is not None:
            cell_pool.write_table(os.path.join(s3c, "kernel_law_by_cell.csv"), kc["summary"], CR.BY_CELL_COLUMNS)
            for r in kc["summary"]:
                log("by cell: %s %s field: %d nodes, %d planes used; median residual %+.4f um (IQR %+.4f to %+.4f)" % (
                    "specimen %d" % r["specimen_id"] if r["specimen_id"] is not None else "cell", r["set"],
                    r["n_nodes"], r["n_points_used"], r["median_residual_um"], r["q25_residual_um"],
                    r["q75_residual_um"]))
    log("pool calibration -> %s" % s3c)
    return meta, done


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--selection", required=True)
    r.add_argument("--out-dir", required=True)
    r.add_argument("--cache-dir", default="allen_cache")
    r.add_argument("--steps", default=",".join(ORDER), help="comma-separated, from: %s" % ",".join(ORDER))
    r.add_argument("--cells", default="", help="comma-separated specimen ids (default: every selected cell)")
    r.add_argument("--task-index", type=int, default=None,
                   help="run the one cell of this 0-based row of the selection (a PBS array task; exclusive with --cells)")
    r.add_argument("--cache-per-cell", action="store_true",
                   help="cache each cell's crops and SWC under <cache-dir>/<specimen_id>/ (davinci: many small files)")
    r.add_argument("--force", default="", help="steps to run again (with every step after each)")
    r.add_argument("--include-set-aside", action="store_true")
    r.add_argument("--config-json", default="")
    r.add_argument("--reuse-search", default="",
                   help="a folder of earlier candidate searches (<set>_search_<id>.csv and .json) the calibration step "
                        "reuses when made from the same trace under the same settings (Cell 15: Cell 7's calibration/)")
    s = sub.add_parser("status")
    s.add_argument("--out-dir", required=True)
    s.add_argument("--selection", default="")
    q = sub.add_parser("pool")
    q.add_argument("--out-dir", required=True)
    q.add_argument("--what", required=True, choices=("coverage", "inputs", "calibration"))
    q.add_argument("--selection", default="")
    q.add_argument("--phantom-rows", nargs="*", default=[], help="Block 6 replicate CSVs: the phantoms' band")
    q.add_argument("--config-json", default="")
    a = ap.parse_args(argv)
    if a.cmd == "status":
        status(a.out_dir, a.selection)
        return 0
    if a.cmd == "pool" and a.what == "coverage":
        pool_coverage(load_config(a.config_json), a.out_dir, a.selection, a.phantom_rows)
        return 0
    if a.cmd == "pool" and a.what == "inputs":
        pool_inputs(load_config(a.config_json), a.out_dir, a.selection)
        return 0
    if a.cmd == "pool":
        pool_calibration(load_config(a.config_json), a.out_dir, a.selection)
        return 0
    cfg = load_config(a.config_json)
    split = (lambda text: [x.strip() for x in text.split(",") if x.strip()])
    source_for = (lambda sid: AllenSource(os.path.join(a.cache_dir, str(int(sid))))) if a.cache_per_cell else None
    run(cfg, a.selection, a.out_dir, AllenSource(a.cache_dir), split(a.steps), [int(c) for c in split(a.cells)],
        split(a.force), a.include_set_aside, task_index=a.task_index, source_for=source_for,
        reuse_search=a.reuse_search or None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
