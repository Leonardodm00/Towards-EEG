#!/usr/bin/env python3
"""S3 across cells -- the resumable per-cell runner and its steps (Block 16.2
(g) in specs/SPEC.md and the blocks that add steps; D-085, D-085.12). Colab:
api.brain-map.org is blocked from the sandbox.

  run     for each cell of selection.csv (Block 16.1) and each named step, in
          order: skip a step already done (its marker in <out>/<sid>/done/),
          unless --force names it (which also clears the markers of the
          steps after it); skip every step after `align` of a cell the
          alignment set aside, unless --include-set-aside; a step that raises
          is recorded as failed (errors.log has the traceback), the cell's
          later steps are skipped, and the next cell starts
  status  the table of every cell's steps, rebuilt from the markers

  pool    the pooled analyses over the cells: --what coverage (Block 17.3-17.5)

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

Example (Colab, after Cell 11)
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

    def planes(self, sid):
        return aio.plane_table(self.images(sid))


class Context:
    """One cell for the steps: its selection record, its folder, the source,
    and its context file (cell.json: the SWC path, the alignment)."""

    def __init__(self, record, cell, source):
        self.record, self.cell, self.source = record, cell, source
        self.sid = cell.sid
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
        k, meds = cell_alignment.darkest(arrays)
        row = cands.iloc[k]

        def get_crop(left, top, width, height, q):
            return aio.fetch_crop(ctx.source.fetcher, int(row["id"]), left, top, width, height, q, rho0)[0]

        rng = np.random.default_rng([int(sv.selection_seed), int(sid)])
        rec, imgs = cell_alignment.align_record(swc, arrays[k], sv.align_downsample,
                                                (int(row["width"]), int(row["height"])), get_crop, rho0, sv, rng)
        rec["projection"] = dict(id=int(row["id"]), image_type=str(row["image_type"]), candidates=[
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


def _say_align(s):
    return "shift (%.1f, %.1f) px, %s, ratio %s, %s%s" % (
        s["shift_x_px"], s["shift_y_px"], "flip" if s["flip_h_px"] is not None else "no flip",
        "%.2f" % s["score_ratio"] if s.get("score_ratio") is not None else "n/a", s["method"],
        "; SET ASIDE (%s)" % ",".join(s["reasons"]) if s["set_aside"] else "")


STEPS = {"align": step_align, "nodes": step_nodes}          # Blocks 18-19 add theirs, in this order
ORDER = ("align", "nodes")
SAY = {"align": _say_align, "nodes": _say_nodes}


# ----------------------------------------------------------------- runner ---

def _status_row(cell, record, steps):
    row = dict(specimen_id=cell.sid, group=record.get("group", ""))
    for s in steps:
        row[s] = "done" if cell.done(s) else "pending"
    return row


def run(cfg, selection_path, out_dir, source, steps, cells=None, force=(), include_set_aside=False, log=print):
    """Block 16.2 (g). Returns the status rows."""
    unknown = [s for s in list(steps) + list(force) if s not in STEPS]
    if unknown:
        raise ValueError("unknown steps %s (known: %s)" % (unknown, list(ORDER)))
    steps = [s for s in ORDER if s in steps]
    sel = survey_io.read_selection(selection_path)
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
        ctx = Context(rec, cell, source)
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
        per_cell[str(cell.sid)] = dict(group=rec.get("group"), n=len(rows),
                                       n_in_S=sum(1 for r in rows if coverage._true(r.get("in_S"))))
    if not rows_by_cell:
        raise RuntimeError("pool coverage: no cell under %s has finished the step nodes" % out_dir)
    nodes = coverage.node_table(rows_by_cell)
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
        bins=bins, verdict=ver, per_cell=per_cell, outside_S=coverage.outside_counts(nodes),
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--selection", required=True)
    r.add_argument("--out-dir", required=True)
    r.add_argument("--cache-dir", default="allen_cache")
    r.add_argument("--steps", default=",".join(ORDER), help="comma-separated, from: %s" % ",".join(ORDER))
    r.add_argument("--cells", default="", help="comma-separated specimen ids (default: every selected cell)")
    r.add_argument("--force", default="", help="steps to run again (with every step after each)")
    r.add_argument("--include-set-aside", action="store_true")
    r.add_argument("--config-json", default="")
    s = sub.add_parser("status")
    s.add_argument("--out-dir", required=True)
    s.add_argument("--selection", default="")
    q = sub.add_parser("pool")
    q.add_argument("--out-dir", required=True)
    q.add_argument("--what", required=True, choices=("coverage",))
    q.add_argument("--selection", default="")
    q.add_argument("--phantom-rows", nargs="*", default=[], help="Block 6 replicate CSVs: the phantoms' band")
    q.add_argument("--config-json", default="")
    a = ap.parse_args(argv)
    if a.cmd == "status":
        status(a.out_dir, a.selection)
        return 0
    if a.cmd == "pool":
        pool_coverage(load_config(a.config_json), a.out_dir, a.selection, a.phantom_rows)
        return 0
    cfg = load_config(a.config_json)
    split = (lambda text: [x.strip() for x in text.split(",") if x.strip()])
    run(cfg, a.selection, a.out_dir, AllenSource(a.cache_dir), split(a.steps), [int(c) for c in split(a.cells)],
        split(a.force), a.include_set_aside)
    return 0


if __name__ == "__main__":
    sys.exit(main())
