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

Steps (in their order)
  align   the cell's global alignment (Block 16.2): the minimum-intensity
          projection at level 4, align_translation with both flips, the
          null's score ratio, the fine stage at level 1 on the cell's box;
          set aside below the ratio threshold or on a search edge.
          <sid>/align/alignment_<sid>.json and overlay_<sid>.png; a hand-made
          <sid>/align/alignment_manual.json is used as given

Example (Colab, after Cell 11)
    python scripts/survey_cells.py run --selection .../survey/selection.csv --out-dir .../survey \
        --cache-dir /content/drive/MyDrive/allen_cache --steps align
    python scripts/survey_cells.py status --out-dir .../survey

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


def _say_align(s):
    return "shift (%.1f, %.1f) px, %s, ratio %s, %s%s" % (
        s["shift_x_px"], s["shift_y_px"], "flip" if s["flip_h_px"] is not None else "no flip",
        "%.2f" % s["score_ratio"] if s.get("score_ratio") is not None else "n/a", s["method"],
        "; SET ASIDE (%s)" % ",".join(s["reasons"]) if s["set_aside"] else "")


STEPS = {"align": step_align}                 # Blocks 17-19 add theirs, in this order
ORDER = ("align",)
SAY = {"align": _say_align}


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
    a = ap.parse_args(argv)
    if a.cmd == "status":
        status(a.out_dir, a.selection)
        return 0
    cfg = load_config(a.config_json)
    split = (lambda text: [x.strip() for x in text.split(",") if x.strip()])
    run(cfg, a.selection, a.out_dir, AllenSource(a.cache_dir), split(a.steps), [int(c) for c in split(a.cells)],
        split(a.force), a.include_set_aside)
    return 0


if __name__ == "__main__":
    sys.exit(main())
