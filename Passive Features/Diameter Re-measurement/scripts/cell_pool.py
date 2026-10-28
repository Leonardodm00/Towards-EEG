#!/usr/bin/env python3
"""S3 across cells, step 0 -- the cell pool, the acquisition check and the
stratified selection (Block 16.1 in specs/SPEC.md; D-065, D-085). Colab:
api.brain-map.org is blocked from the sandbox.

  pool    the human cells with a reconstruction in Allen's database (or a CSV
          with the same columns, --from-csv) -> POOL.csv; prints them by class
          and layer, and the dendrite types no class takes
  check   each pool cell's plane spacing (NeuronReconstruction.scale_factor_z),
          pixel size and usable planes -> ACQ.csv and the summary ACQ.json
          (the distinct values, how many cells match the configuration);
          resumable: the cells already in ACQ.csv are skipped unless --fresh
  select  per neuron class and layer, survey.cells_per_group cells drawn by a
          seeded permutation (all of a smaller group), optionally only among
          the cells whose acquisition matches (--require-match) -> SEL.csv and
          SEL.json (the seed, the rule, the groups, the cells left out)
  pick    the selection records of named cells (--cells 529878215,...), in
          that order, each with its class and group -> SEL.csv and SEL.json:
          a selection for a run on chosen cells, e.g. Colab's pilot (D-099.3)

Example (Colab, after the bootstrap cell)
    python scripts/cell_pool.py pool --out /content/drive/MyDrive/diameters/survey/pool.csv
    python scripts/cell_pool.py check --pool .../survey/pool.csv --out .../survey/acquisition.csv
    python scripts/cell_pool.py select --pool .../survey/pool.csv --acquisition .../survey/acquisition.csv \
        --require-match --out .../survey/selection.csv
    python scripts/cell_pool.py pick --pool .../survey/pool.csv --cells 529878215 --out .../survey_pilot/selection.csv

Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from allen_diameter.analysis import cell_selection  # noqa: E402
from allen_diameter.loading import cell_pool  # noqa: E402
from build_table import load_config  # noqa: E402


def _summary_path(out, given):
    return given or os.path.splitext(out)[0] + ".json"


def _write_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
    os.replace(tmp, path)


def run_pool(cfg, out, from_csv=""):
    sv = cfg.survey
    if from_csv:
        rows = cell_pool.read_table(from_csv, {})
        source = "csv:" + from_csv
    else:
        rows = cell_pool.fetch_specimen_details(page_rows=sv.pool_page_rows)
        source = "api:" + cell_pool.SPECIMEN_DETAIL_MODEL
    pool = cell_pool.pool_records(rows, species=sv.species, require_reconstruction=True)
    cell_pool.write_table(out, pool, cell_pool.POOL_COLUMNS)
    table, unmapped = {}, {}
    for rec in pool:
        cls = cell_selection.neuron_class(rec["dendrite_type"], sv.class_map)
        if cls is None:
            unmapped[str(rec["dendrite_type"])] = unmapped.get(str(rec["dendrite_type"]), 0) + 1
            continue
        key = "%s|%s" % (cls, rec["layer"])
        table[key] = table.get(key, 0) + 1
    print("pool: %d %s cells with a reconstruction (%d rows read, %s) -> %s"
          % (len(pool), sv.species, len(rows), source, out))
    print("  by class and layer: %s" % ", ".join("%s %d" % kv for kv in sorted(table.items())))
    print("  dendrite types without a class (left out of the selection): %s" % (unmapped or "none"))
    return pool


def run_check(cfg, pool_path, out, summary_path="", max_cells=0, fresh=False):
    pool = cell_pool.read_table(pool_path, cell_pool.POOL_TYPES)
    done = []
    if os.path.exists(out) and not fresh:
        done = cell_pool.read_table(out, cell_pool.ACQUISITION_TYPES)
    seen = {int(r["specimen_id"]) for r in done}
    todo = [r for r in pool if int(r["specimen_id"]) not in seen]
    if max_cells:
        todo = todo[:int(max_cells)]
    print("check: %d cells in the pool, %d already checked, %d to check" % (len(pool), len(done), len(todo)), flush=True)
    records = list(done)
    for n, rec in enumerate(todo, start=1):
        a = cell_pool.acquisition_record(rec["specimen_id"])
        records.append(a)
        pixel = a["pixel_values"] if ";" in (a["pixel_values"] or "") else a["pixel_um"]
        print("  %d: dz %s um, pixel %s um, %d planes%s" % (a["specimen_id"], a["dz_um"], pixel, a["n_planes"],
                                                          "; ERROR " + a["error"] if a["error"] else ""), flush=True)
        if n % 10 == 0:
            cell_pool.write_table(out, records, cell_pool.ACQUISITION_COLUMNS)
    cell_pool.write_table(out, records, cell_pool.ACQUISITION_COLUMNS)
    summary = cell_selection.acquisition_summary(records, cfg)
    summary["pool"] = pool_path
    spath = _summary_path(out, summary_path)
    _write_json(summary, spath)
    print("check: %d cells, %d failed; dz (um): %s; pixel (um): %s; %d match the configuration (dz %.4f um, pixel "
          "%.4f um, relative tolerance %g) -> %s, %s"
          % (summary["n_cells"], summary["n_failed"], summary["dz_um_counts"], summary["pixel_um_counts"],
             summary["n_match"], cfg.acquisition.dz_um, cfg.acquisition.res0_um, cfg.survey.acquisition_rel_tol,
             out, spath))
    return records, summary


def run_select(cfg, pool_path, out, acquisition_path="", require_match=False, summary_path=""):
    pool = cell_pool.read_table(pool_path, cell_pool.POOL_TYPES)
    acq = cell_pool.read_table(acquisition_path, cell_pool.ACQUISITION_TYPES) if acquisition_path else None
    sel = cell_selection.select_cells(pool, cfg, acq, require_match)
    cell_pool.write_table(out, sel.selected, cell_pool.SELECTION_COLUMNS)
    reasons = {}
    for why in sel.excluded.values():
        reasons[why] = reasons.get(why, 0) + 1
    summary = dict(seed=sel.seed, cells_per_group=cfg.survey.cells_per_group, class_map=list(cfg.survey.class_map),
                   require_match=bool(require_match), pool=pool_path, acquisition=acquisition_path or None,
                   groups={k: dict(available=a, selected=s) for k, (a, s) in sel.groups.items()},
                   excluded={str(k): v for k, v in sorted(sel.excluded.items())}, excluded_counts=reasons,
                   rule="per (class, layer): the first cells_per_group of numpy default_rng([seed, crc32(key)])"
                        ".permutation over the group's cells in increasing specimen_id; all of a smaller group")
    spath = _summary_path(out, summary_path)
    _write_json(summary, spath)
    print("select: %d groups, %d cells selected (seed %d, %d per group); left out %s -> %s, %s"
          % (len(sel.groups), len(sel.selected), sel.seed, cfg.survey.cells_per_group, reasons or "none", out, spath))
    for k, (a, s) in sel.groups.items():
        ids = [r["specimen_id"] for r in sel.selected if r["group"] == k]
        print("  %-12s %3d of %3d: %s" % (k, s, a, " ".join(str(i) for i in ids)))
    return sel


def run_pick(cfg, pool_path, cells, out, acquisition_path="", require_match=False, summary_path=""):
    """Block 16.1 (e): the selection of the named cells (cell_selection.pick_cells)."""
    pool = cell_pool.read_table(pool_path, cell_pool.POOL_TYPES)
    acq = cell_pool.read_table(acquisition_path, cell_pool.ACQUISITION_TYPES) if acquisition_path else None
    ids = [int(x) for x in str(cells).split(",") if x.strip()]
    if not ids:
        raise ValueError("pick: --cells names no specimen")
    sel = cell_selection.pick_cells(pool, cfg, ids, acq, require_match)
    cell_pool.write_table(out, sel.selected, cell_pool.SELECTION_COLUMNS)
    summary = dict(picked=ids, cells_per_group=None, class_map=list(cfg.survey.class_map),
                   require_match=bool(require_match), pool=pool_path, acquisition=acquisition_path or None,
                   groups={k: dict(available=a, selected=s_) for k, (a, s_) in sel.groups.items()},
                   excluded={}, excluded_counts={},
                   rule="the named cells, in the order given, each with its class and group (pick, D-099.3)")
    spath = _summary_path(out, summary_path)
    _write_json(summary, spath)
    print("pick: %d cells -> %s, %s" % (len(sel.selected), out, spath))
    for r in sel.selected:
        print("  %-12s %s" % (r["group"], r["specimen_id"]))
    return sel


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pool")
    p.add_argument("--out", required=True)
    p.add_argument("--from-csv", default="", help="a CSV with the pool's columns instead of the Allen API")
    c = sub.add_parser("check")
    c.add_argument("--pool", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--summary", default="", help="default: the --out path with .json")
    c.add_argument("--max-cells", type=int, default=0, help="check at most this many new cells (0: all)")
    c.add_argument("--fresh", action="store_true", help="check every cell again")
    s = sub.add_parser("select")
    s.add_argument("--pool", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--acquisition", default="")
    s.add_argument("--require-match", action="store_true", help="only cells whose plane spacing and pixel size match")
    s.add_argument("--seed", type=int, default=None, help="default: survey.selection_seed")
    s.add_argument("--per-group", type=int, default=None, help="default: survey.cells_per_group")
    s.add_argument("--summary", default="", help="default: the --out path with .json")
    k = sub.add_parser("pick")
    k.add_argument("--pool", required=True)
    k.add_argument("--cells", required=True, help="comma-separated specimen ids, in the order wanted")
    k.add_argument("--out", required=True)
    k.add_argument("--acquisition", default="")
    k.add_argument("--require-match", action="store_true", help="refuse a cell whose acquisition does not match")
    k.add_argument("--summary", default="", help="default: the --out path with .json")
    for q in (p, c, s, k):
        q.add_argument("--config-json", default="")
    a = ap.parse_args(argv)
    cfg = load_config(a.config_json)
    if a.cmd == "pool":
        run_pool(cfg, a.out, a.from_csv)
    elif a.cmd == "check":
        run_check(cfg, a.pool, a.out, a.summary, a.max_cells, a.fresh)
    elif a.cmd == "pick":
        run_pick(cfg, a.pool, a.cells, a.out, a.acquisition, a.require_match, a.summary)
    else:
        sv = cfg.survey
        sv = dataclasses.replace(sv, selection_seed=sv.selection_seed if a.seed is None else int(a.seed),
                                 cells_per_group=sv.cells_per_group if a.per_group is None else int(a.per_group))
        cfg = dataclasses.replace(cfg, survey=sv)
        cfg.validate()
        run_select(cfg, a.pool, a.out, a.acquisition, a.require_match, a.summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
