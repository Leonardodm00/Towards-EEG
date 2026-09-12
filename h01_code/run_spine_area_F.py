#!/usr/bin/env python3
"""One shard of the calibrated spine-area campaign, for a PBS array task.

  python3 run_spine_area_F.py --root /path --cell 1302789404 --task 0 --ntasks 40

Stage 0 preparation (ingest, labelling, shaft-continuation demotion, the
minimum-size filter, the skeleton table, the spine ordering) is deterministic
from the neuron CSV and the CLI parameters, so every task recomputes it in a
few seconds and they all agree by construction. Each task then measures a
CONTIGUOUS block of the spatially ordered spine list -- contiguous, not
strided, so neighbouring spines in one task reuse the same CloudVolume chunks
-- and writes its own ledger plus a sidecar of the parameters it used.
merge_spine_area_F.py unions the ledgers, re-checks those sidecars agree, and
runs assemble_cell once.

Nothing here imports matplotlib, plotly or IPython: figures are a pilot-QC
tool and pull in a GUI stack the cluster environment does not need.

Pure ASCII, LF only.
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

RUNNER_VERSION = "run_spine_area_F v1.0"
DEFAULT_CELLS = (1302789404,)


# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(description="Calibrated spine area -> phi^mesh -> F")
    p.add_argument("--root", required=True,
                   help="campaign root; holds neurons/, out/ and the g table")
    p.add_argument("--cell", type=int, required=True, help="neuron id")
    p.add_argument("--task", type=int, default=0, help="shard index, 0-based")
    p.add_argument("--ntasks", type=int, default=1, help="number of shards")
    p.add_argument("--recon-dir", default=None, help="default <root>/neurons")
    p.add_argument("--stage1-dir", default=None, help="default <root>/stage1")
    p.add_argument("--g-table", default=None,
                   help="default <root>/g_table_cyl_2deg.npz")
    p.add_argument("--out-dir", default=None, help="default <root>/out")
    p.add_argument("--subset", type=int, default=None,
                   help="size-stratified subset of the WHOLE cell, then sharded")
    p.add_argument("--pad-nm", type=float, default=500.0)
    p.add_argument("--max-bytes", type=int, default=1 << 30)
    p.add_argument("--checkpoint-every", type=int, default=25)
    p.add_argument("--progress-every", type=int, default=25)
    p.add_argument("--roi-cache", default=None,
                   help="read-only cache of CELL 12 ROIs; default none")
    p.add_argument("--min-spine-metric", default="protrusion_nm",
                   choices=list(("protrusion_nm", "L_skel_nm", "tip_dist_nm",
                                 "n_nodes")))
    p.add_argument("--min-spine-value", type=float, default=None,
                   help="demote spines below this; omit to disable")
    p.add_argument("--axial-window-nm", type=float, default=None,
                   help="cap on |axial offset| for rind triangles; omit = none")
    p.add_argument("--no-shaft-stub-fix", action="store_true",
                   help="skip shaft_continuation demotion (must match Stage 1)")
    p.add_argument("--dry-run", action="store_true",
                   help="prepare, shard and report; touch no network, write no ledger")
    return p


def resolve(args):
    r = args.root
    args.recon_dir = args.recon_dir or os.path.join(r, "neurons")
    args.stage1_dir = args.stage1_dir or os.path.join(r, "stage1")
    args.g_table = args.g_table or os.path.join(r, "g_table_cyl_2deg.npz")
    args.out_dir = args.out_dir or os.path.join(r, "out")
    if not (0 <= args.task < args.ntasks):
        raise SystemExit("--task must satisfy 0 <= task < ntasks")
    for name, path in (("root", r), ("recon-dir", args.recon_dir),
                       ("stage1-dir", args.stage1_dir)):
        if not os.path.isdir(path):
            raise SystemExit("--%s does not exist: %s" % (name, path))
    if not os.path.isfile(args.g_table):
        raise SystemExit("--g-table does not exist: %s" % args.g_table)
    return args


def param_fingerprint(args, sigmas_all):
    """Everything that must agree across shards, as one short hash. The spine
    id list is included, so a shard built from a different CSV or a different
    minimum-size setting cannot be merged with this one."""
    d = {"runner": RUNNER_VERSION, "cell": args.cell, "subset": args.subset,
         "min_spine_metric": args.min_spine_metric,
         "min_spine_value": args.min_spine_value,
         "shaft_stub_fix": not args.no_shaft_stub_fix,
         "pad_nm": args.pad_nm, "axial_window_nm": args.axial_window_nm,
         "g_table_sha": _sha12(args.g_table),
         "sigmas_sha": hashlib.sha256(
             np.asarray(sigmas_all, dtype=np.int64).tobytes()).hexdigest()[:12],
         "n_sigmas": len(sigmas_all)}
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12], d


def _sha12(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:12]


# --------------------------------------------------------------------------- #
def prepare_cell(args, mods):
    """Ingest -> label -> (shaft-stub demotion) -> components. Identical to the
    Colab CELL 4 + CELL 5 path, so the partition matches Stage 1's."""
    sr, s0, shc, sl, sd, sg, mx = (mods[k] for k in
                                   ("sma_run", "s0_ingest", "shaft_continuation",
                                    "spine_labeller", "spine_density",
                                    "spine_geometry", "morphology_exporter"))
    voc = sr.label_vocabularies(sd, sg)
    if not voc["sourced_from_project"]:
        raise SystemExit("label vocabulary fell back to sma_run literals -- "
                         "the Stage 1 modules on --stage1-dir are not the real ones")
    thr = float(mx.SPINE_LENGTH_THRESHOLD_NM)
    src = os.path.join(args.recon_dir, "neuron_%d.csv" % args.cell)
    if not os.path.isfile(src):
        raise SystemExit("neuron CSV not found: %s" % src)
    data_dir = os.path.join(args.out_dir, "s0")
    os.makedirs(data_dir, exist_ok=True)
    csv, _ = s0.write_s0_table(src, data_dir, args.cell)
    nodes = pd.read_csv(csv)
    labelled, _ = sr.label_spines_project(nodes, args.cell, sl, thr,
                                          output_dir=None)
    if not args.no_shaft_stub_fix:
        labelled, _ = shc.demote_shaft_continuations(
            nodes, labelled, spine_density=sd, continuation_threshold_nm=thr)
    nodes["spine_label"] = labelled["annotated_type"].astype(str).str.lower().to_numpy()
    comp = sr.spine_components(
        nodes, sr.spine_mask(nodes, spine_values=voc["spine_labels"]))
    return nodes, labelled, comp


def prepare_all(args, mods):
    """Everything before the first network byte. Returns a dict of state."""
    SAF = mods["h01_spine_area_F"]
    sd = mods["spine_density"]
    nodes, labelled, comp = prepare_cell(args, mods)
    sk_all = SAF.skeleton_spine_table(sd, labelled, nodes, comp)
    short = SAF.short_spine_ids(sk_all, args.min_spine_metric, args.min_spine_value)
    labelled_d, nodes_d, comp_d, dprov = SAF.demote_spines(
        labelled, nodes, comp, sk_all, short, sd.SHAFT_REGEX)
    sk = SAF.skeleton_spine_table(sd, labelled_d, nodes_d, comp_d)
    sigmas_all = SAF.choose_sigmas(sk, args.subset, seed=args.cell)
    fp, fpd = param_fingerprint(args, sigmas_all)
    return {"nodes": nodes_d, "labelled": labelled_d, "comp": comp_d, "sk": sk,
            "sk_all": sk_all, "demoted": dprov, "sigmas_all": sigmas_all,
            "fingerprint": fp, "fingerprint_detail": fpd,
            "axes": SAF.shaft_axes_from_table(sk)}


def shard(sigmas_all, task, ntasks):
    """Contiguous block, so spatial locality (and chunk reuse) survives."""
    if not sigmas_all:
        return []
    return [int(v) for v in np.array_split(np.asarray(sigmas_all, dtype=np.int64),
                                           int(ntasks))[int(task)]]


def task_paths(out_dir, cell, task):
    d = os.path.join(out_dir, "cell%d_shards" % int(cell))
    return {"dir": d,
            "ledger": os.path.join(d, "task_%05d_ledger.npz" % int(task)),
            "meta": os.path.join(d, "task_%05d_meta.json" % int(task))}


# --------------------------------------------------------------------------- #
def import_modules(args):
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (here, args.stage1_dir):          # code dir first, Stage 1 after
        if p not in sys.path:
            sys.path.append(p) if p == args.stage1_dir else sys.path.insert(0, p)
    import importlib
    names = ("h01_spine_area_F", "h01_spine_batch", "h01_spine_roi",
             "h01_area_calibration", "sma_run", "s0_ingest",
             "shaft_continuation", "spine_labeller", "spine_density",
             "spine_geometry", "morphology_exporter")
    mods = {}
    for n in names:
        try:
            mods[n] = importlib.import_module(n)
        except ImportError as exc:
            raise SystemExit("cannot import %r (%s).\n  code dir : %s\n  stage1   : %s"
                             % (n, exc, here, args.stage1_dir))
    return mods


def main(argv=None, reader_factory=None):
    args = resolve(build_parser().parse_args(argv))
    t0 = time.time()
    mods = import_modules(args)
    SAF = mods["h01_spine_area_F"]
    print("%s | %s | cell %d | task %d/%d"
          % (RUNNER_VERSION, SAF.MODULE_VERSION, args.cell, args.task, args.ntasks))

    table, g_lookup, cal_rep = SAF.load_calibration(args.g_table)
    print("g table: %s sha %s  min %.4f med %.4f max %.4f  axes %s"
          % (os.path.basename(cal_rep["path"]), cal_rep["sha256_12"],
             cal_rep["g_min"], cal_rep["g_median"], cal_rep["g_max"],
             cal_rep["axes"]))
    if int(mods["h01_spine_batch"].DEFAULTS["taubin_iterations"]) != 14:
        raise SystemExit("the g table was measured at 14 Taubin iterations but "
                         "h01_spine_batch.DEFAULTS says %s"
                         % mods["h01_spine_batch"].DEFAULTS["taubin_iterations"])

    st = prepare_all(args, mods)
    mine = shard(st["sigmas_all"], args.task, args.ntasks)
    print("cell has %d spines after filtering (%d demoted, %.1f%% of skeleton "
          "spine area); %d selected, this task takes %d"
          % (len(st["sk"]), st["demoted"]["n_spines_demoted"],
             100 * st["demoted"]["A_skel_demoted_um2"]
             / max(st["sk_all"]["A_skel_um2"].sum(), 1e-30),
             len(st["sigmas_all"]), len(mine)))
    print("fingerprint %s  prep %.1f s" % (st["fingerprint"], time.time() - t0))

    if args.dry_run:
        span = "%d ... %d" % (mine[0], mine[-1]) if mine else "none"
        print("DRY RUN: %d sigma(s) in this shard: %s" % (len(mine), span))
        print("DRY RUN: no network touched, no ledger written")
        return 0
    if not mine:
        print("nothing to do for this shard")
        return 0

    paths = task_paths(args.out_dir, args.cell, args.task)
    os.makedirs(paths["dir"], exist_ok=True)
    with open(paths["meta"], "w") as fh:
        json.dump({"fingerprint": st["fingerprint"],
                   "detail": st["fingerprint_detail"], "task": args.task,
                   "ntasks": args.ntasks, "n_shard": len(mine),
                   "host": os.uname().nodename,
                   "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                time.gmtime())}, fh, indent=1)

    factory = SAF.memoized_reader_factory(reader_factory)
    nodes_d, comp_d = st["nodes"], st["comp"]

    def roi_fn(sid):
        return SAF.get_spine_roi(nodes_d, comp_d, sid, args.cell,
                                 roi_dir=args.roi_cache, pad_nm=args.pad_nm,
                                 reader_factory=factory, max_bytes=args.max_bytes,
                                 use_cache=args.roi_cache is not None)

    recs, _ = SAF.measure_all_spines(
        mine, roi_fn, g_lookup, paths["ledger"], cell_id=args.cell,
        checkpoint_every=args.checkpoint_every, progress_every=args.progress_every,
        shaft_axes=st["axes"], axial_window_nm=args.axial_window_nm,
        require_keys=("A_rind_um2",))
    ok = sum(1 for r in recs.values() if r.get("ok"))
    print("task %d done: %d/%d measured, %d failed, %.1f min, ledger %s"
          % (args.task, ok, len(recs), len(recs) - ok, (time.time() - t0) / 60,
             paths["ledger"]))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
