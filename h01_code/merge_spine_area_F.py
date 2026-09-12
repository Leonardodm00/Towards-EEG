#!/usr/bin/env python3
"""Union the shard ledgers of one cell, then assemble F once.

  python3 merge_spine_area_F.py --root /path --cell 1302789404 --ntasks 40

Refuses to merge shards whose parameter fingerprints disagree: a shard built
from a different neuron CSV, a different minimum-size setting or a different g
table would otherwise be silently averaged into the result. Missing shards are
reported and the merge continues, so a campaign with three dead tasks still
yields a usable (and explicitly incomplete) answer.

Pure ASCII, LF only.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

MERGER_VERSION = "merge_spine_area_F v1.0"


def build_parser():
    p = argparse.ArgumentParser(description="Merge shards and assemble F")
    p.add_argument("--root", required=True)
    p.add_argument("--cell", type=int, required=True)
    p.add_argument("--ntasks", type=int, required=True)
    p.add_argument("--recon-dir", default=None)
    p.add_argument("--stage1-dir", default=None)
    p.add_argument("--g-table", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--subset", type=int, default=None)
    p.add_argument("--pad-nm", type=float, default=500.0)
    p.add_argument("--min-spine-metric", default="protrusion_nm")
    p.add_argument("--min-spine-value", type=float, default=None)
    p.add_argument("--axial-window-nm", type=float, default=None)
    p.add_argument("--no-shaft-stub-fix", action="store_true")
    p.add_argument("--kappa-min-per-bin", type=int, default=25)
    p.add_argument("--allow-missing", action="store_true",
                   help="proceed even if some shards never wrote a ledger")
    p.add_argument("--merged-ledger", default=None,
                   help="also write the unioned ledger here")
    return p


def main(argv=None):
    import run_spine_area_F as R           # same directory, shares the CLI logic

    args = build_parser().parse_args(argv)
    args.task, args.ntasks_check = 0, args.ntasks
    args = R.resolve(args)
    mods = R.import_modules(args)
    SAF = mods["h01_spine_area_F"]
    sd = mods["spine_density"]
    t0 = time.time()
    print("%s | %s | cell %d" % (MERGER_VERSION, SAF.MODULE_VERSION, args.cell))

    st = R.prepare_all(args, mods)
    print("fingerprint here: %s (%d spines selected)"
          % (st["fingerprint"], len(st["sigmas_all"])))

    recs, H, missing, bad = {}, {}, [], []
    for k in range(int(args.ntasks)):
        p = R.task_paths(args.out_dir, args.cell, k)
        if not os.path.isfile(p["ledger"]):
            missing.append(k)
            continue
        if os.path.isfile(p["meta"]):
            fp = json.load(open(p["meta"])).get("fingerprint")
            if fp != st["fingerprint"]:
                bad.append((k, fp))
                continue
        r, h = SAF.load_ledger(p["ledger"])
        recs.update(r)
        H.update(h)
    if bad:
        raise SystemExit(
            "shard(s) built with different parameters: %s\n  expected %s\n"
            "  Re-run those tasks with the SAME flags, or delete their ledgers."
            % (", ".join("task %d -> %s" % b for b in bad), st["fingerprint"]))
    if missing and not args.allow_missing:
        raise SystemExit("no ledger for task(s) %s -- re-run them, or pass "
                         "--allow-missing to assemble without them"
                         % ", ".join(str(m) for m in missing))
    if missing:
        print("WARNING: %d shard(s) missing (%s); F below is incomplete"
              % (len(missing), ", ".join(str(m) for m in missing)))

    ok = sum(1 for r in recs.values() if r.get("ok"))
    print("merged %d record(s) from %d shard(s): %d ok, %d failed"
          % (len(recs), int(args.ntasks) - len(missing), ok, len(recs) - ok))
    if args.merged_ledger:
        SAF.save_ledger(args.merged_ledger, recs, H)
        print("wrote merged ledger %s" % args.merged_ledger)

    out = SAF.assemble_cell(sd, st["labelled"], st["nodes"], st["comp"], recs,
                            args.cell, sk=st["sk"],
                            min_per_bin=args.kappa_min_per_bin)
    out["summary"].update({
        "merger_version": MERGER_VERSION, "fingerprint": st["fingerprint"],
        "n_shards_missing": len(missing),
        "min_spine_metric": args.min_spine_metric,
        "min_spine_value": args.min_spine_value,
        "n_spines_demoted_min": st["demoted"]["n_spines_demoted"],
        "A_skel_demoted_um2": st["demoted"]["A_skel_demoted_um2"]})
    paths = SAF.write_cell_outputs(args.out_dir, args.cell, out)

    s = out["summary"]
    print("gate max|diff| %.2e um2 -> PASS"
          % s["attribution_gate"]["max_abs_diff_um2"])
    print("measured %d/%d (%.1f%% of skeleton spine area), %d failed, %d clipped"
          % (s["n_measured"], s["n_spines"], 100 * s["coverage_area"],
             s["n_failed"], s["n_clipped"]))
    print("junction: rind %.1f%% of A_mesh (median, max %.1f%%), %d without an "
          "axis | base frustum %.1f%% of A_skel (pooled)"
          % (100 * s["frac_rind_median"], 100 * s["frac_rind_max"],
             s["n_no_shaft_axis"], 100 * s["base_frac_of_skel_pooled"]))
    print("kappa  raw %.3f | rind removed %.3f | base removed %.3f | both %.3f"
          % (s["kappa_pooled"], s["kappa_pooled_norind"],
             s["kappa_pooled_nobase"], s["kappa_pooled_both"]))
    for lab in ("skel", "mesh_uncalibrated", "mesh_skelfill", "mesh",
                "mesh_norind"):
        print("F_lit %-18s %.4f   F_whole %.4f"
              % (lab, s["F_lit_" + lab], s["F_whole_" + lab]))
    print(out["kappa_table"].to_string(index=False))
    print("wrote %s  (%.1f min)" % (json.dumps(paths, indent=1),
                                    (time.time() - t0) / 60))
    return 0


if __name__ == "__main__":
    sys.exit(main())
