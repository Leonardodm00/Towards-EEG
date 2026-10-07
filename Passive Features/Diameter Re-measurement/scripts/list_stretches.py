#!/usr/bin/env python3
"""List the dendrite stretches of a cell and suggest the node list of the
registration survey and the pilot -- Block 11 in specs/SPEC.md (design
handoff Next actions 2-3: "5-10 stretches: basal, oblique, trunk, tuft").

One row per unbranched dendrite stretch (survey.stretch_table: SWC type,
nodes, length, path distance from the soma, order, Allen's median diameter,
first / middle / last node id, terminal), and a deterministic suggestion
(survey.suggest_nodes): the --include ids first, then per SWC type the
--per-type stretches with at least --min-nodes nodes spread evenly over path
distance, each by its middle node. The suggestion is a starting point, not a
classification: read the table and edit the list. Writes, in --out-dir:

    stretches_<specimen>.csv    the table
    stretches_<specimen>.json   suggested_nodes, nodes_arg (the comma-separated
                                string the other scripts take), the rule and
                                its parameters, counts per type

Example (Colab, after the bootstrap cell)
    python scripts/list_stretches.py --specimen 529878215 --cache-dir /content/drive/MyDrive/allen_cache \
        --out-dir /content/drive/MyDrive/diameters/stretches --include 4505

Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from allen_diameter.analysis import cell, survey  # noqa: E402
from allen_diameter.loading import swc_io, table_io  # noqa: E402

TYPE_NAMES = {3: "basal", 4: "apical"}
RULE = ("the --include ids first, each standing for its stretch; then per SWC type, among the stretches with "
        ">= min_nodes nodes sorted by path distance from the soma, per_type of them at evenly spaced ranks, "
        "each by its middle node")


def run(swc, out_dir, specimen, types=(3, 4), per_type=3, min_nodes=10, include=(), log=print):
    """Write the table and the suggestion; returns (rows, info)."""
    os.makedirs(out_dir, exist_ok=True)
    rows = survey.stretch_table(swc, types)
    nodes = survey.suggest_nodes(swc, types, per_type, min_nodes, include)
    of_id = {int(swc.ids[r]): s for s, run_ in enumerate(cell.stretches(swc, types)) for r in run_}
    chosen = [of_id[n] for n in nodes]                 # rows are in stretch order: rows[s]["stretch"] == s
    by_type = {}
    for t in sorted({r["type"] for r in rows}):
        sel = [r for r in rows if r["type"] == t]
        by_type[TYPE_NAMES.get(t, "type%d" % t)] = dict(
            n_stretches=len(sel), n_eligible=sum(1 for r in sel if r["n_nodes"] >= min_nodes),
            n_nodes=sum(r["n_nodes"] for r in sel), max_order=max(r["order"] for r in sel))
    info = dict(specimen=str(specimen), suggested_nodes=nodes, nodes_arg=",".join(str(n) for n in nodes), rule=RULE,
                params=dict(types=list(types), per_type=int(per_type), min_nodes=int(min_nodes),
                            include=[int(x) for x in include]),
                n_stretches=len(rows), by_type=by_type,
                n_nodes_measured_by_pilot=sum(rows[s]["n_nodes"] for s in set(chosen)))
    table_io.write_rows(rows, os.path.join(out_dir, "stretches_%s.csv" % specimen))
    with open(os.path.join(out_dir, "stretches_%s.json" % specimen), "w") as f:
        json.dump(info, f, indent=1, sort_keys=True)
    log("%d dendrite stretches: %s" % (len(rows), ", ".join(
        "%s %d (%d with >= %d nodes, order <= %d)" % (k, v["n_stretches"], v["n_eligible"], min_nodes, v["max_order"])
        for k, v in by_type.items())))
    log("suggested stretches (node, type, order, nodes, length um, path start um, Allen d um, terminal):")
    for n, s in zip(nodes, chosen):
        r = rows[s]
        log("  %7d  %-6s  %2d  %4d  %7.1f  %7.1f  %5.2f  %s" % (
            n, TYPE_NAMES.get(r["type"], r["type"]), r["order"], r["n_nodes"], r["length_um"], r["path_start_um"],
            r["allen_diameter_um"], "yes" if r["terminal"] else "no"))
    log("the pilot measures every node of these stretches: %d nodes" % info["n_nodes_measured_by_pilot"])
    log('NODES = "%s"' % info["nodes_arg"])
    return rows, info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--specimen", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--swc", default="", help="an SWC on disk instead of fetching it")
    ap.add_argument("--cache-dir", default="allen_cache", help="where the SWC is fetched to (allen_image_io.fetch_swc)")
    ap.add_argument("--types", default="3,4", help="SWC types counted as dendrite")
    ap.add_argument("--per-type", type=int, default=3)
    ap.add_argument("--min-nodes", type=int, default=10)
    ap.add_argument("--include", default="", help="comma-separated node ids that must be in the list (e.g. 4505)")
    a = ap.parse_args(argv)
    if a.swc:
        path = a.swc
    else:
        import allen_image_io as aio      # flat module in src/; needs the network (Colab)
        path = aio.fetch_swc(int(a.specimen), a.cache_dir)
    swc = swc_io.read_swc(path)
    types = tuple(int(t) for t in a.types.split(","))
    include = [int(x) for x in a.include.split(",") if x.strip()]
    run(swc, a.out_dir, a.specimen, types, a.per_type, a.min_nodes, include, log=lambda m: print(m, flush=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
