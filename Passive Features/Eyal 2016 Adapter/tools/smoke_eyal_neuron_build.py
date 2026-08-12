"""
smoke_eyal_neuron_build.py -- NEURON-dependent acceptance test for the
Eyal et al. (2016) morphologies.

Run:
    python3 smoke_eyal_neuron_build.py --archive-root ./eyal_archive
    python3 smoke_eyal_neuron_build.py --archive-root ./eyal_archive --backfill

What it checks, per cell:
    * the Neurolucida .asc imports and instantiates
    * at least one soma section, non-zero basal and apical counts
    * the axon is deleted (Eyal's delete_axon(), no replacement stub)
    * the nseg histogram under Eyal's 1 + 2*int(L/40) rule
    * the input resistance implied by the published triplet theta*
    * the effective-membrane-area difference between the two spine-cutoff
      distance origins (soma(0), as in Eyal's .hoc, versus soma(0.5), as in
      the pipeline's PassiveCell)

Each cell is built in its own child process because NEURON's section
namespace is process-global: instantiating a second morphology in the same
process leaves the first one's sections in h.allsec(), which would silently
corrupt every area and Rin measurement after the first cell.

With --backfill, the measured scalars are merged into each specimen's
metadata.json via the builder's pure-I/O entry point.

ASCII-only, LF-only by construction.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List

import eyal_archive_builder as eab
import eyal_reference_scalars as ers


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive-root", required=True)
    ap.add_argument("--cell-tag", default=None,
                    help="restrict to one cell_tag (default: all six)")
    ap.add_argument("--probe-amp-pA", type=float, default=-100.0)
    ap.add_argument("--backfill", action="store_true",
                    help="merge measured scalars into metadata.json")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    root = Path(args.archive_root)
    entries = [e for e in eab.MANIFEST
               if args.cell_tag is None or e["cell_tag"] == args.cell_tag]
    if not entries:
        print("no cell matches --cell-tag %r" % args.cell_tag)
        return 1

    n_pass = n_fail = 0
    rows: List[Dict[str, Any]] = []

    print("=" * 78)
    print("smoke_eyal_neuron_build -- %d cell(s)" % len(entries))
    print("=" * 78)

    for entry in entries:
        sid = int(entry["specimen_id"])
        cell_dir = root / ("specimen_%d" % sid)
        morph = cell_dir / "morphology.asc"
        tag = entry["cell_tag"]
        try:
            if not morph.is_file():
                raise FileNotFoundError(str(morph))
            sc = ers.run_reference_scalars_isolated(
                morph, entry["reference_published"],
                probe_amp_pA=args.probe_amp_pA, verbose=False)

            counts = sc["n_sections"]
            assert counts["soma"] >= 1, "no soma section"
            assert counts["dend"] > 0, "no basal sections"
            assert counts["apic"] > 0, "no apical sections"
            assert counts["axon"] == 0, "axon was not deleted"
            assert sc["n_segments"] > counts["soma"], "discretisation failed"
            assert sc["rin_MOhm"] > 0.0, "non-positive Rin"

            hist = sc["smoke_test"]["nseg_histogram"]
            rows.append({
                "cell_tag": tag, "specimen_id": sid,
                "soma": counts["soma"], "dend": counts["dend"],
                "apic": counts["apic"], "segments": sc["n_segments"],
                "rin_MOhm": sc["rin_MOhm"],
                "area_um2": sc["area_um2_geometric"],
                "spine_origin_delta_pct": sc["spine_origin_area_delta_pct"],
            })

            print("PASS  %-12s  soma=%d dend=%-3d apic=%-3d  seg=%-5d  "
                  "Rin*=%7.1f MOhm  area=%8.0f um^2  spine-origin %+0.3f%%"
                  % (tag, counts["soma"], counts["dend"], counts["apic"],
                     sc["n_segments"], sc["rin_MOhm"],
                     sc["area_um2_geometric"],
                     sc["spine_origin_area_delta_pct"]))
            if args.verbose:
                print("      nseg histogram: %s" % hist)

            if args.backfill:
                eab.backfill_reference_scalars(cell_dir, sc, verbose=False)
            n_pass += 1

        except Exception as exc:
            n_fail += 1
            print("FAIL  %-12s  %s: %s" % (tag, type(exc).__name__, exc))
            if args.verbose:
                traceback.print_exc()

    print("=" * 78)
    print("pass=%d  fail=%d%s" % (n_pass, n_fail,
                                  "  (metadata.json backfilled)"
                                  if args.backfill and not n_fail else ""))
    if rows:
        deltas = [r["spine_origin_delta_pct"] for r in rows]
        print("spine-cutoff origin, effective-area change from soma(0) to "
              "soma(0.5): min %+0.3f%%, max %+0.3f%%" % (min(deltas),
                                                         max(deltas)))
        print("  (Eyal's .hoc uses 'soma distance()' = soma(0); the "
              "pipeline's _precompute_F_per_segment uses soma(0.5). Same "
              "rule, different origin.)")
    print("=" * 78)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
