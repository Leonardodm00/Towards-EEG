#!/usr/bin/env python3
"""
stamp_phase.py -- advance the ledger's hash chain by one sub-step.

WHY THIS EXISTS
---------------
`build_ledger.py` writes `sha256_pre_s0` from whatever is on disk and leaves
`sha256_post_s02` / `_post_s03` / `_post_s04` as the sentinel "-". That is
correct at S0.0 and wrong from S0.2 onward, for two reasons:

  1. Nothing filled the chain columns. Smoke assertion 8 (byte-identity
     reconstruction from the ledger) is unimplementable without them.
  2. Worse, re-running `build_ledger.py` after S0.2 would silently overwrite
     `sha256_pre_s0` with the POST-transform hash, because that column is
     measured, not remembered. The pre-S0 state would be lost from the
     regenerable artefact and survive only in git.

This tool fixes both. It maintains `tools/phase_hashes.json` as the durable
record of the chain, stamps one named phase column, and refreshes the measured
columns (`parses`, `n_crlf`, `n_nonascii`, ...) which by construction describe
the file as it is NOW. `build_ledger.py --phases tools/phase_hashes.json` then
reproduces the full ledger, chain included, from a clean checkout.

USAGE
-----
    python3 tools/stamp_phase.py --root . --phase s02            # stamp
    python3 tools/stamp_phase.py --root . --phase s02 --check    # no writes

Standard library only, ASCII source, Python 3.8+.
"""

import argparse
import csv
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from s0_ledger import render, scan  # noqa: E402

PHASES = ("pre_s0", "post_s02", "post_s03", "post_s04")
ABSENT = "-"

MEASURED_COLUMNS = ("size_bytes", "is_python", "parses", "n_syntax_warnings",
                    "n_crlf", "n_lf", "n_nonascii", "has_cookie")


def load_phase_file(path):
    if not os.path.isfile(path):
        return {"note": ("durable record of the ledger hash chain; consumed by "
                         "build_ledger.py --phases so that a full regeneration "
                         "from a clean checkout reproduces the chain"),
                "phases": {}}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def measure(root):
    return dict((r.path, r) for r in scan.scan_tree(root))


def refresh_row(row, rec):
    """Update the measured columns of one CSV row from a fresh measurement."""
    changed = []
    new = {
        "size_bytes": str(rec.size),
        "is_python": str(rec.is_python),
        "parses": str(rec.parses),
        "n_syntax_warnings": str(rec.n_syntax_warnings),
        "n_crlf": str(rec.n_crlf),
        "n_lf": str(rec.n_lf),
        "n_nonascii": str(rec.n_nonascii),
        "has_cookie": str(rec.has_cookie),
    }
    for k, v in new.items():
        if row.get(k) != v:
            changed.append("%s %s->%s" % (k, row.get(k), v))
            row[k] = v
    return changed


def main(argv=None):
    ap = argparse.ArgumentParser(description="Stamp one phase of the ledger hash chain.")
    ap.add_argument("--root", default=".")
    ap.add_argument("--ledger", default=None, help="default <root>/ledger.csv")
    ap.add_argument("--phase-file", default=None,
                    help="default <root>/tools/phase_hashes.json")
    ap.add_argument("--phase", required=True, choices=[p for p in PHASES if p != "pre_s0"])
    ap.add_argument("--check", action="store_true", help="report, write nothing")
    args = ap.parse_args(argv)

    ledger_path = args.ledger or os.path.join(args.root, "ledger.csv")
    phase_path = args.phase_file or os.path.join(args.root, "tools", "phase_hashes.json")
    column = "sha256_" + args.phase

    with open(ledger_path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        fieldnames = reader.fieldnames
    if column not in fieldnames:
        print("ledger has no column %s" % column, file=sys.stderr)
        return 2

    records = measure(args.root)
    phase_doc = load_phase_file(phase_path)
    phases = phase_doc.setdefault("phases", {})

    # Freeze pre_s0 the first time we are run, from the committed ledger.
    if "pre_s0" not in phases:
        phases["pre_s0"] = dict((r["path"], r["sha256_pre_s0"]) for r in rows)
        print("froze pre_s0 for %d rows" % len(phases["pre_s0"]))

    stamped, absent, drifted, refreshed = 0, 0, [], 0
    absent_detail = []
    this_phase = {}
    for row in rows:
        path = row["path"]
        rec = records.get(path)
        if rec is None:
            row[column] = ABSENT
            this_phase[path] = ABSENT
            absent += 1
            # Two legitimate reasons for absence, and they are not the same
            # thing: (i) the row is a discard, (ii) the row is a working-branch
            # file keyed by its LOCAL name, whose content S0.1 wrote to the
            # ancestor path -- so the local name never existed in the tree.
            reason = ("discarded at %s" % row["stage"] if row["verdict"] == "discard"
                      else "working-branch alias; content lives at %s"
                           % row["ancestor_path"])
            absent_detail.append("%s  (%s)" % (path, reason))
            continue
        this_phase[path] = rec.sha256
        row[column] = rec.sha256
        stamped += 1
        frozen = phases["pre_s0"].get(path)
        if frozen and frozen != row["sha256_pre_s0"]:
            drifted.append(path)
        if frozen:
            row["sha256_pre_s0"] = frozen      # never let a rescan overwrite it
        if refresh_row(row, rec):
            refreshed += 1

    phases[column.replace("sha256_", "")] = this_phase

    print("phase            : %s" % args.phase)
    print("rows stamped     : %d" % stamped)
    print("rows absent      : %d" % absent)
    for d in absent_detail:
        print("    - %s" % d)
    print("measured columns refreshed on %d row(s)" % refreshed)
    if drifted:
        print("\nWARNING: sha256_pre_s0 in the ledger had drifted from the frozen "
              "record for %d row(s); the frozen value was restored:" % len(drifted))
        for p in drifted[:10]:
            print("  - %s" % p)

    changed_now = [r["path"] for r in rows
                   if r[column] != ABSENT and r[column] != r["sha256_pre_s0"]]
    print("\nfiles whose bytes differ from pre-S0: %d" % len(changed_now))
    for p in changed_now:
        print("  - %s" % p)

    if args.check:
        print("\n--check: nothing written")
        return 0

    render.write_csv_dicts(rows, fieldnames, ledger_path)
    with open(phase_path, "w", encoding="ascii") as fh:
        json.dump(phase_doc, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print("\nwrote %s\nwrote %s" % (ledger_path, phase_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
