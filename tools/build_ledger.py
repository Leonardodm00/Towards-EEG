#!/usr/bin/env python3
"""
build_ledger.py -- stage S0.0 orchestrator for Towards-EEG.

Wires the three layers together and writes the two artefacts:

    scan.scan_tree / scan.scan_files   ->  measurements
    analyse.build_rows                 ->  verdicts
    render.write_markdown / write_csv  ->  LEDGER.md, ledger.csv

Exit status is 0 only if every row carries a final verdict AND every declared
relationship agrees with the measured hashes, so this is usable directly as a
CI gate.

Usage
-----
    python3 tools/build_ledger.py \\
        --root       . \\
        --spec       tools/ancestors.json \\
        --local-dir  ../working_files \\
        --out-dir    .

    python3 tools/build_ledger.py --self-check     # ASCII purity of the tools

Dependencies: Python 3.8+ standard library only.  No numpy, no PyYAML.  This
is deliberate: S0.0 must run inside a bare HPC login shell before any
environment work has been done (S0.8 is blocked and comes later).
"""

import argparse
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from s0_ledger import TOOL_NAME, TOOL_VERSION          # noqa: E402
from s0_ledger import analyse, render, scan            # noqa: E402


def load_spec(path):
    with open(str(path), "r", encoding="utf-8") as fh:
        spec = json.load(fh)
    for key in ("scopes", "working_branch", "transforms"):
        if key not in spec:
            raise KeyError("ancestor spec is missing required key '%s'" % key)
    return spec


def self_check(tools_dir):
    """HPC transfer safety: every .py in the import chain must be pure ASCII.

    A cookie does not help here -- if a cp1252 boundary mangles a multi-byte
    UTF-8 character the declared codec fails to decode it anyway.  ASCII is a
    fixed point of cp1252, latin-1 and UTF-8 alike, so ASCII source cannot be
    corrupted in transit.
    """
    tools_dir = Path(tools_dir)
    ok = True
    for p in sorted(tools_dir.rglob("*.py")):
        data = p.read_bytes()
        bad = [(i + 1, hex(b)) for i, b in enumerate(data) if b > 127]
        if bad:
            ok = False
            print("NON-ASCII  %s  first offenders: %s" % (p, bad[:8]))
        else:
            print("ascii ok   %s" % p)
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build the S0.0 reconciliation ledger.")
    ap.add_argument("--root", help="repository tree to scan")
    ap.add_argument("--spec", help="path to ancestors.json")
    ap.add_argument("--local-dir", help="directory holding the four local working files")
    ap.add_argument("--out-dir", default=".", help="where LEDGER.md and ledger.csv are written")
    ap.add_argument("--as-of", default="post_s01", choices=["pre_s01", "post_s01"],
                    help="which declaration of the working-branch relationship to "
                         "check against; see tools/ancestors.json")
    ap.add_argument("--phases", default=None,
                    help="phase_hashes.json; preserves the hash chain across a "
                         "full regeneration (required from S0.2 onward)")
    ap.add_argument("--self-check", action="store_true",
                    help="byte-scan the tool sources for ASCII purity and exit")
    args = ap.parse_args(argv)

    if args.self_check:
        return 0 if self_check(Path(__file__).resolve().parent) else 1

    for required in ("root", "spec", "local_dir"):
        if getattr(args, required) is None:
            ap.error("--%s is required unless --self-check is given"
                     % required.replace("_", "-"))

    spec = load_spec(args.spec)

    repo_records = scan.scan_tree(args.root)

    local_dir = Path(args.local_dir)
    pairs = []
    for entry in spec["working_branch"]:
        p = local_dir / entry["local"]
        if not p.is_file():
            print("ERROR: working file not found: %s" % p, file=sys.stderr)
            return 2
        pairs.append((p, entry["local"]))
    local_records = scan.scan_files(pairs)

    phases = None
    if args.phases:
        with open(str(args.phases), "r", encoding="utf-8") as fh:
            phases = json.load(fh)

    rows, problems = analyse.build_rows(repo_records, local_records, spec, phases, args.as_of)
    o7 = analyse.find_shadowed_defs(repo_records)
    dup = analyse.find_duplicate_classes(repo_records)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "generated": datetime.datetime.now(datetime.timezone.utc)
                             .strftime("%Y-%m-%d %H:%M:%S UTC"),
        "tool": TOOL_NAME,
        "version": TOOL_VERSION,
        "snapshot": spec.get("snapshot", "unknown"),
        "snapshot_date": spec.get("snapshot_date", "unknown"),
        "root": str(Path(args.root).resolve()),
        "n_repo": len(repo_records),
        "n_local": len(local_records),
    }

    render.write_markdown(rows, o7, dup, problems, meta, out_dir / "LEDGER.md")
    render.write_csv(rows, out_dir / "ledger.csv")

    print("rows              : %d" % len(rows))
    print("shadowed defs (O7): %d across %d files"
          % (len(o7), len({s.path for s in o7})))
    print("duplicate classes : %d names" % len(dup))
    print("written           : %s, %s" % (out_dir / "LEDGER.md", out_dir / "ledger.csv"))
    if problems:
        print("\nEXIT TEST: FAIL -- %d unresolved item(s):" % len(problems))
        for p in problems:
            print("  - %s" % p)
        return 1
    print("\nEXIT TEST: PASS -- all verdicts final, all declarations verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
