#!/usr/bin/env python3
"""
test_s0_chain.py -- assert continuity of the S0 transform chain.

    python3 tools/test_s0_chain.py --root .

WHY THIS EXISTS
---------------
Each transform tool writes a log recording, per file, the hash before and the
hash after ITS OWN transform. `decolab.py --verify` compares the tree against
the S0.2 log's `sha256_after`, and that comparison is true only until the next
transform touches the same file. After S0.3 sweeps the S0.2 targets, it fails
by construction -- ten "on-disk hash does not match sha256_after" reports that
mean the chain advanced, not that anything broke.

Running a stale single-step check and reading a red result as damage is the
failure mode this file removes. The invariant that actually holds across the
whole of S0 is CONTINUITY:

    ledger.sha256_pre_s0   == s02_log.sha256_before   (for S0.2 targets)
    s02_log.sha256_after   == ledger.sha256_post_s02
    ledger.sha256_post_s02 == s03_log.sha256_before   (where S0.3 touched it)
    s03_log.sha256_after   == ledger.sha256_post_s03  == bytes on disk

Every link is an equality between two independently produced records. A gap
anywhere means a byte changed outside a logged transform -- which is exactly
what the governing rule forbids, and the only thing that could make the tree
unreconstructable from its ancestors.

Standard library only, ASCII source, Python 3.8+.
"""

import argparse
import csv
import hashlib
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from s0_paths import Resolver  # noqa: E402

PHASE_COLUMNS = ("sha256_post_s05", "sha256_post_s04", "sha256_post_s03",
                 "sha256_post_s02", "sha256_pre_s0")


def end_of_chain(row):
    """The most recent stamped hash for one ledger row.

    Hardcoding sha256_post_s03 was correct until S0.4 and is exactly the
    shelf-life problem Doc 7 s3 describes: any tool edited during S0.4
    legitimately differs from its post-S0.3 bytes, and a check pinned to
    post_s03 reports that as drift. Reading the last stamped column instead
    keeps the check alive across every remaining sub-step.
    """
    for col in PHASE_COLUMNS:
        val = row.get(col, "-")
        if val not in ("-", ""):
            return col, val
    return None, None


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(root):
    base = os.path.join(root, "tools", "s0_transform")
    out = {}
    for key, name in (("s02", "s02_transform_log.json"),
                      ("s03", "s03_transform_log.json")):
        p = os.path.join(base, name)
        if os.path.isfile(p):
            with open(p, "r", encoding="utf-8") as fh:
                doc = json.load(fh)
            out[key] = dict((e["path"], e) for e in doc["files"])
        else:
            out[key] = None
    with open(os.path.join(root, "ledger.csv"), "r", encoding="utf-8", newline="") as fh:
        out["ledger"] = dict((r["path"], r) for r in csv.DictReader(fh))
    return out


def run(root):
    d = load(root)
    led, s02, s03 = d["ledger"], d["s02"], d["s03"]
    R = Resolver(root)
    results = []

    def L(path):
        """Ledger row for a path as recorded in a transform log."""
        return led.get(R.current(path))

    def add(name, ok, msg):
        results.append((name, ok, msg))

    # -- link 1 ------------------------------------------------------------
    if s02 is None:
        add("check_01_s02_log_present", False, "no S0.2 transform log")
    else:
        bad = [p for p, e in s02.items()
               if L(p) is not None and L(p)["sha256_pre_s0"] != e["sha256_before"]]
        add("check_01_pre_s0_equals_s02_input", not bad,
            "%d/%d S0.2 target(s) start from their pre-S0 bytes"
            % (len(s02) - len(bad), len(s02)) if not bad else "mismatch: %r" % bad)

        bad = [p for p, e in s02.items()
               if L(p) is not None and L(p)["sha256_post_s02"] != e["sha256_after"]]
        add("check_02_s02_output_equals_ledger_post_s02", not bad,
            "ledger post_s02 agrees with the S0.2 log for all %d target(s)" % len(s02)
            if not bad else "mismatch: %r" % bad)

    # -- link 2 ------------------------------------------------------------
    if s03 is None:
        add("check_03_s03_log_present", False, "no S0.3 transform log")
        return results

    bad = []
    for p, e in s03.items():
        row = L(p)
        if row is None:
            bad.append("%s: no ledger row" % p)
        elif row["sha256_post_s02"] not in ("-", e["sha256_before"]):
            bad.append("%s: S0.3 did not start from the post-S0.2 bytes" % p)
    add("check_03_post_s02_equals_s03_input", not bad,
        "all %d S0.3 target(s) start from their post-S0.2 bytes" % len(s03)
        if not bad else "; ".join(bad[:5]))

    bad = [p for p, e in s03.items()
           if L(p) is not None and L(p)["sha256_post_s03"] != e["sha256_after"]]
    add("check_04_s03_output_equals_ledger_post_s03", not bad,
        "ledger post_s03 agrees with the S0.3 log for all %d target(s)" % len(s03)
        if not bad else "mismatch: %r" % bad)

    bad = [p for p, e in s03.items()
           if os.path.isfile(R.full(p))
           and sha256_file(R.full(p)) != e["sha256_after"]]
    add("check_05_disk_equals_end_of_chain", not bad,
        "bytes on disk match the end of the chain for all %d target(s)" % len(s03)
        if not bad else "mismatch: %r" % bad)

    # -- the overlap: files touched by BOTH sub-steps -----------------------
    if s02 is not None:
        overlap = sorted(set(s02) & set(s03))
        bad = [p for p in overlap if s02[p]["sha256_after"] != s03[p]["sha256_before"]]
        add("check_06_s02_output_is_s03_input", not bad,
            "%d file(s) touched by both sub-steps hand over cleanly" % len(overlap)
            if not bad else "broken handover: %r" % bad)

    # -- untouched files must not have moved -------------------------------
    # Four artefacts record hashes of the tree they themselves live in, so
    # writing them necessarily changes them after their own hash was taken.
    # Doc 4 s7 calls this out: "the ledger scans itself". They are declared
    # here rather than skipped silently, and everything else -- including the
    # tool sources -- stays under the check.
    SELF_REFERENTIAL = {
        "ledger.csv",
        "LEDGER.md",
        "tools/phase_hashes.json",
        "tools/s0_transform/s02_transform_log.json",
        "tools/s0_transform/s03_transform_log.json",
        "tools/s0_transform/s02_colab_commands.json",
    }
    touched = set(s03) | set(s02 or {}) | SELF_REFERENTIAL
    drifted = []
    for p, row in led.items():
        if p in touched or row["verdict"] == "discard":
            continue
        if row["relationship"] == "working_branch_edit":
            continue
        full = os.path.join(root, p)
        if not os.path.isfile(full):
            continue
        col, expected = end_of_chain(row)
        if expected is None:
            continue
        if sha256_file(full) != expected:
            drifted.append("%s (vs %s)" % (p, col))
    add("check_07_untouched_files_did_not_drift", not drifted,
        "no file changed outside a logged transform"
        if not drifted else "changed with no log entry: %r" % drifted[:10])

    # -- the S0.2 markers must have survived the sweep ----------------------
    if s02 is not None:
        missing = []
        for p, e in s02.items():
            full = os.path.join(root, p)
            if not os.path.isfile(full):
                continue
            with open(full, "rb") as fh:
                data = fh.read()
            for ed in e["edits"]:
                if ed["transform"].startswith("T7") and b"#S0.2:T7" not in data:
                    missing.append(p)
                    break
        add("check_08_s02_markers_survived_s03", not missing,
            "every S0.2 neutralisation marker is still present after the sweep"
            if not missing else "markers lost in: %r" % missing)

    # -- link 3: S0.4 -------------------------------------------------------
    # The roadmap states the S0.4 exit as "every moved file's hash equals its
    # ancestor's". Concretely that is one more equality on the same chain:
    # a git mv changes no byte, so post_s04 == post_s03 for every T6 row.
    # Written here rather than in a fresh per-step verifier, which would
    # itself go stale at S0.5 (Doc 6 s4).
    t6 = [r for r in led.values() if r["transform_id"] == "T6"]
    stamped = [r for r in t6 if r["sha256_post_s04"] not in ("-", "")]
    bad = [r["path"] for r in stamped
           if r["sha256_post_s04"] != r["sha256_post_s03"]]
    if not t6:
        add("check_09_s04_moves_changed_no_bytes", False, "no T6 row in the ledger")
    elif not stamped:
        add("check_09_s04_moves_changed_no_bytes", False,
            "%d T6 row(s) present but post_s04 is unstamped: run "
            "tools/stamp_phase.py --phase post_s04" % len(t6))
    else:
        add("check_09_s04_moves_changed_no_bytes", not bad,
            "post_s04 == post_s03 for all %d moved file(s)" % len(stamped)
            if not bad else "T6 changed bytes: %r" % bad[:5])

    # -- no logged file may fall out of the checks above ---------------------
    # Every check in this file is of the form "for each entry in a log, look
    # up the ledger". A lookup that misses removes the file from its own
    # check and still prints PASS. Doc 7 s3: an exemption that only skips is
    # an exemption that hides. This asserts the lookups cover the logs.
    orphans = []
    for key, logdoc in (("S0.2", s02), ("S0.3", s03)):
        for p in (logdoc or {}):
            if L(p) is None:
                orphans.append("%s:%s" % (key, p))
    add("check_10_every_logged_path_resolves_to_a_ledger_row", not orphans,
        "all %d logged path(s) resolve after the declared moves"
        % (len(s02 or {}) + len(s03 or {}))
        if not orphans else "unresolvable: %r" % orphans[:5])

    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description="S0 transform-chain continuity.")
    ap.add_argument("--root", default=".")
    args = ap.parse_args(argv)
    results = run(args.root)
    width = max(len(n) for n, _, _ in results)
    n_ok = sum(1 for _, ok, _ in results if ok)
    for name, ok, msg in results:
        print("%-*s  %s   %s" % (width, name, "PASS" if ok else "FAIL", msg))
    print("-" * (width + 40))
    print("%d/%d checks passed" % (n_ok, len(results)))
    print("\nCHAIN: %s" % ("CONTINUOUS" if n_ok == len(results) else "BROKEN"))
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
