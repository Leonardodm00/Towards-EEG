#!/usr/bin/env python3
"""
remove_discards.py -- apply transform T12 for sub-step S0.7.

    python3 tools/s0_transform/remove_discards.py --root . --capture
    python3 tools/s0_transform/remove_discards.py --root . --apply
    python3 tools/s0_transform/remove_discards.py --root . --verify

WHAT T12 IS
-----------
Declared removal of a tracked file that no code path references. Admissible
under the byte-identity rule read in the removal direction: a file nothing
depends on may leave the tree, PROVIDED ITS IDENTITY IS PRESERVED OUTSIDE IT.

WHY CAPTURE AND DELETE ARE SEPARATE INVOCATIONS
-----------------------------------------------
S02_discard.sh was deliberately kept out of decolab.py on the grounds that a
tool which rewrites files should not also be able to delete them. The same
reasoning splits this tool: --capture writes the manifest and removes nothing,
--apply refuses to run unless the manifest already exists on disk and agrees
with every byte it claims to describe. The ordering is the guard. If the
process dies between the two, the manifest survives and nothing was lost; if
they were one step, a crash after the first unlink would leave files deleted
and unrecorded, which is precisely the state S0.7 exists to avoid.

WHY THE MANIFEST CARRIES HASHES AND T11's LOG DID NOT NEED TO
--------------------------------------------------------------
T11 (S0.6) is invertible from its own log: the prior bytes can be reconstructed
from the recorded edits. T12 IS NOT INVERTIBLE -- the bytes genuinely leave the
tree. So the hash cannot be derived after the fact and must be recorded before
the fact. This is the material difference between the two transforms and the
reason finding R-7 matters: tools/s0_ledger/analyse.py builds the ledger by
SCANNING THE TREE, so a deleted file loses its row, and with it sha256_pre_s0.
That already happened once, at S0.2, to Population/population_CHANGED.py --
declared, absent, and carrying no ledger row and no hash anywhere.

Decision N-21 settles the repair: build_ledger.py emits a row for every
manifest entry even though the file is absent, so the ledger remains the single
authority and S0.9's byte-identity reconstruction assertion keeps its subject.

Standard library only, ASCII source, Python 3.8+.
"""

import argparse
import csv
import datetime
import hashlib
import json
import os
import subprocess
import sys

MANIFEST = "s07_discard_manifest.json"
EXIT_BOUND_BYTES = 20 * 1024 * 1024          # roadmap TEEG_03 s3.2

# Columns copied verbatim from the ledger row as it stood at the capture
# commit. Recorded in full rather than as a subset because a row rebuilt from
# a subset would silently differ from every other row in the ledger, and a
# reader could not tell a removed file's record from a truncated one.
ROW_FIELDS = (
    "scope", "stage", "verdict", "relationship", "transform_id",
    "ancestor_path", "ancestor_sha256",
    "sha256_pre_s0", "sha256_post_s02", "sha256_post_s03",
    "sha256_post_s04", "sha256_post_s05", "sha256_post_s06",
    "size_bytes", "is_python", "parses", "n_syntax_warnings",
    "n_crlf", "n_lf", "n_nonascii", "has_cookie", "rationale",
)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tracked_paths(root):
    out = subprocess.run(["git", "ls-files", "-z"], cwd=root,
                         capture_output=True).stdout
    return set(p.decode("utf-8") for p in out.split(b"\0") if p)


def tree_bytes(root):
    """Total size of tracked files present on disk. Excludes .git by
    construction, since git ls-files never lists it."""
    tot = 0
    for p in tracked_paths(root):
        full = os.path.join(root, p)
        if os.path.isfile(full):
            tot += os.path.getsize(full)
    return tot


def load_ledger(root):
    with open(os.path.join(root, "ledger.csv"), "r", encoding="utf-8",
              newline="") as fh:
        return list(csv.DictReader(fh))


def manifest_path(root):
    return os.path.join(root, "tools", "s0_transform", MANIFEST)


def load_manifest(root):
    with open(manifest_path(root), "r", encoding="ascii") as fh:
        return json.load(fh)


def do_capture(root, destination):
    """Write the manifest. Deletes nothing."""
    rows = load_ledger(root)
    D = sorted((r for r in rows if r["verdict"] == "discard"),
               key=lambda r: r["path"])
    if not D:
        raise SystemExit("no discard-verdict row in the ledger; nothing to do")

    entries, total = [], 0
    for r in D:
        rel = r["path"]
        full = os.path.join(root, rel)
        if not os.path.isfile(full):
            raise SystemExit(
                "%s carries verdict discard but is already absent. Capture "
                "must run BEFORE deletion; its bytes can no longer be "
                "recorded." % rel)
        measured = sha256_file(full)
        # The manifest must describe the bytes that are actually there, not
        # the bytes the ledger believes are there. Disagreement means the
        # tree drifted outside a logged transform and is a hard stop.
        if measured != r["sha256_post_s06"]:
            raise SystemExit(
                "%s: on-disk sha256 %s disagrees with ledger post_s06 %s -- "
                "the tree drifted outside a logged transform; stop"
                % (rel, measured[:12], r["sha256_post_s06"][:12]))
        size = os.path.getsize(full)
        if int(r["size_bytes"]) != size:
            raise SystemExit("%s: size on disk %d, ledger says %s"
                             % (rel, size, r["size_bytes"]))
        total += size
        entries.append({
            "path": rel,
            "sha256_at_capture": measured,
            "destination": destination,
            "row": dict((k, r[k]) for k in ROW_FIELDS),
        })

    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                          capture_output=True, text=True).stdout.strip()
    doc = {
        "stage": "S0.7",
        "transform_id": "T12",
        "tool": "tools/s0_transform/remove_discards.py",
        "version": 1,
        "generated": datetime.datetime.now(
            datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "captured_at_commit": head,
        "declaration": (
            "T12 -- declared removal of a tracked file that no code path "
            "references. NOT invertible from its log: the bytes leave the "
            "tree, so every hash here is recorded BEFORE deletion and this "
            "manifest is the only durable record of them outside git history. "
            "Each entry carries the file's complete ledger row as it stood at "
            "captured_at_commit; build_ledger.py re-emits that row even though "
            "the file is absent (decision N-21), so the ledger remains the "
            "single authority and S0.9's byte-identity reconstruction "
            "assertion keeps its subject."),
        "decisions": {
            "N-21": "build_ledger.py emits a row per manifest entry even when "
                    "the file is absent; the ledger stays the single authority.",
            "N-22": "The six plain-text PBS job-log files are included in this "
                    "pass. |D| = 22, not the 16 .zip/.pdf files alone. See "
                    "finding H-7: the set is NOT coextensive with the binary "
                    "payload and 'binary payload' is a misnomer for it.",
            "D-8": "The payload is preserved in a separate repository and "
                   "history is NOT rewritten. .git does not shrink and clone "
                   "cost is unchanged; see exit-scope statement_about_clone_cost.",
        },
        "destination_note": (
            "'unassigned' is a DECLARED GAP, not an oversight. Where the "
            "payload goes, who owns it and how it is cited is an "
            "infrastructure decision outside S0's byte-identity discipline. "
            "Recording a guessed URL would be worse than recording none, "
            "because a wrong destination reads as a settled one."),
        "exit_bound_bytes": EXIT_BOUND_BYTES,
        "n_entries": len(entries),
        "total_bytes": total,
        "entries": entries,
    }
    body = json.dumps(doc, indent=2, sort_keys=True) + "\n"
    if any(ord(c) > 127 for c in body):
        raise SystemExit("non-ASCII would be written to the manifest; stop")
    with open(manifest_path(root), "w", encoding="ascii", newline="") as fh:
        fh.write(body)
    print("captured %d entr(ies), %d bytes (%.2f MB) -> %s"
          % (len(entries), total, total / 1048576.0, MANIFEST))
    print("nothing deleted; run --apply to remove")
    return 0


def do_apply(root):
    """Delete every manifest entry, after verifying each byte-for-byte."""
    if not os.path.isfile(manifest_path(root)):
        raise SystemExit(
            "no manifest at %s. Run --capture first: T12 is not invertible "
            "and deleting before recording destroys the bytes." % MANIFEST)
    doc = load_manifest(root)
    entries = doc["entries"]

    before_tracked = tracked_paths(root)
    before_bytes = tree_bytes(root)

    # -- precondition: every entry present and byte-identical to its record --
    for e in entries:
        full = os.path.join(root, e["path"])
        if not os.path.isfile(full):
            raise SystemExit("%s is already absent; the manifest and the tree "
                             "disagree" % e["path"])
        got = sha256_file(full)
        if got != e["sha256_at_capture"]:
            raise SystemExit("%s: sha256 %s does not match the manifest's %s; "
                             "refusing to delete a file the record does not "
                             "describe" % (e["path"], got[:12],
                                           e["sha256_at_capture"][:12]))
    print("precondition: all %d entr(ies) present and byte-identical to the "
          "manifest" % len(entries))

    for e in entries:
        os.unlink(os.path.join(root, e["path"]))

    # -- postcondition, asserted on the TREE, not on intent ------------------
    # Doc 7 s2's standing rule. The interesting half is the second one: not
    # "did I delete what I meant to" but "did I delete ONLY that".
    declared = set(e["path"] for e in entries)
    still_there = [p for p in declared if os.path.isfile(os.path.join(root, p))]
    if still_there:
        raise SystemExit("POSTCONDITION: %d declared file(s) survived: %r"
                         % (len(still_there), still_there[:5]))
    vanished = set(p for p in before_tracked
                   if not os.path.isfile(os.path.join(root, p)))
    undeclared = sorted(vanished - declared)
    if undeclared:
        raise SystemExit("POSTCONDITION: %d file(s) disappeared that the "
                         "manifest does not declare: %r"
                         % (len(undeclared), undeclared[:5]))
    after_bytes = tree_bytes(root)
    freed = before_bytes - after_bytes
    if freed != doc["total_bytes"]:
        raise SystemExit("POSTCONDITION: freed %d bytes, manifest declares %d"
                         % (freed, doc["total_bytes"]))
    if after_bytes >= EXIT_BOUND_BYTES:
        raise SystemExit("POSTCONDITION: tree is %.2f MB, exit bound is "
                         "%.2f MB" % (after_bytes / 1048576.0,
                                      EXIT_BOUND_BYTES / 1048576.0))
    print("removed   : %d file(s), %.2f MB" % (len(entries), freed / 1048576.0))
    print("tree now  : %.2f MB (bound %.2f MB)"
          % (after_bytes / 1048576.0, EXIT_BOUND_BYTES / 1048576.0))
    print("POSTCONDITION HELD: only the declared files left the tree")
    return 0


def do_verify(root):
    doc = load_manifest(root)
    problems = []
    for e in doc["entries"]:
        if os.path.isfile(os.path.join(root, e["path"])):
            problems.append("%s still present" % e["path"])
        h = e["sha256_at_capture"]
        if len(h) != 64 or any(c not in "0123456789abcdef" for c in h):
            problems.append("%s has a malformed hash" % e["path"])
        if int(e["row"]["size_bytes"]) <= 0:
            problems.append("%s has a non-positive size" % e["path"])
    n = tree_bytes(root)
    if n >= EXIT_BOUND_BYTES:
        problems.append("tree is %.2f MB, bound is %.2f MB"
                        % (n / 1048576.0, EXIT_BOUND_BYTES / 1048576.0))
    for p in problems:
        print("FAIL:", p)
    if problems:
        return 1
    print("verified: %d entr(ies) absent, each with a well-formed hash and "
          "size; tree %.2f MB < %.2f MB"
          % (len(doc["entries"]), n / 1048576.0, EXIT_BOUND_BYTES / 1048576.0))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="S0.7 transform T12.")
    ap.add_argument("--root", default=".")
    ap.add_argument("--destination", default="unassigned",
                    help="where the payload is preserved (D-8); 'unassigned' "
                         "is a declared gap, not a placeholder to be guessed")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--capture", action="store_true")
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--verify", action="store_true")
    args = ap.parse_args(argv)
    if args.capture:
        return do_capture(args.root, args.destination)
    if args.apply:
        return do_apply(args.root)
    return do_verify(args.root)


if __name__ == "__main__":
    sys.exit(main())
