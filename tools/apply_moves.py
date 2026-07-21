#!/usr/bin/env python3
"""
apply_moves.py -- perform the declared path moves (transform T6).

    python3 tools/apply_moves.py --root . --dry-run
    python3 tools/apply_moves.py --root . --apply
    python3 tools/apply_moves.py --root . --verify

WHAT T6 GUARANTEES
------------------
Not one byte of any moved file changes. The tool proves it rather than
asserting it: it hashes every affected file before the move, runs `git mv`,
hashes again, and fails if any digest differs. That equality IS the exit test
for a move-only step -- the same test S0.4 will run over the whole package
layout, which is why this machinery is built here for one directory rather
than improvised later for two hundred files.

WHAT IT KEEPS CONSISTENT
------------------------
A move breaks three things if handled naively:

  1. `ledger.csv` keys rows by path, so a moved file's row would go stale and
     a regeneration would emit a NEW row with no ancestry.
  2. `tools/phase_hashes.json` keys the hash chain by path, so `sha256_pre_s0`
     would be lost for the moved file.
  3. `ancestor_path` for a repository file is its own path, so after the move
     the row would claim to be its own ancestor at the new location, silently
     erasing the fact that it ever lived anywhere else.

This tool rewrites (1) and (2) in place, and `build_ledger.py --moves` handles
(3) on regeneration by resolving a current path back through the move list.

Standard library only, ASCII source, Python 3.8+.
"""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from s0_ledger import render  # noqa: E402


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


REQUIRED_KEYS = ("from", "to", "kind", "transform_id", "stage", "rationale")
KINDS = ("file", "directory")


def _under(inner, outer):
    """True if `inner` is `outer` itself or lies beneath it."""
    outer = outer.rstrip("/")
    return inner == outer or inner.startswith(outer + "/")


def validate_moves(spec):
    """Structural checks on the declared move list.

    With one declared move none of this could go wrong. S0.4 declares many,
    and every failure below is silent rather than loud: a shadowed source
    simply never moves, a collision fails halfway through leaving the tree
    part-moved. Refusing the whole list up front is the only safe behaviour.
    Returns a list of problem strings; empty means the list is well formed.
    """
    moves = spec["moves"]
    problems = []
    for i, mv in enumerate(moves):
        for k in REQUIRED_KEYS:
            if k not in mv:
                problems.append("move %d is missing required key '%s'" % (i, k))
        if mv.get("kind") not in KINDS:
            problems.append("move %d has kind %r, expected one of %r"
                            % (i, mv.get("kind"), KINDS))
        if mv.get("transform_id") != "T6":
            problems.append("move %d has transform_id %r, expected 'T6'"
                            % (i, mv.get("transform_id")))
        if mv.get("from") == mv.get("to"):
            problems.append("move %d is a no-op: from == to == %r"
                            % (i, mv.get("from")))

    for i, a in enumerate(moves):
        for j, b in enumerate(moves):
            if i >= j:
                continue
            if a["from"] == b["from"]:
                problems.append("moves %d and %d share a source: %r"
                                % (i, j, a["from"]))
            if a["to"] == b["to"]:
                problems.append("moves %d and %d collide on a target: %r"
                                % (i, j, a["to"]))
            if a["kind"] == "directory" and _under(b["from"], a["from"]):
                problems.append(
                    "move %d's source %r lies under move %d's directory source "
                    "%r; the outer move runs first and the inner source is gone"
                    % (j, b["from"], i, a["from"]))
            if b["kind"] == "directory" and _under(a["from"], b["from"]):
                problems.append(
                    "move %d's source %r lies under move %d's directory source "
                    "%r; the outer move runs first and the inner source is gone"
                    % (i, a["from"], j, b["from"]))
            if _under(b["from"], a["to"]) or _under(a["from"], b["to"]):
                problems.append("moves %d and %d are chained: one's source is "
                                "inside the other's target" % (i, j))
    return problems


def load_moves(path):
    with open(path, "r", encoding="utf-8") as fh:
        spec = json.load(fh)
    if "moves" not in spec:
        raise KeyError("move spec is missing required key 'moves'")
    problems = validate_moves(spec)
    if problems:
        raise ValueError("move list is not well formed:\n  - "
                         + "\n  - ".join(problems))
    return spec


def affected_files(root, mv):
    """Relative paths under a move source, before the move."""
    src = os.path.join(root, mv["from"])
    if mv["kind"] == "file":
        return [mv["from"]] if os.path.isfile(src) else []
    if not os.path.isdir(src):
        return []
    out = []
    for dirpath, _dirnames, filenames in os.walk(src):
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            out.append(os.path.relpath(full, root).replace(os.sep, "/"))
    return sorted(out)


def remap_with_move(path, moves):
    """Rewrite one path through the move list.

    Returns (new_path, mv) where mv is the move that matched, or None.
    Callers that need the stage or the rationale of the matching move use
    this; `remap` is the boolean-flavoured wrapper kept for compatibility.
    """
    for mv in moves:
        if mv["kind"] == "file":
            if path == mv["from"]:
                return mv["to"], mv
        else:
            prefix = mv["from"].rstrip("/") + "/"
            if path.startswith(prefix):
                return mv["to"].rstrip("/") + "/" + path[len(prefix):], mv
    return path, None


def remap(path, moves):
    """Rewrite one path through the move list. Returns (new_path, moved)."""
    new_path, mv = remap_with_move(path, moves)
    return new_path, mv is not None


def unmap(path, moves):
    """Inverse of remap: current path -> pre-move path."""
    for mv in moves:
        if mv["kind"] == "file":
            if path == mv["to"]:
                return mv["from"], True
        else:
            prefix = mv["to"].rstrip("/") + "/"
            if path.startswith(prefix):
                return mv["from"].rstrip("/") + "/" + path[len(prefix):], True
    return path, False


def git(root, *args):
    r = subprocess.run(["git"] + list(args), cwd=root,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("git %s failed: %s" % (" ".join(args), r.stderr.strip()))
    return r.stdout


def do_moves(root, spec, write):
    moves = spec["moves"]
    report = []
    for mv in moves:
        src_abs = os.path.join(root, mv["from"])
        dst_abs = os.path.join(root, mv["to"])
        files = affected_files(root, mv)

        if not files:
            if os.path.exists(dst_abs):
                report.append({"move": mv, "status": "already applied",
                               "files": affected_files(root,
                                                       {"from": mv["to"],
                                                        "kind": mv["kind"]})})
                continue
            raise RuntimeError("move source does not exist: %s" % mv["from"])

        before = dict((f, sha256_file(os.path.join(root, f))) for f in files)

        if not write:
            report.append({"move": mv, "status": "would move",
                           "files": files, "before": before})
            continue

        if os.path.exists(dst_abs):
            raise RuntimeError("move target already exists: %s" % mv["to"])
        # S0.2c moved a directory to a sibling name, so the destination parent
        # always existed. S0.4 moves into towards_eeg/, which does not exist
        # yet, and `git mv` fails with "destination directory does not exist".
        parent = os.path.dirname(dst_abs)
        if parent and not os.path.isdir(parent):
            os.makedirs(parent)
        git(root, "mv", "--", mv["from"], mv["to"])

        # Pair old to new EXPLICITLY. The previous form zipped sorted(before)
        # against sorted(after), which is correct only when the rename is a
        # constant prefix substitution that happens to preserve sort order --
        # true for one directory move, not true in general, and silently wrong
        # rather than loudly wrong when it fails.
        after = {}
        pairs = []
        for f in files:
            new_f, _ = remap(f, [mv])
            full = os.path.join(root, new_f)
            if not os.path.isfile(full):
                raise RuntimeError("file vanished in the move: %s -> %s" % (f, new_f))
            after[new_f] = sha256_file(full)
            pairs.append((f, new_f))

        for old, new in pairs:
            if before[old] != after[new]:
                raise RuntimeError(
                    "T6 VIOLATED: %s changed bytes during the move (%s -> %s)"
                    % (old, before[old][:12], after[new][:12]))

        report.append({"move": mv, "status": "moved", "files": files,
                       "before": before, "after": after})
    return report


def rewrite_ledger(root, moves, write):
    ledger = os.path.join(root, "ledger.csv")
    with open(ledger, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        fields = reader.fieldnames
    n = 0
    for row in rows:
        new_path, mv = remap_with_move(row["path"], moves)
        if mv is None:
            continue
        n += 1
        # ancestor_path keeps the PRE-move location: that is the whole point of
        # recording a move, and it is what makes the S0.4 exit test expressible.
        if row["ancestor_path"] == row["path"]:
            row["ancestor_path"] = row["path"]
        row["path"] = new_path
        row["relationship"] = "mechanical"
        row["transform_id"] = "T6"
        # The stage comes from the move that actually matched. Hardcoding it
        # was correct while exactly one move existed and would have stamped
        # every S0.4 row with "S0.2c".
        row["rationale"] = (row["rationale"] + "; moved by T6 (git mv) at %s, "
                            "bytes unchanged" % mv.get("stage", "S0.4")).strip("; ")
    if write and n:
        render.write_csv_dicts(rows, fields, ledger)
    return n


def rewrite_phase_hashes(root, moves, write):
    path = os.path.join(root, "tools", "phase_hashes.json")
    if not os.path.isfile(path):
        return 0
    with open(path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    n = 0
    for phase, mapping in doc.get("phases", {}).items():
        new_map = {}
        for k, v in mapping.items():
            nk, moved = remap(k, moves)
            n += bool(moved)
            new_map[nk] = v
        doc["phases"][phase] = new_map
    if write and n:
        with open(path, "w", encoding="ascii") as fh:
            json.dump(doc, fh, indent=2, sort_keys=True)
            fh.write("\n")
    return n


def verify(root, spec):
    """Post-move invariants."""
    moves = spec["moves"]
    problems = []
    for mv in moves:
        if os.path.exists(os.path.join(root, mv["from"])):
            problems.append("source still present: %s" % mv["from"])
        if not os.path.exists(os.path.join(root, mv["to"])):
            problems.append("target absent: %s" % mv["to"])
    ledger = os.path.join(root, "ledger.csv")
    with open(ledger, "r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        stale, moved = remap(row["path"], moves)
        if moved:
            problems.append("ledger row still at a pre-move path: %s" % row["path"])
    # every path in the ledger that is not a discard must exist on disk
    for row in rows:
        if row["verdict"] == "discard" or row["relationship"] == "working_branch_edit":
            continue
        if row["ancestor_path"] != row["path"] and row["transform_id"] != "T6":
            continue
        if not os.path.exists(os.path.join(root, row["path"])):
            problems.append("ledger row has no file: %s" % row["path"])
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser(description="Apply declared path moves (T6).")
    ap.add_argument("--root", default=".")
    ap.add_argument("--moves", default=None, help="default <root>/tools/path_moves.json")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--verify", action="store_true")
    args = ap.parse_args(argv)

    moves_path = args.moves or os.path.join(args.root, "tools", "path_moves.json")
    spec = load_moves(moves_path)

    if args.verify:
        problems = verify(args.root, spec)
        for p in problems:
            print("  - %s" % p)
        print("VERIFY: %s" % ("FAIL" if problems else "PASS"))
        return 1 if problems else 0

    try:
        report = do_moves(args.root, spec, write=bool(args.apply))
    except RuntimeError as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 1

    for r in report:
        print("%-14s %s" % (r["status"], r["move"]["from"]))
        print("%-14s %s" % ("->", r["move"]["to"]))
        for f in r.get("files", []):
            print("               %s" % f)

    n_led = rewrite_ledger(args.root, spec["moves"], write=bool(args.apply))
    n_ph = rewrite_phase_hashes(args.root, spec["moves"], write=bool(args.apply))
    print("\nledger rows remapped      : %d" % n_led)
    print("phase-chain keys remapped : %d" % n_ph)

    if not args.apply:
        print("\n--dry-run: nothing written")
        return 0

    problems = verify(args.root, spec)
    for p in problems:
        print("  - %s" % p)
    if problems:
        print("\nEXIT TEST: FAIL")
        return 1
    print("\nEXIT TEST: PASS -- moved, bytes identical, ledger and chain remapped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
