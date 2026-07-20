#!/usr/bin/env python3
"""
test_s0_moves_smoke.py -- correctness harness for tools/apply_moves.py

    python3 tools/test_s0_moves_smoke.py

Synthetic only: every check builds a throwaway git repository, so nothing here
touches the real tree. Half the checks are mutations -- a move that alters
bytes, a move onto an existing target, a missing source, a stale ledger row --
because a move tool that cannot detect a corrupted move is worth nothing.

Standard library only, ASCII source, Python 3.8+.
"""

import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import apply_moves as M  # noqa: E402

LEDGER_FIELDS = ["path", "scope", "stage", "verdict", "relationship",
                 "transform_id", "ancestor_path", "ancestor_sha256",
                 "sha256_pre_s0", "sha256_post_s02", "sha256_post_s03",
                 "sha256_post_s04", "size_bytes", "is_python", "parses",
                 "n_syntax_warnings", "n_crlf", "n_lf", "n_nonascii",
                 "has_cookie", "rationale"]


def _git(root, *a):
    subprocess.run(["git"] + list(a), cwd=root, check=True,
                   capture_output=True, text=True)


def sandbox(dirname="old dir "):
    """A git repo with one directory holding two files, plus a ledger."""
    root = tempfile.mkdtemp(prefix="s02c_")
    os.makedirs(os.path.join(root, dirname))
    contents = {}
    for i, fn in enumerate(("a.py", "b.py")):
        rel = "%s/%s" % (dirname, fn)
        data = ("x = %d\n" % i).encode("ascii")
        with open(os.path.join(root, rel), "wb") as fh:
            fh.write(data)
        contents[rel] = M.sha256_file(os.path.join(root, rel))

    rows = []
    for rel, sha in contents.items():
        row = dict((k, "") for k in LEDGER_FIELDS)
        row.update({"path": rel, "scope": "colab", "stage": "-",
                    "verdict": "retain", "relationship": "origin",
                    "ancestor_path": rel, "ancestor_sha256": sha,
                    "sha256_pre_s0": sha, "sha256_post_s02": sha,
                    "sha256_post_s03": "-", "sha256_post_s04": "-",
                    "is_python": "True", "parses": "True",
                    "rationale": "fixture"})
        rows.append(row)
    with open(os.path.join(root, "ledger.csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=LEDGER_FIELDS)
        w.writeheader()
        w.writerows(rows)

    os.makedirs(os.path.join(root, "tools"))
    with open(os.path.join(root, "tools", "phase_hashes.json"), "w",
              encoding="ascii") as fh:
        json.dump({"phases": {"pre_s0": dict((k, v) for k, v in contents.items()),
                              "post_s02": dict((k, v) for k, v in contents.items())}},
                  fh, indent=2, sort_keys=True)

    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@t")
    _git(root, "config", "user.name", "t")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "fixture")
    spec = {"moves": [{"from": dirname, "to": "new_dir", "kind": "directory",
                       "transform_id": "T6", "stage": "S0.2c",
                       "rationale": "fixture"}]}
    return root, spec, contents


def check_01_dry_run_writes_nothing():
    root, spec, _ = sandbox()
    try:
        M.do_moves(root, spec, write=False)
        M.rewrite_ledger(root, spec["moves"], write=False)
        M.rewrite_phase_hashes(root, spec["moves"], write=False)
        if not os.path.isdir(os.path.join(root, "old dir ")):
            return False, "dry run moved the directory"
        rows = list(csv.DictReader(open(os.path.join(root, "ledger.csv"))))
        if any(r["path"].startswith("new_dir") for r in rows):
            return False, "dry run rewrote the ledger"
        return True, "tree and ledger untouched"
    finally:
        shutil.rmtree(root)


def check_02_move_preserves_every_byte():
    root, spec, before = sandbox()
    try:
        rep = M.do_moves(root, spec, write=True)
        after = rep[0]["after"]
        if sorted(after.values()) != sorted(before.values()):
            return False, "hashes changed across the move"
        return True, "%d file(s) moved with identical digests" % len(after)
    finally:
        shutil.rmtree(root)


def check_03_ledger_paths_remapped_ancestor_kept():
    root, spec, _ = sandbox()
    try:
        M.do_moves(root, spec, write=True)
        M.rewrite_ledger(root, spec["moves"], write=True)
        rows = list(csv.DictReader(open(os.path.join(root, "ledger.csv"))))
        for r in rows:
            if not r["path"].startswith("new_dir/"):
                return False, "row not remapped: %s" % r["path"]
            if not r["ancestor_path"].startswith("old dir "):
                return False, "ancestor lost for %s" % r["path"]
            if r["transform_id"] != "T6" or r["relationship"] != "mechanical":
                return False, "row not marked as a move: %s" % r["path"]
        return True, "paths remapped, ancestors preserved, T6 recorded"
    finally:
        shutil.rmtree(root)


def check_04_phase_chain_remapped():
    root, spec, _ = sandbox()
    try:
        M.do_moves(root, spec, write=True)
        n = M.rewrite_phase_hashes(root, spec["moves"], write=True)
        doc = json.load(open(os.path.join(root, "tools", "phase_hashes.json")))
        for phase, mapping in doc["phases"].items():
            for k in mapping:
                if not k.startswith("new_dir/"):
                    return False, "%s key not remapped: %s" % (phase, k)
        return True, "%d chain key(s) remapped across 2 phases" % n
    finally:
        shutil.rmtree(root)


def check_05_missing_source_is_an_error():
    root, spec, _ = sandbox()
    try:
        spec = json.loads(json.dumps(spec))
        spec["moves"][0]["from"] = "no such dir"
        try:
            M.do_moves(root, spec, write=False)
        except RuntimeError:
            return True, "MUTATION: missing source rejected"
        return False, "MUTATION SURVIVED: missing source accepted"
    finally:
        shutil.rmtree(root)


def check_06_existing_target_is_an_error():
    root, spec, _ = sandbox()
    try:
        os.makedirs(os.path.join(root, "new_dir"))
        try:
            M.do_moves(root, spec, write=True)
        except RuntimeError:
            return True, "MUTATION: refused to move onto an existing target"
        return False, "MUTATION SURVIVED: clobbered an existing target"
    finally:
        shutil.rmtree(root)


def check_07_byte_change_during_move_is_caught():
    """The load-bearing mutation: T6 must be able to fail."""
    root, spec, _ = sandbox()
    real_git = M.git

    def sabotaging_git(r, *a):
        out = real_git(r, *a)
        if a and a[0] == "mv":
            victim = os.path.join(r, "new_dir", "a.py")
            with open(victim, "ab") as fh:
                fh.write(b"# tampered\n")
        return out
    try:
        M.git = sabotaging_git
        try:
            M.do_moves(root, spec, write=True)
        except RuntimeError as exc:
            if "T6 VIOLATED" in str(exc):
                return True, "MUTATION: byte change during the move was detected"
            return False, "raised, but not the T6 check: %s" % exc
        return False, "MUTATION SURVIVED: bytes changed and the move passed"
    finally:
        M.git = real_git
        shutil.rmtree(root)


def check_08_verify_catches_stale_ledger_row():
    root, spec, _ = sandbox()
    try:
        M.do_moves(root, spec, write=True)
        # deliberately skip the ledger rewrite
        problems = M.verify(root, spec)
        if not problems:
            return False, "MUTATION SURVIVED: stale ledger rows passed verify"
        return True, "MUTATION: stale ledger row detected"
    finally:
        shutil.rmtree(root)


def check_09_idempotent_second_apply():
    root, spec, _ = sandbox()
    try:
        M.do_moves(root, spec, write=True)
        M.rewrite_ledger(root, spec["moves"], write=True)
        rep = M.do_moves(root, spec, write=True)
        if rep[0]["status"] != "already applied":
            return False, "second apply did not recognise the applied state"
        problems = M.verify(root, spec)
        if problems:
            return False, "verify failed after a second apply: %r" % problems
        return True, "second apply is a no-op"
    finally:
        shutil.rmtree(root)


def check_10_remap_unmap_are_inverse():
    moves = [{"from": "old dir ", "to": "new_dir", "kind": "directory"},
             {"from": "x/y.py", "to": "z/w.py", "kind": "file"}]
    for p in ("old dir /a.py", "x/y.py", "untouched/file.py"):
        fwd, moved = M.remap(p, moves)
        back, _ = M.unmap(fwd, moves)
        if back != p:
            return False, "unmap(remap(%r)) == %r" % (p, back)
    return True, "remap and unmap invert on files, dirs and non-targets"


CHECKS = [check_01_dry_run_writes_nothing,
          check_02_move_preserves_every_byte,
          check_03_ledger_paths_remapped_ancestor_kept,
          check_04_phase_chain_remapped,
          check_05_missing_source_is_an_error,
          check_06_existing_target_is_an_error,
          check_07_byte_change_during_move_is_caught,
          check_08_verify_catches_stale_ledger_row,
          check_09_idempotent_second_apply,
          check_10_remap_unmap_are_inverse]


def main():
    width = max(len(c.__name__) for c in CHECKS)
    n_ok = 0
    for c in CHECKS:
        try:
            ok, msg = c()
        except Exception as exc:                       # noqa: BLE001
            ok, msg = False, "raised %s: %s" % (type(exc).__name__, exc)
        n_ok += bool(ok)
        print("%-*s  %s   %s" % (width, c.__name__, "PASS" if ok else "FAIL", msg))
    print("-" * (width + 40))
    print("%d/%d checks passed" % (n_ok, len(CHECKS)))
    return 0 if n_ok == len(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
