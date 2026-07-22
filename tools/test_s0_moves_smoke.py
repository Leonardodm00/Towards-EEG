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
                 "sha256_post_s04", "sha256_post_s05", "sha256_post_s06",
                 "sha256_post_s07", "sha256_post_s08",
                 "size_bytes", "is_python", "parses",
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
                    "sha256_post_s05": "-", "sha256_post_s06": "-",
                    "sha256_post_s07": "-", "sha256_post_s08": "-",
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


def check_11_ledger_rewrite_stays_lf_and_ascii():
    """Regression: a bare csv.DictWriter defaults to CRLF and rewrote the
    whole ledger once. The canonical writer must not, and must refuse
    non-ASCII."""
    root, spec, _ = sandbox()
    try:
        M.do_moves(root, spec, write=True)
        M.rewrite_ledger(root, spec["moves"], write=True)
        data = open(os.path.join(root, "ledger.csv"), "rb").read()
        if b"\r\n" in data:
            return False, "MUTATION SURVIVED: rewrite introduced CRLF"
        if any(b > 127 for b in data):
            return False, "rewrite introduced non-ASCII bytes"
        return True, "ledger rewritten LF-only and pure ASCII"
    finally:
        shutil.rmtree(root)


def check_12_move_into_a_nonexistent_parent():
    """S0.2c moved a directory to a sibling, so the destination parent always
    existed. S0.4 moves into towards_eeg/, which does not. Without a mkdir,
    `git mv` fails with 'destination directory does not exist'."""
    root, spec, before = sandbox()
    try:
        spec = {"moves": [{"from": "old dir /a.py",
                           "to": "towards_eeg/hybrid/population.py",
                           "kind": "file", "transform_id": "T6",
                           "stage": "S0.4", "rationale": "fixture"}]}
        M.do_moves(root, spec, write=True)
        dst = os.path.join(root, "towards_eeg", "hybrid", "population.py")
        if not os.path.isfile(dst):
            return False, "file did not land at the nested destination"
        if M.sha256_file(dst) != before["old dir /a.py"]:
            return False, "bytes changed across a move into a new parent"
        return True, "move into a two-level new parent works, bytes identical"
    finally:
        shutil.rmtree(root)


def check_13_rationale_names_the_moves_own_stage():
    """MUTATION of the old behaviour: the rationale was a hardcoded 'at S0.2c'
    regardless of the move. Applied at S0.4 it mislabels every row."""
    root, spec, _ = sandbox()
    try:
        spec["moves"][0]["stage"] = "S0.4"
        M.do_moves(root, spec, write=True)
        M.rewrite_ledger(root, spec["moves"], write=True)
        rows = list(csv.DictReader(open(os.path.join(root, "ledger.csv"))))
        for r in rows:
            if "at S0.4" not in r["rationale"]:
                return False, ("MUTATION SURVIVED: rationale does not name the "
                               "move's stage: %r" % r["rationale"])
            if "S0.2c" in r["rationale"]:
                return False, "hardcoded S0.2c still present in the rationale"
        return True, "rationale carries the stage declared on the move"
    finally:
        shutil.rmtree(root)


def check_14_validator_rejects_malformed_move_lists():
    """Four ways a multi-move list goes silently wrong. Each must be refused
    before anything is written."""
    base = {"from": "a", "to": "b", "kind": "directory",
            "transform_id": "T6", "stage": "S0.4", "rationale": "x"}

    def mk(*moves):
        return {"moves": [dict(base, **m) for m in moves]}

    cases = [
        ("duplicate source", mk({"from": "a", "to": "b"}, {"from": "a", "to": "c"})),
        ("colliding target", mk({"from": "a", "to": "z"}, {"from": "b", "to": "z"})),
        ("nested source", mk({"from": "a", "to": "b"},
                             {"from": "a/inner", "to": "c", "kind": "file"})),
        ("chained move", mk({"from": "a", "to": "b"}, {"from": "b", "to": "c"})),
        ("no-op", mk({"from": "a", "to": "a"})),
        ("wrong transform", mk({"from": "a", "to": "b", "transform_id": "T9"})),
        ("missing key", {"moves": [{"from": "a", "to": "b", "kind": "file"}]}),
    ]
    survived = [name for name, spec in cases if not M.validate_moves(spec)]
    if survived:
        return False, "MUTATION SURVIVED: validator accepted %r" % survived
    if M.validate_moves(mk({"from": "a", "to": "b"}, {"from": "c", "to": "d"})):
        return False, "validator rejected a well-formed list"
    return True, "%d malformed list(s) refused, well-formed list accepted" % len(cases)


def check_15_real_move_list_is_well_formed_and_reachable():
    """The declared list in the repository, not a fixture. Every source must
    either still exist or have already landed at its target -- a typo in a
    path containing spaces or parentheses is otherwise invisible until
    --apply is halfway through."""
    root = os.path.dirname(_HERE)
    spec_path = os.path.join(root, "tools", "path_moves.json")
    if not os.path.isfile(spec_path):
        return False, "tools/path_moves.json is absent"
    with open(spec_path, "r", encoding="utf-8") as fh:
        spec = json.load(fh)
    problems = M.validate_moves(spec)
    if problems:
        return False, "declared list is malformed: %r" % problems[:3]
    unreachable = []
    for mv in spec["moves"]:
        src = os.path.join(root, mv["from"])
        dst = os.path.join(root, mv["to"])
        if not os.path.exists(src) and not os.path.exists(dst):
            unreachable.append(mv["from"])
    if unreachable:
        return False, ("declared source neither present nor already moved: %r"
                       % unreachable)
    return True, ("%d declared move(s), all well formed and reachable"
                  % len(spec["moves"]))


CHECKS = [check_01_dry_run_writes_nothing,
          check_02_move_preserves_every_byte,
          check_03_ledger_paths_remapped_ancestor_kept,
          check_04_phase_chain_remapped,
          check_05_missing_source_is_an_error,
          check_06_existing_target_is_an_error,
          check_07_byte_change_during_move_is_caught,
          check_08_verify_catches_stale_ledger_row,
          check_09_idempotent_second_apply,
          check_10_remap_unmap_are_inverse,
          check_11_ledger_rewrite_stays_lf_and_ascii,
          check_12_move_into_a_nonexistent_parent,
          check_13_rationale_names_the_moves_own_stage,
          check_14_validator_rejects_malformed_move_lists,
          check_15_real_move_list_is_well_formed_and_reachable]


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
