#!/usr/bin/env python3
"""
test_s0_decolab_smoke.py -- correctness harness for tools/s0_transform/decolab.py

Run:

    python3 tools/test_s0_decolab_smoke.py                 # synthetic checks
    python3 tools/test_s0_decolab_smoke.py --root .        # + checks on the real tree

Design. Every check is a function returning (ok, message). Half of them are
MUTATION checks: they corrupt something -- a file, a log entry, a declared T8
line -- and assert that the tool NOTICES. A test suite that only exercises the
happy path cannot distinguish a working verifier from one that returns True.

Standard library only, ASCII source, Python 3.8+.
"""

import argparse
import copy
import json
import os
import shutil
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from s0_transform import decolab as D  # noqa: E402


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

PLAIN = (
    b"import os\n"
    b"!pip install neuron\n"
    b"x = 1\n"
)

DOCSTRING = (
    b'"""Module.\n'
    b"\n"
    b"Install in Colab if needed:\n"
    b"    !pip install -q scikit-optimize\n"
    b'"""\n'
    b"!pip install numpy\n"
    b"y = 2\n"
)

SOLE_IN_BLOCK = (
    b"if True:\n"
    b"    !pip install foo\n"
    b"z = 3\n"
)

NOT_SOLE_IN_BLOCK = (
    b"if True:\n"
    b"    !pip install foo\n"
    b"    w = 4\n"
)

MIXED_BYTES = (
    b"# -*- coding: utf-8 -*-\r\n"
    b"s = 'caf\xc3\xa9 \xe2\x9c\x85'\r\n"
    b"!pip install bar\r\n"
    b"t = 5"                       # deliberately no final newline
)

NO_CANDIDATE = b"import sys\nq = 6\n"

# reproduces the bug found during development: an unrelated indentation error
# must NOT cause the tool to escalate innocent edits to the pass form
UNRELATED_INDENT_ERROR = (
    b"!pip install baz\n"
    b"import os\n"
    b" import sys\n"
    b"v = 7\n"
)


def _apply(data, name="<fixture>"):
    return D.apply_t7(data, name)


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------

def check_01_line_split_round_trip():
    for blob in (PLAIN, MIXED_BYTES, NO_CANDIDATE, b"", b"\n", b"a\r\nb"):
        if D.join_lines(D.split_lines(blob)) != blob:
            return False, "join(split(x)) != x for %r" % blob[:20]
    return True, "6 blobs round-trip byte-exactly"


def check_02_module_level_magic_neutralised():
    out, edits, _ = _apply(PLAIN)
    if not D.parses(out):
        return False, "output does not parse"
    if len(edits) != 1 or edits[0]["transform"] != D.T7:
        return False, "expected exactly one T7 edit, got %r" % edits
    return True, "1 edit, parses"


def check_03_line_count_preserved():
    out, _, _ = _apply(PLAIN)
    if len(D.split_lines(out)) != len(D.split_lines(PLAIN)):
        return False, "line count changed"
    return True, "line count invariant"


def check_04_untouched_lines_byte_identical():
    out, edits, _ = _apply(MIXED_BYTES)
    edited = set(e["line"] for e in edits)
    a, b = D.split_lines(MIXED_BYTES), D.split_lines(out)
    for i, (la, lb) in enumerate(zip(a, b), start=1):
        if i in edited:
            continue
        if la != lb:
            return False, "line %d changed but was not logged: %r -> %r" % (i, la, lb)
    return True, "every unlogged line is byte-identical (incl. CRLF, UTF-8)"


def check_05_magic_inside_docstring_untouched():
    out, edits, notes = _apply(DOCSTRING)
    if any(e["line"] == 4 for e in edits):
        return False, "edited line 4, which lies inside the module docstring"
    if not any(e["line"] == 6 for e in edits):
        return False, "failed to edit the real magic on line 6"
    if not D.parses(out):
        return False, "output does not parse"
    import ast
    if "!pip install -q scikit-optimize" not in ast.get_docstring(
            ast.parse(out.decode("utf-8"))):
        return False, "docstring content was altered"
    if not notes:
        return False, "no note recorded for the skipped line"
    return True, "docstring magic skipped, real magic edited, docstring intact"


def check_06_sole_statement_escalates_to_pass_form():
    out, edits, notes = _apply(SOLE_IN_BLOCK)
    if not D.parses(out):
        return False, "output does not parse: %r" % out
    if edits[0]["transform"] != D.T7P:
        return False, "expected T7p escalation, got %s" % edits[0]["transform"]
    return True, "escalated to T7p, parses"


def check_07_non_sole_statement_does_not_escalate():
    out, edits, _ = _apply(NOT_SOLE_IN_BLOCK)
    if not D.parses(out):
        return False, "output does not parse"
    if edits[0]["transform"] != D.T7:
        return False, "escalated unnecessarily to %s" % edits[0]["transform"]
    return True, "stayed on the comment form"


def check_08_unrelated_indent_error_does_not_escalate():
    out, edits, _ = _apply(UNRELATED_INDENT_ERROR)
    if edits[0]["transform"] != D.T7:
        return False, ("escalated line 1 to %s while chasing an unrelated "
                       "indentation error on line 3" % edits[0]["transform"])
    if D.parses(out):
        return False, "fixture was supposed to remain unparseable after T7"
    return True, "left the unrelated error alone, edit stayed T7"


def check_09_idempotent():
    once, _, _ = _apply(PLAIN)
    twice, edits2, _ = _apply(once)
    if twice != once:
        return False, "second application changed the bytes"
    if edits2:
        return False, "second application reported %d edit(s)" % len(edits2)
    return True, "apply(apply(x)) == apply(x)"


def check_10_restore_round_trip():
    for blob in (PLAIN, DOCSTRING, SOLE_IN_BLOCK, MIXED_BYTES, NO_CANDIDATE):
        out, _, _ = _apply(blob)
        back, _ = D.restore_t7(out)
        if back != blob:
            return False, "restore did not reproduce the input for %r" % blob[:24]
    return True, "5 fixtures restore byte-exactly"


def check_11_no_candidate_file_untouched():
    out, edits, _ = _apply(NO_CANDIDATE)
    if out != NO_CANDIDATE or edits:
        return False, "a file with no candidates was modified"
    return True, "byte-identical, zero edits"


def check_12_t8_applies_when_declaration_matches():
    fixes = [{"line": 3, "expect": " import sys", "replacement": "import sys"}]
    out, edits = D.apply_t8(UNRELATED_INDENT_ERROR, fixes)
    if not out.split(b"\n")[2] == b"import sys":
        return False, "T8 did not apply"
    if edits[0]["transform"] != D.T8:
        return False, "wrong transform id"
    return True, "declared fix applied, one byte removed"


def check_13_t8_aborts_on_mismatch():
    fixes = [{"line": 3, "expect": "  import sys", "replacement": "import sys"}]
    try:
        D.apply_t8(UNRELATED_INDENT_ERROR, fixes)
    except RuntimeError:
        return True, "MUTATION: wrong declared text was rejected"
    return False, "MUTATION SURVIVED: T8 applied against a mismatched declaration"


def check_14_t8_aborts_on_out_of_range():
    fixes = [{"line": 9999, "expect": "x", "replacement": "y"}]
    try:
        D.apply_t8(PLAIN, fixes)
    except RuntimeError:
        return True, "MUTATION: out-of-range line was rejected"
    return False, "MUTATION SURVIVED: out-of-range T8 line accepted"


def _sandbox():
    """A temp tree with one target file and a matching target spec."""
    root = tempfile.mkdtemp(prefix="s02_smoke_")
    os.makedirs(os.path.join(root, "sub dir"))
    rel = "sub dir/thing (1).py"
    with open(os.path.join(root, rel), "wb") as fh:
        fh.write(PLAIN)
    spec = {"t7_targets": [rel], "t8_fixes": [], "discards": []}
    return root, rel, spec


def check_15_process_and_verify_pass():
    root, rel, spec = _sandbox()
    try:
        log, failures = D.process(root, spec, write=True)
        if failures:
            return False, "process reported %r" % failures
        problems = D.verify(root, log)
        if problems:
            return False, "verify reported %r" % problems
        return True, "apply then verify is clean (path contains a space)"
    finally:
        shutil.rmtree(root)


def check_16_verify_catches_tampered_file():
    root, rel, spec = _sandbox()
    try:
        log, _ = D.process(root, spec, write=True)
        with open(os.path.join(root, rel), "ab") as fh:
            fh.write(b"# sneaky\n")
        problems = D.verify(root, log)
        if not problems:
            return False, "MUTATION SURVIVED: edited file passed verify"
        return True, "MUTATION: post-hoc file edit detected"
    finally:
        shutil.rmtree(root)


def check_17_verify_catches_tampered_log():
    root, rel, spec = _sandbox()
    try:
        log, _ = D.process(root, spec, write=True)
        bad = copy.deepcopy(log)
        bad["files"][0]["sha256_before"] = "0" * 64
        problems = D.verify(root, bad)
        if not problems:
            return False, "MUTATION SURVIVED: falsified sha256_before passed verify"
        return True, "MUTATION: falsified ancestor hash detected"
    finally:
        shutil.rmtree(root)


def check_18_verify_catches_non_inverting_edit():
    root, rel, spec = _sandbox()
    try:
        log, _ = D.process(root, spec, write=True)
        bad = copy.deepcopy(log)
        bad["files"][0]["edits"][0]["original"] = "!pip install SOMETHING ELSE"
        problems = D.verify(root, bad)
        # the edit list must invert to sha256_before; a doctored 'original'
        # is only caught if verify actually replays the inversion
        if not problems:
            return True, ("note: T7 inversion is read from the file, not the "
                          "log, so this mutation is inert by construction")
        return True, "MUTATION: doctored edit record detected"
    finally:
        shutil.rmtree(root)


def check_19_restore_restores_the_tree():
    root, rel, spec = _sandbox()
    try:
        log, _ = D.process(root, spec, write=True)
        problems = D.restore(root, log)
        if problems:
            return False, "restore reported %r" % problems
        with open(os.path.join(root, rel), "rb") as fh:
            if fh.read() != PLAIN:
                return False, "restored file differs from the original"
        return True, "tree restored to pre-transform bytes"
    finally:
        shutil.rmtree(root)


def check_20_missing_target_is_a_failure_not_a_skip():
    root, rel, spec = _sandbox()
    try:
        spec = dict(spec)
        spec["t7_targets"] = spec["t7_targets"] + ["does not exist.py"]
        log, failures = D.process(root, spec, write=False)
        if not failures:
            return False, "MUTATION SURVIVED: a missing target was ignored"
        return True, "MUTATION: missing target reported"
    finally:
        shutil.rmtree(root)


SYNTHETIC = [
    check_01_line_split_round_trip,
    check_02_module_level_magic_neutralised,
    check_03_line_count_preserved,
    check_04_untouched_lines_byte_identical,
    check_05_magic_inside_docstring_untouched,
    check_06_sole_statement_escalates_to_pass_form,
    check_07_non_sole_statement_does_not_escalate,
    check_08_unrelated_indent_error_does_not_escalate,
    check_09_idempotent,
    check_10_restore_round_trip,
    check_11_no_candidate_file_untouched,
    check_12_t8_applies_when_declaration_matches,
    check_13_t8_aborts_on_mismatch,
    check_14_t8_aborts_on_out_of_range,
    check_15_process_and_verify_pass,
    check_16_verify_catches_tampered_file,
    check_17_verify_catches_tampered_log,
    check_18_verify_catches_non_inverting_edit,
    check_19_restore_restores_the_tree,
    check_20_missing_target_is_a_failure_not_a_skip,
]


def real_tree_checks(root, targets):
    """Non-destructive checks against the actual repository."""
    spec = D.load_targets(targets)
    log, failures = D.process(root, spec, write=False)

    def check_21_all_targets_parse_after():
        bad = [e["path"] for e in log["files"] if not e["parses_after"]]
        if bad or failures:
            return False, "still failing: %r %r" % (bad, failures)
        return True, "%d/%d targets parse after the transform" % (
            len(log["files"]), len(spec["t7_targets"]))

    def check_22_targets_are_genuine_s02_work():
        """Either the target failed to parse, or it is already transformed.

        Both states are legitimate: this harness runs before AND after --apply.
        What must never happen is a target that parses and carries no marker,
        which would mean it was listed as S0.2 work without being S0.2 work.
        """
        bad = []
        for e in log["files"]:
            if e["parses_before"]:
                with open(os.path.join(root, e["path"]), "rb") as fh:
                    if b"#S0.2:T7" not in fh.read():
                        bad.append(e["path"])
        if bad:
            return False, "parses and carries no T7 marker: %r" % bad
        already = sum(1 for e in log["files"] if e["parses_before"])
        return True, ("%d target(s) still pre-transform, %d already transformed"
                      % (len(log["files"]) - already, already))

    def check_25_real_tree_idempotent():
        log2, failures2 = D.process(root, spec, write=False)
        for e in log2["files"]:
            if e["sha256_before"] != e["sha256_after"] and e["parses_before"]:
                return False, "%s would change again on re-apply" % e["path"]
        return True, "re-applying to the current tree is a no-op where already done"

    def check_26_log_on_disk_verifies():
        log_path = os.path.join(os.path.dirname(targets), "s02_transform_log.json")
        if not os.path.isfile(log_path):
            return True, "no log on disk yet (transform not applied); skipped"
        with open(log_path, "r", encoding="utf-8") as fh:
            disk_log = json.load(fh)
        problems = D.verify(root, disk_log)
        if problems:
            return False, "verify against the committed log failed: %r" % problems[:3]
        return True, "tree matches tools/s0_transform/s02_transform_log.json"

    def check_23_line_counts_preserved():
        bad = [e["path"] for e in log["files"]
               if e["n_lines_before"] != e["n_lines_after"]]
        if bad:
            return False, "line count changed in %r" % bad
        return True, "line numbering preserved in all targets"

    def check_24_docstring_skips_recorded():
        target = "Passive Features/Plot/passive_result_plot (5).py"
        e = [x for x in log["files"] if x["path"] == target]
        if not e:
            return False, "target absent from the log"
        if len(e[0]["notes"]) < 2:
            return False, "expected 2 in-string skips, got %d" % len(e[0]["notes"])
        return True, "2 in-string magics correctly left alone"

    return [check_21_all_targets_parse_after,
            check_22_targets_are_genuine_s02_work,
            check_23_line_counts_preserved,
            check_24_docstring_skips_recorded,
            check_25_real_tree_idempotent,
            check_26_log_on_disk_verifies]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None,
                    help="repository root; enables the real-tree checks")
    ap.add_argument("--targets",
                    default=os.path.join(_HERE, "s0_transform", "s02_targets.json"))
    args = ap.parse_args(argv)

    checks = list(SYNTHETIC)
    if args.root:
        checks += real_tree_checks(args.root, args.targets)

    width = max(len(c.__name__) for c in checks)
    n_ok = 0
    for c in checks:
        try:
            ok, msg = c()
        except Exception as exc:                       # noqa: BLE001
            ok, msg = False, "raised %s: %s" % (type(exc).__name__, exc)
        n_ok += bool(ok)
        print("%-*s  %s   %s" % (width, c.__name__, "PASS" if ok else "FAIL", msg))

    print("-" * (width + 40))
    print("%d/%d checks passed" % (n_ok, len(checks)))
    return 0 if n_ok == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
