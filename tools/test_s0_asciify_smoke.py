#!/usr/bin/env python3
"""
test_s0_asciify_smoke.py -- correctness harness for tools/s0_transform/asciify.py

    python3 tools/test_s0_asciify_smoke.py                  # synthetic
    python3 tools/test_s0_asciify_smoke.py --root .          # + the real tree

Standard library only, ASCII source, Python 3.8+. Non-ASCII fixtures are built
from escape sequences so that this file itself stays pure ASCII -- the rule the
sweep enforces applies to the tool that enforces it.
"""

import argparse
import base64
import json
import os
import shutil
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from s0_transform import asciify as A  # noqa: E402

MAP_PATH = os.path.join(_HERE, "s0_transform", "s03_translit_map.json")


def load_translit():
    with open(MAP_PATH, "r", encoding="utf-8") as fh:
        return A.Translit(json.load(fh))


def u(s):
    return s.encode("utf-8")


BANNER = u("# " + "\u2550" * 10 + "\n")
GREEK = u("sigma = \u03c3; mu = \u00b5m; Omega = \u03a9\n")
SUPER = u("z\u2080 = \u03a6\u207b\u00b9(x)\n")
EMOJI = u("print('\u2705 done \U0001f680 now')\n")
CRLF = b"a = 1\r\nb = 2\r\n"
MIXED = BANNER + GREEK + SUPER + EMOJI + b"plain = 0\n"


def check_01_round_trip_is_exact():
    t = load_translit()
    for blob in (BANNER, GREEK, SUPER, EMOJI, CRLF, MIXED):
        after, edits, unmapped = A.transform_file(blob, t)
        if unmapped:
            return False, "unmapped: %r" % unmapped
        entry = {"edits": edits}
        back = A.restore_bytes(after, entry)
        if back != blob:
            return False, "restore mismatch for %r" % blob[:30]
    return True, "6 fixtures restore byte-exactly from the log"


def check_02_output_is_pure_ascii_and_lf():
    t = load_translit()
    after, _, _ = A.transform_file(MIXED, t)
    A.assert_clean(after, "<fixture>")
    return True, "output passes the write-time postcondition"


def check_03_ordered_rules_beat_singles():
    t = load_translit()
    after, _, _ = A.transform_file(SUPER, t)
    text = after.decode("ascii")
    if "^-1" not in text:
        return False, "superscript-minus-one did not become ^-1: %r" % text
    if "^-^1" in text:
        return False, "ordered rule lost to the single rules: %r" % text
    return True, "U+207B U+00B9 -> ^-1, not ^-^1"


def check_04_crlf_normalised():
    t = load_translit()
    after, edits, _ = A.transform_file(CRLF, t)
    if b"\r\n" in after:
        return False, "CRLF survived"
    if not all("T10" in e["transforms"] for e in edits):
        return False, "T10 not recorded"
    return True, "CRLF -> LF, recorded as T10"


def check_05_line_count_preserved():
    t = load_translit()
    after, _, _ = A.transform_file(MIXED, t)
    if len(A.split_lines(after)) != len(A.split_lines(MIXED)):
        return False, "line count changed"
    return True, "line numbering preserved"


def check_06_idempotent():
    t = load_translit()
    once, _, _ = A.transform_file(MIXED, t)
    twice, edits2, _ = A.transform_file(once, t)
    if twice != once or edits2:
        return False, "second application changed %d line(s)" % len(edits2)
    return True, "apply(apply(x)) == apply(x)"


def check_07_unmapped_character_is_a_hard_failure():
    t = load_translit()
    exotic = u("x = '\u5b57'\n")            # CJK ideograph, deliberately absent
    _after, _edits, unmapped = A.transform_file(exotic, t)
    if not unmapped:
        return False, "MUTATION SURVIVED: an unmapped character passed silently"
    return True, "MUTATION: unmapped character reported, not passed through"


def check_08_non_ascii_replacement_is_rejected():
    doc = {"ordered_rules": [], "drop_decorative": [],
           "single_rules": [{"from": "\u2550", "to": "\u2014"}]}
    try:
        A.Translit(doc)
    except ValueError:
        return True, "MUTATION: a non-ASCII replacement was rejected at load"
    return False, "MUTATION SURVIVED: map with a non-ASCII replacement loaded"


def check_09_mapped_and_dropped_is_rejected():
    doc = {"ordered_rules": [], "single_rules": [{"from": "\u2705", "to": "[OK]"}],
           "drop_decorative": [{"from": "\u2705"}]}
    try:
        A.Translit(doc)
    except ValueError:
        return True, "MUTATION: contradictory map rejected"
    return False, "MUTATION SURVIVED: character both mapped and dropped"


def check_10_drop_list_deletes():
    t = load_translit()
    after, _, unmapped = A.transform_file(EMOJI, t)
    if unmapped:
        return False, "unmapped: %r" % unmapped
    text = after.decode("ascii")
    if "[OK]" not in text:
        return False, "status glyph did not become a token: %r" % text
    if "done" not in text or "now" not in text:
        return False, "surrounding text damaged: %r" % text
    return True, "status glyph -> [OK]; decorative emoji deleted"


def check_11_invalid_utf8_raises():
    t = load_translit()
    try:
        A.transform_file(b"x = '\xff\xfe'\n", t)
    except RuntimeError:
        return True, "MUTATION: non-UTF-8 input raised instead of being mangled"
    return False, "MUTATION SURVIVED: invalid UTF-8 processed silently"


def _sandbox():
    root = tempfile.mkdtemp(prefix="s03_")
    os.makedirs(os.path.join(root, "sub dir"))
    rel = "sub dir/thing (1).py"
    with open(os.path.join(root, rel), "wb") as fh:
        fh.write(MIXED)
    with open(os.path.join(root, "keep.zip"), "wb") as fh:
        fh.write(b"PK\x03\x04\xe2\x94\x80\r\n")
    import subprocess
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "f"], cwd=root, check=True)
    spec = {"include_globs": ["*.py"],
            "exclude_globs": ["*.zip"], "exclude_paths": []}
    return root, rel, spec


def check_12_binaries_are_never_selected():
    root, rel, spec = _sandbox()
    try:
        chosen = A.select_targets(root, spec)
        if any(p.endswith(".zip") for p in chosen):
            return False, "MUTATION SURVIVED: a binary was selected"
        if rel not in chosen:
            return False, "the .py target was not selected"
        return True, "binary excluded, .py selected (path contains a space)"
    finally:
        shutil.rmtree(root)


def check_13_verify_catches_tampered_file():
    root, rel, spec = _sandbox()
    try:
        t = load_translit()
        log, failures = A.process(root, spec, t, write=True)
        if failures:
            return False, "%r" % failures
        with open(os.path.join(root, rel), "ab") as fh:
            fh.write(b"# extra\n")
        if not A.verify(root, log):
            return False, "MUTATION SURVIVED: tampered file passed verify"
        return True, "MUTATION: post-hoc edit detected"
    finally:
        shutil.rmtree(root)


def check_14_verify_catches_tampered_log():
    root, rel, spec = _sandbox()
    try:
        t = load_translit()
        log, _ = A.process(root, spec, t, write=True)
        bad = json.loads(json.dumps(log))
        e = bad["files"][0]["edits"][0]
        e["original_b64"] = base64.b64encode(b"# lie\n").decode("ascii")
        if not A.verify(root, bad):
            return False, "MUTATION SURVIVED: doctored log passed verify"
        return True, "MUTATION: doctored edit record detected by inversion"
    finally:
        shutil.rmtree(root)


def check_15_restore_restores_the_tree():
    root, rel, spec = _sandbox()
    try:
        t = load_translit()
        log, _ = A.process(root, spec, t, write=True)
        if A.restore(root, log):
            return False, "restore reported problems"
        with open(os.path.join(root, rel), "rb") as fh:
            if fh.read() != MIXED:
                return False, "restored file differs from the original"
        return True, "tree restored to pre-sweep bytes"
    finally:
        shutil.rmtree(root)


SYNTHETIC = [check_01_round_trip_is_exact,
             check_02_output_is_pure_ascii_and_lf,
             check_03_ordered_rules_beat_singles,
             check_04_crlf_normalised,
             check_05_line_count_preserved,
             check_06_idempotent,
             check_07_unmapped_character_is_a_hard_failure,
             check_08_non_ascii_replacement_is_rejected,
             check_09_mapped_and_dropped_is_rejected,
             check_10_drop_list_deletes,
             check_11_invalid_utf8_raises,
             check_12_binaries_are_never_selected,
             check_13_verify_catches_tampered_file,
             check_14_verify_catches_tampered_log,
             check_15_restore_restores_the_tree]


def real_tree(root):
    t = load_translit()
    with open(os.path.join(root, "tools", "s0_transform", "s03_targets.json"),
              "r", encoding="utf-8") as fh:
        spec = json.load(fh)
    log, failures = A.process(root, spec, t, write=False)

    def check_16_map_is_exhaustive_over_the_target_set():
        if failures:
            return False, "%d failure(s), first: %s" % (len(failures), failures[0])
        return True, "no unmapped character in %d file(s)" % len(log["files"])

    def check_17_every_target_would_end_clean():
        return True, ("%d file(s), %d line(s), %d non-ASCII byte(s), %d CRLF line(s)"
                      % (len(log["files"]),
                         sum(e["n_edits"] for e in log["files"]),
                         sum(e["n_nonascii_before"] for e in log["files"]),
                         sum(e["n_crlf_before"] for e in log["files"])))

    def check_18_tools_are_already_clean():
        dirty = [e["path"] for e in log["files"] if e["path"].startswith("tools/")]
        if dirty:
            return False, "S0 tooling is not ASCII/LF clean: %r" % dirty
        return True, "the S0 tooling needs no sweeping; it was written clean"

    return [check_16_map_is_exhaustive_over_the_target_set,
            check_17_every_target_would_end_clean,
            check_18_tools_are_already_clean]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    args = ap.parse_args(argv)
    checks = list(SYNTHETIC) + (real_tree(args.root) if args.root else [])
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
