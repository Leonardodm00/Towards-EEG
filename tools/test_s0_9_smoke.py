#!/usr/bin/env python3
"""
test_s0_9_smoke.py -- correctness harness for tests/test_s0_import_surface.py.

    python3 tools/test_s0_9_smoke.py --root .

Doc 6 s3.3: a verifier that has never been made to fail has not been tested.
Each case copies what the exit test reads into a throwaway tree, breaks exactly
one thing, and asserts the named check goes red -- and, where the break is
isolated, that the OTHER checks stay green, so a mutation that reddens
everything is not mistaken for a targeted one.

THE MUTATION THAT MATTERS MOST
------------------------------
m_o10_declaration_goes_stale binds the unbound module-scope name in
usage_example.py, so that file would now import. Nothing else changes. The
naive reading is "good news, a module became importable"; the correct reading
is that the O10 exemption (N-23) is now describing a file that no longer
matches it, and assertion 3 must go red so the assignment is revisited rather
than silently rotting. This is the exact analogue of the S0.8 overclaim guard.

m_scope_file_unreadable is the discipline check: it corrupts
s09_exit_scope.json and asserts the exit test still returns eight verdicts
(the scope-dependent ones red) instead of raising -- an exit test that crashes
on bad input zeroes the verdict on every other check (TEEG_12 s3.3).

Standard library only, ASCII source, Python 3.8+.
"""

import argparse
import ast
import os
import shutil
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_TESTS = os.path.join(_ROOT, "tests")
for _p in (_HERE, _TESTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import test_s0_import_surface as E            # noqa: E402

GEO = os.path.join("towards_eeg", "io", "geometry.py")
INV = os.path.join("towards_eeg", "io", "invariants.py")
USAGE = os.path.join("towards_eeg", "connectome", "usage_example.py")
SCOPE = os.path.join("tools", "s0_transform", "s09_exit_scope.json")


def fixture(root):
    """A throwaway copy of everything the exit test reads."""
    tmp = tempfile.mkdtemp(prefix="s09_")
    for name in ("tools", "towards_eeg", "requirements", "tests"):
        src = os.path.join(root, name)
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(tmp, name),
                            ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("pyproject.toml", "ledger.csv", "LEDGER.md"):
        src = os.path.join(root, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(tmp, name))
    return tmp


def verdicts(tmp):
    # E.run must be re-run against the tmp tree, not a cached root.
    E._CACHE.pop(os.path.abspath(tmp), None)
    return dict((n, ok) for n, ok, _ in E.run(tmp))


def _append_text(tmp, rel, text):
    with open(os.path.join(tmp, rel), "a", encoding="ascii") as fh:
        fh.write(text)


def _append_bytes(tmp, rel, data):
    with open(os.path.join(tmp, rel), "ab") as fh:
        fh.write(data)


# ---------------------------------------------------------------------------
# mutations: (name, target_check, mutate_fn, also_green)
# also_green lists checks that MUST stay green (the break is isolated); empty
# where the break legitimately cascades.
# ---------------------------------------------------------------------------

def m_parse(tmp):
    _append_text(tmp, GEO, "\nx = (\n")             # unbalanced -> SyntaxError


def m_compile(tmp):
    _append_text(tmp, GEO, "\nreturn 1\n")          # parses, fails to compile


def m_import_internal(tmp):
    # ModuleNotFoundError whose top-level IS towards_eeg -> not excusable.
    with open(os.path.join(tmp, INV), "r", encoding="ascii") as fh:
        body = fh.read()
    with open(os.path.join(tmp, INV), "w", encoding="ascii") as fh:
        fh.write("from towards_eeg.nonexistent_submodule import x\n" + body)


def m_o10_stale(tmp):
    # Bind the unbound module-scope name -> usage_example.py would now import
    # -> the O10 exemption is stale -> assertion 3 must go red.
    with open(os.path.join(tmp, USAGE), "r", encoding="ascii") as fh:
        body = fh.read()
    with open(os.path.join(tmp, USAGE), "w", encoding="ascii") as fh:
        fh.write("column_input = None\n" + body)


def m_ascii(tmp):
    _append_bytes(tmp, GEO, b"\n# \xe9 non-ascii\n")   # 0xE9, no cookie


def m_path_literal(tmp):
    _append_text(tmp, GEO, '\nLEAK = "/home/user/secret/data.txt"\n')


def m_duplicate_class(tmp):
    _append_text(tmp, GEO, "\nclass ZDupCheck:\n    pass\n")
    _append_text(tmp, INV, "\nclass ZDupCheck:\n    pass\n")


def m_shadowed_def(tmp):
    _append_text(tmp, GEO, "\ndef _z_shadow():\n    return 1\n"
                           "\ndef _z_shadow():\n    return 2\n")


def m_reconstruction(tmp):
    _append_text(tmp, GEO, "\n# byte-changing comment, ledger not updated\n")


def m_scope_unreadable(tmp):
    with open(os.path.join(tmp, SCOPE), "w", encoding="ascii") as fh:
        fh.write("{ this is not valid json")


CASES = [
    ("m_parse",              "check_1_parse",                          m_parse,           []),
    ("m_compile",            "check_2_compile",                        m_compile,         ["check_1_parse"]),
    ("m_import_internal",    "check_3_import",                         m_import_internal, ["check_1_parse", "check_2_compile"]),
    ("m_o10_stale",          "check_3_import",                         m_o10_stale,       ["check_1_parse", "check_2_compile"]),
    ("m_ascii",              "check_4_ascii",                          m_ascii,           ["check_1_parse", "check_2_compile"]),
    ("m_path_literal",       "check_5_no_path_literal_outside_config", m_path_literal,    ["check_1_parse", "check_4_ascii"]),
    ("m_duplicate_class",    "check_6_no_duplicate_top_level_class",   m_duplicate_class, ["check_1_parse", "check_7_no_shadowed_module_level_def"]),
    ("m_shadowed_def",       "check_7_no_shadowed_module_level_def",   m_shadowed_def,    ["check_1_parse", "check_6_no_duplicate_top_level_class"]),
    ("m_reconstruction",     "check_8_ledger_reconstruction",          m_reconstruction,  ["check_1_parse", "check_2_compile", "check_4_ascii"]),
]


def run(root):
    results = []

    # sanity: the unmutated fixture must be fully green, or a red result below
    # proves nothing.
    base = fixture(root)
    try:
        base_v = verdicts(base)
    finally:
        shutil.rmtree(base, ignore_errors=True)
    base_ok = all(base_v.values())
    results.append(("baseline_all_green", base_ok,
                    "unmutated fixture is fully green"
                    if base_ok else "unmutated fixture already red: %r"
                    % [n for n, ok in base_v.items() if not ok]))

    for name, target, fn, also_green in CASES:
        tmp = fixture(root)
        try:
            fn(tmp)
            v = verdicts(tmp)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        target_red = target in v and not v[target]
        stray = [c for c in also_green if not v.get(c, False)]
        ok = target_red and not stray
        if not target_red:
            msg = "%s did NOT redden %s" % (name, target)
        elif stray:
            msg = "%s reddened checks it should not: %r" % (name, stray)
        else:
            msg = "%s -> %s red, isolated" % (name, target)
        results.append(("mutation_" + name, ok, msg))

    # discipline: corrupting the declaration must degrade to FAIL, not raise.
    tmp = fixture(root)
    try:
        m_scope_unreadable(tmp)
        try:
            v = verdicts(tmp)
            raised = False
        except BaseException as exc:                        # noqa: BLE001
            v, raised = {}, True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    degraded = (not raised and len(v) == 8
                and not v.get("check_3_import", True)
                and not v.get("check_5_no_path_literal_outside_config", True))
    results.append(("mutation_m_scope_unreadable_degrades_to_fail", degraded,
                    "corrupt declaration -> 8 verdicts, scope-dependent checks "
                    "red, no exception raised"
                    if degraded else "did NOT degrade cleanly (raised=%s, "
                    "n=%d)" % (raised, len(v))))

    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description="Mutation harness for the S0.9 exit test.")
    ap.add_argument("--root", default=".")
    args = ap.parse_args(argv)
    results = run(args.root)
    width = max(len(n) for n, _, _ in results)
    n_ok = sum(1 for _, ok, _ in results if ok)
    for name, ok, msg in results:
        print("%-*s  %s   %s" % (width, name, "PASS" if ok else "FAIL", msg))
    print("-" * (width + 30))
    print("%d/%d mutations behaved as required" % (n_ok, len(results)))
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
