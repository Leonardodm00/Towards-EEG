#!/usr/bin/env python3
"""
test_s0_5_exit.py -- exit test for stage S0.5 (contract schemas pinned into io/).

    python3 tools/test_s0_5_exit.py --root .

WHAT S0.5 CLAIMS
----------------
The C-08, C-09, C-14 and C-15 schemas of TEEG_02 revision 3 are pinned as data
in towards_eeg/io/schemas/; six invariants are registered; validate_instance()
runs them and skips gracefully, and visibly, when the data is absent. Nothing
is populated: no bank is read, no instance is built, no morphology is touched.

WHAT THIS FILE ASSERTS, AND WHAT IT DELIBERATELY DOES NOT
---------------------------------------------------------
That the pinned schemas are present, internally consistent and match the
declared scope; that the register holds exactly the six declared invariants
with exactly the declared clauses; that io/ has not acquired the ability to
read data; and that every declaration in s05_exit_scope.json still applies.

It does NOT re-assert that the invariants are correct. That is fourteen
clauses in both directions and it lives in tools/test_s0_5_smoke.py, because
an exit test that also carried its subject's unit tests would have to be
rewritten whenever either changed.

It also does not assert that any invariant PASSES on real data. There is no
real data. An exit test that required one would be permanently red, which is
the state Doc 6 section 4 warns against.

Standard library only, ASCII source, Python 3.8+.
"""

import argparse
import ast
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from towards_eeg.io import invariants as INV       # noqa: E402
from towards_eeg.io import schema as SCH           # noqa: E402
from towards_eeg.io import validate as VAL         # noqa: E402


def io_source_files(root):
    d = os.path.join(root, "towards_eeg", "io")
    return sorted(os.path.join("towards_eeg", "io", f)
                  for f in os.listdir(d) if f.endswith(".py"))


def run(root):
    with open(os.path.join(root, "tools", "s0_transform", "s05_exit_scope.json"),
              "r", encoding="utf-8") as fh:
        scope = json.load(fh)
    results = []

    def add(name, ok, msg):
        results.append((name, ok, msg))

    # -- 1: the pinned files exist and are ASCII/LF --------------------------
    problems = []
    for rel in scope["schema_files"]:
        full = os.path.join(root, rel)
        if not os.path.isfile(full):
            problems.append("missing: %s" % rel)
            continue
        data = open(full, "rb").read()
        if any(b > 127 for b in data):
            problems.append("non-ASCII bytes in %s" % rel)
        if b"\r\n" in data:
            problems.append("CRLF in %s" % rel)
        try:
            json.loads(data.decode("ascii"))
        except Exception as exc:                            # noqa: BLE001
            problems.append("%s does not parse: %s" % (rel, exc))
    add("check_01_pinned_schema_files_present_and_clean", not problems,
        "%d schema file(s) present, pure ASCII, LF-only, valid JSON"
        % len(scope["schema_files"]) if not problems else "; ".join(problems[:4]))

    # -- 2: the loader agrees with the declared contract list ----------------
    declared = tuple(scope["pinned_contracts"])
    ok = SCH.SCHEMA_IDS == declared
    add("check_02_loader_pins_the_declared_contracts", ok,
        "loader pins %r" % (declared,) if ok
        else "loader pins %r, scope declares %r" % (SCH.SCHEMA_IDS, declared))

    # -- 3: the schemas are internally consistent ----------------------------
    problems = SCH.check_schema_self_consistency()
    add("check_03_schemas_are_self_consistent", not problems,
        "C8.1's biconditional holds, get_idx substrings occur in their section "
        "arrays, Sigma_syn does not leak, every declared enum exists, C-14 and "
        "C-15 are keyed identically, f_implied is qc_only"
        if not problems else "; ".join(problems[:3]))

    # -- 4: the register is exactly the declared six -------------------------
    declared = tuple(scope["registered_invariants"])
    ok = INV.INVARIANT_IDS == declared
    add("check_04_register_is_the_declared_six", ok,
        "%r registered; I-19 absent, and it does not exist" % (declared,) if ok
        else "register is %r, scope declares %r" % (INV.INVARIANT_IDS, declared))

    # -- 5: the clause list is pinned ----------------------------------------
    # A conjunct silently disappearing would shrink an invariant without
    # changing its status. Trap T-11 at the level of the invariant itself.
    rep = VAL.validate_instance({})
    got = dict((r.iid, [c.clause for c in r.clauses]) for r in rep.results)
    problems, n = [], 0
    for iid, clauses in scope["clauses"].items():
        n += len(clauses)
        if got.get(iid) != clauses:
            problems.append("%s has clauses %r, scope declares %r"
                            % (iid, got.get(iid), clauses))
    if n != scope["expected_clause_count"]:
        problems.append("scope lists %d clauses but declares %d"
                        % (n, scope["expected_clause_count"]))
    add("check_05_clauses_match_the_declared_list", not problems,
        "%d clause(s) across %d invariant(s), exactly as declared"
        % (n, len(scope["clauses"])) if not problems else "; ".join(problems[:3]))

    # -- 6: absent data skips, visibly, and is not ok ------------------------
    problems = []
    if rep.ok:
        problems.append("an empty instance reports ok=True")
    if len(rep.skipped) != len(INV.INVARIANT_IDS):
        problems.append("not every invariant skipped: %r" % rep.skipped)
    if rep.checked:
        problems.append("clauses ran on an empty instance: %r" % rep.checked)
    silent = [(r.iid, c.clause) for r in rep.results for c in r.clauses
              if c.status == INV.SKIPPED and not c.reason]
    if silent:
        problems.append("skips with no stated reason: %r" % silent)
    add("check_06_absent_data_skips_visibly_and_is_not_ok", not problems,
        "all %d invariant(s) skip with a stated reason and ok=False; "
        "TEEG_00 section 4's 'skip gracefully' does not mean 'skip quietly'"
        % len(INV.INVARIANT_IDS) if not problems else "; ".join(problems[:3]))

    # -- 7: S0 populates nothing --------------------------------------------
    # The scientific libraries are the cheapest way for schema code to acquire
    # the ability to read data without anyone noticing.
    forbidden = set(scope["s0_populates_nothing"]["forbidden_tokens"])
    problems = []
    for rel in io_source_files(root):
        src = open(os.path.join(root, rel), "r", encoding="utf-8").read()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                mods = [node.module]
            for m in mods:
                if m.split(".")[0] in forbidden:
                    problems.append("%s imports %r" % (rel, m))
    add("check_07_io_cannot_read_data", not problems,
        "no module under towards_eeg/io/ imports %s; S0 populates nothing"
        % ", ".join(sorted(forbidden)[:4]) + ", ..."
        if not problems else "; ".join(problems[:4]))

    # -- 8: no configuration path literal leaked into io/ --------------------
    forbidden = scope["package_relative_resource_access"]["forbidden_in_io"]
    problems = []
    for rel in io_source_files(root):
        src = open(os.path.join(root, rel), "r", encoding="utf-8").read()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            val = getattr(node, "value", None)
            if isinstance(node, ast.Constant) and isinstance(val, str):
                for bad in forbidden:
                    if bad in val:
                        problems.append("%s contains %r in a string literal"
                                        % (rel, bad))
    add("check_08_no_configuration_path_literal_in_io", not problems,
        "io/ contains no absolute or drive path literal; the only path it "
        "builds is package-relative, and that is declared"
        if not problems else "; ".join(problems[:4]))

    # -- 9: the declared not-implemented items are still not implemented -----
    # Self-verifying in the S0.4 pattern: a stale declaration is an error.
    problems = []
    for entry in load_registry_entries():
        if entry.get("implemented", False):
            problems.append("C-15 callable %r is marked implemented; its body "
                            "is deferred to S3" % entry["name"])
        for key in ("form", "constants", "applies_to"):
            if key not in entry:
                problems.append("C-15 callable %r lacks %r, so S3 would have "
                                "to re-derive it" % (entry["name"], key))
    if os.path.isdir(os.path.join(root, "towards_eeg", "structure")):
        stray = [f for f in os.listdir(os.path.join(root, "towards_eeg", "structure"))
                 if f.endswith(".py") and f != "__init__.py"]
        if stray:
            problems.append("towards_eeg/structure/ has acquired %r; the .hoc "
                            "reader is S1's" % stray)
    add("check_09_deferred_items_are_still_deferred", not problems,
        "%d deferred item(s) still deferred, each with its constants pinned "
        "so the assigned stage implements against a declaration"
        % len(scope["not_implemented_by_decision"])
        if not problems else "; ".join(problems[:3]))

    # -- 10: the geometry record has not grown ------------------------------
    from towards_eeg.io import geometry as GEO
    problems = []
    expected_slots = ("sid", "name", "diameter_um", "position_um", "parent")
    if tuple(GEO.Section.__slots__) != expected_slots:
        problems.append("Section carries %r, declared %r"
                        % (tuple(GEO.Section.__slots__), expected_slots))
    for word in scope["geometry_record"]["must_not_grow_to_represent"]:
        src = open(os.path.join(root, "towards_eeg", "io", "geometry.py"),
                   "r", encoding="utf-8").read()
        tree = ast.parse(src)
        names = [n.name for n in ast.walk(tree)
                 if isinstance(n, (ast.ClassDef, ast.FunctionDef))]
        hits = [n for n in names if word.rstrip("s") in n.lower()]
        if hits:
            problems.append("geometry.py has grown %r, which represents %r"
                            % (hits, word))
    ratio = scope["geometry_record"]["somatic_calibre_ratio_default"]
    if GEO.SOMATIC_CALIBRE_RATIO != ratio:
        problems.append("somatic calibre ratio is %r, declared %r"
                        % (GEO.SOMATIC_CALIBRE_RATIO, ratio))
    add("check_10_geometry_record_is_still_minimal", not problems,
        "Section carries exactly the five fields I-15 and I-18 read; the "
        "calibre ratio is the declared %g" % ratio
        if not problems else "; ".join(problems[:3]))

    # -- 11: the S0.4 declaration was updated, not bypassed ------------------
    with open(os.path.join(root, "tools", "s0_transform", "s04_exit_scope.json"),
              "r", encoding="utf-8") as fh:
        s04 = json.load(fh)
    empty = s04["empty_by_construction"]["packages"]
    problems = []
    if "towards_eeg/io" in empty:
        problems.append("s04_exit_scope.json still declares towards_eeg/io "
                        "empty by construction; S0.5 populated it")
    for pkg in ("towards_eeg/structure", "towards_eeg/pointnet",
                "towards_eeg/cosim", "towards_eeg/passive", "towards_eeg/config"):
        if pkg not in empty:
            problems.append("%s was dropped from empty_by_construction without "
                            "being populated" % pkg)
    add("check_11_s04_empty_declaration_was_updated", not problems,
        "towards_eeg/io removed from the S0.4 empty-package declaration, the "
        "other %d untouched" % len(empty)
        if not problems else "; ".join(problems[:3]))

    # -- 12: the schemas are installed, not merely present -------------------
    problems = []
    pyproj = os.path.join(root, "pyproject.toml")
    body = open(pyproj, "r", encoding="utf-8").read()
    if '"towards_eeg.io" = ["schemas/*.json"]' not in body:
        problems.append("pyproject.toml does not declare towards_eeg.io "
                        "package-data; a wheel install would ship an io/ that "
                        "cannot find its own schemas")
    add("check_12_schemas_are_declared_package_data", not problems,
        "schemas/*.json is declared package data, so a non-editable install "
        "carries it" if not problems else "; ".join(problems))

    # -- 13: known gaps are still declared -----------------------------------
    problems = []
    gaps = scope.get("known_gaps", [])
    if not gaps:
        problems.append("no known gaps declared; S0.5 has at least one "
                        "(provenance.json)")
    for g in gaps:
        for key in ("gap", "detail", "assigned_to"):
            if key not in g:
                problems.append("a known gap lacks %r" % key)
    add("check_13_known_gaps_are_declared_with_an_owner", not problems,
        "%d known gap(s), each with a detail and an assigned stage" % len(gaps)
        if not problems else "; ".join(problems[:3]))

    return results


def load_registry_entries():
    return SCH.load_schema("C-15")["callable_registry"]


def main(argv=None):
    ap = argparse.ArgumentParser(description="S0.5 exit test (contract schemas).")
    ap.add_argument("--root", default=".")
    args = ap.parse_args(argv)
    results = run(args.root)
    width = max(len(n) for n, _, _ in results)
    n_ok = sum(1 for _, ok, _ in results if ok)
    for name, ok, msg in results:
        print("%-*s  %s   %s" % (width, name, "PASS" if ok else "FAIL", msg))
    print("-" * (width + 40))
    print("%d/%d checks passed" % (n_ok, len(results)))
    print("\nS0.5 EXIT TEST: %s" % ("PASS" if n_ok == len(results) else "FAIL"))
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
