#!/usr/bin/env python3
"""
test_s0_4_smoke.py -- correctness harness for tools/test_s0_4_exit.py.

    python3 tools/test_s0_4_smoke.py --root .

WHY THIS EXISTS
---------------
The project's standard, stated in Doc 6 s3.3: a verifier that has never been
made to fail has not been tested. test_s0_4_exit.py is twelve checks of new
logic, and twelve checks that always return PASS are indistinguishable from
twelve checks that are correct.

Every case below builds a THROWAWAY copy of only what the exit test reads --
the package tree, tools/, ledger.csv and the three retained forks -- breaks
exactly one thing, and asserts that the named check goes red. Nothing here
touches the real tree.

The fixture is deliberately a copy rather than a synthetic tree: a synthetic
fixture would test the checks against a world the checks were written for,
which proves less than breaking the real one.

Standard library only, ASCII source, Python 3.8+.
"""

import argparse
import csv
import json
import os
import shutil
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import test_s0_4_exit as E  # noqa: E402

FORKS = ["Population/Population_multiMorph.py",
         "Population/README.md",
         "Parames evoked with EEG/Params_evoked_with_EEG_multiMorph.py"]


def fixture(root):
    """A minimal copy of everything test_s0_4_exit.py reads."""
    tmp = tempfile.mkdtemp(prefix="s04_")
    shutil.copytree(os.path.join(root, "towards_eeg"),
                    os.path.join(tmp, "towards_eeg"))
    os.makedirs(os.path.join(tmp, "tools", "s0_transform"))
    for rel in ("tools/path_moves.json",
                "tools/ancestors.json",
                "tools/s0_transform/s04_exit_scope.json",
                "tools/s0_paths.py",
                "tools/apply_moves.py",
                "pyproject.toml",
                "ledger.csv"):
        dst = os.path.join(tmp, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(os.path.join(root, rel), dst)
    shutil.copytree(os.path.join(root, "tools", "s0_ledger"),
                    os.path.join(tmp, "tools", "s0_ledger"))
    for rel in FORKS:
        dst = os.path.join(tmp, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(os.path.join(root, rel), dst)
    return tmp


def verdicts(tmp):
    """{check_name: ok} from one run of the exit test against the fixture."""
    return dict((name, ok) for name, ok, _ in E.run(tmp))


def edit_scope(tmp, fn):
    p = os.path.join(tmp, "tools", "s0_transform", "s04_exit_scope.json")
    with open(p, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    fn(doc)
    with open(p, "w", encoding="ascii", newline="\n") as fh:
        json.dump(doc, fh, indent=2)
        fh.write("\n")


def edit_ledger(tmp, fn):
    p = os.path.join(tmp, "ledger.csv")
    with open(p, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        fields = reader.fieldnames
    fn(rows)
    with open(p, "w", encoding="ascii", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------
# each mutation: (target check, what to break)
# --------------------------------------------------------------------------

def m_baseline(tmp):
    return None                                   # nothing broken


def m_move_a_file_back(tmp):
    src = os.path.join(tmp, "towards_eeg", "hybrid", "population.py")
    dst = os.path.join(tmp, "HybridLFPy Tweaked", "Population_multiMorph.py")
    os.makedirs(os.path.dirname(dst))
    shutil.move(src, dst)
    return "check_01_declared_moves_are_applied"


def m_stray_module_in_package(tmp):
    with open(os.path.join(tmp, "towards_eeg", "hybrid", "sneaked_in.py"),
              "w", encoding="ascii", newline="\n") as fh:
        fh.write("x = 1\n")
    return "check_02_no_undeclared_file_in_the_package"


def m_remove_a_package_marker(tmp):
    os.unlink(os.path.join(tmp, "towards_eeg", "io", "__init__.py"))
    return "check_03_layout_matches_appendix_a"


def m_populate_an_empty_package(tmp):
    # Read the target from the declaration rather than naming a package.
    # This mutation used to hardcode towards_eeg/io, which S0.5 legitimately
    # populated -- so the mutation silently stopped testing anything. A
    # mutation that names a moving target rots into a no-op (trap T-11).
    with open(os.path.join(tmp, "tools", "s0_transform", "s04_exit_scope.json"),
              "r", encoding="utf-8") as fh:
        pkgs = json.load(fh)["empty_by_construction"]["packages"]
    assert pkgs, "no package is declared empty; this mutation has no subject"
    with open(os.path.join(tmp, pkgs[0], "readers.py"),
              "w", encoding="ascii", newline="\n") as fh:
        fh.write("x = 1\n")
    return "check_04_empty_packages_are_still_empty"


def m_illegal_identifier_in_package(tmp):
    d = os.path.join(tmp, "towards_eeg", "bad name")
    os.makedirs(d)
    with open(os.path.join(d, "__init__.py"), "w", encoding="ascii",
              newline="\n") as fh:
        fh.write('""" """\n')
    return "check_05_every_package_path_is_a_legal_identifier"


def m_uncompilable_module(tmp):
    p = os.path.join(tmp, "towards_eeg", "cosim", "__init__.py")
    with open(p, "a", encoding="ascii", newline="\n") as fh:
        fh.write("\ndef (:\n")
    return "check_07_package_parses_and_compiles"


def m_retired_suffix_returns(tmp):
    shutil.move(os.path.join(tmp, "towards_eeg", "hybrid", "params.py"),
                os.path.join(tmp, "towards_eeg", "hybrid",
                             "params_multiMorph.py"))
    return "check_08_retired_suffixes_are_gone"


def m_undeclared_duplicate(tmp):
    edit_scope(tmp, lambda d: d.__setitem__("duplicate_class_allowlist", []))
    return "check_09_duplicates_are_declared_and_still_duplicated"


def m_stale_allowlist_entry(tmp):
    def add_ghost(d):
        d["duplicate_class_allowlist"].append(
            {"name": "GhostClass", "expected_count": 2,
             "paths": ["towards_eeg/hybrid/params.py",
                       "towards_eeg/connectome/connectomics.py"],
             "rationale": "fixture", "assigned_to": "nowhere"})
    edit_scope(tmp, add_ghost)
    return "check_09_duplicates_are_declared_and_still_duplicated"


def m_keep_row_outside_the_package(tmp):
    def bend(rows):
        for r in rows:
            if r["path"] == "Population/Population_multiMorph.py":
                r["verdict"] = "keep"
                r["stage"] = "S0.4"
    edit_ledger(tmp, bend)
    return "check_10_every_keep_row_is_in_the_package"


def m_delete_a_superseded_fork(tmp):
    os.unlink(os.path.join(tmp, "Population", "Population_multiMorph.py"))
    return "check_11_superseded_forks_retained_not_deleted"


def m_ancestor_overwritten(tmp):
    def bend(rows):
        for r in rows:
            if r["path"] == "towards_eeg/hybrid/utility.py":
                r["ancestor_path"] = r["path"]
    edit_ledger(tmp, bend)
    return "check_12_ledger_records_every_move_with_its_ancestor"


def m_packaging_disagrees_with_layout(tmp):
    p = os.path.join(tmp, "pyproject.toml")
    body = open(p, "r", encoding="utf-8").read()
    body = body.replace('include = ["towards_eeg*"]', 'include = ["nothing*"]')
    open(p, "w", encoding="ascii", newline="\n").write(body)
    return "check_13_packaging_declaration_matches_the_layout"


def m_dependencies_pinned_before_b4(tmp):
    p = os.path.join(tmp, "pyproject.toml")
    body = open(p, "r", encoding="utf-8").read()
    body = body.replace("dependencies = []", 'dependencies = ["neuron>=8.0"]')
    open(p, "w", encoding="ascii", newline="\n").write(body)
    return "check_13_packaging_declaration_matches_the_layout"


MUTATIONS = [m_move_a_file_back, m_stray_module_in_package,
             m_remove_a_package_marker, m_populate_an_empty_package,
             m_illegal_identifier_in_package, m_uncompilable_module,
             m_retired_suffix_returns, m_undeclared_duplicate,
             m_stale_allowlist_entry, m_keep_row_outside_the_package,
             m_delete_a_superseded_fork, m_ancestor_overwritten,
             m_packaging_disagrees_with_layout,
             m_dependencies_pinned_before_b4]


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Mutation harness for the S0.4 exit test.")
    ap.add_argument("--root", default=".")
    args = ap.parse_args(argv)

    results = []

    tmp = fixture(args.root)
    try:
        base = verdicts(tmp)
        red = [k for k, ok in base.items() if not ok]
        results.append(("baseline_is_green", not red,
                        "the unmutated fixture passes all %d check(s)" % len(base)
                        if not red else "fixture is already red: %r" % red))
    finally:
        shutil.rmtree(tmp)

    for mut in MUTATIONS:
        tmp = fixture(args.root)
        try:
            target = mut(tmp)
            v = verdicts(tmp)
            if target not in v:
                ok, msg = False, "no such check: %s" % target
            elif v[target]:
                ok, msg = False, "MUTATION SURVIVED: %s still passes" % target
            else:
                collateral = [k for k, good in v.items()
                              if not good and k != target]
                ok = True
                msg = "caught by %s%s" % (
                    target,
                    "" if not collateral else " (also red: %s)" % ", ".join(
                        sorted(collateral)))
        except Exception as exc:                           # noqa: BLE001
            ok, msg = False, "raised %s: %s" % (type(exc).__name__, exc)
        finally:
            shutil.rmtree(tmp)
        results.append((mut.__name__, ok, msg))

    width = max(len(n) for n, _, _ in results)
    n_ok = sum(1 for _, ok, _ in results if ok)
    for name, ok, msg in results:
        print("%-*s  %s   %s" % (width, name, "PASS" if ok else "FAIL", msg))
    print("-" * (width + 40))
    print("%d/%d checks passed" % (n_ok, len(results)))
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
