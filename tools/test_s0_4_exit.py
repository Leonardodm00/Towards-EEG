#!/usr/bin/env python3
"""
test_s0_4_exit.py -- exit test for stage S0.4 (package layout, git mv only).

    python3 tools/test_s0_4_exit.py --root .

WHAT S0.4 CLAIMS
----------------
Nine files moved into towards_eeg/ and not one byte changed. Nine package
markers were created as clause-(3) new infrastructure. Three files that had
verdict 'keep' were demoted to 'retain' because they are superseded forks,
and they are still in the repository.

WHAT THIS FILE ASSERTS, AND WHAT IT DELIBERATELY DOES NOT
---------------------------------------------------------
The byte-identity claim is NOT re-asserted here. It is a link in the chain --
sha256_post_s04 == sha256_post_s03 for every T6 row -- and it lives in
tools/test_s0_chain.py check_09, because a fresh per-step verifier written
here would itself go stale at S0.5 the moment anything moves again (Doc 6 s4,
trap T-2). This file asserts what the chain cannot see: that the layout is the
declared one, that the identifiers are legal, that the retired suffixes are
gone, that nothing 'keep' was left outside the package, and that every
declared exemption still applies.

Smoke assertion 3 (importlib.import_module on every module) is NOT asserted
here and cannot be: the package imports neuron, nest, mpi4py, LFPy and
hybridLFPy, none of which exist in a bare checkout. What is asserted instead
is that every module is REACHABLE by dotted name -- the part S0.4 is
responsible for -- with the actual import deferred to S0.9, where the
environment exists. Saying so explicitly is the point; a check that silently
weakened assertion 3 would be worse than no check.

Every exclusion is self-verifying: a declared duplicate that is no longer
duplicated is an ERROR, not a harmless leftover.

Standard library only, ASCII source, Python 3.8+.
"""

import argparse
import ast
import collections
import csv
import json
import keyword
import os
import py_compile
import re
import sys
import tempfile
import warnings

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from s0_paths import Resolver  # noqa: E402

IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def legal_identifier(name):
    return bool(IDENT.match(name)) and not keyword.iskeyword(name)


def package_modules(root, pkg_root):
    """Every .py under the package root, as repository-relative paths."""
    out = []
    base = os.path.join(root, pkg_root)
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for fn in sorted(filenames):
            if fn.endswith(".py"):
                full = os.path.join(dirpath, fn)
                out.append(os.path.relpath(full, root).replace(os.sep, "/"))
    return sorted(out)


def top_level_names(root, rel):
    """(classes, functions) defined at module scope in one file."""
    with open(os.path.join(root, rel), "r", encoding="utf-8") as fh:
        src = fh.read()
    # E-11: six files raise SyntaxWarning for invalid escape sequences. That
    # is a latent defect carried forward untouched by S0 in both directions,
    # and its noise here would bury a real failure.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        tree = ast.parse(src)
    classes = [n.name for n in tree.body if isinstance(n, ast.ClassDef)]
    funcs = [n.name for n in tree.body
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    return classes, funcs


def run(root):
    with open(os.path.join(root, "tools", "s0_transform", "s04_exit_scope.json"),
              "r", encoding="utf-8") as fh:
        scope = json.load(fh)
    with open(os.path.join(root, "tools", "path_moves.json"),
              "r", encoding="utf-8") as fh:
        moves = json.load(fh)["moves"]
    with open(os.path.join(root, "ledger.csv"), "r", encoding="utf-8",
              newline="") as fh:
        led = list(csv.DictReader(fh))

    pkg_root = scope["package_root"]
    s04_moves = [m for m in moves if m.get("stage") == "S0.4"]
    modules = package_modules(root, pkg_root)
    results = []

    def add(name, ok, msg):
        results.append((name, ok, msg))

    # -- 1: the declared moves happened, in both directions -----------------
    problems = []
    for m in s04_moves:
        if os.path.exists(os.path.join(root, m["from"])):
            problems.append("source still present: %s" % m["from"])
        if not os.path.isfile(os.path.join(root, m["to"])):
            problems.append("target absent: %s" % m["to"])
    add("check_01_declared_moves_are_applied", not problems,
        "%d declared S0.4 move(s) applied, no source left behind" % len(s04_moves)
        if not problems else "; ".join(problems[:5]))

    # -- 2: nothing entered the package that was not declared ---------------
    # "Declared" spans stages, not just this one. A file created by a LATER
    # sub-step and declared as new infrastructure in tools/ancestors.json is
    # declared; requiring it to be an S0.4 move target would force every later
    # stage either to lie about when its files appeared or to weaken this
    # check. What must still fail is a file declared NOWHERE.
    declared_targets = set(m["to"] for m in s04_moves)
    with open(os.path.join(root, "tools", "ancestors.json"),
              "r", encoding="utf-8") as fh:
        anc = json.load(fh)
    declared_new = set(e if isinstance(e, str) else e["path"]
                       for e in anc.get("new_infrastructure", []))
    markers = set(p for p in modules if os.path.basename(p) == "__init__.py")
    undeclared = [p for p in modules if p not in declared_targets
                  and p not in declared_new and p not in markers]
    later = sorted(p for p in modules if p in declared_new)
    add("check_02_no_undeclared_file_in_the_package", not undeclared,
        "every .py in %s is a declared S0.4 move target, a package marker, or "
        "declared new infrastructure from a later sub-step (%d of those)"
        % (pkg_root, len(later))
        if not undeclared else "undeclared: %r" % undeclared)

    # -- 3: the layout is the declared one ----------------------------------
    missing, unmarked = [], []
    for pkg in scope["expected_subpackages"]:
        if not os.path.isdir(os.path.join(root, pkg)):
            missing.append(pkg)
        elif not os.path.isfile(os.path.join(root, pkg, "__init__.py")):
            unmarked.append(pkg)
    add("check_03_layout_matches_appendix_a", not (missing or unmarked),
        "%d subpackage(s) present, each with a package marker"
        % len(scope["expected_subpackages"])
        if not (missing or unmarked)
        else "missing: %r unmarked: %r" % (missing, unmarked))

    # -- 4: the packages declared empty are still empty ---------------------
    # Self-verifying: if one of these acquired a module, the declaration is
    # stale and the exit test must say so rather than skip it.
    stale = []
    for pkg in scope["empty_by_construction"]["packages"]:
        contents = [p for p in modules
                    if p.startswith(pkg + "/") and not p.endswith("__init__.py")]
        if contents:
            stale.append("%s now holds %r" % (pkg, contents))
    add("check_04_empty_packages_are_still_empty", not stale,
        "%d package(s) declared empty are empty; S0 could not populate them"
        % len(scope["empty_by_construction"]["packages"])
        if not stale else "STALE DECLARATION: %s" % "; ".join(stale))

    # -- 5: T-8, package-scoped (decision N-12) -----------------------------
    illegal = []
    for rel in modules:
        parts = rel.split("/")
        for comp in parts[:-1]:
            if not legal_identifier(comp):
                illegal.append("directory %r in %s" % (comp, rel))
        stem = parts[-1][:-3]
        if not legal_identifier(stem):
            illegal.append("stem %r in %s" % (stem, rel))
    add("check_05_every_package_path_is_a_legal_identifier", not illegal,
        "%d module(s) reachable by dotted name; T-8 discharged inside the "
        "package (N-12)" % len(modules)
        if not illegal else "illegal: %r" % illegal[:5])

    # -- 6: reachability by dotted name -------------------------------------
    # NOT smoke assertion 3. See the module docstring: a real import needs
    # neuron/nest/LFPy/hybridLFPy, which S0.4 does not provide. What S0.4 owns
    # is that the dotted name resolves to exactly this file.
    unreachable = []
    for rel in modules:
        dotted = rel[:-3].replace("/", ".")
        if dotted.endswith(".__init__"):
            dotted = dotted[: -len(".__init__")]
        expected = os.path.join(root, rel)
        parts = dotted.split(".")
        walk = os.path.join(root, *parts)
        if not (os.path.isfile(walk + ".py") or
                os.path.isfile(os.path.join(walk, "__init__.py"))):
            unreachable.append(dotted)
        elif os.path.isfile(walk + ".py") and \
                os.path.abspath(walk + ".py") != os.path.abspath(expected):
            unreachable.append("%s resolves elsewhere" % dotted)
    add("check_06_dotted_names_resolve_to_these_files", not unreachable,
        "%d dotted module name(s) resolve; the import itself is S0.9's, once "
        "the environment exists" % len(modules)
        if not unreachable else "unresolved: %r" % unreachable[:5])

    # -- 7: parse and compile ------------------------------------------------
    fd, cfile = tempfile.mkstemp(suffix=".pyc")
    os.close(fd)
    bad_parse, bad_compile = [], []
    try:
        for rel in modules:
            full = os.path.join(root, rel)
            try:
                with open(full, "r", encoding="utf-8") as fh:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", SyntaxWarning)
                        ast.parse(fh.read())
            except (SyntaxError, UnicodeDecodeError) as exc:
                bad_parse.append("%s: %s" % (rel, exc))
                continue
            try:
                py_compile.compile(full, cfile=cfile, doraise=True)
            except Exception:                              # noqa: BLE001
                bad_compile.append(rel)
    finally:
        os.unlink(cfile)
    add("check_07_package_parses_and_compiles", not (bad_parse or bad_compile),
        "%d/%d package module(s) parse and compile; no O9 file is in the package"
        % (len(modules), len(modules))
        if not (bad_parse or bad_compile)
        else "parse: %r compile: %r" % (bad_parse[:3], bad_compile[:3]))

    # -- 8: the retired suffixes are gone ------------------------------------
    survivors = []
    for rel in modules:
        for suf in scope["retired_filename_suffixes"]:
            if suf in os.path.basename(rel):
                survivors.append(rel)
    add("check_08_retired_suffixes_are_gone", not survivors,
        "%s retired inside the package (TEEG_00 s2)"
        % " and ".join(scope["retired_filename_suffixes"])
        if not survivors else "still present: %r" % survivors)

    # -- 9: declared duplicates, self-verifying ------------------------------
    seen_c = collections.defaultdict(list)
    seen_f = collections.defaultdict(list)
    for rel in modules:
        try:
            classes, funcs = top_level_names(root, rel)
        except SyntaxError:
            continue
        for n in classes:
            seen_c[n].append(rel)
        for n in funcs:
            seen_f[n].append(rel)

    def audit(kind, seen, allowlist):
        problems = []
        allowed = {}
        for entry in allowlist:
            allowed[entry["name"]] = entry
            actual = sorted(seen.get(entry["name"], []))
            if len(actual) < 2:
                problems.append(
                    "STALE ALLOWLIST: %s %r is declared duplicated but is "
                    "defined in %d place(s); remove the entry"
                    % (kind, entry["name"], len(actual)))
                continue
            if actual != sorted(entry["paths"]):
                problems.append(
                    "%s %r is duplicated somewhere undeclared: declared %r, "
                    "found %r" % (kind, entry["name"], sorted(entry["paths"]),
                                  actual))
            elif len(actual) != entry["expected_count"]:
                problems.append(
                    "%s %r: declared count %d, found %d"
                    % (kind, entry["name"], entry["expected_count"], len(actual)))
        for name, paths in sorted(seen.items()):
            if len(paths) > 1 and name not in allowed:
                problems.append("UNDECLARED duplicate %s %r in %r"
                                % (kind, name, sorted(paths)))
        return problems

    problems = (audit("class", seen_c, scope["duplicate_class_allowlist"])
                + audit("function", seen_f, scope["duplicate_function_allowlist"]))
    n_allow = (len(scope["duplicate_class_allowlist"])
               + len(scope["duplicate_function_allowlist"]))
    add("check_09_duplicates_are_declared_and_still_duplicated", not problems,
        "%d declared duplicate name(s), each still duplicated at exactly the "
        "declared paths; no undeclared duplicate" % n_allow
        if not problems else "; ".join(problems[:4]))

    # -- 10: 'keep' means 'in the package', and nothing else -----------------
    stray = [r["path"] for r in led
             if r["verdict"] == "keep"
             and not r["path"].startswith(pkg_root + "/")
             and r["stage"] != "S0.1"]
    add("check_10_every_keep_row_is_in_the_package", not stray,
        "every 'keep' row is inside %s (the four S0.1 rows are working-branch "
        "aliases whose content lives at the ancestor path)" % pkg_root
        if not stray else "keep outside the package: %r" % stray)

    # -- 11: demotion is demotion, not deletion ------------------------------
    by_path = dict((r["path"], r) for r in led)
    problems = []
    for p in scope["superseded_forks_that_must_remain_in_the_repository"]:
        if not os.path.isfile(os.path.join(root, p)):
            problems.append("superseded fork was DELETED: %s" % p)
            continue
        row = by_path.get(p)
        if row is None:
            problems.append("no ledger row for %s" % p)
        elif row["verdict"] != "retain":
            problems.append("%s has verdict %r, expected 'retain'"
                            % (p, row["verdict"]))
    add("check_11_superseded_forks_retained_not_deleted", not problems,
        "%d superseded fork(s) still in the repository, verdict 'retain'; S0 "
        "deletes nothing outside the binary payload"
        % len(scope["superseded_forks_that_must_remain_in_the_repository"])
        if not problems else "; ".join(problems))

    # -- 12: the ledger agrees with the move list ----------------------------
    R = Resolver(root)
    problems = []
    for m in s04_moves:
        row = by_path.get(m["to"])
        if row is None:
            problems.append("no ledger row at %s" % m["to"])
            continue
        if row["transform_id"] != "T6":
            problems.append("%s: transform_id %r, expected 'T6'"
                            % (m["to"], row["transform_id"]))
        if R.original(m["to"]) != m["from"]:
            problems.append("%s: does not resolve back to %s"
                            % (m["to"], m["from"]))
        if row["ancestor_path"] == row["path"]:
            problems.append("%s: ancestor_path was overwritten with the "
                            "post-move path" % m["to"])
    add("check_12_ledger_records_every_move_with_its_ancestor", not problems,
        "%d row(s) carry T6 and still name their pre-move ancestor"
        % len(s04_moves) if not problems else "; ".join(problems[:4]))

    # -- 13: the packaging declaration matches the layout --------------------
    # S0.4 is the stage that claims "the repository is installable". Without a
    # pyproject.toml that is an untestable claim; with one that disagrees with
    # the tree it is a false one.
    problems = []
    pyproj = os.path.join(root, "pyproject.toml")
    if not os.path.isfile(pyproj):
        problems.append("pyproject.toml is absent; the packaging claim is "
                        "untestable")
    else:
        with open(pyproj, "r", encoding="utf-8") as fh:
            body = fh.read()
        try:
            import tomllib                                # noqa: PLC0415
            doc = tomllib.loads(body)
        except ImportError:
            # Python 3.8-3.10 have no tomllib. Say so rather than skipping:
            # a check that quietly does less is the failure mode this suite
            # exists to remove.
            doc = None
            problems.append("NOTE: no tomllib on Python %d.%d, so only the "
                            "textual checks below ran"
                            % sys.version_info[:2])
        if doc is not None:
            find = doc.get("tool", {}).get("setuptools", {}) \
                      .get("packages", {}).get("find", {})
            if find.get("include") != ["towards_eeg*"]:
                problems.append("package discovery is not pinned to "
                                "towards_eeg*: %r" % find.get("include"))
            deps = doc.get("project", {}).get("dependencies", None)
            if deps:
                problems.append("dependencies are declared here (%r) but "
                                "belong in the S0.8 lockfiles, and B-4 is "
                                "unanswered" % deps)
            try:
                from setuptools import find_packages       # noqa: PLC0415
                found = sorted(find_packages(where=root,
                                             include=["towards_eeg*"]))
                expected = sorted(p.replace("/", ".")
                                  for p in scope["expected_subpackages"])
                if found != expected:
                    problems.append("setuptools would package %r, layout "
                                    "declares %r" % (found, expected))
            except ImportError:
                problems.append("NOTE: setuptools unavailable, so discovery "
                                "was not exercised")
        elif "towards_eeg*" not in body:
            problems.append("pyproject.toml does not pin discovery to "
                            "towards_eeg*")
    add("check_13_packaging_declaration_matches_the_layout", not problems,
        "pyproject.toml packages exactly the %d declared subpackage(s), with "
        "no dependency list pending B-4"
        % len(scope["expected_subpackages"])
        if not problems else "; ".join(problems[:3]))

    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description="S0.4 exit test (package layout).")
    ap.add_argument("--root", default=".")
    args = ap.parse_args(argv)
    results = run(args.root)
    width = max(len(n) for n, _, _ in results)
    n_ok = sum(1 for _, ok, _ in results if ok)
    for name, ok, msg in results:
        print("%-*s  %s   %s" % (width, name, "PASS" if ok else "FAIL", msg))
    print("-" * (width + 40))
    print("%d/%d checks passed" % (n_ok, len(results)))
    print("\nS0.4 EXIT TEST: %s" % ("PASS" if n_ok == len(results) else "FAIL"))
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
