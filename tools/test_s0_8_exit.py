#!/usr/bin/env python3
"""
test_s0_8_exit.py -- exit test for stage S0.8 (dependency environment pinned).

    python3 tools/test_s0_8_exit.py --root .

WHAT S0.8 CLAIMS
----------------
The dependency graph the package actually imports is resolvable, pinned by
version AND sha256 in requirements/hybrid_stack.lock, and demonstrated to
install and import in a clean environment. It does NOT claim to have closed
B-4, which is about the davinci HPC environment specifically.

WHY THIS TEST DOES NO NETWORK I/O
---------------------------------
It would be natural to make the exit test install the lockfile and import the
stack. It must not. These harnesses run in Colab under TEEG_08, and a check
that downloaded ~400 MB of wheels and built an sdist would take minutes, fail
on any network hiccup, and be red for reasons having nothing to do with S0.8.
Doc 6 section 4 warns against permanently-red tests; a flaky one is worse,
because it teaches people to rerun until green.

The install-and-import was performed ONCE, deliberately, and its result is
recorded in requirements/hybrid_stack.validation.json. This test asserts that
the record exists, is internally consistent with the lockfile, covers the
import surface measured from the source, and IS HONEST ABOUT WHERE IT WAS
PRODUCED. That last one is check_08, and it is the one that matters: a
validation record that quietly claimed to be the HPC environment would make
S0.8 look like it closed B-4 when it did not.

Standard library only, ASCII source, Python 3.8+.
"""

import argparse
import ast
import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

PIN_RE = re.compile(r"^([A-Za-z0-9_.\-]+)==([^\s\\]+)")
HASH_RE = re.compile(r"--hash=sha256:([0-9a-f]{64})")


def parse_lock(text):
    """Return {name_lower: (name, version, [hashes])} and a list of problems."""
    pins, problems, current = {}, [], None
    for i, raw in enumerate(text.split("\n"), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = PIN_RE.match(line)
        if m:
            current = m.group(1)
            pins[current.lower()] = (current, m.group(2), [])
            # an unpinned or range requirement must never appear
            continue
        h = HASH_RE.search(line)
        if h:
            if current is None:
                problems.append("line %d: hash with no preceding package" % i)
            else:
                pins[current.lower()][2].append(h.group(1))
            continue
        if line != "\\":
            problems.append("line %d: unrecognised %r" % (i, line[:50]))
    return pins, problems


def import_surface(root):
    """Third-party top-level modules imported anywhere under towards_eeg/."""
    stdlib = getattr(sys, "stdlib_module_names", None)
    if stdlib is None:                      # Python 3.8/3.9 fallback
        stdlib = set(sys.builtin_module_names)
    out = set()
    for dp, dn, fn in os.walk(os.path.join(root, "towards_eeg")):
        dn[:] = [d for d in dn if d != "__pycache__"]
        for f in fn:
            if not f.endswith(".py"):
                continue
            try:
                tree = ast.parse(open(os.path.join(dp, f), "r", encoding="utf-8",
                                      errors="replace").read())
            except SyntaxError:
                continue
            for n in ast.walk(tree):
                mods = []
                if isinstance(n, ast.Import):
                    mods = [a.name for a in n.names]
                elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
                    mods = [n.module]
                for m in mods:
                    top = str(m).split(".")[0]
                    if top not in stdlib and top != "towards_eeg":
                        out.add(top)
    return out


def run(root):
    tdir = os.path.join(root, "tools", "s0_transform")
    with open(os.path.join(tdir, "s08_exit_scope.json"), "r", encoding="utf-8") as fh:
        scope = json.load(fh)
    lock_rel = scope["lockfile"]
    val_rel = scope["validation_record"]
    results = []

    def add(name, ok, msg):
        results.append((name, ok, msg))

    # -- 1: the lockfile is present and clean -------------------------------
    problems = []
    lock_path = os.path.join(root, lock_rel)
    text = ""
    if not os.path.isfile(lock_path):
        problems.append("missing: %s" % lock_rel)
    else:
        data = open(lock_path, "rb").read()
        if any(b > 127 for b in data):
            problems.append("non-ASCII bytes in %s" % lock_rel)
        if b"\r\n" in data:
            problems.append("CRLF in %s" % lock_rel)
        # Decode leniently. An earlier version used strict ascii here, so a
        # lockfile with one non-ASCII byte made run() RAISE instead of
        # returning a red check_01 -- and an exit test that crashes returns no
        # verdict on any of its other nine checks. Same defect the S0.6
        # mutation harness found in test_s0_6_exit.py. Report red; do not abort.
        text = data.decode("ascii", "replace")
    add("check_01_lockfile_present_and_clean", not problems,
        "%s is present, pure ASCII, LF-only" % lock_rel
        if not problems else "; ".join(problems[:3]))

    pins, parse_problems = parse_lock(text) if text else ({}, ["no lockfile"])

    # -- 2: it parses, with nothing unrecognised ----------------------------
    add("check_02_lockfile_parses", not parse_problems,
        "%d pinned package(s), no unrecognised lines" % len(pins)
        if not parse_problems else "; ".join(parse_problems[:3]))

    # -- 3: EVERY entry is pinned exactly and hashed ------------------------
    # A lockfile with one >= in it is not a lockfile. Hashes because the
    # project's whole discipline is that an artefact is identified by its
    # bytes, and a version number is not its bytes.
    problems = []
    for low, (name, ver, hashes) in sorted(pins.items()):
        if not hashes:
            problems.append("%s has no sha256" % name)
        for h in hashes:
            if len(h) != 64:
                problems.append("%s has a malformed sha256" % name)
    for bad in (">=", "<=", "~=", ">", "<"):
        for line in text.split("\n"):
            s = line.strip()
            if s.startswith("#") or not s:
                continue
            if bad in s and "--hash" not in s:
                problems.append("unpinned requirement: %r" % s[:40])
                break
    add("check_03_every_entry_is_pinned_and_hashed", not problems,
        "all %d entr(ies) pinned with == and carrying a sha256; "
        "--require-hashes is usable" % len(pins)
        if not problems else "; ".join(problems[:4]))

    # -- 4: no package appears twice ----------------------------------------
    names = [PIN_RE.match(l.strip()).group(1).lower()
             for l in text.split("\n")
             if l.strip() and not l.strip().startswith("#") and PIN_RE.match(l.strip())]
    dupes = sorted(set(n for n in names if names.count(n) > 1))
    add("check_04_no_duplicate_pins", not dupes,
        "no package is pinned twice" if not dupes
        else "duplicated: %r" % dupes)

    # -- 5: the lockfile covers the measured import surface -----------------
    # Measured from the SOURCE, not read from a list someone maintains.
    measured = import_surface(root)
    excluded = set(scope["import_surface"]["excluded"].keys())
    covered = set(pins.keys())
    ALIAS = {"nest": "nest-simulator"}
    missing = []
    for mod in sorted(measured - excluded):
        cand = {mod.lower(), ALIAS.get(mod, mod).lower(), mod.lower().replace("_", "-")}
        if not (cand & covered):
            missing.append(mod)
    add("check_05_lockfile_covers_the_import_surface", not missing,
        "every one of the %d third-party module(s) imported under towards_eeg/ "
        "is either pinned or excluded by declaration"
        % len(measured) if not missing else "not covered: %r" % missing)

    # -- 6: the declared exclusions are still real --------------------------
    # A stale exclusion is an error, not a leftover (Doc 7 s3). If a module
    # stopped being imported, its exclusion should go rather than linger.
    stale = sorted(excluded - measured)
    add("check_06_declared_exclusions_are_still_imported", not stale,
        "all %d declared exclusion(s) are still imported somewhere, so none is "
        "a leftover" % len(excluded)
        if not stale else "excluded but no longer imported: %r" % stale)

    # -- 7: the validation record agrees with the lockfile ------------------
    problems = []
    val = None
    vp = os.path.join(root, val_rel)
    if not os.path.isfile(vp):
        problems.append("missing: %s" % val_rel)
    else:
        with open(vp, "r", encoding="ascii") as fh:
            val = json.load(fh)
        if val.get("n_packages") != len(pins):
            problems.append("record says %r package(s), lockfile pins %d"
                            % (val.get("n_packages"), len(pins)))
        for mod, rec in sorted(val.get("modules_imported", {}).items()):
            if not rec.get("ok"):
                problems.append("%s did not import: %s"
                                % (mod, str(rec.get("error"))[:40]))
    add("check_07_validation_record_agrees_with_the_lockfile", not problems,
        "the record covers %d package(s) and every module it lists imported "
        "successfully" % (val.get("n_packages") if val else 0)
        if not problems else "; ".join(problems[:4]))

    # -- 8: the record is HONEST about where it was produced ----------------
    # The check that matters. S0.8 met the roadmap's exit criterion in a clean
    # Linux venv, NOT on davinci. A record that claimed otherwise would make
    # this sub-step look as though it closed B-4.
    problems = []
    if val is None:
        problems.append("no validation record")
    else:
        env = val.get("environment", {})
        if env.get("is_the_hpc_environment") is not False:
            problems.append("the record does not state that it is NOT the HPC "
                            "environment; S0.8 must not appear to close B-4")
        for key in ("kind", "python", "platform", "why_this_matters"):
            if not env.get(key):
                problems.append("environment lacks %r" % key)
    add("check_08_validation_record_does_not_overclaim", not problems,
        "the record states explicitly that it is not the HPC environment and "
        "says why that matters; S0.8 does not appear to close B-4"
        if not problems else "; ".join(problems[:3]))

    # -- 9: B-4 is still declared open, with actionable next steps ----------
    problems = []
    b4 = scope.get("b4_what_remains", {})
    if "OPEN" not in str(b4.get("status", "")).upper():
        problems.append("B-4 is no longer declared open")
    cmds = b4.get("commands_to_run_on_davinci", [])
    if len(cmds) < 3:
        problems.append("fewer than three concrete commands are given")
    if not b4.get("entries_wrong_for_hpc"):
        problems.append("the lockfile entries that are wrong for HPC are not named")
    add("check_09_b4_remains_open_with_next_steps", not problems,
        "B-4 is declared open, the %d lockfile entr(ies) that are wrong for an "
        "HPC node are named, and %d concrete command(s) are given to close it"
        % (len(b4.get("entries_wrong_for_hpc", {})), len(cmds))
        if not problems else "; ".join(problems[:3]))

    # -- 10: pyproject still declares no runtime dependencies ---------------
    # Declaring them would make `pip install towards_eeg` pull neuron and
    # nest-simulator from PyPI, which on davinci is the WRONG action.
    body = open(os.path.join(root, "pyproject.toml"), "r", encoding="utf-8").read()
    inside, declared = False, []
    for line in body.split("\n"):
        s = line.strip()
        if s.startswith("dependencies"):
            inside = True
            if "[]" in s:
                inside = False
            continue
        if inside:
            if s.startswith("]"):
                inside = False
            elif s and not s.startswith("#"):
                declared.append(s)
    add("check_10_pyproject_still_declares_no_runtime_dependencies", not declared,
        "pyproject.toml declares no runtime dependencies, so installing the "
        "package cannot pull a non-MPI NEURON wheel over an HPC module"
        if not declared else "declares: %r" % declared[:4])

    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description="S0.8 exit test (dependency pinning).")
    ap.add_argument("--root", default=".")
    args = ap.parse_args(argv)
    results = run(args.root)
    width = max(len(n) for n, _, _ in results)
    n_ok = sum(1 for _, ok, _ in results if ok)
    for name, ok, msg in results:
        print("%-*s  %s   %s" % (width, name, "PASS" if ok else "FAIL", msg))
    print("-" * (width + 40))
    print("%d/%d checks passed" % (n_ok, len(results)))
    print("\nS0.8 EXIT TEST: %s" % ("PASS" if n_ok == len(results) else "FAIL"))
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
