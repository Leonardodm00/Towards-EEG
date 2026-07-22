#!/usr/bin/env python3
"""
test_s0_6_smoke.py -- correctness harness for tools/test_s0_6_exit.py.

    python3 tools/test_s0_6_smoke.py --root .

WHY THIS EXISTS
---------------
The project's standard, stated in Doc 6 s3.3: a verifier that has never been
made to fail has not been tested. test_s0_6_exit.py is fourteen checks of new
logic, and fourteen checks that always return PASS are indistinguishable from
fourteen checks that are correct.

Every case below builds a THROWAWAY copy of only what the exit test reads,
breaks exactly one thing, and asserts that the named check goes red. Nothing
here touches the real tree.

TRAP T-15, OBSERVED HERE
------------------------
A mutation that names a moving target rots into a no-op and a rotted mutation
reports PASS. Every mutation below that needs a path, a key or a line number
reads it FROM THE DECLARATION -- s06_exit_scope.json, s06_targets.json or
s06_transform_log.json -- and asserts the declaration is non-empty first, so
that a declaration which empties out fails here rather than silently removing
the mutation's subject.

TWO MUTATIONS THAT MATTER MOST
------------------------------
m_config_value_drifts breaks behaviour preservation by ONE character, which is
the failure TEEG_10 s3.3 is written against: a config that resolves to
something almost right is S0.6 silently fixing the Colab-dependence defect.
m_edit_an_undeclared_line changes a byte outside a declared edit, which is what
the byte-identity rule forbids and what value-equality checking cannot see.

Standard library only, ASCII source, Python 3.8+.
"""

import argparse
import json
import os
import shutil
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import test_s0_6_exit as E  # noqa: E402

COPY_FILES = [
    "pyproject.toml",
    "tools/ancestors.json",
    "tools/s0_transform/s05_exit_scope.json",
    "tools/s0_transform/s06_exit_scope.json",
    "tools/s0_transform/s06_targets.json",
    "tools/s0_transform/s06_transform_log.json",
    "tools/s0_transform/s02_transform_log.json",
    "tools/s0_transform/s03_transform_log.json",
]


def fixture(root):
    """A minimal copy of everything test_s0_6_exit.py reads."""
    tmp = tempfile.mkdtemp(prefix="s06_")
    shutil.copytree(os.path.join(root, "towards_eeg"),
                    os.path.join(tmp, "towards_eeg"),
                    ignore=shutil.ignore_patterns("__pycache__"))
    for rel in COPY_FILES:
        dst = os.path.join(tmp, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(os.path.join(root, rel), dst)
    return tmp


def verdicts(tmp):
    """Run the exit test against the fixture and return {check_name: bool}.

    The exit test imports towards_eeg.config to reach resolve(). A previous
    fixture's module is still in sys.modules and would shadow this one, so the
    package is evicted before each run. Without this every mutation after the
    first would be scored against the first fixture's configuration -- a
    harness that silently stops testing, which is the thing this file exists
    to prevent.
    """
    for name in [m for m in list(sys.modules) if m.startswith("towards_eeg")]:
        del sys.modules[name]
    sys.path.insert(0, tmp)
    try:
        return dict((n, ok) for n, ok, _ in E.run(tmp))
    finally:
        sys.path.remove(tmp)
        for name in [m for m in list(sys.modules) if m.startswith("towards_eeg")]:
            del sys.modules[name]


def _scope(tmp):
    with open(os.path.join(tmp, "tools", "s0_transform", "s06_exit_scope.json"),
              "r", encoding="utf-8") as fh:
        return json.load(fh)


def _log(tmp):
    with open(os.path.join(tmp, "tools", "s0_transform", "s06_transform_log.json"),
              "r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path, doc):
    with open(path, "w", encoding="ascii", newline="") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)
        fh.write("\n")


def _first_edited_file(tmp):
    """Path of a file the transform log declares as edited. From the log."""
    files = _log(tmp)["files"]
    assert files, "no edited file declared; this mutation has no subject"
    return files[0]["path"], files[0]


# --------------------------------------------------------------------------
# each mutation: (target check, what to break)
# --------------------------------------------------------------------------

def m_config_file_removed(tmp):
    os.unlink(os.path.join(tmp, _scope(tmp)["config_data_file"]))
    return "check_01_shipped_configuration_present_and_clean"


def m_config_gains_a_key(tmp):
    cfg = os.path.join(tmp, _scope(tmp)["config_data_file"])
    with open(cfg, "r", encoding="ascii") as fh:
        doc = json.load(fh)
    doc["paths"]["an_undeclared_key"] = {"value": "/tmp/x", "description": "x"}
    _write_json(cfg, doc)
    return "check_02_declared_keys_are_the_keys_that_exist"


def m_a_resolve_call_reverts_to_a_literal(tmp):
    """Put one literal back at its call site, exactly as it was."""
    rel, entry = _first_edited_file(tmp)
    ed = entry["edits"][0]
    full = os.path.join(tmp, rel)
    with open(full, "r", encoding="ascii") as fh:
        text = fh.read()
    import base64
    was = base64.b64decode(ed["original_b64"].encode("ascii")).decode("utf-8")
    now = base64.b64decode(ed["replacement_b64"].encode("ascii")).decode("utf-8")
    assert now in text, "declared replacement absent from the fixture"
    with open(full, "w", encoding="ascii", newline="") as fh:
        fh.write(text.replace(now, was, 1))
    return "check_03_every_declared_occurrence_still_applies"


def m_resolver_reads_at_import(tmp):
    """Make the config module eager. Decision N-20 says it must not be."""
    init = os.path.join(tmp, "towards_eeg", "config", "__init__.py")
    with open(init, "a", encoding="ascii", newline="") as fh:
        fh.write("\n\n_EAGER = load_paths()  # mutation: eager, violates N-20\n")
    return "check_04_resolver_reads_nothing_at_import"


def m_config_value_drifts(tmp):
    """Change ONE character of a configured path.

    This is the failure TEEG_10 s3.3 is written against. A configuration that
    resolves to something almost right is not externalisation; it is S0.6
    silently fixing the Colab-dependence defect, and it would be invisible to
    anything that only checked that a config file exists and parses.
    """
    scope = _scope(tmp)
    cfg = os.path.join(tmp, scope["config_data_file"])
    with open(cfg, "r", encoding="ascii") as fh:
        doc = json.load(fh)
    key = scope["path_keys"][0]
    doc["paths"][key]["value"] = doc["paths"][key]["value"].rstrip("/")
    _write_json(cfg, doc)
    return "check_05_shipped_default_resolves_to_the_prior_literal"


def m_edit_an_undeclared_line(tmp):
    """Change a byte OUTSIDE any declared edit.

    Value equality cannot see this: every resolve() call still returns the
    right string. Only inversion catches it, which is why check_06 exists
    alongside check_05 rather than instead of it.
    """
    rel, entry = _first_edited_file(tmp)
    full = os.path.join(tmp, rel)
    with open(full, "r", encoding="ascii") as fh:
        lines = fh.read().split("\n")
    declared = set()
    ins = entry["insert_import"]["after_line"]
    declared.add(ins)
    for ed in entry["edits"]:
        declared.add(ed["line"] - 1 + (1 if ed["line"] - 1 >= ins else 0))
    victim = next(i for i, ln in enumerate(lines)
                  if i not in declared and ln.startswith("import "))
    lines[victim] = lines[victim] + "  # mutation: undeclared edit"
    with open(full, "w", encoding="ascii", newline="") as fh:
        fh.write("\n".join(lines))
    return "check_06_t11_inverts_to_its_prior_bytes"


def m_new_path_literal_in_the_package(tmp):
    """A path literal in a module that is not exempt."""
    scope = _scope(tmp)
    bad = scope["package_sweep"]["forbidden_substrings"][0]
    target = os.path.join(tmp, "towards_eeg", "hybrid", "params.py")
    with open(target, "a", encoding="ascii", newline="") as fh:
        fh.write("\n_SNEAKED = '%sdrive/MyDrive/whatever'\n" % bad)
    return "check_07_no_undeclared_path_literal_in_the_package"


def m_exemption_widens(tmp):
    """Add a SECOND literal to the whole-file-exempt module.

    check_07 cannot see this, by construction: the file is exempt. check_08
    exists precisely so that a file-level exemption cannot quietly cover more
    than the one thing it was granted for.
    """
    target = os.path.join(tmp, "towards_eeg", "connectome",
                          "connectivity_buildup.py")
    with open(target, "a", encoding="ascii", newline="") as fh:
        fh.write("\n_ALSO = '/content/drive/MyDrive/something_else'\n")
    return "check_08_file_exemption_still_covers_exactly_one_literal"


def m_declared_exclusion_goes_stale(tmp):
    """Remove the drive.mount line the exclusion is granted for.

    Doc 7 s3: an exemption that only skips is an exemption that hides. If the
    excluded line disappears, the exclusion is a leftover and must fail.
    """
    with open(os.path.join(tmp, "tools", "s0_transform", "s06_targets.json"),
              "r", encoding="utf-8") as fh:
        want = json.load(fh)["excluded_in_package"][0]["text"]
    target = os.path.join(tmp, "towards_eeg", "connectome",
                          "connectivity_buildup.py")
    with open(target, "r", encoding="ascii") as fh:
        lines = fh.read().split("\n")
    out = [ln for ln in lines if ln.strip() != want]
    assert len(out) < len(lines), "the excluded line was not in the fixture"
    with open(target, "w", encoding="ascii", newline="") as fh:
        fh.write("\n".join(out))
    return "check_09_declared_exclusion_still_present"


def m_s05_carve_out_dropped(tmp):
    """S0.6 stops carrying forward the S0.5 package-relative declaration."""
    p = os.path.join(tmp, "tools", "s0_transform", "s06_exit_scope.json")
    doc = _scope(tmp)
    doc["excluded_by_declaration"] = [
        e for e in doc["excluded_by_declaration"]
        if "towards_eeg/io/schema.py" not in json.dumps(e)]
    _write_json(p, doc)
    return "check_10_s05_package_relative_declaration_honoured"


def m_config_imports_a_scientific_library(tmp):
    init = os.path.join(tmp, "towards_eeg", "config", "__init__.py")
    with open(init, "r", encoding="ascii") as fh:
        body = fh.read()
    with open(init, "w", encoding="ascii", newline="") as fh:
        fh.write("import numpy  # mutation: config acquires the ability to read data\n"
                 + body)
    return "check_11_config_cannot_read_data"


def m_package_data_undeclared(tmp):
    p = os.path.join(tmp, "pyproject.toml")
    with open(p, "r", encoding="utf-8") as fh:
        body = fh.read()
    with open(p, "w", encoding="ascii", newline="") as fh:
        fh.write(body.replace('"towards_eeg.config" = ["paths.json"]',
                              '# removed by mutation'))
    return "check_12_configuration_is_declared_package_data"


def m_an_unimportable_module_starts_importing(tmp):
    """Make one of the two declared-unimportable modules import cleanly.

    This is the mutation that reads oddly: it makes something WORK and the
    check must still go red. That is the point of a self-verifying
    declaration -- the exit test asserts the declaration is still TRUE, so a
    module that starts importing means the H-3 record and the proposed O10
    assignment are stale and must be revisited, not quietly enjoyed.
    """
    target = os.path.join(tmp, "towards_eeg", "connectome",
                          "connectivity_buildup.py")
    with open(target, "w", encoding="ascii", newline="") as fh:
        fh.write("# mutation: body removed so the module imports\n"
                 "drive_mount = '/content/drive'\n")
    return "check_13_unimportability_declaration_still_true"


def m_transform_id_unregistered(tmp):
    p = os.path.join(tmp, "tools", "ancestors.json")
    with open(p, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    tid = _log(tmp)["transform_id"]
    assert tid in doc["transforms"], "the transform id under test is not registered"
    del doc["transforms"][tid]
    _write_json(p, doc)
    return "check_14_every_used_transform_id_is_registered"


MUTATIONS = [m_config_file_removed, m_config_gains_a_key,
             m_a_resolve_call_reverts_to_a_literal, m_resolver_reads_at_import,
             m_config_value_drifts, m_edit_an_undeclared_line,
             m_new_path_literal_in_the_package, m_exemption_widens,
             m_declared_exclusion_goes_stale, m_s05_carve_out_dropped,
             m_config_imports_a_scientific_library, m_package_data_undeclared,
             m_an_unimportable_module_starts_importing,
             m_transform_id_unregistered]


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Mutation harness for the S0.6 exit test.")
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
