#!/usr/bin/env python3
"""
test_s0_8_smoke.py -- correctness harness for tools/test_s0_8_exit.py.

    python3 tools/test_s0_8_smoke.py --root .

Doc 6 s3.3: a verifier that has never been made to fail has not been tested.
Each case copies what the exit test reads, breaks exactly one thing, and
asserts the named check goes red.

THE MUTATION THAT MATTERS MOST
------------------------------
m_record_claims_to_be_the_hpc_env flips one boolean in the validation record.
Nothing else changes: the lockfile is still correct, every package still
resolves, every module still imported. But S0.8 would then APPEAR to have
closed B-4, and the next chat would build on a dependency story that was never
tested on the cluster. Overclaiming is the failure mode this sub-step is most
exposed to, precisely because everything else about it is green.

m_a_pin_becomes_a_range is the other one: `numpy==2.5.1` to `numpy>=2.5.1`
turns a lockfile into a requirements file while still looking like a lockfile,
and every other check stays green.

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

import test_s0_8_exit as E  # noqa: E402


def fixture(root):
    tmp = tempfile.mkdtemp(prefix="s08_")
    for name in ("tools", "towards_eeg", "requirements"):
        shutil.copytree(os.path.join(root, name), os.path.join(tmp, name),
                        ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(os.path.join(root, "pyproject.toml"),
                 os.path.join(tmp, "pyproject.toml"))
    return tmp


def verdicts(tmp):
    return dict((n, ok) for n, ok, _ in E.run(tmp))


def _lock(tmp):
    return os.path.join(tmp, "requirements", "hybrid_stack.lock")


def _val(tmp):
    return os.path.join(tmp, "requirements", "hybrid_stack.validation.json")


def _scope(tmp):
    return os.path.join(tmp, "tools", "s0_transform", "s08_exit_scope.json")


def _read(p):
    with open(p, "r", encoding="ascii") as fh:
        return fh.read()


def _write(p, s):
    with open(p, "w", encoding="ascii", newline="") as fh:
        fh.write(s)


def _rj(p):
    with open(p, "r", encoding="ascii") as fh:
        return json.load(fh)


def _wj(p, d):
    with open(p, "w", encoding="ascii", newline="") as fh:
        json.dump(d, fh, indent=2, sort_keys=True)
        fh.write("\n")


# --------------------------------------------------------------------------

def m_lockfile_gains_non_ascii(tmp):
    p = _lock(tmp)
    with open(p, "ab") as fh:
        fh.write("# caf\u00e9\n".encode("utf-8"))
    return "check_01_lockfile_present_and_clean"


def m_lockfile_gains_a_junk_line(tmp):
    _write(_lock(tmp), _read(_lock(tmp)) + "this is not a requirement line\n")
    return "check_02_lockfile_parses"


def m_a_hash_is_removed(tmp):
    out, dropped = [], False
    for line in _read(_lock(tmp)).split("\n"):
        if not dropped and "--hash=sha256:" in line:
            dropped = True
            continue
        out.append(line)
    assert dropped, "no hash line found to drop"
    # also drop the trailing backslash that continued onto it
    text = "\n".join(out).replace(" \\\n\n", "\n")
    _write(_lock(tmp), text)
    return "check_03_every_entry_is_pinned_and_hashed"


def m_a_pin_becomes_a_range(tmp):
    """A lockfile that still looks like one but no longer locks."""
    _write(_lock(tmp), _read(_lock(tmp)).replace("numpy==", "numpy>=", 1))
    return "check_03_every_entry_is_pinned_and_hashed"


def m_a_package_is_pinned_twice(tmp):
    text = _read(_lock(tmp))
    _write(_lock(tmp), text + "numpy==1.26.4 \\\n    --hash=sha256:" + "0" * 64 + "\n")
    return "check_04_no_duplicate_pins"


def m_a_needed_package_is_dropped(tmp):
    """Remove scipy from the lockfile while the package still imports it."""
    out, skip = [], False
    for line in _read(_lock(tmp)).split("\n"):
        if line.startswith("scipy=="):
            skip = True
            continue
        if skip:
            if "--hash" in line:
                continue
            skip = False
        out.append(line)
    _write(_lock(tmp), "\n".join(out))
    return "check_05_lockfile_covers_the_import_surface"


def m_a_stale_exclusion_lingers(tmp):
    p = _scope(tmp)
    doc = _rj(p)
    doc["import_surface"]["excluded"]["tensorflow"] = \
        "mutation: not imported anywhere, so this exclusion is a leftover"
    _wj(p, doc)
    return "check_06_declared_exclusions_are_still_imported"


def m_record_disagrees_with_the_lockfile(tmp):
    p = _val(tmp)
    doc = _rj(p)
    doc["n_packages"] = int(doc["n_packages"]) + 3
    _wj(p, doc)
    return "check_07_validation_record_agrees_with_the_lockfile"


def m_a_module_did_not_import(tmp):
    p = _val(tmp)
    doc = _rj(p)
    doc["modules_imported"]["hybridLFPy"] = {
        "ok": False, "error": "mutation: ImportError"}
    _wj(p, doc)
    return "check_07_validation_record_agrees_with_the_lockfile"


def m_record_claims_to_be_the_hpc_env(tmp):
    """One boolean. Everything else stays correct.

    S0.8 would then appear to have closed B-4, and the next chat would build on
    a dependency story never tested on the cluster.
    """
    p = _val(tmp)
    doc = _rj(p)
    doc["environment"]["is_the_hpc_environment"] = True
    _wj(p, doc)
    return "check_08_validation_record_does_not_overclaim"


def m_b4_declared_closed(tmp):
    p = _scope(tmp)
    doc = _rj(p)
    doc["b4_what_remains"]["status"] = "B-4 is CLOSED by S0.8."
    _wj(p, doc)
    return "check_09_b4_remains_open_with_next_steps"


def m_pyproject_declares_dependencies(tmp):
    """Make `pip install towards_eeg` pull a non-MPI NEURON wheel."""
    p = os.path.join(tmp, "pyproject.toml")
    with open(p, "r", encoding="utf-8") as fh:
        body = fh.read()
    body = body.replace("dependencies = []",
                        'dependencies = [\n    "neuron==9.0.1",\n    "nest-simulator==3.10.0",\n]')
    with open(p, "w", encoding="ascii", newline="") as fh:
        fh.write(body)
    return "check_10_pyproject_still_declares_no_runtime_dependencies"


MUTATIONS = [m_lockfile_gains_non_ascii, m_lockfile_gains_a_junk_line,
             m_a_hash_is_removed, m_a_pin_becomes_a_range,
             m_a_package_is_pinned_twice, m_a_needed_package_is_dropped,
             m_a_stale_exclusion_lingers, m_record_disagrees_with_the_lockfile,
             m_a_module_did_not_import, m_record_claims_to_be_the_hpc_env,
             m_b4_declared_closed, m_pyproject_declares_dependencies]


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Mutation harness for the S0.8 exit test.")
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
