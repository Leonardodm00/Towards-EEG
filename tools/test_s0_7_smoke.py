#!/usr/bin/env python3
"""
test_s0_7_smoke.py -- correctness harness for tools/test_s0_7_exit.py.

    python3 tools/test_s0_7_smoke.py --root .

WHY THIS EXISTS
---------------
Doc 6 s3.3: a verifier that has never been made to fail has not been tested.
test_s0_7_exit.py is eleven checks of new logic, and eleven checks that always
return PASS are indistinguishable from eleven checks that are correct.

Each case builds a THROWAWAY copy of what the exit test reads, breaks exactly
one thing, and asserts the named check goes red. Nothing touches the real tree.

THE FIXTURE IS A GIT REPOSITORY
-------------------------------
check_03 asks a question about the INDEX -- what does git expect to exist that
does not -- so the fixture must have one. It is built by copying the tree and
running git init/add/commit, then deleting the manifest's entries so that the
index still remembers them. A fixture without an index would make check_03
vacuous and its mutation would report PASS while testing nothing.

THE MUTATION THAT MATTERS MOST
------------------------------
m_delete_a_twenty_third_file removes a file the manifest does not declare.
Checks 01 and 02 stay GREEN under it -- the bound is met even more comfortably
and every declared file is still gone -- so if check_03 did not exist, an
undeclared deletion would land silently. That asymmetry is the reason the
"nothing else" half is written at all.

m_drop_a_hash_from_the_manifest is the other one: it leaves the path in place
and removes only the hash, which is invisible to every check except 05. That is
finding R-7 in miniature -- the record looks present and is not.

Standard library only, ASCII source, Python 3.8+.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import test_s0_7_exit as E  # noqa: E402


def _git(tmp, *args):
    return subprocess.run(["git"] + list(args), cwd=tmp,
                          capture_output=True, text=True)


def fixture(root):
    """A throwaway git repository in the post-S0.7 state.

    The removed files are restored, committed, then deleted again, so that the
    index remembers them exactly as the real repository's does before the S0.7
    commit lands. That is what check_03 interrogates.
    """
    tmp = tempfile.mkdtemp(prefix="s07_")
    for name in ("tools", "towards_eeg"):
        shutil.copytree(os.path.join(root, name), os.path.join(tmp, name),
                        ignore=shutil.ignore_patterns("__pycache__"))
    for rel in ("ledger.csv", "LEDGER.md", "pyproject.toml"):
        src = os.path.join(root, rel)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(tmp, rel))

    with open(os.path.join(tmp, "tools", "s0_transform",
                           "s07_discard_manifest.json"), "r",
              encoding="ascii") as fh:
        man = json.load(fh)

    # Recreate each removed file with its recorded size, so the index carries
    # it, then commit, then delete. Content is irrelevant; only presence in
    # the index and absence from disk matter to check_03.
    for e in man["entries"]:
        full = os.path.join(tmp, e["path"])
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "wb") as fh:
            fh.write(b"\0" * min(int(e["row"]["size_bytes"]), 64))

    _git(tmp, "init", "-q")
    _git(tmp, "config", "user.email", "smoke@example.invalid")
    _git(tmp, "config", "user.name", "smoke")
    _git(tmp, "add", "-A")
    _git(tmp, "commit", "-q", "-m", "fixture")
    for e in man["entries"]:
        os.unlink(os.path.join(tmp, e["path"]))
    return tmp


def verdicts(tmp):
    return dict((n, ok) for n, ok, _ in E.run(tmp))


def _manifest_path(tmp):
    return os.path.join(tmp, "tools", "s0_transform", "s07_discard_manifest.json")


def _scope_path(tmp):
    return os.path.join(tmp, "tools", "s0_transform", "s07_exit_scope.json")


def _load(path):
    with open(path, "r", encoding="ascii") as fh:
        return json.load(fh)


def _dump(path, doc):
    with open(path, "w", encoding="ascii", newline="") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)
        fh.write("\n")


# --------------------------------------------------------------------------
# each mutation: (target check, what to break)
# --------------------------------------------------------------------------

def m_a_big_file_comes_back(tmp):
    """Restore the largest removed file, pushing the tree over the bound."""
    man = _load(_manifest_path(tmp))
    biggest = max(man["entries"], key=lambda e: int(e["row"]["size_bytes"]))
    full = os.path.join(tmp, biggest["path"])
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "wb") as fh:
        fh.write(b"\0" * (21 * 1024 * 1024))
    return "check_01_tracked_tree_is_under_the_bound"


def m_a_declared_file_survives(tmp):
    man = _load(_manifest_path(tmp))
    # Smallest, so the bound is not also breached and the mutation is isolated.
    small = min(man["entries"], key=lambda e: int(e["row"]["size_bytes"]))
    full = os.path.join(tmp, small["path"])
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "wb") as fh:
        fh.write(b"x")
    return "check_02_every_declared_file_is_removed"


def m_delete_a_twenty_third_file(tmp):
    """Remove a file the manifest does not declare.

    Checks 01 and 02 stay GREEN under this: the bound is met even more
    comfortably, and every declared file is still gone. Only check_03 sees it.
    """
    victim = os.path.join(tmp, "towards_eeg", "io", "geometry.py")
    assert os.path.isfile(victim), "the mutation's subject is missing"
    os.unlink(victim)
    return "check_03_nothing_undeclared_was_removed"


def m_manifest_loses_an_entry(tmp):
    p = _manifest_path(tmp)
    man = _load(p)
    dropped = man["entries"].pop()
    # The file stays deleted, so this is a shrinking RECORD, not a restored
    # file: the tree is right and the ledger of it is not.
    _dump(p, man)
    assert dropped
    return "check_04_manifest_is_complete"


def m_drop_a_hash_from_the_manifest(tmp):
    """Leave the entry, remove its hash.

    Finding R-7 in miniature: the record looks present and is not. Invisible
    to every other check.
    """
    p = _manifest_path(tmp)
    man = _load(p)
    man["entries"][0]["row"]["sha256_pre_s0"] = "-"
    _dump(p, man)
    return "check_05_every_entry_carries_a_usable_identity"


def m_a_removed_file_loses_its_ledger_row(tmp):
    """The exact S0.2 failure: delete the file, delete the row.

    This is what a naive S0.7 would have produced for all 22.
    """
    import csv
    p = _manifest_path(tmp)
    victim = _load(p)["entries"][0]["path"]
    lp = os.path.join(tmp, "ledger.csv")
    with open(lp, "r", encoding="utf-8", newline="") as fh:
        rd = csv.DictReader(fh)
        fields, rows = rd.fieldnames, [r for r in rd if r["path"] != victim]
    with open(lp, "w", encoding="ascii", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return "check_06_no_removed_file_lost_its_ledger_row"


def m_manifest_becomes_an_optional_flag(tmp):
    """Regress N-21: make the manifest something you can forget to pass."""
    p = os.path.join(tmp, "tools", "build_ledger.py")
    with open(p, "r", encoding="utf-8") as fh:
        body = fh.read()
    body = body.replace("s07_discard_manifest.json", "SOMETHING_ELSE.json")
    body += '\n# ap.add_argument("--removed")\n'
    with open(p, "w", encoding="ascii", newline="") as fh:
        fh.write(body)
    return "check_07_ledger_reads_the_manifest_unconditionally"


def m_a_dangling_reference_appears(tmp):
    """A surviving source file starts naming a removed one."""
    man = _load(_manifest_path(tmp))
    name = os.path.basename(man["entries"][0]["path"])
    target = os.path.join(tmp, "towards_eeg", "io", "validate.py")
    with open(target, "a", encoding="ascii", newline="") as fh:
        fh.write("\n_DOC = %r  # mutation: dangling reference\n" % name)
    return "check_08_no_surviving_file_references_a_removed_one"


def m_exemption_list_is_padded(tmp):
    """Add an exemption that is never needed.

    A list that can be padded to silence a genuine hit is not an exemption but
    a mute button, so an unused entry must fail.
    """
    p = _scope_path(tmp)
    doc = _load(p)
    doc["reference_exempt_record_artefacts"]["paths"]["pyproject.toml"] = \
        "mutation: never names a removed file"
    _dump(p, doc)
    return "check_08_no_surviving_file_references_a_removed_one"


def m_declaration_file_gains_a_live_reference(tmp):
    """ancestors.json names a removed file OUTSIDE a declaration."""
    man = _load(_manifest_path(tmp))
    name = os.path.basename(man["entries"][0]["path"])
    p = os.path.join(tmp, "tools", "ancestors.json")
    doc = _load(p)
    doc["note"] = doc.get("note", "") + " see also " + name
    _dump(p, doc)
    return "check_08a_declaration_file_names_removals_only_as_declarations"


def m_a_new_pdf_appears(tmp):
    """A surviving file the extension rule claims but the manifest does not.

    Finding H-7's residual risk made concrete: binary_payload.paths is empty,
    so membership rests on an extension rule that would capture this silently.
    """
    with open(os.path.join(tmp, "towards_eeg", "leftover.pdf"), "wb") as fh:
        fh.write(b"%PDF-1.4\n")
    return "check_09_extension_rule_claims_nothing_that_survived"


def m_transform_unregistered(tmp):
    p = os.path.join(tmp, "tools", "ancestors.json")
    doc = _load(p)
    tid = _load(_scope_path(tmp))["transform_id"]
    assert tid in doc["transforms"], "the transform under test is not registered"
    del doc["transforms"][tid]
    _dump(p, doc)
    return "check_10_transform_is_registered"


MUTATIONS = [m_a_big_file_comes_back, m_a_declared_file_survives,
             m_delete_a_twenty_third_file, m_manifest_loses_an_entry,
             m_drop_a_hash_from_the_manifest,
             m_a_removed_file_loses_its_ledger_row,
             m_manifest_becomes_an_optional_flag,
             m_a_dangling_reference_appears, m_exemption_list_is_padded,
             m_declaration_file_gains_a_live_reference,
             m_a_new_pdf_appears, m_transform_unregistered]


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Mutation harness for the S0.7 exit test.")
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
