#!/usr/bin/env python3
"""
test_s0_chain.py -- assert continuity of the S0 transform chain.

    python3 tools/test_s0_chain.py --root .

WHY THIS EXISTS
---------------
Each transform tool writes a log recording, per file, the hash before and the
hash after ITS OWN transform. `decolab.py --verify` compares the tree against
the S0.2 log's `sha256_after`, and that comparison is true only until the next
transform touches the same file. After S0.3 sweeps the S0.2 targets, it fails
by construction -- ten "on-disk hash does not match sha256_after" reports that
mean the chain advanced, not that anything broke.

Running a stale single-step check and reading a red result as damage is the
failure mode this file removes. The invariant that actually holds across the
whole of S0 is CONTINUITY:

    ledger.sha256_pre_s0   == s02_log.sha256_before   (for S0.2 targets)
    s02_log.sha256_after   == ledger.sha256_post_s02
    ledger.sha256_post_s02 == s03_log.sha256_before   (where S0.3 touched it)
    s03_log.sha256_after   == ledger.sha256_post_s03  == bytes on disk

Every link is an equality between two independently produced records. A gap
anywhere means a byte changed outside a logged transform -- which is exactly
what the governing rule forbids, and the only thing that could make the tree
unreconstructable from its ancestors.

Standard library only, ASCII source, Python 3.8+.
"""

import argparse
import base64
import csv
import hashlib
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from s0_paths import Resolver  # noqa: E402

PHASE_COLUMNS = ("sha256_post_s08", "sha256_post_s07", "sha256_post_s06",
                 "sha256_post_s05", "sha256_post_s04", "sha256_post_s03",
                 "sha256_post_s02", "sha256_pre_s0")


def end_of_chain(row):
    """The most recent stamped hash for one ledger row.

    Hardcoding sha256_post_s03 was correct until S0.4 and is exactly the
    shelf-life problem Doc 7 s3 describes: any tool edited during S0.4
    legitimately differs from its post-S0.3 bytes, and a check pinned to
    post_s03 reports that as drift. Reading the last stamped column instead
    keeps the check alive across every remaining sub-step.
    """
    for col in PHASE_COLUMNS:
        val = row.get(col, "-")
        if val not in ("-", ""):
            return col, val
    return None, None


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(root):
    base = os.path.join(root, "tools", "s0_transform")
    out = {}
    for key, name in (("s02", "s02_transform_log.json"),
                      ("s03", "s03_transform_log.json")):
        p = os.path.join(base, name)
        if os.path.isfile(p):
            with open(p, "r", encoding="utf-8") as fh:
                doc = json.load(fh)
            out[key] = dict((e["path"], e) for e in doc["files"])
        else:
            out[key] = None
    with open(os.path.join(root, "ledger.csv"), "r", encoding="utf-8", newline="") as fh:
        out["ledger"] = dict((r["path"], r) for r in csv.DictReader(fh))
    return out


def run(root):
    d = load(root)
    led, s02, s03 = d["ledger"], d["s02"], d["s03"]
    R = Resolver(root)
    results = []

    def L(path):
        """Ledger row for a path as recorded in a transform log."""
        return led.get(R.current(path))

    def add(name, ok, msg):
        results.append((name, ok, msg))

    # -- link 1 ------------------------------------------------------------
    if s02 is None:
        add("check_01_s02_log_present", False, "no S0.2 transform log")
    else:
        bad = [p for p, e in s02.items()
               if L(p) is not None and L(p)["sha256_pre_s0"] != e["sha256_before"]]
        add("check_01_pre_s0_equals_s02_input", not bad,
            "%d/%d S0.2 target(s) start from their pre-S0 bytes"
            % (len(s02) - len(bad), len(s02)) if not bad else "mismatch: %r" % bad)

        bad = [p for p, e in s02.items()
               if L(p) is not None and L(p)["sha256_post_s02"] != e["sha256_after"]]
        add("check_02_s02_output_equals_ledger_post_s02", not bad,
            "ledger post_s02 agrees with the S0.2 log for all %d target(s)" % len(s02)
            if not bad else "mismatch: %r" % bad)

    # -- link 2 ------------------------------------------------------------
    if s03 is None:
        add("check_03_s03_log_present", False, "no S0.3 transform log")
        return results

    bad = []
    for p, e in s03.items():
        row = L(p)
        if row is None:
            bad.append("%s: no ledger row" % p)
        elif row["sha256_post_s02"] not in ("-", e["sha256_before"]):
            bad.append("%s: S0.3 did not start from the post-S0.2 bytes" % p)
    add("check_03_post_s02_equals_s03_input", not bad,
        "all %d S0.3 target(s) start from their post-S0.2 bytes" % len(s03)
        if not bad else "; ".join(bad[:5]))

    bad = [p for p, e in s03.items()
           if L(p) is not None and L(p)["sha256_post_s03"] != e["sha256_after"]]
    add("check_04_s03_output_equals_ledger_post_s03", not bad,
        "ledger post_s03 agrees with the S0.3 log for all %d target(s)" % len(s03)
        if not bad else "mismatch: %r" % bad)

    # The name of this check states the durable invariant; until S0.6 its body
    # implemented the stale one, comparing disk against the S0.3 log's
    # sha256_after. That held while every later sub-step preserved bytes --
    # S0.4 was git mv, S0.5 only added files -- and expired the moment S0.6
    # edited two S0.3 targets under T11. It reported two failures that meant
    # "the chain advanced", which is precisely the misreading Doc 7 s3 warns
    # about. Comparing against end_of_chain() instead keeps it alive for every
    # remaining sub-step. That the S0.3 log itself is honoured is already
    # asserted by check_04, against the ledger rather than against disk.
    bad = []
    for p, e in s03.items():
        full = R.full(p)
        if not os.path.isfile(full):
            continue
        row = L(p)
        if row is None:
            bad.append("%s: no ledger row" % p)
            continue
        col, expected = end_of_chain(row)
        if expected is None:
            continue
        if sha256_file(full) != expected:
            bad.append("%s (vs %s)" % (p, col))
    add("check_05_disk_equals_end_of_chain", not bad,
        "bytes on disk match the end of the chain for all %d target(s)" % len(s03)
        if not bad else "mismatch: %r" % bad)

    # -- the overlap: files touched by BOTH sub-steps -----------------------
    if s02 is not None:
        overlap = sorted(set(s02) & set(s03))
        bad = [p for p in overlap if s02[p]["sha256_after"] != s03[p]["sha256_before"]]
        add("check_06_s02_output_is_s03_input", not bad,
            "%d file(s) touched by both sub-steps hand over cleanly" % len(overlap)
            if not bad else "broken handover: %r" % bad)

    # -- untouched files must not have moved -------------------------------
    # Four artefacts record hashes of the tree they themselves live in, so
    # writing them necessarily changes them after their own hash was taken.
    # Doc 4 s7 calls this out: "the ledger scans itself". They are declared
    # here rather than skipped silently, and everything else -- including the
    # tool sources -- stays under the check.
    SELF_REFERENTIAL = {
        "ledger.csv",
        "LEDGER.md",
        "tools/phase_hashes.json",
        "tools/s0_transform/s02_transform_log.json",
        "tools/s0_transform/s03_transform_log.json",
        "tools/s0_transform/s02_colab_commands.json",
    }
    # S0.6 additions: the T11 targets, the clause-(3) tooling edits it declares,
    # and its own log. Read from the declaration, never from a literal list --
    # trap T-15, a check that names a moving target quietly verifies less.
    s06_declared = set()
    _sc = os.path.join(root, "tools", "s0_transform", "s06_exit_scope.json")
    if os.path.isfile(_sc):
        with open(_sc, "r", encoding="utf-8") as fh:
            _d = json.load(fh)["clause_3_files_changed_at_s06"]
        s06_declared = set(_d["paths"]) | set(_d["self_referential"])
    _sl = os.path.join(root, "tools", "s0_transform", "s06_transform_log.json")
    if os.path.isfile(_sl):
        with open(_sl, "r", encoding="utf-8") as fh:
            s06_declared |= set(e["path"] for e in json.load(fh)["files"])

    s07_declared = set()
    _s7 = os.path.join(root, "tools", "s0_transform", "s07_exit_scope.json")
    if os.path.isfile(_s7):
        with open(_s7, "r", encoding="utf-8") as fh:
            _d7 = json.load(fh)["clause_3_files_changed_at_s07"]
        s07_declared = set(_d7["paths"]) | set(_d7["self_referential"])
    _m7 = os.path.join(root, "tools", "s0_transform", "s07_discard_manifest.json")
    if os.path.isfile(_m7):
        with open(_m7, "r", encoding="utf-8") as fh:
            s07_declared |= set(e["path"] for e in json.load(fh)["entries"])

    s08_declared = set()
    _s8 = os.path.join(root, "tools", "s0_transform", "s08_exit_scope.json")
    if os.path.isfile(_s8):
        with open(_s8, "r", encoding="utf-8") as fh:
            _d8 = json.load(fh)["clause_3_files_changed_at_s08"]
        s08_declared = set(_d8["paths"]) | set(_d8["self_referential"])

    touched = (set(s03) | set(s02 or {}) | SELF_REFERENTIAL | s06_declared
               | s07_declared | s08_declared)
    drifted = []
    for p, row in led.items():
        if p in touched or row["verdict"] == "discard":
            continue
        if row["relationship"] == "working_branch_edit":
            continue
        full = os.path.join(root, p)
        if not os.path.isfile(full):
            continue
        col, expected = end_of_chain(row)
        if expected is None:
            continue
        if sha256_file(full) != expected:
            drifted.append("%s (vs %s)" % (p, col))
    add("check_07_untouched_files_did_not_drift", not drifted,
        "no file changed outside a logged transform"
        if not drifted else "changed with no log entry: %r" % drifted[:10])

    # -- the S0.2 markers must have survived the sweep ----------------------
    if s02 is not None:
        missing = []
        for p, e in s02.items():
            full = os.path.join(root, p)
            if not os.path.isfile(full):
                continue
            with open(full, "rb") as fh:
                data = fh.read()
            for ed in e["edits"]:
                if ed["transform"].startswith("T7") and b"#S0.2:T7" not in data:
                    missing.append(p)
                    break
        add("check_08_s02_markers_survived_s03", not missing,
            "every S0.2 neutralisation marker is still present after the sweep"
            if not missing else "markers lost in: %r" % missing)

    # -- link 3: S0.4 -------------------------------------------------------
    # The roadmap states the S0.4 exit as "every moved file's hash equals its
    # ancestor's". Concretely that is one more equality on the same chain:
    # a git mv changes no byte, so post_s04 == post_s03 for every T6 row.
    # Written here rather than in a fresh per-step verifier, which would
    # itself go stale at S0.5 (Doc 6 s4).
    t6 = [r for r in led.values() if r["transform_id"] == "T6"]
    stamped = [r for r in t6 if r["sha256_post_s04"] not in ("-", "")]
    bad = [r["path"] for r in stamped
           if r["sha256_post_s04"] != r["sha256_post_s03"]]
    if not t6:
        add("check_09_s04_moves_changed_no_bytes", False, "no T6 row in the ledger")
    elif not stamped:
        add("check_09_s04_moves_changed_no_bytes", False,
            "%d T6 row(s) present but post_s04 is unstamped: run "
            "tools/stamp_phase.py --phase post_s04" % len(t6))
    else:
        add("check_09_s04_moves_changed_no_bytes", not bad,
            "post_s04 == post_s03 for all %d moved file(s)" % len(stamped)
            if not bad else "T6 changed bytes: %r" % bad[:5])

    # -- link 4: S0.6 -------------------------------------------------------
    # S0.4 moved files and S0.5 only added them, so each contributed a phase
    # COLUMN but no link: post_s04 == post_s03 is an equality about bytes that
    # did not change. S0.6 is the first sub-step since S0.3 to edit the content
    # of existing files, so it needs a real link, and a link has two halves.
    #
    # Half A is the obvious one: every file T11 touched must have changed, and
    # changed to exactly the bytes the log records.
    # Half B is the one that is easy to omit (TEEG_10 s3.2): every file T11 did
    # NOT touch must be byte-identical across the sub-step. Without half B, a
    # transform could edit an undeclared file and the chain would still report
    # continuous, because nothing would be looking at that file.
    s06_log = os.path.join(root, "tools", "s0_transform", "s06_transform_log.json")
    s06_scope = os.path.join(root, "tools", "s0_transform", "s06_exit_scope.json")
    if not os.path.isfile(s06_log):
        add("check_11_s06_link_present", False, "no S0.6 transform log")
        return results

    with open(s06_log, "r", encoding="utf-8") as fh:
        s06 = dict((e["path"], e) for e in json.load(fh)["files"])
    with open(s06_scope, "r", encoding="utf-8") as fh:
        scope6 = json.load(fh)

    # -- half A -------------------------------------------------------------
    bad = []
    for p, e in s06.items():
        row = led.get(p)
        if row is None:
            bad.append("%s: no ledger row" % p)
            continue
        if row["sha256_post_s05"] != e["sha256_before"]:
            bad.append("%s: T11 did not start from the post-S0.5 bytes" % p)
        if row["sha256_post_s06"] != e["sha256_after"]:
            bad.append("%s: ledger post_s06 disagrees with the T11 log" % p)
        if row["sha256_post_s06"] == row["sha256_post_s05"]:
            bad.append("%s: declared as edited by T11 but no byte changed" % p)
    add("check_11_s06_edited_files_changed_to_the_logged_bytes", not bad,
        "all %d T11 target(s) start from post_s05, end at the logged hash, and "
        "did change" % len(s06) if not bad else "; ".join(bad[:5]))

    # -- half B -------------------------------------------------------------
    declared = set(s06)
    declared |= set(scope6["clause_3_files_changed_at_s06"]["paths"])
    declared |= set(scope6["clause_3_files_changed_at_s06"]["self_referential"])
    drifted = []
    for p, row in led.items():
        if p in declared or row["verdict"] == "discard":
            continue
        a, b = row.get("sha256_post_s05", "-"), row.get("sha256_post_s06", "-")
        if a in ("-", "") or b in ("-", ""):
            continue
        if a != b:
            drifted.append(p)
    add("check_12_s06_changed_nothing_undeclared", not drifted,
        "%d row(s) carry both stamps and every one outside the %d declared "
        "path(s) is byte-identical across S0.6"
        % (sum(1 for r in led.values()
               if r.get("sha256_post_s05", "-") not in ("-", "")
               and r.get("sha256_post_s06", "-") not in ("-", "")), len(declared))
        if not drifted else "changed with no declaration: %r" % drifted[:10])

    # -- half A, strengthened: the transform inverts -------------------------
    # A hash equality proves the file ended where the log says. It does not
    # prove the log DESCRIBES the change. Inverting the recorded edits and
    # recovering sha256_before proves that nothing outside a declared edit
    # moved -- a statement about the whole file, not about five lines.
    bad = []
    for p, e in s06.items():
        full = os.path.join(root, p)
        if not os.path.isfile(full):
            bad.append("%s: absent" % p)
            continue
        try:
            with open(full, "rb") as fh:
                text = fh.read().decode("ascii")
            lines = text.split("\n")
            if text.endswith("\n"):
                lines = lines[:-1]
            idx = e["insert_import"]["after_line"]
            del lines[idx]
            for ed in e["edits"]:
                lines[ed["line"] - 1] = base64.b64decode(
                    ed["original_b64"].encode("ascii")).decode("utf-8")
            back = "\n".join(lines) + ("\n" if e["ends_with_newline"] else "")
            if hashlib.sha256(back.encode("ascii")).hexdigest() != e["sha256_before"]:
                bad.append("%s: inverse does not reproduce sha256_before" % p)
        except Exception as exc:                            # noqa: BLE001
            bad.append("%s: inversion failed: %s" % (p, exc))
    add("check_13_t11_inverts_to_its_prior_bytes", not bad,
        "the recorded inverse of T11 reproduces the pre-S0.6 bytes of all %d "
        "file(s); nothing changed outside a declared edit" % len(s06)
        if not bad else "; ".join(bad[:5]))

    # -- link 5: S0.7 -------------------------------------------------------
    # A removal link is shaped differently from an edit link. For an edit,
    # continuity is "the output hash of one step is the input hash of the
    # next". For a removal there IS no output hash, so the equivalent
    # statement is: the last hash the file ever had is preserved somewhere
    # the ledger can still reach. That is what half A asserts. Half B is the
    # same shape as always -- everything not declared is unchanged.
    m7 = os.path.join(root, "tools", "s0_transform", "s07_discard_manifest.json")
    s7 = os.path.join(root, "tools", "s0_transform", "s07_exit_scope.json")
    if not os.path.isfile(m7):
        add("check_14_s07_link_present", False, "no S0.7 discard manifest")
        return results
    with open(m7, "r", encoding="utf-8") as fh:
        man = json.load(fh)
    with open(s7, "r", encoding="utf-8") as fh:
        scope7 = json.load(fh)
    entries = dict((e["path"], e) for e in man["entries"])

    # -- half A: the chain does not END at a removed file, it is PRESERVED --
    bad = []
    for p, e in entries.items():
        if os.path.isfile(os.path.join(root, p)):
            bad.append("%s: declared removed but still on disk" % p)
            continue
        row = led.get(p)
        if row is None:
            bad.append("%s: removed and has NO LEDGER ROW -- this is exactly "
                       "the row loss S0.7 exists to prevent" % p)
            continue
        if len(row["sha256_pre_s0"]) != 64:
            bad.append("%s: row survives but sha256_pre_s0 was lost" % p)
        if row["sha256_post_s06"] != e["sha256_at_capture"]:
            bad.append("%s: the manifest's captured hash disagrees with the "
                       "last hash the ledger recorded" % p)
        if row["sha256_post_s07"] not in ("-", ""):
            bad.append("%s: absent file carries a post_s07 hash" % p)
    add("check_14_s07_removed_files_keep_their_record", not bad,
        "all %d removed file(s) are absent from disk, still carry a ledger row, "
        "still carry sha256_pre_s0, and hand their last hash to the manifest"
        % len(entries) if not bad else "; ".join(bad[:5]))

    # -- half B: nothing undeclared changed or vanished ---------------------
    declared7 = set(entries)
    declared7 |= set(scope7["clause_3_files_changed_at_s07"]["paths"])
    declared7 |= set(scope7["clause_3_files_changed_at_s07"]["self_referential"])
    drifted = []
    for p, row in led.items():
        if p in declared7:
            continue
        a, b = row.get("sha256_post_s06", "-"), row.get("sha256_post_s07", "-")
        if a in ("-", "") or b in ("-", ""):
            continue
        if a != b:
            drifted.append(p)
    add("check_15_s07_changed_nothing_undeclared", not drifted,
        "every row outside the %d declared path(s) is byte-identical across "
        "S0.7" % len(declared7)
        if not drifted else "changed with no declaration: %r" % drifted[:10])

    # -- link 6: S0.8 -------------------------------------------------------
    # S0.8 adds files and edits its own tooling; it removes nothing and
    # transforms nothing. So it has no "half A" -- there is no transform whose
    # output to check. The whole link IS the "nothing else moved" half, which
    # is the half that is normally easy to omit, and here it is the only one.
    s8 = os.path.join(root, "tools", "s0_transform", "s08_exit_scope.json")
    if os.path.isfile(s8):
        with open(s8, "r", encoding="utf-8") as fh:
            scope8 = json.load(fh)
        d8 = scope8["clause_3_files_changed_at_s08"]
        declared8 = set(d8["paths"]) | set(d8["self_referential"])
        drifted = []
        for p, row in led.items():
            if p in declared8:
                continue
            a, b = row.get("sha256_post_s07", "-"), row.get("sha256_post_s08", "-")
            if a in ("-", "") or b in ("-", ""):
                continue
            if a != b:
                drifted.append(p)
        add("check_16_s08_changed_nothing_undeclared", not drifted,
            "every row outside the %d declared path(s) is byte-identical "
            "across S0.8" % len(declared8)
            if not drifted else "changed with no declaration: %r" % drifted[:10])

    # -- no logged file may fall out of the checks above ---------------------
    # Every check in this file is of the form "for each entry in a log, look
    # up the ledger". A lookup that misses removes the file from its own
    # check and still prints PASS. Doc 7 s3: an exemption that only skips is
    # an exemption that hides. This asserts the lookups cover the logs.
    orphans = []
    for key, logdoc in (("S0.2", s02), ("S0.3", s03)):
        for p in (logdoc or {}):
            if L(p) is None:
                orphans.append("%s:%s" % (key, p))
    add("check_10_every_logged_path_resolves_to_a_ledger_row", not orphans,
        "all %d logged path(s) resolve after the declared moves"
        % (len(s02 or {}) + len(s03 or {}))
        if not orphans else "unresolvable: %r" % orphans[:5])

    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description="S0 transform-chain continuity.")
    ap.add_argument("--root", default=".")
    args = ap.parse_args(argv)
    results = run(args.root)
    width = max(len(n) for n, _, _ in results)
    n_ok = sum(1 for _, ok, _ in results if ok)
    for name, ok, msg in results:
        print("%-*s  %s   %s" % (width, name, "PASS" if ok else "FAIL", msg))
    print("-" * (width + 40))
    print("%d/%d checks passed" % (n_ok, len(results)))
    print("\nCHAIN: %s" % ("CONTINUOUS" if n_ok == len(results) else "BROKEN"))
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
