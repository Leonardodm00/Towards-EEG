#!/usr/bin/env python3
"""
test_s0_7_exit.py -- exit test for stage S0.7 (committed payload removed).

    python3 tools/test_s0_7_exit.py --root .

WHAT S0.7 CLAIMS
----------------
The 22 tracked files carrying verdict 'discard' are removed from the working
tree; the tracked tree is under the 20 MB bound; every removed file's identity
-- its pre-S0 hash, its size and its complete ledger row -- is preserved in a
declared manifest and re-emitted into the ledger; and nothing else was deleted.

WHAT THIS FILE ASSERTS, AND WHAT IT DELIBERATELY DOES NOT
---------------------------------------------------------
That the bound is met by measurement rather than by arithmetic on a document;
that every declared file is gone and NOTHING ELSE is; that the manifest is
complete, well formed, and still agrees with the ledger; and that no removed
file lost its row -- which is finding R-7, the whole reason this sub-step
needed a mechanism rather than an rm.

It does NOT assert anything about .git. D-8 forbids history rewriting, so the
object store is unchanged and a clone still transfers about 82 MB. An exit test
that checked repository size rather than TRACKED TREE size would be permanently
red, which is the state Doc 6 section 4 warns against, and it would be red for
a reason S0.7 is forbidden to fix.

It also does not re-assert chain continuity; that is tools/test_s0_chain.py
checks 14 and 15.

Standard library only, ASCII source, Python 3.8+.
"""

import argparse
import csv
import json
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

HEXDIGITS = set("0123456789abcdef")


def tracked_paths(root):
    out = subprocess.run(["git", "ls-files", "-z"], cwd=root,
                         capture_output=True).stdout
    return set(p.decode("utf-8") for p in out.split(b"\0") if p)


def worktree_files(root):
    """Every file in the working tree, excluding .git and bytecode.

    Deliberately NOT git ls-files. The index only knows about staged paths,
    and TEEG_08's pipeline runs every harness AFTER `git apply` but BEFORE
    `git add -A` -- so a file this sub-step introduces is on disk and invisible
    to the index at exactly the moment the harness runs. A check built on the
    index would pass in the sandbox, where the work is staged, and behave
    differently in Colab, where it is not.

    git ls-files is still the right instrument for check_03, which asks a
    question about the index specifically: what does git expect to exist that
    does not.
    """
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if d not in (".git", "__pycache__")]
        for f in filenames:
            full = os.path.join(dirpath, f)
            if os.path.isfile(full):
                out.append(os.path.relpath(full, root).replace(os.sep, "/"))
    return sorted(out)


def tree_bytes(root):
    """Bytes in the working tree. Excludes .git by construction."""
    return sum(os.path.getsize(os.path.join(root, p))
               for p in worktree_files(root))


def run(root):
    tdir = os.path.join(root, "tools", "s0_transform")
    with open(os.path.join(tdir, "s07_exit_scope.json"), "r", encoding="utf-8") as fh:
        scope = json.load(fh)
    with open(os.path.join(tdir, scope["manifest"].split("/")[-1]), "r",
              encoding="utf-8") as fh:
        man = json.load(fh)
    with open(os.path.join(root, "ledger.csv"), "r", encoding="utf-8",
              newline="") as fh:
        led = dict((r["path"], r) for r in csv.DictReader(fh))

    entries = dict((e["path"], e) for e in man["entries"])
    results = []

    def add(name, ok, msg):
        results.append((name, ok, msg))

    # -- 1: the tree is under the bound, measured --------------------------
    bound = scope["exit_bound_bytes"]
    n = tree_bytes(root)
    ok = n < bound
    add("check_01_tracked_tree_is_under_the_bound", ok,
        "tracked tree is %.2f MB against a bound of %.2f MB, %.2f MB of margin"
        % (n / 1048576.0, bound / 1048576.0, (bound - n) / 1048576.0) if ok
        else "tracked tree is %.2f MB, bound is %.2f MB"
             % (n / 1048576.0, bound / 1048576.0))

    # -- 2: every declared file is gone ------------------------------------
    survivors = [p for p in entries if os.path.isfile(os.path.join(root, p))]
    add("check_02_every_declared_file_is_removed", not survivors,
        "all %d declared file(s) are absent from the working tree" % len(entries)
        if not survivors else "still present: %r" % survivors[:5])

    # -- 3: and NOTHING ELSE is --------------------------------------------
    # The half that is easy to omit. Without it, deleting a 23rd file would
    # still leave checks 1 and 2 green -- the bound would be met even more
    # comfortably, which is the trap.
    tracked = tracked_paths(root)
    vanished = set(p for p in tracked
                   if not os.path.isfile(os.path.join(root, p)))
    undeclared = sorted(vanished - set(entries))
    add("check_03_nothing_undeclared_was_removed", not undeclared,
        "no tracked file is missing except the %d declared" % len(entries)
        if not undeclared else "undeclared removals: %r" % undeclared[:5])

    # -- 4: the manifest is complete and has not shrunk --------------------
    problems = []
    if len(entries) != man["n_entries"]:
        problems.append("manifest lists %d entr(ies) but declares n_entries=%d"
                        % (len(entries), man["n_entries"]))
    declared_n = scope["measured_at_39cd43a_on_21_july_2026"]["discard_rows"]
    if len(entries) != declared_n:
        problems.append("manifest holds %d entr(ies), scope declares %d"
                        % (len(entries), declared_n))
    tot = sum(int(e["row"]["size_bytes"]) for e in entries.values())
    if tot != man["total_bytes"]:
        problems.append("entry sizes sum to %d, manifest declares %d"
                        % (tot, man["total_bytes"]))
    add("check_04_manifest_is_complete", not problems,
        "%d entr(ies) totalling %d bytes (%.2f MB), agreeing with the scope"
        % (len(entries), tot, tot / 1048576.0)
        if not problems else "; ".join(problems[:3]))

    # -- 5: every entry carries a usable identity --------------------------
    # This is equation (7) of TEEG_11: a durable record of (path, hash, size)
    # surviving the file's absence. T12 is NOT invertible, so this record is
    # the only thing standing between S0.7 and permanent loss.
    problems = []
    for p, e in sorted(entries.items()):
        h = e.get("sha256_at_capture", "")
        if len(h) != 64 or any(c not in HEXDIGITS for c in h):
            problems.append("%s: malformed sha256_at_capture" % p)
        pre = e["row"].get("sha256_pre_s0", "")
        if len(pre) != 64 or any(c not in HEXDIGITS for c in pre):
            problems.append("%s: malformed sha256_pre_s0" % p)
        if int(e["row"].get("size_bytes", 0)) <= 0:
            problems.append("%s: non-positive size" % p)
        if "destination" not in e:
            problems.append("%s: no destination field" % p)
    add("check_05_every_entry_carries_a_usable_identity", not problems,
        "all %d entr(ies) carry a well-formed pre-S0 hash, a capture hash, a "
        "positive size and a destination field" % len(entries)
        if not problems else "; ".join(problems[:4]))

    # -- 6: no removed file lost its ledger row (finding R-7) --------------
    problems = []
    for p, e in sorted(entries.items()):
        row = led.get(p)
        if row is None:
            problems.append("%s: REMOVED AND HAS NO LEDGER ROW" % p)
            continue
        if row["sha256_pre_s0"] != e["row"]["sha256_pre_s0"]:
            problems.append("%s: ledger pre_s0 disagrees with the manifest" % p)
        if row["verdict"] != "discard":
            problems.append("%s: row survives with verdict %r" % (p, row["verdict"]))
        if row["sha256_post_s07"] not in ("-", ""):
            problems.append("%s: absent file carries a post_s07 hash %r"
                            % (p, row["sha256_post_s07"][:12]))
    add("check_06_no_removed_file_lost_its_ledger_row", not problems,
        "all %d removed file(s) still carry a ledger row with their pre-S0 "
        "hash; the S0.2 row-loss failure (finding R-7) does not recur"
        % len(entries) if not problems else "; ".join(problems[:4]))

    # -- 7: the ledger cannot be regenerated lossily ------------------------
    # Decision N-21 hinges on build_ledger.py reading the manifest from a
    # FIXED path rather than from a flag. If it were a flag, forgetting it
    # would silently drop 22 rows and the regeneration would look successful.
    body = open(os.path.join(root, "tools", "build_ledger.py"), "r",
                encoding="ascii").read()
    ok = ("s07_discard_manifest.json" in body
          and "removed" in body
          and "--removed" not in body)
    add("check_07_ledger_reads_the_manifest_unconditionally", ok,
        "build_ledger.py reads the manifest from its fixed location, so a "
        "regeneration cannot silently drop the removed rows" if ok
        else "build_ledger.py does not unconditionally read the manifest, or "
             "exposes it as an optional flag")

    # -- 8: no surviving file references a removed one ---------------------
    # The exempt set is READ FROM THE DECLARATION, not hard-coded here. It was
    # hard-coded first, and each run surfaced one more artefact that legitimately
    # names a removed path -- ancestors.json, then phase_hashes.json, then
    # s03_targets.json. Widening a literal set once per failure is how an
    # exemption quietly grows past anyone's intent (trap T-15's family).
    exempt = set(scope["reference_exempt_record_artefacts"]["paths"])
    basenames = dict((os.path.basename(p), p) for p in entries)
    SCANNED = (".py", ".sh", ".toml", ".md", ".json", ".cfg", ".txt",
               ".yaml", ".yml", ".csv")
    problems, used = [], set()
    for p in worktree_files(root):
        if p in entries:
            continue
        full = os.path.join(root, p)
        if not os.path.isfile(full):
            continue
        if os.path.splitext(p)[1].lower() not in SCANNED:
            continue
        try:
            body = open(full, "r", encoding="utf-8", errors="replace").read()
        except Exception:                                   # noqa: BLE001
            continue
        named = [b for b in basenames if b in body]
        if not named:
            continue
        if p in exempt:
            used.add(p)
        else:
            problems.append("%s names removed %s" % (p, named[0]))
    stale = sorted(exempt - used)
    if stale:
        problems.append("exemption(s) declared but never needed: %r" % stale[:4])
    add("check_08_no_surviving_file_references_a_removed_one", not problems,
        "no surviving file names a removed one outside the %d declared record "
        "artefact(s), and every one of those exemptions is actually used"
        % len(exempt) if not problems else "; ".join(problems[:4]))

    # -- 8a: the declaration file's exemption did not widen -----------------
    # Every removed path named in ancestors.json must appear ONLY as a
    # declaration of its own removal -- a 'discards' path, or a
    # superseded/new-infrastructure path. A removed file named anywhere else
    # in the spec would be a live reference wearing a declaration's clothes.
    with open(os.path.join(root, "tools", "ancestors.json"), "r",
              encoding="utf-8") as fh:
        anc_raw = fh.read()
    anc_doc = json.loads(anc_raw)
    legitimate = set()
    for d in anc_doc.get("discards", []):
        legitimate.add(d["path"] if isinstance(d, dict) else d)
    for e in anc_doc.get("new_infrastructure", []):
        legitimate.add(e["path"] if isinstance(e, dict) else e)
    for f in anc_doc.get("superseded_forks", []):
        legitimate.add(f["path"] if isinstance(f, dict) else f)
    # Strip every legitimate declaration, then look for survivors.
    stripped = anc_raw
    for pth in sorted(legitimate, key=len, reverse=True):
        stripped = stripped.replace(pth, "")
    leaked = sorted(b for b in basenames if b in stripped)
    add("check_08a_declaration_file_names_removals_only_as_declarations",
        not leaked,
        "ancestors.json names the removed paths only in its discards, "
        "superseded-fork and new-infrastructure declarations" if not leaked
        else "removed file(s) named outside a declaration: %r" % leaked[:5])

    # -- 9: the manifest and the extension rule agree (finding H-7) --------
    # binary_payload.paths is empty, so membership rests on an extension rule.
    # If a NEW .zip or .pdf appeared, the rule would claim it and the manifest
    # would not. Asserting they agree makes that divergence visible rather
    # than letting the next sub-step act on it.
    anc = anc_doc
    exts = set(x.lower() for x in anc.get("binary_payload", {}).get("extensions", []))
    claimed = set(p for p in worktree_files(root)
                  if os.path.splitext(p)[1].lower() in exts)
    add("check_09_extension_rule_claims_nothing_that_survived", not claimed,
        "the binary_payload extension rule %r claims no surviving tracked "
        "file; the rule and the manifest agree" % sorted(exts) if not claimed
        else "extension rule claims surviving file(s) the manifest does not "
             "list: %r" % sorted(claimed)[:5])

    # -- 10: T12 is registered ---------------------------------------------
    tid = scope["transform_id"]
    ok = tid in anc.get("transforms", {})
    add("check_10_transform_is_registered", ok,
        "%s is registered in ancestors.json; a declaration a tool cannot read "
        "is not a declaration (H-5)" % tid if ok
        else "%s is not in the transform register" % tid)

    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description="S0.7 exit test (payload removal).")
    ap.add_argument("--root", default=".")
    args = ap.parse_args(argv)
    results = run(args.root)
    width = max(len(n) for n, _, _ in results)
    n_ok = sum(1 for _, ok, _ in results if ok)
    for name, ok, msg in results:
        print("%-*s  %s   %s" % (width, name, "PASS" if ok else "FAIL", msg))
    print("-" * (width + 40))
    print("%d/%d checks passed" % (n_ok, len(results)))
    print("\nS0.7 EXIT TEST: %s" % ("PASS" if n_ok == len(results) else "FAIL"))
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
