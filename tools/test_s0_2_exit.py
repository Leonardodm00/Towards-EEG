#!/usr/bin/env python3
"""
test_s0_2_exit.py -- the exit test for stage S0.2.

    python3 tools/test_s0_2_exit.py --root .

Seven assertions. Six are the exit criteria; the seventh is the one that makes
the exemption mechanism trustworthy.

The roadmap states the S0.2 exit as "ast.parse and py_compile succeed on all".
Measurement showed that three targets parse but do not compile, for a reason
that predates S0 and is not S0's to fix: each is a concatenation of several
exported notebooks, carrying `from __future__ import annotations` at interior
positions with none at module head. That is legal per notebook cell and illegal
per module. Under decision B-1 these files are out of the installed package and
are never imported, so py_compile tests a property they are not required to
have. They are therefore exempted -- by name, in
`tools/s0_transform/s02_exit_scope.json`, with a defect id and an onward stage.

AN EXEMPTION THAT ONLY SKIPS IS AN EXEMPTION THAT HIDES.
So check_07 does not take the declaration on trust. For each exempt file it

  (a) confirms the file really does fail py_compile -- an exemption for a file
      that now compiles is stale and must be removed, not left standing;
  (b) neutralises ONLY the interior `from __future__` lines, in memory, and
      confirms the file then compiles.

(b) is the load-bearing half: it proves the declared cause is the ONLY thing
standing between the file and compilation. If some unrelated defect appeared in
an exempt file tomorrow, the exemption would stop covering it and this test
would fail, which is the whole point.

Standard library only, ASCII source, Python 3.8+.
"""

import argparse
import ast
import json
import os
import py_compile
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from s0_transform import decolab as D  # noqa: E402


def _read(path):
    with open(path, "rb") as fh:
        return fh.read()


def _compiles(path, source=None):
    """py_compile on disk, or on substitute source written to a temp file.

    The byte-compiled output goes to a temp path, never to os.devnull:
    py_compile refuses a non-regular cfile, and that refusal is reported as a
    compilation failure, which would silently turn every check below green-red
    for the wrong reason.
    """
    fd, cfile = tempfile.mkstemp(suffix=".pyc")
    os.close(fd)
    src_tmp = None
    try:
        if source is not None:
            fd, src_tmp = tempfile.mkstemp(suffix=".py")
            os.close(fd)
            with open(src_tmp, "wb") as fh:
                fh.write(source)
            path = src_tmp
        try:
            py_compile.compile(path, cfile=cfile, doraise=True)
            return True, ""
        except Exception as exc:                       # noqa: BLE001
            return False, str(exc).strip().splitlines()[-1]
    finally:
        for p_ in (cfile, src_tmp):
            if p_ and os.path.exists(p_):
                os.unlink(p_)


def interior_future_lines(source_bytes):
    """1-based line numbers of `from __future__` statements that are illegal.

    A future import is legal only as the first statement of the module, or
    immediately after a module docstring. Everything else is interior.
    """
    tree = ast.parse(source_bytes.decode("utf-8"))
    body = tree.body
    legal_index = 0
    if (body and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        legal_index = 1
    out = []
    for i, node in enumerate(body):
        if (isinstance(node, ast.ImportFrom) and node.module == "__future__"
                and i != legal_index):
            out.append(node.lineno)
    return out


def neutralise_lines(source_bytes, linenos):
    lines = D.split_lines(source_bytes)
    for ln in linenos:
        content, eol = lines[ln - 1]
        lines[ln - 1] = (b"# exit-test probe: " + content, eol)
    return D.join_lines(lines)


def run(root, log_path, scope_path):
    with open(log_path, "r", encoding="utf-8") as fh:
        log = json.load(fh)
    with open(scope_path, "r", encoding="utf-8") as fh:
        scope = json.load(fh)

    targets = [e["path"] for e in log["files"]]
    exempt = dict((e["path"], e) for e in scope["compile_exempt"])
    results = []

    def check_01_every_target_parses():
        bad = []
        for rel in targets:
            try:
                ast.parse(_read(os.path.join(root, rel)).decode("utf-8"))
            except (SyntaxError, UnicodeDecodeError) as exc:
                bad.append("%s: %s" % (rel, exc))
        if bad:
            return False, "; ".join(bad)
        return True, "ast.parse succeeds on %d/%d targets" % (len(targets), len(targets))

    def check_02_non_exempt_targets_compile():
        bad = []
        for rel in targets:
            if rel in exempt:
                continue
            ok, msg = _compiles(os.path.join(root, rel))
            if not ok:
                bad.append("%s: %s" % (rel, msg))
        n = len(targets) - len(exempt)
        if bad:
            return False, "; ".join(bad)
        return True, "py_compile succeeds on %d/%d non-exempt targets" % (n, n)

    def check_03_discard_is_gone():
        missing = []
        for d in log["discards"]:
            if os.path.exists(os.path.join(root, d["path"])):
                missing.append(d["path"])
        if missing:
            return False, "still present: %r" % missing
        return True, "%d declared discard(s) absent from the tree" % len(log["discards"])

    def check_04_transform_log_verifies():
        problems = D.verify(root, log)
        if problems:
            return False, "%r" % problems[:3]
        return True, "tree matches the transform log; edits invert to the ancestors"

    def check_05_line_numbering_preserved():
        bad = [e["path"] for e in log["files"]
               if e["n_lines_before"] != e["n_lines_after"]]
        if bad:
            return False, "line count changed in %r" % bad
        return True, "line numbering preserved in every target"

    def check_06_hash_chain_stamped():
        import csv
        ledger = os.path.join(root, "ledger.csv")
        with open(ledger, "r", encoding="utf-8", newline="") as fh:
            rows = {r["path"]: r for r in csv.DictReader(fh)}
        bad = []
        for e in log["files"]:
            row = rows.get(e["path"])
            if row is None:
                bad.append("%s: no ledger row" % e["path"])
            elif row["sha256_post_s02"] != e["sha256_after"]:
                bad.append("%s: sha256_post_s02 does not match the transform log"
                           % e["path"])
        for d in log["discards"]:
            row = rows.get(d["path"])
            if row is not None and row["sha256_post_s02"] != "-":
                bad.append("%s: discarded but carries a post_s02 hash" % d["path"])
        if bad:
            return False, "; ".join(bad)
        return True, "ledger sha256_post_s02 agrees with the transform log"

    def check_07_exemptions_are_earned_not_asserted():
        """The self-verifying half. See the module docstring."""
        bad = []
        for rel, decl in exempt.items():
            full = os.path.join(root, rel)
            if not os.path.isfile(full):
                bad.append("%s: exempt file absent" % rel)
                continue
            src = _read(full)
            ok_now, _ = _compiles(full)
            if ok_now:
                bad.append("%s: compiles now -- the exemption is stale and must "
                           "be removed" % rel)
                continue
            if decl["cause"] != "non_leading_future_import":
                bad.append("%s: unknown declared cause %r" % (rel, decl["cause"]))
                continue
            try:
                lines = interior_future_lines(src)
            except (SyntaxError, UnicodeDecodeError) as exc:
                bad.append("%s: no longer parses, so the declared cause cannot "
                           "be the only one: %s" % (rel, exc))
                continue
            if not lines:
                bad.append("%s: declared cause not present in the file" % rel)
                continue
            ok_after, msg = _compiles(full, neutralise_lines(src, lines))
            if not ok_after:
                bad.append("%s: neutralising the %d interior future import(s) "
                           "does NOT make it compile, so the declared cause is "
                           "not the only cause: %s" % (rel, len(lines), msg))
        if bad:
            return False, "; ".join(bad)
        return True, ("%d exemption(s) verified: each fails to compile, and the "
                      "declared cause is provably the sole reason"
                      % len(exempt))

    for fn in (check_01_every_target_parses,
               check_02_non_exempt_targets_compile,
               check_03_discard_is_gone,
               check_04_transform_log_verifies,
               check_05_line_numbering_preserved,
               check_06_hash_chain_stamped,
               check_07_exemptions_are_earned_not_asserted):
        try:
            ok, msg = fn()
        except Exception as exc:                       # noqa: BLE001
            ok, msg = False, "raised %s: %s" % (type(exc).__name__, exc)
        results.append((fn.__name__, ok, msg))
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description="Exit test for stage S0.2.")
    ap.add_argument("--root", default=".")
    ap.add_argument("--log", default=None)
    ap.add_argument("--scope", default=None)
    args = ap.parse_args(argv)

    base = os.path.join(args.root, "tools", "s0_transform")
    log_path = args.log or os.path.join(base, "s02_transform_log.json")
    scope_path = args.scope or os.path.join(base, "s02_exit_scope.json")

    results = run(args.root, log_path, scope_path)
    width = max(len(n) for n, _, _ in results)
    n_ok = sum(1 for _, ok, _ in results if ok)
    for name, ok, msg in results:
        print("%-*s  %s   %s" % (width, name, "PASS" if ok else "FAIL", msg))
    print("-" * (width + 40))
    print("%d/%d checks passed" % (n_ok, len(results)))
    if n_ok == len(results):
        print("\nS0.2 EXIT TEST: PASS")
        return 0
    print("\nS0.2 EXIT TEST: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
