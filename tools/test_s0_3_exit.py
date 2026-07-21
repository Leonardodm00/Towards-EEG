#!/usr/bin/env python3
"""
test_s0_3_exit.py -- the exit test for stage S0.3.

    python3 tools/test_s0_3_exit.py --root .

The roadmap's stated exit is "byte scan clean; no CRLF". Necessary, and not
sufficient: a transliteration that achieved pure ASCII by mangling syntax would
pass it. Deleting every non-ASCII byte would pass it too. So this test asserts
three things the byte scan does not:

  * every swept .py still PARSES and COMPILES (except the S0.2 O9 exemptions);
  * every swept .sh still passes `bash -n`;
  * the sweep introduced no new SyntaxWarning -- the E-11 latent defect must be
    carried forward untouched, not accidentally fixed or worsened, because
    fixing it would be a defect fix and S0 fixes no defects.

And, as with the S0.2 compile exemptions, every file permitted to retain
non-ASCII must match a declared reason. An exclusion list nobody checks is a
list of things that stopped being true.

Standard library only, ASCII source, Python 3.8+.
"""

import argparse
import ast
import fnmatch
import json
import os
import py_compile
import subprocess
import sys
import tempfile
import warnings

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from s0_paths import Resolver  # noqa: E402


def tracked(root):
    out = subprocess.run(["git", "ls-tree", "-r", "--name-only", "HEAD"],
                         cwd=root, capture_output=True, text=True)
    return out.stdout.splitlines()


_RESOLVERS = {}


def resolved(root, rel):
    """Filesystem path of a path AS RECORDED IN THE S0.3 LOG.

    S0.4 moves nine of the swept files. The log keeps naming them by their
    pre-move paths, which is correct -- that is where they were when S0.3
    swept them -- so every filesystem access here has to be resolved forward
    through tools/path_moves.json. Without this the checks below do not fail;
    they quietly cover a smaller set and still print PASS.
    """
    R = _RESOLVERS.get(root)
    if R is None:
        R = _RESOLVERS[root] = Resolver(root)
    return R.full(rel)


def read(root, rel):
    with open(resolved(root, rel), "rb") as fh:
        return fh.read()


def matches(path, globs):
    return any(fnmatch.fnmatch(path, g) or fnmatch.fnmatch(os.path.basename(path), g)
               for g in globs)


def count_syntax_warnings(src, name):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            compile(src, name, "exec")
        except SyntaxError:
            return None
        return sum(1 for x in w if issubclass(x.category, SyntaxWarning))


def run(root):
    with open(os.path.join(root, "tools", "s0_transform", "s03_exit_scope.json"),
              "r", encoding="utf-8") as fh:
        scope = json.load(fh)
    with open(os.path.join(root, "tools", "s0_transform", "s03_transform_log.json"),
              "r", encoding="utf-8") as fh:
        log = json.load(fh)

    files = [p for p in tracked(root) if os.path.isfile(os.path.join(root, p))]
    in_scope = [p for p in files if matches(p, scope["must_be_ascii_and_lf"]["globs"])]
    swept = [e["path"] for e in log["files"]]
    exempt = set(scope["compile_exempt_inherited_from_s02"])
    results = []

    def check_01_scope_is_pure_ascii():
        bad = []
        for p in in_scope:
            data = read(root, p)
            if any(b > 127 for b in data):
                bad.append(p)
        if bad:
            return False, "%d file(s) still carry non-ASCII: %r" % (len(bad), bad[:5])
        return True, "%d in-scope file(s) are pure ASCII" % len(in_scope)

    def check_02_scope_is_lf_only():
        bad = [p for p in in_scope if b"\r\n" in read(root, p)]
        if bad:
            return False, "CRLF remains in %r" % bad[:5]
        return True, "%d in-scope file(s) are LF-only" % len(in_scope)

    def check_03_swept_python_still_parses():
        bad = []
        for p in swept:
            if not p.endswith(".py"):
                continue
            try:
                ast.parse(read(root, p).decode("utf-8"))
            except (SyntaxError, UnicodeDecodeError) as exc:
                bad.append("%s: %s" % (p, exc))
        n = sum(1 for p in swept if p.endswith(".py"))
        if bad:
            return False, "; ".join(bad[:3])
        return True, "%d/%d swept .py still parse" % (n, n)

    def check_04_swept_python_still_compiles():
        fd, cfile = tempfile.mkstemp(suffix=".pyc")
        os.close(fd)
        bad = []
        try:
            for p in swept:
                if not p.endswith(".py") or p in exempt:
                    continue
                try:
                    py_compile.compile(resolved(root, p), cfile=cfile, doraise=True)
                except Exception:                      # noqa: BLE001
                    bad.append(p)
        finally:
            os.unlink(cfile)
        n = sum(1 for p in swept if p.endswith(".py") and p not in exempt)
        if bad:
            return False, "compile broken by the sweep: %r" % bad
        return True, "%d/%d non-exempt swept .py compile" % (n, n)

    def check_05_swept_shell_still_parses():
        sh = [p for p in swept if p.endswith(".sh")]
        if not sh:
            return True, "no .sh in the swept set"
        bad = []
        for p in sh:
            r = subprocess.run(["bash", "-n", resolved(root, p)],
                               capture_output=True, text=True)
            if r.returncode != 0:
                bad.append("%s: %s" % (p, r.stderr.strip()[:80]))
        if bad:
            return False, "; ".join(bad)
        return True, "%d/%d swept .sh pass bash -n" % (len(sh), len(sh))

    def check_06_no_new_syntax_warnings():
        """E-11 is a latent defect carried forward, not touched by S0.3."""
        import base64
        worse = []
        for e in log["files"]:
            if not e["path"].endswith(".py"):
                continue
            after = read(root, e["path"])
            lines = after.split(b"\n")
            before = list(lines)
            for ed in e["edits"]:
                orig = base64.b64decode(ed["original_b64"])
                orig = orig[:-2] if orig.endswith(b"\r\n") else orig.rstrip(b"\n")
                before[ed["line"] - 1] = orig
            try:
                n_before = count_syntax_warnings(b"\n".join(before).decode("utf-8"),
                                                 e["path"])
                n_after = count_syntax_warnings(after.decode("utf-8"), e["path"])
            except UnicodeDecodeError:
                continue
            if n_before is None or n_after is None:
                continue
            if n_after != n_before:
                worse.append("%s: %d -> %d" % (e["path"], n_before, n_after))
        if worse:
            return False, "SyntaxWarning count changed: %r" % worse[:5]
        return True, "SyntaxWarning counts unchanged (E-11 carried forward intact)"

    def check_07_out_of_scope_nonascii_is_declared():
        permitted = scope["nonascii_permitted"]
        undeclared = []
        for p in files:
            if p in in_scope:
                continue
            try:
                data = read(root, p)
            except OSError:
                continue
            if not any(b > 127 for b in data):
                continue
            if any(matches(p, [d["glob"]]) for d in permitted):
                continue
            if p.rsplit(".", 1)[-1].lower() in (
                    "zip", "pdf", "png", "jpg", "jpeg", "npy", "pkl",
                    "tif", "tiff", "dat", "hoc", "pyc"):
                continue
            undeclared.append(p)
        if undeclared:
            return False, ("%d file(s) carry non-ASCII with no declared reason: %r"
                           % (len(undeclared), undeclared[:5]))
        return True, ("every out-of-scope file carrying non-ASCII matches a "
                      "declared exclusion (%d rule(s))" % len(permitted))

    def check_08_gitattributes_covers_the_scope():
        ga = os.path.join(root, ".gitattributes")
        if not os.path.isfile(ga):
            return False, ".gitattributes is absent"
        body = read(root, ".gitattributes").decode("ascii")
        missing = [ext for ext in ("*.py", "*.sh", "*.md", "*.json", "*.csv")
                   if ext not in body]
        if missing:
            return False, "no rule for %r" % missing
        if "*.zip   binary" not in body and "*.zip" not in body:
            return False, "binaries are not marked"
        return True, "line endings pinned by the repository, binaries marked"

    for fn in (check_01_scope_is_pure_ascii, check_02_scope_is_lf_only,
               check_03_swept_python_still_parses, check_04_swept_python_still_compiles,
               check_05_swept_shell_still_parses, check_06_no_new_syntax_warnings,
               check_07_out_of_scope_nonascii_is_declared,
               check_08_gitattributes_covers_the_scope):
        try:
            ok, msg = fn()
        except Exception as exc:                       # noqa: BLE001
            ok, msg = False, "raised %s: %s" % (type(exc).__name__, exc)
        results.append((fn.__name__, ok, msg))
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description="Exit test for stage S0.3.")
    ap.add_argument("--root", default=".")
    args = ap.parse_args(argv)
    results = run(args.root)
    width = max(len(n) for n, _, _ in results)
    n_ok = sum(1 for _, ok, _ in results if ok)
    for name, ok, msg in results:
        print("%-*s  %s   %s" % (width, name, "PASS" if ok else "FAIL", msg))
    print("-" * (width + 40))
    print("%d/%d checks passed" % (n_ok, len(results)))
    print("\nS0.3 EXIT TEST: %s" % ("PASS" if n_ok == len(results) else "FAIL"))
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
