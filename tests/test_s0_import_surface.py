#!/usr/bin/env python3
"""
test_s0_import_surface.py -- exit test for stage S0.9 (import surface).

    python3 tests/test_s0_import_surface.py --root .     # plain harness
    python3 -m pytest tests/test_s0_import_surface.py     # pytest (dev dep)

WHAT S0.9 CLAIMS
----------------
The installed package towards_eeg parses, byte-compiles and imports; every
module is pure ASCII; no filesystem path literal sits outside
towards_eeg/config/; no top-level class name is duplicated; no module-level
definition is shadowed; and every tracked file reconstructs byte-identically
from the ledger's recorded hashes. Eight assertions, from the roadmap
(TEEG_03 s3.2).

WHY THREE OF THE EIGHT CARRY A DECLARED EXEMPTION
------------------------------------------------
S0 fixes no defects, so three assertions cannot pass "as stated" over the
inherited tree and instead assert against a declaration that S0 is forbidden
to fix:

  * assertion 3 (import). Two modules are unimportable BY CONSTRUCTION
    (defect O10, ratified N-23): connectivity_buildup.py imports google.colab
    at module scope, usage_example.py uses an unbound module-scope name. They
    are asserted STATICALLY, never imported -- importing the first under Colab
    would block on drive.mount and run the notebook body. Three further
    modules (hybrid/driver, params, population) are importable in principle
    but pull absent scientific packages, because pyproject declares zero
    runtime dependencies by decision. They are handled by a gated policy: an
    import may fail ONLY with ModuleNotFoundError on an undeclared third-party
    package; any other failure is real.

  * assertion 6 (duplicate class). The class Connectomics is defined in three
    keep-verdict modules. De-duplicating is a content operation for S1.

  * assertion 8 (reconstruction). The three self-referential ledger artifacts
    cannot hash to a column they help produce.

Every exemption is read from tools/s0_transform/s09_exit_scope.json and is
self-verifying: a stale exemption FAILS rather than lingers (Doc 7 s3).

DESIGN NOTES
------------
* Every check DEGRADES TO FAIL on bad input; none raises. An exit test that
  crashes returns no verdict on any of its other checks (Doc 6; the recurring
  self-inflicted bug flagged in TEEG_12 s3.3).
* Imports for assertion 3 run in an ISOLATED SUBPROCESS per module: side
  effects (a hung drive.mount, NEURON initialisation) cannot touch this
  process, and the check works against any --root, including a mutation
  harness tmpdir.
* Assertions 6 and 7 reuse analyse.find_duplicate_classes /
  find_shadowed_defs -- the SAME implementation the ledger uses -- so the test
  and the ledger register cannot drift. Assertion 8 reuses
  test_s0_chain.end_of_chain and the S0.7 discard manifest.

Standard library only, ASCII source, Python 3.8+.
"""

import argparse
import ast
import json
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_TOOLS = os.path.join(_ROOT, "tools")
for _p in (_ROOT, _TOOLS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from s0_ledger import analyse, scan          # noqa: E402
import test_s0_chain as CHAIN                 # noqa: E402

PKG = "towards_eeg"

# A child process that imports one module in isolation and reports the outcome
# as a single JSON line. It catches BaseException so it ALWAYS reports rather
# than dying silently; the parent classifies from the structured result, not
# from parsing a traceback.
_IMPORT_CHILD = (
    "import sys, json, importlib\n"
    "root, mod = sys.argv[1], sys.argv[2]\n"
    "sys.path.insert(0, root)\n"
    "try:\n"
    "    importlib.import_module(mod)\n"
    "    print(json.dumps({'ok': True}))\n"
    "except ModuleNotFoundError as e:\n"
    "    print(json.dumps({'ok': False, 'kind': 'ModuleNotFoundError',\n"
    "                      'name': getattr(e, 'name', None) or ''}))\n"
    "except BaseException as e:\n"
    "    print(json.dumps({'ok': False, 'kind': type(e).__name__,\n"
    "                      'msg': str(e)[:200]}))\n"
)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def package_py_files(root):
    """Sorted repo-relative paths of every .py under the installed package."""
    out = []
    base = os.path.join(root, PKG)
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for f in filenames:
            if f.endswith(".py"):
                full = os.path.join(dirpath, f)
                out.append(os.path.relpath(full, root).replace(os.sep, "/"))
    return sorted(out)


def module_name(rel):
    """towards_eeg/io/schema.py -> towards_eeg.io.schema (drop __init__)."""
    mod = rel[:-3].replace("/", ".")
    if mod.endswith(".__init__"):
        mod = mod[:-len(".__init__")]
    return mod


def module_scope_third_party(root, rel, declared_runtime):
    """Top-level packages imported AT MODULE SCOPE that are neither part of
    towards_eeg nor a declared runtime dependency. Used both by the O10
    staleness guard and the absent-dependency staleness guard.
    """
    tree = ast.parse(open(os.path.join(root, rel), "r", encoding="ascii",
                          errors="replace").read())
    tops = set()
    for n in tree.body:
        if isinstance(n, ast.Import):
            for a in n.names:
                tops.add(a.name.split(".")[0])
        elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
            tops.add(n.module.split(".")[0])
    return sorted(t for t in tops
                  if t != PKG and t not in declared_runtime)


def declared_runtime_deps(root):
    """Top-level names in pyproject [project].dependencies. Empty by decision;
    parsed defensively so the gate stays honest if that ever changes.
    tomllib is 3.11+, so fall back to a minimal extraction for 3.8/3.9."""
    p = os.path.join(root, "pyproject.toml")
    if not os.path.isfile(p):
        return set()
    text = open(p, "r", encoding="utf-8", errors="replace").read()
    try:
        import tomllib
        doc = tomllib.loads(text)
        deps = doc.get("project", {}).get("dependencies", []) or []
        return set(_dep_top(d) for d in deps)
    except Exception:                                       # noqa: BLE001
        pass
    # Minimal fallback: find dependencies = [ ... ] and pull quoted names.
    import re
    m = re.search(r"dependencies\s*=\s*\[(.*?)\]", text, re.DOTALL)
    if not m:
        return set()
    return set(_dep_top(s) for s in re.findall(r"['\"]([^'\"]+)['\"]", m.group(1)))


def _dep_top(spec):
    """'numpy>=1.2' -> 'numpy'."""
    for sep in ("==", ">=", "<=", "~=", ">", "<", "!=", "[", " ", ";"):
        spec = spec.split(sep)[0]
    return spec.strip()


# ---------------------------------------------------------------------------
# the eight assertions
# ---------------------------------------------------------------------------

def run(root):
    results = []

    def add(name, ok, msg):
        results.append((name, ok, msg))

    files = package_py_files(root)

    # -- load the S0.9 declaration and the S0.6 blockers it points at -------
    tdir = os.path.join(root, "tools", "s0_transform")
    scope, blockers, scope_err = {}, {}, None
    try:
        with open(os.path.join(tdir, "s09_exit_scope.json"), "r",
                  encoding="utf-8") as fh:
            scope = json.load(fh)
        src = scope["import_surface"]["unimportable_by_construction"]["blockers_source"]
        s06_file = src.split(":", 1)[0]
        with open(os.path.join(root, s06_file), "r", encoding="utf-8") as fh:
            blockers = json.load(fh)["unimportable_by_construction"]["blockers"]
    except Exception as exc:                                # noqa: BLE001
        scope_err = str(exc)

    # ============ assertion 1: parse =======================================
    bad = []
    for rel in files:
        try:
            ast.parse(open(os.path.join(root, rel), "rb").read())
        except Exception as exc:                            # noqa: BLE001
            bad.append("%s: %s" % (rel, exc))
    add("check_1_parse", not bad,
        "all %d package module(s) ast.parse cleanly" % len(files)
        if not bad else "; ".join(bad[:5]))

    # ============ assertion 2: compile =====================================
    import py_compile
    bad = []
    for rel in files:
        try:
            py_compile.compile(os.path.join(root, rel), doraise=True)
        except Exception as exc:                            # noqa: BLE001
            bad.append("%s: %s" % (rel, str(exc)[:80]))
    add("check_2_compile", not bad,
        "all %d package module(s) byte-compile" % len(files)
        if not bad else "; ".join(bad[:5]))

    # ============ assertion 3: import ======================================
    # Two static exemptions (never imported) + one gated dynamic policy.
    try:
        runtime = declared_runtime_deps(root)
        o10 = set()
        absent_decl = {}
        if scope_err:
            raise RuntimeError("s09 scope unreadable: %s" % scope_err)
        imp = scope["import_surface"]
        o10 = set(imp["unimportable_by_construction"]["files"])
        absent_decl = imp["absent_runtime_dependency"]["known_absent_dependency_modules"]
        absent_decl = dict((k, v) for k, v in absent_decl.items()
                           if k != "note")

        problems = []

        # 3a: O10 -- assert statically that each still fails to import for the
        # declared reason. NEVER import them.
        for rel in sorted(o10):
            b = blockers.get(rel)
            full = os.path.join(root, rel)
            if not os.path.isfile(full):
                problems.append("O10 file absent: %s" % rel)
                continue
            if b is None:
                problems.append("no S0.6 blocker declared for %s" % rel)
                continue
            tree = ast.parse(open(full, "r", encoding="ascii",
                                  errors="replace").read())
            body = [n for n in tree.body
                    if not isinstance(n, (ast.Import, ast.ImportFrom,
                                          ast.FunctionDef, ast.AsyncFunctionDef,
                                          ast.ClassDef, ast.Expr))]
            if not body:
                problems.append("%s: no module-scope body; O10 declaration is "
                                "stale (it may now import)" % rel)
            if b["kind"] == "module_scope_import":
                has = any(isinstance(n, ast.ImportFrom) and n.module == b["module"]
                          for n in tree.body)
                if not has:
                    problems.append("%s: no longer imports %s at module scope; "
                                    "O10 stale" % (rel, b["module"]))
            elif b["kind"] == "module_scope_unbound_name":
                bound = set()
                for n in tree.body:
                    for sub in ast.walk(n):
                        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                            bound.add(sub.id)
                        elif isinstance(sub, (ast.Import, ast.ImportFrom)):
                            for a in sub.names:
                                bound.add((a.asname or a.name).split(".")[0])
                used = False
                for n in tree.body:
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                                      ast.ClassDef)):
                        continue
                    for sub in ast.walk(n):
                        if (isinstance(sub, ast.Name)
                                and isinstance(sub.ctx, ast.Load)
                                and sub.id == b["name"]):
                            used = True
                if not (used and b["name"] not in bound):
                    problems.append("%s: %r no longer unbound at module scope; "
                                    "O10 stale" % (rel, b["name"]))
            else:
                problems.append("%s: unknown blocker kind %r" % (rel, b["kind"]))

        # 3b: known absent-dependency modules -- staleness guard, static.
        for rel in sorted(absent_decl):
            full = os.path.join(root, rel)
            if not os.path.isfile(full):
                problems.append("declared absent-dep module absent: %s" % rel)
                continue
            tp = module_scope_third_party(root, rel, runtime)
            if not tp:
                problems.append("%s: imports no undeclared third-party package "
                                "at module scope; it may now import cleanly and "
                                "the absent-dependency declaration is stale" % rel)

        # 3c: everything else must import, or fail ONLY on the gate.
        skipped = []
        static_exempt = set(o10) | set(absent_decl)
        for rel in files:
            if rel in static_exempt:
                continue
            mod = module_name(rel)
            r = subprocess.run([sys.executable, "-c", _IMPORT_CHILD,
                                os.path.abspath(root), mod],
                               capture_output=True, text=True)
            line = (r.stdout or "").strip().splitlines()
            try:
                res = json.loads(line[-1]) if line else {"ok": False,
                                                          "kind": "no_output",
                                                          "msg": r.stderr[:120]}
            except Exception:                               # noqa: BLE001
                res = {"ok": False, "kind": "unparseable",
                       "msg": (r.stdout or r.stderr)[:120]}
            if res.get("ok"):
                continue
            if res.get("kind") == "ModuleNotFoundError":
                missing = (res.get("name") or "").split(".")[0]
                if missing and missing != PKG and missing not in runtime:
                    skipped.append((mod, missing))     # gated: PASS
                    continue
                problems.append("%s: ModuleNotFoundError on %r, which is a "
                                "towards_eeg submodule or a declared runtime "
                                "dependency -- not excusable"
                                % (mod, res.get("name")))
            else:
                problems.append("%s: %s %s"
                                % (mod, res.get("kind"), res.get("msg", "")))

        n_imp = len(files) - len(static_exempt) - len(skipped)
        add("check_3_import", not problems,
            "%d module(s) import; %d exempt static (O10 + heavy sim); %d skipped "
            "on absent optional deps %r"
            % (n_imp, len(static_exempt), len(skipped),
               [s[1] for s in skipped][:6])
            if not problems else "; ".join(problems[:5]))
    except Exception as exc:                                # noqa: BLE001
        add("check_3_import", False, "import assertion could not run: %s" % exc)

    # ============ assertion 4: ASCII =======================================
    # A module fails only if it carries a byte > 0x7F AND no PEP 263 cookie.
    bad = []
    for rel in files:
        data = open(os.path.join(root, rel), "rb").read()
        if not any(b > 127 for b in data):
            continue
        head = data.split(b"\n", 2)[:2]
        cookie = any(b"coding:" in ln or b"coding=" in ln for ln in head)
        if not cookie:
            first = next(i + 1 for i, b in enumerate(data) if b > 127)
            bad.append("%s: non-ASCII byte at offset %d, no cookie" % (rel, first))
    add("check_4_ascii", not bad,
        "all %d package module(s) are pure ASCII" % len(files)
        if not bad else "; ".join(bad[:5]))

    # ============ assertion 5: no path literal outside config/ =============
    try:
        exempt = {}
        for e in scope.get("path_literal_exemptions", {}).get(
                "declared_literals_outside_config", []):
            exempt.setdefault(e["path"], set()).add(e["literal"])
        hits = []
        still_needed = dict((p, set(v)) for p, v in exempt.items())
        for rel in files:
            if "/config/" in rel:
                continue
            tree = ast.parse(open(os.path.join(root, rel), "r",
                                  encoding="ascii", errors="replace").read())
            for n in ast.walk(tree):
                if isinstance(n, ast.Constant) and isinstance(n.value, str):
                    v = n.value
                    looks_path = v.startswith("/") and "/" in v[1:]
                    if not looks_path:
                        continue
                    if v in exempt.get(rel, set()):
                        still_needed.get(rel, set()).discard(v)
                        continue
                    hits.append("%s: %r" % (rel, v[:40]))
        stale = ["%s: %r" % (p, sorted(s)) for p, s in still_needed.items() if s]
        ok = not hits and not stale
        if hits:
            msg = "undeclared path literal(s): %s" % "; ".join(hits[:5])
        elif stale:
            msg = "declared path-literal exemption no longer present: %s" \
                  % "; ".join(stale)
        else:
            msg = ("no path literal outside config/ except the %d declared "
                   "(S0.6) exemption(s)" % sum(len(v) for v in exempt.values()))
        add("check_5_no_path_literal_outside_config", ok, msg)
    except Exception as exc:                                # noqa: BLE001
        add("check_5_no_path_literal_outside_config", False,
            "path-literal sweep could not run: %s" % exc)

    # ============ assertion 6: no duplicate top-level class ================
    try:
        exempt = set(scope.get("duplicate_class_exemption", {})
                     .get("exempt_class_names", []))
        pkg_records = [r for r in scan.scan_tree(root)
                       if r.path.startswith(PKG + "/")]
        dup = analyse.find_duplicate_classes(pkg_records)   # {name: [(path,ln)]}
        undeclared = sorted(n for n in dup if n not in exempt)
        stale = sorted(n for n in exempt if n not in dup)
        ok = not undeclared and not stale
        if undeclared:
            msg = "undeclared duplicate class name(s): %r" % undeclared
        elif stale:
            msg = ("declared duplicate-class exemption(s) no longer duplicated "
                   "(stale): %r" % stale)
        else:
            msg = ("no duplicate top-level class in the package outside the %d "
                   "declared exemption(s) %r" % (len(exempt), sorted(exempt)))
        add("check_6_no_duplicate_top_level_class", ok, msg)
    except Exception as exc:                                # noqa: BLE001
        add("check_6_no_duplicate_top_level_class", False,
            "duplicate-class sweep could not run: %s" % exc)

    # ============ assertion 7: no shadowed module-level def ================
    try:
        pkg_records = [r for r in scan.scan_tree(root)
                       if r.path.startswith(PKG + "/")]
        shadow = analyse.find_shadowed_defs(pkg_records)
        bad = ["%s:%s (L%s)" % (s.path, s.name,
                                ",".join(str(x) for x in s.lines))
               for s in shadow]
        add("check_7_no_shadowed_module_level_def", not bad,
            "no module-level definition is shadowed in the package (O7 stays "
            "resolved)" if not bad else "shadowed (O7 recurrence): %s"
            % "; ".join(bad[:5]))
    except Exception as exc:                                # noqa: BLE001
        add("check_7_no_shadowed_module_level_def", False,
            "shadowed-def sweep could not run: %s" % exc)

    # ============ assertion 8: ledger reconstruction =======================
    # Every tracked, non-discard, non-self-referential file must hash to its
    # end-of-chain column; every discard file must be absent with its last
    # hash preserved in the S0.7 manifest. Reuses test_s0_chain scaffolding.
    try:
        import csv
        self_ref = set(scope.get("ledger_reconstruction_exemptions", {})
                       .get("self_referential", []))
        with open(os.path.join(root, "ledger.csv"), "r", encoding="ascii",
                  newline="") as fh:
            rows = list(csv.DictReader(fh))
        man_path = os.path.join(root, "tools", "s0_transform",
                                "s07_discard_manifest.json")
        manifest = {}
        if os.path.isfile(man_path):
            with open(man_path, "r", encoding="ascii") as fh:
                manifest = dict((e["path"], e)
                                for e in json.load(fh).get("entries", []))
        problems, n_disk, n_removed = [], 0, 0
        for r in rows:
            p = r["path"]
            full = os.path.join(root, p)
            if r["verdict"] == "discard":
                if os.path.isfile(full):
                    problems.append("%s: discard still on disk" % p)
                    continue
                m = manifest.get(p)
                if m is None:
                    problems.append("%s: discard with no manifest entry" % p)
                elif m.get("sha256_at_capture") != r["sha256_post_s06"]:
                    problems.append("%s: manifest hash disagrees with ledger" % p)
                n_removed += 1
                continue
            if p in self_ref:
                continue
            if not os.path.isfile(full):
                # a working-branch alias row (content lives at the ancestor)
                continue
            col, expected = CHAIN.end_of_chain(r)
            if expected is None:
                problems.append("%s: no stamped hash in the chain" % p)
                continue
            if CHAIN.sha256_file(full) != expected:
                problems.append("%s: disk != %s" % (p, col))
                continue
            n_disk += 1
        add("check_8_ledger_reconstruction", not problems,
            "%d on-disk file(s) hash to their end-of-chain column and %d removed "
            "file(s) keep their hash in the manifest; %d self-referential "
            "artifact(s) exempt" % (n_disk, n_removed, len(self_ref))
            if not problems else "; ".join(problems[:5]))
    except Exception as exc:                                # noqa: BLE001
        add("check_8_ledger_reconstruction", False,
            "reconstruction could not run: %s" % exc)

    return results


# ---------------------------------------------------------------------------
# pytest entry points (collectable without any --root)
# ---------------------------------------------------------------------------

def _default_root():
    return _ROOT


_CACHE = {}


def _verdicts(root=None):
    root = root or _default_root()
    key = os.path.abspath(root)
    if key not in _CACHE:
        _CACHE[key] = dict((n, (ok, msg)) for n, ok, msg in run(root))
    return _CACHE[key]


def _check(name):
    ok, msg = _verdicts()[name]
    assert ok, msg


def test_parse():
    _check("check_1_parse")


def test_compile():
    _check("check_2_compile")


def test_import():
    _check("check_3_import")


def test_ascii():
    _check("check_4_ascii")


def test_no_path_literal_outside_config():
    _check("check_5_no_path_literal_outside_config")


def test_no_duplicate_top_level_class():
    _check("check_6_no_duplicate_top_level_class")


def test_no_shadowed_module_level_def():
    _check("check_7_no_shadowed_module_level_def")


def test_ledger_reconstruction():
    _check("check_8_ledger_reconstruction")


# ---------------------------------------------------------------------------
# plain-harness entry point
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="S0.9 import-surface exit test.")
    ap.add_argument("--root", default=".")
    args = ap.parse_args(argv)
    results = run(args.root)
    width = max(len(n) for n, _, _ in results)
    n_ok = sum(1 for _, ok, _ in results if ok)
    for name, ok, msg in results:
        print("%-*s  %s   %s" % (width, name, "PASS" if ok else "FAIL", msg))
    print("-" * (width + 40))
    print("%d/%d checks passed" % (n_ok, len(results)))
    print("\nS0.9 EXIT TEST: %s" % ("PASS" if n_ok == len(results) else "FAIL"))
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
