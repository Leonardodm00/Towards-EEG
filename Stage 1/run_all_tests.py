#!/usr/bin/env python3
"""run_all_tests -- run every S1 smoke test in one call.

RUN
---
    python3 run_all_tests.py

Exit status is 0 only if EVERY suite passes. Use it as the gate before any
bank run, and as the cluster-side step 5 of the HPC verification block.

WHAT IT RUNS
------------
  0. environment report      -- interpreter, numpy/pandas versions
  1. HPC transfer safety     -- pure-ASCII and zero-CR checks on every .py
                                here (class 1 and class 2 corruption both)
  2. import chain            -- every module imports cleanly
  3. py_compile              -- every .py parses
  4. the three smoke suites  -- spine_geometry, syn_uid, truncation_flag

Steps 1-3 run before any suite, because a transfer-corrupted or
non-importable file produces confusing suite failures that look like logic
bugs. Ordered cheapest-first, hardest-to-diagnose-first.

Pure ASCII source (HPC-safe).
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# Modules shipped as part of S1.0 -- this runner owns these.
OWN_MODULES = ("spine_geometry", "syn_uid", "truncation_flag",
               "s1_plots")

# Pre-existing project modules these depend on. They are NOT shipped with the
# S1 bundle (they are yours, and shipping a copy would invite version drift
# against whatever is current in the repository), so they must be placed on
# the same path. They are checked here anyway, because a transfer-corrupted
# dependency breaks the suites in ways that look like an S1.0 bug.
DEPENDENCIES = ("spine_density", "node_classify")

MODULES = DEPENDENCIES[:1] + OWN_MODULES      # spine_density is imported first

SUITES = ("test_spine_geometry.py", "test_syn_uid.py",
          "test_truncation_flag.py", "test_s1_plots.py")

CHECKED_PY = OWN_MODULES + DEPENDENCIES


def resolve_module_path(name):
    """Absolute path of the file `name` would import from, or None.

    Uses the import system rather than looking next to this script, because
    the modules may legitimately live anywhere on sys.path -- in this project
    they sit in a CODE_DIR on Drive, not alongside the runner. Adjacency is
    the wrong question; importability is the right one.
    """
    try:
        import importlib.util
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.origin:
        return None
    return os.path.abspath(spec.origin)


def resolve_script_path(filename):
    """Absolute path of a test script: next to this runner, else on sys.path."""
    here = os.path.join(HERE, filename)
    if os.path.exists(here):
        return here
    for entry in sys.path:
        if not entry:
            continue
        cand = os.path.join(entry, filename)
        if os.path.exists(cand):
            return os.path.abspath(cand)
    return None


def diagnose_missing_scripts(missing):
    """Say what IS there, rather than only what is not.

    'not found on sys.path' is true but useless when the user is certain the
    file is in the folder -- and they are usually right, because the failure
    is almost always a name that differs by a browser-appended suffix, a
    double extension, or a stray space, none of which are visible at a glance
    in a file listing. Print the directory contents and offer close matches so
    the actual difference is on screen.
    """
    if not missing:
        return
    print("\n  --- what is actually in %s ---" % HERE)
    try:
        entries = sorted(os.listdir(HERE))
    except OSError as exc:
        print("      cannot list the directory: %s" % exc)
        return

    py = [e for e in entries if e.lower().endswith(".py")]
    if py:
        for e in py:
            try:
                size = os.path.getsize(os.path.join(HERE, e))
                print("      %-42s %8d bytes" % (e, size))
            except OSError:
                print("      %-42s   (unreadable)" % e)
    else:
        print("      no .py files at all -- is this the right directory?")

    other = [e for e in entries if not e.lower().endswith(".py")][:12]
    if other:
        print("      (%d non-.py entries, e.g. %s)"
              % (len(entries) - len(py), ", ".join(other[:6])))

    try:
        import difflib
        print("\n  --- closest names present, per missing file ---")
        for m in missing:
            near = difflib.get_close_matches(m, entries, n=3, cutoff=0.55)
            if near:
                print("      %-30s -> %s" % (m, ", ".join(near)))
            else:
                print("      %-30s -> nothing similar" % m)
    except ImportError:
        pass

    print("\n  A browser download that already had the name appends ' (1)'")
    print("  before the extension, which does not match. Rename to the exact")
    print("  filename, or re-copy from the bundle.")


def missing_dependencies():
    """Dependencies Python cannot import, in import order."""
    return [d for d in DEPENDENCIES if resolve_module_path(d) is None]


def banner(text):
    print("\n" + "=" * 66)
    print(text)
    print("=" * 66)


def step_environment():
    banner("0. environment")
    print("python      %s" % sys.version.split()[0])
    print("executable  %s" % sys.executable)
    print("cwd         %s" % HERE)
    for mod in ("numpy", "pandas"):
        try:
            m = __import__(mod)
            print("%-11s %s" % (mod, getattr(m, "__version__", "?")))
        except ImportError:
            print("%-11s MISSING" % mod)
            return False
    return True


def step_transfer_safety():
    """Class 1 (non-ASCII) and class 2 (CR bytes) corruption, both checked.

    A byte-range scan for non-ASCII will never find a CRLF line ending,
    because CR is 0x0D -- inside the ASCII range. They are different bugs and
    need separate checks.

    Each file is located the same way Python locates it, so what gets checked
    is the file that will actually be imported -- not a same-named copy
    sitting next to this runner.
    """
    banner("1. HPC transfer safety (ASCII purity + line endings)")
    ok = True
    targets = []
    for name in CHECKED_PY:
        targets.append((name + ".py", resolve_module_path(name)))
    for suite in list(SUITES) + ["run_all_tests.py"]:
        targets.append((suite, resolve_script_path(suite)))

    missing_scripts = []
    for label, path in targets:
        if path is None or not os.path.exists(path):
            print("  MISSING   %-26s (not found on sys.path)" % label)
            missing_scripts.append(label)
            ok = False
            continue
        data = open(path, "rb").read()
        non_ascii = [(i + 1, hex(b)) for i, b in enumerate(data) if b > 127]
        n_cr = data.count(b"\r")
        status = "ok"
        if non_ascii:
            status = "NON-ASCII %s" % (non_ascii[:5],)
            ok = False
        elif n_cr:
            status = "CRLF (%d CR bytes)" % n_cr
            ok = False
        print("  %-9s %-26s (%d bytes)" % (status, label, len(data)))

    diagnose_missing_scripts(missing_scripts)
    return ok


def step_imports():
    banner("2. import chain")
    ok = True
    sys.path.insert(0, HERE)
    for name in MODULES:
        try:
            mod = __import__(name)
            ver = getattr(mod, "MODULE_VERSION", "(no MODULE_VERSION)")
            print("  ok        %-22s %s" % (name, ver))
        except Exception as exc:                    # noqa: BLE001
            print("  FAILED    %-22s %s: %s"
                  % (name, type(exc).__name__, exc))
            ok = False
    return ok


def step_compile():
    """Compile exactly the files this bundle is responsible for.

    Not everything in HERE: when the modules live in a shared CODE_DIR, that
    directory holds the whole project, and a parse error in an unrelated file
    would be reported here as an S1.0 failure.
    """
    banner("3. py_compile")
    files = []
    for name in CHECKED_PY:
        p = resolve_module_path(name)
        if p:
            files.append(p)
    for suite in list(SUITES) + ["run_all_tests.py"]:
        p = resolve_script_path(suite)
        if p:
            files.append(p)
    if not files:
        print("  FAILED    nothing resolved to compile")
        return False
    proc = subprocess.run([sys.executable, "-m", "py_compile"] + files,
                          capture_output=True, text=True)
    if proc.returncode == 0:
        print("  ok        %d file(s) compile" % len(files))
        return True
    print("  FAILED")
    print(proc.stderr)
    return False


def step_suites():
    banner("4. smoke suites")
    # A fresh interpreter does not inherit sys.path entries added at runtime,
    # so a suite launched as a subprocess cannot see modules on Drive unless
    # the live sys.path is forwarded explicitly.
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    results = []
    for suite in SUITES:
        path = resolve_script_path(suite)
        if path is None:
            print("  MISSING   %-26s (not found on sys.path)" % suite)
            results.append((suite, None, 0, 0))
            continue
        proc = subprocess.run([sys.executable, path],
                              capture_output=True, text=True,
                              cwd=os.path.dirname(path), env=env)
        out = proc.stdout
        n_pass = out.count("  PASS  ")
        n_fail = out.count("  FAIL  ")
        ok = proc.returncode == 0
        print("  %-9s %-26s %3d passed, %d failed"
              % ("ok" if ok else "FAILED", suite, n_pass, n_fail))
        if not ok:
            for line in out.splitlines():
                if line.startswith("  FAIL") or line.startswith("FAILED"):
                    print("      %s" % line.strip())
            if proc.stderr.strip():
                print("      stderr: %s" % proc.stderr.strip()[:500])
        results.append((suite, ok, n_pass, n_fail))
    return results


def step_dependencies():
    """Fail fast, and legibly, when a project dependency cannot be imported.

    Reports the resolved path of each one, so a dependency picked up from an
    unexpected place (a stale copy in the notebook cwd shadowing the real one
    on Drive) is visible rather than silent.
    """
    banner("0b. project dependencies")
    missing = missing_dependencies()
    for d in DEPENDENCIES:
        path = resolve_module_path(d)
        if path:
            print("  ok        %-18s %s" % (d, path))
        else:
            print("  MISSING   %-18s (not importable)" % d)
    if missing:
        print("\n  These are pre-existing project modules, not part of the")
        print("  S1 bundle. They must be importable -- either next to this")
        print("  runner, or on PYTHONPATH. From a notebook, forward sys.path:")
        print("      env = dict(os.environ)")
        print("      env['PYTHONPATH'] = os.pathsep.join(p for p in sys.path"
              " if p)")
        print("      subprocess.run([sys.executable, <runner>], env=env, ...)")
        print("  Not importable: %s" % ", ".join(missing))
        return False
    return True


def main():
    print("S1 module test runner")
    steps_ok = []
    steps_ok.append(("environment", step_environment()))
    deps_ok = step_dependencies()
    steps_ok.append(("dependencies", deps_ok))
    if not deps_ok:
        banner("summary")
        print("  STOPPED -- project dependencies not importable (see above).")
        print("  No suite was run; this is not a test failure.")
        # Exit 3, NOT 2: the interpreter itself returns 2 for 'can't open
        # file', so a runner invoked with a wrong path would otherwise be
        # indistinguishable from a genuine missing dependency.
        return 3
    steps_ok.append(("transfer safety", step_transfer_safety()))
    steps_ok.append(("imports", step_imports()))
    steps_ok.append(("py_compile", step_compile()))

    results = step_suites()

    banner("summary")
    all_ok = True
    for name, ok in steps_ok:
        print("  %-9s %s" % ("ok" if ok else "FAILED", name))
        all_ok = all_ok and ok
    total_pass = sum(r[2] for r in results)
    total_fail = sum(r[3] for r in results)
    n_absent = sum(1 for r in results if r[1] is None)
    for suite, ok, n_pass, n_fail in results:
        if ok is None:
            print("  ABSENT    %-26s (file not found -- setup, not a failure)"
                  % suite)
            all_ok = False
            continue
        print("  %-9s %-26s %3d passed, %d failed"
              % ("ok" if ok else "FAILED", suite, n_pass, n_fail))
        all_ok = all_ok and ok
    print("\n  %d checks passed, %d failed, across %d suite(s) run"
          % (total_pass, total_fail, len(results) - n_absent))
    if n_absent:
        print("  %d suite(s) were never run because the file is not there."
              % n_absent)
    print("\n%s" % ("ALL GREEN" if all_ok else "SOMETHING FAILED -- see above"))
    if all_ok:
        return 0
    # Exit 4 when the ONLY problem is absent files: that is a setup problem
    # with a different fix from a real test failure, and conflating the two
    # sends you looking for a bug in code that never ran.
    real_failure = any(r[1] is False for r in results) or not all(
        ok for _, ok in steps_ok if isinstance(ok, bool) and _ != "transfer safety")
    if n_absent and not real_failure:
        return 4
    return 1


if __name__ == "__main__":
    sys.exit(main())
