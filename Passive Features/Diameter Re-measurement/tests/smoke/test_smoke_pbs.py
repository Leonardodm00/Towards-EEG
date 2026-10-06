"""Smoke test for scripts/pbs/build_table.pbs (hpc-git-delivery gate 5).

The job script is run with bash against a fixture: a stub `conda` whose hook
installs a conda() function that activates by prepending an env's bin to PATH
and RETURNS 1 (davinci's binutils hook does exactly that while activating
correctly), a stub python3 that records its argv, a decoy HOME.

Checks
    test_known_answer   the chunk arithmetic: task 7 of 100 over 2000 runs
                        replicates 140-160; the last task of an uneven split
                        stops at N_TOTAL
    test_reference      the real path: activation's non-zero status tolerated,
                        the env's python runs build_table.py with the expected argv
    test_contract       the header lines that must appear ([diam-table] ...)
    test_edge_cases     reserved CODE / ENV_NAME reported and ignored; wrong
                        DIAM_CODE, bad RUN_TAG, index beyond DIAM_N_TASKS, a
                        missing DIAM_CONFIG_JSON and an env that did not
                        activate all refuse with their exit codes
    test_convergence, test_invariants, test_determinism: skipped (shell glue)

Run
    cd "Passive Features/Diameter Re-measurement"
    python tests/smoke/test_smoke_pbs.py

Keep this file pure ASCII (hpc-python-compat).
"""
from __future__ import annotations

import os
import platform
import subprocess
import sys
import tempfile
import time
import traceback
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
PBS = WS / "scripts" / "pbs" / "build_table.pbs"

CONDA_STUB = """#!/bin/bash
if [ "$1" = "shell.bash" ] && [ "$2" = "hook" ]; then
cat <<'EOS'
conda() {
  if [ "$1" = "activate" ]; then
    if [ -d "$FAKE_ENVS/$2/bin" ]; then export PATH="$FAKE_ENVS/$2/bin:$PATH"; fi
    return 1
  fi
  return 0
}
EOS
fi
exit 0
"""

PY_STUB = """#!/bin/bash
echo "STUB-PY: $*" >> "$STUB_LOG"
if [ "$1" = "-c" ]; then echo "[diam-table] numpy 0 scipy 0 Pillow 0 pandas 0"; fi
exit 0
"""


def fixture(tmp):
    t = Path(tmp)
    code = t / "code"
    (code / "scripts").mkdir(parents=True)
    (code / "scripts" / "build_table.py").write_text("# fixture\n")
    bin_dir = t / "bin"
    bin_dir.mkdir()
    (bin_dir / "conda").write_text(CONDA_STUB)
    (bin_dir / "conda").chmod(0o755)
    env_bin = t / "envs" / "spine_env" / "bin"
    env_bin.mkdir(parents=True)
    (env_bin / "python3").write_text(PY_STUB)
    (env_bin / "python3").chmod(0o755)
    home = t / "home"
    home.mkdir()
    return code, bin_dir, t / "envs", home, t / "stub.log", t / "out"


def run(tmp, extra=None, drop=()):
    code, bin_dir, envs, home, log, out = fixture(tmp)
    env = {"PATH": "%s:/usr/bin:/bin" % bin_dir, "HOME": str(home), "FAKE_ENVS": str(envs), "STUB_LOG": str(log),
           "DIAM_CODE": str(code), "DIAM_OUT": str(out), "PBS_ARRAY_INDEX": "7"}
    env.update(extra or {})
    for k in drop:
        env.pop(k, None)
    p = subprocess.run(["bash", str(PBS)], env=env, capture_output=True, text=True, timeout=60)
    calls = log.read_text().splitlines() if log.exists() else []
    return p.returncode, p.stdout, p.stderr, calls, code, out


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    with tempfile.TemporaryDirectory() as tmp:
        rc, out, err, _calls, _code, outdir = run(tmp, {"DIAM_DRYRUN": "1"})
        assert rc == 0, (rc, out, err)
        assert "[diam-table] task 7 of 100: replicates 140-160 of 2000" in out, out
        assert "DRYRUN: python3 scripts/build_table.py run --start 140 --stop 160 --out-dir %s" % outdir in out, out
    with tempfile.TemporaryDirectory() as tmp:
        rc, out, err, *_ = run(tmp, {"DIAM_DRYRUN": "1", "DIAM_N_TOTAL": "10", "DIAM_N_TASKS": "3", "PBS_ARRAY_INDEX": "2"})
        assert rc == 0 and "replicates 8-10 of 10" in out and "--start 8 --stop 10" in out, (rc, out, err)


def test_reference():
    with tempfile.TemporaryDirectory() as tmp:
        rc, out, err, calls, code, outdir = run(tmp, {"DIAM_CONFIG_JSON": __file__, "DIAM_SEED": "11"})
        assert rc == 0, (rc, out, err)
        want = "STUB-PY: scripts/build_table.py run --start 140 --stop 160 --out-dir %s --config-json %s --seed 11" % (outdir, __file__)
        assert calls[-1] == want, calls
        assert "[diam-table] task 7 done" in out and outdir.is_dir(), out


def test_convergence():
    raise unittest.SkipTest("shell glue: no discretisation")


def test_invariants():
    raise unittest.SkipTest("shell glue")


def test_contract():
    with tempfile.TemporaryDirectory() as tmp:
        rc, out, err, *_ = run(tmp)
        for line in ("[diam-table] code ", "[diam-table] out ", "[diam-table] env    spine_env", "[diam-table] python ",
                     "[diam-table] numpy", "[diam-table] commit "):
            assert line in out, (line, out)


def test_determinism():
    raise unittest.SkipTest("shell glue")


def test_edge_cases():
    with tempfile.TemporaryDirectory() as tmp:
        rc, out, err, calls, *_ = run(tmp, {"CODE": "/elsewhere", "ENV_NAME": "sbi_export"})
        assert rc == 0 and "NOTE: CODE is set (/elsewhere) and is IGNORED" in out and "NOTE: ENV_NAME is set" in out
        assert "[diam-table] env    spine_env" in out and calls and calls[-1].startswith("STUB-PY: scripts/build_table.py")
    cases = [({"DIAM_CODE": "/nonexistent"}, 2, "scripts/build_table.py not found"),
             ({"RUN_TAG": "a b"}, 4, "RUN_TAG must be"),
             ({"PBS_ARRAY_INDEX": "100"}, 4, "-J and DIAM_N_TASKS disagree"),
             ({"DIAM_CONFIG_JSON": "/nonexistent.json"}, 3, "DIAM_CONFIG_JSON set but missing"),
             ({"DIAM_ENV": "no_such_env"}, 5, "not active")]
    for extra, code, msg in cases:
        with tempfile.TemporaryDirectory() as tmp:
            rc, out, err, calls, *_ = run(tmp, extra)
            assert rc == code and msg in err, (extra, rc, out, err)
            assert not any("build_table.py run" in c for c in calls), (extra, calls)


# ---------------------------------------------------------------- runner ---

def main():
    checks = [(name, obj) for name, obj in globals().items() if name.startswith("test_") and callable(obj)]
    print("== %s" % Path(__file__).name)
    print("python %s | %s" % (platform.python_version(), platform.platform()))
    results = []
    for name, func in checks:
        start = time.perf_counter()
        try:
            func()
            status, detail = "PASS", ""
        except unittest.SkipTest as exc:
            status, detail = "SKIP", str(exc)
        except NotImplementedError as exc:
            status, detail = "TODO", str(exc)
        except AssertionError as exc:
            status, detail = "FAIL", str(exc)
        except Exception:
            status, detail = "ERROR", traceback.format_exc()
        results.append((name, status, time.perf_counter() - start, detail))
    width = max([len(n) for n, _, _, _ in results] + [4])
    for name, status, seconds, detail in results:
        lines = detail.strip().splitlines()
        print(("%-5s  %-" + str(width) + "s  %8.3fs  %s") % (status, name, seconds, lines[0] if lines else ""))
    for name, status, _s, detail in results:
        if status in ("FAIL", "ERROR"):
            print("\n---- %s: %s\n%s" % (status, name, detail.strip()))
    counts = {s: sum(1 for r in results if r[1] == s) for s in ("PASS", "FAIL", "ERROR", "TODO", "SKIP")}
    print("\n-- " + ", ".join("%d %s" % (n, s.lower()) for s, n in counts.items()))
    return 1 if (counts["FAIL"] + counts["ERROR"] + counts["TODO"]) else 0


if __name__ == "__main__":
    sys.exit(main())
