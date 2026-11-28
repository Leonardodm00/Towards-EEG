"""Smoke test for the parallel survey runner -- Block 63 in specs/SPEC.md (scripts/survey_parallel.py; D-283,
D-284) -- and for Block 16.2's status.csv written by concurrent runs (survey_io.write_status, D-284).

The survey runner is replaced by a stub (written to a temporary folder at test time) that records its arguments,
writes a done marker per requested step for each of its cells through survey_io, writes status.csv, and can be told
to sleep, to hang, or to exit 2. The checks need no network and no Allen data.

Checks, from the strongest oracle to the weakest
    test_known_answer   partition: 7 cells over 3 workers -> [[c1,c4,c7],[c2,c5],[c3,c6]]; 10 workers -> 7 groups
                        of one; 1 worker -> one group
    test_reference      cell_state against a folder built by hand: done, set_aside, failed, pending
    test_convergence    SKIP: not a discretisation
    test_invariants     a run of 7 cells with 3 workers: every cell in exactly one worker's --cells, the arguments
                        passed through to every worker, every cell done, exit 0, 3 worker logs, the line
                        "[parallel] finished: 7 done"
    test_contract       resumable: a second run launches no worker and says "nothing to do"; --cells runs a subset;
                        a cell outside the selection exits 2
    test_determinism    two runs on fresh folders give the workers the same cells
    test_edge_cases     a worker exiting 2 -> exit 1 with the other cells done; a failed cell (errors.log, no
                        marker) is run again; a set-aside cell is skipped unless --include-set-aside; SIGTERM to
                        the runner -> exit 130 and no worker left alive
    test_status_concurrent  two processes writing status.csv 300 times each into one folder raise nothing, leave a
                        readable table, and touch neither another process's temporary nor a file at the shared
                        name status.csv.tmp that every process used before D-284 (with that name the check fails)

Run (from the package root)
    python3 tests/smoke/test_smoke_survey_parallel.py
        Prints the environment and a PASS / FAIL / ERROR / TODO / SKIP table,
        then the full message of every failure. Exit code 1 if any check
        failed, errored or is still TODO.
    python3 -m pytest tests/smoke/test_smoke_survey_parallel.py -q
        The same checks through pytest.

Keep this file pure ASCII so it survives transfer to the cluster.
"""
from __future__ import annotations

import csv
import importlib.metadata
import json
import multiprocessing
import os
import platform
import signal
import subprocess
import sys
import tempfile
import time
import traceback
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
SRC = WS / "src"
RUNNER = WS / "scripts" / "survey_parallel.py"
for p in (str(SRC), str(WS / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from allen_diameter.loading import cell_pool, survey_io  # noqa: E402
import survey_parallel as SP  # noqa: E402

SEED = 12345            # no randomness in the code under test; reported for the template's line
REPORT_PACKAGES = ("numpy",)
STEPS = ("align", "nodes", "calibration", "screening")
CELLS = (508282493, 526785799, 529878215, 531520401, 531523484, 531526539, 537297470)

STUB = r'''
import json, os, sys, time
sys.path.insert(0, os.environ["STUB_SRC"])
from allen_diameter.loading import survey_io
args = sys.argv[1:]
assert args[0] == "run", args


def val(flag):
    return args[args.index(flag) + 1] if flag in args else None


out, cells, steps = val("--out-dir"), [int(c) for c in val("--cells").split(",")], val("--steps").split(",")
calls = os.environ["STUB_CALLS"]
os.makedirs(calls, exist_ok=True)
with open(os.path.join(calls, "%d.json" % os.getpid()), "w") as f:
    json.dump(dict(argv=args, pid=os.getpid()), f)
time.sleep(float(os.environ.get("STUB_HANG", "0")))
exit2 = set(int(c) for c in os.environ.get("STUB_EXIT2_CELLS", "").split(",") if c)
for c in cells:
    cell = survey_io.CellDir(out, c)
    for s in steps:
        if cell.done(s):
            continue
        time.sleep(float(os.environ.get("STUB_SLEEP", "0.01")))
        survey_io.write_marker(cell, s, {"stub": True}, 0.0, "stub")
    survey_io.write_status(out, [dict(specimen_id=c, group="", **dict((s, "done") for s in steps))], steps)
sys.exit(2 if exit2 & set(cells) else 0)
'''


# --------------------------------------------------------------- helpers ---

def make_case(tmp, cells=CELLS):
    """A selection of `cells` (Block 16.1's columns), an empty out_dir, the stub runner; returns paths."""
    tmp = Path(tmp)
    sel = tmp / "selection.csv"
    rows = [dict(specimen_id=c, species="Homo Sapiens", layer="3", dendrite_type="spiny", reconstruction_type="full",
                 area="MTG", donor_id="1", hemisphere="left", normalized_depth=0.3, neuron_class="exc",
                 group="exc|3", rank=0) for c in cells]
    cell_pool.write_table(str(sel), rows, cell_pool.SELECTION_COLUMNS)
    stub = tmp / "stub_runner.py"
    stub.write_text(STUB)
    out = tmp / "out"
    out.mkdir()
    return dict(sel=str(sel), out=str(out), cache=str(tmp / "cache"), stub=str(stub), calls=str(tmp / "calls"),
                cfg=str(tmp / "cfg.json"))


def run_parallel(case, workers=3, extra=(), env=None, poll="0.2", wait=True):
    """survey_parallel.py in a fresh Python against the stub; returns the CompletedProcess (or the Popen)."""
    cmd = [sys.executable, "-u", str(RUNNER), "--workers", str(workers), "--selection", case["sel"],
           "--out-dir", case["out"], "--cache-dir", case["cache"], "--steps", ",".join(STEPS),
           "--config-json", case["cfg"], "--poll-s", poll, "--runner", case["stub"]] + list(extra)
    e = dict(os.environ, STUB_SRC=str(SRC), STUB_CALLS=case["calls"])
    e.update(env or {})
    if not wait:
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=e)
    return subprocess.run(cmd, capture_output=True, text=True, env=e, timeout=120)


def calls_of(case):
    d = Path(case["calls"])
    return [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))] if d.exists() else []


def cells_of(call):
    a = call["argv"]
    return [int(c) for c in a[a.index("--cells") + 1].split(",")]


def all_done(case, cells):
    return all(survey_io.CellDir(case["out"], c).done(s) for c in cells for s in STEPS)


def _write_status_many(out_dir, n, tag):
    for i in range(n):
        survey_io.write_status(out_dir, [dict(specimen_id=i, group=tag, align="done")], ["align"])


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    c = list(range(1, 8))
    assert SP.partition(c, 3) == [[1, 4, 7], [2, 5], [3, 6]], SP.partition(c, 3)
    assert SP.partition(c, 10) == [[x] for x in c], SP.partition(c, 10)
    assert SP.partition(c, 1) == [c]
    assert SP.partition([], 4) == []


def test_reference():
    with tempfile.TemporaryDirectory() as tmp:
        out = tmp
        a, b, f, p = 1, 2, 3, 4
        for s in STEPS:
            survey_io.write_marker(survey_io.CellDir(out, a), s, {}, 0.0, "h")
        cb = survey_io.CellDir(out, b)
        survey_io.write_marker(cb, "align", {}, 0.0, "h")
        cb.save_context(dict(alignment=dict(set_aside=True, reasons=["score"])))
        cf = survey_io.CellDir(out, f)
        survey_io.write_marker(cf, "align", {}, 0.0, "h")
        cf.log_error("nodes", "Traceback ...")
        got = dict((c, SP.cell_state(out, c, STEPS)) for c in (a, b, f, p))
        assert got == {a: "done", b: "set_aside", f: "failed", p: "pending"}, got
        assert not os.path.exists(os.path.join(out, str(p))), "cell_state created a folder"
        assert SP.counts(out, [a, b, f, p], STEPS) == dict(done=1, set_aside=1, failed=1, pending=1)


def test_convergence():
    raise unittest.SkipTest("not a discretisation")


def test_invariants():
    with tempfile.TemporaryDirectory() as tmp:
        case = make_case(tmp)
        r = run_parallel(case, workers=3)
        assert r.returncode == 0, r.stdout + r.stderr
        calls = calls_of(case)
        assert len(calls) == 3, len(calls)
        seen = sorted(c for call in calls for c in cells_of(call))
        assert seen == sorted(CELLS), seen
        for call in calls:
            a = call["argv"]
            for flag, value in (("--selection", case["sel"]), ("--out-dir", case["out"]),
                                ("--cache-dir", case["cache"]), ("--steps", ",".join(STEPS)),
                                ("--config-json", case["cfg"])):
                assert a[a.index(flag) + 1] == value, (flag, a)
            assert "--cache-per-cell" in a and "--include-set-aside" not in a, a
        assert all_done(case, CELLS)
        assert "[parallel] finished: 7 done, 0 set aside, 0 failed, 0 not done; 3 of 3 workers exited 0" in r.stdout, r.stdout
        logs = sorted(os.listdir(os.path.join(case["out"], "logs")))
        assert len(logs) == 3 and all(x.startswith("parallel_") and x.endswith(".log") for x in logs), logs


def test_contract():
    with tempfile.TemporaryDirectory() as tmp:
        case = make_case(tmp)
        r1 = run_parallel(case, workers=2, extra=["--cells", "%d,%d" % (CELLS[1], CELLS[4])])
        assert r1.returncode == 0, r1.stdout + r1.stderr
        assert sorted(c for call in calls_of(case) for c in cells_of(call)) == sorted([CELLS[1], CELLS[4]])
        assert all_done(case, [CELLS[1], CELLS[4]]) and not all_done(case, [CELLS[0]])
        n_calls = len(calls_of(case))
        r2 = run_parallel(case, workers=3)
        assert r2.returncode == 0, r2.stdout + r2.stderr
        assert len(calls_of(case)) == n_calls + 3, "the second run did not take the 5 cells left"
        assert sorted(c for call in calls_of(case)[n_calls:] for c in cells_of(call)) == sorted(
            set(CELLS) - {CELLS[1], CELLS[4]})
        r3 = run_parallel(case, workers=3)
        assert r3.returncode == 0 and "nothing to do" in r3.stdout, r3.stdout
        assert len(calls_of(case)) == n_calls + 3, "a finished run launched workers again"
        r4 = run_parallel(case, workers=3, extra=["--cells", "123"])
        assert r4.returncode == 2 and "not in the selection" in r4.stderr, (r4.returncode, r4.stderr)


def test_determinism():
    got = []
    for _ in range(2):
        with tempfile.TemporaryDirectory() as tmp:
            case = make_case(tmp)
            r = run_parallel(case, workers=3)
            assert r.returncode == 0, r.stdout + r.stderr
            got.append(sorted(tuple(cells_of(c)) for c in calls_of(case)))
    assert got[0] == got[1], got


def test_edge_cases():
    # a worker exiting 2: exit 1, the other workers' cells done
    with tempfile.TemporaryDirectory() as tmp:
        case = make_case(tmp)
        r = run_parallel(case, workers=3, env=dict(STUB_EXIT2_CELLS=str(CELLS[0])))
        assert r.returncode == 1, (r.returncode, r.stdout)
        assert "worker 0 exited 2" in r.stdout, r.stdout
        assert all_done(case, CELLS), "the stub marks its cells before exiting 2"
    # a failed cell is run again; a set-aside cell is skipped unless --include-set-aside
    with tempfile.TemporaryDirectory() as tmp:
        case = make_case(tmp, cells=CELLS[:3])
        cf = survey_io.CellDir(case["out"], CELLS[0])
        survey_io.write_marker(cf, "align", {}, 0.0, "h")
        cf.log_error("nodes", "Traceback ...")
        cs = survey_io.CellDir(case["out"], CELLS[1])
        survey_io.write_marker(cs, "align", {}, 0.0, "h")
        cs.save_context(dict(alignment=dict(set_aside=True, reasons=["score"])))
        r = run_parallel(case, workers=2)
        assert r.returncode == 0, r.stdout + r.stderr
        ran = sorted(c for call in calls_of(case) for c in cells_of(call))
        assert ran == sorted([CELLS[0], CELLS[2]]), ran
        assert "1 set aside" in r.stdout, r.stdout
        r = run_parallel(case, workers=2, extra=["--include-set-aside"])
        assert r.returncode == 0, r.stdout + r.stderr
        last = calls_of(case)[-1]["argv"]
        assert "--include-set-aside" in last and cells_of(calls_of(case)[-1]) == [CELLS[1]], last
    # SIGTERM to the runner (stopping a Colab cell): exit 130, no worker alive
    with tempfile.TemporaryDirectory() as tmp:
        case = make_case(tmp)
        p = run_parallel(case, workers=3, env=dict(STUB_HANG="60"), wait=False)
        t0 = time.time()
        while len(calls_of(case)) < 3 and time.time() - t0 < 30:
            time.sleep(0.1)
        pids = [c["pid"] for c in calls_of(case)]
        assert len(pids) == 3, "the workers did not start"
        p.send_signal(signal.SIGTERM)
        out, _ = p.communicate(timeout=60)
        assert p.returncode == 130, (p.returncode, out)
        assert "stopped: workers terminated" in out, out
        alive = []
        for pid in pids:
            try:
                os.kill(pid, 0)
                alive.append(pid)
            except ProcessLookupError:
                pass
        assert not alive, "workers still alive: %s" % alive


def test_status_concurrent():
    with tempfile.TemporaryDirectory() as out:
        foreign = os.path.join(out, "status.csv.tmp.999999")
        shared = os.path.join(out, "status.csv.tmp")   # the name every process used before D-284
        for path in (foreign, shared):
            with open(path, "w") as f:
                f.write("another process's half-written table\n")
        ctx = multiprocessing.get_context("spawn")
        procs = [ctx.Process(target=_write_status_many, args=(out, 300, "p%d" % k)) for k in range(2)]
        for q in procs:
            q.start()
        for q in procs:
            q.join(120)
        assert [q.exitcode for q in procs] == [0, 0], [q.exitcode for q in procs]
        with open(os.path.join(out, "status.csv")) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1 and rows[0]["group"] in ("p0", "p1") and rows[0]["align"] == "done", rows
        for path in (foreign, shared):
            assert os.path.exists(path) and open(path).read() == "another process's half-written table\n", (
                "another process's temporary was overwritten or moved: %s" % os.path.basename(path))
        left = [x for x in os.listdir(out) if x.startswith("status.csv.tmp.") and x != os.path.basename(foreign)]
        assert not left, left


# ---------------------------------------------------------------- runner ---

def _environment():
    parts = ["python %s" % platform.python_version()]
    for package in REPORT_PACKAGES:
        try:
            parts.append("%s %s" % (package, importlib.metadata.version(package)))
        except importlib.metadata.PackageNotFoundError:
            parts.append("%s (not installed)" % package)
    parts.append(platform.platform())
    parts.append("seed %d" % SEED)
    return " | ".join(parts)


def main():
    checks = [(name, obj) for name, obj in globals().items()
              if name.startswith("test_") and callable(obj)]
    print("== %s" % Path(__file__).name)
    print(_environment())

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
        except Exception:  # unexpected errors are reported, never hidden
            status, detail = "ERROR", traceback.format_exc()
        results.append((name, status, time.perf_counter() - start, detail))

    width = max([len(name) for name, _, _, _ in results] + [4])
    for name, status, seconds, detail in results:
        lines = detail.strip().splitlines()
        headline = lines[0] if lines else ""
        print(("%-5s  %-" + str(width) + "s  %8.3fs  %s") % (status, name, seconds, headline))

    for name, status, _seconds, detail in results:
        if status in ("FAIL", "ERROR"):
            print("\n---- %s: %s\n%s" % (status, name, detail.strip()))

    counts = dict((s, sum(1 for r in results if r[1] == s)) for s in ("PASS", "FAIL", "ERROR", "TODO", "SKIP"))
    print("\n-- " + ", ".join("%d %s" % (n, s.lower()) for s, n in counts.items()))
    failed = counts["FAIL"] + counts["ERROR"] + counts["TODO"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
