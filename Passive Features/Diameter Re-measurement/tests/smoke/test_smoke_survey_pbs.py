"""Smoke test for the survey jobs on davinci -- scripts/pbs/survey_prepare.pbs,
survey_cells.pbs, survey_pool.pbs and survey_common.sh (Block 16.4 in
specs/SPEC.md; D-099; hpc-git-delivery gate 5).

The job scripts run with bash against a fixture laid out as the repository
(<repo>/Passive Features/Diameter Re-measurement/scripts/..., <repo>/specs/
roadmap/hpc_status.sh): a stub `conda` whose hook installs a conda() that
activates by prepending an env's bin to PATH and RETURNS 1 (davinci's
binutils hook does that while activating correctly), a stub python3 that
records its argv (and fails the API probe on demand), a stub `module`, a
stub roadmap helper that records roadmap_job_start / roadmap_job_end, a
decoy HOME, a selection of three cells.

Checks
    test_known_answer   survey_cells.pbs: task 7 runs `--task-index 7` with
                        `--cache-per-cell` and the default steps; DIAM_CELLS
                        gives `--cells` and no task index, and a task > 0
                        then does nothing; DIAM_FORCE, DIAM_INCLUDE_SET_ASIDE,
                        DIAM_CONFIG_JSON, DIAM_STEPS reach the argv;
                        DIAM_N_TASKS defaults to the selection's rows
    test_reference      the real path of each script: activation's non-zero
                        status tolerated, `module load proxy` before the API
                        probe, the probe before the work, the env's python
                        runs the expected commands in order; the roadmap
                        helper's start and end (exit 0) around the work; the
                        pool job's archive holds status.csv, s3a-s3c, the
                        cells' align and nodes files
    test_contract       the header lines; a dry run prints the commands and
                        starts no python; DIAM_PROXY_MODULE empty loads no
                        module; a failed probe exits 6 before the runner;
                        every script is pure ASCII, LF, bash -n clean, and
                        carries the marker lines where the yaml lists it
    test_edge_cases     reserved CODE / ENV_NAME reported and ignored; a wrong
                        DIAM_CODE (2), a missing selection or config (3), a
                        bad RUN_TAG, a task beyond the selection, -J against
                        DIAM_N_TASKS, bad DIAM_CELLS or DIAM_POOL (4), an env
                        that did not activate (5), a failed analysis in the
                        pool job (1, the others still run)
    test_convergence, test_invariants, test_determinism: skipped (shell glue)

Run
    cd "Passive Features/Diameter Re-measurement"
    python tests/smoke/test_smoke_survey_pbs.py

Keep this file pure ASCII (hpc-python-compat).
"""
from __future__ import annotations

import os
import platform
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
PBS_DIR = WS / "scripts" / "pbs"
SCRIPTS = ("survey_common.sh", "survey_prepare.pbs", "survey_cells.pbs", "survey_pool.pbs")

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
if [ "$1" = "-c" ]; then
  case "$2" in
    *api_query*) if [ "${STUB_API:-ok}" = "fail" ]; then echo "no route" >&2; exit 1; fi; echo "[probe] ok" ;;
    *) echo "[stub] numpy 0 scipy 0 Pillow 0 pandas 0" ;;
  esac
  exit 0
fi
case "$*" in
  *"--what ${STUB_FAIL_POOL:-none}"*) exit 3 ;;
esac
exit 0
"""

MODULE_STUB = """#!/bin/bash
echo "STUB-MODULE: $*" >> "$STUB_LOG"
exit ${STUB_MODULE_RC:-0}
"""

ROADMAP_STUB = """# stub of specs/roadmap/hpc_status.sh: records the calls
roadmap_job_start() { echo "STUB-ROADMAP: start stage=${ROADMAP_STAGE:-} n=${ROADMAP_N_TASKS:-}" >> "$STUB_LOG"; return 0; }
roadmap_job_end() { echo "STUB-ROADMAP: end $1" >> "$STUB_LOG"; return 0; }
"""

SELECTION = ("specimen_id,species,layer,dendrite_type,reconstruction_type,area,donor_id,hemisphere,normalized_depth,"
             "neuron_class,group,rank\n"
             + "".join("%d,Homo Sapiens,3,spiny,full,MTG,9,left,0.3,exc,exc|3,%d\n" % (101 * (k + 1), k)
                       for k in range(9)))                    # nine cells: rows 0-8


# --------------------------------------------------------------- fixture ---

def fixture(tmp, with_roadmap=True):
    t = Path(tmp)
    repo = t / "repo"
    code = repo / "Passive Features" / "Diameter Re-measurement"
    (code / "scripts" / "pbs").mkdir(parents=True)
    for name in SCRIPTS:
        (code / "scripts" / "pbs" / name).write_text((PBS_DIR / name).read_text())
    for name in ("survey_cells.py", "cell_pool.py", "projection_check.py"):
        (code / "scripts" / name).write_text("# fixture\n")
    (code / "src").mkdir()
    if with_roadmap:
        (repo / "specs" / "roadmap").mkdir(parents=True)
        (repo / "specs" / "roadmap" / "hpc_status.sh").write_text(ROADMAP_STUB)
    bin_dir = t / "bin"
    bin_dir.mkdir()
    for name, text in (("conda", CONDA_STUB), ("module", MODULE_STUB)):
        (bin_dir / name).write_text(text)
        (bin_dir / name).chmod(0o755)
    env_bin = t / "envs" / "spine_env" / "bin"
    env_bin.mkdir(parents=True)
    (env_bin / "python3").write_text(PY_STUB)
    (env_bin / "python3").chmod(0o755)
    (t / "home").mkdir()
    survey = t / "survey"
    survey.mkdir()
    (survey / "selection.csv").write_text(SELECTION)
    return dict(repo=repo, code=code, bin=bin_dir, envs=t / "envs", home=t / "home", log=t / "stub.log",
                survey=survey, cache=t / "cache")


def run(tmp, script, extra=None, drop=(), with_roadmap=True, task="7"):
    f = fixture(tmp, with_roadmap)
    env = {"PATH": "%s:/usr/bin:/bin" % f["bin"], "HOME": str(f["home"]), "FAKE_ENVS": str(f["envs"]),
           "STUB_LOG": str(f["log"]), "DIAM_CODE": str(f["code"]), "DIAM_SURVEY": str(f["survey"]),
           "DIAM_CACHE": str(f["cache"]), "PBS_ARRAY_INDEX": task, "ROADMAP_DISABLE": "1"}
    env.update(extra or {})
    for k in drop:
        env.pop(k, None)
    p = subprocess.run(["bash", str(f["code"] / "scripts" / "pbs" / script)], env=env, capture_output=True, text=True,
                       timeout=120, cwd=tmp)
    calls = f["log"].read_text().splitlines() if f["log"].exists() else []
    return p.returncode, p.stdout, p.stderr, calls, f


def py_calls(calls):
    return [c[len("STUB-PY: "):] for c in calls if c.startswith("STUB-PY: ")]


def runner_call(calls):
    hits = [c for c in py_calls(calls) if c.startswith("scripts/survey_cells.py run ")]
    assert len(hits) == 1, hits
    return hits[0].split(" ")


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    with tempfile.TemporaryDirectory() as tmp:
        rc, out, err, calls, f = run(tmp, "survey_cells.pbs", {"DIAM_N_TASKS": "8"})
        assert rc == 0, (out, err)
        argv = runner_call(calls)
        assert argv[argv.index("--task-index") + 1] == "7" and "--cache-per-cell" in argv, argv
        assert argv[argv.index("--cache-dir") + 1] == str(f["cache"]) and argv[argv.index("--out-dir") + 1] == str(f["survey"])
        assert argv[argv.index("--steps") + 1] == "align,nodes,radius,registration,camera,calibration", argv
        assert "--cells" not in argv and "--force" not in argv and "--include-set-aside" not in argv and "--config-json" not in argv
        assert "task 7 of 8 (row 7 of 9)" in out, out
    # the default DIAM_N_TASKS is the selection's rows (9): task 8 runs, task 9 is refused (edge cases)
    with tempfile.TemporaryDirectory() as tmp:
        rc, out, err, calls, f = run(tmp, "survey_cells.pbs", task="8")
        assert rc == 0 and "task 8 of 9 (row 8 of 9)" in out, (rc, out, err)
    with tempfile.TemporaryDirectory() as tmp:
        cfg = os.path.join(tmp, "cfg.json")
        open(cfg, "w").write("{}")
        rc, out, err, calls, f = run(tmp, "survey_cells.pbs", {"DIAM_CELLS": "508305938,576084692", "DIAM_FORCE": "align",
                                                               "DIAM_INCLUDE_SET_ASIDE": "1", "DIAM_CONFIG_JSON": cfg,
                                                               "DIAM_STEPS": "align,nodes"}, task="0")
        assert rc == 0, (out, err)
        argv = runner_call(calls)
        assert argv[argv.index("--cells") + 1] == "508305938,576084692" and "--task-index" not in argv, argv
        assert argv[argv.index("--force") + 1] == "align" and "--include-set-aside" in argv, argv
        assert argv[argv.index("--config-json") + 1] == cfg and argv[argv.index("--steps") + 1] == "align,nodes", argv
        assert "(forced: align) (set-aside cells included)" in out, out
        assert "STUB-ROADMAP: start stage=S3a n=1" in calls, calls
    with tempfile.TemporaryDirectory() as tmp:
        rc, out, err, calls, f = run(tmp, "survey_cells.pbs", {"DIAM_CELLS": "508305938"}, task="1")
        assert rc == 0 and "nothing to do for task 1" in out and not py_calls(calls), (out, calls)


def test_reference():
    # survey_cells.pbs: the order module -> activation check -> roadmap start -> probe -> runner -> roadmap end
    with tempfile.TemporaryDirectory() as tmp:
        rc, out, err, calls, f = run(tmp, "survey_cells.pbs", task="1")
        assert rc == 0, (out, err)
        kinds = []
        for c in calls:
            if c.startswith("STUB-MODULE: load proxy"):
                kinds.append("module")
            elif c.startswith("STUB-PY: -c import numpy"):
                kinds.append("versions")
            elif c.startswith("STUB-PY: -c import sys; sys.path.insert(0, 'src'); import allen_image_io"):
                kinds.append("probe")
            elif c.startswith("STUB-PY: scripts/survey_cells.py run"):
                kinds.append("runner")
            elif c.startswith("STUB-ROADMAP: start"):
                kinds.append("rm-start")
            elif c.startswith("STUB-ROADMAP: end 0"):
                kinds.append("rm-end")
        assert kinds == ["versions", "rm-start", "module", "probe", "runner", "rm-end"], kinds
        assert "STUB-ROADMAP: start stage=S3a n=9" in calls, calls
        assert "python %s" % (f["envs"] / "spine_env" / "bin" / "python3") in out, out
        assert "api.brain-map.org" not in err
    # survey_prepare.pbs: pool, check, select, projections, then the -J line
    with tempfile.TemporaryDirectory() as tmp:
        rc, out, err, calls, f = run(tmp, "survey_prepare.pbs", {"RUN_TAG": "v1"})
        assert rc == 0, (out, err)
        cmds = [c for c in py_calls(calls) if c.startswith("scripts/")]
        assert [c.split(" ")[1] if c.startswith("scripts/cell_pool.py") else "proj" for c in cmds] == \
            ["pool", "check", "select", "proj"], cmds
        assert "--require-match" in cmds[2] and "--check 529878215" in cmds[3] and "/projections" in cmds[3], cmds
        assert "selection: 9 cells" in out and "qsub -J 0-8 -v RUN_TAG=v1 scripts/pbs/survey_cells.pbs" in out, out
        assert "STUB-MODULE: load proxy" in calls and not any(c.startswith("STUB-ROADMAP") for c in calls)
    # survey_pool.pbs: status, the three pools, the archive
    with tempfile.TemporaryDirectory() as tmp:
        f = fixture(tmp)
        sv = f["survey"]
        for d in ("s3a", "s3b", "s3c", "101/align", "101/nodes", "202/align", "101/calibration"):
            (sv / d).mkdir(parents=True)
        (sv / "status.csv").write_text("specimen_id\n")
        (sv / "s3a" / "coverage.json").write_text("{}")
        (sv / "101" / "align" / "overlay_101.png").write_bytes(b"png")
        (sv / "101" / "nodes" / "pilot_101.csv").write_text("a\n")
        (sv / "202" / "align" / "alignment_202.json").write_text("{}")
        (sv / "101" / "calibration" / "scans_101.json").write_text("{}")
        rows = os.path.join(tmp, "rows.csv")
        open(rows, "w").write("x\n")
        env = {"PATH": "%s:/usr/bin:/bin" % f["bin"], "HOME": str(f["home"]), "FAKE_ENVS": str(f["envs"]),
               "STUB_LOG": str(f["log"]), "DIAM_CODE": str(f["code"]), "DIAM_SURVEY": str(sv), "RUN_TAG": "v1",
               "DIAM_PHANTOM_ROWS": rows, "ROADMAP_DISABLE": "1"}
        p = subprocess.run(["bash", str(f["code"] / "scripts" / "pbs" / "survey_pool.pbs")], env=env,
                           capture_output=True, text=True, timeout=120, cwd=tmp)
        assert p.returncode == 0, (p.stdout, p.stderr)
        calls = f["log"].read_text().splitlines()
        cmds = [c for c in py_calls(calls) if c.startswith("scripts/survey_cells.py")]
        assert [c.split(" ")[1] for c in cmds] == ["status", "pool", "pool", "pool"], cmds
        assert [c.split("--what ")[1].split(" ")[0] for c in cmds[1:]] == ["coverage", "inputs", "calibration"], cmds
        assert "--phantom-rows %s" % rows in cmds[1] and "--phantom-rows" not in cmds[2], cmds
        assert "STUB-ROADMAP: start stage=S3c n=1" in calls and "STUB-ROADMAP: end 0" in calls, calls
        with tarfile.open(sv / "results_v1.tgz") as tf:
            names = sorted(tf.getnames())
        assert "status.csv" in names and "s3a/coverage.json" in names and "101/align/overlay_101.png" in names, names
        assert "101/nodes/pilot_101.csv" in names and "202/align/alignment_202.json" in names, names
        assert not any(n.startswith("101/calibration") for n in names), names


def test_convergence():
    raise unittest.SkipTest("shell glue: nothing converges")


def test_invariants():
    raise unittest.SkipTest("shell glue: no invariant beyond the contract")


def test_contract():
    with tempfile.TemporaryDirectory() as tmp:
        rc, out, err, calls, f = run(tmp, "survey_cells.pbs", {"DIAM_DRYRUN": "1"}, task="1")
        assert rc == 0 and not calls, (out, err, calls)
        for line in ("[diam-survey] task 1 of 9 (row 1 of 9)", "[diam-survey] code    ", "[diam-survey] survey  ",
                     "[diam-survey] cache   ", "[diam-survey] env     spine_env", "[diam-survey] steps   align,nodes",
                     "[diam-survey] config  <default>", "[diam-survey] DRYRUN: python3 scripts/survey_cells.py run"):
            assert line in out, (line, out)
        assert "--task-index 1\n" in out and "--cache-per-cell" in out
    with tempfile.TemporaryDirectory() as tmp:
        rc, out, err, calls, f = run(tmp, "survey_prepare.pbs", {"DIAM_DRYRUN": "1"})
        assert rc == 0 and not calls and out.count("DRYRUN: python3") == 4, (out, calls)
    with tempfile.TemporaryDirectory() as tmp:
        rc, out, err, calls, f = run(tmp, "survey_pool.pbs", {"DIAM_DRYRUN": "1", "DIAM_POOL": "coverage,inputs"})
        assert rc == 0 and not calls and out.count("DRYRUN: python3") == 3 and "DRYRUN: tar" in out, (out, calls)
    # no proxy module when DIAM_PROXY_MODULE is empty; a failed probe exits 6 before the runner
    with tempfile.TemporaryDirectory() as tmp:
        rc, out, err, calls, f = run(tmp, "survey_cells.pbs", {"DIAM_PROXY_MODULE": ""}, task="0")
        assert rc == 0 and not any(c.startswith("STUB-MODULE") for c in calls), calls
    with tempfile.TemporaryDirectory() as tmp:
        rc, out, err, calls, f = run(tmp, "survey_cells.pbs", {"STUB_API": "fail"}, task="0")
        assert rc == 6 and "api.brain-map.org is not reachable" in err, (rc, err)
        assert not any(c.startswith("STUB-PY: scripts/") for c in calls), calls
        assert "STUB-ROADMAP: end 6" in calls, calls
    with tempfile.TemporaryDirectory() as tmp:
        rc, out, err, calls, f = run(tmp, "survey_cells.pbs", {"STUB_MODULE_RC": "1"}, task="0")
        assert rc == 0 and "WARNING: module load proxy failed" in out, (rc, out)
    # without the roadmap helper beside the workstream the job says so and runs
    with tempfile.TemporaryDirectory() as tmp:
        rc, out, err, calls, f = run(tmp, "survey_cells.pbs", task="0", with_roadmap=False)
        assert rc == 0 and "no roadmap marker" in out and not any(c.startswith("STUB-ROADMAP") for c in calls)
    # the files themselves
    for name in SCRIPTS:
        data = (PBS_DIR / name).read_bytes()
        assert all(b < 128 for b in data) and b"\r" not in data, name
        assert data.startswith(b"#!/bin/bash\n"), name
        assert subprocess.run(["bash", "-n", str(PBS_DIR / name)], capture_output=True).returncode == 0, name
    for name, stage in (("survey_cells.pbs", "S3a"), ("survey_pool.pbs", "S3c")):
        text = (PBS_DIR / name).read_text()
        assert "export ROADMAP_STAGE=%s" % stage in text and "roadmap_job_start" in text and \
            "trap 'roadmap_job_end $?' EXIT" in text, name
    text = (PBS_DIR / "survey_cells.pbs").read_text()
    assert "#PBS -J 0-50" in text and "#PBS -N diam_survey\n" in text
    assert "#PBS -N diam_survey_pool" in (PBS_DIR / "survey_pool.pbs").read_text()


def test_determinism():
    raise unittest.SkipTest("shell glue: the stubs record one fixed sequence")


def test_edge_cases():
    with tempfile.TemporaryDirectory() as tmp:
        rc, out, err, calls, f = run(tmp, "survey_cells.pbs", {"CODE": "/elsewhere", "ENV_NAME": "sbi_export"}, task="0")
        assert rc == 0 and "NOTE: CODE is set" in out and "NOTE: ENV_NAME is set" in out, out
        assert "python %s" % (f["envs"] / "spine_env" / "bin" / "python3") in out
    cases = [("survey_cells.pbs", {"DIAM_CODE": "/nonexistent"}, 2, "survey_common.sh not found"),
             ("survey_cells.pbs", {"RUN_TAG": "bad tag"}, 4, "RUN_TAG"),
             ("survey_cells.pbs", {"PBS_ARRAY_INDEX": "9"}, 4, "the selection has 9 cells"),
             ("survey_cells.pbs", {"PBS_ARRAY_INDEX": "3", "DIAM_N_TASKS": "2"}, 4, "disagree"),
             ("survey_cells.pbs", {"DIAM_SELECTION": "/nonexistent.csv"}, 3, "selection missing"),
             ("survey_cells.pbs", {"DIAM_CONFIG_JSON": "/nonexistent.json"}, 3, "DIAM_CONFIG_JSON"),
             ("survey_cells.pbs", {"DIAM_CELLS": "12a"}, 4, "DIAM_CELLS"),
             ("survey_cells.pbs", {"DIAM_ENV": "other_env"}, 5, "not active"),
             ("survey_prepare.pbs", {"DIAM_PILOT": "x"}, 4, "DIAM_PILOT"),
             ("survey_pool.pbs", {"DIAM_POOL": "coverage,typo"}, 4, "DIAM_POOL"),
             ("survey_pool.pbs", {"DIAM_SELECTION": "/nonexistent.csv"}, 3, "selection missing")]
    for script, extra, code, text in cases:
        with tempfile.TemporaryDirectory() as tmp:
            rc, out, err, calls, f = run(tmp, script, extra, task=extra.get("PBS_ARRAY_INDEX", "0"))
            assert rc == code and text in err, (script, extra, rc, err)
            assert not any(c.startswith("STUB-PY: scripts/") for c in calls), (script, extra, calls)
    # a wrong workstream (common.sh present, the runner missing) exits 2
    with tempfile.TemporaryDirectory() as tmp:
        f = fixture(tmp)
        (f["code"] / "scripts" / "survey_cells.py").unlink()
        env = {"PATH": "%s:/usr/bin:/bin" % f["bin"], "HOME": str(f["home"]), "FAKE_ENVS": str(f["envs"]),
               "STUB_LOG": str(f["log"]), "DIAM_CODE": str(f["code"]), "DIAM_SURVEY": str(f["survey"]),
               "PBS_ARRAY_INDEX": "0"}
        p = subprocess.run(["bash", str(f["code"] / "scripts" / "pbs" / "survey_cells.pbs")], env=env,
                           capture_output=True, text=True, timeout=120)
        assert p.returncode == 2 and "survey_cells.py not found" in p.stderr, p.stderr
    # the pool job: a failed analysis is reported, the others run, the archive is made, the exit is 1
    with tempfile.TemporaryDirectory() as tmp:
        f = fixture(tmp)
        (f["survey"] / "status.csv").write_text("specimen_id\n")
        env = {"PATH": "%s:/usr/bin:/bin" % f["bin"], "HOME": str(f["home"]), "FAKE_ENVS": str(f["envs"]),
               "STUB_LOG": str(f["log"]), "DIAM_CODE": str(f["code"]), "DIAM_SURVEY": str(f["survey"]),
               "STUB_FAIL_POOL": "inputs", "ROADMAP_DISABLE": "1"}
        p = subprocess.run(["bash", str(f["code"] / "scripts" / "pbs" / "survey_pool.pbs")], env=env,
                           capture_output=True, text=True, timeout=120, cwd=tmp)
        calls = f["log"].read_text().splitlines()
        cmds = [c for c in py_calls(calls) if c.startswith("scripts/survey_cells.py pool")]
        assert p.returncode == 1 and len(cmds) == 3 and "pool inputs FAILED (exit 3)" in p.stdout, (p.stdout, cmds)
        assert (f["survey"] / "results_default.tgz").exists() and "STUB-ROADMAP: end 1" in calls, calls


# ---------------------------------------------------------------- runner ---

def _environment():
    parts = ["python %s" % platform.python_version(), "bash %s" % subprocess.run(
        ["bash", "-c", "echo $BASH_VERSION"], capture_output=True, text=True).stdout.strip(), platform.platform()]
    return " | ".join(parts)


def main():
    checks = [(name, obj) for name, obj in globals().items() if name.startswith("test_") and callable(obj)]
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
