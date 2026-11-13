"""smoke_noise_run.py -- Block 6 smoke suite (R1-R9) for run_noise_fit.py and
submit_noise_fit.sh.

Reduced sizes (`--sizes smoke`: Block 4's F7 and Block 9's P9), in the
sandbox, without NEURON and without PBS. The archive is Block 3's
writer-like fixture (smoke_noise_windows.write_fixture) in a group folder
whose path holds spaces, as the real one does: an ok cell (2 SS sweeps,
K = 63; 5 LS sweeps; tilts planted at sigma_b = 0.1 mV/s; 1.5 s LS tail), a
refused cell (0.5 s LS tail: no LS tilt can be read, D-114) and a broken cell
(its QC table one row short: Block 3 raises LayoutError). The code runs from
a temporary git repository holding copies of the noise-fit modules, the
driver, the job script and the roadmap helper, with a local bare origin, so
that the resume rule's commits and the roadmap markers stay inside the test.
The job script runs against stubs on PATH: `conda` (activation only),
`qsub` (records its arguments, prints a job id) and `python` (records the
`cell` and `merge` runs of run_noise_fit.py and passes everything else to
the real interpreter).

    python smoke_noise_run.py          # about 8 min on 2 cores; numpy, scipy, statsmodels
    python smoke_noise_run.py R5,R7    # a subset (mutation runs)

Each check prints one PASS/FAIL line; the suite ends with
"SMOKE noise_run: n/9 passed". Pure ASCII, LF. Spec: specs/SPEC.md, Block 6
(test oracles R1-R8; R9, the decided defaults, added with the code).
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"           # as the job script does (D-126); before numpy

import contextlib                    # noqa: E402
import csv                           # noqa: E402
import io                            # noqa: E402
import json                          # noqa: E402
import math                          # noqa: E402
import re                            # noqa: E402
import shutil                        # noqa: E402
import stat                          # noqa: E402
import subprocess                    # noqa: E402
import sys                           # noqa: E402
import tempfile                      # noqa: E402
import time                          # noqa: E402
import traceback                     # noqa: E402
from dataclasses import asdict, replace   # noqa: E402
from pathlib import Path             # noqa: E402
from types import SimpleNamespace    # noqa: E402

import numpy as np                   # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import noise_fit as nf               # noqa: E402
import noise_sweep_fit as nsf        # noqa: E402
import noise_windows as nw           # noqa: E402
import run_noise_fit as rnf          # noqa: E402
import smoke_noise_windows as sw     # noqa: E402  (the fixture writer)

RESULTS = []
GROUP = "L3_exc"
ID_OK, ID_REFUSED, ID_BROKEN, ID_MISSING = 900000101, 900000102, 900000103, 900000104
CHAIN = ("noise_stats.py", "noise_law.py", "noise_windows.py", "noise_cell_stats.py", "noise_fit.py",
         "noise_sweep_fit.py", "run_noise_fit.py", "submit_noise_fit.sh")
# the smoke sizes, written out (Block 4's F7, Block 9's P9), independent of run_noise_fit.sizes
SMOKE_CFG = nf.NoiseFitConfig(M_binding=60, R_iter=40, R_final=60, B_boot=400, k_max=2, max_nfev=6)
SMOKE_STAGE = nsf.SweepStageConfig(B_v=4, M_refit=60)
REQUIRED = ("specimen_id", "group", "status", "reason", "code_commit", "code_dirty", "sizes", "seconds", "host",
            "workers", "job_id", "started_utc", "finished_utc", "versions", "windows", "cell_fit", "sweep_stage",
            "tables")
WINDOWS_REQUIRED = ("t_ss_ms", "K", "K_LS", "start_flags", "ss_tilt_flags", "ls_tilt_flags")
LINE = re.compile(r"^NOISE_CELL (\d+): status=(\w+) reason=(.*) seconds=(\d+)$")
BLOCK4_NAMES = ("noise_sigma_f_mV", "noise_rho_f", "noise_sigma_s_mV", "noise_tau_s_ms", "noise_J", "noise_dof",
                "noise_p", "noise_kappa", "noise_n_solves", "noise_flags")


def report(name, ok, note=""):
    RESULTS.append(bool(ok))
    print("%s  %s  -- %s" % ("PASS" if ok else "FAIL", name, note))
    sys.stdout.flush()


def failed_parts(parts):
    return [k for k, v in parts.items() if not v]


# ---------------------------------------------------------------------------
#  Helpers, written independently of the driver
# ---------------------------------------------------------------------------
def norm(x):
    """x through a JSON round trip, NaN and inf as None (the cell JSON's convention)."""
    def fix(v):
        if isinstance(v, dict):
            return {k: fix(w) for k, w in v.items()}
        if isinstance(v, list):
            return [fix(w) for w in v]
        if isinstance(v, float) and not math.isfinite(v):
            return None
        return v
    return fix(json.loads(json.dumps(x)))


def drop_seconds(x):
    if isinstance(x, dict):
        return {k: drop_seconds(v) for k, v in x.items() if k != "seconds"}
    if isinstance(x, list):
        return [drop_seconds(v) for v in x]
    return x


def strict_json(text):
    """True when `text` parses with NaN / Infinity refused."""
    def refuse(tok):
        raise ValueError(tok)
    try:
        json.loads(text, parse_constant=refuse)
        return True
    except ValueError:
        return False


def cell_lines(text):
    return [ln for ln in text.splitlines() if ln.startswith("NOISE_CELL ")]


def git(repo, *args):
    out = subprocess.run(["git", "-C", str(repo), "-c", "user.name=smoke", "-c", "user.email=smoke@localhost"]
                         + list(args), stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    if out.returncode != 0:
        raise RuntimeError("git %s: %s" % (" ".join(args), out.stderr.strip()))
    return out.stdout.strip()


def run_py(args, cwd, env=None, timeout=1800):
    return subprocess.run([sys.executable] + [str(a) for a in args], cwd=str(cwd), env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, universal_newlines=True, timeout=timeout)


@contextlib.contextmanager
def patched(obj, name, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


def in_process(argv):
    """run_noise_fit.main(argv) in this process: (exit code, stdout)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = rnf.main([str(a) for a in argv])
    return rc, buf.getvalue()


def read_csv(path):
    with open(path, newline="") as fh:
        rd = csv.reader(fh)
        header = next(rd)
        return header, [dict(zip(header, row)) for row in rd]


def same_value(text, value):
    """A CSV field against the value it was written from: '' for None or NaN, floats exactly."""
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return text == ""
    if isinstance(value, bool):
        return text == str(value)
    if isinstance(value, int):
        return text == str(value)
    if isinstance(value, float):
        return text != "" and float(text) == value
    return text == str(value)


# ---------------------------------------------------------------------------
#  The fixture: an archive group, a git repository with the code, the stubs
# ---------------------------------------------------------------------------
class Fixture:
    def __init__(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="smoke_noise_run_"))
        self.memo = {}
        self._repo()
        self._archive()
        self._stubs()

    def _repo(self):
        self.origin = self.tmp / "origin.git"
        self.repo = self.tmp / "repo"
        subprocess.run(["git", "init", "-q", "--bare", str(self.origin)], check=True)
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
        git(self.repo, "checkout", "-q", "-b", "main")
        self.code = self.repo / "Passive Features" / "HPC script" / "Ih Fit"
        self.code.mkdir(parents=True)
        for name in CHAIN:
            shutil.copy2(str(HERE / name), str(self.code / name))
        roadmap = HERE.parents[2] / "specs" / "roadmap" / "hpc_status.sh"
        (self.repo / "specs" / "roadmap").mkdir(parents=True)
        self.has_helper = roadmap.is_file()
        if self.has_helper:
            shutil.copy2(str(roadmap), str(self.repo / "specs" / "roadmap" / "hpc_status.sh"))
        (self.repo / "README.md").write_text("smoke repository\n")
        (self.repo / ".gitignore").write_text(".roadmap/\n__pycache__/\n")
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-q", "-m", "A: the code")
        git(self.repo, "remote", "add", "origin", str(self.origin))
        git(self.repo, "push", "-q", "origin", "main")
        self.commit_A = git(self.repo, "rev-parse", "HEAD")

    def _write_cell(self, sid, seed, tail):
        d = self.group / ("specimen_%d" % sid)
        sw.write_fixture(d, np.random.default_rng(seed), sigma_b=0.1, ls_tail_n=tail)
        meta = json.loads((d / "metadata.json").read_text())
        meta["specimen_id"] = sid
        (d / "metadata.json").write_text(json.dumps(meta, indent=1))
        return d

    def _archive(self):
        self.root = self.tmp / "Human Neurons Fitting"
        self.group = self.root / GROUP
        self._write_cell(ID_OK, 2026, sw.LS_LONG_TAIL_N)
        self._write_cell(ID_REFUSED, 14, sw.LS_TAIL_N)
        d = self._write_cell(ID_BROKEN, 15, sw.LS_LONG_TAIL_N)
        lines = (d / "trace_qc_results.csv").read_text().splitlines()
        (d / "trace_qc_results.csv").write_text("\n".join(lines[:-1]) + "\n")     # one QC row short

    def _stubs(self):
        self.stubs = self.tmp / "stubs"
        (self.stubs / "bin").mkdir(parents=True)
        (self.stubs / "condabase" / "etc" / "profile.d").mkdir(parents=True)
        files = {
            "bin/conda": "#!/bin/bash\n"
                         "if [ \"$1\" = info ] && [ \"$2\" = --base ]; then echo \"$FAKE_CONDA_BASE\"; exit 0; fi\n"
                         "exit 0\n",
            "condabase/etc/profile.d/conda.sh":
                "conda() { case \"$1\" in activate) export CONDA_DEFAULT_ENV=\"$2\" ;; esac; return 0; }\n",
            "bin/qsub": "#!/bin/bash\n"
                        "n=$(cat \"$QSUB_LOG.n\" 2>/dev/null || echo 0); n=$((n + 1)); echo $n > \"$QSUB_LOG.n\"\n"
                        "{ printf 'CALL'; printf ' [%s]' \"$@\"; printf '\\n'; } >> \"$QSUB_LOG\"\n"
                        "if [ $n -eq 1 ]; then echo '4242[].testserver'; else echo '4243.testserver'; fi\n",
            "bin/python": "#!/bin/bash\n"
                          "if [ \"${1##*/}\" = run_noise_fit.py ] && { [ \"$2\" = cell ] || [ \"$2\" = merge ]; }; then\n"
                          "    { printf 'RUN'; printf ' [%s]' \"$@\"; "
                          "printf ' ENV[%s/%s/%s]\\n' \"$OMP_NUM_THREADS\" \"$OPENBLAS_NUM_THREADS\" \"$MKL_NUM_THREADS\"; }"
                          " >> \"$PY_LOG\"\n"
                          "    echo \"NOISE_CELL 0: status=ok reason=stub seconds=0\"\n"
                          "    exit \"${PY_STUB_EXIT:-0}\"\n"
                          "fi\n"
                          "exec \"$REAL_PYTHON\" \"$@\"\n",
        }
        for rel, text in files.items():
            p = self.stubs / rel
            p.write_text(text)
            p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    def sh_env(self, **extra):
        """A clean environment for the job script: the stubs first on PATH, a decoy HOME."""
        env = {k: v for k, v in os.environ.items()
               if not (k.startswith(("NOISE_", "PBS_", "ROADMAP_", "IH_")) or k in ("RUN_TAG", "CELLS", "GROUP", "NCPUS",
                                                                                   "WALLTIME", "SIZES", "DRY_RUN",
                                                                                   "CONDA_ENV"))}
        home = self.tmp / "home"
        home.mkdir(exist_ok=True)
        env.update({"PATH": "%s:%s" % (self.stubs / "bin", env.get("PATH", "")), "HOME": str(home),
                    "FAKE_CONDA_BASE": str(self.stubs / "condabase"), "REAL_PYTHON": sys.executable,
                    "QSUB_LOG": str(self.tmp / "qsub.log"), "PY_LOG": str(self.tmp / "py.log"),
                    "IH_CODE_DIR": str(self.code), "ROADMAP_REPO": str(self.repo)})
        env.update({k: str(v) for k, v in extra.items()})
        return env

    def reset_logs(self):
        for name in ("qsub.log", "qsub.log.n", "py.log"):
            p = self.tmp / name
            if p.exists():
                p.unlink()

    def run_sh(self, env, script=None, timeout=600):
        script = script or (self.code / "submit_noise_fit.sh")
        return subprocess.run(["bash", str(script)], cwd=str(self.tmp), env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, universal_newlines=True, timeout=timeout)

    # --- memoised runs -------------------------------------------------------
    def direct(self):
        """The oracle: Blocks 3, 4 and 9 called directly on the ok cell at the smoke sizes."""
        if "direct" not in self.memo:
            d = str(self.group / ("specimen_%d" % ID_OK))
            t0 = time.time()
            ss = nw.load_ss_windows(d)
            nwin = nw.ss_noise_windows(ss)
            ls = nw.load_ls_prewindows(d)
            lay = nf.CellLayout.from_cell(nwin, ls)
            tg = nf.CellTarget.from_cell(nwin, ls, lay)
            cf = nf.fit_cell(tg, lay, SMOKE_CFG)
            st = nsf.fit_sweeps(cf, tg, lay, SMOKE_CFG, SMOKE_STAGE)
            self.memo["direct"] = SimpleNamespace(ss=ss, nwin=nwin, ls=ls, lay=lay, tg=tg, cf=cf, st=st,
                                                  seconds=time.time() - t0)
        return self.memo["direct"]

    def cell_run(self, key, specimen, workers, extra=()):
        """`cell --sizes smoke` through the repository's driver, in a subprocess."""
        if key not in self.memo:
            out = self.tmp / ("out_" + key)
            t0 = time.time()
            proc = run_py([self.code / "run_noise_fit.py", "cell", "--group-dir", self.group, "--specimen", specimen,
                           "--out", out, "--workers", workers, "--sizes", "smoke"] + list(extra), cwd=self.tmp)
            path = out / "cells" / ("specimen_%d.json" % specimen)
            text = path.read_text() if path.exists() else None
            self.memo[key] = SimpleNamespace(proc=proc, out=out, path=path, text=text,
                                             rec=json.loads(text) if text else None, seconds=time.time() - t0)
        return self.memo[key]


# ---------------------------------------------------------------------------
#  R1-R9
# ---------------------------------------------------------------------------
def _r1(fx):
    """One writer-like cell through `cell`: every field, status ok, cell_fit == fit_cell and sweep_stage ==
    fit_sweeps (through to_dict, without the seconds), with 1 and with 2 workers."""
    D = fx.direct()
    exp_cf, exp_st = drop_seconds(norm(D.cf.to_dict())), drop_seconds(norm(D.st.to_dict()))
    exp_rows, exp_cc = norm(D.st.table_rows()), norm(D.st.cell_columns())
    exp_sizes = norm({"name": "smoke", "noise_fit": asdict(SMOKE_CFG), "sweep_stage": asdict(SMOKE_STAGE)})
    bad, notes = [], []
    for w in (1, 2):
        r = fx.cell_run("ok_w%d" % w, ID_OK, w)
        rec = r.rec or {}
        lines = cell_lines(r.proc.stdout)
        m = LINE.match(lines[0]) if len(lines) == 1 else None
        win = rec.get("windows") or {}
        parts = {
            "exit 0": r.proc.returncode == 0,
            "one NOISE_CELL line, ok": bool(m) and m.group(1) == str(ID_OK) and m.group(2) == "ok" and m.group(3) == "",
            "every field": all(k in rec for k in REQUIRED) and "traceback" not in rec
            and all(k in (rec.get("seconds") or {}) for k in ("windows", "cell_fit", "sweep_stage", "total"))
            and all(k in win for k in WINDOWS_REQUIRED),
            "status ok": rec.get("status") == "ok" and rec.get("reason") == "",
            "strict JSON": r.text is not None and strict_json(r.text),
            "cell_fit == fit_cell": drop_seconds(rec.get("cell_fit")) == exp_cf,
            "sweep_stage == fit_sweeps": drop_seconds(rec.get("sweep_stage")) == exp_st,
            "tables == table_rows, cell_columns": rec.get("tables") == {"cell_columns": exp_cc, "sweep_rows": exp_rows},
            "Block 3 summary": win.get("t_ss_ms") == D.nwin.start.t_ss_ms and win.get("K") == D.nwin.ss.n_windows
            and win.get("K_LS") == len(D.ls.pre) and win.get("start_flags") == list(D.nwin.start.flags)
            and win.get("ss_tilt_flags") == list(D.nwin.tilts.flags) and win.get("ls_tilt_flags") == [],
            "sizes": rec.get("sizes") == exp_sizes,
            "commit, workers, ids": rec.get("code_commit") == fx.commit_A and rec.get("code_dirty") is False
            and rec.get("workers") == w and rec.get("specimen_id") == ID_OK and rec.get("group") == GROUP,
        }
        bad += ["w=%d: %s" % (w, k) for k in failed_parts(parts)]
        if r.proc.returncode != 0:
            notes.append(r.proc.stderr[-400:])
        notes.append("w=%d %.0f s" % (w, r.seconds))
    # the pool: --workers processes, forked (the workers run the modules loaded at the start)
    seen = {}

    class _Probe(Exception):
        pass

    class FakePool:
        def __init__(self, max_workers=None, mp_context=None, **kw):
            seen.update(n=max_workers, ctx=mp_context)
            raise _Probe()
    with patched(rnf, "ProcessPoolExecutor", FakePool):
        try:
            rnf.run_sweep_stage(D.cf, D.tg, D.lay, SMOKE_CFG, SMOKE_STAGE, 3, log=lambda s: None)
        except _Probe:
            pass
    import multiprocessing
    if "fork" in multiprocessing.get_all_start_methods():
        if not (seen.get("n") == 3 and seen.get("ctx") is not None and seen["ctx"].get_start_method() == "fork"):
            bad.append("the pool: 3 workers, forked")
    report("R1 a writer-like cell through `cell --sizes smoke` with 1 and 2 workers: exit 0, one NOISE_CELL line, every "
           "field, status ok, strict JSON; cell_fit == fit_cell, sweep_stage == fit_sweeps, tables == table_rows / "
           "cell_columns (to_dict, no seconds); Block 3's summary; the sizes; the commit", not bad,
           "%s; direct fit %.0f s, theta_hat %s; %s" % (bad or "all parts", D.seconds, D.cf.law, "; ".join(notes)))


def _r2(fx):
    """A refused cell: status refused with from_cell's message, exit 0, no fit."""
    r = fx.cell_run("refused", ID_REFUSED, 2)
    d = str(fx.group / ("specimen_%d" % ID_REFUSED))
    ss = nw.load_ss_windows(d)
    nwin = nw.ss_noise_windows(ss)
    ls = nw.load_ls_prewindows(d)
    try:
        nf.CellLayout.from_cell(nwin, ls)
        why = None
    except ValueError as exc:
        why = str(exc)
    rec = r.rec or {}
    lines = cell_lines(r.proc.stdout)
    m = LINE.match(lines[0]) if len(lines) == 1 else None
    secs = rec.get("seconds") or {}
    parts = {
        "the oracle refuses": why is not None and "tail_short" in why,
        "exit 0": r.proc.returncode == 0,
        "line": bool(m) and m.group(2) == "refused" and m.group(3) == why,
        "JSON status, reason": rec.get("status") == "refused" and rec.get("reason") == why,
        "no fit": not any(k in rec for k in ("cell_fit", "sweep_stage", "tables", "traceback"))
        and secs.get("cell_fit") is None and secs.get("sweep_stage") is None and secs.get("windows") is not None,
        "Block 3 summary": (rec.get("windows") or {}).get("ls_tilt_flags")
        == ["ls_tilt_none:%d:tail_short" % s for s in ls.sweep_number]
        and (rec.get("windows") or {}).get("K") == nwin.ss.n_windows,
        "strict JSON": r.text is not None and strict_json(r.text),
    }
    report("R2 a refused cell (the 0.5 s tail, D-114): status refused, reason = from_cell's ValueError, exit 0, no "
           "fit run, Block 3's summary kept", not failed_parts(parts),
           "%s; reason %r" % (failed_parts(parts) or "all parts", (why or "")[:80]))


def _r3(fx):
    """Failed cells: (a) Block 3 raises (LayoutError), (b) Block 4 raises (planted), (c) a Block 9 worker raises."""
    parts = {}
    # (a) a subprocess on the broken archive cell
    r = fx.cell_run("broken", ID_BROKEN, 2)
    rec = r.rec or {}
    lines = cell_lines(r.proc.stdout)
    m = LINE.match(lines[0]) if len(lines) == 1 else None
    parts["(a) Block 3: exit 0, failed, LayoutError, traceback"] = (
        r.proc.returncode == 0 and bool(m) and m.group(2) == "failed"
        and str(rec.get("reason", "")).startswith("LayoutError: QC table has")
        and "load_ss_windows" in str(rec.get("traceback", "")) and "cell_fit" not in rec and "windows" not in rec)
    # (b) a planted exception in Block 4, in process
    D = fx.direct()

    def boom(*a, **k):
        raise RuntimeError("planted R3 failure")
    out_b = fx.tmp / "out_r3b"
    with patched(nf, "fit_cell", boom):
        rc, text = in_process(["cell", "--group-dir", fx.group, "--specimen", ID_OK, "--out", out_b, "--workers", 1,
                               "--sizes", "smoke"])
    rec = json.loads((out_b / "cells" / ("specimen_%d.json" % ID_OK)).read_text())
    m = LINE.match(cell_lines(text)[0]) if len(cell_lines(text)) == 1 else None
    parts["(b) Block 4: exit 0, failed, type and message, traceback"] = (
        rc == 0 and bool(m) and m.group(2) == "failed" and rec.get("status") == "failed"
        and rec.get("reason") == "RuntimeError: planted R3 failure" and "planted R3 failure" in rec.get("traceback", "")
        and "windows" in rec and "cell_fit" not in rec and rec["seconds"]["cell_fit"] is not None)
    # (c) a sweep of Block 9 raising in a worker process: the per-cell result lacks sweep 93's covariance
    cf_missing = replace(D.cf, per_sweep_cov={s: c for s, c in D.cf.per_sweep_cov.items() if s != 93})
    out_c = fx.tmp / "out_r3c"
    with patched(nf, "fit_cell", lambda *a, **k: cf_missing):
        rc, text = in_process(["cell", "--group-dir", fx.group, "--specimen", ID_OK, "--out", out_c, "--workers", 2,
                               "--sizes", "smoke"])
    rec = json.loads((out_c / "cells" / ("specimen_%d.json" % ID_OK)).read_text())
    parts["(c) a Block 9 worker: exit 0, failed, the worker's message and traceback, cell_fit kept"] = (
        rc == 0 and rec.get("status") == "failed"
        and rec.get("reason") == "ValueError: the per-cell result carries no covariance for sweep 93 (per_sweep_cov)"
        and "run_sweep" in rec.get("traceback", "")
        and drop_seconds(rec.get("cell_fit")) == drop_seconds(norm(cf_missing.to_dict()))
        and "sweep_stage" not in rec and rec["seconds"]["sweep_stage"] is not None)
    report("R3 failed cells -- a LayoutError in Block 3 (subprocess), a planted RuntimeError in Block 4 and a ValueError "
           "in a Block 9 worker (in process): status failed with the type, the message and the traceback, exit 0",
           not failed_parts(parts), "%s" % (failed_parts(parts) or "all parts"))


def _r4(fx):
    """The resume rule (D-125.5)."""
    parts = {}
    # (a) a real second run of the 1-worker cell: skipped, the JSON untouched
    r1 = fx.cell_run("ok_w1", ID_OK, 1)
    before = r1.path.read_bytes() if r1.path.exists() else b""
    proc = run_py([fx.code / "run_noise_fit.py", "cell", "--group-dir", fx.group, "--specimen", ID_OK, "--out", r1.out,
                   "--workers", 1, "--sizes", "smoke"], cwd=fx.tmp)
    lines = cell_lines(proc.stdout)
    parts["(a) subprocess: a second `cell` skips, JSON untouched"] = (
        proc.returncode == 0 and len(lines) == 1
        and lines[0].startswith("NOISE_CELL %d: status=ok reason=skipped (same code commit %s and sizes)"
                                % (ID_OK, fx.commit_A[:12]))
        and r1.path.read_bytes() == before and before != b"")
    # (b) in process, the fit replaced by the direct results (instant), the code commit read in the repository
    D = fx.direct()
    calls = {"n": 0}

    def fake_fit(*a, **k):
        calls["n"] += 1
        return D.cf

    out = fx.tmp / "out_r4"
    path = out / "cells" / ("specimen_%d.json" % ID_OK)
    argv = ["cell", "--group-dir", fx.group, "--specimen", ID_OK, "--out", out, "--workers", 1, "--sizes", "smoke"]

    def run(*extra):
        n0 = calls["n"]
        rc, text = in_process(argv + list(extra))
        return rc == 0 and calls["n"] > n0          # True: it ran; False: skipped

    def edit(**kv):
        rec = json.loads(path.read_text())
        for k, v in kv.items():
            if k == "B_v":
                rec["sizes"]["sweep_stage"]["B_v"] = v
            else:
                rec[k] = v
        path.write_text(json.dumps(rec))

    code_file = fx.code / "noise_law.py"
    with patched(rnf, "CODE_DIR", fx.code), patched(nf, "fit_cell", fake_fit), \
            patched(rnf, "run_sweep_stage", lambda *a, **k: D.st):
        parts["(b1) first run runs, commit A"] = run() and json.loads(path.read_text())["code_commit"] == fx.commit_A
        b = path.read_bytes()
        parts["(b2) second run skipped, JSON untouched"] = not run() and path.read_bytes() == b
        parts["(b3) --force re-runs"] = run("--force")
        edit(code_commit="0" * 40)
        parts["(b4) a JSON of another commit re-runs"] = run() and json.loads(path.read_text())["code_commit"] == fx.commit_A
        edit(B_v=5)
        parts["(b5) a JSON of other sizes re-runs"] = run() and not run()
        edit(code_dirty=True)
        parts["(b6) a JSON written from modified code re-runs"] = run() and not run()
        (fx.repo / "README.md").write_text("smoke repository, edited\n")
        git(fx.repo, "commit", "-q", "-am", "B: docs only")
        parts["(b7) a docs commit: the code commit stays A, skipped"] = (rnf.code_commit() == fx.commit_A
                                                                        and not run())
        original = code_file.read_text()
        code_file.write_text(original + "# smoke: an uncommitted change\n")
        parts["(b8) modified code: runs every time, JSON marked"] = (
            rnf.code_dirty() is True and run() and run() and json.loads(path.read_text())["code_dirty"] is True)
        git(fx.repo, "checkout", "--", "Passive Features/HPC script/Ih Fit/noise_law.py")
        parts["(b9) clean again: the dirty JSON re-runs once"] = rnf.code_dirty() is False and run() and not run()
        code_file.write_text(original + "# smoke: a committed change\n")
        git(fx.repo, "commit", "-q", "-am", "C: code")
        commit_c = git(fx.repo, "rev-parse", "HEAD")
        parts["(b10) a code commit re-runs, JSON of commit C"] = (
            rnf.code_commit() == commit_c and run() and json.loads(path.read_text())["code_commit"] == commit_c
            and not run())
        argv_ref = ["cell", "--group-dir", fx.group, "--specimen", ID_REFUSED, "--out", out, "--workers", 1, "--sizes",
                    "smoke"]
        rc1, t1 = in_process(argv_ref)
        rc2, t2 = in_process(argv_ref)
        parts["(b11) a refused JSON is never skipped"] = (rc1 == 0 and rc2 == 0 and "status=refused" in t2
                                                         and "skipped" not in t2)
    srec = {"x": 1}
    ok_prev = {"status": "ok", "code_commit": "c" * 40, "code_dirty": False, "sizes": srec}
    parts["(c) resume_decision: no JSON, failed, no git, git unknown, all equal"] = (
        rnf.resume_decision(None, "c" * 40, False, srec, False)[0] is False
        and rnf.resume_decision(dict(ok_prev, status="failed"), "c" * 40, False, srec, False)[0] is False
        and rnf.resume_decision(ok_prev, None, False, srec, False)[0] is False
        and rnf.resume_decision(ok_prev, "c" * 40, None, srec, False)[0] is False
        and rnf.resume_decision(ok_prev, "c" * 40, False, srec, True)[0] is False
        and rnf.resume_decision(ok_prev, "c" * 40, False, srec, False)[0] is True)
    report("R4 resume (D-125.5): a second `cell` skips (JSON untouched); --force, another commit, other sizes, modified "
           "code re-run; a docs commit does not; a code commit does; a refused JSON is never skipped",
           not failed_parts(parts), "%s" % (failed_parts(parts) or "all parts"))


def _r5(fx):
    """The merge over an ok, a refused, a failed and a missing cell."""
    D = fx.direct()
    runs = {ID_OK: fx.cell_run("ok_w1", ID_OK, 1), ID_REFUSED: fx.cell_run("refused", ID_REFUSED, 2),
            ID_BROKEN: fx.cell_run("broken", ID_BROKEN, 2)}
    missing_json = [sid for sid, r in runs.items() if not r.rec]
    if missing_json:
        report("R5 merge over an ok, a refused, a failed and a missing cell", False,
               "no JSON from the driver for %s: %s" % (missing_json, [runs[s].proc.stderr[-300:] for s in missing_json]))
        return
    out = fx.tmp / "out_r5"
    (out / "cells").mkdir(parents=True)
    for sid, r in runs.items():
        shutil.copy2(str(r.path), str(out / "cells" / r.path.name))
    cohort = [ID_OK, ID_REFUSED, ID_BROKEN, ID_MISSING]
    proc = run_py([fx.code / "run_noise_fit.py", "merge", "--out", out, "--group", GROUP, "--cells",
                   ",".join(str(c) for c in reversed(cohort))], cwd=fx.tmp)
    lines = proc.stdout.splitlines()
    recs = {sid: r.rec for sid, r in runs.items()}
    exp_lines = ["NOISE_MERGE %s: 1 ok, 1 refused, 1 failed, 1 missing" % GROUP,
                 "  refused %d: %s" % (ID_REFUSED, recs[ID_REFUSED]["reason"]),
                 "  failed %d: %s" % (ID_BROKEN, recs[ID_BROKEN]["reason"]),
                 "  missing %d" % ID_MISSING]
    cc = D.st.cell_columns()
    exp_cells_header = (["specimen_id", "status", "reason"] + list(BLOCK4_NAMES) + ["noise_t_ss_ms", "noise_K", "noise_K_LS"]
                        + [k for k in cc if k != "specimen_id"] + ["noise_code_commit"])
    h_cells, rows = read_csv(out / ("noise_cells_%s.csv" % GROUP))
    h_sweeps, srows = read_csv(out / ("noise_sweeps_%s.csv" % GROUP))
    tab = D.st.table_rows()
    parts = {"exit 0, the summary and the listed cells": proc.returncode == 0 and lines == exp_lines,
             "cells: the columns": h_cells == exp_cells_header,
             "cells: one row per cohort cell, in ID order, with its status":
                 [(int(r["specimen_id"]), r["status"]) for r in rows]
                 == [(ID_OK, "ok"), (ID_REFUSED, "refused"), (ID_BROKEN, "failed"), (ID_MISSING, "missing")]}
    if parts["cells: one row per cohort cell, in ID order, with its status"]:
        ok_row, ref_row, bro_row, mis_row = rows
        cf, w = D.cf, recs[ID_OK]["windows"]
        expect_ok = {"reason": "", "noise_sigma_f_mV": cf.law.sigma_f, "noise_rho_f": cf.law.rho_f,
                     "noise_sigma_s_mV": cf.law.sigma_s, "noise_tau_s_ms": 1e3 * cf.law.tau_s, "noise_J": cf.J,
                     "noise_dof": cf.dof, "noise_p": cf.p_value, "noise_kappa": cf.kappa, "noise_n_solves": cf.n_solves,
                     "noise_flags": ";".join(cf.flags), "noise_t_ss_ms": D.nwin.start.t_ss_ms,
                     "noise_K": D.nwin.ss.n_windows, "noise_K_LS": len(D.ls.pre), "noise_code_commit": fx.commit_A}
        expect_ok.update({k: v for k, v in cc.items() if k != "specimen_id"})
        parts["cells: the ok row = Blocks 3, 4 and 9"] = all(same_value(ok_row[k], v) for k, v in expect_ok.items()) \
            and w["t_ss_ms"] == D.nwin.start.t_ss_ms
        wr = recs[ID_REFUSED]["windows"]
        parts["cells: the refused row -- Block 3's columns, no fit"] = (
            ref_row["reason"] == recs[ID_REFUSED]["reason"] and same_value(ref_row["noise_t_ss_ms"], wr["t_ss_ms"])
            and same_value(ref_row["noise_K"], wr["K"]) and same_value(ref_row["noise_K_LS"], wr["K_LS"])
            and all(ref_row[k] == "" for k in BLOCK4_NAMES))
        parts["cells: the failed and the missing rows empty"] = (
            bro_row["reason"] == recs[ID_BROKEN]["reason"]
            and all(bro_row[k] == "" for k in h_cells if k not in ("specimen_id", "status", "reason", "noise_code_commit"))
            and all(mis_row[k] == "" for k in h_cells if k not in ("specimen_id", "status")))
    exp_keys = list(tab[0].keys())
    parts["sweeps: the columns = table_rows' keys"] = h_sweeps == exp_keys == list(rnf.SWEEP_COLUMNS)
    parts["sweeps: the ok cell's rows only, = table_rows (exact)"] = (
        len(srows) == len(tab) and all(all(same_value(s[k], t[k]) for k in exp_keys) for s, t in zip(srows, tab)))
    parts["the driver's Block 9 columns = cell_columns' keys"] = (
        tuple(k for k in cc if k != "specimen_id") == rnf.BLOCK9_CELL_COLUMNS)
    # NaN and None as empty fields (the tables), NaN and inf as null (the JSON): on synthetic values
    rnf.write_csv_atomic(out / "nan_check.csv", ("a", "b", "c", "d"),
                         [{"a": float("nan"), "b": None, "c": float("inf"), "d": 1.5}])
    _hn, nrows = read_csv(out / "nan_check.csv")
    jn = rnf.jsonable({"x": float("nan"), "y": np.float64("-inf"), "z": np.arange(2), "t": (1, np.int64(2)),
                       "b": np.bool_(True), "s": "a"})
    parts["NaN and None as empty fields, NaN and inf as null"] = (
        nrows == [{"a": "", "b": "", "c": "", "d": "1.5"}]
        and jn == {"x": None, "y": None, "z": [0, 1], "t": [1, 2], "b": True, "s": "a"})
    proc2 = run_py([fx.code / "run_noise_fit.py", "merge", "--out", out, "--group", GROUP], cwd=fx.tmp)
    _h, rows2 = read_csv(out / ("noise_cells_%s.csv" % GROUP))
    parts["without --cells: the JSONs found, none missing"] = (
        proc2.returncode == 0 and proc2.stdout.splitlines()[0] == "NOISE_MERGE %s: 1 ok, 1 refused, 1 failed, 0 missing"
        % GROUP and [int(r["specimen_id"]) for r in rows2] == [ID_OK, ID_REFUSED, ID_BROKEN])
    report("R5 merge over an ok, a refused, a failed and a missing cell: the summary and the listed cells; the two "
           "tables' columns; one cell row per cohort cell with its status and values; sweep rows only for the ok cell, "
           "= its table_rows()", not failed_parts(parts), "%s; stderr %s" % (failed_parts(parts) or "all parts",
                                                                            proc.stderr[-300:].strip() or "-"))


def _r6(fx):
    """`list`: sorted by ID (numerically), --cells restricts, an unknown ID refused."""
    g = fx.tmp / "R6 group"
    for name in ("specimen_900000201", "specimen_999999", "specimen_900000203", "specimen_abc"):
        (g / name).mkdir(parents=True)
        (g / name / "ss_pulses.npz").write_bytes(b"")
    (g / "specimen_900000204").mkdir()                      # no ss_pulses.npz: not a cell
    (g / "specimen_900000205").write_text("a file\n")      # not a directory
    empty = fx.tmp / "R6 empty"
    empty.mkdir()
    drv = fx.code / "run_noise_fit.py"

    def ls(*extra, group=g):
        p = run_py([drv, "list", "--group-dir", group] + list(extra), cwd=fx.tmp)
        return p.returncode, p.stdout.split(), p.stderr

    rc, ids, _e = ls()
    rc2, ids2, _e = ls("--cells", "900000203,999999")
    rc3, ids3, _e = ls("--cells", "900000203,900000203")
    rc4, _i, e4 = ls("--cells", "900000204,999999")
    rc5, _i, _e = ls("--cells", "12x")
    rc6, _i, _e = ls(group=fx.tmp / "no such group")
    rc7, _i, _e = ls(group=empty)
    parts = {"sorted by ID": rc == 0 and ids == ["999999", "900000201", "900000203"],
             "--cells restricts, sorted": rc2 == 0 and ids2 == ["999999", "900000203"],
             "--cells deduplicated": rc3 == 0 and ids3 == ["900000203"],
             "an ID not in the group refused, named": rc4 == 2 and "900000204" in e4 and "999999" not in e4,
             "a malformed --cells refused": rc5 == 2,
             "a missing or an empty group refused": rc6 == 2 and rc7 == 2}
    report("R6 `list`: the cohort sorted by ID, --cells restricts it, an unknown ID refused (exit 2)",
           not failed_parts(parts), "%s; listed %s" % (failed_parts(parts) or "all parts", ids))


def _r7_groups(fx):
    root = fx.tmp / "R7 archive"
    if not root.exists():
        for grp, n in (("L3_exc", 3), ("G20", 20), ("G21", 21)):
            for k in range(n):
                d = root / grp / ("specimen_%d" % (900000301 + k))
                d.mkdir(parents=True)
                (d / "ss_pulses.npz").write_bytes(b"")
    return root


def _r7(fx):
    """The job script: syntax, marker lines, the dry run, the cap, a (stubbed) submission, the job roles."""
    script = fx.code / "submit_noise_fit.sh"
    text = script.read_text()
    root = _r7_groups(fx)
    ids = ["900000301", "900000302", "900000303"]
    out_dir = root / "pipeline_outputs_ih" / "noise_fit" / "t7" / "L3_exc"
    parts = {}
    parts["bash -n"] = subprocess.run(["bash", "-n", str(script)]).returncode == 0
    parts["marker lines (N6), BLAS threads 1, the cap"] = all(s in text for s in (
        "export ROADMAP_STAGE=N6", "specs/roadmap/hpc_status.sh", "roadmap_job_start",
        "trap 'roadmap_job_end $?' EXIT", "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1",
        "MAX_CORES=120"))
    # the dry run
    fx.reset_logs()
    p = fx.run_sh(fx.sh_env(RUN_TAG="t7", IH_ARCHIVE_ROOT=root, DRY_RUN=1))
    lines = p.stdout.splitlines()
    sub = [ln for ln in lines if " subjob " in ln and " cell " in ln]
    q = [ln for ln in lines if "DRY RUN:" in ln and " qsub " in ln]
    mrg = [ln for ln in lines if "DRY RUN:   merge:" in ln]
    parts["dry run: one cell command per cohort cell, in order"] = (
        p.returncode == 0 and len(sub) == 3
        and all((" --specimen %s " % i) in s and " --workers 6 " in s and s.endswith(" --sizes full")
                and "R7\\ archive/L3_exc" in s for i, s in zip(ids, sub)))
    parts["dry run: the two qsub lines and the merge's command"] = (
        len(q) == 2 and " -J 0-2 " in q[0] and " select=1:ncpus=6 " in q[0] and " walltime=12:00:00 " in q[0]
        and " -q cpu " in q[0] and " -N noise_fit " in q[0] and "depend=afterany:" in q[1] and " -N noise_merge " in q[1]
        and len(mrg) == 1 and " merge " in mrg[0] and ("--cells " + "\\,".join(ids)) in mrg[0])
    parts["dry run: nothing submitted, nothing written, no fit run"] = (
        not (fx.tmp / "qsub.log").exists() and not (fx.tmp / "py.log").exists()
        and not (fx.tmp / "home" / "noise_fit_runs").exists() and not out_dir.exists()
        and "CLI contract" in p.stdout)
    dry_text = p.stdout
    # the cap (D-125.1)
    fx.reset_logs()
    p21 = fx.run_sh(fx.sh_env(RUN_TAG="t7", GROUP="G21", IH_ARCHIVE_ROOT=root))
    p21d = fx.run_sh(fx.sh_env(RUN_TAG="t7", GROUP="G21", IH_ARCHIVE_ROOT=root, DRY_RUN=1))
    p20d = fx.run_sh(fx.sh_env(RUN_TAG="t7", GROUP="G20", IH_ARCHIVE_ROOT=root, DRY_RUN=1))
    parts["21 cells x 6 cores refused (dry or not), 20 x 6 = 120 accepted"] = (
        p21.returncode == 3 and "126 cores > 120" in p21.stderr and p21d.returncode == 3
        and not (fx.tmp / "qsub.log").exists() and p20d.returncode == 0)
    # a (stubbed) submission
    fx.reset_logs()
    p = fx.run_sh(fx.sh_env(RUN_TAG="t7", IH_ARCHIVE_ROOT=root))
    calls = (fx.tmp / "qsub.log").read_text().splitlines() if (fx.tmp / "qsub.log").exists() else []
    run_files = sorted((fx.tmp / "home" / "noise_fit_runs").glob("t7_L3_exc_*.env"))
    rf = run_files[-1] if run_files else None
    parts["submission: the array, then the merge after it"] = (
        p.returncode == 0 and len(calls) == 2 and rf is not None
        and "[-J] [0-2]" in calls[0] and "[select=1:ncpus=6]" in calls[0] and "[walltime=12:00:00]" in calls[0]
        and ("[NOISE_ROLE=cell,NOISE_RUN_FILE=%s]" % rf) in calls[0]
        and "[depend=afterany:4242[].testserver]" in calls[1] and ("[NOISE_ROLE=merge,NOISE_RUN_FILE=%s]" % rf) in calls[1]
        and (out_dir / "logs" / rf.name).is_file() and (out_dir / "cells").is_dir())
    fx.reset_logs()
    p1 = fx.run_sh(fx.sh_env(RUN_TAG="t7one", IH_ARCHIVE_ROOT=root, CELLS="900000302", ROADMAP_DISABLE=1))
    calls1 = (fx.tmp / "qsub.log").read_text().splitlines() if (fx.tmp / "qsub.log").exists() else []
    parts["one cell: a plain job with NOISE_INDEX=0; ROADMAP_DISABLE passed on"] = (
        p1.returncode == 0 and len(calls1) == 2 and "[-J]" not in calls1[0] and "NOISE_INDEX=0" in calls1[0]
        and ",ROADMAP_DISABLE=1]" in calls1[0] and ",ROADMAP_DISABLE=1]" in calls1[1])
    # the job roles, from a spool copy, with the run file of the submission
    spool = fx.tmp / "spool"
    spool.mkdir(exist_ok=True)
    job = spool / "4242.SC"
    shutil.copy2(str(script), str(job))
    fx.reset_logs()
    pc = fx.run_sh(fx.sh_env(PBS_JOBID="4242[1].testserver", PBS_ARRAY_INDEX=1, NOISE_ROLE="cell",
                             NOISE_RUN_FILE=rf or "", ROADMAP_DISABLE=1, OMP_NUM_THREADS=6), script=job)
    pm = fx.run_sh(fx.sh_env(PBS_JOBID="4243.testserver", NOISE_ROLE="merge", NOISE_RUN_FILE=rf or "",
                             ROADMAP_DISABLE=1), script=job)
    runs = (fx.tmp / "py.log").read_text().splitlines() if (fx.tmp / "py.log").exists() else []
    g = str(root / "L3_exc")
    exp_cell = ("RUN [%s] [cell] [--group-dir] [%s] [--specimen] [900000302] [--out] [%s] [--workers] [6] [--sizes] "
                "[full] ENV[1/1/1]" % (fx.code / "run_noise_fit.py", g, out_dir))
    exp_merge = ("RUN [%s] [merge] [--out] [%s] [--group] [L3_exc] [--cells] [%s] ENV[1/1/1]"
                 % (fx.code / "run_noise_fit.py", out_dir, ",".join(ids)))
    parts["the cell role: the index-th cell, --workers NCPUS, threads 1 (PBS's OMP_NUM_THREADS=6 overridden)"] = (
        pc.returncode == 0 and len(runs) == 2 and runs[0] == exp_cell
        and (out_dir / "logs" / "cell_900000302_4242_1_.testserver.log").is_file())
    parts["the merge role: the cohort's merge"] = pm.returncode == 0 and len(runs) == 2 and runs[1] == exp_merge
    pf = fx.run_sh(fx.sh_env(PBS_JOBID="4242[0].testserver", PBS_ARRAY_INDEX=0, NOISE_ROLE="cell",
                             NOISE_RUN_FILE=rf or "", ROADMAP_DISABLE=1, PY_STUB_EXIT=3), script=job)
    parts["a failing python fails the job (exit 3)"] = pf.returncode == 3
    # refusals
    pq = fx.run_sh(fx.sh_env(PBS_JOBID="99.testserver", RUN_TAG="t7", IH_ARCHIVE_ROOT=root), script=job)
    pt = fx.run_sh(fx.sh_env(IH_ARCHIVE_ROOT=root))
    pt2 = fx.run_sh(fx.sh_env(RUN_TAG="a b", IH_ARCHIVE_ROOT=root))
    pu = fx.run_sh(fx.sh_env(RUN_TAG="t7", IH_ARCHIVE_ROOT=root, CELLS="900000301,123", DRY_RUN=1))
    parts["refused: qsub'd directly, no RUN_TAG, a bad RUN_TAG, an unknown cell"] = (
        pq.returncode == 2 and "NOISE_ROLE is not" in pq.stderr and pt.returncode == 2 and pt2.returncode == 2
        and pu.returncode == 3 and "123" in pu.stderr)
    # the roadmap marker, pushed to the test repository's local origin
    if fx.has_helper:
        rf1 = fx.tmp / "marker.env"
        rf1.write_text("NOISE_CODE_DIR=%s\nNOISE_ARCHIVE_DIR=%s\nNOISE_OUT_DIR=%s\nNOISE_GROUP=L3_exc\nNOISE_NCPUS=6\n"
                       "NOISE_SIZES=full\nNOISE_CONDA_ENV=prova\nNOISE_COHORT=( 900000301 )\n"
                       % (_q(fx.code), _q(g), _q(fx.tmp / "marker out")))
        pk = fx.run_sh(fx.sh_env(PBS_JOBID="777.testserver", NOISE_INDEX=0, NOISE_ROLE="cell", NOISE_RUN_FILE=rf1),
                       script=job)
        pk2 = fx.run_sh(fx.sh_env(PBS_JOBID="778.testserver", NOISE_INDEX=0, NOISE_ROLE="cell", NOISE_RUN_FILE=rf1,
                                  PY_STUB_EXIT=3), script=job)

        def marker(jobid):
            r = subprocess.run(["git", "--git-dir", str(fx.origin), "show", "hpc-status:runs/N6/%s.json" % jobid],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            return json.loads(r.stdout) if r.returncode == 0 else {}
        m1, m2 = marker("777.testserver"), marker("778.testserver")
        parts["the marker: done after a good job, failed after a failing one (local origin)"] = (
            pk.returncode == 0 and pk2.returncode == 3 and m1.get("state") == "done" and m1.get("n_tasks") == 1
            and m2.get("state") == "failed" and m2.get("last_exit") == 3)
    else:
        parts["the marker: specs/roadmap/hpc_status.sh not found next to this checkout"] = False
    report("R7 the job script: bash -n; the marker lines; DRY_RUN=1 prints one `cell` command per cohort cell and the "
           "merge's, submits nothing, writes nothing, runs no fit; 21 cells x 6 cores refused; the submission; the cell "
           "and merge roles; the marker", not failed_parts(parts),
           "%s; dry-run header: %s" % (failed_parts(parts) or "all parts",
                                       " | ".join(ln for ln in dry_text.splitlines() if "cores:" in ln)))
    fx.memo["dry_text"] = dry_text


def _q(path):
    return subprocess.run(["bash", "-c", 'printf %q "$1"', "_", str(path)], stdout=subprocess.PIPE,
                          universal_newlines=True).stdout


def _r8(fx):
    bad = []
    for name in ("run_noise_fit.py", "submit_noise_fit.sh", "smoke_noise_run.py"):
        raw = (HERE / name).read_bytes()
        if b"\r" in raw or any(b > 127 for b in raw):
            bad.append(name)
    report("R8 ASCII/LF: run_noise_fit.py, submit_noise_fit.sh, smoke_noise_run.py", not bad, "bad: %s" % (bad or "none"))


def _r9(fx):
    """The decided defaults (D-122.1, D-125.1, D-125.5, D-126), so that a default changed by mistake fails."""
    full_cfg, full_stage = rnf.sizes("full")
    smoke_cfg, smoke_stage = rnf.sizes("smoke")
    a = rnf.build_parser().parse_args(["cell", "--group-dir", "g", "--specimen", "1", "--out", "o"])
    if "dry_text" not in fx.memo:
        fx.memo["dry_text"] = fx.run_sh(fx.sh_env(RUN_TAG="t9", IH_ARCHIVE_ROOT=_r7_groups(fx), DRY_RUN=1)).stdout
    dry = fx.memo["dry_text"]
    script = (fx.code / "submit_noise_fit.sh").read_text()
    parts = {
        "full = Blocks 4 and 9's defaults: M 3000, R 600/3000, B 2000; B_v 50, M_refit 1000":
            full_cfg == nf.NoiseFitConfig() and full_stage == nsf.SweepStageConfig()
            and (full_cfg.M_binding, full_cfg.R_iter, full_cfg.R_final, full_cfg.B_boot) == (3000, 600, 3000, 2000)
            and (full_stage.B_v, full_stage.M_refit) == (50, 1000),
        "smoke = F7 and P9": smoke_cfg == SMOKE_CFG and smoke_stage == SMOKE_STAGE,
        "the driver: --workers 6, --sizes full": (a.workers, a.sizes, a.force) == (6, "full", False),
        "the resume rule's pathspec": rnf.CODE_PATHSPEC == ("noise_*.py", "run_noise_fit.py"),
        "the script: NCPUS 6, 12:00:00, L3_exc, prova, cap 120, queue cpu":
            "x NCPUS 6 = 18 (cap 120, D-125.1); walltime 12:00:00 per cell job" in dry
            and "group L3_exc," in dry and "conda env prova" in dry and " -q cpu " in dry
            and 'IH_ARCHIVE_ROOT:-/davinci-1/home/ldellamea/Human Neurons Fitting}' in script,
    }
    report("R9 the decided defaults: --sizes full = M 3000, R 600 and 3000, B_boot 2000, B_v 50, M_refit 1000 "
           "(D-125.5, D-122.1); --workers 6 and NCPUS 6 (D-126); WALLTIME 12:00:00; the 120-core cap (D-125.1); "
           "GROUP L3_exc; CONDA_ENV prova", not failed_parts(parts), "%s" % (failed_parts(parts) or "all parts"))


def main(only=None):
    t_all = time.time()
    fx = Fixture()
    print("fixture: %s; repository commit %s; real python %s" % (fx.tmp, fx.commit_A[:12], sys.executable))
    checks = [("R1", _r1), ("R2", _r2), ("R3", _r3), ("R4", _r4), ("R5", _r5), ("R6", _r6), ("R7", _r7),
              ("R8", _r8), ("R9", _r9)]
    for name, fn in checks:
        if only and name not in only:
            continue
        try:
            fn(fx)
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            report("%s crashed" % name, False, "see traceback")
    n_ok = sum(RESULTS)
    print("SMOKE noise_run: %d/%d passed  (%.0f s)" % (n_ok, len(RESULTS), time.time() - t_all))
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    # optional: python smoke_noise_run.py R5,R7   (a subset, for mutation runs)
    sys.exit(main(set(sys.argv[1].split(",")) if len(sys.argv) > 1 else None))
