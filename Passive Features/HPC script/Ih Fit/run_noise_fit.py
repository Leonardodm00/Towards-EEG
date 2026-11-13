"""run_noise_fit -- Block 6 of specs/SPEC.md: the noise-fit run on the cells of
one archive group (D-123 to D-126). The scripts layer of the noise fit: the
only place of it that reads or writes files. submit_noise_fit.sh runs it on
davinci as a PBS array (one subjob per cell) and a merge job.

In plain words: `cell` takes one real cell, reads its SS and LS windows
(Block 3), fits the cell's noise law theta_hat(c) (Block 4), then every
sweep's own law and the pooling per protocol (Block 9, the sweeps spread
over a pool of processes), and writes all of it to one JSON. A cell whose
sweep tilt cannot be read is refused (D-114); a cell whose pipeline raises
is failed; both are recorded with their reason and the run goes on
(D-125.4). `merge` gathers a cohort's JSONs into noise_cells_<GROUP>.csv
(one row per cell) and noise_sweeps_<GROUP>.csv (one row per sweep of every
ok cell) (D-125.2). `list` names the cohort.

    python run_noise_fit.py list  --group-dir DIR [--cells ID,ID]
    python run_noise_fit.py cell  --group-dir DIR --specimen ID --out DIR
                                  [--workers 6] [--force] [--sizes full|smoke]
    python run_noise_fit.py merge --out DIR --group NAME [--cells ID,ID]

`cell` writes <out>/cells/specimen_<id>.json (atomically) and prints
    NOISE_CELL <id>: status=<ok|refused|failed> reason=<...> seconds=<s>
and exits 0 whatever the status; non-zero only for its own usage or I/O
errors. Resume rule (D-125.5): a cell whose JSON has status ok, was written
by the current code commit -- the last commit touching noise_*.py or
run_noise_fit.py, so that a docs or roadmap commit re-runs nothing -- and at
the same sizes is skipped unless --force; a JSON written from modified,
uncommitted code is never skipped, nor is any cell while the code differs
from its commit.

`--sizes full` is Blocks 4 and 9's defaults (D-125.5); `--sizes smoke` the
reduced sizes of their smoke suites (F7, P9), for the tests. `--workers`
is the size of Block 9's process pool (6, D-125.1, D-126).

Pure ASCII, LF. Spec: specs/SPEC.md, Block 6. Smoke: smoke_noise_run.py.
"""
from __future__ import annotations

import os

# D-126: the BLAS threads at 1 -- Block 9's sweeps run in a pool of processes,
# one core each. Set before numpy is imported, so that the driver run by hand
# behaves as under submit_noise_fit.sh (which exports the same three).
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse                                                          # noqa: E402
import csv                                                               # noqa: E402
import json                                                              # noqa: E402
import math                                                              # noqa: E402
import multiprocessing                                                   # noqa: E402
import platform                                                          # noqa: E402
import re                                                                # noqa: E402
import socket                                                            # noqa: E402
import subprocess                                                        # noqa: E402
import sys                                                               # noqa: E402
import time                                                              # noqa: E402
import traceback                                                         # noqa: E402
from concurrent.futures import FIRST_EXCEPTION, ProcessPoolExecutor, wait   # noqa: E402
from dataclasses import asdict                                           # noqa: E402
from datetime import datetime, timezone                                  # noqa: E402
from pathlib import Path                                                 # noqa: E402
from typing import Callable, Dict, List, Optional, Sequence, Tuple       # noqa: E402

import numpy as np                                                       # noqa: E402
import scipy                                                             # noqa: E402

import noise_cell_stats as ncs                                           # noqa: E402
import noise_fit as nf                                                   # noqa: E402
import noise_sweep_fit as nsf                                            # noqa: E402
import noise_windows as nw                                               # noqa: E402

CODE_DIR = Path(__file__).resolve().parent
CODE_PATHSPEC = ("noise_*.py", "run_noise_fit.py")   # the noise-fit code: its whole import chain and this driver
STATUSES = ("ok", "refused", "failed")
DEFAULT_WORKERS = 6                                  # D-125.1, D-126
SIZE_NAMES = ("full", "smoke")
_SPECIMEN_DIR = re.compile(r"^specimen_([0-9]+)$")
_SPECIMEN_JSON = re.compile(r"^specimen_([0-9]+)\.json$")
_GROUP_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")

# The tables (D-125.2). Block 4 has no column method: its columns are named
# here; Block 9's are its cell_columns() and table_rows(), whose keys must be
# these (the merge refuses a JSON whose keys differ; smoke R5 checks them).
BLOCK4_COLUMNS = ("noise_sigma_f_mV", "noise_rho_f", "noise_sigma_s_mV", "noise_tau_s_ms", "noise_J",
                  "noise_dof", "noise_p", "noise_kappa", "noise_n_solves", "noise_flags")
BLOCK3_COLUMNS = ("noise_t_ss_ms", "noise_K", "noise_K_LS")
BLOCK9_CELL_COLUMNS = ("noise_cell_tau2_log_sigma_s_ss", "noise_cell_mu_log_sigma_s_ss",
                       "noise_cell_tau2_log_sigma_s_ls", "noise_cell_mu_log_sigma_s_ls")
CELL_COLUMNS = (("specimen_id", "status", "reason") + BLOCK4_COLUMNS + BLOCK3_COLUMNS + BLOCK9_CELL_COLUMNS
                + ("noise_code_commit",))
SWEEP_COLUMNS = ("specimen_id", "noise_sweep_protocol", "noise_sweep_number", "noise_tilt_mV_per_s",
                 "noise_sweep_sigma_f_mV", "noise_sweep_rho_f", "noise_sweep_sigma_s_raw_mV",
                 "noise_sweep_sigma_s_mV", "noise_sweep_v_log_sigma_s", "noise_sweep_w", "noise_sweep_flags")


# ===========================================================================
#  Small helpers: output, JSON, the code commit, the sizes
# ===========================================================================
def _say(msg: str) -> None:
    print(msg, flush=True)


def _err(msg: str) -> None:
    sys.stderr.write("[run_noise_fit] %s\n" % msg)
    sys.stderr.flush()


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def jsonable(x):
    """A JSON-ready copy: str keys, lists for tuples and arrays, Python
    scalars, NaN and inf as None; any other type is refused (TypeError)."""
    # Custom: json writes NaN as the non-standard token NaN; this pass writes null.
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        f = float(x)
        return f if math.isfinite(f) else None
    if x is None or isinstance(x, str):
        return x
    raise TypeError("not JSON-ready: %s" % type(x).__name__)


def write_json_atomic(path: Path, obj) -> None:
    """Strict JSON (NaN refused), ASCII, through a temporary file and os.replace."""
    text = json.dumps(jsonable(obj), indent=1, allow_nan=False)
    tmp = path.with_name(".%s.%d.tmp" % (path.name, os.getpid()))
    with open(tmp, "w", encoding="ascii", newline="\n") as fh:
        fh.write(text + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def read_json(path: Path) -> Optional[dict]:
    """The JSON at `path`, or None when absent or unreadable."""
    try:
        with open(path, encoding="ascii") as fh:
            out = json.load(fh)
    except (OSError, ValueError):
        return None
    return out if isinstance(out, dict) else None


def _git(args: Sequence[str]) -> Optional[str]:
    try:
        res = subprocess.run(["git", "-C", str(CODE_DIR)] + list(args), stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, universal_newlines=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    return res.stdout.strip() if res.returncode == 0 else None


def code_commit() -> Optional[str]:
    """The last commit that touched the noise-fit code (D-125.5): `git log -1
    --format=%H -- noise_*.py run_noise_fit.py` in CODE_DIR; None without git."""
    return _git(["log", "-1", "--format=%H", "--"] + list(CODE_PATHSPEC)) or None


def code_dirty() -> Optional[bool]:
    """True when a noise-fit file differs from its commit (modified, or a new
    noise_*.py not committed); None without git."""
    out = _git(["status", "--porcelain", "--"] + list(CODE_PATHSPEC))
    return None if out is None else bool(out)


def sizes(name: str) -> Tuple[nf.NoiseFitConfig, nsf.SweepStageConfig]:
    """'full': Blocks 4 and 9's defaults (D-125.5: M = 3000, R = 600 and 3000,
    B_boot = 2000; B_v = 50, M_refit = 1000, D-122.1). 'smoke': the reduced
    sizes of Block 4's F7 and Block 9's P9."""
    if name == "full":
        return nf.NoiseFitConfig(), nsf.SweepStageConfig()
    if name == "smoke":
        return (nf.NoiseFitConfig(M_binding=60, R_iter=40, R_final=60, B_boot=400, k_max=2, max_nfev=6),
                nsf.SweepStageConfig(B_v=4, M_refit=60))
    raise ValueError("sizes must be one of %s, got %r" % (SIZE_NAMES, name))


def sizes_record(name: str, cfg: nf.NoiseFitConfig, stage_cfg: nsf.SweepStageConfig) -> Dict[str, object]:
    return jsonable({"name": name, "noise_fit": asdict(cfg), "sweep_stage": asdict(stage_cfg)})


def resume_decision(prev: Optional[dict], commit: Optional[str], dirty: Optional[bool],
                    sizes_rec: Dict[str, object], force: bool) -> Tuple[bool, str]:
    """(skip, why) by the resume rule (D-125.5): skip only a previous JSON with
    status ok, written by `commit` from clean code, at the same sizes, while
    the code is clean now and --force is not given."""
    # Custom: the resume rule (no library decides it).
    if force:
        return False, "--force"
    if prev is None:
        return False, "no previous JSON"
    if prev.get("status") != "ok":
        return False, "the previous JSON has status %s" % (prev.get("status"),)
    if not commit:
        return False, "the code commit is unknown (no git)"
    if dirty is not False:
        return False, "the noise-fit code differs from its commit"
    if prev.get("code_dirty") is not False:
        return False, "the previous JSON was written from modified code"
    if prev.get("code_commit") != commit:
        return False, "the previous JSON is of commit %s" % (str(prev.get("code_commit"))[:12],)
    if prev.get("sizes") != sizes_rec:
        return False, "the previous JSON is of other sizes"
    return True, "same code commit %s and sizes" % commit[:12]


# ===========================================================================
#  The cohort
# ===========================================================================
def parse_cells(text: str) -> List[int]:
    """'ID,ID,...' -> the distinct IDs in the order given (argparse type)."""
    parts = [p.strip() for p in str(text).split(",")]
    if not parts or any(not re.fullmatch(r"[0-9]+", p) for p in parts):
        raise argparse.ArgumentTypeError("--cells: comma-separated specimen IDs, got %r" % (text,))
    return list(dict.fromkeys(int(p) for p in parts))


def list_cohort(group_dir: Path, cells: Optional[Sequence[int]] = None) -> List[int]:
    """Every specimen_<id>/ of the group that holds ss_pulses.npz, sorted by
    ID; `cells` restricts it (ValueError naming any ID not in the group)."""
    ids = []
    for p in Path(group_dir).iterdir():
        m = _SPECIMEN_DIR.match(p.name)
        if m and p.is_dir() and (p / "ss_pulses.npz").is_file():
            ids.append(int(m.group(1)))
    ids.sort()
    if cells is None:
        return ids
    known = set(ids)
    unknown = [c for c in cells if c not in known]
    if unknown:
        raise ValueError("not in the group (no specimen_<id>/ with ss_pulses.npz): %s"
                         % ",".join(str(c) for c in unknown))
    return sorted(set(int(c) for c in cells))


# ===========================================================================
#  One cell: Blocks 3, 4 and 9
# ===========================================================================
def windows_summary(ss: nw.SSWindows, nwin: nw.SSNoiseWindows, ls: nw.LSWindows,
                    ls_tilts: nw.SweepTilts) -> Dict[str, object]:
    """Block 3's summary for the JSON: t_SS(c), K (the kept SS windows), K_LS,
    the start's and the tilt readings' flags, and the readings themselves."""
    st = nwin.start
    groups = [{"sweep": int(s), "dep": bool(d), "K": int(k), "status": str(g), "t_ms": float(t)}
              for s, d, k, g, t in zip(st.group_sweep, st.group_is_dep, st.group_K, st.group_status,
                                       st.t_group_ms)]
    tilts = nwin.tilts
    return jsonable({
        "fs_Hz": float(ss.fs),
        "t_ss_ms": float(st.t_ss_ms), "t_star_ms": float(st.t_star_ms), "sigma_ss_mV": float(st.sigma_ss_mV),
        "K": int(nwin.ss.n_windows), "K_stored": int(ss.n_windows), "K_LS": int(len(ls.pre)),
        "start_flags": list(st.flags), "start_fallback_reason": str(st.fallback_reason), "start_groups": groups,
        "ss_tilt_flags": [] if tilts is None else list(tilts.flags), "ls_tilt_flags": list(ls_tilts.flags),
        "ss_tilt_mV_per_s": {} if tilts is None else {str(k): v for k, v in tilts.as_dict().items()},
        "ls_tilt_mV_per_s": {str(k): v for k, v in ls_tilts.as_dict().items()},
        "ls_end_status": {str(int(s)): str(e) for s, e in zip(ls.sweep_number, ls.end_status)},
    })


def _sweep_task(cell_fit: nf.CellFitResult, target: nf.CellTarget, layout: nf.CellLayout,
                cfg: nf.NoiseFitConfig, stage_cfg: nsf.SweepStageConfig, protocol: str, sweep: int
                ) -> Dict[str, object]:
    """One sweep of Block 9 (run_sweep) in a worker process."""
    tag = "[noise_fit %d %s %d]" % (layout.specimen_id, protocol.upper(), sweep)
    return nsf.run_sweep(cell_fit, target, layout, cfg, stage_cfg, protocol, sweep,
                         log=lambda s: _say("%s %s" % (tag, s)))


def run_sweep_stage(cell_fit: nf.CellFitResult, target: nf.CellTarget, layout: nf.CellLayout,
                    cfg: nf.NoiseFitConfig, stage_cfg: nsf.SweepStageConfig, workers: int,
                    log: Callable[[str], None] = _say) -> nsf.SweepStageResult:
    """Block 9 on every sweep of S(c) in a pool of `workers` processes
    (D-125.1, D-126): run_sweep per sweep, then pool_cell -- equal to
    nsf.fit_sweeps (Block 9's P9). The rows keep fit_sweeps' order (the
    layout's SS sweeps, then its LS sweeps); the LS sweeps are submitted
    first, being the slower (SPEC 2.5 (h)). The first exception of a sweep is
    raised, after the sweeps not yet started are cancelled. The workers are
    forked where the platform can: they then run the modules this process
    loaded at its start -- the commit its JSON records -- whatever a `git
    pull` changes on disk meanwhile (Python 3.14 made forkserver the Linux
    default; a forkserver or spawned worker would import the files anew)."""
    if int(cell_fit.specimen_id) != int(layout.specimen_id):
        raise ValueError("the per-cell result is of specimen %d, the layout of %d"
                         % (cell_fit.specimen_id, layout.specimen_id))
    t0 = time.time()
    tasks = [("ss", int(s)) for s in layout.sweep_numbers] + [("ls", int(s)) for s in layout.ls_sweep_number]
    order = [i for i, t in enumerate(tasks) if t[0] == "ls"] + [i for i, t in enumerate(tasks) if t[0] == "ss"]
    log("sweep stage: %d SS + %d LS sweeps on %d worker(s)"
        % (len(layout.sweep_numbers), len(layout.ls_sweep_number), workers))
    ctx = multiprocessing.get_context("fork") if "fork" in multiprocessing.get_all_start_methods() else None
    with ProcessPoolExecutor(max_workers=int(workers), mp_context=ctx) as pool:
        futures = {i: pool.submit(_sweep_task, cell_fit, target, layout, cfg, stage_cfg, tasks[i][0], tasks[i][1])
                   for i in order}
        done, pending = wait(list(futures.values()), return_when=FIRST_EXCEPTION)
        failed = [i for i in range(len(tasks)) if futures[i] in done and futures[i].exception() is not None]
        if failed:
            for f in pending:
                f.cancel()
            raise futures[failed[0]].exception()
        rows = [futures[i].result() for i in range(len(tasks))]
    return nsf.pool_cell(rows, cell_fit.law, layout.specimen_id, stage_cfg=stage_cfg, seconds=time.time() - t0)


def process_cell(spec_dir: Path, specimen: int, cfg: nf.NoiseFitConfig, stage_cfg: nsf.SweepStageConfig,
                 workers: int, log: Callable[[str], None] = _say) -> Dict[str, object]:
    """Blocks 3, 4 and 9 on one archive cell (D-123.1). Returns the record's
    status part: status, reason, traceback (failed only), seconds per stage,
    and, as far as the pipeline got, the Block 3 summary (`windows`), Block
    4's `cell_fit`, Block 9's `sweep_stage` and the cell's table rows
    (`tables`). A ValueError of CellLayout.from_cell is a refusal (D-114);
    any other exception a failure (D-125.4)."""
    # Custom: the driver (no library runs this pipeline).
    rec: Dict[str, object] = {"status": None, "reason": ""}
    secs: Dict[str, Optional[float]] = {"windows": None, "cell_fit": None, "sweep_stage": None}
    rec["seconds"] = secs
    stage, t = "windows", time.time()
    try:
        wcfg = cfg.window_cfg
        ss = nw.load_ss_windows(str(spec_dir), wcfg)
        nwin = nw.ss_noise_windows(ss, wcfg)
        ls = nw.load_ls_prewindows(str(spec_dir), wcfg)
        rec["windows"] = windows_summary(ss, nwin, ls, ncs.ls_sweep_tilts(ls))
        if int(ss.specimen_id) != int(specimen):
            raise RuntimeError("metadata.json of specimen_%d names specimen %d" % (specimen, ss.specimen_id))
        log("windows: t_SS %.1f ms, K %d kept of %d, K_LS %d, flags %s"
            % (nwin.start.t_ss_ms, nwin.ss.n_windows, ss.n_windows, len(ls.pre), list(nwin.start.flags)))
        try:
            layout = nf.CellLayout.from_cell(nwin, ls, wcfg)
        except ValueError as exc:                     # D-114: refused, no fit
            secs["windows"] = time.time() - t
            rec["status"], rec["reason"] = "refused", str(exc)
            return rec
        target = nf.CellTarget.from_cell(nwin, ls, layout, wcfg)
        secs["windows"] = time.time() - t

        stage, t = "cell_fit", time.time()
        cell_fit = nf.fit_cell(target, layout, cfg, log=lambda s: log("cell fit: " + s))
        secs["cell_fit"] = time.time() - t
        rec["cell_fit"] = jsonable(cell_fit.to_dict())
        log("cell fit: theta_hat %s, J %.4g, %d solves, flags %s (%.0f s)"
            % (cell_fit.law, cell_fit.J, cell_fit.n_solves, list(cell_fit.flags), secs["cell_fit"]))

        stage, t = "sweep_stage", time.time()
        result = run_sweep_stage(cell_fit, target, layout, cfg, stage_cfg, workers, log)
        secs["sweep_stage"] = time.time() - t
        rec["sweep_stage"] = jsonable(result.to_dict())
        rec["tables"] = jsonable({"cell_columns": result.cell_columns(), "sweep_rows": result.table_rows()})
        rec["status"] = "ok"
    except Exception as exc:                          # noqa: BLE001  (D-125.4: recorded, the run goes on)
        secs[stage] = time.time() - t
        rec["status"], rec["reason"] = "failed", "%s: %s" % (type(exc).__name__, exc)
        rec["traceback"] = traceback.format_exc()
    return rec


def _versions() -> Dict[str, Optional[str]]:
    sm = sys.modules.get("statsmodels")
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "statsmodels": getattr(sm, "__version__", None)}


def _one_line(text: str) -> str:
    return " ".join(str(text).split())


def cmd_cell(args: argparse.Namespace) -> int:
    group_dir = Path(args.group_dir)
    spec_dir = group_dir / ("specimen_%d" % args.specimen)
    if not spec_dir.is_dir():
        _err("no directory %s" % spec_dir)
        return 2
    if args.workers < 1:
        _err("--workers must be >= 1")
        return 2
    cfg, stage_cfg = sizes(args.sizes)
    srec = sizes_record(args.sizes, cfg, stage_cfg)
    commit, dirty = code_commit(), code_dirty()
    cells_dir = Path(args.out) / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    path = cells_dir / ("specimen_%d.json" % args.specimen)
    skip, why = resume_decision(read_json(path), commit, dirty, srec, args.force)
    if skip:
        _say("NOISE_CELL %d: status=ok reason=skipped (%s) seconds=0" % (args.specimen, why))
        return 0
    prefix = "[noise_fit %d]" % args.specimen
    say = lambda s: _say("%s %s" % (prefix, s))       # noqa: E731
    say("run (%s); code %s%s, sizes %s, %d worker(s)"
        % (why, (commit or "unknown")[:12], " (modified)" if dirty else "", args.sizes, args.workers))
    started, t0 = _utc_now(), time.time()
    res = process_cell(spec_dir, args.specimen, cfg, stage_cfg, args.workers, say)
    res["seconds"]["total"] = time.time() - t0
    rec: Dict[str, object] = {"specimen_id": int(args.specimen), "group": Path(os.path.abspath(str(group_dir))).name,
                              "status": res["status"], "reason": res["reason"]}
    if "traceback" in res:
        rec["traceback"] = res["traceback"]
    rec.update({"code_commit": commit, "code_dirty": dirty, "sizes": srec, "seconds": res["seconds"],
                "host": socket.gethostname(), "workers": int(args.workers),
                "job_id": os.environ.get("PBS_JOBID"), "started_utc": started, "finished_utc": _utc_now(),
                "versions": _versions()})
    for key in ("windows", "cell_fit", "sweep_stage", "tables"):
        if key in res:
            rec[key] = res[key]
    write_json_atomic(path, rec)
    _say("NOISE_CELL %d: status=%s reason=%s seconds=%.0f"
         % (args.specimen, rec["status"], _one_line(rec["reason"]), res["seconds"]["total"]))
    return 0


# ===========================================================================
#  The merge (D-125.2)
# ===========================================================================
def cell_row(specimen: int, status: str, reason: str, rec: Optional[dict]) -> Dict[str, object]:
    """One row of noise_cells_<GROUP>.csv from a cell's JSON (None: missing);
    every column the JSON can fill, whatever the status."""
    row: Dict[str, object] = {c: None for c in CELL_COLUMNS}
    row.update({"specimen_id": int(specimen), "status": status, "reason": reason})
    if rec is None:
        return row
    row["noise_code_commit"] = rec.get("code_commit")
    w = rec.get("windows") or {}
    row.update({"noise_t_ss_ms": w.get("t_ss_ms"), "noise_K": w.get("K"), "noise_K_LS": w.get("K_LS")})
    cf = rec.get("cell_fit")
    if cf:
        law = cf["law"]
        tau = law.get("tau_s_s")
        row.update({"noise_sigma_f_mV": law.get("sigma_f_mV"), "noise_rho_f": law.get("rho_f"),
                    "noise_sigma_s_mV": law.get("sigma_s_mV"), "noise_tau_s_ms": None if tau is None else 1e3 * tau,
                    "noise_J": cf.get("J"), "noise_dof": cf.get("dof"), "noise_p": cf.get("p_value"),
                    "noise_kappa": cf.get("kappa"), "noise_n_solves": cf.get("n_solves"),
                    "noise_flags": ";".join(cf.get("flags") or [])})
    tables = rec.get("tables")
    if tables:
        cc = tables["cell_columns"]
        if tuple(cc.keys()) != ("specimen_id",) + BLOCK9_CELL_COLUMNS:
            raise ValueError("specimen %d: Block 9's cell columns %s differ from the merge's"
                             % (specimen, list(cc.keys())))
        row.update({c: cc[c] for c in BLOCK9_CELL_COLUMNS})
    return row


def sweep_rows(specimen: int, rec: dict) -> List[Dict[str, object]]:
    """The rows of noise_sweeps_<GROUP>.csv of one ok cell: Block 9's
    table_rows(), as the cell's JSON holds them."""
    if not rec.get("tables"):
        raise ValueError("specimen %d: status ok but the JSON holds no tables" % specimen)
    rows = rec["tables"]["sweep_rows"]
    for r in rows:
        if tuple(r.keys()) != SWEEP_COLUMNS:
            raise ValueError("specimen %d: Block 9's sweep columns %s differ from the merge's" % (specimen, list(r.keys())))
    return rows


def _csv_value(v) -> str:
    """NaN and None as empty fields; floats by repr (exact round trip)."""
    if v is None:
        return ""
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, float):
        return repr(v) if math.isfinite(v) else ""
    return str(v)


def write_csv_atomic(path: Path, columns: Sequence[str], rows: Sequence[Dict[str, object]]) -> None:
    tmp = path.with_name(".%s.%d.tmp" % (path.name, os.getpid()))
    with open(tmp, "w", encoding="ascii", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns), lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({c: _csv_value(r.get(c)) for c in columns})
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def cmd_merge(args: argparse.Namespace) -> int:
    out = Path(args.out)
    if not out.is_dir():
        _err("no directory %s" % out)
        return 2
    if not _GROUP_NAME.match(args.group):
        _err("--group: letters, digits, '_', '.', '-' only, got %r" % args.group)
        return 2
    found: Dict[int, Path] = {}
    cells_dir = out / "cells"
    if cells_dir.is_dir():
        for p in cells_dir.iterdir():
            m = _SPECIMEN_JSON.match(p.name)
            if m:
                found[int(m.group(1))] = p
    cohort = sorted(set(args.cells)) if args.cells else sorted(found)
    counts = {"ok": 0, "refused": 0, "failed": 0, "missing": 0}
    cell_table, sweep_table, listing = [], [], []
    for sid in cohort:
        rec, status, reason = None, "missing", ""
        if sid in found:
            rec = read_json(found[sid])
            if rec is None:
                status, reason = "failed", "merge: unreadable JSON %s" % found[sid].name
            else:
                status, reason = rec.get("status"), str(rec.get("reason") or "")
                if status not in STATUSES:
                    status, reason, rec = "failed", "merge: unknown status %r" % (status,), None
        counts[status] += 1
        cell_table.append(cell_row(sid, status, reason, rec))
        if status == "ok":
            sweep_table.extend(sweep_rows(sid, rec))
        else:
            listing.append("  %s %d%s" % (status, sid, (": " + _one_line(reason)) if reason else ""))
    write_csv_atomic(out / ("noise_cells_%s.csv" % args.group), CELL_COLUMNS, cell_table)
    write_csv_atomic(out / ("noise_sweeps_%s.csv" % args.group), SWEEP_COLUMNS, sweep_table)
    _say("NOISE_MERGE %s: %d ok, %d refused, %d failed, %d missing"
         % (args.group, counts["ok"], counts["refused"], counts["failed"], counts["missing"]))
    for line in listing:
        _say(line)
    return 0


# ===========================================================================
#  The cohort listing and the command line
# ===========================================================================
def cmd_list(args: argparse.Namespace) -> int:
    group_dir = Path(args.group_dir)
    if not group_dir.is_dir():
        _err("no directory %s" % group_dir)
        return 2
    try:
        ids = list_cohort(group_dir, args.cells)
    except ValueError as exc:
        _err(str(exc))
        return 2
    if not ids:
        _err("no specimen_<id>/ with ss_pulses.npz under %s" % group_dir)
        return 2
    for i in ids:
        _say(str(i))
    return 0


def _specimen_id(text: str) -> int:
    if not re.fullmatch(r"[0-9]+", str(text)):
        raise argparse.ArgumentTypeError("--specimen: a specimen ID (digits), got %r" % (text,))
    return int(text)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="run_noise_fit.py", description="The noise-fit run (SPEC Block 6).")
    sub = p.add_subparsers(dest="command")
    sub.required = True
    pl = sub.add_parser("list", help="the cohort: one specimen ID per line, sorted")
    pl.add_argument("--group-dir", required=True)
    pl.add_argument("--cells", type=parse_cells, default=None)
    pc = sub.add_parser("cell", help="Blocks 3, 4 and 9 on one cell -> <out>/cells/specimen_<id>.json")
    pc.add_argument("--group-dir", required=True)
    pc.add_argument("--specimen", type=_specimen_id, required=True)
    pc.add_argument("--out", required=True)
    pc.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    pc.add_argument("--force", action="store_true")
    pc.add_argument("--sizes", choices=SIZE_NAMES, default="full")
    pm = sub.add_parser("merge", help="the cohort's JSONs -> noise_cells_<GROUP>.csv, noise_sweeps_<GROUP>.csv")
    pm.add_argument("--out", required=True)
    pm.add_argument("--group", required=True)
    pm.add_argument("--cells", type=parse_cells, default=None)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return {"list": cmd_list, "cell": cmd_cell, "merge": cmd_merge}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
