# -*- coding: utf-8 -*-
"""
run_twin_recovery.py -- the Stage-7 run on the twins: plan, one twin (build and fit), merge (Block 8)
=====================================================================================================

specs/SPEC.md, the entry of Block 8 (D-184, D-185, D-188, D-182.3). Entry
point: wiring only -- the plan, the build, the fit and the report are calls of
Blocks 5 (twin_table), 13 (twin_manifest), 12 (twin_build), of
run_ih_fit.main and of ih_recovery_report. Sub-commands:

    python run_twin_recovery.py plan  --group-dir <dir> --noise-dir <dir> --out <dir> [options]
    python run_twin_recovery.py list  --out <dir>
    python run_twin_recovery.py twin  --out <dir> --twin <id> [--force]
    python run_twin_recovery.py merge --out <dir> [--no-plot]

The run directory <out> (SPEC, Block 8, "Outputs"):
    real_cells_<GROUP>.csv, twin_sweeps_<GROUP>.csv, manifest.csv, run_config.json   (plan)
    twins/twin_<id>/specimen_<id>/, fits/twin_<id>/, status/twin_<id>.json            (twin)
    report/, report_without_flagged/, recovery_by_configuration.csv, twins_status.csv (merge)

The options of plan carry the names and the defaults run_ih_recovery.py gives
them: the defaults are read from run_ih_recovery's own parser at run time, so
the two cannot drift apart.

Exit codes: 0 success; 1 a twin that did not end ok (twin), no twin with
results (merge); 2 usage (no run_config.json, a twin id not in the manifest,
missing inputs); 3 refused (plan: a cohort too small, a cell refused by Block 5
or by the loader, a planned run whose options differ without --force).

Pure ASCII, LF (hpc-python-compat).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np    # noqa: E402
import pandas as pd   # noqa: E402

from twin_config import TwinConfig   # noqa: E402

# ---------------------------------------------------------------------------
#  The options of plan, by family (names and defaults: run_ih_recovery.py)
# ---------------------------------------------------------------------------
#: the truth and I_h options: keyword arguments of synth_gt_grid.draw_manifest
#: built as run_ih_recovery.phase_manifest builds them
TRUTH_OPTIONS = ("ra_mode", "cm_phys_lo", "cm_phys_hi", "tau_lo_ms", "tau_hi_ms", "ra_phys_lo", "ra_phys_hi",
                 "gbar_range", "dvh_range", "kappa_range", "gbar_floor")
#: the fit options: passed to run_ih_fit.main as run_ih_recovery.phase_fit passes them
FIT_OPTIONS = ("F", "fit_target", "ss_time_weight", "ss_tau_w_ms", "ss_window_ms", "n_calls", "n_initial",
               "ih_mechanism", "ih_distribution", "ih_regions", "vshift_base", "ehcn", "dt_brief_ms", "dt_long_ms",
               "phase3_subset", "gbar_bounds", "dvh_bounds", "kappa_bounds")
#: the gate's tolerances and switches (run_ih_recovery.phase_report)
TOL_OPTIONS = ("tol_cm_factor", "tol_gbar_factor", "tol_dvh_mV", "tol_kappa_factor", "tol_fp_low_rail_frac",
               "tol_fp_cm_factor", "lenient_rails", "drop_railed")
#: the arm the gate of D-011 applies to (run_ih_recovery's --gate-arm default)
GATE_ARM = "ih6"

EXIT_OK, EXIT_FAILED, EXIT_USAGE, EXIT_REFUSED = 0, 1, 2, 3
REPORT_DIR, REPORT_UNFLAGGED_DIR = "report", "report_without_flagged"
STATUS_FIELDS = ("twin_id", "twin_of_specimen_id", "configuration", "arm", "status", "reason", "build_seconds",
                 "fit_seconds", "built", "fitted", "code_commit", "code_dirty", "host", "job_id", "finished_utc")


class Refused(Exception):
    """A refusal of the entry: exit code and message."""

    def __init__(self, code: int, msg: str):
        super().__init__(msg)
        self.code = int(code)


# ---------------------------------------------------------------------------
#  Small helpers: output, atomic writes, git, versions
# ---------------------------------------------------------------------------
def _say(msg: str) -> None:
    print("[twin_recovery] %s" % msg, flush=True)


def _err(msg: str) -> None:
    sys.stderr.write("[twin_recovery] %s\n" % msg)
    sys.stderr.flush()


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _tmp_of(path: Path) -> Path:
    return path.with_name(".%s.%d.tmp" % (path.name, os.getpid()))


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_of(path)
    with open(tmp, "w", encoding="ascii", newline="\n") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _write_json_atomic(path: Path, obj) -> None:
    _write_text_atomic(path, json.dumps(obj, indent=1, sort_keys=False, default=str) + "\n")


def _write_csv_atomic(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_of(path)
    df.to_csv(tmp, index=False, float_format="%.17g", lineterminator="\n")
    os.replace(tmp, path)


def _read_json(path: Path) -> Optional[dict]:
    try:
        with open(path, encoding="ascii") as fh:
            out = json.load(fh)
    except (OSError, ValueError):
        return None
    return out if isinstance(out, dict) else None


def _git(args: Sequence[str]) -> Optional[str]:
    try:
        res = subprocess.run(["git", "-C", str(HERE)] + list(args), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             universal_newlines=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    return res.stdout.strip() if res.returncode == 0 else None


def code_commit() -> Optional[str]:
    """HEAD of the checkout this file runs from; None without git."""
    return _git(["rev-parse", "HEAD"]) or None


def code_dirty() -> Optional[bool]:
    """True when a file of this folder differs from HEAD (tracked and modified,
    or a new file not ignored); None without git."""
    out = _git(["status", "--porcelain", "--", "."])
    return None if out is None else bool(out)


def versions() -> Dict[str, Optional[str]]:
    import importlib.metadata as im
    out: Dict[str, Optional[str]] = {"python": platform.python_version()}
    for pkg in ("numpy", "scipy", "pandas", "neuron"):
        try:
            out[pkg] = im.version(pkg)
        except im.PackageNotFoundError:
            out[pkg] = None
    return out


# ---------------------------------------------------------------------------
#  The command line
# ---------------------------------------------------------------------------
def recovery_defaults() -> Dict[str, object]:
    """The defaults of run_ih_recovery.py's own parser (its required options
    filled with placeholders: nothing is read or written)."""
    import run_ih_recovery as RIR
    ns = RIR._parse_args(["--output-dir", "-", "--code-dir", "-", "--morph-root", "-"])
    return vars(ns)


def build_parser() -> argparse.ArgumentParser:
    d = recovery_defaults()
    ap = argparse.ArgumentParser(prog="run_twin_recovery.py",
                                 description="Stage 7 on the twins (SPEC, Block 8): plan, twin, merge.")
    sub = ap.add_subparsers(dest="command")
    sub.required = True

    p = sub.add_parser("plan", help="the cohort, the twins' table, the manifest and run_config.json")
    p.add_argument("--group-dir", required=True, help="the archive group of real cells (specimen_<id>/)")
    p.add_argument("--noise-dir", required=True,
                   help="the noise-fit run's group directory (noise_cells_<GROUP>.csv, noise_sweeps_<GROUP>.csv)")
    p.add_argument("--out", required=True, help="the run directory")
    p.add_argument("--group", default=None, help="default: the basename of --group-dir")
    p.add_argument("--cells", default=None, help="comma-separated real specimen ids (default: every ok cell)")
    p.add_argument("--n-per-configuration", type=int, default=10)
    p.add_argument("--seed", type=int, default=d["seed"])
    p.add_argument("--id-base", type=int, default=d["id_base"])
    p.add_argument("--force", action="store_true", help="re-plan a run directory planned with other options")
    # the truth (run_ih_recovery.py's names, types and defaults)
    p.add_argument("--ra-mode", default=d["ra_mode"], choices=["per_cohort", "per_cell"])
    for name in ("cm_phys_lo", "cm_phys_hi", "tau_lo_ms", "tau_hi_ms", "ra_phys_lo", "ra_phys_hi", "gbar_floor"):
        p.add_argument("--" + name.replace("_", "-"), type=float, default=d[name])
    for name in ("gbar_range", "dvh_range", "kappa_range"):
        p.add_argument("--" + name.replace("_", "-"), default=d[name])
    # the fit
    p.add_argument("--F", type=float, default=d["F"])
    p.add_argument("--fit-target", default=d["fit_target"], choices=["dep", "hyp", "both"])
    p.add_argument("--ss-time-weight", default=d["ss_time_weight"], choices=["exp", "gauss", "none"])
    p.add_argument("--ss-tau-w-ms", default=d["ss_tau_w_ms"])
    p.add_argument("--ss-window-ms", default=d["ss_window_ms"])
    p.add_argument("--n-calls", type=int, default=d["n_calls"])
    p.add_argument("--n-initial", type=int, default=d["n_initial"])
    p.add_argument("--ih-mechanism", default=d["ih_mechanism"], choices=["Ih_human", "Ih"])
    p.add_argument("--ih-distribution", default=d["ih_distribution"],
                   choices=["uniform", "eyal_exp_323", "hay_exp_dmax"])
    p.add_argument("--ih-regions", default=d["ih_regions"])
    p.add_argument("--vshift-base", type=float, default=d["vshift_base"])
    p.add_argument("--ehcn", default=d["ehcn"])
    p.add_argument("--dt-brief-ms", type=float, default=d["dt_brief_ms"])
    p.add_argument("--dt-long-ms", type=float, default=d["dt_long_ms"])
    p.add_argument("--phase3-subset", default=d["phase3_subset"])
    p.add_argument("--gbar-bounds", default=d["gbar_bounds"])
    p.add_argument("--dvh-bounds", default=d["dvh_bounds"])
    p.add_argument("--kappa-bounds", default=d["kappa_bounds"])
    # the gate
    for name in ("tol_cm_factor", "tol_gbar_factor", "tol_dvh_mV", "tol_kappa_factor", "tol_fp_low_rail_frac",
                 "tol_fp_cm_factor"):
        p.add_argument("--" + name.replace("_", "-"), type=float, default=d[name])
    p.add_argument("--lenient-rails", action="store_true")
    p.add_argument("--drop-railed", action="store_true")

    p = sub.add_parser("list", help="the twin ids of the manifest, one per line")
    p.add_argument("--out", required=True)

    p = sub.add_parser("twin", help="build one twin (Block 12) and run its arm's fit (run_ih_fit)")
    p.add_argument("--out", required=True)
    p.add_argument("--twin", required=True, type=int)
    p.add_argument("--force", action="store_true", help="rebuild and refit")

    p = sub.add_parser("merge", help="the recovery report (ih_recovery_report)")
    p.add_argument("--out", required=True)
    p.add_argument("--no-plot", action="store_true")
    return ap


# ---------------------------------------------------------------------------
#  plan
# ---------------------------------------------------------------------------
def _parse_cells(text: Optional[str]) -> Optional[List[int]]:
    if text is None or str(text).strip() == "":
        return None
    out: List[int] = []
    for tok in str(text).split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            sid = int(tok)
        except ValueError:
            raise Refused(EXIT_USAGE, "--cells: %r is not a specimen id" % tok)
        if sid not in out:
            out.append(sid)
    return sorted(out)


def plan_options(args) -> Dict[str, object]:
    """Every option of plan with its value (resolved paths), --force and --out
    excluded: the record a later plan is compared with."""
    group_dir = Path(args.group_dir).resolve()
    opts: Dict[str, object] = {
        "group_dir": str(group_dir),
        "noise_dir": str(Path(args.noise_dir).resolve()),
        "group": args.group or group_dir.name,
        "cells": _parse_cells(args.cells),
        "n_per_configuration": int(args.n_per_configuration),
        "seed": int(args.seed),
        "id_base": int(args.id_base),
    }
    for name in TRUTH_OPTIONS + FIT_OPTIONS + TOL_OPTIONS:
        opts[name] = getattr(args, name)
    return opts


def truth_kwargs_of(o) -> Dict[str, object]:
    """synth_gt_grid.draw_manifest's keyword arguments, built as
    run_ih_recovery.phase_manifest builds them, without the noise arguments
    (and without e_pas, cells_per_cohort and max_cells, which are not options
    of plan: the twins' rest is the real cell's, D-192; cells_per_cohort keeps
    draw_manifest's default 10, the value phase_manifest passes)."""
    import run_ih_recovery as RIR
    import synth_gt_grid as G
    ns = argparse.Namespace(**o)
    return dict(
        ra_mode=ns.ra_mode, F=ns.F, cm_phys_lo=ns.cm_phys_lo, cm_phys_hi=ns.cm_phys_hi,
        tau_lo_ms=ns.tau_lo_ms, tau_hi_ms=ns.tau_hi_ms,
        ih_kinetics=ns.ih_mechanism, ih_dist=ns.ih_distribution,
        ih_ehcn_mV=RIR._ehcn_for_truth(ns),
        ih_regions=tuple(r.strip() for r in ns.ih_regions.split(",") if r.strip()),
        ih_vshift_base_mV=ns.vshift_base,
        ih_gbar_range_S_cm2=G._pair_or_none(ns.gbar_range, "--gbar-range"),
        ih_dvh_range_mV=G._pair_or_none(ns.dvh_range, "--dvh-range"),
        ih_kappa_range=G._pair_or_none(ns.kappa_range, "--kappa-range"),
        ih_gbar_floor_S_cm2=ns.gbar_floor,
        ra_phys_lo=ns.ra_phys_lo, ra_phys_hi=ns.ra_phys_hi)


def _paths(out: Path, group: str) -> Dict[str, Path]:
    return dict(real_cells=out / ("real_cells_%s.csv" % group), table=out / ("twin_sweeps_%s.csv" % group),
                manifest=out / "manifest.csv", config=out / "run_config.json")


def cmd_plan(args) -> int:
    import twin_manifest as TM
    import twin_table as TT

    if args.n_initial > args.n_calls:
        raise Refused(EXIT_USAGE, "--n-initial (%d) exceeds --n-calls (%d): gp_minimize refuses this"
                      % (args.n_initial, args.n_calls))
    opts = plan_options(args)
    group, group_dir, noise_dir = str(opts["group"]), Path(opts["group_dir"]), Path(opts["noise_dir"])
    out = Path(args.out).resolve()
    paths = _paths(out, group)

    if paths["manifest"].exists():
        prev = _read_json(paths["config"])
        if prev is not None and prev.get("options") == json.loads(json.dumps(opts)):
            _say("plan: %s is already planned with these options; nothing written" % out)
            return EXIT_OK
        if not args.force:
            if prev is None:
                why = "its run_config.json is missing or unreadable"
            else:
                old = prev.get("options") or {}
                new = json.loads(json.dumps(opts))
                why = "options differ: " + ", ".join(
                    "%s %r -> %r" % (k, old.get(k), new.get(k))
                    for k in sorted(set(old) | set(new)) if old.get(k) != new.get(k))
            raise Refused(EXIT_REFUSED, "plan: %s already holds a manifest and %s; pass --force to re-plan"
                          % (out, why))
        _say("plan: --force -- re-planning %s (existing twins and fits are kept; a twin whose archive or fit "
             "options no longer match the manifest is refused by `twin` until run with --force)" % out)

    if not group_dir.is_dir():
        raise Refused(EXIT_USAGE, "plan: the group directory does not exist: %s" % group_dir)
    cells_csv, sweeps_csv = noise_dir / ("noise_cells_%s.csv" % group), noise_dir / ("noise_sweeps_%s.csv" % group)
    for p in (cells_csv, sweeps_csv):
        if not p.is_file():
            raise Refused(EXIT_USAGE, "plan: the noise-fit table %s does not exist" % p)
    cells_df, sweeps_df = TT.read_noise_tables(cells_csv, sweeps_csv)

    # the cohort
    if opts["cells"] is not None:
        ids = list(opts["cells"])
    else:
        ok = set(int(s) for s, st in zip(cells_df["specimen_id"], cells_df["status"]) if str(st) == "ok")
        ids = []
        for d in sorted(group_dir.glob("specimen_*")):
            try:
                sid = int(d.name[len("specimen_"):])
            except ValueError:
                continue
            if d.is_dir() and sid in ok:
                ids.append(sid)
        ids.sort()
    n = int(opts["n_per_configuration"])
    if len(ids) < n:
        raise Refused(EXIT_REFUSED, "plan: the cohort has %d cell(s) (%s), fewer than --n-per-configuration %d"
                      % (len(ids), ",".join(str(i) for i in ids) or "none", n))
    _say("plan: cohort of %d real cell(s) from %s: %s" % (len(ids), group_dir, " ".join(str(i) for i in ids)))

    import passive_fitting_hpc_fixed as mono
    tcfg = TwinConfig()
    protocols, v_rest = [], {}
    for sid in ids:
        d = group_dir / ("specimen_%d" % sid)
        if not d.is_dir():
            raise Refused(EXIT_REFUSED, "plan: cell %d: %s does not exist" % (sid, d))
        try:
            protocols.append(TT.load_cell_protocol(d, spike_reason_fn=mono.sweep_spike_reason, twin_cfg=tcfg))
            v_rest[sid] = float(mono.load_cell_from_archive(d, verbose=False).v_rest_mV)
        except Exception as exc:   # noqa: BLE001 -- a refusal naming the cell
            raise Refused(EXIT_REFUSED, "plan: cell %d refused: %s: %s" % (sid, type(exc).__name__, exc))
    try:
        table = TT.build_twin_sweep_table(protocols, cells_df, sweeps_df, tcfg)
    except TT.TwinTableError as exc:
        raise Refused(EXIT_REFUSED, "plan: refused by the twins' table (Block 5): %s" % exc)

    rows = []
    for sid in ids:
        t = table[table["specimen_id"] == sid]
        ls = t[t["protocol"] == "LS"]
        flags = set(bool(x) for x in t["cell_tau_s_above_window"])
        if len(flags) != 1:
            raise Refused(EXIT_REFUSED, "plan: cell %d: cell_tau_s_above_window differs between its rows" % sid)
        rows.append(dict(specimen_id=int(sid),
                         swc_path=str(group_dir / ("specimen_%d" % sid) / "reconstruction.swc"),
                         v_rest_mV=v_rest[sid], cell_tau_s_above_window=flags.pop(),
                         n_ls_in_twin=int(ls["in_twin"].sum()), n_ls_left_out=int((~ls["in_twin"]).sum())))
    real_cells = pd.DataFrame(rows, columns=["specimen_id", "swc_path", "v_rest_mV", "cell_tau_s_above_window",
                                             "n_ls_in_twin", "n_ls_left_out"])
    real_cells["specimen_id"] = real_cells["specimen_id"].astype(np.int64)
    real_cells["cell_tau_s_above_window"] = real_cells["cell_tau_s_above_window"].astype(bool)

    try:
        manifest = TM.draw_twin_manifest(real_cells[list(TM.REAL_CELL_COLUMNS)], n_per_configuration=n,
                                         seed=int(opts["seed"]), id_base=int(opts["id_base"]),
                                         truth_kwargs=truth_kwargs_of(opts))
    except ValueError as exc:
        raise Refused(EXIT_REFUSED, "plan: refused by the manifest (Block 13): %s" % exc)

    out.mkdir(parents=True, exist_ok=True)
    _write_csv_atomic(paths["real_cells"], real_cells)
    TT.write_twin_sweep_table(table, paths["table"])
    config = {
        "options": json.loads(json.dumps(opts)),
        "group": group, "group_dir": str(group_dir), "noise_dir": str(noise_dir),
        "noise_cells_csv": str(cells_csv), "noise_sweeps_csv": str(sweeps_csv),
        "twin_config": asdict(tcfg),
        "truth_kwargs": json.loads(json.dumps(truth_kwargs_of(opts))),
        "gate_arm": GATE_ARM,
        "code_commit": code_commit(), "code_dirty": code_dirty(),
        "versions": versions(), "planned_utc": _utc_now(), "host": socket.gethostname(),
    }
    _write_json_atomic(paths["config"], config)
    tmp = _tmp_of(paths["manifest"])
    TM.save_twin_manifest(manifest, tmp)
    os.replace(tmp, paths["manifest"])
    _say("plan: %d twin(s) (%s) on %d real cell(s) -> %s"
         % (len(manifest), ", ".join("%s %d" % (c, int((manifest["configuration"] == c).sum()))
                                      for c in TM.CONFIGURATIONS), len(ids), out))
    return EXIT_OK


# ---------------------------------------------------------------------------
#  The run directory, as written by plan
# ---------------------------------------------------------------------------
def load_run(out) -> Dict[str, object]:
    """run_config.json, the manifest and the paths of a planned run directory;
    Refused(EXIT_USAGE) without run_config.json."""
    import twin_manifest as TM
    out = Path(out).resolve()
    cfg = _read_json(out / "run_config.json")
    if cfg is None:
        raise Refused(EXIT_USAGE, "%s holds no readable run_config.json: run `plan` first" % out)
    paths = _paths(out, str(cfg["group"]))
    if not paths["manifest"].is_file():
        raise Refused(EXIT_USAGE, "%s holds no manifest.csv: run `plan` first" % out)
    return dict(out=out, config=cfg, options=cfg["options"], paths=paths,
                manifest=TM.load_twin_manifest(paths["manifest"]))


def cmd_list(args) -> int:
    run = load_run(args.out)
    for sid in run["manifest"]["specimen_id"]:
        print(int(sid))
    return EXIT_OK


# ---------------------------------------------------------------------------
#  twin
# ---------------------------------------------------------------------------
def fit_argv(options: Dict[str, object], archive_dir, output_dir, arm: str, code_dir=HERE) -> List[str]:
    """The argv of run_ih_fit.main for one twin, in the form
    run_ih_recovery.phase_fit builds it (the '=' form for values that can be
    negative); no --fit-params (the arms ih6 and passive_fullstep)."""
    a = argparse.Namespace(**options)
    argv = ["--archive-dir", str(archive_dir), "--output-dir", str(output_dir),
            "--code-dir", str(code_dir), "--arm", str(arm),
            "--F", str(a.F), "--fit-target", a.fit_target,
            "--ss-time-weight", a.ss_time_weight,
            "--ss-tau-w-ms", str(a.ss_tau_w_ms),
            "--ss-window-ms", str(a.ss_window_ms),
            "--n-calls", str(a.n_calls),
            "--n-initial", str(a.n_initial),
            "--ih-mechanism", a.ih_mechanism,
            "--ih-distribution", a.ih_distribution,
            "--ih-regions", a.ih_regions,
            "--vshift-base=%s" % a.vshift_base,
            "--dt-brief-ms", str(a.dt_brief_ms),
            "--dt-long-ms", str(a.dt_long_ms),
            "--phase3-subset", a.phase3_subset,
            "--gbar-bounds=%s" % a.gbar_bounds,
            "--dvh-bounds=%s" % a.dvh_bounds,
            "--kappa-bounds=%s" % a.kappa_bounds]
    if a.ehcn not in (None, ""):
        argv += ["--ehcn=%s" % a.ehcn]
    return argv


def twin_model(fit_args, configuration: str):
    """(ih_spec, ParamSpec) of the twin's cell, built by the fit's own
    functions from the options the fit receives: run_ih_fit.build_ih_spec, and
    run_ih_fit.build_param_spec on the six names for a configuration with I_h
    (param_spec.make_ih_spec) or the three passive ones for active_fit
    (param_spec.make_passive_spec)."""
    import run_ih_fit as R
    import twin_build as TB
    names = R.IH6_NAMES if TB.HAS_IH[configuration] else R.PASSIVE_NAMES
    pspec = R.build_param_spec(
        names, cm_bounds=R.parse_window(fit_args.cm_bounds), rm_bounds=R.parse_window(fit_args.rm_bounds),
        ra_bounds=R.parse_window(fit_args.ra_bounds), gbar_bounds=R.parse_window(fit_args.gbar_bounds),
        dvh_bounds=R.parse_window(fit_args.dvh_bounds), kappa_bounds=R.parse_window(fit_args.kappa_bounds))
    return R.build_ih_spec(fit_args), pspec


def _clear_sections() -> None:
    import synthetic_ground_truth as sgt
    sgt._clear_neuron_sections()


def _results_row(csv: Path, twin_id: int) -> Optional[pd.DataFrame]:
    """The twin's row(s) of a phase2_results.csv, or None (absent, unreadable, no row)."""
    if not csv.is_file():
        return None
    try:
        df = pd.read_csv(csv)
    except Exception:   # noqa: BLE001 -- an empty or broken file is no result
        return None
    if "specimen_id" not in df.columns:
        return None
    rows = df[pd.to_numeric(df["specimen_id"], errors="coerce") == int(twin_id)]
    return rows if len(rows) else None


def _archive_matches(spec_dir: Path, spec) -> Optional[str]:
    """None when the archive's twin block is this spec's; else why not."""
    meta = _read_json(spec_dir / "metadata.json")
    if meta is None:
        return "metadata.json unreadable"
    tw = meta.get("twin") or {}
    want = dict(twin_of_specimen_id=int(spec.twin_of_specimen_id), configuration=str(spec.configuration),
                theta={k: float(v) for k, v in spec.theta.items()}, F=float(spec.F))
    got = dict(twin_of_specimen_id=tw.get("twin_of_specimen_id"), configuration=tw.get("configuration"),
               theta=tw.get("theta"), F=tw.get("F"))
    if got != want or int(meta.get("specimen_id", -1)) != int(spec.twin_id) \
            or float(meta.get("v_rest_mV", float("nan"))) != float(spec.v_rest_mV):
        return "its metadata (%r) is not the manifest's twin (%r)" % (got, want)
    return None


def cmd_twin(args) -> int:
    import run_ih_fit as R
    import twin_build as TB
    import twin_manifest as TM
    import twin_table as TT

    run = load_run(args.out)
    out, opts, man = run["out"], run["options"], run["manifest"]
    rows = man[man["specimen_id"] == int(args.twin)]
    if len(rows) != 1:
        raise Refused(EXIT_USAGE, "twin %d is not in the manifest %s" % (int(args.twin), run["paths"]["manifest"]))
    row = rows.iloc[0]
    tid, conf, arm = int(row["specimen_id"]), str(row["configuration"]), str(row["arm"])
    real_id = int(row["twin_of_specimen_id"])
    group_dir = out / "twins" / ("twin_%d" % tid)
    spec_dir = group_dir / ("specimen_%d" % tid)
    fit_dir = out / "fits" / ("twin_%d" % tid)
    csv = fit_dir / "phase2_results.csv"
    argv_record = fit_dir / "fit_argv.json"
    status = dict(twin_id=tid, twin_of_specimen_id=real_id, configuration=conf, arm=arm, status="",
                  reason="", build_seconds=0.0, fit_seconds=0.0, built=False, fitted=False,
                  code_commit=code_commit(), code_dirty=code_dirty(), host=socket.gethostname(),
                  job_id=os.environ.get("PBS_JOBID"), finished_utc=None)
    _say("twin %d (%s, arm %s, real cell %d) in %s" % (tid, conf, arm, real_id, out))

    def finish(st: str, reason: str) -> int:
        status.update(status=st, reason=reason, finished_utc=_utc_now())
        _write_json_atomic(out / "status" / ("twin_%d.json" % tid), status)
        _say("TWIN %d: status=%s reason=%s build=%.1fs fit=%.1fs"
             % (tid, st, reason or "-", status["build_seconds"], status["fit_seconds"]))
        return EXIT_OK if st == "ok" else EXIT_FAILED

    argv = fit_argv(opts, group_dir, fit_dir, arm)

    # ---- the build (Block 12) ---------------------------------------------
    t0 = time.time()
    try:
        spec = TM.twin_spec_of(row, F=float(opts["F"]))
        complete = (spec_dir / "metadata.json").is_file()
        if complete and not args.force:
            why = _archive_matches(spec_dir, spec)
            if why is not None:
                return finish("build_failed", "the archive %s exists but %s; run with --force to rebuild"
                              % (spec_dir, why))
            _say("twin %d: archive complete (%s); not rebuilt" % (tid, spec_dir))
        else:
            fit_args = R._parse_args(argv)
            ih_spec, pspec = twin_model(fit_args, conf)
            table = TT.read_twin_sweep_table(run["paths"]["table"])
            real_dir = Path(run["config"]["group_dir"]) / ("specimen_%d" % real_id)
            _clear_sections()
            try:
                TB.build_twin(spec, real_dir, table, group_dir, ih_spec=ih_spec, apply_theta=pspec.apply,
                              cfg=TwinConfig(), overwrite=True,
                              provenance=dict(run_dir=str(out), code_commit=status["code_commit"],
                                              code_dirty=status["code_dirty"], manifest_seed=int(row["manifest_seed"]),
                                              driver="run_twin_recovery.py"))
            finally:
                _clear_sections()
            status["built"] = True
            _say("twin %d: built %s in %.1f s" % (tid, spec_dir, time.time() - t0))
    except Exception as exc:   # noqa: BLE001 -- recorded in the status file, never raised past it
        status["build_seconds"] = time.time() - t0
        traceback.print_exc()
        return finish("build_failed", "%s: %s" % (type(exc).__name__, exc))
    status["build_seconds"] = time.time() - t0

    # ---- the fit (run_ih_fit.main, the arm of the configuration) ----------
    t1 = time.time()
    need_fit = bool(args.force or status["built"] or not csv.exists())
    if not need_fit:
        prev = _read_json(argv_record)
        if prev is not None and prev.get("argv") != argv:
            return finish("fit_failed", "%s was fitted with other options (%s); run with --force to refit"
                          % (csv, argv_record))
        _say("twin %d: %s exists; not refitted" % (tid, csv))
    else:
        fit_dir.mkdir(parents=True, exist_ok=True)
        if csv.exists():
            csv.unlink()
        _write_json_atomic(argv_record, dict(argv=argv, started_utc=_utc_now()))
        _say("twin %d: run_ih_fit.main %s" % (tid, " ".join(argv)))
        exit_note = ""
        try:
            R.main(argv)
        except SystemExit as exc:             # run_ih_fit exits 1 when no cell was fitted
            if exc.code not in (None, 0):
                exit_note = "run_ih_fit exited %r" % (exc.code,)
        except Exception as exc:              # noqa: BLE001
            traceback.print_exc()
            exit_note = "run_ih_fit raised %s: %s" % (type(exc).__name__, exc)
        finally:
            _clear_sections()
        status["fitted"] = True
        status["fit_seconds"] = time.time() - t1
        if exit_note:
            return finish("fit_failed", exit_note)
    status["fit_seconds"] = time.time() - t1
    res = _results_row(csv, tid)
    if res is None:
        return finish("fit_failed", "%s holds no row of twin %d" % (csv, tid))
    return finish("ok", "")


# ---------------------------------------------------------------------------
#  merge
# ---------------------------------------------------------------------------
def _tolerances(o):
    import ih_recovery_report as RR
    return RR.RecoveryTolerances(
        log_ratio_cm=float(np.log(o["tol_cm_factor"])),
        log_ratio_gbar=float(np.log(o["tol_gbar_factor"])),
        abs_dvh_mV=float(o["tol_dvh_mV"]),
        log_ratio_kappa=float(np.log(o["tol_kappa_factor"])),
        fp_min_low_rail_frac=float(o["tol_fp_low_rail_frac"]),
        fp_log_ratio_cm=float(np.log(o["tol_fp_cm_factor"])))


def _box(o) -> Dict[str, tuple]:
    import ih_recovery_report as RR
    box = dict(RR.DEFAULT_BOX)
    for axis, text in (("gbar", o["gbar_bounds"]), ("dv_h", o["dvh_bounds"]), ("kappa_tau", o["kappa_bounds"])):
        lo, hi = (float(x) for x in str(text).split(","))
        box[axis] = (lo, hi)
    return box


def write_one_report(out_dir: Path, manifest: pd.DataFrame, frames: Sequence[pd.DataFrame], o: Dict[str, object],
                     *, plot: bool) -> pd.DataFrame:
    """The report of run_ih_recovery.phase_report on these frames: returns the long table."""
    import ih_recovery_report as RR
    long_df = RR.recovery_long(manifest, list(frames), box=_box(o), strict_rails=not bool(o["lenient_rails"]))
    summary = RR.recovery_summary(long_df, drop_railed=bool(o["drop_railed"]))
    inflation = RR.cm_inflation(long_df)
    fp = RR.fp_control_summary(long_df)
    tol = _tolerances(o)
    verdict, passed = RR.evaluate_gate(summary, fp, arm=GATE_ARM, tol=tol)
    RR.write_report(out_dir, long_df=long_df, summary=summary, inflation=inflation, fp=fp, verdict=verdict,
                    passed=passed, tol=tol)
    print("== %s" % out_dir, flush=True)
    print(RR.format_report(summary, inflation, fp, verdict, passed), flush=True)
    if plot:
        try:
            p = RR.plot_recovery(long_df, out_dir / "recovery.png", tol=tol)
            _say("recovery figure -> %s" % p)
        except Exception as exc:   # noqa: BLE001 -- a figure is not the result
            _say("recovery figure skipped: %s: %s" % (type(exc).__name__, exc))
    return long_df


def arm_frames(manifest: pd.DataFrame, frames: Dict[int, pd.DataFrame]) -> List[pd.DataFrame]:
    """One results frame per arm, as run_ih_recovery.phase_report gives
    recovery_long one frame per arm: the rows of the manifest's twins that have
    results, in the manifest's order, the arms in the order they first appear
    in the manifest."""
    out: List[pd.DataFrame] = []
    for arm in pd.unique(manifest["arm"].astype(str)):
        ids = [int(t) for t, a in zip(manifest["specimen_id"], manifest["arm"]) if str(a) == arm and int(t) in frames]
        if ids:
            out.append(pd.concat([frames[t] for t in ids], ignore_index=True, sort=False))
    return out


def cmd_merge(args) -> int:
    import ih_recovery_report as RR

    run = load_run(args.out)
    out, o, man = run["out"], run["options"], run["manifest"]

    # twins_status.csv: one row per manifest twin
    srows = []
    for _, r in man.iterrows():
        tid = int(r["specimen_id"])
        st = _read_json(out / "status" / ("twin_%d.json" % tid))
        if st is None:
            st = dict(twin_id=tid, twin_of_specimen_id=int(r["twin_of_specimen_id"]),
                      configuration=str(r["configuration"]), arm=str(r["arm"]), status="missing")
        srows.append({k: st.get(k) for k in STATUS_FIELDS})
    _write_csv_atomic(out / "twins_status.csv", pd.DataFrame(srows, columns=list(STATUS_FIELDS)))

    # the results: each twin's row of its own phase2_results.csv, tagged with its arm
    frames: Dict[int, pd.DataFrame] = {}
    for _, r in man.iterrows():
        tid = int(r["specimen_id"])
        csv = out / "fits" / ("twin_%d" % tid) / "phase2_results.csv"
        if _results_row(csv, tid) is None:
            continue
        df = RR.load_arm_results(csv, str(r["arm"]))
        frames[tid] = df[pd.to_numeric(df["specimen_id"], errors="coerce") == tid].reset_index(drop=True)
    n_missing = len(man) - len(frames)
    _say("merge: %d of %d twin(s) with results" % (len(frames), len(man)))
    if not frames:
        _err("merge: no twin has results (fits/twin_<id>/phase2_results.csv with its row); no report written")
        return EXIT_FAILED

    long_df = write_one_report(out / REPORT_DIR, man, arm_frames(man, frames), o, plot=not args.no_plot)

    flagged = set(int(t) for t, f in zip(man["specimen_id"], man["cell_tau_s_above_window"]) if bool(f))
    man_u = man[~man["specimen_id"].isin(flagged)].reset_index(drop=True)
    frames_u = arm_frames(man_u, frames)
    unflagged_dir = out / REPORT_UNFLAGGED_DIR
    if frames_u:
        write_one_report(unflagged_dir, man_u, frames_u, o, plot=not args.no_plot)
    else:
        if unflagged_dir.exists():
            shutil.rmtree(unflagged_dir)
        _write_text_atomic(unflagged_dir / "NO_RESULTS.txt",
                           "No twin of an unflagged real cell (cell_tau_s_above_window False) has results; "
                           "no report.\n")
        _say("merge: no unflagged twin with results; %s/NO_RESULTS.txt" % unflagged_dir)

    join = man[["specimen_id", "configuration", "twin_of_specimen_id", "cell_tau_s_above_window"]]
    by_conf = long_df.merge(join, on="specimen_id", how="left")
    _write_csv_atomic(out / "recovery_by_configuration.csv", by_conf)
    _say("merge: report -> %s; without the %d flagged twin(s) -> %s; %d twin(s) without results"
         % (out / REPORT_DIR, len(flagged), unflagged_dir, n_missing))
    return EXIT_OK


# ---------------------------------------------------------------------------
#  main
# ---------------------------------------------------------------------------
COMMANDS = {"plan": cmd_plan, "list": cmd_list, "twin": cmd_twin, "merge": cmd_merge}


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(COMMANDS[args.command](args))
    except Refused as exc:
        _err(str(exc))
        return exc.code


if __name__ == "__main__":
    sys.exit(main())
