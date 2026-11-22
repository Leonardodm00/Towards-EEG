# -*- coding: utf-8 -*-
"""
run_twin_recovery.py -- the Stage-7 run on the twins: plan, one twin (build and fit), merge (Blocks 8, 17)
=========================================================================================================

specs/SPEC.md, the entries of Block 8 (D-184, D-185, D-188, D-182.3) and of
Block 17 (the second run, SPEC 2.8: D-204, D-206 to D-209). Entry point:
wiring only -- the plan, the build, the fit and the report are calls of
Blocks 5 (twin_table), 13/16 (twin_manifest), 12/18 (twin_build), of
run_ih_fit.main (Block 15) and of ih_recovery_report. Sub-commands:

    python run_twin_recovery.py plan  --group-dir <dir> --noise-dir <dir> --out <dir> [options]
    python run_twin_recovery.py list  --out <dir>
    python run_twin_recovery.py twin  --out <dir> --twin <id> [--force]
    python run_twin_recovery.py merge --out <dir> [--no-plot]

The run directory <out> (SPEC, Block 8, "Outputs"):
    real_cells_<GROUP>.csv, twin_sweeps_<GROUP>.csv, manifest.csv, run_config.json   (plan)
    twins/twin_<id>/specimen_<id>/, fits/twin_<id>/, status/twin_<id>.json            (twin)
    report/, report_without_flagged/, report_no_noise/, recovery_by_configuration.csv,
    recovery_extras.csv, twins_status.csv                                             (merge)

Block 17: four configurations (the fourth, full_no_noise, built without noise
and reported apart in report_no_noise/); the box and the truth of the I_h time
constant given as tau_h_pk in ms (--tau-h-pk-bounds-ms, --tau-h-pk-truth-ms,
converted to kappa_tau with run_ih_fit.tau_h_base_peak_ms); dv_h's truth 0;
the fit of a twin with I_h fitted with I_h evaluated also at the truth
(--eval-theta); the gate on ih5 with the required axes Cm, gbar, kappa_tau.

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
import dataclasses
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
#: (Block 17: --dvh-range dropped, --kappa-range replaced by --tau-h-pk-truth-ms)
TRUTH_OPTIONS = ("ra_mode", "cm_phys_lo", "cm_phys_hi", "tau_lo_ms", "tau_hi_ms", "ra_phys_lo", "ra_phys_hi",
                 "gbar_range", "tau_h_pk_truth_ms", "gbar_floor")
#: the fit options: passed to run_ih_fit.main as run_ih_recovery.phase_fit passes them
#: (Block 17: --kappa-bounds replaced by --tau-h-pk-bounds-ms)
FIT_OPTIONS = ("F", "fit_target", "ss_time_weight", "ss_tau_w_ms", "ss_window_ms", "n_calls", "n_initial",
               "ih_mechanism", "ih_distribution", "ih_regions", "vshift_base", "ehcn", "dt_brief_ms", "dt_long_ms",
               "phase3_subset", "gbar_bounds", "dvh_bounds", "tau_h_pk_bounds_ms")
#: the gate's tolerances and switches (run_ih_recovery.phase_report)
TOL_OPTIONS = ("tol_cm_factor", "tol_gbar_factor", "tol_dvh_mV", "tol_kappa_factor", "tol_fp_low_rail_frac",
               "tol_fp_cm_factor", "lenient_rails", "drop_railed")
#: the arm the gate applies to (Block 17, SPEC 2.8 (g)) and its required axes: dv_h is not an
#: axis of ih5, and its absence does not fail the gate
GATE_ARM = "ih5"
GATE_REQUIRED_AXES = ("Cm", "gbar", "kappa_tau")
#: Block 17's new defaults (D-208; SPEC 2.8 (a), (e)): the gbar box (S/cm^2), tau_h_pk's box and the
#: range of its truth (ms), the number of noise-free twins, the truth of dv_h (mV)
GBAR_BOUNDS_DEFAULT = "1e-6,1e-2"
TAU_H_PK_BOUNDS_MS_DEFAULT = "50,1000"
TAU_H_PK_TRUTH_MS_DEFAULT = "150,700"
N_NO_NOISE_DEFAULT = 5
DVH_TRUTH_MV = 0.0
#: loss_verdict of recovery_extras.csv: one loss lower than the other by more than this fraction of
#: the other (Block 17)
LOSS_VERDICT_REL = 0.02

EXIT_OK, EXIT_FAILED, EXIT_USAGE, EXIT_REFUSED = 0, 1, 2, 3
REPORT_DIR, REPORT_UNFLAGGED_DIR, REPORT_NO_NOISE_DIR = "report", "report_without_flagged", "report_no_noise"
#: the columns of recovery_extras.csv (Block 17), in order
EXTRAS_COLUMNS = ("twin_id", "configuration", "twin_of_specimen_id", "paired_with", "with_noise",
                  "cell_tau_s_above_window", "tau_h_pk_true_ms", "tau_h_pk_est_ms", "loss_at_truth",
                  "loss_at_estimate", "loss_verdict", "v_deepest_train_mV", "gh_deep_true_S_cm2",
                  "gh_deep_est_S_cm2", "ln_gh_deep_ratio")
#: the columns of ih_recovery_report.recovery_summary (its own empty frame)
SUMMARY_COLUMNS = ("arm", "parameter", "n", "n_railed", "median_abs_err", "q25_abs_err", "q75_abs_err",
                   "median_err", "err_kind")
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
            # no pip metadata (NEURON in davinci's env, amendment 10.2): the imported module's own version
            mod = sys.modules.get(pkg)
            out[pkg] = getattr(mod, "__version__", None) if mod is not None else None
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


def _non_negative_int(text: str) -> int:
    """argparse type of --base-seed (amendment 9.2): SeedSequence refuses a negative entry."""
    try:
        v = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError("%r is not an integer" % text)
    if v < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer, got %d" % v)
    return v


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
    p.add_argument("--n-no-noise", type=int, default=N_NO_NOISE_DEFAULT,
                   help="the full_no_noise twins (SPEC 2.8 (e)): the first k full twins copied, built without "
                        "noise; 0 <= k <= --n-per-configuration")
    p.add_argument("--seed", type=int, default=d["seed"])
    p.add_argument("--id-base", type=int, default=d["id_base"])
    p.add_argument("--base-seed", type=_non_negative_int, default=TwinConfig().base_seed,
                   help="root of the twins' noise streams, SPEC 2.7 (e) (amendment 8.3); recorded as "
                        "run_config.json's twin_config.base_seed")
    p.add_argument("--force", action="store_true", help="re-plan a run directory planned with other options")
    # the truth (run_ih_recovery.py's names, types and defaults)
    p.add_argument("--ra-mode", default=d["ra_mode"], choices=["per_cohort", "per_cell"])
    for name in ("cm_phys_lo", "cm_phys_hi", "tau_lo_ms", "tau_hi_ms", "ra_phys_lo", "ra_phys_hi", "gbar_floor"):
        p.add_argument("--" + name.replace("_", "-"), type=float, default=d[name])
    p.add_argument("--gbar-range", default=d["gbar_range"])
    p.add_argument("--tau-h-pk-truth-ms", default=TAU_H_PK_TRUTH_MS_DEFAULT,
                   help="'LO,HI' (ms): the range of the truth of tau_h_pk, drawn by draw_manifest's own law "
                        "through its kappa range (LO/base, HI/base), base = run_ih_fit.tau_h_base_peak_ms "
                        "(SPEC 2.8 (e)); replaces --kappa-range")
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
    p.add_argument("--gbar-bounds", default=GBAR_BOUNDS_DEFAULT, help="S/cm^2 (D-208)")
    p.add_argument("--dvh-bounds", default=d["dvh_bounds"])
    p.add_argument("--tau-h-pk-bounds-ms", default=TAU_H_PK_BOUNDS_MS_DEFAULT,
                   help="'LO,HI' (ms): the box of tau_h_pk, passed to run_ih_fit, which applies it as the box "
                        "(LO/base, HI/base) of kappa_tau (SPEC 2.8 (a)); replaces --kappa-bounds")
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
        "n_no_noise": int(args.n_no_noise),
        "seed": int(args.seed),
        "id_base": int(args.id_base),
        "base_seed": int(args.base_seed),
    }
    for name in TRUTH_OPTIONS + FIT_OPTIONS + TOL_OPTIONS:
        opts[name] = getattr(args, name)
    return opts


def fit_args_of(o):
    """run_ih_fit's parsed options for these plan options (placeholder
    directories, arm ih5): the fit's own parser, read for the I_h spec."""
    import run_ih_fit as R
    return R._parse_args(fit_argv(o, "-", "-", "ih5"))


def tau_h_pk_record(o) -> Dict[str, object]:
    """The box and the truth range of tau_h_pk (ms, as given) and as
    kappa_tau (divided by base = run_ih_fit.tau_h_base_peak_ms of the run's
    I_h spec, SPEC 2.8 (a), (e)): the record of run_config.json."""
    import run_ih_fit as R
    try:
        base = float(R.tau_h_base_peak_ms(R.build_ih_spec(fit_args_of(o))))
        b_lo, b_hi = (float(x) for x in R.parse_window(o["tau_h_pk_bounds_ms"]))
        t_lo, t_hi = (float(x) for x in R.parse_window(o["tau_h_pk_truth_ms"]))
    except (ValueError, TypeError) as exc:
        raise Refused(EXIT_USAGE, "plan: --tau-h-pk-bounds-ms / --tau-h-pk-truth-ms: %s" % exc)
    for name, lo, hi in (("--tau-h-pk-bounds-ms", b_lo, b_hi), ("--tau-h-pk-truth-ms", t_lo, t_hi)):
        if not (np.isfinite(lo) and np.isfinite(hi) and 0.0 < lo < hi):
            raise Refused(EXIT_USAGE, "plan: %s must be LO,HI with 0 < LO < HI (ms), got %r,%r" % (name, lo, hi))
    if not (np.isfinite(base) and base > 0.0):
        raise Refused(EXIT_USAGE, "plan: tau_h_base_peak_ms of the run's I_h spec is %r" % base)
    return dict(base_ms=base, bounds_ms=[b_lo, b_hi], bounds_kappa=[b_lo / base, b_hi / base],
                truth_ms=[t_lo, t_hi], truth_kappa=[t_lo / base, t_hi / base])


def truth_kwargs_of(o, tau_rec: Dict[str, object]) -> Dict[str, object]:
    """synth_gt_grid.draw_manifest's keyword arguments, built as
    run_ih_recovery.phase_manifest builds them, without the noise arguments
    (and without e_pas, cells_per_cohort and max_cells, which are not options
    of plan: the twins' rest is the real cell's, D-192; cells_per_cohort keeps
    draw_manifest's default 10, the value phase_manifest passes). Block 17:
    no dv_h range (draw_manifest's None: dv_h 0, and the manifest's
    dvh_truth_mV = 0); the kappa range is tau_h_pk's truth range as kappa."""
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
        ih_dvh_range_mV=None,
        ih_kappa_range=(float(tau_rec["truth_kappa"][0]), float(tau_rec["truth_kappa"][1])),
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
    tau_rec = tau_h_pk_record(opts)
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
    tcfg = dataclasses.replace(TwinConfig(), base_seed=int(opts["base_seed"]))
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
                                         n_no_noise=int(opts["n_no_noise"]), seed=int(opts["seed"]),
                                         id_base=int(opts["id_base"]), dvh_truth_mV=DVH_TRUTH_MV,
                                         truth_kwargs=truth_kwargs_of(opts, tau_rec))
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
        "truth_kwargs": json.loads(json.dumps(truth_kwargs_of(opts, tau_rec))),
        "tau_h_pk": tau_rec,
        "dvh_truth_mV": DVH_TRUTH_MV,
        "gate_arm": GATE_ARM, "gate_required_axes": list(GATE_REQUIRED_AXES),
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


def recorded_twin_config(config: Dict[str, object]) -> TwinConfig:
    """The TwinConfig recorded in run_config.json (amendment 8.3); Refused(EXIT_REFUSED) when its keys are
    not exactly the dataclass's fields."""
    rec = config.get("twin_config")
    fields = [f.name for f in dataclasses.fields(TwinConfig)]
    if not isinstance(rec, dict) or sorted(rec) != sorted(fields):
        got = sorted(rec) if isinstance(rec, dict) else rec
        raise Refused(EXIT_REFUSED, "run_config.json's twin_config %r does not hold exactly the fields of "
                                    "TwinConfig %r" % (got, fields))
    return TwinConfig(**rec)


def cmd_list(args) -> int:
    run = load_run(args.out)
    for sid in run["manifest"]["specimen_id"]:
        print(int(sid))
    return EXIT_OK


# ---------------------------------------------------------------------------
#  twin
# ---------------------------------------------------------------------------
def fit_argv(options: Dict[str, object], archive_dir, output_dir, arm: str, code_dir=HERE,
             eval_theta: Optional[str] = None) -> List[str]:
    """The argv of run_ih_fit.main for one twin, in the form
    run_ih_recovery.phase_fit builds it (the '=' form for values that can be
    negative); no --fit-params (the arms' own names). Block 17: no
    --kappa-bounds; --tau-h-pk-bounds-ms (the box of kappa_tau given in ms)
    and, when given, --eval-theta."""
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
            "--tau-h-pk-bounds-ms=%s" % a.tau_h_pk_bounds_ms]
    if a.ehcn not in (None, ""):
        argv += ["--ehcn=%s" % a.ehcn]
    if eval_theta is not None:
        argv += ["--eval-theta=%s" % eval_theta]
    return argv


def eval_theta_of(row, arm: str) -> Optional[str]:
    """--eval-theta for one twin (Block 17): the truth in the fitted spec's
    names, for a twin whose configuration has I_h and is fitted with I_h
    (full, full_no_noise) and whose truth holds every fitted name; None
    otherwise (active_fit, passive_fit). Values by repr: they read back as
    the same float64."""
    import run_ih_fit as R
    import twin_build as TB
    import twin_manifest as TM
    conf = str(row["configuration"])
    cfg = R.ARMS[str(arm)]
    theta = TM.twin_theta(row)
    if not (TB.HAS_IH[conf] and cfg.attach_ih and all(n in theta for n in cfg.fit_params)):
        return None
    return ",".join("%s=%r" % (n, float(theta[n])) for n in cfg.fit_params)


def twin_model(fit_args, configuration: str):
    """(ih_spec, ParamSpec) of the twin's cell, built by the fit's own
    functions from the options the fit receives: run_ih_fit.build_ih_spec, and
    run_ih_fit.build_param_spec on the six names for a configuration with I_h
    (param_spec.make_ih_spec) or the three passive ones for active_fit
    (param_spec.make_passive_spec). The kappa_tau box is the fit's own
    (run_ih_fit.kappa_bounds_from_args: --tau-h-pk-bounds-ms, Block 17)."""
    import run_ih_fit as R
    import twin_build as TB
    names = R.IH6_NAMES if TB.HAS_IH[configuration] else R.PASSIVE_NAMES
    pspec = R.build_param_spec(
        names, cm_bounds=R.parse_window(fit_args.cm_bounds), rm_bounds=R.parse_window(fit_args.rm_bounds),
        ra_bounds=R.parse_window(fit_args.ra_bounds), gbar_bounds=R.parse_window(fit_args.gbar_bounds),
        dvh_bounds=R.parse_window(fit_args.dvh_bounds), kappa_bounds=R.kappa_bounds_from_args(fit_args))
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


def _archive_matches(spec_dir: Path, spec, tcfg: TwinConfig, with_noise: bool) -> Optional[str]:
    """None when the archive's twin block is this spec's (and, Block 17, its
    with_noise the manifest's); else why not."""
    meta = _read_json(spec_dir / "metadata.json")
    if meta is None:
        return "metadata.json unreadable"
    tw = meta.get("twin") or {}
    want = dict(twin_of_specimen_id=int(spec.twin_of_specimen_id), configuration=str(spec.configuration),
                theta={k: float(v) for k, v in spec.theta.items()}, F=float(spec.F), base_seed=int(tcfg.base_seed),
                with_noise=bool(with_noise))
    got = dict(twin_of_specimen_id=tw.get("twin_of_specimen_id"), configuration=tw.get("configuration"),
               theta=tw.get("theta"), F=tw.get("F"), base_seed=tw.get("base_seed"),
               with_noise=tw.get("with_noise"))
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
    tcfg = recorded_twin_config(run["config"])
    rows = man[man["specimen_id"] == int(args.twin)]
    if len(rows) != 1:
        raise Refused(EXIT_USAGE, "twin %d is not in the manifest %s" % (int(args.twin), run["paths"]["manifest"]))
    row = rows.iloc[0]
    tid, conf, arm = int(row["specimen_id"]), str(row["configuration"]), str(row["arm"])
    real_id = int(row["twin_of_specimen_id"])
    with_noise = bool(row["with_noise"])
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

    try:
        ev = eval_theta_of(row, arm)
    except Exception as exc:   # noqa: BLE001 -- recorded in the status file, never raised past it
        return finish("build_failed", "the truth of twin %d: %s: %s" % (tid, type(exc).__name__, exc))
    argv = fit_argv(opts, group_dir, fit_dir, arm, eval_theta=ev)

    # ---- the build (Block 12) ---------------------------------------------
    t0 = time.time()
    try:
        spec = TM.twin_spec_of(row, F=float(opts["F"]))
        complete = (spec_dir / "metadata.json").is_file()
        if complete and not args.force:
            why = _archive_matches(spec_dir, spec, tcfg, with_noise)
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
                              cfg=tcfg, overwrite=True, with_noise=with_noise,
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


def _box(config: Dict[str, object]) -> Dict[str, tuple]:
    """The fit's box per axis: the defaults of ih_recovery_report, the run's gbar and dv_h boxes, and
    kappa_tau's box as run_config.json records it (tau_h_pk's box divided by the base, Block 17)."""
    import ih_recovery_report as RR
    o = config["options"]
    box = dict(RR.DEFAULT_BOX)
    for axis, text in (("gbar", o["gbar_bounds"]), ("dv_h", o["dvh_bounds"])):
        lo, hi = (float(x) for x in str(text).split(","))
        box[axis] = (lo, hi)
    lo, hi = config["tau_h_pk"]["bounds_kappa"]
    box["kappa_tau"] = (float(lo), float(hi))
    return box


def summarised_rows(long_df: pd.DataFrame, drop_railed: bool) -> int:
    """The number of rows recovery_summary would summarise (amendment 9.1): the long table's rows that
    are not false-positive controls and, with drop_railed, not railed."""
    if not len(long_df):
        return 0
    keep = ~long_df["is_fp_control"].astype(bool)
    if drop_railed:
        keep &= ~long_df["rail"].astype(bool)
    return int(keep.sum())


def write_one_report(out_dir: Path, manifest: pd.DataFrame, frames: Sequence[pd.DataFrame],
                     config: Dict[str, object], *, plot: bool) -> pd.DataFrame:
    """The report of run_ih_recovery.phase_report on these frames, the gate on GATE_ARM with its
    required axes (Block 17): returns the long table."""
    import ih_recovery_report as RR
    o = config["options"]
    long_df = RR.recovery_long(manifest, list(frames), box=_box(config), strict_rails=not bool(o["lenient_rails"]))
    if summarised_rows(long_df, bool(o["drop_railed"])) > 0:
        summary = RR.recovery_summary(long_df, drop_railed=bool(o["drop_railed"]))
    else:
        # amendments 8.1, 9.1: no row for recovery_summary to summarise (only false-positive controls,
        # or --drop-railed leaves none); it would raise KeyError, so the summary is its own empty frame
        summary = pd.DataFrame(columns=list(SUMMARY_COLUMNS))
    inflation = RR.cm_inflation(long_df)
    fp = RR.fp_control_summary(long_df)
    tol = _tolerances(o)
    verdict, passed = RR.evaluate_gate(summary, fp, arm=GATE_ARM, tol=tol, required_axes=GATE_REQUIRED_AXES)
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
    run = load_run(args.out)
    out, man = run["out"], run["manifest"]

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
        df = load_results_round_trip(csv, str(r["arm"]))
        frames[tid] = df[pd.to_numeric(df["specimen_id"], errors="coerce") == tid].reset_index(drop=True)
    n_missing = len(man) - len(frames)
    _say("merge: %d of %d twin(s) with results" % (len(frames), len(man)))
    if not frames:
        _err("merge: no twin has results (fits/twin_<id>/phase2_results.csv with its row); no report written")
        return EXIT_FAILED

    cfg = run["config"]
    plot = not args.no_plot
    noisy = man["with_noise"].astype(bool).to_numpy()
    man_n = man[noisy].reset_index(drop=True)          # report/, report_without_flagged/: the noisy twins
    man_0 = man[~noisy].reset_index(drop=True)         # report_no_noise/: the full_no_noise twins alone
    flagged = set(int(t) for t, f in zip(man["specimen_id"], man["cell_tau_s_above_window"]) if bool(f))
    man_u = man_n[~man_n["specimen_id"].isin(flagged)].reset_index(drop=True)
    longs = []
    for sub_dir, sub_man, what in ((REPORT_DIR, man_n, "noisy twin"),
                                   (REPORT_UNFLAGGED_DIR, man_u,
                                    "noisy twin of an unflagged real cell (cell_tau_s_above_window False)"),
                                   (REPORT_NO_NOISE_DIR, man_0, "noise-free (full_no_noise) twin")):
        long_df = write_subset_report(out / sub_dir, sub_man, frames, cfg, plot=plot, what=what)
        if sub_dir != REPORT_UNFLAGGED_DIR and long_df is not None:
            longs.append(long_df)

    join = man[["specimen_id", "configuration", "twin_of_specimen_id", "cell_tau_s_above_window"]]
    by_conf = pd.concat(longs, ignore_index=True, sort=False).merge(join, on="specimen_id", how="left")
    _write_csv_atomic(out / "recovery_by_configuration.csv", by_conf)
    _write_csv_atomic(out / "recovery_extras.csv", recovery_extras(man, frames, cfg))
    _say("merge: report -> %s (%d noisy twin(s)); without the flagged twins -> %s; noise-free twins -> %s; "
         "recovery_extras.csv; %d twin(s) without results"
         % (out / REPORT_DIR, len(man_n), out / REPORT_UNFLAGGED_DIR, out / REPORT_NO_NOISE_DIR, n_missing))
    return EXIT_OK


def load_results_round_trip(csv: Path, arm: str) -> pd.DataFrame:
    """ih_recovery_report.load_arm_results with the floats read back exactly (amendment 3:
    float_precision="round_trip", so recovery_extras.csv and the reports carry the fit's values): the
    CSV, tagged with the arm; KeyError without a specimen_id column, as load_arm_results."""
    df = pd.read_csv(csv, float_precision="round_trip")
    if "specimen_id" not in df.columns:
        raise KeyError("{} has no specimen_id column".format(csv))
    df = df.copy()
    df["arm"] = str(arm)
    return df


def write_subset_report(out_dir: Path, sub_man: pd.DataFrame, frames: Dict[int, pd.DataFrame],
                        config: Dict[str, object], *, plot: bool, what: str) -> Optional[pd.DataFrame]:
    """write_one_report on the twins of sub_man that have results (returns the long table), or, when
    none has, NO_RESULTS.txt in place of a report (returns None)."""
    sub_frames = arm_frames(sub_man, frames) if len(sub_man) else []
    if sub_frames:
        return write_one_report(out_dir, sub_man, sub_frames, config, plot=plot)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    _write_text_atomic(out_dir / "NO_RESULTS.txt", "No %s has results; no report.\n" % what)
    _say("merge: no %s with results; %s/NO_RESULTS.txt" % (what, out_dir))
    return None


def _num(r: pd.Series, col: str) -> float:
    """A results row's float column; NaN when absent or not a number."""
    try:
        v = float(r[col]) if col in r.index else float("nan")
    except (TypeError, ValueError):
        return float("nan")
    return v


def loss_verdict(loss_at_truth: float, loss_at_estimate: float, rel: float = LOSS_VERDICT_REL) -> str:
    """'optimiser' when the loss at the truth is lower than at the estimate by more than rel of the
    latter (the optimiser missed a better point), 'data' when the estimate's is lower than the truth's
    by more than rel of the truth's (the data prefer another point), 'tie' otherwise, 'na' when either is
    not finite or is at or above the training loss's penalty (Block 17; amendment 3: a loss of
    run_ih_fit._LOSS_CRASH_PENALTY = 1e6, the value the loss builders return when a simulation fails,
    is not a loss)."""
    import run_ih_fit as R
    lt, le = float(loss_at_truth), float(loss_at_estimate)
    if not (np.isfinite(lt) and np.isfinite(le)):
        return "na"
    if lt >= R._LOSS_CRASH_PENALTY or le >= R._LOSS_CRASH_PENALTY:
        return "na"
    if lt < (1.0 - rel) * le:
        return "optimiser"
    if le < (1.0 - rel) * lt:
        return "data"
    return "tie"


def recovery_extras(man: pd.DataFrame, frames: Dict[int, pd.DataFrame], config: Dict[str, object]) -> pd.DataFrame:
    """recovery_extras.csv (Block 17, SPEC 2.8 (f)): one row per twin with results, in the manifest's
    order. tau_h_pk_true_ms = the truth's kappa_tau x tau_h_base_pk (run_config.json's base_ms), NaN for a
    twin without I_h; tau_h_pk_est_ms, loss_at_estimate, v_deepest_train_mV, gh_deep_est_S_cm2 are the
    fit's columns, loss_at_truth its loss_at_eval_theta; gh_deep_true_S_cm2 = the truth's gbar x
    human_ih_params.minf(v*) with the truth's dv_h (run_ih_fit.gh_at_voltage_S_cm2 on the run's I_h spec),
    NaN without I_h; ln_gh_deep_ratio = ln(est / true) when both are positive and finite."""
    import run_ih_fit as R
    import twin_build as TB
    import twin_manifest as TM
    ih_spec = R.build_ih_spec(fit_args_of(config["options"]))
    base = float(config["tau_h_pk"]["base_ms"])
    rows = []
    for _, m in man.iterrows():
        tid = int(m["specimen_id"])
        if tid not in frames:
            continue
        r = frames[tid].iloc[0]
        conf = str(m["configuration"])
        theta = TM.twin_theta(m)
        has_ih = bool(TB.HAS_IH[conf])
        v_star = _num(r, "v_deepest_train_mV")
        tau_true = float(theta["kappa_tau"]) * base if has_ih else float("nan")
        gh_true = (R.gh_at_voltage_S_cm2(float(theta["gbar"]), v_star, ih_spec, dv_h_mV=float(theta["dv_h"]))
                   if has_ih else float("nan"))
        gh_est = _num(r, "gh_at_v_deepest_S_cm2")
        ok = np.isfinite(gh_true) and np.isfinite(gh_est) and gh_true > 0.0 and gh_est > 0.0
        lt, le = _num(r, "loss_at_eval_theta"), _num(r, "loss_at_estimate")
        rows.append(dict(
            twin_id=tid, configuration=conf, twin_of_specimen_id=int(m["twin_of_specimen_id"]),
            paired_with=int(m["paired_with"]), with_noise=bool(m["with_noise"]),
            cell_tau_s_above_window=bool(m["cell_tau_s_above_window"]),
            tau_h_pk_true_ms=tau_true, tau_h_pk_est_ms=_num(r, "tau_h_pk_ms"),
            loss_at_truth=lt, loss_at_estimate=le, loss_verdict=loss_verdict(lt, le),
            v_deepest_train_mV=v_star, gh_deep_true_S_cm2=float(gh_true), gh_deep_est_S_cm2=gh_est,
            ln_gh_deep_ratio=float(np.log(gh_est / gh_true)) if ok else float("nan")))
    return pd.DataFrame(rows, columns=list(EXTRAS_COLUMNS))


# ---------------------------------------------------------------------------
#  main
# ---------------------------------------------------------------------------
COMMANDS = {"plan": cmd_plan, "list": cmd_list, "twin": cmd_twin, "merge": cmd_merge}


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as exc:      # a usage error (2) or --help (0): returned, not raised, so that
        return int(exc.code or 0)    # main() gives the exit code in process too
    try:
        return int(COMMANDS[args.command](args))
    except Refused as exc:
        _err(str(exc))
        return exc.code


if __name__ == "__main__":
    sys.exit(main())
