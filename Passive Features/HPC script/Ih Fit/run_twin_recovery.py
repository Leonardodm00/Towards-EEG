# -*- coding: utf-8 -*-
"""
run_twin_recovery.py -- the Stage-7 run on the twins: plan, one twin (build and fit), merge (Blocks 8, 17, 22, 28)
=================================================================================================================

specs/SPEC.md, the entries of Block 8 (D-184, D-185, D-188, D-182.3) and of
Block 17 (the second run, SPEC 2.8: D-204, D-206 to D-209) and of Block 22
(the third run, SPEC 2.9: D-234). Entry point:
wiring only -- the plan, the build, the fit and the report are calls of
Blocks 5 (twin_table), 13/16 (twin_manifest), 12/18 (twin_build), of
run_ih_fit.main (Block 15) and of ih_recovery_report. Sub-commands:

    python run_twin_recovery.py plan  --group-dir <dir> --noise-dir <dir> --out <dir> [options]
    python run_twin_recovery.py list  --out <dir>
    python run_twin_recovery.py twin  --out <dir> --twin <id> [--force] [--n-workers N]
    python run_twin_recovery.py merge --out <dir> [--no-plot]
    python run_twin_recovery.py sensitivity --run-dir <dir> --twin <id> [--n-workers N] [--n-noise 200]
                                [--h 1e-3] [--check-h 1e-2]

The run directory <out> (SPEC, Block 8, "Outputs"):
    real_cells_<GROUP>.csv, twin_sweeps_<GROUP>.csv, manifest.csv, run_config.json   (plan)
    twins/twin_<id>/specimen_<id>/, fits/twin_<id>/, status/twin_<id>.json            (twin)
    report/, report_without_flagged/, report_no_noise/, recovery_by_configuration.csv,
    recovery_extras.csv, twins_status.csv                                             (merge)
    sensitivity/twin_<id>/                                                            (sensitivity)
    identifiability.csv, truth_trace_check.csv                                        (merge, Block 28)

Block 17: four configurations (the fourth, full_no_noise, built without noise
and reported apart in report_no_noise/); the box and the truth of the I_h time
constant given as tau_h_pk in ms (--tau-h-pk-bounds-ms, --tau-h-pk-truth-ms,
converted to kappa_tau with run_ih_fit.tau_h_base_peak_ms); dv_h's truth 0;
the fit of a twin with I_h fitted with I_h evaluated also at the truth
(--eval-theta); the gate on ih5 with the required axes Cm, gbar, kappa_tau.

Block 22 (SPEC 2.9): the run's I_h arm is an option of plan (--ih-arm, ih6_d207
by default, or ih5); with ih6_d207 the truth of dv_h is drawn in
--dvh-truth-range-mV, with ih5 it is 0; the gbar box is 1e-5,1e-3 and the
floor of the false-positive control 1e-5; the local refinement (--refine,
--refine-starts, --refine-maxfev) is a fit option recorded in the plan; the
worker processes of one fit (`twin --n-workers`) are not a plan option and
never invalidate a plan or a finished fit; the gate's arm and required axes
follow the plan's I_h arm (2.9 (f)); recovery_extras.csv gains the
refinement's columns and the truth and estimate of dv_h.

Block 28 (SPEC 2.10 (c), (d); D-245, D-246, D-247): --refine accepts trf, the
plan's default, and --refine-max-nfev (30) is a fit option recorded in the plan;
recovery_extras.csv gains the fit's L_2 columns, sqrt_l2_at_truth and
on_bound_<name>; the sub-command `sensitivity` runs run_ih_fit's sensitivity
mode at the twin's truth and adds the linearised scatter of the estimate under
the twin's own noise model (bundle_noise: the per-sweep laws and tilts of the
twins' table, Block 10's draws averaged into the loader's bundles) for the
weightings W0 and W1; merge gathers the sensitivity tables.

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
#: (Block 22: the run's I_h arm and the range of dv_h's truth, SPEC 2.9 (d))
TRUTH_OPTIONS = ("ra_mode", "cm_phys_lo", "cm_phys_hi", "tau_lo_ms", "tau_hi_ms", "ra_phys_lo", "ra_phys_hi",
                 "gbar_range", "tau_h_pk_truth_ms", "gbar_floor", "ih_arm", "dvh_truth_range_mV")
#: the fit options: passed to run_ih_fit.main as run_ih_recovery.phase_fit passes them
#: (Block 17: --kappa-bounds replaced by --tau-h-pk-bounds-ms)
FIT_OPTIONS = ("F", "fit_target", "ss_time_weight", "ss_tau_w_ms", "ss_window_ms", "n_calls", "n_initial",
               "ih_mechanism", "ih_distribution", "ih_regions", "vshift_base", "ehcn", "dt_brief_ms", "dt_long_ms",
               "phase3_subset", "gbar_bounds", "dvh_bounds", "tau_h_pk_bounds_ms",
               "refine", "refine_starts", "refine_maxfev", "refine_max_nfev")
#: the gate's tolerances and switches (run_ih_recovery.phase_report)
TOL_OPTIONS = ("tol_cm_factor", "tol_gbar_factor", "tol_dvh_mV", "tol_kappa_factor", "tol_fp_low_rail_frac",
               "tol_fp_cm_factor", "lenient_rails", "drop_railed")
#: the run's I_h arm (Block 22, SPEC 2.9 (d)): the arm of every configuration but passive_fit
#: (twin_manifest.arm_of); its default
IH_ARM_CHOICES = ("ih5", "ih6_d207")
IH_ARM_DEFAULT = "ih6_d207"
#: the gate's required axes per I_h arm (SPEC 2.8 (g), 2.9 (f)): dv_h is not an axis of ih5, and its
#: absence does not fail that gate; ih6_d207 is gated on dv_h too (D-011's 3 mV)
_GATE_REQUIRED_AXES_OF = {"ih5": ("Cm", "gbar", "kappa_tau"), "ih6_d207": ("Cm", "gbar", "dv_h", "kappa_tau")}
#: the defaults of SPEC 2.8 (a), (e) (Block 17) and 2.9 (a), (b), (d) (Block 22): the gbar box and the
#: floor of the false-positive control (S/cm^2), tau_h_pk's box and the range of its truth (ms), the
#: number of noise-free twins, the range of dv_h's truth under ih6_d207 (mV), the refinement
GBAR_BOUNDS_DEFAULT = "1e-5,1e-3"
GBAR_FLOOR_DEFAULT = 1e-5
TAU_H_PK_BOUNDS_MS_DEFAULT = "50,1000"
TAU_H_PK_TRUTH_MS_DEFAULT = "150,700"
N_NO_NOISE_DEFAULT = 5
DVH_TRUTH_RANGE_MV_DEFAULT = "-5,5"
#: the truth of dv_h (mV) under ih5 (SPEC 2.8 (e))
DVH_TRUTH_MV_IH5 = 0.0
#: Block 28 (SPEC 2.10 (c), D-245): trf is the plan's default; nelder_mead a labelled comparison
REFINE_CHOICES = ("none", "nelder_mead", "trf")
REFINE_DEFAULT = "trf"
REFINE_STARTS_DEFAULT = 3
REFINE_MAXFEV_DEFAULT = 100
#: trf's max_nfev per start (run_ih_fit --refine-max-nfev, TrfConfig.max_nfev; Block 28)
REFINE_MAX_NFEV_DEFAULT = 30
#: the sensitivity sub-command (Block 28): noise realisations K, the central-difference step and the
#: step of fd_step_check (unit coordinates; run_ih_fit --sensitivity-h, --sensitivity-check-h), and the
#: root of the realisations' seeds: realisation k uses base_seed = NOISE_BASE_SEED_ROOT + k (D-247)
N_NOISE_DEFAULT = 200
SENS_H_DEFAULT = 1e-3
SENS_CHECK_H_DEFAULT = 1e-2
NOISE_BASE_SEED_ROOT = 1000000
#: the directory of the sensitivity files under the run directory: <run>/SENS_DIR/twin_<id>/ (amendment 2, D-249)
SENS_DIR = "sensitivity"
#: loss_verdict of recovery_extras.csv: one loss lower than the other by more than this fraction of
#: the other (Block 17)
LOSS_VERDICT_REL = 0.02

EXIT_OK, EXIT_FAILED, EXIT_USAGE, EXIT_REFUSED = 0, 1, 2, 3
REPORT_DIR, REPORT_UNFLAGGED_DIR, REPORT_NO_NOISE_DIR = "report", "report_without_flagged", "report_no_noise"
#: the columns of recovery_extras.csv (Block 17), in order
EXTRAS_COLUMNS = ("twin_id", "configuration", "twin_of_specimen_id", "paired_with", "with_noise",
                  "cell_tau_s_above_window", "tau_h_pk_true_ms", "tau_h_pk_est_ms", "loss_at_truth",
                  "loss_at_estimate", "loss_verdict", "v_deepest_train_mV", "gh_deep_true_S_cm2",
                  "gh_deep_est_S_cm2", "ln_gh_deep_ratio",
                  # Block 22 (SPEC 2.9 (e)): the refinement, from the fit's columns; dv_h for a twin fitted with it
                  "loss_after_search", "refine_improved", "refine_n_evals", "refine_best_start",
                  "dvh_true_mV", "dvh_est_mV",
                  # Block 28 (SPEC 2.10 (c)): the fit's L_2 record; then on_bound_<name> (extras_columns)
                  "l2_after_search", "l2_after_refine", "refine_n_converged", "refine_n_failed",
                  "sqrt_l2_at_estimate", "sqrt_l2_at_truth")
#: the columns of ih_recovery_report.recovery_summary (its own empty frame)
SUMMARY_COLUMNS = ("arm", "parameter", "n", "n_railed", "median_abs_err", "q25_abs_err", "q75_abs_err",
                   "median_err", "err_kind")
STATUS_FIELDS = ("twin_id", "twin_of_specimen_id", "configuration", "arm", "status", "reason", "build_seconds",
                 "fit_seconds", "built", "fitted", "code_commit", "code_dirty", "host", "job_id", "finished_utc")


def gate_arm(ih_arm: str) -> str:
    """The arm the gate applies to: the run's I_h arm (SPEC 2.9 (f)); ValueError for another name."""
    if str(ih_arm) not in _GATE_REQUIRED_AXES_OF:
        raise ValueError("unknown I_h arm %r; choose from %s" % (ih_arm, IH_ARM_CHOICES))
    return str(ih_arm)


def gate_required_axes(ih_arm: str) -> tuple:
    """The gate's required axes for the run's I_h arm (SPEC 2.8 (g), 2.9 (f)): ih5 -> (Cm, gbar,
    kappa_tau); ih6_d207 -> (Cm, gbar, dv_h, kappa_tau); ValueError for another name."""
    return tuple(_GATE_REQUIRED_AXES_OF[gate_arm(ih_arm)])


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


def _positive_int(text: str) -> int:
    """argparse type of `twin --n-workers` (Block 22): an integer >= 1."""
    try:
        v = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError("%r is not an integer" % text)
    if v < 1:
        raise argparse.ArgumentTypeError("must be an integer >= 1, got %d" % v)
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
    for name in ("cm_phys_lo", "cm_phys_hi", "tau_lo_ms", "tau_hi_ms", "ra_phys_lo", "ra_phys_hi"):
        p.add_argument("--" + name.replace("_", "-"), type=float, default=d[name])
    p.add_argument("--gbar-floor", type=float, default=GBAR_FLOOR_DEFAULT,
                   help="S/cm^2: the gbar truth of a twin without I_h (active_fit), the gbar box's lower bound "
                        "(SPEC 2.9 (d)); must lie inside --gbar-bounds")
    p.add_argument("--ih-arm", default=IH_ARM_DEFAULT, choices=list(IH_ARM_CHOICES),
                   help="the run's I_h arm (SPEC 2.9 (d)): the arm of full, active_fit, full_no_noise and the "
                        "gate's arm")
    p.add_argument("--dvh-truth-range-mV", default=DVH_TRUTH_RANGE_MV_DEFAULT,
                   help="'LO,HI' (mV): with --ih-arm ih6_d207 the truth of dv_h of a twin with I_h is drawn "
                        "uniformly in it (draw_manifest's ih_dvh_range_mV); must lie inside --dvh-bounds; "
                        "not used with ih5 (dv_h's truth 0)")
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
    p.add_argument("--refine", default=REFINE_DEFAULT, choices=list(REFINE_CHOICES),
                   help="the local refinement of the fit (SPEC 2.9 (b)), passed to run_ih_fit")
    p.add_argument("--refine-starts", type=int, default=REFINE_STARTS_DEFAULT)
    p.add_argument("--refine-maxfev", type=int, default=REFINE_MAXFEV_DEFAULT,
                   help="nelder_mead only: maxfev of one start")
    p.add_argument("--refine-max-nfev", type=int, default=REFINE_MAX_NFEV_DEFAULT,
                   help="trf only: least_squares' max_nfev of one start (SPEC 2.10 (c)), passed to run_ih_fit")
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
    p.add_argument("--n-workers", type=_positive_int, default=1,
                   help="worker processes of the fit's loss (SPEC 2.9 (c)), passed to run_ih_fit as --n-workers; "
                        "not a plan option: changing it never invalidates the plan or a finished fit")

    p = sub.add_parser("merge", help="the recovery report (ih_recovery_report)")
    p.add_argument("--out", required=True)
    p.add_argument("--no-plot", action="store_true")

    p = sub.add_parser("sensitivity", help="Block 28: run_ih_fit's sensitivity mode at the twin's truth and "
                                           "the predicted scatter of the estimate from the twin's noise model")
    p.add_argument("--run-dir", required=True, help="the run directory (written by plan)")
    p.add_argument("--twin", required=True, type=int)
    p.add_argument("--n-workers", type=_positive_int, default=1,
                   help="worker processes of the trace evaluator, passed to run_ih_fit as --n-workers")
    p.add_argument("--n-noise", type=_positive_int, default=N_NOISE_DEFAULT,
                   help="K, the noise realisations of the predicted scatter (>= 2)")
    p.add_argument("--h", type=float, default=SENS_H_DEFAULT,
                   help="central-difference step, unit coordinates (run_ih_fit --sensitivity-h)")
    p.add_argument("--check-h", type=float, default=SENS_CHECK_H_DEFAULT,
                   help="the step of fd_step_check (run_ih_fit --sensitivity-check-h); 0 = no check")
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
    argv = fit_argv(o, "-", "-", "ih5")
    try:
        return R._parse_args(argv)
    except SystemExit as exc:      # argparse's refusal (its message on stderr): a usage refusal, not an exit
        raise Refused(EXIT_USAGE, "run_ih_fit's parser refused the fit options (exit %r): %s"
                      % (exc.code, " ".join(argv)))


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


def _pair(text, flag: str) -> tuple:
    """'LO,HI' -> (LO, HI) as floats (run_ih_fit.parse_window); Refused(EXIT_USAGE) naming the option."""
    import run_ih_fit as R
    try:
        lo, hi = (float(x) for x in R.parse_window(text))
    except (ValueError, TypeError) as exc:
        raise Refused(EXIT_USAGE, "plan: %s: %s" % (flag, exc))
    if not (np.isfinite(lo) and np.isfinite(hi)):
        raise Refused(EXIT_USAGE, "plan: %s must be finite, got %r" % (flag, text))
    return lo, hi


def dvh_truth_of(o) -> tuple:
    """(ih_dvh_range_mV, dvh_truth_mV) of the plan's I_h arm (Block 22, SPEC 2.9 (d)): ih6_d207 ->
    (the --dvh-truth-range-mV pair, None: the drawer's uniform draw kept); ih5 -> (None, 0.0)."""
    if str(o["ih_arm"]) == "ih6_d207":
        return _pair(o["dvh_truth_range_mV"], "--dvh-truth-range-mV"), None
    if str(o["ih_arm"]) == "ih5":
        return None, DVH_TRUTH_MV_IH5
    raise Refused(EXIT_USAGE, "plan: unknown --ih-arm %r" % (o["ih_arm"],))


def check_boxes(o) -> None:
    """The refusals of Block 22 (exit 3): the floor of gbar outside the gbar box, or (ih6_d207) the
    range of dv_h's truth outside the dv_h box; closed intervals (the default floor is the box's lower
    bound)."""
    g_lo, g_hi = _pair(o["gbar_bounds"], "--gbar-bounds")
    floor = float(o["gbar_floor"])
    if not (np.isfinite(floor) and g_lo <= floor <= g_hi):
        raise Refused(EXIT_REFUSED, "plan: --gbar-floor %r lies outside the gbar box --gbar-bounds [%r, %r] "
                                    "(SPEC 2.9 (d))" % (floor, g_lo, g_hi))
    rng, _ = dvh_truth_of(o)
    if rng is not None:
        d_lo, d_hi = _pair(o["dvh_bounds"], "--dvh-bounds")
        if not (d_lo <= rng[0] and rng[1] <= d_hi):
            raise Refused(EXIT_REFUSED, "plan: --dvh-truth-range-mV [%r, %r] lies outside the dv_h box "
                                        "--dvh-bounds [%r, %r] (SPEC 2.9 (d))" % (rng[0], rng[1], d_lo, d_hi))


def truth_kwargs_of(o, tau_rec: Dict[str, object]) -> Dict[str, object]:
    """synth_gt_grid.draw_manifest's keyword arguments, built as
    run_ih_recovery.phase_manifest builds them, without the noise arguments
    (and without e_pas, cells_per_cohort and max_cells, which are not options
    of plan: the twins' rest is the real cell's, D-192; cells_per_cohort keeps
    draw_manifest's default 10, the value phase_manifest passes). Block 17:
    the kappa range is tau_h_pk's truth range as kappa. Block 22: the dv_h
    range is --dvh-truth-range-mV under ih6_d207 and None (dv_h 0) under ih5
    (dvh_truth_of)."""
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
        ih_dvh_range_mV=dvh_truth_of(o)[0],
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
    check_boxes(opts)
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

    dvh_truth_mV = dvh_truth_of(opts)[1]
    try:
        manifest = TM.draw_twin_manifest(real_cells[list(TM.REAL_CELL_COLUMNS)], n_per_configuration=n,
                                         n_no_noise=int(opts["n_no_noise"]), seed=int(opts["seed"]),
                                         id_base=int(opts["id_base"]), ih_arm=str(opts["ih_arm"]),
                                         dvh_truth_mV=dvh_truth_mV, truth_kwargs=truth_kwargs_of(opts, tau_rec))
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
        "ih_arm": str(opts["ih_arm"]), "dvh_truth_mV": dvh_truth_mV,
        "gate_arm": gate_arm(opts["ih_arm"]), "gate_required_axes": list(gate_required_axes(opts["ih_arm"])),
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
N_WORKERS_FLAG = "--n-workers"


def fit_argv(options: Dict[str, object], archive_dir, output_dir, arm: str, code_dir=HERE,
             eval_theta: Optional[str] = None, n_workers: Optional[int] = None) -> List[str]:
    """The argv of run_ih_fit.main for one twin, in the form
    run_ih_recovery.phase_fit builds it (the '=' form for values that can be
    negative); no --fit-params (the arms' own names). Block 17: no
    --kappa-bounds; --tau-h-pk-bounds-ms (the box of kappa_tau given in ms)
    and, when given, --eval-theta. Block 22: --refine, --refine-starts,
    --refine-maxfev (the plan's fit options) and, when given, --n-workers N
    (the twin's, not the plan's), last. Block 28: --refine-max-nfev after
    --refine-maxfev (absent from a plan made before Block 28, then not
    passed)."""
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
    argv += ["--refine", str(a.refine), "--refine-starts", str(int(a.refine_starts)),
             "--refine-maxfev", str(int(a.refine_maxfev))]
    if getattr(a, "refine_max_nfev", None) is not None:
        argv += ["--refine-max-nfev", str(int(a.refine_max_nfev))]
    if eval_theta is not None:
        argv += ["--eval-theta=%s" % eval_theta]
    if n_workers is not None:
        argv += [N_WORKERS_FLAG, str(int(n_workers))]
    return argv


def without_n_workers(argv: Sequence[str]) -> List[str]:
    """argv without its --n-workers pair (Block 22: the comparison of a finished fit's options with
    the run's ignores the workers, which do not change the fit's result, SPEC 2.9 (c))."""
    out, skip = [], False
    for a in argv:
        if skip:
            skip = False
            continue
        if a == N_WORKERS_FLAG:
            skip = True
            continue
        if a.startswith(N_WORKERS_FLAG + "="):
            continue
        out.append(a)
    return out


def eval_theta_of(row, arm: str) -> Optional[str]:
    """--eval-theta for one twin (Block 17): the truth in the fitted spec's
    names, for a twin whose configuration has I_h and is fitted with I_h
    (full, full_no_noise) and whose truth holds every fitted name; None
    otherwise (active_fit, passive_fit). Values by repr: they read back as
    the same float64. The names are the arm's (run_ih_fit.ARMS): five for
    ih5, the six with dv_h (the manifest's truth) for ih6_d207 (Block 22)."""
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


def build_if_needed(run: Dict[str, object], row, tcfg: TwinConfig, argv: Sequence[str], *, force: bool,
                    status: Dict[str, object]) -> Optional[str]:
    """The build of one twin as `twin` makes it (Block 12; factored out of cmd_twin by Block 28 so that
    `sensitivity` builds the twin the same way): an archive already complete and matching the manifest is
    kept (not with force), else the twin is built from the fit's own model of argv. Sets status["built"]
    and status["build_seconds"]; returns None, or the reason of a build_failed."""
    import run_ih_fit as R
    import twin_build as TB
    import twin_manifest as TM
    import twin_table as TT
    out, opts = run["out"], run["options"]
    tid, conf = int(row["specimen_id"]), str(row["configuration"])
    real_id = int(row["twin_of_specimen_id"])
    with_noise = bool(row["with_noise"])
    group_dir = out / "twins" / ("twin_%d" % tid)
    spec_dir = group_dir / ("specimen_%d" % tid)
    t0 = time.time()
    try:
        spec = TM.twin_spec_of(row, F=float(opts["F"]))
        complete = (spec_dir / "metadata.json").is_file()
        if complete and not force:
            why = _archive_matches(spec_dir, spec, tcfg, with_noise)
            if why is not None:
                return "the archive %s exists but %s; run with --force to rebuild" % (spec_dir, why)
            _say("twin %d: archive complete (%s); not rebuilt" % (tid, spec_dir))
        else:
            fit_args = R._parse_args(list(argv))
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
        return "%s: %s" % (type(exc).__name__, exc)
    status["build_seconds"] = time.time() - t0
    return None


def cmd_twin(args) -> int:
    import run_ih_fit as R

    run = load_run(args.out)
    out, opts, man = run["out"], run["options"], run["manifest"]
    tcfg = recorded_twin_config(run["config"])
    rows = man[man["specimen_id"] == int(args.twin)]
    if len(rows) != 1:
        raise Refused(EXIT_USAGE, "twin %d is not in the manifest %s" % (int(args.twin), run["paths"]["manifest"]))
    row = rows.iloc[0]
    tid, conf, arm = int(row["specimen_id"]), str(row["configuration"]), str(row["arm"])
    real_id = int(row["twin_of_specimen_id"])
    group_dir = out / "twins" / ("twin_%d" % tid)
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
    argv = fit_argv(opts, group_dir, fit_dir, arm, eval_theta=ev, n_workers=int(args.n_workers))

    # ---- the build (Block 12) ---------------------------------------------
    why = build_if_needed(run, row, tcfg, argv, force=bool(args.force), status=status)
    if why is not None:
        return finish("build_failed", why)

    # ---- the fit (run_ih_fit.main, the arm of the configuration) ----------
    t1 = time.time()
    need_fit = bool(args.force or status["built"] or not csv.exists())
    if not need_fit:
        prev = _read_json(argv_record)
        if prev is not None and (not isinstance(prev.get("argv"), list)
                                 or without_n_workers(prev["argv"]) != without_n_workers(argv)):
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


def run_gate(config: Dict[str, object]) -> tuple:
    """(gate arm, required axes) of a planned run (Block 22): those recorded in run_config.json,
    which must equal gate_arm / gate_required_axes of the recorded I_h arm (a run planned by older code
    has no ih_arm: Refused(EXIT_USAGE))."""
    ih_arm = (config.get("options") or {}).get("ih_arm")
    if ih_arm not in IH_ARM_CHOICES:
        raise Refused(EXIT_USAGE, "run_config.json records no I_h arm of %s (got %r): a run planned before "
                                  "Block 22 cannot be merged by this code" % (IH_ARM_CHOICES, ih_arm))
    arm, axes = gate_arm(ih_arm), gate_required_axes(ih_arm)
    if config.get("gate_arm") != arm or list(config.get("gate_required_axes") or []) != list(axes):
        raise Refused(EXIT_USAGE, "run_config.json's gate (%r, %r) is not the gate of its I_h arm %s (%r, %r)"
                      % (config.get("gate_arm"), config.get("gate_required_axes"), ih_arm, arm, list(axes)))
    return arm, axes


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
    """The report of run_ih_recovery.phase_report on these frames, the gate on the run's gate arm with
    its required axes (Block 17; Block 22: run_config.json's gate_arm and gate_required_axes, the
    functions of the plan's I_h arm, SPEC 2.9 (f)): returns the long table."""
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
    arm, axes = run_gate(config)
    verdict, passed = RR.evaluate_gate(summary, fp, arm=arm, tol=tol, required_axes=axes)
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
    arm, axes = run_gate(run["config"])          # refused before anything is written (Block 22)
    _say("merge: the gate on %s, required axes %s" % (arm, ", ".join(axes)))

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

    # Block 28: the sensitivity tables, when any twin has them (written whether or not a fit has results)
    n_sens = gather_sensitivity(out)
    if any(n_sens.values()):
        _say("merge: sensitivity tables of %d twin(s) -> %s, %d twin(s) -> %s"
             % (n_sens["identifiability"], out / "identifiability.csv",
                n_sens["truth_trace_check"], out / "truth_trace_check.csv"))

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
        if any(n_sens.values()):
            _say("merge: the sensitivity tables were written (Block 28); exit 0")
            return EXIT_OK
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


def _bool_or_nan(r: pd.Series, col: str):
    """A results row's bool column (True/False, or the strings a CSV gives back); NaN when absent or
    neither."""
    v = r[col] if col in r.index else None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, str) and v.strip() in ("True", "False"):
        return v.strip() == "True"
    return float("nan")


def _int_or_nan(r: pd.Series, col: str):
    """A results row's integer column as int; NaN when absent, not a number or not integral."""
    v = _num(r, col)
    return int(v) if np.isfinite(v) and float(v) == int(v) else float("nan")


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
    NaN without I_h; ln_gh_deep_ratio = ln(est / true) when both are positive and finite.
    Block 22 (SPEC 2.9 (e)): loss_after_search, refine_improved, refine_n_evals, refine_best_start are
    the fit's columns (NaN when the fit's row has none); for a twin fitted with dv_h (dv_h among the
    fit's param_names) dvh_true_mV is the manifest's ih_dvh_mV (0 for active_fit, SPEC 2.9 (d)) and
    dvh_est_mV the fit's dv_h, NaN otherwise. g_h(v*) at the truth uses the truth's dv_h and at the
    estimate the estimate's (the fit's gh_at_v_deepest_S_cm2, run_ih_fit._fit_record).
    Block 28 (SPEC 2.10 (c)): l2_after_search, l2_after_refine, refine_n_converged, refine_n_failed,
    sqrt_l2_at_estimate are the fit's columns, sqrt_l2_at_truth its sqrt_l2_at_eval_theta (NaN when the
    fit's row has none); on_bound_<name> for every name of the run's I_h arm (extras_columns): the fit's
    flag ("lo", "hi" or ""), "" for a name the twin's arm does not fit."""
    import run_ih_fit as R
    import twin_build as TB
    import twin_manifest as TM
    ih_spec = R.build_ih_spec(fit_args_of(config["options"]))
    base = float(config["tau_h_pk"]["base_ms"])
    bound_names = bound_flag_names(config)
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
        fitted = str(r["param_names"]).split("|") if "param_names" in r.index else []
        with_dvh = "dv_h" in fitted
        rows.append(dict(
            twin_id=tid, configuration=conf, twin_of_specimen_id=int(m["twin_of_specimen_id"]),
            paired_with=int(m["paired_with"]), with_noise=bool(m["with_noise"]),
            cell_tau_s_above_window=bool(m["cell_tau_s_above_window"]),
            tau_h_pk_true_ms=tau_true, tau_h_pk_est_ms=_num(r, "tau_h_pk_ms"),
            loss_at_truth=lt, loss_at_estimate=le, loss_verdict=loss_verdict(lt, le),
            v_deepest_train_mV=v_star, gh_deep_true_S_cm2=float(gh_true), gh_deep_est_S_cm2=gh_est,
            ln_gh_deep_ratio=float(np.log(gh_est / gh_true)) if ok else float("nan"),
            loss_after_search=_num(r, "loss_after_search"), refine_improved=_bool_or_nan(r, "refine_improved"),
            refine_n_evals=_int_or_nan(r, "refine_n_evals"), refine_best_start=_int_or_nan(r, "refine_best_start"),
            dvh_true_mV=float(m["ih_dvh_mV"]) if with_dvh else float("nan"),
            dvh_est_mV=_num(r, "dv_h") if with_dvh else float("nan"),
            l2_after_search=_num(r, "l2_after_search"), l2_after_refine=_num(r, "l2_after_refine"),
            refine_n_converged=_int_or_nan(r, "refine_n_converged"),
            refine_n_failed=_int_or_nan(r, "refine_n_failed"),
            sqrt_l2_at_estimate=_num(r, "sqrt_l2_at_estimate"),
            sqrt_l2_at_truth=_num(r, "sqrt_l2_at_eval_theta"),
            **{"on_bound_" + n: (_flag(r, "on_bound_" + n) if n in fitted else "") for n in bound_names}))
    return pd.DataFrame(rows, columns=extras_columns(config))


def bound_flag_names(config: Dict[str, object]) -> List[str]:
    """The names of recovery_extras.csv's on_bound_<name> columns (Block 28): the fitted names of the
    run's I_h arm (run_ih_fit.ARMS[ih_arm].fit_params, which hold the passive arm's three)."""
    import run_ih_fit as R
    ih_arm = (config.get("options") or {}).get("ih_arm")
    if ih_arm not in IH_ARM_CHOICES:
        raise Refused(EXIT_USAGE, "run_config.json records no I_h arm of %s (got %r)" % (IH_ARM_CHOICES, ih_arm))
    return [str(n) for n in R.ARMS[str(ih_arm)].fit_params]


def extras_columns(config: Dict[str, object]) -> List[str]:
    """The columns of recovery_extras.csv: EXTRAS_COLUMNS, then on_bound_<name> (Block 28)."""
    return list(EXTRAS_COLUMNS) + ["on_bound_" + n for n in bound_flag_names(config)]


def _flag(r: pd.Series, col: str) -> str:
    """A results row's bound flag: 'lo', 'hi' or '' (an empty CSV cell reads back as NaN)."""
    v = r[col] if col in r.index else ""
    if isinstance(v, str):
        return v.strip()
    return ""


# ---------------------------------------------------------------------------
#  sensitivity (Block 28, SPEC 2.10 (d))
# ---------------------------------------------------------------------------
class BundleGridError(ValueError):
    """The loader's bundle grid is not the stored sweep's or window's grid (SPEC Block 28: the mapping of
    the twin's noise onto the bundle would then be a scientific choice the entry does not make)."""


def sens_dir_of(out: Path, twin_id: int) -> Path:
    """<run>/sensitivity/twin_<id>/ (SPEC Block 28, amendment 2: no group level; the run directory is
    already the group's, as <run>/fits and <run>/twins)."""
    return Path(out) / SENS_DIR / ("twin_%d" % int(twin_id))


def sensitivity_theta_of(row, arm: str) -> str:
    """--sensitivity-theta of one twin (Block 28): the twin's truth in the arm's fitted names, values by
    repr. The truth is the manifest's: Cm, Rm, Ra and, for an arm with I_h, gbar, dv_h, kappa_tau from
    its I_h truth columns -- for a twin simulated without I_h (active_fit) those of the false-positive
    control (SPEC 2.7 (i), 2.9 (d)). ValueError for a missing or non-finite value."""
    import run_ih_fit as R
    import twin_manifest as TM
    cols = dict(TM.PASSIVE_TRUTH + TM.IH_TRUTH)
    vals = []
    for n in R.ARMS[str(arm)].fit_params:
        if n not in cols or cols[n] not in row.index:
            raise ValueError("twin %s: no truth column for the fitted name %r" % (row["specimen_id"], n))
        v = float(row[cols[n]])
        if not np.isfinite(v):
            raise ValueError("twin %s: non-finite truth for %r" % (row["specimen_id"], n))
        vals.append("%s=%r" % (n, v))
    return ",".join(vals)


def _ss_sweep_rows(protocol, table, specimen_id: int, sweep: int, *, twin_id: int, base_seed: int) -> np.ndarray:
    """Block 10's noise of every stored window of SS sweep `sweep` of the real cell (K_s, n_w), with the
    positions, period and sampling twin_build.simulate_twin passes."""
    import twin_noise as TN
    import twin_table as TT
    sweeps = np.asarray(protocol.ss_sweep_number, dtype=np.int64)
    pos = np.asarray(protocol.ss_position, dtype=np.int64)
    m = sweeps == int(sweep)
    law, tilt = TT.sweep_law(table, int(specimen_id), "SS", int(sweep))
    return TN.ss_sweep_noise(law, tilt, 1.0 / float(protocol.ss_fs_Hz), int(np.asarray(protocol.ss_t_s).size),
                             pos[m], int(protocol.ss_period_samples), base_seed=int(base_seed),
                             twin_id=int(twin_id), sweep_number=int(sweep))


def bundle_noise(bundle, is_brief: bool, protocol, table, specimen_id: int, *, twin_id: int, base_seed: int,
                 ss_cache: Optional[Dict[int, np.ndarray]] = None) -> np.ndarray:
    """The noise (mV, float64, len(bundle.t)) of one loader bundle of a twin, for one realisation (SPEC
    Block 28, 2.10 (d)): the twin's own per-sweep laws and tilts (twin_table.sweep_law on the real cell
    `specimen_id`, also for a noise-free twin), seeds (base_seed, twin_id) of SPEC 2.7 (e).

    long bundle  the mean over bundle.sweep_numbers of twin_noise.ls_sweep_noise(law, tilt, 1 / fs,
                 n_samples) of each sweep (fs, n_samples: the real cell's CellProtocol, as twin_build
                 passes them)
    brief bundle the mean, over every stored SS window of the bundle's polarity (sign of amplitude_pA),
                 of the rows of twin_noise.ss_sweep_noise of the window's sweep (all the sweep's stored
                 positions passed, as twin_build passes them)

    `protocol`: twin_table.CellProtocol of the real cell; `table`: the twins' table; `ss_cache`: an
    optional dict sweep -> rows, shared by the bundles of ONE realisation. BundleGridError when the
    bundle's samples are not the stored sweep's or window's (length, times, the windows counted, the
    sweeps named)."""
    import twin_noise as TN
    import twin_table as TT
    t = np.asarray(bundle.t, dtype=np.float64)
    if bool(is_brief):
        t_w = np.asarray(protocol.ss_t_s, dtype=np.float64)
        if t.shape != t_w.shape or not np.array_equal(t, t_w):
            raise BundleGridError("brief bundle (%+.6g pA): its %d samples are not the stored window's %d "
                                  "samples" % (float(bundle.amplitude_pA), t.size, t_w.size))
        dep = float(bundle.amplitude_pA) > 0.0
        pol = np.asarray(protocol.ss_polarity_is_dep, dtype=bool)
        sweeps = np.asarray(protocol.ss_sweep_number, dtype=np.int64)
        rows = []
        for s in np.unique(sweeps):
            if ss_cache is not None and int(s) in ss_cache:
                sw = ss_cache[int(s)]
            else:
                sw = _ss_sweep_rows(protocol, table, specimen_id, int(s), twin_id=twin_id, base_seed=base_seed)
                if ss_cache is not None:
                    ss_cache[int(s)] = sw
            keep = pol[sweeps == s] == dep
            if keep.any():
                rows.append(sw[keep])
        n_win = int(sum(r.shape[0] for r in rows))
        named = sorted(int(x) for x in (getattr(bundle, "sweep_numbers", None) or []))
        have = sorted(int(s) for s in np.unique(sweeps[pol == dep]))
        if n_win == 0 or n_win != int(bundle.n_repeats_averaged) or (named and named != have):
            raise BundleGridError("brief bundle (%+.6g pA): %d stored window(s) of its polarity in sweeps %s, "
                                  "the bundle averages %d in sweeps %s"
                                  % (float(bundle.amplitude_pA), n_win, have, int(bundle.n_repeats_averaged),
                                     named))
        return np.ascontiguousarray(np.mean(np.concatenate(rows, axis=0), axis=0), dtype=np.float64)
    fs = float(protocol.ls_fs_Hz)
    nums = [int(x) for x in np.asarray(protocol.ls_sweep_number, dtype=np.int64)]
    named = [int(x) for x in (getattr(bundle, "sweep_numbers", None) or [])]
    if not named:
        raise BundleGridError("long bundle (%+.6g pA) names no sweep" % float(bundle.amplitude_pA))
    draws = []
    for s in named:
        if s not in nums:
            raise BundleGridError("long bundle (%+.6g pA): sweep %d is not a stored LS sweep of cell %d"
                                  % (float(bundle.amplitude_pA), s, int(specimen_id)))
        n = int(np.asarray(protocol.ls_n_samples)[nums.index(s)])
        if n != t.size or not np.array_equal(t, np.arange(n) / fs):
            raise BundleGridError("long bundle (%+.6g pA): its %d samples are not the grid of stored sweep %d "
                                  "(%d samples at %r Hz)" % (float(bundle.amplitude_pA), t.size, s, n, fs))
        law, tilt = TT.sweep_law(table, int(specimen_id), "LS", s)
        draws.append(TN.ls_sweep_noise(law, tilt, 1.0 / fs, n, base_seed=int(base_seed), twin_id=int(twin_id),
                                       sweep_number=s))
    return np.ascontiguousarray(np.mean(np.stack(draws, axis=0), axis=0), dtype=np.float64)


def noise_realisations(bundles: Sequence, is_brief: Sequence[bool], index: Sequence[int], protocol, table,
                       specimen_id: int, *, twin_id: int, n_noise: int):
    """Yield, for k = 0 .. n_noise - 1, the list of bundle_noise of the bundles at `index` (in that order)
    under base_seed = NOISE_BASE_SEED_ROOT + k (SPEC Block 28, D-247)."""
    for k in range(int(n_noise)):
        cache: Dict[int, np.ndarray] = {}
        yield [bundle_noise(bundles[i], bool(is_brief[i]), protocol, table, specimen_id, twin_id=twin_id,
                            base_seed=NOISE_BASE_SEED_ROOT + k, ss_cache=cache) for i in index]


def scatter_tables(sens_dir: Path, specimen_id: int, protocol, table, real_id: int, *, twin_id: int,
                   n_noise: int) -> pd.DataFrame:
    """The rows of identifiability_<specimen>.csv with sd_lin_<name> and corr_<a>_<b> added (SPEC Block
    28): for W0 (meta's protocol) and W1 (ls_tau_d_ms = run_ih_fit.SENS_W1_TAU_D_MS), the layout of the
    stored training bundles (run_ih_fit.layout_from_stored), J = identifiability.jacobian_from_sensitivities
    of the stored sensitivities, R_eta (K, N) from identifiability.noise_residuals of each realisation's
    bundle noise, and identifiability.identifiability_row(J, names, R_eta)'s sd_lin_ and corr_ entries."""
    import identifiability as IDF
    import run_ih_fit as R
    import trace_sensitivity as TS
    sid = int(specimen_id)
    npz = sens_dir / ("bundles_%d.npz" % sid)
    meta = R._read_meta(sens_dir / ("sensitivity_%d.json" % sid))
    sens = TS.load_sensitivities(sens_dir / ("sensitivity_%d.npz" % sid))
    z = R._read_bundles_npz(npz)
    bundles = R.stored_bundles(z)
    is_brief = [bool(x) for x in np.asarray(z["is_brief"], dtype=bool)]
    names = [str(n) for n in meta["names"]]
    weightings = (("W0", None), ("W1", R.SENS_W1_TAU_D_MS))
    layouts, J = {}, {}
    tix = None
    for tag, tau in weightings:
        layouts[tag], ti = R.layout_from_stored(z, meta, ls_tau_d_ms=tau)
        if tix is not None and list(ti) != list(tix):
            raise ValueError("the training bundles of W0 and W1 differ: %s, %s" % (tix, ti))
        tix = list(ti)
        J[tag] = IDF.jacobian_from_sensitivities([sens["S"][i] for i in tix], layouts[tag])
    R_eta = {tag: [] for tag, _ in weightings}
    for noise in noise_realisations(bundles, is_brief, tix, protocol, table, real_id, twin_id=twin_id,
                                    n_noise=n_noise):
        for tag, _ in weightings:
            R_eta[tag].append(IDF.noise_residuals(layouts[tag], noise))
    base = pd.read_csv(sens_dir / ("identifiability_%d.csv" % sid), float_precision="round_trip")
    out_rows = []
    for _, r in base.iterrows():
        tag = str(r["weighting"])
        if tag not in J:
            raise ValueError("identifiability_%d.csv: unknown weighting %r" % (sid, tag))
        row = IDF.identifiability_row(J[tag], names, np.asarray(R_eta[tag], dtype=np.float64))
        add = {k: v for k, v in row.items() if k.startswith("sd_lin_") or k.startswith("corr_")}
        d = dict(r)
        d.update(add)
        out_rows.append(d)
    return pd.DataFrame(out_rows)


def cmd_sensitivity(args) -> int:
    """Block 28: build the twin as `twin` does (a built one reused), run run_ih_fit.main with the twin's
    fit argv plus --sensitivity-theta <truth> and --sensitivity-out <run>/sensitivity/twin_<id>,
    then write identifiability.csv there: the sensitivity mode's rows with sd_lin_<name>, corr_<a>_<b>,
    twin_id, configuration, n_noise. Exit 0 done, 1 failed, 2 usage."""
    import passive_fitting_hpc_fixed as mono
    import run_ih_fit as R
    import twin_table as TT
    run = load_run(args.run_dir)
    out, opts, man = run["out"], run["options"], run["manifest"]
    tcfg = recorded_twin_config(run["config"])
    if int(args.n_noise) < 2:
        raise Refused(EXIT_USAGE, "sensitivity: --n-noise must be >= 2 (the scatter's SD), got %d"
                      % int(args.n_noise))
    rows = man[man["specimen_id"] == int(args.twin)]
    if len(rows) != 1:
        raise Refused(EXIT_USAGE, "twin %d is not in the manifest %s" % (int(args.twin), run["paths"]["manifest"]))
    row = rows.iloc[0]
    tid, conf, arm = int(row["specimen_id"]), str(row["configuration"]), str(row["arm"])
    real_id = int(row["twin_of_specimen_id"])
    twin_group_dir = out / "twins" / ("twin_%d" % tid)
    sdir = sens_dir_of(out, tid)
    status = dict(twin_id=tid, built=False, build_seconds=0.0, code_commit=code_commit(),
                  code_dirty=code_dirty())
    t0 = time.time()
    _say("sensitivity: twin %d (%s, arm %s, real cell %d) -> %s" % (tid, conf, arm, real_id, sdir))

    def failed(why: str) -> int:
        _err("SENSITIVITY %d: failed: %s" % (tid, why))
        return EXIT_FAILED

    try:
        ev = eval_theta_of(row, arm)
        theta = sensitivity_theta_of(row, arm)
    except Exception as exc:   # noqa: BLE001
        return failed("the truth of twin %d: %s: %s" % (tid, type(exc).__name__, exc))
    argv = fit_argv(opts, twin_group_dir, sdir, arm, eval_theta=ev, n_workers=int(args.n_workers))
    why = build_if_needed(run, row, tcfg, argv, force=False, status=status)
    if why is not None:
        return failed("build: " + why)
    sargv = argv + ["--sensitivity-theta=%s" % theta, "--sensitivity-out", str(sdir),
                    "--sensitivity-h", repr(float(args.h)), "--sensitivity-check-h", repr(float(args.check_h))]
    sdir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(sdir / "sensitivity_argv.json", dict(argv=sargv, n_noise=int(args.n_noise),
                                                             started_utc=_utc_now(), **status))
    _say("sensitivity: run_ih_fit.main %s" % " ".join(sargv))
    try:
        R.main(sargv)
    except SystemExit as exc:
        if exc.code not in (None, 0):
            return failed("run_ih_fit exited %r" % (exc.code,))
    except Exception as exc:   # noqa: BLE001
        traceback.print_exc()
        return failed("run_ih_fit raised %s: %s" % (type(exc).__name__, exc))
    finally:
        _clear_sections()
    if not (sdir / ("identifiability_%d.csv" % tid)).is_file():
        return failed("run_ih_fit wrote no identifiability_%d.csv in %s" % (tid, sdir))
    try:
        table = TT.read_twin_sweep_table(run["paths"]["table"])
        real_dir = Path(run["config"]["group_dir"]) / ("specimen_%d" % real_id)
        protocol = TT.load_cell_protocol(real_dir, spike_reason_fn=mono.sweep_spike_reason, twin_cfg=tcfg)
        df = scatter_tables(sdir, tid, protocol, table, real_id, twin_id=tid, n_noise=int(args.n_noise))
    except BundleGridError as exc:
        return failed("SPEC Block 28 STOP: the loader's bundle grid is not the stored grid: %s" % exc)
    except Exception as exc:   # noqa: BLE001
        traceback.print_exc()
        return failed("the predicted scatter: %s: %s" % (type(exc).__name__, exc))
    df.insert(0, "n_noise", int(args.n_noise))
    df.insert(0, "configuration", conf)
    df.insert(0, "twin_id", tid)
    _write_csv_atomic(sdir / "identifiability.csv", df)
    sd_cols = [c for c in df.columns if str(c).startswith("sd_lin_")]
    for _, r in df.iterrows():
        _say("sensitivity: twin %d %s: %s" % (tid, r["weighting"], ", ".join("%s %.4g" % (c, float(r[c]))
                                                                            for c in sd_cols)))
    _say("SENSITIVITY %d: done in %.1f s -> %s" % (tid, time.time() - t0, sdir / "identifiability.csv"))
    return EXIT_OK


def gather_sensitivity(out: Path) -> Dict[str, int]:
    """merge's part of Block 28: every <run>/sensitivity/twin_*/identifiability.csv into
    <run>/identifiability.csv and every truth_trace_check_*.csv of those directories into
    <run>/truth_trace_check.csv (a twin_id column first), in increasing twin id; nothing written
    when no file exists. Returns the number of twins gathered per table."""
    root = Path(out) / SENS_DIR
    dirs = []
    if root.is_dir():
        for d in root.glob("twin_*"):
            try:
                dirs.append((int(d.name[len("twin_"):]), d))
            except ValueError:
                continue
    dirs.sort()
    ident, trace = [], []
    for tid, d in dirs:
        f = d / "identifiability.csv"
        if f.is_file():
            ident.append(pd.read_csv(f, float_precision="round_trip"))
        for g in sorted(d.glob("truth_trace_check_*.csv")):
            df = pd.read_csv(g, float_precision="round_trip")
            df.insert(0, "twin_id", tid)
            trace.append(df)
    n = {"identifiability": 0, "truth_trace_check": 0}
    if ident:
        df = pd.concat(ident, ignore_index=True, sort=False)
        _write_csv_atomic(Path(out) / "identifiability.csv", df)
        n["identifiability"] = int(df["twin_id"].nunique())
    if trace:
        df = pd.concat(trace, ignore_index=True, sort=False)
        _write_csv_atomic(Path(out) / "truth_trace_check.csv", df)
        n["truth_trace_check"] = int(df["twin_id"].nunique())
    return n


# ---------------------------------------------------------------------------
#  main
# ---------------------------------------------------------------------------
COMMANDS = {"plan": cmd_plan, "list": cmd_list, "twin": cmd_twin, "merge": cmd_merge,
            "sensitivity": cmd_sensitivity}


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
