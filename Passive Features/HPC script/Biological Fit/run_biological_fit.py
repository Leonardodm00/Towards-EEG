# -*- coding: utf-8 -*-
"""
run_biological_fit.py
=====================

Phase-2 entrypoint for the passive-fit pipeline on REAL (biological) human
cortical-neuron recordings -- one (layer x type) GROUP per invocation
(called by submit_biological_fit.sh with $GROUP -> --archive-dir <ROOT>/<GROUP>).

This is the biological analogue of run_synth_benchmark.py. It runs the SAME
fitting algorithm that produced the synthetic benchmark report, i.e. the
monolith `passive_fitting_hpc_fixed.py` PATCHED with the multi-protocol,
'relative'-weighted, time-weighted loss (passive_long_step_training) and driven
by the two-pass auto-tau_w loop (cm_profile_sweep). The ONLY differences from
the synthetic driver are the two synthetic-only concerns that do not exist on
real data:

    synthetic driver                        this (biological) driver
    ----------------                        ------------------------
    1. GENERATE archives from a manifest -> removed (real archives already exist)
    2. integrate_long_step (loss patch)  -> identical
    3. load_cells_from_archive           -> identical, specimen_ids=None
       + inject cm_true (ground truth)   -> removed (no truth; bias_log stays None)
    4. two-pass auto-tau_w per cell      -> identical
       + absolute-mV validation gate     -> ADDED (see compute_absolute_gate)
    5. Phase 2.5 (fix Ra @ cohort median)-> identical, ON by default
    6. Phase 3 (subset only)             -> configurable fraction (default 1/2)

Pipeline for the group:
    1. PATCH the monolith loss/split (integrate_long_step); the train/validation
       split is tau_w-independent so this is done once up front.
    2. LOAD every specimen_<id>/ archive in the group directory.
    3. TWO-PASS auto-tau_w, one cell at a time:
         interim cm_profile_sweep over the tau_w grid -> pick sharpest HW_rho
         -> RE-PATCH the loss at that tau_w* (verified) -> fit_one_cell at tau_w*
         -> ABSOLUTE-mV gate: recompute train/valid RMSD in mV at the fit point
            and re-classify with the monolith's calibrated mV thresholds. The
            fit itself stays on the unitless 'relative' loss; only the pass/fail
            gate is put back on a physical (mV) footing, because on real data
            there is no ground truth to fall back on and the mV thresholds are
            calibrated to real voltage error (the 'relative'-loss mismatch is
            documented in the benchmark report, caveats 8.1-8.2).
    4. PHASE 2.5 over the group (fix Ra at the cohort median, refit Cm,Rm;
       mutates results in place). Group == archive dir name (e.g. L3_exc).
    5. PHASE 3 (bootstrap CIs) for a configurable subset (default: half the
       fittable cells; --phase3-subset all|first:N|frac:F|none).

Heavy deps (neuron, the monolith, the sweep) are imported INSIDE main() so the
pure helpers below import without NEURON and are unit-tested by
smoke_run_biological_fit.py.
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ===========================================================================
#  Progress logging -- timestamped + flushed so it lands in the PBS .o file
#  in REAL TIME (Python buffers stdout when it is a file, so flush=True is
#  what makes `tail -f job.o*` actually show progress as it happens).
# ===========================================================================
_T0 = time.time()


def log(msg: str) -> None:
    """Print '[HH:MM:SS | +MM:SS] msg' and flush immediately."""
    now = datetime.now().strftime("%H:%M:%S")
    el = time.time() - _T0
    elapsed = "+{:02d}:{:02d}".format(int(el // 60), int(el % 60))
    print("[{} | {}] {}".format(now, elapsed, msg), flush=True)


# ===========================================================================
#  PURE helpers (NEURON-free; unit-tested by smoke_run_biological_fit.py)
# ===========================================================================
def parse_float_list(s: str) -> List[float]:
    """'2.0,5.0,10.0' -> [2.0, 5.0, 10.0]. Empty tokens are dropped."""
    return [float(x) for x in str(s).split(",") if x.strip() != ""]


def parse_window(s: str) -> Tuple[float, float]:
    """'0.5,100.0' -> (0.5, 100.0)."""
    a = parse_float_list(s)
    if len(a) != 2:
        raise ValueError("window must be 'lo,hi', got {!r}".format(s))
    return (a[0], a[1])


def pick_winning_tau_w(profiles: Sequence, tau_grid: Sequence[float]
                       ) -> Tuple[float, str, Optional[object]]:
    """Per-cell tau_w selection: sharpest C_m profile = smallest finite HW_rho.

    `profiles` is the list of CmProfile objects (one per tau_w) for ONE cell.
    Falls back to the middle of the grid (and flags it) when no profile has a
    finite HW_rho -- a degenerate sweep, which the caller should treat as a
    non-identified cell, not a silent default. With a single-point grid (the
    validated benchmark default tau_w=5.0) the winner is trivially that point.

    Returns (tau_w_star, reason, winner_profile_or_None).
    """
    finite = [p for p in profiles if np.isfinite(getattr(p, "hw_rho", np.nan))]
    if not finite:
        mid = float(sorted(tau_grid)[len(tau_grid) // 2])
        return mid, "no_finite_hw_rho->fallback_mid", None
    winner = min(finite, key=lambda p: p.hw_rho)
    return float(winner.tau_w_ms), "sharpest_hw_rho", winner


def select_phase3_subset(specimen_ids: Sequence[int], spec: str) -> List[int]:
    """Resolve which specimen_ids get Phase 3.

    Unlike the synthetic driver (which selected from a manifest DataFrame), the
    biological driver has no manifest; it selects directly from the list of
    fitted specimen_ids.

    spec grammar (identical to the synthetic driver):
        ""/"none" -> [];  "all" -> every id;
        "first:N" -> first N in the given order;
        "frac:F"  -> ~F fraction, deterministic by specimen_id sort.
    """
    sids = [int(s) for s in specimen_ids]
    spec = (spec or "").strip().lower()
    if spec in ("", "none"):
        return []
    if spec == "all":
        return sids
    if spec.startswith("first:"):
        n = int(spec.split(":", 1)[1])
        return sids[:max(0, n)]
    if spec.startswith("frac:"):
        f = float(spec.split(":", 1)[1])
        k = max(1, int(round(f * len(sids)))) if sids else 0
        return sorted(sids)[:k]
    raise ValueError("unrecognised phase3-subset spec {!r}".format(spec))


def compute_absolute_gate(
    train_bundle_rmsds_mV: Sequence[float],
    valid_bundle_rmsds_mV: Sequence[float],
    *,
    classify_fn: Callable[..., str],
    k_good: float,
    k_fail: float,
    train_fail_mV: float,
    valid_good_mV: float,
) -> Tuple[float, float, float, str]:
    """PURE core of the absolute-mV validation gate (NEURON-free, I/O-free).

    Given per-bundle ABSOLUTE-mV RMSDs (already simulated at the fit point) for
    the training and validation bundles, return
        (train_abs_mV, valid_abs_mV, ratio_abs, status)
    where status = classify_fn(train_abs, valid_abs, k_good, k_fail,
    train_fail_mV, valid_rmsd_good_mV=valid_good_mV).

    Why this exists
    ---------------
    Under the patched 'relative' loss, PassiveFitResult.train_rmsd_mV is the
    UNITLESS relative loss (mean of RMSD/deflection), whereas the held-out
    validation RMSD is in absolute mV. Mixing them makes valid_to_train_ratio
    and the monolith's mV thresholds (train_fail 2.0 mV, valid_good 0.2 mV,
    k_good 3x, k_fail 10x) misfire -- exactly the benchmark report's "almost
    every cell reads good" (caveats 8.1-8.2). On synthetic data the fallback was
    the recovery ratios; on real data there is no ground truth, so the pass/fail
    gate must be put back on a physical footing. This recomputes BOTH sides in
    mV at the fit point and re-classifies with the SAME calibrated thresholds,
    so validation_status means something on real recordings. The fit itself is
    untouched -- it stays the 'relative' quasi-MLE (theory note section 6).

    classify_fn is injected (defaults to mono._classify_fit in main) so this
    stays a single source of truth and is testable with a fake classifier.
    """
    tr = [float(r) for r in train_bundle_rmsds_mV if np.isfinite(r)]
    va = [float(r) for r in valid_bundle_rmsds_mV if np.isfinite(r)]
    train_abs = float(np.mean(tr)) if tr else float("nan")
    valid_abs = float(np.mean(va)) if va else float("nan")
    if np.isfinite(valid_abs) and np.isfinite(train_abs):
        ratio_abs = valid_abs / max(train_abs, 1e-9)
    else:
        ratio_abs = float("inf")
    status = classify_fn(train_abs, valid_abs, k_good, k_fail, train_fail_mV,
                         valid_rmsd_good_mV=valid_good_mV)
    return train_abs, valid_abs, ratio_abs, status


# Serialisable scalar fields extracted from each PassiveFitResult. The block
# after the blank comment are the ones ADDED by this driver / by Phase 2.5.
_RESULT_FIELDS = [
    "specimen_id", "layer", "dendrite_type", "F", "fit_target",
    "cm_uF_per_cm2", "rm_Ohm_cm2", "ra_Ohm_cm",
    "cm_sigma", "rm_sigma", "ra_sigma",
    "train_rmsd_mV", "valid_rmsd_mV", "valid_to_train_ratio",
    "rin_MOhm_allen", "tau_ms_allen", "v_rest_mV",
    "validation_status", "n_calls", "n_initial", "wall_time_s",
    "noise_sigma_mV", "noise_rho_lag1", "error_message",
    # two-pass auto-tau_w:
    "tau_w_chosen_ms", "tau_w_hw_rho", "tau_w_reason",
    # absolute-mV gate (this driver):
    "validation_status_relative", "train_rel_loss",
    "train_rmsd_abs_mV", "valid_rmsd_abs_mV", "valid_to_train_ratio_abs",
    # Phase 2.5 (present after 2.5 only):
    "cm_phase2", "rm_phase2", "ra_phase2", "train_rmsd_phase2",
]

# Fields that are strings (blank rather than NaN when absent).
_STR_FIELDS = {
    "layer", "dendrite_type", "fit_target", "validation_status",
    "validation_status_relative", "tau_w_reason", "error_message",
}


def results_to_dataframe(results: Sequence) -> pd.DataFrame:
    """Extract the serialisable scalar fields from a list of PassiveFitResult
    (duck-typed; missing attributes -> NaN for numeric, '' for string), incl.
    the two-pass tau_w_* fields, the absolute-gate columns, and Phase 2.5's
    *_phase2 stash. Drops gp_result / neuron_cell / opt_inputs (unserialisable).
    """
    rows = []
    for r in results:
        row = {}
        for f in _RESULT_FIELDS:
            val = getattr(r, f, None)
            if val is None:
                row[f] = "" if f in _STR_FIELDS else np.nan
            else:
                row[f] = val
        rows.append(row)
    return pd.DataFrame(rows, columns=_RESULT_FIELDS)


# ===========================================================================
#  Orchestration (NEURON-side; lazy heavy imports)
# ===========================================================================
def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    sys.path.insert(0, args.code_dir)

    # --- lazy heavy imports (only when actually running on the cluster) ------
    import cm_profile_sweep as cps
    import passive_long_step_training as plst
    import passive_fitting_hpc_fixed as mono
    from neuron import h
    h.load_file("stdrun.hoc")

    def _clear():
        for s in list(h.allsec()):
            h.delete_section(sec=s)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tau_grid = parse_float_list(args.tau_w_grid_ms)
    ss_window = parse_window(args.ss_window_ms)
    cm_bounds = mono.DEFAULT_CM_BOUNDS
    rm_bounds = mono.DEFAULT_RM_BOUNDS
    ra_bounds = mono.DEFAULT_RA_BOUNDS
    group_label = Path(args.archive_dir).name

    log("GROUP {}: archive={}  tau_w grid={}  sweep n_grid={}  n_calls={}"
        .format(group_label, args.archive_dir, tau_grid, args.sweep_n_grid,
                args.n_calls))

    # ---- 1. PATCH long-step (split is tau_w-independent; placeholder tau_w) --
    log("[1/5] PATCH -- integrate_long_step (train/validation split + loss) ...")
    ls_kwargs = dict(
        n_long_train=args.n_long_train,
        max_ls_train_deflection_mV=args.ls_deflection_cap,
        max_sag_amplitude_mV=(None if args.max_sag_amplitude_mV < 0
                              else float(args.max_sag_amplitude_mV)),
        r_in_target=args.r_in_target, weighting=args.weighting,
        ss_window_ms=ss_window, ss_time_weight=args.ss_time_weight,
        ls_window_ms_after_onset=args.ls_window_ms,
        ss_t0_ms=(None if args.ss_t0_ms is None else float(args.ss_t0_ms)),
    )
    plst.integrate_long_step(mono, ss_tau_w_ms=tau_grid[0], **ls_kwargs,
                             verbose=True)
    log("[1/5] PATCH -- done.")

    # ---- 2. LOAD this group's archives --------------------------------------
    log("[2/5] LOAD -- reading specimen_* archives + optimiser inputs ...")
    cells_data = mono.load_cells_from_archive(
        args.archive_dir, n_avg_groups=args.n_avg_groups,
        specimen_ids=None, max_cells=args.max_cells, verbose=True)
    if not cells_data:
        log("[ABORT] No cells loaded from archive: {}".format(args.archive_dir))
        sys.exit(1)
    opt_inputs = [mono.prepare_optimiser_inputs(cd, fit_target=args.fit_target)
                  for cd in cells_data]
    log("[2/5] LOAD -- {} cell(s) loaded.".format(len(cells_data)))

    # ---- 3. TWO-PASS auto-tau_w fit (+ absolute gate), one cell at a time ----
    log("[3/5] FIT -- two-pass auto-tau_w over {} cell(s) (sequential) ..."
        .format(len(cells_data)))
    results: List[object] = []
    tau_rows: List[dict] = []
    failed_ids: List[int] = []
    for i, (cd, oi) in enumerate(zip(cells_data, opt_inputs)):
        sid = int(cd.specimen_id)
        t_cell = time.time()
        log("  cell {}/{} (specimen {}): START"
            .format(i + 1, len(cells_data), sid))
        _clear()
        cell = None
        try:
            cell = mono.build_neuron_model(cd.swc_path, F=args.F)
            _assert_loss_live(cps, cell, oi, tau_grid[0], args)

            log("  cell {}/{} ({}): SWEEP ({} tau_w x {} Cm pts) ..."
                .format(i + 1, len(cells_data), sid, len(tau_grid),
                        args.sweep_n_grid))
            t_sw = time.time()
            sweep = cps.sweep_tau_w_per_cell(
                [cps.CellSweepInput(
                    specimen_id=sid, cell=cell, train_bundles=oi.train_bundles,
                    v_rest_mV=float(oi.v_rest_mV),
                    ss_window_ms=tuple(oi.train_window_ms),
                    cm_true=None)],                     # no ground truth on real data
                tau_w_grid_ms=tau_grid, shape=args.ss_time_weight,
                rho=args.sweep_rho, cm_bounds=cm_bounds, n_grid=args.sweep_n_grid,
                rm_bounds=rm_bounds, ra_bounds=ra_bounds,
                r_in_target=args.r_in_target,
                ls_window_ms_after_onset=args.ls_window_ms, verbose=True)
            tau_w_star, reason, winner = pick_winning_tau_w(sweep[sid], tau_grid)
            log("  cell {}/{} ({}): SWEEP done in {:.0f}s -> tau_w*={} ms ({})"
                .format(i + 1, len(cells_data), sid, time.time() - t_sw,
                        tau_w_star, reason))

            # re-patch the loss at tau_w_star and VERIFY it took effect
            plst.integrate_long_step(mono, ss_tau_w_ms=tau_w_star, **ls_kwargs,
                                     verbose=False)
            _verify_tau_w_applied(cps, mono, cell, oi, tau_w_star, args)

            log("  cell {}/{} ({}): FIT (n_calls={}) ..."
                .format(i + 1, len(cells_data), sid, args.n_calls))
            t_fit = time.time()
            fr = mono.fit_one_cell(cell, cd, oi, F=args.F, n_calls=args.n_calls,
                                   n_initial=args.n_initial, seed=i)
            fr.tau_w_chosen_ms = float(tau_w_star)
            fr.tau_w_hw_rho = (float(getattr(winner, "hw_rho", np.nan))
                               if winner else np.nan)
            fr.tau_w_reason = reason

            # absolute-mV validation gate (real-data pass/fail; see helper)
            _apply_absolute_gate(mono, plst, cell, oi, fr, args)

            results.append(fr)
            tau_rows.append(dict(
                specimen_id=sid, tau_w_chosen_ms=tau_w_star, reason=reason,
                hw_rho=(winner.hw_rho if winner else np.nan),
                kappa=(winner.kappa if winner else np.nan)))
            log("  cell {}/{} ({}): FIT done in {:.0f}s -> "
                "Cm={:.3f} Rm={:.0f} Ra={:.0f} status={} "
                "(valid_abs={:.3f} mV) | cell total {:.0f}s"
                .format(i + 1, len(cells_data), sid, time.time() - t_fit,
                        fr.cm_uF_per_cm2, fr.rm_Ohm_cm2, fr.ra_Ohm_cm,
                        fr.validation_status,
                        float(getattr(fr, "valid_rmsd_abs_mV", np.nan)),
                        time.time() - t_cell))
        except Exception as exc:  # noqa: BLE001
            log("  cell {}/{} ({}): FAILED after {:.0f}s -> {}: {}"
                .format(i + 1, len(cells_data), sid, time.time() - t_cell,
                        type(exc).__name__, exc))
            failed_ids.append(sid)
            if args.fail_fast:
                raise
        finally:
            if cell is not None:
                try:
                    cell.destroy()
                except Exception:
                    pass
    _clear()

    if failed_ids:
        (out / "failed_cells.txt").write_text(
            "\n".join(str(s) for s in failed_ids) + "\n")
        log("[3/5] FIT -- {} cell(s) FAILED (see failed_cells.txt): {}"
            .format(len(failed_ids), failed_ids))

    pd.DataFrame(tau_rows).to_csv(out / "tau_w_choice.csv", index=False)
    results_to_dataframe(results).to_csv(out / "phase2_results.csv", index=False)
    log("[3/5] FIT -- done: {} fit(s) -> phase2_results.csv".format(len(results)))

    if not results:
        log("[ABORT] No successful fits; skipping Phase 2.5 / Phase 3.")
        sys.exit(1)

    # ---- 4. PHASE 2.5 (mutates results in place; group = archive dir name) --
    phase2p5_ran = not args.skip_phase2p5
    if phase2p5_ran:
        log("[4/5] PHASE 2.5 -- profile Ra + fix at cohort median + refit ...")
        t_p25 = time.time()
        mono.run_phase2p5_for_group(
            results=results, cells_data=cells_data, opt_inputs=opt_inputs,
            F=args.F, group_label=group_label, output_dir=out,
            n_floor=args.n_floor, n_ra_profile=args.n_ra_profile,
            n_calls=args.n_calls, n_initial=args.n_initial,
            acq_func=mono.DEFAULT_ACQ_FUNC, make_plots=True, seed=0, verbose=True)
        results_to_dataframe(results).to_csv(
            out / "phase2p5_combined_results.csv", index=False)
        log("[4/5] PHASE 2.5 -- done in {:.0f}s".format(time.time() - t_p25))
    else:
        log("[4/5] PHASE 2.5 -- SKIPPED (--skip-phase2p5): Ra free; Phase 3 = 3-D.")

    # ---- 5. PHASE 3 (configurable subset; default half the fittable cells) --
    fittable = [int(r.specimen_id) for r in results
                if r.validation_status in ("good", "to_refine")
                and getattr(r, "gp_result", None) is not None]
    subset = select_phase3_subset(fittable, args.phase3_subset)
    if subset:
        log("[5/5] PHASE 3 -- bootstrap subset ({}/{} fittable): {} ..."
            .format(len(subset), len(fittable), subset))
        t_p3 = time.time()
        _run_phase3_subset(mono, results, cells_data, subset,
                           phase2p5_ran, args, out)
        log("[5/5] PHASE 3 -- done in {:.0f}s".format(time.time() - t_p3))
    else:
        log("[5/5] PHASE 3 -- empty subset (spec={!r}; {} fittable) -> skipped"
            .format(args.phase3_subset, len(fittable)))

    log("DONE group {} -- total {:.0f}s".format(group_label, time.time() - _T0))


# ---------------------------------------------------------------------------
#  Absolute-mV validation gate (NEURON-side wrapper around compute_absolute_gate)
# ---------------------------------------------------------------------------
def _apply_absolute_gate(mono, plst, cell, oi, fr, args) -> None:
    """Recompute train/validation RMSD in ABSOLUTE mV at the fit point and
    re-classify with the monolith's calibrated mV thresholds, then MUTATE fr:

      * stash the fitter's own (relative) status/loss on
        validation_status_relative / train_rel_loss,
      * write train_rmsd_abs_mV / valid_rmsd_abs_mV / valid_to_train_ratio_abs,
      * OVERWRITE validation_status with the absolute-mV verdict (this is the
        status Phase 3's eligibility gate then consumes).

    The gate reflects the Phase-2 (free-Ra) fit; Phase 2.5 (which has no
    validation pass) does not recompute it. The training RMSD here is UNWEIGHTED
    absolute mV (ss_sample_weight_fn=None), matching how the held-out validation
    RMSD is computed, so train/valid are on the same footing.
    """
    fr.validation_status_relative = getattr(fr, "validation_status", "")
    fr.train_rel_loss = float(getattr(fr, "train_rmsd_mV", np.nan))

    if not np.isfinite(getattr(fr, "cm_uF_per_cm2", np.nan)):
        fr.train_rmsd_abs_mV = np.nan
        fr.valid_rmsd_abs_mV = np.nan
        fr.valid_to_train_ratio_abs = float("inf")
        return

    cell.set_passive(Cm=float(fr.cm_uF_per_cm2), Rm=float(fr.rm_Ohm_cm2),
                     Ra=float(fr.ra_Ohm_cm))
    cell.set_e_pas(float(fr.v_rest_mV))

    train_rmsds = []
    for b in oi.train_bundles:
        rmsd, _defl = plst.bundle_rmsd(
            cell, b, float(fr.v_rest_mV),
            ss_window_ms=tuple(oi.train_window_ms),
            ls_window_ms_after_onset=args.ls_window_ms,
            r_in_target=args.r_in_target, ss_sample_weight_fn=None)
        train_rmsds.append(float(rmsd))

    valid_rmsds = []
    for vb in oi.validation_bundles:
        try:
            r = mono._rmsd_for_validation_bundle(
                cell, vb, float(fr.v_rest_mV),
                valid_window_ms_after_onset=mono.DEFAULT_VALID_WINDOW_MS_AFTER_ONSET)
            valid_rmsds.append(float(r))
        except Exception:
            pass

    train_abs, valid_abs, ratio_abs, status_abs = compute_absolute_gate(
        train_rmsds, valid_rmsds,
        classify_fn=mono._classify_fit,
        k_good=mono.DEFAULT_K_GOOD, k_fail=mono.DEFAULT_K_FAIL,
        train_fail_mV=mono.DEFAULT_TRAIN_RMSD_FAIL_MV,
        valid_good_mV=mono.DEFAULT_VALID_RMSD_GOOD_MV)

    fr.train_rmsd_abs_mV = train_abs
    fr.valid_rmsd_abs_mV = valid_abs
    fr.valid_to_train_ratio_abs = ratio_abs
    fr.validation_status = status_abs


# ---------------------------------------------------------------------------
#  NEURON-side fail-loud checks (never silently mis-fit)
# ---------------------------------------------------------------------------
def _assert_loss_live(cps, cell, oi, tau_w_ms, args):
    """Loss must vary across C_m; constant (~1e6) => dead cell / failed sim."""
    L = cps.build_relative_loss_for_tau_w(
        cell, oi.train_bundles, float(oi.v_rest_mV), tau_w_ms=float(tau_w_ms),
        ss_window_ms=tuple(oi.train_window_ms), shape=args.ss_time_weight,
        ls_window_ms_after_onset=args.ls_window_ms, r_in_target=args.r_in_target)
    vals = [float(L(np.log(c), np.log(15000.0), np.log(150.0)))
            for c in (0.5, 1.0, 2.0)]
    if (not np.all(np.isfinite(vals))) or (max(vals) - min(vals) < 1e-9):
        raise RuntimeError(
            "loss constant across C_m {} -> dead cell or simulate() failed; "
            "the sweep would be garbage.".format([round(v, 4) for v in vals]))


def _verify_tau_w_applied(cps, mono, cell, oi, tau_w_star, args):
    """Confirm the monolith's patched loss now uses tau_w_star (fail loud if a
    future double-patch guard ever silently pins one tau_w)."""
    patched = mono._build_loss_function(
        cell, oi.train_bundles, float(oi.v_rest_mV), tuple(oi.train_window_ms))
    ref = cps.build_relative_loss_for_tau_w(
        cell, oi.train_bundles, float(oi.v_rest_mV), tau_w_ms=float(tau_w_star),
        ss_window_ms=tuple(oi.train_window_ms), shape=args.ss_time_weight,
        ls_window_ms_after_onset=args.ls_window_ms, r_in_target=args.r_in_target,
        weighting=args.weighting)
    pt = (np.log(1.5), np.log(8000.0), np.log(200.0))   # generic off-optimum point
    lp, lr = float(patched(*pt)), float(ref(*pt))
    if (not np.isfinite(lp) or not np.isfinite(lr)
            or abs(lp - lr) > 1e-6 * max(abs(lr), 1.0)):
        raise RuntimeError(
            "patched loss ({:.6g}) != tau_w*={} reference ({:.6g}); "
            "integrate_long_step did NOT apply the chosen tau_w. Investigate "
            "before trusting the fit.".format(lp, tau_w_star, lr))


def _run_phase3_subset(mono, results, cells_data, subset_ids,
                       phase2p5_ran, args, out):
    """Bootstrap CIs for the selected subset, mirroring the monolith __main__
    Phase-3 loop (fix_ra iff Phase 2.5 ran). Sequential, one cell at a time."""
    print("\n{0}\n  PHASE 3 -- bootstrap (subset: {1})\n{0}"
          .format("=" * 60, subset_ids))
    common = dict(B=args.bootstrap_B, alpha=0.95, fit_mode="fast",
                  n_calls=args.bootstrap_n_calls,
                  n_initial=args.bootstrap_n_initial,
                  ball_radius_log=0.2, rmsd_reject_mult=5.0, n_workers=1,
                  fix_ra=phase2p5_ran)
    gp_kwargs = dict(n_grid=80, inner_grid_per_axis=30, envelope_k=2.0,
                     n_validation_per_bound=5, validation_ball_logradius=0.05,
                     trust_abs_mv=0.10, trust_zscore=3.0)
    by_sid = {int(cd.specimen_id): cd for cd in cells_data}
    p3_results, rows = [], []
    for i, fr in enumerate(results):
        sid = int(fr.specimen_id)
        if sid not in subset_ids:
            continue
        if (fr.validation_status not in ("good", "to_refine")
                or getattr(fr, "gp_result", None) is None):
            print("[Phase 3] {}: not fittable (status={}) -> skip"
                  .format(sid, fr.validation_status))
            continue
        cd = by_sid[sid]
        pc = None
        try:
            pc = mono.build_neuron_model(cd.swc_path, F=float(fr.F))
            pc.set_passive(fr.cm_uF_per_cm2, fr.rm_Ohm_cm2, fr.ra_Ohm_cm)
            pc.set_e_pas(fr.v_rest_mV)
            fr.neuron_cell = pc
            if args.bootstrap_mode == "parametric":
                bkw = {**common, "bootstrap_mode": "parametric",
                       "swc_path": str(cd.swc_path),
                       "noise_mode": args.noise_mode, "seed": i}
            else:
                bkw = {**common, "bootstrap_mode": "nonparametric",
                       "pulse_pool": cd.ss_individual_pulses,
                       "swc_path": str(cd.swc_path),
                       "n_pulses_per_replicate":
                           max(1, len(cd.ss_individual_pulses) // 3),
                       "n_avg_groups_bootstrap": args.n_avg_groups, "seed": i}
            p3 = mono.phase3_full_for_cell(
                fit_result=fr, root_dir=str(out), bootstrap_kwargs=bkw,
                gp_kwargs={**gp_kwargs, "seed": i}, verbose=True)
            p3_results.append(p3)
            mono.save_replot_bundle(phase3_result=p3, fit_result=fr, cell_data=cd,
                                    F_used=float(fr.F), root_dir=str(out),
                                    verbose=True)
            b = p3.bootstrap
            for ip, p in enumerate(("Cm", "Rm", "Ra")):
                rows.append(dict(specimen_id=sid, parameter=p,
                                 mle=b.mle_physical[ip],
                                 ci_bca_lo=b.ci_bca[p][0], ci_bca_hi=b.ci_bca[p][1],
                                 ci_perc_lo=b.ci_percentile[p][0],
                                 ci_perc_hi=b.ci_percentile[p][1],
                                 ci_norm_lo=b.ci_normal[p][0],
                                 ci_norm_hi=b.ci_normal[p][1],
                                 n_kept=b.n_kept, mode=b.bootstrap_mode))
        except Exception as exc:  # noqa: BLE001
            print("[Phase 3] {} FAILED: {}: {}"
                  .format(sid, type(exc).__name__, exc))
        finally:
            if pc is not None:
                try:
                    pc.destroy()
                except Exception:
                    pass
            fr.neuron_cell = None
    if rows:
        pd.DataFrame(rows).to_csv(out / "phase3_full_summary.csv", index=False)
        print("[Phase 3] CIs -> {}".format(out / "phase3_full_summary.csv"))


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------
def _parse_args(argv):
    ap = argparse.ArgumentParser(
        description="Passive-fit pipeline on real recordings -- one group.")
    ap.add_argument("--archive-dir", required=True,
                    help="Group archive dir: <ROOT>/<GROUP> with specimen_*/.")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--code-dir", required=True,
                    help="Dir containing passive_fitting_hpc_fixed.py, "
                         "cm_profile_sweep.py, passive_long_step_training.py.")
    ap.add_argument("--n-avg-groups", type=int, default=1)
    ap.add_argument("--max-cells", type=int, default=None)
    ap.add_argument("--fail-fast", action="store_true",
                    help="Abort the whole group on the first per-cell error "
                         "(default: log + continue, robust for real data).")
    # fit
    ap.add_argument("--fit-target", default="hyp", choices=["dep", "hyp", "both"])
    ap.add_argument("--F", type=float, default=1.9)
    ap.add_argument("--n-calls", type=int, default=100)
    ap.add_argument("--n-initial", type=int, default=50)
    # two-pass auto-tau_w + multi-protocol loss
    ap.add_argument("--n-long-train", type=int, default=2)
    ap.add_argument("--ls-deflection-cap", type=float, default=12.0)
    ap.add_argument("--max-sag-amplitude-mV", type=float, default=-1.0,
                    help="Optional sag-gated I_h guard ceiling (mV). Negative "
                         "= disabled. Needs sag_ratio on CellData (not carried "
                         "by default); falls back to the deflection cap if NaN.")
    ap.add_argument("--r-in-target", default="peak", choices=["peak", "steady"])
    ap.add_argument("--weighting", default="relative")
    ap.add_argument("--ss-window-ms", default="0.5,100.0")
    ap.add_argument("--ss-t0-ms", default=None)
    ap.add_argument("--ss-time-weight", default="exp",
                    choices=["exp", "gauss", "none"])
    ap.add_argument("--ls-window-ms", type=float, default=150.0)
    ap.add_argument("--tau-w-grid-ms", default="5.0")
    ap.add_argument("--sweep-rho", type=float, default=0.5)
    ap.add_argument("--sweep-n-grid", type=int, default=15)
    # Phase 2.5
    ap.add_argument("--skip-phase2p5", action="store_true")
    ap.add_argument("--n-floor", type=int, default=4)
    ap.add_argument("--n-ra-profile", type=int, default=50)
    # Phase 3 (subset)
    ap.add_argument("--phase3-subset", default="frac:0.5",
                    help="none | all | first:N | frac:F (default frac:0.5).")
    ap.add_argument("--bootstrap-B", type=int, default=200)
    ap.add_argument("--bootstrap-mode", default="nonparametric",
                    choices=["parametric", "nonparametric"])
    ap.add_argument("--noise-mode", default="block",
                    choices=["iid", "ar1", "block"])
    ap.add_argument("--bootstrap-n-calls", type=int, default=60)
    ap.add_argument("--bootstrap-n-initial", type=int, default=20)
    return ap.parse_args(argv)


if __name__ == "__main__":
    main()
