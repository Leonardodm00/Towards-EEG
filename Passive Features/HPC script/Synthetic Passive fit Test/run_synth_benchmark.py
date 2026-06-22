# -*- coding: utf-8 -*-
"""
run_synth_benchmark.py
======================

Phase 2 entrypoint of the da Vinci synthetic passive-fit benchmark — ONE cohort
per invocation (called by submit_synth_benchmark.sh with $GROUP).

Pipeline for the cohort:
    1. GENERATE archives from the manifest rows (gen_from_manifest).
    2. PATCH the monolith with the long-step training + multi-protocol loss
       (passive_long_step_training.integrate_long_step); the train/validation
       SPLIT is tau_w-independent so this is done once.
    3. LOAD the cohort's archives (load_cells_from_archive).
    4. TWO-PASS auto-tau_w, one cell at a time:
         interim cm_profile_sweep over the tau_w grid -> pick sharpest HW_rho
         -> RE-PATCH the loss at that tau_w* (verified) -> fit_one_cell at tau_w*.
    5. PHASE 2.5 over the cohort (run_phase2p5_for_group; mutates results in
       place: cm/rm/ra become the fixed-Ra refit). Cohort == archive dir name.
    6. PHASE 3 (bootstrap CIs) for the held-out calibration subset only.

Heavy deps (neuron, the monolith, the generator) are imported INSIDE main() so
the pure helpers below import without NEURON and are unit-tested by
smoke_run_synth_benchmark.py.
"""

import argparse
import dataclasses
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ===========================================================================
#  PURE helpers (NEURON-free; unit-tested)
# ===========================================================================
def parse_float_list(s: str) -> List[float]:
    """'2.0,5.0,10.0' -> [2.0, 5.0, 10.0]."""
    return [float(x) for x in str(s).split(",") if x.strip() != ""]


def parse_window(s: str) -> Tuple[float, float]:
    """'0.5,100.0' -> (0.5, 100.0)."""
    a = parse_float_list(s)
    if len(a) != 2:
        raise ValueError(f"window must be 'lo,hi', got {s!r}")
    return (a[0], a[1])


def pick_winning_tau_w(profiles: Sequence, tau_grid: Sequence[float]
                       ) -> Tuple[float, str, Optional[object]]:
    """Per-cell tau_w selection: sharpest C_m profile = smallest finite HW_rho.

    `profiles` is the list of CmProfile objects (one per tau_w) for ONE cell.
    Falls back to the middle of the grid (and flags it) when no profile has a
    finite HW_rho -- a degenerate sweep, which the caller should treat as a
    non-identified cell, not a silent default.

    Returns (tau_w_star, reason, winner_profile_or_None).
    """
    finite = [p for p in profiles if np.isfinite(getattr(p, "hw_rho", np.nan))]
    if not finite:
        mid = float(sorted(tau_grid)[len(tau_grid) // 2])
        return mid, "no_finite_hw_rho->fallback_mid", None
    winner = min(finite, key=lambda p: p.hw_rho)
    return float(winner.tau_w_ms), "sharpest_hw_rho", winner


def select_phase3_subset(group_df: pd.DataFrame, spec: str) -> List[int]:
    """Resolve which specimen_ids in this cohort get Phase 3.

    spec grammar:  ""/"none" -> []; "all" -> every cell;
                   "first:N" -> first N by manifest order;
                   "frac:F"  -> ~F fraction, deterministic by specimen_id sort.
    """
    sids = [int(s) for s in group_df["specimen_id"].tolist()]
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
    raise ValueError(f"unrecognised phase3-subset spec {spec!r}")


_RESULT_FIELDS = [
    "specimen_id", "layer", "dendrite_type", "F", "fit_target",
    "cm_uF_per_cm2", "rm_Ohm_cm2", "ra_Ohm_cm",
    "cm_sigma", "rm_sigma", "ra_sigma",
    "train_rmsd_mV", "valid_rmsd_mV", "valid_to_train_ratio",
    "rin_MOhm_allen", "tau_ms_allen", "v_rest_mV",
    "validation_status", "n_calls", "n_initial", "wall_time_s",
    "noise_sigma_mV", "noise_rho_lag1", "error_message",
    # added by this benchmark / by Phase 2.5 (present after 2.5 only):
    "tau_w_chosen_ms", "tau_w_hw_rho", "tau_w_reason",
    "cm_phase2", "rm_phase2", "ra_phase2", "train_rmsd_phase2",
]


def results_to_dataframe(results: Sequence) -> pd.DataFrame:
    """Extract the serialisable scalar fields from a list of PassiveFitResult
    (duck-typed; missing attributes -> NaN/''), incl. the benchmark's tau_w_*
    and Phase 2.5's *_phase2 stash. Drops gp_result / neuron_cell / opt_inputs."""
    rows = []
    for r in results:
        row = {}
        for f in _RESULT_FIELDS:
            val = getattr(r, f, None)
            row[f] = (np.nan if val is None and not f.endswith(("reason", "message"))
                      else ("" if val is None else val))
        rows.append(row)
    return pd.DataFrame(rows, columns=_RESULT_FIELDS)


# ===========================================================================
#  Orchestration (NEURON-side; lazy heavy imports)
# ===========================================================================
def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    sys.path.insert(0, args.code_dir)

    # --- lazy heavy imports (only when actually running on the cluster) ------
    import synth_gt_grid as sgg
    import gen_from_manifest as gfm
    import cm_profile_sweep as cps
    import passive_long_step_training as plst
    import passive_fitting_hpc_fixed as mono
    import synthetic_ground_truth as sgt
    from neuron import h
    h.load_file("stdrun.hoc")

    def _clear():
        for s in list(h.allsec()):
            h.delete_section(sec=s)

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    tau_grid = parse_float_list(args.tau_w_grid_ms)
    ss_window = parse_window(args.ss_window_ms)
    cm_bounds = sgg.DEFAULT_CM_BOUNDS
    rm_bounds = sgg.DEFAULT_RM_BOUNDS
    ra_bounds = sgg.DEFAULT_RA_BOUNDS

    # ---- 1. manifest slice for this cohort ----------------------------------
    manifest = sgg.load_manifest(args.manifest)
    group_df = sgg.group_of(manifest, args.group)
    print(f"\n{'='*60}\n  COHORT {args.group} — {len(group_df)} cell(s)\n{'='*60}")

    # ---- 2. GENERATE archives -----------------------------------------------
    gfm.generate_group(
        group_df, args.archive_dir, sgt=sgt, mono=mono,
        ss_n_repeats=args.ss_n_repeats,
        ls_hyp_amplitudes_pA=parse_float_list(args.ls_hyp_amps),
        clear_fn=_clear, verbose=True)

    # ---- 3. PATCH long-step (split is tau_w-independent; placeholder tau_w) --
    ls_kwargs = dict(
        n_long_train=args.n_long_train,
        max_ls_train_deflection_mV=args.ls_deflection_cap,
        r_in_target=args.r_in_target, weighting=args.weighting,
        ss_window_ms=ss_window, ss_time_weight=args.ss_time_weight,
        ls_window_ms_after_onset=args.ls_window_ms,
        ss_t0_ms=(None if args.ss_t0_ms is None else float(args.ss_t0_ms)),
    )
    plst.integrate_long_step(mono, ss_tau_w_ms=tau_grid[0], **ls_kwargs, verbose=True)

    # ---- 4. LOAD this cohort's archives -------------------------------------
    sids = [int(s) for s in group_df["specimen_id"]]
    cells_data = mono.load_cells_from_archive(
        args.archive_dir, n_avg_groups=args.n_avg_groups, specimen_ids=sids)
    opt_inputs = [mono.prepare_optimiser_inputs(cd, fit_target=args.fit_target)
                  for cd in cells_data]
    cm_true_by_sid = dict(zip(group_df["specimen_id"].astype(int),
                              group_df["cm_true"].astype(float)))

    # ---- 5. TWO-PASS auto-tau_w fit (one cell at a time) --------------------
    results: List[object] = []
    tau_rows: List[dict] = []
    for i, (cd, oi) in enumerate(zip(cells_data, opt_inputs)):
        sid = int(cd.specimen_id)
        print(f"\n--- cell {i+1}/{len(cells_data)} (specimen {sid}) ---")
        _clear()
        cell = mono.build_neuron_model(cd.swc_path, F=args.F)
        try:
            _assert_loss_live(cps, cell, oi, tau_grid[0], args)
            sweep = cps.sweep_tau_w_per_cell(
                [cps.CellSweepInput(
                    specimen_id=sid, cell=cell, train_bundles=oi.train_bundles,
                    v_rest_mV=float(oi.v_rest_mV),
                    ss_window_ms=tuple(oi.train_window_ms),
                    cm_true=cm_true_by_sid.get(sid))],
                tau_w_grid_ms=tau_grid, shape=args.ss_time_weight,
                rho=args.sweep_rho, cm_bounds=cm_bounds, n_grid=args.sweep_n_grid,
                rm_bounds=rm_bounds, ra_bounds=ra_bounds,
                r_in_target=args.r_in_target,
                ls_window_ms_after_onset=args.ls_window_ms, verbose=True)
            tau_w_star, reason, winner = pick_winning_tau_w(sweep[sid], tau_grid)

            # re-patch the loss at tau_w_star and VERIFY it took effect
            plst.integrate_long_step(mono, ss_tau_w_ms=tau_w_star, **ls_kwargs,
                                     verbose=False)
            _verify_tau_w_applied(cps, mono, cell, oi, tau_w_star, args)

            fr = mono.fit_one_cell(cell, cd, oi, F=args.F, n_calls=args.n_calls,
                                   n_initial=args.n_initial, seed=i)
            fr.tau_w_chosen_ms = float(tau_w_star)
            fr.tau_w_hw_rho = float(getattr(winner, "hw_rho", np.nan)) if winner else np.nan
            fr.tau_w_reason = reason
            results.append(fr)
            tau_rows.append(dict(
                specimen_id=sid, tau_w_chosen_ms=tau_w_star, reason=reason,
                hw_rho=(winner.hw_rho if winner else np.nan),
                kappa=(winner.kappa if winner else np.nan),
                bias_log=(getattr(winner, "bias_log", np.nan) if winner else np.nan)))
        finally:
            try:
                cell.destroy()
            except Exception:
                pass
    _clear()

    pd.DataFrame(tau_rows).to_csv(out / "tau_w_choice.csv", index=False)
    results_to_dataframe(results).to_csv(out / "phase2_results.csv", index=False)
    print(f"[Phase 2] {len(results)} fits + tau_w choices -> {out}")

    # ---- 6. PHASE 2.5 (mutates results in place; cohort = archive dir name) -
    phase2p5_ran = not args.skip_phase2p5
    if phase2p5_ran:
        mono.run_phase2p5_for_group(
            results=results, cells_data=cells_data, opt_inputs=opt_inputs,
            F=args.F, group_label=Path(args.archive_dir).name, output_dir=out,
            n_floor=args.n_floor, n_ra_profile=args.n_ra_profile,
            n_calls=args.n_calls, n_initial=args.n_initial,
            acq_func=mono.DEFAULT_ACQ_FUNC, make_plots=True, seed=0, verbose=True)
        results_to_dataframe(results).to_csv(
            out / "phase2p5_combined_results.csv", index=False)
    else:
        print("[Phase 2.5] SKIPPED (--skip-phase2p5): Ra free; Phase 3 = 3-D.")

    # ---- 7. PHASE 3 (calibration subset only) -------------------------------
    subset = select_phase3_subset(group_df, args.phase3_subset)
    if subset:
        _run_phase3_subset(mono, cps, results, cells_data, subset,
                           phase2p5_ran, args, out)
    else:
        print("[Phase 3] no subset for this cohort -> skipped")

    print(f"\n[DONE] cohort {args.group}")


# ---------------------------------------------------------------------------
#  NEURON-side fail-loud checks (the Cell-6 lesson: never silently mis-fit)
# ---------------------------------------------------------------------------
def _assert_loss_live(cps, cell, oi, tau_w_ms, args):
    """Loss must vary across C_m; constant (~1e6) => dead cell / failed sim."""
    L = cps.build_relative_loss_for_tau_w(
        cell, oi.train_bundles, float(oi.v_rest_mV), tau_w_ms=float(tau_w_ms),
        ss_window_ms=tuple(oi.train_window_ms), shape=args.ss_time_weight,
        ls_window_ms_after_onset=args.ls_window_ms, r_in_target=args.r_in_target)
    vals = [float(L(np.log(c), np.log(15000.0), np.log(150.0))) for c in (0.5, 1.0, 2.0)]
    if (not np.all(np.isfinite(vals))) or (max(vals) - min(vals) < 1e-9):
        raise RuntimeError(
            f"loss constant across C_m {[round(v,4) for v in vals]} -> dead cell "
            f"or simulate() failed; the sweep would be garbage.")


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
    if not np.isfinite(lp) or not np.isfinite(lr) or abs(lp - lr) > 1e-6 * max(abs(lr), 1.0):
        raise RuntimeError(
            f"patched loss ({lp:.6g}) != tau_w*={tau_w_star} reference ({lr:.6g}); "
            f"integrate_long_step did NOT apply the chosen tau_w. Investigate "
            f"before trusting the fit.")


def _run_phase3_subset(mono, cps, results, cells_data, subset_ids,
                       phase2p5_ran, args, out):
    """Bootstrap CIs for the calibration subset, mirroring the monolith __main__
    Phase-3 loop (fix_ra iff Phase 2.5 ran)."""
    print(f"\n{'='*60}\n  PHASE 3 — bootstrap (subset: {subset_ids})\n{'='*60}")
    common = dict(B=args.bootstrap_B, alpha=0.95, fit_mode="fast",
                  n_calls=args.bootstrap_n_calls, n_initial=args.bootstrap_n_initial,
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
        if fr.validation_status not in ("good", "to_refine") or fr.gp_result is None:
            print(f"[Phase 3] {sid}: not fittable (status={fr.validation_status}) -> skip")
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
                       "n_pulses_per_replicate": max(1, len(cd.ss_individual_pulses) // 3),
                       "n_avg_groups_bootstrap": args.n_avg_groups, "seed": i}
            p3 = mono.phase3_full_for_cell(
                fit_result=fr, root_dir=str(out), bootstrap_kwargs=bkw,
                gp_kwargs={**gp_kwargs, "seed": i}, verbose=True)
            p3_results.append(p3)
            mono.save_replot_bundle(phase3_result=p3, fit_result=fr, cell_data=cd,
                                    F_used=float(fr.F), root_dir=str(out), verbose=True)
            b = p3.bootstrap
            for ip, p in enumerate(("Cm", "Rm", "Ra")):
                rows.append(dict(specimen_id=sid, parameter=p, mle=b.mle_physical[ip],
                                 ci_bca_lo=b.ci_bca[p][0], ci_bca_hi=b.ci_bca[p][1],
                                 ci_perc_lo=b.ci_percentile[p][0],
                                 ci_perc_hi=b.ci_percentile[p][1],
                                 n_kept=b.n_kept, mode=b.bootstrap_mode))
        except Exception as exc:  # noqa: BLE001
            print(f"[Phase 3] {sid} FAILED: {type(exc).__name__}: {exc}")
        finally:
            if pc is not None:
                try:
                    pc.destroy()
                except Exception:
                    pass
            fr.neuron_cell = None
    if rows:
        pd.DataFrame(rows).to_csv(out / "phase3_full_summary.csv", index=False)
        print(f"[Phase 3] CIs -> {out / 'phase3_full_summary.csv'}")


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------
def _parse_args(argv):
    ap = argparse.ArgumentParser(description="Synthetic passive-fit benchmark — one cohort.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--group", required=True)
    ap.add_argument("--archive-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--code-dir", required=True)
    # generation
    ap.add_argument("--n-avg-groups", type=int, default=1)
    ap.add_argument("--ss-n-repeats", type=int, default=30)
    ap.add_argument("--ls-hyp-amps", default="-10,-30,-50,-70,-90")
    # fit
    ap.add_argument("--fit-target", default="hyp", choices=["dep", "hyp", "both"])
    ap.add_argument("--F", type=float, default=1.9)
    ap.add_argument("--n-calls", type=int, default=100)
    ap.add_argument("--n-initial", type=int, default=50)
    # interim two-pass auto-tau_w
    ap.add_argument("--n-long-train", type=int, default=2)
    ap.add_argument("--ls-deflection-cap", type=float, default=12.0)
    ap.add_argument("--r-in-target", default="peak", choices=["peak", "steady"])
    ap.add_argument("--weighting", default="relative")
    ap.add_argument("--ss-window-ms", default="0.5,100.0")
    ap.add_argument("--ss-t0-ms", default=None)
    ap.add_argument("--ss-time-weight", default="exp", choices=["exp", "gauss", "none"])
    ap.add_argument("--ls-window-ms", type=float, default=150.0)
    ap.add_argument("--tau-w-grid-ms", default="2.0,5.0,10.0")
    ap.add_argument("--sweep-rho", type=float, default=0.5)
    ap.add_argument("--sweep-n-grid", type=int, default=41)
    # Phase 2.5
    ap.add_argument("--skip-phase2p5", action="store_true")
    ap.add_argument("--n-floor", type=int, default=4)
    ap.add_argument("--n-ra-profile", type=int, default=50)
    # Phase 3 (subset)
    ap.add_argument("--phase3-subset", default="")
    ap.add_argument("--bootstrap-B", type=int, default=200)
    ap.add_argument("--bootstrap-mode", default="nonparametric",
                    choices=["parametric", "nonparametric"])
    ap.add_argument("--noise-mode", default="block", choices=["iid", "ar1", "block"])
    ap.add_argument("--bootstrap-n-calls", type=int, default=40)
    ap.add_argument("--bootstrap-n-initial", type=int, default=20)
    return ap.parse_args(argv)


if __name__ == "__main__":
    main()
