# -*- coding: utf-8 -*-
"""
run_ih_recovery.py
==================

Stage 7 (plan section 8): does a six-parameter fit recover a six-parameter
truth on this protocol? Until this has run, a fitted Delta v_h or kappa_tau is
a number the optimiser returned, not a measurement.

Three phases, each doing one thing:

    1. MANIFEST   draw the per-cell ground truth (synth_gt_grid), or load one
                  already drawn. Real archive morphologies; the two kinetic
                  knobs carried as ground truth; a fraction of the cells given
                  NO I_h as the false-positive control.
    2. GENERATE   turn each row into a Phase-0 archive (gen_from_manifest ->
                  synthetic_ground_truth), indistinguishable in shape from a
                  real one.
    3. FIT+REPORT run the campaign's OWN arms over that archive by calling
                  run_ih_fit.main(), then join truth to estimates
                  (ih_recovery_report) and apply the gate.

Why phase 3 calls run_ih_fit rather than fitting here
------------------------------------------------------
A recovery gate is only worth running on the program that will run the
campaign. Re-implementing the fit inside this script would validate a
different program from the one Stage 8 launches, and every later divergence
between them would be invisible. So the arms here are the arms there --
`--arm baseline_runB | passive_fullstep | ih6`, and the 4-D arm is `ih6` with
`--fit-params Cm,Rm,Ra,gbar` -- and each writes the same
`phase2_results.csv` / `ls_roles.csv` / `ls_diagnostics.csv` it writes in
production.

Consistency between the truth and the fit
-----------------------------------------
The mechanism, its reversal potential, its spatial law, its regions and the
configuration shift Delta v_base are CONFIGURED, not fitted (D-005). If the
ground truth uses one setting and the fit another, this script measures
misspecification, not recovery -- a different and much harder question, and
not the one the gate asks. `--allow-misspecification` is the labelled opt-in;
without it, a disagreement is refused before anything is generated.

Cost
----
n_cells x n_arms fits at the campaign's 200/100 budget. The budget is NOT
reduced here: a recovery failure at 60 calls would say nothing about a
campaign at 200. Use `--max-cells` for a shakedown, not a smaller budget.
"""

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


_T0 = time.time()


def log(msg: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    el = time.time() - _T0
    print("[{} | +{:02d}:{:02d}] {}".format(now, int(el // 60), int(el % 60),
                                            msg), flush=True)


# ===========================================================================
#  The arms of the recovery test (PURE: a table, not a computation)
# ===========================================================================
#: name -> (run_ih_fit --arm, --fit-params override or None)
#: The four fits of plan section 8, Stage 7, in the order they are reported.
RECOVERY_ARMS: Dict[str, Tuple[str, Optional[str]]] = {
    "baseline_runB":    ("baseline_runB", None),
    "passive_fullstep": ("passive_fullstep", None),
    "ih4":              ("ih6", "Cm,Rm,Ra,gbar"),
    "ih6":              ("ih6", None),
}


def parse_arms(text: str) -> List[str]:
    """'ih6,passive_fullstep' -> ['ih6', 'passive_fullstep'], validated by
    name so a typo is refused rather than silently dropping an arm."""
    names = [t.strip() for t in str(text).split(",") if t.strip()]
    bad = [n for n in names if n not in RECOVERY_ARMS]
    if bad:
        raise SystemExit("[FATAL] unknown arm(s) {}; choose from {}"
                         .format(bad, sorted(RECOVERY_ARMS)))
    if not names:
        raise SystemExit("[FATAL] --arms is empty")
    return names


def check_consistency(manifest: pd.DataFrame, args) -> List[str]:
    """PURE. Every way the ground truth and the fit can be set up to answer
    different questions. Returns a list of human-readable disagreements; an
    empty list means the fit is looking for what the generator put there.
    """
    out: List[str] = []
    if manifest.empty:
        return ["manifest is empty"]
    ih = manifest[manifest["use_ih"].astype(bool)]
    if ih.empty:
        return out
    mech = sorted({str(v).strip() or "Ih" for v in ih["ih_kinetics"]})
    if mech != [str(args.ih_mechanism)]:
        out.append("mechanism: truth {} vs fit {!r}".format(mech, args.ih_mechanism))
    dist = sorted({str(v).strip() for v in ih["ih_dist"]})
    fit_dist = ("uniform" if args.ih_distribution == "uniform"
                else args.ih_distribution)
    if dist != [fit_dist]:
        out.append("spatial law: truth {} vs fit {!r}".format(dist, fit_dist))
    reg = sorted({str(v).strip() for v in ih["ih_regions"] if str(v).strip()})
    fit_reg = ",".join(r.strip() for r in args.ih_regions.split(",") if r.strip())
    if reg and reg != [fit_reg]:
        out.append("regions: truth {} vs fit {!r}".format(reg, fit_reg))
    vsb = sorted({round(float(v), 6) for v in ih["ih_vshift_base_mV"]
                  if np.isfinite(float(v))})
    if vsb and vsb != [round(float(args.vshift_base), 6)]:
        out.append("vshift_base: truth {} mV vs fit {} mV"
                   .format(vsb, args.vshift_base))
    ehcn = sorted({round(float(v), 4) for v in ih["ih_ehcn_mV"]
                   if np.isfinite(float(v))})
    if args.ehcn not in (None, ""):
        fit_ehcn = [round(float(args.ehcn), 4)]
        if ehcn != fit_ehcn:
            out.append("E_h: truth {} mV vs fit {} mV".format(ehcn, fit_ehcn))
    elif len(ehcn) == 1:
        # the fit will use the mechanism's own default; name it so a
        # mismatch is visible in the log even when it is not refused
        out.append("[note] E_h of the truth is {} mV; the fit takes the "
                   "mechanism default for {} (pass --ehcn to pin it)"
                   .format(ehcn[0], args.ih_mechanism))
    return out


def _is_note(line: str) -> bool:
    return str(line).startswith("[note]")


# ===========================================================================
#  Phases
# ===========================================================================
def phase_noise(args, out: Path) -> Tuple[Dict[int, Dict[str, float]],
                                            Dict[str, float]]:
    """Measure the per-sweep recording noise of the REAL cells whose
    morphologies the cohort is drawn on (noise_calibration). Writes
    noise_table.csv and returns (per-specimen lookup, cohort summary).

    The morphology root IS a real archive group -- `specimen_<id>/` holds the
    SWC and the sweeps side by side -- so the morphologies and the noise come
    from the same cells, and each synthetic cell inherits the noise of the
    real cell whose arbour it borrows. Nothing to configure.

    Returns ({}, {}) when there is nothing to measure (a manifest reused
    without a morph root, --no-measure-noise, or morphologies that are not
    archive cells, as in the smoke suite), and the manifest then falls back
    to the nominal level with a loud label.
    """
    import synth_gt_grid as G
    import noise_calibration as NC
    if args.no_measure_noise or not args.morph_root:
        log("[0/3] NOISE -- not measured (%s)"
            % ("--no-measure-noise" if args.no_measure_noise else "no --morph-root"))
        return {}, {}
    import passive_fitting_hpc_fixed as mono
    swcs = G._resolve_swcs(args.morph_root, args.morph_glob)
    dirs = []
    for sw in swcs:
        if NC.specimen_id_of(sw) is not None and sw.parent not in dirs:
            dirs.append(sw.parent)
    if args.max_cells is not None:
        dirs = dirs[:max(1, int(args.max_cells))]
    if not dirs:
        log("[0/3] NOISE -- no specimen_<id>/ directories under {}; the "
            "morphologies are not archive cells, so there is nothing to "
            "measure".format(args.morph_root))
        return {}, {}
    log("[0/3] NOISE -- per-sweep noise of {} real cell(s), read from their "
        "single sweeps with the fitter's own estimator".format(len(dirs)))
    rows = [NC.measure_specimen_noise(d, mono=mono) for d in dirs]
    table = pd.DataFrame(rows, columns=NC.NOISE_TABLE_COLUMNS)
    table.to_csv(out / "noise_table.csv", index=False)
    summ = NC.summarise_noise_table(table)
    lookup = NC.noise_lookup(table, protocol=args.noise_protocol)
    lookup, warns = apply_cli_acquisition(lookup, args)
    for w in warns:
        log("[0/3] NOISE -- WARNING: " + w)
    log("[0/3] NOISE -- {} usable of {}: LS sigma {:.4f} mV rho {:.3f} @ {:.0f} Hz"
        " | SS sigma {:.4f} mV rho {:.3f} @ {:.0f} Hz | SS pulses/polarity {:.0f}"
        " | LS rho > 0.5 in {:.0f}% of cells -> noise_table.csv"
        .format(len(lookup), len(table), summ["sigma_ls_mV"], summ["rho_ls"],
                summ["fs_ls_Hz"], summ["sigma_ss_mV"], summ["rho_ss"],
                summ["fs_ss_Hz"], summ["n_ss_per_polarity"],
                100.0 * summ["frac_rho_ls_gt_0p5"]))
    # AR(1) adequacy: measured autocorrelation at fixed physical lags beside
    # what AR(1) implies there. measured >> AR(1) at 1 ms = slow correlated
    # power the generator does not reproduce (synthetic data then more
    # informative than real). Reported, never acted on.
    for p in ("ls", "ss"):
        log("[0/3] NOISE -- AR(1) adequacy {}: ".format(p.upper()) + " | ".join(
            "lag {}: measured {:.3f} vs AR(1) {:.3f}".format(
                NC._lag_tag(l), summ["acf_%s_%s" % (p, NC._lag_tag(l))],
                summ["ar1_%s_%s" % (p, NC._lag_tag(l))])
            for l in NC.ACF_LAGS_MS))
    log("[0/3] NOISE -- mode {}: {} cell(s) carry their SS pulses' own noise"
        .format(args.noise_protocol,
                sum(1 for v in lookup.values() if "ss_sigma_mV" in v)))
    return lookup, summ


def apply_cli_acquisition(lookup: Dict[int, Dict[str, float]], args
                          ) -> Tuple[Dict[int, Dict[str, float]], List[str]]:
    """PURE. Keep the precedence command line > measured > default at the
    level of each cell: a protocol constant given on the command line removes
    the corresponding per-cell acquisition key, so the cohort value (the
    command line's) applies to every cell. Returns (new lookup, warnings).

    Forcing a SAMPLING RATE that differs from the one a cell's rho was
    measured at re-creates the transplant D-014 removes -- rho is per
    sample -- so that case is allowed (it is an explicit instruction) and
    reported, cell count and rates included."""
    cli = {"ss_n_repeats": getattr(args, "ss_n_repeats", None),
           "fs_ss_Hz": getattr(args, "ss_sampling_rate_hz", None),
           "fs_ls_Hz": getattr(args, "ls_sampling_rate_hz", None)}
    out: Dict[int, Dict[str, float]] = {}
    clash = {"fs_ss_Hz": 0, "fs_ls_Hz": 0}
    for sid, e in lookup.items():
        e2 = dict(e)
        for key, val in cli.items():
            if val is None or key not in e2:
                continue
            if key in clash and abs(float(e2[key]) - float(val)) > 1e-6:
                clash[key] += 1
            e2.pop(key)
        out[sid] = e2
    warns = ["--%s %g forced on %d cell(s) whose rho was measured at another "
             "rate: their noise is then NOT the recorded noise (rho is per "
             "sample)" % ({"fs_ss_Hz": "ss-sampling-rate-hz",
                            "fs_ls_Hz": "ls-sampling-rate-hz"}[k],
                           float(cli[k]), n)
             for k, n in clash.items() if n]
    return out, warns


def resolve_protocol(args, summ: Dict[str, float]) -> Dict[str, object]:
    """PURE. The COHORT protocol's constants, each with where it came from:
    the command line first, then the measured cohort, then the generator's
    own default. Since D-014 these are the FALLBACK: a synthetic cell whose
    real twin was measured is generated at that twin's own sampling rates and
    SS pulse count (manifest columns acq_*), and only a cell without one uses
    these. The sampling rates matter because a lag-1 correlation read
    at one rate is a different noise at another; the SS repeat count because
    it sets how much the SS bundle's noise is averaged down."""
    out: Dict[str, object] = {}

    def pick(name, cli, measured, default, cast):
        if cli is not None:
            out[name], out[name + "_source"] = cast(cli), "cli"
        elif measured is not None and np.isfinite(measured) and measured > 0:
            out[name], out[name + "_source"] = cast(measured), "measured"
        else:
            out[name], out[name + "_source"] = default, "generator default"

    pick("ss_n_repeats", args.ss_n_repeats, summ.get("n_ss_per_polarity"),
         30, lambda v: max(1, int(round(float(v)))))
    pick("ss_sampling_rate_Hz", args.ss_sampling_rate_hz,
         summ.get("fs_ss_Hz"), None, float)
    pick("ls_sampling_rate_Hz", args.ls_sampling_rate_hz,
         summ.get("fs_ls_Hz"), None, float)
    return out


def phase_manifest(args, out: Path,
                   noise_lookup: Optional[Dict[int, Dict[str, float]]] = None
                   ) -> pd.DataFrame:
    """Draw the cohort, or load one already drawn. Writes manifest.csv."""
    import synth_gt_grid as G
    dest = out / "manifest.csv"
    if args.manifest:
        log("[1/3] MANIFEST -- loading {}".format(args.manifest))
        df = G.load_manifest(args.manifest)
        G.save_manifest(df, dest)
        return G.load_manifest(dest)

    swcs = G._resolve_swcs(args.morph_root, args.morph_glob)
    log("[1/3] MANIFEST -- {} morphology/ies under {}"
        .format(len(swcs), args.morph_root))

    sigma = float(args.noise_sigma)
    baseline = float(args.noise_baseline)
    drift = float(args.noise_drift)
    if not noise_lookup:
        log("[1/3] MANIFEST -- WARNING: noise is NOMINAL (sigma {:.4f} mV per "
            "sweep, rho {:.2f}), NOT MEASURED. A gate that passes on cleaner "
            "noise than the bench says nothing about the campaign."
            .format(sigma, args.noise_rho))

    df = G.draw_manifest(
        swcs, draws_per_morph=args.draws_per_morph, seed=args.seed,
        cells_per_cohort=max(1, args.cells_per_cohort), ra_mode=args.ra_mode,
        e_pas_mV=args.e_pas, F=args.F, max_cells=args.max_cells,
        cm_phys_lo=args.cm_phys_lo, cm_phys_hi=args.cm_phys_hi,
        tau_lo_ms=args.tau_lo_ms, tau_hi_ms=args.tau_hi_ms,
        use_ih=True,
        ih_kinetics=args.ih_mechanism, ih_dist=args.ih_distribution,
        ih_ehcn_mV=_ehcn_for_truth(args),
        ih_regions=tuple(r.strip() for r in args.ih_regions.split(",") if r.strip()),
        ih_vshift_base_mV=args.vshift_base,
        ih_gbar_range_S_cm2=G._pair_or_none(args.gbar_range, "--gbar-range"),
        ih_dvh_range_mV=G._pair_or_none(args.dvh_range, "--dvh-range"),
        ih_kappa_range=G._pair_or_none(args.kappa_range, "--kappa-range"),
        fp_control_frac=args.fp_control_frac,
        ih_gbar_floor_S_cm2=args.gbar_floor,
        ra_phys_lo=args.ra_phys_lo, ra_phys_hi=args.ra_phys_hi,
        noise_sigma_nominal_mV=sigma, noise_baseline_nominal_mV=baseline,
        noise_drift_nominal_mV=drift, noise_cv=args.noise_cv,
        noise_table=noise_lookup, noise_rho_nominal=args.noise_rho,
        id_base=args.id_base)
    G.save_manifest(df, dest)
    # Re-read rather than return the in-memory draw. A CSV is decimal text, so
    # a float survives the round trip to about one ulp and not exactly; if
    # generation used the in-memory values while a later `--manifest` re-run
    # used the file, the two runs would be generating from ground truths that
    # differ in the last bit. The file is the record, so the file is what
    # every phase reads -- including this one.
    df = G.load_manifest(dest)
    n_fp = int(df["is_fp_control"].sum())
    n_meas = int((df["noise_source"] == "measured").sum())
    log("[1/3] MANIFEST -- {} cell(s), {} with I_h, {} false-positive "
        "control(s); noise MEASURED for {} of them -> {}"
        .format(len(df), len(df) - n_fp, n_fp, n_meas, dest))
    return df


def _ehcn_for_truth(args) -> float:
    """E_h injected into the ground truth: whatever the fit will use, so the
    two agree by construction unless the user overrides one of them."""
    if args.ehcn not in (None, ""):
        return float(args.ehcn)
    import human_ih_params as hip
    return float(hip.ehcn_default_mV(args.ih_mechanism))


def phase_generate(args, manifest: pd.DataFrame, archive: Path,
                   protocol: Optional[Dict[str, object]] = None) -> pd.DataFrame:
    """Write one Phase-0 archive per manifest row (NEURON-side)."""
    import gen_from_manifest as GM
    import synthetic_ground_truth as sgt
    import passive_fitting_hpc_fixed as mono
    from neuron import h

    def _clear():
        for s in list(h.allsec()):
            h.delete_section(sec=s)

    hyp = tuple(float(a) for a in str(args.ls_hyp_amps).split(",") if a.strip())
    dep = tuple(float(a) for a in str(args.ls_dep_amps).split(",") if a.strip())
    proto = dict(protocol or resolve_protocol(args, {}))
    n_twin = int(np.isfinite(pd.to_numeric(
        manifest.get("acq_fs_ls_Hz", pd.Series(dtype=float)),
        errors="coerce")).sum())
    n_ss_own = int((manifest.get("noise_ss_source", pd.Series(dtype=str))
                    == "measured").sum())
    log("[2/3] GENERATE -- {} of {} cell(s) at their real twin's own sampling "
        "rates and SS pulse count; {} with the SS pulses' own measured noise"
        .format(n_twin, len(manifest), n_ss_own))
    log("[2/3] GENERATE -- {} cell(s); LS hyp {} pA, LS dep {} pA; cohort "
        "fallback: SS x{} ({}), SS fs {} ({}), LS fs {} ({})".format(
            len(manifest), list(hyp), list(dep),
            proto["ss_n_repeats"], proto["ss_n_repeats_source"],
            proto["ss_sampling_rate_Hz"], proto["ss_sampling_rate_Hz_source"],
            proto["ls_sampling_rate_Hz"], proto["ls_sampling_rate_Hz_source"]))
    meta = GM.generate_group(manifest, archive, sgt=sgt, mono=mono,
                             ss_n_repeats=int(proto["ss_n_repeats"]),
                             ls_hyp_amplitudes_pA=hyp,
                             ls_dep_amplitudes_pA=dep,
                             ss_sampling_rate_Hz=proto["ss_sampling_rate_Hz"],
                             ls_sampling_rate_Hz=proto["ls_sampling_rate_Hz"],
                             clear_fn=_clear, verbose=args.verbose_generate)
    n_ok = int(meta["ok"].sum()) if len(meta) else 0
    if n_ok == 0:
        raise SystemExit("[FATAL] no archive was generated; nothing to fit.")
    log("[2/3] GENERATE -- {}/{} archive(s) written to {}"
        .format(n_ok, len(meta), archive))
    return meta


def phase_fit(args, archive: Path, out: Path, arms: Sequence[str]
              ) -> Dict[str, Path]:
    """Run the campaign's own entrypoint once per arm. Returns arm -> CSV."""
    import run_ih_fit as R
    produced: Dict[str, Path] = {}
    for i, name in enumerate(arms):
        arm_flag, fit_params = RECOVERY_ARMS[name]
        arm_out = out / "arms" / name
        csv = arm_out / "phase2_results.csv"
        if csv.exists() and not args.overwrite:
            log("[3/3] FIT -- arm {} ({}/{}): results exist, reusing {}"
                .format(name, i + 1, len(arms), csv))
            produced[name] = csv
            continue
        argv = ["--archive-dir", str(archive), "--output-dir", str(arm_out),
                "--code-dir", str(args.code_dir), "--arm", arm_flag,
                "--F", str(args.F), "--fit-target", args.fit_target,
                "--ss-time-weight", args.ss_time_weight,
                "--ss-tau-w-ms", str(args.ss_tau_w_ms),
                "--ss-window-ms", str(args.ss_window_ms),
                "--n-calls", str(args.n_calls),
                "--n-initial", str(args.n_initial),
                "--ih-mechanism", args.ih_mechanism,
                "--ih-distribution", args.ih_distribution,
                "--ih-regions", args.ih_regions,
                # '=' form throughout for anything that can be negative:
                # argparse reads a bare '-20.0' as an option name.
                "--vshift-base=%s" % args.vshift_base,
                "--dt-brief-ms", str(args.dt_brief_ms),
                "--dt-long-ms", str(args.dt_long_ms),
                "--phase3-subset", args.phase3_subset,
                "--gbar-bounds=%s" % args.gbar_bounds,
                "--dvh-bounds=%s" % args.dvh_bounds,
                "--kappa-bounds=%s" % args.kappa_bounds]
        if fit_params:
            argv += ["--fit-params", fit_params]
        if args.ehcn not in (None, ""):
            argv += ["--ehcn=%s" % args.ehcn]
        log("[3/3] FIT -- arm {} ({}/{}) -> {}"
            .format(name, i + 1, len(arms), arm_out))
        t0 = time.time()
        try:
            R.main(argv)
        except SystemExit as exc:          # run_ih_fit exits 1 on no fits
            log("[3/3] FIT -- arm {} exited: {}".format(name, exc))
        except Exception as exc:           # noqa: BLE001
            log("[3/3] FIT -- arm {} FAILED: {}: {}"
                .format(name, type(exc).__name__, exc))
            if args.fail_fast:
                raise
        log("[3/3] FIT -- arm {} done in {:.0f}s".format(name, time.time() - t0))
        if csv.exists():
            produced[name] = csv
        else:
            log("[3/3] FIT -- arm {}: no phase2_results.csv; excluded from "
                "the report.".format(name))
    return produced


def phase_report(args, manifest: pd.DataFrame, produced: Dict[str, Path],
                 out: Path) -> bool:
    """Join truth to estimates, apply the gate, write and print the report."""
    import ih_recovery_report as RR
    if not produced:
        raise SystemExit("[FATAL] no arm produced results; nothing to report.")
    frames = [RR.load_arm_results(p, name) for name, p in produced.items()]
    box = dict(RR.DEFAULT_BOX)
    for axis, text in (("gbar", args.gbar_bounds), ("dv_h", args.dvh_bounds),
                       ("kappa_tau", args.kappa_bounds)):
        lo, hi = (float(x) for x in str(text).split(","))
        box[axis] = (lo, hi)
    long_df = RR.recovery_long(manifest, frames, box=box,
                               strict_rails=not args.lenient_rails)
    summary = RR.recovery_summary(long_df, drop_railed=args.drop_railed)
    inflation = RR.cm_inflation(long_df)
    fp = RR.fp_control_summary(long_df)
    tol = RR.RecoveryTolerances(
        log_ratio_cm=float(np.log(args.tol_cm_factor)),
        log_ratio_gbar=float(np.log(args.tol_gbar_factor)),
        abs_dvh_mV=float(args.tol_dvh_mV),
        log_ratio_kappa=float(np.log(args.tol_kappa_factor)),
        fp_min_low_rail_frac=float(args.tol_fp_low_rail_frac),
        fp_log_ratio_cm=float(np.log(args.tol_fp_cm_factor)))
    verdict, passed = RR.evaluate_gate(summary, fp, arm=args.gate_arm, tol=tol)
    RR.write_report(out, long_df=long_df, summary=summary, inflation=inflation,
                    fp=fp, verdict=verdict, passed=passed, tol=tol)
    print(RR.format_report(summary, inflation, fp, verdict, passed), flush=True)
    if not args.no_plot:
        try:
            p = RR.plot_recovery(long_df, out / "recovery.png", tol=tol)
            log("recovery figure -> {}".format(p))
        except Exception as exc:           # noqa: BLE001 -- a figure is not the result
            log("recovery figure skipped: {}: {}".format(type(exc).__name__, exc))
    return passed


# ===========================================================================
#  main
# ===========================================================================
def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    sys.path.insert(0, args.code_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    archive = Path(args.archive_dir) if args.archive_dir else out / "archive"
    arms = parse_arms(args.arms)

    log("STAGE 7 -- synthetic recovery | arms {} | budget {}/{} | out {}"
        .format(arms, args.n_calls, args.n_initial, out))

    noise_lookup, noise_summary = phase_noise(args, out)
    protocol = resolve_protocol(args, noise_summary)
    manifest = phase_manifest(args, out, noise_lookup)

    problems = check_consistency(manifest, args)
    notes = [p for p in problems if _is_note(p)]
    hard = [p for p in problems if not _is_note(p)]
    for p in notes:
        log("consistency " + p)
    if hard:
        msg = ("the ground truth and the fit are configured differently, so "
               "this would measure MISSPECIFICATION, not recovery:\n  "
               + "\n  ".join(hard))
        if not args.allow_misspecification:
            raise SystemExit("[FATAL] " + msg +
                             "\nFix the settings, or pass "
                             "--allow-misspecification to measure it on "
                             "purpose (the result is then labelled as such).")
        log("[WARN] MISSPECIFICATION ON PURPOSE: " + msg)
    (out / "run_config.json").write_text(json.dumps(
        {**vars(args), "arms": arms, "misspecification": hard,
         "protocol": protocol, "noise_summary": noise_summary}, indent=2,
        default=str))

    if not args.skip_generate:
        phase_generate(args, manifest, archive, protocol)
    else:
        log("[2/3] GENERATE -- skipped (--skip-generate); using {}"
            .format(archive))

    produced = phase_fit(args, archive, out, arms)
    passed = phase_report(args, manifest, produced, out)

    if args.clean_archive and not args.skip_generate:
        shutil.rmtree(archive, ignore_errors=True)
        log("archive removed (--clean-archive)")
    log("STAGE 7 -- done in {:.0f}s; gate {}"
        .format(time.time() - _T0, "PASS" if passed else "FAIL"))
    return 0 if passed else 2


def _parse_args(argv):
    ap = argparse.ArgumentParser(
        description="Stage 7: synthetic six-parameter recovery + the gate.")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--code-dir", required=True)
    ap.add_argument("--archive-dir", default=None,
                    help="where the generated archives go (default: "
                         "<output-dir>/archive).")
    # --- cohort ------------------------------------------------------------
    ap.add_argument("--manifest", default=None,
                    help="use an existing manifest instead of drawing one.")
    ap.add_argument("--morph-root", default=None,
                    help="root of the REAL morphologies, e.g. "
                         "<ARCHIVE_ROOT>/L3_exc. Required unless --manifest.")
    ap.add_argument("--morph-glob", default="specimen_*/reconstruction.swc")
    ap.add_argument("--draws-per-morph", type=int, default=1)
    ap.add_argument("--cells-per-cohort", type=int, default=10)
    ap.add_argument("--max-cells", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--id-base", type=int, default=900_000_000)
    ap.add_argument("--ra-mode", default="per_cohort",
                    choices=["per_cohort", "per_cell"])
    ap.add_argument("--e-pas", type=float, default=-73.5)
    ap.add_argument("--cm-phys-lo", type=float, default=0.4)
    ap.add_argument("--cm-phys-hi", type=float, default=1.5)
    ap.add_argument("--tau-lo-ms", type=float, default=3.0)
    ap.add_argument("--tau-hi-ms", type=float, default=40.0)
    ap.add_argument("--ra-phys-lo", type=float, default=100.0)
    ap.add_argument("--ra-phys-hi", type=float, default=500.0)
    # --- the I_h ground truth ----------------------------------------------
    ap.add_argument("--gbar-range", default="2e-5,3e-4",
                    help="'lo,hi' S/cm^2, log-uniform. The default spans the "
                         "human anchors: Rich 5.14e-5, Kalmbach 1e-4, "
                         "Hay 2e-4 (plan section 6.3).")
    ap.add_argument("--dvh-range", default="-5,5",
                    help="'lo,hi' mV, UNIFORM. Inside the fit box [-10,10] so "
                         "the truth is not on a bound.")
    ap.add_argument("--kappa-range", default="0.7,1.4",
                    help="'lo,hi', log-uniform. Inside the fit box [0.5,2].")
    ap.add_argument("--fp-control-frac", type=float, default=0.25,
                    help="fraction of cells with NO I_h (gbar at the floor): "
                         "the false-positive control. 0 disables it.")
    ap.add_argument("--gbar-floor", type=float, default=1e-6)
    # --- the protocol generated --------------------------------------------
    ap.add_argument("--ss-n-repeats", type=int, default=None,
                    help="SS pulses averaged per polarity in the GENERATED "
                         "bundle. Default: the real cells' median (measured), "
                         "else the generator's 30.")
    ap.add_argument("--ss-sampling-rate-hz", type=float, default=None,
                    help="default: the real cells' median, else 50 kHz")
    ap.add_argument("--ls-sampling-rate-hz", type=float, default=None,
                    help="default: the real cells' median, else 20 kHz")
    ap.add_argument("--ls-hyp-amps", default="-10,-30,-50,-70,-90,-110,-150",
                    help="hyperpolarising Long Square amplitudes to GENERATE. "
                         "Must be long enough for the D-006 role split to have "
                         "a validate/train/report partition.")
    ap.add_argument("--ls-dep-amps", default="20,50",
                    help="spike-free depolarising steps: the D-006 validation "
                         "set. An empty list leaves the I_h arms with nothing "
                         "held out.")
    ap.add_argument("--no-measure-noise", action="store_true",
                    help="do NOT read the real cells' per-sweep noise from "
                         "--morph-root; generate at the nominal level instead "
                         "(labelled 'nominal' in the manifest).")
    ap.add_argument("--noise-protocol", default="per_protocol",
                    choices=["per_protocol", "ls", "ss"],
                    help="per_protocol (default, D-014): LS sweeps carry the "
                         "real cell's LS measurement and SS pulses its SS "
                         "measurement. ls / ss: one measurement for both "
                         "(D-012's form). In every mode each synthetic cell "
                         "is generated at its real twin's sampling rates and "
                         "SS pulse count.")
    ap.add_argument("--noise-rho", type=float, default=0.0,
                    help="nominal AR(1) coefficient for a cell with no "
                         "measured noise.")
    ap.add_argument("--noise-sigma", type=float, default=0.05)
    ap.add_argument("--noise-baseline", type=float, default=0.05)
    ap.add_argument("--noise-drift", type=float, default=0.10)
    ap.add_argument("--noise-cv", type=float, default=0.3)
    ap.add_argument("--verbose-generate", action="store_true")
    # --- the fit (must match run_ih_fit's own defaults) ---------------------
    ap.add_argument("--arms", default="baseline_runB,passive_fullstep,ih4,ih6")
    ap.add_argument("--fit-target", default="hyp", choices=["dep", "hyp", "both"])
    # The SS pulses' exponential time-weight in the LOSS, forwarded to
    # run_ih_fit explicitly (its defaults, and submit_ih_fit.sh's) so the gate
    # is run on the loss the campaign runs, and a change to one is visible in
    # the other's run_config.json rather than silently inherited.
    ap.add_argument("--ss-time-weight", default="exp",
                    choices=["exp", "gauss", "none"])
    ap.add_argument("--ss-tau-w-ms", default="5.0")
    ap.add_argument("--ss-window-ms", default="0.5,100.0")
    ap.add_argument("--F", type=float, default=1.9)
    ap.add_argument("--n-calls", type=int, default=200)
    ap.add_argument("--n-initial", type=int, default=100)
    ap.add_argument("--ih-mechanism", default="Ih_human", choices=["Ih_human", "Ih"])
    ap.add_argument("--ih-distribution", default="uniform",
                    choices=["uniform", "eyal_exp_323", "hay_exp_dmax"])
    ap.add_argument("--ih-regions", default="soma,dend,apic")
    ap.add_argument("--vshift-base", type=float, default=0.0)
    ap.add_argument("--ehcn", default=None)
    ap.add_argument("--gbar-bounds", default="1e-6,1e-3")
    ap.add_argument("--dvh-bounds", default="-10.0,10.0")
    ap.add_argument("--kappa-bounds", default="0.5,2.0")
    ap.add_argument("--dt-brief-ms", type=float, default=0.1)
    ap.add_argument("--dt-long-ms", type=float, default=0.1)
    ap.add_argument("--phase3-subset", default="none",
                    help="Phase 3 is OFF here by default: the recovery gate "
                         "is about the point estimate, and bootstrapping "
                         "every synthetic cell costs more than the cohort.")
    ap.add_argument("--skip-generate", action="store_true")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-run an arm whose phase2_results.csv exists.")
    ap.add_argument("--clean-archive", action="store_true")
    ap.add_argument("--fail-fast", action="store_true")
    ap.add_argument("--allow-misspecification", action="store_true")
    # --- the gate ----------------------------------------------------------
    ap.add_argument("--gate-arm", default="ih6")
    ap.add_argument("--tol-cm-factor", type=float, default=1.25)
    ap.add_argument("--tol-gbar-factor", type=float, default=1.25)
    ap.add_argument("--tol-dvh-mV", type=float, default=3.0)
    ap.add_argument("--tol-kappa-factor", type=float, default=1.5)
    ap.add_argument("--tol-fp-low-rail-frac", type=float, default=0.8)
    ap.add_argument("--tol-fp-cm-factor", type=float, default=1.25)
    ap.add_argument("--drop-railed", action="store_true",
                    help="exclude railed axes from the recovery medians. OFF "
                         "by default: dropping the failures is how a gate "
                         "passes. The rail COUNTS are always reported.")
    ap.add_argument("--lenient-rails", action="store_true",
                    help="warn instead of refusing when this report's box "
                         "disagrees with the one the fit used.")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args(argv)
    if not args.manifest and not args.morph_root:
        raise SystemExit("[FATAL] give --morph-root (to draw a cohort) or "
                         "--manifest (to reuse one).")
    # skopt refuses n_calls < n_initial_points; caught HERE, before any cell
    # is generated, because inside the per-cell fit loop the ValueError would
    # be logged and swallowed cell by cell and the run would end with no
    # fits and no verdict.
    if args.n_initial > args.n_calls:
        ap.error("--n-initial (%d) exceeds --n-calls (%d): gp_minimize refuses "
                 "this. For a shakedown lower both, e.g. N_CALLS=20,N_INITIAL=10."
                 % (args.n_initial, args.n_calls))
    return args


if __name__ == "__main__":
    sys.exit(main())
