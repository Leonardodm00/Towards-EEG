#!/usr/bin/env python3
"""Was a twin's poor fit the optimiser's? The fit's own loss at the TRUE parameters and at the ESTIMATE.

For a twin of a Stage-7 run (run_twin_recovery.py), this rebuilds exactly the loss its fit minimised -- from the
twin's recorded fit options (fits/twin_<id>/fit_argv.json), with run_ih_fit's own functions -- and evaluates it at
  truth      the manifest's true parameters (twin_manifest.twin_theta), and
  estimate   the fit's result (fits/twin_<id>/phase2_results.csv).
Reading:  L(truth) clearly BELOW L(estimate)  -> the optimiser stopped short of a better point that exists;
          L(estimate) <= L(truth)             -> the optimiser did its job: the miss is in the data (noise, a faint
                                                 I_h, parameters the steps do not separate), not in the search.
Only twins whose fitted model contains the truth are meaningful: the `full` configuration (I_h traces, ih6 arm).

Usage (from the "Ih Fit" folder, conda env active):
    python dev/2026-11-22_tw1_readout/loss_at_truth.py --run-dir <outputs dir of the run> [--twins 900000005,900000008]
Default twins: every `full` twin with status ok. One row per twin is written to <run dir>/loss_at_truth/twin_<id>.csv
as soon as it is computed (a twin already done is skipped unless --force), and a table is printed at the end.
Cost: two loss evaluations per twin (each simulates the training bundles once at the fit's dt).
Pure ASCII, LF. 2026-11-22, the twins chat; a diagnostic, not pipeline code.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.pop("DISPLAY", None)                      # NEURON exits on an unreachable display
HERE = Path(__file__).resolve().parent
CODE = HERE.parent.parent                            # the "Ih Fit" folder
sys.path.insert(0, str(CODE))

COLUMNS = ["twin_id", "configuration", "twin_of_specimen_id", "n_train_bundles", "loss_truth", "loss_estimate",
           "loss_ratio_truth_over_estimate", "train_rel_loss_recorded", "verdict", "seconds"]


def build_loss(fit_argv, sid):
    """(loss, spec, cell holder) exactly as run_ih_fit.main builds them for this twin."""
    import run_ih_fit as R
    import passive_long_step_training as plst
    import passive_fitting_hpc_fixed as mono
    import ih_mechanism as IM
    from neuron import h
    h.load_file("stdrun.hoc")
    for s in list(h.allsec()):
        h.delete_section(sec=s)
    args = R._parse_args(list(fit_argv))
    cfg = R.resolve_arm_config(args)
    tau_w_ms = R._resolve_scalar_tau_w(args.ss_tau_w_ms)
    spec = R.build_param_spec(cfg.fit_params, cm_bounds=R.parse_window(args.cm_bounds),
                              rm_bounds=R.parse_window(args.rm_bounds), ra_bounds=R.parse_window(args.ra_bounds),
                              gbar_bounds=R.parse_window(args.gbar_bounds), dvh_bounds=R.parse_window(args.dvh_bounds),
                              kappa_bounds=R.parse_window(args.kappa_bounds))
    ih_spec = R.build_ih_spec(args) if cfg.attach_ih else None
    plst.integrate_long_step(
        mono, spec=spec, ih_protocol=cfg.ih_protocol, ls_window_mode=cfg.ls_window_mode,
        n_drop_weakest=args.n_drop_weakest, n_drop_strongest=args.n_drop_strongest,
        dep_n_validation=(None if args.dep_n_validation < 0 else int(args.dep_n_validation)),
        v_trough_min_mV=R.parse_optional_float(args.v_trough_min), train_all_hyp=bool(args.train_all_hyp),
        n_long_train=cfg.n_long_train, max_ls_train_deflection_mV=cfg.ls_deflection_cap_mV,
        r_in_target=args.r_in_target, weighting=args.weighting, ss_window_ms=R.parse_window(args.ss_window_ms),
        ss_time_weight=args.ss_time_weight, ss_tau_w_ms=tau_w_ms, ss_t0_ms=R.parse_optional_float(args.ss_t0_ms),
        ls_window_ms_after_onset=float(cfg.ls_window_ms), dt_brief_ms=float(args.dt_brief_ms),
        dt_long_ms=float(args.dt_long_ms), verbose=False)
    cells = mono.load_cells_from_archive(
        args.archive_dir, n_avg_groups=args.n_avg_groups, ls_max_amplitude_pA=cfg.ls_max_amplitude_pA,
        ls_fallback_amplitude_pA=args.ls_fallback_amplitude_pA, load_depolarising_ls=cfg.load_depolarising_ls,
        ls_dep_max_amplitude_pA=R.parse_optional_float(args.ls_dep_max_amplitude_pA), swc_dir=args.swc_dir,
        specimen_ids=None, max_cells=args.max_cells, verbose=False)
    cd = [c for c in cells if int(c.specimen_id) == int(sid)]
    if len(cd) != 1:
        raise RuntimeError("twin %d: %d archive cell(s) found in %s" % (sid, len(cd), args.archive_dir))
    cd = cd[0]
    oi = mono.prepare_optimiser_inputs(cd, fit_target=args.fit_target)
    cell = mono.build_neuron_model(cd.swc_path, F=args.F)
    if ih_spec is not None:
        IM.attach_ih(cell, ih_spec)
    mono.assert_dt_enforced(cell, dt_values_ms=(args.dt_brief_ms, args.dt_long_ms), v_init_mV=float(oi.v_rest_mV),
                            verbose=False)
    loss = mono._build_loss_function(cell, oi.train_bundles, float(oi.v_rest_mV), tuple(oi.train_window_ms), spec=spec)
    return loss, spec, len(oi.train_bundles), cell


def one_twin(run_dir, row, rtol):
    import twin_manifest as TM
    tid = int(row["specimen_id"])
    if str(row["configuration"]) != "full":
        raise RuntimeError("twin %d is %r: only `full` twins (I_h traces, I_h fit) have their truth inside the "
                           "fitted model" % (tid, row["configuration"]))
    fit_dir = Path(run_dir) / "fits" / ("twin_%d" % tid)
    argv = json.loads((fit_dir / "fit_argv.json").read_text())
    argv = argv["argv"] if isinstance(argv, dict) else argv
    res = pd.read_csv(fit_dir / "phase2_results.csv")
    res = res[res["specimen_id"].astype(int) == tid].iloc[0]
    t0 = time.time()
    loss, spec, n_train, _cell = build_loss(argv, tid)
    truth = TM.twin_theta(row)
    missing = [n for n in spec.names if n not in truth]
    if missing:
        raise RuntimeError("twin %d (%s): the truth has no %s -- its fitted model does not contain the truth"
                           % (tid, row["configuration"], missing))
    est = {n: float(res[n]) for n in spec.names}
    l_true = float(loss(*spec.to_q({n: float(truth[n]) for n in spec.names})))
    l_est = float(loss(*spec.to_q(est)))
    ratio = l_true / l_est if l_est > 0 else float("nan")
    if l_true < l_est * (1.0 - rtol):
        verdict = "OPTIMISER: the truth fits better than the estimate"
    elif l_est < l_true * (1.0 - rtol):
        verdict = "DATA: the estimate fits better than the truth"
    else:
        verdict = "TIE within %g" % rtol
    rec = float(res["train_rel_loss"]) if "train_rel_loss" in res.index else float("nan")
    return dict(twin_id=tid, configuration=str(row["configuration"]), twin_of_specimen_id=int(row["twin_of_specimen_id"]),
                n_train_bundles=n_train, loss_truth=l_true, loss_estimate=l_est, loss_ratio_truth_over_estimate=ratio,
                train_rel_loss_recorded=rec, verdict=verdict, seconds=time.time() - t0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", required=True, help="the run's outputs dir (manifest.csv, fits/, twins/)")
    ap.add_argument("--twins", default="", help="comma-separated twin ids; default: every `full` twin with status ok")
    ap.add_argument("--rtol", type=float, default=0.02, help="relative margin under which the two losses are a tie")
    ap.add_argument("--force", action="store_true", help="recompute twins already done")
    a = ap.parse_args(argv)
    run = Path(a.run_dir)
    man = pd.read_csv(run / "manifest.csv")
    if a.twins.strip():
        ids = [int(x) for x in a.twins.split(",") if x.strip()]
    else:
        ids = [int(x) for x in man[man["configuration"] == "full"]["specimen_id"]]
        st = run / "twins_status.csv"
        if st.exists():
            ok = set(pd.read_csv(st).query("status == 'ok'")["twin_id"].astype(int))
            ids = [i for i in ids if i in ok]
    out = run / "loss_at_truth"
    out.mkdir(exist_ok=True)
    rows = []
    for tid in ids:
        f = out / ("twin_%d.csv" % tid)
        if f.exists() and not a.force:
            rows.append(pd.read_csv(f).iloc[0].to_dict())
            print("[loss_at_truth] twin %d: already done" % tid, flush=True)
            continue
        sel = man[man["specimen_id"].astype(int) == tid]
        if len(sel) != 1:
            print("[loss_at_truth] twin %d: not in the manifest" % tid, flush=True)
            continue
        try:
            r = one_twin(run, sel.iloc[0], a.rtol)
        except Exception as exc:                                 # one twin's failure does not stop the others
            print("[loss_at_truth] twin %d: FAILED %s: %s" % (tid, type(exc).__name__, exc), flush=True)
            continue
        pd.DataFrame([r], columns=COLUMNS).to_csv(f, index=False)
        rows.append(r)
        print("[loss_at_truth] twin %d: L(truth) %.5f  L(estimate) %.5f  ratio %.3f  -> %s  (%.0f s)"
              % (tid, r["loss_truth"], r["loss_estimate"], r["loss_ratio_truth_over_estimate"], r["verdict"], r["seconds"]),
              flush=True)
    if rows:
        t = pd.DataFrame(rows, columns=COLUMNS)
        t.to_csv(out / "loss_at_truth.csv", index=False)
        pd.set_option("display.width", 220)
        print("\n" + t[["twin_id", "n_train_bundles", "loss_truth", "loss_estimate", "loss_ratio_truth_over_estimate",
                        "train_rel_loss_recorded", "verdict"]].round(5).to_string(index=False))
        print("LOSS_AT_TRUTH DONE: %d twin(s) -> %s" % (len(t), out / "loss_at_truth.csv"))
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
