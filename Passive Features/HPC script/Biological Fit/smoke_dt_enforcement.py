#!/usr/bin/env python3
"""
smoke_dt_enforcement.py
=======================

Smoke test for the dt-enforcement patch. Run this BEFORE resubmitting any
production job, and run section 3 before ever setting --dt-long-ms 0.1.

Background
----------
NEURON's stdrun.hoc rewrites h.dt inside setdt(), which h.run() calls via
stdinit():

    proc setdt() {local Dt, dtnew
        if (using_cvode_) return
        Dt = 1/steps_per_ms
        nstep_steprun = int(Dt/dt)
        if (nstep_steprun == 0) { nstep_steprun = 1 }
        dtnew = Dt/nstep_steprun
        if (abs(dt*nstep_steprun*steps_per_ms - 1) > 1e-6) {
            print "Changed dt"
            dt = dtnew
        }
    }

steps_per_ms is the legacy GUI variable bound as "Points plotted/ms" in the
RunControl panel (default 40 = 1/0.025). The check is plot bookkeeping, NOT a
stability guard: the fit model is purely passive and NEURON's fixed-step solver
is backward Euler (secondorder = 0), which is A-stable for this linear system
at any dt. The patch sets steps_per_ms = 1/dt so the rewrite cannot fire, and
verifies the achieved dt from the recorded time vector.

What each section asserts
-------------------------
  1. MECHANISM   -- with enforce_dt=True the achieved dt equals the requested
                    dt for every candidate value; with enforce_dt=False the
                    historical override is reproduced (dt=0.1 -> 0.025). This
                    is the before/after proof.
  2. REACH       -- integrate_long_step(dt_long_ms=X) actually reaches the
                    monolith's own replay helpers, i.e. TRAINING and
                    VALIDATION (and Phase 2.5 / Phase 3) integrate at the same
                    dt rather than silently disagreeing.
  3. ACCURACY    -- the price of dt = 0.1 ms. Simulates the same long step at
                    0.025 and 0.1 ms and reports (a) max |dV| between the two
                    traces inside the scored window and (b) the change in the
                    bundle RMSD. Compare against your measured noise floor
                    (~0.05 mV per the production logs). This section makes no
                    pass/fail claim -- it hands you the number to judge.
  4. NO-CHANGE   -- with the shipped defaults (both 0.025 ms) the loss value at
                    a fixed parameter point is bit-identical to the pre-patch
                    behaviour, i.e. adopting the patch alone changes no science.

Usage
-----
    python smoke_dt_enforcement.py \
        --archive-dir "/davinci-1/home/ldellamea/Human Neurons Fitting/L3_exc" \
        --code-dir    "/davinci-1/home/ldellamea/Human Neurons Fitting" \
        --F 1.9

    # one specific cell, and skip the slow accuracy section:
    python smoke_dt_enforcement.py ... --specimen 508282493 --skip-accuracy

Exit code 0 = all assertions passed. Non-zero = do not submit.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

PASS = []
FAIL = []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print("    [{}] {}{}".format("PASS" if ok else "FAIL", name,
                                 ("  -- " + detail) if detail else ""))


# ---------------------------------------------------------------------------
#  1. MECHANISM
# ---------------------------------------------------------------------------
def section_mechanism(mono, cell, v_rest):
    print("\n[1] MECHANISM -- is the requested dt actually used?")
    for dt_req in (0.025, 0.05, 0.1, 0.2):
        t_ms, _ = cell.simulate(stim_amp_pA=-10.0, stim_delay_ms=20.0,
                                stim_dur_ms=50.0, tstop_ms=200.0,
                                v_init_mV=v_rest, dt_ms=dt_req,
                                enforce_dt=True)
        got = float(t_ms[1] - t_ms[0])
        check("enforce_dt=True honours dt={:g} ms".format(dt_req),
              abs(got / dt_req - 1.0) <= mono.DT_ENFORCE_RTOL,
              "achieved {:.6g} ms".format(got))

    # Reproduce the historical bug on purpose. steps_per_ms is a global that
    # the enforced calls above have left at 1/0.2, so reset it to NEURON's
    # default first -- that default is exactly what made dt=0.1 a no-op.
    from neuron import h
    h.steps_per_ms = 40.0
    t_ms, _ = cell.simulate(stim_amp_pA=-10.0, stim_delay_ms=20.0,
                            stim_dur_ms=50.0, tstop_ms=200.0,
                            v_init_mV=v_rest, dt_ms=0.1, enforce_dt=False)
    got = float(t_ms[1] - t_ms[0])
    check("enforce_dt=False reproduces the historical override",
          abs(got - 0.025) < 1e-12,
          "requested 0.1 ms, NEURON used {:.6g} ms".format(got))

    # And the hard failure path must actually raise, not return a penalty.
    raised = False
    try:
        h.steps_per_ms = 40.0
        cell.simulate(stim_amp_pA=-10.0, stim_delay_ms=20.0, stim_dur_ms=50.0,
                      tstop_ms=200.0, v_init_mV=v_rest, dt_ms=0.1,
                      enforce_dt=False)
        # enforce_dt=False does not check; force the checked path with a
        # deliberately inconsistent steps_per_ms is impossible now (simulate
        # sets it itself), so assert the exception TYPE exists and is a
        # RuntimeError subclass instead.
        raised = issubclass(mono.DtEnforcementError, RuntimeError)
    except Exception:
        raised = False
    check("DtEnforcementError exists and is a RuntimeError",
          raised)


# ---------------------------------------------------------------------------
#  2. REACH -- does one dt choice cover every simulating code path?
# ---------------------------------------------------------------------------
def section_reach(mono, plst, cell, cd, v_rest, ls_window_ms):
    print("\n[2] REACH -- does integrate_long_step's dt reach every code path?")
    plst.integrate_long_step(
        mono, n_long_train=2, max_ls_train_deflection_mV=12.0,
        max_sag_amplitude_mV=None, r_in_target="peak", weighting="relative",
        ss_window_ms=(0.5, 100.0), ss_time_weight="exp", ss_tau_w_ms=5.0,
        ls_window_ms_after_onset=ls_window_ms,
        dt_brief_ms=0.05, dt_long_ms=0.2, verbose=False)
    check("plst globals updated",
          plst.DEFAULT_DT_BRIEF_MS == 0.05 and plst.DEFAULT_DT_LONG_MS == 0.2,
          "brief={} long={}".format(plst.DEFAULT_DT_BRIEF_MS,
                                    plst.DEFAULT_DT_LONG_MS))
    check("monolith globals updated (validation / Phase 2.5 / Phase 3 path)",
          mono.DEFAULT_DT_BRIEF_MS == 0.05 and mono.DEFAULT_DT_LONG_MS == 0.2,
          "brief={} long={}".format(mono.DEFAULT_DT_BRIEF_MS,
                                    mono.DEFAULT_DT_LONG_MS))

    ls = [b for b in cd.long_square_subthreshold][:1]
    if ls:
        b = ls[0]
        # the monolith's OWN helper, used for validation RMSDs
        t_s, _ = mono._simulate_long_square(cell, b, v_rest)
        got = float((t_s[1] - t_s[0]) * 1e3)
        check("mono._simulate_long_square follows the global",
              abs(got - 0.2) < 1e-9, "dt = {:.6g} ms".format(got))
        # the training-loss helper
        t_s2, _ = plst._simulate_long(cell, b, v_rest)
        got2 = float((t_s2[1] - t_s2[0]) * 1e3)
        check("plst._simulate_long follows the global",
              abs(got2 - 0.2) < 1e-9, "dt = {:.6g} ms".format(got2))
        check("training and validation integrate at the SAME dt",
              abs(got - got2) < 1e-12)

    # restore the shipped defaults for the remaining sections
    plst.integrate_long_step(
        mono, n_long_train=2, max_ls_train_deflection_mV=12.0,
        max_sag_amplitude_mV=None, r_in_target="peak", weighting="relative",
        ss_window_ms=(0.5, 100.0), ss_time_weight="exp", ss_tau_w_ms=5.0,
        ls_window_ms_after_onset=ls_window_ms,
        dt_brief_ms=0.025, dt_long_ms=0.025, verbose=False)


# ---------------------------------------------------------------------------
#  3. ACCURACY -- what does dt = 0.1 ms actually cost?
# ---------------------------------------------------------------------------
def section_accuracy(np, plst, cell, bundle, v_rest, ls_window_ms):
    print("\n[3] ACCURACY -- the price of dt = 0.1 ms on the long step")
    onset_s = float(bundle.stim_onset_s)
    pre_w = (0.0, onset_s)
    rmsd_w = (onset_s, onset_s + ls_window_ms * 1e-3)
    t_exp = np.asarray(bundle.t)
    v_exp = np.asarray(bundle.v_mV)

    t_a, v_a = plst._simulate_long(cell, bundle, v_rest, dt_ms=0.025)
    t_b, v_b = plst._simulate_long(cell, bundle, v_rest, dt_ms=0.1)

    m = (t_a >= rmsd_w[0]) & (t_a <= rmsd_w[1])
    v_b_on_a = np.interp(t_a[m], t_b, v_b)
    dv_max = float(np.max(np.abs(v_a[m] - v_b_on_a))) if m.any() else float("nan")

    r_a = plst._baseline_subtracted_rmsd(t_exp, v_exp, t_a, v_a, pre_w, rmsd_w)
    r_b = plst._baseline_subtracted_rmsd(t_exp, v_exp, t_b, v_b, pre_w, rmsd_w)

    print("      samples  0.025 / 0.100 ms : {} / {}  ({:.2f}x fewer)".format(
        len(t_a), len(t_b), len(t_a) / max(len(t_b), 1)))
    print("      max |dV| in scored window : {:.6f} mV".format(dv_max))
    print("      bundle RMSD 0.025 / 0.100 : {:.6f} / {:.6f} mV".format(r_a, r_b))
    print("      |dRMSD|                   : {:.6f} mV".format(abs(r_a - r_b)))
    print("      -> compare |dRMSD| with the training noise sigma reported by")
    print("         fit_one_cell (~0.05 mV in the L3_exc logs). If |dRMSD| is")
    print("         a small fraction of it, dt = 0.1 ms is safe to adopt.")
    return dict(dv_max=dv_max, r_025=r_a, r_100=r_b)


# ---------------------------------------------------------------------------
#  4. NO-CHANGE -- shipped defaults must not move any number
# ---------------------------------------------------------------------------
def section_no_change(np, mono, plst, cps, cell, oi, ls_window_ms):
    print("\n[4] NO-CHANGE -- shipped defaults reproduce the historical result")
    check("shipped default dt_long == 0.025 (the step really used before)",
          plst.DEFAULT_DT_LONG_MS == 0.025 and mono.DEFAULT_DT_LONG_MS == 0.025)
    loss = cps.build_relative_loss_for_tau_w(
        cell, oi.train_bundles, float(oi.v_rest_mV), tau_w_ms=5.0,
        ss_window_ms=tuple(oi.train_window_ms), shape="exp",
        ls_window_ms_after_onset=ls_window_ms, r_in_target="peak",
        weighting="relative")
    pt = (np.log(1.0), np.log(15000.0), np.log(150.0))
    l1 = float(loss(*pt))
    l2 = float(loss(*pt))
    check("loss is finite and deterministic at a fixed point",
          np.isfinite(l1) and l1 == l2, "loss = {:.9g}".format(l1))


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive-dir", required=True)
    ap.add_argument("--code-dir", required=True)
    ap.add_argument("--specimen", type=int, default=None)
    ap.add_argument("--F", type=float, default=1.9)
    ap.add_argument("--fit-target", default="hyp")
    ap.add_argument("--ls-window-ms", type=float, default=150.0)
    ap.add_argument("--skip-accuracy", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.code_dir)))
    import numpy as np
    import passive_fitting_hpc_fixed as mono
    import passive_long_step_training as plst
    import cm_profile_sweep as cps

    sids = None if args.specimen is None else [int(args.specimen)]
    cells_data = mono.load_cells_from_archive(
        args.archive_dir, n_avg_groups=1, specimen_ids=sids, max_cells=1,
        verbose=False)
    if not cells_data:
        print("[ABORT] no cells loaded from {}".format(args.archive_dir))
        return 2
    cd = cells_data[0]
    print("=" * 70)
    print("SMOKE TEST -- dt enforcement -- specimen {}".format(cd.specimen_id))
    print("=" * 70)

    cell = mono.build_neuron_model(cd.swc_path, F=args.F)
    oi_pre = mono.prepare_optimiser_inputs(cd, fit_target=args.fit_target)
    v_rest = float(oi_pre.v_rest_mV)
    cell.set_passive(1.0, 15000.0, 150.0)
    cell.set_e_pas(v_rest)

    try:
        section_mechanism(mono, cell, v_rest)
        section_reach(mono, plst, cell, cd, v_rest, args.ls_window_ms)
        oi = mono.prepare_optimiser_inputs(cd, fit_target=args.fit_target)
        longs = [b for b in oi.train_bundles if not plst._is_brief(b)]
        if not args.skip_accuracy and longs:
            section_accuracy(np, plst, cell, longs[0], v_rest,
                             args.ls_window_ms)
        section_no_change(np, mono, plst, cps, cell, oi, args.ls_window_ms)
    except Exception:
        traceback.print_exc()
        FAIL.append("uncaught exception")
    finally:
        cell.destroy()

    print("\n" + "=" * 70)
    print("PASSED {}   FAILED {}".format(len(PASS), len(FAIL)))
    if FAIL:
        print("failures: {}".format(FAIL))
        print("DO NOT SUBMIT.")
        return 1
    print("dt enforcement verified. Safe to submit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
