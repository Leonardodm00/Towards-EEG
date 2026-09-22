#!/usr/bin/env python3
"""
profile_bottleneck.py
=====================

Standalone diagnostic for the biological passive-fit pipeline. It does NOT
fit anything: it measures where the wall time in Phase 2 actually goes, and
it verifies numerically that the two proposed simulator fixes are safe.

Run it on ONE cell. Expected runtime: a few minutes (it performs a handful of
loss evaluations, not thousands).

What it reports, in order
-------------------------
  A. Model size        : n_sections, n_segments  (the unknown that sets the
                         per-timestep cost; nseg is never set by
                         PassiveCell, so this is 1 segment per section)
  B. Bundle geometry   : for every training bundle, tstop actually requested
                         by _simulate_brief / _simulate_long, the scored
                         window, and the ratio (wasted fraction)
  C. dt override proof : requested dt vs the dt NEURON actually used, read
                         back from the returned time vector. NEURON's
                         stdrun.hoc setdt() silently resets dt whenever
                         abs(dt * nstep_steprun * steps_per_ms - 1) > 1e-6,
                         printing "Changed dt". With steps_per_ms = 40
                         (default) a requested dt = 0.1 ms becomes 0.025 ms.
  D. Timing table      : per-bundle simulate() wall time under
                         (i)   current settings,
                         (ii)  steps_per_ms fixed so dt = 0.1 is honoured,
                         (iii) (ii) + long-step tstop truncated to
                               onset + ls_window + margin.
  E. Correctness check : the LS RMSD under (i), (ii), (iii). The (i) vs
                         truncated-at-dt-0.025 comparison isolates the
                         TRUNCATION error (must be ~0, because for a purely
                         passive model initialised at v_init = e_pas = v_rest
                         the pre-onset solution is exactly constant and the
                         post-window samples are never scored). The
                         dt-0.025 vs dt-0.1 comparison isolates the
                         DISCRETISATION price of honouring dt = 0.1.
  F. Powell budget     : number of REAL loss evaluations consumed by ONE
                         profile_cm grid point, via a counting wrapper. The
                         sweep does n_grid of these.

Separation of concerns: every section is a pure function taking already-built
objects; nothing here loads data and simulates and prints inside one function.

Usage
-----
    python profile_bottleneck.py \
        --archive-dir "/davinci-1/home/ldellamea/Human Neurons Fitting/L3_exc" \
        --code-dir    "/davinci-1/home/ldellamea/Human Neurons Fitting" \
        --F 1.9

Optional:
    --specimen 508282493      pick a specific cell (default: first in archive)
    --ls-window-ms 150.0      must match the production --ls-window-ms
    --margin-ms 20.0          pre-onset padding kept by the truncated sim
    --skip-powell             skip section F (it costs ~80 loss evaluations)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
#  A. Model size
# ---------------------------------------------------------------------------
def report_model_size(h, cell):
    """Count sections and segments of the built model. Pure reporting."""
    n_sec = 0
    n_seg = 0
    nseg_hist = {}
    for sec in h.allsec():
        n_sec += 1
        n_seg += sec.nseg
        nseg_hist[sec.nseg] = nseg_hist.get(sec.nseg, 0) + 1
    print("\n[A] MODEL SIZE")
    print("    sections          : {}".format(n_sec))
    print("    segments (total)  : {}".format(n_seg))
    print("    nseg histogram    : {}".format(
        sorted(nseg_hist.items())))
    print("    soma / dend / apic / axon sections: {} / {} / {} / {}".format(
        len(cell.soma), len(cell.dend), len(cell.apic), len(cell.axon)))
    print("    NOTE: PassiveCell never applies a d_lambda rule, so nseg is")
    print("          whatever Import3d produced (normally 1 per section).")
    return dict(n_sections=n_sec, n_segments=n_seg)


# ---------------------------------------------------------------------------
#  B. Bundle geometry: simulated span vs scored span
# ---------------------------------------------------------------------------
def report_bundle_geometry(plst, train_bundles, ls_window_ms):
    """For each training bundle, print simulated tstop vs scored window."""
    print("\n[B] BUNDLE GEOMETRY  (simulated span vs scored span)")
    rows = []
    for b in train_bundles:
        brief = plst._is_brief(b)
        if brief:
            tstop_ms = 10.0 + float(b.stim_duration_s) * 1e3 + 100.0
            scored_ms = 100.0 - 0.5
            req_dt = 0.025
            kind = "SS brief"
        else:
            tstop_ms = float(b.t[-1]) * 1e3
            scored_ms = ls_window_ms
            req_dt = 0.1
            kind = "LS long"
        onset_ms = float(b.stim_onset_s) * 1e3
        dur_ms = float(b.stim_duration_s) * 1e3
        waste = tstop_ms / max(scored_ms, 1e-9)
        print("    {:9s} amp={:+7.1f} pA  onset={:8.2f} ms  dur={:8.2f} ms  "
              "tstop={:8.1f} ms  scored={:6.1f} ms  ratio={:6.1f}x  "
              "dt_req={:.3f}".format(kind, float(b.amplitude_pA), onset_ms,
                                     dur_ms, tstop_ms, scored_ms, waste,
                                     req_dt))
        rows.append(dict(kind=kind, tstop_ms=tstop_ms, scored_ms=scored_ms,
                         onset_ms=onset_ms, dur_ms=dur_ms, req_dt=req_dt))
    return rows


# ---------------------------------------------------------------------------
#  C. dt override proof
# ---------------------------------------------------------------------------
def prove_dt_override(h, cell, v_rest_mV, tstop_ms=200.0):
    """Request dt = 0.1 ms and read back the dt NEURON actually used.

    The returned time vector is recorded at every fadvance, so
    t[1] - t[0] IS the integration step that was used.
    """
    print("\n[C] dt OVERRIDE PROOF")
    print("    steps_per_ms (NEURON default) = {}".format(h.steps_per_ms))
    for req in (0.025, 0.1):
        t_ms, _ = cell.simulate(stim_amp_pA=-10.0, stim_delay_ms=20.0,
                                stim_dur_ms=50.0, tstop_ms=tstop_ms,
                                v_init_mV=v_rest_mV, dt_ms=req)
        actual = float(t_ms[1] - t_ms[0])
        print("    requested dt = {:.4f} ms  ->  actual dt = {:.4f} ms  "
              "({} samples)  {}".format(
                  req, actual, len(t_ms),
                  "OK" if abs(actual - req) < 1e-9 else "*** OVERRIDDEN ***"))


# ---------------------------------------------------------------------------
#  Patched simulators (proposed fixes), kept OUT of PassiveCell so the
#  diagnostic never mutates production behaviour.
# ---------------------------------------------------------------------------
def simulate_honest_dt(h, cell, stim_amp_pA, stim_delay_ms, stim_dur_ms,
                       tstop_ms, v_init_mV, dt_ms):
    """PassiveCell.simulate with steps_per_ms made consistent with dt.

    Fix 1: set steps_per_ms = 1/dt so stdrun's setdt() leaves dt alone.
    finitialize + continuerun avoids the redundant second initialisation
    that h.run() performs.
    """
    cell._iclamp.delay = float(stim_delay_ms)
    cell._iclamp.dur = float(stim_dur_ms)
    cell._iclamp.amp = float(stim_amp_pA) * 1e-3
    h.dt = float(dt_ms)
    h.steps_per_ms = 1.0 / float(dt_ms)
    h.tstop = float(tstop_ms)
    h.v_init = float(v_init_mV)
    h.finitialize(h.v_init)
    h.continuerun(float(tstop_ms))
    import numpy as np
    return np.array(cell._t_vec), np.array(cell._v_vec)


def simulate_long_truncated(h, cell, bundle, v_rest_mV, ls_window_ms,
                            margin_ms=20.0, dt_ms=0.1):
    """Long step simulated only over [onset - margin, onset + ls_window].

    Why this is exact for a passive model: with e_pas = v_rest and
    v_init = v_rest the model is at equilibrium, so V(t) = v_rest for every
    t < delay, independent of how long that stretch is. The pre-onset span
    therefore contributes a CONSTANT baseline and nothing else, and samples
    after onset + ls_window are never scored.

    The returned time base is shifted back to ABSOLUTE bundle time so the
    downstream np.interp against bundle.t is unchanged.
    """
    import numpy as np
    onset_ms = float(bundle.stim_onset_s) * 1e3
    dur_ms = float(bundle.stim_duration_s) * 1e3
    delay_ms = float(margin_ms)
    tstop_ms = delay_ms + float(ls_window_ms) + float(dt_ms)
    t_ms, v = simulate_honest_dt(h, cell, float(bundle.amplitude_pA),
                                 delay_ms, dur_ms, tstop_ms,
                                 v_rest_mV, dt_ms)
    t_abs_s = (np.asarray(t_ms) + (onset_ms - delay_ms)) * 1e-3
    return t_abs_s, np.asarray(v)


# ---------------------------------------------------------------------------
#  D + E. Timing and correctness for the long steps
# ---------------------------------------------------------------------------
def time_and_check_long(h, np, plst, cell, bundle, v_rest_mV, ls_window_ms,
                        margin_ms, n_repeat=2):
    """Three configurations, timed and scored. Returns a dict of results."""
    onset_s = float(bundle.stim_onset_s)
    pre_w = (0.0, onset_s)
    rmsd_w = (onset_s, onset_s + ls_window_ms * 1e-3)
    t_exp = np.asarray(bundle.t)
    v_exp = np.asarray(bundle.v_mV)

    def score(t_s, v):
        return plst._baseline_subtracted_rmsd(t_exp, v_exp, t_s, v,
                                              pre_w, rmsd_w,
                                              sample_weight_fn=None)

    def timeit(fn):
        best = float("inf")
        out = None
        for _ in range(n_repeat):
            t0 = time.time()
            out = fn()
            best = min(best, time.time() - t0)
        return best, out

    # (i) current production path
    t_cur, (ts_cur, vs_cur) = timeit(
        lambda: plst._simulate_long(cell, bundle, v_rest_mV))
    # (iii-a) truncated but still at dt = 0.025 -> isolates truncation error
    t_tr25, (ts_tr25, vs_tr25) = timeit(
        lambda: simulate_long_truncated(h, cell, bundle, v_rest_mV,
                                        ls_window_ms, margin_ms, 0.025))
    # (iii-b) truncated AND honest dt = 0.1 -> the proposed configuration
    t_tr10, (ts_tr10, vs_tr10) = timeit(
        lambda: simulate_long_truncated(h, cell, bundle, v_rest_mV,
                                        ls_window_ms, margin_ms, 0.1))

    r_cur, r_tr25, r_tr10 = score(ts_cur, vs_cur), score(ts_tr25, vs_tr25), \
        score(ts_tr10, vs_tr10)
    return dict(
        amp=float(bundle.amplitude_pA),
        t_cur=t_cur, t_tr25=t_tr25, t_tr10=t_tr10,
        n_cur=len(ts_cur), n_tr25=len(ts_tr25), n_tr10=len(ts_tr10),
        r_cur=r_cur, r_tr25=r_tr25, r_tr10=r_tr10)


def report_long_table(rows):
    print("\n[D/E] LONG-STEP TIMING AND RMSD EQUIVALENCE")
    print("    cfg-i   = production  (tstop = full sweep, dt requested 0.1 "
          "-> actually 0.025)")
    print("    cfg-ii  = truncated window, dt = 0.025  (isolates TRUNCATION)")
    print("    cfg-iii = truncated window, dt = 0.100  (proposed)")
    for r in rows:
        print("\n    LS amp = {:+.1f} pA".format(r["amp"]))
        print("      samples   i/ii/iii : {} / {} / {}".format(
            r["n_cur"], r["n_tr25"], r["n_tr10"]))
        print("      wall  [s] i/ii/iii : {:.3f} / {:.3f} / {:.3f}".format(
            r["t_cur"], r["t_tr25"], r["t_tr10"]))
        print("      speedup     ii/iii : {:.1f}x / {:.1f}x".format(
            r["t_cur"] / max(r["t_tr25"], 1e-9),
            r["t_cur"] / max(r["t_tr10"], 1e-9)))
        print("      RMSD [mV] i/ii/iii : {:.6f} / {:.6f} / {:.6f}".format(
            r["r_cur"], r["r_tr25"], r["r_tr10"]))
        print("      |i - ii|  (truncation error)    = {:.3e} mV".format(
            abs(r["r_cur"] - r["r_tr25"])))
        print("      |ii - iii| (dt discretisation)  = {:.3e} mV".format(
            abs(r["r_tr25"] - r["r_tr10"])))


# ---------------------------------------------------------------------------
#  F. Powell evaluation budget for ONE profile grid point
# ---------------------------------------------------------------------------
def count_powell_evals(np, cps, loss, cm_bounds, rm_bounds, ra_bounds):
    """Run the inner minimisation for a SINGLE Cm grid node with a counter."""
    from scipy.optimize import minimize

    calls = {"n": 0}

    def counted(cm_log, rm_log, ra_log):
        calls["n"] += 1
        return loss(cm_log, rm_log, ra_log)

    lcm = float(np.log(np.sqrt(cm_bounds[0] * cm_bounds[1])))   # mid grid node
    warm = np.array([np.log(15000.0), np.log(150.0)], dtype=float)
    t0 = time.time()
    res = minimize(lambda x: float(counted(lcm, float(x[0]), float(x[1]))),
                   warm, method="Powell",
                   bounds=[(np.log(rm_bounds[0]), np.log(rm_bounds[1])),
                           (np.log(ra_bounds[0]), np.log(ra_bounds[1]))],
                   options={"maxiter": 200, "xtol": 1e-4, "ftol": 1e-4})
    dt_s = time.time() - t0
    print("\n[F] POWELL BUDGET FOR ONE Cm GRID NODE")
    print("    loss evaluations : {}".format(calls["n"]))
    print("    wall time        : {:.1f} s  -> {:.2f} s per evaluation"
          .format(dt_s, dt_s / max(calls["n"], 1)))
    print("    scipy nfev       : {}   success={}".format(
        getattr(res, "nfev", "n/a"), getattr(res, "success", "n/a")))
    print("    NOTE: scipy Powell caps on maxfev (default 2 * 1000 = 2000),")
    print("          NOT on maxiter=200. A hard grid node can burn 2000 evals.")
    return calls["n"], dt_s


# ---------------------------------------------------------------------------
#  Orchestration
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive-dir", required=True)
    ap.add_argument("--code-dir", required=True)
    ap.add_argument("--specimen", type=int, default=None)
    ap.add_argument("--F", type=float, default=1.9)
    ap.add_argument("--fit-target", default="hyp")
    ap.add_argument("--ls-window-ms", type=float, default=150.0)
    ap.add_argument("--margin-ms", type=float, default=20.0)
    ap.add_argument("--n-long-train", type=int, default=2)
    ap.add_argument("--n-repeat", type=int, default=2)
    ap.add_argument("--skip-powell", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.code_dir)))
    import numpy as np
    from neuron import h
    import passive_fitting_hpc_fixed as mono
    import passive_long_step_training as plst
    import cm_profile_sweep as cps

    # Patch exactly as production does, so train_bundles match the real run.
    plst.integrate_long_step(
        mono, n_long_train=args.n_long_train,
        max_ls_train_deflection_mV=12.0, max_sag_amplitude_mV=None,
        r_in_target="peak", weighting="relative",
        ss_window_ms=(0.5, 100.0), ss_time_weight="exp", ss_tau_w_ms=5.0,
        ls_window_ms_after_onset=args.ls_window_ms, verbose=False)

    sids = None if args.specimen is None else [int(args.specimen)]
    cells_data = mono.load_cells_from_archive(
        args.archive_dir, n_avg_groups=1, specimen_ids=sids,
        max_cells=1, verbose=False)
    if not cells_data:
        print("[ABORT] no cells loaded from {}".format(args.archive_dir))
        sys.exit(1)
    cd = cells_data[0]
    oi = mono.prepare_optimiser_inputs(cd, fit_target=args.fit_target)
    print("=" * 72)
    print("DIAGNOSTIC for specimen {}".format(cd.specimen_id))
    print("=" * 72)

    cell = mono.build_neuron_model(cd.swc_path, F=args.F)
    v_rest = float(oi.v_rest_mV)

    report_model_size(h, cell)
    report_bundle_geometry(plst, oi.train_bundles, args.ls_window_ms)
    prove_dt_override(h, cell, v_rest)

    # Put the cell at a plausible operating point before timing.
    cell.set_passive(1.0, 15000.0, 150.0)
    cell.set_e_pas(v_rest)

    rows = []
    for b in oi.train_bundles:
        if plst._is_brief(b):
            t0 = time.time()
            for _ in range(args.n_repeat):
                plst._simulate_brief(cell, b, v_rest)
            print("\n    SS brief simulate(): {:.3f} s per call".format(
                (time.time() - t0) / args.n_repeat))
            continue
        rows.append(time_and_check_long(h, np, plst, cell, b, v_rest,
                                        args.ls_window_ms, args.margin_ms,
                                        n_repeat=args.n_repeat))
    report_long_table(rows)

    if not args.skip_powell:
        loss = cps.build_relative_loss_for_tau_w(
            cell, oi.train_bundles, v_rest, tau_w_ms=5.0,
            ss_window_ms=tuple(oi.train_window_ms), shape="exp",
            ls_window_ms_after_onset=args.ls_window_ms,
            r_in_target="peak", weighting="relative")
        n_ev, dt_s = count_powell_evals(np, cps, loss, mono.DEFAULT_CM_BOUNDS,
                                        mono.DEFAULT_RM_BOUNDS,
                                        mono.DEFAULT_RA_BOUNDS)
        print("\n[SUMMARY] projected sweep cost = n_grid x {} evals x {:.1f} s"
              .format(n_ev, dt_s / max(n_ev, 1)))
        print("          with n_grid = 15 -> {:.1f} h"
              .format(15 * dt_s / 3600.0))

    cell.destroy()


if __name__ == "__main__":
    main()
