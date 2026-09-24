"""smoke_ih_fit.py -- Stage 1 smoke suite of the I_h fitting pipeline.

Checks S1-S5 and S10 of TEEG_Ih_fit_staged_plan.md section 9 (S6-S9 arrive
with Stages 2-7). Self-contained: builds its own ball-and-stick cell; needs
NEURON and the compiled mechanisms in ./x86_64 (run ``nrnivmodl mod`` in this
directory first, or pass --build).

Run:
    cd "<repo>/Passive Features/HPC script/Ih Fit"
    nrnivmodl mod                # once, on the cluster (architecture-specific)
    python smoke_ih_fit.py       # expect: "smoke_ih_fit: 6/6 passed"

Every check prints PASS/FAIL and one line of evidence; the exit code is 0
only if all pass. Numbers quoted in the checks were computed on 2026-09-22
from the repository's reference kinetics (human_ih_params.py).

    S1  compiled mInf/mTau == pure-Python reference for both mechanisms, at
        defaults AND at non-default (vshift, vshift_minf, tau_scale)
    S2  spatial laws: uniform == 1 everywhere; Eyal Eq. 4 factors 1.217 /
        4.803 / 45.27 at 0 / 323 / 1000 um; soma+basal carry gbar
    S3  rest balance: with no stimulus max|v - V_rest| < 1e-3 mV over 2 s,
        for both mechanisms x dv_h in {-10,0,10} x kappa in {0.5,2} x two
        laws; and NEURON's large-dt implicit steady-state init agrees
    S4  sag: fraction monotone in gbar and ~0 at the floor; m_inf(rest)
        monotone in dv_h and the sag fraction responds to dv_h; the recovery
        time t63 monotone in kappa_tau; rebound > 0 after offset
    S5  Kalmbach convention: Ih with vshift = +20 gives mInf(-73.5) = 0.166
        and V1/2 = -90.3 mV (compiled == reference); unshifted V1/2 = -110.3
    S6  ParamSpec identity: the 3-D passive spec reproduces
        PassiveSearchSpace's skopt dimensions exactly, and the spec-driven
        loss equals the legacy closure at random parameters
    S7  loader + roles: the legacy loader call is unchanged by Stage 2; the
        I_h call admits every hyperpolarising amplitude, builds spike-free
        depolarising bundles, and assign_ls_roles puts each sweep in the
        role D-006 prescribes
    S10 byte safety: pure ASCII in every .py, zero CR in .py/.mod/.sh
    S8  Phase 3 over the generic vector: a 3-D bootstrap still yields three
        correctly named columns, a 6-D one yields six, and the LINEAR axis
        (dv_h) is stored through its own transform rather than exp() -- the
        one place where a silent bug would have produced plausible, wrong
        confidence intervals
    S11 six-parameter wiring: a 6-D fit_one_cell runs end to end on a
        synthetic I_h cell and returns a theta with all six axes
    S12 the campaign entrypoint trains the SS pulses with the exponential
        time-weight and records where the morphology came from
    S13 the depolarising-sweep spike screen (D-016): noise at 50-200 kHz and
        a bridge-balance step are not spikes, an action potential peaking
        below the -20 mV catch-all is; the rule it replaced flags the noise;
        the loader and the role assignment report why a sweep was dropped;
        check_dep_sweeps.py reports both screens
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.chdir(HERE)                     # NEURON loads ./x86_64 from the CWD
sys.path.insert(0, str(HERE))

import numpy as np                 # noqa: E402

RESULTS = []


#: The value both loss builders return when a simulation raises.
#: It is FINITE, so `isfinite` is never enough to prove a loss is live.
_LOSS_CRASH_PENALTY = 1e6


def report(name: str, ok: bool, evidence: str) -> None:
    RESULTS.append(bool(ok))
    print("%s  %s  --  %s" % ("PASS" if ok else "FAIL", name, evidence))


# ---------------------------------------------------------------------------
def ensure_build(build: bool) -> None:
    special = HERE / "x86_64" / "special"
    if special.exists():
        return
    if not build:
        sys.exit("x86_64/special not found; run 'nrnivmodl mod' here or pass --build")
    r = subprocess.run(["nrnivmodl", "mod"], cwd=str(HERE), capture_output=True, text=True)
    if r.returncode != 0 or not special.exists():
        sys.exit("nrnivmodl failed:\n" + r.stdout[-2000:] + r.stderr[-2000:])


def make_cell(F: float = 1.9, nseg: int = 41):
    """The monolith's own PassiveCell on a ball-and-stick SWC.

    Import3d leaves nseg = 1 per section (the pipeline's standing state); the
    smoke fixture raises it to `nseg` per dendritic section so that the rest
    balance and the spatial laws are exercised across a real gradient, and
    re-runs the monolith's per-segment F precompute, which keys on (section,
    seg.x) and would otherwise KeyError on the new segments."""
    import passive_fitting_hpc_fixed as mono
    from synthetic_ground_truth import write_ball_and_stick_swc
    tmp = Path(tempfile.mkdtemp(prefix="smoke_ih_"))
    swc = write_ball_and_stick_swc(tmp / "bas.swc", soma_r_um=10.0,
                                   dend_len_um=400.0, dend_r_um=1.0,
                                   apic_len_um=1000.0, apic_r_um=1.2, step_um=20.0)
    cell = mono.build_neuron_model(swc, F=F)
    for sec in cell.dend + cell.apic:
        sec.nseg = int(nseg)
    cell._precompute_F_per_segment()
    return cell


# ---------------------------------------------------------------------------
def make_ih_archive(tmp: Path, *, specimen_id: int = 900000101,
                    gbar: float = 1.2e-4, dv_h: float = 0.0,
                    kappa: float = 1.0, verbose: bool = False) -> Path:
    """A synthetic Phase-0 archive cell WITH I_h, spiking depolarising steps
    included, so the loader's spike filter and the role assignment are
    exercised on data of the shape the real archive has."""
    import synthetic_ground_truth as sgt
    swc = sgt.write_ball_and_stick_swc(tmp / "ih.swc", soma_r_um=10.0,
                                       dend_len_um=400.0, dend_r_um=1.0,
                                       apic_len_um=600.0, apic_r_um=1.2,
                                       step_um=20.0)
    ih = sgt.IhConfig(gIhbar_S_cm2=gbar, ehcn_mV=-49.85, distribution="uniform",
                      mechanism="Ih_human", vshift_minf_mV=dv_h, tau_scale=kappa)
    gt = sgt.GroundTruthParams(cm_uF_cm2=0.9, rm_Ohm_cm2=30000.0, ra_Ohm_cm=200.0,
                               e_pas_mV=-78.0, ih=ih, active=True,
                               active_regions=("soma", "axon"))
    proto = sgt.ProtocolConfig(ss_n_repeats=6,
                               ls_hyp_amplitudes_pA=(-10., -30., -50., -70., -90., -110., -150.),
                               ls_dep_amplitudes_pA=(20., 50., 200.))
    syn = sgt.generate_synthetic_cell(swc, gt, proto=proto,
                                      noise=sgt.NoiseConfig(sigma_mV=0.05, seed=7),
                                      specimen_id=specimen_id, verbose=verbose)
    d = tmp / ("specimen_%d" % specimen_id)
    sgt.write_archive_cell(syn, d, verbose=verbose)
    sgt._clear_neuron_sections()
    return d


def check_S1() -> None:
    from neuron import h
    import human_ih_params as H
    grid = np.arange(-130.0, -40.0 + 1e-9, 5.0)
    settings = [dict(vshift=0.0, vshift_minf=0.0, tau_scale=1.0),
                dict(vshift=20.0, vshift_minf=0.0, tau_scale=1.0),
                dict(vshift=0.0, vshift_minf=-10.0, tau_scale=1.0),
                dict(vshift=0.0, vshift_minf=10.0, tau_scale=0.5),
                dict(vshift=5.0, vshift_minf=3.0, tau_scale=2.0)]
    worst = 0.0
    n = 0
    for mech in H.IH_MECHANISMS:
        sec = h.Section(name="s1_" + mech)
        sec.insert(mech)
        mobj = getattr(sec(0.5), mech)
        for st in settings:
            mobj.vshift = st["vshift"]; mobj.vshift_minf = st["vshift_minf"]; mobj.tau_scale = st["tau_scale"]
            for v in grid:
                h.finitialize(float(v))          # INITIAL -> rates() at v
                mi = float(mobj.mInf); mt = float(mobj.mTau)
                ri = H.minf(v, mech, vshift_mV=st["vshift"], vshift_minf_mV=st["vshift_minf"])
                rt = H.mtau_ms(v, mech, vshift_mV=st["vshift"], tau_scale=st["tau_scale"])
                worst = max(worst, abs(mi - ri) / max(abs(ri), 1e-12),
                            abs(mt - rt) / max(abs(rt), 1e-12))
                n += 1
        h.delete_section(sec=sec)
    report("S1 compiled kinetics == Python reference", worst < 1e-9,
           "%d (mechanism, setting, v) points, worst rel. dev. %.2e" % (n, worst))


def check_S2() -> None:
    from neuron import h
    import ih_mechanism as IM
    ok = True; notes = []
    f0, f323, f1000 = IM.eyal_factor_323(0.0), IM.eyal_factor_323(323.0), IM.eyal_factor_323(1000.0)
    ok &= abs(f0 - 1.2174) < 1e-3 and abs(f323 - 4.8031) < 1e-3 and abs(f1000 - 45.271) < 2e-2
    notes.append("Eq.4 factors %.3f/%.3f/%.2f" % (f0, f323, f1000))
    cell = make_cell()
    cell.set_passive(1.0, 30000.0, 150.0)
    # uniform
    spec_u = IM.IhSpec(mechanism="Ih_human", distribution="uniform")
    fac = IM.attach_ih(cell, spec_u)
    IM.set_ih(cell, 1e-4, 0.0, 1.0)
    vals = [getattr(seg, "Ih_human").gIhbar for sec in cell.soma + cell.dend + cell.apic for seg in sec]
    ok &= all(abs(f - 1.0) < 1e-12 for f in fac.values()) and max(abs(x - 1e-4) for x in vals) < 1e-18
    ok &= all(not sec.has_membrane("Ih_human") for sec in cell.axon)
    notes.append("uniform: %d segs at gbar, axon stub excluded" % len(vals))
    # Eyal law
    h.distance(0, cell.soma[0](0.5))
    spec_e = IM.IhSpec(mechanism="Ih_human", distribution="eyal_exp_323")
    fac = IM.attach_ih(cell, spec_e)
    IM.set_ih(cell, 1e-4, 0.0, 1.0)
    worst = 0.0
    for sec in cell.apic:
        for seg in sec:
            d = h.distance(seg.x, sec=sec)
            worst = max(worst, abs(fac[(sec.name(), seg.x)] - IM.eyal_factor_323(d)))
            worst = max(worst, abs(getattr(seg, "Ih_human").gIhbar - 1e-4 * IM.eyal_factor_323(d)) / 1e-4)
    for sec in cell.soma + cell.dend:
        for seg in sec:
            worst = max(worst, abs(fac[(sec.name(), seg.x)] - 1.0))
    ok &= worst < 1e-9
    notes.append("Eyal law per-segment worst dev %.1e" % worst)
    cell.destroy()
    report("S2 spatial laws", ok, "; ".join(notes))


def check_S3() -> None:
    from neuron import h
    import ih_mechanism as IM
    v_rest = -73.5
    worst_drift = 0.0; worst_ss = 0.0; n = 0
    for mech in ("Ih_human", "Ih"):
        for law in ("uniform", "eyal_exp_323"):
            cell = make_cell()
            cell.set_passive(1.0, 30000.0, 150.0)
            spec = IM.IhSpec(mechanism=mech, distribution=law)
            IM.attach_ih(cell, spec)
            for dv in (-10.0, 0.0, 10.0):
                for kappa in (0.5, 2.0):
                    IM.set_ih(cell, 1e-4, dv, kappa)
                    IM.balance_e_pas(cell, v_rest)
                    drift = IM.rest_drift(cell, v_rest, t_ms=2000.0, dt_ms=0.1)
                    worst_drift = max(worst_drift, drift)
                    # large-dt implicit steady-state initialisation cross-check
                    h.finitialize(v_rest)
                    h.secondorder = 0
                    dt_saved = h.dt
                    h.dt = 1e4
                    for _ in range(20):
                        h.fadvance()
                    h.dt = dt_saved
                    ss_dev = max(abs(seg.v - v_rest) for sec in cell.soma + cell.dend + cell.apic for seg in sec)
                    worst_ss = max(worst_ss, ss_dev)
                    n += 1
            cell.destroy()
    ok = worst_drift < 1e-3 and worst_ss < 1e-3
    report("S3 rest balance keeps V_rest stationary", ok,
           "%d configs; max |v-V_rest| over 2 s = %.2e mV; large-dt SS init dev %.2e mV" % (n, worst_drift, worst_ss))


def _step_response(cell, v_rest, amp_pA=-50.0, onset=100.0, dur=1000.0, post=400.0, dt=0.1):
    import ih_mechanism as IM
    t, v = cell.simulate(stim_amp_pA=amp_pA, stim_delay_ms=onset, stim_dur_ms=dur,
                         tstop_ms=onset + dur + post, v_init_mV=v_rest, dt_ms=dt)
    return IM.sag_metrics(np.asarray(t), np.asarray(v), v_rest_mV=v_rest, onset_ms=onset, offset_ms=onset + dur)


def check_S4() -> None:
    import ih_mechanism as IM
    import human_ih_params as H
    v_rest = -73.5
    cell = make_cell()
    cell.set_passive(1.0, 30000.0, 150.0)
    spec = IM.IhSpec(mechanism="Ih_human", distribution="uniform")
    IM.attach_ih(cell, spec)
    notes = []; ok = True
    # (a) sag monotone in gbar, ~0 at the floor
    sags = []
    for g in (1e-6, 5e-5, 1e-4, 2e-4):
        IM.set_ih(cell, g, 0.0, 1.0); IM.balance_e_pas(cell, v_rest)
        sags.append(_step_response(cell, v_rest)["sag_fraction"])
    ok &= all(b > a for a, b in zip(sags, sags[1:])) and sags[0] < 5e-3
    notes.append("sag vs gbar " + "/".join("%.3f" % s for s in sags))
    # (b) m_inf(rest) monotone in dv_h and sag responds to dv_h
    mrest = [H.minf(v_rest, "Ih_human", vshift_minf_mV=dv) for dv in (-10.0, 0.0, 10.0)]
    sag_dv = []
    for dv in (-10.0, 0.0, 10.0):
        IM.set_ih(cell, 1e-4, dv, 1.0); IM.balance_e_pas(cell, v_rest)
        sag_dv.append(_step_response(cell, v_rest)["sag_fraction"])
    ok &= mrest[0] < mrest[1] < mrest[2] and (max(sag_dv) - min(sag_dv)) > 0.01
    notes.append("m_inf(rest) %.3f/%.3f/%.3f; sag %.3f/%.3f/%.3f at dv -10/0/+10"
                 % (mrest[0], mrest[1], mrest[2], sag_dv[0], sag_dv[1], sag_dv[2]))
    # (c) t63 recovery monotone in kappa; rebound > 0
    t63 = []; reb = None
    for k in (0.5, 1.0, 2.0):
        IM.set_ih(cell, 1e-4, 0.0, k); IM.balance_e_pas(cell, v_rest)
        m = _step_response(cell, v_rest)
        t63.append(m["t63_recovery_ms"]); reb = m["rebound_mV"]
    ok &= all(np.isfinite(t63)) and t63[0] < t63[1] < t63[2] and reb > 0.0
    notes.append("t63 %.0f/%.0f/%.0f ms at kappa 0.5/1/2; rebound %.2f mV" % (t63[0], t63[1], t63[2], reb))
    cell.destroy()
    report("S4 sag responds to gbar, dv_h, kappa_tau", ok, "; ".join(notes))


def check_S5() -> None:
    from neuron import h
    import human_ih_params as H
    sec = h.Section(name="s5"); sec.insert("Ih")
    mobj = sec(0.5).Ih
    mobj.vshift = 20.0
    h.finitialize(-73.5)
    mi = float(mobj.mInf)
    vh_sh = H.v_half_mV("Ih", vshift_mV=20.0); vh_0 = H.v_half_mV("Ih")
    ok = abs(mi - 0.166) < 2e-3 and abs(vh_sh + 90.3) < 0.2 and abs(vh_0 + 110.3) < 0.2
    h.delete_section(sec=sec)
    report("S5 Kalmbach shift convention (vshift = +20)", ok,
           "compiled mInf(-73.5) = %.3f; V1/2 shifted %.1f, unshifted %.1f mV" % (mi, vh_sh, vh_0))


def check_S6() -> None:
    import numpy as _np
    import param_spec as PS
    import passive_fitting_hpc_fixed as mono
    ok = True; notes = []
    a = PS.PASSIVE_3D.as_skopt_dimensions()
    b = mono.PassiveSearchSpace().as_skopt_dimensions()
    same_dims = (len(a) == len(b) and all(
        x.bounds == y.bounds and x.name == y.name and x.prior == y.prior
        for x, y in zip(a, b)))
    ok &= same_dims
    notes.append("skopt dims identical to PassiveSearchSpace: %s" % same_dims)
    # round-trip of the 6-D spec
    s6 = PS.make_ih_spec()
    th = {"Cm": 0.9, "Rm": 30000.0, "Ra": 200.0, "gbar": 1.3e-4,
          "dv_h": -4.0, "kappa_tau": 1.7}
    back = s6.to_physical(s6.to_q(th))
    ok &= all(abs(back[k] - v) <= 1e-12 * max(abs(v), 1.0) for k, v in th.items())
    ok &= s6.names == ("Cm", "Rm", "Ra", "gbar", "dv_h", "kappa_tau")
    ok &= PS.make_ih_spec(kappa_bounds=None).n == 5
    notes.append("6-D round-trip exact; freezing kappa gives 5 axes")
    # The spec-driven loss, called with the legacy 3 positional args.
    #
    # ORDER MATTERS: make_ih_archive() runs the synthetic generator, which
    # builds its OWN NEURON cell and clears h.allsec(). A cell built BEFORE it
    # keeps a Python handle to an IClamp whose section no longer exists, so
    # every simulate() raises "point process not located in a section", the
    # loss builder swallows it and returns its 1e6 penalty -- which is finite.
    # The archive is therefore generated first, and the check below asserts
    # the loss is BELOW the penalty and VARIES, because `isfinite` alone
    # passes on 1e6 and would hide exactly this.
    tmp = Path(tempfile.mkdtemp(prefix="smoke_s6_"))
    d = make_ih_archive(tmp, specimen_id=900000106)
    cell = make_cell()
    cd = mono.load_cell_from_archive(d, verbose=False)
    oi = mono.prepare_optimiser_inputs(cd, fit_target="hyp")
    loss_spec = mono._build_loss_function(cell=cell, train_bundles=oi.train_bundles,
                                          v_rest_mV=oi.v_rest_mV,
                                          train_window_ms=oi.train_window_ms,
                                          spec=PS.PASSIVE_3D)
    rng = _np.random.default_rng(4)
    lo = _np.log([0.3, 1e3, 50.0]); hi = _np.log([3.0, 1e5, 1000.0])
    qs = lo + rng.random((8, 3)) * (hi - lo)
    vals = [float(loss_spec(*map(float, q))) for q in qs]
    live = (all(_np.isfinite(vals)) and max(vals) < _LOSS_CRASH_PENALTY
            and (max(vals) - min(vals)) > 1e-9)
    ok &= live
    notes.append("spec loss live at %d random theta: range %.4f..%.4f "
                 "(no 1e6 penalty; legacy 3-arg call works)"
                 % (len(vals), min(vals), max(vals)))
    cell.destroy()
    report("S6 ParamSpec is the passive spec", ok, "; ".join(notes))


def check_S7() -> None:
    import numpy as _np
    import passive_fitting_hpc_fixed as mono
    import passive_long_step_training as plst
    ok = True; notes = []
    tmp = Path(tempfile.mkdtemp(prefix="smoke_s7_"))
    d = make_ih_archive(tmp, specimen_id=900000107)

    cd_old = mono.load_cell_from_archive(d, verbose=False)
    cd_new = mono.load_cell_from_archive(d, ls_max_amplitude_pA=None,
                                         load_depolarising_ls=True, verbose=False)
    amps_old = [round(b.amplitude_pA) for b in cd_old.long_square_subthreshold]
    amps_new = [round(b.amplitude_pA) for b in cd_new.long_square_subthreshold]
    ok &= amps_old == [-10, -30, -50, -70, -90]          # the 100 pA cap
    ok &= amps_new == [-10, -30, -50, -70, -90, -110, -150]
    ok &= len(cd_old.long_square_depolarising) == 0       # legacy untouched
    notes.append("hyp amps legacy %s -> no-cap %s" % (amps_old, amps_new))
    # identical arrays for the amplitudes both loaders kept
    for b0 in cd_old.long_square_subthreshold:
        b1 = next(b for b in cd_new.long_square_subthreshold
                  if abs(b.amplitude_pA - b0.amplitude_pA) < 1e-9)
        ok &= _np.array_equal(b0.v_mV, b1.v_mV) and _np.array_equal(b0.t, b1.t)
    # depolarising: spike-free only
    dep = cd_new.long_square_depolarising
    ok &= len(dep) >= 1
    peaks = [float(_np.max(b.v_mV)) for b in dep]
    ok &= all(pk <= mono.DEFAULT_SPIKE_V_THRESHOLD_MV for pk in peaks)
    notes.append("dep bundles %s pA, peaks %s mV (all subthreshold)"
                 % ([round(b.amplitude_pA) for b in dep],
                    [round(pk, 1) for pk in peaks]))
    # troughs monotone in |amplitude|
    tr = [mono.bundle_trough_mV(b) for b in cd_new.long_square_subthreshold]
    ok &= all(b < a for a, b in zip(tr, tr[1:]))
    # roles
    roles = plst.assign_ls_roles(cd_new, verbose=False)
    tr_amps = [round(b.amplitude_pA) for b in roles["train"]]
    va_amps = [round(b.amplitude_pA) for b in roles["validate"]]
    rp_amps = [round(b.amplitude_pA) for b in roles["report"]]
    ok &= tr_amps == [-30, -50, -70, -90, -110]     # h_2 .. h_{n-1}
    ok &= -10 in va_amps and all(a > 0 for a in va_amps if a != -10)
    ok &= rp_amps == [-150]                         # h_n withheld
    notes.append("roles train %s | validate %s | report %s" % (tr_amps, va_amps, rp_amps))
    # the opt-in trains on everything
    roles_all = plst.assign_ls_roles(cd_new, train_all_hyp=True, verbose=False)
    ok &= len(roles_all["train"]) == 7 and not roles_all["report"]
    # window modes widen monotonically
    b = cd_new.long_square_subthreshold[0]
    w_leg = plst.ls_rmsd_window_s(b, "after_onset", 60.0)
    w_step = plst.ls_rmsd_window_s(b, "step", 60.0)
    w_sweep = plst.ls_rmsd_window_s(b, "sweep", 60.0)
    ok &= w_leg[1] < w_step[1] <= w_sweep[1] and w_leg[0] == w_step[0] == w_sweep[0]
    notes.append("windows after_onset %.0f ms < step %.0f ms <= sweep %.0f ms"
                 % ((w_leg[1] - w_leg[0]) * 1e3, (w_step[1] - w_step[0]) * 1e3,
                    (w_sweep[1] - w_sweep[0]) * 1e3))
    report("S7 loader admits every amplitude; roles per D-006", ok, "; ".join(notes))


def check_S11() -> None:
    import numpy as _np
    import param_spec as PS
    import passive_fitting_hpc_fixed as mono
    import passive_long_step_training as plst
    import ih_mechanism as IM
    tmp = Path(tempfile.mkdtemp(prefix="smoke_s11_"))
    d = make_ih_archive(tmp, specimen_id=900000111, gbar=1.2e-4, dv_h=0.0, kappa=1.0)

    spec = PS.make_ih_spec()
    plst.integrate_long_step(mono, spec=spec, ih_protocol=True,
                             ls_window_mode="step", r_in_target="peak",
                             weighting="relative", ss_window_ms=(0.5, 100.0),
                             ss_time_weight="exp", ss_tau_w_ms=5.0,
                             dt_brief_ms=0.1, dt_long_ms=0.1, verbose=False)
    cd = mono.load_cell_from_archive(d, ls_max_amplitude_pA=None,
                                     load_depolarising_ls=True, verbose=False)
    oi = mono.prepare_optimiser_inputs(cd, fit_target="hyp")
    cell = mono.build_neuron_model(cd.swc_path, F=1.9)
    IM.attach_ih(cell, IM.IhSpec(mechanism="Ih_human", distribution="uniform"))
    fr = mono.fit_one_cell(cell, cd, oi, spec=spec, F=1.9,
                           n_calls=14, n_initial=8, seed=0, verbose=False)
    ok = (fr.gp_result is not None
          and set(fr.params) == set(spec.names)
          and all(_np.isfinite(v) for v in fr.params.values())
          # below the penalty, not merely finite: a fit whose every simulation
          # raised would report 1e6 and pass an isfinite test
          and _np.isfinite(fr.train_rmsd_mV)
          and float(fr.train_rmsd_mV) < _LOSS_CRASH_PENALTY
          and len(fr.sigmas_by_name) == 6
          and oi.param_spec is spec
          and len(oi.report_bundles) == 1)
    summ = IM.ih_rest_summary(cell, oi.v_rest_mV, fr.params["Rm"])
    cell.destroy()
    report("S11 six-parameter fit runs end to end", ok,
           "theta = %s; loss %.4f; gh_rest/g_pas %.2f; %d train / %d valid / %d report bundles"
           % (", ".join("%s=%.3g" % (k, fr.params[k]) for k in spec.names),
              fr.train_rmsd_mV, summ["gh_rest_over_gpas"], len(oi.train_bundles),
              len(oi.validation_bundles), len(oi.report_bundles)))


def _fit_for_bootstrap(d: Path, spec, *, n_calls: int, seed: int = 0):
    """A short fit on one archive cell, set up so Phase 3 can run on it."""
    import passive_fitting_hpc_fixed as mono
    import passive_long_step_training as plst
    import ih_mechanism as IM
    import param_spec as PS
    is_ih = spec.n > 3
    plst.integrate_long_step(mono, spec=spec, ih_protocol=is_ih,
                             ls_window_mode="step" if is_ih else "after_onset",
                             ls_window_ms_after_onset=60.0,
                             r_in_target="peak", weighting="relative",
                             ss_window_ms=(0.5, 100.0), ss_time_weight="exp",
                             ss_tau_w_ms=5.0, dt_brief_ms=0.1, dt_long_ms=0.1,
                             verbose=False)
    cd = mono.load_cell_from_archive(
        d, ls_max_amplitude_pA=(None if is_ih else 100.0),
        load_depolarising_ls=is_ih, verbose=False)
    oi = mono.prepare_optimiser_inputs(cd, fit_target="hyp")
    cell = mono.build_neuron_model(cd.swc_path, F=1.9)
    if is_ih:
        IM.attach_ih(cell, IM.IhSpec(mechanism="Ih_human", distribution="uniform"))
    fr = mono.fit_one_cell(cell, cd, oi, spec=spec, F=1.9, n_calls=n_calls,
                           n_initial=max(n_calls // 2, 4), seed=seed, verbose=False)
    return mono, cd, oi, cell, fr


def check_S8() -> None:
    import numpy as _np
    import param_spec as PS
    tmp = Path(tempfile.mkdtemp(prefix="smoke_s8_"))
    d = make_ih_archive(tmp, specimen_id=900000108)
    ok = True; notes = []

    # --- 3-D: the legacy shape, still three correctly named columns -------
    spec3 = PS.PASSIVE_3D
    mono, cd, oi, cell, fr = _fit_for_bootstrap(d, spec3, n_calls=12)
    b3 = mono.bootstrap_ci_for_cell(
        fit_result=fr, bootstrap_mode="nonparametric", B=14, fit_mode="fast",
        n_calls=8, n_initial=3, seed=1, fix_ra=False, verbose=False,
        pulse_pool=cd.ss_individual_pulses, root_dir=str(tmp),
        n_pulses_per_replicate=len(cd.ss_individual_pulses),
        save_pickle=False, save_plots=False)
    ok &= tuple(b3.param_names) == ("Cm", "Rm", "Ra")
    ok &= b3.samples.shape[1] == 3 and set(b3.ci_bca) == {"Cm", "Rm", "Ra"}
    ok &= _np.allclose(b3.samples, _np.exp(b3.samples_log))   # all-log spec
    notes.append("3-D: %s, samples %s, exp() still exact"
                 % (tuple(b3.param_names), b3.samples.shape))
    cell.destroy()

    # --- 6-D: six columns, and dv_h NOT exponentiated ---------------------
    spec6 = PS.make_ih_spec()
    mono, cd, oi, cell, fr = _fit_for_bootstrap(d, spec6, n_calls=14)
    b6 = mono.bootstrap_ci_for_cell(
        fit_result=fr, bootstrap_mode="nonparametric", B=14, fit_mode="fast",
        n_calls=8, n_initial=3, seed=1, fix_ra=False, verbose=False,
        pulse_pool=cd.ss_individual_pulses, root_dir=str(tmp),
        n_pulses_per_replicate=len(cd.ss_individual_pulses),
        save_pickle=False, save_plots=False)
    names = tuple(b6.param_names)
    ok &= names == spec6.names and b6.samples.shape[1] == 6
    ok &= set(b6.ci_bca) == set(spec6.names)
    i_dv = names.index("dv_h")
    lo, hi = spec6.axis("dv_h").lo, spec6.axis("dv_h").hi
    dv_col = b6.samples[:, i_dv]
    ok &= bool(_np.all(dv_col >= lo - 1e-9) and _np.all(dv_col <= hi + 1e-9))
    # ... and it is NOT the exponential of the q column
    ok &= not _np.allclose(dv_col, _np.exp(b6.samples_log[:, i_dv]))
    # the log axes ARE
    i_g = names.index("gbar")
    ok &= _np.allclose(b6.samples[:, i_g], _np.exp(b6.samples_log[:, i_g]))
    notes.append("6-D: %d columns; dv_h in [%.1f, %.1f] mV (range %.2f..%.2f), "
                 "not exp'd; gbar is" % (b6.samples.shape[1], lo, hi,
                                         float(dv_col.min()), float(dv_col.max())))
    cell.destroy()
    report("S8 Phase 3 follows the spec, linear axis included", ok, "; ".join(notes))


def check_S12() -> None:
    """Stage 6: the arm table, the spec builder, and one end-to-end run of the
    orchestrator that writes the three CSVs a campaign is read from."""
    import pandas as _pd
    import run_ih_fit as R
    ok = True; notes = []

    # --- (a) the arm table, PURE ------------------------------------------
    base = ["--archive-dir", "/a", "--output-dir", "/o", "--code-dir", "."]
    expect = {
        "baseline_runB":    (("Cm", "Rm", "Ra"), False, False, "after_onset",
                             60.0, 100.0, False, "legacy_early"),
        "passive_fullstep": (("Cm", "Rm", "Ra"), False, True, "step",
                             150.0, None, True, "same_window"),
        "ih6":              (R.IH6_NAMES, True, True, "step",
                             150.0, None, True, "same_window"),
    }
    for arm, exp in expect.items():
        c = R.resolve_arm_config(R._parse_args(base + ["--arm", arm]))
        got = (c.fit_params, c.attach_ih, c.ih_protocol, c.ls_window_mode,
               c.ls_window_ms, c.ls_max_amplitude_pA, c.load_depolarising_ls,
               c.gate_valid_via)
        if got != exp:
            ok = False; notes.append("arm %s: %s != %s" % (arm, got, exp))
    # --fit-params drops the mechanism with gbar, and brings it back with it
    c4 = R.resolve_arm_config(R._parse_args(
        base + ["--arm", "ih6", "--fit-params", "Cm,Rm,Ra,gbar"]))
    c3 = R.resolve_arm_config(R._parse_args(
        base + ["--arm", "ih6", "--fit-params", "Cm,Rm,Ra"]))
    if not (c4.attach_ih and not c3.attach_ih):
        ok = False; notes.append("--fit-params did not re-derive attach_ih")

    # --- (b) the spec builder accepts 4 shapes and rejects the rest --------
    bnd = dict(cm_bounds=(0.3, 3.0), rm_bounds=(1e3, 1e5), ra_bounds=(50., 1e3),
               gbar_bounds=(1e-6, 1e-3), dvh_bounds=(-10., 10.),
               kappa_bounds=(0.5, 2.0))
    for nm, n_exp in ((("Cm", "Rm", "Ra"), 3), (("Cm", "Rm", "Ra", "gbar"), 4),
                      (("Cm", "Rm", "Ra", "gbar", "dv_h"), 5),
                      (R.IH6_NAMES, 6)):
        if R.build_param_spec(nm, **bnd).n != n_exp:
            ok = False; notes.append("spec %s wrong n" % (nm,))
    for bad in (("Cm", "Ra", "Rm"), ("Cm", "Rm", "Ra", "dv_h"),
                ("Cm", "Rm", "Ra", "gbarr")):
        try:
            R.build_param_spec(bad, **bnd)
            ok = False; notes.append("accepted bad axis list %s" % (bad,))
        except ValueError:
            pass
    # a tau_w GRID must be refused, not silently collapsed to its first entry
    try:
        R._resolve_scalar_tau_w("3,5,7")
        ok = False; notes.append("a tau_w grid was accepted")
    except SystemExit:
        pass
    # skopt refuses n_calls < n_initial; the parser must, before any cell
    _base = ["--archive-dir", "/a", "--output-dir", "/o", "--code-dir", "."]
    for _bad in (["--n-calls", "20"],                        # 100 initial > 20
                 ["--bootstrap-n-calls", "10"]):             # 30 initial > 10
        try:
            R._parse_args(_base + _bad)
            ok = False; notes.append("parser accepted %s" % " ".join(_bad))
        except SystemExit:
            pass
    R._parse_args(_base + ["--n-calls", "20", "--n-initial", "10"])

    # --- (c) end to end: ih6 on one synthetic I_h cell ---------------------
    tmp = Path(tempfile.mkdtemp(prefix="smoke_s12_"))
    make_ih_archive(tmp, specimen_id=900000112, gbar=1.2e-4, dv_h=0.0, kappa=1.0)
    out = tmp / "out"
    # Capture every loss the orchestrator builds, to check the SS pulses are
    # trained on WITH their exponential time-weight -- asserted on the program
    # the campaign runs, not on a hand-built loss.
    import passive_long_step_training as _plst
    _orig_bmpl = _plst.build_multi_protocol_loss
    _built = []

    def _spy(cell, train_bundles, v_rest_mV, **kw):
        _built.append((list(train_bundles), kw.get("ss_sample_weight_fn"),
                       tuple(kw.get("ss_window_ms", ()))))
        return _orig_bmpl(cell, train_bundles, v_rest_mV, **kw)
    _plst.build_multi_protocol_loss = _spy
    try:
        R.main(["--archive-dir", str(tmp), "--output-dir", str(out),
                "--code-dir", str(HERE), "--arm", "ih6",
                "--n-calls", "14", "--n-initial", "8", "--phase3-subset", "none",
                "--dt-brief-ms", "0.1", "--dt-long-ms", "0.1"])
    finally:
        _plst.build_multi_protocol_loss = _orig_bmpl
    # (d) the SS pulses are in the training set, exponentially weighted
    if not _built:
        ok = False; notes.append("no loss was built through build_multi_protocol_loss")
    for _tb, _w, _win in _built:
        _ss = [b for b in _tb if _plst._is_brief(b) and b.polarity == "hyp"]
        _ls = [b for b in _tb if not _plst._is_brief(b)]
        if not (_ss and _ls):
            ok = False; notes.append("training set lacks SS hyp pulses or long "
                                     "steps (%d SS, %d LS)" % (len(_ss), len(_ls)))
        if _w is None:
            ok = False; notes.append("SS time-weight is None (uniform)")
            continue
        _t0 = _win[0] * 1e-3
        _v = _w(np.array([_t0, _t0 + 5e-3, _t0 + 50e-3]))
        if not (abs(_v[0] - 1.0) < 1e-12 and abs(_v[1] - np.exp(-1.0)) < 1e-12
                and abs(_v[2] - np.exp(-10.0)) < 1e-15):
            ok = False; notes.append("SS weight is not exp(-(t-t0)/5 ms): %s" % _v)
    if _built:
        notes.append("SS weight exp(-(t-%.1f ms)/5 ms) on %d SS hyp bundle(s) "
                     "beside %d long step(s), in all %d loss build(s)"
                     % (_built[0][2][0],
                        len([b for b in _built[0][0] if _plst._is_brief(b)]),
                        len([b for b in _built[0][0] if not _plst._is_brief(b)]),
                        len(_built)))

    res = _pd.read_csv(out / "phase2_results.csv")
    roles = _pd.read_csv(out / "ls_roles.csv")
    diag = _pd.read_csv(out / "ls_diagnostics.csv")

    # every fitted axis is a column, and named as the spec names it
    missing = [n for n in R.IH6_NAMES if n not in res.columns]
    if missing:
        ok = False; notes.append("theta columns missing: %s" % missing)
    if "rail_gbar" not in res.columns:
        ok = False; notes.append("rail_gbar missing (a railed gbar would be "
                                 "unreadable from the value alone)")
    # the derived I_h quantities are written EXACTLY once
    for col in ("m_inf_at_rest", "gh_rest_over_gpas", "e_pas_soma_mV",
                "rest_drift_mV"):
        if col not in res.columns:
            ok = False; notes.append("%s missing" % col)
        if ("ih_" + col) in res.columns:
            ok = False; notes.append("%s duplicated with an ih_ prefix" % col)
    if str(res["arm"].iloc[0]) != "ih6" or "Ih_human" not in str(res["ih_label"].iloc[0]):
        ok = False; notes.append("arm / ih_label not stamped on the row")
    # D-013: which arbours the fit was run on is a column, not a log line
    if str(res.get("morphology_source", _pd.Series([""])).iloc[0]) != "archive":
        ok = False; notes.append("morphology_source not 'archive' on a raw-SWC fit")
    if float(res["tau_w_chosen_ms"].iloc[0]) != 5.0 \
            or str(res["tau_w_reason"].iloc[0]) != "fixed_by_cli":
        ok = False; notes.append("tau_w provenance not recorded")
    # the fit actually simulated: below the crash penalty, not merely finite
    if float(res["train_rel_loss"].iloc[0]) >= _LOSS_CRASH_PENALTY:
        ok = False; notes.append("training loss sat at the 1e6 crash penalty")
    # the D-006 role split, audited
    got_roles = {(int(r.amplitude_pA), r.role) for r in roles.itertuples()}
    for amp, role in ((-10, "validate"), (-30, "train"), (-110, "train"),
                      (-150, "report"), (20, "validate"), (50, "validate")):
        if (amp, role) not in got_roles:
            ok = False; notes.append("role %s pA -> %s not recorded" % (amp, role))
    # one diagnostic row per LONG bundle, with model AND data sag side by side
    if len(diag) != len(roles):
        ok = False; notes.append("ls_diagnostics %d rows vs %d long bundles"
                                 % (len(diag), len(roles)))
    for col in ("exp_sag_fraction", "sim_sag_fraction", "sag_fraction_error",
                "rmsd_charge_mV", "rmsd_sag_mV", "rmsd_rebound_mV"):
        if col not in diag.columns or not np.isfinite(diag[col]).all():
            ok = False; notes.append("%s absent or non-finite" % col)
    rep = diag[diag["role"] == "report"]
    if len(rep) != 1:
        ok = False; notes.append("expected exactly one report-only step")

    notes.append("theta = %s" % ", ".join(
        "%s=%.4g" % (n, float(res[n].iloc[0])) for n in R.IH6_NAMES))
    notes.append("gh/g_pas=%.3f, drift=%.1e mV"
                 % (float(res["gh_rest_over_gpas"].iloc[0]),
                    float(res["rest_drift_mV"].iloc[0])))
    notes.append("report-only %+.0f pA: sag exp %.3f vs sim %.3f"
                 % (float(rep["amplitude_pA"].iloc[0]),
                    float(rep["exp_sag_fraction"].iloc[0]),
                    float(rep["sim_sag_fraction"].iloc[0])))
    report("S12 orchestrator: arms, spec builder, end-to-end CSVs", ok,
           "; ".join(notes))


def _ar1_noise(n: int, sigma: float, rho: float, rng) -> "np.ndarray":
    """Stationary AR(1) noise (scipy.signal.lfilter), for the S13 fixtures."""
    from scipy.signal import lfilter
    if rho == 0.0:
        return rng.normal(0.0, sigma, n)
    x = rng.normal(0.0, sigma * math.sqrt(1.0 - rho * rho), n)
    x[0] = rng.normal(0.0, sigma)
    return lfilter([1.0], [1.0, -rho], x)


def _passive_dep_sweep(fs: float, amp_mV: float = 15.0, tau_ms: float = 20.0,
                       onset_ms: float = 100.0, dur_ms: float = 1000.0,
                       post_ms: float = 200.0, v0: float = -73.5):
    """A clean subthreshold depolarising step response (single exponential),
    1.3 s long like the generator's Long Square sweep."""
    t = np.arange(0.0, (onset_ms + dur_ms + post_ms) * 1e-3, 1.0 / fs) * 1e3
    v = np.full(t.size, v0)
    on = (t >= onset_ms) & (t < onset_ms + dur_ms)
    v[on] += amp_mV * (1.0 - np.exp(-(t[on] - onset_ms) / tau_ms))
    off = t >= onset_ms + dur_ms
    v[off] += (amp_mV * (1.0 - math.exp(-dur_ms / tau_ms))
               * np.exp(-(t[off] - onset_ms - dur_ms) / tau_ms))
    return t, v, on


def _dep_archive(tmp: Path, sid: int, *, fs: float, sigma: float, rho: float) -> Path:
    """An ACTIVE archive cell with I_h, recorded at `fs` with AR(1) noise:
    +20 and +50 pA are subthreshold, +200 pA fires."""
    import synthetic_ground_truth as sgt
    swc = sgt.write_ball_and_stick_swc(tmp / ("dep_%d.swc" % sid), soma_r_um=10.0,
                                       dend_len_um=400.0, dend_r_um=1.0,
                                       apic_len_um=600.0, apic_r_um=1.2,
                                       step_um=20.0)
    ih = sgt.IhConfig(gIhbar_S_cm2=1.2e-4, ehcn_mV=-49.85, distribution="uniform",
                      mechanism="Ih_human", vshift_minf_mV=0.0, tau_scale=1.0)
    gt = sgt.GroundTruthParams(cm_uF_cm2=0.9, rm_Ohm_cm2=30000.0, ra_Ohm_cm=200.0,
                               e_pas_mV=-78.0, ih=ih, active=True,
                               active_regions=("soma", "axon"))
    proto = sgt.ProtocolConfig(ss_n_repeats=4, ls_hyp_amplitudes_pA=(-30.0, -70.0),
                               ls_dep_amplitudes_pA=(20.0, 50.0, 200.0),
                               ss_sampling_rate_Hz=fs, ls_sampling_rate_Hz=fs)
    syn = sgt.generate_synthetic_cell(swc, gt, proto=proto,
                                      noise=sgt.NoiseConfig(sigma_mV=sigma,
                                                            rho_lag1=rho, seed=sid % 1000),
                                      specimen_id=sid, verbose=False)
    d = tmp / ("specimen_%d" % sid)
    sgt.write_archive_cell(syn, d, verbose=False)
    sgt._clear_neuron_sections()
    return d


def check_S13() -> None:
    """D-016: the depolarising-sweep spike screen. Noise cannot pass it at
    any sampling rate the Allen data use, a bridge-balance step cannot, an
    action potential cannot evade it; the rule it replaced fails the noise
    case; the loader and the role assignment say WHY a sweep was dropped;
    check_dep_sweeps.py reports both screens."""
    import contextlib
    import io
    import passive_fitting_hpc_fixed as mono
    import passive_long_step_training as plst
    import check_dep_sweeps as CDS
    ok, notes = True, []
    rng = np.random.default_rng(20260923)

    # (a) noise alone, on a subthreshold step: L3_exc medians at 50 kHz, the
    # same process at 200 kHz (rho transferred as an OU: rho ** (1/4)), R8's
    # 100 kHz twin, and white 0.07 mV at 200 kHz
    cases = [(50e3, 0.0594, 0.668, False), (200e3, 0.0594, 0.668 ** 0.25, True),
             (100e3, 0.064, 0.80, True), (200e3, 0.07, 0.0, True)]
    worst_filt = 0.0
    for fs, sig, rho, legacy_must_flag in cases:
        new_flags = old_flags = 0
        for _ in range(8):
            _t, v, _on = _passive_dep_sweep(fs)
            v = v + _ar1_noise(v.size, sig, rho, rng)
            new_flags += int(mono.sweep_has_spike(v, fs))
            old_flags += int(mono.sweep_has_spike_legacy(v, fs))
            d, filt = mono.filtered_dvdt_mV_per_ms(v, fs)
            ok &= bool(filt)
            worst_filt = max(worst_filt, float(np.max(d)))
        ok &= new_flags == 0
        if legacy_must_flag:
            ok &= old_flags == 8
        notes.append("%.0f kHz noise: new %d/8, old %d/8 flagged"
                     % (fs / 1e3, new_flags, old_flags))
    ok &= worst_filt < 0.5 * mono.DEFAULT_SPIKE_DVDT_MV_PER_MS
    notes.append("max filtered dV/dt of noise %.1f mV/ms (threshold %g)"
                 % (worst_filt, mono.DEFAULT_SPIKE_DVDT_MV_PER_MS))

    # (b) a bridge-balance step: 3 mV instantaneous jump for the whole step,
    # at rest and on a step that plateaus at -28 mV. The second crosses the
    # dV/dt threshold AND has a sweep maximum above -30 mV, so only the 5 ms
    # time test keeps it out (without it -- max_interval huge -- it is an AP)
    for amp_mV in (15.0, 45.5):
        _t, v, on = _passive_dep_sweep(50e3, amp_mV=amp_mV)
        v = v + _ar1_noise(v.size, 0.0594, 0.668, rng)
        v[on] += 3.0
        why = mono.sweep_spike_reason(v, 50e3)
        why_nowin = mono.sweep_spike_reason(v, 50e3, max_interval_ms=1e9)
        ok &= why == "" and mono.sweep_has_spike_legacy(v, 50e3)
        if amp_mV > 40.0:
            ok &= why_nowin == "action_potential" and float(np.max(v)) > -30.0
        notes.append("3 mV bridge step, plateau %.0f mV: new '%s' (no time test: "
                     "'%s'), legacy flagged" % (-73.5 + amp_mV, why, why_nowin))
    # (b2) the height rule and the dV/dt < 0 rule, noise-free so each is
    # decided by the rule alone. A fast 1.8 mV step on a -28 mV plateau
    # crosses 20 mV/ms but rises < 2 mV: not an AP; 2.6 mV is. At 20 kHz (no
    # filter, raw differences): two 1.2 mV one-sample jumps with a flat run
    # between are ONE event of 2.4 mV (the second crossing is dropped: dV/dt
    # never fell below 0) -> AP; with a 0.1 mV dip between they are two
    # events of 1.2 mV -> not an AP.
    t, v0, _on = _passive_dep_sweep(50e3, amp_mV=45.5)
    for jump, want in ((1.8, ""), (2.6, "action_potential")):
        v = v0.copy()
        v[t >= 600.0] += jump
        d, _f = mono.filtered_dvdt_mV_per_ms(v, 50e3)
        ok &= float(np.max(d)) >= mono.DEFAULT_SPIKE_DVDT_MV_PER_MS
        ok &= mono.sweep_spike_reason(v, 50e3) == want
    v = np.full(400, -31.0)
    v[200:] += 1.2
    v[205:] += 1.2
    ok &= mono.sweep_spike_reason(v, 20e3) == "action_potential"
    v = np.full(400, -31.0)
    v[200:] += 1.2
    v[202:] -= 0.1
    v[205:] += 1.2
    ok &= mono.sweep_spike_reason(v, 20e3) == ""
    notes.append("height rule 1.8 / 2.6 mV and the dV/dt < 0 rule decide as specified")

    # (c) action potentials: one overshooting to ~+25 mV (reported as an AP,
    # not as the catch-all), one peaking at -25 mV (below the catch-all, so
    # only the event can catch it); at 50 kHz and at 20 kHz, where the 10 kHz
    # filter cannot apply (unfiltered path)
    for fs in (50e3, 20e3):
        for peak_mV in (25.0, -25.0):
            t, v, _on = _passive_dep_sweep(fs)
            base = float(v[np.argmin(np.abs(t - 600.0))])
            rise = (t >= 599.5) & (t < 600.0)
            fall = (t >= 600.0) & (t < 601.5)
            h = peak_mV - base
            v[rise] += h * (t[rise] - 599.5) / 0.5
            v[fall] += h * (1.0 - (t[fall] - 600.0) / 1.5)
            why = mono.sweep_spike_reason(v, fs)
            _d, filt = mono.filtered_dvdt_mV_per_ms(v, fs)
            ok &= why == "action_potential" and filt == (fs > 20e3)
            notes.append("AP peaking at %.1f mV, %.0f kHz: '%s' (filtered=%s)"
                         % (float(np.max(v)), fs / 1e3, why, filt))

    # (d) the loader on ACTIVE archive cells with realistic noise: +20/+50 kept,
    # +200 dropped (by the cap by default, as an action potential without it)
    tmp = Path(tempfile.mkdtemp(prefix="smoke_s13_"))
    d50 = _dep_archive(tmp, 900000131, fs=50e3, sigma=0.0594, rho=0.668)
    d100 = _dep_archive(tmp, 900000132, fs=100e3, sigma=0.064, rho=0.80)
    for dd in (d50, d100):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cd = mono.load_cell_from_archive(dd, ls_max_amplitude_pA=None,
                                             load_depolarising_ls=True, verbose=True)
            cd_nocap = mono.load_cell_from_archive(dd, ls_max_amplitude_pA=None,
                                                   load_depolarising_ls=True,
                                                   ls_dep_max_amplitude_pA=None,
                                                   verbose=False)
        amps = [round(b.amplitude_pA) for b in cd.long_square_depolarising]
        sc, sc2 = cd.ls_dep_screen, cd_nocap.ls_dep_screen
        ok &= amps == [20, 50]
        ok &= (sc.get("n_depolarising") == 3 and sc.get("n_above_cap") == 1
               and sc.get("n_kept") == 2)
        ok &= (sc2.get("n_kept") == 2 and sc2.get("n_above_cap") == 0
               and sc2.get("n_action_potential", 0) == 1
               and sc2.get("n_peak_above_threshold", 0) == 0)
        ok &= "2 kept" in buf.getvalue()
        notes.append("%s: dep kept %s; no cap -> +200 pA reported as %d AP"
                     % (dd.name, amps, sc2.get("n_action_potential", 0)))
    # (e) the role assignment names the cause
    import copy
    cd_none = copy.copy(cd)
    cd_none.long_square_depolarising = []
    cd_none.ls_dep_screen = dict(cd.ls_dep_screen, n_kept=0, n_action_potential=2)
    msgs = []
    cd_unasked = copy.copy(cd_none)
    cd_unasked.ls_dep_screen = {}
    cd_empty = copy.copy(cd_none)
    cd_empty.ls_dep_screen = {"n_depolarising": 0, "n_above_cap": 0, "n_kept": 0}
    for obj in (cd_none, cd_unasked, cd_empty):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            plst.assign_ls_roles(obj, verbose=True)
        msgs.append(buf.getvalue())
    ok &= ("screened out" in msgs[0] and "action potential" in msgs[0]
           and "not asked" in msgs[1] and "holds no depolarising" in msgs[2])
    notes.append("roles warning names the cause")
    # (f) check_dep_sweeps on the same two cells: the new screen keeps +20/+50
    # in both, the legacy screen loses the 100 kHz pair
    out_csv = tmp / "dep_screen.csv"
    with contextlib.redirect_stdout(io.StringIO()):
        rc = CDS.main(["--group-dir", str(tmp), "--out", str(out_csv),
                       "--code-dir", str(HERE)])
    import pandas as _pd
    df = _pd.read_csv(out_csv)
    ok &= rc == 0 and list(df.columns) == CDS.SWEEP_COLUMNS
    under = df[df["above_cap"].astype(str) == "False"]
    ok &= len(under) == 4 and (under["kept_new"].astype(str) == "True").all()
    ok &= (under["reason"] == "subthreshold").all()   # a plain CSV read, no NaN
    k100 = under[under["specimen_id"] == 900000132]
    ok &= (k100["kept_legacy"].astype(str) == "False").all()
    notes.append("check_dep_sweeps: new keeps %d/4, legacy keeps %d/4"
                 % ((under["kept_new"].astype(str) == "True").sum(),
                    (under["kept_legacy"].astype(str) == "True").sum()))
    report("S13 spike screen (D-016): noise and bridge steps pass, APs caught, "
           "causes reported", ok, "; ".join(notes))


def check_S10() -> None:
    files = ["ih_mechanism.py", "human_ih_params.py", "smoke_ih_fit.py",
             "regression_passive_identity.py", "synthetic_ground_truth.py",
             "param_spec.py", "passive_fitting_hpc_fixed.py",
             "passive_long_step_training.py",
             "run_ih_fit.py", "check_dep_sweeps.py",
             "mod/Ih.mod", "mod/Ih_human.mod"]
    bad = []
    for f in files:
        p = HERE / f
        if not p.exists():
            bad.append(f + " missing"); continue
        b = p.read_bytes()
        if f.endswith(".py") and any(c > 127 for c in b):
            bad.append(f + " non-ASCII")
        if b.count(b"\r"):
            bad.append(f + " CR bytes")
    report("S10 byte safety (ASCII .py, LF-only .py/.mod)", not bad, ", ".join(bad) if bad else "%d files clean" % len(files))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true", help="run nrnivmodl mod if x86_64/special is missing")
    args = ap.parse_args()
    ensure_build(args.build)
    from neuron import h  # noqa: F401  (loads ./x86_64)
    for fn in (check_S1, check_S2, check_S3, check_S4, check_S5,
               check_S6, check_S7, check_S8, check_S11, check_S12,
               check_S13, check_S10):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            report(fn.__name__, False, "raised %s: %s" % (type(e).__name__, e))
    n_ok = sum(RESULTS)
    print("smoke_ih_fit: %d/%d passed" % (n_ok, len(RESULTS)))
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
