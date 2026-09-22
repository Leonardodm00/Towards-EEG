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
    S10 byte safety: pure ASCII in every .py, zero CR in .py/.mod/.sh
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


def check_S10() -> None:
    files = ["ih_mechanism.py", "human_ih_params.py", "smoke_ih_fit.py",
             "regression_passive_identity.py", "synthetic_ground_truth.py",
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
    for fn in (check_S1, check_S2, check_S3, check_S4, check_S5, check_S10):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            report(fn.__name__, False, "raised %s: %s" % (type(e).__name__, e))
    n_ok = sum(RESULTS)
    print("smoke_ih_fit: %d/%d passed" % (n_ok, len(RESULTS)))
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
