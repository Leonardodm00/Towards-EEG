# -*- coding: utf-8 -*-
"""
smoke_ih_recovery.py -- Stage 7's own smoke suite (R1-R13).

Stage 7 produces a VERDICT, so its arithmetic has to be checked against
fixtures whose answer is known by construction, not merely inspected. Nine
checks:

    R1  the fourth RNG stream does not move the first three, and a legacy
        manifest still draws exactly what the untouched oracle draws
    R2  the Stage-7 manifest: ranges, the false-positive selection, the
        save/load round trip, and the back-fill of an old manifest
    R3  the generator plumbing: the knobs reach IhConfig, a legacy row still
        gives the published model, and the depolarising steps reach the
        protocol
    R4  the recovery arithmetic, on a fixture with a hand-computed answer,
        including the one that matters most: dv_h uses a DIFFERENCE and not a
        log ratio
    R5  the C_m-inflation read-out and the false-positive table
    R6  the gate, in all four of its outcomes
    R7  the consistency check refuses a truth/fit mismatch, item by item
    R8  end to end: draw, generate, fit two arms, report
    R9  byte safety

Run:  python smoke_ih_recovery.py            (add --build for nrnivmodl)
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from math import log
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
#: The untouched synthetic benchmark, used by R1 to prove a legacy manifest
#: still draws what it always drew. It sits beside this folder in the repo;
#: IH_ORACLE_DIR overrides that for a checkout where it does not.
ORACLE = Path(os.environ.get("IH_ORACLE_DIR",
                             str(HERE.parent / "Synthetic Passive fit Test")))

RESULTS = []


def report(name, ok, evidence):
    RESULTS.append(bool(ok))
    print("%s  %s  --  %s" % ("PASS" if ok else "FAIL", name, evidence),
          flush=True)


def ensure_build(build: bool) -> None:
    if (HERE / "x86_64" / "special").exists():
        return
    if not build:
        print("[smoke] x86_64/special missing; re-run with --build", flush=True)
        return
    subprocess.run(["nrnivmodl", "mod"], cwd=str(HERE), check=True)


def _swcs(tmp: Path, n: int = 2):
    """n distinct ball-and-stick morphologies, so the cohort has more than
    one and `_unique_morph_names` is exercised."""
    import synthetic_ground_truth as sgt
    out = []
    for i in range(n):
        d = tmp / ("m%d" % i)
        d.mkdir(parents=True, exist_ok=True)
        out.append(sgt.write_ball_and_stick_swc(
            d / "reconstruction.swc", soma_r_um=10.0,
            dend_len_um=300.0 + 100.0 * i, dend_r_um=1.0,
            apic_len_um=500.0 + 100.0 * i, apic_r_um=1.2, step_um=20.0))
    return out


def _archive_cell(tmp: Path, sid: int, *, sigma: float, rho: float,
                  ss_sigma: float = None, ss_rho: float = None,
                  ss_n: int = 8, fs: float = 50000.0, dend_len: float = 300.0):
    """A synthetic stand-in for a REAL archive cell: specimen_<sid>/ holding
    reconstruction.swc and the sweeps, generated with KNOWN per-sweep noise --
    (sigma, rho) on the Long Square sweeps and, when given, a DIFFERENT
    (ss_sigma, ss_rho) on the SS pulses, all at sampling rate `fs`.
    Passive, so it generates in seconds; the noise is what is being tested."""
    import synthetic_ground_truth as sgt
    import passive_fitting_hpc_fixed as mono
    swc_tmp = tmp / ("_swc_%d.swc" % sid)
    sgt.write_ball_and_stick_swc(swc_tmp, soma_r_um=10.0, dend_len_um=dend_len,
                                 dend_r_um=1.0, apic_len_um=500.0,
                                 apic_r_um=1.2, step_um=20.0)
    gt = sgt.GroundTruthParams(cm_uF_cm2=0.9, rm_Ohm_cm2=30000.0,
                               ra_Ohm_cm=200.0, e_pas_mV=-73.5)
    proto = sgt.ProtocolConfig(ss_n_repeats=ss_n,
                               ls_hyp_amplitudes_pA=(-30.0, -70.0, -110.0),
                               ss_sampling_rate_Hz=fs, ls_sampling_rate_Hz=fs)
    kw = {}
    if ss_sigma is not None:
        kw["ss_noise"] = sgt.NoiseConfig(
            sigma_mV=ss_sigma, rho_lag1=(rho if ss_rho is None else ss_rho),
            seed=sid % 1000)
    syn = sgt.generate_synthetic_cell(
        swc_tmp, gt, proto=proto,
        noise=sgt.NoiseConfig(sigma_mV=sigma, rho_lag1=rho, seed=sid % 1000),
        specimen_id=sid, passive_cell_factory=mono.build_neuron_model,
        verbose=False, **kw)
    d = tmp / ("specimen_%d" % sid)
    sgt.write_archive_cell(syn, d, verbose=False)
    sgt._clear_neuron_sections()
    return d


# ===========================================================================
def check_R1() -> None:
    """The new stream is APPENDED, so nothing drawn before Stage 7 moves."""
    import importlib.util
    import synth_gt_grid as G

    ss = np.random.SeedSequence(12345)
    three = [s.entropy if hasattr(s, "entropy") else s for s in
             np.random.SeedSequence(12345).spawn(3)]
    four = [s.entropy if hasattr(s, "entropy") else s for s in
            np.random.SeedSequence(12345).spawn(4)]
    same_children = all(
        np.array_equal(np.asarray(a.spawn_key), np.asarray(b.spawn_key))
        for a, b in zip(np.random.SeedSequence(12345).spawn(3),
                        np.random.SeedSequence(12345).spawn(4)))
    # the decisive form: the same draws come out of children 0..2
    d3 = [np.random.default_rng(s).random(5)
          for s in np.random.SeedSequence(12345).spawn(3)]
    d4 = [np.random.default_rng(s).random(5)
          for s in np.random.SeedSequence(12345).spawn(4)][:3]
    streams_equal = all(np.array_equal(a, b) for a, b in zip(d3, d4))

    # and the whole legacy manifest, against the untouched oracle
    tmp = Path(tempfile.mkdtemp(prefix="smoke_r1_"))
    swcs = _swcs(tmp, 3)
    legacy_cols = ["specimen_id", "cm_true", "rm_true", "ra_true",
                   "ih_gihbar_S_cm2", "noise_sigma_mV",
                   "noise_baseline_sigma_mV", "noise_drift_sigma_mV",
                   "noise_seed"]
    new = G.draw_manifest(swcs, seed=7, cells_per_cohort=2, use_ih=True)
    ok_oracle, note = True, "oracle not present; compared streams only"
    spec_path = ORACLE / "synth_gt_grid.py"
    if spec_path.exists():
        spec = importlib.util.spec_from_file_location("_oracle_grid", spec_path)
        Gold = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(Gold)
        old = Gold.draw_manifest(swcs, seed=7, cells_per_cohort=2, use_ih=True)
        diffs = []
        for c in legacy_cols:
            a = np.asarray(old[c], dtype=float)
            b = np.asarray(new[c], dtype=float)
            if not np.allclose(a, b, rtol=0, atol=0):
                diffs.append("%s: max|d|=%.3e" % (c, np.max(np.abs(a - b))))
        ok_oracle = not diffs
        note = ("every legacy column identical to the oracle over %d cells"
                % len(new)) if ok_oracle else ("; ".join(diffs))
    report("R1 the appended RNG stream moves nothing",
           bool(same_children and streams_equal and ok_oracle),
           "spawn(3) children == spawn(4)[:3]: %s; same draws: %s; %s"
           % (same_children, streams_equal, note))


def check_R2() -> None:
    import synth_gt_grid as G
    tmp = Path(tempfile.mkdtemp(prefix="smoke_r2_"))
    swcs = _swcs(tmp, 8)
    ok, notes = True, []

    df = G.draw_manifest(
        swcs, seed=3, cells_per_cohort=4, use_ih=True,
        ih_kinetics="Ih_human", ih_dist="uniform", ih_ehcn_mV=-49.85,
        ih_gbar_range_S_cm2=(2e-5, 3e-4), ih_dvh_range_mV=(-5.0, 5.0),
        ih_kappa_range=(0.7, 1.4), fp_control_frac=0.25,
        ih_gbar_floor_S_cm2=1e-6)

    # every Stage-7 column present
    for c in ("ih_regions", "ih_vshift_base_mV", "ih_dvh_mV", "ih_kappa_tau",
              "is_fp_control"):
        if c not in df.columns:
            ok = False; notes.append("column %s missing" % c)

    fp = df["is_fp_control"].to_numpy(dtype=bool)
    # selection is by INDEX: every 4th cell at frac 0.25
    expect_fp = (np.arange(len(df)) % 4) == 3
    if not np.array_equal(fp, expect_fp):
        ok = False; notes.append("FP selection %s != every-4th %s"
                                 % (fp.astype(int), expect_fp.astype(int)))
    notes.append("%d/%d cells are FP controls" % (fp.sum(), len(df)))

    # the FP cells carry the floor and the published kinetics, exactly
    g = df["ih_gihbar_S_cm2"].to_numpy(dtype=float)
    if not np.allclose(g[fp], 1e-6, rtol=0, atol=0):
        ok = False; notes.append("FP gbar %s != floor" % g[fp])
    if not (np.allclose(df["ih_dvh_mV"].to_numpy(dtype=float)[fp], 0.0)
            and np.allclose(df["ih_kappa_tau"].to_numpy(dtype=float)[fp], 1.0)):
        ok = False; notes.append("FP cells carry a non-trivial dv_h or kappa")

    # the I_h cells stay inside the drawn ranges
    d = df["ih_dvh_mV"].to_numpy(dtype=float)[~fp]
    k = df["ih_kappa_tau"].to_numpy(dtype=float)[~fp]
    gg = g[~fp]
    if not (np.all((d >= -5.0) & (d <= 5.0)) and np.all((k >= 0.7) & (k <= 1.4))
            and np.all((gg >= 2e-5) & (gg <= 3e-4))):
        ok = False; notes.append("a drawn value escaped its range")
    notes.append("dv_h in [%.2f, %.2f] mV, kappa in [%.2f, %.2f], "
                 "gbar in [%.2e, %.2e]"
                 % (d.min(), d.max(), k.min(), k.max(), gg.min(), gg.max()))
    # dv_h must be drawn UNIFORMLY (linear), not log-uniformly: a log draw
    # could not produce a negative value at all
    if not (d.min() < 0.0 < d.max()):
        ok = False; notes.append("dv_h never changes sign -- drawn in log?")

    # save -> load round trip. A CSV is DECIMAL TEXT: it does not promise
    # bit-exactness, and measurement here puts the loss at about one ulp
    # (~5e-17 relative). So the tolerance is 1e-12 relative, and the thing
    # that actually has to hold -- that every phase reads the FILE rather
    # than the in-memory draw -- is asserted by R8 through phase_manifest.
    p = tmp / "manifest.csv"
    G.save_manifest(df, p)
    back = G.load_manifest(p)
    worst = 0.0
    for c in ("ih_dvh_mV", "ih_kappa_tau", "ih_gihbar_S_cm2"):
        a = back[c].to_numpy(dtype=float); b = df[c].to_numpy(dtype=float)
        scale = np.maximum(np.abs(b), 1e-300)
        worst = max(worst, float(np.max(np.abs(a - b) / scale)))
        if not np.allclose(a, b, rtol=1e-12, atol=0.0):
            ok = False; notes.append("round trip changed %s beyond 1e-12" % c)
    if not np.array_equal(back["is_fp_control"].to_numpy(dtype=bool), fp):
        ok = False; notes.append("round trip changed is_fp_control")
    notes.append("CSV round trip worst relative change %.1e" % worst)

    # a PRE-Stage-7 manifest still loads, back-filled to the published model
    legacy = pd.read_csv(p).drop(columns=list(G._STAGE7_DEFAULTS))
    lp = tmp / "legacy.csv"
    legacy.to_csv(lp, index=False)
    lb = G.load_manifest(lp)
    if not (np.allclose(lb["ih_dvh_mV"].to_numpy(dtype=float), 0.0)
            and np.allclose(lb["ih_kappa_tau"].to_numpy(dtype=float), 1.0)
            and not lb["is_fp_control"].any()):
        ok = False; notes.append("legacy manifest did not back-fill to the "
                                 "published model")
    report("R2 manifest carries the Stage-7 ground truth", ok, "; ".join(notes))


def check_R3() -> None:
    import gen_from_manifest as GM
    import synthetic_ground_truth as sgt
    ok, notes = True, []

    legacy = dict(use_ih=True, ih_gihbar_S_cm2=2e-4, ih_ehcn_mV=-45.0,
                  ih_dist="uniform", ih_kinetics="Ih", cm_true=0.9,
                  rm_true=30000.0, ra_true=200.0, e_pas_mV=-70.0, F=1.9)
    a = GM.row_to_gt_kwargs(legacy)["ih"]
    if not (a["vshift_mV"] == 0.0 and a["vshift_minf_mV"] == 0.0
            and a["tau_scale"] == 1.0 and a["mechanism"] == "Ih"
            and "regions" not in a):
        ok = False; notes.append("legacy row no longer gives the published model")
    notes.append("legacy row -> published model")

    new = dict(legacy, ih_kinetics="Ih_human", ih_ehcn_mV=-49.85,
               ih_vshift_base_mV=20.0, ih_dvh_mV=-3.25, ih_kappa_tau=1.18,
               ih_regions="soma,dend", is_fp_control=False)
    b = GM.row_to_gt_kwargs(new)["ih"]
    if not (b["vshift_mV"] == 20.0 and b["vshift_minf_mV"] == -3.25
            and b["tau_scale"] == 1.18 and b["regions"] == ("soma", "dend")):
        ok = False; notes.append("Stage-7 row lost a knob: %s" % b)
    notes.append("stage7 row -> dv_h %.2f, kappa %.2f, vshift %.0f, regions %s"
                 % (b["vshift_minf_mV"], b["tau_scale"], b["vshift_mV"],
                    b["regions"]))
    # a NaN column (pandas writes NaN for a blank) must fall back, not crash
    c = GM.row_to_gt_kwargs(dict(new, ih_dvh_mV=float("nan"),
                                 ih_kappa_tau=float("nan")))["ih"]
    if not (c["vshift_minf_mV"] == 0.0 and c["tau_scale"] == 1.0):
        ok = False; notes.append("NaN knob did not fall back to the default")

    # the depolarising steps must reach ProtocolConfig -- without them the
    # D-006 validation set is empty
    proto = GM.build_proto(sgt, ss_n_repeats=4,
                           ls_hyp_amplitudes_pA=(-10., -50.),
                           ls_dep_amplitudes_pA=(20., 50.))
    if tuple(proto.ls_dep_amplitudes_pA) != (20.0, 50.0):
        ok = False; notes.append("ls_dep_amplitudes_pA not forwarded")
    legacy_proto = GM.build_proto(sgt, ss_n_repeats=4,
                                  ls_hyp_amplitudes_pA=(-10., -50.))
    if tuple(legacy_proto.ls_dep_amplitudes_pA) != ():
        ok = False; notes.append("default dep amps changed")
    notes.append("dep steps reach the protocol; default still empty")
    report("R3 the knobs reach the generator", ok, "; ".join(notes))


# ---------------------------------------------------------------------------
def _fixture():
    """A manifest and one arm's results whose every error is known by hand.

    4 cells; cell 4 is the false-positive control.
        Cm    truth 1.0 everywhere; est 1.25, 1.0, 0.8, 1.0
              -> log errors +log1.25, 0, -log1.25, 0
              -> over the 3 I_h cells: median |err| = log 1.25 EXACTLY,
                 median signed err = 0 (so the inflation factor is 1.0)
        dv_h  truth -4, 0, +4; est -1, +2, +4
              -> differences +3, +2, 0 -> median |err| = 2.0 mV.
              A log ratio is UNDEFINED here (truth 0 on cell 2), so a report
              that silently logged this axis cannot produce 2.0.
        gbar  truth 1e-4 thrice; est 1.5e-4, 1e-4, 1e-4 -> median |err| = 0
        FP    truth gbar 1e-6 (the box floor); est 1e-6 -> low rail
    """
    man = pd.DataFrame(dict(
        specimen_id=[1, 2, 3, 4],
        cm_true=[1.0, 1.0, 1.0, 1.0],
        rm_true=[30000.0] * 4, ra_true=[200.0] * 4,
        ih_gihbar_S_cm2=[1e-4, 1e-4, 1e-4, 1e-6],
        ih_dvh_mV=[-4.0, 0.0, 4.0, 0.0],
        ih_kappa_tau=[1.0, 1.0, 1.0, 1.0],
        is_fp_control=[False, False, False, True],
        use_ih=[True] * 4, ih_kinetics=["Ih_human"] * 4,
        ih_dist=["uniform"] * 4, ih_regions=["soma,dend,apic"] * 4,
        ih_vshift_base_mV=[0.0] * 4, ih_ehcn_mV=[-49.85] * 4))
    res = pd.DataFrame(dict(
        specimen_id=[1, 2, 3, 4],
        param_names=["Cm|Rm|Ra|gbar|dv_h|kappa_tau"] * 4,
        Cm=[1.25, 1.0, 0.8, 1.0],
        Rm=[30000.0] * 4, Ra=[200.0] * 4,
        gbar=[1.5e-4, 1e-4, 1e-4, 1e-6],
        dv_h=[-1.0, 2.0, 4.0, 0.0],
        kappa_tau=[1.0, 1.0, 1.0, 1.0],
        rail_gbar=[False, False, False, True],
        rail_Cm=[False] * 4, rail_Rm=[False] * 4, rail_Ra=[False] * 4,
        rail_dv_h=[False] * 4, rail_kappa_tau=[False] * 4,
        validation_status=["good"] * 4, arm=["ih6"] * 4))
    return man, res


def check_R4() -> None:
    import ih_recovery_report as RR
    man, res = _fixture()
    ok, notes = True, []
    long_df = RR.recovery_long(man, [res])
    s = RR.recovery_summary(long_df)

    def med(par):
        g = s[(s["arm"] == "ih6") & (s["parameter"] == par)]
        return (float(g["median_abs_err"].iloc[0]), int(g["n"].iloc[0]),
                str(g["err_kind"].iloc[0]))

    m_cm, n_cm, k_cm = med("Cm")
    if not (abs(m_cm - log(1.25)) < 1e-12 and n_cm == 3 and k_cm == "log_ratio"):
        ok = False; notes.append("Cm: %.6f over %d (%s), expected %.6f over 3"
                                 % (m_cm, n_cm, k_cm, log(1.25)))
    notes.append("Cm median |log ratio| = %.6f over %d I_h cells (FP excluded)"
                 % (m_cm, n_cm))

    m_dv, n_dv, k_dv = med("dv_h")
    if not (abs(m_dv - 2.0) < 1e-12 and n_dv == 3 and k_dv == "difference"):
        ok = False
        notes.append("dv_h: %.6f (%s), expected 2.0 mV as a DIFFERENCE"
                     % (m_dv, k_dv))
    notes.append("dv_h median |difference| = %.4f mV, not logged (a truth of "
                 "0 mV would make a log ratio undefined)" % m_dv)

    m_g, _, _ = med("gbar")
    if abs(m_g - 0.0) > 1e-12:
        ok = False; notes.append("gbar median %.6f, expected 0" % m_g)

    # the FP cell is excluded from the recovery, present in the long table
    n_rows_fp = int((long_df["is_fp_control"]).sum())
    if n_rows_fp != 6:
        ok = False; notes.append("FP cell has %d axis rows, expected 6" % n_rows_fp)

    # a box that disagrees with the fit's must REFUSE, not mislabel the rails
    raised = False
    try:
        RR.recovery_long(man, [res], box={**RR.DEFAULT_BOX, "gbar": (1e-8, 1e-3)})
    except ValueError:
        raised = True
    if not raised:
        ok = False; notes.append("a mismatched box did not refuse")
    notes.append("a box disagreeing with the fit's is refused")
    report("R4 recovery arithmetic matches the hand-computed fixture", ok,
           "; ".join(notes))


def check_R5() -> None:
    import ih_recovery_report as RR
    man, res = _fixture()
    ok, notes = True, []
    # a second arm: every I_h cell's Cm inflated by exactly 1.4
    res2 = res.copy()
    res2["arm"] = "passive_fullstep"
    res2["param_names"] = "Cm|Rm|Ra"
    res2["Cm"] = [1.4, 1.4, 1.4, 1.0]
    long_df = RR.recovery_long(man, [res, res2])
    infl = RR.cm_inflation(long_df)

    def fac(arm):
        return float(infl[infl["arm"] == arm]["median_factor"].iloc[0])

    if abs(fac("passive_fullstep") - 1.4) > 1e-9:
        ok = False; notes.append("passive inflation %.4f, expected 1.40"
                                 % fac("passive_fullstep"))
    if abs(fac("ih6") - 1.0) > 1e-9:
        ok = False; notes.append("ih6 inflation %.4f, expected 1.00" % fac("ih6"))
    notes.append("C_m factor: passive_fullstep %.3f, ih6 %.3f (I_h cells only)"
                 % (fac("passive_fullstep"), fac("ih6")))
    # the FP cell must not enter the inflation
    n = int(infl[infl["arm"] == "ih6"]["n"].iloc[0])
    if n != 3:
        ok = False; notes.append("inflation used %d cells, expected 3" % n)

    fp = RR.fp_control_summary(long_df)
    row = fp[fp["arm"] == "ih6"].iloc[0]
    if not (int(row["n_fp_cells"]) == 1 and float(row["low_rail_frac"]) == 1.0
            and abs(float(row["median_cm_abs_log_ratio"])) < 1e-12):
        ok = False; notes.append("FP table wrong: %s" % row.to_dict())
    notes.append("FP control: %d cell, gbar low-railed %.0f%%, C_m error %.1e"
                 % (row["n_fp_cells"], 100 * row["low_rail_frac"],
                    row["median_cm_abs_log_ratio"]))
    report("R5 C_m inflation and the false-positive table", ok, "; ".join(notes))


def check_R6() -> None:
    import ih_recovery_report as RR
    ok, notes = True, []
    tol = RR.RecoveryTolerances()

    def summ(cm, gbar, dvh, kappa):
        return pd.DataFrame([
            dict(arm="ih6", parameter=p, n=10, median_abs_err=v,
                 err_kind="x", median_err=v, q25_abs_err=v, q75_abs_err=v,
                 n_railed=0)
            for p, v in (("Cm", cm), ("gbar", gbar), ("dv_h", dvh),
                         ("kappa_tau", kappa))])

    fp_ok = pd.DataFrame([dict(arm="ih6", n_fp_cells=5, n_gbar_low_rail=5,
                               n_gbar_high_rail=0, low_rail_frac=1.0,
                               median_cm_abs_log_ratio=0.0,
                               median_cm_factor=1.0)])
    fp_bad = pd.DataFrame([dict(arm="ih6", n_fp_cells=5, n_gbar_low_rail=1,
                                n_gbar_high_rail=0, low_rail_frac=0.2,
                                median_cm_abs_log_ratio=0.0,
                                median_cm_factor=1.0)])

    # (a) everything inside tolerance -> pass
    v, p = RR.evaluate_gate(summ(0.10, 0.10, 1.0, 0.20), fp_ok, tol=tol)
    if not p:
        ok = False; notes.append("a passing cohort did not pass")
    # (b) kappa outside -> FAIL, and kappa is the freezable axis
    v, p = RR.evaluate_gate(summ(0.10, 0.10, 1.0, 0.90), fp_ok, tol=tol)
    frozen = RR.frozen_axes_on_failure(v)
    if p or frozen != ["kappa_tau"]:
        ok = False; notes.append("kappa failure: pass=%s frozen=%s" % (p, frozen))
    notes.append("kappa outside tolerance -> FAIL, freeze %s" % frozen)
    # (c) C_m outside -> FAIL and NOTHING is freezable
    v, p = RR.evaluate_gate(summ(0.90, 0.10, 1.0, 0.20), fp_ok, tol=tol)
    if p or RR.frozen_axes_on_failure(v) != []:
        ok = False; notes.append("a C_m failure offered a freeze")
    notes.append("C_m outside tolerance -> FAIL, nothing freezable")
    # (d) the false-positive control alone can fail the gate
    v, p = RR.evaluate_gate(summ(0.10, 0.10, 1.0, 0.20), fp_bad, tol=tol)
    if p:
        ok = False; notes.append("a failed FP control still passed")
    notes.append("FP low-rail 20%% -> FAIL")
    # (e) a required axis absent is a FAIL, never a silent pass
    part = summ(0.10, 0.10, 1.0, 0.20)
    part = part[part["parameter"] != "gbar"]
    v, p = RR.evaluate_gate(part, fp_ok, tol=tol)
    if p or "FAIL (axis absent)" not in set(v["verdict"]):
        ok = False; notes.append("a missing gbar row passed the gate")
    notes.append("a missing axis is a FAIL, not a pass")
    report("R6 the gate, in each of its outcomes", ok, "; ".join(notes))


def check_R7() -> None:
    import run_ih_recovery as RIR
    ok, notes = True, []
    man = pd.DataFrame(dict(
        use_ih=[True, True], ih_kinetics=["Ih_human"] * 2,
        ih_dist=["uniform"] * 2, ih_regions=["soma,dend,apic"] * 2,
        ih_vshift_base_mV=[0.0] * 2, ih_ehcn_mV=[-49.85] * 2))

    class A:
        ih_mechanism = "Ih_human"; ih_distribution = "uniform"
        ih_regions = "soma,dend,apic"; vshift_base = 0.0; ehcn = "-49.85"
    hard = [p for p in RIR.check_consistency(man, A) if not p.startswith("[note]")]
    if hard:
        ok = False; notes.append("a consistent setup was flagged: %s" % hard)
    notes.append("matched truth/fit -> no complaint")

    for attr, val, word in (("ih_mechanism", "Ih", "mechanism"),
                            ("ih_distribution", "eyal_exp_323", "spatial law"),
                            ("ih_regions", "soma,dend", "regions"),
                            ("vshift_base", 20.0, "vshift_base"),
                            ("ehcn", "-45.0", "E_h")):
        class B(A):
            pass
        setattr(B, attr, val)
        hits = [p for p in RIR.check_consistency(man, B)
                if not p.startswith("[note]")]
        if not any(word in h for h in hits):
            ok = False; notes.append("%s mismatch not caught (got %s)"
                                     % (word, hits))
    notes.append("every configured-object mismatch is caught by name")

    # the arm table resolves, and a typo is refused
    if RIR.parse_arms("ih6,ih4") != ["ih6", "ih4"]:
        ok = False; notes.append("parse_arms lost an arm")
    try:
        RIR.parse_arms("ih6,ih7")
        ok = False; notes.append("parse_arms accepted a typo")
    except SystemExit:
        pass
    if RIR.RECOVERY_ARMS["ih4"] != ("ih6", "Cm,Rm,Ra,gbar"):
        ok = False; notes.append("the 4-D arm is not ih6 minus the kinetics")
    notes.append("arms: %s" % ", ".join(sorted(RIR.RECOVERY_ARMS)))
    report("R7 truth/fit consistency is enforced before anything is generated",
           ok, "; ".join(notes))


def _estimator_band(n: int, m: int, rho: float, *, reps: int = 2000,
                    seed: int = 0, q: float = 0.001):
    """Sampling distribution of the calibration estimator under the
    generator's own noise law, for a window of `n` samples and `m` traces:
    median over traces of std(ddof=1), and of the lag-1 autocorrelation, of a
    stationary AR(1) with unit variance -- exactly what
    sgt.add_recording_noise injects and mono._estimate_noise reads.
    Returns ((sigma_lo, sigma_hi), (rho_lo, rho_hi), sigma_mean): the [q, 1-q]
    quantiles of sigma_hat/sigma and of rho_hat.

    Why not a fixed tolerance: demeaning a short, strongly correlated window
    biases sigma_hat LOW (the window mean absorbs part of the variance that
    ddof=1 only corrects for independent samples), and the scatter grows as
    rho -> 1. A 10 ms SS window at 50 kHz and rho = 0.9 has a mean bias of a
    few per cent and a 99.8 % band about 17 % wide; a hand-picked 6 % would
    fail a correct estimator. The band is the estimator's, not a choice."""
    from scipy.signal import lfilter
    rng = np.random.default_rng(seed)
    r = float(np.clip(rho, -0.999, 0.999))
    s_med = np.empty(reps); r_med = np.empty(reps)
    for k in range(reps):
        x = rng.normal(0.0, np.sqrt(1.0 - r * r), size=(m, n))
        x[:, 0] = rng.normal(0.0, 1.0, size=m)          # stationary start
        x = lfilter([1.0], [1.0, -r], x, axis=1)
        c = x - x.mean(axis=1, keepdims=True)
        s_med[k] = np.median(x.std(axis=1, ddof=1))
        r_med[k] = np.median((c[:, 1:] * c[:, :-1]).mean(axis=1)
                             / (c * c).mean(axis=1))
    return ((float(np.quantile(s_med, q)), float(np.quantile(s_med, 1 - q))),
            (float(np.quantile(r_med, q)), float(np.quantile(r_med, 1 - q))),
            float(s_med.mean()))


def _acf_gap_band(n: int, m: int, rho: float, lags, *, reps: int = 2000,
                  seed: int = 0, q: float = 0.001):
    """Sampling distribution, under the generator's AR(1) law, of the adequacy
    diagnostic noise_calibration writes for one protocol: the median over the
    m traces of the lag-L autocorrelation MINUS the median over traces of
    rho1_hat^L. Zero in expectation only asymptotically; this band is what a
    correct diagnostic on genuinely AR(1) noise of that window length and
    trace count produces. Returns {L: (lo, hi)}, the [q, 1-q] quantiles."""
    from scipy.signal import lfilter
    rng = np.random.default_rng(seed)
    r = float(np.clip(rho, -0.999, 0.999))
    gaps = {int(L): np.empty(reps) for L in lags}
    for k in range(reps):
        x = rng.normal(0.0, np.sqrt(1.0 - r * r), size=(m, n))
        x[:, 0] = rng.normal(0.0, 1.0, size=m)
        x = lfilter([1.0], [1.0, -r], x, axis=1)
        c = x - x.mean(axis=1, keepdims=True)
        v = (c * c).mean(axis=1)
        r1 = (c[:, 1:] * c[:, :-1]).mean(axis=1) / v
        for L in gaps:
            rl = (c[:, L:] * c[:, :-L]).mean(axis=1) / v
            gaps[L][k] = np.median(rl) - np.median(r1 ** L)
    return {L: (float(np.quantile(g, q)), float(np.quantile(g, 1 - q)))
            for L, g in gaps.items()}


def check_R10() -> None:
    """CLOSURE: inject known per-sweep (sigma, rho), write an archive, read it
    back with noise_calibration. The measurement is only worth anything if it
    is the inverse of the injection -- within the estimator's own sampling
    distribution at the window length and trace count actually used."""
    import passive_fitting_hpc_fixed as mono
    import noise_calibration as NC
    tmp = Path(tempfile.mkdtemp(prefix="smoke_r10_"))
    ok, notes = True, []
    for sid, sig, rho in ((900000701, 0.05, 0.0), (900000702, 0.08, 0.7),
                          (900000703, 0.12, 0.9)):
        d = _archive_cell(tmp, sid, sigma=sig, rho=rho)
        r = NC.measure_specimen_noise(d, mono=mono)
        # the window the estimator read, against the protocol that wrote it:
        # SS pre-pad 10 ms, LS onset 100 ms less the 5 ms guard, at 50 kHz
        if not (int(r["n_pre_ss"]) == 500 and int(r["n_pre_ls"]) == 4750
                and int(r["n_ls_single"]) == 3 and int(r["n_ss_pulses"]) == 16):
            ok = False
            notes.append("%s: windows %s/%s samples, %s LS sweeps, %s pulses "
                         "(expect 4750/500, 3, 16)" % (sid, r["n_pre_ls"],
                         r["n_pre_ss"], r["n_ls_single"], r["n_ss_pulses"]))
        nm = {"ls": (int(r["n_pre_ls"]), int(r["n_ls_single"])),
              "ss": (int(r["n_pre_ss"]), int(r["n_ss_pulses"]))}
        for proto in ("ls", "ss"):
            n_, m_ = nm[proto]
            (slo, shi), (rlo, rhi), smean = _estimator_band(
                n_, m_, rho, seed=sid % 1000)
            s_hat = float(r["sigma_%s_mV" % proto]) / sig
            r_hat = float(r["rho_%s" % proto])
            inside = (slo <= s_hat <= shi) and (rlo <= r_hat <= rhi)
            ok &= inside
            notes.append("%s rho %.1f: sigma x%.3f in [%.3f, %.3f] (mean %.3f), "
                         "rho %.3f in [%.3f, %.3f] over %d x %d samples%s"
                         % (proto.upper(), rho, s_hat, slo, shi, smean, r_hat,
                            rlo, rhi, m_, n_, "" if inside else "  <-- OUTSIDE"))
        if int(r["n_ss_hyp"]) != 8 or float(r["fs_ls_Hz"]) != 50000.0:
            ok = False
            notes.append("%s: SS per polarity %s (expect 8), LS fs %s"
                         % (sid, r["n_ss_hyp"], r["fs_ls_Hz"]))
        # the AR(1) adequacy diagnostic is SILENT on AR(1) noise: measured
        # minus AR(1)-implied autocorrelation inside its own sampling band
        lags = {NC._lag_tag(l): int(round(l * 1e-3 * 50000.0))
                for l in NC.ACF_LAGS_MS}
        for proto in ("ls", "ss"):
            n_, m_ = nm[proto]
            band = _acf_gap_band(n_, m_, rho, sorted(set(lags.values())),
                                 seed=sid % 1000 + 7)
            for tag, L in lags.items():
                gap = (float(r["acf_%s_%s" % (proto, tag)])
                       - float(r["ar1_%s_%s" % (proto, tag)]))
                lo, hi = band[L]
                if not (lo <= gap <= hi):
                    ok = False
                    notes.append("%s rho %.1f %s lag %s: acf-AR(1) gap %.3f "
                                 "outside [%.3f, %.3f]  <-- OUTSIDE"
                                 % (sid, rho, proto.upper(), tag, gap, lo, hi))
        notes.append("rho %.1f: AR(1) diagnostic gaps LS %s | SS %s (all in band)"
                     % (rho, " ".join("%+.3f" % (float(r["acf_ls_" + t])
                                                  - float(r["ar1_ls_" + t]))
                                      for t in lags),
                        " ".join("%+.3f" % (float(r["acf_ss_" + t])
                                            - float(r["ar1_ss_" + t]))
                                 for t in lags)))
    # the diagnostic's lag-1 IS the fitter's rho, sweep by sweep
    d = tmp / "specimen_900000703"
    cd = NC.load_for_noise(d, mono=mono)
    sweeps = ([b for b in cd.long_square_subthreshold
               if int(getattr(b, "n_repeats_averaged", 1)) == 1]
              + [NC._pulse_as_bundle(mono, p) for p in cd.ss_individual_pulses])
    worst = max(abs(NC.lag_autocorr(NC.pre_window_samples(mono, b), 1)
                    - mono._estimate_noise(b)[1]) for b in sweeps)
    if not worst <= 1e-12:
        ok = False
    notes.append("lag_autocorr(.,1) vs the fitter's rho over %d sweeps: worst "
                 "|diff| %.1e" % (len(sweeps), worst))
    # the generator's legacy path is untouched: no ss_noise == ss_noise=None
    # == the same law passed explicitly, bit for bit, SS and LS alike
    import synthetic_ground_truth as sgt
    swc = tmp / "_swc_legacy.swc"
    sgt.write_ball_and_stick_swc(swc, soma_r_um=10.0, dend_len_um=300.0,
                                 dend_r_um=1.0, apic_len_um=500.0,
                                 apic_r_um=1.2, step_um=20.0)
    gt = sgt.GroundTruthParams(cm_uF_cm2=0.9, rm_Ohm_cm2=30000.0,
                               ra_Ohm_cm=200.0, e_pas_mV=-73.5)
    pr = sgt.ProtocolConfig(ss_n_repeats=3, ls_hyp_amplitudes_pA=(-50.0,))
    law = sgt.NoiseConfig(sigma_mV=0.08, rho_lag1=0.7, seed=5)

    def _gen(**kw):
        syn = sgt.generate_synthetic_cell(
            swc, gt, proto=pr, noise=law, specimen_id=1,
            passive_cell_factory=mono.build_neuron_model, verbose=False, **kw)
        sgt._clear_neuron_sections()
        return ([np.asarray(q["v"]) for q in syn.ss_individual_pulses]
                + [np.asarray(b.v_mV) for b in syn.long_square_subthreshold])
    t0, t1, t2 = _gen(), _gen(ss_noise=None), _gen(ss_noise=law)
    same = (len(t0) == len(t1) == len(t2)
            and all(np.array_equal(a, b) and np.array_equal(a, c)
                    for a, b, c in zip(t0, t1, t2)))
    if not same:
        ok = False
    notes.append("generator: no ss_noise == ss_noise=None == same law explicit, "
                 "bit for bit over %d traces: %s" % (len(t0), same))
    # and the object the fitter REPORTS is not this one: an averaged SS
    # bundle carries sigma/sqrt(N), which is why run B's number is not used
    d = tmp / "specimen_900000702"
    cd = mono.load_cell_from_archive(d, verbose=False)
    avg = [mono._estimate_noise(b)[0] for b in cd.square_subthreshold
           if b.polarity == "hyp"]
    if not (avg and avg[0] < 0.08 / 2.0):
        ok = False; notes.append("averaged SS bundle sigma is not the "
                                 "per-sweep 0.08 / sqrt(8)")
    notes.append("averaged SS bundle reads %.4f mV against per-sweep 0.08 "
                 "(0.08/sqrt(8) = %.4f): the two objects differ by sqrt(N)"
                 % (avg[0] if avg else float("nan"), 0.08 / np.sqrt(8.0)))
    report("R10 noise calibration inverts the generator's injection", ok,
           "; ".join(notes))


def check_R11() -> None:
    """Each synthetic cell inherits the noise of the real cell whose
    morphology it borrows; a cell with no row falls back and says so; and
    switching the table on moves no ground-truth draw."""
    import synth_gt_grid as G
    tmp = Path(tempfile.mkdtemp(prefix="smoke_r11_"))
    ok, notes = True, []
    swcs = []
    for sid in (900000801, 900000802, 900000803):
        d = tmp / ("specimen_%d" % sid)
        d.mkdir()
        import synthetic_ground_truth as sgt
        swcs.append(sgt.write_ball_and_stick_swc(
            d / "reconstruction.swc", soma_r_um=10.0, dend_len_um=300.0,
            dend_r_um=1.0, apic_len_um=500.0, apic_r_um=1.2, step_um=20.0))
    # 801: LS + its own SS noise + acquisition; 802: LS + acquisition only
    # (its SS pulses inherit); 803: absent (nominal)
    table = {900000801: {"sigma_mV": 0.071, "rho_lag1": 0.62,
                         "ss_sigma_mV": 0.103, "ss_rho_lag1": 0.88,
                         "fs_ss_Hz": 200000.0, "fs_ls_Hz": 200000.0,
                         "ss_n_repeats": 12.0},
             900000802: {"sigma_mV": 0.094, "rho_lag1": 0.81,
                         "fs_ss_Hz": 50000.0, "fs_ls_Hz": 50000.0,
                         "ss_n_repeats": 20.0}}
    kw = dict(seed=5, cells_per_cohort=2, draws_per_morph=2, use_ih=True,
              ih_kinetics="Ih_human", ih_dist="uniform", ih_ehcn_mV=-49.85,
              ih_gbar_range_S_cm2=(2e-5, 3e-4), ih_dvh_range_mV=(-5.0, 5.0),
              ih_kappa_range=(0.7, 1.4), fp_control_frac=0.25)
    a = G.draw_manifest(swcs, noise_table=table, noise_rho_nominal=0.3, **kw)
    b = G.draw_manifest(swcs, **kw)                       # no table
    # per-cell assignment, by the specimen id in the morphology path
    for sid, want in table.items():
        rows = a[a["swc"].str.contains("specimen_%d" % sid)]
        if not (len(rows) == 2 and (rows["noise_source"] == "measured").all()
                and np.allclose(rows["noise_sigma_mV"], want["sigma_mV"])
                and np.allclose(rows["noise_rho_lag1"], want["rho_lag1"])):
            ok = False; notes.append("specimen %d not assigned its own noise" % sid)
    miss = a[a["swc"].str.contains("specimen_900000803")]
    if not ((miss["noise_source"] == "nominal").all()
            and np.allclose(miss["noise_rho_lag1"], 0.3)):
        ok = False; notes.append("a cell with no table row did not fall back")
    # D-014: the SS pulses' own noise and the acquisition twin
    r1 = a[a["swc"].str.contains("specimen_900000801")]
    r2 = a[a["swc"].str.contains("specimen_900000802")]
    if not ((r1["noise_ss_source"] == "measured").all()
            and np.allclose(r1["noise_ss_sigma_mV"], 0.103)
            and np.allclose(r1["noise_ss_rho_lag1"], 0.88)
            and np.allclose(r1["acq_fs_ss_Hz"], 200000.0)
            and np.allclose(r1["acq_ss_n_repeats"], 12.0)):
        ok = False; notes.append("801's SS noise / acquisition not carried")
    if not ((r2["noise_ss_source"] == "inherits").all()
            and r2["noise_ss_sigma_mV"].isna().all()
            and np.allclose(r2["acq_fs_ls_Hz"], 50000.0)):
        ok = False; notes.append("802 (no SS measurement) not labelled 'inherits'")
    if not ((miss["noise_ss_source"] == "inherits").all()
            and miss["acq_fs_ls_Hz"].isna().all()):
        ok = False; notes.append("a nominal cell acquired a twin's acquisition")
    notes.append("SS noise measured for %d, inherited for %d; acquisition twin "
                 "on %d cells" % ((a["noise_ss_source"] == "measured").sum(),
                                  (a["noise_ss_source"] == "inherits").sum(),
                                  a["acq_fs_ls_Hz"].notna().sum()))
    notes.append("%d measured, %d nominal (fallback labelled)"
                 % ((a["noise_source"] == "measured").sum(),
                    (a["noise_source"] == "nominal").sum()))
    # the table moves no ground-truth draw
    for c in ("cm_true", "rm_true", "ra_true", "ih_gihbar_S_cm2", "ih_dvh_mV",
              "ih_kappa_tau", "is_fp_control", "noise_seed"):
        if not np.array_equal(a[c].to_numpy(), b[c].to_numpy()):
            ok = False; notes.append("the noise table moved %s" % c)
    notes.append("every ground-truth column identical with and without the table")
    # a pre-calibration manifest loads as white, nominal noise
    pth = tmp / "m.csv"
    G.save_manifest(b, pth)
    legacy = pd.read_csv(pth).drop(columns=[
        "noise_rho_lag1", "noise_source", "noise_ss_sigma_mV",
        "noise_ss_rho_lag1", "noise_ss_source", "acq_ss_n_repeats",
        "acq_fs_ss_Hz", "acq_fs_ls_Hz"])
    legacy.to_csv(pth, index=False)
    back = G.load_manifest(pth)
    if not (np.allclose(back["noise_rho_lag1"], 0.0)
            and (back["noise_source"] == "nominal").all()
            and (back["noise_ss_source"] == "inherits").all()
            and back["noise_ss_sigma_mV"].isna().all()
            and back["acq_fs_ls_Hz"].isna().all()):
        ok = False; notes.append("legacy manifest did not back-fill to one "
                                 "white nominal law at the cohort protocol")
    # a noise table survives a CSV round trip: an empty `error` cell reads
    # back as NaN, which is truthy -- the lookup must not drop every cell
    import noise_calibration as NC
    nt = pd.DataFrame([{**{c: np.nan for c in NC.NOISE_TABLE_COLUMNS},
                        "specimen_id": 11, "sigma_ls_mV": 0.07, "rho_ls": 0.6,
                        "sigma_ss_mV": 0.09, "rho_ss": 0.8, "fs_ls_Hz": 5e4,
                        "fs_ss_Hz": 5e4, "n_ss_hyp": 10, "error": ""},
                       {**{c: np.nan for c in NC.NOISE_TABLE_COLUMNS},
                        "specimen_id": 12, "sigma_ls_mV": 0.08, "rho_ls": 0.7,
                        "error": ""},
                       {**{c: np.nan for c in NC.NOISE_TABLE_COLUMNS},
                        "specimen_id": 13, "sigma_ls_mV": 0.08, "rho_ls": 0.7,
                        "error": "OSError: unreadable"}],
                      columns=NC.NOISE_TABLE_COLUMNS)
    ntp = tmp / "noise_table.csv"
    nt.to_csv(ntp, index=False)
    nt_back = pd.read_csv(ntp)
    for mode in NC.NOISE_PROTOCOLS:
        mem, disk = NC.noise_lookup(nt, protocol=mode), NC.noise_lookup(nt_back, protocol=mode)
        if mem != disk:
            ok = False; notes.append("%s lookup changes across a CSV round "
                                     "trip: %s vs %s" % (mode, mem, disk))
    pp = NC.noise_lookup(nt_back)
    if not (sorted(pp) == [11, 12] and "ss_sigma_mV" in pp[11]
            and "ss_sigma_mV" not in pp[12] and pp[11]["ss_n_repeats"] == 10.0):
        ok = False; notes.append("per_protocol lookup after reload: %s" % pp)
    notes.append("noise table: lookup identical in memory and after a CSV "
                 "round trip in all %d modes; the errored cell dropped"
                 % len(NC.NOISE_PROTOCOLS))
    report("R11 each synthetic cell inherits its real cell's noise", ok,
           "; ".join(notes))


def check_R12() -> None:
    """rho and the sampling rates reach the generator; the protocol constants
    resolve CLI > measured > generator default, and say which."""
    import gen_from_manifest as GM
    import synthetic_ground_truth as sgt
    import run_ih_recovery as RIR
    ok, notes = True, []
    row = dict(noise_sigma_mV=0.07, noise_baseline_sigma_mV=0.07,
               noise_drift_sigma_mV=0.14, noise_seed=3, noise_rho_lag1=0.66)
    if GM.row_to_noise_kwargs(row)["rho_lag1"] != 0.66:
        ok = False; notes.append("rho_lag1 did not reach NoiseConfig")
    legacy = {k: v for k, v in row.items() if k != "noise_rho_lag1"}
    if GM.row_to_noise_kwargs(legacy)["rho_lag1"] != 0.0:
        ok = False; notes.append("a legacy row is no longer white noise")
    GM.build_noise(sgt, GM.row_to_noise_kwargs(row))       # constructs
    p = GM.build_proto(sgt, ss_n_repeats=5, ls_hyp_amplitudes_pA=(-50.0,),
                       ss_sampling_rate_Hz=200000.0, ls_sampling_rate_Hz=50000.0)
    q = GM.build_proto(sgt, ss_n_repeats=5, ls_hyp_amplitudes_pA=(-50.0,))
    d = sgt.ProtocolConfig()
    if not (p.ss_sampling_rate_Hz == 200000.0 and p.ls_sampling_rate_Hz == 50000.0
            and q.ss_sampling_rate_Hz == d.ss_sampling_rate_Hz
            and q.ls_sampling_rate_Hz == d.ls_sampling_rate_Hz):
        ok = False; notes.append("sampling rates not forwarded / defaults moved")
    notes.append("rho and both fs reach the generator; defaults unchanged")
    # D-014: the SS pulses' own noise, and the acquisition twin
    ssrow = dict(row, noise_ss_sigma_mV=0.11, noise_ss_rho_lag1=0.9)
    sk = GM.row_to_ss_noise_kwargs(ssrow)
    if not (sk and sk["sigma_mV"] == 0.11 and sk["rho_lag1"] == 0.9
            and sk["baseline_sigma_mV"] == 0.07 and sk["seed"] == 3):
        ok = False; notes.append("SS noise kwargs wrong: %s" % sk)
    if (GM.row_to_ss_noise_kwargs(row) is not None
            or GM.row_to_ss_noise_kwargs(dict(ssrow, noise_ss_rho_lag1=np.nan))
            is not None):
        ok = False; notes.append("a row without SS noise did not inherit")
    acq = GM.row_to_acquisition(dict(acq_ss_n_repeats=12.0,
                                     acq_fs_ss_Hz=200000.0,
                                     acq_fs_ls_Hz=np.nan))
    if acq != {"ss_n_repeats": 12, "ss_sampling_rate_Hz": 200000.0}:
        ok = False; notes.append("acquisition mapping wrong: %s" % acq)
    if GM.row_to_acquisition(row) != {}:
        ok = False; notes.append("a legacy row gained an acquisition")
    notes.append("SS noise kwargs and the acquisition twin map as specified")
    # CLI > measured, cell by cell; a forced rate that differs is reported
    look = {1: {"sigma_mV": .07, "rho_lag1": .6, "fs_ss_Hz": 2e5,
                "fs_ls_Hz": 2e5, "ss_n_repeats": 12.0},
            2: {"sigma_mV": .08, "rho_lag1": .7, "fs_ss_Hz": 5e4,
                "fs_ls_Hz": 5e4, "ss_n_repeats": 20.0}}
    base_ = ["--output-dir", "/o", "--code-dir", ".", "--morph-root", "/m"]
    same, w0 = RIR.apply_cli_acquisition(look, RIR._parse_args(base_))
    forced, w1 = RIR.apply_cli_acquisition(look, RIR._parse_args(
        base_ + ["--ss-sampling-rate-hz", "50000", "--ss-n-repeats", "8"]))
    if not (same == look and not w0
            and all("fs_ss_Hz" not in e and "ss_n_repeats" not in e
                    and "fs_ls_Hz" in e for e in forced.values())
            and len(w1) == 1 and "1 cell(s)" in w1[0]):
        ok = False; notes.append("CLI precedence per cell wrong: %s | %s"
                                 % (forced, w1))
    notes.append("CLI overrides the twin cell by cell; forcing 50 kHz on the "
                 "200 kHz cell is reported")
    base = ["--output-dir", "/o", "--code-dir", ".", "--morph-root", "/m"]
    meas = {"n_ss_per_polarity": 5.0, "fs_ss_Hz": 200000.0, "fs_ls_Hz": 50000.0}
    r1 = RIR.resolve_protocol(RIR._parse_args(base), meas)
    r2 = RIR.resolve_protocol(RIR._parse_args(base + ["--ss-n-repeats", "12"]), meas)
    r3 = RIR.resolve_protocol(RIR._parse_args(base), {})
    if not (r1["ss_n_repeats"] == 5 and r1["ss_n_repeats_source"] == "measured"
            and r2["ss_n_repeats"] == 12 and r2["ss_n_repeats_source"] == "cli"
            and r3["ss_n_repeats"] == 30
            and r3["ss_n_repeats_source"] == "generator default"
            and r1["ls_sampling_rate_Hz"] == 50000.0):
        ok = False; notes.append("protocol precedence wrong: %s | %s | %s"
                                 % (r1, r2, r3))
    notes.append("precedence cli > measured > default, source recorded")
    try:                           # skopt refuses n_calls < n_initial
        RIR._parse_args(base + ["--n-calls", "20"])
        ok = False; notes.append("n_initial 100 > n_calls 20 accepted")
    except SystemExit:
        notes.append("n_initial > n_calls refused at parse time")
    report("R12 rho and the real protocol reach the generator", ok,
           "; ".join(notes))


def check_R13() -> None:
    """The AR(1) adequacy diagnostic can FAIL: silent on an AR(1) trace, it
    flags a trace carrying the same fast AR(1) plus a slow component that an
    exponential fitted at lag 1 cannot represent. NumPy only."""
    import noise_calibration as NC
    from scipy.signal import lfilter
    ok, notes = True, []
    rng = np.random.default_rng(13)
    fs, n = 50000.0, 50000                   # 1 s window at 50 kHz
    lag = int(round(1.0e-3 * fs))            # 1 ms = 50 samples

    def ar1(rho, size, sd):
        e = rng.normal(0.0, sd * np.sqrt(1.0 - rho * rho), size)
        e[0] = rng.normal(0.0, sd)
        return lfilter([1.0], [1.0, -rho], e)
    fast = ar1(0.8, n, 0.05)                 # correlation time ~ 0.09 ms
    slow = ar1(np.exp(-1.0 / (0.005 * fs)), n, 0.05)   # OU, tau = 5 ms
    for label, x, want_flag in (("AR(1) alone", fast, False),
                                ("AR(1) + 5 ms OU", fast + slow, True)):
        r1 = NC.lag_autocorr(x, 1)
        acf, ar1_implied = NC.lag_autocorr(x, lag), r1 ** lag
        flagged = (acf - ar1_implied) > 0.1
        if flagged != want_flag:
            ok = False
        notes.append("%s: rho1 %.3f, acf(1 ms) %.3f vs AR(1) %.3f -> %s"
                     % (label, r1, acf, ar1_implied,
                        "FLAGGED" if flagged else "silent"))
    report("R13 the AR(1) adequacy diagnostic flags a second timescale", ok,
           "; ".join(notes))


def check_R8() -> None:
    """End to end on a 4-cell cohort at a toy budget, and a CLOSURE through
    the whole chain: two real-like cells recorded with KNOWN, protocol-
    specific noise at DIFFERENT sampling rates and SS pulse counts are
    measured; each synthetic twin is generated at its real cell's
    acquisition with per-protocol noise; two arms are fitted through the
    campaign's own entrypoint (SS pulses exponentially weighted); the report
    and gate are written -- and then the SYNTHETIC archives are measured
    again and must carry the noise and acquisition of their real twins."""
    import run_ih_recovery as RIR
    import noise_calibration as NC
    import passive_fitting_hpc_fixed as mono
    import passive_long_step_training as _plst
    tmp = Path(tempfile.mkdtemp(prefix="smoke_r8_"))
    real = {900000901: dict(ls=(0.07, 0.60), ss=(0.10, 0.80), fs=50000.0,
                            n=6, dl=300.0),
            900000902: dict(ls=(0.09, 0.75), ss=(0.06, 0.50), fs=100000.0,
                            n=4, dl=400.0)}
    for sid, c in real.items():
        _archive_cell(tmp, sid, sigma=c["ls"][0], rho=c["ls"][1],
                      ss_sigma=c["ss"][0], ss_rho=c["ss"][1],
                      ss_n=c["n"], fs=c["fs"], dend_len=c["dl"])
    out = tmp / "out"
    _orig = _plst.build_multi_protocol_loss
    _built = []

    def _spy(cell, train_bundles, v_rest_mV, **kw):
        _built.append((kw.get("ss_sample_weight_fn"),
                       tuple(kw.get("ss_window_ms", ())),
                       sum(1 for x in train_bundles if _plst._is_brief(x))))
        return _orig(cell, train_bundles, v_rest_mV, **kw)
    _plst.build_multi_protocol_loss = _spy
    try:
        rc = RIR.main([
            "--output-dir", str(out), "--code-dir", str(HERE),
            "--morph-root", str(tmp),
            "--morph-glob", "specimen_*/reconstruction.swc",
            "--draws-per-morph", "2", "--cells-per-cohort", "2",
            "--max-cells", "4", "--seed", "11", "--fp-control-frac", "0.25",
            # '=' form: argparse reads a bare '-10,...' as an option
            "--ls-hyp-amps=-10,-30,-50,-70,-90,-110,-150",
            "--ls-dep-amps=20,50",
            "--arms", "passive_fullstep,ih6",
            "--n-calls", "14", "--n-initial", "8",
            "--dt-brief-ms", "0.1", "--dt-long-ms", "0.1",
            "--phase3-subset", "none", "--no-plot"])
    finally:
        _plst.build_multi_protocol_loss = _orig
    ok, notes = True, []
    man = pd.read_csv(out / "manifest.csv")
    # phase_manifest must hand every later phase the frame it READ BACK from
    # this file, so a re-run with --manifest generates the same ground truth
    import synth_gt_grid as _G
    reread = _G.load_manifest(out / "manifest.csv")
    if not np.allclose(reread["ih_dvh_mV"].to_numpy(dtype=float),
                       man["ih_dvh_mV"].to_numpy(dtype=float),
                       rtol=0, atol=0, equal_nan=True):
        ok = False; notes.append("the written manifest does not re-read equal")
    if not (out / "noise_table.csv").exists():
        ok = False; notes.append("noise_table.csv not written")
    nt = pd.read_csv(out / "noise_table.csv")

    def _twin(swc):
        return next(s_ for s_ in real if ("specimen_%d" % s_) in str(swc))

    def _in_band(r_, proto, sig, rho, seed):
        n_ = int(r_["n_pre_" + proto])
        m_ = int(r_["n_ls_single"] if proto == "ls" else r_["n_ss_pulses"])
        (slo, shi), (rlo, rhi), _m = _estimator_band(n_, m_, rho, seed=seed)
        sh, rh = float(r_["sigma_%s_mV" % proto]) / sig, float(r_["rho_" + proto])
        return (slo <= sh <= shi) and (rlo <= rh <= rhi), sh, rh

    # (1) the REAL cells were read right: noise per protocol, and acquisition
    for sid, c in real.items():
        r_ = nt[nt["specimen_id"] == sid].iloc[0]
        for proto in ("ls", "ss"):
            inside, sh, rh = _in_band(r_, proto, c[proto][0], c[proto][1],
                                      sid % 1000)
            if not inside:
                ok = False; notes.append("real %d %s read (x%.3f, %.3f) outside "
                                         "its band" % (sid, proto, sh, rh))
        if not (float(r_["fs_ls_Hz"]) == c["fs"] == float(r_["fs_ss_Hz"])
                and int(r_["n_ss_hyp"]) == c["n"]):
            ok = False; notes.append("real %d acquisition read as %s/%s Hz, "
                                     "SS x%s" % (sid, r_["fs_ls_Hz"],
                                                 r_["fs_ss_Hz"], r_["n_ss_hyp"]))
    # (2) the manifest hands each twin its real cell's numbers, per protocol
    if not ((man["noise_source"] == "measured").all()
            and (man["noise_ss_source"] == "measured").all()):
        ok = False; notes.append("sources: %s / %s"
                                 % (man["noise_source"].value_counts().to_dict(),
                                    man["noise_ss_source"].value_counts().to_dict()))
    for _, mr in man.iterrows():
        sid = _twin(mr["swc"])
        r_ = nt[nt["specimen_id"] == sid].iloc[0]
        want = ((mr["noise_sigma_mV"], r_["sigma_ls_mV"]),
                (mr["noise_rho_lag1"], r_["rho_ls"]),
                (mr["noise_ss_sigma_mV"], r_["sigma_ss_mV"]),
                (mr["noise_ss_rho_lag1"], r_["rho_ss"]),
                (mr["acq_fs_ls_Hz"], real[sid]["fs"]),
                (mr["acq_fs_ss_Hz"], real[sid]["fs"]),
                (mr["acq_ss_n_repeats"], real[sid]["n"]))
        if not all(np.isclose(float(x), float(y), rtol=1e-12, atol=0)
                   for x, y in want):
            ok = False; notes.append("synthetic %d does not carry twin %d's "
                                     "measurement" % (mr["specimen_id"], sid))
    # (3) the SYNTHETIC archives: generated at the twin's acquisition, and
    # measured back at the twin's per-protocol noise
    n_checked = 0
    for _, mr in man.iterrows():
        sid = _twin(mr["swc"])
        d = out / "archive" / ("specimen_%d" % int(mr["specimen_id"]))
        r_ = NC.measure_specimen_noise(d, mono=mono)
        if not (float(r_["fs_ls_Hz"]) == real[sid]["fs"]
                and float(r_["fs_ss_Hz"]) == real[sid]["fs"]
                and int(r_["n_ss_hyp"]) == real[sid]["n"]):
            ok = False; notes.append("synthetic %d generated at %s Hz, SS x%s; "
                                     "its twin at %s Hz, SS x%s"
                                     % (mr["specimen_id"], r_["fs_ls_Hz"],
                                        r_["n_ss_hyp"], real[sid]["fs"],
                                        real[sid]["n"]))
        for proto, sig, rho in (("ls", mr["noise_sigma_mV"], mr["noise_rho_lag1"]),
                                ("ss", mr["noise_ss_sigma_mV"],
                                 mr["noise_ss_rho_lag1"])):
            inside, sh, rh = _in_band(r_, proto, float(sig), float(rho),
                                      int(mr["specimen_id"]) % 1000)
            if not inside:
                ok = False; notes.append("synthetic %d %s reads (x%.3f, %.3f) "
                                         "against injected (%.4f, %.3f)"
                                         % (mr["specimen_id"], proto, sh, rh,
                                            sig, rho))
        n_checked += 1
    notes.append("closure real -> measure -> twin -> measure: %d synthetic "
                 "archives at their twin's rate (50 / 100 kHz) and SS count "
                 "(6 / 4), LS and SS noise each in band" % n_checked)
    # (4) every fit trained the SS pulses with the exponential weight
    bad = [w for w, win, nss in _built
           if w is None or nss == 0
           or abs(w(np.array([win[0] * 1e-3 + 5e-3]))[0] - np.exp(-1.0)) > 1e-12]
    if not _built or bad:
        ok = False; notes.append("%d of %d loss builds lack the exp SS weight"
                                 % (len(bad), len(_built)))
    notes.append("SS pulses exp-weighted (tau_w 5 ms) in all %d loss builds"
                 % len(_built))
    # (5) the cohort protocol is only the fallback: median of the twins
    cfg = json.loads((out / "run_config.json").read_text())
    if not (cfg["protocol"]["ss_n_repeats"] == 5
            and cfg["protocol"]["ss_n_repeats_source"] == "measured"
            and cfg["noise_protocol"] == "per_protocol"):
        ok = False; notes.append("cohort fallback / mode: %s, %s"
                                 % (cfg["protocol"], cfg.get("noise_protocol")))
    if int(man["is_fp_control"].sum()) != 1:
        ok = False; notes.append("expected exactly 1 FP control, got %d"
                                 % man["is_fp_control"].sum())
    for name in ("recovery_per_cell.csv", "recovery_summary.csv",
                 "cm_inflation.csv", "fp_control.csv", "gate_verdict.csv",
                 "gate_verdict.txt"):
        if not (out / name).exists():
            ok = False; notes.append("%s not written" % name)
    if ok:
        per = pd.read_csv(out / "recovery_per_cell.csv")
        summ = pd.read_csv(out / "recovery_summary.csv")
        arms = sorted(per["arm"].unique())
        if arms != ["ih6", "passive_fullstep"]:
            ok = False; notes.append("arms in the report: %s" % arms)
        ih6 = summ[summ["arm"] == "ih6"]
        axes = sorted(ih6["parameter"].unique())
        if axes != ["Cm", "Ra", "Rm", "dv_h", "gbar", "kappa_tau"]:
            ok = False; notes.append("ih6 axes in the summary: %s" % axes)
        p3 = summ[(summ["arm"] == "passive_fullstep")]
        if sorted(p3["parameter"].unique()) != ["Cm", "Ra", "Rm"]:
            ok = False; notes.append("the 3-D arm reported I_h axes")
        dv = ih6[ih6["parameter"] == "dv_h"]
        if len(dv) and str(dv["err_kind"].iloc[0]) != "difference":
            ok = False; notes.append("dv_h reported as a log ratio end to end")
        infl = pd.read_csv(out / "cm_inflation.csv")
        notes.append("C_m factor " + ", ".join(
            "%s %.2f" % (r.arm, r.median_factor) for r in infl.itertuples()))
        notes.append("gate exit code %d (%s)"
                     % (rc, (out / "gate_verdict.txt").read_text().splitlines()[0]))
        notes.append("%d per-cell rows over %d cells" % (len(per), len(man)))
    report("R8 end to end: measure, twin, generate, fit two arms, report", ok,
           "; ".join(notes))


def check_R9() -> None:
    files = ["synth_gt_grid.py", "gen_from_manifest.py", "noise_calibration.py",
             "ih_recovery_report.py", "run_ih_recovery.py",
             "smoke_ih_recovery.py", "submit_ih_recovery.sh"]
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
    report("R9 byte safety (ASCII .py, LF-only)", not bad,
           ", ".join(bad) if bad else "%d files clean" % len(files))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="skip R8 (the only check that simulates)")
    args = ap.parse_args()
    ensure_build(args.build)
    from neuron import h  # noqa: F401  (loads ./x86_64)
    checks = [check_R1, check_R2, check_R3, check_R4, check_R5, check_R6,
              check_R7, check_R10, check_R11, check_R12, check_R13]
    if not args.quick:
        checks.append(check_R8)
    checks.append(check_R9)
    for fn in checks:
        try:
            fn()
        except BaseException as e:  # noqa: BLE001
            # BaseException, not Exception: argparse and several code paths
            # raise SystemExit, which is NOT an Exception. Catching only
            # Exception let a SystemExit inside a check kill the suite with
            # no FAIL line and a zero-ish looking transcript -- a failure that
            # does not look like one. KeyboardInterrupt is re-raised so Ctrl-C
            # still works.
            if isinstance(e, KeyboardInterrupt):
                raise
            report(fn.__name__, False, "raised %s: %s" % (type(e).__name__, e))
    n_ok = sum(RESULTS)
    print("smoke_ih_recovery: %d/%d passed" % (n_ok, len(RESULTS)))
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
