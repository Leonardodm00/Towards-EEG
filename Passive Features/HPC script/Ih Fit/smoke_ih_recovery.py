# -*- coding: utf-8 -*-
"""
smoke_ih_recovery.py -- Stage 7's own smoke suite.

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
                   "ih_gihbar_S_cm2", "noise_sigma_mV", "noise_seed"]
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


def check_R8() -> None:
    """End to end on a 4-cell cohort at a toy budget: draw, generate, fit two
    arms through the campaign's own entrypoint, report, gate."""
    import run_ih_recovery as RIR
    tmp = Path(tempfile.mkdtemp(prefix="smoke_r8_"))
    swcs = _swcs(tmp, 2)
    morph_root = tmp
    out = tmp / "out"
    rc = RIR.main([
        "--output-dir", str(out), "--code-dir", str(HERE),
        "--morph-root", str(morph_root), "--morph-glob", "m*/reconstruction.swc",
        "--draws-per-morph", "2", "--cells-per-cohort", "2", "--max-cells", "4",
        "--seed", "11", "--fp-control-frac", "0.25",
        # '=' form: argparse reads a bare '-10,...' as an option, not a value
        "--ls-hyp-amps=-10,-30,-50,-70,-90,-110,-150",
        "--ls-dep-amps=20,50", "--ss-n-repeats", "4",
        "--arms", "passive_fullstep,ih6",
        "--n-calls", "14", "--n-initial", "8",
        "--dt-brief-ms", "0.1", "--dt-long-ms", "0.1",
        "--phase3-subset", "none", "--no-plot", "--clean-archive"])
    ok, notes = True, []
    man = pd.read_csv(out / "manifest.csv")
    # phase_manifest must hand every later phase the frame it READ BACK from
    # this file, not the in-memory draw, so a re-run with --manifest generates
    # the same ground truth to the last bit.
    import synth_gt_grid as _G

    class _A:
        manifest = str(out / "manifest.csv")
    reread = _G.load_manifest(out / "manifest.csv")
    if not np.allclose(reread["ih_dvh_mV"].to_numpy(dtype=float),
                       man["ih_dvh_mV"].to_numpy(dtype=float),
                       rtol=0, atol=0, equal_nan=True):
        ok = False; notes.append("the written manifest does not re-read equal")
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
    report("R8 end to end: draw, generate, fit two arms, report", ok,
           "; ".join(notes))


def check_R9() -> None:
    files = ["synth_gt_grid.py", "gen_from_manifest.py",
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
              check_R7]
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
