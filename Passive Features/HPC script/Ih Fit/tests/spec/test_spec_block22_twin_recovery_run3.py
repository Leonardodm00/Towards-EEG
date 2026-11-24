"""Spec-derived tests of Blocks 20 (run_ih_fit.py, param_spec.py: the arm ih6_d207,
the refinement's wiring and columns) and 22 (run_twin_recovery.py,
submit_twin_recovery.sh: the third Stage-7 run), SPEC 2.9. Integration tester,
2026-11-24.

The end-to-end part runs the real entry point (`run_twin_recovery.py plan /
twin / merge`) on the writer-like fixture group of the Block 8 tests (4 real
cells, ball-and-stick morphologies, hand-written noise tables) at the smallest
budget, with the run's defaults (I_h arm ih6_d207, refinement nelder_mead) and a
small refinement budget. Expected values: SPEC 2.9 (a)-(f) and the entries'
Public API written out; m_inf(v) = 1 / (1 + exp((v - v_h - dv_h) / k)), v_h =
-90.87 mV, k = 8.05 mV (2.8 (a), 2.9 (a)); the fit's own CSVs for the columns
the merge copies; the refinement's bookkeeping from refine_starts.csv.
"""
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("neuron")
pytest.importorskip("skopt")

import param_spec as PS            # noqa: E402
import run_ih_fit as R             # noqa: E402
import run_twin_recovery as rtr    # noqa: E402
from test_spec_block8_twin_recovery import group, _results, TRUTH_COL  # noqa: E402,F401  (the fixture group)

CODE_DIR = Path(__file__).resolve().parents[2]
BASE = ["--archive-dir", "/nonexistent", "--output-dir", "/nonexistent", "--code-dir", str(CODE_DIR)]
SMALL22 = ["--n-per-configuration", "1", "--n-no-noise", "1", "--n-calls", "4", "--n-initial", "4",
           "--refine-maxfev", "6", "--refine-starts", "2"]
IDS4 = (900000000, 900000100, 900000200, 900000300)
IH6 = ("Cm", "Rm", "Ra", "gbar", "dv_h", "kappa_tau")


def _minf(v, dvh=0.0):
    return 1.0 / (1.0 + math.exp((v - (-90.87) - dvh) / 8.05))


def _plan(group, out, *extra):
    return rtr.main(["plan", "--group-dir", str(group["gdir"]), "--noise-dir", str(group["ndir"]), "--out", str(out)]
                    + SMALL22 + list(extra))


# ============================================================================ Block 20, pure

def test_b20_arm_table_and_defaults():
    a = R.ARMS
    assert a["ih6_d207"].fit_params == IH6 == R.IH6_NAMES
    assert (a["ih6_d207"].role_rule, a["ih6_d207"].weighting, a["ih6_d207"].ls_sample_weight,
            a["ih6_d207"].refine) == ("d207", "ls_amplitude", "peak_decay", "nelder_mead")
    assert a["ih5"].refine == "nelder_mead" and a["passive_fullstep"].refine == "nelder_mead"
    assert a["ih6"].refine == "none" and a["baseline_runB"].refine == "none"
    # ih6_d207 is ih6 apart from the name, the protocol and the refinement
    for f in a["ih6"].__dataclass_fields__:
        if f not in ("name", "role_rule", "weighting", "ls_sample_weight", "refine"):
            assert getattr(a["ih6_d207"], f) == getattr(a["ih6"], f), f
    p = R._parse_args(BASE + ["--arm", "ih6_d207"])
    assert (p.refine, p.refine_starts, p.refine_maxfev, p.n_workers) == (None, 3, 100, 1)
    assert p.gbar_bounds == "1e-5,1e-3" and PS.DEFAULT_GBAR_BOUNDS == (1e-5, 1e-3)
    spec = R.build_spec_from_args(p, R.resolve_arm_config(p))
    assert spec.names == IH6
    g, d = spec.axis("gbar"), spec.axis("dv_h")
    assert (g.lo, g.hi, g.log) == (1e-5, 1e-3, True)
    assert (d.lo, d.hi, d.log) == (-10.0, 10.0, False)
    assert R.resolve_arm_config(R._parse_args(BASE + ["--arm", "ih6", "--refine", "nelder_mead"])).refine == \
        "nelder_mead"
    assert R.resolve_arm_config(R._parse_args(BASE + ["--arm", "ih5", "--refine", "none"])).refine == "none"
    cfg = R.refine_config_from_args(R._parse_args(BASE + ["--arm", "ih5", "--refine-starts", "2",
                                                          "--refine-maxfev", "7"]), R.ARMS["ih5"])
    assert (cfg.n_starts, cfg.maxfev_per_start, cfg.min_start_dist_unit, cfg.simplex_step_unit, cfg.xatol_unit,
            cfg.fatol_rel) == (2, 7, 0.1, 0.05, 1e-3, 1e-4)
    assert R.refine_config_from_args(p, R.ARMS["ih6"]) is None


@pytest.mark.parametrize("extra", [["--refine", "bfgs"], ["--refine-starts", "0"], ["--refine-maxfev", "0"],
                                   ["--n-workers", "0"], ["--n-workers", "1.5"], ["--arm", "ih7"]])
def test_b20_cli_refusals(extra):
    with pytest.raises(SystemExit) as e:
        R._parse_args(BASE + ["--arm", "ih6_d207"] + extra)
    assert e.value.code == 2


def test_b20_refine_record_rules():
    class GP:
        pass
    class FR:
        pass
    fr = FR()
    fr.gp_result = None
    rec = R.refine_record(fr, R.ARMS["ih6_d207"], 3)
    assert rec["refine"] == "nelder_mead" and math.isnan(rec["loss_after_search"])
    assert math.isnan(rec["loss_after_refine"]) and rec["n_workers"] == 3
    fr.gp_result = GP()
    fr.gp_result.fun = 0.25
    rec = R.refine_record(fr, R.ARMS["ih6"], 1)
    assert (rec["refine"], rec["loss_after_search"], rec["loss_after_refine"], rec["refine_improved"],
            rec["refine_n_starts"], rec["refine_n_evals"], rec["refine_best_start"]) == \
        ("none", 0.25, 0.25, False, 0, 0, -1)
    assert tuple(R.REFINE_RECORD_COLUMNS) == ("refine", "loss_after_search", "loss_after_refine", "refine_improved",
                                              "refine_n_starts", "refine_n_evals", "refine_best_start", "n_workers")


def test_b20_eval_theta_six_names():
    th = R.parse_eval_theta("Cm=1,Rm=2e4,Ra=100,gbar=1e-4,dv_h=-2.5,kappa_tau=1", R.ARMS["ih6_d207"].fit_params)
    assert th["dv_h"] == -2.5
    R._parse_args(BASE + ["--arm", "ih6_d207", "--eval-theta=Cm=1,Rm=2e4,Ra=100,gbar=1e-4,dv_h=-2.5,kappa_tau=1"])
    with pytest.raises(SystemExit):
        R._parse_args(BASE + ["--arm", "ih6_d207", "--eval-theta", "Cm=1,Rm=2e4,Ra=100,gbar=1e-4,kappa_tau=1"])


# ============================================================================ Block 22, pure

def test_b22_plan_parser_defaults_and_gate_functions():
    ap = rtr.build_parser()
    a = ap.parse_args(["plan", "--group-dir", "g", "--noise-dir", "n", "--out", "o"])
    assert a.ih_arm == "ih6_d207" and a.dvh_truth_range_mV == "-5,5"
    assert a.gbar_bounds == "1e-5,1e-3" and a.gbar_floor == 1e-5
    assert (a.refine, a.refine_starts, a.refine_maxfev) == ("nelder_mead", 3, 100)
    assert a.gbar_range == "2e-5,3e-4"                       # 2.9 (d): the truth's range unchanged
    assert a.tol_dvh_mV == 3.0                               # 2.9 (f): D-011's 3 mV
    assert not hasattr(a, "n_workers")                       # not a plan option
    t = ap.parse_args(["twin", "--out", "o", "--twin", "1"])
    assert t.n_workers == 1
    for bad in ("0", "-2", "x"):
        with pytest.raises(SystemExit):
            ap.parse_args(["twin", "--out", "o", "--twin", "1", "--n-workers", bad])
    assert rtr.gate_arm("ih6_d207") == "ih6_d207" and rtr.gate_arm("ih5") == "ih5"
    assert tuple(rtr.gate_required_axes("ih6_d207")) == ("Cm", "gbar", "dv_h", "kappa_tau")
    assert tuple(rtr.gate_required_axes("ih5")) == ("Cm", "gbar", "kappa_tau")
    with pytest.raises(ValueError):
        rtr.gate_arm("ih6")


def test_b22_without_n_workers():
    argv = ["--arm", "ih5", "--n-workers", "3", "--refine", "none"]
    assert rtr.without_n_workers(argv) == ["--arm", "ih5", "--refine", "none"]
    assert rtr.without_n_workers(["--n-workers=2", "--x", "1"]) == ["--x", "1"]


@pytest.mark.parametrize("extra,code", [
    (["--gbar-floor", "1e-6"], 3), (["--gbar-floor", "2e-3"], 3),
    (["--dvh-truth-range-mV=-12,5"], 3), (["--dvh-truth-range-mV=-5,11", "--dvh-bounds=-10,10"], 3),
    (["--dvh-truth-range-mV=-5,5", "--dvh-bounds=-4,4"], 3),
    (["--refine-maxfev", "0"], 2), (["--refine", "bfgs"], 2), (["--ih-arm", "ih6"], 2),
])
def test_b22_plan_refusals(group, tmp_path, extra, code):
    out = tmp_path / "r"
    assert _plan(group, out, *extra) == code
    assert not (out / "manifest.csv").exists()


def test_b22_plan_ih5_and_range_ignored(group, tmp_path):
    out = tmp_path / "p5"
    # under ih5 the dv_h range is not used (dv_h 0), so an out-of-box range is not a refusal
    assert _plan(group, out, "--ih-arm", "ih5", "--seed", "3") == 0
    rc = json.loads((out / "run_config.json").read_text())
    assert rc["ih_arm"] == "ih5" and rc["gate_arm"] == "ih5" and rc["dvh_truth_mV"] == 0.0
    assert rc["gate_required_axes"] == ["Cm", "gbar", "kappa_tau"]
    assert rc["truth_kwargs"]["ih_dvh_range_mV"] is None
    man = pd.read_csv(out / "manifest.csv", float_precision="round_trip")
    assert set(man["arm"]) == {"ih5", "passive_fullstep"} and (man["ih_dvh_mV"] == 0.0).all()


# ============================================================================ end to end (step 6)

@pytest.fixture(scope="module")
def e2e22(group, tmp_path_factory):
    """Plan with the defaults; the four twins built and fitted, the full twin with --n-workers 2; merge. Then a
    second run directory with the same plan, every twin with --n-workers 1."""
    runs = []
    for name, workers in (("w2", 2), ("w1", 1)):
        out = tmp_path_factory.mktemp("e22") / name
        assert _plan(group, out, "--seed", "4") == 0
        codes = [rtr.main(["twin", "--out", str(out), "--twin", str(t), "--n-workers", str(workers)])
                 for t in IDS4]
        assert rtr.main(["merge", "--out", str(out), "--no-plot"]) == 0
        runs.append((out, codes, workers))
    return runs


def test_e2e_plan_records(e2e22):
    out, codes, _ = e2e22[0]
    assert codes == [0, 0, 0, 0], codes
    rc = json.loads((out / "run_config.json").read_text())
    assert rc["ih_arm"] == "ih6_d207" and rc["gate_arm"] == "ih6_d207"
    assert rc["gate_required_axes"] == ["Cm", "gbar", "dv_h", "kappa_tau"]
    assert rc["dvh_truth_mV"] is None
    o = rc["options"]
    assert (o["gbar_bounds"], o["gbar_floor"], o["refine"], o["dvh_truth_range_mV"]) == \
        ("1e-5,1e-3", 1e-5, "nelder_mead", "-5,5")
    tk = rc["truth_kwargs"]
    assert tk["ih_dvh_range_mV"] == [-5.0, 5.0] and tk["ih_gbar_floor_S_cm2"] == 1e-5
    assert tk["ih_gbar_range_S_cm2"] == [2e-5, 3e-4]
    man = pd.read_csv(out / "manifest.csv", float_precision="round_trip")
    assert list(man["arm"]) == ["ih6_d207", "passive_fullstep", "ih6_d207", "ih6_d207"]
    fu, pf, af, nn = (man.iloc[i] for i in range(4))
    assert -5.0 <= fu["ih_dvh_mV"] <= 5.0 and fu["ih_dvh_mV"] != 0.0
    assert -5.0 <= pf["ih_dvh_mV"] <= 5.0
    assert af["ih_dvh_mV"] == 0.0 and af["ih_gihbar_S_cm2"] == 1e-5
    assert nn["ih_dvh_mV"] == fu["ih_dvh_mV"]
    for r in (fu, pf):
        assert 2e-5 <= r["ih_gihbar_S_cm2"] <= 3e-4
    # boundary Block 21 -> 12: the twin is built at the manifest's truth, dv_h included (archive metadata)
    for _, r in man.iterrows():
        tid = int(r["specimen_id"])
        meta = json.loads((out / "twins" / ("twin_%d" % tid) / ("specimen_%d" % tid) / "metadata.json").read_text())
        th = meta["twin"]["theta"]
        if r["configuration"] != "active_fit":
            assert th["dv_h"] == float(r["ih_dvh_mV"]) and th["gbar"] == float(r["ih_gihbar_S_cm2"])
        else:
            assert "dv_h" not in th


def test_e2e_fit_command_line(e2e22):
    out, _, workers = e2e22[0]
    man = pd.read_csv(out / "manifest.csv", float_precision="round_trip")
    for _, row in man.iterrows():
        tid = int(row["specimen_id"])
        rec = json.loads((out / "fits" / ("twin_%d" % tid) / "fit_argv.json").read_text())
        argv = rec["argv"]
        assert argv[argv.index("--arm") + 1] == row["arm"]
        assert "--gbar-bounds=1e-5,1e-3" in argv or argv[argv.index("--gbar-bounds") + 1] == "1e-5,1e-3"
        assert argv[argv.index("--refine") + 1] == "nelder_mead"
        assert argv[argv.index("--refine-starts") + 1] == "2" and argv[argv.index("--refine-maxfev") + 1] == "6"
        assert argv[argv.index("--n-workers") + 1] == str(workers)
        ev = [x for x in argv if x.startswith("--eval-theta")]
        if row["configuration"] in ("full", "full_no_noise"):
            th = dict(kv.split("=") for kv in ev[0].split("=", 1)[1].split(","))
            assert list(th) == list(IH6)
            assert float(th["dv_h"]) == float(row["ih_dvh_mV"])
        else:
            assert ev == []
        R._parse_args(argv)


def test_e2e_phase2_refinement_columns(e2e22):
    out, _, workers = e2e22[0]
    man = pd.read_csv(out / "manifest.csv", float_precision="round_trip")
    for _, row in man.iterrows():
        tid = int(row["specimen_id"])
        fd = out / "fits" / ("twin_%d" % tid)
        res = pd.read_csv(fd / "phase2_results.csv", float_precision="round_trip")
        r = res[res["specimen_id"] == tid].iloc[0]
        cols = list(res.columns)
        # Block 20's columns, after Block 15's, in the stated order
        i0 = cols.index("refine")
        assert cols[i0:i0 + 8] == list(R.REFINE_RECORD_COLUMNS)
        assert cols.index("gh_at_v_deepest_S_cm2") < i0
        names = r["param_names"].split("|")
        exp = ["Cm", "Rm", "Ra"] if row["configuration"] == "passive_fit" else list(IH6)
        assert names == exp
        assert r["refine"] == "nelder_mead" and int(r["n_workers"]) == workers
        ls, lr = r["loss_after_search"], r["loss_after_refine"]
        assert np.isfinite(ls) and lr <= ls
        # 2.9 (e): train_rel_loss and loss_at_estimate are at q_hat; 1e-9 relative (the entry's F2)
        assert r["train_rel_loss"] == pytest.approx(lr, rel=1e-9)
        assert r["loss_at_estimate"] == pytest.approx(lr, rel=1e-9)
        st = pd.read_csv(fd / "refine_starts.csv", float_precision="round_trip")
        assert list(st.columns) == (["specimen_id", "start", "search_index", "f0", "fun", "nfev", "nit", "status"]
                                    + ["x0_" + n for n in names] + ["x_" + n for n in names])
        assert len(st) == int(r["refine_n_starts"]) and 1 <= len(st) <= 2
        assert int(r["refine_n_evals"]) == int(st["nfev"].sum())
        assert (st["f0"].min() == ls)                    # the first start is the search's best
        assert list(st["start"]) == list(range(len(st)))
        if bool(r["refine_improved"]):
            k = int(r["refine_best_start"])
            assert 0 <= k < len(st)
            assert st.loc[k, "fun"] == lr and lr < ls
            for n in names:                              # the estimate in physical units is the best start's
                assert st.loc[k, "x_" + n] == pytest.approx(r[n], rel=1e-12, abs=1e-12)
        else:
            assert int(r["refine_best_start"]) == -1 and lr == ls
        # the x0 of the first start is a point of the box, physical units
        for n in names:
            ax = PS.make_ih_spec(gbar_bounds=(1e-5, 1e-3)).axis(n) if n != "dv_h" else None
        if "gbar" in names:
            assert (st["x0_gbar"] >= 1e-5 * (1 - 1e-12)).all() and (st["x0_gbar"] <= 1e-3 * (1 + 1e-12)).all()
            assert 1e-5 * (1 - 1e-12) <= r["gbar"] <= 1e-3 * (1 + 1e-12)
        if "dv_h" in names:
            assert -10.0 <= r["dv_h"] <= 10.0


def test_e2e_workers_do_not_change_the_fit(e2e22):
    (a, _, _), (b, _, _) = e2e22
    for t in IDS4:
        ra = pd.read_csv(a / "fits" / ("twin_%d" % t) / "phase2_results.csv", float_precision="round_trip")
        rb = pd.read_csv(b / "fits" / ("twin_%d" % t) / "phase2_results.csv", float_precision="round_trip")
        skip = {"n_workers", "wall_time_s"}
        cols = [c for c in ra.columns if c not in skip and "seconds" not in c and "time" not in c]
        pd.testing.assert_frame_equal(ra[cols], rb[cols])
        sa = pd.read_csv(a / "fits" / ("twin_%d" % t) / "refine_starts.csv", float_precision="round_trip")
        sb = pd.read_csv(b / "fits" / ("twin_%d" % t) / "refine_starts.csv", float_precision="round_trip")
        pd.testing.assert_frame_equal(sa, sb)


def test_e2e_rerun_with_other_workers_does_not_refit(e2e22):
    out, _, _ = e2e22[0]
    tid = IDS4[0]
    p = out / "fits" / ("twin_%d" % tid) / "phase2_results.csv"
    m0 = os.path.getmtime(p)
    assert rtr.main(["twin", "--out", str(out), "--twin", str(tid), "--n-workers", "3"]) == 0
    assert os.path.getmtime(p) == m0
    st = json.loads((out / "status" / ("twin_%d.json" % tid)).read_text())
    assert st["status"] == "ok"


def test_e2e_merge_extras_and_gate(e2e22):
    out, _, _ = e2e22[0]
    man = pd.read_csv(out / "manifest.csv", float_precision="round_trip")
    ex = pd.read_csv(out / "recovery_extras.csv", float_precision="round_trip")
    for c in ("loss_after_search", "refine_improved", "refine_n_evals", "refine_best_start", "dvh_true_mV",
              "dvh_est_mV"):
        assert c in ex.columns
    for _, e in ex.iterrows():
        tid = int(e["twin_id"])
        m = man[man["specimen_id"] == tid].iloc[0]
        r = pd.read_csv(out / "fits" / ("twin_%d" % tid) / "phase2_results.csv", float_precision="round_trip").iloc[0]
        assert e["loss_after_search"] == r["loss_after_search"]
        assert bool(e["refine_improved"]) == bool(r["refine_improved"])
        assert int(e["refine_n_evals"]) == int(r["refine_n_evals"])
        assert int(e["refine_best_start"]) == int(r["refine_best_start"])
        conf = m["configuration"]
        if conf == "passive_fit":
            assert math.isnan(e["dvh_true_mV"]) and math.isnan(e["dvh_est_mV"])
            continue
        assert e["dvh_true_mV"] == m["ih_dvh_mV"] and e["dvh_est_mV"] == r["dv_h"]
        v = e["v_deepest_train_mV"]
        if conf != "active_fit":
            # 2.9 (e): g_h(v*) at the truth with the truth's own dv_h
            assert e["gh_deep_true_S_cm2"] == pytest.approx(m["ih_gihbar_S_cm2"] * _minf(v, m["ih_dvh_mV"]),
                                                            rel=1e-12)
        # at the estimate with the estimate's dv_h (the fit's column)
        assert r["gh_at_v_deepest_S_cm2"] == pytest.approx(r["gbar"] * _minf(v, r["dv_h"]), rel=1e-12)
        assert e["gh_deep_est_S_cm2"] == r["gh_at_v_deepest_S_cm2"]
    ver = pd.read_csv(out / "report" / "gate_verdict.csv")
    sub = ver[ver["arm"] == "ih6_d207"]
    assert set(["Cm", "gbar", "dv_h", "kappa_tau"]) <= set(sub["parameter"])
    assert float(sub.loc[sub["parameter"] == "dv_h", "tolerance"].iloc[0]) == 3.0


def test_e2e_two_runs_identical_manifest_and_twins(e2e22):
    (a, _, _), (b, _, _) = e2e22
    for f in ("manifest.csv", "twin_sweeps_G.csv"):
        pd.testing.assert_frame_equal(pd.read_csv(a / f), pd.read_csv(b / f))
    ea = pd.read_csv(a / "recovery_extras.csv", float_precision="round_trip")
    eb = pd.read_csv(b / "recovery_extras.csv", float_precision="round_trip")
    pd.testing.assert_frame_equal(ea, eb)


# ============================================================================ T6: the gate on synthetic frames

@pytest.mark.parametrize("dv_err,passes", [(5.0, False), (0.8, True)])
def test_b22_gate_requires_dvh(group, tmp_path, dv_err, passes):
    out = tmp_path / "g"
    assert _plan(group, out, "--seed", "4") == 0
    man = pd.read_csv(out / "manifest.csv", float_precision="round_trip")
    for _, row in man.iterrows():
        tid = int(row["specimen_id"])
        conf = row["configuration"]
        if conf == "passive_fit":
            names = ("Cm", "Rm", "Ra")
        else:
            names = IH6
        vals = {n: float(row[TRUTH_COL[n]]) for n in names}
        if conf in ("full", "full_no_noise"):
            vals["dv_h"] = float(row["ih_dvh_mV"]) + dv_err
        if conf == "active_fit":
            vals["gbar"] = 1e-5
        _results(out, tid, names, vals, row["arm"])
    assert rtr.main(["merge", "--out", str(out), "--no-plot"]) == 0
    ver = pd.read_csv(out / "report" / "gate_verdict.csv")
    sub = ver[(ver["arm"] == "ih6_d207") & (ver["parameter"] == "dv_h")]
    assert len(sub) == 1
    assert str(sub["verdict"].iloc[0]).lower().startswith("pass") == passes, sub.to_dict("records")
    for a in ("Cm", "gbar", "kappa_tau"):
        v = ver[(ver["arm"] == "ih6_d207") & (ver["parameter"] == a)]
        assert str(v["verdict"].iloc[0]).lower().startswith("pass"), (a, v.to_dict("records"))


# ============================================================================ the submitter (text and bash -n)

def test_b22_submitter_text_and_syntax():
    p = CODE_DIR / "submit_twin_recovery.sh"
    raw = p.read_bytes()
    assert all(c < 128 for c in raw) and b"\r" not in raw
    txt = raw.decode()
    assert 'NCPUS="${NCPUS:-3}"' in txt
    assert '--n-workers "$4"' in txt
    subprocess.run(["bash", "-n", str(p)], check=True)
    for f in ("run_twin_recovery.py", "run_ih_fit.py", "local_refine.py", "param_spec.py",
              "passive_long_step_training.py", "twin_manifest.py"):
        raw = (CODE_DIR / f).read_bytes()
        assert all(c < 128 for c in raw) and b"\r" not in raw, f


# ============================================================================ swappability: the refinement off

def test_swap_refine_none_through_the_plan(group, tmp_path):
    """Directive audit (swappability): the refinement switched off through the plan's option; the rest of the
    pipeline runs and the columns take the values the Block 20 entry states for 'refinement off'."""
    out = tmp_path / "rn"
    assert _plan(group, out, "--seed", "4", "--refine", "none") == 0
    tid = 900000100                                   # passive_fit: the cheapest fit
    assert rtr.main(["twin", "--out", str(out), "--twin", str(tid)]) == 0
    fd = out / "fits" / ("twin_%d" % tid)
    r = pd.read_csv(fd / "phase2_results.csv", float_precision="round_trip").iloc[0]
    assert r["refine"] == "none" and not bool(r["refine_improved"])
    assert (int(r["refine_n_starts"]), int(r["refine_n_evals"]), int(r["refine_best_start"])) == (0, 0, -1)
    assert r["loss_after_refine"] == r["loss_after_search"] == pytest.approx(r["train_rel_loss"], rel=1e-15)
    st = pd.read_csv(fd / "refine_starts.csv")
    assert len(st) == 0
    assert list(st.columns) == (["specimen_id", "start", "search_index", "f0", "fun", "nfev", "nit", "status"]
                                + ["x0_" + n for n in ("Cm", "Rm", "Ra")] + ["x_" + n for n in ("Cm", "Rm", "Ra")])
    assert rtr.main(["merge", "--out", str(out), "--no-plot"]) == 0
