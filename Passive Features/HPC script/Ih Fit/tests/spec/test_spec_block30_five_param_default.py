"""Spec-derived tests of Block 30 (SPEC 2.9 (d) as amended, the Block 30 entry):
five parameters by default (D-282) and the five-parameter fit on twins whose
truth has a dv_h (plan --ih5-dvh-truth-range-mV). Integration tester,
2026-11-28 (incremental run, base 8b05c56, commit e05b5ec).

Expected values come from the entry written out (defaults, exit codes, the
recorded key), from synth_gt_grid.draw_manifest called directly with the
arguments the Block 13 entry states (the helper of the Block 21 tests), from
the ih6_d207 manifest of the same arguments (the oracle the entry names), and
from m_inf(v) = 1 / (1 + exp((v - v_h - dv_h) / k)), v_h = -90.87 mV, k = 8.05 mV
(2.8 (a), 2.9 (a)) for g_h(v*) at the truth.
"""
import inspect
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

import twin_manifest as tm
from test_spec_block21_twin_manifest_ih_arm import _real, _draw_ref, TRUTH_COLS, ID_BASE  # noqa: F401

CODE_DIR = Path(__file__).resolve().parents[2]
IH5 = ("Cm", "Rm", "Ra", "gbar", "kappa_tau")
KW = {"ih_dvh_range_mV": (-5.0, 5.0)}


# ============================================================================ Block 30 / twin_manifest (pure)

def test_manifest_default_arm_is_ih5():
    assert tm.DEFAULT_IH_ARM == "ih5"
    sig = inspect.signature(tm.draw_twin_manifest)
    assert sig.parameters["ih_arm"].default == "ih5"
    assert sig.parameters["allow_dvh_truth_with_ih5"].default is False
    df = tm.draw_twin_manifest(_real(), n_per_configuration=3, n_no_noise=2, seed=1)
    exp = {"full": "ih5", "passive_fit": "passive_fullstep", "active_fit": "ih5", "full_no_noise": "ih5"}
    assert list(df["arm"]) == [exp[c] for c in df["configuration"]]
    assert (df["ih_dvh_mV"] == 0.0).all()


def test_flag_lifts_refusal_and_changes_nothing_else():
    real = _real()
    ref6 = tm.draw_twin_manifest(real, n_per_configuration=4, n_no_noise=2, seed=7, ih_arm="ih6_d207",
                                 dvh_truth_mV=None, truth_kwargs=KW)
    got = tm.draw_twin_manifest(real, n_per_configuration=4, n_no_noise=2, seed=7, ih_arm="ih5",
                                dvh_truth_mV=None, allow_dvh_truth_with_ih5=True, truth_kwargs=KW)
    # every column but arm equal, bit for bit (the entry's oracle)
    cols = [c for c in ref6.columns if c != "arm"]
    assert list(got.columns) == list(ref6.columns)
    pd.testing.assert_frame_equal(got[cols], ref6[cols])
    exp = {"full": "ih5", "passive_fit": "passive_fullstep", "active_fit": "ih5", "full_no_noise": "ih5"}
    assert list(got["arm"]) == [exp[c] for c in got["configuration"]]
    # independent reference: the drawer called directly (Blocks 13 / 21 seed rule) for full and passive_fit
    for q, conf in enumerate(("full", "passive_fit")):
        cells, t = _draw_ref(real, 4, 7, q, KW)
        sub = got[got["configuration"] == conf]
        assert_array_equal(sub["twin_of_specimen_id"].to_numpy(), cells)
        assert_array_equal(sub["ih_dvh_mV"].to_numpy(), t["ih_dvh_mV"].to_numpy())
    full = got[got["configuration"] == "full"]
    dv = full["ih_dvh_mV"].to_numpy()
    assert np.all((dv >= -5.0) & (dv <= 5.0)) and len(set(dv)) > 1 and np.all(dv != 0.0)
    # active_fit: dv_h 0 whatever the options; full_no_noise copies its partner's dv_h
    assert (got.loc[got["configuration"] == "active_fit", "ih_dvh_mV"] == 0.0).all()
    nn = got[got["configuration"] == "full_no_noise"]
    for _, r in nn.iterrows():
        partner = got[got["specimen_id"] == r["paired_with"]].iloc[0]
        for c in TRUTH_COLS:
            assert r[c] == partner[c], c


def test_flag_false_refuses_as_before():
    with pytest.raises(ValueError, match="ih5"):
        tm.draw_twin_manifest(_real(), n_per_configuration=2, n_no_noise=1, seed=0, ih_arm="ih5",
                              dvh_truth_mV=None, truth_kwargs=KW)


def test_flag_with_fixed_value_and_with_ih6():
    real = _real()
    a = tm.draw_twin_manifest(real, n_per_configuration=2, n_no_noise=1, seed=0, ih_arm="ih5",
                              dvh_truth_mV=2.5, allow_dvh_truth_with_ih5=True)
    has_ih = a["configuration"].map(tm.HAS_IH).to_numpy(dtype=bool)
    assert (a.loc[has_ih, "ih_dvh_mV"] == 2.5).all() and (a.loc[~has_ih, "ih_dvh_mV"] == 0.0).all()
    # under ih6_d207 the flag changes nothing
    b1 = tm.draw_twin_manifest(real, n_per_configuration=2, n_no_noise=1, seed=0, ih_arm="ih6_d207",
                               truth_kwargs=KW)
    b2 = tm.draw_twin_manifest(real, n_per_configuration=2, n_no_noise=1, seed=0, ih_arm="ih6_d207",
                               truth_kwargs=KW, allow_dvh_truth_with_ih5=True)
    pd.testing.assert_frame_equal(b1, b2)
    # under ih5 with dv_h 0 the flag changes nothing either
    c1 = tm.draw_twin_manifest(real, n_per_configuration=2, n_no_noise=1, seed=0, ih_arm="ih5", dvh_truth_mV=0.0)
    c2 = tm.draw_twin_manifest(real, n_per_configuration=2, n_no_noise=1, seed=0, ih_arm="ih5", dvh_truth_mV=0.0,
                               allow_dvh_truth_with_ih5=True)
    pd.testing.assert_frame_equal(c1, c2)


@pytest.mark.parametrize("bad", [1, 0, "True", None, 1.0])
def test_flag_non_bool_refused(bad):
    with pytest.raises(ValueError):
        tm.draw_twin_manifest(_real(), n_per_configuration=2, n_no_noise=1, seed=0, allow_dvh_truth_with_ih5=bad)


def test_flag_round_trip(tmp_path):
    df = tm.draw_twin_manifest(_real(), n_per_configuration=2, n_no_noise=1, seed=3, ih_arm="ih5",
                               allow_dvh_truth_with_ih5=True, truth_kwargs=KW)
    p = tmp_path / "m.csv"
    tm.save_twin_manifest(df, p)
    pd.testing.assert_frame_equal(tm.load_twin_manifest(p), df)


# ============================================================================ run_ih_fit default arm

def test_run_ih_fit_default_arm_ih5():
    import run_ih_fit as R
    p = R._parse_args(["--archive-dir", "/x", "--output-dir", "/x", "--code-dir", str(CODE_DIR)])
    assert p.arm == "ih5"
    spec = R.build_spec_from_args(p, R.resolve_arm_config(p))
    assert spec.names == IH5
    # the other arms still selectable by name, unchanged
    assert R.ARMS["ih6"].fit_params == ("Cm", "Rm", "Ra", "gbar", "dv_h", "kappa_tau")
    assert R.ARMS["ih6_d207"].fit_params == ("Cm", "Rm", "Ra", "gbar", "dv_h", "kappa_tau")


def test_submit_ih_fit_not_changed():
    # NOT changed by Block 30: submit_ih_fit.sh's default ARM=ih6 (it passes --arm explicitly)
    txt = (CODE_DIR / "submit_ih_fit.sh").read_text()
    assert 'ARM="${ARM:-ih6}"' in txt or "ARM:-ih6}" in txt
    assert '--arm                    "$ARM"' in txt
    r = subprocess.run(["git", "diff", "--quiet", "8b05c56", "e05b5ec", "--", "submit_ih_fit.sh"], cwd=CODE_DIR)
    assert r.returncode == 0


# ============================================================================ run_twin_recovery (plan level)

neuron = pytest.importorskip("neuron")
pytest.importorskip("skopt")
import run_twin_recovery as rtr                                        # noqa: E402
from test_spec_block8_twin_recovery import group                       # noqa: E402,F401  (the fixture group)

SMALL30 = ["--n-per-configuration", "1", "--n-no-noise", "1", "--n-calls", "4", "--n-initial", "4",
           "--refine-max-nfev", "2", "--refine-starts", "1"]
IDS4 = (900000000, 900000100, 900000200, 900000300)


def _plan(group, out, *extra):
    return rtr.main(["plan", "--group-dir", str(group["gdir"]), "--noise-dir", str(group["ndir"]), "--out", str(out)]
                    + SMALL30 + list(extra))


def _man(out):
    return pd.read_csv(out / "manifest.csv", float_precision="round_trip")


def test_plan_defaults_are_ih5(group, tmp_path):
    assert rtr.IH_ARM_DEFAULT == "ih5"
    assert rtr.build_parser().parse_args(["plan", "--group-dir", "g", "--noise-dir", "n", "--out", "o"]).ih_arm == "ih5"
    out = tmp_path / "d"
    assert _plan(group, out, "--seed", "4") == 0
    rc = json.loads((out / "run_config.json").read_text())
    assert rc["ih_arm"] == "ih5" and rc["gate_arm"] == "ih5" and rc["dvh_truth_mV"] == 0.0
    assert rc["gate_required_axes"] == ["Cm", "gbar", "kappa_tau"]
    assert "ih5_dvh_truth_range_mV" not in rc["options"]
    m = _man(out)
    assert list(m["arm"]) == ["ih5", "passive_fullstep", "ih5", "ih5"] and (m["ih_dvh_mV"] == 0.0).all()
    # the default plan equals the explicit ih5 plan: re-planning with --ih-arm ih5 is a no-op
    assert _plan(group, out, "--seed", "4", "--ih-arm", "ih5") == 0


def test_plan_option_gives_ih6_truths(group, tmp_path):
    o5, o6 = tmp_path / "b5", tmp_path / "b6"
    assert _plan(group, o5, "--seed", "4", "--ih-arm", "ih5", "--ih5-dvh-truth-range-mV=-5,5") == 0
    assert _plan(group, o6, "--seed", "4", "--ih-arm", "ih6_d207") == 0
    m5, m6 = _man(o5), _man(o6)
    cols = [c for c in m6.columns if c != "arm"]
    pd.testing.assert_frame_equal(m5[cols], m6[cols])
    assert list(m5["arm"]) == ["ih5", "passive_fullstep", "ih5", "ih5"]
    full = m5[m5["configuration"] == "full"].iloc[0]
    assert full["ih_dvh_mV"] != 0.0 and -5.0 <= full["ih_dvh_mV"] <= 5.0
    rc = json.loads((o5 / "run_config.json").read_text())
    assert rc["options"]["ih5_dvh_truth_range_mV"] == "-5,5"
    assert rc["truth_kwargs"]["ih_dvh_range_mV"] == [-5.0, 5.0] and rc["dvh_truth_mV"] is None
    assert rc["gate_arm"] == "ih5" and rc["gate_required_axes"] == ["Cm", "gbar", "kappa_tau"]
    # the same plan again is a no-op; the plan without the option on that directory is refused (options differ)
    assert _plan(group, o5, "--seed", "4", "--ih-arm", "ih5", "--ih5-dvh-truth-range-mV=-5,5") == 0
    assert _plan(group, o5, "--seed", "4", "--ih-arm", "ih5") == 3
    # determinism: a second directory with the same plan gives the same manifest
    o5b = tmp_path / "b5b"
    assert _plan(group, o5b, "--seed", "4", "--ih5-dvh-truth-range-mV=-5,5") == 0   # default arm = ih5
    pd.testing.assert_frame_equal(_man(o5b), m5)


@pytest.mark.parametrize("extra, code", [
    (["--ih-arm", "ih6_d207", "--ih5-dvh-truth-range-mV=-5,5"], 2),   # the other arm's option is --dvh-truth-range-mV
    (["--ih5-dvh-truth-range-mV=-11,5"], 3),                          # outside --dvh-bounds (default -10,10)
    (["--ih5-dvh-truth-range-mV=-5,10.5"], 3),
    (["--ih5-dvh-truth-range-mV=-2,2", "--dvh-bounds=-1,1"], 3),
    (["--ih5-dvh-truth-range-mV=a,b"], 2),
    (["--ih5-dvh-truth-range-mV=1"], 2),
    (["--ih5-dvh-truth-range-mV=nan,1"], 2),
])
def test_plan_option_refusals(group, tmp_path, extra, code):
    out = tmp_path / "r"
    assert _plan(group, out, *extra) == code
    assert not (out / "manifest.csv").exists()


def test_plan_option_closed_box(group, tmp_path):
    # Block 22: closed intervals -- the range equal to the box is accepted
    out = tmp_path / "c"
    assert _plan(group, out, "--ih5-dvh-truth-range-mV=-10,10") == 0
    m = _man(out)
    dv = m.loc[m["configuration"] == "full", "ih_dvh_mV"].to_numpy()
    assert np.all((dv >= -10.0) & (dv <= 10.0))


def test_eval_and_sensitivity_theta_five_names(group, tmp_path):
    out = tmp_path / "e"
    assert _plan(group, out, "--seed", "4", "--ih5-dvh-truth-range-mV=-5,5") == 0
    m = _man(out)
    for _, row in m.iterrows():
        ev = rtr.eval_theta_of(row, row["arm"])
        if row["configuration"] in ("full", "full_no_noise"):
            th = dict(kv.split("=") for kv in ev.split(","))
            assert tuple(th) == IH5
            assert float(th["gbar"]) == float(row["ih_gihbar_S_cm2"])
            assert float(th["kappa_tau"]) == float(row["ih_kappa_tau"])
            assert float(th["Cm"]) == float(row["cm_true"])
        else:
            assert ev is None
        if row["configuration"] != "passive_fit":
            st = dict(kv.split("=") for kv in rtr.sensitivity_theta_of(row, row["arm"]).split(","))
            assert tuple(st) == IH5


def test_submitter_dry_run_with_option(group, tmp_path):
    """Step 6 for the cluster entry: DRY_RUN=1 PLAN_ARGS='--ih-arm ih5 --ih5-dvh-truth-range-mV=-5,5' under stub
    qsub/conda (as the Block 8/22 tests run it): it completes, plans nothing, and its plan accepts the option."""
    import sys
    stub = tmp_path / "bin"
    stub.mkdir()
    cb = tmp_path / "condabase" / "etc" / "profile.d"
    cb.mkdir(parents=True)
    (cb / "conda.sh").write_text("conda() { case \"$1\" in activate) export CONDA_DEFAULT_ENV=\"$2\" ;; esac; "
                                 "return 0; }\n")
    files = {"conda": "#!/bin/bash\nif [ \"$1\" = info ] && [ \"$2\" = --base ]; then echo \"$FAKE_CONDA_BASE\"; "
                      "exit 0; fi\nexit 0\n",
             "qsub": "#!/bin/bash\necho QSUB-CALLED >> \"$QSUB_LOG\"\necho 1.x\n",
             "python": "#!/bin/bash\nexec \"$REAL_PYTHON\" \"$@\"\n"}
    for name, text in files.items():
        p = stub / name
        p.write_text(text)
        p.chmod(0o755)
    base = tmp_path / "base"
    (base / "noise_fit" / "NF" / "G").mkdir(parents=True)
    for f in group["ndir"].iterdir():
        shutil.copy(f, base / "noise_fit" / "NF" / "G" / f.name)
    home = tmp_path / "home"
    home.mkdir()
    qlog = tmp_path / "qsub.log"
    env = {k: v for k, v in os.environ.items() if not k.startswith(("TWIN_", "PBS_", "ROADMAP_", "IH_"))}
    env.update(PATH="%s:%s" % (stub, os.environ["PATH"]), RUN_TAG="t30", NOISE_TAG="NF", GROUP="G", HOME=str(home),
               DRY_RUN="1", IH_CODE_DIR=str(CODE_DIR), IH_ARCHIVE_ROOT=str(group["gdir"].parent),
               IH_OUTPUT_BASE=str(base), ROADMAP_DISABLE="1", FAKE_CONDA_BASE=str(tmp_path / "condabase"),
               REAL_PYTHON=sys.executable, QSUB_LOG=str(qlog),
               PLAN_ARGS="--ih-arm ih5 --ih5-dvh-truth-range-mV=-5,5 " + " ".join(SMALL30))
    env.pop("DISPLAY", None)
    r = subprocess.run(["bash", str(CODE_DIR / "submit_twin_recovery.sh")], cwd=CODE_DIR, env=env,
                       capture_output=True, text=True, timeout=600)
    txt = r.stdout + r.stderr
    assert r.returncode == 0, txt[-3000:]
    assert "twin" in txt and "merge" in txt
    assert not (base / "twin_recovery" / "t30" / "G" / "manifest.csv").exists()
    assert not qlog.exists()                                   # nothing submitted
    # the space-separated form (argparse reads "-5,5" as an option): what the user gets if PLAN_ARGS omits '='
    env["PLAN_ARGS"] = "--ih-arm ih5 --ih5-dvh-truth-range-mV -5,5 " + " ".join(SMALL30)
    r2 = subprocess.run(["bash", str(CODE_DIR / "submit_twin_recovery.sh")], cwd=CODE_DIR, env=env,
                        capture_output=True, text=True, timeout=600)
    assert r2.returncode != 0                                  # refused (documented form is '=LO,HI')


# ============================================================================ end to end (step 6, step 7)

@pytest.fixture(scope="module")
def e2e30(group, tmp_path_factory):
    out = tmp_path_factory.mktemp("e30") / "run"
    assert _plan(group, out, "--seed", "4", "--ih5-dvh-truth-range-mV=-5,5") == 0
    codes = {t: rtr.main(["twin", "--out", str(out), "--twin", str(t)]) for t in IDS4}
    assert rtr.main(["merge", "--out", str(out), "--no-plot"]) == 0
    return out, codes


def test_e2e_build_has_dvh_fit_has_five(e2e30):
    out, codes = e2e30
    assert all(c == 0 for c in codes.values()), codes
    m = _man(out)
    for _, row in m.iterrows():
        tid = int(row["specimen_id"])
        meta = json.loads((out / "twins" / ("twin_%d" % tid) / ("specimen_%d" % tid) / "metadata.json").read_text())
        th = meta["twin"]["theta"]
        if row["configuration"] in ("full", "full_no_noise", "passive_fit"):
            assert th["dv_h"] == float(row["ih_dvh_mV"]) and th["dv_h"] != 0.0     # the generating truth
        argv = json.loads((out / "fits" / ("twin_%d" % tid) / "fit_argv.json").read_text())["argv"]
        assert argv[argv.index("--arm") + 1] == row["arm"]
        ev = [x for x in argv if x.startswith("--eval-theta")]
        if row["configuration"] in ("full", "full_no_noise"):
            names = [kv.split("=")[0] for kv in ev[0].split("=", 1)[1].split(",")]
            assert tuple(names) == IH5
        res = pd.read_csv(out / "fits" / ("twin_%d" % tid) / "phase2_results.csv", float_precision="round_trip")
        r = res[res["specimen_id"] == tid].iloc[0]
        if row["arm"] == "ih5":
            assert tuple(str(r["param_names"]).split("|")) == IH5
            for c in ("Cm", "Rm", "Ra", "gbar", "kappa_tau", "loss_at_estimate"):
                assert np.isfinite(float(r[c])), c


def test_e2e_merge_extras_and_gate(e2e30):
    out, _ = e2e30
    m = _man(out)
    rc = json.loads((out / "run_config.json").read_text())
    ex = pd.read_csv(out / "recovery_extras.csv", float_precision="round_trip")
    assert len(ex) == 4
    assert "on_bound_dv_h" not in ex.columns and all(("on_bound_" + n) in ex.columns for n in IH5)
    for _, e in ex.iterrows():
        row = m[m["specimen_id"] == e["twin_id"]].iloc[0]
        # dvh_true_mV / dvh_est_mV only for a twin fitted with dv_h (Block 22): none here
        assert np.isnan(e["dvh_true_mV"]) and np.isnan(e["dvh_est_mV"])
        if row["configuration"] in ("full", "full_no_noise"):
            # g_h(v*) at the truth with the TRUTH's dv_h (Block 17, 22), recomputed by hand
            v = float(e["v_deepest_train_mV"])
            dvh = float(row["ih_dvh_mV"])
            # m_inf of Ih_human at the run's base shift (0 by default): v_h = -90.87, k = 8.05
            assert float(rc["options"]["vshift_base"]) == 0.0
            exp = float(row["ih_gihbar_S_cm2"]) / (1.0 + math.exp((v + 90.87 - dvh) / 8.05))
            # 1e-3 relative: v_h and k given to 4 significant digits in the spec
            assert abs(e["gh_deep_true_S_cm2"] / exp - 1.0) < 1e-3
            assert np.isfinite(e["loss_at_truth"])
    # the gate: on ih5 (2.8 (g)), no dv_h axis even though the twins' truth has one (Block 30: model error)
    for sub in ("report", "report_without_flagged", "report_no_noise"):
        gv = pd.read_csv(out / sub / "gate_verdict.csv")
        assert set(gv["arm"]) == {"ih5"}
        params = set(gv["parameter"])
        assert not any("dv_h" in p for p in params), params
        for p in ("Cm", "gbar", "kappa_tau"):
            assert p in params
    summ = pd.read_csv(out / "report" / "recovery_summary.csv")
    assert set(summ["parameter"]) == set(IH5)


def test_e2e_status_and_reproducible_plan(e2e30, group, tmp_path):
    out, _ = e2e30
    st = pd.read_csv(out / "twins_status.csv")
    assert (st["status"] == "ok").all()
    o2 = tmp_path / "again"
    assert _plan(group, o2, "--seed", "4", "--ih5-dvh-truth-range-mV=-5,5") == 0
    pd.testing.assert_frame_equal(_man(o2), _man(out))


def test_swap_range_and_inflation_together(group, tmp_path):
    """Step 8 swappability: another range and Block 29's inflation through the plan alone; both recorded."""
    out = tmp_path / "sw"
    assert _plan(group, out, "--seed", "2", "--ih5-dvh-truth-range-mV=-2,3", "--twin-diam-factor", "1.5") == 0
    rc = json.loads((out / "run_config.json").read_text())
    assert rc["options"]["ih5_dvh_truth_range_mV"] == "-2,3" and rc["options"]["twin_diam_factor"] == 1.5
    m = _man(out)
    dv = m.loc[m["configuration"].isin(["full", "passive_fit", "full_no_noise"]), "ih_dvh_mV"].to_numpy()
    assert np.all((dv >= -2.0) & (dv <= 3.0))


def test_changed_files_ascii_lf():
    for f in ("run_ih_fit.py", "twin_manifest.py", "run_twin_recovery.py", "submit_twin_recovery.sh"):
        b = (CODE_DIR / f).read_bytes()
        assert all(x < 128 for x in b), f
        assert b"\r" not in b, f


# ============================================================================ finding reproducers

def test_finding_replan_of_pre_block30_default_run_refused(group, tmp_path):
    """A run planned before Block 30 with the plan's defaults recorded ih_arm ih6_d207. The submitter always runs
    plan first with PLAN_ARGS; re-submitting that run with the same PLAN_ARGS (no --ih-arm) is now refused (exit 3,
    options differ: ih_arm) -- the default change reaches existing run directories. Recorded as a NOTE (spec gap):
    the entry says only that a plan without the new option compares equal."""
    out = tmp_path / "old"
    assert _plan(group, out, "--seed", "4", "--ih-arm", "ih6_d207") == 0     # what a pre-Block-30 default plan wrote
    assert _plan(group, out, "--seed", "4") == 3                              # the same command line today
