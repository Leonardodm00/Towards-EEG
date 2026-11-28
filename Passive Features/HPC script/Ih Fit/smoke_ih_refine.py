"""Smoke test for the refinement in the fit, the arm ih6_d207 and the new box of gbar -- Block 20 in
specs/SPEC.md (F1-F7), SPEC 2.9 (a), (b), (e).

The fixture: a noise-free twin archive built at a known truth by Block 12's functions, on the
fixture cell of smoke_ih5.py (two SS sweeps with trace-QC drops; LS sweeps of -130 ... +60 pA, so
n = 5 hyperpolarising steps), with a ball-and-stick morphology. The truth is the fit's own
six-parameter model with dv_h = -3 mV (THETA6 below): no noise, no tilt.

Checks
    test_F1_arm_table      (pure) each arm's names, protocol (role rule, weighting, LS sample
                           weight) and refine against SPEC Block 20 written out here; the parser's
                           defaults; each override; refusals exit 2
    test_F2_ih6_d207_fit   run_ih_fit.main --arm ih6_d207 --n-calls 8 --n-initial 4
                           --refine-maxfev 12 --refine-starts 2 on the fixture: runs; six names;
                           ls_roles.csv follows d207; the new columns present, after the old ones;
                           loss_after_refine <= loss_after_search; loss_at_estimate == train_rel_loss
                           == loss_after_refine (1e-9 relative); refine_starts.csv has <= 2 rows with
                           finite entries; mono.gp_minimize is the original object after the fit, and
                           also after a fit_one_cell that raises (it is wrapped during the call)
    test_F3_refine_none    the same fit with --refine none reproduces, in every pre-existing column,
                           the contract commit's run_ih_fit.py run with the same configuration (the
                           contract code has no arm ih6_d207: --arm ih6 with the d207 protocol given
                           by its overrides and --gbar-bounds=1e-5,1e-3); columns arm and
                           wall_time_s excluded; ls_roles.csv and ls_diagnostics.csv identical
    test_F4_workers        the F2 fit with --n-workers 2 gives the same phase2_results.csv as with
                           --n-workers 1 in every column but n_workers and wall_time_s, and the same
                           refine_starts.csv (needs Block 19's pool: before its merge the check runs
                           against whatever passive_long_step_training.py is here)
    test_F5_eval_theta     --eval-theta with the six names at the truth: finite, and below the loss
                           at the truth with dv_h moved by +5 mV
    test_F6_gbar_box       the default box of gbar is (1e-5, 1e-3): param_spec, the CLI default, the
                           spec of every I_h arm, and the run header of F2
    test_F7_ascii_lf       the Block 20 files are pure ASCII with LF line endings

Run (from this folder: NEURON loads ./x86_64 from the working directory)
    python smoke_ih_refine.py [--build] [--only F1,F2]
        Prints the environment and a PASS / FAIL / ERROR / TODO / SKIP table, then the full
        message of every failure. Exit code 1 if any check failed, errored or is still TODO.

Keep this file pure ASCII (hpc-python-compat).
"""
from __future__ import annotations

import argparse
import atexit
import contextlib
import importlib.metadata
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.chdir(HERE)                     # NEURON loads ./x86_64 from the CWD
sys.path.insert(0, str(HERE))

import numpy as np                 # noqa: E402
import pandas as pd                # noqa: E402
from numpy.testing import assert_allclose  # noqa: E402

import run_ih_fit as R             # noqa: E402
import smoke_ih5 as S5             # noqa: E402  (the fixture cell's layout and the twins' table)

SEED = 12345
REPORT_PACKAGES = ("numpy", "scipy", "pandas", "scikit-optimize", "neuron")
#: the contract commit of Block 20 (F3 runs its run_ih_fit.py)
CONTRACT = "33a6a70"
_CODE_REL = "Passive Features/HPC script/Ih Fit"

TWIN_ID = 900002001
THETA6 = dict(Cm=1.0, Rm=20000.0, Ra=150.0, gbar=1.2e-4, dv_h=-3.0, kappa_tau=1.3)
F2_OPTS = ["--n-calls", "8", "--n-initial", "4", "--refine-maxfev", "12", "--refine-starts", "2"]
NEW_COLUMNS = R.REFINE_RECORD_COLUMNS
_LOSS_CRASH_PENALTY = 1e6

_STATE = {}


def _theta_arg(theta):
    return ",".join("%s=%r" % (k, float(v)) for k, v in theta.items())


def _clear():
    import synthetic_ground_truth as sgt
    sgt._clear_neuron_sections()


def _fixture():
    """The group directory holding the noise-free twin at THETA6 (built once)."""
    if "group" in _STATE:
        return _STATE
    import ih_mechanism as IM
    import param_spec as PS
    import smoke_noise_windows as snw
    import synthetic_ground_truth as sgt
    import twin_build as tb
    import twin_table as tt
    from twin_config import TwinConfig
    root = Path(tempfile.mkdtemp(prefix="smoke ih refine "))  # a space, as on the cluster's paths
    atexit.register(shutil.rmtree, str(root), True)
    real = root / "real" / ("specimen_%d" % S5.REAL_ID)
    snw.write_fixture(real, np.random.default_rng(SEED), with_response=True, dropped=S5.SS_DROPPED,
                      sweeps=S5.SS_SWEEPS, ls_amps=S5.LS_AMPS)
    sgt.write_ball_and_stick_swc(real / "reconstruction.swc")
    meta = json.load(open(real / "metadata.json"))
    protocol = tt.load_cell_protocol(real, spike_reason_fn=lambda v_mV, sampling_rate_Hz: "")
    table_path = root / "twin_sweeps_FIX.csv"
    tt.write_twin_sweep_table(S5._hand_table(protocol), table_path)
    table = tt.read_twin_sweep_table(table_path)
    cfg = TwinConfig()
    spec = tb.TwinSpec(twin_id=TWIN_ID, twin_of_specimen_id=S5.REAL_ID, configuration="full",
                       theta=dict(THETA6), v_rest_mV=S5.V_REST, F=S5.F)
    _clear()
    cell = tb.build_twin_cell(real / "reconstruction.swc", spec, ih_spec=IM.IhSpec(),
                              apply_theta=PS.make_ih_spec().apply, cfg=cfg)
    traces = tb.simulate_twin(cell, spec, protocol, table, cfg=cfg, with_noise=False)
    _clear()
    group = root / "group FIX"
    tb.write_twin_archive(group, spec, traces, swc_path=real / "reconstruction.swc", real_metadata=meta,
                          features=dict(rin_MOhm=1.0, tau_ms=1.0, sag_ratio=0.0), cfg=cfg)
    _STATE.update(root=root, group=group, n_out=0)
    return _STATE


def _new_out(tag):
    st = _fixture()
    st["n_out"] += 1
    return st["root"] / ("out_%s_%d" % (tag, st["n_out"]))


def _fit_args(out, arm, *extra):
    st = _fixture()
    return (["--archive-dir", str(st["group"]), "--output-dir", str(out), "--code-dir", str(HERE),
             "--arm", arm, "--phase3-subset", "none"] + list(extra))


def _run_main(argv):
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            R.main(argv)
    except BaseException:
        sys.stdout.write(buf.getvalue()[-4000:])
        raise
    finally:
        _clear()
    return buf.getvalue()


def _sub(cmd, label):
    r = subprocess.run(cmd, cwd=str(HERE), capture_output=True, text=True, timeout=3600)
    if r.returncode != 0:
        raise AssertionError("%s exited %d:\n%s\n%s" % (label, r.returncode, r.stdout[-3000:], r.stderr[-3000:]))
    return r.stdout


def _f2_run():
    """The F2 fit (with --eval-theta at the truth, for F5), in process, run once."""
    if "f2" in _STATE:
        return _STATE["f2"]
    import passive_fitting_hpc_fixed as mono
    orig = mono.gp_minimize
    out = _new_out("f2")
    t0 = time.time()
    # [adapted in Block 27: --refine nelder_mead, the refinement this suite tests; trf is the default now]
    logtxt = _run_main(_fit_args(out, "ih6_d207", *(F2_OPTS + ["--refine", "nelder_mead",
                                                               "--eval-theta", _theta_arg(THETA6)])))
    secs = time.time() - t0
    _STATE["f2"] = dict(out=out, res=pd.read_csv(out / "phase2_results.csv", float_precision="round_trip"),
                        roles=pd.read_csv(out / "ls_roles.csv"),
                        starts=pd.read_csv(out / "refine_starts.csv", float_precision="round_trip"),
                        log=logtxt, restored=(mono.gp_minimize is orig), secs=secs)
    return _STATE["f2"]


def check_close(name, actual, expected, rtol=0.0, atol=0.0):
    assert_allclose(np.asarray(actual), np.asarray(expected), rtol=rtol, atol=atol, err_msg=name)


def _parse(extra):
    return R._parse_args(["--archive-dir", "/a", "--output-dir", "/o", "--code-dir", "."] + list(extra))


# ---------------------------------------------------------------- checks ---
def test_F1_arm_table():
    six = ("Cm", "Rm", "Ra", "gbar", "dv_h", "kappa_tau")
    five = ("Cm", "Rm", "Ra", "gbar", "kappa_tau")
    three = ("Cm", "Rm", "Ra")
    # SPEC Blocks 15 and 20, written out: (names, role_rule, weighting, ls_sample_weight, refine)
    # [adapted in Block 27: the default refinement of the three D-207 arms is trf (SPEC 2.10 (c))]
    expect = {
        "ih6_d207": (six, "d207", "ls_amplitude", "peak_decay", "trf"),
        "ih5": (five, "d207", "ls_amplitude", "peak_decay", "trf"),
        "passive_fullstep": (three, "d207", "ls_amplitude", "peak_decay", "trf"),
        "ih6": (six, "d006", "relative", "none", "none"),
        "baseline_runB": (three, "d006", "relative", "none", "none"),
    }
    bad = []
    if set(R.ARMS) != set(expect):
        bad.append("arms %s" % sorted(R.ARMS))
    for arm, exp in expect.items():
        c = R.resolve_arm_config(_parse(["--arm", arm]))
        got = (tuple(c.fit_params), c.role_rule, c.weighting, c.ls_sample_weight, c.refine)
        if got != exp:
            bad.append("arm %s: %s != %s" % (arm, got, exp))
    # ih6_d207 is ih6 in everything else
    a, b = R.ARMS["ih6_d207"], R.ARMS["ih6"]
    for f in ("attach_ih", "ih_protocol", "ls_window_mode", "ls_window_ms", "ls_max_amplitude_pA",
              "load_depolarising_ls", "n_long_train", "ls_deflection_cap_mV", "gate_valid_via"):
        if getattr(a, f) != getattr(b, f):
            bad.append("ih6_d207.%s = %r differs from ih6's %r" % (f, getattr(a, f), getattr(b, f)))
    # the parser's defaults
    p = _parse(["--arm", "ih6_d207"])
    for name, want in (("refine", None), ("refine_starts", 3), ("refine_maxfev", 100), ("n_workers", 1),
                       ("gbar_bounds", "1e-5,1e-3"), ("dvh_bounds", "-10.0,10.0"), ("arm", "ih6_d207")):
        if getattr(p, name) != want:
            bad.append("default --%s = %r, not %r" % (name.replace("_", "-"), getattr(p, name), want))
    # renewed 2026-11-28 (Block 30, D-282): the default arm is ih5 (was ih6)
    if _parse([]).arm != "ih5":
        bad.append("the default --arm is not ih5")
    # overrides
    for arm in expect:
        for val in ("none", "nelder_mead"):
            c = R.resolve_arm_config(_parse(["--arm", arm, "--refine", val]))
            if c.refine != val or c.fit_params != R.ARMS[arm].fit_params or c.role_rule != R.ARMS[arm].role_rule:
                bad.append("%s --refine %s -> %s" % (arm, val, c))
    # [adapted in Block 27: ih5's own refinement is trf; Nelder-Mead is asked for by --refine]
    _a = _parse(["--arm", "ih5", "--refine", "nelder_mead", "--refine-starts", "2", "--refine-maxfev", "12"])
    rc = R.refine_config_from_args(_a, R.resolve_arm_config(_a))
    import local_refine as LR
    if rc != LR.RefineConfig(n_starts=2, maxfev_per_start=12):
        bad.append("refine config %s" % (rc,))
    if R.refine_config_from_args(_parse(["--arm", "ih6"]), R.ARMS["ih6"]) is not None:
        bad.append("ih6 has a refinement")
    # --eval-theta names the six for ih6_d207
    _parse(["--arm", "ih6_d207", "--eval-theta", _theta_arg(THETA6)])
    # refusals: exit 2
    five_theta = {k: v for k, v in THETA6.items() if k != "dv_h"}
    for extra in (["--refine", "powell"], ["--refine-starts", "0"], ["--refine-maxfev", "0"],
                  ["--n-workers", "0"], ["--n-workers", "-1"], ["--n-workers", "1.5"],
                  ["--eval-theta", _theta_arg(five_theta)]):
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                _parse(["--arm", "ih6_d207"] + extra)
            bad.append("parser accepted %s" % extra)
        except SystemExit as exc:
            if exc.code != 2:
                bad.append("%s exited %r, not 2" % (extra, exc.code))
    assert not bad, "; ".join(bad)


def test_F2_ih6_d207_fit():
    st = _f2_run()
    res, roles, starts = st["res"], st["roles"], st["starts"]
    bad = []
    if (st["out"] / "failed_cells.txt").exists():
        bad.append("failed_cells.txt written")
    assert len(res) == 1, "phase2_results.csv has %d rows" % len(res)
    row = res.iloc[0]
    if str(row["arm"]) != "ih6_d207" or str(row["param_names"]) != "|".join(R.IH6_NAMES):
        bad.append("arm %r / param_names %r" % (row["arm"], row["param_names"]))
    got = {float(r.amplitude_pA): str(r.role) for r in roles.itertuples()}
    if got != S5.D207_ROLES_N5:
        bad.append("ls_roles.csv %s != d207 %s" % (got, S5.D207_ROLES_N5))
    for c in NEW_COLUMNS:
        if c not in res.columns:
            bad.append("column %s missing" % c)
    cols = list(res.columns)
    if any(cols.index(c) < cols.index(o) for c in NEW_COLUMNS if c in cols for o in R.FIT_RECORD_COLUMNS):
        bad.append("the Block 20 columns are not after the existing ones")
    if str(row.get("refine")) != "nelder_mead" or int(row.get("n_workers", -9)) != 1:
        bad.append("refine %r, n_workers %r" % (row.get("refine"), row.get("n_workers")))
    ls, lr = float(row["loss_after_search"]), float(row["loss_after_refine"])
    le, tr = float(row["loss_at_estimate"]), float(row["train_rel_loss"])
    if not (np.isfinite(ls) and np.isfinite(lr) and lr <= ls and lr < _LOSS_CRASH_PENALTY):
        bad.append("loss_after_refine %r, loss_after_search %r" % (lr, ls))
    if bool(row["refine_improved"]) != (lr < ls):
        bad.append("refine_improved %r with %r < %r" % (row["refine_improved"], lr, ls))
    # refine_starts.csv: at most 2 rows, finite, consistent with the row
    if len(starts) > 2 or len(starts) != int(row["refine_n_starts"]) or len(starts) == 0:
        bad.append("refine_starts.csv %d rows, refine_n_starts %r" % (len(starts), row["refine_n_starts"]))
    want_cols = R.refine_starts_columns(R.build_spec_from_args(_parse(["--arm", "ih6_d207"]), R.ARMS["ih6_d207"]))
    if list(starts.columns) != want_cols:
        bad.append("refine_starts.csv columns %s" % list(starts.columns))
    # [adapted in Block 27: the columns Block 27 appends are NaN under Nelder-Mead by contract]
    num = starts[list(R.REFINE_STARTS_HEAD[1:]) + [c for c in starts.columns if c.startswith("x0_")
                                                   or (c.startswith("x_") and not c.startswith("x_solver_"))]
                 ].to_numpy(dtype=float)
    if not np.all(np.isfinite(num)):
        bad.append("refine_starts.csv has non-finite entries")
    if int(starts["nfev"].sum()) != int(row["refine_n_evals"]):
        bad.append("sum nfev %d != refine_n_evals %r" % (int(starts["nfev"].sum()), row["refine_n_evals"]))
    if lr < ls and float(starts["fun"].min()) != lr:
        bad.append("the best start's fun %r != loss_after_refine %r" % (float(starts["fun"].min()), lr))
    if not st["restored"]:
        bad.append("mono.gp_minimize is not the original object after the fit")
    # restored also when fit_one_cell raises, and wrapped while it runs
    import passive_fitting_hpc_fixed as mono
    orig_gp, orig_fit = mono.gp_minimize, mono.fit_one_cell
    seen = []

    def _boom(*a, **k):
        seen.append(getattr(mono.gp_minimize, "wrapped_minimizer", None) is orig_gp)
        raise RuntimeError("planted failure inside fit_one_cell")
    mono.fit_one_cell = _boom
    try:
        try:
            _run_main(_fit_args(_new_out("boom"), "ih6_d207", "--n-calls", "4", "--n-initial", "2"))
        except SystemExit:
            pass                                        # no successful fit: main exits 1
    finally:
        mono.fit_one_cell = orig_fit
    if seen != [True]:
        bad.append("gp_minimize not wrapped during fit_one_cell: %s" % seen)
    if mono.gp_minimize is not orig_gp:
        mono.gp_minimize = orig_gp
        bad.append("mono.gp_minimize not restored after an exception")
    print("  [F2] %.0f s; search %.10g -> refined %.10g (improved %s; %d start(s), %d local evaluations); "
          "loss_at_estimate %.17g, train_rel_loss %.17g" % (st["secs"], ls, lr, row["refine_improved"],
                                                             len(starts), int(row["refine_n_evals"]), le, tr))
    print("  [F2] estimate %s" % ", ".join("%s=%.4g" % (n, float(row[n])) for n in R.IH6_NAMES))
    assert not bad, "; ".join(bad)
    # same deterministic loss at the same q up to the exp/log round trip of the estimate (SPEC 1e-9)
    check_close("F2_loss_at_estimate", le, tr, rtol=1e-9)
    check_close("F2_train_rel_loss", tr, lr, rtol=1e-9)


def _contract_run_ih_fit(tmp):
    try:
        top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=str(HERE), capture_output=True,
                             text=True, check=True).stdout.strip()
        src = subprocess.run(["git", "show", "%s:%s/run_ih_fit.py" % (CONTRACT, _CODE_REL)], cwd=top,
                             capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise unittest.SkipTest("the contract commit %s is not readable through git here: %s" % (CONTRACT, exc))
    path = Path(tmp) / "run_ih_fit_contract.py"
    path.write_bytes(src)
    return path


def _f5_eval_shifted():
    return dict(THETA6, dv_h=THETA6["dv_h"] + 5.0)


def test_F3_refine_none():
    tmp = Path(tempfile.mkdtemp(prefix="smoke_ih_refine_contract_"))
    atexit.register(shutil.rmtree, str(tmp), True)
    old_py = _contract_run_ih_fit(tmp)
    out_old, out_new = _new_out("f3_contract"), _new_out("f3_now")
    ev = ["--eval-theta", _theta_arg(_f5_eval_shifted())]
    _sub([sys.executable, str(old_py)] + _fit_args(out_old, "ih6", "--ls-role-rule", "d207",
                                                   "--weighting", "ls_amplitude", "--ls-sample-weight",
                                                   "peak_decay", "--gbar-bounds=1e-5,1e-3",
                                                   "--n-calls", "8", "--n-initial", "4", *ev),
         "contract run_ih_fit.py")
    _sub([sys.executable, str(HERE / "run_ih_fit.py")] + _fit_args(out_new, "ih6_d207", "--refine", "none",
                                                                  *(F2_OPTS + ev)), "run_ih_fit.py")
    old = pd.read_csv(out_old / "phase2_results.csv", dtype=str, keep_default_na=False)
    new = pd.read_csv(out_new / "phase2_results.csv", dtype=str, keep_default_na=False)
    bad = []
    if list(new.columns[:len(old.columns)]) != list(old.columns):
        bad.append("the old columns are not the first ones, in their order")
    n_cmp = 0
    for c in old.columns:
        if c in ("wall_time_s", "arm"):          # the clock; the arm's name (ih6 there, ih6_d207 here)
            continue
        n_cmp += 1
        if c not in new.columns or list(old[c]) != list(new[c]):
            bad.append("%s: %s != %s" % (c, list(old[c]), list(new.get(c, []))))
    for f in ("ls_roles.csv", "ls_diagnostics.csv"):
        if (out_old / f).read_bytes() != (out_new / f).read_bytes():
            bad.append("%s differs" % f)
    row = new.iloc[0]
    want = dict(refine="none", refine_improved="False", refine_n_starts="0", refine_n_evals="0",
                refine_best_start="-1", n_workers="1")
    for c, w in want.items():
        if str(row.get(c)) != w:
            bad.append("%s = %r, not %r" % (c, row.get(c), w))
    if not (row["loss_after_search"] == row["loss_after_refine"]
            and float(row["loss_after_search"]) == float(row["train_rel_loss"])):
        bad.append("refine none: loss_after_search %s, loss_after_refine %s, train_rel_loss %s"
                   % (row["loss_after_search"], row["loss_after_refine"], row["train_rel_loss"]))
    st = pd.read_csv(out_new / "refine_starts.csv")
    if len(st) != 0 or "x0_dv_h" not in st.columns:
        bad.append("refine_starts.csv without refinement: %d rows, columns %s" % (len(st), list(st.columns)))
    _STATE["f3_new"] = new
    print("  [F3] %d old columns compared text for text (wall_time_s, arm excluded); refine none: "
          "loss_after_search = loss_after_refine = %s" % (n_cmp, row["loss_after_search"]))
    assert not bad, "; ".join(bad)


def test_F4_workers():
    outs = {}
    for w in (1, 2):
        out = _new_out("f4_w%d" % w)
        _sub([sys.executable, str(HERE / "run_ih_fit.py")] + _fit_args(out, "ih6_d207", *(F2_OPTS + [
            "--n-workers", str(w)])), "run_ih_fit.py --n-workers %d" % w)
        outs[w] = out
    a = pd.read_csv(outs[1] / "phase2_results.csv", dtype=str, keep_default_na=False)
    b = pd.read_csv(outs[2] / "phase2_results.csv", dtype=str, keep_default_na=False)
    bad = []
    if list(a.columns) != list(b.columns):
        bad.append("columns differ")
    for c in a.columns:
        if c in ("wall_time_s", "n_workers"):
            continue
        if list(a[c]) != list(b.get(c, [])):
            bad.append("%s: %s != %s" % (c, list(a[c]), list(b.get(c, []))))
    if list(a["n_workers"]) != ["1"] or list(b["n_workers"]) != ["2"]:
        bad.append("n_workers %s / %s" % (list(a["n_workers"]), list(b["n_workers"])))
    if (outs[1] / "refine_starts.csv").read_bytes() != (outs[2] / "refine_starts.csv").read_bytes():
        bad.append("refine_starts.csv differs")
    print("  [F4] W=1 and W=2: %d columns compared text for text; wall %s s vs %s s"
          % (len(a.columns) - 2, a["wall_time_s"][0], b["wall_time_s"][0]))
    assert not bad, "; ".join(bad[:5])


def test_F5_eval_theta():
    st = _f2_run()
    l_truth = float(st["res"].iloc[0]["loss_at_eval_theta"])
    if "f3_new" in _STATE:
        l_shift = float(_STATE["f3_new"].iloc[0]["loss_at_eval_theta"])
    else:
        out = _new_out("f5_shift")
        _run_main(_fit_args(out, "ih6_d207", "--refine", "none", "--n-calls", "4", "--n-initial", "2",
                            "--eval-theta", _theta_arg(_f5_eval_shifted())))
        l_shift = float(pd.read_csv(out / "phase2_results.csv").iloc[0]["loss_at_eval_theta"])
    print("  [F5] loss at the truth %.6g; at the truth with dv_h + 5 mV %.6g" % (l_truth, l_shift))
    assert np.isfinite(l_truth) and l_truth < _LOSS_CRASH_PENALTY, l_truth
    assert l_truth < l_shift, "loss at the truth %g not below the dv_h + 5 mV one %g" % (l_truth, l_shift)


def test_F6_gbar_box():
    import param_spec as PS
    bad = []
    if tuple(PS.DEFAULT_GBAR_BOUNDS) != (1e-5, 1e-3):
        bad.append("param_spec.DEFAULT_GBAR_BOUNDS = %s" % (PS.DEFAULT_GBAR_BOUNDS,))
    ax = PS.make_ih_spec().axis("gbar")
    if (ax.lo, ax.hi) != (1e-5, 1e-3) or not ax.log:
        bad.append("make_ih_spec() gbar box %s" % ((ax.lo, ax.hi, ax.log),))
    for arm in ("ih6_d207", "ih5", "ih6"):
        a = _parse(["--arm", arm])
        g = R.build_spec_from_args(a, R.resolve_arm_config(a)).axis("gbar")
        if (g.lo, g.hi) != (1e-5, 1e-3):
            bad.append("%s spec gbar box %s" % (arm, (g.lo, g.hi)))
    a = _parse(["--arm", "ih6_d207"])
    dv = R.build_spec_from_args(a, R.resolve_arm_config(a)).axis("dv_h")
    if (dv.lo, dv.hi, dv.log) != (-10.0, 10.0, False):
        bad.append("dv_h axis %s" % ((dv.lo, dv.hi, dv.log),))
    logtxt = _f2_run()["log"]
    hdr = [ln for ln in logtxt.splitlines() if "box: " in ln]
    if not hdr or "gbar[1e-05, 0.001]" not in hdr[0]:
        bad.append("run header box line %r" % (hdr[:1],))
    else:
        print("  [F6] %s" % hdr[0].split("box: ", 1)[1][:160])
    assert not bad, "; ".join(bad)


def test_F7_ascii_lf():
    bad = []
    for f in ("local_refine.py", "run_ih_fit.py", "param_spec.py", "smoke_local_refine.py",
              "smoke_ih_refine.py", "smoke_ih5.py", "smoke_ih_fit.py"):
        b = (HERE / f).read_bytes()
        if any(c > 127 for c in b):
            bad.append(f + " non-ASCII")
        if b"\r" in b:
            bad.append(f + " CR bytes")
    assert not bad, ", ".join(bad)


# ---------------------------------------------------------------- runner ---
def _environment():
    parts = ["python %s" % platform.python_version()]
    for package in REPORT_PACKAGES:
        try:
            parts.append("%s %s" % (package, importlib.metadata.version(package)))
        except importlib.metadata.PackageNotFoundError:
            parts.append("%s (not installed)" % package)
    parts.append(platform.platform())
    parts.append("seed %d" % SEED)
    return " | ".join(parts)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true", help="run nrnivmodl mod if x86_64/special is missing")
    ap.add_argument("--only", default="", help="comma-separated check ids to run, e.g. F1,F2")
    args = ap.parse_args(argv)
    S5.ensure_build(args.build)
    only = [t.strip() for t in args.only.split(",") if t.strip()]
    checks = [(name, obj) for name, obj in globals().items()
              if name.startswith("test_") and callable(obj)
              and (not only or any(name.startswith("test_%s_" % o) for o in only))]
    print("== %s" % Path(__file__).name)
    print(_environment())
    results = []
    for name, func in checks:
        start = time.perf_counter()
        try:
            func()
            status, detail = "PASS", ""
        except unittest.SkipTest as exc:
            status, detail = "SKIP", str(exc)
        except NotImplementedError as exc:
            status, detail = "TODO", str(exc)
        except AssertionError as exc:
            status, detail = "FAIL", str(exc)
        except Exception:  # unexpected errors are reported, never hidden
            status, detail = "ERROR", traceback.format_exc()
        results.append((name, status, time.perf_counter() - start, detail))
    width = max([len(n) for n, _, _, _ in results] + [4])
    for name, status, seconds, detail in results:
        lines = detail.strip().splitlines()
        headline = lines[0] if lines else ""
        print(("%-5s  %-*s  %8.3fs  %s" % (status, width, name, seconds, headline[:200])).rstrip())
    for name, status, _s, detail in results:
        if status in ("FAIL", "ERROR"):
            print("\n---- %s: %s\n%s" % (status, name, detail.strip()))
    counts = {s: sum(1 for r in results if r[1] == s) for s in ("PASS", "FAIL", "ERROR", "TODO", "SKIP")}
    print("\n-- " + ", ".join("%d %s" % (n, s.lower()) for s, n in counts.items()))
    return 1 if counts["FAIL"] + counts["ERROR"] + counts["TODO"] else 0


if __name__ == "__main__":
    sys.exit(main())
