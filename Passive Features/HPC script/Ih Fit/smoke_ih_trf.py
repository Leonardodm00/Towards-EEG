"""Smoke test for the fit's trf refinement, the bound flags and the sensitivity mode of
run_ih_fit.py -- Block 27 in specs/SPEC.md (U1-U10), SPEC 2.10 (a), (c), (d).

The fixture: a noise-free twin archive built at a known truth by Block 12's functions, as
smoke_ih_refine.py builds it (the fixture cell of smoke_ih5.py: two SS sweeps with trace-QC drops;
LS sweeps of -130 ... +60 pA, so n = 5 hyperpolarising steps; a ball-and-stick morphology), at
the truth of smoke_ih5.py (THETA below: dv_h = 0, so that the truth lies in the arm ih5's model
and U5 and U8 can test against it). No noise, no tilt.

Checks
    test_U1_arm_table       (pure) refine of every arm (trf on ih5, ih6_d207, passive_fullstep;
                            none on ih6, baseline_runB); REFINE_CHOICES; --refine overrides;
                            refine_config_from_args gives a TrfConfig / RefineConfig / None with
                            the CLI's values; the parser's defaults; refusals exit 2
    test_U2_trf_fit         run_ih_fit.main --arm ih5 --n-calls 12 --n-initial 8 --refine-starts 2
                            --refine-max-nfev 8 (and --eval-theta at the truth, for U8): runs; the
                            Block 27 columns present (after Block 20's) and finite;
                            l2_after_refine <= l2_after_search; loss_at_estimate == loss_after_refine
                            (1e-9); sqrt_l2_at_estimate**2 == l2_after_refine (1e-9);
                            sqrt_l2_at_estimate >= loss_at_estimate; refine_starts.csv rows with the
                            appended columns, finite; mono.gp_minimize and mono._build_loss_function
                            are the original objects after the fit
    test_U3_refine_none     the same fit with --refine none reproduces, in every pre-existing column,
                            the contract commit's run_ih_fit.py with the same options (wall_time_s
                            excluded); the trf run's loss_after_search equals the search's best
                            (1e-9 relative): the search is untouched
    test_U4_workers         the U2 fit with --n-workers 2 gives the same phase2_results.csv as with 1
                            in every column but n_workers and wall_time_s, and the same
                            refine_starts.csv
    test_U5_start_near_truth  a canned global result (a fake gp_minimize: one point, the truth moved
                            by 0.05 in every unit coordinate): the estimate is within 5e-3 of the
                            truth (unit box, max-norm), the start converged
    test_U6_bound_flags     hand cases on bound_flag and bound_flags
    test_U7_nelder_mead     --refine nelder_mead runs and fills the appended columns as stated
    test_U8_sensitivity     the sensitivity mode at the truth: the four files exist; fd_step_check
                            < 0.05; sqrt_l2_at_theta == U2's sqrt_l2_at_eval_theta (1e-9);
                            sigma_1 >= ... >= sigma_p > 0; W1 differs from W0; layout_from_stored
                            reproduces the live residual layout (array_equal on sqrt_a, masks, D);
                            the truth trace check's maximum (below 1e-6 mV, or reported)
    test_U9_box             --sensitivity-theta outside the box: exit 2 (parser and main)
    test_U10_ascii_lf       the Block 27 files are pure ASCII with LF line endings

Run (from this folder: NEURON loads ./x86_64 from the working directory)
    python smoke_ih_trf.py [--build] [--only U1,U2]
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
#: the contract commit of Block 27 (U3 runs its run_ih_fit.py)
CONTRACT = "87df33a"
_CODE_REL = "Passive Features/HPC script/Ih Fit"

TWIN_ID = 900002701
THETA = dict(S5.THETA)                         # dv_h = 0: the truth lies in the ih5 model
THETA5 = {n: THETA[n] for n in R.IH5_NAMES}
U2_OPTS = ["--n-calls", "12", "--n-initial", "8", "--refine-starts", "2", "--refine-max-nfev", "8"]
_LOSS_CRASH_PENALTY = 1e6
#: Every fit of this suite integrates at the twin's own step (TwinConfig.neuron_dt_ms, 0.025 ms),
#: not at the CLI's default 0.1 ms: only then is the noise-free fixture's truth the minimum of the
#: fit's loss, which U5 and U8 test against. Measured by the implementer at the default 0.1 ms: the
#: truth trace check's maximum 0.158 mV (the SS pulse's discretisation) and U5's estimate 2.1e-2
#: from the truth (unit, max-norm) at an L2 below the truth's (3.1e-6 vs 6.7e-6). At 0.025 ms the
#: traces agree to 1e-13 mV inside every RMSD window but the last sample of each long step: the
#: loader's step lasts 0.99998 s, the twin's 1 s (U8 reports the maximum, at the offset).
FIT_DT = ["--dt-brief-ms", "0.025", "--dt-long-ms", "0.025"]

_STATE = {}


def _theta_arg(theta):
    return ",".join("%s=%r" % (k, float(v)) for k, v in theta.items())


def _clear():
    import synthetic_ground_truth as sgt
    sgt._clear_neuron_sections()


def _fixture():
    """The group directory holding the noise-free twin at THETA (built once), as smoke_ih_refine.py
    builds its fixture."""
    if "group" in _STATE:
        return _STATE
    import ih_mechanism as IM
    import param_spec as PS
    import smoke_noise_windows as snw
    import synthetic_ground_truth as sgt
    import twin_build as tb
    import twin_table as tt
    from twin_config import TwinConfig
    root = Path(tempfile.mkdtemp(prefix="smoke ih trf "))   # a space, as on the cluster's paths
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
                       theta=dict(THETA), v_rest_mV=S5.V_REST, F=S5.F)
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
             "--arm", arm, "--phase3-subset", "none"] + FIT_DT + list(extra))


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


def _read(out, name, **kw):
    return pd.read_csv(out / name, float_precision="round_trip", **kw)


def _u2_run():
    """The U2 fit (with --eval-theta at the truth, for U8), in process, run once."""
    if "u2" in _STATE:
        return _STATE["u2"]
    import passive_fitting_hpc_fixed as mono
    orig_gp, orig_build = mono.gp_minimize, mono._build_loss_function
    out = _new_out("u2")
    t0 = time.time()
    logtxt = _run_main(_fit_args(out, "ih5", *(U2_OPTS + ["--eval-theta", _theta_arg(THETA5)])))
    secs = time.time() - t0
    # the shim of the contract restores the builder patched by integrate_long_step: the object
    # in place before the fit (main's own integrate_long_step installs a fresh one each run)
    _STATE["u2"] = dict(out=out, res=_read(out, "phase2_results.csv"),
                        flags=pd.read_csv(out / "phase2_results.csv", dtype=str, keep_default_na=False),
                        starts=_read(out, "refine_starts.csv"), log=logtxt, secs=secs,
                        gp_restored=(mono.gp_minimize is orig_gp),
                        builder_after=mono._build_loss_function, builder_before=orig_build)
    return _STATE["u2"]


def check_close(name, actual, expected, rtol=0.0, atol=0.0):
    assert_allclose(np.asarray(actual), np.asarray(expected), rtol=rtol, atol=atol, err_msg=name)


def _parse(extra):
    return R._parse_args(["--archive-dir", "/a", "--output-dir", "/o", "--code-dir", "."] + list(extra))


def _exit_code(fn):
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            fn()
    except SystemExit as exc:
        return exc.code
    return None


def _spec(arm="ih5", *extra):
    a = _parse(["--arm", arm] + list(extra))
    return R.build_spec_from_args(a, R.resolve_arm_config(a))


def _unit(spec, theta):
    out = []
    for ax in spec.axes:
        lo, hi = ax.q_bounds()
        out.append((ax.to_q(float(theta[ax.name])) - lo) / (hi - lo))
    return np.asarray(out, dtype=float)


# ---------------------------------------------------------------- checks ---
def test_U1_arm_table():
    import local_refine as LR
    expect = {"ih5": "trf", "ih6_d207": "trf", "passive_fullstep": "trf", "ih6": "none",
              "baseline_runB": "none"}
    bad = []
    if tuple(R.REFINE_CHOICES) != ("none", "nelder_mead", "trf"):
        bad.append("REFINE_CHOICES %s" % (R.REFINE_CHOICES,))
    if set(R.ARMS) != set(expect):
        bad.append("arms %s" % sorted(R.ARMS))
    for arm, want in expect.items():
        c = R.resolve_arm_config(_parse(["--arm", arm]))
        if c.refine != want or R.ARMS[arm].refine != want:
            bad.append("arm %s refine %r, not %r" % (arm, c.refine, want))
        for val in R.REFINE_CHOICES:
            c = R.resolve_arm_config(_parse(["--arm", arm, "--refine", val]))
            if (c.refine != val or c.fit_params != R.ARMS[arm].fit_params
                    or c.weighting != R.ARMS[arm].weighting):
                bad.append("%s --refine %s -> %s" % (arm, val, c))
    p = _parse(["--arm", "ih5"])
    for name, want in (("refine", None), ("refine_starts", 3), ("refine_maxfev", 100), ("refine_max_nfev", 30),
                       ("sensitivity_theta", None), ("sensitivity_out", None), ("sensitivity_h", 1e-3),
                       ("sensitivity_check_h", 1e-2), ("n_workers", 1)):
        if getattr(p, name) != want:
            bad.append("default --%s = %r, not %r" % (name.replace("_", "-"), getattr(p, name), want))
    opts = ["--refine-starts", "2", "--refine-max-nfev", "8", "--refine-maxfev", "12"]
    a = _parse(["--arm", "ih5"] + opts)
    rc = R.refine_config_from_args(a, R.resolve_arm_config(a))
    if rc != LR.TrfConfig(n_starts=2, max_nfev=8):
        bad.append("trf config %s" % (rc,))
    a = _parse(["--arm", "ih5", "--refine", "nelder_mead"] + opts)
    rc = R.refine_config_from_args(a, R.resolve_arm_config(a))
    if rc != LR.RefineConfig(n_starts=2, maxfev_per_start=12):
        bad.append("nelder_mead config %s" % (rc,))
    a = _parse(["--arm", "ih5", "--refine", "none"] + opts)
    if R.refine_config_from_args(a, R.resolve_arm_config(a)) is not None:
        bad.append("--refine none has a config")
    # the sensitivity options parse; refusals exit 2
    _parse(["--arm", "ih5", "--sensitivity-theta", _theta_arg(THETA5), "--sensitivity-out", "/s"])
    refusals = (["--refine", "lm"], ["--refine-max-nfev", "0"], ["--refine-max-nfev", "-3"],
                ["--sensitivity-theta", _theta_arg(THETA5)],                       # no --sensitivity-out
                ["--sensitivity-theta", _theta_arg(THETA), "--sensitivity-out", "/s"],   # dv_h not in ih5
                ["--sensitivity-theta", _theta_arg(THETA5).replace("Cm=", "Cm:"), "--sensitivity-out", "/s"],
                ["--sensitivity-h", "0"], ["--sensitivity-h", "0.6"], ["--sensitivity-check-h", "-0.01"],
                ["--sensitivity-check-h", "0.7"])
    for extra in refusals:
        code = _exit_code(lambda: _parse(["--arm", "ih5"] + extra))
        if code != 2:
            bad.append("%s exited %r, not 2" % (extra, code))
    if _exit_code(lambda: _parse(["--arm", "ih5", "--sensitivity-check-h", "0"])) is not None:
        bad.append("--sensitivity-check-h 0 refused")
    assert not bad, "; ".join(bad)


def test_U2_trf_fit():
    st = _u2_run()
    res, starts, flags = st["res"], st["starts"], st["flags"]
    bad = []
    if (st["out"] / "failed_cells.txt").exists():
        bad.append("failed_cells.txt written")
    assert len(res) == 1, "phase2_results.csv has %d rows" % len(res)
    row = res.iloc[0]
    spec = _spec("ih5")
    if str(row["refine"]) != "trf" or str(row["param_names"]) != "|".join(R.IH5_NAMES):
        bad.append("refine %r, param_names %r" % (row["refine"], row["param_names"]))
    new = list(R.TRF_RECORD_COLUMNS) + ["on_bound_" + n for n in spec.names]
    cols = list(res.columns)
    for c in new:
        if c not in cols:
            bad.append("column %s missing" % c)
    if all(c in cols for c in new):
        if cols[-len(new):] != new:
            bad.append("the Block 27 columns are not the last ones, in order: %s" % cols[-len(new):])
        if min(cols.index(c) for c in new) < max(cols.index(c) for c in R.REFINE_RECORD_COLUMNS):
            bad.append("the Block 27 columns are not after the Block 20 ones")
    for c in R.TRF_RECORD_COLUMNS:
        if c in cols and not np.isfinite(float(row[c])):
            bad.append("%s = %r not finite" % (c, row[c]))
    for n in spec.names:
        v = str(flags.iloc[0].get("on_bound_" + n, "?"))
        if v not in ("", "lo", "hi"):
            bad.append("on_bound_%s = %r" % (n, v))
    l2s, l2r = float(row["l2_after_search"]), float(row["l2_after_refine"])
    ls, lr = float(row["loss_after_search"]), float(row["loss_after_refine"])
    le, sq = float(row["loss_at_estimate"]), float(row["sqrt_l2_at_estimate"])
    if not l2r <= l2s:
        bad.append("l2_after_refine %r > l2_after_search %r" % (l2r, l2s))
    if bool(row["refine_improved"]) != (l2r < l2s):
        bad.append("refine_improved %r with l2 %r -> %r" % (row["refine_improved"], l2s, l2r))
    if not sq >= le:
        bad.append("sqrt_l2_at_estimate %r < loss_at_estimate %r" % (sq, le))
    if int(row["refine_n_failed"]) != 0:
        bad.append("refine_n_failed %r on the noise-free fixture" % row["refine_n_failed"])
    # refine_starts.csv: the appended columns, finite, consistent with the row
    if list(starts.columns) != R.refine_starts_columns(spec):
        bad.append("refine_starts.csv columns %s" % list(starts.columns))
    if not (1 <= len(starts) <= 2) or len(starts) != int(row["refine_n_starts"]):
        bad.append("refine_starts.csv %d rows, refine_n_starts %r" % (len(starts), row["refine_n_starts"]))
    num = starts.drop(columns=["specimen_id", "converged"]).to_numpy(dtype=float)
    if not np.all(np.isfinite(num)):
        bad.append("refine_starts.csv has non-finite entries")
    if int(starts["n_sims"].sum()) != int(row["refine_n_evals"]):
        bad.append("sum n_sims %d != refine_n_evals %r" % (int(starts["n_sims"].sum()), row["refine_n_evals"]))
    if int(starts["converged"].astype(bool).sum()) != int(row["refine_n_converged"]):
        bad.append("converged starts %d != refine_n_converged %r"
                   % (int(starts["converged"].astype(bool).sum()), row["refine_n_converged"]))
    if float(starts["l2"].min()) != l2r:
        bad.append("lowest start l2 %r != l2_after_refine %r" % (float(starts["l2"].min()), l2r))
    if float(starts["l2_0"].iloc[0]) != l2s or float(starts["f0"].iloc[0]) != ls:
        bad.append("first start l2_0 %r / f0 %r != l2_after_search %r / loss_after_search %r"
                   % (float(starts["l2_0"].iloc[0]), float(starts["f0"].iloc[0]), l2s, ls))
    sv = starts[["sv_%d" % (k + 1) for k in range(len(spec.names))]].to_numpy(dtype=float)
    if not np.all(np.diff(sv, axis=1) <= 0):
        bad.append("sv not descending: %s" % sv)
    if not st["gp_restored"]:
        bad.append("mono.gp_minimize is not the original object after the fit")
    if getattr(st["builder_after"], "wrapped_builder", None) is not None:
        bad.append("mono._build_loss_function is still the capturing shim after the fit")
    # restored also when fit_one_cell raises, and the shims in place while it runs
    import passive_fitting_hpc_fixed as mono
    orig_gp, orig_fit = mono.gp_minimize, mono.fit_one_cell
    seen = []

    def _boom(*a, **k):
        seen.append((getattr(mono.gp_minimize, "wrapped_minimizer", None) is orig_gp,
                     getattr(mono._build_loss_function, "wrapped_builder", None) is not None))
        raise RuntimeError("planted failure inside fit_one_cell")
    mono.fit_one_cell = _boom
    try:
        try:
            _run_main(_fit_args(_new_out("boom"), "ih5", "--n-calls", "4", "--n-initial", "2"))
        except SystemExit:
            pass                                        # no successful fit: main exits 1
    finally:
        mono.fit_one_cell = orig_fit
    if seen != [(True, True)]:
        bad.append("shims not in place during fit_one_cell: %s" % seen)
    if mono.gp_minimize is not orig_gp:
        mono.gp_minimize = orig_gp
        bad.append("mono.gp_minimize not restored after an exception")
    if getattr(mono._build_loss_function, "wrapped_builder", None) is not None:
        mono._build_loss_function = mono._build_loss_function.wrapped_builder
        bad.append("mono._build_loss_function not restored after an exception")
    print("  [U2] %.0f s; L: search %.10g -> refined %.10g; L2: search %.10g -> refined %.10g; %d start(s), "
          "%d converged, %d residual vectors; sqrt(L2) at estimate %.10g; flags %s"
          % (st["secs"], ls, lr, l2s, l2r, len(starts), int(row["refine_n_converged"]),
             int(row["refine_n_evals"]), sq,
             ",".join("%s=%s" % (n, flags.iloc[0]["on_bound_" + n] or "-") for n in spec.names)))
    for k, s in starts.iterrows():
        print("  [U2] start %d: status %d, converged %s, optimality %.3g, nfev %d, njev %d, n_sims %d, l2 %.6g -> %.6g"
              % (k, int(s["status"]), bool(s["converged"]), float(s["optimality"]), int(s["nfev"]), int(s["nit"]),
                 int(s["n_sims"]), float(s["l2_0"]), float(s["l2"])))
    print("  [U2] estimate %s" % ", ".join("%s=%.6g" % (n, float(row[n])) for n in R.IH5_NAMES))
    assert not bad, "; ".join(bad)
    # same deterministic loss at the same q up to the exp/log round trip of the estimate (SPEC 1e-9)
    check_close("U2_loss_at_estimate", le, lr, rtol=1e-9)
    check_close("U2_train_rel_loss", float(row["train_rel_loss"]), lr, rtol=1e-9)
    check_close("U2_sqrt_l2", sq ** 2, l2r, rtol=1e-9)


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


def test_U3_refine_none():
    tmp = Path(tempfile.mkdtemp(prefix="smoke_ih_trf_contract_"))
    atexit.register(shutil.rmtree, str(tmp), True)
    old_py = _contract_run_ih_fit(tmp)
    out_old, out_new = _new_out("u3_contract"), _new_out("u3_now")
    ev = ["--eval-theta", _theta_arg(THETA5)]
    # the contract's CLI has no --refine-max-nfev: the options it knows, the same values
    old_opts = ["--n-calls", "12", "--n-initial", "8", "--refine-starts", "2"]
    _sub([sys.executable, str(old_py)] + _fit_args(out_old, "ih5", "--refine", "none", *(old_opts + ev)),
         "contract run_ih_fit.py")
    _sub([sys.executable, str(HERE / "run_ih_fit.py")] + _fit_args(out_new, "ih5", "--refine", "none",
                                                                  *(U2_OPTS + ev)), "run_ih_fit.py")
    old = pd.read_csv(out_old / "phase2_results.csv", dtype=str, keep_default_na=False)
    new = pd.read_csv(out_new / "phase2_results.csv", dtype=str, keep_default_na=False)
    bad = []
    if list(new.columns[:len(old.columns)]) != list(old.columns):
        bad.append("the old columns are not the first ones, in their order")
    n_cmp = 0
    for c in old.columns:
        if c == "wall_time_s":
            continue
        n_cmp += 1
        if c not in new.columns or list(old[c]) != list(new[c]):
            bad.append("%s: %s != %s" % (c, list(old[c]), list(new.get(c, []))))
    for f in ("ls_roles.csv", "ls_diagnostics.csv"):
        if (out_old / f).read_bytes() != (out_new / f).read_bytes():
            bad.append("%s differs" % f)
    row = new.iloc[0]
    for c, w in (("refine", "none"), ("l2_after_search", ""), ("l2_after_refine", ""),
                 ("refine_n_converged", "0"), ("refine_n_failed", "0")):
        if str(row.get(c)) != w:
            bad.append("%s = %r, not %r" % (c, row.get(c), w))
    if not np.isfinite(float(row["sqrt_l2_at_estimate"])):
        bad.append("sqrt_l2_at_estimate %r without refinement" % row["sqrt_l2_at_estimate"])
    st = pd.read_csv(out_new / "refine_starts.csv")
    spec = _spec("ih5")
    if len(st) != 0 or list(st.columns) != R.refine_starts_columns(spec):
        bad.append("refine_starts.csv without refinement: %d rows, columns %s" % (len(st), list(st.columns)))
    # the trf run's loss_after_search = the search's best: same search, same best point; the two
    # values are the same L by two routes (the loss; l_and_l2 of the residual vector), 1e-9 relative
    trf_ls = float(_u2_run()["res"].iloc[0]["loss_after_search"])
    best = float(row["loss_after_search"])
    print("  [U3] %d old columns compared text for text (wall_time_s excluded); search best %.17g, "
          "trf loss_after_search %.17g (rel. diff %.2e)" % (n_cmp, best, trf_ls, abs(trf_ls - best) / best))
    assert not bad, "; ".join(bad[:6])
    check_close("U3_search_best", trf_ls, best, rtol=1e-9)


def test_U4_workers():
    outs = {}
    for w in (1, 2):
        out = _new_out("u4_w%d" % w)
        _sub([sys.executable, str(HERE / "run_ih_fit.py")] + _fit_args(out, "ih5", *(U2_OPTS + [
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
    print("  [U4] W=1 and W=2: %d columns compared text for text; wall %s s vs %s s; l2_after_refine %s"
          % (len(a.columns) - 2, a["wall_time_s"][0], b["wall_time_s"][0], a["l2_after_refine"][0]))
    assert not bad, "; ".join(bad[:5])


def test_U5_start_near_truth():
    import fd_jacobian as FJ
    import passive_fitting_hpc_fixed as mono
    spec = _spec("ih5")
    bounds = [ax.q_bounds() for ax in spec.axes]
    u_t = _unit(spec, THETA5)
    step = np.where(u_t + 0.05 <= 1.0, 0.05, -0.05)
    u_s = u_t + step
    q_s = [float(v) for v in FJ.to_q(u_s, bounds)]
    orig = mono.gp_minimize
    canned = {}

    def _fake(func, dimensions, **kw):
        # a real (short) search for the GP models fit_one_cell reads afterwards, then the canned
        # global result: one evaluated point, the start
        res = orig(func, dimensions, **kw)
        f = float(func(list(q_s)))
        res.x_iters, res.func_vals, res.x, res.fun = [list(q_s)], np.asarray([f]), list(q_s), f
        canned["f"] = f
        return res
    out = _new_out("u5")
    mono.gp_minimize = _fake
    try:
        _run_main(_fit_args(out, "ih5", "--n-calls", "4", "--n-initial", "2", "--refine-starts", "1"))
    finally:
        mono.gp_minimize = orig
    row = _read(out, "phase2_results.csv").iloc[0]
    starts = _read(out, "refine_starts.csv")
    u_hat = _unit(spec, {n: float(row[n]) for n in spec.names})
    dist = float(np.max(np.abs(u_hat - u_t)))
    dist0 = float(np.max(np.abs(u_s - u_t)))
    print("  [U5] start at %.3g (unit, max-norm) from the truth, L %.6g; estimate at %.3e, L2 %.6g -> %.6g; "
          "status %d, converged %s, nfev %d, njev %d, n_sims %d; per axis %s"
          % (dist0, canned.get("f", float("nan")), dist, float(row["l2_after_search"]),
             float(row["l2_after_refine"]), int(starts["status"][0]), bool(starts["converged"][0]),
             int(starts["nfev"][0]), int(starts["nit"][0]), int(starts["n_sims"][0]),
             ", ".join("%s %.2e" % (n, abs(a - b)) for n, a, b in zip(spec.names, u_hat, u_t))))
    assert len(starts) == 1 and int(starts["search_index"][0]) == 0, starts
    check_close("U5_start", [float(starts["x0_" + n][0]) for n in spec.names],
                [spec.to_physical(q_s)[n] for n in spec.names], rtol=1e-12)
    assert bool(starts["converged"][0]), "the start did not converge (status %d)" % int(starts["status"][0])
    # 5e-3 (SPEC U5): ten times the gap between scipy's xtol (1e-4, relative) and the 1e-3 FD step
    assert dist < 5e-3, "estimate %.3e from the truth (unit, max-norm)" % dist


def test_U6_bound_flags():
    bad = []
    for u, want in ((0.0, "lo"), (0.0099, "lo"), (0.01, ""), (0.0100001, ""), (0.5, ""), (0.99, ""),
                    (0.9900001, "hi"), (1.0, "hi"), (-0.1, "lo"), (1.2, "hi"), (float("nan"), ""),
                    (float("inf"), "")):
        got = R.bound_flag(u)
        if got != want:
            bad.append("bound_flag(%r) = %r, not %r" % (u, got, want))
    spec5 = _spec("ih5")
    cases = [
        (dict(THETA5), {}),                                          # the truth: interior
        (dict(THETA5, Cm=0.3, Rm=1e5), {"Cm": "lo", "Rm": "hi"}),     # on the faces (log axes)
        (dict(THETA5, gbar=1.0e-5 * 1.04), {"gbar": "lo"}),           # u = ln 1.04 / ln 100 = 0.0085
        (dict(THETA5, gbar=1.0e-5 * 1.06), {}),                       # u = 0.0127
        (dict(THETA5, gbar=1.0e-3 / 1.04), {"gbar": "hi"}),
        ({k: v for k, v in THETA5.items() if k != "Ra"}, {}),        # missing: ""
        (dict(THETA5, Ra=float("nan")), {}),
    ]
    for theta, want in cases:
        got = R.bound_flags(spec5, theta)
        exp = {"on_bound_" + n: want.get(n, "") for n in spec5.names}
        if got != exp:
            bad.append("bound_flags(%s) = %s, not %s" % (theta, got, exp))
    spec6 = _spec("ih6_d207")                                        # dv_h: linear axis [-10, 10]
    for dv, want in ((-10.0, "lo"), (-9.9, "lo"), (-9.7, ""), (9.7, ""), (9.9, "hi"), (10.0, "hi"), (0.0, "")):
        got = R.bound_flags(spec6, dict(THETA, dv_h=dv))["on_bound_dv_h"]
        if got != want:
            bad.append("dv_h=%g -> %r, not %r" % (dv, got, want))
    if R.BOUND_FLAG_UNIT != 0.01:
        bad.append("BOUND_FLAG_UNIT %r" % R.BOUND_FLAG_UNIT)
    assert not bad, "; ".join(bad)


def test_U7_nelder_mead():
    out = _new_out("u7")
    _run_main(_fit_args(out, "ih5", "--refine", "nelder_mead", "--n-calls", "6", "--n-initial", "3",
                        "--refine-starts", "2", "--refine-maxfev", "8"))
    row = _read(out, "phase2_results.csv").iloc[0]
    st = _read(out, "refine_starts.csv")
    spec = _spec("ih5")
    bad = []
    if str(row["refine"]) != "nelder_mead":
        bad.append("refine %r" % row["refine"])
    for c in ("l2_after_search", "l2_after_refine"):
        if not np.isnan(float(row[c])):
            bad.append("%s = %r, not NaN" % (c, row[c]))
    for c in ("refine_n_converged", "refine_n_failed"):
        if int(row[c]) != 0:
            bad.append("%s = %r, not 0" % (c, row[c]))
    if not np.isfinite(float(row["sqrt_l2_at_estimate"])):
        bad.append("sqrt_l2_at_estimate %r" % row["sqrt_l2_at_estimate"])
    if list(st.columns) != R.refine_starts_columns(spec) or len(st) != int(row["refine_n_starts"]) or len(st) == 0:
        bad.append("refine_starts.csv %d rows, columns %s" % (len(st), list(st.columns)))
    nan_cols = ["l2_0", "l2", "optimality"] + ["sv_%d" % (k + 1) for k in range(len(spec.names))]
    for c in nan_cols:
        if not st[c].isna().all():
            bad.append("%s not NaN: %s" % (c, list(st[c])))
    if list(st["converged"].astype(bool)) != list(st["status"] == 0):
        bad.append("converged %s, status %s" % (list(st["converged"]), list(st["status"])))
    if list(st["n_sims"]) != list(st["nfev"]):
        bad.append("n_sims %s != nfev %s" % (list(st["n_sims"]), list(st["nfev"])))
    for n in spec.names:
        if list(st["x_solver_" + n]) != list(st["x_" + n]):
            bad.append("x_solver_%s != x_%s" % (n, n))
        if list(st["active_" + n]) != [0] * len(st):
            bad.append("active_%s %s" % (n, list(st["active_" + n])))
    print("  [U7] nelder_mead: %d start(s), status %s, n_sims %s; loss %.6g -> %.6g; sqrt_l2_at_estimate %.6g"
          % (len(st), list(st["status"]), list(st["n_sims"]), float(row["loss_after_search"]),
             float(row["loss_after_refine"]), float(row["sqrt_l2_at_estimate"])))
    assert not bad, "; ".join(bad)


def _u8_run():
    if "u8" in _STATE:
        return _STATE["u8"]
    sens_dir = _new_out("u8_sens")
    out = _new_out("u8")
    t0 = time.time()
    logtxt = _run_main(_fit_args(out, "ih5", "--sensitivity-theta", _theta_arg(THETA5),
                                 "--sensitivity-out", str(sens_dir)))
    _STATE["u8"] = dict(dir=sens_dir, out=out, log=logtxt, secs=time.time() - t0)
    return _STATE["u8"]


def test_U8_sensitivity():
    import passive_fitting_hpc_fixed as mono
    import trace_sensitivity as TS
    st = _u8_run()
    d = st["dir"]
    sens_files = sorted(d.glob("sensitivity_*.npz"))
    assert len(sens_files) == 1, "sensitivity files %s" % sens_files
    sid = int(sens_files[0].stem.split("_")[1])
    bad = []
    for name in ("sensitivity_%d.npz", "sensitivity_%d.json", "bundles_%d.npz", "identifiability_%d.csv",
                 "truth_trace_check_%d.csv"):
        if not (d / (name % sid)).exists():
            bad.append("%s missing" % (name % sid))
    if (st["out"] / "phase2_results.csv").exists():
        bad.append("a fit was made in sensitivity mode (phase2_results.csv)")
    assert not bad, "; ".join(bad)
    meta = json.loads((d / ("sensitivity_%d.json" % sid)).read_text())
    sens = TS.load_sensitivities(d / ("sensitivity_%d.npz" % sid))
    if sens["meta"] != meta:
        bad.append("the npz's meta differs from the json")
    spec = _spec("ih5")
    if sens["names"] != list(spec.names) or meta["names"] != list(spec.names):
        bad.append("names %s / %s" % (sens["names"], meta["names"]))
    check_close("U8_q", sens["q"], spec.to_q(THETA5), rtol=0, atol=0)
    for k in ("specimen_id", "bounds", "theta", "v_rest_mV", "arm", "ss_window_ms", "ls_window_mode",
              "ls_window_ms_after_onset", "r_in_target", "weighting", "ls_sample_weight", "ls_tau_d_ms",
              "ss_time_weight", "ss_tau_w_ms", "ss_t0_ms", "dt_brief_ms", "dt_long_ms", "h", "check_h", "n_workers"):
        if k not in meta:
            bad.append("meta lacks %s" % k)
    if (meta.get("arm"), meta.get("weighting"), meta.get("ls_sample_weight"), meta.get("h"), meta.get("check_h")) != \
            ("ih5", "ls_amplitude", "peak_decay", 1e-3, 1e-2):
        bad.append("meta %s" % meta)
    idf = pd.read_csv(d / ("identifiability_%d.csv" % sid), float_precision="round_trip")
    tt = pd.read_csv(d / ("truth_trace_check_%d.csv" % sid), float_precision="round_trip")
    p = len(spec.names)
    if list(idf["weighting"]) != ["W0", "W1"]:
        bad.append("weightings %s" % list(idf["weighting"]))
    w0, w1 = idf.iloc[0], idf.iloc[1]
    if float(w1["ls_tau_d_ms"]) != 1e12 or float(w0["ls_tau_d_ms"]) != 500.0:
        bad.append("ls_tau_d_ms %r / %r" % (w0["ls_tau_d_ms"], w1["ls_tau_d_ms"]))
    for w in (w0, w1):
        sig = np.asarray([float(w["sigma_%d" % (k + 1)]) for k in range(p)])
        if not (np.all(np.diff(sig) <= 0) and sig[-1] > 0):
            bad.append("%s sigma %s" % (w["weighting"], sig))
    sig0 = np.asarray([float(w0["sigma_%d" % (k + 1)]) for k in range(p)])
    sig1 = np.asarray([float(w1["sigma_%d" % (k + 1)]) for k in range(p)])
    if np.array_equal(sig0, sig1):
        bad.append("W1 equals W0")
    fd = float(w0["fd_step_check"])
    if not fd < 0.05:
        bad.append("fd_step_check %r" % fd)
    for c in ("fd_step_check", "loss_at_theta", "sqrt_l2_at_theta", "v_deepest_train_mV", "m_inf_at_v_deepest"):
        if not np.isnan(float(w1[c])):
            bad.append("W1 %s = %r, not NaN" % (c, w1[c]))
    # sqrt(L2) at the truth: the stored traces through the stored layout vs the fit's own residuals
    sq_fit = float(_u2_run()["res"].iloc[0]["sqrt_l2_at_eval_theta"])
    sq_sens = float(w0["sqrt_l2_at_theta"])
    # layout_from_stored vs the live layout of the fit's own loss
    lay, tix = R.layout_from_stored(d / ("bundles_%d.npz" % sid), meta)
    a = _parse(_fit_args(st["out"], "ih5"))
    cfg = R.resolve_arm_config(a)
    cells = R.load_group_cells(mono, a, cfg, verbose=False)
    oi = mono.prepare_optimiser_inputs(cells[0], fit_target=a.fit_target)
    live = mono._build_loss_function(None, oi.train_bundles, float(oi.v_rest_mV), tuple(oi.train_window_ms),
                                     spec=spec).make_residuals().layout
    if tix != list(range(len(oi.train_bundles))):
        bad.append("train_index %s for %d training bundles" % (tix, len(oi.train_bundles)))
    if len(lay.masks) != len(live.masks):
        bad.append("layout bundles %d vs live %d" % (len(lay.masks), len(live.masks)))
    else:
        for b in range(len(live.masks)):
            if not (np.array_equal(lay.sqrt_a[b], live.sqrt_a[b]) and np.array_equal(lay.masks[b], live.masks[b])):
                bad.append("bundle %d: sqrt_a or masks differ" % b)
        if not np.array_equal(lay.D, live.D):
            bad.append("D differs: %s vs %s" % (lay.D, live.D))
    lay1, _ = R.layout_from_stored(d / ("bundles_%d.npz" % sid), meta, ls_tau_d_ms=1e12)
    if all(np.array_equal(x, y) for x, y in zip(lay1.sqrt_a, lay.sqrt_a)):
        bad.append("the W1 layout equals the W0 one")
    # the truth trace check
    tmax = float(tt["max_abs_mV"].max())
    n_b = len(sens["traces0"])
    if len(tt) != n_b or list(tt["bundle"]) != list(range(n_b)):
        bad.append("truth_trace_check rows %d for %d bundles" % (len(tt), n_b))
    print("  [U8] %.0f s; %d bundles (%d training); sigma W0 = [%s], ratio %.3g; W1 = [%s]; fd_step_check %.3e"
          % (st["secs"], n_b, len(tix), ", ".join("%.4g" % v for v in sig0), float(w0["sigma_ratio"]),
             ", ".join("%.4g" % v for v in sig1), fd))
    print("  [U8] sqrt_l2_at_theta %.17g; the fit's sqrt_l2_at_eval_theta %.17g (rel. diff %.2e); loss_at_theta %.6g"
          % (sq_sens, sq_fit, abs(sq_sens - sq_fit) / max(abs(sq_fit), 1e-300), float(w0["loss_at_theta"])))
    print("  [U8] truth trace check: max |d| %.3e mV (bundle %d); per bundle %s"
          % (tmax, int(tt["max_abs_mV"].idxmax()), ", ".join("%.2e" % v for v in tt["max_abs_mV"])))
    print("  [U8] truth trace RMS (mV): brief windows max %.2e; long steps first 5 ms max %.2e, rest of step max "
          "%.2e, after offset max %.2e" % (tt["rms_window"].max(), tt["rms_first5ms"].max(),
                                           tt["rms_rest_of_step"].max(), tt["rms_after_offset"].max()))
    if not tmax < 1e-6:
        print("  [U8] NOTE: the truth trace check's maximum %.3e mV is not below 1e-6 mV (SPEC U8: reported)"
              % tmax)
    assert not bad, "; ".join(bad)
    # the same residual vector by two routes (stored traces + stored layout; the fit's residual
    # function at --eval-theta): same simulations, same layout arithmetic; 1e-9 relative (SPEC U8)
    check_close("U8_sqrt_l2", sq_sens, sq_fit, rtol=1e-9)


def test_U9_box():
    out_box = dict(THETA5, Cm=3.5)                           # the box of Cm is [0.3, 3.0]
    bad = []
    code = _exit_code(lambda: _parse(["--arm", "ih5", "--sensitivity-theta", _theta_arg(out_box),
                                      "--sensitivity-out", "/s"]))
    if code != 2:
        bad.append("parser: Cm=3.5 exited %r" % code)
    kb = dict(THETA5, kappa_tau=1.3)                         # inside [0.5, 2], outside a tau box of 50-300 ms
    code = _exit_code(lambda: _parse(["--arm", "ih5", "--tau-h-pk-bounds-ms", "50,300", "--sensitivity-theta",
                                      _theta_arg(kb), "--sensitivity-out", "/s"]))
    if code != 2:
        bad.append("parser: kappa_tau outside the box in ms exited %r" % code)
    st = _fixture()
    code = _exit_code(lambda: R.main(["--archive-dir", str(st["group"]), "--output-dir", str(_new_out("u9")),
                                      "--code-dir", str(HERE), "--arm", "ih5", "--sensitivity-theta",
                                      _theta_arg(dict(THETA5, gbar=2e-3)), "--sensitivity-out",
                                      str(_new_out("u9s"))]))
    if code != 2:
        bad.append("main: gbar=2e-3 exited %r" % code)
    # the faces themselves are inside
    face = dict(THETA5, Cm=0.3, Rm=1e5)
    if _exit_code(lambda: _parse(["--arm", "ih5", "--sensitivity-theta", _theta_arg(face),
                                  "--sensitivity-out", "/s"])) is not None:
        bad.append("a point on the faces was refused")
    assert not bad, "; ".join(bad)


def test_U10_ascii_lf():
    bad = []
    for f in ("run_ih_fit.py", "smoke_ih_trf.py", "smoke_ih_refine.py", "smoke_ih5.py", "smoke_ih_fit.py"):
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
    ap.add_argument("--only", default="", help="comma-separated check ids to run, e.g. U1,U2")
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
