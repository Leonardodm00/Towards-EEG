"""Smoke test for the five-parameter arm ih5 and what a fit records -- Block 15 in specs/SPEC.md (I1-I8).

The fixture: a noise-free twin archive built at a known truth by Block 12's own functions
(twin_build.build_twin_cell + simulate_twin(with_noise=False) + write_twin_archive) on one
real-layout fixture cell (smoke_noise_windows.write_fixture: two SS sweeps with trace-QC drops;
LS sweeps of -130, -110, -90, -70, -50, 0, +30 and +60 pA, i.e. n = 5 hyperpolarising steps)
with a ball-and-stick reconstruction.swc (synthetic_ground_truth.write_ball_and_stick_swc).
The truth is the fit's own model, I_h with dv_h = 0: THETA below. No noise and no tilt.

Checks
    test_I1_arm_table          (pure, no NEURON) each arm's names and protocol (role rule,
                               weighting, LS sample weight) against SPEC Block 15 written out
                               here; the defaults None keep the arm's; each override applies
    test_I2_tau_base_peak      tau_h_base_peak_ms(IhSpec()) equals a brute-force maximum of
                               human_ih_params.mtau_ms over a 0.01 mV grid of [-140, -40] mV
                               (1e-3 ms) and lies in [340, 346] ms; --tau-h-pk-bounds-ms 50,1000
                               gives the kappa box (50/base, 1000/base) in the spec, whatever
                               --kappa-bounds says
    test_I3_ih5_fit            run_ih_fit.main --arm ih5 --n-calls 6 --n-initial 3 on the
                               fixture: it runs; the spec has the five names; every set_ih call
                               puts dv_h = 0 on the cell; ls_roles.csv follows d207 (SPEC 2.8 (b)
                               for n = 5: train h2, h4, h5; validate h1, h3 and the depolarising
                               steps; nothing report-only); the new columns are present and
                               finite; loss_at_estimate = train_rel_loss (1e-9 relative)
    test_I4_eval_theta         --eval-theta at the truth: loss_at_eval_theta finite, below
                               loss_at_estimate of that short fit and below the loss at the
                               truth with gbar x 3 (a second run of the same builder); the two
                               runs differ in nothing else (the estimate is back on the cell);
                               malformed --eval-theta values exit 2
    test_I5_gh_at_v_deepest    gh_at_v_deepest_S_cm2 = gbar x human_ih_params.minf(v*) recomputed
                               (1e-12 relative), v* recomputed from ls_roles.csv's train rows
    test_I6_ih6_regression     the arm ih6 with no new option (and --gbar-bounds=1e-6,1e-3)
                               reproduces the phase2_results.csv of the contract commit's
                               run_ih_fit.py on the fixture: every old column text-identical
                               except wall_time_s; ls_roles.csv and ls_diagnostics.csv identical
    test_I7_noise_free         the noise-free archive is fitted without error (row present,
                               no failed_cells.txt, empty error_message; the noise estimate
                               may be 0)
    test_I8_ascii_lf           the four Block 15 files are pure ASCII with LF line endings

Run (from this folder: NEURON loads ./x86_64 from the working directory)
    python smoke_ih5.py [--build] [--only I1,I2]
        --build runs `nrnivmodl mod` first when ./x86_64/special is missing.
        Prints the environment and a PASS / FAIL / ERROR / TODO / SKIP table,
        then the full message of every failure. Exit code 1 if any check
        failed, errored or is still TODO.

Debug
    A failed check_close() saves both arrays to smoke_artifacts/<check name>.npz.

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
import re
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

SEED = 12345
REPORT_PACKAGES = ("numpy", "scipy", "pandas", "scikit-optimize", "neuron")
ARTIFACT_DIR = HERE / "smoke_artifacts"
_SAVED_ARTIFACTS = []
#: the contract commit of Block 15 (I6 runs its run_ih_fit.py)
CONTRACT = "aac46d3"
_CODE_REL = "Passive Features/HPC script/Ih Fit"

# ---------------------------------------------------------------- fixture ---
REAL_ID = 900000003                       # write_fixture's specimen id (the "real" cell)
TWIN_ID = 900001501
SS_SWEEPS = (92, 93)
SS_DROPPED = {92: (2, 3, 11, 20), 93: (0, 5, 27)}
LS_AMPS = (-130.0, -110.0, -90.0, -70.0, -50.0, 0.0, 30.0, 60.0)  # sweeps 54..61; 0 pA is no step
V_REST = -60.0                            # the fixture's metadata v_rest_mV
F = 1.9
THETA = dict(Cm=1.0, Rm=20000.0, Ra=150.0, gbar=1.2e-4, dv_h=0.0, kappa_tau=1.3)
THETA5 = {n: THETA[n] for n in R.IH5_NAMES}
#: SPEC 2.8 (b) for n = 5 (|I| increasing: h1 = -50 ... h5 = -130 pA), m = ceil(5/2) = 3:
#: train h2, h4, h5; validate h1, h3; the depolarising steps validate (dep_n_validation 3).
D207_ROLES_N5 = {-50.0: "validate", -70.0: "train", -90.0: "validate", -110.0: "train",
                 -130.0: "train", 30.0: "validate", 60.0: "validate"}
FIT_BUDGET = ["--n-calls", "6", "--n-initial", "3"]
_LOSS_CRASH_PENALTY = 1e6                 # run_ih_fit._LOSS_CRASH_PENALTY
NEW_COLUMNS = R.FIT_RECORD_COLUMNS
NEW_FLOAT_COLUMNS = ("tau_h_pk_ms", "tau_h_base_pk_ms", "ls_tau_d_ms", "loss_at_estimate",
                     "loss_at_eval_theta", "v_deepest_train_mV", "gh_at_v_deepest_S_cm2")

_STATE = {}


def _theta_arg(theta):
    return ",".join("%s=%r" % (k, float(v)) for k, v in theta.items())


def _hand_table(protocol):
    """The twins' table: every SS sweep and every LS step of the fixture, with a
    placeholder law (the twin is built without noise, so the law is never drawn)."""
    import smoke_noise_windows as snw
    import twin_table as tt
    law = dict(sigma_f_mV=0.03, sigma_m_mV=0.02, tau_m_s=7.1e-5, sigma_s_mV=0.08, tau_s_s=0.21,
               tilt_mV_per_s=0.0, law_source="sweep", noise_flags="", cell_tau_s_above_window=False)
    rows = []
    for s in SS_SWEEPS:
        rows.append(dict(specimen_id=REAL_ID, protocol="SS", sweep_number=s, n_windows=37 - len(SS_DROPPED[s]),
                         amplitude_pA=np.nan, onset_s=np.nan, duration_s=np.nan, n_samples=snw.PRE_N + snw.POST_N,
                         fs_Hz=snw.FS, real_spike_reason="", in_twin=True, **law))
    for k, amp in enumerate(LS_AMPS):
        if amp == 0.0:
            continue                          # not a step: no row (Block 5)
        rows.append(dict(specimen_id=REAL_ID, protocol="LS", sweep_number=54 + k, n_windows=1, amplitude_pA=amp,
                         onset_s=snw.LS_PRE_N / snw.FS,
                         duration_s=(snw.LS_PRE_N + snw.LS_STEP_N) / snw.FS - snw.LS_PRE_N / snw.FS,
                         n_samples=snw.LS_PRE_N + snw.LS_STEP_N + snw.LS_TAIL_N, fs_Hz=snw.FS,
                         real_spike_reason="", in_twin=True, **law))
    df = pd.DataFrame(rows)[list(tt.TWIN_SWEEP_COLUMNS)]
    for c in ("specimen_id", "sweep_number", "n_windows", "n_samples"):
        df[c] = df[c].astype(np.int64)
    for c in ("in_twin", "cell_tau_s_above_window"):
        df[c] = df[c].astype(bool)
    return df


def _clear():
    import synthetic_ground_truth as sgt
    sgt._clear_neuron_sections()


def _fixture():
    """The group directory holding the noise-free twin (built once)."""
    if "group" in _STATE:
        return _STATE
    import ih_mechanism as IM
    import param_spec as PS
    import smoke_noise_windows as snw
    import synthetic_ground_truth as sgt
    import twin_build as tb
    import twin_table as tt
    from twin_config import TwinConfig
    root = Path(tempfile.mkdtemp(prefix="smoke ih5 "))       # a space, as on the cluster's paths
    atexit.register(shutil.rmtree, str(root), True)
    real = root / "real" / ("specimen_%d" % REAL_ID)
    snw.write_fixture(real, np.random.default_rng(SEED), with_response=True, dropped=SS_DROPPED,
                      sweeps=SS_SWEEPS, ls_amps=LS_AMPS)
    sgt.write_ball_and_stick_swc(real / "reconstruction.swc")
    meta = json.load(open(real / "metadata.json"))
    protocol = tt.load_cell_protocol(real, spike_reason_fn=lambda v_mV, sampling_rate_Hz: "")
    table_path = root / "twin_sweeps_FIX.csv"
    tt.write_twin_sweep_table(_hand_table(protocol), table_path)
    table = tt.read_twin_sweep_table(table_path)
    cfg = TwinConfig()
    spec = tb.TwinSpec(twin_id=TWIN_ID, twin_of_specimen_id=REAL_ID, configuration="full",
                       theta=dict(THETA), v_rest_mV=V_REST, F=F)
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
             "--arm", arm] + FIT_BUDGET + ["--phase3-subset", "none"] + list(extra))


def _run_main(argv):
    """run_ih_fit.main in process, its log kept (printed on failure)."""
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


def _ih5_run():
    """The fit of I3 (with --eval-theta at the truth, for I4), run once; the
    dv_h of every set_ih call recorded by a spy."""
    if "ih5" in _STATE:
        return _STATE["ih5"]
    import ih_mechanism as IM
    out = _new_out("ih5")
    calls = []
    orig = IM.set_ih

    def _spy(cell, gbar_S_cm2, dv_h_mV=0.0, kappa_tau=1.0):
        calls.append((float(gbar_S_cm2), float(dv_h_mV), float(kappa_tau)))
        return orig(cell, gbar_S_cm2, dv_h_mV, kappa_tau)
    IM.set_ih = _spy
    try:
        logtxt = _run_main(_fit_args(out, "ih5", "--tau-h-pk-bounds-ms", "50,1000",
                                     "--eval-theta", _theta_arg(THETA5)))
    finally:
        IM.set_ih = orig
    res = pd.read_csv(out / "phase2_results.csv")
    roles = pd.read_csv(out / "ls_roles.csv")
    _STATE["ih5"] = dict(out=out, res=res, roles=roles, set_ih=calls, log=logtxt)
    return _STATE["ih5"]


def check_close(name, actual, expected, rtol=0.0, atol=0.0):
    """assert_allclose that keeps the evidence (smoke_artifacts/<name>.npz) on failure."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    try:
        assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=name)
    except AssertionError:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        path = ARTIFACT_DIR / ("%s.npz" % re.sub(r"[^A-Za-z0-9_.-]+", "_", name))
        np.savez(path, actual=actual, expected=expected)
        _SAVED_ARTIFACTS.append(path)
        raise


def _parse(extra):
    return R._parse_args(["--archive-dir", "/a", "--output-dir", "/o", "--code-dir", "."] + list(extra))


# ---------------------------------------------------------------- checks ---
def test_I1_arm_table():
    # SPEC Block 15, written out: (fit_params, role_rule, weighting, ls_sample_weight)
    expect = {
        "ih5": (("Cm", "Rm", "Ra", "gbar", "kappa_tau"), "d207", "ls_amplitude", "peak_decay"),
        "passive_fullstep": (("Cm", "Rm", "Ra"), "d207", "ls_amplitude", "peak_decay"),
        "ih6": (("Cm", "Rm", "Ra", "gbar", "dv_h", "kappa_tau"), "d006", "relative", "none"),
        "baseline_runB": (("Cm", "Rm", "Ra"), "d006", "relative", "none"),
        # Block 20 (SPEC 2.9 (a)): a fifth arm [adapted in Block 20]
        "ih6_d207": (("Cm", "Rm", "Ra", "gbar", "dv_h", "kappa_tau"), "d207", "ls_amplitude", "peak_decay"),
    }
    bad = []
    assert R.IH5_NAMES == ("Cm", "Rm", "Ra", "gbar", "kappa_tau"), R.IH5_NAMES
    assert set(R.ARMS) == set(expect), sorted(R.ARMS)
    for arm, exp in expect.items():
        c = R.resolve_arm_config(_parse(["--arm", arm]))
        got = (tuple(c.fit_params), c.role_rule, c.weighting, c.ls_sample_weight)
        if got != exp:
            bad.append("arm %s: %s != %s" % (arm, got, exp))
    # ih5 is ih6 in everything else ("as ih6 with ...")
    a5, a6 = R.ARMS["ih5"], R.ARMS["ih6"]
    for f in ("attach_ih", "ih_protocol", "ls_window_mode", "ls_window_ms", "ls_max_amplitude_pA",
              "load_depolarising_ls", "n_long_train", "ls_deflection_cap_mV", "gate_valid_via"):
        if getattr(a5, f) != getattr(a6, f):
            bad.append("ih5.%s = %r differs from ih6's %r" % (f, getattr(a5, f), getattr(a6, f)))
    # the parser's defaults are None (= the arm's) and --ls-tau-d-ms is 500
    a = _parse(["--arm", "ih5"])
    for name, want in (("ls_role_rule", None), ("weighting", None), ("ls_sample_weight", None),
                       ("ls_tau_d_ms", 500.0), ("tau_h_pk_bounds_ms", None), ("eval_theta", None),
                       ("gbar_bounds", "1e-5,1e-3")):        # Block 20, SPEC 2.9 (a) [adapted]
        if getattr(a, name) != want:
            bad.append("default --%s = %r, not %r" % (name.replace("_", "-"), getattr(a, name), want))
    import param_spec as PS
    if tuple(PS.DEFAULT_GBAR_BOUNDS) != (1e-5, 1e-3):              # Block 20 [adapted]
        bad.append("param_spec.DEFAULT_GBAR_BOUNDS = %s" % (PS.DEFAULT_GBAR_BOUNDS,))
    # each override applies, alone, on both protocols
    for arm in ("ih5", "ih6"):
        for flag, field, val in (("--ls-role-rule", "role_rule", "d006"), ("--ls-role-rule", "role_rule", "d207"),
                                 ("--weighting", "weighting", "relative"),
                                 ("--weighting", "weighting", "ls_amplitude"),
                                 ("--ls-sample-weight", "ls_sample_weight", "none"),
                                 ("--ls-sample-weight", "ls_sample_weight", "peak_decay")):
            c = R.resolve_arm_config(_parse(["--arm", arm, flag, val]))
            base = R.ARMS[arm]
            others = [f for f in ("role_rule", "weighting", "ls_sample_weight") if f != field]
            if getattr(c, field) != val or any(getattr(c, f) != getattr(base, f) for f in others):
                bad.append("%s %s %s -> %s" % (arm, flag, val, c))
    # a refused choice exits 2
    for extra in (["--ls-role-rule", "d999"], ["--ls-sample-weight", "uniform"]):
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                _parse(["--arm", "ih5"] + extra)
            bad.append("parser accepted %s" % extra)
        except SystemExit as exc:
            if exc.code != 2:
                bad.append("%s exited %r, not 2" % (extra, exc.code))
    assert not bad, "; ".join(bad)


def test_I2_tau_base_peak():
    import human_ih_params as HIP
    import ih_mechanism as IM
    base = R.tau_h_base_peak_ms(IM.IhSpec())
    # brute force: 0.01 mV grid of [-140, -40] mV (10001 points), the mechanism's own mtau at kappa = 1
    grid = np.linspace(-140.0, -40.0, 10001)
    brute = max(HIP.mtau_ms(float(v), "Ih_human", vshift_mV=0.0, tau_scale=1.0) for v in grid)
    # 1e-3 ms (SPEC I2): both scan the same 0.01 mV grid, so the difference is grid-point
    # roundoff only; a 0.1 mV scan would miss by about 1.8e-3 ms.
    assert abs(base - brute) <= 1e-3, "tau_h_base_peak_ms %.6f vs brute force %.6f ms" % (base, brute)
    assert 340.0 <= base <= 346.0, "tau_h_base_pk = %.4f ms not in [340, 346]" % base
    # the box in ms replaces --kappa-bounds
    a = _parse(["--arm", "ih5", "--tau-h-pk-bounds-ms", "50,1000", "--kappa-bounds", "0.5,2.0"])
    spec = R.build_spec_from_args(a, R.resolve_arm_config(a))
    ax = spec.axis("kappa_tau")
    # same float arithmetic as the code (LO / base): exact up to one rounding
    check_close("I2_kappa_box", [ax.lo, ax.hi], [50.0 / base, 1000.0 / base], rtol=1e-15)
    assert ax.log, "kappa_tau axis is not log-scaled"
    assert spec.names == R.IH5_NAMES, spec.names
    # the default box of gbar is Block 20's (SPEC 2.9 (a)) [adapted in Block 20; was 1e-6, 1e-2]
    check_close("I2_gbar_box", [spec.axis("gbar").lo, spec.axis("gbar").hi], [1e-5, 1e-3], rtol=0.0)
    # without the option the kappa box is --kappa-bounds
    a = _parse(["--arm", "ih5", "--kappa-bounds", "0.5,2.0"])
    ax = R.build_spec_from_args(a, R.resolve_arm_config(a)).axis("kappa_tau")
    assert (ax.lo, ax.hi) == (0.5, 2.0), (ax.lo, ax.hi)
    print("  [I2] tau_h_base_pk = %.6f ms (brute force %.6f ms); kappa box [%.6f, %.6f] = [50, 1000] ms"
          % (base, brute, 50.0 / base, 1000.0 / base))


def test_I3_ih5_fit():
    st = _ih5_run()
    res, roles = st["res"], st["roles"]
    bad = []
    if (st["out"] / "failed_cells.txt").exists():
        bad.append("failed_cells.txt written")
    assert len(res) == 1, "phase2_results.csv has %d rows" % len(res)
    row = res.iloc[0]
    if str(row["arm"]) != "ih5":
        bad.append("arm column %r" % row["arm"])
    if str(row["param_names"]) != "|".join(R.IH5_NAMES):
        bad.append("param_names %r" % row["param_names"])
    if "dv_h" in res.columns:
        bad.append("a dv_h column was written for ih5")
    # dv_h on the cell is 0 at every evaluation
    dvs = sorted({c[1] for c in st["set_ih"]})
    if not st["set_ih"] or dvs != [0.0]:
        bad.append("set_ih dv_h values %s over %d calls" % (dvs, len(st["set_ih"])))
    # ls_roles.csv follows d207 (SPEC 2.8 (b), written out above for n = 5); asserted last,
    # so that every other part of the check is evaluated first
    role_bad = []
    got = {float(r.amplitude_pA): str(r.role) for r in roles.itertuples()}
    if got != D207_ROLES_N5:
        role_bad.append("ls_roles.csv roles %s != d207 %s" % (got, D207_ROLES_N5))
    for r in roles.itertuples():
        if str(r.role) == "train" and "d207" not in str(r.reason):
            role_bad.append("train row %+.0f pA reason %r does not name d207" % (r.amplitude_pA, r.reason))
    # the new columns: present, finite floats, the protocol strings
    for c in NEW_COLUMNS:
        if c not in res.columns:
            bad.append("column %s missing" % c)
    for c in NEW_FLOAT_COLUMNS:
        if c in res.columns and not np.isfinite(float(row[c])):
            bad.append("%s = %r not finite" % (c, row[c]))
    for c, want in (("ls_role_rule", "d207"), ("weighting", "ls_amplitude"), ("ls_sample_weight", "peak_decay")):
        if c in res.columns and str(row[c]) != want:
            bad.append("%s = %r, not %r" % (c, row[c], want))
    if "ls_tau_d_ms" in res.columns and float(row["ls_tau_d_ms"]) != 500.0:
        bad.append("ls_tau_d_ms = %r" % row["ls_tau_d_ms"])
    # the new columns are written after every existing one
    # (Block 20's columns come after Block 15's: they are not "existing" here [adapted in Block 20])
    # (and Block 27's after Block 20's [adapted in Block 27])
    old_last = max(list(res.columns).index(c) for c in res.columns
                   if c not in NEW_COLUMNS and c not in R.REFINE_RECORD_COLUMNS
                   and c not in R.TRF_RECORD_COLUMNS and not c.startswith("on_bound_"))
    if min(list(res.columns).index(c) for c in NEW_COLUMNS if c in res.columns) < old_last:
        bad.append("the Block 15 columns are not after the existing ones")
    assert not bad, "; ".join(bad)
    # tau_h_pk = kappa_tau x base (1e-12: one product of float64)
    check_close("I3_tau_h_pk", float(row["tau_h_pk_ms"]),
                float(row["kappa_tau"]) * float(row["tau_h_base_pk_ms"]), rtol=1e-12)
    # loss_at_estimate = train_rel_loss: the same deterministic loss at the same q, up to the
    # exp/log round trip of the estimate (a few ulp in q); 1e-9 relative (SPEC I3)
    le, tr = float(row["loss_at_estimate"]), float(row["train_rel_loss"])
    assert tr < _LOSS_CRASH_PENALTY, "train_rel_loss %g at the crash penalty" % tr
    check_close("I3_loss_at_estimate", le, tr, rtol=1e-9)
    print("  [I3] theta = %s; tau_h_pk = %.4g ms; loss at estimate %.17g = train_rel_loss %.17g"
          % (", ".join("%s=%.4g" % (n, float(row[n])) for n in R.IH5_NAMES), float(row["tau_h_pk_ms"]), le, tr))
    assert not role_bad, "; ".join(role_bad)


def test_I4_eval_theta():
    st = _ih5_run()
    row = st["res"].iloc[0]
    l_truth, l_est = float(row["loss_at_eval_theta"]), float(row["loss_at_estimate"])
    # the same builder at the truth with gbar x 3: a second run with that --eval-theta
    th3 = dict(THETA5, gbar=3.0 * THETA5["gbar"])
    out3 = _new_out("ih5_gbar3")
    _run_main(_fit_args(out3, "ih5", "--tau-h-pk-bounds-ms", "50,1000", "--eval-theta", _theta_arg(th3)))
    res3 = pd.read_csv(out3 / "phase2_results.csv")
    l_g3 = float(res3.iloc[0]["loss_at_eval_theta"])
    # --eval-theta changes nothing else: the two runs fit the same cell with the same seed, so
    # every other column (the gate, the I_h summary, ...) is identical, and ls_diagnostics.csv too
    # (the estimate is back on the cell after the loss at the given theta was evaluated)
    # (sqrt_l2_at_eval_theta is the same point's sqrt(L_2) [adapted in Block 27])
    leak = [c for c in res3.columns if c not in ("loss_at_eval_theta", "sqrt_l2_at_eval_theta", "wall_time_s")
            and not (str(res3.iloc[0][c]) == str(row[c]))]
    if (out3 / "ls_diagnostics.csv").read_bytes() != (st["out"] / "ls_diagnostics.csv").read_bytes():
        leak.append("ls_diagnostics.csv")
    assert not leak, "--eval-theta changed other outputs: %s" % leak
    print("  [I4] loss at the truth %.6g; at the 6-call estimate %.6g; at the truth with gbar x 3 %.6g"
          % (l_truth, l_est, l_g3))
    assert np.isfinite(l_truth) and l_truth < _LOSS_CRASH_PENALTY, l_truth
    assert l_truth < l_est, "loss at the truth %g not below the estimate's %g" % (l_truth, l_est)
    assert l_truth < l_g3, "loss at the truth %g not below gbar x 3's %g" % (l_truth, l_g3)
    # malformed --eval-theta: exit 2 (argparse's usage error)
    full = _theta_arg(THETA5)
    malformed = [
        _theta_arg({k: v for k, v in THETA5.items() if k != "kappa_tau"}),     # a name missing
        full + ",dv_h=0.0",                                                     # a name not in ih5
        full + ",Cm=1.0",                                                       # a name twice
        full.replace("Rm=", "Rm:"),                                             # not name=value
        full.replace("Ra=150.0", "Ra=abc"),                                     # not a number
        full.replace("gbar=0.00012", "gbar=nan"),                               # not finite
        full.replace("Cm=1.0", "Cm=-1.0"),                                      # a positive quantity < 0
        "",                                                                     # nothing
    ]
    bad = []
    for m in malformed:
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                _parse(["--arm", "ih5", "--eval-theta", m])
            bad.append("accepted %r" % m)
        except SystemExit as exc:
            if exc.code != 2:
                bad.append("%r exited %r" % (m, exc.code))
    # and the arm's own names are what counts: ih6 needs dv_h
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            _parse(["--arm", "ih6", "--eval-theta", full])
        bad.append("ih6 accepted an eval-theta without dv_h")
    except SystemExit as exc:
        if exc.code != 2:
            bad.append("ih6 without dv_h exited %r" % exc.code)
    _parse(["--arm", "ih6", "--eval-theta", full + ",dv_h=-1.5"])          # a signed shift is accepted
    assert not bad, "; ".join(bad)


def test_I5_gh_at_v_deepest():
    import human_ih_params as HIP
    st = _ih5_run()
    row, roles = st["res"].iloc[0], st["roles"]
    v_star = float(roles.loc[roles["role"] == "train", "v_trough_mV"].min())
    check_close("I5_v_deepest", float(row["v_deepest_train_mV"]), v_star, rtol=0.0)
    gh = float(row["gbar"]) * HIP.minf(v_star, "Ih_human", vshift_mV=0.0, vshift_minf_mV=0.0)
    # 1e-12 relative: one product, the CSV round trip of pandas is exact for float64
    check_close("I5_gh", float(row["gh_at_v_deepest_S_cm2"]), gh, rtol=1e-12)
    print("  [I5] v* = %.4f mV, gbar = %.4g, m_inf(v*) = %.5f, g_h(v*) = %.5g S/cm^2"
          % (v_star, float(row["gbar"]), gh / float(row["gbar"]), gh))


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


def _sub(cmd, label):
    r = subprocess.run(cmd, cwd=str(HERE), capture_output=True, text=True, timeout=3600)
    if r.returncode != 0:
        raise AssertionError("%s exited %d:\n%s\n%s" % (label, r.returncode, r.stdout[-3000:], r.stderr[-3000:]))


def test_I6_ih6_regression():
    st = _fixture()
    tmp = Path(tempfile.mkdtemp(prefix="smoke_ih5_contract_"))
    atexit.register(shutil.rmtree, str(tmp), True)
    old_py = _contract_run_ih_fit(tmp)
    out_old, out_new = _new_out("ih6_contract"), _new_out("ih6_now")
    # each in its own process: no NEURON state shared between the two fits
    _sub([sys.executable, str(old_py)] + _fit_args(out_old, "ih6"), "contract run_ih_fit.py")
    _sub([sys.executable, str(HERE / "run_ih_fit.py")] + _fit_args(out_new, "ih6", "--gbar-bounds=1e-6,1e-3"),
         "run_ih_fit.py")
    bad = []
    old = pd.read_csv(out_old / "phase2_results.csv", dtype=str, keep_default_na=False)
    new = pd.read_csv(out_new / "phase2_results.csv", dtype=str, keep_default_na=False)
    if len(old) != len(new):
        bad.append("rows %d vs %d" % (len(old), len(new)))
    if list(new.columns[:len(old.columns)]) != list(old.columns):
        bad.append("the old columns are not the first ones, in their order")
    n_cmp = 0
    for c in old.columns:
        if c == "wall_time_s":                 # the fit's own clock: never reproducible
            continue
        if c not in new.columns:
            bad.append("column %s gone" % c)
            continue
        n_cmp += 1
        if list(old[c]) != list(new[c]):
            bad.append("%s: %s != %s" % (c, list(old[c]), list(new[c])))
    for f in ("ls_roles.csv", "ls_diagnostics.csv"):
        if (out_old / f).read_bytes() != (out_new / f).read_bytes():
            bad.append("%s differs" % f)
    row = new.iloc[0]
    for c, want in (("ls_role_rule", "d006"), ("weighting", "relative"), ("ls_sample_weight", "none")):
        if str(row.get(c, "")) != want:
            bad.append("ih6 %s = %r" % (c, row.get(c, "")))
    if str(row.get("loss_at_eval_theta", "x")) != "":
        bad.append("loss_at_eval_theta %r without --eval-theta (NaN expected)" % row.get("loss_at_eval_theta"))
    print("  [I6] %d old columns compared text for text (wall_time_s excluded); %d new columns"
          % (n_cmp, len(new.columns) - len(old.columns)))
    assert not bad, "; ".join(bad)


def test_I7_noise_free():
    st = _ih5_run()
    row = st["res"].iloc[0]
    bad = []
    if (st["out"] / "failed_cells.txt").exists():
        bad.append("failed_cells.txt: %s" % (st["out"] / "failed_cells.txt").read_text().strip())
    msg = row.get("error_message", "")
    if isinstance(msg, str) and msg.strip():
        bad.append("error_message %r" % msg)
    # (validation_status grades the fit's quality, here of a 6-call fit; it is not an error)
    # the archive is noise-free: the fit's noise estimate is ~0 (or NaN), never an error
    sig = float(row["noise_sigma_mV"])
    print("  [I7] noise-free archive fitted: status %s, noise_sigma_mV = %r, error_message %r"
          % (row["validation_status"], sig, msg))
    assert not bad, "; ".join(bad)


def test_I8_ascii_lf():
    bad = []
    for f in ("run_ih_fit.py", "param_spec.py", "smoke_ih5.py", "smoke_ih_fit.py"):
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


def ensure_build(build: bool) -> None:
    special = HERE / "x86_64" / "special"
    if special.exists():
        return
    if not build:
        sys.exit("x86_64/special not found; run 'nrnivmodl mod' here or pass --build")
    r = subprocess.run(["nrnivmodl", "mod"], cwd=str(HERE), capture_output=True, text=True)
    if r.returncode != 0 or not special.exists():
        sys.exit("nrnivmodl failed:\n" + r.stdout[-2000:] + r.stderr[-2000:])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true", help="run nrnivmodl mod if x86_64/special is missing")
    ap.add_argument("--only", default="", help="comma-separated check ids to run, e.g. I1,I2")
    args = ap.parse_args(argv)
    ensure_build(args.build)
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
    for path in _SAVED_ARTIFACTS:
        print("-- arrays from a failed comparison: %s" % path)
    return 1 if counts["FAIL"] + counts["ERROR"] + counts["TODO"] else 0


if __name__ == "__main__":
    sys.exit(main())
