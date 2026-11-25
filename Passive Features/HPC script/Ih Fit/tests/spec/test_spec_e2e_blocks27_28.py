"""End to end through the real entry points of Blocks 27 and 28 (NEURON; SPEC
2.10, section 6). Integration tester, 2026-11-25.

`python run_ih_fit.py` (a trf fit, then the sensitivity mode) on the noise-free
fixture twin of smoke_ih_trf.py, and `python run_twin_recovery.py plan /
sensitivity / merge` on the fixture group of smoke_twin_recovery.py. Only the
INPUTS come from those smokes' fixture builders. Checks: every output exists, is
finite, has the columns / keys / shapes the entries state; two runs with the same
seed are identical; the boundary Block 27 -> Block 23: the sensitivity mode's
stored layout (meta + bundles) reproduces the loss the fit minimises (loss and
sqrt(L2) at the same point equal to the fit's eval-theta values).
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal

HERE = Path(__file__).resolve().parents[2]

pytest.importorskip("neuron")

_S = {}


def _fx():
    if "fx" not in _S:
        import smoke_ih_trf as U
        st = U._fixture()
        _S["fx"] = (U, st)
    return _S["fx"]


def _run(args, timeout=3000):
    r = subprocess.run([sys.executable, str(HERE / "run_ih_fit.py")] + [str(a) for a in args], cwd=str(HERE),
                       capture_output=True, text=True, timeout=timeout)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return r


FIT_OPTS = ["--n-calls", "10", "--n-initial", "6", "--refine-starts", "2", "--refine-max-nfev", "4"]


def _fit(tag, *extra):
    key = "fit_" + tag
    if key not in _S:
        U, st = _fx()
        out = st["root"] / ("tester_" + tag)
        _run(U._fit_args(out, "ih5", *(FIT_OPTS + ["--eval-theta", U._theta_arg(U.THETA5)] + list(extra))))
        _S[key] = out
    return _S[key]


def test_trf_fit_outputs_and_reproducibility():
    U, st = _fx()
    a, b = _fit("a"), _fit("b")
    ra = pd.read_csv(a / "phase2_results.csv", float_precision="round_trip")
    rb = pd.read_csv(b / "phase2_results.csv", float_precision="round_trip")
    drop = [c for c in ra.columns if "wall" in c or c.endswith("_s") and "time" in c]
    pd.testing.assert_frame_equal(ra.drop(columns=drop), rb.drop(columns=drop))
    sa = pd.read_csv(a / "refine_starts.csv", float_precision="round_trip")
    sb = pd.read_csv(b / "refine_starts.csv", float_precision="round_trip")
    pd.testing.assert_frame_equal(sa, sb)
    r = ra.iloc[0]
    new = ["l2_after_search", "l2_after_refine", "refine_n_converged", "refine_n_failed", "sqrt_l2_at_estimate",
           "sqrt_l2_at_eval_theta"]
    for c in new:
        assert c in ra.columns and np.isfinite(float(r[c])), c
    # column order: Block 27's after Block 20's
    cols = list(ra.columns)
    assert cols.index("l2_after_search") > cols.index("n_workers")
    # 2.10 (a), (c) identities (1e-9 relative: the same float64 quantity recomputed)
    assert float(r["l2_after_refine"]) <= float(r["l2_after_search"])
    assert_allclose(float(r["sqrt_l2_at_estimate"]) ** 2, float(r["l2_after_refine"]), rtol=1e-9)
    assert_allclose(float(r["loss_at_estimate"]), float(r["loss_after_refine"]), rtol=1e-9)
    assert_allclose(float(r["train_rel_loss"]), float(r["loss_after_refine"]), rtol=1e-9)
    assert float(r["sqrt_l2_at_estimate"]) >= float(r["loss_at_estimate"]) * (1 - 1e-12)
    assert float(r["sqrt_l2_at_eval_theta"]) >= float(r["loss_at_eval_theta"]) * (1 - 1e-12)
    # the bound flags recomputed from the estimate in the spec's box
    import run_ih_fit as R
    a_ = R._parse_args(U._fit_args(a, "ih5", *FIT_OPTS))
    spec = R.build_spec_from_args(a_, R.resolve_arm_config(a_))
    flags = pd.read_csv(a / "phase2_results.csv", dtype=str, keep_default_na=False).iloc[0]
    theta = {n: float(r[n]) for n in spec.names}
    want = R.bound_flags(spec, theta)
    for n in spec.names:
        assert flags["on_bound_" + n] == want["on_bound_" + n]
    assert set(sa.columns) >= {"l2_0", "l2", "converged", "optimality", "n_sims"}
    assert int(sa["n_sims"].sum()) == int(r["refine_n_evals"])
    assert float(sa["l2"].min()) == pytest.approx(float(r["l2_after_refine"]), rel=1e-12)
    assert float(sa["l2_0"].iloc[0]) == pytest.approx(float(r["l2_after_search"]), rel=1e-12)


def _sens(tag, W):
    key = "sens_" + tag
    if key not in _S:
        U, st = _fx()
        out = st["root"] / ("tester_sens_" + tag)
        _run(U._fit_args(out, "ih5", "--n-workers", str(W), "--sensitivity-theta", U._theta_arg(U.THETA5),
                         "--sensitivity-out", str(out / "sens")))
        _S[key] = out / "sens"
    return _S[key]


def test_sensitivity_mode_files_determinism_and_the_boundary_with_the_fit():
    U, st = _fx()
    d1, d2 = _sens("w1", 1), _sens("w2", 2)
    import run_ih_fit as R
    import trace_sensitivity as TS
    files = sorted(p.name for p in d1.iterdir())
    sid = int(files[0].split("_")[-1].split(".")[0])
    for f in ("sensitivity_%d.npz", "sensitivity_%d.json", "bundles_%d.npz", "identifiability_%d.csv",
              "truth_trace_check_%d.csv"):
        assert (d1 / (f % sid)).is_file(), f
    # W = 1 and W = 2: the same arrays exactly (2.9 (c), Block 23's exactness) and run-to-run identical
    s1 = TS.load_sensitivities(d1 / ("sensitivity_%d.npz" % sid))
    s2 = TS.load_sensitivities(d2 / ("sensitivity_%d.npz" % sid))
    for b in range(len(s1["S"])):
        assert np.array_equal(s1["S"][b], s2["S"][b]) and np.array_equal(s1["traces0"][b], s2["traces0"][b])
        assert s1["S"][b].shape == (s1["t_list"][b].size, 5) and np.all(np.isfinite(s1["S"][b]))
    meta = json.loads((d1 / ("sensitivity_%d.json" % sid)).read_text())
    for k in ("specimen_id", "names", "bounds", "theta", "v_rest_mV", "arm", "ss_window_ms", "ls_window_mode",
              "ls_window_ms_after_onset", "r_in_target", "weighting", "ls_sample_weight", "ls_tau_d_ms",
              "dt_brief_ms", "dt_long_ms", "h", "check_h", "n_workers"):
        assert k in meta, k
    assert meta["names"] == list(R.IH5_NAMES) and meta == s1["meta"]
    with np.load(d1 / ("bundles_%d.npz" % sid), allow_pickle=False) as z:
        for k in ("is_train", "is_brief", "amplitude_pA", "stim_onset_s", "stim_duration_s", "n_repeats_averaged",
                  "sweep_numbers"):
            assert k in z.files
        nb = z["is_train"].size
        assert len(s1["S"]) == nb
    idf = pd.read_csv(d1 / ("identifiability_%d.csv" % sid), float_precision="round_trip")
    assert idf["weighting"].tolist() == ["W0", "W1"]
    sig = idf[["sigma_%d" % k for k in range(1, 6)]].to_numpy()
    assert np.all(np.diff(sig, axis=1) <= 0) and np.all(sig > 0)
    assert not np.array_equal(sig[0], sig[1])
    w0 = idf.iloc[0]
    assert float(w0["fd_step_check"]) < 0.05
    # the boundary: the stored protocol reproduces the fit's loss at the same point (fit a ran --eval-theta
    # at the same theta, same dt): L and sqrt(L2) equal to 1e-9 relative
    fa = pd.read_csv(_fit("a") / "phase2_results.csv", float_precision="round_trip").iloc[0]
    assert_allclose(float(w0["loss_at_theta"]), float(fa["loss_at_eval_theta"]), rtol=1e-9)
    assert_allclose(float(w0["sqrt_l2_at_theta"]), float(fa["sqrt_l2_at_eval_theta"]), rtol=1e-9)
    ttc = pd.read_csv(d1 / ("truth_trace_check_%d.csv" % sid))
    assert len(ttc) == nb and np.all(np.isfinite(ttc["max_abs_mV"]))
    _S["ttc_max"] = float(ttc["max_abs_mV"].max())
    _S["ttc_first5"] = float(np.nanmax(ttc["rms_first5ms"]))
    print("truth trace check: max |d| %.4g mV, max rms_first5ms %.3g mV" % (_S["ttc_max"], _S["ttc_first5"]))


# --------------------------------------------------------------- Block 28 --
def test_twin_recovery_sensitivity_end_to_end_reproducible():
    import smoke_twin_recovery as STR
    import run_twin_recovery as T
    out = STR._new_out("tester_sens")
    code, so, se = STR._plan(out, "--refine-max-nfev", "2")
    assert code == 0, se[-2000:]
    cfg = json.loads((out / "run_config.json").read_text())
    assert cfg["options"]["refine"] == "trf" and int(cfg["options"]["refine_max_nfev"]) == 2
    man = STR._manifest(out)
    tid = int(man.loc[man["configuration"] == "full_no_noise", "specimen_id"].iloc[0])
    r1 = STR._sub(["sensitivity", "--run-dir", out, "--twin", tid, "--n-noise", 12])
    assert r1.returncode == 0, r1.stdout[-3000:] + r1.stderr[-3000:]
    f = T.sens_dir_of(out, tid) / "identifiability.csv"
    a = pd.read_csv(f, float_precision="round_trip")
    r2 = STR._sub(["sensitivity", "--run-dir", out, "--twin", tid, "--n-noise", 12])
    assert r2.returncode == 0
    b = pd.read_csv(f, float_precision="round_trip")
    pd.testing.assert_frame_equal(a, b)                         # same seeds, same numbers
    names = [c[len("sd_lin_"):] for c in a.columns if c.startswith("sd_lin_")]
    import run_ih_fit as R
    arm = str(man.loc[man["specimen_id"] == tid, "arm"].iloc[0])
    assert names == list(R.ARMS[arm].fit_params)
    sd = a[["sd_lin_" + n for n in names]].to_numpy()
    assert np.all(np.isfinite(sd)) and np.all(sd > 0)
    assert a["twin_id"].tolist() == [tid, tid] and a["n_noise"].tolist() == [12, 12]
    assert a["weighting"].tolist() == ["W0", "W1"]
    corr = a[[c for c in a.columns if c.startswith("corr_")]].to_numpy()
    assert np.all(np.abs(corr) <= 1 + 1e-12)
    # merge with sensitivity files only (no fit has run): exit 0, the two tables written
    code, so, se = STR._run(["merge", "--out", out, "--no-plot"])
    assert code == 0, se[-2000:]
    m = pd.read_csv(out / "identifiability.csv")
    assert m["twin_id"].unique().tolist() == [tid]
    assert (out / "truth_trace_check.csv").is_file()
