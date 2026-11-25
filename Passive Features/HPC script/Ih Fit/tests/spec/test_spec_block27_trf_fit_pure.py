"""Spec-derived tests of Block 27 (run_ih_fit.py: the trf refinement's records,
the bound flags, the sensitivity mode's pure helpers; SPEC 2.10 and the Block 27
entry) that need no NEURON. Integration tester, 2026-11-25.

Oracles: the entry's tables (arms, CLI defaults, column order), the bound rule
written out (u < 0.01 'lo', u > 0.99 'hi'), 2.10 (a) through Block 23's
`residual_layout` for the stored bundles (the reference the entry names), and
hand computations of the truth trace check on synthetic arrays.
"""
import argparse
import json
import types

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import local_refine as LR
import passive_long_step_training as plst
import run_ih_fit as R
from smoke_ls_protocol import ls_bundle, ss_bundle


def _parse(*extra):
    return R._parse_args(["--archive-dir", "/a", "--output-dir", "/o", "--code-dir", "."] + list(extra))


def test_arm_table_and_cli_defaults():
    assert R.REFINE_CHOICES == ("none", "nelder_mead", "trf")
    for arm, ref in (("ih5", "trf"), ("ih6_d207", "trf"), ("passive_fullstep", "trf"), ("ih6", "none"),
                     ("baseline_runB", "none")):
        assert R.ARMS[arm].refine == ref
    a = _parse("--arm", "ih5")
    assert a.refine_max_nfev == 30 and a.refine_starts == 3 and a.refine_maxfev == 100
    assert a.sensitivity_h == 1e-3 and a.sensitivity_check_h == 1e-2 and a.sensitivity_theta is None
    c = R.refine_config_from_args(a, R.resolve_arm_config(a))
    assert isinstance(c, LR.TrfConfig) and c.max_nfev == 30 and c.n_starts == 3
    a = _parse("--arm", "ih5", "--refine", "nelder_mead", "--refine-starts", "2", "--refine-maxfev", "7")
    c = R.refine_config_from_args(a, R.resolve_arm_config(a))
    assert isinstance(c, LR.RefineConfig) and c.n_starts == 2 and c.maxfev_per_start == 7
    a = _parse("--arm", "ih5", "--refine", "none")
    assert R.refine_config_from_args(a, R.resolve_arm_config(a)) is None
    a = _parse("--arm", "ih6", "--refine-max-nfev", "5")
    assert R.refine_config_from_args(a, R.resolve_arm_config(a)) is None
    a = _parse("--arm", "ih6", "--refine", "trf", "--refine-max-nfev", "5")
    c = R.refine_config_from_args(a, R.resolve_arm_config(a))
    assert isinstance(c, LR.TrfConfig) and c.max_nfev == 5
    for bad in (["--refine-max-nfev", "0"], ["--sensitivity-h", "0"], ["--sensitivity-h", "0.6"],
                ["--sensitivity-check-h", "-1"], ["--refine", "lbfgs"]):
        with pytest.raises(SystemExit) as ei:
            _parse("--arm", "ih5", *bad)
        assert ei.value.code == 2


def test_sensitivity_theta_box_and_grammar():
    th = "Cm=1.0,Rm=20000.0,Ra=150.0,gbar=1e-4,kappa_tau=1.0"
    a = _parse("--arm", "ih5", "--sensitivity-theta", th, "--sensitivity-out", "/x")
    assert a.sensitivity_theta == th
    spec = R.build_spec_from_args(a, R.resolve_arm_config(a))
    # the box's own corners are inside (closed box): no exit
    lo = ",".join("%s=%r" % (ax.name, ax.lo) for ax in spec.axes)
    _parse("--arm", "ih5", "--sensitivity-theta", lo, "--sensitivity-out", "/x")
    for bad in ("Cm=1.0,Rm=20000.0,Ra=150.0,gbar=1.0,kappa_tau=1.0",       # gbar above the box
                "Cm=1.0,Rm=20000.0,Ra=150.0,gbar=1e-4",                     # a name missing
                "Cm=1.0,Rm=20000.0,Ra=150.0,gbar=1e-4,kappa_tau=1.0,dv_h=0"):  # a name too many
        with pytest.raises(SystemExit) as ei:
            _parse("--arm", "ih5", "--sensitivity-theta", bad, "--sensitivity-out", "/x")
        assert ei.value.code == 2
    with pytest.raises(SystemExit):
        _parse("--arm", "ih5", "--sensitivity-theta", th)                   # no --sensitivity-out


@pytest.mark.parametrize("u,flag", [(0.0, "lo"), (0.0099, "lo"), (0.01, ""), (0.5, ""), (0.99, ""),
                                    (0.9901, "hi"), (1.0, "hi"), (float("nan"), ""), (-0.2, "lo")])
def test_bound_flag(u, flag):
    assert R.bound_flag(u) == flag


def test_bound_flags_on_the_spec():
    a = _parse("--arm", "ih6_d207")
    spec = R.build_spec_from_args(a, R.resolve_arm_config(a))
    theta = {}
    want = {}
    for i, ax in enumerate(spec.axes):
        qlo, qhi = ax.q_bounds()
        u = [0.005, 0.995, 0.5, 0.011, 0.989, 0.0][i]
        theta[ax.name] = ax.to_physical(qlo + u * (qhi - qlo))
        want["on_bound_" + ax.name] = "lo" if u < 0.01 else ("hi" if u > 0.99 else "")
    assert R.bound_flags(spec, theta) == want
    assert set(R.bound_flags(spec, {}).values()) == {""}
    assert list(R.bound_flags(spec, theta)) == ["on_bound_" + n for n in spec.names]


def test_refine_starts_columns_order():
    a = _parse("--arm", "ih5")
    spec = R.build_spec_from_args(a, R.resolve_arm_config(a))
    n = list(spec.names)
    exp = (["specimen_id", "start", "search_index", "f0", "fun", "nfev", "nit", "status"] + ["x0_" + x for x in n]
           + ["x_" + x for x in n] + ["l2_0", "l2", "converged", "optimality", "n_sims"]
           + ["x_solver_" + x for x in n] + ["active_" + x for x in n] + ["sv_%d" % k for k in range(1, 6)])
    assert R.refine_starts_columns(spec) == exp


def _trf_fr(spec):
    """A fit result carrying a real TrfResult of a toy residual function in the spec's q box."""
    bounds = [ax.q_bounds() for ax in spec.axes]
    lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
    target = lo + 0.37 * (hi - lo)
    f = lambda q: (np.asarray(q) - target) / (hi - lo)
    rng = np.random.default_rng(0)
    X = lo + rng.random((12, len(bounds))) * (hi - lo)
    F = np.array([np.linalg.norm(f(x)) for x in X])

    def res(Q):
        return np.array([f(q) for q in Q])
    rr = LR.refine_trf(res, X, F, bounds, LR.TrfConfig(n_starts=2, max_nfev=10),
                       loss_of_row=lambda r: float(np.sum(np.abs(r))))
    gp = types.SimpleNamespace(refine=rr, fun=rr.fun, func_vals=F, x=rr.x)
    return types.SimpleNamespace(gp_result=gp), rr


def test_records_under_trf():
    a = _parse("--arm", "ih5")
    cfg = R.resolve_arm_config(a)
    spec = R.build_spec_from_args(a, cfg)
    fr, rr = _trf_fr(spec)
    rec = R.refine_record(fr, cfg, 3)
    assert rec["refine"] == "trf" and rec["n_workers"] == 3
    assert rec["loss_after_search"] == rr.l_search and rec["loss_after_refine"] == rr.fun
    assert rec["refine_n_evals"] == rr.n_sims and rec["refine_n_starts"] == len(rr.starts)
    assert rec["refine_improved"] == rr.improved and rec["refine_best_start"] == rr.best_start
    t = R.trf_record(fr, 4)
    assert t == dict(l2_after_search=rr.l2_search, l2_after_refine=rr.l2,
                     refine_n_converged=sum(1 for s in rr.starts if s["converged"]), refine_n_failed=4)
    rows = R.refine_start_rows(fr, spec, 77)
    assert len(rows) == len(rr.starts)
    cols = R.refine_starts_columns(spec)
    for row, st in zip(rows, rr.starts):
        assert list(row) == cols
        assert row["f0"] == st["l_0"] and row["fun"] == st["l"] and row["nit"] == st["njev"]
        xs = spec.to_physical(st["x_solver"])
        assert all(row["x_solver_" + n] == xs[n] for n in spec.names)
        assert [row["sv_%d" % (k + 1)] for k in range(5)] == [float(v) for v in st["sv"]]
    # without trf: NaN / 0 defaults
    none = types.SimpleNamespace(gp_result=types.SimpleNamespace(refine=None, fun=0.5))
    t = R.trf_record(none, 9)
    assert np.isnan(t["l2_after_search"]) and np.isnan(t["l2_after_refine"])
    assert t["refine_n_converged"] == 0 and t["refine_n_failed"] == 0
    rec = R.refine_record(none, R.resolve_arm_config(_parse("--arm", "ih6")), 1)
    assert rec["loss_after_search"] == rec["loss_after_refine"] == 0.5 and rec["refine_best_start"] == -1


# ------------------------------------------------------ the stored bundles ---
def _bundles():
    return [ss_bundle(-200.0), ss_bundle(-150.0), ls_bundle(-30.0, sag=0.1), ls_bundle(-70.0, sag=0.3),
            ls_bundle(-110.0, sag=0.5), ls_bundle(20.0, sag=0.0), ss_bundle(100.0)]


IS_TRAIN = [True, True, True, False, True, False, False]


def _meta(**over):
    m = dict(specimen_id=1, names=["Cm", "Rm", "Ra"], ss_window_ms=[0.5, 100.0], ls_window_mode="step",
             ls_window_ms_after_onset=150.0, r_in_target="peak", weighting="ls_amplitude",
             ls_sample_weight="peak_decay", ls_tau_d_ms=500.0, ss_time_weight="exp", ss_tau_w_ms=5.0,
             ss_t0_ms=None)
    m.update(over)
    return m


def test_bundles_npz_round_trip_and_layout_from_stored(tmp_path):
    bs = _bundles()
    for k, b in enumerate(bs):
        b.sweep_numbers = [10 + k, 20 + k]
        b.n_repeats_averaged = k + 1
    p = tmp_path / "bundles_1.npz"
    R.write_bundles_npz(p, bs, IS_TRAIN)
    with np.load(p, allow_pickle=False) as z:
        assert z["is_train"].dtype == bool and z["is_brief"].tolist() == [True, True, False, False, False, False, True]
        assert json.loads(str(z["sweep_numbers"])) == [[10 + k, 20 + k] for k in range(7)]
        assert z["n_repeats_averaged"].tolist() == list(range(1, 8))
    meta = _meta()
    lay, tix = R.layout_from_stored(p, meta)
    assert tix == [0, 1, 2, 4]
    train = [bs[i] for i in tix]
    ref = plst.residual_layout(train, ss_window_ms=(0.5, 100.0), ls_window_mode="step", weighting="ls_amplitude",
                               ls_sample_weight="peak_decay", ls_tau_d_ms=500.0,
                               ss_sample_weight_fn=plst.exp_time_weight(0.5e-3, 5e-3))
    for a_, b_ in zip(lay.sqrt_a, ref.sqrt_a):
        assert_array_equal(a_, b_)
    for a_, b_ in zip(lay.masks, ref.masks):
        assert_array_equal(a_, b_)
    assert_array_equal(lay.D, ref.D)
    assert_array_equal(lay.c, ref.c)
    # overrides: W1 (no decay after the peak) changes only the long bundles' sqrt_a
    l1, _ = R.layout_from_stored(p, json.dumps(meta), ls_tau_d_ms=1e12)
    assert_array_equal(l1.sqrt_a[0], lay.sqrt_a[0])
    assert not np.array_equal(l1.sqrt_a[2], lay.sqrt_a[2])
    # with ls_tau_d_ms = 1e12 the weight after the peak is exp(-t/1e9 s) = 1 to 1e-12
    l0, _ = R.layout_from_stored(p, meta, ls_sample_weight="none")
    assert not np.array_equal(l0.sqrt_a[2], lay.sqrt_a[2])
    # the meta as a JSON file path
    jp = tmp_path / "sensitivity_1.json"
    jp.write_text(json.dumps(meta))
    l2, _ = R.layout_from_stored(p, str(jp))
    assert_array_equal(l2.D, lay.D)


def test_identifiability_table_rows(tmp_path):
    bs = _bundles()
    p = tmp_path / "b.npz"
    R.write_bundles_npz(p, bs, IS_TRAIN)
    meta = _meta()
    rng = np.random.default_rng(3)
    S = [rng.normal(size=(b.t.size, 3)) for b in bs]
    Sc = [s * (1 + 1e-3) for s in S]
    tr0 = [np.asarray(b.v_mV) - np.mean(b.v_mV[:5]) + 0.01 * rng.normal(size=b.t.size) for b in bs]
    rows = R.identifiability_table(p, meta, dict(S=S, traces0=tr0), dict(S=Sc, traces0=tr0), ["Cm", "Rm", "Ra"], 1,
                                   v_deepest_mV=-90.0, m_inf_at_v_deepest=0.4)
    assert [r["weighting"] for r in rows] == ["W0", "W1"]
    assert rows[0]["ls_tau_d_ms"] == 500.0 and rows[1]["ls_tau_d_ms"] == 1e12
    lay, tix = R.layout_from_stored(p, meta)
    import identifiability as IDF
    J = IDF.jacobian_from_sensitivities([S[i] for i in tix], lay)
    sv = np.linalg.svd(J, compute_uv=False)
    assert_allclose([rows[0]["sigma_%d" % k] for k in (1, 2, 3)], sv, rtol=1e-12)
    assert_allclose(rows[0]["fd_step_check"], 1e-3, rtol=1e-9)
    r = np.concatenate([lay.sqrt_a[k] * (tr0[i][lay.masks[k]] - lay.v_exp[k]) / lay.D[k] for k, i in enumerate(tix)])
    L2 = float(np.sum(r ** 2))
    L = sum(lay.c[k] * np.sqrt(np.sum(r[lay.slices[k]] ** 2) / lay.c[k]) for k in range(len(tix)))
    assert_allclose(rows[0]["sqrt_l2_at_theta"], np.sqrt(L2), rtol=1e-12)
    assert_allclose(rows[0]["loss_at_theta"], L, rtol=1e-12)
    assert rows[0]["v_deepest_train_mV"] == -90.0 and rows[0]["m_inf_at_v_deepest"] == 0.4
    assert "fd_step_check" not in rows[1]
    rows_nc = R.identifiability_table(p, meta, dict(S=S, traces0=tr0), None, ["Cm", "Rm", "Ra"], 1)
    assert np.isnan(rows_nc[0]["fd_step_check"])


def test_truth_trace_rows_by_hand():
    bs = [ls_bundle(-50.0), ss_bundle(-200.0)]
    tr = []
    for b in bs:
        v = np.asarray(b.v_mV)
        pm = (b.t >= (0.0 if b.stim_duration_s > 0.01 else -0.01)) & (b.t <= (b.stim_onset_s if b.stim_duration_s > 0.01 else 0.0))
        d = np.zeros_like(v)
        d[b.t.size // 2] = 0.3                       # one planted difference
        tr.append(v - v[pm].mean() + d)
    rows = R.truth_trace_rows(tr, bs, [True, False], (0.5, 100.0), 5)
    long_, brief = rows
    t = bs[0].t
    assert long_["max_abs_mV"] == pytest.approx(0.3, abs=1e-12)
    assert long_["t_max_s"] == t[t.size // 2]
    on, off = 0.1, 0.6
    seg = (t >= on + 5e-3) & (t <= off)
    assert long_["rms_rest_of_step"] == pytest.approx(np.sqrt(0.09 / seg.sum()), rel=1e-9)
    assert long_["rms_first5ms"] == pytest.approx(0.0, abs=1e-12)
    assert long_["rms_after_offset"] == pytest.approx(0.0, abs=1e-12)
    assert np.isnan(long_["rms_window"])
    tb = bs[1].t
    w = (tb >= 0.5e-3) & (tb <= 0.1)
    k = tb.size // 2
    assert brief["rms_window"] == pytest.approx(np.sqrt(0.09 / w.sum()) if w[k] else 0.0, rel=1e-9)
    assert np.isnan(brief["rms_first5ms"])
