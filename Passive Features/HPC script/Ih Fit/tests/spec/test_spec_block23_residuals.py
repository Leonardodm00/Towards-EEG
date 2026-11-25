"""Spec-derived tests of Block 23 (passive_long_step_training.py: bundle_trace,
residual_layout, assemble_residuals, l_and_l2, build_trace_evaluator,
build_multi_protocol_residuals, loss.make_residuals; SPEC 2.10 (a) and the Block
23 entry). Integration tester, 2026-11-25.

Oracle: 2.10 (a) re-implemented here with numpy from the definitions (windows,
pre-window means with both ends included, linear interpolation of the simulated
trace on the recorded samples, c_b of `relative` / `ls_amplitude`, the sample
weights of 2.8 (d) and an SS exponential weight written out by hand), applied to
the analytic stub cell of the existing smokes (no NEURON; only the INPUT comes
from smoke_ls_protocol). The scalar loss of Block 14 (`build_multi_protocol_loss`)
is the reference for L; pooled evaluations are compared with `==`.
"""
import multiprocessing
import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import param_spec as PS
import passive_long_step_training as plst
from smoke_ls_protocol import V_REST, StubCell, ls_bundle, ss_bundle

DT = 0.1
SS_WIN = (0.5, 100.0)
SPEC = PS.PASSIVE_3D


def _train():
    return [ss_bundle(-200.0), ss_bundle(-150.0), ls_bundle(-30.0, sag=0.1), ls_bundle(-70.0, sag=0.3),
            ls_bundle(-110.0, sag=0.5)]


def _pts(n=6, seed=7):
    rng = np.random.default_rng(seed)
    lo = np.array([a.q_bounds()[0] for a in SPEC.axes])
    hi = np.array([a.q_bounds()[1] for a in SPEC.axes])
    P = lo + rng.uniform(0, 1, (n, 3)) * (hi - lo)
    P[0] = lo                    # two points on faces
    P[1] = hi
    return P


# ------------------------------------------------------------ the oracle ---
def _ss_w(t, t0=0.5e-3, tau=5e-3):
    return np.exp(-np.maximum(t - t0, 0.0) / tau)


def _peak_decay_w(b, tau_d_s):
    """2.8 (d) by hand: boxcar of 2h+1 samples (h = round(0.5 ms / dt), finite samples
    covered), extreme over [t_on + 1 ms, t_off], earliest tie; the weight."""
    t = np.asarray(b.t, float); v = np.asarray(b.v_mV, float)
    dt = float(np.median(np.diff(t)))
    h = int(round(1.0e-3 / 2 / dt))
    sm = np.array([np.mean(v[max(0, i - h):i + h + 1]) for i in range(v.size)])
    on, off = b.stim_onset_s, b.stim_onset_s + b.stim_duration_s
    m = (t >= on + 1e-3) & (t <= off)
    idx = np.flatnonzero(m)
    k = idx[int(np.argmax(sm[idx])) if b.amplitude_pA > 0 else int(np.argmin(sm[idx]))]
    tpk = t[k]

    def w(tt):
        tt = np.asarray(tt, float)
        return np.where(tt < on, 0.0, np.where(tt <= tpk, (tt - on) / (tpk - on), np.exp(-(tt - tpk) / tau_d_s)))
    return w


def _oracle(cell, q, bundles, *, weighting, ls_mode, ls_after_ms=150.0, ss_fn=None, ls_w="none", tau_d_ms=500.0):
    """(r, L, L2, per-bundle (c_b, D_b, sum r^2)) of 2.10 (a), from the definitions."""
    SPEC.apply_q(cell, q, V_REST)
    n = len(bundles)
    brief = [b.stim_duration_s < 0.01 for b in bundles]
    if weighting == "relative":
        c = np.full(n, 1.0 / n)
    else:
        nls = sum(1 for x in brief if not x)
        nss = n - nls
        tot = sum(abs(b.amplitude_pA) for b, x in zip(bundles, brief) if not x)
        c = np.array([1.0 / (nss + nls) if x else nls * abs(b.amplitude_pA) / tot / (nss + nls)
                      for b, x in zip(bundles, brief)])
    rs, per = [], []
    for b, x, cb in zip(bundles, brief, c):
        t = np.asarray(b.t, float); v = np.asarray(b.v_mV, float)
        if x:
            tm, vs = cell.simulate(b.amplitude_pA, 10.0, b.stim_duration_s * 1e3, 10.0 + b.stim_duration_s * 1e3 + 100.0,
                                   v_init_mV=V_REST, dt_ms=DT)
            ts = tm * 1e-3 - 10e-3
            pre = (-10e-3, 0.0); win = (SS_WIN[0] * 1e-3, SS_WIN[1] * 1e-3)
            wf = ss_fn
        else:
            tm, vs = cell.simulate(b.amplitude_pA, b.stim_onset_s * 1e3, b.stim_duration_s * 1e3, t[-1] * 1e3,
                                   v_init_mV=V_REST, dt_ms=DT)
            ts = tm * 1e-3
            pre = (0.0, b.stim_onset_s)
            on = b.stim_onset_s
            win = (on, on + ls_after_ms * 1e-3) if ls_mode == "after_onset" else (on, on + b.stim_duration_s)
            wf = _peak_decay_w(b, tau_d_ms * 1e-3) if ls_w == "peak_decay" else None
        pms = (ts >= pre[0]) & (ts <= pre[1])
        pme = (t >= pre[0]) & (t <= pre[1])
        wm = (t >= win[0]) & (t <= win[1])
        vhat = np.interp(t[wm], ts, vs - vs[pms].mean())
        vexp = (v - v[pme].mean())[wm]
        w = np.ones(wm.sum()) if wf is None else wf(t[wm])
        D = np.max(np.abs(vexp))
        r = np.sqrt(cb * w / w.sum()) * (vhat - vexp) / D
        rs.append(r)
        per.append((cb, D, float(np.sum(r * r))))
    r = np.concatenate(rs)
    L2 = float(np.sum(r * r))
    L = float(sum(cb * np.sqrt(s / cb) for cb, D, s in per))
    return r, L, L2, per


PROTOCOLS = [
    dict(weighting="relative", ls_mode="after_onset", ss_fn=None, ls_w="none"),
    dict(weighting="ls_amplitude", ls_mode="step", ss_fn=_ss_w, ls_w="peak_decay"),
    dict(weighting="ls_amplitude", ls_mode="step", ss_fn=None, ls_w="none"),
]


def _kw(p):
    return dict(weighting=p["weighting"], ls_window_mode=p["ls_mode"], ss_sample_weight_fn=p["ss_fn"],
                ls_sample_weight=p["ls_w"], ss_window_ms=SS_WIN)


@pytest.mark.parametrize("prot", PROTOCOLS)
def test_residual_vector_equals_the_definition(prot):
    tb = _train()
    cell = StubCell()
    R = plst.build_multi_protocol_residuals(cell, tb, V_REST, dt_brief_ms=DT, dt_long_ms=DT, **_kw(prot))
    loss = plst.build_multi_protocol_loss(StubCell(), tb, V_REST, dt_brief_ms=DT, dt_long_ms=DT, **_kw(prot))
    P = _pts()
    rows = R(P)
    assert rows.shape == (len(P), R.n_residuals) and rows.dtype == np.float64
    assert_allclose(R.layout.c.sum(), 1.0, rtol=1e-14)
    for k, q in enumerate(P):
        r, L, L2, per = _oracle(StubCell(), q, tb, weighting=prot["weighting"], ls_mode=prot["ls_mode"],
                                ss_fn=prot["ss_fn"], ls_w=prot["ls_w"])
        # same arithmetic up to summation order: 1e-12 relative on entries (a few ulps
        # of the D-scaled differences), 1e-10 on the sums as the entry asks.
        assert_allclose(rows[k], r, rtol=1e-12, atol=1e-14 * np.max(np.abs(r)))
        Lc, L2c = R.l_and_l2(rows[k])
        assert_allclose(L2c, L2, rtol=1e-10)
        assert_allclose(Lc, L, rtol=1e-10)
        assert Lc <= np.sqrt(L2c) * (1 + 1e-15)
        # L equals the scalar loss of 2.8 (Block 14's code) at the same point
        assert_allclose(Lc, loss(*q), rtol=1e-10)
        # per bundle: sum r^2 = c_b (RMSD_b / D_b)^2 with bundle_rmsd (Block 14)
        SPEC.apply_q(cell, q, V_REST)
        for b, bd in enumerate(tb):
            fn = None if bd.stim_duration_s >= 0.01 else prot["ss_fn"]
            rmsd, defl = plst.bundle_rmsd(cell, bd, V_REST, ss_window_ms=SS_WIN, ls_window_mode=prot["ls_mode"],
                                          ss_sample_weight_fn=fn, dt_brief_ms=DT, dt_long_ms=DT,
                                          ls_sample_weight=prot["ls_w"])
            sl = R.layout.slices[b]
            assert_allclose(np.sum(rows[k][sl] ** 2), R.layout.c[b] * (rmsd / defl) ** 2, rtol=1e-10)
            assert_allclose(R.layout.D[b], per[b][1], rtol=1e-14)


def test_layout_fields_by_hand():
    tb = _train()
    lay = plst.residual_layout(tb, ss_window_ms=SS_WIN, ls_window_mode="step", weighting="ls_amplitude",
                               ls_sample_weight="peak_decay", ss_sample_weight_fn=_ss_w)
    c_ss = 1.0 / 5
    tot = 30.0 + 70.0 + 110.0
    assert_allclose(lay.c, [c_ss, c_ss, 3 * 30 / tot / 5, 3 * 70 / tot / 5, 3 * 110 / tot / 5], rtol=1e-14)
    start = 0
    for b, bd in enumerate(tb):
        t = bd.t
        if bd.stim_duration_s < 0.01:
            pm = (t >= -10e-3) & (t <= 0.0); wm = (t >= 0.5e-3) & (t <= 0.1)
            w = _ss_w(t[wm])
        else:
            pm = (t >= 0.0) & (t <= bd.stim_onset_s)
            wm = (t >= bd.stim_onset_s) & (t <= bd.stim_onset_s + bd.stim_duration_s)
            w = _peak_decay_w(bd, 0.5)(t[wm])
        assert_array_equal(lay.masks[b], wm)
        assert_array_equal(lay.pre_masks[b], pm)
        assert_allclose(lay.sqrt_a[b], np.sqrt(lay.c[b] * w / w.sum()), rtol=1e-13)
        assert_allclose(lay.v_exp[b], (bd.v_mV - bd.v_mV[pm].mean())[wm], rtol=0, atol=1e-12)
        assert lay.slices[b] == slice(start, start + int(wm.sum()))
        start += int(wm.sum())
    assert lay.n_residuals == start


def test_bundle_trace_definition():
    cell = StubCell()
    SPEC.apply_q(cell, _pts()[3], V_REST)
    for bd in _train():
        tr = plst.bundle_trace(cell, bd, V_REST, dt_brief_ms=DT, dt_long_ms=DT)
        t = bd.t
        if bd.stim_duration_s < 0.01:
            tm, vs = cell.simulate(bd.amplitude_pA, 10.0, 0.5, 110.5, v_init_mV=V_REST, dt_ms=DT)
            ts = tm * 1e-3 - 10e-3; pre = (-10e-3, 0.0)
        else:
            tm, vs = cell.simulate(bd.amplitude_pA, bd.stim_onset_s * 1e3, bd.stim_duration_s * 1e3, t[-1] * 1e3,
                                   v_init_mV=V_REST, dt_ms=DT)
            ts = tm * 1e-3; pre = (0.0, bd.stim_onset_s)
        pm = (ts >= pre[0]) & (ts <= pre[1])
        assert tr.shape == t.shape and tr.dtype == np.float64
        assert_allclose(tr, np.interp(t, ts, vs - vs[pm].mean()), rtol=0, atol=1e-12)


@pytest.mark.parametrize("W", [2, 3])
def test_pooled_equals_serial_exactly(W):
    tb = _train()
    kw = _kw(PROTOCOLS[1])
    c1, cw = StubCell(), StubCell()
    R1 = plst.build_multi_protocol_residuals(c1, tb, V_REST, dt_brief_ms=DT, dt_long_ms=DT, **kw)
    RW = plst.build_multi_protocol_residuals(cw, tb, V_REST, dt_brief_ms=DT, dt_long_ms=DT, n_workers=W, **kw)
    try:
        P = _pts()
        a, b = R1(P), RW(P)
        assert np.array_equal(a, b)
        ev1 = R1.evaluator(P)
        evW = RW.evaluator(P)
        for x, y in zip(ev1, evW):
            assert all(np.array_equal(u, v) for u, v in zip(x, y))
        # the parent's cell is at Q[-1] in both
        th = SPEC.to_physical(P[-1])
        for c in (c1, cw):
            assert (c.Cm, c.Rm, c.Ra) == (th["Cm"], th["Rm"], th["Ra"])
        assert RW.n_points == 2 * len(P) and RW.evaluator.n_workers == W
        assert len(multiprocessing.active_children()) <= W
    finally:
        RW.close()
    assert multiprocessing.active_children() == []


class _FailingCell(StubCell):
    """simulate raises (or returns NaN) when Cm exceeds a threshold."""

    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def simulate(self, *a, **k):
        t, v = super().simulate(*a, **k)
        if self.Cm > 2.0:
            if self.mode == "raise":
                raise RuntimeError("planted failure")
            v = v.copy(); v[5] = np.nan
        return t, v


@pytest.mark.parametrize("mode", ["raise", "nan"])
@pytest.mark.parametrize("W", [1, 2])
def test_failed_point_constant_row(mode, W):
    tb = _train()
    R = plst.build_multi_protocol_residuals(_FailingCell(mode), tb, V_REST, dt_brief_ms=DT, dt_long_ms=DT,
                                            n_workers=W, fail_l2=1e6, **_kw(PROTOCOLS[0]))
    ref = plst.build_multi_protocol_residuals(StubCell(), tb, V_REST, dt_brief_ms=DT, dt_long_ms=DT,
                                              **_kw(PROTOCOLS[0]))
    try:
        P = _pts()
        cm = np.exp(P[:, 0])
        bad = cm > 2.0
        assert bad.any() and (~bad).any()
        rows = R(P)
        good = ref(P)
        for k in range(len(P)):
            if bad[k]:
                assert np.all(rows[k] == np.sqrt(1e6 / R.n_residuals))
                assert_allclose(R.l_and_l2(rows[k])[1], 1e6, rtol=1e-12)
            else:
                assert np.array_equal(rows[k], good[k])
        assert R.n_failed == int(bad.sum())
    finally:
        R.close()


class DtEnforcementError(RuntimeError):
    pass


class _DtCell(StubCell):
    def simulate(self, *a, **k):
        raise DtEnforcementError("dt not enforced (planted)")


@pytest.mark.parametrize("W", [1, 2])
def test_dt_enforcement_reaches_the_caller(W):
    R = plst.build_multi_protocol_residuals(_DtCell(), _train(), V_REST, dt_brief_ms=DT, dt_long_ms=DT,
                                            n_workers=W, **_kw(PROTOCOLS[0]))
    try:
        with pytest.raises(Exception) as ei:
            R(_pts()[:2])
        assert type(ei.value).__name__ == "DtEnforcementError"
    finally:
        R.close()


def test_refusals():
    tb = _train()
    with pytest.raises(ValueError):
        plst.residual_layout(tb, weighting="equal")
    with pytest.raises(ValueError):
        plst.residual_layout([])
    bad = ls_bundle(-50.0); bad.v_mV = bad.v_mV.copy(); bad.v_mV[1500] = np.nan   # in the window
    with pytest.raises(ValueError):
        plst.residual_layout([bad], ls_window_mode="step")
    flat = ls_bundle(-50.0, v=np.full(9000, V_REST))                               # D_b = 0
    with pytest.raises(ValueError):
        plst.residual_layout([flat], ls_window_mode="step")
    for w in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            plst.build_trace_evaluator(StubCell(), tb, V_REST, n_workers=w)
    ev = plst.build_trace_evaluator(StubCell(), tb, V_REST, dt_brief_ms=DT, dt_long_ms=DT)
    with pytest.raises(ValueError):
        ev(np.zeros((2, 2)))
    with pytest.raises(ValueError):
        ev(np.array([[0.0, np.nan, 1.0]]))
    with pytest.raises(ValueError):
        plst.build_multi_protocol_residuals(StubCell(), tb, V_REST, fail_l2=0.0)
    for wt in ("equal", "custom"):
        kw = dict(weights=[1.0] * 5) if wt == "custom" else {}
        L = plst.build_multi_protocol_loss(StubCell(), tb, V_REST, weighting=wt, **kw)
        with pytest.raises(ValueError):
            L.make_residuals()


def test_make_residuals_is_the_loss_protocol_and_loss_unchanged():
    tb = _train()
    kw = _kw(PROTOCOLS[1])
    for W in (1, 2):
        L = plst.build_multi_protocol_loss(StubCell(), tb, V_REST, dt_brief_ms=DT, dt_long_ms=DT, n_workers=W, **kw)
        R = L.make_residuals()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                for q in _pts()[:3]:               # alternate the loss's pool and the residuals' pool
                    lv = L(*q)
                    row = R(q[None, :])[0]
                    assert_allclose(R.l_and_l2(row)[0], lv, rtol=1e-10)
                    assert R.evaluator.n_workers == W
        finally:
            R.close()
            L.close()
