"""Smoke test for the residual vector and the trace evaluator -- Block 23 in
specs/SPEC.md (section 2.10 (a)).

bundle_trace, residual_layout, assemble_residuals, l_and_l2,
build_trace_evaluator, build_multi_protocol_residuals and
loss.make_residuals in passive_long_step_training.py: the residual vector
r(q) whose squared norm is L_2 = sum_b c_b (RMSD_b / D_b)^2 and from which
the loss L of section 2.8 is recovered, on a pool of (point, bundle) tasks
whose values are EXACTLY those of the serial loop.

Checks (the oracles of the Block 23 entry, T1-T10)
    T1   >= 6 points of the six-parameter box (two on its faces), protocols
         (ls_amplitude, peak_decay, window step, SS exponential weight),
         (relative, none) and, beyond the entry, (relative, none, window
         after_onset, no SS weight, r_in_target steady): per bundle
         sum_s r_{b,s}^2 == c_b (RMSD_b / D_b)^2 from bundle_rmsd (relative
         1e-10), D_b equal to bundle_rmsd's deflection; l_and_l2(row)[0]
         equals the loss of build_multi_protocol_loss (relative 1e-10);
         sum_b c_b = 1; L <= sqrt(L_2)
    T2   resid(Q) with n_workers 2 and 3 np.array_equal to n_workers 1 for a
         Q of 6 points; ev(Q) likewise (every trace array_equal)
    T3   the parent's cell is at Q[-1] after a call (soma cm, g_pas,
         gIhbar, e_pas), W = 1 and W = 2
    T4   a point made to fail (a patched simulation raising, and one
         returning NaN, for one point): its row is the constant
         sqrt(fail_l2 / N), L_2 = fail_l2, n_failed counts it, the other
         rows unchanged; a DtEnforcementError (the monolith's class, and an
         unpicklable one of that name) reaches the caller under that name,
         serial and pooled
    T5   the refusals of residual_layout, of n_workers, of Q, of fail_l2 and
         of make_residuals under weighting 'equal' / 'custom'
    T6   loss(q) of build_multi_protocol_loss equals the contract commit's bit
         for bit at 4 points, W = 1 and 2; smoke_ls_pool.py and
         smoke_ls_protocol.py still pass
    T7   n_points equals the number of rows asked; close() twice; a call
         after close() works and equals the serial value
    T8   a loss with a live pool and its make_residuals() evaluated
         alternately, warnings as errors: values equal the serial ones, no
         warning, at most one pool's workers alive
    T9   bundle_trace on a long bundle: its window samples minus v_exp,
         weighted, reproduce bundle_rmsd (1e-10) and equal an independent
         subtraction-then-interpolation; constant outside the simulated
         interval for a brief bundle
    T10  ASCII / LF of the block's files

Run (from this directory; NEURON loads ./x86_64 from the CWD; compile it
with `nrnivmodl mod` if it is missing)
    env -u DISPLAY python3 smoke_residuals.py
        Prints the environment and a PASS / FAIL / ERROR / TODO / SKIP table,
        then the full message of every failure. Exit code 1 if any check
        failed, errored or is still TODO.
    T6 needs git (it reads the contract commit's module with `git show`) and
    runs the two other smokes as subprocesses (several minutes).

Keep this file pure ASCII.
"""
from __future__ import annotations

import importlib.metadata
import importlib.util
import multiprocessing
import os
import platform
import signal
import subprocess
import sys
import tempfile
import time
import traceback
import unittest
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.chdir(HERE)                      # NEURON loads ./x86_64 from the CWD
sys.path.insert(0, str(HERE))

import numpy as np                  # noqa: E402

import passive_long_step_training as plst  # noqa: E402
# The fixtures smoke_ls_pool.py uses (one NEURON ball-and-stick cell with
# I_h, the bundles of smoke_ls_protocol.py, the six-parameter spec).
from smoke_ls_pool import (V_REST, DT, SPEC6, neuron_cell, box_points,  # noqa: E402
                           ls_bundle, ss_bundle, _train_set, _CELL)

CONTRACT = "43ae62f"
SEED = 20261125
REPORT_PACKAGES = ("numpy", "scipy", "NEURON")
# Float64 roundoff: a sum of ~1e4 squared terms taken in two different
# orders (np.mean in bundle_rmsd, a weighted sum here) differs by a few
# 1e-16 relative; 1e-10 (the entry's tolerance) leaves six orders of margin
# and still catches any wrong weight, window or normalisation (>= 1e-3).
RTOL = 1e-10


# --------------------------------------------------------------- helpers ---

def expect_raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc:
        return
    raise AssertionError("%s did not raise %s with %r %r"
                         % (getattr(fn, "__name__", fn), exc.__name__, a, k))


def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-300)


class _Timeout(Exception):
    pass


def _alarm(seconds):
    def _h(signum, frame):
        raise _Timeout("timed out after %d s (a hang)" % seconds)
    signal.signal(signal.SIGALRM, _h)
    signal.alarm(int(seconds))


def _children():
    return {p.pid: p for p in multiprocessing.active_children()}


_CONTRACT_MOD = None


def contract_module():
    """passive_long_step_training.py as of the contract commit."""
    global _CONTRACT_MOD
    if _CONTRACT_MOD is None:
        src = subprocess.check_output(
            ["git", "show", "%s:./passive_long_step_training.py" % CONTRACT],
            cwd=str(HERE))
        d = Path(tempfile.mkdtemp(prefix="smoke_resid_contract_"))
        p = d / "plst_contract_b23.py"
        p.write_bytes(src)
        spec = importlib.util.spec_from_file_location("plst_contract_b23", str(p))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _CONTRACT_MOD = mod
    return _CONTRACT_MOD


# The protocols: (name, keyword arguments shared by the loss, the layout and
# the residuals). The SS exponential weight starts at the window start
# (0.5 ms), tau_w 5 ms, as integrate_long_step builds it by default.
SSW = plst.exp_time_weight(5e-4, 5e-3)
PROTOCOLS = (
    ("ls_amplitude/peak_decay/step/ss_exp",
     dict(weighting="ls_amplitude", ls_sample_weight="peak_decay",
          ls_window_mode="step", ss_sample_weight_fn=SSW)),
    ("relative/none/step/ss_exp",
     dict(weighting="relative", ls_sample_weight="none",
          ls_window_mode="step", ss_sample_weight_fn=SSW)),
    ("relative/none/after_onset/no_ss/steady",
     dict(weighting="relative", ls_sample_weight="none",
          ls_window_mode="after_onset", ss_sample_weight_fn=None,
          r_in_target="steady")),
)
P_LS = PROTOCOLS[0][1]


def rmsd_kw(kw):
    """The bundle_rmsd arguments of a protocol."""
    return {k: v for k, v in kw.items() if k != "weighting"}


def loss_of(tb, W, kw, mod=plst, spec=SPEC6):
    _mono, cell = neuron_cell()
    a = dict(kw, dt_brief_ms=DT, dt_long_ms=DT, spec=spec)
    if mod is plst:
        a["n_workers"] = W
    return mod.build_multi_protocol_loss(cell, tb, V_REST, **a)


def resid_of(tb, W, kw, spec=SPEC6, fail_l2=100.0):
    _mono, cell = neuron_cell()
    return plst.build_multi_protocol_residuals(
        cell, tb, V_REST, dt_brief_ms=DT, dt_long_ms=DT, spec=spec,
        n_workers=W, fail_l2=fail_l2, **kw)


def ev_of(tb, W, spec=SPEC6):
    _mono, cell = neuron_cell()
    return plst.build_trace_evaluator(cell, tb, V_REST, spec=spec,
                                      dt_brief_ms=DT, dt_long_ms=DT,
                                      n_workers=W)


def soma_state(cell):
    seg = cell.soma[0](0.5)
    return (seg.cm, seg.g_pas, seg.Ih_human.gIhbar, seg.e_pas)


def Q6():
    return np.array(box_points(SPEC6)[:4] + box_points(SPEC6)[-2:], dtype=float)


# ------------------------------------------------------------ the checks ---

def test_T1_residuals_reproduce_rmsd_and_loss():
    _mono, cell = neuron_cell()
    pts = box_points(SPEC6)                      # 6 inner + 2 on faces
    assert len(pts) >= 6
    notes = []
    for name, kw in PROTOCOLS:
        tb = _train_set()
        loss = loss_of(tb, 1, kw)
        R = loss.make_residuals()
        lay = R.layout
        assert abs(float(np.sum(lay.c)) - 1.0) < 1e-14, lay.c   # a few ulp
        assert lay.c.dtype == np.float64 and lay.c.shape == (len(tb),)
        assert lay.D.dtype == np.float64 and lay.D.shape == (len(tb),)
        rows = R(np.array(pts))
        assert rows.shape == (len(pts), lay.n_residuals) and rows.dtype == np.float64
        assert R.n_failed == 0
        worst = 0.0
        for k, q in enumerate(pts):
            SPEC6.apply_q(cell, q, V_REST)
            for b, bun in enumerate(tb):
                rmsd, defl = plst.bundle_rmsd(cell, bun, V_REST, dt_brief_ms=DT,
                                              dt_long_ms=DT, **rmsd_kw(kw))
                assert lay.D[b] == defl, (name, b, lay.D[b], defl)
                rb = rows[k, lay.slices[b]]
                lhs = float(np.sum(rb * rb))
                rhs = float(lay.c[b] * (rmsd / defl) ** 2)
                worst = max(worst, rel(lhs, rhs))
                assert rel(lhs, rhs) < RTOL, (name, k, b, lhs, rhs)
            L, L2 = R.l_and_l2(rows[k])
            lv = loss(*q)
            assert lv < 1e6
            worst = max(worst, rel(L, lv))
            assert rel(L, lv) < RTOL, (name, k, L, lv)
            assert rel(L2, float(np.sum(rows[k] ** 2))) == 0.0
            # L <= sqrt(L_2) (Cauchy-Schwarz with sum c_b = 1); roundoff slack
            assert L <= np.sqrt(L2) * (1 + 1e-12), (name, k, L, L2)
        notes.append("%s: N=%d, worst rel %.1e" % (name, lay.n_residuals, worst))
    print("    T1 " + "; ".join(notes))


def _ev_equal(a, b):
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if (x is None) != (y is None):
            return False
        if x is not None:
            if len(x) != len(y) or not all(np.array_equal(u, v) and u.dtype == v.dtype
                                           for u, v in zip(x, y)):
                return False
    return True


def test_T2_pooled_equals_serial():
    tb = _train_set()
    Q = Q6()
    for name, kw in PROTOCOLS[:2]:
        rows = {}
        for W in (1, 2, 3):
            R = resid_of(tb, W, kw)
            try:
                rows[W] = R(Q)
            finally:
                R.close()
        assert np.array_equal(rows[2], rows[1]), name
        assert np.array_equal(rows[3], rows[1]), name
    evs = {}
    for W in (1, 2, 3):
        ev = ev_of(tb, W)
        try:
            evs[W] = ev(Q)
        finally:
            ev.close()
    assert all(x is not None for x in evs[1])
    assert all(len(x) == len(tb) for x in evs[1])
    for x in evs[1]:
        for b, tr in enumerate(x):
            assert tr.dtype == np.float64 and tr.shape == (len(tb[b].t),)
    assert _ev_equal(evs[2], evs[1]) and _ev_equal(evs[3], evs[1])


def test_T3_parent_cell_at_last_point():
    _mono, cell = neuron_cell()
    tb = _train_set()
    Q = Q6()
    SPEC6.apply_q(cell, tuple(Q[-1]), V_REST)
    want = soma_state(cell)
    for W in (1, 2):
        SPEC6.apply_q(cell, tuple(Q[0]), V_REST)     # move the cell elsewhere
        assert soma_state(cell) != want
        R = resid_of(tb, W, P_LS)
        try:
            R(Q)
            assert soma_state(cell) == want, (W, soma_state(cell), want)
            ev = R.evaluator
            SPEC6.apply_q(cell, tuple(Q[0]), V_REST)
            ev(Q[::-1])                               # last row is Q[0] now
            SPEC6.apply_q(cell, tuple(Q[0]), V_REST)
            want0 = soma_state(cell)
            SPEC6.apply_q(cell, tuple(Q[-1]), V_REST)
            ev(Q[::-1])
            assert soma_state(cell) == want0, W
        finally:
            R.close()


def _target_cm(q):
    _mono, cell = neuron_cell()
    SPEC6.apply_q(cell, tuple(q), V_REST)
    return cell.soma[0](0.5).cm


def _patch_long(target_cm, action):
    """_simulate_long that, for the point whose soma cm is target_cm, raises
    action (a class) or returns a NaN trace (action == 'nan')."""
    orig = plst._simulate_long

    def f(cell, b, *a, **k):
        if cell.soma[0](0.5).cm == target_cm:
            if action == "nan":
                t, v = orig(cell, b, *a, **k)
                v = np.array(v, dtype=float)
                v[len(v) // 2] = np.nan
                return t, v
            raise action("planted failure")
        return orig(cell, b, *a, **k)
    return orig, f


def test_T4_failed_point_and_dt_error():
    import passive_fitting_hpc_fixed as mono
    tb = _train_set()
    Q = Q6()
    bad = 2
    tcm = _target_cm(Q[bad])
    assert sum(_target_cm(q) == tcm for q in Q) == 1     # one point only
    ref = {}
    for W in (1, 2):
        R = resid_of(tb, W, P_LS)
        try:
            ref[W] = R(Q)
        finally:
            R.close()
    for action in (ValueError, "nan"):
        for W in (1, 2):
            for fail_l2 in (100.0, 7.0):
                orig, f = _patch_long(tcm, action)
                plst._simulate_long = f                  # before the fork
                try:
                    R = resid_of(tb, W, P_LS, fail_l2=fail_l2)
                    try:
                        rows = R(Q)
                    finally:
                        R.close()
                finally:
                    plst._simulate_long = orig
                N = R.n_residuals
                const = np.full(N, np.sqrt(fail_l2 / N))
                assert np.array_equal(rows[bad], const), (action, W)
                _L, L2 = R.l_and_l2(rows[bad])
                # N equal squares summed: roundoff only (N ~ 1.7e4 terms)
                assert rel(L2, fail_l2) < 1e-12, (L2, fail_l2)
                assert R.n_failed == 1 and R.evaluator.n_failed == 1, R.n_failed
                keep = [k for k in range(len(Q)) if k != bad]
                assert np.array_equal(rows[keep], ref[1][keep]), (action, W)

    class DtEnforcementError(RuntimeError):            # local: cannot be pickled
        pass

    for cls in (mono.DtEnforcementError, DtEnforcementError):
        for W in (1, 2):
            orig, f = _patch_long(tcm, cls)
            plst._simulate_long = f
            try:
                R = resid_of(tb, W, P_LS)
                try:
                    R(Q)
                except Exception as exc:  # noqa: BLE001 - the outcome under test
                    assert type(exc).__name__ == "DtEnforcementError", type(exc)
                    if cls is mono.DtEnforcementError:
                        assert isinstance(exc, cls), type(exc)
                else:
                    raise AssertionError("DtEnforcementError swallowed (W=%d)" % W)
                finally:
                    R.close()
            finally:
                plst._simulate_long = orig
    # the evaluator itself (not only resid) raises it, serial and pooled
    for W in (1, 2):
        orig, f = _patch_long(tcm, mono.DtEnforcementError)
        plst._simulate_long = f
        try:
            ev = ev_of(tb, W)
            try:
                expect_raises(mono.DtEnforcementError, ev, Q)
            finally:
                ev.close()
        finally:
            plst._simulate_long = orig


def test_T5_refusals():
    _mono, cell = neuron_cell()
    tb = _train_set()
    lay = plst.residual_layout
    for w in ("equal", "custom", "bogus"):
        expect_raises(ValueError, lay, tb, weighting=w)
    expect_raises(ValueError, lay, [])
    expect_raises(ValueError, lay, tb, ls_sample_weight="bogus")
    expect_raises(ValueError, lay, tb, ls_window_mode="bogus")
    expect_raises(ValueError, lay, tb, ls_sample_weight="peak_decay", ls_tau_d_ms=0.0)
    # no pre-window sample: a brief bundle recorded from 1 ms after onset
    s = ss_bundle(-100.0)
    m = s.t > 1e-3
    s.t, s.v_mV = s.t[m], s.v_mV[m]
    expect_raises(ValueError, lay, [s])
    # fewer than 2 window samples
    expect_raises(ValueError, lay, [ss_bundle(-100.0)], ss_window_ms=(200.0, 300.0))
    # a non-finite recorded sample in the window, then in the pre-window
    for where in (0.02, -0.005):
        s = ss_bundle(-100.0)
        s.v_mV = s.v_mV.copy()
        s.v_mV[np.flatnonzero(s.t >= where)[0]] = np.nan
        expect_raises(ValueError, lay, [s])
    lb = ls_bundle(-70.0, sag=0.3)
    lb.v_mV = lb.v_mV.copy()
    lb.v_mV[np.flatnonzero(lb.t > 0.15)[0]] = np.inf
    expect_raises(ValueError, lay, [lb])
    # sample weights non-finite, of zero sum
    expect_raises(ValueError, lay, [ss_bundle(-100.0)],
                  ss_sample_weight_fn=lambda t: np.full_like(t, np.nan))
    expect_raises(ValueError, lay, [ss_bundle(-100.0)],
                  ss_sample_weight_fn=lambda t: np.zeros_like(t))
    # D_b not positive: a flat recorded trace
    s = ss_bundle(-100.0)
    s.v_mV = np.full_like(s.t, V_REST)
    expect_raises(ValueError, lay, [s])
    # ls_amplitude with no positive long-square amplitude
    try:
        lay([ss_bundle(-100.0), ls_bundle(0.0, sag=0.1)], weighting="ls_amplitude")
    except ValueError as exc:
        assert "ls_amplitude" in str(exc), str(exc)
    else:
        raise AssertionError("ls_amplitude with a zero amplitude accepted")
    # the same refusals through build_multi_protocol_residuals
    expect_raises(ValueError, plst.build_multi_protocol_residuals, cell, tb, V_REST,
                  weighting="equal")
    expect_raises(ValueError, plst.build_multi_protocol_residuals, cell, [], V_REST)
    for bad in (0, -1, 1.5, True):
        expect_raises(ValueError, plst.build_trace_evaluator, cell, tb, V_REST,
                      n_workers=bad)
        expect_raises(ValueError, plst.build_multi_protocol_residuals, cell, tb,
                      V_REST, n_workers=bad)
    for bad in (0.0, -1.0, np.nan, np.inf):
        expect_raises(ValueError, plst.build_multi_protocol_residuals, cell, tb,
                      V_REST, fail_l2=bad)
    # Q: not 2-D, wrong number of columns, non-finite
    ev = ev_of(tb, 1)
    R = resid_of(tb, 1, P_LS)
    q = np.array(box_points(SPEC6)[0])
    for badQ in (q, q[None, :5], np.vstack([q, q])[None], np.array([[np.nan] * 6])):
        expect_raises(ValueError, ev, badQ)
        expect_raises(ValueError, R, badQ)
    assert ev.n_points == 0 and R.n_points == 0
    # make_residuals refused for the weightings without a residual vector
    for w, extra in (("equal", {}), ("custom", {"weights": [1.0] * len(tb)})):
        L = plst.build_multi_protocol_loss(cell, tb, V_REST, weighting=w,
                                           spec=SPEC6, dt_brief_ms=DT,
                                           dt_long_ms=DT, **extra)
        expect_raises(ValueError, L.make_residuals)


def test_T6_loss_unchanged_and_other_smokes():
    old = contract_module()
    pts = box_points(SPEC6)[:4]
    for name, kw in PROTOCOLS:
        tb = _train_set()
        l0 = loss_of(tb, 1, kw, mod=old)
        v0 = [l0(*q) for q in pts]
        for W in (1, 2):
            lw = loss_of(tb, W, kw)
            try:
                vw = [lw(*q) for q in pts]
            finally:
                lw.close()
            assert vw == v0, (name, W, vw, v0)
        assert max(v0) < 1e6
    env = dict(os.environ)
    env.pop("DISPLAY", None)
    for smoke, need in (("smoke_ls_protocol.py", "9 pass"),
                        ("smoke_ls_pool.py", "13 pass")):
        r = subprocess.run([sys.executable, smoke], cwd=str(HERE), env=env,
                           capture_output=True, text=True, timeout=3600)
        tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-500:]
        print("    T6 %s: %s (exit %d)" % (smoke, tail, r.returncode))
        assert r.returncode == 0 and need in r.stdout, r.stdout[-3000:] + r.stderr[-2000:]


def test_T7_counters_and_close():
    tb = _train_set()
    Q = Q6()
    R1 = resid_of(tb, 1, P_LS)
    ref = R1(Q)
    before = set(_children())
    R = resid_of(tb, 2, P_LS)
    assert R.n_points == 0 and R.n_failed == 0
    assert set(_children()) == before, "the pool must not exist before the first call"
    R(Q)
    R(Q[:2])
    R(Q[:0])                                    # zero rows: a valid (0, N)
    assert R(Q[:0]).shape == (0, R.n_residuals)
    assert R.n_points == len(Q) + 2 and R.evaluator.n_points == len(Q) + 2
    kids = {pid for pid in _children() if pid not in before}
    assert 1 <= len(kids) <= 2, kids
    R.close()
    R.close()                                   # twice: harmless
    time.sleep(0.05)
    assert not [pid for pid in _children() if pid not in before]
    again = R(Q)                                # a new pool
    assert np.array_equal(again, ref)
    assert R.n_points == 2 * len(Q) + 2
    R.close()
    assert not [pid for pid in _children() if pid not in before]
    ev = ev_of(tb, 1)
    assert ev.n_workers == 1 and ev.close() is None and ev.close() is None
    assert R.evaluator.n_workers == 2 and isinstance(R.evaluator.n_workers, int)


def test_T8_loss_and_its_residuals_alternately():
    tb = _train_set()
    pts = box_points(SPEC6)
    before = set(_children())
    L1 = loss_of(tb, 1, P_LS)
    R1 = L1.make_residuals()
    L2 = loss_of(tb, 2, P_LS)
    R2 = L2.make_residuals()
    assert R2.evaluator.n_workers == 2 and R1.evaluator.n_workers == 1
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            for r in range(3):
                q = pts[r]
                v = L2(*q)
                assert len([p for p in _children() if p not in before]) <= 2
                assert v == L1(*q), (r, v)
                Q = np.array([pts[r], pts[r + 3]])
                rows = R2(Q)
                assert len([p for p in _children() if p not in before]) <= 2
                assert np.array_equal(rows, R1(Q)), r
    finally:
        L2.close()
        R2.close()
    assert not [pid for pid in _children() if pid not in before]


def test_T9_bundle_trace():
    _mono, cell = neuron_cell()
    q = box_points(SPEC6)[1]
    SPEC6.apply_q(cell, q, V_REST)
    for lsw in ("peak_decay", "none"):
        b = ls_bundle(-110.0, sag=0.5)
        lay = plst.residual_layout([b], ls_window_mode="step", ls_sample_weight=lsw)
        tr = plst.bundle_trace(cell, b, V_REST, dt_brief_ms=DT, dt_long_ms=DT)
        assert tr.dtype == np.float64 and tr.shape == (len(b.t),)
        d = tr[lay.masks[0]] - lay.v_exp[0]
        t_w = b.t[lay.masks[0]]
        wfn = plst._ls_weight_fn(b, lsw, 500.0)
        w = np.ones_like(t_w) if wfn is None else wfn(t_w)
        mine = float(np.sqrt(np.sum(w * d * d) / np.sum(w)))
        rmsd, _defl = plst.bundle_rmsd(cell, b, V_REST, dt_brief_ms=DT, dt_long_ms=DT,
                                       ls_window_mode="step", ls_sample_weight=lsw)
        assert rel(mine, rmsd) < RTOL, (lsw, mine, rmsd)
        # independent: subtract the simulated pre-window mean, interpolate
        t_s, v = plst._simulate_long(cell, b, V_REST, dt_ms=DT)
        pm = (t_s >= 0.0) & (t_s <= b.stim_onset_s)
        ind = np.interp(b.t, t_s, v - np.mean(v[pm]))
        assert np.array_equal(tr, ind)
    # a brief bundle recorded past both ends of the simulated interval
    s = ss_bundle(-200.0)
    fs = 1e4
    s.t = np.arange(-0.03, 0.2, 1.0 / fs)
    s.v_mV = np.full_like(s.t, V_REST)
    tr = plst.bundle_trace(cell, s, V_REST, dt_brief_ms=DT, dt_long_ms=DT)
    t_s, v = plst._simulate_brief(cell, s, V_REST, dt_ms=DT)
    pm = (t_s >= -10e-3) & (t_s <= 0.0)
    bs = np.mean(v[pm])
    lo, hi = s.t < t_s[0], s.t > t_s[-1]
    assert lo.sum() > 10 and hi.sum() > 10, (t_s[0], t_s[-1])
    assert np.all(tr[lo] == v[0] - bs) and np.all(tr[hi] == v[-1] - bs)
    assert np.all(np.isfinite(tr))


def test_T10_ascii_lf():
    for f in ("passive_long_step_training.py", "smoke_residuals.py"):
        b = (HERE / f).read_bytes()
        assert all(c < 128 for c in b), f + " non-ASCII"
        assert b"\r" not in b, f + " CR bytes"


# ---------------------------------------------------------------- runner ---

def _environment():
    parts = ["python %s" % platform.python_version()]
    for package in REPORT_PACKAGES:
        try:
            parts.append("%s %s" % (package, importlib.metadata.version(package)))
        except importlib.metadata.PackageNotFoundError:
            parts.append("%s (not installed)" % package)
    parts.append(platform.platform())
    parts.append("%d cores" % (os.cpu_count() or 0))
    parts.append("seed %d, contract %s" % (SEED, CONTRACT))
    return " | ".join(parts)


def main():
    checks = [(name, obj) for name, obj in globals().items()
              if name.startswith("test_") and callable(obj)]
    only = sys.argv[1:]
    if only:
        checks = [(n, f) for n, f in checks if any(n.startswith("test_" + o) for o in only)]
    print("== %s" % Path(__file__).name)
    print(_environment())
    results = []
    for name, func in checks:
        start = time.perf_counter()
        try:
            _alarm(3600)
            func()
            status, detail = "PASS", ""
        except unittest.SkipTest as exc:
            status, detail = "SKIP", str(exc)
        except NotImplementedError as exc:
            status, detail = "TODO", str(exc)
        except AssertionError as exc:
            status, detail = "FAIL", str(exc) or traceback.format_exc()
        except Exception:  # unexpected errors are reported, never hidden
            status, detail = "ERROR", traceback.format_exc()
        finally:
            signal.alarm(0)
        results.append((name, status, time.perf_counter() - start, detail))
    if "cell" in _CELL:
        _CELL["cell"].destroy()
    width = max([len(n) for n, _, _, _ in results] + [4])
    for name, status, seconds, detail in results:
        lines = detail.strip().splitlines()
        head = lines[0] if lines else ""
        print(("%-5s  %-*s  %8.3fs  %s" % (status, width, name, seconds, head)).rstrip())
    for name, status, _s, detail in results:
        if status in ("FAIL", "ERROR"):
            print("\n---- %s: %s\n%s" % (status, name, detail.strip()))
    counts = {s: sum(1 for r in results if r[1] == s)
              for s in ("PASS", "FAIL", "ERROR", "TODO", "SKIP")}
    print("\n-- " + ", ".join("%d %s" % (n, s.lower()) for s, n in counts.items()))
    return 1 if (counts["FAIL"] + counts["ERROR"] + counts["TODO"]) else 0


if __name__ == "__main__":
    sys.exit(main())
