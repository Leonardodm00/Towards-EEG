"""Smoke test for the loss of SPEC 2.8 -- Block 14 in specs/SPEC.md.

The role rule `d207` (2.8 (b)), the step weight `ls_amplitude` (2.8 (c)) and
the long square's sample weight `peak_decay` (2.8 (d)) in
passive_long_step_training.py, and the regression oracles that keep every
default where it was at the contract commit.

Checks (the oracles of the Block 14 entry)
    LP1  roles for n = 0..7 under d207 against 2.8 (b) written out by hand,
         with the depolarising steps and the records; d006 unchanged against
         a table computed from the contract commit's code
    LP2  ls_peak_time_s: a known trough (exact to one sample), noise spikes
         shorter than the boxcar that must not move it, a monotone trace (the
         last sample), a depolarising step (the maximum), the tie rule
    LP3  peak_decay_weight: 0 at onset, 1 at the peak, exp(-1) at
         t_pk + tau_d, linear before, continuous, 0 before onset
    LP4  the weighted RMSD equals the formula recomputed with numpy (1e-12
         relative); "none" equals the unweighted RMSD bit for bit
    LP5  ls_amplitude equals the formula of 2.8 (c) from the per-bundle
         RMSDs and deflections; equal amplitudes reproduce relative (1e-12);
         only SS or only LS bundles
    LP6  defaults: the loss of a fixture cell (a stub cell, and the
         monolith's NEURON cell) equals the loss of the contract commit's code
         for three parameter vectors, bit for bit, directly and through
         integrate_long_step; roles and splits equal too
    LP7  integrate_long_step with the new options: the patched training loss
         uses them; validation RMSDs through the patched monolith are
         unchanged by ls_sample_weight
    LP8  refusals
    LP9  ASCII / LF of the block's files

Run (from this directory; NEURON loads ./x86_64 from the CWD)
    env -u DISPLAY python3 smoke_ls_protocol.py
        Prints the environment and a PASS / FAIL / ERROR / TODO / SKIP table,
        then the full message of every failure. Exit code 1 if any check
        failed, errored or is still TODO.
    LP6 needs git (it reads the contract commit's module with `git show`).

Debug
    A failed check_close() saves both arrays to smoke_artifacts/<name>.npz.

Keep this file pure ASCII.
"""
from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
import traceback
import types
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.chdir(HERE)                      # NEURON loads ./x86_64 from the CWD
sys.path.insert(0, str(HERE))

import numpy as np                  # noqa: E402
from numpy.testing import assert_allclose  # noqa: E402

import passive_long_step_training as plst  # noqa: E402

CONTRACT = "aac46d3"
SEED = 20261122
REPORT_PACKAGES = ("numpy", "scipy", "NEURON")
ARTIFACT_DIR = HERE / "smoke_artifacts"
_SAVED_ARTIFACTS = []


# --------------------------------------------------------------- helpers ---

def check_close(name, actual, expected, rtol=1e-12, atol=0.0):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    try:
        assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=name)
    except AssertionError:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
        path = ARTIFACT_DIR / ("%s.npz" % safe)
        np.savez(path, actual=actual, expected=expected)
        _SAVED_ARTIFACTS.append(path)
        raise


def expect_raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc:
        return
    raise AssertionError("%s did not raise %s" % (getattr(fn, "__name__", fn),
                                                  exc.__name__))


_CONTRACT_MOD = None


def contract_module():
    """passive_long_step_training.py as of the contract commit, imported
    under another name from a temporary file."""
    global _CONTRACT_MOD
    if _CONTRACT_MOD is None:
        src = subprocess.check_output(
            ["git", "show", "%s:./passive_long_step_training.py" % CONTRACT],
            cwd=str(HERE))
        d = Path(tempfile.mkdtemp(prefix="smoke_lp_contract_"))
        p = d / "plst_contract.py"
        p.write_bytes(src)
        spec = importlib.util.spec_from_file_location("plst_contract", str(p))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _CONTRACT_MOD = mod
    return _CONTRACT_MOD


# --------------------------------------------------------------- fixtures --
V_REST = -70.0


def sag_trace(t, amp_pA, on, dur, *, r_MOhm=100.0, tau_m=0.02, sag=0.3,
              tau_h=0.15):
    """A recorded long square with a sag: charging, a trough, relaxation,
    and the return after the offset (mV)."""
    v = np.full_like(t, V_REST)
    s = t - on
    m = (s >= 0) & (s <= dur)
    dv = amp_pA * r_MOhm * 1e-3
    shape = (1 - np.exp(-s[m] / tau_m)) * (1 - sag * (1 - np.exp(-s[m] / tau_h)))
    v[m] = V_REST + dv * shape
    end = V_REST + dv * (1 - np.exp(-dur / tau_m)) * (1 - sag * (1 - np.exp(-dur / tau_h)))
    a = s > dur
    v[a] = V_REST + (end - V_REST) * np.exp(-(s[a] - dur) / tau_m)
    return v


def ls_bundle(amp_pA, *, fs=1e4, T=0.9, on=0.1, dur=0.5, v=None, **kw):
    t = np.arange(int(round(T * fs))) / fs
    if v is None:
        v = sag_trace(t, amp_pA, on, dur, **kw)
    return types.SimpleNamespace(
        t=t, v_mV=np.asarray(v, dtype=float), amplitude_pA=float(amp_pA),
        polarity="hyp" if amp_pA < 0 else "dep", stim_onset_s=float(on),
        stim_duration_s=float(dur), i_pA=np.zeros_like(t),
        n_repeats_averaged=1, sweep_numbers=[0], sampling_rate_Hz=fs,
        stimulus_name="Long Square")


def ss_bundle(amp_pA, *, fs=1e4):
    t = np.arange(-0.01, 0.1, 1.0 / fs)
    v = np.full_like(t, V_REST)
    m = t >= 0
    v[m] = V_REST + amp_pA * 1e-3 * 0.5 * np.exp(-t[m] / 0.02)
    return types.SimpleNamespace(
        t=t, v_mV=v, amplitude_pA=float(amp_pA),
        polarity="hyp" if amp_pA < 0 else "dep", stim_onset_s=0.0,
        stim_duration_s=0.0005, i_pA=np.zeros_like(t), n_repeats_averaged=1,
        sweep_numbers=[0], sampling_rate_Hz=fs, stimulus_name="Short Square")


class StubCell:
    """The duck-typed cell API, analytic RC response (no NEURON)."""

    def __init__(self):
        self.Cm, self.Rm, self.Ra, self.e = 1.0, 30000.0, 100.0, V_REST

    def set_passive(self, Cm, Rm, Ra):
        self.Cm, self.Rm, self.Ra = float(Cm), float(Rm), float(Ra)

    def set_e_pas(self, e):
        self.e = float(e)

    def simulate(self, stim_amp_pA, stim_delay_ms, stim_dur_ms, tstop_ms,
                 v_init_mV=-70.0, dt_ms=None, enforce_dt=True):
        dt = 0.1 if dt_ms is None else float(dt_ms)
        t = np.arange(0.0, tstop_ms + 0.5 * dt, dt)
        tau = self.Rm * self.Cm * 1e-3                    # ms
        r = self.Rm / 300.0 + 0.01 * self.Ra              # MOhm
        dv = stim_amp_pA * r * 1e-3
        v = np.full_like(t, v_init_mV)
        s = t - stim_delay_ms
        on = (s >= 0) & (s <= stim_dur_ms)
        v[on] = v_init_mV + dv * (1 - np.exp(-s[on] / tau))
        endv = dv * (1 - np.exp(-stim_dur_ms / tau))
        a = s > stim_dur_ms
        v[a] = v_init_mV + endv * np.exp(-(s[a] - stim_dur_ms) / tau)
        return t, v


def roles_cell(n, n_dep=4):
    hyp = [ls_bundle(-10.0 * (i + 1)) for i in range(n)][::-1]   # unsorted
    dep = [ls_bundle(20.0 * (i + 1), sag=0.0) for i in range(n_dep)]
    return types.SimpleNamespace(
        square_subthreshold=[ss_bundle(-200.0), ss_bundle(200.0)],
        long_square_subthreshold=hyp, long_square_depolarising=dep,
        rin_MOhm=100.0, tau_ms=20.0, v_rest_mV=V_REST, sag_ratio=0.3,
        ls_dep_screen={"n_depolarising": n_dep})


Q3 = [np.log([1.0, 30000.0, 150.0]), np.log([0.6, 12000.0, 400.0]),
      np.log([2.2, 80000.0, 60.0])]


# ------------------------------------------------------------ the checks ---

def _d207_by_hand(n):
    """2.8 (b) written out: n -> {i (1-based): (role, reason)}."""
    table = {
        0: {},
        1: {1: ("train", "h_1 of 1 (d207)")},
        2: {1: ("validate", "weakest step (held out)"),
            2: ("train", "h_2 of 2 (d207)")},
        3: {1: ("validate", "weakest step (held out)"),
            2: ("validate", "middle step h_m (held out, d207)"),
            3: ("train", "h_3 of 3 (d207)")},
        4: {1: ("validate", "weakest step (held out)"),
            2: ("validate", "middle step h_m (held out, d207)"),
            3: ("train", "h_3 of 4 (d207)"), 4: ("train", "h_4 of 4 (d207)")},
        5: {1: ("validate", "weakest step (held out)"),
            2: ("train", "h_2 of 5 (d207)"),
            3: ("validate", "middle step h_m (held out, d207)"),
            4: ("train", "h_4 of 5 (d207)"), 5: ("train", "h_5 of 5 (d207)")},
        6: {1: ("validate", "weakest step (held out)"),
            2: ("train", "h_2 of 6 (d207)"),
            3: ("validate", "middle step h_m (held out, d207)"),
            4: ("train", "h_4 of 6 (d207)"), 5: ("train", "h_5 of 6 (d207)"),
            6: ("train", "h_6 of 6 (d207)")},
        7: {1: ("validate", "weakest step (held out)"),
            2: ("train", "h_2 of 7 (d207)"), 3: ("train", "h_3 of 7 (d207)"),
            4: ("validate", "middle step h_m (held out, d207)"),
            5: ("train", "h_5 of 7 (d207)"), 6: ("train", "h_6 of 7 (d207)"),
            7: ("train", "h_7 of 7 (d207)")},
    }
    return table[n]


def _d006_contract_table(n):
    """Computed from the contract commit's code (aac46d3), default arguments,
    for n = 0..7: h_1 of 1 / h_1, h_2 of 2 trained; otherwise h_1 validates,
    h_n is report-only, h_2..h_{n-1} train."""
    if n == 0:
        return {}
    if n <= 2:
        return {i: ("train", "h_%d of %d" % (i, n)) for i in range(1, n + 1)}
    out = {1: ("validate", "weakest step (held out)"),
           n: ("report", "strongest step (PIR conductances unmodelled)")}
    for i in range(2, n):
        out[i] = ("train", "h_%d of %d" % (i, n))
    return out


_DEP_RECORDS = [(20.0, "validate", "spike-free depolarising step"),
                (40.0, "validate", "spike-free depolarising step"),
                (60.0, "validate", "spike-free depolarising step"),
                (80.0, "unused", "beyond dep_n_validation")]


def _check_roles(n, rule, expected):
    cd = roles_cell(n)
    kw = {} if rule is None else {"role_rule": rule}
    r = plst.assign_ls_roles(cd, verbose=False, **kw)
    recs = r["records"]
    hyp = [(x["amplitude_pA"], x["role"], x["reason"]) for x in recs
           if x["polarity"] == "hyp"]
    want = [(-10.0 * i, expected[i][0], expected[i][1])
            for i in range(1, n + 1)]
    assert hyp == want, "%s n=%d: %s != %s" % (rule, n, hyp, want)
    dep = [(x["amplitude_pA"], x["role"], x["reason"]) for x in recs
           if x["polarity"] == "dep"]
    assert dep == _DEP_RECORDS, "%s n=%d dep records %s" % (rule, n, dep)
    for role in ("train", "report"):
        got = [b.amplitude_pA for b in r[role]]
        exp = [-10.0 * i for i in range(1, n + 1) if expected[i][0] == role]
        assert got == exp, "%s n=%d %s %s != %s" % (rule, n, role, got, exp)
    got_v = [b.amplitude_pA for b in r["validate"]]
    exp_v = ([-10.0 * i for i in range(1, n + 1) if expected[i][0] == "validate"]
             + [20.0, 40.0, 60.0])
    assert got_v == exp_v, "%s n=%d validate %s != %s" % (rule, n, got_v, exp_v)
    assert r["n_hyp"] == n and r["n_dep"] == 4
    assert all(set(x) == {"amplitude_pA", "polarity", "role", "reason",
                          "v_trough_mV"} for x in recs)


def test_LP1_roles():
    for n in range(8):
        _check_roles(n, "d207", _d207_by_hand(n))
        _check_roles(n, None, _d006_contract_table(n))
        _check_roles(n, "d006", _d006_contract_table(n))
    # d207 ignores n_drop_weakest / n_drop_strongest
    a = plst.assign_ls_roles(roles_cell(6), role_rule="d207", n_drop_weakest=3,
                             n_drop_strongest=0, verbose=False)
    b = plst.assign_ls_roles(roles_cell(6), role_rule="d207", verbose=False)
    assert [x["role"] for x in a["records"]] == [x["role"] for x in b["records"]]
    # the floor keeps its meaning under d207: a training step below it -> report
    cd = roles_cell(6)
    r = plst.assign_ls_roles(cd, role_rule="d207", v_trough_min_mV=-74.0,
                             verbose=False)
    got = {x["amplitude_pA"]: x["role"] for x in r["records"]
           if x["polarity"] == "hyp"}
    # the fixture's troughs are -70 + 0.0861 amp mV (-73.45 at -40 pA,
    # -74.31 at -50, -75.17 at -60): h_5 and h_6 fall below -74 mV
    want = {-10.0: "validate", -20.0: "train", -30.0: "validate",
            -40.0: "train", -50.0: "report", -60.0: "report"}
    assert got == want, got
    # split_train_validation_ih passes role_rule on
    tr, va, rp, ro = plst.split_train_validation_ih(roles_cell(5),
                                                    role_rule="d207",
                                                    verbose=False)
    assert [b.amplitude_pA for b in tr] == [-200.0, -20.0, -40.0, -50.0]
    assert [b.amplitude_pA for b in va] == [200.0, -10.0, -30.0, 20.0, 40.0, 60.0]
    assert rp == []


def _gauss_dip(t, t0, depth, width=0.05, sign=-1.0):
    return V_REST + sign * depth * np.exp(-((t - t0) / width) ** 2)


def test_LP2_peak_time():
    fs = 50e3
    t = np.arange(int(0.8 * fs)) / fs
    on, dur = 0.1, 0.5
    # (a) a known trough at a grid sample; Gaussian dip, symmetric about it,
    # so the centred boxcar keeps its minimum there exactly
    k_pk = int(round(0.3123 * fs))
    v = _gauss_dip(t, t[k_pk], 10.0)
    b = ls_bundle(-50.0, fs=fs, T=0.8, on=on, dur=dur, v=v)
    tp = plst.ls_peak_time_s(b)
    assert abs(tp - t[k_pk]) <= 1.0 / fs, (tp, t[k_pk])
    # (b) noise spikes (3 samples = 0.06 ms << 1 ms) deeper than the trough
    v2 = v.copy()
    for kk in (int(0.2 * fs), int(0.4 * fs), int(0.55 * fs)):
        v2[kk:kk + 3] -= 13.0
    assert np.argmin(v2[(t >= on + 1e-3) & (t <= on + dur)]) != \
        k_pk - int(np.flatnonzero(t >= on + 1e-3)[0]), "spikes too weak to test"
    tp2 = plst.ls_peak_time_s(ls_bundle(-50.0, fs=fs, T=0.8, on=on, dur=dur, v=v2))
    assert abs(tp2 - t[k_pk]) <= 1.0 / fs, (tp2, t[k_pk])
    # (c) monotone (decreasing over the whole sweep): the interval's last sample
    v3 = V_REST - 20.0 * t
    tp3 = plst.ls_peak_time_s(ls_bundle(-50.0, fs=fs, T=0.8, on=on, dur=dur, v=v3))
    last = t[(t >= on + 1e-3) & (t <= on + dur)][-1]
    assert tp3 == last, (tp3, last)
    # (d) depolarising: the maximum
    v4 = _gauss_dip(t, t[k_pk], 10.0, sign=+1.0)
    tp4 = plst.ls_peak_time_s(ls_bundle(+50.0, fs=fs, T=0.8, on=on, dur=dur, v=v4))
    assert abs(tp4 - t[k_pk]) <= 1.0 / fs, (tp4, t[k_pk])
    # the same depolarising trace read as hyperpolarising gives an end, not
    # the bump: the polarity decides
    tp4h = plst.ls_peak_time_s(ls_bundle(-50.0, fs=fs, T=0.8, on=on, dur=dur, v=v4))
    assert abs(tp4h - t[k_pk]) > 0.05
    # (e) ties: a flat plateau of integer value (sums exact in float64): the
    # earliest sample whose 51-sample boxcar lies inside the plateau
    v5 = np.full_like(t, V_REST)
    ka, kb = int(0.25 * fs), int(0.45 * fs)
    v5[ka:kb] = V_REST - 10.0
    tp5 = plst.ls_peak_time_s(ls_bundle(-50.0, fs=fs, T=0.8, on=on, dur=dur, v=v5))
    half = int(round(0.5e-3 * fs))
    assert tp5 == t[ka + half], (tp5, t[ka + half])
    # (f) the 1 ms guard: a trough before t_on + 1 ms is not seen
    v6 = V_REST - 5.0 * np.exp(-((t - on - 0.2) / 0.05) ** 2)
    # deep samples in [t_on, t_on + 0.4 ms): outside the interval and out
    # of reach of the boxcar centred at t_on + 1 ms (it reaches 0.5 ms back)
    v6[(t >= on) & (t < on + 0.0004)] = V_REST - 50.0
    tp6 = plst.ls_peak_time_s(ls_bundle(-50.0, fs=fs, T=0.8, on=on, dur=dur, v=v6))
    assert abs(tp6 - (on + 0.2)) <= 1.0 / fs, tp6
    # brute-force reference of the smoothing rule on a coarse trace
    fs7 = 2e3
    t7 = np.arange(int(0.8 * fs7)) / fs7
    rng = np.random.default_rng(SEED)
    v7 = sag_trace(t7, -60.0, on, dur) + rng.normal(0, 0.4, t7.size)
    b7 = ls_bundle(-60.0, fs=fs7, T=0.8, on=on, dur=dur, v=v7)
    h7 = int(round(0.5 * 3.0e-3 * fs7))
    idx = np.flatnonzero((t7 >= on + 1e-3) & (t7 <= on + dur))
    sm = [np.mean(v7[max(i - h7, 0):i + h7 + 1]) for i in idx]
    ref = t7[idx[int(np.argmin(sm))]]
    got7 = plst.ls_peak_time_s(b7, smooth_ms=3.0)
    assert got7 == ref, (got7, ref)


def test_LP3_peak_decay_weight():
    on, pk, tau = 0.1, 0.25, 0.5
    w = plst.peak_decay_weight(on, pk, tau)
    assert w(np.array([on]))[0] == 0.0
    assert w(np.array([pk]))[0] == 1.0
    check_close("LP3 e^-1", w(np.array([pk + tau]))[0], np.exp(-1.0), rtol=1e-15)
    tl = np.linspace(on, pk, 101)
    check_close("LP3 linear", w(tl), (tl - on) / (pk - on), rtol=1e-14, atol=1e-15)
    eps = 1e-9
    assert abs(w(np.array([pk + eps]))[0] - 1.0) < 1e-8
    assert abs(w(np.array([pk - eps]))[0] - 1.0) < 1e-7
    tb = np.array([-1.0, 0.0, on - 1e-12])
    assert np.all(w(tb) == 0.0)
    td = np.linspace(pk, pk + 3, 50)
    check_close("LP3 decay", w(td), np.exp(-(td - pk) / tau), rtol=1e-15)
    out = w(np.linspace(0, 2, 1000))
    assert out.shape == (1000,) and out.dtype == np.float64
    assert np.all(out >= 0) and np.all(out <= 1)


def _ref_rmsd(cell, b, weighted, tau_d_s=0.5, mode="after_onset", win_ms=150.0):
    """The formula of 2.8 (d), recomputed with numpy from the stub's own
    simulation (independent of bundle_rmsd's code path)."""
    t_ms, vs = cell.simulate(stim_amp_pA=b.amplitude_pA,
                             stim_delay_ms=b.stim_onset_s * 1e3,
                             stim_dur_ms=b.stim_duration_s * 1e3,
                             tstop_ms=b.t[-1] * 1e3, v_init_mV=V_REST,
                             dt_ms=plst.DEFAULT_DT_LONG_MS)
    ts = t_ms * 1e-3
    on = b.stim_onset_s
    be = np.mean(b.v_mV[(b.t >= 0) & (b.t <= on)])
    bs = np.mean(vs[(ts >= 0) & (ts <= on)])
    end = {"after_onset": on + win_ms * 1e-3, "step": on + b.stim_duration_s,
           "sweep": b.t[-1]}[mode]
    m = (b.t >= on) & (b.t <= end)
    d = (b.v_mV - be)[m] - np.interp(b.t[m], ts, vs - bs)
    if not weighted:
        return np.sqrt(np.mean(d * d))
    tg = b.t[m]
    tpk = plst.ls_peak_time_s(b)
    w = np.zeros_like(tg)
    r = (tg >= on) & (tg <= tpk)
    w[r] = (tg[r] - on) / (tpk - on)
    a = tg > tpk
    w[a] = np.exp(-(tg[a] - tpk) / tau_d_s)
    return np.sqrt(np.sum(w * d * d) / np.sum(w))


def test_LP4_weighted_rmsd():
    cell = StubCell()
    cell.set_passive(0.8, 25000.0, 120.0)
    for amp in (-30.0, -90.0, +40.0):
        b = ls_bundle(amp, sag=0.3 if amp < 0 else 0.0)
        for mode in ("after_onset", "step", "sweep"):
            r_w, d_w = plst.bundle_rmsd(cell, b, V_REST, ls_window_mode=mode,
                                        ls_sample_weight="peak_decay")
            check_close("LP4 weighted %g %s" % (amp, mode), r_w,
                        _ref_rmsd(cell, b, True, mode=mode), rtol=1e-12)
            r_n, d_n = plst.bundle_rmsd(cell, b, V_REST, ls_window_mode=mode,
                                        ls_sample_weight="none")
            r_0, d_0 = plst.bundle_rmsd(cell, b, V_REST, ls_window_mode=mode)
            assert r_n == r_0 and d_n == d_0
            check_close("LP4 unweighted %g %s" % (amp, mode), r_n,
                        _ref_rmsd(cell, b, False, mode=mode), rtol=1e-12)
            assert d_w == d_n                        # D_b is unchanged
            assert r_w != r_n                        # the weight is live
    # tau_d reaches the weight
    b = ls_bundle(-50.0)
    r1, _ = plst.bundle_rmsd(cell, b, V_REST, ls_window_mode="step",
                             ls_sample_weight="peak_decay", ls_tau_d_ms=80.0)
    check_close("LP4 tau_d", r1, _ref_rmsd(cell, b, True, tau_d_s=0.08,
                                           mode="step"), rtol=1e-12)
    # brief bundles ignore ls_sample_weight
    s = ss_bundle(-200.0)
    assert (plst.bundle_rmsd(cell, s, V_REST, ls_sample_weight="peak_decay")
            == plst.bundle_rmsd(cell, s, V_REST))
    # bit-for-bit against the contract's bundle_rmsd at its defaults
    old = contract_module()
    for mode in ("after_onset", "step", "sweep"):
        assert (plst.bundle_rmsd(cell, b, V_REST, ls_window_mode=mode)
                == old.bundle_rmsd(cell, b, V_REST, ls_window_mode=mode))


def _train_set():
    # sag that grows with |I| (as I_h's does), so r_b / D_b differs per step
    return [ss_bundle(-200.0), ss_bundle(-150.0),
            ls_bundle(-30.0, sag=0.1), ls_bundle(-70.0, sag=0.3),
            ls_bundle(-110.0, sag=0.5)]


def test_LP5_ls_amplitude():
    cell = StubCell()
    kw = dict(ls_window_mode="step", ss_sample_weight_fn=plst.exp_time_weight(5e-4, 5e-3))
    q = Q3[1]
    for lsw in ("none", "peak_decay"):
        tb = _train_set()
        L = plst.build_multi_protocol_loss(cell, tb, V_REST, weighting="ls_amplitude",
                                           ls_sample_weight=lsw, **kw)(*q)
        cell.set_passive(*np.exp(q)); cell.set_e_pas(V_REST)
        rd = [plst.bundle_rmsd(cell, b, V_REST, ls_window_mode="step",
                               ss_sample_weight_fn=kw["ss_sample_weight_fn"],
                               ls_sample_weight=lsw) for b in tb]
        rel = [r / max(abs(d), 1e-6) for r, d in rd]
        ss_idx = [0, 1]
        ls_idx = [2, 3, 4]
        amps = np.array([abs(tb[i].amplitude_pA) for i in ls_idx])
        om = amps / amps.sum()
        want = (sum(rel[i] for i in ss_idx)
                + 3 * sum(o * rel[i] for o, i in zip(om, ls_idx))) / 5.0
        check_close("LP5 formula %s" % lsw, L, want, rtol=1e-12)
        Lrel = plst.build_multi_protocol_loss(cell, tb, V_REST, weighting="relative",
                                              ls_sample_weight=lsw, **kw)(*q)
        assert abs(L - Lrel) > 1e-6 * Lrel, "unequal amplitudes must differ"
    # equal amplitudes reproduce relative
    tb = [ss_bundle(-200.0), ls_bundle(-60.0, tau_h=0.1), ls_bundle(-60.0, sag=0.1),
          ls_bundle(-60.0)]
    for qq in Q3:
        a = plst.build_multi_protocol_loss(cell, tb, V_REST, weighting="ls_amplitude", **kw)(*qq)
        b = plst.build_multi_protocol_loss(cell, tb, V_REST, weighting="relative", **kw)(*qq)
        check_close("LP5 equal amplitudes", a, b, rtol=1e-12)
    # only SS: the relative loss; only LS: n_LS sum omega r/D / n_LS
    ss_only = [ss_bundle(-200.0), ss_bundle(-120.0)]
    a = plst.build_multi_protocol_loss(cell, ss_only, V_REST, weighting="ls_amplitude", **kw)(*q)
    b = plst.build_multi_protocol_loss(cell, ss_only, V_REST, weighting="relative", **kw)(*q)
    check_close("LP5 SS only", a, b, rtol=1e-12)
    ls_only = [ls_bundle(-20.0), ls_bundle(-90.0)]
    a = plst.build_multi_protocol_loss(cell, ls_only, V_REST, weighting="ls_amplitude", **kw)(*q)
    cell.set_passive(*np.exp(q)); cell.set_e_pas(V_REST)
    rel = [r / max(abs(d), 1e-6) for r, d in
           (plst.bundle_rmsd(cell, x, V_REST, ls_window_mode="step") for x in ls_only)]
    want = (2 * (20.0 / 110.0 * rel[0] + 90.0 / 110.0 * rel[1])) / 2.0
    check_close("LP5 LS only", a, want, rtol=1e-12)


def _neuron_cell():
    import passive_fitting_hpc_fixed as mono
    from synthetic_ground_truth import write_ball_and_stick_swc
    tmp = Path(tempfile.mkdtemp(prefix="smoke_lp_"))
    swc = write_ball_and_stick_swc(tmp / "bas.swc", soma_r_um=10.0,
                                   dend_len_um=300.0, dend_r_um=1.0,
                                   apic_len_um=500.0, apic_r_um=1.2, step_um=20.0)
    return mono, mono.build_neuron_model(swc, F=1.9)


def test_LP6_defaults_bit_for_bit():
    old = contract_module()
    # roles and splits under several argument sets
    for n in range(8):
        for kw in ({}, {"n_drop_weakest": 2, "n_drop_strongest": 0},
                   {"train_all_hyp": True}, {"v_trough_min_mV": -74.0},
                   {"dep_n_validation": None}):
            a = plst.assign_ls_roles(roles_cell(n), verbose=False, **kw)
            b = old.assign_ls_roles(roles_cell(n), verbose=False, **kw)
            # repr: NaN troughs (the depolarising rows) compare equal
            ra = [repr((x["amplitude_pA"], x["role"], x["reason"], x["v_trough_mV"]))
                  for x in a["records"]]
            rb = [repr((x["amplitude_pA"], x["role"], x["reason"], x["v_trough_mV"]))
                  for x in b["records"]]
            assert ra == rb, (n, kw)
    # the stub cell, all window modes and weightings, three q
    cell = StubCell()
    tb = _train_set()
    cell.set_passive(*np.exp(Q3[2])); cell.set_e_pas(V_REST)
    for b in tb:
        for mode in ("after_onset", "step", "sweep"):
            assert (plst.bundle_rmsd(cell, b, V_REST, ls_window_mode=mode)
                    == old.bundle_rmsd(cell, b, V_REST, ls_window_mode=mode))
    for wt in ("relative", "equal"):
        for mode in ("after_onset", "step"):
            la = plst.build_multi_protocol_loss(cell, tb, V_REST, weighting=wt,
                                                ls_window_mode=mode)
            lb = old.build_multi_protocol_loss(cell, tb, V_REST, weighting=wt,
                                               ls_window_mode=mode)
            for q in Q3:
                assert la(*q) == lb(*q), (wt, mode)
    # the monolith's NEURON cell, directly and through integrate_long_step
    mono, ncell = _neuron_cell()
    try:
        tb = [ss_bundle(-200.0), ls_bundle(-30.0), ls_bundle(-70.0)]
        la = plst.build_multi_protocol_loss(ncell, tb, V_REST, dt_brief_ms=0.1,
                                            dt_long_ms=0.1)
        lb = old.build_multi_protocol_loss(ncell, tb, V_REST, dt_brief_ms=0.1,
                                           dt_long_ms=0.1)
        va = [la(*q) for q in Q3]
        vb = [lb(*q) for q in Q3]
        assert va == vb, (va, vb)
        assert max(va) < 1e6 and len(set(va)) == 3, va      # live, not penalty
        cd = roles_cell(5)
        vals = {}
        for name, mod in (("new", plst), ("old", old)):
            mod.integrate_long_step(mono, ih_protocol=True, ls_window_mode="step",
                                    dt_brief_ms=0.1, dt_long_ms=0.1,
                                    verbose=False)
            oi = mono.prepare_optimiser_inputs(cd, fit_target="hyp")
            L = mono._build_loss_function(cell=ncell, train_bundles=oi.train_bundles,
                                          v_rest_mV=V_REST,
                                          train_window_ms=oi.train_window_ms)
            vals[name] = ([L(*q) for q in Q3],
                          [b.amplitude_pA for b in oi.train_bundles],
                          [b.amplitude_pA for b in oi.validation_bundles],
                          [b.amplitude_pA for b in oi.report_bundles])
        assert vals["new"] == vals["old"], vals
        assert max(vals["new"][0]) < 1e6
    finally:
        plst.integrate_long_step(mono, verbose=False)
        ncell.destroy()


def test_LP7_integrate_options():
    mono, ncell = _neuron_cell()
    try:
        cd = roles_cell(6)
        common = dict(ih_protocol=True, ls_window_mode="step", dt_brief_ms=0.1,
                      dt_long_ms=0.1, verbose=False, ss_time_weight="exp",
                      ss_tau_w_ms=5.0)
        vb = [b for b in cd.long_square_subthreshold] + cd.long_square_depolarising
        res = {}
        for lsw in ("none", "peak_decay"):
            plst.integrate_long_step(mono, role_rule="d207", weighting="ls_amplitude",
                                     ls_sample_weight=lsw, **common)
            oi = mono.prepare_optimiser_inputs(cd, fit_target="hyp")
            L = mono._build_loss_function(cell=ncell, train_bundles=oi.train_bundles,
                                          v_rest_mV=V_REST,
                                          train_window_ms=oi.train_window_ms)
            direct = plst.build_multi_protocol_loss(
                ncell, oi.train_bundles, V_REST, ss_window_ms=oi.train_window_ms,
                ls_window_mode="step", weighting="ls_amplitude",
                ss_sample_weight_fn=plst.exp_time_weight(0.5e-3, 5e-3),
                dt_brief_ms=0.1, dt_long_ms=0.1, ls_sample_weight=lsw)
            lv = [L(*q) for q in Q3]
            assert lv == [direct(*q) for q in Q3], lsw
            assert max(lv) < 1e6
            # roles: d207 reached assign_ls_roles
            hyp = [(x["amplitude_pA"], x["role"]) for x in oi.ls_roles["records"]
                   if x["polarity"] == "hyp"]
            assert hyp == [(-10.0, "validate"), (-20.0, "train"), (-30.0, "validate"),
                           (-40.0, "train"), (-50.0, "train"), (-60.0, "train")], hyp
            assert [b.amplitude_pA for b in oi.train_bundles] == \
                [-200.0, -20.0, -40.0, -50.0, -60.0]
            assert oi.report_bundles == []
            # validation RMSDs through the patched monolith, and the
            # module-level bundle_rmsd at its defaults (the gate's path)
            ncell.set_passive(1.0, 30000.0, 150.0); ncell.set_e_pas(V_REST)
            val = [mono._rmsd_for_validation_bundle(ncell, b, V_REST, 150.0)
                   for b in oi.validation_bundles]
            gate = [plst.bundle_rmsd(ncell, b, V_REST, ls_window_mode="step",
                                     dt_brief_ms=0.1, dt_long_ms=0.1)[0] for b in vb]
            res[lsw] = (lv, val, gate)
        assert res["none"][1] == res["peak_decay"][1], "validation RMSD moved"
        assert res["none"][2] == res["peak_decay"][2], "gate RMSD moved"
        assert all(np.isfinite(res["none"][1]))
        assert res["none"][0] != res["peak_decay"][0], "peak_decay not reached"
        # the default options give a different (relative, d006) training loss
        plst.integrate_long_step(mono, **common)
        oi = mono.prepare_optimiser_inputs(cd, fit_target="hyp")
        assert [b.amplitude_pA for b in oi.report_bundles] == [-60.0]
        # role_rule is not applied without ih_protocol
        plst.integrate_long_step(mono, role_rule="d207", verbose=False,
                                 dt_brief_ms=0.1, dt_long_ms=0.1)
        oi = mono.prepare_optimiser_inputs(cd, fit_target="hyp")
        assert oi.ls_roles is None
    finally:
        plst.integrate_long_step(mono, verbose=False)
        ncell.destroy()


def test_LP8_refusals():
    V = ValueError
    cd = roles_cell(4)
    expect_raises(V, plst.assign_ls_roles, cd, role_rule="d208", verbose=False)
    expect_raises(V, plst.assign_ls_roles, cd, role_rule="d207",
                  train_all_hyp=True, verbose=False)
    expect_raises(V, plst.ls_peak_time_s, ss_bundle(-200.0))
    # a long bundle (>= 10 ms) on a 5 ms grid: 2 samples in [t_on + 1 ms, t_off]
    short = ls_bundle(-50.0, fs=200.0, T=0.3, on=0.1, dur=0.012)
    assert not plst._is_brief(short)
    assert len(np.flatnonzero((short.t >= 0.101) & (short.t <= 0.112))) == 2
    expect_raises(V, plst.ls_peak_time_s, short)
    short2 = ls_bundle(-50.0, fs=200.0, T=0.3, on=0.1, dur=0.016)
    assert len(np.flatnonzero((short2.t >= 0.101) & (short2.t <= 0.116))) == 3
    plst.ls_peak_time_s(short2)                        # 3+ samples: accepted
    expect_raises(V, plst.ls_peak_time_s, ls_bundle(-50.0), smooth_ms=0.0)
    expect_raises(V, plst.peak_decay_weight, 0.1, 0.1, 0.5)
    expect_raises(V, plst.peak_decay_weight, 0.2, 0.1, 0.5)
    expect_raises(V, plst.peak_decay_weight, 0.1, 0.2, 0.0)
    expect_raises(V, plst.peak_decay_weight, 0.1, 0.2, -1.0)
    expect_raises(V, plst.peak_decay_weight, 0.1, float("nan"), 0.5)
    cell = StubCell()
    expect_raises(V, plst.bundle_rmsd, cell, ls_bundle(-50.0), V_REST,
                  ls_sample_weight="peak")
    expect_raises(V, plst.build_multi_protocol_loss, cell, _train_set(), V_REST,
                  ls_sample_weight="uniform")
    expect_raises(V, plst.build_multi_protocol_loss, cell, _train_set(), V_REST,
                  ls_sample_weight="peak_decay", ls_tau_d_ms=0.0)
    expect_raises(V, plst.build_multi_protocol_loss, cell, [short], V_REST,
                  ls_sample_weight="peak_decay")      # raised at build time
    zero = ls_bundle(-50.0); zero.amplitude_pA = 0.0
    expect_raises(V, plst.build_multi_protocol_loss, cell, [zero], V_REST,
                  weighting="ls_amplitude")
    expect_raises(V, plst.build_multi_protocol_loss, cell, [], V_REST,
                  weighting="ls_amplitude")
    dummy = object()                                   # refused before mono is touched
    expect_raises(V, plst.integrate_long_step, dummy, role_rule="x")
    expect_raises(V, plst.integrate_long_step, dummy, role_rule="d207",
                  train_all_hyp=True)
    expect_raises(V, plst.integrate_long_step, dummy, ls_sample_weight="x")
    expect_raises(V, plst.integrate_long_step, dummy,
                  ls_sample_weight="peak_decay", ls_tau_d_ms=-5.0)


def test_LP9_ascii_lf():
    for f in ("passive_long_step_training.py", "smoke_ls_protocol.py"):
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
    parts.append("seed %d, contract %s" % (SEED, CONTRACT))
    return " | ".join(parts)


def main():
    checks = [(name, obj) for name, obj in globals().items()
              if name.startswith("test_") and callable(obj)]
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
            status, detail = "FAIL", str(exc) or traceback.format_exc()
        except Exception:  # unexpected errors are reported, never hidden
            status, detail = "ERROR", traceback.format_exc()
        results.append((name, status, time.perf_counter() - start, detail))
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
    for path in _SAVED_ARTIFACTS:
        print("-- arrays from a failed comparison: %s" % path)
    return 1 if (counts["FAIL"] + counts["ERROR"] + counts["TODO"]) else 0


if __name__ == "__main__":
    sys.exit(main())
