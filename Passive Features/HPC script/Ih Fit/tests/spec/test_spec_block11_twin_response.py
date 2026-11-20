"""Spec-derived tests of Block 11 (synthetic_ground_truth.simulate_ss_clean,
simulate_ls_clean; SPEC 2.7 (f) and the Block 11 entry with amendments 2 and 4).
Integration tester, 2026-11-20.

Two independent oracles:
(1) a duck-typed cell (the interface `simulate(stim_amp_pA, stim_delay_ms,
    stim_dur_ms, tstop_ms, v_init_mV, dt_ms) -> (t_ms, v)`) whose "voltage" is
    NEURON's clock itself, v = t: the output must then equal the NEURON-clock
    time of each output sample as the spec defines it (settle_ms - t_s[0] 1e3 +
    t_s[j] 1e3 for SS; settle_ms + j / f_s 1e3 for LS), and the recorded stimulus
    arguments the spec's delay and duration;
(2) a one-compartment passive NEURON cell written here, against the analytic RC
    response V(t) = v0 + A R (1 - exp(-t / tau)) during the pulse.
"""
import math

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import synthetic_ground_truth as sgt

FS = 50000.0
T_SS = (np.arange(10500) - 500) / FS          # the archive's SS window axis


class ClockCell:
    """v = NEURON's clock (ms) on a fixed-step grid k dt; records the call."""
    def __init__(self):
        self.calls = []

    def simulate(self, stim_amp_pA, stim_delay_ms, stim_dur_ms, tstop_ms, v_init_mV, dt_ms):
        self.calls.append(dict(amp=stim_amp_pA, delay=stim_delay_ms, dur=stim_dur_ms, tstop=tstop_ms,
                               v0=v_init_mV, dt=dt_ms))
        n = int(math.floor(tstop_ms / dt_ms + 1e-9)) + 1
        t = np.arange(n) * dt_ms
        return t, t.copy()


@pytest.mark.parametrize("settle", [0.0, 37.5])
def test_ss_clock_alignment_and_current(settle):
    c = ClockCell()
    v, i = sgt.simulate_ss_clean(c, amplitude_pA=-200.0, duration_s=5e-4, t_s=T_SS, v_init_mV=-70.0,
                                 dt_ms=0.025, settle_ms=settle)
    exp = settle - T_SS[0] * 1e3 + T_SS * 1e3
    # tolerance: float64 roundoff of the times (~1e-13 ms at 250 ms); the
    # interpolation of the identity is exact
    assert_allclose(v, exp, rtol=0, atol=1e-9)
    call = c.calls[0]
    assert call["amp"] == -200.0 and call["v0"] == -70.0 and call["dt"] == 0.025
    assert abs(call["delay"] - (settle + 10.0)) < 1e-12 and abs(call["dur"] - 0.5) < 1e-12
    assert call["tstop"] >= exp[-1]
    on = np.flatnonzero(i)
    assert on[0] == 500 and on[-1] == 524 and np.all(i[on] == -200.0)   # amendment 2 (ii): 25 samples
    assert v.dtype == np.float64 and i.dtype == np.float64 and v.shape == i.shape == T_SS.shape


@pytest.mark.parametrize("settle", [0.0, 6000.0])
def test_ls_clock_alignment_and_current(settle):
    c = ClockCell()
    n = 3000
    v, i = sgt.simulate_ls_clean(c, amplitude_pA=30.0, onset_s=0.01, duration_s=0.02, n_samples=n, fs_Hz=FS,
                                 v_init_mV=-65.0, dt_ms=0.025, settle_ms=settle)
    exp = settle + np.arange(n) / FS * 1e3
    assert_allclose(v, exp, rtol=0, atol=1e-8)          # roundoff of ms times up to 6060 ms
    call = c.calls[0]
    assert abs(call["delay"] - (settle + 10.0)) < 1e-9 and abs(call["dur"] - 20.0) < 1e-9
    on = np.flatnonzero(i)
    assert on[0] == 500 and on[-1] == 500 + 1000 - 1 and np.all(i[on] == 30.0)


def test_real_ls_geometry_current():
    c = ClockCell()
    _, i = sgt.simulate_ls_clean(c, amplitude_pA=-110.0, onset_s=0.27, duration_s=1.0, n_samples=363500,
                                 fs_Hz=FS, v_init_mV=-70.0, dt_ms=0.025)
    on = np.flatnonzero(i)
    assert on[0] == 13500 and on[-1] == 63499


@pytest.mark.parametrize("kw", [
    dict(t_s=np.array([0.0, 1e-5])), dict(t_s=np.array([-1e-5])),
    dict(t_s=np.r_[T_SS[:100], T_SS[100] + 1e-7]), dict(duration_s=0.0), dict(dt_ms=0.0),
    dict(settle_ms=-1.0), dict(settle_ms=float("nan")), dict(amplitude_pA=float("inf")),
    dict(duration_s=1.0),    # does not fit in the window
])
def test_ss_refusals(kw):
    a = dict(amplitude_pA=1.0, duration_s=5e-4, t_s=T_SS, v_init_mV=-70.0, dt_ms=0.025, settle_ms=0.0)
    a.update(kw)
    with pytest.raises(ValueError):
        sgt.simulate_ss_clean(ClockCell(), **a)


@pytest.mark.parametrize("kw", [dict(n_samples=1), dict(n_samples=10.0), dict(fs_Hz=0.0), dict(onset_s=-0.1),
                                dict(onset_s=0.05, duration_s=0.02), dict(duration_s=1e-6)])
def test_ls_refusals(kw):
    a = dict(amplitude_pA=1.0, onset_s=0.01, duration_s=0.02, n_samples=3000, fs_Hz=FS, v_init_mV=-70.0,
             dt_ms=0.025)
    a.update(kw)
    with pytest.raises(ValueError):
        sgt.simulate_ls_clean(ClockCell(), **a)


# ---------------------------------------------------------------------------
#  (2) A one-compartment passive NEURON cell against the analytic RC response
# ---------------------------------------------------------------------------
class SomaCell:
    """One cylindrical soma (pas only); diameter = length = 20 um."""
    CM, RM, EPAS = 1.0, 20000.0, -70.0       # uF/cm2, Ohm cm2, mV -> tau = 20 ms

    def __init__(self):
        from neuron import h
        h.load_file("stdrun.hoc")
        self.h = h
        self.sec = h.Section(name="tester_soma")
        self.sec.L = self.sec.diam = 20.0
        self.sec.nseg = 1
        self.sec.cm = self.CM
        self.sec.insert("pas")
        self.sec(0.5).pas.g = 1.0 / self.RM
        self.sec(0.5).pas.e = self.EPAS
        self.stim = h.IClamp(self.sec(0.5))
        area_cm2 = math.pi * 20e-4 * 20e-4
        self.R_MOhm = self.RM / area_cm2 / 1e6
        self.tau_ms = self.RM * self.CM * 1e-3

    def simulate(self, stim_amp_pA, stim_delay_ms, stim_dur_ms, tstop_ms, v_init_mV, dt_ms):
        h = self.h
        self.stim.amp = stim_amp_pA * 1e-3
        self.stim.delay = stim_delay_ms
        self.stim.dur = stim_dur_ms
        tv = h.Vector().record(h._ref_t)
        vv = h.Vector().record(self.sec(0.5)._ref_v)
        h.dt = dt_ms
        h.steps_per_ms = 1.0 / dt_ms
        h.secondorder = 0
        h.finitialize(v_init_mV)
        h.continuerun(tstop_ms)
        return np.array(tv), np.array(vv)


@pytest.fixture(scope="module")
def soma():
    pytest.importorskip("neuron")
    return SomaCell()


def test_ls_against_analytic_rc(soma):
    """A 20 ms step of -50 pA from 10 ms: V = v0 + A R (1 - exp(-(t - t_on)/tau))
    on the step. Backward Euler at dt = 0.0025 ms has a relative error near
    dt / (2 tau) = 6e-5 of the deflection; a one-sample (20 us) shift would give
    about 1e-3 of it: tolerance 3e-4 of |A R|."""
    A = -50.0
    n = 2000
    v, _ = sgt.simulate_ls_clean(soma, amplitude_pA=A, onset_s=0.01, duration_s=0.02, n_samples=n, fs_Hz=FS,
                                 v_init_mV=soma.EPAS, dt_ms=0.0025)
    t = np.arange(n) / FS * 1e3
    dv = A * 1e-3 * soma.R_MOhm      # nA * MOhm = mV
    exp = soma.EPAS + np.where(t >= 10.0, dv * (1 - np.exp(-np.clip(t - 10.0, 0, None) / soma.tau_ms)), 0.0)
    exp = np.where(t >= 30.0, soma.EPAS + dv * (1 - math.exp(-20.0 / soma.tau_ms))
                   * np.exp(-(t - 30.0) / soma.tau_ms), exp)
    assert np.max(np.abs(v - exp)) < 3e-4 * abs(dv)
    assert np.all(np.abs(v[t < 10.0] - soma.EPAS) < 1e-9)       # baseline = v_init (passive, v0 = e_pas)


def test_ss_against_analytic_rc_and_linearity(soma):
    """The 0.5 ms pulse of the SS axis: the decay after the pulse is
    dv (1 - e^{-0.5/tau}) e^{-(t - 0.5)/tau}; same tolerance logic. Linearity:
    v(2A) - v0 = 2 (v(A) - v0) to 1e-9 mV (linear system, roundoff)."""
    A = 200.0
    v1, _ = sgt.simulate_ss_clean(soma, amplitude_pA=A, duration_s=5e-4, t_s=T_SS[:3000], v_init_mV=soma.EPAS,
                                  dt_ms=0.0025)
    v2, _ = sgt.simulate_ss_clean(soma, amplitude_pA=2 * A, duration_s=5e-4, t_s=T_SS[:3000],
                                  v_init_mV=soma.EPAS, dt_ms=0.0025)
    assert_allclose(v2 - soma.EPAS, 2 * (v1 - soma.EPAS), rtol=0, atol=1e-9)
    t = T_SS[:3000] * 1e3
    dv = A * 1e-3 * soma.R_MOhm
    peak = dv * (1 - math.exp(-0.5 / soma.tau_ms))
    exp = np.where(t < 0, soma.EPAS,
                   np.where(t <= 0.5, soma.EPAS + dv * (1 - np.exp(-np.clip(t, 0, None) / soma.tau_ms)),
                            soma.EPAS + peak * np.exp(-(t - 0.5) / soma.tau_ms)))
    late = t > 1.0
    assert np.max(np.abs(v1[late] - exp[late])) < 3e-3 * peak
    assert np.all(np.abs(v1[t < 0] - soma.EPAS) < 1e-9)


def test_settle_neutral_on_passive(soma):
    """Amendment 2/4: settle 0 and 500 ms give the same LS voltages to 1e-7 mV."""
    a, _ = sgt.simulate_ls_clean(soma, amplitude_pA=-30.0, onset_s=0.005, duration_s=0.01, n_samples=1500,
                                 fs_Hz=FS, v_init_mV=soma.EPAS, dt_ms=0.025, settle_ms=0.0)
    b, _ = sgt.simulate_ls_clean(soma, amplitude_pA=-30.0, onset_s=0.005, duration_s=0.01, n_samples=1500,
                                 fs_Hz=FS, v_init_mV=soma.EPAS, dt_ms=0.025, settle_ms=500.0)
    assert np.max(np.abs(a - b)) < 1e-7
    c, _ = sgt.simulate_ls_clean(soma, amplitude_pA=-30.0, onset_s=0.005, duration_s=0.01, n_samples=1500,
                                 fs_Hz=FS, v_init_mV=soma.EPAS, dt_ms=0.025, settle_ms=0.0)
    assert_array_equal(a, c)      # no state between calls
