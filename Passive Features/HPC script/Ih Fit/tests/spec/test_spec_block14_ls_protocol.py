"""Spec-derived tests of Block 14 (passive_long_step_training.py; SPEC 2.8 (b), (c),
(d) and the Block 14 entry). Integration tester, 2026-11-22.

No NEURON: the simulator is stubbed where a loss is built (the stub returns a
trace planted in the test), so every expected value is the spec's formula
written out here with numpy. The D-006 rule and the default ('none',
'relative') behaviour are compared with the module as it stood at the base
commit 216ecf1 (read with `git show`), the reference the spec names ("at the
contract commit").
"""
import importlib.util
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose

import passive_long_step_training as plst

BASE = "216ecf13ccca455903b2cb05d740dd64f3feff08"
CODE_DIR = Path(__file__).resolve().parents[2]
FS = 50000.0


# ---------------------------------------------------------------------------
#  helpers
# ---------------------------------------------------------------------------
def _old_module(tmp_path_factory=None, _cache={}):
    """passive_long_step_training.py as at BASE, imported under another name."""
    if "m" in _cache:
        return _cache["m"]
    top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=CODE_DIR, capture_output=True,
                         text=True, check=True).stdout.strip()
    rel = CODE_DIR.relative_to(top) / "passive_long_step_training.py"
    src = subprocess.run(["git", "show", "%s:%s" % (BASE, rel.as_posix())], cwd=top, capture_output=True,
                         check=True).stdout
    import tempfile
    d = Path(tempfile.mkdtemp())
    p = d / "plst_base.py"
    p.write_bytes(src)
    spec = importlib.util.spec_from_file_location("plst_base", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    _cache["m"] = m
    return m


def _ls(amp, onset=0.1, dur=0.5, tail=0.2, v=None, fs=FS):
    n = int(round((onset + dur + tail) * fs)) + 1
    t = np.arange(n) / fs
    if v is None:
        v = np.full(n, -70.0)
    return SimpleNamespace(t=t, v_mV=np.asarray(v, float), amplitude_pA=float(amp), stim_onset_s=float(onset),
                           stim_duration_s=float(dur), polarity="hyp" if amp < 0 else "dep")


def _hyp_trace(t, onset, dur, amp, tau=0.03):
    """A plain RC-like trace: deflection amp * 0.1 mV/pA, charging with tau, recovering after the step."""
    v = np.full(t.size, -70.0)
    on = (t >= onset) & (t <= onset + dur)
    v[on] += 0.1 * amp * (1 - np.exp(-(t[on] - onset) / tau))
    after = t > onset + dur
    v[after] += 0.1 * amp * (1 - np.exp(-dur / tau)) * np.exp(-(t[after] - onset - dur) / tau)
    return v


def _cell(hyp_amps, dep_amps=()):
    hyp = []
    for a in hyp_amps:
        b = _ls(a)
        b.v_mV = _hyp_trace(b.t, b.stim_onset_s, b.stim_duration_s, a)
        hyp.append(b)
    dep = [_ls(a) for a in dep_amps]
    rng = np.random.default_rng(0)
    rng.shuffle(hyp)            # the rule sorts by |I| itself
    return SimpleNamespace(long_square_subthreshold=hyp, long_square_depolarising=dep, ls_dep_screen={})


def _recs(rs):
    """records with NaN made comparable"""
    return [{k: ("nan" if isinstance(v, float) and v != v else v) for k, v in r.items()} for r in rs]


def _d207_expected(n):
    """SPEC 2.8 (b) written out: 1-based roles of h_1..h_n."""
    if n == 0:
        return []
    m = math.ceil(n / 2)
    train = [i for i in range(2, n + 1) if i != m]
    if not train:
        return ["train"] * n
    return ["validate" if i in (1, m) else "train" for i in range(1, n + 1)]


# ---------------------------------------------------------------------------
#  (b) roles
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n", range(0, 9))
def test_d207_roles_by_hand(n):
    amps = [-10.0 * (k + 1) for k in range(n)]
    cd = _cell(amps, dep_amps=(20.0, 40.0, 60.0, 80.0))
    out = plst.assign_ls_roles(cd, role_rule="d207", verbose=False,
                               n_drop_weakest=3, n_drop_strongest=2)   # ignored under d207
    hyp_recs = [r for r in out["records"] if r["polarity"] == "hyp"]
    assert [r["amplitude_pA"] for r in hyp_recs] == sorted(amps, key=abs)   # h_1..h_n by |I|
    assert [r["role"] for r in hyp_recs] == _d207_expected(n)
    assert out["report"] == []                                           # no report-only bundle
    trained = sorted(float(b.amplitude_pA) for b in out["train"])
    assert trained == sorted(r["amplitude_pA"] for r in hyp_recs if r["role"] == "train")
    for r in hyp_recs:
        assert "d207" in r["reason"] or r["reason"] == "weakest step (held out)"
    # the depolarising steps exactly as under D-006: dep_n_validation = 3 smallest
    dep_recs = [r for r in out["records"] if r["polarity"] == "dep"]
    old = _old_module().assign_ls_roles(cd, verbose=False)
    assert _recs(dep_recs) == _recs([r for r in old["records"] if r["polarity"] == "dep"])
    assert sum(1 for b in out["validate"] if b.amplitude_pA > 0) == 3


def test_d207_spec_examples():
    """The examples written in 2.8 (b): n = 6: h2,h4,h5,h6; 5: h2,h4,h5; 3: h3; 2: h2."""
    for n, want in ((6, [2, 4, 5, 6]), (5, [2, 4, 5]), (3, [3]), (2, [2]), (1, [1])):
        got = [i + 1 for i, r in enumerate(_d207_expected(n)) if r == "train"]
        assert got == want
        cd = _cell([-10.0 * (k + 1) for k in range(n)])
        out = plst.assign_ls_roles(cd, role_rule="d207", verbose=False)
        assert sorted(abs(b.amplitude_pA) / 10 for b in out["train"]) == want


@pytest.mark.parametrize("n", range(0, 8))
def test_d006_unchanged_against_base(n):
    cd = _cell([-10.0 * (k + 1) for k in range(n)], dep_amps=(20.0, 40.0))
    new = plst.assign_ls_roles(cd, verbose=False)
    old = _old_module().assign_ls_roles(cd, verbose=False)
    assert _recs(new["records"]) == _recs(old["records"])
    for k in ("train", "validate", "report"):
        assert [id(b) for b in new[k]] == [id(b) for b in old[k]]
    # the floor too
    new = plst.assign_ls_roles(cd, verbose=False, v_trough_min_mV=-73.0)
    old = _old_module().assign_ls_roles(cd, verbose=False, v_trough_min_mV=-73.0)
    assert _recs(new["records"]) == _recs(old["records"])


def test_d207_floor_demotes_a_training_step():
    cd = _cell([-10.0, -20.0, -30.0, -40.0, -50.0, -60.0])
    floor = -74.5        # troughs ~ -70 + 0.1 * amp: -75 and -76 (h5, h6) are below it
    out = plst.assign_ls_roles(cd, role_rule="d207", verbose=False, v_trough_min_mV=floor)
    recs = [r for r in out["records"] if r["polarity"] == "hyp"]
    roles = [r["role"] for r in recs]
    assert roles == ["validate", "train", "validate", "train", "report", "report"]


def test_role_refusals():
    cd = _cell([-10.0, -20.0, -30.0])
    with pytest.raises(ValueError):
        plst.assign_ls_roles(cd, role_rule="d207", train_all_hyp=True, verbose=False)
    with pytest.raises(ValueError):
        plst.assign_ls_roles(cd, role_rule="D207", verbose=False)


# ---------------------------------------------------------------------------
#  (d) the peak and the weight
# ---------------------------------------------------------------------------
def _peak_oracle(b, smooth_ms=1.0):
    """2.8 (d) written out: the boxcar of 2h+1 samples (h = round(smooth_ms / 2 / dt), the as-built
    width), averaging the finite samples it covers, evaluated at every grid sample of
    [t_on + 1 ms, t_off]; argmin (argmax when depolarising), earliest on ties."""
    t, v = b.t, b.v_mV
    dt = float(np.median(np.diff(t)))
    h = int(round(smooth_ms * 1e-3 / 2 / dt))
    t_on = b.stim_onset_s
    t_off = t_on + b.stim_duration_s
    idx = np.flatnonzero((t >= t_on + 1e-3) & (t <= t_off))
    sm = np.array([np.nanmean(v[max(j - h, 0):j + h + 1]) for j in idx])
    k = int(np.argmax(sm)) if b.amplitude_pA > 0 else int(np.argmin(sm))
    return float(t[idx[k]])


def test_peak_time_known_trough_and_spikes():
    b = _ls(-50.0)
    t = b.t
    b.v_mV = -70.0 + 1000.0 * (t - 0.2) ** 2 - 40.0 * ((t >= 0.1) & (t <= 0.6))
    j_tr = int(np.argmin(np.abs(t - 0.2)))
    assert plst.ls_peak_time_s(b) == t[j_tr]
    # spikes of 5 samples (0.1 ms) much deeper than the trough elsewhere in the step do not move it
    v2 = b.v_mV.copy()
    for tc in (0.35, 0.45, 0.55):
        j = int(round(tc * FS))
        v2[j:j + 5] -= 50.0
    b2 = _ls(-50.0, v=v2)
    assert plst.ls_peak_time_s(b2) == t[j_tr]


def test_peak_time_random_trace_matches_oracle():
    rng = np.random.default_rng(7)
    for amp in (-90.0, 60.0):
        b = _ls(amp)
        b.v_mV = _hyp_trace(b.t, 0.1, 0.5, amp) + 0.3 * rng.standard_normal(b.t.size)
        b.v_mV[rng.integers(0, b.t.size, 20)] = np.nan            # non-finite samples are skipped
        assert plst.ls_peak_time_s(b) == _peak_oracle(b)


def test_peak_time_monotone_tie_and_guard():
    b = _ls(-50.0)
    t = b.t
    on = (t >= 0.1) & (t <= 0.6)
    b.v_mV = np.where(on, -70.0 - 10.0 * (t - 0.1), -70.0)         # monotone: the last sample of the interval
    idx = np.flatnonzero((t >= 0.1 + 1e-3) & (t <= 0.6))
    assert plst.ls_peak_time_s(b) == _peak_oracle(b)
    assert abs(plst.ls_peak_time_s(b) - t[idx[-1]]) <= 0.5e-3      # within the boxcar's half-width of the end
    flat = _ls(-50.0)                                               # constant: every sample ties -> the earliest
    assert plst.ls_peak_time_s(flat) == t[idx[0]]
    # a deep dip inside the 1 ms guard is outside the search interval, but the boxcar reaches it
    g = _ls(-50.0)
    g.v_mV = np.where(on, -71.0, -70.0)
    j = int(round(0.1007 * FS))
    g.v_mV[j] = -200.0
    assert plst.ls_peak_time_s(g) == _peak_oracle(g)
    assert plst.ls_peak_time_s(g) < 0.1 + 1.5e-3


def test_peak_time_refusals():
    brief = _ls(-200.0, dur=0.0005)
    with pytest.raises(ValueError):
        plst.ls_peak_time_s(brief)
    coarse = _ls(-50.0, onset=0.02, dur=0.012, tail=0.02, fs=200.0)  # dt 5 ms: 2 samples in [1, 12] ms
    with pytest.raises(ValueError):
        plst.ls_peak_time_s(coarse)


def test_peak_decay_weight_values():
    t_on, t_pk, tau = 0.27, 0.42, 0.5
    w = plst.peak_decay_weight(t_on, t_pk, tau)
    t = np.linspace(0.0, 2.0, 20001)
    exp = np.where(t < t_on, 0.0, np.where(t <= t_pk, (t - t_on) / (t_pk - t_on), np.exp(-(t - t_pk) / tau)))
    assert_allclose(w(t), exp, rtol=1e-15, atol=0)                  # the same arithmetic: roundoff only
    assert w(np.array([t_on]))[0] == 0.0 and w(np.array([t_pk]))[0] == 1.0
    assert w(np.array([t_pk + tau]))[0] == pytest.approx(math.exp(-1), rel=1e-14)
    assert w(np.array([t_on - 1e-9]))[0] == 0.0
    for bad in ((0.3, 0.3, 0.5), (0.3, 0.2, 0.5), (0.1, 0.2, 0.0), (0.1, 0.2, -1.0), (0.1, float("nan"), 0.5)):
        with pytest.raises(ValueError):
            plst.peak_decay_weight(*bad)


# ---------------------------------------------------------------------------
#  (d) the weighted RMSD; 'none' bit for bit
# ---------------------------------------------------------------------------
def _stub_sim(monkeypatch, module, sim):
    monkeypatch.setattr(module, "_simulate_long", lambda cell, b, v_rest, dt_ms=None: sim[id(b)])


@pytest.mark.parametrize("mode", ["step", "sweep", "after_onset"])
def test_weighted_rmsd_formula(monkeypatch, mode):
    rng = np.random.default_rng(3)
    b = _ls(-80.0)
    b.v_mV = _hyp_trace(b.t, 0.1, 0.5, -80.0) + 0.05 * rng.standard_normal(b.t.size)
    ts = np.arange(0, b.t[-1] + 1e-4, 1e-4)                        # a simulator grid of 0.1 ms
    vs = _hyp_trace(ts, 0.1, 0.5, -80.0, tau=0.035) - 3.0
    _stub_sim(monkeypatch, plst, {id(b): (ts, vs)})
    r, _ = plst.bundle_rmsd(None, b, -70.0, ls_window_mode=mode, ls_sample_weight="peak_decay", ls_tau_d_ms=500.0)
    # expected: baseline means on [0, t_on], window, interpolation, weight -- written out
    pre = (0.0, 0.1)
    be = b.v_mV[(b.t >= pre[0]) & (b.t <= pre[1])].mean()
    bs = vs[(ts >= pre[0]) & (ts <= pre[1])].mean()
    w_end = {"step": 0.6, "sweep": b.t[-1], "after_onset": 0.1 + 0.150}[mode]
    m = (b.t >= 0.1) & (b.t <= w_end)
    d = (b.v_mV - be)[m] - np.interp(b.t[m], ts, vs - bs)
    t_pk = _peak_oracle(b)
    tt = b.t[m]
    w = np.where(tt < 0.1, 0, np.where(tt <= t_pk, (tt - 0.1) / (t_pk - 0.1), np.exp(-(tt - t_pk) / 0.5)))
    exp = math.sqrt(np.sum(w * d * d) / np.sum(w))
    assert r == pytest.approx(exp, rel=1e-12)                       # 1e-12: summation order only
    # 'none' and the default: the unweighted RMSD, bit for bit with the base commit's function
    old = _old_module()
    _stub_sim(monkeypatch, old, {id(b): (ts, vs)})
    r0, d0 = old.bundle_rmsd(None, b, -70.0, ls_window_mode=mode)
    r1, d1 = plst.bundle_rmsd(None, b, -70.0, ls_window_mode=mode)
    r2, d2 = plst.bundle_rmsd(None, b, -70.0, ls_window_mode=mode, ls_sample_weight="none")
    assert r1 == r0 and r2 == r0 and d1 == d0 and d2 == d0
    assert r0 == pytest.approx(math.sqrt(np.mean(d * d)), rel=1e-12)
    # the deflection is never weighted
    _, dw = plst.bundle_rmsd(None, b, -70.0, ls_window_mode=mode, ls_sample_weight="peak_decay")
    assert dw == d0


def test_weight_depends_on_the_recorded_trace_only(monkeypatch):
    """t_pk and w_b come from the recorded trace: two different model traces leave w unchanged, so the
    weighted RMSD of a model equal to the data is 0 and of data + c (in the window) is |c|."""
    b = _ls(-80.0)
    b.v_mV = _hyp_trace(b.t, 0.1, 0.5, -80.0)
    ts = b.t.copy()
    _stub_sim(monkeypatch, plst, {id(b): (ts, b.v_mV.copy())})
    r, _ = plst.bundle_rmsd(None, b, -70.0, ls_window_mode="step", ls_sample_weight="peak_decay")
    assert r == 0.0
    v2 = b.v_mV.copy()
    v2[ts > 0.1 + 1e-9] += 0.7                                     # a constant offset inside the window only
    _stub_sim(monkeypatch, plst, {id(b): (ts, v2)})
    r, _ = plst.bundle_rmsd(None, b, -70.0, ls_window_mode="step", ls_sample_weight="peak_decay")
    assert r == pytest.approx(0.7, rel=1e-12)


# ---------------------------------------------------------------------------
#  (c) the step weight ls_amplitude
# ---------------------------------------------------------------------------
class _NoSpec:
    def apply_q(self, cell, q, v_rest):
        return {}


def _plant(monkeypatch, planted):
    def fake(cell, b, v_rest_mV, **kw):
        return planted[id(b)]
    monkeypatch.setattr(plst, "bundle_rmsd", fake)


def _brief(amp=-200.0):
    t = np.arange(-0.01, 0.2, 1 / FS)
    return SimpleNamespace(t=t, v_mV=np.full(t.size, -70.0), amplitude_pA=amp, stim_onset_s=0.0,
                           stim_duration_s=0.0005, polarity="hyp")


@pytest.mark.parametrize("n_ss,ls_amps", [(2, (-30.0, -70.0, -110.0)), (1, (-50.0,)), (0, (-30.0, -90.0)),
                                          (3, ())])
def test_ls_amplitude_formula(monkeypatch, n_ss, ls_amps):
    rng = np.random.default_rng(len(ls_amps) + 10 * n_ss)
    ss = [_brief() for _ in range(n_ss)]
    ls = [_ls(a) for a in ls_amps]
    planted = {id(b): (float(rng.uniform(0.1, 2.0)), float(rng.uniform(1.0, 20.0) * (-1) ** k))
               for k, b in enumerate(ss + ls)}
    _plant(monkeypatch, planted)
    bundles = ss + ls
    L = plst.build_multi_protocol_loss(None, bundles, -70.0, weighting="ls_amplitude", spec=_NoSpec())(0.0)
    tot = sum(abs(a) for a in ls_amps)
    term = lambda b: planted[id(b)][0] / max(abs(planted[id(b)][1]), 1e-6)   # noqa: E731
    exp = (sum(term(b) for b in ss) + len(ls) * sum(abs(b.amplitude_pA) / tot * term(b) for b in ls)) / \
        (len(ss) + len(ls))
    assert L == pytest.approx(exp, rel=1e-12)
    rel = plst.build_multi_protocol_loss(None, bundles, -70.0, weighting="relative", spec=_NoSpec())(0.0)
    assert rel == pytest.approx(np.mean([term(b) for b in bundles]), rel=1e-12)
    if len(set(abs(a) for a in ls_amps)) <= 1:                      # equal amplitudes: the relative loss
        assert L == pytest.approx(rel, rel=1e-12)


def test_ls_amplitude_equal_amplitudes_is_relative(monkeypatch):
    ss = [_brief(), _brief()]
    ls = [_ls(-60.0), _ls(-60.0), _ls(-60.0)]
    rng = np.random.default_rng(1)
    planted = {id(b): (float(rng.uniform(0.1, 2)), float(rng.uniform(1, 9))) for b in ss + ls}
    _plant(monkeypatch, planted)
    a = plst.build_multi_protocol_loss(None, ss + ls, -70.0, weighting="ls_amplitude", spec=_NoSpec())(0.0)
    b = plst.build_multi_protocol_loss(None, ss + ls, -70.0, weighting="relative", spec=_NoSpec())(0.0)
    assert a == pytest.approx(b, rel=1e-12)


def test_loss_refusals_at_build(monkeypatch):
    zero = _ls(0.0)
    zero.amplitude_pA = 0.0
    with pytest.raises(ValueError):
        plst.build_multi_protocol_loss(None, [_brief(), zero], -70.0, weighting="ls_amplitude", spec=_NoSpec())
    with pytest.raises(ValueError):
        plst.build_multi_protocol_loss(None, [_ls(-50.0)], -70.0, ls_sample_weight="peakdecay", spec=_NoSpec())
    coarse = _ls(-50.0, onset=0.02, dur=0.012, tail=0.02, fs=200.0)   # no usable peak
    with pytest.raises(ValueError):
        plst.build_multi_protocol_loss(None, [coarse], -70.0, ls_sample_weight="peak_decay", spec=_NoSpec())


def test_defaults_loss_bit_for_bit_against_base(monkeypatch):
    """LP6 in spirit: every new argument at its default, the loss equals the base commit's, bit for bit
    (the simulator stubbed identically in both modules)."""
    rng = np.random.default_rng(5)
    bundles = [_brief(), _ls(-30.0), _ls(-70.0)]
    sims = {}
    for b in bundles:
        b.v_mV = b.v_mV + 0.1 * rng.standard_normal(b.t.size)
        sims[id(b)] = (b.t.copy(), b.v_mV + rng.normal(0, 0.3))
    old = _old_module()
    for mod in (plst, old):
        monkeypatch.setattr(mod, "_simulate_long", lambda cell, b, v, dt_ms=None: sims[id(b)])
        monkeypatch.setattr(mod, "_simulate_brief", lambda cell, b, v, dt_ms=None, **k: sims[id(b)])
    for wt in ("relative", "equal"):
        a = plst.build_multi_protocol_loss(None, bundles, -70.0, weighting=wt, spec=_NoSpec())(0.0)
        b = old.build_multi_protocol_loss(None, bundles, -70.0, weighting=wt, spec=_NoSpec())(0.0)
        assert a == b, wt


def test_ascii_lf():
    raw = (CODE_DIR / "passive_long_step_training.py").read_bytes()
    assert all(c < 128 for c in raw) and b"\r" not in raw
