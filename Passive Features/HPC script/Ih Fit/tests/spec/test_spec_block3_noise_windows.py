"""Spec-derived tests, Block 3: noise_windows and noise_cell_stats (SPEC 2.2
(a)-(g), 2.4 (TL1), (TL2), (OF1), section 3, Block 3's entry). Fixtures are
the tester's own writer-like cells (tests/spec/_fixtures.py); expected values
come from numpy/scipy/statsmodels and hand computations.
"""
import csv
import json
import math
import os
import shutil

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.optimize import curve_fit, minimize_scalar
from statsmodels.tsa.stattools import acf

import _fixtures as fx
import noise_cell_stats as ncs
import noise_windows as nw
from noise_windows import DecayFit, LayoutError, WindowConfig

DROP = [(0, 3), (0, 4), (1, 10), (1, 11), (1, 12)]


@pytest.fixture(scope="module")
def cell(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("b3"))
    d = fx.write_cell(root, specimen_id=111, ss_tilts=(0.1, -0.05), drop=DROP, ls_tilts=(0.05, -0.02, 0.0, 0.08))
    return d


@pytest.fixture(scope="module")
def ss(cell):
    return nw.load_ss_windows(cell)


def _copy(cell, tmp_path, name):
    dst = os.path.join(str(tmp_path), name)
    shutil.copytree(cell, dst)
    return dst


def _planted_positions():
    return [k for k in range(37) if k not in (3, 4)] + [k for k in range(37) if k not in (10, 11, 12)]


# --- O3.1 configuration -------------------------------------------------------
def test_O3_1_window_config_defaults():
    c = WindowConfig()
    expected = dict(t_ss_floor_ms=80.0, kappa_tr=0.1, fit_start_ms=5.0, decay_tau_bounds_ms=(1.0, 1000.0),
                    decay_grid_n=301, min_window_ms=50.0, fallback_t_ms=80.0, min_windows_per_group=5,
                    ss_pre_ms=10.0, ls_pre_gap_ms=5.0, ls_first_ms=50.0, ls_last_ms=132.5, ls_end_ms=500.0,
                    ls_polarities=("hyp", "dep"), ls_amp_min_pA=5.0, ss_period_samples=None, min_overlap=100,
                    transient_bin_ms=5.0, transient_floor_z=None, transient_fallback_t_ms=190.0)
    for k, v in expected.items():
        assert getattr(c, k) == v, k
    with pytest.raises(Exception):
        c.kappa_tr = 0.2


# --- O3.2 layout --------------------------------------------------------------
def test_O3_2_layout_read_back(ss):
    assert ss.n_windows == 74 - len(DROP)
    assert ss.period_samples == fx.P and ss.train_length == 37 and ss.pre_n == fx.PRE_N
    assert ss.n_raw == 74 and ss.n_sweeps == 2 and ss.fs == fx.FS and ss.dt == pytest.approx(1 / fx.FS, rel=1e-15)
    assert ss.position.tolist() == _planted_positions()
    assert ss.sweep_number.tolist() == [31] * 35 + [35] * 34
    assert_array_equal(ss.polarity_is_dep, np.asarray(_planted_positions()) % 2 == 0)
    assert ss.t[fx.PRE_N] == 0.0 and ss.v.shape == (69, fx.N_W) and ss.v.dtype == np.float64


def test_O3_2_visiting_order_from_metadata(tmp_path):
    d = fx.write_cell(str(tmp_path), specimen_id=5, sweeps=(35, 31), drop=[(0, 2)])
    s = nw.load_ss_windows(d)
    assert s.sweep_number.tolist() == [35] * 36 + [31] * 37
    assert s.position.tolist()[:3] == [0, 1, 3]
    d2 = fx.write_cell(str(tmp_path), specimen_id=6, sweeps=(31, 35), listed_sweeps=False)
    assert nw.load_ss_windows(d2).sweep_number.tolist() == [31] * 37 + [35] * 37


def _edit_json(d, fn):
    p = os.path.join(d, "metadata.json")
    m = json.load(open(p))
    fn(m)
    json.dump(m, open(p, "w"))


def _edit_csv(d, fn):
    p = os.path.join(d, "trace_qc_results.csv")
    rows = list(csv.DictReader(open(p, newline="")))
    fn(rows)
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _edit_npz(d, fn):
    p = os.path.join(d, "ss_pulses.npz")
    z = dict(np.load(p))
    fn(z)
    np.savez(p, **z)


TAMPERS = {
    "rows_vs_n_raw": lambda d: _edit_json(d, lambda m: m["ss_extraction"].update(n_pulses_raw=75)),
    "rows_vs_ampdur": lambda d: _edit_json(d, lambda m: m["ss_extraction"].update(n_pulses_ampdur_qc=73)),
    "passed_vs_K": lambda d: _edit_csv(d, lambda r: r[0].update(passed="False")),
    "polarity_vs_npz": lambda d: _edit_npz(d, lambda z: z["polarity_is_dep"].__setitem__(5, ~z["polarity_is_dep"][5])),
    "train_not_alternating": lambda d: _edit_csv(d, lambda r: r[3].update(polarity="dep")),   # row 3 is a dropped hyp
    "sweep_not_listed": lambda d: _edit_json(d, lambda m: m["ss_extraction"].update(sweep_numbers=[31, 36])),
    "n_raw_not_multiple": lambda d: _edit_json(d, lambda m: m["ss_extraction"].update(sweep_numbers=[31, 35, 40])),
    "overlap_disagrees": lambda d: _edit_npz(d, lambda z: z["v"].__setitem__((7, slice(10200, 10210)), 0.0)),
}


@pytest.mark.parametrize("name", sorted(TAMPERS))
def test_O3_2_layout_refusals(cell, tmp_path, name):
    d = _copy(cell, tmp_path, name)
    TAMPERS[name](d)
    with pytest.raises(LayoutError):
        nw.load_ss_windows(d)


def test_O3_2_wrong_period_and_no_adjacent_pair(cell, tmp_path):
    with pytest.raises(LayoutError):
        nw.load_ss_windows(cell, WindowConfig(ss_period_samples=10024))
    assert nw.load_ss_windows(cell, WindowConfig(ss_period_samples=10025)).period_samples == 10025
    drop = [(s, k) for s in (0, 1) for k in range(37) if k % 2 == 1]       # no two stored windows adjacent
    d = fx.write_cell(str(tmp_path), specimen_id=9, drop=drop)
    with pytest.raises(LayoutError):
        nw.load_ss_windows(d)


# --- O3.3 the exact decay fit --------------------------------------------------
T_FIT = np.arange(250, 10000) / 50.0          # ms: t >= 5 ms and t < 200 ms at 50 kHz


@pytest.mark.parametrize("A,tau,B", [(0.4, 28.143, -70.3), (-0.4, 28.143, -70.3), (1.2, 5.0, 0.0), (0.05, 300.0, 3.0)])
def test_O3_3_fit_decay_noise_free_exact(A, tau, B):
    y = B + A * np.exp(-T_FIT / tau)
    f = nw.fit_decay(T_FIT, y)
    assert f.ok and not f.at_bound
    # 1e-9 relative (SPEC W2): a noise-free model is recovered to the polish tolerance
    assert f.tau_ms == pytest.approx(tau, rel=1e-9)
    assert f.A_mV == pytest.approx(A, rel=1e-8)
    assert f.B_mV == pytest.approx(B, rel=1e-9, abs=1e-9)


def _rss(t, y, tau):
    X = np.column_stack([np.ones_like(t), np.exp(-t / tau)])
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ c
    return float(r @ r), c


def test_O3_3_fit_decay_noisy_is_the_least_squares_optimum():
    rng = np.random.default_rng(31)
    y = -70.0 + 0.3 * np.exp(-T_FIT / 25.0) + 0.02 * rng.standard_normal(T_FIT.size)
    f = nw.fit_decay(T_FIT, y)
    assert f.ok
    # independent optimum: dense profile (own lstsq) + scipy bounded search, and curve_fit from the truth
    grid = np.geomspace(f.tau_ms / 2, f.tau_ms * 2, 20001)
    dense = np.array([_rss(T_FIT, y, g)[0] for g in grid[::20]])
    assert f.rss <= dense.min() * (1 + 1e-12)
    res = minimize_scalar(lambda lt: _rss(T_FIT, y, math.exp(lt))[0], bounds=(math.log(10), math.log(60)),
                          method="bounded", options={"xatol": 1e-12})
    assert f.tau_ms == pytest.approx(math.exp(res.x), rel=1e-6)
    p, _ = curve_fit(lambda t, B, A, tau: B + A * np.exp(-t / tau), T_FIT, y, p0=[-70.0, 0.3, 25.0])
    assert f.tau_ms == pytest.approx(p[2], rel=1e-5)
    rss_cf = float(np.sum((y - (p[0] + p[1] * np.exp(-T_FIT / p[2]))) ** 2))
    assert f.rss <= rss_cf * (1 + 1e-12)
    A, B, rss = nw.profile_decay(T_FIT, y, 30.0)
    r2, c = _rss(T_FIT, y, 30.0)
    assert rss == pytest.approx(r2, rel=1e-10) and A == pytest.approx(c[1], rel=1e-9) and B == pytest.approx(c[0], rel=1e-12)


@pytest.mark.parametrize("y", [-70.0 + 0.001 * T_FIT, np.full(T_FIT.size, -70.0),
                               -70.0 + 0.1 * np.exp(-T_FIT / 5000.0)])
def test_O3_3_no_resolvable_decay_is_a_failed_fit(y):
    f = nw.fit_decay(T_FIT, y)
    assert not f.ok


# --- O3.4 - O3.7 the start with a fit double --------------------------------------
class PlantedFit:
    """A `fit` double returning planted (A, tau) per call, in call order, and
    recording what it was given."""
    def __init__(self, plan):
        self.plan = list(plan)
        self.calls = []

    def __call__(self, t_ms, y, cfg):
        self.calls.append((np.array(t_ms), np.array(y)))
        A, tau = self.plan[len(self.calls) - 1]
        if A is None:
            return DecayFit(float("nan"), float("nan"), float("nan"), float("nan"), False, True, "double: failed")
        return DecayFit(A, tau, -70.0, 0.0, True, False, "double")


def _sigma_ss(ss, keep):
    return float(np.median(ss.v[keep, :ss.pre_n].std(axis=1, ddof=1)))


def _plan_for(ts, sigma, tau=28.0):
    return [(0.1 * sigma * math.exp(t / tau), tau) if t > 0 else (0.05 * sigma, tau) for t in ts]


@pytest.mark.parametrize("ts,t_ss,flags,reason", [
    ((100, 120, 140, 160), 130.0, (), ""),                       # even count: mean of the two middle values
    ((10, 20, 30, 40), 80.0, ("ss_start_floor",), ""),
    ((0, 0, 0, 0), 80.0, ("ss_start_floor",), ""),              # |A| <= kappa sigma -> t = 0
    ((151, 151, 151, 151), 80.0, ("ss_start_fallback",), "short_window"),
    ((149, 149, 149, 149), 149.0, (), ""),
    ((149.999, 149.999, 300, 1), 149.999, (), ""),              # median 149.999: 50.001 ms left, kept
    ((150.001, 150.001, 300, 1), 80.0, ("ss_start_fallback",), "short_window"),
])
def test_O3_5_start_rule(ss, ts, t_ss, flags, reason):
    keep = np.ones(ss.n_windows, bool)
    sig = _sigma_ss(ss, keep)
    dbl = PlantedFit(_plan_for(ts, sig))
    st = nw.decay_start(ss, WindowConfig(), fit=dbl)
    assert st.sigma_ss_mV == pytest.approx(sig, rel=1e-15)
    # 1e-10: t = tau ln(exp(t / tau)) round trip in float64
    assert_allclose(st.t_group_ms, ts, rtol=1e-10, atol=1e-10)
    assert st.t_ss_ms == pytest.approx(t_ss, abs=1e-8)
    assert tuple(f for f in st.flags if f.startswith("ss_start")) == flags
    assert st.fallback_reason == reason


def test_O3_5_group_order_inputs_and_failed_fit(ss):
    sig = _sigma_ss(ss, np.ones(ss.n_windows, bool))
    plan = _plan_for((100, 120, 140, 160), sig)
    plan[1] = (None, None)                                   # group 2 (sweep 31, hyp) fails
    dbl = PlantedFit(plan)
    st = nw.decay_start(ss, WindowConfig(), fit=dbl)
    assert st.group_sweep.tolist() == [31, 31, 35, 35] and st.group_is_dep.tolist() == [True, False, True, False]
    assert st.group_status == ("ok", "fit", "ok", "ok")
    assert "ss_group_dropped:31:hyp:fit" in st.flags
    assert st.t_ss_ms == pytest.approx(140.0, abs=1e-9)      # median of (100, 140, 160)
    assert st.keep.all()                                     # a failed fit keeps its windows (D-106.2)
    t0, y0 = dbl.calls[0]
    assert t0[0] == pytest.approx(5.0) and t0[-1] == pytest.approx(199.98) and t0.size == 9750
    g = (ss.sweep_number == 31) & ss.polarity_is_dep
    # the double sees the group's per-sample median window on t >= 5 ms (index 500 + 250 = 750)
    assert_allclose(y0, np.median(ss.v[g][:, 750:], axis=0), rtol=0, atol=0)


def test_O3_5_no_valid_group(ss):
    st = nw.decay_start(ss, WindowConfig(), fit=PlantedFit([(None, None)] * 4))
    assert st.t_ss_ms == 80.0 and math.isnan(st.t_star_ms)
    assert st.fallback_reason == "no_valid_group" and "ss_start_fallback" in st.flags


@pytest.mark.parametrize("n_hyp_left,kept", [(5, True), (4, False)])
def test_O3_6_small_group_boundary(tmp_path, n_hyp_left, kept):
    hyp = [k for k in range(37) if k % 2 == 1]                     # 18 hyp positions per train
    drop = [(1, k) for k in hyp[n_hyp_left:]]
    d = fx.write_cell(str(tmp_path), specimen_id=20 + n_hyp_left, drop=drop)
    s = nw.load_ss_windows(d)
    keep = nw.keep_mask(s)
    g = (s.sweep_number == 35) & ~s.polarity_is_dep
    assert g.sum() == n_hyp_left and keep[g].all() == kept and keep[~g].all()
    st = nw.decay_start(s, WindowConfig(), fit=PlantedFit(_plan_for((100,) * 4, _sigma_ss(s, keep))))
    assert ("ss_group_dropped:35:hyp:few_windows" in st.flags) == (not kept)
    assert st.sigma_ss_mV == pytest.approx(_sigma_ss(s, keep), rel=1e-15)
    nwin = nw.ss_noise_windows(s, WindowConfig(), fit=PlantedFit(_plan_for((100,) * 4, _sigma_ss(s, keep))))
    assert nwin.ss.n_windows == int(keep.sum()) and nwin.X_tilde.shape[0] == int(keep.sum())


# --- O3.8 / O3.9 (TL1) and (OF1) ---------------------------------------------------
def test_O3_8_tl1_against_lstsq(ss):
    jf = 500 + 4000
    m = fx.N_W - jf
    centres = (ss.position * fx.P + jf + 0.5 * (m - 1)) * ss.dt
    ref = fx.tl1_reference(ss.v[:, jf:].mean(axis=1), centres, ss.sweep_number, ss.polarity_is_dep)
    t = nw.read_ss_tilts(ss)
    assert t.protocol == "ss" and t.sweep_number.tolist() == [31, 35] and t.flags == ()
    for s, b, se, n in zip(t.sweep_number, t.b_mV_per_s, t.se_mV_per_s, t.n_windows):
        assert b == pytest.approx(ref[int(s)][0], rel=1e-10)
        assert se == pytest.approx(ref[int(s)][1], rel=1e-8)
        assert n == int((ss.sweep_number == s).sum())
    # invariance: a constant per (sweep, polarity) group leaves the slope (TL1)
    v2 = ss.v.copy()
    for s in (31, 35):
        for p in (True, False):
            v2[(ss.sweep_number == s) & (ss.polarity_is_dep == p)] += 0.37 * s * (1 + p)
    t2 = nw.ss_sweep_tilts(v2[:, jf:], ss, jf)
    assert_allclose(t2.b_mV_per_s, t.b_mV_per_s, rtol=1e-9, atol=1e-12)


def test_O3_8_tl1_degenerate_sweeps():
    sw, b, se, n, flags = nw.tl1_fit(np.array([1.0, 2.0, 3.0]), np.array([0.0, 1.0, 2.0]), np.array([7, 7, 7]),
                                     np.array([True, False, True]))
    assert math.isnan(b[0]) and flags == ("ss_tilt_none:7",)               # n = 3 <= 3 columns
    sw, b, se, n, flags = nw.tl1_fit(np.array([1.0, 2.0, 3.5]), np.array([0.0, 1.0, 2.0]), np.array([7, 7, 7]),
                                     np.array([True, True, True]))
    assert b[0] == pytest.approx(1.25, rel=1e-12) and flags == ()          # one intercept, OLS slope


def test_O3_9_offsets(ss):
    b = {31: 0.1, 35: float("nan")}
    off = nw.tilt_offsets(ss.position, ss.sweep_number, fx.P, ss.dt, b)
    for s in (31, 35):
        g = ss.sweep_number == s
        tk = ss.position[g] * fx.P * ss.dt
        exp = b[s] * (tk - tk.mean()) if np.isfinite(b[s]) else np.zeros(g.sum())
        assert_allclose(off[g], exp, rtol=0, atol=1e-15)
    tl = nw.read_ss_tilts(ss)
    corr = nw.remove_tilt_offsets(ss, tl)
    o2 = nw.tilt_offsets(ss.position, ss.sweep_number, fx.P, ss.dt, tl.as_dict())
    assert_allclose(corr.v, ss.v - o2[:, None], rtol=0, atol=0)


# --- O3.10 / O3.11 / O3.17 the subtraction, the enlarged window, the pipeline -----------
def test_O3_10_median_subtraction():
    rng = np.random.default_rng(32)
    X = rng.standard_normal((20, 50))
    dep = np.array([True, False] * 10)
    sw = np.array([1] * 11 + [2] * 9)
    Xt, med = nw.subtract_median_response(X, dep, sw)
    for (s, p), m in med.items():
        g = (sw == s) & (dep == p)
        assert_array_equal(m, np.median(X[g], axis=0))
        assert np.max(np.abs(np.median(Xt[g], axis=0))) < 1e-14
    shift = X.copy()
    for s in (1, 2):
        for p in (True, False):
            shift[(sw == s) & (dep == p)] += rng.standard_normal(50)
    assert_allclose(nw.subtract_median_response(shift, dep, sw)[0], Xt, rtol=0, atol=1e-12)


def test_O3_11_enlarged_window_formula(ss):
    for t_ss in (80.0, 129.627, 151.7):
        X, j0 = nw.enlarged_windows(ss, t_ss)
        assert j0 == 500 + int(round(t_ss * 50.0))
        assert_array_equal(X, ss.v[:, j0:])
        assert X.shape == (ss.n_windows, fx.N_W - j0)
        # ends at the last stored sample, 0.5 ms before the next onset (200.5 ms)
        assert ss.t[-1] * 1e3 == pytest.approx(199.98)


def test_O3_17_ss_noise_windows_against_reimplementation(ss):
    cfg = WindowConfig()
    res = nw.ss_noise_windows(ss, cfg)
    keep = nw.keep_mask(ss, cfg)
    kept_v = ss.v[keep]
    assert_array_equal(res.ss.v, kept_v)
    jf = 4500
    m = fx.N_W - jf
    centres = (ss.position[keep] * fx.P + jf + 0.5 * (m - 1)) * ss.dt
    ref = fx.tl1_reference(kept_v[:, jf:].mean(axis=1), centres, ss.sweep_number[keep], ss.polarity_is_dep[keep])
    assert_allclose(res.tilts.b_mV_per_s, [ref[31][0], ref[35][0]], rtol=1e-10)
    off = np.zeros(int(keep.sum()))
    for s in (31, 35):
        g = ss.sweep_number[keep] == s
        tk = ss.position[keep][g] * fx.P * ss.dt
        off[g] = ref[s][0] * (tk - tk.mean())
    j0 = 500 + int(round(res.start.t_ss_ms * 50.0))
    assert res.j0 == j0
    Xc = (kept_v - off[:, None])[:, j0:]
    Xt = np.empty_like(Xc)
    for s in (31, 35):
        for p in (True, False):
            g = (ss.sweep_number[keep] == s) & (ss.polarity_is_dep[keep] == p)
            Xt[g] = Xc[g] - np.median(Xc[g], axis=0)
    # 1e-12 mV: the offsets enter through a slope computed by two different lstsq calls
    assert_allclose(res.X_tilde, Xt, rtol=0, atol=1e-12)
    st = nw.decay_start(ss, cfg, tilts=res.tilts)
    assert st.t_ss_ms == res.start.t_ss_ms and st.flags == res.start.flags


# --- O3.12 / O3.13 / O3.14 statistics ------------------------------------------------
def test_O3_12_window_stats_and_labels():
    rng = np.random.default_rng(33)
    X = rng.standard_normal((4, 6000)).cumsum(axis=1) * 0.01 + rng.standard_normal((4, 6000))
    st = ncs.ss_window_stats(X, 50000.0)
    for k in range(4):
        assert st[k, 0] == pytest.approx(X[k].std(ddof=1), rel=1e-14)
        ref = acf(X[k], adjusted=False, nlags=500, fft=False)[[1, 5, 50, 500]]
        assert_allclose(st[k, 1:], ref, rtol=0, atol=1e-12)
    assert ncs.STAT_LABELS[:5] == ("ss_sd", "ss_rho_0.02ms", "ss_rho_0.1ms", "ss_rho_1ms", "ss_rho_10ms")
    assert len(ncs.STAT_LABELS) == 15 and ncs.STAT_BLOCK == ("ss",) * 5 + ("ls",) * 10
    pre = rng.standard_normal(13250).cumsum() * 0.001 + rng.standard_normal(13250)
    row = ncs.ls_window_stats([pre], 50000.0)[0]
    first, last = pre[:2500], pre[-6625:]
    exp = [pre.std(ddof=1)] + list(acf(pre, adjusted=False, nlags=50, fft=False)[[1, 5, 50]]) + [first.std(ddof=1)] \
        + [last.std(ddof=1)] + list(acf(last, adjusted=False, nlags=500, fft=False)[[1, 5, 50, 500]])
    assert_allclose(row, exp, rtol=0, atol=1e-12)


def test_O3_12_statistic_vector_and_per_sweep():
    rng = np.random.default_rng(34)
    S = rng.standard_normal((9, 5))
    S[2, 4] = np.nan
    S[:, 3] = np.nan                                   # a statistic with no finite window
    L = rng.standard_normal((3, 10))
    sv = ncs.statistic_vector(S, 3565, L, 13250, fs=50000.0)
    assert sv.d == 15 and sv.labels == ncs.STAT_LABELS
    assert_allclose(sv.q[:5][[0, 1, 2, 4]], np.nanmedian(S[:, [0, 1, 2, 4]], axis=0), rtol=0, atol=0)
    assert math.isnan(sv.q[3]) and sv.K[3] == 0 and sv.K[4] == 8 and sv.K[0] == 9 and sv.K[5] == 3
    assert sv.n.tolist() == [3565] * 5 + [13250] * 4 + [2500] + [6625] * 5
    sw = np.array([4, 4, 4, 9, 9, 9, 9, 2, 2])
    ps = ncs.per_sweep_vectors(S[:, [0, 1]], sw)
    assert sorted(ps) == [2, 4, 9]
    for s in (2, 4, 9):
        assert_allclose(ps[s], np.median(S[sw == s][:, [0, 1]], axis=0), rtol=0, atol=0)


def test_O3_14_bootstrap_se():
    K, sig = 60, 0.7
    ratios = []
    for seed in range(8):
        A = np.random.default_rng(100 + seed).normal(0, sig, size=(K, 2))
        se = ncs.bootstrap_median_se(A, 2000, np.random.default_rng(seed))
        ratios.append(se / (1.2533 * sig / math.sqrt(K)))
    # 20 %: the bootstrap SE of a median is noisy at K = 60; averaged over 8 data sets
    assert np.all(np.abs(np.mean(ratios, axis=0) - 1.0) < 0.2)
    A = np.random.default_rng(1).normal(size=(30, 3))
    assert_array_equal(ncs.bootstrap_median_se(A, 500, np.random.default_rng(5)),
                       ncs.bootstrap_median_se(A, 500, np.random.default_rng(5)))
    assert not np.array_equal(ncs.bootstrap_median_se(A, 500, np.random.default_rng(5)),
                              ncs.bootstrap_median_se(A, 500, np.random.default_rng(6)))


# --- O3.15 / O3.16 LS windows and (TL2) -------------------------------------------------
def test_O3_15_ls_prewindows_and_end_windows(cell):
    ls = nw.load_ls_prewindows(cell)
    assert ls.sweep_number.tolist() == [60, 61, 63]                     # 0 pA sweep skipped
    assert ls.polarity.tolist() == ["hyp", "dep", "hyp"]
    assert ls.n_pre.tolist() == [13250] * 3 and all(p.size == 13250 for p in ls.pre)
    n = int(round((0.27 + 1.0 + 1.5) * fx.FS))
    assert ls.n_samples.tolist() == [n] * 3 and ls.end_status.tolist() == ["ok"] * 3
    assert all(e.size == 25000 for e in ls.end)
    assert_allclose(ls.end_start_s, (n - 25000) / fx.FS, rtol=1e-15)
    assert_allclose(ls.step_end_s, (13500 + 50000) / fx.FS, rtol=1e-15)
    first, last = nw.ls_segments(ls.pre[0], ls.fs)
    assert first.size == 2500 and last.size == 6625
    assert_array_equal(last, ls.pre[0][-6625:])


@pytest.mark.parametrize("tail_n,status", [(25000, "tail_short"), (25001, "ok")])
def test_O3_15_end_window_boundary(tmp_path, tail_n, status):
    d = fx.write_cell(str(tmp_path), specimen_id=40, ls=((60, -50.0),), ls_tail_s=tail_n / fx.FS)
    ls = nw.load_ls_prewindows(d)
    assert ls.end_status.tolist() == [status]      # first sample at t_off -> not usable (2.4: "at or before")


def test_O3_15_prewindow_is_half_open_for_every_onset(tmp_path):
    """[t_0, t_on - 5 ms) holds on - 250 samples at 50 kHz for any onset index.
    (Finding: the float comparison includes the sample at t_on - 5 ms for
    some onsets, e.g. 12504.)"""
    for on in (13500, 12504, 12518):
        d = fx.write_cell(str(tmp_path), specimen_id=on, ls=((60, -50.0),), ls_onset_index=on)
        ls = nw.load_ls_prewindows(d)
        assert int(ls.n_pre[0]) == on - 250, "onset %d: %d samples" % (on, ls.n_pre[0])


def test_O3_15_nonfinite_current_sample_does_not_drop_a_sweep_silently(cell, tmp_path):
    """SPEC 2.2: the pre-window of EVERY LS sweep enters. One NaN at the end of
    sweep 61's current (its voltage finite, its step intact) must not make the
    sweep vanish without a flag. (Finding: it is skipped silently.)"""
    d = _copy(cell, tmp_path, "nan_i")
    p = os.path.join(d, "ls_sweeps.npz")
    z = dict(np.load(p))
    z["i_1"][-1] = np.nan
    np.savez(p, **z)
    ls = nw.load_ls_prewindows(d)
    tilts = ncs.ls_sweep_tilts(ls)
    present = 61 in ls.sweep_number.tolist()
    flagged = any(":61" in f for f in tilts.flags)
    assert present or flagged, "sweep 61 dropped without a flag: %s" % ls.sweep_number.tolist()


def _ls_reference_tilt(v, on, fs, n_end=25000):
    pre = v[:on - 250]
    jj = np.flatnonzero(np.isfinite(pre))
    fin = np.flatnonzero(np.isfinite(v))
    last = fin[-1] + 1
    end = v[last - n_end:last]
    t_pre = jj.mean() / fs
    t_end = (last - n_end + (n_end - 1) / 2.0) / fs
    return (end.mean() - pre[jj].mean()) / (t_end - t_pre), t_end - t_pre


def test_O3_16_tl2_against_hand_computation_and_l3exc_geometry(tmp_path):
    d = fx.write_cell(str(tmp_path), specimen_id=50, ls=((60, -50.0), (61, 30.0)), ls_tail_s=6.0,
                      ls_tilts=(0.07, -0.03), ls_v_nan_tail=0)
    z = np.load(os.path.join(d, "ls_sweeps.npz"))
    ls = nw.load_ls_prewindows(d)
    t = ncs.ls_sweep_tilts(ls)
    assert t.protocol == "ls" and t.flags == () and t.n_windows.tolist() == [2, 2] and np.all(np.isnan(t.se_mV_per_s))
    for k in range(2):
        b, dtc = _ls_reference_tilt(z["v_%d" % k], 13500, fx.FS)
        assert dtc == pytest.approx(6.8875, abs=1e-12)                  # SPEC 2.4's L3_exc Delta_t_c
        assert t.b_mV_per_s[k] == pytest.approx(b, rel=1e-10)


def test_O3_16_tl2_nan_voltage_tail_and_flags(tmp_path):
    d = fx.write_cell(str(tmp_path), specimen_id=51, ls=((60, -50.0), (61, 30.0)), ls_tail_s=1.5,
                      ls_v_nan_tail=5000)
    z = np.load(os.path.join(d, "ls_sweeps.npz"))
    ls = nw.load_ls_prewindows(d)
    t = ncs.ls_sweep_tilts(ls)
    for k in range(2):
        b, _ = _ls_reference_tilt(z["v_%d" % k], 13500, fx.FS)
        assert t.b_mV_per_s[k] == pytest.approx(b, rel=1e-10)
    d2 = fx.write_cell(str(tmp_path), specimen_id=52, ls=((60, -50.0),), ls_tail_s=0.5)
    t2 = ncs.ls_sweep_tilts(nw.load_ls_prewindows(d2))
    assert math.isnan(t2.b_mV_per_s[0]) and t2.flags == ("ls_tilt_none:60:tail_short",)
    ls3 = nw.load_ls_prewindows(d)
    ls3.end[0] = ls3.end[0].copy()
    ls3.end[0][100] = np.nan
    t3 = ncs.ls_sweep_tilts(ls3)
    assert math.isnan(t3.b_mV_per_s[0]) and "ls_tilt_none:60:nonfinite" in t3.flags


def test_O3_16_validation_target_ls_tilt_error_sd():
    """SPEC 2.4: error SD of (TL2) at theta_cf on the L3_exc geometry is
    0.0045 mV/s: SD((e_end - e_pre) / Delta_t_c), from (L3) by my own sums."""
    law = fx.THETA_CF
    n1, n2, off = 13250, 25000, 338500

    def var_mean(n):
        k = np.arange(1, n)
        return (n * fx.gamma(law, fx.DT, 0) + 2 * np.sum((n - k) * fx.gamma(law, fx.DT, k))) / n ** 2

    dmin, dmax = off - (n1 - 1), off + n2 - 1
    dd = np.arange(dmin, dmax + 1)
    cnt = np.minimum(n1 - 1, off + n2 - 1 - dd) - np.maximum(0, off - dd) + 1
    cov = np.sum(np.maximum(cnt, 0) * fx.gamma(law, fx.DT, dd)) / (n1 * n2)
    sd = math.sqrt(var_mean(n1) + var_mean(n2) - 2 * cov) / 6.8875
    assert round(sd, 4) == 0.0045


# --- legacy functions kept for the dev checks ------------------------------------------
def test_legacy_subtract_mean_response_zero_mean():
    rng = np.random.default_rng(35)
    X = rng.standard_normal((12, 30))
    dep = np.array([True, False] * 6)
    Xt, md, mh = nw.subtract_mean_response(X, dep)
    assert np.max(np.abs(Xt[dep].mean(axis=0))) < 1e-15 and np.max(np.abs(Xt[~dep].mean(axis=0))) < 1e-15


def test_O3_5_zero_prepulse_sd_follows_st1_arithmetic(ss):
    """(ST1) with sigma_SS_hat = 0 and a valid fit: |A| > 0 = kappa sigma_SS, so
    t_sp = tau ln(|A| / 0) = +inf, the median is +inf and (ST2) falls back to
    80 ms ('short_window'). (Finding: decay_start raises ZeroDivisionError --
    a Python float division that its np.errstate(divide='ignore') does not
    cover.)"""
    v = ss.v.copy()
    v[:, :ss.pre_n] = -70.0
    flat = nw.SSWindows(**{**ss.__dict__, "v": v})
    dbl = lambda t, y, cfg: DecayFit(0.4, 28.0, -70.0, 0.0, True, False, "double")
    st = nw.decay_start(flat, WindowConfig(), fit=dbl)
    assert st.t_ss_ms == 80.0 and st.fallback_reason == "short_window"
