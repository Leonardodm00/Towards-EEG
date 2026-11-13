"""Spec-derived tests, Block 4: noise_fit (SPEC 2.3, (F1), the iteration,
Block 4's entry). The cell is the tester's writer-like fixture read through
Block 3. Expected values: my own simulator loop on the same unit draws
(scipy.signal.lfilter), numpy.median / numpy.cov / scipy hdquantiles /
scipy chi2, my own closed-form moments, and residual doubles with known
minima. Reduced sizes throughout (M, R, B small).

The per-replicate draw granularity (one draw_units(rng, 1, L_s) per SS sweep
in layout order, then draw_units(rng, K_LS, n_ls)) is not in SPEC.md; it is
taken from simulate_cell_windows' docstring (see Problems in the spec).
"""
import json
import math

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import chi2
from scipy.stats.mstats import hdquantiles
from statsmodels.tsa.stattools import acf

import _fixtures as fx
import noise_cell_stats as ncs
import noise_fit as nf
import noise_windows as nw
from noise_law import NoiseLaw

LAW = NoiseLaw(*fx.THETA_CF)


@pytest.fixture(scope="module")
def cellpack(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("b4"))
    d = fx.write_cell(root, specimen_id=222, ss_tilts=(0.1, -0.05), drop=[(0, 3), (0, 4), (1, 10), (1, 11), (1, 12)],
                      ls_tilts=(0.05, -0.02, 0.0, 0.08))
    ss = nw.load_ss_windows(d)
    nwin = nw.ss_noise_windows(ss)
    ls = nw.load_ls_prewindows(d)
    layout = nf.CellLayout.from_cell(nwin, ls)
    target = nf.CellTarget.from_cell(nwin, ls, layout)
    return dict(dir=d, ss=ss, nwin=nwin, ls=ls, layout=layout, target=target)


# --- O4.1 configuration and coordinates ---------------------------------------
def test_O4_1_config_defaults_and_coordinates():
    c = nf.NoiseFitConfig()
    exp = dict(M_binding=3000, R_iter=600, R_final=3000, B_boot=2000, k_max=6, tol_u=0.01, starts_after_first=1,
               max_nfev=15, ftol=1e-4, xtol=1e-3, gtol=1e-8, fd_rel_step=(1e-3, 1e-2, 1e-3, 1e-3),
               rho_bounds=(1e-3, 0.999), chunk_windows=500, seed=20261023)
    for k, v in exp.items():
        assert getattr(c, k) == v, k
    assert c.log_sigma_bounds == pytest.approx((math.log(1e-4), math.log(10)), rel=1e-15)
    assert c.log_tau_bounds == pytest.approx((math.log(1e-4), math.log(10)), rel=1e-15)
    u = nf.law_to_u(LAW)
    assert_allclose(u, [math.log(0.038), 0.175, math.log(0.0508), math.log(0.0366)], rtol=1e-15)
    assert nf.u_to_law(u).as_array() == pytest.approx(LAW.as_array(), rel=1e-14)


# --- O4.2 the layout --------------------------------------------------------------
def test_O4_2_layout_fields(cellpack):
    L, nwin = cellpack["layout"], cellpack["nwin"]
    assert L.K == nwin.ss.n_windows == 69 and L.K_LS == 3 and L.n_ls == 13250
    assert L.j_floor == 4500 and L.j0 == nwin.j0 and L.m == fx.N_W - nwin.j0 and L.j_floor <= L.j0
    assert L.n_sweeps == 2 and L.sweep_numbers == (31, 35) and L.L_s == 36 * fx.P + fx.N_W
    assert_array_equal(L.position, nwin.ss.position)
    assert_array_equal(L.ss_tilt, [nwin.tilts.as_dict()[31], nwin.tilts.as_dict()[35]])
    lt = ncs.ls_sweep_tilts(cellpack["ls"]).as_dict()
    assert L.ls_sweep_number.tolist() == [60, 61, 63]
    assert_array_equal(L.ls_tilt, [lt[60], lt[61], lt[63]])


def test_O4_2_layout_refusals(cellpack):
    nwin, ls = cellpack["nwin"], cellpack["ls"]
    ls_t = ncs.ls_sweep_tilts(ls)
    bad = nw.SweepTilts("ss", np.array([31]), np.array([0.1]), np.array([0.0]), np.array([30]), ())
    with pytest.raises(ValueError):                                     # SS sweep 35 has no tilt
        nf.CellLayout.from_windows(nwin.ss, ls, nwin.start.t_ss_ms, bad, ls_t)
    nan_t = nw.SweepTilts("ss", np.array([31, 35]), np.array([0.1, np.nan]), np.zeros(2), np.array([30, 30]),
                          ("ss_tilt_none:35",))
    with pytest.raises(ValueError, match="ss_tilt_none:35"):
        nf.CellLayout.from_windows(nwin.ss, ls, nwin.start.t_ss_ms, nan_t, ls_t)
    import copy
    ls2 = copy.deepcopy(ls)
    ls2.pre[1] = ls2.pre[1][:-1]
    with pytest.raises(ValueError):                                     # unequal LS pre-window lengths
        nf.CellLayout.from_windows(nwin.ss, ls2, nwin.start.t_ss_ms, nwin.tilts, ls_t)


def test_O4_2_unreadable_ls_tilt_refused(tmp_path):
    d = fx.write_cell(str(tmp_path), specimen_id=223, ls=((60, -50.0), (61, 30.0)), ls_tail_s=0.5)
    nwin = nw.ss_noise_windows(nw.load_ss_windows(d))
    with pytest.raises(ValueError, match="tail_short"):
        nf.CellLayout.from_cell(nwin, nw.load_ls_prewindows(d))


# --- O4.3 the simulator against an independent loop --------------------------------
def _independent_replicate(law, L, seed):
    rng = np.random.default_rng(seed)
    P, j0, jf, n_w, Ls, dt = L.period_samples, L.j0, L.j_floor, L.n_w, L.L_s, L.dt
    Xf = np.empty((L.K, n_w - jf))
    tc = (np.arange(Ls) - (Ls - 1) / 2.0) * dt
    for s_idx in range(L.n_sweeps):
        xf = rng.standard_normal((1, Ls)); xs = rng.standard_normal((1, Ls)); rng.standard_normal(1); rng.standard_normal(1)
        tr = fx.ar1_from_units(xf, law.sigma_f, law.rho_f)[0]
        eta = law.sigma_s * math.sqrt(-math.expm1(-2 * dt / law.tau_s)) * xs[0]
        eta[0] = law.sigma_s * xs[0, 0]
        from scipy.signal import lfilter
        tr = tr + lfilter([1.0], [1.0, -math.exp(-dt / law.tau_s)], eta) + L.ss_tilt[s_idx] * tc
        for k in np.flatnonzero(L.sweep_index == s_idx):
            p = int(L.position[k])
            Xf[k] = tr[p * P + jf:p * P + n_w]
    m_f = n_w - jf
    centres = (L.position * P + jf + 0.5 * (m_f - 1)) * dt
    ref = fx.tl1_reference(Xf.mean(axis=1), centres, L.sweep_number, L.polarity_is_dep)
    off = np.zeros(L.K)
    for s in L.sweep_numbers:
        g = L.sweep_number == s
        tk = L.position[g] * P * dt
        off[g] = ref[s][0] * (tk - tk.mean())
    X = Xf[:, j0 - jf:] - off[:, None]
    Xt = np.empty_like(X)
    for s in L.sweep_numbers:
        for pol in (True, False):
            g = (L.sweep_number == s) & (L.polarity_is_dep == pol)
            if g.any():
                Xt[g] = X[g] - np.median(X[g], axis=0)
    xf = rng.standard_normal((L.K_LS, L.n_ls)); xs = rng.standard_normal((L.K_LS, L.n_ls))
    tcl = (np.arange(L.n_ls) - (L.n_ls - 1) / 2.0) * dt
    from scipy.signal import lfilter
    eta = law.sigma_s * math.sqrt(-math.expm1(-2 * dt / law.tau_s)) * xs
    eta[:, 0] = law.sigma_s * xs[:, 0]
    lsx = fx.ar1_from_units(xf, law.sigma_f, law.rho_f) + lfilter([1.0], [1.0, -math.exp(-dt / law.tau_s)], eta, axis=1) \
        + L.ls_tilt[:, None] * tcl[None, :]
    return Xt, lsx, {s: ref[s][0] for s in L.sweep_numbers}


def _stats_ref(X, lags):
    out = np.empty((X.shape[0], 1 + len(lags)))
    for k in range(X.shape[0]):
        out[k, 0] = X[k].std(ddof=1)
        out[k, 1:] = acf(X[k], adjusted=False, nlags=max(lags), fft=False)[lags]
    return out


def test_O4_3_simulate_cell_windows_equals_independent_loop(cellpack):
    L = cellpack["layout"]
    ss_st, ls_st, Xt, lsx, tilts = nf.simulate_cell_windows(LAW, L, np.random.default_rng(77), return_windows=True,
                                                             return_tilts=True)
    Xr, lsr, tr = _independent_replicate(LAW, L, 77)
    assert ss_st.shape == (69, 5) and ls_st.shape == (3, 10) and Xt.shape == (69, L.m) and lsx.shape == (3, 13250)
    # 1e-12 mV: the same draws through two lfilter calls and two lstsq slopes
    assert_allclose(Xt, Xr, rtol=0, atol=1e-12)
    assert_allclose(lsx, lsr, rtol=0, atol=1e-12)
    for s in L.sweep_numbers:
        assert tilts[s] == pytest.approx(tr[s], rel=1e-9)
    assert_allclose(ss_st, _stats_ref(Xr, [1, 5, 50, 500]), rtol=0, atol=1e-10)
    assert_allclose(ls_st, ncs.ls_window_stats(list(lsr), fx.FS), rtol=0, atol=1e-10)


# --- O4.5 / O4.6 binding function and HD median -----------------------------------
def test_O4_5_binding_function_crn_pools_and_hd(cellpack):
    L = cellpack["layout"]
    cfg = nf.NoiseFitConfig(M_binding=40)
    q1, pss, pls = nf.binding_function(LAW, L, cfg, 5, return_pool=True)
    q2 = nf.binding_function(LAW, L, cfg, 5)
    assert_array_equal(q1, q2)
    assert not np.array_equal(q1, nf.binding_function(LAW, L, cfg, 6))
    assert pss.shape == (math.ceil(40 / 69) * 69, 5) and pls.shape == (math.ceil(40 / 3) * 3, 10)
    ref = [float(hdquantiles(c, prob=[0.5])[0]) for c in pss.T] + [float(hdquantiles(c, prob=[0.5])[0]) for c in pls.T]
    assert_allclose(q1, ref, rtol=1e-12, atol=0)
    qs = nf.binding_function(LAW, L, cfg, 5, which="ss")
    ql = nf.binding_function(LAW, L, cfg, 5, which="ls")
    assert_array_equal(qs[:5], q1[:5]) and np.all(np.isnan(qs[5:]))
    assert_array_equal(ql[5:], q1[5:]) and np.all(np.isnan(ql[:5]))


def test_O4_5_ls_pool_cycles_the_tilts(cellpack):
    """F5's oracle: at a noise-free law the LS pool row i is the pure ramp of
    LS sweep i mod K_LS, SD |b| dt sqrt(n (n + 1) / 12)."""
    L = cellpack["layout"]
    law0 = NoiseLaw(0.0, 0.5, 0.0, 0.04)
    _q, _pss, pls = nf.binding_function(law0, L, nf.NoiseFitConfig(M_binding=7), 3, which="ls", return_pool=True)
    n = L.n_ls
    for i in range(pls.shape[0]):
        b = L.ls_tilt[i % 3]
        assert pls[i, 0] == pytest.approx(abs(b) * L.dt * math.sqrt(n * (n + 1) / 12.0), rel=1e-12)


def test_O4_5_note_q_sim_depends_on_chunk_windows(cellpack):
    """NOTE: chunk_windows, a memory knob, changes the LS draws and hence
    q_sim for a given seed (the CRN hold within one config)."""
    L = cellpack["layout"]
    a = nf.binding_function(LAW, L, nf.NoiseFitConfig(M_binding=30), 9, which="ls")
    b = nf.binding_function(LAW, L, nf.NoiseFitConfig(M_binding=30, chunk_windows=7), 9, which="ls")
    assert not np.array_equal(a, b)


def test_O4_6_hd_median():
    rng = np.random.default_rng(41)
    A = rng.standard_normal((200, 3))
    A[5, 1] = np.nan
    got = nf.hd_median(A)
    assert got[0] == pytest.approx(float(hdquantiles(A[:, 0], prob=[0.5])[0]), rel=1e-12)
    assert got[1] == pytest.approx(float(hdquantiles(A[np.isfinite(A[:, 1]), 1], prob=[0.5])[0]), rel=1e-12)
    assert nf.hd_median(np.array([[2.5, np.nan]]))[0] == 2.5 and math.isnan(nf.hd_median(np.array([[2.5, np.nan]]))[1])


# --- O4.7 / O4.8 Sigma_hat and the closed forms -----------------------------------------
def test_O4_7_simulated_covariance(cellpack):
    L = cellpack["layout"]
    sr = nf.simulated_covariance(LAW, L, 6, np.random.default_rng(8), per_sweep=True)
    assert sr.vectors.shape == (6, 15) and sr.Sigma.shape == (15, 15)
    assert_allclose(sr.Sigma, np.cov(sr.vectors, rowvar=False, ddof=1), rtol=0, atol=0)
    assert sorted(sr.per_sweep) == [31, 35, 60, 61, 63]
    assert sr.per_sweep[31].shape == (6, 5) and sr.per_sweep[60].shape == (6, 10)
    rng = np.random.default_rng(8)
    for r in range(6):
        s_st, l_st = nf.simulate_cell_windows(LAW, L, rng)
        assert_array_equal(sr.per_sweep[35][r], np.median(s_st[L.sweep_number == 35], axis=0))
        assert_array_equal(sr.per_sweep[61][r], l_st[1])
        assert_allclose(sr.vectors[r], np.concatenate([np.median(s_st, axis=0), np.median(l_st, axis=0)]), rtol=0, atol=0)
    sr2 = nf.simulated_covariance(LAW, L, 6, np.random.default_rng(8))
    assert_array_equal(sr2.Sigma, sr.Sigma)


def test_O4_8_closed_form_vector(cellpack):
    L = cellpack["layout"]
    q = nf.closed_form_vector(LAW, L)
    th = fx.THETA_CF
    exp = []
    for n, lags in ((L.m, [1, 5, 50, 500]), (13250, [1, 5, 50]), (2500, []), (6625, [1, 5, 50, 500])):
        _, s0, sl = fx.demeaned_moments(th, fx.DT, n, lags or [1])
        exp += [math.sqrt(s0 / (n - 1))] + (list(sl / s0) if lags else [])
    assert_allclose(q, exp, rtol=1e-9)


def test_O4_8_closed_form_start_recovers_theta(cellpack):
    L = cellpack["layout"]
    star = NoiseLaw(0.05, 0.3, 0.07, 0.025)
    q = nf.closed_form_vector(star, L)
    SE = 0.01 * np.abs(q)
    got = nf.closed_form_start(q, SE, np.ones(15, bool), L, nf.NoiseFitConfig())
    assert_allclose(nf.law_to_u(got), nf.law_to_u(star), rtol=0, atol=1e-4)


# --- O4.9 the inner solve with residual doubles -------------------------------------------
def test_O4_9_solve_inner_bookkeeping(cellpack):
    L = cellpack["layout"]
    cfg = nf.NoiseFitConfig(max_nfev=50)
    x1, x2 = math.log(0.02), math.log(0.08)

    def res(u):     # local minimum near x1 (J > 0), global at x2 (J = 0) in the 4th coordinate
        return np.array([u[0] - math.log(0.05), u[1] - 0.3, u[2] - math.log(0.06),
                         10 * (u[3] - x1) * (u[3] - x2), 0.3 * (u[3] - x2)])

    start = NoiseLaw(0.05, 0.3, 0.06, 0.04)
    r = nf.solve_inner(None, None, [start, NoiseLaw(0.05, 0.3, 0.06, 0.02), NoiseLaw(0.05, 0.3, 0.06, 0.08)],
                       L, cfg, 0, None, residual=res)
    Js = [J for _u0, J in r.starts]
    assert r.J == min(Js) and r.start_index == int(np.argmin(Js))
    assert r.J == pytest.approx(float(np.sum(res(r.u) ** 2)), rel=1e-12, abs=1e-30)
    assert abs(r.u[3] - x2) < 1e-4 and r.at_bound == ()

    def res_out(u):  # optimum beyond the upper rho bound -> at_bound
        return np.array([u[0] - math.log(0.05), u[1] - 1.5, u[2] - math.log(0.06), u[3] - math.log(0.04)])
    r2 = nf.solve_inner(None, None, [start], L, cfg, 0, None, residual=res_out)
    assert r2.at_bound == ("rho_f",) and r2.u[1] == pytest.approx(0.999, abs=1e-6)


# --- O4.10 the iteration ---------------------------------------------------------------------
SMALL = dict(M_binding=60, R_iter=40, R_final=60, B_boot=400, k_max=2, max_nfev=6)


@pytest.mark.slow
def test_O4_10_fit_cell_reduced(cellpack):
    L, T = cellpack["layout"], cellpack["target"]
    cfg = nf.NoiseFitConfig(**SMALL)
    res = nf.fit_cell(T, L, cfg)
    seed_b = int(np.random.SeedSequence([cfg.seed, 222]).spawn(3)[0].generate_state(1, dtype=np.uint64)[0])
    assert_array_equal(res.q_sim, nf.binding_function(res.law, L, cfg, seed_b))
    mask = np.isfinite(T.q_hat)
    r = (res.q_sim - T.q_hat)[mask]
    J = float(r @ np.linalg.solve(res.Sigma, r))
    assert res.J == pytest.approx(J, rel=1e-8)
    assert res.dof == 11 and res.p_value == pytest.approx(chi2.sf(res.J, 11), rel=1e-12)
    sd = np.sqrt(np.diag(res.Sigma))
    assert res.kappa == pytest.approx(np.linalg.cond(res.Sigma / np.outer(sd, sd)), rel=1e-10)
    h = res.history
    assert h[0]["weight"] == "W0" and h[-1]["weight"] == "final" and len(h) == res.n_solves
    sig = [x for x in h if x["weight"].startswith("Sigma")]
    du = [float(np.max(np.abs(sig[0]["u"] - h[0]["u"])))] + \
         [float(np.max(np.abs(sig[i]["u"] - sig[i - 1]["u"]))) for i in range(1, len(sig))]
    if res.converged:
        assert du[-1] <= cfg.tol_u and all(d > cfg.tol_u for d in du[:-1]) and "no_plateau" not in res.flags
    else:
        assert len(sig) == cfg.k_max and all(d > cfg.tol_u for d in du) and "no_plateau" in res.flags
    assert sorted(res.per_sweep_cov) == [31, 35, 60, 61, 63]
    assert res.per_sweep_cov[31].shape == (5, 5) and res.per_sweep_cov[60].shape == (10, 10)
    assert res.ss_tilt == {31: float(L.ss_tilt[0]), 35: float(L.ss_tilt[1])}
    s = json.dumps(res.to_dict(), allow_nan=False)
    assert json.loads(s)["specimen_id"] == 222


@pytest.mark.slow
def test_O4_10_masks_and_no_plateau(cellpack):
    L, T = cellpack["layout"], cellpack["target"]
    q = T.q_hat.copy()
    q[[4, 14]] = np.nan
    T2 = nf.CellTarget(q_hat=q, per_window_ss=T.per_window_ss, per_window_ls=T.per_window_ls)
    cfg = nf.NoiseFitConfig(**dict(SMALL, k_max=0, max_nfev=1))
    res = nf.fit_cell(T2, L, cfg)
    assert res.Sigma.shape == (13, 13) and res.dof == 9 and res.n_solves == 2
    assert res.stats_dropped == ("ss_rho_10ms", "ls_last_rho_10ms") and "no_plateau" in res.flags
    assert np.isnan(res.SE[[4, 14]]).all() and np.isfinite(np.delete(res.SE, [4, 14])).all()
    assert any(f.startswith("stats_dropped:") for f in res.flags)


def _cell_with_ls_rate(tmp_path, fs_ls):
    """Tester's cell whose LS sweeps are genuinely sampled at fs_ls (SS at 50 kHz)."""
    import os
    d = fx.write_cell(str(tmp_path), specimen_id=224, ls=((60, -50.0), (61, 30.0)))
    rng = np.random.default_rng(5)
    arr = {}
    for j, amp in enumerate((-50.0, 30.0)):
        on, ns, nt = int(0.27 * fs_ls), int(1.0 * fs_ls), int(1.5 * fs_ls)
        n = on + ns + nt
        sf, rf, s_s, ts = fx.THETA_CF
        x = fx.ar1_from_units(rng.standard_normal((1, n)), sf, rf)[0] \
            + fx.ar1_from_units(rng.standard_normal((1, n)), s_s, math.exp(-1 / fs_ls / ts))[0] - 70.3
        i = np.zeros(n)
        i[on:on + ns] = amp
        arr["v_%d" % j], arr["i_%d" % j] = x, i
    arr["index"] = np.arange(2)
    np.savez(os.path.join(d, "ls_sweeps.npz"), **arr)
    p = os.path.join(d, "metadata.json")
    m = json.load(open(p))
    for e in m["ls_sweeps"]:
        e["sampling_rate_Hz"] = fs_ls
    json.dump(m, open(p, "w"))
    return d


def test_O4_2_ls_rate_differs_from_ss_rate_is_not_silent(tmp_path):
    """Boundary Block 3 -> Block 4: CellTarget computes the LS statistics at
    ls.fs, while CellLayout takes fs and dt from the SS windows and the
    simulator uses them for the LS pre-windows. With LS sweeps at 100 kHz and
    SS at 50 kHz the two sides compare different lags and window durations;
    the layout must refuse such a cell (or simulate the LS part at ls.fs).
    (Finding: accepted silently; the simulated LS window lasts 530 ms instead
    of 265 ms and its lags are 1, 5, 50 samples instead of 2, 10, 100.)"""
    d = _cell_with_ls_rate(tmp_path, 100000.0)
    nwin = nw.ss_noise_windows(nw.load_ss_windows(d))
    ls = nw.load_ls_prewindows(d)
    assert ls.fs == 100000.0 and nwin.ss.fs == 50000.0
    try:
        L = nf.CellLayout.from_cell(nwin, ls)
    except ValueError:
        return
    assert L.n_ls * L.dt == pytest.approx(ls.n_pre[0] / ls.fs), "LS pre-window simulated at the SS sampling interval"


def test_O4_7_ls_sweep_numbered_like_an_ss_sweep(tmp_path):
    """per_sweep (and CellFitResult.per_sweep_cov) are keyed by the bare sweep
    number for both protocols. A cell whose LS sweep carries the number of an
    SS sweep is accepted by CellLayout; the per-sweep covariance run must then
    keep the two apart (Block 9 keys twins and seeds by (protocol, sweep)).
    (Finding: np.vstack raises on the mixed (5,) and (10,) rows -- in fit_cell
    this happens only at the final Sigma_hat step.)"""
    d = fx.write_cell(str(tmp_path), specimen_id=225, sweeps=(31, 35), ls=((31, -50.0), (61, 30.0)))
    nwin = nw.ss_noise_windows(nw.load_ss_windows(d))
    L = nf.CellLayout.from_cell(nwin, nw.load_ls_prewindows(d))
    sr = nf.simulated_covariance(LAW, L, 3, np.random.default_rng(0), per_sweep=True)
    assert len(sr.per_sweep) == 4
