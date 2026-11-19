"""Block 4 (noise_fit.py) against SPEC 2.6 (d), (h) and the unchanged parts of 2.3.

Oracles derived from the spec:
  * u = (log sigma_f, log sigma_m, log tau_m, log sigma_s, log tau_s), natural logs.
  * bounds: log 1e-4 .. log 10 for the three SDs; log 4e-5 .. log 1e-3 for tau_m;
    log 1e-3 .. log 10 for tau_s; bounds() -> (lo (5,), hi (5,)) in the order of u.
  * diff_step = (1e-3, 3e-3, 3e-3, 1e-3, 1e-3); ftol 1e-4, xtol 1e-3, gtol 1e-8, max_nfev 15 unchanged.
  * closed-form start: lowest closed-form cost over the 4 points (0.035, 0.017, tau_m, 0.06, tau_s),
    (tau_m, tau_s) in {1e-4, 6e-4} x {0.03, 0.2}; then tau_s / 2 and 2 tau_s, clipped into the bounds.
  * dof = d_eff - 5; the fit needs d_eff >= 6.
  * at_bound names log_sigma_f, log_sigma_m, log_tau_m, log_sigma_s, log_tau_s.
  * the JSON law dict keys sigma_f_mV, sigma_m_mV, tau_m_s, sigma_s_mV, tau_s_s.
  * the simulator uses the three-component law (2.6 (b)) at the cell's layout; q_sim (21,), CRN.
"""
import math
from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import optimize

import noise_cell_stats as ncs
import noise_fit as nf
import noise_law as nl

LOG = math.log


def test_config_defaults():
    c = nf.NoiseFitConfig()
    assert tuple(c.fd_rel_step) == (1e-3, 3e-3, 3e-3, 1e-3, 1e-3)
    assert c.log_sigma_bounds == (LOG(1e-4), LOG(10))
    assert_allclose(c.log_tau_m_bounds, (LOG(4e-5), LOG(1e-3)), rtol=1e-15)
    assert_allclose(c.log_tau_s_bounds, (LOG(1e-3), LOG(10)), rtol=1e-15)
    assert not hasattr(c, "rho_bounds") and not hasattr(c, "log_tau_bounds")
    assert (c.M_binding, c.R_iter, c.R_final, c.B_boot, c.k_max, c.tol_u, c.starts_after_first, c.max_nfev,
            c.ftol, c.xtol, c.gtol) == (3000, 600, 3000, 2000, 6, 0.01, 1, 15, 1e-4, 1e-3, 1e-8)
    lo, hi = c.bounds()
    assert lo.shape == (5,) and hi.shape == (5,)
    assert_allclose(lo, [LOG(1e-4), LOG(1e-4), LOG(4e-5), LOG(1e-4), LOG(1e-3)], rtol=1e-15)
    assert_allclose(hi, [LOG(10), LOG(10), LOG(1e-3), LOG(10), LOG(10)], rtol=1e-15)


def test_u_law_round_trip_and_order():
    law = nl.NoiseLaw(0.03, 0.02, 2e-4, 0.05, 0.07)
    u = nf.law_to_u(law)
    assert_allclose(u, [LOG(0.03), LOG(0.02), LOG(2e-4), LOG(0.05), LOG(0.07)], rtol=1e-15)
    back = nf.u_to_law(u)
    assert_allclose(back.as_array(), law.as_array(), rtol=1e-14)
    with pytest.raises(ValueError):
        nf.u_to_law(u[:4])
    assert tuple(nf.U_NAMES) == ("log_sigma_f", "log_sigma_m", "log_tau_m", "log_sigma_s", "log_tau_s")


def test_closed_form_vector_independent(cell):
    lay = cell["layout"]
    law = cell["law"]
    q = nf.closed_form_vector(law, lay)
    assert q.shape == (21,)
    dt = lay.dt
    exp = [nl.closed_form_sd(law, dt, lay.m)] + list(nl.closed_form_acf(law, dt, lay.m, [1, 2, 5, 15, 50, 500]))
    exp += [nl.closed_form_sd(law, dt, lay.n_ls)] + list(nl.closed_form_acf(law, dt, lay.n_ls, [1, 2, 5, 15, 50]))
    exp += [nl.closed_form_sd(law, dt, 2500), nl.closed_form_sd(law, dt, 6625)]
    exp += list(nl.closed_form_acf(law, dt, 6625, [1, 2, 5, 15, 50, 500]))
    assert_allclose(q, exp, rtol=1e-14)


def test_closed_form_start_is_best_of_four_points(cell):
    lay = cell["layout"]
    cfg = nf.NoiseFitConfig()
    truth = nl.NoiseLaw(0.03, 0.012, 2.5e-4, 0.05, 0.06)
    q_hat = nf.closed_form_vector(truth, lay) * (1 + 0.01 * np.random.default_rng(4).standard_normal(21))
    SE = np.abs(q_hat) * 0.02 + 1e-3
    mask = np.ones(21, dtype=bool)
    got = nf.closed_form_start(q_hat, SE, mask, lay, cfg)
    lo, hi = cfg.bounds()
    points = [nl.NoiseLaw(0.035, 0.017, tm, 0.06, ts) for tm in (1e-4, 6e-4) for ts in (0.03, 0.2)]

    def fun(u):
        qq = nf.closed_form_vector(nf.u_to_law(u), lay)
        return (qq - q_hat) / SE
    best = None
    for p in points:
        u0 = np.clip(nf.law_to_u(p), lo, hi)
        r = optimize.least_squares(fun, u0, bounds=(lo, hi), method="trf")
        if best is None or r.cost < best.cost:
            best = r
    # same optimiser on the same problem: identical to roundoff (1e-10 in u)
    assert_allclose(nf.law_to_u(got), best.x, atol=1e-10)


def test_starts_clipped_into_tau_s_bounds():
    # 2.6 (d): "that start with tau_s / 2 and 2 tau_s (each clipped into the bounds)"
    cfg = nf.NoiseFitConfig()
    near_lo = nl.NoiseLaw(0.03, 0.01, 2e-4, 0.05, 1.5e-3)
    s = nf._starts_around(near_lo, 3, cfg)
    assert len(s) == 3 and s[0] == near_lo
    assert_allclose([x.tau_s for x in s], [1.5e-3, 1e-3, 3e-3], rtol=1e-12)
    near_hi = replace(near_lo, tau_s=8.0)
    assert_allclose([x.tau_s for x in nf._starts_around(near_hi, 3, cfg)], [8.0, 4.0, 10.0], rtol=1e-12)
    assert nf._starts_around(near_lo, 1, cfg) == [near_lo]


def test_solve_inner_passes_spec_steps_and_names_bounds(cell, monkeypatch):
    lay = cell["layout"]
    cfg = nf.NoiseFitConfig()
    seen = {}
    real = optimize.least_squares

    def spy(fun, x0, **kw):
        seen.update(kw)
        return real(fun, x0, **kw)
    monkeypatch.setattr(nf.optimize, "least_squares", spy)
    lo, hi = cfg.bounds()
    target = np.array([lo[0] - 1, 0.5 * (lo[1] + hi[1]), hi[2] + 1, 0.5 * (lo[3] + hi[3]), lo[4] - 1])
    res = nf.solve_inner(np.zeros(21), np.eye(5), [nl.NoiseLaw(0.03, 0.01, 2e-4, 0.05, 0.05)], lay, cfg, 1,
                         np.ones(21, bool), residual=lambda u: u - target)
    assert list(seen["diff_step"]) == [1e-3, 3e-3, 3e-3, 1e-3, 1e-3]
    assert (seen["ftol"], seen["xtol"], seen["gtol"], seen["max_nfev"], seen["method"]) == (1e-4, 1e-3, 1e-8, 15, "trf")
    assert_allclose(seen["bounds"][0], lo) and assert_allclose(seen["bounds"][1], hi)
    assert set(res.at_bound) == {"log_sigma_f", "log_tau_m", "log_tau_s"}


def test_simulator_shapes_and_mid_component(cell):
    lay = cell["layout"]
    ss, ls = nf.simulate_cell_windows(cell["law"], lay, np.random.default_rng(1))
    assert ss.shape == (lay.K, 7) and ls.shape == (lay.K_LS, 14)
    # the LS traces of the simulator = Block 2's simulator on the same stream (parts=('ls',): no SS draw first)
    out = nf.simulate_cell_windows(cell["law"], lay, np.random.default_rng(9), parts=("ls",), return_windows=True)
    u = nl.draw_units(np.random.default_rng(9), lay.K_LS, lay.n_ls)
    exp = nl.simulate_traces(cell["law"], lay.dt, u.fast, u.mid, u.slow, tilt_mV_per_s=lay.ls_tilt)
    assert_allclose(out[3], exp, rtol=0, atol=1e-12)
    # SS: a middle-only law gives a strongly correlated lag-1 (exp(-0.02/0.3) = 0.936 before demeaning);
    # a white-only law gives |rho_1| near 0. Medians over 63 windows: verdicts far from the thresholds.
    mid = nl.NoiseLaw(0.0, 0.03, 3e-4, 0.0, 0.04)
    white = nl.NoiseLaw(0.03, 0.0, 3e-4, 0.0, 0.04)
    z = replace(lay, ss_tilt=np.zeros(lay.n_sweeps), ls_tilt=np.zeros(lay.K_LS))
    s_mid, _ = nf.simulate_cell_windows(mid, z, np.random.default_rng(2), parts=("ss",))
    s_wh, _ = nf.simulate_cell_windows(white, z, np.random.default_rng(2), parts=("ss",))
    assert np.median(s_mid[:, 1]) > 0.85
    assert abs(np.median(s_wh[:, 1])) < 0.05


def test_binding_function_crn_and_shape(cell):
    lay = cell["layout"]
    cfg = nf.NoiseFitConfig(M_binding=60)
    a = nf.binding_function(cell["law"], lay, cfg, 123)
    b = nf.binding_function(cell["law"], lay, cfg, 123)
    c = nf.binding_function(cell["law"], lay, cfg, 124)
    assert a.shape == (21,) and np.all(np.isfinite(a))
    assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_fit_needs_six_statistics_and_dof(cell):
    lay, tg = cell["layout"], cell["target"]
    cfg = nf.NoiseFitConfig(M_binding=30, R_iter=10, R_final=12, B_boot=50, k_max=0, max_nfev=1)
    q5 = tg.q_hat.copy()
    keep = np.zeros(21, bool)
    keep[[0, 1, 2, 7, 8]] = True
    q5[~keep] = np.nan
    with pytest.raises(ValueError):
        nf.fit_cell(replace(tg, q_hat=q5), lay, cfg)
    q6 = tg.q_hat.copy()
    keep[13] = True
    q6[~keep] = np.nan
    res = nf.fit_cell(replace(tg, q_hat=q6), lay, cfg)
    assert res.dof == 1 and res.Sigma.shape == (6, 6) and res.u.shape == (5,)
    d = res.to_dict()
    assert list(d["law"].keys()) == ["sigma_f_mV", "sigma_m_mV", "tau_m_s", "sigma_s_mV", "tau_s_s"]


def test_swap_bounds_and_steps_through_config(cell, monkeypatch):
    # directive audit (swappability): a different tau_m range and steps given through NoiseFitConfig reach the solver
    lay = cell["layout"]
    cfg = nf.NoiseFitConfig(log_tau_m_bounds=(LOG(1e-4), LOG(2e-3)), fd_rel_step=(1e-2,) * 5)
    seen = {}
    real = optimize.least_squares

    def spy(fun, x0, **kw):
        seen.update(kw)
        return real(fun, x0, **kw)
    monkeypatch.setattr(nf.optimize, "least_squares", spy)
    res = nf.solve_inner(np.zeros(21), np.eye(5), [nl.NoiseLaw(0.03, 0.01, 5e-5, 0.05, 0.05)], lay, cfg, 1,
                         np.ones(21, bool), residual=lambda u: u - nf.law_to_u(nl.NoiseLaw(0.03, 0.01, 1.5e-3, 0.05, 0.05)))
    assert_allclose(seen["bounds"][0][2], LOG(1e-4)) and assert_allclose(seen["bounds"][1][2], LOG(2e-3))
    assert list(seen["diff_step"]) == [1e-2] * 5
    assert_allclose(res.law.tau_m, 1.5e-3, rtol=1e-6)        # inside the swapped range, outside the default


def test_inner_solve_recovers_its_own_binding_value(cell):
    # (F1) with q_hat := q_sim(theta*) on the same seed: the minimum J = 0 is at u(theta*); a trf solve from
    # a start 0.15 away in every coordinate must return near it. Tolerance 0.02 in u: twice the stop rule's 0.01
    # (the solver stops on xtol 1e-3 relative, the cost is flat to Monte Carlo noise only through CRN).
    lay = cell["layout"]
    cfg = nf.NoiseFitConfig(M_binding=40, max_nfev=15)
    truth = nl.NoiseLaw(0.035, 0.017, 3e-4, 0.055, 0.04)
    q = nf.binding_function(truth, lay, cfg, 99, return_pool=True)
    q_hat, pool_ss, pool_ls = q
    sd = np.concatenate([pool_ss.std(axis=0, ddof=1) / math.sqrt(lay.K), pool_ls.std(axis=0, ddof=1)])
    start = nf.u_to_law(nf.law_to_u(truth) + np.array([0.15, -0.15, 0.15, -0.15, 0.15]))
    res = nf.solve_inner(q_hat, np.diag(sd), [start], lay, cfg, 99, np.ones(21, bool))
    assert res.J < 1e-2, res.J
    assert_allclose(res.u, nf.law_to_u(truth), atol=0.02)
