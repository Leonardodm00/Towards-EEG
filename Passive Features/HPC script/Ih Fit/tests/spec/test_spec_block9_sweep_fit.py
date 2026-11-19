"""Block 9 (noise_sweep_fit.py) against SPEC 2.6 (e), (g), (h) and the unchanged parts of 2.5.

Oracles derived from the spec:
  * u_s = (log sigma_f, log sigma_m, log sigma_s); theta(u_s) holds tau_m and tau_s at the cell's.
  * diff_step (1e-3, 3e-3, 1e-3); the three SD bounds; Block 4's ftol, xtol, gtol, max_nfev.
  * failure: status <= 0 -> no_convergence; a coordinate at a bound -> at_bound:<names>; then
    sigma_f_hat(c, s) < 0.5 sigma_f_hat(c) -> degenerate:sigma_f (strict); a failed sweep is not
    bootstrapped; its twin carries theta_hat(c) whole; raw, v, w columns NaN.
  * one bootstrap gives v_m and v_s, ddof 1, over the same B_ok refits.
  * the pooling (P3), (P4), (D-076.1) applied twice, independently, per protocol.
  * table_rows() and cell_columns() exactly the columns of 2.6 (g), in order.
  * SweepStageConfig defaults: B_v 50, M_refit 1000, frac 0.1, min_stats 4, min_sigma_f_ratio 0.5.
"""
import math
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import optimize

import noise_fit as nf
import noise_law as nl
import noise_sweep_fit as nsf

SWEEP_COLS = ["specimen_id", "noise_sweep_protocol", "noise_sweep_number", "noise_tilt_mV_per_s",
              "noise_sweep_sigma_f_mV", "noise_sweep_sigma_m_raw_mV", "noise_sweep_sigma_m_mV",
              "noise_sweep_v_log_sigma_m", "noise_sweep_w_m", "noise_sweep_sigma_s_raw_mV",
              "noise_sweep_sigma_s_mV", "noise_sweep_v_log_sigma_s", "noise_sweep_w", "noise_sweep_flags"]
CELL_COLS = ["specimen_id", "noise_cell_tau2_log_sigma_m_ss", "noise_cell_mu_log_sigma_m_ss",
             "noise_cell_tau2_log_sigma_m_ls", "noise_cell_mu_log_sigma_m_ls", "noise_cell_tau2_log_sigma_s_ss",
             "noise_cell_mu_log_sigma_s_ss", "noise_cell_tau2_log_sigma_s_ls", "noise_cell_mu_log_sigma_s_ls"]
CELL_LAW = nl.NoiseLaw(0.035, 0.017, 3e-4, 0.055, 0.04)


def dl_reference(y, v):
    """DerSimonian-Laird written out from (P3)-(P4) and (D-076.1)."""
    y, v = np.asarray(y, float), np.asarray(v, float)
    w = 1 / v
    yfe = np.sum(w * y) / np.sum(w)
    Q = np.sum(w * (y - yfe) ** 2)
    raw = (Q - (len(y) - 1)) / (np.sum(w) - np.sum(w ** 2) / np.sum(w))
    t2 = max(0.0, raw)
    ws = 1 / (v + t2)
    mu = np.sum(ws * y) / np.sum(ws)
    lam = t2 / (t2 + v)
    return raw, t2, mu, lam, mu + lam * (y - mu)


def test_config_defaults():
    c = nsf.SweepStageConfig()
    assert (c.B_v, c.M_refit, c.max_failed_refit_frac, c.min_stats, c.min_sigma_f_ratio) == (50, 1000, 0.1, 4, 0.5)
    assert tuple(nsf.U_NAMES) == ("log_sigma_f", "log_sigma_m", "log_sigma_s")


def test_pool_protocol_against_written_out_dl():
    rng = np.random.default_rng(7)
    for k in range(2, 9):
        for _ in range(30):
            y = rng.normal(-3, 0.3, k)
            v = rng.uniform(0.002, 0.08, k)
            p = nsf.pool_protocol(y, v)
            raw, t2, mu, lam, ys = dl_reference(y, v)
            # sums of k terms: roundoff ~1e-15 of the scale; 1e-12 ample
            assert_allclose([p.tau2_raw, p.tau2, p.mu], [raw, t2, mu], rtol=1e-11, atol=1e-12)
            assert_allclose(p.lam, lam, atol=1e-12)
            assert_allclose(p.y_shrunk, ys, atol=1e-12)
    p1 = nsf.pool_protocol([-3.0], [0.01])
    assert p1.k == 1 and p1.mu == -3.0 and math.isnan(p1.tau2) and np.isnan(p1.lam[0]) and p1.y_shrunk[0] == -3.0
    p0 = nsf.pool_protocol([], [])
    assert p0.k == 0 and math.isnan(p0.mu) and math.isnan(p0.tau2)


def _row(pr, s, sf, lm, vm, ls, vs, failed=False, reason=""):
    return {"specimen_id": 1, "protocol": pr, "sweep": s, "tilt_mV_per_s": 0.01 * s, "failed": failed,
            "reason": reason, "sigma_f_mV": sf, "sigma_m_mV": math.exp(lm), "sigma_s_mV": math.exp(ls),
            "log_sigma_m": lm, "log_sigma_s": ls, "v_log_sigma_m": vm, "v_log_sigma_s": vs,
            "flags": (["sweep_fit_failed:%s:%d:%s" % (pr, s, reason)] if failed else []), "seconds": 0.0}


def test_pool_cell_twice_independent_twins_and_columns():
    rows = [_row("ss", 92, 0.034, -4.0, 0.02, -2.9, 0.01), _row("ss", 93, 0.036, -4.3, 0.05, -2.7, 0.03),
            _row("ss", 94, 0.035, -4.1, 0.04, -3.1, 0.005),
            _row("ls", 54, 0.033, -3.9, 0.03, -3.0, 0.02), _row("ls", 55, 0.031, -4.6, 0.06, -2.5, 0.04),
            _row("ls", 56, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"),
                 failed=True, reason="degenerate:sigma_f")]
    res = nsf.pool_cell(rows, CELL_LAW, 1)
    for pr, idx in (("ss", [0, 1, 2]), ("ls", [3, 4])):
        for co, ky, kv in (("sigma_m", "log_sigma_m", "v_log_sigma_m"), ("sigma_s", "log_sigma_s", "v_log_sigma_s")):
            raw, t2, mu, lam, ys = dl_reference([rows[i][ky] for i in idx], [rows[i][kv] for i in idx])
            p = res.pools[pr][co]
            assert_allclose([p.tau2_raw, p.tau2, p.mu], [raw, t2, mu], rtol=1e-11, atol=1e-12)
            assert_allclose(p.y_shrunk, ys, atol=1e-12)
        for j, i in enumerate(idx):
            tw = res.twin_laws[(pr, rows[i]["sweep"])]
            exp = (rows[i]["sigma_f_mV"], math.exp(res.pools[pr]["sigma_m"].y_shrunk[j]), CELL_LAW.tau_m,
                   math.exp(res.pools[pr]["sigma_s"].y_shrunk[j]), CELL_LAW.tau_s)
            assert_allclose(tw.as_array(), exp, rtol=1e-14)
    assert res.twin_laws[("ls", 56)] == CELL_LAW
    tr = res.table_rows()
    assert all(list(r.keys()) == SWEEP_COLS for r in tr)
    failed = [r for r in tr if r["noise_sweep_number"] == 56][0]
    for k in ("noise_sweep_sigma_m_raw_mV", "noise_sweep_v_log_sigma_m", "noise_sweep_w_m",
              "noise_sweep_sigma_s_raw_mV", "noise_sweep_v_log_sigma_s", "noise_sweep_w"):
        assert math.isnan(failed[k]), k
    assert (failed["noise_sweep_sigma_f_mV"], failed["noise_sweep_sigma_m_mV"], failed["noise_sweep_sigma_s_mV"]) == (
        CELL_LAW.sigma_f, CELL_LAW.sigma_m, CELL_LAW.sigma_s)
    ok = [r for r in tr if r["noise_sweep_number"] == 93][0]
    assert ok["noise_sweep_protocol"] == "SS"
    assert_allclose(ok["noise_sweep_w_m"], res.pools["ss"]["sigma_m"].lam[1])
    assert_allclose(ok["noise_sweep_w"], res.pools["ss"]["sigma_s"].lam[1])
    cc = res.cell_columns()
    assert list(cc.keys()) == CELL_COLS
    assert cc["noise_cell_tau2_log_sigma_m_ls"] == res.pools["ls"]["sigma_m"].tau2
    assert cc["noise_cell_mu_log_sigma_s_ss"] == res.pools["ss"]["sigma_s"].mu


def test_k1_rule_for_both_coordinates():
    rows = [_row("ss", 92, 0.034, -4.0, 0.02, -2.9, 0.01)]
    res = nsf.pool_cell(rows, CELL_LAW, 1)
    tw = res.twin_laws[("ss", 92)]
    assert_allclose(tw.as_array(), [0.034, math.exp(-4.0), CELL_LAW.tau_m, math.exp(-2.9), CELL_LAW.tau_s], rtol=1e-14)
    assert math.isnan(res.pools["ss"]["sigma_m"].tau2) and math.isnan(res.pools["ls"]["sigma_s"].mu)


# ------------------------------------------------------------- with the fixture cell
def _stub_cell_fit(lay, R=40):
    sig = nf.simulated_covariance(CELL_LAW, lay, R, np.random.default_rng(3), per_sweep=True)
    # per_sweep holds each sweep's (R, N) vectors; per_sweep_cov is their covariance (ddof 1), 2.5 (c)
    cov = {s: np.cov(v, rowvar=False, ddof=1) for s, v in sig.per_sweep.items()}
    return SimpleNamespace(law=CELL_LAW, per_sweep_cov=cov, specimen_id=lay.specimen_id,
                           config=nf.NoiseFitConfig(M_binding=40))


@pytest.fixture(scope="module")
def stub(cell):
    return _stub_cell_fit(cell["layout"])


def test_fit_sweep_holds_taus_and_passes_spec_steps(cell, stub, monkeypatch):
    lay, tg = cell["layout"], cell["target"]
    cfg = nf.NoiseFitConfig(M_binding=40, max_nfev=3)
    s = lay.sweep_numbers[0]
    sub = nsf.sweep_layout(lay, "ss", s)
    q, mask = nsf.sweep_target(tg, lay, "ss", s)
    seen = {}
    real = optimize.least_squares

    def spy(fun, x0, **kw):
        seen.update(kw)
        seen["x0"] = np.array(x0)
        return real(fun, x0, **kw)
    monkeypatch.setattr(nsf.optimize, "least_squares", spy)
    start = nl.NoiseLaw(0.03, 0.02, 2.2e-4, 0.05, 0.07)
    f = nsf.fit_sweep(q, mask, stub.per_sweep_cov[s], sub, start, cfg, 11)
    assert_allclose(seen["diff_step"], [1e-3, 3e-3, 1e-3])
    assert_allclose(seen["bounds"][0], [math.log(1e-4)] * 3) and assert_allclose(seen["bounds"][1], [math.log(10)] * 3)
    assert_allclose(seen["x0"], [math.log(0.03), math.log(0.02), math.log(0.05)], rtol=1e-14)
    assert (seen["ftol"], seen["xtol"], seen["gtol"], seen["max_nfev"]) == (1e-4, 1e-3, 1e-8, 3)
    assert f.law.tau_m == 2.2e-4 and f.law.tau_s == 0.07
    assert_allclose([math.log(f.law.sigma_f), math.log(f.law.sigma_m), math.log(f.law.sigma_s)], f.u, rtol=1e-14)
    assert f.u.shape == (3,) and f.dof == f.d - 3 and f.d == 7


def _fake_fit(sf, failed=False, reason=""):
    def fake(q, mask, Sigma, sub, law_start, cfg, seed):
        pr, s = nsf._sweep_of(sub)
        u = np.array([math.log(sf), math.log(0.015), math.log(0.05)])
        return nsf.SweepFit(protocol=pr, sweep=s, u=u, law=nsf._law_of(u, law_start.tau_m, law_start.tau_s), J=1.0,
                            d=int(mask.sum()), dof=int(mask.sum()) - 3, n_calls=1, status=1, at_bound=(),
                            failed=failed, reason=reason)
    return fake


def test_degenerate_sigma_f_rule(cell, stub, monkeypatch):
    lay, tg = cell["layout"], cell["target"]
    cfg = nf.NoiseFitConfig(M_binding=20)
    st = nsf.SweepStageConfig(B_v=3, M_refit=20)
    calls = []
    monkeypatch.setattr(nsf, "bootstrap_v", lambda *a, **k: calls.append(1) or nsf.SweepBootstrap(
        0.01, 0.02, np.zeros(3), np.zeros(3), 3, 3, ()))
    s = lay.sweep_numbers[0]
    monkeypatch.setattr(nsf, "fit_sweep", _fake_fit(0.5 * CELL_LAW.sigma_f * 0.999))
    row = nsf.run_sweep(stub, tg, lay, cfg, st, "ss", s)
    assert row["failed"] and row["reason"] == "degenerate:sigma_f" and not calls
    assert "sweep_fit_failed:ss:%d:degenerate:sigma_f" % s in row["flags"]
    monkeypatch.setattr(nsf, "fit_sweep", _fake_fit(0.5 * CELL_LAW.sigma_f))     # strict '<': exactly half passes
    row = nsf.run_sweep(stub, tg, lay, cfg, st, "ss", s)
    assert not row["failed"] and calls
    # status / at_bound come first: a sweep failing at a bound keeps that reason
    monkeypatch.setattr(nsf, "fit_sweep", _fake_fit(0.001, True, "at_bound:log_sigma_f"))
    row = nsf.run_sweep(stub, tg, lay, cfg, st, "ss", s)
    assert row["reason"] == "at_bound:log_sigma_f"


def test_bootstrap_one_set_of_refits_two_variances(cell, stub):
    lay, tg = cell["layout"], cell["target"]
    cfg = nf.NoiseFitConfig(M_binding=20)
    st = nsf.SweepStageConfig(B_v=12, M_refit=20)
    s = int(lay.ls_sweep_number[0])
    sub = nsf.sweep_layout(lay, "ls", s)
    q, mask = nsf.sweep_target(tg, lay, "ls", s)
    fit = _fake_fit(0.035)(q, mask, None, sub, CELL_LAW, cfg, 0)
    seen = []

    def refit(qb, mb, Sigma, sub_, law_start, cfg_, seed):
        b = len(seen)
        seen.append((cfg_.M_binding, seed, law_start))
        u = np.array([math.log(0.035), -4.0 + 0.1 * b, -3.0 - 0.05 * b ** 1.5])
        fail = (b == 4)
        return nsf.SweepFit("ls", s, u, nsf._law_of(u, 1e-4, 1.0), 0.0, 14, 11, 1, 1, (), fail,
                            "no_convergence" if fail else "")
    bs = nsf.bootstrap_v(fit, mask, stub.per_sweep_cov[s], sub, CELL_LAW, cfg, st, 77, 88, refit=refit)
    ym = np.array([-4.0 + 0.1 * b for b in range(12)])
    ys = np.array([-3.0 - 0.05 * b ** 1.5 for b in range(12)])
    ok = np.arange(12) != 4
    assert bs.B == 12 and bs.B_ok == 11
    assert np.isnan(bs.y_m[4]) and np.isnan(bs.y_s[4])
    assert_allclose(bs.v_m, np.var(ym[ok], ddof=1), rtol=1e-13)
    assert_allclose(bs.v_s, np.var(ys[ok], ddof=1), rtol=1e-13)
    assert all(m == 20 and sd == 77 and ls_ == CELL_LAW for m, sd, ls_ in seen)
    assert bs.flags == ()                         # 1 of 12 = 8.3 % <= 10 %: not flagged


def test_table_columns_on_real_stage(cell, stub):
    # a tiny end-to-end stage on the fixture: every row's keys, twins, finiteness
    lay, tg = cell["layout"], cell["target"]
    cfg = nf.NoiseFitConfig(M_binding=20, max_nfev=2)
    st = nsf.SweepStageConfig(B_v=2, M_refit=20)
    res = nsf.fit_sweeps(stub, tg, lay, cfg, st)
    tr = res.table_rows()
    assert len(tr) == lay.n_sweeps + lay.K_LS
    assert all(list(r.keys()) == SWEEP_COLS for r in tr)
    assert list(res.cell_columns().keys()) == CELL_COLS
    for r in res.rows:
        tw = res.twin_laws[(r["protocol"], r["sweep"])]
        assert tw.tau_m == CELL_LAW.tau_m and tw.tau_s == CELL_LAW.tau_s
        for key in ("sigma_m_mV", "tau_m_s", "log_sigma_m", "v_log_sigma_m", "bootstrap_y_m", "lambda_m",
                    "sigma_m_tilde_mV"):
            assert key in r, key
        assert "rho_f" not in r


def test_swap_min_sigma_f_ratio(cell, stub, monkeypatch):
    lay, tg = cell["layout"], cell["target"]
    monkeypatch.setattr(nsf, "bootstrap_v", lambda *a, **k: nsf.SweepBootstrap(0.01, 0.02, np.zeros(3), np.zeros(3), 3, 3, ()))
    monkeypatch.setattr(nsf, "fit_sweep", _fake_fit(0.8 * CELL_LAW.sigma_f))
    s = lay.sweep_numbers[0]
    row = nsf.run_sweep(stub, tg, lay, nf.NoiseFitConfig(M_binding=20), nsf.SweepStageConfig(min_sigma_f_ratio=0.9),
                        "ss", s)
    assert row["reason"] == "degenerate:sigma_f"


def test_zero_bootstrap_variance_fails_the_sweep():
    # 2.6 (e) as amended 2026-11-17 (amendments 4-5, D-155; the final check's finding F1): a bootstrap variance
    # that is NaN or below V_MIN = 1e-12 fails the sweep with 'v_none'; pool_cell, given such a row made elsewhere,
    # marks it failed, leaves it out of both poolings and gives its twin theta_hat(c); it does not raise.
    # (Rewritten by the orchestrating session on 2026-11-19 when the tests were committed: the tester's version
    # recorded the gap, pool_cell raising ValueError at commit 17d8205.)
    assert nsf.V_MIN == 1e-12
    for v0 in (0.0, 1e-31):
        rows = [_row("ss", 92, 0.034, -4.0, v0, -2.9, 0.01), _row("ss", 93, 0.036, -4.3, 0.05, -2.7, 0.03)]
        res = nsf.pool_cell(rows, CELL_LAW, 1)
        r92 = [r for r in res.rows if r["sweep"] == 92][0]
        assert r92["failed"] and r92["reason"] == "v_none"
        assert "sweep_fit_failed:ss:92:v_none" in r92["flags"]
        assert res.twin_laws[("ss", 92)] == CELL_LAW
        for co in ("sigma_m", "sigma_s"):
            assert res.pools["ss"][co].k == 1
