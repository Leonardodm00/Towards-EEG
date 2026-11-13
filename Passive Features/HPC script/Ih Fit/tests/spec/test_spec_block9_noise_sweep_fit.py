"""Spec-derived tests, Block 9: noise_sweep_fit (SPEC 2.5 (P0)-(P4), items
(a)-(g), Block 9's entry with its as-built note). The cell is the tester's
writer-like fixture through Blocks 3-4 with a stub per-cell result (the
law, per-sweep covariances from simulated_covariance); expected values come
from exact rational arithmetic, statsmodels, numpy, SeedSequence written out,
and refit doubles. Reduced sizes throughout.
"""
import json
import math
from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from statsmodels.stats.meta_analysis import combine_effects

import _fixtures as fx
import noise_cell_stats as ncs
import noise_fit as nf
import noise_sweep_fit as nsf
import noise_windows as nw
from noise_law import NoiseLaw

CELL_LAW = NoiseLaw(0.038 * 1.05, 0.195, 0.0508 * 0.8, 0.0366)     # theta_hat(c): set apart from the truth
CFG = nf.NoiseFitConfig(M_binding=30, max_nfev=4)


@pytest.fixture(scope="module")
def pack(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("b9"))
    d = fx.write_cell(root, specimen_id=333, ss_tilts=(0.1, -0.05), drop=[(0, 3), (0, 4), (1, 10), (1, 11), (1, 12)],
                      ls=((60, -50.0), (61, 30.0), (63, -90.0)), ls_tilts=(0.05, -0.02, 0.08))
    nwin = nw.ss_noise_windows(nw.load_ss_windows(d))
    ls = nw.load_ls_prewindows(d)
    layout = nf.CellLayout.from_cell(nwin, ls)
    target = nf.CellTarget.from_cell(nwin, ls, layout)
    sr = nf.simulated_covariance(CELL_LAW, layout, 40, np.random.default_rng(3), per_sweep=True)
    cov = {s: np.cov(v, rowvar=False, ddof=1) for s, v in sr.per_sweep.items()}
    stub = nf.CellFitResult(specimen_id=333, law=CELL_LAW, u=nf.law_to_u(CELL_LAW), J=1.0, dof=11, p_value=0.5,
                            kappa=10.0, Sigma=sr.Sigma, labels_kept=ncs.STAT_LABELS, stats_dropped=(),
                            q_hat=target.q_hat, q_sim=target.q_hat, SE=np.ones(15), ss_tilt={}, ls_tilt={},
                            n_solves=3, converged=True, flags=(), history=[], per_sweep_cov=cov, config=CFG,
                            seconds=0.0)
    return dict(layout=layout, target=target, stub=stub, cov=cov)


# --- O9.1 configuration -----------------------------------------------------------
def test_O9_1_defaults_and_validation():
    c = nsf.SweepStageConfig()
    assert (c.B_v, c.M_refit, c.max_failed_refit_frac, c.min_stats, c.seed) == (50, 1000, 0.1, 4, 20261030)
    nsf.SweepStageConfig(M_refit=None)
    for kw in (dict(B_v=1), dict(M_refit=0), dict(max_failed_refit_frac=1.1), dict(max_failed_refit_frac=-0.1),
               dict(min_stats=0)):
        with pytest.raises(ValueError):
            nsf.SweepStageConfig(**kw)


# --- O9.2 - O9.4 restriction, target, seeds ------------------------------------------
def test_O9_2_sweep_layout(pack):
    L = pack["layout"]
    sub = nsf.sweep_layout(L, "ss", 35)
    g = L.sweep_number == 35
    assert sub.K == int(g.sum()) and sub.K_LS == 0 and sub.sweep_numbers == (35,)
    assert_array_equal(sub.position, L.position[g])
    assert_array_equal(sub.ss_tilt, L.ss_tilt[[1]])
    for f in ("fs", "dt", "n_w", "pre_n", "j0", "j_floor", "period_samples", "train_length", "n_ls", "specimen_id"):
        assert getattr(sub, f) == getattr(L, f), f
    sl = nsf.sweep_layout(L, "ls", 61)
    assert sl.K == 0 and sl.K_LS == 1 and sl.ls_sweep_number.tolist() == [61] and sl.ls_tilt[0] == L.ls_tilt[1]
    for bad in (("ss", 36), ("ls", 31), ("xx", 31)):
        with pytest.raises(ValueError):
            nsf.sweep_layout(L, *bad)


def test_O9_3_sweep_target(pack):
    L, T = pack["layout"], pack["target"]
    q, m = nsf.sweep_target(T, L, "ss", 31)
    assert_array_equal(q[:5], np.median(T.per_window_ss[L.sweep_number == 31], axis=0))
    assert np.all(np.isnan(q[5:])) and m.tolist() == [True] * 5 + [False] * 10
    q, m = nsf.sweep_target(T, L, "ls", 63)
    assert_array_equal(q[5:], T.per_window_ls[2]) and np.all(np.isnan(q[:5])) and m.sum() == 10


def test_O9_4_sweep_seeds():
    a = nsf.sweep_seeds(20261030, 333, "ss", 31)
    kids = np.random.SeedSequence([20261030, 333, 0, 31]).spawn(2)
    assert a == tuple(int(k.generate_state(1, dtype=np.uint64)[0]) for k in kids)
    assert nsf.sweep_seeds(20261030, 333, "ls", 31) != a and nsf.sweep_seeds(20261030, 333, "ss", 35) != a


# --- O9.5 the per-sweep fit -----------------------------------------------------------
def _J(u, sub, pr, cfg, seed, q_hat, mask, Sigma_s, tau_s):
    q = nf.binding_function(NoiseLaw(math.exp(u[0]), u[1], math.exp(u[2]), tau_s), sub, cfg, seed, which=pr)
    r = (q - q_hat)[mask]
    S = Sigma_s[np.ix_(mask[nsf._SLICE[pr]], mask[nsf._SLICE[pr]])]
    return float(r @ np.linalg.solve(S, r))


@pytest.mark.parametrize("pr,sweep", [("ss", 31), ("ls", 61)])
def test_O9_5_self_consistency(pack, pr, sweep):
    L, cov = pack["layout"], pack["cov"]
    sub = nsf.sweep_layout(L, pr, sweep)
    seed_b, _ = nsf.sweep_seeds(1, 333, pr, sweep)
    star = NoiseLaw(0.04, 0.2, 0.045, CELL_LAW.tau_s)
    cfg = replace(CFG, M_binding=60, max_nfev=15)
    q_hat = nf.binding_function(star, sub, cfg, seed_b, which=pr)
    mask = np.isfinite(q_hat)
    start = NoiseLaw(0.04, 0.2, 0.045 * math.exp(0.2), CELL_LAW.tau_s)      # 0.2 away in log sigma_s
    f = nsf.fit_sweep(q_hat, mask, cov[sweep], sub, start, cfg, seed_b)
    assert not f.failed and f.law.tau_s == CELL_LAW.tau_s
    assert np.max(np.abs(f.u - nf.law_to_u(star)[:3])) < 0.01 and f.J < 1e-3
    assert f.d == int(mask.sum()) and f.dof == f.d - 3
    assert f.J == pytest.approx(_J(f.u, sub, pr, cfg, seed_b, q_hat, mask, cov[sweep], star.tau_s), rel=1e-6, abs=1e-12)


def test_O9_5_failure_rules(pack):
    L, T, cov = pack["layout"], pack["target"], pack["cov"]
    sub = nsf.sweep_layout(L, "ls", 60)
    seed_b, _ = nsf.sweep_seeds(1, 333, "ls", 60)
    q, m = nsf.sweep_target(T, L, "ls", 60)
    f = nsf.fit_sweep(q, m, cov[60], sub, NoiseLaw(1.0, 0.9, 1.0, CELL_LAW.tau_s), replace(CFG, max_nfev=1), seed_b)
    assert f.failed and f.reason == "no_convergence" and f.status == 0
    q_hi = q.copy()
    q_hi[[5, 9, 10]] = 1400.0                                            # the three LS SDs, far above any law in the bounds
    f2 = nsf.fit_sweep(q_hi, m, cov[60], sub, CELL_LAW, replace(CFG, max_nfev=15), seed_b)
    assert f2.failed and f2.reason.startswith("at_bound:")
    row = nsf.run_sweep(pack["stub"], nf.CellTarget(q_hat=T.q_hat, per_window_ss=T.per_window_ss,
                                                    per_window_ls=np.where(np.arange(10) < 7, np.nan, T.per_window_ls)),
                        L, CFG, nsf.SweepStageConfig(B_v=2), "ls", 61)
    assert row["failed"] and row["reason"] == "stats" and row["n_calls"] == 0 and row["d"] == 3
    assert "sweep_fit_failed:ls:61:stats" in row["flags"]


# --- O9.6 the bootstrap machinery with refit doubles ---------------------------------------
class RefitDouble:
    def __init__(self, fail_every=0, pr="ss"):
        self.fail_every, self.pr, self.calls = fail_every, pr, []

    def __call__(self, q_b, m_b, Sigma_s, sub, law_start, cfg, seed_b):
        self.calls.append((q_b.copy(), Sigma_s, law_start, cfg.M_binding, seed_b))
        i = len(self.calls)
        fail = bool(self.fail_every) and (i % self.fail_every == 0)
        first = q_b[0] if self.pr == "ss" else q_b[5]
        return nsf.SweepFit(self.pr, 0, np.array([0.0, 0.0, math.log(first)]), law_start, 0.0, int(m_b.sum()),
                            int(m_b.sum()) - 3, 0, 1, (), fail, "x" if fail else "")


def _fit_ok(pr, sweep, law):
    return nsf.SweepFit(pr, sweep, nf.law_to_u(law)[:3], law, 0.1, 5, 2, 1, 1, (), False, "")


@pytest.mark.parametrize("pr,sweep", [("ss", 35), ("ls", 63)])
def test_O9_6_bootstrap_against_independent_loop(pack, pr, sweep):
    L, cov = pack["layout"], pack["cov"]
    sub = nsf.sweep_layout(L, pr, sweep)
    law_s = NoiseLaw(0.041, 0.21, 0.05, CELL_LAW.tau_s)
    seed_b, seed_v = nsf.sweep_seeds(9, 333, pr, sweep)
    dbl = RefitDouble(pr=pr)
    mask = np.zeros(15, bool)
    mask[nsf._SLICE[pr]] = True
    st = nsf.SweepStageConfig(B_v=6, M_refit=17)
    boot = nsf.bootstrap_v(_fit_ok(pr, sweep, law_s), mask, cov[sweep], sub, CELL_LAW, CFG, st, seed_b, seed_v, refit=dbl)
    rng = np.random.default_rng(seed_v)
    y = []
    for b in range(6):
        s_st, l_st = nf.simulate_cell_windows(law_s, sub, rng, parts=(pr,))
        y.append(math.log(np.median(s_st[:, 0]) if pr == "ss" else l_st[0, 0]))
    assert_allclose(boot.y, y, rtol=0, atol=1e-15)
    assert boot.v == pytest.approx(np.var(y, ddof=1), rel=1e-13) and boot.B_ok == 6 and boot.flags == ()
    for (_q, S, ls_, M, sb) in dbl.calls:
        assert S is cov[sweep] and ls_ == CELL_LAW and M == 17 and sb == seed_b


@pytest.mark.parametrize("every,B,flag", [(5, 10, "v_refits_failed:2/10"), (10, 10, None), (1, 4, "v_none")])
def test_O9_6_failed_refits(pack, every, B, flag):
    L, cov = pack["layout"], pack["cov"]
    sub = nsf.sweep_layout(L, "ls", 60)
    mask = np.zeros(15, bool)
    mask[5:] = True
    boot = nsf.bootstrap_v(_fit_ok("ls", 60, CELL_LAW), mask, cov[60], sub, CELL_LAW, CFG,
                           nsf.SweepStageConfig(B_v=B, M_refit=5), 1, 2, refit=RefitDouble(every, "ls"))
    n_fail = B // every
    assert boot.B_ok == B - n_fail and int(np.isnan(boot.y).sum()) == n_fail
    if flag is None:
        assert boot.flags == ()                                # exactly 10 % is not "more than 10 %"
    else:
        assert flag in boot.flags
    if B - n_fail >= 2:
        assert boot.v == pytest.approx(np.var(boot.y[np.isfinite(boot.y)], ddof=1), rel=1e-13)
    else:
        assert math.isnan(boot.v)
    with pytest.raises(ValueError):
        nsf.bootstrap_v(replace(_fit_ok("ls", 60, CELL_LAW), failed=True), mask, cov[60], sub, CELL_LAW, CFG,
                        nsf.SweepStageConfig(B_v=2), 1, 2, refit=RefitDouble(0, "ls"))


# --- O9.7 the pooling ------------------------------------------------------------------------
def test_O9_7_pool_protocol_exact_and_statsmodels():
    rng = np.random.default_rng(71)
    n_clip = 0
    for _ in range(200):
        k = int(rng.integers(2, 13))
        y = rng.normal(-3.0, rng.uniform(0.01, 0.4), size=k)
        v = rng.uniform(0.002, 0.08, size=k)
        p = nsf.pool_protocol(y, v)
        Q, raw, t2, mu, lam, ysh, scale_num, den = fx.dl_exact(y, v)
        scale = scale_num / den
        # 1e-13 of the formula's scale: float roundoff on k-term sums (exact reference in Fractions)
        assert abs(p.tau2_raw - raw) <= 1e-13 * scale and abs(p.Q - Q) <= 1e-12 * scale_num
        assert p.tau2 == max(0.0, p.tau2_raw)
        assert p.mu == pytest.approx(mu, rel=1e-12, abs=1e-12)
        assert_allclose(p.lam, lam, rtol=1e-12, atol=1e-15)
        assert_allclose(p.y_shrunk, ysh, rtol=1e-12, atol=1e-12)
        sm = combine_effects(y, v, method_re="dl")
        assert abs(sm.tau2 - p.tau2_raw) <= 1e-10 * scale
        if p.tau2_raw < 0:
            n_clip += 1
            assert np.all(p.lam == 0) and p.mu == pytest.approx(sm.mean_effect_fe, rel=1e-12)
        else:
            assert p.mu == pytest.approx(sm.mean_effect_re, rel=1e-12)
        lo, hi = np.minimum(p.mu, y), np.maximum(p.mu, y)
        assert np.all(p.y_shrunk >= lo - 1e-12) and np.all(p.y_shrunk <= hi + 1e-12)
        perm = rng.permutation(k)
        pp = nsf.pool_protocol(y[perm], v[perm])
        assert pp.tau2_raw == pytest.approx(p.tau2_raw, rel=1e-9, abs=1e-15)
        assert_allclose(pp.y_shrunk, p.y_shrunk[perm], rtol=1e-9, atol=1e-12)   # permutation-equivariant
        ps = nsf.pool_protocol(y + 2.5, v)
        assert ps.tau2_raw == pytest.approx(p.tau2_raw, rel=1e-6, abs=1e-12) and ps.mu == pytest.approx(p.mu + 2.5, rel=1e-12)
    assert 0 < n_clip < 200


def test_O9_7_pool_edge_cases():
    p1 = nsf.pool_protocol([-3.1], [0.01])
    assert p1.k == 1 and p1.mu == -3.1 and math.isnan(p1.tau2) and math.isnan(p1.lam[0]) and p1.y_shrunk[0] == -3.1
    p0 = nsf.pool_protocol([], [])
    assert p0.k == 0 and math.isnan(p0.mu) and math.isnan(p0.tau2)
    for y, v in (([1.0, 2.0], [0.1, 0.0]), ([1.0, 2.0], [0.1, -1.0]), ([1.0, np.nan], [0.1, 0.1]), ([1.0, 2.0], [0.1, np.inf])):
        with pytest.raises(ValueError):
            nsf.pool_protocol(y, v)


# --- O9.8 pool_cell, the tables, fit_sweeps ----------------------------------------------------
def _row(pr, s, failed, y=float("nan"), v=float("nan"), sf=0.04, rf=0.2):
    return {"protocol": pr, "sweep": s, "failed": failed, "log_sigma_s": y, "v_log_sigma_s": v, "sigma_f_mV": sf,
            "rho_f": rf, "sigma_s_mV": math.exp(y) if y == y else float("nan"), "tilt_mV_per_s": 0.01,
            "flags": ["sweep_fit_failed:%s:%d:x" % (pr, s)] if failed else [], "seconds": 1.0, "reason": "x" if failed else ""}


def test_O9_8_pool_cell_twins_and_columns():
    rows = [_row("ss", 31, False, -3.0, 0.01, 0.041, 0.21), _row("ss", 35, False, -2.7, 0.02, 0.039, 0.18),
            _row("ss", 36, True), _row("ls", 60, False, -2.9, 0.05)]
    res = nsf.pool_cell(rows, CELL_LAW, 333)
    p = nsf.pool_protocol([-3.0, -2.7], [0.01, 0.02])
    assert res.pools["ss"].tau2_raw == p.tau2_raw and res.pools["ls"].k == 1
    tw = res.twin_laws[("ss", 31)]
    assert (tw.sigma_f, tw.rho_f, tw.tau_s) == (0.041, 0.21, CELL_LAW.tau_s)
    assert tw.sigma_s == pytest.approx(math.exp(p.y_shrunk[0]), rel=1e-15)
    assert res.twin_laws[("ss", 36)] == CELL_LAW and res.twin_laws[("ls", 60)].sigma_s == pytest.approx(math.exp(-2.9))
    tab = {(r["noise_sweep_protocol"], r["noise_sweep_number"]): r for r in res.table_rows()}
    t36 = tab[("SS", 36)]
    assert math.isnan(t36["noise_sweep_sigma_s_raw_mV"]) and t36["noise_sweep_sigma_s_mV"] == CELL_LAW.sigma_s
    assert tab[("SS", 31)]["noise_sweep_sigma_s_raw_mV"] == pytest.approx(math.exp(-3.0))
    assert tab[("SS", 31)]["noise_sweep_w"] == pytest.approx(p.lam[0])
    need = {"noise_sweep_sigma_f_mV", "noise_sweep_rho_f", "noise_sweep_sigma_s_raw_mV", "noise_sweep_sigma_s_mV",
            "noise_sweep_v_log_sigma_s", "noise_sweep_w", "noise_sweep_flags", "noise_tilt_mV_per_s"}
    assert need <= set(t36)
    cc = res.cell_columns()
    assert cc["noise_cell_tau2_log_sigma_s_ss"] == res.pools["ss"].tau2 and math.isnan(cc["noise_cell_tau2_log_sigma_s_ls"])
    assert cc["noise_cell_mu_log_sigma_s_ls"] == -2.9
    json.dumps(res.to_dict(), allow_nan=False)


@pytest.mark.slow
def test_O9_8_fit_sweeps_end_to_end(pack):
    L, T, stub = pack["layout"], pack["target"], pack["stub"]
    st = nsf.SweepStageConfig(B_v=3, M_refit=30)
    a = nsf.fit_sweeps(stub, T, L, CFG, st)
    b = nsf.fit_sweeps(stub, T, L, CFG, st)
    strip = lambda d: {k: v for k, v in d.items() if k != "seconds"}
    da, db = a.to_dict(), b.to_dict()
    da["rows"] = [strip(r) for r in da["rows"]]
    db["rows"] = [strip(r) for r in db["rows"]]
    assert strip(da) == strip(db)                                         # bit-identical twice
    rows = [nsf.run_sweep(stub, T, L, CFG, st, "ss", s) for s in L.sweep_numbers] + \
           [nsf.run_sweep(stub, T, L, CFG, st, "ls", int(s)) for s in L.ls_sweep_number]
    c = nsf.pool_cell(rows, stub.law, 333, stage_cfg=st).to_dict()
    c["rows"] = [strip(r) for r in c["rows"]]
    assert strip(c) == strip(da)
    assert [(r["protocol"], r["sweep"]) for r in a.rows] == [("ss", 31), ("ss", 35), ("ls", 60), ("ls", 61), ("ls", 63)]
    for pr in ("ss", "ls"):
        ok = [r for r in a.rows if r["protocol"] == pr and not r["failed"]]
        p = nsf.pool_protocol([r["log_sigma_s"] for r in ok], [r["v_log_sigma_s"] for r in ok])
        assert a.pools[pr].k == p.k and (p.k < 2 or a.pools[pr].tau2_raw == p.tau2_raw)
    with pytest.raises(ValueError):
        nsf.fit_sweeps(replace(stub, specimen_id=334), T, L, CFG, st)
