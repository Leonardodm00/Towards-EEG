"""smoke_noise_fit.py -- Block 4 smoke suite (F1-F10) for noise_fit.py.

Reduced sizes throughout (M, R, iterations), on the layout of Block 3's
writer-like fixture (2 SS sweeps x 37 pulses with 11 planted drops, K = 63;
5 LS pre-windows of 13250 samples; a planted tilt). Each check prints one
PASS/FAIL line and the suite ends with "SMOKE noise_fit: n/10 passed".

    python smoke_noise_fit.py            # about 2 min, no NEURON needed
    python smoke_noise_fit.py F1,F5      # a subset (mutation runs)

Pure ASCII, LF. Spec: specs/SPEC.md, Block 4 (test oracles F1-F10).
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
from scipy.stats.mstats import hdquantiles

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import noise_cell_stats as ncs      # noqa: E402
import noise_fit as nf              # noqa: E402
import noise_law as nl              # noqa: E402
import noise_windows as nw          # noqa: E402
import smoke_noise_windows as sw    # noqa: E402  (the fixture writer)

RESULTS = []
Z_SE = 3.5
THETA_CF = sw.THETA_CF
SIGMA_B_PLANTED = 0.1               # mV/s
WHITE = nl.NoiseLaw(0.05, 1e-3, 0.0, 0.04)   # one-component oracle law (zero slow part)


def report(name, ok, note=""):
    RESULTS.append(bool(ok))
    print("%s  %s  -- %s" % ("PASS" if ok else "FAIL", name, note))
    sys.stdout.flush()


def build_cell(d, rng, sigma_b=SIGMA_B_PLANTED, law=THETA_CF):
    """Fixture -> (layout, target, ss, truth) at t_SS = 80 ms (the floor)."""
    truth = sw.write_fixture(d, rng, law=law, sigma_b=sigma_b)
    ss = nw.load_ss_windows(d)
    ls = nw.load_ls_prewindows(d)
    lay = nf.CellLayout.from_windows(ss, ls, 80.0)
    X, _j0 = nw.enlarged_windows(ss, 80.0)
    Xt, _, _ = nw.subtract_mean_response(X, ss.polarity_is_dep)
    ss_stats = ncs.ss_window_stats(Xt, ss.fs)
    ls_stats = ncs.ls_window_stats(ls.pre, ls.fs)
    V, N = ncs.window_mean_semivariogram(Xt, ss, 4)
    tg = nf.CellTarget.from_stats(ss_stats, ls_stats, lay, V, N)
    return lay, tg, ss, truth


# ---------------------------------------------------------------------------
def _f1(lay):
    """Cutting: bit-identical to an independent loop over the stored positions."""
    seed = 11
    rng = np.random.default_rng(seed)
    ss_stats, ls_stats, Xt = nf.simulate_cell_windows(THETA_CF, 0.07, lay, rng, return_windows=True)
    rng2 = np.random.default_rng(seed)
    X = np.empty((lay.K, lay.m))
    for s_idx, _sw in enumerate(lay.sweep_numbers):
        units = nl.draw_units(rng2, 1, lay.L_s)
        tr = nl.simulate_sweep(THETA_CF, lay.dt, lay.L_s, units, 0, sigma_b=0.07)
        for k in range(lay.K):
            if lay.sweep_index[k] == s_idx:
                p = int(lay.position[k])
                start = p * lay.period_samples + lay.pre_n + int(round(80.0e-3 * lay.fs))
                X[k] = tr[start:start + lay.m]
    # the per-polarity mean window, written out
    Xt2 = X.copy()
    for mask in (lay.polarity_is_dep, ~lay.polarity_is_dep):
        Xt2[mask] -= X[mask].mean(axis=0)
    same = np.array_equal(Xt, Xt2)
    shapes = ss_stats.shape == (lay.K, 5) and ls_stats.shape == (lay.K_LS, 10)
    finite = np.all(np.isfinite(ss_stats)) and np.all(np.isfinite(ls_stats))
    report("F1 cutting: simulate_cell_windows == independent cut at the stored positions (bit for bit); shapes",
           same and shapes and finite, "identical %s, shapes %s %s" % (same, ss_stats.shape, ls_stats.shape))


def _f2(lay):
    """Subtraction: mean sigma_hat^2 = sigma_f^2 (1 - 1 / K_p) at a white law."""
    rng = np.random.default_rng(21)
    n_rep = 40
    vals = []
    for _ in range(n_rep):
        s, _l = nf.simulate_cell_windows(WHITE, 0.0, lay, rng, parts=("ss",))
        vals.append(s[:, 0] ** 2)
    vals = np.array(vals)                                   # (n_rep, K)
    K_dep = int(lay.polarity_is_dep.sum())
    K_hyp = int(lay.K - K_dep)
    expected = WHITE.sigma_f ** 2 * np.where(lay.polarity_is_dep, 1.0 - 1.0 / K_dep, 1.0 - 1.0 / K_hyp)
    mean_all = vals.mean()
    exp_all = expected.mean()
    se = vals.std(ddof=1) / math.sqrt(vals.size)
    ok1 = abs(mean_all - exp_all) <= Z_SE * se
    ok2 = (WHITE.sigma_f ** 2 - mean_all) > Z_SE * se
    report("F2 subtraction: pooled mean sigma_hat^2 = sigma_f^2 (1 - 1/K_p) within 3.5 SE; below sigma_f^2",
           ok1 and ok2, "mean %.6g expected %.6g (sigma_f^2 %.6g) SE %.2g" % (mean_all, exp_all, WHITE.sigma_f ** 2, se))


def _f3(lay):
    """CRN: identical twice, different at another seed, continuous in theta."""
    cfg = nf.NoiseFitConfig(M_binding=60)
    q1, pool_ss, pool_ls = nf.binding_function(THETA_CF, 0.05, lay, cfg, 5, return_pool=True)
    q2 = nf.binding_function(THETA_CF, 0.05, lay, cfg, 5)
    q3 = nf.binding_function(THETA_CF, 0.05, lay, cfg, 6)
    law_eps = nl.NoiseLaw(THETA_CF.sigma_f * (1 + 1e-6), THETA_CF.rho_f * (1 + 1e-6),
                          THETA_CF.sigma_s * (1 + 1e-6), THETA_CF.tau_s * (1 + 1e-6))
    q4 = nf.binding_function(law_eps, 0.05, lay, cfg, 5)
    se_mc = np.concatenate([pool_ss.std(axis=0, ddof=1) * 1.2533 / math.sqrt(pool_ss.shape[0]),
                            pool_ls.std(axis=0, ddof=1) * 1.2533 / math.sqrt(pool_ls.shape[0])])
    ident = np.array_equal(q1, q2)
    diff = not np.array_equal(q1, q3)
    cont = np.all(np.abs(q4 - q1) < 0.1 * se_mc)
    # determinism of Sigma_hat on one seed
    s1 = nf.simulated_covariance(THETA_CF, 0.05, lay, 4, np.random.default_rng(9))
    s2 = nf.simulated_covariance(THETA_CF, 0.05, lay, 4, np.random.default_rng(9))
    det = np.array_equal(s1.vectors, s2.vectors)
    report("F3 CRN: binding bit-identical at one seed, different at another, continuous in theta; Sigma deterministic",
           ident and diff and cont and det,
           "identical %s differs %s max|dq|/SE_MC %.3g Sigma deterministic %s" % (ident, diff, float(np.max(np.abs(q4 - q1) / se_mc)), det))


def _f4(lay):
    """Sigma_hat at the white law against the asymptotic variance of the median."""
    R = 120
    sig = nf.simulated_covariance(WHITE, 0.0, lay, R, np.random.default_rng(31), per_sweep=True)
    m, K = lay.m, lay.K
    var_sd_window = WHITE.sigma_f ** 2 / (2.0 * (m - 1))           # Var(sigma_hat) of one white window
    expected = (math.pi / 2.0) * var_sd_window / K                   # the median over K windows
    got = sig.Sigma[0, 0]
    ok1 = abs(got / expected - 1.0) <= 0.4
    # LS: one window per sweep -> the single-window variance of sigma_hat
    ls_rows = sig.per_sweep[int(lay.ls_sweep_number[0])]
    var_ls = ls_rows[:, 0].var(ddof=1)
    exp_ls = WHITE.sigma_f ** 2 / (2.0 * (lay.n_ls - 1))
    ok2 = abs(var_ls / exp_ls - 1.0) <= 0.4
    S = sig.Sigma
    spd = np.allclose(S, S.T) and np.all(np.linalg.eigvalsh(S) > 0)
    kappa = nf.correlation_condition_number(S)
    keys = set(sig.per_sweep)
    want = set(int(s) for s in lay.sweep_numbers) | set(int(s) for s in lay.ls_sweep_number)
    shapes = all(sig.per_sweep[s].shape == (R, 5) for s in lay.sweep_numbers) and \
        all(sig.per_sweep[int(s)].shape == (R, 10) for s in lay.ls_sweep_number)
    report("F4 Sigma_hat: Var(ss_sd) and the LS single-window Var(sd) within 40 % of the white-law values; SPD; per-sweep vectors",
           ok1 and ok2 and spd and np.isfinite(kappa) and keys == want and shapes,
           "ss_sd var ratio %.3f, ls_sd var ratio %.3f, SPD %s, kappa %.3g, sweeps %s" % (got / expected, var_ls / exp_ls, spd, kappa, sorted(keys)))


def _f5(lay, ss):
    """Tilt: (T1) on 200 replicate cells at a known sigma_b; clipping."""
    rng = np.random.default_rng(41)
    n_rep = 200
    d2, npairs = [], 0
    for _ in range(n_rep):
        _s, _l, Xt = nf.simulate_cell_windows(THETA_CF, SIGMA_B_PLANTED, lay, rng, parts=("ss",), return_windows=True)
        V_r, N_r = ncs.window_mean_semivariogram(Xt, ss, 4)
        d2.append(V_r * N_r)
        npairs += N_r
    V_hat = float(np.sum(d2) / npairs)
    tilt = nf.tilt_from_semivariogram(V_hat, npairs, THETA_CF, lay, 4)
    # SE by the delta method: each pair's half squared difference has mean V and variance 2 V^2
    se_V = V_hat * math.sqrt(2.0 / npairs)
    se_sb2 = 2.0 * se_V / tilt.h_star ** 2
    se_sb = se_sb2 / (2.0 * SIGMA_B_PLANTED)
    ok1 = abs(tilt.sigma_b - SIGMA_B_PLANTED) <= Z_SE * se_sb and tilt.flags == ()
    clipped = nf.tilt_from_semivariogram(0.5 * tilt.V_model, npairs, THETA_CF, lay, 4)
    ok2 = clipped.sigma_b == 0.0 and "tilt_clipped" in clipped.flags
    none = nf.tilt_from_semivariogram(float("nan"), 0, THETA_CF, lay, 4)
    ok3 = none.sigma_b == 0.0 and "tilt_no_pairs" in none.flags
    ok4 = abs(tilt.h_star - 4 * lay.period_samples * lay.dt) < 1e-12
    report("F5 tilt: sigma_b_hat within 3.5 SE of the planted 0.1 mV/s over 200 replicate cells; clipped / no-pair flags; h* = 4 P dt",
           ok1 and ok2 and ok3 and ok4,
           "sigma_b_hat %.4f (SE %.4f, %d pairs, V_hat %.4g, V_model %.4g) flags %s" % (tilt.sigma_b, se_sb, npairs, V_hat, tilt.V_model, tilt.flags))


def _f6(lay):
    """Self-consistency of the inner solve: q_hat := q_sim(theta*) exactly."""
    cfg = nf.NoiseFitConfig(M_binding=60, max_nfev=10)
    seed = 77
    q_star, pool_ss, pool_ls = nf.binding_function(THETA_CF, 0.05, lay, cfg, seed, return_pool=True)
    se = np.concatenate([pool_ss.std(axis=0, ddof=1) * 1.2533 / math.sqrt(lay.K),
                         pool_ls.std(axis=0, ddof=1) * 1.2533 / math.sqrt(lay.K_LS)])
    mask = np.isfinite(q_star) & (se > 0)
    C = np.diag(se[mask])
    starts = [nl.NoiseLaw(THETA_CF.sigma_f, THETA_CF.rho_f, THETA_CF.sigma_s, THETA_CF.tau_s / 2.0),
              nl.NoiseLaw(THETA_CF.sigma_f, THETA_CF.rho_f, THETA_CF.sigma_s, THETA_CF.tau_s * 2.0)]
    t0 = time.time()
    res = nf.solve_inner(q_star, C, starts, 0.05, lay, cfg, seed, mask)
    du = np.abs(res.u - nf.law_to_u(THETA_CF))
    ok = np.all(du <= cfg.tol_u) and len(res.starts) == 2 and res.start_index in (0, 1) and res.J < 1e-3
    report("F6 inner solve: from tau_s/2 and 2 tau_s back to theta* within 0.01 in u on an exactly attainable target; two-start bookkeeping",
           ok, "max|du| %.4f J %.3g start %d nfev %d (%.0f s)" % (du.max(), res.J, res.start_index, res.nfev, time.time() - t0))


def _f7(lay, tg, truth):
    """A reduced fit end to end."""
    cfg = nf.NoiseFitConfig(M_binding=60, R_iter=40, R_final=60, B_boot=400, k_max=2, max_nfev=6,
                            starts_after_first=1)
    lines = []
    t0 = time.time()
    res = nf.fit_cell(tg, lay, cfg, log=lines.append)
    secs = time.time() - t0
    du = np.abs(res.u - nf.law_to_u(THETA_CF))
    tol = np.array([0.04, 0.03, 0.4, 1.0])
    ok_u = np.all(du <= tol)
    # the tilt: the plumbing (final law, h*, clipping) recomputed here
    V_model = nl.semivariogram(res.law, lay.dt, lay.m, lay.m, 4 * lay.period_samples)
    sb2 = 2.0 * (tg.V_hat - V_model) / (4 * lay.period_samples * lay.dt) ** 2
    ok_sb = abs(res.sigma_b - math.sqrt(max(sb2, 0.0))) < 1e-12 and (("tilt_clipped" in res.flags) == (sb2 < 0))
    # J reported = |C^-1 (q_sim - q_hat)|^2 with the final Sigma_hat and q_sim at the solution
    r = (res.q_sim - res.q_hat)[np.isfinite(res.SE)]
    J_check = float(r @ np.linalg.solve(res.Sigma, r))
    ok_J = abs(J_check - res.J) <= 1e-8 * max(1.0, res.J)
    ok_fields = (ok_J and res.n_solves == len(res.history) and res.n_solves >= 2 and np.isfinite(res.J) and np.isfinite(res.kappa)
                 and res.Sigma.shape == (15, 15) and res.dof == 11 and 0.0 <= res.p_value <= 1.0
                 and res.stats_dropped == () and len(res.labels_kept) == 15 and np.all(np.isfinite(res.SE))
                 and np.all(np.isfinite(res.q_sim)) and len(res.per_sweep_cov) == 7
                 and res.history[0]["weight"] == "W0" and res.history[-1]["weight"] == "final")
    try:
        txt = json.dumps(res.to_dict())
        back = json.loads(txt)
        ok_json = abs(back["law"]["sigma_f_mV"] - res.law.sigma_f) < 1e-15 and back["n_solves"] == res.n_solves \
            and len(back["Sigma"]) == 15 and len(back["history"]) == res.n_solves
    except Exception as exc:  # noqa: BLE001
        ok_json = False
        lines.append("to_dict failed: %r" % (exc,))
    report("F7 reduced fit: |u_hat - u*| within (0.04, 0.03, 0.4, 1.0); tilt plumbing; J = r^T Sigma^-1 r at the solution; every field; JSON round trip",
           ok_u and ok_sb and ok_fields and ok_json,
           "du %s theta_hat %s sigma_b %.4f J %.3g (recomputed %.3g) p %.3f kappa %.3g solves %d converged %s flags %s (%.0f s)"
           % (np.array2string(du, precision=3), res.law, res.sigma_b, res.J, J_check, res.p_value, res.kappa, res.n_solves,
              res.converged, res.flags, secs))
    for ln in lines:
        print("      " + ln)


def _f8():
    """hd_median against hdquantiles, NaN handling, closeness to the median."""
    rng = np.random.default_rng(51)
    A = rng.standard_normal((3000, 4)) * np.array([1.0, 0.1, 10.0, 0.05]) + np.array([0.0, 5.0, -3.0, 0.5])
    ref = np.array([float(np.asarray(hdquantiles(A[:, j], prob=[0.5]))[0]) for j in range(4)])
    got = nf.hd_median(A)
    ok1 = np.allclose(got, ref, rtol=0, atol=1e-12)
    B = A.copy(); B[:100, 1] = np.nan
    got_nan = nf.hd_median(B)
    ref_nan = float(np.asarray(hdquantiles(A[100:, 1], prob=[0.5]))[0])
    ok2 = abs(got_nan[1] - ref_nan) < 1e-12 and np.allclose(got_nan[[0, 2, 3]], got[[0, 2, 3]])
    # closeness to the sample median on symmetric data: within half a standard error of the median
    se_med = 1.2533 * A.std(axis=0, ddof=1) / math.sqrt(A.shape[0])
    ok3 = np.all(np.abs(got - np.median(A, axis=0)) < 0.5 * se_med)
    one = nf.hd_median(np.array([[3.0, np.nan]]))
    ok4 = one[0] == 3.0 and np.isnan(one[1])
    report("F8 hd_median: equals hdquantiles (1e-12), ignores NaN per column, within half an SE of the sample median, single value",
           ok1 and ok2 and ok3 and ok4, "max|HD - median|/SE %.3f" % float(np.max(np.abs(got - np.median(A, axis=0)) / se_med)))


def _f9(lay, tg):
    """Masks: two NaN statistics are dropped everywhere."""
    q = tg.q_hat.copy()
    j1 = ncs.STAT_LABELS.index("ss_rho_10ms")
    j2 = ncs.STAT_LABELS.index("ls_last_rho_10ms")
    q[j1] = np.nan; q[j2] = np.nan
    tg2 = nf.CellTarget(q_hat=q, per_window_ss=tg.per_window_ss, per_window_ls=tg.per_window_ls,
                        V_hat=tg.V_hat, N_pairs=tg.N_pairs)
    cfg = nf.NoiseFitConfig(M_binding=60, R_iter=20, R_final=20, B_boot=200, k_max=0, max_nfev=1)
    t0 = time.time()
    res = nf.fit_cell(tg2, lay, cfg)
    ok = (res.Sigma.shape == (13, 13) and set(res.stats_dropped) == {"ss_rho_10ms", "ls_last_rho_10ms"}
          and len(res.labels_kept) == 13 and res.dof == 9 and np.isnan(res.SE[j1]) and np.isnan(res.SE[j2])
          and any(f.startswith("stats_dropped:") for f in res.flags) and "no_plateau" in res.flags
          and res.n_solves == 2)
    report("F9 masks: two NaN statistics -> d_eff 13, Sigma (13, 13), labels, SE NaN, flag; k_max 0 -> no_plateau",
           ok, "dropped %s dof %d solves %d flags %s (%.0f s)" % (res.stats_dropped, res.dof, res.n_solves, res.flags, time.time() - t0))


def _f10():
    bad = []
    for name in ("noise_fit.py", "smoke_noise_fit.py"):
        raw = (HERE / name).read_bytes()
        if b"\r" in raw or any(b > 127 for b in raw):
            bad.append(name)
    report("F10 ASCII + LF on the two files", not bad, "ok" if not bad else "bad: %s" % bad)


def main(only=None):
    t_all = time.time()
    tmp = tempfile.mkdtemp(prefix="smoke_noise_fit_")
    lay, tg, ss, truth = build_cell(tmp, np.random.default_rng(2026))
    print("fixture: K %d (dep %d), K_LS %d, m %d, L_s %d, V_hat %.4g (%d pairs)"
          % (lay.K, int(lay.polarity_is_dep.sum()), lay.K_LS, lay.m, lay.L_s, tg.V_hat, tg.N_pairs))
    checks = [("F1", lambda: _f1(lay)), ("F2", lambda: _f2(lay)), ("F3", lambda: _f3(lay)),
              ("F4", lambda: _f4(lay)), ("F5", lambda: _f5(lay, ss)), ("F6", lambda: _f6(lay)),
              ("F7", lambda: _f7(lay, tg, truth)), ("F8", _f8), ("F9", lambda: _f9(lay, tg)), ("F10", _f10)]
    for name, fn in checks:
        if only and name not in only:
            continue
        try:
            fn()
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            report("%s crashed" % name, False, "see traceback")
    n_ok = sum(RESULTS)
    print("SMOKE noise_fit: %d/%d passed  (%.0f s)" % (n_ok, len(RESULTS), time.time() - t_all))
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    # optional: python smoke_noise_fit.py F1,F5   (a subset, for mutation runs)
    sys.exit(main(set(sys.argv[1].split(",")) if len(sys.argv) > 1 else None))
