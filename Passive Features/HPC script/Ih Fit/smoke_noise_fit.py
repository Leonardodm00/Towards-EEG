"""smoke_noise_fit.py -- Block 4 smoke suite (F1-F12) for noise_fit.py.

Reduced sizes throughout (M, R, iterations), on Block 3's writer-like
fixture read through Block 3's pipeline (2 SS sweeps x 37 pulses with 11
planted drops, K = 63; t_SS(c) of D-111; the kept windows, the tilt offsets
removed and the median subtraction (D-115); per-sweep tilts planted at
sigma_b = 0.1 mV/s and read by (TL1) at the 80 ms floor and (TL2), the LS
tail 1.5 s so that every LS tilt can be read; 5 LS
pre-windows of 13250 samples). The noise is the three-component law of SPEC
2.6 (a) (THETA_CF of smoke_noise_windows: sigma_f 0.035 mV, sigma_m 0.017 mV,
tau_m 0.3 ms, sigma_s 0.055 mV, tau_s 40 ms), fitted in the five coordinates
of 2.6 (d) on the 21 statistics of 2.6 (c) (every count from noise_cell_stats).
Each check prints one PASS/FAIL line and the suite ends with
"SMOKE noise_fit: n/12 passed".

    python smoke_noise_fit.py            # a few minutes, no NEURON needed
    python smoke_noise_fit.py F1,F5      # a subset (mutation runs)

Pure ASCII, LF. Spec: specs/SPEC.md, Block 4 (test oracles F1-F11, re-aimed
by the note of 2026-11-17) and 2.6 (d) (F12, the config's defaults).
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
import time
import traceback
from dataclasses import replace
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
SIGMA_B_PLANTED = 0.1               # mV/s: the fixture's per-sweep tilts are draws of N(0, 0.1^2)
N_STATS = len(ncs.STAT_LABELS)      # d (21, SPEC 2.6 (c)); every count below is taken from noise_cell_stats
N_SS = ncs.N_SS_STATS
N_LS = ncs.N_LS_STATS
N_PARAMS = 5                        # theta = (sigma_f, sigma_m, tau_m, sigma_s, tau_s), SPEC 2.6 (a)
WHITE = nl.NoiseLaw(0.05, 0.0, 3e-4, 0.0, 0.04)   # one-component oracle law: white only (middle and slow off)
SILENT = nl.NoiseLaw(0.0, 0.0, 3e-4, 0.0, 1.0)    # no noise at all: only the tilts remain
# F7's tolerance on |u_hat - u*| per coordinate (log sigma_f, log sigma_m, log tau_m, log sigma_s, log tau_s), from
# the fixture's own Monte Carlo spread (measured 2026-11-17 in the sandbox, numpy 2.5.3, scipy 1.18.1): build_cell
# with 40 other fixture seeds (1-40), each fitted with F7's reduced sizes (M 60, R 40/60, B 400, k_max 2, max_nfev 6;
# about 45 s each). The spread is bimodal. In 34 of 40 fits u_hat - u* had mean (-0.002, -0.022, -0.108, +0.032,
# +0.017) and SD (0.006, 0.045, 0.200, 0.151, 0.353), max |.| (0.010, 0.144, 0.523, 0.354, 0.760); the tolerance is
# |mean| + 4 SD of that mode, rounded up to 0.01. In the other 6 (seeds 1, 3, 6, 12, 28, 37) the slow part ran away
# along its ridge (sigma_s 7-15 times too large, tau_s 60 times too long or at the 10 s bound, flag
# at_bound:log_tau_s in 3): +1.9 to +2.7 in log sigma_s, +4.1 to +5.5 in log tau_s, sigma_f absorbing 0.03-0.08.
# The suite's fixture (seed 2026) is deterministic and falls in the first mode (u_hat - u* = (+0.011, -0.078, -0.087,
# -0.115, -0.525)); a defect that throws it into the second mode, or permutes coordinates, fails the check. The
# middle part is determined at these sizes about as well as the slow part's main mode (log tau_m SD 0.20).
F7_TOL_U = np.array([0.03, 0.21, 0.91, 0.64, 1.43])


def report(name, ok, note=""):
    RESULTS.append(bool(ok))
    print("%s  %s  -- %s" % ("PASS" if ok else "FAIL", name, note))
    sys.stdout.flush()


def build_cell(d, rng, sigma_b=SIGMA_B_PLANTED, law=THETA_CF, dropped=sw.DROPPED, ls_tail_n=sw.LS_LONG_TAIL_N):
    """Fixture -> (layout, target, nwin, ls, truth) through Block 3's pipeline."""
    truth = sw.write_fixture(d, rng, law=law, sigma_b=sigma_b, dropped=dropped, ls_tail_n=ls_tail_n)
    ss = nw.load_ss_windows(d)
    ls = nw.load_ls_prewindows(d)
    nwin = nw.ss_noise_windows(ss)
    lay = nf.CellLayout.from_cell(nwin, ls)
    tg = nf.CellTarget.from_cell(nwin, ls, lay)
    return lay, tg, nwin, ls, truth


def zero_tilts(lay):
    return replace(lay, ss_tilt=np.zeros(lay.n_sweeps), ls_tilt=np.zeros(lay.K_LS))


def groups_of(lay):
    """{(sweep, is_dep): row indices} of the layout's windows."""
    out = {}
    for k, key in enumerate(zip(lay.sweep_number.tolist(), lay.polarity_is_dep.tolist())):
        out.setdefault((int(key[0]), bool(key[1])), []).append(k)
    return {g: np.array(v) for g, v in out.items()}


def c_median(K, n_draw=1_000_000, chunk=100_000, seed=7):
    """c(K) = E[(x_1 - med(x_1..x_K))^2] / sigma^2 for K iid normal values (oracle; numpy only).
    Returns (via the identity 1 + Var(med) - 2/K, its SE; the direct estimate, its SE).
    The identity: cov(x_bar, med) = Var(x_bar) = 1/K, since x_bar is the efficient unbiased estimator
    and med is unbiased (Rao-Blackwell); then E[(x_1 - med)^2] = 1 + Var(med) - 2 cov(x_1, med)."""
    rng = np.random.default_rng(seed + K)
    meds, direct = [], []
    for _ in range(n_draw // chunk):
        x = rng.standard_normal((chunk, K))
        md = np.median(x, axis=1)
        meds.append(md)
        direct.append((x[:, 0] - md) ** 2)
    meds = np.concatenate(meds)
    direct = np.concatenate(direct)
    v = meds.var(ddof=1)
    se_v = v * math.sqrt(2.0 / (meds.size - 1))
    return 1.0 + v - 2.0 / K, se_v, direct.mean(), direct.std(ddof=1) / math.sqrt(direct.size)


# ---------------------------------------------------------------------------
def _f1(lay, t_ss_ms):
    """Cutting, the replicate's own tilt and offsets (D-115.3) and the subtraction: equal to an independent loop."""
    seed = 11
    rng = np.random.default_rng(seed)
    ss_stats, ls_stats, Xt, ls_tr, tilts_read = nf.simulate_cell_windows(THETA_CF, lay, rng, return_windows=True,
                                                                          return_tilts=True)
    rng2 = np.random.default_rng(seed)
    j0 = lay.pre_n + int(round(t_ss_ms * 1e-3 * lay.fs))
    jf = lay.pre_n + int(round(nw.WindowConfig().t_ss_floor_ms * 1e-3 * lay.fs))
    mf = lay.n_w - jf
    Xf = np.empty((lay.K, mf))
    for s_idx, _sw in enumerate(lay.sweep_numbers):
        units = nl.draw_units(rng2, 1, lay.L_s)
        tr = nl.simulate_sweep(THETA_CF, lay.dt, lay.L_s, units, 0, tilt_mV_per_s=float(lay.ss_tilt[s_idx]))
        for k in range(lay.K):
            if lay.sweep_index[k] == s_idx:
                start = int(lay.position[k]) * lay.period_samples + jf
                Xf[k] = tr[start:start + mf]
    means = Xf.mean(axis=1)
    centres = (lay.position * lay.period_samples + jf + 0.5 * (mf - 1)) * lay.dt
    X = Xf[:, j0 - jf:].copy()
    b_loop = {}
    for s in lay.sweep_numbers:                                  # (TL1) by its closed form, (OF1) written out
        g = np.flatnonzero(lay.sweep_number == s)
        b_loop[s] = sw._fw_slope(centres[g], means[g], lay.polarity_is_dep[g])
        tk = lay.position[g] * lay.period_samples * lay.dt
        X[g] -= (b_loop[s] * (tk - tk.mean()))[:, None]
    Xt2 = X.copy()
    for idx in groups_of(lay).values():                         # each group's per-sample median, written out
        Xt2[idx] = X[idx] - np.median(X[idx], axis=0)
    err = float(np.abs(Xt - Xt2).max())
    tilt_err = max(abs(tilts_read[int(s)] - b_loop[s]) for s in lay.sweep_numbers)
    same = err <= 1e-12 and tilt_err <= 1e-12 and lay.j_floor == jf
    units = nl.draw_units(rng2, lay.K_LS, lay.n_ls)
    ls_tr2 = nl.simulate_traces(THETA_CF, lay.dt, units.fast, units.mid, units.slow, tilt_mV_per_s=lay.ls_tilt)
    same_ls = np.array_equal(ls_tr, ls_tr2) and np.array_equal(ls_stats, ncs.ls_window_stats(list(ls_tr2), lay.fs))
    shapes = ss_stats.shape == (lay.K, N_SS) and ls_stats.shape == (lay.K_LS, N_LS) and Xt.shape == (lay.K, lay.m)
    finite = np.all(np.isfinite(ss_stats)) and np.all(np.isfinite(ls_stats))
    report("F1 cutting (D-115.3): X_tilde == independent loop (cut from the floor, the replicate's (TL1) by its closed "
           "form, offsets, each group's median) within 1e-12; the replicate's tilts likewise; LS bit for bit; shapes",
           same and same_ls and shapes and finite,
           "max |X_tilde diff| %.1e mV, max |tilt diff| %.1e mV/s, j_floor %d; identical LS %s, shapes %s %s %s"
           % (err, tilt_err, lay.j_floor, same_ls, ss_stats.shape, ls_stats.shape, Xt.shape))


def _f2(lay):
    """Subtraction: mean sigma_hat^2 = sigma_f^2 c(K_g) at a white law, c from an independent Monte Carlo; with
    zero tilts and with tilts of 0.1 mV/s in size (D-115: the offsets removed, the tilt leaves the expectation)."""
    lay0 = zero_tilts(lay)
    lay_t = replace(lay0, ss_tilt=np.where(np.arange(lay0.n_sweeps) % 2 == 0, 0.1, -0.1))
    rng = np.random.default_rng(21)
    n_rep = 40
    vals = np.array([nf.simulate_cell_windows(WHITE, lay0, rng, parts=("ss",))[0][:, 0] ** 2 for _ in range(n_rep)])
    vals_t = np.array([nf.simulate_cell_windows(WHITE, lay_t, rng, parts=("ss",))[0][:, 0] ** 2 for _ in range(n_rep)])
    g = groups_of(lay0)
    K_of = np.empty(lay0.K, dtype=int)
    for idx in g.values():
        K_of[idx] = idx.size
    cs = {K: c_median(int(K)) for K in sorted(set(K_of.tolist()))}
    c_win = np.array([cs[K][0] for K in K_of])
    se_c = math.sqrt(np.mean([cs[K][1] ** 2 for K in K_of]))      # the oracle's own Monte Carlo SE
    expected = WHITE.sigma_f ** 2 * c_win.mean()
    mean_all = vals.mean()
    se = math.sqrt((vals.std(ddof=1) / math.sqrt(vals.size)) ** 2 + (WHITE.sigma_f ** 2 * se_c) ** 2)
    ok1 = abs(mean_all - expected) <= Z_SE * se
    mean_t = vals_t.mean()
    se_t = math.sqrt((vals_t.std(ddof=1) / math.sqrt(vals_t.size)) ** 2 + (WHITE.sigma_f ** 2 * se_c) ** 2)
    ok1 = ok1 and abs(mean_t - expected) <= Z_SE * se_t
    exp_mean_sub = WHITE.sigma_f ** 2 * np.mean(1.0 - 1.0 / K_of)
    ok2 = abs(mean_all - exp_mean_sub) > Z_SE * se                  # the median, not the mean, was subtracted
    agree = all(abs(cs[K][0] - cs[K][2]) <= Z_SE * math.hypot(cs[K][1], cs[K][3]) for K in cs)
    report("F2 subtraction (D-103.1, D-115): pooled mean sigma_hat^2 = sigma_f^2 c(K_g) within 3.5 SE, without and "
           "with 0.1 mV/s tilts; not the mean subtraction's sigma_f^2 (1 - 1/K_g); c's identity = its direct Monte Carlo",
           ok1 and ok2 and agree,
           "mean %.7g, tilted %.7g, expected %.7g (mean subtraction %.7g) SE %.2g; c(K) %s"
           % (mean_all, mean_t, expected, exp_mean_sub, se,
              {K: "%.5f (direct %.5f)" % (cs[K][0], cs[K][2]) for K in cs}))


def _f3(lay):
    """CRN: identical twice, different at another seed, continuous in theta."""
    cfg = nf.NoiseFitConfig(M_binding=60)
    q1, pool_ss, pool_ls = nf.binding_function(THETA_CF, lay, cfg, 5, return_pool=True)
    q2 = nf.binding_function(THETA_CF, lay, cfg, 5)
    q3 = nf.binding_function(THETA_CF, lay, cfg, 6)
    law_eps = nl.NoiseLaw.from_array(THETA_CF.as_array() * (1 + 1e-6))
    q4 = nf.binding_function(law_eps, lay, cfg, 5)
    se_mc = np.concatenate([pool_ss.std(axis=0, ddof=1) * 1.2533 / math.sqrt(pool_ss.shape[0]),
                            pool_ls.std(axis=0, ddof=1) * 1.2533 / math.sqrt(pool_ls.shape[0])])
    ident = np.array_equal(q1, q2)
    diff = not np.array_equal(q1, q3)
    cont = np.all(np.abs(q4 - q1) < 0.1 * se_mc)
    # determinism of Sigma_hat on one seed
    s1 = nf.simulated_covariance(THETA_CF, lay, 4, np.random.default_rng(9))
    s2 = nf.simulated_covariance(THETA_CF, lay, 4, np.random.default_rng(9))
    det = np.array_equal(s1.vectors, s2.vectors)
    report("F3 CRN: binding bit-identical at one seed, different at another, continuous in theta; Sigma deterministic",
           ident and diff and cont and det,
           "identical %s differs %s max|dq|/SE_MC %.3g Sigma deterministic %s" % (ident, diff, float(np.max(np.abs(q4 - q1) / se_mc)), det))


def _f4(lay):
    """Sigma_hat at the white law against the asymptotic variance of the median."""
    lay0 = zero_tilts(lay)
    R = 120
    sig = nf.simulated_covariance(WHITE, lay0, R, np.random.default_rng(31), per_sweep=True)
    m, K = lay0.m, lay0.K
    g = groups_of(lay0)
    c_of = {int(idx.size): c_median(int(idx.size), n_draw=200_000)[0] for idx in g.values()}
    c_bar = float(np.mean([c_of[int(idx.size)] for idx in g.values() for _ in idx]))
    var_sd_window = c_bar * WHITE.sigma_f ** 2 / (2.0 * (m - 1))   # Var(sigma_hat) of one subtracted white window
    expected = (math.pi / 2.0) * var_sd_window / K                   # the median over K windows
    got = sig.Sigma[0, 0]
    ok1 = abs(got / expected - 1.0) <= 0.4
    ls_rows = sig.per_sweep[int(lay0.ls_sweep_number[0])]
    var_ls = ls_rows[:, 0].var(ddof=1)
    exp_ls = WHITE.sigma_f ** 2 / (2.0 * (lay0.n_ls - 1))
    ok2 = abs(var_ls / exp_ls - 1.0) <= 0.4
    S = sig.Sigma
    spd = np.allclose(S, S.T) and np.all(np.linalg.eigvalsh(S) > 0)
    kappa = nf.correlation_condition_number(S)
    keys = set(sig.per_sweep)
    want = set(int(s) for s in lay0.sweep_numbers) | set(int(s) for s in lay0.ls_sweep_number)
    shapes = all(sig.per_sweep[s].shape == (R, N_SS) for s in lay0.sweep_numbers) and \
        all(sig.per_sweep[int(s)].shape == (R, N_LS) for s in lay0.ls_sweep_number) and \
        sig.Sigma.shape == (N_STATS, N_STATS)
    report("F4 Sigma_hat: Var(ss_sd) and the LS single-window Var(sd) within 40 % of the white-law values; SPD; per-sweep vectors",
           ok1 and ok2 and spd and np.isfinite(kappa) and keys == want and shapes,
           "ss_sd var ratio %.3f, ls_sd var ratio %.3f, SPD %s, kappa %.3g, sweeps %s" % (got / expected, var_ls / exp_ls, spd, kappa, sorted(keys)))


def _f5(lay, nwin, truth):
    """The real tilts in the simulator (D-107.2)."""
    # (a) the LS traces carry b_k (t - t_bar) exactly: the same draws with and without the tilts
    _s, _l, _x, tr_b = nf.simulate_cell_windows(THETA_CF, lay, np.random.default_rng(5), parts=("ls",), return_windows=True)
    _s, _l, _x, tr_0 = nf.simulate_cell_windows(THETA_CF, zero_tilts(lay), np.random.default_rng(5), parts=("ls",),
                                                return_windows=True)
    t_c = (np.arange(lay.n_ls) - 0.5 * (lay.n_ls - 1)) * lay.dt
    err_a = float(np.max(np.abs((tr_b - tr_0) - lay.ls_tilt[:, None] * t_c[None, :])))
    ok_a = err_a <= 1e-12
    # (b) at a noise-free law the binding pool's LS row i is the ramp of LS sweep i mod K_LS
    cfg = nf.NoiseFitConfig(M_binding=2 * lay.K_LS + 1)
    _q, _ps, pool_ls = nf.binding_function(SILENT, lay, cfg, 3, which="ls", return_pool=True)
    n = lay.n_ls
    want = np.abs(lay.ls_tilt[np.arange(pool_ls.shape[0]) % lay.K_LS]) * lay.dt * math.sqrt(n * (n + 1) / 12.0)
    ok_b = pool_ls.shape[0] == 3 * lay.K_LS and np.allclose(pool_ls[:, 0], want, rtol=1e-12, atol=0)
    # (c) Block 3's SS reading (TL1) on the simulator's windows recovers each sweep's planted tilt on average
    rng = np.random.default_rng(55)
    n_rep = 100
    reads = []
    for _ in range(n_rep):
        out = nf.simulate_cell_windows(THETA_CF, lay, rng, parts=("ss",), return_tilts=True)
        reads.append([out[-1][int(s)] for s in lay.sweep_numbers])      # the replicate's own (TL1), D-115.3
    reads = np.array(reads)
    z = (reads.mean(axis=0) - lay.ss_tilt) / (reads.std(axis=0, ddof=1) / math.sqrt(n_rep))
    ok_c = bool(np.all(np.abs(z) <= Z_SE))
    report("F5 real tilts (D-107.2): LS traces carry b (t - t_bar) exactly; the binding pool cycles the LS tilts; "
           "each replicate's own (TL1) recovers each SS sweep's tilt", ok_a and ok_b and ok_c,
           "max |diff - b (t - t_bar)| %.1e mV; pool rows %d, ramp SDs exact %s; SS tilts %s mV/s, z %s"
           % (err_a, pool_ls.shape[0], ok_b, np.round(lay.ss_tilt, 4).tolist(), np.round(z, 2).tolist()))


def _f6(lay):
    """Self-consistency of the inner solve: q_hat := q_sim(theta*) exactly."""
    cfg = nf.NoiseFitConfig(M_binding=60, max_nfev=10)
    seed = 77
    q_star, pool_ss, pool_ls = nf.binding_function(THETA_CF, lay, cfg, seed, return_pool=True)
    se = np.concatenate([pool_ss.std(axis=0, ddof=1) * 1.2533 / math.sqrt(lay.K),
                         pool_ls.std(axis=0, ddof=1) * 1.2533 / math.sqrt(lay.K_LS)])
    mask = np.isfinite(q_star) & (se > 0)
    C = np.diag(se[mask])
    starts = [replace(THETA_CF, tau_s=THETA_CF.tau_s / 2.0), replace(THETA_CF, tau_s=THETA_CF.tau_s * 2.0)]
    t0 = time.time()
    res = nf.solve_inner(q_star, C, starts, lay, cfg, seed, mask)
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
    tol = F7_TOL_U
    ok_u = np.all(du <= tol)
    # J reported = |C^-1 (q_sim - q_hat)|^2 with the final Sigma_hat and q_sim at the solution
    r = (res.q_sim - res.q_hat)[np.isfinite(res.SE)]
    J_check = float(r @ np.linalg.solve(res.Sigma, r))
    ok_J = abs(J_check - res.J) <= 1e-8 * max(1.0, res.J)
    # q_sim is the binding function at the reported law: recomputed here on the cell's binding seed (CRN), bit for bit
    seed_binding = nf._child_seeds(cfg.seed, lay.specimen_id, 3)[0]
    ok_q = np.array_equal(nf.binding_function(res.law, lay, cfg, seed_binding), res.q_sim)
    ok_J = ok_J and ok_q
    ok_tilt = (res.ss_tilt == {int(s): float(b) for s, b in zip(lay.sweep_numbers, lay.ss_tilt)}
               and res.ls_tilt == {int(s): float(b) for s, b in zip(lay.ls_sweep_number, lay.ls_tilt)}
               and not any(f.startswith("tilt_") for f in res.flags) and not hasattr(res, "sigma_b"))
    ok_fields = (ok_J and res.n_solves == len(res.history) and res.n_solves >= 2 and np.isfinite(res.J) and np.isfinite(res.kappa)
                 and res.Sigma.shape == (N_STATS, N_STATS) and res.dof == N_STATS - N_PARAMS
                 and res.u.shape == (N_PARAMS,) and 0.0 <= res.p_value <= 1.0
                 and res.stats_dropped == () and res.labels_kept == ncs.STAT_LABELS and np.all(np.isfinite(res.SE))
                 and np.all(np.isfinite(res.q_sim)) and len(res.per_sweep_cov) == 7
                 and res.history[0]["weight"] == "W0" and res.history[-1]["weight"] == "final")
    try:
        txt = json.dumps(res.to_dict())
        back = json.loads(txt)
        law_keys = ("sigma_f_mV", "sigma_m_mV", "tau_m_s", "sigma_s_mV", "tau_s_s")      # SPEC 2.6 (g)
        ok_json = (tuple(back["law"]) == law_keys
                   and np.array_equal(np.array([back["law"][k] for k in law_keys]), res.law.as_array())
                   and back["n_solves"] == res.n_solves and back["dof"] == N_STATS - N_PARAMS
                   and len(back["Sigma"]) == N_STATS and len(back["u"]) == N_PARAMS
                   and len(back["history"]) == res.n_solves
                   and "sigma_b_mV_per_s" not in back and len(back["ls_tilt_mV_per_s"]) == lay.K_LS)
    except Exception as exc:  # noqa: BLE001
        ok_json = False
        lines.append("to_dict failed: %r" % (exc,))
    report("F7 reduced fit: |u_hat - u*| within %s per coordinate of u; q_sim = q_sim(theta_hat), J = r^T Sigma^-1 r there; "
           "dof = d - 5; the layout's tilts carried, no tilt estimate; every field; the law dict of 2.6 (g); JSON round trip"
           % (tuple(float(x) for x in F7_TOL_U),),
           ok_u and ok_tilt and ok_fields and ok_json,
           "du %s theta_hat %s J %.3g (recomputed %.3g) p %.3f kappa %.3g solves %d converged %s flags %s (%.0f s)"
           % (np.array2string(du, precision=3), res.law, res.J, J_check, res.p_value, res.kappa, res.n_solves,
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
    """Masks: two NaN statistics are dropped everywhere; dof = d_eff - 5 (SPEC 2.6 (d))."""
    q = tg.q_hat.copy()
    j1 = ncs.STAT_LABELS.index("ss_rho_10ms")
    j2 = ncs.STAT_LABELS.index("ls_last_rho_10ms")
    q[j1] = np.nan; q[j2] = np.nan
    tg2 = nf.CellTarget(q_hat=q, per_window_ss=tg.per_window_ss, per_window_ls=tg.per_window_ls)
    cfg = nf.NoiseFitConfig(M_binding=60, R_iter=20, R_final=20, B_boot=200, k_max=0, max_nfev=1)
    t0 = time.time()
    res = nf.fit_cell(tg2, lay, cfg)
    d_eff = N_STATS - 2
    ok = (res.Sigma.shape == (d_eff, d_eff) and set(res.stats_dropped) == {"ss_rho_10ms", "ls_last_rho_10ms"}
          and len(res.labels_kept) == d_eff and res.dof == d_eff - N_PARAMS
          and np.isnan(res.SE[j1]) and np.isnan(res.SE[j2])
          and any(f.startswith("stats_dropped:") for f in res.flags) and "no_plateau" in res.flags
          and res.n_solves == 2)
    # d_eff >= 6 required (SPEC 2.6 (d)): a q_hat with 5 finite statistics is refused before any simulation
    q5 = np.full(N_STATS, np.nan)
    keep5 = [ncs.STAT_LABELS.index(l) for l in ("ss_sd", "ss_rho_0.02ms", "ss_rho_1ms", "ls_sd", "ls_last_sd")]
    q5[keep5] = tg.q_hat[keep5]
    try:
        nf.fit_cell(nf.CellTarget(q_hat=q5, per_window_ss=tg.per_window_ss, per_window_ls=tg.per_window_ls), lay, cfg)
        ok_min, why = False, "not refused"
    except ValueError as exc:
        why = str(exc)
        ok_min = "at least %d" % (N_PARAMS + 1) in why
    ok = ok and ok_min
    notes_min = "d_eff 5: %s" % why
    report("F9 masks: two NaN statistics -> d_eff %d, Sigma (%d, %d), dof %d, labels, SE NaN, flag; k_max 0 -> no_plateau; "
           "d_eff 5 refused" % (d_eff, d_eff, d_eff, d_eff - N_PARAMS), ok, "dropped %s dof %d solves %d flags %s; %s (%.0f s)"
           % (res.stats_dropped, res.dof, res.n_solves, res.flags, notes_min, time.time() - t0))


def _f10():
    bad = []
    for name in ("noise_fit.py", "smoke_noise_fit.py"):
        raw = (HERE / name).read_bytes()
        if b"\r" in raw or any(b > 127 for b in raw):
            bad.append(name)
    report("F10 ASCII + LF on the two files", not bad, "ok" if not bad else "bad: %s" % bad)


def _f11(tmp):
    """The layout (D-106, D-107.2): kept windows only; unreadable or missing tilts refused; tilts aligned."""
    notes = []
    # (a) the small-group cell: sweep 93 keeps 3 hyp windows, which leave the layout
    lay_s, _tg, nwin_s, _ls, truth_s = build_cell(str(Path(tmp) / "small"), np.random.default_rng(13),
                                                  dropped=sw.SMALL_DROPPED)
    no_small = not np.any((lay_s.sweep_number == 93) & ~lay_s.polarity_is_dep)
    rows = nf.simulate_cell_windows(THETA_CF, lay_s, np.random.default_rng(3), parts=("ss",))[0].shape[0]
    ok_a = lay_s.K == truth_s["K"] - 3 == nwin_s.ss.n_windows and no_small and rows == lay_s.K
    notes.append("small-group cell: K %d of %d, rows %d" % (lay_s.K, truth_s["K"], rows))
    # (b) the 0.5 s-tail cell: every LS tilt NaN -> refused with the reason
    try:
        build_cell(str(Path(tmp) / "short"), np.random.default_rng(14), ls_tail_n=sw.LS_TAIL_N)
        ok_b, why = False, "not refused"
    except ValueError as exc:
        why = str(exc)
        ok_b = "tail_short" in why
    notes.append("short tail: %s" % why[:90])
    # (c) a tilt missing for a sweep -> refused
    d_base = Path(tmp) / "base"
    ss = nw.load_ss_windows(str(d_base)); ls = nw.load_ls_prewindows(str(d_base))
    nwin = nw.ss_noise_windows(ss)
    ss_t = nwin.tilts                                            # the reading of D-115.2
    ls_t = ncs.ls_sweep_tilts(ls)
    short = replace(ls_t, sweep_number=ls_t.sweep_number[1:], b_mV_per_s=ls_t.b_mV_per_s[1:])
    try:
        nf.CellLayout.from_windows(nwin.ss, ls, nwin.start.t_ss_ms, ss_t, short)
        ok_c = False
    except ValueError as exc:
        ok_c = "no tilt reading" in str(exc)
    # (d) aligned with the sweep numbers
    lay = nf.CellLayout.from_windows(nwin.ss, ls, nwin.start.t_ss_ms, ss_t, ls_t)
    b_ss, b_ls = ss_t.as_dict(), ls_t.as_dict()
    ok_d = (all(lay.ss_tilt[i] == b_ss[s] for i, s in enumerate(lay.sweep_numbers))
            and all(lay.ls_tilt[k] == b_ls[int(s)] for k, s in enumerate(lay.ls_sweep_number)))
    # (e) from_cell takes Block 3's reading (D-115.2), never a re-reading on X_tilde; j_floor is the floor's sample
    lay_c = nf.CellLayout.from_cell(nwin, ls)
    ok_e = (np.array_equal(lay_c.ss_tilt, np.array([nwin.tilts.as_dict()[s] for s in lay_c.sweep_numbers]))
            and lay_c.j_floor == lay_c.pre_n + int(round(nw.WindowConfig().t_ss_floor_ms * 1e-3 * lay_c.fs)))
    report("F11 layout (D-106, D-107.2, D-115.2): kept windows only; an unreadable LS tilt and a missing tilt refused; "
           "tilts aligned; from_cell's SS tilts = Block 3's reading; j_floor",
           ok_a and ok_b and ok_c and ok_d and ok_e,
           "; ".join(notes) + "; missing refused %s; aligned %s; from_cell = nwin.tilts and j_floor %s" % (ok_c, ok_d, ok_e))


def _f12(lay, tg):
    """NoiseFitConfig()'s defaults, the bounds, the coordinates and the starts against SPEC 2.6 (d), (h)."""
    cfg = nf.NoiseFitConfig()
    L = math.log
    notes = []
    # (a) the fields of 2.6 (d), (h), written out here; the fields 2.6 leaves standing (Block 4 entry, 2.3 (d))
    want = {"fd_rel_step": (1e-3, 3e-3, 3e-3, 1e-3, 1e-3), "log_sigma_bounds": (L(1e-4), L(10.0)),
            "log_tau_m_bounds": (L(4e-5), L(1e-3)), "log_tau_s_bounds": (L(1e-3), L(10.0)),
            "M_binding": 3000, "R_iter": 600, "R_final": 3000, "B_boot": 2000, "k_max": 6, "tol_u": 0.01,
            "starts_after_first": 1, "max_nfev": 15, "ftol": 1e-4, "xtol": 1e-3, "gtol": 1e-8}
    bad = [k for k, v in want.items() if getattr(cfg, k, None) != v]
    gone = not hasattr(cfg, "rho_bounds") and not hasattr(cfg, "log_tau_bounds")
    ok_a = not bad and gone
    notes.append("fields %s, old bounds removed %s" % ("ok" if not bad else "differ: %s" % bad, gone))
    # (b) bounds() in the order of u: three SDs [1e-4, 10] mV, tau_m [4e-5, 1e-3] s, tau_s [1e-3, 10] s
    lo, hi = cfg.bounds()
    lo_w = np.array([L(1e-4), L(1e-4), L(4e-5), L(1e-4), L(1e-3)])
    hi_w = np.array([L(10.0), L(10.0), L(1e-3), L(10.0), L(10.0)])
    ok_b = lo.shape == (N_PARAMS,) and hi.shape == (N_PARAMS,) and np.array_equal(lo, lo_w) and np.array_equal(hi, hi_w)
    # (c) the coordinates: u = (log sigma_f, log sigma_m, log tau_m, log sigma_s, log tau_s), by field name
    law = nl.NoiseLaw(0.031, 0.017, 2.5e-4, 0.062, 0.045)
    u = nf.law_to_u(law)
    u_w = np.array([L(law.sigma_f), L(law.sigma_m), L(law.tau_m), L(law.sigma_s), L(law.tau_s)])
    back = nf.u_to_law(u_w)
    ok_c = (u.shape == (N_PARAMS,) and np.array_equal(u, u_w)
            and np.allclose([back.sigma_f, back.sigma_m, back.tau_m, back.sigma_s, back.tau_s],
                            [0.031, 0.017, 2.5e-4, 0.062, 0.045], rtol=1e-14, atol=0))   # exp(log x): roundoff only
    try:
        nf.u_to_law(u_w[:4])
        ok_c = False
    except ValueError:
        pass
    # (d) at_bound names: a residual pulling one coordinate past its bound, the others interior
    names = ("log_sigma_f", "log_sigma_m", "log_tau_m", "log_sigma_s", "log_tau_s")
    hits = []
    for j, target in ((2, L(1e-2)), (1, L(1e-6)), (4, L(100.0))):
        tgt = nf.law_to_u(THETA_CF)
        tgt[j] = target
        r = nf.solve_inner(np.zeros(N_STATS), np.eye(1), [THETA_CF], lay, cfg, 0, np.ones(N_STATS, bool),
                           residual=lambda x, t=tgt: x - t)
        hits.append(r.at_bound)
    ok_d = tuple(nf.U_NAMES) == names and hits == [("log_tau_m",), ("log_sigma_m",), ("log_tau_s",)]
    notes.append("at_bound %s" % hits)
    # (e) the three starts of the first solve: tau_s / 2 and 2 tau_s, clipped into [1e-3, 10] s
    s_mid = nf._starts_around(THETA_CF, 3, cfg)
    s_lo = nf._starts_around(replace(THETA_CF, tau_s=1.5e-3), 3, cfg)
    s_hi = nf._starts_around(replace(THETA_CF, tau_s=8.0), 3, cfg)
    ok_e = ([x.tau_s for x in s_mid] == [THETA_CF.tau_s, THETA_CF.tau_s / 2, THETA_CF.tau_s * 2]
            and all(replace(x, tau_s=THETA_CF.tau_s) == THETA_CF for x in s_mid)
            and abs(s_lo[1].tau_s - 1e-3) <= 1e-15 and s_lo[2].tau_s == 3e-3      # exp(log 1e-3): roundoff only
            and s_hi[1].tau_s == 4.0 and abs(s_hi[2].tau_s - 10.0) <= 1e-12
            and len(nf._starts_around(THETA_CF, 1, cfg)) == 1)
    # (f) the closed-form start: the 4 initial points of 2.6 (d); the lowest-cost solution kept; recovers theta on q_cf
    pts = [(0.035, 0.017, tm, 0.06, ts) for tm in (1e-4, 6e-4) for ts in (0.03, 0.2)]
    ok_pts = sorted(tuple(p.as_array()) for p in nf.CF_INITIAL_POINTS) == sorted(pts)
    # (f1) on the fixture's own q_hat (not attainable by q_cf), W_0 from a window bootstrap: the four single-start
    # solutions differ in their last digits (cost differences of about 1e-9 relative, far above the roundoff of the
    # cost, 1e-15), and closed_form_start must return, bit for bit, the one of lowest cost
    SE_d = np.concatenate([ncs.bootstrap_median_se(tg.per_window_ss, 400, np.random.default_rng(1)),
                           ncs.bootstrap_median_se(tg.per_window_ls, 400, np.random.default_rng(2))])
    mask_d = np.isfinite(tg.q_hat) & np.isfinite(SE_d) & (SE_d > 0)
    cost = lambda lw: float(np.sum(((nf.closed_form_vector(lw, lay) - tg.q_hat)[mask_d] / SE_d[mask_d]) ** 2))
    best = nf.closed_form_start(tg.q_hat, SE_d, mask_d, lay, cfg)
    singles = [nf.closed_form_start(tg.q_hat, SE_d, mask_d, lay, cfg, u0=nf.law_to_u(p)) for p in nf.CF_INITIAL_POINTS]
    c_single = [cost(x) for x in singles]
    ok_f1 = best == singles[int(np.argmin(c_single))]
    # (f2) on q_cf(theta*) (exactly attainable, any positive weights) the start recovers theta* to 1e-3 in u
    # (least_squares' default tolerances, 1e-8 on the cost)
    q_cf = nf.closed_form_vector(THETA_CF, lay)
    SE = 0.02 * np.abs(q_cf) + 1e-4
    du_cf = float(np.max(np.abs(nf.law_to_u(nf.closed_form_start(q_cf, SE, np.isfinite(q_cf), lay, cfg))
                                - nf.law_to_u(THETA_CF))))
    ok_f = ok_pts and ok_f1 and du_cf <= 1e-3
    notes.append("closed form: 4 points %s, best = lowest of %s: %s; on q_cf(theta*) max |du| %.1e"
                 % (ok_pts, ["%.10f" % c for c in c_single], ok_f1, du_cf))
    report("F12 config and coordinates (SPEC 2.6 (d), (h)): NoiseFitConfig() defaults; bounds() (5,); law_to_u / u_to_law "
           "order; at_bound names; starts tau_s/2, 2 tau_s clipped; closed-form start = best of the 4 initial points",
           ok_a and ok_b and ok_c and ok_d and ok_e and ok_f,
           "; ".join(notes) + "; bounds %s coords %s starts %s" % (ok_b, ok_c, ok_e))


def main(only=None):
    t_all = time.time()
    tmp = tempfile.mkdtemp(prefix="smoke_noise_fit_")
    lay, tg, nwin, ls, truth = build_cell(str(Path(tmp) / "base"), np.random.default_rng(2026))
    print("fixture: K %d (dep %d), K_LS %d, t_SS %.1f ms, m %d, L_s %d, SS tilts %s, LS tilts %s mV/s"
          % (lay.K, int(lay.polarity_is_dep.sum()), lay.K_LS, nwin.start.t_ss_ms, lay.m, lay.L_s,
             np.round(lay.ss_tilt, 4).tolist(), np.round(lay.ls_tilt, 4).tolist()))
    checks = [("F1", lambda: _f1(lay, nwin.start.t_ss_ms)), ("F2", lambda: _f2(lay)), ("F3", lambda: _f3(lay)),
              ("F4", lambda: _f4(lay)), ("F5", lambda: _f5(lay, nwin, truth)), ("F6", lambda: _f6(lay)),
              ("F7", lambda: _f7(lay, tg, truth)), ("F8", _f8), ("F9", lambda: _f9(lay, tg)), ("F10", _f10),
              ("F11", lambda: _f11(tmp)), ("F12", lambda: _f12(lay, tg))]
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
