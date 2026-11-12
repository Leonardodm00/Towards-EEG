"""smoke_noise_sweep_fit.py -- Block 9 smoke suite (P1-P11) for noise_sweep_fit.py.

Reduced sizes throughout (M, B_v), on Block 4's smoke fixture read through
Block 3's pipeline (smoke_noise_fit.build_cell: 2 SS sweeps, 92 and 93, of
30 and 33 kept windows; 5 LS pre-windows of 13250 samples, sweeps 54-58;
theta_cf noise; per-sweep tilts planted at sigma_b = 0.1 mV/s and read by
(TL1) and (TL2)), with a stub per-cell result: theta_hat(c) a law set apart
from the fixture's truth theta_cf (sigma_f x 1.05, rho_f + 0.02, sigma_s x
0.8; tau_s as theta_cf's), per_sweep_cov from
`simulated_covariance(per_sweep=True)` at R = 200. Each check prints one
PASS/FAIL line and the suite ends with "SMOKE noise_sweep_fit: n/11 passed".

    python smoke_noise_sweep_fit.py           # about 8 min on one core, no NEURON needed
    python smoke_noise_sweep_fit.py P3,P6     # a subset (mutation runs)

Needs numpy, scipy, statsmodels (P3's oracle). Pure ASCII, LF. Spec:
specs/SPEC.md, Block 9 (test oracles P1-P11).
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
import time
import traceback
import warnings
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.stats import chi2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import noise_cell_stats as ncs      # noqa: E402
import noise_fit as nf              # noqa: E402
import noise_law as nl              # noqa: E402
import noise_sweep_fit as nsf       # noqa: E402
import noise_windows as nw          # noqa: E402
import smoke_noise_fit as sf        # noqa: E402  (the fixture builder)
import smoke_noise_windows as sw    # noqa: E402  (the closed-form slope of (TL1))

RESULTS = []
THETA_CF = sf.THETA_CF
LAW_CELL = nl.NoiseLaw(THETA_CF.sigma_f * 1.05, THETA_CF.rho_f + 0.02, THETA_CF.sigma_s * 0.8, THETA_CF.tau_s)
STAGE_SEED = 20261030
R_STUB = 200
STEP = 0.02                         # central-difference step of the sandwich's Jacobian (2.5 (h))


def report(name, ok, note=""):
    RESULTS.append(bool(ok))
    print("%s  %s  -- %s" % ("PASS" if ok else "FAIL", name, note))
    sys.stdout.flush()


def stub_cell_fit(lay, tg, per_sweep_cov, cfg):
    """A CellFitResult carrying theta_hat(c) = LAW_CELL and the given per-sweep covariances."""
    return nf.CellFitResult(
        specimen_id=lay.specimen_id, law=LAW_CELL, u=nf.law_to_u(LAW_CELL), J=float("nan"), dof=11,
        p_value=float("nan"), kappa=float("nan"), Sigma=np.eye(nf.N_STATS), labels_kept=ncs.STAT_LABELS,
        stats_dropped=(), q_hat=tg.q_hat, q_sim=np.full(nf.N_STATS, np.nan), SE=np.full(nf.N_STATS, np.nan),
        ss_tilt={int(s): float(b) for s, b in zip(lay.sweep_numbers, lay.ss_tilt)},
        ls_tilt={int(s): float(b) for s, b in zip(lay.ls_sweep_number, lay.ls_tilt)},
        n_solves=0, converged=True, flags=("stub",), history=[], per_sweep_cov=per_sweep_cov, config=cfg,
        seconds=0.0)


def all_sweeps(lay):
    return [("ss", int(s)) for s in lay.sweep_numbers] + [("ls", int(s)) for s in lay.ls_sweep_number]


def independent_seeds(specimen_id, protocol, sweep, seed=STAGE_SEED):
    """The spec's formula for (seed_binding_s, seed_v_s), written out."""
    kids = np.random.SeedSequence([seed, int(specimen_id), 0 if protocol == "ss" else 1, int(sweep)]).spawn(2)
    return tuple(int(c.generate_state(1, dtype=np.uint64)[0]) for c in kids)


def law_u3(u, tau_s):
    return nl.NoiseLaw(math.exp(u[0]), float(u[1]), math.exp(u[2]), tau_s)


def sandwich(u3, tau_s, sub, cfg, seed, protocol, mask, Sigma_w, Sigma_star):
    """V = A Sigma* A^T, A = (G^T W G)^{-1} G^T W, W = Sigma_w^{-1}, G the central-difference Jacobian
    of q_sim(. | s) in u_s at u3 (step 0.02), on the kept statistics (2.5 (h))."""
    G = np.empty((int(mask.sum()), 3))
    for j in range(3):
        up, dn = np.array(u3, dtype=float), np.array(u3, dtype=float)
        up[j] += STEP
        dn[j] -= STEP
        qp = nf.binding_function(law_u3(up, tau_s), sub, cfg, seed, which=protocol)[mask]
        qm = nf.binding_function(law_u3(dn, tau_s), sub, cfg, seed, which=protocol)[mask]
        G[:, j] = (qp - qm) / (2.0 * STEP)
    W = np.linalg.inv(Sigma_w)
    A = np.linalg.solve(G.T @ W @ G, G.T @ W)
    return A @ Sigma_star @ A.T


def protocol_block(Sigma_s, protocol, mask):
    """The kept part of a protocol-sized covariance, for the oracles."""
    sl = slice(0, nf.N_SS) if protocol == "ss" else slice(nf.N_SS, nf.N_STATS)
    return np.asarray(Sigma_s)[np.ix_(mask[sl], mask[sl])]


# ---------------------------------------------------------------------------
def _p1(fx):
    """The restriction: the sweep's windows and tilt only; the simulator on it equals an independent loop."""
    lay, notes, ok = fx.lay, [], True
    cfgw = nw.WindowConfig()
    jf = lay.pre_n + int(round(cfgw.t_ss_floor_ms * 1e-3 * lay.fs))
    j0 = lay.pre_n + int(round(fx.nwin.start.t_ss_ms * 1e-3 * lay.fs))
    same_fields = ("specimen_id", "fs", "dt", "n_w", "pre_n", "j0", "j_floor", "period_samples", "train_length",
                   "n_ls")
    for i, s in enumerate(lay.sweep_numbers):
        sub = nsf.sweep_layout(lay, "ss", s)
        keep = lay.sweep_number == s
        ok_f = (sub.sweep_numbers == (s,) and sub.K == int(keep.sum()) and np.array_equal(sub.position, lay.position[keep])
                and np.array_equal(sub.polarity_is_dep, lay.polarity_is_dep[keep]) and np.all(sub.sweep_number == s)
                and np.all(sub.sweep_index == 0) and sub.ss_tilt.tolist() == [float(lay.ss_tilt[i])]
                and sub.K_LS == 0 and sub.ls_tilt.size == 0
                and all(getattr(sub, f) == getattr(lay, f) for f in same_fields))
        seed = 31 + int(s)
        _ss, _ls, Xt, _tr = nf.simulate_cell_windows(THETA_CF, sub, np.random.default_rng(seed), parts=("ss",),
                                                     return_windows=True)
        rng2 = np.random.default_rng(seed)                      # the independent loop, on the same draws
        mf = lay.n_w - jf
        units = nl.draw_units(rng2, 1, lay.L_s)
        tr = nl.simulate_sweep(THETA_CF, lay.dt, lay.L_s, units, 0, tilt_mV_per_s=float(lay.ss_tilt[i]))
        pos, dep = lay.position[keep], lay.polarity_is_dep[keep]
        Xf = np.array([tr[int(p) * lay.period_samples + jf:int(p) * lay.period_samples + jf + mf] for p in pos])
        centres = (pos * lay.period_samples + jf + 0.5 * (mf - 1)) * lay.dt
        b = sw._fw_slope(centres, Xf.mean(axis=1), dep)
        tk = pos * lay.period_samples * lay.dt
        X = Xf[:, j0 - jf:] - (b * (tk - tk.mean()))[:, None]
        X2 = X.copy()
        for p in (True, False):
            g = dep == p
            if g.any():
                X2[g] = X[g] - np.median(X[g], axis=0)
        err = float(np.abs(Xt - X2).max())
        ok = ok and ok_f and err <= 1e-12
        notes.append("SS %d: K_s %d, tilt %.4f, fields %s, max|diff| %.1e mV" % (s, sub.K, sub.ss_tilt[0], ok_f, err))
    for k, s in enumerate(lay.ls_sweep_number.tolist()):
        sub = nsf.sweep_layout(lay, "ls", s)
        ok_f = (sub.K == 0 and sub.n_sweeps == 0 and sub.ss_tilt.size == 0 and sub.ls_sweep_number.tolist() == [s]
                and sub.ls_tilt.tolist() == [float(lay.ls_tilt[k])]
                and all(getattr(sub, f) == getattr(lay, f) for f in same_fields))
        seed = 71 + int(s)
        _ss, _ls, _Xt, tr = nf.simulate_cell_windows(THETA_CF, sub, np.random.default_rng(seed), parts=("ls",),
                                                     return_windows=True)
        units = nl.draw_units(np.random.default_rng(seed), 1, lay.n_ls)
        tr2 = nl.simulate_traces(THETA_CF, lay.dt, units.fast, units.slow, tilt_mV_per_s=np.array([lay.ls_tilt[k]]))
        err = float(np.abs(tr - tr2).max())
        ok = ok and ok_f and err <= 1e-12
        if k == 0:
            notes.append("LS %d: tilt %.4f, fields %s, max|diff| %.1e mV" % (s, sub.ls_tilt[0], ok_f, err))
    try:
        nsf.sweep_layout(lay, "ss", 12345)
        refused = False
    except ValueError:
        refused = True
    report("P1 restriction: the sweep's kept windows and tilt only, nothing of the other protocol; the simulator "
           "on it equals an independent loop (floor cut, (TL1) closed form, offsets, per-group median; LS traces)",
           ok and refused, "; ".join(notes) + "; unknown sweep refused %s" % refused)


def _p2(fx):
    """Targets: an SS sweep's medians over its windows, an LS sweep's row; the mask."""
    lay, tg, ok, notes = fx.lay, fx.tg, True, []
    stats_ss = ncs.ss_window_stats(fx.nwin.X_tilde, lay.fs)
    for s in lay.sweep_numbers:
        q, m = nsf.sweep_target(tg, lay, "ss", s)
        ref = np.median(stats_ss[lay.sweep_number == s], axis=0)
        ok = ok and np.array_equal(q[:nf.N_SS], ref) and np.all(np.isnan(q[nf.N_SS:])) \
            and m.tolist() == [True] * nf.N_SS + [False] * nf.N_LS
    for k, s in enumerate(lay.ls_sweep_number.tolist()):
        q, m = nsf.sweep_target(tg, lay, "ls", s)
        ref = ncs.ls_window_stats([fx.ls.pre[k]], lay.fs)[0]
        ok = ok and np.array_equal(q[nf.N_SS:], ref) and np.all(np.isnan(q[:nf.N_SS])) \
            and m.tolist() == [False] * nf.N_SS + [True] * nf.N_LS
    # a statistic without a finite value on the sweep leaves the mask
    pw = tg.per_window_ss.copy()
    s0 = lay.sweep_numbers[0]
    pw[lay.sweep_number == s0, 4] = np.nan
    tg2 = nf.CellTarget(q_hat=tg.q_hat, per_window_ss=pw, per_window_ls=tg.per_window_ls)
    q2, m2 = nsf.sweep_target(tg2, lay, "ss", s0)
    ok_nan = bool(np.isnan(q2[4]) and not m2[4] and m2[:4].all())
    notes.append("NaN statistic dropped %s" % ok_nan)
    report("P2 targets: SS = numpy's median over the sweep's rows of ss_window_stats (exactly); LS = the window's "
           "row; mask False outside the protocol and where not finite", ok and ok_nan, "; ".join(notes))


def _p3():
    """The pooling against statsmodels (raw tau2, the FE and RE means) and (D-076.1) written out."""
    from statsmodels.stats.meta_analysis import combine_effects
    rng = np.random.default_rng(303)
    worst = {"exact": 0.0, "tau2": 0.0, "mu": 0.0, "lam": 0.0}
    n_neg = n_pos = 0
    ok = True
    for _ in range(200):
        k = int(rng.integers(2, 13))
        v = 10.0 ** rng.uniform(-3, -1, k)
        y = rng.normal(-3.0, 0.05, k) + rng.normal(0.0, 1.0, k) * np.sqrt(v) * rng.uniform(0.3, 2.0)
        P = nsf.pool_protocol(y, v)
        with warnings.catch_warnings():                          # statsmodels' sqrt of a negative RE variance
            warnings.simplefilter("ignore", RuntimeWarning)
            ref = combine_effects(y, v, method_re="dl")
        w = 1.0 / v
        denom = w.sum() - (w * w).sum() / w.sum()
        # (P3) in exact rational arithmetic on the same float inputs (fractions): the reference for the
        # roundoff. Tolerances relative to the formula's own scale, (|Q| + k - 1) / denom, since (P3)
        # subtracts k - 1 from Q: 1e-13 against the exact value (float64 roundoff on k-term sums);
        # 1e-10 against statsmodels, whose own value is off the exact one by up to 1.1e-12 of that scale
        # on these draws (measured 2026-11-12: its expected-Q term is summed differently).
        yf, wf = [Fraction(float(a)) for a in y], [1 / Fraction(float(a)) for a in v]
        swf = sum(wf)
        mf = sum(a * b for a, b in zip(wf, yf)) / swf
        exact = float((sum(a * (b - mf) ** 2 for a, b in zip(wf, yf)) - (k - 1))
                      / (swf - sum(a * a for a in wf) / swf))
        scale = abs(exact) + (P.Q + (k - 1)) / denom
        e_x = abs(P.tau2_raw - exact) / scale
        e_t = abs(P.tau2_raw - float(ref.tau2)) / scale
        worst["exact"] = max(worst["exact"], e_x)
        worst["tau2"] = max(worst["tau2"], e_t)
        ok = ok and e_x <= 1e-13 and e_t <= 1e-10
        if P.tau2_raw >= 0:
            n_pos += 1
            e_m = abs(P.mu - float(ref.mean_effect_re))
            ok = ok and P.tau2 == P.tau2_raw
        else:
            n_neg += 1
            e_m = abs(P.mu - float(ref.mean_effect_fe))
            ok = ok and P.tau2 == 0.0 and np.all(P.lam == 0.0)
        worst["mu"] = max(worst["mu"], e_m)
        ok = ok and e_m <= 1e-12
        lam = P.tau2 / (P.tau2 + v)                              # (D-076.1) written out
        ys = P.mu + lam * (y - P.mu)
        e_l = max(float(np.abs(P.lam - lam).max()), float(np.abs(P.y_shrunk - ys).max()))
        worst["lam"] = max(worst["lam"], e_l)
        between = np.all((P.y_shrunk - P.mu) * (y - P.y_shrunk) >= -1e-15)
        ok = ok and e_l <= 1e-12 and between and P.k == k
    P1 = nsf.pool_protocol([-2.9], [0.01])
    ok1 = P1.k == 1 and P1.mu == -2.9 and math.isnan(P1.tau2) and math.isnan(P1.tau2_raw) \
        and np.isnan(P1.lam).all() and P1.y_shrunk.tolist() == [-2.9]
    P0 = nsf.pool_protocol([], [])
    ok0 = P0.k == 0 and all(math.isnan(x) for x in (P0.Q, P0.tau2_raw, P0.tau2, P0.mu)) and P0.lam.size == 0
    try:
        nsf.pool_protocol([1.0, 2.0], [0.1, 0.0])
        ok_v = False
    except ValueError:
        ok_v = True
    report("P3 pooling: tau2_raw = the exact rational (P3) (1e-13 of its scale) and statsmodels' DL tau2 (1e-10); mu = "
           "its RE mean when tau2_raw >= 0, its FE mean and lambda 0 when clipped (1e-12); (D-076.1) written out; "
           "y_shrunk between mu and y; k = 1, k = 0; v > 0",
           ok and ok1 and ok0 and ok_v and n_neg > 10 and n_pos > 10,
           "200 draws (%d clipped, %d not); worst errors tau2 vs exact %.1e, vs statsmodels %.1e, mu %.1e, lambda/y "
           "%.1e; k=1 %s, k=0 %s, v=0 refused %s"
           % (n_neg, n_pos, worst["exact"], worst["tau2"], worst["mu"], worst["lam"], ok1, ok0, ok_v))


def _p4(fx):
    """Self-consistency: q_hat_s := q_sim(theta* | s) exactly; the solve returns u(theta*), J ~ 0."""
    lay, notes, ok = fx.lay, [], True
    cfg = nf.NoiseFitConfig(M_binding=60)
    for pr, s in (("ss", lay.sweep_numbers[0]), ("ls", int(lay.ls_sweep_number[0]))):
        sub = nsf.sweep_layout(lay, pr, s)
        seed_b, _ = nsf.sweep_seeds(STAGE_SEED, lay.specimen_id, pr, s)
        q_star = nf.binding_function(THETA_CF, sub, cfg, seed_b, which=pr)
        mask = np.isfinite(q_star)
        start = nl.NoiseLaw(THETA_CF.sigma_f, THETA_CF.rho_f, THETA_CF.sigma_s * math.exp(0.2), THETA_CF.tau_s)
        t0 = time.time()
        fit = nsf.fit_sweep(q_star, mask, fx.psc[s], sub, start, cfg, seed_b)
        du = np.abs(fit.u - nf.law_to_u(THETA_CF)[:3])
        # J recomputed at the reported solution: (q_sim - q_hat)^T Sigma_s^-1 (q_sim - q_hat)
        r = (nf.binding_function(fit.law, sub, cfg, seed_b, which=pr) - q_star)[mask]
        J_re = float(r @ np.linalg.solve(protocol_block(fx.psc[s], pr, mask), r))
        # J against its recomputation: relative 1e-6 (the same deterministic residual; float64 roundoff of the
        # quadratic form is far below), no absolute floor -- J itself is about 1e-8 here
        ok_s = (np.all(du <= 0.01) and fit.J < 1e-3 and not fit.failed and fit.law.tau_s == THETA_CF.tau_s
                and abs(fit.J - J_re) <= 1e-6 * J_re and fit.d == int(mask.sum()) and fit.dof == fit.d - 3)
        ok = ok and ok_s
        notes.append("%s %d: |du| %s J %.1e (recomputed %.1e) calls %d status %d (%.0f s)"
                     % (pr.upper(), s, np.array2string(du, precision=4), fit.J, J_re, fit.n_calls, fit.status,
                        time.time() - t0))
    report("P4 self-consistency (M = 60): q_hat_s = q_sim(theta* | s); from 0.2 away in log sigma_s the solve "
           "returns u within 0.01, J < 1e-3 (and = r^T Sigma_s^-1 r), tau_s held, not failed; SS and LS",
           ok, "; ".join(notes))


def _p5(fx):
    """Recovery: every sweep fitted at M = 100 from the stub theta_hat(c) lies within 4 linearised SDs of theta_cf."""
    lay, notes, ok = fx.lay, [], True
    cfg = nf.NoiseFitConfig(M_binding=100)
    t0 = time.time()
    star = nf.simulated_covariance(THETA_CF, lay, R_STUB, np.random.default_rng(505), per_sweep=True)
    u_cf = nf.law_to_u(THETA_CF)[:3]
    worst = 0.0
    for pr, s in all_sweeps(lay):
        sub = nsf.sweep_layout(lay, pr, s)
        q, m = nsf.sweep_target(fx.tg, lay, pr, s)
        seed_b, _ = nsf.sweep_seeds(STAGE_SEED, lay.specimen_id, pr, s)
        fit = nsf.fit_sweep(q, m, fx.psc[s], sub, LAW_CELL, cfg, seed_b)
        Sigma_star = np.cov(star.per_sweep[s], rowvar=False, ddof=1)
        V = sandwich(u_cf, THETA_CF.tau_s, sub, cfg, seed_b, pr, m, protocol_block(fx.psc[s], pr, m),
                     protocol_block(Sigma_star, pr, m))
        z = np.abs(fit.u - u_cf) / np.sqrt(np.diag(V))
        r = (nf.binding_function(fit.law, sub, cfg, seed_b, which=pr) - q)[m]
        J_re = float(r @ np.linalg.solve(protocol_block(fx.psc[s], pr, m), r))
        worst = max(worst, float(z.max()))
        ok = ok and np.all(z < 4.0) and not fit.failed and abs(fit.J - J_re) <= 1e-6 * J_re
        notes.append("%s %d z %s J %.3g%s" % (pr.upper(), s, np.array2string(z, precision=2), fit.J,
                                             " FAILED" if fit.failed else ""))
    report("P5 recovery (M = 100): every sweep's |u_hat_s - u(theta_cf)| below 4 linearised SDs (the sandwich at "
           "the same M and seed), none failed; J = r^T Sigma_s^-1 r at the solution (1e-6)", ok,
           "; ".join(notes) + "; worst z %.2f (%.0f s)" % (worst, time.time() - t0))


def _p6(fx):
    """The bootstrap's machinery with a test double for the refit: stream, law, reduction, ddof, the flags."""
    lay, notes = fx.lay, []
    cfg = nf.NoiseFitConfig(M_binding=60)
    stage = nsf.SweepStageConfig(B_v=10, M_refit=40)
    law_fit = nl.NoiseLaw(THETA_CF.sigma_f, THETA_CF.rho_f, THETA_CF.sigma_s * 1.1, LAW_CELL.tau_s)
    calls = []

    def make_double(fail_every=None, fail_all=False):
        counter = [0]

        def double(q_b, m_b, Sigma_s, sub, law_start, cfg_r, seed_r):
            b = counter[0]
            counter[0] += 1
            calls.append((int(seed_r), cfg_r.M_binding, law_start, Sigma_s))
            j = int(np.flatnonzero(m_b)[0])                     # the protocol's first statistic: its SD
            failed = fail_all or (fail_every is not None and b % fail_every == fail_every - 1)
            return SimpleNamespace(failed=failed, u=np.array([np.nan, np.nan, math.log(q_b[j])]))
        return double

    ok = True
    for pr, s in (("ss", lay.sweep_numbers[0]), ("ls", int(lay.ls_sweep_number[0]))):
        sub = nsf.sweep_layout(lay, pr, s)
        q, m = nsf.sweep_target(fx.tg, lay, pr, s)
        seed_b, seed_v = nsf.sweep_seeds(STAGE_SEED, lay.specimen_id, pr, s)
        ok_seeds = (seed_b, seed_v) == independent_seeds(lay.specimen_id, pr, s)
        fit = nsf.SweepFit(protocol=pr, sweep=s, u=nf.law_to_u(law_fit)[:3], law=law_fit, J=0.0, d=int(m.sum()),
                           dof=int(m.sum()) - 3, n_calls=0, status=1, at_bound=(), failed=False, reason="")
        calls.clear()
        boot = nsf.bootstrap_v(fit, m, fx.psc[s], sub, LAW_CELL, cfg, stage, seed_b, seed_v, refit=make_double())
        rng = np.random.default_rng(seed_v)                     # the independent loop
        y_ref = np.empty(stage.B_v)
        for b in range(stage.B_v):
            ss_st, ls_st = nf.simulate_cell_windows(law_fit, sub, rng, parts=(pr,))
            y_ref[b] = math.log(np.median(ss_st[:, 0]) if pr == "ss" else ls_st[0, 0])
        args_ok = all(c[0] == seed_b and c[1] == stage.M_refit and c[2] is LAW_CELL and c[3] is fx.psc[s] for c in calls)
        ok_y = (float(np.abs(boot.y - y_ref).max()) <= 1e-12 and boot.v == float(np.var(y_ref, ddof=1))
                and boot.B == boot.B_ok == stage.B_v and boot.flags == () and len(calls) == stage.B_v)
        ok = ok and ok_seeds and args_ok and ok_y
        notes.append("%s %d: max|y - loop| %.1e, v %.3g, seeds %s, refit args (seed, M_refit, start, Sigma_s) %s"
                     % (pr.upper(), s, float(np.abs(boot.y - y_ref).max()), boot.v, ok_seeds, args_ok))
        if pr == "ss":
            b5 = nsf.bootstrap_v(fit, m, fx.psc[s], sub, LAW_CELL, cfg, stage, seed_b, seed_v,
                                 refit=make_double(fail_every=5))
            keep = np.ones(stage.B_v, dtype=bool)
            keep[4::5] = False
            ok5 = (b5.B_ok == 8 and np.all(np.isnan(b5.y[~keep])) and np.array_equal(b5.y[keep], y_ref[keep])
                   and b5.v == float(np.var(y_ref[keep], ddof=1)) and b5.flags == ("v_refits_failed:2/10",))
            b10 = nsf.bootstrap_v(fit, m, fx.psc[s], sub, LAW_CELL, cfg, stage, seed_b, seed_v,
                                  refit=make_double(fail_every=10))
            ok10 = b10.B_ok == 9 and b10.flags == ()            # exactly 10 %: not "more than 10 %"
            ball = nsf.bootstrap_v(fit, m, fx.psc[s], sub, LAW_CELL, cfg, stage, seed_b, seed_v,
                                   refit=make_double(fail_all=True))
            okall = ball.B_ok == 0 and math.isnan(ball.v) and "v_none" in ball.flags
            ok = ok and ok5 and ok10 and okall
            notes.append("every 5th fails: B_ok %d flags %s; 1 of 10: flags %s; all: v %s flags %s"
                         % (b5.B_ok, b5.flags, b10.flags, ball.v, ball.flags))
            # through run_sweep: the sweep fails with the reason v_none, flagged
            cell = stub_cell_fit(lay, fx.tg, fx.psc, cfg)
            calls.clear()
            row = nsf.run_sweep(cell, fx.tg, lay, cfg, stage, pr, s, refit=make_double(fail_all=True))
            ok_row = (row["failed"] and row["reason"] == "v_none" and row["B_ok"] == 0
                      and "sweep_fit_failed:ss:%d:v_none" % s in row["flags"] and all(c[0] == seed_b for c in calls))
            ok = ok and ok_row
            notes.append("run_sweep: reason %s flags %s" % (row["reason"], row["flags"]))
    report("P6 bootstrap machinery (test double): y = an independent loop over seed_v_s at theta_hat(c, s); v with "
           "ddof 1; refits get seed_binding_s, M_refit, theta_hat(c), Sigma_s; 2/10 failed -> flag, 1/10 -> none; all "
           "failed -> v NaN, v_none", ok, "; ".join(notes))


def _p7(fx):
    """The bootstrap against the linearisation: B_v = 60 real refits at M = 100, first SS sweep."""
    lay = fx.lay
    s = lay.sweep_numbers[0]
    cfg = nf.NoiseFitConfig(M_binding=100)
    stage = nsf.SweepStageConfig(B_v=60, M_refit=100)
    sub = nsf.sweep_layout(lay, "ss", s)
    q, m = nsf.sweep_target(fx.tg, lay, "ss", s)
    seed_b, seed_v = nsf.sweep_seeds(STAGE_SEED, lay.specimen_id, "ss", s)
    t0 = time.time()
    fit = nsf.fit_sweep(q, m, fx.psc[s], sub, LAW_CELL, cfg, seed_b)
    boot = nsf.bootstrap_v(fit, m, fx.psc[s], sub, LAW_CELL, cfg, stage, seed_b, seed_v)
    t_boot = time.time() - t0
    rng = np.random.default_rng(707)                          # Sigma*_s at theta_hat(c, s): 300 replicate sweeps
    reps = np.array([np.median(nf.simulate_cell_windows(fit.law, sub, rng, parts=("ss",))[0], axis=0)
                     for _ in range(300)])
    Sigma_star = np.cov(reps, rowvar=False, ddof=1)
    V = sandwich(fit.u, LAW_CELL.tau_s, sub, cfg, seed_b, "ss", m, protocol_block(fx.psc[s], "ss", m),
                 Sigma_star[np.ix_(m[:nf.N_SS], m[:nf.N_SS])])
    v_lin = float(V[2, 2])
    dof = boot.B_ok - 1
    lo, hi = v_lin * chi2.ppf(0.0005, dof) / dof, v_lin * chi2.ppf(0.9995, dof) / dof
    ok = (not fit.failed) and boot.B_ok >= 54 and lo <= boot.v <= hi
    report("P7 bootstrap vs linearisation (SS sweep %d, B_v = 60, M = 100): v inside v_lin chi2_q(B_ok-1)/(B_ok-1), "
           "q = 0.0005, 0.9995" % s, ok,
           "v %.4g (B_ok %d) v_lin %.4g, interval [%.4g, %.4g], ratio %.2f; fit u %s (%.0f s)"
           % (boot.v, boot.B_ok, v_lin, lo, hi, boot.v / v_lin, np.array2string(fit.u, precision=4), t_boot))


def _p8(fx):
    """A failed sweep: at a bound, not bootstrapped, out of the pooling, its twin theta_hat(c); no_convergence; stats."""
    lay, notes = fx.lay, []
    cfg = nf.NoiseFitConfig(M_binding=30)
    stage = nsf.SweepStageConfig(B_v=2, M_refit=30)
    cell = stub_cell_fit(lay, fx.tg, fx.psc, cfg)
    s_bad, s_ok = lay.sweep_numbers[0], lay.sweep_numbers[1]
    pw = fx.tg.per_window_ss.copy()
    # 1400 mV: 100 times above the largest SS SD a law in the bounds produces (about 14 mV at sigma_f = sigma_s
    # = 10 mV). A target 100 times BELOW the smallest is not used: there the whitened cost is flat near the
    # lower bounds and the solve stops by xtol short of them (checked 2026-11-12: u = (-8.3, 0.19, -7.9)).
    pw[lay.sweep_number == s_bad, 0] = 1400.0
    tg_bad = nf.CellTarget(q_hat=fx.tg.q_hat, per_window_ss=pw, per_window_ls=fx.tg.per_window_ls)
    t0 = time.time()
    row_bad = nsf.run_sweep(cell, tg_bad, lay, cfg, stage, "ss", s_bad)
    row_ok = nsf.run_sweep(cell, tg_bad, lay, cfg, stage, "ss", s_ok)
    res = nsf.pool_cell([row_bad, row_ok], LAW_CELL, lay.specimen_id)
    flag = [f for f in row_bad["flags"] if f.startswith("sweep_fit_failed:ss:%d:at_bound:" % s_bad)]
    ok_a = (row_bad["failed"] and str(row_bad["reason"]).startswith("at_bound:") and len(flag) == 1
            and row_bad["B_ok"] == 0 and row_bad["bootstrap_y"] == [] and not row_ok["failed"]
            and res.pools["ss"].sweeps == (s_ok,) and res.pools["ss"].k == 1
            and res.twin_laws[("ss", s_bad)] == LAW_CELL and math.isnan(res.rows[0]["lambda"])
            and res.twin_laws[("ss", s_ok)] == nl.NoiseLaw(row_ok["sigma_f_mV"], row_ok["rho_f"],
                                                         row_ok["sigma_s_mV"], LAW_CELL.tau_s))
    notes.append("at a bound: reason %s, flag %s, pooled sweeps %s, twin = theta_hat(c) %s (%.0f s)"
                 % (row_bad["reason"], flag, res.pools["ss"].sweeps, res.twin_laws[("ss", s_bad)] == LAW_CELL,
                    time.time() - t0))
    sub = nsf.sweep_layout(lay, "ss", s_ok)
    q, m = nsf.sweep_target(fx.tg, lay, "ss", s_ok)
    seed_b, _ = nsf.sweep_seeds(STAGE_SEED, lay.specimen_id, "ss", s_ok)
    f1 = nsf.fit_sweep(q, m, fx.psc[s_ok], sub, LAW_CELL, nf.NoiseFitConfig(M_binding=30, max_nfev=1), seed_b)
    ok_b = f1.failed and f1.reason == "no_convergence" and f1.status <= 0
    notes.append("max_nfev 1: status %d reason %s" % (f1.status, f1.reason))
    pw3 = fx.tg.per_window_ss.copy()
    pw3[lay.sweep_number == s_ok, 3:] = np.nan          # 3 of 5 statistics left: d_s < 4
    tg3 = nf.CellTarget(q_hat=fx.tg.q_hat, per_window_ss=pw3, per_window_ls=fx.tg.per_window_ls)
    row3 = nsf.run_sweep(cell, tg3, lay, cfg, stage, "ss", s_ok)
    ok_c = row3["failed"] and row3["reason"] == "stats" and row3["n_calls"] == 0 and row3["d"] == 3 \
        and "sweep_fit_failed:ss:%d:stats" % s_ok in row3["flags"]
    notes.append("d_s 3: reason %s calls %d" % (row3["reason"], row3["n_calls"]))
    report("P8 a failed sweep: an SD 100x above the bounds' largest -> at_bound, flagged, not bootstrapped, out of "
           "S'_pr, twin = theta_hat(c); max_nfev 1 -> no_convergence; d_s < 4 -> stats", ok_a and ok_b and ok_c,
           "; ".join(notes))


ROW_FIELDS = ("protocol", "sweep", "K_s", "n_LS", "tilt_mV_per_s", "d", "sigma_f_mV", "rho_f", "sigma_s_mV",
              "tau_s_s", "J", "dof", "n_calls", "status", "at_bound", "failed", "reason", "v_log_sigma_s", "B_ok",
              "n_refits_failed", "lambda", "sigma_s_tilde_mV", "twin_law", "flags")
TABLE_COLUMNS = ("noise_sweep_sigma_f_mV", "noise_sweep_rho_f", "noise_sweep_sigma_s_raw_mV", "noise_sweep_sigma_s_mV",
                 "noise_sweep_v_log_sigma_s", "noise_sweep_w", "noise_sweep_flags")
CELL_COLUMNS = ("noise_cell_tau2_log_sigma_s_ss", "noise_cell_tau2_log_sigma_s_ls", "noise_cell_mu_log_sigma_s_ss",
                "noise_cell_mu_log_sigma_s_ls")


def _comparable(res):
    d = res.to_dict()
    d.pop("seconds")
    for r in d["rows"]:
        r.pop("seconds")
    return json.dumps(d, sort_keys=True)


def _p9(fx):
    """The stage end to end on the fixture."""
    lay = fx.lay
    cfg = nf.NoiseFitConfig(M_binding=60)
    stage = nsf.SweepStageConfig(B_v=4, M_refit=60)
    cell = stub_cell_fit(lay, fx.tg, fx.psc, cfg)
    t0 = time.time()
    res = nsf.fit_sweeps(cell, fx.tg, lay, cfg, stage)
    t1 = time.time() - t0
    order = [(r["protocol"], r["sweep"]) for r in res.rows]
    ok_rows = order == all_sweeps(lay) and all(all(f in r for f in ROW_FIELDS) for r in res.rows)
    ok_pool = True
    for pr in nsf.PROTOCOLS:
        rr = [r for r in res.rows if r["protocol"] == pr and not r["failed"]]
        P = nsf.pool_protocol([r["log_sigma_s"] for r in rr], [r["v_log_sigma_s"] for r in rr])
        Q = res.pools[pr]
        ok_pool = ok_pool and Q.k == len(rr) and Q.sweeps == tuple(r["sweep"] for r in rr) \
            and np.array_equal(Q.y_shrunk, P.y_shrunk) and (Q.mu == P.mu or (math.isnan(Q.mu) and math.isnan(P.mu)))
        for j, r in enumerate(rr):
            twin = res.twin_laws[(pr, r["sweep"])]
            ok_pool = ok_pool and twin == nl.NoiseLaw(r["sigma_f_mV"], r["rho_f"], math.exp(P.y_shrunk[j]),
                                                      cell.law.tau_s)
    n_failed = sum(bool(r["failed"]) for r in res.rows)
    t0 = time.time()
    rows = [nsf.run_sweep(cell, fx.tg, lay, cfg, stage, pr, s) for pr, s in all_sweeps(lay)]
    res2 = nsf.pool_cell(rows, cell.law, lay.specimen_id, stage_cfg=stage)
    t2 = time.time() - t0
    ok_same = _comparable(res) == _comparable(res2)
    back = json.loads(json.dumps(res.to_dict()))
    ok_json = back == res.to_dict() or json.dumps(back, sort_keys=True) == json.dumps(res.to_dict(), sort_keys=True)
    tab = res.table_rows()
    cc = res.cell_columns()
    ok_tab = (len(tab) == len(res.rows) and all(all(c in t for c in TABLE_COLUMNS) for t in tab)
              and all(c in cc for c in CELL_COLUMNS)
              and all(t["noise_sweep_sigma_s_mV"] == res.twin_laws[(r["protocol"], r["sweep"])].sigma_s
                      for t, r in zip(tab, res.rows)))
    ok = ok_rows and ok_pool and ok_same and ok_json and ok_tab and n_failed == 0
    report("P9 the stage end to end (M = 60, B_v = 4, M_refit = 60): a row per sweep of S(c) with every field; each "
           "protocol's pool = pool_protocol on the rows; theta_tilde by (P0); run_sweep + pool_cell == fit_sweeps "
           "(bit for bit, two runs); JSON round trip; the table's columns", ok,
           "rows %s fields %s pools %s same %s json %s table %s failed %d; SS tau2 %.3g mu %.4f, LS tau2 %.3g mu %.4f "
           "(%.0f s + %.0f s)" % (len(res.rows), ok_rows, ok_pool, ok_same, ok_json, ok_tab, n_failed,
                                  res.pools["ss"].tau2, res.pools["ss"].mu, res.pools["ls"].tau2,
                                  res.pools["ls"].mu, t1, t2))


def _p10():
    bad = []
    for name in ("noise_sweep_fit.py", "smoke_noise_sweep_fit.py"):
        raw = (HERE / name).read_bytes()
        if b"\r" in raw or any(b > 127 for b in raw):
            bad.append(name)
    report("P10 ASCII + LF on the two files", not bad, "ok" if not bad else "bad: %s" % bad)


def _p11():
    c = nsf.SweepStageConfig()
    ok = c.B_v == 50 and c.M_refit == 1000 and c.max_failed_refit_frac == 0.1 and c.min_stats == 4 \
        and c.seed == STAGE_SEED
    report("P11 the decided defaults (D-122.1-D-122.2): B_v 50, M_refit 1000, max_failed_refit_frac 0.1 "
           "(and min_stats 4, seed 20261030)", ok, "%s" % (c,))


def main(only=None):
    t_all = time.time()
    tmp = tempfile.mkdtemp(prefix="smoke_noise_sweep_fit_")
    lay, tg, nwin, ls, truth = sf.build_cell(str(Path(tmp) / "base"), np.random.default_rng(2026))
    t0 = time.time()
    psc = None
    if not only or only & {"P4", "P5", "P6", "P7", "P8", "P9"}:
        sig = nf.simulated_covariance(LAW_CELL, lay, R_STUB, np.random.default_rng(909), per_sweep=True)
        psc = {s: np.cov(v, rowvar=False, ddof=1) for s, v in sig.per_sweep.items()}
    fx = SimpleNamespace(lay=lay, tg=tg, nwin=nwin, ls=ls, truth=truth, psc=psc)
    print("fixture: SS sweeps %s (K_s %s), LS sweeps %s (n_LS %d), tilts SS %s LS %s mV/s; stub per_sweep_cov at "
          "R = %d (%.0f s)" % (lay.sweep_numbers, [int((lay.sweep_number == s).sum()) for s in lay.sweep_numbers],
                              lay.ls_sweep_number.tolist(), lay.n_ls, np.round(lay.ss_tilt, 4).tolist(),
                              np.round(lay.ls_tilt, 4).tolist(), R_STUB, time.time() - t0))
    checks = [("P1", lambda: _p1(fx)), ("P2", lambda: _p2(fx)), ("P3", _p3), ("P4", lambda: _p4(fx)),
              ("P5", lambda: _p5(fx)), ("P6", lambda: _p6(fx)), ("P7", lambda: _p7(fx)), ("P8", lambda: _p8(fx)),
              ("P9", lambda: _p9(fx)), ("P10", _p10), ("P11", _p11)]
    for name, fn in checks:
        if only and name not in only:
            continue
        try:
            fn()
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            report("%s crashed" % name, False, "see traceback")
    n_ok = sum(RESULTS)
    print("SMOKE noise_sweep_fit: %d/%d passed  (%.0f s)" % (n_ok, len(RESULTS), time.time() - t_all))
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    # optional: python smoke_noise_sweep_fit.py P3,P6   (a subset, for mutation runs)
    sys.exit(main(set(sys.argv[1].split(",")) if len(sys.argv) > 1 else None))
