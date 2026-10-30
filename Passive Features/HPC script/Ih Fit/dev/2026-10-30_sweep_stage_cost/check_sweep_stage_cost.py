"""check_sweep_stage_cost.py -- the cost of Block 9's ("Block 4b") per-sweep stage, measured.

Standalone dev check (not pipeline code), written 2026-10-30 by the noise-fit chat
for the SPEC entry of the per-sweep stage (D-116) and the questions put to the user
on its bootstrap. It times, on Block 3's writer-like fixture cell (2 SS sweeps of 37
pulses with 11 planted drops, K = 63; 5 LS pre-windows of 13250 samples; theta_cf
noise; per-sweep tilts at sigma_b = 0.1 mV/s), through Block 4's own functions:

  E  one evaluation of the per-sweep binding function q_sim(theta | s) at M = 3000:
     an SS sweep alone (ceil(M / K_s) replicate sweeps, the SS part of
     `noise_fit.binding_function` on a layout restricted to the sweep) and an LS
     sweep alone (M pre-windows with that sweep's tilt); the cell's own SS and LS
     parts for comparison;
  R  one bootstrap replicate of a sweep (simulated, reduced to its statistic vector);
  S  one per-sweep solve as D-116.1 states it -- u_s = (log sigma_f, rho_f,
     log sigma_s), tau_s held, W_s = the inverse of the sweep's simulated covariance
     (R = 200 replicate cells at the start law), `least_squares(method='trf')` with
     Block 4's steps and tolerances -- at a reduced M = 300, counting every call of
     the residual (the Jacobian's included); started from a "cell law" set apart
     from the fixture's truth (sigma_f x 1.05, rho_f + 0.02, sigma_s x 0.8), as
     theta_hat(c) is apart from a sweep's own estimate. With common random numbers
     the binding function is smooth in theta, so the count carries to M = 3000.

The layout restriction and the solve are written here for the measurement only;
Block 9's code is not written (its SPEC entry is drafted).

    python check_sweep_stage_cost.py      # about 3 min on one core

Needs the Ih Fit folder two levels up (or TEEG_IH_FIT). Pure ASCII, LF.
"""
from __future__ import annotations

import math
import os
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy import linalg, optimize

IH_FIT = Path(os.environ.get("TEEG_IH_FIT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(IH_FIT))

import noise_cell_stats as ncs      # noqa: E402
import noise_fit as nf              # noqa: E402
import noise_law as nl              # noqa: E402
import smoke_noise_fit as sf        # noqa: E402  (the fixture builder)

SEED = 20261053


def sweep_layout(lay, protocol, sweep):
    """The cell's layout restricted to one sweep (measurement only)."""
    if protocol == "ss":
        keep = lay.sweep_number == sweep
        i = lay.sweep_numbers.index(int(sweep))
        return replace(lay, sweep_numbers=(int(sweep),), sweep_index=np.zeros(int(keep.sum()), dtype=int),
                       position=lay.position[keep], polarity_is_dep=lay.polarity_is_dep[keep],
                       sweep_number=lay.sweep_number[keep], ss_tilt=lay.ss_tilt[[i]],
                       ls_sweep_number=np.zeros(0, dtype=int), ls_tilt=np.zeros(0))
    k = list(lay.ls_sweep_number.tolist()).index(int(sweep))
    return replace(lay, sweep_numbers=(), sweep_index=np.zeros(0, dtype=int), position=np.zeros(0, dtype=int),
                   polarity_is_dep=np.zeros(0, dtype=bool), sweep_number=np.zeros(0, dtype=int),
                   ss_tilt=np.zeros(0), ls_sweep_number=lay.ls_sweep_number[[k]], ls_tilt=lay.ls_tilt[[k]])


def timed(fn, reps=2):
    out, best = None, float("inf")
    for _ in range(reps):
        t0 = time.time()
        out = fn()
        best = min(best, time.time() - t0)
    return out, best


def solve_sweep(q_hat_s, Sigma_s, mask, sub, cfg, seed, law_start, tau_s):
    """D-116.1 at the given M (measurement only): returns (u, status, calls, seconds)."""
    C, _ = nf._cholesky_lower(Sigma_s)
    lo, hi = [np.asarray(b)[:3] for b in cfg.bounds()]
    calls = [0]

    def law_of(u):
        return nl.NoiseLaw(math.exp(u[0]), float(u[1]), math.exp(u[2]), tau_s)

    def fun(u):
        calls[0] += 1
        q = nf.binding_function(law_of(u), sub, cfg, seed, which="ss" if mask[0] else "ls")
        return linalg.solve_triangular(C, q[mask] - q_hat_s[mask], lower=True)

    u0 = np.clip(nf.law_to_u(law_start)[:3], lo, hi)
    t0 = time.time()
    res = optimize.least_squares(fun, u0, bounds=(lo, hi), method="trf", diff_step=list(cfg.fd_rel_step[:3]),
                                 ftol=cfg.ftol, xtol=cfg.xtol, gtol=cfg.gtol, max_nfev=cfg.max_nfev)
    return res.x, int(res.status), int(res.nfev), int(res.njev), calls[0], time.time() - t0


def main():
    rng = np.random.default_rng(SEED)
    with tempfile.TemporaryDirectory() as d:
        lay, tg, nwin, ls, truth = sf.build_cell(Path(d), rng)
    law = sf.THETA_CF
    print("fixture: K = %d SS windows in sweeps %s (K_s = %s); K_LS = %d, n_LS = %d; m = %d samples"
          % (lay.K, lay.sweep_numbers, [int((lay.sweep_number == s).sum()) for s in lay.sweep_numbers],
             lay.K_LS, lay.n_ls, lay.m))
    s_ss = lay.sweep_numbers[0]
    s_ls = int(lay.ls_sweep_number[0])
    sub_ss = sweep_layout(lay, "ss", s_ss)
    sub_ls = sweep_layout(lay, "ls", s_ls)
    cfg = nf.NoiseFitConfig()
    seed = 12345

    # --- E: one evaluation at M = 3000
    _q, t_ss = timed(lambda: nf.binding_function(law, sub_ss, cfg, seed, which="ss"))
    _q, t_ls = timed(lambda: nf.binding_function(law, sub_ls, cfg, seed, which="ls"))
    _q, t_cell_ss = timed(lambda: nf.binding_function(law, lay, cfg, seed, which="ss"), reps=1)
    _q, t_cell_ls = timed(lambda: nf.binding_function(law, lay, cfg, seed, which="ls"), reps=1)
    print("E  one evaluation at M = 3000: SS sweep %d alone %.2f s (%d replicate sweeps); LS sweep %d alone %.2f s"
          " (3000 windows); the cell's SS part %.2f s (%d replicate cells x %d sweeps), LS part %.2f s"
          % (s_ss, t_ss, math.ceil(3000 / sub_ss.K), s_ls, t_ls, t_cell_ss, math.ceil(3000 / lay.K),
             lay.n_sweeps, t_cell_ls))

    # --- R: one bootstrap replicate of a sweep (simulation and statistics only)
    r_rng = np.random.default_rng(SEED + 1)
    _o, t_rep_ss = timed(lambda: nf.simulate_cell_windows(law, sub_ss, r_rng, parts=("ss",)), reps=3)
    _o, t_rep_ls = timed(lambda: nf.simulate_cell_windows(law, sub_ls, r_rng, parts=("ls",)), reps=3)
    print("R  one replicate sweep: SS %.3f s, LS %.3f s" % (t_rep_ss, t_rep_ls))

    # --- S: one per-sweep solve at M = 300 (the call count carries to M = 3000)
    cfg300 = replace(cfg, M_binding=300)
    law_cell = nl.NoiseLaw(law.sigma_f * 1.05, law.rho_f + 0.02, law.sigma_s * 0.8, law.tau_s)
    t0 = time.time()
    sig = nf.simulated_covariance(law_cell, lay, 200, np.random.default_rng(SEED + 2), per_sweep=True)
    t_sig = time.time() - t0
    per_ss = ncs.per_sweep_vectors(tg.per_window_ss, lay.sweep_number)
    q_ss = np.full(nf.N_STATS, np.nan)
    q_ss[:nf.N_SS] = per_ss[s_ss]
    q_ls = np.full(nf.N_STATS, np.nan)
    q_ls[nf.N_SS:] = tg.per_window_ls[0]
    mask_ss = np.zeros(nf.N_STATS, dtype=bool)
    mask_ss[:nf.N_SS] = True
    mask_ls = ~mask_ss
    out = []
    for name, q_s, mask, sub, s in (("SS", q_ss, mask_ss, sub_ss, s_ss), ("LS", q_ls, mask_ls, sub_ls, s_ls)):
        Sigma_s = np.cov(sig.per_sweep[int(s)], rowvar=False, ddof=1)
        u, status, nfev, njev, calls, secs = solve_sweep(q_s, Sigma_s, mask, sub, cfg300, seed, law_cell,
                                                         law_cell.tau_s)
        out.append((name, calls))
        print("S  %s sweep %d at M = 300: status %d, trf nfev %d, njev %d, residual calls %d, %.1f s; "
              "u = (%.4f, %.4f, %.4f); truth (%.4f, %.4f, %.4f); start (%.4f, %.4f, %.4f)"
              % (name, s, status, nfev, njev, calls, secs, u[0], u[1], u[2],
                 math.log(law.sigma_f), law.rho_f, math.log(law.sigma_s),
                 math.log(law_cell.sigma_f), law_cell.rho_f, math.log(law_cell.sigma_s)))
    print("   (Sigma_hat_s from 200 replicate cells: %.1f s)" % t_sig)

    # --- the cost of one solve at M = 3000 and of the bootstrap, per sweep
    calls_ss, calls_ls = out[0][1], out[1][1]
    solve_ss = calls_ss * t_ss
    solve_ls = calls_ls * t_ls
    print("C  one solve at M = 3000 (the call count of S x the evaluation of E): SS %.1f min, LS %.1f min"
          % (solve_ss / 60, solve_ls / 60))
    for B in (50, 100, 200):
        print("C  bootstrap of one sweep at B = %d, full M, refits like the main solve: SS %.1f h, LS %.1f h"
              " (one core); SD(v)/v about %.2f for a normal log sigma_s_hat" % (
                  B, B * solve_ss / 3600, B * solve_ls / 3600, math.sqrt(2.0 / (B - 1))))


if __name__ == "__main__":
    main()
