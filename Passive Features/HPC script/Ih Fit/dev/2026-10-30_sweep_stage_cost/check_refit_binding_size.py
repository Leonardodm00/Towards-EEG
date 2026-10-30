"""check_refit_binding_size.py -- does the bootstrap's v(c, s) depend on the refits' binding size M?

Standalone dev check (not pipeline code), written 2026-10-30 by the noise-fit chat
for the questions put to the user on the per-sweep stage's bootstrap (D-116.2).

To first order the bootstrap's refits spread as the linearised (sandwich) variance

    V(u_hat_s) = A Sigma*_s A^T,   A = (G^T W G)^{-1} G^T W,   W = Sigma_s^{-1},

with G (d_s x 3) the Jacobian of the per-sweep binding function q_sim(theta | s) in
u_s = (log sigma_f, rho_f, log sigma_s) (tau_s held), Sigma_s the sweep's simulated
covariance at the start law (Block 4's weight, D-116.1) and Sigma*_s the covariance
of a replicate sweep's statistic vector at the law the replicates are drawn from
(theta_hat(c, s) in the stage; theta_cf here). The refits' M enters only through G
(the common random numbers freeze the binding function's Monte Carlo error, which
then shifts every refit alike). This script computes G by central differences
(step 0.02 in each coordinate of u) at theta_cf for M = 1000 and M = 3000 on three
binding seeds each, and prints V's diagonal: the spread over seeds at one M is the
Monte Carlo error of v from the binding function; the difference between the M's is
what choosing M = 1000 for the refits would change.

Fixture: Block 3's writer-like cell (smoke_noise_fit.build_cell): SS sweep 92
(K_s = 30 windows), LS sweep 54 (one pre-window of 13250 samples); Sigma_s and
Sigma*_s from R = 300 replicate cells each (simulated_covariance, per_sweep).

    python check_refit_binding_size.py      # about 6 min on one core

Needs the Ih Fit folder two levels up (or TEEG_IH_FIT). Pure ASCII, LF.
"""
from __future__ import annotations

import math
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

IH_FIT = Path(os.environ.get("TEEG_IH_FIT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(IH_FIT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import noise_fit as nf              # noqa: E402
import noise_law as nl              # noqa: E402
import smoke_noise_fit as sf        # noqa: E402
from check_sweep_stage_cost import sweep_layout   # noqa: E402

SEED = 20261054
STEP = 0.02
R_COV = 300


def law_of(u, tau_s):
    return nl.NoiseLaw(math.exp(u[0]), float(u[1]), math.exp(u[2]), tau_s)


def jacobian(u0, tau_s, sub, cfg, seed, which, mask):
    G = np.empty((int(mask.sum()), 3))
    for j in range(3):
        up, dn = u0.copy(), u0.copy()
        up[j] += STEP
        dn[j] -= STEP
        qp = nf.binding_function(law_of(up, tau_s), sub, cfg, seed, which=which)[mask]
        qm = nf.binding_function(law_of(dn, tau_s), sub, cfg, seed, which=which)[mask]
        G[:, j] = (qp - qm) / (2.0 * STEP)
    return G


def sandwich(G, Sigma, Sigma_star):
    W = np.linalg.inv(Sigma)
    A = np.linalg.solve(G.T @ W @ G, G.T @ W)
    return A @ Sigma_star @ A.T


def main():
    rng = np.random.default_rng(SEED)
    with tempfile.TemporaryDirectory() as d:
        lay, tg, nwin, ls, truth = sf.build_cell(Path(d), rng)
    law = sf.THETA_CF
    u0 = nf.law_to_u(law)[:3]
    law_cell = nl.NoiseLaw(law.sigma_f * 1.05, law.rho_f + 0.02, law.sigma_s * 0.8, law.tau_s)
    t0 = time.time()
    sig_w = nf.simulated_covariance(law_cell, lay, R_COV, np.random.default_rng(SEED + 1), per_sweep=True)
    sig_s = nf.simulated_covariance(law, lay, R_COV, np.random.default_rng(SEED + 2), per_sweep=True)
    print("Sigma_s (start law) and Sigma*_s (theta_cf), R = %d each: %.0f s" % (R_COV, time.time() - t0))
    s_ss, s_ls = lay.sweep_numbers[0], int(lay.ls_sweep_number[0])
    mask_ss = np.zeros(nf.N_STATS, dtype=bool)
    mask_ss[:nf.N_SS] = True
    cases = (("SS", s_ss, sweep_layout(lay, "ss", s_ss), "ss", mask_ss),
             ("LS", s_ls, sweep_layout(lay, "ls", s_ls), "ls", ~mask_ss))
    print("V = A Sigma* A^T at theta_cf; columns: SD of log sigma_f_hat, rho_f_hat, log sigma_s_hat (the last is"
          " sqrt(v)); one row per (M, binding seed)")
    for name, s, sub, which, mask in cases:
        Sigma = np.cov(sig_w.per_sweep[int(s)], rowvar=False, ddof=1)
        Sigma_star = np.cov(sig_s.per_sweep[int(s)], rowvar=False, ddof=1)
        rows = {}
        for M in (1000, 3000):
            cfg = nf.NoiseFitConfig(M_binding=M)
            for k in range(3):
                t0 = time.time()
                G = jacobian(u0, law.tau_s, sub, cfg, 1000 + k, which, mask)
                V = sandwich(G, Sigma, Sigma_star)
                sd = np.sqrt(np.diag(V))
                rows.setdefault(M, []).append(V[2, 2])
                print("  %s sweep %d  M = %4d  seed %d:  SD = (%.4f, %.4f, %.4f)   v = %.5f   (%.0f s)"
                      % (name, s, M, 1000 + k, sd[0], sd[1], sd[2], V[2, 2], time.time() - t0))
        v1, v3 = np.array(rows[1000]), np.array(rows[3000])
        print("  %s: v over seeds -- M = 1000 mean %.5f (spread %.1f %% of the mean, max - min); M = 3000 mean %.5f"
              " (spread %.1f %%); ratio of the means M1000 / M3000 = %.3f"
              % (name, v1.mean(), 100 * np.ptp(v1) / v1.mean(), v3.mean(), 100 * np.ptp(v3) / v3.mean(),
                 v1.mean() / v3.mean()))


if __name__ == "__main__":
    main()
