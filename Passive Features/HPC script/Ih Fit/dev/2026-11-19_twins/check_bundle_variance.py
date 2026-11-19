"""Standalone check (not pipeline code), 2026-11-19, the twins chat.

The variance of the average of K equally correlated window means:
    Var[mean_K] = (v / K) * (1 + (K - 1) * rho)
(v the variance of one window mean, rho the pairwise correlation between window means), against a Monte Carlo
estimate, at the K and rho quoted in the chat: K = 19 windows (one polarity of a 37-pulse SS train, dep), rho = 0
(independent twin windows, D-180.2) and rho = 0.965 (the slow part of 571430815 between pulses 200.5 ms apart
under its nf2 law, decision log D-180, "Why").

    python3 check_bundle_variance.py
"""
import numpy as np

SEED = 20261119
K = 19
V = 1.0            # variance of one window mean (arbitrary unit)
N_REP = 400000     # Monte Carlo repetitions
TOL_REL = 0.01     # fixed before running: 400000 repetitions give a relative SE of sqrt(2/N) = 0.0022

rng = np.random.default_rng(SEED)
ok = True
for rho in (0.0, 0.965):
    common = rng.standard_normal((N_REP, 1))
    own = rng.standard_normal((N_REP, K))
    m = np.sqrt(V) * (np.sqrt(rho) * common + np.sqrt(1.0 - rho) * own)   # Var = V, pairwise corr = rho
    mc = m.mean(axis=1).var(ddof=1)
    pred = V / K * (1.0 + (K - 1) * rho)
    good = abs(mc / pred - 1.0) < TOL_REL
    ok &= good
    print("rho = %.3f  K = %d : Monte Carlo %.5f  predicted %.5f  ratio to independent (V/K) %.2f  %s"
          % (rho, K, mc, pred, pred / (V / K), "PASS" if good else "FAIL"))
print("ALL PASS" if ok else "FAIL")
