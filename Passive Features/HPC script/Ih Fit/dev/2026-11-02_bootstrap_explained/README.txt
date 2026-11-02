dev/2026-11-02_bootstrap_explained -- the check behind the explanation of the per-sweep bootstrap
=================================================================================================

Standalone check (not pipeline code), written 2026-11-02 by the noise-fit chat when the
user asked for the per-sweep stage's bootstrap (SPEC section 2.5 (d), Block 9 = "Block
4b"; D-116.2) to be put in context so that they could answer its four open points (SPEC
section 8). It models only the pooling (SPEC (P3)-(P4), (D-076.1)): the per-sweep
estimates are normal around their true values with the variance v measured on Block 4's
fixture (dev/2026-10-30_sweep_stage_cost: SS sweep of 30 windows 0.0046, LS pre-window
0.026), and the bootstrap's estimate of v is v chi^2_{B-1} / (B - 1). No Block-9 code
exists.

Files
-----
check_bootstrap_B.py           (seed 20261102; about 2 s)
bootstrap_B_2026-11-02.txt     its output

What came out (run here, 2026-11-02)
------------------------------------
A  SD(v_hat)/v = sqrt(2 / (B - 1)) confirmed by Monte Carlo: 0.325 / 0.202 / 0.142 / 0.100
   at B = 20 / 50 / 100 / 200.
B  RMS error of the shrunk log sigma_s against the truth (x 100; 40000 cells per row):
   B hardly matters. Against v known: SS with 2 sweeps within 0.05 at every B from 20 on
   (5.50-6.92); SS with 10 sweeps +0.10 at B = 50, +0.04 at B = 100 (2.75-6.56); LS with
   5 sweeps +0.16 at B = 50, +0.08 at B = 100 (9.01-14.36). The raw per-sweep error is
   6.7-6.8 (SS) and 16.1-16.2 (LS). Side result: with 2 SS sweeps the pooling helps when
   the true between-sweep SD tau is small (5.50 against 6.73 at tau = 0) and costs a
   little at tau = 0.20 (6.92 against 6.78) -- a property of pooling two sweeps, not of B.
C  one refit stuck at the lower bound of sigma_s (1e-4 mV) kept in v multiplies it by 170
   / 85 (SS, B = 50 / 100) and 31 / 16 (LS): the sweep would be pooled to its protocol's
   mean almost entirely.

Hypotheses: normal sampling errors; v equal across a protocol's sweeps and as measured at
theta_cf on the fixture's layout; the bootstrap's own errors other than its finite B (the
binding seed, v evaluated at the estimate) left out.
