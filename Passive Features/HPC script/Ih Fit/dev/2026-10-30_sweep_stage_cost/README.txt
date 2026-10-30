dev/2026-10-30_sweep_stage_cost -- the cost and the precision of the per-sweep stage (Block 9, "Block 4b")
=========================================================================================================

Standalone checks (not pipeline code), written 2026-10-30 by the noise-fit chat for the
SPEC entry of the per-sweep stage (SPEC section 2.5 (h), Block 9; decision D-116) and the
questions put to the user on its bootstrap. They need the Ih Fit folder two levels up (or
TEEG_IH_FIT) and run Block 4's own functions on Block 4's smoke fixture
(smoke_noise_fit.build_cell: 2 SS sweeps of 37 pulses with 11 planted drops, 30 and 33
kept windows; 5 LS pre-windows of 13250 samples at 50 kHz; theta_cf noise; per-sweep
tilts at sigma_b = 0.1 mV/s). The restriction of a layout to one sweep and the 3-coordinate
solve are written in the scripts for the measurement only; Block 9's code is not written.

Files
-----
check_sweep_stage_cost.py         (seed 20261053; about 3 min) parts E, R, S, C below
sweep_stage_cost_2026-10-30.txt   its output
check_refit_binding_size.py       (seed 20261054; about 6 min) part V below
refit_binding_size_2026-10-30.txt its output

What came out (run here, 2026-10-30, one core)
----------------------------------------------
E  one evaluation of the per-sweep binding function at M = 3000: SS sweep 92 alone
   3.54 s (100 replicate sweeps); LS sweep 54 alone 8.55 s (3000 windows); the cell's
   own SS part 3.78 s (48 replicate cells x 2 sweeps) and LS part 10.85 s.
R  one replicate sweep (the bootstrap's draw): SS 0.042 s, LS 0.005 s.
S  one solve of D-116.1 (u_s = (log sigma_f, rho_f, log sigma_s), tau_s held, W_s from
   200 replicate cells, Block 4's trf settings) at M = 300 from a start set apart from the
   truth (sigma_f x 1.05, rho_f + 0.02, sigma_s x 0.8): SS 16 residual calls (4 trf
   iterations, status 3), LS 28 calls (7 iterations, status 2).
C  hence one solve at M = 3000: SS 0.9 min, LS 4.0 min; the bootstrap of one sweep with
   refits like the main solve: B = 50 -> 0.8 h (SS), 3.3 h (LS); B = 100 -> 1.6 h, 6.6 h;
   B = 200 -> 3.1 h, 13.3 h. SD(v)/v = sqrt(2 / (B - 1)) for a normal log sigma_s_hat:
   0.20, 0.14, 0.10. The fixture cell (2 SS + 5 LS sweeps) at B = 100: about 36 h on one
   core with M = 3000 refits, about 13 h with M = 1000 (an evaluation at M = 1000 took
   about 1.25 s for the SS sweep and 3.1 s for the LS sweep in part V).
V  the linearised (sandwich) variance V = A Sigma* A^T, A = (G^T W G)^-1 G^T W, G the
   Jacobian of the per-sweep binding function at theta_cf (central differences, step
   0.02 in u), W from the sweep's covariance at the start law, Sigma* at theta_cf
   (R = 300 each), three binding seeds at M = 1000 and M = 3000:
     SS sweep 92: sqrt(v) = 0.067-0.069; v range over seeds 7.4 % (M = 1000) and 2.1 %
                  (M = 3000) of the mean; mean v at M = 1000 / at M = 3000 = 1.023
     LS sweep 54: sqrt(v) = 0.150-0.171; v range over seeds 14.1 % and 25.0 %; ratio 0.995
   So the refits' binding size barely moves v (-0.5 % to +2.3 %), while the binding
   function's Monte Carlo error already leaves several to 25 % on it: a bootstrap much
   more precise than B = 100 (SD(v)/v = 0.14) buys little. Per-sweep SD of
   log sigma_s_hat: about 0.07 for an SS sweep of 30 windows, about 0.16 for one LS
   pre-window (tau_s held).

Caveats: one solve per protocol was timed (the iteration count of a real sweep from the
cell's estimate may differ); the timings are this sandbox's (the cluster's cores differ);
a real cell's cost scales with its number of LS sweeps, which the fixture fixes at 5.
