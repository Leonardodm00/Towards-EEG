dev/2026-11-17_three_component_law -- does a third ("middle") noise component remove the misfit of the nf1 run?
===============================================================================================================

Standalone checks (not pipeline code), written 2026-11-17 by the noise-fit chat at the user's request ("Go on with
the test", 12:30), after the nf1 run's read-back (the Block 6 handoff, section 4): the two-component law of SPEC 2.1
is rejected in all 18 L3_exc cells, its SS autocorrelation at 1 ms above the data's in 16. The user's proposal: three
components -- the fast part as it is, a middle part with a time scale of at most 1 ms, the slow part above 1 ms.
Nothing in the pipeline is changed; no decision is taken here.

Input: the 18 cell JSONs of the run (`$OUT/cells/specimen_<id>.json`; not in the repository). The scripts need the
Ih Fit folder two levels up (or TEEG_IH_FIT).

Laws
----
M2   fast AR(1) (sigma_f, rho_f) + slow OU (sigma_s, tau_s)                                    4 parameters (SPEC 2.1)
M3   M2 + middle OU (sigma_m, tau_m), tau_m in [0.04 ms, 1 ms], tau_s in [1 ms, 10 s]          6 parameters
M3w  M3 with the fast part white (rho_f fixed at 0.001)                                        5 parameters
Independent stationary Gaussian components: gamma(k) = sigma_f^2 rho_f^k + sigma_m^2 exp(-k dt / tau_m)
+ sigma_s^2 exp(-k dt / tau_s) (mV^2). The 0.04 ms lower bound of tau_m (2 samples) is the assistant's choice.
J = r^T Sigma^-1 r, r = q(theta) - q_hat(c) (15 statistics), Sigma the run's stored covariance -- simulated at the
run's theta_hat(c) under M2, an approximation for M3.

Files
-----
check_three_component_cf.py                    stage 1: closed form, no simulation (2.4 min, 2 cores)
three_component_cf_2026-11-17.txt / .csv       its output
check_three_component_sim.py                   stage 2: the pipeline's binding function with the middle component
                                               added by wrapping noise_law.simulate_traces; --refine fits M3 by simulation
three_component_sim_2026-11-17.txt             (a), (b) on the 18 cells, (b) at stage 1's law as it stands
three_component_sim_refined_2026-11-17.txt / .csv   (a), (b), (c) on the 18 cells (42 min of CPU, 2 cores)

Run:
    python3 check_three_component_cf.py  <cells dir> cf.csv
    python3 check_three_component_sim.py <cells dir> cf.csv sim.csv --refine

What came out (run here, 2026-11-17; numpy 2.5.3, scipy 1.18.1)
----------------------------------------------------------------
Stage 1 (closed-form statistics, corrected by each cell's stored simulation so that they equal q_sim at theta_hat(c)):
  median J: M2 refitted 116.9 (the run's 119.2: the approximation moves M2's optimum little), M3 36.7, M3w 36.7.
  J(M2) - J(M3) exceeds chi2_2's 99 % point (9.21) in 18 of 18 cells; J(M3) / J(M2) median 0.44 (0.10-0.91).
  J(M3) below chi2_9's 99 % point (21.7) in 4 cells. M3w within 1 of M3 except 529878215 (+10) and 569844159 (+41).

Stage 2 (simulation, M = 3000, each cell's binding seed, the layout rebuilt from the JSON -- group sizes, t_SS and
tilts the cell's, the kept windows' positions spread evenly because the JSON does not store them):
  (a) M2 at the run's theta_hat(c): J(a) / J(run) median 1.01, range 0.97-1.07 -> the rebuilt layout is adequate.
  (b) M3 at stage 1's law, not refined: median J 46.1, below (a) in 16 of 18 (above in 539653924 and 571430815,
      where stage 1 ran tau_s to its 10 s bound).
  (c) M3 refined by simulation (least_squares trf, at most 6 iterations at M = 1000, started at stage 1's law, or
      at the run's slow part where stage 1 hit the tau_s bound; J re-evaluated at M = 3000):
      median J 37.9 against 120.7 for (a); (c) / (a) median 0.36, range 0.08-0.87; below (a) in 18 of 18;
      below chi2_9's 99 % point in 5 of 18 (537313600, 548408551, 548473051, 571430815, 571691416).
      Still far: 569844159 398, 529878215 220, 571483807 159, 541545716 144, 526785799 103.
      SS rho(1 ms): standardised residual median +2.27 -> -0.50; the law above the data in 16 -> 6 cells.
      The 15 statistics' relative error, median per cell: 1.2-16 %, median 5.8 %.
      Fitted: sigma_m 0.012-0.024 mV (median 0.017); tau_m 0.07-0.11 ms in 8 cells, 0.15-0.39 ms in 4, 0.77-1.0 ms
      in 6 (3 at the 1 ms bound: 529878215, 548408551, 569844159); sigma_f 0.033-0.039 mV with rho_f 0.001-0.23
      (below 0.02 in 4 cells); tau_s 10 ms-650 ms in 17 cells (median 118 ms), at the 10 s bound in 569844159
      (571624532: 189 ms, against 1863 ms under M2).

Reading
-------
A middle component is supported in every cell and removes the systematic excess of the 1 ms autocorrelation; it cuts
J to about a third. It does not make the law pass: 13 of 18 cells stay rejected at 1 %, the five largest being cells
whose SS and LS statistics disagree under any one law per cell (the handoff's section 4). The middle time scale is
not one number across cells (0.07 to the 1 ms cap).

Limits: Sigma is M2's (not re-simulated under M3); (c) is a short refinement, so its J is an upper bound of M3's
minimum; the layout is rebuilt; the statistic set (lags 0.02, 0.1, 1, 10 ms) was chosen for four parameters; the
per-sweep stage (Block 9) is not tested (the JSONs do not hold the per-sweep targets).
