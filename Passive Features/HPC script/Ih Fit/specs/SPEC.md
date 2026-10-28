# SPEC -- Ih Fit / Stage-7 per-cell noise-law fit (D-017, D-034, D-048 to D-056)

Single source of truth for what the noise-fit code of `Passive Features/HPC
script/Ih Fit/` must do. The code follows this file; when the two disagree,
fix whichever is wrong and record why in the project's decision log
(`TEEG_decisions_and_ideas_log.md`, project root; entries cited as D-nnn).
The integration tester reads only the repository, so every constraint the
code must satisfy is stated here, including those that come from decisions.

Notation rules: every symbol is defined at first use; a conditional quantity
keeps its conditioning every time it appears (write q_m(theta | n_m), never
q_m); quantifiers ("for each fixed theta", "for all m") are stated explicitly.
Plain ASCII throughout (cluster rule, hpc-python-compat).

Blocks are appended as they are written (section 5). Status `drafted` means
the entry was written before the code; `smoke-tested <date>` means the block's
smoke suite passed in the sandbox on that date.

Last reviewed at commit: (branch `sci/noise-fit`, baseline `9ed4da4` = the
content of `IhFit_main_2026-10-16.zip`, a git archive of `main`; Block 1 reached
`main` as `e8ed84e`, and Block 2 and the tools below were verified with `git am`
on `main` at `e8ed84e` on 2026-10-20 and reached `main` as `36ba658` and
`ca028ea`)

## 0. Coverage

Paths this spec covers (relative to `Passive Features/HPC script/Ih Fit/`):

- `noise_stats.py` (Block 1)
- `smoke_noise_stats.py` (Block 1)
- the two lag-1 estimator lines of `passive_fitting_hpc_fixed.py`
  (`_estimate_noise`, `_estimate_residual_noise_at_mle`) and the helpers
  `_estimator_band`, `_acf_gap_band` of `smoke_ih_recovery.py` (Block 1)
- `noise_calibration.lag_autocorr` (Block 1); the rest of
  `noise_calibration.py` is NOT yet covered
- `noise_law.py`, `smoke_noise_law.py` (Block 2)
- `noise_fit_gates.py`, `smoke_noise_fit_gates.py`, `cluster_gates_noise_fit.sh`
  (tools: the cluster-side checks and the archive inspection, section 5)
- `noise_windows.py`, `noise_cell_stats.py`, `smoke_noise_windows.py` (Block 3)
- `noise_fit.py`, `smoke_noise_fit.py` (Block 4: the per-cell fit)
- (to be added block by block) `noise_sweep_fit.py` (Block 4b), `run_noise_fit.py`, the `sens_*.py`
  scripts, the OU term of `synthetic_ground_truth.add_recording_noise`

Everything else in `Ih Fit/` (the I_h fitter, the loader, the recovery gate,
the earlier smoke suites) is specified by its own docstrings and
`CHANGELOG_ih_fit.md` and is out of scope for the tester. `Biological Fit/`
is never changed (D-029 (i)).

## 1. Scientific goal

Stage 7 of the I_h-aware passive fit generates synthetic twins of the real
Allen L3_exc cells and asks whether the six-parameter fit recovers a known
truth (D-011). Each twin must carry the recording noise of its own real cell
(D-017). This code (a) computes the noise statistics of a real cell with the
pipeline's own estimators, (b) fits, per cell, the four-parameter noise law
(D-017.1) to those statistics by indirect inference with the simulated
inverse-covariance weighting (D-033, D-048 to D-051, D-056), (c) calibrates
the per-sweep tilt (D-034), (d) writes the law into the Stage-7 manifest so
the generator injects it, and (e) measures in silico how a noise component
the law lacks moves the fit (D-055). Consumers: `gen_from_manifest.py` /
`synthetic_ground_truth.py` (the twins), `run_ih_recovery.py` (the gate),
the noise table and manifest CSVs, the S0-S3 reports.

[2026-10-21, D-073 (user): each twin sweep is the twin of one real sweep and
carries noise calibrated on that sweep, not a per-cell average. Which
parameters are estimated per sweep and which per cell is open; the
assistant's proposal (decision log, D-073 comments) keeps (b) as the pooled
per-cell stage and adds a per-sweep stage for sigma_f, rho_f and sigma_s with
tau_s and sigma_b per cell. Until the user settles the form, Blocks 3-4 keep
every statistic per sweep as well as per cell, so either form can be built on
them. D-072 (user): no shape test anywhere in the noise fit.]

## 2. Mathematical formulation

### 2.1 The law (D-017.1, D-034)

For one sweep sampled every Delta_t (s) at t_j = j Delta_t, j = 0..n-1, with
mid-time t_bar, the injected noise (mV) is

    e_j = e_fast_j + e_slow_j + a + b (t_j - t_bar),                    (L1)

    e_fast_j = rho_f e_fast_{j-1} + eta_j,   eta_j  ~ N(0, sigma_f^2 (1 - rho_f^2)),
    e_slow_j = rho_s e_slow_{j-1} + eta'_j,  eta'_j ~ N(0, sigma_s^2 (1 - rho_s^2)),
    rho_s = exp(-Delta_t / tau_s),                                     (L2)

both recursions started in their stationary laws (e_fast_0 ~ N(0, sigma_f^2),
e_slow_0 ~ N(0, sigma_s^2)), driven by independent iid Gaussian innovations;
a ~ N(0, sigma_a^2) (mV) and b ~ N(0, sigma_b(c)^2) (mV/s) drawn once per
sweep, independent across sweeps. theta = (sigma_f, rho_f, sigma_s, tau_s) in
Theta = R_{>0} x (0, 1) x R_{>0} x R_{>0} is the per-cell parameter vector;
sigma_b(c) is calibrated separately (D-034.1, section 2.4), sigma_a is never
needed (every statistic demeans). For each fixed theta and every lag k >= 0
(samples) the autocovariance of the stationary part is

    gamma(k | theta) = sigma_f^2 rho_f^k + sigma_s^2 exp(-k Delta_t / tau_s)  (mV^2).  (L3)

### 2.2 The window statistics (D-028, D-029, D-051)

[2026-10-28: the window start and the subtraction of this section are replaced by D-095 with D-103 -- the per-sample median window per SS sweep and polarity over [-10, 200) ms; per group a single exponential with a free baseline fitted on [5, 200) ms and its point value t_sp = tau_hat ln(|A_hat| / (kappa_tr sigma_SS_hat(c))); t_SS(c) = max(80 ms, max t_sp); 80 ms also the fallback, flagged; that median window subtracted on [t_SS(c), 200). To be rewritten here before the code; open: a failed group and the fallback's trigger.] [2026-10-28 (later): D-104 -- the fallback applies when the window [t_SS(c), 200) ms would be shorter than 50 ms (min_window_ms = 50), and with no valid group; a group whose fit fails or that has fewer than 5 windows (min_windows_per_group, PROVISIONAL) is flagged and left out of the maximum.]

For a window x_1..x_n (mV) with mean x_bar and u_j = x_j - x_bar:

    sigma_hat = sqrt( S_0 / (n - 1) ),           S_0 = sum_{j=1}^{n} u_j^2,          (S1)
    rho_hat(L) = S_L / S_0,                      S_L = sum_{j=1}^{n-L} u_j u_{j+L},  (S2)

(S2) is the textbook estimator, computed by
`statsmodels.tsa.stattools.acf(x, adjusted=False, nlags=L)[L]` through the
single wrapper `noise_stats.sample_acf` (D-028, D-029 (iv)); it is NaN when
L < 1 or n - L < 10, and at every lag when the window has a non-finite sample
or S_0 = 0 (D-029 (iii)). The estimator it replaces,
rho_adj(L) = (S_L/(n-L)) / (S_0/n), relates to it by rho_hat(L) = (n-L)/n *
rho_adj(L) exactly, window by window (D-028.2). No opt-in for rho_adj exists
anywhere in `Ih Fit/` (D-029 (ii)).

The per-cell statistic vector q_hat(c) in R^d, d = 15 (D-051 (i)), each entry
the median over the cell's K_m windows of statistic m: SS enlarged window
[t_SS(c), 200) ms after each pulse, after the per-polarity mean-response
subtraction (D-051.1): sigma_hat, rho_hat at 0.02, 0.1, 1, 10 ms; LS 265 ms
pre-window: sigma_hat, rho_hat at 0.02, 0.1, 1 ms; LS first 50 ms: sigma_hat;
LS last 132.5 ms: sigma_hat, rho_hat at 0.02, 0.1, 1, 10 ms. Lags in samples
are round(lag_ms * 1e-3 * f_s). (Blocks 3 onward; stated here so Block 1's
estimator is seen in its use.)

Window geometry (Block 3; D-051, D-072, D-076). A stored SS window holds
n_w = pre_n + post_n samples, t = 0 at the pulse onset, pre_n = round(10 ms
f_s), post_n = round(200 ms f_s). The enlarged window of a pulse is its
samples with t >= t_SS(c): m = n_w - pre_n - round(t_SS f_s / 1e3) samples,
ending 0.5 ms before the next pulse's onset (onset period P = 10025 samples
at 50 kHz, read from the data). t_SS(c) = max(t_floor, t_tr(c)), t_floor =
80 ms (D-076 (ii)); t_tr(c) is the first 5 ms bin boundary after which
|mean of r_odd over the bin| < kappa_tr sigma_SS(c) in every later bin (the bin
MEAN is Block 3's reading of D-051's "|r_odd| in 5 ms bins"), with r_odd(t) =
(r_dep(t) - r_hyp(t)) / 2 the odd part of the polarity means of the windows
(each minus its own pre-pulse mean), sigma_SS(c) the median pre-pulse SD,
kappa_tr = 0.1 PROVISIONAL (D-051 (iv)). Found by the smoke (W2, fixture of 63
windows at the cohort law): r_odd's own noise floor is 0.17 sigma_SS, above
kappa_tr sigma_SS, so the test as written fails on noise alone [corrected
2026-10-23: 0.17 sigma_SS is the option's floor estimate below (the per-sample
SD of r_even over t >= 0) on one fixture realisation, not the SD of the test's
statistic; see the next sentence]. Simulation 2026-10-23 (standalone
`check_findings_2026-10-23.py`, not in the repository; theta_cf, 200 cells
per layout, the archive's layout): under noise alone the SD of a 5 ms bin mean
of r_odd in [100, 200) ms is about sigma_s sqrt(2 / K) (the slow component,
5 ms << tau_s; x sqrt(2) from the pre-pulse baseline), simulated 0.30, 0.24,
0.15 and 0.11 sigma_SS at K = 40, 63, 199 and 348 -- at or above kappa_tr
sigma_SS at every L3_exc window count (passing on noise needs K >> 2
(sigma_s / (kappa_tr sigma_SS))^2, about 320) -- and the test as written puts
t_SS >= 190 ms (`ss_enlarged_none`) in 51-90 % of noise-only cells; even
without noise the fixture's RC response (113 MOhm, 28.1 ms) falls below 0.1
sigma_SS only at 130 ms. The option
`WindowConfig.transient_floor_z` raises the threshold to z x the per-sample
SD of the even part r_even = (r_dep + r_hyp) / 2 over t >= 0 (same noise, no
response); default None (D-051 as written), the user's call. [corrected
2026-10-23: r_even does not carry r_odd's noise near the window end -- a
window's last ~40 ms correlate, through the slow component and the 9.5 ms
overlap, with the next (opposite-polarity) window's pre-pulse baseline, which
inflates r_odd's noise there and deflates r_even's (per-bin SD at 195 ms
0.28 against 0.08 sigma_SS at K = 63, sigma_b = 0), and r_even carries the
tilt's ramp; with z = 3 the option still puts t_SS >= 190 ms in 4-13 % of
noise-only cells. A per-bin threshold max(kappa_tr sigma_SS, z x the SD of
r_odd's bin mean simulated at the cell's layout and law) gives at z = 3.5 at
most 0.3 % (K = 63 and 348, the true law), t_SS 80 ms (K = 63) and about
100 ms (K = 348) on the fixture's response. Open (section 8).] The subtracted quantity (D-051.1)
is the per-polarity MEAN WINDOW -- for each sample j, the mean over that
polarity's K_p windows -- a vector over time, computed over the whole cell
and subtracted from every window of that polarity (D-076 (ii), confirmed by
the user); each window's own scalar mean is removed by (S1)-(S2) themselves.
If t_SS(c) >= 190 ms the cell is flagged `ss_enlarged_none` (fallback of
D-051 (i), not built in Block 3: the flag is raised). Per sweep s the same
statistics are kept as medians over that sweep's windows (SS: 5 per sweep;
LS: the 10 values of that one window), the inputs of the per-sweep stage of
D-076 (i). The LS pre-window of EVERY Long Square sweep enters, hyperpolarising
and depolarising (D-073 needs a twin of each; the assistant's choice, D-076
open points), one window each: [t_0, t_on - 5 ms) with t_on the first sample
where |i| exceeds half its maximum (the fitter's rule, `_estimate_noise`).

### 2.3 The fit (D-033, D-048 to D-050, D-056)

Binding function: q_m(theta | n_m) = med(T_m | theta, n_m), the single-window
median of statistic m under the law (analytic level), estimated by the
Harrell-Davis median over M = 3000 windows simulated with common random
numbers (D-056 (iii)). Cost, for each symmetric positive-definite W:

    J_W(theta | c) = (q_sim(theta) - q_hat(c))^T W (q_sim(theta) - q_hat(c)),   (F1)

iteration (D-048, D-049, D-050): W_0 = diag(1/SE_hat_m^2) from a window-level
bootstrap (B = 2000) of the cell's own per-window values; for k >= 0,
Sigma_hat_k = sample covariance (ddof 1) of the statistic vectors of R = 600
cells simulated at theta_hat_k with c's own n_m, K_m and sweep structure
(D-056 (i)), W_{k+1} = Sigma_hat_k^{-1}, theta_hat_{k+1} = argmin J_{W_{k+1}};
stop at the first k >= 1 with max_i |u_{k,i} - u_{k-1,i}| <= 0.01 in the fit
coordinates u = (log sigma_f, rho_f, log sigma_s, log tau_s), or k = k_max = 6
(flag `no_plateau`); final step at R = 3000 whose theta_hat, J and
kappa(C_hat) (condition number of the correlation matrix) are reported. Inner
problem: `scipy.optimize.least_squares(method='trf')` on the whitened residual
C^{-1}(q_sim - q_hat), Sigma_hat = C C^T (Cholesky), default finite-difference
Jacobian, three starts (closed form; tau_s/2; 2 tau_s), lowest J kept.

Block 4's realisation of (F1) and the iteration (2026-10-23; D-048 to D-050,
D-056 as above, the rest the assistant's choices, flagged). (a) Binding
function: the SS part is computed on M_cells = ceil(M / K) replicate cells
simulated at theta with the cell's own layout (n_sweeps continuous sweeps of
(train_length - 1) P + n_w samples each, D-056 (i); the enlarged windows cut
at the stored positions; the per-polarity mean window of the replicate's own
windows subtracted, D-051 (iv)) and pooled to M_cells K windows; the LS part
on M independent windows of n_LS samples, one tilt draw each (an LS
pre-window is its own sweep); each part's Harrell-Davis median per statistic
(`scipy.stats.mstats.hdquantiles`, prob 0.5) is q_sim(theta). Common random
numbers by re-seeding: every evaluation rebuilds `default_rng(seed_binding)`
and draws in the fixed order of `noise_law.draw_units`, so q_sim(theta) is a
deterministic function of theta for a given cell and seed (nothing is stored:
3000 windows of 13250 samples would hold 1.3 GB of unit draws). (b) Sigma_hat:
R replicate cells at the real layout, each reduced to its statistic vector
by `noise_cell_stats.statistic_vector`; sample covariance (ddof 1); W = its
inverse through the Cholesky factor C (Sigma_hat = C C^T, residual whitened
by `solve_triangular`). (c) Statistics with no finite data value (K_m = 0 or
a non-finite SE) are dropped from q_hat, q_sim and Sigma_hat (flag
`stats_dropped` with their labels); d_eff <= 15. (d) Fit coordinates u =
(log sigma_f, rho_f, log sigma_s, log tau_s) with bounds log 1e-4 <= log
sigma <= log 10 (mV), 1e-3 <= rho_f <= 0.999, log 1e-4 <= log tau_s <= log
10 (s); `least_squares(method='trf', diff_step=(1e-3, 1e-2, 1e-3, 1e-3),
ftol=1e-4, xtol=1e-3, gtol=1e-8, max_nfev=15)` -- the finite-difference step
is set explicitly because q_sim is a Monte Carlo quantity (the default step,
sqrt(eps), would read roundoff), the tolerances match the stop rule (0.01 in
u) rather than the default 1e-8, and `max_nfev` counts trf iterations (scipy
does not count the Jacobian's evaluations), each costing 5 evaluations of
q_sim with the 2-point Jacobian: at most about 75 per start. (e) Starts: the closed-form start
is the argmin of (F1) with W_0 on the closed-form statistics q_cf (eq. 10;
instant), then that start with tau_s / 2 and 2 tau_s -- three starts for the
first solve (W_0); the later solves (W from Sigma_hat_k) start once, from
theta_hat_k (assistant's choice, `starts_after_first = 1`; W changes little
between iterations and a solve costs about an hour at M = 3000). (f) Tilt:
after every solve, sigma_b_hat^2 = 2 [V_hat_c(h*) - V(h* | theta_hat, 0)] /
h*^2 with h* = 4 P Delta_t, clipped at 0 (flag `tilt_clipped`; `tilt_no_pairs`
and sigma_b = 0 when V_hat has no pair); the next simulation uses it. (g)
Stop: the first Sigma_hat-based solve k >= 1 with max_i |u_{k+1,i} - u_{k,i}|
<= 0.01, or k_max = 6 solves (flag `no_plateau`); then the final step --
Sigma_hat at R = 3000 and theta_hat, one more solve, then sigma_b re-read at
the solution; the reported pair is (theta_hat, sigma_b_hat) and q_sim and J
are evaluated at that pair with the final Sigma_hat (the solve's own J, at
the previous sigma_b, stays in the history); kappa of the correlation matrix
of Sigma_hat (`numpy.linalg.cond`); and, as a goodness check only (no
decision uses it yet; the user's chi-square alarm is open), J ~ chi^2 with
d_eff - 4 degrees of freedom, its tail probability reported.
(h) Cost (sandbox timings 2026-10-23, one core, the fixture's 2-sweep layout,
K = 63, 5 LS windows): one evaluation of q_sim at M = 3000 is 16 s (SS 3.9 s
on 48 replicate cells, LS 12.3 s on 3000 windows of 13250 samples); one
replicate cell for Sigma_hat is 0.10 s (R = 600: 1 min; R = 3000: 5 min); a
10-sweep cell (K about 330) costs about 0.7 s per replicate (R = 600: 7 min;
R = 3000: 35 min) and 19 s per evaluation. A solve is at most about 75
evaluations per start (20 min), typically 30 (the smoke's solves take 6
trf iterations). A 2-sweep cell takes 1-3.5 h on one core, a 10-sweep cell
2-5.5 h; the campaign runs the 18 cells as independent jobs (Block 6).
Options not taken, for the user: a batched FFT autocorrelation in
`noise_stats` (most of the LS 12 s; it would be a second implementation of
(S2), D-029 (iv)); exact Markov jumps over the unused samples of a sweep
(halves the SS simulation; more bookkeeping). Finding (smoke F7): sigma_b_hat
of a 2-sweep cell rests on two tilt draws (14 of the 18 L3_exc cells), so
(T1) returns, for such a cell, essentially the mean square of two values of
b -- on the fixture 0.17 mV/s for a planted 0.1 -- while on 200 replicate
cells (F5) it is exact to 1 %; a per-cell sigma_b (D-034) is therefore a
2-degree-of-freedom estimate for most cells. Open for the user (section 8).

### 2.4 The tilt (D-034.1, D-051 (iii))

[2026-10-28: for the twins superseded by D-102 -- each twin sweep carries the realised tilt of its real sweep, b_hat(c, s); no sigma_b(c) is computed for them. (T1) stays in the fit's alternation until D-102's open point on the fit's simulator is answered.]

    sigma_b_hat(c)^2 = 2 [ V_hat_c(h*) - V(h* | theta_hat_c, 0) ] / h*^2,        (T1)

V_hat_c(h) = half the mean squared difference of two enlarged-window means h
apart in one sweep, same polarity (computed level); V(h | theta, 0) the same
quantity under the stationary part of (L1), computed EXACTLY for the sampled
law (`noise_law.semivariogram`: (1/2)[Var(x_bar_1) + Var(x_bar_2) - 2 Cov]
with (7a)-type sums over the sample pairs of the two windows; topic doc eq.
(25) is its continuous-time form, kept as `semivariogram_continuous` for the
tests, equal to it within O(Delta_t/tau_s) for the slow component -- Block 2
refinement, no decision changed); h* = 0.8 s PROVISIONAL (D-051). Alternated
with theta_hat at every iteration. [2026-10-20, data: the onset period is
200.5 ms, so the within-sweep lags are multiples of 0.2005 s and the
same-polarity ones even multiples; h* = 0.8 s is realised as 4 periods,
0.802 s, and V is evaluated at the realised lag]

### Notation

| Symbol | Meaning | Type / shape | Units |
|---|---|---|---|
| c | a real cell (Allen specimen) | id | -- |
| Delta_t, f_s | sampling interval, rate (1/Delta_t) | float | s, Hz |
| n, n_m | samples in a window; of the windows behind statistic m | int | samples |
| K_m | number of windows behind statistic m in cell c | int | -- |
| x, u | a window; the window minus its mean | float[n] | mV |
| S_0, S_L | sums of (S1), (S2) | float | mV^2 |
| L | lag | int in [1, n-1] | samples |
| P, k | onset period of the SS train; position of a stored window in its sweep's train (0-based) | int; int | samples; -- |
| n_w, pre_n, m | samples of a stored SS window; of its pre-pulse part; of the enlarged window | int | samples |
| t_SS(c), t_tr(c), t_floor | start of the enlarged window; the transient test's time; the floor (80 ms) | float | ms |
| r_odd(t), sigma_SS(c), kappa_tr | odd part of the polarity-mean responses; median pre-pulse SD; the test's factor (0.1, provisional) | float[n_w]; float; float | mV; mV; -- |
| x_tilde, mean window | a window minus its polarity's mean window (D-051.1); the per-sample mean over a polarity's windows | float[m]; float[m] | mV |
| q_hat(c, s) | the statistic vector of sweep s of c (SS: 5 medians over the sweep's windows; LS: the 10 values of its one window) | float[5] or float[10] | mixed |
| V_hat_c(h*), N_pairs | the data semivariogram at h* = 4 P (same polarity, same sweep, positions 4 apart); the pairs it averages | float; int | mV^2; -- |
| sigma_hat, rho_hat(L) | window SD (ddof 1); textbook autocorrelation | float | mV; -- |
| rho_adj(L) | the pre-D-028 estimator (test oracle only) | float | -- |
| theta | (sigma_f, rho_f, sigma_s, tau_s); `noise_law.NoiseLaw` | float[4] | mV, --, mV, s |
| xi | unit (standard-normal) draws of the simulator: fast, slow (n_traces, n); tilt, offset (n_traces,); `noise_law.UnitDraws` | float arrays | -- |
| E[S_0 \| theta, n], E[S_L \| theta, n], Var(x_bar \| theta, n) | closed-form moments of a demeaned window, topic doc (7a), (7b), (9) | float | mV^2 |
| q_cf_m(theta \| n_m) | closed-form statistic: sqrt(E[S_0]/(n-1)) for an SD, E[S_L]/E[S_0] for an autocorrelation (eq. 10); the fit's starting point only | float | mV; -- |
| V(h \| theta, sigma_b) | semivariogram of two window means h apart in one trace (T1) | float | mV^2 |
| gamma(k \| theta) | autocovariance (L3) | float | mV^2 |
| a, b, sigma_b(c) | per-sweep offset, tilt, the tilt's SD | float | mV, mV/s, mV/s |
| q_hat(c), q_sim(theta), q_m(theta \| n_m) | the cell's statistic vector; its model estimate; the analytic single-window median | float[d]; float[d]; float | mixed |
| W, Sigma_hat, J | weight, simulated covariance, cost | float[d,d]; float[d,d]; float | mixed |
| M, R, B, k_max | binding windows; cells per Sigma_hat; bootstrap draws; iteration cap | int | -- |
| u | the fit coordinates (log sigma_f, rho_f, log sigma_s, log tau_s) | float[4] | log mV, --, log mV, log s |
| M_cells, n_LS, L_s | replicate cells behind the SS binding function (ceil(M / K)); samples of an LS pre-window; samples of a simulated SS sweep ((train_length - 1) P + n_w) | int | --; samples; samples |
| C, kappa(C_hat), d_eff | Cholesky factor of Sigma_hat; condition number of its correlation matrix; statistics kept after dropping the non-finite ones | float[d_eff, d_eff]; float; int | mixed; --; -- |
| h* | the tilt's lag, 4 P Delta_t | float | s |
| seed_binding, seed_Sigma_k, seed_boot | per-cell child seeds (`SeedSequence([seed, specimen_id])`): the binding function's common random numbers; Sigma_hat's draws at iteration k; the bootstrap | int | -- |

## 3. Data contracts

| Item | Source / format | Shape and axis order | dtype | Units | Conventions |
|---|---|---|---|---|---|
| a window `x` for `sample_acf` | any 1-D array (data or simulation) | (n,) | float64 | mV (any unit; result dimensionless) | lags in samples, >= 1; NaN rules of 2.2 |
| LS pre-window | `noise_calibration.pre_window_samples(mono, bundle)` | (n,), n = 13250 at 50 kHz for 265 ms | float64 | mV, LJP-corrected | [t_0, t_on - 5 ms), t_0 the first stored sample |
| SS stored pulse window | `ss_pulses.npz` Format A via `load_cell_from_archive` | per pulse (n_w,), t in s, t = 0 at pulse onset, window (-10, +200) ms | float64 | mV | polarity 'dep'/'hyp'; sweep number NOT read by the loader (open, D-056) |
| SS sweep layout (data inspected 2026-10-20, two cells) | `ss_pulses.npz` keys `sweep_number` (int64, one per stored window, windows in onset order); `metadata.json['ss_extraction']` (`n_pulses_raw`, `n_pulses_ampdur_qc`, `n_pulses_qc`, `n_dep_pre_trace_qc`, `n_hyp_pre_trace_qc`, `trace_qc` with `engine`, `alpha`); `trace_qc_results.csv` (one row per pulse that entered the trace QC, in the same order, columns `pulse_index`, `polarity`, `passed`, `reason`, ...) | per SS sweep a train of 37 pulses of 0.5 ms, polarity alternating from dep (19 dep, 18 hyp), onset to onset 10025 samples = 200.5 ms at 50 kHz; the stored windows are the train's pulses that passed the writer's trace QC (engine v2, alpha 0.5), in order | -- | -- | 508282493: 2 sweeps, 74 pulses detected, 74 after the amplitude/duration QC, 40 stored (34 dropped by the trace QC); 569844159: 10 sweeps, 370, 370, 348 (22 dropped). Every gap is a trace-QC drop, and the position of every stored window in its train follows from `trace_qc_results.csv` (the stored windows are its `passed` rows, in order). Sources: cluster G12 (polarity strings, spacing) and the Google Drive copy of the two cells' archive files (counts, QC table), which reproduces the G12 strings exactly. All 18 L3_exc cells checked on the cluster 2026-10-21 (decision log D-072): every cell carries `trace_qc_results.csv` with as many rows as `n_pulses_raw` and `n_pulses_ampdur_qc`, a multiple of 37 -- 2 SS sweeps in 14 cells, 6 in 571430815, 10 in 569844159, 571483807, 571624532; engine v2, alpha 0.5 everywhere; 6-50 % of pulses dropped by the trace QC (median 27 %). Block 3 still checks each cell's per-sweep split and polarity sequence. |

## 4. Global conventions

- Units used internally: mV, s (lags in samples where the estimator needs them; ms only in printed output and column names ending in `_ms`).
- Floating-point precision: float64.
- Randomness: a `numpy.random.Generator` passed explicitly; per-cell streams spawned from (config seed, specimen id); no module-level RNG.
- Pure ASCII `.py`, LF line endings (hpc-python-compat).
- Binding decisions:
  - D-028: the sample autocorrelation is (S2), computed by statsmodels, through `noise_stats.sample_acf` only.
  - D-029: (i) `Biological Fit/` unchanged; (ii) no adjusted-form opt-in; (iii) NaN when n - L < 10, flat or non-finite window; (iv) wrapper = `noise_stats.sample_acf`; (v) statsmodels installed in the cluster env `prova` before any run.
  - D-010 (iii): no check certifies a loss or a statistic by `isfinite` alone.
  - D-017, D-033, D-034, D-048 to D-051, D-055, D-056: as stated in section 2 (blocks 2 onward).
  - D-072: the stored SS windows are the data as the archive wrote them (trace-QC'd when archived); no shape test is applied anywhere in the noise fit -- not to the data, not in the binding function's or Sigma_hat's simulator, not to the twins; each window's position in its 37-pulse train comes from `trace_qc_results.csv`.
  - D-073: each twin sweep is calibrated on its own real sweep (form open, section 8); statistics are kept per sweep as well as per cell.

## 5. Blocks

### Block 1: textbook sample autocorrelation through one wrapper (D-028, D-029)

- Module: `noise_stats.py`; call sites changed: `passive_fitting_hpc_fixed._estimate_noise` (lag-1 of a pre-stimulus window), `passive_fitting_hpc_fixed._estimate_residual_noise_at_mle` (lag-1 of the pooled residuals), `noise_calibration.lag_autocorr`, `smoke_ih_recovery._estimator_band` / `_acf_gap_band` (the Monte Carlo bands R8/R10 compare the data's statistic with).
- Public API: `sample_acf(x: array-like 1-D, lags: int | Sequence[int]) -> np.ndarray` (float, one value per lag, in the order given).
- Inputs: `x` (n,) float, any unit; `lags` in samples.
- Outputs: rho_hat(L) of (S2) per lag; NaN per the rules of 2.2; |rho_hat| <= 1 wherever finite.
- Parameters: `MIN_PRODUCTS = 10` (D-029 (iii)); no other knob.
- Library calls relied on: `statsmodels.tsa.stattools.acf(x, adjusted=False, nlags=max_lag, fft=True)`.
- Custom code: none (the flat-window guard only avoids statsmodels' 0/0 RuntimeWarning).
- Test oracles (`smoke_noise_stats.py`, N1-N6): N1 equals `acf(x, adjusted=False)` and the hand sum (S2) to 1e-14 at n in {12, 500, 4750, 13250} and lags up to n-10 (float64 roundoff on sums of ~1e4 terms); N2 equals rho_adj(L) * (n-L)/n to 1e-12 (D-028.2; the old formula written out in the test only); N3 NaN rules (n - L < 10, L < 1, flat window, non-finite sample; a scalar lag gives a length-1 array); N4 |rho_hat| <= 1 on 200 random windows; N5 the default is textbook: on a window where the two forms differ by more than 1e-3 at lag n-10, the result equals the textbook value, not the adjusted one; N6 this file, `noise_stats.py` and the four changed files are pure ASCII and LF.
- Data flow: consumed by every autocorrelation statistic of blocks 3 onward; by the fitter's `noise_rho_lag1` / `residuals_rho_lag1`; by the noise table's `rho_*`, `acf_*`, `ar1_*` columns (all move by (n-L)/n, D-028.2).
- Confirmed: yes, 2026-10-07 (D-028, D-029 by the user) and 2026-10-16 (never applied: ships here).
- Status: smoke-tested 2026-10-18 (sandbox: N1-N6 6/6, mutation-tested; `smoke_ls_baseline_qc` 10/10, `smoke_ih_fit` 12/12, `smoke_ih_recovery` 14/14; skipped: `regression_passive_identity.py`, oracle folder and archive on the cluster).

### Block 2: the noise law -- simulator from unit draws and closed-form moments (D-017, D-034, D-056 (i))

- Module: `noise_law.py` (model layer: no I/O, no plotting, no module-level RNG); smoke `smoke_noise_law.py`.
- Public API: `NoiseLaw(sigma_f, rho_f, sigma_s, tau_s)` (frozen dataclass; `validate` in `__post_init__`; `as_array`/`from_array`; `rho_s(dt)`, `tau_f(dt)`); `autocovariance(law, dt, lags) -> float[len(lags)]` (L3); `draw_units(rng, n_traces, n_samples) -> UnitDraws` (fast, slow (n_traces, n); tilt, offset (n_traces,); drawn in that fixed order); `simulate_traces(law, dt, xi_fast, xi_slow, *, xi_tilt=None, sigma_b=0.0, xi_offset=None, sigma_a=0.0) -> float64[n_traces, n]` (L1)-(L2); `simulate_sweep(law, dt, n_samples, units, index, *, sigma_b, sigma_a) -> float64[n]` (one continuous trace, D-056 (i)); `var_window_mean(law, dt, n)` (7a); `expected_s0(law, dt, n)` (7b); `expected_sl(law, dt, n, lags)` (9); `closed_form_sd(law, dt, n)`; `closed_form_acf(law, dt, n, lags)` (10); `visible_share(z)` (8); `window_mean_cov(law, dt, n1, n2, offset)`; `semivariogram(law, dt, n1, n2, offset, sigma_b=0.0)` (T1's V, exact for the sampled law); `semivariogram_continuous(law, dt, T1, T2, h, sigma_b=0.0)` (topic doc eq. 25 as written; oracle).
- Inputs: theta in mV, --, mV, s with rho_f the lag-1 coefficient AT the `dt` passed (D-014; the transfer between sampling rates, D-017 (f), is not done here); dt in s; unit draws float (n_traces, n >= 2); sigma_b in mV/s, sigma_a in mV; lags and offsets in samples.
- Outputs: traces in mV, float64, (n_traces, n); each recursion started in its stationary law (e_0 = sigma xi_0); the tilt of trace r is sigma_b xi_tilt[r] (t_j - t_bar), t_bar = (n-1) dt / 2 the trace's own mid-time, so it sums to zero over the trace (any constant is invisible to every demeaned statistic and to the within-trace semivariogram); moments in mV^2; `expected_sl`/`closed_form_acf` NaN outside 1 <= L <= n-1.
- Parameters: none (no knob; sizes come from the caller's config, Block 4).
- Library calls relied on: `scipy.signal.lfilter([1], [1, -rho], eta, axis=-1)` for both AR(1) recursions; `numpy.expm1` for 1 - rho_s^2 and z - 1 + e^{-z}.
- Custom code (marked `# Custom:` at the definition): the exact first moments (7a), (7b), (9) of a demeaned window (zeta_j from the cumulative sum of gamma, O(n)); the window-mean covariance by lag counts over sample pairs (O(n1 + n2)). No library computes either.
- Test oracles (`smoke_noise_law.py`, M1-M9): M1 brute force -- (7a), (7b), (9) against the full n x n covariance matrix (E[S_L] = sum_j (P Gamma P)_{j,j+L}) at n = 240, 5 lags, 3 laws, 1e-10 relative; M2 Monte Carlo (4000 traces x 2000 samples, cohort law) -- mean S_0, S_1, S_50 and the across-trace autocovariance at lags 0, 1, 50 within 3.5 SE of the closed forms; M3 stationarity -- first- and last-sample variance equal gamma(0) within 3.5 SE; M4 window means -- `window_mean_cov` equals brute force (1e-10; overlapping, disjoint, negative offset), the Monte Carlo covariance of two 190 ms means 200 ms apart within 3.5 SE, exact-discrete V against eq. (25) within 3 dt/tau_s (slow-only law); M5 tilt -- Monte Carlo V equals V(h|theta,0) + sigma_b^2 h^2/2 within 3.5 SE, the tilt sums to 0; M6 contract -- shapes, float64, finite, bit-identical on the same draws, different on others, `simulate_sweep` == a row of `simulate_traces`, six invalid laws refused, draw order fixed; M7 visible share -- E[S_0]/(n gamma(0)) equals v(z) within 2z/n at z in {0.25, 1.26, 6.67}; M8 regression -- q_cf at theta_cf = (0.0380 mV, 0.175, 0.0508 mV, 36.6 ms) reproduces the 14 q^cf values of topic doc 3.5.3 within 3e-4 mV and 3e-3; M9 ASCII/LF. Mutation-tested 2026-10-19: a non-stationary start (M2, M3 fail), the Var term dropped from (9) (M1, M2, M8), an uncentred tilt (M5), 1 - rho_s in place of 1 - rho_s^2 (M2, M4, M5), an off-by-one pair count (M4 brute force) each FAIL.
- Data flow: consumed by Block 3 (`noise_windows` cuts the enlarged SS and LS windows from traces of this simulator at the real sweep structure), Block 4 (`noise_fit`: binding function and Sigma_hat from `simulate_traces` on frozen `UnitDraws`; `closed_form_sd`/`closed_form_acf` for the start; `semivariogram` for (T1)), Block 5 (the generator's OU term through the same simulator).
- Confirmed: yes -- D-017, D-034, D-056 (i) (user); the exact-discrete V in place of eq. (25)'s continuous form is the assistant's refinement, recorded in 2.4 and in the decision log's changelog (2026-10-19).
- Status: smoke-tested 2026-10-19 (sandbox: M1-M9 9/9, about 5 s; mutation-tested as above).

### Block 3: windows and statistics -- the archive's SS layout, the enlarged windows, the LS pre-windows and the D-051 statistic vector (D-051, D-072, D-073, D-076)

- Modules: `noise_windows.py` (data layer: loading, layout, windows, the transient test, the subtraction; no fitting, no plotting), `noise_cell_stats.py` (analysis layer: the statistics per window, per sweep and per cell, the bootstrap SE, the data semivariogram; no I/O); smoke `smoke_noise_windows.py`.
- Public API (`noise_windows`): `WindowConfig` (frozen dataclass: `t_ss_floor_ms=80.0` (D-076), `kappa_tr=0.1` and `transient_bin_ms=5.0` (D-051, provisional), `ss_pre_ms=10.0`, `ls_pre_gap_ms=5.0`, `ls_first_ms=50.0`, `ls_last_ms=132.5`, `ls_polarities=("hyp", "dep")`, `ls_amp_min_pA=5.0`, `ss_period_samples=None` (measured), `min_overlap=100`, `fallback_t_ms=190.0`, `transient_floor_z=None`); `load_ss_windows(spec_dir, cfg) -> SSWindows` (fields: `specimen_id`, `fs`, `dt`, `t` (n_w,), `v` (K, n_w) mV, `polarity_is_dep` (K,), `sweep_number` (K,), `position` (K,) int, `train_length` int, `period_samples` int, `pre_n` int, `n_raw`, `n_sweeps`); `load_ls_prewindows(spec_dir, cfg) -> LSWindows` (fields: `specimen_id`, `fs`, `dt`, `sweep_number` (K_LS,), `amplitude_pA` (K_LS,), `polarity` (K_LS,) str, `pre` list of K_LS float64 arrays, `n_pre` (K_LS,), `onset_s` (K_LS,)); `measure_period(ss, min_overlap) -> int`; `transient_start(ss, cfg) -> TransientResult` (`t_tr_ms`, `t_ss_ms`, `sigma_ss_mV`, `r_odd` (n_w,), `bin_edges_ms`, `bin_exceeds` (n_bins,) bool, `flags` tuple of 'transient_fail' / 'ss_enlarged_none' / 'one_polarity' / 'floor_binds' / 'noise_floor_threshold', `threshold_mV`, `floor_mV`); `enlarged_windows(ss, t_ss_ms) -> (X (K, m), j0)`; `subtract_mean_response(X, polarity_is_dep) -> (X_tilde, mean_dep (m,), mean_hyp (m,))` (D-051.1; a polarity with no window gets a zero mean); `ls_segments(pre, fs, cfg) -> (first (n1,), last (n2,))`; `LayoutError`.
- Public API (`noise_cell_stats`): `SS_LAGS_MS=(0.02, 0.1, 1.0, 10.0)`, `LS_FULL_LAGS_MS=(0.02, 0.1, 1.0)`, `LS_LAST_LAGS_MS=(0.02, 0.1, 1.0, 10.0)`, `STAT_LABELS` (15, in the order of 2.2), `STAT_BLOCK` (15 of 'ss'/'ls'); `window_stats(X, fs, lags_ms) -> (K, 1 + len(lags))` (sigma_hat then rho_hat per lag, through `noise_stats.sample_acf`); `ss_window_stats(X_tilde, fs) -> (K, 5)`; `ls_window_stats(pre_list, fs, cfg) -> (K_LS, 10)`; `StatisticVector` (`q` (15,), `labels`, `block`, `n` (15,) window lengths, `K` (15,) counts, `d`); `statistic_vector(ss_stats, m, ls_stats, n_ls) -> StatisticVector` (NaN-ignoring medians; a statistic with no window is NaN and its K is 0); `per_sweep_vectors(stats, sweep_number) -> dict sweep -> median vector`; `bootstrap_median_se(per_window (K, d), B, rng) -> (d,)` (window-level bootstrap of the median, D-049's W_0 input); `window_mean_semivariogram(X_tilde, ss, lag_periods=4) -> (V_hat, N_pairs)` (half the mean squared difference of the window means of same-polarity, same-sweep windows whose positions differ by `lag_periods`; NaN with 0 pairs).
- Inputs: a Phase-0 specimen directory with `ss_pulses.npz` (Format A, `sweep_number`), `trace_qc_results.csv`, `metadata.json` (`ss_extraction.n_pulses_raw`, `n_pulses_ampdur_qc`, `n_pulses_qc`, `ls_sweeps`), `ls_sweeps.npz`. Voltages in mV as stored (LJP-corrected by the writer), f_s in Hz.
- Layout rule (D-072): the QC table's rows are the sweep trains in order, each of length n_raw / n_sweeps (an integer, else `LayoutError`); row r's position is r mod train_length and its sweep the n-th entry of `metadata.json['ss_extraction']['sweep_numbers']` (the writer's visiting order; the npz's sweep numbers ascending when that list is absent); the `passed` rows, in order, are the stored windows. Refused (`LayoutError`, with the reason) when: rows != n_pulses_raw or != n_pulses_ampdur_qc; passed rows != K; the passed rows' polarities or sweep counts differ from the npz; a train's polarity does not alternate; an npz sweep number not in the metadata list; the measured period disagrees with `ss_period_samples` when given, or no adjacent stored pair exists to measure it; two adjacent stored windows disagree on their overlap.
- Outputs: windows in mV, float64; positions 0-based; statistics dimensionless except the SDs (mV); the semivariogram in mV^2; every array's first axis is the window.
- Parameters: the `WindowConfig` fields above; `transient_bin_ms` bins start at t = 0; the last bin is the one ending at or before 200 ms.
- Library calls relied on: `numpy` (loading, medians with `nanmedian`), `statsmodels` through `noise_stats.sample_acf`, `csv`/`json` (standard library).
- Custom code (marked `# Custom:`): the layout reconstruction from the QC table (no library reads the writer's files); the overlap test of adjacent windows (as in `noise_fit_gates.onset_spacing`, re-implemented without the tools module so the data layer imports nothing from the tools); the transient test; the window-mean semivariogram over position pairs.
- Test oracles (`smoke_noise_windows.py`, W1-W10): W1 layout -- a writer-like fixture (2 sweeps x 37 pulses, period 10025, RC response + Block-2 noise, 11 pulses dropped by a planted QC table, stored windows in order) is read back with the planted positions and period exactly; tampered tables (a wrong count, a swapped polarity, a non-alternating train, a wrong `ss_period_samples`) each raise `LayoutError`; W2 transient test -- on the noise-free single-pulse RC response (white noise of SD sigma_SS before t = 0 only) the test returns exactly the first bin boundary after which every bin mean of the response is below kappa sigma_SS; floored at 80 ms when earlier; flagged `ss_enlarged_none` when the response lasts past 190 ms; with Block-2 noise and 63 windows the test as written is noise-limited (reported) and with `transient_floor_z = 3` it returns the response's own time at that threshold within two bins, no fail flag; W3 subtraction -- the mean of the subtracted windows is 0 per sample and polarity (1e-12); with a response that is a common vector per polarity (the single-pulse RC response) the statistics after subtraction equal the noise's alone to 1e-10, and on the fixture's stored windows (whose previous-pulse tails differ, pulses having been dropped) within 1e-3; W4 statistics -- `window_stats` equals `np.std(ddof=1)` and `sample_acf` per window (1e-14); the 15 labels, blocks, n and K are those of 2.2; the LS pre-window of a fixture sweep equals `noise_calibration.pre_window_samples(mono, bundle)` for the same sweep loaded through the monolith (the fitter's rule; SkipTest when NEURON is not importable); W5 per-sweep vectors -- keys are the sweep numbers, each the median over that sweep's windows, and the cell vector equals the median over all windows; W6 bootstrap SE -- on K = 60 iid Gaussian window values the SE of the median is within 20 % of 1.2533 sigma / sqrt(K) (sampling spread at B = 2000 and 40 repeats), bit-identical for the same seed; W7 semivariogram -- on 400 sweeps simulated with `noise_law.simulate_sweep` at a known law the pair count is the fixture's and V_hat equals `noise_law.semivariogram(law, dt, m, m, 4 P)` within 3.5 SE; W8 enlarged window -- m = n_w - pre_n - round(t_SS f_s / 1e3), the first sample is the first with t >= t_SS; W9 LS -- every sweep of the fixture gives one pre-window of 13250 samples at 50 kHz (265 ms), hyp and dep both read, a sweep below 5 pA skipped, the first 50 ms and last 132.5 ms segments of the right length; W10 ASCII/LF.
- Data flow: consumes the archive (section 3) and `noise_law` (W7 only); produces q_hat(c), q_hat(c, s), the per-window matrices, SE_hat (W_0 of Block 4), V_hat_c(h*), the layout (positions, period, train length, K per sweep and polarity) that Block 4's simulator reproduces, t_SS(c) and the flags; consumed by Block 4 and by Block 6's driver.
- Confirmed: D-051, D-072, D-073, D-076 (user). The assistant's choices, flagged: every LS sweep's pre-window enters (hyp and dep); the cell-level mean window is subtracted for the per-sweep statistics too; `fallback_t_ms = 190` raises the flag and Block 3 still returns the (short) windows.
- Status: smoke-tested 2026-10-23 (sandbox: W1-W10 10/10, about 20 s, the monolith oracle of W4 run; the fixture writer gained a `sigma_b` option for Block 4's smoke, default 0 = unchanged; mutation-tested with eight planted defects -- positions off by one, a polarity skipped in the subtraction, lag truncation, cross-sweep semivariogram pairs, the LS pre-window gap dropped, the floor not applied, an overlap test on the head only, the bin max in place of the mean -- each FAILs at least one check). Not run here: the real archive (cluster).

### Block 4: the per-cell fit -- binding function, simulated covariance, tilt and the iteration (D-033, D-048 to D-051, D-056)

- Module: `noise_fit.py` (model and analysis layer: the simulator of a cell's windows at the real layout, the binding function, Sigma_hat, the tilt, the inner solve and the iteration; no I/O beyond an optional log callable, no plotting); smoke `smoke_noise_fit.py`. The per-sweep stage of D-076 (i) is Block 4b (`noise_sweep_fit.py`), specified when written.
- Public API: `NoiseFitConfig` (frozen dataclass: `M_binding=3000`, `R_iter=600`, `R_final=3000`, `B_boot=2000`, `k_max=6`, `tol_u=0.01`, `starts_after_first=1`, `max_nfev=15`, `ftol=1e-4`, `xtol=1e-3`, `gtol=1e-8`, `fd_rel_step=(1e-3, 1e-2, 1e-3, 1e-3)`, `log_sigma_bounds=(log 1e-4, log 10)`, `rho_bounds=(1e-3, 0.999)`, `log_tau_bounds=(log 1e-4, log 10)`, `lag_periods=4`, `chunk_windows=500`, `seed=20261023`); `CellLayout` (frozen: `specimen_id`, `fs`, `dt`, `n_w`, `pre_n`, `j0`, `m`, `period_samples`, `train_length`, `sweep_numbers` (n_sweeps,), `positions` tuple of int arrays (one per sweep, the stored positions), `polarity_is_dep` (K,), `sweep_number` (K,) per window, `n_ls`, `ls_sweep_number` (K_LS,); `K`, `K_LS`, `L_s` properties; `from_windows(ss: SSWindows, ls: LSWindows, t_ss_ms) -> CellLayout`, refusing LS pre-windows of unequal length); `CellTarget` (`q_hat` (15,), `per_window_ss` (K, 5), `per_window_ls` (K_LS, 10), `V_hat`, `N_pairs`, `labels`; `from_stats(...)`); `simulate_cell_windows(law, sigma_b, layout, rng) -> (ss_stats (K, 5), ls_stats (K_LS, 10))` (one replicate cell; the sweeps continuous; the mean window of the replicate's own windows subtracted); `binding_function(law, sigma_b, layout, cfg, seed) -> q_sim (15,)` (CRN by re-seeding; parts selectable by `which='ss'|'ls'|'both'`); `hd_median(A (N, d)) -> (d,)` (Harrell-Davis, NaN-ignoring per column); `simulated_covariance(law, sigma_b, layout, R, rng, per_sweep=False) -> SigmaResult` (`Sigma` (15, 15), `vectors` (R, 15), `per_sweep` dict sweep -> (R, 5 or 10) when asked); `closed_form_vector(law, layout) -> (15,)`; `tilt_from_semivariogram(V_hat, N_pairs, law, layout, lag_periods) -> TiltResult` (`sigma_b`, `V_model`, `h_star`, `flags`); `solve_inner(q_hat, C, law_starts, sigma_b, layout, cfg, seed, mask, residual=None) -> InnerResult` (`law`, `u`, `J`, `nfev`, `start_index`, `status`, `message`, `starts` (u_0, J) per start, `at_bound`); `closed_form_start(q_hat, SE, mask, layout, cfg) -> NoiseLaw`; `law_to_u`, `u_to_law`; `correlation_condition_number(Sigma)`; `fit_cell(target, layout, cfg, log=None) -> CellFitResult` (`specimen_id`, `law`, `u`, `sigma_b`, `J` (at the reported pair), `dof`, `p_value`, `kappa`, `Sigma` (d_eff, d_eff), `labels_kept`, `stats_dropped`, `q_hat`, `q_sim`, `SE` (W_0's; NaN where dropped), `n_solves`, `converged`, `flags` ('stats_dropped:<labels>', 'se_from_simulation:<labels>' when a finite statistic had no bootstrap SE -- a single window -- and W_0 took the simulated single-window spread x 1.2533 / sqrt(K) at the closed-form start, 'no_plateau', 'sigma_jitter', 'at_bound:<coordinates>', 'tilt_clipped', 'tilt_no_pairs'), `history` (per solve: weight, R, u, J, sigma_b, nfev, start_index, status, at_bound, tilt_flags, seconds), `per_sweep_cov` dict sweep -> covariance of its vector (from the final Sigma_hat run, Block 4b's weights), `config`, `seconds`; `to_dict()` JSON-ready, NaN as null).
- Inputs: Block 3's `SSWindows`, `LSWindows`, t_SS(c), the per-window statistic matrices, V_hat_c(h*) and N_pairs; theta in mV, --, mV, s; sigma_b in mV/s.
- Outputs: theta_hat as `noise_law.NoiseLaw`; u in the fit coordinates; sigma_b_hat in mV/s; J dimensionless; Sigma in the statistics' units squared.
- Parameters: the `NoiseFitConfig` fields above; defaults are D-048 to D-051 and D-056 (M, R, B, k_max, tol, three starts) and the assistant's choices of 2.3 (bounds, steps, tolerances, warm starts, chunking).
- Library calls relied on: `scipy.optimize.least_squares(method='trf', bounds, diff_step, ftol, xtol, gtol, max_nfev)`; `scipy.linalg.cholesky`, `scipy.linalg.solve_triangular`; `scipy.stats.mstats.hdquantiles`; `scipy.stats.chi2.sf`; `numpy.cov`, `numpy.linalg.cond`; `numpy.random.SeedSequence`; Blocks 1-3 for the statistics, the law and the layout.
- Custom code (marked `# Custom:`): the layout-true cutting of the enlarged windows from a continuous simulated sweep (position k -> samples [k P + j0, k P + n_w)); the iteration of 2.3 (no library implements indirect inference with a simulated-covariance weight).
- Test oracles (`smoke_noise_fit.py`, F1-F10, reduced sizes -- the layout of Block 3's fixture, 2 x 37 pulses with 11 drops, K = 63, 5 LS windows of 13250 samples): F1 cutting -- the windows `simulate_cell_windows` cuts from a sweep equal, bit for bit, the windows cut by an independent loop over the stored positions from the same `simulate_sweep` trace; F2 subtraction -- at a white law (rho_f = 1e-3, sigma_s = 1e-6 mV) the pooled mean of sigma_hat^2 over the SS windows of 300 replicate cells equals sigma_f^2 (1 - 1 / K_p) (K_p the polarity's windows; the exact expectation of S_0 / (n - 1) after subtracting the mean of K_p iid windows) within 3.5 SE, and is lower than sigma_f^2 by more than 3.5 SE; F3 CRN -- `binding_function` is bit-identical twice at one theta and seed, differs at another seed, and moves by less than 0.1 times its Monte Carlo SE (from the pooled windows) for a 1e-6 relative change of theta; `simulated_covariance` is bit-identical on one seed; F4 Sigma_hat -- at the white law (sigma_s = 0, rho_f = 1e-3), with R = 120, the variance of the SS SD statistic (the median over K windows of sigma_hat) is within 40 % of (pi / 2) sigma_f^2 / (2 (m - 1) K) (asymptotic variance of the median of K iid near-normal estimates) and the LS single-window variance of sigma_hat within 40 % of sigma_f^2 / (2 (n_LS - 1)); Sigma_hat is symmetric positive definite and `cond` finite; the per-sweep vectors carry every SS and LS sweep with shapes (R, 5) and (R, 10); F5 tilt -- 200 replicate cells from `simulate_cell_windows` at sigma_b = 0.1 mV/s (9600 pairs) give, through Block 3's `window_mean_semivariogram` pooled over the replicates and `tilt_from_semivariogram`, sigma_b within 3.5 SE (SE by the delta method: each pair's half squared difference has variance 2 V^2), sigma_b = 0 with `tilt_clipped` when V_hat < V_model, `tilt_no_pairs` without a pair, h* = 4 P Delta_t; F6 self-consistency of the inner solve -- q_hat := `binding_function(theta*, seed)` exactly, C = diag of the simulated single-window spread / sqrt(K), starts at tau_s / 2 and 2 tau_s: `solve_inner` returns u within 0.01 of u(theta*) with J < 1e-3 (M = 60; the optimiser plumbing: coordinates, bounds, steps, the start bookkeeping); F7 a reduced fit -- the fixture cell (theta_cf, sigma_b = 0.1 mV/s planted in its 2 SS and 6 LS sweeps), fitted with M = 60, R_iter = 40, R_final = 60, B = 400, k_max = 2, max_nfev = 6, one start after the first: `fit_cell` returns a `CellFitResult` with every field, |u_hat - u*| <= (0.04, 0.03, 0.4, 1.0) (4 x the whole-cell identifiability SDs of `sweep_identifiability.py`, inflated for M = 60), sigma_b_hat equal to (T1) recomputed at the reported law (clipping included; its value is not tested: two tilt draws, see 2.3 (h)), J equal to r^T Sigma_hat^-1 r at the reported (theta_hat, sigma_b_hat) within 1e-8, history length = n_solves with the first solve 'W0' and the last 'final', dof 11, kappa finite, 7 per-sweep covariances, `to_dict` round-trips through JSON; F8 `hd_median` equals `hdquantiles(prob=0.5)` (1e-12), ignores NaN per column, lies within half a standard error of the median (1.2533 sigma / sqrt(N)) of the sample median on symmetric data, and returns a single value as is; F9 masks -- a q_hat with two NaN entries, fitted with k_max = 0 and max_nfev = 1, gives d_eff = 13, those labels in `stats_dropped` and the flag, Sigma (13, 13), SE NaN there, dof 9, two solves and 'no_plateau'; F10 ASCII/LF. The smoke takes about 100 s (a subset by name: `python smoke_noise_fit.py F1,F5`).
- Data flow: consumes Blocks 1-3; produces theta_hat(c), sigma_b_hat(c), Sigma_hat (final) with the per-sweep covariances for Block 4b, the fit report fields for the noise table and manifest (Block 6: columns `noise_sigma_f_mV`, `noise_rho_f`, `noise_sigma_s_mV`, `noise_tau_s_ms`, `noise_sigma_b_mV_per_s`, `noise_J`, `noise_dof`, `noise_p`, `noise_kappa`, `noise_n_solves`, `noise_flags`); consumed by Block 4b and Block 6.
- Confirmed: D-033, D-048 to D-051, D-056 (user). The assistant's choices, flagged: items (c)-(e), (h) of 2.3; the chi-square tail probability as a report field only.
- Status: smoke-tested 2026-10-23 (sandbox: F1-F10 10/10, about 100 s; mutation-tested with ten planted defects -- the cut off by one, the subtraction skipped, CRN broken (a fresh generator), the tilt's sign, h* with the window length added, exp on rho_f in `u_to_law`, the mask not applied to Sigma_hat, the sample median for Harrell-Davis, J reported with W_0's weights, sigma_b not fed to the SS simulator -- each FAILs the check named for it). Not run here: a fit at the full M and R (hours; the cluster, Block 6's S0).

### Tools (not blocks): cluster-side checks and the SS archive inspection

- Files: `noise_fit_gates.py` (read-only helpers), `cluster_gates_noise_fit.sh` (the runner), `smoke_noise_fit_gates.py` (T1-T8).
- `python noise_fit_gates.py bytes [--folder DIR]` -> `BYTES: ok|FAIL (...)`, exit 0/1: top-level `*.py` pure ASCII; `*.sh`, `*.pbs`, `mod/*.mod` without CR.
- `python noise_fit_gates.py compile [--folder DIR]` -> `COMPILE: ok|FAIL (...)`, exit 0/1: every top-level `*.py` compiles in memory (no `.pyc`).
- `python noise_fit_gates.py ss-pulses DIR [DIR ...]` -> per specimen: the keys, dtypes and shapes of `ss_pulses.npz` (Format A or B); per sweep number (or, without one, per chain of adjacent windows) the window count, the polarity string (`+` dep, `-` hyp), the adjacent consecutive pairs and their onset spacing in ms, the gaps; the `metadata.json` keys about SS; then one line per cell, `SS_PULSES <id>: format=A|B sweep_number=yes|no n_pulses=.. n_sweeps|n_chains=.. per_sweep|per_chain=.. adjacent=a/b opposite_polarity_adjacent=c/a spacing_ms=.. fs_Hz=.. window_ms=..`; exit 0 when every cell was read and carries sweep numbers, else 1. Adjacency: windows k, k+1 of one sweep whose onsets are D samples apart (D < window length n) agree exactly on n - D samples of voltage and of current; the smallest D >= 1 with at least `MIN_OVERLAP` = 100 agreeing samples is the spacing (custom: no library function; a 16-sample vectorised pre-filter keeps it fast on ADC-quantised traces, where single values recur).
- `bash cluster_gates_noise_fit.sh` (FULL) or `QUICK=1 bash ...`: one `PASS|FAIL|SKIP  <check>  -- <seen>` line per check -- G0 interpreter, G1 numpy/scipy/statsmodels, G2 Blocks 1-4 committed (commit subjects) and no tracked file of the folder modified, G3 bytes, G4 compile, G5 `smoke_noise_stats` 6/6, G6 `smoke_noise_law` 9/9, G6b `smoke_noise_windows` 10/10 (Block 3, added 2026-10-23), and in FULL mode G6c `smoke_noise_fit` 10/10 (Block 4, about 2 min, added 2026-10-23), G7 NEURON/skopt and `x86_64/special` (built with `nrnivmodl mod` under the submit scripts' `.nrnivmodl.lock` only when missing or older than `mod/*.mod`), G8 `smoke_ih_fit` 12/12, G9 `smoke_ls_baseline_qc` 10/10, G10 `smoke_ih_recovery` 14/14, G11 `regression_passive_identity.py` PASS on `$IH_ARCHIVE_GROUP/specimen_$REG_SPECIMEN`, G12 `noise_fit_gates.py ss-pulses` on `$NPZ_SPECIMENS`; then `GATES noise-fit: p passed, f failed, s skipped` and `VERDICT: SETTLED|NOT SETTLED -- <reason>` (SETTLED only in FULL mode with nothing failed or skipped). A check passes only on a line it must see (the suite's own summary line), never on an exit status. Logs in `$GATE_LOG_DIR` (default `~/ih_fit_gate_logs/<date_time>/`), never in the repository; no tracked file is written. Exit 0 iff nothing failed.
- Defaults: `IH_ARCHIVE_GROUP=/davinci-1/home/ldellamea/Human Neurons Fitting/L3_exc`, `REG_SPECIMEN=508282493`, `NPZ_SPECIMENS="508282493 569844159"`, `PYTHON=python`.
- Test oracles (`smoke_noise_fit_gates.py`): T1 planted CR and non-ASCII files flagged, nothing else; T2 a syntax error named, no `.pyc`; T3 a writer-like Format-A fixture (two sweeps of an ADC-quantised trace, spacings 200.000 and 200.040 ms, one pulse dropped) gives the exact expected summary and the gap; T4 Format B the same; T5 no `sweep_number`: three adjacency chains; T6 no false adjacency in 200 independent quantised pairs, a 50-sample overlap refused unless the minimum is lowered; T7 the runner's QUICK mode passes G0, G1, G3-G6b and cannot settle, a bad interpreter fails G0 with exit 1; T8 ASCII/LF. Mutation-tested 2026-10-20 (no overlap minimum, sweep numbers ignored, a wrong ms scale, `mod/` not scanned, a check certified on exit status: each caught).
- Status: smoke-tested 2026-10-20 (sandbox: T1-T8 8/8). The FULL runner was run in the sandbox on `main` + Block 2 + these tools, with `Biological Fit/` and `Synthetic Passive fit Test/` from `main` and a synthetic archive group; its result is in the decision log's changelog of 2026-10-20. Not run here: G11 on a real cell and G12 on the real archive (cluster). Run on the cluster 2026-10-20 (FULL, `main` at `ca028ea`, env `prova`): 13 passed, 0 failed, 0 skipped, VERDICT SETTLED; G11 PASS on 508282493; G12 read 508282493 and 569844159 (section 3).

## 6. Pipeline entry points

| Command | Purpose | Minimal configuration for testing | Expected runtime |
|---|---|---|---|
| `python smoke_noise_stats.py` | Block 1 smoke (N1-N6) | none | seconds |
| `python smoke_noise_law.py` | Block 2 smoke (M1-M9) | none | about 5 s |
| `python smoke_noise_fit_gates.py` | tools smoke (T1-T8) | none | about 15 s |
| `python smoke_noise_windows.py` | Block 3 smoke (W1-W10) | none (W4's oracle needs NEURON for the monolith import, else skipped) | about 30 s |
| `python smoke_noise_fit.py` | Block 4 smoke (F1-F10, reduced M and R) | none | about 100 s |
| `bash cluster_gates_noise_fit.sh` (`QUICK=1` for the fast half) | every cluster-side check of Blocks 1-4 and the SS archive read, one PASS/FAIL/SKIP line each and a VERDICT | the conda env active; on the cluster nothing else | about 17 min (QUICK < 1 min) |
| `python smoke_ih_recovery.py`, `python smoke_ih_fit.py`, `python smoke_ls_baseline_qc.py` | existing suites, re-run after Block 1 (R8/R10 bands now through `sample_acf`) | NEURON + compiled `mod/` | minutes |
| `python regression_passive_identity.py --ref-dir "../Biological Fit" --archive-cell <dir>` | bit-for-bit gate against the oracle folder | the oracle folder and one archive cell | minutes |

## 7. Not executable in the sandbox

- `regression_passive_identity.py`: `Biological Fit/` is not in the zip baseline and the archive is on the cluster; skipped here, to be run on the cluster (command in README).
- Anything reading the real archive (`noise_calibration.py --group-dir`, S3, G12 of `cluster_gates_noise_fit.sh`).
- G11 of `cluster_gates_noise_fit.sh` on a real archive cell (here only on a synthetic cell, with `Biological Fit/` from `main`).

## 8. Open questions

- ~~Does `ss_pulses.npz` carry sweep numbers? The loader reads none (D-056 open point; cluster check requested 2026-10-18).~~ [answered at source 2026-10-20; confirmed on the files 2026-10-20 by G12 on the cluster: both cells read carry `sweep_number`] The archive writer (`Passive Features/Allen Institute Data/save_alleninstitute_data (6).py`, `_archive_one_cell`, every copy in that file) stores `sweep_number` (int64, one per pulse) in Format A and B; a sweep's windows in onset order (`_detect_pulses_in_current` scans the current trace); each window (-10, +200) ms around its onset; pulses failing the amplitude/duration QC (200 +/- 30 pA, 0.5 +/- 0.5 ms) or too close to a trace boundary dropped silently; onset times not stored. Its docstring, citing the Allen tech paper: all 20 pulses of the SS protocol, alternating polarity, 200 ms apart, lie in ONE sweep. [corrected 2026-10-20: one sweep holds the whole train, but in the archive's cells the train is 37 pulses, 200.5 ms onset to onset, and the writer's trace QC drops pulses as well (section 3, data inspected)] Adjacency and the true onset spacing follow from the exact overlap of consecutive windows (tools, section 5).
- Block 5 (the twins' generator): `synthetic_ground_truth.simulate_ss_protocol` gives every SS pulse its own sweep number and independent noise. Under D-017 (per-sweep offset and tilt) and D-056 (i) (continuous sweeps in the fit's simulator), the twins' SS sweeps should carry the real sweep structure -- 20 pulses per sweep [corrected 2026-10-20: 37 pulses, 200.5 ms onset to onset, section 3] with the slow component and the tilt continuous across them -- or the twins' SS windows (and the SS bundles the I_h fit averages) are independent where the real cells' are not. To settle with the user when Block 5 is specified.
- Root `specs/SPEC.md` Coverage pointing to this file (user's call, D-056 (ii)).
- (added 2026-10-20) The stored SS windows are a selected subset: they passed the writer's trace QC (section 3), a test on the lag-1 structure of each pulse's residual after a bounded single-exponential fit over [offset + 2 ms, + 40 ms], the timescales of the law's slow part. The fit's simulator does not apply it. Check SQC (standalone script `sqc_trace_qc_selection.py`, not in the repository; the repository's `trace_qc_v2.py` with the archive's configuration on 2-sweep cells simulated with the Block-2 law, RC response 113 MOhm / 28.1 ms) compared the D-051 SS statistics of the kept windows with a random subset of the same per-polarity counts: Gaussian noise of the law alone makes the QC drop 12.9 % of pulses (theta_cf) and 12.0 % (sigma_s = 0.025 mV); the selection shifts the medians by -0.17 to -0.19 per-cell SD at t_SS = 40 ms, by at most 0.09 SD at t_SS = 80 ms and 0.08 SD at 120 ms, and V_hat at 4 periods by at most 0.07 SD (for a 10-sweep cell the SD is about 2.2 times smaller). The archive's own drop rate in 508282493 (46 %) is far above the noise-only rate, so most of its drops remove structure the law does not have; the check does not cover those. [decided 2026-10-21, D-072: no shape test anywhere in the noise fit; every L3_exc cell's layout is reconstructable, so no fallback is needed and Block 3 refuses a cell whose table, counts or polarity sequence disagree.] Still open: the floor t_SS(c) >= 80 ms and the per-cell excess-drop label (the assistant's proposal).
- (added 2026-10-23, Block 4) The tilt sigma_b(c) of a 2-sweep cell (14 of 18) is estimated from two draws of b (2.3 (h)): keep per cell as D-034 says (and report its uncertainty), pool the tilt across cells (a cohort sigma_b or a shrinkage towards it, as D-076 does for sigma_s per sweep), or draw on the LS sweeps' slopes as well (each LS pre-window is 265 ms: a slope of 0.1 mV/s moves its mean by 0.027 mV, within the pre-window SD; weak)? [corrected 2026-10-23: the LS pre-window slopes are not an option -- excluded by D-034's status (D-017 note 4: a tilt calibrated from pre-window slopes would be too large by more than an order of magnitude).] The user's call. Simulation 2026-10-23 (`check_findings_2026-10-23.py`; theta_cf, sigma_b = 0.1 mV/s, t_SS = 80 ms, (T1) at the true law): sigma_b_hat / sigma_b has 95 % range 0.00-1.95 with 2 sweeps (clipped at 0 in 3.5-4.9 % of cells), 0.42-1.56 with 6 and 0.59-1.44 with 10 -- the spread of sqrt(chi^2_S / S) from the S realised tilts, plus the noise term in the lower tail -- and sigma_b_hat^2 is unbiased (mean ratio 0.97-1.01); pooled over the cohort's 64 SS sweeps under a common sigma_b, 0.82-1.17. A sigma_b error of that size (0 or 0.17 in the simulator for a cell at 0.1) moves u_hat by at most 0.18 of its SD (linearised at theta_cf, fixture layout): what is at stake is the twins' tilt, not theta_hat. Also open: the batched FFT autocorrelation and the Markov-jump simulator of 2.3 (h), both unneeded at the measured cost. [answered 2026-10-28 (later): D-100 (user, "Let's do A and B for the 2 sweep cells") -- for the 14 cells with two SS sweeps the twins' tilt SD is taken both per cell (D-034, variant a) and as one cohort value (variant b); partial pooling not taken. The assistant's reading, to confirm before Blocks 5-6 are specified: two labelled variants of twins per 2-sweep cell, sharing their manifest row and unit draws; the cohort value max(0, sum_c S_eff(c) sigma_b_hat_u(c)^2 / sum_c S_eff(c)) over the 18 cells, from the unclipped (T1) values; each real cell's own fit (2.3) keeps its own sigma_b_hat. Blocks 3-4 then also export sigma_b_hat_u(c)^2 and the per-sweep pair counts; a cohort step follows the per-cell jobs (Block 6).] [corrected 2026-10-28 (later, 2): D-100 was the assistant's misreading of the user's answer and is withdrawn. D-102 (user): each twin sweep carries the realised tilt of its own real sweep, b_hat(c, s), and no statistic of the tilt (no sigma_b(c)) is computed for the twins; on an SS sweep b_hat(c, s) is the least-squares slope of the window means on time with one intercept per polarity (proposal). Open: the Long Square sweeps' tilts and the fit's simulator (the same b_hat(c, s), proposal, or the data detrended).]
- (added 2026-10-21) D-073's per-sweep calibration: which of (sigma_f, rho_f, sigma_s, tau_s, sigma_b) are estimated per sweep and which per cell; raw or partially pooled per-sweep values; the twins' protocol twinning (LS amplitudes of the real cell instead of the cohort's `--ls-hyp-amps`; SS sweeps as 37-pulse trains at the real stored positions). Evidence for the proposal: `sweep_identifiability.py` (standalone, not in the repository): from one sweep at theta_cf the fast pair is identified to 1-3 % (sigma_f) and +/-0.012-0.014 (rho_f), sigma_s to 18-30 % (6-13 % with tau_s held), tau_s only within a factor of about 2 (log-SD 0.49-0.72).
