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
- `noise_sweep_fit.py`, `smoke_noise_sweep_fit.py` (Block 9, "Block 4b": the per-sweep
  stage; SPEC entry drafted 2026-10-30; the code written 2026-11-12)
- `run_noise_fit.py`, `submit_noise_fit.sh`, `smoke_noise_run.py` (Block 6: the noise-fit run;
  SPEC entry drafted 2026-11-12; the code written and smoke-tested 2026-11-13)
- `twin_config.py` (the twins' parameters; orchestrator's file), `twin_table.py`, `smoke_twin_table.py` (Block 5: the
  twins' table; SPEC entry drafted 2026-11-19)
- `twin_noise.py`, `smoke_twin_noise.py` (Block 10: the twin's noise; drafted 2026-11-19)
- `synthetic_ground_truth.simulate_ss_clean`, `synthetic_ground_truth.simulate_ls_clean`, `smoke_twin_response.py`
  (Block 11: the twin's clean responses; drafted 2026-11-19); the rest of `synthetic_ground_truth.py` is NOT covered
- (to be added block by block) [2026-11-12: the `sens_*.py` scripts dropped for now, D-125.3] the OU term of `synthetic_ground_truth.add_recording_noise`

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
them. D-072 (user): no shape test anywhere in the noise fit.] [2026-10-30: the
form was settled by D-076 (i) (2026-10-23: sigma_f and rho_f per sweep,
sigma_s per sweep with tau_s held and shrunk towards the cell, tau_s per cell)
and its estimation by D-116; the per-sweep stage is section 2.5 and Block 9.]

## 2. Mathematical formulation

### 2.1 The law (D-017.1, D-034)

[2026-11-17, D-152 to D-154: superseded in part by section 2.6 (the three-component law, 21 statistics, five fit coordinates, three per-sweep coordinates). Where this section and 2.6 differ, 2.6 holds; what 2.6 does not mention stands.]

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
sigma_b(c) is calibrated separately (D-034.1, section 2.4) [superseded
2026-10-29 by D-102, D-107: in the twins and in the fit's simulator b is not
drawn; every sweep carries its real sweep's tilt b_hat(c, s), read as in 2.4,
and no sigma_b(c) is computed], sigma_a is never needed (every statistic
demeans). For each fixed theta and every lag k >= 0
(samples) the autocovariance of the stationary part is

    gamma(k | theta) = sigma_f^2 rho_f^k + sigma_s^2 exp(-k Delta_t / tau_s)  (mV^2).  (L3)

### 2.2 The window statistics (D-028, D-029, D-051, D-095, D-103, D-104, D-106, D-110, D-111, D-115)

[2026-11-17, D-152 to D-154: superseded in part by section 2.6 (the three-component law, 21 statistics, five fit coordinates, three per-sweep coordinates). Where this section and 2.6 differ, 2.6 holds; what 2.6 does not mention stands.]

[2026-10-28 to 2026-10-29: the window start and the subtraction were replaced by D-095 with D-103, D-104 and D-106; this section is rewritten for them (2026-10-29, Block 3), below.]
[2026-10-30 (later): D-115 -- each kept window's tilt offset is removed before the median window, step (b2) below; (c), (d) and (g) take the corrected windows x'_k.]

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
[t_SS(c), 200) ms after each kept pulse, after the subtraction of its group's
median window (D-103.1; below): sigma_hat, rho_hat at 0.02, 0.1, 1, 10 ms; LS 265 ms
pre-window: sigma_hat, rho_hat at 0.02, 0.1, 1 ms; LS first 50 ms: sigma_hat;
LS last 132.5 ms: sigma_hat, rho_hat at 0.02, 0.1, 1, 10 ms. Lags in samples
are round(lag_ms * 1e-3 * f_s). (Blocks 3 onward; stated here so Block 1's
estimator is seen in its use.)

Window geometry, start and subtraction (Block 3; D-051, D-072, D-076 (ii),
D-095, D-103, D-104, D-106, D-110, D-111; rewritten 2026-10-29). A stored SS window holds
n_w = pre_n + post_n samples, t = 0 at the pulse onset, pre_n = round(10 ms
f_s), post_n = round(200 ms f_s), so the window ends at T_end = post_n / f_s =
200 ms; the onset period P = 10025 samples at 50 kHz is read from the data.
For each SS sweep s of a cell c and each polarity p in {dep, hyp}, the group
(s, p) holds the K_{s,p} stored windows x_k(t), k in (s, p), on t in [-10,
200) ms. For each cell, in this order:

 (a) Small groups (D-104.2, D-106.1). A group with K_{s,p} below
     min_windows_per_group (5, PROVISIONAL) is dropped -- flag
     `ss_group_dropped:<sweep>:<polarity>:few_windows` -- and its windows leave
     everything below: sigma_SS_hat(c), the start, the statistics per window,
     per sweep and per cell, the tilt reading (2.4) and the layout handed to
     Block 4 (whose simulator then cuts no window at those positions). The
     others are "the kept windows".
 (b) sigma_SS_hat(c) (D-095.2), mV: the median, over the kept windows, of
     the SD (ddof 1) of each window's pre-pulse segment (its first pre_n
     samples, t in [-10, 0) ms).
 (b2) The tilt offsets (D-115). For each SS sweep s with kept windows,
     b_hat_s is (TL1) of 2.4 on the kept windows of s cut at the floor,
     [t_floor, 200) ms with t_floor = t_ss_floor_ms = 80 ms, as stored (no
     subtraction; read before the start exists, D-115.2). Every kept window
     k of s is replaced, on all its n_w samples, by

         x'_k(t) = x_k(t) - b_hat_s (t^c_k - t_bar_s),                        (OF1)

     t^c_k and t_bar_s those of (TL1) (2.4): window k's centre time in the
     sweep and their mean over the sweep's kept windows; t^c_k - t_bar_s
     equals the difference of the windows' start times pos_k P Delta_t, so
     the offsets do not depend on the cut. A sweep whose tilt is
     NaN (`ss_tilt_none:<sweep>`) keeps its windows as stored and its flag;
     the fit then refuses the cell (D-114.2). sigma_SS_hat(c) of (b) is the
     same on x' (a window's SD does not see a constant). From here on x'_k
     stands for x_k: in (c), (d) and (g). Why: the per-sample median is
     equivariant under a shift common to a group's windows (the ramp b (t -
     t_k) inside each window passes through it) but not under the offsets
     b (t_k - t_bar), which differ between windows (dev checks
     `dev/2026-10-30_offsets_and_fit/`, the decision log's note under
     D-103).
 (c) The median window (D-103.1; on x'_k, (b2)): for each kept group and each sample t in
     [-10, 200) ms, m_{s,p}(t) = med_{k in (s,p)} x_k(t) (`numpy.median` over
     the group's windows; for an even K_{s,p} the mean of the two middle
     values), mV.
 (d) The start of a group (D-103.2, D-110). The model m_{s,p}(t) = B + A exp(-t
     / tau) (t, tau in ms; A, B in mV) is fitted by EXACT least squares to the
     samples t_j with t_j >= fit_start_ms (5 ms, PROVISIONAL) and t_j < T_end,
     for tau in decay_tau_bounds_ms = [1, 1000] ms (PROVISIONAL; D-110 as confirmed by D-114.1), by variable
     projection (D-110.1): at each tau the linear least-squares (A_hat(tau),
     B_hat(tau)) in closed form (the centred regression of y on e_j(tau) =
     exp(-t_j / tau)) and the profiled residual sum of squares RSS(tau); tau_hat
     minimises RSS(tau) -- the minimum of a grid of decay_grid_n = 301 points
     log-spaced over the bounds (PROVISIONAL), refined by
     `scipy.optimize.minimize_scalar(method='bounded')` on log tau between the
     grid minimum's two neighbours, then polished by `scipy.optimize.brentq` to
     the root of

         dRSS/dtau = -(2 A_hat(tau) / tau^2) sum_j r_j(tau) e_j(tau) t_j            (ST0)

     (r_j the residuals; the envelope theorem) where it changes sign within a
     relative 1e-6 of the refined point. No starting value enters. A grid
     minimum at the grid's first or last point -- no interior minimum of RSS in
     the bounds, i.e. no resolvable decay -- is a failed fit (D-110.2). Then,
     with kappa_tr = 0.1 (PROVISIONAL, D-051),

         t_{s,p} = tau_hat ln( |A_hat| / (kappa_tr sigma_SS_hat(c)) )   if |A_hat| > kappa_tr sigma_SS_hat(c),
         t_{s,p} = 0                                                   otherwise.       (ST1)

     A group whose fit failed is invalid -- flag
     `ss_group_dropped:<sweep>:<polarity>:fit` -- left out of the median below,
     its windows kept in the statistics (D-106.2). [2026-10-29 (later, 5):
     replaces the three-start `curve_fit` of `f293d4a` (D-110: "it must be
     fitted precisely").]
 (e) The start of the cell (D-103.3, D-104.1, D-111), with t_floor =
     t_ss_floor_ms = 80 ms (D-076 (ii)):

         t_star(c) = max{ t_floor, median over the valid groups of t_{s,p} },
         t_SS(c)   = t_star(c)                if T_end - t_star(c) >= min_window_ms (50 ms),
         t_SS(c)   = fallback_t_ms = 80 ms     otherwise, and when no group is valid.     (ST2)

     The median is `numpy.median` (for an even number of valid groups, the
     mean of the two middle values). Flags: `ss_start_floor` when a group is
     valid and the median is below t_floor; `ss_start_fallback` when the
     fallback applies, the result naming why ('short_window' or
     'no_valid_group'). t_SS(c) is computed on the real windows only; the
     fit's simulator cuts its windows at the real cell's t_SS(c) (D-104).
     [2026-10-29 (later, 5): the median replaces the maximum of D-103.3, D-111:
     on noisy cells each group's t_{s,p} scatters by about 18 ms, and the
     maximum sent 47-96 % of simulated cells to the fallback (dev check
     `dev/2026-10-29_start_noise/`).]
 (f) The enlarged window of a kept pulse: its samples with t >= t_SS(c), j0 =
     pre_n + round(t_SS f_s / 1e3), m = n_w - j0 samples, ending 0.5 ms
     before the next pulse's onset.
 (g) The subtraction (D-103.1, D-106.2; on x'_k, (b2)): x_tilde_k(t) = x_k(t) - m_{s,p}(t)
     on [t_SS(c), 200) ms for every kept window k in (s, p), computed as the
     per-sample median of the group's enlarged windows, which is m_{s,p}
     restricted to that range (the median is taken sample by sample). Each
     window's own scalar mean is removed by (S1)-(S2) themselves. The fit's
     simulator applies the same function to its own windows, group by group
     (D-051 (iv), D-033).

Per sweep s the statistics are kept as medians over that sweep's kept
windows (SS: 5 per sweep; LS: the 10 values of that one window), the inputs
of the per-sweep stage of D-076 (i). [superseded 2026-10-29 by D-095, D-103,
D-104, D-106: the transient test of D-051 (iv) -- the 5 ms bins of the odd part
r_odd of the polarity means against kappa_tr sigma_SS, its noise-limited
behaviour found by the smoke and by simulation, the option
`transient_floor_z` -- with D-051 (i)'s fallback at 190 ms (flag
`ss_enlarged_none`), and the per-polarity mean window over the cell
(D-051.1). Their text and findings: this file at commit 1aabe3e, 2.2. The
functions `noise_windows.transient_start` and `subtract_mean_response` are
kept, unused by the pipeline, so that the dev checks of 2026-10-23 to
2026-10-29 re-run.] The LS pre-window of EVERY Long Square sweep enters,
hyperpolarising and depolarising (D-073 needs a twin of each; the assistant's
choice, D-076 open points), one window each: [t_0, t_on - 5 ms) with t_on the
first sample where |i| exceeds half its maximum (the fitter's rule,
`_estimate_noise`).

### 2.3 The fit (D-033, D-048 to D-050, D-056)

[2026-11-17, D-152 to D-154: superseded in part by section 2.6 (the three-component law, 21 statistics, five fit coordinates, three per-sweep coordinates). Where this section and 2.6 differ, 2.6 holds; what 2.6 does not mention stands.]

[2026-10-29: D-107 -- the tilt SD and its alternation leave the fit: every simulated sweep, SS and LS, carries the tilt of its real sweep, b_hat(c, s), the same in every replicate. Rewritten 2026-10-29 (later, 5) with Block 4: items (a), (b), (f), (g) below; the layout is the kept windows' (D-106) and the simulator subtracts the group median window (D-103.1).]

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

Block 4's realisation of (F1) and the iteration (2026-10-23, rewritten
2026-10-29 (later, 5); D-048 to D-050, D-056, D-103, D-106, D-107.2 as
stated, the rest the assistant's choices, flagged). The cell's layout is
that of its KEPT windows (2.2 (a), D-106.1): the simulator cuts no window at
a dropped group's positions. Every simulated sweep carries its real sweep's
tilt, b_hat(c, s) of 2.4, as a fixed number (D-107.2; `noise_law`'s
`tilt_mV_per_s`), in every replicate and at every theta; a cell with a tilt
that could not be read (an LS end window too short, (TL2)) is refused with
the reason (D-114.2: the assistant's proposal of 2026-10-29 (later, 4),
confirmed by the user 2026-10-30). (a) Binding function: the SS part is computed on M_cells = ceil(M / K)
replicate cells simulated at theta with the cell's own layout (n_sweeps
continuous sweeps of (train_length - 1) P + n_w samples each, D-056 (i), sweep
s carrying b_hat(c, s); the enlarged windows cut at the kept positions; the
group median window of the replicate's own windows subtracted per sweep and
polarity, D-103.1, after the replicate has read its own b_hat*_s by (TL1) on its windows
cut at the floor and removed its own offsets (OF1), D-115.3) and pooled to M_cells K windows; the LS part on
N_LS = ceil(M / K_LS) K_LS windows of n_LS samples, window i carrying the
tilt of the cell's LS sweep i mod K_LS (ceil(M / K_LS) replicate sets of the
cell's LS sweeps; an LS pre-window is simulated on its own: its statistics
demean, so only the tilt's slope matters, not where in the sweep the window
sits); each part's Harrell-Davis median per statistic
(`scipy.stats.mstats.hdquantiles`, prob 0.5) is q_sim(theta). Common random
numbers by re-seeding: every evaluation rebuilds `default_rng(seed_binding)`
and draws in the fixed order of `noise_law.draw_units` (the unit tilt draw is
made and left unused, so the order is that of 2026-10-23), so q_sim(theta) is
a deterministic function of theta for a given cell and seed (nothing is
stored: 3000 windows of 13250 samples would hold 1.3 GB of unit draws). (b)
Sigma_hat: R replicate cells at the real layout with the real tilts, each
reduced to its statistic vector by `noise_cell_stats.statistic_vector`;
sample covariance (ddof 1); W = its inverse through the Cholesky factor C
(Sigma_hat = C C^T, residual whitened by `solve_triangular`). (c) Statistics
with no finite data value (K_m = 0 or a non-finite SE) are dropped from q_hat,
q_sim and Sigma_hat (flag `stats_dropped` with their labels); d_eff <= 15.
(d) Fit coordinates u = (log sigma_f, rho_f, log sigma_s, log tau_s) with
bounds log 1e-4 <= log sigma <= log 10 (mV), 1e-3 <= rho_f <= 0.999, log 1e-4
<= log tau_s <= log 10 (s); `least_squares(method='trf', diff_step=(1e-3,
1e-2, 1e-3, 1e-3), ftol=1e-4, xtol=1e-3, gtol=1e-8, max_nfev=15)` -- the
finite-difference step is set explicitly because q_sim is a Monte Carlo
quantity (the default step, sqrt(eps), would read roundoff), the tolerances
match the stop rule (0.01 in u) rather than the default 1e-8, and `max_nfev`
counts trf iterations (scipy does not count the Jacobian's evaluations),
each costing 5 evaluations of q_sim with the 2-point Jacobian: at most about
75 per start. (e) Starts: the closed-form start is the argmin of (F1) with
W_0 on the closed-form statistics q_cf (eq. 10; instant), then that start
with tau_s / 2 and 2 tau_s -- three starts for the first solve (W_0); the
later solves (W from Sigma_hat_k) start once, from theta_hat_k (assistant's
choice, `starts_after_first = 1`; W changes little between iterations and a
solve costs about an hour at M = 3000). (f) [removed 2026-10-29 (later, 5),
D-107.2: the tilt is not estimated inside the fit -- no (T1), no
alternation, no sigma_b_hat; the simulator carries the real sweeps' tilts.]
(g) Stop: the first Sigma_hat-based solve k >= 1 with max_i |u_{k+1,i} -
u_{k,i}| <= 0.01, or k_max = 6 solves (flag `no_plateau`); then the final
step -- Sigma_hat at R = 3000 and theta_hat, one more solve; q_sim and J are
evaluated at the final theta_hat with the final Sigma_hat; kappa of the
correlation matrix of Sigma_hat (`numpy.linalg.cond`); and, as a goodness
check only (no decision uses it yet; the user's chi-square alarm is open), J
~ chi^2 with d_eff - 4 degrees of freedom, its tail probability reported.
(h) Cost (sandbox timings 2026-10-23, one core, the fixture's 2-sweep layout,
K = 63, 5 LS windows; the median subtraction and the fixed tilts change them
little): one evaluation of q_sim at M = 3000 is 16 s (SS 3.9 s on 48
replicate cells, LS 12.3 s on 3000 windows of 13250 samples); one replicate
cell for Sigma_hat is 0.10 s (R = 600: 1 min; R = 3000: 5 min); a 10-sweep
cell (K about 330) costs about 0.7 s per replicate (R = 600: 7 min; R = 3000:
35 min) and 19 s per evaluation. A solve is at most about 75 evaluations per
start (20 min), typically 30 (the smoke's solves take 6 trf iterations). A
2-sweep cell takes 1-3.5 h on one core, a 10-sweep cell 2-5.5 h; the campaign
runs the 18 cells as independent jobs (Block 6). Options not taken, for the
user: a batched FFT autocorrelation in `noise_stats` (most of the LS 12 s; it
would be a second implementation of (S2), D-029 (iv)); exact Markov jumps
over the unused samples of a sweep (halves the SS simulation; more
bookkeeping). [superseded 2026-10-29 by D-102 / D-107: the finding of smoke
F7 of 2026-10-23 -- sigma_b_hat of a 2-sweep cell rests on two tilt draws --
no longer concerns the fit, which estimates no tilt SD.]

### 2.4 The tilt of each sweep (D-102, D-107; rewritten 2026-10-29)

The tilt b of law (L1) is read on every real sweep and copied, not drawn:
the twin of sweep s (Block 5, D-102.1) and every simulated copy of that sweep
inside the fit (Block 4, D-107.2) carry b_hat(c, s) as a fixed number, the
same in every replicate. No statistic of the tilt is computed: no sigma_b(c),
no (T1).

SS sweep (D-102.2). For each SS sweep s of c that has kept windows (2.2 (a)),

    (b_hat(c, s), alpha_hat_dep, alpha_hat_hyp)
        = argmin over (b, alpha_dep, alpha_hyp) of
          sum_{k in s} ( <x_tilde_k> - alpha_{p(k)} - b (t^c_k - t_bar_s) )^2,        (TL1)

with <x_tilde_k> (mV) the mean of window k's samples on [t_floor, 200) ms as
stored, t_floor = 80 ms (D-115.2: read before the offsets and the start;
[superseded 2026-10-30 (later): the mean of the subtracted samples on
[t_SS(c), 200) ms, which after (OF1) would see no tilt]), t^c_k = (pos_k P +
j_f + (m_f - 1) / 2) Delta_t (s) its centre time in the sweep, j_f = pre_n +
round(t_floor f_s / 1e3), m_f = n_w - j_f (origin: the first sample of the stored window at
position 0; the slope does not depend on the origin), t_bar_s the mean of
the sweep's t^c_k, and p(k) the window's polarity; one intercept per polarity
present in the sweep. Solved by `numpy.linalg.lstsq`. NaN, flag
`ss_tilt_none:<sweep>`, when the design's rank is below its number of
columns or the sweep has no more windows than columns. Reported with the OLS
standard error se = sqrt( s^2 [(X^T X)^-1]_00 ), s^2 = RSS / (n - columns),
which treats the window means as independent (adjacent ones correlate weakly
through the slow component: an approximation, for the report only). (TL1)
is invariant to the group's median subtraction: it shifts every window of a
group by one constant, which the polarity's intercept absorbs. Precision at
theta_cf (dev check `dev/2026-10-29_ls_tilt/check_ss_tilt_reading.py`, on the
tilt study's 2-, 6- and 10-sweep reference layouts): RMS error 0.0027-0.0036
mV/s at t_SS = 80 ms, 0.0031-0.0041 at 130 ms, 0.0034-0.0044 at 150 ms; the
reading of D-115.2 is the 80 ms one. The same regression serves the offsets
(OF1) and every replicate of the fit's simulator (2.3 (a), D-115.3), through
one function (`noise_windows.tl1_fit`).

LS sweep (D-107.1). For each LS sweep s (every sweep of 2.2's LS rule),

    b_hat(c, s) = ( ybar_end(c, s) - ybar_pre(c, s) ) / Delta_t_c(c, s)   (mV/s),        (TL2)

with ybar_pre (mV) the mean of the finite samples of the pre-window [t_0,
t_on - 5 ms), ybar_end (mV) the mean of the end window -- the last
ls_end_ms = 500 ms of the stored sweep, ending at its last finite sample
(length PROVISIONAL, D-107 open (b); `ls_baseline_qc.END_WINDOW_MS`, the
Allen convention) -- and Delta_t_c (s) the distance between the two windows'
centres (the means of their sample times). NaN when the end window does not
lie entirely after the step -- its first sample at or before t_off, the
first sample after the last one with |i| above half its maximum -- or the
sweep is shorter than the window (flag `ls_tilt_none:<sweep>:tail_short`), or
when a mean is not finite (`ls_tilt_none:<sweep>:nonfinite`). (TL2) equals
`ls_baseline_qc.baseline_record`'s `delta_end_mV` divided by the distance
between its two windows' centres (the smoke's oracle). On the archive's
L3_exc geometry (pre-window [0, 265) ms, a 1 s step from 270 ms, 6.000 s
stored after it) Delta_t_c = 6.8875 s; under (L1) b_hat = b + (e_end -
e_pre) / Delta_t_c, unbiased if the step leaves no after-effect 5.5-6 s
later (a hypothesis, D-107.1); error SD 0.0045 mV/s at theta_cf (dev check
`dev/2026-10-29_ls_tilt/check_ls_tilt_reading.py`).

[superseded 2026-10-29 by D-102 and D-107: the calibration of sigma_b(c) from
the window-mean semivariogram, (T1) sigma_b_hat(c)^2 = 2 [V_hat_c(h*) - V(h* |
theta_hat_c, 0)] / h*^2 at h* = 4 P Delta_t, alternated with theta_hat
(D-034.1, D-051 (iii)); its text: this file at commit 1aabe3e, 2.4. The
closed form `noise_law.semivariogram` (Block 2) and the data side
`noise_cell_stats.window_mean_semivariogram` (Block 3) are kept, unused by
the pipeline, for the dev checks of the tilt study; Block 4's
`tilt_from_semivariogram` and the alternation leave with Block 4's rewrite.]

### 2.5 The per-sweep stage (D-073, D-076 (i), D-115, D-116; Block 9, "Block 4b"; drafted 2026-10-30)

[2026-11-17, D-152 to D-154: superseded in part by section 2.6 (the three-component law, 21 statistics, five fit coordinates, three per-sweep coordinates). Where this section and 2.6 differ, 2.6 holds; what 2.6 does not mention stands.]

For each real cell c, once the per-cell fit of 2.3 has returned theta_hat(c)
= (sigma_f_hat, rho_f_hat, sigma_s_hat, tau_s_hat)(c), and for each sweep s
in S(c) = S_SS(c) u S_LS(c) -- S_SS(c) the SS sweeps with kept windows (2.2
(a), D-106.1), S_LS(c) the LS sweeps of 2.2's LS rule -- the stage gives the
law that the twin of s carries (D-076 (i); Block 5):

    theta_tilde(c, s) = (sigma_f_hat(c, s), rho_f_hat(c, s), sigma_s_tilde(c, s), tau_s_hat(c)).    (P0)

Below, pr in {SS, LS} is a protocol (not the polarity p of 2.2) and
tau2_hat_pr(c) a between-sweep variance of log sigma_s (not the slow time
constant tau_s).

(a) Target. q_hat(c, s) of the Notation: for s in S_SS(c) the 5 SS
statistics of 2.2, each the median over the sweep's kept windows after the
offsets (OF1) and the subtraction (D-115, D-103.1) -- Block 3's
`per_sweep_vectors`; for s in S_LS(c) the 10 LS statistics of its one
pre-window. A statistic without a finite value on the sweep is dropped from
q_hat(c, s), q_sim(theta | s) and Sigma_hat_s (as 2.3 (c)); d_s statistics are
kept, and d_s >= 4 is required (one degree of freedom over the three
coordinates; else the sweep fails with the reason `stats`; assistant's
choice).

(b) Binding function. For each fixed theta and each sweep s, q_sim(theta |
s) is the Harrell-Davis median per statistic over the pool of the sweep's own
simulator: for s in S_SS(c), ceil(M / K_s) replicate sweeps (K_s the sweep's
kept windows), each one continuous sweep of L_s samples carrying b_hat(c, s),
cut at the sweep's kept positions from the floor, its own (TL1) and offsets
(OF1) removed and its own group median windows subtracted (D-115.3, D-103.1),
pooled to ceil(M / K_s) K_s windows; for s in S_LS(c), M pre-windows of n_LS
samples, each carrying b_hat(c, s). This is 2.3 (a)'s simulator on the cell's
layout restricted to the sweep: the sweeps are independent under (L1)-(L2) and
every step of the processing is taken per sweep, so a sweep simulated alone
has the law of that sweep simulated in its cell [reasoning]. Common random
numbers per sweep: the pool is rebuilt from seed_binding(c, s) at every
evaluation, so q_sim(theta | s) is a deterministic function of theta.

(c) The fit (D-116.1). In the coordinates u_s = (log sigma_f, rho_f, log
sigma_s) -- the first three of 2.3 (d), with their bounds -- with tau_s held at
tau_s_hat(c) and theta(u_s) = (exp u_s1, u_s2, exp u_s3, tau_s_hat(c)),

    J_s(u_s | c, s) = (q_sim(theta(u_s) | s) - q_hat(c, s))^T W_s (q_sim(theta(u_s) | s) - q_hat(c, s)),    (P1)

W_s = Sigma_hat_s^{-1} fixed: Sigma_hat_s is the sample covariance (ddof 1) of
sweep s's statistic vector over the R_final replicate cells of Block 4's final
Sigma_hat run (`CellFitResult.per_sweep_cov[s]`), restricted to the kept
statistics and whitened through its Cholesky factor as in 2.3 (b). That run is
simulated at the law the final solve starts from, theta_hat_K, which the final
solve moves to theta_hat(c) by about tol_u when converged; this is the reading
of D-116.1's "Sigma_hat_s(theta_hat(c))" (assistant's, flagged). One call of
`least_squares(method='trf')` with Block 4's diff_step for these coordinates
(1e-3, 1e-2, 1e-3), its ftol, xtol, gtol and max_nfev, started at
u_s(theta_hat(c)); no second start, no iteration of the weight.
theta_hat(c, s) = theta(u_hat_s); J_s is reported with dof_s = d_s - 3 (a
report only). The sweep's fit fails (D-116.4) when `least_squares` returns
status <= 0 (max_nfev reached, or improper input; reason `no_convergence`) or
a coordinate of u_hat_s lies within 1e-3 of its range from a bound (Block 4's
`at_bound` rule; reason `at_bound:<coordinates>`).

(d) The sampling variance (D-116.2). For b = 1..B_v, one replicate of sweep
s is simulated at theta_hat(c, s) (tau_s = tau_s_hat(c)) with the sweep's
layout and tilt, on the stream seed_v(c, s) (the replicates drawn in the order
b = 1..B_v), reduced to its statistic vector q*_b(c, s) as in (a) and refitted
by (c) -- the same q_sim(. | s) (seed_binding(c, s)), W_s and start
u_s(theta_hat(c)) [corrected 2026-11-11, D-122.1: q_sim(. | s) is (b)'s binding
function at M = M_refit = 1000 on the same stream seed_binding(c, s); the main
solve (c) keeps Block 4's M_binding = 3000] -- giving y*_b = log sigma_s_hat*_b (the third coordinate).
Then

    v(c, s) = (1 / (B_ok - 1)) sum_{b ok} (y*_b - ybar*)^2,   ybar* = (1 / B_ok) sum_{b ok} y*_b,      (P2)

over the B_ok refits that did not fail by (c)'s rule; v(c, s) is NaN when
B_ok < 2, and the sweep then fails with the reason `v_none`. A sweep whose fit
failed is not bootstrapped. [open, section 8: B_v (proposal 100 [corrected
2026-11-02: proposal 50 -- the shrunk value's RMS error at B_v = 50 lies within
0.16 percentage points of its value with v known, `dev/2026-11-02_bootstrap_explained/`]); the refits'
binding size (proposal M = 1000, the alternative the main solve's M); a failed
refit (proposal: left out of (P2) and counted, the sweep flagged
`v_refits_failed:<n>/<B_v>` when more than 10 % of its refits fail).]
[answered 2026-11-11, D-122.1-D-122.2 (user): B_v = 50; the refits' binding size
M_refit = 1000; a failed refit is left out of (P2) and counted, the sweep
flagged `v_refits_failed:<n>/<B_v>` when more than 10 % of its B_v refits fail
(a report: the sweep stays in the pooling); B_ok < 2 gives v NaN and the
failed sweep `v_none`, as above.]

(e) The pooling per protocol (D-116.3; (D-076.1)). For pr in {SS, LS} let
S'_pr(c) be the sweeps of protocol pr whose fit did not fail, k_pr =
|S'_pr(c)|, and for s in S'_pr(c) y_s = log sigma_s_hat(c, s), v_s = v(c, s),
w_s = 1 / v_s. For k_pr >= 2, with every sum over s in S'_pr(c),

    ybar_FE,pr(c)  = sum_s w_s y_s / sum_s w_s,
    Q_pr(c)        = sum_s w_s (y_s - ybar_FE,pr(c))^2,
    tau2_raw_pr(c) = (Q_pr(c) - (k_pr - 1)) / (sum_s w_s - sum_s w_s^2 / sum_s w_s),                 (P3)
    tau2_hat_pr(c) = max(0, tau2_raw_pr(c)),
    mu_hat_pr(c)   = sum_s w*_s y_s / sum_s w*_s,   w*_s = 1 / (v_s + tau2_hat_pr(c)),                (P4)
    lambda(c, s)   = tau2_hat_pr(c) / (tau2_hat_pr(c) + v_s),
    log sigma_s_tilde(c, s) = mu_hat_pr(c) + lambda(c, s) (y_s - mu_hat_pr(c)),                        (D-076.1)

the DerSimonian-Laird moment estimate clipped at zero, with the random-effects
mean recomputed at the clipped value (statsmodels' `combine_effects(...,
method_re="dl")` returns tau2_raw_pr unclipped and uses it in its random-effects
weights: D-076's clarification of 2026-10-26). With tau2_hat_pr(c) = 0 every
sweep of the protocol takes mu_hat_pr(c), the fixed-effect mean. k_pr = 1:
mu_hat_pr(c) = y_s, tau2_hat_pr(c) and lambda NaN, sigma_s_tilde(c, s) =
sigma_s_hat(c, s) (unshrunk; D-116, reasoning). k_pr = 0: nothing is pooled
(mu_hat_pr, tau2_hat_pr NaN). sigma_f and rho_f are not pooled (D-076 (i)).

(f) A failed sweep (D-116.4). Flagged `sweep_fit_failed:<pr>:<sweep>:<reason>`
and left out of S'_pr(c); its twin takes the cell's values, theta_tilde(c, s) =
theta_hat(c) [the reading of "the cell's values" is open, section 8; the
alternative keeps (sigma_f_hat, rho_f_hat, tau_s_hat)(c) and takes sigma_s =
exp mu_hat_pr(c) when k_pr >= 1]. [answered 2026-11-11, D-122.3 (user): the
cell's whole law, theta_tilde(c, s) = theta_hat(c); the alternative is not
taken. A sweep whose bootstrap ends in `v_none` (d) is a failed sweep too.]

(g) Outputs. Per sweep: the protocol, the sweep number, K_s (SS) or n_LS
(LS), b_hat(c, s), d_s, theta_hat(c, s), J_s, dof_s, the residual calls, the
solver's status, at_bound, failed and its reason, v(c, s), B_ok and the
failed refits, lambda(c, s), sigma_s_tilde(c, s), theta_tilde(c, s), the
flags; per cell and protocol: k_pr, Q_pr, tau2_raw_pr, tau2_hat_pr,
mu_hat_pr. Table columns (D-076, D-116): per sweep `noise_sweep_sigma_f_mV`,
`noise_sweep_rho_f`, `noise_sweep_sigma_s_raw_mV`, `noise_sweep_sigma_s_mV`
(the shrunk value), `noise_sweep_v_log_sigma_s`, `noise_sweep_w` (lambda),
`noise_sweep_flags`; per cell `noise_cell_tau2_log_sigma_s_ss`,
`noise_cell_tau2_log_sigma_s_ls` and, the assistant's addition,
`noise_cell_mu_log_sigma_s_ss`, `noise_cell_mu_log_sigma_s_ls`.

(h) Cost and precision (sandbox 2026-10-30, one core, Block 4's smoke
fixture: SS sweeps of 30 and 33 kept windows, LS pre-windows of 13250
samples; dev checks `dev/2026-10-30_sweep_stage_cost/`). One evaluation of
q_sim(theta | s) at M = 3000 takes 3.5 s for an SS sweep (100 replicate
sweeps) and 8.6 s for an LS sweep (3000 windows; the cell's own SS and LS
parts 3.8 s and 10.9 s); one solve of (c) took 16 calls of the residual for an
SS sweep (4 trf iterations) and 28 for an LS sweep (7), from a start 0.22 away
in log sigma_s -- about 0.9 min and 4.0 min at M = 3000; a replicate sweep
costs 0.04 s (SS) and 0.005 s (LS). The bootstrap is B_v solves per sweep: at
B_v = 100, about 1.6 h (SS) and 6.6 h (LS) per sweep with refits at M = 3000,
a third of that at M = 1000; the fixture cell (2 SS, 5 LS sweeps) about 36 h
or 13 h on one core. The (cell, sweep) pairs are independent jobs (Block 6).
For a normal log sigma_s_hat, SD(v) / v = sqrt(2 / (B_ok - 1)), 0.14 at 100
[textbook: (B_ok - 1) v / sigma^2 ~ chi^2_{B_ok - 1}]. The linearised
variance V = A Sigma*_s A^T, A = (G^T W_s G)^{-1} G^T W_s, G the Jacobian of
q_sim(. | s) in u_s (central differences, step 0.02), Sigma*_s the replicate
sweep's covariance at theta_cf, gives sqrt(v) = 0.067-0.069 (SS sweep, K_s =
30) and 0.150-0.171 (LS sweep); over three binding seeds its range (max -
min) is 2-7 % of the mean (SS) and 14-25 % (LS), and the mean of three seeds
moves between M = 3000 and M = 1000 by -0.5 % (LS) and +2.3 % (SS)
(`check_refit_binding_size.py`).

### 2.6 The three-component law and what it changes (D-152, D-153, D-154; written 2026-11-17)

Why: the nf1 run (2026-11-17) rejected the law of 2.1 in all 18 L3_exc cells, its SS autocorrelation at 1 ms above
the data's in 16; a middle component removes that excess (`dev/2026-11-17_three_component_law/`). This section is the
contract of the change. It replaces the clauses of 2.1, 2.2, 2.3 and 2.5 it names; everything else in them stands
(the windows, the start, the subtraction, the tilts, the iteration, the bootstrap, the pooling formulas).

(a) The law (replaces (L1)-(L3) and theta of 2.1; D-152). For one sweep sampled every Delta_t (s), t_j = j Delta_t,
j = 0..n-1, mid-time t_bar, the injected noise (mV) is

    e_j = e_fast_j + e_mid_j + e_slow_j + a + b (t_j - t_bar),                                    (L1')

    e_fast_j ~ N(0, sigma_f^2), independent across j (white),
    e_mid_j  = rho_m e_mid_{j-1}  + eta_j,   eta_j  ~ N(0, sigma_m^2 (1 - rho_m^2)),  rho_m = exp(-Delta_t / tau_m),
    e_slow_j = rho_s e_slow_{j-1} + eta'_j,  eta'_j ~ N(0, sigma_s^2 (1 - rho_s^2)),  rho_s = exp(-Delta_t / tau_s),  (L2')

the two recursions started in their stationary laws (e_mid_0 ~ N(0, sigma_m^2), e_slow_0 ~ N(0, sigma_s^2)); the three
components independent, each driven by its own iid standard-normal draws; a and b as in 2.1 and 2.4 (b fixed per
sweep, never drawn). theta = (sigma_f, sigma_m, tau_m, sigma_s, tau_s), in mV, mV, s, mV, s: five parameters, in
this order everywhere (arrays, dataclass fields, fit coordinates). rho_f no longer exists. For each fixed theta and
Delta_t and every lag k (samples, any sign):

    gamma(k | theta) = sigma_f^2 [k = 0] + sigma_m^2 exp(-|k| Delta_t / tau_m) + sigma_s^2 exp(-|k| Delta_t / tau_s)  (mV^2),  (L3')

[k = 0] being 1 at lag 0 and 0 elsewhere. Reading: the white part adds variance at lag 0 only; the middle and slow
parts are exponentials of the lag. Every closed-form moment of Block 2 (Var(x_bar), E[S_0], E[S_L], the closed-form
SD and autocorrelation, the legacy semivariogram functions) keeps its formula in terms of gamma and takes (L3').
The fit's domain (D-152): sigma_f, sigma_m, sigma_s in [1e-4, 10] mV; tau_m in [4e-5, 1e-3] s (fixed in seconds; 2 samples at the archive's 50 kHz
to 1 ms); tau_s in [1e-3, 10] s. `NoiseLaw` itself accepts any sigma >= 0 and any tau > 0 (a zero SD switches its
component off, for oracles), and does not enforce the fit's bounds.

(b) The simulator (replaces 2.1's recursions in code; Block 2). Unit draws per trace: fast, mid, slow (n_traces, n),
tilt, offset (n_traces,), drawn from the generator in THIS order (fast, mid, slow, tilt, offset), so a generator
state gives the same draws for any law. e_fast = sigma_f xi_fast; e_mid and e_slow by the stationary AR(1) recursion
(`scipy.signal.lfilter`, as today's slow part). The draw order changes every simulated number against the
two-component code: results of the nf1 run are not reproduced bit for bit, by design.

(c) The statistics (replaces 2.2's lag lists and d; D-153.2, D-154.1). Autocorrelation lags, ms: SS enlarged window
(0.02, 0.04, 0.1, 0.3, 1, 10); LS pre-window (0.02, 0.04, 0.1, 0.3, 1); LS last 132.5 ms (0.02, 0.04, 0.1, 0.3, 1,
10); in samples round(lag_ms * 1e-3 * f_s): 1, 2, 5, 15, 50, 500 at 50 kHz. The statistic vector has d = 21 entries
in this order: SS: sigma_hat, rho_hat at the 6 SS lags (7); LS pre-window: sigma_hat, rho_hat at its 5 lags (6); LS
first 50 ms: sigma_hat (1); LS last 132.5 ms: sigma_hat, rho_hat at its 6 lags (7). N_SS = 7 statistics per SS
window, N_LS = 14 per LS window. Labels: `ss_sd`, `ss_rho_<lag>ms`, `ls_sd`, `ls_rho_<lag>ms`, `ls_first_sd`,
`ls_last_sd`, `ls_last_rho_<lag>ms`, <lag> printed with `%g` (0.02, 0.04, 0.1, 0.3, 1, 10). The estimators (S1),
(S2), their NaN rules and the medians are unchanged. No code may assume 15, 5 or 10: the counts come from
`noise_cell_stats` (`STAT_LABELS`, `N_SS_STATS`, `N_LS_STATS`).

(d) The per-cell fit (replaces 2.3 (d), (e) in part, and the dof; Block 4). Fit coordinates

    u = (log sigma_f, log sigma_m, log tau_m, log sigma_s, log tau_s)        (natural logs of mV, mV, s, mV, s)

with bounds log 1e-4 <= log sigma <= log 10 for the three SDs, log 4e-5 <= log tau_m <= log 1e-3, log 1e-3 <= log
tau_s <= log 10. `least_squares(method='trf')` as in 2.3 (d) with diff_step = (1e-3, 3e-3, 3e-3, 1e-3, 1e-3)
(PROVISIONAL, [orchestrator]: the middle part's coordinates get a step three times larger because the statistics
respond more weakly to them); ftol, xtol, gtol, max_nfev unchanged (a trf iteration now costs 6 evaluations of
q_sim). The stop rule is max_i |Delta u_i| <= 0.01 over the five coordinates. Coordinate names, for `at_bound`:
`log_sigma_f`, `log_sigma_m`, `log_tau_m`, `log_sigma_s`, `log_tau_s`; the rule (within 1e-3 of the range from a
bound) is unchanged; a cell at the tau_m cap is flagged `at_bound:log_tau_m` and kept. Starts: the closed-form start
is the lowest-cost solution of the closed-form problem (2.3 (e): W_0 on the closed-form statistics q_cf, instant)
over the 4 initial points sigma_f = 0.035 mV, sigma_m = 0.017 mV, sigma_s = 0.06 mV with (tau_m, tau_s) in
{1e-4, 6e-4} s x {0.03, 0.2} s (PROVISIONAL, [orchestrator]: the values of the dev check); then, as D-056, that
start with tau_s / 2 and 2 tau_s (each clipped into the bounds) -- three starts for the first solve, one afterwards.
dof = d_eff - 5 (16 when no statistic is dropped); the fit needs d_eff >= 6. Everything else of 2.3 -- W_0, the
binding function and its common random numbers, Sigma_hat, the iteration, the final step, J, kappa, the flags --
is unchanged, with the law of (a) and the vector of (c).

(e) The per-sweep stage (replaces 2.5 (P0), (c), (d), (e), (f) in part; Block 9; D-153.1, D-154.2, D-154.3). For
each sweep s of cell c, with tau_m and tau_s held at the cell's tau_m_hat(c), tau_s_hat(c):

    u_s = (log sigma_f, log sigma_m, log sigma_s),   theta(u_s) = (exp u_s1, exp u_s2, tau_m_hat(c), exp u_s3, tau_s_hat(c)),
    theta_tilde(c, s) = (sigma_f_hat(c, s), sigma_m_tilde(c, s), tau_m_hat(c), sigma_s_tilde(c, s), tau_s_hat(c)).   (P0')

The fit (P1) is unchanged in form, with diff_step (1e-3, 3e-3, 1e-3) and the three SD bounds; dof_s = d_s - 3
(SS: 7 - 3 = 4; LS: 14 - 3 = 11 when no statistic is dropped); d_s >= 4 still required. Failure (D-116.4, D-154.3):
status <= 0 (`no_convergence`), a coordinate at a bound (`at_bound:<coordinates>`), or, checked next,
sigma_f_hat(c, s) < min_sigma_f_ratio * sigma_f_hat(c) with min_sigma_f_ratio = 0.5 (reason `degenerate:sigma_f`).
The bootstrap (P2) is run once per sweep as today; each successful refit gives both y_m*_b = log sigma_m_hat*_b
and y_s*_b = log sigma_s_hat*_b, and

    v_m(c, s), v_s(c, s) = the sample variances (ddof 1) of y_m*_b and of y_s*_b over the B_ok refits;

either NaN (B_ok < 2) or below V_MIN = 1e-12 (log mV)^2 (every successful refit returned the same value, up to roundoff: an SD of log sigma under 1e-6, where real bootstrap SDs are 0.03 to 0.8; the pooling's weights 1 / v would be absurd or not exist; the floor is a module constant, amendment 5) fails the sweep with `v_none` [the second case added 2026-11-17, D-155 [orchestrator], after the final check's finding F1: `pool_cell` raised on v = 0 and the whole cell was lost]. The pooling (P3), (P4), (D-076.1) is applied per protocol
TWICE, independently: to (y_m, v_m) = (log sigma_m_hat(c, s), v_m(c, s)) giving tau2_hat_m,pr(c), mu_hat_m,pr(c),
lambda_m(c, s), sigma_m_tilde(c, s); and to (y_s, v_s) as today. sigma_f is not pooled. k_pr = 1 and k_pr = 0 as
in 2.5 (e), for both. A failed sweep's twin carries theta_hat(c) whole (D-122.3).

(f) Cost. One trf iteration of the per-cell solve costs 6 evaluations (was 5), of the per-sweep solve 4 (unchanged);
each window statistic set has 2-3 more autocorrelations. Expected run cost 20-40 % above nf1's [reasoning]; the
pilot measures it.

(g) Outputs (replaces the column lists of 2.5 (g) and of Blocks 4 and 6). Per cell (`noise_cells_<GROUP>.csv`):
`specimen_id`, `status`, `reason`, `noise_sigma_f_mV`, `noise_sigma_m_mV`, `noise_tau_m_ms`, `noise_sigma_s_mV`,
`noise_tau_s_ms`, `noise_J`, `noise_dof`, `noise_p`, `noise_kappa`, `noise_n_solves`, `noise_flags`,
`noise_t_ss_ms`, `noise_K`, `noise_K_LS`, `noise_cell_tau2_log_sigma_m_ss`, `noise_cell_mu_log_sigma_m_ss`,
`noise_cell_tau2_log_sigma_m_ls`, `noise_cell_mu_log_sigma_m_ls`, `noise_cell_tau2_log_sigma_s_ss`,
`noise_cell_mu_log_sigma_s_ss`, `noise_cell_tau2_log_sigma_s_ls`, `noise_cell_mu_log_sigma_s_ls`,
`noise_code_commit`. Per sweep (`noise_sweeps_<GROUP>.csv`): `specimen_id`, `noise_sweep_protocol`,
`noise_sweep_number`, `noise_tilt_mV_per_s`, `noise_sweep_sigma_f_mV`, `noise_sweep_sigma_m_raw_mV`,
`noise_sweep_sigma_m_mV`, `noise_sweep_v_log_sigma_m`, `noise_sweep_w_m`, `noise_sweep_sigma_s_raw_mV`,
`noise_sweep_sigma_s_mV`, `noise_sweep_v_log_sigma_s`, `noise_sweep_w`, `noise_sweep_flags`. The `_mV` columns
without `raw` are the twin's theta_tilde(c, s) (the cell's values for a failed sweep); the `raw`, `v` and `w`
columns are NaN for a failed sweep; `w_m` and `w` are lambda_m and lambda. `noise_rho_f` and
`noise_sweep_rho_f` no longer exist. JSON law dicts carry the keys `sigma_f_mV`, `sigma_m_mV`, `tau_m_s`,
`sigma_s_mV`, `tau_s_s`.

(i) Dev checks. The scripts under `dev/` dated before 2026-11-17 were written for the two-component law and no longer run against `noise_law.py` (`NoiseLaw` has five fields); each re-runs at the commit it was written at. The statements of 2.2 and 2.4 that legacy functions are kept "so that the dev checks re-run" hold in that sense only [final check, F3].

(h) Code interfaces changed by this section (the blocks' entries carry the rest unchanged).
Block 2, `noise_law.py`: `NoiseLaw(sigma_f, sigma_m, tau_m, sigma_s, tau_s)` (frozen dataclass; finite values;
sigmas >= 0; taus > 0; `as_array() -> (5,)`, `from_array((5,))`, `rho_m(dt)`, `rho_s(dt)`; `tau_f` removed);
`autocovariance(law, dt, lags)` = (L3'); `UnitDraws(fast, mid, slow, tilt, offset)`; `draw_units(rng, n_traces,
n_samples)` in the order of (b); `simulate_traces(law, dt, xi_fast, xi_mid, xi_slow, *, xi_tilt=None, sigma_b=0.0,
xi_offset=None, sigma_a=0.0, tilt_mV_per_s=None)`; `simulate_sweep(law, dt, n_samples, units, index=0, *,
sigma_b=0.0, sigma_a=0.0, tilt_mV_per_s=None)`; the closed-form functions with their present signatures.
Block 3, `noise_cell_stats.py`: the lag tuples of (c) (set by the contract commit); nothing else.
Block 4, `noise_fit.py`: `NoiseFitConfig` -- `fd_rel_step` (5 entries, (d)), `log_sigma_bounds` as today,
`log_tau_m_bounds = (log 4e-5, log 1e-3)`, `log_tau_s_bounds = (log 1e-3, log 10)` (`rho_bounds`,
`log_tau_bounds` removed), `bounds() -> (lo (5,), hi (5,))`; `law_to_u(law) -> (5,)`, `u_to_law(u (5,))`;
`closed_form_start` over the 4 initial points; `fit_cell` and `CellFitResult` as today with u (5,), the law dict
of (g), dof = d_eff - 5.
Block 9, `noise_sweep_fit.py`: `SweepStageConfig.min_sigma_f_ratio = 0.5`; `SweepFit.u (3,)` in the order of (e);
`SweepBootstrap` -- `v_m`, `v_s` (floats), `y_m`, `y_s` ((B,), NaN where a refit failed), `B`, `B_ok`, `flags`
(the field `v` and `y` are replaced); `SweepStageResult.pools`: dict protocol ('ss', 'ls') -> dict coordinate
('sigma_m', 'sigma_s') -> `PoolResult`; a row's keys: those of today with `rho_f` removed and `sigma_m_mV`,
`tau_m_s`, `log_sigma_m`, `v_log_sigma_m`, `bootstrap_y_m`, `lambda_m`, `sigma_m_tilde_mV` added (`bootstrap_y`,
`v_log_sigma_s`, `lambda`, `sigma_s_tilde_mV` keep their names and mean the slow part); `table_rows()` and
`cell_columns()` return exactly the per-sweep and Block 9 per-cell columns of (g), in that order (`cell_columns()`
led by `specimen_id`).
Block 6, `run_noise_fit.py`: the columns of (g); the smoke sizes unchanged.

### 2.7 The twins (Blocks 5, 10, 11, 12; D-180, D-182, D-183, D-186, D-187; written 2026-11-19)

A twin g is a synthetic copy of one real cell c = c(g): its responses are simulated in NEURON on c's archive
morphology with the twin's chosen parameters, and the noise of each sweep or window comes from c's recordings through
the noise fit (D-183). Several twins may share one real cell (D-188); the twin id g is an integer distinct from every
real specimen id only by the manifest's convention (Block 8), and enters this section only through the seeds, (e).

(a) What a twin has (D-186.1). For each real cell c, the twin g has every stored SS window of c -- window k of SS
sweep s, of polarity p(k), at position pos_k (0-based) in its train of period P samples, n_w samples at interval
Delta_t = 1 / f_s(c), sample j = 0 .. n_w - 1 at time t_j = (j - pre_n) Delta_t from the pulse onset -- and every
stored Long Square sweep s of c, of amplitude A(c, s) (pA, signed), onset t_on(c, s), duration D(c, s) and n(c, s)
samples at the same interval. Nothing is resampled: the noise is drawn in the twin's own samples (D-186.3).
[amended 2026-11-19, amendment 1, D-190 (user): of the stored Long Square sweeps the twin has every one whose
amplitude A(c, s) is at most `TwinConfig.ls_dep_min_amplitude_pA` (5 pA, the loader's threshold for a depolarising
sweep) and, of the depolarising ones (A(c, s) above it), those for which the fitter's loader's spike test on the REAL
voltage trace, `passive_fitting_hpc_fixed.sweep_spike_reason(v, f_s)` with its default thresholds, returns the empty
string. A depolarising sweep with a reason (`action_potential`, `peak_above_threshold`, `unusable`) is left out of the
twin; the twins' table keeps its row, with the reason and `in_twin` = False (D-190.2).]

(b) The law and the tilt of a twin sweep (D-073, D-076 (i), D-122.3, D-153). For each real cell c and each of its
sweeps s (SS or LS),

    theta_tilde(c, s) = ( sigma_f_hat(c, s), sigma_m_tilde(c, s), tau_m_hat(c), sigma_s_tilde(c, s), tau_s_hat(c) )   (P0')

read from the noise-fit run's tables (section 3): the three SDs from the per-sweep table's `noise_sweep_sigma_f_mV`,
`noise_sweep_sigma_m_mV`, `noise_sweep_sigma_s_mV` (mV), the two time scales from the per-cell table's
`noise_tau_m_ms`, `noise_tau_s_ms` divided by 1000 (s). For a failed sweep (its `noise_sweep_flags` holds an item
`sweep_fit_failed:...`) the per-sweep table's three SD columns already hold the cell's theta_hat(c): they are taken as
they are, never replaced a second time, and the row is marked `law_source = cell`. The tilt b_hat(c, s) (mV/s) is the
per-sweep table's `noise_tilt_mV_per_s` ((TL1) for SS, (TL2) for LS, 2.4). A stored sweep without a row, a row without
a stored sweep, a cell that is not `ok` in the per-cell table, or a law or tilt that is not finite is refused: no
value is substituted. [data inspected 2026-11-19, the nf2 run's 18 JSONs: in every cell every stored SS window belongs
to a sweep with a row (K = K_stored = the sum of the rows' K_s), every LS sweep has a row, and no tilt is missing; 64
SS and 525 LS rows, 9 LS rows failed.]

(c) An SS window of a twin (D-180, D-186.2, D-187). For each twin g of c, each SS sweep s of c and each stored window
k of s,

    y(g, k)[j] = r(g, p(k))[j] + e(g, k)[j] + b_hat(c, s) ( u(k, j) - u_bar(c, s) ),   j = 0 .. n_w - 1,      (TW1)

with r(g, p)[j] (mV) the clean response of twin g to one pulse of polarity p from rest (Block 11, (f)); e(g, k) one
draw of n_w samples of the stationary law theta_tilde(c, s) (L1', 2.6 (a), started in its stationary law), drawn
independently for every window k -- no correlation between windows (D-180.2); u(k, j) = (pos_k P + j) Delta_t (s) the
time of sample j of window k counted from the first sample of the stored window at position 0; and u_bar(c, s) the
mean of u(k, j) over all stored windows k of sweep s and all j, which equals mean_k(pos_k) P Delta_t + (n_w - 1)
Delta_t / 2. The tilt term is the real sweep's tilt line read at the window's real place: it equals (OF1)'s offset
b_hat(c, s) (t_k - t_bar_s) (`noise_windows.tilt_offsets`, t_k = pos_k P Delta_t) plus the ramp b_hat(c, s) (j - (n_w
- 1) / 2) Delta_t inside the window, and it sums to zero over the sweep's stored samples. There is no offset term
(D-183.3) and no random tilt.

(d) A Long Square sweep of a twin (D-182). For each twin g of c and each stored LS sweep s of c,

    y(g, s)[j] = r(g, s)[j] + e(g, s)[j] + b_hat(c, s) ( j - (n(c, s) - 1) / 2 ) Delta_t,   j = 0 .. n(c, s) - 1,   (TW2)

with r(g, s) the clean response to the step of amplitude A(c, s) on [t_on(c, s), t_on(c, s) + D(c, s)) (Block 11) and
e(g, s) ONE continuous draw of n(c, s) samples of theta_tilde(c, s) over the whole sweep. Beyond the 265 ms pre-window
this is an extrapolation of the law, accepted (D-182.2). A cell with tau_s_hat(c) > 265 ms (`TwinConfig.slow_flag_tau_s`)
is flagged in the twins' table (D-182.3; in the nf2 run 571430815, 526785799, 571624532, 548408551).

(e) Seeds (D-186.6). The unit draws of e(g, k) come from `numpy.random.default_rng(numpy.random.SeedSequence([base,
g, 0, s, pos_k]))` and those of e(g, s) (LS) from `SeedSequence([base, g, 1, s])`, with base =
`TwinConfig.base_seed`, 0 / 1 the protocol code of SS / LS (as `noise_sweep_fit.sweep_seeds`) and s the real sweep
number; from each generator the draws are taken by `noise_law.draw_units(rng, 1, n)` (order fast, mid, slow, tilt,
offset; the last two unused). Adding or removing a cell, a sweep or a window therefore changes no other window's or
sweep's random part; only u_bar(c, s), a constant of the sweep, depends on which windows of s are stored.

(f) The clean responses (D-183.1, D-187). SS: for each polarity p present among c's stored windows, ONE pulse of
amplitude A_p(c) = the mean of the stored windows' `peak_pA` of polarity p (pA, signed) and duration d_p(c) = the mean
of their `stim_duration_s` -- the values the fitter's loader gives the bundle it simulates
(`passive_fitting_hpc_fixed.load_cell_from_archive`) -- is simulated from the twin's resting state on the stored
window's time axis; every stored window of polarity p carries that response. No 37-pulse train is simulated (D-187.2).
LS: one simulation per stored sweep (sweeps of equal amplitude, onset, duration and length share one), from the
resting state, over the whole n(c, s) samples. The resting state is the twin's settled potential v_rest(g)
(`synthetic_ground_truth.measure_rin_tau_sag`), with every mechanism initialised at it.
[corrected 2026-11-19, amendment 2, D-191 [orchestrator]: `finitialize` at one uniform potential is NOT the settled
state of a cell whose rest is not uniform -- uniform e_pas with I_h absent from the axon and the spine factor scaling
g_pas but not gIhbar. Measured on the smoke's ball-and-stick I_h cell (Block 11's implementer; again by the
orchestrator, `dev/2026-11-19_twins/rest_models.py`): started uniformly at its measured soma rest the cell drifts by
0.056 mV before the onset, and its baseline-subtracted SS response differs from the settled cell's by up to 0.067 mV
(RMS 0.059) on 0.5-100 ms -- more than the smallest sigma_f of the nf2 run (0.031 mV). The resting state of a twin is
therefore the state reached after `TwinConfig.settle_ms` of simulation without stimulus from the uniform start:
every run of Block 11 begins `settle_ms` before its first output sample (drift then 6e-13 mV on the same cell).]

(g) What is not tested (D-184.4, D-189). No statistic of a twin's noise is compared with the noise fit's simulator or
with the real cell: Blocks 5, 10 and 11 carry code checks only. The twins' SS windows are independent, so a twin's SS
bundle averages its slow noise down as 1 / K whatever the real cell's does (a limit of Stage 7, D-189). A change of
holding potential between sweeps is not represented (D-183).

### Notation

[2026-11-17: rows written before section 2.6 describe the two-component law where they name rho_f, vectors of 5, 10 or 15 statistics, or unit draws of two components; under 2.6 the vectors hold 7 (SS), 14 (LS) and 21 statistics, the unit draws are fast, mid, slow, and the rows marked (2.6) below hold.]

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
| t_SS(c), t_star(c), t_floor | start of the enlarged window (ST2); max(t_floor, the median of the valid groups' starts) before the fallback (D-111); the floor (80 ms) | float | ms |
| (s, p), K_{s,p} | a group: the stored windows of SS sweep s with polarity p; its window count | --; int | --; -- |
| m_{s,p}(t) | the group's median window, per sample over its windows (D-103.1) | float[n_w] (or its enlarged part, float[m]) | mV |
| A_hat, B_hat, tau_hat; t_{s,p}; RSS(tau) | the decay fit B + A exp(-t / tau) of m_{s,p} (exact least squares, D-110); the group's start (ST1); the profiled residual sum of squares | float | mV, mV, ms; ms; mV^2 |
| T_end, min_window_ms, fallback_t_ms | end of the stored window (200 ms); the shortest window kept (50 ms); the fallback start (80 ms) | float | ms |
| t_tr(c), r_odd(t) | [legacy, superseded by D-103] the 5 ms bin test's time; the odd part of the polarity means | float; float[n_w] | ms; mV |
| sigma_SS_hat(c), kappa_tr | median pre-pulse SD over the kept windows (D-095.2); the start's tolerance factor (0.1, provisional) | float; float | mV; -- |
| x_tilde | a kept window's enlarged part minus its group's median window (D-103.1); [legacy] minus its polarity's mean window over the cell (D-051.1) | float[m] | mV |
| x'_k; j_f, m_f | a kept window minus its tilt offset (OF1), D-115.1 (t^c_k, t_bar_s as in (TL1)); the floor's first sample in a stored window and the samples from it to the end (2.4) | float[n_w]; int | mV; samples |
| q_hat(c, s) | the statistic vector of sweep s of c (SS: 5 medians over the sweep's windows; LS: the 10 values of its one window) | float[5] or float[10] | mixed |
| V_hat_c(h*), N_pairs | [legacy, superseded by D-102 / D-107] the data semivariogram at h* = 4 P (same polarity, same sweep, positions 4 apart); the pairs it averages | float; int | mV^2; -- |
| sigma_hat, rho_hat(L) | window SD (ddof 1); textbook autocorrelation | float | mV; -- |
| rho_adj(L) | the pre-D-028 estimator (test oracle only) | float | -- |
| theta [until 2026-11-17] | (sigma_f, rho_f, sigma_s, tau_s); `noise_law.NoiseLaw` | float[4] | mV, --, mV, s |
| theta (2.6, D-152) | (sigma_f, sigma_m, tau_m, sigma_s, tau_s): white fast SD, middle OU SD and correlation time, slow OU SD and correlation time; `noise_law.NoiseLaw` | float[5] | mV, mV, s, mV, s |
| u, u_s (2.6 (d), (e)) | the per-cell fit coordinates (log sigma_f, log sigma_m, log tau_m, log sigma_s, log tau_s); the per-sweep ones (log sigma_f, log sigma_m, log sigma_s) | float[5]; float[3] | natural logs of mV, mV, s, mV, s |
| v_m(c, s), v_s(c, s); lambda_m, sigma_m_tilde | (2.6 (e)) the bootstrap variances of log sigma_m_hat and log sigma_s_hat of sweep s; the middle part's shrinkage weight and shrunk SD | float | (log mV)^2; --, mV |
| xi | unit (standard-normal) draws of the simulator: fast, slow (n_traces, n); tilt, offset (n_traces,); `noise_law.UnitDraws` | float arrays | -- |
| E[S_0 \| theta, n], E[S_L \| theta, n], Var(x_bar \| theta, n) | closed-form moments of a demeaned window, topic doc (7a), (7b), (9) | float | mV^2 |
| q_cf_m(theta \| n_m) | closed-form statistic: sqrt(E[S_0]/(n-1)) for an SD, E[S_L]/E[S_0] for an autocorrelation (eq. 10); the fit's starting point only | float | mV; -- |
| V(h \| theta, sigma_b) | semivariogram of two window means h apart in one trace ((T1), legacy) | float | mV^2 |
| gamma(k \| theta) | autocovariance (L3) | float | mV^2 |
| a, b, sigma_b(c) | per-sweep offset, tilt, the tilt's SD (sigma_b(c): legacy, D-102) | float | mV, mV/s, mV/s |
| b_hat(c, s); alpha_hat_p; t^c_k, t_bar_s | the tilt read on sweep s (TL1 for SS, TL2 for LS); the polarity intercepts of (TL1); a window's centre time, their mean over the sweep | float | mV/s; mV; s |
| ybar_pre, ybar_end, Delta_t_c | an LS sweep's pre-window and end-window means; the distance between the windows' centres (6.8875 s on L3_exc) | float | mV, mV, s |
| q_hat(c), q_sim(theta), q_m(theta \| n_m) | the cell's statistic vector; its model estimate; the analytic single-window median | float[d]; float[d]; float | mixed |
| W, Sigma_hat, J | weight, simulated covariance, cost | float[d,d]; float[d,d]; float | mixed |
| M, R, B, k_max | binding windows; cells per Sigma_hat; bootstrap draws; iteration cap | int | -- |
| u | the fit coordinates (log sigma_f, rho_f, log sigma_s, log tau_s) | float[4] | log mV, --, log mV, log s |
| M_cells, n_LS, L_s | replicate cells behind the SS binding function (ceil(M / K)); samples of an LS pre-window; samples of a simulated SS sweep ((train_length - 1) P + n_w) | int | --; samples; samples |
| C, kappa(C_hat), d_eff | Cholesky factor of Sigma_hat; condition number of its correlation matrix; statistics kept after dropping the non-finite ones | float[d_eff, d_eff]; float; int | mixed; --; -- |
| h* | [legacy] the tilt's lag, 4 P Delta_t | float | s |
| seed_binding, seed_Sigma_k, seed_boot | per-cell child seeds (`SeedSequence([seed, specimen_id])`): the binding function's common random numbers; Sigma_hat's draws at iteration k; the bootstrap | int | -- |
| S(c), S_SS(c), S_LS(c); pr; S'_pr(c), k_pr | (2.5) the sweeps of c in the per-sweep stage, all and by protocol; a protocol, SS or LS (not a polarity p); the sweeps of protocol pr whose fit did not fail, their count | sets; label; set, int | -- |
| q_sim(theta \| s), K_s, d_s | the per-sweep binding function (2.5 (b)); the sweep's kept SS windows; its statistics kept (>= 4) | float[15] (NaN outside the sweep's protocol); int; int | mixed; --; -- |
| u_s, theta_hat(c, s), J_s, dof_s | the per-sweep fit coordinates (log sigma_f, rho_f, log sigma_s); the sweep's estimate, with tau_s_hat(c); the cost (P1) at it; d_s - 3 | float[3]; `NoiseLaw`; float; int | log mV, --, log mV; mV, --, mV, s; --; -- |
| W_s, Sigma_hat_s | the sweep's weight and simulated covariance (Block 4's `per_sweep_cov[s]`, kept statistics) | float[d_s, d_s] | mixed |
| B_v, B_ok, q*_b(c, s), y*_b, v(c, s) | the bootstrap's refits per sweep (not the window bootstrap's B of W_0); those that did not fail; the b-th replicate's statistic vector; its refit's log sigma_s_hat; the sampling variance (P2) | int; int; float[15]; float; float | --; --; mixed; log mV; (log mV)^2 |
| y_s, v_s, w_s, w*_s; Q_pr(c) | log sigma_s_hat(c, s), v(c, s) and the fixed- and random-effects weights of 2.5 (e); Cochran's Q | float | log mV, (log mV)^2, (log mV)^-2, (log mV)^-2; -- |
| tau2_raw_pr(c), tau2_hat_pr(c), mu_hat_pr(c) | the DerSimonian-Laird moment estimate of the between-sweep variance of log sigma_s in protocol pr (P3), raw and clipped at 0 (not tau_s); the random-effects mean (P4) | float | (log mV)^2; (log mV)^2; log mV |
| lambda(c, s), sigma_s_tilde(c, s), theta_tilde(c, s) | the shrinkage weight; the shrunk slow SD (D-076.1); the law the twin of s carries (P0) | float; float; `NoiseLaw` | --; mV; mV, --, mV, s |
| seed_binding(c, s), seed_v(c, s) | per-sweep child seeds: the per-sweep binding function's common random numbers; the bootstrap's replicates | int | -- |

## 3. Data contracts

| Item | Source / format | Shape and axis order | dtype | Units | Conventions |
|---|---|---|---|---|---|
| a window `x` for `sample_acf` | any 1-D array (data or simulation) | (n,) | float64 | mV (any unit; result dimensionless) | lags in samples, >= 1; NaN rules of 2.2 |
| LS pre-window | `noise_calibration.pre_window_samples(mono, bundle)` | (n,), n = 13250 at 50 kHz for 265 ms | float64 | mV, LJP-corrected | [t_0, t_on - 5 ms), t_0 the first stored sample |
| LS end window (D-107.1) | `ls_sweeps.npz` (`v_<index>`, `i_<index>`, `index` from `metadata.json['ls_sweeps']`), read by `noise_windows.load_ls_prewindows` | (n_end,), n_end = round(500 ms f_s) = 25000 at 50 kHz, the last n_end samples ending at the sweep's last finite sample | float64 | mV, LJP-corrected | must lie entirely after the step (first sample after t_off); t in s from the sweep's first sample; L3_exc sweeps hold 6.000 s after the step (living handoff 4.6) |
| SS stored pulse window | `ss_pulses.npz` Format A via `load_cell_from_archive` | per pulse (n_w,), t in s, t = 0 at pulse onset, window (-10, +200) ms | float64 | mV | polarity 'dep'/'hyp'; sweep number NOT read by the loader (open, D-056) |
| SS sweep layout (data inspected 2026-10-20, two cells) | `ss_pulses.npz` keys `sweep_number` (int64, one per stored window, windows in onset order); `metadata.json['ss_extraction']` (`n_pulses_raw`, `n_pulses_ampdur_qc`, `n_pulses_qc`, `n_dep_pre_trace_qc`, `n_hyp_pre_trace_qc`, `trace_qc` with `engine`, `alpha`); `trace_qc_results.csv` (one row per pulse that entered the trace QC, in the same order, columns `pulse_index`, `polarity`, `passed`, `reason`, ...) | per SS sweep a train of 37 pulses of 0.5 ms, polarity alternating from dep (19 dep, 18 hyp), onset to onset 10025 samples = 200.5 ms at 50 kHz; the stored windows are the train's pulses that passed the writer's trace QC (engine v2, alpha 0.5), in order | -- | -- | 508282493: 2 sweeps, 74 pulses detected, 74 after the amplitude/duration QC, 40 stored (34 dropped by the trace QC); 569844159: 10 sweeps, 370, 370, 348 (22 dropped). Every gap is a trace-QC drop, and the position of every stored window in its train follows from `trace_qc_results.csv` (the stored windows are its `passed` rows, in order). Sources: cluster G12 (polarity strings, spacing) and the Google Drive copy of the two cells' archive files (counts, QC table), which reproduces the G12 strings exactly. All 18 L3_exc cells checked on the cluster 2026-10-21 (decision log D-072): every cell carries `trace_qc_results.csv` with as many rows as `n_pulses_raw` and `n_pulses_ampdur_qc`, a multiple of 37 -- 2 SS sweeps in 14 cells, 6 in 571430815, 10 in 569844159, 571483807, 571624532; engine v2, alpha 0.5 everywhere; 6-50 % of pulses dropped by the trace QC (median 27 %). Block 3 still checks each cell's per-sweep split and polarity sequence. |

| the noise-fit run's tables (run tag nf2; data inspected 2026-11-19, the user's upload `nf2_L3_exc.tar.gz`) | `noise_cells_<GROUP>.csv`: one row per cell, columns `specimen_id`, `status`, `reason`, `noise_sigma_f_mV`, `noise_sigma_m_mV`, `noise_tau_m_ms`, `noise_sigma_s_mV`, `noise_tau_s_ms`, ... (`run_noise_fit.CELL_COLUMNS`); `noise_sweeps_<GROUP>.csv`: one row per sweep, `run_noise_fit.SWEEP_COLUMNS`; `cells/specimen_<id>.json`: `sweep_stage.twin_laws["<ss or ls>:<sweep>"]` = {`sigma_f_mV`, `sigma_m_mV`, `tau_m_s`, `sigma_s_mV`, `tau_s_s`} and `sweep_stage.rows[]` (`protocol`, `sweep`, `tilt_mV_per_s`, `failed`, `reason`, `K_s`) | 18 cell rows; 589 sweep rows (64 SS + 525 LS); 9 LS rows failed (`at_bound:log_sigma_m` 3, `degenerate:sigma_f` 5, `no_convergence` 1) | float64 | mV; ms in the per-cell table's `_ms` columns, s in the JSON's `_s` fields; mV/s | `noise_sweep_protocol` is `SS` / `LS` (upper case), the JSON's `protocol` `ss` / `ls`; an empty CSV cell is a missing value; a failed row's SD columns hold theta_hat(c) |
| a real cell's stimulation protocol (data inspected 2026-11-19, 508282493, through `noise_windows.load_ss_windows` and `load_ls_prewindows`) | `ss_pulses.npz` (`t`, `v`, `i_pA`, `polarity_is_dep`, `peak_pA`, `stim_duration_s`, `sampling_rate_Hz`, `sweep_number`), `trace_qc_results.csv`, `ls_sweeps.npz` (`v_<index>`, `i_<index>`), `metadata.json['ls_sweeps']` (`index`, `sweep_number`, `detected_amplitude_pA`, `sampling_rate_Hz`, `n_samples`) | SS: `t` (10500,), `v` (40, 10500); pre_n = 500, period 10025 samples, train of 37; sweeps 92 (23 windows) and 93 (17); `peak_pA` = +/-200.0000027 in every window, `stim_duration_s` = 0.0005, the current on at samples 500..524. LS: 22 sweeps (54..75), each 363500 samples, the step on at samples 13500..63499 (onset 0.27 s, duration 1.0 s), amplitudes -110 to +310 pA in steps of 20 | float64, int64, bool | mV (LJP-corrected), pA, s, Hz | the LS amplitudes include depolarising steps on which the real cell may spike; the fitter's loader keeps spiking sweeps out by a test on the voltage (`sweep_has_spike`), which a twin without Na/Kv does not fail for spikes; the depolarising steps are validation only (section 8) |
| the twins' table (Block 5) | `twin_sweeps_<GROUP>.csv`, written by `twin_table.write_twin_sweep_table` | one row per real sweep (SS and LS), columns `twin_table.TWIN_SWEEP_COLUMNS` in that order | as the entry of Block 5 states | mV, s, mV/s, pA, Hz | floats written with 17 significant digits (`%.17g`), so that reading returns the same float64 |

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
  - D-073: each twin sweep is calibrated on its own real sweep (form open, section 8); statistics are kept per sweep as well as per cell. [2026-10-30: the form is D-076 (i), estimated as D-116 states, section 2.5.]
  - D-116: the per-sweep stage of 2.5 (Block 9): one solve per sweep at the cell's weights from theta_hat(c), tau_s held at tau_s_hat(c); v(c, s) by a parametric bootstrap; (D-076.1) per protocol, the SS and the LS sweeps separately; a failed sweep flagged, left out of the pooling, its twin at the cell's values.
  - D-095, D-103, D-104, D-106, D-110 (confirmed by D-114.1), D-111: the SS window start and subtraction of 2.2 (a)-(g): the median window per sweep and polarity, its decay fitted by exact least squares (variable projection, no starting values; a minimum at a bound is a failed fit), the point value (ST1), the MEDIAN over valid groups with the 80 ms floor, the 80 ms fallback below a 50 ms window, groups under 5 windows out of everything, a failed fit out of the median only.
  - D-102, D-107: every sweep's tilt is read (2.4: (TL1) for SS, (TL2) for LS) and copied into the twins and the fit's simulator; no statistic of the tilt is computed.
  - D-114.2: a cell with a kept sweep whose tilt cannot be read (`ss_tilt_none:<sweep>`, `ls_tilt_none:<sweep>:<reason>`) is refused by the fit (`CellLayout.from_windows` raises, quoting the sweep and the flag); no tilt is substituted.
  - D-180, D-182, D-183, D-186, D-187 (the twins, section 2.7): each SS window's noise an independent draw of its sweep's law with the real sweep's tilt line at the window's real place; each LS sweep one continuous draw over the whole sweep; no offset; every response simulated on the real morphology, each SS pulse alone from rest; all stored SS windows and LS sweeps; noise at the twin's own sampling rate; LS amplitude and timing from the real cell; seeds per twin sweep.
  - D-184.4, D-189: no statistic of a twin's noise is compared with the noise fit's simulator or with the real cell; the SS bundles' cross-window correlation is not measured.
  - D-184, D-188 (Stage 7, Block 8, to specify): three configurations of 10 twins each (`full`, `passive_fit`, `active_fit`); D-185: one PBS job per twin, 120 cores in all.

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

[2026-11-17, D-152: the law is 2.6 (a), the simulator 2.6 (b), the interfaces 2.6 (h). The smoke `smoke_noise_law.py` is re-aimed at them: every oracle of this entry that names the fast AR(1) part or rho_f is restated for the white fast part and the middle OU (the closed-form moments against the brute-force covariance matrix and Monte Carlo; the simulator's sample autocovariance against (L3') at lags 0, 1, 2, 5, 15, 50, 500; the draw order; a component switched off by a zero SD; `from_array` / `as_array` round trip on 5 entries). Status: smoke-tested 2026-11-17 (sandbox: `smoke_noise_law.py` M1-M11 11/11, about 3 s; five planted defects -- the middle part with rho_s, mid and slow draws swapped, a non-stationary first sample, the white term at every lag, `from_array` permuted -- each caught; commit `de61d4b`). As built [orchestrator, accepted at the gate]: the check against the topic document's 14 two-component values (old M8) is replaced by the white part's analytic moments (E[S_0] = (n-1) sigma_f^2, E[S_L] = -(n-L) sigma_f^2 / n) and the additivity of every moment over the three components; `semivariogram_continuous` (legacy) takes the white part at its sampled value sigma_f^2 dt / T per window; `simulate_traces` requires all three draw arrays.]

- Module: `noise_law.py` (model layer: no I/O, no plotting, no module-level RNG); smoke `smoke_noise_law.py`.
- Public API: `NoiseLaw(sigma_f, rho_f, sigma_s, tau_s)` (frozen dataclass; `validate` in `__post_init__`; `as_array`/`from_array`; `rho_s(dt)`, `tau_f(dt)`); `autocovariance(law, dt, lags) -> float[len(lags)]` (L3); `draw_units(rng, n_traces, n_samples) -> UnitDraws` (fast, slow (n_traces, n); tilt, offset (n_traces,); drawn in that fixed order); `simulate_traces(law, dt, xi_fast, xi_slow, *, xi_tilt=None, sigma_b=0.0, xi_offset=None, sigma_a=0.0, tilt_mV_per_s=None) -> float64[n_traces, n]` (L1)-(L2); `simulate_sweep(law, dt, n_samples, units, index, *, sigma_b, sigma_a, tilt_mV_per_s=None) -> float64[n]` (one continuous trace, D-056 (i)); `tilt_mV_per_s` (added 2026-10-29, D-102.1, D-107.2): the tilt as a FIXED number per trace (mV/s; `simulate_traces`: (n_traces,), `simulate_sweep`: a float), each trace's real sweep's tilt read by Block 3 (SPEC 2.4) -- the alternative to the random tilt `sigma_b xi_tilt`, with which it cannot be combined; `var_window_mean(law, dt, n)` (7a); `expected_s0(law, dt, n)` (7b); `expected_sl(law, dt, n, lags)` (9); `closed_form_sd(law, dt, n)`; `closed_form_acf(law, dt, n, lags)` (10); `visible_share(z)` (8); `window_mean_cov(law, dt, n1, n2, offset)`; `semivariogram(law, dt, n1, n2, offset, sigma_b=0.0)` (T1's V, exact for the sampled law); `semivariogram_continuous(law, dt, T1, T2, h, sigma_b=0.0)` (topic doc eq. 25 as written; oracle).
- Inputs: theta in mV, --, mV, s with rho_f the lag-1 coefficient AT the `dt` passed (D-014; the transfer between sampling rates, D-017 (f), is not done here); dt in s; unit draws float (n_traces, n >= 2); sigma_b in mV/s, sigma_a in mV; lags and offsets in samples.
- Outputs: traces in mV, float64, (n_traces, n); each recursion started in its stationary law (e_0 = sigma xi_0); the tilt of trace r is sigma_b xi_tilt[r] (t_j - t_bar), or tilt_mV_per_s[r] (t_j - t_bar) when the fixed tilts are passed, t_bar = (n-1) dt / 2 the trace's own mid-time, so it sums to zero over the trace (any constant is invisible to every demeaned statistic and to the within-trace semivariogram); moments in mV^2; `expected_sl`/`closed_form_acf` NaN outside 1 <= L <= n-1. Refused (ValueError): `tilt_mV_per_s` together with sigma_b > 0, of the wrong shape, or with a non-finite entry. The fixed tilt consumes no unit draw (`UnitDraws.tilt` is left unused).
- Parameters: none (no knob; sizes come from the caller's config, Block 4).
- Library calls relied on: `scipy.signal.lfilter([1], [1, -rho], eta, axis=-1)` for both AR(1) recursions; `numpy.expm1` for 1 - rho_s^2 and z - 1 + e^{-z}.
- Custom code (marked `# Custom:` at the definition): the exact first moments (7a), (7b), (9) of a demeaned window (zeta_j from the cumulative sum of gamma, O(n)); the window-mean covariance by lag counts over sample pairs (O(n1 + n2)). No library computes either.
- Test oracles (`smoke_noise_law.py`, M1-M10): M1 brute force -- (7a), (7b), (9) against the full n x n covariance matrix (E[S_L] = sum_j (P Gamma P)_{j,j+L}) at n = 240, 5 lags, 3 laws, 1e-10 relative; M2 Monte Carlo (4000 traces x 2000 samples, cohort law) -- mean S_0, S_1, S_50 and the across-trace autocovariance at lags 0, 1, 50 within 3.5 SE of the closed forms; M3 stationarity -- first- and last-sample variance equal gamma(0) within 3.5 SE; M4 window means -- `window_mean_cov` equals brute force (1e-10; overlapping, disjoint, negative offset), the Monte Carlo covariance of two 190 ms means 200 ms apart within 3.5 SE, exact-discrete V against eq. (25) within 3 dt/tau_s (slow-only law); M5 tilt -- Monte Carlo V equals V(h|theta,0) + sigma_b^2 h^2/2 within 3.5 SE, the tilt sums to 0; M6 contract -- shapes, float64, finite, bit-identical on the same draws, different on others, `simulate_sweep` == a row of `simulate_traces`, six invalid laws refused, draw order fixed; M7 visible share -- E[S_0]/(n gamma(0)) equals v(z) within 2z/n at z in {0.25, 1.26, 6.67}; M8 regression -- q_cf at theta_cf = (0.0380 mV, 0.175, 0.0508 mV, 36.6 ms) reproduces the 14 q^cf values of topic doc 3.5.3 within 3e-4 mV and 3e-3; M9 ASCII/LF; M10 fixed tilt (added 2026-10-29) -- the traces with `tilt_mV_per_s = b` minus the same draws without a tilt equal b[r] (t_j - t_bar) to 1e-12 mV (float64 roundoff on an addition of mV-sized terms), for tilts of both signs and a zero; `tilt_mV_per_s = sigma_b xi_tilt` gives traces bit-identical to the pair (xi_tilt, sigma_b); `simulate_sweep` with a scalar tilt equals row `index` of `simulate_traces`; both tilts at once, a wrong shape and a NaN entry each raise ValueError. Mutation-tested 2026-10-19: a non-stationary start (M2, M3 fail), the Var term dropped from (9) (M1, M2, M8), an uncentred tilt (M5), 1 - rho_s in place of 1 - rho_s^2 (M2, M4, M5), an off-by-one pair count (M4 brute force) each FAIL.
- Data flow: consumed by Block 3 (`noise_windows` cuts the enlarged SS and LS windows from traces of this simulator at the real sweep structure), Block 4 (`noise_fit`: binding function and Sigma_hat from `simulate_traces` on frozen `UnitDraws`; `closed_form_sd`/`closed_form_acf` for the start; `semivariogram` for (T1)), Block 5 (the generator's OU term through the same simulator).
- Confirmed: yes -- D-017, D-034, D-056 (i) (user); the exact-discrete V in place of eq. (25)'s continuous form is the assistant's refinement, recorded in 2.4 and in the decision log's changelog (2026-10-19).
- Status: smoke-tested 2026-10-19 (sandbox: M1-M9 9/9, about 5 s; mutation-tested as above); the fixed tilt smoke-tested 2026-10-29 (sandbox: M1-M10 10/10; M10 mutation-tested with four planted defects -- an uncentred tilt, its sign flipped, the exclusivity check removed, `simulate_sweep` dropping the tilt -- each FAILs M10).

### Block 3: windows and statistics -- the archive's SS layout, the enlarged windows, the LS pre-windows and the D-051 statistic vector (D-051, D-072, D-073, D-076; window start and subtraction D-095, D-103, D-104, D-106, D-110, D-111, D-115; per-sweep tilts D-102, D-107, D-115)

[2026-11-17, D-153.2, D-154.1: the lag lists and d = 21 are 2.6 (c); `noise_cell_stats.py`'s three lag tuples are set by the contract commit. The fixture writer of `smoke_noise_windows.py` takes the law of 2.6 (a) (its `THETA_CF`-style default becomes a five-parameter law: sigma_f 0.035 mV, sigma_m 0.017 mV, tau_m 0.3 ms, sigma_s 0.055 mV, tau_s 40 ms [orchestrator]) and the simulator of 2.6 (b); the smokes of Blocks 1 and 3 (`smoke_noise_stats.py`, `smoke_noise_windows.py`) pass with the new counts, their oracles otherwise unchanged. Status: smoke-tested 2026-11-17 (sandbox: `smoke_noise_stats.py` 6/6, `smoke_noise_windows.py` W1-W15 15/15 on the five-parameter fixture, the 21 labels compared with 2.6 (c); commit `de61d4b`).]

- Modules: `noise_windows.py` (data layer: loading, layout, windows, the window start, the subtraction; no fitting of the noise law, no plotting), `noise_cell_stats.py` (analysis layer: the statistics per window, per sweep and per cell, the bootstrap SE, the per-sweep tilts; no I/O); smoke `smoke_noise_windows.py`.
- Public API (`noise_windows`): `WindowConfig` (frozen dataclass; the pipeline's fields: `t_ss_floor_ms=80.0` (D-076 (ii), D-103.3), `kappa_tr=0.1` (D-051, PROVISIONAL), `fit_start_ms=5.0` (D-103.2, PROVISIONAL), `decay_tau_bounds_ms=(1.0, 1000.0)`, `decay_grid_n=301` (D-110, PROVISIONAL; `decay_tau0_ms` and `decay_maxfev` of `f293d4a` retired), `min_window_ms=50.0` (D-104.1), `fallback_t_ms=80.0` (D-103.3; was 190), `min_windows_per_group=5` (D-104.2, PROVISIONAL), `ss_pre_ms=10.0`, `ls_pre_gap_ms=5.0`, `ls_first_ms=50.0`, `ls_last_ms=132.5`, `ls_end_ms=500.0` (D-107.1, PROVISIONAL length), `ls_polarities=("hyp", "dep")`, `ls_amp_min_pA=5.0`, `ss_period_samples=None` (measured), `min_overlap=100`; legacy fields read only by `transient_start`: `transient_bin_ms=5.0`, `transient_floor_z=None`, `transient_fallback_t_ms=190.0`); `load_ss_windows(spec_dir, cfg) -> SSWindows` (fields: `specimen_id`, `fs`, `dt`, `t` (n_w,), `v` (K, n_w) mV, `polarity_is_dep` (K,), `sweep_number` (K,), `position` (K,) int, `train_length` int, `period_samples` int, `pre_n` int, `n_raw`, `n_sweeps`); `measure_period(ss, min_overlap) -> int`; `DecayFit` (`A_mV`, `tau_ms`, `B_mV`, `rss`, `ok`, `at_bound`, `message`); `fit_decay(t_ms (n,), y_mV (n,), cfg) -> DecayFit` (2.2 (d), (ST0): the exact least-squares fit of one median window's fitted samples by variable projection; `profile_decay(t_ms, y_mV, tau_ms) -> (A, B, RSS)`, the closed form at one tau); `StartResult` (`t_ss_ms`, `t_star_ms` (max(80 ms, the median of the valid groups' starts); NaN when no group is valid), `sigma_ss_mV`, `keep` (K,) bool (False for the windows of the small groups), `group_sweep` (G,), `group_is_dep` (G,), `group_K` (G,), `group_status` (G,) 'ok' / 'few_windows' / 'fit', `A_mV`, `tau_ms`, `B_mV`, `t_group_ms` (G,) (NaN where no fit), `flags` tuple of 'ss_group_dropped:<sweep>:<polarity>:<few_windows or fit>', 'ss_start_floor', 'ss_start_fallback', `fallback_reason` ('', 'short_window', 'no_valid_group')); `decay_start(ss, cfg, fit=fit_decay, tilts=None) -> StartResult` (2.2 (a)-(e); groups in the order of the sweeps' first windows, dep before hyp; `fit` a callable with `fit_decay`'s signature, swappable for a test double; with `tilts` (a `SweepTilts`) the kept groups' median windows are taken on the windows corrected by (OF1) over the kept set, D-115.1 -- without, the plain rule); `select_windows(ss, keep) -> SSWindows` (a copy holding the kept windows, every other field unchanged); `enlarged_windows(ss, t_ss_ms) -> (X (K, m), j0)`; `subtract_median_response(X, polarity_is_dep, sweep_number) -> (X_tilde (K, m), medians dict (sweep, is_dep) -> (m,))` (2.2 (g)); `SSNoiseWindows` (`start`, `ss` (the kept `SSWindows`, as stored), `X_tilde` (K_kept, m, from the corrected windows), `j0`, `tilts` (the SS reading of D-115.2, a `SweepTilts`); `m` property); `ss_noise_windows(ss, cfg, fit=fit_decay) -> SSNoiseWindows` (2.2 (a)-(g) in order, D-115: the small groups out, (TL1) on the kept windows at the floor, the offsets (OF1), the start on the corrected windows, the enlarged windows, the subtraction -- the one entry point of the pipeline and of Block 4's driver); the SS tilt reading and the offsets, shared by the data and Block 4's simulator (D-115.3): `SweepTilts` and `ss_sweep_tilts(X, ss, j0) -> SweepTilts` ((TL1) on the window means of X cut at j0; moved here from `noise_cell_stats` on 2026-10-30, which re-exports both), `tl1_fit(means (K,) mV, centres (K,) s, sweep_number (K,), polarity_is_dep (K,)) -> (sweeps tuple, b (S,) mV/s, se (S,), n (S,) int, flags tuple)` ((TL1)'s regression), `read_ss_tilts(ss, cfg) -> SweepTilts` ((TL1) at the floor: `ss_sweep_tilts(ss.v[:, j_f:], ss, j_f)`, j_f = pre_n + round(t_ss_floor_ms f_s / 1e3)), `tilt_offsets(position (K,), sweep_number (K,), period_samples, dt, b: dict sweep -> mV/s) -> (K,) mV` ((OF1)'s offsets, b_s (pos_k P Delta_t - their mean over the sweep's windows given); 0 for a sweep whose b is absent or NaN), `remove_tilt_offsets(ss, tilts) -> SSWindows` (a copy with (OF1) applied to every window, all samples); `load_ls_prewindows(spec_dir, cfg) -> LSWindows` (fields: `specimen_id`, `fs`, `dt`, `sweep_number` (K_LS,), `amplitude_pA` (K_LS,), `polarity` (K_LS,) str, `pre` list of K_LS float64 arrays, `n_pre` (K_LS,), `onset_s` (K_LS,), and for (TL2) `end` list of K_LS float64 arrays (empty where the end window is not usable), `end_start_s` (K_LS,) (NaN there), `step_end_s` (K_LS,) (t_off), `n_samples` (K_LS,), `end_status` (K_LS,) 'ok' / 'tail_short'); `ls_segments(pre, fs, cfg) -> (first (n1,), last (n2,))`; `LayoutError`. Legacy, kept unused by the pipeline for the dev checks of 2026-10-23 to 2026-10-29 (superseded by D-103): `transient_start(ss, cfg) -> TransientResult` (the 5 ms bin test of D-051 (iv); its fallback flag `ss_enlarged_none` at `transient_fallback_t_ms`), `subtract_mean_response(X, polarity_is_dep) -> (X_tilde, mean_dep, mean_hyp)` (D-051.1).
- Public API (`noise_cell_stats`): `SS_LAGS_MS=(0.02, 0.1, 1.0, 10.0)`, `LS_FULL_LAGS_MS=(0.02, 0.1, 1.0)`, `LS_LAST_LAGS_MS=(0.02, 0.1, 1.0, 10.0)`, `STAT_LABELS` (15, in the order of 2.2), `STAT_BLOCK` (15 of 'ss'/'ls'); `window_stats(X, fs, lags_ms) -> (K, 1 + len(lags))` (sigma_hat then rho_hat per lag, through `noise_stats.sample_acf`); `ss_window_stats(X_tilde, fs) -> (K, 5)`; `ls_window_stats(pre_list, fs, cfg) -> (K_LS, 10)`; `StatisticVector` (`q` (15,), `labels`, `block`, `n` (15,) window lengths, `K` (15,) counts, `d`); `statistic_vector(ss_stats, m, ls_stats, n_ls) -> StatisticVector` (NaN-ignoring medians; a statistic with no window is NaN and its K is 0); `per_sweep_vectors(stats, sweep_number) -> dict sweep -> median vector`; `bootstrap_median_se(per_window (K, d), B, rng) -> (d,)` (window-level bootstrap of the median, D-049's W_0 input); `SweepTilts` (`protocol` 'ss' / 'ls', `sweep_number` (S,), `b_mV_per_s` (S,), `se_mV_per_s` (S,) (OLS SE for SS; NaN for LS), `n_windows` (S,) (SS: the sweep's kept windows; LS: 2), `flags` tuple; `as_dict() -> {sweep: b}`) and `ss_sweep_tilts(X, ss, j0) -> SweepTilts` ((TL1), `ss` the kept `SSWindows` aligned with the rows of `X`), both defined in `noise_windows` since 2026-10-30 (D-115) and re-exported here; `ls_sweep_tilts(ls) -> SweepTilts` ((TL2)). Legacy: `window_mean_semivariogram(X_tilde, ss, lag_periods=4) -> (V_hat, N_pairs)` ((T1)'s data side, superseded by D-102 / D-107).
- Inputs: a Phase-0 specimen directory with `ss_pulses.npz` (Format A, `sweep_number`), `trace_qc_results.csv`, `metadata.json` (`ss_extraction.n_pulses_raw`, `n_pulses_ampdur_qc`, `n_pulses_qc`, `ls_sweeps`), `ls_sweeps.npz`. Voltages in mV as stored (LJP-corrected by the writer), f_s in Hz.
- Layout rule (D-072): the QC table's rows are the sweep trains in order, each of length n_raw / n_sweeps (an integer, else `LayoutError`); row r's position is r mod train_length and its sweep the n-th entry of `metadata.json['ss_extraction']['sweep_numbers']` (the writer's visiting order; the npz's sweep numbers ascending when that list is absent); the `passed` rows, in order, are the stored windows. Refused (`LayoutError`, with the reason) when: rows != n_pulses_raw or != n_pulses_ampdur_qc; passed rows != K; the passed rows' polarities or sweep counts differ from the npz; a train's polarity does not alternate; an npz sweep number not in the metadata list; the measured period disagrees with `ss_period_samples` when given, or no adjacent stored pair exists to measure it; two adjacent stored windows disagree on their overlap.
- Outputs: windows in mV, float64; positions 0-based; statistics dimensionless except the SDs (mV); tilts in mV/s; times of the start in ms; every array's first axis is the window (or the group, or the sweep).
- Parameters: the `WindowConfig` fields above.
- Library calls relied on: `numpy` (loading, `median`, `nanmedian`, `geomspace`, `linalg.lstsq`), `scipy.optimize.minimize_scalar` (method 'bounded') and `scipy.optimize.brentq` (the 1-D steps of D-110), `statsmodels` through `noise_stats.sample_acf`, `csv`/`json` (standard library).
- Custom code (marked `# Custom:`): the layout reconstruction from the QC table (no library reads the writer's files); the overlap test of adjacent windows (as in `noise_fit_gates.onset_spacing`, re-implemented without the tools module so the data layer imports nothing from the tools); the variable projection of (ST0) (no SciPy routine profiles out the linear parameters of a separable model; its 1-D steps are SciPy's) and the start rule (ST1)-(ST2) (groups, median, floor, fallback); the end window of (TL2) (the rule of `ls_baseline_qc.baseline_record`, re-implemented so the data layer imports no pandas); the legacy transient test and semivariogram.
- Test oracles (`smoke_noise_windows.py`, W1-W15; fixtures: Block 3's writer-like cell -- 2 SS sweeps x 37 pulses, period 10025, RC response 113 MOhm / 28.143 ms + Block-2 noise at theta_cf, 11 pulses dropped by a planted QC table (groups of 16, 14, 17, 16 windows), six LS sweeps of 270 ms + 1 s step + 0.5 s tail; and a second cell with a 1.5 s LS tail and per-sweep tilts planted at sigma_b = 0.1 mV/s): W1 layout -- read back with the planted positions and period exactly; a wrong count, a swapped polarity, a non-alternating train, a wrong `ss_period_samples` each raise `LayoutError`; a false 16-sample head rejected; W2 start (D-103, D-104, D-110, D-111) -- on noise-free planted groups (B + A exp(-t / tau) for t >= 0, white noise of a known SD before t = 0 only, and a 1 mV artefact on [0, 5) ms that the fit must not see) `decay_start` returns A, tau, B to 1e-9 relative, sigma_SS_hat = the median pre-pulse SD (1e-15), and t_{s,p} within one sample (0.02 ms) of tau ln(A / (kappa_tr sigma_SS_hat)) (tau = 28.143 ms, A = 0.4 mV: about 130 ms); a fast decay (tau = 5 ms) gives t_SS = 80 ms with `ss_start_floor`; groups planted at 100, 120, 140 and 160 ms give t_SS = 130 ms (the median of an even count, D-111); every group planted at 151 ms falls back to 80 ms with `ss_start_fallback` ('short_window'), at 149 ms keeps t_SS = 149 ms and no fallback flag; a ramp (no decay) puts every group's minimum at the upper bound: all flagged `fit`, `ss_start_fallback` ('no_valid_group'), every window kept; a fit double failing the first group flags it `fit`, keeps its windows and takes the median over the other groups; the precision of the fit (D-110) on each noisy group of the fixture: the residual's derivative (ST0) at tau_hat below 1e-8 of its scale, equal to a central finite difference of RSS (1e-6 relative), `scipy.optimize.curve_fit` started at the solution returning it (tau within 1e-7 relative, RSS not lower by more than 1e-12 relative), and no point of a dense grid (20001 points over [tau_hat / 2, 2 tau_hat]) below RSS(tau_hat); every group's fit valid with tau_hat within 25 % of 28.143 ms (reported: t_{s,p}, t_SS); W3 subtraction (D-103.1) -- the per-sample median of the subtracted windows is 0 per group (1e-12); the returned medians equal `numpy.median` per group computed by an independent loop (exactly); a different common vector added to each group leaves X_tilde unchanged (1e-12) and the statistics of response + noise after the subtraction equal those of the noise alone after it (1e-10); W4 statistics -- `window_stats` equals `np.std(ddof=1)` and `sample_acf` per window (1e-14); the 15 labels, blocks, n and K of 2.2; the LS pre-window equals `noise_calibration.pre_window_samples(mono, bundle)` on the same sweep through the monolith (skipped without NEURON); W5 per-sweep vectors -- through `ss_noise_windows`: keys are the sweep numbers, each the median over that sweep's kept windows; the cell vector the median over all kept windows; W6 bootstrap SE -- K = 60 iid Gaussian values: the mean SE within 20 % of 1.2533 sigma / sqrt(K), bit-identical for one seed; W7 [legacy (T1)] semivariogram -- 400 sweeps at a known law: the pair count is the layout's and V_hat equals `noise_law.semivariogram(law, dt, m, m, 4 P)` within 3.5 SE; W8 enlarged window -- m = n_w - pre_n - round(t_SS f_s / 1e3), the first sample at t >= t_SS, ending 0.5 ms before the next onset; W9 LS -- every sweep gives a 13250-sample pre-window (265 ms at 50 kHz), hyp and dep read, the 0 pA sweep skipped, segments of 2500 and 6625 samples; with the 1.5 s tail each end window holds 25000 samples, starts at n_samples - 25000 and after the step's end ('ok'); with the 0.5 s tail it starts at the step's end ('tail_short'); W10 ASCII/LF; W11 SS tilt (TL1, the reading of D-115.2: `ss_noise_windows`' `tilts`, on the kept windows at the 80 ms floor) -- equals the within-polarity (Frisch-Waugh) closed form sum (t - t_bar_p)(y - ybar_p) / sum (t - t_bar_p)^2 (1e-12 mV/s), the same on the floor windows after each group's median subtraction (1e-12: (TL1) does not see a group constant); on the tilted cell |b_hat - b| < 4 x the exact SD of (TL1) under the law at the fixture's layout and the floor (from `noise_law.var_window_mean` and `window_mean_cov`), each sweep; a sweep with one polarity takes one intercept; a sweep of 2 windows gives NaN with `ss_tilt_none`; W12 LS tilt (TL2) -- equals `ls_baseline_qc.baseline_record`'s `delta_end_mV` / (centre of its end window - centre of its pre-window) on every LS sweep (1e-12 mV/s), a sweep ending in 5000 non-finite samples included (the end window ends at its last finite sample; skipped without pandas); on the tilted cell |b_hat - b| < 4 x the exact SD sqrt(V_pre + V_end - 2 C) / Delta_t_c; with the 0.5 s tail every LS tilt is NaN with `ls_tilt_none:<sweep>:tail_short`, as the oracle's flag; W13 small groups (D-106) -- a fixture whose sweep 93 keeps 3 hyp windows: flag `ss_group_dropped:93:hyp:few_windows`; the kept `SSWindows`, X_tilde, the statistic vector's K, the per-sweep vector of sweep 93 and its tilt reading (`tilts`) exclude the 3 windows; sigma_SS_hat is the median over the kept windows; sweep 93's tilt uses its dep windows only; W14 [legacy] `transient_start` (noise-free time exact -- the pre-pulse noise zero-mean per window, so r_odd is the response exactly -- floored at 80 ms, `ss_enlarged_none` at 190 ms) and `subtract_mean_response` (zero mean per polarity, 1e-12) behave as at commit 1aabe3e; W15 the offsets (D-115) -- on a noise-free cell without response with a planted tilt per sweep: `read_ss_tilts` returns each tilt to 1e-9 relative and `ss_noise_windows`' X_tilde is 0 (1e-12), where the plain subtraction of the uncorrected windows leaves the offsets (above 0.01 mV); on the noisy tilted cell: `tilts` = `read_ss_tilts` on the kept windows exactly, the offsets = b_hat_s (t_k - t_bar_s) from an independent loop (1e-15), X_tilde = an independent re-implementation (offsets removed, cut at j0, numpy's per-group median subtracted; 1e-12), and the start = `decay_start` on an `SSWindows` holding the corrected windows (t_SS and every group's fit equal); `tilt_offsets` gives 0 for a NaN tilt.
- Data flow: consumes the archive (section 3) and `noise_law` (W7, W11, W12 only); produces the kept windows, X_tilde, t_SS(c) with its flags and per-group fits (A_hat, tau_hat, B_hat, t_{s,p}: noise-table columns `noise_ss_enl_t0_ms` and per group), q_hat(c), q_hat(c, s), the per-window matrices, SE_hat (W_0 of Block 4), the per-sweep tilts b_hat(c, s) of both protocols (the per-sweep table's `noise_tilt_mV_per_s`; Block 4's simulator, Block 5's twins), and the layout (positions, period, train length, K per sweep and polarity of the kept windows) that Block 4's simulator reproduces; consumed by Block 4, Block 5 and Block 6's driver.
- Confirmed: D-051, D-072, D-073, D-076, D-095, D-102, D-103, D-104, D-106, D-107 (user). The assistant's choices, flagged: every LS sweep's pre-window enters (hyp and dep); the reading of D-110's "fitted precisely" as the exact least-squares optimum by variable projection, its bounds and grid (2.2 (d)); a minimum at a bound as a failed fit (D-110.2); sigma_SS_hat over the kept windows (the reading of D-106.1's "no per-cell statistic"); the OLS SE of (TL1) as a report; t_off of (TL2) as the first sample after the last with |i| above half its maximum; the legacy functions kept.
- Status: smoke-tested 2026-10-23 (W1-W10 10/10; mutation-tested with eight planted defects). Rewritten for D-095, D-102 to D-107 and smoke-tested 2026-10-29 (sandbox: W1-W14 14/14, about 20 s; W4's monolith oracle skipped, no NEURON here; W12's `ls_baseline_qc` oracle run; mutation-tested with thirteen planted defects -- the median replaced by the mean, groups by polarity only, tau dropped from (ST1), the fallback never applied, the floor not applied, small groups kept, sigma_SS_hat over all windows, the fit from t = 0, one intercept in (TL1), the end window's start for its centre, the end window from the last sample instead of the last finite one, `>=` for `>` in the end window's test, a failed fit counted as valid -- each FAILs at least one check; Blocks 2 and 4 suites unchanged, 10/10 each). Finding (dev check `dev/2026-10-29_start_noise/check_start_noise.py`, the decision log): on noisy cells the maximum over groups put t_star past 150 ms, hence the 80 ms fallback, in 47-96 % of simulated cells; answered by D-111 (the median). D-110 (the exact fit) and D-111 (the median) implemented and smoke-tested 2026-10-29 (later, 5) (W1-W14 14/14; W2 (g): stationarity 1.5e-14, the derivative formula against finite differences 2.8e-10, curve_fit from the solution moves nothing, no dense-grid point lower; six planted defects -- the median replaced by the maximum, no refinement, no polish, no bound check, an uncentred regression, a wrong derivative -- each FAIL W2). Not run here: the real archive (cluster). D-115 (the tilt offsets removed before the median; the SS reading at the 80 ms floor) implemented and smoke-tested 2026-10-30 (later) (W1-W15 15/15; W11 rewritten for the floor reading with its floor index written out, W13 reads `tilts`, W15 new; six planted defects -- no offsets in the subtraction, the start ignoring the reading, the reading from 5 ms, the offsets' sign, the offsets not centred, one intercept in (TL1) -- each FAILs W11 or W15; Blocks 2 and 4 suites 10/10, 11/11).

### Block 4: the per-cell fit -- binding function, simulated covariance and the iteration (D-033, D-048 to D-051, D-056; the layout and the simulator D-103, D-106, D-107.2, D-115.3)

[2026-11-17, D-152, D-153, D-154: the coordinates, bounds, steps, starts and dof are 2.6 (d), the interfaces 2.6 (h). `smoke_noise_fit.py` is re-aimed: F6's self-consistency and F7's reduced fit on the five coordinates (F7's tolerance on |u_hat - u*| restated by the implementer from the fixture's own Monte Carlo spread and reported), F9's dof = d_eff - 5, every count from `noise_cell_stats`; a new check that `NoiseFitConfig()` carries the bounds and steps of 2.6 (d). Status: smoke-tested 2026-11-17 (sandbox: `smoke_noise_fit.py` F1-F12 12/12 in 106 s; six planted defects -- sigma_m and tau_m swapped in `u_to_law`, the old dof, the mid draws not passed, the closed-form start keeping the last point, tau_s / 2 not clipped, a wrong law-dict key -- each caught; commit `bdd8011`). As built [orchestrator, accepted at the gate]: the 4 initial points are the module constant `CF_INITIAL_POINTS` (also the fallback law); new names `N_PARAMS`, `U_NAMES`, `LOG_4E5`, `LOG_1E3`; `u_to_law` refuses a shape other than (5,). F7's reduced fit (M = 60, R 40/60) recovered (0.0354 mV, 0.0157 mV, 0.275 ms, 0.0490 mV, 23.7 ms) for the truth (0.035, 0.017, 0.3 ms, 0.055, 40 ms), J 22.3 on 16 dof; its tolerance |u_hat - u*| <= (0.03, 0.21, 0.91, 0.64, 1.43) is the mean + 4 SD of 34 of 40 fixture seeds fitted at the reduced sizes. Finding, to watch on the pilot: in the other 6 of 40 seeds the slow part ran along its ridge (log tau_s +4.1 to +5.5, three at the 10 s bound); at these sizes the loose coordinates are the slow part's, not the middle's (SD 0.045 in log sigma_m, 0.20 in log tau_m). Not run here: a fit at the full M and R.]

- Module: `noise_fit.py` (model and analysis layer: the simulator of a cell's windows at the real layout, the binding function, Sigma_hat, the inner solve and the iteration; no I/O beyond an optional log callable, no plotting); smoke `smoke_noise_fit.py`. The per-sweep stage of D-076 (i) is Block 4b (`noise_sweep_fit.py`), specified when written [2026-10-30: specified as Block 9, section 2.5; drafted].
- Public API: `NoiseFitConfig` (frozen dataclass: `M_binding=3000`, `R_iter=600`, `R_final=3000`, `B_boot=2000`, `k_max=6`, `tol_u=0.01`, `starts_after_first=1`, `max_nfev=15`, `ftol=1e-4`, `xtol=1e-3`, `gtol=1e-8`, `fd_rel_step=(1e-3, 1e-2, 1e-3, 1e-3)`, `log_sigma_bounds=(log 1e-4, log 10)`, `rho_bounds=(1e-3, 0.999)`, `log_tau_bounds=(log 1e-4, log 10)`, `chunk_windows=500`, `seed=20261023`, `window_cfg`; `lag_periods` retired with (T1)); `CellLayout` (frozen: `specimen_id`, `fs`, `dt`, `n_w`, `pre_n`, `j0`, `j_floor` (the floor's first sample, pre_n + round(t_ss_floor_ms f_s / 1e3) <= j0; D-115.3), `period_samples`, `train_length`, `sweep_numbers` (n_sweeps,) the SS sweeps with kept windows, `sweep_index` (K,), `position` (K,), `polarity_is_dep` (K,), `sweep_number` (K,), `n_ls`, `ls_sweep_number` (K_LS,), `ss_tilt` (n_sweeps,) and `ls_tilt` (K_LS,) the real sweeps' tilts in mV/s, aligned with `sweep_numbers` and `ls_sweep_number` (D-107.2); `K`, `K_LS`, `m`, `n_sweeps`, `L_s` properties; `from_windows(ss_kept: SSWindows, ls: LSWindows, t_ss_ms, ss_tilts: SweepTilts, ls_tilts: SweepTilts, window_cfg=WindowConfig()) -> CellLayout` (refuses, with the reason: LS pre-windows of unequal length; a sweep of `ss_kept` or `ls` without a tilt, or with a NaN one, quoting the reading's flag); `from_cell(nwin: SSNoiseWindows, ls: LSWindows, window_cfg=WindowConfig()) -> CellLayout` (the SS tilts are Block 3's reading `nwin.tilts` (D-115.2; never re-read on X_tilde, where (OF1) has removed them; `read_ss_tilts(nwin.ss)` when `tilts` is None), the LS tilts `ls_sweep_tilts(ls)`, then `from_windows(nwin.ss, ls, nwin.start.t_ss_ms, ...)`)); `CellTarget` (`q_hat` (15,), `per_window_ss` (K, 5), `per_window_ls` (K_LS, 10), `labels`; `from_stats(ss_stats, ls_stats, layout, window_cfg)`, `from_cell(nwin, ls, layout, window_cfg)` (the statistics of the kept, subtracted windows and of the LS pre-windows); V_hat and N_pairs retired with (T1)); `simulate_cell_windows(law, layout, rng, *, parts=('ss', 'ls'), window_cfg, return_windows=False, return_tilts=False) -> (ss_stats (K, 5), ls_stats (K_LS, 10))`, with `return_windows` also `X_tilde` (K, m) and `ls_traces` (K_LS, n_ls), with `return_tilts` last the replicate's own SS reading (dict sweep -> mV/s) (one replicate cell: the sweeps continuous, sweep s with tilt `layout.ss_tilt[s]`, the windows cut at the kept positions from the floor; the replicate's own (TL1) at the floor (`tl1_fit`) and offsets (OF1) (`tilt_offsets`) removed, D-115.3; the enlarged parts; the group median window of the replicate's own corrected windows subtracted; LS window k with tilt `layout.ls_tilt[k]`); `binding_function(law, layout, cfg, seed, *, which='both', return_pool=False) -> q_sim (15,)` (CRN by re-seeding; with `return_pool` also the pools (M_cells K, 5) and (N_LS, 10), LS row i with the tilt of LS sweep i mod K_LS); `hd_median(A (N, d)) -> (d,)` (Harrell-Davis, NaN-ignoring per column); `simulated_covariance(law, layout, R, rng, *, per_sweep=False, window_cfg) -> SigmaResult` (`Sigma` (15, 15), `vectors` (R, 15), `per_sweep` dict sweep -> (R, 5 or 10) when asked); `closed_form_vector(law, layout, window_cfg) -> (15,)`; `solve_inner(q_hat, C, law_starts, layout, cfg, seed, mask, residual=None) -> InnerResult` (`law`, `u`, `J`, `nfev`, `start_index`, `status`, `message`, `starts` (u_0, J) per start, `at_bound`); `closed_form_start(q_hat, SE, mask, layout, cfg) -> NoiseLaw`; `law_to_u`, `u_to_law`; `correlation_condition_number(Sigma)`; `fit_cell(target, layout, cfg, log=None) -> CellFitResult` (`specimen_id`, `law`, `u`, `J` (at the reported theta_hat), `dof`, `p_value`, `kappa`, `Sigma` (d_eff, d_eff), `labels_kept`, `stats_dropped`, `q_hat`, `q_sim`, `SE` (W_0's; NaN where dropped), `ss_tilt` and `ls_tilt` (dict sweep -> mV/s: the tilts the simulator carried), `n_solves`, `converged`, `flags` ('stats_dropped:<labels>', 'se_from_simulation:<labels>' when a finite statistic had no bootstrap SE -- a single window -- and W_0 took the simulated single-window spread x 1.2533 / sqrt(K) at the closed-form start, 'no_plateau', 'sigma_jitter', 'at_bound:<coordinates>'), `history` (per solve: weight, R, u, J, nfev, start_index, status, at_bound, seconds), `per_sweep_cov` dict sweep -> covariance of its vector (from the final Sigma_hat run, Block 4b's weights), `config`, `seconds`; `to_dict()` JSON-ready, NaN as null). Retired with (T1) (D-107.2): `tilt_from_semivariogram`, `TiltResult`, the result's `sigma_b` and the flags 'tilt_clipped', 'tilt_no_pairs'.
- Inputs: Block 3's `SSNoiseWindows` (the kept windows, X_tilde, t_SS(c)), `LSWindows`, the per-sweep tilts; theta in mV, --, mV, s; tilts in mV/s.
- Outputs: theta_hat as `noise_law.NoiseLaw`; u in the fit coordinates; J dimensionless; Sigma in the statistics' units squared.
- Parameters: the `NoiseFitConfig` fields above; defaults are D-048 to D-051 and D-056 (M, R, B, k_max, tol, three starts) and the assistant's choices of 2.3 (bounds, steps, tolerances, warm starts, chunking).
- Library calls relied on: `scipy.optimize.least_squares(method='trf', bounds, diff_step, ftol, xtol, gtol, max_nfev)`; `scipy.linalg.cholesky`, `scipy.linalg.solve_triangular`; `scipy.stats.mstats.hdquantiles`; `scipy.stats.chi2.sf`; `numpy.cov`, `numpy.linalg.cond`; `numpy.random.SeedSequence`; Blocks 1-3 for the statistics, the law, the subtraction and the tilts.
- Custom code (marked `# Custom:`): the layout-true cutting of the enlarged windows from a continuous simulated sweep (position k -> samples [k P + j0, k P + n_w)); the cycling of the LS tilts through the binding pool; the iteration of 2.3 (no library implements indirect inference with a simulated-covariance weight).
- Test oracles (`smoke_noise_fit.py`, F1-F11, reduced sizes; the cell: Block 3's writer-like fixture with a planted tilt (sigma_b = 0.1 mV/s, realised per sweep) and a 1.5 s LS tail, read through Block 3's pipeline -- t_SS(c) of D-111, the kept windows, the median subtraction, the tilts of (TL1) and (TL2) -- K = 63, 5 LS windows of 13250 samples): F1 cutting -- `simulate_cell_windows`' X_tilde equals an independent loop (1e-12 mV): `simulate_sweep` with each sweep's tilt, the windows cut at the kept positions from the floor, each sweep's slope by the within-polarity closed form on the floor windows' means and its offsets removed (D-115.3), the enlarged parts, each group's per-sample `numpy.median` subtracted; its LS statistics equal `ls_window_stats` of `simulate_traces` with the LS tilts on the same draws; shapes and finiteness; F2 subtraction -- at a white law (rho_f = 1e-3, sigma_s = 0), both with every tilt 0 and with the fixture's tilts scaled to 0.1 mV/s in size (D-115: the offsets removed, the tilt leaves the expectation alone), the pooled mean of sigma_hat^2 over the SS windows of 40 replicate cells equals the mean over windows of sigma_f^2 c(K_g) within 3.5 SE, c(K) = E[(x_1 - med(x_1..x_K))^2] / sigma^2 for K iid normal values by an independent Monte Carlo (numpy, 200000 draws per K; also against the closed form 1 + Var(med) / sigma^2 - 2 / K, reasoning: cov(x_bar, med) = Var(x_bar) for the efficient mean), and differs from the mean subtraction's sigma_f^2 (1 - 1 / K_g) by more than 3.5 SE; F3 CRN -- `binding_function` is bit-identical twice at one theta and seed, differs at another seed, and moves by less than 0.1 times its Monte Carlo SE for a 1e-6 relative change of theta; `simulated_covariance` is bit-identical on one seed; F4 Sigma_hat -- at the white law with zero tilts, R = 120: the variance of the SS SD statistic within 40 % of (pi / 2) sigma_f^2 c_bar / (2 (m - 1) K) (c_bar the windows' mean c(K_g)), the LS single-window variance of sigma_hat within 40 % of sigma_f^2 / (2 (n_LS - 1)); Sigma_hat symmetric positive definite, `cond` finite; the per-sweep vectors carry every SS and LS sweep with shapes (R, 5) and (R, 10); F5 the real tilts in the simulator (D-107.2) -- on the same draws, the LS traces with the cell's tilts minus those with zero tilts equal b_k (t_j - t_bar) to 1e-12 mV; at a noise-free law (sigma_f = sigma_s = 0) the binding function's LS pool row i has the SD of the pure ramp of LS sweep i mod K_LS, |b| dt sqrt(n (n + 1) / 12), to 1e-12 relative, every sweep once per replicate set; over 100 replicate cells the mean of each replicate's own SS reading (`return_tilts`, (TL1) at the floor, D-115.3) lies within 3.5 SE of each sweep's planted tilt; F6 self-consistency of the inner solve -- q_hat := `binding_function(theta*, seed)` exactly, C = diag of the simulated single-window spread / sqrt(K), starts at tau_s / 2 and 2 tau_s: `solve_inner` returns u within 0.01 of u(theta*) with J < 1e-3 (M = 60); F7 a reduced fit -- the fixture cell fitted with M = 60, R_iter = 40, R_final = 60, B = 400, k_max = 2, max_nfev = 6, one start after the first: `fit_cell` returns a `CellFitResult` with every field, |u_hat - u*| <= (0.04, 0.03, 0.4, 1.0), q_sim equal, bit for bit, to `binding_function` at the reported theta_hat on the cell's binding seed, J equal to r^T Sigma_hat^-1 r there within 1e-8, history length = n_solves with the first solve 'W0' and the last 'final', dof 11, kappa finite, 7 per-sweep covariances, the result's tilts equal to the layout's, no tilt flag, `to_dict` round-trips through JSON; F8 `hd_median` equals `hdquantiles(prob=0.5)` (1e-12), ignores NaN per column, lies within half a standard error of the median of the sample median on symmetric data, and returns a single value as is; F9 masks -- a q_hat with two NaN entries, fitted with k_max = 0 and max_nfev = 1, gives d_eff = 13, those labels in `stats_dropped` and the flag, Sigma (13, 13), SE NaN there, dof 9, two solves and 'no_plateau'; F10 ASCII/LF; F11 the layout (D-106, D-107.2) -- on Block 3's small-group cell (sweep 93 keeps 3 hyp windows) the layout holds the kept windows only (K - 3, no position of the dropped group) and the simulator returns that many rows; on the 0.5 s-tail cell (every LS tilt NaN, 'tail_short') `from_cell` refuses with the reason; a tilt missing for a sweep is refused; the layout's tilts are aligned with its sweep numbers; `from_cell`'s SS tilts are Block 3's `nwin.tilts` exactly (D-115.2) and `j_floor` is the floor's sample.
- Data flow: consumes Blocks 1-3; produces theta_hat(c), Sigma_hat (final) with the per-sweep covariances for Block 4b, the fit report fields for the noise table and manifest (Block 6: columns `noise_sigma_f_mV`, `noise_rho_f`, `noise_sigma_s_mV`, `noise_tau_s_ms`, `noise_J`, `noise_dof`, `noise_p`, `noise_kappa`, `noise_n_solves`, `noise_flags`; `noise_sigma_b_mV_per_s` retired, D-107; the per-sweep tilts are Block 3's, `noise_tilt_mV_per_s` in the per-sweep table); consumed by Block 4b and Block 6.
- Confirmed: D-033, D-048 to D-051, D-056, D-103, D-106, D-107, D-114.2 (a cell with an unreadable tilt refused; confirmed 2026-10-30) (user). The assistant's choices, flagged: items (c)-(e), (h) of 2.3; the LS tilts cycled through the binding pool by replicate sets; the unit tilt draw kept in the draw order; the chi-square tail probability as a report field only.
- Status: smoke-tested 2026-10-23 (F1-F10 10/10; mutation-tested with ten planted defects). Rewritten for D-103, D-106, D-107.2 and smoke-tested 2026-10-29 (later, 5) (sandbox: F1-F11 11/11, about 100 s; Blocks 2-3 suites 10/10, 14/14; mutation-tested with seven planted defects -- the mean subtraction in the simulator, the SS tilts dropped, the LS tilts dropped, the LS tilts not cycled, a NaN tilt accepted, the SS tilts misaligned, q_sim reported at the start law -- each FAILs the check named for it, the last once F7 recomputes q_sim at the reported law). Found by the smoke: the fixture cell's start falls back to 80 ms (its sweep 92 carries a -0.25 mV/s tilt, which bends two of its four groups' fits; the median is 161 ms); see the dev check `dev/2026-10-29_start_noise/check_start_detrended.py`. Not run here: a fit at the full M and R (hours; the cluster, Block 6's S0). D-115.3 (each replicate's own reading at the floor and its offsets) implemented and smoke-tested 2026-10-30 (later) (F1-F11 11/11, about 2 min; F1 against an independent loop with the closed-form slope (4.4e-16 mV), F2 also with 0.1 mV/s tilts (the pooled mean 0.0024134 against 0.0024151 expected), F5 the replicate's own reading, F11 `from_cell`'s tilts and `j_floor`; three planted defects -- `from_cell` re-reading on X_tilde, the simulator without the offsets, its offsets from the layout's tilts -- each FAILs F1, F2 or F11). With the offsets removed the fixture cell's median start is 151.7 ms (161.0 ms without): it still falls back, now by 1.7 ms.

### Block 9: the per-sweep stage ("Block 4b") -- each sweep's law, its bootstrap variance and the pooling per protocol (D-073, D-076 (i), D-115, D-116)

[2026-11-17, D-153.1, D-154.2, D-154.3: the coordinates, the two bootstrap variances, the two poolings, the degenerate rule and the columns are 2.6 (e), (g), (h). `smoke_noise_sweep_fit.py` is re-aimed: P4/P5 on (log sigma_f, log sigma_m, log sigma_s); P6 the two variances from one set of refits; P3 unchanged (one pooling function, used twice); P8 also a planted sigma_f below half the cell's -> `degenerate:sigma_f`, not bootstrapped, twin = theta_hat(c); P9 the pools as protocol -> coordinate -> PoolResult and the columns of 2.6 (g); P11 also `min_sigma_f_ratio` = 0.5. Status: smoke-tested 2026-11-17 (sandbox: `smoke_noise_sweep_fit.py` P1-P11 11/11 in 388 s; six planted defects -- tau_m not held, v_m from the slow coordinate, the degenerate rule removed, the middle pooling given the slow one's tau2, the wrong three steps, `cell_columns` reordered -- each caught; commit `1b07fb1`). As built [orchestrator, accepted at the gate]: `degenerate:sigma_f` is tested on the sweep's own fit only (in `run_sweep`), with a strict '<'; a bootstrap refit is never failed by it; a refit counts when both its log sigma_m and log sigma_s are finite; new names `U_NAMES`, `COORDS`; the row's keys add `sigma_m_mV`, `tau_m_s`, `log_sigma_m`, `v_log_sigma_m`, `bootstrap_y_m`, `lambda_m`, `sigma_m_tilde_mV`. Measured on the fixture: P5's worst z 1.53 over the three coordinates; P7's bootstrap over linearised variance 1.02 (log sigma_m) and 0.90 (log sigma_s). Not run here: the stage at the full sizes. Amendments 4-5 (2026-11-17, D-155 [orchestrator], after the final check's F1): a bootstrap variance that is NaN or below `V_MIN` = 1e-12 fails the sweep with `v_none`, in `bootstrap_v`, `run_sweep` and `pool_cell` (rows made elsewhere); P6 covers a variance of exactly 0 and one of roundoff size (8.8e-31); P1-P11 11/11 on the merged code (494 s), `smoke_noise_run.py` 9/9 (478 s); commits `3aea91c`, `e9a6469`. Final check 2026-11-17 (an independent Opus agent, `specs/test_reports/2026-11-17_1700_incremental.md`): PASS WITH ISSUES, no blocker or major finding; 54 spec tests of its own pass.]

Numbered 9, not 4b, because the roadmap tool and the tester's `changed_blocks.py` read integer block numbers only and 5-8 are held by the roadmap's later stages (2026-10-30, the assistant's choice); "Block 4b" in the decision log (D-082 to D-116) and the handoffs is this block. It runs after Block 4 and before Block 5.

- Module: `noise_sweep_fit.py` (model and analysis layer: the per-sweep fit, its bootstrap and the pooling; no I/O beyond an optional log callable, no plotting); smoke `smoke_noise_sweep_fit.py`.
- Public API (drafted; the open points of section 8 change defaults, not the interface): `SweepStageConfig` (frozen dataclass: `B_v=100` [open; proposal 50 since 2026-11-02, section 8] [corrected 2026-11-11, D-122.1: `B_v=50`], `M_refit=None` (the refits' binding size; None takes the main fit's `M_binding`) [open; proposal 1000] [corrected 2026-11-11, D-122.1: `M_refit=1000`; None still means the main fit's `M_binding`], `max_failed_refit_frac=0.1` [open] [decided 2026-11-11, D-122.2], `min_stats=4`, `seed=20261030`); `sweep_layout(layout: CellLayout, protocol: str, sweep: int) -> CellLayout` (the cell's layout restricted to one sweep: 'ss' -- its kept windows (positions, polarities, sweep numbers) and its tilt, no LS window; 'ls' -- its one pre-window and its tilt, no SS window; every other field unchanged); `sweep_target(target: CellTarget, layout: CellLayout, protocol, sweep) -> (q_hat_s (15,), mask (15,) bool)` (2.5 (a); NaN and False outside the sweep's protocol and where a statistic is not finite); `sweep_seeds(seed, specimen_id, protocol, sweep) -> (seed_binding_s, seed_v_s)` (`SeedSequence([seed, specimen_id, 0 for SS or 1 for LS, sweep]).spawn(2)`); `SweepFit` (`protocol`, `sweep`, `u` (3,), `law` (`NoiseLaw`, tau_s = tau_s_hat(c)), `J`, `d`, `dof`, `n_calls`, `status`, `at_bound`, `failed`, `reason`); `fit_sweep(q_hat_s, mask, Sigma_s, sub: CellLayout, law_start: NoiseLaw, cfg: NoiseFitConfig, seed_binding_s: int) -> SweepFit` ((P1), the failure rule of 2.5 (c)); `SweepBootstrap` (`v`, `B`, `B_ok`, `y` (B,) NaN where a refit failed, `flags`); `bootstrap_v(fit, mask, Sigma_s, sub, law_start, cfg, stage_cfg, seed_binding_s, seed_v_s, refit=fit_sweep) -> SweepBootstrap` ((P2); `refit` a callable with `fit_sweep`'s signature, swappable for a test double); `PoolResult` (`protocol`, `sweeps`, `k`, `Q`, `tau2_raw`, `tau2`, `mu`, `lam` (k,), `y_shrunk` (k,)); `pool_protocol(y (k,), v (k,)) -> PoolResult` ((P3), (P4), (D-076.1), the k = 1 and k = 0 rules); `run_sweep(cell_fit: CellFitResult, target, layout, cfg, stage_cfg, protocol, sweep, log=None) -> dict` (one sweep's fit and bootstrap: the per-sweep fields of 2.5 (g); one cluster job per (cell, sweep)); `pool_cell(rows, cell_law: NoiseLaw, specimen_id) -> SweepStageResult` (the pooling per protocol over the rows, the failed sweeps' twins by 2.5 (f), theta_tilde(c, s) by (P0)); `SweepStageResult` (`specimen_id`, `rows` (list of dict), `pools` (dict protocol -> PoolResult), `twin_laws` (dict (protocol, sweep) -> `NoiseLaw`), `flags`, `config`, `seconds`; `to_dict()` JSON-ready, NaN as null; `table_rows()` the columns of 2.5 (g)); `fit_sweeps(cell_fit, target, layout, cfg, stage_cfg, log=None) -> SweepStageResult` (`run_sweep` on every sweep of S(c), then `pool_cell`). [as built 2026-11-12, `noise_sweep_fit.py`: the API above, with the assistant's additions (flagged) -- `SweepStageConfig()` defaults `B_v=50`, `M_refit=1000` (D-122.1; None still takes `M_binding`), validated (B_v >= 2, M_refit None or >= 1, the fraction in [0, 1], min_stats >= 1); `SweepFit.flags` ('sigma_jitter' when W_s's Cholesky needed Block 4's jitter); `SweepFit.n_calls` counts every call of the residual, the finite-difference Jacobian's included (least_squares' nfev does not); `pool_protocol(y, v, *, protocol='', sweeps=())` (refuses v <= 0 or a non-finite value); `run_sweep(..., log=None, *, refit=fit_sweep)` (passed to `bootstrap_v`; the smoke's double); `pool_cell(rows, cell_law, specimen_id, *, stage_cfg=None, seconds=None)`; `fit_sweeps(cell_fit, target, layout, cfg=None, stage_cfg=SweepStageConfig(), log=None)` (cfg None: the per-cell fit's own config; refuses a result of another specimen); `SweepStageResult.cell_columns()` (the per-cell columns of 2.5 (g); `table_rows()` the per-sweep ones, plus `specimen_id`, `noise_sweep_protocol` ('SS' or 'LS'), `noise_sweep_number` and the sweep's `noise_tilt_mV_per_s` of Block 4's data flow; `noise_sweep_sigma_f_mV`, `noise_sweep_rho_f`, `noise_sweep_sigma_s_mV` are the twin's theta_tilde(c, s) -- theta_hat(c)'s for a failed sweep (D-122.3) -- and `noise_sweep_sigma_s_raw_mV` the sweep's own fit, NaN for a failed sweep); a row's keys: specimen_id, protocol ('ss' or 'ls'), sweep, K_s, n_LS, tilt_mV_per_s, d, labels_kept, u, sigma_f_mV, rho_f, sigma_s_mV, tau_s_s, log_sigma_s, J, dof, n_calls, status, at_bound, failed, reason ('', 'stats', 'no_convergence', 'at_bound:<coordinates>', 'v_none'), v_log_sigma_s, B_v, B_ok, n_refits_failed, bootstrap_y, flags, seconds, and after `pool_cell` lambda, sigma_s_tilde_mV (NaN for a failed sweep), twin_law; a bootstrap replicate with fewer than min_stats finite statistics counts as a failed refit (never seen; the assistant's choice).]
- Inputs: Block 4's `CellFitResult` (theta_hat(c), `per_sweep_cov`), `CellTarget` and `CellLayout` of the cell; Block 4's `NoiseFitConfig` (M, bounds, steps, tolerances, max_nfev, seed); theta in mV, --, mV, s.
- Outputs: the per-sweep laws (`NoiseLaw`, tau_s = tau_s_hat(c)); log sigma in log mV, v in (log mV)^2; the pooling's numbers per protocol; the per-sweep and per-cell columns of 2.5 (g).
- Parameters: `SweepStageConfig` above; `NoiseFitConfig`'s `M_binding`, `log_sigma_bounds`, `rho_bounds`, `fd_rel_step[:3]`, `ftol`, `xtol`, `gtol`, `max_nfev`, `seed`.
- Library calls relied on: `scipy.optimize.least_squares(method='trf', bounds, diff_step, ftol, xtol, gtol, max_nfev)`; `scipy.linalg.solve_triangular` and Block 4's Cholesky helper; `numpy.var(ddof=1)`; `numpy.random.SeedSequence`; Block 4's `binding_function`, `simulate_cell_windows`, `law_to_u`; Block 3's `per_sweep_vectors`; `statsmodels.stats.meta_analysis.combine_effects` as the smoke's oracle of (P3)'s raw value.
- Custom code (marked `# Custom:`): the layout restriction; (P3)-(P4) with the clip and the recomputed random-effects mean (statsmodels' `combine_effects` neither clips nor recomputes; its raw tau2 is the smoke's oracle); the bootstrap loop (no library bootstraps an indirect-inference estimator); the failed-sweep rule.
- Test oracles (`smoke_noise_sweep_fit.py`, P1-P10, reduced sizes, about 5 min (estimate); the cell: Block 4's smoke fixture -- 2 SS sweeps of 30 and 33 kept windows, 5 LS pre-windows, planted tilts -- with a stub per-cell result: theta_hat(c) a law set apart from the fixture's truth theta_cf (sigma_f x 1.05, rho_f + 0.02, sigma_s x 0.8), `per_sweep_cov` from `simulated_covariance(per_sweep=True)` at R = 200): P1 restriction -- `sweep_layout` keeps exactly the sweep's kept windows, its tilt and nothing of the other protocol; `simulate_cell_windows` on it equals, on the same draws, an independent loop (`simulate_sweep` with the sweep's tilt, the windows cut from the floor, the within-polarity closed-form slope (TL1) and its offsets, numpy's per-group median subtracted; 1e-12 mV) and, for an LS sweep, `simulate_traces` with its tilt (1e-12 mV); P2 targets -- an SS target equals numpy's median over the sweep's rows of `ss_window_stats` (exactly), an LS target the window's row; the mask is False outside the protocol; P3 pooling -- on random (y, v), k = 2..12, 200 draws: tau2_raw equals `combine_effects(y, v, method_re="dl").tau2` (1e-12 relative: float64 roundoff on sums of k terms); where it is >= 0, mu equals `mean_effect_re` (1e-12); where it is < 0, tau2 = 0, mu equals `mean_effect_fe` and every lambda is 0; lambda and y_shrunk equal (D-076.1) written out in the test; y_shrunk lies between mu and y; k = 1 gives mu = y, tau2 NaN and y_shrunk = y; k = 0 all NaN; P4 self-consistency -- q_hat_s := q_sim(theta* | s) exactly on the sweep's binding seed: `fit_sweep` from a start 0.2 away in log sigma_s returns u within 0.01 of u(theta*), J < 1e-3, not failed (M = 60), for one SS and one LS sweep; P5 recovery -- every sweep of the fixture fitted at M = 100 from the stub theta_hat(c): |u_hat_s - u(theta_cf)| below 4 times the linearised SD of each coordinate (2.5 (h)'s sandwich at the same M and seed), not failed; P6 the bootstrap's machinery -- with `refit` replaced by a test double returning the log of the replicate's first statistic (the SD) as its third coordinate: `bootstrap_v`'s y equals an independent loop over the same stream (`simulate_cell_windows` on the restricted layout at theta_hat(c, s) with tau_s_hat(c) and the sweep's tilt; 1e-12), v its variance with ddof 1; a double failing every 5th refit gives B_ok = 0.8 B_v, those y NaN, v over the others and the flag `v_refits_failed` (20 % > 10 %); a double failing every refit gives v NaN and the reason `v_none`; P7 the bootstrap against the linearisation -- for the fixture's first SS sweep, B_v = 60 real refits at M = 100: v lies inside v_lin chi2_q(B_ok - 1) / (B_ok - 1) for q in {0.0005, 0.9995}, v_lin the sandwich at the same M and seed (loose: the sandwich is first-order); P8 a failed sweep -- a target whose SD statistics are 100 times below what any law in the bounds produces drives the solve to a bound: flagged `sweep_fit_failed:<pr>:<sweep>:at_bound:...`, not bootstrapped, out of S'_pr(c), its twin law by 2.5 (f); a solve with max_nfev = 1 fails with `no_convergence`; P9 the stage end to end -- `fit_sweeps` on the fixture (M = 60, B_v = 8, M_refit = 60): a row per sweep of S(c) with every field of 2.5 (g); each protocol's pool equals `pool_protocol` on the rows' y and v; theta_tilde(c, s) = (sigma_f_hat, rho_f_hat, sigma_s_tilde, tau_s_hat(c)); bit-identical twice; `run_sweep` per sweep then `pool_cell` equals `fit_sweeps`; `to_dict` round-trips through JSON; P10 ASCII/LF; P11 the decided defaults (added 2026-11-11, D-122) -- `SweepStageConfig()` has `B_v` = 50, `M_refit` = 1000 and `max_failed_refit_frac` = 0.1, so a default changed by mistake FAILs. [corrected 2026-11-12, as implemented: P3 -- tau2_raw against (P3) in exact rational arithmetic (`fractions`, the same float inputs) at 1e-13 of the formula's scale (|Q| + k - 1) / (sum w - sum w^2 / sum w), and against statsmodels' DL tau2 at 1e-10 of it, statsmodels' own value being off the exact one by up to 1.1e-12 of that scale on these draws (measured; the 1e-12 first drafted failed on the oracle's roundoff, not on (P3)); mu against statsmodels at 1e-12; P4 and P5 -- also J = r^T Sigma_s^-1 r recomputed at the solution, relative 1e-6 with no absolute floor (J is about 1e-8 in P4); P6 -- also the seeds against the spec's SeedSequence formula written out, the refit's arguments (seed_binding_s, M_refit, theta_hat(c), Sigma_s), one failed refit in 10 (exactly 10 %) not flagged, and the LS sweep 54 as well as the SS sweep 92; P8 -- the SD target 100 times ABOVE the largest SS SD a law in the bounds produces (1400 mV; about 14 mV at sigma_f = sigma_s = 10 mV), which ends at the bounds of all three coordinates in 16 calls: a target 100 times below leaves the whitened cost flat near the lower bounds and the solve stops on xtol short of them (u = (-8.3, 0.19, -7.9) at M = 30, checked 2026-11-12); also d_s = 3 fails with `stats` and no solve; P9 -- B_v = 4 (not 8), the two runs compared bit for bit being `fit_sweeps` and `run_sweep` per sweep then `pool_cell`, with the table's and the cell's columns.]
- Data flow: consumes Blocks 2-4 (the law; Block 3's per-sweep vectors through `CellTarget`; Block 4's layout, simulator, binding function, theta_hat(c) and per-sweep covariances); produces theta_tilde(c, s) for Block 5's twins and the per-sweep and per-cell columns for Block 6's tables; consumed by Blocks 5 and 6.
- Confirmed: D-073, D-076 (i), D-116.1-D-116.3 (user); D-116.4 with its reading open (section 8) [corrected 2026-11-11: D-116.4's reading confirmed and the open points decided, D-122.1-D-122.3 (user): B_v = 50, M_refit = 1000, a failed refit left out and counted, a failed sweep's twin at theta_hat(c)]. The assistant's choices, flagged: Block 4's simulator restricted to one sweep (2.5 (b)); Sigma_hat_s at the law of Block 4's final Sigma_hat run (2.5 (c)); the failure rule's tolerance (Block 4's `at_bound`); d_s >= 4; the seeds; the k = 1 and k = 0 rules; the per-cell mu columns; B_v, M_refit and the failed refits (open, section 8).
- Status: drafted 2026-10-30 (SPEC only; the code waits for the answers of section 8) [updated 2026-11-11: answered, D-122; the code is next (D-123: then Block 6's driver for the noise fit of n real cells, then Block 5)]. Cost and precision measured on the fixture with Block 4's functions (2.5 (h); `dev/2026-10-30_sweep_stage_cost/`). [2026-11-12: the code written and smoke-tested (sandbox, numpy 2.5.3, scipy 1.18.1, statsmodels 0.15.0: P1-P11 11/11 in 404 s on one core; P5's worst z 2.34, P7's v / v_lin = 1.19 inside [0.50, 1.72] (B_ok 60), P9's pools SS tau2 0 (clipped), mu -2.881, LS tau2 0.0060, mu -2.968; mutation-tested with 17 planted defects -- the SS restriction with the first sweep's tilt, with the first polarities; the SS target a mean; the seeds without the protocol; tau_s not held; J reported as the cost; no at_bound failure; the refits at the main M; the replicates at theta_hat(c); v with ddof 0; the flag at 10 % or more; tau2 not clipped; mu not recomputed; a failed sweep's twin at the pooled sigma_s; the twin with the raw sigma_s; no d_s rule; the LS sweeps left out -- each FAILs the check named for it, J's once P4 lost its absolute floor). Not run here: the stage at the full sizes (M = 3000, B_v = 50, M_refit = 1000: hours per sweep, 2.5 (h)) and on a real cell -- Block 6's run on davinci (D-123, D-124).]

### Block 6: the noise-fit run -- the driver, its PBS array and the merge (D-123, D-124, D-125)

[2026-11-17, D-152 to D-154: the tables' columns are 2.6 (g); `run_noise_fit.py` and `smoke_noise_run.py` follow. A run under the two-component law (nf1) is not resumed: the code commit differs. Status: smoke-tested 2026-11-17 (sandbox: `smoke_noise_run.py` R1-R9 9/9 in 455 s on the new columns; written by the orchestrating session; the gate script's expected lines for Blocks 2 and 4 are now 11/11 and 12/12). Not run here: the run on davinci.]

- Module: `run_noise_fit.py` (scripts layer: Blocks 3, 4 and 9 on one archive cell, the cohort listing and the merge; the only place of the noise fit that reads or writes files), `submit_noise_fit.sh` (the PBS array and the merge job); smoke `smoke_noise_run.py`.
- Command line: `python run_noise_fit.py list --group-dir DIR [--cells ID,ID]` -> one specimen ID per line, sorted: every `specimen_<id>/` of the group that holds `ss_pulses.npz`; `--cells` restricts it (the pilot, D-124.3), an ID not in the group refused. `python run_noise_fit.py cell --group-dir DIR --specimen ID --out DIR [--workers N] [--force] [--sizes full|smoke]` -> `<out>/cells/specimen_<id>.json`, written atomically (a temporary file, then `os.replace`), and one line `NOISE_CELL <id>: status=<ok|refused|failed> reason=<...> seconds=<s>`; exit 0 for every status (D-125.4: the run goes on), non-zero only for the driver's own usage or I/O errors; a cell whose JSON has status ok and the current code commit is skipped unless `--force` (D-125.5). `python run_noise_fit.py merge --out DIR --group NAME [--cells ID,ID]` -> the two tables and `NOISE_MERGE <GROUP>: <a> ok, <b> refused, <c> failed, <d> missing`, then one line per refused, failed or missing cell. [as built 2026-11-13, `run_noise_fit.py`: this command line, with the assistant's additions (flagged) -- `list` refuses an empty cohort (exit 2) and deduplicates `--cells`; `cell` treats a missing `specimen_<id>/` as a usage error (exit 2, no JSON) and a `metadata.json` naming another specimen as a failure (RuntimeError); before its NOISE_CELL line `cell` prints progress lines `[noise_fit <id>] ...` (Block 4's and 9's log callables; a worker's prefixed `[noise_fit <id> <SS|LS> <sweep>]`), and a skipped cell's line reads `status=ok reason=skipped (same code commit <12 hex> and sizes) seconds=0`; the resume rule (`resume_decision`) also requires the same `sizes` record and clean code -- a JSON written from code that differed from its commit (`code_dirty`) is re-run, and nothing is skipped while `git status` shows a noise-fit file modified -- so that a smoke-size or hand-edited result never stands in for a full-size committed one; `merge` without `--cells` takes the JSONs found as the cohort, writes its rows sorted by ID, counts an unreadable JSON or an unknown status as failed, and stops (exit 1) on Block 9 column keys that differ from its constants; a missing cell's listing line carries no reason.]
- The per-cell pipeline (D-123.1): `noise_windows.load_ss_windows(spec_dir)` -> `ss_noise_windows(ss)` -> `load_ls_prewindows(spec_dir)` -> `noise_fit.CellLayout.from_cell(nwin, ls)` -- a `ValueError` here is a refusal (D-114): status `refused`, the message as reason, no fit -> `CellTarget.from_cell(nwin, ls, layout)` -> `fit_cell(target, layout, NoiseFitConfig())` -> Block 9 on every sweep of S(c) in a `concurrent.futures.ProcessPoolExecutor` of `--workers` processes (`run_sweep` per sweep, then `pool_cell(rows, cell_fit.law, specimen_id, stage_cfg=SweepStageConfig())`, equal to `fit_sweeps`: Block 9's P9). Any other exception in Blocks 3, 4 or 9: status `failed`, the exception's type and message as reason, its traceback in the JSON. [as built: the pool's workers are forked where the platform can (`multiprocessing.get_context('fork')`): they run the modules loaded at the cell's start -- the commit its JSON records -- whatever a `git pull` changes on disk meanwhile, where a forkserver worker (Linux's default from Python 3.14) would import the files anew; the LS sweeps are submitted first (the slower, 2.5 (h)) and the rows kept in `fit_sweeps`' order; the first exception of a sweep cancels the sweeps not yet started and is raised with the worker's traceback (concurrent.futures' remote traceback, so the JSON's traceback shows the frame in `run_sweep`); the LS tilts are read once more for the summary (`ls_sweep_tilts`, which raises no ValueError: the refusal stays `from_cell`'s); `fit_cell` and the stage take the `--sizes` configuration (`full` = `NoiseFitConfig()` and `SweepStageConfig()`).]
- The cell JSON: `specimen_id`, `group`, `status`, `reason`, `traceback` (failed only), `code_commit` (the last commit that touched the noise-fit code, `git log -1 --format=%H -- noise_*.py run_noise_fit.py`, so that a docs or roadmap commit between the pilot and the run does not re-run the pilot's cells; D-125.5), `sizes` (`NoiseFitConfig` and `SweepStageConfig` as dicts), `seconds` per stage (windows, cell fit, sweep stage), `host`, `workers`, the Block 3 summary (t_SS(c), K, K_LS, the start's and the tilt readings' flags), `cell_fit` (`CellFitResult.to_dict()`, present once the cell fit ran) and `sweep_stage` (`SweepStageResult.to_dict()`, present once the stage ran). [as built: also `code_dirty` (`git status --porcelain` on the same pathspec; null without git), `job_id` (PBS_JOBID), `started_utc`, `finished_utc`, `versions` (python, numpy, scipy, statsmodels), `seconds.total` and, for a failed cell, the seconds of the stage that raised; `tables` -- Block 9's `cell_columns()` and `table_rows()`, computed when the stage ends and read by the merge (no result object rebuilt from its dict); the summary `windows` holds `t_ss_ms`, `K` (the kept SS windows), `K_LS`, `start_flags`, `ss_tilt_flags`, `ls_tilt_flags` and, added, `fs_Hz`, `t_star_ms`, `sigma_ss_mV`, `K_stored`, `start_fallback_reason`, `start_groups` (per group: sweep, polarity, K, status, t_ms), the per-sweep tilts of both protocols and `ls_end_status`; strict JSON, ASCII, NaN and inf as null.]
- The tables (D-125.2), `csv.DictWriter`, NaN and None written as empty fields: `noise_cells_<GROUP>.csv`, one row per cohort cell -- `specimen_id`, `status`, `reason`, Block 4's columns (`noise_sigma_f_mV`, `noise_rho_f`, `noise_sigma_s_mV`, `noise_tau_s_ms`, `noise_J`, `noise_dof`, `noise_p`, `noise_kappa`, `noise_n_solves`, `noise_flags`), `noise_t_ss_ms`, `noise_K`, `noise_K_LS`, Block 9's `cell_columns()`, `noise_code_commit`; `noise_sweeps_<GROUP>.csv`, one row per sweep of every cell with status ok -- Block 9's `table_rows()`. [as built: a cell row holds every column its JSON can fill whatever the status (a refused cell keeps Block 3's three, a failed one what its pipeline reached); Block 9's four columns in `cell_columns()` order without its duplicate `specimen_id`; floats written by repr (an exact round trip), LF line ends, each file replaced atomically.]
- The job script (`hpc-python-compat`, `hpc-git-delivery`): run by hand it is the submitter -- it lists the cohort (`list`), refuses a submission whose cells x `NCPUS` exceed 120 (D-125.1), and submits the array (`qsub -J 0-<n-1>`, `-l select=1:ncpus=$NCPUS`, `-q cpu`, `-l walltime=$WALLTIME`) and the merge job (`-W depend=afterany:<array id>`); run by PBS with `PBS_ARRAY_INDEX` it runs `cell` on the index-th cell with `--workers $NCPUS`. Knobs (environment): `RUN_TAG` (required), `GROUP` (L3_exc), `CELLS` (the pilot's IDs; empty for the whole group), `IH_CODE_DIR`, `IH_ARCHIVE_ROOT` (`/davinci-1/home/ldellamea/Human Neurons Fitting`), `IH_OUTPUT_BASE`, `CONDA_ENV` (prova), `NCPUS` (6), `WALLTIME` (12:00:00 until the pilot sets it, D-125.5), `DRY_RUN=1` (prints every command, submits nothing, runs no Python); `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` set to 1; the roadmap marker lines (`subproject-roadmap`, `references/hpc_marker.md`; job `noise_fit`, stage N6). Outputs under `$IH_OUTPUT_BASE/noise_fit/<RUN_TAG>/<GROUP>/`: `cells/`, the two tables, `logs/`. [as built 2026-11-13, `submit_noise_fit.sh`, the assistant's choices (flagged): the role by `NOISE_ROLE` (cell, merge; unset = the submitter, refused when `PBS_JOBID` is set); the submitter freezes every resolved path and the cohort in a run file `$NOISE_RUN_DIR/<RUN_TAG>_<GROUP>_<UTC>.env` (default `NOISE_RUN_DIR=$HOME/noise_fit_runs`, refused with a space or a comma; a copy in `logs/`) and passes the jobs only its path (`-v NOISE_ROLE=...,NOISE_RUN_FILE=...`), because the archive and code paths hold spaces that `qsub -v` does not carry, and so that a job runs exactly the cohort submitted; a one-cell cohort goes as a plain job with `NOISE_INDEX=0` (PBS refuses an array of one); `DRY_RUN=1` runs `run_noise_fit.py list` and checks the cell and merge argument lists against `run_noise_fit.py`'s own parser -- read-only Python, so 'runs no Python' above is read as 'runs no fit, no merge, no qsub' -- prints the run file, the two qsubs and every command, and writes nothing; added knobs `SIZES` (full; smoke for a quick check of the PBS path, whose JSONs the resume rule never lets stand in for full-size ones), `NOISE_RUN_DIR` and `ROADMAP_DISABLE` (passed to the jobs); the merge job 1 core, 01:00:00; `-k eo` (PBS's own logs in $HOME) and each job's output teed to `logs/cell_<id>_<job>.log` or `logs/merge_<job>.log`; exit codes 2 usage, 3 refused (archive, cohort, core cap), 4 qsub, 5 conda; the conda block of `submit_ih_fit.sh` with `set -u` relaxed across it (hpc-git-delivery); the driver also sets the three BLAS variables to 1 by default before numpy is imported.]
- Inputs: the archive group, `<IH_ARCHIVE_ROOT>/<GROUP>/specimen_<id>/` with `ss_pulses.npz` (Format A), `trace_qc_results.csv`, `metadata.json`, `ls_sweeps.npz` (Block 3's contracts). Outputs: as above; units as Blocks 4 and 9 state them.
- Parameters: Blocks 4 and 9's defaults at full size (D-125.5); `--sizes smoke` swaps in the smoke's reduced sizes, recorded in the JSON's `sizes`; `--workers` 6 (D-125.1, the assistant's reading).
- Library calls relied on: `argparse`, `json`, `csv`, `concurrent.futures.ProcessPoolExecutor`, `subprocess` (the code commit), `os.replace`, `traceback`; Blocks 3, 4 and 9.
- Custom code (marked): the driver itself (no library runs this pipeline); the resume rule.
- Test oracles (`smoke_noise_run.py`, reduced sizes, sandbox, Block 3's fixture writer): R1 one writer-like cell through `cell --sizes smoke` -> a JSON with every field and status ok, its `cell_fit` equal to `fit_cell` run directly at the same sizes and its `sweep_stage` equal to `fit_sweeps` (both compared through `to_dict()` without the seconds), with 1 and with 2 workers; R2 a refused cell (Block 3's 0.5 s-tail fixture: no LS tilt can be read) -> status refused with the reason, exit 0, no fit run; R3 a failed cell (a planted exception) -> status failed with its type, message and traceback, exit 0; R4 resume -> a second `cell` skips the cell, `--force` re-runs it, a JSON of another commit is re-run; R5 merge over an ok, a refused, a failed and a missing cell -> the two tables' columns, one cell row per cohort cell with its status, sweep rows only for the ok cell and equal to its `table_rows()`, the summary line and the listed cells; R6 `list` -> the cohort sorted, `--cells` restricts it, an unknown ID refused; R7 the job script -> `bash -n`; `DRY_RUN=1` prints one `cell` command per cohort cell and the merge's, submits nothing and runs no Python; 21 cells x 6 cores refused; the marker lines present; R8 ASCII/LF of the three files. [as built 2026-11-13: R1-R8 as listed; R1 also: the pool takes `--workers` processes, forked; R3 in three cases -- a LayoutError of Block 3 (a QC table one row short, in a subprocess), a RuntimeError planted in Block 4 and a ValueError raised in a Block 9 worker (in process); R4 also: other sizes, modified code (re-run every time, the JSON marked), a docs commit (no re-run), a code commit (re-run, the JSON of the new commit), a refused JSON (never skipped), and `resume_decision` on its own; R5 also: NaN and None as empty fields and NaN as null, the merge without `--cells`; R7 also: a stubbed submission (the array then the merge after it), a one-cell plain job, the cell and merge roles from a spool copy with PBS's `OMP_NUM_THREADS=6` overridden, a failing python failing the job, the refusals, and the roadmap marker pushed to a local origin (done, then failed); R9 added: the decided defaults (D-122.1, D-125.1, D-125.5, D-126). The code runs from a temporary git repository holding copies of the noise-fit modules, the driver, the script and the roadmap helper, with a local bare origin; `conda`, `qsub` and `python` are stubs on PATH for the script's tests.]
- Data flow: consumes the archive and Blocks 3, 4 and 9; produces the per-cell JSONs and the two tables for Block 5 (the twins) and Block 7 (the record).
- Confirmed: D-123, D-124, D-125.1 (layout (a), at most 120 cores), D-125.2, D-125.4, D-125.5 (user). The assistant's choices, flagged: 6 cores per cell with Block 9's sweeps in a process pool (D-125.1's reading) [confirmed 2026-11-12, D-126]; the command line, the JSON's fields and the status names; the resume rule by code commit; the submitter/job split of the script; the default walltime; the oracles. [2026-11-13: the additions of the as-built notes above are the assistant's choices, flagged, not confirmed.]
- Status: drafted 2026-11-12 (SPEC only; the code next). Not run here: the pilot and the run (davinci). [2026-11-13: the code written and smoke-tested (sandbox; Python 3.13.18, numpy 2.5.3, scipy 1.18.1, statsmodels 0.15.0; 2 cores): R1-R9 9/9 in 508 s, and 9/9 on the final code in 675 s with two other runs sharing the cores; mutation-tested with 26 planted defects -- the rows in submission order, the pool not forked, a refusal recorded as failed, a failed cell exiting 1, no traceback, the resume rule ignoring the commit, the modified code or the sizes or skipping a refused JSON, the code commit taken from HEAD, the merge counting refused as failed, writing tau_s in s, keeping the --cells order, keeping NaN in the JSON or writing nan in the CSV, the cap refusing 120 cores, a dry run that submits, no BLAS exports, the array 0-N, --workers 1 in the cell role, afterok, the marker of N5, the cohort sorted as text, an unknown --cells ID dropped, --workers 1 by default, B_v = 100 at full size -- each FAILs the check named for it (25 run before the last edit, which forked the pool and guarded R3 and R5 against crashing; the no-traceback and NaN-kept defects re-run after it as clean FAILs, and the unforked pool added). Not run here: the job script under PBS, the pilot and the 18-cell run (davinci).]

### Block 5: the twins' table -- each twin sweep's law, tilt and protocol (D-073, D-076 (iii), D-122.3, D-182.3, D-186)

- Module: `twin_table.py` (loading layer: the only twin module that reads the real archive and the noise-fit tables; no NEURON, no plotting); smoke `smoke_twin_table.py`. Parameters in `twin_config.TwinConfig`.
- Public API:
  - `class TwinTableError(RuntimeError)`.
  - `CellProtocol` (frozen dataclass): `specimen_id: int`; `ss_fs_Hz: float`; `ss_t_s` float64 (n_w,) s, 0 at the pulse onset; `ss_pre_n: int`; `ss_period_samples: int`; `ss_train_length: int`; per stored window, in the archive's order, (K,): `ss_sweep_number` int64, `ss_position` int64 (0-based), `ss_polarity_is_dep` bool, `ss_peak_pA` float64 (signed), `ss_stim_duration_s` float64; `ls_fs_Hz: float`; per stored LS sweep, in the order of `metadata.json['ls_sweeps']`, (K_LS,): `ls_index` int64 (the `<index>` of `v_<index>`), `ls_sweep_number` int64, `ls_amplitude_pA` float64 (signed, `detected_amplitude_pA`), `ls_onset_s` float64, `ls_duration_s` float64 (= step end - onset), `ls_n_samples` int64.
  - `load_cell_protocol(spec_dir, cfg: noise_windows.WindowConfig = WindowConfig()) -> CellProtocol`.
  - `read_noise_tables(cells_csv, sweeps_csv) -> Tuple[pandas.DataFrame, pandas.DataFrame]` (the per-cell and the per-sweep table as written by `run_noise_fit.py merge`).
  - `sweep_laws_from_tables(cells: DataFrame, sweeps: DataFrame, cfg: TwinConfig = TwinConfig()) -> DataFrame`: one row per row of `sweeps`, columns `specimen_id`, `protocol`, `sweep_number`, `sigma_f_mV`, `sigma_m_mV`, `tau_m_s`, `sigma_s_mV`, `tau_s_s`, `tilt_mV_per_s`, `law_source`, `noise_flags`, `cell_tau_s_above_window` ((P0') of 2.7 (b)).
  - `build_twin_sweep_table(protocols: Sequence[CellProtocol], cells: DataFrame, sweeps: DataFrame, cfg: TwinConfig = TwinConfig()) -> DataFrame` with columns `TWIN_SWEEP_COLUMNS`, the rows of the cells in `protocols` only.
  - `TWIN_SWEEP_COLUMNS = ("specimen_id", "protocol", "sweep_number", "n_windows", "amplitude_pA", "onset_s", "duration_s", "n_samples", "fs_Hz", "sigma_f_mV", "sigma_m_mV", "tau_m_s", "sigma_s_mV", "tau_s_s", "tilt_mV_per_s", "law_source", "noise_flags", "cell_tau_s_above_window")`.
  - `write_twin_sweep_table(table: DataFrame, path) -> None`; `read_twin_sweep_table(path) -> DataFrame` (same dtypes and values as written).
  - `sweep_law(table: DataFrame, specimen_id: int, protocol: str, sweep_number: int) -> Tuple[noise_law.NoiseLaw, float]` (the law in mV, mV, s, mV, s and the tilt in mV/s of one row; `KeyError` when the row is absent, `TwinTableError` when it is not unique).
- Inputs: an archive cell directory as section 3 states (read only through `noise_windows.load_ss_windows` and `load_ls_prewindows`, plus `peak_pA` and `stim_duration_s` of `ss_pulses.npz` and `index` of `metadata.json['ls_sweeps']`); the two noise-fit tables (section 3).
- Outputs: the table, one row per real sweep. `specimen_id` int64 (the REAL cell); `protocol` `SS` | `LS` (upper case); `sweep_number` int64; `n_windows` int64 (SS: the sweep's stored windows; LS: 1); `amplitude_pA`, `onset_s`, `duration_s` float64 (LS: `ls_amplitude_pA`, `ls_onset_s`, `ls_duration_s`; SS: NaN -- the windows' polarity, amplitude and place stay in `CellProtocol`); `n_samples` int64 (SS: n_w; LS: the sweep's samples); `fs_Hz` float64 (the protocol's); the law and tilt of 2.7 (b), float64, mV / s / mV/s; `law_source` `sweep` | `cell`; `noise_flags` str (the per-sweep table's `noise_sweep_flags`, empty string when none); `cell_tau_s_above_window` bool, `tau_s_s > cfg.slow_flag_tau_s`, the same on every row of a cell. Rows sorted by `specimen_id`, then `SS` before `LS`, then `sweep_number`.
- Refused (`TwinTableError`, naming the cell and the sweep): a cell absent from the per-cell table or whose `status` is not `ok`; a stored SS or LS sweep without exactly one row of its protocol in the per-sweep table; a row of the per-sweep table (of a cell in `protocols`) without a stored sweep; a sweep of `metadata.json['ls_sweeps']` that `load_ls_prewindows` does not return (the twin must have every stored LS sweep, D-186.1); a law that `NoiseLaw` refuses or a tilt that is not finite; a failed row whose three SDs differ from the per-cell table's by more than `cfg.failed_law_rtol` (relative) -- the check that the row already holds theta_hat(c).
- Parameters: `TwinConfig.slow_flag_tau_s` (0.265 s), `TwinConfig.failed_law_rtol` (1e-9).
- Library calls relied on: `noise_windows.load_ss_windows`, `noise_windows.load_ls_prewindows`; `pandas.read_csv`, `DataFrame.to_csv(float_format="%.17g")`; `noise_law.NoiseLaw`.
- Custom code: none (a join and checks).
- Test oracles (`smoke_twin_table.py`, TT1-TT9): TT1 reference on a fixture -- two fixture cells written by `smoke_noise_windows.write_fixture` and two tables written by hand in the formats of section 3; every cell of the built table equals the value planted in the test (floats exactly); TT2 units and case -- `tau_*_s` equals the `_ms` column / 1000 exactly, `protocol` upper case, `sweep_law` returns the row's `NoiseLaw` and tilt exactly; TT3 a failed sweep -- `law_source` is `cell`, its law equals the cell's, and a failed row whose SD was altered is refused; TT4 each refusal of the list above raises `TwinTableError` naming the cell (and the sweep where there is one); TT5 the CSV round trip returns equal values, dtypes and column order; TT6 the flag -- `tau_s_s` = 0.2650 gives False, 0.2651 True, on every row of the cell; TT7 `load_cell_protocol` on the fixture returns the planted positions, polarities, sweep numbers, LS amplitudes, onsets, durations and lengths, and K and K_LS; TT8 the real run (needs `--nf2-dir`, and `--archive-cell` for its second half; reported SKIP with the reason when absent): `sweep_laws_from_tables` on the nf2 tables gives 589 rows (64 SS, 525 LS, 9 with `law_source = cell`) and each row's law equals `sweep_stage.twin_laws["<pr>:<sweep>"]` of its cell's JSON to 1e-12 relative (float64 through a CSV of 17 digits and a division by 1000) and its tilt equals `sweep_stage.rows[].tilt_mV_per_s` exactly; the four cells of D-182.3 and no other are flagged; with the archive cell 508282493, `build_twin_sweep_table` gives 2 SS rows (23 and 17 windows, n_samples 10500) and 22 LS rows (onset 0.27 s, duration 1.0 s, 363500 samples, amplitudes -110 to +310 pA); TT9 the two files are pure ASCII and LF.
- Data flow: consumes Block 6's tables and Block 3's loaders; consumed by Block 12 (the build takes each sweep's law and tilt from the table and the windows' and sweeps' protocol from `CellProtocol`) and Block 8 (the run writes the table beside the manifest; the report reads `cell_tau_s_above_window`).
- Amendment 1 (2026-11-19, D-190, user: a twin leaves out the depolarising LS sweeps the fitter's loader drops for the real cell). (i) `CellProtocol` gains `ls_spike_reason`, (K_LS,) of str: the empty string for a sweep with `ls_amplitude_pA` <= `TwinConfig.ls_dep_min_amplitude_pA` (not tested) and for a depolarising sweep the test passes, else the reason the test returns. (ii) `load_cell_protocol(spec_dir, cfg: WindowConfig = WindowConfig(), *, spike_reason_fn, twin_cfg: TwinConfig = TwinConfig()) -> CellProtocol`: `spike_reason_fn(v_mV: float64 (n,), sampling_rate_Hz: float) -> str` is a REQUIRED keyword (no default: the caller passes the loader's `passive_fitting_hpc_fixed.sweep_spike_reason`; Block 5 does not import the monolith, which imports NEURON); it is called once for every stored LS sweep with `ls_amplitude_pA` > `twin_cfg.ls_dep_min_amplitude_pA`, on that sweep's stored voltage `v_<index>` of `ls_sweeps.npz` (all its samples, as the loader passes it) and its sampling rate, and for no other sweep; a return value that is not a str is refused (`TwinTableError`). (iii) `TWIN_SWEEP_COLUMNS` gains two columns at its end, `real_spike_reason` (str; the empty string for an SS row) and `in_twin` (bool: `real_spike_reason` is empty; True for an SS row); `build_twin_sweep_table` fills them from `CellProtocol.ls_spike_reason`; the CSV round trip keeps them. The table still has one row per real sweep. (iv) Oracles added: TT10 -- on a fixture with hyperpolarising and depolarising LS sweeps and a `spike_reason_fn` written in the test (a reason for a chosen sweep), the reasons and `in_twin` equal what the test planted, the function is called exactly for the sweeps above the threshold (a sweep of amplitude exactly `ls_dep_min_amplitude_pA` is not tested), and a non-str return is refused; TT8 gains a third part, run when `--archive-cell` is given and `passive_fitting_hpc_fixed` imports (NEURON, scikit-optimize, matplotlib; SKIP with the reason otherwise): with `spike_reason_fn = passive_fitting_hpc_fixed.sweep_spike_reason`, the number of depolarising rows with `in_twin` True equals the loader's own `ls_dep_screen['n_kept']` for that cell loaded with `load_depolarising_ls=True` and no amplitude cap, and the number of rows per reason equals `ls_dep_screen['n_<reason>']` (the loader itself as the independent reference). Blocks affected: 5 (this entry), 12 (the build makes only the rows with `in_twin` True). Blocks 10 and 11 are not affected.
- Amendment 3 (2026-11-19, D-191.4 [orchestrator]; found at the gate). A sweep of `metadata.json['ls_sweeps']` that `noise_windows.load_ls_prewindows` does not return is no longer always refused: when its `detected_amplitude_pA` is missing, not finite, or smaller in size than `WindowConfig.ls_amp_min_pA` (5 pA: not a step), it is left out of `CellProtocol` and of the table without a row -- it has no noise law (the noise fit never saw it) and the fitter's loader ignores it as well (it takes hyperpolarising sweeps below -5 pA and depolarising ones above +5 pA); any other sweep the loader does not return (a step whose onset or end cannot be detected, or whose pre-window is too short) is still refused (`TwinTableError`), because a sweep the fit would use has no law. `ls_index` and the other `ls_*` fields are matched to the returned sweeps by sweep number, in the order of `metadata.json['ls_sweeps']`. The table has one row per real sweep that is a step. Oracle: TT4's refusal uses a step the loader cannot return (not a 0 pA sweep), and TT7 (or a new TT11) checks that a 0 pA sweep and a sweep without an amplitude listed in the metadata are left out and that `ls_index` of the remaining sweeps is still right. Blocks affected: 5 only.
- Confirmed: yes -- D-186.1, D-186.4, D-186.5, D-182.3, D-190 (user, 2026-11-19); the column names, the refusals and the SS rows' NaN protocol columns are the orchestrator's.
- Status: drafted

### Block 10: the twin's noise -- independent SS windows, one continuous LS draw, the tilt line and the seeds (D-180, D-182, D-183.3, D-186)

- Module: `twin_noise.py` (model layer: no I/O, no NEURON, no module-level RNG); smoke `smoke_twin_noise.py`.
- Public API:
  - `twin_rng(base_seed: int, twin_id: int, protocol: str, sweep_number: int, position: Optional[int] = None) -> numpy.random.Generator` (2.7 (e)).
  - `ss_sweep_noise(law: NoiseLaw, tilt_mV_per_s: float, dt: float, n_w: int, position: array-like int (K,), period_samples: int, *, base_seed: int, twin_id: int, sweep_number: int) -> float64 (K, n_w)`: for ONE SS sweep, row k = e(g, k) + the tilt term of (TW1) for the window at `position[k]`.
  - `ls_sweep_noise(law: NoiseLaw, tilt_mV_per_s: float, dt: float, n_samples: int, *, base_seed: int, twin_id: int, sweep_number: int) -> float64 (n_samples,)`: e(g, s) + the tilt term of (TW2).
- Inputs: `law` in mV, mV, s, mV, s; `tilt_mV_per_s` mV/s (finite); `dt` s (> 0); `position` the 0-based places of ALL the stored windows of the sweep (u_bar(c, s) is taken over the positions passed), unique, >= 0; `period_samples` >= 1; `protocol` `SS` | `LS`; `base_seed`, `twin_id`, `sweep_number`, `position` non-negative integers.
- Outputs: mV, float64, finite. `ss_sweep_noise`: rows in the order of `position`; each row's random part is a stationary draw of `law` (first sample in the stationary law) from its own generator, so two rows are independent and a row's random part does not depend on which other positions are passed; the tilt term sums to zero over the K n_w samples. `ls_sweep_noise`: one continuous stationary draw over the n_samples, plus `tilt_mV_per_s (j - (n_samples - 1) / 2) dt`. No offset term in either.
- Refused (`ValueError`): `position` empty, not unique or negative; `n_w` or `n_samples` < 2; `dt` <= 0 or a tilt that is not finite; `twin_rng` with protocol `SS` and no position, with protocol `LS` and a position, with an unknown protocol or a negative integer.
- Parameters: none of its own (`TwinConfig.base_seed` is passed in by the caller).
- Library calls relied on: `numpy.random.SeedSequence`, `numpy.random.default_rng`; `noise_law.draw_units(rng, 1, n)`, `noise_law.simulate_traces(..., tilt_mV_per_s=...)`, `noise_law.simulate_sweep(..., tilt_mV_per_s=...)`; `noise_windows.tilt_offsets` for (OF1)'s offsets.
- Custom code: none (compositions of Block 2's simulator and (OF1)).
- Test oracles (`smoke_twin_noise.py`, TN1-TN9): TN1 reference -- every row of `ss_sweep_noise` equals (TW1) written out in the test from `twin_rng`, `noise_law.draw_units`, `noise_law.simulate_traces` without a tilt and the explicit line b (u(k, j) - u_bar) to 1e-12 mV (float64 roundoff on sums of mV-sized terms), and `ls_sweep_noise` equals (TW2) written out the same way to 1e-12 mV; TN2 moments -- over 4000 windows of 2000 samples at dt = 20 us (distinct twin ids, a cohort-like law, no tilt) the mean demeaned sum of squares equals `noise_law.expected_s0` and the across-window autocovariance at lags 0, 1, 5, 50 equals `noise_law.autocovariance` within 3.5 SE; TN3 independence -- with a slow-dominated law (sigma_s 1 mV, tau_s 1 s, for which adjacent windows of a continuous train would correlate at about 0.8) the correlation between the window means of two positions of one sweep, over 2000 twin ids, is zero within 3.5 SE (SE = 1 / sqrt(2000)); TN4 the tilt -- noise with tilt b minus noise with tilt 0 on the same seeds equals b (u(k, j) - u_bar) to 1e-12 mV for tilts of both signs and zero, sums to zero over the sweep to 1e-9 mV, and for LS equals b (j - (n - 1) / 2) dt; TN5 seeds -- the same arguments give identical arrays; changing any one of base_seed, twin_id, sweep_number, position or protocol changes the draw; a window's random part (tilt 0) is identical whether its position is passed alone or among others; TN6 contract -- shapes, float64, finiteness, and every refusal above; TN7 zeros -- a law with the three SDs at 0 and tilt 0 gives exact zeros, and with the cohort-like law the mean of 4000 window means is zero within 3.5 SE (no offset); TN8 the LS draw is continuous over the whole sweep -- with (sigma_s 1 mV, tau_s 1 s) and n_samples covering 3 s at dt = 1 ms, over 2000 twin ids the variance at the first and at the last sample equals gamma(0) and the covariance between samples 1 s apart equals sigma_s^2 exp(-1) within 3.5 SE; TN9 the two files are pure ASCII and LF.
- Data flow: consumes Block 2 (`noise_law`) and `noise_windows.tilt_offsets`; consumed by Block 12 (the build adds these arrays to Block 11's clean responses).
- Confirmed: yes -- D-180, D-182, D-183.3, D-186.2, D-186.3, D-186.6 (user); the entropy lists of 2.7 (e) and u_bar taken over the stored windows are the orchestrator's.
- As built [orchestrator, accepted at the gate 2026-11-19]: also refused (`ValueError`) -- `period_samples` < 1; a seed argument, a sweep number or a position that is not an integer (a bool, a float, a `position` array of non-integer dtype: never rounded); a `dt` that is not finite; a protocol that is not exactly `SS` or `LS` (case-sensitive, the keys of `twin_config.PROTOCOL_CODE`). TN4's sum-to-zero bound (1e-9 mV) is applied to the LS line as well. A failed comparison saves its arrays to `smoke_artifacts/` (git-ignored).
- Status: smoke-tested 2026-11-19 (sandbox: `smoke_twin_noise.py` TN1-TN9 9/9, about 5 s, run by the implementer and again by the orchestrator at the gate; six planted defects -- one generator shared by a sweep's windows, the tilt line uncentred, (OF1)'s offsets omitted, the SS windows cut from one continuous train, the LS sweep drawn as two independent halves, the LS sweep seeded with the SS code -- each made the run fail; TN3 measured 0.027 against a bound of 0.078 where a continuous train gives 0.880; commit `ddf2e48`, merged `9aa9c70`)

### Block 11: the twin's clean responses -- one SS pulse per polarity from rest and each Long Square sweep at its real amplitude and timing (D-183.1, D-186.4, D-187)

- Module: `synthetic_ground_truth.py`, two new functions; every existing function and class of the file is left as it is, bit for bit (the legacy path of old manifests). Smoke `smoke_twin_response.py` (needs NEURON; the I_h cases need the compiled mechanisms, `nrnivmodl mod` -> `./x86_64`).
- Public API:
  - `simulate_ss_clean(cell, *, amplitude_pA: float, duration_s: float, t_s: float64 (n_w,), v_init_mV: float, dt_ms: float = 0.025) -> Tuple[float64 (n_w,), float64 (n_w,)]`: (v_mV, i_pA) of one pulse on the stored window's time axis.
  - `simulate_ls_clean(cell, *, amplitude_pA: float, onset_s: float, duration_s: float, n_samples: int, fs_Hz: float, v_init_mV: float, dt_ms: float = 0.025) -> Tuple[float64 (n_samples,), float64 (n_samples,)]`: (v_mV, i_pA) of one Long Square sweep.
- Inputs: `cell` as `build_synthetic_cell` returns it (it has `simulate(stim_amp_pA, stim_delay_ms, stim_dur_ms, tstop_ms, v_init_mV, dt_ms)`); `amplitude_pA` signed; `t_s` uniform, s, 0 at the pulse onset, `t_s[0]` < 0 (the stored window's axis: -10 ms to +199.98 ms at 50 kHz in L3_exc); `onset_s`, `duration_s` s; `fs_Hz` Hz; `v_init_mV` the twin's resting potential (mV; `measure_rin_tau_sag`); `dt_ms` NEURON's fixed step (`TwinConfig.neuron_dt_ms`).
- Outputs: `v_mV` mV, float64, finite: NEURON's somatic trace of ONE run from `v_init_mV` (stimulus delay `-t_s[0]` for SS, `onset_s` for LS), read at the output samples by `_resample`'s linear interpolation (`numpy.interp`) as the legacy functions do; SS sample j at `t_s[j]`, LS sample j at `j / fs_Hz`. `i_pA`: SS, `amplitude_pA` where `0 <= t_s <= duration_s` and 0 elsewhere (the legacy rule); LS, `amplitude_pA` on the samples `j_on .. j_on + n_on - 1` with `j_on = round(onset_s fs_Hz)`, `n_on = round(duration_s fs_Hz)`, and 0 elsewhere (the real archive's layout: 13500..63499). Neither function adds noise, averages, or keeps state between calls.
- Refused (`ValueError`): `t_s` with fewer than 2 samples, not uniform (relative spread of its steps above 1e-6) or with `t_s[0]` >= 0; `n_samples` < 2; `fs_Hz`, `duration_s` or `dt_ms` <= 0; `onset_s` < 0 or `j_on + n_on` > `n_samples`; a non-finite argument.
- Parameters: `TwinConfig.neuron_dt_ms` (0.025 ms, the generator's value), passed by the caller.
- Library calls relied on: `cell.simulate` (NEURON, fixed step), `numpy.interp` through the existing `_resample` or the same call.
- Custom code: none.
- Test oracles (`smoke_twin_response.py`, TR1-TR9; cells on `write_ball_and_stick_swc`, passive and with I_h through `GroundTruthParams.ih`): TR1 legacy parity -- with zero noise, the `v_mV` of `simulate_ss_clean` on the legacy grid equals the bundle of `simulate_ss_protocol` (per polarity) and the `v_mV` of `simulate_ls_clean` with the legacy onset, duration and length equals the bundle of `simulate_ls_protocol`, to 1e-9 mV (the same NEURON run and the same interpolation; the two output time axes agree to float roundoff, at most 1e-12 ms, and the trace's slope is at most tens of mV/ms), for the passive and the I_h cell; TR2 linearity -- for the passive cell, v(2A) - v_init equals 2 (v(A) - v_init) to 1e-8 mV, SS and LS (a linear system under a fixed-step implicit scheme; roundoff); TR3 baseline -- every sample before the onset equals `v_init_mV` to 1e-9 mV for the passive cell (v_init = e_pas) and to 5e-3 mV for the I_h cell started at its measured rest (a sixth of the smallest sigma_f of the nf2 run, 0.031 mV); TR4 steady state -- for the passive cell the deflection at the end of a 1 s step of -50 pA equals `measure_rin_tau_sag`'s R_in times the amplitude to 1e-3 relative (two runs at different steps, 0.025 and 0.05 ms); TR5 I_h -- for a hyperpolarising 1 s step on the I_h cell the minimum during the step lies below the value at the step's end by more than 1e-3 mV (sag) and the maximum after the step lies above `v_init_mV` by more than 1e-3 mV (rebound), and for the passive cell neither happens: the minimum during the step is not below the value at its end by more than 1e-9 mV and the maximum after the step does not exceed `v_init_mV` by more than 1e-9 mV; TR6 contract -- shapes, float64, finiteness; `i_pA` on exactly the samples stated, for the legacy geometry and for the real one (n_samples 363500, onset 0.27 s, duration 1.0 s, 50 kHz; the SS axis of 10500 samples from -10 ms); TR7 no state -- two identical calls give identical arrays, and a call made after a call at another amplitude equals a call on a freshly built cell to 1e-12 mV; TR8 every refusal above; TR9 `smoke_twin_response.py` is pure ASCII and LF, and `git diff <contract commit> -- synthetic_ground_truth.py` shows additions only (no line of the legacy code removed or changed).
- Data flow: consumes `build_synthetic_cell`, `measure_rin_tau_sag` (existing); consumed by Block 12 (one `simulate_ss_clean` per polarity with A_p(c), d_p(c) of 2.7 (f); one `simulate_ls_clean` per distinct (amplitude, onset, duration, length)).
- Amendment 2 (2026-11-19, D-191 [orchestrator]; the issue raised by Block 11's implementer, evidence in 2.7 (f)). (i) Both functions gain the keyword `settle_ms: float = 0.0` (ms, >= 0, finite; `TwinConfig.settle_ms` = 6000, the settle of `measure_rin_tau_sag`, is passed by the caller): NEURON is started at `v_init_mV` and runs `settle_ms` with no stimulus before the first output sample, so the stimulus delay is `settle_ms - t_s[0] 1e3` (SS) or `settle_ms + onset_s 1e3` (LS) and output sample j is read `settle_ms` later on NEURON's clock; shapes, grids and `i_pA` are unchanged; with `settle_ms` = 0 the functions are exactly as before (TR1's parity is tested at `settle_ms` = 0). A negative or non-finite `settle_ms` is refused. (ii) The SS current trace follows the real archive, not the legacy rule [data inspected, 508282493: the current is on samples 500..524, and t_s[525] = 0.0005 exactly, so the closed rule gives 26 samples]: `i_pA` is `amplitude_pA` on the samples `j0 .. j0 + n_on - 1`, `j0 = round(-t_s[0] f_s)` (the sample at t = 0), `n_on = round(duration_s f_s)`, f_s = 1 / the step of `t_s`, and 0 elsewhere; TR1 compares the voltages only; TR6 asserts 500..524 on the real axis. (iii) TR3 restated: passive cell, every sample before the onset equals `v_init_mV` to 1e-9 mV at `settle_ms` = 0; I_h cell (uniform e_pas, I_h in soma and dendrites, F = 1.9) started at its measured rest with `settle_ms` = 6000: every sample before the onset equals the first output sample to 1e-6 mV (no drift) and equals `v_init_mV` to 5e-3 mV (`measure_rin_tau_sag`'s rest is the settled soma potential); the same cell at `settle_ms` = 0 drifts by more than 5e-3 mV before the onset (the defect this amendment removes: the check must see it). (iv) TR10 added, the settle is converged and neutral: for the passive cell `settle_ms` = 0 and 500 give the same SS and LS voltages to 1e-7 mV [corrected 2026-11-19, amendment 4, D-191.3 [orchestrator]: first written 1e-9 mV; NEURON's recorded clock is a running sum t += dt and lags k dt by 3.3e-10 ms after 500 ms (5.2e-8 ms after 6000 ms), so a run read through the legacy interpolation on the recorded clock is displaced by that much: times the trace's slope (16 mV/ms on the SS pulse edge) it gave 5.3e-9 mV (SS) and 1.6e-8 mV (LS), measured by the implementer; read on the exact step clock the same runs agree to 1e-12 mV, so the settle itself is neutral. The legacy read-out on the recorded clock is kept (parity with the legacy generator, TR1); at the real settle the displacement is about 1e-6 mV, against sigma_f >= 0.031 mV]; for the I_h cell `settle_ms` = 6000 and 12000 give the same voltages to 1e-4 mV (measured 3e-6 mV). Blocks affected: 11 (this entry), 12 (the build passes `TwinConfig.settle_ms`). Blocks 5 and 10 are not affected.
- Confirmed: yes -- D-183.1, D-186.4, D-187 (user); the half-open current intervals (from the real archive's layout, data inspected), the legacy interpolation kept and the settle of amendment 2 (D-191) are the orchestrator's.
- Status: drafted

### Blocks 12 and 8 (wave 2; entries to be written when Blocks 5, 10 and 11 are merged)

Block 12, the twin build (`twin_build.py`, `gen_from_manifest.py`): for one manifest row, the clean responses of Block 11 plus the noise of Block 10, with each sweep's law and tilt from Block 5's table, written as an archive cell the fitter's loader reads like a real one (2.7). Block 8, the Stage-7 run (`synth_gt_grid.py`, `run_ih_recovery.py`, `submit_ih_recovery.sh`, `ih_recovery_report.py`): the manifest of D-184 and D-188 (three configurations of 10 twins on cells drawn at random from the 18), one PBS job per twin that builds it and runs its fit, a merge job that writes the report (D-185). Their interfaces are written here before their code.

### Tools (not blocks): cluster-side checks and the SS archive inspection

- Files: `noise_fit_gates.py` (read-only helpers), `cluster_gates_noise_fit.sh` (the runner), `smoke_noise_fit_gates.py` (T1-T8).
- `python noise_fit_gates.py bytes [--folder DIR]` -> `BYTES: ok|FAIL (...)`, exit 0/1: top-level `*.py` pure ASCII; `*.sh`, `*.pbs`, `mod/*.mod` without CR.
- `python noise_fit_gates.py compile [--folder DIR]` -> `COMPILE: ok|FAIL (...)`, exit 0/1: every top-level `*.py` compiles in memory (no `.pyc`).
- `python noise_fit_gates.py ss-pulses DIR [DIR ...]` -> per specimen: the keys, dtypes and shapes of `ss_pulses.npz` (Format A or B); per sweep number (or, without one, per chain of adjacent windows) the window count, the polarity string (`+` dep, `-` hyp), the adjacent consecutive pairs and their onset spacing in ms, the gaps; the `metadata.json` keys about SS; then one line per cell, `SS_PULSES <id>: format=A|B sweep_number=yes|no n_pulses=.. n_sweeps|n_chains=.. per_sweep|per_chain=.. adjacent=a/b opposite_polarity_adjacent=c/a spacing_ms=.. fs_Hz=.. window_ms=..`; exit 0 when every cell was read and carries sweep numbers, else 1. Adjacency: windows k, k+1 of one sweep whose onsets are D samples apart (D < window length n) agree exactly on n - D samples of voltage and of current; the smallest D >= 1 with at least `MIN_OVERLAP` = 100 agreeing samples is the spacing (custom: no library function; a 16-sample vectorised pre-filter keeps it fast on ADC-quantised traces, where single values recur).
- `bash cluster_gates_noise_fit.sh` (FULL) or `QUICK=1 bash ...`: one `PASS|FAIL|SKIP  <check>  -- <seen>` line per check -- G0 interpreter, G1 numpy/scipy/statsmodels, G2 Blocks 1-4 and 9 committed (commit subjects; Block 9's `block 9: the per-sweep stage` added 2026-11-12, over every commit of the folder since 2026-10-29: a fixed window of 40 commits would lose Block 1 as documentation commits accumulate) and no tracked file of the folder modified, G3 bytes, G4 compile, G5 `smoke_noise_stats` 6/6, G6 `smoke_noise_law` 10/10 (M10 added 2026-10-29), G6b `smoke_noise_windows` 15/15 (Block 3, added 2026-10-23; 14 checks since 2026-10-29), and in FULL mode G6c `smoke_noise_fit` 11/11 (Block 4, about 2 min, added 2026-10-23; 11 checks since 2026-10-29), G6d `smoke_noise_sweep_fit` 11/11 (Block 9, about 7 min, added 2026-11-12), G7 NEURON/skopt and `x86_64/special` (built with `nrnivmodl mod` under the submit scripts' `.nrnivmodl.lock` only when missing or older than `mod/*.mod`), G8 `smoke_ih_fit` 12/12, G9 `smoke_ls_baseline_qc` 10/10, G10 `smoke_ih_recovery` 14/14, G11 `regression_passive_identity.py` PASS on `$IH_ARCHIVE_GROUP/specimen_$REG_SPECIMEN`, G12 `noise_fit_gates.py ss-pulses` on `$NPZ_SPECIMENS`; then `GATES noise-fit: p passed, f failed, s skipped` and `VERDICT: SETTLED|NOT SETTLED -- <reason>` (SETTLED only in FULL mode with nothing failed or skipped). A check passes only on a line it must see (the suite's own summary line), never on an exit status. Logs in `$GATE_LOG_DIR` (default `~/ih_fit_gate_logs/<date_time>/`), never in the repository; no tracked file is written. Exit 0 iff nothing failed.
- Defaults: `IH_ARCHIVE_GROUP=/davinci-1/home/ldellamea/Human Neurons Fitting/L3_exc`, `REG_SPECIMEN=508282493`, `NPZ_SPECIMENS="508282493 569844159"`, `PYTHON=python`.
- Test oracles (`smoke_noise_fit_gates.py`): T1 planted CR and non-ASCII files flagged, nothing else; T2 a syntax error named, no `.pyc`; T3 a writer-like Format-A fixture (two sweeps of an ADC-quantised trace, spacings 200.000 and 200.040 ms, one pulse dropped) gives the exact expected summary and the gap; T4 Format B the same; T5 no `sweep_number`: three adjacency chains; T6 no false adjacency in 200 independent quantised pairs, a 50-sample overlap refused unless the minimum is lowered; T7 the runner's QUICK mode passes G0, G1, G3-G6b and cannot settle, a bad interpreter fails G0 with exit 1; T8 ASCII/LF. Mutation-tested 2026-10-20 (no overlap minimum, sweep numbers ignored, a wrong ms scale, `mod/` not scanned, a check certified on exit status: each caught).
- Status: smoke-tested 2026-10-20 (sandbox: T1-T8 8/8). The FULL runner was run in the sandbox on `main` + Block 2 + these tools, with `Biological Fit/` and `Synthetic Passive fit Test/` from `main` and a synthetic archive group; its result is in the decision log's changelog of 2026-10-20. Not run here: G11 on a real cell and G12 on the real archive (cluster). Run on the cluster 2026-10-20 (FULL, `main` at `ca028ea`, env `prova`): 13 passed, 0 failed, 0 skipped, VERDICT SETTLED; G11 PASS on 508282493; G12 read 508282493 and 569844159 (section 3). [2026-11-12: G2 and G6d extended to Block 9 (FULL mode now 14 checks); T1-T8 8/8 after the change (sandbox); not run on davinci since.]

## 6. Pipeline entry points

| Command | Purpose | Minimal configuration for testing | Expected runtime |
|---|---|---|---|
| `python smoke_noise_stats.py` | Block 1 smoke (N1-N6) | none | seconds |
| `python smoke_noise_law.py` | Block 2 smoke (M1-M11) | none | about 5 s |
| `python smoke_noise_fit_gates.py` | tools smoke (T1-T8) | none | about 15 s |
| `python smoke_noise_windows.py` | Block 3 smoke (W1-W15) | none (W4's oracle needs NEURON for the monolith import, W12's pandas for `ls_baseline_qc`, else skipped) | about 20 s |
| `python smoke_noise_fit.py` | Block 4 smoke (F1-F11, reduced M and R) | none | about 100 s |
| `python smoke_noise_sweep_fit.py` | Block 9 smoke (P1-P11, reduced M and B_v; `P3,P6` runs a subset) | statsmodels (P3's oracle) | about 7 min |
| `python smoke_noise_run.py` | Block 6 smoke (R1-R9 at `--sizes smoke`; `R5,R7` runs a subset) | statsmodels, git, bash; no NEURON, no PBS | about 8.5 min on 2 cores |
| `RUN_TAG=<tag> [CELLS=<id,id>] DRY_RUN=1 bash submit_noise_fit.sh` | the noise-fit run's dry run: the cohort, the 120-core cap, the CLI contract, every qsub and command; nothing written or submitted | the conda env; the archive group | seconds |
| `bash cluster_gates_noise_fit.sh` (`QUICK=1` for the fast half) | every cluster-side check of Blocks 1-4 and 9 and the SS archive read, one PASS/FAIL/SKIP line each and a VERDICT | the conda env active; on the cluster nothing else | about 25 min (QUICK < 1 min) [2026-11-12: about 17 min before G6d] |
| `python smoke_ih_recovery.py`, `python smoke_ih_fit.py`, `python smoke_ls_baseline_qc.py` | existing suites, re-run after Block 1 (R8/R10 bands now through `sample_acf`) | NEURON + compiled `mod/` | minutes |
| `python regression_passive_identity.py --ref-dir "../Biological Fit" --archive-cell <dir>` | bit-for-bit gate against the oracle folder | the oracle folder and one archive cell | minutes |
| `python smoke_twin_table.py [--nf2-dir <dir>] [--archive-cell <specimen dir>]` | Block 5 smoke (TT1-TT9); TT8 on the real nf2 tables and one real archive cell when the paths are given | pandas; no NEURON | seconds |
| `python smoke_twin_noise.py` | Block 10 smoke (TN1-TN9) | none | under a minute |
| `python smoke_twin_response.py [--build]` | Block 11 smoke (TR1-TR9) | NEURON; `./x86_64` (`--build` runs `nrnivmodl mod`) | minutes |

## 7. Not executable in the sandbox

- `regression_passive_identity.py`: `Biological Fit/` is not in the zip baseline and the archive is on the cluster; skipped here, to be run on the cluster (command in README).
- Anything reading the real archive (`noise_calibration.py --group-dir`, S3, G12 of `cluster_gates_noise_fit.sh`).
- G11 of `cluster_gates_noise_fit.sh` on a real archive cell (here only on a synthetic cell, with `Biological Fit/` from `main`).
- The noise-fit run itself (Block 6): `submit_noise_fit.sh` under PBS on the real archive -- the pilot and the 18 cells (D-124, D-125.5); here only against stubs of `conda`, `qsub` and `python`, and `run_noise_fit.py` on the fixture at `--sizes smoke`.
- The twins on the real archive (Blocks 5, 12, 8): the 18 cells' archive is on the cluster. In the sandbox: the nf2 tables and JSONs and the archive cell 508282493 (the user's upload of 2026-11-19, kept outside the repository), on which `smoke_twin_table.py`'s TT8 runs.

## 8. Open questions

- ~~Does `ss_pulses.npz` carry sweep numbers? The loader reads none (D-056 open point; cluster check requested 2026-10-18).~~ [answered at source 2026-10-20; confirmed on the files 2026-10-20 by G12 on the cluster: both cells read carry `sweep_number`] The archive writer (`Passive Features/Allen Institute Data/save_alleninstitute_data (6).py`, `_archive_one_cell`, every copy in that file) stores `sweep_number` (int64, one per pulse) in Format A and B; a sweep's windows in onset order (`_detect_pulses_in_current` scans the current trace); each window (-10, +200) ms around its onset; pulses failing the amplitude/duration QC (200 +/- 30 pA, 0.5 +/- 0.5 ms) or too close to a trace boundary dropped silently; onset times not stored. Its docstring, citing the Allen tech paper: all 20 pulses of the SS protocol, alternating polarity, 200 ms apart, lie in ONE sweep. [corrected 2026-10-20: one sweep holds the whole train, but in the archive's cells the train is 37 pulses, 200.5 ms onset to onset, and the writer's trace QC drops pulses as well (section 3, data inspected)] Adjacency and the true onset spacing follow from the exact overlap of consecutive windows (tools, section 5).
- Block 5 (the twins' generator): `synthetic_ground_truth.simulate_ss_protocol` gives every SS pulse its own sweep number and independent noise. Under D-017 (per-sweep offset and tilt) and D-056 (i) (continuous sweeps in the fit's simulator), the twins' SS sweeps should carry the real sweep structure -- 20 pulses per sweep [corrected 2026-10-20: 37 pulses, 200.5 ms onset to onset, section 3] with the slow component and the tilt continuous across them -- or the twins' SS windows (and the SS bundles the I_h fit averages) are independent where the real cells' are not. To settle with the user when Block 5 is specified. [answered 2026-11-19: D-180, D-186.2 (each SS window's noise independent, the tilt line at the window's real place), D-187 (each pulse simulated alone from rest, no train), D-189 (the bundles' cross-window correlation not measured: a limit of Stage 7); section 2.7.]
- Root `specs/SPEC.md` Coverage pointing to this file (user's call, D-056 (ii)).
- (added 2026-11-19, Block 5) A twin has every stored LS sweep of its real cell (D-186.1), the depolarising ones included, simulated without Na/Kv; the fitter's loader drops a real sweep on which the cell spiked (`sweep_has_spike`) but will not drop its twin. Whether the twins should omit the sweeps the real cell's fit never sees is put to the user (2026-11-19). [answered at source 2026-11-19, `passive_long_step_training.assign_ls_roles`: the depolarising long steps are VALIDATION only, and at most the `dep_n_validation` = 3 smallest spike-free ones; training uses the SS hyperpolarising pulses and the hyperpolarising steps h_2 .. h_{n-1}. The fitted parameters therefore do not depend on the larger depolarising sweeps of a twin, and the validation set differs from the real cell's only if the real cell spiked on one of its three smallest depolarising steps. The twins keep every stored sweep (D-186.1); nothing to decide.]
- (added 2026-10-20) The stored SS windows are a selected subset: they passed the writer's trace QC (section 3), a test on the lag-1 structure of each pulse's residual after a bounded single-exponential fit over [offset + 2 ms, + 40 ms], the timescales of the law's slow part. The fit's simulator does not apply it. Check SQC (standalone script `sqc_trace_qc_selection.py`, not in the repository; the repository's `trace_qc_v2.py` with the archive's configuration on 2-sweep cells simulated with the Block-2 law, RC response 113 MOhm / 28.1 ms) compared the D-051 SS statistics of the kept windows with a random subset of the same per-polarity counts: Gaussian noise of the law alone makes the QC drop 12.9 % of pulses (theta_cf) and 12.0 % (sigma_s = 0.025 mV); the selection shifts the medians by -0.17 to -0.19 per-cell SD at t_SS = 40 ms, by at most 0.09 SD at t_SS = 80 ms and 0.08 SD at 120 ms, and V_hat at 4 periods by at most 0.07 SD (for a 10-sweep cell the SD is about 2.2 times smaller). The archive's own drop rate in 508282493 (46 %) is far above the noise-only rate, so most of its drops remove structure the law does not have; the check does not cover those. [decided 2026-10-21, D-072: no shape test anywhere in the noise fit; every L3_exc cell's layout is reconstructable, so no fallback is needed and Block 3 refuses a cell whose table, counts or polarity sequence disagree.] Still open: the floor t_SS(c) >= 80 ms and the per-cell excess-drop label (the assistant's proposal).
- (added 2026-10-23, Block 4) The tilt sigma_b(c) of a 2-sweep cell (14 of 18) is estimated from two draws of b (2.3 (h)): keep per cell as D-034 says (and report its uncertainty), pool the tilt across cells (a cohort sigma_b or a shrinkage towards it, as D-076 does for sigma_s per sweep), or draw on the LS sweeps' slopes as well (each LS pre-window is 265 ms: a slope of 0.1 mV/s moves its mean by 0.027 mV, within the pre-window SD; weak)? [corrected 2026-10-23: the LS pre-window slopes are not an option -- excluded by D-034's status (D-017 note 4: a tilt calibrated from pre-window slopes would be too large by more than an order of magnitude).] The user's call. Simulation 2026-10-23 (`check_findings_2026-10-23.py`; theta_cf, sigma_b = 0.1 mV/s, t_SS = 80 ms, (T1) at the true law): sigma_b_hat / sigma_b has 95 % range 0.00-1.95 with 2 sweeps (clipped at 0 in 3.5-4.9 % of cells), 0.42-1.56 with 6 and 0.59-1.44 with 10 -- the spread of sqrt(chi^2_S / S) from the S realised tilts, plus the noise term in the lower tail -- and sigma_b_hat^2 is unbiased (mean ratio 0.97-1.01); pooled over the cohort's 64 SS sweeps under a common sigma_b, 0.82-1.17. A sigma_b error of that size (0 or 0.17 in the simulator for a cell at 0.1) moves u_hat by at most 0.18 of its SD (linearised at theta_cf, fixture layout): what is at stake is the twins' tilt, not theta_hat. Also open: the batched FFT autocorrelation and the Markov-jump simulator of 2.3 (h), both unneeded at the measured cost. [answered 2026-10-28 (later): D-100 (user, "Let's do A and B for the 2 sweep cells") -- for the 14 cells with two SS sweeps the twins' tilt SD is taken both per cell (D-034, variant a) and as one cohort value (variant b); partial pooling not taken. The assistant's reading, to confirm before Blocks 5-6 are specified: two labelled variants of twins per 2-sweep cell, sharing their manifest row and unit draws; the cohort value max(0, sum_c S_eff(c) sigma_b_hat_u(c)^2 / sum_c S_eff(c)) over the 18 cells, from the unclipped (T1) values; each real cell's own fit (2.3) keeps its own sigma_b_hat. Blocks 3-4 then also export sigma_b_hat_u(c)^2 and the per-sweep pair counts; a cohort step follows the per-cell jobs (Block 6).] [corrected 2026-10-28 (later, 2): D-100 was the assistant's misreading of the user's answer and is withdrawn. D-102 (user): each twin sweep carries the realised tilt of its own real sweep, b_hat(c, s), and no statistic of the tilt (no sigma_b(c)) is computed for the twins; on an SS sweep b_hat(c, s) is the least-squares slope of the window means on time with one intercept per polarity (proposal). Open: the Long Square sweeps' tilts and the fit's simulator (the same b_hat(c, s), proposal, or the data detrended).] [2026-10-29: D-107 -- an LS sweep's tilt is the line through its pre-window and end-window means; the fit's simulator gives every simulated sweep its real sweep's tilt, so the tilt SD leaves the fit.]
- (added 2026-10-21) D-073's per-sweep calibration: which of (sigma_f, rho_f, sigma_s, tau_s, sigma_b) are estimated per sweep and which per cell; raw or partially pooled per-sweep values; the twins' protocol twinning (LS amplitudes of the real cell instead of the cohort's `--ls-hyp-amps`; SS sweeps as 37-pulse trains at the real stored positions). Evidence for the proposal: `sweep_identifiability.py` (standalone, not in the repository): from one sweep at theta_cf the fast pair is identified to 1-3 % (sigma_f) and +/-0.012-0.014 (rho_f), sigma_s to 18-30 % (6-13 % with tau_s held), tau_s only within a factor of about 2 (log-SD 0.49-0.72). [answered: D-076 (i) (2026-10-23) for the form, D-076 (iii) for the LS amplitudes, D-116 (2026-10-30) for the estimation; section 2.5, Block 9. The SS twins' structure stays with Block 5.]
- (added 2026-10-30, Block 9 = "Block 4b"; put to the user with the SPEC entry) The per-sweep stage's open points (2.5 (d), (f); cost and precision in 2.5 (h)): (1) B_v, the bootstrap's refits per sweep (proposal 100: SD(v) / v about 0.14, comparable to what the binding function's Monte Carlo error already leaves on v) [corrected 2026-11-02: proposal 50 -- what matters downstream is the shrunk sigma_s_tilde, and its RMS error moves by at most 0.16 percentage points between B_v = 50 and v known (SS with 2 or 10 sweeps, LS with 5; tau 0-0.2; normal errors, v as measured), `dev/2026-11-02_bootstrap_explained/check_bootstrap_B.py`; at B_v = 50 with M = 1000 refits the fixture cell costs about 6.6 h on one core]; (2) the refits' binding size (proposal M = 1000: the linearised v moves by -0.5 % to +2.3 % against M = 3000, for a third of the cost; the alternative is the main solve's M); (3) a refit that fails (proposal: left out of (P2) and counted, the sweep flagged above 10 % -- v is then the variance of the estimates that would enter the pooling, as D-116.4 leaves a failed sweep out; the alternative keeps the at-bound values in v) [2026-11-02: one refit stuck at the lower bound of sigma_s, kept in v, multiplies v by 85 (SS) or 16 (LS) at B_v = 100 and by 170 or 31 at 50, `check_bootstrap_B.py` part C]; (4) D-116.4's "the cell's values" for a failed sweep's twin: theta_hat(c) for sigma_f, rho_f and sigma_s (the assistant's first reading), or theta_hat(c) for sigma_f and rho_f with sigma_s = exp mu_hat_pr(c), the pooled value of its protocol (theta_hat(c)'s when k_pr = 0) -- the value (D-076.1) gives a sweep that carries no information (v -> infinity) and the one that keeps the SS/LS difference D-116.3 separates. [answered 2026-11-11, D-122 (user): (1) B_v = 50, (2) M_refit = 1000, (3) left out of (P2) and counted, flagged above 10 %, (4) the cell's whole law theta_hat(c) -- 2.5 (d), (f) and Block 9's entry updated]
- (added 2026-11-11, D-123) The order after Block 9's code (user: "I would fit the noise features on n cells in order then to use these fits to construct the twins on which to run the synthetic recovery run"; the assistant's reading, to confirm): Block 6's driver and PBS job fit Blocks 3, 4 and 9 on n real cells on davinci first; Block 5's twins are built from that run's per-sweep laws and tilts; then Stage 7. Open: n and which cells (the L3_exc group holds 18); whether the S0-S3 checks (D-055) run before the twins; the job layout (per cell for Blocks 3-4, per (cell, sweep) for Block 9's bootstrap, 2.5 (h)); a 1-2 cell pilot before the n-cell run. Block 5's own question above (the twins' SS sweep structure) stays open.
