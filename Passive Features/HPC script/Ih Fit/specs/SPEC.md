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
- (to be added block by block) `noise_windows.py`,
  `noise_fit.py`, `run_noise_fit.py`, `smoke_noise_fit.py`, the `sens_*.py`
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

### 2.4 The tilt (D-034.1, D-051 (iii))

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

### Tools (not blocks): cluster-side checks and the SS archive inspection

- Files: `noise_fit_gates.py` (read-only helpers), `cluster_gates_noise_fit.sh` (the runner), `smoke_noise_fit_gates.py` (T1-T8).
- `python noise_fit_gates.py bytes [--folder DIR]` -> `BYTES: ok|FAIL (...)`, exit 0/1: top-level `*.py` pure ASCII; `*.sh`, `*.pbs`, `mod/*.mod` without CR.
- `python noise_fit_gates.py compile [--folder DIR]` -> `COMPILE: ok|FAIL (...)`, exit 0/1: every top-level `*.py` compiles in memory (no `.pyc`).
- `python noise_fit_gates.py ss-pulses DIR [DIR ...]` -> per specimen: the keys, dtypes and shapes of `ss_pulses.npz` (Format A or B); per sweep number (or, without one, per chain of adjacent windows) the window count, the polarity string (`+` dep, `-` hyp), the adjacent consecutive pairs and their onset spacing in ms, the gaps; the `metadata.json` keys about SS; then one line per cell, `SS_PULSES <id>: format=A|B sweep_number=yes|no n_pulses=.. n_sweeps|n_chains=.. per_sweep|per_chain=.. adjacent=a/b opposite_polarity_adjacent=c/a spacing_ms=.. fs_Hz=.. window_ms=..`; exit 0 when every cell was read and carries sweep numbers, else 1. Adjacency: windows k, k+1 of one sweep whose onsets are D samples apart (D < window length n) agree exactly on n - D samples of voltage and of current; the smallest D >= 1 with at least `MIN_OVERLAP` = 100 agreeing samples is the spacing (custom: no library function; a 16-sample vectorised pre-filter keeps it fast on ADC-quantised traces, where single values recur).
- `bash cluster_gates_noise_fit.sh` (FULL) or `QUICK=1 bash ...`: one `PASS|FAIL|SKIP  <check>  -- <seen>` line per check -- G0 interpreter, G1 numpy/scipy/statsmodels, G2 Blocks 1-2 committed and no tracked file of the folder modified, G3 bytes, G4 compile, G5 `smoke_noise_stats` 6/6, G6 `smoke_noise_law` 9/9, and in FULL mode G7 NEURON/skopt and `x86_64/special` (built with `nrnivmodl mod` under the submit scripts' `.nrnivmodl.lock` only when missing or older than `mod/*.mod`), G8 `smoke_ih_fit` 12/12, G9 `smoke_ls_baseline_qc` 10/10, G10 `smoke_ih_recovery` 14/14, G11 `regression_passive_identity.py` PASS on `$IH_ARCHIVE_GROUP/specimen_$REG_SPECIMEN`, G12 `noise_fit_gates.py ss-pulses` on `$NPZ_SPECIMENS`; then `GATES noise-fit: p passed, f failed, s skipped` and `VERDICT: SETTLED|NOT SETTLED -- <reason>` (SETTLED only in FULL mode with nothing failed or skipped). A check passes only on a line it must see (the suite's own summary line), never on an exit status. Logs in `$GATE_LOG_DIR` (default `~/ih_fit_gate_logs/<date_time>/`), never in the repository; no tracked file is written. Exit 0 iff nothing failed.
- Defaults: `IH_ARCHIVE_GROUP=/davinci-1/home/ldellamea/Human Neurons Fitting/L3_exc`, `REG_SPECIMEN=508282493`, `NPZ_SPECIMENS="508282493 569844159"`, `PYTHON=python`.
- Test oracles (`smoke_noise_fit_gates.py`): T1 planted CR and non-ASCII files flagged, nothing else; T2 a syntax error named, no `.pyc`; T3 a writer-like Format-A fixture (two sweeps of an ADC-quantised trace, spacings 200.000 and 200.040 ms, one pulse dropped) gives the exact expected summary and the gap; T4 Format B the same; T5 no `sweep_number`: three adjacency chains; T6 no false adjacency in 200 independent quantised pairs, a 50-sample overlap refused unless the minimum is lowered; T7 the runner's QUICK mode passes G0, G1, G3-G6 and cannot settle, a bad interpreter fails G0 with exit 1; T8 ASCII/LF. Mutation-tested 2026-10-20 (no overlap minimum, sweep numbers ignored, a wrong ms scale, `mod/` not scanned, a check certified on exit status: each caught).
- Status: smoke-tested 2026-10-20 (sandbox: T1-T8 8/8). The FULL runner was run in the sandbox on `main` + Block 2 + these tools, with `Biological Fit/` and `Synthetic Passive fit Test/` from `main` and a synthetic archive group; its result is in the decision log's changelog of 2026-10-20. Not run here: G11 on a real cell and G12 on the real archive (cluster). Run on the cluster 2026-10-20 (FULL, `main` at `ca028ea`, env `prova`): 13 passed, 0 failed, 0 skipped, VERDICT SETTLED; G11 PASS on 508282493; G12 read 508282493 and 569844159 (section 3).

## 6. Pipeline entry points

| Command | Purpose | Minimal configuration for testing | Expected runtime |
|---|---|---|---|
| `python smoke_noise_stats.py` | Block 1 smoke (N1-N6) | none | seconds |
| `python smoke_noise_law.py` | Block 2 smoke (M1-M9) | none | about 5 s |
| `python smoke_noise_fit_gates.py` | tools smoke (T1-T8) | none | about 15 s |
| `bash cluster_gates_noise_fit.sh` (`QUICK=1` for the fast half) | every cluster-side check of Blocks 1-2 and the SS archive read, one PASS/FAIL/SKIP line each and a VERDICT | the conda env active; on the cluster nothing else | about 15 min (QUICK < 1 min) |
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
- (added 2026-10-21) D-073's per-sweep calibration: which of (sigma_f, rho_f, sigma_s, tau_s, sigma_b) are estimated per sweep and which per cell; raw or partially pooled per-sweep values; the twins' protocol twinning (LS amplitudes of the real cell instead of the cohort's `--ls-hyp-amps`; SS sweeps as 37-pulse trains at the real stored positions). Evidence for the proposal: `sweep_identifiability.py` (standalone, not in the repository): from one sweep at theta_cf the fast pair is identified to 1-3 % (sigma_f) and +/-0.012-0.014 (rho_f), sigma_s to 18-30 % (6-13 % with tau_s held), tau_s only within a factor of about 2 (log-SD 0.49-0.72).
