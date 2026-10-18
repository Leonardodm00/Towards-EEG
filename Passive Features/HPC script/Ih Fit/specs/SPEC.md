# SPEC -- Ih Fit / Stage-7 per-cell noise-law fit (D-017, D-048 to D-056)

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
content of `IhFit_main_2026-10-16.zip`, a git archive of `main`)

## 0. Coverage

Paths this spec covers (relative to `Passive Features/HPC script/Ih Fit/`):

- `noise_stats.py` (Block 1)
- `smoke_noise_stats.py` (Block 1)
- the two lag-1 estimator lines of `passive_fitting_hpc_fixed.py`
  (`_estimate_noise`, `_estimate_residual_noise_at_mle`) and the helpers
  `_estimator_band`, `_acf_gap_band` of `smoke_ih_recovery.py` (Block 1)
- `noise_calibration.lag_autocorr` (Block 1); the rest of
  `noise_calibration.py` is NOT yet covered
- (to be added block by block) `noise_law.py`, `noise_windows.py`,
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
quantity under the stationary part of (L1) (closed form, topic doc eq. 25);
h* = 0.8 s PROVISIONAL (D-051). Alternated with theta_hat at every iteration.

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
| theta | (sigma_f, rho_f, sigma_s, tau_s) | float[4] | mV, --, mV, s |
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

## 6. Pipeline entry points

| Command | Purpose | Minimal configuration for testing | Expected runtime |
|---|---|---|---|
| `python smoke_noise_stats.py` | Block 1 smoke (N1-N6) | none | seconds |
| `python smoke_ih_recovery.py`, `python smoke_ih_fit.py`, `python smoke_ls_baseline_qc.py` | existing suites, re-run after Block 1 (R8/R10 bands now through `sample_acf`) | NEURON + compiled `mod/` | minutes |
| `python regression_passive_identity.py --ref-dir "../Biological Fit" --archive-cell <dir>` | bit-for-bit gate against the oracle folder | the oracle folder and one archive cell | minutes |

## 7. Not executable in the sandbox

- `regression_passive_identity.py`: `Biological Fit/` is not in the zip baseline and the archive is on the cluster; skipped here, to be run on the cluster (command in README).
- Anything reading the real archive (`noise_calibration.py --group-dir`, S3).

## 8. Open questions

- Does `ss_pulses.npz` carry sweep numbers? The loader reads none (D-056 open point; cluster check requested 2026-10-18).
- Root `specs/SPEC.md` Coverage pointing to this file (user's call, D-056 (ii)).
