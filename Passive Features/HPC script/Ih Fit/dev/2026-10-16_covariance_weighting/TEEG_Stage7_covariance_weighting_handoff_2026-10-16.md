# TEEG -- Stage 7 per-cell law fit with simulated inverse-covariance weighting (D-048): implementation handoff (2026-10-16)

| Date | Change |
|---|---|
| 2026-10-16 | Created for a new chat, from the noise chat of 2026-10-13/16 (bootstrap types, weights, covariance, condition number, missing-noise sensitivity). D-048 logged the same turn. Nothing here is implemented; the repo was **not** opened in that chat (the attach request was denied on 2026-10-13), so every file and function name below is taken from the decision log and the earlier handoffs, tagged [log], and must be checked against `main` before coding. |
| 2026-10-16 | Not in project knowledge: the upload was refused because project knowledge is full. This file lives in the repo at `Passive Features/HPC script/Ih Fit/dev/2026-10-16_covariance_weighting/`, committed by the user, with the sandbox checks of §6 under `checks/` and the bias document of 2026-10-11 under `bias_doc_2026-10-11/`. §0 and §6 now point there instead of to zips. The `.py` files of the bias document were made ASCII-only for the cluster (comments transliterated, non-ASCII plot labels written as `\u` escapes; the parsed code is identical). |

**Basis tags.** **[run]** executed in the assistant's sandbox (scripts in
`checks/` next to this file, §6); **[user]** the user's decision or report; **[KB]** project
knowledge; **[log]** the decision log's text; **[FT]** full text read through
PubMed; **[memory]** recalled, not read this chat; **[open]** not decided or
not verified.

## 0. Read first, in this order

| Doc | What for |
|---|---|
| `TEEG_decisions_and_ideas_log.md` (project **root**, not the stale `claude/` copy) | D-017 (the law), D-028 (textbook autocorrelation through statsmodels), D-033 (indirect inference, no ABC), D-034 (tilt carries the seconds-scale variance; enlarged SS windows), **D-048** (this handoff's decision). Re-read the last changelog row before logging: the diameter chat logs in parallel. **Next free IDs on 2026-10-16: D-049, I-003.** |
| `claude/TEEG_Stage7_noise_model_2026-09-28.md` (v3.2) | the law §3.3; why a window statistic is not a parameter §3.4; the inference and the 14 statistics §3.5 (cost eq. (14), cohort fit §3.5.3); tilt §3.6; enlarged SS windows §3.7; per-cell calibration §3.8 |
| `claude/TEEG_Stage7_noise_handoff_2026-10-01.md` | state of the Ih Fit code, paths, the `ls_baseline_qc.py` results, §5 implementation list items 1-7 (this handoff replaces its "weights" question, nothing else) |
| `claude/TEEG_Ih_fit_staged_plan.md` | stage structure and smoke-suite map |
| repo `Passive Features/HPC script/Ih Fit/README.md`, `CHANGELOG_ih_fit.md` | how each script runs on the cluster |

In the repo next to this file (`Passive Features/HPC script/Ih Fit/dev/2026-10-16_covariance_weighting/`; project knowledge was full on 2026-10-16):
`README.md` (layout, how to run); the check folders of §6 under `checks/`; the bias document
of 2026-10-11 under `bias_doc_2026-10-11/` (the content of `TEEG_bias_of_a_statistic_2026-10-11.zip`,
never saved to project knowledge). Still only sent to the user as files, ask for them if
needed: `TEEG_noise_theory_handoff_2026-10-09.md` with `TEEG_noise_theory_checks_2026-10-09.zip`.

## 1. What is decided and what is not

| Item | State | Basis |
|---|---|---|
| Estimator family | indirect inference: $\hat\theta(c)$ minimises a weighted squared distance between $c$'s statistics and their medians under the law, binding function by simulated medians through the pipeline's own estimators | D-033 [log] |
| Weighting | $W = \hat\Sigma^{-1}$, $\hat\Sigma$ the covariance of the statistic vector over $R$ cells simulated at the current $\hat\theta$, iterated from a first step weighted by the real data until the objective plateaus (§2) | **D-048 [user]** |
| Hand-set $\omega_m$ (0.005 mV SDs, 0.05 autocorrelations) | demoted to a labelled comparison; never the default | D-048 [log] |
| In-silico sensitivity check | required before the fit is trusted on real cells: twins with a noise component the law lacks or only approximates (the tilt of D-034) | **D-048 [user]** |
| $W_0$ ("weights derived by the real data") | **[open]** -- the assistant's reading: the cell's own window-level SEs on the diagonal, $W_0 = \mathrm{diag}(1/\widehat{\mathrm{SE}}_m^2)$; the alternative is the hand-set $\omega_m$. Ask the user in the first exchange. |
| $R$, plateau criterion, iteration cap | **[open]**; proposals in §2 |
| Statistic set | 14 of §3.5.3 (SS 10 ms window: $\hat\sigma$, $\hat\rho$ at 0.02, 0.1, 1 ms; LS 265 ms: the same four; LS first 50 ms: $\hat\sigma$; LS last 132.5 ms: $\hat\sigma$, $\hat\rho$ at 0.02, 0.1, 1, 10 ms) [KB]; D-034's enlarged SS windows proposed to join, **[open]** |
| Free parameters | $\theta = (\sigma_f, \rho_f, \sigma_s, \tau_s)$, $p$ = 4; $\sigma_b$ of the tilt is calibrated separately from the SS variogram (D-034.1), not inside this fit | D-017, D-034 [log] |
| D-017 open points (a), (b), (e), (f), (g) | unchanged | 2026-10-01 handoff §3 |

## 2. The procedure to implement

Notation: cell $c$; statistic $m = 1..d$ with window length $n_m$ (samples) and
$K_m$ windows; $\hat q_m(c)$ = median over the $K_m$ windows of statistic $m$
computed on the real data (computed level); $q_m(\theta) = \operatorname{med}(T_m \mid \theta, n_m)$
the single-window median under the law (analytic level, estimated by simulation
with common random numbers so that the objective is a deterministic function of
$\theta$, §3.5.2 of the topic doc); $\Sigma(\theta, \mathbf n, \mathbf K) =
\operatorname{Cov}(\hat{\mathbf q} \mid \theta, \mathbf n, \mathbf K)$, $d \times d$
(analytic level); $\hat\Sigma$ its sample estimate from $R$ simulated cells
(computed level).

```
inputs: q_hat(c) [d], n_m, K_m, W_0
k = 0; theta_0 = argmin_theta (q(theta) - q_hat)' W_0 (q(theta) - q_hat)
repeat
    simulate R cells at theta_k, each with c's own n_m and K_m, through the pipeline's estimators (D-028)
    Sigma_k = sample covariance (ddof 1) of the R statistic vectors [d x d]
    W_{k+1} = inv(Sigma_k)            # after the diagnostics of section 3
    theta_{k+1}, J_{k+1} = argmin / min of (q(theta) - q_hat)' W_{k+1} (q(theta) - q_hat)
    k += 1
until |J_k - J_{k-1}| <= tol_J * max(1, J_k)  and  max_i |theta_k,i - theta_{k-1,i}| / scale_i <= tol_theta,  or k == k_max
write theta_k, J_k, k, R, kappa(C_k), the per-iteration trace
```

Proposals for the open knobs, each to be confirmed and logged:

| Knob | Proposal | Why |
|---|---|---|
| $W_0$ | $\mathrm{diag}(1/\widehat{\mathrm{SE}}_m^2)$ from a window-level bootstrap of the cell's own $K_m$ windows (B = 2000) | the per-cell *full* bootstrap covariance is useless (rank $\le K_m - 1$; the window-level bootstrap covariance had median $\kappa$ $8.2 \times 10^6$ and gave a fit as poor as the hand weights, 16.3 ms vs 17.1 ms) but its *diagonal* is fine: in the check, diagonal $1/\mathrm{SE}^2$ gave 15.5 ms against 17.0 ms hand-set [run, `check_cov_source.py`, `check_weights.py`] |
| $R$ | 600 for the iterations, 3000 for the final $\hat\Sigma$ | two-step at $R$ = 600 matched the oracle (7.25 vs 7.61 ms); the Monte Carlo error of $\hat\Sigma^{-1}$ in the cost was within 0.76-1.36 at $R$ = 600 and 0.88-1.14 at $R$ = 3000 [run, `check_cov_source.py`, `check_cov_structure.py` A] |
| tol_J, tol_theta, k_max | $10^{-2}$ relative on $J$, $10^{-2}$ on each $\theta_i$ relative to its scale, $k_{\max}$ = 6 | [reasoning]; the two-step already captures most of the gain, so the plateau should come at $k$ = 2-3; a cell that does not plateau is flagged, not forced |
| the $\chi^2$ reading of $J_k$ | report $J_k$ and $P(\chi^2_{d-p} > J_k)$ per cell; flag below 0.01 | with $W = \Sigma^{-1}$ at the truth, $J$ had mean 9.22 against $\chi^2_9$'s 9 (10 statistics, 1 free parameter) [run]; the $\chi^2_{d-p}$ law of the minimised objective is Hansen's $J$ test, stated for the asymptotic regime [memory; Kaczmarska et al. cite Hansen 1982 and Hall 2005 for consistency, FT] -- treat as a calibrated alarm, not a test |

Why the iteration is worth it (numbers from the ten LS statistics, $K$ = 6, only
$\tau_s$ free at $\hat\theta_{\rm sim}$, 1500 simulated cells) [run,
`check_weights.py`]: robust SD of $\hat\tau_s$ 16.98 ms hand-set, 15.53 ms
diagonal, **7.57 ms** full $\Sigma^{-1}$. The gain comes from the contrasts
between autocorrelations of the same window (correlations 0.993-0.997;
participation ratio of the correlation matrix 2.07 of 10): the diagonal
weighting double-counts nine statistics that move together. With all four
parameters free the gain was **not measured** [open].

The iterated form of the two-step is known in the GMM literature as iterated
GMM (Hansen, Heaton & Yaron 1996) [memory, not read; PubMed returned nothing
usable for the phrase on 2026-10-16]. Nothing here depends on that reference.

## 3. Diagnostics that must run inside the loop

| Diagnostic | Rule | Basis |
|---|---|---|
| rank and conditioning of $\hat\Sigma_k$ | refuse to invert if rank $< d$; report $\kappa(\hat C_k)$ of the **correlation** matrix (unit-free; $\kappa(\hat\Sigma)$ depends on mV vs $\mu$V: $6.7 \times 10^4$ vs $3.6 \times 10^6$ for the same ten statistics, while the fit is unit-invariant) | [run, `check_cov_structure.py` E, P18] |
| what the thin contrasts are | print the loadings of the smallest eigenvector of $\hat C_k$: for the ten LS statistics it is mainly $\hat\rho(0.1\,\mathrm{ms}) - \hat\rho(0.02\,\mathrm{ms})$ on the 265 ms window, with $\mu_{\min}$ = 0.0025 | [run] |
| misfit alarm | $J_k$ against $\chi^2_{d-p}$; a missing noise source raises $J$ far before it ruins $\hat\theta$ (+1 % variance per statistic: mean $J$ 8.9 $\to$ 18.8, $\chi^2_9$'s 95 % point 16.9; robust SD of $\hat\tau_s$ 6.3 $\to$ 8.7 ms, oracle 7.8 ms) | [run, `check_missing_noise.py`] |
| reported precision | **do not** report $(G^\top \hat\Sigma^{-1} G)^{-1/2}$ as the SD of $\hat\theta$: it was optimistic even when the law is true (5.07 ms linearised vs 6.38 ms Gaussian-input fit vs 7.57 ms simulated medians) and does not move when the law is wrong | [run, `check_cov_structure.py` P12/P15; `check_missing_noise.py`] |
| sign of a model error | a noise source the simulator lacks always *understates* $\hat\Sigma$ along the contrasts it reaches, and understating is the unbounded direction: the weight on an axis becomes $1/(1 - \kappa\varepsilon)$ times too large (overstating: $1/(1 + \kappa\varepsilon)$, bounded) | [run, `fig_condition_number.py`; derivation in the chat of 2026-10-14] |

## 4. The in-silico sensitivity check (D-048, "during the implementation")

Design, to be agreed with the user before coding; predictions fixed before each
run, as in every check of §6.

| Step | What | Truth known | Measured |
|---|---|---|---|
| S0 | closure: twins from the law (D-017.1) at $\hat\theta_{\rm sim}$ with the real $n_m$, $K_m$ (5-6 LS windows, the SS windows of D-034), fitted by §2 | $\theta$ | bias and robust SD of each $\hat\theta_i$ over $\ge$ 300 twins; $J$ against $\chi^2_{d-p}$; iterations to plateau |
| S1 | the tilt as an approximation: twins carrying a seconds-scale OU (the alternative D-034 rejected: $\sigma$ 0.142-0.293 mV at $\tau$ 3-30 s, matched to the cohort's 6.89 s change [run, 2026-10-01 handoff §4.6]) instead of the tilt, fitted with the law that has the tilt | $\theta$ | the same, plus which statistics absorb the error |
| S2 | a component the law lacks: a weak sweep-locked component (not excluded in 5 of 18 cells [run, cluster 09-28]); independent extra scatter added to each statistic (the `check_missing_noise.py` model, +0.1, 1, 10 % variance) as the generic case | $\theta$ | the same; the $J$ alarm's hit rate at each level |
| S3 | real cells: $J_k$ and its $\chi^2$ tail per cell on L3_exc; the cells with the trend share above its null (526785799, 571430815) read separately | -- | which cells the alarm flags |

S0 is the closure smoke of the 2026-10-01 handoff §5 item 6, now with the
D-048 weighting. S1 and S2 answer the user's question ("how much a not
captured or only approximated noise component affects the estimation of the
law"); their numbers decide whether the alarm threshold and $k_{\max}$ are
adequate.

## 5. Where it goes in the code [log; not verified against the repo this chat]

| Piece | Where | Note |
|---|---|---|
| per-cell fit, the loop of §2 | `Ih Fit/noise_calibration.py` or a module beside it (D-033) | the binding function with common random numbers already exists in the sandbox scripts (§6, `check_weights.py`: `windows_crn`, `binding`, `stats10`) and in the 2026-10-08 scripts zip; port, do not rewrite |
| estimators | `noise_stats.sample_acf` (statsmodels, `adjusted=False`, D-028) on data and simulations alike | the sandbox used a NumPy copy asserted equal to statsmodels on a few windows |
| generator | `synthetic_ground_truth.add_recording_noise`, `NoiseConfig` (D-017); the tilt through `drift_sigma_mV` (D-034) | the fit's simulator must be the twins' generator, or a function both call, so that S0 closes by construction |
| output columns | the noise table beside the existing columns, names that cannot collide with `noise_ss_*` (handoff §5 item 2): `noise_fast_sigma_mV`, `noise_fast_rho_lag1`, `noise_slow_sigma_mV`, `noise_slow_tau_ms`, plus `noise_fit_J`, `noise_fit_J_pchi2`, `noise_fit_iters`, `noise_fit_R`, `noise_fit_kappa_corr`, `noise_fit_weighting` (`sigma_inv` default; `diag_se`, `hand_omega` as comparisons) | the decided default is the default of the script (D-048) |
| smoke | `smoke_ih_recovery.py` or a new `smoke_noise_fit.py`: S0 closure, assertion of the default weighting, rank refusal, the $\chi^2$ alarm on a planted misfit | |
| delivery | git patch against current `main`, `git apply --check` on a fresh clone, smoke on that clone, ASCII-only Python, LF-only shell; `hpc-python-compat` and `hpc-git-delivery` skills | 2026-10-01 handoff §5 |

Cost on the cluster [reasoning, from sandbox timings]: one binding-function
grid (36 $\tau_s$ values, 3000 windows each, ten statistics) took about 60 s on
two cores; one $\hat\Sigma$ at $R$ = 600, $K$ = 6 took about 5 s. With four
free parameters the binding function is evaluated inside the optimiser, not
on a grid: budget it before choosing $R$ and $k_{\max}$.

## 6. The checks behind every number here (sandbox runs; scripts in `checks/` next to this file)

| Folder in `checks/` (zip it came from) | Scripts | What they established |
|---|---|---|
| `weights_1013/` (`TEEG_covariance_checks_2026-10-14.zip`, which superseded `TEEG_weights_check_2026-10-14.zip`) | `check_weights.py` (P1-P3), `check_cov_source.py` (P4-P6), `check_sigma_theta.py` (P7-P9), `check_conditioning.py`, `check_cov_structure.py` (P10-P18), `check_missing_noise.py` (PM1-PM3), `boot_model.py` | hand vs diagonal vs full weighting; where $\hat\Sigma$ can come from; $\Sigma$ depends on $\theta$ (SE of the 1 ms autocorrelation median changes by 1.61$\times$ over $\tau_s$ 20-80 ms); perturbation bound; Monte Carlo error vs $\kappa$; missing-noise sensitivity. Failed predictions kept in the docstrings: P12/P15 (linearised SD optimistic for $\Sigma^{-1}$). |
| `boot_1013/` (`TEEG_bootstrap_types_checks_2026-10-13.zip`) | `check_boot_types.py`, `fig01_two_worlds.py`, `fig02_block_length.py` | iid resampling of samples destroys the autocorrelation (SD $\approx 1/\sqrt n$); block bootstraps reach at most two thirds of the true SD; window-level bootstrap of the median over $K$ = 6 is unbiased in SE but spread 0.58-1.52 |
| `boot_err_1014/` (`TEEG_boot_errors_check_2026-10-14.zip`) | `check_boot_errors.py`, `check_info_vs_scheme.py` | sampling vs internal error of a bootstrap SE (E&N §3): internal RMS $\propto 1/\sqrt B$, 0.019 at B = 2000; E&N's jackknife estimate matches it |
| `cond_fig_1014/` (`TEEG_condition_number_fig_2026-10-14.zip`) | `fig_condition_number.py` | the two readings of $\kappa$; the worst-case perturbation reaches the bound exactly |

All run with Python 3, NumPy 2.5.3, SciPy 1.18.1; every script has its
predictions in the docstring and its log next to it. They are standalone
checks, not pipeline code. Paths are resolved from each script's own location;
`cond_fig_1014` reads `../weights_1013/results/weights.npz`, so the folders must
stay siblings. `boot_1013` also needs `statsmodels` and `arch`.

## 7. Traps

- **Statistic is not parameter; computed is not analytic.** $\hat q_m(c)$ is one number from one cell; $q_m(\theta)$ and $\Sigma(\theta, \mathbf n, \mathbf K)$ are properties of the law; $\hat\Sigma$ is their Monte Carlo estimate. Keep the three apart in names.
- **$\hat\Sigma$ from the cell's own windows is not an option** (rank $\le K_m - 1$); only its diagonal, for $W_0$.
- **Units.** $\kappa(\hat\Sigma)$ is meaningless across unit choices; report $\kappa(\hat C)$.
- **The nominal SD from $\hat\Sigma$ is not the precision of $\hat\theta$.** Precision comes from the closure run S0 (and, per cell, from a parametric bootstrap at $\hat\theta(c)$ if wanted; its plug-in error is set by $\hat\theta(c)$, not by $B$).
- **$J$ is an alarm, not a verdict.** Its $\chi^2_{d-p}$ reading assumes the law is true and $R$ large; S2 calibrates it.
- **Two copies of the decision log** (root is live); **two copies of the paths reference** (both about the H01 pipeline).
- **`noise_ss_*` means Square Subthreshold**, not Short Square.

## 8. First exchange with the user, in order

1. Confirm $W_0$ (diagonal of the cell's own window-level SEs, or the hand $\omega_m$); log the answer (D-049).
2. Confirm $R$, the plateau tolerances and $k_{\max}$; log.
3. Confirm the statistic set, including whether D-034's enlarged SS windows join (D-034 "what it does not decide"); log.
4. Agree S0-S3 of §4 and their predictions before writing them.
5. Open the repo (`Leonardodm00/Towards-EEG`, `main`), verify every name in §5, then start from the 2026-10-01 handoff §5 item 1.
