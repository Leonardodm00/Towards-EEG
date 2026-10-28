# Recording noise for the Stage-7 synthetic twins: two timescales, the variance at seconds, and how the noise law is inferred

**Date:** 2026-09-28; **v3, 2026-10-08**, v3.1 and v3.2 2026-10-09, v3.3 2026-10-10, v3.4 2026-10-16 (body rewritten to the current approach; the changelog
says what changed). **Project:** Towards EEG -- I_h-aware passive fit (`Passive Features/HPC
script/Ih Fit/`), Stage 7 (synthetic recovery gate). **Decision records:** D-005, D-006, D-011,
D-012, D-014 to D-017, D-028, D-029, D-033, D-034 and D-048 of `TEEG_decisions_and_ideas_log.md` (project
root). **Implementation state:** the per-cell noise calibration (§3.8) is not written; the
textbook autocorrelation (D-028, D-029) exists as a patch on `main` at `b355d8e`, pending the
user's push and the cluster gates; the generator in the repository still injects the
single-timescale law of D-014 (§3.1.3); the weighting of D-048 (§3.5.6) is not implemented. [Checked
2026-10-16 **[src]**: the laptop clone's `Ih Fit` on `main`, pulled that morning, holds no
`noise_stats.py`, so the D-028 patch has not reached `main` there; `noise_calibration.py` measures
per-specimen noise and fits nothing yet; the cluster's copy was not checked.]

| Version | Date | Change | Basis |
|---|---|---|---|
| v1 | 2026-09-28 | Created. | [run; cluster] |
| v2 | 2026-09-28 | Revised after two independent checks: the closed-form shortcut checked by Monte Carlo and replaced by simulated medians; five statements of D-017 corrected. | [run] |
| v2, markers | 2026-10-08 | Superseded passages marked (D-028, D-033, D-034, D-017 note 4). | [log] |
| **v3** | 2026-10-08 | **Body rewritten to the current approach at the user's request; superseded text is removed, not marked.** What changed: (i) the sample autocorrelation is the textbook estimator, through statsmodels (D-028, D-029): (3) and (10) restated with it, (14) generalised to either way of computing the model statistics, every autocorrelation value converted exactly by (18), the cohort law and the single-AR(1) comparison refitted; v2's $\tau_s$ (37.3 ms) lies inside the refit's Monte Carlo spread over five draws (35.6-45.6 ms) (§3.5.3). (ii) The per-cell law is fitted by our own indirect inference, point estimates, no ABC (D-033). (iii) Creep versus wander is settled by the LS tail over 6.89 s (D-017 note 4): no cell continues its pre-window trend; the two high-trend cells are slow wander. v2's per-cell creep scenarios, its `vm_delta_mv` route and its four-law trend-share table are removed; its account of a baseline error against the I_h sag is condensed into §3.6.2. (iv) 17 or 18 of the 18 cells carry more variance at seconds than the two-component law allows; the per-sweep tilt carries it, calibrated per cell from the semivariogram of SS window means (D-034; §3.6); model order is not selected per sweep (§3.5.4). (v) The SS noise windows are enlarged where the pulse transient has died out (D-034; §3.7). (vi) New: demeaning as a filter and the zero-sum identity (§3.4.1-3.4.2); window length against bias and precision (§3.4.3); what resting noise is physically (§3.3.3); the Monte Carlo spread of the fit (§3.5.3); how the tilt is matched to the loss (§3.6.5). (vii) v2's Bessel lag-1 values (0.49/0.59/0.59) were not reproduced by any digital realisation tried and are replaced by the range over those realisations (§3.3.3); notation and glossary follow the new material. Numbers under a law are recomputed under the textbook cohort law of §3.5.3 unless stated. An independent agent checked the draft against the computation logs; its findings were corrected before publication. | [user; log; run; cluster] |
| v3.1 | 2026-10-09 | §3.3.3 and §4 item 4: a resonant membrane is still a continuum of sinusoids, concentrated around $\pm\nu_0$; its autocorrelation is a damped oscillation, a pure damped cosine only for two Lorentzian lines at $\pm\nu_0$ (a noise-driven damped oscillator adds a sine term); the trough-time bound is removed, the first-zero bound kept. | [run; reasoning] |
| v3.2 | 2026-10-09 | §3.3.3, §1, §5, §6 and Appendix B: the resonant frequency reported by Moradi Chameh et al. is the impedance peak $\nu_p$ (ZAP stimulus), not the ringing frequency $\nu_0$ of the autocorrelation; for a noise-driven oscillator $\nu_0^2 = \nu_p^2 + (2\pi\tau)^{-2}$, so v3.1's step from "few above 2 Hz" to a first zero no earlier than 0.125 s did not hold. Replaced by that relation and worked cases (for $\nu_p$ = 2 Hz the first zero falls at 96-132 ms for the oscillator and at 99-121 ms for two Lorentzian lines, $\tau$ = 50-100 ms), with the 265 ms window's median sample autocorrelation at those lags; "the centre of the peak gives the oscillation" restated (the spectral peak lies below $\nu_0$); "every sample autocorrelation" restated as the median. Found by an independent check of the theory handoff of 2026-10-09. | [KB; derived; run] |
| v3.3 | 2026-10-10 | §1, §2, §3.4.3, §4 item 6, §5, §6 and Appendix B: (24b) added, the information on $1/\tau$ that a demeaned OU window keeps (its REML information). (24) ignores each window's own mean, which costs between $\tau^2/2$ and $2\tau^2$ of information per window. The precision of 5-6 LS pre-windows is restated as 22-25 % by (24) and at least 27.5-30.1 % for unbiased estimators computed from the demeaned-window statistics of (12) (17 SS sweeps of enlarged windows: 3.5-4.8 % and at least 4.6-7.6 %); the contrast information reproduces the widening of REML's spread in Check D's second run. Derived and checked in `TEEG_fisher_information_2026-10-10.md` (closed form against dense explicit-contrast computations, error falling as $\Delta t^2$; independently reviewed). | [derived; run] |
| v3.4 | 2026-10-16 | Built on the project's v3.3 (the copy attached on 2026-10-16 was v3.1; diffed, it differs from v3.3 only by the v3.2-v3.3 changes). **New §3.5.6, eqs. (28)-(30)**: the per-cell fit weights its statistics by $\hat\Sigma^{-1}$, the covariance of the statistic vector simulated at the current estimate and refined iteratively from a first step built from the cell's own data (D-048); why (robust SD of $\hat\tau_s$ 7.57 ms against 15.53 ms diagonal and 16.98 ms hand-set, only $\tau_s$ free); why $\hat\Sigma$ must come from simulation; the two readings of the condition number and the perturbation bound; Monte Carlo error of $\hat\Sigma$ not amplified by it; sensitivity to a missing noise source and to one biased statistic; the linearised precision optimistic. **§3.4 and §3.4.2 [corrected]**: the expectation of (3) has no *elementary* closed form but an exact one-dimensional integral for Gaussian windows; demeaning is the larger of two sources of its bias (62 % in the worked case), the ratio's weighting by $S_0$ most of the rest (from `TEEG_bias_of_a_statistic_2026-10-11.md`). **§3.5.1-3.5.2, §3.5.5, §3.8.1, §5 (d) and the approximations [corrected]**: the hand-set $\omega_m$ are no longer the per-cell default. Header: implementation state checked against the laptop clone. §1, §2, §3.8.2, §4 item 13, §5 (h), §6 and Appendix B follow. Every number checked against the logs of the checks before writing. | [user; run; derived; src; PubMed FT; KB] |

**Abstract.** Stage 7 asks whether the six-parameter I_h fit recovers a known ground truth from
synthetic twins of real Allen layer-3 excitatory (L3_exc) cells; the answer is only as
trustworthy as the recording noise the twins carry. This document states the noise law the
twins carry -- a fast AR(1) component, a slow Ornstein-Uhlenbeck (OU) component, a per-sweep
offset and a per-sweep linear tilt (D-017, D-034) -- and how it is inferred from the statistics
the pipeline computes on windows of 10-265 ms: indirect inference with the textbook sample
autocorrelation (D-028, D-033), the statistics weighted by their simulated inverse covariance, refined
iteratively (D-048). It establishes (i) why a single AR(1) fails on L3_exc; (ii) why a
window statistic is not a parameter: demeaning acts as a high-pass filter, so a window shows only
part of a slow component and its autocorrelation is biased low; (iii) how long windows must be
relative to a timescale, for bias and for precision; (iv) what the 6.89 s LS tail showed: no
creep, but more variance at seconds than the two-component law allows, in 17 or 18 of the 18 cells; (v) how the
per-sweep tilt is calibrated to carry that variance, and what a tilt cannot reproduce; (vi) the
enlarged SS windows that measure it; (vii) what the noise is, physically, at rest; (viii) why the
statistics are weighted by their inverse covariance, and what that weighting is sensitive to. **Covered:**
the noise model, its estimation and identifiability, the seconds scale, the open points.
**Excluded:** the code of the per-cell calibration (not written), the depolarising-sweep spike
screen (D-016), the I_h mechanism and the fitter, and every group other than L3_exc.

---

## 1 Notation and symbols

| Symbol | Name / Meaning | Type & domain | Units | First used in § |
|---|---|---|---|---|
| $c$; $\mathcal C$; $c'$ | a real archive cell; the L3_exc cohort (18 cells); the synthetic twin of $c$ | element of $\mathcal C$; finite set; a synthetic cell | -- | 3.1.1 |
| $C_m, R_m, R_a$ | specific membrane capacitance, specific membrane resistance, axial resistivity (fitted) | $\mathbb R_{>0}$ each | µF/cm², Ω·cm², Ω·cm | 3.1.1 |
| $\bar g_h, \Delta v_h, \kappa_\tau$ | I_h density, shift of its activation curve, scale of its time constant (fitted) | $\mathbb R_{\ge 0}$; $\mathbb R$; $\mathbb R_{>0}$ | S/cm²; mV; dimensionless | 3.1.1 |
| $t$ | time; origin stated where used (Conventions) | $\mathbb R$ | s | 3.1.2 |
| $t_0$ | first sample of a stored sweep | $\mathbb R$ | s | 3.1.2 |
| $t_{\rm on}, t_{\rm off}$ | onset and end of a Long Square step | $\mathbb R$ | s | 3.1.2 |
| $w(t)$ | time weight of the SS loss (configured) | $(0, 1]$ | dimensionless | 3.1.2 |
| $N$ | number of SS pulses averaged into one bundle | $\mathbb N_{\ge 1}$ | -- | 3.1.2 |
| $\sigma$, $\rho$, $\tau$ | SD, lag-1 coefficient and correlation time of a generic AR(1) or OU component ($\tau$ in §3.3.3's resonant forms: the envelope's decay time) | $\mathbb R_{>0}$; $(-1, 1)$; $\mathbb R_{>0}$ | mV; dimensionless; s | 3.1.2 |
| $\hat\theta_{\rm sim}$, $\hat\theta_{\rm cf}$ | cohort estimates of $\theta$ by simulated medians and by closed form (computed level; diagnostics only) | $\Theta$ | mixed | 3.1.2 (defined 3.5.3) |
| $f_s$; $\Delta t$ | sampling rate; sampling interval $1/f_s$ | $\mathbb R_{>0}$ | Hz; s | 3.2.1 |
| $t_{\rm SS}(c)$ | start of cell $c$'s enlarged SS window after the previous pulse | $[10, 200)$ | ms | 3.2.1 |
| $n$; $T$ | window length; window duration $n\,\Delta t$ | $\mathbb N_{\ge 2}$; $\mathbb R_{>0}$ | samples; s | 3.2.1 |
| $i, j$ | sample indices, $\{1, \dots, n\}$ in a window; $j = 0, 1, \dots$ in a sweep in (5a)-(5b) | $\mathbb N$ | -- | 3.2.2 |
| $x_j$; $\mathbf x$; $\bar x$; $u_j$ | voltage sample $j$ of a window (LJP-corrected); the window as a vector; its mean; $x_j - \bar x$ | $\mathbb R$; $\mathbb R^n$; $\mathbb R$; $\mathbb R$ | mV | 3.2.2 |
| $S_0$; $S_L$ | $\sum_j u_j^2$; $\sum_{j=1}^{n-L} u_j u_{j+L}$ | $\mathbb R_{\ge 0}$; $\mathbb R$ | mV² | 3.2.2 |
| $L$; $\ell$ | lag of a statistic (reported only for $n - L \ge 10$, D-029); physical lag $L\,\Delta t$ | $\{1, \dots, n - 1\}$; $\mathbb R_{\ge 0}$ | samples; s | 3.2.2 |
| $\hat\sigma$ | sample SD of a window, ddof 1 (computed level) | $\mathbb R_{\ge 0}$ | mV | 3.1.3 (defined 3.2.2) |
| $\hat\rho(L)$ | textbook sample autocorrelation of a window, eq. (3) (computed level) | $[-1, 1]$ | dimensionless | 3.2.2 |
| $\hat\rho_{\rm adj}(L)$ | the pre-D-028 estimator, used only to convert old values by (18) | $\mathbb R$ | dimensionless | 3.2.2 |
| $\hat{\mathbf x}^{\rm lin}$; $\operatorname{var}(\cdot)$ | least-squares line through a window; sample variance over a window's entries (ddof 0) | $\mathbb R^n$; operator | mV; mV² | 3.2.2 |
| $s_{\rm trend}$; $s_{\rm lock}$ | trend share of a window, eq. (4); locked share of a cell, D-016.4 (computed level) | $[0, 1]$; $\mathbb R$, at most 1 | dimensionless | 3.2.2 |
| $K$ | number of single LS sweeps (or SS pulses) of one cell | $\mathbb N_{\ge 1}$ | -- | 3.2.2 |
| $\Delta_{\rm end}$ | LS tail change of one sweep: end-window mean minus pre-window mean, signed (computed level) | $\mathbb R$ | mV | 3.2.2 |
| $\hat b$; $\Delta_{\rm creep}$ | least-squares slope of one pre-window; $\hat b \times 6.89$ s, what a continuing creep would give | $\mathbb R$ | mV/s; mV | 3.2.2 |
| $e_j$ | total noise at sample $j$ of a sweep, (5a) | $\mathbb R$ | mV | 3.3.1 |
| $e^{\rm fast}_j, e^{\rm slow}_j$; $e^{\rm stat}_j$, $e^{\rm stat}(t)$ | fast and slow stationary components; their sum, at sample $j$ or time $t$ | $\mathbb R$ | mV | 3.3.1 |
| $\sigma_f, \rho_f, \tau_f$ | fast component: SD; lag-1 coefficient at $\Delta t$; correlation time $-\Delta t/\ln\rho_f$ | $\mathbb R_{>0}$; $(0, 1)$; $\mathbb R_{>0}$ | mV; dimensionless; s (quoted in µs) | 3.3.1 |
| $\sigma_s, \tau_s, \rho_s$ | slow component: SD; correlation time; lag-1 coefficient $\mathrm e^{-\Delta t/\tau_s}$ | $\mathbb R_{>0}$; $\mathbb R_{>0}$; $(0, 1)$ | mV; s (quoted in ms); dimensionless | 3.3.1 |
| $\eta_j, \eta'_j$ | independent Gaussian innovations of the two recursions | $\mathbb R$ | mV | 3.3.1 |
| $\mathcal N(0, \cdot)$ | normal law, mean 0, variance as argument | law on $\mathbb R$ | -- | 3.3.1 |
| $\theta$; $\Theta$ | noise-law parameters $(\sigma_f, \rho_f, \sigma_s, \tau_s)$; their space $\mathbb R_{>0} \times (0,1) \times \mathbb R_{>0} \times \mathbb R_{>0}$ | $\theta \in \Theta$ | mixed | 3.3.1; $\Theta$ 3.5.1 |
| $k$ | generic lag | $\mathbb Z$ | samples | 3.3.1 |
| $\gamma(k \mid \theta)$ | autocovariance of $e^{\rm stat}$ at lag $k$, for each fixed $\theta$ (analytic level) | $\mathbb R$ | mV² | 3.3.1 |
| $a$; $b$ | per-sweep offset; per-sweep tilt | $\mathbb R$ | mV; mV/s | 3.3.1 |
| $t_j$; $\bar t$; $T_{\rm sw}$ | time of sample $j$ from the sweep's start; the sweep's mid-time; a generated trace's duration | $\mathbb R$; $\mathbb R$; $\mathbb R_{>0}$ | s | 3.3.1; $T_{\rm sw}$ 3.3.2 |
| $\sigma_a$; $\sigma_b(c)$ | SD of $a$; SD of $b$ for cell $c$ (analytic level; estimated by (26)) | $\mathbb R_{\ge 0}$ | mV; mV/s | 3.3.2 |
| $\delta$; $\sigma_\delta$ | tilt span $b\,T_{\rm sw}$; its SD $\sigma_b T_{\rm sw}$, the generator's `drift_sigma_mV` | $\mathbb R$; $\mathbb R_{\ge 0}$ | mV | 3.3.2 |
| $h$; $\lvert h\rvert$ | lag between two times or two window centres (signed in (19)-(21)); its magnitude | $\mathbb R$; $\mathbb R_{\ge 0}$ | s | 3.3.2 |
| $Q_i$, $Q$; $\tilde t_i$; $\lambda$ | charge of random kick $i$, and a generic kick charge distributed as each $Q_i$; its arrival time; the kicks' Poisson rate | $\mathbb R$; $\mathbb R$; $\mathbb R_{>0}$ | C; s; s⁻¹ | 3.3.3 |
| $C_{\rm cell}$; $R_{\rm cell}$; $\tau_m$ | whole-cell capacitance; input resistance; $R_{\rm cell}C_{\rm cell}$ for an isopotential membrane | $\mathbb R_{>0}$ | F; Ω; s | 3.3.3 |
| $\mathcal V(t)$; $\mathcal V_{\rm rest}$ | membrane voltage; its resting mean | $\mathbb R$ | mV | 3.3.3 |
| $\mathbb E$, $\operatorname{Var}$, $\operatorname{Cov}$; $\mathbb 1$ | expectation, variance, covariance over the random quantities of the equation where they appear (the noise law with the window fixed, unless stated: the kicks in (19), the frequency in (21)); indicator | operators | -- | 3.3.3 |
| $\nu$; $P(\nu \mid \theta)$ | frequency; two-sided power spectral density of the law's stationary part | $\mathbb R$; $\mathbb R_{\ge 0}$ | Hz; mV²/Hz | 3.3.3 |
| $\gamma_{\rm OU}(h \mid \sigma, \tau)$; $P_{\rm OU}(\nu \mid \sigma, \tau)$ | autocovariance and two-sided spectral density of one OU component (analytic level) | $\mathbb R$; $\mathbb R_{\ge 0}$ | mV²; mV²/Hz | 3.3.3 |
| $\nu_c$; $\nu_0$; $\nu_p$ | Lorentzian corner $1/(2\pi\tau)$; the oscillation (ringing) frequency of a resonant autocorrelation; the frequency at which the impedance peaks, the resonant frequency reported experimentally | $\mathbb R_{>0}$ | Hz | 3.3.3 |
| $F$ | a random frequency drawn from $P_{\rm OU}(\nu \mid \sigma, \tau)/\sigma^2$, a Cauchy law | $\mathbb R$ | Hz | 3.3.3 |
| $\alpha, \beta$ | opening and closing rates of a two-state channel | $\mathbb R_{>0}$ | s⁻¹ | 3.3.3 |
| $z$; $v(z)$ | $T/\tau$; visible share, eq. (8) | $\mathbb R_{>0}$; $[0, 1)$ | dimensionless | 3.4.1 |
| $x(t)$ | a stationary process in continuous time; $\bar x$ is then its mean over $[0, T]$ | $\mathbb R$ | mV | 3.4.1 |
| $H_T(\nu)$; $\nu_{1/2}$ | power transfer of demeaning a window of duration $T$; its half-power frequency $0.443/T$ | $[0, 1]$; $\mathbb R_{>0}$ | dimensionless; Hz | 3.4.1 |
| $\zeta_j$ | $\sum_{i=1}^{n}\gamma(\lvert i - j\rvert \mid \theta)$ | $\mathbb R$ | mV² | 3.4.2 |
| $\rho(\ell \mid \theta)$ | true autocorrelation of the law at lag $\ell$ (analytic level) | $[-1, 1]$ | dimensionless | 3.4.2 |
| $T_{\rm tot}$; $\hat\tau$; $N_{\rm w}$ | total observed duration; an estimate of $\tau$ (computed level); number of windows a total duration is cut into | $\mathbb R_{>0}$; $\mathbb R_{>0}$; $\mathbb N_{\ge 1}$ | s; s; -- | 3.4.3 |
| $D$ | $\sigma^2/\tau$ of a slow OU component | $\mathbb R_{>0}$ | mV²/s | 3.4.3 |
| $I_{\rm REML}(T)$ | information on $1/\tau$ kept by one window of duration $T$ with its own unknown mean (its REML information), pure OU, $D$ fixed, eq. (24b) (analytic level) | $\mathbb R_{\ge 0}$ | s² | 3.4.3 |
| $m$; $\hat q_m$; $q_m(\theta)$ | index of a matched statistic; its measured value (a cohort or per-cell median); its value under $\theta$ through the pipeline's estimator | $\{1, \dots, 14\}$; $\mathbb R$; $\mathbb R$ | mV or dimensionless | 3.5.1 |
| $\omega_m$ | residual scale of statistic $m$ (configured: 0.005 mV for SDs, 0.05 for autocorrelations) | $\mathbb R_{>0}$ | as $\hat q_m$ | 3.5.1 |
| $\hat\theta$; $\mu, \hat\mu$ | an estimate of $\theta$; a window's unknown mean and its estimate | $\Theta$; $\mathbb R$ | mixed; mV | 3.5.1 |
| $\Gamma(\theta, n)$; $\mathbf 1$; $\det$, $^\top$ | covariance matrix of a window; all-ones vector; determinant, transpose | SPD $n \times n$; $\mathbb R^n$; operators | mV²; --; -- | 3.5.1 |
| $\arg\min$, $\arg\max$ | the argument at which a function attains its minimum, maximum | operators | -- | 3.5.1 |
| $J(\theta)$; $J^{\rm cf}$, $J^{\rm sim}$ | cost (14); with the closed form, with simulated medians | $\mathbb R_{\ge 0}$ | dimensionless | 3.5.2 |
| $q^{\rm cf}_m(\theta)$; $q^{\rm sim}_m(\theta)$; $M$ | closed-form approximation of $q_m$; median of the estimator over $M$ simulated windows; $M$ | $\mathbb R$; $\mathbb R$; $\mathbb N$ | as $\hat q_m$ | 3.5.2 |
| $K_{\rm OU}$; $\sigma_3, \tau_3$ | number of OU components of a candidate law; SD and correlation time of a hypothetical seconds-scale OU (not part of the law, D-034) | $\mathbb N_{\ge 1}$; $\mathbb R_{>0}$ | --; mV, s | 3.5.4 |
| $n_{\rm par}$; $n_{\rm obs}$; $\hat{\mathcal L}$ | free parameters (of a BIC model; of the fit (28), §3.5.6); observations entering a likelihood; maximised likelihood (in BIC) | $\mathbb N$; $\mathbb N$; $\mathbb R_{>0}$ | -- | 3.5.4 |
| $d$; $\hat{\mathbf q}(c)$; $\mathbf q(\theta)$ | number of statistics in a fit (14 in §3.5.3, 10 in the checks of §3.5.6; not the $d$ of an integral); cell $c$'s statistic vector, each entry the median over its windows (computed level); the model's vector $(q_1(\theta), \dots, q_d(\theta))^\top$ (analytic level) | $\mathbb N_{\ge 1}$; $\mathbb R^d$; $\mathbb R^d$ | --; as $\hat q_m$ | 3.5.6 |
| $\mathbf n$, $\mathbf K$; $n_m$, $K_m$ | window lengths and window counts behind the $d$ statistics of one cell, $\mathbf n(c)$, $\mathbf K(c)$ for cell $c$; their entries for statistic $m$ | $\mathbb N^d$ | samples; -- | 3.5.6 |
| $W$; $J_W(\theta \mid c)$; $\hat\theta_W(c)$ | weighting matrix; the weighted cost (28); its minimiser for cell $c$ | symmetric positive-definite $d \times d$; $\mathbb R_{\ge 0}$; $\Theta$ | inverse squares of the statistics' units; dimensionless; mixed | 3.5.6 |
| $\Sigma(\theta \mid \mathbf n, \mathbf K)$; $\hat\Sigma(\theta \mid \mathbf n, \mathbf K)$; $\Sigma_{\rm used}$, $\Delta\Sigma$ | covariance matrix of $\hat{\mathbf q}(c)$ for a cell whose law is $\theta$ (analytic level); its sample estimate, ddof 1, from $R$ cells simulated at $\theta$ (computed level); the matrix actually inverted, and its deviation $\Sigma_{\rm used} - \Sigma$ | symmetric positive-definite $d \times d$; symmetric $d \times d$ for $\Delta\Sigma$ | products of the statistics' units | 3.5.6 |
| $\Sigma^{\rm corr}$; $\mathrm{SE}_m$; $\widehat{\mathrm{SE}}_m(c)$ | correlation matrix of $\Sigma$; $\Sigma_{mm}^{1/2}$ (analytic level); the window-level bootstrap SE of $\hat q_m(c)$ (computed level) | $d \times d$, unit diagonal; $\mathbb R_{>0}$; $\mathbb R_{>0}$ | dimensionless; as $\hat q_m$ | 3.5.6 |
| $R$; $\iota$; $W^{[\iota]}$, $\hat\theta^{[\iota]}(c)$ | number of cells simulated for one $\hat\Sigma$ (bare $R$, unrelated to $R_m$, $R_a$, $R_{\rm cell}$); iteration index of (29); the weighting and the estimate at iteration $\iota$ | $\mathbb N_{\ge 2}$; $\mathbb N$; as $W$; $\Theta$ | -- | 3.5.6 |
| $\operatorname{cond}(A)$; $\Lambda_{\max}(A)$, $\Lambda_{\min}(A)$; $\mathbf v_{\min}(A)$; $\operatorname{diag}(\cdot)$ | condition number $\Lambda_{\max}(A)/\Lambda_{\min}(A)$ of a symmetric positive-definite matrix $A$; its largest and smallest eigenvalue; a unit eigenvector for $\Lambda_{\min}(A)$; the diagonal matrix of a vector's entries, or of a matrix's diagonal | $[1, \infty)$; $\mathbb R_{>0}$; $\mathbb R^d$; operator | dimensionless; units of $A$; --; -- | 3.5.6 |
| $\varepsilon$; $\lVert\cdot\rVert_2$; $\mathbf a$ | relative size $\lVert\Delta\Sigma\rVert_2/\lVert\Sigma\rVert_2$ of a perturbation; spectral norm; a vector of coefficients combining the statistics (bold; not the offset $a$) | $\mathbb R_{\ge 0}$; operator; $\mathbb R^d \setminus \{\mathbf 0\}$ | dimensionless; --; -- | 3.5.6 |
| $\chi^2_{d - n_{\rm par}}$; $\nabla_{\tau_s}\mathbf q$; $N_{\rm boot}$ | chi-squared law with $d - n_{\rm par}$ degrees of freedom; derivative of $\mathbf q(\theta)$ with respect to $\tau_s$ at the truth; number of bootstrap replicates (not $B(t \mid \theta)$ of §3.6.2) | law on $\mathbb R_{\ge 0}$; $\mathbb R^d$; $\mathbb N$ | --; per s; -- | 3.5.6 |
| $\bar x_{\rm pre}$; $B(t \mid \theta)$ | mean of the LS pre-window; RMS of the baseline-subtracted stationary noise at $t$ after onset | $\mathbb R$; $\mathbb R_{\ge 0}$ | mV | 3.6.2 |
| $V(h \mid \theta, \sigma_b)$ | semivariogram of two window means $h$ apart in one sweep, under the law (analytic level), eq. (25) | $\mathbb R_{\ge 0}$ | mV² | 3.6.4 |
| $\bar x_{(1)}, \bar x_{(2)}$; $T_1, T_2$; $g$ | the two window means; their durations; the gap between the windows | $\mathbb R$; $\mathbb R_{>0}$; $\mathbb R_{\ge 0}$ | mV; s; s | 3.6.4 |
| $\operatorname{Var}_T$; $\operatorname{Cov}_{T_1,T_2}(h)$ | variance of one OU component's window mean; covariance of two | $\mathbb R_{\ge 0}$; $\mathbb R$ | mV² | 3.6.4 |
| $\hat V_c(h)$; $h^*$; $\hat\theta_c$ | cell $c$'s semivariogram computed from its SS window means (computed level); the calibration lag; $c$'s fitted law | $\mathbb R_{\ge 0}$; $\mathbb R_{>0}$; $\Theta$ | mV²; s; mixed | 3.6.4 |
| $\bar r_{\rm dep}(t), \bar r_{\rm hyp}(t)$; $\bar r_{\rm odd}(t)$ | polarity-mean responses after each window's pre-pulse mean is removed; their odd part, eq. (27) | $\mathbb R$ | mV | 3.7 |
| $\kappa_{\rm tr}$; $\hat\sigma_{\rm SS}$ | threshold factor of the transient test (configured); the cell's median SS $\hat\sigma$ | $\mathbb R_{>0}$; $\mathbb R_{\ge 0}$ | dimensionless; mV | 3.7 |
| $\Pr(\cdot)$; $p$ | probability; p-value of a test | $[0, 1]$ | -- | 3.2.5 |
| $\ln$; $\propto$; $\operatorname{SD}(\cdot)$ | natural logarithm; proportional to; standard deviation of an estimator over repeated data | operators; relation | -- | 3.3.1; 2; 3.4.3 |
| $\hat\sigma_{\rm LS}$; $\hat\rho_{\rm LS}(\cdot)$, $\hat\rho_{\rm SS}(\cdot)$ | a cell's medians over its LS sweeps or SS pulses (Appendix A) | $\mathbb R_{\ge 0}$; $[-1, 1]$ | mV; dimensionless | App. A |

### Conventions

- **Units.** Equations are unit-consistent with times in seconds; numbers are quoted in ms or µs
  where that reads better. $\Delta t$ = 20 µs (50 kHz) for every L3_exc LS sweep and, at the cohort
  median, for the SS pulses.
- **Time origin.** $t$ is measured from the SS pulse onset in (1) and §3.7, from the start of the
  stored sweep in (5a)-(5b), and from the LS step onset in (15)-(16).
- **Subscripts.** $s$ slow and $f$ fast; **SS** the Allen *Square Subthreshold* protocol (0.5 ms
  pulses to $\pm$200 pA, 20 per sweep, 200 ms apart), not the Allen *Short Square*; **LS** the
  *Long Square* protocol (1 s steps).
- **Levels.** Hats mark numbers computed from recorded or simulated data (computed level); a symbol
  written with "$\mid \theta$" is a property of the law (analytic level). A per-cell value is the
  median over the cell's sweeps or pulses ($s_{\rm lock}$: one value per cell); a cohort value is
  the median over cells of the per-cell values.
- **Lags, and one declared abuse of notation.** $k$ is a generic lag and $L$ a statistic's lag, in
  samples; $\hat\rho(\ell)$ with $\ell$ in ms means $\hat\rho(L)$ at $L = \ell/\Delta t$.
- **The letter $\rho$.** $\rho_f$, $\rho_s$ and a bare $\rho$ (a generic AR(1)) are lag-1
  coefficients; $\rho(\ell \mid \theta)$ is the
  law's autocorrelation; $\hat\rho(L)$ is the statistic (3).
- **Renamings.** D-017.1 writes the components $f_j$ and $s_j$; here $e^{\rm fast}_j$ and
  $e^{\rm slow}_j$, keeping $f$ and $s$ free. D-034 writes the enlarged window's start $t_0(c)$;
  here $t_{\rm SS}(c)$, because $t_0$ is a stored sweep's first sample.
- **Equation numbers.** (1)-(16) keep their v2 numbers; (3) and (10) now use the
  textbook estimator, and (14) is the cost for either way of computing $q_m$; (17) is
  unused; (18)-(27) are new, in order of appearance; (28)-(30), in §3.5.6, are added in v3.4.
- **"Mixture"** means a sum of independent components, not a probabilistic mixture of laws.
- **Source tags.** [KB] project knowledge base, full text; [PubMed FT] full text read through PubMed
  Central; [PubMed abstract only] flagged, never used for a number; [src] read from code;
  [cluster] computed by our scripts on the real archive and pasted by the user; [run] computed in
  this analysis (Appendix B); [user] decision; [reasoning] the assistant's argument; [derived] a
  short derivation by the assistant, checked where stated; [textbook] from memory, not searched.
  From v3.4, [run] also covers the checks of 2026-10-11 to 2026-10-14 listed in Appendix B.

---

## 2 Glossary

Ordered alphabetically.

- **ABC (approximate Bayesian computation).** Keep the parameter draws whose simulated summary
  statistic lies near the observed one; the kept draws approximate a posterior. Not used (D-033).
  §3.5.5.
- **AR(1).** First-order autoregressive sequence: each sample is $\rho$ times the previous one plus
  independent Gaussian noise; autocorrelation $\rho^k$ at lag $k$. §3.2.3.
- **Autocorrelation.** *Flag: two senses.* *True*: the law's correlation with itself at a lag.
  *Sample*: the statistic (3) computed from one window. They differ systematically on short windows.
  §3.4.2.
- **Autocovariance.** The covariance of the process with itself at a given lag; eq. (6). §3.3.1.
- **Bessel filter.** A low-pass filter with nearly linear phase; the Allen recordings were
  Bessel-filtered at 10 kHz before digitisation. How it is realised (analog, or digital at a given
  rate and normalisation) changes its lag-1 correlation. §3.3.3.
- **BIC (Bayesian information criterion).** A likelihood penalised by the number of parameters, used
  to choose a model order; consistent only when the true model is among the candidates. §3.5.4.
- **Bundle / sweep.** A *sweep* is one recorded trace; a *bundle* is what the fitter fits: an
  average of $N$ SS pulses, or a single LS sweep. §3.1.2.
- **Campbell's theorem.** The covariance of a sum of randomly timed, identically shaped kicks is the
  kick rate times the overlap integral of the kick shape. §3.3.3.
- **Common random numbers.** Reusing the same random draws for every parameter value of a
  simulation-based fit, so that the cost changes only because the parameters change. §3.5.2.
- **Condition number.** For a symmetric positive-definite matrix, its largest eigenvalue over its
  smallest. Two readings: how much more one combination of the statistics scatters than another, and
  how much a relative error in the matrix can grow in its inverse, (30). Depends on units; that of the
  correlation matrix does not. §3.5.6.
- **Cost profile.** The minimum cost over the other parameters at fixed values of one; shows how
  firmly the data pin it. §3.5.3.
- **Creep.** *Flag: also called "drift" here.* A change of the baseline approximately linear across
  a sweep; it continues through the step. Not found in any L3_exc cell. §3.6.1.
- **Demeaning.** Subtracting a window's own mean before computing a statistic; in expectation a
  high-pass filter (22). §3.4.1.
- **Direct fit.** *Flag: two senses.* As in Zeraati et al.: fit the law's true autocorrelation to
  the sample autocorrelation, eq. (11). As opposed to indirect inference in statistics: estimate
  through the likelihood. §3.5.1.
- **Fluctuation (noise) analysis.** Reading channel kinetics off the spectrum or autocorrelation of
  current noise. §3.5.5.
- **I_h, sag, rebound.** The hyperpolarisation-activated cation current; the relaxation back
  toward rest during a hyperpolarising step that it produces; the depolarisation after the step.
  §3.1.1.
- **Identifiability (weak, one-sided).** A parameter fitted almost equally well over a broad range;
  $\tau_s$ is bounded well from below and poorly from above. §3.5.3.
- **Indirect inference.** Choose the parameters under which the model reproduces a chosen statistic
  *as the same estimator computes it*, here through closed-form approximations or simulated medians.
  §3.5.1.
- **Iterated weighting (two-step; iterated GMM).** Estimate with a simple weighting, compute the
  covariance of the statistics under that estimate (here by simulation), refit with its inverse, and
  repeat until the cost stops changing; the two-step stops after one refit (D-048). §3.5.6.
- **Layer labels.** L3_exc: the Allen layer-3 excitatory group of our archive; L2&3, L3c (deep
  layer 3) and L5: the groups of Moradi Chameh et al.; L4/5: the rat groups of the noise papers.
  §3.3.3.
- **Likelihood (Gaussian); REML.** The probability density of the raw samples as a function of the
  parameters; REML integrates each window's unknown mean out instead of maximising over it; this
  changes the bias that profiling the mean puts on $\hat\tau$ but does not remove it. Its information,
  (24b), is what a demeaned window keeps on $1/\tau$ (§3.4.3). §3.5.1.
- **LJP (liquid-junction potential).** The voltage offset between pipette and bath solutions; the
  archive is corrected by +14 mV; it does not affect the noise statistics. §3.2.2.
- **Locked share.** The share of a cell's pre-window variance that repeats from sweep to sweep
  (D-016.4); detects a transient locked to the start of the stored sweep. §3.2.4.
- **Long Square (LS).** Allen protocol of 1 s current steps, one amplitude per sweep, where I_h is
  identified. §3.1.2.
- **Lorentzian.** Power spectrum $\propto 1/(1 + (\nu/\nu_c)^2)$, flat below the corner and falling
  as $\nu^{-2}$ above it: the spectrum of an OU process. §3.3.3.
- **LS tail change.** The signed change of one LS sweep's baseline from the pre-window to the last
  500 ms of the stored sweep, 6.89 s later. §3.2.5.
- **Model order.** The number of OU components of a candidate law. §3.5.4.
- **Noise.** *Flag: broader than "instrument noise".* The total pre-stimulus fluctuation of one
  sweep, recording chain plus membrane (D-015 (iii)). §3.1.3.
- **Ornstein-Uhlenbeck (OU) process.** Stationary Gaussian process with autocorrelation
  $\mathrm e^{-|h|/\tau}$; sampled every $\Delta t$ it is exactly an AR(1) with
  $\rho = \mathrm e^{-\Delta t/\tau}$; the voltage of an RC membrane driven by white current noise.
  §3.3.1.
- **Overidentification (J) alarm.** With more statistics than parameters and the optimal weighting,
  twice the minimised cost is asymptotically $\chi^2$ with $d - n_{\rm par}$ degrees of freedom when the
  model is right (Hansen); used here as an alarm for misfit, not as a test. §3.5.6.
- **Pre-window.** The samples before a stimulus used as baseline and as noise sample: $t < 0$ for an
  SS pulse, $[t_0, t_{\rm on} - 5\ \text{ms})$ for an LS sweep. §3.1.2.
- **QC (Allen).** Cell-level criteria on the whole recording (e.g. electrode drift below 1 mV per
  10 min) and sweep-level criteria on noise and voltage stability. §3.6.6.
- **Ratio of expectations.** The ratio of the expectations of a statistic's numerator and
  denominator, used in place of the expectation of the ratio; exact only for windows much longer
  than the slowest timescale. §3.4.2.
- **Recovery gate (Stage 7).** Fits synthetic cells of known ground truth with the production
  fitter and compares estimates with truth (D-011). §3.1.1.
- **Resonance (subthreshold).** A peak of the membrane impedance at a non-zero frequency; it would
  make resting noise a damped oscillation. §3.3.3.
- **Semivariogram.** Half the mean squared difference of a quantity at two times, as a function of
  their separation; here of window means within one SS sweep, eq. (25). §3.6.4.
- **Simulated medians.** Our implementation of indirect inference: the model's value of each
  statistic is the median of the pipeline's estimator over simulated windows of the real length.
  §3.5.2.
- **Spectral representation.** Any stationary process is a superposition of random-phase sinusoids,
  each frequency weighted by the spectrum. §3.3.3.
- **Square Subthreshold (SS).** Allen protocol of 0.5 ms pulses to $\pm$200 pA, 20 per sweep, 200 ms
  apart. §3.1.2.
- **Stationarity.** *Flag: weak sense.* Mean and autocovariance do not change with time within the
  record; for a Gaussian process this fixes the whole law. §3.3.1.
- **Textbook estimator.** The sample autocorrelation $S_L/S_0$, both sums divided by the same $n$;
  bounded by 1 (D-028). §3.2.2.
- **Tilt.** The per-sweep linear term $b\,(t_j - \bar t)$ of (5a); the law's carrier of the
  variance at seconds (D-034). §3.3.2.
- **Transient test.** Decides from where after an SS pulse its response has died out, eq. (27).
  §3.7.
- **Trend share.** The share of a window's variance that a least-squares line explains, eq. (4).
  §3.2.2.
- **Visible share.** The fraction of a slow component's variance that survives demeaning a window,
  eq. (8). §3.4.1.
- **Wander.** *Flag: used here for a stationary fluctuation only.* A slow stationary fluctuation that
  returns towards the cell's mean level, as against a creep. §3.6.1.
- **Weighting matrix.** The matrix $W$ of (28) that sets how much each statistic, and each
  combination of statistics, counts in the fit; optimal at the inverse of the statistics' covariance.
  §3.5.6.
- **Wiener-Khinchin theorem.** The autocovariance and the power spectrum of a stationary process are
  a Fourier pair. §3.3.3.
- **Window statistic.** Any SD or autocorrelation the pipeline computes on a window. §3.2.2.
- **Window-level bootstrap.** Resampling a cell's whole windows with replacement, not the samples
  within a window, so that each window keeps its own autocorrelation. §3.5.6.
- **Zero-sum identity.** The textbook sample autocorrelation of any window sums to $-\tfrac12$ over
  all lags, eq. (23). §3.4.2.

---

## 3 Main body

### 3.1 Context: why the synthetic twins need the real noise

*This section establishes what the recovery gate tests, how the fitter uses the pre-stimulus
window, and how the noise model reached its present form.*

#### 3.1.1 The recovery gate

The I_h-aware fit estimates six parameters per cell: $C_m$, $R_m$, $R_a$, the I_h density
$\bar g_h$, the shift $\Delta v_h$ of its activation curve and a scale $\kappa_\tau$ on its time
constant (D-005). Stage 7 generates synthetic cells with known parameters on the real archive
morphologies, fits them with the production fitter and compares estimates with truth; a quarter
of the cohort carries no I_h, as a false-positive control (D-011) **[KB]**. Each synthetic cell
$c'$ is the twin of a real cell $c \in \mathcal C$: same morphology, sampling rates and SS pulse
count (D-014), and its own real cell's noise law (D-017). In plain words: the gate asks "if the
truth were known, would we find it?", and noise is what makes the data hard; a twin with cleaner
noise than its cell would pass a gate that the real data fail (D-011 (iii) **[KB]**).

#### 3.1.2 What the fitter does with the pre-window

Both protocols use the samples before the stimulus twice: as the **baseline** subtracted from the
trace, and as the **sample of noise** that the calibration reads.

- **SS** (Square Subthreshold): 0.5 ms pulses to $\pm$200 pA, 20 per sweep, 200 ms apart **[KB]**.
  $N$ pulses are averaged into a bundle, whose fast noise has SD $\sigma/\sqrt N$ (D-012). The
  residual is a weighted RMS deviation over 0.5-100 ms after each trace's own pre-window mean
  ($t < 0$, 10 ms) is removed, with $t$ from the pulse onset and

$$ w(t) = \mathrm e^{-(t - 0.5\ \text{ms})/5\ \text{ms}}, \qquad 0.5\ \text{ms} \le t \le 100\ \text{ms}, \tag{1} $$

  so it is judged almost entirely on its first 10-15 ms (D-014, D-015 **[KB, src]**).
- **LS** (Long Square): 1 s steps, one amplitude per sweep, from $-110$ pA (or $-220$ pA for some
  low-input-resistance human cells) **[KB]**. Each LS bundle is a single sweep (D-012); the
  residual spans the whole step $[t_{\rm on}, t_{\rm off}]$ (D-006) after the mean of the
  pre-window $[t_0, t_{\rm on} - 5\ \text{ms})$ is subtracted **[src]**.

**Consequence.** The SS residual is not fast noise only: under the cohort law
$\hat\theta_{\rm sim}$ (§3.5.3) the slow component supplies 26 %, 42 %, 53 % and 59 % of the
baseline-subtracted noise variance of one trace at 0.5, 5, 10 and 15 ms after the pulse **[run]**.
The LS residual spans a second after a baseline taken 5-270 ms before the step, so it sees every
slow fluctuation -- on exactly the timescale on which $\bar g_h$, $\Delta v_h$ and $\kappa_\tau$ are
identified -- and nothing averages it down **[reasoning]**.

#### 3.1.3 The noise object, and how the model got here

The object to reproduce is the pre-stimulus fluctuation of a **single sweep**: recording chain
plus membrane, not an instrument-only noise (D-015 (iii), **[user]**).

| Record | What it fixed |
|---|---|
| D-012 | per-sweep noise of the real twin, read with the fitter's estimator; run B's averaged-bundle SD is not a generator input |
| D-014 | acquisition twins: each twin at its cell's own sampling rates and SS pulse count; AR(1) adequacy reported |
| D-015 | the loss weight (1) never enters the noise; the fast noise is stationary; the noise object is one sweep's total fluctuation |
| D-016 | AR(1) found inadequate on L3_exc |
| D-017 | per-cell law: fast AR(1) + slow OU + per-sweep offset and tilt, fitted to the cell's own window statistics; no cohort law; the twin's LS pre-window is its cell's (265 ms) |
| D-028, D-029 | the sample autocorrelation is the textbook estimator, through statsmodels |
| D-033 | the per-cell law is our own indirect-inference point estimate; no ABC |
| D-034 | the variance at seconds is carried by the per-sweep tilt, calibrated from SS window means; SS noise windows enlarged where the pulse transient has died out |

The generator in the repository still implements D-014: a stationary AR(1) per protocol, a
per-sweep offset and tilt of nominal size (0.05 and 0.10 mV scaled by $\hat\sigma/0.05$ mV, "not
read off the recordings" **[src]**), and the LS step 100 ms into the synthetic sweep, i.e. a 95 ms
pre-window **[src]**.

---

### 3.2 What was measured on the real cells

*This section establishes the data, the estimators, and the facts the law must reproduce:
correlation that outlives AR(1), an SD that grows with the window, and a baseline that moves over
seconds by more than the two-component law allows.*

#### 3.2.1 Cells, protocols, windows

The L3_exc cohort has 18 cells; every LS sweep is sampled at $f_s$ = 50 kHz, and the SS pulses at
50 kHz at the cohort median (per-cell SS rates not verified) **[cluster]**. The Allen recordings
were Bessel-filtered at 10 kHz and digitised at 200 or 50 kHz, with the bias current held stable
for at least 1 s before each stimulus, in a bath containing kynurenic acid and picrotoxin to block
fast glutamatergic and GABAergic transmission **[KB]**.

| Window | Definition | Samples $n$ | Per cell |
|---|---|---|---|
| SS pre-pulse | the 10 ms before a pulse | 500 | 37-348 pulses (about 2-17 sweeps) |
| SS enlarged (D-034) | $[t_{\rm SS}(c), 200)$ ms after the previous pulse (§3.7) | at most 9500 | up to 20 per sweep |
| LS pre-window | $[t_0, t_{\rm on} - 5\ \text{ms})$, 265 ms in every cell | 13 250 | 5-6 single sweeps |
| LS first segment | its first 50 ms | 2500 | same sweeps |
| LS last segment | its last 132.5 ms | 6625 | same sweeps |
| LS end window | the last 500 ms of the stored sweep | 25 000 | same sweeps |

The stored LS sweep starts at the AllenSDK `index_range` start, documented as the end of the test
pulse, so the onset sits 270 ms into it **[src]**; every stored sweep holds 6.000 s after the
step, so the end window starts 5.5 s after the step and its centre lies 6.89 s after the
pre-window's **[cluster, 2026-10-01]**.

#### 3.2.2 The estimators

For a window $x_1, \dots, x_n$ with mean $\bar x$ and $u_j = x_j - \bar x$ **[src]**:

$$ \hat\sigma = \sqrt{\frac{S_0}{n - 1}}, \qquad S_0 = \sum_{j=1}^{n} u_j^2, \tag{2} $$

$$ \hat\rho(L) = \frac{S_L}{S_0}, \qquad S_L = \sum_{j=1}^{n-L} u_j\,u_{j+L}, \tag{3} $$

$$ s_{\rm trend} = 1 - \frac{\operatorname{var}(\mathbf x - \hat{\mathbf x}^{\rm lin})}{\operatorname{var}(\mathbf x)}. \tag{4} $$

(3) is the textbook sample autocorrelation, computed by
`statsmodels.tsa.stattools.acf(x, adjusted=False)` through `noise_stats.sample_acf`, which returns
NaN when $n - L < 10$, for a flat window or a non-finite sample (D-028, D-029 **[src]**); it lies
in $[-1, 1]$. Values computed before D-028 with $\hat\rho_{\rm adj}(L) = \bigl(S_L/(n-L)\bigr)/(S_0/n)$
convert exactly, window by window, and for medians over windows of equal duration at a physical lag, or of equal $n$ at a lag in samples:

$$ \hat\rho(L) = \frac{n-L}{n}\,\hat\rho_{\rm adj}(L) = \Bigl(1 - \frac{\ell}{T}\Bigr)\hat\rho_{\rm adj}(L), \qquad \ell = L\,\Delta t, \tag{18} $$

so at a physical lag the factor depends on the window's duration only, not on $f_s$ **[derived]**.
Every autocorrelation value below is textbook; values measured before D-028 are converted by (18)
**[cluster; converted]**, and the re-run of the noise table on the cluster (`noise_L3_v3.csv`,
pending) must reproduce them to rounding. $\hat{\mathbf x}^{\rm lin}$ in (4) is the least-squares
line (`scipy.signal.detrend`). The **locked share** $s_{\rm lock}$ (D-016.4) is the share of a
cell's pre-window variance that repeats from sweep to sweep: 0 in expectation when nothing
repeats, 1 when the sweeps are identical; one value per cell. The **LS tail change** of one sweep
(`ls_baseline_qc.py`) is $\Delta_{\rm end}$, the mean of the end window minus the mean of the
pre-window, signed, beside $\Delta_{\rm creep} = \hat b \times 6.89$ s, what a creep continuing at
the pre-window's least-squares slope $\hat b$ would give **[src]**. Every other per-cell value is
the median over the cell's $K$ sweeps (LS) or pulses (SS).

#### 3.2.3 The cohort's numbers and the failure of AR(1)

Cohort medians **[cluster, 2026-09-23 and 2026-09-28; autocorrelations converted by (18)]**. The
"AR(1)" values are, per sweep, $\hat\rho(1)^L$, what an AR(1) with that sweep's lag-1 coefficient
would give, then the median over sweeps and cells **[src]**.

| Window | $\hat\sigma$ (mV) | $\hat\rho(1)$ | $\hat\rho(0.1\ \text{ms})$ | $\hat\rho(1\ \text{ms})$ | $\hat\rho(10\ \text{ms})$ | $\hat\rho(100\ \text{ms})$ |
|---|---|---|---|---|---|---|
| SS 10 ms | 0.0405 | 0.274 | 0.153 (AR(1): 0.002) | 0.032 (0.000) | -- | -- |
| LS 265 ms | 0.0594 | 0.668 | 0.613 (0.134) | 0.564 (0.000) | -- | -- |
| LS first 50 ms | 0.0465 | -- | -- | -- | -- | -- |
| LS last 132.5 ms | 0.0551 | 0.593 | 0.514 (0.079) | 0.445 (0.000) | 0.226 (0.000) | $-0.022$ |

Three facts follow. (i) **The correlation outlives AR(1) by orders of magnitude**: 0.445 at 1 ms
and 0.226 at 10 ms in the last segment, where any AR(1) matched at lag 1 gives 0.000. (ii) **The SD
grows with the window**: 0.0465 mV (first 50 ms), 0.0551 (last 132.5 ms), 0.0594 mV (all 265 ms);
noise whose correlation times are all far below 50 ms would give the same SD in each. The SS
10 ms window, 0.0405 mV, continues the progression if the two protocols share one law (open point
(b), §5). (iii) All 18 cells have LS $\hat\rho(1) > 0.5$ **[cluster]**: not a few outliers. In
plain words: the noise has a second, slow ingredient that a short window barely sees.

#### 3.2.4 Start-of-sweep transient and sweep locking

- **No transient at the cohort level.** The stationary law fitted to all 14 statistics (§3.5.3)
  reproduces the first segment's SD, 0.0460 mV predicted against 0.0465 mV measured, as well as the
  last segment's, 0.0552 against 0.0551 mV **[run; cluster]** -- a consistency check on cohort
  medians, both SDs being fitted targets. That the first segment is quieter than the last is not
  itself evidence: a stationary slow component makes a shorter window quieter (§3.4.1).
- **Weak locking in five cells.** The cohort median of $s_{\rm lock}$ is 0.02. Under
  $\hat\theta_{\rm sim}$ the per-cell null for 265 ms windows has a 95th percentile of 0.121
  ($K$ = 5) and 0.094 ($K$ = 6) and a 99th of 0.191 and 0.148 **[run]**. Five cells, all with
  $K$ = 6, lie above the 95th percentile against about one expected by chance: 529878215 (0.186,
  above the 99th), 571691416 (0.148, at the 99th), 548425467 (0.118), 571624532 (0.105) and
  537313600 (0.104) **[cluster; run]**. A weak sweep-locked component is not excluded in these
  cells; since no cell creeps (§3.2.5), a start-of-sweep transient is the candidate left. Slower
  laws widen the null, so this is indicative **[reasoning]**.

#### 3.2.5 The baseline over seconds: the LS tail

`ls_baseline_qc.py` was run on L3_exc on 2026-10-01 (commit `3dfa263`): 104 hyperpolarising
single LS sweeps in 18 cells (D-017 note 4) **[cluster]**.

- **No continuing creep, in any cell.** $\Delta_{\rm end}$ and $\Delta_{\rm creep}$ agree in sign
  in 53 of 104 sweeps (binomial, two-sided $p$ = 0.92). With these windows a stationary wander of
  any timescale (OU from 37 ms to 30 s, and a random walk) gives an agreement of 0.49-0.53, because
  the later change is dominated by the return towards the cell's mean level, which is unrelated to
  the local slope; a continuing creep gives agreement near 1 **[run, 2026-10-01]**. In the three
  sweeps of 526785799 with $|\Delta_{\rm creep}|$ of 4.8-6.2 mV, $|\Delta_{\rm end}|$ is at most
  0.42 mV; in 571430815, $|\Delta_{\rm creep}|$ 5.3-23.9 mV stands against $|\Delta_{\rm end}|$
  0.30-0.81 mV; the least-squares slope of $\Delta_{\rm end}$ on $\Delta_{\rm creep}$ over these 10
  sweeps is 0.005 **[cluster; run]**. The two cells with a high trend share (526785799: 0.58;
  571430815: 0.64; cohort median 0.14) are slow wander, not creep.
- **More variance at seconds than the law allows, in 17 or 18 of the 18 cells.** The per-cell median of
  $|\Delta_{\rm end}|$ is 0.048-0.620 mV, cohort median 0.128 mV **[cluster]**. Under
  $\hat\theta_{\rm sim}$ alone a single sweep's $|\Delta_{\rm end}|$ has median 0.023 mV, and the
  per-cell median over 5 or 6 sweeps has median 0.024 mV and 99th percentile 0.056 mV ($K$ = 5) or
  0.051 mV ($K$ = 6) **[run]**; against nulls of this kind (99th percentiles 0.046-0.056 mV under
  the laws of 2026-09-28), 17 or 18 of the 18 cells lie above (D-017 note 4) **[cluster; run]**; the
  lowest cell, 0.048 mV, lies below $\hat\theta_{\rm sim}$'s 99th percentiles but above its 96th-98th
  ($\Pr$ = 0.019-0.035) **[run]**.
  In plain words: the baseline wanders back, but by more than a 40 ms process can.
- **Hint, not significant.** In 526785799, $\Delta_{\rm end}$ rises with step size (Spearman 0.90,
  $p$ = 0.04, 5 sweeps, uncorrected). The protocol runs from $-110$ pA upwards in 20 pA increments
  **[KB]**, so amplitude and recording order are collinear, and a step-induced after-effect cannot
  be told from a drift over the protocol; the per-cell median of $\Delta_{\rm end}$ is positive in
  12 of 18 cells ($p$ = 0.24) **[cluster; run]**.

---

### 3.3 The noise law

*This section establishes the law adopted in D-017 and D-034, and what its components are
physically.*

#### 3.3.1 Fast AR(1) plus slow OU

For one sweep with samples $j = 0, 1, \dots$ at times $t_j = j\,\Delta t$ from the start of the
sweep and mid-time $\bar t$, the noise is

$$ e_j = e^{\rm fast}_j + e^{\rm slow}_j + a + b\,(t_j - \bar t), \tag{5a} $$

$$ e^{\rm fast}_j = \rho_f\,e^{\rm fast}_{j-1} + \eta_j,\ \ \eta_j \sim \mathcal N\bigl(0, \sigma_f^2(1 - \rho_f^2)\bigr); \qquad e^{\rm slow}_j = \rho_s\,e^{\rm slow}_{j-1} + \eta'_j,\ \ \eta'_j \sim \mathcal N\bigl(0, \sigma_s^2(1 - \rho_s^2)\bigr), \tag{5b} $$

with $\rho_s = \mathrm e^{-\Delta t/\tau_s}$, both recursions started in their stationary laws and
driven by independent innovations. The slow recursion is exactly an OU process of SD $\sigma_s$ and
correlation time $\tau_s$ observed every $\Delta t$ **[textbook]**; likewise the fast one, with
$\tau_f = -\Delta t/\ln\rho_f$. For each fixed $\theta = (\sigma_f, \rho_f, \sigma_s, \tau_s)$ and
every lag $k \ge 0$, the stationary part $e^{\rm stat}_j = e^{\rm fast}_j + e^{\rm slow}_j$ has
autocovariance

$$ \gamma(k \mid \theta) = \sigma_f^2\,\rho_f^{\,k} + \sigma_s^2\,\mathrm e^{-k\,\Delta t/\tau_s}, \qquad \gamma(-k \mid \theta) = \gamma(k \mid \theta), \tag{6} $$

a sum of two independent OU processes -- the generative model of Zeraati et al. for timescale
estimation **[PubMed FT]**. Both components are Gaussian and weakly (second-order) stationary, so
the mean and (6) determine the stationary part completely; the autocorrelation is the model, not a
summary of it. The cohort estimate, a diagnostic and never a generator input (D-017 (ii)), is
$\hat\theta_{\rm sim}$ = (0.0383 mV, 0.179, 0.0547 mV, 39.7 ms), i.e. a fast correlation time of
11.6 µs (§3.5.3).

#### 3.3.2 Per-sweep offset and tilt (D-034)

The last two terms of (5a) are drawn once per sweep, independently across sweeps.
$a \sim \mathcal N(0, \sigma_a^2)$ cancels in every residual and every window statistic, since each
trace's own pre-window mean is removed and every statistic demeans: $\sigma_a$ needs no
calibration **[src; derived]**. $b \sim \mathcal N(0, \sigma_b(c)^2)$ (mV/s); the generator sets
the span $\delta = b\,T_{\rm sw} \sim \mathcal N(0, \sigma_\delta^2)$ over the whole generated
trace, so $\sigma_\delta = \sigma_b\,T_{\rm sw}$ with $T_{\rm sw}$ that trace's own duration
**[src]**. The tilt is the law's only carrier of the variance at seconds (§3.2.5): there is no third
OU component (D-034). Its semivariogram grows as $\tfrac12\sigma_b^2h^2$ at lag $h$, while an OU's
grows at most linearly and saturates; unless the variance at seconds is itself a linear creep, which
the LS tail excludes (§3.2.5), a tilt therefore reproduces it at one lag only;
§3.6.4-3.6.5 calibrate it there.

#### 3.3.3 What the noise is physically

*Interpretation; the generator needs the statistics, not the physics.*

- **Memory, not a signal.** "At rest" means that the mean does not move, not that the voltage does
  not. Membrane and recording chain are kicked continuously by small random currents; each kick
  charges the capacitance, and the charge leaks out with a time constant. For kicks of charge
  $Q_i$ at Poisson times $\tilde t_i$ with rate $\lambda$ onto an isopotential membrane with capacitance
  $C_{\rm cell}$, input resistance $R_{\rm cell}$ and $\tau_m = R_{\rm cell}C_{\rm cell}$,

$$ \mathcal V(t) - \mathcal V_{\rm rest} = \sum_i \frac{Q_i}{C_{\rm cell}}\,\mathrm e^{-(t - \tilde t_i)/\tau_m}\,\mathbb 1_{t \ge \tilde t_i} \ \Longrightarrow\ \operatorname{Cov}\bigl(\mathcal V(t), \mathcal V(t + h)\bigr) = \frac{\lambda\,\mathbb E[Q^2]\,\tau_m}{2C_{\rm cell}^2}\,\mathrm e^{-|h|/\tau_m} \tag{19} $$

  by Campbell's theorem **[textbook]**. Two samples $h$ apart share every kick that arrived
  before the earlier one, attenuated by $\mathrm e^{-h/\tau_m}$: that shared part is the
  correlation. Many small kicks give an OU process, the stochastic leaky integrator used as the
  standard model of subthreshold membrane fluctuations (Jahn et al. 2011 **[PubMed FT]**; in their
  turtle motoneurons under network activity a state-dependent noise fitted better). Even a passive
  resistor-capacitor at thermal equilibrium has such a voltage, with a correlation time equal to
  the product of its resistance and capacitance (Johnson-Nyquist noise) **[textbook]**.
- **The spectral picture.** By Wiener-Khinchin **[textbook]**,

$$ \gamma_{\rm OU}(h \mid \sigma, \tau) = \sigma^2\,\mathrm e^{-|h|/\tau} \ \Longleftrightarrow\ P_{\rm OU}(\nu \mid \sigma, \tau) = \frac{2\sigma^2\tau}{1 + (2\pi\nu\tau)^2}, \qquad \nu_c = \frac{1}{2\pi\tau}, \tag{20} $$

  for one OU component, two-sided ($\int P_{\rm OU}\,d\nu = \sigma^2$). Any stationary process is a
  superposition of sinusoids with random phases, weighted by its spectrum (spectral
  representation) **[textbook]**. A few sinusoids never decorrelate: one random-phase sinusoid has
  autocorrelation $\cos 2\pi\nu h$, back to 1 every period. A continuum dephases: for a frequency
  $F$ drawn from the density $P_{\rm OU}(\nu \mid \sigma, \tau)/\sigma^2$, a Cauchy law of scale $\nu_c$,

$$ \mathbb E[\cos(2\pi F h)] = \mathrm e^{-|h|/\tau} \tag{21} $$

  (the Cauchy characteristic function) **[textbook; checked by simulation, run]**. The Lorentzian
  (20) has no peak, so the noise wanders without a preferred period. Conversely, a resonant membrane is still a continuum of
  random-phase sinusoids, but concentrated near $\pm\nu_0$, the frequency at which the autocorrelation
  oscillates; the width of that concentration gives the forgetting. The autocorrelation is then a damped oscillation, exactly
  $\mathrm e^{-|h|/\tau}\cos 2\pi\nu_0 h$ for two Lorentzian lines of half-width $1/(2\pi\tau)$ at
  $\pm\nu_0$, and $\mathrm e^{-|h|/\tau}[\cos 2\pi\nu_0 h + \sin(2\pi\nu_0|h|)/(2\pi\nu_0\tau)]$ for a
  noise-driven, underdamped oscillator ringing at $\nu_0$ **[textbook; checked numerically, run]**; in both cases
  the autocorrelation first turns negative no earlier than $1/(4\nu_0)$ **[derived]**. Human pyramidal cells can show an
  I_h-linked subthreshold resonance: a non-zero resonant frequency in 27 % of L2&3, 47 % of L3c and
  40 % of L5 cells, few above 2 Hz (Moradi Chameh et al. 2021 **[KB]**; on h-current kinetics and
  resonance, Rich et al. 2021 **[KB]**). That resonant frequency is $\nu_p$, the peak of the impedance profile
  measured with a ZAP stimulus **[KB]**, not the ringing frequency $\nu_0$. For the oscillator, whose voltage
  spectrum under white current noise peaks where its impedance does, $\nu_0^2 = \nu_p^2 + (2\pi\tau)^{-2}$: the
  ringing is faster than the peak **[derived; checked numerically, run]**; for two Lorentzian lines the summed
  spectrum also peaks below $\nu_0$. A low $\nu_p$ therefore does not by itself
  bound the first zero. A barely resonant cell with $\nu_p$ = 2 Hz and $\tau$ = 50 ms (impedance peak 1.4 % above
  its value at 0 Hz) rings at 3.76 Hz and first crosses zero at 96 ms; with $\tau$ = 100 ms, at 2.56 Hz and 132 ms;
  two Lorentzian lines peaking at 2 Hz cross zero at 99-121 ms for $\tau$ = 50-100 ms **[run]**. Either way the
  crossing falls where demeaning already pulls the sample autocorrelation of a 265 ms window negative: under
  $\hat\theta_{\rm sim}$ its median at 96 and 132 ms is $-0.07$, below zero in 78 % and 75 % of windows
  **[run]**. Within a window the two are hard to tell apart, and only the SS semivariogram (§3.6.4)
  could show a resonance, as an overshoot **[reasoning]**.
- **Sources in these recordings.** With fast synaptic transmission blocked in the Allen bath
  **[KB]**, spontaneous fast synaptic events should be largely absent **[reasoning]**; intrinsic
  channel noise and the recording chain remain. A two-state channel's current has autocorrelation
  $\mathrm e^{-(\alpha + \beta)|h|}$ for opening and closing rates $\alpha$, $\beta$
  **[textbook]**; an exponentially correlated current is the usual approximation to random channel
  gating (Tuckwell et al. 2002 **[PubMed abstract only]**), and through the membrane it gives a
  voltage autocorrelation that combines $\mathrm e^{-|h|/\tau_m}$ and $\mathrm e^{-(\alpha+\beta)|h|}$
  **[derived]**. In rat L4/5 pyramidal neurons in slices at room temperature, somatic
  voltage-noise spectra (2-min records) were flat up to an average cut-off of 1.4 Hz at rest and
  fell more steeply than one Lorentzian above it (Yaron-Jakoubovitch et al. 2008 **[PubMed FT]**);
  in the same preparation voltage noise grows with depolarisation (Jacobson et al. 2005 **[PubMed
  abstract only]**). Rat, room temperature: qualitative only.
- **The fast component.** The cohort estimate of $\rho_f$, 0.179, means a correlation time near
  12 µs, far too fast for a membrane. Pure white noise through a 10 kHz Bessel low-pass (the Allen
  filter **[KB]**; order and realisation not stated there) has a lag-1 (20 µs) correlation of
  0.33-0.87 across the digital realisations tried (2, 4 or 8 poles; phase, magnitude or delay
  normalisation) **[run]**; the estimate is below all of them, so part of the fast variance is not
  shaped by the filter. Source not identified.
- **The slow component.** The cohort estimate of $\tau_s$, about 40 ms (corner 4.0 Hz), is the candidate for membrane
  memory: for an isopotential membrane driven by white current noise the voltage is an OU process
  with $\tau = \tau_m$ **[textbook]**; a cell with dendrites has several modes, so $\tau_s$ would at
  best track the slowest. Testable once per-cell fits exist: per-cell $\tau_s$ against $R_mC_m$
  **[reasoning]**.
- **The variance at seconds** (§3.2.5): origin not identified, membrane or recording chain.

---

### 3.4 Why a measured statistic is not a parameter

*This section establishes the mechanism behind the biases -- demeaning [corrected 2026-10-16: the
larger of their two sources, §3.4.2] -- its exact expectations,
its spectral form, an exact constraint on the sample autocorrelation, and how long windows must be
for bias and for precision.*

#### 3.4.1 Demeaning hides slow variance

Regardless of its amplitude, a fluctuation much slower than the window moves the window's mean
with it, so subtracting the mean removes it; conversely, a fluctuation much faster than the window
averages out of the mean and survives demeaning intact. For a stationary zero-mean process with
autocovariance $\gamma(k \mid \theta)$ and a window of $n$ samples **[textbook]**:

$$ \operatorname{Var}(\bar x \mid \theta, n) = \frac{1}{n^2}\sum_{k=-(n-1)}^{n-1} (n - \lvert k\rvert)\,\gamma(\lvert k\rvert \mid \theta), \tag{7a} $$

$$ \mathbb E[S_0 \mid \theta, n] = n\,\gamma(0 \mid \theta) - n\operatorname{Var}(\bar x \mid \theta, n), \qquad \mathbb E[\hat\sigma^2 \mid \theta, n] = \frac{\mathbb E[S_0 \mid \theta, n]}{n - 1}. \tag{7b} $$

For one OU component with correlation time $\tau$ in a window of duration $T = n\,\Delta t = z\tau$,
the fraction of its variance that survives demeaning, the **visible share**, is **[derived from
textbook OU results; checked against (7a)-(7b)]**

$$ v(z) = 1 - \frac{2}{z^2}\bigl(z - 1 + \mathrm e^{-z}\bigr), \qquad v(z) \approx \frac{z}{3}\ \ (z \ll 1), \qquad v(z) \approx 1 - \frac{2}{z}\ \ (z \gg 1). \tag{8} $$

Under $\hat\theta_{\rm sim}$ ($\tau_s$ = 39.7 ms) **[run; cluster]**:

| Window | $z = T/\tau_s$ | $v(z)$ | $\sqrt{\mathbb E[\hat\sigma^2 \mid \hat\theta_{\rm sim}, n]}$ (mV) | median $\hat\sigma$, simulated (mV) | measured (mV) |
|---|---|---|---|---|---|
| SS 10 ms | 0.25 | 7.9 % | 0.0413 | 0.0408 | 0.0405 |
| LS first 50 ms | 1.26 | 31.5 % | 0.0491 | 0.0460 | 0.0465 |
| LS last 132.5 ms | 3.33 | 57.4 % | 0.0565 | 0.0552 | 0.0551 |
| LS full 265 ms | 6.67 | 74.5 % | 0.0608 | 0.0594 | 0.0594 |

In plain words: a slow wobble is invisible in a short window because the window's own average moves
with it; as windows lengthen, more of it shows, and the length at which it appears is $\tau_s$.
This is fact (ii) of §3.2.3, reproduced.

**The spectral form.** In expectation, demeaning acts on the power spectrum like a high-pass
filter. For a stationary process with two-sided spectral density $P(\nu \mid \theta)$, in
continuous time,

$$ \mathbb E\Bigl[\frac1T\int_0^T \bigl(x(t) - \bar x\bigr)^2 dt \;\Big|\; \theta\Bigr] = \int_{-\infty}^{\infty} P(\nu \mid \theta)\,H_T(\nu)\,d\nu, \qquad H_T(\nu) = 1 - \Bigl(\frac{\sin \pi\nu T}{\pi\nu T}\Bigr)^2, \tag{22} $$

because $\bar x$ is the output of a boxcar average of length $T$, whose power transfer is the
squared sinc **[textbook]**; for an OU component (22) equals $\sigma^2 v(T/\tau)$ **[derived;
checked numerically to six decimals, run]**. $H_T$ is not a sharp threshold: it rises from 0 at
$\nu = 0$ as $(\pi\nu T)^2/3$, passes $\tfrac12$ at $\nu_{1/2} = 0.443/T$, and equals 1 at
every non-zero integer multiple of $1/T$, where a sinusoid has a whole number of periods in the window and zero mean
**[derived]**. Two objects are not to be confused: $\nu_{1/2}$ is a property of the window alone
(44 Hz for 10 ms, 3.3 Hz for 132.5 ms, 2.3 Hz for 190 ms, 1.7 Hz for 265 ms), while $v(T/\tau)$ is
the fraction of one component's power that passes, the Lorentzian-weighted average of $H_T$. With
the slow corner at 4.0 Hz, 94 % of the slow power lies below $\nu_{1/2}$ of the 10 ms window and
8 % survives it; for the 265 ms window, 25 % lies below and 75 % survives **[run]**.

#### 3.4.2 The autocorrelation is a biased ratio

The numerator and denominator of (3) are quadratic forms in the data, so their expectations are
exact **[textbook]**. With $\zeta_j = \sum_{i=1}^{n}\gamma(\lvert i - j\rvert \mid \theta)$,

$$ \mathbb E[S_L \mid \theta, n] = (n - L)\,\gamma(L \mid \theta) - \frac{1}{n}\sum_{j=1}^{n-L}\zeta_j - \frac{1}{n}\sum_{j=L+1}^{n}\zeta_j + (n - L)\operatorname{Var}(\bar x \mid \theta, n), \tag{9} $$

checked under $\hat\theta_{\rm sim}$ against Monte Carlo means of $S_0$ and $S_{50}$ over 4000 SS and 1500 LS windows, to within about two standard errors **[run]**. The
ratio (3) has no closed-form expectation for a two-timescale law: it is tractable for simple cases
such as a single-timescale Markov process but not for processes with several timescales (Zeraati
et al. 2022 **[PubMed FT]**). [corrected 2026-10-16: no *elementary* closed form. For a Gaussian
window the expectation has an exact one-dimensional integral representation, eq. (11) of
`TEEG_bias_of_a_statistic_2026-10-11.md`, proved there and checked against Monte Carlo (0.30597 against
0.30480, SE 0.00073, under $\hat\theta_{\rm sim}$ on a 50 ms window) **[derived; run]**. Its cost, an
eigendecomposition of an $n \times n$ matrix, not the absence of a formula, is why Monte Carlo is used.] Its ratio of expectations,

$$ \mathbb E[\hat\rho(L) \mid \theta, n] \approx \frac{\mathbb E[S_L \mid \theta, n]}{\mathbb E[S_0 \mid \theta, n]}, \tag{10} $$

is biased high at our window lengths (§3.5.2).

**The zero-sum identity.** For every window $\sum_j u_j = 0$, so
$0 = (\sum_j u_j)^2 = S_0 + 2\sum_{L=1}^{n-1}S_L$ and

$$ \sum_{L=1}^{n-1} \hat\rho(L) = -\tfrac12 \tag{23} $$

exactly, for the textbook estimator **[derived; checked to twelve decimals, run]**. The sample
autocorrelation of any window must go negative somewhere and sum to $-\tfrac12$, so a window only a
few $\tau$ long cannot hold the law's long positive tail. For a pure OU on windows of $5\tau$, the
median of (3) at $h = 0.5\tau$, $\tau$ and $2\tau$ is 0.286, $-0.013$ and $-0.152$ against the true
0.607, 0.368 and 0.135, and it crosses zero at $0.96\tau$ **[run]**. Under $\hat\theta_{\rm sim}$
the true $\rho(10\ \text{ms} \mid \hat\theta_{\rm sim})$ is 0.521, while the median of (3) on the
132.5 ms segment is 0.225 (5-95 %: 0.019-0.508); at 100 ms the true value is 0.054 and the median
$-0.031$ (5-95 %: $-0.197$ to 0.063) **[run]**. The measured $-0.022$ at 100 ms (§3.2.3) is what the
law predicts, not anti-correlation, and that column is too broad to be a target. Zeraati et al.
find the same: the bias of the sample autocorrelation arises primarily from the deviation of the
sample mean from the true mean, and grows as the series shortens relative to the timescale
**[PubMed FT]**. [corrected 2026-10-16: demeaning is the larger of two sources, not the only one. For
$\hat\rho(1\ \text{ms})$ on the 265 ms window under $\hat\theta_{\rm sim}$ (3200 windows), computing (3) with
the law's true mean in place of each window's own removes 62 % of the bias (Monte Carlo SE 1.4 %); of the
$-0.041$ that remains, the ratio's weighting by $S_0$ makes $-0.039$ and the factor $(n - L)/n$ $-0.002$
**[run; `TEEG_bias_of_a_statistic_2026-10-11.md` §3.5]**.]

#### 3.4.3 How long a window must be

Two requirements, set by two different lengths. **Bias** is set by the length $T$ of each window,
since each window carries its own unknown mean; **precision** is set by the total observed time
$T_{\rm tot}$, summed over windows. For a pure OU observed continuously for $T_{\rm tot} \gg \tau$,
the Fisher information on the rate $1/\tau$ is $T_{\rm tot}\tau/2$, so that

$$ \frac{\operatorname{SD}(\hat\tau)}{\tau} \approx \sqrt{\frac{2\tau}{T_{\rm tot}}} \tag{24} $$

**[derived; that the variance of an OU rate estimate depends on the rate and the sampling scheme,
not on the mean or the noise level: Kenney et al. 2020, PubMed FT]**. A pure-OU check (200
replicates per case; each window with its own unknown mean; $\hat\tau$ by REML, i.e. the exact
likelihood with the mean integrated out, or by a direct exponential fit to the window's sample
autocorrelation over lags up to $3\tau$) **[run]**:

| Window $T$ | REML: median $\hat\tau/\tau$ [10-90 %] | direct fit: median $\hat\tau/\tau$ |
|---|---|---|
| $2\tau$ | 0.64 [0.19, 16.15] | 0.17 |
| $5\tau$ | 0.87 [0.37, 3.07] | 0.36 |
| $10\tau$ | 0.83 [0.52, 1.73] | 0.49 |
| $30\tau$ | 0.94 [0.68, 1.39] | 0.75 |
| $100\tau$ | 0.99 [0.81, 1.21] | 0.90 |

In a second run (120 replicates per case), with $T_{\rm tot} = 100\tau$ cut into $N_{\rm w}$
windows, REML stays within [0.80, 1.20] for $N_{\rm w} \le 10$ (windows of $10\tau$ or more),
[0.78, 1.31] at $N_{\rm w}$ = 20 and [0.73, 1.40] at $N_{\rm w}$ = 50, while the direct fit falls to
0.93, 0.73, 0.54, 0.34 and 0.17 at $N_{\rm w}$ = 1, 4, 10, 20 and 50: pooling short
windows does not remove a per-window bias. REML's spread follows (24) from about $30\tau$ in total
**[run]**. A direct fit at $5\tau$ returns a decay 2.8 times faster than the truth -- the zero-sum
identity (23) at work.

**What each window's own mean costs.** Each of the 14 statistics of (12) is computed on a demeaned
window, so it is a function of the within-window contrasts and can use at most their information,
the information of the REML likelihood (data-processing inequality **[textbook]**). For a pure OU
with $D$ fixed, observed continuously on a window of duration $T = z\tau$ with its own unknown mean,
that information on the rate $1/\tau$ is

$$ I_{\rm REML}(T) = \tau^2\Bigl[\frac{z-3}{2} + \frac{3-\mathrm e^{-z}}{z+2} + \frac{2}{(z+2)^2}\Bigr], \tag{24b} $$

short of the known-mean $(T+\tau)\tau/2$ by between $\tau^2/2$ ($z \to 0$) and $2\tau^2$
($z \to \infty$): a long window with its own mean is worth a known-mean window about $4\tau$
shorter, and a window much shorter than $\tau$ keeps almost nothing -- under $\hat\theta_{\rm sim}$,
1 % of the known-mean value at 10 ms and 58 % at 265 ms **[derived; checked against dense
computations, run]**. (24) ignores this loss. At Check D's sampling, with $D$ unknown, the same
contrast information predicts REML's widening spread in the second run above: SDs of 0.19 and 0.25
at $N_{\rm w}$ = 20 and 50, against 0.21 and 0.26 from the 10-90 % ranges divided by 2.563, where
(24) gives 0.14 for both **[run]**.

For us: an LS pre-window is about $7\tau_s$, a cell's 5-6 of them $33$-$40\,\tau_s$ in total: 22-25 %
by (24), but at least 27.5-30.1 % for an unbiased estimator computed from these demeaned windows, by
(24b), both for a pure OU **[derived; run]**. 17 SS sweeps of enlarged windows hold 34-65 s, about
$850$-$1650\,\tau_s$: 3.5-4.8 % by (24), and at least 4.6-7.6 % by (24b) with the dependence
between windows included **[derived; run]**. Indirect inference carries the per-window bias on both sides of the comparison, so
it needs windows of at least about $\tau$, for the decay to be visible at all, and enough
$T_{\rm tot}$. A seconds-scale component ($\tau \approx 3$ s) is $0.09\tau$ per LS window; below
$\tau$ a window sees only $D = \sigma^2/\tau$ (§3.5.4), which is why the seconds scale is read
between windows (§3.6.4).

---

### 3.5 Inferring the law

*This section establishes how $\theta$ is estimated (indirect inference, D-033), how the model's
statistics are computed, the cohort estimate with its Monte Carlo spread, why model order is not
selected per sweep, where the method comes from, and (v3.4) how the statistics are weighted (D-048).*

#### 3.5.1 Three routes: direct fit, indirect inference, likelihood

Let $\hat q_m$, $m = 1, \dots, 14$, be measured statistics and $q_m(\theta)$ the value statistic $m$
takes under law $\theta$ when computed by the pipeline's estimator on windows of the real length.

$$ \text{direct fit:}\quad \hat\theta = \arg\min_{\theta \in \Theta} \sum_{L}\bigl[\hat\rho(L) - \gamma(L \mid \theta)/\gamma(0 \mid \theta)\bigr]^2, \tag{11} $$

$$ \text{indirect inference:}\quad \hat\theta = \arg\min_{\theta \in \Theta} \sum_{m=1}^{14}\Bigl[\frac{q_m(\theta) - \hat q_m}{\omega_m}\Bigr]^2, \tag{12} $$

$$ \text{likelihood:}\quad (\hat\theta, \hat\mu) = \arg\max_{\theta \in \Theta,\ \mu \in \mathbb R}\ -\tfrac12\ln\det\Gamma(\theta, n) - \tfrac12(\mathbf x - \mu\mathbf 1)^{\top}\Gamma(\theta, n)^{-1}(\mathbf x - \mu\mathbf 1). \tag{13} $$

- **Direct fit (11)** treats the sample autocorrelation as the law's own; on windows not much
  longer than $\tau_s$ it returns too weak or too fast a slow component (§3.4.2-3.4.3).
- **Indirect inference (12)** does not correct the estimator: it predicts what the estimator
  returns under $\theta$ -- same windows, same demeaning -- and matches that prediction. In plain
  words: *which noise law, if it were the real one, would make our own estimator return exactly
  these numbers on windows exactly this long?* If the model is right and $q_m(\theta)$ is the
  estimator's exact value, the bias is carried on both sides of the comparison and cancels.
  Choosing the parameters under which the expected value of a calibration statistic matches its
  observed value is indirect inference; Weltz, Laber & Volfovsky take the expectation over data
  simulated from the model (Weltz et al. 2024 **[PubMed FT]**). **Decided (D-033):** the per-cell
  law is a least-squares point estimate of this kind; no ABC posterior, no abcTau. The hand-set
  scales $\omega_m$ of (12) are the cohort fit's (§3.5.3); for the per-cell fit (12) is replaced by
  the weighted form (28), with $W$ from the iteration (29) (D-048, §3.5.6).
- **Likelihood (13)** uses all the information if the model is right and gives standard errors
  **[textbook]**, but every nuisance -- offset, tilt, any transient -- must then be in the model, and
  the per-window mean must be handled: profiling $\mu$ biases $\hat\tau_s$ low, and integrating it
  out (REML) changes that bias without removing it (§3.5.4) **[run]**. Zeraati et al. name the trade-off: an
  autoregressive fit avoids the autocorrelation's bias, while the autocorrelation lets factors
  unrelated to the dynamics, such as slow drifts, be removed by resampling **[PubMed FT]**. Used
  here as a diagnostic only (§3.5.4).

#### 3.5.2 Closed form against simulated medians

The cost is

$$ J(\theta) = \tfrac12\sum_{m=1}^{14}\Bigl[\frac{q_m(\theta) - \hat q_m}{\omega_m}\Bigr]^2, \tag{14} $$

with $\omega_m$ = 0.005 mV for SDs and 0.05 for autocorrelations (configured by hand) [corrected
2026-10-16: for the per-cell fit these scales are a labelled comparison only; the default weighting is
$\hat\Sigma^{-1}$ by (29) (D-048, §3.5.6)]. Two ways to
compute $q_m(\theta)$:

- **closed form** $q^{\rm cf}_m(\theta)$: $\sqrt{\mathbb E[\hat\sigma^2 \mid \theta, n]}$ from
  (7a)-(7b) for the SDs and the ratio of expectations (10) for the autocorrelations -- fast, but
  neither is the median of the estimator;
- **simulated medians** $q^{\rm sim}_m(\theta)$: the median of the pipeline's own estimator over
  $M$ simulated windows of the real length, with common random numbers, so that $J^{\rm sim}$ is a
  deterministic function of $\theta$.

Monte Carlo check under $\hat\theta_{\rm sim}$ (4000 SS and 1500 LS windows) **[run]**:
$\sqrt{\mathbb E[\hat\sigma^2 \mid \hat\theta_{\rm sim}, n]}$ matches the mean of the SD estimator to within 0.2-2.2 %, and its
median sits 1.3-3.7 % below its mean; the ratio of expectations (10) overshoots the median of the
textbook autocorrelation by 0.018-0.065 across the ten autocorrelation statistics, worst (0.065)
at 10 ms in the last segment -- more than $\omega_m$ itself. The closed form is therefore a
starting point; whether the per-cell fit uses simulated medians, and with which $M$, is open within
D-033 (§5).

#### 3.5.3 The cohort fit (a diagnostic)

The 14 targets $\hat q_m$ are the cohort medians of §3.2.3: SS 10 ms window ($\hat\sigma$;
$\hat\rho$ at 0.02, 0.1, 1 ms), LS 265 ms window (the same four), LS first 50 ms ($\hat\sigma$), LS
last 132.5 ms ($\hat\sigma$; $\hat\rho$ at 0.02, 0.1, 1, 10 ms). Fitted with
`scipy.optimize.least_squares` **[run]**:

| | $\hat\theta_{\rm cf}$ (closed form) | $\hat\theta_{\rm sim}$ (simulated medians) |
|---|---|---|
| $\sigma_f$ | 0.0380 mV | 0.0383 mV |
| $\rho_f$ ($\tau_f$) | 0.175 (11.5 µs) | 0.179 (11.6 µs) |
| $\sigma_s$ | 0.0508 mV | 0.0547 mV |
| $\tau_s$ | 36.6 ms | 39.7 ms |
| cost | $J^{\rm cf}$ = 1.73 | $J^{\rm sim}$ = 1.14 |

The closed-form fit reaches the same optimum from six starts ($\tau_s$ from 2 to 100 ms); the
simulated-medians fit ($M$ = 600 SS and 300 LS windows) from four starts ($\tau_s$ of 20, 37 and
80 ms, and $\hat\theta_{\rm cf}$) ends at 39.6-39.9 ms. The best single AR(1) reaches $J^{\rm cf}$ = 212
($\sigma$ = 0.0505 mV, $\rho$ = 0.780): it predicts $-0.001$ where the last segment measures 0.445 at
1 ms and 0.226 at 10 ms, and the same SD, 0.050 mV, in the 10 ms and 265 ms windows, where the data
give 0.0405 and 0.0594 mV **[run]**.

**Monte Carlo spread.** Refitting $\hat\theta_{\rm sim}$ with four other draws of the common random
numbers gives $\tau_s$ = 35.6, 39.6, 40.4 and 45.6 ms and $\sigma_s$ = 0.0556-0.0597 mV, while
$\sigma_f$ and $\rho_f$ move by at most 0.0005 mV and 0.006 **[run]**. At $M$ = 300 LS windows, then,
the simulated-medians estimate of $\tau_s$ carries a Monte Carlo error of several ms. For per-cell fits, $M$ must make this
error small against the per-cell statistical uncertainty (§3.8.2) -- open (§5).

**What pins $\tau_s$.** Profile of $J^{\rm sim}$, the other three parameters refitted at each fixed
$\tau_s$ **[run]**:

| $\tau_s$ (ms) | 10 | 20 | 30 | **39.7** | 45 | 60 | 100 | 300 |
|---|---|---|---|---|---|---|---|---|
| $J^{\rm sim}$ | 7.78 | 2.21 | 1.38 | **1.14** | 1.27 | 1.64 | 1.95 | 2.94 |
| $\sigma_s$ (mV) | 0.044 | 0.048 | 0.051 | **0.055** | 0.057 | 0.061 | 0.073 | 0.114 |

$\tau_s$ is bounded well from below -- a faster component would show most of its variance within
50 ms, leaving no room for the SD's growth from 0.0465 to 0.0594 mV between 50 and 265 ms -- and
poorly from above: with a larger $\sigma_s$, a slower component raises the cost only slowly,
because 265 ms is the longest LS window **[run; reasoning]**. $J$ is a scaled distance between
medians, not a likelihood, so no interval is read from it.

Model against target **[cluster; run]**:

| Statistic | target $\hat q_m$ | $q^{\rm cf}_m(\hat\theta_{\rm cf})$ | $q^{\rm sim}_m(\hat\theta_{\rm sim})$ |
|---|---|---|---|
| SS $\hat\sigma$ (mV) | 0.0405 | 0.0408 | 0.0408 |
| SS $\hat\rho$(0.02 ms) | 0.274 | 0.280 | 0.271 |
| SS $\hat\rho$(0.1 ms) | 0.153 | 0.123 | 0.109 |
| SS $\hat\rho$(1 ms) | 0.032 | 0.070 | 0.058 |
| LS $\hat\sigma$ (mV) | 0.0594 | 0.0584 | 0.0594 |
| LS $\hat\rho$(0.02 ms) | 0.668 | 0.650 | 0.658 |
| LS $\hat\rho$(0.1 ms) | 0.613 | 0.574 | 0.583 |
| LS $\hat\rho$(1 ms) | 0.564 | 0.553 | 0.560 |
| first 50 ms $\hat\sigma$ (mV) | 0.0465 | 0.0480 | 0.0460 |
| last 132.5 ms $\hat\sigma$ (mV) | 0.0551 | 0.0546 | 0.0552 |
| last $\hat\rho$(0.02 ms) | 0.593 | 0.599 | 0.604 |
| last $\hat\rho$(0.1 ms) | 0.514 | 0.512 | 0.515 |
| last $\hat\rho$(1 ms) | 0.445 | 0.487 | 0.485 |
| last $\hat\rho$(10 ms) | 0.226 | 0.272 | 0.240 |

The remaining systematic misfit is in the SS window at 0.1 and 1 ms (0.109 against 0.153; 0.058
against 0.032), where the exponential shape of the fast component is the likely limitation
(§3.3.3), and in the last segment at 1 ms (0.485 against 0.445) **[reasoning]**.

#### 3.5.4 Model order is not selected per sweep

The law's order is fixed -- two OU components plus the tilt -- and is not chosen per sweep, for
example by the elbow of a BIC curve over sums of $K_{\rm OU}$ OU processes (D-034 (i), **[user]**).
The reasons follow. The checks ran under
the cohort law fitted on 2026-09-28 with the pre-D-028 estimator, (0.0381 mV, 0.179, 0.0560 mV,
37.3 ms), within 7 % of $\hat\theta_{\rm sim}$ in every parameter; the first was recomputed under
$\hat\theta_{\rm sim}$ **[run]**.

- **Inside a window, a seconds-scale component shows only through $D = \sigma_3^2/\tau_3$.** Adding
  an OU of $\sigma_3$ = 0.142 mV and $\tau_3$ = 3 s to $\hat\theta_{\rm sim}$ changes the 14
  statistics (ratio of expectations) by up to 0.055, but an OU of 30 s with the same $D$ differs
  from it by less than 0.001 in every statistic **[run]**: its shape is invisible within 265 ms.
- **BIC needs a likelihood,** $\mathrm{BIC} = n_{\rm par}\ln n_{\rm obs} - 2\ln\hat{\mathcal L}$ with $n_{\rm par}$ free
  parameters and $n_{\rm obs}$ the observations entering the likelihood (Vrieze 2012 **[PubMed
  FT]**). It selects consistently only when the true model is among the candidates, and the
  comparison of $K_{\rm OU}$ against $K_{\rm OU} + 1$ components is non-regular, the extra SD sitting
  on its boundary at 0 **[textbook]**.
- **The seconds scale is not detected per sweep or per cell.** With the exact Gaussian likelihood
  (celerite2, each window's mean profiled), BIC chose $K_{\rm OU}$ = 2 on 24 of 24 single 265 ms
  windows, both without and with the 3 s component, and on 12 of 12 sets of 5 windows in either
  case **[run]**.
- **A non-exponential fast part creates spurious order.** With the fast component replaced by
  a digital 4-pole Bessel-filtered noise (designed at 50 kHz with a 10 kHz phase-normalised
  cutoff; lag-1 0.811) plus white noise, mixed to the same SD and lag-1 correlation, BIC chose
  $K_{\rm OU}$ = 3 in 12 of 12 cells, splitting the fast part into components at the 3 µs bound and
  at 23-26 µs **[run]**.
- **Per-sweep $\hat\tau_s$ is too noisy to be a per-sweep quantity, and its bias depends on how each
  window's mean is handled.** With the mean profiled, the 10/50/90 % quantiles over 24 single windows
  are 10.4/24.3/34.5 ms (truth 37.3 ms); integrating it out (REML) gives 11.4/30.7/46.7 ms; over 12
  sets of five windows, 21.3/31.4/36.8 ms profiled against 25.5/42.1/52.7 ms by REML **[run]**: REML
  moves the median from 35 % to 18 % low on one window, and from 16 % low to 13 % high on five.

The seconds scale is therefore read between windows (§3.6.4); the per-cell law keeps two OU
components and the tilt.

#### 3.5.5 Where the method comes from

- **The model.** Zeraati, Engel & Levina 2022 define timescales as the exponential decay rates of
  the autocorrelation and use a sum of OU processes, one per timescale, as the generative model
  **[PubMed FT]**; (6) is their two-timescale form sampled at $\Delta t$. The older biophysical
  root is fluctuation analysis: Anderson & Stevens 1973 obtained a channel-closing rate from the
  spectrum of end-plate current noise **[PubMed abstract only]**.
- **The principle.** Zeraati et al. show that fitting exponentials directly to the sample
  autocorrelation biases timescales, simulate the generative model with the same duration and
  number of trials as the data, compute the same summary statistic on both, and sample an
  approximate posterior by adaptive ABC; the posteriors peak near the true timescales when the same
  statistic is used on both sides **[PubMed FT]**. Indirect inference rests on the same principle
  and solves for a point estimate instead (Weltz et al. 2024 **[PubMed FT]**; the comparison is
  **[reasoning]**; the method's original references, Gouriéroux, Monfort & Renault 1993 and Smith
  1993, are cited from memory, not read).
- **What is ours.** The SDs of four windows of different length as statistics (the
  variance-versus-window signature that pins $\tau_s$ from below); the closed form as a starting
  point; least squares with hand-set scales in place of ABC (D-033) [corrected 2026-10-16: weighted
  by the simulated inverse covariance, D-048, §3.5.6], which gives a point estimate
  only: no posterior, no interval, no test of one timescale against two. Zeraati et al.'s package
  abcTau (BSD-3-Clause, GitHub, not on PyPI **[src]**) is not used.

#### 3.5.6 Weighting the statistics: the iterated simulated inverse covariance (D-048)

*Added in v3.4. This subsection establishes how the statistics are weighted against each other in the
per-cell fit, why the hand-set scales of (12) and (14) stop being the default, and what the new
weighting is sensitive to. Every number below comes from checks with **only $\tau_s$ free**, the ten LS
statistics of §3.5.3 (265 ms window: $\hat\sigma$ and $\hat\rho$ at 0.02, 0.1, 1 ms; first 50 ms:
$\hat\sigma$; last 132.5 ms: $\hat\sigma$ and $\hat\rho$ at 0.02, 0.1, 1, 10 ms), medians over $K$ = 6
windows, truth $\hat\theta_{\rm sim}$; with all four parameters free, or with the SS statistics, nothing
here was measured **[open]**.*

**The weighted cost.** For cell $c$, collect its $d$ statistics into $\hat{\mathbf q}(c) \in \mathbb R^d$,
each the median over the cell's $K_m$ windows of length $n_m$ (computed level), and the model's values
into $\mathbf q(\theta) = (q_1(\theta), \dots, q_d(\theta))^\top$ (analytic level, computed by simulated
medians, §3.5.2). For every symmetric positive-definite $W \in \mathbb R^{d \times d}$,

$$ J_W(\theta \mid c) = \tfrac12\bigl(\mathbf q(\theta) - \hat{\mathbf q}(c)\bigr)^{\top} W \bigl(\mathbf q(\theta) - \hat{\mathbf q}(c)\bigr), \qquad \hat\theta_W(c) = \arg\min_{\theta \in \Theta} J_W(\theta \mid c). \tag{28} $$

The cost (14) is (28) with $W = \operatorname{diag}(\omega_1^{-2}, \dots, \omega_{14}^{-2})$ and the cohort medians in
place of $\hat{\mathbf q}(c)$; D-048.1 writes the same cost without the factor $\tfrac12$, which does not move
the minimiser.

**Which $W$.** Let $\Sigma(\theta \mid \mathbf n, \mathbf K) = \operatorname{Cov}\bigl(\hat{\mathbf q}(c) \mid \theta, \mathbf n, \mathbf K\bigr)$
be the covariance matrix of the statistic vector of a cell whose law is $\theta$ and whose statistics come
from windows of lengths $\mathbf n = (n_1, \dots, n_d)$ and counts $\mathbf K = (K_1, \dots, K_d)$ (analytic
level). The weighting that minimises the variance of the estimates is the inverse of the covariance
matrix of the fitting statistics (Kaczmarska, Isham & Northrop 2015 **[PubMed FT]**, who cite Hansen and
Hall for the estimator's consistency; Hansen 1982 from memory, not read); $W = \operatorname{diag}(\mathrm{SE}_1^{-2}, \dots, \mathrm{SE}_d^{-2})$, with
$\mathrm{SE}_m = \Sigma_{mm}^{1/2}$, keeps only its diagonal. In plain words: $\Sigma^{-1}$ gives each
combination of statistics a weight inversely proportional to how much that combination scatters from
cell to cell under the law, so statistics that move together are counted once, and the contrasts between
them, which scatter little, weigh most.

Measured on 1500 simulated cells **[run, `check_weights.py`]**:

| Weighting | robust SD of $\hat\tau_s$ | 10-90 % of $\hat\tau_s$ | median $\hat\tau_s$ (truth 39.74 ms) |
|---|---|---|---|
| hand-set $\omega_m$ | 16.98 ms | 23.2-66.7 ms | 39.02 ms |
| $\operatorname{diag}(\mathrm{SE}_m^{-2})$ | 15.53 ms | 23.8-63.7 ms | 38.90 ms |
| $\Sigma^{-1}$ | **7.57 ms** | 31.9-51.3 ms | 38.79 ms |

The hand-set scales lie within a factor 2 of the SEs ($\omega_m/\mathrm{SE}_m$ = 0.65-1.85), so the hand
weighting is close to the diagonal one; the factor of about 2 comes from the correlations. The medians
of $\hat\rho$ at 0.02, 0.1 and 1 ms within one window correlate at 0.993-0.997, and the participation
ratio of the correlation matrix is 2.07 of 10 **[run, `check_cov_structure.py` E]**: the diagonal
counts as ten what is about two. Kaczmarska et al. used the diagonal, following Jesus & Chandler's
finding that it is close to optimal for their rainfall statistics **[PubMed FT; Jesus & Chandler not
read]**; for our statistics, several of which are near-duplicates, it is not **[run; reasoning]**.

**Why $\hat\Sigma$ comes from simulation.** $\Sigma$ depends on $\theta$: the SE of the 1 ms
autocorrelation median on the 265 ms window changes by a factor 1.61 as $\tau_s$ goes from 20 to 80 ms
**[run, `check_sigma_theta.py`]**. One cell's own windows cannot estimate it: the sample covariance of
$K$ = 6 windows has rank at most 5 $< d$ (5 in every simulated cell), and a window-level bootstrap
covariance of the cell's medians is near-singular (condition number, defined below, median
$8.2 \times 10^6$, 10-90 % $5.8 \times 10^5$-$1.2 \times 10^{17}$, against $6.7 \times 10^4$ for the true
$\Sigma$) and fits as badly as the hand weights (robust SD of $\hat\tau_s$ 16.33 against 17.08 ms on the same
cells) **[run, `check_cov_source.py`]**. Its diagonal is usable: the window-level bootstrap SE of a median
over $K$ = 6 windows, $\widehat{\mathrm{SE}}_m(c)$, is centred on the true SD (median ratio 0.99) but spreads
0.58-1.52 (10-90 %) from cell to cell; its Monte Carlo (internal) error at $N_{\rm boot}$ = 2000 replicates is 0.019 of
the true SD, against an SD of 0.365 for the cell-to-cell (sampling) part (Efron & Narasimhan 2020 §3 for the
split **[KB]**) **[run, `boot_1013` E2, `boot_err_1014`]**. Resampling the samples of one window does not
estimate the SE at all: iid resampling gives an SD of 0.0087 ($\approx 1/\sqrt n$) against the true 0.115,
and moving, circular and stationary block bootstraps reach at most about two thirds of the true SD
**[run, `boot_1013` E1]**, because the window's own slow component is not resampled.

**The procedure (D-048, [user]).**

$$ \hat\theta^{[0]}(c) = \hat\theta_{W^{[0]}}(c); \qquad W^{[\iota+1]} = \hat\Sigma\bigl(\hat\theta^{[\iota]}(c) \bigm| \mathbf n(c), \mathbf K(c)\bigr)^{-1}, \quad \hat\theta^{[\iota+1]}(c) = \hat\theta_{W^{[\iota+1]}}(c), \qquad \iota = 0, 1, 2, \dots \tag{29} $$

where $\hat\Sigma(\theta \mid \mathbf n, \mathbf K)$ is the sample covariance (ddof 1) of the statistic vectors
of $R$ cells simulated from the law at $\theta$ with the cell's own $\mathbf n(c)$ and $\mathbf K(c)$,
through the pipeline's estimators (D-028); the iteration stops when the minimised cost plateaus. $W^{[0]}$
is built from the cell's real data alone: $\operatorname{diag}\bigl(\widehat{\mathrm{SE}}_m(c)^{-2}\bigr)$ (the
assistant's reading) or the hand-set $\omega_m$ -- **[open]**, as are $R$, the plateau tolerances and the
iteration cap (proposals in `TEEG_Stage7_covariance_weighting_handoff_2026-10-16.md`: $R$ = 600 per
iteration and 3000 for the last $\hat\Sigma$; 1 % on the cost and on each component of $\hat\theta$; at
most 6 iterations **[reasoning]**). One step from a first estimate, at $R$ = 600, gave a robust SD of 7.25 ms
against 7.61 ms with the true $\Sigma$ on the same cells **[run, `check_cov_source.py`]**. Kaczmarska et
al. describe this two-step form -- artificial data simulated at an initial estimate obtained with the
diagonal weighting, then a refit with the full matrix computed from the simulated data -- and its cost in
computation time **[PubMed FT]**; its iterated form is iterated GMM (Hansen, Heaton & Yaron 1996, from
memory, not read). At $W = \Sigma^{-1}$ and the true law, $2J_W$ at its minimum is asymptotically
$\chi^2_{d - n_{\rm par}}$ (Hansen's overidentification statistic) **[textbook, from memory]**; here it had
mean 9.22 and median 7.51 against $\chi^2_9$'s 9.00 and 8.34 **[run, `check_weights.py`]**, so it serves as a
misfit alarm, not as a test.

**The condition number, two readings.** For a symmetric positive-definite $A$,
$\operatorname{cond}(A) = \Lambda_{\max}(A)/\Lambda_{\min}(A)$. *Statistical reading:* for every non-zero
$\mathbf a \in \mathbb R^d$, $\Lambda_{\min}(\Sigma) \le \mathbf a^\top\Sigma\,\mathbf a/\mathbf a^\top\mathbf a \le \Lambda_{\max}(\Sigma)$
(Rayleigh quotient) **[textbook]**, so $\operatorname{cond}(\Sigma)$ is the ratio between the variances of the
widest and the thinnest unit-norm combination of the statistics, and $\Sigma^{-1}$ weights the thinnest
$\operatorname{cond}(\Sigma)$ times the widest. *Numerical reading:* for a deviation
$\Delta\Sigma = \Sigma_{\rm used} - \Sigma$ of relative size $\varepsilon = \lVert\Delta\Sigma\rVert_2/\lVert\Sigma\rVert_2$,
for every $\Delta\Sigma$ with $\operatorname{cond}(\Sigma)\,\varepsilon < 1$,

$$ \frac{\lVert \Sigma_{\rm used}^{-1} - \Sigma^{-1}\rVert_2}{\lVert \Sigma^{-1}\rVert_2} \le \frac{\operatorname{cond}(\Sigma)\,\varepsilon}{1 - \operatorname{cond}(\Sigma)\,\varepsilon}, \tag{30} $$

with equality for $\Delta\Sigma = -\varepsilon\,\Lambda_{\max}(\Sigma)\,\mathbf v_{\min}(\Sigma)\mathbf v_{\min}(\Sigma)^\top$
**[textbook, from memory; equality checked to $10^{-9}$, run, `fig_condition_number.py`]**: an error that is
a fraction $\varepsilon$ of the largest variance, aimed at the thinnest axis, removes a fraction
$\operatorname{cond}(\Sigma)\,\varepsilon$ of the smallest. $\operatorname{cond}(\Sigma)$ depends on units --
$6.67 \times 10^4$ with the SDs in mV, $3.61 \times 10^6$ in µV, while the fit is unit-invariant (the same
$\hat\tau_s$ for 11 of 11 inputs) -- so the number to report is the unit-free
$\operatorname{cond}(\Sigma^{\rm corr})$ of the correlation matrix, $2.60 \times 10^3$ here **[run,
`check_cov_structure.py` E]**. Its thinnest axis, $\Lambda_{\min}(\Sigma^{\rm corr})$ = 0.0025, is mainly
$\hat\rho(0.1\ \text{ms}) - \hat\rho(0.02\ \text{ms})$ on the 265 ms window **[run]**.

**Monte Carlo error is not amplified by the condition number.** With $\hat\Sigma$ the sample covariance of
$R$ draws, $\Sigma^{-1/2}\hat\Sigma\,\Sigma^{-1/2}$ is the sample covariance of $R$ whitened draws, whose law
does not involve $\Sigma$ **[derived]**: the distortion of the cost along any direction depends on $R$ and
$d$ only (eigenvalues equal to those of the identity case to $2.9 \times 10^{-12}$). At $R$ = 600 the ratio of
the cost under $\hat\Sigma^{-1}$ to the cost under $\Sigma^{-1}$ lies in 0.76-1.36 over all directions (2.5 % of
minima to 97.5 % of maxima), at $R$ = 3000 in 0.88-1.14; the relative error of $\hat\Sigma^{-1}$ has median
0.072 and 0.029, although $\operatorname{cond}(\Sigma)\,\varepsilon$ has median 3683 and 1489 **[run,
`check_cov_structure.py` A; Gaussian draws]**. (30) is a worst case for an error aimed at the thin axis;
Monte Carlo error is relative axis by axis. Independent errors of the same size on each entry would make
$\hat\Sigma$ indefinite in 300 of 300 replicates **[run]**.

**What $\Sigma^{-1}$ is sensitive to: model error.**

- *A noise source the simulator lacks.* It adds a positive semi-definite matrix to the true $\Sigma$, so
  $\Delta\Sigma$ is negative semi-definite: $\hat\Sigma$ is understated along every contrast the source
  reaches, which is the unbounded side of (30) -- along an eigen-axis the weight becomes
  $1/(1 - \operatorname{cond}(\Sigma)\,\varepsilon)$ times too large, while an overstatement would give the
  bounded $1/(1 + \operatorname{cond}(\Sigma)\,\varepsilon)$ **[derived; run]**. With independent extra
  scatter raising every statistic's variance by a fraction ($\Sigma = \Sigma_{\rm used} + 0.01\operatorname{diag}(\Sigma_{\rm used})$
  for 1 %), Gaussian inputs, 3000 replicates **[run, `check_missing_noise.py`]**:

| Extra variance per statistic | weight on the thinnest axis of $\Sigma^{\rm corr}$, used/right | mean minimised $2J$ under $\Sigma_{\rm used}^{-1}$ | robust SD of $\hat\tau_s$: $\Sigma_{\rm used}^{-1}$ / true $\Sigma^{-1}$ / diagonal |
|---|---|---|---|
| 0 | 1.00 | 8.93 | 6.27 / 6.27 / 16.62 ms |
| 0.1 % | 1.40 | 9.81 | 6.66 / 6.70 / 16.74 ms |
| 1 % | 5.02 | 18.83 | 8.65 / 7.83 / 16.47 ms |
| 10 % | 41.18 | 112.30 | 24.34 / 10.07 / 16.01 ms |

  ($\chi^2_9$'s 95 % point is 16.9.) The cost rises well before the estimate degrades; at 10 % the full
  weighting is worse than the diagonal. The precision computed from $\Sigma_{\rm used}$ alone,
  $\bigl((\nabla_{\tau_s}\mathbf q)^\top\Sigma_{\rm used}^{-1}\nabla_{\tau_s}\mathbf q\bigr)^{-1/2}$ = 5.07 ms, is the same at every level.
- *A biased statistic.* With a noise-free input shifted by one SE in one statistic at a time (the kind of
  misfit §3.5.3 shows at SS 0.1 and 1 ms), $\hat\tau_s$ moves by up to 56.5 ms under $\Sigma^{-1}$, against at
  most 2.8 ms with the diagonal and 4.7 ms with the hand-set weights **[run, `check_cov_structure.py` C]**.
  Every shift above 8 ms comes with a minimised $2J$ of 113-238, far above $\chi^2_9$; the 7.3 ms shift
  from the 10 ms autocorrelation comes with 4.1 and would not be flagged **[run]**.

In plain words: the full weighting buys about a factor 2 in precision when the law is right, by trusting
the contrasts between near-duplicate statistics, and those are exactly what a missing or misshapen noise
component disturbs. That is why D-048 requires the in-silico sensitivity check before the fit is trusted
on real cells, and why the minimised cost is read against $\chi^2_{d - n_{\rm par}}$ as an alarm
**[reasoning]**. The check is planned (handoff §4: S0 closure; S1 the tilt replaced by a seconds-scale OU;
S2 a missing component, a sweep-locked transient or extra scatter; S3 the alarm on the real cells), not run.

**Reported precision.** The linearised SD
$\bigl((\nabla_{\tau_s}\mathbf q)^\top\Sigma^{-1}\nabla_{\tau_s}\mathbf q\bigr)^{-1/2}$ is optimistic even when the law
is right: 5.07 ms, against 6.38 ms for the same nonlinear fit with Gaussian inputs and 7.57 ms with
simulated medians; for the diagonal and hand-set weightings the linearised (sandwich) SD agrees with the
simulated one within 3-7 % **[run, `check_cov_structure.py` B, D]**. Per-cell precision under (29) is
therefore taken from simulation (closure runs, or a parametric bootstrap at $\hat\theta(c)$), not from the
linearised formula **[reasoning]**.

---

### 3.6 The seconds scale and the per-sweep tilt

*This section establishes what a creep and a wander are, how each enters the LS loss, how large the
variance at seconds is, how the tilt is calibrated from the SS window means, and what a tilt can
and cannot reproduce.*

#### 3.6.1 Creep and wander

A **creep** is a change of the baseline that is approximately linear across the sweep: inside a
window an offset plus a line, and it continues through the step. A **wander** is a stationary
fluctuation whose correlation time exceeds the window: inside the window it looks like a random
walk, and over seconds it returns towards the cell's mean level. One 265 ms window cannot tell
them apart, and the trend share (4) responds to both: under $\hat\theta_{\rm sim}$ the cohort
median of per-cell 5-sweep medians has a null of 0.074-0.078 $\pm$ 0.018-0.020 ($\Pr(\ge 0.14)$ = 0.001-0.006), against the measured 0.14; with a seconds-scale OU of 0.142 mV and 3 s added, the null
becomes 0.111 $\pm$ 0.030, $\Pr(\ge 0.14)$ = 0.17 **[run; cluster]**. The excess linear structure is
what the variance at seconds produces. The LS tail decided the question (§3.2.5): no creep. Here
*drift* means a creep, *slow noise* a stationary component however slow; the tilt $b$ is the law's
carrier of the variance at seconds, whatever that variance's shape (D-034).

#### 3.6.2 How the slow terms enter the Long Square loss

The LS residual is scored over the 1 s step after the pre-window mean $\bar x_{\rm pre}$ is
subtracted (§3.1.2). Measure $t$ from the step onset.

- **Offset:** $a$ cancels exactly.
- **Stationary noise:** the baseline-subtracted stationary noise at time $t$ has RMS
  $B(t \mid \theta)$ with

$$ B(t \mid \theta)^2 = \gamma(0 \mid \theta) + \operatorname{Var}(\bar x_{\rm pre} \mid \theta, n) - 2\operatorname{Cov}\bigl(e^{\rm stat}(t), \bar x_{\rm pre} \mid \theta, n\bigr), \tag{15} $$

  bounded: under $\hat\theta_{\rm sim}$, 0.067 mV at onset, 0.071 mV at 50 ms and 0.072 mV from
  100 ms on, the fast part 0.038 mV and the slow part 0.061 mV in quadrature **[run]**. The SD of
  the step-mean baseline error (step mean minus pre-window mean) is 0.031 mV **[run]**.
- **Tilt:** relative to the pre-window mean, whose time is the window's midpoint 137.5 ms before
  onset, it adds

$$ b\,(t + 0.1375\ \text{s}), \qquad 0 \le t \le 1\ \text{s}, \tag{16} $$

  an error that grows linearly over the step. Any variance at seconds adds an error that grows over
  the step; its size and shape are the subject of the rest of this section.

**Against the I_h signal.** In human cortex the L2&3 sag voltage amplitude is 1.7 $\pm$ 1.1 mV (56
cells) and the L3c one 2.3 $\pm$ 0.7 mV (15 cells), measured with 600 ms steps down to $-400$ pA; in
13 L2&3 cells ZD7288 reduced it from 1.3 $\pm$ 0.9 to 0.2 $\pm$ 0.3 mV (Moradi Chameh et al. 2021
**[KB]**). The Allen LS steps start at $-110$ pA (or $-220$ pA in some low-resistance human
cells) **[KB]**, so for comparable cells the sag at our amplitudes is smaller still **[reasoning]**. A baseline error that grows over the step competes with it: if the baseline
rises during a hyperpolarising step, the late response rises toward rest as sag does and the fit
reads more I_h; conversely, if it sinks, the sag is masked and the fit reads less I_h. Each LS
amplitude is a single sweep, so nothing averages the error out; across a cell's sweeps, errors of
random sign add variance to the estimate, while an error of consistent sign biases it **[reasoning;
not simulated]**.

#### 3.6.3 How large the variance at seconds is, and where it can sit

An OU component $(\sigma_3, \tau_3)$ added to $\hat\theta_{\rm sim}$ (without refitting) and sized to
reproduce the cohort's 6.89 s change (median $|\Delta_{\rm end}|$ 0.128 mV) **[run]**:

| $\tau_3$ | $\sigma_3$ (mV) | 265 ms window, $\sqrt{\mathbb E[\hat\sigma^2 \mid \hat\theta_{\rm sim}, \sigma_3, \tau_3, n]}$ (mV; without the component 0.0608) | step-mean error of the component (mV) |
|---|---|---|---|
| 0.3 s | 0.159 | 0.099 | 0.143 |
| 1 s | 0.140 | 0.073 | 0.104 |
| 3 s | 0.142 | 0.066 | 0.070 |
| 10 s | 0.189 | 0.063 | 0.054 |
| 30 s | 0.294 | 0.063 | 0.049 |

A component of 0.3 or 1 s would raise the 265 ms window's SD by 60 % or 20 %, far beyond its
per-cell precision (§3.8.2), so, if the variance at seconds is stationary noise, it sits at
$\tau_3 \gtrsim 3$ s **[run; reasoning]**. Its effect on the fit depends on its shape: a step-mean error of 0.049-0.070 mV for
an OU of 3-30 s, against 0.031 mV from the 40 ms component **[run]**. The shape is what the SS
window means measure.

#### 3.6.4 Calibrating the tilt from the SS window means (D-034)

Within one SS sweep the means of successive noise windows sample the baseline every 200 ms over
about 4 s. For two windows of durations $T_1$, $T_2$ whose centres lie $h$ apart in one sweep, the
semivariogram of their means under law (5a) is

$$ V(h \mid \theta, \sigma_b) = \tfrac12\,\mathbb E\bigl[(\bar x_{(2)} - \bar x_{(1)})^2\bigr] = \tfrac12\sum_{\text{OU comps}}\Bigl[\operatorname{Var}_{T_1} + \operatorname{Var}_{T_2} - 2\operatorname{Cov}_{T_1,T_2}(h)\Bigr] + \tfrac12\sigma_b^2h^2, \tag{25} $$

where, for one OU component $(\sigma, \tau)$, $\operatorname{Var}_T = 2\sigma^2(z - 1 + \mathrm e^{-z})/z^2$
with $z = T/\tau$, and $\operatorname{Cov}_{T_1,T_2}(h) = \sigma^2\tau^2(1 - \mathrm e^{-T_1/\tau})(1 -
\mathrm e^{-T_2/\tau})\,\mathrm e^{-g/\tau}/(T_1T_2)$ with gap $g = h - (T_1 + T_2)/2$ **[derived
from textbook OU results; $z - 1 + \mathrm e^{-z}$ computed as $z + \operatorname{expm1}(-z)$]**.
The tilt term is exact, since the mean of a line over a window is its value at the window's centre;
the offset cancels within a sweep. Per cell, $\hat V_c(h)$ is computed from the window means
(computed level), and the tilt is calibrated as

$$ \hat\sigma_b(c)^2 = \frac{2\,\bigl[\hat V_c(h^*) - V(h^* \mid \hat\theta_c, 0)\bigr]}{h^{*2}}, \tag{26} $$

i.e. (D-034.1), with the cell's fitted stationary law $\hat\theta_c$ plugged in for its own part and
$h^*$ the calibration lag (§3.6.5).

**What $V$ separates** (190 ms windows, under $\hat\theta_{\rm sim}$; each added component except the
last sized to the cohort's 6.89 s change) **[run]**:

| Law | $V$(0.2 s) | $V$(0.64 s) | $V$(1 s) | $V$(3.8 s) (mV²) | median $\lvert\Delta_{\rm end}\rvert$ (mV) |
|---|---|---|---|---|---|
| $\hat\theta_{\rm sim}$ alone | 0.0009 | 0.0010 | 0.0010 | 0.0010 | 0.023 |
| + OU 3 s, $\sigma_3$ = 0.142 mV | 0.0018 | 0.0044 | 0.0063 | 0.0151 | 0.128 |
| + OU 30 s, $\sigma_3$ = 0.294 mV | 0.0013 | 0.0026 | 0.0036 | 0.0111 | 0.128 |
| + tilt, $\sigma_b$ = 0.027 mV/s | 0.0009 | 0.0011 | 0.0014 | 0.0063 | 0.128 |
| + tilt, $\sigma_b$ = 0.119 mV/s (§3.6.5) | 0.0012 | 0.0039 | 0.0081 | 0.1032 | 0.554 |

With the 10 ms pre-pulse windows the floor is 0.0027-0.0028 mV², three times higher **[run]**.
Two components of equal size at 6.89 s already differ at 1 s: across simulated cells with 17 SS
sweeps, the 10-90 % band of $\hat V_c$(1 s) is 0.0051-0.0077 mV² for the 3 s component and
0.0030-0.0043 mV² for the 30 s one, so with many sweeps the semivariogram tells a 3 s from a 30 s
timescale; with 2 sweeps the bands overlap (0.0036-0.0101 against 0.0020-0.0055 mV²) **[run]**. The calibration
(26) needs only the level of $\hat V_c$ at $h^*$, not that timescale.

#### 3.6.5 What a tilt can and cannot reproduce

A tilt's semivariogram grows as $h^2$ and a seconds-scale OU's at most as $h$, so, if the variance at
seconds is OU-like rather than a creep (§3.2.5), a tilt reproduces it at one lag only **[derived]**. The LS loss reads the baseline error at lags
0.14-1.14 s from the pre-window centre, through the step-averaged variance of the baseline-subtracted noise **[reasoning]**. With the seconds-scale part taken as an OU of 3-30 s
matched to the 6.89 s change, under $\hat\theta_{\rm sim}$ **[run]**:

- **matched to the 6.89 s change**, $\sigma_b$ = 0.027 mV/s ($\sigma_\delta$ = 0.20 mV over a
  7.27 s LS sweep): the step-mean error is 0.017 mV against 0.049-0.070 mV, 3-4 times too small;
- **matched to the step-averaged point-error variance**, $\sigma_b$ = 0.083-0.119 mV/s: the tilt's
  point error is then 0.47-0.49 of the OU's at step onset and 1.20-1.23 at its end, and the 6.89 s
  change is over-predicted (median 0.39-0.56 mV against 0.128 mV);
- **read from (26)** with 190 ms windows: $h^*$ = 0.8 s gives $\sigma_b$ = 0.081-0.116 mV/s, within
  3 % of the step-averaged match for every OU of 3-30 s; $h^*$ = 0.6 s gives 11-13 % more and
  $h^*$ = 1.0 s 12-13 % less.

The recommendation for D-034's open calibration lag is therefore $h^*$ = 0.8 s with 190 ms windows,
recomputed for the windows actually used; the twin's post-step stretch then carries too much drift,
which only the report-only rebound residuals read (D-006) **[reasoning]**.

#### 3.6.6 Membrane or electrode?

The Allen cell-level QC required the final voltage to lie within 1 mV of the zeroed electrode
voltage for every 10 minutes of recording **[KB]**: a bound on the net electrode offset, which
excludes an electrode creep of one sign over minutes but not within-sweep changes of varying sign
**[reasoning]**. With no creep in any cell (§3.2.5), the question becomes the origin of the wander,
membrane (§3.3.3) or recording chain: not settled. The generator docstring's "electrode/seal
drift" is unsupported **[src; reasoning]**. Whether every stored sweep passed the Allen sweep-level
QC is not verified: neither the archive writer nor AllenSDK's `get_ephys_sweeps` filters on it
**[src]**; the archive's own LS tail (signed, from the 265 ms pre-window to the last 500 ms of the stored
sweep) answers the creep question directly; the Allen `vm_delta_mv` (unsigned, two 500 ms windows)
is a different statistic with its own null and is not needed.

---

### 3.7 The enlarged SS windows (D-034)

*This section establishes the window geometry, the test that decides where the pulse transient has
died out, and what the enlarged windows buy.*

**Geometry.** The archive stores each SS pulse's $(-10, +200)$ ms window, in temporal order with
its sweep number (Format A of `ss_pulses.npz`) **[src]**. With pulses 200 ms apart, each pulse's window
ends at the next pulse's onset and the next window starts 10 ms earlier, so consecutive windows
overlap by 10 ms. The noise window before a pulse becomes $[t_{\rm SS}(c), 200)$ ms after the
preceding pulse, the late part of that pulse's stored window, with $t_{\rm SS}(c)$ set by the test
below; the late part after the last pulse of a sweep is used too.

**Transient test** (proposal; thresholds open). Per cell, with each window minus its own pre-pulse
mean, let $\bar r_{\rm dep}(t)$ and $\bar r_{\rm hyp}(t)$ be the means over depolarising and over
hyperpolarising pulses, and

$$ \bar r_{\rm odd}(t) = \tfrac12\bigl[\bar r_{\rm dep}(t) - \bar r_{\rm hyp}(t)\bigr]. \tag{27} $$

The pulse response changes sign with the pulse, while noise and anything polarity-independent
cancel in expectation, so (27) isolates the response **[reasoning]**. $t_{\rm SS}(c)$ is the
earliest time after which $|\bar r_{\rm odd}|$, in 5 ms bins, stays below
$\kappa_{\rm tr}\hat\sigma_{\rm SS}$ (proposal $\kappa_{\rm tr}$ = 0.1), with the across-pulse
residual variance flat from $t_{\rm SS}(c)$ on. Format A stores no pulse times **[src]**: two stored
windows are consecutive only if their 10 ms overlap is identical within one sweep number.

**What they buy** (per-cell SD of the median textbook $\hat\rho$ over simulated cells,
$\hat\theta_{\rm sim}$, pulse response assumed removed exactly) **[run]**:

| $t_{\rm SS}$ | window | $\hat\rho$(1 ms): 2 / 17 sweeps | $\hat\rho$(10 ms): 2 / 17 sweeps |
|---|---|---|---|
| 10 ms | 190 ms | 0.022 / 0.009 | 0.027 / 0.011 |
| 50 ms | 150 ms | 0.031 / 0.012 | 0.035 / 0.012 |
| 100 ms | 100 ms | 0.035 / 0.011 | 0.033 / 0.010 |

against 0.059-0.064 at 1 ms (full LS window) and 0.077-0.091 at 10 ms (last LS segment) from 5-6
LS pre-windows (§3.8.2). The L3_exc cells have 37-348 SS pulses, about 2-17 sweeps **[cluster]**.
The enlarged windows serve the calibration of the tilt (§3.6.4) and, if adopted (§5 (d)), D-033's
statistic set; with windows of 100-190 ms they also see 50-67 % of the slow component's variance, against 8 %
in 10 ms (8) **[run]**.

---

### 3.8 Consequences for the per-cell calibration

*This section establishes what is decided and how precisely one cell's statistics constrain its
law; what is open is in §5.*

#### 3.8.1 What is decided

(i) Each twin carries its **own** real cell's law (5a)-(5b) with (6), calibrated on that cell's
window statistics; (ii) **no cohort-level law**: $\hat\theta_{\rm cf}$ and $\hat\theta_{\rm sim}$
are diagnostics, never generator inputs; (iii) the twin's LS pre-window is its cell's, so the
synthetic onset moves from 100 ms to `ls_pre_ms` + 5 ms = 270 ms (D-017, **[user]**). The sample
autocorrelation is (3), through `noise_stats.sample_acf`, on data and simulations alike (D-028,
D-029). The law is fitted by indirect inference (12), point estimates (D-033) [corrected 2026-10-16:
with the weighted cost (28) and $W$ from the iteration (29), the hand-set scales of (12) kept as a
comparison only (D-048, §3.5.6)]. The variance at
seconds is carried by the tilt, $\sigma_b$ from (26); the SS noise windows are enlarged where the
transient test passes (D-034).

#### 3.8.2 Per-cell precision

Under $\hat\theta_{\rm sim}$, SD across simulated cells of the per-cell median statistic **[run]**:

| Per-cell statistic | $K$ = 5 LS sweeps | $K$ = 6 LS sweeps |
|---|---|---|
| full-window $\hat\sigma$ | 0.0044 mV | 0.0041 mV |
| full-window $\hat\rho$(1 ms) | 0.064 | 0.059 |
| last-segment $\hat\sigma$ | 0.0043 mV | 0.0040 mV |
| last-segment $\hat\rho$(1 ms) | 0.080 | 0.072 |
| last-segment $\hat\rho$(10 ms) | 0.091 | 0.077 |
| trend share | 0.077 | 0.062 |

For SS, with 37 pulses the per-cell median $\hat\sigma$ varies by 0.0004 mV and $\hat\rho(1)$ by
0.015; with 348 pulses, by 0.0001 mV and 0.005 **[run]**. The cells differ by more than these
uncertainties (last-segment $\hat\rho$(1 ms) 0.23-0.87, Appendix A), so per-cell laws are
informative; a per-cell $\tau_s$ from LS windows alone carries real uncertainty; the enlarged SS
windows cut the per-cell SD of the autocorrelation statistics about 1.7- to 3.4-fold with 2 SS
sweeps and 5- to 9-fold with 17 (§3.7). With the weighting of D-048, the robust SD of $\hat\tau_s$ from
the ten LS statistics of 6 sweeps falls from 15.53-16.98 ms to 7.57 ms, with only $\tau_s$ free
(§3.5.6) **[run]**.

---

## 4 Summary of results

1. **AR(1) fails on L3_exc.** The last 132.5 ms of the LS pre-window keep textbook autocorrelations
   of 0.445 at 1 ms and 0.226 at 10 ms where AR(1) gives 0.000; within the pre-window the SD grows
   from 0.0465 mV (50 ms) to 0.0551 (132.5 ms) and 0.0594 mV (265 ms) (§3.2.3).
2. **The estimator.** $\hat\rho(L) = S_L/S_0$, eq. (3), through statsmodels (D-028, D-029); older
   values convert exactly by (18), the factor $1 - \ell/T$ depending on the window's duration only
   (§3.2.2).
3. **The law.** Fast AR(1) + slow OU + per-sweep offset + per-sweep tilt, (5a)-(5b), with
   autocovariance (6); the offset cancels everywhere; the tilt alone carries the variance at
   seconds (D-034) (§3.3.1-3.3.2).
4. **Resting noise is memory.** Random kicks onto a leaky membrane give an exponentially decaying
   covariance, (19); equivalently a Lorentzian spectrum, (20), a continuum of random-phase sinusoids
   that dephase into $\mathrm e^{-|h|/\tau}$, (21); a resonant membrane concentrates them around
   $\pm\nu_0$ and gives a damped oscillation (§3.3.3).
5. **Demeaning is a filter.** Exact expectations (7a)-(7b), (9); visible share (8); in spectral
   form (22), a high-pass with half-power point $0.443/T$ (§3.4.1). The sample autocorrelation of
   every window sums to $-\tfrac12$, (23), so short windows bias it low: on the 132.5 ms segment
   the law's true 0.521 at 10 ms reads 0.225 (§3.4.2).
6. **Window length.** Bias is set per window, precision by the total time, (24), less what each
   window's own mean costs, (24b): between $\tau^2/2$ and $2\tau^2$ of information per window, so 5-6
   LS pre-windows give at least 27.5-30.1 % rather than 22-25 % for unbiased estimators; with each
   window's mean unknown, REML is within 6 % from $30\tau$, while a direct fit is still 10 % low at
   $100\tau$ and returns $0.36\tau$ at $5\tau$ (§3.4.3).
7. **Inference.** Indirect inference (12), point estimates (D-033); the ratio of expectations (10)
   overshoots the estimator's median by 0.018-0.065, so the closed form is a starting point
   (§3.5.1-3.5.2).
8. **Cohort law** (diagnostic): $\hat\theta_{\rm sim}$ = (0.0383 mV, 0.179, 0.0547 mV, 39.7 ms),
   $J^{\rm sim}$ = 1.14; Monte Carlo spread of $\tau_s$ over five draws 35.6-45.6 ms at $M$ = 300 LS
   windows; $\tau_s$ bounded well below, loosely above (§3.5.3).
9. **No per-sweep model order.** Within a window a seconds-scale OU shows only through
   $\sigma^2/\tau$; BIC on exact likelihoods never detected it per sweep or per cell, and a
   non-exponential fast part creates spurious order (§3.5.4).
10. **No creep; variance at seconds beyond the law in 17 or 18 of the 18 cells.** Sign agreement
    53/104 between the tail change and a continuing creep; per-cell median $|\Delta_{\rm end}|$
    0.048-0.620 mV against a null with 99th percentile 0.051-0.056 mV, the lowest cell at its
    96th-98th; if stationary, the component sits at $\tau \gtrsim 3$ s (§3.2.5, §3.6.3).
11. **The tilt.** Calibrated from the SS window-mean semivariogram, (25)-(26); it matches the
    variance at seconds at one lag only; read at $h^*$ = 0.8 s with 190 ms windows it reproduces
    the step-averaged baseline error within 3 %; matched to that error, the tilt's point error is
    0.47-0.49 of an OU's at step onset and 1.20-1.23 at its end (§3.6.4-3.6.5).
12. **Enlarged SS windows.** Transient test (27); the per-cell SD of the autocorrelation
    statistics falls about 1.7- to 3.4-fold (2 sweeps) and 5- to 9-fold (17 sweeps) against 5-6
    LS pre-windows, for any start from 10 to 100 ms (§3.7, §3.8.2).
13. **Weighting (D-048).** The per-cell fit minimises (28) with $W = \hat\Sigma^{-1}$, $\hat\Sigma$ simulated
    at the current estimate with the cell's own windows and refined by (29) from a first step built from
    the cell's data. With only $\tau_s$ free it halves the robust SD of $\hat\tau_s$ (7.57 against
    15.53-16.98 ms), because the statistics of one window correlate at 0.993-0.997. It is fragile to model
    error: a missing source understates $\hat\Sigma$ on the unbounded side of (30) (+1 % variance per
    statistic: mean minimised $2J$ 8.9 to 18.8, robust SD 6.3 to 8.7 ms), and one statistic biased by one SE
    can move $\hat\tau_s$ by 56.5 ms. Hence the $\chi^2_{d - n_{\rm par}}$ alarm and the in-silico
    sensitivity check before use on real cells (§3.5.6).

---

## 5 Open points, caveats and assumptions

**Decisions open.**
- (a) **The two high-trend cells** (526785799, 571430815): keep with their calibrated law, flagged
  in the gate report, or exclude. Recommended: keep; both are slow wander, not creep (§3.2.5);
  excluding them from the real fit is a separate QC question.
- (b) **One law per cell for both protocols** (SS and LS from one process, each read through its
  own windows), superseding D-014's per-protocol AR(1) pair: recommended, and needed for an
  SS-calibrated $\sigma_b$ to serve the LS sweeps (D-034 comment (b)).
- (c) **The tilt's calibration lag $h^*$** (D-034): recommended 0.8 s with 190 ms windows,
  recomputed for the windows used (§3.6.5); $\kappa_{\rm tr}$ and the bin width of the transient
  test (§3.7).
- (d) **Within D-033:** closed form or simulated medians, and $M$ (§3.5.2-3.5.3); the statistic set
  (proposed: add the enlarged SS windows' statistics); the scales $\omega_m$ [corrected 2026-10-16:
  the weighting is decided, D-048: $\hat\Sigma^{-1}$ by (29); open within it: $W^{[0]}$, $R$, the plateau
  tolerances and iteration cap, the threshold of the $\chi^2$ alarm (§3.5.6)].
- (e) **$\tau_s$'s loose upper side** (§3.5.3): the enlarged SS windows are at most 190 ms long and
  do not bound it through window length; the SS semivariogram at 0.2-0.6 s could, if the per-cell
  fit uses it (not computed).
- (f) Cells whose two protocols were sampled at different rates (none in L3_exc; other groups mix
  50 and 200 kHz, D-014): how the fast component transfers.
- (g) D-016 (d) (noise differing between the sweeps of one cell) and (e) (LJP level of the spike
  test); D-015's stationarity across a response.
- (h) **The in-silico sensitivity check of D-048** (S0-S3 of
  `TEEG_Stage7_covariance_weighting_handoff_2026-10-16.md`): how much a noise component the law lacks,
  or only approximates (the tilt), moves $\hat\theta$ and its reported precision under (29); and the gain
  and the sensitivities of $\hat\Sigma^{-1}$ with all four parameters free and the SS statistics included
  (only $\tau_s$ free was measured, §3.5.6). Not run.

**Assumptions.**
- The components are stationary, Gaussian and independent, and the pre-stimulus noise is the noise
  during the step. Membrane noise can depend on voltage (in rat L4/5 it grows with depolarisation:
  Jacobson et al. 2005 **[PubMed abstract only]**; Yaron-Jakoubovitch et al. 2008 **[PubMed
  FT]**); our steps hyperpolarise, and the dependence there is not measured.
- The fast component is exponential (AR(1)); the SS misfit at 0.1 and 1 ms (§3.5.3) says this is
  approximate, and a non-exponential fast part creates spurious model order (§3.5.4).
- The tilt is a deliberate simplification: it matches the variance at seconds at one lag (§3.6.5).
  The shape of that variance is unknown; OUs of 3-30 s stand in for it in every calculation here.
- The enlarged-window simulations assume the pulse response is removed exactly (§3.7).
- The weighting $\hat\Sigma^{-1}$ assumes that the simulator's law is the recordings' law along the thin
  contrasts of $\Sigma^{\rm corr}$; a source the simulator lacks is amplified there, not averaged
  (§3.5.6).

**Approximations and their regime.**
- Closed form: exact for $\mathbb E[S_0]$, $\mathbb E[S_L]$; the ratio (10) overshoots the median
  by 0.018-0.065 at our lengths and is exact only for $T/\tau_s \to \infty$.
- Simulated medians: Monte Carlo error of several ms in $\tau_s$ at $M$ = 300 LS windows (§3.5.3).
- The scales $\omega_m$ are set by hand; the statistics overlap (the last segment is part of the
  full window), so a diagonal weighting double-counts. $J$ is not a likelihood. [corrected 2026-10-16:
  no longer the per-cell default (D-048). The double counting was measured: correlations 0.993-0.997
  within one window, robust SD of $\hat\tau_s$ 15.53 ms diagonal against 7.57 ms with $\Sigma^{-1}$ (§3.5.6).
  Under $W = \Sigma^{-1}$ and the true law, $2J$ at its minimum is close to $\chi^2_{d - n_{\rm par}}$ (mean
  9.22 against 9), so it serves as a misfit alarm; it is still not a likelihood.]
- (30) is a worst case for an error aimed at the thinnest axis; Monte Carlo error in $\hat\Sigma$ is
  relative axis by axis and is not amplified by $\operatorname{cond}(\Sigma)$; the numbers of §3.5.6 come
  from Gaussian draws or Gaussian inputs where stated there, and all of them have only $\tau_s$ free.
- The cohort targets are medians of per-cell medians and describe no single cell.
- (22) is continuous-time; (24) holds for a pure, continuously observed OU with $T_{\rm tot} \gg
  \tau$ and every window's mean known; (24b) for a pure OU with $D$ fixed and each window's mean
  unknown. The precisions derived from either bound unbiased estimators only, and the fast component
  and the other parameters lower them further (not computed).
- Of the resonance relations of §3.3.3, $\nu_0^2 = \nu_p^2 + (2\pi\tau)^{-2}$ is the second-order oscillator's; for
  two Lorentzian lines the spectral peak also lies below $\nu_0$, by another relation. A membrane with I_h has
  another impedance, for which the relation between its peak and the autocorrelation's ringing was not
  computed.
- §3.5.4's likelihood checks ran under the cohort law of 2026-09-28, within 7 % of
  $\hat\theta_{\rm sim}$.

**Not verified.**
- The per-cell SS sampling rates (only the cohort median was pasted); (18) is exact at physical
  lags whatever the rate, but the lag-1-sample column assumes 50 kHz.
- The re-run noise table (`noise_L3_v3.csv`) against the converted values of §3.2.3 and Appendix A.
- Whether every stored sweep passed the Allen sweep-level QC (§3.6.6).
- The physical origin of each component (§3.3.3).
- Whether the textbook-autocorrelation patch (`noise_stats.py`) has reached `main`: absent from the
  laptop clone's `Ih Fit` on 2026-10-16 **[src]**; the cluster not checked.
- The values of the last row of Appendix A.2 beyond $s_{\rm lock}$ (the pasted output was
  truncated).

---

## 6 References

**Project knowledge base (full text).**
- Allen Institute for Brain Science. *Allen Cell Types Database -- Electrophysiology technical
  white paper*, October 2017, v.5 (`CellTypes_Ephys_Overview 1.pdf`): recording, filtering and
  bath solution (synaptic blockers), QC criteria, stimulus protocols.
- Moradi Chameh, H. et al. (2021). Diversity amongst human cortical pyramidal neurons revealed via
  their sag currents and frequency preferences. *Nat Commun* 12, 2497.
  [DOI](https://doi.org/10.1038/s41467-021-22741-9).
- Rich, S. et al. (2021). Modeling reveals human-rodent differences in h-current kinetics
  influencing resonance in cortical layer 5 neurons. *Cereb Cortex* 31, 845-872.
- `TEEG_decisions_and_ideas_log.md`: D-005, D-006, D-011, D-012, D-014 to D-017 (status notes
  1-4), D-028, D-029, D-033, D-034, D-048; `claude/TEEG_acf_textbook_change_plan_2026-10-07.md`.
- Efron, B. & Narasimhan, B. (2020). The automatic construction of bootstrap confidence intervals
  (`The automatic construction of bootstrap confidence intervals.pdf`): §3, the internal (Monte Carlo)
  against the sampling error of a bootstrap estimate.

**PubMed Central, full text read.**
- Zeraati, R., Engel, T. A. & Levina, A. (2022). A flexible Bayesian framework for unbiased
  estimation of timescales. *Nat Comput Sci* 2, 193-204.
  [DOI](https://doi.org/10.1038/s43588-022-00214-3).
- Weltz, J., Laber, E. & Volfovsky, A. (2024). Hidden population estimation with indirect
  inference and auxiliary information. *Proc Mach Learn Res* 244, 3730-3746.
  [PMC12448677](https://pmc.ncbi.nlm.nih.gov/articles/PMC12448677/).
- Yaron-Jakoubovitch, A., Jacobson, G. A., Koch, C., Segev, I. & Yarom, Y. (2008). A paradoxical
  isopotentiality: a spatially uniform noise spectrum in neocortical pyramidal cells. *Front Cell
  Neurosci* 2, 3. [DOI](https://doi.org/10.3389/neuro.03.003.2008).
- Jahn, P., Berg, R. W., Hounsgaard, J. & Ditlevsen, S. (2011). Motoneuron membrane potentials
  follow a time inhomogeneous jump diffusion process. *J Comput Neurosci* 31, 563-579.
  [DOI](https://doi.org/10.1007/s10827-011-0326-z).
- Kenney, T., Gao, J. & Gu, H. (2020). Application of OU processes to modelling temporal dynamics
  of the human microbiome, and calculating optimal sampling schemes. *BMC Bioinformatics* 21, 450.
  [DOI](https://doi.org/10.1186/s12859-020-03747-4). Used for a non-numeric statement only; its
  equations were not legible in the connector's text.
- Vrieze, S. I. (2012). Model selection and psychological theory: a discussion of the differences
  between the Akaike information criterion (AIC) and the Bayesian information criterion (BIC).
  PMC3366160. [DOI](https://doi.org/10.1037/a0027127).
- Kaczmarska, J. M., Isham, V. S. & Northrop, P. (2015). Local generalised method of moments: an
  application to point process-based rainfall models. *Environmetrics* 26, 312-325. PMC4975607.
  [DOI](https://doi.org/10.1002/env.2338). The optimal weighting matrix, the two-step procedure, and
  Jesus & Chandler's finding that the diagonal was close to optimal in their setting.

**PubMed, abstract only (flagged; no number used).**
- Jacobson, G. A. et al. (2005). Subthreshold voltage noise of rat neocortical pyramidal neurones.
  *J Physiol* 564, 145-160. [DOI](https://doi.org/10.1113/jphysiol.2004.080903).
- Anderson, C. R. & Stevens, C. F. (1973). Voltage clamp analysis of acetylcholine produced
  end-plate current fluctuations at frog neuromuscular junction. *J Physiol* 235, 655-691.
  [DOI](https://doi.org/10.1113/jphysiol.1973.sp010410).
- Tuckwell, H. C., Wan, F. Y. M. & Rospars, J.-P. (2002). A spatial stochastic neuronal model with
  Ornstein-Uhlenbeck input current. *Biol Cybern* 86, 137-145.
  [DOI](https://doi.org/10.1007/s004220100283).

**Companion documents and checks (not in project knowledge; delivered for commit at
`Passive Features/HPC script/Ih Fit/dev/2026-10-16_covariance_weighting/`).**
`TEEG_bias_of_a_statistic_2026-10-11.md` (folder `bias_doc_2026-10-11/`);
`TEEG_Stage7_covariance_weighting_handoff_2026-10-16.md`; the checks listed in Appendix B.

**Data.** The recordings are the Allen Cell Types Database (Allen Institute for Brain Science),
whose electrophysiology publication is the white paper above. *Licence and attribution:* the Allen
Institute Terms of Use allow the content to be used, copied, distributed and adapted for research or
other noncommercial purposes, forbid commercial redistribution without written permission, and
require citation according to the Allen Institute Citation Policy (read through a summarising fetch
on 2026-09-28; the citation string specific to the Cell Types Database was not found on the pages
fetched).

**Code and software read [src].** `Ih Fit/noise_calibration.py`, `noise_stats.py` (patch on
`b355d8e`), `ls_baseline_qc.py`; `passive_fitting_hpc_fixed._estimate_noise`;
`synthetic_ground_truth.add_recording_noise`, `ProtocolConfig`; the Phase-0 archive writer
(`save_alleninstitute_data (6).py`: `_extract_windows_around_pulses`, Format A of
`ss_pulses.npz`); AllenSDK 2.16.2 (`get_ephys_sweeps`); statsmodels 0.15.0 (`acf`); abcTau
repository (commit `72343a8`). 2026-10-16: the file list of `Ih Fit` on the laptop clone's `main` and
the function definitions of `noise_calibration.py`.

**From memory, not read.** Gouriéroux, C., Monfort, A. & Renault, E. (1993). Indirect inference.
*J Appl Econom* 8, S85-S118; Smith, A. A. (1993). Estimating nonlinear time-series models using
simulated vector autoregressions. *J Appl Econom* 8, S63-S84. Hansen, L. P. (1982). Large sample
properties of generalized method of moments estimators. *Econometrica* 50, 1029-1054; Hansen, L. P.,
Heaton, J. & Yaron, A. (1996). Finite-sample properties of some alternative GMM estimators. *J Bus
Econ Stat* 14, 262-280. Jesus & Chandler, cited by Kaczmarska et al. for the diagonal weighting: not
identified further, not read.

**Textbook, from memory.** Exact discretisation of the OU process as an AR(1); the variance of the
mean of a stationary sequence (7a) and its OU form (8); Wiener-Khinchin and the Lorentzian
spectrum (20); the spectral representation of stationary processes; the Cauchy characteristic
function (21); the autocorrelation of a noise-driven damped oscillator; Campbell's theorem (19); Johnson-Nyquist noise of an RC circuit; the autocorrelation
of a two-state Markov channel; the boxcar's sinc² transfer behind (22); the OU Fisher information
behind (24); the Gaussian (Slepian-Bangs) information and the integrated-likelihood form of REML
behind (24b); the non-regularity of a variance component tested at zero; the single-compartment RC
argument for $\tau_m$; the Rayleigh quotient and the perturbation bound (30) for the inverse of a
matrix; the law of a whitened sample covariance; the asymptotic $\chi^2$ law of the minimised,
optimally weighted GMM cost.

**Searches run.** 2026-09-28: PubMed, whole-cell drift and baseline stability, subthreshold
voltage noise in neocortical pyramidal neurons (Jacobson 2005 and Yaron-Jakoubovitch 2008 taken
from "voltage noise pyramidal neurones", 81 records), timescale estimation (Zeraati 2022), indirect
inference (Weltz 2024 among 7 records), Anderson & Stevens by citation, and several queries with no
usable return; bioRxiv, neuroscience, last 30 days, nothing relevant. 2026-10-01: PubMed,
within-sweep drift in slice recordings (0 records) and slow afterdepolarisation after
hyperpolarisation (5, none used); bioRxiv, neuroscience, last 30 days, nothing relevant.
2026-10-08: PubMed, BIC and model selection (Vrieze 2012), OU parameter estimation and its
precision (12 records: Kenney 2020 read, Tuckwell 2002 abstract only, the rest off topic), Jahn et
al. by author (1); bioRxiv, neuroscience, last 14 days (15 preprints), nothing relevant; the
connector has no keyword search. Data repositories: the Allen Brain Map API (EphysSweep records; 2026-09-28 and 2026-10-01,
metadata only, no value used); none on 2026-10-08. 2026-10-09: project knowledge, Moradi Chameh et al.'s
definition of the resonant frequency (the impedance peak of a ZAP response). 2026-10-11: see
`TEEG_bias_of_a_statistic_2026-10-11.md`, which lists its own. 2026-10-13/14: PubMed full text of
Kaczmarska et al. 2015 retrieved and read (the other full texts retrieved in that discussion are not
used here). 2026-10-16: a PubMed search for iterated GMM returned nothing usable (recorded in the
covariance-weighting handoff); PubMed metadata of PMID 27563266, which gives the authors as Kaczmarska,
Isham & Northrop.

---

## Appendix A -- Per-cell statistics of L3_exc [cluster; autocorrelations converted by (18)]

**A.1 First noise table (2026-09-23).** LS values: medians over each cell's single sweeps; SS
values: medians over its pulses. $\hat\rho(1)$ is lag 1 sample; the SS lag-1 column assumes 50 kHz.

| Specimen | LS sweeps | $\hat\sigma_{\rm LS}$ (mV) | $\hat\rho_{\rm LS}(1)$ | $\hat\rho_{\rm LS}(1\ \text{ms})$ | SS pulses | $\hat\sigma_{\rm SS}$ (mV) | $\hat\rho_{\rm SS}(1)$ | $\hat\rho_{\rm SS}(1\ \text{ms})$ |
|---|---|---|---|---|---|---|---|---|
| 508282493 | 6 | 0.0541 | 0.541 | 0.431 | 40 | 0.0417 | 0.242 | 0.040 |
| 526785799 | 5 | 0.0698 | 0.756 | 0.662 | 47 | 0.0403 | 0.284 | 0.041 |
| 529878215 | 6 | 0.0698 | 0.765 | 0.707 | 39 | 0.0386 | 0.252 | 0.066 |
| 531520401 | 5 | 0.0536 | 0.635 | 0.432 | 40 | 0.0407 | 0.287 | 0.019 |
| 531523484 | 6 | 0.0593 | 0.633 | 0.509 | 37 | 0.0411 | 0.265 | 0.031 |
| 531526539 | 6 | 0.0596 | 0.633 | 0.497 | 46 | 0.0421 | 0.328 | 0.024 |
| 537297470 | 6 | 0.0549 | 0.647 | 0.556 | 66 | 0.0377 | 0.209 | 0.021 |
| 537313600 | 6 | 0.0490 | 0.545 | 0.420 | 47 | 0.0383 | 0.255 | 0.013 |
| 539653924 | 6 | 0.0648 | 0.711 | 0.632 | 47 | 0.0389 | 0.214 | 0.015 |
| 541545716 | 5 | 0.0945 | 0.853 | 0.790 | 56 | 0.0425 | 0.348 | 0.004 |
| 548408551 | 6 | 0.0559 | 0.635 | 0.535 | 52 | 0.0391 | 0.238 | 0.044 |
| 548425467 | 6 | 0.0549 | 0.675 | 0.564 | 69 | 0.0387 | 0.279 | 0.032 |
| 548473051 | 6 | 0.0585 | 0.662 | 0.564 | 56 | 0.0383 | 0.230 | 0.027 |
| 569844159 | 6 | 0.0522 | 0.618 | 0.464 | 348 | 0.0450 | 0.422 | 0.048 |
| 571430815 | 5 | 0.2282 | 0.979 | 0.962 | 199 | 0.0493 | 0.535 | 0.187 |
| 571483807 | 6 | 0.0691 | 0.759 | 0.603 | 303 | 0.0456 | 0.405 | 0.106 |
| 571624532 | 6 | 0.0736 | 0.782 | 0.721 | 317 | 0.0426 | 0.268 | 0.034 |
| 571691416 | 6 | 0.0762 | 0.801 | 0.735 | 58 | 0.0403 | 0.339 | 0.090 |

**A.2 Pre-window structure (2026-09-28).** LS columns: medians over each cell's single sweeps;
$\hat\sigma_{\rm SS}$: median over its pulses; $s_{\rm lock}$: one value per cell. SDs in mV. The
100 ms column is shown for completeness: under $\hat\theta_{\rm sim}$ the estimator's 5-95 % range
there is $[-0.197, 0.063]$ per window (§3.4.2).

| Specimen | pre-window (ms) | $\hat\sigma$ first 50 ms | $\hat\sigma$ last 132.5 ms | $\hat\sigma_{\rm SS}$ | $s_{\rm lock}$ | $s_{\rm trend}$ | last $\hat\rho(1\ \text{ms})$ | last $\hat\rho(10\ \text{ms})$ | last $\hat\rho(100\ \text{ms})$ |
|---|---|---|---|---|---|---|---|---|---|
| 508282493 | 265 | 0.045 | 0.049 | 0.042 | $-0.068$ | 0.149 | 0.321 | 0.107 | $-0.020$ |
| 526785799 | 265 | 0.043 | 0.046 | 0.040 | $-0.082$ | **0.582** | 0.229 | 0.196 | $-0.018$ |
| 529878215 | 265 | 0.060 | 0.066 | 0.039 | **0.186** | 0.089 | 0.584 | 0.343 | $-0.156$ |
| 531520401 | 265 | 0.047 | 0.055 | 0.041 | $-0.027$ | 0.128 | 0.293 | 0.181 | $-0.004$ |
| 531523484 | 265 | 0.047 | 0.050 | 0.041 | 0.031 | 0.139 | 0.341 | 0.146 | 0.005 |
| 531526539 | 265 | 0.047 | 0.056 | 0.042 | $-0.038$ | 0.097 | 0.417 | 0.247 | $-0.023$ |
| 537297470 | 265 | 0.041 | 0.060 | 0.038 | $-0.068$ | 0.040 | 0.642 | 0.442 | $-0.032$ |
| 537313600 | 265 | 0.039 | 0.046 | 0.038 | 0.104 | 0.112 | 0.339 | 0.186 | $-0.056$ |
| 539653924 | 265 | 0.047 | 0.060 | 0.039 | 0.005 | 0.146 | 0.559 | 0.407 | $-0.046$ |
| 541545716 | 265 | 0.048 | 0.077 | 0.042 | $-0.039$ | 0.137 | 0.691 | 0.334 | $-0.069$ |
| 548408551 | 265 | 0.041 | 0.050 | 0.039 | $-0.060$ | 0.192 | 0.484 | 0.256 | 0.010 |
| 548425467 | 265 | 0.046 | 0.049 | 0.039 | 0.118 | 0.168 | 0.401 | 0.205 | 0.030 |
| 548473051 | 265 | 0.043 | 0.047 | 0.038 | 0.080 | 0.214 | 0.353 | 0.185 | $-0.019$ |
| 569844159 | 265 | 0.045 | 0.051 | 0.045 | 0.045 | 0.071 | 0.419 | 0.186 | $-0.017$ |
| 571430815 | 265 | 0.093 | 0.127 | 0.049 | $-0.097$ | **0.637** | 0.870 | 0.436 | $-0.177$ |
| 571483807 | 265 | 0.056 | 0.061 | 0.046 | 0.047 | 0.179 | 0.469 | 0.133 | 0.030 |
| 571624532 | 265 | 0.046 | 0.072 | 0.043 | 0.105 | 0.058 | 0.671 | 0.352 | $-0.060$ |
| 571691416 | 265 | 0.057 | 0.076 | 0.040 | 0.148 | not in the pasted output | -- | -- | -- |

---

## Appendix B -- The scripts behind the numbers

Every number tagged [run] comes from the scripts below (fixed seeds); they are diagnostics, not the
calibration, and not part of the repository [corrected 2026-10-16: the checks of 2026-10-11 to 2026-10-14
are delivered for commit at `Passive Features/HPC script/Ih Fit/dev/2026-10-16_covariance_weighting/`,
as standalone checks, not pipeline code; the earlier bundles are still not in the repository].

| Bundle | Scripts | Used in |
|---|---|---|
| 2026-10-01 read-out of `ls_baseline_qc.py` (`lsqc_readout/readout.py`, `sign_check.py`) | sign agreement, wander nulls, per-cell tail statistics | §3.2.5 |
| `TEEG_Stage7_noise_scripts_2026-10-08.zip`, folder `mo/` | `common.py` (laws, exact expectations, simulation, textbook ACF), `exact_lik.py` (celerite2 likelihood, BIC) with `smoke_exact_lik.py`, checks A-D (`checkA_window_identifiability.py`, `checkB_order_selection.py`, `checkB2_reml.py`, `checkC_ss_enlarged.py`, `checkD_window_vs_tau.py`), `checkC2_t0.py`, `verify_msg29.py`, `fig_noise_shape.py` | §3.3.3, §3.4, §3.5.4 |
| `check_resonant_acf.py` (sent 2026-10-09); `check_resonance_peak.py` (`TEEG_noise_theory_checks_2026-10-09.zip`) | two Lorentzian lines against a noise-driven damped oscillator; the impedance peak against the ringing frequency, and the 265 ms window's median sample autocorrelation at the first zeros | §3.3.3 |
| `fisher_doc_checks.py` with its log (sent 2026-10-10 with `TEEG_fisher_information_2026-10-10.md`) | (24b) against dense explicit-contrast computations of the REML information; the contrast information of Check D's designs; the enlarged-window designs | §3.4.3 |
| same zip, folder `v3/` | `targets.py`, `model.py`, `v3_fit.py` (cohort fits, profile), `validate_sim.py`, `stable.py`, `derived.py`, `derived_fix2.py`, `derived_mc.py`, `final.py`, `fix_review.py`, `q23.py`, `convert_appendix.py`, with their logs | every number under $\hat\theta_{\rm sim}$; Appendix A |
| dev folder, `bias_doc_2026-10-11/scripts/` (`check01`-`check04`, `fig01`-`fig05`, `bias_common.py`) | the bias of $\hat\rho(1\ \text{ms})$ on the 265 ms window, its split between demeaning and the ratio, the exact integral against Monte Carlo | §3.4.2 corrections |
| dev folder, `checks/weights_1013/` (`check_weights.py`, `check_cov_source.py`, `check_sigma_theta.py`, `check_conditioning.py`, `check_cov_structure.py`, `check_missing_noise.py`, `boot_model.py`; `results/weights.npz`) | the three weightings; where $\hat\Sigma$ can come from; $\Sigma$ against $\theta$; (30) under random perturbations; Monte Carlo error against the condition number, linearised SDs, biased-statistic shifts, units and correlations; the missing-noise table | §3.5.6 |
| dev folder, `checks/boot_1013/`, `checks/boot_err_1014/` | bootstrap schemes within one window and over a cell's windows; sampling against internal error | §3.5.6 |
| dev folder, `checks/cond_fig_1014/fig_condition_number.py` | equality in (30); the figure of the two readings of the condition number | §3.5.6 |

To reproduce: `python <script>` from inside its folder (the `v3/` scripts import `../mo/common.py`);
`derived*.py`, `final.py` and `fix_review.py` read `fit.json`, written by `v3_fit.py` (about 9
minutes). `derived.py` gives the visible shares, the closed-form check, the demeaning bias and the
seconds-scale sizes; `derived_fix2.py` the baseline error, the SS slow shares and the tilt-matching
criteria (exact points, via `stable.py`); `derived_mc.py` the per-cell precision, the trend-share and
locked-share nulls, the enlarged-window precision, the 3 s semivariogram bands and the Monte Carlo
spread; `final.py` the step-mean error and the slow power below $\nu_{1/2}$; `fix_review.py` the
semivariogram table, the matched 30 s bands, the single AR(1), the lowest cell's percentile, the
Bessel realisations and the check of (9); `q23.py` the numbers of (22) and the $5\tau$ OU of §3.4.2;
`mo/checkD_window_vs_tau.py` the table of §3.4.3; `fisher_doc_checks.py` runs on its own (about
35 s) and gives (24b) and its comparison with Check D. The dev-folder checks: see that folder's
`README.md`; each script states its predictions in its docstring and its log sits in `logs/`;
`check_weights.py` (about 2 min) writes `results/weights.npz`, which the other `weights_1013` scripts and
`cond_fig_1014` read.
