Weighting check behind the answer of 2026-10-13 12:46 (TEEG, Stage 7, cost (13)/TD (14)).

scripts/check_weights.py  predictions P1-P3 fixed in the docstring; one-parameter fit (tau_s free) of the ten
                          LS statistics, medians over K = 6 windows, under three weightings: hand-set omega_m,
                          diagonal 1/SE^2, full Sigma^-1. Run: python3 check_weights.py (about 2 min).
scripts/boot_model.py     model and statistic helpers (same as the bootstrap check).
logs/check_weights.log    the reported run (all predictions passed).
results/weights.npz       SEs, correlation matrix, binding function, fitted tau_s per cell and weighting.
Software: Python 3, NumPy 2.5.3, SciPy 1.18.1.

Added 2026-10-14:
scripts/check_cov_source.py  where Sigma can come from: one cell's six windows (plain covariance; window-level
                             bootstrap) vs simulation at a first-step tau_hat (two-step). Predictions P4-P6 in the
                             docstring. Run after check_weights.py: python3 check_cov_source.py (about 3 min).
logs/check_cov_source.log    the reported run (all predictions passed).

Added 2026-10-14 (answer of 09:13: construction of Sigma, condition number, correlations):
scripts/check_sigma_theta.py     P7-P9 behind the answer of 09:04 (Sigma vs theta; independent 1 % errors on the entries
                                 of Sigma vs Sigma^-1; the 2 x 2 identity). Not in the earlier zip. Note added on P8
                                 (docstring). Run: python3 check_sigma_theta.py (about 1 min).
scripts/check_conditioning.py    eigenvalues and condition numbers of Sigma and of the correlation matrix; the exact
                                 perturbation bound kappa*eps/(1 - kappa*eps) against 2000 random symmetric perturbations.
                                 Runs 1-2 tested a first-order version and failed outside its domain (docstring).
                                 Run: python3 check_conditioning.py (a few seconds).
scripts/check_cov_structure.py   P10-P18: Monte Carlo error of a simulated Sigma_hat vs kappa, incl. P17 (independent of kappa) (A); linearised fit and the
                                 diagonal weighting's nominal SD (B); sensitivity to a one-SE bias in one statistic (C);
                                 Gaussian-input diagnosis (D); units, correlations, P18 (fit independent of units) (E). P12 for Sigma^-1 and P15 failed:
                                 the linearised model is optimistic for Sigma^-1 (docstring). Run: python3
                                 check_cov_structure.py (about 6 s, peak memory about 160 MB).
logs/                            check_*.log are the reported runs; check_*_runN.log are earlier runs kept for the record.
All scripts run after check_weights.py, which writes results/weights.npz (shipped, so they also run without it).

Added 2026-10-14 17:30 (consolidated explanation of the condition number):
scripts/check_missing_noise.py   a noise source the simulator lacks: Sigma_true = Sigma_used + s2*diag(Sigma_used) for
                                 s2 = 0, 0.1 %, 1 %, 10 %; exact weight ratios on the thinnest/widest axis and E[J];
                                 Gaussian-input fits with W = Sigma_used^-1, the oracle Sigma_true^-1 and the diagonal.
                                 Predictions PM1-PM3 in the docstring (all held). Run: python3 check_missing_noise.py
                                 (about 30 s).
logs/check_missing_noise.log     the reported run.
