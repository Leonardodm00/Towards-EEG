Check behind the answer of 2026-10-14 10:26 (TEEG, bootstrap: sampling error vs internal error, E&N sec. 3).

scripts/check_boot_errors.py  window-level bootstrap of the median of K = 6 LS windows of rho_hat(1 ms) at theta_sim
                              (the cells of E2, boot_1013/check_boot_types.py, same seed). The ideal bootstrap
                              (B = infinity) is computed exactly by enumerating the 462 count vectors of a resample of
                              six; the error of SE_B is split into sampling (SE_inf - SD_true) and internal
                              (SE_B - SE_inf); Efron & Narasimhan's jackknife estimate of internal error (J = 10
                              groups of the B replicates) is compared with the actual one. Predictions P0-P4 are
                              fixed in the docstring. Run: python3 check_boot_errors.py (about 20 s, 270 MB).
scripts/boot_model.py         model and statistic helpers (copied unchanged from boot_1013).
logs/check_boot_errors.log    the reported run (all predictions held).
Software: Python 3, NumPy 2.5.3, SciPy 1.18.1.

Added 2026-10-14 10:37 (answer to: does the sampling error depend only on the information in the data?):
scripts/check_info_vs_scheme.py  A: the same window-level bootstrap with K = 6 and K = 24 windows per cell (more
                                 information, same plug-in rule); B: the iid world (one window's samples resampled
                                 with replacement) against the true SD of T at n = 13 250 and n = 53 000 (same data
                                 kind, wrong plug-in rule). Predictions PA, PB in the docstring. Run: python3
                                 check_info_vs_scheme.py (about 30 s, 260 MB).
logs/check_info_vs_scheme.log    the reported run (both predictions held).
