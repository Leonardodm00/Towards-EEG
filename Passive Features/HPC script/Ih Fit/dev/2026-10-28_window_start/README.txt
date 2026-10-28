TEEG -- window-start (SS transient) checks, 2026-10-23 to 2026-10-28 -- standalone, not pipeline code

In the repository since 2026-10-28 (D-097: every document also in the repository), as delivered in
TEEG_window_start_checks_2026-10-28.zip except the default Ih Fit path of check_findings_2026-10-23.py
(two levels up instead of the sandbox clone). The scripts' docstrings still say "not in the repository":
they were written in the sandbox. Nothing here is imported by the Ih Fit code or runs in a job.

Behind: the explanation of the two findings of Blocks 3-4 (2026-10-23), the explanation of the window-start rule
(2026-10-26), of sigma_SS_hat (2026-10-28), and decision D-095 (2026-10-28) that replaces the rule.
Every script uses the repository's own functions (noise_law, noise_windows, noise_cell_stats, noise_fit) through
check_findings_2026-10-23.py, which finds the Ih Fit folder two levels up (this folder sits in Ih Fit/dev/), or IH_FIT_DIR if set:
    export IH_FIT_DIR="/path/to/Towards-EEG/Passive Features/HPC script/Ih Fit"
then run any script from this folder with python (numpy, scipy, statsmodels, matplotlib). theta_cf = (0.0380 mV, 0.175,
0.0508 mV, 36.6 ms) throughout; the archive's SS layout (37-pulse trains, 10025-sample period, stored (-10, +200) ms windows).

| script | output | what it shows | time |
|---|---|---|---|
| check_findings_2026-10-23.py (A, B, C, D) | results_2026-10-23.txt | A: the transient test as written is noise-limited (bin-mean SD 0.30/0.24/0.15/0.11 sigma_SS at K 40/63/199/348; fallback 56-90 % of noise-only cells); B: sigma_b_hat spread with 2-10 sweeps; C: what sigma_b moves in theta_hat; D: a per-bin simulated null (z 3.5) | ~17 min |
| fig_transient_floor.py | fig_transient_floor.png, .log | the test statistic's noise against its threshold at K 63 and K 348 | ~1.5 min |
| check_per_window_fit.py | per_window_fit_2026-10-26.txt | plain mean-window subtraction vs per-window gain/shift fits under amplitude jitter (plain: bias <= 0.033 SD at cv 0.2; gain fits -1.3 to -2.3 SD) | ~4 min |
| check_jitter_leak.py | jitter_leak_2026-10-26.txt | the leak law: E[sigma_hat^2] rises by delta_k = cv^2 [...] exactly (L1a); medians move 0.57-0.64 of first order; option (b)'s start reads the mean response, not cv (L2) | ~4 min |
| check_jitter_leak_run1.py | jitter_leak_2026-10-26_run1_L1_failed.txt | run 1, the median-law claim that FAILED (kept as a record) | ~3 min |
| check_sigma_ss.py | sigma_ss_2026-10-28.txt | sigma_SS_hat's make-up: one 10 ms segment's expected variance 87 % fast, 13 % slow (8.5 % of sigma_s^2 visible); median 0.0402 mV +/- 1.1 % across cells at K 63 | ~40 s |
| check_d095_rc_start.py | d095_rc_start_2026-10-28.txt (+ run1, run2 failed) | D-095's start for a noiseless RC response at kappa_tr 0.1: 130 ms for the fixture (28.1 ms, 113 MOhm); past 190 ms from tau_m ~46 ms; a pre-pulse-level baseline delays the crossing (the previous pulse's tail) | seconds |
| draw_decision_map.py | noise_fit_decision_map.png | schematic of where the 2026-10-23 choices sit in the chain | seconds |

Runs that failed are kept with their outputs, as the explanation reported them (Check by computation: a failed
check corrects the claim).
