dev/2026-10-30_offsets_and_fit -- the checks behind two explanations of 2026-10-30
=====================================================================================

Standalone checks (not pipeline code), written 2026-10-30 by the noise-fit chat when
the user asked (1) to explain the proposal to remove each SS window's tilt offset
before the per-sample median window (D-103.1), and (2) to explain the window-start
decay fit of D-110 (confirmed that day, D-114). They need the Ih Fit folder two levels
up (or TEEG_IH_FIT) and the simulator of ../2026-10-29_start_noise/check_start_noise.py.

Simulated cells: check_start_noise.py's -- 2 SS sweeps of the archive's 37-pulse train
(+/-200 pA x 0.5 ms, 10025 samples onset to onset at 50 kHz), each a continuous sweep:
the RC response of the whole train (113 MOhm, tau_m 28.143 ms) plus Block-2 noise at
theta_cf = (0.0380 mV, 0.175, 0.0508 mV, 36.6 ms); each pulse dropped with probability
0.27; the tilt b per sweep as stated in each part.

Files
-----
check_offsets_mechanism.py       (seed 20261051; about 4 min) parts M, E, F below;
offsets_mechanism_2026-10-30.txt its output: PM1, PM2, PE1, PF1, PF2 all PASS
check_fit_profile.py             (seed 20261052; about 8 min; --figure-only: seconds)
fit_profile_2026-10-30.txt       its output: PU1, PU2, PT1 all PASS
fig_fit_profile.png              the figure of the fit's explanation (made by it)
inspect_failed_group.py          (about 1 min) the one failed fit of check_fit_profile.py,
failed_group_2026-10-30.txt      refitted with the offsets removed; its output

What came out (run here, 2026-10-30)
------------------------------------
(1) The tilt and the median window (check_offsets_mechanism.py)
  M  which windows supply the per-sample median of a group, on [5, 200) ms (theta_cf
     noise, no response, the same b in both sweeps, 400 groups per b): the central
     third of a group's windows in onset order supplies 0.330 of the median at b = 0
     (it holds 0.329 of the windows) and 0.914 at b = 0.1 mV/s (it holds 0.334); by
     fifths of the train at 0.1 mV/s: 0.002, 0.134, 0.721, 0.140, 0.003.
  E  the subtraction (white law sigma_f 0.05 mV, rho_f 1e-3, sigma_s 0; windows from
     80 ms; 60 cells per b): mean sigma_hat^2 / sigma_f^2 = 0.9616 / 0.9881 / 1.2625 at
     b = 0 / 0.02 / 0.1 mV/s with the plain median; 0.9616 / 0.9604 / 0.9619 with the
     true offsets removed first; 0.9616 / 0.9604 / 0.9619 with the ESTIMATED offsets
     removed first (b_hat_s by (TL1) on the windows cut at 80 ms).
  F  the start (D-110, D-111) on 300 tilted 2-sweep cells (sigma_b 0.1 mV/s, RC
     response), paired: SD of t_sp - t0 50.6 ms as implemented, 21.9 ms with the
     estimated offsets removed first; fallback 16.3 % -> 3.7 % (41 cells rescued, 3
     lost). Untilted reference (300 cells): 18.6 ms either way; fallback 4.3 % -> 5.0 %
     (0 rescued, 2 lost).

(2) The exact decay fit (check_fit_profile.py)
  Worked case (the figure): the first sweep's depolarising group of the first
  untilted cell, K = 13 windows: the grid minimum at tau = 27.542 ms (point 144 of
  301), the bounded Brent step in [26.915, 28.184] ms -> 27.257604 ms (8 profile
  evaluations), the root of dRSS/dtau at 27.257604 ms; A_hat 0.3667 mV, B_hat
  -59.9871 mV, RSS 3.795 mV^2 over 9750 samples; sigma_SS_hat 0.0402 mV; t_sp 123.0 ms
  (noiseless t0 129.5 ms).
  On 600 groups per tilt (150 2-sweep cells, sigma_b 0 and 0.1 mV/s):
    every untilted group has exactly one interior minimum of RSS(tau) on a 3001-point
    grid over [1, 1000] ms; with the tilt 599 do and 1 has none (RSS(tau) falls to the
    bound: a failed fit); no point of the 3001-point grid lies below the exact fit's
    RSS in any group; the three-start curve_fit of f293d4a (copied in the script)
    reached the same optimum (to 1e-6 of the RSS) in every group where its tau stayed
    in [1, 1000] ms and never beat the exact fit; in the one tilted group without an
    interior minimum it returned tau = 9.2e6 ms with A = 1.5e4 mV (a straight line in
    disguise), where the exact fit reports a failed fit.
  That group (inspect_failed_group.py; tilted cell 47, sweep 101, dep, K = 11 at
  positions 0, 4, 8, 10, 12, 18, 24, 26, 28, 34, 36; b_hat 0.115 mV/s): RSS(tau)
  falls from 123.5 mV^2 at 1 ms to 50.2 mV^2 at 1000 ms; with the estimated tilt
  offsets removed first it fits normally: tau_hat 29.25 ms, A_hat 0.3766 mV, RSS
  5.3 mV^2. The one failed fit of the 1200 is the tilt-offset effect of (1).

Re-run (from this folder):
    python check_offsets_mechanism.py    (about 4 min)
    python check_fit_profile.py          (about 8 min)
    python inspect_failed_group.py       (about 1 min)
Pure ASCII, LF.
