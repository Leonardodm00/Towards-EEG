dev/2026-10-29_start_noise -- how the SS window start of D-103 / D-104 behaves on noisy cells
==========================================================================================

Standalone checks (not pipeline code), written 2026-10-29 by the noise-fit chat right
after Block 3 implemented the start (noise_windows.decay_start, commit f293d4a on
branch sci/noise-fit-d103). They need the Ih Fit folder two levels up (or
TEEG_IH_FIT pointing to it) with that Block 3: noise_windows, noise_law and
smoke_noise_windows (for the fixture's RC membrane).

Simulated cells: S SS sweeps (S = 2, 6, 10) of the archive's train (37 pulses of
+/-200 pA x 0.5 ms, 10025 samples onset to onset at 50 kHz), each a continuous
sweep: the RC response of the whole train (113 MOhm, tau_m 28.143 ms) plus Block-2
noise at theta_cf = (0.0380 mV, 0.175, 0.0508 mV, 36.6 ms); each pulse dropped with
probability 0.27; tilt sigma_b 0 or 0.1 mV/s. t0 = the noiseless start at the cell's
own sigma_SS_hat (129.5 ms at 0.0402 mV).

Files
-----
check_start_noise.py         the start as implemented (V0): per-group point values
                             t_sp and the cell's t_star, t_SS; predictions P1-P3
start_noise_2026-10-29.txt   its output (150 cells per row; P1, P2, P3 PASS; 262 s)
check_start_options.py       the same cells, four variants beside V0 (seed 20261042,
                             100 cells per row): V1 the median over groups; V2 the
                             decay with a linear term, maximum; V3 V2's fits,
                             median; V4 one fit per polarity on the cell's pooled
                             median window (departs from D-095's per-sweep grouping)
start_options_2026-10-29.txt its output (no pass/fail: a comparison)

What came out (run here, 2026-10-29)
------------------------------------
Each group's point value is unbiased (median of t_sp - t0 within 1 ms of 0) but
scatters: SD about 18 ms without a tilt, 40-55 ms with sigma_b = 0.1 mV/s (the
tilt's ramp inside the window bends the fit; a few groups fit tau in the 1e5 ms
range). The maximum over groups (D-103.3) therefore lands late, and past 150 ms the
80 ms fallback of D-104 takes over:

  fallback share (V0, as implemented):  sigma_b 0: 47 % (2 sweeps), 90 % (6), 96 % (10)
                                        sigma_b 0.1: 69 %, 99 %, 100 %
  median of t_star - t0 (V0):           sigma_b 0: +19.7, +33.0, +38.4 ms
                                        sigma_b 0.1: +31.0, +72.1, +88.5 ms

The variants (start_options, 100 cells per row; median of t_star - t0 in ms /
fallback share):

  sigma_b  sweeps   V0 max        V1 median    V2 line+max    V3 line+med   V4 pooled+max
  0        2        +19.8 / 49 %  -0.3 /  3 %  +39.2 /  67 %  -2.0 / 16 %   +7.3 / 12 %
  0        6        +33.6 / 84 %  +0.0 /  0 %  +68.0 /  97 %  -1.3 /  3 %   +4.5 /  0 %
  0        10       +35.8 / 97 %  -0.1 /  0 %  +83.7 / 100 %  +1.4 /  0 %   +3.1 /  0 %
  0.1      2        +30.6 / 66 %  +2.4 / 13 %  +61.7 /  84 %  +4.8 / 26 %  +11.4 / 30 %
  0.1      6        +63.7 / 97 %  +1.5 /  3 % +132.2 / 100 %  +3.6 / 15 %   +5.6 / 11 %
  0.1      10       +82.8 / 99 %  +0.3 /  3 % +193.0 / 100 %  -1.3 /  5 %   +6.4 /  1 %

In short: the median over groups (V1) stays within 2.4 ms of t0 and falls back in
0-13 % of cells; the pooled fit (V4) within 3-11 ms, 0-30 %; the linear term (V2,
V3) adds a parameter's noise and does not help. Which rule to use is the user's
call (D-103.3 chose the maximum); nothing in the pipeline changed.

After D-110 and D-111 (2026-10-29, later, 5: the exact fit, the median over groups)
--------------------------------------------------------------------------------
check_start_noise.py and check_start_options.py now compute the maximum rule and the
variants themselves from Block 3's group starts (exact fits since D-110); their
output files are the runs of f293d4a (three-start fits).

check_start_decided.py        the rule as decided and implemented (Block 3 at 7672ffc):
start_decided_2026-10-29.txt  150 cells per row, seed 20261043. P1, P2 PASS; P3 FAIL:
  sigma_b  sweeps  t_star - t0 median  fallback
  0        2       +0.6 ms              3 %
  0        6       +0.1 ms              0 %
  0        10      +0.4 ms              0 %
  0.1      2       +3.5 ms             16 %   (P3 predicted at most 15 %)
  0.1      6       +0.9 ms              1 %
  0.1      10      -0.7 ms              0 %
  No fit at a bound except 1 of 3000 groups (10 sweeps, tilted).

check_start_detrended.py      the tilt's ramp INSIDE the window removed before the fit
start_detrended_2026-10-29.txt (seed 20261044): no help -- 2 sweeps at 0.1 mV/s fall back
                              in 15 % as decided and 17 % with the ramp removed. The
                              assistant's first explanation (the ramp bends the fit) was wrong.

check_median_offsets.py       the tilt's spread of the windows' BASELINES (b (t_k - t_bar),
median_offsets_2026-10-29.txt up to +/-0.36 mV across a train at 0.1 mV/s) degrades the
                              per-sample median over a group's windows (seed 20261045):
  A (start): SD of t_sp - t0 at sigma_b 0.1: 35.9 ms (2 sweeps) and 41.6 ms (6) as
    decided, 22.2 and 21.8 ms with each window's tilt offset removed first (19.2 and
    18.1 ms without a tilt); fallback 8 -> 4 % and 2 -> 0 %.
  B (subtraction, white law): mean sigma_hat^2 / sigma_f^2 after the median
    subtraction 0.961 (b 0), 0.990 (0.02 mV/s), 1.267 (0.1 mV/s); with the offsets
    removed 0.961, 0.962, 0.961.
  Predictions: PA1 FAIL (offsets removed 22.2 ms against "below 22"), PB1 FAIL (1.27
  against "above 1.5": the inflation is real but smaller than predicted), PB2 PASS.
  The fit stays consistent either way (its simulator carries the same tilts and
  subtracts alike); what the spread costs is precision of the SS statistics and the
  start's scatter. Remedy (a proposal for the user): remove each window's tilt offset
  before the per-sample median, in the start and the subtraction, data and
  simulator alike.

Re-run (from this folder):
    python check_start_noise.py      (about 5 min)
    python check_start_options.py    (about 6 min)
    python check_start_decided.py    (about 7 min)
    python check_start_detrended.py  (about 9 min)
    python check_median_offsets.py   (about 4 min)
Pure ASCII, LF.
