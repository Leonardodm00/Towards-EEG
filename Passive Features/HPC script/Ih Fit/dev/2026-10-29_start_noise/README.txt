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

Re-run (from this folder):
    python check_start_noise.py      (about 5 min)
    python check_start_options.py    (about 6 min)
Pure ASCII, LF.
