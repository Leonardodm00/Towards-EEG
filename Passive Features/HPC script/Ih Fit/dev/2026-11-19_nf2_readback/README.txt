dev/2026-11-19_nf2_readback -- the nf2 run (three-component law, D-152 to D-155) read back, and the slow-ridge check
====================================================================================================================

Standalone checks, not pipeline code (2026-11-19, the noise-fit chat). Nothing here is imported by the pipeline.

Files
-----
readback2_2026-11-19.txt        the user's paste (2026-11-19, 12:12) of the second read-back one-liner of the Block 6
                                handoff, section 4, over the 18 cell JSONs of $OUT/cells/ (run tag nf2): per cell the
                                iteration history (weight, R, J, max |Delta u|, the law per solve) and per protocol the
                                per-sweep fits (J_s / dof_s, the count with p < 0.01). Its first line is the tail of the
                                command.
parse_readback2.py              parses it and compares with nf1 (needs nf1's noise_cells_L3_exc.csv and
                                noise_sweeps_L3_exc.csv, which are in the archive, not in the repository):
                                    python3 -I parse_readback2.py readback2_2026-11-19.txt <nf1 cells csv> <nf1 sweeps csv>
nf2_cells_from_paste_2026-11-19.csv   its per-cell table (final law, J, Sigma solves, nf1 J and slow part, sigma_s^2/tau_s)
check_slow_ridge.py             closed-form check (noise_law.expected_s0 / expected_sl) for 571430815: the slow part moved
                                along sigma_s^2 / tau_s = const from the nf2 fit (1.615 mV, 5.656 s) to tau_s = 1 s and
                                0.2 s, the 21 statistics compared against the data's bootstrap SE (nf1 JSON, the 15
                                statistics nf1 and nf2 share), and the slow part over a 7.42 s SS train:
                                    python3 check_slow_ridge.py <Ih Fit folder> <nf1 specimen_571430815.json>
check_slow_ridge_2026-11-19.txt its output (numpy 2.5.3, scipy 1.18.1)

What came out
-------------
- J/dof lower than nf1 in 18 of 18 cells: median 10.8 -> 2.65; J < 32.0 (chi2_16 99 %) in 2 cells, < 39.25 (99.9 %) in 9.
- No coordinate at a bound; tau_m 0.053-0.111 ms, tau_s 11.2 ms - 5.66 s.
- Per-sweep fits: p < 0.01 in 12 of 64 SS sweeps (nf1 44 of 64) and 117 of 516 LS sweeps (nf1 289 of 515).
- 15 of 18 cells `no_plateau`; over the rounds Sigma_2..final tau_s moved by more than 20 % in 7 cells while
  sigma_s^2 / tau_s moved by 5-29 % (5 % in 526785799 and 571430815).
- The check: in 571430815, tau_s from 5.66 s to 1 s at fixed sigma_s^2 / tau_s moves no statistic by more than
  0.35 SE (to 0.2 s: 1.92 SE, with the other parameters not refitted); over a 7.42 s train the slow part's SD is
  0.92, 0.59 and 0.30 mV, its correlation between pulses 200.5 ms apart 0.965, 0.818 and 0.367.
Reading [reasoning]: where tau_s is long against the windows (<= 265 ms) the run determines sigma_s^2 / tau_s, not
sigma_s and tau_s apart; the slow noise of a twin longer than the windows rests on an extrapolation.
Limits: closed-form statistics (mean removed, tilt not included), one cell, the other parameters not refitted.
