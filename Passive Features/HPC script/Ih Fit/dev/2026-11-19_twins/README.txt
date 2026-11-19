dev/2026-11-19_twins -- standalone checks made while settling Block 5 (the twins) with the user
===============================================================================================

Standalone checks, not pipeline code (2026-11-19, the twins chat). Nothing here is imported by the pipeline.

Files
-----
ls_slow_span.py               the slow OU part of a Long Square twin over the span the fit reads: per cell, the SD of
                              [X(t) - mean of X over the pre-window [0, 0.265) s] at the end of the 1 s step (t = 1.27 s)
                              and its RMS over the step, for the law as fitted and for the law moved along
                              sigma_s^2 / tau_s = const to tau_s = min(tau_s, 0.265 s); exact OU covariance, quadrature
                              over the pre-window. Input: the nf2 per-cell table as pasted.
                                  python3 ls_slow_span.py ../2026-11-19_nf2_readback/nf2_cells_from_paste_2026-11-19.csv
ls_slow_span_2026-11-19.txt   its output (numpy 2.5.3); the numbers of decision D-182
check_bundle_variance.py      Var[mean of K equally correlated window means] = (v / K) (1 + (K - 1) rho) against a
                              Monte Carlo estimate at K = 19, rho = 0 and rho = 0.965 (seed and tolerance in the file)
                                  python3 check_bundle_variance.py

What came out
-------------
- Option (a) of D-182 (the law as fitted) against the short end of the ridge: 0.954 against 0.457 mV at the end of the
  step in 571430815, 0.297 against 0.168 mV in 526785799, 0.148 against 0.128 mV in 571624532, 0.156 against 0.150 mV
  in 548408551; the same under both in the other 14 cells (0.049 to 0.155 mV).
- The bundle formula holds at both correlations (ratio to the Monte Carlo value within 1 %): a bundle of 19 windows
  whose window means correlate at 0.965 has 18.4 times the variance of a bundle of 19 independent windows.
Limits: the slow part alone; a point value against the pre-window mean, not the fit's loss; the nf2 laws as pasted.
