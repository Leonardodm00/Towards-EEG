TEEG -- the tilt readings of D-102 and D-107 (Long Square and SS sweeps), 2026-10-29 -- standalone, not pipeline code

In the repository since 2026-10-29 (D-097). Written in the sandbox for the noise-fit chat's record of D-107 (the user,
2026-10-29: a Long Square sweep's tilt is the line through a start-window mean and an end-window mean). Nothing here is
imported by the Ih Fit code or runs in a job. The script imports noise_law (Block 2) from the Ih Fit folder two levels
up (or TEEG_IH_FIT) and runs from this folder with python (numpy, scipy).

| script | output | what it shows | time |
|---|---|---|---|
| check_ls_tilt_reading.py | ls_tilt_reading_2026-10-29.txt | at theta_cf = (0.0380 mV, 0.175, 0.0508 mV, 36.6 ms), 50 kHz, the line through the means of the 265 ms pre-window and the last 500 ms of a 7.270 s LS sweep (centres 6.8875 s apart) reads the sweep's tilt with an error SD of 0.0045 mV/s (exact, from noise_law's window-mean moments; 0.00449 over 2000 simulated sweeps, seed 20261029); the two window means are independent (covariance 1.7e-82 mV^2) | seconds |

| check_ss_tilt_reading.py (added 2026-10-29, later) | ss_tilt_reading_2026-10-29.txt | the SS reading of D-102.2 under D-103 (theta_cf, sigma_b 0.1 mV/s, the tilt study's four reference layouts, 2000 or 400 cells, seed 20261040): the per-(s,p) median, the per-polarity mean or no subtraction give the same slopes (2e-16 mV/s); RMS error 0.0027-0.0036 mV/s at 80 ms, 0.0031-0.0041 at 130 ms, 0.0034-0.0044 at 150 ms, each within 3 SE of the exact value from the law's window-mean covariances. Needs the tilt study's scripts (tilt_common.py), passed as its argument or found in ../2026-10-26_tilt_sigma_b/scripts | ~8 min |

Re-run on 2026-10-29 from this folder: check_ls_tilt_reading.py output identical, timing aside; check_ss_tilt_reading.py run once (P1-P3 PASS).
