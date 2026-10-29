TEEG -- the Long Square tilt reading of D-107, 2026-10-29 -- standalone, not pipeline code

In the repository since 2026-10-29 (D-097). Written in the sandbox for the noise-fit chat's record of D-107 (the user,
2026-10-29: a Long Square sweep's tilt is the line through a start-window mean and an end-window mean). Nothing here is
imported by the Ih Fit code or runs in a job. The script imports noise_law (Block 2) from the Ih Fit folder two levels
up (or TEEG_IH_FIT) and runs from this folder with python (numpy, scipy).

| script | output | what it shows | time |
|---|---|---|---|
| check_ls_tilt_reading.py | ls_tilt_reading_2026-10-29.txt | at theta_cf = (0.0380 mV, 0.175, 0.0508 mV, 36.6 ms), 50 kHz, the line through the means of the 265 ms pre-window and the last 500 ms of a 7.270 s LS sweep (centres 6.8875 s apart) reads the sweep's tilt with an error SD of 0.0045 mV/s (exact, from noise_law's window-mean moments; 0.00449 over 2000 simulated sweeps, seed 20261029); the two window means are independent (covariance 1.7e-82 mV^2) | seconds |

Re-run on 2026-10-29 from this folder: output identical, timing aside.
