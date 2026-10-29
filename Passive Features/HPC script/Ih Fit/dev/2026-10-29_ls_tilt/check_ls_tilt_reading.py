"""check_ls_tilt_reading.py -- standalone check (not pipeline code), 2026-10-29:
how precisely D-107's rule reads the realised tilt of a Long Square sweep.

Rule (D-107, the user): the tilt of a Long Square sweep is the slope of the
line through two window means, one at the start and one at the end of the
stored sweep,
    b_hat = (ybar_end - ybar_pre) / dt_c,
with ybar_pre the mean over the 265 ms pre-window (samples [0, 13250) at
50 kHz), ybar_end the mean over the last 500 ms of the stored sweep, and dt_c
the distance between the two windows' centres. Archive geometry of the
L3_exc LS sweeps (the living handoff, s.4.6; D-017): pre-window [0, 265) ms,
step onset at 270 ms, step 1 s, 6.000 s stored after the step, so the sweep
is 7.270 s and the end window is [6770, 7270) ms; dt_c = 6.8875 s.

Under law (L1) with the tilt b (t - t_bar): the mean of a line over a window
is its value at the window's centre, so b_hat = b + (e_end - e_pre) / dt_c,
with e_pre, e_end the window means of the stationary part; the error has
variance (V_pre + V_end - 2 C) / dt_c^2, V and C from noise_law's exact
var_window_mean and window_mean_cov for the sampled law.

Truth (chosen): theta_cf = (0.0380 mV, 0.175, 0.0508 mV, 36.6 ms), rho_f at
dt = 20 us (50 kHz). Predictions, fixed before running:
 P1  the covariance C of the two window means is below 1e-12 mV^2 (6.5 s
     apart against a 36.6 ms correlation time), so the two windows can be
     simulated as independent stationary traces;
 P2  the Monte Carlo SD of b_hat - b over 2000 simulated sweeps equals the
     exact value within 3 standard errors of an SD (SE = SD / sqrt(2 (N - 1))).
Seed 20261029. numpy, scipy; noise_law from the Ih Fit folder (two levels
up, or TEEG_IH_FIT).

Run (from this folder):  python check_ls_tilt_reading.py      (about 1 min)
Pure ASCII, LF.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

IH_FIT = Path(os.environ.get("TEEG_IH_FIT", str(Path(__file__).resolve().parents[2])))
sys.path.insert(0, str(IH_FIT))
import noise_law as nl  # noqa: E402

CFG = dict(theta_cf=(0.0380, 0.175, 0.0508, 0.0366), fs=50000.0, pre_ms=265.0, end_ms=500.0,
           sweep_ms=7270.0, n_sweeps=2000, chunk=250, seed=20261029, cov_tol=1e-12, n_se=3.0,
           sigma_b_ref=(0.1, 0.027))


def geometry():
    dt = 1.0 / CFG["fs"]
    n_pre = int(round(CFG["pre_ms"] * 1e-3 * CFG["fs"]))
    n_end = int(round(CFG["end_ms"] * 1e-3 * CFG["fs"]))
    n_tot = int(round(CFG["sweep_ms"] * 1e-3 * CFG["fs"]))
    off_end = n_tot - n_end                                         # start of the end window, samples
    dt_c = (off_end + 0.5 * (n_end - 1) - 0.5 * (n_pre - 1)) * dt   # s, centre to centre
    return dt, n_pre, n_end, off_end, dt_c


def exact_sd(law, dt, n_pre, n_end, off_end, dt_c):
    v_pre = nl.var_window_mean(law, dt, n_pre)
    v_end = nl.var_window_mean(law, dt, n_end)
    cov = nl.window_mean_cov(law, dt, n_pre, n_end, off_end)
    return np.sqrt(v_pre + v_end - 2.0 * cov) / dt_c, v_pre, v_end, cov


def monte_carlo(law, dt, n_pre, n_end, dt_c, rng):
    err = []
    for start in range(0, CFG["n_sweeps"], CFG["chunk"]):
        m = min(CFG["chunk"], CFG["n_sweeps"] - start)
        u_pre = nl.draw_units(rng, m, n_pre)
        u_end = nl.draw_units(rng, m, n_end)
        e_pre = nl.simulate_traces(law, dt, u_pre.fast, u_pre.slow).mean(axis=1)
        e_end = nl.simulate_traces(law, dt, u_end.fast, u_end.slow).mean(axis=1)
        err.append((e_end - e_pre) / dt_c)                         # b_hat - b, mV/s
    return np.concatenate(err)


def main():
    t0 = time.time()
    law = nl.NoiseLaw(*CFG["theta_cf"])
    dt, n_pre, n_end, off_end, dt_c = geometry()
    sd_ex, v_pre, v_end, cov = exact_sd(law, dt, n_pre, n_end, off_end, dt_c)
    print("geometry: pre-window %d samples, end window %d samples starting at sample %d; centres %.4f s apart"
          % (n_pre, n_end, off_end, dt_c))
    print("exact (noise_law): SD of the pre-window mean %.4f mV, of the end-window mean %.4f mV, covariance %.2e mV^2"
          % (np.sqrt(v_pre), np.sqrt(v_end), cov))
    ok1 = abs(cov) < CFG["cov_tol"]
    print("P1 covariance below %.0e mV^2: %s" % (CFG["cov_tol"], "PASS" if ok1 else "FAIL"))
    err = monte_carlo(law, dt, n_pre, n_end, dt_c, np.random.default_rng(CFG["seed"]))
    sd_mc = err.std(ddof=1)
    se = sd_ex / np.sqrt(2.0 * (err.size - 1))
    ok2 = abs(sd_mc - sd_ex) < CFG["n_se"] * se
    print("P2 SD of b_hat - b: exact %.5f mV/s, Monte Carlo %.5f mV/s (N = %d, SE %.5f; mean %+.5f) %s"
          % (sd_ex, sd_mc, err.size, se, err.mean(), "PASS" if ok2 else "FAIL"))
    for sb in CFG["sigma_b_ref"]:
        print("   relative to a tilt SD of %.3f mV/s: %.1f %%" % (sb, 100.0 * sd_ex / sb))
    print("ALL %s   (%.0f s)" % ("PASS" if ok1 and ok2 else "FAIL", time.time() - t0))
    return ok1 and ok2


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
