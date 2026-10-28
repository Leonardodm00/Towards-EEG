"""check_sigma_ss.py -- standalone check (not pipeline code; not in the
repository), 2026-10-28: what sigma_SS_hat(c) of the transient test is made of.

sigma_SS_hat(c) = median over the cell's K stored SS windows of the SD (ddof 1)
of each window's first pre_n = 500 samples (t in [-10, 0) ms at 50 kHz), on
the stored voltage as it is (noise_windows.transient_start).

Reported, at theta_cf:
 (i)  closed form (noise_law.expected_s0, eq. (7b) of the topic doc):
      E[sigma_hat^2] of one 500-sample window, and its split into the fast
      and the slow component (the two are independent, so E[S_0] adds);
      the slow component's visible share E[sigma_hat^2 | slow] / sigma_s^2;
 (ii) Monte Carlo with the pipeline's own estimator: sigma_SS_hat(c) on
      simulated cells at the archive's layout "2 sweeps, K 63" (continuous
      sweeps, check_findings_2026-10-23.py), sigma_b = 0 and 0.1 mV/s:
      median and 10-90 % range across cells, against sqrt(E[sigma_hat^2]).
Claim checked (tolerance fixed before running): the Monte Carlo median of
sigma_SS_hat lies within 2 % of sqrt(E[sigma_hat^2]) (the median of SDs sits
slightly below the root of the mean variance), and sigma_b = 0.1 mV/s moves it
by less than 0.5 % (a 10 ms ramp adds (sigma_b x 10 ms)^2 / 12 = 8e-8 mV^2).

Run (from this folder, next to check_findings_2026-10-23.py):
    python check_sigma_ss.py       (about 1 min)
Pure ASCII, LF.
"""
import importlib.util
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("chk", HERE / "check_findings_2026-10-23.py")
chk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(chk)        # sets sys.path to the repository's Ih Fit folder
import noise_law as nl               # noqa: E402
import noise_windows as nw           # noqa: E402

CFG = dict(seed=20261028, n_cells=500, layout="2 sweeps, K 63", sigma_b=(0.0, 0.1), tol_rel=0.02, tol_tilt=0.005)


def closed_form():
    law, dt, n = chk.LAW, chk.DT, chk.PRE_N
    fast = nl.NoiseLaw(law.sigma_f, law.rho_f, 0.0, law.tau_s)
    slow = nl.NoiseLaw(0.0, law.rho_f, law.sigma_s, law.tau_s)
    e_all, e_f, e_s = (nl.expected_s0(m, dt, n) / (n - 1) for m in (law, fast, slow))
    return e_all, e_f, e_s


def monte_carlo(sigma_b, rng):
    sweep, pos = chk.build_layout(chk.CFG["layouts"][CFG["layout"]], rng)
    out = []
    for _ in range(CFG["n_cells"]):
        v = chk.simulate_noise_windows(sigma_b, sweep, pos, rng)
        a = nw.transient_start(chk.make_ss(v, sweep, pos), nw.WindowConfig(t_ss_floor_ms=0.0))
        out.append(a.sigma_ss_mV)
    return np.array(out), pos.size


def main():
    t0 = time.time()
    law = chk.LAW
    e_all, e_f, e_s = closed_form()
    print("theta_cf: sigma_f %.4f mV, rho_f %.3f, sigma_s %.4f mV, tau_s %.1f ms; pre-window %d samples (%.0f ms)"
          % (law.sigma_f, law.rho_f, law.sigma_s, 1e3 * law.tau_s, chk.PRE_N, 1e3 * chk.PRE_N * chk.DT))
    print("closed form E[sigma_hat^2] = %.4e mV^2 (sqrt %.4f mV): fast %.4e (%.1f %%), slow %.4e (%.1f %%); "
          "fast visible share %.3f, slow visible share %.3f of sigma_s^2"
          % (e_all, np.sqrt(e_all), e_f, 100 * e_f / e_all, e_s, 100 * e_s / e_all,
             e_f / law.sigma_f ** 2, e_s / law.sigma_s ** 2))
    rng = np.random.default_rng(CFG["seed"])
    meds = {}
    for sb in CFG["sigma_b"]:
        s, K = monte_carlo(sb, rng)
        meds[sb] = np.median(s)
        print("Monte Carlo, %s (K = %d), sigma_b %.2f mV/s, %d cells: sigma_SS_hat median %.4f mV, 10-90 %% %.4f-%.4f "
              "(spread +/- %.1f %%); median / sqrt(E) = %.4f"
              % (CFG["layout"], K, sb, s.size, meds[sb], np.percentile(s, 10), np.percentile(s, 90),
                 50 * (np.percentile(s, 90) - np.percentile(s, 10)) / meds[sb], meds[sb] / np.sqrt(e_all)))
    ok1 = abs(meds[0.0] / np.sqrt(e_all) - 1.0) <= CFG["tol_rel"]
    ok2 = abs(meds[0.1] / meds[0.0] - 1.0) <= CFG["tol_tilt"]
    print("claim (median within 2 %% of sqrt(E)): %s; claim (tilt moves it < 0.5 %%): %s  (%.0f s)"
          % ("PASS" if ok1 else "FAIL", "PASS" if ok2 else "FAIL", time.time() - t0))
    return ok1 and ok2


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
