#!/usr/bin/env python3
"""Check by computation: is only sigma_s^2 / tau_s determined when tau_s is much longer than the windows?
Closed-form statistics (noise_law.expected_s0 / expected_sl, window mean removed, as Block 3's S_0, S_L) of
571430815's nf2 law and of the same law with the slow part moved along sigma_s^2 / tau_s = const to tau_s = 0.2 s
and 1 s; differences against the data's bootstrap SE of each statistic (nf1 JSON, the 15 statistics nf1 and nf2
share; Block 3 unchanged for them). Then the slow part over 7.4 s (a continuous 37-pulse SS train) and the
correlation of the slow part between pulses 200.5 ms apart.
Usage: python3 check_slow_ridge.py <Ih Fit code dir> <nf1 specimen_571430815.json>"""
import json, math, sys
import numpy as np
sys.path.insert(0, sys.argv[1])
import noise_law as nl
import noise_cell_stats as ncs

rec = json.load(open(sys.argv[2]))
fs = float(rec["windows"]["fs_Hz"]); dt = 1.0 / fs
m_ss = 10500 - (500 + int(round(rec["windows"]["t_ss_ms"] * 1e-3 * fs)))
n_ls, n_first, n_last = 13250, int(round(0.050 * fs)), int(round(0.1325 * fs))
SE = dict(zip(rec["cell_fit"]["labels_kept"], rec["cell_fit"]["SE"]))

def stats(law):
    out = {}
    def sd(n): return math.sqrt(nl.expected_s0(law, dt, n) / (n - 1))
    def acf(n, lags_ms):
        lags = ncs.lags_in_samples(lags_ms, fs)
        return dict(zip(lags_ms, nl.expected_sl(law, dt, n, lags) / nl.expected_s0(law, dt, n)))
    out["ss_sd"] = sd(m_ss)
    out.update({"ss_rho_%gms" % k: v for k, v in acf(m_ss, ncs.SS_LAGS_MS).items()})
    out["ls_sd"] = sd(n_ls)
    out.update({"ls_rho_%gms" % k: v for k, v in acf(n_ls, ncs.LS_FULL_LAGS_MS).items()})
    out["ls_first_sd"] = sd(n_first); out["ls_last_sd"] = sd(n_last)
    out.update({"ls_last_rho_%gms" % k: v for k, v in acf(n_last, ncs.LS_LAST_LAGS_MS).items()})
    return out

base = dict(sigma_f=0.0315, sigma_m=0.0178, tau_m=0.078e-3)          # 571430815, nf2 final
ratio = 1.615 ** 2 / 5.6563                                        # mV^2/s, nf2 final
laws = {"nf2 (1.615 mV, 5.656 s)": nl.NoiseLaw(base["sigma_f"], base["sigma_m"], base["tau_m"], 1.615, 5.6563)}
for ts in (1.0, 0.2):
    laws["ridge (%.3f mV, %g s)" % (math.sqrt(ratio * ts), ts)] = nl.NoiseLaw(base["sigma_f"], base["sigma_m"], base["tau_m"], math.sqrt(ratio * ts), ts)
names = list(laws)
Q = {k: stats(v) for k, v in laws.items()}
print("SS window m = %d samples (%.1f ms); LS %d, first %d, last %d" % (m_ss, 1e3 * m_ss / fs, n_ls, n_first, n_last))
print("%-20s %10s %10s %10s | %9s %9s" % ("statistic", "nf2", "tau 1 s", "tau 0.2 s", "d(1s)/SE", "d(.2s)/SE"))
worst = {}
for lab in Q[names[0]]:
    a, b, c = (Q[n][lab] for n in names)
    se = SE.get(lab)
    z1 = (b - a) / se if se else float("nan"); z2 = (c - a) / se if se else float("nan")
    worst[lab] = (z1, z2)
    print("%-20s %10.5f %10.5f %10.5f | %9.2f %9.2f" % (lab, a, b, c, z1, z2))
fin = [v for v in worst.values() if np.isfinite(v[0])]
print("max |difference| / SE over the %d statistics with an SE: tau 1 s %.2f, tau 0.2 s %.2f" % (len(fin), max(abs(v[0]) for v in fin), max(abs(v[1]) for v in fin)))
print()
for k, law in laws.items():
    slow = nl.NoiseLaw(0.0, 0.0, base["tau_m"], law.sigma_s, law.tau_s)
    n_train = int(round(37 * 0.2005 * fs))
    sd_train = math.sqrt(nl.expected_s0(slow, dt, n_train) / (n_train - 1))
    sd_200 = math.sqrt(nl.expected_s0(slow, dt, int(0.2 * fs)) / (int(0.2 * fs) - 1))
    print("%-24s slow part: SD within a 200 ms window %.4f mV; within a 7.42 s train %.4f mV; correlation between pulses 200.5 ms apart %.3f"
          % (k, sd_200, sd_train, math.exp(-0.2005 / law.tau_s)))
