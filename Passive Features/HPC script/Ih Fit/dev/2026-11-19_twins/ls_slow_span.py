"""Standalone check (not pipeline code): the slow OU part of an LS twin over the span the fit reads.
For each cell's nf2 slow part (sigma_s, tau_s): SD of [X(t) - mean of X over the pre-window [0, 0.265) s]
at the end of the 1 s step (t = 1.27 s) and its RMS over the step [0.27, 1.27] s, for
(A1) the law as fitted and (A2) the law moved along sigma_s^2/tau_s = const to tau_s = min(tau_s, 0.265 s).
Exact for a stationary OU process, by quadrature of its covariance sigma^2 exp(-|u-v|/tau)."""
import csv, sys
import numpy as np

def dev_sd(sigma, tau, t_pts, T=0.265, n=2001):
    u = np.linspace(0.0, T, n)
    # Var(mean over [0,T]) = sigma^2 * 2 tau^2/T^2 * (T/tau - 1 + exp(-T/tau))
    z = T / tau
    var_mean = sigma**2 * 2.0 / z**2 * (z - 1.0 + np.exp(-z))
    out = []
    for t in t_pts:
        cov = sigma**2 * np.trapezoid(np.exp(-np.abs(t - u) / tau), u) / T
        out.append(sigma**2 + var_mean - 2.0 * cov)
    return np.sqrt(np.array(out))

rows = list(csv.DictReader(open(sys.argv[1])))
t_step = np.linspace(0.27, 1.27, 201)
print("specimen   sigma_s_mV  tau_s_ms | A1: SD@end  RMS(step) | A2(tau<=265ms): sigma_s  SD@end  RMS(step)")
for r in sorted(rows, key=lambda r: -float(r["ts"])):
    s, tau = float(r["ss"]), float(r["ts"]) * 1e-3
    d1 = dev_sd(s, tau, t_step)
    tau2 = min(tau, 0.265); s2 = s * np.sqrt(tau2 / tau)
    d2 = dev_sd(s2, tau2, t_step)
    print("%s  %9.4f  %8.1f | %8.3f  %8.3f | %8.4f  %8.3f  %8.3f" % (
        r["sid"], s, tau * 1e3, d1[-1], np.sqrt(np.mean(d1**2)), s2, d2[-1], np.sqrt(np.mean(d2**2))))
