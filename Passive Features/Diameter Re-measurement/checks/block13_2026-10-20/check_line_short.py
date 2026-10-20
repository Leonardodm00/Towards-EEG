"""P(round 1 ends 'line_short') for a phantom whose start radius is r_br = d/2 + eps,
eps ~ N(0, 0.2^2) redrawn until r_br > 0 (D-062.1), with gradient_iteration's rule:
line_short iff round(h / step) < 1, h = m * d0 / 2 = m * r_br (d0 = 2 r_br), m = 3, step = 0.1144 um.
Exact: truncated normal cdf; Monte Carlo with the redraw loop as a cross-check.

Run (from anywhere): python checks/block13_2026-10-20/check_line_short.py
Pure ASCII (hpc-python-compat)."""
import numpy as np
from scipy import stats

m, step, sd = 3.0, 0.1144, 0.2
# round(x) < 1 <=> x < 0.5 (Python rounds 0.5 to 0): h < 0.5 step <=> r_br < 0.5 step / m
r_crit = 0.5 * step / m
rng = np.random.default_rng(0)
print("r_crit = %.5f um (line of < 3 samples)" % r_crit)
for d in (0.2, 0.3, 0.4, 0.6, 1.0):
    a = (0.0 - d / 2) / sd                      # truncation at r_br > 0
    tn = stats.truncnorm(a, np.inf, loc=d / 2, scale=sd)
    p_exact = tn.cdf(r_crit)
    # Monte Carlo with the redraw rule itself
    n, out = 200000, np.empty(200000)
    k = 0
    while k < n:
        x = d / 2 + sd * rng.standard_normal(2 * (n - k))
        x = x[x > 0][: n - k]
        out[k:k + x.size] = x
        k += x.size
    h = m * out
    p_mc = np.mean(np.array([round(v) for v in (h / step)]) < 1)
    print("d = %.1f um: P(line_short) exact %.4f, Monte Carlo %.4f (n = %d)" % (d, p_exact, p_mc, n))
