"""check11_second_order.py -- standalone check for the brief chat answer of
2026-10-28 on second-order stationarity (not pipeline code).
Claims, predictions fixed before running (|z| < 3 against the formula):
 OU (sigma2 = 1, tau = 20 samples, started stationary): Var Z(t) = 1 at t = 10 and
   t = 900, and Cov(Z(t), Z(t+20)) = exp(-1) at t = 10 and t = 900 (position-free).
 random walk (q = 1 per sample): Var Z(t) = q t at t = 100 and t = 900 (grows).
 random slope (s = 0.01 per sample, record centre t_bar = 500): Var Z(t) =
   s^2 (t - t_bar)^2 at t = 400 and t = 100 (depends on position).
Seed 20261040; numpy only."""
import numpy as np
rng = np.random.default_rng(20261040)
n_rec, n_len, tau = 20000, 1001, 20.0
rho = np.exp(-1 / tau)
z = np.empty((n_rec, n_len)); z[:, 0] = rng.standard_normal(n_rec)
e = np.sqrt(1 - rho ** 2) * rng.standard_normal((n_rec, n_len))
for j in range(1, n_len):
    z[:, j] = rho * z[:, j - 1] + e[:, j]
w = np.cumsum(rng.standard_normal((n_rec, n_len)), axis=1)          # w[:, t-1] = Z(t), t = 1..n_len
b = 0.01 * rng.standard_normal((n_rec, 1)); sl = b * (np.arange(n_len)[None, :] - 500)
ok = True
def t_(label, x, pred):
    global ok
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(x.size)
    good = abs(m - pred) < 3 * se; ok &= good
    print("%-34s %.4f +/- %.4f, predicted %.4f -> %s" % (label, m, se, pred, "PASS" if good else "FAIL"))
for t in (10, 900):
    t_("OU   Var Z(%d)" % t, z[:, t] ** 2, 1.0)
    t_("OU   Cov(Z(%d), Z(%d))" % (t, t + 20), z[:, t] * z[:, t + 20], np.exp(-1.0))
for t in (100, 900):
    t_("walk Var Z(%d)" % t, w[:, t - 1] ** 2, float(t))
for t in (400, 100):
    t_("slope Var Z(%d), t - t_bar = %d" % (t, t - 500), sl[:, t] ** 2, 1e-4 * (t - 500) ** 2)
print("ALL", "PASS" if ok else "FAIL")
