"""check10_nugget_offset.py -- standalone check for the brief chat answer of
2026-10-28 on "on top of any nugget" (not pipeline code).
Claim: adding independent white noise of variance eta2 to a process adds the
constant eta2 to its semivariogram at every lag h >= 1 and leaves gamma(0) = 0,
so the shape (here a random walk's q h / 2) is unchanged, only shifted up.
Prediction, fixed before running: gamma_hat(h) of random walk + noise minus
gamma_hat(h) of the same random walk without noise equals eta2 within 3 SE at
h = 1, 10, 100 (common random numbers). Seed 20261038; numpy only."""
import numpy as np
rng = np.random.default_rng(20261038)
q, eta2, n_rec, n_len = 1.0, 0.25, 2000, 2000
w = np.cumsum(np.sqrt(q) * rng.standard_normal((n_rec, n_len)), axis=1)
y = w + np.sqrt(eta2) * rng.standard_normal((n_rec, n_len))
g = lambda x, h: 0.5 * np.mean((x[:, h:] - x[:, :-h]) ** 2, axis=1)
ok = True
for h in (1, 10, 100):
    d = g(y, h) - g(w, h)
    m, se = d.mean(), d.std(ddof=1) / np.sqrt(n_rec)
    gy = g(y, h).mean()
    good = abs(m - eta2) < 3 * se
    ok &= good
    print("h = %3d: gamma_hat(walk + noise) = %.3f (predicted eta2 + q h / 2 = %.3f); shift over the noise-free walk = %.4f +/- %.4f, predicted %.2f -> %s"
          % (h, gy, eta2 + q * h / 2, m, se, eta2, "PASS" if good else "FAIL"))
print("ALL", "PASS" if ok else "FAIL")
