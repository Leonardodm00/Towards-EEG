"""check09_variogram_vs_autocov.py -- standalone check written for the brief
chat comparison of the semivariogram and the autocovariance on 2026-10-28
(not pipeline code, not in the repository).

Estimators, per record x_1..x_n (h in samples):
  gamma_hat(h) = 1/(2(n-h)) sum_t (x_{t+h} - x_t)^2                 (no mean)
  C_hat(h)     = 1/(n-h) sum_t (x_t - xbar)(x_{t+h} - xbar)          (record's own mean)
  C_known(h)   = 1/(n-h) sum_t x_t x_{t+h}                           (true mean 0 used)
Claims tested, predictions fixed before running:
 B  short records (OU, sigma2 = 1, tau = 20 samples, n = 100 = 5 tau): C_hat is
    biased low (z < -5 at h = 0 and h = tau, against C(h) = exp(-h/tau)), by
    about Var(xbar) = 2 (z - 1 + e^-z)/z^2 = 0.32 at z = n/tau = 5;
    gamma_hat is unbiased for gamma(h) = 1 - exp(-h/tau) (|z| < 3).
 C  white measurement noise (eta2 = 0.25) on long records: in C_known it sits
    at h = 0 only (C(0) = 1.25, C(1) = exp(-1/20)); in gamma_hat it is a
    constant nugget at every h >= 1 (gamma(1) = 0.25 + 1 - exp(-1/20)); |z| < 3.
 D  random walk (q = 1): C_hat at h = 10 grows with the record length (ratio
    between n = 4000 and n = 500 larger than 5: no stationary covariance);
    gamma_hat(10) = q h / 2 = 5 at both lengths (|z| < 3).
Seed 20261037; numpy only.  Run:  python check09_variogram_vs_autocov.py
"""
import time

import numpy as np

CFG = dict(seed=20261037, tau=20.0, n_short=100, n_rec_short=20000, eta2=0.25, n_long=4000,
           n_rec_long=400, q=1.0, rw_lengths=(500, 4000), n_rec_rw=2000, h_rw=10)


def ou(rng, n_rec, n_len, tau):
    rho = np.exp(-1.0 / tau)
    z = np.empty((n_rec, n_len))
    z[:, 0] = rng.standard_normal(n_rec)
    e = np.sqrt(1 - rho ** 2) * rng.standard_normal((n_rec, n_len))
    for j in range(1, n_len):
        z[:, j] = rho * z[:, j - 1] + e[:, j]
    return z


def gam(x, h):
    return 0.5 * np.mean((x[:, h:] - x[:, :-h]) ** 2, axis=1) if h else np.zeros(x.shape[0])


def cov(x, h, known_mean=False):
    u = x if known_mean else x - x.mean(axis=1, keepdims=True)
    return np.mean(u[:, h:] * u[:, :u.shape[1] - h], axis=1)


def test(label, per, pred, rule):
    m, se = per.mean(), per.std(ddof=1) / np.sqrt(per.size)
    z = (m - pred) / se
    ok = rule(z)
    print("%-34s mean %.4f +/- %.4f, predicted %.4f, z = %+7.2f -> %s" % (label, m, se, pred, z, "PASS" if ok else "FAIL"))
    return ok


def main():
    t0 = time.time()
    rng = np.random.default_rng(CFG["seed"])
    tau, ok = CFG["tau"], True
    # B: short records, mean estimated vs differences
    x = ou(rng, CFG["n_rec_short"], CFG["n_short"], tau)
    zz = CFG["n_short"] / tau
    print("B  OU, n = %d = %.0f tau; Var(xbar) = %.3f" % (CFG["n_short"], zz, 2 * (zz - 1 + np.exp(-zz)) / zz ** 2))
    for h in (0, int(tau)):
        ok &= test("   C_hat(%d) (own mean)" % h, cov(x, h), np.exp(-h / tau), lambda z: z < -5)
    for h in (1, int(tau)):
        ok &= test("   gamma_hat(%d) (no mean)" % h, gam(x, h), 1 - np.exp(-h / tau), lambda z: abs(z) < 3)
    # C: white noise: spike in C at 0, nugget in gamma at every h >= 1
    y = ou(rng, CFG["n_rec_long"], CFG["n_long"], tau) + np.sqrt(CFG["eta2"]) * rng.standard_normal((CFG["n_rec_long"], CFG["n_long"]))
    print("C  OU + white noise eta2 = %.2f, n = %d" % (CFG["eta2"], CFG["n_long"]))
    ok &= test("   C_known(0)", cov(y, 0, True), 1 + CFG["eta2"], lambda z: abs(z) < 3)
    ok &= test("   C_known(1)", cov(y, 1, True), np.exp(-1 / tau), lambda z: abs(z) < 3)
    ok &= test("   gamma_hat(1)", gam(y, 1), CFG["eta2"] + 1 - np.exp(-1 / tau), lambda z: abs(z) < 3)
    ok &= test("   gamma_hat(5)", gam(y, 5), CFG["eta2"] + 1 - np.exp(-5 / tau), lambda z: abs(z) < 3)
    # D: random walk: C_hat depends on the record length, gamma_hat does not
    h = CFG["h_rw"]
    print("D  random walk q = %.0f, h = %d" % (CFG["q"], h))
    c_by_n = {}
    for n in CFG["rw_lengths"]:
        w = np.cumsum(np.sqrt(CFG["q"]) * rng.standard_normal((CFG["n_rec_rw"], n)), axis=1)
        c_by_n[n] = cov(w, h).mean()
        print("   n = %4d: mean C_hat(%d) = %.1f" % (n, h, c_by_n[n]))
        ok &= test("   gamma_hat(%d), n = %d" % (h, n), gam(w, h), CFG["q"] * h / 2, lambda z: abs(z) < 3)
    ratio = c_by_n[CFG["rw_lengths"][1]] / c_by_n[CFG["rw_lengths"][0]]
    okr = ratio > 5
    ok &= okr
    print("   C_hat ratio n = %d / n = %d: %.2f (> 5: %s)" % (CFG["rw_lengths"][1], CFG["rw_lengths"][0], ratio, okr))
    print("\nALL %s   (%.1f s)" % ("PASS" if ok else "FAIL", time.time() - t0))


if __name__ == "__main__":
    main()
