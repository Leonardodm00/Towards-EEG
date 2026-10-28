"""check08_semivariogram_shapes.py -- standalone check written for the brief
chat explanation of the semivariogram on 2026-10-28 (not pipeline code, not in
the repository).

Claims tested (gamma(h) = 1/2 E[(Z(t+h) - Z(t))^2], h in samples):
 C1  stationary process with white measurement noise: OU (AR(1)), variance
     sigma2 = 1, correlation time tau = 20 samples, plus white noise eta2 = 0.25.
     For h >= 1, gamma(h) = eta2 + sigma2 (1 - exp(-h/tau)) = C(0) - C(h):
     a jump at h -> 0 (the nugget, eta2), a plateau at C(0) = sigma2 + eta2
     (the sill) once h >> tau (the range).
 C2  random walk, increment variance q = 1 per sample: gamma(h) = q h / 2,
     linear and unbounded (no sill, no finite variance).
 C3  one random slope per record, b ~ N(0, s2), s = 0.01 per sample, plus white
     noise eta2 = 0.25: gamma(h) = eta2 + s2 h^2 / 2, quadratic.
 C4  a constant offset per record (a ~ N(0, 10^2)) leaves the empirical
     semivariogram unchanged: differences cancel it, no mean is estimated.
Tolerance, fixed before running: |gamma_hat - gamma| < 3 Monte Carlo SE
(SE from the spread across independent records) for C1-C3; relative
difference < 1e-9 for C4. Seed 20261036. numpy only.

Run:  python check08_semivariogram_shapes.py      (a few seconds)
"""
import time

import numpy as np

CFG = dict(seed=20261036, n_rec=400, n_len=4000, tau=20.0, sigma2=1.0, eta2=0.25,
           q=1.0, s=0.01, n_rec_tilt=4000, n_len_tilt=500,
           lags_ou=(1, 20, 160), lags_rw=(10, 100, 1000), lags_tilt=(50, 100, 400), offset_sd=10.0)


def ou(rng, n_rec, n_len, tau, sigma2):
    """AR(1) = OU sampled every sample, started in its stationary law."""
    rho = np.exp(-1.0 / tau)
    z = np.empty((n_rec, n_len))
    z[:, 0] = np.sqrt(sigma2) * rng.standard_normal(n_rec)
    innov = np.sqrt(sigma2 * (1 - rho ** 2)) * rng.standard_normal((n_rec, n_len))
    for j in range(1, n_len):
        z[:, j] = rho * z[:, j - 1] + innov[:, j]
    return z


def gamma_hat(z, h):
    """Empirical semivariogram at lag h: per record, half the mean squared
    difference; returns (mean over records, Monte Carlo SE over records)."""
    per = 0.5 * np.mean((z[:, h:] - z[:, :-h]) ** 2, axis=1)
    return per.mean(), per.std(ddof=1) / np.sqrt(per.size), per


def report(name, h, g, se, pred):
    ok = abs(g - pred) < 3 * se
    print("%-4s h = %5d: gamma_hat = %.5f +/- %.5f, predicted %.5f, z = %+.2f -> %s"
          % (name, h, g, se, pred, (g - pred) / se, "PASS" if ok else "FAIL"))
    return ok


def main():
    t0 = time.time()
    rng = np.random.default_rng(CFG["seed"])
    ok = True
    # C1: OU + white noise
    z = ou(rng, CFG["n_rec"], CFG["n_len"], CFG["tau"], CFG["sigma2"])
    z = z + np.sqrt(CFG["eta2"]) * rng.standard_normal(z.shape)
    for h in CFG["lags_ou"]:
        g, se, _ = gamma_hat(z, h)
        ok &= report("C1", h, g, se, CFG["eta2"] + CFG["sigma2"] * (1 - np.exp(-h / CFG["tau"])))
    print("     sill C(0) = sigma2 + eta2 = %.3f; nugget eta2 = %.3f" % (CFG["sigma2"] + CFG["eta2"], CFG["eta2"]))
    # C4: offset invariance on the C1 records
    zo = z + CFG["offset_sd"] * rng.standard_normal((z.shape[0], 1))
    rel = max(abs(gamma_hat(zo, h)[0] / gamma_hat(z, h)[0] - 1) for h in CFG["lags_ou"])
    ok4 = rel < 1e-9
    ok &= ok4
    print("C4   per-record offsets of SD %.0f: largest relative change of gamma_hat = %.1e -> %s"
          % (CFG["offset_sd"], rel, "PASS" if ok4 else "FAIL"))
    # C2: random walk
    w = np.cumsum(np.sqrt(CFG["q"]) * rng.standard_normal((CFG["n_rec"], CFG["n_len"])), axis=1)
    for h in CFG["lags_rw"]:
        g, se, _ = gamma_hat(w, h)
        ok &= report("C2", h, g, se, CFG["q"] * h / 2)
    # C3: random slope per record + white noise
    n, L = CFG["n_rec_tilt"], CFG["n_len_tilt"]
    b = CFG["s"] * rng.standard_normal((n, 1))
    tt = np.arange(L)[None, :] - (L - 1) / 2
    x = b * tt + np.sqrt(CFG["eta2"]) * rng.standard_normal((n, L))
    for h in CFG["lags_tilt"]:
        g, se, _ = gamma_hat(x, h)
        ok &= report("C3", h, g, se, CFG["eta2"] + CFG["s"] ** 2 * h ** 2 / 2)
    print("\nALL %s   (%.1f s)" % ("PASS" if ok else "FAIL", time.time() - t0))


if __name__ == "__main__":
    main()
