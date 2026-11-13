"""check01_generic.py -- standalone check behind TEEG_semivariogram_2026-11-13.md
(sections 3.1-3.4). Not pipeline code: nothing in the Ih Fit pipeline imports it.

Claims tested on simulated records; the predictions and the tolerance were fixed
before the first run. Lags h and record lengths n are in samples, and
gamma(h) = (1/2) E[(X(t+h) - X(t))^2].

 A  white noise of variance eta2 = 0.25: gamma(h) = eta2 at every h >= 1 (a pure nugget).
 B  OU (AR(1) sampled every sample, variance sigma2 = 1, correlation time tau = 20,
    started in its stationary law) plus white noise eta2 = 0.25:
    gamma(h) = eta2 + sigma2 (1 - exp(-h / tau)) = C(0) - C(h) for h >= 1, rising to
    the sill C(0) = sigma2 + eta2 = 1.25; at the practical range h = 3 tau = 60 it has
    covered 1 - exp(-3) = 0.950 of the partial sill sigma2.
 C  random walk, increments of variance q = 1 per sample: gamma(h) = q h / 2 (no sill).
 D  one random slope per record, b ~ N(0, s^2) with s = 0.01 per sample, plus white
    noise eta2 = 0.25: gamma(h) = eta2 + s^2 h^2 / 2.
 E  a constant offset per record (SD 10) leaves the estimate unchanged: relative
    change below 1e-9 (floating-point roundoff only).
 F  short OU records (n = 100 = 5 tau, sigma2 = 1, no nugget):
    F0  the exact expectation of the autocovariance computed with the record's own
        mean,
          E[C_own(h)] = C(h) - [sum_{t=1}^{n-h} z_t + sum_{t=h+1}^{n} z_t] / (n (n - h)) + Var(xbar),
          z_t = sum_{s=1}^{n} C(t - s),   Var(xbar) = sum_t z_t / n^2,
        written here independently, equals noise_law.expected_sl / (n - h) of the
        repository (Block 2, eq. (9) of the topic doc) to 1e-12 relative;
    F1  the simulated mean of C_own(h) agrees with E[C_own(h)] at h = 0, 5, 20, and
        at h = 0 falls short of C(0) = 1 by about Var(xbar) (0.321 here);
    F2  the semivariogram estimate is unbiased for gamma(h) = 1 - exp(-h / tau) at
        h = 1, 5, 20.
 G  random walk (q = 1): the autocovariance at h = 10 computed with the record's own
    mean grows with the record length (mean at n = 4000 over mean at n = 500 above 5),
    while the semivariogram estimate stays at q h / 2 = 5 at both lengths.

Tolerance: |estimate - prediction| < 3.5 Monte Carlo SE, the SE taken from the spread
of the per-record estimates across independent records, for every comparison in
A-D, F1, F2, G; E and F0 as stated. Seed 20261113. numpy (and the repository's
noise_law for F0 only).

Run:  python check01_generic.py            (about 10 s; prints PASS/FAIL per line)
Repository path for F0: environment variable TEEG_IH_FIT, else the Ih Fit folder
three levels above this file.

Pure ASCII, LF.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CFG = dict(
    seed=20261113,
    z_tol=3.5,                     # Monte Carlo SEs
    n_rec=400, n_len=4000,         # A, B, C: records x samples
    eta2=0.25, sigma2=1.0, tau=20.0, q=1.0,
    lags_white=(1, 10, 100),
    lags_ou=(1, 5, 20, 60, 160),
    lags_rw=(10, 100, 1000),
    n_rec_slope=4000, n_len_slope=500, s=0.01, lags_slope=(50, 100, 400),
    offset_sd=10.0,
    n_short=100, n_rec_short=20000, lags_cov_short=(0, 5, 20), lags_gam_short=(1, 5, 20),
    rw_lengths=(500, 4000), n_rec_rw=2000, h_rw=10,
)
REPO_IH_FIT = Path(os.environ.get("TEEG_IH_FIT", str(Path(__file__).resolve().parents[3])))


# ---------------------------------------------------------------------------
# Simulation (data generation)
# ---------------------------------------------------------------------------
def ar1(rng: np.random.Generator, n_rec: int, n_len: int, tau: float, sigma2: float) -> np.ndarray:
    """AR(1) = OU sampled every sample: rho = exp(-1 / tau), started stationary."""
    rho = np.exp(-1.0 / tau)
    x = np.empty((n_rec, n_len))
    x[:, 0] = np.sqrt(sigma2) * rng.standard_normal(n_rec)
    innov = np.sqrt(sigma2 * (1.0 - rho ** 2)) * rng.standard_normal((n_rec, n_len))
    for j in range(1, n_len):
        x[:, j] = rho * x[:, j - 1] + innov[:, j]
    return x


def random_walk(rng: np.random.Generator, n_rec: int, n_len: int, q: float) -> np.ndarray:
    return np.cumsum(np.sqrt(q) * rng.standard_normal((n_rec, n_len)), axis=1)


# ---------------------------------------------------------------------------
# Estimators (analysis)
# ---------------------------------------------------------------------------
def gamma_hat(x: np.ndarray, h: int) -> np.ndarray:
    """Matheron estimate per record: (1 / (2 (n - h))) sum_t (x_{t+h} - x_t)^2."""
    d = x[:, h:] - x[:, :-h]
    return 0.5 * np.mean(d ** 2, axis=1)


def cov_own_mean(x: np.ndarray, h: int) -> np.ndarray:
    """Sample autocovariance per record with the record's own mean, divisor n - h."""
    xc = x - x.mean(axis=1, keepdims=True)
    n = x.shape[1]
    return np.sum(xc[:, : n - h] * xc[:, h:], axis=1) / (n - h)


def expected_cov_own_mean(cov: np.ndarray, n: int, h: int) -> float:
    """Exact E[C_own(h)] for a stationary process with autocovariance cov[k], k = 0..n-1."""
    idx = np.arange(n)
    C = cov[np.abs(idx[:, None] - idx[None, :])]          # n x n covariance matrix
    z = C.sum(axis=1)                                      # z_t, t = 1..n (0-based here)
    var_mean = z.sum() / n ** 2
    head = z[: n - h].sum()                                # sum_{t=1}^{n-h} z_t
    tail = z[h:].sum()                                     # sum_{t=h+1}^{n} z_t
    return float(cov[h] - (head + tail) / (n * (n - h)) + var_mean)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
RESULTS = []


def compare(label: str, per_record: np.ndarray, predicted: float) -> None:
    m = float(per_record.mean())
    se = float(per_record.std(ddof=1) / np.sqrt(per_record.size))
    z = (m - predicted) / se
    ok = abs(z) < CFG["z_tol"]
    RESULTS.append(ok)
    print("%-44s %11.5f +/- %9.5f   predicted %11.5f   z = %+6.2f  %s"
          % (label, m, se, predicted, z, "PASS" if ok else "FAIL"))


def flag(label: str, ok: bool, detail: str) -> None:
    RESULTS.append(bool(ok))
    print("%-44s %s  %s" % (label, detail, "PASS" if ok else "FAIL"))


# ---------------------------------------------------------------------------
# The checks
# ---------------------------------------------------------------------------
def main() -> int:
    t0 = time.time()
    c = CFG
    rng = np.random.default_rng(c["seed"])

    # A: white noise
    w = np.sqrt(c["eta2"]) * rng.standard_normal((c["n_rec"], c["n_len"]))
    for h in c["lags_white"]:
        compare("A  white noise, h = %d" % h, gamma_hat(w, h), c["eta2"])

    # B: OU + white noise (the same white noise as A, independent of the OU)
    y = ar1(rng, c["n_rec"], c["n_len"], c["tau"], c["sigma2"]) + w
    for h in c["lags_ou"]:
        pred = c["eta2"] + c["sigma2"] * (1.0 - np.exp(-h / c["tau"]))
        compare("B  OU + nugget, h = %d" % h, gamma_hat(y, h), pred)
    frac = 1.0 - np.exp(-3.0)
    print("   sill C(0) = %.3f, nugget %.3f, practical range 3 tau = %d samples (%.3f of the partial sill)"
          % (c["sigma2"] + c["eta2"], c["eta2"], int(3 * c["tau"]), frac))

    # C: random walk
    rw = random_walk(rng, c["n_rec"], c["n_len"], c["q"])
    for h in c["lags_rw"]:
        compare("C  random walk, h = %d" % h, gamma_hat(rw, h), c["q"] * h / 2.0)

    # D: random slope + white noise
    n_s = c["n_len_slope"]
    t = np.arange(n_s) - 0.5 * (n_s - 1)
    b = c["s"] * rng.standard_normal((c["n_rec_slope"], 1))
    sl = b * t[None, :] + np.sqrt(c["eta2"]) * rng.standard_normal((c["n_rec_slope"], n_s))
    for h in c["lags_slope"]:
        compare("D  random slope + nugget, h = %d" % h, gamma_hat(sl, h),
                c["eta2"] + 0.5 * c["s"] ** 2 * h ** 2)

    # E: per-record offsets cancel exactly
    off = c["offset_sd"] * rng.standard_normal((c["n_rec"], 1))
    worst = 0.0
    for h in c["lags_ou"]:
        g0, g1 = gamma_hat(y, h), gamma_hat(y + off, h)
        worst = max(worst, float(np.max(np.abs(g1 - g0) / g0)))
    flag("E  offsets SD 10 added per record", worst < 1e-9, "largest relative change %.1e" % worst)

    # F: short OU records, the record's own mean
    n = c["n_short"]
    cov = c["sigma2"] * np.exp(-np.arange(n) / c["tau"])
    var_mean = float(cov[np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])].sum() / n ** 2)
    sys.path.insert(0, str(REPO_IH_FIT))
    try:
        import noise_law as nl   # noqa: E402  (F0 only)
        # sigma_f = 0 switches the fast component off (its rho_f must still lie in (0, 1));
        # dt = 1 "s" per sample, so tau_s = tau samples
        law = nl.NoiseLaw(0.0, 0.5, float(np.sqrt(c["sigma2"])), c["tau"] * 1.0)
        worst0 = 0.0
        for h in c["lags_cov_short"]:
            mine = expected_cov_own_mean(cov, n, h)
            if h == 0:
                repo = nl.expected_s0(law, 1.0, n) / n
            else:
                repo = float(nl.expected_sl(law, 1.0, n, [h])[0]) / (n - h)
            worst0 = max(worst0, abs(mine - repo) / abs(repo))
        flag("F0 E[C_own(h)]: own formula = noise_law", worst0 < 1e-12, "largest relative difference %.1e" % worst0)
    except ImportError as exc:   # the repository is needed for F0 only
        flag("F0 E[C_own(h)]: own formula = noise_law", False, "noise_law not importable (%s)" % exc)
    print("   Var(xbar) at n = %d = 5 tau: %.4f" % (n, var_mean))
    ys = ar1(rng, c["n_rec_short"], n, c["tau"], c["sigma2"])
    for h in c["lags_cov_short"]:
        compare("F1 C_own(%d), own mean (true C = %.4f)" % (h, cov[h]), cov_own_mean(ys, h),
                expected_cov_own_mean(cov, n, h))
    for h in c["lags_gam_short"]:
        compare("F2 gamma_hat(%d), no mean" % h, gamma_hat(ys, h), c["sigma2"] * (1.0 - np.exp(-h / c["tau"])))

    # G: random walk, record length
    means = []
    for n_len in c["rw_lengths"]:
        rwx = random_walk(rng, c["n_rec_rw"], n_len, c["q"])
        means.append(float(cov_own_mean(rwx, c["h_rw"]).mean()))
        compare("G  gamma_hat(%d), random walk, n = %d" % (c["h_rw"], n_len), gamma_hat(rwx, c["h_rw"]),
                c["q"] * c["h_rw"] / 2.0)
        print("   mean C_own(%d) at n = %d: %.1f" % (c["h_rw"], n_len, means[-1]))
    ratio = means[1] / means[0]
    flag("G  C_own(10) grows with n (ratio > 5)", ratio > 5.0, "ratio n = %d / n = %d: %.2f"
         % (c["rw_lengths"][1], c["rw_lengths"][0], ratio))

    print("\nALL %s   (%d checks, %.1f s)" % ("PASS" if all(RESULTS) else "FAIL", len(RESULTS), time.time() - t0))
    return 0 if all(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
