"""Check by computation: can the dense thin-plate spline (scipy RBFInterpolator, kernel thin_plate_spline,
no `neighbors`) take G * n_real = 1e6 rows, and what does the local kernel window hold at that N?

(1) Time two dense fits (N = 3000, 6000; 3 inputs, smoothing 10) here, read the exponent, extrapolate to
    N = 1e5 and 1e6; the dense system's memory is 8 N^2 bytes (float64) -- SciPy also forms the (N+4)x(N+4) system.
(2) The design's kernel window: fraction of rows within the Gaussian kernel of bandwidths (0.15, 5 deg, 0.3) in
    (ln d, phi, ln alpha_hat) at three query points, from 2e5 draws of the D-024 laws with alpha_hat = true alpha
    (the stand-in the table chat used), scaled to N = 1e6.
(3) The support bracket in ln d at fixed (phi, alpha_hat): ln(mu_max/mu_min) / ln(d_max/d_min).
"""
import time
import numpy as np
from scipy.interpolate import RBFInterpolator

SEED = 20261020
D_RANGE, PHI_RANGE_DEG, MU_RANGE = (0.2, 4.0), (0.0, 90.0), (0.3, 3.0)
BW = (0.15, np.radians(5.0), 0.3)   # D-054 (ii), raw coordinates (ln d, rad, ln alpha_hat)

def draws(n, rng):
    d = np.exp(rng.uniform(np.log(D_RANGE[0]), np.log(D_RANGE[1]), n))
    phi = np.radians(rng.uniform(*PHI_RANGE_DEG, n))
    mu = np.exp(rng.uniform(np.log(MU_RANGE[0]), np.log(MU_RANGE[1]), n))
    alpha = mu * d / np.cos(phi)
    return np.column_stack([np.log(d), phi, np.log(alpha)])

def time_fit(N, rng):
    X = draws(N, rng)
    u = (X - X.min(0)) / (X.max(0) - X.min(0))
    y = 1 + 0.06 / np.exp(X[:, 0]) + 0.15 * np.sin(X[:, 1]) ** 2 + 0.05 * X[:, 2] + rng.normal(0, 0.03, N)
    t0 = time.perf_counter()
    RBFInterpolator(u, y, kernel="thin_plate_spline", smoothing=10.0)
    return time.perf_counter() - t0

rng = np.random.default_rng(SEED)
t3, t6 = time_fit(3000, rng), time_fit(6000, rng)
p = np.log(t6 / t3) / np.log(2.0)
print("(1) dense fit: N=3000 %.2f s, N=6000 %.2f s, exponent %.2f" % (t3, t6, p))
for N in (1e5, 1e6):
    print("    N=%.0e: time ~ %.3g h (extrapolated at exponent %.2f), dense system memory 8 N^2 = %.3g GB"
          % (N, t6 * (N / 6000) ** p / 3600, p, 8 * N ** 2 / 1e9))
print("    handoff's T6 (table chat): 0.27, 2.85, 24.8 s at 1500, 3000, 6000, exponent 3.1 [KB]")

n = 200_000
X = draws(n, rng)
for name, q in (("d=1 um, 30 deg, alpha=1.2", (np.log(1.0), np.radians(30), np.log(1.2))),
                ("d=0.5 um, 10 deg, alpha=0.3", (np.log(0.5), np.radians(10), np.log(0.3))),
                ("d=3 um, 45 deg, alpha=4", (np.log(3.0), np.radians(45), np.log(4.0)))):
    w = np.exp(-0.5 * (((X - np.asarray(q)) / np.asarray(BW)) ** 2).sum(1))
    mass = w.sum() / n                      # mean kernel weight per row
    neff = w.sum() ** 2 / (w ** 2).sum()    # at n rows
    print("(2) %-30s mean weight/row %.2e -> kernel mass at N=1e6: %.0f; N_eff(local) at N=1e6 ~ %.0f"
          % (name, mass, mass * 1e6, neff * 1e6 / n))
print("(3) support bracket in ln d at fixed (phi, alpha_hat): ln(mu_max/mu_min)/ln(d_max/d_min) = %.3f"
      % (np.log(MU_RANGE[1] / MU_RANGE[0]) / np.log(D_RANGE[1] / D_RANGE[0])))
