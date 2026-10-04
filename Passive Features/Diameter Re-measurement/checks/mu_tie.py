"""Eq. 11 fitted three ways on noise-free profiles (B fixed, D-018):
 A  free alpha per node:            params (d, alpha, v0)
 M  free mu per node, alpha = mu d / cos(phi_i), phi_i plugged in:  params (d, mu, v0)
 S  shared mu fixed (true, or +20 %), alpha = mu d / cos(phi_i):     params (d, v0)
Truth rendered with sigma_true = 0.10 um; fits use sigma_fit (0.10 or 0.125)."""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
dx = 0.002
vf = np.arange(-4, 4, dx)
vn = np.arange(-26, 27) * 0.1144 + 0.03          # sample positions, sub-pixel offset
def model(d, a, v0, s):
    u = np.clip(1 - (2 * (vf - v0) / d) ** 2, 0, None)
    T = np.exp(-a * np.sqrt(u))
    return np.interp(vn, vf, gaussian_filter1d(T, s / dx, mode="nearest"))
PHI = np.deg2rad(15.0)
def fit(y, s, mode, mu_fix=None, d0=0.8):
    if mode == "A":
        f = lambda p: model(p[0], p[1], p[2], s) - y
        r = least_squares(f, [d0, 1.0, 0.0], bounds=([0.05, 0, -1], [6, 20, 1]))
    elif mode == "M":
        f = lambda p: model(p[0], p[1] * p[0] / np.cos(PHI), p[2], s) - y
        r = least_squares(f, [d0, 1.0, 0.0], bounds=([0.05, 0, -1], [6, 30, 1]))
    else:
        f = lambda p: model(p[0], mu_fix * p[0] / np.cos(PHI), p[1], s) - y
        r = least_squares(f, [d0, 0.0], bounds=([0.05, -1], [6, 1]))
    return r.x[0]
print("d_hat / d_true   (phi = 15 deg, sigma_true = 0.10 um)")
for mu in (0.6, 3.0):
    print("\nmu_true = %.1f /um" % mu)
    print("%5s %6s | %-24s | %-24s" % ("d", "alpha", "sigma_fit = 0.100", "sigma_fit = 0.125"))
    print("%5s %6s | %6s %5s %5s %6s | %6s %5s %5s %6s" % ("", "", "A", "M", "S", "S+20%", "A", "M", "S", "S+20%"))
    for d in (0.3, 0.5, 1.0, 2.0):
        a = mu * d / np.cos(PHI)
        y = model(d, a, 0.0, 0.10)
        row = []
        for s in (0.10, 0.125):
            row += [fit(y, s, "A") / d, fit(y, s, "M") / d,
                    fit(y, s, "S", mu) / d, fit(y, s, "S", 1.2 * mu) / d]
        print("%5.2f %6.2f | %6.3f %5.3f %5.3f %6.3f | %6.3f %5.3f %5.3f %6.3f" % (d, a, *row))
