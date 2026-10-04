"""Half-depth width (FWHM of the dip) of the Eq. 11 profile vs true d, darkness alpha, blur sigma."""
import numpy as np
from scipy.ndimage import gaussian_filter1d
dx = 0.0005
v = np.arange(-4, 4, dx)
def profile(d, a, s):
    u = np.clip(1 - (2 * v / d) ** 2, 0, None)
    T = np.exp(-a * np.sqrt(u))
    return gaussian_filter1d(T, s / dx, mode="nearest") if s > 0 else T
def fwhm(y):
    lvl = (1 + y.min()) / 2           # half depth below background (B = 1)
    idx = np.where(y <= lvl)[0]
    return (idx[-1] - idx[0]) * dx
print("FWHM / d   (sigma in um)")
print("%6s %6s | %8s %8s %8s" % ("d", "alpha", "s=0", "s=0.08", "s=0.10"))
for d in (0.3, 0.5, 1.0, 2.0):
    for a in (0.1, 1.0, 3.0):
        print("%6.2f %6.1f | %8.3f %8.3f %8.3f" % (d, a, *[fwhm(profile(d, a, s)) / d for s in (0, 0.08, 0.10)]))
