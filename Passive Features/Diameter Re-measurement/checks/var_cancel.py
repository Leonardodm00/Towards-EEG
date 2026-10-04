"""Does subtracting the in-focus width cancel the tube's own width?
Dip = (1 - exp(-alpha s_d)) * g_sigma (exact for the 2-D incoherent model: 1 - T*g = (1-T)*g).
Width measured two ways: raw second moment over +-3 um, and a fitted Gaussian.
Report: [w2(sigma) - w2(sigma0)] / [sigma^2 - sigma0^2]  (1.000 = perfect cancellation)."""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
dx = 0.001; v = np.arange(-3, 3, dx)
def dip(d, a, s):
    u = np.clip(1 - (2 * v / d) ** 2, 0, None)
    return gaussian_filter1d(1 - np.exp(-a * np.sqrt(u)), s / dx, mode="constant")
def m2(y): return np.sum(v**2 * y) / np.sum(y)
def g2(y):
    p, _ = curve_fit(lambda x, A, s: A * np.exp(-x**2 / (2*s**2)), v, y, p0=[y.max(), 0.15])
    return p[1]**2
s0 = 0.10
print("in-focus check: m2 - s0^2 vs d^2/16 (faint) / d^2/12 (opaque)")
for d in (0.3, 0.5):
    for a in (0.1, 5.0):
        print("  d=%.1f a=%.1f: m2-s0^2=%.5f  d2/16=%.5f d2/12=%.5f" % (d, a, m2(dip(d,a,s0))-s0**2, d*d/16, d*d/12))
print("\nratio of width increments to true sigma^2 increment, sigma0=0.10")
print("%5s %5s %6s | %8s %8s" % ("d", "alpha", "sigma", "moment", "gaussfit"))
for d in (0.3, 0.5, 1.0):
    for a in (0.1, 2.0):
        y0 = dip(d, a, s0)
        for s in (0.15, 0.25, 0.4):
            y = dip(d, a, s)
            print("%5.1f %5.1f %6.2f | %8.3f %8.3f" % (d, a, s, (m2(y)-m2(y0))/(s*s-s0*s0), (g2(y)-g2(y0))/(s*s-s0*s0)))
