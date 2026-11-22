"""Membrane areas of the soma shapes the brightfield data allow (2026-11-22): spheroids with the silhouette's
equivalent radius a and half-heights c from the stems' depth spread to a sphere, and the stacked-outline cylinder.
Closed form for the spheroid, checked against a numerical surface integral. Pure ASCII."""
import math
import numpy as np
from scipy.integrate import quad

A_ALLEN = 4 * math.pi * 5.9694 ** 2


def spheroid_area(a, c):
    """surface area of the spheroid with equatorial radius a and polar semi-axis c (um2)"""
    if abs(a - c) < 1e-12:
        return 4 * math.pi * a * a
    if c < a:                                   # oblate
        e = math.sqrt(1 - c * c / (a * a))
        return 2 * math.pi * a * a * (1 + (1 - e * e) / e * math.atanh(e))
    e = math.sqrt(1 - a * a / (c * c))          # prolate
    return 2 * math.pi * a * a * (1 + c / (a * e) * math.asin(e))


def spheroid_area_numeric(a, c):
    """2 pi int x(t) sqrt(x'^2 + z'^2) dt over the meridian x = a sin t, z = c cos t, t in [0, pi]"""
    f = lambda t: 2 * math.pi * a * math.sin(t) * math.hypot(a * math.cos(t), c * math.sin(t))
    return quad(f, 0, math.pi, epsabs=1e-10)[0]


rows = []
for a in (5.95, 6.2, 6.55):                     # the silhouette's equivalent radius, k = 111-121
    for c, why in ((3.45, "stems' spread 6.9 um"), (4.5, "+-4.5 um"), (a, "sphere")):
        S, Sn = spheroid_area(a, c), spheroid_area_numeric(a, c)
        assert abs(S - Sn) < 1e-6 * S, (a, c, S, Sn)
        rows.append((a, c, why, S, S / A_ALLEN))
cyl = 2 * math.pi * 6.5 * 7.28 + 2 * math.pi * 6.5 ** 2
print("Allen's sphere (r 5.9694): %.1f um2" % A_ALLEN)
print("  a (um)  c (um)  shape                 area (um2)  / Allen")
for a, c, why, S, q in rows:
    print("  %5.2f   %5.2f   %-20s  %7.1f     %.2f" % (a, c, why, S, q))
print("  cylinder r 6.5, height 7.28 (stacked outlines 107-133)  %7.1f     %.2f" % (cyl, cyl / A_ALLEN))
