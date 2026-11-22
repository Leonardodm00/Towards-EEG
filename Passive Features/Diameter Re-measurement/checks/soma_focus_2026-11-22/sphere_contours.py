"""Check (2026-11-22): on a simulated brightfield stack of a sphere (radius 6 um), does the half-depth contour of each plane follow
the true cross-section radius sqrt(R^2 - z^2) away from the equator? Same image model as
checks/soma_focus_2026-11-22/check_detail_energy.py (no dendrite)."""
import sys
import numpy as np
sys.path.insert(0, sys.argv[1])
import check_detail_energy as C
from scipy.ndimage import gaussian_filter

c = dict(C.CFG)
B, X, Y, zs = C.stack(c, with_dendrite=False)
p = c["res_um"]
R = c["R_um"]
n = B.shape[1]
row = n // 2
print("plane z (um)  true r  half-depth r  (the contour's radius along +x, on the plane smoothed by 0.3 um)")
for j in range(0, len(zs), 3):
    S = gaussian_filter(B[j], 0.3 / p)
    prof = S[row, n // 2:]
    rho = np.arange(prof.size) * p
    lo = np.percentile(S[(X ** 2 + Y ** 2) <= (0.6 * R) ** 2], 10)
    hi = np.median(S[(X ** 2 + Y ** 2 >= (1.5 * R) ** 2) & (X ** 2 + Y ** 2 <= (1.8 * R) ** 2)])
    lev = 0.5 * (lo + hi)
    k = np.flatnonzero((prof >= lev) & (rho >= 0.5 * R))
    rc = rho[k[0]] if k.size else np.nan
    zt = zs[j]
    tr = np.sqrt(max(R * R - zt * zt, 0.0))
    print("%+6.2f       %5.2f   %5.2f   (contrast %.0f gl)" % (zt, tr, rc, hi - lo))
