"""Focus curve of flat tubes in the partition renderer over +-6 planes (Debye kernel,
0.79 continuation; light travels toward +z, so negative planes are on the light side),
and the opaque-limit leak of the partition predicted by the skin-crossing formula (S9)
of docs/TEEG_diameter_implementation_handoff_2026-10-06.md (compare with the
mu*d = 50 partition dip P* printed by followup_checks.py). Reference output in
focus_scan.out. Run from inside checks/ (imports optics_points_check):
    python focus_scan.py
"""
import numpy as np
from scipy import ndimage
from optics_points_check import render_partition_points, V_PROF, DZ, directions, chord_interval

# renderer focus curve over +-6 planes (axis on plane 0), Debye kernel with 0.79 continuation
for d, mu in ((1.0, 1.5), (2.0, 1.0), (2.0, 0.1)):
    r = d / 2
    F = []
    ks = np.arange(-6, 7)
    for k in ks:
        I = render_partition_points(r, 0.0, mu, k * DZ, "debye_0.79", U=6.0, hu=0.05,
                                    hv=r / 40, dzeta=0.01, v_eval=V_PROF)
        F.append(-np.log(ndimage.gaussian_filter1d(I, 1.0, mode="nearest").min()))
    F = np.array(F)
    print(f"d={d} mu*d={mu*d:.1f}: argmax plane {ks[np.argmax(F)]:+d}; F = " +
          " ".join(f"{x:.3f}" for x in F))

# analytic leak check (ray world, opaque flat tube, node plane, centre point):
# partition absorbed fraction of a ray entering at v_e = 1/(1 + |dz_lo/dv| * |s_v|/s_z)
S, W = directions()
Wst = W / S[:, 2]; Wst /= Wst.sum()
r = 0.5
t1, t2, good = chord_interval(np.zeros(3), S, r, 0.0, 6.0)
v_e = t1 * S[:, 1]
slope = np.abs(v_e) / np.sqrt(np.clip(r**2 - v_e**2, 1e-12, None))
absorbed = 1.0 / (1.0 + slope * np.abs(S[:, 1]) / S[:, 2])
print(f"opaque limit, centre dip predicted by the skin-crossing formula: {np.sum(Wst * absorbed):.3f}")
