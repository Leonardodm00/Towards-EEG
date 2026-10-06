"""Follow-up sandbox checks (2026-10-06): grid convergence of A2, why the partition
saturates for dark tubes (ray world), the focus-curve shift it causes, and the
in-focus centre dip of the partition renderer vs stain darkness (Debye kernel).
Reference output in followup_checks.out. Run from inside checks/ (imports
optics_points_check):
    python followup_checks.py
Note: the 3-point parabola vertex of section 3 is meaningless when F(-1,0,+1) is
not peaked at 0 (d = 2, mu*d = 2 prints +212 um); focus_scan.py scans +-6 planes."""
import numpy as np
from optics_points_check import (render_partition_points, fit_profile, V_PROF, DZ,
                                 directions, ray_models)
from scipy import ndimage

# 1. convergence of A2 (halve grid and slab thickness)
for d, ph, (hu, hv, dz) in ((0.5, 45, (0.01, 0.005, 0.01)), (2.0, 20, (0.02, 0.01, 0.02))):
    r, phi = d / 2, np.radians(ph)
    I = render_partition_points(r, phi, 1.0, 0.0, "debye_0.79", hu=hu, hv=hv, dzeta=dz)
    p, _ = fit_profile(I, V_PROF, phi, 0.080, d, 1.0)
    print(f"convergence d={d} phi={ph} (0.79 rule, halved grid): d^/d {p[0] / d:.3f}, dip {1 - I[26]:.3f}")

# 2. ray world, flat d=1 tube, node plane, v=0: share of rays that enter the tube through
#    its upper half (entry point above the axis) -- the rays the partition lets through.
#    This first hypothesis is REFUTED (it prints 0.000); the leak mechanism is the
#    skin-crossing formula (S9) of the implementation handoff, checked in focus_scan.py.
S, W = directions()
Wst = W / S[:, 2]; Wst /= Wst.sum()
r = 0.5
P = np.array([0.0, 0.0, 0.0])
from optics_points_check import chord_interval
t1, t2, good = chord_interval(P, S, r, 0.0, 6.0)
z_entry = P[2] + t1 * S[:, 2]
print(f"ray world, d=1, node plane, centre point: share of ray weight entering above the axis "
      f"(W) {np.sum(W[good & (z_entry > 0)]):.3f}")
for mu in (1.5, 3.0, 10.0, 50.0):
    L_, P_, Gv_, G_ = ray_models(0.0, 0.0, r, 0.0, mu, S, Wst)
    print(f"   mu*d={mu * 1.0:5.1f}: centre dip, partition P* {1 - P_:.3f}, true history Gv* {1 - Gv_:.3f}")

# 3. focus curve of a dark flat tube in the renderer (Debye kernel, 0.79 continuation):
#    F_k = -ln(min smoothed profile / B) at planes -1, 0, +1 (axis on plane 0); parabola vertex (handoff Eq. 2)
for d, mu in ((0.5, 1.0), (1.0, 1.5), (2.0, 1.0), (2.0, 0.1)):
    r = d / 2
    F = []
    for k in (-1, 0, 1):
        I = render_partition_points(r, 0.0, mu, k * DZ, "debye_0.79", U=6.0, hu=0.05,
                                    hv=r / 40, dzeta=0.01, v_eval=V_PROF)
        Is = ndimage.gaussian_filter1d(I, 1.0, mode="nearest")
        F.append(-np.log(Is.min()))
    Fm, F0, Fp = F
    zv = DZ * (Fm - Fp) / (2 * (Fm - 2 * F0 + Fp))
    print(f"focus curve d={d} mu*d={mu * d:.2f}: F(-1,0,+1) = {Fm:.3f}, {F0:.3f}, {Fp:.3f}; "
          f"sub-plane depth {zv:+.3f} um (negative = toward the light)")

# 4. in-focus centre dip of the partition renderer (Debye kernel, 0.79 continuation), flat
#    tubes, plane through the axis, v = 0, vs mu*d; "column" = 1 - exp(-mu*d), what a
#    vertical ray through the centre absorbs. (Inline run of 2026-10-06, committed here.)
for d in (0.5, 1.0, 2.0):
    r = d / 2
    out = []
    for mu_d in (0.5, 1.5, 3.0, 10.0, 50.0):
        I = render_partition_points(r, 0.0, mu_d / d, 0.0, "debye_0.79", U=6.0, hu=0.05,
                                    hv=r / 60, dzeta=min(0.01, 0.2 / (mu_d / d)),
                                    v_eval=np.array([0.0]))
        out.append((mu_d, 1 - I[0], 1 - np.exp(-mu_d)))
    print(f"renderer (Debye kernel), flat d={d}, axis plane, centre: " +
          "; ".join(f"mu*d={m:g}: dip {x:.3f} (column {c:.3f})" for m, x, c in out))
