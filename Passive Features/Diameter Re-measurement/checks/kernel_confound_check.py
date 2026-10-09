"""Partition with the production-like Gaussian kernel vs the ray world (2026-10-09).

NOT pipeline code. Theory chat check for the D-024 (i) study (theory handoff,
Open work 1): what a comparison of the two renderers measures when the
partition keeps the Gaussian Debye-table kernel instead of the kernel of the
ray world's own rays ((S7)/(S8) of the implementation handoff). Reference
output in kernel_confound_check.out. Run from inside checks/ (imports
optics_points_check, like focus_scan.py):
    python kernel_confound_check.py      # about a minute

Cases: those of optics_points_check.py section BC (d = 1 um flat at four
darknesses; d = 0.5 um at 20 deg), plus thick flat tubes (d = 2 and 3 um at
mu d = 0.5 and 3; added 2026-10-09 later, for the user's question whether the
ray world could build the table for thick or dark dendrites; partition grid as
section A2 uses for d >= 1, which its halved-grid run showed converged at
d = 2 um). Same fit for both renderers (handoff Eq. 11
as in D-019.1, sigma_fit = 0.080 um, Bbar = B), same profile samples, same
true mu (no mu matching). Partition: procedure Eqs. 5-6 with the ideal-Debye
core table and the 0.79 continuation, U = 6 um. Ray world: (S6), the G column
of optics_points_check.ray_models, post-blurred by 0.08 um as in section BC.
Compare with optics_points_check.py `matched`, where the partition gets the
faint-limit kernel of the same rays and mu is matched through mu_hat.
"""
import numpy as np

from optics_points_check import V_PROF, directions, fit_profile, ray_profiles, render_partition_points

S, W = directions()
v_fine = np.arange(-3.5, 3.5001, 0.01)
print("fit: sigma_fit 0.080 um, Bbar = B; node plane; same true mu for both renderers")
print("case                     partition (Gaussian Debye kernel) | ray world G       | gap")
print("                         d^/d    mu^/mu                    | d^/d    mu^/mu    | partition - ray")
CASES = ((1.0, 0, 0.05), (1.0, 0, 0.5), (1.0, 0, 1.5), (1.0, 0, 3.0), (0.5, 20, 0.5),
         (2.0, 0, 0.5), (2.0, 0, 3.0), (3.0, 0, 0.5), (3.0, 0, 3.0))
for d, ph_deg, mud in CASES:
    r, phi, mu = d / 2, np.radians(ph_deg), mud / d
    hu, hv, dz = (0.02, 0.01, 0.02) if d <= 1.0 else (0.04, 0.02, 0.04)
    I_P = render_partition_points(r, phi, mu, 0.0, "debye_0.79", U=6.0, hu=hu, hv=hv, dzeta=dz)
    pP, _ = fit_profile(I_P, V_PROF, phi, 0.080, d, mu)
    prof, _ = ray_profiles(r, phi, mu, 0.0, S, W, v_fine)
    pG, _ = fit_profile(prof["G"], V_PROF, phi, 0.080, d, mu)
    print(f"d={d} phi={ph_deg:2d} mu*d={mud:4.2f}     {pP[0] / d:6.3f}  {pP[1] / mu:6.3f}"
          f"                    | {pG[0] / d:6.3f}  {pG[1] / mu:6.3f}    | {pP[0] / d - pG[0] / d:+.3f}")
