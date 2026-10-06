"""Reach of faint thick calibration nodes beyond the +-3 planes of the thin ones, and the
tilt range a calibrated depth range covers (decision D-026, comments (c) and (h)).

Part 1 (comment (c); 0.5 um nodes for the ladder of comment (g), added later on 2026-10-06).
How many planes from focus a faint dendrite of diameter 0.8 um (and 0.5 um) keeps a centre
dip at least as deep as a 0.3 um dendrite has 3 planes (0.84 um) from focus.
Model: the partition renderer of procedure Eqs. 5-6 as implemented in
optics_points_check.py (circular Gaussian kernels; illustrative ideal-Debye core table,
continued linearly beyond 0.84 um with slope 0.79 or 1.21), flat tube (phi = 0),
half-length U = 6 um, output planes on the light side of the tube (z_k = -n * dz),
noise-free, no camera chain. The reference dip is computed with the 0.79 continuation.
These are mechanisms and orders of magnitude, not calibrated values: the run assumes a
far-field law, so it shows the gain of a thicker node, not the real usable range.

Part 2 (comment (h)). By (S5) of the implementation handoff, the farthest slab from the
plane through the node's axis lies at U tan(phi) + r / cos(phi). For U = 10 um (D-024),
the largest tilt whose reach stays within a calibrated depth range R, for R = 0.84 um
(thin nodes) and for the reaches found in part 1.

Part 3 (comment (b)). Two algebraic statements, checked numerically. (i) Faint flat round
tube of radius r, Gaussian kernel whose width is linear in depth across the tube,
sigma(delta) = sigma_A + g (delta - 0.84) for delta > 0.84 um: the variance across the
branch of the faint dip (a mixture over depth of box profiles, weights = chord lengths)
equals r^2/4 + sigma(delta_0)^2 + g^2 r^2/4, delta_0 = axis distance from the plane.
(ii) Under the same law, an axis-depth error s gives the same squared widths as the
anchor sigma_A replaced by sigma_A + g s.

Run from inside checks/ (imports optics_points_check.py from this folder; a few seconds):
    python calib_reach_check.py
Reference output: calib_reach_check.out (2026-10-06).
"""
import os
import sys

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from optics_points_check import DZ, render_partition_points, sigma_r  # noqa: E402

D_REF, MU_REF, N_REF = 0.3, 0.6, 3      # thin reference node: um, 1/um, planes from focus
THICK = ((0.8, 0.6), (0.8, 0.3), (0.5, 0.6), (0.5, 0.3))  # (d in um, mu in 1/um) of the thicker nodes
RULES = ("debye_0.79", "disc_1.21")     # continuation slopes 0.79 and 1.21 beyond 0.84 um
N_MAX = 30                              # planes scanned (8.4 um)
U_PHANTOM = 10.0                        # D-024 phantom half-length, um
R_THIN = 0.84                           # depth range of the thin-node calibration, um
COVER_D = (0.3, 0.5, 0.8)               # diameters for part 2, um


def centre_dip(d, mu, n_planes, rule):
    """1 - I/B at the tube centre in the plane n_planes * DZ from the axis, light side."""
    r = d / 2
    I = render_partition_points(r, 0.0, mu, -n_planes * DZ, rule, U=6.0,
                                v_eval=np.array([0.0]), hu=0.05, hv=r / 50, dzeta=0.01)
    return 1.0 - I[0]


def reach(d, mu, rule, ref):
    """Last plane index n in 0..N_MAX whose centre dip is >= ref, and all the dips."""
    dips = np.array([centre_dip(d, mu, n, rule) for n in range(N_MAX + 1)])
    ok = np.nonzero(dips >= ref)[0]
    return (int(ok.max()) if ok.size else -1), dips


def max_tilt_deg(R, d, U=U_PHANTOM):
    """Largest phi (deg) with U tan(phi) + r / cos(phi) <= R ((S5), plane through the axis)."""
    r = d / 2
    if r > R:
        return float("nan")
    return brentq(lambda p: U * np.tan(np.radians(p)) + r / np.cos(np.radians(p)) - R,
                  0.0, 89.9)


def faint_dip_variance(r, g, sigma_a, delta0, n=4001):
    """Variance across the branch of the faint dip of a flat round tube (part 3 (i)).

    Mixture over depth offsets w in (-r, r): each slab is a box of half-width
    sqrt(r^2 - w^2) (variance (r^2 - w^2) / 3), weighted by its chord length and blurred
    by a Gaussian of width sigma_a + g (delta0 + w - 0.84). All slabs are centred at v = 0.
    """
    w = np.linspace(-r, r, n)[1:-1]
    half = np.sqrt(r ** 2 - w ** 2)
    wt = half / half.sum()
    sig = sigma_a + g * (delta0 + w - 0.84)
    return float(np.sum(wt * (half ** 2 / 3 + sig ** 2)))


def part3():
    r, g, sigma_a = 0.4, 0.79, 0.603
    print("Part 3. Comment (b), faint flat round tube, linear kernel law:")
    for delta0 in (1.5, 3.0, 4.5):
        num = faint_dip_variance(r, g, sigma_a, delta0)
        pred = r ** 2 / 4 + (sigma_a + g * (delta0 - 0.84)) ** 2 + g ** 2 * r ** 2 / 4
        print(f"  (i) delta_0={delta0} um: mixture variance {num:.6f} um^2, "
              f"r^2/4 + sigma^2 + g^2 r^2/4 = {pred:.6f} um^2, difference {num - pred:.1e}")
    s = 0.07
    x = np.linspace(0.0, 3.6, 13)
    diff = (sigma_a + g * (x + s)) ** 2 - ((sigma_a + g * s) + g * x) ** 2
    print(f"  (ii) axis error s={s} um vs anchor shifted by g*s: max |difference| of squared "
          f"widths {np.abs(diff).max():.1e} um^2")


def main():
    ref = centre_dip(D_REF, MU_REF, N_REF, "debye_0.79")
    print(f"Part 1. Reference: d={D_REF} um, mu={MU_REF}/um, {N_REF} planes from focus: "
          f"centre dip {ref:.4f} B")
    reaches = []
    for rule in RULES:
        for d, mu in THICK:
            last, dips = reach(d, mu, rule, ref)
            dz = last * DZ
            reaches.append(dz)
            note = " (scan limit reached)" if last == N_MAX else ""
            print(f"  {rule}: d={d} mu={mu} (mu*d={mu * d:.2f}): in-focus dip {dips[0]:.4f}; "
                  f"dip >= reference out to plane {last}{note} (|delta| = {dz:.2f} um), "
                  f"sigma_r there {float(sigma_r(dz, rule)):.2f} um; dips at planes 3/6/10/15: "
                  + "/".join(f"{dips[k]:.4f}" for k in (3, 6, 10, 15)))
    r_list = [R_THIN] + sorted({round(x, 2) for x in reaches})
    print(f"Part 2. Largest tilt covered, U = {U_PHANTOM} um, plane through the node's axis:")
    for d in COVER_D:
        print(f"  d={d} um: " + ", ".join(f"R={R:.2f} um -> {max_tilt_deg(R, d):.1f} deg"
                                         for R in r_list))
    part3()


if __name__ == "__main__":
    main()
