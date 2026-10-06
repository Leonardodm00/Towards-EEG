"""Width statistics of calibration profiles (decision D-026 comment (a); decision D-027).

Part 1 (D-026 comment (a)): why the windowed second moment of a dip depends on the window W.
  (i) A toy line-spread function with a Gaussian core (sigma 0.08 um) continued by a |v|^-2
      tail from |v| = 0.25 um on (continuous there): sqrt(V_W) for W = 0.75 ... 6 um. The full
      second moment is infinite; V_W grows roughly in proportion to W.
  (ii) A Gaussian dip of width sigma = sqrt(0.080^2 + d^2/16) (d = 0.3 um, faint) and area
      A = mu pi d^2 / 4 (mu = 0.6/um, units of B * um), plus a constant background offset eps
      (units of B) over |v| <= W:  V_W = (sigma^2 A + 2 eps W^3 / 3) / (A + 2 eps W).
Part 2 (D-027): a thin node (d = 0.3 um, mu = 0.6/um) tilted by phi = 5, 10, 20, 30 deg, profile
  across the branch at the node, planes 1, 2, 3, 6, 9 on the light side. Depth-only Gaussian
  kernel: the partition renderer of optics_points_check.py, ideal-Debye core table continued
  with slope 0.79, U = 8 um, noise-free, no camera chain. Ratio tilted / flat of two widths:
  the square root of the second moment over |v| <= 5 sigma_r(|delta|) (the flat value), and
  the Gaussian-core width (Gaussian fitted to the samples at or above half maximum).
Part 3 (both): the profile of part 2 at plane 1 and 30 deg, measured over two windows,
  |v| <= 5 sigma_r (0.61 um) and |v| <= 1.0 um: the second-moment ratio moves with the window,
  the core width does not.
These are model numbers (orders of magnitude), not calibrated values.

Run from inside checks/ (imports optics_points_check.py from this folder; seconds):
    python calib_width_check.py
Reference output: calib_width_check.out (2026-10-06).
"""
import os
import sys

import numpy as np
from scipy.optimize import curve_fit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from optics_points_check import DZ, render_partition_points, sigma_r  # noqa: E402

D_THIN, MU_THIN = 0.3, 0.6          # thin calibration node: um, 1/um
PLANES = (1, 2, 3, 6, 9)            # plane offsets from the axis, light side
TILTS_DEG = (5, 10, 20, 30)
RULE = "debye_0.79"                 # continuation slope 0.79 beyond 0.84 um


def toy_lsf(x, s0=0.08, a=0.25):
    """Gaussian core of width s0 for |x| <= a, kappa / x^2 beyond (continuous at a)."""
    kappa = np.exp(-a ** 2 / (2 * s0 ** 2)) * a ** 2
    return np.where(np.abs(x) <= a, np.exp(-x ** 2 / (2 * s0 ** 2)),
                    kappa / np.maximum(x ** 2, a ** 2))


def windowed_sd(x, y):
    """sqrt of the second moment of y about 0 over the samples x (equal spacing)."""
    return float(np.sqrt(np.sum(x ** 2 * y) / np.sum(y)))


def dip_profile(n_plane, phi_deg, d=D_THIN, mu=MU_THIN, U=8.0, half_width=None):
    """v samples (um) and dip 1 - I/B across the branch at the node, plane n_plane * DZ below.
    The samples span |v| <= half_width, by default 5 sigma_r(n_plane * DZ)."""
    if half_width is None:
        half_width = 5 * float(sigma_r(n_plane * DZ, RULE))
    v = np.linspace(-half_width, half_width, 241)
    r = d / 2
    I = render_partition_points(r, np.radians(phi_deg), mu, -n_plane * DZ, RULE, U=U,
                                v_eval=v, hu=0.04, hv=r / 25, dzeta=0.01)
    return v, 1.0 - I


def core_width(v, D):
    """Width of a zero-centred Gaussian fitted to the samples at or above half maximum."""
    keep = D >= 0.5 * D.max()
    (_, s), _ = curve_fit(lambda x, a, w: a * np.exp(-x ** 2 / (2 * w ** 2)),
                          v[keep], D[keep], p0=(D.max(), windowed_sd(v, D)))
    return abs(float(s))


def part1():
    print("Part 1. Windowed second moment")
    for W in (0.75, 1.5, 3.0, 6.0):
        x = np.linspace(-W, W, 200001)
        print(f"  (i) toy core + |v|^-2 tail, W = {W} um: sqrt(V_W) = {windowed_sd(x, toy_lsf(x)):.3f} um")
    A = MU_THIN * np.pi * D_THIN ** 2 / 4
    sig = np.sqrt(0.080 ** 2 + D_THIN ** 2 / 16)
    for eps in (0.0, 0.0025, 0.005):
        for W in (1.5, 3.0):
            v_w = (sig ** 2 * A + eps * 2 * W ** 3 / 3) / (A + eps * 2 * W)
            print(f"  (ii) Gaussian dip sigma {sig:.3f} um, area {A:.4f} B*um, offset {eps:.4f} B, "
                  f"W = {W} um: sqrt(V_W) = {np.sqrt(v_w):.3f} um")


def part2():
    print("Part 2. Tilted thin node: width ratio tilted / flat (sqrt of 2nd moment ; core width)")
    for n in PLANES:
        v0, D0 = dip_profile(n, 0.0)
        m0, c0 = windowed_sd(v0, D0), core_width(v0, D0)
        cells = []
        for ph in TILTS_DEG:
            v, D = dip_profile(n, ph)
            cells.append(f"{ph:2d} deg: {windowed_sd(v, D) / m0:.3f} ; {core_width(v, D) / c0:.3f}")
        print(f"  plane {n} (|delta| = {n * DZ:.2f} um; flat: sqrt(m2) {m0:.3f} um, core {c0:.3f} um): "
              + " | ".join(cells))


def part3():
    print("Part 3. Same profile, two windows (plane 1, 30 deg vs flat)")
    for hw in (None, 1.0):
        v0, D0 = dip_profile(1, 0.0, half_width=hw)
        v, D = dip_profile(1, 30.0, half_width=hw)
        print(f"  window |v| <= {v0[-1]:.2f} um: sqrt(m2) ratio {windowed_sd(v, D) / windowed_sd(v0, D0):.3f}; "
              f"core ratio {core_width(v, D) / core_width(v0, D0):.3f}")


if __name__ == "__main__":
    part1()
    part2()
    part3()
