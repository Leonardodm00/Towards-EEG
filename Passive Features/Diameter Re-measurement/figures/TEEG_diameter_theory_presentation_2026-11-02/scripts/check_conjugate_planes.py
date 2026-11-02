"""check_conjugate_planes.py -- does an aplanatic NA 1.4 objective image a
DEFOCUSED axial point to a point anywhere? (TEEG, 2026-11-02)

Model (model level): ideal objective obeying the sine condition for its focal
plane (aplanatic), object space oil (n = n_oil), image space air (n' = 1),
lateral magnification m; index-matched specimen; scalar optics.

Claims tested; tolerances fixed before running:
  G1 (rays) a ray leaving the axial point at defocus delta with angle theta
     crosses the focal plane at x_f = delta tan(theta); the objective images
     x_f to y = m x_f (aplanatic: stigmatic near the axis) and, by the sine
     condition, sends it out at sin(theta') = n sin(theta)/m. Its axial crossing
     in image space then lies dz' = y / tan(theta') from the camera plane, which
     is (m^2/n) delta / cos(theta) up to the small-angle error of theta'
     (relative error < 1e-3 at m = 63). So the crossings spread by the factor
     1/cos(theta_o) = 2.62 at NA 1.4: no single conjugate point.
  W1 (waves) refocusing the camera adds a phase proportional to sin^2(theta)
     (image space is paraxial at m = 63); the defocus phase is
     k n delta (1 - cos theta). Best peak over all camera refocus values:
     NA 1.4, delta = 0.3 um: < 0.95 (refocusing cannot restore the point);
     NA 0.1 at the same edge phase: > 0.999 (paraxial: a conjugate point exists).
Added after the first run, because W1 FAILED there (refocusing restored 0.990 at
0.3 um: the claim is corrected, not the tolerance); tolerance fixed before W1b ran:
  W1b at NA 1.4 the best-refocus peak stays >= 0.95 for |delta| <= 0.6 um, and the
     defocus at which it first falls below 0.8 lies in (0.6, 5] um.
No random numbers. Run: python3 check_conjugate_planes.py
"""
import numpy as np
from scipy import optimize

CFG = dict(lam=0.55, n=1.515, na=1.4, m=63.0, n_gl=600)


def crossing_ratio(theta, cfg):
    """dz'(theta)/dz'(0) from the exact ray geometry, and the closed form 1/cos(theta)."""
    n, m, delta = cfg["n"], cfg["m"], 1.0
    y = m * delta * np.tan(theta)
    th_p = np.arcsin(n * np.sin(theta) / m)
    dz = y / np.tan(th_p)
    dz0 = (m**2 / n) * delta
    return dz / dz0, 1 / np.cos(theta)


def best_refocus_peak(delta, na, cfg):
    """max over c of |< exp(i[k n delta (1 - sqrt(1-s^2)) - c s^2]) >|^2,
    average over a pupil uniform in direction sines, |s| <= na/n."""
    s_o = na / cfg["n"]
    x, w = np.polynomial.legendre.leggauss(cfg["n_gl"])
    s2 = 0.5 * s_o**2 * (x + 1)                  # uniform in |s|^2 <=> uniform on the disc
    w = 0.5 * w
    kn = 2 * np.pi * cfg["n"] / cfg["lam"]
    phi = kn * delta * (1 - np.sqrt(1 - s2))
    peak = lambda c: abs(np.sum(w * np.exp(1j * (phi - c * s2))) / np.sum(w)) ** 2
    c0 = phi[-1] / s2[-1]
    res = optimize.minimize_scalar(lambda c: -peak(c), bounds=(0, 3 * c0), method="bounded",
                                   options=dict(xatol=1e-10))
    return peak(res.x), peak(0.0)


def main(cfg):
    th_o = np.arcsin(cfg["na"] / cfg["n"])
    th = np.linspace(1e-4, th_o, 2001)
    exact, closed = crossing_ratio(th, cfg)
    g1 = np.max(np.abs(exact / closed - 1)) < 1e-3
    print(f"G1: max |exact/closed - 1| = {np.max(np.abs(exact / closed - 1)):.2e}; "
          f"crossing spread at the edge = {exact[-1]:.3f} (1/cos theta_o = {1 / np.cos(th_o):.3f})  "
          f"{'PASS' if g1 else 'FAIL'}")

    kn = 2 * np.pi * cfg["n"] / cfg["lam"]
    edge_phase = kn * 0.3 * (1 - np.cos(th_o))
    p14, p14_0 = best_refocus_peak(0.3, 1.4, cfg)
    th_01 = np.arcsin(0.1 / cfg["n"])
    d01 = edge_phase / (kn * (1 - np.cos(th_01)))
    p01, p01_0 = best_refocus_peak(d01, 0.1, cfg)
    w1 = p14 < 0.95 and p01 > 0.999
    print(f"W1: edge phase {edge_phase:.2f} rad. NA 1.4, delta 0.3 um: peak {p14_0:.3f} unrefocused, "
          f"{p14:.3f} at the best camera refocus. NA 0.1, delta {d01:.1f} um (same edge phase): "
          f"{p01_0:.3f} -> {p01:.4f}  {'PASS' if w1 else 'FAIL'}")
    ds = np.round(np.arange(0.05, 5.0001, 0.05), 3)
    pk = np.array([best_refocus_peak(d, 1.4, cfg)[0] for d in ds])
    below = ds[pk < 0.8]
    d80 = below[0] if below.size else np.inf
    w1b = bool(np.all(pk[ds <= 0.6] >= 0.95)) and 0.6 < d80 <= 5.0
    for d in (0.15, 0.3, 0.6, 1.0):
        print(f"    NA 1.4, delta {d} um: best-refocus peak {pk[np.argmin(abs(ds - d))]:.3f}")
    print(f"W1b: peak >= 0.95 up to 0.6 um: {bool(np.all(pk[ds <= 0.6] >= 0.95))}; first below 0.8 at "
          f"delta = {d80} um  {'PASS' if w1b else 'FAIL'}")
    print("ALL PASS" if (g1 and w1 and w1b) else "FAILED: W1 is the original claim; W1b tests the corrected one"
          if (g1 and w1b) else "SOME FAILED")


if __name__ == "__main__":
    main(CFG)
