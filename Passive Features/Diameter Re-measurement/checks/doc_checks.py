"""Checks behind the three documents of 2026-10-04 (b-table procedure, its maths, the optics).

C1  absorbed-light partition: sum_j dA_j = 1 - exp(-sum_j a_j) exactly (telescoping).
C2  single slab: the partition rendering equals B (T * g) of Eq. 11.
C3  faint stain: partition rendering -> linear sum of blurred absorbances.
C4  V additivity under convolution for non-Gaussian shapes (semicircle, box) and kernels
    (Gaussian, uniform, semicircle).
C5  windowed second moment of the scalar Debye LSF vs defocus, against the geometric
    semicircle law tan^2(theta) dz^2 / 4, and against the Gaussian-core sigma.
C6  acceptance half-angle in the specimen for several mounting-medium indices.
"""
import io, contextlib
import numpy as np
from scipy.ndimage import gaussian_filter1d

rng = np.random.default_rng(0)
print("=== C1 partition telescopes ===")
for _ in range(3):
    a = rng.uniform(0, 1.5, 12)
    Tprev = np.exp(-np.concatenate([[0], np.cumsum(a)[:-1]]))
    dA = Tprev * (1 - np.exp(-a))
    print("sum dA = %.12f   1-exp(-sum a) = %.12f" % (dA.sum(), 1 - np.exp(-a.sum())))

# 1-D cross-section test bed: v grid (um), tube of diameter d, column absorbance per slab
dv = 0.002
v = np.arange(-3, 3 + dv, dv)


def render_partition(a_slabs, sig_slabs, B=1.0):
    """a_slabs: (J, len(v)) absorbance of each slab along v; sig_slabs: blur sigma per slab."""
    Tprev = np.ones_like(v)
    out = np.zeros_like(v)
    for a_j, s_j in zip(a_slabs, sig_slabs):
        dA = Tprev * (1 - np.exp(-a_j))
        out += gaussian_filter1d(dA, s_j / dv, mode="constant") if s_j > 0 else dA
        Tprev = Tprev * np.exp(-a_j)
    return B * (1 - out)


print("\n=== C2 single slab equals B (T * g) ===")
d, mu, sig = 0.5, 3.0, 0.08
s_d = np.sqrt(np.clip(1 - (2 * v / d) ** 2, 0, None))
alpha = mu * d
T = np.exp(-alpha * s_d)
I_eq11 = 1.0 * gaussian_filter1d(T, sig / dv, mode="nearest")
I_part = render_partition([alpha * s_d], [sig])
print("max |difference| = %.2e" % np.max(np.abs(I_eq11 - I_part)))

print("\n=== C3 faint limit: partition vs linear sum, 9 slabs, different sigmas ===")
J = 9
for mu in (0.05, 0.6, 3.0):
    zeta = np.linspace(-d / 2, d / 2, J + 1); dz_s = np.diff(zeta); zc = 0.5 * (zeta[1:] + zeta[:-1])
    # chord of each slab at offset v: inside where v^2 + z^2 <= R^2
    a_slabs = np.array([mu * dz_j * ((v ** 2 + zc_j ** 2) <= (d / 2) ** 2) for dz_j, zc_j in zip(dz_s, zc)])
    sigs = 0.08 + 1.2 * np.abs(zc)
    I_part = render_partition(a_slabs, sigs)
    I_lin = 1 - sum(gaussian_filter1d(a_j, s / dv, mode="constant") for a_j, s in zip(a_slabs, sigs))
    dip_p, dip_l = 1 - I_part, 1 - I_lin
    print("mu = %.2f /um: centre-line absorbance %.3f; max dip partition %.4f, linear %.4f, rel diff %.1f%%"
          % (mu, mu * d, dip_p.max(), dip_l.max(), 100 * (dip_l.max() - dip_p.max()) / dip_p.max()))

print("\n=== C4 V adds for non-Gaussian shapes and kernels ===")


def V(w):
    A = w.sum(); m = (v * w).sum() / A
    return ((v - m) ** 2 * w).sum() / A


R = 0.25
shapes = {"semicircle": np.sqrt(np.clip(R ** 2 - v ** 2, 0, None)), "box": (np.abs(v) <= R).astype(float)}
kernels = {"gauss s=0.1": np.exp(-v ** 2 / (2 * 0.1 ** 2)),
           "uniform half-width 0.2": (np.abs(v) <= 0.2).astype(float),
           "semicircle R=0.3": np.sqrt(np.clip(0.3 ** 2 - v ** 2, 0, None))}
for sn, w in shapes.items():
    for kn, g in kernels.items():
        g = g / g.sum()
        c = np.convolve(w, g, mode="same")
        print("%-10s * %-22s V(w*g) = %.6f   V(w)+V(g) = %.6f" % (sn, kn, V(c), V(w) + V(g)))
print("semicircle V = R^2/4 = %.6f (d^2/16); box V = R^2/3 = %.6f (d^2/12)" % (R ** 2 / 4, R ** 2 / 3))

print("\n=== C5 Debye LSF: windowed second moment vs defocus ===")
with contextlib.redirect_stdout(io.StringIO()):
    exec(open("psf_check.py").read().split('print("=== Part A')[0])
theta = np.arcsin(NA / N_OIL)
print("theta = %.2f deg, tan(theta)/2 = %.3f" % (np.degrees(theta), np.tan(theta) / 2))
x = np.linspace(-3, 3, 601)
rgrid = np.linspace(0, 8.0, 4001)
for dz in (0.0, 0.14, 0.28, 0.42, 0.56, 0.84):
    Ig = debye_I(rgrid, dz=dz)
    f = lambda rr, Ig=Ig: np.interp(rr, rgrid, Ig, right=0.0)
    y = np.linspace(0, 8.0, 4001)
    lsf = np.array([2 * np.trapezoid(f(np.sqrt(xi ** 2 + y ** 2)), y) for xi in x])
    out = []
    for W in (1.5, 3.0):
        m = np.abs(x) <= W
        out.append(np.sqrt(np.sum(x[m] ** 2 * lsf[m]) / np.sum(lsf[m])))
    print("dz = %.2f um: sqrt(V) window +-1.5 = %.3f, +-3 = %.3f um; geometric tan(theta)|dz|/2 = %.3f um"
          % (dz, out[0], out[1], np.tan(theta) * dz / 2))

print("\n=== C6 acceptance half-angle in the specimen, NA = 1.4 ===")
for n_m in (1.33, 1.42, 1.45, 1.49, 1.515):
    if n_m > 1.4:
        th = np.degrees(np.arcsin(1.4 / n_m))
        print("n_mount = %.3f: theta_mount = %.1f deg, tan/2 = %.2f" % (n_m, th, np.tan(np.radians(th)) / 2))
    else:
        print("n_mount = %.3f: NA 1.4 > n_mount -> every propagating angle accepted; effective NA <= %.3f" % (n_m, n_m))
