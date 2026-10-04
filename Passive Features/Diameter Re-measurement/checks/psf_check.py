"""Numerical checks for the blur budget of the Allen 63x brightfield profiles.

Part A: Airy (paraxial) and scalar high-NA (Debye) in-focus PSF at NA 1.4:
        FWHM, first zero, line-spread function (LSF) FWHM and Gaussian sigma.
Part B: defocused scalar high-NA LSF at dz = 0.14 um (half the plane step).
Units: um. lambda is a free parameter (brightfield, broadband); 0.55 um used.
"""
import numpy as np
from scipy.special import j1, j0
from scipy.integrate import quad
from scipy.optimize import brentq, curve_fit

NA, N_OIL, LAM = 1.4, 1.515, 0.55


def airy_I(r):
    v = 2 * np.pi * NA * np.asarray(r, float) / LAM
    out = np.ones_like(v)
    nz = v != 0
    out[nz] = (2 * j1(v[nz]) / v[nz]) ** 2
    return out


def debye_I(r, dz=0.0, n_theta=4000):
    """Scalar Debye integral, aplanatic sqrt(cos) apodisation, intensity."""
    alpha = np.arcsin(NA / N_OIL)
    th = np.linspace(0, alpha, n_theta)
    k = 2 * np.pi / LAM
    r = np.atleast_1d(np.asarray(r, float))
    w = np.sqrt(np.cos(th)) * np.sin(th) * np.exp(1j * k * N_OIL * dz * np.cos(th))
    I = np.empty(r.size)
    for a in range(0, r.size, 200):
        rc = r[a:a + 200]
        A = np.trapezoid(w[None, :] * j0(k * N_OIL * rc[:, None] * np.sin(th)[None, :]), th, axis=1)
        I[a:a + 200] = np.abs(A) ** 2
    return I


def fwhm_of(fun, xmax):
    x = np.linspace(0, xmax, 200001)
    y = fun(x)
    y = y / y[0]
    i = np.argmax(y < 0.5)
    return 2 * np.interp(0.5, [y[i], y[i - 1]], [x[i], x[i - 1]])


def lsf_from_radial(I_radial, x, ymax=8.0, ny=8001):
    y = np.linspace(0, ymax, ny)
    out = []
    for xi in x:
        rr = np.sqrt(xi ** 2 + y ** 2)
        out.append(2 * np.trapezoid(I_radial(rr), y))
    out = np.array(out)
    return out / out[0]


def gauss(x, s):
    return np.exp(-x ** 2 / (2 * s ** 2))


print("=== Part A: in focus, lambda = %.2f um, NA = %.2f ===" % (LAM, NA))
f_airy = fwhm_of(airy_I, 1.0)
z1 = brentq(lambda r: j1(2 * np.pi * NA * r / LAM), 0.1, 0.4)
print("Airy PSF: FWHM = %.4f um (= %.4f lam/NA); first zero = %.4f um (= %.4f lam/NA)"
      % (f_airy, f_airy * NA / LAM, z1, z1 * NA / LAM))
print("  sigma from FWHM matching: %.4f um" % (f_airy / 2.3548))

x = np.linspace(0, 0.6, 301)
lsf_a = lsf_from_radial(airy_I, x)
f_lsf_a = 2 * np.interp(0.5, lsf_a[::-1], x[::-1])
core = x <= f_lsf_a  # fit the core out to ~1 FWHM
s_fit_a, _ = curve_fit(gauss, x[core], lsf_a[core], p0=[0.08])
print("Airy LSF: FWHM = %.4f um; sigma(FWHM-match) = %.4f; sigma(LSQ core fit) = %.4f"
      % (f_lsf_a, f_lsf_a / 2.3548, s_fit_a[0]))

I0 = debye_I(np.array([0.0]))[0]
rgrid = np.linspace(0, 8.0, 4001)
Igrid = debye_I(rgrid) / I0
rf = np.linspace(0, 0.3, 3001); If = debye_I(rf) / I0
i = np.argmax(If < 0.5); f_deb = 2 * np.interp(0.5, [If[i], If[i-1]], [rf[i], rf[i-1]])
print("Debye (high-NA scalar) PSF: FWHM = %.4f um" % f_deb)
deb_interp = lambda rr: np.interp(rr, rgrid, Igrid, right=0.0)
lsf_d = lsf_from_radial(deb_interp, x)
f_lsf_d = 2 * np.interp(0.5, lsf_d[::-1], x[::-1])
core = x <= f_lsf_d
s_fit_d, _ = curve_fit(gauss, x[core], lsf_d[core], p0=[0.08])
print("Debye LSF: FWHM = %.4f um; sigma(FWHM-match) = %.4f; sigma(LSQ core fit) = %.4f"
      % (f_lsf_d, f_lsf_d / 2.3548, s_fit_d[0]))

print("\n=== Part B: defocus, Debye LSF ===")
for dz in (0.0, 0.07, 0.14, 0.28):
    Ig = debye_I(rgrid, dz=dz)
    Ig = Ig / I0  # normalise to in-focus peak to keep energy scale
    f = lambda rr, Ig=Ig: np.interp(rr, rgrid, Ig, right=0.0)
    l = lsf_from_radial(f, x)
    l = l / l[0]
    fw = 2 * np.interp(0.5, l[::-1], x[::-1])
    c = x <= fw
    s, _ = curve_fit(gauss, x[c], l[c], p0=[0.08])
    print("dz = %.2f um: LSF FWHM = %.4f um, sigma(LSQ core) = %.4f um, sigma^2 excess over focus = %.5f um^2"
          % (dz, fw, s[0], s[0] ** 2 - s_fit_d[0] ** 2))

print("\nlambda scaling: sigma is proportional to lambda; at 0.45 / 0.65 um multiply by %.3f / %.3f"
      % (0.45 / LAM, 0.65 / LAM))
