"""optics_model.py -- scalar wave-optics model of TEEG's brightfield microscope.

Model level (analytic definitions); every function evaluates them numerically
(computed level). Physics only: no checks, no plotting (see
check_optics_claims.py and make_figures.py).

Conventions (TEEG renderers document 2026-10-09, sine-condition document
2026-10-12):
  lam   vacuum wavelength (um); n = n_oil, specimen index-matched to the oil;
  s     direction sines (s_x, s_y); spatial frequency nu = n s / lam (cycles/um);
  objective pupil: |nu| <= NA_o / lam (pupil uniform in direction sines);
  Koehler source: plane waves nu_c uniform on |nu_c| <= NA_c / lam (the sine
  document's W written in frequency); S = NA_c / NA_o.
"""
import numpy as np
from scipy import special


# ---------------------------------------------------------------- diffraction
def _gl(a, b, n):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (b - a) * (x + 1) + a, 0.5 * (b - a) * w


def psf_field(r, delta, cfg, n_gl=400, apod=None):
    """Scalar Debye amplitude U(r, delta) = iint_{|s|<=s_o} P(s) exp[i k n (s.x + sqrt(1-|s|^2) delta)] d^2s,
    azimuth integrated analytically; r array (um), delta scalar (um).
    apod: None (P = 1, pupil uniform in s) or a function of |s| giving P."""
    s_o = cfg["na_o"] / cfg["n"]
    rho, w = _gl(0.0, s_o, n_gl)
    kn = 2 * np.pi * cfg["n"] / cfg["lam"]
    p = np.ones_like(rho) if apod is None else apod(rho)
    wts = p * np.exp(1j * kn * np.sqrt(1 - rho**2) * delta) * 2 * np.pi * rho * w
    return special.j0(kn * np.outer(np.atleast_1d(r), rho)) @ wts


def psf_total_power(cfg):
    """iint |U|^2 d^2x for P = 1 (Parseval); the same at every delta."""
    s_o = cfg["na_o"] / cfg["n"]
    return np.pi * s_o**2 * (cfg["lam"] / cfg["n"]) ** 2


def airy_intensity(r, cfg):
    """[2 J1(beta)/beta]^2, beta = 2 pi NA_o r / lam."""
    b = 2 * np.pi * cfg["na_o"] * np.abs(np.asarray(r, float)) / cfg["lam"]
    out = np.ones_like(b)
    nz = b > 0
    out[nz] = (2 * special.j1(b[nz]) / b[nz]) ** 2
    return out


def encircled_energy(R, delta, cfg):
    """Fraction of the spot's power inside radius R (R increasing, starting at 0)."""
    I = np.abs(psf_field(R, delta, cfg)) ** 2
    f = I * 2 * np.pi * R
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(R))])
    return cum / psf_total_power(cfg)


def encircled_energy_airy(R, cfg):
    """Rayleigh's closed form for the in-focus uniform pupil: 1 - J0^2 - J1^2."""
    b = 2 * np.pi * cfg["na_o"] * np.asarray(R, float) / cfg["lam"]
    return 1 - special.j0(b) ** 2 - special.j1(b) ** 2


def encircled_energy_geometric(R, delta, cfg):
    """lam -> 0 limit of the same model: rays uniform in s land at |x| = |delta| tan(theta)."""
    s_o = cfg["na_o"] / cfg["n"]
    R = np.asarray(R, float)
    s = R / np.sqrt(delta**2 + R**2)
    return np.minimum(1.0, (s / s_o) ** 2)


def radius_at_fraction(R, ee, frac=0.5):
    """First R where the encircled energy reaches frac (linear interpolation)."""
    i = int(np.argmax(ee >= frac))
    return R[i - 1] + (frac - ee[i - 1]) * (R[i] - R[i - 1]) / (ee[i] - ee[i - 1])


# ------------------------------------------------------------------ coherence
def coherence_function(dx, na_c, lam):
    """Gamma(dx) = <exp(i 2 pi nu_c . dx)>_W, nu_c uniform on |nu_c| <= NA_c/lam
    (van Cittert-Zernike for Koehler illumination): 2 J1(beta_c)/beta_c."""
    b = 2 * np.pi * na_c * np.abs(np.asarray(dx, float)) / lam
    out = np.ones_like(b)
    nz = b > 0
    out[nz] = 2 * special.j1(b[nz]) / b[nz]
    return out


def source_quadrature(radius, n_rho, n_psi):
    """Nodes (nu_cx, nu_cy) and weights (sum 1) of the uniform measure on the disc
    |nu_c| <= radius: Gauss-Legendre in rho^2, equally spaced azimuths offset by
    half a step -- the construction of the project's directions() (sine doc 3.9)."""
    rho2, w = _gl(0.0, radius**2, n_rho)
    rho = np.sqrt(rho2)
    psi = (np.arange(n_psi) + 0.5) * 2 * np.pi / n_psi
    R, PSI = np.meshgrid(rho, psi, indexing="ij")
    W = np.repeat(w[:, None] / (radius**2 * n_psi), n_psi, axis=1)
    return (R * np.cos(PSI)).ravel(), (R * np.sin(PSI)).ravel(), W.ravel()


def lens_area(d, r1, r2):
    """Area of the intersection of two discs (radii r1, r2) whose centres are d apart."""
    d = np.atleast_1d(np.asarray(d, float))
    A = np.zeros_like(d)
    small, big = min(r1, r2), max(r1, r2)
    A[d <= big - small] = np.pi * small**2
    m = (d > big - small) & (d < r1 + r2)
    dd = d[m]
    t1 = r1**2 * np.arccos(np.clip((dd**2 + r1**2 - r2**2) / (2 * dd * r1), -1, 1))
    t2 = r2**2 * np.arccos(np.clip((dd**2 + r2**2 - r1**2) / (2 * dd * r2), -1, 1))
    t3 = 0.5 * np.sqrt(np.clip((-dd + r1 + r2) * (dd + r1 - r2) * (dd - r1 + r2) * (dd + r1 + r2), 0, None))
    A[m] = t1 + t2 - t3
    return A


def transfer_absorption(nu, S, cfg):
    """H_S(nu): first-order transfer of a weak absorber, in focus, uniform pupil,
    Koehler source of coherence parameter S: the fraction of the condenser disc
    (radius S NA_o/lam) that stays inside the pupil (radius NA_o/lam) when
    shifted by nu. S = 0: the pupil itself (coherent); S = 1: the incoherent OTF."""
    a_o = cfg["na_o"] / cfg["lam"]
    nu = np.abs(np.atleast_1d(np.asarray(nu, float)))
    if S == 0:
        return (nu <= a_o).astype(float)
    return lens_area(nu, S * a_o, a_o) / (np.pi * (S * a_o) ** 2)


def otf_closed_form(nu, cfg):
    """Incoherent OTF of a circular pupil: (2/pi)[arccos u - u sqrt(1-u^2)], u = nu/(2 NA_o/lam)."""
    u = np.clip(np.abs(np.asarray(nu, float)) / (2 * cfg["na_o"] / cfg["lam"]), 0, 1)
    return (2 / np.pi) * (np.arccos(u) - u * np.sqrt(1 - u**2))


# ------------------------------------------------ imaging of a straight branch
def profile_grid(cfg):
    """Across-branch coordinate v (um, centred) and FFT frequencies nu_x (cycles/um)."""
    N, L = cfg["n_v"], cfg["L_v"]
    v = (np.arange(N) - N // 2) * (L / N)
    return v, np.fft.fftfreq(N, d=L / N)


def tube_amplitude(v, d, mu, phase_peak=0.0):
    """Thin-object (projection) amplitude transmittance of a flat tube of diameter d:
    t(v) = exp(-mu l(v)/2 + i phase_peak l(v)/d), l(v) = vertical chord (renderers (S3))."""
    r = d / 2
    ell = 2 * np.sqrt(np.clip(r**2 - v**2, 0, None))
    return np.exp(-mu * ell / 2 + 1j * phase_peak * ell / d)


def _pupil_window(fx, fy2, a_o, kn, lam_n, delta):
    f2 = fx**2 + fy2
    inside = f2 <= a_o**2
    P = np.zeros(fx.shape, complex)
    P[inside] = np.exp(1j * kn * np.sqrt(1 - lam_n**2 * f2[inside]) * delta)
    return P


def abbe_image(t, nu, S, cfg, delta=0.0, n_rho=32, n_psi=64, batch=256):
    """I(v)/B of a y-invariant thin object t(v) under Koehler illumination:
    Abbe's sum over source points nu_c (uniform on |nu_c| <= S NA_o/lam) of the
    coherent images |F^-1[t~(nu_x) P(nu_x + nu_cx, nu_cy)]|^2; defocus delta (um)
    enters as the pupil phase k n sqrt(1 - (lam |f|/n)^2) delta."""
    a_o = cfg["na_o"] / cfg["lam"]
    kn = 2 * np.pi * cfg["n"] / cfg["lam"]
    lam_n = cfg["lam"] / cfg["n"]
    T_hat = np.fft.fft(np.fft.ifftshift(t))
    if S == 0:
        cx, cy, w = np.zeros(1), np.zeros(1), np.ones(1)
    else:
        cx, cy, w = source_quadrature(S * a_o, n_rho, n_psi)
    I = np.zeros(t.shape)
    for k in range(0, cx.size, batch):
        fx = nu[None, :] + cx[k:k + batch, None]
        fy2 = (cy[k:k + batch, None]) ** 2 * np.ones_like(fx)
        P = _pupil_window(fx, fy2, a_o, kn, lam_n, delta)
        U = np.fft.ifft(T_hat[None, :] * P, axis=1)
        I += (w[k:k + batch, None] * np.abs(U) ** 2).sum(axis=0)
    return np.fft.fftshift(I)


def incoherent_image(t, nu, cfg):
    """The renderers' incoherent form B (T * h_0)/B for a straight branch:
    T = |t|^2 blurred by the line-spread function, i.e. multiplied by the OTF along nu_x."""
    T_hat = np.fft.fft(np.fft.ifftshift(np.abs(t) ** 2))
    return np.real(np.fft.fftshift(np.fft.ifft(T_hat * otf_closed_form(nu, cfg))))


def darkfield_image(t, nu, cfg, cx_max=40.0, dcx=0.02, n_cy=48, batch=512):
    """Image formed by the illumination directions OUTSIDE the objective's cone,
    |nu_c| > NA_o/lam, at the S = 1 source density 1/(pi (NA_o/lam)^2). For a
    y-invariant object only |nu_cy| < NA_o/lam contributes; nu_cx is summed by the
    midpoint rule out to cx_max (truncation), both signs (mirror for a symmetric t)."""
    a_o = cfg["na_o"] / cfg["lam"]
    T_hat = np.fft.fft(np.fft.ifftshift(t))
    ys, wy = _gl(-a_o, a_o, n_cy)
    I = np.zeros(t.shape)
    for ncy, wyk in zip(ys, wy):
        b = np.sqrt(a_o**2 - ncy**2)
        cxs = np.arange(b + dcx / 2, cx_max, dcx)
        for k in range(0, cxs.size, batch):
            fx = nu[None, :] + cxs[k:k + batch, None]
            P = (fx**2 <= b**2).astype(float)
            U = np.fft.ifft(T_hat[None, :] * P, axis=1)
            I += wyk * dcx * (np.abs(U) ** 2).sum(axis=0)
    I = np.fft.fftshift(I)
    I = I + np.roll(I[::-1], 1)          # the nu_cx < 0 half, by the tube's symmetry
    return I / (np.pi * a_o**2)
