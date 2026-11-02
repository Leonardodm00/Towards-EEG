"""theory_figs_model.py -- the mathematics behind Figures 7-11 of the diameter theory pack (TEEG, 2026-11-02).

Pure functions and documented constants; no plotting (make_theory_figures.py) and no checks
(check_theory_figures.py). Every formula names the repository location it reproduces.
Repository: Leonardodm00/Towards-EEG, folder "Passive Features/Diameter Re-measurement" (= DR below).
Units: um for lengths, um^-1 for mu, radians unless a name ends in _deg.
"""
import numpy as np
from scipy.special import beta as beta_fn, gamma as gamma_fn, iv, modstruve
from scipy.optimize import brentq

# --------------------------------------------------------------------------- T1: frames and the measuring line
def chord_ratio(eps, phi):
    """Chord / d of a line through the node centre misaligned by eps from the measuring axis y_hat, for a
    straight round tube tilted by phi out of the image plane.

    DR/docs/handoff_diameter_remeasurement.md (main), Math reference Eq. 8: a line at angle alpha to e_u cuts
    the in-plane ellipse (u sin phi)^2 + v^2 <= r^2 (Eq. 6) with half-length r / sqrt(cos^2 alpha sin^2 phi
    + sin^2 alpha). With alpha = 90 deg - eps the ratio chord/d is 1 / sqrt(sin^2 eps sin^2 phi + cos^2 eps):
    1 at eps = 0 for every phi (Eq. 7, width invariance), 1/cos(eps) for a flat branch (phi = 0)."""
    eps, phi = np.asarray(eps, float), np.asarray(phi, float)
    return 1.0 / np.sqrt(np.sin(eps) ** 2 * np.sin(phi) ** 2 + np.cos(eps) ** 2)


def ellipse_section(r, phi, n=400):
    """Boundary (u, v) of the focal-plane section (u sin phi)^2 + v^2 = r^2 (handoff Eq. 6), phi in (0, pi/2]."""
    t = np.linspace(0, 2 * np.pi, n)
    return r / np.sin(phi) * np.cos(t), r * np.sin(t)


# --------------------------------------------------------------------------- T2: the blurred absorbing tube
def s_d(v, d):
    """Normalised chord sqrt(1 - (2v/d)^2) inside |v| <= d/2, 0 outside (mathematics Eq. 8: l(v) = d s_d(v)/cos phi)."""
    x = 1.0 - (2.0 * np.asarray(v, float) / d) ** 2
    return np.sqrt(np.clip(x, 0.0, None))


def transmission(v, d, alpha):
    """T(v) = exp(-alpha s_d(v)), alpha = mu d / cos phi (SPEC (3); D-019.1)."""
    return np.exp(-alpha * s_d(v, d))


def gauss_kernel(z, sigma):
    return np.exp(-0.5 * (z / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)


def tube_profile(v, d, alpha, kernel, v0=0.0, n_psi=600):
    """I_model / B_bar of SPEC (B3.1) / handoff-10-16 (H4) for any unit-area kernel callable:
    1 - (d/2) int_{-pi/2}^{pi/2} (1 - exp(-alpha cos psi)) cos psi kernel(v - v0 - (d/2) sin psi) dpsi,
    by Gauss-Legendre in psi (the substitution u = (d/2) sin psi)."""
    x, w = np.polynomial.legendre.leggauss(n_psi)
    psi, w = x * np.pi / 2, w * np.pi / 2
    core = (1.0 - np.exp(-alpha * np.cos(psi))) * np.cos(psi) * w
    v = np.atleast_1d(np.asarray(v, float))
    K = kernel(v[:, None] - v0 - 0.5 * d * np.sin(psi)[None, :])
    return 1.0 - 0.5 * d * (K * core[None, :]).sum(axis=1)


def dip_area_exact(d, alpha, precise=True):
    """int (1 - T) dv = (pi d / 2)(I_1(alpha) - L_1(alpha)) (SPEC Block 3, the exact dip area of (B3.1)).
    I_1 and L_1 both grow like e^alpha while their difference stays O(1), so in double precision the
    difference loses about log10(I_1) digits (7e-5 relative at alpha = 30, check P2): precise=True evaluates
    it with mpmath at 40 digits; precise=False is the plain scipy form, kept for check P2."""
    if precise:
        import mpmath as mp
        mp.mp.dps = 40
        return float(mp.pi / 2 * d * (mp.besseli(1, alpha) - mp.struvel(1, alpha)))
    return 0.5 * np.pi * d * (iv(1, alpha) - modstruve(1, alpha))


def v_tube_bounds(d):
    """Normalised squared width of the unblurred dip: d^2/16 (faint, a semicircle) to d^2/12 (opaque, a box)
    (mathematics s.3.3, l.460-466)."""
    d = np.asarray(d, float)
    return d ** 2 / 16.0, d ** 2 / 12.0


SIGMA_FIT_UM = 0.099      # config.py MeasureConfig: the Gaussian width of the pilot fit and the SGT starts (PROVISIONAL)
PROFILE_HALF_UM, PROFILE_N = 3.0, 53     # SPEC 2.2 l.114-116: 53 bilinear samples over +-3 um


# --------------------------------------------------------------------------- T5: the skewed generalized-t family
SGT_XI_ZERO = 1e-12   # as model/blur_kernels.py: xi at or below this is the xi = 0 form of (H1)


def sgt_core(x, beta, xi):
    """(H1) of the 10-16 handoff s.3.2 (= SPEC Block 12, D-052.1)."""
    x = np.abs(np.asarray(x, float))
    if xi > SGT_XI_ZERO:
        c = beta * xi ** (1.0 / beta) / (2.0 * beta_fn(1.0 / beta, 1.0 / xi))
        return c * (1.0 + xi * x ** beta) ** (-(1.0 / xi + 1.0 / beta))
    return beta / (2.0 * gamma_fn(1.0 / beta)) * np.exp(-x ** beta)


def sgt_unit_pdf(x, beta, xi, zeta):
    """(H2): the two-piece skewed density, unit integral, mode at 0, right-to-left half-width ratio zeta^2."""
    x = np.asarray(x, float)
    pref = 2.0 / (zeta + 1.0 / zeta)
    return pref * np.where(x >= 0, sgt_core(x / zeta, beta, xi), sgt_core(zeta * x, beta, xi))


def sgt_unit_fwhm(beta, xi, zeta):
    """W(kappa) of (H3), with 2^e - 1 as expm1(e ln 2) (the 10-16 handoff's cancellation note)."""
    if xi > SGT_XI_ZERO:
        e = xi * beta / (beta + xi)
        return (zeta + 1.0 / zeta) * (np.expm1(e * np.log(2.0)) / xi) ** (1.0 / beta)
    return (zeta + 1.0 / zeta) * np.log(2.0) ** (1.0 / beta)


def sgt_kernel(z, w, beta, xi, zeta):
    """g_{w,kappa}(z) = (W/w) f(W z / w | kappa) of (H3): the member of FWHM w (um)."""
    W = sgt_unit_fwhm(beta, xi, zeta)
    return W / w * sgt_unit_pdf(W * np.asarray(z, float) / w, beta, xi, zeta)


# The members named in the 10-16 handoff s.3.2 (l.101-106), kappa = (beta, xi, zeta)
SGT_MEMBERS = [
    ("Gaussian", (2.0, 0.0, 1.0)),
    ("Laplace", (1.0, 0.0, 1.0)),
    ("Student $t_4$", (2.0, 0.5, 1.0)),
    ("Lorentzian ($t_1$)", (2.0, 2.0, 1.0)),
    ("gen. normal, $\\beta$ = 4", (4.0, 0.0, 1.0)),
    ("skewed Gaussian, $\\zeta$ = 1.5", (2.0, 0.0, 1.5)),
]


# --------------------------------------------------------------------------- T3/T6/T10: the defocus kernel law
# Configured renderer kernel, config.py RendererConfig l.186-191 at 5015806 (ideal Debye, 550 nm, NOT calibrated)
CONFIGURED_KNOTS_UM = (0.0, 0.14, 0.28, 0.42, 0.56, 0.84)
CONFIGURED_SIGMA_UM = (0.080, 0.086, 0.122, 0.262, 0.438, 0.603)
CONFIGURED_GAMMA = 0.79
DELTA0_UM = 0.84          # the anchor / junction (D-080.1); 3 planes of 0.28 um
PLANE_UM = 0.28
P_FAR = 16                # SPEC 15.1: far scan k* +- 16 (4.48 um)
FAR_MIN_K_OFFSET = 5      # SPEC 15.3: far fit planes |k - k*| >= 5

# Pilot 529878215, Cell 7 on Colab, 2026-10-30 (decision log row "2026-10-30 (later)", l.139; SPEC 584 for the SE)
PILOT_KNOTS_UM = (0.0, 0.28, 0.56, 0.84)
PILOT_SIGMA_UM = (0.080, 0.143, 0.384, 0.547)    # sigma_r(0) = 0.080 is configured, not measured
PILOT_ANCHOR_SE_UM = 0.0438                       # after D-115.4 (1000 of 1000 resamples); 0.041 before
PILOT_GAMMA_HAT, PILOT_GAMMA_T1 = 0.408, (-0.01, 0.83)   # diagnostic on 2 far nodes; no production kernel written

# First stage of Block 10 on noise-free phantoms rendered with the configured kernel (partition, true line, alpha 0.1)
# checks/d094_2026-10-28/dark_node_growth.out (branch): sigma_hat_r at 0.28, 0.56, 0.84 um
PHANTOM_FIRST_STAGE = {0.3: (0.092, 0.389, 0.597), 0.5: (0.106, 0.357, 0.601)}

# Prior slopes from geometry: mathematics Eq. 17 table (l.520-524) and the sine-condition doc (l.644-648)
PRIOR_SLOPES = [
    ("uniform disc", 1.21),
    ("isotropic cone ($W^*$)", 0.90),
    ("$\\cos\\vartheta$-weighted cone, NA$_c$ 1.12", 0.4728),
    ("$\\cos\\vartheta$-weighted cone, NA$_c$ 1.05", 0.4259),
]
QUARTER_WAVE_UM = 0.1469   # this chat's check_diffraction.py: defocus at which the edge phase error is pi/2


def sigma_r_law(delta, knots, sigmas, gamma):
    """sigma_r(delta): linear interpolation of the table in |delta|, then sigma_M + gamma (|delta| - t_M)
    beyond the last knot (DL handoff s.4.2 l.115; D-080.1 for the rendered tables)."""
    a = np.abs(np.asarray(delta, float))
    out = np.interp(a, knots, sigmas)
    far = a > knots[-1]
    out[far] = sigmas[-1] + gamma * (a[far] - knots[-1])
    return out


# --------------------------------------------------------------------------- T4.3: the inversion (illustration)
# mathematics s.3.6 (l.712-715): "b* = 1.15 and beta = 0.3 give -4.3 % (numbers chosen, not measured)".
TOY = dict(b_star=1.15, d_star=1.0, beta=0.3, tau=0.10)
D_RANGE_UM = (0.2, 4.0)    # config.py TableConfig: the table's d range (d_max TEMPORARY, D-086)


def b_toy(d, b_star=TOY["b_star"], d_star=TOY["d_star"], beta=TOY["beta"]):
    """A bias factor with constant elasticity beta = d ln b / d ln d (Eq. 21's beta) and b(d*) = b*."""
    return b_star * (np.asarray(d, float) / d_star) ** beta


def m_toy(d):
    """m(d) = d b(d): the mean measured diameter (mathematics Eq. 19)."""
    return np.asarray(d, float) * b_toy(d)


def invert(d_hat, m=m_toy, lo=D_RANGE_UM[0], hi=D_RANGE_UM[1]):
    """Root of m(d) = d_hat on the table's d range (mathematics Eq. 20; SPEC 14.5); None when out of range."""
    f = lambda d: m(d) - d_hat
    if f(lo) * f(hi) > 0:
        return None
    return brentq(f, lo, hi, xtol=1e-14, rtol=1e-14)


def shortcut(d_hat):
    """The superseded shortcut d_hat / b(d_hat) (mathematics s.3.6, 'Error of the handoff's shortcut')."""
    return d_hat / b_toy(d_hat)


def eq21(b_star=TOY["b_star"], beta=TOY["beta"]):
    """Mathematics Eq. 21: relative error of the shortcut, full form and first order."""
    full = -beta * (b_star - 1) / (1 + beta * (b_star - 1))
    return full, -beta * (b_star - 1)
