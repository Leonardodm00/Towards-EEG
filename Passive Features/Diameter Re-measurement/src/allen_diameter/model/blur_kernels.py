"""Blur-kernel families for the fit variants -- Block 11 in specs/SPEC.md
(D-046, the user's request of 2026-10-13, 14:50; a diagnostic: Block 5's fit
keeps its Gaussian, D-026).

A kernel is a family F, a full width at half maximum w (um) and a shape s:

    g(z) = f_F(z / a + m_F(s); s) / a,     a = w / W_F(s),

with f_F( . ; s) the family's density at unit scale, m_F(s) its mode and
W_F(s) its FWHM, so that for every family and shape g has unit integral, FWHM
w and its maximum at z = 0 (the tube's axis v0 stays where an infinitely thin
line's blurred image is darkest; only the skew-normal's mode is not 0).
Families, all from scipy:

    gauss     scipy.stats.norm                      no shape
    lorentz   scipy.stats.cauchy                    no shape
    voigt     scipy.special.voigt_profile           s = r, the ratio of the Lorentzian component's FWHM
                                                    to the Gaussian component's (unit scale: sigma =
                                                    1 / (2 sqrt(2 ln 2)), gamma = r / 2)
    student   scipy.stats.t                         s = nu, the degrees of freedom
    gennorm   scipy.stats.gennorm                   s = beta (1 Laplace, 2 Gaussian, large flat-topped)
    skewnorm  scipy.stats.skewnorm                  s = a, the skew (0 Gaussian)

W_F(s) has a closed form for gauss, lorentz, student and gennorm; for voigt
and skewnorm it is found numerically (scipy.optimize.brentq for f = f(mode)/2
on each side of the mode). m_F(s) is 0 except for skewnorm, where it is the
root of d/dz ln f = -z + a phi(a z) / Phi(a z) (scipy.optimize.brentq; phi,
Phi the standard normal density and distribution).

The skewed generalized t (SGT, D-052.1; Block 12, 2026-10-18) is the kernel
of the deliverable fit (analysis/fit_sgt.py). Its shape is a VECTOR
kappa = (beta, xi, zeta), so it has its own functions below and is not one
of FAMILIES (the D-046 grid's one-shape families, unchanged). With B the beta
and Gamma the gamma function, for beta > 0, xi >= 0, zeta > 0 and every x:

    h(x) = beta xi^(1/beta) / (2 B(1/beta, 1/xi)) (1 + xi |x|^beta)^-(1/xi + 1/beta)   (xi > 0)
    h(x) = beta / (2 Gamma(1/beta)) exp(-|x|^beta)                                    (xi = 0)   (H1)
    f(x | kappa) = 2 / (zeta + 1/zeta) h(x / zeta)  (x >= 0),   h(zeta x)  (x < 0)               (H2)
    W(kappa) = (zeta + 1/zeta) ((2^e - 1) / xi)^(1/beta),  e = xi beta / (beta + xi)  (xi > 0)
    W(kappa) = (zeta + 1/zeta) (ln 2)^(1/beta)                                         (xi = 0)
    g(z) = (W / w) f(W z / w | kappa)      (the kernel of FWHM w, um)                            (H3)

f has unit integral, its mode at 0 and right-to-left half-width ratio
zeta^2. Members: (2, 0, 1) Gaussian, (1, 0, 1) Laplace, (2, 2/nu, 1) Student
t_nu, (2, 2, 1) Lorentzian, (beta, 0, 1) generalized normal. xi at or below
SGT_XI_ZERO is the xi = 0 form; 2^e - 1 is computed as expm1(e ln 2), which
keeps W continuous as xi -> 0 (the naive form loses about 1e-7 relative at
small xi). Ported from checks/estimator_2026-10-15/sgt_kernel.py (shapes
named (p, r, gamma) there), with the same arithmetic. Custom: scipy.stats
has no generalized t and no SGT (checked there: t, gennorm, genhyperbolic,
jf_skew_t, johnsonsu, pearson3).

Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import functools
import math

import numpy as np
from scipy import optimize, special, stats

FAMILIES = ("gauss", "lorentz", "voigt", "student", "gennorm", "skewnorm")
NO_SHAPE = ("gauss", "lorentz")
# D-046, the user's choices of 2026-10-13: representative fixed members, and the range of a fitted shape
FIXED_SHAPES = {"voigt": (1.0,), "student": (3.0,), "gennorm": (1.0, 4.0)}
SHAPE_BOUNDS = {"voigt": (0.01, 100.0), "student": (1.0, 100.0), "gennorm": (0.5, 8.0), "skewnorm": (-10.0, 10.0)}
ALWAYS_FITTED = ("skewnorm",)                      # a skew across the branch has no sign known in advance
# a fitted shape is a fit parameter through this map (log10 for the ratios spanning decades)
SHAPE_SCALE = {"voigt": "log10", "student": "log10", "gennorm": "linear", "skewnorm": "linear"}
# the member nearest the Gaussian, from which a fitted shape also starts (PROVISIONAL, the assistant's)
GAUSS_LIKE = {"voigt": 0.05, "student": 30.0, "gennorm": 2.0, "skewnorm": 0.0}
_FWHM_GAUSS = 2.0 * math.sqrt(2.0 * math.log(2.0))
_SIG_V = 1.0 / _FWHM_GAUSS                        # the Voigt's Gaussian component at unit scale: FWHM 1


def check_kernel(family, shape):
    """(family, shape as float or None); ValueError for an unknown family, a
    shape given to a family without one, a missing shape, or a shape outside
    the family's bounds (SHAPE_BOUNDS)."""
    if family not in FAMILIES:
        raise ValueError("unknown kernel family %r (one of %s)" % (family, ", ".join(FAMILIES)))
    if family in NO_SHAPE:
        if shape is not None:
            raise ValueError("%s has no shape parameter, got %r" % (family, shape))
        return family, None
    if shape is None:
        raise ValueError("%s needs a shape parameter" % family)
    s = float(shape)
    lo, hi = SHAPE_BOUNDS[family]
    if not (math.isfinite(s) and lo <= s <= hi):
        raise ValueError("%s shape must lie in [%g, %g], got %r" % (family, lo, hi, shape))
    return family, s


def unit_pdf(family, shape, z):
    """f_F(z; s): the family's density at unit scale (location 0), at z (array)."""
    z = np.asarray(z, dtype=float)
    if family == "gauss":
        return stats.norm.pdf(z)
    if family == "lorentz":
        return stats.cauchy.pdf(z)
    if family == "voigt":
        return special.voigt_profile(z, _SIG_V, 0.5 * shape)
    if family == "student":
        return stats.t.pdf(z, shape)
    if family == "gennorm":
        return stats.gennorm.pdf(z, shape)
    if family == "skewnorm":
        return stats.skewnorm.pdf(z, shape)
    raise ValueError("unknown kernel family %r" % (family,))


@functools.lru_cache(maxsize=4096)
def unit_mode(family, shape):
    """m_F(s), the mode of the unit-scale density: 0 for the symmetric
    families; for skewnorm the root of -z + a phi(a z) / Phi(a z) in [-2, 2]
    (it lies in (0, 1) for a > 0, and m(-a) = -m(a))."""
    family, shape = check_kernel(family, shape)
    if family != "skewnorm" or shape == 0.0:
        return 0.0
    a = abs(shape)

    def dlog(z):                                   # d/dz ln f(z; a) for a > 0
        return -z + a * math.exp(stats.norm.logpdf(a * z) - special.log_ndtr(a * z))
    m = optimize.brentq(dlog, -2.0, 2.0, xtol=1e-15, rtol=4.0 * np.finfo(float).eps)
    return math.copysign(m, shape)


def _numeric_fwhm(f, mode):
    """FWHM of a unimodal density f on R with its maximum at mode: the two
    half-maximum points by brentq, bracketed by doubling a span from 1."""
    half = 0.5 * float(f(mode))
    span = 1.0
    while f(mode - span) > half or f(mode + span) > half:
        span *= 2.0
        if span > 1e6:
            raise ValueError("no half-maximum point within 1e6 of the mode")
    left = optimize.brentq(lambda x: f(x) - half, mode - span, mode, xtol=1e-14, rtol=4.0 * np.finfo(float).eps)
    right = optimize.brentq(lambda x: f(x) - half, mode, mode + span, xtol=1e-14, rtol=4.0 * np.finfo(float).eps)
    return right - left


@functools.lru_cache(maxsize=4096)
def unit_fwhm(family, shape):
    """W_F(s), the FWHM of the unit-scale density: closed forms for gauss
    (2 sqrt(2 ln 2)), lorentz (2), student (2 sqrt(nu (2^(2/(nu+1)) - 1))) and
    gennorm (2 (ln 2)^(1/beta)); numeric for voigt and skewnorm."""
    family, shape = check_kernel(family, shape)
    if family == "gauss":
        return _FWHM_GAUSS
    if family == "lorentz":
        return 2.0
    if family == "student":
        return 2.0 * math.sqrt(shape * (2.0 ** (2.0 / (shape + 1.0)) - 1.0))
    if family == "gennorm":
        return 2.0 * math.log(2.0) ** (1.0 / shape)
    return _numeric_fwhm(lambda x: float(unit_pdf(family, shape, x)), unit_mode(family, shape))


def kernel_pdf(family, fwhm_um, shape=None):
    """g: the kernel of the family with FWHM fwhm_um (um) and the given shape,
    a vectorised function of the offset z (um) with unit integral and its
    maximum at z = 0. ValueError for a FWHM that is not finite and positive,
    and as check_kernel."""
    family, shape = check_kernel(family, shape)
    w = float(fwhm_um)
    if not (math.isfinite(w) and w > 0):
        raise ValueError("the kernel's FWHM must be finite and > 0, got %r" % (fwhm_um,))
    a = w / unit_fwhm(family, shape)
    m = unit_mode(family, shape)

    def g(z):
        return unit_pdf(family, shape, np.asarray(z, dtype=float) / a + m) / a
    return g


def gauss_fwhm(sigma_um):
    """The FWHM of a Gaussian of standard deviation sigma_um: 2 sqrt(2 ln 2) sigma."""
    return _FWHM_GAUSS * float(sigma_um)


def shape_to_param(family, shape):
    """The fit parameter of a fitted shape (SHAPE_SCALE: log10 or as is)."""
    return math.log10(shape) if SHAPE_SCALE[family] == "log10" else float(shape)


def param_to_shape(family, p):
    """The inverse of shape_to_param."""
    return 10.0 ** float(p) if SHAPE_SCALE[family] == "log10" else float(p)


def kernel_label(family, shape=None, shape_fitted=False):
    """A short ASCII label: 'gauss', 'voigt r=1', 'gennorm beta fit', ..."""
    sym = {"voigt": "r", "student": "nu", "gennorm": "beta", "skewnorm": "a"}
    if family in NO_SHAPE:
        return family
    if shape_fitted:
        return "%s %s fit" % (family, sym[family])
    return "%s %s=%g" % (family, sym[family], shape)


# ---------------------------------------------------- the SGT (D-052.1) --
SGT = "sgt"              # the family's name (not in FAMILIES: its shape is the vector (beta, xi, zeta))
SGT_XI_ZERO = 1e-12      # xi at or below this is the generalized-normal limit xi = 0, (H1) second form


def check_sgt_shape(beta, xi, zeta):
    """(beta, xi, zeta) as floats; ValueError unless all three are finite,
    beta > 0, xi >= 0 and zeta > 0 (the family's domain; the fit's box is
    narrower, MeasureConfig.sgt_*_bounds)."""
    b, x, z = float(beta), float(xi), float(zeta)
    if not (math.isfinite(b) and math.isfinite(x) and math.isfinite(z)) or not (b > 0 and x >= 0 and z > 0):
        raise ValueError("SGT shape needs finite beta > 0, xi >= 0, zeta > 0; got %r" % ((beta, xi, zeta),))
    return b, x, z


def _sgt_log_c(beta, xi):
    """ln of the core's normaliser: beta xi^(1/beta) / (2 B(1/beta, 1/xi)), or
    beta / (2 Gamma(1/beta)) in the xi = 0 form."""
    if xi <= SGT_XI_ZERO:
        return math.log(beta) - math.log(2.0) - special.gammaln(1.0 / beta)
    return math.log(beta) + math.log(xi) / beta - math.log(2.0) - special.betaln(1.0 / beta, 1.0 / xi)


def _sgt_log_core(x, beta, xi):
    """ln of (1 + xi |x|^beta)^-(1/xi + 1/beta), or -|x|^beta in the xi = 0 form (x array)."""
    ax = np.abs(x)
    if xi <= SGT_XI_ZERO:
        return -(ax ** beta)
    return -(1.0 / xi + 1.0 / beta) * np.log1p(xi * ax ** beta)


def sgt_core_hwhm(beta, xi):
    """Half width at half maximum of the symmetric core h at unit scale:
    (expm1(e ln 2) / xi)^(1/beta), e = 1 / (1/xi + 1/beta) = xi beta / (beta + xi),
    or (ln 2)^(1/beta) in the xi = 0 form."""
    beta, xi, _ = check_sgt_shape(beta, xi, 1.0)
    if xi <= SGT_XI_ZERO:
        return math.log(2.0) ** (1.0 / beta)
    e = 1.0 / (1.0 / xi + 1.0 / beta)
    return (math.expm1(e * math.log(2.0)) / xi) ** (1.0 / beta)


def sgt_unit_fwhm(beta, xi, zeta):
    """W(kappa) of (H3): the FWHM of f( . | kappa), (zeta + 1/zeta) x sgt_core_hwhm(beta, xi)."""
    beta, xi, zeta = check_sgt_shape(beta, xi, zeta)
    return sgt_core_hwhm(beta, xi) * (zeta + 1.0 / zeta)


def sgt_unit_pdf(x, beta, xi, zeta):
    """f(x | kappa) of (H2) at x (array or scalar): unit integral, mode 0."""
    beta, xi, zeta = check_sgt_shape(beta, xi, zeta)
    s = zeta + 1.0 / zeta
    x = np.asarray(x, dtype=float)
    u = np.where(x >= 0.0, x / zeta, x * zeta)
    return np.exp(_sgt_log_c(beta, xi) + math.log(2.0 / s) + _sgt_log_core(u, beta, xi))


def sgt_kernel_pdf(fwhm_um, beta, xi, zeta):
    """g of (H3): the SGT kernel of FWHM fwhm_um (um) and shape (beta, xi,
    zeta), a vectorised function of the offset z (um, array in, array of the
    same shape out, 1/um) with unit integral and its maximum at z = 0.
    ValueError for a FWHM that is not finite and positive, and as
    check_sgt_shape."""
    beta, xi, zeta = check_sgt_shape(beta, xi, zeta)
    w = float(fwhm_um)
    if not (math.isfinite(w) and w > 0):
        raise ValueError("the kernel's FWHM must be finite and > 0, got %r" % (fwhm_um,))
    s = zeta + 1.0 / zeta
    a = w / (sgt_core_hwhm(beta, xi) * s)
    log_lead = _sgt_log_c(beta, xi) + math.log(2.0 / s) - math.log(a)

    def g(z):
        x = np.asarray(z, dtype=float) / a
        u = np.where(x >= 0.0, x / zeta, x * zeta)
        return np.exp(log_lead + _sgt_log_core(u, beta, xi))
    return g
