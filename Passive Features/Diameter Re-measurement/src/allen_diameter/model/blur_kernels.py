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
