"""Residual weights about the centre and the near-bump cut -- Block 39 in specs/SPEC.md (D-169.4, D-169.5,
D-171.1 to D-171.4; the details D-174 [orchestrator]).

A clear node's line profiles I_k(v_n) (grey levels; planes k, line samples v_n in um) can still carry the
shadow of a structure that is not in the trace. Two tools keep it out of the fit:

    allen_fwhm_um     FWHM_A = sqrt((c D_A)^2 + (2 sqrt(2 ln 2) sigma_r(0))^2), c = sqrt(3) / 2: the half-depth
                      width of the in-focus dip of a faint tube of Allen's diameter D_A blurred by a Gaussian
                      of SD sigma_r(0) (variances added, as for two Gaussians).
    centre_weights    w_n = exp(-|v_n - v_c| / lambda), lambda = kappa x FWHM_A.
    bump_cut / bump_mask
                      a second dip beside the central one, rising at least m noise SDs above the lightest
                      point that separates the two: the samples from that point outwards get no weight. The
                      dip searched is the mean over the used planes, smoothed along the line: on a single
                      plane's samples a rise of 2 noise SDs is met by the noise alone in every profile
                      (checks/d169_2026-11-18/bump_false_positive_check.py).

Pure ASCII, LF, for the cluster. No file access, no plotting.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

CHORD_FWHM_FACTOR = math.sqrt(3.0) / 2.0             # half-depth width of 2 sqrt(r^2 - v^2), in diameters
GAUSS_FWHM_FACTOR = 2.0 * math.sqrt(2.0 * math.log(2.0))   # a Gaussian's FWHM in SDs


def allen_fwhm_um(d_allen_um, sigma_r0_um):
    """FWHM_A (um) of the in-focus dip expected from Allen's diameter d_allen_um (um, > 0) and the kernel's
    in-focus SD sigma_r0_um (um, >= 0)."""
    d, s = float(d_allen_um), float(sigma_r0_um)
    if not (math.isfinite(d) and d > 0):
        raise ValueError("d_allen_um must be finite and > 0, got %r" % (d_allen_um,))
    if not (math.isfinite(s) and s >= 0):
        raise ValueError("sigma_r0_um must be finite and >= 0, got %r" % (sigma_r0_um,))
    return math.hypot(CHORD_FWHM_FACTOR * d, GAUSS_FWHM_FACTOR * s)


def _line(v_um):
    v = np.array(v_um, dtype=float, copy=True).reshape(-1)
    if v.size < 3 or not np.all(np.isfinite(v)) or not np.all(np.diff(v) > 0):
        raise ValueError("v_um must be at least 3 finite, strictly increasing offsets")
    return v


def centre_weights(v_um, centre_um, lambda_um):
    """w_n = exp(-|v_n - centre_um| / lambda_um), shape (n_v,), dimensionless in (0, 1]. lambda_um > 0 (um)."""
    v = _line(v_um)
    c, lam = float(centre_um), float(lambda_um)
    if not math.isfinite(c):
        raise ValueError("centre_um must be finite")
    if not (math.isfinite(lam) and lam > 0):
        raise ValueError("lambda_um must be finite and > 0, got %r" % (lambda_um,))
    return np.exp(-np.abs(v - c) / lam)


# Custom: the cut rule of SPEC Block 39 (central dip's end at a fraction of its depth, then the nearest local
# maximum of the dip whose prominence reaches m noise SDs, cut at the lightest sample before it). Checked:
# scipy.signal.find_peaks / peak_prominences give the maxima and their prominences, used below; no library
# function returns the cut.
def _side_cut(D, v, m_sigma, level):
    """One side, ordered outwards from the central extreme (D[0] = the central depth): the offset v of the
    cut, or None. D: dip depths (gl), v: their offsets (um)."""
    below = np.flatnonzero(D <= level)
    if below.size == 0:
        return None
    e = int(below[0])                           # the central dip has ended here
    seg = D[e:]
    if seg.size < 3:
        return None
    peaks, _props = find_peaks(seg, prominence=m_sigma)
    if peaks.size == 0:
        return None
    b = int(peaks[0])                           # the bump nearest the centre
    return float(v[e + int(np.argmin(seg[:b + 1]))])


def bump_cut(dip_gl, sigma_gl, v_um, centre_um, window_um, m_sd, half_depth):
    """(v_lo, v_hi) in um for one dip profile: the samples outside [v_lo, v_hi] lie from a near bump outwards
    (-inf / +inf on a side without one).

    dip_gl (n_v,) = background - intensity (grey levels), finite; sigma_gl > 0 the noise SD (gl) the rise is
    counted in; v_um (n_v,) strictly increasing (um); centre_um the dip's expected centre (um); window_um > 0:
    the central extreme is the deepest sample within it of the centre; m_sd > 0: a bump's prominence is at
    least m_sd x sigma_gl; half_depth in (0, 1): the central dip ends at that fraction of its depth."""
    v = _line(v_um)
    D = np.array(dip_gl, dtype=float, copy=True).reshape(-1)
    if D.shape != v.shape or not np.all(np.isfinite(D)):
        raise ValueError("dip_gl must be finite with v_um's shape %s" % (v.shape,))
    sig, c, win = float(sigma_gl), float(centre_um), float(window_um)
    m, h = float(m_sd), float(half_depth)
    if not (math.isfinite(sig) and sig > 0):
        raise ValueError("sigma_gl must be finite and > 0")
    if not (math.isfinite(c) and math.isfinite(win) and win > 0):
        raise ValueError("centre_um must be finite and window_um finite and > 0")
    if not (math.isfinite(m) and m > 0 and 0 < h < 1):
        raise ValueError("m_sd must be > 0 and half_depth in (0, 1)")
    near = np.flatnonzero(np.abs(v - c) <= win)
    if near.size == 0:
        near = np.array([int(np.argmin(np.abs(v - c)))])
    n_c = int(near[np.argmax(D[near])])
    D_c = float(D[n_c])
    if D_c < m * sig:                            # no readable central dip: no cut
        return -math.inf, math.inf
    level = h * D_c
    hi = _side_cut(D[n_c:], v[n_c:], m * sig, level)
    lo = _side_cut(D[n_c::-1], v[n_c::-1], m * sig, level)
    return (-math.inf if lo is None else lo), (math.inf if hi is None else hi)


def mean_dip(profiles_gl, B_gl, smooth_samples, planes=None):
    """The dip the bumps are searched on, (n_v,) grey levels: mean over the used planes k of B_k - I_k(v_n),
    smoothed along the line by a Gaussian of SD smooth_samples line samples (scipy.ndimage.gaussian_filter1d,
    mode "nearest"; 0 = not smoothed). profiles_gl (n_planes, n_v); B_gl scalar or (n_planes,); planes
    (n_planes,) bool or None (all)."""
    P = np.array(profiles_gl, dtype=float, copy=True)
    if P.ndim != 2 or P.shape[0] == 0 or P.shape[1] < 3 or not np.all(np.isfinite(P)):
        raise ValueError("profiles_gl must be finite, (n_planes, n_v) with n_v >= 3, got %s" % (P.shape,))
    B = np.broadcast_to(np.asarray(B_gl, dtype=float), (P.shape[0],))
    if not np.all(np.isfinite(B)):
        raise ValueError("B_gl must be finite")
    if planes is None:
        use = np.ones(P.shape[0], dtype=bool)
    else:
        use = np.array(planes, copy=True)
        if use.shape != (P.shape[0],) or use.dtype != bool:
            raise ValueError("planes must be a bool array of shape (%d,)" % P.shape[0])
    if not np.any(use):
        raise ValueError("no plane used")
    s = float(smooth_samples)
    if not (math.isfinite(s) and s >= 0):
        raise ValueError("smooth_samples must be finite and >= 0")
    D = (B[use][:, None] - P[use]).mean(axis=0)
    return gaussian_filter1d(D, s, mode="nearest") if s > 0 else D


def bump_mask(profiles_gl, B_gl, sigma_gl, v_um, centre_um, window_um, m_sd, half_depth, smooth_samples,
              planes=None):
    """(keep, v_lo, v_hi): keep (n_v,) bool = v_lo <= v_n <= v_hi, the same for every plane: bump_cut on
    mean_dip(profiles_gl, B_gl, smooth_samples, planes), the rise counted in SDs sigma_gl of one line sample
    of one plane (the fit's sigma). m_sd = 0 disables (all True, infinite cuts)."""
    v = _line(v_um)
    if not (math.isfinite(float(m_sd)) and float(m_sd) >= 0):
        raise ValueError("m_sd must be finite and >= 0")
    D = mean_dip(profiles_gl, B_gl, smooth_samples, planes)
    if D.shape != v.shape:
        raise ValueError("profiles_gl must have n_v = %d line samples, got %d" % (v.size, D.size))
    v_lo, v_hi = -math.inf, math.inf
    if float(m_sd) > 0:
        v_lo, v_hi = bump_cut(D, sigma_gl, v, centre_um, window_um, m_sd, half_depth)
    keep = (v >= v_lo) & (v <= v_hi)
    return keep, v_lo, v_hi
