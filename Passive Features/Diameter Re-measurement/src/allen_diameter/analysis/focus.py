"""Focus scores and the sharpest plane -- Block 5 in specs/SPEC.md (D-030; handoff Eqs. 1-2).

For node j with radius r_j (Allen's radius on real nodes, the true radius on
phantoms) and plane k, with the raw profile I_{j,k}(v) along node j's
measuring line (no along-branch averaging; v in um, v = 0 at the node):

    I~_{j,k} = I_{j,k} smoothed by Gaussian weights of s = focus_smooth_px samples,
    B_{j,k}  = median of I_{j,k} at |v| > focus_bg_ends_um (focus_bg_rule
               "profile_ends_median", D-023), or the node's B_bar ("same_as_bbar").

Scores, one per rule of focus_rule (the first is the deliverable, D-030):

    "gradient_energy"   G_{j,k} = B_{j,k}^-2 * integral over |v| <= h_j of (dI~_{j,k}/dv)^2 dv   [1/um]
                        h_j = r_j + focus_grad_margin_um   (focus_grad_window "radius_margin")
                        h_j = the whole profile             ("whole_profile")
    "dip_depth"         F_{j,k} = -ln(max(min_v I~_{j,k}, smallest positive float) / B_{j,k})
                        (handoff Eq. 1 as written; the labelled comparison)

k*_j = argmax_k of the configured score. The derivative is numpy.gradient
(central differences, one-sided at the two ends) and the integral
scipy.integrate.trapezoid over the samples with |v| <= h_j.

Sub-plane depth (Eq. 2): the vertex of the parabola through the three best
scores, clipped to +-dz/2; the middle of the plateau when 3 or more
consecutive planes have a score >= (1 - focus_plateau_rel) x the maximum.

Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import math

import numpy as np
from scipy import integrate, ndimage

_TINY = np.finfo(float).tiny
_V_TOL_UM = 1e-9      # a sample at exactly |v| = h_j is inside the window


def _smoothed(I, cfg):
    return ndimage.gaussian_filter1d(I, cfg.focus_smooth_px, mode="nearest") if cfg.focus_smooth_px > 0 else I


def plane_background(I_raw, v, cfg, B_override=None):
    """B_{j,k} of one plane by focus_bg_rule (D-023); NaN when it is undefined."""
    if cfg.focus_bg_rule == "profile_ends_median":
        ends = np.abs(v) > cfg.focus_bg_ends_um
        return float(np.median(I_raw[ends])) if ends.any() else float("nan")
    if cfg.focus_bg_rule == "same_as_bbar":
        return float(B_override) if B_override is not None else float("nan")
    raise ValueError("unknown focus_bg_rule %r" % (cfg.focus_bg_rule,))


def grad_half_width(v, cfg, radius_um=None):
    """h_j (um): the half-width of the gradient-energy window."""
    if cfg.focus_grad_window == "whole_profile":
        return float(np.max(np.abs(v)))
    if cfg.focus_grad_window == "radius_margin":
        if radius_um is None or not (math.isfinite(float(radius_um)) and float(radius_um) >= 0):
            raise ValueError("focus_grad_window 'radius_margin' needs the node's radius (finite, >= 0), got %r"
                             % (radius_um,))
        return float(radius_um) + cfg.focus_grad_margin_um
    raise ValueError("unknown focus_grad_window %r" % (cfg.focus_grad_window,))


def dip_depth(I_raw, v, cfg, B_override=None):
    """(F, min of the smoothed profile, B) for one plane (handoff Eq. 1); F is
    NaN when the profile has a non-finite sample or B is not positive."""
    I = np.asarray(I_raw, dtype=float)
    v = np.asarray(v, dtype=float)
    if not np.all(np.isfinite(I)):
        return float("nan"), float("nan"), float("nan")
    B = plane_background(I, v, cfg, B_override)
    I_min = float(_smoothed(I, cfg).min())
    if not (B > 0):
        return float("nan"), I_min, B
    return float(-np.log(max(I_min, _TINY) / B)), I_min, B


def gradient_energy(I_raw, v, cfg, B_override=None, radius_um=None):
    """(G in 1/um, B) for one plane (D-030); G is NaN when the profile has a
    non-finite sample or B is not positive. v must increase strictly."""
    I = np.asarray(I_raw, dtype=float)
    v = np.asarray(v, dtype=float)
    if I.shape != v.shape or v.ndim != 1 or v.size < 3 or np.any(np.diff(v) <= 0):
        raise ValueError("gradient_energy: I and v must be 1-D of the same length >= 3, v increasing")
    h = grad_half_width(v, cfg, radius_um)
    if not np.all(np.isfinite(I)):
        return float("nan"), float("nan")
    B = plane_background(I, v, cfg, B_override)
    if not (B > 0):
        return float("nan"), B
    slope = np.gradient(_smoothed(I, cfg), v)
    inside = np.abs(v) <= h + _V_TOL_UM
    if np.count_nonzero(inside) < 2:
        raise ValueError("gradient_energy: the window |v| <= %g um holds fewer than 2 samples" % h)
    return float(integrate.trapezoid(slope[inside] ** 2, v[inside])) / B ** 2, B


def plane_scores(I_raw, v, cfg, B_override=None, radius_um=None):
    """Every rule's score of one plane, {rule: score}, keyed as config.FOCUS_RULES."""
    return {"gradient_energy": gradient_energy(I_raw, v, cfg, B_override, radius_um)[0],
            "dip_depth": dip_depth(I_raw, v, cfg, B_override)[0]}


def best_plane(F, z, cfg, dz):
    """Sharpest plane and sub-plane depth from focus scores F over planes at
    depths z (consecutive planes dz apart; NaN = missing or unusable).

    Returns (index of k*, z_sub, plateau (bool), at_edge (bool)); index is
    None when every score is NaN. at_edge: k* within edge_margin_planes of
    the first or last usable plane (the end of the searched range, or of the
    stack when planes are missing), i.e. the search may have been cut short.
    """
    F = np.asarray(F, dtype=float)
    z = np.asarray(z, dtype=float)
    ok = np.isfinite(F)
    if not ok.any():
        return None, float("nan"), False, False
    k = int(np.nanargmax(F))
    usable = np.flatnonzero(ok)
    at_edge = k - int(usable[0]) < cfg.edge_margin_planes or int(usable[-1]) - k < cfg.edge_margin_planes
    if not cfg.subplane_depth:
        return k, float(z[k]), False, at_edge
    F0 = F[k]
    level = (1.0 - cfg.focus_plateau_rel) * F0
    lo = k
    while lo - 1 >= 0 and ok[lo - 1] and F[lo - 1] >= level:
        lo -= 1
    hi = k
    while hi + 1 < F.size and ok[hi + 1] and F[hi + 1] >= level:
        hi += 1
    if F0 > 0 and hi - lo + 1 >= 3:
        return k, float(np.mean(z[lo:hi + 1])), True, at_edge
    if k == 0 or k == F.size - 1 or not (ok[k - 1] and ok[k + 1]):
        return k, float(z[k]), False, at_edge
    Fm, Fp = F[k - 1], F[k + 1]
    den = Fm - 2.0 * F0 + Fp
    if not (den < 0):
        return k, float(z[k]), False, at_edge
    shift = float(np.clip(0.5 * (Fm - Fp) / den, -0.5, 0.5))
    return k, float(z[k] + dz * shift), False, at_edge


def plane_differences(stack, valid=None, mask=None):
    """Consecutive-plane differences of a z-stack (the user's proposal of
    2026-10-08; a diagnostic, not a focus rule).

    stack: (n, H, W) planes in increasing plane index, consecutive planes dz
    apart, grey levels; valid: (n,) bool or None (all valid); mask: (H, W)
    bool or None (every pixel), the pixels counted. For n = 0 .. n-2:

        D_n(p) = I_{n+1}(p) - I_n(p)
        S+_n   = (1/N) * sum over p of max(D_n(p), 0)     [grey levels per pixel]
        S-_n   = (1/N) * sum over p of max(-D_n(p), 0)

    with N the number of pixels counted. A pair touching an invalid plane, or
    with a non-finite sample among the counted pixels, is NaN. Reversing the
    plane order swaps S+ and S-; S+_n - S-_n is the mean of D_n. Returns
    dict(diff=(n-1, H, W) float, D_n with NaN planes for unusable pairs,
    pos=(n-1,) S+, neg=(n-1,) S-).
    """
    a = np.asarray(stack, dtype=float)
    if a.ndim != 3 or a.shape[0] < 2:
        raise ValueError("plane_differences: stack must be (n >= 2, H, W), got shape %r" % (a.shape,))
    ok = np.ones(a.shape[0], dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    if ok.shape != (a.shape[0],):
        raise ValueError("plane_differences: valid must have one entry per plane")
    m = np.ones(a.shape[1:], dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if m.shape != a.shape[1:] or not m.any():
        raise ValueError("plane_differences: mask must be (H, W) with at least one pixel counted")
    d = np.diff(a, axis=0)
    inside = d[:, m]                                       # (n-1, N)
    use = ok[:-1] & ok[1:] & np.all(np.isfinite(inside), axis=1)
    pos = np.full(d.shape[0], np.nan)
    neg = np.full(d.shape[0], np.nan)
    pos[use] = np.clip(inside[use], 0.0, None).mean(axis=1)
    neg[use] = np.clip(-inside[use], 0.0, None).mean(axis=1)
    d[~use] = np.nan
    return dict(diff=d, pos=pos, neg=neg)


def difference_dip(S):
    """Index of the dip of a difference curve S (n,) (plane_differences' pos or
    neg): the minimum between its two largest local maxima, the two ends
    counting as local maxima; the global minimum when fewer than two maxima
    exist or the two are adjacent; None when S has no finite value. NaN
    entries are skipped. The reading of the proposal (PROVISIONAL, assistant's):
    far from focus consecutive planes differ little, each side of focus has a
    maximum of change, and the focal pair is the dip between the two."""
    S = np.asarray(S, dtype=float)
    idx = np.flatnonzero(np.isfinite(S))
    if idx.size == 0:
        return None
    s = S[idx]
    n = s.size
    peaks = [i for i in range(n) if (i == 0 or s[i] >= s[i - 1]) and (i == n - 1 or s[i] >= s[i + 1])]
    top2 = sorted(sorted(peaks, key=lambda i: (-s[i], i))[:2])
    if len(top2) < 2 or top2[1] - top2[0] < 2:
        return int(idx[int(np.argmin(s))])
    a, b = top2
    return int(idx[a + 1 + int(np.argmin(s[a + 1:b]))])
