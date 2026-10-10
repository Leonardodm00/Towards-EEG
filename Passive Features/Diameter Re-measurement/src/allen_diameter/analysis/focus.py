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


def profile_areas(profiles, v, valid=None):
    """Area under each plane's profile and its change between consecutive
    planes (the user's evaluation of 2026-10-08, 16:26; a diagnostic).

    profiles: (n, M) I_k(v_m), the profile of plane k along the measuring line
    (grey levels), planes in increasing, consecutive index; v: (M,) offsets in
    um, increasing; valid: (n,) bool or None.

        A_k  = integral over v of I_k(v) dv     (trapezoid; grey levels x um)
        dA_n = A_{k_n + 1} - A_{k_n}

    A plane that is invalid or holds a non-finite sample has A_k = NaN, and so
    have the two differences that touch it. Returns dict(area=(n,), d_area=(n-1,)).
    Under a blur that conserves the light of each plane (a normalised kernel,
    no truncation by the window) A_k does not change with the plane; what it
    sees is light leaving or entering the window and changes of the whole
    plane's brightness."""
    P = np.asarray(profiles, dtype=float)
    v = np.asarray(v, dtype=float)
    if P.ndim != 2 or v.ndim != 1 or P.shape[1] != v.size or v.size < 2 or np.any(np.diff(v) <= 0):
        raise ValueError("profile_areas: profiles must be (n, M) with v (M,) increasing, M >= 2")
    ok = np.ones(P.shape[0], dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    if ok.shape != (P.shape[0],):
        raise ValueError("profile_areas: valid must have one entry per profile")
    ok = ok & np.all(np.isfinite(P), axis=1)
    area = np.full(P.shape[0], np.nan)
    area[ok] = integrate.trapezoid(P[ok], v, axis=1)
    return dict(area=area, d_area=np.diff(area))


def grey_histogram(values, bin_width=1.0):
    """The finite values counted in bins of width w = bin_width centred on the
    multiples of w: value x goes to bin j = floor(x / w + 1/2), the half-open
    interval [(j - 1/2) w, (j + 1/2) w). Returns (centres (m,) = j w, counts (m,))
    from the lowest to the highest occupied bin, empty bins in between
    included; two empty arrays when no value is finite."""
    # Custom: bins centred on the grey levels, half-open on every side. Checked: numpy.histogram (its last bin is
    # closed, and edges at half-integers in floating point are what this avoids); numpy.bincount does the counting.
    if not bin_width > 0:
        raise ValueError("grey_histogram: bin_width must be > 0, got %r" % (bin_width,))
    x = np.asarray(values, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.empty(0), np.empty(0, dtype=np.int64)
    idx = np.floor(x / float(bin_width) + 0.5).astype(np.int64)
    lo = int(idx.min())
    counts = np.bincount(idx - lo)
    return (lo + np.arange(counts.size)) * float(bin_width), counts


def histogram_entropy(values, bin_width=1.0):
    """Shannon entropy of the grey-level histogram of `values` (the user's
    proposal of 2026-10-08, 17:07; a diagnostic). Returns (H in bits, N, m).

    With the bins of grey_histogram (for 8-bit grey levels and bin_width 1,
    one bin per grey level; a bilinear sample goes to the nearest level),
    p_j = n_j / N over the N finite values and

        H = -sum over occupied bins j of p_j log2 p_j       [bits]

    (scipy.stats.entropy, base 2): the plug-in estimate of the entropy of the
    distribution the values were drawn from, biased low when N is not large
    against the number m of occupied bins. Non-finite values are dropped;
    (NaN, 0, 0) when none is left. Adding a multiple of bin_width to every
    value, or permuting the values, leaves H unchanged."""
    from scipy import stats
    _, counts = grey_histogram(values, bin_width)
    occupied = counts[counts > 0]
    if occupied.size == 0:
        return float("nan"), 0, 0
    return float(stats.entropy(occupied, base=2)), int(occupied.sum()), int(occupied.size)


def plane_entropies(profiles, stack, strip, valid=None, bin_width=1.0):
    """Histogram entropies plane by plane (the user's proposal of 2026-10-08,
    17:07; a diagnostic): of the samples along the measuring line and of the
    pixels in the strip around it.

    profiles: (n, M) I_k(v_m), the bilinear samples along the measuring line in
    plane k (profiles.sample_profile; grey levels); stack: (n, H, W) the planes;
    strip: (H, W) bool, the pixels of the strip (profiles.stripe_mask), at
    least one; valid: (n,) bool or None (all valid). For each plane k:

        H_line_k  = histogram_entropy(I_k(v_1), ..., I_k(v_M))      [bits]
        H_strip_k = histogram_entropy(I_k(p), p in the strip)        [bits]

    A plane that is invalid, or holds a non-finite value among the samples a
    quantity uses, has that quantity NaN (and N, m = 0), so that every finite
    entropy rests on the same number of samples. Returns dict(h_line, n_line,
    m_line, h_strip, n_strip, m_strip), each (n,)."""
    P = np.asarray(profiles, dtype=float)
    S = np.asarray(stack, dtype=float)
    m = np.asarray(strip, dtype=bool)
    if P.ndim != 2 or S.ndim != 3 or P.shape[0] != S.shape[0]:
        raise ValueError("plane_entropies: profiles (n, M) and stack (n, H, W) must have one row per plane")
    if m.shape != S.shape[1:] or not m.any():
        raise ValueError("plane_entropies: strip must be (H, W) with at least one pixel")
    ok = np.ones(S.shape[0], dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    if ok.shape != (S.shape[0],):
        raise ValueError("plane_entropies: valid must have one entry per plane")
    out = {key: np.full(S.shape[0], np.nan) for key in ("h_line", "h_strip")}
    out.update({key: np.zeros(S.shape[0], dtype=int) for key in ("n_line", "m_line", "n_strip", "m_strip")})
    for k in range(S.shape[0]):
        if not ok[k]:
            continue
        for name, x in (("line", P[k]), ("strip", S[k][m])):
            if np.all(np.isfinite(x)):
                out["h_" + name][k], out["n_" + name][k], out["m_" + name][k] = histogram_entropy(x, bin_width)
    return out


def difference_dip(S):
    """Index of the dip of a curve S (n,) over consecutive planes or plane
    pairs (plane_differences' pos or neg; plane_entropies' h_line or h_strip):
    the minimum between its two largest local maxima, the two ends counting as
    local maxima; the global minimum when fewer than two maxima exist or the
    two are adjacent; None when S has no finite value. NaN entries are skipped.
    The readings (PROVISIONAL, assistant's): far from focus consecutive planes
    differ little, each side of focus has a maximum of change, and the focal
    pair is the dip between the two; the entropy of a dark tube's
    neighbourhood is lowest in focus, rises on both sides as the blur spreads
    the tube's grey levels over more pixels, and falls again far out as the
    tube fades into the background noise (synthetic tubes, 2026-10-08)."""
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


# ---- the gradient energy over the whole line and its blend with the strip entropy (D-040, 2026-10-09) ----

def plane_gradient_energies(profiles, v, B, cfg, valid=None):
    """G_k = B_k^-2 * integral over the whole line of (dI~_k/dv)^2 dv, plane
    by plane (D-040, a diagnostic): gradient_energy with focus_grad_window
    "whole_profile" and focus_bg_rule "same_as_bbar", B_k given as the
    override, so that between two lines of one node only the line differs.

    profiles: (n, M) I_k(v_m), grey levels; v: (M,) um, strictly increasing,
    M >= 3; B: (n,) each plane's background, grey levels; cfg: MeasureConfig
    (focus_smooth_px is used); valid: (n,) bool or None (all valid). A plane
    that is invalid, holds a non-finite sample, or has B_k not finite and
    positive gets NaN. Returns (n,) in 1/um."""
    import dataclasses
    P = np.asarray(profiles, dtype=float)
    v = np.asarray(v, dtype=float)
    B = np.asarray(B, dtype=float)
    if P.ndim != 2 or v.ndim != 1 or P.shape[1] != v.size or B.shape != (P.shape[0],):
        raise ValueError("plane_gradient_energies: profiles (n, M), v (M,) and B (n,) must agree")
    if v.size < 3 or np.any(np.diff(v) <= 0):
        raise ValueError("plane_gradient_energies: v must increase strictly and hold at least 3 samples")
    ok = np.ones(P.shape[0], dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    if ok.shape != (P.shape[0],):
        raise ValueError("plane_gradient_energies: valid must have one entry per plane")
    whole = dataclasses.replace(cfg, focus_grad_window="whole_profile", focus_bg_rule="same_as_bbar")
    G = np.full(P.shape[0], np.nan)
    for k in range(P.shape[0]):
        if ok[k] and math.isfinite(B[k]) and B[k] > 0:
            G[k] = gradient_energy(P[k], v, whole, B_override=float(B[k]))[0]
    return G


def sigmoid_weight(d_um, d0_um, s_um):
    """w(d) = 1 / (1 + exp((d - d0) / s)) (D-040): the gradient energy's weight
    in the blend, 1/2 at d = d0, going from 1 (thin) to 0 (thick) over a few
    s; scipy.special.expit((d0 - d) / s). s must be finite and > 0, d0 finite;
    NaN for a non-finite d."""
    from scipy.special import expit
    s, d0 = float(s_um), float(d0_um)
    if not (math.isfinite(s) and s > 0 and math.isfinite(d0)):
        raise ValueError("sigmoid_weight: s must be finite and > 0 and d0 finite, got s=%r, d0=%r" % (s_um, d0_um))
    d = float(d_um)
    return float(expit((d0 - d) / s)) if math.isfinite(d) else float("nan")


def minmax_scores(S, valid=None, higher_is_better=True):
    """A curve rescaled to [0, 1] over its usable entries (finite, and valid
    when given): (S - min) / (max - min), or (max - S) / (max - min) when
    higher_is_better is False (the curve inverted); every usable entry 0 when
    the curve is flat there (it carries no information); NaN elsewhere.
    Returns (n,)."""
    S = np.asarray(S, dtype=float)
    use = np.isfinite(S) if valid is None else (np.isfinite(S) & np.asarray(valid, dtype=bool))
    out = np.full(S.shape, np.nan)
    if not use.any():
        return out
    lo, hi = float(S[use].min()), float(S[use].max())
    if hi > lo:
        out[use] = (S[use] - lo) / (hi - lo) if higher_is_better else (hi - S[use]) / (hi - lo)
    else:
        out[use] = 0.0
    return out


def blend_scores(G, H, w):
    """The min-max blend of the gradient energy and the entropy (D-040; the
    form of D-041): over V, the planes where both G and H are finite,
    g = minmax_scores(G) (1 at the largest gradient energy) and
    h = minmax_scores(H) (0 at the lowest entropy), J = w g - (1 - w) h,
    largest where G is high and H low; NaN outside V. w in [0, 1]. Returns
    dict(g, h, J, V), V a bool mask."""
    G = np.asarray(G, dtype=float)
    H = np.asarray(H, dtype=float)
    if G.ndim != 1 or G.shape != H.shape:
        raise ValueError("blend_scores: G and H must be 1-D of the same length")
    w = float(w)
    if not 0.0 <= w <= 1.0:
        raise ValueError("blend_scores: w must lie in [0, 1], got %r" % (w,))
    V = np.isfinite(G) & np.isfinite(H)
    g = minmax_scores(G, V)
    h = minmax_scores(H, V)
    return dict(g=g, h=h, J=np.where(V, w * g - (1.0 - w) * h, np.nan), V=V)
