"""Focus score and sharpest plane -- Block 5 in specs/SPEC.md (handoff Eqs. 1-2).

For node j and plane k, with the raw profile I_{j,k}(v) along node j's
measuring line (no along-branch averaging, Eq. 1 as written):

    I~ = I smoothed by Gaussian weights of s = focus_smooth_px samples,
    B_{j,k} = median of I at |v| > focus_bg_ends_um  (focus_bg_rule
              "profile_ends_median", D-023), or the node's B_bar ("same_as_bbar"),
    F_{j,k} = -ln(max(min I~, smallest positive float) / B_{j,k}),
    k*_j = argmax_k F_{j,k}.

Sub-plane depth (Eq. 2): the vertex of the parabola through the three best
scores, clipped to +-dz/2; the middle of the plateau when 3 or more
consecutive planes have F >= (1 - focus_plateau_rel) F_max.

Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

_TINY = np.finfo(float).tiny


def focus_score(I_raw, v, cfg, B_override=None):
    """(F, min of the smoothed profile, B) for one plane; F is NaN when the
    profile has a NaN sample or B is not positive."""
    I = np.asarray(I_raw, dtype=float)
    v = np.asarray(v, dtype=float)
    if not np.all(np.isfinite(I)):
        return float("nan"), float("nan"), float("nan")
    smooth = ndimage.gaussian_filter1d(I, cfg.focus_smooth_px, mode="nearest") if cfg.focus_smooth_px > 0 else I
    if cfg.focus_bg_rule == "profile_ends_median":
        ends = np.abs(v) > cfg.focus_bg_ends_um
        B = float(np.median(I[ends])) if ends.any() else float("nan")
    elif cfg.focus_bg_rule == "same_as_bbar":
        B = float(B_override) if B_override is not None else float("nan")
    else:
        raise ValueError("unknown focus_bg_rule %r" % (cfg.focus_bg_rule,))
    I_min = float(smooth.min())
    if not (B > 0):
        return float("nan"), I_min, B
    return float(-np.log(max(I_min, _TINY) / B)), I_min, B


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
