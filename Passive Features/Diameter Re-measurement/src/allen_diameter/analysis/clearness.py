"""The image-based clearness screen of a prepared node -- Block 40 in specs/SPEC.md (D-192 [orchestrator],
PROVISIONAL; the handoff's open point O2).

The trace-based rule (contamination.pick_clear_examples) only knows the traced neighbours; a structure that is
not in the trace still darkens the line. This screen reads the recorded image alone, with no forward model and
no diameter: a node is judged clean when its dip is ONE dip that is symmetric and returns to the background --
on the measuring line in the in-focus planes, and on parallel lines along the dendrite in the sharpest plane --
and when that dip keeps its width, depth and position along the dendrite.

For one dip profile D(v_n) = B - I(v_n) (grey levels; v_n in um), smoothed along the line:

    n_c   = argmax of D over |v_n - c0| <= window_um          the central extreme, depth D_c = D(v_{n_c})
    x_L(f), x_R(f)                                            where D first falls to f D_c going outwards from
                                                              n_c on each side (linear interpolation; NaN when
                                                              the line ends first)
    W     = x_R(1/2) - x_L(1/2)                               the half-depth width (um)
    c     = [x_L(1/2) + x_R(1/2)] / 2                         the dip's centre (um)
    S     = max over f in skew_levels of |[x_L(f) + x_R(f)] / 2 - c| / W          the skew (0 for a symmetric dip)
    E     = max over |v_n - c| >= outside_fwhm W of |D(v_n)| / D_c                the excess outside the dip

Along the dendrite, on the sharpest plane, with the same features (W_m, D_{c,m}, c_m) on the lines at
u_m = m along_step_um, |u_m| <= along_half_um:

    U_W = (max_m W_m - min_m W_m) / median_m W_m,   U_D likewise for the depth,
    U_c = (max_m c_m - min_m c_m) / median_m W_m.

Score Q = the largest of E / outside_max and S / skew_max over the judged profiles and of U_W / width_range_max,
U_D / depth_range_max, U_c / centre_range_max; +inf when a judged feature is not finite. Clean: Q <= 1 and the
central depth on the sharpest plane at least min_depth_sd noise SDs.

Pure ASCII, LF, for the cluster. No file access, no plotting.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d

from . import profiles

FEATURES = ("depth_gl", "fwhm_um", "centre_um", "skew", "outside")
CRITERIA = ("outside", "skew", "width_range", "depth_range", "centre_range")
REASON_NOT_FINITE = "not_finite"
REASON_FAINT = "faint"


def _line(v_um):
    v = np.array(v_um, dtype=float, copy=True).reshape(-1)
    if v.size < 5 or not np.all(np.isfinite(v)) or not np.all(np.diff(v) > 0):
        raise ValueError("v_um must be at least 5 finite, strictly increasing offsets")
    return v


# Custom: the outward walk to a fraction of the depth with linear interpolation, and the features built on it.
# Checked: scipy.signal.peak_widths returns the width of a peak at a relative height measured from the peak's
# prominence base, not from the background level 0 that B fixes here, and it does not stop at the first
# crossing going outwards when a second dip follows; no library function returns the skew or the excess outside.
def _crossing(D, v, n_c, step, level):
    """The offset where D first falls to `level` going from sample n_c in direction step (+1 or -1): linear
    interpolation between the last sample above the level and the first at or below it; NaN when the line
    ends first."""
    n = int(n_c)
    while 0 <= n + step < D.size and D[n + step] > level:
        n += step
    if not 0 <= n + step < D.size:
        return float("nan")
    a, b = float(D[n]), float(D[n + step])
    return float(v[n] + (v[n + step] - v[n]) * (a - level) / (a - b))


def smooth_dip(dip_gl, smooth_samples):
    """dip_gl (..., n_v) smoothed along its last axis by a Gaussian of SD smooth_samples line samples
    (scipy.ndimage.gaussian_filter1d, mode "nearest"; 0 = unchanged, a float64 copy). NaN stays where it is
    and spreads to its neighbours."""
    D = np.array(dip_gl, dtype=float, copy=True)
    s = float(smooth_samples)
    if not (math.isfinite(s) and s >= 0):
        raise ValueError("smooth_samples must be finite and >= 0")
    return gaussian_filter1d(D, s, axis=-1, mode="nearest") if s > 0 else D


def dip_features(dip_gl, v_um, centre_um=0.0, window_um=1.0, skew_levels=(0.75, 0.25), outside_fwhm=1.0):
    """The features of one dip profile -> dict(depth_gl, fwhm_um, centre_um, skew, outside, x_lo_um, x_hi_um).

    dip_gl (n_v,) = background - intensity (grey levels), already smoothed if it is to be; v_um (n_v,)
    strictly increasing (um); centre_um: where the dip is expected (um); window_um > 0: the central extreme is
    the deepest sample within it of centre_um; skew_levels: two fractions of the depth, one in (0.5, 1) and one
    in (0, 0.5); outside_fwhm > 0. depth_gl = D_c; fwhm_um = W; centre_um = c; skew = S; outside = E (the
    formulas of the module's docstring); x_lo_um, x_hi_um = x_L(1/2), x_R(1/2). A profile with a NaN, a
    central depth <= 0, a level the line never falls to on one side, or no sample outside the dip on one of
    the two sides gives NaN for what cannot be formed (depth_gl stays when it is finite)."""
    v = _line(v_um)
    D = np.array(dip_gl, dtype=float, copy=True).reshape(-1)
    if D.shape != v.shape:
        raise ValueError("dip_gl must have v_um's shape %s, got %s" % (v.shape, D.shape))
    c0, win, a = float(centre_um), float(window_um), float(outside_fwhm)
    lv = tuple(float(f) for f in skew_levels)
    if not (math.isfinite(c0) and math.isfinite(win) and win > 0):
        raise ValueError("centre_um must be finite and window_um finite and > 0")
    if not (len(lv) == 2 and 0.5 < lv[0] < 1 and 0 < lv[1] < 0.5):
        raise ValueError("skew_levels must be two fractions, one in (0.5, 1) then one in (0, 0.5), got %r" % (lv,))
    if not (math.isfinite(a) and a > 0):
        raise ValueError("outside_fwhm must be finite and > 0")
    nan = float("nan")
    out = dict(depth_gl=nan, fwhm_um=nan, centre_um=nan, skew=nan, outside=nan, x_lo_um=nan, x_hi_um=nan)
    if not np.all(np.isfinite(D)):
        return out
    near = np.flatnonzero(np.abs(v - c0) <= win)
    if near.size == 0:
        near = np.array([int(np.argmin(np.abs(v - c0)))])
    n_c = int(near[np.argmax(D[near])])
    D_c = float(D[n_c])
    out["depth_gl"] = D_c
    if not D_c > 0:
        return out
    x_lo, x_hi = _crossing(D, v, n_c, -1, 0.5 * D_c), _crossing(D, v, n_c, +1, 0.5 * D_c)
    out.update(x_lo_um=x_lo, x_hi_um=x_hi)
    if not (math.isfinite(x_lo) and math.isfinite(x_hi)):
        return out
    W, c = x_hi - x_lo, 0.5 * (x_lo + x_hi)
    out.update(fwhm_um=W, centre_um=c)
    mids = [0.5 * (_crossing(D, v, n_c, -1, f * D_c) + _crossing(D, v, n_c, +1, f * D_c)) for f in lv]
    if all(math.isfinite(m) for m in mids):
        out["skew"] = max(abs(m - c) for m in mids) / W
    left, right = v - c <= -a * W, v - c >= a * W
    if np.any(left) and np.any(right):
        out["outside"] = float(np.max(np.abs(D[left | right]))) / D_c
    return out


def strip_profiles(plane, left, top, res0_um, origin_xy, y_hat, e_u, v_um, n_avg, step_um, u_um):
    """The profiles on lines parallel to the measuring line, (n_u, n_v) grey levels: for each offset u of u_um
    (um, along the axis' projection e_u) profiles.sample_profile of `plane` on the line through
    origin_xy + u e_u along y_hat -- the measuring line's own read (the mean of 2 n_avg + 1 bilinear profiles
    step_um apart along e_u), moved along the dendrite. plane (H, W); left, top: the patch's corner in
    full-resolution pixels; res0_um: um per pixel; origin_xy (2,) um. NaN where a read leaves the patch."""
    img = np.asarray(plane, dtype=float)
    if img.ndim != 2:
        raise ValueError("plane must be (H, W), got %s" % (img.shape,))
    v = _line(v_um)
    o = np.asarray(origin_xy, dtype=float).reshape(-1)
    yh = np.asarray(y_hat, dtype=float).reshape(-1)
    eu = np.asarray(e_u, dtype=float).reshape(-1)
    u = np.atleast_1d(np.asarray(u_um, dtype=float))
    if o.shape != (2,) or yh.shape != (2,) or eu.shape != (2,) or not np.all(np.isfinite(np.concatenate([o, yh, eu]))):
        raise ValueError("origin_xy, y_hat and e_u must be finite 2-vectors")
    if u.ndim != 1 or u.size == 0 or not np.all(np.isfinite(u)):
        raise ValueError("u_um must be finite offsets, at least one")
    p = float(res0_um)
    if not (math.isfinite(p) and p > 0) or int(n_avg) < 0 or not (math.isfinite(float(step_um)) and float(step_um) >= 0):
        raise ValueError("res0_um must be > 0, n_avg >= 0 and step_um >= 0")
    frame = SimpleNamespace(left=int(left), top=int(top), res_um_px=p)
    return np.stack([profiles.sample_profile(img, frame, o + uu * eu, yh, eu, v, int(n_avg), float(step_um))
                     for uu in u])


def along_offsets(along_half_um, along_step_um):
    """u_m = m along_step_um, m = -M..M, M = floor(along_half_um / along_step_um) (um); (0,) when M = 0."""
    L, s = float(along_half_um), float(along_step_um)
    if not (math.isfinite(L) and L >= 0 and math.isfinite(s) and s > 0):
        raise ValueError("along_half_um must be >= 0 and along_step_um > 0")
    M = int(math.floor(L / s + 1e-9))
    return np.arange(-M, M + 1) * s


@dataclass(frozen=True)
class ClearnessResult:
    """What screen_node returns. Arrays of one node; n_planes its recorded planes, n_u the lines along the axis.

    clean: the verdict; score: Q (dimensionless; <= 1 passes; inf when a judged feature is not finite);
    reasons: the criteria above their threshold, of CRITERIA, REASON_NOT_FINITE, REASON_FAINT;
    ratios: {criterion: its largest value over its threshold} (NaN when it could not be formed);
    line: {feature: (n_planes,)} of FEATURES on the measuring line, every recorded plane;
    focus: (n_planes,) bool, the planes the line is judged on; k_index: the sharpest plane's index;
    u_um (n_u,); strips_gl (n_u, n_v): the profiles along the axis on the sharpest plane (grey levels);
    along: {feature: (n_u,)} of FEATURES on those; width_range, depth_range, centre_range: U_W, U_D, U_c;
    depth_sd: the central depth on the sharpest plane in noise SDs."""
    clean: bool
    score: float
    reasons: Tuple[str, ...]
    ratios: dict
    line: dict
    focus: np.ndarray
    k_index: int
    u_um: np.ndarray
    strips_gl: np.ndarray
    along: dict
    width_range: float
    depth_range: float
    centre_range: float
    depth_sd: float


def _features_rows(dips, v, q):
    rows = [dip_features(d, v, 0.0, q.window_um, q.skew_levels, q.outside_fwhm) for d in dips]
    return {name: np.array([r[name] for r in rows], dtype=float) for name in FEATURES}


def _range_over_median(x, scale):
    """(max - min) of x over the median of scale; NaN unless all are finite and the median is > 0."""
    x, scale = np.asarray(x, dtype=float), np.asarray(scale, dtype=float)
    if x.size == 0 or not (np.all(np.isfinite(x)) and np.all(np.isfinite(scale))):
        return float("nan")
    med = float(np.median(scale))
    return float((np.max(x) - np.min(x)) / med) if med > 0 else float("nan")


def screen_node(block, ks, k_star, B_gl, sigma_gl, left, top, res0_um, origin_xy, y_hat, e_u, v_um, n_avg,
                step_um, ccfg, profiles_gl=None):
    """The clearness screen of one prepared node -> ClearnessResult (the rule of the module's docstring).

    block (n_planes, H, W): the node's recorded patch, grey levels; ks (n_planes,) int: its planes; k_star: the
    first stage's sharpest plane (the nearest recorded plane is used when it is not in ks); B_gl (n_planes,):
    the backgrounds at the line's origin (gl, > 0); sigma_gl > 0: the noise SD of one line sample (gl);
    left, top, res0_um, origin_xy, y_hat, e_u, v_um, n_avg, step_um: the patch's frame and the measuring
    line, as strip_profiles; ccfg: a ClearnessConfig; profiles_gl (n_planes, n_v) or None: the measuring line
    as recorded (None: read here from block with u = 0).

    Judged: E and S of the measuring line on the planes |k - k*| <= focus_half_planes and of every line along
    the axis on the sharpest plane; U_W, U_D, U_c of the lines along the axis; the depth floor."""
    q = ccfg
    blk = np.asarray(block, dtype=float)
    k_arr = np.asarray(ks).reshape(-1)
    if blk.ndim != 3 or blk.shape[0] != k_arr.size or k_arr.size == 0:
        raise ValueError("block must be (n_planes, H, W) with one plane per entry of ks")
    B = np.array(B_gl, dtype=float, copy=True).reshape(-1)
    if B.shape != (k_arr.size,) or not np.all(np.isfinite(B)) or np.any(B <= 0):
        raise ValueError("B_gl must be (n_planes,), finite and > 0")
    sig = float(sigma_gl)
    if not (math.isfinite(sig) and sig > 0):
        raise ValueError("sigma_gl must be finite and > 0")
    v = _line(v_um)
    j = int(np.argmin(np.abs(k_arr.astype(float) - float(k_star))))
    if profiles_gl is None:
        P = np.stack([strip_profiles(blk[i], left, top, res0_um, origin_xy, y_hat, e_u, v, n_avg, step_um, [0.0])[0]
                      for i in range(k_arr.size)])
    else:
        P = np.array(profiles_gl, dtype=float, copy=True)
        if P.shape != (k_arr.size, v.size):
            raise ValueError("profiles_gl must be (n_planes, n_v) = (%d, %d)" % (k_arr.size, v.size))
    line = _features_rows(smooth_dip(B[:, None] - P, q.smooth_samples), v, q)
    focus = np.abs(k_arr.astype(int) - int(k_arr[j])) <= int(q.focus_half_planes)
    u = along_offsets(q.along_half_um, q.along_step_um)
    strips = strip_profiles(blk[j], left, top, res0_um, origin_xy, y_hat, e_u, v, n_avg, step_um, u)
    along = _features_rows(smooth_dip(B[j] - strips, q.smooth_samples), v, q)

    U_W = _range_over_median(along["fwhm_um"], along["fwhm_um"])
    U_D = _range_over_median(along["depth_gl"], along["depth_gl"])
    U_c = _range_over_median(along["centre_um"], along["fwhm_um"])
    judged_E = np.concatenate([line["outside"][focus], along["outside"]])
    judged_S = np.concatenate([line["skew"][focus], along["skew"]])

    def worst(x):
        return float(np.max(x)) if np.all(np.isfinite(x)) else float("nan")
    ratios = dict(outside=worst(judged_E) / q.outside_max, skew=worst(judged_S) / q.skew_max,
                  width_range=U_W / q.width_range_max, depth_range=U_D / q.depth_range_max,
                  centre_range=U_c / q.centre_range_max)
    reasons = [name for name in CRITERIA if math.isfinite(ratios[name]) and ratios[name] > 1.0]
    finite = all(math.isfinite(r) for r in ratios.values())
    if not finite:
        reasons.append(REASON_NOT_FINITE)
    score = max(ratios.values()) if finite else float("inf")
    depth_sd = float(line["depth_gl"][j]) / sig
    if not (math.isfinite(depth_sd) and depth_sd >= q.min_depth_sd):
        reasons.append(REASON_FAINT)
    return ClearnessResult(clean=not reasons, score=float(score), reasons=tuple(reasons), ratios=ratios, line=line,
                           focus=focus, k_index=j, u_um=u, strips_gl=strips, along=along, width_range=U_W,
                           depth_range=U_D, centre_range=U_c, depth_sd=depth_sd)
