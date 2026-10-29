"""Independent references for the spec-derived tests of Block 10.S (the stripe)
and its neighbours (tester, 2026-10-28). Written from specs/SPEC.md only:
nothing here calls the code under test.

  ref_sigma_r        Block 4's kernel law: linear interpolation of the table in
                     |delta|, the linear / frozen / proportional continuation
  ref_shadow_depth   10.I / 10.S: B int_{-r}^{r} (1 - exp(-2 mu sqrt(r^2 - v'^2)))
                     phi_sigma(rho - v') dv' by scipy.integrate.quad
  ref_margin         rho_tau: D(rho) = tau by scipy.optimize.brentq; -1.0 when
                     D(0) <= tau
  seg_dist           2-D distance from points to a segment (closed form)
  ref_patch_bounds   10.P: the pixels whose centres lie within +-half of (cx, cy)
  pixel_centres_ref  section 3 / Block 11: pixel (row, col) centred on
                     ((left + col) p, (top + row) p)
  ref_stripe         10.S's walk, written from the spec's words (columns one
                     pixel wide along v, centred on multiples of p)

Pure ASCII.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
from scipy import integrate, optimize, stats

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
for _p in (WS / "src", WS / "scripts", WS / "tests" / "smoke"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


# --------------------------------------------------------- Block 4's law ----

def ref_sigma_r(delta, rcfg):
    """sigma_r(delta) of Block 4 (section 2.3): linear interpolation of
    (kernel_table_delta_um, kernel_table_sigma_um) at |delta| <= delta_max,
    continued by kernel_continuation beyond."""
    d = np.abs(np.asarray(delta, dtype=float))
    xs = np.asarray(rcfg.kernel_table_delta_um, dtype=float)
    ys = np.asarray(rcfg.kernel_table_sigma_um, dtype=float)
    dmax, smax = xs[-1], ys[-1]
    inside = np.interp(np.minimum(d, dmax), xs, ys)
    if rcfg.kernel_continuation == "linear":
        beyond = smax + rcfg.kernel_continuation_slope * (d - dmax)
    elif rcfg.kernel_continuation == "frozen":
        beyond = np.full_like(d, smax)
    elif rcfg.kernel_continuation == "proportional":
        beyond = smax * d / dmax
    else:
        raise ValueError(rcfg.kernel_continuation)
    return np.where(d <= dmax, inside, beyond)


# ------------------------------------------------------- the shadow (10.I) --

def ref_shadow_depth(rho, r, mu, sigma, B):
    """The predicted dip (gl) at lateral distance rho (um) of a flat tube of
    radius r (um), absorption mu (1/um), blurred by a Gaussian of SD sigma (um),
    background B (gl): quad of the integral of 10.I with the substitution
    v' = r sin(psi), which removes the endpoint square root."""
    rho, r, mu, sigma, B = (float(x) for x in (rho, r, mu, sigma, B))

    def f(psi):
        vp = r * math.sin(psi)
        return ((1.0 - math.exp(-2.0 * mu * r * math.cos(psi))) * stats.norm.pdf(rho - vp, scale=sigma)
                * r * math.cos(psi))
    pts = []
    if -r < rho < r:
        pts = [math.asin(rho / r)]
    val, _err = integrate.quad(f, -0.5 * math.pi, 0.5 * math.pi, points=pts or None, epsabs=1e-14, epsrel=1e-12,
                               limit=400)
    return B * val


def ref_margin(r, mu, sigma, B, tau):
    """rho_tau: the root of D(rho) = tau on rho >= 0 (D decreasing), -1.0 when
    D(0) <= tau (10.S)."""
    d0 = ref_shadow_depth(0.0, r, mu, sigma, B)
    if not d0 > tau:
        return -1.0
    hi = r + 12.0 * sigma
    while ref_shadow_depth(hi, r, mu, sigma, B) > tau:
        hi *= 2.0
    return optimize.brentq(lambda x: ref_shadow_depth(x, r, mu, sigma, B) - tau, 0.0, hi, xtol=1e-12, rtol=1e-12)


# ------------------------------------------------------------ geometry -----

def seg_dist(px, py, a, b):
    """Distance (um) from the points (px, py) to the segment a-b (2-vectors)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = b - a
    L2 = float(d @ d)
    if L2 == 0.0:
        return np.hypot(px - a[0], py - a[1])
    t = np.clip(((px - a[0]) * d[0] + (py - a[1]) * d[1]) / L2, 0.0, 1.0)
    return np.hypot(px - (a[0] + t * d[0]), py - (a[1] + t * d[1]))


def ref_patch_bounds(cx, cy, half_um, p):
    """(left, top, width, height): the full-resolution pixels whose centres
    (index x p) lie within +-half_um of (cx, cy) in both axes (10.P). Exact
    integer arithmetic on a fine rational grid is not needed: the tests keep
    the centre away from the half-pixel ties."""
    left = math.ceil((cx - half_um) / p)
    right = math.floor((cx + half_um) / p)
    top = math.ceil((cy - half_um) / p)
    bottom = math.floor((cy + half_um) / p)
    return left, top, right - left + 1, bottom - top + 1


def pixel_centres_ref(shape, left, top, p):
    H, W = shape
    rows, cols = np.mgrid[0:H, 0:W]
    return (left + cols) * p, (top + rows) * p


# --------------------------------------------------------- the stripe -------

def ref_stripe(nb, ow, left, top, p, cx, cy, theta, widths, min_pixels, min_side):
    """10.S's stripe written from the spec: for each W of widths, the band
    |u| <= W cut into pixel columns along v (one pixel wide: column j holds
    the pixels with round(v / p) = j); from column 0 outward on each side the
    walk passes the columns the own branch shades (and no neighbour), then the
    clear columns, and stops at the first column a neighbour or the own branch
    shades or that has no band pixel (the patch's edge). N(W) = the band's
    pixels in the run columns; W* the W of largest N, the smaller on a tie.
    Returns dict(w, v_own_plus, v_own_minus, l_plus, l_minus, side_plus,
    side_minus, n, ok, reason, mask)."""
    H, Wd = nb.shape
    px, py = pixel_centres_ref((H, Wd), left, top, p)
    dx, dy = px - cx, py - cy
    v = -math.sin(theta) * dx + math.cos(theta) * dy
    u = math.cos(theta) * dx + math.sin(theta) * dy
    col = np.rint(v / p).astype(int)
    best = None
    for w in sorted(float(x) for x in widths):
        band = np.abs(u) <= w
        cols_present = set(np.unique(col[band]).tolist())
        own_cols = set(np.unique(col[band & ow]).tolist())
        nb_cols = set(np.unique(col[band & nb]).tolist())

        def walk(step):
            j = 0
            while j in cols_present and j in own_cols and j not in nb_cols:
                j += step
            j_own = j
            while j in cols_present and j not in own_cols and j not in nb_cols:
                j += step
            return j_own, j
        jo_p, je_p = walk(+1)
        jo_m, je_m = walk(-1)
        run = set(range(jo_p, je_p)) | set(range(je_m + 1, jo_m + 1))
        mask = band & np.isin(col, np.array(sorted(run), dtype=int)) & ~(nb | ow)
        n = int(mask.sum())
        rec = dict(w=w, v_own_plus=jo_p * p, v_own_minus=-jo_m * p, l_plus=je_p * p, l_minus=-je_m * p,
                   side_plus=(je_p - jo_p) * p, side_minus=(jo_m - je_m) * p, n=n, mask=mask)
        if best is None or n > best["n"]:
            best = rec
    reason = ""
    if best["n"] < min_pixels:
        reason = "too_few"
    elif min(best["side_plus"], best["side_minus"]) < min_side:
        reason = "side_short"
    best["ok"] = reason == ""
    best["reason"] = reason
    return best


def ws_path():
    return WS


def env_line():
    import platform
    import scipy
    return "python %s | numpy %s | scipy %s | %s" % (platform.python_version(), np.__version__, scipy.__version__,
                                                    platform.platform())


if __name__ == "__main__":
    print(env_line(), os.getcwd())
