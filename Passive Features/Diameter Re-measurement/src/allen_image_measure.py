"""
allen_image_measure.py -- measurement on image arrays. No IO, no plotting.

Everything here takes an array plus a micron-per-pixel scale and returns
numbers, so it can be unit-tested on synthetic bars of known width.

Convention: these images are BRIGHTFIELD -- the DAB-filled neuron is DARK on a
bright background. Every function below works on "darkness", defined as
    d(p) = background - I(p)
with the background estimated locally from the ends of the profile, so a wider
dendrite means a taller peak. Do not pass raw intensity.

Known duplication: allen_stack_radius_refit.py carries its own _fwhm with the
same definition. They should be unified once the stack refit moves into this
package; until then, changing one means changing the other.
"""
import math

import numpy as np
from scipy import ndimage

MIN_DARKNESS = 12.0        # 8-bit levels below which a profile is not trusted
BG_ENDS_PX = 5             # how many pixels at each end define the background


def line_profile(img, p0_colrow, p1_colrow, res_um_px, n=None):
    """Interpolated intensity along the segment p0 -> p1, both (col, row) in
    ARRAY coordinates. Returns (t_um, intensity), t measured from p0."""
    (c0, r0), (c1, r1) = p0_colrow, p1_colrow
    length_px = math.hypot(c1 - c0, r1 - r0)
    if n is None:
        n = max(8, int(round(length_px * 2)) + 1)      # ~2 samples per pixel
    cols = np.linspace(c0, c1, n)
    rows = np.linspace(r0, r1, n)
    vals = ndimage.map_coordinates(img.astype(np.float32), [rows, cols], order=1,
                                   mode="nearest")
    t_um = np.linspace(0.0, length_px * res_um_px, n)
    return t_um, vals


def perpendicular_profile(img, centre_colrow, tangent_colrow, res_um_px,
                          half_len_um=3.0):
    """Profile ACROSS a process: sampled along the in-plane normal of the given
    tangent, centred on `centre_colrow`. This is the quantity whose FWHM is the
    diameter. Returns (t_um, intensity) with t = 0 at the centre."""
    tc, tr = tangent_colrow
    norm = math.hypot(tc, tr)
    if norm < 1e-9:
        raise ValueError("tangent has zero length; cannot define a normal")
    nc, nr = -tr / norm, tc / norm                    # in-plane normal
    half_px = half_len_um / res_um_px
    c, r = centre_colrow
    p0 = (c - nc * half_px, r - nr * half_px)
    p1 = (c + nc * half_px, r + nr * half_px)
    t_um, vals = line_profile(img, p0, p1, res_um_px)
    return t_um - t_um[-1] / 2.0, vals


def to_darkness(intensity):
    """Background-subtracted darkness, background from the profile's ends."""
    v = np.asarray(intensity, dtype=np.float64)
    k = min(BG_ENDS_PX, max(1, len(v) // 6))
    bg = np.median(np.r_[v[:k], v[-k:]])
    return bg - v


def fwhm_um(t_um, intensity):
    """Full width at half maximum of the darkness profile, in um.

    Returns (width_um, info). width_um is NaN when the profile is too faint or
    never returns to background on both sides -- both of which are real,
    reportable outcomes, not errors to be silently filled in.
    """
    t = np.asarray(t_um, dtype=np.float64)
    d = to_darkness(intensity)
    peak = float(d.max())
    info = dict(peak_darkness=peak, reason="")
    if peak < MIN_DARKNESS:
        info["reason"] = "faint (peak %.1f < %.1f levels)" % (peak, MIN_DARKNESS)
        return float("nan"), info
    c = int(np.argmax(d))
    half = peak / 2.0
    info["half_level"] = half
    li = c
    while li > 0 and d[li] > half:
        li -= 1
    ri = c
    while ri < len(d) - 1 and d[ri] > half:
        ri += 1
    if li == 0 or ri == len(d) - 1:
        info["reason"] = "profile never returns to background within the window"
        return float("nan"), info
    # linear interpolation of the two half-crossings
    tl = t[li] + (half - d[li]) * (t[li + 1] - t[li]) / (d[li + 1] - d[li] + 1e-12)
    tr = t[ri] - (half - d[ri]) * (t[ri] - t[ri - 1]) / (d[ri - 1] - d[ri] + 1e-12)
    info.update(t_left=float(tl), t_right=float(tr), t_peak=float(t[c]))
    return float(tr - tl), info


def estimate_tangent(img, centre_colrow, res_um_px, search_um=2.0, n_angles=72):
    """Crude local orientation: the angle whose PERPENDICULAR profile is
    narrowest is the one lying along the process. Returns (tangent_col,
    tangent_row) as a unit vector. Useful when clicking a point by eye."""
    best = (float("inf"), (1.0, 0.0))
    for a in np.linspace(0, math.pi, n_angles, endpoint=False):
        tan = (math.cos(a), math.sin(a))
        try:
            t, v = perpendicular_profile(img, centre_colrow, tan, res_um_px, search_um)
            w, _ = fwhm_um(t, v)
        except ValueError:
            continue
        if np.isfinite(w) and w < best[0]:
            best = (w, tan)
    return best[1]


def centre_of_dark_mass(img, percentile=99.0):
    """(col, row) of the centroid of the darkest `100-percentile`% of pixels.

    On a projection of a filled neuron this lands on the cell, which saves
    guessing crop coordinates by eye. Returns floats in ARRAY coordinates.
    """
    a = np.asarray(img, dtype=np.float64)
    dark = a.max() - a
    thr = np.percentile(dark, percentile)
    rows, cols = np.nonzero(dark >= thr)
    if len(rows) == 0:
        return (a.shape[1] / 2.0, a.shape[0] / 2.0)
    w = dark[rows, cols]
    return (float(np.average(cols, weights=w)), float(np.average(rows, weights=w)))
