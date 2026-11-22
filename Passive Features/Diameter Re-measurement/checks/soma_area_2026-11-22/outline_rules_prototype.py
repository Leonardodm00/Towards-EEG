"""Prototype (exploration, not pipeline code): two outline rules per plane on the pilot's soma."""
import json, sys
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open(sys.argv[1]))
z = np.load(sys.argv[2], allow_pickle=False)
B, ks, left, top, p = z["block"].astype(float), list(z["ks"]), int(z["left"]), int(z["top"]), float(z["res0_um"])
cx, cy, cz = d["centre_um"]
r = d["radius_um"]
roots = np.array(d["primary_dendrite_roots_um"]) - np.array([cx, cy])
c_col, c_row = cx / p - left, cy / p - top                  # the centre in pixel coordinates of the crop
SMOOTH_UM, NTH, RHO = 0.3, 120, np.linspace(0.3 * r, 1.8 * r, 200)
TH = np.linspace(0, 2 * np.pi, NTH, endpoint=False)
root_th = np.arctan2(roots[:, 1], roots[:, 0])
STEM_HALF = np.deg2rad(14.0)
cut = np.zeros(NTH, bool)
for t in root_th:
    cut |= np.abs(np.angle(np.exp(1j * (TH - t)))) < STEM_HALF


def interp_cut(rad):
    """fill the stem sectors by periodic linear interpolation"""
    good = ~cut & np.isfinite(rad)
    th_g = np.concatenate([TH[good] - 2 * np.pi, TH[good], TH[good] + 2 * np.pi])
    r_g = np.tile(rad[good], 3)
    return np.interp(TH, th_g, r_g)


def profiles(img):
    S = gaussian_filter(img, SMOOTH_UM / p)
    X = c_col + (RHO[None, :] * np.cos(TH)[:, None]) / p
    Y = c_row + (RHO[None, :] * np.sin(TH)[:, None]) / p
    return map_coordinates(S, [Y.ravel(), X.ravel()], order=1).reshape(NTH, RHO.size)


def edge_rule(P):
    g = np.gradient(P, RHO, axis=1)                        # gl per um, positive where it brightens outward
    j = np.argmax(g, axis=1)
    return interp_cut(RHO[j]), g[np.arange(NTH), j]


def half_rule(P, img):
    inner = P[:, RHO <= 0.6 * r]
    lo = np.percentile(inner, 10)
    ring = P[:, RHO >= 1.5 * r]
    hi = np.median(ring[~cut])
    lev = 0.5 * (lo + hi)
    rad = np.full(NTH, np.nan)
    for i in range(NTH):
        above = np.flatnonzero(P[i] >= lev)
        above = above[RHO[above] >= 0.5 * r]               # the first outward crossing beyond half the traced radius
        if above.size:
            rad[i] = RHO[above[0]]
    return interp_cut(rad), lev


def area_perim(rad):
    dth = 2 * np.pi / NTH
    A = 0.5 * np.sum(rad ** 2) * dth
    drd = np.gradient(np.concatenate([rad[-1:], rad, rad[:1]]), dth)[1:-1]
    P = np.sum(np.sqrt(rad ** 2 + drd ** 2)) * dth
    return A, P


sel = list(range(99, 142, 2))
rows = []
fig, axes = plt.subplots(4, 6, figsize=(18, 12.5), layout="constrained")
win_lo, win_hi = 55, 100
for a, k in zip(axes.ravel(), sel):
    j = ks.index(k)
    P = profiles(B[j])
    re, gmax = edge_rule(P)
    rh, lev = half_rule(P, B[j])
    Ae, Pe = area_perim(re)
    Ah, Ph = area_perim(rh)
    rows.append((k, Ae, Pe, Ah, Ph, np.median(gmax[~cut])))
    H, W = B[j].shape
    x = (left + np.arange(W)) * p - cx
    y = (top + np.arange(H)) * p - cy
    a.imshow(B[j], cmap="gray", vmin=win_lo, vmax=win_hi, extent=(x[0], x[-1], y[-1], y[0]), interpolation="nearest")
    for rad, col, ls in ((re, "#3a86ff", "-"), (rh, "#ef6f3c", "-")):
        a.plot(np.append(rad, rad[0]) * np.cos(np.append(TH, TH[0])), np.append(rad, rad[0]) * np.sin(np.append(TH, TH[0])), ls, color=col, lw=1.0)
    t = np.linspace(0, 2 * np.pi, 181)
    a.plot(r * np.cos(t), r * np.sin(t), "--", color="#ffd23f", lw=0.7)
    a.plot(roots[:, 0], roots[:, 1], "o", mfc="none", mec="#ffd23f", ms=3)
    a.set_xlim(-10, 10); a.set_ylim(10, -10); a.tick_params(labelsize=6)
    a.set_title("k = %d: A edge %.0f, half %.0f um2" % (k, Ae, Ah), fontsize=8.5)
for a in axes.ravel()[len(sel):]:
    a.set_axis_off()
fig.suptitle("prototype on the pilot: blue = steepest rise along rays, orange = half-depth contour; dashed = Allen's "
             "sphere (r 5.97); stem sectors +-14 deg about the traced roots interpolated")
fig.savefig(sys.argv[3], dpi=95)
print(" k    A_edge  P_edge  A_half  P_half  median max-gradient (gl/um)")
for row in rows:
    print("%3d  %6.1f  %6.1f  %6.1f  %6.1f  %6.1f" % row)
print("Allen's sphere: equatorial area %.1f um2, circumference %.1f um" % (np.pi * r * r, 2 * np.pi * r))
