"""Figure: which pixels the pilot's soma fit uses -- the region R split by where its pixels lie relative to the fitted
outline. Standalone figure script (not pipeline code).

R = the pixels within 13 um of Allen's soma centre, minus the mask (every traced process in the crop, at every depth,
at its traced radius + 2 um; the segments from the soma node to the stems' first nodes excepted), as fit_soma.py
builds it. Rebuilt here from the CORRECTED SWC of the 2026-11-24 upload (the original is not in the sandbox), with each moved
stem's inner stretch restored from the JSON (original first node to moved one), so R is approximate near the soma;
its pixel count is printed beside the run's. Categories by the distance to the fitted outline (the
ellipse a, b, phi of the fit at its centre): inside (> 1 um inside), edge (within 1 um), outside (> 1 um beyond).
Usage: python3 fig_region.py SOMA_FIT_JSON CORRECTED_SWC OUT_PNG
"""
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
from matplotlib.colors import ListedColormap                      # noqa: E402
from matplotlib.patches import Patch                              # noqa: E402

sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..", "src"))
from allen_diameter.loading import swc_io                         # noqa: E402
from allen_diameter.analysis import soma_view                     # noqa: E402

CFG = dict(widen_um=2.0, region_um=13.0, rim_um=1.0, view_half_um=15.0)
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
MASKED, OUTSIDE, EDGE, INSIDE = "#c9c8c2", "#2a78d6", "#eb6834", "#1baf7a"   # grey; categorical slots 1, 2, 3


def classify(run, swc):
    """(codes (H, W) int, extent, geometry): 0 outside the disc, 1 masked, 2 R outside, 3 R edge, 4 R inside."""
    crop, fit = run["crop"], run["fit"]
    left, top, W, H, p = crop["left"], crop["top"], crop["width"], crop["height"], crop["res0_um"]
    xyz = np.asarray(swc.xyz, dtype=float)
    index_of = {int(i): k for k, i in enumerate(swc.ids)}
    parent_index = np.array([index_of.get(int(pp), -1) for pp in np.asarray(swc.parent)])
    w = CFG["widen_um"]
    seg = soma_view.trace_segments(xyz, np.asarray(swc.radius), parent_index, np.asarray(swc.types),
                                   (left * p - w, (left + W) * p + w, top * p - w, (top + H) * p + w))
    mask = soma_view.segment_mask((H, W), left, top, p, seg["p1"], seg["p2"], seg["r1"] + w, seg["r2"] + w)
    # restore the stems' inner stretch the corrected SWC no longer has: each moved stem from its original first node
    # (old_um, inside the fitted surface) to its moved one (new_um), at the first node's traced radius -- the original
    # segments (old first node, removed nodes, child) ran along that line, the move being along the stem
    rad = {int(i): float(r) for i, r in zip(swc.ids, swc.radius)}
    st = [s for s in run.get("stems", []) if s["moved_um"] > 0]
    if st:
        p1 = np.array([s["old_um"][:2] for s in st])
        p2 = np.array([s["new_um"][:2] for s in st])
        r = np.array([rad.get(int(s["id"]), 0.5) for s in st])
        mask |= soma_view.segment_mask((H, W), left, top, p, p1, p2, r + w, r + w)
    cs = run["start"]["centre_um"]
    disc = soma_view.disk_mask((H, W), left, top, p, cs[:2], CFG["region_um"])
    x0, y0, z0, a, b, c, phi = fit["theta"][:7]
    yy, xx = np.mgrid[0:H, 0:W]
    X, Y = (left + xx) * p, (top + yy) * p
    u = (X - x0) * np.cos(phi) + (Y - y0) * np.sin(phi)
    v = -(X - x0) * np.sin(phi) + (Y - y0) * np.cos(phi)
    rho = np.sqrt((u / a) ** 2 + (v / b) ** 2)
    dist = (rho - 1.0) * np.sqrt(u ** 2 + v ** 2) / np.maximum(rho, 1e-9)       # um, approximately, to the outline
    codes = np.zeros((H, W), dtype=int)
    codes[disc & mask] = 1
    region = disc & ~mask
    codes[region & (dist > CFG["rim_um"])] = 2
    codes[region & (np.abs(dist) <= CFG["rim_um"])] = 3
    codes[region & (dist < -CFG["rim_um"])] = 4
    extent = (left * p - p / 2, (left + W) * p - p / 2, (top + H) * p - p / 2, top * p - p / 2)
    geo = dict(p=p, centre=cs, x0=x0, y0=y0, a=a, b=b, phi=phi, n_run=fit["n_samples"] / len(fit["ks"]),
               sil=rho < 1.0)
    return codes, extent, geo


def main(json_path, swc_path, out_png):
    run = json.load(open(json_path))
    swc = swc_io.read_swc(swc_path)
    codes, extent, g = classify(run, swc)
    p2 = g["p"] ** 2
    n = {k: int(np.sum(codes == k)) for k in range(5)}
    nR = n[2] + n[3] + n[4]
    disc_um2 = (n[1] + nR) * p2
    sil_in_R = int(np.sum(g["sil"] & (codes >= 2)))
    sil_n = int(np.sum(g["sil"]))
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "text.color": INK})
    fig = plt.figure(figsize=(12.4, 5.6), facecolor=SURFACE)
    ax = fig.add_axes([0.05, 0.11, 0.42, 0.78])
    cmap = ListedColormap([SURFACE, MASKED, OUTSIDE, EDGE, INSIDE])
    ax.imshow(codes, cmap=cmap, vmin=-0.5, vmax=4.5, extent=extent, interpolation="nearest")
    t = np.linspace(0, 2 * np.pi, 400)
    ex = g["x0"] + g["a"] * np.cos(t) * np.cos(g["phi"]) - g["b"] * np.sin(t) * np.sin(g["phi"])
    ey = g["y0"] + g["a"] * np.cos(t) * np.sin(g["phi"]) + g["b"] * np.sin(t) * np.cos(g["phi"])
    ax.plot(ex, ey, color=INK, lw=1.6)
    cx, cy = g["centre"][:2]
    ax.plot(cx + CFG["region_um"] * np.cos(t), cy + CFG["region_um"] * np.sin(t), color=INK2, lw=1.0, ls="--")
    ax.plot([cx], [cy], marker="+", color=INK, ms=9, mew=1.4)
    h = CFG["view_half_um"]
    ax.set_xlim(cx - h, cx + h)
    ax.set_ylim(cy + h, cy - h)
    ax.set_aspect("equal")
    ax.set_xlabel("x (um, image frame)")
    ax.set_ylabel("y (um)")
    ax.set_title("(a) the fit's pixels R, by where they lie", loc="left", fontsize=11)
    ax.text(cx + 0.75 * h, cy + 0.93 * h, "5 um", ha="center", va="bottom", fontsize=9, color=INK)
    ax.plot([cx + 0.75 * h - 2.5, cx + 0.75 * h + 2.5], [cy + 0.96 * h] * 2, color=INK, lw=2.5)
    handles = [Patch(color=OUTSIDE, label="R, more than 1 um outside the outline: %d %%" % round(100 * n[2] / nR)),
               Patch(color=EDGE, label="R, within 1 um of the outline: %d %%" % round(100 * n[3] / nR)),
               Patch(color=INSIDE, label="R, more than 1 um inside: %d %%" % round(100 * n[4] / nR)),
               Patch(color=MASKED, label="masked: Allen's traced processes (+2 um), every depth"),
               plt.Line2D([], [], color=INK, lw=1.6, label="the fitted outline (a %.1f, b %.1f um)" % (g["a"], g["b"])),
               plt.Line2D([], [], color=INK2, lw=1.0, ls="--", label="the 13 um disc about Allen's soma centre (+)")]
    leg = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.53, 0.90), frameon=False, fontsize=9.5)
    for txt in leg.get_texts():
        txt.set_color(INK)
    # (b) where the disc's area, R's pixels and the outline's area go
    bx = fig.add_axes([0.62, 0.12, 0.28, 0.40])
    rows = [("13 um disc\n(%.0f um^2)" % disc_um2,
             [(n[1], MASKED, "masked"), (n[2], OUTSIDE, ""), (n[3], EDGE, ""), (n[4], INSIDE, "")], n[1] + nR,
             "R %d %%" % round(100 * nR / (n[1] + nR))),
            ("R\n(%.0f um^2)" % (nR * p2),
             [(n[2], OUTSIDE, "outside"), (n[3], EDGE, ""), (n[4], INSIDE, "")], nR,
             "edge %d %%\ninside %d %%" % (round(100 * n[3] / nR), round(100 * n[4] / nR))),
            ("soma outline\n(%.0f um^2)" % (sil_n * p2),
             [(sil_n - sil_in_R, MASKED, "masked"), (sil_in_R, INSIDE, "")], sil_n,
             "in R %d %%" % round(100 * sil_in_R / sil_n))]
    for i_row, (name, parts, tot, note) in enumerate(rows):
        left0 = 0.0
        for val, col, lab in parts:
            frac = val / tot
            bx.barh(i_row, frac, left=left0, color=col, edgecolor=SURFACE, linewidth=2, height=0.62)
            if lab:
                bx.text(left0 + frac / 2, i_row, "%s %d %%" % (lab, round(100 * frac)), ha="center", va="center",
                        fontsize=9, color=INK if col == MASKED else "#ffffff")
            left0 += frac
        if note:
            bx.text(1.01, i_row, note, ha="left", va="center", fontsize=9, color=INK)
    bx.set_yticks(range(len(rows)))
    bx.set_yticklabels([r[0] for r in rows])
    bx.invert_yaxis()
    bx.set_xlim(0, 1)
    bx.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    bx.set_xticklabels(["0", "25 %", "50 %", "75 %", "100 %"])
    for s in ("top", "right"):
        bx.spines[s].set_visible(False)
    bx.set_title("(b) where the area goes", loc="left", fontsize=11)
    fig.suptitle("Most pixels the soma fit uses are background around the soma (R rebuilt: %d px per plane; "
                 "the run's %d)" % (nR, g["n_run"]), x=0.05, ha="left", fontsize=11.5)
    fig.savefig(out_png, dpi=150, facecolor=SURFACE)
    print("R %d px (%.1f um^2): outside %.1f %%, edge %.1f %%, inside %.1f %%; disc %.0f um^2 masked %.1f %%; "
          "outline %.1f um^2, %.1f %% in R" % (nR, nR * p2, 100 * n[2] / nR, 100 * n[3] / nR, 100 * n[4] / nR,
                                               disc_um2, 100 * n[1] / (n[1] + nR), sil_n * p2,
                                               100 * sil_in_R / sil_n))


if __name__ == "__main__":
    main(*sys.argv[1:4])
