#!/usr/bin/env python3
"""Plane-to-plane evaluations around measured nodes, the user's proposals of
2026-10-08 -- Block 11 in specs/SPEC.md. Diagnostics; nothing here changes the
measurement.

--evaluation profile (default; the proposal of 16:26): in every plane
k_swc - H .. k_swc + H, the profile I_k(v) along the node's measuring line (the
line the focus scores use: through the SWC node, across the node's fitted
heading, |v| <= --profile-half-um, bilinear), its area A_k = int I_k dv, and
dA = A_{k+1} - A_k. Beside it, the same areas with each plane divided by its
own background B_k and multiplied by the planes' mean background, so that a
plane-wide change of brightness does not enter: B_k is the interquartile mean
(25th-75th percentile) of the block's pixels farther than Allen's radius +
--bg-margin-um from every traced dendrite segment in the block and from the
soma; the margin shrinks (3, 2, 1, 0.5 um) until 5 % of the block's pixels
are left.
The plane of the smallest area is framed (blue: as measured; violet:
normalised).

--evaluation image (the proposal of 15:45): the positive part of
I_{k+1} - I_k over the pixels of the square, its mean S+ and the mean S- of
the negative part (focus.plane_differences), with the dip of S+.

--evaluation entropy (the proposal of 17:07): in every plane, the Shannon
entropy (bits) of the grey-level histogram, one bin per grey level
(focus.plane_entropies), of two sample sets: the bilinear samples along the
measuring line (the profile of the profile evaluation), and the pixels of the
strip centred on the line: within the line's half-length across the branch
and within S = --stripe-half-um (default 1 um) along it (profiles.stripe_mask).
The plane each entropy curve picks is framed (blue: line; amber: strip):
--entropy-pick min, its global minimum (the default since 2026-10-08,
evening: on real trunks the curves have no W shape), or dip, the minimum
between its two largest maxima (focus.difference_dip; the W of thin tubes,
whose far planes fade into the noise). Writes planeentropy_<id>.png and
planeentropy_<specimen>.json instead of planediff_*.

--evaluation gradient (D-040, the user's request of 2026-10-09, 16:19): the
gradient energy G = B_k^-2 * integral of (dI~/dv)^2 over the whole line
(focus.plane_gradient_energies), with B_k each plane's background of the
profile evaluation, on the lines of half-length --grad-lines-um (3, 5) and on
the line sized from the node's fitted diameter, --line-mult x d_hat / 2 (2:
[-d_hat, d_hat]); each line frames its maximum. Writes planegrad_*.
--evaluation blend (D-040): on that d-line, G, the strip's entropy and their
min-max blend J = w g + (1 - w) eta, w = 1 / (1 + exp((d_hat - d0) / s)) with
d0 = --sigmoid-d0-um (1.5) and s = --sigmoid-s-um (0.3); frames the planes G,
J and the entropy pick. Writes planeblend_*. The planes of both figures are
drawn without the line. --nodes bydiameter --pilot-csv takes the nodes from
bins of the pilot's d_hat (--dhat-bins, --per-bin, --add-nodes).

Each node is located as the pilot measures it (survey.node_planes: the same
stretch and block request, so with the pilot's image cache its planes come
from disk); the planes outside the pilot's range are fetched, one crop each.
The square: +-half_um about the SWC node (default block_half_um, 5 um),
widened to the measuring line's half-length + 2 pixels for the profile and
entropy evaluations; a square reaching beyond the pilot's block is fetched as
such (one new crop per plane). The records of the profile and entropy
evaluations carry the line's offsets and every plane's profile (v_um,
profiles). Writes, in --out-dir: planediff_<id>.png and
planediff_<specimen>.json. --hide-line (profile and entropy evaluations)
draws the planes without the measuring line and the strip's outline, so that
the focus can be judged by eye (the user's request of 2026-10-09); the frames
stay, and nothing measured changes.

Example (Colab, after Cells 0 and 1)
    python scripts/plane_differences.py --specimen 529878215 --nodes 2,3 \
        --cache-dir /content/drive/MyDrive/allen_cache --out-dir /content/drive/MyDrive/diameters/pilot/planediff
    python scripts/plane_differences.py --specimen 529878215 --nodes 2,3 --evaluation entropy \
        --cache-dir /content/drive/MyDrive/allen_cache --out-dir /content/drive/MyDrive/diameters/pilot/planeentropy

Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
for p in (SRC, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from allen_diameter.analysis import background  # noqa: E402
from allen_diameter.analysis import cell  # noqa: E402
from allen_diameter.analysis import focus  # noqa: E402
from allen_diameter.analysis import profiles  # noqa: E402
from allen_diameter.analysis import survey  # noqa: E402
from build_table import load_config  # noqa: E402

# (key, colour, linestyle, short tag on the panel, legend label); the order is the order of precedence of a frame
_MARKS_PIPELINE = (("k_star", "#ff1744", "-", "k*", "k* (gradient energy, D-030)"),
                   ("k_star_depth", "#7a7a7a", "--", "dip", "plane of the dip depth"),
                   ("k_swc", "#1a1a19", ":", "SWC", "the SWC node's plane"))
_MARKS_AREA = (("k_min_area", "#2a78d6", "-", "minA", "smallest area under the profile, as measured"),
               ("k_min_area_norm", "#4a3aa7", "--", "minA/B", "smallest area, background-normalised"))
_PICK_WORDS = {"min": "lowest entropy", "dip": "entropy dip (between its two largest maxima)"}
_PREFIX = {"profile": "planediff", "image": "planediff", "entropy": "planeentropy", "gradient": "planegrad",
           "blend": "planeblend"}


def _marks_entropy(pick):
    w = _PICK_WORDS[pick]
    return (("k_h_line", "#2a78d6", "-", "Hl", w + " along the measuring line"),
            ("k_h_strip", "#eda100", "--", "Hs", w + " in the strip"))


def entropy_pick(S, rule="min"):
    """Index of the plane an entropy curve S (n,) picks: rule "min", its
    smallest finite value (the first of equal ones; the default since
    2026-10-08, evening, on real trunk curves that have no W shape); rule
    "dip", focus.difference_dip (the minimum between the two largest maxima,
    for the W of thin tubes whose far planes fade into the noise). None when S
    has no finite value."""
    if rule not in _PICK_WORDS:
        raise ValueError("entropy pick must be one of %s, got %r" % (", ".join(sorted(_PICK_WORDS)), rule))
    S = np.asarray(S, dtype=float)
    if not np.isfinite(S).any():
        return None
    return int(np.nanargmin(S)) if rule == "min" else focus.difference_dip(S)


def node_stack(swc, provider, cfg, node_id, transform=None, planes_half=6, half_um=None, band_um=0.0,
               line_half_um=None, d_line_mult=None):
    """The planes k_swc - planes_half .. k_swc + planes_half of node node_id's
    block, cropped to the square +-half um about the node, with what the
    evaluations need: the pixel mask (all True, or within band_um of the traced
    stretch when band_um > 0), the crop's frame and extent, the node's xy and
    heading, the stretch's traced path and radii. half = half_um (default
    block_half_um), raised to line_half_um + 2 pixels when a measuring line of
    that half-length must fit in the square, and to d_line_mult * d_hat / 2 + 2
    pixels for the line sized from the node's fitted diameter (D-040; when
    d_hat is finite and positive). The square is cut from the
    pilot's block (from the cache on real data) when it lies inside it, and
    requested from the provider otherwise (new crops on real data); the pilot's
    block is returned as well, for the background levels. ValueError for a
    node survey.node_planes refuses."""
    from allen_image_io import CropFrame
    from allen_diameter.plotting.figures import image_extent_um
    pl = survey.node_planes(swc, provider, cfg, node_id, transform)
    frame, (H, W) = pl["frame"], pl["block"].shape[1:]
    k_swc = int(pl["k_swc"])
    k_lo, k_hi = k_swc - int(planes_half), k_swc + int(planes_half)
    block, ks, valid, _ = provider(frame.left, frame.top, W, H, k_lo, k_hi)
    p = float(frame.res_um_px)
    half = float(cfg.measure.block_half_um if half_um is None else half_um)
    if line_half_um is not None:
        half = max(half, float(line_half_um) + 2.0 * p)
    if d_line_mult is not None:
        d = float(pl["result"].d_hat_um)
        if math.isfinite(d) and d > 0:
            half = max(half, 0.5 * float(d_line_mult) * d + 2.0 * p)
    o = np.asarray(pl["branch"].xyz_um[pl["index"], :2], dtype=float)
    c0 = int(math.ceil((o[0] - half) / p - frame.left))
    c1 = int(math.floor((o[0] + half) / p - frame.left)) + 1
    r0 = int(math.ceil((o[1] - half) / p - frame.top))
    r1 = int(math.floor((o[1] + half) / p - frame.top)) + 1
    if c1 - c0 < 2 or r1 - r0 < 2:
        raise ValueError("node %d: the square +-%g um about the node holds fewer than 2 x 2 pixels" % (node_id, half))
    fetched = not (c0 >= 0 and r0 >= 0 and c1 <= W and r1 <= H)
    if fetched:
        square, ks_sq, valid_sq, _ = provider(int(frame.left + c0), int(frame.top + r0), c1 - c0, r1 - r0, k_lo, k_hi)
        if not np.array_equal(np.asarray(ks_sq), np.asarray(ks)):
            raise ValueError("node %d: the square's planes differ from the block's" % node_id)
        stack = np.asarray(square, dtype=float)
        valid = np.asarray(valid, dtype=bool) & np.asarray(valid_sq, dtype=bool)
    else:
        stack = np.asarray(block[:, r0:r1, c0:c1], dtype=float)
    sub = CropFrame(int(frame.left + c0), int(frame.top + r0), 0, p)
    xyz, radius = pl["branch"].xyz_um, np.asarray(pl["branch"].radius_um, dtype=float)
    som = np.asarray(swc.types) == 1
    soma = np.column_stack([cell.to_image_um(swc.xyz[som], cfg.acquisition.res0_um, **(transform or {}))[:, :2],
                            np.asarray(swc.radius, dtype=float)[som]]) if som.any() else np.empty((0, 3))
    if band_um and band_um > 0:
        mask = background.mask_near_branch(stack.shape[1:], sub.left, sub.top, p, xyz, np.zeros(xyz.shape[0]),
                                           float(band_um))
    else:
        mask = np.ones(stack.shape[1:], dtype=bool)
    return dict(stack=stack, ks=np.asarray(ks), valid=np.asarray(valid, dtype=bool), mask=mask, frame=sub,
                extent=image_extent_um(sub, stack.shape[1:]), k_swc=k_swc, result=pl["result"], o=o,
                theta=float(pl["result"].theta_rad), xyz=xyz, radius=radius, block=np.asarray(block, dtype=float),
                block_frame=frame, segments=np.asarray(pl["segments"], dtype=float), soma=soma,
                square_half_um=half, square_fetched=bool(fetched), allen_radius_um=float(pl["allen_radius_um"]),
                node_type=int(np.asarray(swc.types)[np.flatnonzero(np.asarray(swc.ids) == int(node_id))[0]]))


def far_from_dendrites(shape, frame, segments, margin_um, discs=()):
    """(H, W) bool: True where a pixel centre is farther than max(r0, r1) +
    margin_um from every traced dendrite segment (the rows of
    survey.node_planes' segments: x0, y0, z0, x1, y1, z1, r0, r1, in_stretch)
    and farther than r + margin_um from every disc (x, y, r) (the soma)."""
    H, W = int(shape[0]), int(shape[1])
    p = float(frame.res_um_px)
    rows, cols = np.mgrid[0:H, 0:W]
    px, py = (frame.left + cols) * p, (frame.top + rows) * p
    near = np.zeros((H, W), dtype=bool)
    for s in np.atleast_2d(segments):
        if s.size < 8:
            continue
        near |= background.segment_distance(px, py, s[0:2], s[3:5]) <= max(s[6], s[7]) + margin_um
    for x, y, r in np.atleast_2d(np.asarray(discs, dtype=float)).reshape(-1, 3):
        near |= np.hypot(px - x, py - y) <= r + margin_um
    return ~near


def plane_background(block, valid, far, min_frac=0.05):
    """B_k (n,): the interquartile mean (25th-75th percentile) of each valid
    plane's pixels in `far`; NaN when fewer than min_frac of the pixels are far."""
    B = np.full(block.shape[0], np.nan)
    if far.mean() < min_frac:
        return B
    for k in range(block.shape[0]):
        if valid[k]:
            x = np.sort(block[k][far])
            lo, hi = int(np.floor(0.25 * x.size)), int(np.ceil(0.75 * x.size))
            B[k] = float(x[lo:max(hi, lo + 1)].mean())
    return B


def line_profiles(st, cfg, profile_half_um=None):
    """The node's measuring line (through the node, across its fitted heading,
    |v| <= h, h = profile_half_um or the measurement's) and the bilinear
    profile I_k(v) on it in every valid plane of node_stack's square (NaN rows
    for the invalid planes). Returns (h, v, y_hat, e_u, prof (n, M))."""
    m = cfg.measure
    h = float(m.profile_half_um if profile_half_um is None else profile_half_um)
    v = profiles.profile_offsets(dataclasses.replace(m, profile_half_um=h))
    th = st["theta"]
    y_hat, e_u = np.array([-math.sin(th), math.cos(th)]), np.array([math.cos(th), math.sin(th)])
    stack, valid, fr = st["stack"], st["valid"], st["frame"]
    prof = np.full((stack.shape[0], v.size), np.nan)
    for k in range(stack.shape[0]):
        if valid[k]:
            prof[k] = profiles.sample_profile(stack[k], fr, st["o"], y_hat, e_u, v)
    return h, v, y_hat, e_u, prof


def block_background(st, bg_margin_um=4.0):
    """Each plane's background B_k (the profile evaluation's, D-036): plane_background
    over the pilot block's pixels farther than Allen's radius + the margin from
    every traced dendrite segment and from the soma. The margin shrinks
    (bg_margin_um, then 3, 2, 1, 0.5 um below it) until 5 % of the block's
    pixels are far: next to the soma a wide margin leaves no pixel. Returns
    (B (n,), the far fraction, the margin used or None when no plane has a
    background)."""
    margins = [float(bg_margin_um)] + [x for x in (3.0, 2.0, 1.0, 0.5) if x < float(bg_margin_um)]
    for margin in margins:
        far = far_from_dendrites(st["block"].shape[1:], st["block_frame"], st["segments"], margin, st["soma"])
        if far.mean() >= 0.05:
            break
    B = plane_background(st["block"], st["valid"], far)
    return B, float(far.mean()), (margin if np.isfinite(B).any() else None)


def d_line_half_um(d_hat_um, mult):
    """Half-length of the measuring line sized from the fitted diameter (D-040):
    mult * d_hat / 2, so that mult = 2 gives the line [-d_hat, d_hat].
    ValueError unless d_hat and mult are finite and positive."""
    d, m = float(d_hat_um), float(mult)
    if not (math.isfinite(d) and d > 0 and math.isfinite(m) and m > 0):
        raise ValueError("the d-line needs a finite positive d_hat and multiplier, got d_hat=%r, mult=%r"
                         % (d_hat_um, mult))
    return 0.5 * m * d


def profile_evaluation(st, cfg, profile_half_um=None, bg_margin_um=4.0):
    """Profiles along the node's measuring line in every plane, their areas as
    measured and background-normalised, and each plane's background B_k."""
    h, v, y_hat, e_u, prof = line_profiles(st, cfg, profile_half_um)
    valid = st["valid"]
    B, bg_frac, margin_used = block_background(st, bg_margin_um)
    area = focus.profile_areas(prof, v, valid)["area"]
    ok = valid & np.isfinite(B) & (B > 0)
    area_norm = np.full_like(area, np.nan)
    if ok.any():
        scale = np.where(ok, np.nanmean(B[ok]) / np.where(ok, B, 1.0), np.nan)
        area_norm = focus.profile_areas(prof * scale[:, None], v, ok)["area"]
    line = (st["o"] - h * y_hat, st["o"] + h * y_hat)
    return dict(v=v, prof=prof, area=area, area_norm=area_norm, background=B, line=line, half_um=h,
                bg_frac=bg_frac, bg_margin_used=margin_used)


def entropy_evaluation(st, cfg, profile_half_um=None, stripe_half_um=1.0, bin_width=1.0):
    """The entropies of the grey-level histograms, plane by plane, of the
    samples along the node's measuring line and of the pixels of its strip
    (|v| <= h across the branch, |u| <= stripe_half_um along it, in the square
    of node_stack), with the histograms themselves and the strip's outline.
    ValueError when the strip holds no pixel of the square."""
    if not stripe_half_um > 0:
        raise ValueError("stripe_half_um must be > 0, got %r" % (stripe_half_um,))
    h, v, y_hat, e_u, prof = line_profiles(st, cfg, profile_half_um)
    stack, valid, fr, o = st["stack"], st["valid"], st["frame"], st["o"]
    strip = profiles.stripe_mask(stack.shape[1:], fr, o, y_hat, e_u, h, stripe_half_um)
    if not strip.any():
        raise ValueError("the strip (+-%g um across, +-%g um along) holds no pixel of the square" % (h, stripe_half_um))
    ent = focus.plane_entropies(prof, stack, strip, valid, bin_width)
    hist_line = [focus.grey_histogram(prof[k], bin_width) if np.isfinite(ent["h_line"][k]) else None
                 for k in range(stack.shape[0])]
    hist_strip = [focus.grey_histogram(stack[k][strip], bin_width) if np.isfinite(ent["h_strip"][k]) else None
                  for k in range(stack.shape[0])]
    corners = np.array([o + a * h * y_hat + c * float(stripe_half_um) * e_u
                        for a, c in ((-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1))])
    ent.update(v=v, prof=prof, line=(o - h * y_hat, o + h * y_hat), half_um=h, stripe_half_um=float(stripe_half_um),
               bin_width=float(bin_width), strip=strip, outline=corners, hist_line=hist_line, hist_strip=hist_strip)
    return ent


# the colours of the gradient energy's fixed lines, in order (at most three, each its own colour), and of the d-line
_GRAD_FIXED_COLOURS = (("#2a78d6", "o-", "-"), ("#4a3aa7", "s--", "--"), ("#008300", "D:", ":"))
_GRAD_D_COLOUR = ("#1baf7a", "^-", "-.")


def _d_line_or_error(st, cfg, line_mult):
    """(d_hat, h) for the line sized from the fitted diameter, h = line_mult * d_hat / 2; ValueError when d_hat is
    unusable or the line would hold fewer than 3 samples."""
    d = float(st["result"].d_hat_um)
    h = d_line_half_um(d, line_mult)
    if int(round(h / float(cfg.measure.profile_step_um))) < 1:
        raise ValueError("the d-line (+-%.3g um) holds fewer than 3 samples" % h)
    return d, h


def fixed_lines_um(lines_um):
    """The gradient evaluation's fixed half-lengths as floats: at most three
    (one colour each), distinct, finite and positive; none is allowed (the
    d-line alone). ValueError otherwise."""
    out = [float(h) for h in lines_um]
    if len(out) > len(_GRAD_FIXED_COLOURS) or len(set(out)) != len(out) or \
            any(not (math.isfinite(h) and h > 0) for h in out):
        raise ValueError("lines_um: at most %d distinct positive half-lengths, got %r" % (len(_GRAD_FIXED_COLOURS),
                                                                                          list(lines_um)))
    return out


def gradient_evaluation(st, cfg, lines_um=(3.0, 5.0), line_mult=2.0, bg_margin_um=4.0):
    """The gradient energy over the whole line, plane by plane
    (focus.plane_gradient_energies; D-040), for the lines of half-length
    lines_um and for the line sized from the node's fitted diameter, h_d =
    line_mult * d_hat / 2 (key "Gd"), every line with each plane's background
    of block_background. A node without a usable d-line keeps the fixed lines,
    and d_line_note says why; ValueError when no line is left. Returns
    dict(lines=[dict(key, half_um, v, prof, G)], background, bg_frac,
    bg_margin_used, d_hat_um, d_line_half_um or None, d_line_note or None)."""
    lines_um = fixed_lines_um(lines_um)
    B, bg_frac, margin_used = block_background(st, bg_margin_um)
    specs, note, hd = [("G%g" % h, h) for h in lines_um], None, None
    try:
        hd = _d_line_or_error(st, cfg, line_mult)[1]
        specs.append(("Gd", hd))
    except ValueError as e:
        note = str(e)
    if not specs:
        raise ValueError("no line to evaluate: no fixed line, and no d-line (%s)" % note)
    lines = []
    for key, h in specs:
        _h, v, _y, _e, prof = line_profiles(st, cfg, h)
        lines.append(dict(key=key, half_um=h, v=v, prof=prof,
                          G=focus.plane_gradient_energies(prof, v, B, cfg.measure, st["valid"])))
    return dict(lines=lines, background=B, bg_frac=bg_frac, bg_margin_used=margin_used,
                d_hat_um=float(st["result"].d_hat_um), d_line_half_um=hd, d_line_note=note)


def blend_evaluation(st, cfg, line_mult=2.0, stripe_half_um=1.0, bin_width=1.0, d0_um=1.5, s_um=0.3, bg_margin_um=4.0):
    """On the line sized from the node's fitted diameter, h = line_mult * d_hat
    / 2 (D-040): the gradient energy over the whole line (with each plane's
    background of block_background), the entropy of the strip's pixels
    (entropy_evaluation: |v| <= h across, |u| <= stripe_half_um along), and
    their min-max blend J = w g + (1 - w) eta with w = focus.sigmoid_weight(
    d_hat, d0_um, s_um) (focus.blend_scores). ValueError without a usable
    d-line or with an empty strip. Returns entropy_evaluation's dict with G,
    g, eta, J, V, w, background, bg_frac, bg_margin_used, d_hat_um, line_mult,
    d0_um, s_um."""
    d, h = _d_line_or_error(st, cfg, line_mult)
    w = focus.sigmoid_weight(d, d0_um, s_um)
    ent = entropy_evaluation(st, cfg, h, stripe_half_um, bin_width)
    B, bg_frac, margin_used = block_background(st, bg_margin_um)
    G = focus.plane_gradient_energies(ent["prof"], ent["v"], B, cfg.measure, st["valid"])
    ent.update(focus.blend_scores(G, ent["h_strip"], w))
    ent.update(G=G, w=w, background=B, bg_frac=bg_frac, bg_margin_used=margin_used, d_hat_um=d,
               line_mult=float(line_mult), d0_um=float(d0_um), s_um=float(s_um))
    return ent


def _truthy(x):
    return x is True or (isinstance(x, str) and x.strip() == "True") or (isinstance(x, (int, np.integer)) and x == 1)


def thin_reference_node(r, max_2r_um=0.6, max_dhat_um=1.0):
    """Whether a pilot row is a thin node on which the pipeline's two focus rules agree: in_S true,
    2 * allen_radius_um <= max_2r_um, d_hat_um <= max_dhat_um, a finite z_sub_um, k_star == k_star_depth and
    steep false. A row lacking a field does not qualify."""
    try:
        return (_truthy(r["in_S"]) and 2.0 * float(r["allen_radius_um"]) <= float(max_2r_um)
                and float(r["d_hat_um"]) <= float(max_dhat_um) and math.isfinite(float(r["z_sub_um"]))
                and int(r["k_star"]) == int(r["k_star_depth"]) and not _truthy(r["steep"]))
    except (KeyError, TypeError, ValueError):
        return False


def select_thin_nodes(swc, rows, max_2r_um=0.6, max_dhat_um=1.0, per_stretch=3, max_nodes=12, types=(3, 4)):
    """Node ids for the thin-dendrite test of the entropy (the user's request of 2026-10-09: where the
    gradient energy works). The pilot rows that pass thin_reference_node; in each unbranched stretch
    (cell.stretches, in its order, proximal first) up to per_stretch of them spread evenly along it
    (survey.spread_ranks over the stretch's qualifying nodes); of those, over all stretches in order, at most
    max_nodes spread evenly. Thresholds PROVISIONAL, the assistant's."""
    by_id = {int(r["node_id"]): r for r in rows}
    picked = []
    for run in cell.stretches(swc, types):
        cand = [int(swc.ids[i]) for i in run if int(swc.ids[i]) in by_id
                and thin_reference_node(by_id[int(swc.ids[i])], max_2r_um, max_dhat_um)]
        picked += [cand[j] for j in survey.spread_ranks(len(cand), int(per_stretch))] if cand else []
    return [picked[j] for j in survey.spread_ranks(len(picked), int(max_nodes))] if picked else []


def diameter_candidate(r):
    """Whether a pilot row may enter the comparison across diameters (D-040;
    PROVISIONAL, the assistant's): fit_status converged, a finite z_sub_um, a
    finite positive d_hat_um, steep false and no flag 'crossing' (a crossing
    neurite widens the fit). A row lacking a field does not qualify."""
    try:
        d = float(r["d_hat_um"])
        flags = r.get("flags")
        flags = [] if flags is None or (isinstance(flags, float) and math.isnan(flags)) else str(flags).split(";")
        return (str(r["fit_status"]) == "converged" and math.isfinite(float(r["z_sub_um"])) and math.isfinite(d)
                and d > 0 and not _truthy(r["steep"]) and "crossing" not in flags)
    except (KeyError, TypeError, ValueError):
        return False


def select_nodes_by_diameter(swc, rows, bins_um=(0.8, 1.0, 1.5, 2.0, 3.0), per_bin=2, add_nodes=(2, 3), types=(3, 4)):
    """Node ids for the comparison across diameters (D-040; PROVISIONAL, the
    assistant's): the pilot rows passing diameter_candidate, in stretch order
    (cell.stretches, proximal first, then along each stretch), split by their
    d_hat into the bins (0, e_1], (e_1, e_2], ..., (e_n, inf) for the strictly
    increasing edges bins_um; per bin up to per_bin of them spread evenly over
    that order (survey.spread_ranks); then the ids of add_nodes the pilot holds,
    whatever their row says. Returned by increasing pilot d_hat, ties by id."""
    edges = [float(e) for e in bins_um]
    if not edges or any(not (math.isfinite(e) and e > 0) for e in edges) or any(b <= a for a, b in zip(edges, edges[1:])):
        raise ValueError("bins_um must be positive, finite and strictly increasing, got %r" % (bins_um,))
    if int(per_bin) < 0:
        raise ValueError("per_bin must be >= 0, got %r" % (per_bin,))
    by_id = {int(r["node_id"]): r for r in rows}
    order = [int(swc.ids[i]) for run in cell.stretches(swc, types) for i in run]
    cand = [n for n in order if n in by_id and diameter_candidate(by_id[n])]
    bounds = [0.0] + edges + [float("inf")]
    picked = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        inbin = [n for n in cand if lo < float(by_id[n]["d_hat_um"]) <= hi]
        picked += [inbin[j] for j in survey.spread_ranks(len(inbin), int(per_bin))] if inbin else []
    picked += [int(n) for n in add_nodes if int(n) in by_id and int(n) not in picked]

    def by_dhat(n):
        try:
            d = float(by_id[n]["d_hat_um"])
        except (KeyError, TypeError, ValueError):
            d = float("nan")
        return (d if math.isfinite(d) else float("inf"), n)
    return sorted(picked, key=by_dhat)


_PICK_KEYS = {"entropy": ("k_h_line", "k_h_strip"), "profile": ("k_min_area", "k_min_area_norm"), "image": (),
              "blend": ("k_Gd", "k_blend", "k_Hd"), "gradient": ()}   # gradient: the keys depend on the lines


def pick_summary(records, evaluation, keys=None, extra=()):
    """(rows, agreement): one row per measured node -- id, SWC type, Allen's 2r, the fitted d, the SWC plane,
    k*, the dip depth's plane, the evaluation's picked planes (keys, default _PICK_KEYS[evaluation]) and each
    pick minus k* (planes), and the record's fields named in extra -- and, per pick, dict(n, within_1,
    median_abs) over the nodes with both planes."""
    keys = _PICK_KEYS[evaluation] if keys is None else tuple(keys)
    rows = []
    for r in records:
        if "skipped" in r:
            continue
        row = dict(node_id=r["node_id"], node_type=r.get("node_type"), allen_2r_um=2.0 * r.get("allen_radius_um", np.nan),
                   d_hat_um=r.get("d_hat_um"), k_swc=r["k_swc"], k_star=r["k_star"], k_star_depth=r["k_star_depth"])
        for key in keys:
            row[key] = r.get(key)
            row[key + "_minus_k_star"] = None if r.get(key) is None else int(r[key]) - int(r["k_star"])
        for x in extra:
            row[x] = r.get(x)
        rows.append(row)
    agreement = {}
    for key in keys:
        d = np.array([abs(x[key + "_minus_k_star"]) for x in rows if x[key + "_minus_k_star"] is not None], dtype=float)
        agreement[key] = dict(n=int(d.size), within_1=int((d <= 1).sum()),
                              median_abs=float(np.median(d)) if d.size else float("nan"))
    return rows, agreement


def _argmin_plane(ks, A):
    A = np.asarray(A, dtype=float)
    return None if not np.isfinite(A).any() else int(ks[int(np.nanargmin(A))])


def _argmax_plane(ks, A):
    """The plane of the largest finite value of A (the first of equal values); None without a finite value."""
    A = np.asarray(A, dtype=float)
    return None if not np.isfinite(A).any() else int(ks[int(np.nanargmax(A))])


def _gradient_line_specs(lines, line_mult):
    """(key, colour, curve style, frame linestyle, tag, legend label) of each line of gradient_evaluation."""
    out, n_fixed = [], 0
    for ln in lines:
        if ln["key"] == "Gd":
            colour, style, ls = _GRAD_D_COLOUR
            label = "G over the whole d-line, full width %g x d_hat" % float(line_mult)
        else:
            colour, style, ls = _GRAD_FIXED_COLOURS[n_fixed]
            n_fixed += 1
            label = "G over the whole +-%g um line" % ln["half_um"]
        out.append((ln["key"], colour, style, ls, ln["key"], label))
    return out


_MARKS_BLEND = (("k_Gd", _GRAD_D_COLOUR[0], "-.", "Gd", "G alone, over the whole d-line"),
                ("k_blend", "#008300", "-", "J", "the blend J = w g + (1 - w) eta"),
                ("k_Hd", "#eda100", "--", "Hs", "the strip's entropy alone (lowest), on the d-line"))


def _profile_record(v, prof):
    """The measuring line's offsets (um) and every plane's profile (grey levels, 2 decimals; NaN rows for the
    invalid planes), for reading the profiles back without the images."""
    P = np.asarray(prof, dtype=float)
    return dict(v_um=[round(float(x), 4) for x in v], profiles=[[round(float(x), 2) for x in row] for row in P])


def _marks(at, specs):
    """Frames (one per plane; the first mark sets the colour, the tags are joined) and curve lines."""
    frames, lines = {}, []
    for key, colour, ls, tag, label in specs:
        k = at.get(key)
        if k is None:
            continue
        lines.append((k, colour, ls, label))
        frames[k] = (frames[k][0], frames[k][1], frames[k][2] + " " + tag) if k in frames else (colour, ls, tag)
    return frames, lines


def _pipeline_at(res, k_swc):
    return dict(k_star=int(res.k_star) if math.isfinite(res.z_sub_um) else None, k_star_depth=int(res.k_star_depth),
                k_swc=int(k_swc))


def run(swc, provider, cfg, node_ids, out_dir, specimen, transform=None, planes_half=6, half_um=None, band_um=0.0,
        evaluation="profile", profile_half_um=None, bg_margin_um=4.0, dpi=110, log=print, stripe_half_um=1.0,
        entropy_bin_gl=1.0, entropy_pick_rule="min", show_line=True, grad_lines_um=(3.0, 5.0), line_mult=2.0,
        sigmoid_d0_um=1.5, sigmoid_s_um=0.3):
    """One figure per node id; returns the per-node records (also written to
    <prefix>_<specimen>.json: planediff for the profile and image evaluations,
    planeentropy, planegrad, planeblend). show_line=False: the profile and
    entropy figures draw the planes without the measuring line and the strip's
    outline (presentation only; the records do not change); the gradient and
    blend figures always do. grad_lines_um, line_mult: the gradient
    evaluation's fixed half-lengths and the d-line's multiplier (D-040);
    sigmoid_d0_um, sigmoid_s_um: the blend's weight w(d_hat)."""
    from allen_diameter.plotting import figures as fg
    import matplotlib.pyplot as plt
    if evaluation not in _PREFIX:
        raise ValueError("evaluation must be one of %s, got %r" % (", ".join(sorted(_PREFIX)), evaluation))
    if entropy_pick_rule not in _PICK_WORDS:
        raise ValueError("entropy_pick_rule must be one of %s, got %r" % (", ".join(sorted(_PICK_WORDS)),
                                                                         entropy_pick_rule))
    focus.sigmoid_weight(sigmoid_d0_um, sigmoid_d0_um, sigmoid_s_um)        # refuses s <= 0 before any node
    grad_lines_um = fixed_lines_um(grad_lines_um) if evaluation == "gradient" else grad_lines_um   # before any node
    line_h, d_mult = None, None
    if evaluation in ("profile", "entropy"):       # the square must hold the measuring line
        line_h = float(cfg.measure.profile_half_um if profile_half_um is None else profile_half_um)
    elif evaluation == "gradient":
        line_h, d_mult = max(grad_lines_um, default=None), float(line_mult)
    elif evaluation == "blend":
        d_mult = float(line_mult)
    os.makedirs(out_dir, exist_ok=True)
    records = []
    for nid in node_ids:
        try:
            st = node_stack(swc, provider, cfg, nid, transform, planes_half, half_um, band_um, line_h, d_mult)
        except ValueError as e:
            log("[planediff] node %d skipped: %s" % (nid, e))
            records.append(dict(node_id=int(nid), skipped=str(e)))
            continue
        r, ks = st["result"], st["ks"]
        at = _pipeline_at(r, st["k_swc"])
        rec = dict(node_id=int(nid), evaluation=evaluation, ks=[int(k) for k in ks],
                   valid=[bool(x) for x in st["valid"]], k_star=int(r.k_star), k_star_depth=int(r.k_star_depth),
                   k_swc=int(st["k_swc"]), z_sub_um=float(r.z_sub_um), n_missing=int((~st["valid"]).sum()),
                   half_um=float(st["square_half_um"]), square_fetched=st["square_fetched"],
                   node_type=st["node_type"], allen_radius_um=st["allen_radius_um"], d_hat_um=float(r.d_hat_um))
        if evaluation == "profile":
            ev = profile_evaluation(st, cfg, profile_half_um, bg_margin_um)
            at.update(k_min_area=_argmin_plane(ks, ev["area"]), k_min_area_norm=_argmin_plane(ks, ev["area_norm"]))
            frames, lines = _marks(at, _MARKS_AREA + _MARKS_PIPELINE)
            na = lambda x: "n/a" if x is None else "%d" % x  # noqa: E731
            label = ("node %d: planes %d..%d; profile +-%.1f um across heading %.0f deg; smallest area at k %s "
                     "(normalised: k %s); k* %d, dip-depth plane %d, SWC plane %d"
                     % (nid, ks[0], ks[-1], ev["half_um"], math.degrees(st["theta"]), na(at["k_min_area"]),
                        na(at["k_min_area_norm"]), r.k_star, r.k_star_depth, st["k_swc"]))
            fig = fg.profile_area_figure([dict(label=label, stack=st["stack"], ks=ks, valid=st["valid"],
                                               extent=st["extent"], line=ev["line"], frames=frames, lines=lines,
                                               v=ev["v"], prof=ev["prof"], area=ev["area"], area_norm=ev["area_norm"],
                                               background=ev["background"], k_ref=st["k_swc"])],
                                         "Area under the profile along the measuring line, specimen %s" % specimen,
                                         show_line=show_line)
            B = ev["background"]
            rec.update(profile_half_um=ev["half_um"], theta_rad=st["theta"], area=[float(x) for x in ev["area"]],
                       area_norm=[float(x) for x in ev["area_norm"]], background=[float(x) for x in B],
                       bg_frac=ev["bg_frac"], bg_margin_um=float(bg_margin_um), bg_margin_used=ev["bg_margin_used"],
                       k_min_area=at["k_min_area"],
                       k_min_area_norm=at["k_min_area_norm"], **_profile_record(ev["v"], ev["prof"]))
            fin = B[np.isfinite(B)]
            msg = ("smallest area at k %s, normalised k %s; background %s"
                   % (na(at["k_min_area"]), na(at["k_min_area_norm"]),
                      "%.1f..%.1f gl (margin %g um)" % (fin.min(), fin.max(), ev["bg_margin_used"]) if fin.size
                      else "n/a (no pixel far from the traced dendrites)"))
        elif evaluation == "entropy":
            ev = entropy_evaluation(st, cfg, profile_half_um, stripe_half_um, entropy_bin_gl)
            d_line = entropy_pick(ev["h_line"], entropy_pick_rule)
            d_strip = entropy_pick(ev["h_strip"], entropy_pick_rule)
            at.update(k_h_line=None if d_line is None else int(ks[d_line]),
                      k_h_strip=None if d_strip is None else int(ks[d_strip]))
            frames, lines = _marks(at, _marks_entropy(entropy_pick_rule) + _MARKS_PIPELINE)
            word = _PICK_WORDS[entropy_pick_rule]
            na = lambda x: "n/a" if x is None else "%d" % x  # noqa: E731
            n_line = int(ev["n_line"].max())
            n_strip = int(ev["n_strip"].max())
            label = ("node %d: planes %d..%d; line +-%.1f um across heading %.0f deg (%d samples), strip +-%.1f um "
                     "along it (%d pixels); %s at k %s (line), k %s (strip); k* %d, dip-depth plane %d, SWC plane %d"
                     % (nid, ks[0], ks[-1], ev["half_um"], math.degrees(st["theta"]), n_line, ev["stripe_half_um"],
                        n_strip, _PICK_WORDS[entropy_pick_rule].split(" (")[0], na(at["k_h_line"]),
                        na(at["k_h_strip"]), r.k_star, r.k_star_depth, st["k_swc"]))
            fig = fg.entropy_figure([dict(label=label, stack=st["stack"], ks=ks, valid=st["valid"], extent=st["extent"],
                                          line=ev["line"], outline=ev["outline"], frames=frames, lines=lines,
                                          hist_line=ev["hist_line"], hist_strip=ev["hist_strip"],
                                          h_line=ev["h_line"], h_strip=ev["h_strip"], dip_line=d_line,
                                          dip_strip=d_strip, n_line=n_line, n_strip=n_strip, k_ref=st["k_swc"],
                                          pick_word=word)],
                                    "Entropy of the grey levels along the measuring line and in its strip, specimen %s"
                                    % specimen, show_line=show_line)
            rec.update(profile_half_um=ev["half_um"], stripe_half_um=ev["stripe_half_um"],
                       entropy_bin_gl=ev["bin_width"], theta_rad=st["theta"],
                       h_line=[float(x) for x in ev["h_line"]], h_strip=[float(x) for x in ev["h_strip"]],
                       n_line=[int(x) for x in ev["n_line"]], n_strip=[int(x) for x in ev["n_strip"]],
                       m_line=[int(x) for x in ev["m_line"]], m_strip=[int(x) for x in ev["m_strip"]],
                       entropy_pick=entropy_pick_rule, k_h_line=at["k_h_line"], k_h_strip=at["k_h_strip"],
                       **_profile_record(ev["v"], ev["prof"]))
            msg = ("%s at k %s along the line (%d samples), k %s in the strip (%d pixels)"
                   % (_PICK_WORDS[entropy_pick_rule].split(" (")[0], na(at["k_h_line"]), n_line, na(at["k_h_strip"]),
                      n_strip))
        elif evaluation == "gradient":
            try:
                ev = gradient_evaluation(st, cfg, grad_lines_um, line_mult, bg_margin_um)
            except ValueError as e:                 # no fixed line and no d-line for this node
                log("[planediff] node %d skipped: %s" % (nid, e))
                records.append(dict(node_id=int(nid), skipped=str(e)))
                continue
            specs = _gradient_line_specs(ev["lines"], line_mult)
            for ln in ev["lines"]:
                at["k_" + ln["key"]] = _argmax_plane(ks, ln["G"])
            frames, lines = _marks(at, [("k_" + key, colour, ls, tag, label)
                                        for key, colour, _st, ls, tag, label in specs] + list(_MARKS_PIPELINE))
            na = lambda x: "n/a" if x is None else "%d" % x  # noqa: E731
            longest = max(ev["lines"], key=lambda x: x["half_um"])
            picks = ", ".join("%s k %s" % ("+-%.2f um (d-line)" % ln["half_um"] if ln["key"] == "Gd" else
                                           "+-%g um" % ln["half_um"], na(at["k_" + ln["key"]])) for ln in ev["lines"])
            if ev["d_line_note"]:
                picks += "; no d-line: " + ev["d_line_note"]
            label = ("node %d: planes %d..%d; d_hat %.2f um; G over the whole line: %s; k* %d, dip-depth plane %d, "
                     "SWC plane %d" % (nid, ks[0], ks[-1], ev["d_hat_um"], picks, r.k_star, r.k_star_depth, st["k_swc"]))
            variants = [dict(key=key, label=label_, colour=colour, style=style, half_um=ln["half_um"], G=ln["G"],
                             pick=None if at["k_" + key] is None else int(at["k_" + key]) - int(ks[0]))
                        for (key, colour, style, _ls, _tag, label_), ln in zip(specs, ev["lines"])]
            fig = fg.gradient_lines_figure([dict(label=label, stack=st["stack"], ks=ks, valid=st["valid"],
                                                 extent=st["extent"], frames=frames, lines=lines, variants=variants,
                                                 v=longest["v"], prof=longest["prof"], background=ev["background"],
                                                 k_ref=st["k_swc"])],
                                           "Gradient energy over the whole measuring line: fixed lines and the line "
                                           "sized from d_hat, specimen %s" % specimen)
            rec.update(lines_um=[float(h) for h in grad_lines_um], line_mult=float(line_mult), theta_rad=st["theta"],
                       d_line_half_um=ev["d_line_half_um"], d_line_note=ev["d_line_note"],
                       background=[float(x) for x in ev["background"]], bg_frac=ev["bg_frac"],
                       bg_margin_um=float(bg_margin_um), bg_margin_used=ev["bg_margin_used"],
                       pick_keys=["k_" + ln["key"] for ln in ev["lines"]],
                       **{"G_" + ln["key"]: [float(x) for x in ln["G"]] for ln in ev["lines"]},
                       **{"k_" + ln["key"]: at["k_" + ln["key"]] for ln in ev["lines"]},
                       **_profile_record(longest["v"], longest["prof"]))
            msg = "d_hat %.2f um; G picks: %s" % (ev["d_hat_um"], picks)
        elif evaluation == "blend":
            try:
                ev = blend_evaluation(st, cfg, line_mult, stripe_half_um, entropy_bin_gl, sigmoid_d0_um, sigmoid_s_um,
                                      bg_margin_um)
            except ValueError as e:
                log("[planediff] node %d skipped: %s" % (nid, e))
                records.append(dict(node_id=int(nid), skipped=str(e)))
                continue
            at.update(k_Gd=_argmax_plane(ks, ev["G"]), k_blend=_argmax_plane(ks, ev["J"]),
                      k_Hd=_argmin_plane(ks, ev["h_strip"]))
            frames, lines = _marks(at, _MARKS_BLEND + _MARKS_PIPELINE)
            na = lambda x: "n/a" if x is None else "%d" % x  # noqa: E731
            n_line, n_strip = int(ev["n_line"].max()), int(ev["n_strip"].max())
            idx = lambda k: None if k is None else int(k) - int(ks[0])  # noqa: E731
            label = ("node %d: planes %d..%d; d_hat %.2f um, line +-%.2f um (%d samples), strip +-%.1f um along it (%d "
                     "pixels); w %.2f; G k %s, blend k %s, strip entropy k %s; k* %d, dip-depth plane %d, SWC plane %d"
                     % (nid, ks[0], ks[-1], ev["d_hat_um"], ev["half_um"], n_line, ev["stripe_half_um"], n_strip,
                        ev["w"], na(at["k_Gd"]), na(at["k_blend"]), na(at["k_Hd"]), r.k_star, r.k_star_depth,
                        st["k_swc"]))
            fig = fg.blend_figure([dict(label=label, stack=st["stack"], ks=ks, valid=st["valid"], extent=st["extent"],
                                        frames=frames, lines=lines, v=ev["v"], prof=ev["prof"], half_um=ev["half_um"],
                                        g=ev["g"], eta=ev["eta"], J=ev["J"], w=ev["w"], d_hat_um=ev["d_hat_um"],
                                        d0_um=ev["d0_um"], s_um=ev["s_um"], pick_g=idx(at["k_Gd"]),
                                        pick_eta=idx(at["k_Hd"]), pick_J=idx(at["k_blend"]), k_ref=st["k_swc"])],
                                  "The line sized from d_hat: gradient energy, sigmoid-weighted min-max blend and strip "
                                  "entropy, specimen %s" % specimen)
            rec.update(profile_half_um=ev["half_um"], d_line_half_um=ev["half_um"], line_mult=ev["line_mult"],
                       stripe_half_um=ev["stripe_half_um"], entropy_bin_gl=ev["bin_width"], theta_rad=st["theta"],
                       w=ev["w"], d0_um=ev["d0_um"], s_um=ev["s_um"],
                       G_Gd=[float(x) for x in ev["G"]], h_strip=[float(x) for x in ev["h_strip"]],
                       n_strip=[int(x) for x in ev["n_strip"]], g=[float(x) for x in ev["g"]],
                       eta=[float(x) for x in ev["eta"]], J=[float(x) for x in ev["J"]],
                       background=[float(x) for x in ev["background"]], bg_frac=ev["bg_frac"],
                       bg_margin_um=float(bg_margin_um), bg_margin_used=ev["bg_margin_used"],
                       pick_keys=list(_PICK_KEYS["blend"]), k_Gd=at["k_Gd"], k_blend=at["k_blend"], k_Hd=at["k_Hd"],
                       **_profile_record(ev["v"], ev["prof"]))
            msg = ("d_hat %.2f um, line +-%.2f um (%d samples), strip %d pixels; w %.2f; G k %s, blend k %s, strip "
                   "entropy k %s" % (ev["d_hat_um"], ev["half_um"], n_line, n_strip, ev["w"], na(at["k_Gd"]),
                                     na(at["k_blend"]), na(at["k_Hd"])))
        else:
            res = focus.plane_differences(st["stack"], st["valid"], st["mask"])
            dip = focus.difference_dip(res["pos"])
            frames, lines = _marks(at, _MARKS_PIPELINE)
            where = "the whole square" if not band_um else "pixels within %.2g um of the stretch" % band_um
            label = ("node %d: planes %d..%d, %s (%d pixels); k* %d, dip-depth plane %d, SWC plane %d"
                     % (nid, ks[0], ks[-1], where, int(st["mask"].sum()), r.k_star, r.k_star_depth, st["k_swc"]))
            fig = fg.plane_difference_figure([dict(label=label, stack=st["stack"], ks=ks, valid=st["valid"], res=res,
                                                   dip=dip, frames=frames, lines=lines, extent=st["extent"])],
                                             "Consecutive-plane differences, specimen %s" % specimen)
            rec.update(pos=[float(x) for x in res["pos"]], neg=[float(x) for x in res["neg"]],
                       dip_pair=None if dip is None else [int(ks[dip]), int(ks[dip + 1])],
                       n_pixels=int(st["mask"].sum()), band_um=float(band_um))
            msg = "%d pixels; dip of S+ at %s" % (rec["n_pixels"], "none" if dip is None else
                                                 "%d->%d" % tuple(rec["dip_pair"]))
        path = os.path.join(out_dir, "%s_%d.png" % (_PREFIX[evaluation], nid))
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        rec["png"] = path
        log("[planediff] node %d: planes %d..%d (%d missing); %s; k* %d, dip-depth plane %d, SWC plane %d"
            % (nid, ks[0], ks[-1], rec["n_missing"], msg, rec["k_star"], rec["k_star_depth"], rec["k_swc"]))
        records.append(rec)
    with open(os.path.join(out_dir, "%s_%s.json" % (_PREFIX[evaluation], specimen)), "w") as f:
        json.dump(records, f, indent=1, sort_keys=True, allow_nan=True)
    done = [x for x in records if "png" in x]
    keys = list(_PICK_KEYS[evaluation])
    if evaluation == "gradient":                    # every key any node has, in the order of the lines
        keys = [k for k in ["k_G%g" % float(h) for h in grad_lines_um] + ["k_Gd"]
                if any(k in x.get("pick_keys", ()) for x in done)]
    if len(done) > 1 and keys:
        from allen_diameter.loading import table_io
        extra = {"gradient": ("d_line_half_um",), "blend": ("d_line_half_um", "w")}.get(evaluation, ())
        rows, agreement = pick_summary(records, evaluation, keys, extra)
        table_io.write_rows(rows, os.path.join(out_dir, "%s_summary_%s.csv" % (_PREFIX[evaluation], specimen)))
        log("[planediff] summary over %d nodes: %s" % (len(rows), "; ".join(
            "%s within 1 plane of k* in %d of %d (median |diff| %.1f planes)"
            % (key, a["within_1"], a["n"], a["median_abs"]) for key, a in agreement.items())))
        fig = None
        if evaluation == "entropy":
            fig = fg.entropy_summary_figure(done, "Entropy picks against k* on %d nodes, specimen %s (%s)"
                                            % (len(done), specimen, _PICK_WORDS[entropy_pick_rule].split(" (")[0]))
        elif evaluation == "gradient":
            meta = [dict(key="G%g" % float(h), half_um=float(h)) for h in grad_lines_um] + [dict(key="Gd", half_um=None)]
            specs = _gradient_line_specs([x for x in meta if "k_" + x["key"] in keys], line_mult)
            series = [dict(curve="G_" + key, pick="k_" + key, label=label, colour=colour, style=style)
                      for key, colour, style, _ls, _tag, label in specs]
            fig = fg.picks_summary_figure(done, series, "Gradient energy over the whole line: the plane each line "
                                                        "picks, on %d nodes by d_hat, specimen %s" % (len(done), specimen))
        elif evaluation == "blend":
            series = [dict(curve="G_Gd", pick="k_Gd", label="G alone, over the whole d-line", colour=_GRAD_D_COLOUR[0],
                           style=_GRAD_D_COLOUR[1]),
                      dict(curve="J", pick="k_blend", label="the blend J = w g + (1 - w) eta", colour="#008300",
                           style="D-"),
                      dict(curve="h_strip", pick="k_Hd", label="the strip's entropy alone (inverted: lowest on top)",
                           colour="#eda100", style="s--", lower_is_better=True)]
            fig = fg.picks_summary_figure(done, series, "The d-line: the plane each method picks, on %d nodes by d_hat "
                                                        "(w = sigmoid weight of G), specimen %s" % (len(done), specimen),
                                          weight_key="w")
        if fig is not None:
            fig.savefig(os.path.join(out_dir, "%s_summary_%s.png" % (_PREFIX[evaluation], specimen)), dpi=dpi)
            plt.close(fig)
    return records


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--specimen", required=True)
    ap.add_argument("--nodes", required=True, help="comma-separated node ids; 'thin' (needs --pilot-csv): thin "
                                                    "nodes where k* and the dip depth agree, spread over the stretches; "
                                                    "'bydiameter' (needs --pilot-csv): nodes spread over bins of d_hat")
    ap.add_argument("--pilot-csv", default="", help="the pilot's per-node CSV (pilot_<specimen>.csv), for --nodes thin "
                                                    "or bydiameter")
    ap.add_argument("--max-nodes", type=int, default=12, help="--nodes thin: at most this many nodes")
    ap.add_argument("--per-stretch", type=int, default=3, help="--nodes thin: at most this many per stretch")
    ap.add_argument("--thin-max-2r-um", type=float, default=0.6, help="--nodes thin: Allen's 2r at most, um")
    ap.add_argument("--thin-max-dhat-um", type=float, default=1.0, help="--nodes thin: the pilot's fitted d at most, um")
    ap.add_argument("--dhat-bins", default="0.8,1.0,1.5,2.0,3.0",
                    help="--nodes bydiameter: increasing bin edges of d_hat, um (bins (0, e1], ..., (e_n, inf))")
    ap.add_argument("--per-bin", type=int, default=2, help="--nodes bydiameter: at most this many nodes per bin")
    ap.add_argument("--add-nodes", default="2,3",
                    help="--nodes bydiameter: node ids always added (default: the trunks 2 and 3); '' for none")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--evaluation", choices=("profile", "image", "entropy", "gradient", "blend"), default="profile")
    ap.add_argument("--grad-lines-um", default="3,5",
                    help="gradient evaluation: half-lengths of the fixed lines, um (at most 3)")
    ap.add_argument("--line-mult", type=float, default=2.0,
                    help="gradient and blend evaluations: the d-line's full width in units of d_hat (2: [-d_hat, d_hat])")
    ap.add_argument("--sigmoid-d0-um", type=float, default=1.5,
                    help="blend evaluation: d_hat at which the gradient energy's weight is 1/2, um")
    ap.add_argument("--sigmoid-s-um", type=float, default=0.3,
                    help="blend evaluation: width of the sigmoid w(d) = 1 / (1 + exp((d - d0) / s)), um")
    ap.add_argument("--planes-half", type=int, default=6, help="planes on each side of the SWC plane")
    ap.add_argument("--profile-half-um", type=float, default=None,
                    help="profile and entropy evaluations: half-length of the measuring line, um "
                         "(default: profile_half_um)")
    ap.add_argument("--stripe-half-um", type=float, default=1.0,
                    help="entropy evaluation: half-width of the strip along the branch, um")
    ap.add_argument("--entropy-bin-gl", type=float, default=1.0,
                    help="entropy evaluation: width of a histogram bin, grey levels")
    ap.add_argument("--entropy-pick", choices=("min", "dip"), default="min",
                    help="entropy evaluation: the plane framed on each curve -- its global minimum (default), or "
                         "its dip between the two largest maxima")
    ap.add_argument("--hide-line", action="store_true",
                    help="profile and entropy evaluations: draw the planes without the measuring line and the "
                         "strip's outline, to judge the focus by eye (the frames stay; nothing measured changes)")
    ap.add_argument("--bg-margin-um", type=float, default=4.0,
                    help="profile evaluation: B_k uses the block's pixels farther than Allen's radius + this from "
                         "every traced dendrite")
    ap.add_argument("--half-um", type=float, default=None, help="half-width of the square, um "
                                                                "(default: the measurement's block_half_um)")
    ap.add_argument("--band-um", type=float, default=0.0, help="image evaluation, > 0: count only the pixels "
                                                               "within this distance of the traced stretch")
    ap.add_argument("--swc", default="")
    ap.add_argument("--cache-dir", default="allen_cache")
    ap.add_argument("--shift-x", type=float, default=0.0, help="full-res px (global alignment)")
    ap.add_argument("--shift-y", type=float, default=0.0)
    ap.add_argument("--flip-h", type=float, default=None)
    ap.add_argument("--z0", type=float, default=0.0)
    ap.add_argument("--config-json", default="")
    a = ap.parse_args(argv)
    cfg = load_config(a.config_json)
    mode = a.nodes.strip().lower()
    thin, bydiam = mode == "thin", mode == "bydiameter"
    if (thin or bydiam) and not a.pilot_csv:
        ap.error("--nodes %s needs --pilot-csv" % mode)
    try:
        grad_lines = [float(x) for x in a.grad_lines_um.split(",") if x.strip()]
        dhat_bins = [float(x) for x in a.dhat_bins.split(",") if x.strip()]
        add_nodes = [int(x) for x in a.add_nodes.split(",") if x.strip()]
    except ValueError as e:
        ap.error("--grad-lines-um, --dhat-bins and --add-nodes take comma-separated numbers: %s" % e)
    node_ids = [] if (thin or bydiam) else [int(x) for x in a.nodes.split(",") if x.strip()]
    if not (thin or bydiam) and not node_ids:
        print("[planediff] no node given: nothing to do", flush=True)
        return 0
    import allen_image_io as aio
    import run_cell
    from allen_diameter.loading import swc_io, table_io
    swc = swc_io.read_swc(a.swc or aio.fetch_swc(int(a.specimen), a.cache_dir))
    if thin:
        node_ids = select_thin_nodes(swc, table_io.read_rows([a.pilot_csv]), a.thin_max_2r_um, a.thin_max_dhat_um,
                                     a.per_stretch, a.max_nodes, cfg.acquisition.dendrite_swc_types)
        print("[planediff] thin nodes (in S, Allen 2r <= %g um, fitted d <= %g um, k* = dip-depth plane, flat; up to "
              "%d per stretch, %d in all): %s" % (a.thin_max_2r_um, a.thin_max_dhat_um, a.per_stretch, a.max_nodes,
                                                  ",".join(str(n) for n in node_ids) or "none"), flush=True)
        if not node_ids:
            return 0
    if bydiam:
        rows = table_io.read_rows([a.pilot_csv])
        node_ids = select_nodes_by_diameter(swc, rows, dhat_bins, a.per_bin, add_nodes, cfg.acquisition.dendrite_swc_types)
        dh = {int(r["node_id"]): r.get("d_hat_um") for r in rows}
        fmt = lambda x: "%.2f" % float(x) if isinstance(x, (int, float)) else "n/a"  # noqa: E731
        print("[planediff] nodes by diameter (bins of d_hat at %s um, up to %d per bin, converged, flat, not crossing; "
              "plus %s): %s" % (",".join("%g" % e for e in dhat_bins), a.per_bin,
                                ",".join(str(n) for n in add_nodes) or "none",
                                ", ".join("%d (%s um)" % (n, fmt(dh.get(n))) for n in node_ids) or "none"), flush=True)
        if not node_ids:
            return 0
    fetcher = aio.HttpFetcher(cache_dir=a.cache_dir)
    planes = aio.plane_table(aio.list_images(int(a.specimen)))
    provider = run_cell.real_provider(fetcher, planes, cfg.acquisition.res0_um)
    t0 = time.perf_counter()
    recs = run(swc, provider, cfg, node_ids, a.out_dir, a.specimen,
               dict(shift_full_px=(a.shift_x, a.shift_y), flip_y_full_h=a.flip_h, z0_um=a.z0),
               a.planes_half, a.half_um, a.band_um, a.evaluation, a.profile_half_um, a.bg_margin_um,
               log=lambda m: print(m, flush=True), stripe_half_um=a.stripe_half_um, entropy_bin_gl=a.entropy_bin_gl,
               entropy_pick_rule=a.entropy_pick, show_line=not a.hide_line, grad_lines_um=grad_lines,
               line_mult=a.line_mult, sigmoid_d0_um=a.sigmoid_d0_um, sigmoid_s_um=a.sigmoid_s_um)
    n_ok = sum(1 for r in recs if "png" in r)
    print("[planediff] wrote %d figures (%d skipped) in %s; crops: %d from the cache, %d downloaded (%.1f MB); %.1f min"
          % (n_ok, len(recs) - n_ok, a.out_dir, fetcher.n_cache_hits, fetcher.n_requests,
             fetcher.bytes_downloaded / 1e6, (time.perf_counter() - t0) / 60), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
