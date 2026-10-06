"""
allen_image_align.py -- put an SWC skeleton into image pixel coordinates.

Separate module because alignment is its own concern: it consumes an image
(from allen_image_io) and an SWC (from allen_image_plot.read_swc) and produces
a transform that allen_image_plot.overlay_swc and any later measurement both
use. It fetches nothing and draws nothing.

The transform, stated once and carried everywhere
-------------------------------------------------
For an SWC node at (x_um, y_um) and an image whose full-resolution scale is
res0 um/px and whose full height is H_full px, the FULL-RESOLUTION pixel is

    x_px = x_um / res0 + dx                                              (1)
    y_px = (H_full - y_um / res0) + dy   if flip_y                       (2a)
         = y_um / res0 + dy              otherwise                       (2b)

with (dx, dy) the alignment offset in full-resolution pixels and flip_y a
boolean. That (dx, dy, flip_y) triple is the ENTIRE output of this module, and
is exactly what overlay_swc takes as shift_px / flip_y_full_h.

Whether Allen SWC coordinates are already in this frame (dx = dy = 0, no flip)
is NOT assumed -- `diagnose_frame` measures it, and `align_translation` finds
the residual offset by cross-correlation.
"""
import math

import numpy as np
from scipy import ndimage, signal

DEND_TYPES = (3, 4)          # basal, apical; 1 = soma, 2 = axon


# ------------------------------------------------------------- diagnostics
def diagnose_frame(swc, res0_um_px, width_px, height_px):
    """Compare the SWC bounding box with the image extent, before any fitting.

    A direct mapping means the SWC box sits inside the image box. If it does
    not, no translation will fix it and the units or the origin are different.
    Returns a dict; read `verdict` first.
    """
    x, y = swc["x"].values, swc["y"].values
    img_w_um, img_h_um = width_px * res0_um_px, height_px * res0_um_px
    bx, by = (x.min(), x.max()), (y.min(), y.max())
    fits = (bx[0] >= -img_w_um * 0.1 and bx[1] <= img_w_um * 1.1 and
            by[0] >= -img_h_um * 0.1 and by[1] <= img_h_um * 1.1)
    spans_ok = (bx[1] - bx[0]) <= img_w_um and (by[1] - by[0]) <= img_h_um
    if fits and spans_ok:
        verdict = "SWC is in image microns; expect a small offset only"
    elif spans_ok:
        verdict = ("SWC spans fit but sit outside the image box -- a translated "
                   "origin (e.g. soma-centred coordinates)")
    else:
        verdict = ("SWC is LARGER than the image -- units or scale differ; a "
                   "translation will not fix this")
    return dict(swc_x_um=bx, swc_y_um=by, image_um=(img_w_um, img_h_um),
                swc_span_um=(bx[1] - bx[0], by[1] - by[0]), verdict=verdict)


# ---------------------------------------------------------------- rendering
def render_skeleton_mask(swc, shape, res_um_px, offset_px=(0.0, 0.0),
                         flip_y_rows=None, types=DEND_TYPES):
    """Binary mask of the skeleton drawn into an array of `shape` (rows, cols).

    res_um_px is the scale OF THAT ARRAY; offset_px and flip_y_rows are in the
    same array pixels (not full-resolution ones)."""
    rows, cols = shape
    mask = np.zeros(shape, dtype=bool)
    xs = swc["x"].values / res_um_px + offset_px[0]
    ys = swc["y"].values / res_um_px
    ys = (flip_y_rows - ys if flip_y_rows is not None else ys) + offset_px[1]
    idx = {int(i): k for k, i in enumerate(swc["id"].values)}
    typ = swc["type"].values
    for k, p in enumerate(swc["parent"].values):
        if int(p) not in idx or int(typ[k]) not in types:
            continue
        kp = idx[int(p)]
        n = max(2, int(math.hypot(xs[k] - xs[kp], ys[k] - ys[kp]) * 2) + 1)
        cc = np.linspace(xs[kp], xs[k], n).round().astype(int)
        rr = np.linspace(ys[kp], ys[k], n).round().astype(int)
        ok = (cc >= 0) & (cc < cols) & (rr >= 0) & (rr < rows)
        mask[rr[ok], cc[ok]] = True
    return mask


def darkness_map(img, bg_px=41):
    """Background-flattened darkness, >= 0. The background is a median filter,
    so a slow illumination gradient does not dominate the correlation."""
    a = img.astype(np.float32)
    bg = ndimage.median_filter(a, size=bg_px, mode="nearest")
    return np.clip(bg - a, 0, None)


# ---------------------------------------------------------------- alignment
def align_translation(swc, img, res_um_px, flip_y_rows=None,
                      max_shift_um=250.0, try_flip=True, refine_px=2,
                      types=DEND_TYPES):
    """Find the (dx, dy) in ARRAY pixels, and the y-flip, that best place the
    skeleton on the dark pixels. Cross-correlation, then a small local refine.

    Returns dict(dx_px, dy_px, flip_y, score, score_noflip, score_flip),
    where score is the mean darkness under the skeleton -- higher is better,
    and a good alignment on these images is many times the value of a wrong one.
    """
    D = darkness_map(img)
    rows, cols = img.shape[:2]
    best = None
    scores = {}
    for flip in ((False, True) if try_flip else (False,)):
        fr = (rows if flip else None) if flip_y_rows is None else (flip_y_rows if flip else None)
        S = render_skeleton_mask(swc, img.shape, res_um_px, (0.0, 0.0), fr, types)
        if not S.any():
            scores[flip] = -np.inf
            continue
        corr = signal.fftconvolve(D, S[::-1, ::-1].astype(np.float32), mode="same")
        m = int(min(max_shift_um / res_um_px, rows // 2 - 1, cols // 2 - 1))
        cy, cx = rows // 2, cols // 2
        win = corr[cy - m:cy + m + 1, cx - m:cx + m + 1]
        py, px = np.unravel_index(int(np.argmax(win)), win.shape)
        dx, dy = float(px - m), float(py - m)
        # local refine on the actual objective (mean darkness under the mask)
        bs, bv = (dx, dy), -np.inf
        for ddx in range(-refine_px, refine_px + 1):
            for ddy in range(-refine_px, refine_px + 1):
                Sm = render_skeleton_mask(swc, img.shape, res_um_px,
                                          (dx + ddx, dy + ddy), fr, types)
                v = float(D[Sm].mean()) if Sm.any() else -np.inf
                if v > bv:
                    bv, bs = v, (dx + ddx, dy + ddy)
        scores[flip] = bv
        if best is None or bv > best[2]:
            best = (bs, flip, bv)
    (dx, dy), flip, score = best
    return dict(dx_px=dx, dy_px=dy, flip_y=flip, score=score,
                score_noflip=scores.get(False, float("nan")),
                score_flip=scores.get(True, float("nan")))


def to_full_res_shift(result, frame):
    """Convert an alignment found on a downsampled array into the full-res
    (dx, dy) that overlay_swc and every later crop expect."""
    f = frame.factor
    return (result["dx_px"] * f, result["dy_px"] * f)


def swc_to_full_px(swc, res0_um_px, shift_full_px=(0.0, 0.0), flip_y_full_h=None):
    """Apply equations (1) and (2) -- the one place the transform is computed."""
    x = swc["x"].values / res0_um_px + shift_full_px[0]
    y = swc["y"].values / res0_um_px
    if flip_y_full_h is not None:
        y = flip_y_full_h - y
    return x, y + shift_full_px[1]


# =========================================================================
# LOCAL REGISTRATION CHECK -- is this traced path on the process I'm looking at?
# =========================================================================
#
# Global alignment gives one (dx, dy) for the whole cell from a 1.83 um/px
# overview. Before measuring a diameter at full resolution, each path has to be
# checked locally: an unbranched stretch of the skeleton is slid sideways, and
# along z, and the image darkness under it is recorded as a function of the
# shift. A path lying on its process gives a sharp peak at zero shift; a path
# lying alongside gives the peak at the gap; a path over nothing gives no peak.
# The same path dropped at random nearby places supplies the null: how large a
# peak arises by coincidence in this particular patch of tissue.
#
# Conventions: positions in FULL-RES pixels unless named *_um; the normal n is
# the tangent rotated +90 deg in the (col, row) image frame, so a positive
# lateral offset s always points to the same side along the whole path.

def path_through_node(swc, node_id, n_each_way=15, types=DEND_TYPES):
    """The unbranched stretch of skeleton containing node_id, ordered
    proximal -> distal. Walks up to n_each_way nodes toward the soma (stopping
    at, and including, a branch point) and down while there is exactly one
    dendritic child (stopping at a branch point or a tip)."""
    ids = swc["id"].astype(int).values
    parent = dict(zip(ids, swc["parent"].astype(int).values))
    typ = dict(zip(ids, swc["type"].astype(int).values))
    children = {}
    for i, p in parent.items():
        children.setdefault(p, []).append(i)
    node_id = int(node_id)
    if node_id not in typ:
        raise KeyError("node %d not in SWC" % node_id)
    up, cur = [], node_id
    for _ in range(n_each_way):
        p = parent.get(cur, -1)
        if p not in typ or typ[p] not in types:
            break
        up.append(p)
        cur = p
        if len([c for c in children.get(p, []) if typ.get(c) in types]) > 1:
            break
    down, cur = [], node_id
    for _ in range(n_each_way):
        ch = [c for c in children.get(cur, []) if typ.get(c) in types]
        if len(ch) != 1:
            break
        cur = ch[0]
        down.append(cur)
    order = list(reversed(up)) + [node_id] + down
    row_of = {int(i): k for k, i in enumerate(ids)}
    return swc.iloc[[row_of[i] for i in order]].reset_index(drop=True)


def densify_path(cols, rows, zs_um, step_px):
    """Points every ~step_px along the polyline, each with its interpolated z
    and the unit tangent of its segment. Returns (N, 5): col, row, z, tc, tr."""
    cols, rows, zs_um = map(np.asarray, (cols, rows, zs_um))
    if len(cols) < 2:
        raise ValueError("a path needs at least two nodes")
    pts, tc, tr = [], 1.0, 0.0
    for i in range(len(cols) - 1):
        dc, dr = cols[i + 1] - cols[i], rows[i + 1] - rows[i]
        L = math.hypot(dc, dr)
        if L < 1e-9:
            continue
        tc, tr = dc / L, dr / L
        n = max(1, int(math.ceil(L / step_px)))
        for j in range(n):
            f = j / n
            pts.append((cols[i] + f * dc, rows[i] + f * dr,
                        zs_um[i] + f * (zs_um[i + 1] - zs_um[i]), tc, tr))
    pts.append((cols[-1], rows[-1], zs_um[-1], tc, tr))
    return np.array(pts, dtype=np.float64)


def ridge_darkness(img, res_um_px, max_width_um=4.0):
    """Darkness of features NARROWER than max_width_um, >= 0.

    Background = grey closing (a rolling-ball in all but name): it erases every
    dark feature thinner than the structuring element, so bg - img keeps the
    processes and discards the slow gradient. Unlike a median background it
    does not fail when a thick process fills much of the window. Works on a 2-D
    plane or, per plane, on a 3-D block."""
    k = int(round(max_width_um / res_um_px)) | 1
    a = np.asarray(img, dtype=np.float32)
    size = (1, k, k) if a.ndim == 3 else (k, k)
    return ndimage.grey_closing(a, size=size) - a


def plan_path_block(path, res0_um_px, dz_um, shift_full_px=(0.0, 0.0),
                    flip_y_full_h=None, margin_um=20.0, dz_max_um=4.0,
                    k_min=0, k_max=10 ** 9, z0_um=0.0):
    """The xy crop and plane range that contain a path plus every lateral,
    axial and null shift the check will try. Pure arithmetic: tested offline."""
    x, y = swc_to_full_px(path, res0_um_px, shift_full_px, flip_y_full_h)
    m = int(math.ceil(margin_um / res0_um_px))
    k = (path["z"].values - z0_um) / dz_um
    pad = int(math.ceil(dz_max_um / dz_um)) + 2
    k_lo = max(int(math.floor(k.min())) - pad, int(k_min))
    k_hi = min(int(math.ceil(k.max())) + pad, int(k_max))
    left, top = int(math.floor(x.min())) - m, int(math.floor(y.min())) - m
    width = int(math.ceil(x.max() - x.min())) + 2 * m
    height = int(math.ceil(y.max() - y.min())) + 2 * m
    return dict(left=left, top=top, width=width, height=height, k_lo=k_lo, k_hi=k_hi,
                n_planes=k_hi - k_lo + 1,
                path_len_um=float(np.sum(np.hypot(np.diff(x), np.diff(y))) * res0_um_px))


def _profile(D2, cols, rows, nc, nr, offsets_px, min_inside=0.7):
    """Mean of D2 along the path shifted by each offset along its normal."""
    out = np.full(len(offsets_px), np.nan)
    for i, s in enumerate(offsets_px):
        v = ndimage.map_coordinates(D2, [rows + s * nr, cols + s * nc], order=1,
                                    mode="constant", cval=np.nan)
        if np.mean(np.isfinite(v)) >= min_inside:
            out[i] = np.nanmean(v)
    return out


def _peak(x, y, flank_min):
    """argmax, baseline from the flanks (>= flank_min away), robust spread,
    contrast, SNR and FWHM of the peak above baseline."""
    y = np.asarray(y, dtype=np.float64)
    if not np.any(np.isfinite(y)):
        return dict(x_star=np.nan, peak=np.nan, base=np.nan, contrast=np.nan,
                    snr=np.nan, width=np.nan)
    j = int(np.nanargmax(y))
    fl = y[(np.abs(x - x[j]) >= flank_min) & np.isfinite(y)]
    if len(fl) < 3:
        fl = y[np.isfinite(y)]
    base = float(np.median(fl))
    spread = float(1.4826 * np.median(np.abs(fl - base))) + 1e-6
    contrast = float(y[j] - base)
    half = base + contrast / 2.0
    lo = j
    while lo > 0 and np.isfinite(y[lo - 1]) and y[lo - 1] > half:
        lo -= 1
    hi = j
    while hi < len(y) - 1 and np.isfinite(y[hi + 1]) and y[hi + 1] > half:
        hi += 1
    width = float(x[hi] - x[lo]) if (lo > 0 and hi < len(y) - 1) else np.nan
    return dict(x_star=float(x[j]), peak=float(y[j]), base=base, contrast=contrast,
                snr=contrast / spread, width=width)


def registration_check(path, block, ks, valid, frame, dz_um, node_id=None,
                       shift_full_px=(0.0, 0.0), flip_y_full_h=None, z0_um=0.0,
                       s_max_um=4.0, s_step_um=0.1, dz_max_um=4.0,
                       n_null=60, null_shift_um=(9.0, 16.0), null_cone_deg=45.0, seed=0,
                       max_width_um=4.0, lateral_tol_um=0.5, snap_max_um=3.0,
                       alpha=0.05, min_coverage=0.5, lateral_window_um=0.4):
    """Is `path` lying on the process it traces? See the block comment above.

    Two independent gates, both required for ON / ALONGSIDE:
      p_value  <= alpha        the peak is larger than random placements of the
                               same path shape produce in this patch of tissue
      coverage >= min_coverage the peak comes from the path FOLLOWING a ridge,
                               not from crossing a speck or another process once
    Neither alone is enough: a path over empty tissue that happens to cross two
    specks at the same offset can beat most random placements (p small) while
    covering 11% of its length; a path over noise can have half its samples
    above a near-zero threshold (coverage ~50%) while p is unremarkable.

    The flank SNR of the averaged profile is reported but NOT used: averaging
    along the path makes the flanks so smooth that one speck scores SNR > 50.

    Returns a dict. Read, in order:
      verdict           ON / ALONGSIDE / NOT ON a visible process
      lateral_offset_um s* -- how far sideways the process is (0 = on it)
      p_value           fraction of random placements that matched as well
      coverage          fraction of the path individually on the ridge at s*
      z_null            (contrast - null median) / null MAD -- effect size
      z_offset_um       where along z the process is sharpest, relative to SWC z
    """
    res0 = frame.res0_um_px
    x, y = swc_to_full_px(path, res0, shift_full_px, flip_y_full_h)
    cols_a, rows_a = frame.to_array_xy(x, y)
    S = densify_path(cols_a, rows_a, path["z"].values, step_px=0.25 / res0)
    c, r, z, tc, tr = S.T
    nc, nr = -tr, tc

    D3 = ridge_darkness(block, res0, max_width_um)
    D3[~valid] = np.nan
    mip = np.nanmax(D3, axis=0)
    mip = np.where(np.isfinite(mip), mip, 0.0).astype(np.float32)

    s_um = np.arange(-s_max_um, s_max_um + 1e-9, s_step_um)
    L = _profile(mip, c, r, nc, nr, s_um / res0)
    lat = _peak(s_um, L, flank_min=2.0)

    # null: the same path shape, rigidly translated ACROSS the tissue.
    # Translations are drawn within +/- null_cone_deg of the path's mean normal:
    # a translation ALONG the path would leave it lying on the very process under
    # test (exactly so for a straight trunk) and the null would re-find the real
    # match. The minimum distance (9 um) exceeds the lateral search (4 um) plus
    # the snap limit (3 um), so a null copy cannot reach back to the process.
    rng = np.random.default_rng(seed)
    H, W = mip.shape
    ev = np.linalg.eigh(np.cov(np.vstack([c - c.mean(), r - r.mean()])))[1][:, -1]
    mean_normal = math.atan2(ev[0], -ev[1])          # tangent (ev) rotated +90 deg
    L_null, c_null = [], []
    tries = 0
    while len(c_null) < n_null and tries < 40 * n_null:
        tries += 1
        a = (mean_normal + math.radians(rng.uniform(-null_cone_deg, null_cone_deg))
             + (math.pi if rng.random() < 0.5 else 0.0))
        d = rng.uniform(*null_shift_um) / res0
        cc, rr = c + d * math.cos(a), r + d * math.sin(a)
        reach = s_max_um / res0 + 2
        if cc.min() < reach or rr.min() < reach or cc.max() > W - reach or rr.max() > H - reach:
            continue
        Ln = _profile(mip, cc, rr, nc, nr, s_um / res0)
        pk = _peak(s_um, Ln, flank_min=2.0)
        if np.isfinite(pk["contrast"]):
            L_null.append(Ln)
            c_null.append(pk["contrast"])
    c_null = np.array(c_null)
    p_value = (1 + np.sum(c_null >= lat["contrast"])) / (1 + len(c_null)) if len(c_null) else np.nan
    if len(c_null) >= 5:
        med = float(np.median(c_null))
        mad = float(1.4826 * np.median(np.abs(c_null - med))) + 1e-6
        z_null = (lat["contrast"] - med) / mad
    else:
        z_null = np.nan

    # coverage: is the path ON the ridge along its length, or crossing it once?
    s_star_px = (lat["x_star"] if np.isfinite(lat["x_star"]) else 0.0) / res0
    v = ndimage.map_coordinates(mip, [r + s_star_px * nr, c + s_star_px * nc], order=1,
                                mode="constant", cval=np.nan)
    coverage = float(np.nanmean(v >= lat["base"] + 0.5 * lat["contrast"])) \
        if np.isfinite(lat["contrast"]) and lat["contrast"] > 0 else 0.0

    # z: at the laterally corrected positions, where is the process sharpest?
    win = 2 * int(round(lateral_window_um / res0)) + 1
    D3m = ndimage.maximum_filter(np.nan_to_num(D3, nan=-1.0), size=(1, win, win))
    K = int(math.ceil(dz_max_um / dz_um))
    dk = np.arange(-K, K + 1)
    cs = np.clip(np.round(c + s_star_px * nc).astype(int), 0, W - 1)
    rs = np.clip(np.round(r + s_star_px * nr).astype(int), 0, H - 1)
    kf = (z - z0_um) / dz_um
    Z = np.full(len(dk), np.nan)
    for i, d_ in enumerate(dk):
        j = np.round(kf + d_).astype(int) - int(ks[0])
        ok = (j >= 0) & (j < len(ks))
        ok[ok] &= valid[j[ok]]
        if ok.mean() >= 0.5:
            Z[i] = float(np.mean(D3m[j[ok], rs[ok], cs[ok]]))
    zst = _peak(dk * dz_um, Z, flank_min=1.5)

    # verdicts
    passes_p = bool(np.isfinite(p_value) and p_value <= alpha)
    passes_cov = bool(coverage >= min_coverage)
    significant = passes_p and passes_cov
    s_star = lat["x_star"]
    if not significant:
        why = []
        if not passes_p:
            why.append("p=%.3f > %.2f" % (p_value, alpha))
        if not passes_cov:
            why.append("coverage %.0f%% < %.0f%%" % (100 * coverage, 100 * min_coverage))
        verdict = ("NOT ON A VISIBLE PROCESS (%s): wrong structure, faint fill, or "
                   "outside the z-band" % ", ".join(why))
    elif abs(s_star) <= lateral_tol_um:
        verdict = "ON THE PROCESS (lateral offset %+.2f um)" % s_star
    elif abs(s_star) <= snap_max_um:
        verdict = ("ALONGSIDE: a ridge %+.2f um to the side -- snap needed before "
                   "measuring" % s_star)
    else:
        verdict = "FAR: nearest matching ridge %+.2f um away -- treat as unregistered" % s_star
    if not significant:
        z_verdict = "z not assessed (path is not on a process)"
    elif np.isfinite(zst["contrast"]) and zst["contrast"] > 0.25 * lat["contrast"]:
        z_verdict = ("SWC z consistent (%+.2f um)" % zst["x_star"] if abs(zst["x_star"]) <= 1.0
                     else "SWC z off by %+.2f um -- use z + offset for this path" % zst["x_star"])
    else:
        z_verdict = "no clear focus peak along z"
    if significant and abs(s_star) > lateral_tol_um and np.isfinite(zst["x_star"]) \
            and abs(zst["x_star"]) > 2.0:
        verdict += " [and focus is %+.1f um off in z: probably a DIFFERENT process]" % zst["x_star"]

    # what to draw
    k_node = None
    if node_id is not None and int(node_id) in set(path["id"].astype(int)):
        zn = float(path.loc[path["id"].astype(int) == int(node_id), "z"].iloc[0])
        k_node = int(round((zn - z0_um) / dz_um + (zst["x_star"] / dz_um
                                                   if np.isfinite(zst["x_star"]) else 0)))
    else:
        k_node = int(round(np.median(kf) + (zst["x_star"] / dz_um if np.isfinite(zst["x_star"]) else 0)))
    jn = int(np.clip(k_node - int(ks[0]), 0, len(ks) - 1))

    return dict(verdict=verdict, z_verdict=z_verdict,
                lateral_offset_um=s_star, p_value=float(p_value), coverage=coverage,
                z_null=float(z_null), contrast=lat["contrast"],
                flank_snr=lat["snr"], n_null=int(len(c_null)),
                passes_p=passes_p, passes_coverage=passes_cov,
                peak_width_um=lat["width"], z_offset_um=zst["x_star"], z_snr=zst["snr"],
                path_nodes=int(len(path)), path_len_um=float(np.sum(np.hypot(np.diff(cols_a), np.diff(rows_a))) * res0),
                # arrays for plotting
                mip=mip, s_um=s_um, L=L, L_null=np.array(L_null), base=lat["base"],
                dz_axis_um=dk * dz_um, Z=Z, samples=S, best_plane=block[jn], best_k=int(ks[jn]),
                node_id=node_id)
