"""Each S3 cell's global alignment, SWC to image -- Block 16.2 (a)-(e) in
specs/SPEC.md (D-085.2). Functions of arrays, SWC tables and the
configuration; nothing here fetches or writes (the fine crop comes through a
callable).

  xy_projections   the SubImages that are xy projections (the planes' size)
  choose_projection  the one taken: of image_type survey.projection_xy_type
                   ('MinimumIntensityProjection - xy'), else the darkest
                   (D-098.2; D-089.2's rule is the fallback)
  darkest          the candidate with the lowest median grey level
  coarse_alignment allen_image_align.align_translation on the projection
  search_score     the coarse search's best mean darkness under the mask
  block_shuffle    the darkness map with its square tiles permuted
  shuffle_null     the search's best score on block-shuffled maps
  fine_crop        the full-res box of the skeleton under a transform
  fine_level       the finest downsample level the fine crop may use
  fine_search      the fine stage's direct search
  align_record     coarse, null, fine and the verdict, as one record
  verdict          set aside or not, with the reasons
  full_transform   the keyword arguments of cell.to_image_um

The transform is Block 0's (allen_image_align, equations (1)-(2)): full-res
X = x / res0 + dx, Y = (H - y / res0) + dy with a y-flip, y / res0 + dy
without. Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import math

import numpy as np

import allen_image_align as aia
import allen_image_io as aio

TYPES = (3, 4)               # SWC basal and apical dendrite: the skeleton aligned (Block 0's default)


def xy_projections(images, rel_tol):
    """The rows of list_images' table, other than Primary, whose width and
    height each lie within rel_tol (relative) of the usable Primary planes'
    median width and height (plane_table): the xy projections."""
    planes = aio.plane_table(images)
    W, H = float(planes["width"].median()), float(planes["height"].median())
    other = images[images["image_type"] != "Primary"]
    ok = ((other["width"].astype(float) - W).abs() <= rel_tol * W) & \
         ((other["height"].astype(float) - H).abs() <= rel_tol * H)
    return other[ok].reset_index(drop=True)


def darkest(arrays):
    """(index of the array with the lowest median grey level, the medians)."""
    meds = [float(np.median(np.asarray(a, dtype=float))) for a in arrays]
    if not meds:
        raise ValueError("darkest: no image")
    return int(np.argmin(meds)), meds


def _type_key(t):
    return str(t).strip().lower()


def choose_projection(image_types, arrays, xy_type):
    """Block 16.2 (a) (D-098.2): (index, rule, medians) of the xy projection
    to align on, among candidates with these image_types and arrays (the
    same order). rule "type": the candidate whose image_type equals xy_type
    (case and outer spaces ignored; several such: the darkest of them);
    rule "darkest": no candidate has that type, so the darkest of all
    (D-089.2's rule, which on Allen's data takes the inverted
    'MaximumIntensityProjection - xy' where one exists, Cell 11b,
    2026-10-28). medians: every candidate's median grey level."""
    k_all, meds = darkest(arrays)
    if len(image_types) != len(meds):
        raise ValueError("choose_projection: %d types for %d images" % (len(image_types), len(meds)))
    want = _type_key(xy_type)
    typed = [i for i, t in enumerate(image_types) if _type_key(t) == want]
    if typed:
        return min(typed, key=lambda i: (meds[i], i)), "type", meds
    return k_all, "darkest", meds


def _finite(v):
    v = float(v)
    return v if math.isfinite(v) else None


def coarse_alignment(swc, P, rho_c, f_c, scfg):
    """Block 16.2 (b): align_translation on P (array px of rho_c um, factor f_c
    to full resolution), both flips; the full-res shift and flip height, and
    whether the shift lies on the search window's edge."""
    out = aia.align_translation(swc, P, rho_c, max_shift_um=scfg.align_max_shift_um, try_flip=True, refine_px=2,
                                types=TYPES)
    rows, cols = P.shape[:2]
    M = int(min(scfg.align_max_shift_um / rho_c, rows // 2 - 1, cols // 2 - 1))
    dx, dy, flip = float(out["dx_px"]), float(out["dy_px"]), bool(out["flip_y"])
    return dict(downsample=int(round(math.log2(f_c))), dx_px=dx, dy_px=dy, flip=flip, score=_finite(out["score"]),
                score_noflip=_finite(out["score_noflip"]), score_flip=_finite(out["score_flip"]), half_window_px=M,
                at_border=bool(max(abs(dx), abs(dy)) >= M - 1), shift_x_px=f_c * dx, shift_y_px=f_c * dy,
                flip_h_px=float(f_c * rows) if flip else None)


def search_masks(swc, shape, rho, types=TYPES):
    """The skeleton's masks at offset 0 without and with the y-flip (flip
    rows = the array's rows), as align_translation renders them."""
    return [aia.render_skeleton_mask(swc, shape, rho, (0.0, 0.0), None, types),
            aia.render_skeleton_mask(swc, shape, rho, (0.0, 0.0), shape[0], types)]


def search_score(D, masks, M):
    """Block 16.2 (c): the best mean of D under a mask over the integer shifts
    |dx|, |dy| <= M and the masks (both flips): align_translation's FFT
    cross-correlation and window, without its refinement, divided by the
    mask's pixel count."""
    from scipy import signal
    rows, cols = D.shape
    cy, cx = rows // 2, cols // 2
    best = -np.inf
    for S in masks:
        n = int(S.sum())
        if n == 0:
            continue
        corr = signal.fftconvolve(D, S[::-1, ::-1].astype(np.float32), mode="same")
        best = max(best, float(corr[cy - M:cy + M + 1, cx - M:cx + M + 1].max()) / n)
    return best


def block_shuffle(D, block_px, rng):
    """D with its square tiles of block_px permuted by rng (the margins that do
    not fill a tile stay in place)."""
    b = int(block_px)
    H, W = D.shape
    nh, nw = H // b, W // b
    out = np.array(D, dtype=float, copy=True)
    tiles = out[:nh * b, :nw * b].reshape(nh, b, nw, b).swapaxes(1, 2).reshape(nh * nw, b, b)
    out[:nh * b, :nw * b] = tiles[rng.permutation(nh * nw)].reshape(nh, nw, b, b).swapaxes(1, 2).reshape(nh * b, nw * b)
    return out


def shuffle_null(D, masks, M, block_px, rng, n):
    """Block 16.2 (c): search_score on n block-shuffled copies of D -- the
    maximum the same search reaches on the image's own content without the
    cell's geometry."""
    return np.array([search_score(block_shuffle(D, block_px, rng), masks, M) for _ in range(int(n))])


def _full_px(swc, rho0, transform, types=TYPES):
    sel = swc[swc["type"].isin(types)]
    x, y = aia.swc_to_full_px(sel, rho0, transform["shift_full_px"], transform["flip_y_full_h"])
    return sel, np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def fine_crop(swc, rho0, transform, margin_um, extent_px):
    """Block 16.2 (d): the full-res box (left, top, width, height) of the
    skeleton's dendrite nodes under transform, widened by margin_um and
    clipped to the image extent (W, H)."""
    _sel, x, y = _full_px(swc, rho0, transform)
    if x.size == 0:
        raise ValueError("fine_crop: the SWC has no dendrite node (types %s)" % (TYPES,))
    m = float(margin_um) / rho0
    W, H = int(extent_px[0]), int(extent_px[1])
    left, top = max(0, int(math.floor(x.min() - m))), max(0, int(math.floor(y.min() - m)))
    right, bottom = min(W, int(math.ceil(x.max() + m))), min(H, int(math.ceil(y.max() + m)))
    if right <= left or bottom <= top:
        raise ValueError("fine_crop: the skeleton lies outside the projection under the coarse transform")
    return left, top, right - left, bottom - top


def fine_level(width, height, q_coarse, scfg):
    """The smallest level q >= align_fine_downsample, below q_coarse, whose
    returned image passes Block 0's size guard (check_request_size)."""
    for q in range(int(scfg.align_fine_downsample), int(q_coarse)):
        try:
            aio.check_request_size(int(width), int(height), q)
        except ValueError:
            continue
        return q
    raise ValueError("fine_level: a %d x %d px crop is too large below level %d" % (width, height, q_coarse))


def fine_search(D, mask, half_px):
    """Block 16.2 (d): S(a, b) = the mean of D over the mask's pixels moved by
    a columns and b rows (those inside D), for |a|, |b| <= half_px. Returns
    (a*, b*, S(a*, b*), S(0, 0), S) with S indexed [b + half_px, a + half_px].
    Custom: align_translation's FFT pads the crop to twice its size
    (gigabytes at level 1) and searches a global window. Checked:
    scipy.signal.fftconvolve, scipy.signal.correlate (same padding)."""
    r, c = np.nonzero(mask)
    if r.size == 0:
        raise ValueError("fine_search: the skeleton's mask is empty in the crop")
    H, W = D.shape
    A = int(half_px)
    S = np.full((2 * A + 1, 2 * A + 1), -np.inf)
    for bi in range(2 * A + 1):
        rr = r + (bi - A)
        okr = (rr >= 0) & (rr < H)
        for ai in range(2 * A + 1):
            cc = c + (ai - A)
            ok = okr & (cc >= 0) & (cc < W)
            if ok.any():
                S[bi, ai] = float(D[rr[ok], cc[ok]].mean())
    j = np.unravel_index(int(np.argmax(S)), S.shape)
    return int(j[1] - A), int(j[0] - A), float(S[j]), float(S[A, A]), S


def crop_swc(swc, rho0, transform, left, top):
    """The SWC with x, y moved to um in the frame of a full-res crop at (left,
    top) under transform, flip applied: render_skeleton_mask then needs no
    offset and no flip."""
    out = swc.copy()
    x, y = aia.swc_to_full_px(swc, rho0, transform["shift_full_px"], transform["flip_y_full_h"])
    out["x"] = (np.asarray(x, dtype=float) - left) * rho0
    out["y"] = (np.asarray(y, dtype=float) - top) * rho0
    return out


def verdict(rec, scfg):
    """(set_aside, reasons) of Block 16.2 (e)."""
    reasons = []
    R = rec.get("score_ratio")
    if R is None or not math.isfinite(float(R)):
        reasons.append("no_null")
    elif float(R) < scfg.align_min_score_ratio:
        reasons.append("low_score_ratio")
    if rec["coarse"]["at_border"]:
        reasons.append("coarse_at_border")
    if (rec.get("fine") or {}).get("at_border"):
        reasons.append("fine_at_border")
    return bool(reasons), reasons


def full_transform(rec):
    """The keyword arguments of cell.to_image_um for an alignment record."""
    h = rec.get("flip_h_px")
    return dict(shift_full_px=(float(rec["shift_x_px"]), float(rec["shift_y_px"])),
                flip_y_full_h=None if h is None else float(h), z0_um=float(rec.get("z0_um", 0.0)))


def align_record(swc, P, q_c, extent_px, get_crop, rho0, scfg, rng):
    """Block 16.2 (b)-(e) on one cell: swc (Block 0's SWC table, um), P the
    projection at level q_c, extent_px its full-res (width, height),
    get_crop(left, top, width, height, q) -> the projection's crop at level q
    (array), rho0 the full-res pixel (um), scfg the SurveyConfig, rng the
    null's Generator. Returns (record, images): images holds the arrays and
    frames the overlay draws."""
    f_c = 2 ** int(q_c)
    rho_c = f_c * rho0
    co = coarse_alignment(swc, P, rho_c, f_c, scfg)
    D = aia.darkness_map(P)
    masks = search_masks(swc, P.shape, rho_c)
    observed = search_score(D, masks, co["half_window_px"])
    block = max(2, int(round(scfg.align_null_block_um / rho_c)))
    nul = shuffle_null(D, masks, co["half_window_px"], block, rng, scfg.align_null_n)
    nul = nul[np.isfinite(nul)]
    med = float(np.median(nul)) if nul.size else float("nan")
    mad = float(1.4826 * np.median(np.abs(nul - med))) if nul.size else float("nan")
    R = observed / med if (nul.size >= 10 and med > 0 and math.isfinite(observed)) else float("nan")
    coarse_t = dict(shift_full_px=(co["shift_x_px"], co["shift_y_px"]), flip_y_full_h=co["flip_h_px"])
    left, top, w, h = fine_crop(swc, rho0, coarse_t, scfg.align_fine_margin_um, extent_px)
    q_f = fine_level(w, h, q_c, scfg)
    f_f = 2 ** q_f
    A = np.asarray(get_crop(left, top, w, h, q_f))
    Df = aia.ridge_darkness(A, f_f * rho0, scfg.align_fine_ridge_um)
    mask = aia.render_skeleton_mask(crop_swc(swc, rho0, coarse_t, left, top), A.shape, f_f * rho0, (0.0, 0.0), None,
                                    TYPES)
    half = int(scfg.align_fine_max_shift_um / (f_f * rho0))
    a, b, S_best, S00, _S = fine_search(Df, mask, half)
    rec = dict(method="auto", shift_x_px=co["shift_x_px"] + f_f * a, shift_y_px=co["shift_y_px"] + f_f * b,
               flip_h_px=co["flip_h_px"], z0_um=float(scfg.align_z0_um), coarse=co,
               null=dict(n=int(nul.size), block_px=block, observed=_finite(observed), median=_finite(med),
                         mad=_finite(mad)), score_ratio=_finite(R),
               fine=dict(downsample=q_f, crop=[left, top, w, h], dx_px=a, dy_px=b, score=_finite(S_best),
                         score_coarse=_finite(S00), half_window_px=half, at_border=bool(max(abs(a), abs(b)) >= half)))
    rec["set_aside"], rec["reasons"] = verdict(rec, scfg)
    images = dict(coarse=P, coarse_frame=aio.CropFrame(0, 0, int(q_c), float(rho0)), fine=A,
                  fine_frame=aio.CropFrame(left, top, q_f, float(rho0)), coarse_transform=coarse_t)
    return rec, images
