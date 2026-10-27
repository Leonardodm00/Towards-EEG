"""What Allen's projections are -- Block 16.3 in specs/SPEC.md (D-090.1;
2026-10-27). Each xy projection of a cell against pixelwise operators over
its planes, on one full-resolution window. Arrays in, records out: nothing
here fetches, reads or writes a file.

  OPERATORS         min, mean, median, max over the usable planes
  plane_operators   the operators' images of a stack (n_planes, H, W)
  shift_match       a projection crop (with a margin of s px) against one
                    operator image: the best integer offset, r, the line, RMS
  classify          the best operator, its lead over the next, the verdict
  check_projection  every operator for one projection crop
  place_window      the window's corner, moved inside the images
  aligned_crop      the projection over the window at an offset
  soma_center       the SWC's soma in full-res pixels under Block 0's transform

Offsets: (a, b) is the projection's pixel minus the planes' pixel that hold
the same content, P(x + a, y + b) ~ O(x, y).

Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import math

import numpy as np

OPERATORS = ("min", "mean", "median", "max")
NAN = float("nan")


def plane_operators(stack, valid=None):
    """{op: (H, W) float64} over the planes of stack (n_planes, H, W) whose
    valid entry is True (all of them when valid is None): the pixelwise
    minimum, mean, median and maximum. Fewer than 2 valid planes, or a
    non-finite value among them: ValueError."""
    s = np.asarray(stack)
    if s.ndim != 3:
        raise ValueError("plane_operators: the stack must be (n_planes, H, W), got shape %s" % (s.shape,))
    v = np.ones(s.shape[0], dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    if v.shape != (s.shape[0],):
        raise ValueError("plane_operators: valid needs one entry per plane (%d), got shape %s" % (s.shape[0], v.shape))
    if int(v.sum()) < 2:
        raise ValueError("plane_operators: %d valid planes; at least 2 are needed" % int(v.sum()))
    if not v.all():
        s = s[v]
    if np.issubdtype(s.dtype, np.floating) and not np.isfinite(s).all():
        raise ValueError("plane_operators: the stack holds non-finite values")
    return dict(min=s.min(axis=0).astype(np.float64), mean=s.mean(axis=0, dtype=np.float64),
                median=np.median(s, axis=0).astype(np.float64), max=s.max(axis=0).astype(np.float64))


def _correlation(p, oc, so):
    """Pearson r of the flat array p with the centred oc (norm so); NaN when p is constant."""
    if np.ptp(p) == 0:
        return NAN
    pc = p - p.mean()
    return float(np.dot(pc, oc) / (math.sqrt(float(np.dot(pc, pc))) * so))


def shift_match(P_big, O, max_shift):
    """P_big (H + 2s, W + 2s): the projection's crop over the window widened
    by s = max_shift px on every side; O (H, W): an operator's image over the
    window. For every integer (a, b) with |a|, |b| <= s, r(a, b) = the Pearson
    correlation of P_big[s + b : s + b + H, s + a : s + a + W] with O (NaN when
    either is constant). At the first maximum in the order b = -s..s, then
    a = -s..s: dict(r, shift_x_px=a, shift_y_px=b, slope, intercept, rms_gl),
    the line P ~ intercept + slope O by least squares and the RMS of its
    residual (grey levels). No finite r: every value NaN, the shifts None."""
    s = int(max_shift)
    P_big = np.asarray(P_big, dtype=np.float64)
    O = np.asarray(O, dtype=np.float64)
    if O.ndim != 2 or s < 0:
        raise ValueError("shift_match: O must be 2-D and max_shift >= 0")
    H, W = O.shape
    if P_big.shape != (H + 2 * s, W + 2 * s):
        raise ValueError("shift_match: P_big must be %s for O %s and max_shift %d, got %s"
                         % ((H + 2 * s, W + 2 * s), O.shape, s, P_big.shape))
    out = dict(r=NAN, shift_x_px=None, shift_y_px=None, slope=NAN, intercept=NAN, rms_gl=NAN)
    o = O.ravel()
    if np.ptp(o) == 0 or not (np.isfinite(o).all() and np.isfinite(P_big).all()):
        return out
    oc = o - o.mean()
    so = math.sqrt(float(np.dot(oc, oc)))
    best = None
    for b in range(-s, s + 1):
        for a in range(-s, s + 1):
            r = _correlation(P_big[s + b:s + b + H, s + a:s + a + W].ravel(), oc, so)
            if math.isfinite(r) and (best is None or r > best[0]):
                best = (r, a, b)
    if best is None:
        return out
    r, a, b = best
    p = P_big[s + b:s + b + H, s + a:s + a + W].ravel()
    slope = float(np.dot(p - p.mean(), oc) / np.dot(oc, oc))
    intercept = float(p.mean() - slope * o.mean())
    rms = float(np.sqrt(np.mean((p - intercept - slope * o) ** 2)))
    return dict(r=float(r), shift_x_px=int(a), shift_y_px=int(b), slope=slope, intercept=intercept, rms_gl=rms)


def classify(by_op, min_r, min_ratio):
    """(best, margin, ratio, verdict) from {op: shift_match dict}: the
    operator of the largest finite r (r1), the lead r1 - r2 over the next
    finite one, and ratio = (1 - r2^2) / (1 - r1^2), how many times the
    best's unexplained variance the next operator's line leaves (inf when r1
    = 1 or the best is the only finite one). "matches <op>" when r1 >= min_r
    and ratio >= min_ratio, otherwise "unclear: best <op> r <r1>, ratio <q>";
    no finite r: (None, NaN, NaN, "unclear: no finite r"). The ratio, not the
    lead, decides: two operators' images can correlate at 0.95-0.97 while
    the wrong one's line leaves several times the residual (Block 16.3 (b))."""
    fin = sorted(((float(v["r"]), op) for op, v in by_op.items()
                  if v.get("r") is not None and math.isfinite(float(v["r"]))), reverse=True)
    if not fin:
        return None, NAN, NAN, "unclear: no finite r"
    r1, op1 = fin[0]
    if len(fin) > 1:
        r2 = fin[1][0]
        margin = r1 - r2
        unexplained = 1.0 - r1 * r1
        ratio = (1.0 - r2 * r2) / unexplained if unexplained > 0 else float("inf")
    else:
        margin = ratio = float("inf")
    if r1 >= float(min_r) and ratio >= float(min_ratio):
        return op1, margin, ratio, "matches %s" % op1
    return op1, margin, ratio, "unclear: best %s r %.4f, ratio %.3g" % (op1, r1, ratio)


def check_projection(P_big, ops, max_shift, min_r, min_ratio):
    """Every operator of ops (in OPERATORS' order) against one projection
    crop: dict(by_operator {op: shift_match}, best, margin, ratio, verdict)."""
    by_op = {op: shift_match(P_big, ops[op], max_shift) for op in OPERATORS if op in ops}
    best, margin, ratio, verdict = classify(by_op, min_r, min_ratio)
    return dict(by_operator=by_op, best=best, margin=margin, ratio=ratio, verdict=verdict)


def place_window(cx, cy, half, margin, width, height):
    """(left, top), full-res px, of the window of 2 half x 2 half px centred at
    the nearest pixel of (cx, cy) (halves up), moved so that the window
    widened by margin px on every side lies inside [0, width) x [0, height).
    ValueError when it cannot fit or the centre is not finite."""
    h, m = int(half), int(margin)
    if h < 1 or m < 0:
        raise ValueError("place_window: half >= 1 and margin >= 0, got %r, %r" % (half, margin))
    if int(width) < 2 * (h + m) or int(height) < 2 * (h + m):
        raise ValueError("place_window: a window of %d px with a margin of %d px does not fit in %d x %d px"
                         % (2 * h, m, int(width), int(height)))
    if not (math.isfinite(float(cx)) and math.isfinite(float(cy))):
        raise ValueError("place_window: the centre (%r, %r) is not finite" % (cx, cy))
    left = min(max(int(math.floor(float(cx) + 0.5)) - h, m), int(width) - 2 * h - m)
    top = min(max(int(math.floor(float(cy) + 0.5)) - h, m), int(height) - 2 * h - m)
    return left, top


def aligned_crop(P_big, a, b, max_shift):
    """The projection over the window at offset (a, b) (both None: no
    offset): P_big[s + b : s + b + H, s + a : s + a + W] with H, W the shape of
    P_big less 2 s."""
    s = int(max_shift)
    P = np.asarray(P_big)
    a, b = (0, 0) if a is None or b is None else (int(a), int(b))
    if abs(a) > s or abs(b) > s:
        raise ValueError("aligned_crop: offset (%d, %d) beyond the margin %d" % (a, b, s))
    H, W = P.shape[0] - 2 * s, P.shape[1] - 2 * s
    return P[s + b:s + b + H, s + a:s + a + W]


def soma_center(swc, rho0, shift_full_px=(0.0, 0.0), flip_y_full_h=None):
    """(cx, cy), full-res px: the mean of the SWC's soma rows -- type 1 with
    parent -1, else every type-1 row -- under Block 0's transform
    (allen_image_align.swc_to_full_px). ValueError without a type-1 row."""
    import allen_image_align as aia
    t = np.asarray(swc["type"].values)
    rows = swc[(t == 1) & (np.asarray(swc["parent"].values) == -1)]
    if not len(rows):
        rows = swc[t == 1]
    if not len(rows):
        raise ValueError("soma_center: the SWC has no soma (type-1) row; give the window's centre (--center X,Y)")
    x, y = aia.swc_to_full_px(rows, float(rho0), tuple(shift_full_px), flip_y_full_h)
    return float(np.mean(x)), float(np.mean(y))
