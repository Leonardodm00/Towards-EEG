"""The calibration's patch, its planar far-pixel background, and the
isolation and straightness criteria -- Block 10.P and 10.I in specs/SPEC.md
(D-078.1, D-079, D-081.1-3; 2026-10-23), shared by the near scan (Block 10)
and the far scan (Block 15).

  fetch_patch          the +-half_um square of pixels about the node's centre
                       over the stage's planes, in the fetch_zblock contract
  segment_rows         the scene's segments that can reach the patch (their
                       bounding box grown by reach_um meets it), as the rows
                       of survey.node_planes: x0 y0 z0 x1 y1 z1 r0 r1 own
  far_pixel_mask       (n, H, W) bool, True where a pixel centre lies farther
                       than max(r0, r1) + n_sigma sigma_r(delta_q,k) from every
                       traced segment q (the node's own branch included: its
                       shadow is the dip) and farther than r_s + n_sigma
                       sigma_soma from every soma disc, plane by plane
  planar_background    the least-squares plane b0 + bx (x - cx) + by (y - cy)
                       through a plane's far pixels, refitted once without the
                       pixels beyond trim_sigmas x 1.4826 MAD
  background_on_line   the plane read at points of the measuring line
  isolation_depth      the deepest shadow (gl) a traced neighbour -- not the
                       node's own branch -- throws into the patch in any of
                       the stage's planes, under the renderer's current law
  straightness_excess  how far the trace strays from its own straight line
                       within +-half_um along the branch (um)

All depths of defocus use the renderer's current law model.kernel.sigma_r.
Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import math

import numpy as np
from numpy.polynomial.legendre import leggauss

from . import background, path
from ..model.kernel import sigma_r

OWN = 1.0          # the "own" column of a segment row: both ends on the node's branch


# ------------------------------------------------------------ the patch ----

def patch_bounds(cx, cy, half_um, res0_um):
    """(left, top, width, height) in full-resolution pixels of the square whose
    pixel centres (left + col) p, (top + row) p lie within +-half_um of (cx, cy)."""
    p = float(res0_um)
    left = int(math.ceil((cx - half_um) / p - 1e-9))
    right = int(math.floor((cx + half_um) / p + 1e-9))
    top = int(math.ceil((cy - half_um) / p - 1e-9))
    bottom = int(math.floor((cy + half_um) / p + 1e-9))
    return left, top, right - left + 1, bottom - top + 1


def fetch_patch(provider, cx, cy, half_um, k_lo, k_hi, res0_um):
    """provider(left, top, width, height, k_lo, k_hi) -> (block, ks, valid, frame) for the patch."""
    left, top, width, height = patch_bounds(cx, cy, half_um, res0_um)
    return provider(left, top, width, height, int(k_lo), int(k_hi))


def pixel_centres(shape, frame):
    """(px, py) (H, W) um: the pixel centres of a crop."""
    H, W = int(shape[0]), int(shape[1])
    p = float(frame.res_um_px)
    rows, cols = np.mgrid[0:H, 0:W]
    return (frame.left + cols) * p, (frame.top + rows) * p


def segment_rows(scene, frame, shape, branch, reach_um):
    """(n', 9) float64 rows x0 y0 z0 x1 y1 z1 r0 r1 own of the scene's segments
    whose xy bounding box, grown by reach_um, meets the crop (frame, shape);
    own = 1.0 when both ends' ids are among branch.ids (plane_iteration.
    scene_segments' rule, with the reach)."""
    H, W = int(shape[0]), int(shape[1])
    if scene.seg_xyz.shape[0] == 0:
        return np.empty((0, 9))
    p = float(frame.res_um_px)
    a, b = scene.seg_xyz[:, :3], scene.seg_xyz[:, 3:]
    g = float(reach_um)
    x_lo, x_hi = (frame.left - 0.5) * p - g, (frame.left + W - 0.5) * p + g
    y_lo, y_hi = (frame.top - 0.5) * p - g, (frame.top + H - 0.5) * p + g
    meets = ((np.maximum(a[:, 0], b[:, 0]) >= x_lo) & (np.minimum(a[:, 0], b[:, 0]) <= x_hi)
             & (np.maximum(a[:, 1], b[:, 1]) >= y_lo) & (np.minimum(a[:, 1], b[:, 1]) <= y_hi))
    in_branch = {int(x) for x in np.asarray(branch.ids).ravel()}
    own = np.array([float(int(c) in in_branch and int(q) in in_branch) for c, q in scene.seg_ids])
    return np.column_stack([a, b, scene.seg_radius[:, 0], scene.seg_radius[:, 1], own])[meets]


def segment_margins(rows, planes_z, n_sigma, rcfg):
    """(n', n_planes) um: max(r0, r1) + n_sigma sigma_r(delta_q,k), delta_q,k =
    the larger |z_end - z_k| of the two ends (the wider blur)."""
    rows = np.atleast_2d(np.asarray(rows, dtype=float)).reshape(-1, 9)
    z = np.asarray(planes_z, dtype=float).reshape(1, -1)
    if rows.shape[0] == 0:
        return np.empty((0, z.size))
    d = np.maximum(np.abs(rows[:, 2:3] - z), np.abs(rows[:, 5:6] - z))
    return np.maximum(rows[:, 6], rows[:, 7])[:, None] + float(n_sigma) * sigma_r(d, rcfg)


def far_pixel_mask(shape, frame, rows, soma, planes_z, cfg, sigma_soma_um=None):
    """(n_planes, H, W) bool: the far pixels of each plane (10.P). rows: the
    segment rows of segment_rows; soma (m, 3) x, y, r; planes_z (n_planes,) um;
    sigma_soma_um: the soma's blur (default sigma_r at the largest |z_node -
    z_k| the caller passes through planes_z's spread: here the stage's largest
    offset is taken as half the span of planes_z)."""
    H, W = int(shape[0]), int(shape[1])
    px, py = pixel_centres(shape, frame)
    z = np.asarray(planes_z, dtype=float).reshape(-1)
    rc, n_sigma = cfg.renderer, cfg.calibration.far_pixel_sigmas
    near = np.zeros((z.size, H, W), dtype=bool)
    rows = np.atleast_2d(np.asarray(rows, dtype=float)).reshape(-1, 9)
    if rows.shape[0]:
        margins = segment_margins(rows, z, n_sigma, rc)
        for q, s in enumerate(rows):
            dist = background.segment_distance(px, py, s[0:2], s[3:5])
            for k in range(z.size):
                near[k] |= dist <= margins[q, k]
    soma = np.atleast_2d(np.asarray(soma, dtype=float)).reshape(-1, 3)
    if soma.shape[0]:
        if sigma_soma_um is None:
            span = 0.5 * (z.max() - z.min()) if z.size > 1 else 0.0
            sigma_soma_um = float(sigma_r(span, rc))
        for x, y, r in soma:
            d = np.hypot(px - x, py - y)
            near |= (d <= r + n_sigma * float(sigma_soma_um))[None]
    return ~near


# --------------------------------------------------------- the background --

def planar_background(plane2d, far, frame, cx, cy, trim_sigmas, min_frac):
    """(b (3,), n_used, ok): the least-squares plane B(x, y) = b0 + bx (x - cx) +
    by (y - cy) through the far pixels of one plane (numpy lstsq), refitted once
    without the pixels whose residual exceeds trim_sigmas x 1.4826 x MAD. ok is
    False (b NaN) when fewer than min_frac of the pixels are far, or fewer than
    6 remain after trimming."""
    img = np.asarray(plane2d, dtype=float)
    far = np.asarray(far, dtype=bool)
    nan3 = np.full(3, np.nan)
    if far.shape != img.shape or far.mean() < float(min_frac) or far.sum() < 6:
        return nan3, 0, False
    px, py = pixel_centres(img.shape, frame)
    x, y, val = (px[far] - cx), (py[far] - cy), img[far]
    keep = np.isfinite(val)
    if keep.sum() < 6:
        return nan3, 0, False
    A = np.column_stack([np.ones(keep.sum()), x[keep], y[keep]])
    b = np.linalg.lstsq(A, val[keep], rcond=None)[0]
    res = val[keep] - A @ b
    mad = float(np.median(np.abs(res - np.median(res))))
    if mad > 0:
        inner = np.abs(res) <= float(trim_sigmas) * 1.4826 * mad
        if inner.sum() >= 6 and inner.sum() < inner.size:
            b = np.linalg.lstsq(A[inner], val[keep][inner], rcond=None)[0]
            return b, int(inner.sum()), True
    return b, int(keep.sum()), True


def background_on_line(b, cx, cy, xy):
    """B(x, y) at the points xy (n, 2) um: b0 + bx (x - cx) + by (y - cy)."""
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    return b[0] + b[1] * (xy[:, 0] - cx) + b[2] * (xy[:, 1] - cy)


# ---------------------------------------------------------- isolation -----

def _rect_distance(a, b, lo, hi):
    """Distance (um) from the segment a-b (2-vectors) to the axis-aligned
    rectangle [lo, hi] (2-vectors): 0 when they meet, else the minimum over the
    rectangle's corners to the segment and the segment's ends to the rectangle
    (both convex: the minimum is attained at a vertex of one of them)."""
    a, b, lo, hi = (np.asarray(v, dtype=float) for v in (a, b, lo, hi))
    ends = np.array([a, b])
    if np.any(np.all((ends >= lo) & (ends <= hi), axis=1)):
        return 0.0
    # does the segment cross the rectangle's boundary? test the four edges
    corners = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
    for i in range(4):
        c, d = corners[i], corners[(i + 1) % 4]
        if _segments_cross(a, b, c, d):
            return 0.0
    d_corner = min(float(background.segment_distance(np.array([c[0]]), np.array([c[1]]), a, b)[0]) for c in corners)
    d_end = min(float(np.hypot(*(np.clip(e, lo, hi) - e))) for e in ends)
    return min(d_corner, d_end)


def _segments_cross(a, b, c, d):
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
    return (o1 * o2 < 0) and (o3 * o4 < 0)


_GL = leggauss(16)


def shadow_depth(rho, r, mu, sigma, B):
    """The dip (gl) of a flat tube of radius r (um) and absorption mu (1/um),
    blurred by a Gaussian of SD sigma (um), at lateral distance rho (um) from
    its axis: B int_{-r}^{r} (1 - exp(-2 mu sqrt(r^2 - v'^2))) phi_sigma(rho -
    v') dv' (Gauss-Legendre, 16 nodes). Custom: the partition integral of the
    renderer's faint tube in closed quadrature; no library function."""
    x, w = _GL
    vp = r * x
    deficit = 1.0 - np.exp(-2.0 * mu * np.sqrt(np.maximum(r * r - vp * vp, 0.0)))
    s = max(float(sigma), 1e-6)
    g = np.exp(-0.5 * ((rho - vp) / s) ** 2) / (math.sqrt(2.0 * math.pi) * s)
    return float(B * r * np.sum(w * deficit * g))


def isolation_depth(branch, index, scene, cx, cy, half_um, planes_z, B_bar, mu, cfg):
    """(max depth gl, worst id): the deepest shadow a traced neighbour throws
    into the node's +-half_um patch in any plane of planes_z (10.I, D-081.1).
    Neighbours: the scene's segments with at least one end off branch.ids
    (the node's own branch is excluded) and the soma discs; a segment's
    radius is the larger of its ends', its defocus from plane k is tried at
    both ends' depths; a soma disc is a tube of radius r_s at the blur of the
    stage's largest offset. B_bar and mu are the pilot's. Returns (0.0, None)
    when nothing is traced near the patch."""
    rc = cfg.renderer
    z = np.asarray(planes_z, dtype=float).reshape(-1)
    lo = np.array([cx - half_um, cy - half_um])
    hi = np.array([cx + half_um, cy + half_um])
    in_branch = {int(x) for x in np.asarray(branch.ids).ravel()}
    worst, worst_id = 0.0, None
    for q in range(scene.seg_xyz.shape[0]):
        c_id, p_id = int(scene.seg_ids[q, 0]), int(scene.seg_ids[q, 1])
        if c_id in in_branch and p_id in in_branch:
            continue
        a, b = scene.seg_xyz[q, :3], scene.seg_xyz[q, 3:]
        rho = _rect_distance(a[:2], b[:2], lo, hi)
        r = float(max(scene.seg_radius[q, 0], scene.seg_radius[q, 1]))
        depth = 0.0
        for z_end in (a[2], b[2]):
            for s in np.atleast_1d(sigma_r(np.abs(z_end - z), rc)):
                if rho > r + 6.0 * s:            # the Gaussian beyond 6 sigma: below 1e-8 of the dip
                    continue
                depth = max(depth, shadow_depth(rho, r, mu, s, B_bar))
        if depth > worst:
            worst, worst_id = depth, c_id
    span = 0.5 * (z.max() - z.min()) if z.size > 1 else 0.0
    sig_soma = float(sigma_r(span, rc))
    for x, y, r in np.atleast_2d(np.asarray(scene.soma, dtype=float)).reshape(-1, 3):
        rho = float(np.hypot(*(np.clip([x, y], lo, hi) - np.array([x, y]))))
        depth = shadow_depth(rho, float(r), mu, sig_soma, B_bar)
        if depth > worst:
            worst, worst_id = depth, "soma"
    return float(worst), worst_id


# -------------------------------------------------------- straightness ----

def straightness_excess(branch, index, half_um):
    """The largest distance (um) of the branch's traced points with |s - s_index|
    <= half_um along the trace from the total-least-squares line through those
    points (analysis.path.tls_direction; 10.I, D-081.3): the trace's own
    departure from straightness, independent of the pilot's heading and tilt
    estimates. 0.0 with fewer than three points."""
    s = np.asarray(branch.s_um, dtype=float)
    sel = np.abs(s - s[int(index)]) <= float(half_um)
    P = np.asarray(branch.xyz_um, dtype=float)[sel]
    if P.shape[0] < 3:
        return 0.0
    try:
        t = path.tls_direction(P)
    except ValueError:
        return 0.0
    Q = P - P.mean(axis=0)
    perp = Q - np.outer(Q @ t, t)
    return float(np.max(np.linalg.norm(perp, axis=1)))
