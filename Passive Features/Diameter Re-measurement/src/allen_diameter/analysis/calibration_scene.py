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
  isolation_breakdown  every traced neighbour's predicted shadow (gl) in the
                       patch, the deepest over the stage's planes (D-098.4)
  shadow_depths        shadow_depth over an array of lateral distances
  shadow_margins       rho_tau for arrays of tubes: the lateral distance where
                       each tube's predicted shadow falls to tau (Block 10.S,
                       D-101.2); shadow_margin, one tube
  shadow_masks         (n, H, W) bool x 2: the pixels each plane's neighbours
                       shade, and those the node's own fitted cylinder shades
  stripe_support       the stripe along the measuring line with most clear
                       pixels, one plane (Block 10.S, D-101.3-4)
  patch_frame          the patch's shape and frame without fetching it
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


def segment_rows(scene, frame, shape, branch, reach_um, return_index=False):
    """(n', 9) float64 rows x0 y0 z0 x1 y1 z1 r0 r1 own of the scene's segments
    whose xy bounding box, grown by reach_um, meets the crop (frame, shape);
    own = 1.0 when both ends' ids are among branch.ids (plane_iteration.
    scene_segments' rule, with the reach). return_index: also the rows'
    indices in the scene's segment arrays (n',) int (2026-10-28, for the
    isolation figure of 10.F (5))."""
    H, W = int(shape[0]), int(shape[1])
    if scene.seg_xyz.shape[0] == 0:
        return (np.empty((0, 9)), np.empty(0, dtype=int)) if return_index else np.empty((0, 9))
    p = float(frame.res_um_px)
    a, b = scene.seg_xyz[:, :3], scene.seg_xyz[:, 3:]
    g = float(reach_um)
    x_lo, x_hi = (frame.left - 0.5) * p - g, (frame.left + W - 0.5) * p + g
    y_lo, y_hi = (frame.top - 0.5) * p - g, (frame.top + H - 0.5) * p + g
    meets = ((np.maximum(a[:, 0], b[:, 0]) >= x_lo) & (np.minimum(a[:, 0], b[:, 0]) <= x_hi)
             & (np.maximum(a[:, 1], b[:, 1]) >= y_lo) & (np.minimum(a[:, 1], b[:, 1]) <= y_hi))
    in_branch = {int(x) for x in np.asarray(branch.ids).ravel()}
    own = np.array([float(int(c) in in_branch and int(q) in in_branch) for c, q in scene.seg_ids])
    rows = np.column_stack([a, b, scene.seg_radius[:, 0], scene.seg_radius[:, 1], own])[meets]
    return (rows, np.flatnonzero(meets)) if return_index else rows


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
_GL_CACHE = {16: _GL}


def _gauss_legendre(r, sigma):
    """The Gauss-Legendre rule for the tube integral: 16 nodes, more when the
    blur is narrow against the tube (node spacing about sigma / 3 or finer,
    so that a Gaussian of SD sigma is resolved on [-r, r]); at most 512."""
    n = int(min(512, max(16, math.ceil(6.0 * r / max(float(sigma), 1e-6)))))
    if n not in _GL_CACHE:
        _GL_CACHE[n] = leggauss(n)
    return _GL_CACHE[n]


def shadow_depths(rho, r, mu, sigma, B):
    """shadow_depth over an array of lateral distances rho (um): the dip (gl) of
    a flat tube of radius r (um) and absorption mu (1/um), blurred by a Gaussian
    of SD sigma (um), B int_{-r}^{r} (1 - exp(-2 mu sqrt(r^2 - v'^2)))
    phi_sigma(rho - v') dv' (Gauss-Legendre; 16 nodes, more for a blur narrow
    against the tube, _gauss_legendre), shaped as rho."""
    x, w = _gauss_legendre(r, sigma)
    rho = np.asarray(rho, dtype=float)
    vp = r * x
    deficit = 1.0 - np.exp(-2.0 * mu * np.sqrt(np.maximum(r * r - vp * vp, 0.0)))
    s = max(float(sigma), 1e-6)
    g = np.exp(-0.5 * ((rho[..., None] - vp) / s) ** 2) / (math.sqrt(2.0 * math.pi) * s)
    return B * r * np.sum(w * deficit * g, axis=-1)


def shadow_depth(rho, r, mu, sigma, B):
    """The dip (gl) of a flat tube of radius r (um) and absorption mu (1/um),
    blurred by a Gaussian of SD sigma (um), at lateral distance rho (um) from
    its axis: B int_{-r}^{r} (1 - exp(-2 mu sqrt(r^2 - v'^2))) phi_sigma(rho -
    v') dv' (Gauss-Legendre, 16 nodes). Custom: the partition integral of the
    renderer's faint tube in closed quadrature; no library function."""
    return float(shadow_depths(np.array(float(rho)), r, mu, sigma, B))


# ------------------------------------------------------ the stripe (10.S) --

def _depths_rule(rho, r, sigma, mu, B, n):
    """shadow_depths for arrays rho, r, sigma (m,) of m tubes, all on the n-node Gauss-Legendre rule."""
    x, w = _GL_CACHE[n] if n in _GL_CACHE else _GL_CACHE.setdefault(n, leggauss(n))
    vp = r[:, None] * x[None, :]
    deficit = 1.0 - np.exp(-2.0 * mu * np.sqrt(np.maximum(r[:, None] ** 2 - vp ** 2, 0.0)))
    s = np.maximum(sigma, 1e-6)[:, None]
    g = np.exp(-0.5 * ((rho[:, None] - vp) / s) ** 2) / (math.sqrt(2.0 * math.pi) * s)
    return B * r * np.sum(w * deficit * g, axis=1)


def shadow_margins(r, mu, sigma, B, tau, n_iter=48):
    """rho_tau (um) for arrays r and sigma of the same shape: for each tube
    (radius r, absorption mu, blur sigma, background B) the lateral distance at
    which its predicted shadow (shadow_depths, on _gauss_legendre's rule for
    that tube) falls to tau (gl), by bisection on [0, r + 6 sigma] -- the shadow
    decreases with rho >= 0 (a symmetric unimodal deficit blurred by a
    Gaussian, Wintner). -1.0 where the shadow is at most tau everywhere
    (nothing is shaded) or r <= 0; r + 6 sigma where it is still above tau
    there (never, for tau > 0: the Gaussian tail is below 1e-8 of the dip);
    all -1.0 when B or mu is not finite and positive. Block 10.S, D-101.2.
    Custom: a bisection vectorized over many tubes (one call per plane for
    every segment of a cell); scipy.optimize.brentq / bisect are scalar, and
    scipy.optimize.elementwise.find_root needs SciPy >= 1.15, which Colab and
    the cluster's environment do not guarantee."""
    r = np.asarray(r, dtype=float)
    s = np.maximum(np.asarray(sigma, dtype=float), 1e-6)
    r, s = np.broadcast_arrays(r, s)
    out = np.full(r.shape, -1.0)
    B, mu, tau = float(B), float(mu), float(tau)
    if not (math.isfinite(B) and B > 0 and math.isfinite(mu) and mu > 0):
        return out
    fr, fs, res = r.ravel(), s.ravel(), out.ravel()
    need = np.clip(np.ceil(6.0 * fr / fs), 16, 512).astype(int)       # _gauss_legendre's rule, element by element
    valid = fr > 0
    for n in np.unique(need[valid]):
        idx = np.flatnonzero(valid & (need == n))
        for c in range(0, idx.size, 20000):
            j = idx[c:c + 20000]
            rr, ss = fr[j], fs[j]
            lo, hi = np.zeros(j.size), rr + 6.0 * ss
            d0 = _depths_rule(lo, rr, ss, mu, B, int(n))
            dh = _depths_rule(hi, rr, ss, mu, B, int(n))
            m = np.where(d0 <= tau, -1.0, np.where(dh > tau, hi, np.nan))
            todo = np.isnan(m)
            if todo.any():
                a, b, r2, s2 = lo[todo], hi[todo], rr[todo], ss[todo]
                for _ in range(int(n_iter)):
                    mid = 0.5 * (a + b)
                    above = _depths_rule(mid, r2, s2, mu, B, int(n)) > tau
                    a = np.where(above, mid, a)
                    b = np.where(above, b, mid)
                m[todo] = 0.5 * (a + b)
            res[j] = m
    return res.reshape(r.shape)


def shadow_margin(r, mu, sigma, B, tau, n_iter=48):
    """rho_tau (um) of one tube: shadow_margins for a single (r, sigma)."""
    return float(shadow_margins(np.array([float(r)]), mu, np.array([float(sigma)]), B, tau, n_iter)[0])


def _window(frame, shape, x0, x1, y0, y1):
    """(r0, r1, c0, c1): the pixel rows and columns whose centres can lie in [x0, x1] x [y0, y1] (um), padded by one
    pixel and clipped to the crop; empty when r0 >= r1 or c0 >= c1."""
    p = float(frame.res_um_px)
    H, W = int(shape[0]), int(shape[1])
    c0 = max(0, int(math.ceil(x0 / p - frame.left)) - 1)
    c1 = min(W, int(math.floor(x1 / p - frame.left)) + 2)
    r0 = max(0, int(math.ceil(y0 / p - frame.top)) - 1)
    r1 = min(H, int(math.floor(y1 / p - frame.top)) + 2)
    return r0, r1, c0, c1


def shadow_masks(shape, frame, rows, soma, planes_z, B_bar, mu, cfg, d_own_um=None, sigma_soma_um=None):
    """(neighbour (n, H, W) bool, own (n, H, W) bool): the pixels of each plane
    that a traced neighbour shades -- pixel centres within rho_tau of the
    segment, rho_tau = shadow_margin at the segment's larger end radius, the
    wider end's blur sigma_r(delta_q,k), the pilot's B_bar and mu, tau =
    isolation_depth_gl (Block 10.S, D-101.2) -- and those the node's own
    branch shades, with the radius max(d_own_um / 2, the end radius): the
    fitted cylinder (D-101.4; d_own_um None or not finite: Allen's radius).
    rows: segment_rows (its own column tells the two apart); soma (m, 3) x, y,
    r, shaded as a tube of radius r_s at sigma_soma_um (default: sigma_r at
    half the span of planes_z, 10.I's convention -- a caller testing one plane
    of a stage passes the stage's own, as stripe_reasons does). Without a
    finite positive B_bar and mu (a phantom's node, or a pilot without a fit)
    the shadow cannot be predicted: the margin is then 10.P's geometric r +
    far_pixel_sigmas x sigma_r (the same rule for the own branch and the
    neighbours). The margins of every segment and plane come from one
    vectorized shadow_margins call; each segment's distances are computed on
    the pixels within its largest margin of it only (exact: a pixel outside
    that window is farther than every margin)."""
    H, W = int(shape[0]), int(shape[1])
    px, py = pixel_centres(shape, frame)
    z = np.asarray(planes_z, dtype=float).reshape(-1)
    rc, tau = cfg.renderer, float(cfg.calibration.isolation_depth_gl)
    n_sigma = float(cfg.calibration.far_pixel_sigmas)
    predicted = all(math.isfinite(float(x)) and float(x) > 0 for x in (B_bar, mu))
    nb = np.zeros((z.size, H, W), dtype=bool)
    own = np.zeros((z.size, H, W), dtype=bool)
    r_own = 0.5 * float(d_own_um) if d_own_um is not None and math.isfinite(float(d_own_um)) and d_own_um > 0 else 0.0
    rows = np.atleast_2d(np.asarray(rows, dtype=float)).reshape(-1, 9)

    def margins_of(r, sigma):
        if predicted:
            return shadow_margins(r, float(mu), sigma, float(B_bar), tau)
        return r + n_sigma * sigma

    if rows.shape[0]:
        is_own = rows[:, 8] >= 0.5
        r = np.maximum(rows[:, 6], rows[:, 7])
        r = np.where(is_own, np.maximum(r, r_own), r)
        delta = np.maximum(np.abs(rows[:, 2:3] - z[None, :]), np.abs(rows[:, 5:6] - z[None, :]))   # (n_rows, n)
        margins = margins_of(np.broadcast_to(r[:, None], delta.shape), sigma_r(delta, rc))
        # each segment's largest distance to the crop (at a corner: the distance to a segment is convex)
        cx_ = (frame.left + np.array([0, W - 1, 0, W - 1])) * float(frame.res_um_px)
        cy_ = (frame.top + np.array([0, 0, H - 1, H - 1])) * float(frame.res_um_px)
        for q in np.flatnonzero(np.any(margins >= 0, axis=1)):
            sg, mq = rows[q], margins[q]
            target = own if is_own[q] else nb
            far_corner = float(np.max(background.segment_distance(cx_, cy_, sg[0:2], sg[3:5])))
            whole = mq >= far_corner                      # these planes are shaded on the whole crop
            target[whole] = True
            if whole.all():
                continue
            reach = float(mq[~whole].max())
            if reach < 0:
                continue
            r0, r1, c0, c1 = _window(frame, (H, W), min(sg[0], sg[3]) - reach, max(sg[0], sg[3]) + reach,
                                     min(sg[1], sg[4]) - reach, max(sg[1], sg[4]) + reach)
            if r0 >= r1 or c0 >= c1:
                continue
            dist = background.segment_distance(px[r0:r1, c0:c1], py[r0:r1, c0:c1], sg[0:2], sg[3:5])
            for k in np.flatnonzero((mq >= 0) & ~whole):
                target[k, r0:r1, c0:c1] |= dist <= mq[k]
    soma = np.atleast_2d(np.asarray(soma, dtype=float)).reshape(-1, 3)
    if soma.shape[0]:
        if sigma_soma_um is None:
            span = 0.5 * (z.max() - z.min()) if z.size > 1 else 0.0
            sigma_soma_um = float(sigma_r(span, rc))
        ms = np.atleast_1d(margins_of(soma[:, 2], np.full(soma.shape[0], float(sigma_soma_um))))
        for (x, y, _r), margin in zip(soma, ms):
            if margin >= 0:
                r0, r1, c0, c1 = _window(frame, (H, W), x - margin, x + margin, y - margin, y + margin)
                if r0 < r1 and c0 < c1:
                    nb[:, r0:r1, c0:c1] |= (np.hypot(px[r0:r1, c0:c1] - x, py[r0:r1, c0:c1] - y) <= margin)[None]
    return nb, own


def fitted_own_rows(rows, cx, cy):
    """The own branch's rows moved across to the pilot's fitted centre (Block 10.S, D-101.4, the fitted
    cylinder; the tester's F2 of 2026-10-28): the offset is from the trace's point nearest (cx, cy) to (cx, cy),
    applied to every own row (x and y only). (n_own, 9); empty when there is no own row or the centre is
    not finite."""
    rows = np.atleast_2d(np.asarray(rows, dtype=float)).reshape(-1, 9)
    own = rows[rows[:, 8] >= 0.5]
    if not own.shape[0] or not (math.isfinite(float(cx)) and math.isfinite(float(cy))):
        return np.empty((0, 9))
    c = np.array([float(cx), float(cy)])
    a, b = own[:, 0:2], own[:, 3:5]
    ab = b - a
    t = np.clip(np.sum((c - a) * ab, axis=1) / np.maximum(np.sum(ab * ab, axis=1), 1e-12), 0.0, 1.0)
    p = a + t[:, None] * ab
    j = int(np.argmin(np.hypot(*(c - p).T)))
    off = c - p[j]
    moved = own.copy()
    moved[:, [0, 3]] += off[0]
    moved[:, [1, 4]] += off[1]
    return moved


def _walk(present, blocked_own, blocked_nb, j0, step, n):
    """One side of the stripe walk over the columns j0, j0 + step, ... (n of
    them): (j_own, j_end) -- j_own the first column after the own cylinder,
    the contiguous block of own-shaded columns about the origin (a column a
    neighbour shades ends it); j_end the first column at or after j_own that
    is shaded (by anyone) or absent."""
    j, taken = j0, 0
    while taken < n and present[j] and blocked_own[j] and not blocked_nb[j]:
        j += step
        taken += 1
    j_own = j
    while taken < n and present[j] and not blocked_own[j] and not blocked_nb[j]:
        j += step
        taken += 1
    return j_own, j


def stripe_support(neighbour2d, own2d, frame, cx, cy, theta, cfg):
    """The stripe of one plane (Block 10.S, D-101.3-4): in the measuring line's
    frame (v along (-sin theta, cos theta), u along (cos theta, sin theta),
    origin (cx, cy)), for each half-width W of stripe_half_widths_um the band
    |u| <= W is cut into pixel columns along v; from v = 0 outward on each
    side the walk passes the columns the own branch shades (to v_own) and
    goes on to the first column a neighbour or the own branch shades, or the
    patch's edge (L); the band's clear pixels in the run columns count; the W
    with most pixels wins (the smaller on a tie). The own cylinder is the
    contiguous block of own-shaded columns about the origin (_walk). Returns (mask (H, W) bool,
    info): info = dict(w_um, v_own_minus_um, v_own_plus_um, l_minus_um,
    l_plus_um, side_minus_um, side_plus_um, n_pixels, ok, reason) with ok when
    n_pixels >= stripe_min_pixels and both sides' runs >= stripe_min_side_um,
    reason "" | "too_few" | "side_short"."""
    cal = cfg.calibration
    nb = np.asarray(neighbour2d, dtype=bool)
    ow = np.asarray(own2d, dtype=bool)
    H, W_ = nb.shape
    p = float(frame.res_um_px)
    px, py = pixel_centres(nb.shape, frame)
    dx, dy = px - float(cx), py - float(cy)
    nv = np.array([-math.sin(theta), math.cos(theta)])
    tv = np.array([math.cos(theta), math.sin(theta)])
    v = dx * nv[0] + dy * nv[1]
    u = dx * tv[0] + dy * tv[1]
    col = np.rint(v / p).astype(int)              # pixel columns along the line, 0 at the origin
    jmin, jmax = int(col.min()), int(col.max())
    ncol = jmax - jmin + 1
    clear = ~(nb | ow)
    best = None
    for w_um in sorted(float(x) for x in cal.stripe_half_widths_um):
        band = np.abs(u) <= w_um
        cj = col[band] - jmin
        present = np.bincount(cj, minlength=ncol) > 0
        b_own = np.bincount(cj, weights=ow[band].astype(float), minlength=ncol) > 0
        b_nb = np.bincount(cj, weights=nb[band].astype(float), minlength=ncol) > 0
        n_clear = np.bincount(cj, weights=clear[band].astype(float), minlength=ncol)
        # pad one absent column at each end so the walk stops at the patch's edge
        present = np.concatenate([[False], present, [False]])
        b_own = np.concatenate([[False], b_own, [False]])
        b_nb = np.concatenate([[False], b_nb, [False]])
        n_clear = np.concatenate([[0.0], n_clear, [0.0]])
        j0 = -jmin + 1                              # the origin's column in the padded arrays
        jo_p, je_p = _walk(present, b_own, b_nb, j0, +1, ncol + 1)
        jo_m, je_m = _walk(present, b_own, b_nb, j0, -1, ncol + 1)
        run_p = range(jo_p, je_p)                   # columns of the + side's run (origin column included once)
        run_m = range(je_m + 1, jo_m + 1)
        cols = set(run_p) | set(run_m)
        n = float(sum(n_clear[j] for j in cols))
        info = dict(w_um=w_um, v_own_plus_um=(jo_p - j0) * p, v_own_minus_um=(j0 - jo_m) * p,
                    l_plus_um=(je_p - j0) * p, l_minus_um=(j0 - je_m) * p,
                    side_plus_um=(je_p - jo_p) * p, side_minus_um=(jo_m - je_m) * p, n_pixels=int(round(n)),
                    _cols=cols, _band=band)
        if best is None or n > best["n_pixels"]:
            best = info
    cols, band = best.pop("_cols"), best.pop("_band")
    mask = band & clear & np.isin(col - jmin + 1, np.fromiter(cols, dtype=int, count=len(cols)))
    best["n_pixels"] = int(mask.sum())
    reason = ""
    if best["n_pixels"] < int(cal.stripe_min_pixels):
        reason = "too_few"
    elif min(best["side_plus_um"], best["side_minus_um"]) < float(cal.stripe_min_side_um):
        reason = "side_short"
    best["ok"] = reason == ""
    best["reason"] = reason
    return mask, best


def patch_frame(cx, cy, half_um, res0_um):
    """((H, W), CropFrame) of the patch about (cx, cy) without fetching it: the
    frame fetch_patch's provider returns (downsample 0)."""
    from allen_image_io import CropFrame
    left, top, width, height = patch_bounds(cx, cy, half_um, res0_um)
    return (int(height), int(width)), CropFrame(int(left), int(top), 0, float(res0_um))


def isolation_breakdown(branch, index, scene, cx, cy, half_um, planes_z, B_bar, mu, cfg):
    """Every traced neighbour's predicted shadow in the node's +-half_um patch
    (10.I, D-081.1), one dict per neighbour in the scene's order -- the
    segments not of the node's own branch, then the soma discs: kind
    ("segment" or "soma"), index (in scene.seg_xyz or scene.soma), child_id
    and parent_id (None for a soma), rho_um (lateral distance to the patch
    square, 0 when it crosses it), r_um, depth_gl (the deepest over the
    planes of planes_z; 0.0 when the neighbour lies beyond 6 sigma of the
    patch in every plane). A segment's radius is the larger of its ends', its
    defocus from plane k is tried at both ends' depths; a soma disc is a
    tube of radius r_s at the blur of the stage's largest offset. B_bar and
    mu are the pilot's. isolation_depth is the deepest of these; the figure
    of 10.F (5) draws them (D-098.4)."""
    rc = cfg.renderer
    z = np.asarray(planes_z, dtype=float).reshape(-1)
    lo = np.array([cx - half_um, cy - half_um])
    hi = np.array([cx + half_um, cy + half_um])
    in_branch = {int(x) for x in np.asarray(branch.ids).ravel()}
    out = []
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
        out.append(dict(kind="segment", index=int(q), child_id=c_id, parent_id=p_id, rho_um=float(rho), r_um=r,
                        depth_gl=float(depth)))
    span = 0.5 * (z.max() - z.min()) if z.size > 1 else 0.0
    sig_soma = float(sigma_r(span, rc))
    for m, (x, y, r) in enumerate(np.atleast_2d(np.asarray(scene.soma, dtype=float)).reshape(-1, 3)):
        rho = float(np.hypot(*(np.clip([x, y], lo, hi) - np.array([x, y]))))
        depth = shadow_depth(rho, float(r), mu, sig_soma, B_bar)
        out.append(dict(kind="soma", index=int(m), child_id=None, parent_id=None, rho_um=rho, r_um=float(r),
                        depth_gl=float(depth)))
    return out


def isolation_depth(branch, index, scene, cx, cy, half_um, planes_z, B_bar, mu, cfg):
    """(max depth gl, worst id): the deepest shadow a traced neighbour throws
    into the node's +-half_um patch in any plane of planes_z (10.I, D-081.1)
    -- the deepest of isolation_breakdown's, the first in its order on a tie;
    the id is the segment's child id or "soma". Returns (0.0, None) when
    nothing is traced near the patch."""
    worst, worst_id = 0.0, None
    for rec in isolation_breakdown(branch, index, scene, cx, cy, half_um, planes_z, B_bar, mu, cfg):
        if rec["depth_gl"] > worst:
            worst, worst_id = rec["depth_gl"], (rec["child_id"] if rec["kind"] == "segment" else "soma")
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
