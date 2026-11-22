"""The soma in the morphology -- Block 47 in specs/SPEC.md (D-215.2, D-216.3, D-222.5).

The fitted ellipsoid (Block 46) enters the reconstruction as the sphere of equal membrane area, r_eq = sqrt(S / 4 pi),
at the fitted centre: the SWC's first type-1 node (row order) is moved there with radius r_eq, any other type-1 node is
removed and its children are re-parented to the first. Each stem's first node moves along the stem's own direction onto
the fitted surface; when the stem's next nodes lie inside the surface they are removed and the first node is placed
where the stem leaves it (D-222.5). Radii of the stems are kept.

Frames. The image frame of Blocks 44-46 (x, y um from the full image's pixel centres, z um in stage units) and the SWC
frame, related by analysis.cell.to_image_um (a shift, an optional reflection of y, z - z0); image_to_swc_um is its
inverse (lengths are kept). The ellipsoid E(theta_g), theta_g = (x0, y0, z0, a, b, c, phi, mu) (Block 45), in its own
frame
    u = (x - x0) cos phi + (y - y0) sin phi,   v = -(x - x0) sin phi + (y - y0) cos phi,   w = z - z0,
    Q(P) = (u/a)^2 + (v/b)^2 + (w/c)^2 - 1        (negative inside, 0 on the surface).
A line P + t d (d != 0): Q(P + t d) = A t^2 + 2 B t + C, A = (d_u/a)^2 + (d_v/b)^2 + (d_w/c)^2,
B = P_u d_u/a^2 + P_v d_v/b^2 + P_w d_w/c^2, C = Q(P); roots t = (-B -+ sqrt(B^2 - A C)) / A, NaN when B^2 < A C.

The stems: the rows whose parent is a type-1 node and whose own type is not 1 (the axon included). For a stem with first
node P1 and P2 its first child in row order, d = (P2 - P1) / |P2 - P1| (P1 a leaf: d from the fitted centre to P1):
    |Q(P1)| <= Q_SURFACE_TOL   unmoved, "on surface";
    Q(P1) > 0 (outside)        P1 -> P1 + t* d, t* the root of smaller |t|; no root: unmoved, "missed";
    Q(P1) < 0 (inside)         t* the positive root (the stem leaves the surface there); t* < |P2 - P1|:
                               P1 -> P1 + t* d; otherwise the chain P1, P2, ..., Pm is walked while each node from P2 on
                               is inside (Q < 0) and has exactly one child, Pm the first node with Q >= 0; P1 moves to
                               the crossing on the segment P(m-1) Pm, P2 .. P(m-1) are removed, Pm is re-parented to P1.
A node inside with two or more children before the exit -- P1 itself included -- leaves the stem unmoved,
"branch inside" (D-222.5: "a branch point inside the soma leaves its stem unmoved, flagged"); a leaf inside -- P1 itself
included -- unmoved, "ends inside"; otherwise "moved". Outlines for the figures: the silhouette (the in-plane ellipse
(a, b, phi) about (x0, y0)), the section at a depth z (that ellipse scaled by sqrt(1 - ((z - z0)/c)^2)) and the side
outline (the ellipse (a, c) in the (u, w) plane).

Library calls: numpy; analysis.cell.to_image_um (the forward map). Custom code (marked below): the line-ellipsoid
quadratic, the stems' walk. The SWC editor is loading.swc_io.edited_lines.

No file access, no plotting. Pure ASCII, LF, for the cluster.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .cell import to_image_um

Q_SURFACE_TOL = 1e-12        # SPEC Block 47: |Q| <= this counts as on the surface
FLAGS = ("moved", "on surface", "missed", "branch inside", "ends inside")
_THETA_NAMES = ("x0_um", "y0_um", "z0_um", "a_um", "b_um", "c_um", "phi_rad", "mu_per_um")


@dataclass(frozen=True)
class SomaEdit:
    """The corrected soma (SPEC Block 47). xyz (N, 3) SWC frame um, radius (N,) um, parent (N,) ids, keep (N,) bool --
    every row of the input SWC, edited where moved or re-parented; soma_row the kept type-1 row; stems a tuple of dicts
    (row, id, type, flag in {"moved", "on surface", "missed", "branch inside", "ends inside"}, old_um (3,), new_um (3,)
    -- the first node before and after, SWC frame um -- moved_um, removed_ids); removed_soma_ids the other type-1
    nodes' ids."""
    xyz: np.ndarray
    radius: np.ndarray
    parent: np.ndarray
    keep: np.ndarray
    soma_row: int
    stems: Tuple[dict, ...]
    removed_soma_ids: Tuple[int, ...]


# ------------------------------------------------------------------ checks --

def _finite(x, name):
    """x as a float when it is one finite number."""
    try:
        v = np.asarray(x, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("%s must be one finite number, got %r" % (name, x)) from None
    if v.ndim != 0 or not math.isfinite(float(v)):
        raise ValueError("%s must be one finite number, got %r" % (name, x))
    return float(v)


def _positive(x, name):
    """x as a float when it is one finite number > 0."""
    v = _finite(x, name)
    if v <= 0.0:
        raise ValueError("%s must be > 0, got %r" % (name, x))
    return v


def _theta(theta_g):
    """theta_g (8,) as eight floats (x0, y0, z0, a, b, c, phi, mu), checked as Block 45 states."""
    try:
        th = np.asarray(theta_g, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("theta_g must be 8 numbers %s" % (_THETA_NAMES,)) from None
    if th.shape != (8,):
        raise ValueError("theta_g must have shape (8,) %s, got shape %s" % (_THETA_NAMES, th.shape))
    x0, y0, z0, a, b, c, phi, mu = (float(t) for t in th)
    if not all(math.isfinite(t) for t in (x0, y0, z0, phi)):
        raise ValueError("theta_g: x0, y0, z0 and phi must be finite, got %r" % ((x0, y0, z0, phi),))
    if not all(math.isfinite(t) and t > 0.0 for t in (a, b, c)):
        raise ValueError("theta_g: the semi-axes a, b, c must be finite and > 0 (um), got %r" % ((a, b, c),))
    if not (math.isfinite(mu) and mu >= 0.0):
        raise ValueError("theta_g: mu must be finite and >= 0 (1/um), got %r" % (mu,))
    return x0, y0, z0, a, b, c, phi, mu


def _points(P, name):
    """P as a finite float64 array of shape (n, 3)."""
    try:
        A = np.asarray(P, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("%s must be an (n, 3) array of numbers" % name) from None
    if A.ndim != 2 or A.shape[1] != 3:
        raise ValueError("%s must have shape (n, 3), got %s" % (name, A.shape))
    if not np.all(np.isfinite(A)):
        raise ValueError("%s must be finite" % name)
    return A


def _directions(d, n):
    """d (3,) or (n, 3) as a finite (n, 3) array with no zero row."""
    try:
        D = np.asarray(d, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("d must be an array of numbers") from None
    if D.shape == (3,):
        D = np.broadcast_to(D, (n, 3))
    elif D.shape != (n, 3):
        raise ValueError("d must have shape (3,) or (%d, 3), got %s" % (n, D.shape))
    if not np.all(np.isfinite(D)):
        raise ValueError("d must be finite")
    if np.any(np.all(D == 0.0, axis=1)):
        raise ValueError("every direction d must be nonzero")
    return D


def _alignment(res0_um, shift_full_px, flip_y_full_h, z0_um):
    """The arguments of analysis.cell.to_image_um, checked: res0 > 0, a finite shift (2,), flip None or finite, z0."""
    p = _positive(res0_um, "res0_um")
    try:
        s = np.asarray(shift_full_px, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("shift_full_px must be two finite numbers, got %r" % (shift_full_px,)) from None
    if s.shape != (2,) or not np.all(np.isfinite(s)):
        raise ValueError("shift_full_px must be two finite numbers, got %r" % (shift_full_px,))
    flip = None if flip_y_full_h is None else _finite(flip_y_full_h, "flip_y_full_h")
    return p, (float(s[0]), float(s[1])), flip, _finite(z0_um, "z0_um")


def _count(n, name, floor):
    """n as an int when it is a whole number >= floor (an int, or a float with an integral value; not a bool)."""
    if isinstance(n, (bool, np.bool_)):
        raise ValueError("%s must be a whole number >= %d, got %r" % (name, floor, n))
    try:
        v = float(n)
    except (TypeError, ValueError):
        raise ValueError("%s must be a whole number >= %d, got %r" % (name, floor, n)) from None
    if not (math.isfinite(v) and v == math.floor(v) and v >= floor):
        raise ValueError("%s must be a whole number >= %d, got %r" % (name, floor, n))
    return int(v)


# ------------------------------------------------------------------ frames --

def image_to_swc_um(xyz_img, res0_um, shift_full_px=(0.0, 0.0), flip_y_full_h=None, z0_um=0.0):
    """-> (n, 3) float64 um in the SWC frame: the inverse of analysis.cell.to_image_um with the same arguments,
    x = (x_img / p - s_x) p, y = (F - (y_img / p - s_y)) p when flipped (else (y_img / p - s_y) p), z = z_img + z0.
    xyz_img (n, 3) um, image frame. ValueError for points not (n, 3) or not finite, res0_um not > 0, a bad alignment."""
    P = _points(xyz_img, "xyz_img")
    p, (sx, sy), flip, z0 = _alignment(res0_um, shift_full_px, flip_y_full_h, z0_um)
    x = (P[:, 0] / p - sx) * p
    y = P[:, 1] / p - sy
    if flip is not None:
        y = flip - y
    return np.column_stack([x, y * p, P[:, 2] + z0])


def _rotate(dx, dy, dz, phi):
    """(n, 3): (dx, dy, dz) turned into the ellipsoid's axes, u = dx cos phi + dy sin phi, v = -dx sin phi + dy cos phi."""
    cp, sp = math.cos(phi), math.sin(phi)
    return np.column_stack([dx * cp + dy * sp, -dx * sp + dy * cp, dz])


def fit_frame(P_img, theta_g):
    """-> (n, 3) um: the points P_img (n, 3) in the fitted ellipsoid's frame, (u, v, w) = Block 45's rotated in-plane
    coordinates about (x0, y0) and w = z - z0. theta_g (8,) as Block 45."""
    x0, y0, z0, _a, _b, _c, phi, _mu = _theta(theta_g)
    P = _points(P_img, "P_img")
    return _rotate(P[:, 0] - x0, P[:, 1] - y0, P[:, 2] - z0, phi)


def _q(U, a, b, c):
    """Q from the ellipsoid-frame coordinates U (n, 3)."""
    return (U[:, 0] / a) ** 2 + (U[:, 1] / b) ** 2 + (U[:, 2] / c) ** 2 - 1.0


def ellipsoid_q(P_img, theta_g):
    """-> (n,): Q = (u/a)^2 + (v/b)^2 + (w/c)^2 - 1 at the points P_img (n, 3) um (negative inside, 0 on the surface)."""
    _x0, _y0, _z0, a, b, c, _phi, _mu = _theta(theta_g)
    return _q(fit_frame(P_img, theta_g), a, b, c)


def line_crossings(P_img, d, theta_g):
    """-> (t_lo, t_hi) (n,) each, t_lo <= t_hi: the roots of Q(P + t d) = 0, NaN where the line misses (B^2 < A C).
    P_img (n, 3) um, d (n, 3) or (3,) nonzero directions (not normalised: t is in units of |d|). ValueError for a zero
    or non-finite direction, shapes that disagree, a bad theta_g."""
    _x0, _y0, _z0, a, b, c, phi, _mu = _theta(theta_g)
    P = _points(P_img, "P_img")
    D = _directions(d, P.shape[0])
    axes = np.array([a, b, c])
    Ps = fit_frame(P, theta_g) / axes                      # (P_u/a, P_v/b, P_w/c)
    Ds = _rotate(D[:, 0], D[:, 1], D[:, 2], phi) / axes    # (d_u/a, d_v/b, d_w/c)
    A = np.sum(Ds * Ds, axis=1)                            # > 0: d != 0
    B = np.sum(Ps * Ds, axis=1)
    C = np.sum(Ps * Ps, axis=1) - 1.0
    disc = B * B - A * C
    hit = disc >= 0.0
    # Custom: no library routine solves a batch of real quadratics with NaN for a miss (numpy.roots takes one
    # polynomial and returns complex roots). Checked: numpy.roots, numpy.polynomial.polynomial.polyroots. The roots
    # (-B -+ sqrt(B^2 - A C)) / A are taken in the cancellation-free form q = -(B + sign(B) sqrt(B^2 - A C)),
    # t = q / A and C / q: the same two numbers, without the loss of digits of -B + sqrt(...) when B^2 >> |A C|.
    # q = 0 only when B = 0 and B^2 = A C, so C = 0: the double root t = 0.
    sq = np.sqrt(np.where(hit, disc, 0.0))
    q = -(B + np.copysign(sq, B))
    nz = q != 0.0
    t1 = q / A
    t2 = np.where(nz, C / np.where(nz, q, 1.0), t1)
    t_lo = np.where(hit, np.minimum(t1, t2), np.nan)
    t_hi = np.where(hit, np.maximum(t1, t2), np.nan)
    return t_lo, t_hi


# ---------------------------------------------------------------- outlines --

def ellipse_outline(theta_g, z_um=None, n=256):
    """-> (x (n,), y (n,)) um image frame: the fitted silhouette (z_um None) or the section at depth z_um (the
    silhouette scaled by sqrt(1 - ((z - z0)/c)^2); empty (0,) arrays when |z - z0| >= c). A closed outline: n points
    at the angles linspace(0, 2 pi, n), the last repeating the first. ValueError for n not a whole number >= 3, z_um
    not finite, a bad theta_g."""
    x0, y0, z0, a, b, c, phi, _mu = _theta(theta_g)
    n = _count(n, "n", 3)
    scale = 1.0
    if z_um is not None:
        r = (_finite(z_um, "z_um") - z0) / c
        if abs(r) >= 1.0:
            return np.empty(0), np.empty(0)
        scale = math.sqrt(1.0 - r * r)
    t = np.linspace(0.0, 2.0 * math.pi, n)
    u, v = scale * a * np.cos(t), scale * b * np.sin(t)
    cp, sp = math.cos(phi), math.sin(phi)
    return x0 + u * cp - v * sp, y0 + u * sp + v * cp


def side_outline(theta_g, n=256):
    """-> (u (n,), w (n,)) um: the ellipse (a, c) in the ellipsoid's (u, w) plane, about its centre (0, 0); closed as
    ellipse_outline. ValueError for n not a whole number >= 3, a bad theta_g."""
    _x0, _y0, _z0, a, _b, c, _phi, _mu = _theta(theta_g)
    n = _count(n, "n", 3)
    t = np.linspace(0.0, 2.0 * math.pi, n)
    return a * np.cos(t), c * np.sin(t)


# ------------------------------------------------------------------- stems --

def stem_rows(swc):
    """-> ndarray[int]: the rows of the stems' first nodes (parent a type-1 node, own type not 1), ascending."""
    types = np.asarray(swc.types)
    pidx = np.asarray(swc.parent_index())
    parent_soma = np.zeros(types.shape, dtype=bool)
    has = pidx >= 0
    parent_soma[has] = types[pidx[has]] == 1
    return np.flatnonzero(parent_soma & (types != 1))


def _children(pidx):
    """Each row's children, ascending rows."""
    out = [[] for _ in range(len(pidx))]
    for r, q in enumerate(pidx):
        if q >= 0:
            out[int(q)].append(r)
    return out


def _stem(r1, P, Q, children, theta):
    """One stem (SPEC Block 47): -> (flag, P1's new position (3,) image frame or None, rows removed, the row
    re-parented to P1 or None). P (N, 3) image frame, Q (N,) at P, both of the input SWC."""
    x0, y0, z0 = theta[0], theta[1], theta[2]
    P1, q1, kids = P[r1], float(Q[r1]), children[r1]
    if abs(q1) <= Q_SURFACE_TOL:
        return "on surface", None, (), None
    if q1 > 0.0:                                           # outside: the nearer crossing along the stem's line
        d = (P[kids[0]] - P1) if kids else (P1 - np.array([x0, y0, z0]))
        L = float(np.linalg.norm(d))
        if L == 0.0:                                       # no direction (P2 on P1): no line, no crossing
            return "missed", None, (), None
        d = d / L
        t_lo, t_hi = (float(t[0]) for t in line_crossings(P1[None, :], d, theta))
        if not math.isfinite(t_lo):
            return "missed", None, (), None
        t = t_lo if abs(t_lo) <= abs(t_hi) else t_hi        # both roots on one side (C > 0)
        return "moved", P1 + t * d, (), None
    # inside (D-222.5): a branch point or a leaf inside leaves the stem unmoved, P1 included
    if len(kids) == 0:
        return "ends inside", None, (), None
    if len(kids) >= 2:
        return "branch inside", None, (), None
    r2 = kids[0]
    seg = P[r2] - P1
    L = float(np.linalg.norm(seg))
    if L > 0.0:
        d = seg / L
        t_star = float(line_crossings(P1[None, :], d, theta)[1][0])   # the positive root (C < 0)
        if t_star < L:
            return "moved", P1 + t_star * d, (), None
    # Custom: the stems' walk (a chain of single children until the first node with Q >= 0); no library function.
    # Checked: none applies (a tree walk on the SWC's parent ids). The exit lies at or beyond P2 (or P2 is on P1).
    chain, seen = [r1, r2], {r1, r2}
    while Q[chain[-1]] < 0.0:
        nxt = children[chain[-1]]
        if len(nxt) == 0:
            return "ends inside", None, (), None
        if len(nxt) >= 2:
            return "branch inside", None, (), None
        if nxt[0] in seen:
            raise ValueError("the SWC's parent ids form a cycle through row %d" % nxt[0])
        chain.append(nxt[0])
        seen.add(nxt[0])
    r_prev, r_m = chain[-2], chain[-1]                     # Q(P(m-1)) < 0 <= Q(Pm)
    seg = P[r_m] - P[r_prev]
    s = float(line_crossings(P[r_prev][None, :], seg, theta)[1][0])
    s = min(max(s, 0.0), 1.0)                              # the root in [0, 1] (rounding at the ends)
    return "moved", P[r_prev] + s * seg, tuple(chain[1:-1]), r_m


def _kept_soma_parent(swc, soma_row, soma_id, removed_ids):
    """The kept soma's parent id: its own, or, when that is a removed type-1 node, the nearest ancestor that is not
    one (-1 when there is none or the chain returns to the kept soma)."""
    index = swc.index_of()
    p, seen = int(swc.parent[soma_row]), set()
    while p in removed_ids and p not in seen:
        seen.add(p)
        p = int(swc.parent[index[p]])
    return -1 if (p in removed_ids or p == soma_id) else p


def correct_soma(swc, theta_g, r_eq_um, res0_um, shift_full_px=(0.0, 0.0), flip_y_full_h=None, z0_um=0.0):
    """-> SomaEdit: the SWC (loading.swc_io.SWC, not mutated) with its soma replaced by the sphere of radius r_eq_um at
    the fitted centre and its stems' first nodes on the fitted surface (SPEC Block 47); theta_g (8,) in the image frame;
    res0_um and the alignment (shift_full_px, flip_y_full_h, z0_um) as analysis.cell.to_image_um. Rows not edited keep
    their input values bit for bit. ValueError for no type-1 node, a bad theta_g, r_eq_um not finite and > 0, res0_um
    not > 0, a bad alignment, non-finite SWC coordinates."""
    theta = _theta(theta_g)
    x0, y0, z0, a, b, c, phi, _mu = theta
    r_eq = _positive(r_eq_um, "r_eq_um")
    p, shift, flip, z0a = _alignment(res0_um, shift_full_px, flip_y_full_h, z0_um)
    frame = dict(shift_full_px=shift, flip_y_full_h=flip, z0_um=z0a)
    ids = np.asarray(swc.ids, dtype=np.int64)
    types = np.asarray(swc.types)
    soma_rows = np.flatnonzero(types == 1)
    if soma_rows.size == 0:
        raise ValueError("the SWC has no type-1 (soma) node")
    xyz_in = _points(swc.xyz, "swc.xyz")
    xyz = np.array(xyz_in, dtype=float, copy=True)
    radius = np.array(swc.radius, dtype=float, copy=True)
    parent = np.array(swc.parent, dtype=np.int64, copy=True)
    keep = np.ones(ids.size, dtype=bool)

    # the soma: the first type-1 node at the fitted centre with radius r_eq; the others removed, children re-parented
    soma_row, removed_rows = int(soma_rows[0]), soma_rows[1:]
    soma_id = int(ids[soma_row])
    xyz[soma_row] = image_to_swc_um(np.array([[x0, y0, z0]]), p, **frame)[0]
    radius[soma_row] = r_eq
    keep[removed_rows] = False
    removed_soma_ids = tuple(int(i) for i in ids[removed_rows])
    if removed_rows.size:
        parent[np.isin(parent, ids[removed_rows])] = soma_id
        parent[soma_row] = _kept_soma_parent(swc, soma_row, soma_id, set(removed_soma_ids))

    # the stems, on the input's positions and topology
    P = to_image_um(xyz_in, p, **frame)
    Q = _q(_rotate(P[:, 0] - x0, P[:, 1] - y0, P[:, 2] - z0, phi), a, b, c)
    children = _children(np.asarray(swc.parent_index()))
    stems = []
    for r1 in (int(r) for r in stem_rows(swc)):
        flag, new_img, removed, r_m = _stem(r1, P, Q, children, theta)
        if new_img is not None:
            xyz[r1] = image_to_swc_um(new_img[None, :], p, **frame)[0]
        if removed:
            keep[list(removed)] = False
        if r_m is not None:
            parent[r_m] = ids[r1]
        old = np.array(xyz_in[r1], dtype=float)
        new = np.array(xyz[r1], dtype=float)
        stems.append({"row": r1, "id": int(ids[r1]), "type": int(types[r1]), "flag": flag, "old_um": old,
                      "new_um": new, "moved_um": float(np.linalg.norm(new - old)),
                      "removed_ids": tuple(int(ids[k]) for k in removed)})
    return SomaEdit(xyz=xyz, radius=radius, parent=parent, keep=keep, soma_row=soma_row, stems=tuple(stems),
                    removed_soma_ids=removed_soma_ids)
