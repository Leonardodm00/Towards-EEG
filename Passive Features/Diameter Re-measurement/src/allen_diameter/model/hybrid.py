"""The hybrid absorption -- Block 21 in specs/SPEC.md (D-084, D-092; the
renderers document of 2026-10-09, s.3.8, Eqs. 13-14, on main; I-004).

Each slab's absorbed light is taken from the condenser's rays; Block 4 then
blurs it with the defocus kernel (procedure Eq. 6), unchanged. Directions s
are unit vectors with s_z > 0 and weights W (Block 9's quadrature of the
evenly filled cone); the light travels along +s for light_direction +1 and
along -s for -1. For a point p and a direction s, with [t1, t2] the interval
of the line p + t s inside the tube (Block 2 line_interval; t = arc length),

    l_back(p; s) = |[t1, t2] & (-inf, 0]|   (light +1)
                   |[t1, t2] & [0, +inf)|   (light -1)

is the length inside the tube of the ray that reaches p, before it reaches p
(-t1, resp. t2, for p inside the tube), and

    h(p) = < exp(-mu l_back(p; s)) / s_z >_W                    (ray_history)

the light each ray has left at p (its own history) times its path per unit
depth, averaged over the cone. For slab j at a column (x, y) whose vertical
line meets the tube inside the slab over the length a_j / mu (S4), with
midpoint zbar_j,

    Delta A_j(x, y) = a_j(x, y) h(x, y, zbar_j(x, y))           (slab_absorbed)

-- the document's Eq. 13, a midpoint rule in depth, with the slab's inside
part in place of the whole slab. As mu -> 0, h -> <1/s_z>_W (1.447 at NA 1.4,
n_oil 1.515), the faint limit of the ray world (S10).

l_back is Block 2's chord (line_interval: squashed section, both end cuts)
written for many points and directions at once: the quadratic's
point-direction cross term is a matrix product. The smoke test checks it
against line_interval itself.

Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import numpy as np

from . import geometry

_CHUNK_ENTRIES = 1 << 20      # points x directions per chunk (~8 MB per float64 array)
_TINY = 1e-15                 # geometry._TINY: a direction that does not move along the cut's coordinate


def _points(points):
    P = np.asarray(points, dtype=float)
    if P.ndim == 1 and P.size == 3:
        P = P.reshape(1, 3)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("points must have shape (n, 3), got %s" % (P.shape,))
    if not np.all(np.isfinite(P)):
        raise ValueError("points must be finite")
    return P


def _directions(S, W=None):
    S = np.asarray(S, dtype=float)
    if S.ndim != 2 or S.shape[1] != 3 or S.shape[0] < 1:
        raise ValueError("directions S must have shape (N, 3) with N >= 1, got %s" % (S.shape,))
    if not np.all(np.isfinite(S)) or not np.all(S[:, 2] > 0):
        raise ValueError("directions must be finite with s_z > 0")
    if np.max(np.abs((S * S).sum(axis=1) - 1.0)) > 1e-12:
        raise ValueError("directions must be unit vectors")
    if W is None:
        return S, None
    W = np.asarray(W, dtype=float)
    if W.shape != (S.shape[0],) or not np.all(np.isfinite(W)) or np.any(W < 0) or abs(W.sum() - 1.0) > 1e-12:
        raise ValueError("weights W must be (N,), finite, >= 0 and sum to 1")
    return S, W


def _light(light_direction):
    if light_direction not in (+1, -1):
        raise ValueError("light_direction must be +1 or -1")
    return int(light_direction)


def _back_chunk(P, S, tube, light):
    """(n, N) l_back for points P (n, 3) and unit directions S (N, 3).

    line_interval's algebra: q0 = S^-1 (P - c), s0 = S^-1 s with
    S = diag(1, 1, k); A = |s0|^2 - (s0.t0)^2, B = q0.s0 - (q0.t0)(s0.t0),
    C = |q0|^2 - (q0.t0)^2 - r^2; t1,2 = (-B -+ sqrt(B^2 - A C)) / A; a line
    parallel to the axis is inside for all t when C < 0; then the end cut."""
    k = tube.aspect
    sq = np.array([1.0, 1.0, 1.0 / k])
    q = P - np.asarray(tube.c, dtype=float)
    q0 = q * sq
    s0 = S * sq
    t0 = tube.t_hat0
    st = s0 @ t0                                        # (N,)
    qt = q0 @ t0                                        # (n,)
    ss = (s0 * s0).sum(axis=1)
    A = ss - st ** 2                                    # (N,)
    C = (q0 * q0).sum(axis=1) - qt ** 2 - tube.r ** 2   # (n,)
    B = q0 @ s0.T                                       # (n, N), the cross term
    B -= np.outer(qt, st)
    par = A <= 1e-14 * np.maximum(ss, 1e-300)           # (N,): along the axis
    Asafe = np.where(par, 1.0, A)
    disc = B * B
    disc -= np.outer(C, Asafe)
    ok = (disc > 0) & ~par[None, :]
    sqd = np.sqrt(np.where(ok, disc, 0.0))
    inside_par = par[None, :] & (C < 0)[:, None]
    t1 = np.where(ok, (-B - sqd) / Asafe, np.where(inside_par, -np.inf, 0.0))
    t2 = np.where(ok, (-B + sqd) / Asafe, np.where(inside_par, np.inf, 0.0))
    hit = ok | inside_par
    U = tube.half_length
    if U is not None:
        if tube.end_cut == "axial":
            a, b = qt, st                               # the axial coordinate qt + t st within [-U, U]
        else:
            eu = tube.e_u
            a, b = q @ eu, S @ eu                       # |u| <= U, u in real space
        moving = np.abs(b) > _TINY                      # (N,)
        bsafe = np.where(moving, b, 1.0)
        ta = (-U - a)[:, None] / bsafe[None, :]
        tb = (U - a)[:, None] / bsafe[None, :]
        within = (np.abs(a) <= U)[:, None]
        c_lo = np.where(moving[None, :], np.minimum(ta, tb), np.where(within, -np.inf, np.inf))
        c_hi = np.where(moving[None, :], np.maximum(ta, tb), np.where(within, np.inf, -np.inf))
        t1 = np.maximum(t1, c_lo)
        t2 = np.minimum(t2, c_hi)
        hit &= t2 > t1
    with np.errstate(invalid="ignore"):                 # inf - inf on lines that miss; masked by hit
        if light == +1:
            ell = np.minimum(t2, 0.0) - t1              # |[t1, t2] & (-inf, 0]|
        else:
            ell = t2 - np.maximum(t1, 0.0)              # |[t1, t2] & [0, inf)|
        return np.where(hit, np.clip(ell, 0.0, None), 0.0)


def back_lengths(points, tube, S, light_direction=+1):
    """l_back (um), (n, N): for each point (n, 3) um and unit direction of S
    (N, 3), s_z > 0, the length inside the tube of the ray that reaches the
    point, before it reaches it (light along +s for light_direction +1, -s for
    -1); 0 where the line misses the tube, inf inside an uncapped tube along a
    direction parallel to its axis. For tests and small cases: ray_history
    reduces chunk by chunk without holding (n, N)."""
    P = _points(points)
    S, _ = _directions(S)
    light = _light(light_direction)
    out = np.empty((P.shape[0], S.shape[0]))
    step = max(1, _CHUNK_ENTRIES // S.shape[0])
    for a in range(0, P.shape[0], step):
        out[a:a + step] = _back_chunk(P[a:a + step], S, tube, light)
    return out


def ray_history(points, tube, mu, S, W, light_direction=+1):
    """h(p) = < exp(-mu l_back(p; s)) / s_z >_W at each point (n, 3) um, (n,)
    dimensionless; mu (1/um) >= 0. At mu = 0, <1/s_z>_W exactly."""
    P = _points(points)
    S, W = _directions(S, W)
    light = _light(light_direction)
    mu = float(mu)
    if not (np.isfinite(mu) and mu >= 0):
        raise ValueError("mu must be finite and >= 0")
    w_path = W / S[:, 2]                                # weight x path per unit depth
    if mu == 0.0:
        return np.full(P.shape[0], float(w_path.sum()))
    out = np.empty(P.shape[0])
    step = max(1, _CHUNK_ENTRIES // S.shape[0])
    for a in range(0, P.shape[0], step):
        ell = _back_chunk(P[a:a + step], S, tube, light)
        ell *= -mu
        out[a:a + step] = np.exp(ell, out=ell) @ w_path
    return out


def slab_absorbed(X, Y, z_lo, z_hi, zeta, dzeta, mu, tube, S, W, light_direction=+1):
    """Delta A_j of the hybrid for one slab [zeta - dzeta/2, zeta + dzeta/2]
    on the columns (X, Y) (um; broadcast to the shape of z_lo) whose vertical
    lines are inside the tube on [z_lo, z_hi] (Block 2 column_interval; a
    column that misses has z_lo = z_hi): a_j h(X, Y, zbar_j) with a_j Block 2's
    slab_absorbance and zbar_j the midpoint of the slab's inside part; 0 where
    a_j = 0. Same shape as z_lo."""
    z_lo = np.asarray(z_lo, dtype=float)
    z_hi = np.asarray(z_hi, dtype=float)
    if z_lo.shape != z_hi.shape:
        raise ValueError("z_lo and z_hi must have the same shape")
    X, Y = np.broadcast_to(X, z_lo.shape), np.broadcast_to(Y, z_lo.shape)
    a = geometry.slab_absorbance(z_lo, z_hi, float(zeta), float(dzeta), float(mu))
    out = np.zeros(z_lo.shape)
    m = a > 0
    if m.any():
        lo = np.maximum(z_lo[m], zeta - 0.5 * dzeta)
        hi = np.minimum(z_hi[m], zeta + 0.5 * dzeta)
        pts = np.column_stack([X[m], Y[m], 0.5 * (lo + hi)])
        out[m] = a[m] * ray_history(pts, tube, mu, S, W, light_direction)
    return out
