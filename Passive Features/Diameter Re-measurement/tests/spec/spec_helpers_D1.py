"""Independent references for run D1's spec-derived tests (Blocks 0, 1, 4-9, 21
and their boundaries; integration tester, 2026-11-13). Written from
specs/SPEC.md (and, for Block 0, the module docstrings the spec names as its
contract) only: nothing here calls the code under test.

  ref_choose_swc_file  Block 0 / D-098.1: drop a basename containing "marker"
                       (any case); the first file of type 303941301 (compared
                       as an integer), else the first path ending in .swc
  ref_swc_to_full_um   Block 0 equations (1)-(2) of allen_image_align, times p_x
  axis_dir             (cos phi cos theta, cos phi sin theta, sin phi)
  ref_line_tube        (S2)/(S6) for a round tube (k = 1) with the axial cut:
                       the parameter interval of P + t s inside the tube, by
                       the quadratic of the distance to the axis and the slab
                       |(q . a_hat)| <= U
  ref_column           (S3) the vertical line's [z_lo, z_hi] (ref_line_tube)
  ref_back_length      section 2.3: |[t1, t2] cap (-inf, 0]| (light +1) or
                       |[t1, t2] cap [0, inf)| (light -1), unit directions
  ref_h                section 2.3 (H): <exp(-mu l_back) / s_z>_W
  ref_directions       Block 9: Gauss-Legendre in rho^2 on [0, s_m^2] x
                       azimuths (m + 1/2) 2 pi / n_psi, weights summing to 1
  ref_mean_inverse_cos (2 / s_m^2)(1 - sqrt(1 - s_m^2))  (S10)
  ref_shadow_per_length section 2.3 / Block 21 (c): (p/s_z) int_{-r}^{r}
                       (1 - exp(-2 mu sqrt(r^2 - b^2)/p)) db, one direction
  ref_frustum          Block 8: pi (r1 + r2) sqrt(h^2 + (r1 - r2)^2)
  ref_stretches        Block 8's stretch rule, walked by hand
  write_swc_text       a hand-made SWC file

Pure ASCII.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from scipy import integrate

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
for _p in (WS / "src", WS / "scripts", WS / "tests" / "smoke"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

SWC_TYPE_ID = 303941301


# ------------------------------------------------------------- Block 0 ----

def _basename(path):
    return str(path).replace("\\", "/").split("/")[-1]


def ref_choose_swc_file(files):
    """D-098.1 as SPEC Block 0 states it."""
    keep = [f for f in files if "marker" not in _basename(f.get("path", "")).lower()]
    for f in keep:
        t = f.get("well_known_file_type_id")
        try:
            if t is not None and int(t) == SWC_TYPE_ID:
                return f
        except (TypeError, ValueError):
            pass
    for f in keep:
        if str(f.get("path", "")).endswith(".swc"):
            return f
    return None


def ref_swc_to_full_um(xyz, res0, shift=(0.0, 0.0), flip_h=None, z0=0.0):
    """allen_image_align's (1)-(2) in full-resolution pixels, times res0 (um);
    z - z0 for the depth (SPEC Block 8, Frame)."""
    xyz = np.asarray(xyz, dtype=float)
    x_px = xyz[:, 0] / res0 + shift[0]
    if flip_h is None:
        y_px = xyz[:, 1] / res0 + shift[1]
    else:
        y_px = (flip_h - xyz[:, 1] / res0) + shift[1]
    return np.stack([x_px * res0, y_px * res0, xyz[:, 2] - z0], axis=1)


# --------------------------------------------------------- tube geometry ----

def axis_dir(phi, theta):
    return np.array([math.cos(phi) * math.cos(theta),
                     math.cos(phi) * math.sin(theta),
                     math.sin(phi)])


def ref_line_tube(P, s, c, r, phi, theta, U=None):
    """(t1, t2, hit) of the line P + t s inside the round tube of radius r
    about the axis through c along axis_dir(phi, theta), cut at axial arc
    length |(x - c) . a_hat| <= U (None: uncapped). P (n, 3), s (3,) or
    (n, 3). Lines that miss: t1 = t2 = 0, hit False."""
    P = np.atleast_2d(np.asarray(P, dtype=float))
    s = np.broadcast_to(np.asarray(s, dtype=float), P.shape)
    a = axis_dir(phi, theta)
    q = P - np.asarray(c, dtype=float)
    sa = s @ a
    qa = q @ a
    s_perp = s - sa[:, None] * a
    q_perp = q - qa[:, None] * a
    A = np.einsum("ij,ij->i", s_perp, s_perp)
    B = 2.0 * np.einsum("ij,ij->i", q_perp, s_perp)
    C = np.einsum("ij,ij->i", q_perp, q_perp) - r * r
    disc = B * B - 4.0 * A * C
    with np.errstate(divide="ignore", invalid="ignore"):
        sq = np.sqrt(np.maximum(disc, 0.0))
        t1 = (-B - sq) / (2.0 * A)
        t2 = (-B + sq) / (2.0 * A)
    par = A <= 1e-300
    # a line parallel to the axis: inside for all t when C < 0
    t1 = np.where(par, np.where(C < 0, -np.inf, 0.0), t1)
    t2 = np.where(par, np.where(C < 0, np.inf, 0.0), t2)
    hit = np.where(par, C < 0, disc > 0)
    if U is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            lo = np.where(sa != 0, np.minimum((-U - qa) / sa, (U - qa) / sa), -np.inf)
            hi = np.where(sa != 0, np.maximum((-U - qa) / sa, (U - qa) / sa), np.inf)
        cap_ok = np.where(sa != 0, True, np.abs(qa) <= U)
        t1n = np.maximum(t1, lo)
        t2n = np.minimum(t2, hi)
        hit = hit & cap_ok & (t2n > t1n)
        t1, t2 = t1n, t2n
    t1 = np.where(hit, t1, 0.0)
    t2 = np.where(hit, t2, 0.0)
    return t1, t2, hit


def ref_column(x, y, c, r, phi, theta, U=None):
    """(z_lo, z_hi, inside) of the vertical lines through (x, y) (S3)."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    P = np.stack([x, y, np.zeros_like(x)], axis=1)
    t1, t2, hit = ref_line_tube(P, np.array([0.0, 0.0, 1.0]), c, r, phi, theta, U)
    z_lo = np.where(hit, t1, c[2])
    z_hi = np.where(hit, t2, c[2])
    return z_lo, z_hi, hit


def ref_back_length(P, S, c, r, phi, theta, U=None, light=+1):
    """(n, N) l_back for unit directions S (N, 3) (section 2.3)."""
    P = np.atleast_2d(np.asarray(P, dtype=float))
    S = np.atleast_2d(np.asarray(S, dtype=float))
    out = np.zeros((P.shape[0], S.shape[0]))
    for m in range(S.shape[0]):
        t1, t2, hit = ref_line_tube(P, S[m], c, r, phi, theta, U)
        if light > 0:
            lb = np.maximum(0.0, np.minimum(t2, 0.0) - t1)
        else:
            lb = np.maximum(0.0, t2 - np.maximum(t1, 0.0))
        out[:, m] = np.where(hit, lb, 0.0)
    return out


def ref_h(P, S, W, mu, c, r, phi, theta, U=None, light=+1):
    lb = ref_back_length(P, S, c, r, phi, theta, U, light)
    with np.errstate(invalid="ignore"):
        e = np.exp(-mu * lb) if mu > 0 else np.ones_like(lb)
    return e @ (np.asarray(W, dtype=float) / np.asarray(S, dtype=float)[:, 2])


# ------------------------------------------------------------- Block 9 ----

def ref_directions(n_rho, n_psi, na, n_oil):
    sm = na / n_oil
    x, w = np.polynomial.legendre.leggauss(n_rho)
    rho2 = sm * sm * (x + 1.0) / 2.0
    wr = w / 2.0
    psi = (np.arange(n_psi) + 0.5) * 2.0 * math.pi / n_psi
    rho = np.sqrt(rho2)
    sx = (rho[:, None] * np.cos(psi)[None, :]).ravel()
    sy = (rho[:, None] * np.sin(psi)[None, :]).ravel()
    sz = np.repeat(np.sqrt(1.0 - rho2), n_psi)
    W = np.repeat(wr, n_psi) / n_psi
    return np.stack([sx, sy, sz], axis=1), W


def ref_mean_inverse_cos(na, n_oil):
    sm2 = (na / n_oil) ** 2
    return (2.0 / sm2) * (1.0 - math.sqrt(1.0 - sm2))


def sort_rows(A):
    A = np.asarray(A, dtype=float)
    order = np.lexsort(np.round(A[:, ::-1], 12).T)
    return A[order]


# ----------------------------------------------------- Block 21 invariant ----

def ref_shadow_per_length(mu, r, s):
    """Per unit length of an infinite flat tube along x: the total shadow of
    one direction s = (s_x, s_v, s_z), (p/s_z) int_{-r}^{r} (1 - exp(-2 mu
    sqrt(r^2 - b^2)/p)) db with p = sqrt(s_v^2 + s_z^2) (SPEC Block 21 (c))."""
    p = math.hypot(s[1], s[2])

    def f(b):
        return 1.0 - math.exp(-2.0 * mu * math.sqrt(max(r * r - b * b, 0.0)) / p)

    val, _err = integrate.quad(f, -r, r, epsabs=1e-14, epsrel=1e-13, limit=200)
    return (p / s[2]) * val


# ------------------------------------------------------------- Block 8 ----

def ref_frustum(r1, r2, h):
    return math.pi * (r1 + r2) * math.sqrt(h * h + (r1 - r2) ** 2)


def ref_stretches(ids, types, parents, dend=(3, 4)):
    """Maximal unbranched runs of dendrite nodes, proximal to distal, as
    lists of ids (SPEC Block 8, Stretches)."""
    ids = list(ids)
    types = dict(zip(ids, types))
    parent = dict(zip(ids, parents))
    children = {i: [] for i in ids}
    for i in ids:
        p = parent[i]
        if p in children:
            children[p].append(i)

    def dch(i):
        return [k for k in children[i] if types[k] in dend]

    def starts(i):
        p = parent[i]
        if types[i] not in dend:
            return False
        if p not in types or types[p] not in dend:
            return True
        return len(dch(p)) >= 2

    out = []
    for i in ids:
        if not starts(i):
            continue
        run = [i]
        cur = i
        while True:
            dc = dch(cur)
            if len(dc) != 1:
                break
            cur = dc[0]
            run.append(cur)
        out.append(run)
    return out


def write_swc_text(path, rows, comments=("# hand-made SWC for the D1 spec tests",)):
    """rows: (id, type, x, y, z, radius_token, parent); the radius as a token
    string so that its decimals are chosen by the test."""
    lines = list(comments)
    for (i, t, x, y, z, rad, p) in rows:
        lines.append("%d %d %s %s %s %s %d" % (i, t, repr(float(x)), repr(float(y)),
                                               repr(float(z)), rad, p))
    with open(path, "w", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    return path
