"""Independent references for the spec-derived tests of Blocks 8 (amendment),
44 (amendment 3), 46 (amendment 1), 47, 48 and the soma fit's entry point
(tester, 2026-11-23). Written from specs/SPEC.md only: nothing here calls the
code under test.

  ref_to_image        Block 47's frame map: x_img = (x/p + s_x) p,
                      y_img = ((F - y/p if flipped else y/p) + s_y) p,
                      z_img = z - z0
  ref_frame, ref_q    Block 47's (u, v, w) and Q = (u/a)^2 + (v/b)^2 + (w/c)^2 - 1
  ref_roots           the roots of Q(P + t d) = 0 by numpy.roots, one line at a time
  ref_area            the ellipsoid's area: closed forms for the sphere and the
                      spheroids, scipy.integrate.dblquad of the parametric surface
  ref_window          the plane window |k dz - z| <= half with exact rationals
                      (fractions) and a tolerance of 1e-9 of a plane
  ref_start_theta     Block 46's start (D-219.5, amendment 1) written out with numpy
  ref_edit_text       Block 8 amendment / Block 47: the SWC text with tokens 3-5, 6
                      and 7 of the kept data lines replaced, the removed rows'
                      lines dropped, every other byte verbatim
  make_swc_text       an SWC text from rows (id, type, x, y, z, r, parent)

Pure ASCII.
"""
from __future__ import annotations

import math
import re
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy import integrate

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
for _p in (WS / "src", WS / "scripts", WS / "tests" / "smoke", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


# ----------------------------------------------------------- Block 47 frames --

def ref_to_image(xyz, p, shift=(0.0, 0.0), flip=None, z0=0.0):
    P = np.asarray(xyz, dtype=float)
    x = (P[:, 0] / p + shift[0]) * p
    yy = P[:, 1] / p
    if flip is not None:
        yy = flip - yy
    y = (yy + shift[1]) * p
    return np.column_stack([x, y, P[:, 2] - z0])


def ref_frame(P, theta):
    x0, y0, z0, a, b, c, phi, mu = (float(t) for t in np.asarray(theta, dtype=float)[:8])
    P = np.asarray(P, dtype=float)
    dx, dy, dz = P[:, 0] - x0, P[:, 1] - y0, P[:, 2] - z0
    u = dx * math.cos(phi) + dy * math.sin(phi)
    v = -dx * math.sin(phi) + dy * math.cos(phi)
    return np.column_stack([u, v, dz])


def ref_q(P, theta):
    a, b, c = float(theta[3]), float(theta[4]), float(theta[5])
    U = ref_frame(P, theta)
    return (U[:, 0] / a) ** 2 + (U[:, 1] / b) ** 2 + (U[:, 2] / c) ** 2 - 1.0


def ref_roots(P, d, theta):
    """(t_lo, t_hi) by numpy.roots of A t^2 + 2 B t + C with the coefficients of Block 47 (NaN for complex roots)."""
    a, b, c, phi = (float(theta[j]) for j in (3, 4, 5, 6))
    P = np.atleast_2d(np.asarray(P, dtype=float))
    D = np.broadcast_to(np.asarray(d, dtype=float), P.shape)
    U = ref_frame(P, theta)
    cp, sp = math.cos(phi), math.sin(phi)
    Du = np.column_stack([D[:, 0] * cp + D[:, 1] * sp, -D[:, 0] * sp + D[:, 1] * cp, D[:, 2]])
    ax = np.array([a, b, c])
    A = np.sum((Du / ax) ** 2, axis=1)
    B = np.sum(U * Du / ax ** 2, axis=1)
    C = np.sum((U / ax) ** 2, axis=1) - 1.0
    lo, hi = np.full(len(P), np.nan), np.full(len(P), np.nan)
    for i in range(len(P)):
        r = np.roots([A[i], 2 * B[i], C[i]])
        if np.all(np.abs(np.imag(r)) == 0):
            r = np.sort(np.real(r))
            lo[i], hi[i] = r[0], r[-1]
    return lo, hi


# -------------------------------------------------------------- Block 46 area --

def ref_area(a, b, c):
    """The ellipsoid's area: closed forms (sphere, spheroids) or dblquad of |r_theta x r_phi|."""
    a, b, c = sorted((float(a), float(b), float(c)), reverse=True)
    if a == b == c:
        return 4 * math.pi * a * a
    if b == c:                      # prolate about the a axis
        e = math.sqrt(1 - (c / a) ** 2)
        return 2 * math.pi * c * c * (1 + a / (c * e) * math.asin(e))
    if a == b:                      # oblate about the c axis
        e = math.sqrt(1 - (c / a) ** 2)
        return 2 * math.pi * a * a * (1 + (1 - e * e) / e * math.atanh(e))

    def elem(th, ph):               # th in [0, pi], ph in [0, 2 pi]
        st, ct, sp, cp = math.sin(th), math.cos(th), math.sin(ph), math.cos(ph)
        rt = np.array([a * ct * cp, b * ct * sp, -c * st])
        rp = np.array([-a * st * sp, b * st * cp, 0.0])
        return float(np.linalg.norm(np.cross(rt, rp)))
    val, _err = integrate.dblquad(lambda ph, th: elem(th, ph), 0.0, math.pi, 0.0, 2 * math.pi,
                                  epsabs=0.0, epsrel=1e-12)
    return val


# -------------------------------------------------------------- plane windows --

def ref_window(z, half, dz, ks):
    """The ks with |k dz - z| <= half, decided on the exact rationals of the floats with a tolerance of 1e-9 of a
    plane (SPEC Block 46 amendment 1 (iii): Block 44's closed window)."""
    Z, H, DZ = Fraction(z), Fraction(half), Fraction(dz)
    tol = Fraction(1, 10 ** 9) * DZ
    return [int(k) for k in ks if abs(Fraction(int(k)) * DZ - Z) <= H + tol]


# ------------------------------------------------------------- Block 46 start --

def ref_start_theta(plane, left, top, p, centre, rs, region, slope0, far_r=1.5, core_r=0.5, pct=5.0,
                    clip=(0.05, 0.95), none=0.5):
    """SPEC Block 46 'Start' with amendment 1's settings, written out with numpy."""
    H, W = plane.shape
    xs, ys, zs = centre
    x = (left + np.arange(W)) * p
    y = (top + np.arange(H)) * p
    X, Y = np.meshgrid(x, y)
    Xr, Yr, Dr = X[region], Y[region], plane[region].astype(float)
    rho = np.sqrt((Xr - xs) ** 2 + (Yr - ys) ** 2)
    far = rho > far_r * rs
    B_far = float(np.median(Dr[far])) if far.any() else float(np.median(Dr))
    core = rho <= core_r * rs
    if core.any():
        varrho = float(np.clip(np.percentile(Dr[core], pct) / B_far, clip[0], clip[1]))
    else:
        varrho = none
    mu0 = -math.log(varrho) / (2 * rs)
    near = rho <= far_r * rs
    w = np.maximum(0.0, 1.0 - Dr[near] / B_far)
    a0 = b0 = rs
    phi0 = 0.0
    if w.sum() > 0:
        xm = np.sum(w * Xr[near]) / w.sum()
        ym = np.sum(w * Yr[near]) / w.sum()
        cxx = np.sum(w * (Xr[near] - xm) ** 2) / w.sum()
        cyy = np.sum(w * (Yr[near] - ym) ** 2) / w.sum()
        cxy = np.sum(w * (Xr[near] - xm) * (Yr[near] - ym)) / w.sum()
        tr, det = cxx + cyy, cxx * cyy - cxy * cxy
        disc = math.sqrt(max(0.0, tr * tr / 4 - det))
        l1, l2 = tr / 2 + disc, tr / 2 - disc
        if l2 > 0:
            phi0 = 0.5 * math.atan2(2 * cxy, cxx - cyy)
            a0 = rs * (l1 / l2) ** 0.25
            b0 = rs * (l2 / l1) ** 0.25
    return np.array([xs, ys, zs, a0, b0, rs, phi0, mu0, slope0], dtype=float)


# ----------------------------------------------------------- Block 8 / 47 SWC --

_TOK = re.compile(r"\S+")


def make_swc_text(rows, header="# tester's cell\n", sep=" ", fmt="%.4f"):
    out = [header]
    for (i, t, x, y, z, r, pa) in rows:
        out.append(sep.join(["%d" % i, "%d" % t, fmt % x, fmt % y, fmt % z, fmt % r, "%d" % pa]) + "\n")
    return "".join(out)


def _decimals(tok, floor=4):
    if "." in tok and "e" not in tok.lower():
        return max(len(tok.split(".")[1]), floor)
    return floor


def ref_edit_text(text, xyz=None, radius=None, parent=None, keep=None):
    """The text with, on each kept data line (node n in file order), tokens 3-5 replaced by xyz[n] (each with the
    decimals of the token it replaces, at least 4), token 6 by radius[n], token 7 by '%d' % parent[n]; the lines of
    the removed rows dropped; every other character verbatim."""
    lines = text.splitlines(keepends=True)
    out, n = [], 0
    for line in lines:
        s = line.strip()
        if not s or s.startswith("#"):
            out.append(line)
            continue
        i = n
        n += 1
        if keep is not None and not keep[i]:
            continue
        spans = [m.span() for m in _TOK.finditer(line)]
        repl = {}
        if xyz is not None:
            for j in range(3):
                a, b = spans[2 + j]
                repl[2 + j] = "%.*f" % (_decimals(line[a:b]), float(xyz[i][j]))
        if radius is not None:
            a, b = spans[5]
            repl[5] = "%.*f" % (_decimals(line[a:b]), float(radius[i]))
        if parent is not None:
            repl[6] = "%d" % int(parent[i])
        pieces, pos = [], 0
        for j, (a, b) in enumerate(spans):
            pieces.append(line[pos:a])
            pieces.append(repl.get(j, line[a:b]))
            pos = b
        pieces.append(line[pos:])
        out.append("".join(pieces))
    return "".join(out)
