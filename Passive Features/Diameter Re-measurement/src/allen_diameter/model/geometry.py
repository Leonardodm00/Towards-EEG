"""Geometry of a straight phantom tube -- Block 2 in specs/SPEC.md.

Closed forms (S1)-(S6) of docs/TEEG_diameter_implementation_handoff_2026-10-06.md,
generalised in two ways that the configuration needs:

* a cross-section squashed along the GLOBAL z axis by a factor k
  (RendererConfig.cross_section_aspect; procedure s.3.5: mounting shrinkage
  acts along z). The phantom is a round tube of radius r in unsquashed space,
  mapped by S = diag(1, 1, k) about its node point. Its observed tilt is phi,
  and its tilt in unsquashed space is phi0 = arctan(tan(phi) / k). At k = 1
  every formula reduces to the handoff's.
* two end cuts (RendererConfig.end_cut): "vertical" keeps |u| <= U (the
  handoff's (S5) convention); "axial" keeps |s0| <= U, where s0 is the
  arc length along the axis in unsquashed space (caps perpendicular to the
  axis), so the depth span stays bounded as phi -> 90 deg.

Frame: x, y specimen-referred um; z in stage um; c = node point on the axis.
Heading theta in (-pi, pi]; tilt phi in [0, pi/2). For every point,
    u = (x - c_x) cos(theta) + (y - c_y) sin(theta)        (S1)
    v = -(x - c_x) sin(theta) + (y - c_y) cos(theta)
    w = z - c_z
and in unsquashed space w0 = w / k. Membership (handoff Eq. 6 / (S2)):
    v^2 + (w0 cos(phi0) - u sin(phi0))^2 <= r^2, plus the end cut.
The vertical line through (x, y) is inside the tube iff |v| <= r and
    z in [z_axis(u) - h(v), z_axis(u) + h(v)],  z_axis(u) = c_z + u tan(phi),
    h(v) = sqrt(r^2 - v^2) * sqrt(k^2 + tan(phi)^2)                  (S3)
(= sqrt(r^2 - v^2) / cos(phi) at k = 1), intersected with the end cut.

Library calls: numpy only. Custom code: all of it -- these are the closed
forms of the spec; the smoke test checks them against an independent
membership test, brute-force sampling and the theory chat's
checks/stack_geometry_check.py.

Pure ASCII (hpc-python-compat).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

END_CUTS = ("axial", "vertical")
_TINY = 1e-15


@dataclass(frozen=True)
class Tube:
    """One straight phantom tube.

    c          node point on the axis, (3,) um (x, y, z; z in stage units)
    r          radius, um (> 0); the lateral width along e_v is d = 2 r for
               every phi and every aspect (handoff Eq. 7)
    phi        observed tilt out of the image plane, rad, in [0, pi/2)
    theta      heading of the axis' projection, rad
    aspect     z-squash factor k > 0 (1 = round)
    half_length  U, um, or None for an infinite tube
    end_cut    "axial" or "vertical" (ignored when half_length is None)
    """

    c: Tuple[float, float, float]
    r: float
    phi: float
    theta: float
    aspect: float = 1.0
    half_length: Optional[float] = None
    end_cut: str = "axial"

    def __post_init__(self):
        if len(self.c) != 3:
            raise ValueError("c must have 3 components")
        if not (self.r > 0):
            raise ValueError("r must be > 0, got %r" % (self.r,))
        if not (0.0 <= self.phi < 0.5 * math.pi):
            raise ValueError("phi must be in [0, pi/2), got %r" % (self.phi,))
        if not (self.aspect > 0):
            raise ValueError("aspect must be > 0, got %r" % (self.aspect,))
        if self.half_length is not None and not (self.half_length > 0):
            raise ValueError("half_length must be > 0 or None")
        if self.end_cut not in END_CUTS:
            raise ValueError("end_cut must be one of %r" % (END_CUTS,))

    @property
    def d(self) -> float:
        return 2.0 * self.r

    @property
    def phi0(self) -> float:
        """Tilt in unsquashed space, arctan(tan(phi) / k)."""
        return math.atan(math.tan(self.phi) / self.aspect)

    @property
    def e_u(self) -> np.ndarray:
        return np.array([math.cos(self.theta), math.sin(self.theta), 0.0])

    @property
    def e_v(self) -> np.ndarray:
        return np.array([-math.sin(self.theta), math.cos(self.theta), 0.0])

    @property
    def t_hat(self) -> np.ndarray:
        """Observed (squashed-space) axis direction, unit vector."""
        return axis_direction(self.phi, self.theta)

    @property
    def t_hat0(self) -> np.ndarray:
        """Axis direction in unsquashed space, unit vector."""
        return axis_direction(self.phi0, self.theta)

    @property
    def half_chord_factor(self) -> float:
        """sqrt(k^2 + tan(phi)^2): h(v) / sqrt(r^2 - v^2) of (S3)."""
        return math.sqrt(self.aspect ** 2 + math.tan(self.phi) ** 2)


def axis_direction(phi: float, theta: float) -> np.ndarray:
    """t_hat = (cos phi cos theta, cos phi sin theta, sin phi) (handoff Eq. 5)."""
    return np.array([math.cos(phi) * math.cos(theta), math.cos(phi) * math.sin(theta), math.sin(phi)])


def to_local(x, y, z, tube: Tube):
    """(S1): local coordinates (u, v, w) of points, broadcasting x, y, z."""
    dx = np.asarray(x, dtype=float) - tube.c[0]
    dy = np.asarray(y, dtype=float) - tube.c[1]
    ct, st = math.cos(tube.theta), math.sin(tube.theta)
    u = dx * ct + dy * st
    v = -dx * st + dy * ct
    w = np.asarray(z, dtype=float) - tube.c[2]
    return u, v, w


def inside_tube(x, y, z, tube: Tube) -> np.ndarray:
    """3-D membership by handoff Eq. 6 in unsquashed space, plus the end cut:
    |q0|^2 - (q0 . t0)^2 <= r^2 with q0 = S^-1 (p - c)."""
    q = np.stack(np.broadcast_arrays(np.asarray(x, float) - tube.c[0],
                                     np.asarray(y, float) - tube.c[1],
                                     (np.asarray(z, float) - tube.c[2]) / tube.aspect), axis=-1)
    t0 = tube.t_hat0
    qt = q @ t0
    inside = (q * q).sum(-1) - qt ** 2 <= tube.r ** 2
    if tube.half_length is not None:
        if tube.end_cut == "axial":
            inside &= np.abs(qt) <= tube.half_length
        else:
            inside &= np.abs(q @ tube.e_u) <= tube.half_length
    return inside


def column_interval(x, y, tube: Tube):
    """(S3) generalised: for each vertical line through (x, y), the depth
    interval [z_lo, z_hi] (um, stage units) where the line is inside the tube.

    Returns (z_lo, z_hi, inside); where the line misses the tube,
    inside is False and z_lo = z_hi = c_z (zero length, never NaN).
    """
    u, v, _ = to_local(x, y, 0.0, tube)
    u, v = np.broadcast_arrays(u, v)
    r = tube.r
    inside = np.abs(v) <= r
    half = np.sqrt(np.clip(r * r - v * v, 0.0, None)) * tube.half_chord_factor
    z_axis = tube.c[2] + u * math.tan(tube.phi)
    lo = z_axis - half
    hi = z_axis + half
    U = tube.half_length
    if U is not None:
        if tube.end_cut == "vertical":
            inside = inside & (np.abs(u) <= U)
        else:
            s0, c0, k = math.sin(tube.phi0), math.cos(tube.phi0), tube.aspect
            if s0 > _TINY:
                cap_lo = tube.c[2] + k * (-U - u * c0) / s0
                cap_hi = tube.c[2] + k * (U - u * c0) / s0
                lo = np.maximum(lo, cap_lo)
                hi = np.minimum(hi, cap_hi)
                inside = inside & (hi > lo)
            else:  # phi = 0: the caps are vertical planes at |u| = U
                inside = inside & (np.abs(u) * c0 <= U)
    cz = tube.c[2]
    z_lo = np.where(inside, lo, cz)
    z_hi = np.where(inside, hi, cz)
    return z_lo, z_hi, inside


def depth_extent(tube: Tube) -> Tuple[float, float]:
    """(S5) generalised: (z_min, z_max) over the whole tube, um.

    vertical cut: c_z -+ (U tan(phi) + r sqrt(k^2 + tan^2 phi));
    axial cut:    c_z -+ k (U sin(phi0) + r cos(phi0));
    no cut:       infinite unless phi = 0.
    For a plane at z_k the farthest tube point is at
    max(|z_k - z_min|, |z_max - z_k|) (impl-handoff (S5))."""
    U, r, cz = tube.half_length, tube.r, tube.c[2]
    if U is None:
        if tube.phi == 0.0:
            h = r * tube.aspect
            return cz - h, cz + h
        return -math.inf, math.inf
    if tube.end_cut == "vertical":
        h = U * math.tan(tube.phi) + r * tube.half_chord_factor
    else:
        h = tube.aspect * (U * math.sin(tube.phi0) + r * math.cos(tube.phi0))
    return cz - h, cz + h


def lateral_half_extent(tube: Tube) -> Tuple[float, float]:
    """Half-extents (along e_u, along e_v) of the tube's footprint, um."""
    U, r = tube.half_length, tube.r
    if U is None:
        return math.inf, r
    if tube.end_cut == "vertical":
        return U, r
    return U * math.cos(tube.phi0) + r * math.sin(tube.phi0), r


def slab_centres(z_min: float, z_max: float, dzeta: float) -> np.ndarray:
    """Centres zeta_j of slabs of thickness dzeta tiling [z_min, z_max]:
    zeta_j = z_min + (j + 1/2) dzeta, j = 0 .. J-1, J = ceil((z_max - z_min) / dzeta)."""
    if not (dzeta > 0) or not (z_max >= z_min) or not (math.isfinite(z_min) and math.isfinite(z_max)):
        raise ValueError("need finite z_min <= z_max and dzeta > 0")
    J = max(1, int(math.ceil((z_max - z_min) / dzeta - 1e-12)))
    return z_min + (np.arange(J) + 0.5) * dzeta


def slab_absorbance(z_lo, z_hi, zeta: float, dzeta: float, mu: float) -> np.ndarray:
    """(S4): a_j = mu * |[z_lo, z_hi] intersect [zeta - dzeta/2, zeta + dzeta/2]|."""
    lo = np.maximum(z_lo, zeta - 0.5 * dzeta)
    hi = np.minimum(z_hi, zeta + 0.5 * dzeta)
    return mu * np.clip(hi - lo, 0.0, None)


def transmitted_before(z_lo, z_hi, zeta: float, dzeta: float, mu: float, light_direction: int = +1) -> np.ndarray:
    """(S4): T_<j, the fraction of the incident light that reaches slab j.

    light_direction = +1: light travels toward +z, so the tube below the slab,
    [z_lo, min(z_hi, zeta - dzeta/2)], has absorbed already; -1: toward -z,
    the tube above the slab, [max(z_lo, zeta + dzeta/2), z_hi]."""
    if light_direction == +1:
        length = np.minimum(z_hi, zeta - 0.5 * dzeta) - z_lo
    elif light_direction == -1:
        length = z_hi - np.maximum(z_lo, zeta + 0.5 * dzeta)
    else:
        raise ValueError("light_direction must be +1 or -1")
    return np.exp(-mu * np.clip(length, 0.0, None))


def line_interval(P, s, tube: Tube):
    """(S6) chord, generalised: the parameter interval [t1, t2] of the lines
    P + t s that lies inside the tube (end cut included).

    P, s: (..., 3) arrays (broadcast); s need not be unit -- the arc length
    inside the tube in real (squashed) space is (t2 - t1) |s|.
    Returns (t1, t2, hit); t1 = t2 = 0 where the line misses.
    """
    P = np.asarray(P, dtype=float)
    s = np.asarray(s, dtype=float)
    k = tube.aspect
    q = P - np.asarray(tube.c, dtype=float)
    sq = np.array([1.0, 1.0, 1.0 / k])
    q0, s0 = q * sq, s * sq
    q0, s0 = np.broadcast_arrays(q0, s0)
    t0 = tube.t_hat0
    st = s0 @ t0
    qt = q0 @ t0
    A = (s0 * s0).sum(-1) - st ** 2
    B = (q0 * s0).sum(-1) - qt * st
    C = (q0 * q0).sum(-1) - qt ** 2 - tube.r ** 2
    scale = (s0 * s0).sum(-1)
    par = A <= 1e-14 * np.maximum(scale, 1e-300)
    disc = B * B - A * C
    ok = ~par & (disc > 0)
    sqd = np.sqrt(np.where(ok, disc, 0.0))
    Asafe = np.where(par, 1.0, A)
    inside_par = par & (C < 0)   # strictly inside, like disc > 0 for crossing lines
    t1 = np.where(ok, (-B - sqd) / Asafe, np.where(inside_par, -np.inf, 0.0))
    t2 = np.where(ok, (-B + sqd) / Asafe, np.where(inside_par, np.inf, 0.0))
    hit = ok | inside_par
    U = tube.half_length
    if U is not None:
        if tube.end_cut == "axial":
            a, b = qt, st          # s0(t) = qt + t st must lie in [-U, U]
        else:
            eu = tube.e_u
            a, b = q @ eu, np.broadcast_to(s @ eu, np.shape(qt))
        with np.errstate(divide="ignore", invalid="ignore"):
            ta = (-U - a) / b
            tb = (U - a) / b
        moving = np.abs(b) > _TINY
        c_lo = np.where(moving, np.minimum(ta, tb), np.where(np.abs(a) <= U, -np.inf, np.inf))
        c_hi = np.where(moving, np.maximum(ta, tb), np.where(np.abs(a) <= U, np.inf, -np.inf))
        t1 = np.maximum(t1, c_lo)
        t2 = np.minimum(t2, c_hi)
        hit = hit & (t2 > t1)
    t1 = np.where(hit, t1, 0.0)
    t2 = np.where(hit, t2, 0.0)
    return t1, t2, hit
