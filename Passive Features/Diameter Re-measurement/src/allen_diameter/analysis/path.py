"""Local branch direction -- Block 5 in specs/SPEC.md (handoff Eqs. 4-5; D6).

Total-least-squares line through 3-D points (um, global frame -- never in
pixel or plane units, D6: the 0.28 / 0.1144 anisotropy would distort the
tilt): t_hat = leading eigenvector of the points' covariance. Then

    theta = atan2(t_y, t_x),  phi = arcsin |t_z|,
    y_hat = (-sin theta, cos theta)  (measuring axis),  e_u = (cos theta, sin theta).

Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import math

import numpy as np


def window(s, i, half):
    """Indices j with |s_j - s_i| <= half (s: path length, um)."""
    s = np.asarray(s, dtype=float)
    return np.flatnonzero(np.abs(s - s[i]) <= half + 1e-12 * max(1.0, abs(half)))


def tls_direction(points):
    """Unit leading eigenvector (3,) of the covariance of points (m, 3), m >= 2,
    by numpy.linalg.eigh; ValueError when the points coincide."""
    P = np.asarray(points, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3 or P.shape[0] < 2 or not np.all(np.isfinite(P)):
        raise ValueError("tls_direction needs at least 2 finite 3-D points")
    Q = P - P.mean(axis=0)
    w, V = np.linalg.eigh(Q.T @ Q / P.shape[0])
    if not (w[-1] > 0):
        raise ValueError("tls_direction: the points coincide")
    t = V[:, -1]
    return t / np.linalg.norm(t)


def orient(t, first, last):
    """t or -t, whichever has a non-negative component along last - first."""
    t = np.asarray(t, dtype=float)
    return -t if float(np.dot(t, np.asarray(last, float) - np.asarray(first, float))) < 0 else t


def angles(t):
    """(theta rad, phi rad, y_hat (2,), e_u (2,)) of a 3-D unit direction (handoff Eq. 5)."""
    t = np.asarray(t, dtype=float)
    theta = math.atan2(t[1], t[0])
    phi = math.asin(min(1.0, abs(float(t[2]))))
    return theta, phi, np.array([-math.sin(theta), math.cos(theta)]), np.array([math.cos(theta), math.sin(theta)])


def circular_mean(thetas):
    """Mean direction of headings (rad); None for an empty input."""
    th = np.asarray(thetas, dtype=float)
    if th.size == 0:
        return None
    return math.atan2(float(np.mean(np.sin(th))), float(np.mean(np.cos(th))))
