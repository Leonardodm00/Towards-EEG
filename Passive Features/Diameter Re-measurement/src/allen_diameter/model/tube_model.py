"""Blurred-tube forward model -- Block 3 in specs/SPEC.md.

Handoff Eqs. 10-11 with the background fixed (D-018.2) and the darkness
parameterised by the absorption coefficient (D-019.1).

For a straight round tube of diameter d (um) crossed by the measuring axis,
the stained path along the vertical line at offset v (um) from the node is
(d / cos phi) s_d(v - v0), with s_d(u) = sqrt(1 - (2u/d)^2) for |u| <= d/2 and
0 otherwise (handoff Eq. 10). Beer-Lambert absorption followed by a Gaussian
blur g_sigma (standard deviation sigma, unit integral) gives, for each fixed
(d, alpha, v0, sigma, B_bar) and every v,

    I_model(v) = B_bar (T * g_sigma)(v),     T(v) = exp(-alpha s_d(v - v0)),
    alpha = mu d / cos(phi)                   (D-019.1, phi in [0, pi/2)).

Because 1 - T vanishes for |v - v0| > d/2 and g_sigma has unit integral, the
substitution u = (d/2) sin(psi) turns the convolution into

    I_model(v) = B_bar [1 - (d/2) Int_{-pi/2}^{pi/2} (1 - exp(-alpha cos psi))
                          cos(psi) g_sigma(v - v0 - (d/2) sin psi) dpsi]     (B3.1)

whose integrand is analytic in psi (the square root of s_d at the dome's
edges is gone). (B3.1) is evaluated with an N-node Gauss-Legendre rule in
psi; the caller chooses N (n_quadrature_nodes) and holds it fixed during a
fit, so the model is a smooth function of (d, alpha, v0).

Units: um for lengths, 1/um for mu, grey levels for B_bar and I_model;
alpha is dimensionless.

Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import functools
import math

import numpy as np
from scipy.special import roots_legendre

_SQRT_2PI = math.sqrt(2.0 * math.pi)
_ROWS_PER_CHUNK = 2048   # memory tile of model_profile: at most 2048 x N float64 temporaries


def _finite_scalar(x, name):
    x = float(x)
    if not math.isfinite(x):
        raise ValueError("%s must be finite, got %r" % (name, x))
    return x


def alpha_from_mu(mu, d, phi):
    """Centre-line absorbance alpha = mu d / cos(phi) (D-019.1).

    mu (1/um) >= 0, d (um) > 0, phi (rad) in [0, pi/2); broadcasting arrays
    or scalars. Raises ValueError for phi outside [0, pi/2), where the
    vertical path d / cos(phi) of handoff Eq. 10 is undefined.
    """
    phi_a = np.asarray(phi, dtype=float)
    if not np.all(np.isfinite(phi_a)) or np.any(phi_a < 0.0) or np.any(phi_a >= 0.5 * math.pi):
        raise ValueError("phi must lie in [0, pi/2) rad")
    return np.asarray(mu, dtype=float) * np.asarray(d, dtype=float) / np.cos(phi_a)


def n_quadrature_nodes(d_hi, sigma, min_nodes, nodes_per_sigma):
    """N = max(min_nodes, ceil(nodes_per_sigma * d_hi / sigma)).

    d_hi (um): the largest diameter the model will be evaluated at (in a fit,
    the upper bound of d); sigma (um): the blur. The Gaussian of (B3.1) is
    resolved by about nodes_per_sigma / 2.5 nodes per sigma at the centre of
    the dome, where the Gauss-Legendre nodes are sparsest.
    """
    d_hi = _finite_scalar(d_hi, "d_hi")
    sigma = _finite_scalar(sigma, "sigma")
    nodes_per_sigma = _finite_scalar(nodes_per_sigma, "nodes_per_sigma")
    if not (d_hi > 0 and sigma > 0 and nodes_per_sigma > 0) or int(min_nodes) < 1:
        raise ValueError("n_quadrature_nodes: need d_hi, sigma, nodes_per_sigma > 0 and min_nodes >= 1")
    return max(int(min_nodes), int(math.ceil(nodes_per_sigma * d_hi / sigma)))


@functools.lru_cache(maxsize=32)
def _rule(n_nodes):
    x, w = roots_legendre(n_nodes)
    psi = 0.5 * math.pi * x
    out = (np.sin(psi), np.cos(psi), 0.5 * math.pi * w)
    for a in out:
        a.setflags(write=False)
    return out


def quadrature_rule(n_nodes):
    """(sin psi_m, cos psi_m, W_m), m = 1..N: the Gauss-Legendre rule
    (scipy.special.roots_legendre) mapped from [-1, 1] to psi in
    [-pi/2, pi/2], weights scaled by pi/2. Cached per N; read-only arrays."""
    n = int(n_nodes)
    if n != n_nodes or n < 1:
        raise ValueError("n_nodes must be a positive integer, got %r" % (n_nodes,))
    return _rule(n)


def _check_shape_params(d, alpha):
    d = _finite_scalar(d, "d")
    alpha = _finite_scalar(alpha, "alpha")
    if not (d > 0):
        raise ValueError("d must be > 0, got %r" % d)
    if alpha < 0:
        raise ValueError("alpha must be >= 0, got %r" % alpha)
    return d, alpha


def transmittance(v, d, alpha, v0):
    """T(v) = exp(-alpha s_d(v - v0)), the unblurred transmittance of handoff
    Eq. 11 (dimensionless); equals 1 for |v - v0| >= d/2."""
    d, alpha = _check_shape_params(d, alpha)
    v0 = _finite_scalar(v0, "v0")
    u = (np.asarray(v, dtype=float) - v0) / (0.5 * d)
    return np.exp(-alpha * np.sqrt(np.clip(1.0 - u * u, 0.0, None)))


def model_profile(v, d, alpha, v0, sigma, B_bar, n_nodes):
    """(B3.1): B_bar (T * g_sigma)(v) by an n_nodes-point Gauss-Legendre rule.

    v: array (any shape) or scalar, um, finite. d > 0 (um), alpha >= 0,
    v0 (um), sigma > 0 (um), B_bar > 0 (grey levels). Returns float64 of the
    shape of v, in grey levels: in (0, B_bar] exactly, in [0, B_bar] up to
    quadrature roundoff (of order 1e-13 B_bar where the tube is opaque).
    """
    d, alpha = _check_shape_params(d, alpha)
    v0 = _finite_scalar(v0, "v0")
    sigma = _finite_scalar(sigma, "sigma")
    B_bar = _finite_scalar(B_bar, "B_bar")
    if not (sigma > 0 and B_bar > 0):
        raise ValueError("sigma and B_bar must be > 0")
    v_arr = np.asarray(v, dtype=float)
    if not np.all(np.isfinite(v_arr)):
        raise ValueError("v must be finite")
    sin_psi, cos_psi, w = quadrature_rule(n_nodes)
    # (d/2) W_m (1 - exp(-alpha cos psi_m)) cos psi_m; expm1 keeps the faint limit exact
    kern = (0.5 * d) * w * cos_psi * (-np.expm1(-alpha * cos_psi))
    shift = (0.5 * d) * sin_psi
    flat = v_arr.reshape(-1)
    dip = np.empty(flat.shape, dtype=float)
    for start in range(0, flat.size, _ROWS_PER_CHUNK):
        z = (flat[start:start + _ROWS_PER_CHUNK, None] - v0) - shift[None, :]
        g = np.exp(-0.5 * (z / sigma) ** 2) / (sigma * _SQRT_2PI)
        dip[start:start + _ROWS_PER_CHUNK] = g @ kern
    return (B_bar * (1.0 - dip)).reshape(v_arr.shape)
