"""The same-node clearance as a product -- SPEC Block 37 (D-151, D-156).

The node's features are estimated by the joint fit of all its planes with the whole own shadow (Blocks 29,
30); no fit reads a cleared profile (D-156.1, D-156.3). After the fit, the own shadow at the estimate, split
by depth (line_fast.line_shadow_split, Block 35 amendment 2), gives the cleared profiles, the in-band model,
the residual and the tube's chord at each plane.

Mathematics (SPEC Block 37). For plane k and sample n of the measuring line, with the profile I_k(v_n) (gl),
the background B_k (gl) and the split S_in_k(v_n), S_out_k(v_n) (dimensionless) of the own shadow:

    cleared   I_clr_k = I_k + B_k S_out_k                (the sign of Block 28: light added back)
    in-band   M_in_k  = B_k (1 - S_in_k)
    whole     M_k     = B_k (1 - S_in_k - S_out_k)
    residual  R_k     = I_k - M_k                        (= I_clr_k - M_in_k)

per plane: in_peak = B_k max_n S_in_k, out_peak = B_k max_n S_out_k (gl); in_share = S_in_k(v_n*) /
(S_in_k(v_n*) + S_out_k(v_n*)) at n* = argmax_n (S_in_k + S_out_k) (the first index on a tie, numpy's
argmax), NaN where that maximum is not positive; residual_rms = sqrt(mean_n R_k^2) (gl). The chord of a
round tube of diameter d and tilt phi at depth z_k - z_ax from its axis where it crosses the line:
c_k = 2 sqrt(max(d^2 / 4 - (z_k - z_ax)^2 cos^2 phi, 0)) (um) [D-156.5]. No clipping; float64 out.

No file access, no plotting, no configuration (the band and the layer thickness act in Block 35's split).
Keep this file pure ASCII so it survives transfer to the cluster.
"""
from __future__ import annotations

import dataclasses

import numpy as np

# A split entry may be a rounding below 0 (Block 35 sums slab terms); below this it is an error (SPEC Block 37).
SPLIT_NEG_TOL = 1e-9


@dataclasses.dataclass(frozen=True)
class OwnClearance:
    """cleared_gl, model_in_gl, model_gl, residual_gl  (n_planes, n_v) float64, gl:
           I + B S_out;  B (1 - S_in);  B (1 - S_in - S_out);  I - model_gl (= cleared_gl - model_in_gl)
    chord_um         (n_planes,) um: the tube's chord on the line at each plane's depth
    in_peak_gl       (n_planes,) gl: B_k max_n S_in
    out_peak_gl      (n_planes,) gl: B_k max_n S_out
    in_share         (n_planes,): S_in / (S_in + S_out) at the whole shadow's maximum; NaN where that
                     maximum is not positive
    residual_rms_gl  (n_planes,) gl: the root mean square of residual_gl over the line
    layers_gl        (n_layers, n_planes, n_v) gl, B_k times each layer's deficit, or None
    """

    cleared_gl: np.ndarray
    model_in_gl: np.ndarray
    model_gl: np.ndarray
    residual_gl: np.ndarray
    chord_um: np.ndarray
    in_peak_gl: np.ndarray
    out_peak_gl: np.ndarray
    in_share: np.ndarray
    residual_rms_gl: np.ndarray
    layers_gl: object


# ------------------------------------------------------------- validation ---

def _real_array(x, name):
    """float64 copy of a real, finite array; ValueError otherwise."""
    a = np.asarray(x)
    if a.dtype.kind not in "iuf":
        raise ValueError("%s must be real-valued; got dtype %s" % (name, a.dtype))
    a = a.astype(np.float64)                             # a copy: the input is never changed
    if not np.all(np.isfinite(a)):
        raise ValueError("%s must be finite" % name)
    return a


def _real_scalar(x, name):
    a = _real_array(x, name)
    if a.ndim != 0:
        raise ValueError("%s must be a scalar; got shape %r" % (name, a.shape))
    return float(a)


def _split_array(x, name, shape):
    a = _real_array(x, name)
    if a.shape != tuple(shape):
        raise ValueError("%s must have shape %r; got %r" % (name, tuple(shape), a.shape))
    if np.any(a < -SPLIT_NEG_TOL):
        raise ValueError("%s has an entry below -%g (a shadow is an absorbed fraction)" % (name, SPLIT_NEG_TOL))
    return a


# ------------------------------------------------------------------ chord ---

def chord_um(d_um, dz_um, phi_rad=0.0):
    """c = 2 sqrt(max(d^2 / 4 - (dz cos phi)^2, 0)) (um), the shape of dz_um: the part of the measuring line
    inside a round tube of diameter d_um (um, > 0) and tilt phi_rad at the depth dz_um = z_k - z_ax (um)
    from the axis' depth where it crosses the line (SPEC Block 37). A point at offset v on the line and depth
    z_k is at distance sqrt(v^2 + (dz cos phi)^2) from the axis. ValueError for d_um <= 0 or a non-finite
    input."""
    d = _real_scalar(d_um, "d_um")
    if d <= 0.0:
        raise ValueError("d_um must be > 0 (um); got %r" % d)
    phi = _real_scalar(phi_rad, "phi_rad")
    dz = _real_array(dz_um, "dz_um")
    a = np.abs(dz * np.cos(phi))                         # the depth offset's component across the axis, um
    half = 0.5 * d
    # (d/2 - a)(d/2 + a) = d^2/4 - a^2, written as a product for accuracy near the tube's edge
    # np.asarray: a 0-d dz gives a 0-d ndarray, not a numpy scalar (the contract returns an ndarray)
    return np.asarray(2.0 * np.sqrt(np.maximum((half - a) * (half + a), 0.0)), dtype=np.float64)


# ---------------------------------------------------------------- product ---

def clear_profiles(profiles_gl, B_gl, s_in, s_out, planes_z, d_um, z_ax_um, phi_rad=0.0, layers=None):
    """-> OwnClearance (SPEC Block 37). profiles_gl, s_in, s_out (n_planes, n_v); B_gl (n_planes,) gl > 0;
    planes_z (n_planes,) um; d_um > 0, z_ax_um, phi_rad the estimate's diameter, axis depth at the line and
    tilt; layers (n_layers, n_planes, n_v) dimensionless or None. The inputs are not changed. ValueError for
    shapes that disagree, a non-finite input, B <= 0, d <= 0, a split entry below -1e-9."""
    I = _real_array(profiles_gl, "profiles_gl")
    if I.ndim != 2:
        raise ValueError("profiles_gl must be (n_planes, n_v); got shape %r" % (I.shape,))
    n_planes, n_v = I.shape
    if n_v < 1:
        raise ValueError("profiles_gl must have at least one sample on the line (n_v >= 1)")
    S_in = _split_array(s_in, "s_in", I.shape)
    S_out = _split_array(s_out, "s_out", I.shape)
    B = _real_array(B_gl, "B_gl")
    if B.shape != (n_planes,):
        raise ValueError("B_gl must have shape (%d,); got %r" % (n_planes, B.shape))
    if np.any(B <= 0.0):
        raise ValueError("B_gl must be > 0 (gl)")
    z = _real_array(planes_z, "planes_z")
    if z.shape != (n_planes,):
        raise ValueError("planes_z must have shape (%d,); got %r" % (n_planes, z.shape))
    d = _real_scalar(d_um, "d_um")
    if d <= 0.0:
        raise ValueError("d_um must be > 0 (um); got %r" % d)
    z_ax = _real_scalar(z_ax_um, "z_ax_um")
    phi = _real_scalar(phi_rad, "phi_rad")
    L = None
    if layers is not None:
        L = _real_array(layers, "layers")
        if L.ndim != 3 or L.shape[1:] != I.shape:
            raise ValueError("layers must be (n_layers, %d, %d); got %r" % (n_planes, n_v, L.shape))
        if np.any(L < -SPLIT_NEG_TOL):
            raise ValueError("layers has an entry below -%g" % SPLIT_NEG_TOL)

    Bc = B[:, None]                                      # (n_planes, 1) gl
    whole = S_in + S_out                                 # the whole own shadow, dimensionless
    cleared = I + Bc * S_out
    model_in = Bc * (1.0 - S_in)
    model = Bc * (1.0 - S_in - S_out)
    residual = I - model

    in_peak = B * S_in.max(axis=1)
    out_peak = B * S_out.max(axis=1)
    n_star = np.argmax(whole, axis=1)                    # first index on a tie
    rows = np.arange(n_planes)
    w_star = whole[rows, n_star]
    share = np.full(n_planes, np.nan)
    pos = w_star > 0.0
    share[pos] = S_in[rows, n_star][pos] / w_star[pos]
    rms = np.sqrt(np.mean(residual * residual, axis=1))

    chord = chord_um(d, z - z_ax, phi)
    layers_gl = None if L is None else B[None, :, None] * L

    return OwnClearance(cleared_gl=cleared, model_in_gl=model_in, model_gl=model, residual_gl=residual,
                        chord_um=chord, in_peak_gl=in_peak, out_peak_gl=out_peak, in_share=share,
                        residual_rms_gl=rms, layers_gl=layers_gl)
