"""Defocus kernel width sigma_r(delta) -- Block 4 in specs/SPEC.md.

The rendering kernel for a slab at signed defocus delta (um, stage units) from
the plane being rendered is a circular Gaussian G of standard deviation
sigma_r(delta) (procedure s.3.4, Gaussian family). sigma_r is read from the
table (kernel_table_delta_um, kernel_table_sigma_um) up to the last knot
delta_max, in one of two ways chosen by kernel_family:

    "gaussian_table"          linear interpolation in |delta| (the first reading);
    "gaussian_table_spline"   amendment 4 (D-181, D-265, D-272): the not-a-knot cubic spline of the growth of
                              the variance G(delta) = sigma_r(delta)^2 - sigma_r(0)^2 through the knot values on
                              the signed knots -delta_max .. delta_max (Block 10's reading of its own growth
                              curve), sigma_r = sqrt(sigma_r(0)^2 + max(G, 0)).

Beyond the last knot it is continued as

    sigma_r(delta) = sigma_r(delta_max) + gamma (|delta| - delta_max)

with gamma = kernel_continuation_slope ("linear"), gamma = 0 ("frozen") or
gamma = sigma_r(delta_max) / delta_max ("proportional"); the three are one
family (impl-handoff, Configuration; mathematics Eq. 17 for the slopes). With
the spline reading a fourth continuation exists, "tangent" (D-265): G continues
along the spline's tangent at the last knot, G(delta) = G(delta_max) +
G'(delta_max) (|delta| - delta_max), which is how Block 10 continues its growth
curve; it is used by the near stage of the kernel calibration's step 2.

The table is the ideal-Debye core width at 550 nm (mathematics s.3.4), NOT a
calibration; Block 10 replaces it.

Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
from scipy.interpolate import CubicSpline

TABLE_FAMILIES = ("gaussian_table", "gaussian_table_spline")    # the families read from the table of knots
SPLINE_MIN_KNOTS = 3                                            # the knot at 0 and at least two more


def growth_spline(rcfg):
    """The not-a-knot cubic spline of G(delta) = sigma_r(delta)^2 - sigma_r(0)^2 (um^2) through the table's knot
    values on the signed knots -delta_max .. delta_max (amendment 4; D-181.1, D-265.1): a scipy CubicSpline,
    even in delta because its data are. ValueError: fewer than SPLINE_MIN_KNOTS knots."""
    knots = np.asarray(rcfg.kernel_table_delta_um, dtype=float)
    values = np.asarray(rcfg.kernel_table_sigma_um, dtype=float)
    if knots.size < SPLINE_MIN_KNOTS:
        raise ValueError("the spline reading needs at least %d knots, got %d" % (SPLINE_MIN_KNOTS, knots.size))
    g = values ** 2 - values[0] ** 2
    x = np.concatenate([-knots[:0:-1], knots])
    y = np.concatenate([g[:0:-1], g])
    return CubicSpline(x, y, bc_type="not-a-knot")


def sigma_r(delta, rcfg):
    """sigma_r(delta) in um for an array (any shape) or scalar delta in um.

    rcfg: a RendererConfig. Even in delta. kernel_family "gaussian_table": linear in |delta| between the knots;
    "gaussian_table_spline": the spline of the growth of the variance (module docstring), the growth floored at 0
    so that sigma_r >= sigma_r(0). Raises NotImplementedError for the "empirical" kernel family (Block 10) and
    ValueError for an unknown continuation, for "tangent" without the spline reading, and for a delta that is
    not finite.
    """
    if rcfg.kernel_family not in TABLE_FAMILIES:
        raise NotImplementedError("kernel_family %r is not available before Block 10" % (rcfg.kernel_family,))
    spline = rcfg.kernel_family == "gaussian_table_spline"
    knots = np.asarray(rcfg.kernel_table_delta_um, dtype=float)
    values = np.asarray(rcfg.kernel_table_sigma_um, dtype=float)
    d = np.abs(np.asarray(delta, dtype=float))
    if not np.all(np.isfinite(d)):
        raise ValueError("delta must be finite")
    d_max, s_max = knots[-1], values[-1]
    if rcfg.kernel_continuation == "linear":
        gamma = float(rcfg.kernel_continuation_slope)
    elif rcfg.kernel_continuation == "frozen":
        gamma = 0.0
    elif rcfg.kernel_continuation == "proportional":
        gamma = s_max / d_max
    elif rcfg.kernel_continuation == "tangent":
        if not spline:
            raise ValueError("kernel_continuation 'tangent' needs kernel_family 'gaussian_table_spline', got %r"
                             % (rcfg.kernel_family,))
        gamma = None
    else:
        raise ValueError("unknown kernel_continuation %r" % (rcfg.kernel_continuation,))
    if not spline:
        inside = np.interp(d, knots, values)
        return np.where(d > d_max, s_max + gamma * (d - d_max), inside)
    sp = growth_spline(rcfg)
    var0 = float(values[0]) ** 2
    inside = np.sqrt(var0 + np.maximum(sp(np.minimum(d, d_max)), 0.0))
    if gamma is None:                                    # D-265: the growth along the spline's tangent at d_max
        g_out = float(sp(d_max)) + float(sp(d_max, 1)) * (d - d_max)
        outside = np.sqrt(var0 + np.maximum(g_out, 0.0))
    else:
        outside = s_max + gamma * (d - d_max)
    return np.where(d > d_max, outside, inside)


def with_in_focus_width(rcfg, sigma_r0_um):
    """rcfg with its Gaussian kernel table re-anchored at the in-focus width sigma_r0_um (um) -- Block 4,
    amendment 1 (D-165).

    The growth of the variance with defocus is kept: for every knot delta_m,
    sigma'(delta_m) = sqrt(sigma(delta_m)^2 - sigma(0)^2 + sigma_r0_um^2), i.e. Block 10's
    sigma(delta_m)^2 = sigma(0)^2 + G(delta_m) with another sigma(0) and the same G. sigma_r0_um of the result is
    the new width; the continuation beyond the last knot (kind and slope) is unchanged. The in-focus width is
    configured, never identified by the stacks (config.py), and a node's fitted diameter is confounded with it:
    this is the knob to try other values. The family (the reading between knots) is kept. ValueError: a
    kernel family that is not read from the table (TABLE_FAMILIES), a width
    that is not finite and > 0, a knot whose new variance is <= 0.
    """
    if rcfg.kernel_family not in TABLE_FAMILIES:
        raise ValueError("with_in_focus_width needs a table family %r, got %r" % (TABLE_FAMILIES, rcfg.kernel_family))
    s0 = float(sigma_r0_um)
    if not (math.isfinite(s0) and s0 > 0.0):
        raise ValueError("sigma_r0_um must be finite and > 0 (um), got %r" % (sigma_r0_um,))
    old = np.asarray(rcfg.kernel_table_sigma_um, dtype=float)
    var = old ** 2 - old[0] ** 2 + s0 ** 2
    if np.any(var <= 0.0):
        raise ValueError("with_in_focus_width: a knot's variance would be <= 0 at sigma_r0_um = %g um" % s0)
    new = np.sqrt(var)
    new[0] = s0                                    # exactly: the table's first value and sigma_r0_um are one number
    return dataclasses.replace(rcfg, sigma_r0_um=s0, kernel_table_sigma_um=tuple(float(x) for x in new))


def with_kernel(rcfg, sigma_r0_um, growth_scale=1.0):
    """rcfg with its Gaussian kernel table re-anchored at the in-focus width sigma_r0_um (um) and the growth of its
    variance with defocus scaled by growth_scale -- Block 4, amendment 2 (D-201: the in-focus width calibrated on
    thick nodes, with a free growth scale).

    For every knot delta_m, with c = growth_scale:

        sigma'(delta_m) = sqrt(c (sigma(delta_m)^2 - sigma(0)^2) + sigma_r0_um^2),

    i.e. Block 10's sigma(delta_m)^2 = sigma(0)^2 + G(delta_m) with another sigma(0) and the growth c G. Beyond the
    last knot the continuation keeps its kind: "linear" with its slope times sqrt(c) (where the growth dominates,
    sigma' tends to sqrt(c) sigma), "frozen" unchanged, "proportional" recomputed by sigma_r from the new table.
    growth_scale = 1 gives with_in_focus_width's table bit for bit; "tangent" follows the new table by itself.
    ValueError: a kernel family that is not read from the table (TABLE_FAMILIES), a width or a scale that is not finite and > 0, a knot whose new variance is <= 0.
    """
    if rcfg.kernel_family not in TABLE_FAMILIES:
        raise ValueError("with_kernel needs a table family %r, got %r" % (TABLE_FAMILIES, rcfg.kernel_family))
    s0, c = float(sigma_r0_um), float(growth_scale)
    if not (math.isfinite(s0) and s0 > 0.0):
        raise ValueError("sigma_r0_um must be finite and > 0 (um), got %r" % (sigma_r0_um,))
    if not (math.isfinite(c) and c > 0.0):
        raise ValueError("growth_scale must be finite and > 0, got %r" % (growth_scale,))
    old = np.asarray(rcfg.kernel_table_sigma_um, dtype=float)
    var = c * (old ** 2 - old[0] ** 2) + s0 ** 2      # c = 1: with_in_focus_width's expression, the same floats
    if np.any(var <= 0.0):
        raise ValueError("with_kernel: a knot's variance would be <= 0 at sigma_r0_um = %g um, growth_scale = %g"
                         % (s0, c))
    new = np.sqrt(var)
    new[0] = s0                                    # the table's first value and sigma_r0_um are one number
    repl = dict(sigma_r0_um=s0, kernel_table_sigma_um=tuple(float(x) for x in new))
    if rcfg.kernel_continuation == "linear":
        repl["kernel_continuation_slope"] = float(rcfg.kernel_continuation_slope) * math.sqrt(c)
    return dataclasses.replace(rcfg, **repl)
