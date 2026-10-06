"""Defocus kernel width sigma_r(delta) -- Block 4 in specs/SPEC.md.

The rendering kernel for a slab at signed defocus delta (um, stage units) from
the plane being rendered is a circular Gaussian G of standard deviation
sigma_r(delta) (procedure s.3.4, Gaussian family). sigma_r is read from the
table (kernel_table_delta_um, kernel_table_sigma_um) by linear interpolation
in |delta| up to the last knot delta_max, and continued beyond it as

    sigma_r(delta) = sigma_r(delta_max) + gamma (|delta| - delta_max)

with gamma = kernel_continuation_slope ("linear"), gamma = 0 ("frozen") or
gamma = sigma_r(delta_max) / delta_max ("proportional"); the three are one
family (impl-handoff, Configuration; mathematics Eq. 17 for the slopes).

The table is the ideal-Debye core width at 550 nm (mathematics s.3.4), NOT a
calibration; Block 10 replaces it.

Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import numpy as np


def sigma_r(delta, rcfg):
    """sigma_r(delta) in um for an array (any shape) or scalar delta in um.

    rcfg: a RendererConfig. Even in delta. Raises NotImplementedError for the
    "empirical" kernel family (Block 10) and ValueError for an unknown
    continuation.
    """
    if rcfg.kernel_family != "gaussian_table":
        raise NotImplementedError("kernel_family %r is not available before Block 10" % (rcfg.kernel_family,))
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
    else:
        raise ValueError("unknown kernel_continuation %r" % (rcfg.kernel_continuation,))
    inside = np.interp(d, knots, values)
    return np.where(d > d_max, s_max + gamma * (d - d_max), inside)
