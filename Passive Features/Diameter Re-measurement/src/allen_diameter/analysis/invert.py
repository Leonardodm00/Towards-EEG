"""Correction by inversion of the mean response -- Block 7 in specs/SPEC.md
(procedure Eq. 1, s.3.9; mathematics Eqs. 20-21; D5).

At a node with fitted d_hat and tilt phi, the corrected diameter d_tilde
solves m_hat(d, phi | C) = d_hat on the table's [d_min, d_max]
(mathematics Eq. 20), m_hat = d * b_hat. The sign changes of m_hat - d_hat on
inversion_grid_points log-spaced diameters decide: none -> out_of_domain,
more than one -> non_monotone, exactly one -> scipy.optimize.brentq in that
bracket. At the root: large_correction when |b_hat - 1| > bias_flag_threshold
(D5), high_failure when the local failure rate > max_failure_rate.

The handoff's shortcut d_hat / b_hat(d_hat, phi) looks b up at the measured
diameter; its relative error is -beta (b - 1) / (1 + beta (b - 1)) with
beta = dln b / dln d (mathematics Eq. 21). It is kept for comparison only.

Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.optimize import brentq

from .phantoms import reject_reasons


@dataclass(frozen=True)
class Inversion:
    d_tilde_um: float          # the corrected diameter; NaN when not inverted or flagged (then filled)
    d_root_um: float           # the root of m_hat = d_hat even when flagged (diagnostics); NaN when none
    b_hat: float               # b_hat at the root (NaN when none)
    flags: Tuple[str, ...]


def shortcut(d_hat, phi, table):
    """The handoff's d_hat / b_hat(d_hat, phi) -- comparison only."""
    return float(d_hat) / float(table.b_hat(float(d_hat), float(phi)))


def invert_node(d_hat, phi, table, cfg):
    """Inversion of one node (see the module docstring); cfg: DiameterConfig."""
    c = cfg.correction
    nan = float("nan")
    d_hat, phi = float(d_hat), float(phi)
    if not (math.isfinite(d_hat) and d_hat > 0 and math.isfinite(phi)):
        return Inversion(nan, nan, nan, ("no_estimate",))
    flags = []
    lo_phi, hi_phi = table.phi_range_rad
    if not (lo_phi <= phi <= hi_phi):
        flags.append("phi_out_of_domain")
    d_lo, d_hi = table.d_range
    grid = np.exp(np.linspace(math.log(d_lo), math.log(d_hi), int(c.inversion_grid_points)))
    g = table.m_hat(grid, np.full(grid.shape, phi)) - d_hat
    side = np.where(g >= 0.0, 1, -1)              # a zero counts as above: one bracket per crossing
    changes = np.flatnonzero(side[:-1] != side[1:])
    if changes.size == 0:
        return Inversion(nan, nan, nan, tuple(flags + ["out_of_domain"]))
    if changes.size > 1:
        return Inversion(nan, nan, nan, tuple(flags + ["non_monotone"]))
    k = int(changes[0])
    root = float(brentq(lambda d: float(table.m_hat(d, phi)) - d_hat, grid[k], grid[k + 1],
                        xtol=1e-12, rtol=4 * np.finfo(float).eps))
    b = float(table.b_hat(root, phi))
    if abs(b - 1.0) > c.bias_flag_threshold:
        flags.append("large_correction")
    if float(table.failure_rate(root, phi)) > c.max_failure_rate:
        flags.append("high_failure")
    return Inversion(nan if flags else root, root, b, tuple(flags))


def correct_nodes(results, table, cfg):
    """Inversion of every NodeResult; nodes outside S keep their reasons as flags.
    Refuses a table built for another estimator (procedure s.3.11)."""
    table.check_estimator(cfg)
    out = []
    for res in results:
        reasons = reject_reasons(res)
        if reasons:
            out.append(Inversion(float("nan"), float("nan"), float("nan"), tuple(reasons)))
        else:
            out.append(invert_node(res.d_hat_um, res.phi_rad, table, cfg))
    return out
