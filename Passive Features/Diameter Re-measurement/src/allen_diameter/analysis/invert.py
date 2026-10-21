"""Correction by inversion of the mean response -- Blocks 7 and 14.5 in
specs/SPEC.md (procedure Eq. 1, s.3.9; mathematics Eqs. 20-21; D5, D-030,
D-063.1, D-063.4, D-071.4).

A node with fitted d_hat, mu_hat and tilt phi is inverted along the table's
line (., phi, ln(mu_hat d_hat)) (D-063.1): the corrected diameter d_tilde
solves m_hat(d, phi, mu_hat d_hat | C) = d_hat on the table's [d_min, d_max]
(mathematics Eq. 20), m_hat = d b_hat. The sign changes of m_hat - d_hat on
inversion_grid_points log-spaced diameters decide: none -> out_of_domain,
more than one -> non_monotone, exactly one -> scipy.optimize.brentq in that
bracket. Flags:

  no_estimate          d_hat, mu_hat or phi not finite, d_hat <= 0 or mu_hat d_hat <= 0
  phi_out_of_domain    phi outside the table's tilt axis (clamped for the root)
  mu_d_out_of_domain   ln(mu_hat d_hat) outside the third axis: a LABEL only
                       (D-071.4): clamped to the axis' nearest end, the node
                       inverted there and not filled for it (the clamp is the
                       assistant's)
  out_of_domain, non_monotone
  large_correction     |b_hat - 1| > bias_flag_threshold at the root (D5)
  high_failure         the failure rate at the root > max_failure_rate
  off_support          the kernel mass at the root below the table's q_loo
                       (D-063.4), or a grid point around the root without
                       support (the support floor of Block 14.4)

Every flag but the labels sends the node to the fill (d_tilde NaN, the root
kept as d_root_um).

The handoff's shortcut d_hat / b_hat(d_hat, phi, mu_hat d_hat) looks b up at
the measured diameter; its relative error is -beta (b - 1) / (1 + beta (b - 1))
with beta = dln b / dln d (mathematics Eq. 21). It is kept for comparison only.

Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.optimize import brentq

from .phantoms import reject_reasons

LABEL_FLAGS = ("mu_d_out_of_domain",)     # flags that do not send a node to the fill (D-071.4)


@dataclass(frozen=True)
class Inversion:
    d_tilde_um: float          # the corrected diameter; NaN when not inverted or flagged (then filled)
    d_root_um: float           # the root of m_hat = d_hat even when flagged (diagnostics); NaN when none
    b_hat: float               # b_hat at the root (NaN when none)
    flags: Tuple[str, ...]


def shortcut(d_hat, phi, mu_hat, table):
    """The handoff's d_hat / b_hat(d_hat, phi, mu_hat d_hat) -- comparison only."""
    return float(d_hat) / float(table.b_hat(float(d_hat), float(phi), float(mu_hat) * float(d_hat)))


def invert_node(d_hat, phi, mu_hat, table, cfg):
    """Inversion of one node (see the module docstring); cfg: DiameterConfig."""
    c = cfg.correction
    nan = float("nan")
    d_hat, phi, mu_hat = float(d_hat), float(phi), float(mu_hat)
    mu_d = mu_hat * d_hat
    if not (math.isfinite(d_hat) and d_hat > 0 and math.isfinite(phi) and math.isfinite(mu_d) and mu_d > 0):
        return Inversion(nan, nan, nan, ("no_estimate",))
    flags = []
    lo_phi, hi_phi = table.phi_range_rad
    if not (lo_phi <= phi <= hi_phi):
        flags.append("phi_out_of_domain")
        phi = min(max(phi, lo_phi), hi_phi)
    lo_md, hi_md = table.mu_d_range
    if not (lo_md <= mu_d <= hi_md):
        flags.append("mu_d_out_of_domain")
        mu_d = min(max(mu_d, lo_md), hi_md)
    d_lo, d_hi = table.d_range
    grid = np.exp(np.linspace(math.log(d_lo), math.log(d_hi), int(c.inversion_grid_points)))
    g = table.m_hat(grid, phi, mu_d) - d_hat
    side = np.where(g >= 0.0, 1, -1)              # a zero counts as above: one bracket per crossing
    changes = np.flatnonzero(side[:-1] != side[1:])
    if changes.size == 0:
        return Inversion(nan, nan, nan, tuple(flags + ["out_of_domain"]))
    if changes.size > 1:
        return Inversion(nan, nan, nan, tuple(flags + ["non_monotone"]))
    k = int(changes[0])
    root = float(brentq(lambda d: float(table.m_hat(d, phi, mu_d)) - d_hat, grid[k], grid[k + 1],
                        xtol=1e-12, rtol=4 * np.finfo(float).eps))
    b = float(table.b_hat(root, phi, mu_d))
    if abs(b - 1.0) > c.bias_flag_threshold:
        flags.append("large_correction")
    if float(table.failure_rate(root, phi, mu_d)) > c.max_failure_rate:
        flags.append("high_failure")
    if float(table.mass(root, phi, mu_d)) < table.q_loo or float(table.support(root, phi, mu_d)) < 1.0 - 1e-12:
        flags.append("off_support")
    fill = [f for f in flags if f not in LABEL_FLAGS]
    return Inversion(nan if fill else root, root, b, tuple(flags))


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
            out.append(invert_node(res.d_hat_um, res.phi_rad, res.mu_hat_per_um, table, cfg))
    return out
