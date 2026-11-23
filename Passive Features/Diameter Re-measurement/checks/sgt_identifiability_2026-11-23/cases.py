#!/usr/bin/env python3
"""Standalone check (NOT pipeline code), 2026-11-23: the cases of the SGT identifiability check (D-222).

Geometries: d in {0.4, 0.8, 1.5, 2.5} um x mu in {0.5, 1.5} /um x the axis depth z_ax in {0.0, 0.5} um (planes
-4..4 at k dz, so 0.0 is on the plane of best focus k* = 0 and 0.5 um is 0.5 um off it); the axis 0.04 um aside
of the line's origin (a generic sub-pixel offset). Kernel truths (beta, xi, zeta), as the brief names them:
gauss (2, 0, 1), heavy (2, 0.5, 1), b13 (1.3, 0, 1) -- the brief's "flatter core"; at equal FWHM a generalized
normal with beta < 2 has a sharper, cusp-like core with heavier shoulders than the Gaussian (a flatter core is
beta > 2) -- and skew (2, 0.3, 1.3), the zeta option's case.

Pure ASCII, LF.
"""
from __future__ import annotations

import numpy as np

V0_UM = 0.04
D_LIST = (0.4, 0.8, 1.5, 2.5)
MU_LIST = (0.5, 1.5)
Z_LIST = (0.0, 0.5)
TRUTHS = {"gauss": (2.0, 0.0, 1.0), "heavy": (2.0, 0.5, 1.0), "b13": (1.3, 0.0, 1.0), "skew": (2.0, 0.3, 1.3)}
TRUTH_ORDER = ("gauss", "heavy", "b13", "skew")
SEED = 20261123


def geometries():
    out = []
    for d in D_LIST:
        for mu in MU_LIST:
            for z in Z_LIST:
                out.append(dict(d=d, mu=mu, z_ax=z, v0=V0_UM))
    return out


def geom_id(g):
    return "d%.1f_mu%.1f_z%.1f" % (g["d"], g["mu"], g["z_ax"])


def case_index(g, truth):
    """A stable integer for the seeds: geometry index * 10 + truth index."""
    gi = [geom_id(x) for x in geometries()].index(geom_id(g))
    return gi * 10 + TRUTH_ORDER.index(truth)


def phi_true(g, truth):
    return np.array([g["d"], g["mu"], g["v0"], g["z_ax"]] + list(TRUTHS[truth]), float)


def models_for(truth):
    """The fitted models whose Fisher information and recovery the check reports for a truth: z1 = (beta, xi)
    free with zeta = 1 (the default of D-222), zf = (beta, xi, zeta) free (the option). z1 is the true model only
    for a symmetric truth."""
    return ("z1", "zf") if TRUTHS[truth][2] == 1.0 else ("zf",)
