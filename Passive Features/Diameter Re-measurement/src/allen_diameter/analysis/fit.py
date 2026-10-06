"""Per-node diameter fit -- Block 3 in specs/SPEC.md (D-018.2, D-019.1).

For each fixed node i, with the profile samples (v_n, I_i(v_n)), the tilt
phi_i (rad, from the line fit), the fixed background B_bar_i (D-018.1) and
the fixed blur sigma_fit (MeasureConfig.sigma_fit_um):

    (d_hat, mu_hat, v0_hat) = argmin_{(d, mu, v0) in box}
        sum_n [ I_i(v_n) - I_model(v_n | d, alpha = mu d / cos phi_i, v0) ]^2    (D-019.1)

with I_model of model/tube_model.py (B3.1). The residuals handed to
scipy.optimize.least_squares are divided by B_bar_i, a positive constant,
which leaves the minimiser unchanged. One run per start (MeasureConfig.
fit_multistart_factors); the reported run is the converged one of least
residual sum of squares, the first in start order on ties.

When no start converges, the finite run of least residual sum of squares is
reported with status "failed". A parameter within fit_at_bound_rel_tol of
its bound range from a bound makes the status "at_bound"; for a dark tube
(alpha above about 5) the profile hardly depends on mu, and the fit may stop
at mu's upper bound with a sound d_hat.

fit_profile_alpha is the alpha-parameterised form of D-018.2, kept only as a
labelled comparison: with no bound active the two forms have the same
minimiser in (d, v0) (bijection, D-019).

Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.optimize import least_squares

from ..model import tube_model

FIT_STATUSES = ("converged", "at_bound", "failed")
FIT_FORMS = ("mu", "alpha")


@dataclass(frozen=True)
class FitResult:
    """Outcome of one per-node fit (units in the field names; alpha_hat and,
    in the alpha form, mu_hat_per_um are derived)."""

    d_hat_um: float
    mu_hat_per_um: float
    v0_hat_um: float
    alpha_hat: float          # mu_hat d_hat / cos(phi); diagnostics only (D-019)
    phi_rad: float            # the tilt the fit was given
    form: str                 # "mu" (deliverable) or "alpha" (D-018.2 comparison)
    status: str               # one of FIT_STATUSES
    at_bound: Tuple[str, ...]  # e.g. ("mu_lo",); empty when no parameter is at a bound
    rss_gl2: float            # sum_n (I_n - I_model(v_n))^2, grey levels squared
    n_samples: int
    n_starts: int
    best_start: int           # index into the starts (fit_multistart_factors order)
    n_converged: int          # starts with scipy status > 0 and a finite result
    nfev: int                 # residual evaluations summed over starts (scipy's nfev: the 2-point Jacobian's are not counted)
    n_nodes: int              # Gauss-Legendre nodes of the model quadrature
    message: str              # scipy's message for the reported run


def _check_profile(v, I, phi, B_bar):
    v = np.asarray(v, dtype=float)
    I = np.asarray(I, dtype=float)
    if v.ndim != 1 or I.shape != v.shape:
        raise ValueError("v and I must be 1-D arrays of one shape, got %r and %r" % (v.shape, I.shape))
    if v.size < 4:
        raise ValueError("the fit needs at least 4 samples, got %d" % v.size)
    if not (np.all(np.isfinite(v)) and np.all(np.isfinite(I))):
        raise ValueError("non-finite profile sample (flag the node upstream instead)")
    if np.any(np.diff(v) <= 0):
        raise ValueError("v must increase strictly")
    phi = float(phi)
    if not (0.0 <= phi < 0.5 * math.pi):
        raise ValueError("phi must lie in [0, pi/2) rad, got %r" % phi)
    B_bar = float(B_bar)
    if not (math.isfinite(B_bar) and B_bar > 0):
        raise ValueError("B_bar must be finite and > 0, got %r" % B_bar)
    return v, I, phi, B_bar


def _bounds(cfg, form, cos_phi):
    (d_lo, d_hi), (mu_lo, mu_hi), (v0_lo, v0_hi) = (cfg.fit_d_bounds_um, cfg.fit_mu_bounds_per_um,
                                                    cfg.fit_v0_bounds_um)
    if form == "mu":
        dark = (mu_lo, mu_hi)
    elif form == "alpha":
        # the box containing the image of the mu box under (d, mu) -> (d, mu d / cos phi)
        dark = (mu_lo * d_lo / cos_phi, mu_hi * d_hi / cos_phi)
    else:
        raise ValueError("form must be one of %r, got %r" % (FIT_FORMS, form))
    lb = np.array([d_lo, dark[0], v0_lo], dtype=float)
    ub = np.array([d_hi, dark[1], v0_hi], dtype=float)
    return lb, ub


def half_depth_width(v, I, B_bar, n_star):
    """Width (um) of the dip at half depth around sample n_star, or None.

    Level h = (B_bar + I[n_star]) / 2. On each side, the sample nearest to
    n_star with I >= h and its inner neighbour bracket the crossing, which is
    placed by linear interpolation. None when I[n_star] >= B_bar (no dip) or
    when one side has no sample with I >= h.
    """
    v = np.asarray(v, dtype=float)
    I = np.asarray(I, dtype=float)
    I_min = I[n_star]
    if not (I_min < B_bar):
        return None
    h = 0.5 * (B_bar + I_min)
    above = I >= h
    left = np.flatnonzero(above[:n_star])
    right = np.flatnonzero(above[n_star + 1:])
    if left.size == 0 or right.size == 0:
        return None
    nl = int(left[-1])                 # I[nl] >= h > I[nl + 1]
    nr = int(n_star + 1 + right[0])    # I[nr - 1] < h <= I[nr]
    x_left = v[nl] + (h - I[nl]) * (v[nl + 1] - v[nl]) / (I[nl + 1] - I[nl])
    x_right = v[nr - 1] + (h - I[nr - 1]) * (v[nr] - v[nr - 1]) / (I[nr] - I[nr - 1])
    return float(x_right - x_left)


def start_points(v, I, phi, B_bar, cfg, form="mu"):
    """Start vectors of the fit, one row per multistart factor: (n_starts, 3)
    float64 with columns (d um, mu 1/um or alpha, v0 um), each inside its bounds.

    n_star = the smallest sample among those with v inside the v0 bounds;
    v0 starts at v[n_star]. d0 by cfg.fit_start_rule: "profile" = sqrt(max(w^2
    - 8 ln2 sigma_fit^2, 0)) with w = half_depth_width (cfg.fit_d0_um when w is
    None); "fixed" = cfg.fit_d0_um. Start k has d = d0 f_k and a darkness
    whose centre-line absorbance equals the observed depth,
    alpha0 = -ln(min(1, max(I[n_star] / B_bar, smallest positive float))).
    """
    v, I, phi, B_bar = _check_profile(v, I, phi, B_bar)
    cos_phi = math.cos(phi)
    lb, ub = _bounds(cfg, form, cos_phi)
    window = np.flatnonzero((v >= lb[2]) & (v <= ub[2]))
    if window.size == 0:
        raise ValueError("no profile sample inside the v0 bounds %r" % (cfg.fit_v0_bounds_um,))
    n_star = int(window[np.argmin(I[window])])
    if cfg.fit_start_rule == "profile":
        w = half_depth_width(v, I, B_bar, n_star)
        if w is None:
            d0 = cfg.fit_d0_um
        else:
            d0 = math.sqrt(max(w * w - 8.0 * math.log(2.0) * cfg.sigma_fit_um ** 2, 0.0))
    elif cfg.fit_start_rule == "fixed":
        d0 = cfg.fit_d0_um
    else:
        raise ValueError("unknown fit_start_rule %r" % (cfg.fit_start_rule,))
    ratio = min(1.0, max(I[n_star] / B_bar, np.finfo(float).tiny))
    alpha0 = -math.log(ratio)
    rows = []
    for f in cfg.fit_multistart_factors:
        d_s = min(max(d0 * f, lb[0]), ub[0])
        dark = alpha0 * cos_phi / d_s if form == "mu" else alpha0
        rows.append((d_s, min(max(dark, lb[1]), ub[1]), v[n_star]))
    return np.array(rows, dtype=float)


def _fit(v, I, phi, B_bar, cfg, form):
    if cfg.mu_mode != "per_node":
        raise NotImplementedError("mu_mode %r (a shared mu) is a planned comparison, not in Block 3"
                                  % (cfg.mu_mode,))
    v, I, phi, B_bar = _check_profile(v, I, phi, B_bar)
    sigma = cfg.sigma_fit_um
    n_nodes = tube_model.n_quadrature_nodes(cfg.fit_d_bounds_um[1], sigma, cfg.fit_quad_min_nodes,
                                            cfg.fit_quad_nodes_per_sigma)
    cos_phi = math.cos(phi)
    lb, ub = _bounds(cfg, form, cos_phi)
    starts = start_points(v, I, phi, B_bar, cfg, form)

    if form == "mu":
        def residuals(p):
            alpha = tube_model.alpha_from_mu(p[1], p[0], phi)
            return (tube_model.model_profile(v, p[0], alpha, p[2], sigma, B_bar, n_nodes) - I) / B_bar
    else:
        def residuals(p):
            return (tube_model.model_profile(v, p[0], p[1], p[2], sigma, B_bar, n_nodes) - I) / B_bar

    runs = [least_squares(residuals, x0, bounds=(lb, ub), method="trf", jac="2-point", x_scale="jac",
                          ftol=cfg.fit_tol, xtol=cfg.fit_tol, gtol=cfg.fit_tol, max_nfev=cfg.fit_max_nfev)
            for x0 in starts]
    finite = [k for k, r in enumerate(runs) if np.all(np.isfinite(r.x)) and np.isfinite(r.cost)]
    converged = [k for k in finite if runs[k].status > 0]
    pool = converged if converged else finite
    best = min(pool, key=lambda k: (runs[k].cost, k)) if pool else 0
    r = runs[best]

    d_hat, dark_hat, v0_hat = (float(x) for x in r.x)
    if form == "mu":
        mu_hat, alpha_hat = dark_hat, float(tube_model.alpha_from_mu(dark_hat, d_hat, phi))
    else:
        mu_hat, alpha_hat = dark_hat * cos_phi / d_hat, dark_hat
    names = ("d", form, "v0")
    at_bound = []
    for name, x, lo, hi in zip(names, r.x, lb, ub):
        tol = cfg.fit_at_bound_rel_tol * (hi - lo)
        if x - lo <= tol:
            at_bound.append(name + "_lo")
        if hi - x <= tol:
            at_bound.append(name + "_hi")
    if not converged:
        status = "failed"
    elif at_bound:
        status = "at_bound"
    else:
        status = "converged"
    return FitResult(d_hat_um=d_hat, mu_hat_per_um=mu_hat, v0_hat_um=v0_hat, alpha_hat=alpha_hat,
                     phi_rad=phi, form=form, status=status, at_bound=tuple(at_bound),
                     rss_gl2=float(np.sum((np.asarray(r.fun) * B_bar) ** 2)), n_samples=int(v.size),
                     n_starts=len(runs), best_start=int(best), n_converged=len(converged),
                     nfev=int(sum(int(q.nfev) for q in runs)), n_nodes=int(n_nodes), message=str(r.message))


def fit_profile(v, I, phi, B_bar, cfg):
    """The deliverable fit (D-019.1): parameters (d, mu, v0), mu per node (D-023).

    v (n,) strictly increasing um; I (n,) finite grey levels; phi rad in
    [0, pi/2); B_bar > 0 grey levels; cfg a MeasureConfig. Returns FitResult.
    """
    return _fit(v, I, phi, B_bar, cfg, "mu")


def fit_profile_alpha(v, I, phi, B_bar, cfg):
    """The alpha form of D-018.2 (labelled comparison only): parameters
    (d, alpha, v0); mu_hat is reported as alpha_hat cos(phi) / d_hat."""
    return _fit(v, I, phi, B_bar, cfg, "alpha")
