"""Fit variants at one plane: other blur-kernel families and centre-weighted
fits -- Block 11 in specs/SPEC.md (D-046, the user's request of 2026-10-13,
14:50, and the answers to six questions; a diagnostic that changes no
measurement: Block 5's fit, its Gaussian kernel (D-026) and the estimator
hash stay as they are).

Inputs are Block 5's at one plane: the profile (v_n, I_n), n = 1..N (um,
grey levels), the tilt phi (rad) and the background B_bar (grey levels). A
variant is a kernel (family F, FWHM w, shape s; model/blur_kernels.py), a
width mode (w fixed at the FWHM of Block 5's Gaussian, 2 sqrt(2 ln 2)
sigma_fit, or fitted in FITTED_FWHM_BOUNDS_UM), a shape (fixed, or fitted
in blur_kernels.SHAPE_BOUNDS through blur_kernels.SHAPE_SCALE) and a
weighting (none, or omega_n = exp(-|v_n - v0| / lambda), lambda in um). Its
parameters theta = (d, mu, v0[, w][, t(s)]) minimise

    sum_n omega_n [(I_model(v_n | d, mu d / cos phi, v0; g_{F,w,s}) - I_n) / B_bar]^2

with I_model = tube_model.model_profile_kernel and omega_n = 1 unweighted;
d, mu, v0 in Block 5's bounds. Unweighted: least_squares (trf, 2-point
Jacobian, Block 5's tolerances) from Block 5's start points (fit.start_points)
completed with w0 = the fixed FWHM and s0 = blur_kernels.GAUSS_LIKE, and from
warm starts (other variants' solutions, listed per variant in fit_grid); the
converged run of least cost is reported (the first in start order on ties),
else the finite run of least cost with status "failed", as Block 5. Weighted,
the user's choice (the weights centred on the fit's own axis): iteratively
re-weighted, omega^(t)_n = exp(-|v_n - v0^(t)| / lambda) and theta^(t+1) the
minimiser with omega^(t) from theta^(t), theta^(0) the unweighted fit of the
same kernel and width mode, until |v0^(t+1) - v0^(t)| and |d^(t+1) - d^(t)|
are both <= IRLS_TOL_UM (irls_ok True) or after IRLS_MAX_ITER iterations
(irls_ok False). The variant gauss, fixed width, unweighted reproduces
fit.fit_profile (the baseline, d_base).

Quadrature: n_quadrature_nodes(d_hi, w_lo / (2 sqrt(2 ln 2)), ...) with w_lo
the narrowest width the fit can take (sigma_fit itself at fixed width, as
Block 5; FITTED_FWHM_BOUNDS_UM[0] when the width is fitted). A generalized
normal with beta <= 1 has a cusp at its peak, where the rule converges
slowly (0.05-0.27 grey levels against a direct convolution at w = 0.1 um,
d = 2 um); tripling or sextupling the nodes moved its fitted d by at most
0.003 um on synthetic profiles (2026-10-13), so the rule is kept.

Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import least_squares

from ..model import blur_kernels as bk
from ..model import tube_model
from . import fit as block5

WEIGHT_LAMBDAS_UM = (0.25, 0.5, 1.0, 2.0)      # D-046, the user's scan
FITTED_FWHM_BOUNDS_UM = (0.1, 2.0)             # D-046 (PROVISIONAL, the assistant's range)
IRLS_TOL_UM = 1e-4                             # D-046 (PROVISIONAL)
IRLS_MAX_ITER = 30                             # D-046 (PROVISIONAL)
WIDTH_MODES = ("fixed", "fitted")
# the kernel configurations (family, fixed shape or None, shape fitted), in the figures' order
KERNEL_CONFIGS = (("gauss", None, False), ("lorentz", None, False), ("voigt", 1.0, False), ("student", 3.0, False),
                  ("gennorm", 1.0, False), ("gennorm", 4.0, False), ("voigt", None, True), ("student", None, True),
                  ("gennorm", None, True), ("skewnorm", None, True))
BASE_KERNEL = ("gauss", None, False)


@dataclass(frozen=True)
class VariantSpec:
    """One variant: kernel (family, shape, shape_fitted), width mode, weighting
    (lam_um None = unweighted)."""

    family: str
    shape: Optional[float]
    shape_fitted: bool
    width_fitted: bool
    lam_um: Optional[float] = None

    @property
    def kernel(self):
        return (self.family, self.shape, self.shape_fitted)

    @property
    def kernel_label(self):
        return bk.kernel_label(self.family, self.shape, self.shape_fitted)

    @property
    def width_mode(self):
        return "fitted" if self.width_fitted else "fixed"

    @property
    def weight_label(self):
        return "none" if self.lam_um is None else "lam=%g" % self.lam_um

    @property
    def key(self):
        return "%s|w %s|%s" % (self.kernel_label, self.width_mode, self.weight_label)

    def unweighted(self):
        return VariantSpec(self.family, self.shape, self.shape_fitted, self.width_fitted, None)


@dataclass(frozen=True)
class VariantFit:
    """One variant's fit (units in the field names; shape None when the family has none)."""

    spec: VariantSpec
    d_um: float
    mu_per_um: float
    v0_um: float
    fwhm_um: float
    shape: Optional[float]
    alpha_hat: float
    status: str                       # converged / at_bound / failed, as Block 5
    at_bound: Tuple[str, ...]
    rss_w_gl2: float                  # sum_n omega_n (I_model - I_n)^2 (the unweighted rss when omega = 1)
    rss_gl2: float                    # sum_n (I_model - I_n)^2
    n_irls: int                       # 0 unweighted
    irls_ok: Optional[bool]           # None unweighted
    nfev: int
    n_starts: int
    best_start: str
    n_nodes: int
    message: str
    model_gl: np.ndarray = field(repr=False, compare=False)   # I_model(v_n), (N,)


def check_spec(spec):
    """ValueError for a spec that is not a variant: unknown family, a fixed
    shape outside its bounds or given with shape_fitted, a family without a
    shape given one, skewnorm with a fixed shape, lambda not finite > 0."""
    if spec.shape_fitted:
        if spec.family in bk.NO_SHAPE or spec.shape is not None:
            raise ValueError("%s: a fitted shape takes no fixed value, and gauss/lorentz have none" % (spec,))
    else:
        if spec.family in bk.ALWAYS_FITTED:
            raise ValueError("%s: the skew is always fitted (D-046)" % spec.family)
        bk.check_kernel(spec.family, spec.shape)
    if spec.family not in bk.FAMILIES:
        raise ValueError("unknown kernel family %r" % (spec.family,))
    if spec.lam_um is not None and not (math.isfinite(spec.lam_um) and spec.lam_um > 0):
        raise ValueError("lambda must be finite and > 0, got %r" % (spec.lam_um,))
    return spec


def variant_grid(kernels=KERNEL_CONFIGS, width_modes=WIDTH_MODES, lambdas_um=WEIGHT_LAMBDAS_UM):
    """The variants, kernel by kernel, then width mode, then weighting
    (unweighted first, then lambdas_um in order): 10 x 2 x 5 = 100 by default."""
    out = []
    for fam, shape, fitted in kernels:
        for wm in width_modes:
            if wm not in WIDTH_MODES:
                raise ValueError("width mode must be one of %r, got %r" % (WIDTH_MODES, wm))
            for lam in (None,) + tuple(float(x) for x in lambdas_um):
                out.append(check_spec(VariantSpec(fam, shape, fitted, wm == "fitted", lam)))
    return out


def fixed_fwhm_um(cfg):
    """The fixed width: the FWHM of Block 5's Gaussian, 2 sqrt(2 ln 2) sigma_fit (um)."""
    return bk.gauss_fwhm(cfg.sigma_fit_um)


def n_nodes_for(spec, cfg):
    """Gauss-Legendre nodes for the spec (module docstring)."""
    d_hi = cfg.fit_d_bounds_um[1]
    if spec.width_fitted:
        sigma = FITTED_FWHM_BOUNDS_UM[0] / bk.gauss_fwhm(1.0)
    else:
        sigma = cfg.sigma_fit_um
    return tube_model.n_quadrature_nodes(d_hi, sigma, cfg.fit_quad_min_nodes, cfg.fit_quad_nodes_per_sigma)


class _Layout:
    """theta <-> (d, mu, v0, w, s) for one spec, with its bounds."""

    def __init__(self, spec, cfg):
        self.spec, self.w_fixed = spec, fixed_fwhm_um(cfg)
        (d_lo, d_hi), (mu_lo, mu_hi), (v_lo, v_hi) = cfg.fit_d_bounds_um, cfg.fit_mu_bounds_per_um, cfg.fit_v0_bounds_um
        names, lb, ub = ["d", "mu", "v0"], [d_lo, mu_lo, v_lo], [d_hi, mu_hi, v_hi]
        if spec.width_fitted:
            names.append("w")
            lb.append(FITTED_FWHM_BOUNDS_UM[0])
            ub.append(FITTED_FWHM_BOUNDS_UM[1])
        if spec.shape_fitted:
            lo, hi = bk.SHAPE_BOUNDS[spec.family]
            names.append("s")
            lb.append(bk.shape_to_param(spec.family, lo))
            ub.append(bk.shape_to_param(spec.family, hi))
        self.names, self.lb, self.ub = tuple(names), np.array(lb, dtype=float), np.array(ub, dtype=float)

    def unpack(self, p):
        d, mu, v0 = float(p[0]), float(p[1]), float(p[2])
        i = 3
        w = self.w_fixed
        if self.spec.width_fitted:
            w, i = float(p[i]), i + 1
        s = self.spec.shape
        if self.spec.shape_fitted:
            lo, hi = bk.SHAPE_BOUNDS[self.spec.family]
            s = min(max(bk.param_to_shape(self.spec.family, p[i]), lo), hi)   # 10**log10(hi) may round above hi
        return d, mu, v0, w, s

    def pack(self, d, mu, v0, w, s):
        p = [d, mu, v0]
        if self.spec.width_fitted:
            p.append(w)
        if self.spec.shape_fitted:
            lo, hi = bk.SHAPE_BOUNDS[self.spec.family]
            s0 = bk.GAUSS_LIKE[self.spec.family] if s is None else min(max(float(s), lo), hi)
            p.append(bk.shape_to_param(self.spec.family, s0))
        return np.clip(np.array(p, dtype=float), self.lb, self.ub)


def _model(lay, p, v, phi, B_bar, n_nodes):
    d, mu, v0, w, s = lay.unpack(p)
    g = bk.kernel_pdf(lay.spec.family, w, s)
    alpha = tube_model.alpha_from_mu(mu, d, phi)
    return tube_model.model_profile_kernel(v, d, alpha, v0, g, B_bar, n_nodes)


def centre_weights(v, v0_um, lam_um):
    """omega_n = exp(-|v_n - v0| / lambda), the centre weights (D-046)."""
    return np.exp(-np.abs(np.asarray(v, dtype=float) - float(v0_um)) / float(lam_um))


def _runs(lay, v, I, phi, B_bar, cfg, n_nodes, starts, omega):
    sw = 1.0 if omega is None else np.sqrt(omega)

    def residuals(p):
        return sw * (_model(lay, p, v, phi, B_bar, n_nodes) - I) / B_bar
    max_nfev = max(int(cfg.fit_max_nfev), 100 * lay.lb.size)
    return [least_squares(residuals, x0, bounds=(lay.lb, lay.ub), method="trf", jac="2-point", x_scale="jac",
                          ftol=cfg.fit_tol, xtol=cfg.fit_tol, gtol=cfg.fit_tol, max_nfev=max_nfev)
            for _, x0 in starts]


def _result(spec, lay, run, runs, label, converged, v, I, phi, B_bar, cfg, n_nodes, omega, n_irls, irls_ok):
    x = np.asarray(run.x, dtype=float)
    if not np.all(np.isfinite(x)):     # no finite run: every number NaN, status failed
        nan = float("nan")
        return VariantFit(spec=spec, d_um=nan, mu_per_um=nan, v0_um=nan, fwhm_um=nan, shape=None, alpha_hat=nan,
                          status="failed", at_bound=(), rss_w_gl2=nan, rss_gl2=nan, n_irls=int(n_irls),
                          irls_ok=None if irls_ok is None else False, nfev=int(sum(int(q.nfev) for q in runs)),
                          n_starts=len(runs), best_start=label, n_nodes=int(n_nodes), message=str(run.message),
                          model_gl=np.full(np.shape(v), nan))
    d, mu, v0, w, s = lay.unpack(x)
    at_bound = []
    for name, xi, lo, hi in zip(lay.names, x, lay.lb, lay.ub):
        tol = cfg.fit_at_bound_rel_tol * (hi - lo)
        if xi - lo <= tol:
            at_bound.append(name + "_lo")
        if hi - xi <= tol:
            at_bound.append(name + "_hi")
    status = "failed" if not converged else ("at_bound" if at_bound else "converged")
    model = _model(lay, x, v, phi, B_bar, n_nodes)
    r2 = (model - I) ** 2
    return VariantFit(spec=spec, d_um=d, mu_per_um=mu, v0_um=v0, fwhm_um=w, shape=s,
                      alpha_hat=float(tube_model.alpha_from_mu(mu, d, phi)), status=status, at_bound=tuple(at_bound),
                      rss_w_gl2=float(np.sum(r2 if omega is None else omega * r2)), rss_gl2=float(np.sum(r2)),
                      n_irls=int(n_irls), irls_ok=irls_ok, nfev=int(sum(int(q.nfev) for q in runs)),
                      n_starts=len(runs), best_start=label, n_nodes=int(n_nodes), message=str(run.message),
                      model_gl=model)


def fit_unweighted(v, I, phi, B_bar, cfg, spec, warm=()):
    """The unweighted fit of spec (lam_um ignored): starts from Block 5's
    start points, then from warm, a sequence of (label, VariantFit) whose
    (d, mu, v0, w, s) seed this spec's parameters. Returns VariantFit."""
    v, I, phi, B_bar = block5._check_profile(v, I, phi, B_bar)
    spec = check_spec(spec.unweighted())
    lay, n_nodes = _Layout(spec, cfg), n_nodes_for(spec, cfg)
    starts = [("block5:%d" % k, lay.pack(row[0], row[1], row[2], lay.w_fixed, None))
              for k, row in enumerate(block5.start_points(v, I, phi, B_bar, cfg, "mu"))]
    for label, f in warm:
        if f is not None and np.isfinite(f.d_um):
            starts.append(("warm:" + label, lay.pack(f.d_um, f.mu_per_um, f.v0_um, f.fwhm_um, f.shape)))
    runs = _runs(lay, v, I, phi, B_bar, cfg, n_nodes, starts, None)
    finite = [k for k, r in enumerate(runs) if np.all(np.isfinite(r.x)) and np.isfinite(r.cost)]
    converged = [k for k in finite if runs[k].status > 0]
    pool = converged if converged else finite
    best = min(pool, key=lambda k: (runs[k].cost, k)) if pool else 0
    return _result(spec, lay, runs[best], runs, starts[best][0], bool(converged), v, I, phi, B_bar, cfg, n_nodes,
                   None, 0, None)


def fit_weighted(v, I, phi, B_bar, cfg, spec, start, tol_um=IRLS_TOL_UM, max_iter=IRLS_MAX_ITER):
    """spec's centre-weighted fit by iteratively re-weighted least squares
    (module docstring) from start, the unweighted VariantFit of the same
    kernel and width mode. A start without a finite d, or a run that does
    not converge, ends it with status "failed" and irls_ok False."""
    v, I, phi, B_bar = block5._check_profile(v, I, phi, B_bar)
    spec = check_spec(spec)
    if spec.lam_um is None:
        raise ValueError("fit_weighted needs a spec with lam_um")
    lay, n_nodes = _Layout(spec, cfg), n_nodes_for(spec, cfg)
    p = lay.pack(start.d_um, start.mu_per_um, start.v0_um, start.fwhm_um, start.shape)
    runs_all, ok, t = [], False, 0
    omega = centre_weights(v, p[2], spec.lam_um)
    run, converged = None, False
    if np.all(np.isfinite(p)) and np.isfinite(start.d_um):
        for t in range(1, int(max_iter) + 1):
            omega = centre_weights(v, p[2], spec.lam_um)
            run = _runs(lay, v, I, phi, B_bar, cfg, n_nodes, [("irls", p)], omega)[0]
            runs_all.append(run)
            converged = bool(run.status > 0 and np.all(np.isfinite(run.x)) and np.isfinite(run.cost))
            if not converged:
                break
            done = abs(run.x[2] - p[2]) <= tol_um and abs(run.x[0] - p[0]) <= tol_um
            p = np.array(run.x, dtype=float)
            if done:
                ok = True
                break
    if run is None:                    # nothing to iterate from: report the start, failed
        run = _FakeRun(p)
    return _result(spec, lay, run, runs_all, "irls", converged, v, I, phi, B_bar, cfg, n_nodes, omega, t,
                   bool(ok and converged))


class _FakeRun:
    """A stand-in for a least_squares result when no run was made."""

    def __init__(self, x):
        self.x, self.status, self.nfev, self.message = np.asarray(x, dtype=float), 0, 0, "no finite start"


def _warm_starts(spec, fits_of):
    """The warm starts of an unweighted spec, by kernel and width mode (fits_of
    returns the unweighted VariantFit of a (kernel, width_fitted))."""
    kern = spec.kernel
    if spec.width_fitted:
        return [("fixed width", fits_of(kern, False))]
    if spec.shape_fitted:
        members = [("%s s=%g" % (spec.family, s), fits_of((spec.family, s, False), False))
                   for s in bk.FIXED_SHAPES.get(spec.family, ())]
        return members + [("gauss", fits_of(BASE_KERNEL, False))]
    return []


class PlaneFits:
    """The variants' fits on one profile (one plane's Block 5 inputs), made on
    demand and kept: the unweighted fit once per kernel and width mode, with
    the warm starts _warm_starts lists (made first, and kept too), and a
    weighted fit once per spec, from the unweighted fit of its kernel and width
    mode. log(spec, fit) is called after each fit made, when given."""

    def __init__(self, v, I, phi, B_bar, cfg, log=None):
        self.v, self.I, self.phi, self.B_bar, self.cfg, self.log = v, I, phi, B_bar, cfg, log
        self._unweighted, self._weighted = {}, {}

    def unweighted(self, kern, width_fitted):
        """The unweighted VariantFit of kernel kern (family, shape, shape_fitted) at the width mode."""
        k = (tuple(kern), bool(width_fitted))
        if k not in self._unweighted:
            spec = check_spec(VariantSpec(kern[0], kern[1], kern[2], bool(width_fitted), None))
            warm = _warm_starts(spec, self.unweighted)
            self._unweighted[k] = fit_unweighted(self.v, self.I, self.phi, self.B_bar, self.cfg, spec, warm)
            if self.log is not None:
                self.log(spec, self._unweighted[k])
        return self._unweighted[k]

    def fit(self, spec):
        """spec's VariantFit (unweighted, or re-weighted from the unweighted fit of its kernel and width mode)."""
        spec = check_spec(spec)
        u = self.unweighted(spec.kernel, spec.width_fitted)
        if spec.lam_um is None:
            return u
        if spec not in self._weighted:
            self._weighted[spec] = fit_weighted(self.v, self.I, self.phi, self.B_bar, self.cfg, spec, u)
            if self.log is not None:
                self.log(spec, self._weighted[spec])
        return self._weighted[spec]

    @property
    def base(self):
        """The unweighted fixed-width Gaussian: Block 5's fit."""
        return self.unweighted(BASE_KERNEL, False)


def fit_grid(v, I, phi, B_bar, cfg, specs=None, log=None):
    """Every variant of specs (default variant_grid()) on one profile; returns
    (fits in specs' order, base), base the unweighted fixed-width Gaussian
    fit (computed whether or not specs holds it). An unweighted fit is made
    once per kernel and width mode and seeds the weighted ones; log(spec, fit)
    is called after each fit when given (PlaneFits)."""
    specs = variant_grid() if specs is None else [check_spec(s) for s in specs]
    pf = PlaneFits(v, I, phi, B_bar, cfg, log)
    base = pf.base
    return [pf.fit(spec) for spec in specs], base


def variant_curve(fit, v, phi, B_bar):
    """The fitted model of a VariantFit at any offsets v (um), grey levels."""
    spec = fit.spec
    g = bk.kernel_pdf(spec.family, fit.fwhm_um, fit.shape)
    return tube_model.model_profile_kernel(v, fit.d_um, fit.alpha_hat, fit.v0_um, g, float(B_bar), fit.n_nodes)


def variant_row(fit, base):
    """The scalar outcome of one variant as a flat dict (CSV and JSON)."""
    s = fit.spec
    ratio = fit.d_um / base.d_um if (np.isfinite(fit.d_um) and np.isfinite(base.d_um) and base.d_um > 0) else float("nan")
    return dict(variant=s.key, family=s.family, kernel=s.kernel_label, shape_mode="fitted" if s.shape_fitted else "fixed",
                width_mode=s.width_mode, weight=s.weight_label, lam_um=s.lam_um, d_um=fit.d_um,
                mu_per_um=fit.mu_per_um, v0_um=fit.v0_um, fwhm_um=fit.fwhm_um, shape=fit.shape,
                alpha_hat=fit.alpha_hat, status=fit.status, at_bound=";".join(fit.at_bound), rss_w_gl2=fit.rss_w_gl2,
                rss_gl2=fit.rss_gl2, n_irls=fit.n_irls, irls_ok=fit.irls_ok, nfev=fit.nfev, n_starts=fit.n_starts,
                best_start=fit.best_start, n_nodes=fit.n_nodes, d_base_um=base.d_um, ratio=ratio)
