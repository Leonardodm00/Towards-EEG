"""The fit at one plane, D-052.2 -- Block 12 in specs/SPEC.md.

A straight tube blurred by the skewed generalized-t kernel of
model/blur_kernels.py (D-052.1), with the kernel's width and its three shape
parameters fitted per node, and the residuals centre-weighted by iteratively
re-weighted least squares (IRLS) re-centred on the fit's own axis.

Inputs, for one node and one plane (Block 5's): the profile (v_n, I_n),
n = 1..N (um, grey levels), the tilt phi (rad, the pilot's) and the background
B_bar (grey levels, D-018.1), all held fixed. With theta = (d, mu, v0, w,
beta, xi, zeta) and alpha = mu d / cos(phi), for every v (um) the model is
(H4)

    I_model(v | theta) = B_bar [1 - (d/2) Int_{-pi/2}^{pi/2} (1 - exp(-alpha cos psi))
                                cos(psi) g_{w,kappa}(v - v0 - (d/2) sin psi) dpsi]

(tube_model.model_profile_kernel with g = blur_kernels.sgt_kernel_pdf(w, beta,
xi, zeta); alpha by tube_model.alpha_from_mu), and for weights omega the
objective is (H5)

    Q(theta | omega) = (1/2) sum_n omega_n [(I_model(v_n | theta) - I_n) / B_bar]^2,

the cost of scipy.optimize.least_squares with the residuals
sqrt(omega_n) (I_model(v_n | theta) - I_n) / B_bar. The search coordinates
are q = (d, mu, v0, w, log10 beta, xi, ln zeta) (PARAM_NAMES) in the box of
box(cfg). Every solve is trf with a 2-point Jacobian, x_scale "jac",
ftol = xtol = gtol = fit_tol and max_nfev = sgt_max_nfev (D-054 (iv)), from a
start clipped into the box shrunk by START_MARGIN.

Steps of fit_plane (D-052.2; the start rule is PROVISIONAL, the sandbox
check's of 2026-10-15, checks/estimator_2026-10-15/identifiability_sgt.py):
  1. the two Gaussian fits of fit_variants (D-046) on the same profile:
     Block 5's Gaussian at sigma_fit (gauss_fixed) and the Gaussian of fitted
     width (gauss_fitted, warm-started from gauss_fixed); both are kept in
     the result as labelled comparisons;
  2. the unweighted fit theta^(0) = argmin Q(theta | 1): one solve from
     gauss_fitted's (d, mu, v0, w) with each shape of sgt_start_shapes, then
     one from gauss_fixed's (d, mu, v0) at w = sgt_start_w_um with the shape
     sgt_base_start_shape; the converged solve (scipy status > 0, finite q
     and cost) of least cost is reported, else the finite one of least cost,
     the first in start order on ties;
  3. the IRLS from theta^(0): omega^(t)_n = exp(-|v_n - v0^(t)| / lambda),
     lambda = sgt_lambda_um, v0^(t) the centre of iterate t; one solve from
     q^(t) per iteration; it ends with irls_ok True when d and v0 both moved by
     at most sgt_irls_tol_um, and with irls_ok False when a solve did not
     converge or after sgt_irls_max_iter solves. The last solve is reported:
     the node's fit at the plane.

Status (D-053 (ii)), one of fit.FIT_STATUSES: "failed" when the fit's own
rule is not met (unweighted: no solve converged; IRLS: irls_ok False);
otherwise "at_bound" when d, mu, v0 or w lies within fit_at_bound_rel_tol x
its range of a bound (STATUS_PARAMS); otherwise "converged". at_bound names
every coordinate at a bound, the shapes included; a shape at its box edge
never changes the status (xi = 0 is the generalized-normal edge and beta = 1
the Laplace: members of the family, not failures).

Quadrature: N = max(fit_quad_min_nodes, ceil(fit_quad_nodes_per_sigma x d_hi
/ sigma_min)), d_hi = fit_d_bounds_um[1], sigma_min = sgt_w_bounds_um[0] /
(2 sqrt(2 ln 2)) the Gaussian SD at the narrowest width (848 at the
defaults). The cusped members (beta near 1) converge slowly under this rule
(8.3e-4 grey levels on a 1.5 um tube against a direct quadrature, the
2026-10-15 check); the rule is kept as decided (D-052.2).

Custom: the IRLS loop (no scipy routine re-centres its weights on the fit's
own axis); the solves, the kernel's special functions and the Gaussian start
fits are library or Block 3 / Block 11 code.

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
from . import fit_variants as fv

PARAM_NAMES = ("d", "mu", "v0", "w", "log10_beta", "xi", "ln_zeta")   # the search coordinates q
BOUND_NAMES = ("d", "mu", "v0", "w", "beta", "xi", "zeta")            # their names in at_bound (_lo / _hi)
STATUS_PARAMS = ("d", "mu", "v0", "w")      # a bound hit here makes the status at_bound (D-053 (ii))
WEIGHTINGS = ("irls", "none")               # the deliverable (D-052.2), then the labelled comparison
START_MARGIN = 1e-9      # starts are clipped into the box shrunk by this (in q), as the D-052 check's solves were


@dataclass(frozen=True)
class SGTFit:
    """One SGT fit at a plane (units in the field names; beta, xi, zeta, alpha_hat dimensionless)."""

    weighting: str                     # "irls" (the deliverable) or "none" (theta^(0), the comparison)
    d_um: float
    mu_per_um: float
    v0_um: float
    w_um: float                        # the kernel's FWHM
    beta: float
    xi: float
    zeta: float
    alpha_hat: float                   # mu d / cos(phi)
    phi_rad: float
    q: Tuple[float, ...]               # the 7 search coordinates as the solver returned them (PARAM_NAMES)
    status: str                        # converged / at_bound / failed (module docstring, D-053 (ii))
    at_bound: Tuple[str, ...]          # every coordinate at a bound, shapes included, e.g. ("xi_lo",)
    solve_ok: bool                     # unweighted: some solve converged; IRLS: every weighted solve converged
    irls_ok: Optional[bool]            # None unweighted; True when the IRLS rule was met
    n_irls: int                        # weighted solves made (0 unweighted)
    irls_path: Tuple[Tuple[float, float], ...]   # (d, v0) after each weighted solve
    cost: float                        # Q of the reported solve (for the IRLS at the weights it was solved with)
    rss_gl2: float                     # sum_n (I_model(v_n) - I_n)^2, grey levels squared, unweighted
    nfev: int                          # residual evaluations summed over the solves (scipy's count)
    n_starts: int
    best_start: str
    n_nodes: int
    message: str
    model_gl: np.ndarray = field(repr=False, compare=False)   # I_model(v_n), (N,)


@dataclass(frozen=True)
class PlaneSGT:
    """The fits of fit_plane at one plane: the deliverable and its labelled comparisons."""

    final: SGTFit                      # the IRLS fit: the node's fit at the plane (D-052.2)
    unweighted: SGTFit                 # theta^(0), the unweighted SGT (comparison, and the IRLS start)
    gauss_fitted: "fv.VariantFit"      # the Gaussian of fitted width (comparison, and the starts)
    gauss_fixed: "fv.VariantFit"       # Block 5's Gaussian at sigma_fit at this plane (comparison, and a start)


def box(cfg):
    """(lb, ub): the box of the search coordinates q (PARAM_NAMES order), float64 (7,) each."""
    (d_lo, d_hi), (mu_lo, mu_hi), (v_lo, v_hi) = cfg.fit_d_bounds_um, cfg.fit_mu_bounds_per_um, cfg.fit_v0_bounds_um
    (w_lo, w_hi), (b_lo, b_hi) = cfg.sgt_w_bounds_um, cfg.sgt_beta_bounds
    (x_lo, x_hi), (z_lo, z_hi) = cfg.sgt_xi_bounds, cfg.sgt_ln_zeta_bounds
    lb = np.array([d_lo, mu_lo, v_lo, w_lo, math.log10(b_lo), x_lo, z_lo], dtype=float)
    ub = np.array([d_hi, mu_hi, v_hi, w_hi, math.log10(b_hi), x_hi, z_hi], dtype=float)
    return lb, ub


def to_q(d, mu, v0, w, beta, xi, zeta):
    """The search coordinates q of theta = (d, mu, v0, w, beta, xi, zeta), float64 (7,)."""
    return np.array([d, mu, v0, w, math.log10(beta), xi, math.log(zeta)], dtype=float)


def from_q(q):
    """theta = (d, mu, v0, w, beta, xi, zeta) from the search coordinates q."""
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]), 10.0 ** float(q[4]), float(q[5]),
            math.exp(float(q[6])))


def n_nodes_for(cfg):
    """Gauss-Legendre nodes of the model quadrature (module docstring; 848 at the defaults)."""
    sigma_min = cfg.sgt_w_bounds_um[0] / bk.gauss_fwhm(1.0)
    return tube_model.n_quadrature_nodes(cfg.fit_d_bounds_um[1], sigma_min, cfg.fit_quad_min_nodes,
                                         cfg.fit_quad_nodes_per_sigma)


def sgt_model(v, q, phi, B_bar, n_nodes):
    """(H4) at the search coordinates q: I_model(v | theta) in grey levels, the shape of v."""
    d, mu, v0, w, beta, xi, zeta = from_q(q)
    g = bk.sgt_kernel_pdf(w, beta, xi, zeta)
    return tube_model.model_profile_kernel(v, d, tube_model.alpha_from_mu(mu, d, phi), v0, g, B_bar, n_nodes)


def objective(v, I, phi, B_bar, q, omega, n_nodes):
    """Q(theta | omega) of (H5) at q; omega None = all weights 1."""
    r = (sgt_model(v, q, phi, B_bar, n_nodes) - np.asarray(I, dtype=float)) / float(B_bar)
    w = 1.0 if omega is None else np.asarray(omega, dtype=float)
    return 0.5 * float(np.sum(w * r * r))


def _solve(v, I, phi, B_bar, cfg, q0, omega, n_nodes, lb, ub):
    sw = 1.0 if omega is None else np.sqrt(omega)

    def residuals(q):
        return sw * (sgt_model(v, q, phi, B_bar, n_nodes) - I) / B_bar
    q0 = np.clip(np.asarray(q0, dtype=float), lb + START_MARGIN, ub - START_MARGIN)
    return least_squares(residuals, q0, bounds=(lb, ub), method="trf", jac="2-point", x_scale="jac",
                         ftol=cfg.fit_tol, xtol=cfg.fit_tol, gtol=cfg.fit_tol, max_nfev=int(cfg.sgt_max_nfev))


def _solve_ok(run):
    return bool(run.status > 0 and np.all(np.isfinite(run.x)) and np.isfinite(run.cost))


def _bound_hits(q, lb, ub, rel_tol):
    out = []
    for name, x, lo, hi in zip(BOUND_NAMES, q, lb, ub):
        tol = rel_tol * (hi - lo)
        if x - lo <= tol:
            out.append(name + "_lo")
        if hi - x <= tol:
            out.append(name + "_hi")
    return tuple(out)


def _status(rule_met, at_bound):
    if not rule_met:
        return "failed"
    hit = {name.rsplit("_", 1)[0] for name in at_bound}
    return "at_bound" if hit & set(STATUS_PARAMS) else "converged"


def _nan_fit(weighting, phi, n_starts, nfev, n_nodes, message, v, irls_ok, n_irls=0, path=()):
    nan = float("nan")
    return SGTFit(weighting=weighting, d_um=nan, mu_per_um=nan, v0_um=nan, w_um=nan, beta=nan, xi=nan, zeta=nan,
                  alpha_hat=nan, phi_rad=float(phi), q=(nan,) * len(PARAM_NAMES), status="failed", at_bound=(),
                  solve_ok=False, irls_ok=irls_ok, n_irls=int(n_irls), irls_path=tuple(path), cost=nan, rss_gl2=nan,
                  nfev=int(nfev), n_starts=int(n_starts), best_start="", n_nodes=int(n_nodes), message=str(message),
                  model_gl=np.full(np.shape(v), nan))


def _make_fit(weighting, run, solve_ok, irls_ok, n_irls, path, label, n_starts, nfev, v, I, phi, B_bar, cfg, n_nodes,
              lb, ub):
    q = np.asarray(run.x, dtype=float)
    if not np.all(np.isfinite(q)):         # a solve that returned no numbers: report it failed, with NaNs
        return _nan_fit(weighting, phi, n_starts, nfev, n_nodes, str(run.message), v,
                        None if weighting == "none" else False, n_irls, path)
    d, mu, v0, w, beta, xi, zeta = from_q(q)
    at_bound = _bound_hits(q, lb, ub, cfg.fit_at_bound_rel_tol)
    rule_met = solve_ok if weighting == "none" else bool(irls_ok)
    model = sgt_model(v, q, phi, B_bar, n_nodes)
    return SGTFit(weighting=weighting, d_um=d, mu_per_um=mu, v0_um=v0, w_um=w, beta=beta, xi=xi, zeta=zeta,
                  alpha_hat=float(tube_model.alpha_from_mu(mu, d, phi)), phi_rad=float(phi),
                  q=tuple(float(x) for x in q), status=_status(rule_met, at_bound), at_bound=at_bound,
                  solve_ok=bool(solve_ok), irls_ok=irls_ok, n_irls=int(n_irls), irls_path=tuple(path),
                  cost=float(run.cost), rss_gl2=float(np.sum((model - I) ** 2)), nfev=int(nfev),
                  n_starts=int(n_starts), best_start=str(label), n_nodes=int(n_nodes), message=str(run.message),
                  model_gl=model)


def start_points(gauss_fitted, gauss_fixed, cfg):
    """[(label, q)]: gauss_fitted's (d, mu, v0, w) with each shape of
    sgt_start_shapes, then gauss_fixed's (d, mu, v0) at w = sgt_start_w_um
    with sgt_base_start_shape (D-052.2, PROVISIONAL). A start with a
    non-finite entry (a Gaussian fit without numbers) is left out."""
    out = []
    g = gauss_fitted
    for k, (beta, xi, zeta) in enumerate(cfg.sgt_start_shapes):
        out.append(("gauss_fitted:%d" % k, to_q(g.d_um, g.mu_per_um, g.v0_um, g.fwhm_um, beta, xi, zeta)))
    b = gauss_fixed
    beta, xi, zeta = cfg.sgt_base_start_shape
    out.append(("gauss_fixed", to_q(b.d_um, b.mu_per_um, b.v0_um, cfg.sgt_start_w_um, beta, xi, zeta)))
    return [(label, q) for label, q in out if np.all(np.isfinite(q))]


def fit_unweighted(v, I, phi, B_bar, cfg, starts):
    """theta^(0): one unweighted solve per start (label, q); the converged
    solve of least cost, else the finite one of least cost, the first in
    start order on ties. Returns SGTFit (weighting "none")."""
    v, I, phi, B_bar = block5._check_profile(v, I, phi, B_bar)
    lb, ub = box(cfg)
    n_nodes = n_nodes_for(cfg)
    runs = [_solve(v, I, phi, B_bar, cfg, q0, None, n_nodes, lb, ub) for _, q0 in starts]
    nfev = sum(int(r.nfev) for r in runs)
    finite = [k for k, r in enumerate(runs) if np.all(np.isfinite(r.x)) and np.isfinite(r.cost)]
    converged = [k for k in finite if runs[k].status > 0]
    pool = converged if converged else finite
    if not pool:
        return _nan_fit("none", phi, len(runs), nfev, n_nodes, "no finite start or solve", v, None)
    best = min(pool, key=lambda k: (runs[k].cost, k))
    return _make_fit("none", runs[best], bool(converged), None, 0, (), starts[best][0], len(runs), nfev, v, I, phi,
                     B_bar, cfg, n_nodes, lb, ub)


def fit_irls(v, I, phi, B_bar, cfg, start):
    """The centre-weighted fit by IRLS (module docstring, step 3) from start,
    the unweighted SGTFit of the same profile. A start without finite
    coordinates gives a failed fit with NaNs. Returns SGTFit (weighting "irls")."""
    v, I, phi, B_bar = block5._check_profile(v, I, phi, B_bar)
    lb, ub = box(cfg)
    n_nodes = n_nodes_for(cfg)
    q = np.array(start.q, dtype=float)
    if not np.all(np.isfinite(q)):
        return _nan_fit("irls", phi, 0, 0, n_nodes, "no finite start", v, False)
    runs, path, ok, solves_ok = [], [], False, True
    for _ in range(int(cfg.sgt_irls_max_iter)):
        omega = fv.centre_weights(v, q[2], cfg.sgt_lambda_um)
        run = _solve(v, I, phi, B_bar, cfg, q, omega, n_nodes, lb, ub)
        runs.append(run)
        path.append((float(run.x[0]), float(run.x[2])))
        if not _solve_ok(run):
            solves_ok = False
            break
        done = abs(run.x[0] - q[0]) <= cfg.sgt_irls_tol_um and abs(run.x[2] - q[2]) <= cfg.sgt_irls_tol_um
        q = np.array(run.x, dtype=float)
        if done:
            ok = True
            break
    return _make_fit("irls", runs[-1], solves_ok, bool(ok and solves_ok), len(runs), path, "irls", 1,
                     sum(int(r.nfev) for r in runs), v, I, phi, B_bar, cfg, n_nodes, lb, ub)


def fit_plane(v, I, phi, B_bar, cfg):
    """The D-052.2 fit at one plane (module docstring, steps 1-3) on the
    profile (v, I) with the tilt phi and the background B_bar; cfg a
    MeasureConfig. Returns PlaneSGT, whose final is the node's fit."""
    v, I, phi, B_bar = block5._check_profile(v, I, phi, B_bar)
    pf = fv.PlaneFits(v, I, phi, B_bar, cfg)
    gauss_fixed = pf.base
    gauss_fitted = pf.unweighted(fv.BASE_KERNEL, True)
    unweighted = fit_unweighted(v, I, phi, B_bar, cfg, start_points(gauss_fitted, gauss_fixed, cfg))
    final = fit_irls(v, I, phi, B_bar, cfg, unweighted)
    return PlaneSGT(final=final, unweighted=unweighted, gauss_fitted=gauss_fitted, gauss_fixed=gauss_fixed)


def fit_curve(fit, v, B_bar):
    """The fitted model of an SGTFit at any offsets v (um), grey levels (NaN for a fit without numbers)."""
    if not np.all(np.isfinite(fit.q)):
        return np.full(np.shape(v), float("nan"))
    return sgt_model(v, np.asarray(fit.q, dtype=float), fit.phi_rad, float(B_bar), fit.n_nodes)
