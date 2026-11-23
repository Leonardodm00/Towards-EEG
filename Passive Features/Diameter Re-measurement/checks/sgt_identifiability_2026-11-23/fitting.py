#!/usr/bin/env python3
"""Standalone check (NOT pipeline code), 2026-11-23 (D-221, D-222): the per-node joint fit across planes on the
1-D model of sgt_line_model.py, as Block 30 fits it, with the kernel's shape optionally free.

    min_q  sum_{k in K} sum_n [I_k(v_n) - B_k(q) tau_k(v_n; q)]^2 / sigma^2,
    B_k(q) = sum_n I_k tau_k / sum_n tau_k^2      (one scale per plane solved exactly: variable projection)

q = (d, mu, v0, z_ax) and, when the shape is free, Block 12's search coordinates (log10 beta, xi, ln zeta);
scipy.optimize.least_squares(method="trf", jac="2-point", x_scale="jac", max_nfev = inference.max_nfev),
scipy's default tolerances (Block 30's call). Bounds: Block 30's InferenceConfig d_bounds_um, mu_bounds_per_um,
v0_bounds_um and z_bounds_um about the start's depth; Block 12's MeasureConfig sgt_beta_bounds (in log10),
sgt_xi_bounds and sgt_ln_zeta_bounds. The start is clipped into the box shrunk by 1e-9 (Block 12's
START_MARGIN). The covariance is Block 30's: (J^T J)^{-1} with J the solver's Jacobian of the projected
residual (Golub-Pereyra) at the solution, rows and columns of a parameter at a bound set to NaN; mapped to the
natural parameters (beta = 10^q, zeta = e^q) by the chain rule.

Pure ASCII, LF.
"""
from __future__ import annotations

import math
import time

import numpy as np
from scipy.optimize import least_squares

import sgt_line_model as M

START_MARGIN = 1e-9
SHAPE_Q = {"beta": "log10_beta", "xi": "xi", "zeta": "ln_zeta"}


def box(spec, setup, z_centre):
    """(lb, ub) in the search coordinates of spec."""
    icfg, mcfg = setup.cfg.inference, setup.cfg.measure
    lo = [icfg.d_bounds_um[0], icfg.mu_bounds_per_um[0], icfg.v0_bounds_um[0], z_centre + icfg.z_bounds_um[0]]
    hi = [icfg.d_bounds_um[1], icfg.mu_bounds_per_um[1], icfg.v0_bounds_um[1], z_centre + icfg.z_bounds_um[1]]
    for name in spec.free:
        if name == "beta":
            lo.append(math.log10(mcfg.sgt_beta_bounds[0]))
            hi.append(math.log10(mcfg.sgt_beta_bounds[1]))
        elif name == "xi":
            lo.append(mcfg.sgt_xi_bounds[0])
            hi.append(mcfg.sgt_xi_bounds[1])
        elif name == "zeta":
            lo.append(mcfg.sgt_ln_zeta_bounds[0])
            hi.append(mcfg.sgt_ln_zeta_bounds[1])
    return np.array(lo, float), np.array(hi, float)


def to_q(phi, spec):
    q = list(phi[:4])
    for name, val in zip(spec.free, phi[4:]):
        q.append(math.log10(val) if name == "beta" else (math.log(val) if name == "zeta" else val))
    return np.array(q, float)


def from_q(q, spec):
    phi = list(q[:4])
    for name, val in zip(spec.free, q[4:]):
        phi.append(10.0 ** val if name == "beta" else (math.exp(val) if name == "zeta" else val))
    return np.array(phi, float)


def dphi_dq(q, spec):
    """Diagonal of d phi / d q."""
    g = [1.0, 1.0, 1.0, 1.0]
    for name, val in zip(spec.free, q[4:]):
        g.append(10.0 ** val * math.log(10.0) if name == "beta" else (math.exp(val) if name == "zeta" else 1.0))
    return np.array(g, float)


def fit(I_obs, spec, setup, phi0, planes, max_nfev=None):
    """Fit spec to I_obs (n_k, n_v) gl over the planes kept (bool (n_k,)). phi0: the start in natural
    parameters (d, mu, v0, z_ax[, shape]); the z_ax box is about phi0's z_ax. Returns a dict."""
    keep = np.asarray(planes, bool)
    I = np.asarray(I_obs, float)[keep]
    s = setup.sigma_gl
    lb, ub = box(spec, setup, float(phi0[3]))
    q0 = to_q(np.asarray(phi0, float), spec)
    margin = START_MARGIN * (ub - lb)
    q0 = np.clip(q0, lb + margin, ub - margin)
    nonfinite = {"n": 0}

    def residuals(q):
        phi = from_q(q, spec)
        try:
            t = M.tau_phi(phi, spec, setup)[keep]
        except ValueError:
            nonfinite["n"] += 1
            return np.full(I.size, 1e6)
        Bk = M.project_background(I, t)
        return ((I - Bk[:, None] * t) / s).ravel()

    t0 = time.perf_counter()
    mx = int(setup.cfg.inference.max_nfev) if max_nfev is None else int(max_nfev)
    res = least_squares(residuals, q0, bounds=(lb, ub), method="trf", jac="2-point", x_scale="jac", max_nfev=mx)
    secs = time.perf_counter() - t0
    q = np.array(res.x, float)
    phi = from_q(q, spec)
    active = np.asarray(res.active_mask) != 0
    names = spec.names
    at_bound = tuple("%s_%s" % (names[i], "lo" if res.active_mask[i] < 0 else "hi")
                     for i in range(q.size) if active[i])
    J = np.asarray(res.jac, float)
    cov_q = np.full((q.size, q.size), np.nan)
    if np.all(np.isfinite(J)) and np.linalg.matrix_rank(J) == J.shape[1]:
        try:
            cov_q = np.linalg.inv(J.T @ J)
        except np.linalg.LinAlgError:
            pass
    cov_q[active, :] = np.nan
    cov_q[:, active] = np.nan
    g = dphi_dq(q, spec)
    cov = cov_q * np.outer(g, g)
    with np.errstate(invalid="ignore"):
        se = np.sqrt(np.diag(cov))
    # the conditional SE of d: the parameters at a bound held fixed (J^T J over the free columns)
    se_d_cond = float("nan")
    free_cols = np.flatnonzero(~active)
    if np.all(np.isfinite(J)) and 0 in free_cols:
        Jc = J[:, free_cols]
        if np.linalg.matrix_rank(Jc) == Jc.shape[1]:
            try:
                se_d_cond = float(math.sqrt(np.linalg.inv(Jc.T @ Jc)[0, 0]))
            except np.linalg.LinAlgError:
                pass
    t = M.tau_phi(phi, spec, setup)[keep]
    Bk = M.project_background(I, t)
    chi2 = float(np.sum(((I - Bk[:, None] * t) / s) ** 2))
    status = "failed" if res.status <= 0 else ("at_bound" if at_bound else "converged")
    return dict(phi=phi, q=q, se=se, se_d_cond=se_d_cond, cov=cov, B=Bk, chi2=chi2, n_res=int(I.size), status=status,
                scipy_status=int(res.status), at_bound=at_bound, nfev=int(res.nfev), njev=int(res.njev or 0),
                seconds=secs, names=names, nonfinite=nonfinite["n"])


def linearised_shift(y_true, spec, setup, phi_true, B, planes, steps=None):
    """The first Gauss-Newton step of spec from phi_true on noise-free data y_true (n_k, n_v) gl, with the
    planes' backgrounds as free linear parameters: Delta = (J^T J)^{-1} J^T (y_true - B tau(phi_true)).
    Returns (Delta_phi (p,), Delta_B (n_k kept,))."""
    keep = np.asarray(planes, bool)
    t0, Jt = M.jacobian_tau(phi_true, spec, setup, steps=steps, lower=M.lower_bounds(spec))
    t0, Jt = t0[keep], Jt[keep]
    n_k, n_v, p = Jt.shape
    Bk = np.full(n_k, float(B))
    Jphi = (Bk[:, None, None] * Jt).reshape(n_k * n_v, p)
    JB = np.zeros((n_k * n_v, n_k))
    for k in range(n_k):
        JB[k * n_v:(k + 1) * n_v, k] = t0[k]
    Jf = np.hstack([Jphi, JB])
    r = (np.asarray(y_true, float)[keep] - Bk[:, None] * t0).ravel()
    delta, *_ = np.linalg.lstsq(Jf, r, rcond=None)
    return delta[:p], delta[p:]
