#!/usr/bin/env python3
"""Checks behind the explanation of Cell 4m's fit objective (2026-10-15).

A standalone check, not pipeline code: it imports allen_diameter (commit
9e3296c, branch sci/diameter-pipeline) from --repo and tests five claims of
the explanation on simulated profiles whose truth is chosen here (CONFIG).

Truth (D-046's synthetic thin tube, reused): d 0.6 um, mu 1.0 /um, v0 0.1 um,
phi 0.2 rad, B_bar 180 grey levels, a Gaussian blur of FWHM 0.5 um (about
twice Block 5's 0.233 um), sampled at Block 5's 53 offsets (+-3 um, step
0.1144 um), plus Gaussian noise of SD 2 grey levels (seed in CONFIG).

C1 cost      scipy's cost is 0.5 sum_n omega_n r_n^2 with
             r_n = (I_model(v_n) - I_n) / B_bar, and the CSV's rss_w_gl2 is
             2 B_bar^2 x that sum's argument (unweighted exactly; weighted with
             the last iteration's weights)
C2 balance   the objective with the weights moving with v0,
             Phi(theta) = 0.5 sum_n omega_n(v0) r_n(theta)^2, has
             dPhi/dv0 = (1 / (2 lam)) sum_n omega_n sgn(v_n - v0) r_n^2
                        + dF(theta; omega)/dv0   (weights frozen in the second term);
             at an interior IRLS fixed point theta* the second term is 0.
             Cases: (A) noise-free, model right: both terms 0, and minimising
             Phi jointly stays at theta*; (B, recorded) a darker neighbour 1.3 um
             away: the fit merges both dips and stops at the v0 bound, outside
             the hypothesis; (B1) a skewed true blur, noise-free, and (B2) a
             fainter neighbour 1.8 um away, with noise: interior fixed points
             with the first term not 0, and the joint minimum moves (B1, B2
             added after B's run, their criteria fixed before they ran)
C3 valley    the unweighted objective profiled along d (minimised over mu, v0
             and, when fitted, w): sharp at the fixed width, flat at a fitted
             width; w-hat along the valley
C4 moments   faint limit: the dip's variance is sigma_g^2 + d^2 / 16; at the
             fixed width the leading-order d-hat is
             sqrt(d^2 + 16 (sigma_true^2 - sigma_fit^2))
C5 spread    Monte Carlo over noise draws: d-hat of Block 5's fit, of the
             Gaussian at a fitted width, and of the same with lambda 0.25 um

The pass criteria below (TOL) were fixed before the first run.

Outcome of the 2026-10-15 run (deterministic; objective_check.json has every number):
    C1 PASS (cost to 1e-16 relative; rss_w_gl2 exact unweighted, 6e-8 weighted).
    C2 FAIL as pre-registered, and the claim was corrected: the identity holds in
       all seven cases; at the interior fixed points (B2, W25, W50, B2_25) the
       first term is not 0 (2e-5 to 2.3e-3) and the frozen term is <= 2e-10,
       but the joint minimum moved by at most 0.0011 um in v0 and 0.0055 um in
       d (criterion 0.005 um not met), and the global probe found a slide only
       in B (the merged neighbour at the v0 bound); B1 ended with mu at its
       bound (hypothesis not met).
    C3 PASS (Delta chi^2 <= 1 interval 4.7 x wider at the fitted width).
    C4 FAIL as pre-registered, and the claim was corrected: the faint-limit
       variance identity holds (<= 7e-5 relative), but the leading-order
       fixed-width d-hat overstates Block 5's by 9-14 %.
    C5 PASS (SD of d-hat 0.206 um at the fitted width against 0.021 um).

Run (several minutes on one CPU):
    python check_objective.py --repo "<repo>/Passive Features/Diameter Re-measurement" --out <dir>
then the figures:
    python fig_objective.py --res <dir>/objective_check.npz --out <dir>

Pure ASCII.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
from scipy import integrate
from scipy.optimize import least_squares

CONFIG = dict(
    seed=20261015,
    d_um=0.6, mu_per_um=1.0, v0_um=0.1, phi_rad=0.2, B_gl=180.0,   # D-046's thin tube
    true_fwhm_um=0.5, noise_sd_gl=2.0,
    sim_nodes=2000,                                   # quadrature of the simulated truth
    lambdas_um=(0.25, 0.5, 1.0, 2.0),                 # D-046's scan (for the weights panel)
    lam_balance_um=0.5,                               # C2
    neighbour=dict(d_um=0.8, mu_per_um=1.5, v0_um=1.4),   # C2 (B): outside the v0 bounds +-1 um
    neighbour_far=dict(d_um=0.5, mu_per_um=0.5, v0_um=1.9),   # C2 (B2), added 2026-10-15 after case B
    skew_kernel=("skewnorm", 4.0),                    # C2 (B1), added 2026-10-15 after case B: the true blur's shape
    irls_tight_tol_um=1e-9, irls_tight_max_iter=300,  # C2: the exact fixed point
    d_grid_um=(0.10, 1.30, 0.025),                    # C3: start, stop (inclusive), step
    w_starts_um=(0.2331, 0.6, 1.4),                   # C3 starts of w (mu from the depth rule)
    mu_start_factors=(0.5, 2.0),                      # C3 extra starts of mu (x the depth rule; w 0.6 um when fitted)
    lam_valley_um=0.25,                               # C3, C5
    faint_alpha=1e-3,                                 # C4
    moment_cases=((0.6, 0.5), (0.3, 0.3), (1.0, 0.2331)),   # C4: (d um, FWHM um)
    floor_d_um=(0.1, 0.3, 0.6, 1.0),                  # C4: fixed-width d-hat against the leading order
    n_mc=40,                                          # C5
)
# C2's cases. A and B were in the first run; B ended at a bound (the unweighted fit merged the
# tube and the neighbour, v0 at +1 um), where the claim's hypothesis (an interior fixed point)
# fails, so it is kept as a record and B1, B2 were added, with the criteria below fixed before
# they were run (2026-10-15).
CASES_C2 = dict(A=dict(kind="balanced", noise=False, neighbour=None, kernel=None),
                B=dict(kind="recorded", noise=True, neighbour="neighbour", kernel=None, seed_tail=999),
                B1=dict(kind="unbalanced", noise=False, neighbour=None, kernel="skew_kernel"),
                B2=dict(kind="unbalanced", noise=True, neighbour="neighbour_far", kernel=None, seed_tail=998),
                # added 2026-10-15 after B1, B2, criterion fixed before they ran: the global joint minimum
                # of Phi "slides onto the background" (D-046's statement) when it lies >= 0.2 um from v0*
                # with a lower Phi; W = the worked profile (the C1, C3 draw)
                W25=dict(kind="probe", noise=True, neighbour=None, kernel=None, worked=True, lam=0.25),
                W50=dict(kind="probe", noise=True, neighbour=None, kernel=None, worked=True, lam=0.5),
                B2_25=dict(kind="probe", noise=True, neighbour="neighbour_far", kernel=None, seed_tail=998, lam=0.25))
TOL = dict(
    cost_rel=1e-12,          # C1: float64 roundoff of a 53-term sum
    rss_rel_unweighted=1e-12,
    rss_rel_weighted=1e-3,   # C1: the weights' centre moved by <= 1e-4 um in the last IRLS step
    ref_um=1e-6,             # C1: SPEC's baseline equality
    deriv_rel=1e-4,          # C2: central difference with h = 1e-6 um
    balanced_abs=1e-9,       # C2 (A): |dPhi/dv0| at a noise-free fixed point (residuals ~ solver tolerance)
    joint_still_um=1e-3,     # C2 (A): the joint minimum stays
    joint_moves_um=5e-3,     # C2 (B): the joint minimum moves by at least this in v0
    slide_um=0.2,            # C2 probes: the global joint minimum this far from v0* = a slide
    valley_ratio=3.0,        # C3: the Delta chi^2 <= 1 interval at fitted width >= 3 x the fixed width's
    fixed_bias_um=0.7,       # C3, C5: Block 5's d-hat above this (the overestimate D-046 saw)
    moment_rel=5e-3,         # C4: faint-limit variance identity
    floor_rel=0.05,          # C4: leading-order fixed-width d-hat within 5 %
    spread_ratio=2.0,        # C5: SD of d-hat at fitted width >= 2 x Block 5's
)


# ---------------------------------------------------------------- loading ---

def load(repo):
    src = os.path.join(repo, "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from allen_diameter.config import default_config
    from allen_diameter.analysis import fit as block5
    from allen_diameter.analysis import fit_variants as FV
    from allen_diameter.analysis import profiles
    from allen_diameter.model import blur_kernels as BK
    from allen_diameter.model import tube_model as TM
    return SimpleNamespace(cfg=default_config().measure, block5=block5, FV=FV, profiles=profiles, BK=BK, TM=TM)


# ------------------------------------------------------------- simulation ---

def simulate(P, c, rng=None, noise=True, neighbour=False, mu=None, d=None, fwhm=None, kernel=None):
    """Block 5's offsets and a profile of the chosen truth (grey levels). neighbour: False, True
    (CONFIG["neighbour"]) or a CONFIG key; kernel: None (Gaussian) or (family, shape)."""
    v = P.profiles.profile_offsets(P.cfg)
    d = c["d_um"] if d is None else d
    mu = c["mu_per_um"] if mu is None else mu
    fam, shp = ("gauss", None) if kernel is None else kernel
    g = P.BK.kernel_pdf(fam, c["true_fwhm_um"] if fwhm is None else fwhm, shp)
    phi, B = c["phi_rad"], c["B_gl"]
    I = P.TM.model_profile_kernel(v, d, mu * d / math.cos(phi), c["v0_um"], g, B, c["sim_nodes"])
    if neighbour:
        nb = c["neighbour"] if neighbour is True else c[neighbour]
        I = I + (P.TM.model_profile_kernel(v, nb["d_um"], nb["mu_per_um"] * nb["d_um"] / math.cos(phi),
                                           nb["v0_um"], g, B, c["sim_nodes"]) - B)
    if noise:
        I = I + rng.normal(0.0, c["noise_sd_gl"], v.size)
    return v, I


def spec(P, width_fitted, lam=None):
    return P.FV.VariantSpec("gauss", None, False, bool(width_fitted), lam)


# ----------------------------------------------------------------- checks ---

def check_cost(P, c, v, I):
    """C1."""
    phi, B, cfg, FV = c["phi_rad"], c["B_gl"], P.cfg, P.FV
    pf = FV.PlaneFits(v, I, phi, B, cfg)
    b5 = P.block5.fit_profile(v, I, phi, B, cfg)
    base = pf.base
    rows = []
    for lam in (None, c["lam_balance_um"]):
        s = spec(P, True, lam)
        f = pf.fit(s)
        lay, nn = FV._Layout(s, cfg), FV.n_nodes_for(s, cfg)
        p = lay.pack(f.d_um, f.mu_per_um, f.v0_um, f.fwhm_um, f.shape)
        om = None if lam is None else FV.centre_weights(v, f.v0_um, lam)
        run = FV._runs(lay, v, I, phi, B, cfg, nn, [("at", p)], om)[0]
        r = (FV._model(lay, run.x, v, phi, B, nn) - I) / B
        w = np.ones_like(v) if om is None else om
        F_hand = 0.5 * float(np.sum(w * r * r))
        rss_hand = float(np.sum(w * (f.model_gl - I) ** 2))
        rows.append(dict(lam_um=lam, n_params=int(lay.lb.size), n_nodes=int(nn), n_starts=int(f.n_starts),
                         best_start=f.best_start, d_um=f.d_um, fwhm_um=f.fwhm_um, v0_um=f.v0_um,
                         status=f.status, n_irls=f.n_irls, irls_ok=f.irls_ok,
                         cost_scipy=float(run.cost), cost_hand=F_hand,
                         cost_rel=abs(float(run.cost) - F_hand) / F_hand,
                         rss_w_gl2=f.rss_w_gl2, rss_hand_gl2=rss_hand,
                         rss_rel=abs(f.rss_w_gl2 - rss_hand) / rss_hand))
    ok = (rows[0]["cost_rel"] <= TOL["cost_rel"] and rows[1]["cost_rel"] <= TOL["cost_rel"]
          and rows[0]["rss_rel"] <= TOL["rss_rel_unweighted"] and rows[1]["rss_rel"] <= TOL["rss_rel_weighted"]
          and abs(base.d_um - b5.d_hat_um) <= TOL["ref_um"])
    return dict(rows=rows, ref_d_um=base.d_um, block5_d_um=b5.d_hat_um, ref_diff_um=abs(base.d_um - b5.d_hat_um),
                block5_status=b5.status, block5_n_nodes=b5.n_nodes, fixed_fwhm_um=FV.fixed_fwhm_um(P.cfg),
                ok=bool(ok))


def _phi_parts(P, lay, p, v, I, phi, B, nn, lam):
    d, mu, v0, w, s = lay.unpack(p)
    r = (P.FV._model(lay, p, v, phi, B, nn) - I) / B
    om = np.exp(-np.abs(v - v0) / lam)
    return 0.5 * float(np.sum(om * r * r)), r, om, v0


def global_joint(P, c, lay, nn, v, I, lam, p_star):
    """The joint minimum of Phi from starts across the v0 box: v0 in {-0.9, -0.5, v0*, 0.5, 0.9} um,
    each with theta*'s (d, mu, w) and with a vanishing tube (mu = 0). Returns (Phi, theta) of the best."""
    phi, B, FV = c["phi_rad"], c["B_gl"], P.FV

    def res(p):
        v0_ = lay.unpack(p)[2]
        return np.exp(-np.abs(v - v0_) / (2.0 * lam)) * (FV._model(lay, p, v, phi, B, nn) - I) / B
    best = None
    for v0s in (-0.9, -0.5, float(p_star[2]), 0.5, 0.9):
        for mu0 in (float(p_star[1]), 0.0):
            q = p_star.copy()
            q[1], q[2] = mu0, v0s
            q = np.clip(q, lay.lb, lay.ub)
            run = least_squares(res, q, bounds=(lay.lb, lay.ub), method="trf", jac="2-point", x_scale="jac",
                                ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=3000)
            val = _phi_parts(P, lay, run.x, v, I, phi, B, nn, lam)[0]
            if best is None or val < best[0]:
                best = (val, run.x.copy())
    return best


def check_balance(P, c, v, I, case):
    """C2 for one profile (revised 2026-10-15 after case B, see CASES_C2). The IRLS
    fixed point is computed twice: with the pipeline's tolerance (1e-4 um, 30
    iterations) and tightly (1e-9 um, 300 iterations; the property tested is the
    exact fixed point's). At the tight fixed point theta*: the total derivative
    of Phi in v0 (central difference), the weight term
    (1 / (2 lam)) sum_n omega_n sgn(v_n - v0*) r_n^2 and the frozen-weight term
    dF(theta; omega*)/dv0 (central difference); the identity total = weight +
    frozen holds everywhere, and at an interior fixed point frozen = 0. Then Phi
    is minimised jointly from theta*."""
    phi, B, cfg, FV = c["phi_rad"], c["B_gl"], P.cfg, P.FV
    lam = CASES_C2[case].get("lam", c["lam_balance_um"])
    s = spec(P, True, lam)
    pf = FV.PlaneFits(v, I, phi, B, cfg)
    f_pipe = pf.fit(s)
    u = pf.fit(spec(P, True, None))
    f = FV.fit_weighted(v, I, phi, B, cfg, s, u, tol_um=c["irls_tight_tol_um"], max_iter=c["irls_tight_max_iter"])
    lay, nn = FV._Layout(s, cfg), FV.n_nodes_for(s, cfg)
    p0 = lay.pack(f.d_um, f.mu_per_um, f.v0_um, f.fwhm_um, f.shape)
    Phi0, r, om, v0 = _phi_parts(P, lay, p0, v, I, phi, B, nn, lam)
    weight_term = float(np.sum(om * np.sign(v - v0) * r * r)) / (2.0 * lam)
    h = 1e-6
    pp, pm = p0.copy(), p0.copy()
    pp[2] += h
    pm[2] -= h
    total = (_phi_parts(P, lay, pp, v, I, phi, B, nn, lam)[0] - _phi_parts(P, lay, pm, v, I, phi, B, nn, lam)[0]) / (2 * h)

    def frozen(p):
        rr = (FV._model(lay, p, v, phi, B, nn) - I) / B
        return 0.5 * float(np.sum(om * rr * rr))
    frozen_term = (frozen(pp) - frozen(pm)) / (2 * h)

    def res(p):
        v0_ = lay.unpack(p)[2]
        return np.exp(-np.abs(v - v0_) / (2.0 * lam)) * (FV._model(lay, p, v, phi, B, nn) - I) / B
    run = least_squares(res, p0, bounds=(lay.lb, lay.ub), method="trf", jac="2-point", x_scale="jac",
                        ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=3000)
    dj, muj, v0j, wj, _ = lay.unpack(run.x)
    PhiJ = _phi_parts(P, lay, run.x, v, I, phi, B, nn, lam)[0]
    PhiG, xg = global_joint(P, c, lay, nn, v, I, lam, p0)
    dg, mug, v0g, wg, _ = lay.unpack(xg)
    slide = bool(abs(v0g - f.v0_um) >= TOL["slide_um"] and PhiG < Phi0 - 1e-12)
    scale = max(abs(total), abs(weight_term), abs(frozen_term))
    decomposition_ok = abs(total - (weight_term + frozen_term)) <= TOL["deriv_rel"] * scale + TOL["balanced_abs"]
    interior = bool(f.status == "converged" and f.irls_ok)
    out = dict(case=case, lam_um=lam, unweighted_d_um=u.d_um, unweighted_v0_um=u.v0_um, unweighted_status=u.status,
               pipeline_irls_d_um=f_pipe.d_um, pipeline_irls_v0_um=f_pipe.v0_um, pipeline_n_irls=f_pipe.n_irls,
               irls_d_um=f.d_um, irls_v0_um=f.v0_um, irls_fwhm_um=f.fwhm_um, irls_mu=f.mu_per_um,
               n_irls=f.n_irls, irls_ok=f.irls_ok, status=f.status, at_bound=list(f.at_bound), interior=interior,
               dPhi_dv0_total=total, weight_term=weight_term, frozen_term=frozen_term,
               decomposition_ok=bool(decomposition_ok),
               Phi_fixed_point=Phi0, joint_d_um=dj, joint_v0_um=v0j, joint_fwhm_um=wj, joint_mu=muj,
               Phi_joint=PhiJ, joint_status=int(run.status), joint_nfev=int(run.nfev),
               dv0_joint_um=v0j - f.v0_um, dd_joint_um=dj - f.d_um,
               Phi_global=PhiG, global_d_um=dg, global_v0_um=v0g, global_fwhm_um=wg, global_mu=mug, slide=slide)
    kind = CASES_C2[case]["kind"]
    if kind == "balanced":
        out["ok"] = bool(decomposition_ok and abs(weight_term) <= TOL["balanced_abs"]
                         and abs(v0j - f.v0_um) <= TOL["joint_still_um"] and abs(dj - f.d_um) <= TOL["joint_still_um"])
    elif kind == "unbalanced":
        out["ok"] = bool(interior and decomposition_ok
                         and abs(frozen_term) <= 1e-3 * abs(weight_term) + TOL["balanced_abs"]
                         and abs(weight_term) > TOL["balanced_abs"]
                         and PhiJ < Phi0 and abs(v0j - f.v0_um) >= TOL["joint_moves_um"])
    else:                       # "recorded" / "probe": only the identity is a pass criterion; slide is reported
        out["ok"] = bool(decomposition_ok)
    out["kind"] = kind
    return out


def profile_along_d(P, c, v, I, width_fitted, omega=None, d_grid=None):
    """C3: min over (mu, v0[, w]) of 0.5 sum omega_n r_n^2 at each fixed d (Gaussian kernel)."""
    phi, B, cfg, FV, BK, TM = c["phi_rad"], c["B_gl"], P.cfg, P.FV, P.BK, P.TM
    w_fixed = FV.fixed_fwhm_um(cfg)
    nn = FV.n_nodes_for(spec(P, width_fitted), cfg)
    (mu_lo, mu_hi), (v_lo, v_hi) = cfg.fit_mu_bounds_per_um, cfg.fit_v0_bounds_um
    lb = [mu_lo, v_lo] + ([FV.FITTED_FWHM_BOUNDS_UM[0]] if width_fitted else [])
    ub = [mu_hi, v_hi] + ([FV.FITTED_FWHM_BOUNDS_UM[1]] if width_fitted else [])
    sw = 1.0 if omega is None else np.sqrt(omega)
    win = np.flatnonzero((v >= v_lo) & (v <= v_hi))
    n_star = int(win[np.argmin(I[win])])
    alpha0 = -math.log(min(1.0, max(I[n_star] / B, np.finfo(float).tiny)))
    rows, prev = [], None
    for d in d_grid:
        def res(q, d=d):
            w = q[2] if width_fitted else w_fixed
            al = TM.alpha_from_mu(q[0], d, phi)
            return sw * (TM.model_profile_kernel(v, d, al, q[1], BK.kernel_pdf("gauss", w), B, nn) - I) / B
        mu0 = alpha0 * math.cos(phi) / d
        starts = [] if prev is None else [prev]
        for w0 in (c["w_starts_um"] if width_fitted else (w_fixed,)):
            starts.append(np.array([mu0, v[n_star]] + ([w0] if width_fitted else []), dtype=float))
        for fct in c["mu_start_factors"]:
            starts.append(np.array([mu0 * fct, v[n_star]] + ([0.6] if width_fitted else []), dtype=float))
        best = None
        for q0 in starts:
            q0 = np.clip(q0, lb, ub)
            r = least_squares(res, q0, bounds=(lb, ub), method="trf", jac="2-point", x_scale="jac",
                              ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=800)
            if np.isfinite(r.cost) and (best is None or r.cost < best.cost):
                best = r
        prev = best.x.copy()
        rows.append([d, best.cost, best.x[0], best.x[1], best.x[2] if width_fitted else w_fixed])
    return np.array(rows)        # columns: d, F (dimensionless), mu, v0, w


def interval_width(dgrid, dchi2, level=1.0):
    """Width (um) of {d : Delta chi^2 <= level} on the grid (linear interpolation at the crossings)."""
    inside = dchi2 <= level
    idx = np.flatnonzero(inside)
    if idx.size == 0:
        return float("nan"), float("nan"), float("nan")
    i0, i1 = idx[0], idx[-1]

    def cross(a, b):
        return dgrid[a] + (level - dchi2[a]) * (dgrid[b] - dgrid[a]) / (dchi2[b] - dchi2[a])
    lo = dgrid[0] if i0 == 0 else cross(i0 - 1, i0)
    hi = dgrid[-1] if i1 == dgrid.size - 1 else cross(i1, i1 + 1)
    return float(hi - lo), float(lo), float(hi)


def dip_variance(P, d, fwhm, alpha):
    """Second central moment (um^2) of the noise-free dip 1 - I_model / B_bar on a wide fine grid."""
    vv = np.linspace(-8.0, 8.0, 16001)
    D = 1.0 - P.TM.model_profile_kernel(vv, d, alpha, 0.0, P.BK.kernel_pdf("gauss", fwhm), 1.0, 4000)
    m0 = integrate.trapezoid(D, vv)
    m1 = integrate.trapezoid(vv * D, vv) / m0
    return float(integrate.trapezoid((vv - m1) ** 2 * D, vv) / m0)


def check_moments(P, c):
    """C4."""
    k = 2.0 * math.sqrt(2.0 * math.log(2.0))
    rows = []
    for d, fwhm in c["moment_cases"]:
        sg = fwhm / k
        pred = sg * sg + d * d / 16.0
        faint = dip_variance(P, d, fwhm, c["faint_alpha"])
        dark = dip_variance(P, d, fwhm, c["mu_per_um"] * d / math.cos(c["phi_rad"]))
        rows.append(dict(d_um=d, fwhm_um=fwhm, predicted_um2=pred, faint_um2=faint, faint_rel=abs(faint - pred) / pred,
                         dark_alpha=c["mu_per_um"] * d / math.cos(c["phi_rad"]), dark_um2=dark,
                         dark_rel=(dark - pred) / pred))
    s_fit, s_true = P.cfg.sigma_fit_um, c["true_fwhm_um"] / k
    floor = []
    for d in c["floor_d_um"]:
        for label, mu in (("faint", 0.01 * math.cos(c["phi_rad"]) / d), ("D-046 darkness", c["mu_per_um"])):
            v, I = simulate(P, c, noise=False, d=d, mu=mu)
            b5 = P.block5.fit_profile(v, I, c["phi_rad"], c["B_gl"], P.cfg)
            lead = math.sqrt(d * d + 16.0 * (s_true ** 2 - s_fit ** 2))
            floor.append(dict(d_um=d, darkness=label, alpha=mu * d / math.cos(c["phi_rad"]), d_hat_um=b5.d_hat_um,
                              status=b5.status, leading_order_um=lead, rel=(b5.d_hat_um - lead) / lead))
    ok = all(r["faint_rel"] <= TOL["moment_rel"] for r in rows) and \
        all(abs(r["rel"]) <= TOL["floor_rel"] for r in floor if r["darkness"] == "faint")
    return dict(moments=rows, floor=floor, sigma_fit_um=s_fit, sigma_true_um=s_true, ok=bool(ok))


def monte_carlo(P, c):
    """C5: d-hat over noise draws for Block 5, the Gaussian at a fitted width, and that with lambda 0.25 um."""
    FV, cfg, phi, B = P.FV, P.cfg, c["phi_rad"], c["B_gl"]
    keys = ("block5", "fitted", "fitted_lam")
    out = {k: dict(d=[], w=[], status=[], at_bound=[], irls_ok=[]) for k in keys}
    for n in range(c["n_mc"]):
        rng = np.random.default_rng([c["seed"], n])
        v, I = simulate(P, c, rng=rng)
        pf = FV.PlaneFits(v, I, phi, B, cfg)
        for k, f in zip(keys, (pf.base, pf.fit(spec(P, True)), pf.fit(spec(P, True, c["lam_valley_um"])))):
            out[k]["d"].append(f.d_um)
            out[k]["w"].append(f.fwhm_um)
            out[k]["status"].append(f.status)
            out[k]["at_bound"].append(";".join(f.at_bound))
            out[k]["irls_ok"].append(f.irls_ok)
    summ = {}
    for k in keys:
        d = np.array(out[k]["d"], dtype=float)
        w = np.array(out[k]["w"], dtype=float)
        summ[k] = dict(mean_d=float(np.nanmean(d)), sd_d=float(np.nanstd(d, ddof=1)), median_d=float(np.nanmedian(d)),
                       q10_d=float(np.nanquantile(d, 0.1)), q90_d=float(np.nanquantile(d, 0.9)),
                       min_d=float(np.nanmin(d)), max_d=float(np.nanmax(d)),
                       mean_w=float(np.nanmean(w)), q10_w=float(np.nanquantile(w, 0.1)), q90_w=float(np.nanquantile(w, 0.9)),
                       n_at_bound=int(sum(1 for s in out[k]["status"] if s == "at_bound")),
                       n_failed=int(sum(1 for s in out[k]["status"] if s == "failed")),
                       bounds_hit=sorted(set(x for s in out[k]["at_bound"] for x in s.split(";") if x)),
                       n_irls_not_settled=int(sum(1 for x in out[k]["irls_ok"] if x is False)))
    ok = (summ["fitted"]["sd_d"] >= TOL["spread_ratio"] * summ["block5"]["sd_d"]
          and summ["block5"]["mean_d"] >= TOL["fixed_bias_um"])
    return dict(raw={k: dict(d=out[k]["d"], w=out[k]["w"], status=out[k]["status"], at_bound=out[k]["at_bound"])
                     for k in keys}, summary=summ, ok=bool(ok))


# ------------------------------------------------------------------- main ---

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="", help="comma list of C1..C5 to run, merged into an existing JSON")
    a = ap.parse_args(argv)
    todo = set(x.strip() for x in a.only.split(",") if x.strip()) or {"C1", "C2", "C3", "C4", "C5"}
    os.makedirs(a.out, exist_ok=True)
    P, c = load(a.repo), CONFIG
    t0 = time.time()
    rng = np.random.default_rng(c["seed"])
    v, I = simulate(P, c, rng=rng)                           # the worked profile
    jpath = os.path.join(a.out, "objective_check.json")
    res = dict(config=c, tol=TOL)
    if todo != {"C1", "C2", "C3", "C4", "C5"} and os.path.exists(jpath):
        with open(jpath) as fh:
            res = json.load(fh)
        res["config"], res["tol"] = c, TOL

    if "C1" in todo:
        res["C1"] = check_cost(P, c, v, I)
        print("C1 cost: %s (%.0f s)" % ("PASS" if res["C1"]["ok"] else "FAIL", time.time() - t0), flush=True)

    if "C2" in todo:
        res["C2"] = {}
        for case, cs in CASES_C2.items():
            if cs.get("worked"):
                vc, Ic = v, I
            else:
                rng_c = np.random.default_rng([c["seed"], cs["seed_tail"]]) if cs["noise"] else None
                vc, Ic = simulate(P, c, rng=rng_c, noise=cs["noise"], neighbour=cs["neighbour"] or False,
                                  kernel=None if cs["kernel"] is None else c[cs["kernel"]])
            res["C2"][case] = check_balance(P, c, vc, Ic, case)
        res["C2"]["ok"] = bool(all(res["C2"][k]["ok"] for k in CASES_C2))
        print("C2 balance: %s (%.0f s)" % ("PASS" if res["C2"]["ok"] else "FAIL", time.time() - t0), flush=True)

    if not {"C3", "C5"} <= todo:          # the figures' arrays need C3 and C5
        for k in ("C4",):
            if k in todo:
                res[k] = check_moments(P, c)
                print("C4 moments: %s (%.0f s)" % ("PASS" if res[k]["ok"] else "FAIL", time.time() - t0), flush=True)
        with open(jpath, "w") as fh:
            json.dump(res, fh, indent=1, default=float)
        print("ran %s; merged into %s" % (",".join(sorted(todo)), jpath))
        return 0
    lo, hi, st = c["d_grid_um"]
    dgrid = np.round(np.arange(lo, hi + 0.5 * st, st), 10)
    prof_fixed = profile_along_d(P, c, v, I, False, d_grid=dgrid)
    prof_fit = profile_along_d(P, c, v, I, True, d_grid=dgrid)
    pf = P.FV.PlaneFits(v, I, c["phi_rad"], c["B_gl"], P.cfg)
    f25 = pf.fit(spec(P, True, c["lam_valley_um"]))
    s2 = c["noise_sd_gl"] ** 2
    B2 = c["B_gl"] ** 2
    chi_fixed = 2.0 * B2 * (prof_fixed[:, 1] - prof_fixed[:, 1].min()) / s2
    chi_fit = 2.0 * B2 * (prof_fit[:, 1] - prof_fit[:, 1].min()) / s2
    wf, lof, hif = interval_width(dgrid, chi_fixed)
    wt, lot, hit = interval_width(dgrid, chi_fit)
    k = 2.0 * math.sqrt(2.0 * math.log(2.0))
    s2_valley = (prof_fit[:, 4] / k) ** 2 + dgrid ** 2 / 16.0
    in_valley = chi_fit <= 4.0
    res["C3"] = dict(argmin_fixed_um=float(dgrid[np.argmin(chi_fixed)]), argmin_fitted_um=float(dgrid[np.argmin(chi_fit)]),
                     interval_fixed_um=[wf, lof, hif], interval_fitted_um=[wt, lot, hit],
                     ratio=wt / wf if wf > 0 else float("nan"),
                     w_at_min_fitted_um=float(prof_fit[np.argmin(chi_fit), 4]),
                     s2_valley_um2_range=[float(s2_valley[in_valley].min()), float(s2_valley[in_valley].max())],
                     lam_fixed_point_v0_um=f25.v0_um, lam_fixed_point_d_um=f25.d_um, lam_fixed_point_fwhm_um=f25.fwhm_um)
    res["C3"]["ok"] = bool(res["C3"]["ratio"] >= TOL["valley_ratio"] and res["C3"]["argmin_fixed_um"] >= TOL["fixed_bias_um"])
    print("C3 valley: %s (%.0f s)" % ("PASS" if res["C3"]["ok"] else "FAIL", time.time() - t0), flush=True)

    res["C4"] = check_moments(P, c)
    print("C4 moments: %s (%.0f s)" % ("PASS" if res["C4"]["ok"] else "FAIL", time.time() - t0), flush=True)

    res["C5"] = monte_carlo(P, c)
    print("C5 spread: %s (%.0f s)" % ("PASS" if res["C5"]["ok"] else "FAIL", time.time() - t0), flush=True)

    # arrays for the figures (the worked profile, its two fits, the weights, the profiles along d, the draws)
    base = pf.base
    fu = pf.fit(spec(P, True))
    vv = np.linspace(v[0], v[-1], 601)
    np.savez(os.path.join(a.out, "objective_check.npz"),
             v=v, I=I, vv=vv, B=c["B_gl"], noise_sd=c["noise_sd_gl"],
             ref_curve=P.FV.variant_curve(base, vv, c["phi_rad"], c["B_gl"]),
             fit_curve=P.FV.variant_curve(fu, vv, c["phi_rad"], c["B_gl"]),
             ref_model=base.model_gl, fit_model=fu.model_gl,
             ref_theta=np.array([base.d_um, base.mu_per_um, base.v0_um, base.fwhm_um]),
             fit_theta=np.array([fu.d_um, fu.mu_per_um, fu.v0_um, fu.fwhm_um]),
             lambdas=np.array(c["lambdas_um"]), true_d=c["d_um"], true_fwhm=c["true_fwhm_um"], true_v0=c["v0_um"],
             dgrid=dgrid, chi_fixed=chi_fixed, chi_fit=chi_fit, prof_fixed=prof_fixed, prof_fit=prof_fit,
             lam_valley=c["lam_valley_um"],
             mc_block5=np.array(res["C5"]["raw"]["block5"]["d"]), mc_fitted=np.array(res["C5"]["raw"]["fitted"]["d"]),
             mc_fitted_lam=np.array(res["C5"]["raw"]["fitted_lam"]["d"]),
             mc_w_fitted=np.array(res["C5"]["raw"]["fitted"]["w"]),
             mc_w_fitted_lam=np.array(res["C5"]["raw"]["fitted_lam"]["w"]),
             mc_bound_block5=np.array([s != "" for s in res["C5"]["raw"]["block5"]["at_bound"]]),
             mc_bound_fitted=np.array([s != "" for s in res["C5"]["raw"]["fitted"]["at_bound"]]),
             mc_bound_fitted_lam=np.array([s != "" for s in res["C5"]["raw"]["fitted_lam"]["at_bound"]]),
             fit25_theta=np.array([f25.d_um, f25.mu_per_um, f25.v0_um, f25.fwhm_um]),
             fit_status=np.array([base.status, fu.status, f25.status]),
             sigma_fit=P.cfg.sigma_fit_um, mu_hi=P.cfg.fit_mu_bounds_per_um[1])
    worked = dict(N=int(v.size), step_um=float(v[1] - v[0]), half_um=float(v[-1]),
                  ref=dict(d_um=base.d_um, mu=base.mu_per_um, v0_um=base.v0_um, fwhm_um=base.fwhm_um, status=base.status,
                           n_nodes=base.n_nodes, n_starts=base.n_starts, rss_gl2=base.rss_gl2),
                  fitted=dict(d_um=fu.d_um, mu=fu.mu_per_um, v0_um=fu.v0_um, fwhm_um=fu.fwhm_um, status=fu.status,
                              n_nodes=fu.n_nodes, n_starts=fu.n_starts, best_start=fu.best_start, rss_gl2=fu.rss_gl2),
                  weights={})
    for lam in c["lambdas_um"]:
        fl = pf.fit(spec(P, True, lam))
        om = P.FV.centre_weights(v, fl.v0_um, lam)
        worked["weights"]["%g" % lam] = dict(d_um=fl.d_um, fwhm_um=fl.fwhm_um, v0_um=fl.v0_um, n_irls=fl.n_irls,
                                             irls_ok=fl.irls_ok, status=fl.status, sum_w=float(om.sum()),
                                             n_eff=float(om.sum() ** 2 / np.sum(om ** 2)),
                                             w_at_0p3um=math.exp(-0.3 / lam), w_at_1um=math.exp(-1.0 / lam))
    res["worked"] = worked
    res["seconds"] = time.time() - t0
    with open(jpath, "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print("all: %s; %.0f s; results in %s" % ("PASS" if all(res[k]["ok"] for k in ("C1", "C2", "C3", "C4", "C5"))
                                              else "see FAIL above", res["seconds"], a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
