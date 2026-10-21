"""Smoke test for the phantoms and the bias table -- Blocks 6 and 14 in
specs/SPEC.md (procedure s.3.5, 3.7, 3.8; D-024, D-057.1, D-059 to D-063,
D-071).

Fixture (Block 14.4): the G x n_real design on the reduced ranges of gate 2
(d 0.4-3.5 um, phi 0-20 deg, mu 0.3-3 /um), b(x) = 1 + 0.06/d + 0.15
sin^2 phi + 0.05 ln(mu_hat d_hat) at the rows' own coordinates, mu_hat d_hat =
mu d exp(N(0, 0.1^2)) per realization, y = b + u_g + eps (a draw-level offset
u_g of correlation 0.2, tau = 0.03), failures with probability 0.3 below 0.6
um. "Inside the band": grid points at least one bandwidth from every end of
the axes and of the band ln mu_min <= x3 - x1 <= ln mu_max.

Checks
    test_known_answer   on the fixture (c = 1), at the grid points inside the
                        band: the noise b_hat - b_hat_0 (b_hat_0 the fit of the
                        same design without noise) over the sandwich SE has SD
                        0.8-1.25, at most 12 % beyond 2 and none beyond 5; the
                        bias |b_hat_0 - b| <= 0.003 (the O(h^2) smoothing bias
                        of the local linear fit); the median tau_hat
                        there within 10 % of tau; the failure rate at every
                        grid point inside the band within 4 SD of its
                        expectation given the draws; m_hat = d b_hat; noise-
                        free ratios linear in x (and constant) are reproduced
                        exactly at every supported grid point (1e-10); on
                        check_b_target.py's toy (the d_hat and mu_hat d_hat
                        errors correlated, rho = -0.9) the table follows the
                        conditional target b(x), not the draws' mean relation,
                        which differs by 0.10-0.13 at the band's ends (within
                        0.03 of the first, at least 0.07 from the second)
    test_reference      the grid arrays (b_hat, both SEs, tau_hat, N_eff, the
                        mass, the failure rate) equal a brute-force evaluation
                        of their definitions point by point (1e-12 relative),
                        under table_phi_axis "true" and "measured"; the
                        smoother weights sum to one and annihilate a linear b
                        (1e-10); the sandwich by factors equals the sandwich by
                        pairs (1e-12); the interpolant reproduces the grid
                        (1e-10); the failure count's coordinates are the true
                        ones; q_loo is the 5th percentile of the rows'
                        leave-one-out masses (1e-3); the grid sizes at the
                        default ranges ((41, 37, 37) at c = 1); the CV scores
                        equal those of tables fitted on each fold's complement
                        (1e-9); the per-draw table by hand; the draws follow
                        their laws (Kolmogorov-
                        Smirnov on 2000 draws g = 1..2000, realization 1); Block
                        14.2: the branch radius's noise at d = 0.2 um follows
                        the normal truncated below at -d/2 (KS against
                        scipy.stats.truncnorm, 4000 realizations) and the
                        rejections per kept draw average Phi(-d/2s) / (1 -
                        Phi(-d/2s)) within 4 SE; D-062's keys and draw orders,
                        recomputed from NumPy alone, give the same features,
                        nuisances and radius
    test_convergence    the sandwich SE falls as 1/sqrt(G): G = 500 vs 2000
                        (within 25 %); over 60 tables of the fixture the
                        sandwich SE matches the across-table SD of b_hat at five
                        points inside the band (pooled within 25 %, each within
                        40 %) while the independent-rows SE falls short
    test_invariants     Block 14.2: a draw depends only on (seed, g) and a
                        realization only on (seed, g, r), in any order; a draw's
                        realizations share its features and differ in their
                        nuisances; the feature, nuisance and radius streams
                        differ for g, r in 1..3; index 0 is refused; the phantom
                        branch lies on the tube's axis at the configured spacing
                        and carries radius_um at every node; sd 0 gives d/2;
                        the grid design visits cell (g - 1) mod (n_d n_phi);
                        Block 14.4: the table does not depend on the rows'
                        order (1e-10); the folds keep a draw's realizations
                        together
    test_contract       two rendered realizations end to end: row contract;
                        CSV and table files round trip exactly; a table file of
                        Block 6's format is refused by name; the CLI renders
                        draws by --draw-start/--draw-stop and merges them with
                        synthetic rows of more draws (Block 14.6: the draws
                        short of n_realizations reported, the per-draw table
                        written, a (draw, realization) seen twice refused); a
                        changed estimator
                        setting is refused; a merge under another simulation
                        configuration is refused, under another correction
                        setting accepted
    test_determinism    a rendered realization repeats bit for bit; a table fit
                        repeats bit for bit
    test_edge_cases     row provenance; unsupported estimators raise; the
                        cross-validated multiplier on pure noise is the grid's
                        largest and flagged at its edge; tps_spline (the
                        small-table path) fits with folds by draw and refuses
                        more than tps_max_rows rows in S; too few rows; bad
                        inputs; failure_rate_ignore: () (the default, D-057.1)
                        counts every row; a named screen ("dark" or
                        "stack_edge") removes the rows it rejected, whatever
                        else they failed; "crossing" and an unlabelled
                        rejection count as failures

Run
    cd "Passive Features/Diameter Re-measurement"
    python tests/smoke/test_smoke_table.py

Keep this file pure ASCII (hpc-python-compat).
"""
from __future__ import annotations

import dataclasses
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
import traceback
import unittest
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
SRC = WS / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from allen_diameter.analysis import phantoms as PH  # noqa: E402
from allen_diameter.analysis import table as TB  # noqa: E402
from allen_diameter.analysis.table import check_rows_config, fit_table, row_provenance, table_inputs  # noqa: E402
from allen_diameter.config import default_config  # noqa: E402
from allen_diameter.loading import table_io  # noqa: E402

SEED = 20261006
REPORT_PACKAGES = ("numpy", "scipy")
PX = 0.1144
TAU = 0.03


def b_true(x1, x2, x3):
    return 1.0 + 0.06 / np.exp(x1) + 0.15 * np.sin(x2) ** 2 + 0.05 * x3


def reduced_cfg(**correction):
    """The default configuration on gate 2's reduced ranges (d 0.4-3.5 um, phi 0-20 deg), c fixed at 1 unless
    the correction overrides say otherwise."""
    base = default_config()
    ph = dataclasses.replace(base.phantom, d_range_um=(0.4, 3.5), phi_range_deg=(0.0, 20.0))
    cc = dataclasses.replace(base.correction, **dict(dict(bandwidth_cv_grid=(1.0,)), **correction))
    return dataclasses.replace(base, phantom=ph, correction=cc)


def fixture(G, n_real, seed, cfg, tau=TAU, rho=0.2, eta=0.1, p_fail=None, b_fn=b_true, first=1):
    """Rows of the G x n_real design on cfg's ranges (see the module docstring); p_fail(d) the failure
    probability of a row (default 0.3 below 0.6 um), b_fn(x1, x2, x3) the mean ratio."""
    rng = np.random.default_rng(seed)
    ph = cfg.phantom
    (dlo, dhi), (mlo, mhi) = ph.d_range_um, ph.mu_range_per_um
    p_fail = p_fail or (lambda d: 0.3 if d < 0.6 else 0.0)
    rows = []
    for g in range(first, first + G):
        d = math.exp(rng.uniform(math.log(dlo), math.log(dhi)))
        phi = math.radians(rng.uniform(*ph.phi_range_deg))
        mu = math.exp(rng.uniform(math.log(mlo), math.log(mhi)))
        u = rng.normal(0.0, math.sqrt(rho) * tau)
        for r in range(1, n_real + 1):
            md = mu * d * math.exp(rng.normal(0.0, eta))
            y = float(b_fn(math.log(d), phi, math.log(md))) + u + rng.normal(0.0, math.sqrt(1.0 - rho) * tau)
            fail = rng.random() < p_fail(d)
            rows.append(dict(draw_index=g, realization_index=r, d_um=d, phi_rad=phi, theta_rad=0.5, mu_per_um=mu,
                             d_hat_um=y * d, mu_hat_per_um=md / (y * d), meas_phi_rad=phi, ratio=y, in_S=not fail,
                             reject="faint" if fail else ""))
    return rows


def target_toy(G, k, seed, cfg, rho=-0.9, s1=0.18, s2=0.4):
    """check_b_target.py's toy on the three coordinates (Block 14.4): a realization's fitted ln(mu_hat d_hat) =
    ln(mu_g d_g) + s2 eta2 and its ratio y = b_true(ln d_g, phi_g, ln(mu_g d_g)) + s1 eta1, corr(eta1, eta2) = rho,
    every row in S; and (target, draw-mean) at (d, phi, l): b(x) = b_true(m) + s1 rho (l - m) / s2 and b_true(m),
    m = E[ln(mu d) | l], the mean of N(l, s2^2) truncated to the design's [ln(mu_min d), ln(mu_max d)]."""
    rng = np.random.default_rng(seed)
    ph = cfg.phantom
    (dlo, dhi), (mlo, mhi) = ph.d_range_um, ph.mu_range_per_um
    d = np.exp(rng.uniform(math.log(dlo), math.log(dhi), G))
    phi = np.radians(rng.uniform(*ph.phi_range_deg, G))
    mu = np.exp(rng.uniform(math.log(mlo), math.log(mhi), G))
    e1 = rng.standard_normal((G, k))
    e2 = rho * e1 + math.sqrt(1.0 - rho ** 2) * rng.standard_normal((G, k))
    rows = []
    for g in range(G):
        for r in range(k):
            y = float(b_true(math.log(d[g]), phi[g], math.log(mu[g] * d[g]))) + s1 * e1[g, r]
            md = mu[g] * d[g] * math.exp(s2 * e2[g, r])
            rows.append(dict(draw_index=g + 1, realization_index=r + 1, d_um=d[g], phi_rad=phi[g], mu_per_um=mu[g],
                             d_hat_um=y * d[g], mu_hat_per_um=md / (y * d[g]), ratio=y, in_S=True, reject=""))

    def answer(dq, phq, l):
        L0, L1 = math.log(mlo * dq), math.log(mhi * dq)
        a, b = (L0 - l) / s2, (L1 - l) / s2
        m = l + s2 * (stats.norm.pdf(a) - stats.norm.pdf(b)) / (stats.norm.cdf(b) - stats.norm.cdf(a))
        return float(b_true(math.log(dq), phq, m) + s1 * rho * (l - m) / s2), float(b_true(math.log(dq), phq, m))
    return rows, answer


def grid_points(T):
    return np.stack(np.meshgrid(*T.axes, indexing="ij"), -1)


def inside_band(T, cfg, margin=1.0):
    """Grid points at least `margin` bandwidths from every end of the axes and of the design's band."""
    P, h = grid_points(T), T.meta["bandwidth"]
    (l1, u1), (l2, u2), _ = TB.design_ranges(cfg)
    lm, um = (math.log(x) for x in cfg.phantom.mu_range_per_um)
    x1, x2, x3 = P[..., 0], P[..., 1], P[..., 2]
    return ((x1 >= l1 + margin * h[0]) & (x1 <= u1 - margin * h[0]) & (x2 >= l2 + margin * h[1])
            & (x2 <= u2 - margin * h[1]) & (x3 - x1 >= lm + margin * h[2]) & (x3 - x1 <= um - margin * h[2]))


def brute_force(T, rows, cfg):
    """The table's arrays from their definitions, point by point over the grid (the residuals from the table's
    own interpolant at the rows, clamped to the grid), and the smoother weights' sums and linear moments."""
    R = TB.table_rows(rows, cfg)
    h = np.array(T.meta["bandwidth"])
    P = grid_points(T).reshape(-1, 3)
    Xc = TB.clamp_to_grid(T.axes, R.X)
    e = R.y - T.at("b_hat", np.exp(Xc[:, 0]), Xc[:, 1], np.exp(Xc[:, 2]))
    out = {k: np.full(P.shape[0], np.nan) for k in TB.ARRAYS}
    ell_sum, ell_lin = [], []
    draws = np.unique(R.draw)
    for i, x in enumerate(P):
        U = (R.X - x) / h
        w = np.exp(-0.5 * (U * U).sum(1))
        Z = np.column_stack([np.ones(U.shape[0]), U])
        A = Z.T @ (w[:, None] * Z)
        neff = w.sum() ** 2 / (w * w).sum()
        sv = np.linalg.svd(A, compute_uv=False)
        if sv[-1] / sv[0] >= TB.RCOND_MIN and neff >= cfg.correction.min_neff:
            ell = np.linalg.solve(A, Z.T * w[None, :])[0]
            out["b_hat"][i] = ell @ R.y
            out["se_hat"][i] = math.sqrt(sum(float(ell[R.draw == g] @ e[R.draw == g]) ** 2 for g in draws))
            tau2 = (w * e * e).sum() / w.sum() * neff / (neff - 1)
            out["tau_hat"][i], out["se_indep"][i], out["n_eff"][i] = math.sqrt(tau2), math.sqrt(tau2 / neff), neff
            ell_sum.append(ell.sum() - 1.0)
            ell_lin.append(np.abs(ell @ U).max())
        wf = np.exp(-0.5 * (((R.X_fail - x) / h) ** 2).sum(1))
        out["mass"][i], out["p_fail"][i] = wf.sum(), (wf * R.failed).sum() / wf.sum()
    return out, np.array(ell_sum), np.array(ell_lin)


def small_cfg(**phantom):
    """Short rendered replicates: low tilts, short tubes, smaller block, p_x/8 everywhere; 2 realizations a draw."""
    base = default_config()
    rc = dataclasses.replace(base.renderer, U_um=4.0, pad_um=1.0, h_g_um_thin=PX / 8)
    ms = dataclasses.replace(base.measure, block_half_um=3.5)
    ph = dataclasses.replace(base.phantom, phi_range_deg=(0.0, 30.0), d_range_um=(0.5, 1.2), nodes_each_way=4,
                             n_realizations=2, **phantom)
    return dataclasses.replace(base, renderer=rc, measure=ms, phantom=ph)


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    cfg = reduced_cfg()
    rows = fixture(1000, 5, SEED, cfg)
    T = fit_table(rows, cfg)
    T0 = fit_table(fixture(1000, 5, SEED, cfg, tau=0.0), cfg)        # the same design without noise: the bias
    P = grid_points(T)
    ins = inside_band(T, cfg) & T.supported & T0.supported
    assert ins.sum() >= 500, ins.sum()
    # the noise against the sandwich SE (the SE covers the noise; the O(h^2) bias is checked on its own)
    z = (T.arrays["b_hat"] - T0.arrays["b_hat"])[ins] / T.arrays["se_hat"][ins]
    assert 0.8 <= z.std() <= 1.25 and (np.abs(z) > 2).mean() <= 0.12 and np.abs(z).max() <= 5.0, (
        "noise / SE", z.std(), (np.abs(z) > 2).mean(), np.abs(z).max())
    bias = np.abs(T0.arrays["b_hat"] - b_true(P[..., 0], P[..., 1], P[..., 2]))[ins]
    assert bias.max() <= 0.003, ("bias at c = 1", bias.max())
    assert abs(np.median(T.arrays["tau_hat"][ins]) / TAU - 1) <= 0.10, np.median(T.arrays["tau_hat"][ins])
    # the failure rate against its expectation given the draws: p_n = 0.3 below 0.6 um, at the true coordinates
    R = TB.table_rows(rows, cfg)
    h = np.array(T.meta["bandwidth"])
    pn = np.where(np.exp(R.X_fail[:, 0]) < 0.6, 0.3, 0.0)
    Q, F = P[ins], T.arrays["p_fail"][ins]
    zf = []
    for x, got in zip(Q[:: max(1, Q.shape[0] // 200)], F[:: max(1, Q.shape[0] // 200)]):
        w = np.exp(-0.5 * (((R.X_fail - x) / h) ** 2).sum(1))
        sd = math.sqrt((w * w * pn * (1 - pn)).sum()) / w.sum()
        zf.append(abs(got - (w * pn).sum() / w.sum()) / (sd + 1e-12))     # 1e-12: the rate's rounding far away
    assert zf and max(zf) <= 4.0, max(zf)
    assert float(T.failure_rate(2.0, 0.1, 2.0)) <= 1e-6 and float(T.failure_rate(0.45, 0.1, 0.5)) > 0.1
    d, phi, md = np.array([0.7, 1.5]), np.radians([5.0, 12.0]), np.array([0.8, 2.0])
    assert np.array_equal(T.m_hat(d, phi, md), d * T.b_hat(d, phi, md))
    # the target is the conditional mean at the fitted coordinate, not the draws' mean relation (D-060.1, D-063.1;
    # check_b_target.py): with the d_hat and mu_hat d_hat errors correlated (rho = -0.9), at the band's ends the two
    # differ by 0.13 and the table follows the first (8000 draws x 5, c = 0.7071, the CV's choice at 20000 draws)
    toy_cfg = reduced_cfg(bandwidth_cv_grid=(0.7071,))
    toy_rows, answer = target_toy(8000, 5, SEED + 2, toy_cfg)
    Tt = fit_table(toy_rows, toy_cfg)
    for dq, phd, l in ((1.0, 10.0, -0.05), (1.0, 10.0, math.log(3.0)), (1.0, 10.0, math.log(0.3)),
                       (2.0, 8.0, math.log(6.0) - 0.1)):
        bt, bd = answer(dq, math.radians(phd), l)
        bh = float(Tt.b_hat(dq, math.radians(phd), math.exp(l)))
        assert abs(bh - bt) <= 0.03 and (abs(bt - bd) < 0.01 or abs(bh - bd) >= 0.07), (dq, phd, l, bh, bt, bd)
    # noise-free ratios linear in x, and constant: reproduced exactly where the grid has support
    for fn in (lambda x1, x2, x3: 1.1 + 0.2 * x1 - 0.3 * x2 + 0.07 * x3, lambda x1, x2, x3: 1.07 + 0.0 * x1):
        Tl = fit_table(fixture(300, 3, SEED + 1, cfg, tau=0.0, b_fn=fn), cfg)
        Pl = grid_points(Tl)
        ok = Tl.supported
        err = np.abs(Tl.arrays["b_hat"][ok] - fn(Pl[..., 0], Pl[..., 1], Pl[..., 2])[ok])
        assert ok.sum() > 100 and err.max() <= 1e-10, (ok.sum(), err.max())


def test_reference():
    for axis in ("true", "measured"):
        base = default_config()
        cfg = dataclasses.replace(base, phantom=dataclasses.replace(
            base.phantom, d_range_um=(0.5, 2.0), phi_range_deg=(0.0, 30.0), mu_range_per_um=(0.5, 2.0)),
            correction=dataclasses.replace(base.correction, bandwidth_cv_grid=(2.0,), min_neff=3.0,
                                           table_phi_axis=axis))
        rows = fixture(40, 4, SEED + 3, cfg)
        if axis == "measured":
            rng = np.random.default_rng(SEED + 4)
            for r in rows:
                r["meas_phi_rad"] = abs(r["phi_rad"] + rng.normal(0.0, 0.03))
        T = fit_table(rows, cfg)
        bf, ell_sum, ell_lin = brute_force(T, rows, cfg)
        for k in TB.ARRAYS:
            got, want = T.arrays[k].ravel(), bf[k]
            fin = np.isfinite(want)
            assert np.array_equal(np.isfinite(got), fin) and fin.sum() > 100, (axis, k, np.isfinite(got).sum(),
                                                                                fin.sum())
            rel = np.max(np.abs(got[fin] - want[fin])) / np.max(np.abs(want[fin]))
            assert rel <= 1e-12, (axis, k, rel)
        assert np.abs(ell_sum).max() <= 1e-10 and ell_lin.max() <= 1e-10, (axis, np.abs(ell_sum).max(), ell_lin.max())
        if axis == "true":
            R = TB.table_rows(rows, cfg)
            Xc = TB.clamp_to_grid(T.axes, R.X)
            e = R.y - T.at("b_hat", np.exp(Xc[:, 0]), Xc[:, 1], np.exp(Xc[:, 2]))
            B1 = TB.sandwich_true(T.axes, T.meta["bandwidth"], R.X, e, R.draw)
            B2 = TB.sandwich_pairs(T.axes, T.meta["bandwidth"], R.X, e, R.draw)
            assert np.max(np.abs(B1 - B2)) <= 1e-12 * np.max(np.abs(B2)), "sandwich by factors vs by pairs"
        # the interpolant reproduces the grid where it has support
        P = grid_points(T)[T.supported]
        rep = T.b_hat(np.exp(P[:, 0]), P[:, 1], np.exp(P[:, 2]))
        assert np.max(np.abs(rep - T.arrays["b_hat"][T.supported])) <= 1e-10, "grid reproduction"
        # the failure count's coordinates are the true (ln d_g, phi_g, ln(mu_g d_g)) of every row (D-062.5), and
        # q_loo the 5th percentile of the rows' leave-one-out masses (D-063.4; M at the rows by the interpolant)
        R = TB.table_rows(rows, cfg)
        d = np.array([r["d_um"] for r in rows])
        want = table_inputs(d, [r["phi_rad"] for r in rows], [r["mu_per_um"] for r in rows], d)
        assert np.array_equal(R.X_fail, want), (axis, "failure coordinates")
        m_rows = np.array([np.exp(-0.5 * (((R.X_fail - x) / np.array(T.meta["bandwidth"])) ** 2).sum(1)).sum()
                           for x in R.X_fail])
        q = np.percentile(m_rows - 1.0, cfg.correction.loo_mass_percentile)
        assert abs(T.q_loo / q - 1) <= 1e-3, (axis, T.q_loo, q)
    # the grid at the default ranges: spacing at most c h_k / 2, (41, 37, 37) points at c = 1 (D-071.3)
    cfg = default_config()
    for c, want in ((1.0, (41, 37, 37)), (2.0, (21, 19, 19)), (0.5, (81, 73, 72))):
        axes = TB.grid_axes(np.empty((0, 3)), cfg, c)
        h = TB.bandwidths(cfg, c)
        assert tuple(a.size for a in axes) == want and all(np.diff(a).max() <= 0.5 * hk * (1 + 1e-12)
                                                           for a, hk in zip(axes, h)), (c, [a.size for a in axes])
    # the cross-validation's sums (the whole set's less fold f's) against tables fitted on each fold's complement;
    # without the fit's noise in mu_hat d_hat (eta = 0) every axis is the design's, the same for every subset
    cv = reduced_cfg(bandwidth_cv_grid=(0.7071, 1.4142))
    rows = fixture(150, 3, SEED + 21, cv, eta=0.0)
    R = TB.table_rows(rows, cv)
    mult, scores, edge = TB.choose_multiplier(R, cv)
    K = cv.correction.spline_cv_folds
    fold = TB.fold_of_draw(R.draw, K)
    for c, score in zip(cv.correction.bandwidth_cv_grid, scores):
        axes, h = TB.grid_axes(R.X, cv, c), TB.bandwidths(cv, c)
        err = 0.0
        for f in range(K):
            tr = fold != f
            assert [a.size for a in TB.grid_axes(R.X[tr], cv, c)] == [a.size for a in axes]
            b, _, _, ok = TB.local_linear(TB.grid_sums(axes, h, R.X[tr], TB.LL_SPECS, {"y": R.y[tr]})[0],
                                          cv.correction.min_neff)
            pred = TB.interpolant(axes, TB.filled(b, ok))(TB.clamp_to_grid(axes, R.X[~tr]))
            err += float(np.sum((R.y[~tr] - pred) ** 2))
        assert abs(err / R.y.size / score - 1) <= 1e-9, (c, err / R.y.size, score)
    assert mult == cv.correction.bandwidth_cv_grid[int(np.argmin(scores))] and edge
    # the per-draw table (D-060.2): counts, the mean and SD of a draw's ratios in S, the failure share
    pd_rows = TB.per_draw_table(rows, cv)
    assert [r["draw_index"] for r in pd_rows] == sorted({r["draw_index"] for r in rows})
    for got in pd_rows[:20]:
        mine = [r for r in rows if r["draw_index"] == got["draw_index"]]
        ys = np.array([r["ratio"] for r in mine if r["in_S"]])
        assert got["n_rows"] == len(mine) and got["n_S"] == ys.size and got["d_um"] == mine[0]["d_um"]
        assert abs(got["failure_share"] - (1 - ys.size / len(mine))) <= 1e-15
        if ys.size:
            assert got["y_mean"] == float(ys.mean()) and (ys.size < 2 or got["y_sd"] == float(ys.std(ddof=1)))
    cfg = default_config()
    ph, acq = cfg.phantom, cfg.acquisition
    draws = [PH.draw_replicate(cfg, SEED, g, 1)[0] for g in range(1, 2001)]
    lo, hi = ph.d_range_um
    mlo, mhi = ph.mu_range_per_um
    laws = {
        "ln d": [(math.log(x.d_um) - math.log(lo)) / (math.log(hi) - math.log(lo)) for x in draws],
        "phi": [math.degrees(x.phi_rad) / 90.0 for x in draws],
        "theta": [x.theta_rad / math.pi for x in draws],
        "ln mu": [(math.log(x.mu_per_um) - math.log(mlo)) / (math.log(mhi) - math.log(mlo)) for x in draws],
        "x offset": [x.cx_um / acq.res0_um + 0.5 for x in draws],
        "y offset": [x.cy_um / acq.res0_um + 0.5 for x in draws],
        "depth": [x.cz_um / acq.dz_um + 0.5 for x in draws],
    }
    for name, u in laws.items():
        p = stats.kstest(u, "uniform").pvalue
        assert p > 1e-3, (name, p)
    # Block 14.2 (D-062.1): r_br = d/2 + eps, eps ~ N(0, sd^2) redrawn until positive; at d = 0.2 um, sd = 0.2 um a
    # draw is rejected with probability Phi(-0.5) = 0.31, so the kept eps follow the normal truncated below at -d/2
    d0, sd = 0.2, ph.radius_noise_sd_um
    reals = [PH.draw_replicate(cfg, SEED, g, 1, d_override_um=d0)[0] for g in range(1, 4001)]
    eps = np.array([x.radius_um - 0.5 * d0 for x in reals])
    p = stats.kstest(eps, stats.truncnorm(-0.5 * d0 / sd, np.inf, loc=0.0, scale=sd).cdf).pvalue
    assert p > 0.01 and np.all(eps > -0.5 * d0), ("radius law", p)
    q = 1.0 - stats.norm.cdf(-0.5 * d0 / sd)             # the probability that a draw is kept
    n_rej = np.array([x.radius_redraws for x in reals], dtype=float)
    mean, se = (1 - q) / q, math.sqrt((1 - q) / q ** 2 / n_rej.size)
    assert abs(n_rej.mean() - mean) <= 4 * se, ("rejections per kept draw", n_rej.mean(), mean, se)
    # D-062's keys and orders, recomputed from NumPy alone: the features of draw g from default_rng([seed, g]) in
    # the order d, phi, theta, mu; the nuisances of realization r from default_rng([seed, g, r]) in the order
    # (cx, cy), cz, the returned generator positioned after them; the radius from default_rng([seed, g, r, 1])
    assert ph.d_log_uniform and ph.mu_log_uniform and ph.subpixel_offset and ph.axis_depth_jitter
    for g, r in ((1, 1), (5, 3), (12, 10)):
        x, gen = PH.draw_replicate(cfg, SEED, g, r)
        f = np.random.default_rng([SEED, g])
        want = (math.exp(f.uniform(math.log(lo), math.log(hi))), math.radians(f.uniform(*ph.phi_range_deg)),
                f.uniform(*ph.theta_range_rad), math.exp(f.uniform(math.log(mlo), math.log(mhi))))
        assert (x.d_um, x.phi_rad, x.theta_rad, x.mu_per_um) == want, ("feature key", g, r)
        n = np.random.default_rng([SEED, g, r])
        off = n.uniform(-0.5 * acq.res0_um, 0.5 * acq.res0_um, 2)
        cz = n.uniform(-0.5 * acq.dz_um, 0.5 * acq.dz_um)
        assert (x.cx_um, x.cy_um, x.cz_um) == (off[0], off[1], cz) and gen.random() == n.random(), ("nuisance key", g, r)
        e = np.random.default_rng([SEED, g, r, 1])
        for k in range(100):
            rb = 0.5 * want[0] + ph.radius_noise_sd_um * e.standard_normal()
            if rb > 0:
                break
        assert (x.radius_um, x.radius_redraws) == (rb, k), ("radius key", g, r)


def test_convergence():
    cfg = reduced_cfg()
    pt = (1.0, math.radians(10.0), 1.0)
    se = [float(fit_table(fixture(G, 5, SEED + G, cfg), cfg).se_hat(*pt)) for G in (500, 2000)]
    assert abs(se[0] / se[1] / 2.0 - 1) <= 0.25, se
    # the sandwich SE against the across-table SD of b_hat, 60 tables of 600 draws x 5 (D-063.3, D-071.2)
    pts = np.array([[1.0, 10.0, 1.0], [0.7, 8.0, 0.8], [2.0, 12.0, 2.5], [1.4, 5.0, 1.2], [0.9, 15.0, 0.6]])
    d, phi, md = pts[:, 0], np.radians(pts[:, 1]), pts[:, 2]
    B, S, SI = [], [], []
    for k in range(60):
        T = fit_table(fixture(600, 5, SEED + 100 + k, cfg), cfg)
        B.append(T.b_hat(d, phi, md))
        S.append(T.se_hat(d, phi, md))
        SI.append(T.se_indep(d, phi, md))
    B, S, SI = np.array(B), np.array(S), np.array(SI)
    sd = B.std(axis=0, ddof=1)
    ratio = S.mean(axis=0) / sd
    pooled = math.sqrt(np.mean(S.mean(axis=0) ** 2) / np.mean(sd ** 2))
    pooled_indep = math.sqrt(np.mean(SI.mean(axis=0) ** 2) / np.mean(sd ** 2))
    assert abs(pooled - 1) <= 0.25 and np.all(np.abs(ratio - 1) <= 0.40), (pooled, ratio)
    assert pooled_indep < pooled, ("the independent-rows SE ignores the draw-level correlation", pooled_indep, pooled)


def test_invariants():
    cfg = default_config()
    a = PH.draw_replicate(cfg, SEED, 7, 2)[0]
    for g, r in ((3, 1), (11, 4), (7, 1), (1, 1)):
        PH.draw_replicate(cfg, SEED, g, r)
    assert PH.draw_replicate(cfg, SEED, 7, 2)[0] == a
    assert PH.draw_replicate(cfg, SEED + 1, 7, 2)[0] != a
    # a draw's realizations share its features and differ in their nuisances and radius
    reals = [PH.draw_replicate(cfg, SEED, 7, r)[0] for r in (1, 2, 3)]
    assert len({(x.d_um, x.phi_rad, x.theta_rad, x.mu_per_um) for x in reals}) == 1
    assert len({(x.cx_um, x.cy_um, x.cz_um) for x in reals}) == 3 and len({x.radius_um for x in reals}) == 3
    assert PH.draw_features(cfg, SEED, 7) == (a.d_um, a.phi_rad, a.theta_rad, a.mu_per_um)
    # the streams of D-062's keys are distinct for indices from 1 (NumPy pads a key with zeros: [s, g, 0] = [s, g])
    first = {}
    for g in (1, 2, 3):
        first[(g,)] = np.random.default_rng([SEED, g]).random()
        for r in (1, 2, 3):
            first[(g, r)] = np.random.default_rng([SEED, g, r]).random()
            first[(g, r, 1)] = np.random.default_rng([SEED, g, r, 1]).random()
    assert len(set(first.values())) == len(first), "two generator keys give one stream"
    assert np.random.default_rng([SEED, 7, 0]).random() == np.random.default_rng([SEED, 7]).random()  # why from 1
    for bad in ((0, 1), (1, 0), (-2, 1), (1.5, 1)):
        try:
            PH.draw_replicate(cfg, SEED, *bad)
        except ValueError:
            continue
        raise AssertionError("index %r accepted" % (bad,))
    zero = dataclasses.replace(cfg, phantom=dataclasses.replace(cfg.phantom, radius_noise_sd_um=0.0))
    z = PH.draw_replicate(zero, SEED, 7, 2)[0]
    assert z.radius_um == 0.5 * z.d_um and z.radius_redraws == 0 and (z.cx_um, z.cz_um) == (a.cx_um, a.cz_um)
    # the grid design (an option; D-024 is random): draw g takes cell (g - 1) mod (n_d n_phi), d-major, so the
    # draws 1..n_d n_phi visit every cell once, in order, and draw n_d n_phi + 1 starts again
    grid = dataclasses.replace(cfg, phantom=dataclasses.replace(cfg.phantom, design="grid"))
    gd, gp = grid.phantom.grid_d_um, grid.phantom.grid_phi_deg
    cells = [PH.draw_features(grid, SEED, g)[:2] for g in range(1, len(gd) * len(gp) + 2)]
    want = [(float(x), math.radians(y)) for x in gd for y in gp]
    assert cells[:-1] == want and cells[-1] == want[0], "grid design cells"
    draw, rng = PH.draw_replicate(cfg, SEED, 5, 1)
    br = PH.phantom_branch(draw, cfg, rng)
    assert np.all(np.asarray(br.radius_um) == draw.radius_um) and draw.radius_um > 0
    tube = PH.phantom_tube(draw, cfg)
    q = br.xyz_um - np.array(tube.c)
    t = tube.t_hat
    off_axis = np.linalg.norm(q - (q @ t)[:, None] * t[None, :], axis=1)
    assert off_axis.max() <= 1e-12 and np.allclose(np.diff(br.s_um), cfg.phantom.phantom_node_step_um, atol=1e-12)
    assert np.allclose(br.xyz_um[cfg.phantom.nodes_each_way], tube.c, atol=0)
    # Block 14.4: the rows' order does not matter; the folds keep a draw's realizations together
    rc = reduced_cfg()
    rows = fixture(300, 3, SEED + 5, rc)
    perm = [rows[i] for i in np.random.default_rng(SEED).permutation(len(rows))]
    T1, T2 = fit_table(rows, rc), fit_table(perm, rc)
    for k in TB.ARRAYS:
        v1, v2 = T1.arrays[k], T2.arrays[k]
        fin = np.isfinite(v1)
        assert np.array_equal(fin, np.isfinite(v2)) and np.max(np.abs(v1[fin] - v2[fin])) <= 1e-10 * np.max(
            np.abs(v1[fin])), k
    assert TB.fold_of_draw([1, 1, 2, 6, 7, 5], 5).tolist() == [0, 0, 1, 0, 1, 4]


def test_contract():
    cfg = small_cfg()
    rows = [PH.run_replicate(cfg, SEED, 1, r) for r in (1, 2)]
    for r in rows:
        for k in ("draw_index", "realization_index", "seed", "d_um", "phi_rad", "theta_rad", "mu_per_um", "cx_um",
                  "cy_um", "cz_um", "radius_um", "radius_redraws", "d_hat_um", "mu_hat_per_um", "fit_status", "flags",
                  "in_S", "reject", "ratio", "meas_phi_rad", "seconds"):
            assert k in r, k
        assert isinstance(r["in_S"], bool) and r["in_S"] == (r["reject"] == "")
        assert abs(r["ratio"] - r["d_hat_um"] / r["d_um"]) <= 1e-15 * abs(r["ratio"])
    assert [(r["draw_index"], r["realization_index"]) for r in rows] == [(1, 1), (1, 2)]
    assert rows[0]["d_um"] == rows[1]["d_um"] and rows[0]["radius_um"] != rows[1]["radius_um"]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "rows.csv")
        table_io.write_rows(rows, path)
        back = table_io.read_rows(path)
        for a, b in zip(rows, back):
            for k, v in a.items():
                assert (v == b[k]) or (isinstance(v, float) and math.isnan(v) and math.isnan(b[k])), (k, v, b[k])
        rc = reduced_cfg()
        T = fit_table(fixture(300, 3, SEED, rc), rc)
        stem = os.path.join(tmp, "table")
        table_io.save_table(T, stem)
        T2 = table_io.load_table(stem)
        D, P, M = np.meshgrid(np.linspace(0.4, 3.5, 5), np.radians(np.linspace(0, 20, 5)), np.linspace(0.2, 6.0, 4))
        for name in TB.ARRAYS:
            assert np.array_equal(T.at(name, D, P, M), T2.at(name, D, P, M), equal_nan=True), name
        assert T2.estimator_hash == T.estimator_hash and T2.q_loo == T.q_loo and T2.meta["n_all"] == 900
        T2.check_estimator(rc)
        try:
            T2.check_estimator(dataclasses.replace(rc, measure=dataclasses.replace(rc.measure, sigma_fit_um=0.125)))
        except ValueError:
            pass
        else:
            raise AssertionError("a table must refuse a fit with another sigma_fit")
        # a table file of Block 6's format (no grid) is refused by name
        meta = json.load(open(stem + ".json"))
        meta.pop("format")
        json.dump(meta, open(os.path.join(tmp, "old.json"), "w"))
        import shutil
        shutil.copy(stem + ".npz", os.path.join(tmp, "old.npz"))      # the arrays are fine: the format key decides
        try:
            table_io.load_table(os.path.join(tmp, "old"))
        except ValueError as exc:
            assert "rebuild" in str(exc), exc
        else:
            raise AssertionError("a table file without the grid format was loaded")
        # the CLI: two chunks of one draw x 2 realizations rendered, then the merge with synthetic rows of 200
        # more draws (a table needs draws in general position; the rendered rows test the plumbing)
        cfgf = small_cfg()
        cj = os.path.join(tmp, "cfg.json")
        with open(cj, "w") as f:
            f.write(cfgf.to_json())
        script = str(WS / "scripts" / "build_table.py")
        out = os.path.join(tmp, "tab")
        for a, b in ((1, 2), (2, 3)):
            subprocess.run([sys.executable, script, "run", "--draw-start", str(a), "--draw-stop", str(b), "--out-dir",
                            out, "--config-json", cj], check=True, capture_output=True)
        refused = subprocess.run([sys.executable, script, "run", "--draw-start", "0", "--draw-stop", "1", "--out-dir",
                                  out, "--config-json", cj], capture_output=True, text=True)
        assert refused.returncode != 0 and "--draw-start" in refused.stderr, refused.stderr[-300:]
        h = cfgf.signature_hash("estimator")
        import glob as _glob
        rendered = table_io.read_rows(sorted(_glob.glob(os.path.join(out, "rows_*.csv"))))
        assert sorted((r["draw_index"], r["realization_index"]) for r in rendered) == [(1, 1), (1, 2), (2, 1), (2, 2)]
        # 2026-10-07: every row carries the simulation hash
        assert {r["sim_hash"] for r in rendered} == {row_provenance(cfgf)}, {r["sim_hash"] for r in rendered}
        synth = [dict(r, sim_hash=row_provenance(cfgf)) for r in fixture(200, 1, SEED + 7, cfgf, first=100)]
        table_io.write_rows(synth, os.path.join(out, "rows_%s_000100_000300.csv" % h))
        res = subprocess.run([sys.executable, script, "merge", "--out-dir", out, "--config-json", cj], check=True,
                             capture_output=True, text=True)
        meta = json.load(open(os.path.join(out, "bias_table_%s.json" % h)))
        assert meta["n_all"] == 204 and meta["estimator_hash"] == h and meta["estimator"] == "local_linear", meta["n_all"]
        assert meta["n_kept"] == sum(1 for r in rendered + synth if r["in_S"]), meta["n_kept"]
        # Block 14.6: the draws short of n_realizations reported, the per-draw table written, a pair seen twice refused
        assert "204 rows of 202 draws from 3 files" in res.stdout, res.stdout[-400:]
        assert "note: 200 draws have fewer than n_realizations = 2 rows" in res.stdout, res.stdout[-400:]
        per = table_io.read_rows(os.path.join(out, "draws_%s.csv" % h))
        assert [r["draw_index"] for r in per] == [1, 2] + list(range(100, 300)), [r["draw_index"] for r in per][:5]
        assert per[0]["n_rows"] == 2 and per[2]["n_rows"] == 1 and per[0]["n_S"] == sum(
            1 for r in rendered if r["draw_index"] == 1 and r["in_S"]), per[0]
        first = sorted(_glob.glob(os.path.join(out, "rows_%s_000001_*.csv" % h)))[0]
        twice = os.path.join(out, "rows_%s_000001_000002_again.csv" % h)
        shutil.copy(first, twice)
        res = subprocess.run([sys.executable, script, "merge", "--out-dir", out, "--config-json", cj],
                             capture_output=True, text=True)
        assert res.returncode != 0 and "merge refused" in res.stderr and "more than once" in res.stderr, res.stderr[-300:]
        os.remove(twice)
        # the merge refuses rows rendered under another simulation configuration -- the dangerous case is the same
        # estimator hash, so the row files are found -- while a change of the correction settings, which only the
        # merge reads, is accepted
        other = dataclasses.replace(cfgf, phantom=dataclasses.replace(cfgf.phantom, d_range_um=(0.5, 2.0)))
        corr = dataclasses.replace(cfgf, correction=dataclasses.replace(cfgf.correction, bandwidth_cv_grid=(1.0,)))
        assert other.signature_hash("estimator") == h and corr.signature_hash("estimator") == h
        for c, ok in ((other, False), (corr, True)):
            cj2 = os.path.join(tmp, "cfg2.json")
            with open(cj2, "w") as f:
                f.write(c.to_json())
            res = subprocess.run([sys.executable, script, "merge", "--out-dir", out, "--config-json", cj2],
                                 capture_output=True, text=True)
            assert (res.returncode == 0) == ok, (ok, res.returncode, res.stderr[-300:])
            if not ok:
                assert "merge refused" in res.stderr and "--config-json" in res.stderr, res.stderr[-300:]


def test_determinism():
    cfg = small_cfg()
    a, b = PH.run_replicate(cfg, SEED, 3, 2), PH.run_replicate(cfg, SEED, 3, 2)
    a.pop("seconds"), b.pop("seconds")
    assert a == b or all((a[k] == b[k]) or (isinstance(a[k], float) and math.isnan(a[k]) and math.isnan(b[k])) for k in a)
    rc = reduced_cfg()
    rows = fixture(200, 3, SEED + 9, rc)
    T1, T2 = fit_table(rows, rc), fit_table(rows, rc)
    assert all(np.array_equal(T1.arrays[k], T2.arrays[k], equal_nan=True) for k in TB.ARRAYS) and T1.q_loo == T2.q_loo


def test_edge_cases():
    # row provenance (2026-10-07): rows without the column (older runs) are counted, not refused; one row from
    # another configuration refuses the lot; the prefix keeps a numeric-looking hash a string through the CSV
    cfg0 = default_config()
    good = row_provenance(cfg0)
    assert check_rows_config([dict(sim_hash=good), dict(sim_hash=good)], cfg0) == 0
    assert check_rows_config([dict(sim_hash=good), dict(), dict(sim_hash="")], cfg0) == 2
    try:
        check_rows_config([dict(sim_hash=good), dict(sim_hash="sim-0000000000000000")], cfg0)
    except ValueError:
        pass
    else:
        raise AssertionError("rows from another configuration were accepted")
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "r.csv")
        table_io.write_rows([dict(sim_hash="sim-1234567890123456"), dict(sim_hash="sim-1e10000000000000")], p)
        assert [r["sim_hash"] for r in table_io.read_rows(p)] == ["sim-1234567890123456", "sim-1e10000000000000"]
    rc = reduced_cfg()
    rows = fixture(200, 3, SEED, rc)
    for change in (dict(response_estimator="grid_mean"), dict(table_statistic="median")):
        cfg = dataclasses.replace(rc, correction=dataclasses.replace(rc.correction, **change))
        try:
            fit_table(rows, cfg)
        except NotImplementedError:
            continue
        raise AssertionError("unsupported estimator accepted: %r" % change)
    # the cross-validated multiplier: on pure noise (b constant) the most smoothing wins, at the grid's edge
    cv = reduced_cfg(bandwidth_cv_grid=(0.5, 1.0, 2.0))
    T = fit_table(fixture(400, 3, SEED + 11, cv, b_fn=lambda x1, x2, x3: 1.0 + 0.0 * x1), cv)
    assert T.meta["multiplier"] == 2.0 and T.meta["cv_at_grid_edge"] and len(T.meta["cv_scores"]) == 3, T.meta
    assert T.meta["cv_parameter"] == "bandwidth_multiplier" and T.meta["cv_grid"] == [0.5, 1.0, 2.0]
    # tps_spline, the small-table path (D-067.3): folds by draw, the extended smoothing grid, its cap
    tp = reduced_cfg(response_estimator="tps_spline")
    small = fixture(60, 3, SEED + 13, tp)
    Tt = fit_table(small, tp)
    assert Tt.meta["estimator"] == "tps_spline" and Tt.meta["se_method"] == "independent_rows"
    assert len(Tt.meta["cv_scores"]) == len(tp.correction.spline_smoothing_grid) == 11 and Tt.meta["smoothing"] > 0
    assert np.array_equal(Tt.arrays["se_hat"], Tt.arrays["se_indep"], equal_nan=True)
    assert np.array_equal(np.isfinite(Tt.arrays["b_hat"]), Tt.supported) and not Tt.supported.all()  # the floor
    # folds by draw (D-062.3): with a strong draw-level offset (rho 0.8) on a constant b the realizations of a draw
    # are near-copies; folds by draw see it and smooth (10 here; 1e4, the grid's end, on 4 other seeds) where folds
    # by row chose 0.001-0.01 (checks/table_2026-10-21/check_tps_folds.py; 0.01 on this seed)
    noisy = fixture(100, 5, SEED + 15, tp, rho=0.8, b_fn=lambda x1, x2, x3: 1.0 + 0.0 * x1, p_fail=lambda d: 0.0)
    Tn = fit_table(noisy, tp)
    assert Tn.meta["smoothing"] >= 1.0, (Tn.meta["smoothing"], Tn.meta["cv_scores"])
    lam, sc, edge = TB.choose_smoothing(np.random.default_rng(SEED).random((40, 3)), np.ones(40), np.arange(1, 41),
                                        (1.0, 10.0), 5)
    assert edge and sc.size == 2, "a two-value grid's choice is at an end"
    try:
        fit_table(small, dataclasses.replace(tp, correction=dataclasses.replace(tp.correction, tps_max_rows=50)))
    except ValueError as exc:
        assert "tps_max_rows" in str(exc), exc
    else:
        raise AssertionError("tps_spline accepted more rows than tps_max_rows")
    # failure_rate_ignore: the default () counts every row (D-057.1); a named screen removes the rows it rejected,
    # whatever else they failed; a row failing other screens, or with no reason given, is a failure
    cfg = reduced_cfg()
    assert default_config().correction.failure_rate_ignore == ()
    base = [dict(r, in_S=True, reject="") for r in rows]
    nan = float("nan")
    extra = [dict(draw_index=9000 + k, d_um=1.0, phi_rad=0.2, mu_per_um=1.0, d_hat_um=nan, mu_hat_per_um=nan,
                  ratio=nan, in_S=False, reject=why) for k, why in
             enumerate(("dark", "dark", "stack_edge;dark", "crossing", ""))]
    r0, r2 = TB.table_rows(base, cfg), TB.table_rows(base + extra, cfg)
    assert r2.X_fail.shape[0] == r0.X_fail.shape[0] + 5 and int(r2.failed.sum()) == int(r0.failed.sum()) + 5
    assert r2.n_rows == len(base) + 5 and r2.y.size == r0.y.size
    for named, left in ((("dark",), 2), (("stack_edge",), 4)):
        r1 = TB.table_rows(base + extra, dataclasses.replace(cfg, correction=dataclasses.replace(
            cfg.correction, failure_rate_ignore=named)))
        assert r1.X_fail.shape[0] == r0.X_fail.shape[0] + left, (named, r1.X_fail.shape[0] - r0.X_fail.shape[0])
    for bad in (lambda: fit_table(rows[:6], default_config()), lambda: table_inputs([0.0], [0.1], [1.0], [1.0]),
                lambda: table_inputs([1.0], [np.nan], [1.0], [1.0]), lambda: table_inputs([1.0], [0.1], [0.0], [1.0]),
                lambda: table_inputs([1.0], [0.1], [np.inf], [1.0]), lambda: fit_table([], rc)):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("bad input accepted")


# ---------------------------------------------------------------- runner ---

def _environment():
    parts = ["python %s" % platform.python_version()]
    for package in REPORT_PACKAGES:
        try:
            parts.append("%s %s" % (package, importlib.metadata.version(package)))
        except importlib.metadata.PackageNotFoundError:
            parts.append("%s (not installed)" % package)
    parts.append(platform.platform())
    parts.append("seed %d" % SEED)
    return " | ".join(parts)


def main():
    checks = [(name, obj) for name, obj in globals().items()
              if name.startswith("test_") and callable(obj)]
    print("== %s" % Path(__file__).name)
    print(_environment())
    results = []
    for name, func in checks:
        start = time.perf_counter()
        try:
            func()
            status, detail = "PASS", ""
        except unittest.SkipTest as exc:
            status, detail = "SKIP", str(exc)
        except NotImplementedError as exc:
            status, detail = "TODO", str(exc)
        except AssertionError as exc:
            status, detail = "FAIL", str(exc)
        except Exception:
            status, detail = "ERROR", traceback.format_exc()
        results.append((name, status, time.perf_counter() - start, detail))
    width = max([len(n) for n, _, _, _ in results] + [4])
    for name, status, seconds, detail in results:
        lines = detail.strip().splitlines()
        headline = lines[0] if lines else ""
        print(("%-5s  %-" + str(width) + "s  %8.3fs  %s") % (status, name, seconds, headline))
    for name, status, _s, detail in results:
        if status in ("FAIL", "ERROR"):
            print("\n---- %s: %s\n%s" % (status, name, detail.strip()))
    counts = {s: sum(1 for r in results if r[1] == s) for s in ("PASS", "FAIL", "ERROR", "TODO", "SKIP")}
    print("\n-- " + ", ".join("%d %s" % (n, s.lower()) for s, n in counts.items()))
    return 1 if (counts["FAIL"] + counts["ERROR"] + counts["TODO"]) else 0


if __name__ == "__main__":
    sys.exit(main())
