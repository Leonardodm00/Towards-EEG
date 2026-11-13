"""Spec-derived tests of Block 14 (the design, the coordinates, the local
linear estimator, the inversion, the jobs) -- integration tester, run D3,
2026-11-13. Expected values come from specs/SPEC.md (section 2.4, Block 14,
Block 7's inversion rule), from NumPy / SciPy or by hand; the code under test
only produces the outputs that are checked.

Run (from the workstream directory):
    python -m pytest tests/spec/test_spec_block14_table.py -q
No rendering: the rows are synthetic (seconds to a minute).
"""
from __future__ import annotations

import csv
import dataclasses
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import stats

import spec_helpers_D3 as H

from allen_diameter import config as C
from allen_diameter.analysis import invert, node_pipeline as NP, phantoms, table
from allen_diameter.analysis.cell import node_row
from allen_diameter.config import config_from_dict, default_config
from allen_diameter.loading import table_io

CFG = default_config()
WS = str(H.WS)
NAN = float("nan")


# ----------------------------------------------------- 14.1 selection ----

def _res(**kw):
    """A NodeResult with every required field (values irrelevant to the checks unless given)."""
    base = dict(node_id=1, type=3, x_um=0.0, y_um=0.0, z_um=0.0, path_um=0.0, reg_verdict="", s_star_um=NAN,
                dz_star_um=NAN, k_star=0, z_sub_um=0.0, cx_um=0.0, cy_um=0.0, cz_um=0.0, theta_rad=0.0, phi_rad=0.2,
                steep=False, vertical=False, B_bar=200.0, B_bar_region="block_masked", d_hat_um=1.0,
                mu_hat_per_um=1.0, v0_hat_um=0.0, alpha_hat=1.0, fit_status="converged", flags=(), fit=None,
                focus_rule="gradient_energy", focus_planes=np.arange(3), focus_score=np.ones(3),
                focus_depth=np.ones(3), k_star_depth=0)
    base.update(kw)
    return NP.NodeResult(**base)


def test_selection_flags_and_reasons():
    # O14.1 (D-030, D-096): S's flags; dark and faint are labels; the status, the registration, the flags in order.
    assert C.SELECTION_FLAGS == ("crossing", "stack_edge", "bbar_few", "profile_nan")
    assert phantoms.SELECTION_FLAGS is C.SELECTION_FLAGS
    assert CFG.correction.failure_rate_ignore == ()
    assert CFG.estimator_signature()["selection"] == {"flags": list(C.SELECTION_FLAGS)}
    assert phantoms.reject_reasons(_res(flags=("dark", "faint", "steep", "vertical", "iter_cycle"))) == []
    assert phantoms.reject_reasons(_res(flags=("profile_nan", "faint", "crossing"))) == ["crossing", "profile_nan"]
    assert phantoms.reject_reasons(_res(fit_status="at_bound", flags=("bbar_few",))) == ["status:at_bound", "bbar_few"]
    assert phantoms.reject_reasons(_res(fit_status="none", reg_verdict="FAR: the trace is 3 um away")) \
        == ["status:none", "registration:FAR"]
    assert phantoms.reject_reasons(_res(reg_verdict="ON THE PROCESS (lateral offset +0.00 um)")) == []


# -------------------------------------------------------- 14.2 design ----

def test_draws_recomputed_from_numpy():
    # O14.2: D-062's keys and orders, recomputed from NumPy alone (features, nuisances, the returned generator's
    # position, the radius); exact up to the last bit of exp(log) (1e-15 relative).
    ph, acq = CFG.phantom, CFG.acquisition
    for seed in (20261006, 7):
        for g in (1, 2, 17, 99999):
            for r in (1, 2, 10):
                dr, rng = phantoms.draw_replicate(CFG, seed, g, r)
                d, phi, th, mu = H.np_features(seed, g, ph)
                cx, cy, cz, ref = H.np_nuisances(seed, g, r, acq.res0_um, acq.dz_um)
                rb, k = H.np_radius(seed, g, r, d, ph.radius_noise_sd_um)
                assert_allclose([dr.d_um, dr.phi_rad, dr.theta_rad, dr.mu_per_um], [d, phi, th, mu], rtol=1e-15)
                assert_allclose([dr.cx_um, dr.cy_um, dr.cz_um], [cx, cy, cz], rtol=1e-15, atol=0)
                assert dr.radius_um == rb and dr.radius_redraws == k
                assert (dr.draw_index, dr.realization_index, dr.seed) == (g, r, seed)
                assert rng.standard_normal() == ref.standard_normal()          # positioned for the jitter


def test_d_override_draws_the_radius_about_it():
    # O14.2: with d_override_um the radius is drawn about the overridden d (the same key [seed, g, r, 1]); the other
    # features and the nuisances are the draw's.
    dr = phantoms.draw_replicate(CFG, 777, 5, 1, d_override_um=1.7)[0]
    base = phantoms.draw_replicate(CFG, 777, 5, 1)[0]
    rb, k = H.np_radius(777, 5, 1, 1.7, CFG.phantom.radius_noise_sd_um)
    assert dr.d_um == 1.7 and dr.radius_um == rb and dr.radius_redraws == k
    assert (dr.phi_rad, dr.theta_rad, dr.mu_per_um, dr.cx_um, dr.cy_um, dr.cz_um) == \
        (base.phi_rad, base.theta_rad, base.mu_per_um, base.cx_um, base.cy_um, base.cz_um)


def test_draw_indices_order_and_sharing():
    # O14.2: index 0 (or not an integer >= 1) refused; reproducible whatever the order; realizations share the
    # features and differ in the nuisances; the three streams distinct for g, r in {1, 2, 3}.
    for g, r in ((0, 1), (1, 0), (-1, 1), (1.5, 1), (True, 1)):
        with pytest.raises(ValueError):
            phantoms.draw_replicate(CFG, 1, g, r)
    a = phantoms.draw_replicate(CFG, 5, 3, 2)[0]
    for g in (9, 1, 3):
        phantoms.draw_replicate(CFG, 5, g, 1)
    assert phantoms.draw_replicate(CFG, 5, 3, 2)[0] == a
    firsts = set()
    for g in (1, 2, 3):
        ds = [phantoms.draw_replicate(CFG, 5, g, r)[0] for r in (1, 2, 3)]
        assert len({(x.d_um, x.phi_rad, x.theta_rad, x.mu_per_um) for x in ds}) == 1
        assert len({(x.cx_um, x.cy_um, x.cz_um) for x in ds}) == 3 and len({x.radius_um for x in ds}) == 3
        firsts.add(ds[0].d_um)
        for x in ds:
            firsts.update({x.cx_um, x.radius_um})
    assert len(firsts) == 3 + 9 + 9


def test_radius_redraw_law():
    # O14.2: at d = 0.2 um, sd 0.2 um the kept eps follows N(0, sd^2) truncated below at -d/2 (KS against
    # scipy.stats.truncnorm, 4000 draws, p > 0.01); the mean number of rejections per kept draw is
    # Phi(-d/(2 sd)) / (1 - Phi(-d/(2 sd))) within 4 SE (the redraw count is geometric).
    d, sd = 0.2, CFG.phantom.radius_noise_sd_um
    eps, rej = [], []
    for g in range(1, 4001):
        rb, k = phantoms.branch_radius(CFG, 11, g, 1, d)
        eps.append(rb - d / 2)
        rej.append(k)
    eps, rej = np.array(eps), np.array(rej)
    assert np.all(eps > -d / 2)
    p = stats.kstest(eps, stats.truncnorm(-d / 2 / sd, np.inf, loc=0, scale=sd).cdf).pvalue
    assert p > 0.01, p
    q = stats.norm.cdf(-d / (2 * sd))
    assert abs(rej.mean() - q / (1 - q)) < 4 * rej.std(ddof=1) / math.sqrt(rej.size)
    zero = dataclasses.replace(CFG, phantom=dataclasses.replace(CFG.phantom, radius_noise_sd_um=0.0))
    assert phantoms.branch_radius(zero, 11, 3, 2, 0.7) == (0.35, 0)
    dr = phantoms.draw_replicate(zero, 11, 3, 2)[0]
    assert dr.radius_um == dr.d_um / 2 and dr.radius_redraws == 0


def test_phantom_branch_and_tube():
    # O14.2: every node carries radius_um; nodes every phantom_node_step_um along the axis through c, the middle
    # one at c; the tube's radius d/2.
    dr, rng = phantoms.draw_replicate(CFG, 3, 4, 5)
    br = phantoms.phantom_branch(dr, CFG, rng)
    ph = CFG.phantom
    assert np.all(br.radius_um == dr.radius_um) and len(br.ids) == 2 * ph.nodes_each_way + 1
    t = np.array([math.cos(dr.phi_rad) * math.cos(dr.theta_rad), math.cos(dr.phi_rad) * math.sin(dr.theta_rad),
                  math.sin(dr.phi_rad)])
    c = np.array([dr.cx_um, dr.cy_um, dr.cz_um])
    ref = c[None, :] + (np.arange(-ph.nodes_each_way, ph.nodes_each_way + 1) * ph.phantom_node_step_um)[:, None] * t
    assert_allclose(br.xyz_um, ref, atol=1e-12)
    tube = phantoms.phantom_tube(dr, CFG)
    assert tube.r == 0.5 * dr.d_um


def test_grid_design_cells():
    # O14.2: the grid design visits cell (g - 1) mod (n_d n_phi), then draws theta and mu.
    cfg = dataclasses.replace(CFG, phantom=dataclasses.replace(CFG.phantom, design="grid"))
    ph = cfg.phantom
    nd, nphi = len(ph.grid_d_um), len(ph.grid_phi_deg)
    for g in (1, 2, 11, nd * nphi, nd * nphi + 1, 257):
        cell_ = (g - 1) % (nd * nphi)
        d, phi, th, mu = phantoms.draw_features(cfg, 13, g)
        assert d == ph.grid_d_um[cell_ // nphi] and phi == math.radians(ph.grid_phi_deg[cell_ % nphi])
        rng = np.random.default_rng([13, g])
        assert th == rng.uniform(*ph.theta_range_rad)
        assert mu == math.exp(rng.uniform(math.log(ph.mu_range_per_um[0]), math.log(ph.mu_range_per_um[1])))


# --------------------------------------------------- 14.3 coordinates ----

def test_table_inputs():
    # O14.3
    X = table.table_inputs([0.5, 2.0], 0.3, [1.2, 0.4], [0.6, 2.2])
    assert_allclose(X, [[math.log(0.5), 0.3, math.log(0.72)], [math.log(2.0), 0.3, math.log(0.88)]], rtol=1e-15)
    assert X.shape == (2, 3)
    for args in ((0.0, 0.1, 1, 1), (-1, 0.1, 1, 1), (np.inf, 0.1, 1, 1), (NAN, 0.1, 1, 1), (1, NAN, 1, 1),
                 (1, np.inf, 1, 1), (1, 0.1, 0.0, 1), (1, 0.1, -1, 1), (1, 0.1, NAN, 1), (1, 0.1, np.inf, 1),
                 (1, 0.1, 1, 0.0)):
        with pytest.raises(ValueError):
            table.table_inputs(*args)


def _rows(G=120, n_real=3, seed=1, fail=0.1, linear=False, measured_noise=0.0, cfg=CFG):
    """Synthetic replicate rows of the D-059 design: b = 1 + 0.06/d + 0.15 sin^2 phi + 0.05 ln(mu d) (section 8's
    fixture), a draw-level and a realization-level noise, mu_hat d_hat varying within a draw."""
    rng = np.random.default_rng(seed)
    ph = cfg.phantom
    out = []
    for g in range(1, G + 1):
        d, phi, th, mu = H.np_features(seed, g, ph)
        off = rng.normal(0, 0.03)
        for r in range(1, n_real + 1):
            mu_hat = mu * math.exp(rng.normal(0, 0.1))
            meas = phi + rng.normal(0, measured_noise) if measured_noise else phi
            if linear:                                   # linear in the table's coordinates (the measured tilt)
                y = 1.0 + 0.1 * math.log(d) - 0.2 * meas + 0.05 * math.log(mu_hat * d)
            else:
                y = (1 + 0.06 / d + 0.15 * math.sin(phi) ** 2 + 0.05 * math.log(mu * d)) * (1 + off) \
                    + rng.normal(0, 0.02)
            d_hat = d * y
            if linear:
                mu_hat = mu_hat * d / d_hat                      # so that mu_hat d_hat is the value used above
            bad = rng.random() < fail
            out.append(dict(draw_index=g, realization_index=r, d_um=d, phi_rad=phi,
                            meas_phi_rad=meas, theta_rad=th, mu_per_um=mu,
                            d_hat_um=d_hat, mu_hat_per_um=mu_hat, ratio=d_hat / d, in_S=not bad,
                            reject="crossing" if bad else ""))
    return out


def test_table_rows_and_failures():
    # O14.3: rows in S give X, y, draw; every counted row gives X_fail at the true coordinates; a row in S without
    # a finite mu_hat d_hat > 0 counts as a failure; a screen of failure_rate_ignore removes the rows it rejected;
    # a list of dicts and a dict of columns give the same.
    rows = _rows(G=10, n_real=2, fail=0.3)
    rows[0]["mu_hat_per_um"] = NAN
    rows[1]["in_S"] = True
    rows[1]["mu_hat_per_um"] = 0.0
    R = table.table_rows(rows, CFG)
    ok = [r["in_S"] and np.isfinite(r["mu_hat_per_um"]) and r["mu_hat_per_um"] > 0 for r in rows]
    S = [r for r, k in zip(rows, ok) if k]
    assert_allclose(R.X, [[math.log(r["d_um"]), r["phi_rad"], math.log(r["mu_hat_per_um"] * r["d_hat_um"])]
                          for r in S], rtol=1e-15)
    assert_allclose(R.y, [r["ratio"] for r in S])
    assert_array_equal(R.draw, [r["draw_index"] for r in S])
    assert_allclose(R.X_fail, [[math.log(r["d_um"]), r["phi_rad"], math.log(r["mu_per_um"] * r["d_um"])]
                               for r in rows], rtol=1e-15)
    assert_array_equal(R.failed, [not k for k in ok])
    assert R.n_rows == 20 and R.n_draws_rows == 10
    cols = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    R2 = table.table_rows(cols, CFG)
    assert_array_equal(R2.X, R.X)
    assert_array_equal(R2.failed, R.failed)
    ign = dataclasses.replace(CFG, correction=dataclasses.replace(CFG.correction, failure_rate_ignore=("crossing",)))
    R3 = table.table_rows(rows, ign)
    keep = [k or "crossing" not in r["reject"] for r, k in zip(rows, ok)]
    assert R3.X_fail.shape[0] == sum(keep)


# ----------------------------------------------------- 14.4 estimator ----

def _expected_axes(X, cfg, c):
    """The grid of D-071.3 with the implemented extent (a): each axis spans the design's range joined with the
    0.1-99.9 percentile range of the rows' coordinate, n = ceil(L / (c h / 2)) + 1."""
    ph, cc = cfg.phantom, cfg.correction
    (dlo, dhi), (mlo, mhi) = ph.d_range_um, ph.mu_range_per_um
    rng_ = [(math.log(dlo), math.log(dhi)), (math.radians(ph.phi_range_deg[0]), math.radians(ph.phi_range_deg[1])),
            (math.log(mlo * dlo), math.log(mhi * dhi))]
    h = [c * cc.tau_kernel_bandwidth[0], c * math.radians(cc.tau_kernel_bandwidth[1]), c * cc.tau_kernel_bandwidth[2]]
    axes = []
    for k, (lo, hi) in enumerate(rng_):
        qlo, qhi = np.percentile(X[:, k], (0.1, 99.9))
        lo, hi = min(lo, qlo), max(hi, qhi)
        axes.append(np.linspace(lo, hi, math.ceil((hi - lo) / (0.5 * h[k]) - 1e-9) + 1))
    return axes, np.array(h)


@pytest.fixture(scope="module")
def fitted():
    rows = _rows(G=300, n_real=3, seed=3)
    R = table.table_rows(rows, CFG)
    T = table.build_table(R, CFG, "local_linear", 1.0)
    return rows, R, T


def _inside(T, X):
    """X clamped strictly inside the axes (1e-12), as (d, phi, mu d) query arguments."""
    lo = np.array([a[0] for a in T.axes]) + 1e-12
    hi = np.array([a[-1] for a in T.axes]) - 1e-12
    Xc = np.clip(X, lo, hi)
    return np.exp(Xc[:, 0]), Xc[:, 1], np.exp(Xc[:, 2])


def test_grid_axes_rule(fitted):
    # O14.4: the axes by the spec's rule; (41, 37, 37) at the defaults when the rows stay inside the band.
    rows, R, T = fitted
    axes, _ = _expected_axes(R.X, CFG, 1.0)
    for a, b in zip(T.axes, axes):
        assert_allclose(a, b, rtol=0, atol=1e-14)
    inside = R.X.copy()
    inside[:, 2] = np.clip(inside[:, 2], math.log(0.3 * 0.2) + 0.1, math.log(3.0 * 4.0) - 0.1)
    assert tuple(a.size for a in table.grid_axes(inside, CFG, 1.0)) == (41, 37, 37)


def test_local_linear_against_brute_force(fitted):
    # O14.4: at supported grid points b_hat, N_eff, tau_hat, the sandwich SE by draw, the independent-rows SE, the
    # mass and the failure rate equal their definitions evaluated row by row (1e-8 relative: normal equations
    # against lstsq on rcond >= 1e-12 systems); q_loo within 2e-3 (it reads M at the rows by the cubic interpolant,
    # whose error at spacing h/2 is ~ (1/2)^4 / 384 x 3 ~ 5e-4 of M).
    rows, R, T = fitted
    axes, h = _expected_axes(R.X, CFG, 1.0)
    e = R.y - T.b_hat(*_inside(T, R.X))
    rng = np.random.default_rng(0)
    sup = np.argwhere(T.supported)
    assert sup.shape[0] > 100
    n_checked = 0
    for i, j, k in sup[rng.choice(sup.shape[0], 60, replace=False)]:
        x = np.array([axes[0][i], axes[1][j], axes[2][k]])
        b, ell, w, A = H.brute_local_linear(R.X, R.y, x, h)
        if np.linalg.cond(A) > 1e8:          # normal equations vs lstsq agree to ~cond(A) eps: keep 1e-8 meaningful
            continue
        n_checked += 1
        assert abs(T.arrays["b_hat"][i, j, k] - b) <= 1e-8 * abs(b)
        neff = w.sum() ** 2 / (w * w).sum()
        assert abs(T.arrays["n_eff"][i, j, k] / neff - 1) < 1e-10
        tau = math.sqrt((w * e * e).sum() / w.sum() * neff / (neff - 1))
        assert abs(T.arrays["tau_hat"][i, j, k] / tau - 1) < 1e-8
        assert abs(T.arrays["se_indep"][i, j, k] / (tau / math.sqrt(neff)) - 1) < 1e-8
        var = sum(np.sum(ell[R.draw == g] * e[R.draw == g]) ** 2 for g in np.unique(R.draw))
        assert abs(T.arrays["se_hat"][i, j, k] / math.sqrt(var) - 1) < 1e-8
        assert abs(sum(ell) - 1) < 1e-10 and np.max(np.abs(ell @ (R.X - x))) < 1e-10      # weights' identities
        U = (R.X_fail - x) / h
        wf = np.exp(-0.5 * np.sum(U * U, axis=1))
        assert abs(T.arrays["mass"][i, j, k] / wf.sum() - 1) < 1e-10
        assert abs(T.arrays["p_fail"][i, j, k] - (wf * R.failed).sum() / wf.sum()) < 1e-10
    assert n_checked >= 25, n_checked
    M_rows = np.array([np.exp(-0.5 * np.sum(((R.X_fail - x) / h) ** 2, axis=1)).sum() for x in R.X_fail])
    q = np.percentile(M_rows - 1.0, CFG.correction.loo_mass_percentile)
    assert abs(T.q_loo / q - 1) < 2e-3, (T.q_loo, q)


def test_support_floor(fitted):
    # O14.4: where N_eff < min_neff (or A is near singular) the arrays of the rows in S are NaN, the mass and the
    # failure rate are kept.
    rows, R, T = fitted
    axes, h = _expected_axes(R.X, CFG, 1.0)
    neff = np.empty(T.supported.shape)
    U = [((R.X[:, k][None, :] - axes[k][:, None]) / h[k]) for k in range(3)]
    E = [np.exp(-0.5 * u * u) for u in U]
    sw = np.einsum("in,jn,kn->ijk", E[0], E[1], E[2])
    sw2 = np.einsum("in,jn,kn->ijk", E[0] ** 2, E[1] ** 2, E[2] ** 2)
    neff = sw ** 2 / sw2
    low = neff < CFG.correction.min_neff
    assert low.any() and not T.supported[low].any()
    for name in table.S_ARRAYS:
        assert np.all(np.isnan(T.arrays[name][~T.supported]))
        assert np.all(np.isfinite(T.arrays[name][T.supported]))
    assert np.all(np.isfinite(T.arrays["mass"]))
    assert np.all(np.isfinite(T.arrays["p_fail"][T.arrays["mass"] > 0]))


def test_linear_response_reproduced():
    # O14.4: the local linear fit reproduces a noise-free linear b at every supported grid point (1e-10), under
    # both tilt axes; the sandwich SE is then 0 up to rounding.
    for axis in ("true", "measured"):
        cfg = dataclasses.replace(CFG, correction=dataclasses.replace(CFG.correction, table_phi_axis=axis))
        rows = _rows(G=150, n_real=3, seed=4, fail=0.0, linear=True, measured_noise=0.01 if axis == "measured" else 0)
        R = table.table_rows(rows, cfg)
        T = table.build_table(R, cfg, "local_linear", 1.0)
        L1, L2, L3 = np.meshgrid(*T.axes, indexing="ij")
        ref = 1.0 + 0.1 * L1 - 0.2 * L2 + 0.05 * L3
        s = T.supported
        assert np.max(np.abs(T.arrays["b_hat"][s] - ref[s])) < 1e-10, axis
        # (the SE on this noise-free response is not 0 near the support's edge: finding of the report, its reproducer)


def test_sandwich_measured_axis_against_brute_force():
    # O14.4: under table_phi_axis "measured" (the sandwich by pairs) the SE equals its definition (1e-8).
    cfg = dataclasses.replace(CFG, correction=dataclasses.replace(CFG.correction, table_phi_axis="measured"))
    rows = _rows(G=300, n_real=3, seed=6, measured_noise=0.02)
    R = table.table_rows(rows, cfg)
    T = table.build_table(R, cfg, "local_linear", 1.0)
    axes, h = _expected_axes(R.X, cfg, 1.0)
    e = R.y - T.b_hat(*_inside(T, R.X))
    sup = np.argwhere(T.supported)
    n_checked = 0
    for i, j, k in sup[np.random.default_rng(1).choice(sup.shape[0], 40, replace=False)]:
        x = np.array([axes[0][i], axes[1][j], axes[2][k]])
        b, ell, w, A = H.brute_local_linear(R.X, R.y, x, h)
        if np.linalg.cond(A) > 1e8:
            continue
        n_checked += 1
        var = sum(np.sum(ell[R.draw == g] * e[R.draw == g]) ** 2 for g in np.unique(R.draw))
        assert abs(T.arrays["se_hat"][i, j, k] / math.sqrt(var) - 1) < 1e-8
    assert n_checked >= 10, n_checked


def test_cross_validation_choice():
    # O14.4: the multiplier is the smallest CV score's, the scores are stored, a choice at either end is flagged;
    # one grid value fixes c; folds by draw (g - 1) mod K.
    assert_array_equal(table.fold_of_draw([1, 2, 5, 6, 11], 5), [0, 1, 4, 0, 0])
    rows = _rows(G=150, n_real=3, seed=8)
    T = table.fit_table(rows, CFG)
    grid = list(CFG.correction.bandwidth_cv_grid)
    sc = T.meta["cv_scores"]
    assert len(sc) == len(grid) and T.meta["multiplier"] == grid[int(np.argmin(sc))]
    assert T.meta["cv_at_grid_edge"] == (int(np.argmin(sc)) in (0, len(grid) - 1))
    one = dataclasses.replace(CFG, correction=dataclasses.replace(CFG.correction, bandwidth_cv_grid=(1.4142,)))
    T1 = table.fit_table(rows, one)
    assert T1.meta["multiplier"] == 1.4142 and T1.meta["cv_scores"] == [] and T1.meta["cv_at_grid_edge"] is False


def test_interpolant_and_file_round_trip(fitted):
    # O14.4: the interpolant reproduces the grid (1e-12 at the nodes); NaN outside the grid; save/load is exact; a
    # table file without the format key is refused; the per-draw table by hand.
    rows, R, T = fitted
    s = np.argwhere(T.supported)[:200]
    q = (np.exp(T.axes[0][s[:, 0]]), T.axes[1][s[:, 1]], np.exp(T.axes[2][s[:, 2]]))
    ok = (np.log(q[0]) == T.axes[0][s[:, 0]]) & (np.log(q[2]) == T.axes[2][s[:, 2]])   # exact exp/log round trips
    got = T.b_hat(*q)
    assert_allclose(got[ok], T.arrays["b_hat"][tuple(s[ok].T)], rtol=1e-12)
    assert np.isnan(T.b_hat(10.0, 0.3, 1.0)) and np.isnan(T.b_hat(1.0, 2.0, 1.0))
    with tempfile.TemporaryDirectory() as tmp:
        stem = os.path.join(tmp, "t")
        table_io.save_table(T, stem)
        U = table_io.load_table(stem)
        for k in table.ARRAYS:
            assert_array_equal(U.arrays[k], T.arrays[k])
        assert_array_equal(U.supported, T.supported)
        for a, b in zip(U.axes, T.axes):
            assert_array_equal(a, b)
        assert U.q_loo == T.q_loo and U.estimator_hash == T.estimator_hash
        x = (np.array([0.5, 1.3]), np.array([0.2, 0.9]), np.array([0.5, 2.0]))
        assert_array_equal(U.b_hat(*x), T.b_hat(*x))
        with open(stem + ".json") as f:
            meta = json.load(f)
        meta.pop("format")
        with open(stem + ".json", "w") as f:
            json.dump(meta, f)
        with pytest.raises(ValueError):
            table_io.load_table(stem)
    per = table.per_draw_table(rows, CFG)
    g1 = [r for r in rows if r["draw_index"] == 1]
    ys = [r["ratio"] for r in g1 if r["in_S"]]
    assert per[0]["draw_index"] == 1 and per[0]["n_rows"] == len(g1) and per[0]["n_S"] == len(ys)
    assert_allclose(per[0]["y_mean"], np.mean(ys)) if ys else None
    assert_allclose(per[0]["failure_share"], 1 - len(ys) / len(g1))
    assert len(per) == 300


# ----------------------------------------------------- 14.5 inversion ----

AX = (np.linspace(math.log(0.2), math.log(4.0), 41), np.linspace(0.0, math.pi / 2, 37),
      np.linspace(math.log(0.06), math.log(12.0), 37))


def test_inversion_root_and_flags():
    # O14.5: the root of d b_hat = d_hat; m_hat(d_tilde) = d_hat (1e-9); b = 1 + 0.1/d gives d_tilde = d_hat - 0.1
    # within the cubic interpolant's error (~1e-7 um here); the flags by their rules.
    T = H.grid_table(AX, lambda l1, l2, l3: 1.0 + 0.1 * np.exp(-l1))
    for d_hat in (0.7, 1.0, 2.7):                      # roots 0.6, 0.9, 2.6: |b - 1| <= 0.17, no large_correction
        inv = invert.invert_node(d_hat, 0.3, 1.0, T, CFG)
        assert inv.flags == () and abs(inv.d_tilde_um - (d_hat - 0.1)) < 1e-6
        assert abs(float(T.m_hat(inv.d_tilde_um, 0.3, d_hat)) - d_hat) < 1e-9 and inv.d_root_um == inv.d_tilde_um
        assert abs(invert.shortcut(d_hat, 0.3, 1.0, T) - d_hat / float(T.b_hat(d_hat, 0.3, d_hat))) < 1e-15
    for args in ((NAN, 0.3, 1.0), (0.0, 0.3, 1.0), (-1.0, 0.3, 1.0), (1.0, NAN, 1.0), (1.0, 0.3, 0.0),
                 (1.0, 0.3, NAN), (1.0, 0.3, -2.0), (np.inf, 0.3, 1.0)):
        inv = invert.invert_node(*args, T, CFG)
        assert inv.flags == ("no_estimate",) and math.isnan(inv.d_tilde_um)
    inv = invert.invert_node(1.0, 1.7, 1.0, T, CFG)                        # tilt beyond the axis: filled
    assert inv.flags == ("phi_out_of_domain",) and math.isnan(inv.d_tilde_um) and np.isfinite(inv.d_root_um)
    inv = invert.invert_node(1.0, 0.3, 40.0, T, CFG)                       # mu d beyond the axis: a label
    assert inv.flags == ("mu_d_out_of_domain",) and abs(inv.d_tilde_um - 0.9) < 1e-6
    inv = invert.invert_node(9.0, 0.3, 1.0, T, CFG)
    assert inv.flags == ("out_of_domain",) and math.isnan(inv.d_tilde_um) and math.isnan(inv.d_root_um)
    # m = d (1 + 0.5 sin(8 ln d)) = 1 has several roots in [0.67, 2] (by hand: 1/d spans [0.5, 1.5] while the sine
    # runs through more than one period of ln d there)
    Tn = H.grid_table(AX, lambda l1, l2, l3: 1.0 + 0.5 * np.sin(8 * l1))
    grid = np.exp(np.linspace(math.log(0.2), math.log(4.0), 256))
    m = grid * (1 + 0.5 * np.sin(8 * np.log(grid))) - 1.0
    assert np.count_nonzero(np.diff(np.sign(m))) > 1
    assert invert.invert_node(1.0, 0.3, 1.0, Tn, CFG).flags == ("non_monotone",)
    Tl = H.grid_table(AX, lambda l1, l2, l3: 1.3 + 0 * l1)
    inv = invert.invert_node(1.3, 0.3, 1.0, Tl, CFG)
    assert inv.flags == ("large_correction",) and math.isnan(inv.d_tilde_um) and abs(inv.d_root_um - 1.0) < 1e-9
    Tf = H.grid_table(AX, lambda l1, l2, l3: 1.0 + 0 * l1, p_fail=0.5 * np.ones((41, 37, 37)))
    assert invert.invert_node(1.0, 0.3, 1.0, Tf, CFG).flags == ("high_failure",)
    Tm = H.grid_table(AX, lambda l1, l2, l3: 1.0 + 0 * l1, mass=np.full((41, 37, 37), 3.0), q_loo=5.0)
    assert invert.invert_node(1.0, 0.3, 1.0, Tm, CFG).flags == ("off_support",)
    sup = np.ones((41, 37, 37), bool)
    i0 = int(np.argmin(np.abs(AX[0] - math.log(1.0))))
    sup[i0 - 1:i0 + 2, :, :] = False
    Ts = H.grid_table(AX, lambda l1, l2, l3: 1.0 + 0 * l1, supported=sup)
    assert invert.invert_node(1.0, 0.3, 1.0, Ts, CFG).flags == ("off_support",)


def test_mu_d_clamp_inverts_at_any_axis_end():
    # O14.5 (D-071.4): a node whose ln(mu_hat d_hat) lies outside the table's third axis is clamped to the axis'
    # nearest end, inverted there and not filled for it -- whatever floating-point value the end has. The third
    # axis' ends are set by the rows' 0.1 / 99.9 percentiles (14.4 (a)), arbitrary floats; with narrow phantom
    # ranges (a reduced table) they lie in (-1, 1), where about half of the floats a have log(exp(a)) != a
    # (52 % of 100,000 sampled; none of 100,000 with 1 <= |a| <= 6).
    cand = np.random.default_rng(2).uniform(-1.0, 1.0, 4000)
    lo_end = next(a for a in cand if a < 0.0 and math.log(math.exp(a)) < a)     # its exp/log round trip falls below
    hi_end = next(a for a in cand if a > 0.3 and math.log(math.exp(a)) > a)     # ... above
    ax = (AX[0], AX[1], np.linspace(lo_end, hi_end, 37))
    T = H.grid_table(ax, lambda l1, l2, l3: 1.0 + 0.1 * np.exp(-l1))
    for mu_hat in (0.5 * math.exp(lo_end), 3.0 * math.exp(hi_end)):      # d_hat = 1: mu d below / above the axis
        inv = invert.invert_node(1.0, 0.3, mu_hat, T, CFG)
        assert inv.flags == ("mu_d_out_of_domain",) and abs(inv.d_tilde_um - 0.9) < 1e-6, (mu_hat, inv)


def test_correct_nodes():
    # O14.5: another estimator signature is refused; a node outside S carries its reasons and is not inverted;
    # a dark and faint node in S is inverted and keeps its labels in the per-node row; the node's own mu_hat is used.
    h = CFG.signature_hash("estimator")
    T = H.grid_table(AX, lambda l1, l2, l3: 1.0 + 0.1 * l3 + 0 * l1, estimator_hash=h)
    with pytest.raises(ValueError):
        invert.correct_nodes([_res()], H.grid_table(AX, lambda a, b, c: 1 + 0 * a, estimator_hash="other"), CFG)
    nodes = [_res(flags=("dark", "faint"), d_hat_um=1.2, mu_hat_per_um=2.0),
             _res(flags=("crossing",)), _res(fit_status="failed"),
             _res(d_hat_um=1.2, mu_hat_per_um=0.5)]
    out = invert.correct_nodes(nodes, T, CFG)
    assert np.isfinite(out[0].d_tilde_um) and out[0].flags == ()
    assert out[0].d_tilde_um == invert.invert_node(1.2, 0.2, 2.0, T, CFG).d_tilde_um
    assert out[1].flags == ("crossing",) and math.isnan(out[1].d_tilde_um)
    assert out[2].flags == ("status:failed",) and math.isnan(out[2].d_tilde_um)
    assert abs(out[3].d_tilde_um - 1.2 / (1 + 0.1 * math.log(0.6))) < 1e-6
    assert abs(out[0].d_tilde_um - 1.2 / (1 + 0.1 * math.log(2.4))) < 1e-6
    row = node_row(nodes[0], out[0], out[0].d_tilde_um, "self", 0.3, 0.099)
    assert {"dark", "faint"} <= set(row["flags"].split(";")) and abs(row["mu_d_hat"] - 2.4) < 1e-15


# ------------------------------------------------------------ 14.6 jobs ----

def _write_rows(path, rows, sim):
    keys = list(rows[0]) + ["sim_hash"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(dict(r, sim_hash=sim))


def _merge(out_dir, *extra):
    return subprocess.run([sys.executable, "scripts/build_table.py", "merge", "--out-dir", out_dir, *extra], cwd=WS,
                          capture_output=True, text=True, timeout=900)


def test_merge_outputs_and_refusals():
    # O14.6: merge writes the table and draws_<hash>.csv; refuses another sim_hash and a (draw, realization) seen
    # twice; notes the draws short of n_realizations.
    h = CFG.signature_hash("estimator")
    sim = table.row_provenance(CFG)
    rows = _rows(G=150, n_real=2, seed=9)
    short = [r for r in rows if not (r["draw_index"] == 7 and r["realization_index"] == 2)]
    with tempfile.TemporaryDirectory() as tmp:
        _write_rows(os.path.join(tmp, "rows_%s_000001_000076.csv" % h), [r for r in short if r["draw_index"] <= 75], sim)
        _write_rows(os.path.join(tmp, "rows_%s_000076_000151.csv" % h), [r for r in short if r["draw_index"] > 75], sim)
        p = _merge(tmp)
        assert p.returncode == 0, p.stdout + p.stderr
        assert "fewer than n_realizations" in p.stdout
        for name in ("bias_table_%s.npz" % h, "bias_table_%s.json" % h, "draws_%s.csv" % h):
            assert os.path.exists(os.path.join(tmp, name)), name
        T = table_io.load_table(os.path.join(tmp, "bias_table_%s" % h))
        assert T.meta["n_all"] == len(short) and T.estimator_hash == h
        with open(os.path.join(tmp, "draws_%s.csv" % h)) as f:
            assert sum(1 for _ in f) == 150 + 1
        _write_rows(os.path.join(tmp, "rows_%s_000007_000008.csv" % h), [r for r in rows if r["draw_index"] == 7], sim)
        p = _merge(tmp)
        assert p.returncode != 0 and "merge refused" in (p.stdout + p.stderr)
        os.remove(os.path.join(tmp, "rows_%s_000007_000008.csv" % h))
        _write_rows(os.path.join(tmp, "rows_%s_000200_000201.csv" % h),
                    [dict(rows[0], draw_index=200)], "sim-0000000000000000")
        p = _merge(tmp)
        assert p.returncode != 0 and "merge refused" in (p.stdout + p.stderr)


def _pbs(env):
    e = dict(os.environ, DIAM_DRYRUN="1", DIAM_CODE=WS, **{k: str(v) for k, v in env.items()})
    for k in ("CODE", "ROOT", "ENV_NAME"):
        e.pop(k, None)
    return subprocess.run(["bash", "scripts/pbs/build_table.pbs"], cwd=WS, env=e, capture_output=True, text=True,
                          timeout=60)


def test_pbs_chunking_by_draw():
    # O14.6: task i runs draws O + i c + 1 .. O + min((i + 1) c, N), c = ceil(N / T); start >= 1.
    for N, T, O in ((10, 3, 0), (10, 3, 5), (7, 7, 100), (5, 8, 0)):
        c = -(-N // T)
        for i in range(T):
            lo, hi = O + i * c + 1, O + min((i + 1) * c, N)
            p = _pbs(dict(DIAM_N_DRAWS=N, DIAM_N_TASKS=T, DIAM_DRAW_OFFSET=O, PBS_ARRAY_INDEX=i))
            assert p.returncode == 0, p.stderr
            if lo > hi:
                assert "nothing to do" in p.stdout
                continue
            m = re.search(r"--draw-start (\d+) --draw-stop (\d+)", p.stdout)
            assert m and (int(m.group(1)), int(m.group(2)) - 1) == (lo, hi), (N, T, O, i, p.stdout)


def test_gate2_configuration_new_schema():
    # O14.6: configs/gate2_reduced.json in the new schema.
    with open(os.path.join(WS, "configs", "gate2_reduced.json")) as f:
        cfg = config_from_dict(json.load(f))
    ph, cc = cfg.phantom, cfg.correction
    assert (ph.n_draws, ph.n_realizations, ph.radius_noise_sd_um) == (48, 5, 0.2)
    assert cc.failure_rate_ignore == () and cc.response_estimator == "local_linear"
    assert len(cc.tau_kernel_bandwidth) == 3 and len(cc.bandwidth_cv_grid) >= 1
    assert cc.min_neff > 0 and 0 <= cc.loo_mass_percentile <= 100 and cc.tps_max_rows >= 1
    d = json.load(open(os.path.join(WS, "configs", "gate2_reduced.json")))
    bad = json.loads(json.dumps(d))
    bad["phantom"]["n_replicates"] = 240
    with pytest.raises(ValueError):
        config_from_dict(bad)
