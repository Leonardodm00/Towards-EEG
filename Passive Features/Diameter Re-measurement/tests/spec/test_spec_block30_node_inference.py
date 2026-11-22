"""Spec-derived tests, Block 30 (the per-node fit) -- tester 2026-11-13.

Oracles (SPEC.md Block 30), with a forward model built here from Block 3's
tube_model.model_profile (one Gaussian whose width grows with |z_k - z_ax|):
noise-free recovery from a start 30 % off in d and mu [relative 1e-6; 1e-6
um on v0, z_ax]; on 300 seeded noisy sets (SD 3 gl, B = 210, d = 1 um) the
interval d +- se covers the truth in 60-76 %, the mean d within 3 SE, chi2_k/N
averages 1 within 10 %; cov = (J^T J)^-1 against an independent central-
difference Jacobian [relative 1e-3]; select_planes by hand; at_bound; a NaN
forward ends failed; planes all False raises; determinism; inputs unchanged.
Boundary 26 -> 30: line_profiles as `forward` on a noise-free lone node.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import nc_helpers as H
from allen_diameter.analysis import node_inference as NI
from allen_diameter.model.tube_model import model_profile

B, SD = 210.0, 3.0
KS = np.array([-2, -1, 0, 1, 2])
DZ = 0.28
V = np.arange(-20, 21) * 0.1144


def forward(d, mu, v0, z):
    out = []
    for k in KS:
        s = math.sqrt(0.12 ** 2 + (0.9 * (k * DZ - z)) ** 2)
        out.append(model_profile(V, d, mu * d, v0, s, 1.0, 96))
    return np.stack(out)


TRUTH = np.array([1.0, 0.8, 0.07, 0.05])


def test_noise_free_recovery():
    cfg = H.cfg_with()
    I = B * forward(*TRUTH)
    th0 = TRUTH * np.array([1.3, 0.7, 1.0, 1.0])
    fit = NI.fit_node(I, B, SD, forward, th0, cfg)
    print(fit.status, fit.theta)
    assert fit.status == "converged"
    assert_allclose(fit.theta[:2], TRUTH[:2], rtol=1e-6)
    assert_allclose(fit.theta[2:], TRUTH[2:], atol=1e-6)


def test_coverage_bias_chi2_cov():
    cfg = H.cfg_with()
    rng = np.random.default_rng(3030)
    I0 = B * forward(*TRUTH)
    cover, ds, chi = 0, [], []
    for it in range(300):
        I = I0 + rng.normal(0, SD, size=I0.shape)
        Ic = I.copy()
        fit = NI.fit_node(I, B, SD, forward, TRUTH, cfg, planes=np.ones(KS.size, bool))
        assert_array_equal(I, Ic)
        assert fit.status == "converged"
        cover += abs(fit.theta[0] - TRUTH[0]) <= fit.se[0]
        ds.append(fit.theta[0])
        chi.append(np.nanmean(fit.chi2_per_plane))
        if it == 0:   # independent Gauss-Newton covariance from central differences at theta_hat
            th = fit.theta
            J = np.empty((I.size, 4))
            for p in range(4):
                h = 1e-6 * max(1.0, abs(th[p]))
                e = np.zeros(4); e[p] = h
                J[:, p] = (-(B * forward(*(th + e))) + B * forward(*(th - e))).ravel() / (2 * h * SD)
            cov = np.linalg.inv(J.T @ J)
            assert_allclose(fit.se, np.sqrt(np.diag(cov)), rtol=1e-3)
    ds = np.asarray(ds)
    frac = cover / 300.0
    z = (ds.mean() - TRUTH[0]) / (ds.std(ddof=1) / math.sqrt(300))
    print("coverage %.3f, mean d %+.2f SE, mean chi2/N %.3f" % (frac, z, np.mean(chi)))
    assert 0.60 <= frac <= 0.76
    assert abs(z) <= 3.0
    assert abs(np.mean(chi) - 1.0) <= 0.10


def test_select_planes_by_hand_and_contract():
    cfg = H.cfg_with()
    tau0 = forward(*TRUTH)
    Bk = np.array([200.0, 205.0, 210.0, 215.0, 220.0])
    for sig in (1.0, 10.0, 60.0, 200.0):
        keep = NI.select_planes(tau0, Bk, sig, cfg)
        exp = Bk * np.max(1 - tau0, axis=1) >= cfg.inference.plane_snr_min * sig
        assert_array_equal(keep, exp)
    I = B * forward(*TRUTH)
    f1 = NI.fit_node(I, B, SD, forward, TRUTH * 1.05, cfg)
    f2 = NI.fit_node(I, B, SD, forward, TRUTH * 1.05, cfg)
    assert_array_equal(f1.theta, f2.theta) and f1.status == f2.status
    assert f1.cov.shape == (4, 4) and f1.se.shape == (4,) and f1.chi2_per_plane.shape == (KS.size,)


def test_at_bound_failed_and_errors():
    cfg = H.cfg_with()
    I = B * forward(7.0, 0.3, 0.0, 0.0)                      # d beyond the 6 um bound
    fit = NI.fit_node(I, B, SD, forward, (5.0, 0.3, 0.0, 0.0), cfg)
    assert fit.status == "at_bound" and "d_hi" in fit.at_bound
    assert np.all(np.isnan(fit.cov[0])) and np.all(np.isnan(fit.cov[:, 0]))
    fit = NI.fit_node(I, B, SD, lambda *a: np.full((KS.size, V.size), np.nan), TRUTH, cfg,
                      planes=np.ones(KS.size, bool))
    assert fit.status == "failed"
    with pytest.raises(ValueError):
        NI.fit_node(I, B, SD, forward, TRUTH, cfg, planes=np.zeros(KS.size, bool))


def test_boundary_line_render_as_forward():
    """Block 26 -> Block 30: line_profiles (linear, partition-free cheap mode) as `forward` of a lone flat node."""
    from allen_diameter.model import geometry
    from allen_diameter.model.line_render import line_profiles
    cfg = H.cfg_with({"inference.multistart_factors": [1.0], "inference.max_nfev": 60,
                      "renderer.fft_split_sigma_um": 50.0})
    th = 0.3
    yh, eu = np.array([-math.sin(th), math.cos(th)]), np.array([math.cos(th), math.sin(th)])
    o = np.array([0.0, 0.0])
    v = np.arange(-12, 13) * 0.1144
    ks = np.array([-1, 0, 1])

    def fwd(d, mu, v0, z):
        c = o + v0 * yh
        tube = geometry.Tube(c=(float(c[0]), float(c[1]), float(z)), r=0.5 * d, phi=0.0, theta=th, half_length=3.0)
        return line_profiles(tube, mu, ks, o, yh, eu, v, 0, 0.1144, cfg, absorption="linear")

    truth = np.array([0.5, 1.0, 0.04, 0.03])
    I = B * fwd(*truth)
    fit = NI.fit_node(I, B, SD, fwd, truth * np.array([1.15, 0.9, 1.0, 1.0]), cfg, planes=np.ones(3, bool))
    print("line-render forward fit:", fit.status, fit.theta, "nfev", fit.nfev, "cost", fit.cost)
    assert fit.status == "converged"
    assert_allclose(fit.theta[:2], truth[:2], rtol=1e-3)
    assert_allclose(fit.theta[2:], truth[2:], atol=1e-3)


def test_nan_forward_after_the_start_ends_failed():
    """SPEC Block 30: 'a forward returning NaN ends failed'. Here the forward is finite at the start
    and NaN from its second call on (inside the finite-difference Jacobian)."""
    cfg = H.cfg_with()
    I = B * forward(*TRUTH) + 1.0
    n = {"c": 0}

    def fwd(*a):
        n["c"] += 1
        out = forward(*a)
        return out * np.nan if n["c"] > 1 else out
    fit = NI.fit_node(I, B, SD, fwd, TRUTH * 1.2, cfg, planes=np.ones(KS.size, bool))
    assert fit.status == "failed"


# ---------------------------------------------------------------------------------------------------------------
# Tester 2026-11-22 (incremental at a749cc9): Block 30, amendment 2 (D-169.4, D-171.2) -- fit_node(weights,
# z_box_centre_um) and fit_node_recentred. Oracles: weights of ones = None [exact theta and cost, cov 1e-9];
# the weighted cost and chi2_k/N by their formulas; the sandwich A^-1 B A^-1 against an independent central-
# difference Jacobian; samples of zero weight do not matter [exact]; coverage of d +- se with centre weights
# [60-76 %, the entry's band]; the re-centred fit from 0.2 um off ends within 1e-3 um of the centre in <= 3 solves.
# ---------------------------------------------------------------------------------------------------------------

def _w(lam=0.7, vc=0.07):
    return np.exp(-np.abs(V - vc) / lam)


def test_weights_of_ones_equal_none():
    cfg = H.cfg_with()
    I = B * forward(*TRUTH) + np.random.default_rng(1).normal(0, SD, (KS.size, V.size))
    a = NI.fit_node(I, B, SD, forward, TRUTH * 1.1, cfg)
    b = NI.fit_node(I, B, SD, forward, TRUTH * 1.1, cfg, weights=np.ones(V.size))
    assert_array_equal(a.theta, b.theta)
    assert a.cost == b.cost
    assert_allclose(b.cov, a.cov, rtol=1e-9)
    assert_allclose(b.chi2_per_plane, a.chi2_per_plane, rtol=1e-12)


def test_zero_weight_samples_do_not_matter():
    cfg = H.cfg_with()
    rng = np.random.default_rng(2)
    I = B * forward(*TRUTH) + rng.normal(0, SD, (KS.size, V.size))
    w = (np.abs(V) < 1.5).astype(float)
    I2 = I.copy()
    I2[:, w == 0] += rng.uniform(-80, 80, (KS.size, int((w == 0).sum())))
    a = NI.fit_node(I, B, SD, forward, TRUTH * 1.1, cfg, weights=w)
    b = NI.fit_node(I2, B, SD, forward, TRUTH * 1.1, cfg, weights=w)
    assert_array_equal(a.theta, b.theta) and a.cost == b.cost


def test_weighted_cost_chi2_and_sandwich_by_hand():
    cfg = H.cfg_with()
    rng = np.random.default_rng(3)
    I = B * forward(*TRUTH) + rng.normal(0, SD, (KS.size, V.size))
    w2 = np.tile(_w(), (KS.size, 1))
    w2[1] *= 0.5                                         # a (n_planes, n_v) form, planes weighted differently
    fit = NI.fit_node(I, B, SD, forward, TRUTH, cfg, planes=np.ones(KS.size, bool), weights=w2)
    th = fit.theta
    r = (I - B * forward(*th)) / SD
    assert_allclose(fit.cost, np.sum(w2 * r ** 2), rtol=1e-12)
    assert_allclose(fit.chi2_per_plane, np.sum(w2 * r ** 2, axis=1) / np.sum(w2, axis=1), rtol=1e-12)
    sw = np.sqrt(w2).ravel()
    Jw = np.empty((I.size, 4))
    for p in range(4):
        h = 1e-6 * max(1.0, abs(th[p]))
        e = np.zeros(4)
        e[p] = h
        rp = (np.sqrt(w2) * (I - B * forward(*(th + e))) / SD).ravel()
        rm = (np.sqrt(w2) * (I - B * forward(*(th - e))) / SD).ravel()
        Jw[:, p] = (rp - rm) / (2 * h)
    A = np.linalg.inv(Jw.T @ Jw)
    JB = sw[:, None] * Jw
    cov = A @ (JB.T @ JB) @ A
    # the solver's 2-point Jacobian against a central difference: 1e-3 relative on the covariance's diagonal
    assert_allclose(np.diag(fit.cov), np.diag(cov), rtol=2e-3)
    assert_allclose(fit.se, np.sqrt(np.diag(cov)), rtol=1e-3)


def test_weighted_coverage():
    cfg = H.cfg_with()
    rng = np.random.default_rng(3031)
    I0 = B * forward(*TRUTH)
    w = _w()
    cover = 0
    for _ in range(300):
        fit = NI.fit_node(I0 + rng.normal(0, SD, I0.shape), B, SD, forward, TRUTH, cfg,
                          planes=np.ones(KS.size, bool), weights=w)
        assert fit.status == "converged"
        cover += abs(fit.theta[0] - TRUTH[0]) <= fit.se[0]
    frac = cover / 300.0
    print("weighted coverage %.3f" % frac)
    assert 0.60 <= frac <= 0.76                          # Block 30's band at n = 300 (nominal 68 %)


def test_recentred_fit():
    cfg = H.cfg_with()
    I = B * forward(*TRUTH)
    start = TRUTH.copy()
    start[2] += 0.2
    lam = 0.7
    fit, info = NI.fit_node_recentred(I, B, SD, forward, start, cfg, V, lam)
    assert set(info) >= {"n_solves", "centre_um", "converged", "lambda_um", "nfev", "weights"}
    assert info["n_solves"] <= 3 and info["lambda_um"] == lam
    assert abs(fit.theta[2] - TRUTH[2]) <= 1e-3 and abs(info["centre_um"] - TRUTH[2]) <= 1e-2
    assert_allclose(info["weights"], np.exp(-np.abs(V - info["centre_um"]) / lam), rtol=1e-15)
    keep = np.abs(V) < 1.8
    fit2, info2 = NI.fit_node_recentred(I, B, SD, forward, start, cfg, V, lam, keep=keep)
    assert np.all(info2["weights"][~keep] == 0)
    # centre_offset_um maps v0 onto the line: v_c = v0 + offset
    fit3, info3 = NI.fit_node_recentred(I, B, SD, forward, TRUTH, cfg, V, lam, centre_offset_um=-0.3)
    assert abs(info3["centre_um"] - (fit3.theta[2] - 0.3)) <= cfg.inference.weight_irls_tol_um + 1e-12


@pytest.mark.parametrize("bad", [-np.ones(V.size), np.full(V.size, np.nan), np.ones(V.size - 1),
                                 np.zeros(V.size), np.ones((2, V.size))])
def test_weights_refusals(bad):
    cfg = H.cfg_with()
    I = B * forward(*TRUTH)
    with pytest.raises(ValueError):
        NI.fit_node(I, B, SD, forward, TRUTH, cfg, weights=bad)
