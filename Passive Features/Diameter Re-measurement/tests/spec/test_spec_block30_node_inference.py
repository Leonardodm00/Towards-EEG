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


# ------------------------------------------- amendment 3 (wave 7; tester, 2026-11-23 13:55) ---
# The SGT option and the error bars on the fine lattice. Stand-in forward models from tests/spec/wave7_refs.py:
# each plane's profile is (3'') with phi = 0 and an SGT kernel of FWHM 0.25 + 0.8 |z_k - z_ax| um, written from
# Block 12's (H1)-(H3) with its own Gauss-Legendre rule (nothing from blur_kernels or the code under test).

import dataclasses  # noqa: E402

import wave7_refs as W7  # noqa: E402

ZP = KS * DZ
SH_TRUTH = (1.4, 0.4, 1.0)
TH7 = np.array([1.0, 1.5, 0.05, 0.1])


def _cfg(**inf):
    cfg = H.cfg_with()
    if inf:
        cfg = dataclasses.replace(cfg, inference=dataclasses.replace(cfg.inference, **inf))
        cfg.validate()
    return cfg


def _phi(fit, skew_fit):
    """The search vector at the estimate, rebuilt from the reported theta and shape (amendment 3 (A))."""
    b, xi, z = fit.shape
    ph = list(fit.theta) + [math.log10(b), xi]
    if skew_fit:
        ph.append(math.log(z))
    return np.array(ph, dtype=float)


def _args(ph, skew_fit, zeta0):
    return tuple(ph[:4]) + (10.0 ** ph[4], ph[5], math.exp(ph[6]) if skew_fit else zeta0)


def _box(cfg, z_c, skew_fit):
    i = cfg.inference
    lo = [i.d_bounds_um[0], i.mu_bounds_per_um[0], i.v0_bounds_um[0], z_c + i.z_bounds_um[0],
          i.sgt_log10_beta_bounds[0], i.sgt_xi_bounds[0]] + ([i.sgt_ln_zeta_bounds[0]] if skew_fit else [])
    hi = [i.d_bounds_um[1], i.mu_bounds_per_um[1], i.v0_bounds_um[1], z_c + i.z_bounds_um[1],
          i.sgt_log10_beta_bounds[1], i.sgt_xi_bounds[1]] + ([i.sgt_ln_zeta_bounds[1]] if skew_fit else [])
    return np.array(lo), np.array(hi)


def _fine_cov_by_hand(fwd_cov, ph, I, Bk, sigma, planes, cfg, z_c, skew_fit, zeta0, weights=None, active=()):
    """Amendment 3 (D) written out: J of r = (I - B tau)/sigma by central differences of fwd_cov with the absolute
    steps cfg.inference.cov_steps, the one-sided three-point rule inward within two steps of a bound; (J^T J)^-1,
    or the sandwich with weights; NaN rows/columns for the parameters named in active."""
    n = ph.size
    h = np.array(cfg.inference.cov_steps[:n], dtype=float)
    lo, hi = _box(cfg, z_c, skew_fit)

    def tau(p):
        return np.asarray(fwd_cov(*_args(p, skew_fit, zeta0)), dtype=float)[planes]

    cols = []
    for i in range(n):
        e = np.zeros(n)
        e[i] = h[i]
        if ph[i] - lo[i] <= 2 * h[i]:
            dt = (-3 * tau(ph) + 4 * tau(ph + e) - tau(ph + 2 * e)) / (2 * h[i])
        elif hi[i] - ph[i] <= 2 * h[i]:
            dt = (-3 * tau(ph) + 4 * tau(ph - e) - tau(ph - 2 * e)) / (-2 * h[i])
        else:
            dt = (tau(ph + e) - tau(ph - e)) / (2 * h[i])
        cols.append((-(np.asarray(Bk)[planes][:, None] * dt) / sigma).ravel())
    J = np.stack(cols, axis=1)
    if weights is None:
        cov = np.linalg.inv(J.T @ J)
    else:
        sw = np.sqrt(np.broadcast_to(np.asarray(weights, float), I.shape)[planes]).ravel()
        Jw = sw[:, None] * J
        A = np.linalg.inv(Jw.T @ Jw)
        JB = sw[:, None] * Jw
        cov = A @ (JB.T @ JB) @ A
    names = list(NI.PARAM_NAMES) + ["log10_beta", "xi", "ln_zeta"][:n - 4]
    for nm in active:
        i = names.index(nm.rsplit("_", 1)[0])
        cov[i, :] = np.nan
        cov[:, i] = np.nan
    return cov


def _cov_close(a, b, tol):
    """Error bars (sqrt of the diagonals) within tol relative; off-diagonals within tol of sqrt(C_ii C_jj)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert _same_nan(a, b)
    da, db = np.diag(a), np.diag(b)
    ok = ~np.isnan(da)
    assert_allclose(np.sqrt(da[ok]), np.sqrt(db[ok]), rtol=tol)
    scale = np.sqrt(np.outer(np.abs(db), np.abs(db)))
    m = ~np.isnan(a)
    assert np.all(np.abs(a[m] - b[m]) <= tol * scale[m]), np.max(np.abs(a[m] - b[m]) / scale[m])


def _same_nan(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return a.shape == b.shape and np.array_equal(np.isnan(a), np.isnan(b))


_B30_CASES = r'''
import os, sys
import numpy as np
sys.path.insert(0, os.environ["SPEC_DIR"])
import wave7_refs as W7
from allen_diameter.config import default_config
from allen_diameter.analysis import node_inference as NI

KS = np.array([-2, -1, 0, 1, 2]); DZ = 0.28
V = np.arange(-20, 21) * 0.1144


def b30_cases():
    cfg = default_config()
    fwd = W7.make_gauss_forward(V, KS * DZ)
    rng = np.random.default_rng(7301)
    I = 210.0 * fwd(1.0, 0.8, 0.07, 0.05) + rng.normal(0.0, 3.0, (KS.size, V.size))
    th0 = np.array([1.2, 0.6, 0.0, 0.0])
    out = {}
    w = np.where(np.abs(V) < 1.6, 1.0, 0.0) * np.exp(-np.abs(V) / 0.9)
    fits = {"plain": NI.fit_node(I, 210.0, 3.0, fwd, th0, cfg),
            "planes": NI.fit_node(I, 210.0, 3.0, fwd, th0, cfg, planes=np.array([True, True, True, False, True])),
            "weighted": NI.fit_node(I, 210.0, 3.0, fwd, th0, cfg, weights=w, z_box_centre_um=0.02)}
    rec, info = NI.fit_node_recentred(I, 210.0, 3.0, fwd, th0, cfg, V, 0.7, centre_offset_um=-0.1,
                                      keep=np.abs(V) < 2.0)
    fits["recentred"] = rec
    for k in ("n_solves", "centre_um", "converged", "lambda_um", "nfev", "weights"):
        out["info_" + k] = np.asarray(info[k])
    for name, f in fits.items():
        for fld in ("theta", "cov", "se", "cost", "chi2_per_plane", "planes", "n_starts", "nfev"):
            out["%s_%s" % (name, fld)] = np.asarray(getattr(f, fld))
        out["%s_status" % name] = np.asarray(f.status)
        out["%s_at_bound" % name] = np.asarray(";".join(f.at_bound))
    return out, fits


if __name__ == "__main__":
    np.savez(os.path.join(os.environ["OUT"], "b30_base.npz"), **b30_cases()[0])
'''


def test_am3_gaussian_path_bit_identical_to_base(tmp_path, monkeypatch):
    """B30.1 (amendment 3 (vii)): with shape0 None and forward_cov None, fit_node (plain, given planes, weighted with
    a box centre) and fit_node_recentred give, bit for bit, what the base commit's package (f1bc7cb, before wave 7,
    in a subprocess) gives on the same stand-in forward and seeded data; and the Gaussian NodeFit has shape and
    se_shape None, param_names = PARAM_NAMES, cov_search equal to cov (NaN in the same places), cov_source "fit"."""
    monkeypatch.setenv("SPEC_DIR", str(H.HERE))
    W7.run_at_commit(_B30_CASES, tmp_path)
    ns = {}
    exec(compile(_B30_CASES, "b30_cases", "exec"), ns)
    head, fits = ns["b30_cases"]()
    with np.load(tmp_path / "b30_base.npz") as z:
        assert sorted(z.files) == sorted(head)
        for key in z.files:
            assert_array_equal(np.asarray(head[key]), z[key], err_msg=key)
    for f in fits.values():
        assert f.shape is None and f.se_shape is None and tuple(f.param_names) == tuple(NI.PARAM_NAMES)
        assert f.cov_source == "fit"
        assert _same_nan(f.cov_search, f.cov) and np.array_equal(np.nan_to_num(f.cov_search), np.nan_to_num(f.cov))


@pytest.mark.parametrize("skew", ["off", "fit"])
def test_am3_parametrisation_spy(skew):
    """B30.2: under the SGT every forward call has the seven arguments (d, mu, v0, z_ax, beta, xi, zeta) with beta =
    10^b inside 10^sgt_log10_beta_bounds, xi inside sgt_xi_bounds, zeta = zeta0 at every call under "off" and inside
    exp(sgt_ln_zeta_bounds) under "fit"; forward_cov is never evaluated outside the box (the one-sided rule);
    param_names = PARAM_NAMES + ("log10_beta", "xi"[, "ln_zeta"])."""
    cfg = _cfg(sgt_skew=skew)
    calls, ccalls = [], []
    fwd = W7.make_sgt_forward(V, ZP, calls=calls)
    fcov = W7.make_sgt_forward(V, ZP, n_nodes=600, calls=ccalls)
    I = B * fwd(*TH7, 1.4, 0.4, 1.2 if skew == "fit" else 1.0)
    calls.clear()
    fit = NI.fit_node(I, B, SD, fwd, TH7 * np.array([1.1, 0.9, 1, 1]), cfg, shape0=(2.0, 0.0, 1.0),
                      forward_cov=fcov)
    i = cfg.inference
    names = tuple(NI.PARAM_NAMES) + ("log10_beta", "xi") + (("ln_zeta",) if skew == "fit" else ())
    assert tuple(fit.param_names) == names
    assert fit.cov_search.shape == (len(names), len(names)) and fit.cov_source == "fine"
    for c in calls + ccalls:
        assert len(c) == 7
        assert 10 ** i.sgt_log10_beta_bounds[0] - 1e-12 <= c[4] <= 10 ** i.sgt_log10_beta_bounds[1] + 1e-12
        assert i.sgt_xi_bounds[0] <= c[5] <= i.sgt_xi_bounds[1]
        if skew == "off":
            assert c[6] == 1.0
        else:
            assert math.exp(i.sgt_ln_zeta_bounds[0]) - 1e-12 <= c[6] <= math.exp(i.sgt_ln_zeta_bounds[1]) + 1e-12
    assert len(ccalls) >= 2 * len(names) - 1


def test_am3_noise_free_recovery():
    """B30.8 (spec (i)): noise-free recovery of (d, mu, v0, z_ax, beta, xi) = (1.0, 1.5, 0.05, 0.1, 1.4, 0.4) from the
    Gaussian member with theta 15 % off, and with zeta = 1.3 fitted ("fit") [relative 1e-5 on d, beta, xi, zeta;
    1e-5 absolute on v0, z_ax]."""
    for skew, zeta in (("off", 1.0), ("fit", 1.3)):
        cfg = _cfg(sgt_skew=skew)
        fwd = W7.make_sgt_forward(V, ZP)
        I = B * fwd(*TH7, 1.4, 0.4, zeta)
        fit = NI.fit_node(I, B, SD, fwd, TH7 * np.array([1.15, 0.85, 1.0, 1.0]), cfg, shape0=(2.0, 0.0, 1.0))
        print(skew, fit.status, fit.theta, fit.shape, fit.nfev)
        assert fit.status == "converged" and fit.at_bound == ()
        assert_allclose(fit.theta[:2], TH7[:2], rtol=1e-5)
        assert_allclose(fit.theta[2:], TH7[2:], atol=1e-5)
        assert_allclose(fit.shape, [1.4, 0.4, zeta], rtol=1e-5)


def test_am3_gaussian_truth_ends_at_xi_lo():
    """B30.8 / B30.3 / B30.4 (spec (ii), (B), (C)): on a Gaussian truth the SGT fit started at the Gaussian fit's
    estimate (the caller's sequence (G): one start, the Gaussian member) ends at xi = 0 with xi_lo alone in at_bound and
    status "converged" (xi = 0 is a member, D-225.5), beta = 2 [1e-5], d the Gaussian fit's [relative 1e-6]; the
    covariance rows/columns of xi NaN; se_shape = (beta ln10 se(b), NaN (at a bound), NaN (held)); cov = cov_search's
    theta block; se = sqrt(diag(cov))."""
    cfg = _cfg()
    fwd4 = W7.make_gauss_forward(V, ZP)
    fwd7 = W7.make_sgt_forward(V, ZP)
    I = B * fwd4(*TH7)
    g = NI.fit_node(I, B, SD, fwd4, TH7 * np.array([1.2, 0.8, 1, 1]), cfg)
    one = dataclasses.replace(cfg, inference=dataclasses.replace(cfg.inference, multistart_factors=(1.0,)))
    s = NI.fit_node(I, B, SD, fwd7, g.theta, one, planes=g.planes, z_box_centre_um=float(g.theta[3]),
                    shape0=(2.0, 0.0, 1.0))
    print(s.status, s.at_bound, s.shape, s.theta, g.theta)
    assert s.status == "converged" and tuple(s.at_bound) == ("xi_lo",)
    assert s.shape[1] == 0.0 or abs(s.shape[1]) < 1e-9
    assert abs(s.shape[0] - 2.0) <= 1e-5 and s.shape[2] == 1.0
    assert_allclose(s.theta[0], g.theta[0], rtol=1e-6)
    ix = list(s.param_names).index("xi")
    assert np.all(np.isnan(s.cov_search[ix, :])) and np.all(np.isnan(s.cov_search[:, ix]))
    ib = list(s.param_names).index("log10_beta")
    assert_allclose(s.se_shape[0], s.shape[0] * math.log(10.0) * math.sqrt(s.cov_search[ib, ib]), rtol=1e-12)
    assert math.isnan(s.se_shape[1]) and math.isnan(s.se_shape[2])
    assert _same_nan(s.cov, s.cov_search[:4, :4])
    assert_allclose(np.nan_to_num(s.cov), np.nan_to_num(s.cov_search[:4, :4]), rtol=0, atol=0)
    assert_allclose(s.se, np.sqrt(np.diag(s.cov)), rtol=1e-15)


def test_am3_other_bounds_give_at_bound():
    """B30.3: any active bound other than xi_lo gives "at_bound" (D-225.5): a tube wider than d_bounds_um's upper
    end, and a kernel whose tails are heavier than sgt_xi_bounds allow (xi = 2.6 > 2)."""
    cfg = _cfg()
    fwd = W7.make_sgt_forward(V, ZP)
    I = B * fwd(6.8, 0.5, 0.05, 0.1, 2.0, 0.3, 1.0)
    f1 = NI.fit_node(I, B, SD, fwd, np.array([5.5, 0.5, 0.05, 0.1]), cfg, shape0=(2.0, 0.3, 1.0))
    print(f1.status, f1.at_bound)
    assert "d_hi" in f1.at_bound and f1.status == "at_bound"
    I2 = B * fwd(*TH7, 2.0, 2.6, 1.0)
    f2 = NI.fit_node(I2, B, SD, fwd, TH7, cfg, shape0=(2.0, 1.0, 1.0))
    print(f2.status, f2.at_bound, f2.shape)
    assert "xi_hi" in f2.at_bound and f2.status == "at_bound"


@pytest.mark.parametrize("skew,weighted", [("off", False), ("fit", False), ("off", True), ("fit", True)])
def test_am3_fine_covariance_by_hand(skew, weighted):
    """B30.5: with forward_cov (here the stand-in on a finer quadrature, a model distinct from forward) the
    covariance in the search parameters equals amendment 3 (D) written out by hand at the estimate -- central
    differences of forward_cov with the absolute steps inference.cov_steps, one-sided inward within two steps of a
    bound, (J^T J)^-1 or the weighted sandwich on the solve's planes and weights -- [relative 1e-9: the same
    arithmetic in another order]; cov_source "fine"; se_shape by the delta method from it."""
    cfg = _cfg(sgt_skew=skew)
    fwd = W7.make_sgt_forward(V, ZP)
    fcov = W7.make_sgt_forward(V, ZP, n_nodes=900)
    zeta = 1.25 if skew == "fit" else 1.0
    rng = np.random.default_rng(31 if weighted else 32)
    I = B * fwd(*TH7, 1.6, 0.5, zeta) + rng.normal(0.0, SD, (KS.size, V.size))
    w = np.exp(-np.abs(V - 0.05) / 0.8) if weighted else None
    planes = np.array([True, True, True, True, False])
    fit = NI.fit_node(I, B, SD, fwd, TH7, cfg, planes=planes, weights=w, shape0=(2.0, 0.0, 1.0), forward_cov=fcov)
    assert fit.cov_source == "fine" and fit.status in ("converged", "at_bound")
    ph = _phi(fit, skew == "fit")
    ref = _fine_cov_by_hand(fcov, ph, I, np.full(KS.size, B), SD, planes, cfg, float(TH7[3]), skew == "fit", 1.0,
                            weights=w, active=fit.at_bound)
    print(skew, weighted, fit.at_bound, np.sqrt(np.abs(np.diag(fit.cov_search))))
    assert _same_nan(fit.cov_search, ref)
    m = ~np.isnan(ref)
    assert_allclose(fit.cov_search[m], ref[m], rtol=1e-9, atol=1e-30)
    sd = np.sqrt(np.diag(fit.cov_search))
    assert_allclose(fit.se_shape[0], fit.shape[0] * math.log(10.0) * sd[4], rtol=1e-12, equal_nan=True)
    assert_allclose(fit.se_shape[1], sd[5], rtol=1e-12, equal_nan=True)
    if skew == "fit":
        assert_allclose(fit.se_shape[2], fit.shape[2] * sd[6], rtol=1e-12, equal_nan=True)
    else:
        assert math.isnan(fit.se_shape[2])


def test_am3_forward_cov_equals_forward():
    """B30.6 (spec (iv)): with forward_cov = forward the fine error bars agree with (J^T J)^-1 of a Jacobian
    Richardson-extrapolated by hand (central differences at h and h/2) [relative 1e-3] and with the solver's own
    error bars (cov_source "fit") [relative 1e-3], with the same NaN pattern; the Gaussian path likewise. The error
    bars (the SEs) are compared relatively; the off-diagonal entries scaled by sqrt(C_ii C_jj) (a near-zero
    covariance has no meaningful relative error)."""
    cfg = _cfg()
    fwd = W7.make_sgt_forward(V, ZP)
    rng = np.random.default_rng(33)
    I = B * fwd(*TH7, 1.6, 0.5, 1.0) + rng.normal(0.0, SD, (KS.size, V.size))
    a = NI.fit_node(I, B, SD, fwd, TH7, cfg, shape0=(2.0, 0.0, 1.0))
    b = NI.fit_node(I, B, SD, fwd, TH7, cfg, shape0=(2.0, 0.0, 1.0), forward_cov=fwd)
    assert a.cov_source == "fit" and b.cov_source == "fine"
    assert_array_equal(a.theta, b.theta)
    assert _same_nan(a.cov_search, b.cov_search)
    _cov_close(b.cov_search, a.cov_search, 1e-3)
    ph = _phi(b, False)
    h = np.array(cfg.inference.cov_steps[:6])
    planes = b.planes

    def tau(p):
        return fwd(*_args(p, False, 1.0))[planes]

    cols = []
    for i in range(6):
        e = np.zeros(6)
        d1 = (tau(ph + h[i] * np.eye(6)[i]) - tau(ph - h[i] * np.eye(6)[i])) / (2 * h[i])
        d2 = (tau(ph + 0.5 * h[i] * np.eye(6)[i]) - tau(ph - 0.5 * h[i] * np.eye(6)[i])) / h[i]
        cols.append((-(B * (4 * d2 - d1) / 3.0) / SD).ravel())
        del e
    J = np.stack(cols, axis=1)
    rich = np.linalg.inv(J.T @ J)
    for nm in b.at_bound:
        i = list(b.param_names).index(nm.rsplit("_", 1)[0])
        rich[i, :] = np.nan
        rich[:, i] = np.nan
    assert _same_nan(b.cov_search, rich)
    _cov_close(b.cov_search, rich, 1e-3)
    fwd4 = W7.make_gauss_forward(V, ZP)
    I4 = B * fwd4(*TH7) + rng.normal(0.0, SD, (KS.size, V.size))
    g1 = NI.fit_node(I4, B, SD, fwd4, TH7, cfg)
    g2 = NI.fit_node(I4, B, SD, fwd4, TH7, cfg, forward_cov=fwd4)
    assert g2.cov_source == "fine" and g2.shape is None and _same_nan(g2.cov_search, g2.cov)
    _cov_close(g2.cov, g1.cov, 1e-3)


def test_am3_nan_forward_cov():
    """B30.5: forward_cov returning a non-finite value at a point it needs gives an all-NaN covariance (cov,
    cov_search, se, se_shape), cov_source "fine", the estimate unchanged."""
    cfg = _cfg()
    fwd = W7.make_sgt_forward(V, ZP)
    I = B * fwd(*TH7, 1.6, 0.5, 1.0)
    a = NI.fit_node(I, B, SD, fwd, TH7, cfg, shape0=(2.0, 0.0, 1.0))
    n = {"k": 0}

    def bad(*args):
        n["k"] += 1
        out = fwd(*args)
        if n["k"] == 3:
            out = out.copy()
            out[1, 7] = np.nan
        return out
    b = NI.fit_node(I, B, SD, fwd, TH7, cfg, shape0=(2.0, 0.0, 1.0), forward_cov=bad)
    assert_array_equal(a.theta, b.theta) and b.cov_source == "fine"
    for arr in (b.cov, b.cov_search, b.se, b.se_shape):
        assert np.all(np.isnan(arr))


def test_am3_recentred_with_shape():
    """B30.7 (spec (E), (vi)): fit_node_recentred passes shape0 and forward_cov on; forward_cov is evaluated only
    after the last solve (never between solves), once, with the last solve's weights; the result is where fit_node
    ends with those weights started at it [relative 1e-6: the solver's tolerances], and its error bars equal
    fit_node's fine error bars there."""
    cfg = _cfg()
    log = []
    fwd = W7.make_sgt_forward(V, ZP)

    def f_spy(*a):
        log.append("f")
        return fwd(*a)

    def c_spy(*a):
        log.append("c")
        return fwd(*a)
    rng = np.random.default_rng(34)
    I = B * fwd(*TH7, 1.6, 0.5, 1.0) + rng.normal(0.0, SD, (KS.size, V.size))
    start = TH7.copy()
    start[2] += 0.15
    fit, info = NI.fit_node_recentred(I, B, SD, f_spy, start, cfg, V, 0.7, shape0=(2.0, 0.0, 1.0), forward_cov=c_spy)
    print(info["n_solves"], fit.status, fit.theta, fit.shape)
    assert fit.cov_source == "fine" and fit.shape is not None
    first_c = log.index("c")
    assert "f" not in log[first_c:], "forward evaluated after the error bars began"
    assert info["n_solves"] >= 2
    one = dataclasses.replace(cfg, inference=dataclasses.replace(cfg.inference, multistart_factors=(1.0,)))
    again = NI.fit_node(I, B, SD, fwd, fit.theta, one, planes=fit.planes, weights=info["weights"],
                        z_box_centre_um=float(start[3]), shape0=tuple(fit.shape), forward_cov=fwd)
    assert_allclose(again.theta, fit.theta, rtol=1e-6, atol=1e-7)
    assert_allclose(again.shape, fit.shape, rtol=1e-6, atol=1e-7)
    _cov_close(again.cov_search, fit.cov_search, 1e-3)


@pytest.mark.parametrize("bad", [(2.0, 0.0), (2.0, 0.0, 1.0, 1.0), (0.0, 0.0, 1.0), (-2.0, 0.1, 1.0),
                                 (2.0, 0.0, 0.0), (2.0, math.nan, 1.0), (math.inf, 0.0, 1.0), (20.0, 0.0, 1.0),
                                 (0.5, 0.0, 1.0), (2.0, -0.1, 1.0), (2.0, 2.5, 1.0), (2.0, 0.0, 3.0),
                                 "gauss", None])
def test_am3_shape0_refusals(bad):
    """B30.9 (F): ValueError for a shape0 that is not three finite numbers with beta > 0, zeta > 0, or that lies
    outside the box (beta in [1, 10], xi in [0, 2], zeta in [e^-0.7, e^0.7] at the defaults; a held zeta too);
    None is the Gaussian fit."""
    cfg = _cfg()
    fwd = W7.make_sgt_forward(V, ZP)
    I = B * fwd(*TH7, 1.6, 0.5, 1.0)
    if bad is None:
        assert NI.fit_node(I, B, SD, W7.make_gauss_forward(V, ZP), TH7, cfg, shape0=None).shape is None
        return
    with pytest.raises(ValueError):
        NI.fit_node(I, B, SD, fwd, TH7, cfg, shape0=bad)


def test_am3_determinism_and_inputs_unchanged():
    """B30.10: two SGT fits with fine error bars are bit-identical; the inputs are not modified."""
    cfg = _cfg(sgt_skew="fit")
    fwd = W7.make_sgt_forward(V, ZP)
    rng = np.random.default_rng(35)
    I = B * fwd(*TH7, 1.6, 0.5, 1.1) + rng.normal(0.0, SD, (KS.size, V.size))
    Bk = np.full(KS.size, B)
    th0 = TH7.copy()
    I0, B0, t0 = I.copy(), Bk.copy(), th0.copy()
    a = NI.fit_node(I, Bk, SD, fwd, th0, cfg, shape0=(2.0, 0.0, 1.0), forward_cov=fwd)
    b = NI.fit_node(I, Bk, SD, fwd, th0, cfg, shape0=(2.0, 0.0, 1.0), forward_cov=fwd)
    for fld in ("theta", "cov", "se", "chi2_per_plane", "shape", "se_shape", "cov_search"):
        assert_array_equal(getattr(a, fld), getattr(b, fld))
    assert a.cost == b.cost and a.status == b.status and a.at_bound == b.at_bound
    assert_array_equal(I, I0) and assert_array_equal(Bk, B0) is None and np.array_equal(th0, t0)


def test_am3_sgt_coverage():
    """B30.8 (spec (iii)): on the entry's 300 seeded noisy sets (SD 3 gl, B 210, a Gaussian truth at d = 1 um) the
    SGT fit with zeta held, started as the caller does (from the Gaussian fit's estimate, the Gaussian member, one
    start, the same planes), gives d +- se covering the truth in 60-76 % (nominal 68 %; the band is +-3 binomial SDs
    at n = 300) and its mean d within 3 standard errors of the truth. Reported: the share ending at xi_lo."""
    cfg = _cfg()
    one = dataclasses.replace(cfg, inference=dataclasses.replace(cfg.inference, multistart_factors=(1.0,)))
    fwd4 = W7.make_gauss_forward(V, ZP, n_nodes=200)
    fwd7 = W7.make_sgt_forward(V, ZP, n_nodes=200)
    truth = np.array([1.0, 0.8, 0.07, 0.05])
    I0 = B * fwd4(*truth)
    rng = np.random.default_rng(3031)
    cover, ds, at_xi = 0, [], 0
    for _ in range(300):
        I = I0 + rng.normal(0.0, SD, I0.shape)
        g = NI.fit_node(I, B, SD, fwd4, truth * np.array([1.1, 0.9, 1, 1]), cfg)
        s = NI.fit_node(I, B, SD, fwd7, g.theta, one, planes=g.planes, z_box_centre_um=float(g.theta[3]),
                        shape0=(2.0, 0.0, 1.0))
        cover += abs(s.theta[0] - truth[0]) <= s.se[0]
        ds.append(s.theta[0])
        at_xi += "xi_lo" in s.at_bound
    ds = np.array(ds)
    frac = cover / 300.0
    z = (ds.mean() - truth[0]) / (ds.std(ddof=1) / math.sqrt(ds.size))
    print("SGT coverage %.3f, mean d %+.2f SE, %d of 300 at xi_lo" % (frac, z, at_xi))
    assert 0.60 <= frac <= 0.76 and abs(z) <= 3.0
