"""Smoke test for the per-node fit of the forward model -- Block 30 in specs/SPEC.md
(D-129.1, D-130.O9, D-132.2).

The forward model is built here from Block 3's tube_model.model_profile: one
Gaussian blur whose SD grows with the plane's defocus from the axis,
sigma_k(z_ax) = sqrt(S0^2 + (GROWTH (z_k - z_ax))^2), tau_k = model_profile / B
at alpha = mu d (tilt 0). Block 26 (the real forward) is not imported.

Checks
    test_known_answer   noise-free recovery of (d, mu, v0, z_ax) from a start
                        30 % off in d and mu [relative 1e-6; 1e-6 um on v0, z_ax]
    test_reference      the Gauss-Newton covariance against (J^T J)^{-1} with J
                        from independent central differences
    test_convergence    skipped (not a discretisation)
    test_invariants     300 seeded noisy sets (SD 3 gl, B 210, d 1 um):
                        coverage of d_hat +- se in [60, 76] %, mean d_hat within
                        3 SE of the truth, mean chi2_k / N within 10 % of 1
    test_contract       NodeFit fields, shapes, dtypes; select_planes against
                        the rule written by hand; inputs unchanged
    test_determinism    two identical calls are bit-identical
    test_edge_cases     truth beyond a bound -> at_bound (and NaN cov row);
                        NaN forward -> failed (planes given or selected);
                        planes all False -> ValueError; invalid inputs
    test_weights        amendment 2 (D-169.4, D-171.2): weights of ones = no
                        weights; noise-free recovery under centre weights; a
                        second dip beside the tube pulls the unweighted fit's d
                        (+5 %, +220 % in the two cases) and the cut with the
                        centre weights removes most of it (+2 %, +16 %); the
                        sandwich covariance covers the truth in
                        [60, 76] % of 300 noisy sets; the refusals
    test_recentred      fit_node_recentred started 0.2 um off the centre ends on
                        it within 3 solves; its record; keep honoured

Amendment 3 (wave 7; D-221.2-3, D-222.1-2, D-224.1-2, D-225.5-6): the SGT option
and the error bars on the fine lattice. The SGT stand-in is Block 3's rule
(model_profile: the tube's chord profile on the Gauss-Legendre grid in psi,
u = v0 + (d/2) sin psi) with the Gaussian replaced by
blur_kernels.sgt_kernel_pdf of the law's FWHM 2 sqrt(2 ln 2) sigma_k(z_ax),
evaluated at v_n - u (the side u > 0 towards increasing v is the wide one for
zeta > 1, as Block 49's kernel); at the Gaussian member (2, 0, 1) it equals
make_forward to 2.2e-16. forward_cov is the same stand-in, or the stand-in on
a rule of twice the nodes (the "fine lattice").
    test_sgt_known_answer   (i) noise-free recovery of (d, mu, v0, z_ax, beta,
                            xi) = (1.0, 1.5, 0.05, 0.1, 1.4, 0.4) from the
                            Gaussian member at theta 15 % off, and with "fit"
                            of zeta = 1.3 [relative 1e-5]; forward calls reported
    test_sgt_gaussian_truth (ii) a Gaussian truth: the SGT fit of sequence (G)
                            ends at xi = 0 with xi_lo the only active bound,
                            "converged", beta 2 [1e-5], d the Gaussian fit's
                            [relative 1e-6]; also from an off start; another
                            bound (log10_beta_hi, alone or with xi_lo) gives
                            "at_bound"
    test_sgt_coverage       (iii) the entry's 300 seeded sets, the SGT fit of
                            sequence (G), zeta held: coverage of d +- se in
                            [60, 76] %, mean d within 3 SE
    test_fine_errors        (iv) forward_cov = forward: the error bars against
                            (J^T J)^{-1} of a Richardson-extrapolated Jacobian
                            [relative 1e-3]; the one-sided rule at a lower bound
                            (xi at xi_lo) and near an upper bound (v0), against
                            the rule written here [1e-10]; the Gaussian path;
                            cov_source; a non-finite forward_cov
    test_fine_sandwich      (v) the weighted sandwich with the shape against a
                            hand-written computation [1e-10]; se_shape by the
                            delta method [1e-12]
    test_sgt_recentred      (vi) fit_node_recentred with the shape ends where
                            fit_node with its last weights ends [1e-8]; the
                            solve sequence (each later solve one start from the
                            previous shape) and the error bars after the last
                            solve, bit for bit
    test_gaussian_fields    (vii) a Gaussian NodeFit: shape None, cov_search
                            equal to cov, the default fields (the entry's tests
                            above are the Gaussian path's)
    test_sgt_contract       (viii) fields and shapes, determinism, the inputs
                            unchanged, a failed SGT fit, the ValueErrors

Run (from the code folder "Passive Features/Diameter Re-measurement")
    python3 tests/smoke/test_smoke_node_inference.py
    python3 -m pytest tests/smoke/test_smoke_node_inference.py -q

Debug
    A failed check_close() saves both arrays to
        tests/smoke/artifacts/<check name>.npz

Pure ASCII, for the cluster.
"""
from __future__ import annotations

import dataclasses
import importlib.metadata
import platform
import re
import sys
import time
import traceback
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal  # noqa: F401

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
SRC = WS / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from allen_diameter.analysis import line_weights as LW  # noqa: E402
from allen_diameter.analysis import node_inference as NI  # noqa: E402
from allen_diameter.config import default_config  # noqa: E402
from allen_diameter.model import blur_kernels as BK  # noqa: E402
from allen_diameter.model import tube_model as M  # noqa: E402

SEED = 20261113
REPORT_PACKAGES = ("numpy", "scipy")
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
_SAVED_ARTIFACTS = []  # files written by check_close during this run

CFG = default_config()
B_GL = 210.0
NOISE_GL = 3.0
TRUTH = np.array([1.0, 1.0, 0.1, 0.07])     # d um, mu 1/um, v0 um, z_ax um
START = np.array([1.3, 0.7, 0.0, 0.0])      # 30 % off in d and mu; z bounds are about z0 = 0
Z_PLANES = 0.28 * np.arange(-3, 4)          # um, plane spacing dz of section 2.1
V = 0.1144 * np.arange(-26, 27)             # um, Block 5's offsets, step p_x, |v| <= 3 um
S0 = 0.1                                    # um, the in-focus blur
GROWTH = 0.4                                # um of blur SD per um of defocus
N_NODES = M.n_quadrature_nodes(CFG.inference.d_bounds_um[1], S0, CFG.measure.fit_quad_min_nodes,
                               CFG.measure.fit_quad_nodes_per_sigma)
N_SETS = 300


def make_forward(z_planes=Z_PLANES, v=V):
    def forward(d, mu, v0, z_ax):
        out = np.empty((z_planes.size, v.size))
        for k, zk in enumerate(z_planes):
            sig = np.sqrt(S0 ** 2 + (GROWTH * (zk - z_ax)) ** 2)
            out[k] = M.model_profile(v, d, mu * d, v0, sig, 1.0, N_NODES)
        return out
    return forward


FORWARD = make_forward()


def noisy_set(rng):
    return B_GL * FORWARD(*TRUTH) + rng.normal(0.0, NOISE_GL, size=(Z_PLANES.size, V.size))


# --------------------------------------------------------------- helpers ---

def make_rng(seed: int = SEED) -> np.random.Generator:
    """Fresh generator, to be passed explicitly to the code under test."""
    return np.random.default_rng(seed)


def check_close(name, actual, expected, rtol=1e-12, atol=0.0):
    """numpy.testing.assert_allclose that keeps the evidence on failure."""
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    try:
        assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=name)
    except AssertionError:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
        path = ARTIFACT_DIR / f"{safe_name}.npz"
        np.savez(path, actual=actual, expected=expected)
        _SAVED_ARTIFACTS.append(path)
        raise


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    I = B_GL * FORWARD(*TRUTH)
    fit = NI.fit_node(I, np.full(Z_PLANES.size, B_GL), NOISE_GL, FORWARD, START, CFG)
    assert fit.status == "converged", (fit.status, fit.at_bound)
    err = fit.theta - TRUTH
    print("    noise-free recovery: theta_hat - truth = %s, nfev %d" % (np.array2string(err, precision=3), fit.nfev))
    # SPEC Block 30: relative 1e-6 on d and mu; absolute 1e-6 um on v0 and z_ax
    check_close("known_answer_d_mu", fit.theta[:2], TRUTH[:2], rtol=1e-6)
    check_close("known_answer_v0_z", fit.theta[2:], TRUTH[2:], rtol=0.0, atol=1e-6)


def _central_jacobian(I, B, keep, theta, h=1e-6):
    cols = []
    for i in range(4):
        e = np.zeros(4)
        e[i] = h * max(1.0, abs(theta[i]))
        rp = ((I - B[:, None] * FORWARD(*(theta + e))) / NOISE_GL)[keep].ravel()
        rm = ((I - B[:, None] * FORWARD(*(theta - e))) / NOISE_GL)[keep].ravel()
        cols.append((rp - rm) / (2.0 * e[i]))
    return np.stack(cols, axis=1)


def test_reference():
    rng = make_rng(SEED + 1)
    I = noisy_set(rng)
    B = np.full(Z_PLANES.size, B_GL)
    fit = NI.fit_node(I, B, NOISE_GL, FORWARD, START, CFG)
    assert fit.status == "converged", fit.status
    J = _central_jacobian(I, B, fit.planes, fit.theta)
    cov_ref = np.linalg.inv(J.T @ J)
    # 2-point (scipy) against central differences: truncation O(h) with h ~ 1.5e-8 relative,
    # roundoff ~ eps / h; 1e-4 relative is far above both and far below any formula error
    # entries are compared on the scale of the correlation: atol = 1e-4 sqrt(C_ii C_jj)
    scale = np.sqrt(np.outer(np.diag(cov_ref), np.diag(cov_ref)))
    check_close("reference_cov", fit.cov / scale, cov_ref / scale, rtol=0.0, atol=1e-4)
    r = ((I - B[:, None] * FORWARD(*fit.theta)) / NOISE_GL)
    check_close("reference_cost", fit.cost, np.sum(r[fit.planes] ** 2), rtol=1e-12)
    check_close("reference_chi2_per_plane", fit.chi2_per_plane, np.mean(r ** 2, axis=1), rtol=1e-12)


def test_convergence():
    raise unittest.SkipTest("not a discretisation: a least-squares fit")


def test_invariants():
    rng = make_rng(SEED)
    B = np.full(Z_PLANES.size, B_GL)
    d_hat, se, chi2, status = [], [], [], []
    for _ in range(N_SETS):
        fit = NI.fit_node(noisy_set(rng), B, NOISE_GL, FORWARD, START, CFG)
        status.append(fit.status)
        d_hat.append(fit.theta[0])
        se.append(fit.se[0])
        chi2.append(np.nanmean(fit.chi2_per_plane))
    d_hat, se, chi2 = np.array(d_hat), np.array(se), np.array(chi2)
    n_conv = sum(s == "converged" for s in status)
    cover = np.mean(np.abs(d_hat - TRUTH[0]) <= se)
    mean_d = np.mean(d_hat)
    sem = np.std(d_hat, ddof=1) / np.sqrt(N_SETS)
    print("    %d sets: %d converged; coverage of d_hat +- se = %.3f (nominal 0.683); "
          "mean d_hat - d = %.5f um (SE %.5f um, %.2f SE); mean se = %.5f um, SD of d_hat = %.5f um; "
          "mean chi2_k/N = %.4f" % (N_SETS, n_conv, cover, mean_d - TRUTH[0], sem, (mean_d - TRUTH[0]) / sem,
                                     np.mean(se), np.std(d_hat, ddof=1), np.mean(chi2)))
    assert n_conv == N_SETS, "not every set converged: %s" % sorted(set(status))
    # SPEC Block 30: coverage in [60, 76] % (binomial SD 2.7 % at 300 sets: about +-3 SD)
    assert 0.60 <= cover <= 0.76, "coverage %.3f outside [0.60, 0.76]" % cover
    # SPEC Block 30: mean d_hat within 3 standard errors of the truth
    assert abs(mean_d - TRUTH[0]) <= 3.0 * sem, "bias %.5f um > 3 SE (%.5f um)" % (mean_d - TRUTH[0], sem)
    # SPEC Block 30: chi2_k / N averages 1 within 10 %
    assert abs(np.mean(chi2) - 1.0) <= 0.10, "mean chi2/N = %.4f" % np.mean(chi2)


def test_contract():
    rng = make_rng(SEED + 2)
    I = noisy_set(rng)
    B = np.full(Z_PLANES.size, B_GL)
    I_copy, B_copy, start_copy = I.copy(), B.copy(), START.copy()
    fit = NI.fit_node(I, B, NOISE_GL, FORWARD, START, CFG)
    assert_array_equal(I, I_copy)
    assert_array_equal(B, B_copy)
    assert_array_equal(START, start_copy)
    assert dataclasses.is_dataclass(fit) and type(fit).__dataclass_params__.frozen
    assert fit.theta.shape == (4,) and fit.theta.dtype == np.float64
    assert fit.cov.shape == (4, 4) and fit.cov.dtype == np.float64
    assert fit.se.shape == (4,)
    assert fit.chi2_per_plane.shape == (Z_PLANES.size,)
    assert fit.planes.shape == (Z_PLANES.size,) and fit.planes.dtype == bool
    assert fit.status in NI.FIT_STATUSES and isinstance(fit.at_bound, tuple)
    assert fit.n_starts == len(CFG.inference.multistart_factors) and fit.nfev > 0
    assert np.all(np.isfinite(fit.theta)) and np.all(np.isfinite(fit.cov)) and np.isfinite(fit.cost)
    check_close("contract_cov_symmetric", fit.cov, fit.cov.T, rtol=1e-10)
    check_close("contract_se", fit.se, np.sqrt(np.diag(fit.cov)), rtol=1e-15)
    # an InferenceConfig gives the same as the DiameterConfig
    fit2 = NI.fit_node(I, B, NOISE_GL, FORWARD, START, CFG.inference)
    assert_array_equal(fit.theta, fit2.theta)
    # select_planes against the rule written by hand, on a defocus ladder and a threshold
    # that cuts through it; B per plane varies
    tau0 = make_forward(z_planes=0.28 * np.arange(-12, 13))(*TRUTH)
    Bk = np.linspace(150.0, 230.0, tau0.shape[0])
    cfg_cut = dataclasses.replace(CFG.inference, plane_snr_min=20.0)
    keep = NI.select_planes(tau0, Bk, NOISE_GL, cfg_cut)
    expect = np.zeros(tau0.shape[0], dtype=bool)
    for k in range(tau0.shape[0]):
        peak = 0.0
        for n in range(tau0.shape[1]):
            peak = max(peak, Bk[k] * (1.0 - tau0[k, n]))
        expect[k] = peak >= 20.0 * NOISE_GL
    assert 0 < expect.sum() < expect.size, "the threshold should cut the ladder"
    assert keep.dtype == bool
    assert_array_equal(keep, expect)
    # the fit uses planes=None -> select_planes(forward(theta0))
    fit3 = NI.fit_node(make_forward(z_planes=0.28 * np.arange(-12, 13))(*TRUTH) * Bk[:, None], Bk, NOISE_GL,
                       make_forward(z_planes=0.28 * np.arange(-12, 13)), TRUTH, cfg_cut)
    assert_array_equal(fit3.planes, expect)
    assert np.all(np.isnan(fit3.chi2_per_plane[~expect])) and np.all(np.isfinite(fit3.chi2_per_plane[expect]))


def test_determinism():
    rng = make_rng(SEED + 3)
    I = noisy_set(rng)
    a = NI.fit_node(I, B_GL, NOISE_GL, FORWARD, START, CFG)
    b = NI.fit_node(I, B_GL, NOISE_GL, FORWARD, START, CFG)
    for f in ("theta", "cov", "se", "chi2_per_plane", "planes"):
        assert_array_equal(getattr(a, f), getattr(b, f))
    assert (a.cost, a.status, a.at_bound, a.nfev) == (b.cost, b.status, b.at_bound, b.nfev)


def _raises(exc, func, *args, **kw):
    try:
        func(*args, **kw)
    except exc:
        return
    raise AssertionError("%s not raised by %s" % (exc.__name__, getattr(func, "__name__", func)))


def test_edge_cases():
    B = np.full(Z_PLANES.size, B_GL)
    I = B_GL * FORWARD(*TRUTH)
    # truth beyond a bound: d = 1 um with d_hi = 0.8 um
    cfg_b = dataclasses.replace(CFG, inference=dataclasses.replace(CFG.inference, d_bounds_um=(0.1, 0.8)))
    fit = NI.fit_node(I, B, NOISE_GL, FORWARD, START, cfg_b)
    assert fit.status == "at_bound" and "d_hi" in fit.at_bound, (fit.status, fit.at_bound, fit.theta)
    assert np.all(np.isnan(fit.cov[0, :])) and np.all(np.isnan(fit.cov[:, 0])) and np.isnan(fit.se[0])
    # a forward returning NaN ends failed without raising, planes given and planes selected
    nan_fwd = lambda d, mu, v0, z: np.full((Z_PLANES.size, V.size), np.nan)  # noqa: E731
    fit = NI.fit_node(I, B, NOISE_GL, nan_fwd, START, CFG, planes=np.ones(Z_PLANES.size, dtype=bool))
    assert fit.status == "failed" and np.all(np.isnan(fit.theta)), fit.status
    fit = NI.fit_node(I, B, NOISE_GL, nan_fwd, START, CFG)
    assert fit.status == "failed", fit.status
    # NaN only away from the start: the trf step is shrunk, the fit still ends
    def half_nan(d, mu, v0, z):
        out = FORWARD(d, mu, v0, z)
        return out if d < 1.5 else out * np.nan
    fit = NI.fit_node(I, B, NOISE_GL, half_nan, START, CFG)
    assert fit.status in NI.FIT_STATUSES
    # planes all False
    _raises(ValueError, NI.fit_node, I, B, NOISE_GL, FORWARD, START, CFG, planes=np.zeros(Z_PLANES.size, dtype=bool))
    cfg_none = dataclasses.replace(CFG.inference, plane_snr_min=1e9)
    _raises(ValueError, NI.fit_node, I, B, NOISE_GL, FORWARD, START, cfg_none)
    # invalid inputs
    _raises(ValueError, NI.fit_node, I, -B, NOISE_GL, FORWARD, START, CFG)
    _raises(ValueError, NI.fit_node, I, B, 0.0, FORWARD, START, CFG)
    _raises(ValueError, NI.fit_node, I[:, :, None], B, NOISE_GL, FORWARD, START, CFG)
    _raises(ValueError, NI.fit_node, I, B[:-1], NOISE_GL, FORWARD, START, CFG)
    _raises(ValueError, NI.fit_node, I, B, NOISE_GL, FORWARD, START[:3], CFG)
    bad = I.copy()
    bad[0, 0] = np.nan
    _raises(ValueError, NI.fit_node, bad, B, NOISE_GL, FORWARD, START, CFG)
    _raises(ValueError, NI.fit_node, I[:-1], B[:-1], NOISE_GL, FORWARD, START, CFG)  # forward shape mismatch
    _raises(ValueError, NI.fit_node, I, B, NOISE_GL, FORWARD, START, CFG, planes=np.ones(Z_PLANES.size))
    _raises(ValueError, NI.select_planes, np.ones(5), B_GL, NOISE_GL, CFG)
    _raises(ValueError, NI.select_planes, np.full((2, 3), np.nan), B_GL, NOISE_GL, CFG)


# ---------------------------------------------------------------- runner ---

def test_weights():
    B = np.full(Z_PLANES.size, B_GL)
    I = B_GL * FORWARD(*TRUTH)
    # (a) weights of ones are the unweighted fit: the same residual vector, so theta bit-identical; cov equal
    #     up to the rounding of A^{-1} A A^{-1} against A^{-1} [1e-9 relative]
    rng = make_rng(SEED + 1)
    In = I + rng.normal(0.0, NOISE_GL, size=I.shape)
    f0 = NI.fit_node(In, B, NOISE_GL, FORWARD, START, CFG)
    f1 = NI.fit_node(In, B, NOISE_GL, FORWARD, START, CFG, weights=np.ones(V.size))
    f2 = NI.fit_node(In, B, NOISE_GL, FORWARD, START, CFG, weights=np.ones((Z_PLANES.size, V.size)))
    assert_array_equal(f1.theta, f0.theta)
    assert_array_equal(f2.theta, f0.theta)
    check_close("weights_ones_cov", f1.cov, f0.cov, rtol=1e-9, atol=1e-18)
    check_close("weights_ones_chi2", f1.chi2_per_plane, f0.chi2_per_plane, rtol=1e-12)
    assert f1.cost == f0.cost
    # (b) noise-free recovery under the centre weights of Block 39 [as test_known_answer]
    w = LW.centre_weights(V, TRUTH[2], 0.6)
    fw = NI.fit_node(I, B, NOISE_GL, FORWARD, START, CFG, weights=w)
    assert fw.status == "converged", (fw.status, fw.at_bound)
    check_close("weights_known_d_mu", fw.theta[:2], TRUTH[:2], rtol=1e-6)
    check_close("weights_known_v0_z", fw.theta[2:], TRUTH[2:], rtol=0.0, atol=1e-6)
    # (c) a second dip beside the tube, not in the model (Gaussian; every plane): the cut and the weights remove
    #     most of its pull on d, not all of it -- the second dip's near flank lies inside the cut. Two cases
    #     (checks/d169_2026-11-18/bump_bias_check.py): 1.3 um aside, SD 0.3 um, 60 gl; 1.5 um aside, SD 0.6 um, 80 gl
    lam = 1.25 * LW.allen_fwhm_um(TRUTH[0], S0)
    for sep, sd, depth, lo_none, hi_both in ((1.3, 0.3, 60.0, 0.04, 0.03), (1.5, 0.6, 80.0, 1.0, 0.20)):
        Ib = I - depth * np.exp(-0.5 * ((V - (TRUTH[2] + sep)) / sd) ** 2)[None, :]
        fu = NI.fit_node(Ib, B, NOISE_GL, FORWARD, START, CFG)
        keep, v_lo, v_hi = LW.bump_mask(Ib, B, NOISE_GL, V, TRUTH[2], 0.9, 2.0, 0.5, 1.0)
        assert np.isfinite(v_hi) and not np.isfinite(v_lo) and TRUTH[2] < v_hi < TRUTH[2] + sep, (v_lo, v_hi)
        fk = NI.fit_node(Ib, B, NOISE_GL, FORWARD, START, CFG, weights=keep.astype(float))
        fc = NI.fit_node(Ib, B, NOISE_GL, FORWARD, START, CFG, weights=LW.centre_weights(V, TRUTH[2], lam) * keep)
        rel = [f.theta[0] / TRUTH[0] - 1.0 for f in (fu, fk, fc)]
        print("    second dip %.1f um aside (SD %.1f um, %.0f gl): d off by %+.1f %% unweighted, %+.1f %% cut at "
              "%.2f um, %+.1f %% cut + centre weights" % (sep, sd, depth, 100 * rel[0], 100 * rel[1], v_hi,
                                                         100 * rel[2]))
        assert rel[0] > lo_none, "the unweighted fit should be pulled by the second dip, got %.3f" % rel[0]
        assert abs(rel[2]) < hi_both and abs(rel[2]) < abs(rel[1]) < abs(rel[0]), rel
    # (d) the sandwich covariance: coverage of d_hat +- se under centre weights [60, 76] % at 300 sets
    #     (binomial SD 2.7 %), and the naive (J_w^T J_w)^{-1} would not be the estimator's covariance
    rng = make_rng(SEED + 2)
    d_hat, se = [], []
    for _ in range(N_SETS):
        f = NI.fit_node(noisy_set(rng), B, NOISE_GL, FORWARD, START, CFG, weights=w)
        assert f.status == "converged", f.status
        d_hat.append(f.theta[0])
        se.append(f.se[0])
    d_hat, se = np.array(d_hat), np.array(se)
    cover = np.mean(np.abs(d_hat - TRUTH[0]) <= se)
    print("    centre weights (lambda 0.6 um), %d sets: coverage %.3f; mean se %.5f um, SD of d_hat %.5f um; "
          "unweighted SD would be smaller (the weights cost precision)" % (N_SETS, cover, se.mean(),
                                                                          d_hat.std(ddof=1)))
    assert 0.60 <= cover <= 0.76, "coverage %.3f outside [0.60, 0.76]" % cover
    assert abs(se.mean() / d_hat.std(ddof=1) - 1.0) < 0.15, (se.mean(), d_hat.std(ddof=1))
    # (e) chi2_k / N is the weighted mean; a plane of zero weight is NaN
    w2 = np.tile(w, (Z_PLANES.size, 1))
    w2[0] = 0.0
    fz = NI.fit_node(In, B, NOISE_GL, FORWARD, START, CFG, weights=w2, planes=np.ones(Z_PLANES.size, dtype=bool))
    assert np.isnan(fz.chi2_per_plane[0]) and np.all(np.isfinite(fz.chi2_per_plane[1:]))
    r = (In - B_GL * FORWARD(*fz.theta)) / NOISE_GL
    check_close("weights_chi2", fz.chi2_per_plane[1:], (w2[1:] * r[1:] ** 2).sum(axis=1) / w2[1:].sum(axis=1),
                rtol=1e-9)
    check_close("weights_cost", fz.cost, float((w2 * r ** 2).sum()), rtol=1e-9)
    # (f) refusals
    for bad in (-np.ones(V.size), np.full(V.size, np.nan), np.ones(V.size + 1), np.zeros(V.size)):
        _raises(ValueError, NI.fit_node, In, B, NOISE_GL, FORWARD, START, CFG, weights=bad)


def test_recentred():
    B = np.full(Z_PLANES.size, B_GL)
    I = B_GL * FORWARD(*TRUTH)
    start = np.array([1.3, 0.7, TRUTH[2] - 0.2, 0.0])       # the weights start 0.2 um off the tube's centre
    fit, info = NI.fit_node_recentred(I, B, NOISE_GL, FORWARD, start, CFG, V, 0.6)
    print("    re-centred fit from 0.2 um off: v0 error %.2e um, %d solves, converged %s"
          % (fit.theta[2] - TRUTH[2], info["n_solves"], info["converged"]))
    assert fit.status == "converged" and info["converged"] and 2 <= info["n_solves"] <= CFG.inference.weight_irls_max_iter
    assert abs(info["centre_um"] - TRUTH[2]) <= 1e-3 and abs(fit.theta[2] - TRUTH[2]) <= 1e-6
    check_close("recentred_d_mu", fit.theta[:2], TRUTH[:2], rtol=1e-6)
    check_close("recentred_weights", info["weights"], LW.centre_weights(V, info["centre_um"], 0.6), rtol=1e-12)
    assert set(info) == {"n_solves", "centre_um", "converged", "lambda_um", "nfev", "weights"} and info["lambda_um"] == 0.6
    # the offset maps the fit's v0 onto the line's coordinate: weights centred at v0 + offset
    fit2, info2 = NI.fit_node_recentred(I, B, NOISE_GL, FORWARD, TRUTH, CFG, V, 0.6, centre_offset_um=0.3)
    assert abs(info2["centre_um"] - (TRUTH[2] + 0.3)) <= 1e-3
    # keep: the samples where it is False carry zero weight
    keep = V < TRUTH[2] + 0.8
    fit3, info3 = NI.fit_node_recentred(I, B, NOISE_GL, FORWARD, start, CFG, V, 0.6, keep=keep)
    assert np.all(info3["weights"][~keep] == 0) and np.all(info3["weights"][keep] > 0)
    check_close("recentred_keep_d", fit3.theta[0], TRUTH[0], rtol=1e-6)
    # the z box stays about the first start (not the re-starts)
    assert abs(fit.theta[3] - TRUTH[3]) <= 1e-6
    # determinism and inputs unchanged
    I0, s0 = I.copy(), start.copy()
    fit4, _ = NI.fit_node_recentred(I, B, NOISE_GL, FORWARD, start, CFG, V, 0.6)
    assert_array_equal(fit4.theta, fit.theta)
    assert_array_equal(I, I0)
    assert_array_equal(start, s0)
    _raises(ValueError, NI.fit_node_recentred, I, B, NOISE_GL, FORWARD, start, CFG, V, 0.6, keep=np.ones(3, dtype=bool))
    _raises(ValueError, NI.fit_node_recentred, I, B, NOISE_GL, FORWARD, start, CFG, V, -1.0)


# ------------------------------------ amendment 3: the SGT option, the fine-lattice error bars ---

FWHM_PER_SD = 2.0 * np.sqrt(2.0 * np.log(2.0))       # a Gaussian's FWHM over its SD: the SGT's FWHM is the law's (D-222.2)
GAUSS_SHAPE = tuple(CFG.inference.sgt_start_shape)    # (2, 0, 1), the Gaussian member: the SGT fit's start (D-224.1)
SGT_THETA = np.array([1.0, 1.5, 0.05, 0.1])           # oracle (i): d um, mu 1/um, v0 um, z_ax um
SGT_SHAPE = (1.4, 0.4, 1.0)                           # (beta, xi, zeta), zeta held under sgt_skew "off"
SGT_SHAPE_SKEW = (1.4, 0.4, 1.3)                      # with "fit" of zeta
OFF15 = np.array([1.15, 0.85, 1.15, 0.85])            # the start: theta 15 % off
CFG_SKEW = dataclasses.replace(CFG.inference, sgt_skew="fit")
CFG_ONE = dataclasses.replace(CFG.inference, multistart_factors=(1.0,))   # sequence (G): one start
STEPS = np.array(CFG.inference.cov_steps, dtype=float)                    # (d, mu, v0, z_ax, log10 beta, xi, ln zeta)


def make_sgt_forward(z_planes=Z_PLANES, v=V, n_nodes=N_NODES):
    """forward(d, mu, v0, z_ax, beta, xi, zeta): Block 3's model_profile rule (the chord profile on the
    Gauss-Legendre grid in psi, u = v0 + (d/2) sin psi, absorbed fraction 1 - exp(-mu d cos psi)) with the
    Gaussian replaced by the SGT kernel of FWHM 2 sqrt(2 ln 2) sigma_k(z_ax), sigma_k as make_forward's, read at
    v_n - u (offsets u > 0 towards increasing v take the kernel's right side, the wide one for zeta > 1, as Block
    49's kernel does)."""
    sin_psi, cos_psi, wq = M.quadrature_rule(n_nodes)

    def forward(d, mu, v0, z_ax, beta, xi, zeta):
        r = 0.5 * d
        mass = r * wq * cos_psi * (-np.expm1(-mu * d * cos_psi))     # absorbed fraction x du per node (um)
        dv = (v[:, None] - v0) - (r * sin_psi)[None, :]              # from each chord node to each sample (um)
        out = np.empty((z_planes.size, v.size))
        for k, zk in enumerate(z_planes):
            fwhm = FWHM_PER_SD * np.sqrt(S0 ** 2 + (GROWTH * (zk - z_ax)) ** 2)
            out[k] = 1.0 - BK.sgt_kernel_pdf(fwhm, beta, xi, zeta)(dv) @ mass
        return out
    return forward


SGT_FORWARD = make_sgt_forward()
SGT_FORWARD_FINE = make_sgt_forward(n_nodes=2 * N_NODES)   # forward_cov: the same model on a rule of twice the nodes


def gauss_member(d, mu, v0, z_ax):
    """The SGT stand-in at the Gaussian member, the forward of the Gaussian fit of sequence (G)."""
    return SGT_FORWARD(d, mu, v0, z_ax, *GAUSS_SHAPE)


class Recorder(object):
    """A forward that records every call's arguments (the calls reported; the points the error bars read)."""

    def __init__(self, f):
        self.f, self.calls = f, []

    def __call__(self, *args):
        self.calls.append(tuple(float(a) for a in args))
        return self.f(*args)


def search_vector(fit):
    """phi = (d, mu, v0, z_ax[, log10 beta, xi[, ln zeta]]) of a NodeFit (amendment 3 (A))."""
    if fit.shape is None:
        return np.array(fit.theta, dtype=float)
    s = [np.log10(fit.shape[0]), fit.shape[1], np.log(fit.shape[2])][:len(fit.param_names) - 4]
    return np.concatenate([fit.theta, s])


def tau_at(forward, phi, zeta_held=1.0):
    """forward at the search vector phi (4, 6 or 7 entries)."""
    if phi.size == 4:
        return forward(*phi)
    zeta = np.exp(phi[6]) if phi.size == 7 else zeta_held
    return forward(phi[0], phi[1], phi[2], phi[3], 10.0 ** phi[4], phi[5], zeta)


def residual_jacobian(forward, phi, planes, steps, rules=None, B=B_GL, sqrt_w=None, zeta_held=1.0):
    """The Jacobian of r = sqrt(w) (I - B tau) / sigma on the planes used, written here from SPEC Block 30,
    amendment 3 (D): rules[i] "c" central, (f(phi + h) - f(phi - h)) / (2 h); "+" / "-" the one-sided three-point
    rule inward, (-3 f(phi) + 4 f(phi +- h) - f(phi +- 2 h)) / (+-2 h). B: scalar or (n_planes,); sqrt_w:
    (n_planes, n_v) or None."""
    n = phi.size
    rules = ("c",) * n if rules is None else rules
    Bk = np.full(Z_PLANES.size, float(B)) if np.ndim(B) == 0 else np.asarray(B, dtype=float)
    cols = []
    for i in range(n):
        e = np.zeros(n)
        e[i] = steps[i]
        if rules[i] == "c":
            dtau = (tau_at(forward, phi + e, zeta_held) - tau_at(forward, phi - e, zeta_held)) / (2.0 * steps[i])
        else:
            s = 1.0 if rules[i] == "+" else -1.0
            dtau = (-3.0 * tau_at(forward, phi, zeta_held) + 4.0 * tau_at(forward, phi + s * e, zeta_held)
                    - tau_at(forward, phi + 2.0 * s * e, zeta_held)) / (2.0 * s * steps[i])
        col = -Bk[planes][:, None] * dtau[planes] / NOISE_GL
        if sqrt_w is not None:
            col = sqrt_w[planes] * col
        cols.append(col.ravel())
    return np.stack(cols, axis=1)


def scaled(cov):
    """cov_ij / sqrt(cov_ii cov_jj): the entries compared on the scale of the correlation (NaN rows stay NaN)."""
    sd = np.sqrt(np.diag(cov))
    return cov / np.outer(sd, sd)


def blank(cov, idx):
    """cov with the rows and columns idx set to NaN (the parameters at a bound, amendment 3 (B))."""
    out = np.array(cov, dtype=float, copy=True)
    out[idx, :] = np.nan
    out[:, idx] = np.nan
    return out


def test_sgt_known_answer():
    B = np.full(Z_PLANES.size, B_GL)
    th0 = SGT_THETA * OFF15
    for label, icfg, shape_t in (("zeta held", CFG.inference, SGT_SHAPE), ("zeta fitted", CFG_SKEW, SGT_SHAPE_SKEW)):
        I = B_GL * SGT_FORWARD(*SGT_THETA, *shape_t)
        rec = Recorder(SGT_FORWARD)
        t0 = time.perf_counter()
        fit = NI.fit_node(I, B, NOISE_GL, rec, th0, icfg, shape0=GAUSS_SHAPE)
        seconds = time.perf_counter() - t0
        n_p = 7 if icfg.sgt_skew == "fit" else 6
        print("    (i) %s: theta - truth = %s, shape / truth - 1 = %s; %s; %d starts, nfev %d, %d forward calls "
              "(the reported calls of an SGT fit), %.2f s"
              % (label, np.array2string(fit.theta - SGT_THETA, precision=2),
                 np.array2string(fit.shape / np.array(shape_t) - 1.0, precision=2), fit.status, fit.n_starts,
                 fit.nfev, len(rec.calls), seconds))
        assert fit.status == "converged" and fit.at_bound == (), (fit.status, fit.at_bound)
        assert fit.param_names == NI.PARAM_NAMES + NI.SHAPE_NAMES[:n_p - 4] and fit.cov_search.shape == (n_p, n_p)
        # SPEC Block 30, amendment 3 (i): relative 1e-5 on d, beta, xi, zeta (the truth is the exact minimum of the
        # stand-in, so the bound is the solver's convergence, not a discretisation)
        check_close("sgt_known_d", fit.theta[0], SGT_THETA[0], rtol=1e-5)
        check_close("sgt_known_shape", fit.shape, shape_t, rtol=1e-5)
        # beyond the oracle, the same standard: mu relative 1e-5, v0 and z_ax 1e-5 um
        check_close("sgt_known_mu", fit.theta[1], SGT_THETA[1], rtol=1e-5)
        check_close("sgt_known_v0_z", fit.theta[2:], SGT_THETA[2:], rtol=0.0, atol=1e-5)
        # every forward call inside the shape's box (the solver never leaves it; 1e-12: the 10^b, e^l round off)
        a = np.array(rec.calls)
        assert np.all((a[:, 4] >= 1.0 - 1e-12) & (a[:, 4] <= 10.0 * (1 + 1e-12))), "beta left [1, 10]"
        assert np.all((a[:, 5] >= 0.0) & (a[:, 5] <= 2.0)), "xi left [0, 2]"
        if icfg.sgt_skew == "off":
            assert np.all(a[:, 6] == 1.0), "zeta not held at zeta0 under 'off'"
        else:
            assert np.all(np.abs(np.log(a[:, 6])) <= 0.7 + 1e-12), "zeta left exp([-0.7, 0.7])"


def test_sgt_gaussian_truth():
    B = np.full(Z_PLANES.size, B_GL)
    I = B_GL * gauss_member(*TRUTH)
    # sequence (G) of amendment 3: the Gaussian fit as today, then the SGT fit from its estimate with the
    # Gaussian member, one start, the same planes and the same z box (D-224.1)
    fg = NI.fit_node(I, B, NOISE_GL, gauss_member, START, CFG)
    assert fg.status == "converged", fg.status
    rec = Recorder(SGT_FORWARD)
    fs = NI.fit_node(I, B, NOISE_GL, rec, fg.theta, CFG_ONE, planes=fg.planes, z_box_centre_um=START[3],
                     shape0=GAUSS_SHAPE)
    print("    (ii) Gaussian truth, sequence (G): %s, at_bound %s, shape %s, d / d_gauss - 1 = %.1e; %d forward calls"
          % (fs.status, fs.at_bound, np.array2string(fs.shape, precision=6), fs.theta[0] / fg.theta[0] - 1.0,
             len(rec.calls)))
    # D-225.5: xi = 0 is a member (the generalized normal), so xi_lo alone leaves the fit "converged"
    assert fs.at_bound == ("xi_lo",) and fs.status == "converged", (fs.status, fs.at_bound)
    # "ends at xi = 0": within scipy's active threshold, xtol (1e-8) x max(1, |lower bound|)
    assert 0.0 <= fs.shape[1] <= 1e-8, fs.shape
    check_close("gauss_truth_beta", fs.shape[0], 2.0, rtol=0.0, atol=1e-5)    # SPEC: beta = 2 [1e-5]
    check_close("gauss_truth_d", fs.theta[0], fg.theta[0], rtol=1e-6)          # SPEC: d the Gaussian fit's [rel 1e-6]
    assert fs.shape[2] == 1.0 and np.isnan(fs.se_shape[2])                     # zeta held
    # (B): the row and column of a parameter at a bound are NaN, xi_lo included; the rest finite
    assert np.all(np.isnan(fs.cov_search[5, :])) and np.all(np.isnan(fs.cov_search[:, 5])) and np.isnan(fs.se_shape[1])
    assert np.all(np.isfinite(fs.cov_search[:5, :5])) and np.isfinite(fs.se_shape[0]) and np.all(np.isfinite(fs.se))
    # from a start away from the Gaussian member (theta 15 % off, every start): the shape walks to the edge
    fo = NI.fit_node(I, B, NOISE_GL, SGT_FORWARD, TRUTH * OFF15, CFG, shape0=SGT_SHAPE)
    print("    (ii) from (1.4, 0.4, 1) and theta 15 %% off: %s, at_bound %s, shape %s, d / d_gauss - 1 = %.1e"
          % (fo.status, fo.at_bound, np.array2string(fo.shape, precision=6), fo.theta[0] / fg.theta[0] - 1.0))
    assert fo.at_bound == ("xi_lo",) and fo.status == "converged", (fo.status, fo.at_bound)
    assert 0.0 <= fo.shape[1] <= 1e-8, fo.shape
    check_close("gauss_truth_off_beta", fo.shape[0], 2.0, rtol=0.0, atol=1e-5)
    check_close("gauss_truth_off_d", fo.theta[0], fg.theta[0], rtol=1e-6)
    # any other active bound gives "at_bound" (B): beta beyond the box (12 > 10), alone and with xi_lo
    for xi_t, expect in ((0.4, ("log10_beta_hi",)), (0.0, ("log10_beta_hi", "xi_lo"))):
        Ib = B_GL * SGT_FORWARD(*SGT_THETA, 12.0, xi_t, 1.0)
        fb = NI.fit_node(Ib, B, NOISE_GL, SGT_FORWARD, SGT_THETA * OFF15, CFG, shape0=GAUSS_SHAPE)
        print("    beta 12 > 10, xi %.1f: %s, at_bound %s, shape %s" % (xi_t, fb.status, fb.at_bound,
                                                                        np.array2string(fb.shape, precision=4)))
        assert fb.status == "at_bound" and fb.at_bound == expect, (fb.status, fb.at_bound)
        check_close("beta_at_hi", fb.shape[0], 10.0, rtol=1e-7)
        assert np.all(np.isnan(fb.cov_search[4, :])) and np.isnan(fb.se_shape[0])
        assert np.all(np.isfinite(fb.cov_search[:4, :4]))


def test_sgt_coverage():
    rng = make_rng(SEED)          # the entry's 300 sets: test_invariants' seed and generator
    B = np.full(Z_PLANES.size, B_GL)
    d_hat, se, status, at_b, calls = [], [], [], [], []
    for _ in range(N_SETS):
        I = noisy_set(rng)
        fg = NI.fit_node(I, B, NOISE_GL, FORWARD, START, CFG)          # the entry's Gaussian fit
        rec = Recorder(SGT_FORWARD)
        fs = NI.fit_node(I, B, NOISE_GL, rec, fg.theta, CFG_ONE, planes=fg.planes, z_box_centre_um=START[3],
                         shape0=GAUSS_SHAPE)                             # sequence (G), zeta held
        d_hat.append(fs.theta[0])
        se.append(fs.se[0])
        status.append(fs.status)
        at_b.append(fs.at_bound)
        calls.append(len(rec.calls))
    d_hat, se = np.array(d_hat), np.array(se)
    cover = np.mean(np.abs(d_hat - TRUTH[0]) <= se)
    mean_d = np.mean(d_hat)
    sem = np.std(d_hat, ddof=1) / np.sqrt(N_SETS)
    n_xi = sum(a == ("xi_lo",) for a in at_b)
    print("    (iii) %d sets, SGT fit (zeta held) after the Gaussian fit: %d converged (%d with xi_lo alone); coverage "
          "of d_hat +- se = %.3f (nominal 0.683); mean d_hat - d = %.5f um (%.2f SE); mean se = %.5f um, SD of "
          "d_hat = %.5f um; %.1f forward calls per SGT fit" % (N_SETS, status.count("converged"), n_xi, cover,
                                                                 mean_d - TRUTH[0], (mean_d - TRUTH[0]) / sem,
                                                                 np.mean(se), np.std(d_hat, ddof=1), np.mean(calls)))
    assert status.count("converged") == N_SETS, "not every set converged: %s" % sorted(set(status))
    # SPEC amendment 3 (iii): coverage in [60, 76] % (binomial SD 2.7 % at 300 sets) and the mean within 3 SE
    assert 0.60 <= cover <= 0.76, "coverage %.3f outside [0.60, 0.76]" % cover
    assert abs(mean_d - TRUTH[0]) <= 3.0 * sem, "bias %.5f um > 3 SE (%.5f um)" % (mean_d - TRUTH[0], sem)


def test_fine_errors():
    B = np.full(Z_PLANES.size, B_GL)
    th0 = SGT_THETA * OFF15
    # (a) an interior estimate of the seven parameters, forward_cov = forward
    rng = make_rng(SEED + 10)
    I = B_GL * SGT_FORWARD(*SGT_THETA, *SGT_SHAPE_SKEW) + rng.normal(0.0, NOISE_GL, size=(Z_PLANES.size, V.size))
    plain = NI.fit_node(I, B, NOISE_GL, SGT_FORWARD, th0, CFG_SKEW, shape0=GAUSS_SHAPE)
    rec = Recorder(SGT_FORWARD)
    fine = NI.fit_node(I, B, NOISE_GL, SGT_FORWARD, th0, CFG_SKEW, shape0=GAUSS_SHAPE, forward_cov=rec)
    assert plain.cov_source == "fit" and fine.cov_source == "fine"
    for f in ("theta", "shape", "chi2_per_plane", "planes"):        # the fit itself is not touched
        assert_array_equal(getattr(fine, f), getattr(plain, f))
    assert (fine.cost, fine.status, fine.at_bound, fine.nfev) == (plain.cost, plain.status, plain.at_bound, plain.nfev)
    assert fine.status == "converged" and fine.at_bound == (), (fine.status, fine.at_bound)
    phi = search_vector(fine)
    ic = CFG_SKEW
    lo = np.array([ic.d_bounds_um[0], ic.mu_bounds_per_um[0], ic.v0_bounds_um[0], th0[3] + ic.z_bounds_um[0],
                   ic.sgt_log10_beta_bounds[0], ic.sgt_xi_bounds[0], ic.sgt_ln_zeta_bounds[0]])
    hi = np.array([ic.d_bounds_um[1], ic.mu_bounds_per_um[1], ic.v0_bounds_um[1], th0[3] + ic.z_bounds_um[1],
                   ic.sgt_log10_beta_bounds[1], ic.sgt_xi_bounds[1], ic.sgt_ln_zeta_bounds[1]])
    assert np.all(phi - lo > 2.0 * STEPS) and np.all(hi - phi > 2.0 * STEPS), "the case must be interior"
    assert len(rec.calls) == 2 * 7, len(rec.calls)          # central differences only: two calls per parameter
    J = residual_jacobian(SGT_FORWARD, phi, fine.planes, STEPS)
    # the rule as written here, at the same steps [1e-10: rounding of phi's 10^b, e^l round trip]
    check_close("fine_central_rule", scaled(fine.cov_search), scaled(np.linalg.inv(J.T @ J)), rtol=0.0, atol=1e-10)
    # SPEC amendment 3 (iv): against (J^T J)^{-1} of a Richardson-extrapolated Jacobian, (4 J(h/2) - J(h)) / 3
    # [relative 1e-3]. The stand-in is C1 but not C2 in d and v0 at beta < 2 (the kernel's mode crosses the
    # quadrature nodes): two Richardson pairs, (h, h/2) and (h/2, h/4), differ by up to 1.7e-4 in the error bars
    # (measured on this geometry when the test was written), the floor of this reference
    JR = (4.0 * residual_jacobian(SGT_FORWARD, phi, fine.planes, STEPS / 2.0) - J) / 3.0
    cov_R = np.linalg.inv(JR.T @ JR)
    se_fine, se_R = np.sqrt(np.diag(fine.cov_search)), np.sqrt(np.diag(cov_R))
    se_fit = np.sqrt(np.diag(plain.cov_search))
    print("    (iv) error bars on the stand-in, se / se_Richardson - 1: fine %s; the solver's own (cov_source "
          "'fit') %s; %d forward_cov calls (the reported calls of the error bars)"
          % (np.array2string(se_fine / se_R - 1.0, precision=1), np.array2string(se_fit / se_R - 1.0, precision=1),
             len(rec.calls)))
    check_close("fine_se_richardson", se_fine, se_R, rtol=1e-3)
    check_close("fine_corr_richardson", scaled(fine.cov_search), scaled(cov_R), rtol=0.0, atol=1e-3)
    check_close("fine_theta_block", fine.cov, fine.cov_search[:4, :4], rtol=0.0, atol=0.0)
    # (b) a parameter at its lower bound takes the one-sided rule upwards: xi at xi_lo (a Gaussian truth, sequence (G))
    Ig = B_GL * gauss_member(*TRUTH)
    fg = NI.fit_node(Ig, B, NOISE_GL, gauss_member, START, CFG)
    rec = Recorder(SGT_FORWARD)
    fs = NI.fit_node(Ig, B, NOISE_GL, SGT_FORWARD, fg.theta, CFG_ONE, planes=fg.planes, z_box_centre_um=START[3],
                     shape0=GAUSS_SHAPE, forward_cov=rec)
    assert fs.at_bound == ("xi_lo",) and fs.cov_source == "fine", (fs.at_bound, fs.cov_source)
    phi = search_vector(fs)
    xis = np.array([c[5] for c in rec.calls])
    assert np.all(xis >= 0.0), "forward_cov read beyond the bound xi = 0"
    assert len(rec.calls) == 1 + 2 + 2 * 5, len(rec.calls)   # f(phi) once, xi + h, xi + 2h, five central pairs
    assert np.allclose(np.unique(xis), [phi[5], phi[5] + STEPS[5], phi[5] + 2.0 * STEPS[5]], rtol=0.0, atol=1e-15)
    J = residual_jacobian(SGT_FORWARD, phi, fs.planes, STEPS, rules=("c",) * 5 + ("+",))
    check_close("fine_one_sided_lo", scaled(fs.cov_search), scaled(blank(np.linalg.inv(J.T @ J), [5])),
                rtol=0.0, atol=1e-10)
    # (c) within two steps of an upper bound (v0 0.003 um under it, 2 x 0.002 um the margin), not at it: the
    #     one-sided rule downwards, and the row stays finite
    icfg = dataclasses.replace(CFG.inference, v0_bounds_um=(-1.0, SGT_THETA[2] + 0.003))
    Iu = B_GL * SGT_FORWARD(*SGT_THETA, *SGT_SHAPE)
    rec = Recorder(SGT_FORWARD)
    fu = NI.fit_node(Iu, B, NOISE_GL, SGT_FORWARD, th0, icfg, shape0=GAUSS_SHAPE, forward_cov=rec)
    assert fu.status == "converged" and fu.at_bound == (), (fu.status, fu.at_bound)
    phi = search_vector(fu)
    v0s = np.array([c[2] for c in rec.calls])
    assert np.all(v0s <= SGT_THETA[2] + 0.003) and len(rec.calls) == 1 + 2 + 2 * 5
    assert np.allclose(np.unique(v0s), [phi[2] - 2.0 * STEPS[2], phi[2] - STEPS[2], phi[2]], rtol=0.0, atol=1e-15)
    J = residual_jacobian(SGT_FORWARD, phi, fu.planes, STEPS[:6], rules=("c", "c", "-", "c", "c", "c"))
    check_close("fine_one_sided_hi", scaled(fu.cov_search), scaled(np.linalg.inv(J.T @ J)), rtol=0.0, atol=1e-10)
    assert np.all(np.isfinite(fu.cov_search))
    # (d) the Gaussian path: forward_cov of forward's arity (4), the first four steps; the fit unchanged
    rng = make_rng(SEED + 11)
    In = noisy_set(rng)
    g0 = NI.fit_node(In, B, NOISE_GL, FORWARD, START, CFG)
    rec = Recorder(FORWARD)
    g1 = NI.fit_node(In, B, NOISE_GL, FORWARD, START, CFG, forward_cov=rec)
    assert_array_equal(g1.theta, g0.theta)
    assert g0.cov_source == "fit" and g1.cov_source == "fine" and len(rec.calls) == 8 and g1.shape is None
    J = residual_jacobian(FORWARD, g1.theta, g1.planes, STEPS[:4])
    check_close("fine_gauss", scaled(g1.cov), scaled(np.linalg.inv(J.T @ J)), rtol=0.0, atol=1e-10)
    assert_array_equal(g1.cov_search, g1.cov)
    # (e) a non-finite forward_cov at a point the differences need: the error bars all NaN, the fit unchanged
    def nan_beyond(*a):
        out = SGT_FORWARD(*a)
        return out * np.nan if a[0] > fine.theta[0] + 0.5 * STEPS[0] else out
    fn = NI.fit_node(I, B, NOISE_GL, SGT_FORWARD, th0, CFG_SKEW, shape0=GAUSS_SHAPE, forward_cov=nan_beyond)
    assert fn.cov_source == "fine" and fn.status == fine.status
    assert_array_equal(fn.theta, fine.theta)
    for f in ("cov_search", "cov", "se", "se_shape"):
        assert np.all(np.isnan(getattr(fn, f))), f
    # ... and NaN on a plane the fit does not use is not needed: the error bars as with the clean model
    planes = np.ones(Z_PLANES.size, dtype=bool)
    planes[0] = False

    def nan_plane0(*a):
        out = SGT_FORWARD(*a)
        out[0] = np.nan
        return out
    fp = NI.fit_node(I, B, NOISE_GL, SGT_FORWARD, th0, CFG_SKEW, planes=planes, shape0=GAUSS_SHAPE,
                     forward_cov=nan_plane0)
    fq = NI.fit_node(I, B, NOISE_GL, SGT_FORWARD, th0, CFG_SKEW, planes=planes, shape0=GAUSS_SHAPE,
                     forward_cov=SGT_FORWARD)
    assert np.all(np.isfinite(fp.cov_search))
    assert_array_equal(fp.cov_search, fq.cov_search)


def test_fine_sandwich():
    rng = make_rng(SEED + 12)
    Bk = np.linspace(190.0, 230.0, Z_PLANES.size)                 # backgrounds per plane
    I = Bk[:, None] * SGT_FORWARD(*SGT_THETA, *SGT_SHAPE_SKEW) + rng.normal(0.0, NOISE_GL, size=(Z_PLANES.size, V.size))
    w = np.tile(LW.centre_weights(V, SGT_THETA[2], 0.6), (Z_PLANES.size, 1))
    w[:, V > SGT_THETA[2] + 1.5] = 0.0                            # a cut: samples of zero weight
    w[1] *= 0.5                                                   # weights per plane
    planes = np.ones(Z_PLANES.size, dtype=bool)
    planes[0] = False                                             # a plane not used
    rec = Recorder(SGT_FORWARD_FINE)
    fit = NI.fit_node(I, Bk, NOISE_GL, SGT_FORWARD, SGT_THETA * OFF15, CFG_SKEW, planes=planes, weights=w,
                      shape0=GAUSS_SHAPE, forward_cov=rec)
    assert fit.status == "converged" and fit.at_bound == () and fit.cov_source == "fine", (fit.status, fit.at_bound)
    assert len(rec.calls) == 2 * 7
    phi = search_vector(fit)
    sw = np.sqrt(w)
    # SPEC amendment 2: A = J_w^T J_w, B = (sqrt(w) J_w)^T (sqrt(w) J_w), cov = A^{-1} B A^{-1}, J_w the Jacobian of
    # sqrt(w) r / sigma, here by the central differences of forward_cov (amendment 3 (D)), written by hand
    J_w = residual_jacobian(SGT_FORWARD_FINE, phi, planes, STEPS, B=Bk, sqrt_w=sw)
    A_inv = np.linalg.inv(J_w.T @ J_w)
    JB = sw[planes].ravel()[:, None] * J_w
    C = A_inv @ (JB.T @ JB) @ A_inv
    print("    (v) weighted sandwich with the shape: max |scaled difference| = %.1e; se %s, se_shape %s"
          % (np.max(np.abs(scaled(fit.cov_search) - scaled(C))), np.array2string(fit.se, precision=4),
             np.array2string(fit.se_shape, precision=4)))
    # [1e-10]: the same arithmetic, up to the rounding of phi's 10^b, e^l round trip
    check_close("sandwich_hand", scaled(fit.cov_search), scaled(C), rtol=0.0, atol=1e-10)
    check_close("sandwich_diag", np.diag(fit.cov_search), np.diag(C), rtol=1e-10)
    # (C): theta's block, its SE, and se_shape by the delta method: beta ln 10 se(b), se(xi), zeta se(l)
    assert_array_equal(fit.cov, fit.cov_search[:4, :4])
    check_close("sandwich_se", fit.se, np.sqrt(np.diag(fit.cov)), rtol=1e-15)
    sd = np.sqrt(np.diag(fit.cov_search))
    check_close("sandwich_se_shape", fit.se_shape,
                [fit.shape[0] * np.log(10.0) * sd[4], sd[5], fit.shape[2] * sd[6]], rtol=1e-12)


def test_sgt_recentred():
    B = np.full(Z_PLANES.size, B_GL)
    th0 = np.array([1.15, 1.275, SGT_THETA[2] - 0.2, 0.085])     # the weights start 0.2 um off the tube's centre
    I = B_GL * SGT_FORWARD(*SGT_THETA, *SGT_SHAPE_SKEW)
    rec = Recorder(SGT_FORWARD_FINE)
    fit, info = NI.fit_node_recentred(I, B, NOISE_GL, SGT_FORWARD, th0, CFG_SKEW, V, 0.6, shape0=GAUSS_SHAPE,
                                      forward_cov=rec)
    print("    (vi) re-centred SGT fit from 0.2 um off: %d solves, converged %s, d error %.1e um, shape %s; %d "
          "forward_cov calls" % (info["n_solves"], info["converged"], fit.theta[0] - SGT_THETA[0],
                                 np.array2string(fit.shape, precision=6), len(rec.calls)))
    assert fit.status == "converged" and info["converged"]
    assert 2 <= info["n_solves"] <= CFG.inference.weight_irls_max_iter
    assert set(info) == {"n_solves", "centre_um", "converged", "lambda_um", "nfev", "weights"}
    # (E): the error bars once, after the last solve (seven parameters, interior: 14 calls)
    assert fit.cov_source == "fine" and len(rec.calls) == 2 * 7, (fit.cov_source, len(rec.calls))
    check_close("recentred_sgt_truth", np.concatenate([fit.theta, fit.shape]),
                np.concatenate([SGT_THETA, SGT_SHAPE_SKEW]), rtol=1e-5)
    # SPEC amendment 3 (vi): it ends where fit_node with its last weights ends [1e-8] (from theta0, every start)
    ref = NI.fit_node(I, B, NOISE_GL, SGT_FORWARD, th0, CFG_SKEW, planes=fit.planes, weights=info["weights"],
                      z_box_centre_um=th0[3], shape0=GAUSS_SHAPE, forward_cov=SGT_FORWARD_FINE)
    check_close("recentred_sgt_theta", fit.theta, ref.theta, rtol=1e-8)
    check_close("recentred_sgt_shape", fit.shape, ref.shape, rtol=1e-8)
    check_close("recentred_sgt_cov", scaled(fit.cov_search), scaled(ref.cov_search), rtol=0.0, atol=1e-8)
    check_close("recentred_sgt_se", fit.se, ref.se, rtol=1e-8)
    # on a noisy set fit_node from theta0 agrees only to the solver's tolerance (about 1e-6 in the shape, measured
    # when the test was written); the sequence of (E) written out with fit_node is the same computation, bit for bit
    rng = make_rng(SEED + 13)
    In = I + rng.normal(0.0, NOISE_GL, size=I.shape)
    fr, ir = NI.fit_node_recentred(In, B, NOISE_GL, SGT_FORWARD, th0, CFG_SKEW, V, 0.6, shape0=GAUSS_SHAPE,
                                   forward_cov=SGT_FORWARD_FINE)
    one = dataclasses.replace(CFG_SKEW, multistart_factors=(1.0,))
    w = LW.centre_weights(V, th0[2], 0.6)
    f = NI.fit_node(In, B, NOISE_GL, SGT_FORWARD, th0, CFG_SKEW, weights=w, z_box_centre_um=th0[3],
                    shape0=GAUSS_SHAPE, forward_cov=SGT_FORWARD_FINE if ir["n_solves"] == 1 else None)
    for k in range(1, ir["n_solves"]):
        w = LW.centre_weights(V, f.theta[2], 0.6)
        f = NI.fit_node(In, B, NOISE_GL, SGT_FORWARD, f.theta, one, planes=f.planes, weights=w,
                        z_box_centre_um=th0[3], shape0=f.shape,
                        forward_cov=SGT_FORWARD_FINE if k == ir["n_solves"] - 1 else None)
    print("    (vi) noisy set: %d solves; the sequence written out equal bit for bit" % ir["n_solves"])
    for name in ("theta", "shape", "cov_search", "se_shape", "cov", "se", "chi2_per_plane", "planes"):
        assert_array_equal(getattr(fr, name), getattr(f, name))
    assert (fr.cost, fr.status, fr.at_bound, fr.cov_source) == (f.cost, f.status, f.at_bound, f.cov_source)
    assert_array_equal(ir["weights"], w)
    # without forward_cov the solver's own error bars, and the same estimate
    fx, _ = NI.fit_node_recentred(In, B, NOISE_GL, SGT_FORWARD, th0, CFG_SKEW, V, 0.6, shape0=GAUSS_SHAPE)
    assert fx.cov_source == "fit"
    assert_array_equal(fx.theta, fr.theta)
    assert_array_equal(fx.shape, fr.shape)
    _raises(ValueError, NI.fit_node_recentred, I, B, NOISE_GL, SGT_FORWARD, th0, CFG_SKEW, V, 0.6,
            shape0=(20.0, 0.0, 1.0))


def test_gaussian_fields():
    B = np.full(Z_PLANES.size, B_GL)
    rng = make_rng(SEED + 2)
    I = noisy_set(rng)
    fits = [NI.fit_node(I, B, NOISE_GL, FORWARD, START, CFG),
            NI.fit_node(I, B, NOISE_GL, FORWARD, START, CFG, weights=LW.centre_weights(V, TRUTH[2], 0.6)),
            NI.fit_node_recentred(I, B, NOISE_GL, FORWARD, START, CFG, V, 0.6)[0],
            NI.fit_node(I, B, NOISE_GL, lambda d, mu, v0, z: np.full(I.shape, np.nan), START, CFG)]
    for fit in fits:
        # SPEC amendment 3 (C), (vii): a Gaussian fit has shape and se_shape None, param_names PARAM_NAMES and
        # cov_search equal to cov; its error bars are the fit's own
        assert fit.shape is None and fit.se_shape is None and fit.param_names == NI.PARAM_NAMES
        assert fit.cov_source == "fit" and fit.cov_search.shape == (4, 4)
        assert_array_equal(fit.cov_search, fit.cov)
        assert fit.cov_search is not fit.cov
    assert fits[-1].status == "failed"
    # a NodeFit built without the new fields (other blocks' tests do) takes the Gaussian's defaults
    nf = NI.NodeFit(theta=np.zeros(4), cov=np.eye(4), se=np.ones(4), cost=0.0, chi2_per_plane=np.ones(7),
                    planes=np.ones(7, dtype=bool), status="converged", at_bound=(), n_starts=1, nfev=1)
    assert nf.shape is None and nf.se_shape is None and nf.param_names == NI.PARAM_NAMES
    assert nf.cov_search is None and nf.cov_source == "fit"


def test_sgt_contract():
    B = np.full(Z_PLANES.size, B_GL)
    rng = make_rng(SEED + 14)
    I = B_GL * SGT_FORWARD(*SGT_THETA, *SGT_SHAPE_SKEW) + rng.normal(0.0, NOISE_GL, size=(Z_PLANES.size, V.size))
    th0 = SGT_THETA * OFF15
    shape0 = np.array(GAUSS_SHAPE)
    w = LW.centre_weights(V, SGT_THETA[2], 0.6)
    copies = [x.copy() for x in (I, B, th0, shape0, w)]
    a = NI.fit_node(I, B, NOISE_GL, SGT_FORWARD, th0, CFG_SKEW, weights=w, shape0=shape0, forward_cov=SGT_FORWARD_FINE)
    b = NI.fit_node(I, B, NOISE_GL, SGT_FORWARD, th0, CFG_SKEW, weights=w, shape0=shape0, forward_cov=SGT_FORWARD_FINE)
    for name in ("theta", "cov", "se", "chi2_per_plane", "planes", "shape", "se_shape", "cov_search"):
        assert_array_equal(getattr(a, name), getattr(b, name))
    assert (a.cost, a.status, a.at_bound, a.nfev, a.param_names, a.cov_source) == \
        (b.cost, b.status, b.at_bound, b.nfev, b.param_names, b.cov_source)
    for x, x0 in zip((I, B, th0, shape0, w), copies):
        assert_array_equal(x, x0)
    # the fields (F)
    assert a.status == "converged" and a.param_names == ("d", "mu", "v0", "z", "log10_beta", "xi", "ln_zeta")
    assert a.cov_source in NI.COV_SOURCES and a.status in NI.FIT_STATUSES
    assert a.theta.shape == (4,) and a.theta.dtype == np.float64 and a.cov.shape == (4, 4)
    assert a.shape.shape == (3,) and a.shape.dtype == np.float64 and a.se_shape.shape == (3,)
    assert a.cov_search.shape == (7, 7) and np.all(np.isfinite(a.cov_search))
    # symmetric to rounding: A^{-1} B A^{-1} is not symmetrised; compared on the correlation's scale [1e-12]
    check_close("contract_sgt_symmetric", scaled(a.cov_search), scaled(a.cov_search).T, rtol=0.0, atol=1e-12)
    assert_array_equal(a.cov, a.cov_search[:4, :4])
    # a DiameterConfig and its InferenceConfig give the same fit
    c = NI.fit_node(I, B, NOISE_GL, SGT_FORWARD, th0, dataclasses.replace(CFG, inference=CFG_SKEW), weights=w,
                    shape0=shape0, forward_cov=SGT_FORWARD_FINE)
    assert_array_equal(c.theta, a.theta)
    assert_array_equal(c.cov_search, a.cov_search)
    # zeta held ("off"): six search parameters, zeta = zeta0 passed to the forward as given, se_shape NaN for it
    rec = Recorder(SGT_FORWARD)
    h = NI.fit_node(I, B, NOISE_GL, rec, th0, CFG_ONE, shape0=(2.0, 0.0, 1.2))
    assert h.cov_search.shape == (6, 6) and h.shape[2] == 1.2 and np.isnan(h.se_shape[2])
    assert all(cl[6] == 1.2 for cl in rec.calls)
    # the planes are selected on the start, the shape included (A)
    tau0 = SGT_FORWARD(*th0, *GAUSS_SHAPE)
    peak = B_GL * np.max(1.0 - tau0, axis=1) / NOISE_GL
    cut = 0.5 * (np.sort(peak)[1] + np.sort(peak)[2])           # between the two weakest planes and the rest
    cfg_cut = dataclasses.replace(CFG_SKEW, plane_snr_min=float(cut))
    fc = NI.fit_node(I, B, NOISE_GL, SGT_FORWARD, th0, cfg_cut, shape0=GAUSS_SHAPE)
    assert_array_equal(fc.planes, NI.select_planes(tau0, B, NOISE_GL, cfg_cut))
    assert 0 < fc.planes.sum() < Z_PLANES.size
    # a failed SGT fit (a forward that is NaN everywhere): NaN estimate, the SGT's names, forward_cov not read
    nan7 = lambda *args: np.full((Z_PLANES.size, V.size), np.nan)  # noqa: E731
    rec = Recorder(SGT_FORWARD)
    for planes in (None, np.ones(Z_PLANES.size, dtype=bool)):
        f = NI.fit_node(I, B, NOISE_GL, nan7, th0, CFG_SKEW, planes=planes, shape0=GAUSS_SHAPE, forward_cov=rec)
        assert f.status == "failed" and np.all(np.isnan(f.theta)) and np.all(np.isnan(f.shape)), f.status
        assert f.cov_search.shape == (7, 7) and np.all(np.isnan(f.cov_search)) and np.all(np.isnan(f.se_shape))
        assert f.param_names == NI.PARAM_NAMES + NI.SHAPE_NAMES and f.cov_source == "fine"
    assert len(rec.calls) == 0
    # (F) ValueError: shape0 not three finite numbers with beta > 0, zeta > 0, or outside the box
    for bad in ((2.0, 0.0), (2.0, 0.0, 1.0, 1.0), (np.nan, 0.0, 1.0), (2.0, np.inf, 1.0), (0.0, 0.0, 1.0),
                (-2.0, 0.0, 1.0), (2.0, 0.0, 0.0), (2.0, 0.0, -1.0), ("a", 0.0, 1.0),
                (20.0, 0.0, 1.0), (0.5, 0.0, 1.0), (2.0, -0.1, 1.0), (2.0, 3.0, 1.0)):
        for icfg in (CFG.inference, CFG_SKEW):
            _raises(ValueError, NI.fit_node, I, B, NOISE_GL, SGT_FORWARD, th0, icfg, shape0=bad)
    # zeta outside exp(sgt_ln_zeta_bounds): refused under "fit" and, held, under "off" (config.validate's rule
    # for sgt_start_shape)
    for icfg in (CFG.inference, CFG_SKEW):
        _raises(ValueError, NI.fit_node, I, B, NOISE_GL, SGT_FORWARD, th0, icfg, shape0=(2.0, 0.0, 3.0))
        _raises(ValueError, NI.fit_node, I, B, NOISE_GL, SGT_FORWARD, th0, icfg, shape0=(2.0, 0.0, np.exp(0.7 + 1e-9)))
    # the box's edges are inside, and a start an ulp outside (the 10^b, e^l round off) is taken into it
    quick = dataclasses.replace(CFG_SKEW, multistart_factors=(1.0,), max_nfev=3)
    for edge in ((1.0, 0.0, np.exp(-0.7)), (10.0, 2.0, np.exp(0.7)), (2.0, 0.0, np.exp(0.7) * (1.0 + 1e-15))):
        f = NI.fit_node(I, B, NOISE_GL, SGT_FORWARD, th0, quick, shape0=edge)
        assert f.status in NI.FIT_STATUSES
    # forward_cov: callable or None; the steps it needs finite and > 0
    _raises(TypeError, NI.fit_node, I, B, NOISE_GL, SGT_FORWARD, th0, CFG_SKEW, shape0=GAUSS_SHAPE, forward_cov=1.0)
    _raises(TypeError, NI.fit_node_recentred, I, B, NOISE_GL, SGT_FORWARD, th0, CFG_SKEW, V, 0.6,
            shape0=GAUSS_SHAPE, forward_cov=1.0)
    for steps in ((1e-3,) * 6, (2e-3, 1e-3, 0.0, 2e-3, 1e-3, 1e-3, 1e-3), (2e-3, np.nan, 2e-3, 2e-3, 1e-3, 1e-3, 1e-3)):
        _raises(ValueError, NI.fit_node, I, B, NOISE_GL, SGT_FORWARD, th0,
                dataclasses.replace(CFG_SKEW, cov_steps=steps), shape0=GAUSS_SHAPE, forward_cov=SGT_FORWARD)


def _environment():
    parts = [f"python {platform.python_version()}"]
    for package in REPORT_PACKAGES:
        try:
            parts.append(f"{package} {importlib.metadata.version(package)}")
        except importlib.metadata.PackageNotFoundError:
            parts.append(f"{package} (not installed)")
    parts.append(platform.platform())
    parts.append(f"seed {SEED}")
    return " | ".join(parts)


def main():
    checks = [(name, obj) for name, obj in globals().items()
              if name.startswith("test_") and callable(obj)]
    print(f"== {Path(__file__).name}")
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
        except Exception:  # unexpected errors are reported, never hidden
            status, detail = "ERROR", traceback.format_exc()
        results.append((name, status, time.perf_counter() - start, detail))

    width = max([len(name) for name, _, _, _ in results] + [4])
    for name, status, seconds, detail in results:
        lines = detail.strip().splitlines()
        headline = lines[0] if lines else ""
        print(f"{status:<5}  {name:<{width}}  {seconds:8.3f}s  {headline}".rstrip())

    for name, status, _seconds, detail in results:
        if status in ("FAIL", "ERROR"):
            print(f"\n---- {status}: {name}\n{detail.strip()}")

    counts = {s: sum(1 for r in results if r[1] == s)
              for s in ("PASS", "FAIL", "ERROR", "TODO", "SKIP")}
    print("\n-- " + ", ".join(f"{n} {s.lower()}" for s, n in counts.items()))
    for path in _SAVED_ARTIFACTS:
        print(f"-- arrays from a failed comparison: {path}")
    failed = counts["FAIL"] + counts["ERROR"] + counts["TODO"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
