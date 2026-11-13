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

from allen_diameter.analysis import node_inference as NI  # noqa: E402
from allen_diameter.config import default_config  # noqa: E402
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
