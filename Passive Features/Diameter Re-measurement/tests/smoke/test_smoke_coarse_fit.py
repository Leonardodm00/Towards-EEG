"""Smoke test for the coarse refinement of a node's features -- Block 29 in specs/SPEC.md.

The shadow model `predict` is the closed form the Block 29 entry describes:
a flat tube of diameter d and coefficient mu at lateral shift v0, its shadow
in plane k = calibration_scene.shadow_depths(|v_n - v0|, d/2, mu,
sigma_r(|z_k - dz|), 1) (dimensionless, B = 1). Block 25 (model/shadow.py) is
not imported.

Checks
    test_known_answer   noise-free recovery of (d, mu, v0, dz) from a start
                        30 % off [relative 1e-6 on d, mu; 1e-6 um on shifts]
    test_reference      cost and rms_gl against the objective written by hand
    test_convergence    skipped: not a discretisation
    test_invariants     100 seeded noisy draws (SD 3 gl, B = 210): mean d-hat
                        within 3 standard errors of the truth; the inputs are
                        not changed
    test_contract       keys, types, status names; offset_xyz
    test_determinism    identical output for identical input
    test_edge_cases     truth outside the bounds -> at_bound with the name;
                        a NaN predict -> failed without raising; ValueError
                        for shapes that disagree, B <= 0, a start outside the
                        bounds

Run (from "Passive Features/Diameter Re-measurement")
    python3 tests/smoke/test_smoke_coarse_fit.py
    python3 -m pytest tests/smoke/test_smoke_coarse_fit.py -q

Debug
    A failed check_close() saves both arrays to
        tests/smoke/artifacts/<check name>.npz

Pure ASCII, for the cluster.
"""
from __future__ import annotations

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
SRC = HERE.parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from allen_diameter.analysis import coarse_fit as CF  # noqa: E402
from allen_diameter.analysis.calibration_scene import shadow_depths  # noqa: E402
from allen_diameter.analysis.profiles import profile_offsets  # noqa: E402
from allen_diameter.config import default_config  # noqa: E402
from allen_diameter.model.kernel import sigma_r  # noqa: E402

SEED = 12345
REPORT_PACKAGES = ("numpy", "scipy")
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
_SAVED_ARTIFACTS = []  # files written by check_close during this run

B_GL = 210.0          # the entry's background (gl)
NOISE_SD_GL = 3.0     # the entry's noise SD (gl)
N_DRAWS = 100         # the entry's number of noisy draws
TRUTH = (0.8, 1.5, 0.12, 0.1)   # (d um, mu 1/um, v0 um, dz um), inside the default bounds
PLANES = np.arange(-3, 4)       # plane indices about the node's plane


# --------------------------------------------------------------- helpers ---

def make_rng(seed: int = SEED) -> np.random.Generator:
    """Fresh generator, to be passed explicitly to the code under test."""
    return np.random.default_rng(seed)


def check_close(name, actual, expected, rtol=1e-12, atol=0.0):
    """numpy.testing.assert_allclose that keeps the evidence on failure.

    On failure both arrays are saved to ARTIFACT_DIR/<name>.npz before the
    AssertionError propagates. Justify rtol and atol at the call site
    (float64 roundoff, discretisation-error estimate, statistical spread).
    """
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


def observed_order(h, err):
    """Least-squares slope p of log e(h) = log C + p * log h.

    h: step sizes (any order); err: positive errors e(h) at those steps.
    """
    h = np.asarray(h, dtype=float)
    err = np.asarray(err, dtype=float)
    if np.any(err <= 0.0):
        raise ValueError("observed_order needs strictly positive errors")
    slope, _intercept = np.polyfit(np.log(h), np.log(err), 1)
    return float(slope)


# -------------------------------------------------------------- fixtures ---

def make_case(cfg=None):
    """(cfg, v (n_v,), z (n_planes,), predict): the entry's closed-form model."""
    cfg = default_config() if cfg is None else cfg
    v = profile_offsets(cfg.measure)
    z = PLANES * float(cfg.acquisition.dz_um)
    rcfg = cfg.renderer

    def predict(d, mu, v0, dz):
        sig = sigma_r(np.abs(z - dz), rcfg)
        return np.stack([shadow_depths(np.abs(v - v0), d / 2.0, mu, s, 1.0) for s in sig])

    return cfg, v, z, predict


def observed(predict, theta, n_planes):
    B = np.full(n_planes, B_GL)
    return B[:, None] * (1.0 - predict(*theta)), B


def theta_of(out):
    return np.array([out[k] for k in CF.PARAM_NAMES])


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    # Noise-free data: the truth is a zero of every residual, so the solver
    # reaches it to its own termination tolerance (xtol 1e-8 relative); the
    # entry's tolerances are 1e-6 relative on d and mu, 1e-6 um on the shifts.
    cfg, v, z, predict = make_case()
    truths = [TRUTH, (1.6, 0.6, -0.2, -0.15), (0.4, 3.0, 0.05, 0.2)]
    for truth in truths:
        I, B = observed(predict, truth, z.size)
        for factor in (1.3, 0.7):
            th0 = np.array(truth) * factor
            out = CF.refine_node(I, B, v, predict, th0, cfg)
            est = theta_of(out)
            tag = "truth %r start x%.1f" % (truth, factor)
            assert out["status"] == "converged", (tag, out)
            check_close("known_answer d mu " + tag, est[:2], truth[:2], rtol=1e-6, atol=0.0)
            check_close("known_answer shifts " + tag, est[2:], truth[2:], rtol=0.0, atol=1e-6)
            print("    %s: d %.12f mu %.12f v0 %.12f dz %.12f; err %s; nfev %d"
                  % (tag, est[0], est[1], est[2], est[3], np.array2string(est - np.array(truth), precision=2), out["nfev"]))


def test_reference():
    # cost = sum_k sum_n w_n r_kn^2 and rms_gl = sqrt(mean r_kn^2), written as
    # loops at the returned parameters; float64 roundoff over ~370 terms: 1e-12.
    cfg, v, z, predict = make_case()
    I, B = observed(predict, TRUTH, z.size)
    I = I + make_rng().normal(0.0, NOISE_SD_GL, I.shape)
    out = CF.refine_node(I, B, v, predict, np.array(TRUTH) * 1.3, cfg)
    s = predict(*theta_of(out))
    lam = cfg.measure.sgt_lambda_um
    cost, ss, n = 0.0, 0.0, 0
    for k in range(z.size):
        for j in range(v.size):
            r = I[k, j] - B[k] * (1.0 - s[k, j])
            cost += np.exp(-abs(v[j]) / lam) * r * r
            ss += r * r
            n += 1
    check_close("reference cost", out["cost"], cost, rtol=1e-12)
    check_close("reference rms", out["rms_gl"], np.sqrt(ss / n), rtol=1e-12)
    print("    cost %.6f gl^2, rms %.4f gl (noise SD %.1f)" % (out["cost"], out["rms_gl"], NOISE_SD_GL))


def test_convergence():
    raise unittest.SkipTest("not a discretisation: a wrapper of scipy.optimize.least_squares")


def test_invariants():
    # Over N_DRAWS seeded noisy data sets the mean d-hat lies within 3
    # standard errors (SD of the d-hats / sqrt(N_DRAWS)) of the truth.
    cfg, v, z, predict = make_case()
    I0, B = observed(predict, TRUTH, z.size)
    rng = make_rng()
    th0 = np.array(TRUTH) * 1.3
    est = []
    for _ in range(N_DRAWS):
        I = I0 + rng.normal(0.0, NOISE_SD_GL, I0.shape)
        I_copy, B_copy, v_copy, th_copy = I.copy(), B.copy(), v.copy(), th0.copy()
        out = CF.refine_node(I, B, v, predict, th0, cfg)
        assert_array_equal(I, I_copy)
        assert_array_equal(B, B_copy)
        assert_array_equal(v, v_copy)
        assert_array_equal(th0, th_copy)
        assert out["status"] == "converged", out
        est.append(theta_of(out))
    est = np.array(est)
    mean = est.mean(axis=0)
    se = est.std(axis=0, ddof=1) / np.sqrt(N_DRAWS)
    bias = mean - np.array(TRUTH)
    for name, m, s, b in zip(CF.PARAM_NAMES, mean, se, bias):
        print("    %-9s mean %.5f  SE %.5f  bias %+.5f (%+.2f SE)" % (name, m, s, b, b / s))
    assert abs(bias[0]) <= 3.0 * se[0], "mean d-hat %.6f, truth %.6f, SE %.6f" % (mean[0], TRUTH[0], se[0])


def test_contract():
    cfg, v, z, predict = make_case()
    I, B = observed(predict, TRUTH, z.size)
    out = CF.refine_node(I, B, v, predict, np.array(TRUTH) * 1.3, cfg)
    assert set(out) == {"d_um", "mu_per_um", "v0_um", "dz_um", "cost", "rms_gl", "status", "at_bound", "nfev"}, out
    for k in ("d_um", "mu_per_um", "v0_um", "dz_um", "cost", "rms_gl"):
        assert isinstance(out[k], float) and np.isfinite(out[k]), (k, out[k])
    assert out["status"] in CF.STATUSES and out["status"] in ("converged", "at_bound", "failed")
    assert isinstance(out["at_bound"], tuple) and isinstance(out["nfev"], int) and out["nfev"] >= 1
    # offset_xyz: (v0 y_x, v0 y_y, dz), shape (3,), float64; exact products
    y = np.array([-np.sin(0.7), np.cos(0.7), 0.0])
    off = CF.offset_xyz(0.25, -0.1, y)
    assert off.shape == (3,) and off.dtype == np.float64
    assert_array_equal(off, np.array([0.25 * y[0], 0.25 * y[1], -0.1]))
    assert_array_equal(CF.offset_xyz(0.25, -0.1, y[:2]), off)
    for bad in ((np.nan, 0.0, y), (0.1, 0.0, np.ones(4))):
        try:
            CF.offset_xyz(*bad)
        except ValueError:
            pass
        else:
            raise AssertionError("offset_xyz accepted %r" % (bad,))


def test_determinism():
    cfg, v, z, predict = make_case()
    I, B = observed(predict, TRUTH, z.size)
    I = I + make_rng().normal(0.0, NOISE_SD_GL, I.shape)
    a = CF.refine_node(I, B, v, predict, np.array(TRUTH) * 0.7, cfg)
    b = CF.refine_node(I, B, v, predict, np.array(TRUTH) * 0.7, cfg)
    assert a == b, (a, b)


def _raises(func, *args):
    try:
        func(*args)
    except ValueError as exc:
        return str(exc)
    raise AssertionError("no ValueError for %r" % (args[:2] if len(args) > 1 else args,))


def test_edge_cases():
    cfg, v, z, predict = make_case()
    d_hi = cfg.cleaning.refine_d_bounds_um[1]
    # truth beyond the upper diameter bound: at_bound, named
    truth = (d_hi + 1.0, 1.0, 0.0, 0.1)
    I, B = observed(predict, truth, z.size)
    out = CF.refine_node(I, B, v, predict, (d_hi - 1.0, 1.0, 0.0, 0.1), cfg)
    assert out["status"] == "at_bound" and "d_um" in out["at_bound"], out
    print("    beyond the d bound: %s %r, d %.6f" % (out["status"], out["at_bound"], out["d_um"]))
    # truth beyond the dz bound
    dz_hi = cfg.cleaning.refine_dz_bounds_um[1]
    truth = (0.8, 1.5, 0.0, dz_hi + 0.5)
    I, B = observed(predict, truth, z.size)
    out = CF.refine_node(I, B, v, predict, (0.8, 1.5, 0.0, 0.0), cfg)
    assert out["status"] == "at_bound" and "dz_um" in out["at_bound"], out
    # a predict returning NaN: failed, no exception
    I, B = observed(predict, TRUTH, z.size)

    def nan_predict(d, mu, v0, dz):
        return np.full((z.size, v.size), np.nan)

    out = CF.refine_node(I, B, v, nan_predict, TRUTH, cfg)
    assert out["status"] == "failed", out

    def nan_away(d, mu, v0, dz):  # finite at the start, NaN once d moves
        return predict(d, mu, v0, dz) if d == 1.04 else np.full((z.size, v.size), np.nan)

    out = CF.refine_node(I, B, v, nan_away, (1.04, 1.5, 0.12, 0.1), cfg)
    assert out["status"] == "failed", out
    # ValueError: shapes that disagree, B <= 0, a start outside the bounds
    th0 = np.array(TRUTH)
    _raises(CF.refine_node, I[:, :-1], B, v, predict, th0, cfg)
    _raises(CF.refine_node, I, B[:-1], v, predict, th0, cfg)
    _raises(CF.refine_node, I[0], B, v, predict, th0, cfg)
    _raises(CF.refine_node, I, B, v, predict, th0[:3], cfg)
    Bz = B.copy()
    Bz[2] = 0.0
    _raises(CF.refine_node, I, Bz, v, predict, th0, cfg)
    _raises(CF.refine_node, I, -B, v, predict, th0, cfg)
    _raises(CF.refine_node, I, B, v, predict, (d_hi + 0.5, 1.5, 0.12, 0.1), cfg)
    _raises(CF.refine_node, I, B, v, predict, (0.8, 1.5, 0.12, dz_hi + 0.5), cfg)
    _raises(CF.refine_node, I, B, v, predict, (0.8, 1.5, 0.12, np.nan), cfg)


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
