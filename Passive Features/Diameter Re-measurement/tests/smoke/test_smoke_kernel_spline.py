"""Smoke test for the spline reading of the kernel table -- Block 4, amendment 4 in specs/SPEC.md
(D-181, D-265, D-272): model.kernel.sigma_r with kernel_family "gaussian_table_spline" and the "tangent"
continuation, and growth_spline.

Checks
    test_known_answer   D-265's numbers on the pilot cell's table (the decision log, 2026-11-27): the tangent of
                        G at 0.84 um is 0.469 um; sigma_r = 0.618, 0.669, 0.723 um at 0.98, 1.12, 1.28 um; and the
                        knot values exactly
    test_reference      Block 10's own reading of a growth curve (calibration._basis, "symmetric", "cubic":
                        one not-a-knot spline per knot value, continued along the tangent), an independent
                        construction, inside and beyond the knots
    test_convergence    not a discretisation (SkipTest)
    test_invariants     even in delta; continuous at the last knot for every continuation, C1 in G for
                        "tangent"; sigma_r >= sigma_r(0) everywhere; the linear continuation's values
    test_contract       shapes, dtype, finiteness; scalar in, 0-d array out as before
    test_determinism    no randomness: two calls agree bit for bit; the first reading is unchanged bit for bit
    test_edge_cases     "tangent" without the spline family, fewer than three knots, a non-finite delta, the
                        "empirical" family; with_in_focus_width and with_kernel keep the family and the growth

Run (from Passive Features/Diameter Re-measurement/)
    python tests/smoke/test_smoke_kernel_spline.py
    python -m pytest tests/smoke/test_smoke_kernel_spline.py -q

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

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from allen_diameter.config import default_config  # noqa: E402
from allen_diameter.model import kernel as K  # noqa: E402
from allen_diameter.analysis import calibration as CAL  # noqa: E402

SEED = 12345  # no randomness in this block
REPORT_PACKAGES = ("numpy", "scipy")
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
_SAVED_ARTIFACTS = []  # files written by check_close during this run


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


# ---------------------------------------------------------------- checks ---

PILOT = (0.080, 0.1536, 0.3924, 0.5620)      # um at 0, 0.28, 0.56, 0.84 um: the pilot cell's table (D-265)
KNOTS = (0.0, 0.28, 0.56, 0.84)


def rcfg_for(table=PILOT, knots=KNOTS, family="gaussian_table_spline", continuation="tangent", slope=0.79):
    r = default_config().renderer
    return dataclasses.replace(r, kernel_family=family, kernel_table_delta_um=tuple(knots),
                               kernel_table_sigma_um=tuple(table), sigma_r0_um=table[0],
                               kernel_continuation=continuation, kernel_continuation_slope=slope)


def test_known_answer():
    r = rcfg_for()
    # exact at the knots: the spline interpolates G, and sqrt(s0^2 + (s^2 - s0^2)) returns s to a few ulp
    check_close("knots", K.sigma_r(np.array(KNOTS), r), np.array(PILOT), rtol=1e-13)
    sp = K.growth_spline(r)
    # the decision log gives three digits: 0.469 um, and 0.618, 0.669, 0.723 um (D-265), so atol = half a unit
    # of the last digit
    check_close("tangent_dG", float(sp(0.84, 1)), 0.469, rtol=0.0, atol=5e-4)
    check_close("beyond", K.sigma_r(np.array([0.98, 1.12, 1.28]), r), np.array([0.618, 0.669, 0.723]), rtol=0.0,
                atol=5e-4)


def test_reference():
    # Block 10 reads its growth curve as B @ G with one not-a-knot spline per knot value on the signed knots and a
    # linear continuation along the tangent (calibration._basis); for equally spaced knots this is the same curve
    # built another way. Tolerance: two spline solves of a 7-point system in float64, 1e-12 um^2 in G.
    r = rcfg_for()
    g = np.array(PILOT) ** 2 - PILOT[0] ** 2
    delta = np.linspace(-1.4, 1.4, 281)
    B, _ = CAL._basis(delta, "symmetric", 0.28, 3, "cubic")
    g_ref = B @ g
    s = K.sigma_r(delta, r)
    check_close("G_vs_block10", s ** 2 - PILOT[0] ** 2, np.maximum(g_ref, 0.0), rtol=0.0, atol=1e-12)
    # where Block 10's curve is negative the renderer's growth is floored at 0 (D-275): on the pilot table this
    # happens only next to focus, inside the first knot interval, and by less than 1e-3 um^2
    neg = g_ref < 0.0
    assert np.all(np.abs(delta[neg]) < 0.28) and float(np.min(g_ref)) > -1e-3
    print("    pilot table: Block 10's spline of G is negative for |delta| < %.3f um, min %.3e um^2 at %.3f um; "
          "floored: sigma_r = sigma_r(0) there (unfloored it would be %.4f um)"
          % (float(np.max(np.abs(delta[neg]))), float(np.min(g_ref)), float(abs(delta[np.argmin(g_ref)])),
             float(np.sqrt(PILOT[0] ** 2 + np.min(g_ref)))))
    # the default table (six knots, unequal steps): against scipy's spline evaluated directly
    d = default_config().renderer
    rd = dataclasses.replace(d, kernel_family="gaussian_table_spline")
    x = np.linspace(0.0, d.kernel_table_delta_um[-1], 400)
    gd = K.growth_spline(rd)(x)
    check_close("default_table", K.sigma_r(x, rd), np.sqrt(d.sigma_r0_um ** 2 + np.maximum(gd, 0.0)), rtol=1e-13)
    print("    default table: min of the spline's G between knots %.3e um^2 (the floor acts where this is < 0)"
          % float(np.min(gd)))


def test_convergence():
    raise unittest.SkipTest("the spline reading is not a discretisation")


def test_invariants():
    x = np.linspace(0.0, 2.0, 801)
    for cont in ("tangent", "linear", "frozen", "proportional"):
        r = rcfg_for(continuation=cont)
        s = K.sigma_r(x, r)
        assert_array_equal(K.sigma_r(-x, r), s)                                  # even
        assert np.all(s >= PILOT[0] - 1e-15), cont                              # never below the in-focus width
        eps = 1e-9                                                              # continuity at the last knot
        check_close("continuity_" + cont, K.sigma_r(0.84 + eps, r), K.sigma_r(0.84 - eps, r), rtol=0.0, atol=1e-7)
    r = rcfg_for(continuation="tangent")
    sp = K.growth_spline(r)
    h = 1e-5                                                                     # C1 in G across 0.84 um
    g = lambda d: float(K.sigma_r(d, r)) ** 2 - PILOT[0] ** 2                    # noqa: E731
    left, right = (g(0.84) - g(0.84 - h)) / h, (g(0.84 + h) - g(0.84)) / h
    check_close("C1_in_G", [left, right], [float(sp(0.84, 1))] * 2, rtol=0.0, atol=1e-4)   # one-sided difference, O(h G'')
    rl = rcfg_for(continuation="linear", slope=0.79)                             # D-265.3: gamma beyond the table
    check_close("linear_beyond", K.sigma_r(np.array([1.0, 1.5]), rl), 0.5620 + 0.79 * (np.array([1.0, 1.5]) - 0.84),
                rtol=1e-14)


def test_contract():
    r = rcfg_for()
    out = K.sigma_r(np.zeros((3, 4, 5)) + 0.3, r)
    assert out.shape == (3, 4, 5) and out.dtype == np.float64 and np.all(np.isfinite(out))
    assert np.ndim(K.sigma_r(0.3, r)) == 0 and float(K.sigma_r(0.0, r)) == PILOT[0]
    assert isinstance(K.TABLE_FAMILIES, tuple) and "gaussian_table_spline" in K.TABLE_FAMILIES


def test_determinism():
    r = rcfg_for()
    x = np.linspace(-2, 2, 101)
    assert_array_equal(K.sigma_r(x, r), K.sigma_r(x, r))
    # the first reading is unchanged bit for bit: np.interp inside, s_max + gamma (d - d_max) beyond
    d = default_config().renderer
    kn, va = np.array(d.kernel_table_delta_um), np.array(d.kernel_table_sigma_um)
    a = np.abs(x)
    ref = np.where(a > kn[-1], va[-1] + d.kernel_continuation_slope * (a - kn[-1]), np.interp(a, kn, va))
    assert d.kernel_family == "gaussian_table"
    assert_array_equal(K.sigma_r(x, d), ref)


def test_edge_cases():
    def raises(exc, fn):
        try:
            fn()
        except exc:
            return
        raise AssertionError("%s not raised" % exc.__name__)
    raises(ValueError, lambda: K.sigma_r(0.1, rcfg_for(family="gaussian_table", continuation="tangent")))
    raises(ValueError, lambda: K.sigma_r(0.1, rcfg_for(table=(0.08, 0.3), knots=(0.0, 0.84))))
    raises(ValueError, lambda: K.sigma_r(np.array([0.1, np.nan]), rcfg_for()))
    raises(NotImplementedError, lambda: K.sigma_r(0.1, dataclasses.replace(rcfg_for(), kernel_family="empirical")))
    raises(ValueError, lambda: K.sigma_r(0.1, dataclasses.replace(rcfg_for(), kernel_continuation="bogus")))
    # a table whose spline dips below 0 between knots: the growth is floored, sigma_r = sigma_r(0) there
    dip = rcfg_for(table=(0.080, 0.0801, 0.60, 0.61))
    x = np.linspace(0.0, 0.84, 400)
    assert float(np.min(K.growth_spline(dip)(x))) < 0.0
    assert float(np.min(K.sigma_r(x, dip))) == 0.080
    # re-anchoring keeps the family, the continuation and the growth at the knots
    r = rcfg_for()
    for new in (K.with_in_focus_width(r, 0.12), K.with_kernel(r, 0.12, 1.0)):
        assert new.kernel_family == "gaussian_table_spline" and new.kernel_continuation == "tangent"
        check_close("growth_kept", np.array(new.kernel_table_sigma_um) ** 2 - 0.12 ** 2,
                    np.array(PILOT) ** 2 - PILOT[0] ** 2, rtol=0.0, atol=1e-15)


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
