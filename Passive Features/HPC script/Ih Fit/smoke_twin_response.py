"""Smoke test for <ALGORITHM> -- Block <N> in specs/SPEC.md.

Adapt this file for one algorithm: replace every EDIT marker and every
NotImplementedError. A check that does not apply is not deleted; make it
raise unittest.SkipTest("<reason>") so the summary shows why it was skipped.
Until a check is adapted it is reported as TODO and the run fails, so an
unadapted copy can never pass.

Checks, from the strongest oracle to the weakest
    test_known_answer   analytic or tabulated solution
    test_reference      independent implementation (library function, or
                        brute force on a tiny input)
    test_convergence    discretisations only: observed order p in
                        e(h) ~ C * h**p, where e(h) is the error at step h
    test_invariants     conservation laws, symmetries, bounds, normalisation
    test_contract       output shapes, dtypes and finiteness
    test_determinism    identical output for an identical seed
    test_edge_cases     degenerate or boundary inputs named in the spec

Run
    python tests/smoke/test_smoke_<algorithm>.py
        Prints the environment and a PASS / FAIL / ERROR / TODO / SKIP table,
        then the full message of every failure. Exit code 1 if any check
        failed, errored or is still TODO.
    python -m pytest tests/smoke -q
        The same checks through pytest (TODO checks show up as failures).

Debug
    A failed check_close() saves both arrays to
        tests/smoke/artifacts/<check name>.npz
    Inspect it with
        d = np.load("tests/smoke/artifacts/<check name>.npz")
        diff = d["actual"] - d["expected"]
        np.unravel_index(np.argmax(np.abs(diff)), diff.shape)

Add a check
    Define a function test_<something>() with no arguments that raises
    AssertionError on failure; the runner below and pytest both pick it up.

Keep this file pure ASCII (no Greek letters, dashes or arrows outside ASCII)
so it survives transfer to the cluster.
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

# EDIT: import the code under test, for example
# from mypackage.model.integrators import trapezoid

SEED = 12345  # EDIT if SPEC.md fixes a seed
REPORT_PACKAGES = ("numpy", "scipy")  # EDIT: packages whose versions matter
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

def test_known_answer():
    # EDIT: an input whose exact output is known analytically or tabulated,
    # e.g. check_close("known_answer", algorithm(x), exact, rtol=...)
    raise NotImplementedError("EDIT: known-answer check")


def test_reference():
    # EDIT: compare with an independent implementation on a tiny input,
    # e.g. a SciPy routine, or a brute-force loop that is obviously correct.
    raise NotImplementedError("EDIT: reference-implementation check")


def test_convergence():
    # EDIT (discretisations): errors at decreasing step sizes h, e.g.
    #   hs = np.array([...]); errs = np.array([abs(run(h) - exact) for h in hs])
    #   p = observed_order(hs, errs)
    #   assert abs(p - EXPECTED_ORDER) < 0.1, f"observed order {p:.3f}"
    # Not a discretisation: raise unittest.SkipTest("<reason>").
    raise NotImplementedError("EDIT: convergence check")


def test_invariants():
    # EDIT: properties that must hold for any valid input: conservation,
    # symmetry, bounds, normalisation, linearity, exactness on a class of
    # inputs the method integrates or fits exactly.
    raise NotImplementedError("EDIT: invariant check")


def test_contract():
    # EDIT: output shape and axis order, dtype, finiteness, units where they
    # are checkable, e.g.
    #   out = algorithm(x)
    #   assert out.shape == (n_trials, n_bins), out.shape
    #   assert out.dtype == np.float64, out.dtype
    #   assert np.all(np.isfinite(out))
    raise NotImplementedError("EDIT: output-contract check")


def test_determinism():
    # EDIT: two runs with the same seed must give bit-identical output, e.g.
    #   a = algorithm(x, rng=make_rng()); b = algorithm(x, rng=make_rng())
    #   assert_array_equal(a, b)
    # No randomness: raise unittest.SkipTest("deterministic algorithm").
    raise NotImplementedError("EDIT: determinism check")


def test_edge_cases():
    # EDIT: degenerate and boundary inputs named in SPEC.md (empty input,
    # single sample, zero-length interval, constant signal, ...), each with
    # its expected behaviour: a value, or a specific exception.
    raise NotImplementedError("EDIT: edge-case check")


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
