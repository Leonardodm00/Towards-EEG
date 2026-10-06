"""Smoke test for the inversion, flags and fill -- Block 7 in specs/SPEC.md
(procedure Eq. 1, s.3.9; mathematics Eqs. 20-21; D5).

Checks
    test_known_answer   on a table fitted to an exact b = 1 + c0/d, d_tilde
                        recovers d to 1e-3; the shortcut d_hat / b(d_hat) has
                        the closed-form error c0^2 / (d (d + 2 c0)) and agrees
                        with mathematics Eq. 21 to first order
    test_reference      fill rules on hand-made stretches, values by hand
    test_convergence    skipped: no discretisation parameter
    test_invariants     m_hat(d_tilde) = d_hat at every inverted node
    test_contract       Inversion fields; correct_nodes keeps the selection
                        reasons of nodes outside S
    test_determinism    skipped: deterministic closed forms and root finding
    test_edge_cases     non-monotone table, out-of-domain d_hat, large
                        correction, high failure rate, tilt outside the table,
                        missing estimate, mismatched estimator signature

Run
    cd "Passive Features/Diameter Re-measurement"
    python tests/smoke/test_smoke_invert.py

Keep this file pure ASCII (hpc-python-compat).
"""
from __future__ import annotations

import dataclasses
import importlib.metadata
import math
import platform
import sys
import time
import traceback
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
SRC = WS / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from allen_diameter.analysis import fill as FI  # noqa: E402
from allen_diameter.analysis import invert as IV  # noqa: E402
from allen_diameter.analysis.node_pipeline import NodeResult  # noqa: E402
from allen_diameter.analysis.table import fit_table  # noqa: E402
from allen_diameter.config import default_config, with_sigma_fit  # noqa: E402

SEED = 20261006
REPORT_PACKAGES = ("numpy", "scipy")


def cfg_with(smoothing="1e-8", **correction):
    cfg = default_config()
    return dataclasses.replace(cfg, correction=dataclasses.replace(cfg.correction, spline_smoothing=smoothing,
                                                                   **correction))


def exact_table(b_fn, n=600, seed=SEED, fail_fn=None, cfg=None):
    """A table fitted to noise-free ratios b_fn(d, phi) over the default design ranges."""
    cfg = cfg or cfg_with()
    rng = np.random.default_rng(seed)
    lo, hi = cfg.phantom.d_range_um
    d = np.exp(rng.uniform(math.log(lo), math.log(hi), n))
    phi = np.radians(rng.uniform(0, 90, n))
    fail = np.zeros(n, bool) if fail_fn is None else fail_fn(d, phi)
    rows = [dict(d_um=a, phi_rad=b, ratio=float(b_fn(a, b)), in_S=not f) for a, b, f in zip(d, phi, fail)]
    return fit_table(rows, cfg), cfg


def node(d_hat, phi, status="converged", flags=()):
    nan = float("nan")
    return NodeResult(0, 3, 0.0, 0.0, 0.0, 0.0, "", nan, nan, 0, 0.0, 0.0, 0.0, 0.0, 0.0, phi, False, False, 200.0,
                      "block_masked", d_hat, 1.0, 0.0, 1.0, status, tuple(flags), None, np.empty(0))


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    c0 = 0.1
    T, cfg = exact_table(lambda d, p: 1.0 + c0 / d)
    for d in (0.4, 0.8, 1.5, 3.0):
        phi = math.radians(25.0)
        inv = IV.invert_node(d + c0, phi, T, cfg)        # d_hat = m(d) = d b(d) = d + c0
        assert abs(inv.d_root_um / d - 1) <= 1e-3, (d, inv)
        if not inv.flags:
            assert inv.d_tilde_um == inv.d_root_um
        rel = IV.shortcut(d + c0, phi, T) / d - 1
        exact = c0 ** 2 / (d * (d + 2 * c0))
        b, beta = 1 + c0 / d, -(c0 / d) / (1 + c0 / d)
        eq21 = -beta * (b - 1) / (1 + beta * (b - 1))
        assert abs(rel - exact) <= 1e-3, ("shortcut vs closed form", d, rel, exact)
        assert abs(rel - eq21) <= 2 * (b - 1) ** 2, ("Eq. 21 to first order", d, rel, eq21)


def test_reference():
    cfg = cfg_with()
    nan = float("nan")
    d, src = FI.fill_stretch([1.0, nan, 1.2, nan, nan, 0.8], 0.5, cfg)
    assert np.allclose(d, [1.0, 1.1, 1.1, 1.0, 1.0, 0.8]) and list(src) == ["self", "neighbours", "self",
                                                                             "neighbours", "neighbours", "self"], (d, src)
    d, src = FI.fill_stretch([nan, nan, nan], [0.4, 0.5, 0.6], cfg)
    assert np.allclose(d, [0.4, 0.5, 0.6]) and set(src) == {"allen"}
    d, src = FI.fill_stretch([1.0, 1.0, 3.0, 1.0, 1.0], 0.5, cfg)
    assert np.allclose(d, 1.0), "the running median removes a one-node spike"
    allen = dataclasses.replace(cfg, correction=dataclasses.replace(cfg.correction, fill_policy="allen_only"))
    d, src = FI.fill_stretch([1.0, nan, 1.2], 0.5, allen)
    assert np.allclose(d, [1.0, 1.0, 1.2]) and src[1] == "allen", d    # median of (1.0, 0.5, 1.2) at the middle
    none = dataclasses.replace(cfg, correction=dataclasses.replace(cfg.correction, fill_policy="none"))
    d, src = FI.fill_stretch([1.0, nan, 1.2, 1.4], 0.5, none)
    assert math.isnan(d[1]) and src[1] == "none" and np.allclose(d[[0, 2, 3]], [1.0, 1.2, 1.4])


def test_convergence():
    raise unittest.SkipTest("no discretisation parameter in Block 7")


def test_invariants():
    T, cfg = exact_table(lambda d, p: 1.05 + 0.08 / d + 0.1 * np.sin(p) ** 2)
    rng = np.random.default_rng(SEED)
    for _ in range(20):
        d, phi = math.exp(rng.uniform(math.log(0.3), math.log(3.5))), math.radians(rng.uniform(0, 85))
        d_hat = float(T.m_hat(d, phi))
        inv = IV.invert_node(d_hat, phi, T, cfg)
        assert abs(float(T.m_hat(inv.d_root_um, phi)) - d_hat) <= 1e-9 * d_hat and abs(inv.d_root_um / d - 1) <= 1e-9


def test_contract():
    T, cfg = exact_table(lambda d, p: 1.0 + 0.05 / d)
    out = IV.correct_nodes([node(1.05, 0.2), node(1.05, 0.2, flags=("faint",)), node(float("nan"), 0.2, status="none")],
                           T, cfg)
    assert isinstance(out[0], IV.Inversion) and abs(out[0].d_tilde_um - 1.0) <= 1e-3 and out[0].flags == ()
    assert math.isnan(out[1].d_tilde_um) and out[1].flags == ("faint",)
    assert math.isnan(out[2].d_tilde_um) and out[2].flags == ("status:none",)


def test_determinism():
    raise unittest.SkipTest("deterministic closed forms and root finding")


def test_edge_cases():
    # non-monotone: b peaks at d = 1, so m = d b falls after it
    T, cfg = exact_table(lambda d, p: 1.0 + 0.8 * np.exp(-((np.log(d)) / 0.2) ** 2))
    assert "non_monotone" in IV.invert_node(1.6, 0.3, T, cfg).flags
    T, cfg = exact_table(lambda d, p: 1.0 + 0.05 / d)
    lo, hi = T.d_range
    assert "out_of_domain" in IV.invert_node(hi * 1.3, 0.3, T, cfg).flags
    assert "out_of_domain" in IV.invert_node(lo * 0.5, 0.3, T, cfg).flags
    assert IV.invert_node(float("nan"), 0.3, T, cfg).flags == ("no_estimate",)
    assert "phi_out_of_domain" in IV.invert_node(1.0, math.radians(95.0), T, cfg).flags
    T, cfg = exact_table(lambda d, p: 1.3 + 0.0 * d)
    inv = IV.invert_node(1.3, 0.3, T, cfg)
    assert "large_correction" in inv.flags and math.isnan(inv.d_tilde_um) and abs(inv.d_root_um - 1.0) <= 1e-3, inv
    T, cfg = exact_table(lambda d, p: 1.0 + 0.0 * d, fail_fn=lambda d, p: p > np.radians(60.0))
    assert "high_failure" in IV.invert_node(1.0, math.radians(80.0), T, cfg).flags
    assert "high_failure" not in IV.invert_node(1.0, math.radians(20.0), T, cfg).flags
    try:
        IV.correct_nodes([node(1.0, 0.2)], T, with_sigma_fit(cfg, 0.125))
    except ValueError:
        pass
    else:
        raise AssertionError("a table must refuse a fit with another estimator")


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
