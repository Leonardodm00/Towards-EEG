"""Smoke test for the residual weights about the centre and the near-bump cut -- Block 39 in specs/SPEC.md
(D-169.4, D-169.5, D-171.1 to D-171.4; the details D-174 [orchestrator]).

Checks
    test_known_answer   allen_fwhm_um against the arithmetic by hand; the weights at the
                        centre and one lambda away; the cut of a two-dip profile at the
                        lightest sample between the dips, found by hand
    test_reference      mean_dip against its definition written out; the chord factor
                        sqrt(3)/2 against the half-depth width of a faint tube's chord
                        profile blurred by nothing, solved numerically
    test_convergence    skipped (not a discretisation)
    test_invariants     a single dip: no cut; a flat-bottomed dip with texture inside its
                        half depth: no cut; a second dip below m sigma: no cut, and a cut
                        once m is lowered; 200 seeded noisy clean dips in 9 planes: no cut;
                        the mirrored profile gives the mirrored cut
    test_contract       shapes, dtypes, m_sd = 0 keeps all, planes honoured, inputs unchanged
    test_determinism    two identical calls are identical
    test_edge_cases     the refusals; a centre outside the line; a plane without a readable dip

Run (from the code folder "Passive Features/Diameter Re-measurement")
    python3 tests/smoke/test_smoke_line_weights.py
    python3 -m pytest tests/smoke/test_smoke_line_weights.py -q

Pure ASCII, for the cluster.
"""
from __future__ import annotations

import importlib.metadata
import math
import platform
import sys
import time
import traceback
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
SRC = WS / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from allen_diameter.analysis import line_weights as LW  # noqa: E402

SEED = 20261118
REPORT_PACKAGES = ("numpy", "scipy")
_SAVED_ARTIFACTS = []
V = 0.1144 * np.arange(-26, 27)             # um, the pilot's line: 53 samples, step p_x
B_GL, SIGMA = 200.0, 2.5                    # gl
WINDOW = 0.5                                # um


def gauss(c, s, a):
    return a * np.exp(-0.5 * ((V - c) / s) ** 2)


def _raises(exc, func, *args, **kw):
    try:
        func(*args, **kw)
    except exc:
        return
    raise AssertionError("%s not raised by %s" % (exc.__name__, getattr(func, "__name__", func)))


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    # FWHM_A = sqrt((sqrt(3)/2 D)^2 + (2 sqrt(2 ln 2) s)^2): D 0.45, s 0.080 -> sqrt(0.151875 + 0.035489) = 0.43286;
    # D 1.2 -> sqrt(1.08 + 0.035489) = 1.05617 (by hand, 1e-4 um)
    assert_allclose(LW.allen_fwhm_um(0.45, 0.080), 0.43286, atol=1e-4)
    assert_allclose(LW.allen_fwhm_um(1.2, 0.080), 1.05617, atol=1e-4)
    assert_allclose(LW.allen_fwhm_um(1.0, 0.0), math.sqrt(3.0) / 2.0, rtol=1e-15)
    w = LW.centre_weights(V, V[26], 0.6)
    assert w[26] == 1.0 and np.all((w > 0) & (w <= 1.0))
    assert_allclose(LW.centre_weights(np.array([0.0, 0.6, 1.2]), 0.0, 0.6), [1.0, math.exp(-1.0), math.exp(-2.0)],
                    rtol=1e-15)
    # two dips: 40 gl at 0 (SD 0.25 um), 25 gl at +1.3 um (SD 0.3 um). By hand: the lightest sample between
    # them is the argmin of the summed dip on 0 < v < 1.3
    D = gauss(0.0, 0.25, 40.0) + gauss(1.3, 0.3, 25.0)
    between = np.flatnonzero((V > 0.0) & (V < 1.3))
    v_hand = V[between[np.argmin(D[between])]]
    lo, hi = LW.bump_cut(D, SIGMA, V, 0.0, WINDOW, 2.0, 0.5)
    print("    two dips: cut at %.4f um (by hand %.4f um), none on the other side (%s)" % (hi, v_hand, lo))
    assert hi == v_hand and lo == -math.inf
    keep, v_lo, v_hi = LW.bump_mask((B_GL - D)[None, :], B_GL, SIGMA, V, 0.0, WINDOW, 2.0, 0.5, 0.0)
    assert (v_lo, v_hi) == (lo, hi)
    assert_array_equal(keep, V <= v_hand)


def test_reference():
    rng = np.random.default_rng(SEED)
    P = B_GL - gauss(0.0, 0.3, 30.0)[None, :] + rng.normal(0.0, SIGMA, (5, V.size))
    Bk = np.array([198.0, 199.0, 200.0, 201.0, 202.0])
    use = np.array([True, False, True, True, False])
    hand = np.mean([Bk[k] - P[k] for k in (0, 2, 3)], axis=0)
    assert_allclose(LW.mean_dip(P, Bk, 0.0, use), hand, rtol=1e-13)
    # the smoothing against a Gaussian kernel written out: scipy's gaussian_filter1d truncates the kernel at
    # 4 SDs (its default) and normalises it (interior samples: the edge mode does not reach them)
    s = 1.0
    j = np.arange(-4, 5)
    ker = np.exp(-0.5 * (j / s) ** 2)
    ker /= ker.sum()
    sm = LW.mean_dip(P, Bk, s, use)
    for n in (20, 26, 33):
        assert_allclose(sm[n], float(np.sum(ker * hand[n - 4:n + 5])), rtol=1e-12)
    # the chord factor: half-depth width of 2 sqrt(r^2 - v^2) is where sqrt(r^2 - v^2) = r / 2
    r = 0.5
    v_half = brentq(lambda x: math.sqrt(r * r - x * x) - 0.5 * r, 0.0, r)
    assert_allclose(2.0 * v_half / (2.0 * r), LW.CHORD_FWHM_FACTOR, rtol=1e-12)
    assert_allclose(LW.GAUSS_FWHM_FACTOR, 2.354820045, rtol=1e-9)


def test_convergence():
    raise unittest.SkipTest("not a discretisation")


def test_invariants():
    one = gauss(0.0, 0.25, 40.0)
    assert LW.bump_cut(one, SIGMA, V, 0.0, WINDOW, 2.0, 0.5) == (-math.inf, math.inf)
    # a wide flat-bottomed dip (2 um, 60 gl) with texture of 3 noise SDs inside: nothing below half depth
    # between the texture's maxima, so no cut
    flat = 60.0 / (1.0 + np.exp((np.abs(V) - 1.0) / 0.08))
    tex = 3.0 * SIGMA * np.cos(2.0 * math.pi * V / 0.5) * (np.abs(V) < 0.9)
    assert LW.bump_cut(flat + tex, SIGMA, V, 0.0, WINDOW, 2.0, 0.5) == (-math.inf, math.inf)
    # ... and without the half-depth condition (the level at 0.999 of the depth) the same texture is cut
    lo, hi = LW.bump_cut(flat + tex, SIGMA, V, 0.0, WINDOW, 2.0, 0.999)
    assert math.isfinite(lo) or math.isfinite(hi)
    # a second dip of 1.5 noise SDs: below m = 2, cut at m = 1
    small = one + gauss(1.3, 0.3, 1.5 * SIGMA)
    assert LW.bump_cut(small, SIGMA, V, 0.0, WINDOW, 2.0, 0.5) == (-math.inf, math.inf)
    lo, hi = LW.bump_cut(small, SIGMA, V, 0.0, WINDOW, 1.0, 0.5)
    assert lo == -math.inf and 0.0 < hi < 1.3
    # noise alone: 200 seeded clean dips in 9 planes, the rule of bump_mask (mean, smoothed by 1 sample)
    rng = np.random.default_rng(SEED)
    depth = 40.0 * np.exp(-0.5 * ((np.arange(9) - 4) / 2.0) ** 2)
    n_cut = 0
    for _ in range(200):
        P = B_GL - depth[:, None] * gauss(0.0, 0.25, 1.0)[None, :] + rng.normal(0.0, SIGMA, (9, V.size))
        keep, _lo, _hi = LW.bump_mask(P, B_GL, SIGMA, V, 0.0, WINDOW, 2.0, 0.5, 1.0)
        n_cut += not keep.all()
    print("    noise alone, 9 planes, 200 draws: %d cut" % n_cut)
    assert n_cut == 0
    # mirror symmetry (V is symmetric about 0)
    D = one + gauss(1.3, 0.3, 25.0)
    lo, hi = LW.bump_cut(D, SIGMA, V, 0.0, WINDOW, 2.0, 0.5)
    lo_m, hi_m = LW.bump_cut(D[::-1].copy(), SIGMA, V, 0.0, WINDOW, 2.0, 0.5)
    assert_allclose([lo_m, hi_m], [-hi, -lo], rtol=0, atol=1e-12)


def test_contract():
    P = B_GL - (gauss(0.0, 0.25, 40.0) + gauss(-1.4, 0.3, 30.0))[None, :] * np.ones((4, 1))
    P0, V0 = P.copy(), V.copy()
    keep, v_lo, v_hi = LW.bump_mask(P, B_GL, SIGMA, V, 0.0, WINDOW, 2.0, 0.5, 1.0)
    assert keep.shape == V.shape and keep.dtype == bool and isinstance(v_lo, float) and isinstance(v_hi, float)
    assert math.isfinite(v_lo) and v_hi == math.inf and keep.sum() == np.sum(V >= v_lo)
    keep0, lo0, hi0 = LW.bump_mask(P, B_GL, SIGMA, V, 0.0, WINDOW, 0.0, 0.5, 1.0)
    assert keep0.all() and (lo0, hi0) == (-math.inf, math.inf)
    # planes: the bump only in plane 3; without that plane no cut
    Q = B_GL - gauss(0.0, 0.25, 40.0)[None, :] * np.ones((4, 1))
    Q[3] -= gauss(-1.4, 0.3, 120.0)
    assert not LW.bump_mask(Q, B_GL, SIGMA, V, 0.0, WINDOW, 2.0, 0.5, 1.0)[0].all()
    assert LW.bump_mask(Q, B_GL, SIGMA, V, 0.0, WINDOW, 2.0, 0.5, 1.0, planes=np.array([True, True, True, False]))[0].all()
    w = LW.centre_weights(V, 0.2, 0.5)
    assert w.shape == V.shape and w.dtype == np.float64
    assert_array_equal(P, P0)
    assert_array_equal(V, V0)


def test_determinism():
    rng = np.random.default_rng(SEED)
    P = B_GL - gauss(0.0, 0.25, 40.0)[None, :] - gauss(1.2, 0.3, 30.0)[None, :] + rng.normal(0.0, SIGMA, (6, V.size))
    a = LW.bump_mask(P, B_GL, SIGMA, V, 0.0, WINDOW, 2.0, 0.5, 1.0)
    b = LW.bump_mask(P, B_GL, SIGMA, V, 0.0, WINDOW, 2.0, 0.5, 1.0)
    assert_array_equal(a[0], b[0])
    assert a[1:] == b[1:]


def test_edge_cases():
    D = gauss(0.0, 0.25, 40.0)
    _raises(ValueError, LW.allen_fwhm_um, 0.0, 0.08)
    _raises(ValueError, LW.allen_fwhm_um, float("nan"), 0.08)
    _raises(ValueError, LW.allen_fwhm_um, 0.5, -0.1)
    _raises(ValueError, LW.centre_weights, V, 0.0, 0.0)
    _raises(ValueError, LW.centre_weights, V[::-1], 0.0, 0.5)
    _raises(ValueError, LW.centre_weights, V, float("inf"), 0.5)
    _raises(ValueError, LW.bump_cut, D[:-1], SIGMA, V, 0.0, WINDOW, 2.0, 0.5)
    _raises(ValueError, LW.bump_cut, D, 0.0, V, 0.0, WINDOW, 2.0, 0.5)
    _raises(ValueError, LW.bump_cut, D, SIGMA, V, 0.0, WINDOW, 0.0, 0.5)
    _raises(ValueError, LW.bump_cut, D, SIGMA, V, 0.0, WINDOW, 2.0, 1.0)
    _raises(ValueError, LW.bump_cut, D, SIGMA, V, 0.0, 0.0, 2.0, 0.5)
    P = (B_GL - D)[None, :]
    _raises(ValueError, LW.bump_mask, P, B_GL, SIGMA, V, 0.0, WINDOW, -1.0, 0.5, 1.0)
    _raises(ValueError, LW.bump_mask, P, B_GL, SIGMA, V, 0.0, WINDOW, 2.0, 0.5, -1.0)
    _raises(ValueError, LW.bump_mask, P, B_GL, SIGMA, V, 0.0, WINDOW, 2.0, 0.5, 1.0, planes=np.array([False]))
    _raises(ValueError, LW.bump_mask, P[:, :-1], B_GL, SIGMA, V, 0.0, WINDOW, 2.0, 0.5, 1.0)
    # a centre outside the line: the nearest sample is taken as the central extreme; no error
    assert LW.bump_cut(D, SIGMA, V, 10.0, WINDOW, 2.0, 0.5) == (-math.inf, math.inf)
    # no readable central dip (shallower than m sigma): no cut even with a deep dip aside
    weak = gauss(0.0, 0.25, 1.0 * SIGMA) + gauss(1.5, 0.3, 50.0)
    assert LW.bump_cut(weak, SIGMA, V, 0.0, WINDOW, 2.0, 0.5) == (-math.inf, math.inf)


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
