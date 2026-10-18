# -*- coding: utf-8 -*-
"""
smoke_noise_stats.py -- smoke suite for noise_stats.sample_acf (N1-N6)
======================================================================

Block 1 of specs/SPEC.md (D-028, D-029). No NEURON, no archive, seconds.

  N1  library oracle: sample_acf equals statsmodels acf(x, adjusted=False)
      and the hand-written textbook sum (D-028.1) at n in {12, 500, 4750,
      13250} and lags up to n - 10, to 1e-14 (float64 roundoff on sums of up
      to 1.3e4 terms of order 1)
  N2  relation to the estimator it replaces (D-028.2): textbook =
      (n - L)/n * adjusted, window by window, to 1e-12; the adjusted form is
      written out HERE as an oracle only, it exists nowhere in Ih Fit/
  N3  NaN rules (D-029 (iii)): NaN at a lag with n - L < 10 and at L < 1
      while other lags of the same call stay finite; NaN everywhere for a
      flat window and for a window with a non-finite sample; a scalar lag
      returns a length-1 array; an int lag inside a float array of lags is
      accepted
  N4  |rho_hat(L)| <= 1 at every finite lag on 200 random AR(1) windows
  N5  the default is textbook: on a window where the two forms differ by
      more than 1e-3 at lag n - 10, the result equals the textbook value and
      differs from the adjusted one -- a change of estimator cannot pass
  N6  byte safety: this file, noise_stats.py, noise_calibration.py,
      passive_fitting_hpc_fixed.py and smoke_ih_recovery.py are pure ASCII
      with LF line endings; and the three call sites import sample_acf
      (text check, no NEURON needed)

Run from this folder (needs numpy, statsmodels):

    python smoke_noise_stats.py

Expected last line: "SMOKE noise_stats: 6/6 passed". Exit 1 on any FAIL.
"""

import sys
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from noise_stats import MIN_PRODUCTS, sample_acf  # noqa: E402

RESULTS = []


def report(name, ok, note=""):
    RESULTS.append((name, bool(ok), note))
    print("%s  %s%s" % ("PASS" if ok else "FAIL", name, ("  -- " + note) if note else ""))


def textbook_hand(x, L):
    """D-028.1 written out: sum of n-L lag products over sum of n squares."""
    u = np.asarray(x, float) - np.mean(x)
    return float(np.dot(u[L:], u[:-L]) / np.dot(u, u))


def adjusted_old(x, L):
    """The pre-D-028 estimator, as an oracle only: mean of the n-L products
    over the mean square of all n samples (= (n/(n-L)) * textbook)."""
    u = np.asarray(x, float) - np.mean(x)
    return float(np.mean(u[L:] * u[:-L]) / np.mean(u * u))


def ar1(rng, n, rho, sigma=1.0):
    from scipy.signal import lfilter
    e = rng.normal(0.0, sigma * np.sqrt(1 - rho * rho), n)
    e[0] = rng.normal(0.0, sigma)
    return lfilter([1.0], [1.0, -rho], e)


def _n1():
    from statsmodels.tsa.stattools import acf
    rng = np.random.default_rng(1)
    worst_lib = worst_hand = 0.0
    for n in (12, 500, 4750, 13250):
        x = ar1(rng, n, 0.6)
        lags = np.unique(np.clip(np.r_[1, 2, 5, 50, 500, n - 10], 1, n - MIN_PRODUCTS))
        r = sample_acf(x, lags)
        ref = acf(x, adjusted=False, nlags=int(lags.max()), fft=True)[lags]
        worst_lib = max(worst_lib, float(np.max(np.abs(r - ref))))
        worst_hand = max(worst_hand, max(abs(r[i] - textbook_hand(x, int(L))) for i, L in enumerate(lags)))
    # tolerance: float64 roundoff on sums of up to 1.3e4 terms of order 1
    report("N1 library oracle: sample_acf == statsmodels acf(adjusted=False) == hand sum",
           worst_lib <= 1e-14 and worst_hand <= 1e-14,
           "max |diff| lib %.1e, hand %.1e" % (worst_lib, worst_hand))


def _n2():
    rng = np.random.default_rng(2)
    worst = 0.0
    for n in (12, 500, 13250):
        x = ar1(rng, n, 0.9)
        for L in (1, 3, n - MIN_PRODUCTS):
            tb = float(sample_acf(x, L)[0])
            worst = max(worst, abs(tb - (n - L) / n * adjusted_old(x, L)))
    report("N2 D-028.2: textbook == (n-L)/n * adjusted, window by window",
           worst <= 1e-12, "max |diff| %.1e" % worst)


def _n3():
    rng = np.random.default_rng(3)
    n = 60
    x = ar1(rng, n, 0.5)
    r = sample_acf(x, [0, 1, n - MIN_PRODUCTS, n - MIN_PRODUCTS + 1, n - 1, n, -3])
    rules = (np.isnan(r[0]) and np.isfinite(r[1]) and np.isfinite(r[2])
             and np.isnan(r[3]) and np.isnan(r[4]) and np.isnan(r[5]) and np.isnan(r[6]))
    flat = np.all(np.isnan(sample_acf(np.full(n, 3.0), [1, 2])))
    bad = x.copy(); bad[7] = np.nan
    nonfinite = np.all(np.isnan(sample_acf(bad, [1, 2])))
    scalar = sample_acf(x, 1)
    shape_ok = scalar.shape == (1,) and np.isfinite(scalar[0])
    floats = sample_acf(x, np.array([1.0, 2.0]))
    float_ok = floats.shape == (2,) and np.all(np.isfinite(floats))
    report("N3 NaN rules: n-L<10, L<1, flat, non-finite; scalar lag -> length-1",
           rules and flat and nonfinite and shape_ok and float_ok,
           "lags [0,1,%d,%d,%d,%d,-3] -> %s" % (n - 10, n - 9, n - 1, n, np.round(r, 3).tolist()))


def _n4():
    rng = np.random.default_rng(4)
    worst = 0.0
    for _ in range(200):
        n = int(rng.integers(20, 3000))
        x = ar1(rng, n, float(rng.uniform(-0.95, 0.99)))
        r = sample_acf(x, np.arange(1, n - MIN_PRODUCTS + 1))
        worst = max(worst, float(np.nanmax(np.abs(r))))
    report("N4 |rho_hat| <= 1 on 200 random AR(1) windows (every finite lag)",
           worst <= 1.0 + 1e-12, "max |rho_hat| %.6f" % worst)


def _n5():
    # a short, strongly correlated window: at lag n-10 the two forms differ
    # by the factor (n-L)/n = 10/60; the difference is far above 1e-3
    rng = np.random.default_rng(5)
    n = 60
    x = np.cumsum(rng.normal(size=n))           # random walk: large |rho| at long lags
    L = n - MIN_PRODUCTS
    tb, adj = textbook_hand(x, L), adjusted_old(x, L)
    got = float(sample_acf(x, L)[0])
    report("N5 the default is textbook (D-029 (ii)): equals the textbook value, not the adjusted one",
           abs(tb - adj) > 1e-3 and abs(got - tb) <= 1e-14 and abs(got - adj) > 1e-3,
           "textbook %.4f, adjusted %.4f, got %.4f" % (tb, adj, got))


def _n6():
    files = ["smoke_noise_stats.py", "noise_stats.py", "noise_calibration.py",
             "passive_fitting_hpc_fixed.py", "smoke_ih_recovery.py"]
    bad = []
    for f in files:
        b = (HERE / f).read_bytes()
        if any(c > 127 for c in b):
            bad.append(f + ":non-ascii")
        if b"\r\n" in b:
            bad.append(f + ":crlf")
    callers = {"noise_calibration.py": "from noise_stats import sample_acf",
               "passive_fitting_hpc_fixed.py": "from noise_stats import sample_acf",
               "smoke_ih_recovery.py": "from noise_stats import sample_acf"}
    for f, needle in callers.items():
        if needle not in (HERE / f).read_text(encoding="ascii", errors="strict"):
            bad.append(f + ":no-import")
    # the old hand-written form must be gone from the two production modules
    for f, needle in (("noise_calibration.py", "np.mean(u[lag:] * u[:-lag]) / var"),
                      ("passive_fitting_hpc_fixed.py", "np.mean(s_centered[1:] * s_centered[:-1]) / var"),
                      ("passive_fitting_hpc_fixed.py", "np.sum(c[:-1] * c[1:]) / (np.sum(c**2) + 1e-12)")):
        if needle in (HERE / f).read_text(encoding="ascii"):
            bad.append(f + ":old-estimator-present")
    report("N6 ASCII + LF on %d files; the three call sites import sample_acf; old forms gone" % len(files),
           not bad, ", ".join(bad) if bad else "ok")


def main():
    for fn in (_n1, _n2, _n3, _n4, _n5, _n6):
        try:
            fn()
        except BaseException as exc:          # SystemExit included (D-010 (iii) lineage)
            traceback.print_exc()
            report(fn.__name__ + " raised", False, repr(exc))
    n_ok = sum(1 for _, ok, _ in RESULTS if ok)
    print("SMOKE noise_stats: %d/%d passed" % (n_ok, len(RESULTS)))
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
