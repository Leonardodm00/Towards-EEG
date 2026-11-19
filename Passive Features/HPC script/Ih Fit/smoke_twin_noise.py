"""Smoke test for the twin's noise -- Block 10 in specs/SPEC.md (TN1-TN9).

SPEC 2.7 (c), (d), (e) and the entry "### Block 10: the twin's noise"
(D-180, D-182, D-183.3, D-186). No NEURON, no archive; truths are chosen,
so every check has a known answer. COHORT is the five-parameter fixture law
of the Block 3 note of 2026-11-17 (sigma_f 0.035 mV, sigma_m 0.017 mV,
tau_m 0.3 ms, sigma_s 0.055 mV, tau_s 40 ms).

Checks (each named after its oracle in the Block 10 entry)
    test_TN1_reference     every row of ss_sweep_noise equals (TW1) written
                           out here (SeedSequence of 2.7 (e), draw_units,
                           simulate_traces without a tilt, the explicit line
                           b (u(k, j) - u_bar)); ls_sweep_noise equals (TW2)
                           written out the same way; 1e-12 mV
    test_TN2_moments       4000 windows of 2000 samples at dt = 20 us,
                           distinct twin ids, COHORT, no tilt: mean demeaned
                           S_0 = expected_s0, across-window autocovariance at
                           lags 0, 1, 5, 50 = autocovariance, within 3.5 SE
    test_TN3_independence  slow-dominated law (sigma_s 1 mV, tau_s 1 s):
                           correlation of the window means of two adjacent
                           positions of one sweep over 2000 twin ids is 0
                           within 3.5 / sqrt(2000)
    test_TN4_tilt          noise(b) - noise(0) on the same seeds equals
                           b (u - u_bar) (SS) and b (j - (n - 1) / 2) dt (LS)
                           to 1e-12 mV; the SS tilt sums to 0 to 1e-9 mV
    test_TN5_seeds         reproducible; each seed argument changes the draw;
                           a window's draw does not depend on the others passed
    test_TN6_contract      shapes, float64, finiteness, every refusal
    test_TN7_zeros         zero SDs and zero tilt give exact zeros; COHORT:
                           mean of 4000 window means is 0 within 3.5 SE
    test_TN8_ls_continuous LS (sigma_s 1 mV, tau_s 1 s), 3 s at dt = 1 ms,
                           2000 twin ids: variance of the first and last
                           sample = gamma(0), covariance of samples 1 s apart
                           = sigma_s^2 exp(-1), within 3.5 SE
    test_TN9_ascii_lf      both files pure ASCII, LF line endings
    test_convergence       SKIP: not a discretisation

Run (from the code folder "Passive Features/HPC script/Ih Fit")
    python smoke_twin_noise.py
        Prints the environment and a PASS / FAIL / ERROR / TODO / SKIP table,
        then the full message of every failure. Exit code 1 if any check
        failed, errored or is still TODO.
    python -m pytest smoke_twin_noise.py -q
        The same checks through pytest.

Debug
    A failed check_close() saves both arrays to
        smoke_artifacts/<check name>.npz   (git-ignored)

Keep this file pure ASCII so it survives transfer to the cluster.
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
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import noise_law  # noqa: E402
from noise_law import NoiseLaw  # noqa: E402
from twin_noise import ls_sweep_noise, ss_sweep_noise, twin_rng  # noqa: E402

SEED = 20261119  # TwinConfig.base_seed's default, used as base_seed below
REPORT_PACKAGES = ("numpy", "scipy")
ARTIFACT_DIR = HERE / "smoke_artifacts"
_SAVED_ARTIFACTS = []  # files written by check_close during this run

# Fixtures (chosen truths)
COHORT = NoiseLaw(sigma_f=0.035, sigma_m=0.017, tau_m=3e-4, sigma_s=0.055, tau_s=0.040)
SLOW = NoiseLaw(sigma_f=0.035, sigma_m=0.017, tau_m=3e-4, sigma_s=1.0, tau_s=1.0)
SLOW_ONLY = NoiseLaw(sigma_f=0.0, sigma_m=0.0, tau_m=3e-4, sigma_s=1.0, tau_s=1.0)
DT = 2e-5           # s, 50 kHz (the archive's rate)
N_W = 10500         # samples of a stored SS window (L3_exc archive)
PERIOD = 10025      # samples, onset to onset (Block 3 fixture)
Z = 3.5             # Monte Carlo acceptance: 3.5 SE, one-sided p ~ 2e-4 per comparison


# --------------------------------------------------------------- helpers ---

def check_close(name, actual, expected, rtol=1e-12, atol=0.0):
    """numpy.testing.assert_allclose that keeps the evidence on failure.

    On failure both arrays are saved to ARTIFACT_DIR/<name>.npz before the
    AssertionError propagates. Justify rtol and atol at the call site.
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


def within_se(name, values, truth, z=Z):
    """Assert mean(values) is within z standard errors of `truth`."""
    values = np.asarray(values, dtype=float)
    mean = values.mean()
    se = values.std(ddof=1) / np.sqrt(values.size)
    assert abs(mean - truth) <= z * se, (
        "%s: mean %.6g vs truth %.6g, |diff| = %.3g SE (SE %.3g)"
        % (name, mean, truth, abs(mean - truth) / se, se))
    return mean, se


def reference_ss(law, b, dt, n_w, position, P, base, g, s):
    """(TW1) without the response, written out from SPEC 2.7 (c), (e)."""
    position = list(position)
    rows = []
    for p in position:
        rng = np.random.default_rng(np.random.SeedSequence([base, g, 0, s, p]))
        u = noise_law.draw_units(rng, 1, n_w)
        rows.append(noise_law.simulate_traces(law, dt, u.fast, u.mid, u.slow)[0])
    e = np.array(rows)
    j = np.arange(n_w, dtype=float)
    u_kj = np.array([(p * P + j) * dt for p in position])        # s, (K, n_w)
    u_bar = u_kj.mean()
    # the closed form of u_bar stated in 2.7 (c)
    u_bar_cf = np.mean(position) * P * dt + (n_w - 1) * dt / 2.0
    assert abs(u_bar - u_bar_cf) <= 1e-12 * max(1.0, abs(u_bar_cf)), (u_bar, u_bar_cf)
    return e + b * (u_kj - u_bar)


def reference_ls(law, b, dt, n, base, g, s):
    """(TW2) without the response, written out from SPEC 2.7 (d), (e)."""
    rng = np.random.default_rng(np.random.SeedSequence([base, g, 1, s]))
    u = noise_law.draw_units(rng, 1, n)
    e = noise_law.simulate_traces(law, dt, u.fast, u.mid, u.slow)[0]
    return e + b * (np.arange(n, dtype=float) - (n - 1) / 2.0) * dt


# ---------------------------------------------------------------- checks ---

def test_TN1_reference():
    # twin_rng is the SeedSequence of 2.7 (e)
    for args, ent in (((SEED, 4, "SS", 93, 11), [SEED, 4, 0, 93, 11]),
                      ((SEED, 4, "LS", 93), [SEED, 4, 1, 93])):
        a = twin_rng(*args).standard_normal(64)
        r = np.random.default_rng(np.random.SeedSequence(ent)).standard_normal(64)
        assert_array_equal(a, r)
    # SS: unsorted positions with a gap; tolerance 1e-12 mV: float64 roundoff on
    # sums of terms of at most a few mV (tilt 0.8 mV/s x ~7 s).
    pos = [5, 0, 17, 2, 36]
    for b in (0.83, -0.41):
        got = ss_sweep_noise(COHORT, b, DT, N_W, pos, PERIOD, base_seed=SEED, twin_id=4, sweep_number=93)
        ref = reference_ss(COHORT, b, DT, N_W, pos, PERIOD, SEED, 4, 93)
        check_close("TN1_ss_b%g" % b, got, ref, rtol=0.0, atol=1e-12)
    # LS: the real geometry's length (363500 samples)
    n = 363500
    for b in (0.37, -0.12):
        got = ls_sweep_noise(COHORT, b, DT, n, base_seed=SEED, twin_id=4, sweep_number=41)
        ref = reference_ls(COHORT, b, DT, n, SEED, 4, 41)
        check_close("TN1_ls_b%g" % b, got, ref, rtol=0.0, atol=1e-12)


_TN2_CACHE = {}


def _cohort_windows():
    """4000 windows of 2000 samples at 20 us, one per twin id, COHORT, b = 0."""
    if "x" not in _TN2_CACHE:
        n, N = 2000, 4000
        x = np.empty((N, n))
        for g in range(N):
            x[g] = ss_sweep_noise(COHORT, 0.0, DT, n, [7], PERIOD, base_seed=SEED, twin_id=g, sweep_number=93)[0]
        _TN2_CACHE["x"] = x
    return _TN2_CACHE["x"]


def test_TN2_moments():
    x = _cohort_windows()
    N, n = x.shape
    xc = x - x.mean(axis=1, keepdims=True)
    s0 = np.sum(xc * xc, axis=1)
    m, se = within_se("TN2 S_0", s0, noise_law.expected_s0(COHORT, DT, n))
    report = ["S_0 %.5g (E %.5g, SE %.2g)" % (m, noise_law.expected_s0(COHORT, DT, n), se)]
    for L in (0, 1, 5, 50):
        prod = np.mean(x[:, :n - L] * x[:, L:], axis=1)      # per-window lag-product average
        g = float(noise_law.autocovariance(COHORT, DT, [L])[0])
        m, se = within_se("TN2 gamma(%d)" % L, prod, g)
        report.append("gamma(%d) %.4g (%.4g)" % (L, m, g))
    print("    TN2: " + "; ".join(report))


def test_TN3_independence():
    # Adjacent positions 0 and 1 of one sweep; tilt 0. A continuous train would
    # correlate their means at rho_cont (printed; about 0.8 by the entry).
    N = 2000
    m = np.empty((N, 2))
    for g in range(N):
        y = ss_sweep_noise(SLOW, 0.0, DT, N_W, [0, 1], PERIOD, base_seed=SEED, twin_id=g, sweep_number=93)
        m[g] = y.mean(axis=1)
    r = np.corrcoef(m[:, 0], m[:, 1])[0, 1]
    rho_cont = noise_law.window_mean_cov(SLOW, DT, N_W, N_W, PERIOD) / noise_law.var_window_mean(SLOW, DT, N_W)
    print("    TN3: corr = %.4f (bound %.4f); a continuous train: %.3f" % (r, Z / np.sqrt(N), rho_cont))
    assert rho_cont > 0.5, "fixture has no power: continuous-train correlation %.3f" % rho_cont
    assert abs(r) <= Z / np.sqrt(N), "TN3: window means correlate, r = %.4f" % r


def test_TN4_tilt():
    pos = np.array([12, 3, 0, 7, 30])
    j = np.arange(N_W, dtype=float)
    u = (pos[:, None] * PERIOD + j[None, :]) * DT
    kw = dict(base_seed=SEED, twin_id=9, sweep_number=93)
    y0 = ss_sweep_noise(COHORT, 0.0, DT, N_W, pos, PERIOD, **kw)
    for b in (1.7, -0.6, 0.0):
        yb = ss_sweep_noise(COHORT, b, DT, N_W, pos, PERIOD, **kw)
        line = yb - y0
        # 1e-12 mV: roundoff of adding a <= 3 mV line to sub-mV noise
        check_close("TN4_ss_line_b%g" % b, line, b * (u - u.mean()), rtol=0.0, atol=1e-12)
        # sum of 52500 terms of <= 3 mV: roundoff ~ 1e-11, bound 1e-9 mV
        assert abs(line.sum()) <= 1e-9, "TN4: SS tilt sums to %.3g mV" % line.sum()
    n = 50001
    l0 = ls_sweep_noise(COHORT, 0.0, DT, n, base_seed=SEED, twin_id=9, sweep_number=41)
    for b in (1.7, -0.6, 0.0):
        lb = ls_sweep_noise(COHORT, b, DT, n, base_seed=SEED, twin_id=9, sweep_number=41)
        check_close("TN4_ls_line_b%g" % b, lb - l0, b * (np.arange(n) - (n - 1) / 2.0) * DT, rtol=0.0, atol=1e-12)
        assert abs((lb - l0).sum()) <= 1e-9, "TN4: LS tilt sums to %.3g mV" % (lb - l0).sum()


def test_TN5_seeds():
    pos = [0, 4, 9]
    base = dict(base_seed=SEED, twin_id=3, sweep_number=93)
    a = ss_sweep_noise(COHORT, 0.5, DT, 2000, pos, PERIOD, **base)
    assert_array_equal(a, ss_sweep_noise(COHORT, 0.5, DT, 2000, pos, PERIOD, **base))
    la = ls_sweep_noise(COHORT, 0.5, DT, 3000, base_seed=SEED, twin_id=3, sweep_number=41)
    assert_array_equal(la, ls_sweep_noise(COHORT, 0.5, DT, 3000, base_seed=SEED, twin_id=3, sweep_number=41))
    # each seed argument changes the SS draw (all rows differ)
    for key, val in (("base_seed", SEED + 1), ("twin_id", 4), ("sweep_number", 94)):
        kw = dict(base)
        kw[key] = val
        c = ss_sweep_noise(COHORT, 0.5, DT, 2000, pos, PERIOD, **kw)
        assert np.all(np.any(c != a, axis=1)), "TN5: changing %s left a row unchanged" % key
    for key, val in (("base_seed", SEED + 1), ("twin_id", 4), ("sweep_number", 42)):
        kw = dict(base_seed=SEED, twin_id=3, sweep_number=41)
        kw[key] = val
        assert np.any(ls_sweep_noise(COHORT, 0.5, DT, 3000, **kw) != la), "TN5: LS %s" % key
    # position changes a window's draw: tilt 0 so only the random part is compared
    w0 = ss_sweep_noise(COHORT, 0.0, DT, 2000, [4], PERIOD, **base)[0]
    w1 = ss_sweep_noise(COHORT, 0.0, DT, 2000, [5], PERIOD, **base)[0]
    assert np.any(w0 != w1), "TN5: position does not change the draw"
    # protocol changes the generator (SS position p vs LS of the same sweep)
    s_draw = twin_rng(SEED, 3, "SS", 93, 0).standard_normal(100)
    l_draw = twin_rng(SEED, 3, "LS", 93).standard_normal(100)
    assert np.any(s_draw != l_draw), "TN5: protocol does not change the draw"
    # a window's random part does not depend on the other positions passed
    many = ss_sweep_noise(COHORT, 0.0, DT, 2000, [9, 4, 0, 22], PERIOD, **base)
    assert_array_equal(many[1], w0)
    alone9 = ss_sweep_noise(COHORT, 0.0, DT, 2000, [9], PERIOD, **base)[0]
    assert_array_equal(many[0], alone9)


def _raises(fn, *args, **kw):
    try:
        fn(*args, **kw)
    except ValueError:
        return
    raise AssertionError("no ValueError for %s(%r, %r)" % (fn.__name__, args, kw))


def test_TN6_contract():
    kw = dict(base_seed=SEED, twin_id=1, sweep_number=93)
    y = ss_sweep_noise(COHORT, 0.2, DT, 500, np.array([2, 0, 1], dtype=np.int32), PERIOD, **kw)
    assert y.shape == (3, 500) and y.dtype == np.float64 and np.all(np.isfinite(y)), (y.shape, y.dtype)
    z = ls_sweep_noise(COHORT, 0.2, DT, 700, **kw)
    assert z.shape == (700,) and z.dtype == np.float64 and np.all(np.isfinite(z)), (z.shape, z.dtype)
    # refusals of ss_sweep_noise
    _raises(ss_sweep_noise, COHORT, 0.2, DT, 500, [], PERIOD, **kw)                 # empty
    _raises(ss_sweep_noise, COHORT, 0.2, DT, 500, [1, 2, 1], PERIOD, **kw)          # not unique
    _raises(ss_sweep_noise, COHORT, 0.2, DT, 500, [0, -1], PERIOD, **kw)            # negative
    _raises(ss_sweep_noise, COHORT, 0.2, DT, 1, [0], PERIOD, **kw)                  # n_w < 2
    _raises(ss_sweep_noise, COHORT, 0.2, 0.0, 500, [0], PERIOD, **kw)               # dt = 0
    _raises(ss_sweep_noise, COHORT, 0.2, -DT, 500, [0], PERIOD, **kw)               # dt < 0
    _raises(ss_sweep_noise, COHORT, np.nan, DT, 500, [0], PERIOD, **kw)             # tilt NaN
    _raises(ss_sweep_noise, COHORT, np.inf, DT, 500, [0], PERIOD, **kw)             # tilt inf
    _raises(ss_sweep_noise, COHORT, 0.2, DT, 500, [0], 0, **kw)                     # period < 1
    for key in ("base_seed", "twin_id", "sweep_number"):
        bad = dict(kw)
        bad[key] = -1
        _raises(ss_sweep_noise, COHORT, 0.2, DT, 500, [0], PERIOD, **bad)
        _raises(ls_sweep_noise, COHORT, 0.2, DT, 500, **bad)
    # refusals of ls_sweep_noise
    _raises(ls_sweep_noise, COHORT, 0.2, DT, 1, **kw)                               # n < 2
    _raises(ls_sweep_noise, COHORT, 0.2, 0.0, 500, **kw)
    _raises(ls_sweep_noise, COHORT, 0.2, -DT, 500, **kw)
    _raises(ls_sweep_noise, COHORT, np.nan, DT, 500, **kw)
    _raises(ls_sweep_noise, COHORT, -np.inf, DT, 500, **kw)
    # refusals of twin_rng
    _raises(twin_rng, SEED, 1, "SS", 93)                    # SS without a position
    _raises(twin_rng, SEED, 1, "LS", 93, 0)                 # LS with one
    _raises(twin_rng, SEED, 1, "XX", 93, 0)                 # unknown protocol
    _raises(twin_rng, SEED, 1, "ss", 93, 0)                 # lower case is not the contract's code
    _raises(twin_rng, -1, 1, "LS", 93)
    _raises(twin_rng, SEED, -1, "LS", 93)
    _raises(twin_rng, SEED, 1, "LS", -93)
    _raises(twin_rng, SEED, 1, "SS", 93, -2)
    assert isinstance(twin_rng(SEED, 1, "SS", 93, 0), np.random.Generator)


def test_TN7_zeros():
    zero = NoiseLaw(sigma_f=0.0, sigma_m=0.0, tau_m=3e-4, sigma_s=0.0, tau_s=0.04)
    y = ss_sweep_noise(zero, 0.0, DT, 2000, [0, 3, 8], PERIOD, base_seed=SEED, twin_id=2, sweep_number=93)
    assert np.all(y == 0.0), "TN7: SS not exactly zero (max |y| %.3g)" % np.abs(y).max()
    z = ls_sweep_noise(zero, 0.0, DT, 5000, base_seed=SEED, twin_id=2, sweep_number=41)
    assert np.all(z == 0.0), "TN7: LS not exactly zero (max |z| %.3g)" % np.abs(z).max()
    means = _cohort_windows().mean(axis=1)
    m, se = within_se("TN7 grand mean", means, 0.0)
    print("    TN7: mean of window means %.3g mV (SE %.2g)" % (m, se))


def test_TN8_ls_continuous():
    N, dt = 2000, 1e-3
    n = 3001                                           # j = 0..3000: 3 s
    x = np.empty((N, n))
    for g in range(N):
        x[g] = ls_sweep_noise(SLOW_ONLY, 0.0, dt, n, base_seed=SEED, twin_id=g, sweep_number=41)
    g0 = float(noise_law.autocovariance(SLOW_ONLY, dt, [0])[0])
    g1 = float(noise_law.autocovariance(SLOW_ONLY, dt, [1000])[0])
    assert abs(g1 - np.exp(-1.0)) < 1e-12 and abs(g0 - 1.0) < 1e-12
    rep = []
    for name, a, b, truth in (("var first", 0, 0, g0), ("var last", n - 1, n - 1, g0),
                              ("cov 0-1000", 0, 1000, g1), ("cov 1000-2000", 1000, 2000, g1),
                              ("cov 2000-3000", 2000, 3000, g1)):
        m, se = within_se("TN8 " + name, x[:, a] * x[:, b], truth)
        rep.append("%s %.3f (%.3f)" % (name, m, truth))
    print("    TN8: " + "; ".join(rep))


def test_TN9_ascii_lf():
    for name in ("twin_noise.py", "smoke_twin_noise.py"):
        raw = (HERE / name).read_bytes()
        bad = [i for i, ch in enumerate(raw) if ch > 127]
        assert not bad, "%s: non-ASCII byte at offset %d" % (name, bad[0])
        assert b"\r" not in raw, "%s: CR byte present" % name


def test_convergence():
    raise unittest.SkipTest("not a discretisation: exact draws of the sampled law")


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
