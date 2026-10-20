"""
smoke_noise_law.py -- smoke suite for noise_law (M1-M9)
========================================================

Block 2 of specs/SPEC.md (D-017, D-034, D-056). No NEURON, no archive; about
one minute on one core. Truths are chosen, so every check has a known answer.

  M1  brute force: Var(x_bar), E[S_0] and E[S_L] of eqs. (7a), (7b), (9) equal
      the same moments computed from the full n x n covariance matrix Gamma
      (E[S_L] = sum_j (P Gamma P)_{j,j+L}, P the centering matrix) at n = 240,
      lags {1, 2, 5, 50, 200}, for the cohort law, a slow-only and a fast-only
      law, to 1e-10 relative (float64 on sums of 6e4 terms)
  M2  Monte Carlo of the simulator against the closed forms: 4000 traces of
      n = 2000 at dt = 20 us, cohort law: the sample mean of S_0 and of S_L
      (L = 1, 50) within 3.5 standard errors of (7b) and (9); the across-trace
      autocovariance at lags 0, 1, 50 within 3.5 SE of gamma (L3); the grand
      mean within 3.5 SE of 0 (Monte Carlo spread: 3.5 SE, one-sided
      probability 2e-4 per comparison)
  M3  stationarity: the across-trace variance of the FIRST sample and of the
      LAST sample both equal gamma(0) within 3.5 SE (a non-stationary start
      fails the first)
  M4  window means of one trace: window_mean_cov equals brute force to 1e-10
      (overlapping and disjoint windows, negative offset); the Monte Carlo
      covariance of two 190 ms window means 200 ms apart (dt = 0.1 ms, 4000
      traces) within 3.5 SE of it; and the exact-discrete semivariogram agrees
      with the continuous-time eq. (25) for a slow-only law to within
      3 dt/tau_s relative (the discretisation error of a sampled window mean)
  M5  tilt: with sigma_b > 0 the Monte Carlo semivariogram of two window means
      equals V(h | theta, 0) + sigma_b^2 h^2 / 2 within 3.5 SE, and the tilt
      term of every trace sums to 0 over the trace (|sum| <= 1e-9 x sigma_b T)
  M6  contract and determinism: (n_traces, n) float64 finite output; the same
      unit draws give bit-identical traces and different draws differ;
      simulate_sweep equals row `index` of simulate_traces; invalid laws
      (rho_f outside (0, 1), tau_s <= 0, negative sigma, NaN) raise ValueError;
      draw_units has the documented shapes and order
  M7  visible share: for a slow-only law E[S_0]/(n gamma(0)) equals v(z) of
      eq. (8) at z = T/tau in {0.25, 1.26, 6.67} within 2 z/n (the
      discrete-continuous gap), n = 4000
  M8  regression against the topic document: closed_form_sd and
      closed_form_acf at the cohort closed-form estimate theta_cf =
      (0.0380 mV, 0.175, 0.0508 mV, 36.6 ms) reproduce the 14 q^cf values of
      topic doc 3.5.3 (SS 10 ms, LS 265 ms, first 50 ms, last 132.5 ms at
      50 kHz) within 0.0003 mV and 0.003 (the table's rounding plus theta_cf's
      own 3-digit rounding)
  M9  byte safety: this file and noise_law.py are pure ASCII with LF endings

Run from this folder (needs numpy, scipy):

    python smoke_noise_law.py

Expected last line: "SMOKE noise_law: 9/9 passed". Exit 1 on any FAIL.
"""

import sys
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from noise_law import (NoiseLaw, UnitDraws, autocovariance, closed_form_acf, closed_form_sd,  # noqa: E402
                       draw_units, expected_s0, expected_sl, semivariogram, semivariogram_continuous,
                       simulate_sweep, simulate_traces, var_window_mean, visible_share, window_mean_cov)

RESULTS = []
DT = 20e-6                                              # s, 50 kHz
THETA_SIM = NoiseLaw(0.0383, 0.179, 0.0547, 0.0397)     # cohort diagnostic (topic doc 3.5.3)
THETA_CF = NoiseLaw(0.0380, 0.175, 0.0508, 0.0366)      # cohort closed-form estimate
Z_SE = 3.5                                              # Monte Carlo tolerance in standard errors


def report(name, ok, note=""):
    RESULTS.append((name, bool(ok), note))
    print("%s  %s%s" % ("PASS" if ok else "FAIL", name, ("  -- " + note) if note else ""))


def gamma_matrix(law, dt, n):
    idx = np.arange(n)
    return autocovariance(law, dt, idx[:, None] - idx[None, :])


def brute_moments(law, dt, n, lags):
    """Var(x_bar), E[S_0], E[S_L] from the full covariance matrix."""
    G = gamma_matrix(law, dt, n)
    P = np.eye(n) - np.full((n, n), 1.0 / n)
    PGP = P @ G @ P
    var_mean = float(np.ones(n) @ G @ np.ones(n)) / n ** 2
    e_s0 = float(np.trace(PGP))
    e_sl = np.array([float(np.sum(np.diag(PGP, k=int(L)))) for L in lags])
    return var_mean, e_s0, e_sl


def _m1():
    n, lags = 240, np.array([1, 2, 5, 50, 200])
    worst = 0.0
    for law in (THETA_SIM, NoiseLaw(0.0, 0.5, 0.05, 0.0397), NoiseLaw(0.04, 0.6, 0.0, 1.0)):
        vb, s0b, slb = brute_moments(law, DT, n, lags)
        worst = max(worst, abs(var_window_mean(law, DT, n) - vb) / vb,
                    abs(expected_s0(law, DT, n) - s0b) / s0b,
                    float(np.max(np.abs(expected_sl(law, DT, n, lags) - slb) / s0b)))
    report("M1 brute force: (7a), (7b), (9) == full-covariance moments at n=240, 3 laws",
           worst <= 1e-10, "max rel |diff| %.1e" % worst)


def _sample_stats(x, lags):
    u = x - x.mean(axis=1, keepdims=True)
    s0 = np.einsum("ij,ij->i", u, u)
    sl = np.array([np.einsum("ij,ij->i", u[:, L:], u[:, :-L]) for L in lags])
    return s0, sl


def _m2():
    rng = np.random.default_rng(20261019)
    R, n, lags = 4000, 2000, [1, 50]
    units = draw_units(rng, R, n)
    x = simulate_traces(THETA_SIM, DT, units.fast, units.slow)
    s0, sl = _sample_stats(x, lags)
    notes, ok = [], True
    for name, sample, exact in (("S0", s0, expected_s0(THETA_SIM, DT, n)),
                                ("S1", sl[0], expected_sl(THETA_SIM, DT, n, 1)[0]),
                                ("S50", sl[1], expected_sl(THETA_SIM, DT, n, 50)[0])):
        z = (sample.mean() - exact) / (sample.std(ddof=1) / np.sqrt(R))
        ok &= abs(z) <= Z_SE
        notes.append("%s z=%+.2f" % (name, z))
    for L in (0, 1, 50):
        prod = x[:, L:] * x[:, :n - L] if L else x * x
        acov = prod.mean(axis=1)                              # per-trace lag product mean
        z = (acov.mean() - autocovariance(THETA_SIM, DT, L)) / (acov.std(ddof=1) / np.sqrt(R))
        ok &= abs(z) <= Z_SE
        notes.append("gamma(%d) z=%+.2f" % (L, z))
    m = x.mean(axis=1)
    z = m.mean() / (m.std(ddof=1) / np.sqrt(R))
    ok &= abs(z) <= Z_SE
    notes.append("mean z=%+.2f" % z)
    report("M2 Monte Carlo (R=4000, n=2000) vs (7b), (9), (L3): all within %.1f SE" % Z_SE, ok, "; ".join(notes))


def _m3():
    rng = np.random.default_rng(3)
    R, n = 20000, 60
    units = draw_units(rng, R, n)
    x = simulate_traces(THETA_SIM, DT, units.fast, units.slow)
    g0 = autocovariance(THETA_SIM, DT, 0)
    notes, ok = [], True
    for name, col in (("first", x[:, 0]), ("last", x[:, -1])):
        v = col ** 2
        z = (v.mean() - g0) / (v.std(ddof=1) / np.sqrt(R))
        ok &= abs(z) <= Z_SE
        notes.append("%s-sample var z=%+.2f" % (name, z))
    # a recursion started at 0 instead of the stationary law would give first-sample variance 0
    report("M3 stationarity: first and last sample variance == gamma(0) within %.1f SE" % Z_SE, ok, "; ".join(notes))


def _m4():
    # (a) brute force on small windows, overlapping and disjoint, negative offset
    n = 120
    G = gamma_matrix(THETA_SIM, DT, n)
    worst = 0.0
    for n1, n2, off in ((30, 30, 50), (30, 40, 20), (25, 25, 0), (40, 20, -10), (50, 50, 70)):
        lo = min(0, off)
        i1 = np.arange(0, n1) - lo
        i2 = np.arange(off, off + n2) - lo
        brute = G[np.ix_(i1, i2)].mean()
        worst = max(worst, abs(window_mean_cov(THETA_SIM, DT, n1, n2, off) - brute) / abs(brute))
    ok_a = worst <= 1e-10
    # (b) Monte Carlo: two 190 ms windows 200 ms apart in one trace, dt = 0.1 ms
    dt, n1, off = 1e-4, 1900, 2000
    rng = np.random.default_rng(4)
    R = 4000
    units = draw_units(rng, R, off + n1)
    x = simulate_traces(THETA_SIM, dt, units.fast, units.slow)
    m1, m2 = x[:, :n1].mean(axis=1), x[:, off:off + n1].mean(axis=1)
    prod = (m1 - m1.mean()) * (m2 - m2.mean())
    cov_exact = window_mean_cov(THETA_SIM, dt, n1, n1, off)
    z = (prod.mean() * R / (R - 1) - cov_exact) / (prod.std(ddof=1) / np.sqrt(R))
    ok_b = abs(z) <= Z_SE
    # (c) exact discrete vs continuous eq. (25), slow-only law
    slow = NoiseLaw(0.0, 0.5, 0.0547, 0.0397)
    v_disc = semivariogram(slow, dt, n1, n1, off)
    v_cont = semivariogram_continuous(slow, dt, n1 * dt, n1 * dt, off * dt)
    rel = abs(v_disc - v_cont) / v_cont
    tol_c = 3.0 * dt / slow.tau_s
    ok_c = rel <= tol_c
    report("M4 window means: cov == brute force (1e-10); MC cov within %.1f SE; discrete V vs eq. (25)" % Z_SE,
           ok_a and ok_b and ok_c,
           "brute rel %.1e; MC z=%+.2f; |V_disc-V_cont|/V %.2e (tol %.2e); corr(adjacent means) %.3f"
           % (worst, z, rel, tol_c, cov_exact / var_window_mean(THETA_SIM, dt, n1)))


def _m5():
    dt, n1, off, sigma_b = 1e-4, 1900, 2000, 0.1          # mV/s
    rng = np.random.default_rng(5)
    R = 4000
    units = draw_units(rng, R, off + n1)
    x = simulate_traces(THETA_SIM, dt, units.fast, units.slow, xi_tilt=units.tilt, sigma_b=sigma_b)
    d = x[:, off:off + n1].mean(axis=1) - x[:, :n1].mean(axis=1)
    half_sq = 0.5 * d ** 2
    v_exact = semivariogram(THETA_SIM, dt, n1, n1, off, sigma_b=sigma_b)
    v_stat = semivariogram(THETA_SIM, dt, n1, n1, off)
    h = off * dt
    z = (half_sq.mean() - v_exact) / (half_sq.std(ddof=1) / np.sqrt(R))
    ok_mc = abs(z) <= Z_SE and abs(v_exact - v_stat - 0.5 * sigma_b ** 2 * h ** 2) <= 1e-15
    # the tilt alone sums to zero over the trace
    n = off + n1
    tilt_only = simulate_traces(NoiseLaw(0.0, 0.5, 0.0, 1.0), dt, units.fast[:5], units.slow[:5],
                                xi_tilt=units.tilt[:5], sigma_b=sigma_b)
    ok_sum = float(np.max(np.abs(tilt_only.sum(axis=1)))) <= 1e-9 * sigma_b * n * dt
    report("M5 tilt: MC semivariogram == V(h|theta,0) + sigma_b^2 h^2/2 within %.1f SE; tilt sums to 0" % Z_SE,
           ok_mc and ok_sum, "z=%+.2f; tilt share of V %.2f; max |sum tilt| %.1e mV"
           % (z, 0.5 * sigma_b ** 2 * h ** 2 / v_exact, float(np.max(np.abs(tilt_only.sum(axis=1))))))


def _m6():
    rng = np.random.default_rng(6)
    R, n = 7, 500
    u = draw_units(rng, R, n)
    shapes_ok = (u.fast.shape == (R, n) and u.slow.shape == (R, n) and u.tilt.shape == (R,)
                 and u.offset.shape == (R,))
    # fixed order: the same generator state gives the same fast block first
    u2 = draw_units(np.random.default_rng(6), R, n)
    order_ok = np.array_equal(u.fast, u2.fast) and np.array_equal(u.offset, u2.offset)
    x = simulate_traces(THETA_SIM, DT, u.fast, u.slow, xi_tilt=u.tilt, sigma_b=0.05,
                        xi_offset=u.offset, sigma_a=0.3)
    y = simulate_traces(THETA_SIM, DT, u.fast, u.slow, xi_tilt=u.tilt, sigma_b=0.05,
                        xi_offset=u.offset, sigma_a=0.3)
    contract_ok = x.shape == (R, n) and x.dtype == np.float64 and np.all(np.isfinite(x))
    det_ok = np.array_equal(x, y)
    other = draw_units(np.random.default_rng(7), R, n)
    diff_ok = not np.array_equal(x, simulate_traces(THETA_SIM, DT, other.fast, other.slow,
                                                     xi_tilt=other.tilt, sigma_b=0.05,
                                                     xi_offset=other.offset, sigma_a=0.3))
    sweep_ok = np.array_equal(simulate_sweep(THETA_SIM, DT, n, u, 3, sigma_b=0.05, sigma_a=0.3), x[3])
    bad = 0
    for args in ((0.04, 1.0, 0.05, 0.04), (0.04, 0.0, 0.05, 0.04), (0.04, 0.2, 0.05, 0.0),
                 (-0.04, 0.2, 0.05, 0.04), (0.04, 0.2, float("nan"), 0.04), (0.04, -0.3, 0.05, 0.04)):
        try:
            NoiseLaw(*args)
        except ValueError:
            bad += 1
    rt = NoiseLaw.from_array(THETA_SIM.as_array()) == THETA_SIM
    report("M6 contract, determinism, simulate_sweep, validation (6 invalid laws), draw order",
           shapes_ok and order_ok and contract_ok and det_ok and diff_ok and sweep_ok and bad == 6 and rt,
           "invalid laws refused %d/6" % bad)


def _m7():
    n, tau = 4000, 0.0397
    worst, notes, ok = 0.0, [], True
    for z in (0.25, 1.26, 6.67):
        dt = z * tau / n
        law = NoiseLaw(0.0, 0.5, 0.0547, tau)
        share = expected_s0(law, dt, n) / (n * autocovariance(law, dt, 0))
        diff = abs(share - float(visible_share(z)))
        ok &= diff <= 2.0 * z / n
        notes.append("z=%.2f: %.4f vs v(z) %.4f" % (z, share, float(visible_share(z))))
        worst = max(worst, diff)
    report("M7 visible share (8): E[S_0]/(n gamma(0)) == v(z) within 2z/n, slow-only law", ok,
           "; ".join(notes) + "; max |diff| %.1e" % worst)


def _m8():
    # topic doc 3.5.3, column q^cf(theta_cf), 50 kHz: windows and lags in samples
    fs = 50_000.0
    ms = lambda x: int(round(x * 1e-3 * fs))  # noqa: E731
    table = [  # (n, [lags], sd_ref, [acf_refs])
        (ms(10), [ms(0.02), ms(0.1), ms(1)], 0.0408, [0.280, 0.123, 0.070]),
        (ms(265), [ms(0.02), ms(0.1), ms(1)], 0.0584, [0.650, 0.574, 0.553]),
        (ms(50), [], 0.0480, []),
        (ms(132.5), [ms(0.02), ms(0.1), ms(1), ms(10)], 0.0546, [0.599, 0.512, 0.487, 0.272]),
    ]
    worst_sd = worst_acf = 0.0
    for n, lags, sd_ref, acf_ref in table:
        worst_sd = max(worst_sd, abs(closed_form_sd(THETA_CF, DT, n) - sd_ref))
        if lags:
            worst_acf = max(worst_acf, float(np.max(np.abs(closed_form_acf(THETA_CF, DT, n, lags) - np.array(acf_ref)))))
    # tolerance: the table rounds to 3 decimals (SDs to 4) and theta_cf itself to 3 digits
    report("M8 regression: q^cf(theta_cf) reproduces the 14 topic-doc values (SD 3e-4 mV, acf 3e-3)",
           worst_sd <= 3e-4 and worst_acf <= 3e-3, "max |diff| SD %.1e mV, acf %.1e" % (worst_sd, worst_acf))


def _m9():
    bad = []
    for f in ("smoke_noise_law.py", "noise_law.py"):
        b = (HERE / f).read_bytes()
        if any(c > 127 for c in b):
            bad.append(f + ":non-ascii")
        if b"\r\n" in b:
            bad.append(f + ":crlf")
    report("M9 ASCII + LF on noise_law.py and this file", not bad, ", ".join(bad) if bad else "ok")


def main():
    for fn in (_m1, _m2, _m3, _m4, _m5, _m6, _m7, _m8, _m9):
        try:
            fn()
        except BaseException as exc:          # SystemExit included (D-010 (iii) lineage)
            traceback.print_exc()
            report(fn.__name__ + " raised", False, repr(exc))
    n_ok = sum(1 for _, ok, _ in RESULTS if ok)
    print("SMOKE noise_law: %d/%d passed" % (n_ok, len(RESULTS)))
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
