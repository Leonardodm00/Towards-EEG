"""
smoke_noise_law.py -- smoke suite for noise_law (M1-M11)
=========================================================

Block 2 of specs/SPEC.md (D-017, D-034, D-056; the three-component law of
SPEC 2.6 (a), (b), (h), D-152; re-aimed 2026-11-17 as the Block 2 note of that
date says). No NEURON, no archive; under a minute on one core. Truths are
chosen, so every check has a known answer. THETA below is the fixture law of
the Block 3 note of 2026-11-17 (sigma_f 0.035 mV, sigma_m 0.017 mV, tau_m
0.3 ms, sigma_s 0.055 mV, tau_s 40 ms).

  M1  brute force: Var(x_bar), E[S_0] and E[S_L] of eqs. (7a), (7b), (9) equal
      the same moments computed from the full n x n covariance matrix Gamma of
      (L3') (E[S_L] = sum_j (P Gamma P)_{j,j+L}, P the centering matrix) at
      n = 240, lags {1, 2, 5, 15, 50, 200}, for THETA and three one-component
      laws (white, middle, slow), to 1e-10 relative (float64 on sums of 6e4
      terms)
  M2  Monte Carlo of the simulator against (L3') and the closed forms: 4000
      traces of n = 2000 at dt = 20 us, THETA: the across-trace mean of the
      per-trace lag-product average at lags 0, 1, 2, 5, 15, 50, 500 within
      3.5 SE of gamma (L3'); the sample mean of S_0 and S_L (L = 1, 15, 50)
      within 3.5 SE of (7b) and (9); the grand mean within 3.5 SE of 0
      (Monte Carlo spread: 3.5 SE, one-sided probability 2e-4 per comparison)
  M3  stationarity: for THETA and for each one-component law, the
      across-trace variance of the FIRST and of the LAST sample equal
      gamma(0) within 3.5 SE (a recursion started off its stationary law
      fails the first)
  M4  window means of one trace: window_mean_cov equals brute force to 1e-10
      (overlapping and disjoint windows, negative offset); the Monte Carlo
      covariance of two 190 ms window means 200 ms apart (dt = 0.1 ms, 4000
      traces) within 3.5 SE of it; the exact-discrete semivariogram agrees
      with the continuous-time eq. (25) for a slow-only law within 3 dt/tau_s
      relative (the discretisation error of a sampled window mean), and for a
      white-only law to 1e-12 relative (both take its sampled value)
  M5  tilt: with sigma_b > 0 the Monte Carlo semivariogram of two window means
      equals V(h | theta, 0) + sigma_b^2 h^2 / 2 within 3.5 SE, and the tilt
      term of every trace sums to 0 over the trace (|sum| <= 1e-9 x sigma_b T)
  M6  contract and determinism: (n_traces, n) float64 finite output; the same
      unit draws give bit-identical traces and different draws differ;
      simulate_sweep equals row `index` of simulate_traces; six invalid laws
      (tau_m <= 0, tau_s <= 0, a negative sigma of each kind, NaN, inf) raise
      ValueError; as_array / from_array round trip on 5 entries in the order
      (sigma_f, sigma_m, tau_m, sigma_s, tau_s), 4 or 6 entries refused
  M7  visible share: for a slow-only law E[S_0]/(n gamma(0)) equals v(z) of
      eq. (8) at z = T/tau in {0.25, 1.26, 6.67} within 2 z/n (the
      discrete-continuous gap), n = 4000
  M8  analytic white part and additivity (replaces the regression against
      topic doc 3.5.3, whose 14 values belong to the retired two-component
      law): for a white-only law E[S_0] = (n - 1) sigma_f^2, E[S_L] = -(n - L)
      sigma_f^2 / n, closed_form_sd = sigma_f and closed_form_acf = -(n - L) /
      (n (n - 1)) at n in {12, 500, 13250} to 10 n eps relative (the
      cancellation of cumulative sums of n terms at L near n); for THETA, Var(x_bar), E[S_0] and E[S_L]
      equal the sums of those of its three one-component laws to 1e-12
      relative (the moments are linear in gamma)
  M9  byte safety: this file and noise_law.py are pure ASCII with LF endings
  M10 fixed tilt (D-102.1, D-107.2; added 2026-10-29): with tilt_mV_per_s = b
      the traces minus the same draws without a tilt equal b[r] (t_j - t_bar)
      to 1e-12 mV (roundoff of one float64 addition of mV-sized terms), for
      tilts of both signs and a zero; tilt_mV_per_s = sigma_b xi_tilt is
      bit-identical to the pair (xi_tilt, sigma_b); simulate_sweep with a
      scalar tilt equals row `index` of simulate_traces; both tilts at once,
      a wrong shape and a NaN entry each raise ValueError
  M11 draw order and components switched off (SPEC 2.6 (b), (a)): draw_units
      returns exactly rng.standard_normal drawn fast (n_traces, n), mid, slow,
      tilt (n_traces,), offset in that order from the same seed; with a zero
      SD, a component's draws do not touch the traces (bit-identical when they
      are replaced) and its term is absent from (L3'); the traces of a
      one-component law equal that component written out here (white:
      sigma_f xi; OU: the recursion e_0 = sigma xi_0, e_j = rho e_{j-1} +
      sigma sqrt(1 - rho^2) xi_j by a Python loop) to 1e-14 mV (float64
      roundoff of the recursion accumulated over 400 samples)

Run from this folder (needs numpy, scipy):

    python smoke_noise_law.py

Expected last line: "SMOKE noise_law: 11/11 passed". Exit 1 on any FAIL.
"""

import sys
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from noise_law import (NoiseLaw, autocovariance, closed_form_acf, closed_form_sd,  # noqa: E402
                       draw_units, expected_s0, expected_sl, semivariogram, semivariogram_continuous,
                       simulate_sweep, simulate_traces, var_window_mean, visible_share, window_mean_cov)

RESULTS = []
DT = 20e-6                                              # s, 50 kHz
THETA = NoiseLaw(0.035, 0.017, 3e-4, 0.055, 0.040)      # the fixture law of the Block 3 note (2026-11-17)
WHITE = NoiseLaw(0.035, 0.0, 3e-4, 0.0, 0.040)          # THETA's three components, one at a time
MID = NoiseLaw(0.0, 0.017, 3e-4, 0.0, 0.040)
SLOW = NoiseLaw(0.0, 0.0, 3e-4, 0.055, 0.040)
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


def _sim(law, dt, u, **kw):
    return simulate_traces(law, dt, u.fast, u.mid, u.slow, **kw)


def _m1():
    n, lags = 240, np.array([1, 2, 5, 15, 50, 200])
    worst = 0.0
    for law in (THETA, WHITE, MID, SLOW):
        vb, s0b, slb = brute_moments(law, DT, n, lags)
        # E[S_L] is scaled by E[S_0]: for the white law E[S_L] = -(n-L) sigma^2 / n is small, not 0
        worst = max(worst, abs(var_window_mean(law, DT, n) - vb) / vb,
                    abs(expected_s0(law, DT, n) - s0b) / s0b,
                    float(np.max(np.abs(expected_sl(law, DT, n, lags) - slb) / s0b)))
    report("M1 brute force: (7a), (7b), (9) == full-covariance moments of (L3') at n=240, 4 laws",
           worst <= 1e-10, "max rel |diff| %.1e" % worst)


def _sample_stats(x, lags):
    u = x - x.mean(axis=1, keepdims=True)
    s0 = np.einsum("ij,ij->i", u, u)
    sl = np.array([np.einsum("ij,ij->i", u[:, L:], u[:, :-L]) for L in lags])
    return s0, sl


def _m2():
    rng = np.random.default_rng(20261117)
    R, n, lags = 4000, 2000, [1, 15, 50]
    units = draw_units(rng, R, n)
    x = _sim(THETA, DT, units)
    notes, ok = [], True
    # (L3') at the lags of SPEC's Block 2 note: 0, 1, 2, 5, 15, 50, 500
    for L in (0, 1, 2, 5, 15, 50, 500):
        prod = x[:, L:] * x[:, :n - L] if L else x * x
        acov = prod.mean(axis=1)                              # per-trace lag-product mean (traces independent)
        z = (acov.mean() - autocovariance(THETA, DT, L)) / (acov.std(ddof=1) / np.sqrt(R))
        ok &= abs(z) <= Z_SE
        notes.append("gamma(%d) z=%+.2f" % (L, z))
    s0, sl = _sample_stats(x, lags)
    exact_sl = expected_sl(THETA, DT, n, lags)
    for name, sample, exact in [("S0", s0, expected_s0(THETA, DT, n))] + [
            ("S%d" % L, sl[i], exact_sl[i]) for i, L in enumerate(lags)]:
        z = (sample.mean() - exact) / (sample.std(ddof=1) / np.sqrt(R))
        ok &= abs(z) <= Z_SE
        notes.append("%s z=%+.2f" % (name, z))
    m = x.mean(axis=1)
    z = m.mean() / (m.std(ddof=1) / np.sqrt(R))
    ok &= abs(z) <= Z_SE
    notes.append("mean z=%+.2f" % z)
    report("M2 Monte Carlo (R=4000, n=2000) vs (L3') at 7 lags, (7b), (9): all within %.1f SE" % Z_SE, ok,
           "; ".join(notes))


def _m3():
    rng = np.random.default_rng(3)
    R, n = 20000, 60
    units = draw_units(rng, R, n)
    notes, ok = [], True
    for lname, law in (("theta", THETA), ("white", WHITE), ("mid", MID), ("slow", SLOW)):
        x = _sim(law, DT, units)
        g0 = autocovariance(law, DT, 0)
        for name, col in (("first", x[:, 0]), ("last", x[:, -1])):
            v = col ** 2
            z = (v.mean() - g0) / (v.std(ddof=1) / np.sqrt(R))
            ok &= abs(z) <= Z_SE
            notes.append("%s %s z=%+.2f" % (lname, name, z))
    # a recursion started off its stationary law (e_0 = sigma sqrt(1-rho^2) xi_0) gives a first-sample variance
    # sigma^2 (1 - rho^2): for the middle part 0.064 sigma_m^2, about 66 SE below gamma(0) at R = 20000
    report("M3 stationarity: first and last sample variance == gamma(0) within %.1f SE, theta and each component"
           % Z_SE, ok, "; ".join(notes))


def _m4():
    # (a) brute force on small windows, overlapping and disjoint, negative offset
    n = 120
    G = gamma_matrix(THETA, DT, n)
    worst = 0.0
    for n1, n2, off in ((30, 30, 50), (30, 40, 20), (25, 25, 0), (40, 20, -10), (50, 50, 70)):
        lo = min(0, off)
        i1 = np.arange(0, n1) - lo
        i2 = np.arange(off, off + n2) - lo
        brute = G[np.ix_(i1, i2)].mean()
        worst = max(worst, abs(window_mean_cov(THETA, DT, n1, n2, off) - brute) / abs(brute))
    ok_a = worst <= 1e-10
    # (b) Monte Carlo: two 190 ms windows 200 ms apart in one trace, dt = 0.1 ms
    dt, n1, off = 1e-4, 1900, 2000
    rng = np.random.default_rng(4)
    R = 4000
    units = draw_units(rng, R, off + n1)
    x = _sim(THETA, dt, units)
    m1, m2 = x[:, :n1].mean(axis=1), x[:, off:off + n1].mean(axis=1)
    prod = (m1 - m1.mean()) * (m2 - m2.mean())
    cov_exact = window_mean_cov(THETA, dt, n1, n1, off)
    z = (prod.mean() * R / (R - 1) - cov_exact) / (prod.std(ddof=1) / np.sqrt(R))
    ok_b = abs(z) <= Z_SE
    # (c) exact discrete vs continuous eq. (25): slow-only law within 3 dt/tau_s; white-only law exactly
    slow = NoiseLaw(0.0, 0.0, 3e-4, 0.0547, 0.0397)
    v_disc = semivariogram(slow, dt, n1, n1, off)
    v_cont = semivariogram_continuous(slow, dt, n1 * dt, n1 * dt, off * dt)
    rel = abs(v_disc - v_cont) / v_cont
    tol_c = 3.0 * dt / slow.tau_s
    vw_disc = semivariogram(WHITE, dt, n1, n1, off)
    vw_cont = semivariogram_continuous(WHITE, dt, n1 * dt, n1 * dt, off * dt)
    rel_w = abs(vw_disc - vw_cont) / vw_cont
    ok_c = rel <= tol_c and rel_w <= 1e-12          # white: both sigma_f^2 / n1, float64 roundoff
    report("M4 window means: cov == brute force (1e-10); MC cov within %.1f SE; discrete V vs eq. (25)" % Z_SE,
           ok_a and ok_b and ok_c,
           "brute rel %.1e; MC z=%+.2f; slow |V_disc-V_cont|/V %.2e (tol %.2e); white %.1e; corr(adjacent means) %.3f"
           % (worst, z, rel, tol_c, rel_w, cov_exact / var_window_mean(THETA, dt, n1)))


def _m5():
    dt, n1, off, sigma_b = 1e-4, 1900, 2000, 0.1          # mV/s
    rng = np.random.default_rng(5)
    R = 4000
    units = draw_units(rng, R, off + n1)
    x = _sim(THETA, dt, units, xi_tilt=units.tilt, sigma_b=sigma_b)
    d = x[:, off:off + n1].mean(axis=1) - x[:, :n1].mean(axis=1)
    half_sq = 0.5 * d ** 2
    v_exact = semivariogram(THETA, dt, n1, n1, off, sigma_b=sigma_b)
    v_stat = semivariogram(THETA, dt, n1, n1, off)
    h = off * dt
    z = (half_sq.mean() - v_exact) / (half_sq.std(ddof=1) / np.sqrt(R))
    ok_mc = abs(z) <= Z_SE and abs(v_exact - v_stat - 0.5 * sigma_b ** 2 * h ** 2) <= 1e-15
    # the tilt alone sums to zero over the trace
    n = off + n1
    silent = NoiseLaw(0.0, 0.0, 3e-4, 0.0, 1.0)
    tilt_only = simulate_traces(silent, dt, units.fast[:5], units.mid[:5], units.slow[:5],
                                xi_tilt=units.tilt[:5], sigma_b=sigma_b)
    ok_sum = float(np.max(np.abs(tilt_only.sum(axis=1)))) <= 1e-9 * sigma_b * n * dt
    report("M5 tilt: MC semivariogram == V(h|theta,0) + sigma_b^2 h^2/2 within %.1f SE; tilt sums to 0" % Z_SE,
           ok_mc and ok_sum, "z=%+.2f; tilt share of V %.2f; max |sum tilt| %.1e mV"
           % (z, 0.5 * sigma_b ** 2 * h ** 2 / v_exact, float(np.max(np.abs(tilt_only.sum(axis=1))))))


def _m6():
    rng = np.random.default_rng(6)
    R, n = 7, 500
    u = draw_units(rng, R, n)
    shapes_ok = (u.fast.shape == (R, n) and u.mid.shape == (R, n) and u.slow.shape == (R, n)
                 and u.tilt.shape == (R,) and u.offset.shape == (R,))
    x = _sim(THETA, DT, u, xi_tilt=u.tilt, sigma_b=0.05, xi_offset=u.offset, sigma_a=0.3)
    y = _sim(THETA, DT, u, xi_tilt=u.tilt, sigma_b=0.05, xi_offset=u.offset, sigma_a=0.3)
    contract_ok = x.shape == (R, n) and x.dtype == np.float64 and np.all(np.isfinite(x))
    det_ok = np.array_equal(x, y)
    other = draw_units(np.random.default_rng(7), R, n)
    diff_ok = not np.array_equal(x, _sim(THETA, DT, other, xi_tilt=other.tilt, sigma_b=0.05,
                                         xi_offset=other.offset, sigma_a=0.3))
    sweep_ok = np.array_equal(simulate_sweep(THETA, DT, n, u, 3, sigma_b=0.05, sigma_a=0.3), x[3])
    bad = 0
    for args in ((0.04, 0.02, 0.0, 0.05, 0.04), (0.04, 0.02, 3e-4, 0.05, -0.04), (-0.04, 0.02, 3e-4, 0.05, 0.04),
                 (0.04, -0.02, 3e-4, 0.05, 0.04), (0.04, 0.02, 3e-4, float("nan"), 0.04),
                 (0.04, 0.02, float("inf"), 0.05, 0.04)):
        try:
            NoiseLaw(*args)
        except ValueError:
            bad += 1
    neg_s = 0
    try:
        NoiseLaw(0.04, 0.02, 3e-4, -0.05, 0.04)
    except ValueError:
        neg_s = 1
    arr = THETA.as_array()
    rt = (NoiseLaw.from_array(arr) == THETA and arr.shape == (5,) and arr.dtype == np.float64
          and list(arr) == [0.035, 0.017, 3e-4, 0.055, 0.040])
    wrong_len = 0
    for k in (4, 6):
        try:
            NoiseLaw.from_array(np.ones(k))
        except ValueError:
            wrong_len += 1
    ok = (shapes_ok and contract_ok and det_ok and diff_ok and sweep_ok and bad == 6 and neg_s == 1 and rt
          and wrong_len == 2)
    report("M6 contract, determinism, simulate_sweep, validation (6 invalid laws + negative sigma_s), "
           "as_array/from_array on 5 entries", ok,
           "invalid laws refused %d/6, negative sigma_s %d/1; round trip %s; 4 and 6 entries refused %d/2"
           % (bad, neg_s, rt, wrong_len))


def _m7():
    n, tau = 4000, 0.0397
    worst, notes, ok = 0.0, [], True
    for z in (0.25, 1.26, 6.67):
        dt = z * tau / n
        law = NoiseLaw(0.0, 0.0, 3e-4, 0.0547, tau)
        share = expected_s0(law, dt, n) / (n * autocovariance(law, dt, 0))
        diff = abs(share - float(visible_share(z)))
        ok &= diff <= 2.0 * z / n
        notes.append("z=%.2f: %.4f vs v(z) %.4f" % (z, share, float(visible_share(z))))
        worst = max(worst, diff)
    report("M7 visible share (8): E[S_0]/(n gamma(0)) == v(z) within 2z/n, slow-only law", ok,
           "; ".join(notes) + "; max |diff| %.1e" % worst)


def _m8():
    # (a) the white part in closed form: for iid samples of variance s2, E[S_0] = (n-1) s2 and
    # E[S_L] = sum_j E[u_j u_{j+L}] = (n-L) (-s2/n); hence sd = sigma_f and acf = -(n-L) / (n (n-1))
    # tolerance 10 n eps relative: at L near n, (9) subtracts cumulative sums of n terms of size sigma_f^2 to
    # leave -(n-L) sigma_f^2 / n, so its relative roundoff grows like n eps (n eps = 2.9e-12 at n = 13250)
    s2 = WHITE.sigma_f ** 2
    worst_w = 0.0
    eps = np.finfo(float).eps
    for n in (12, 500, 13250):
        lags = np.array([1, 2, 5, n // 2, n - 1])
        worst_w = max(worst_w, (1.0 / (10.0 * n * eps)) * max(abs(expected_s0(WHITE, DT, n) / ((n - 1) * s2) - 1.0),
                      float(np.max(np.abs(expected_sl(WHITE, DT, n, lags) / (-(n - lags) * s2 / n) - 1.0))),
                      abs(closed_form_sd(WHITE, DT, n) / WHITE.sigma_f - 1.0),
                      float(np.max(np.abs(closed_form_acf(WHITE, DT, n, lags) / (-(n - lags) / (n * (n - 1.0))) - 1.0))),
                      abs(var_window_mean(WHITE, DT, n) / (s2 / n) - 1.0)))
    # (b) additivity: every moment is linear in gamma, and gamma of THETA is the sum of its components'
    worst_add = 0.0
    for n in (240, 13250):
        lags = np.array([1, 2, 5, 15, 50, 200])
        parts = (WHITE, MID, SLOW)
        worst_add = max(worst_add,
                        abs(var_window_mean(THETA, DT, n) / sum(var_window_mean(p, DT, n) for p in parts) - 1.0),
                        abs(expected_s0(THETA, DT, n) / sum(expected_s0(p, DT, n) for p in parts) - 1.0),
                        float(np.max(np.abs(expected_sl(THETA, DT, n, lags) - sum(expected_sl(p, DT, n, lags)
                                                                                  for p in parts))
                                     / expected_s0(THETA, DT, n))))
    # tolerance 1e-12: float64 roundoff of O(n) cumulative sums (n <= 13250) of terms of one sign or nearly so
    report("M8 analytic white part (E[S_0], E[S_L], sd, acf, Var mean) and additivity over the 3 components",
           worst_w <= 1.0 and worst_add <= 1e-12,
           "white max rel / (10 n eps) %.2f; additivity max rel %.1e" % (worst_w, worst_add))


def _m9():
    bad = []
    for f in ("smoke_noise_law.py", "noise_law.py"):
        b = (HERE / f).read_bytes()
        if any(c > 127 for c in b):
            bad.append(f + ":non-ascii")
        if b"\r" in b:
            bad.append(f + ":cr")
    report("M9 ASCII + LF on noise_law.py and this file", not bad, ", ".join(bad) if bad else "ok")


def _m10():
    rng = np.random.default_rng(10)
    R, n = 4, 3000
    u = draw_units(rng, R, n)
    b = np.array([0.1, -0.25, 0.0, 1.7])                                   # mV/s, both signs and a zero
    with_tilt = _sim(THETA, DT, u, tilt_mV_per_s=b)
    without = _sim(THETA, DT, u)
    t_bar = 0.5 * (n - 1) * DT
    expect = b[:, None] * (np.arange(n) * DT - t_bar)[None, :]
    err = float(np.max(np.abs((with_tilt - without) - expect)))
    ok_formula = err <= 1e-12
    # the fixed tilt sigma_b xi_tilt reproduces the random-tilt path bit for bit
    sb = 0.08
    ok_same = np.array_equal(_sim(THETA, DT, u, tilt_mV_per_s=sb * u.tilt),
                             _sim(THETA, DT, u, xi_tilt=u.tilt, sigma_b=sb))
    ok_sweep = np.array_equal(simulate_sweep(THETA, DT, n, u, 1, tilt_mV_per_s=float(b[1])), with_tilt[1])
    refused = 0
    for kwargs in (dict(tilt_mV_per_s=b, xi_tilt=u.tilt, sigma_b=0.1), dict(tilt_mV_per_s=b[:3]),
                   dict(tilt_mV_per_s=np.array([0.1, np.nan, 0.0, 0.2]))):
        try:
            _sim(THETA, DT, u, **kwargs)
        except ValueError:
            refused += 1
    report("M10 fixed tilt: b (t - t_bar) added exactly; = the random-tilt path for sigma_b xi_tilt; "
           "simulate_sweep row; 3 misuses refused",
           ok_formula and ok_same and ok_sweep and refused == 3,
           "max |diff - b (t - t_bar)| %.1e mV; bit-identical %s; sweep row %s; refused %d/3"
           % (err, ok_same, ok_sweep, refused))


def _ou_loop(sigma, tau, dt, xi):
    """The stationary recursion written out sample by sample (oracle for the lfilter path)."""
    rho = np.exp(-dt / tau)
    out = np.empty_like(xi)
    out[:, 0] = sigma * xi[:, 0]
    for j in range(1, xi.shape[1]):
        out[:, j] = rho * out[:, j - 1] + sigma * np.sqrt(1.0 - rho ** 2) * xi[:, j]
    return out


def _m11():
    R, n = 3, 400
    # (a) the draw order fast, mid, slow, tilt, offset (SPEC 2.6 (b))
    u = draw_units(np.random.default_rng(11), R, n)
    g = np.random.default_rng(11)
    ref = [g.standard_normal((R, n)), g.standard_normal((R, n)), g.standard_normal((R, n)),
           g.standard_normal(R), g.standard_normal(R)]
    ok_order = all(np.array_equal(a, b) for a, b in zip((u.fast, u.mid, u.slow, u.tilt, u.offset), ref))
    # (b) a zero SD: the component's draws do not touch the traces; its term is absent from (L3')
    other = draw_units(np.random.default_rng(12), R, n)
    ok_off, notes = True, []
    for name, law, swap in (("fast", NoiseLaw(0.0, 0.017, 3e-4, 0.055, 0.04), dict(xi_fast=other.fast)),
                            ("mid", NoiseLaw(0.035, 0.0, 3e-4, 0.055, 0.04), dict(xi_mid=other.mid)),
                            ("slow", NoiseLaw(0.035, 0.017, 3e-4, 0.0, 0.04), dict(xi_slow=other.slow))):
        base = dict(xi_fast=u.fast, xi_mid=u.mid, xi_slow=u.slow)
        a = simulate_traces(law, DT, **base)
        base.update(swap)
        b = simulate_traces(law, DT, **base)
        k = np.array([0, 1, 15, 500])
        g_exp = (law.sigma_f ** 2 * (k == 0) + law.sigma_m ** 2 * np.exp(-k * DT / law.tau_m)
                 + law.sigma_s ** 2 * np.exp(-k * DT / law.tau_s))
        same = np.array_equal(a, b) and np.allclose(autocovariance(law, DT, k), g_exp, rtol=1e-15, atol=0)
        ok_off &= same
        notes.append("%s off: %s" % (name, same))
    # (c) one-component laws equal the component written out here
    err = max(float(np.max(np.abs(simulate_traces(WHITE, DT, u.fast, u.mid, u.slow) - WHITE.sigma_f * u.fast))),
              float(np.max(np.abs(simulate_traces(MID, DT, u.fast, u.mid, u.slow)
                                  - _ou_loop(MID.sigma_m, MID.tau_m, DT, u.mid)))),
              float(np.max(np.abs(simulate_traces(SLOW, DT, u.fast, u.mid, u.slow)
                                  - _ou_loop(SLOW.sigma_s, SLOW.tau_s, DT, u.slow)))))
    full = simulate_traces(THETA, DT, u.fast, u.mid, u.slow)
    err_sum = float(np.max(np.abs(full - (THETA.sigma_f * u.fast + _ou_loop(THETA.sigma_m, THETA.tau_m, DT, u.mid)
                                          + _ou_loop(THETA.sigma_s, THETA.tau_s, DT, u.slow)))))
    # 1e-14 mV: lfilter and the loop order their operations differently; the roundoff of the recursion
    # accumulates over the 400 samples to some tens of ulp of traces of about 0.2 mV (1 ulp = 2.8e-17 mV)
    ok_one = err <= 1e-14 and err_sum <= 1e-14
    report("M11 draw order fast, mid, slow, tilt, offset; a zero SD switches its component off; components = "
           "written-out recursions", ok_order and ok_off and ok_one,
           "order %s; %s; max |one-component - loop| %.1e mV, sum %.1e mV" % (ok_order, "; ".join(notes), err, err_sum))


def main():
    for fn in (_m1, _m2, _m3, _m4, _m5, _m6, _m7, _m8, _m9, _m10, _m11):
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
