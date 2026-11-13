"""check02_window_means.py -- standalone check behind TEEG_semivariogram_2026-11-13.md
(sections 3.5-3.6). Not pipeline code: nothing in the Ih Fit pipeline imports it.

Configuration: the project's Short Square (SS) layout and the law theta_cf.
  fs = 50 kHz (dt = 20 us). One SS sweep is a train of 37 pulses, onset to onset
  P = 10025 samples (200.5 ms), polarity alternating from depolarising, so two
  windows of the same polarity lie an even number L of periods apart. The noise
  window of pulse k is [80, 200) ms after its onset: m = 6000 samples, T = 120 ms.
  The trace is law (L1) of the SPEC: a fast AR(1) and a slow OU at
  theta_cf = (sigma_f, rho_f, sigma_s, tau_s) = (0.0380 mV, 0.175, 0.0508 mV, 36.6 ms),
  plus a per-sweep offset a ~ N(0, 1 mV^2) and a per-sweep tilt b (t - t_bar),
  b ~ N(0, sigma_b^2), sigma_b = 0.1 mV/s. All windows kept (no trace-QC drops), no
  pulse response. The simulator here is independent of the repository's
  (scipy.signal.lfilter AR(1) recursions, stationary starts); the repository's
  noise_law (Block 2) supplies the predictions.

Claims tested; predictions and tolerances fixed before the first run:
 (a) at every same-polarity lag L = 2, 4, ..., 36 periods (h = L P dt = 0.401 ...
     7.218 s), the Matheron estimate over the same-sweep pairs,
       V_hat(h) = (1 / (2 N)) sum_k (xbar_{k+L} - xbar_k)^2,
     agrees with noise_law.semivariogram(theta_cf, dt, m, m, L P, sigma_b) within
     3.5 SE (SE from the spread of the per-sweep estimates across sweeps), with the
     tilt and without it (common random numbers: the same traces minus the tilt).
 (b) the stationary floor V(h | theta_cf, 0) is flat over these lags: relative
     spread over L = 2..36 below 1e-3, and equal to Var(xbar) of one 120 ms window
     mean (noise_law.var_window_mean) within 1e-3 relative; at h* = 4 periods it is
     1.1125e-3 mV^2 within 1e-7 mV^2 (the tilt study's value).
 (c) the tilt term: V(h* | theta_cf, sigma_b) - V(h* | theta_cf, 0) = sigma_b^2 h*^2 / 2
     = 3.21602e-3 mV^2 at h* = 0.802 s (relative difference below 1e-12); and, on
     the simulated sweeps, (D-034.1) with the true theta,
       2 [V_hat(h*) - V(h* | theta_cf, 0)] / h*^2,
     agrees with the realised mean square of the sweeps' tilts within 3.5 SE.
 (d) Allan variance: on continuous stationary traces at theta_cf, one half the
     mean squared difference of consecutive, non-overlapping 120 ms averages
     agrees within 3.5 SE with noise_law.semivariogram(theta_cf, dt, m, m, m),
     the window-mean semivariogram of two adjacent windows (h = T).
 (e) shapes, formula evaluations only: the excess of a seconds-scale OU sized as in
     the topic doc's section 3.6.3 (3 s, 0.142 mV; 30 s, 0.294 mV), computed exactly
     for the sampled law, agrees with the continuous-time eq. (25) within 1e-3
     relative; its log-log slope between consecutive same-polarity lags lies in
     [0.2, 1.15] for both; the point slope (central difference, dh = 1e-4 s, of the
     continuous form) is 1.047 and 1.104 at h = 0.401 s and 0.242 and 0.890 at
     h = 7.218 s (3 s and 30 s), each within 0.005; the tilt term's slope is 2
     (1e-9).
 (f) the averaging shift (added after the first run of (a)-(e), its prediction
     fixed before its own first run): for T << h << tau the OU excess is close to
     sigma^2 (h - T/3) / tau, so for an OU of 3000 s the point slope at h = 0.401 s
     lies within [1.109, 1.113] (the limit h / (h - T/3) = 1.111).
Seed 20261114. Writes ../results/check02_window_means.json (read by
fig02_window_mean_semivariogram.py).

Run:  python check02_window_means.py        (about 1 min, about 0.5 GB of memory)
Repository path: environment variable TEEG_IH_FIT, else the Ih Fit folder three
levels above this file.

Pure ASCII, LF.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CFG = dict(
    seed=20261114,
    z_tol=3.5,
    fs=50000.0,                                 # Hz
    period=10025,                               # samples onset to onset (200.5 ms)
    train=37,                                   # pulses per SS sweep
    win_start=4000, win_end=10000,              # samples after the onset: [80, 200) ms
    theta_cf=(0.0380, 0.175, 0.0508, 0.0366),   # sigma_f mV, rho_f, sigma_s mV, tau_s s
    sigma_b=0.1,                                # mV/s
    offset_sd=1.0,                              # mV
    n_sweeps=400, batch=20,
    lag_star=4,                                 # periods (h* = 0.802 s)
    floor_ref=1.1125e-3, floor_tol=1e-7,        # mV^2 (tilt study, check03)
    av_traces=60, av_bins=100,                  # Allan variance traces
    ou_seconds=((3.0, 0.142), (30.0, 0.294)),   # (tau_3 s, sigma_3 mV), topic doc 3.6.3
    slope_band=(0.2, 1.15),
    point_slopes={3.0: (1.047, 0.242), 30.0: (1.104, 0.890)},   # at h = 0.401 s and 7.218 s
    point_slope_tol=0.005,
    tau_long=3000.0, slope_long_band=(1.109, 1.113),            # (f), added 2026-11-13 before its first run
)
REPO_IH_FIT = Path(os.environ.get("TEEG_IH_FIT", str(Path(__file__).resolve().parents[3])))
OUT = Path(__file__).resolve().parent.parent / "results" / "check02_window_means.json"

sys.path.insert(0, str(REPO_IH_FIT))
import noise_law as nl   # noqa: E402

DT = 1.0 / CFG["fs"]
P = CFG["period"]
M = CFG["win_end"] - CFG["win_start"]
N_SWEEP = (CFG["train"] - 1) * P + CFG["win_end"]          # samples from the first onset
LAW = nl.NoiseLaw(*CFG["theta_cf"])
LAGS = np.arange(2, CFG["train"], 2)                         # even lags in periods: 2..36


# ---------------------------------------------------------------------------
# Simulation (independent of the repository's simulator)
# ---------------------------------------------------------------------------
def ar1_rows(rng: np.random.Generator, rows: int, n: int, sigma: float, rho: float) -> np.ndarray:
    """rows x n stationary AR(1): x_0 ~ N(0, sigma^2), x_j = rho x_{j-1} + e_j."""
    e = rng.standard_normal((rows, n))
    e[:, 0] *= sigma
    e[:, 1:] *= sigma * np.sqrt(1.0 - rho ** 2)
    return lfilter([1.0], [1.0, -rho], e, axis=1)


def stationary_traces(rng: np.random.Generator, rows: int, n: int) -> np.ndarray:
    fast = ar1_rows(rng, rows, n, LAW.sigma_f, LAW.rho_f)
    slow = ar1_rows(rng, rows, n, LAW.sigma_s, LAW.rho_s(DT))
    return fast + slow


def window_means(x: np.ndarray) -> np.ndarray:
    """(rows, 37) means of the noise windows [k P + 4000, k P + 10000)."""
    return np.stack([x[:, k * P + CFG["win_start"]: k * P + CFG["win_end"]].mean(axis=1)
                     for k in range(CFG["train"])], axis=1)


# ---------------------------------------------------------------------------
# Estimators (analysis)
# ---------------------------------------------------------------------------
def per_sweep_semivariogram(wm: np.ndarray, L: int) -> np.ndarray:
    """Per sweep: (1 / (2 N_s)) sum over same-sweep pairs L positions apart."""
    d = wm[:, L:] - wm[:, :-L]
    return 0.5 * np.mean(d ** 2, axis=1)


def mean_se(v: np.ndarray):
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(v.size))


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
RESULTS = []


def report(label: str, ok: bool, detail: str) -> None:
    RESULTS.append(bool(ok))
    print("%-58s %s  %s" % (label, detail, "PASS" if ok else "FAIL"))


def main() -> int:
    t0 = time.time()
    c = CFG
    rng = np.random.default_rng(c["seed"])
    t = np.arange(N_SWEEP) * DT
    t_bar = 0.5 * (N_SWEEP - 1) * DT

    # ---- simulate the sweeps (common random numbers: with and without the tilt)
    S, B = c["n_sweeps"], c["batch"]
    wm0 = np.empty((S, c["train"]))
    wm1 = np.empty((S, c["train"]))
    b_all = np.empty(S)
    for i0 in range(0, S, B):
        rows = min(B, S - i0)
        x = stationary_traces(rng, rows, N_SWEEP)
        a = c["offset_sd"] * rng.standard_normal((rows, 1))
        b = c["sigma_b"] * rng.standard_normal((rows, 1))
        x += a
        wm0[i0:i0 + rows] = window_means(x)
        x += b * (t - t_bar)[None, :]
        wm1[i0:i0 + rows] = window_means(x)
        b_all[i0:i0 + rows] = b[:, 0]
        del x
    print("simulated %d sweeps of %d samples (%.1f s)" % (S, N_SWEEP, time.time() - t0))

    # ---- (a) estimator against the exact window-mean semivariogram, every even lag
    h = LAGS * P * DT
    rows_out = []
    zmax0 = zmax1 = 0.0
    for L, hh in zip(LAGS, h):
        v0, se0 = mean_se(per_sweep_semivariogram(wm0, L))
        v1, se1 = mean_se(per_sweep_semivariogram(wm1, L))
        p0 = nl.semivariogram(LAW, DT, M, M, int(L) * P, 0.0)
        p1 = nl.semivariogram(LAW, DT, M, M, int(L) * P, c["sigma_b"])
        z0, z1 = (v0 - p0) / se0, (v1 - p1) / se1
        zmax0, zmax1 = max(zmax0, abs(z0)), max(zmax1, abs(z1))
        rows_out.append(dict(L=int(L), h_s=float(hh), pairs_per_sweep=int(c["train"] - L),
                             V_hat_notilt=v0, se_notilt=se0, V_notilt=p0,
                             V_hat_tilt=v1, se_tilt=se1, V_tilt=p1))
        print("   L = %2d (h = %.3f s, %2d pairs/sweep): no tilt %.4e +/- %.1e (pred %.4e, z %+5.2f) | "
              "tilt %.4e +/- %.1e (pred %.4e, z %+5.2f)" % (L, hh, c["train"] - L, v0, se0, p0, z0, v1, se1, p1, z1))
    report("(a) V_hat = V(h | theta_cf, 0) at L = 2..36", zmax0 < c["z_tol"], "max |z| %.2f" % zmax0)
    report("(a) V_hat = V(h | theta_cf, 0.1 mV/s) at L = 2..36", zmax1 < c["z_tol"], "max |z| %.2f" % zmax1)

    # ---- (b) the floor
    floor = np.array([r["V_notilt"] for r in rows_out])
    vwm = nl.var_window_mean(LAW, DT, M)
    spread = float((floor.max() - floor.min()) / floor.mean())
    dev = float(np.max(np.abs(floor - vwm)) / vwm)
    floor_star = float(floor[LAGS == c["lag_star"]][0])
    report("(b) floor flat over L = 2..36 (relative spread)", spread < 1e-3, "%.2e" % spread)
    report("(b) floor = Var(xbar) of one 120 ms window", dev < 1e-3,
           "Var(xbar) %.5e mV^2, largest relative deviation %.2e" % (vwm, dev))
    report("(b) floor at h* = 1.1125e-3 mV^2", abs(floor_star - c["floor_ref"]) < c["floor_tol"],
           "%.6e mV^2" % floor_star)
    fast_share = nl.var_window_mean(nl.NoiseLaw(LAW.sigma_f, LAW.rho_f, 0.0, LAW.tau_s), DT, M) / vwm
    print("   share of the fast AR(1) in Var(xbar): %.2e" % fast_share)

    # ---- (c) the tilt term and (D-034.1) on the simulated sweeps
    h_star = c["lag_star"] * P * DT
    tilt_term = nl.semivariogram(LAW, DT, M, M, c["lag_star"] * P, c["sigma_b"]) - floor_star
    pred_tilt = 0.5 * c["sigma_b"] ** 2 * h_star ** 2
    report("(c) tilt term at h* = sigma_b^2 h*^2 / 2", abs(tilt_term / pred_tilt - 1) < 1e-12,
           "%.6e mV^2 (h* = %.3f s; ratio to the floor %.2f)" % (tilt_term, h_star, tilt_term / floor_star))
    v_s = per_sweep_semivariogram(wm1, c["lag_star"])
    s2_s = 2.0 * (v_s - floor_star) / h_star ** 2                 # per-sweep (D-034.1), true theta
    diff = s2_s - b_all ** 2
    m_d, se_d = mean_se(diff)
    s2_hat = float(s2_s.mean())
    msq = float(np.mean(b_all ** 2))
    report("(c) (D-034.1) = realised mean square of the tilts", abs(m_d / se_d) < c["z_tol"],
           "%.5f (mV/s)^2 against %.5f, difference %+.5f +/- %.5f" % (s2_hat, msq, m_d, se_d))

    # ---- (d) Allan variance = semivariogram of adjacent windows
    nb = c["av_bins"]
    av_parts = []
    for i0 in range(0, c["av_traces"], 10):                 # 10 traces of nb * M samples at a time
        rows = min(10, c["av_traces"] - i0)
        xa = stationary_traces(rng, rows, nb * M)
        bins = xa.reshape(rows, nb, M).mean(axis=2)          # consecutive, non-overlapping T-averages
        av_parts.append(0.5 * np.mean(np.diff(bins, axis=1) ** 2, axis=1))
        del xa
    av = np.concatenate(av_parts)
    av_m, av_se = mean_se(av)
    av_pred = nl.semivariogram(LAW, DT, M, M, M, 0.0)
    report("(d) Allan variance at T = 120 ms = V(h = T)", abs(av_m - av_pred) / av_se < c["z_tol"],
           "%.5e +/- %.1e against %.5e mV^2 (z %+.2f)" % (av_m, av_se, av_pred, (av_m - av_pred) / av_se))

    # ---- (e) shapes: seconds-scale OU excess against the tilt
    shapes = {}
    T = M * DT
    for tau3, sig3 in c["ou_seconds"]:
        ou = nl.NoiseLaw(0.0, 0.5, sig3, tau3)          # sigma_f = 0: the OU alone
        exact = np.array([nl.semivariogram(ou, DT, M, M, int(L) * P) for L in LAGS])
        cont = np.array([nl.semivariogram_continuous(ou, DT, T, T, float(hh)) for hh in h])
        rel = float(np.max(np.abs(exact / cont - 1)))
        report("(e) OU %g s: exact = continuous eq. (25)" % tau3, rel < 1e-3, "largest relative difference %.1e" % rel)
        sl = np.diff(np.log(exact)) / np.diff(np.log(h))
        lo, hi = c["slope_band"]
        report("(e) OU %g s: log-log slopes in [%.2f, %.2f]" % (tau3, lo, hi), bool(np.all((sl >= lo) & (sl <= hi))),
               "%.3f ... %.3f" % (sl.min(), sl.max()))

        def point_slope(hh, d=1e-4):
            vp = nl.semivariogram_continuous(ou, DT, T, T, hh + d)
            vm = nl.semivariogram_continuous(ou, DT, T, T, hh - d)
            v = nl.semivariogram_continuous(ou, DT, T, T, hh)
            return (vp - vm) / (2 * d) * hh / v
        ps = (point_slope(float(h[0])), point_slope(float(h[-1])))
        want = c["point_slopes"][tau3]
        ok = all(abs(p - w) < c["point_slope_tol"] for p, w in zip(ps, want))
        report("(e) OU %g s: point slopes at 0.401 / 7.218 s" % tau3, ok,
               "%.3f / %.3f (predicted %.3f / %.3f)" % (ps[0], ps[1], want[0], want[1]))
        shapes["OU_%gs" % tau3] = dict(tau_s=tau3, sigma_mV=sig3, excess=exact.tolist(),
                                        slopes=sl.tolist(), point_slopes=list(ps))
    # (f) the averaging shift: for T << h << tau the excess is ~ sigma^2 (h - T/3) / tau,
    #     so the point slope at h = 0.401 s tends to h / (h - T/3) = 1.111 as tau grows
    ou_long = nl.NoiseLaw(0.0, 0.5, 1.0, c["tau_long"])

    def point_slope_long(hh, d=1e-4):
        vp = nl.semivariogram_continuous(ou_long, DT, T, T, hh + d)
        vm = nl.semivariogram_continuous(ou_long, DT, T, T, hh - d)
        return (vp - vm) / (2 * d) * hh / nl.semivariogram_continuous(ou_long, DT, T, T, hh)
    ps_long = point_slope_long(float(h[0]))
    lim = float(h[0] / (h[0] - T / 3.0))
    lo, hi = c["slope_long_band"]
    report("(f) OU %g s: point slope at 0.401 s -> h / (h - T/3)" % c["tau_long"], lo <= ps_long <= hi,
           "%.4f (limit %.4f; band [%.3f, %.3f])" % (ps_long, lim, lo, hi))
    shapes["OU_long"] = dict(tau_s=c["tau_long"], point_slope=ps_long, limit=lim)

    tilt_ex = np.array([nl.semivariogram(LAW, DT, M, M, int(L) * P, c["sigma_b"]) - nl.semivariogram(LAW, DT, M, M, int(L) * P)
                        for L in LAGS])
    sl_t = np.diff(np.log(tilt_ex)) / np.diff(np.log(h))
    report("(e) tilt term: log-log slope 2", bool(np.max(np.abs(sl_t - 2.0)) < 1e-9), "max |slope - 2| %.1e" % np.max(np.abs(sl_t - 2.0)))
    shapes["tilt"] = dict(sigma_b=c["sigma_b"], excess=tilt_ex.tolist())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(config={k: (list(v) if isinstance(v, tuple) else v) for k, v in CFG.items()
                                           if k != "point_slopes"},
                                   dt=DT, m=M, n_sweep=N_SWEEP, lags=rows_out, floor_var_window_mean=vwm,
                                   allan=dict(T_s=T, mean=av_m, se=av_se, predicted=av_pred),
                                   d0341=dict(estimate=s2_hat, realised_mean_square=msq, diff=m_d, se=se_d),
                                   shapes=shapes), indent=1))
    print("\nwrote %s" % OUT)
    print("ALL %s   (%d checks, %.1f s)" % ("PASS" if all(RESULTS) else "FAIL", len(RESULTS), time.time() - t0))
    return 0 if all(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
