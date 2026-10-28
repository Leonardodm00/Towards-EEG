"""check_per_window_fit.py -- standalone check (not pipeline code): does a
per-window fit of the pulse response, on top of the mean-window subtraction
(D-051.1), remove pulse-to-pulse response variability without removing noise?

Simulated cells at the archive's SS layout (2 sweeps x 37 alternating pulses,
the fixture's 11 missing pulses, K = 63), noise theta_cf, an RC response
(113 MOhm, 28.1 ms, +/-200 pA x 0.5 ms) whose amplitude varies from pulse to
pulse by a factor a_k ~ N(1, cv^2). The five SS statistics (sigma_hat, rho_hat
at 0.02, 0.1, 1, 10 ms) on [80, 200) ms, per-cell medians, are compared with
the same statistics of the same noise draws without any response, after the
plain subtraction (what the fit's simulator produces). Methods:
  plain     : window minus the polarity's mean window (D-051.1)
  gain_late : plus, per window, the least-squares gain a_k of the mean window
              fitted on [80, 200) ms itself, residual x - a_k m
  gain_early: the gain fitted on [0, 80) ms (where the response is large),
              applied on [80, 200) ms
  shift     : the time shift (|d| <= 0.5 ms) maximising the correlation of the
              window with the mean window over [0, 200) ms, residual x(t) - m(t - d)
Bias is reported in units of the per-cell SD of each statistic.

Run:  python check_per_window_fit.py      (about 2 min)
"""
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("chk", HERE / "check_findings_2026-10-23.py")
chk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chk)          # reuses its layout, response and noise simulators
import noise_cell_stats as ncs        # noqa: E402  (path set by chk)

CFG = dict(seed=20261026, n_rep=200, cv_list=(0.0, 0.05, 0.20), t_ss_ms=80.0, early_ms=(0.0, 80.0),
           max_shift_ms=0.5, layout="2 sweeps, K 63")
FS, PRE_N, N_W = chk.FS, chk.PRE_N, chk.N_W
J_SS = PRE_N + int(round(CFG["t_ss_ms"] * 1e-3 * FS))
J_E0 = PRE_N + int(round(CFG["early_ms"][0] * 1e-3 * FS))
J_E1 = PRE_N + int(round(CFG["early_ms"][1] * 1e-3 * FS))
MAXS = int(round(CFG["max_shift_ms"] * 1e-3 * FS))


def polarity_means(V, dep):
    M = np.empty_like(V)
    for mask in (dep, ~dep):
        M[mask] = V[mask].mean(axis=0)
    return M


def residuals(V, dep, method):
    """V (K, n_w) full windows; returns residual windows on [t_SS, 200) ms."""
    M = polarity_means(V, dep)
    if method == "plain":
        return (V - M)[:, J_SS:]
    if method in ("gain_late", "gain_early"):
        sl = slice(J_SS, N_W) if method == "gain_late" else slice(J_E0, J_E1)
        x = V[:, sl] - V[:, sl].mean(axis=1, keepdims=True)
        m = M[:, sl] - M[:, sl].mean(axis=1, keepdims=True)
        a = np.sum(x * m, axis=1) / np.sum(m * m, axis=1)
        return V[:, J_SS:] - a[:, None] * M[:, J_SS:]
    if method == "shift":
        out = np.empty((V.shape[0], N_W - J_SS))
        lo, hi = PRE_N + MAXS, N_W - MAXS
        for k in range(V.shape[0]):
            best, bd = -np.inf, 0
            x = V[k, lo:hi] - V[k, lo:hi].mean()
            for d in range(-MAXS, MAXS + 1):
                m = M[k, lo - d:hi - d]
                c = np.dot(x, m - m.mean())
                if c > best:
                    best, bd = c, d
            shifted = np.roll(M[k], bd)          # m(t - d); the wrapped samples lie outside [t_SS, 200)
            out[k] = V[k, J_SS:] - shifted[J_SS:]
        return out
    raise ValueError(method)


def cell_stats(R):
    return np.median(ncs.ss_window_stats(R, FS), axis=0)


def main():
    t0 = time.time()
    rng = np.random.default_rng(CFG["seed"])
    sweep, pos = chk.build_layout(chk.CFG["layouts"][CFG["layout"]], rng)
    dep = pos % 2 == 0
    RW = chk.response_windows(pos)
    methods = ("plain", "gain_late", "gain_early", "shift")
    truth, res = [], {(cv, m): [] for cv in CFG["cv_list"] for m in methods}
    for _ in range(CFG["n_rep"]):
        noise = chk.simulate_noise_windows(0.0, sweep, pos, rng)
        truth.append(cell_stats(residuals(noise, dep, "plain")))
        for cv in CFG["cv_list"]:
            a = 1.0 + cv * rng.standard_normal(pos.size)
            V = noise + a[:, None] * RW
            for m in methods:
                res[(cv, m)].append(cell_stats(residuals(V, dep, m)))
    truth = np.array(truth)
    sd = truth.std(axis=0, ddof=1)
    labels = ("sd", "rho0.02", "rho0.1", "rho1", "rho10")
    print("K = %d, t_SS = %.0f ms, %d cells, theta_cf; bias = mean(stat - noise-only plain stat) / per-cell SD"
          % (pos.size, CFG["t_ss_ms"], CFG["n_rep"]))
    print("%-6s %-11s " % ("cv", "method") + " ".join("%9s" % l for l in labels))
    for cv in CFG["cv_list"]:
        for m in methods:
            b = (np.array(res[(cv, m)]) - truth).mean(axis=0) / sd
            print("%-6.2f %-11s " % (cv, m) + " ".join("%9.3f" % x for x in b))
    print("(%.0f s)" % (time.time() - t0))


if __name__ == "__main__":
    main()
