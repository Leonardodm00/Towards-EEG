"""check_start_options.py -- standalone check (not pipeline code), 2026-10-29:
the SS window start of D-103 / D-104 against four variants, on the same
simulated cells as check_start_noise.py (its simulator, seed 20261042).

Variants (each gives t_star; then the 50 ms rule and the 80 ms floor and
fallback of (ST2) as implemented):
  V0  as implemented (Block 3): per group the decay B + A exp(-t / tau),
      the MAXIMUM over groups of t_sp;
  V1  the same fits, the MEDIAN over groups of t_sp;
  V2  per group the decay with a linear term, B + A exp(-t / tau) + beta t
      (the sweep's tilt inside the window), the maximum over groups;
  V3  V2's fits, the median over groups;
  V4  one fit per polarity on the CELL's median window (all sweeps pooled;
      departs from D-095's "per polarity and per sweep"), the maximum over
      the two polarities.
Each start uses t_sp = tau_hat ln(|A_hat| / (kappa_tr sigma_SS_hat)) (ST1).
Reported per layout and tilt: the median of t_star - t0 and the fallback
share (t0 the noiseless start at the cell's sigma_SS_hat, as in
check_start_noise.py). No pass/fail: a comparison for the user's decision.
[2026-10-29 (later, 5): V0, V1 and V4 use Block 3's fit, exact since D-110
(the output file is the run of f293d4a, three-start fits); V2, V3 keep their
own three-start fit with the linear term.]
Run (from this folder; about 10 min):  python check_start_options.py
Pure ASCII, LF.
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import check_start_noise as csn            # noqa: E402  (simulator, t0, the Ih Fit imports)

nw = csn.nw
CFG = dict(seed=20261042, n_cells=100, tau0_ms=(5.0, 20.0, 80.0), maxfev=2000)   # V2/V3's own fit settings
WCFG = nw.WindowConfig()


def fit_with_line(t, y, cfg):
    """B + A exp(-t / tau) + beta t, from the same three tau_0 starts (linear LS
    for A, B, beta at each), the converged start of smallest RSS."""
    best = None
    f = lambda tt, B, A, tau, beta: B + A * np.exp(-tt / tau) + beta * tt      # noqa: E731
    for tau0 in CFG["tau0_ms"]:
        basis = np.column_stack([np.exp(-t / tau0), np.ones_like(t), t])
        a0, b0, c0 = np.linalg.lstsq(basis, y, rcond=None)[0]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", OptimizeWarning)
                with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
                    p = curve_fit(f, t, y, p0=(b0, a0, tau0, c0), method="lm", maxfev=CFG["maxfev"])[0]
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            continue
        B, A, tau, beta = (float(x) for x in p)
        if not (np.all(np.isfinite(p)) and tau > 0):
            continue
        rss = float(np.sum((y - f(t, B, A, tau, beta)) ** 2))
        if best is None or rss < best[3]:
            best = (A, tau, B, rss)
    if best is None:
        nan = float("nan")
        return nw.DecayFit(nan, nan, nan, nan, False, False, "no start")
    return nw.DecayFit(best[0], best[1], best[2], best[3], True, False, "")


def t_star_of(t_sp, how):
    t_sp = t_sp[np.isfinite(t_sp)]
    if t_sp.size == 0:
        return float("nan")
    return max(WCFG.t_ss_floor_ms, float(np.max(t_sp) if how == "max" else np.median(t_sp)))


def falls_back(t_star):
    return (not np.isfinite(t_star)) or (200.0 - t_star < WCFG.min_window_ms)


def pooled_starts(ss, sigma_hat):
    t_ms = ss.t * 1e3
    sel = np.arange(ss.n_w) >= ss.pre_n + int(round(WCFG.fit_start_ms * 1e-3 * ss.fs))
    out = []
    for dep in (True, False):
        med = np.median(ss.v[ss.polarity_is_dep == dep], axis=0)
        f = nw.fit_decay(t_ms[sel], med[sel], WCFG)
        if f.ok:
            thr = WCFG.kappa_tr * sigma_hat
            out.append(f.tau_ms * np.log(abs(f.A_mV) / thr) if abs(f.A_mV) > thr else 0.0)
    return np.array(out)


def main():
    t_start = time.time()
    rng = np.random.default_rng(CFG["seed"])
    resp = csn.train_response()
    names = ("V0 max", "V1 median", "V2 line+max", "V3 line+median", "V4 pooled+max")
    print("rows: sigma_b, sweeps; per variant the median of t_star - t0 (ms) and the fallback share")
    for sb in csn.CFG["sigma_b"]:
        for S in csn.CFG["sweeps"]:
            dev = {n: [] for n in names}
            fall = {n: 0 for n in names}
            for c in range(CFG["n_cells"]):
                ss = csn.simulate_cell(S, sb, resp, rng)
                st0 = nw.decay_start(ss)
                st2 = nw.decay_start(ss, fit=fit_with_line)
                t0 = csn.t0_of(st0.sigma_ss_mV)
                ts = {"V0 max": t_star_of(st0.t_group_ms, "max"), "V1 median": t_star_of(st0.t_group_ms, "median"),
                      "V2 line+max": t_star_of(st2.t_group_ms, "max"),
                      "V3 line+median": t_star_of(st2.t_group_ms, "median"),
                      "V4 pooled+max": t_star_of(pooled_starts(ss, st0.sigma_ss_mV), "max")}
                for n in names:
                    if np.isfinite(ts[n]):
                        dev[n].append(ts[n] - t0)
                    fall[n] += int(falls_back(ts[n]))
            line = "sigma_b %.1f, %2d sweeps:" % (sb, S)
            for n in names:
                line += "  %s %+6.1f / %3.0f %%" % (n, float(np.median(dev[n])), 100.0 * fall[n] / CFG["n_cells"])
            print(line + "   (%.0f s)" % (time.time() - t_start))
    print("noiseless t0 at sigma_SS_hat 0.0402 mV: %.1f ms; the window keeps >= 50 ms only for t_star <= 150 ms"
          % csn.t0_of(0.0402))


if __name__ == "__main__":
    main()
