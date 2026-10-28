"""check_fallback_window.py -- standalone check (not pipeline code), 2026-10-28:
where D-104's fallback starts to apply on the RC test cell.

Rule (D-103, D-104): t_SS(c) = max(80 ms, max t_sp), with t_sp the point value
tau_hat ln(|A_hat| / (kappa_tr sigma_SS_hat(c))) of a single exponential with a
free baseline fitted on [5, 200) ms; when the window [t_SS(c), 200) ms would
be shorter than 50 ms -- a start later than 150 ms -- the start falls back to
80 ms (flagged). Computed here on the noiseless RC test cell of
check_d095_rc_start.py (SS train of 37 alternating pulses, +/-200 pA x 0.5 ms,
200.5 ms onset to onset; sigma_SS_hat = 0.0402 mV): for kappa_tr in
(0.05, 0.1, 0.2), the membrane time constant tau_m (at R_in = 113 MOhm) and
the input resistance R_in (at tau_m = 28.143 ms) from which the start passes
150 ms, by brentq on the train's analytic start tau_m ln(A / (kappa_tr
sigma_SS)), A = V0 / (1 + exp(-P / tau_m)), V0 = Q R_in / tau_m.

Prediction, fixed before running:
 P1  at each parameter found, the fitted crossing (curve_fit, as in
     check_d095_rc_start.py) lies within 0.5 ms of 150 ms.

Run (from this folder, next to check_d095_rc_start.py):
    python check_fallback_window.py      (seconds; numpy, scipy)
Pure ASCII, LF.
"""
import importlib.util
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("rcs", HERE / "check_d095_rc_start.py")
rcs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rcs)

CFG = dict(kappas=(0.05, 0.1, 0.2), r_fix_mohm=113.0, tau_fix_ms=28.143, window_end_ms=200.0,
           min_window_ms=50.0, tol_ms=0.5)


def analytic_start(kappa, r_mohm, tau_ms):
    thr = kappa * rcs.CFG["sigma_ss_mV"]
    v0 = rcs.CFG["q_pC"] * 1e-12 * r_mohm * 1e6 / (tau_ms * 1e-3) * 1e3            # mV
    a = v0 / (1.0 + np.exp(-rcs.P / rcs.FS * 1e3 / tau_ms))                         # mV
    return tau_ms * np.log(a / thr) if a > thr else 0.0


def fitted_start(kappa, r_mohm, tau_ms):
    rcs.CFG["kappa"] = kappa
    return rcs.crossings(r_mohm, tau_ms)[1]


def main():
    t_lim = CFG["window_end_ms"] - CFG["min_window_ms"]
    ok = True
    print("fallback when the window [t_SS, %.0f) ms is shorter than %.0f ms, i.e. a start later than %.0f ms; "
          "sigma_SS_hat = %.4f mV" % (CFG["window_end_ms"], CFG["min_window_ms"], t_lim, rcs.CFG["sigma_ss_mV"]))
    for k in CFG["kappas"]:
        tau_star = brentq(lambda t: analytic_start(k, CFG["r_fix_mohm"], t) - t_lim, 5.0, 200.0)
        r_star = brentq(lambda r: analytic_start(k, r, CFG["tau_fix_ms"]) - t_lim, 1.0, 5000.0)
        f_tau = fitted_start(k, CFG["r_fix_mohm"], tau_star)
        f_r = fitted_start(k, r_star, CFG["tau_fix_ms"])
        good = abs(f_tau - t_lim) <= CFG["tol_ms"] and abs(f_r - t_lim) <= CFG["tol_ms"]
        ok &= good
        print("P1 kappa_tr %.2f: fallback from tau_m = %.1f ms at %.0f MOhm (fitted start %.1f ms), "
              "or from R_in = %.0f MOhm at tau_m %.1f ms (fitted start %.1f ms) %s"
              % (k, tau_star, CFG["r_fix_mohm"], f_tau, r_star, CFG["tau_fix_ms"], f_r, "PASS" if good else "FAIL"))
    print("ALL", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
