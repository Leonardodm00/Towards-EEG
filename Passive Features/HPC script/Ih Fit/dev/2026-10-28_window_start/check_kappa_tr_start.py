"""check_kappa_tr_start.py -- standalone check (not pipeline code), 2026-10-28:
what kappa_tr does to the SS window start of D-095 as settled by D-102.

Rule (D-102): for each group (s, p) a single exponential B + A exp(-t / tau)
with a free baseline is fitted to the median window on [5, 200) ms, and the
group's start is the point value
    t_sp = tau_hat ln(|A_hat| / (kappa_tr sigma_SS_hat(c)))
(0 when |A_hat| <= kappa_tr sigma_SS_hat(c)); t_SS(c) = max(80 ms, max t_sp).
Here on the noiseless RC test cell of check_d095_rc_start.py (SS train of
37 alternating pulses, +/-200 pA x 0.5 ms, 200.5 ms onset to onset; a
noiseless window, so its median over windows equals the window itself),
sigma_SS_hat = 0.0402 mV, kappa_tr in (0.05, 0.1, 0.2).

Predictions, fixed before running:
 P1  at 113 MOhm, tau_m = 28.143 ms, the fitted crossing equals the train's
     analytic value tau_m ln(A / (kappa sigma_SS)), A = V0 / (1 + exp(-P / tau_m)),
     within 0.5 ms at every kappa_tr;
 P2  halving kappa_tr moves the fitted crossing later by tau_m ln 2 (19.51 ms)
     within 1 ms;
 P3  the tau_m at which the analytic start reaches 190 ms (113 MOhm; found by
     brentq) gives a fitted crossing within 0.5 ms of 190 ms.

Run (from this folder, next to check_d095_rc_start.py):
    python check_kappa_tr_start.py      (seconds; numpy, scipy)
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

CFG = dict(kappas=(0.05, 0.1, 0.2), r_mohm=113.0, tau_ms=28.143, t_limit_ms=190.0, tol_ms=0.5, tol_shift_ms=1.0)


def analytic_start(kappa, r_mohm, tau_ms):
    thr = kappa * rcs.CFG["sigma_ss_mV"]
    v0 = rcs.CFG["q_pC"] * 1e-12 * r_mohm * 1e6 / (tau_ms * 1e-3) * 1e3            # mV
    a = v0 / (1.0 + np.exp(-rcs.P / rcs.FS * 1e3 / tau_ms))                         # mV, alternating tails
    return tau_ms * np.log(a / thr) if a > thr else 0.0, a


def fitted_start(kappa, r_mohm, tau_ms):
    rcs.CFG["kappa"] = kappa                       # crossings() reads its threshold from the module's CFG
    return rcs.crossings(r_mohm, tau_ms)[1]


def main():
    ok, fit = True, {}
    print("sigma_SS_hat = %.4f mV; RC test cell %.0f MOhm; crossing = start of the noise window before the 80 ms floor"
          % (rcs.CFG["sigma_ss_mV"], CFG["r_mohm"]))
    for k in CFG["kappas"]:
        t_an, a = analytic_start(k, CFG["r_mohm"], CFG["tau_ms"])
        fit[k] = fitted_start(k, CFG["r_mohm"], CFG["tau_ms"])
        good = abs(fit[k] - t_an) <= CFG["tol_ms"]
        ok &= good
        print("P1 kappa_tr %.2f (tolerance %.4f mV): tau_m %.1f ms, |A| %.4f mV -> fitted %.1f ms, analytic %.1f ms %s"
              % (k, k * rcs.CFG["sigma_ss_mV"], CFG["tau_ms"], a, fit[k], t_an, "PASS" if good else "FAIL"))
    for k_lo, k_hi in ((0.05, 0.1), (0.1, 0.2)):
        shift = fit[k_lo] - fit[k_hi]
        good = abs(shift - CFG["tau_ms"] * np.log(2.0)) <= CFG["tol_shift_ms"]
        ok &= good
        print("P2 kappa_tr %.2f -> %.2f: start later by %.2f ms (tau_m ln 2 = %.2f ms) %s"
              % (k_hi, k_lo, shift, CFG["tau_ms"] * np.log(2.0), "PASS" if good else "FAIL"))
    for k in CFG["kappas"]:
        tau_star = brentq(lambda t: analytic_start(k, CFG["r_mohm"], t)[0] - CFG["t_limit_ms"], 5.0, 200.0)
        t_f = fitted_start(k, CFG["r_mohm"], tau_star)
        good = abs(t_f - CFG["t_limit_ms"]) <= CFG["tol_ms"]
        ok &= good
        print("P3 kappa_tr %.2f: the start reaches %.0f ms at tau_m = %.1f ms (fitted %.1f ms) %s"
              % (k, CFG["t_limit_ms"], tau_star, t_f, "PASS" if good else "FAIL"))
    print("ALL", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
