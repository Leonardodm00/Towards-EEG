"""check_d095_rc_start.py -- standalone check (not pipeline code; not in the
repository), 2026-10-28: where the window start of D-095 would fall for a
noiseless RC pulse response, as a function of the membrane time constant and
the input resistance, under two readings of "the exponential decay".

D-095 (the user, 2026-10-28): t_SS(c) is the time at which the exponential
decay of the mean window's response falls within kappa_tr * sigma_SS_hat(c).
Read here in its simplest form (point values, no confidence band, no 80 ms
floor), on the noiseless mean window of the train's interior pulse (position
18) for an RC membrane driven by the SS train (37 alternating pulses of
+/-200 pA x 0.5 ms, 200.5 ms onset to onset):
  (pre)  the decay measured against the mean window's own pre-pulse level
         (t in [-10, 0) ms): the last sample with |w(t)| >= kappa sigma_SS;
  (fit)  the decay of a fitted single exponential b + A exp(-t / tau) on
         [5, 200) ms (scipy.optimize.curve_fit, baseline b free): the time
         tau_hat ln(|A_hat| / (kappa sigma_SS)).
Claim checked (tolerance fixed before running): every earlier pulse's tail
is the same exponential, so the train's window is exactly b + A exp(-t/tau)
with b = 0 and, the polarities alternating, A = V0 / (1 + exp(-P / tau_m))
for a long train (P = 200.5 ms, V0 = Q R_in / tau_m, Q = 0.1 pC); the (fit)
reading must agree with tau_m ln(A / (kappa sigma_SS)) within 0.5 ms.
Run 2 compared it with the single-pulse value tau_m ln(V0 / (kappa sigma_SS))
within 1 ms and FAILED at tau_m = 60 ms (229.0 against 230.8 ms; output kept
as d095_rc_start_2026-10-28_run2_failed.txt): the alternating tails lower A
by 3.5 % there, 2 ms of crossing time; the claim is corrected to the train's
amplitude. The (pre) reading is reported, no
pass/fail: its pre-pulse segment holds the previous, opposite-polarity
pulse's tail, which stays in w(t) as a constant offset.
Run 1 of this file compared the (pre) reading with the analytic value and
FAILED (output kept as d095_rc_start_2026-10-28_run1_failed.txt): that
comparison was the wrong expectation for (pre), and the failure is the finding.
sigma_SS_hat = 0.0402 mV (the median at theta_cf, check_sigma_ss.py).

Run (from this folder, next to check_findings_2026-10-23.py):
    python check_d095_rc_start.py      (seconds)
Pure ASCII, LF.
"""
import importlib.util
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("chk", HERE / "check_findings_2026-10-23.py")
chk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(chk)
import smoke_noise_windows as snw    # noqa: E402

CFG = dict(sigma_ss_mV=0.0402, kappa=0.1, q_pC=0.1, position=18,
           tau_ms=(20.0, 28.143, 35.0, 40.0, 45.0, 50.0, 60.0), r_fix_mohm=113.0,
           r_mohm=(60.0, 113.0, 200.0, 300.0), tau_fix_ms=28.143, tol_ms=0.5)
FS, PRE_N, N_W, P, T_TRAIN, L_SWEEP = chk.FS, chk.PRE_N, chk.N_W, chk.P, chk.T_TRAIN, chk.L_SWEEP
T_MS = (np.arange(N_W) - PRE_N) / FS * 1e3


def mean_window(r_mohm, tau_ms):
    """Noiseless window of the train's interior pulse (the mean window of its polarity)."""
    i = np.zeros(L_SWEEP)
    for k in range(T_TRAIN):
        s0 = k * P + PRE_N
        i[s0:s0 + chk.CFG["pulse_n"]] = chk.CFG["pulse_pA"] if k % 2 == 0 else -chk.CFG["pulse_pA"]
    v = snw.rc_response(i, r_mohm=r_mohm, tau_ms=tau_ms)
    return v[CFG["position"] * P + np.arange(N_W)]


def crossings(r_mohm, tau_ms):
    w = mean_window(r_mohm, tau_ms)
    thr = CFG["kappa"] * CFG["sigma_ss_mV"]
    # (pre): against the window's own pre-pulse level
    wp = w - w[:PRE_N].mean()
    above = np.flatnonzero((T_MS >= 0.5) & (np.abs(wp) >= thr))
    t_pre = T_MS[above[-1]] + 1e3 / FS if above.size else 0.0
    # (fit): single exponential with a free baseline on [5, 200) ms
    m = T_MS >= 5.0
    f = lambda t, b, a, tau: b + a * np.exp(-t / tau)
    with warnings.catch_warnings():                       # noiseless data: zero residual, no covariance
        warnings.simplefilter("ignore", OptimizeWarning)
        (b, a, tau_hat), _ = curve_fit(f, T_MS[m], w[m], p0=(0.0, w[m][0], tau_ms))
    t_fit = tau_hat * np.log(abs(a) / thr) if abs(a) > thr else 0.0
    v0 = CFG["q_pC"] * 1e-12 * r_mohm * 1e6 / (tau_ms * 1e-3) * 1e3        # mV
    a_train = v0 / (1.0 + np.exp(-P / FS * 1e3 / tau_ms))                   # alternating tails, long train
    t_an = tau_ms * np.log(a_train / thr) if a_train > thr else 0.0
    t_one = tau_ms * np.log(v0 / thr) if v0 > thr else 0.0
    return t_pre, t_fit, t_an, t_one, v0, b, tau_hat


def main():
    ok = True
    print("kappa_tr sigma_SS = %.4f mV; noiseless RC response, SS train position %d; crossing times in ms "
          "(pre: against the pre-pulse level; fit: fitted exponential, free baseline; train analytic and one pulse)"
          % (CFG["kappa"] * CFG["sigma_ss_mV"], CFG["position"]))
    cases = [(CFG["r_fix_mohm"], t) for t in CFG["tau_ms"]] + [(r, CFG["tau_fix_ms"]) for r in CFG["r_mohm"]]
    for r, tau in cases:
        t_pre, t_fit, t_an, t_one, v0, b, tau_hat = crossings(r, tau)
        good = abs(t_fit - t_an) <= CFG["tol_ms"]
        ok &= good
        print("R_in %5.0f MOhm, tau_m %5.1f ms: V0 %.3f mV | pre %6.1f | fit %6.1f (b %.1e mV, tau_hat %.2f ms) | "
              "train analytic %6.1f (one pulse %6.1f) | %s%s"
              % (r, tau, v0, t_pre, t_fit, b, tau_hat, t_an, t_one, "ok" if good else "FAIL",
                 "   [fit start at or past 190 ms: fallback]" if t_fit >= 190.0 else ""))
    print("fit reading against the train's analytic crossing within %.1f ms: %s" % (CFG["tol_ms"], "PASS" if ok else "FAIL"))
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
