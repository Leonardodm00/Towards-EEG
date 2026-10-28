"""check02_lag_choice.py -- what (T1) reads when the variance at seconds is
not a tilt, and which lag h* makes the twins' tilt carry the Long Square
step's baseline error. Formulas only (continuous time), no random numbers.

Setting. The variance at seconds is taken to be an OU component of SD
sigma_3 and correlation time tau_3 (3, 10 or 30 s; sigma_3 = 0.142, 0.189,
0.294 mV sized in the topic doc v3 section 3.6.3 to the cohort's 6.89 s
change). The twin carries a tilt b ~ N(0, sigma_b^2) instead.
 * Step match (the D-034 note of 2026-10-08): the Long Square loss reads the
   baseline-subtracted error over the 1 s step, t in [0, 1] s after onset,
   with the baseline the mean of the 265 ms pre-window [-0.270, -0.005] s.
   For the OU: B3(t)^2 = sigma^2 + Var(pre mean) - 2 Cov(e(t), pre mean);
   for the tilt: (sigma_b (t + 0.1375 s))^2. sigma_b(match) equates the two
   averages over the step.
 * What (T1) reads at lag h with SS windows of duration T: the OU's excess
   semivariogram V3(h) = Var_T - Cov_T(h) (topic doc eq. 25), so
   sigma_b(read)^2 = 2 V3(h) / h^2.
Reported: the ratio sigma_b(read at h) / sigma_b(match) for T = 190 ms
(t_SS = 10 ms, the topic doc's case) and T = 120 ms (t_SS = 80 ms, D-076).

Predictions, fixed before running: (a) the T = 190 ms numbers of the topic
doc v3 section 3.6.5 are reproduced: sigma_b(match) 0.083-0.119 mV/s and
sigma_b(read at 0.8 s) 0.081-0.116 mV/s over the three OUs (to the 3
digits quoted); point-error ratios 0.47-0.49 at onset, 1.20-1.23 at the end;
(b) the closed forms agree with noise_law.semivariogram_continuous (for V3)
and with numerical quadrature (for B3^2) to 1e-9 relative.
Not predicted (the finding): the ratio at h* = 0.802 s for T = 120 ms.

Run:  python check02_lag_choice.py    (seconds)
"""
import numpy as np
from scipy import integrate

import tilt_common as tc

CFG = dict(
    ous=((3.0, 0.142), (10.0, 0.189), (30.0, 0.294)),   # (tau_3 s, sigma_3 mV)
    pre=(-0.270, -0.005),                              # s, LS pre-window relative to onset
    step=(0.0, 1.0),                                   # s, the scored step
    T_list=(0.190, 0.120),                             # s, SS window durations
    h_doc=(0.6, 0.8, 1.0),                             # s, the lags quoted in the topic doc
    lags_periods=(2, 4, 6, 8),                         # same-polarity lags available
)
T_PRE = CFG["pre"][1] - CFG["pre"][0]
C_PRE = 0.5 * (CFG["pre"][0] + CFG["pre"][1])          # -0.1375 s


# ---------------------------------------------------------------- formulas (model layer)
def var_mean(sigma, tau, T):
    z = T / tau
    return 2.0 * sigma ** 2 * (z + np.expm1(-z)) / z ** 2


def cov_point_pre(sigma, tau, t):
    """Cov(e(t), mean of e over the pre-window), t >= pre end."""
    a, b = CFG["pre"]
    return sigma ** 2 * tau / T_PRE * np.exp(-t / tau) * (np.exp(b / tau) - np.exp(a / tau))


def B3_sq(sigma, tau, t):
    return sigma ** 2 + var_mean(sigma, tau, T_PRE) - 2.0 * cov_point_pre(sigma, tau, t)


def step_avg_B3_sq(sigma, tau):
    t0, t1 = CFG["step"]
    # closed form of the step average of B3^2
    a, b = CFG["pre"]
    I = sigma ** 2 * tau / T_PRE * (np.exp(b / tau) - np.exp(a / tau)) * tau * (np.exp(-t0 / tau) - np.exp(-t1 / tau))
    return sigma ** 2 + var_mean(sigma, tau, T_PRE) - 2.0 * I / (t1 - t0)


def tilt_step_factor():
    """Step average of (t - C_PRE)^2 over [0, 1] s (s^2)."""
    t0, t1 = CFG["step"]
    return ((t1 - C_PRE) ** 3 - (t0 - C_PRE) ** 3) / 3.0 / (t1 - t0)


def V3(sigma, tau, T, h):
    g = h - T
    cov = sigma ** 2 * tau ** 2 * np.expm1(-T / tau) ** 2 * np.exp(-g / tau) / T ** 2
    return var_mean(sigma, tau, T) - cov


def sigma_b_match(sigma, tau):
    return np.sqrt(step_avg_B3_sq(sigma, tau) / tilt_step_factor())


def sigma_b_read(sigma, tau, T, h):
    return np.sqrt(2.0 * V3(sigma, tau, T, h)) / h


# ---------------------------------------------------------------- checks of the formulas
def oracle_checks():
    worst_v, worst_b = 0.0, 0.0
    for tau, sigma in CFG["ous"]:
        law = tc.R.nl.NoiseLaw(0.0, 0.5, sigma, tau)                 # one OU component (fast part off)
        for T in CFG["T_list"]:
            for h in (0.401, 0.802, 1.203):
                ref = tc.R.nl.semivariogram_continuous(law, tc.DT, T, T, h, 0.0)
                worst_v = max(worst_v, abs(V3(sigma, tau, T, h) - ref) / ref)
        # quadrature of the step average of B3^2, independent of the closed form; the
        # double integral is taken over w < u and doubled (the integrand's kink on the
        # diagonal |u - w| = 0 limited a square-domain dblquad to ~2e-9 in the first run)
        a, b = CFG["pre"]
        cov_q = lambda t: integrate.quad(lambda u: sigma ** 2 * np.exp(-(t - u) / tau), a, b, epsabs=0, epsrel=1e-13)[0] / T_PRE
        var_q = 2.0 * integrate.dblquad(lambda w, u: sigma ** 2 * np.exp(-(u - w) / tau), a, b, lambda u: a, lambda u: u,
                                        epsabs=0, epsrel=1e-13)[0] / T_PRE ** 2
        avg_q = integrate.quad(lambda t: sigma ** 2 + var_q - 2 * cov_q(t), *CFG["step"], epsabs=0, epsrel=1e-12)[0]
        worst_b = max(worst_b, abs(avg_q - step_avg_B3_sq(sigma, tau)) / avg_q)
    print("(b) closed forms: V3 against noise_law.semivariogram_continuous, max rel diff %.1e; step-averaged B3^2"
          " against quadrature, max rel diff %.1e  -> %s" % (worst_v, worst_b, "PASS" if max(worst_v, worst_b) < 1e-9 else "FAIL"))


def main():
    print("tilt step factor <(t + 0.1375)^2> over the step = %.5f s^2" % tilt_step_factor())
    oracle_checks()
    print("\n(a) T = 190 ms (topic doc v3 section 3.6.5):")
    for tau, sigma in CFG["ous"]:
        sm = sigma_b_match(sigma, tau)
        on = sm * (0 - C_PRE) / np.sqrt(B3_sq(sigma, tau, 0.0))
        end = sm * (1 - C_PRE) / np.sqrt(B3_sq(sigma, tau, 1.0))
        rd = {h: sigma_b_read(sigma, tau, 0.190, h) for h in CFG["h_doc"]}
        print("   OU %4.0f s, sigma_3 %.3f mV: sigma_b(match) %.3f mV/s; read at 0.6/0.8/1.0 s: %.3f / %.3f / %.3f"
              " (ratio %.3f / %.3f / %.3f); point error tilt/OU at onset %.2f, at end %.2f"
              % (tau, sigma, sm, rd[0.6], rd[0.8], rd[1.0], rd[0.6] / sm, rd[0.8] / sm, rd[1.0] / sm, on, end))
    print("\n(finding) ratio sigma_b(read at h) / sigma_b(match), same-polarity lags of the 200.5 ms train:")
    for T in CFG["T_list"]:
        for tau, sigma in CFG["ous"]:
            sm = sigma_b_match(sigma, tau)
            row = ["%.3f" % (sigma_b_read(sigma, tau, T, tc.h_of(L)) / sm) for L in CFG["lags_periods"]]
            print("   T = %3.0f ms, OU %4.0f s: lags %s periods (%s s): %s"
                  % (T * 1e3, tau, list(CFG["lags_periods"]), ", ".join("%.3f" % tc.h_of(L) for L in CFG["lags_periods"]), " / ".join(row)))
    # the two-lag estimator (no theta_hat): 2 [V(4P) - V(2P)] / (h4^2 - h2^2), under the OU at seconds
    h2, h4 = tc.h_of(2), tc.h_of(4)
    for T in CFG["T_list"]:
        row = []
        for tau, sigma in CFG["ous"]:
            s2 = 2.0 * (V3(sigma, tau, T, h4) - V3(sigma, tau, T, h2)) / (h4 ** 2 - h2 ** 2)
            row.append(np.sqrt(s2) / sigma_b_match(sigma, tau))
        print("   two-lag estimator (2 and 4 periods), T = %3.0f ms: read / match = %s (OU 3 / 10 / 30 s)"
              % (T * 1e3, " / ".join("%.3f" % x for x in row)))
    # the lag at which read = match, per OU and window
    for T in CFG["T_list"]:
        hs = []
        for tau, sigma in CFG["ous"]:
            sm = sigma_b_match(sigma, tau)
            f = lambda h: sigma_b_read(sigma, tau, T, h) - sm
            from scipy.optimize import brentq
            hs.append(brentq(f, T + 1e-3, 5.0))
        print("   T = %3.0f ms: the lag where read = match: %s s (OU 3 / 10 / 30 s)" % (T * 1e3, " / ".join("%.3f" % x for x in hs)))


if __name__ == "__main__":
    main()
