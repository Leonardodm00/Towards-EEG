"""check_table_inversion.py -- three claims of the correction-table explanation,
tested on simulated data with a known truth (2026-10-18; standalone, not pipeline code).

Truth (chosen): b(d) = 1 + c0/d + c2 sin^2(phi) at fixed phi, m(d) = d b(d); the
fitted diameter of one replicate is d_hat = m(d) + d tau eps, eps ~ N(0, 1), so that
tau is the SD of the ratio d_hat / d, as in mathematics Eq. 19 and procedure Eq. 7.
Inversion: d_tilde solves m(d_tilde) = d_hat (scipy.optimize.brentq on the exact m).

C1  mean correction: E[d_tilde]/d - 1 equals the Jensen term
    0.5 (m^-1)''(m(d)) Var(d_hat) / d  with (m^-1)'' = -m''/(m')^3   [mathematics s.3.6]
    within 3 Monte Carlo SE (both tau).
C2  corrected scatter: SD(d_tilde)/d equals tau / [b (1 + beta)],
    beta = d ln b / d ln d  [procedure s.3.8], within 2 % at tau = 0.03 (first-order
    propagation; at tau = 0.2 the curvature of m^-1 adds to the spread, reported only).
C3  shortcut error: (d_hat / b(d_hat) - d*)/d* on the noise-free d_hat = m(d) equals
    -beta (b - 1) / (1 + beta (b - 1)) of mathematics Eq. 21 up to the second-order
    remainder of its linearisation of b: tolerance (b - 1)^2 (the smoke suite uses 2 (b - 1)^2).

Run:  python3 check_table_inversion.py
"""
import numpy as np
from scipy.optimize import brentq

CFG = dict(seed=20261018, n=200_000, c0=0.06, c2=0.15, phi=0.3,
           d_list=(0.5, 1.5), tau_list=(0.03, 0.2), d_bracket=(0.05, 8.0))


def b(d):
    return 1.0 + CFG["c0"] / d + CFG["c2"] * np.sin(CFG["phi"]) ** 2


def m(d):
    return d * b(d)


def m1(d):            # dm/dd = b + d b' = 1 + c2 sin^2 phi   (the c0/d term drops out)
    return 1.0 + CFG["c2"] * np.sin(CFG["phi"]) ** 2


def m2(d):            # d^2 m / dd^2 = 0 for this fixture: use a second fixture term below
    return 0.0


def beta(d):          # d ln b / d ln d = d b'(d) / b(d)
    return (-CFG["c0"] / d) / b(d)


def invert(y):
    lo, hi = CFG["d_bracket"]
    return brentq(lambda d: m(d) - y, lo, hi, xtol=1e-12)


def main():
    rng = np.random.default_rng(CFG["seed"])
    rows = []
    # C1/C2 need a curved m: add a quadratic bias term so (m^-1)'' != 0
    CFG["c3"] = 0.05                                   # b += c3 * d  (bias growing with d)
    global b, m, m1, m2, beta

    def b(d):
        return 1.0 + CFG["c0"] / d + CFG["c2"] * np.sin(CFG["phi"]) ** 2 + CFG["c3"] * d

    def m(d):
        return d * b(d)

    def m1(d):
        return 1.0 + CFG["c2"] * np.sin(CFG["phi"]) ** 2 + 2.0 * CFG["c3"] * d

    def m2(d):
        return 2.0 * CFG["c3"]

    def beta(d):
        return d * (-CFG["c0"] / d ** 2 + CFG["c3"]) / b(d)

    for d in CFG["d_list"]:
        for tau in CFG["tau_list"]:
            eps = rng.standard_normal(CFG["n"])
            d_hat = m(d) + d * tau * eps
            d_hat = d_hat[(d_hat > m(CFG["d_bracket"][0])) & (d_hat < m(CFG["d_bracket"][1]))]
            d_til = np.array([invert(y) for y in d_hat[:40_000]])     # 40k roots per case
            rel_bias = d_til.mean() / d - 1.0
            se_bias = d_til.std(ddof=1) / np.sqrt(len(d_til)) / d
            var_dhat = (d * tau) ** 2
            jensen = 0.5 * (-m2(d) / m1(d) ** 3) * var_dhat / d
            sd_rel = d_til.std(ddof=1) / d
            sd_pred = tau / (b(d) * (1.0 + beta(d)))
            ok1 = abs(rel_bias - jensen) <= 3 * se_bias
            ok2 = abs(sd_rel / sd_pred - 1) <= 0.02 if tau == 0.03 else True
            rows.append(("C1 d=%.1f tau=%.2f" % (d, tau), ok1,
                         "E[d~]/d-1 = %+.5f (MC SE %.5f); Jensen %+.5f" % (rel_bias, se_bias, jensen)))
            rows.append(("C2 d=%.1f tau=%.2f" % (d, tau), ok2,
                         "SD(d~)/d = %.5f; tau/[b(1+beta)] = %.5f%s" % (sd_rel, sd_pred, "" if tau == 0.03 else "  (not tested at tau 0.2)")))
    for d in CFG["d_list"]:
        y = m(d)                       # noise-free measurement; d* = d exactly
        d_short = y / b(y)
        err = d_short / d - 1.0
        bb, be = b(d), beta(d)
        pred = -be * (bb - 1) / (1 + be * (bb - 1))
        rows.append(("C3 shortcut d=%.1f" % d, abs(err - pred) <= (bb - 1) ** 2,
                     "(d_s - d*)/d* = %+.5f; Eq. 21 %+.5f (first order %+.5f); tol (b-1)^2 = %.5f" % (err, pred, -be * (bb - 1), (bb - 1) ** 2)))
    print("%-24s %-5s %s" % ("check", "pass", "detail"))
    for name, ok, det in rows:
        print("%-24s %-5s %s" % (name, "PASS" if ok else "FAIL", det))


if __name__ == "__main__":
    main()
