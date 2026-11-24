"""SGT members of (H1) (symmetric, zeta = 1): FWHM per SD against the Gaussian's 2.355, and where the SD is infinite.

h(x) ~ (1 + xi |x|^beta)^-(1/xi + 1/beta) (xi > 0), exp(-|x|^beta) (xi = 0). Its tails fall as |x|^-(1 + beta/xi), so
the variance is finite only for xi < beta / 2. For each member: the FWHM W (numerical root of h = h(0)/2) and the SD
(numerical integral, when finite); W / SD is the FWHM a member would have if matched to the law by its SD instead.
Usage: python3 fwhm_vs_sd.py
"""
import numpy as np
from scipy import integrate, optimize


def h(x, beta, xi):
    x = np.abs(x)
    return np.exp(-x ** beta) if xi == 0 else (1.0 + xi * x ** beta) ** (-(1.0 / xi + 1.0 / beta))


def fwhm(beta, xi):
    return 2.0 * optimize.brentq(lambda x: h(x, beta, xi) - 0.5, 1e-9, 1e3)


def sd(beta, xi, upper=np.inf):
    if xi > 0 and xi >= beta / 2.0:
        return np.inf
    m0 = 2 * integrate.quad(lambda x: h(x, beta, xi), 0, upper, limit=500)[0]
    m2 = 2 * integrate.quad(lambda x: x * x * h(x, beta, xi), 0, upper, limit=500)[0]
    return np.sqrt(m2 / m0)


print("Gaussian check: W/SD = %.4f (2 sqrt(2 ln 2) = %.4f)" % (fwhm(2, 0) / sd(2, 0), 2 * np.sqrt(2 * np.log(2))))
for beta, xi, label in ((10.0, 1.834, "node 5353's fit"), (2.0, 0.5, "Gaussian core, power tails"),
                        (2.0, 1.0, "xi = beta/2"), (1.0, 2.0, "box corner"), (10.0, 0.0, "flat top, light tails")):
    W, s = fwhm(beta, xi), sd(beta, xi)
    r = W / s if np.isfinite(s) else 0.0
    print("beta %5.2f xi %5.3f (%s): W/SD %s -> SD-matched FWHM = %s x the law's FWHM"
          % (beta, xi, label, "%.3f" % r if r else "0 (SD infinite)",
             "%.2f" % (r / (2 * np.sqrt(2 * np.log(2)))) if r else "undefined"))
# the windowed SD of an infinite-variance member grows with the window
for U in (5.0, 50.0, 500.0):
    m0 = 2 * integrate.quad(lambda x: h(x, 2.0, 1.0), 0, U, limit=500)[0]
    m2 = 2 * integrate.quad(lambda x: x * x * h(x, 2.0, 1.0), 0, U, limit=500)[0]
    print("beta 2, xi 1: SD within |x| <= %5.0f (units of h): %.3f" % (U, np.sqrt(m2 / m0)))
