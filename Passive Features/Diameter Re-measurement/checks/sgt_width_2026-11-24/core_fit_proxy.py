"""Does the FWHM rule (D-222.2) reproduce the calibration's own width estimate when the true blur is an SGT?

The calibration (SPEC Block 10, dip_statistics) fits a Gaussian a exp(-(v - v_c)^2 / (2 s^2)) by least squares to the
dip within |v - v_peak| <= 1.0 x FWHM (samples 0.1144 um apart) and takes s as the width; the law's sigma_r is built
from those s. The SGT option then gives the SGT kernel the FWHM 2 sqrt(2 ln 2) sigma_r. Here the true blur IS an SGT
of FWHM W (a point-like object: the kernel itself is the dip); the Gaussian-core fit is run on it, and the ratio
2.355 s / W says how far the SGT's FWHM under the rule is from its true FWHM (1 = the rule is exact). Noise-free;
sampled; v_c and v_peak at a sample, and half a sample off.
Usage: python3 core_fit_proxy.py
"""
import numpy as np
from scipy import optimize

STEP = 0.1144          # um, the profile's sampling (profile_step_um)
K = 2.0 * np.sqrt(2.0 * np.log(2.0))


def h(x, beta, xi):
    x = np.abs(x)
    return np.exp(-x ** beta) if xi == 0 else (1.0 + xi * x ** beta) ** (-(1.0 / xi + 1.0 / beta))


def unit_fwhm(beta, xi):
    return 2.0 * optimize.brentq(lambda x: h(x, beta, xi) - 0.5, 1e-9, 1e3)


def core_fit_s(beta, xi, W, offset):
    """The calibration's estimator on the SGT of FWHM W um sampled every STEP um, centre at offset samples."""
    Wu = unit_fwhm(beta, xi)
    v = (np.arange(-400, 401) + offset) * STEP
    w = h(v * Wu / W, beta, xi)
    n0 = int(np.argmax(w))
    win = np.abs(v - v[n0]) <= 1.0 * W                # core_fit_fwhm_factor 1.0 x FWHM (the true FWHM here)
    vv, ww = v[win], w[win]
    res = lambda p: p[0] * np.exp(-(vv - p[1]) ** 2 / (2 * p[2] ** 2)) - ww
    fit = optimize.least_squares(res, [1.0, 0.0, W / K], xtol=1e-12, ftol=1e-12, gtol=1e-12)
    return abs(fit.x[2]), int(win.sum())


print("ratio = 2.355 s / W: the FWHM the rule gives the SGT, per its true FWHM (1 = exact)")
for beta, xi, label in ((2.0, 0.0, "Gaussian"), (10.0, 1.834, "node 5353's fit"), (2.0, 0.5, "Gaussian core, power tails"),
                        (1.0, 0.0, "peaked (Laplace)"), (10.0, 0.0, "flat top, light tails"), (4.0, 1.0, "mid")):
    out = []
    for W in (0.3, 0.6, 1.5):
        for off in (0.0, 0.5):
            s, n = core_fit_s(beta, xi, W, off)
            out.append(K * s / W)
    print("beta %5.2f xi %5.3f (%-26s): ratio %.3f to %.3f over FWHM 0.3, 0.6, 1.5 um, centred on or between samples"
          % (beta, xi, label, min(out), max(out)))
