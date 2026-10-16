"""Shared pieces for the scripts of TEEG_bias_of_a_statistic_2026-10-11.md.

Model, statistic and exact expectations only; no plotting, no reporting.

Law (Stage-7 topic doc, eqs. (5b)/(6)): stationary fast AR(1) + slow OU, sampled every dt, one independent
window per draw; the per-sweep offset cancels on demeaning and the tilt is left out (not part of this question).
Statistic (topic doc eq. (3)): textbook sample autocorrelation T = S_L / S_0 of the demeaned window.
Exact expectations of S_0 and S_L (topic doc eq. (9)) for any stationary autocovariance.
"""
import numpy as np
from scipy.signal import lfilter

DT = 20e-6                                   # s, sampling interval of the LS pre-window (50 kHz)
N_LS = 13_250                                # samples, 265 ms LS pre-window
LAG_1MS = 50                                 # samples, 1 ms at DT

# cohort laws of topic doc sec. 3.5.3 (sigma in mV, tau in s); theta_sim with the unrounded tau_s
THETA_SIM = dict(sigma_f=0.0383, rho_f=0.179, sigma_s=0.0547, tau_s=0.03974)
THETA_CF = dict(sigma_f=0.0380, rho_f=0.175, sigma_s=0.0508, tau_s=0.0366)
# best single AR(1) of topic doc sec. 3.5.3 (a law that does not fit the cohort), written as a degenerate theta
AR1_BEST = dict(sigma_f=0.0505, rho_f=0.780, sigma_s=0.0, tau_s=1.0)


# --- model -----------------------------------------------------------------------------------------
def simulate_windows(th, dt, n, n_win, rng):
    """n_win independent stationary windows of the two-component law, shape (n_win, n), in mV.

    Draw order (fast innovations, fast start, slow innovations, slow start) is fixed, so the same seed
    and chunking reproduce the same windows (and those of the session's earlier scripts of 2026-10-11, not shipped).
    A component with sigma = 0 still consumes its random numbers (keeps the draw order) and adds zeros.
    """
    rs = np.exp(-dt / th["tau_s"])
    out = np.zeros((n_win, n))
    for sigma, a in ((th["sigma_f"], th["rho_f"]), (th["sigma_s"], rs)):
        e = rng.normal(0.0, sigma * np.sqrt(1 - a**2), (n_win, n))
        e[:, 0] = rng.normal(0.0, sigma, n_win)            # stationary start
        out += lfilter([1.0], [1.0, -a], e, axis=1)
    return out


def autocov(th, dt, n):
    """gamma(k | theta), k = 0..n-1, topic doc eq. (6), in mV^2."""
    k = np.arange(n)
    return th["sigma_f"]**2 * th["rho_f"]**k + th["sigma_s"]**2 * np.exp(-k * dt / th["tau_s"])


def true_acf(th, dt, lag):
    """rho(lag | theta) = gamma(lag | theta) / gamma(0 | theta)."""
    g = autocov(th, dt, lag + 1)
    return g[lag] / g[0]


# --- exact expectations of the quadratic forms ----------------------------------------------------------
def expected_S(gam, n, L):
    """E[S_L | theta, n], topic doc eq. (9), from the autocovariance gam[k], k = 0..n-1.

    zeta_j = sum_i gamma(|i - j|) is a sliding sum of length n over gamma(|m|), m = -(n-1)..(n-1).
    """
    g_full = np.concatenate([gam[:0:-1], gam])
    cs = np.concatenate([[0.0], np.cumsum(g_full)])
    j = np.arange(1, n + 1)
    zeta = cs[2 * n - j] - cs[n - j]                         # zeta_j, j = 1..n
    var_mean = zeta.sum() / n**2                              # Var(x_bar | theta, n)
    return (n - L) * gam[L] - zeta[: n - L].sum() / n - zeta[L:].sum() / n + (n - L) * var_mean


def closed_form_acf(th, dt, n, L):
    """q_cf = E[S_L | theta, n] / E[S_0 | theta, n], topic doc eq. (10)."""
    gam = autocov(th, dt, n)
    return expected_S(gam, n, L) / expected_S(gam, n, 0)


# --- statistic -------------------------------------------------------------------------------------
def quadratic_forms(x, lag):
    """S_0 and S_L per window (rows), each window demeaned; T = S_L / S_0 is topic doc eq. (3)."""
    u = x - x.mean(axis=1, keepdims=True)
    return np.sum(u * u, axis=1), np.sum(u[:, :-lag] * u[:, lag:], axis=1)


def monte_carlo(th, dt, n, lag, M, seed, chunk=200):
    """S_0 and S_L over M simulated windows, generated in chunks (fixed seed -> fixed windows)."""
    rng = np.random.default_rng(seed)
    S0, SL = [], []
    for start in range(0, M, chunk):
        m = min(chunk, M - start)
        s0, sl = quadratic_forms(simulate_windows(th, dt, n, m, rng), lag)
        S0.append(s0)
        SL.append(sl)
    return np.concatenate(S0), np.concatenate(SL)
