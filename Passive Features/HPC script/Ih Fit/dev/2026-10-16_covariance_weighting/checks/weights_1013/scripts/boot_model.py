"""Model and statistic for the bootstrap-types checks of 2026-10-13. No resampling, no plotting.

Law: the Stage-7 two-component noise law (topic doc TD (5b)/(6)), stationary fast AR(1) + slow OU,
sampled every dt; the per-sweep offset cancels on demeaning and the tilt is left out.
Statistic: the textbook sample autocorrelation T = S_L / S_0 of a demeaned window (D-028.1,
statsmodels acf(adjusted=False)); computed here with NumPy for speed and asserted equal to
statsmodels on a few windows in check_boot_types.py.

The generator is copied from bias_doc/scripts/bias_common.py (2026-10-11), same draw order, so the
same seed and chunking give the same windows as that script.
"""
import numpy as np
from scipy.signal import lfilter


def simulate_windows(th, dt, n, n_win, rng):
    """n_win independent stationary windows of the two-component law, shape (n_win, n), in mV."""
    rs = np.exp(-dt / th["tau_s"])
    out = np.zeros((n_win, n))
    for sigma, a in ((th["sigma_f"], th["rho_f"]), (th["sigma_s"], rs)):
        e = rng.normal(0.0, sigma * np.sqrt(1 - a**2), (n_win, n))
        e[:, 0] = rng.normal(0.0, sigma, n_win)            # stationary start
        out += lfilter([1.0], [1.0, -a], e, axis=1)
    return out


def true_acf(th, dt, lag):
    """rho(lag | theta) = gamma(lag | theta) / gamma(0 | theta), the target g(theta) of the bias doc."""
    g0 = th["sigma_f"]**2 + th["sigma_s"]**2
    gl = th["sigma_f"]**2 * th["rho_f"]**lag + th["sigma_s"]**2 * np.exp(-lag * dt / th["tau_s"])
    return gl / g0


def acf_tb(x, lag):
    """T = S_L / S_0 along the last axis, each row demeaned by its own mean (textbook ACF)."""
    u = x - x.mean(axis=-1, keepdims=True)
    s0 = np.einsum("...j,...j->...", u, u)
    sl = np.einsum("...j,...j->...", u[..., :-lag], u[..., lag:])
    return sl / s0


def lag_sums_about(x, centre, lag):
    """S0~ and SL~ about a FIXED centre (no re-demeaning): used for the join identity of block worlds."""
    v = x - centre
    s0 = np.einsum("...j,...j->...", v, v)
    sl = np.einsum("...j,...j->...", v[..., :-lag], v[..., lag:])
    return s0, sl


def circular_wrap_sum(x, lag):
    """W_L = sum_{j=n-L+1}^{n} u_j u_{j+L-n}: the lag-L products that wrap from the end to the start."""
    u = x - x.mean()
    return float(np.dot(u[-lag:], u[:lag]))


def simulate_T(th, dt, n, lag, M, seed, chunk=200):
    """T over M independent windows from p(. | theta, n), generated in chunks (fixed seed -> fixed windows)."""
    rng = np.random.default_rng(seed)
    out = []
    for start in range(0, M, chunk):
        m = min(chunk, M - start)
        out.append(acf_tb(simulate_windows(th, dt, n, m, rng), lag))
    return np.concatenate(out)
