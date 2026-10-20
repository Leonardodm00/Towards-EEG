"""noise_law.py -- the recording-noise law of the Stage-7 twins: simulator and
closed-form moments (D-017, D-034, D-056; specs/SPEC.md Block 2).

Model layer only: no file I/O, no plotting, no module-level randomness. Every
function takes the law and the sampling interval explicitly; random numbers
enter as UNIT draws (standard normal arrays) made by the caller, so that the
same draws can be reused at every theta (common random numbers, D-050) and
the generator (Block 5) and the fit (Block 4) share one simulator.

The law (SPEC 2.1, eqs. L1-L3). For one trace sampled every dt (s) at
t_j = j dt, j = 0..n-1, with mid-time t_bar = (n-1) dt / 2,

    e_j = e_fast_j + e_slow_j + a + b (t_j - t_bar)                       (L1)
    e_fast_j = rho_f e_fast_{j-1} + eta_j,   eta_j  ~ N(0, sigma_f^2 (1 - rho_f^2))
    e_slow_j = rho_s e_slow_{j-1} + eta'_j,  eta'_j ~ N(0, sigma_s^2 (1 - rho_s^2))
    rho_s = exp(-dt / tau_s)                                              (L2)

both recursions started in their stationary laws, a ~ N(0, sigma_a^2) (mV)
and b ~ N(0, sigma_b^2) (mV/s) drawn once per trace. theta = (sigma_f, rho_f,
sigma_s, tau_s) in mV, --, mV, s; rho_f is the lag-1 coefficient AT THE dt
PASSED IN (D-014: a per-sample quantity; its transfer between sampling rates
is D-017 (f), open, and is not done here). For every lag k >= 0 (samples)

    gamma(k | theta, dt) = sigma_f^2 rho_f^k + sigma_s^2 exp(-k dt / tau_s)  (L3)

Closed forms (topic doc eqs. (7a), (7b), (9), (10), (8), (25); SPEC 2.4):
exact expectations of a demeaned window's S_0 and S_L, the ratio of
expectations used as the closed-form starting point of the fit, the visible
share of one OU component, and the semivariogram of two window means of one
trace, exact for the sampled law (discrete time) with the continuous-time
form of eq. (25) kept as an oracle.

Any constant added to a trace (the offset a, or the tilt's value at a
window's centre) is invisible to every demeaned statistic and to the
within-trace semivariogram; the simulator therefore uses the trace's own
mid-time as t_bar whatever the trace stands for (a whole sweep, or one LS
pre-window simulated on its own).

Pure ASCII (hpc-python-compat).
"""
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np
from scipy.signal import lfilter

ArrayLike = Union[int, Sequence[int], np.ndarray]


# ===========================================================================
#  The law
# ===========================================================================
@dataclass(frozen=True)
class NoiseLaw:
    """theta = (sigma_f, rho_f, sigma_s, tau_s): the stationary part of (L1).

    sigma_f : mV, SD of the fast AR(1) component (>= 0; 0 switches it off)
    rho_f   : lag-1 coefficient of the fast component at the sampling interval
              the law is used with, in (0, 1)
    sigma_s : mV, SD of the slow OU component (>= 0; 0 switches it off)
    tau_s   : s, correlation time of the slow component (> 0)
    The fit lives in the interior (sigma_f, sigma_s > 0); zero SDs are allowed
    so that one-component laws can serve as oracles and comparisons.
    """
    sigma_f: float
    rho_f: float
    sigma_s: float
    tau_s: float

    def __post_init__(self):
        vals = (self.sigma_f, self.rho_f, self.sigma_s, self.tau_s)
        if not all(np.isfinite(v) for v in vals):
            raise ValueError("NoiseLaw: every parameter must be finite, got %r" % (vals,))
        if self.sigma_f < 0 or self.sigma_s < 0:
            raise ValueError("NoiseLaw: sigma_f and sigma_s must be >= 0 (mV)")
        if not 0.0 < self.rho_f < 1.0:
            raise ValueError("NoiseLaw: rho_f must lie in (0, 1), got %r" % (self.rho_f,))
        if not self.tau_s > 0.0:
            raise ValueError("NoiseLaw: tau_s must be > 0 (s), got %r" % (self.tau_s,))

    def as_array(self) -> np.ndarray:
        """(4,) float64: (sigma_f, rho_f, sigma_s, tau_s) in mV, --, mV, s."""
        return np.array([self.sigma_f, self.rho_f, self.sigma_s, self.tau_s], dtype=float)

    @classmethod
    def from_array(cls, theta: np.ndarray) -> "NoiseLaw":
        theta = np.asarray(theta, dtype=float).ravel()
        if theta.shape != (4,):
            raise ValueError("NoiseLaw.from_array: expected 4 entries, got shape %r" % (theta.shape,))
        return cls(float(theta[0]), float(theta[1]), float(theta[2]), float(theta[3]))

    def rho_s(self, dt: float) -> float:
        """Lag-1 coefficient of the slow component at interval dt (s): exp(-dt/tau_s)."""
        return float(np.exp(-dt / self.tau_s))

    def tau_f(self, dt: float) -> float:
        """Correlation time (s) of the fast component at interval dt: -dt / ln(rho_f)."""
        return float(-dt / np.log(self.rho_f))


def autocovariance(law: NoiseLaw, dt: float, lags: ArrayLike) -> np.ndarray:
    """gamma(k | theta, dt) of (L3) at each lag in `lags` (samples, any sign), mV^2."""
    k = np.abs(np.asarray(lags, dtype=float))
    return law.sigma_f ** 2 * law.rho_f ** k + law.sigma_s ** 2 * np.exp(-k * dt / law.tau_s)


# ===========================================================================
#  Simulator (unit draws in, traces out)
# ===========================================================================
@dataclass(frozen=True)
class UnitDraws:
    """Standard-normal draws for `simulate_traces`: fast, slow (n_traces, n),
    tilt and offset (n_traces,). Made once by `draw_units`, reused at every
    theta (common random numbers)."""
    fast: np.ndarray
    slow: np.ndarray
    tilt: np.ndarray
    offset: np.ndarray


def draw_units(rng: np.random.Generator, n_traces: int, n_samples: int) -> UnitDraws:
    """Draw the unit noise for n_traces traces of n_samples each from `rng`,
    in the fixed order fast, slow, tilt, offset (so a given generator state
    gives the same draws whatever the law)."""
    if n_traces < 1 or n_samples < 2:
        raise ValueError("draw_units: need n_traces >= 1 and n_samples >= 2")
    fast = rng.standard_normal((n_traces, n_samples))
    slow = rng.standard_normal((n_traces, n_samples))
    tilt = rng.standard_normal(n_traces)
    offset = rng.standard_normal(n_traces)
    return UnitDraws(fast, slow, tilt, offset)


def _stationary_ar1(sigma: float, rho: float, one_minus_rho2: float, xi: np.ndarray) -> np.ndarray:
    """Stationary AR(1) of SD sigma and lag-1 coefficient rho driven by unit
    draws xi (n_traces, n): e_0 = sigma xi_0, e_j = rho e_{j-1} +
    sigma sqrt(1 - rho^2) xi_j. The recursion is scipy.signal.lfilter with
    b = [1], a = [1, -rho] along the last axis."""
    if sigma == 0.0:
        return np.zeros_like(xi)
    eta = (sigma * np.sqrt(one_minus_rho2)) * xi
    eta[:, 0] = sigma * xi[:, 0]
    return lfilter([1.0], [1.0, -rho], eta, axis=-1)


def simulate_traces(law: NoiseLaw, dt: float, xi_fast: np.ndarray, xi_slow: np.ndarray, *,
                    xi_tilt: Optional[np.ndarray] = None, sigma_b: float = 0.0,
                    xi_offset: Optional[np.ndarray] = None, sigma_a: float = 0.0) -> np.ndarray:
    """Traces of (L1)-(L2) from unit draws.

    law, dt   : the law and the sampling interval (s); rho_f is read at this dt.
    xi_fast, xi_slow : (n_traces, n) standard-normal draws of the two recursions.
    xi_tilt, sigma_b : (n_traces,) unit draws and the tilt SD (mV/s); the tilt
                of trace r is sigma_b xi_tilt[r] (t_j - t_bar), t_bar the
                trace's mid-time (n - 1) dt / 2, so it sums to zero over j.
    xi_offset, sigma_a : (n_traces,) unit draws and the offset SD (mV).
    Returns (n_traces, n) float64 noise traces in mV. Each component is left
    out when its SD is 0 (then its draws may be None).
    """
    xi_fast = np.asarray(xi_fast, dtype=float)
    xi_slow = np.asarray(xi_slow, dtype=float)
    if xi_fast.ndim != 2 or xi_fast.shape != xi_slow.shape or xi_fast.shape[1] < 2:
        raise ValueError("simulate_traces: xi_fast and xi_slow must both be (n_traces, n >= 2)")
    if not dt > 0.0:
        raise ValueError("simulate_traces: dt must be > 0 (s)")
    n_traces, n = xi_fast.shape
    rho_s = law.rho_s(dt)
    out = _stationary_ar1(law.sigma_f, law.rho_f, 1.0 - law.rho_f ** 2, xi_fast)
    out = out + _stationary_ar1(law.sigma_s, rho_s, -np.expm1(-2.0 * dt / law.tau_s), xi_slow)
    if sigma_b < 0.0 or sigma_a < 0.0:
        raise ValueError("simulate_traces: sigma_b and sigma_a must be >= 0")
    if sigma_b > 0.0:
        if xi_tilt is None or np.shape(xi_tilt) != (n_traces,):
            raise ValueError("simulate_traces: xi_tilt must be (n_traces,) when sigma_b > 0")
        t_centred = (np.arange(n, dtype=float) - 0.5 * (n - 1)) * dt          # s, zero-mean
        out = out + (sigma_b * np.asarray(xi_tilt, dtype=float))[:, None] * t_centred[None, :]
    if sigma_a > 0.0:
        if xi_offset is None or np.shape(xi_offset) != (n_traces,):
            raise ValueError("simulate_traces: xi_offset must be (n_traces,) when sigma_a > 0")
        out = out + (sigma_a * np.asarray(xi_offset, dtype=float))[:, None]
    return out


def simulate_sweep(law: NoiseLaw, dt: float, n_samples: int, units: UnitDraws, index: int = 0, *,
                   sigma_b: float = 0.0, sigma_a: float = 0.0) -> np.ndarray:
    """One continuous trace of n_samples (the D-056 (i) simulator for one
    sweep): row `index` of `units` through `simulate_traces`. Returns (n,)."""
    if units.fast.shape[1] != n_samples:
        raise ValueError("simulate_sweep: units were drawn for %d samples, not %d"
                         % (units.fast.shape[1], n_samples))
    out = simulate_traces(law, dt, units.fast[index:index + 1], units.slow[index:index + 1],
                          xi_tilt=units.tilt[index:index + 1], sigma_b=sigma_b,
                          xi_offset=units.offset[index:index + 1], sigma_a=sigma_a)
    return out[0]


# ===========================================================================
#  Closed-form moments of a demeaned window (topic doc eqs. 7a, 7b, 9, 10, 8)
# ===========================================================================
# Custom: exact first moments of the demeaned-window sums S_0 and S_L under a
# stationary Gaussian law; no library computes them (statsmodels gives the
# estimators, not their expectations). Checked: statsmodels.tsa.stattools,
# scipy.stats, numpy.cov. Verified against brute force (full covariance
# matrix) and Monte Carlo in smoke_noise_law.py.
def var_window_mean(law: NoiseLaw, dt: float, n: int) -> float:
    """Var(x_bar | theta, n) of the mean of n consecutive samples, eq. (7a):
    (1/n^2) sum_{k=-(n-1)}^{n-1} (n - |k|) gamma(|k|). mV^2."""
    n = int(n)
    if n < 1:
        raise ValueError("var_window_mean: n must be >= 1")
    g = autocovariance(law, dt, np.arange(n))
    weights = np.full(n, 2.0)
    weights[0] = 1.0
    return float(np.dot(weights * (n - np.arange(n)), g) / n ** 2)


def expected_s0(law: NoiseLaw, dt: float, n: int) -> float:
    """E[S_0 | theta, n] = n gamma(0) - n Var(x_bar | theta, n), eq. (7b). mV^2."""
    n = int(n)
    if n < 2:
        raise ValueError("expected_s0: n must be >= 2")
    return float(n * autocovariance(law, dt, 0) - n * var_window_mean(law, dt, n))


def expected_sl(law: NoiseLaw, dt: float, n: int, lags: ArrayLike) -> np.ndarray:
    """E[S_L | theta, n] of eq. (9) at each lag L in `lags` (samples, 1 <= L <= n-1):

        (n-L) gamma(L) - (1/n) sum_{j=1}^{n-L} zeta_j - (1/n) sum_{j=L+1}^{n} zeta_j
        + (n-L) Var(x_bar | theta, n),      zeta_j = sum_{i=1}^{n} gamma(|i-j|).

    zeta_j is formed from the cumulative sum C_m = sum_{k=0}^{m} gamma(k) as
    C_{j-1} + C_{n-j} - gamma(0). Returns float array, NaN where L is outside
    [1, n-1]. mV^2."""
    n = int(n)
    lags = np.atleast_1d(np.asarray(lags, dtype=int))
    out = np.full(lags.shape, np.nan)
    if n < 2:
        return out
    g = autocovariance(law, dt, np.arange(n))
    cum = np.cumsum(g)                                   # C_m, m = 0..n-1
    j = np.arange(1, n + 1)
    zeta = cum[j - 1] + cum[n - j] - g[0]                # zeta_j, j = 1..n
    zcum = np.concatenate([[0.0], np.cumsum(zeta)])      # zcum[m] = sum_{j=1}^{m} zeta_j
    var_mean = var_window_mean(law, dt, n)
    ok = (lags >= 1) & (lags <= n - 1)
    L = lags[ok]
    head = zcum[n - L]                                   # sum_{j=1}^{n-L}
    tail = zcum[n] - zcum[L]                             # sum_{j=L+1}^{n}
    out[ok] = (n - L) * g[L] - (head + tail) / n + (n - L) * var_mean
    return out


def closed_form_sd(law: NoiseLaw, dt: float, n: int) -> float:
    """sqrt(E[sigma_hat^2 | theta, n]) = sqrt(E[S_0 | theta, n] / (n - 1)), mV:
    the closed-form approximation of the SD statistic (topic doc 3.5.2)."""
    return float(np.sqrt(expected_s0(law, dt, n) / (n - 1)))


def closed_form_acf(law: NoiseLaw, dt: float, n: int, lags: ArrayLike) -> np.ndarray:
    """Ratio of expectations E[S_L]/E[S_0], eq. (10): the closed-form
    approximation of the textbook autocorrelation statistic. It is biased
    high against the estimator's median at the project's window lengths
    (0.018-0.065, topic doc 3.5.2), so it is a starting point for the fit,
    not its binding function (D-033, D-056)."""
    return expected_sl(law, dt, n, lags) / expected_s0(law, dt, n)


def visible_share(z: ArrayLike) -> np.ndarray:
    """v(z) of eq. (8): the fraction of one OU component's variance that
    survives demeaning a window of duration T = z tau (continuous time):
    1 - 2 (z - 1 + e^{-z}) / z^2, with z - 1 + e^{-z} as z + expm1(-z)."""
    z = np.asarray(z, dtype=float)
    return 1.0 - 2.0 * (z + np.expm1(-z)) / z ** 2


# ===========================================================================
#  Window means of one trace: covariance and semivariogram (eq. 25, exact)
# ===========================================================================
# Custom: the covariance of two window means of a sampled stationary process,
# summed over sample pairs by their lag count (a trapezoid), O(n) instead of
# O(n1 n2); no library routine. Checked: numpy, scipy.signal. Verified against
# brute force, Monte Carlo and the continuous-time eq. (25) in the smoke test.
def window_mean_cov(law: NoiseLaw, dt: float, n1: int, n2: int, offset: int) -> float:
    """Cov(x_bar_1, x_bar_2 | theta) of the means of two windows of one trace:
    window 1 = samples [0, n1), window 2 = samples [offset, offset + n2)
    (offset in samples, any sign; the windows may overlap). mV^2."""
    n1, n2, offset = int(n1), int(n2), int(offset)
    if n1 < 1 or n2 < 1:
        raise ValueError("window_mean_cov: n1 and n2 must be >= 1")
    d = np.arange(offset - (n1 - 1), offset + n2)                  # j - i over all pairs
    lo = np.maximum(0, offset - d)
    hi = np.minimum(n1 - 1, offset + n2 - 1 - d)
    count = np.maximum(0, hi - lo + 1).astype(float)              # pairs at each lag d
    return float(np.dot(count, autocovariance(law, dt, d)) / (n1 * n2))


def semivariogram(law: NoiseLaw, dt: float, n1: int, n2: int, offset: int, sigma_b: float = 0.0) -> float:
    """V(h | theta, sigma_b) of eq. (25), exact for the sampled law: half the
    expected squared difference of the two window means of `window_mean_cov`
    plus the tilt's (1/2) sigma_b^2 h^2, with h (s) the distance between the
    windows' centres, (offset + (n2 - 1)/2 - (n1 - 1)/2) dt. mV^2."""
    v1 = var_window_mean(law, dt, n1)
    v2 = var_window_mean(law, dt, n2)
    cov = window_mean_cov(law, dt, n1, n2, offset)
    h = (offset + 0.5 * (n2 - 1) - 0.5 * (n1 - 1)) * dt
    return float(0.5 * (v1 + v2 - 2.0 * cov) + 0.5 * sigma_b ** 2 * h ** 2)


def semivariogram_continuous(law: NoiseLaw, dt: float, T1: float, T2: float, h: float,
                             sigma_b: float = 0.0) -> float:
    """Eq. (25) as written in the topic doc (continuous time), summed over the
    two OU components (the fast one with tau_f = -dt / ln rho_f):
    (1/2) sum_comp [Var_T1 + Var_T2 - 2 Cov_T1,T2(h)] + (1/2) sigma_b^2 h^2,
    Var_T = 2 sigma^2 (z - 1 + e^{-z}) / z^2 (z = T/tau),
    Cov_T1,T2(h) = sigma^2 tau^2 (1 - e^{-T1/tau}) (1 - e^{-T2/tau}) e^{-g/tau} / (T1 T2),
    gap g = h - (T1 + T2)/2 >= 0. Oracle for `semivariogram`: for the slow
    component (tau_s >> dt) the two agree to O(dt/tau_s); for the fast
    component, whose tau_f is below dt, the continuous form does not
    approximate the sampled law, but that component's share of a window mean
    is of order sigma_f^2 / n and negligible at the project's windows. mV^2."""
    if h < 0.5 * (T1 + T2):
        raise ValueError("semivariogram_continuous: the windows overlap (h < (T1 + T2)/2)")
    g = h - 0.5 * (T1 + T2)
    total = 0.0
    for sigma, tau in ((law.sigma_f, law.tau_f(dt)), (law.sigma_s, law.tau_s)):
        if sigma == 0.0:
            continue
        z1, z2 = T1 / tau, T2 / tau
        var1 = 2.0 * sigma ** 2 * (z1 + np.expm1(-z1)) / z1 ** 2
        var2 = 2.0 * sigma ** 2 * (z2 + np.expm1(-z2)) / z2 ** 2
        cov = sigma ** 2 * tau ** 2 * (-np.expm1(-z1)) * (-np.expm1(-z2)) * np.exp(-g / tau) / (T1 * T2)
        total += 0.5 * (var1 + var2 - 2.0 * cov)
    return float(total + 0.5 * sigma_b ** 2 * h ** 2)
