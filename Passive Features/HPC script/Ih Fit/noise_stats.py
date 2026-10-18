"""noise_stats.py -- sample statistics of recording-noise windows (D-028, D-029).

One function, sample_acf, is the only autocorrelation estimator of Ih Fit/.
It is the textbook (biased) estimator, computed by statsmodels:

    rho_hat(L) = sum_{j=1}^{n-L} u_j u_{j+L} / sum_{j=1}^{n} u_j^2,
    u_j = x_j - mean(x),

i.e. statsmodels.tsa.stattools.acf(x, adjusted=False)[L]. It replaces the
hand-written estimator that divided the lag sum by n - L (D-028.2:
textbook = (n - L)/n * old, window by window). Spec: specs/SPEC.md, Block 1.
Pure ASCII.
"""
from typing import Sequence, Union

import numpy as np
from statsmodels.tsa.stattools import acf

# D-029 (iii): NaN at any lag with fewer than this many lag products (n - L).
MIN_PRODUCTS = 10


def sample_acf(x: np.ndarray, lags: Union[int, Sequence[int]]) -> np.ndarray:
    """Textbook sample autocorrelation of ONE window at each lag in `lags`.

    x     : 1-D array of samples (any unit; the result is dimensionless).
    lags  : int or sequence of ints, in samples.
    return: float array, one value per lag, in the order given. NaN where
            L < 1 or n - L < MIN_PRODUCTS, and everywhere when the window
            has a non-finite sample or zero variance (as the code it
            replaces did). One statsmodels call per window, nlags = max lag.
    """
    x = np.asarray(x, dtype=float).ravel()
    lags = np.atleast_1d(np.asarray(lags, dtype=int))
    out = np.full(lags.shape, np.nan)
    n = x.size
    ok = (lags >= 1) & (n - lags >= MIN_PRODUCTS)
    if not ok.any() or not np.all(np.isfinite(x)):
        return out
    u = x - x.mean()
    if not float(np.dot(u, u)) > 0.0:
        return out
    r = acf(x, adjusted=False, nlags=int(lags[ok].max()), fft=True)
    out[ok] = r[lags[ok]]
    return out
