"""Bootstrap worlds built from data, with library resamplers only (arch, SciPy). No model, no plotting.

Each function takes data and returns resampled data or bootstrap replicates; what is computed on
the replicates is the caller's.
"""
import numpy as np
from scipy import stats
from arch.bootstrap import (IIDBootstrap, MovingBlockBootstrap, CircularBlockBootstrap,
                            StationaryBootstrap)

# scheme name -> arch class; "iid" resamples single samples, the others blocks of b samples
# (for "sb", b is the MEAN block length: arch's StationaryBootstrap restarts with probability 1/b)
SCHEMES = {
    "iid": IIDBootstrap,
    "mbb": MovingBlockBootstrap,
    "cbb": CircularBlockBootstrap,
    "sb": StationaryBootstrap,
}


def resample_window(x, scheme, b, B, seed):
    """B bootstrap copies of one window x (1-D), shape (B, n), through arch's public generator."""
    cls = SCHEMES[scheme]
    rng = np.random.default_rng(seed)
    bs = cls(x, seed=rng) if scheme == "iid" else cls(int(b), x, seed=rng)
    return np.stack([data[0] for data, _ in bs.bootstrap(B)])


def window_level_bootstrap(values, statistic, B, seed, level=0.95):
    """Nonparametric bootstrap with the WINDOW as the unit (scipy.stats.bootstrap, percentile method).

    values: the K per-window statistics of one cell. Returns (standard error, CI low, CI high,
    bootstrap distribution).
    """
    res = stats.bootstrap((np.asarray(values),), statistic, n_resamples=B, vectorized=True,
                          method="percentile", confidence_level=level,
                          random_state=np.random.default_rng(seed))
    return (res.standard_error, res.confidence_interval.low, res.confidence_interval.high,
            res.bootstrap_distribution)
