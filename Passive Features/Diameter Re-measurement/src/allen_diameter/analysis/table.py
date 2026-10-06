"""The bias table b(d, phi | C) -- Block 6 in specs/SPEC.md (procedure s.3.8;
SPEC section 2.4).

From the replicates retained by the selection S, a smoothing thin-plate spline
b_hat(ln d, phi) through (ln d_n, phi_n, d_hat_n / d_n), inputs ln(d / um) and
phi in rad, d and phi the phantom's TRUE values (procedure Eq. 2); the
smoothing is chosen by k-fold cross-validation over spline_smoothing_grid.
m_hat(d, phi) = d * b_hat. Around any (d, phi), Gaussian weights
w_n = exp(-((ln d_n - ln d)/h_d)^2 / 2 - ((phi_n - phi)/h_phi)^2 / 2) give

    tau_hat^2 = sum w r^2 / sum w * N_eff / (N_eff - 1),  r_n = ratio_n - b_hat(x_n),
    N_eff = (sum w)^2 / sum w^2,  SE = tau_hat / sqrt(N_eff)          (retained replicates)
    failure rate = sum w (1 - in_S) / sum w                           (all replicates)

Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.interpolate import RBFInterpolator


def table_inputs(d_um, phi_rad):
    """(n, 2) inputs (ln d, phi) of the spline; d um > 0, phi rad (broadcasting)."""
    d, phi = np.broadcast_arrays(np.asarray(d_um, dtype=float), np.asarray(phi_rad, dtype=float))
    if np.any(~(d > 0)) or not np.all(np.isfinite(phi)):
        raise ValueError("table inputs need d > 0 and finite phi")
    return np.column_stack([np.log(d.ravel()), phi.ravel()])


def _spline(X, y, lam):
    return RBFInterpolator(X, y, kernel="thin_plate_spline", smoothing=float(lam))


def choose_smoothing(X, y, grid, folds, seed):
    """(smoothing, CV mean squared error per grid value) by k-fold CV; folds
    assigned by a permutation from numpy.random.default_rng(seed). A value whose
    fit fails (singular system) scores inf. The first minimum wins."""
    n = X.shape[0]
    if n < 2 * folds:
        raise ValueError("cross-validation needs at least 2 x folds retained replicates, got %d" % n)
    fold = np.empty(n, dtype=int)
    fold[np.random.default_rng(seed).permutation(n)] = np.arange(n) % folds
    scores = []
    for lam in grid:
        err = 0.0
        try:
            for f in range(folds):
                tr = fold != f
                err += float(np.sum((_spline(X[tr], y[tr], lam)(X[~tr]) - y[~tr]) ** 2))
        except np.linalg.LinAlgError:
            err = math.inf
        scores.append(err / n)
    k = int(np.argmin(scores))
    if not math.isfinite(scores[k]):
        raise ValueError("no smoothing value gives a finite cross-validation error")
    return float(grid[k]), np.array(scores)


class BiasTable:
    """b_hat, m_hat, tau_hat, SE and failure rate of one configuration C."""

    def __init__(self, X_kept, ratio_kept, X_all, ok_all, smoothing, bandwidth, d_range, phi_range_rad,
                 signature, estimator_hash, cv_grid=(), cv_scores=()):
        self.X_kept = np.asarray(X_kept, dtype=float)
        self.ratio_kept = np.asarray(ratio_kept, dtype=float)
        self.X_all = np.asarray(X_all, dtype=float)
        self.ok_all = np.asarray(ok_all, dtype=bool)
        self.smoothing = float(smoothing)
        self.bandwidth = (float(bandwidth[0]), float(bandwidth[1]))     # (ln d, rad)
        self.d_range = (float(d_range[0]), float(d_range[1]))
        self.phi_range_rad = (float(phi_range_rad[0]), float(phi_range_rad[1]))
        self.signature = signature
        self.estimator_hash = str(estimator_hash)
        self.cv_grid = np.asarray(cv_grid, dtype=float)
        self.cv_scores = np.asarray(cv_scores, dtype=float)
        self._rbf = _spline(self.X_kept, self.ratio_kept, self.smoothing)
        self._resid = self.ratio_kept - self._rbf(self.X_kept)

    def b_hat(self, d_um, phi_rad):
        shape = np.broadcast(np.asarray(d_um), np.asarray(phi_rad)).shape
        return self._rbf(table_inputs(d_um, phi_rad)).reshape(shape)

    def m_hat(self, d_um, phi_rad):
        return np.asarray(d_um, dtype=float) * self.b_hat(d_um, phi_rad)

    def _weights(self, d_um, phi_rad, X):
        Q = table_inputs(d_um, phi_rad)
        z0 = (Q[:, 0:1] - X[None, :, 0]) / self.bandwidth[0]
        z1 = (Q[:, 1:2] - X[None, :, 1]) / self.bandwidth[1]
        return np.exp(-0.5 * (z0 ** 2 + z1 ** 2)), np.broadcast(np.asarray(d_um), np.asarray(phi_rad)).shape

    def n_eff(self, d_um, phi_rad):
        w, shape = self._weights(d_um, phi_rad, self.X_kept)
        return ((w.sum(1) ** 2) / np.maximum((w ** 2).sum(1), np.finfo(float).tiny)).reshape(shape)

    def tau_hat(self, d_um, phi_rad):
        w, shape = self._weights(d_um, phi_rad, self.X_kept)
        sw = w.sum(1)
        neff = sw ** 2 / np.maximum((w ** 2).sum(1), np.finfo(float).tiny)
        var = (w * self._resid[None, :] ** 2).sum(1) / np.maximum(sw, np.finfo(float).tiny)
        with np.errstate(divide="ignore", invalid="ignore"):
            var = np.where(neff > 1, var * neff / (neff - 1), np.nan)
        return np.sqrt(var).reshape(shape)

    def se_hat(self, d_um, phi_rad):
        return self.tau_hat(d_um, phi_rad) / np.sqrt(self.n_eff(d_um, phi_rad))

    def failure_rate(self, d_um, phi_rad):
        w, shape = self._weights(d_um, phi_rad, self.X_all)
        return ((w * (~self.ok_all)[None, :]).sum(1) / np.maximum(w.sum(1), np.finfo(float).tiny)).reshape(shape)

    def check_estimator(self, cfg):
        """ValueError unless cfg's estimator signature is the table's (procedure s.3.11)."""
        h = cfg.signature_hash("estimator")
        if h != self.estimator_hash:
            raise ValueError("bias table built for estimator %s, the fit uses %s: rebuild the table or use the "
                             "matching configuration" % (self.estimator_hash, h))


def fit_table(rows, cfg):
    """BiasTable from replicate rows (dicts with d_um, phi_rad or meas_phi_rad -- by
    correction.table_phi_axis -- ratio, in_S)."""
    c = cfg.correction
    if c.response_estimator != "tps_spline" or c.table_statistic != "mean":
        raise NotImplementedError("response_estimator %r / table_statistic %r: only tps_spline / mean in Block 6"
                                  % (c.response_estimator, c.table_statistic))
    d = np.array([float(r["d_um"]) for r in rows])
    phi_key = {"true": "phi_rad", "measured": "meas_phi_rad"}[c.table_phi_axis]
    phi = np.array([float(r[phi_key]) for r in rows])
    ratio = np.array([float(r["ratio"]) for r in rows])
    ok = np.array([bool(r["in_S"]) for r in rows]) & np.isfinite(ratio) & np.isfinite(phi)
    phi = np.where(np.isfinite(phi), phi, 0.0)       # rows without a measured tilt are outside S anyway
    X = table_inputs(d, phi)
    if c.spline_smoothing == "cv":
        lam, scores = choose_smoothing(X[ok], ratio[ok], c.spline_smoothing_grid, c.spline_cv_folds, cfg.phantom.seed)
        grid = c.spline_smoothing_grid
    else:
        lam, scores, grid = float(c.spline_smoothing), (), ()
    ph = cfg.phantom
    return BiasTable(X[ok], ratio[ok], X, ok, lam, (c.tau_kernel_bandwidth[0], math.radians(c.tau_kernel_bandwidth[1])),
                     ph.d_range_um, (math.radians(ph.phi_range_deg[0]), math.radians(ph.phi_range_deg[1])),
                     cfg.full_signature(), cfg.signature_hash("estimator"), grid, scores)
