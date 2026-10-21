"""check_tps_folds.py -- why the spline's cross-validation folds by draw (D-062.3), on the tps_spline path of
Block 14.4 (2026-10-21; standalone).

100 draws x 5 realizations of the smoke suite's fixture with a constant b and a strong draw-level offset
(correlation 0.8, tau = 0.03), on gate 2's reduced ranges: the smoothing chosen by table.choose_smoothing (folds
by draw) against the same 5-fold CV with folds by row (row n in fold n mod 5), four seeds. Folds by row score each
held-out row against near-copies of it in the training folds and choose too little smoothing.

Run (from this folder):  python3 check_tps_folds.py   (about a minute)
"""
import os
import sys

import numpy as np
from scipy.interpolate import RBFInterpolator

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(WS, "src"))
sys.path.insert(0, os.path.join(WS, "tests", "smoke"))

import test_smoke_table as T  # noqa: E402
from allen_diameter.analysis import table as TB  # noqa: E402


def by_row(Xs, y, grid, folds):
    fold = np.arange(y.size) % folds
    scores = []
    for lam in grid:
        err = 0.0
        for f in range(folds):
            tr = fold != f
            err += float(np.sum((RBFInterpolator(Xs[tr], y[tr], kernel="thin_plate_spline", smoothing=lam)(Xs[~tr])
                                 - y[~tr]) ** 2))
        scores.append(err)
    return grid[int(np.argmin(scores))]


def main():
    cfg = T.reduced_cfg(response_estimator="tps_spline")
    grid, K = cfg.correction.spline_smoothing_grid, cfg.correction.spline_cv_folds
    for seed in range(4):
        rows = T.fixture(100, 5, 500 + seed, cfg, rho=0.8, b_fn=lambda a, b, c: 1.0 + 0.0 * a, p_fail=lambda d: 0.0)
        R = TB.table_rows(rows, cfg)
        axes = TB.grid_axes(R.X, cfg, 1.0)
        lo = np.array([a[0] for a in axes])
        span = np.array([a[-1] - a[0] for a in axes])
        Xs = (R.X - lo) / span
        lam_draw, _, edge = TB.choose_smoothing(Xs, R.y, R.draw, grid, K)
        print("seed %d: folds by draw choose %g (at the grid's end: %s), folds by row %g"
              % (seed, lam_draw, edge, by_row(Xs, R.y, grid, K)))


if __name__ == "__main__":
    main()
