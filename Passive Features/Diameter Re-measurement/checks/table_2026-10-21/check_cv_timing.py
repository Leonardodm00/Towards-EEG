"""check_cv_timing.py -- the cost of Block 14.4's table at a production-like size (2026-10-21; standalone).

20,000 draws x 5 realizations of the smoke suite's fixture (tests/smoke/test_smoke_table.py: b = 1 + 0.06/d
+ 0.15 sin^2 phi + 0.05 ln(mu_hat d_hat), a draw-level offset of correlation 0.2, tau = 0.03, 20 % failures) on
the DEFAULT ranges (d 0.2-4 um, phi 0-90 deg, mu 0.3-3 /um): times table.choose_multiplier (the K-fold CV over
bandwidth_cv_grid) and table.build_table at the chosen c, and prints the CV scores. Both costs are linear in the
rows (the kernel sums are matrix products over the rows), so the times scale to 10^6 rows by the row ratio.

Run (from this folder):  python3 check_cv_timing.py   (a few minutes on 2 cores)
"""
import os
import platform
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(WS, "src"))
sys.path.insert(0, os.path.join(WS, "tests", "smoke"))

import test_smoke_table as T  # noqa: E402
from allen_diameter.analysis import table as TB  # noqa: E402
from allen_diameter.config import default_config  # noqa: E402

G, K_REAL, SEED = 20000, 5, 11


def main():
    cfg = default_config()
    print("python %s, numpy %s, %d cores" % (platform.python_version(), np.__version__, os.cpu_count()))
    rows = T.fixture(G, K_REAL, SEED, cfg, p_fail=lambda d: 0.2)
    R = TB.table_rows(rows, cfg)
    t0 = time.perf_counter()
    c, scores, edge = TB.choose_multiplier(R, cfg)
    t1 = time.perf_counter()
    tab = TB.build_table(R, cfg, "local_linear", c, dict(cv_parameter="bandwidth_multiplier",
                                                         cv_grid=cfg.correction.bandwidth_cv_grid,
                                                         cv_scores=scores, cv_at_grid_edge=edge))
    t2 = time.perf_counter()
    print("%d rows, %d in S; grid at c: %s" % (R.n_rows, R.y.size, tab.meta["grid_shape"]))
    print("CV over %s: scores %s (tau^2 = %.1e); chosen c = %g (at an edge: %s); %.1f s"
          % (list(cfg.correction.bandwidth_cv_grid), np.round(scores, 7).tolist(), T.TAU ** 2, c, edge, t1 - t0))
    print("table at c: %.1f s; scaled to 1e6 rows: CV %.0f s, table %.0f s"
          % (t2 - t1, (t1 - t0) * 1e6 / R.y.size, (t2 - t1) * 1e6 / R.y.size))


if __name__ == "__main__":
    main()
