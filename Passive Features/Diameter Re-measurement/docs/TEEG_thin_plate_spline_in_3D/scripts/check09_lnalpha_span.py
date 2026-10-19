"""check09_lnalpha_span.py -- what sets the scale of the third axis under D-054 (i)'s range normalisation
(2026-10-18; standalone). Document section 3.12 (2).

On the fixture's design (tps_common.design, the D-024 (iii) laws; alpha = mu d / cos phi), for the ten tables of
check07 (default_rng([20261018, b])): the span max - min of ln alpha over the 1500 rows, the largest tilt drawn,
the span of the 1st-99th percentiles as a share of the full span, and the share of rows with u3 < 0.5 after the
min-max normalisation. Prediction (reasoning, fixed before running): -ln cos(phi) is unbounded as phi -> 90 deg,
so the span is set by the single most tilted draw and most rows sit in the lower part of the u3 axis.
No tolerance: a description.
Run (same folder):  python3 check09_lnalpha_span.py      (seconds)
"""
import math

import numpy as np

from tps_common import design

CFG = dict(table_seed=20261018, tables=10, n=1500)


def main():
    spans, shares, lowers, tilts = [], [], [], []
    for b in range(CFG["tables"]):
        X = design(np.random.default_rng([CFG["table_seed"], b]), CFG["n"])
        la = X[:, 2]
        span = la.max() - la.min()
        p1, p99 = np.percentile(la, [1, 99])
        u3 = (la - la.min()) / span
        spans.append(span)
        shares.append((p99 - p1) / span)
        lowers.append(float(np.mean(u3 < 0.5)))
        tilts.append(math.degrees(X[:, 1].max()))
        print("table %d: ln alpha span %.2f (max tilt %.3f deg); 1-99 percentile span / span = %.2f; "
              "rows with u3 < 0.5: %.0f %%" % (b, span, tilts[-1], shares[-1], 100 * lowers[-1]))
    print("over tables: span %.2f-%.2f; percentile share %.2f-%.2f; rows with u3 < 0.5: %.0f-%.0f %%"
          % (min(spans), max(spans), min(shares), max(shares), 100 * min(lowers), 100 * max(lowers)))


if __name__ == "__main__":
    main()
