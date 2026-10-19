"""check07b_paired.py -- the paired, per-table reading of check07_kernel_comparison.py (2026-10-18; standalone).
Document section 3.11. Reads check07.log (run check07 first) and, for each table and variant, recomputes the
effective degrees of freedom df = trace(H) at the smoothing that variant's CV chose (same tables: the fixture of
tps_common.py drawn with default_rng([20261018, b]); range normalisation of D-054 (i)).
Reports, against 'thin_plate_spline': per-table differences of RMS and slope RMS, how many tables favour each
variant, a two-sided sign test (scipy.stats.binomtest), the median difference, and the median df.
Added after the first run: for each table where 'thin_plate_spline''s CV chose a smoothing other than the modal 10,
the same table refitted at lambda_s = 0.1, 1 and 10 (RMS on check07's 3000 evaluation points), so that what the CV
choice cost within that table is measured, not inferred across tables.
No tolerance: a description of the logged comparison.
Run (same folder):  python3 check07b_paired.py      (about 1 min)
"""
import re

import numpy as np
from scipy.stats import binomtest

from scipy.interpolate import RBFInterpolator

from check07_kernel_comparison import CFG as C7
from tps_common import Scaler, design, hat_trace, tau_of_d, truth

CFG = dict(log="check07.log", table_seed=20261018, n=1500,
           variants=(("thin_plate_spline", 1), ("linear", 1), ("linear", 0)))
ROW = re.compile(r"table (\d+): (.*)")
CELL = re.compile(r"(\S+)/deg(\d) lam (\S+) rms (\S+) slope (\S+)")


def parse():
    rows = {}
    for line in open(CFG["log"]):
        m = ROW.match(line.strip())
        if not m:
            continue
        cells = [CELL.search(c) for c in m.group(2).split("|")]
        rows[int(m.group(1))] = [(c.group(1), int(c.group(2)), float(c.group(3)), float(c.group(4)),
                                  float(c.group(5))) for c in cells]
    return rows


def main():
    rows = parse()
    tables = sorted(rows)
    df = np.zeros((len(tables), len(CFG["variants"])))
    for i, b in enumerate(tables):
        rng = np.random.default_rng([CFG["table_seed"], b])
        X = design(rng, CFG["n"])
        _ = truth(X[:, 0], X[:, 1], X[:, 2]) + tau_of_d(np.exp(X[:, 0])) * rng.standard_normal(CFG["n"])
        Xs = Scaler(X)(X)
        for j, (kernel, degree) in enumerate(CFG["variants"]):
            df[i, j] = hat_trace(Xs, rows[b][j][2], kernel=kernel, degree=degree)
    print("per-table df at the CV smoothing (tps, linear deg1, linear deg0):")
    for i, b in enumerate(tables):
        print("  table %d: lam %s -> df %s" % (b, [rows[b][j][2] for j in range(3)], np.round(df[i], 1)))
    print("median df: %s ; mean df: %s" % (np.round(np.median(df, axis=0), 1), np.round(df.mean(axis=0), 1)))
    tps_med = {k: float(np.median([rows[b][0][k] for b in tables])) for k in (3, 4)}
    print("tps medians over tables: RMS %.4f, slope RMS %.4f" % (tps_med[3], tps_med[4]))
    for j in (1, 2):
        name = "%s/deg%d" % CFG["variants"][j]
        for k, label in ((3, "RMS"), (4, "slope RMS")):
            d = np.array([rows[b][j][k] - rows[b][0][k] for b in tables])
            wins = int(np.sum(d > 0))
            p = binomtest(wins, len(d), 0.5).pvalue
            med = float(np.median(d))
            print("%-14s %-9s: differences to tps %s; tps lower in %d of %d tables, sign test p = %.3f; "
                  "median difference %+.4f (%.0f %% of the tps median)"
                  % (name, label, np.round(d, 4).tolist(), wins, len(d), p, med, 100 * med / tps_med[k]))
    others = [rows[b][0][3] for b in tables if rows[b][0][2] != 0.1]
    t8 = [rows[b][0][3] for b in tables if rows[b][0][2] == 0.1]
    print("tps RMS where CV chose 0.1: %s; mean of the other tables %.4f; ratio %.2f"
          % (t8, float(np.mean(others)), t8[0] / float(np.mean(others))))
    for b in tables:                                   # within-table cost of the CV choice (added after the first run)
        lam_cv = rows[b][0][2]
        if lam_cv == 10.0:
            continue
        r = {lam: rms_at(b, lam) for lam in (0.1, 1.0, 10.0)}
        print("table %d, thin_plate_spline refitted: RMS at lambda 0.1 / 1 / 10 = %.4f / %.4f / %.4f; CV chose %g, "
              "its RMS is %.2f times that at lambda 10" % (b, r[0.1], r[1.0], r[10.0], lam_cv, r[lam_cv] / r[10.0]))


def rms_at(b, lam, kernel="thin_plate_spline", degree=1):
    """RMS(b_hat - b) of table b at a fixed smoothing, on check07's evaluation points (same draws as check07)."""
    rng = np.random.default_rng([C7["table_seed"], b])
    X = design(rng, C7["n"])
    y = truth(X[:, 0], X[:, 1], X[:, 2]) + tau_of_d(np.exp(X[:, 0])) * rng.standard_normal(C7["n"])
    sc = Scaler(X)
    E = design(np.random.default_rng(C7["eval_seed"]), C7["n_eval"])
    f = RBFInterpolator(sc(X), y, kernel=kernel, degree=degree, smoothing=lam)
    return float(np.sqrt(np.mean((f(sc(E)) - truth(E[:, 0], E[:, 1], E[:, 2])) ** 2)))


if __name__ == "__main__":
    main()
