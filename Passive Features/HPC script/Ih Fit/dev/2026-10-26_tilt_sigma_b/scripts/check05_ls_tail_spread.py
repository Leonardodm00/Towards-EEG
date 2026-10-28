"""check05_ls_tail_spread.py -- is the spread of the Long Square tail change
across the 18 L3_exc cells what one common variance at seconds would give?

Data (not re-read here; quoted from the decision log D-017 note 4 and the
noise handoff section 4.6, cluster run of 2026-10-01): per-cell median of
|delta_end| over the cell's 5-6 hyperpolarising LS sweeps, minimum 0.048 mV,
maximum 0.620 mV, cohort median 0.128 mV; sweeps per cell: 6 in 14 cells,
5 in 4 (topic doc Appendix A.1). The per-cell values themselves are not in
project knowledge, so only these three order statistics are used.

Model. Within a cell, delta_end of one sweep ~ N(0, s_c^2) (Gaussian under
the noise law with a tilt or an OU at seconds; mean zero: the per-cell
median is positive in 12 of 18 cells, p = 0.24). Homogeneous: one s for all
cells. Heterogeneous: s_c = exp(omega z_c), z_c ~ N(0, 1). The statistics
max/min, max/median and median/min of the 18 per-cell medians do not depend
on the common scale.

Predictions, fixed before running: none on the numbers; the question is
whether the observed max/min = 12.9 is improbable under homogeneity
(P < 0.01 taken as "not one common law"), and which omega makes it typical.

Run:  python check05_ls_tail_spread.py     (seconds)
"""
import numpy as np

CFG = dict(seed=20261029, n_rep=200000, n_sweeps=[6] * 14 + [5] * 4,
           observed=dict(min=0.048, max=0.620, median=0.128), omegas=np.round(np.arange(0.0, 1.61, 0.1), 2))


def per_cell_medians(rng, omega, n_rep):
    n = np.array(CFG["n_sweeps"])
    s = np.exp(omega * rng.standard_normal((n_rep, n.size)))
    med = np.empty((n_rep, n.size))
    for j, k in enumerate(n):
        med[:, j] = np.median(np.abs(rng.standard_normal((n_rep, k))), axis=1) * s[:, j]
    return med


def stats(med):
    mn, mx, md = med.min(axis=1), med.max(axis=1), np.median(med, axis=1)
    return mx / mn, mx / md, md / mn


def main():
    rng = np.random.default_rng(CFG["seed"])
    o = CFG["observed"]
    obs = (o["max"] / o["min"], o["max"] / o["median"], o["median"] / o["min"])
    print("observed: max/min %.1f, max/median %.2f, median/min %.2f" % obs)
    r = stats(per_cell_medians(rng, 0.0, CFG["n_rep"]))
    print("homogeneous (one s for all cells, %d cohorts): P(max/min >= obs) = %.2e; P(max/median >= obs) = %.2e;"
          " P(median/min >= obs) = %.3f; median max/min %.2f [95th pct %.2f]"
          % (CFG["n_rep"], np.mean(r[0] >= obs[0]), np.mean(r[1] >= obs[1]), np.mean(r[2] >= obs[2]),
             np.median(r[0]), np.quantile(r[0], 0.95)))
    print("heterogeneous, s_c = exp(omega z_c):")
    rows = []
    for om in CFG["omegas"]:
        r = stats(per_cell_medians(rng, om, 40000))
        lo, mid, hi = np.quantile(r[0], (0.05, 0.5, 0.95))
        rows.append((om, lo, mid, hi))
        print("   omega %.1f: max/min 5/50/95 %% = %.1f / %.1f / %.1f; P(max/min >= obs) = %.3f; P(max/median >= obs) = %.3f"
              % (om, lo, mid, hi, np.mean(r[0] >= obs[0]), np.mean(r[1] >= obs[1])))
    ok = [om for om, lo, mid, hi in rows if lo <= obs[0] <= hi]
    best = min(rows, key=lambda x: abs(np.log(x[2] / obs[0])))
    print("omega for which the observed max/min lies inside the 5-95 %% band: %.1f-%.1f; median match at omega %.1f"
          % (min(ok), max(ok), best[0]))


if __name__ == "__main__":
    main()
