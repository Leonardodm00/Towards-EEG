"""check_inverse_vs_classical.py -- why Step 5 inverts the mean response (classical calibration)
instead of regressing d on d_hat across the phantoms (the inverse estimator): tested on simulated
data with a known truth (2026-10-18; standalone, not pipeline code).

Truth (1-D, fixed phi and alpha_hat): b(d) = 1 + 0.06/d, m(d) = d + 0.06 (linear, so the classical
inverse m^-1(d_hat) = d_hat - 0.06 has no Jensen gap); d_hat = m(d) + d tau(d) eps, eps ~ N(0, 1),
tau(d) = clip(0.009 (d/1.5)^-3.27, 0.009, 0.30) as in check_table_se_coverage.py.
Phantoms ("design"): ln d ~ U[ln 0.2, ln 4], N = 400 000. Inverse estimator: E_design[d | d_hat],
estimated by the mean of d in 400 log-spaced bins of d_hat (linear interpolation between bin centres).
Two "real" populations the design does not resemble: thin, ln d ~ N(ln 0.6, 0.25^2); thick,
ln d ~ N(ln 2.0, 0.15^2); 200 000 nodes each, same measurement model.

Claim: the classical inverse is unbiased for both populations (|mean relative error| <= 3 MC SE);
the inverse estimator's mean error depends on the population (it is the posterior mean under the
design's law of d, Bayes' rule) and exceeds 3 MC SE for the thin one, where the noise is large.

Run:  python3 check_inverse_vs_classical.py      (seconds)
"""
import math

import numpy as np

CFG = dict(seed=20261018, n_design=400_000, n_real=200_000, n_bins=400, d_range=(0.2, 4.0),
           pops={"thin": (math.log(0.6), 0.25), "thick": (math.log(2.0), 0.15)})


def tau(d):
    return np.clip(0.009 * (d / 1.5) ** (-3.27), 0.009, 0.30)


def measure(d, rng):
    return d + 0.06 + d * tau(d) * rng.standard_normal(d.size)


def main():
    rng = np.random.default_rng(CFG["seed"])
    lo, hi = CFG["d_range"]
    d_des = np.exp(rng.uniform(math.log(lo), math.log(hi), CFG["n_design"]))
    dh_des = measure(d_des, rng)
    ok = dh_des > 0
    edges = np.exp(np.linspace(math.log(dh_des[ok].min()), math.log(dh_des[ok].max()), CFG["n_bins"] + 1))
    idx = np.clip(np.digitize(dh_des[ok], edges) - 1, 0, CFG["n_bins"] - 1)
    cnt = np.bincount(idx, minlength=CFG["n_bins"])
    mean_d = np.bincount(idx, weights=d_des[ok], minlength=CFG["n_bins"]) / np.maximum(cnt, 1)
    centres = np.sqrt(edges[:-1] * edges[1:])
    use = cnt >= 20
    print("%-6s %12s %12s %12s %12s" % ("pop", "classical", "MC SE", "inverse est.", "MC SE"))
    for name, (mu, sd) in CFG["pops"].items():
        d = np.exp(rng.normal(mu, sd, CFG["n_real"]))
        d = d[(d >= lo) & (d <= hi)]
        dh = measure(d, rng)
        keep = dh > 0
        d, dh = d[keep], dh[keep]
        e_cl = (dh - 0.06 - d) / d
        e_inv = (np.interp(dh, centres[use], mean_d[use]) - d) / d
        se_cl, se_inv = e_cl.std(ddof=1) / math.sqrt(d.size), e_inv.std(ddof=1) / math.sqrt(d.size)
        print("%-6s %+12.5f %12.5f %+12.5f %12.5f   classical %s, inverse %s" % (
            name, e_cl.mean(), se_cl, e_inv.mean(), se_inv,
            "unbiased (PASS)" if abs(e_cl.mean()) <= 3 * se_cl else "biased (FAIL)",
            "biased" if abs(e_inv.mean()) > 3 * se_inv else "not distinguishable from 0"))


if __name__ == "__main__":
    main()
