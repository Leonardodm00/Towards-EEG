"""check04_pooling_options.py -- per-cell, cohort, or partially pooled
sigma_b: which gives the twins the tilt SD closest to each cell's own, for an
18-cell cohort with the L3_exc sweep counts (14 cells of 2 SS sweeps, one of
6, three of 10; stored windows K per cell from the archive).

Truth (chosen): sigma_b(c) = sigma_med exp(omega z_c), z_c ~ N(0, 1) iid;
omega is the log-SD of sigma_b across cells (0 = one common sigma_b). The
stationary law is theta_cf in every cell (an assumption: the real cells'
laws differ). Each cell's noise is a row of its cohort bank (make_banks.py
cohort), the tilts are fresh draws, and (T1) is applied at the true law,
or (variant) at V0 perturbed by theta_hat's error (check03 (d)).

Options (each returns sigma_b_tilde(c)):
 (a) per cell, (T1) clipped at 0 (D-034 as implemented);
 (b) one cohort value: the S_eff-weighted mean of the unclipped per-cell
     estimates, clipped at 0;
 (c-DL) the D-076 machinery on log sigma_b^2: y_c = log max(s2_c, floor_c)
     minus the chi-square log-bias, v_c = trigamma(S_eff/2), DerSimonian-Laird
     omega^2 (statsmodels, clipped at 0 here), shrinkage towards the
     random-effects mean;
 (c-EB) moderated variances (Smyth 2004 / limma): scaled-inverse-chi-square
     prior fitted by the moments of log s2, posterior value
     (d0 s0^2 + S_eff s2_c)/(d0 + S_eff).
floor_c = the SD of the cell's noise-only (T1) (its bank at sigma_b = 0), the
assistant's choice; it only enters the logs of (c).

Metric (per cell, over replicates): the ratio sigma_b_tilde / sigma_b; the
share outside a factor 2 (zeros counted outside), the share of zeros, the
median and 90th percentile of |log ratio| (symmetric: a factor 2 too large
and too small count alike; a zero is infinitely far), the median ratio and
its 5-95 % range, and the RMS relative error (which weighs overestimates more
than underestimates: a zero costs 1, a factor 3 too large costs 2). Reported
for the 2-sweep and the 10-sweep cells. Prediction, fixed before running: with omega = 0, (b) and both (c)
beat (a) for 2-sweep cells; with omega = 1, (b) is worst; the finding is
where (c) sits between.

Also asserted: statsmodels 0.15 combine_effects(method_re='dl') returns the
UNCLIPPED moment estimate (negative when the spread is below the sampling
variance) and uses it in its random-effects weights.

Run:  python make_banks.py cohort (first), then  python check04_pooling_options.py   (~2 min)
"""
import time
import warnings
from pathlib import Path

import numpy as np
from statsmodels.stats.meta_analysis import combine_effects

import tilt_common as tc

# statsmodels warns (sqrt of a negative variance) exactly when its unclipped tau2 is negative;
# that case is the one asserted below and handled by the clip in tilt_common.pool_dl_log
warnings.filterwarnings("ignore", category=RuntimeWarning, module="statsmodels")

CFG = dict(
    seed=20261028, n_rep=4000, sigma_med=0.1,
    scenarios=(  # (label, omega, sigma_med, theta_error)
        ("omega 0", 0.0, 0.1, False), ("omega 0.25", 0.25, 0.1, False), ("omega 0.5", 0.5, 0.1, False),
        ("omega 1.0", 1.0, 0.1, False), ("omega 0.5, sigma_med 0.05", 0.5, 0.05, False),
        ("omega 0.5, theta_hat error", 0.5, 0.1, True),
    ),
    bank_dir=Path(__file__).resolve().parent.parent / "results" / "banks",
    theta_err_file=Path(__file__).resolve().parent.parent / "results" / "check03_theta_err.npz",
    options=("a: per cell", "b: cohort", "c-DL: log, DerSimonian-Laird", "c-EB: moderated variance"),
)


def assert_statsmodels_unclipped():
    v = np.array([0.5, 1, 2, 4, 0.3, 1.5])
    y = np.array([0.1, -0.1, 0.05, 0.0, 0.12, -0.08])
    r = combine_effects(y, v, method_re="dl")
    neg = r.tau2 < 0
    w_neg = np.any(np.asarray(r.weights_re) < 0)
    print("statsmodels combine_effects(dl) on a homogeneous toy: tau2 = %.3f (negative: %s), negative RE weights: %s"
          " -> the clip at 0 must be ours" % (r.tau2, neg, w_neg))


def assert_log_chi2_moments():
    """E[log(chi2_nu/nu)] = psi(nu/2) - log(nu/2), Var = psi_1(nu/2): the bias and the
    sampling variance the DL form uses (Monte Carlo, 10^6 draws, own seed; tolerance 3 SE)."""
    rng = np.random.default_rng(7)
    for nu in (2.0, 6.0, 10.0):
        x = np.log(rng.chisquare(nu, 1_000_000) / nu)
        m, v = tc.log_chi2_moments(nu)
        se_m, se_v = x.std() / 1e3, np.sqrt((np.mean((x - x.mean()) ** 4) - x.var() ** 2) / 1e6)
        ok = abs(x.mean() - m) < 3 * se_m and abs(x.var() - v) < 3 * se_v
        print("log(chi2_%g/%g): mean %.4f (formula %.4f), variance %.4f (formula %.4f) -> %s"
              % (nu, nu, x.mean(), float(m), x.var(), float(v), "PASS" if ok else "FAIL"))


def load_cohort():
    cells = []
    for spec, S, K in tc.CFG["cohort"]:
        z = np.load(CFG["bank_dir"] / ("cohort_c%d.npz" % spec))
        lay = tc.Layout(str(spec), z["sweep"], z["pos"], int(z["n_sweeps"]))
        d, s_of = tc.pairs_from_means(z["M"], lay, 4)
        npp = np.bincount(s_of, minlength=S)
        V0 = tc.v_model(tc.LAW_CF, 4)
        floor = float(((np.mean(d ** 2, axis=1) - 2 * V0) / tc.h_of(4) ** 2).std(ddof=1))
        cells.append(dict(spec=spec, S=S, K=K, d=d, s_of=s_of, Se=tc.s_eff(npp), N=int(npp.sum()), floor=floor, V0=V0))
    return cells


def run(cells, label, omega, sigma_med, theta_err, rng, sd_logV0):
    n_rep, C = CFG["n_rep"], len(cells)
    sig = sigma_med * np.exp(omega * rng.standard_normal((n_rep, C)))            # truth per replicate and cell
    s2 = np.empty((n_rep, C))
    for j, c in enumerate(cells):
        rows = rng.integers(0, c["d"].shape[0], n_rep)
        b = sig[:, j:j + 1] * rng.standard_normal((n_rep, c["S"]))
        V0 = c["V0"] * (np.exp(sd_logV0 * rng.standard_normal(n_rep)) if theta_err else 1.0)
        s2[:, j] = tc.sigma_b2_from_pairs(c["d"][rows], c["s_of"], b, 4, V0)
    nu = np.array([c["Se"] for c in cells])
    floor = np.array([c["floor"] for c in cells])
    est = {o: np.empty((n_rep, C)) for o in CFG["options"]}
    om2, d0 = np.empty(n_rep), np.empty(n_rep)
    for r in range(n_rep):
        est["a: per cell"][r] = np.sqrt(np.clip(s2[r], 0, None))
        est["b: cohort"][r] = np.sqrt(tc.pool_cohort(s2[r], nu))
        v, om2[r], _ = tc.pool_dl_log(s2[r], nu, floor)
        est["c-DL: log, DerSimonian-Laird"][r] = np.sqrt(v)
        v, d0[r], _ = tc.pool_eb_chi2(s2[r], nu, floor)
        est["c-EB: moderated variance"][r] = np.sqrt(v)
    S = np.array([c["S"] for c in cells])
    print("\n== %s (sigma_med %.3f mV/s, %d cohorts): DL omega_hat^2 = 0 in %.0f %%, median sqrt(omega_hat^2)/2 = %.2f"
          " (the implied log-SD of sigma_b); EB d0 = inf in %.0f %%, median finite d0 %.1f"
          % (label, sigma_med, n_rep, 100 * np.mean(om2 == 0), np.median(np.sqrt(om2) / 2), 100 * np.mean(~np.isfinite(d0)),
             np.median(d0[np.isfinite(d0)]) if np.any(np.isfinite(d0)) else np.nan))
    out = {}
    for grp, mask in (("2-sweep", S == 2), ("10-sweep", S == 10)):
        for o in CFG["options"]:
            ratio = est[o][:, mask] / sig[:, mask]
            rms = np.sqrt(np.mean((ratio - 1) ** 2))
            off2 = np.mean((ratio < 0.5) | (ratio > 2))
            zero = np.mean(ratio == 0)
            med = np.median(ratio)
            q = np.quantile(ratio, (0.05, 0.95))
            with np.errstate(divide="ignore"):
                alog = np.abs(np.log(ratio))                    # a zero estimate counts as infinitely far
            mal, p90 = np.median(alog), np.quantile(alog, 0.9)
            out[(grp, o)] = (rms, off2, zero, med, mal, p90)
            print("   %-8s %-30s outside x2 %5.1f %% | zero %4.1f %% | median |log ratio| %.2f | 90th pct %s | median ratio %.2f"
                  " | 5-95 %% %.2f-%.2f | RMS rel err %.3f"
                  % (grp, o, 100 * off2, 100 * zero, mal, ("%.2f" % p90) if np.isfinite(p90) else "inf", med, q[0], q[1], rms))
    return out


def main():
    t0 = time.time()
    assert_statsmodels_unclipped()
    assert_log_chi2_moments()
    cells = load_cohort()
    print("cohort: " + ", ".join("%d:%dsw K%d S_eff %.2f" % (c["spec"], c["S"], c["K"], c["Se"]) for c in cells))
    print("floor (SD of noise-only sigma_b_hat^2), (mV/s)^2: 2-sweep cells %.4f-%.4f, 10-sweep %.4f-%.4f"
          % (min(c["floor"] for c in cells if c["S"] == 2), max(c["floor"] for c in cells if c["S"] == 2),
             min(c["floor"] for c in cells if c["S"] == 10), max(c["floor"] for c in cells if c["S"] == 10)))
    sd_logV0 = float(np.load(CFG["theta_err_file"])["sd_logV0"]) if CFG["theta_err_file"].exists() else 0.25
    print("theta_hat error variant: SD(log V0) = %.3f per cell (check03 (d); fixture layout, applied to every cell)" % sd_logV0)
    rng = np.random.default_rng(CFG["seed"])
    res = {}
    for label, omega, smed, terr in CFG["scenarios"]:
        res[label] = run(cells, label, omega, smed, terr, rng, sd_logV0)
    # save for the figure
    keys = [k for k in res]
    arr = np.array([[[res[k][(g, o)] for o in CFG["options"]] for g in ("2-sweep", "10-sweep")] for k in keys])
    np.savez(CFG["bank_dir"].parent / "check04_results.npz", arr=arr, scenarios=np.array(keys), options=np.array(CFG["options"]))
    print("\n(%.0f s)" % (time.time() - t0))


if __name__ == "__main__":
    main()
