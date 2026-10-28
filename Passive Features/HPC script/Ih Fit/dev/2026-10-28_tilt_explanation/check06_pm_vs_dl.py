"""check06_pm_vs_dl.py -- standalone check written for the chat explanation of
sigma_b(c) on 2026-10-28 (not pipeline code, not in the repository).

Question: inside option (c) (partial pooling of log sigma_b(c)^2 across the
18 cells), does the between-cell variance estimator matter? DerSimonian-Laird
(the D-076 machinery, as in check04 of the tilt study) against Paule-Mandel,
the estimator Veroniki et al. 2016 (Res Synth Methods 7:55, full text) find to
have the more favourable profile in general and statsmodels' default.

Both forms: y_c = log max(s2_c, floor_c) - E[log(chi2_nu/nu)],
v_c = psi_1(nu_c / 2), nu_c = S_eff(c); tau^2 (here Omega^2) from
statsmodels.stats.meta_analysis.combine_effects(method_re="dl" | "pm"),
clipped at 0; random-effects mean mu with the clipped value; shrunk
y_c -> mu + Omega^2 / (Omega^2 + v_c) (y_c - mu); sigma_b_tilde = exp(y/2).

Truth (chosen, as check04): the 18 L3_exc layouts (cohort banks written by
make_banks.py cohort: noise-only window means, theta_cf), sigma_b(c) =
0.1 exp(omega z_c) mV/s, z_c ~ N(0, 1); tilts drawn fresh; (T1) at the true
law; omega in (0, 0.25, 0.5, 1.0); 4000 cohorts per omega; seed 20261034.

Predictions, fixed before running:
 P1  statsmodels' PM tau2 is never negative (counted over every cohort).
 P2  on the share of 2-sweep cells whose sigma_b_tilde is off by more than a
     factor 2, PM and DL differ by less than 3 percentage points at every omega.
 P3  (a) per cell reproduces check04's shares (25.1 / 25.4 / 25.7 / 30.5 % for
     2-sweep cells) within 2 points (another seed, same design).

Run (from anywhere; the tilt study's scripts folder is passed explicitly;
its cohort banks must exist: `python make_banks.py cohort` there first, ~8 min;
this script ~10 s; numpy, scipy, statsmodels 0.15):
    python check06_pm_vs_dl.py "<.../Ih Fit/dev/2026-10-26_tilt_sigma_b/scripts>"
Result of the run of 2026-10-28: check06.log (P1-P3 PASS).
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np

CFG = dict(seed=20261034, n_rep=4000, sigma_med=0.1, omegas=(0.0, 0.25, 0.5, 1.0),
           check04_off2_a=(25.1, 25.4, 25.7, 30.5))


def pool_log(s2, nu, floor, method, tc, combine_effects):
    """Option (c) on log sigma_b^2 with the between-cell estimator `method`
    ('dl' or 'pm'); returns (sigma_b^2 per cell, clipped Omega^2, raw tau2)."""
    bias, var = tc.log_chi2_moments(nu)
    y = np.log(np.maximum(s2, floor)) - bias
    raw = float(combine_effects(y, var, method_re=method).tau2)
    om2 = max(0.0, raw)
    w = 1.0 / (var + om2)
    mu = float(np.sum(w * y) / np.sum(w))
    return np.exp(mu + om2 / (om2 + var) * (y - mu)), om2, raw


def main(scripts_dir: Path):
    sys.path.insert(0, str(scripts_dir))
    import tilt_common as tc                                  # the tilt study's shared code
    import check04_pooling_options as c4                      # its cohort loader (bank paths)
    from statsmodels.stats.meta_analysis import combine_effects
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="statsmodels")
    t0 = time.time()
    cells = c4.load_cohort()
    S = np.array([c["S"] for c in cells])
    nu = np.array([c["Se"] for c in cells])
    floor = np.array([c["floor"] for c in cells])
    rng = np.random.default_rng(CFG["seed"])
    n_rep, C = CFG["n_rep"], len(cells)
    print("cells: %d (2-sweep %d, 6-sweep %d, 10-sweep %d); S_eff 2-sweep %.2f-%.2f"
          % (C, np.sum(S == 2), np.sum(S == 6), np.sum(S == 10), nu[S == 2].min(), nu[S == 2].max()))
    ok_p1, ok_p2, ok_p3 = True, True, True
    for omega, ref_a in zip(CFG["omegas"], CFG["check04_off2_a"]):
        sig = CFG["sigma_med"] * np.exp(omega * rng.standard_normal((n_rep, C)))
        s2 = np.empty((n_rep, C))
        for j, c in enumerate(cells):
            rows = rng.integers(0, c["d"].shape[0], n_rep)
            b = sig[:, j:j + 1] * rng.standard_normal((n_rep, c["S"]))
            s2[:, j] = tc.sigma_b2_from_pairs(c["d"][rows], c["s_of"], b, 4, c["V0"])
        est = {k: np.empty((n_rep, C)) for k in ("a", "dl", "pm")}
        om = {k: np.empty(n_rep) for k in ("dl", "pm")}
        neg_pm = 0
        for r in range(n_rep):
            est["a"][r] = np.sqrt(np.clip(s2[r], 0, None))
            for k in ("dl", "pm"):
                v, om[k][r], raw = pool_log(s2[r], nu, floor, k, tc, combine_effects)
                est[k][r] = np.sqrt(v)
                if k == "pm" and raw < 0:
                    neg_pm += 1
        ok_p1 &= neg_pm == 0
        print("\n== omega %.2f: Omega_hat^2 = 0 in DL %.0f %% / PM %.0f %% of cohorts; median Omega_hat/2 DL %.2f / PM %.2f;"
              " PM raw tau2 < 0 in %d cohorts"
              % (omega, 100 * np.mean(om["dl"] == 0), 100 * np.mean(om["pm"] == 0),
                 np.median(np.sqrt(om["dl"]) / 2), np.median(np.sqrt(om["pm"]) / 2), neg_pm))
        off = {}
        for grp, mask in (("2-sweep", S == 2), ("10-sweep", S == 10)):
            for k, name in (("a", "a: per cell"), ("dl", "c-DL"), ("pm", "c-PM")):
                ratio = est[k][:, mask] / sig[:, mask]
                off2 = 100 * np.mean((ratio < 0.5) | (ratio > 2))
                with np.errstate(divide="ignore"):
                    mal = np.median(np.abs(np.log(ratio)))
                rms = np.sqrt(np.mean((ratio - 1) ** 2))
                off[(grp, k)] = off2
                print("   %-8s %-12s outside x2 %5.1f %% | zero %4.1f %% | median |log ratio| %.2f | median ratio %.2f | RMS rel err %.3f"
                      % (grp, name, off2, 100 * np.mean(ratio == 0), mal, np.median(ratio), rms))
        d = abs(off[("2-sweep", "pm")] - off[("2-sweep", "dl")])
        ok_p2 &= d < 3.0
        ok_p3 &= abs(off[("2-sweep", "a")] - ref_a) < 2.0
        print("   P2 |PM - DL| on 2-sweep cells = %.1f points (< 3: %s); P3 (a) %.1f vs check04 %.1f (within 2: %s)"
              % (d, d < 3.0, off[("2-sweep", "a")], ref_a, abs(off[("2-sweep", "a")] - ref_a) < 2.0))
    print("\nP1 %s  P2 %s  P3 %s   (%.0f s)" % ("PASS" if ok_p1 else "FAIL", "PASS" if ok_p2 else "FAIL",
                                             "PASS" if ok_p3 else "FAIL", time.time() - t0))


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
