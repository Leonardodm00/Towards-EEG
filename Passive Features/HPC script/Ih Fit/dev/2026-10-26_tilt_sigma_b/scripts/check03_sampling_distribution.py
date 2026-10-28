"""check03_sampling_distribution.py -- how precisely (T1) fixes sigma_b(c)
for the L3_exc sweep counts, what the error is made of, and how it compares
with the best one could do from the same sweeps.

Data: the reference banks of make_banks.py (theta_cf, t_SS = 80 ms, random
drops: 2 sweeps K 40 and K 63, 6 sweeps K 199, 10 sweeps K 348). The tilt
b_s = sigma_b xi_s is added exactly to every pair difference (check01 (i)).

Predictions, fixed before running:
 (b) at sigma_b = 0.1 mV/s, sigma_b_hat^2 / sigma_b^2 has mean 1 [within
     3 SE]; the spread of sigma_b_hat / sigma_b follows sqrt(chi2_Seff/Seff),
     S_eff = (sum N_s)^2 / sum N_s^2, plus a small noise term: the variance
     of sigma_b_hat^2/sigma_b^2 exceeds 2/S_eff by less than 15 % at 2 sweeps;
     the three terms (tilt draws, cross, noise) add to the total.
 (c) the noise-only floor: SD of sigma_b_hat^2 at sigma_b = 0 (no prediction
     of its size beyond "below sigma_b^2 = 0.01 (mV/s)^2"); the clip rate.
 (e) the two-lag estimator 2[(V_hat(4P) - V_hat(2P)) - (V0(4P) - V0(2P))] /
     (h4^2 - h2^2), which needs no theta_hat, is unbiased under the tilt model
     too but wider than (T1).
 (f) per-sweep least-squares slopes of the window means on time (polarity
     intercepts) recover each realised b_s with an error far below sigma_b
     (RMS < 0.01 mV/s); their mean square has the spread of chi2_S/S, the
     least an estimator of sigma_b^2 from S exactly known draws can have.
 (d) theta_hat's error moves V(h*|theta_hat, 0): linearised at theta_cf with
     the SD(u_hat) of check C of 2026-10-23 (fixture cell of smoke_noise_fit:
     2 sweeps, K 63, 5 LS windows) and correlation -1/0/+1 between log sigma_s
     and log tau_s; the SD of the induced error in sigma_b_hat^2 is reported
     against the floor of (c).

Run:  python make_banks.py ref  (first), then  python check03_sampling_distribution.py   (~1 min)
"""
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import chi2

import tilt_common as tc

CFG = dict(
    seed=20261027, sigma_b=0.1, sigma_b_grid=(0.025, 0.05, 0.1, 0.2),
    banks=("2sw_K40", "2sw_K63", "6sw_K199", "10sw_K348"),
    bank_dir=Path(__file__).resolve().parent.parent / "results" / "banks",
    q=(0.025, 0.5, 0.975),
    # (d): SD(u_hat) at theta_cf, fixture cell, from check C of check_findings_2026-10-23.py
    sd_u_checkC=(0.005, 0.007, 0.082, 0.206),
)


def load(name):
    z = np.load(CFG["bank_dir"] / ("ref_%s.npz" % name))
    lay = tc.Layout(name, z["sweep"], z["pos"], int(z["n_sweeps"]))
    return z["M"], lay


def slopes_per_sweep(M: np.ndarray, lay: tc.Layout) -> np.ndarray:
    """OLS slope (mV/s) of the window means on window-centre time, per sweep,
    with one intercept per polarity: (n_cells, S). numpy.linalg.lstsq."""
    c = tc.window_centres(lay)
    dep = (lay.pos % 2 == 0)
    out = np.empty((M.shape[0], lay.n_sweeps))
    for s in range(lay.n_sweeps):
        k = np.flatnonzero(lay.sweep == s)
        X = np.column_stack([dep[k].astype(float), (~dep[k]).astype(float), c[k]])
        coef, *_ = np.linalg.lstsq(X, M[:, k].T, rcond=None)
        out[:, s] = coef[2]
    return out


def part_b_c_e_f(rng):
    h4, h2 = tc.h_of(4), tc.h_of(2)
    V04, V02 = tc.v_model(tc.LAW_CF, 4), tc.v_model(tc.LAW_CF, 2)
    print("V(h|theta_cf,0), 120 ms windows: %.6e mV^2 at 2 periods, %.6e at 4 periods (difference %.1e)" % (V02, V04, V04 - V02))
    rows = {}
    for name in CFG["banks"]:
        M0, lay = load(name)
        n, S = M0.shape[0], lay.n_sweeps
        d4, s4 = tc.pairs_from_means(M0, lay, 4)
        d2, s2 = tc.pairs_from_means(M0, lay, 2)
        npp = np.bincount(s4, minlength=S)
        lam = npp / npp.sum()
        Se = tc.s_eff(npp)
        N = int(npp.sum())
        sb = CFG["sigma_b"]
        xi = rng.standard_normal((n, S))
        b = sb * xi
        est = tc.sigma_b2_from_pairs(d4, s4, b, 4, V04)                 # unclipped (T1)
        ratio = np.sqrt(np.clip(est, 0, None)) / sb
        qs = np.quantile(ratio, CFG["q"])
        mean_r, se_r = est.mean() / sb ** 2, est.std(ddof=1) / np.sqrt(n) / sb ** 2
        ref = tc.chi2_quantiles(Se, CFG["q"])
        tilt_only = np.sqrt((xi ** 2 * lam).sum(axis=1))                 # sigma_b_hat/sigma_b with no noise
        q_tilt = np.quantile(tilt_only, CFG["q"])
        # decomposition of sigma_b_hat^2 / sigma_b^2
        T = (xi ** 2 * lam).sum(axis=1)
        Q = (np.mean(d4 ** 2, axis=1) - 2 * V04) / h4 ** 2 / sb ** 2
        C = est / sb ** 2 - T - Q
        v_tot, v_T, v_C, v_Q = est.var(ddof=1) / sb ** 4, T.var(ddof=1), C.var(ddof=1), Q.var(ddof=1)
        a_T, a_C, a_Q = 2 * np.sum(lam ** 2), 8 * V04 / (N * h4 ** 2 * sb ** 2), 8 * V04 ** 2 / (N * h4 ** 4 * sb ** 4)
        print("\n== %s: K %d, pairs per sweep %s (N %d), S_eff %.2f, %d cells" % (name, lay.K, npp.tolist(), N, Se, n))
        print("(b) sigma_b %.2f mV/s: sigma_b_hat/sigma_b 2.5/50/97.5 %% = %.2f / %.2f / %.2f; clipped at 0 in %.1f %%;"
              " mean sigma_b_hat^2/sigma_b^2 = %.3f (SE %.3f) -> %s"
              % (sb, qs[0], qs[1], qs[2], 100 * np.mean(est <= 0), mean_r, se_r, "PASS" if abs(mean_r - 1) < 3 * se_r else "FAIL"))
        print("    references: sqrt(chi2_Seff/Seff) %.2f / %.2f / %.2f; tilt draws alone (exact weights) %.2f / %.2f / %.2f"
              % (ref[0], ref[1], ref[2], q_tilt[0], q_tilt[1], q_tilt[2]))
        print("    variance of sigma_b_hat^2/sigma_b^2: total %.4f = tilt %.4f + cross %.4f + noise %.4f (+ covariances %.4f);"
              " approximations 2 sum lam^2 = %.4f, 8V0/(N h^2 sb^2) = %.4f, 8V0^2/(N h^4 sb^4) = %.4f; excess over 2/S_eff: %.1f %% -> %s"
              % (v_tot, v_T, v_C, v_Q, v_tot - v_T - v_C - v_Q, a_T, a_C, a_Q, 100 * (v_tot / a_T - 1),
                 "PASS" if (S > 2 or v_tot / a_T - 1 < 0.15) else "FAIL"))
        # (c) floor and clip rates
        Q0 = (np.mean(d4 ** 2, axis=1) - 2 * V04) / h4 ** 2                 # sigma_b = 0, (mV/s)^2
        sdQ = Q0.std(ddof=1)
        clip = []
        for sbg in CFG["sigma_b_grid"]:
            e = tc.sigma_b2_from_pairs(d4, s4, sbg * xi, 4, V04)
            clip.append(100 * np.mean(e <= 0))
        print("(c) floor: SD of sigma_b_hat^2 at sigma_b = 0: %.5f (mV/s)^2 (sqrt %.4f mV/s); 95th pct %.5f (sqrt %.4f mV/s);"
              " clipped at sigma_b = %s mV/s: %s %%"
              % (sdQ, np.sqrt(sdQ), np.quantile(Q0, 0.95), np.sqrt(max(np.quantile(Q0, 0.95), 0)),
                 list(CFG["sigma_b_grid"]), " / ".join("%.1f" % x for x in clip)))
        # (e) two-lag estimator
        npp2 = np.bincount(s2, minlength=S)
        D4 = d4 + b[:, s4] * h4
        D2 = d2 + b[:, s2] * h2
        est2 = 2 * ((0.5 * np.mean(D4 ** 2, axis=1) - 0.5 * np.mean(D2 ** 2, axis=1)) - (V04 - V02)) / (h4 ** 2 - h2 ** 2)
        q2 = np.quantile(np.sqrt(np.clip(est2, 0, None)) / sb, CFG["q"])
        print("(e) two-lag estimator (pairs at 2 periods %s): mean ratio %.3f (SE %.3f); sigma_b_hat/sigma_b %.2f / %.2f / %.2f;"
              " clipped %.1f %%; SD of its sigma_b^2 at sigma_b = 0: %.5f (mV/s)^2"
              % (npp2.tolist(), est2.mean() / sb ** 2, est2.std(ddof=1) / np.sqrt(n) / sb ** 2, q2[0], q2[1], q2[2],
                 100 * np.mean(est2 <= 0),
                 (2 * ((0.5 * np.mean(d4 ** 2, 1) - 0.5 * np.mean(d2 ** 2, 1)) - (V04 - V02)) / (h4 ** 2 - h2 ** 2)).std(ddof=1)))
        # (f) per-sweep slopes
        c = tc.window_centres(lay)
        Mb = M0 + (b[:, lay.sweep] * c[None, :])
        bh = slopes_per_sweep(Mb, lay)
        err = bh - b
        ms = np.mean(bh ** 2, axis=1)
        qf = np.quantile(np.sqrt(ms) / sb, CFG["q"])
        qS = tc.chi2_quantiles(S, CFG["q"])
        print("(f) per-sweep slopes: RMS(b_hat_s - b_s) = %.4f mV/s -> %s; sqrt(mean_s b_hat_s^2)/sigma_b %.2f / %.2f / %.2f"
              " against sqrt(chi2_S/S), S = %d: %.2f / %.2f / %.2f"
              % (np.sqrt(np.mean(err ** 2)), "PASS" if np.sqrt(np.mean(err ** 2)) < 0.01 else "FAIL",
                 qf[0], qf[1], qf[2], S, qS[0], qS[1], qS[2]))
        rows[name] = dict(ratio=ratio, Se=Se, S=S, sdQ=sdQ, clip=clip, est=est)
    np.savez(CFG["bank_dir"].parent / "check03_ratios.npz",
             **{k + "_ratio": v["ratio"] for k, v in rows.items()},
             **{k + "_Se": v["Se"] for k, v in rows.items()},
             **{k + "_sdQ": v["sdQ"] for k, v in rows.items()})
    return rows


def part_d():
    """theta_hat error propagated into V(h*|theta_hat, 0), linearised. The SDs
    of u_hat = (log sigma_f, rho_f, log sigma_s, log tau_s) are those of check C
    of check_findings_2026-10-23.py (efficient bound at theta_cf on the fixture
    cell of smoke_noise_fit: 2 SS sweeps, K 63, 5 LS windows); their correlations
    were not printed then, so the induced SD is given for correlation -1, 0, +1
    between log sigma_s and log tau_s (sigma_f, rho_f taken independent). The
    full re-computation (Jacobian and Sigma_hat, ~3 min) was not run here."""
    sd_u = np.array(CFG["sd_u_checkC"])
    nl = tc.R.nl
    def logV0(u):
        law = nl.NoiseLaw(float(np.exp(u[0])), float(u[1]), float(np.exp(u[2])), float(np.exp(u[3])))
        return np.log(tc.v_model(law, 4))
    u0 = np.array([np.log(tc.LAW_CF.sigma_f), tc.LAW_CF.rho_f, np.log(tc.LAW_CF.sigma_s), np.log(tc.LAW_CF.tau_s)])
    g = np.empty(4)
    for i in range(4):
        up, um = u0.copy(), u0.copy(); up[i] += 1e-5; um[i] -= 1e-5
        g[i] = (logV0(up) - logV0(um)) / 2e-5
    V0 = float(np.exp(logV0(u0)))
    h = tc.h_of(4)
    out = {}
    for rho in (-1.0, 0.0, 1.0):
        C = np.diag(sd_u ** 2)
        C[2, 3] = C[3, 2] = rho * sd_u[2] * sd_u[3]
        sd_log = float(np.sqrt(g @ C @ g))
        out[rho] = (sd_log, 2 * V0 * sd_log / h ** 2)
    print("\n(d) d log V(h*|theta,0) / d u at theta_cf (120 ms windows) = %s  [log sigma_f, rho_f, log sigma_s, log tau_s]"
          % np.array2string(g, precision=3))
    for rho, (sl, s2) in out.items():
        print("    corr(log sigma_s, log tau_s) = %+.0f: SD(log V0_hat) = %.3f; induced SD of sigma_b_hat^2 = %.5f (mV/s)^2 (sqrt %.4f mV/s)"
              % (rho, sl, s2, np.sqrt(s2)))
    np.savez(CFG["bank_dir"].parent / "check03_theta_err.npz", sd_logV0=out[0.0][0], sd_logV0_hi=out[1.0][0],
             sd_logV0_lo=out[-1.0][0], g=g)


def main(which="bd"):
    t0 = time.time()
    rng = np.random.default_rng(CFG["seed"])
    if "b" in which:
        part_b_c_e_f(rng)
    if "d" in which:
        part_d()
    print("\n(%.0f s)" % (time.time() - t0))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "bd")
