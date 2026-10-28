"""check01_tilt_identities.py -- the identities the estimator (T1) rests on,
checked on simulated cells with the pipeline's own functions.

Predictions, fixed before running (tolerances in brackets):
 (i)   With the same unit draws, every same-polarity pair difference of
       enlarged-window means with the tilt equals the one without it plus
       b_s h, h = 4 P Delta_t = 0.802 s  [max abs error < 1e-12 mV].
 (ii)  The per-polarity mean-window subtraction (D-051.1) cancels in every
       same-polarity difference: V_hat from the subtracted windows equals V_hat
       from the raw windows [relative difference < 1e-9]; and V_hat computed
       from the pair list here equals noise_cell_stats.window_mean_semivariogram
       [relative difference < 1e-12, same N_pairs].
 (iii) Under the stationary law alone (sigma_b = 0) the mean of V_hat over
       cells equals V(h* | theta_cf, 0) of noise_law.semivariogram
       [within 3 Monte Carlo SE].
 (iv)  Given the realised tilts, the mean over noise draws of sigma_b_hat^2
       equals sum_s (N_s/N) b_s^2 [within 3 Monte Carlo SE], i.e. (T1) reads
       the realised tilts' pair-weighted mean square, not sigma_b^2 itself.
 (v)   A line of slope b over a window of duration T adds b^2 T^2 / 12 to
       the window's demeaned variance (discrete: b^2 dt^2 (m^2 - 1)/12)
       [exact to 1e-12 relative].

Run:  python check01_tilt_identities.py      (about 1 min)
"""
import time

import numpy as np

import tilt_common as tc

CFG = dict(seed=20261026, layout=(2, 40), n_cells_iii=1500, n_noise_iv=1500, sigma_b=0.1)


def main():
    t0 = time.time()
    rng = np.random.default_rng(CFG["seed"])
    lay = tc.build_layout(*CFG["layout"], rng)
    lag = tc.CFG["lag_periods"]
    h = tc.h_of(lag)
    a, b_idx, s_of = tc.pair_index(lay, lag)
    print("layout %s: K = %d, pairs per sweep %s, h* = %.4f s, window %d samples (%.1f ms)"
          % (lay.name, lay.K, tc.pairs_per_sweep(lay, lag).tolist(), h, tc.M_WIN, tc.M_WIN * tc.DT * 1e3))

    # (i) and (ii): one cell, same unit draws with and without the tilt
    v0, units = tc.simulate_windows(tc.LAW_CF, lay, rng, 0.0)
    v1, _ = tc.simulate_windows(tc.LAW_CF, lay, rng, CFG["sigma_b"], units=units)
    m0, m1 = tc.window_means(v0, lay), tc.window_means(v1, lay)
    d0, d1 = m0[b_idx] - m0[a], m1[b_idx] - m1[a]
    bs = CFG["sigma_b"] * units.tilt
    err = np.max(np.abs(d1 - (d0 + bs[s_of] * h)))
    print("(i)   max |D_with - (D_without + b_s h)| = %.2e mV  (b_s = %s mV/s)  -> %s"
          % (err, np.array2string(bs, precision=4), "PASS" if err < 1e-12 else "FAIL"))
    ss = tc.make_ss(v1, lay)
    X, _ = tc.R.nw.enlarged_windows(ss, tc.CFG["t_ss_ms"])
    Xt, _, _ = tc.R.nw.subtract_mean_response(X, ss.polarity_is_dep)
    V_sub, N_sub = tc.R.ncs.window_mean_semivariogram(Xt, ss, lag)
    V_raw, N_raw = tc.R.ncs.window_mean_semivariogram(X, ss, lag)
    V_here = 0.5 * np.mean(d1 ** 2)
    r1 = abs(V_sub - V_raw) / V_sub
    r2 = abs(V_here - V_sub) / V_sub
    print("(ii)  V_hat subtracted %.6e, raw %.6e (rel diff %.1e); pair list here %.6e (rel diff %.1e); N %d / %d / %d  -> %s"
          % (V_sub, V_raw, r1, V_here, r2, N_sub, N_raw, d1.size,
             "PASS" if (r1 < 1e-9 and r2 < 1e-12 and N_sub == N_raw == d1.size) else "FAIL"))

    # (iii) mean V_hat under the stationary law alone against the exact V(h* | theta, 0)
    bank = tc.noise_bank(tc.LAW_CF, lay, CFG["n_cells_iii"], rng)[lag][0]
    Vh = 0.5 * np.mean(bank ** 2, axis=1)
    V0 = tc.v_model(tc.LAW_CF, lag)
    se = Vh.std(ddof=1) / np.sqrt(Vh.size)
    z = (Vh.mean() - V0) / se
    print("(iii) mean V_hat (sigma_b = 0, %d cells) = %.6e mV^2 +/- %.1e (SE); V(h*|theta_cf,0) = %.6e; z = %+.2f  -> %s"
          % (Vh.size, Vh.mean(), se, V0, z, "PASS" if abs(z) < 3 else "FAIL"))
    print("      noise floor of (T1) in sigma_b^2 units, 2 V0 / h*^2 = %.5f (mV/s)^2, i.e. sqrt = %.4f mV/s"
          % (2 * V0 / h ** 2, np.sqrt(2 * V0 / h ** 2)))

    # (iv) conditional on fixed realised tilts, average over noise draws
    b_fixed = np.array([0.13, -0.04])                       # mV/s, one draw per sweep (chosen)
    npp = tc.pairs_per_sweep(lay, lag)
    target = float(np.sum(npp / npp.sum() * b_fixed ** 2))
    s2 = tc.sigma_b2_from_pairs(bank[:CFG["n_noise_iv"]], s_of, np.tile(b_fixed, (CFG["n_noise_iv"], 1)), lag, V0)
    se4 = s2.std(ddof=1) / np.sqrt(s2.size)
    z4 = (s2.mean() - target) / se4
    print("(iv)  b = %s mV/s: mean sigma_b_hat^2 over %d noise draws = %.6f +/- %.6f; sum (N_s/N) b_s^2 = %.6f; z = %+.2f  -> %s"
          % (b_fixed.tolist(), s2.size, s2.mean(), se4, target, z4, "PASS" if abs(z4) < 3 else "FAIL"))
    print("      (sigma_b^2 itself would be whatever law drew these b: the estimator cannot see it beyond the S draws)")

    # (v) the tilt's share of a window's demeaned variance
    m = tc.M_WIN
    t = np.arange(m) * tc.DT
    line = CFG["sigma_b"] * t
    var_line = np.var(line)                                   # ddof 0, the demeaned mean square
    pred = CFG["sigma_b"] ** 2 * tc.DT ** 2 * (m ** 2 - 1) / 12.0
    win_var = tc.LAW_CF.sigma_f ** 2 + tc.LAW_CF.sigma_s ** 2 * float(tc.R.nl.visible_share(m * tc.DT / tc.LAW_CF.tau_s))
    print("(v)   sigma_b = %.2f mV/s over %.0f ms: demeaned variance of the line %.4e mV^2 (pred %.4e, rel err %.1e) -> %s;"
          " share of a window's expected demeaned variance under theta_cf (%.4e mV^2): %.2f %%"
          % (CFG["sigma_b"], m * tc.DT * 1e3, var_line, pred, abs(var_line - pred) / pred,
             "PASS" if abs(var_line - pred) / pred < 1e-12 else "FAIL", win_var, 100 * var_line / win_var))
    print("(%.0f s)" % (time.time() - t0))


if __name__ == "__main__":
    main()
