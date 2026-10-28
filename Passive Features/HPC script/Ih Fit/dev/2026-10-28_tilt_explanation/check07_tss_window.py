"""check07_tss_window.py -- standalone check written for the chat explanation of
sigma_b(c) on 2026-10-28 (not pipeline code, not in the repository).

Question: D-095 may start the SS noise window later than the 80 ms of the
tilt study (the handoff of 2026-10-28, section 4: about 130 ms for the RC
fixture at kappa_tr = 0.1), so the enlarged windows shorten from 120 ms to
about 70 ms. What does that do to (T1) for a 2-sweep cell?

Truth (chosen): the stationary law theta_cf, the two 2-sweep reference
layouts of make_banks.py (the same layout draws: SeedSequence([20261026, 1]),
spawn 0 and 1), sigma_b = 0 (noise floor) and 0.1 mV/s; (T1) at the true law.
Each simulated cell is cut at t_SS = 80 and 130 ms from the SAME traces
(common random numbers), so the two columns differ by the window only.
Sanity: at 80 ms the first 50 cells reproduce the stored reference bank
bit for bit (same seeds, same code).

Predictions, fixed before running:
 P1  the floor rises: the 95th percentile of sigma_b_hat under sigma_b = 0
     is larger at 130 ms than at 80 ms, by less than 30 % (V0 grows about 35 %,
     the floor about as its square root).
 P2  at sigma_b = 0.1 mV/s the 2.5 / 50 / 97.5 % points of sigma_b_hat /
     sigma_b move by less than 0.10 (the spread is the two tilt draws').

Run (the tilt study's scripts folder is passed explicitly; its reference banks
must exist for the sanity line: `python make_banks.py ref` there first, ~7 min;
this script ~3 min; numpy, scipy):
    python check07_tss_window.py "<.../Ih Fit/dev/2026-10-26_tilt_sigma_b/scripts>"
Result of the run of 2026-10-28: check07.log (P1, P2 PASS; sanity PASS).
"""
import sys
import time
from pathlib import Path

import numpy as np

CFG = dict(bank_seed=20261026, layouts=(("2sw_K40", 2, 40), ("2sw_K63", 2, 63)), n_cells=2000,
           t_ss_ms=(80.0, 130.0), sigma_b=0.1, n_tilt_draws=4, tilt_seed=20261035, sanity_cells=50)


def set_window(tc, t_ss_ms):
    """Point the tilt study's module at a window start (its globals J0, M_WIN)."""
    tc.CFG["t_ss_ms"] = t_ss_ms
    tc.J0 = tc.PRE_N + int(round(t_ss_ms * 1e-3 * tc.FS))
    tc.M_WIN = tc.N_W - tc.J0


def main(scripts_dir: Path):
    sys.path.insert(0, str(scripts_dir))
    import tilt_common as tc
    t0 = time.time()
    ss = np.random.SeedSequence([CFG["bank_seed"], 1]).spawn(4)
    ok1, ok2 = True, True
    for (name, S, K), s in zip(CFG["layouts"], ss[:2]):
        rng = np.random.default_rng(s)
        lay = tc.build_layout(S, K, rng, name)                      # same draw as make_banks ref
        M = {t: np.empty((CFG["n_cells"], lay.K)) for t in CFG["t_ss_ms"]}
        for c in range(CFG["n_cells"]):
            v, _ = tc.simulate_windows(tc.LAW_CF, lay, rng, 0.0)     # same rng stream as make_banks
            for t in CFG["t_ss_ms"]:
                set_window(tc, t)
                M[t][c] = tc.window_means(v, lay)
        ref = np.load(scripts_dir.parent / "results" / "banks" / ("ref_%s.npz" % name))["M"]
        same = np.array_equal(ref[:CFG["sanity_cells"]], M[80.0][:CFG["sanity_cells"]])
        print("%s: K %d, sanity (80 ms reproduces the stored bank, %d cells): %s"
              % (name, lay.K, CFG["sanity_cells"], "PASS" if same else "FAIL"))
        q = {}
        trng = np.random.default_rng(CFG["tilt_seed"])
        b_unit = trng.standard_normal((CFG["n_tilt_draws"] * CFG["n_cells"], S))
        for t in CFG["t_ss_ms"]:
            set_window(tc, t)
            d, s_of = tc.pairs_from_means(M[t], lay, 4)
            V0 = tc.v_model(tc.LAW_CF, 4)
            s2_0 = tc.sigma_b2_from_pairs(d, s_of, np.zeros((d.shape[0], S)), 4, V0)
            d_rep = np.tile(d, (CFG["n_tilt_draws"], 1))
            s2 = tc.sigma_b2_from_pairs(d_rep, s_of, CFG["sigma_b"] * b_unit, 4, V0)
            r = np.sqrt(np.clip(s2, 0, None)) / CFG["sigma_b"]
            q[t] = dict(V0=V0, m=tc.M_WIN, floor95=np.sqrt(max(np.quantile(s2_0, 0.95), 0)),
                        sd0=s2_0.std(ddof=1), qs=np.quantile(r, (0.025, 0.5, 0.975)), clip=np.mean(s2 <= 0),
                        mean=np.mean(s2) / CFG["sigma_b"] ** 2)
            print("   t_SS %5.1f ms (window %5.1f ms): V0 = %.4e mV^2 | floor: SD of sigma_b_hat^2 at 0 = %.5f (mV/s)^2,"
                  " 95th pct of sigma_b_hat = %.4f mV/s | at sigma_b 0.1: ratio 2.5/50/97.5 %% = %.2f / %.2f / %.2f,"
                  " clipped %.1f %%, mean sigma_b_hat^2/sigma_b^2 = %.3f"
                  % (t, tc.M_WIN * tc.DT * 1e3, V0, q[t]["sd0"], q[t]["floor95"], *q[t]["qs"], 100 * q[t]["clip"], q[t]["mean"]))
        rise = q[130.0]["floor95"] / q[80.0]["floor95"] - 1
        move = np.max(np.abs(q[130.0]["qs"] - q[80.0]["qs"]))
        ok1 &= 0 < rise < 0.30
        ok2 &= move < 0.10
        print("   P1 floor rise %+.1f %% (0 to 30 %%: %s); V0 rise %+.1f %%; P2 largest quantile shift %.3f (< 0.10: %s)"
              % (100 * rise, 0 < rise < 0.30, 100 * (q[130.0]["V0"] / q[80.0]["V0"] - 1), move, move < 0.10))
    print("\nP1 %s  P2 %s   (%.0f s)" % ("PASS" if ok1 else "FAIL", "PASS" if ok2 else "FAIL", time.time() - t0))


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
