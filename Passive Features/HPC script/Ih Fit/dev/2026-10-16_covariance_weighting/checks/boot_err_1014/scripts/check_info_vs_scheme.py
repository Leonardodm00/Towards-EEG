"""Does the sampling (plug-in) error depend only on how much information the data carry? (check, 2026-10-14)

Two experiments on the Stage-7 law at theta_sim, statistic T = rho_hat(1 ms) (textbook autocorrelation, lag 50
samples at 50 kHz), as in check_boot_errors.py.

A. More information, same plug-in rule. Window-level bootstrap SE of the median of K windows, B = 2000
   (internal error about 0.02 x the true SE, check_boot_errors.py), for K = 6 and K = 24 windows per cell,
   built from the same 20 000 windows of 265 ms (3333 and 833 cells). Spread over cells of SE_B / SD_true(K),
   SD_true(K) = SD over cells of the cell medians.
B. Same data, wrong plug-in rule. The iid world: resample the n samples of ONE window with replacement (this
   destroys the autocorrelation T measures). SD(T*) over B = 200 replicates, for 10 windows, at n = 13 250
   (265 ms) and n = 53 000 (1.06 s), against the true SD(T) over M windows at each n.

Predictions, fixed BEFORE the first run:
  PA the SD over cells of SE_B / SD_true is at least 15 % smaller with K = 24 than with K = 6 (ratio < 0.85):
     more information, smaller sampling error.
  PB at both n, true SD(T) / median iid SD(T*) > 5, and the ratio at n = 53 000 is at least 0.8 x the ratio at
     n = 13 250: four times more data does not remove the error of the wrong plug-in rule.

Run:  python check_info_vs_scheme.py      (about 1 min; needs boot_model.py next to it)
"""
import numpy as np

import boot_model as bm

CONFIG = dict(
    dt=20e-6, lag=50,
    theta_sim=dict(sigma_f=0.0383, rho_f=0.179, sigma_s=0.0547, tau_s=0.03974),   # TD 3.5.3
    n_A=13_250, M_A=20_000, seed_A=2026101301, K_list=(6, 24), B_A=2000, seed_bootA=2026101440,
    n_list_B=(13_250, 53_000), M_B=2000, seed_B=2026101441, n_win_B=10, B_B=200, seed_bootB=2026101442,
)


# ---------------- A ----------------
def window_level_se(cells, B, rng, chunk=100):
    out = np.zeros(len(cells))
    K = cells.shape[1]
    for s in range(0, len(cells), chunk):
        v = cells[s:s + chunk]
        idx = rng.integers(0, K, size=(len(v), B, K))
        out[s:s + len(v)] = np.median(np.take_along_axis(v[:, None, :], idx, axis=2), axis=2).std(axis=1, ddof=1)
    return out


def part_a(cfg):
    T = bm.simulate_T(cfg["theta_sim"], cfg["dt"], cfg["n_A"], cfg["lag"], cfg["M_A"], cfg["seed_A"])
    rng = np.random.default_rng(cfg["seed_bootA"])
    spread = {}
    for K in cfg["K_list"]:
        cells = T[: (T.size // K) * K].reshape(-1, K)
        sd_true = np.median(cells, axis=1).std(ddof=1)
        r = window_level_se(cells, cfg["B_A"], rng) / sd_true
        q = np.quantile(r, [0.1, 0.5, 0.9])
        spread[K] = r.std(ddof=1)
        print("  K = %2d (%4d cells): SD_true %.4f; SE_B/SD_true median %.3f, 10%% %.3f, 90%% %.3f; SD over cells %.3f"
              % (K, len(cells), sd_true, q[1], q[0], q[2], spread[K]))
    ratio = spread[24] / spread[6]
    print("  PA spread K = 24 / K = 6 = %.2f" % ratio)
    return {"PA": bool(ratio < 0.85)}


# ---------------- B ----------------
def part_b(cfg):
    rng_b = np.random.default_rng(cfg["seed_bootB"])
    ratios = {}
    for i, n in enumerate(cfg["n_list_B"]):
        T = bm.simulate_T(cfg["theta_sim"], cfg["dt"], n, cfg["lag"], cfg["M_B"], cfg["seed_B"] + i, chunk=100)
        sd_true = T.std(ddof=1)
        rng_w = np.random.default_rng(cfg["seed_B"] + 100 + i)
        wins = bm.simulate_windows(cfg["theta_sim"], cfg["dt"], n, cfg["n_win_B"], rng_w)
        sd_iid = []
        for x in wins:
            Ts = np.array([bm.acf_tb(x[rng_b.integers(0, n, n)], cfg["lag"]) for _ in range(cfg["B_B"])])
            sd_iid.append(Ts.std(ddof=1))
        med_iid = np.median(sd_iid)
        ratios[n] = sd_true / med_iid
        print("  n = %6d: true SD(T) %.4f (M = %d); iid world SD(T*) median %.4f over %d windows (1/sqrt(n) = %.4f); "
              "ratio %.1f" % (n, sd_true, cfg["M_B"], med_iid, cfg["n_win_B"], 1 / np.sqrt(n), ratios[n]))
    n0, n1 = cfg["n_list_B"]
    ok = bool(min(ratios.values()) > 5 and ratios[n1] >= 0.8 * ratios[n0])
    print("  PB ratio at n = %d / ratio at n = %d = %.2f" % (n1, n0, ratios[n1] / ratios[n0]))
    return {"PB": ok}


def main():
    cfg = CONFIG
    ok = {}
    print("== A. more information (K windows), same plug-in rule (window-level resampling), B = %d" % cfg["B_A"])
    ok.update(part_a(cfg))
    print("\n== B. same data, wrong plug-in rule (iid resampling of one window's samples), B = %d" % cfg["B_B"])
    ok.update(part_b(cfg))
    print("\nALL PREDICTIONS:", ok)


if __name__ == "__main__":
    main()
