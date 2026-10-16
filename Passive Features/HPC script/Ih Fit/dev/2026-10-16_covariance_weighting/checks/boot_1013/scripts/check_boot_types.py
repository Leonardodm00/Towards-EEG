"""Which bootstrap world reproduces the sampling law of the Stage-7 statistic? (check, 2026-10-13)

Statistic: T = rho_hat(1 ms), textbook sample autocorrelation (D-028.1) of one 265 ms Long Square
pre-window (n = 13 250 samples at 50 kHz, lag L = 50 samples), under the cohort law theta_sim of the
topic doc (section 3.5.3). The true sampling law of T is estimated by Monte Carlo over M fresh windows
(E0). Bootstrap worlds are then built from ONE window at a time (E1) or from the K = 6 windows of
one simulated cell (E2), and compared with E0.

Predictions, fixed BEFORE the first run (tolerances included):
  P1  iid world (single samples resampled): for every window, |mean T*| < 0.01 and
      SD(T*) in [0.85, 1.15] / sqrt(n)  (sample ACF of iid data: mean ~ -1/n, SD ~ 1/sqrt(n);
      textbook, from memory).
  P2  circular-block world (CBB, block b samples, n = k b exactly): about the window's own mean xbar,
      E*[S0~*] = S0 and E*[SL~*] = (1 - L/b)_+ (S_L + W_L), W_L the L wrap-around products
      (own derivation: pairs inside a block keep their product, pairs across a join are products of
      independent uniform draws, whose mean is 0). Monte Carlo z-scores |z| < 4 for every (window, b).
  P3  stationary world (SB, mean block b, restart probability p = 1/b):
      E*[SL~*] = (1 - L/n)(1 - p)^L (S_L + W_L), E*[S0~*] = S0; |z| < 4 for every (window, b).
  P4  window-level world, mean of K = 6: median over cells of SE_boot(mean)/(SD(T)/sqrt(6)) in
      [0.78, 0.92]; its 10 % quantile in [0.42, 0.62] and 90 % quantile in [1.10, 1.40]
      (normal theory: sqrt((K-1)/K) * s/sigma, s/sigma ~ sqrt(chi2_5/5): 0.85, 0.52, 1.24).
  P5  window-level world, median of K = 6: at most 21 distinct bootstrap medians in every cell
      (the median of 6 resampled values is one value or the mean of two: 6 + 15 cases).
  P6  E0 reproduces the bias doc: bias of T at theta_sim within 3 combined SE of the bias doc's
      -0.1082 (MC SE 0.0021, M = 3200), at theta_cf within 3 combined SE of -0.1027 (MC SE 0.0014,
      M = 6400), both read from bias_doc/scripts/logs/check01.txt.
      [corrected 2026-10-13 after the first run: that run used the document's rounded values
      -0.108 (0.002) and -0.103 (0.001) and passed with z = +0.81 and +2.98; the unrounded reference
      values from the log are used from the second run on. Tolerance unchanged.]
Measured without prediction: SD(T*) and mean(T*) - T(x) in the block worlds (MBB, CBB, SB) against
block length; the window-level SE and percentile-interval coverage for the median of K = 6.

Parametric world: here P_hat = p(. | theta_hat, n) with theta_hat HELD at a cohort law (theta_sim,
i.e. the truth, and theta_cf, another fit of the same cohort), not fitted per window or per cell; its
law of T* is then E0 at that theta_hat, and the plug-in error of a per-cell theta_hat is not included.

Run:  python check_boot_types.py            (about 3-6 min on 2 cores; writes ../results, ../logs)
"""
import time
from pathlib import Path

import numpy as np
from statsmodels.tsa.stattools import acf as sm_acf

import boot_model as bm
import boot_worlds as bw

# ---------------- configuration (all parameters here) ----------------
CONFIG = dict(
    dt=20e-6,                       # s, 50 kHz (LS pre-window sampling)
    n=13_250,                       # samples, 265 ms
    lag=50,                         # samples, 1 ms
    theta_sim=dict(sigma_f=0.0383, rho_f=0.179, sigma_s=0.0547, tau_s=0.03974),   # TD 3.5.3
    theta_cf=dict(sigma_f=0.0380, rho_f=0.175, sigma_s=0.0508, tau_s=0.0366),     # TD 3.5.3
    M_true=20_000,                  # windows for the true law (E0), per theta
    seed_true_sim=2026101301,
    seed_true_cf=2026101302,
    n_win=40,                       # windows each turned into a bootstrap world (E1)
    seed_win=2026101303,
    seed_boot=2026101304,           # root of the SeedSequence for all resampling
    B=500,                          # replicates per (window, scheme, b)
    blocks=[50, 125, 250, 530, 1325, 2650, 6625],   # samples; all divide n (1 ... 132.5 ms)
    schemes_block=["mbb", "cbb", "sb"],
    K=6,                            # windows per cell (5-6 LS pre-windows per cohort cell, TD 3.2.1)
    B_cell=2000,                    # replicates per cell (E2)
    seed_cell=2026101305,
    level=0.95,
)
HERE = Path(__file__).resolve().parent
OUT_RES = HERE.parent / "results" / "boot_types.npz"


def true_law(cfg):
    """E0: Monte Carlo law of T at theta_sim and theta_cf."""
    out = {}
    for key, seed in (("sim", cfg["seed_true_sim"]), ("cf", cfg["seed_true_cf"])):
        th = cfg["theta_" + key]
        T = bm.simulate_T(th, cfg["dt"], cfg["n"], cfg["lag"], cfg["M_true"], seed)
        g = bm.true_acf(th, cfg["dt"], cfg["lag"])
        out[key] = dict(T=T, g=g, mean=T.mean(), sd=T.std(ddof=1), med=np.median(T),
                        bias=T.mean() - g, se_bias=T.std(ddof=1) / np.sqrt(T.size))
    return out


def window_worlds(cfg):
    """E1: for each of n_win windows, iid and block worlds; replicate summaries per (scheme, b)."""
    rng = np.random.default_rng(cfg["seed_win"])
    X = bm.simulate_windows(cfg["theta_sim"], cfg["dt"], cfg["n"], cfg["n_win"], rng)
    L, n, B = cfg["lag"], cfg["n"], cfg["B"]
    # smoke: our T equals statsmodels' textbook acf on three windows
    for x in X[:3]:
        assert abs(bm.acf_tb(x, L) - sm_acf(x, adjusted=False, nlags=L, fft=False)[L]) < 1e-12
    T_obs = bm.acf_tb(X, L)
    S0 = np.array([bm.lag_sums_about(x, x.mean(), L)[0] for x in X])
    SL = np.array([bm.lag_sums_about(x, x.mean(), L)[1] for x in X])
    WL = np.array([bm.circular_wrap_sum(x, L) for x in X])
    labels = [("iid", 1)] + [(s, b) for s in cfg["schemes_block"] for b in cfg["blocks"]]
    seeds = np.random.SeedSequence(cfg["seed_boot"]).spawn(len(X) * len(labels))
    shape = (len(X), len(labels))
    res = {k: np.zeros(shape) for k in ("Tm", "Tsd", "S0m", "S0sd", "SLm", "SLsd")}
    k = 0
    for i, x in enumerate(X):
        xbar = x.mean()
        for j, (scheme, b) in enumerate(labels):
            xs = bw.resample_window(x, scheme, b, B, seeds[k]); k += 1
            Ts = bm.acf_tb(xs, L)
            s0, sl = bm.lag_sums_about(xs, xbar, L)
            res["Tm"][i, j], res["Tsd"][i, j] = Ts.mean(), Ts.std(ddof=1)
            res["S0m"][i, j], res["S0sd"][i, j] = s0.mean(), s0.std(ddof=1)
            res["SLm"][i, j], res["SLsd"][i, j] = sl.mean(), sl.std(ddof=1)
    return dict(labels=labels, T_obs=T_obs, S0=S0, SL=SL, WL=WL, **res)


def join_predictions(cfg, w):
    """Predicted E*[SL~*] (P2 for cbb, P3 for sb) and E*[S0~*] = S0, per (window, label); NaN otherwise."""
    L, n = cfg["lag"], cfg["n"]
    predL = np.full(w["Tm"].shape, np.nan)
    for j, (scheme, b) in enumerate(w["labels"]):
        if scheme == "cbb":
            predL[:, j] = max(0.0, 1 - L / b) * (w["SL"] + w["WL"])
        elif scheme == "sb":
            predL[:, j] = (1 - L / n) * (1 - 1 / b) ** L * (w["SL"] + w["WL"])
    return predL


def cell_worlds(cfg, T_sim):
    """E2: cells of K windows from the E0 draws; window-level bootstrap of the median and the mean."""
    K, B = cfg["K"], cfg["B_cell"]
    cells = T_sim[: (T_sim.size // K) * K].reshape(-1, K)
    seeds = np.random.SeedSequence(cfg["seed_cell"]).spawn(2 * len(cells))
    se_med, lo_med, hi_med, ndist = (np.zeros(len(cells)) for _ in range(4))
    se_mean, lo_mean, hi_mean = (np.zeros(len(cells)) for _ in range(3))
    for c, v in enumerate(cells):
        se_med[c], lo_med[c], hi_med[c], dist = bw.window_level_bootstrap(
            v, np.median, B, seeds[2 * c], cfg["level"])
        ndist[c] = np.unique(np.round(dist, 14)).size
        se_mean[c], lo_mean[c], hi_mean[c], _ = bw.window_level_bootstrap(
            v, np.mean, B, seeds[2 * c + 1], cfg["level"])
    return dict(cells=cells, se_med=se_med, lo_med=lo_med, hi_med=hi_med, ndist=ndist,
                se_mean=se_mean, lo_mean=lo_mean, hi_mean=hi_mean)


def report(cfg, e0, w, predL, e2):
    L, n = cfg["lag"], cfg["n"]
    ok = {}
    print("== E0: true law of T = rho_hat(1 ms), n = %d, M = %d windows per theta" % (n, cfg["M_true"]))
    for key in ("sim", "cf"):
        d = e0[key]
        print("  theta_%-3s g = %.4f  E[T] = %.4f  SD(T) = %.4f  med(T) = %.4f  bias = %+.4f (SE %.4f)"
              % (key, d["g"], d["mean"], d["sd"], d["med"], d["bias"], d["se_bias"]))
    ref = {"sim": (-0.1082, 0.0021), "cf": (-0.1027, 0.0014)}   # bias_doc/scripts/logs/check01.txt
    for key, (b0, s0) in ref.items():
        z = (e0[key]["bias"] - b0) / np.hypot(e0[key]["se_bias"], s0)
        ok["P6_" + key] = abs(z) < 3
        print("  P6 theta_%s: bias vs bias doc %+.3f: z = %+.2f -> %s" % (key, b0, z, ok["P6_" + key]))

    print("\n== E1: bootstrap worlds built from one window (%d windows, B = %d each)" % (len(w["T_obs"]), cfg["B"]))
    print("  observed T over the windows: mean %.4f, SD %.4f" % (w["T_obs"].mean(), w["T_obs"].std(ddof=1)))
    j0 = 0
    Tm, Tsd = w["Tm"][:, j0], w["Tsd"][:, j0]
    ok["P1"] = bool(np.all(np.abs(Tm) < 0.01) and np.all((Tsd > 0.85 / np.sqrt(n)) & (Tsd < 1.15 / np.sqrt(n))))
    print("  iid: mean T* in [%.5f, %.5f]; SD(T*) in [%.5f, %.5f] vs 1/sqrt(n) = %.5f -> P1 %s"
          % (Tm.min(), Tm.max(), Tsd.min(), Tsd.max(), 1 / np.sqrt(n), ok["P1"]))
    zmax = {}
    for scheme in ("cbb", "sb"):
        idx = [j for j, (s, b) in enumerate(w["labels"]) if s == scheme]
        zL = (w["SLm"][:, idx] - predL[:, idx]) / (w["SLsd"][:, idx] / np.sqrt(cfg["B"]))
        z0 = (w["S0m"][:, idx] - w["S0"][:, None]) / (w["S0sd"][:, idx] / np.sqrt(cfg["B"]))
        zmax[scheme] = (np.nanmax(np.abs(zL)), np.nanmax(np.abs(z0)))
    ok["P2"] = bool(max(zmax["cbb"]) < 4)
    ok["P3"] = bool(max(zmax["sb"]) < 4)
    print("  P2 cbb join identity: max|z| SL~ %.2f, S0~ %.2f -> %s" % (*zmax["cbb"], ok["P2"]))
    print("  P3 sb  join identity: max|z| SL~ %.2f, S0~ %.2f -> %s" % (*zmax["sb"], ok["P3"]))
    sd_true, bias_true = e0["sim"]["sd"], e0["sim"]["bias"]
    print("  true law (E0, theta_sim): SD(T) = %.4f, bias = %+.4f;  parametric at theta_cf: SD %.4f, bias %+.4f"
          % (sd_true, bias_true, e0["cf"]["sd"], e0["cf"]["bias"]))
    print("  per scheme and block length: median [10%%, 90%%] over windows of SD(T*) and of mean(T*) - T(x)")
    for j, (s, b) in enumerate(w["labels"]):
        sd = w["Tsd"][:, j]; bi = w["Tm"][:, j] - w["T_obs"]
        print("   %-3s b = %5d (%6.1f ms): SD(T*) %.4f [%.4f, %.4f]   bias* %+.4f [%+.4f, %+.4f]"
              % (s, b, b * cfg["dt"] * 1e3, np.median(sd), *np.quantile(sd, [0.1, 0.9]),
                 np.median(bi), *np.quantile(bi, [0.1, 0.9])))
    # decomposition of the cbb bias estimate: join part = ratio of the join identity (P2) minus T(x);
    # the rest is the bootstrap world's own demeaning (and the ratio's nonlinearity)
    print("  cbb bias* split, median over windows: total = join part + rest")
    for j, (s, b) in enumerate(w["labels"]):
        if s != "cbb":
            continue
        join = max(0.0, 1 - L / b) * (w["SL"] + w["WL"]) / w["S0"] - w["T_obs"]
        tot = w["Tm"][:, j] - w["T_obs"]
        print("   b = %5d (%6.1f ms): total %+.4f = join %+.4f + rest %+.4f"
              % (b, b * cfg["dt"] * 1e3, np.median(tot), np.median(join), np.median(tot - join)))

    print("\n== E2: window-level bootstrap, cells of K = %d windows (%d cells, B = %d each)"
          % (cfg["K"], len(e2["cells"]), cfg["B_cell"]))
    K = cfg["K"]
    med_cells = np.median(e2["cells"], axis=1)
    mean_cells = e2["cells"].mean(axis=1)
    sd_med_true, sd_mean_true = med_cells.std(ddof=1), mean_cells.std(ddof=1)
    r_mean = e2["se_mean"] / (e0["sim"]["sd"] / np.sqrt(K))
    q = np.quantile(r_mean, [0.1, 0.5, 0.9])
    ok["P4"] = bool(0.78 <= q[1] <= 0.92 and 0.42 <= q[0] <= 0.62 and 1.10 <= q[2] <= 1.40)
    print("  mean of K: true SD %.4f (= SD(T)/sqrt(K) %.4f); SE_boot/true: median %.3f, 10%% %.3f, 90%% %.3f -> P4 %s"
          % (sd_mean_true, e0["sim"]["sd"] / np.sqrt(K), q[1], q[0], q[2], ok["P4"]))
    r_med = e2["se_med"] / sd_med_true
    qm = np.quantile(r_med, [0.1, 0.5, 0.9])
    ok["P5"] = bool(e2["ndist"].max() <= 21)
    cover_med = np.mean((e2["lo_med"] <= e0["sim"]["med"]) & (e0["sim"]["med"] <= e2["hi_med"]))
    cover_mean = np.mean((e2["lo_mean"] <= e0["sim"]["mean"]) & (e0["sim"]["mean"] <= e2["hi_mean"]))
    print("  median of K: true SD %.4f; SE_boot/true: median %.3f, 10%% %.3f, 90%% %.3f"
          % (sd_med_true, qm[1], qm[0], qm[2]))
    print("  distinct bootstrap medians per cell: max %d, median %d -> P5 %s"
          % (e2["ndist"].max(), np.median(e2["ndist"]), ok["P5"]))
    print("  95%% percentile-interval coverage: median of K covers med(T) in %.3f of cells; mean of K covers E[T] in %.3f"
          % (cover_med, cover_mean))
    print("\nALL PREDICTIONS:", ok)
    return ok


def main():
    t0 = time.time()
    cfg = CONFIG
    e0 = true_law(cfg)
    print("[E0 done, %.0f s]" % (time.time() - t0))
    w = window_worlds(cfg)
    print("[E1 done, %.0f s]" % (time.time() - t0))
    predL = join_predictions(cfg, w)
    e2 = cell_worlds(cfg, e0["sim"]["T"])
    print("[E2 done, %.0f s]\n" % (time.time() - t0))
    ok = report(cfg, e0, w, predL, e2)
    OUT_RES.parent.mkdir(exist_ok=True)
    np.savez(OUT_RES, labels=np.array(["%s_%d" % lb for lb in w["labels"]]),
             blocks=np.array(cfg["blocks"]), dt=cfg["dt"], n=cfg["n"], lag=cfg["lag"],
             T_obs=w["T_obs"], Tm=w["Tm"], Tsd=w["Tsd"], S0=w["S0"], SL=w["SL"], WL=w["WL"],
             true_sd_sim=e0["sim"]["sd"], true_bias_sim=e0["sim"]["bias"],
             true_sd_cf=e0["cf"]["sd"], true_bias_cf=e0["cf"]["bias"],
             g_sim=e0["sim"]["g"], mean_sim=e0["sim"]["mean"],
             ok=np.array([str(ok)]))
    print("saved", OUT_RES, "| total %.0f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
