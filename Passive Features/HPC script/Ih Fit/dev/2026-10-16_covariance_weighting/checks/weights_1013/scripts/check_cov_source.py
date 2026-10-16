"""Where can the covariance Sigma of the statistics come from? Data of one cell vs simulation (check, 2026-10-14).

Same configuration as check_weights.py (ten LS statistics, medians over K = 6 windows, only tau_s
free, theta_sim otherwise; binding function and oracle Sigma read from ../results/weights.npz).
Weightings compared on R simulated cells:
  oracle  Sigma at the true tau_s (from check_weights.py; not available in practice)
  twostep Sigma simulated at the grid point nearest a first-step tau_hat (hand weights), refit
  boot    Sigma estimated from the cell's OWN six windows by a window-level bootstrap of the medians
  diag    1/SE^2 at the truth (reference), hand  hand-set omega_m (reference)

Predictions, fixed BEFORE the first run:
  P4  robust SD(tau_hat) under 'twostep' <= 1.2 x robust SD under 'oracle'.
  P5  robust SD(tau_hat) under 'boot' > 1.2 x robust SD under 'oracle' (one cell's six windows are
      too few to estimate a 10 x 10 covariance of medians).
  P6  the plain sample covariance of one cell's six single-window statistic vectors has rank <= 5
      (K - 1), so it cannot be inverted (linear algebra; checked on every cell).

Run (after check_weights.py):  python check_cov_source.py     (about 2-3 min)
"""
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

import check_weights as cw

CFG = dict(cw.CONFIG)
CFG.update(R=600, seed_cells=2026101401, M_grid_cells=600, seed_grid=2026101402,
           B_boot=1000, seed_boot=2026101403)
RES = Path(__file__).resolve().parent.parent / "results"


def cells_stats(cfg, tau, R, seed, chunk_cells=100):
    """Per-window statistics of R cells of K windows at one tau, shape (R, K, 10)."""
    rng = np.random.default_rng(seed)
    out = []
    for start in range(0, R, chunk_cells):
        m = min(chunk_cells, R - start)
        (_, x), = cw.windows_crn(cfg, [tau], m * cfg["K"], rng)
        out.append(cw.stats10(x, cfg).reshape(m, cfg["K"], 10))
    return np.concatenate(out)


def boot_cov_of_medians(S, B, rng):
    """Window-level bootstrap covariance of the 10 medians from one cell's K x 10 statistics."""
    K = S.shape[0]
    idx = rng.integers(0, K, (B, K))
    med = np.median(S[idx], axis=1)                       # (B, 10)
    return np.cov(med, rowvar=False)


def fit_one(qhat, Q_fine, tau_fine, W):
    r = Q_fine - qhat[None, :]
    J = np.einsum("gi,ij,gj->g", r, W, r)
    return tau_fine[J.argmin()]


def main():
    t0 = time.time()
    d = np.load(RES / "weights.npz")
    tau_grid, Qg = d["tau_grid"], d["Qg"]
    Sigma_or = d["corr"] * np.outer(d["se"], d["se"])
    tau_fine = np.geomspace(tau_grid[0], tau_grid[-1], CFG["n_fine"])
    Q_fine = CubicSpline(np.log(tau_grid), Qg, axis=0)(np.log(tau_fine))
    tau0 = CFG["theta_sim"]["tau_s"]

    S = cells_stats(CFG, tau0, CFG["R"], CFG["seed_cells"])          # (R, K, 10)
    qhat = np.median(S, axis=1)
    print("[cells: %.0f s]" % (time.time() - t0))
    # Sigma on the grid by simulation (for the two-step weighting)
    Sig_grid = []
    for g, tau in enumerate(tau_grid):
        med = np.median(cells_stats(CFG, tau, CFG["M_grid_cells"], CFG["seed_grid"] + g), axis=1)
        Sig_grid.append(np.cov(med, rowvar=False))
    print("[Sigma on the grid: %.0f s]" % (time.time() - t0))

    W_or, W_diag = np.linalg.inv(Sigma_or), np.diag(1 / d["se"]**2)
    W_hand = np.diag(1 / d["omega"]**2)
    rng = np.random.default_rng(CFG["seed_boot"])
    out = {k: np.zeros(CFG["R"]) for k in ("hand", "diag", "oracle", "twostep", "boot")}
    rank_plain, cond_boot = np.zeros(CFG["R"], int), np.zeros(CFG["R"])
    for c in range(CFG["R"]):
        q = qhat[c]
        out["hand"][c] = t1 = fit_one(q, Q_fine, tau_fine, W_hand)
        out["diag"][c] = fit_one(q, Q_fine, tau_fine, W_diag)
        out["oracle"][c] = fit_one(q, Q_fine, tau_fine, W_or)
        g = np.argmin(np.abs(np.log(tau_grid) - np.log(t1)))
        out["twostep"][c] = fit_one(q, Q_fine, tau_fine, np.linalg.inv(Sig_grid[g]))
        Sb = boot_cov_of_medians(S[c], CFG["B_boot"], rng)
        cond_boot[c] = np.linalg.cond(Sb)
        out["boot"][c] = fit_one(q, Q_fine, tau_fine, np.linalg.pinv(Sb))
        rank_plain[c] = np.linalg.matrix_rank(np.cov(S[c], rowvar=False))
    print("[fits: %.0f s]\n" % (time.time() - t0))

    rsd = {k: cw.robust_sd(v) for k, v in out.items()}
    for k, v in out.items():
        print("  %-7s median %.2f ms, robust SD %.2f ms, 10-90%% [%.1f, %.1f] ms"
              % (k, 1e3 * np.median(v), 1e3 * rsd[k], *(1e3 * np.quantile(v, [0.1, 0.9]))))
    ok = dict(P4=bool(rsd["twostep"] <= 1.2 * rsd["oracle"]),
              P5=bool(rsd["boot"] > 1.2 * rsd["oracle"]),
              P6=bool(rank_plain.max() <= CFG["K"] - 1))
    print("  plain per-cell sample covariance (K = %d windows): rank max %d, min %d"
          % (CFG["K"], rank_plain.max(), rank_plain.min()))
    print("  per-cell bootstrap covariance of the medians: condition number median %.1e, 10-90%% [%.1e, %.1e]"
          " (oracle Sigma: %.1e)" % (np.median(cond_boot), *np.quantile(cond_boot, [0.1, 0.9]),
                                     np.linalg.cond(Sigma_or)))
    print("\nALL PREDICTIONS:", ok, "| total %.0f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
