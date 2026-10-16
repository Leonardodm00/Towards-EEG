"""Does the choice of omega_m (the weights of the indirect-inference cost) matter? (check, 2026-10-13)

Configuration: the ten Long Square statistics of the topic doc (TD section 3.5.3) on 265 ms
pre-windows (n = 13 250 samples at 50 kHz): sigma_hat and rho_hat at 0.02, 0.1, 1 ms on the whole
window; sigma_hat on the first 50 ms; sigma_hat and rho_hat at 0.02, 0.1, 1, 10 ms on the last
132.5 ms. Data per cell: the median over K = 6 windows of each statistic (TD 3.2.2). Law: the
two-component law at theta_sim with ONLY tau_s free (sigma_f, rho_f, sigma_s held at theta_sim),
so the fit is one-dimensional and cheap. Binding function q_m(tau_s) = med(T_m | theta(tau_s), n_m),
the single-window median, by simulation with common random numbers on a log grid of tau_s.

Fit per cell: tau_hat = argmin_tau (q(tau) - q_hat)^T W (q(tau) - q_hat) with three weightings:
  hand  W = diag(1/omega_m^2), omega_m = 0.005 mV for SDs, 0.05 for autocorrelations (TD 3.5.2);
  diag  W = diag(1/SE_m^2), SE_m the SD of the median-of-K statistic at the truth (simulated);
  full  W = Sigma^{-1}, Sigma the covariance of the ten medians-of-K at the truth (simulated).

Predictions, fixed BEFORE the first run:
  P1 (consistency): for each weighting |median(tau_hat) - tau_s0| < 0.25 * robust SD(tau_hat),
     robust SD = (q90 - q10) / 2.563.
  P2 (efficiency, Hansen / GMM theory as stated in Kaczmarska et al. 2015): robust SD under 'full'
     <= 1.05 x min(robust SD under 'diag', robust SD under 'hand').
  P3 (chi-square reading): with W = Sigma^{-1}, the minimised cost has mean within [0.8, 1.2] x
     (m - p) = [7.2, 10.8] (m = 10 statistics, p = 1 parameter).
Measured without prediction: SE_m against omega_m; correlations between the medians.

Run:  python check_weights.py      (about 3-5 min on 2 cores; writes ../results, prints a report)
"""
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import lfilter

import boot_model as bm

CONFIG = dict(
    dt=20e-6, n=13_250,
    theta_sim=dict(sigma_f=0.0383, rho_f=0.179, sigma_s=0.0547, tau_s=0.03974),   # TD 3.5.3
    first_n=2500, last_n=6625,                       # 50 ms and 132.5 ms segments of the window
    lags_full=(1, 5, 50), lags_last=(1, 5, 50, 500),  # 0.02, 0.1, 1 (and 10) ms
    omega_sd=0.005, omega_acf=0.05,                  # TD 3.5.2 (mV; dimensionless)
    tau_grid=np.geomspace(0.010, 0.200, 36),         # s, binding-function grid
    M_bind=3000, seed_bind=2026101310,
    K=6,
    R_data=1500, seed_data=2026101311,               # replicate cells fitted
    R_cov=3000, seed_cov=2026101312,                 # cells for Sigma at the truth
    n_fine=2000,
)
NAMES = ["sd 265", "acf .02 265", "acf .1 265", "acf 1 265", "sd first50",
         "sd last132", "acf .02 last", "acf .1 last", "acf 1 last", "acf 10 last"]
IS_SD = np.array([1, 0, 0, 0, 1, 1, 0, 0, 0, 0], bool)
OUT = Path(__file__).resolve().parent.parent / "results" / "weights.npz"


# ---------------- model (common random numbers over tau_s) ----------------
def windows_crn(cfg, taus, n_win, rng):
    """Yield (tau, windows) for each tau in taus, with the SAME random numbers for every tau."""
    th, n, dt = cfg["theta_sim"], cfg["n"], cfg["dt"]
    ef = rng.normal(0.0, th["sigma_f"] * np.sqrt(1 - th["rho_f"]**2), (n_win, n))
    ef[:, 0] = rng.normal(0.0, th["sigma_f"], n_win)
    fast = lfilter([1.0], [1.0, -th["rho_f"]], ef, axis=1)
    z = rng.normal(0.0, 1.0, (n_win, n))           # unit innovations for the slow OU
    for tau in taus:
        a = np.exp(-dt / tau)
        e = th["sigma_s"] * np.sqrt(1 - a**2) * z
        e[:, 0] = th["sigma_s"] * z[:, 0]           # stationary start
        yield tau, fast + lfilter([1.0], [1.0, -a], e, axis=1)


# ---------------- statistics (pipeline estimators: SD ddof=1, textbook ACF) ----------------
def stats10(x, cfg):
    """The ten LS statistics per window (rows), shape (n_win, 10)."""
    first, last = x[:, :cfg["first_n"]], x[:, -cfg["last_n"]:]
    cols = [x.std(axis=1, ddof=1)]
    cols += [bm.acf_tb(x, L) for L in cfg["lags_full"]]
    cols += [first.std(axis=1, ddof=1), last.std(axis=1, ddof=1)]
    cols += [bm.acf_tb(last, L) for L in cfg["lags_last"]]
    return np.column_stack(cols)


def cell_medians(cfg, tau, R, seed, chunk_cells=100):
    """Medians over K windows for R independent cells at one tau, shape (R, 10); generated in chunks."""
    rng = np.random.default_rng(seed)
    out = []
    for start in range(0, R, chunk_cells):
        m = min(chunk_cells, R - start)
        (_, x), = windows_crn(cfg, [tau], m * cfg["K"], rng)
        out.append(np.median(stats10(x, cfg).reshape(m, cfg["K"], 10), axis=1))
    return np.concatenate(out)


# ---------------- fit ----------------
def binding(cfg, chunk=500):
    """q_m(tau) on the grid: single-window medians over M_bind windows (CRN), shape (n_grid, 10)."""
    rng = np.random.default_rng(cfg["seed_bind"])
    S = [[] for _ in cfg["tau_grid"]]
    for start in range(0, cfg["M_bind"], chunk):
        m = min(chunk, cfg["M_bind"] - start)
        for g, (_, x) in enumerate(windows_crn(cfg, cfg["tau_grid"], m, rng)):
            S[g].append(stats10(x, cfg))
    return np.array([np.median(np.concatenate(s), axis=0) for s in S])


def fit_all(qhat, tau_fine, Q_fine, W):
    """tau_hat per cell by minimising the quadratic form on the fine grid; also the minimum."""
    r = Q_fine[None, :, :] - qhat[:, None, :]                  # (R, n_fine, 10)
    J = np.einsum("rgi,ij,rgj->rg", r, W, r)
    k = J.argmin(axis=1)
    return tau_fine[k], J[np.arange(len(k)), k], k


def robust_sd(v):
    q10, q90 = np.quantile(v, [0.1, 0.9])
    return (q90 - q10) / 2.563


def main():
    t0 = time.time()
    cfg = CONFIG
    tau0 = cfg["theta_sim"]["tau_s"]
    Qg = binding(cfg)
    print("[binding function: %.0f s]" % (time.time() - t0))
    lg = np.log(cfg["tau_grid"])
    tau_fine = np.geomspace(cfg["tau_grid"][0], cfg["tau_grid"][-1], cfg["n_fine"])
    Q_fine = CubicSpline(lg, Qg, axis=0)(np.log(tau_fine))
    cov_cells = cell_medians(cfg, tau0, cfg["R_cov"], cfg["seed_cov"])
    Sigma = np.cov(cov_cells, rowvar=False)
    se = np.sqrt(np.diag(Sigma))
    corr = Sigma / np.outer(se, se)
    qhat = cell_medians(cfg, tau0, cfg["R_data"], cfg["seed_data"])
    print("[cells simulated: %.0f s]" % (time.time() - t0))
    omega = np.where(IS_SD, cfg["omega_sd"], cfg["omega_acf"])
    Ws = {"hand": np.diag(1 / omega**2), "diag": np.diag(1 / se**2), "full": np.linalg.inv(Sigma)}

    print("\n== SE of each median-of-%d statistic at the truth vs the hand-set omega_m" % cfg["K"])
    for nm, s, o in zip(NAMES, se, omega):
        print("  %-13s SE = %.4f   omega = %.3f   omega/SE = %.2f" % (nm, s, o, o / s))
    print("  largest |corr| between medians: " + ", ".join(
        "%s~%s %.2f" % (NAMES[i], NAMES[j], corr[i, j])
        for i, j in sorted(((i, j) for i in range(10) for j in range(i + 1, 10)),
                           key=lambda ij: -abs(corr[ij]))[:5]))
    print("  condition number of Sigma: %.1e" % np.linalg.cond(Sigma))

    ok, res = {}, {}
    print("\n== fits of %d replicate cells, tau_s0 = %.2f ms (grid %.0f-%.0f ms)"
          % (cfg["R_data"], 1e3 * tau0, 1e3 * cfg["tau_grid"][0], 1e3 * cfg["tau_grid"][-1]))
    for name, W in Ws.items():
        th, Jmin, k = fit_all(qhat, tau_fine, Q_fine, W)
        rsd = robust_sd(th)
        res[name] = dict(tau=th, J=Jmin, rsd=rsd)
        at_edge = np.mean((k == 0) | (k == len(tau_fine) - 1))
        ok["P1_" + name] = bool(abs(np.median(th) - tau0) < 0.25 * rsd)
        print("  %-4s median %.2f ms, mean %.2f ms, SD %.2f ms, robust SD %.2f ms, 10-90%% [%.1f, %.1f] ms, at grid edge %.3f"
              " -> P1 %s" % (name, 1e3 * np.median(th), 1e3 * th.mean(), 1e3 * th.std(ddof=1), 1e3 * rsd,
                             *(1e3 * np.quantile(th, [0.1, 0.9])), at_edge, ok["P1_" + name]))
    ok["P2"] = bool(res["full"]["rsd"] <= 1.05 * min(res["diag"]["rsd"], res["hand"]["rsd"]))
    print("  P2 robust SD full %.2f vs diag %.2f, hand %.2f ms -> %s"
          % (1e3 * res["full"]["rsd"], 1e3 * res["diag"]["rsd"], 1e3 * res["hand"]["rsd"], ok["P2"]))
    Jf = res["full"]["J"]
    ok["P3"] = bool(7.2 <= Jf.mean() <= 10.8)
    print("  P3 minimised cost with W = Sigma^-1: mean %.2f, median %.2f (chi2_9: 9.00, 8.34) -> %s"
          % (Jf.mean(), np.median(Jf), ok["P3"]))
    print("\nALL PREDICTIONS:", ok)
    OUT.parent.mkdir(exist_ok=True)
    np.savez(OUT, names=np.array(NAMES), se=se, omega=omega, corr=corr, tau_grid=cfg["tau_grid"], Qg=Qg,
             **{f"tau_{k}": v["tau"] for k, v in res.items()}, **{f"J_{k}": v["J"] for k, v in res.items()})
    print("saved", OUT, "| total %.0f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
