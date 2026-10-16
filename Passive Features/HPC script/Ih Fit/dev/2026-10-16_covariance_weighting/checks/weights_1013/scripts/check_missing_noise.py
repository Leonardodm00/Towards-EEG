"""A noise source the simulator lacks: what it does to the weighting and the fit (check, 2026-10-14 17:30).

Reads ../results/weights.npz (check_weights.py: ten LS medians, K = 6, only tau_s free at theta_sim). Sigma there
plays Sigma_used (what the simulator gives). The recordings are assumed to carry, on top, independent extra scatter
that raises each statistic's variance by a fraction s2:
    Sigma_true = Sigma_used + s2 * diag(Sigma_used),   so E = Sigma_used - Sigma_true = -s2 * diag(Sigma_used).
In standardised units (each statistic divided by its SE) this adds s2 to every direction, so along the eigen-axis
i of the correlation matrix C the used variance is mu_i and the true one mu_i + s2 (exact: s2*I commutes with C).

Reported:
  exact   weight ratio used/right along C's thinnest axis, (mu_min + s2)/mu_min, and along the widest;
  exact   E[J(tau_s0)] with W = Sigma_used^-1 under the true law = tr(Sigma_used^-1 Sigma_true) = d + s2 tr(C^-1);
  run     Gaussian inputs q_hat ~ N(q(tau_s0), Sigma_true), R draws, fitted on the fine grid with
          W = Sigma_used^-1 (what you would do), W = Sigma_true^-1 (oracle), W = diag(1/SE_used^2):
          robust SD of tau_hat; mean minimised cost under W = Sigma_used^-1 against chi2_9 (mean 9, 95 % 16.9);
          the nominal SD (G^T Sigma_used^-1 G)^-1/2 you would report.

Predictions, fixed BEFORE the first run (s2 = 0.01, i.e. +1 % variance on each statistic):
  PM1 mean minimised cost under W = Sigma_used^-1 exceeds 15 (it would be about 9 without the extra noise).
  PM2 the robust SD of tau_hat under W = Sigma_used^-1 is at least 0.95 x the oracle's (no free lunch from a wrong W).
  PM3 the robust SD of tau_hat under W = Sigma_used^-1 is larger at s2 = 0.01 than at s2 = 0, while the nominal SD
      (computed from Sigma_used alone) is the same in both by construction: the reported precision does not move
      while the actual one worsens. [Edited before the first run: comparing nominal and obtained SD at s2 = 0.01
      alone would not isolate the noise, since the linearised SD is already optimistic at s2 = 0
      (check_cov_structure.py, P12/P15).]
Also run, without prediction, s2 = 0.001 and 0.1; s2 = 0 is the baseline.

Run (after check_weights.py):  python check_missing_noise.py     (about 30 s)
"""
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

CONFIG = dict(s2_list=(0.0, 0.001, 0.01, 0.1), tau0=0.03974, n_fine=2000, R=3000, seed=2026101470, chunk=250)
RES = Path(__file__).resolve().parent.parent / "results" / "weights.npz"


def load():
    d = np.load(RES)
    se, C = d["se"], d["corr"]
    return dict(se=se, C=C, S=C * np.outer(se, se), tau_grid=d["tau_grid"], Qg=d["Qg"])


def fit(qh, Qf, tau_fine, W, chunk):
    th, Jm = [], []
    for s in range(0, len(qh), chunk):
        r = Qf[None, :, :] - qh[s:s + chunk, None, :]
        J = np.einsum("rgi,ij,rgj->rg", r, W, r)
        k = J.argmin(axis=1)
        th.append(tau_fine[k]); Jm.append(J[np.arange(len(k)), k])
    return np.concatenate(th), np.concatenate(Jm)


def rsd(v):
    q10, q90 = np.quantile(v, [0.1, 0.9])
    return (q90 - q10) / 2.563


def main():
    cfg, D = CONFIG, load()
    S, C, se = D["S"], D["C"], D["se"]
    mu = np.linalg.eigvalsh(C)
    spl = CubicSpline(np.log(D["tau_grid"]), D["Qg"], axis=0)
    tau_fine = np.geomspace(D["tau_grid"][0], D["tau_grid"][-1], cfg["n_fine"])
    Qf, q0 = spl(np.log(tau_fine)), spl(np.log(cfg["tau0"]))
    G = spl(np.log(cfg["tau0"]), 1) / cfg["tau0"]
    W_used, W_diag = np.linalg.inv(S), np.diag(1 / se**2)
    nominal = 1e3 / np.sqrt(G @ W_used @ G)
    rng = np.random.default_rng(cfg["seed"])
    ok = {}
    print("C: mu_max %.3f, mu_min %.4f, tr(C^-1) %.1f; nominal SD with W = Sigma_used^-1: %.2f ms"
          % (mu.max(), mu.min(), np.trace(np.linalg.inv(C)), nominal))
    for s2 in cfg["s2_list"]:
        S_true = S + s2 * np.diag(np.diag(S))
        ratio_thin, ratio_wide = (mu.min() + s2) / mu.min(), (mu.max() + s2) / mu.max()
        EJ = np.trace(W_used @ S_true)
        qh = q0 + rng.standard_normal((cfg["R"], len(q0))) @ np.linalg.cholesky(S_true).T
        res = {}
        for name, W in (("used", W_used), ("oracle", np.linalg.inv(S_true)), ("diag", W_diag)):
            th, Jm = fit(qh, Qf, tau_fine, W, cfg["chunk"])
            res[name] = (1e3 * rsd(th), Jm.mean(), 1e3 * np.median(th))
        print("\ns2 = %.3f (+%.1f %% variance on each statistic): weight used/right on the thinnest axis %.2f, "
              "on the widest %.4f; exact E[J(tau_s0)] with W = Sigma_used^-1: %.2f (10 without the noise)"
              % (s2, 100 * s2, ratio_thin, ratio_wide, EJ))
        for name, (sd, Jmean, med) in res.items():
            print("  W = %-6s robust SD(tau_hat) %.2f ms, median %.2f ms, mean minimised cost %.2f" % (name, sd, med, Jmean))
        if s2 == 0.0:
            base_sd = res["used"][0]
        if s2 == 0.01:
            ok["PM1"] = bool(res["used"][1] > 15)
            ok["PM2"] = bool(res["used"][0] >= 0.95 * res["oracle"][0])
            ok["PM3"] = bool(res["used"][0] > base_sd)
            print("  nominal SD reported with W = Sigma_used^-1: %.2f ms (any s2); obtained %.2f ms here vs %.2f ms at s2 = 0"
                  % (nominal, res["used"][0], base_sd))
    print("\nALL PREDICTIONS:", ok)


if __name__ == "__main__":
    main()
