"""Three small checks for the answer of 2026-10-14 09:10 (Sigma vs theta; condition number; correlations).

Same statistics and law as check_weights.py (ten LS statistics, medians over K = 6, theta_sim with only
tau_s varied). Predictions, fixed BEFORE running:
  P7  Sigma depends on the TRUE law: the SE of the median of rho_hat(1 ms) on the 265 ms window changes
      by more than 10 % (max/min > 1.10) across true tau_s in {20, 39.74, 80} ms.
  P8  conditioning: a symmetric 1 % multiplicative perturbation of the true Sigma changes Sigma^-1 by
      more than 1 % in relative spectral norm (amplification > 1) and by no more than the first-order
      bound kappa * ||dSigma|| / ||Sigma|| (checked on the median over perturbations).
  P9  2 x 2 identity: for unit-SE statistics with correlation c,
      r^T Sigma^-1 r = (r1 + r2)^2 / (2 (1 + c)) + (r1 - r2)^2 / (2 (1 - c)), to 1e-10.
Run (after check_weights.py):  python check_sigma_theta.py   (about 1 min)

[Note added 2026-10-14, after check_conditioning.py and check_cov_structure.py: P8 perturbs each entry of
Sigma independently by about 1 %, so ||dSigma||_2 / ||Sigma||_2 is of order 1e-2 and kappa * eps >> 1. There
the first-order expression kappa * eps is not a bound (the exact bound kappa*eps / (1 - kappa*eps) needs
kappa * eps < 1), so P8's comparison with it proves nothing. And an independent error on each entry is not
how Monte Carlo error enters a simulated Sigma_hat: see check_cov_structure.py, part A, where the sampling
error of Sigma_hat changes the cost r^T Sigma^-1 r by at most about 14 % (R = 3000), while independent entry
errors of the same per-entry size make Sigma + dSigma indefinite in 300 of 300 draws.]
"""
from pathlib import Path

import numpy as np

import check_weights as cw
import check_cov_source as cs

RES = Path(__file__).resolve().parent.parent / "results"
CFG = dict(cw.CONFIG)
TAUS = [0.020, 0.03974, 0.080]
R, SEED, SEED_PERT, N_PERT = 600, 2026101410, 2026101411, 200


def main():
    ok = {}
    # P7: Sigma at three true laws
    se_acf1, conds = [], []
    for k, tau in enumerate(TAUS):
        med = np.median(cs.cells_stats(CFG, tau, R, SEED + k), axis=1)
        S = np.cov(med, rowvar=False)
        se = np.sqrt(np.diag(S))
        se_acf1.append(se[3]); conds.append(np.linalg.cond(S))
        print("  true tau_s %5.1f ms: SE med sd265 %.4f mV, SE med acf1ms265 %.4f, kappa(Sigma) %.1e"
              % (1e3 * tau, se[0], se[3], conds[-1]))
    ok["P7"] = bool(max(se_acf1) / min(se_acf1) > 1.10)
    print("  P7 max/min SE(acf 1 ms, 265) = %.2f -> %s" % (max(se_acf1) / min(se_acf1), ok["P7"]))

    # P8: conditioning of the true Sigma (from check_weights.py)
    d = np.load(RES / "weights.npz")
    Sig = d["corr"] * np.outer(d["se"], d["se"])
    lam = np.linalg.eigvalsh(Sig); lamc = np.linalg.eigvalsh(d["corr"])
    kappa, kappa_c = lam.max() / lam.min(), lamc.max() / lamc.min()
    print("  kappa(Sigma) = %.2e (largest/smallest eigenvalue); kappa(correlation matrix) = %.2e" % (kappa, kappa_c))
    rng = np.random.default_rng(SEED_PERT)
    inv = np.linalg.inv(Sig)
    amp, bound = [], []
    for _ in range(N_PERT):
        E = rng.normal(0, 0.01, Sig.shape); E = (E + E.T) / 2
        dS = Sig * E                                    # 1 % multiplicative, symmetric
        rel_in = np.linalg.norm(dS, 2) / np.linalg.norm(Sig, 2)
        rel_out = np.linalg.norm(np.linalg.inv(Sig + dS) - inv, 2) / np.linalg.norm(inv, 2)
        amp.append(rel_out / rel_in); bound.append(kappa)
    amp = np.array(amp)
    ok["P8"] = bool(np.median(amp) > 1 and np.median(amp) <= kappa)
    print("  1 %% perturbation: relative change of Sigma^-1 / relative change of Sigma: median %.0f, 10-90%% [%.0f, %.0f]"
          " (bound kappa = %.0f) -> P8 %s" % (np.median(amp), *np.quantile(amp, [0.1, 0.9]), kappa, ok["P8"]))

    # P9: 2 x 2 identity
    c = 0.99
    S2 = np.array([[1, c], [c, 1]])
    r = rng.normal(size=(1000, 2))
    lhs = np.einsum("ni,ij,nj->n", r, np.linalg.inv(S2), r)
    rhs = (r[:, 0] + r[:, 1])**2 / (2 * (1 + c)) + (r[:, 0] - r[:, 1])**2 / (2 * (1 - c))
    ok["P9"] = bool(np.max(np.abs(lhs - rhs)) < 1e-10)
    print("  P9 2x2 identity, c = %.2f: max |difference| %.1e; weight on difference / weight on sum = %.0f -> %s"
          % (c, np.max(np.abs(lhs - rhs)), (1 + c) / (1 - c), ok["P9"]))
    print("\nALL PREDICTIONS:", ok)


if __name__ == "__main__":
    main()
