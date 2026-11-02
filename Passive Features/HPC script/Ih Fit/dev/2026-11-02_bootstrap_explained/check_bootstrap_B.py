"""check_bootstrap_B.py -- what the bootstrap size B and a failed refit do to the per-sweep stage.

Standalone check (not pipeline code), written 2026-11-02 by the noise-fit chat for the
explanation of the per-sweep stage's bootstrap (SPEC section 2.5, Block 9 = "Block 4b";
D-116) given so that the user can answer its four open points. No Block-9 code exists;
this script models only the pooling of SPEC (P3)-(P4) and (D-076.1).

Model (the hypotheses every number below carries), for one cell and one protocol with
k sweeps:
  true value      y_true_s = mu + delta_s,  delta_s ~ N(0, tau^2)   (log sigma_s, log mV)
  per-sweep fit   y_s = y_true_s + eps_s,   eps_s ~ N(0, v)          (sampling error)
  bootstrap       v_hat_s = v * X_s / (B - 1),  X_s ~ chi^2_{B-1}     (the sample variance
                  of B normal refits whose variance is v; the binding seed's error and the
                  dependence of v on the estimate are left out)
  pooling         DerSimonian-Laird on (y, v_hat), tau2 clipped at 0, the random-effects
                  mean recomputed, then (D-076.1); B = inf means v known exactly.
v per sweep from dev/2026-10-30_sweep_stage_cost (linearised, theta_cf): SS sweep of 30
windows 0.0046 (SD 0.068), LS pre-window 0.026 (SD 0.16).

Part A: SD(v_hat) / v against sqrt(2 / (B - 1)) (textbook), by Monte Carlo.
Part B: RMS error of the shrunk log sigma_s (log units: 0.01 = about 1 % in sigma_s)
        against the truth, for B = 20, 50, 100, 200 and v known; the raw per-sweep
        estimate for reference.
Part C: one refit stuck at the lower bound of sigma_s (1e-4 mV, SPEC 2.3 (d)) inside v.

    python check_bootstrap_B.py      # about 20 s

Pure ASCII, LF.
"""
from __future__ import annotations

import numpy as np

SEED = 20261102
N_CELLS = 40000
B_LIST = (20, 50, 100, 200)
TAUS = (0.0, 0.05, 0.10, 0.20)
CASES = (("SS", 2, 0.0046), ("SS", 10, 0.0046), ("LS", 5, 0.026))
LOG_SIGMA_S = np.log(0.0508)        # theta_cf's sigma_s, mV
LOG_LOWER = np.log(1e-4)            # the fit's lower bound on sigma_s, mV


def pool(y, v):
    """(P3)-(P4) and (D-076.1), vectorised over cells: y, v of shape (n, k)."""
    w = 1.0 / v
    sw = w.sum(axis=1)
    ybar = (w * y).sum(axis=1) / sw
    Q = (w * (y - ybar[:, None]) ** 2).sum(axis=1)
    k = y.shape[1]
    tau2 = np.maximum(0.0, (Q - (k - 1)) / (sw - (w ** 2).sum(axis=1) / sw))
    ws = 1.0 / (v + tau2[:, None])
    mu = (ws * y).sum(axis=1) / ws.sum(axis=1)
    lam = tau2[:, None] / (tau2[:, None] + v)
    return mu[:, None] + lam * (y - mu[:, None])


def main():
    rng = np.random.default_rng(SEED)

    print("A  SD(v_hat)/v of the sample variance of B normal values (Monte Carlo, 200000 sets) "
          "against sqrt(2/(B-1))")
    for B in B_LIST:
        s2 = rng.chisquare(B - 1, 200000) / (B - 1)
        print("   B = %3d   Monte Carlo %.4f   formula %.4f" % (B, s2.std(), np.sqrt(2.0 / (B - 1))))

    print("B  RMS error of the shrunk log sigma_s against the truth (x 100, log units; "
          "%d cells per row)" % N_CELLS)
    print("   protocol  k   tau   raw     " + "  ".join("B=%-4d" % B for B in B_LIST) + "  v known")
    for name, k, v in CASES:
        for tau in TAUS:
            y_true = LOG_SIGMA_S + rng.normal(0.0, tau, (N_CELLS, k))
            y = y_true + rng.normal(0.0, np.sqrt(v), (N_CELLS, k))
            row = ["   %-8s %2d  %.2f  %6.2f" % (name, k, tau, 100 * np.sqrt(np.mean((y - y_true) ** 2)))]
            for B in B_LIST:
                v_hat = v * rng.chisquare(B - 1, (N_CELLS, k)) / (B - 1)
                row.append("%6.2f" % (100 * np.sqrt(np.mean((pool(y, v_hat) - y_true) ** 2))))
            v_known = np.full((N_CELLS, k), v)
            row.append("%6.2f" % (100 * np.sqrt(np.mean((pool(y, v_known) - y_true) ** 2))))
            print("  ".join(row))

    print("C  one refit stuck at the lower bound of sigma_s (log 1e-4 mV) among B, kept in v "
          "(mean over 20000 bootstraps)")
    for name, _k, v in (("SS", 2, 0.0046), ("LS", 5, 0.026)):
        for B in (50, 100):
            reps = LOG_SIGMA_S + rng.normal(0.0, np.sqrt(v), (20000, B))
            reps[:, 0] = LOG_LOWER
            v_in = reps.var(axis=1, ddof=1).mean()
            print("   %s B = %3d: v with the stuck refit %.3f against %.4f without (x %.0f)"
                  % (name, B, v_in, v, v_in / v))


if __name__ == "__main__":
    main()
