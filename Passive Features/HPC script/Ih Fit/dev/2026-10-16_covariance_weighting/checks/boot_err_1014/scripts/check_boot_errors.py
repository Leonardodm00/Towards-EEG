"""Sampling error vs internal error of a bootstrap estimate (check, 2026-10-14).

Setting: the window-level bootstrap of E2 in boot_1013/check_boot_types.py. Statistic per window:
T = rho_hat(1 ms), textbook autocorrelation of a 265 ms LS pre-window (n = 13 250 samples at 50 kHz) drawn
from the Stage-7 law at theta_sim; per cell: the median of T over K = 6 windows. The same 20 000 windows as
E0/E2 (same seed), grouped into 3333 cells.

Bootstrap output studied: SE_B(v) = SD (ddof 1) of the median of K windows resampled with replacement from
the cell's own six values v, over B replicates. Because K = 6, the ideal bootstrap (B -> infinity) is
computed exactly: the 462 count vectors of a resample of six, with multinomial weights.
  target       SD_true = SD over cells of the cell medians (the sampling SD of the median of six)
  sampling     SE_inf(v) - SD_true   (depends on the cell's six values only: the plug-in, B = infinity)
  internal     SE_B(v) - SE_inf(v)   (Monte Carlo, from stopping at B replicates)
  E&N's internal-error estimate ("jacksd", Efron & Narasimhan sec. 3): split the B replicates at random
  into J groups, delete each in turn, recompute SE, jackknife standard error (their 3.9) over the J values.

Predictions, fixed BEFORE the first run:
  P0 (implementation) for three cells, SE_inf by enumeration matches a 10^6-replicate Monte Carlo within 0.5 %.
  P1 the root-mean-square over cells of the internal part scales as 1/sqrt(B): RMS(B = 100) / RMS(B = 1600)
     lies in [3.2, 4.8] (4 +- 20 %).
  P2 at B = 2000 (as in E2) the RMS internal part is below 1/10 of the SD over cells of the sampling part.
  P3 law of total variance: Var over cells of SE_B / SD_true equals Var of SE_inf / SD_true plus the mean
     squared internal part, within +-10 %, at every B.
  P4 E&N's jackknife: RMS over cells of jacksd (B = 2000, J = 10) is within [0.75, 1.33] x the RMS internal part.
Measured without prediction: quantiles of SE_inf / SD_true over cells (E2 reported, at B = 2000, median 0.989,
10 % 0.584, 90 % 1.523).

Run:  python check_boot_errors.py      (about 30 s; needs boot_model.py next to it)
"""
import itertools
import math

import numpy as np

import boot_model as bm

CONFIG = dict(
    dt=20e-6, n=13_250, lag=50,                                                    # s; samples; samples (1 ms)
    theta_sim=dict(sigma_f=0.0383, rho_f=0.179, sigma_s=0.0547, tau_s=0.03974),   # TD 3.5.3
    M=20_000, seed_true=2026101301,                                                # same windows as E0 / E2
    K=6,
    B_list=(25, 100, 400, 1600, 2000), J=10, seed_boot=2026101430,
    B_check=1_000_000, seed_check=2026101431,
)


# ---------------- data ----------------
def make_cells(cfg):
    T = bm.simulate_T(cfg["theta_sim"], cfg["dt"], cfg["n"], cfg["lag"], cfg["M"], cfg["seed_true"])
    K = cfg["K"]
    return T[: (T.size // K) * K].reshape(-1, K)


# ---------------- ideal bootstrap (B = infinity), exact for the median of K ----------------
def compositions(K):
    """All count vectors (c_1..c_K), sum K, with multinomial probabilities; and for each, the positions
    (in the sorted sample) of the two middle order statistics of the resample."""
    counts, w = [], []
    for c in itertools.product(range(K + 1), repeat=K):
        if sum(c) == K:
            counts.append(c)
            w.append(math.factorial(K) / np.prod([math.factorial(ci) for ci in c]) / K**K)
    counts = np.array(counts); w = np.array(w)
    cum = np.cumsum(counts, axis=1)
    lo = np.argmax(cum >= K // 2, axis=1)          # 3rd smallest of 6 (1-based)
    hi = np.argmax(cum >= K // 2 + 1, axis=1)      # 4th smallest of 6
    return w, lo, hi


def se_ideal(cells, comp):
    w, lo, hi = comp
    vs = np.sort(cells, axis=1)
    med = 0.5 * (vs[:, lo] + vs[:, hi])            # (cells, 462)
    m = med @ w
    return np.sqrt(((med - m[:, None])**2) @ w)


# ---------------- Monte Carlo bootstrap and E&N's jackknife of its replicates ----------------
def se_mc(cells, B, rng, J=None, chunk=200):
    se, jsd = np.zeros(len(cells)), np.zeros(len(cells))
    K = cells.shape[1]
    for s in range(0, len(cells), chunk):
        v = cells[s:s + chunk]
        idx = rng.integers(0, K, size=(len(v), B, K))
        med = np.median(np.take_along_axis(v[:, None, :], idx, axis=2), axis=2)   # (chunk, B)
        se[s:s + len(v)] = med.std(axis=1, ddof=1)
        if J:
            groups = rng.permutation(B) % J                                        # random partition, B/J each
            sj = np.stack([med[:, groups != j].std(axis=1, ddof=1) for j in range(J)], axis=1)
            jsd[s:s + len(v)] = np.sqrt((J - 1) / J * ((sj - sj.mean(axis=1, keepdims=True))**2).sum(axis=1))
    return se, jsd


def main():
    cfg = CONFIG
    cells = make_cells(cfg)
    med_cells = np.median(cells, axis=1)
    sd_true = med_cells.std(ddof=1)
    comp = compositions(cfg["K"])
    print("cells %d of K = %d windows; SD_true (SD of the cell median over cells) = %.4f; %d count vectors, "
          "weights sum %.12f" % (len(cells), cfg["K"], sd_true, len(comp[0]), comp[0].sum()))
    ok = {}

    se_inf = se_ideal(cells, comp)
    rng_c = np.random.default_rng(cfg["seed_check"])
    rel = []
    for c in range(3):
        idx = rng_c.integers(0, cfg["K"], size=(cfg["B_check"], cfg["K"]))
        rel.append(np.median(cells[c][idx], axis=1).std(ddof=1) / se_inf[c] - 1)
    ok["P0"] = bool(np.max(np.abs(rel)) < 0.005)
    print("P0 enumeration vs 1e6-replicate Monte Carlo, three cells: relative differences %s -> %s"
          % (np.array2string(np.array(rel), precision=4), ok["P0"]))

    samp = se_inf / sd_true - 1
    q = np.quantile(se_inf / sd_true, [0.1, 0.5, 0.9])
    print("sampling part: SE_inf / SD_true over cells: median %.3f, 10%% %.3f, 90%% %.3f; SD over cells %.3f"
          % (q[1], q[0], q[2], samp.std(ddof=1)))

    rng = np.random.default_rng(cfg["seed_boot"])
    rms, p3 = {}, {}
    for B in cfg["B_list"]:
        J = cfg["J"] if B == 2000 else None
        se_B, jsd = se_mc(cells, B, rng, J)
        inter = (se_B - se_inf) / sd_true
        rms[B] = np.sqrt(np.mean(inter**2))
        lhs = np.var(se_B / sd_true, ddof=1)
        rhs = np.var(se_inf / sd_true, ddof=1) + np.mean(inter**2)
        p3[B] = abs(lhs / rhs - 1) <= 0.10
        print("B = %5d: internal part RMS %.4f (mean %+.4f); Var(SE_B/SD) %.4f vs Var(SE_inf/SD) + mean int^2 %.4f"
              % (B, rms[B], inter.mean(), lhs, rhs))
        if J:
            qB = np.quantile(se_B / sd_true, [0.1, 0.5, 0.9])
            print("          SE_B / SD_true at B = %d: median %.3f, 10%% %.3f, 90%% %.3f (E2: 0.989, 0.584, 1.523)"
                  % (B, qB[1], qB[0], qB[2]))
            rj = np.sqrt(np.mean((jsd / sd_true)**2))
            ok["P4"] = bool(0.75 <= rj / rms[B] <= 1.33)
            print("          E&N jackknife (J = %d): RMS jacksd / SD_true %.4f vs RMS internal part %.4f (ratio %.2f) -> P4 %s"
                  % (J, rj, rms[B], rj / rms[B], ok["P4"]))
    ok["P1"] = bool(3.2 <= rms[100] / rms[1600] <= 4.8)
    ok["P2"] = bool(rms[2000] < 0.1 * samp.std(ddof=1))
    ok["P3"] = bool(all(p3.values()))
    print("P1 RMS internal B = 100 / B = 1600 = %.2f (1/sqrt(B) predicts 4) -> %s" % (rms[100] / rms[1600], ok["P1"]))
    print("P2 RMS internal at B = 2000, %.4f, vs SD of the sampling part %.3f -> %s" % (rms[2000], samp.std(ddof=1), ok["P2"]))
    print("P3 total variance = sampling + internal within 10 %% at every B -> %s" % ok["P3"])
    print("\nALL PREDICTIONS:", ok)


if __name__ == "__main__":
    main()
