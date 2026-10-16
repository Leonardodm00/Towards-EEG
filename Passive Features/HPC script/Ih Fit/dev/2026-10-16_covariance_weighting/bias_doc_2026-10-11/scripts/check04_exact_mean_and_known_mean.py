"""Check 4: (a) an exact expression for E[T | theta, n] exists for Gaussian windows; (b) how much of the bias each
window's own mean causes, against the ratio itself.

(a) For x ~ N(0, Gamma) and T = S_L / S_0 with S_0 = x'Cx, S_L = x'(C D C)x (C centring, D the symmetrised lag-L
    shift), whitening x = R z (Gamma = R R') and diagonalising P = R'CR = U diag(lam) U' give, with
    qd_i = (U' R'CDCR U)_ii,
        E[T | theta, n] = int_0^inf  prod_i (1 + 2 s lam_i)^(-1/2)  sum_i qd_i / (1 + 2 s lam_i)  ds,
    from 1/S_0 = int_0^inf exp(-s S_0) ds and a Gaussian integral (document eq. (11), proof in \u00a73.6).
    Computed for the toy of \u00a73.8 (pure OU, unit SD, tau = 40 ms, n = 663 at 0.2 ms, L = 50) and for theta_sim on a
    50 ms window (n = 2500 at 20 us, L = 50); compared with Monte Carlo means of independent windows.
(b) Under theta_sim on the 265 ms window (n = 13 250, L = 50), on the windows of check01 (seed 20261011, M = 3200):
    the statistic with each window's own mean (the recipe) against the same ratio with the true mean, 0, used
    instead, on the same windows; with the matching ratios of expectations, so that the bias splits as in document
    eq. (8); the share of the bias that a known mean removes, with its delta-method Monte Carlo SE, and the same
    share on 3200 fresh windows (seed 20261016).
(c) The toy's exact bias (3), E[T | tau, n] from (11) minus exp(-10 ms / tau), at tau = 10, 20, 40, 80, 150 ms.
Predictions fixed before running: (a) exact and Monte Carlo agree within 3 Monte Carlo SE; (b) a known mean removes
part, not all, of the bias (the ratio's own S_0-weighting remains), and the two runs agree within 3 SE.
Run:  python3 check04_exact_mean_and_known_mean.py      (about 1-2 min; prints and writes logs/check04.txt)
"""
import numpy as np
from scipy.integrate import quad
from scipy.linalg import cholesky, eigh
from scipy.signal import lfilter
from bias_common import DT, N_LS, LAG_1MS, THETA_SIM, autocov, true_acf, simulate_windows, closed_form_acf

CONFIG = dict(toy=dict(dt=2e-4, n=663, lag=50, tau=0.040, M=200_000, seed=11, chunk=20_000),
              short=dict(dt=DT, n=2500, lag=LAG_1MS, M=40_000, seed=12, chunk=2_000),
              known=dict(dt=DT, n=N_LS, lag=LAG_1MS, M=3200, seed=20261011, seed_fresh=20261016, chunk=200),
              toy_tau_grid=(0.010, 0.020, 0.040, 0.080, 0.150),
              out="logs/check04.txt")


# --- exact expectation of the ratio (Gaussian windows) -------------------------------------------
def exact_mean_ratio(gam, n, L):
    """E[S_L/S_0] for a stationary Gaussian window with autocovariance gam[k], k = 0..n-1 (demeaned recipe)."""
    k = np.arange(n)
    Gam = gam[np.abs(k[:, None] - k[None, :])]
    R = cholesky(Gam, lower=True)
    CR = R - R.mean(axis=0, keepdims=True)                 # C R: centre each column
    P = CR.T @ CR                                            # R' C R  (C symmetric, idempotent)
    DCR = np.zeros_like(CR)                                  # D (C R), D = (E_L + E_L')/2
    DCR[:-L] += 0.5 * CR[L:]
    DCR[L:] += 0.5 * CR[:-L]
    Q = CR.T @ DCR                                           # R' C D C R
    lam, U = eigh(P)
    lam = np.clip(lam, 0.0, None)                            # one eigenvalue is 0 (the mean direction)
    qd = np.einsum("ij,ij->j", U, Q @ U)                     # diag(U' Q U)

    def integrand(v):                                        # s = exp(v), ds = s dv
        s = np.exp(v)
        w = 1.0 + 2.0 * s * lam
        return np.exp(-0.5 * np.sum(np.log(w))) * np.sum(qd / w) * s

    scale = 1.0 / np.max(lam)
    val, err = quad(integrand, np.log(scale) - 40, np.log(scale) + 40, limit=400, epsabs=1e-12, epsrel=1e-10)
    return val, err


# --- Monte Carlo references -----------------------------------------------------------------------
def acf_stat(x, L, mean=None):
    u = x - (x.mean(axis=1, keepdims=True) if mean is None else mean)
    return np.sum(u[:, :-L] * u[:, L:], axis=1) / np.sum(u * u, axis=1)


def mc_toy(c):
    rng = np.random.default_rng(c["seed"])
    a = np.exp(-c["dt"] / c["tau"])
    out = []
    for _ in range(c["M"] // c["chunk"]):
        z = rng.standard_normal((c["chunk"], c["n"]))
        e = z * np.sqrt(1 - a**2)
        e[:, 0] = z[:, 0]
        out.append(acf_stat(lfilter([1.0], [1.0, -a], e, axis=1), c["lag"]))
    return np.concatenate(out)


def mc_theta(th, c, mean=None):
    rng = np.random.default_rng(c["seed"])
    out = []
    for _ in range(c["M"] // c["chunk"]):
        out.append(acf_stat(simulate_windows(th, c["dt"], c["n"], c["chunk"], rng), c["lag"], mean))
    return np.concatenate(out)


def mc_theta_pair(th, c, seed):
    """The recipe (each window's own mean) and the same ratio with the true mean 0, on the same windows."""
    rng = np.random.default_rng(seed)
    own, true = [], []
    for _ in range(c["M"] // c["chunk"]):
        x = simulate_windows(th, c["dt"], c["n"], c["chunk"], rng)
        own.append(acf_stat(x, c["lag"]))
        true.append(acf_stat(x, c["lag"], 0.0))
    return np.concatenate(own), np.concatenate(true)


def share_removed(T_own, T_true, rho):
    """r = mean(T_own - T_true) / (mean(T_own) - rho), with its delta-method Monte Carlo SE (paired windows):
    SE(r) = sd(d - r e) / (sqrt(M) |mean(e)|), d = T_own - T_true, e = T_own - rho."""
    d, e = T_own - T_true, T_own - rho
    r = d.mean() / e.mean()
    return r, (d - r * e).std(ddof=1) / (np.sqrt(d.size) * abs(e.mean()))


def main(cfg=CONFIG):
    lines = []
    # (a) toy
    c = cfg["toy"]
    gam = np.exp(-np.arange(c["n"]) * c["dt"] / c["tau"])
    ex, err = exact_mean_ratio(gam, c["n"], c["lag"])
    T = mc_toy(c)
    se = T.std(ddof=1) / np.sqrt(T.size)
    lines.append(f"(a) toy (OU, tau 40 ms, n {c['n']}, L {c['lag']}): exact E[T] = {ex:.5f} (quad err {err:.1e});"
                 f" Monte Carlo mean over {T.size} windows = {T.mean():.5f} (SE {se:.5f}); difference {(T.mean()-ex)/se:+.2f} SE")
    # (a) theta_sim, 50 ms
    c = cfg["short"]
    gam = autocov(THETA_SIM, c["dt"], c["n"])
    ex, err = exact_mean_ratio(gam, c["n"], c["lag"])
    T = mc_theta(THETA_SIM, c)
    se = T.std(ddof=1) / np.sqrt(T.size)
    qcf = closed_form_acf(THETA_SIM, c["dt"], c["n"], c["lag"])
    lines.append(f"(a) theta_sim, 50 ms (n {c['n']}, L {c['lag']}): exact E[T] = {ex:.5f} (quad err {err:.1e});"
                 f" Monte Carlo mean over {T.size} windows = {T.mean():.5f} (SE {se:.5f}); difference {(T.mean()-ex)/se:+.2f} SE;"
                 f" ratio of expectations q_cf = {qcf:.5f}")
    # (b) own mean against true mean, 265 ms
    c = cfg["known"]
    rho = true_acf(THETA_SIM, c["dt"], c["lag"])
    T_own, T_true = mc_theta_pair(THETA_SIM, c, c["seed"])
    q_own = closed_form_acf(THETA_SIM, c["dt"], c["n"], c["lag"])
    q_true = (c["n"] - c["lag"]) / c["n"] * rho                 # E[sum x_j x_{j+L}] / E[sum x_j^2] with known mean 0
    for name, T, q in (("own window mean (the recipe)", T_own, q_own), ("true mean 0", T_true, q_true)):
        se = T.std(ddof=1) / np.sqrt(T.size)
        lines.append(f"(b) {name}: mean T = {T.mean():.4f} (SE {se:.4f}); bias = {T.mean()-rho:+.4f};"
                     f" ratio of expectations {q:.4f} -> its gap to rho {q-rho:+.4f}; mean T minus it {T.mean()-q:+.4f}")
    d = T_own - T_true
    r, r_se = share_removed(T_own, T_true, rho)
    lines.append(f"(b) per-window difference own - true mean: mean {d.mean():+.4f} (SE {d.std(ddof=1)/np.sqrt(d.size):.4f});"
                 f" share of the recipe's bias removed by a known mean = {r:.3f} (delta-method MC SE {r_se:.3f})")
    T_own2, T_true2 = mc_theta_pair(THETA_SIM, c, c["seed_fresh"])
    r2, r2_se = share_removed(T_own2, T_true2, rho)
    lines.append(f"(b) the same with fresh windows (seed {c['seed_fresh']}, M {T_own2.size}): bias {T_own2.mean()-rho:+.4f},"
                 f" known-mean bias {T_true2.mean()-rho:+.4f}, share removed {r2:.3f} (SE {r2_se:.3f});"
                 f" difference between the two runs {(r2 - r) / np.hypot(r_se, r2_se):+.2f} SE")
    # (c) the toy's exact bias (3) via (11) on a grid of tau, against the law's autocorrelation at the lag
    c = cfg["toy"]
    parts = []
    for tau in cfg["toy_tau_grid"]:
        gam = np.exp(-np.arange(c["n"]) * c["dt"] / tau)
        ex, _ = exact_mean_ratio(gam, c["n"], c["lag"])
        parts.append(f"{1e3 * tau:.0f} ms: E[T] {ex:.4f}, rho {np.exp(-c['lag'] * c['dt'] / tau):.4f}, bias {ex - np.exp(-c['lag'] * c['dt'] / tau):+.3f}")
    lines.append("(c) toy, exact bias (3) via (11): " + "; ".join(parts))
    text = "\n".join(lines)
    print(text)
    with open(cfg["out"], "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
