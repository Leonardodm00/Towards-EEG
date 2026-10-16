"""Check 1: the bias of T = rho_hat(1 ms) on a 265 ms LS pre-window, computed without a closed form.

What it tests (document \u00a73.3, \u00a73.6, \u00a73.7):
  (a) Monte Carlo estimate (document eq. (10)) of bias(T; g | theta, n) = E[T | theta, n] - rho(1 ms | theta) and its Monte Carlo SE
      (Morris et al. 2019, Table 6) at M = 200, 800, 2000, 3200 nested windows, theta = theta_sim;
      the first 2000 windows reproduce the run of 2026-10-11 (bias -0.1100, mean 0.5444).
  (b) The error of single windows (10-90 % range, share above the target, also over the first 2000 windows) and the
      MSE split bias^2 + Var, for one window and for the average of K = 6 windows.
  (c) The ratio of expectations q_cf (topic doc eq. (10)) against E[T]: the size of the covariance term of the
      identity q_cf = E[T] + Cov(S_0, T) / E[S_0] (document eq. (8), an algebraic identity, proved in the text);
      its Monte Carlo side equals the ratio of the Monte Carlo means of S_L and S_0, so comparing it with q_cf
      checks topic doc eq. (9), not the identity.
  (d) Topic doc eq. (9) against Monte Carlo means of S_0 and S_L (differences in standard-error units).
  (e) The same bias at theta_cf (another fitted law; M = 6400) and under the best single AR(1) (a wrong law).
Prediction fixed before running: (a) the estimates agree within 3 Monte Carlo SE and the SE halves when M
quadruples; (c) both sides agree within 3 SE of the right side; (d) differences below 3 SE.
Run:  python3 check01_bias_mc_vs_closed_form.py      (about 10 s; prints and writes logs/check01.txt)
"""
import numpy as np
from bias_common import DT, N_LS, LAG_1MS, THETA_SIM, THETA_CF, AR1_BEST
from bias_common import true_acf, autocov, expected_S, closed_form_acf, monte_carlo

CONFIG = dict(dt=DT, n=N_LS, lag=LAG_1MS,
              M_sim=3200, M_report=(200, 800, 2000, 3200), seed_sim=20261011,
              M_cf=6400, M_other=1600, seed_cf=20261012, seed_ar1=20261013,
              K_avg=6, n_first=2000, out="logs/check01.txt")


def bias_block(label, th, cfg, M, seed, M_report):
    """Monte Carlo bias table, closed form, the covariance term of identity (8) and a TD (9) check, for one law."""
    lines = []
    rho = true_acf(th, cfg["dt"], cfg["lag"])
    q_cf = closed_form_acf(th, cfg["dt"], cfg["n"], cfg["lag"])
    S0, SL = monte_carlo(th, cfg["dt"], cfg["n"], cfg["lag"], M, seed)
    T = SL / S0
    lines.append(f"[{label}]  rho(1 ms | theta) = {rho:.6f}")
    for m in M_report:
        b, se = T[:m].mean() - rho, T[:m].std(ddof=1) / np.sqrt(m)
        lines.append(f"  M = {m:5d}: mean T = {T[:m].mean():.4f}  bias = {b:+.4f}  (MC SE {se:.4f})")
    if len(M_report) > 1:   # nested subsets: SE of (mean of first m) - (mean of all M) is s * sqrt(1/m - 1/M)
        m0, s_all = M_report[0], T.std(ddof=1)
        d = T[:m0].mean() - T.mean()
        lines.append(f"  nested check: mean(first {m0}) - mean(all {M}) = {d:+.4f}, "
                     f"SE of that difference {s_all * np.sqrt(1 / m0 - 1 / M):.4f}, ratio {d / (s_all * np.sqrt(1 / m0 - 1 / M)):+.2f}")
    med = np.median(T)
    lines.append(f"  median T (M = {M}) = {med:.4f};  median bias = {med - rho:+.4f}")
    lines.append(f"  closed form q_cf = {q_cf:.4f};  q_cf - rho = {q_cf - rho:+.4f}  "
                 f"({100 * (q_cf - rho) / (T.mean() - rho):.0f} % of the MC bias);  q_cf - mean T = {q_cf - T.mean():+.4f};"
                 f"  q_cf - median T = {q_cf - med:+.4f}")
    # identity (8): q_cf = E[T] + Cov(S0, T)/E[S0]; its Monte Carlo side is mean(S_L)/mean(S_0) (a TD (9) check)
    rhs_terms = T + (S0 - S0.mean()) * (T - T.mean()) / S0.mean()
    rhs, rhs_se = rhs_terms.mean(), rhs_terms.std(ddof=1) / np.sqrt(T.size)
    cov_term = np.mean((S0 - S0.mean()) * (T - T.mean())) / S0.mean()
    lines.append(f"  identity (8): covariance term Cov(S0,T)/E[S0] (MC) = {cov_term:+.4f} against q_cf - mean T = {q_cf - T.mean():+.4f};"
                 f"  MC ratio of means mean(S_L)/mean(S_0) = {rhs:.4f} (SE {rhs_se:.4f}) vs q_cf {q_cf:.4f} from TD (9):"
                 f" difference in SE units {(rhs - q_cf) / rhs_se:+.2f}")
    gam = autocov(th, cfg["dt"], cfg["n"])
    for name, S, L in (("S_0", S0, 0), ("S_L", SL, cfg["lag"])):
        e = expected_S(gam, cfg["n"], L)
        z = (S.mean() - e) / (S.std(ddof=1) / np.sqrt(S.size))
        lines.append(f"  TD eq. (9) {name}: analytic {e:.5f} mV^2, MC mean {S.mean():.5f} mV^2, difference in SE units {z:+.2f}")
    return lines, T, rho


def error_block(T, rho, K, n_first):
    """Single-window errors and the MSE split, for one window and for the mean of K windows."""
    err = T - rho
    q10, q50, q90 = np.quantile(err, [0.1, 0.5, 0.9])
    bias, var = err.mean(), T.var(ddof=1)
    mse1, mseK = bias**2 + var, bias**2 + var / K
    return [f"  single-window error t(x) - rho: 10/50/90 % = {q10:+.3f} / {q50:+.3f} / {q90:+.3f};"
            f"  share of windows above the target = {np.mean(err > 0):.3f} (first {n_first}: {np.mean(err[:n_first] > 0):.4f});"
            f"  min/max = {err.min():+.3f} / {err.max():+.3f}",
            f"  SD(T) = {np.sqrt(var):.4f};  MSE split, one window: bias^2 = {bias**2:.5f}, Var = {var:.5f} "
            f"(bias^2 share {bias**2 / mse1:.2f});  mean of K = {K}: Var/K = {var / K:.5f} (bias^2 share {bias**2 / mseK:.2f})"]


def main(cfg=CONFIG):
    out = []
    lines, T, rho = bias_block("theta_sim", THETA_SIM, cfg, cfg["M_sim"], cfg["seed_sim"], cfg["M_report"])
    out += lines + error_block(T, rho, cfg["K_avg"], cfg["n_first"])
    lines_cf, T_cf, rho_cf = bias_block("theta_cf", THETA_CF, cfg, cfg["M_cf"], cfg["seed_cf"], (cfg["M_cf"],))
    out += lines_cf
    b1, b2 = T.mean() - rho, T_cf.mean() - rho_cf
    se12 = np.sqrt(T.var(ddof=1) / T.size + T_cf.var(ddof=1) / T_cf.size)
    out.append(f"  bias at theta_sim minus bias at theta_cf = {b1 - b2:+.4f} (SE {se12:.4f}; {(b1 - b2) / se12:+.1f} SE)")
    out += bias_block("best single AR(1), a wrong law", AR1_BEST, cfg, cfg["M_other"], cfg["seed_ar1"],
                      (cfg["M_other"],))[0]
    text = "\n".join(out)
    print(text)
    with open(cfg["out"], "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
