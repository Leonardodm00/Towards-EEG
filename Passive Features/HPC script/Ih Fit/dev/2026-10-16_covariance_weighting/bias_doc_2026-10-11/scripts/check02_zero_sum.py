"""Check 2: the zero-sum identity of the textbook sample autocorrelation (topic doc eq. (23), document eq. (7)).

For every window with S_0 > 0:  sum_{L=1}^{n-1} rho_hat(L) = -1/2  exactly.
Tested on windows of three laws (theta_sim on a 265 ms window, white noise, the best single AR(1)).
Prediction fixed before running: |sum + 1/2| < 1e-9 for every window.
Run:  python3 check02_zero_sum.py      (a few seconds; prints and writes logs/check02.txt)
"""
import numpy as np
from bias_common import DT, N_LS, THETA_SIM, AR1_BEST, simulate_windows

CONFIG = dict(dt=DT, n=N_LS, n_win=5, seed=20261015, out="logs/check02.txt")


def acf_all_lags(x):
    """rho_hat(L), L = 0..n-1, of one window: S_L / S_0 with the window demeaned (FFT, zero-padded)."""
    u = x - x.mean()
    n = u.size
    f = np.fft.rfft(u, 2 * n)
    s = np.fft.irfft(f * np.conj(f), 2 * n)[:n]          # S_L for L = 0..n-1
    return s / s[0]


def main(cfg=CONFIG):
    rng = np.random.default_rng(cfg["seed"])
    laws = {"theta_sim": THETA_SIM, "AR(1) best": AR1_BEST}
    lines, worst = [], 0.0
    for name, th in laws.items():
        X = simulate_windows(th, cfg["dt"], cfg["n"], cfg["n_win"], rng)
        sums = [acf_all_lags(x)[1:].sum() for x in X]
        worst = max(worst, max(abs(s + 0.5) for s in sums))
        lines.append(f"{name}: sums over L = 1..n-1: " + ", ".join(f"{s:.12f}" for s in sums))
    W = rng.standard_normal((cfg["n_win"], cfg["n"]))
    sums = [acf_all_lags(x)[1:].sum() for x in W]
    worst = max(worst, max(abs(s + 0.5) for s in sums))
    lines.append("white noise: sums over L = 1..n-1: " + ", ".join(f"{s:.12f}" for s in sums))
    lines.append(f"largest |sum + 1/2| = {worst:.2e}")
    text = "\n".join(lines)
    print(text)
    with open(cfg["out"], "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
