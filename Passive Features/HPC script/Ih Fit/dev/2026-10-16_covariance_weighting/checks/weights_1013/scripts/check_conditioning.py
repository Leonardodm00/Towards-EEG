"""Conditioning of the covariance of the ten LS medians (check, 2026-10-14).

Reads ../results/weights.npz (check_weights.py). Reports: eigenvalues and condition number of Sigma
(mixed units) and of the correlation matrix (unit-free); and, for random symmetric perturbations
dSigma with ||dSigma||_2 / ||Sigma||_2 = eps, the relative change of Sigma^{-1} against the
first-order bound kappa * eps (textbook perturbation bound, from memory).
Prediction fixed before running: the observed relative change never exceeds 1.1 * kappa * eps
(first-order bound; eps small) and its maximum reaches at least 0.1 * kappa * eps.
[Result of the first run, 2026-10-14: FAILED for Sigma at eps = 1e-4 (median change 1.6, max 2.9e3 against
the bound 6.7), held for the correlation matrix (max 0.24 <= 0.26). Cause: the first-order bound needs
kappa * eps << 1, and for Sigma kappa * eps = 6.7 > 1 (the perturbation exceeds the smallest eigenvalue).
From the second run the script also tests eps = 1e-6 (kappa * eps = 0.067) for Sigma, inside the bound's
domain; the eps = 1e-4 row is kept as the out-of-domain illustration. The second run also exceeded the
first-order bound for the correlation matrix (max 0.32 > 0.29) at kappa * eps = 0.26, which is not << 1.
From the third run the test uses the exact bound kappa*eps / (1 - kappa*eps), valid for kappa*eps < 1
(textbook, from memory); rows with kappa*eps >= 1 have no bound and are reported only.]
"""
import numpy as np

CONFIG = dict(eps=1e-4, n_pert=2000, seed=2026101410)
d = np.load(__file__.rsplit("/scripts/", 1)[0] + "/results/weights.npz")
Sigma = d["corr"] * np.outer(d["se"], d["se"])
C = d["corr"]
for name, A in (("Sigma (mixed units)", Sigma), ("correlation matrix (unit-free)", C)):
    ev = np.linalg.eigvalsh(A)
    print("%-32s eigenvalues %s" % (name, np.array2string(np.sort(ev)[::-1], precision=4)))
    print("%-32s kappa = %.3g" % ("", ev.max() / ev.min()))
rng = np.random.default_rng(CONFIG["seed"])
for name, A, eps in (("Sigma", Sigma, CONFIG["eps"]), ("Sigma", Sigma, 1e-6), ("corr", C, CONFIG["eps"])):
    kappa, Ainv, nA = np.linalg.cond(A), np.linalg.inv(A), np.linalg.norm(A, 2)
    rel = []
    for _ in range(CONFIG["n_pert"]):
        E = rng.normal(size=A.shape); E = (E + E.T) / 2
        E *= eps * nA / np.linalg.norm(E, 2)
        rel.append(np.linalg.norm(np.linalg.inv(A + E) - Ainv, 2) / np.linalg.norm(Ainv, 2))
    rel = np.array(rel); x = kappa * eps
    bound = x / (1 - x) if x < 1 else np.inf
    ok = (rel.max() <= bound) if x < 1 else "no bound (kappa*eps >= 1)"
    print("%-6s eps = %.0e (kappa*eps = %.3g): rel. change of inverse median %.2e, max %.2e; bound %.3g -> %s"
          % (name, eps, x, np.median(rel), rel.max(), bound, ok))
